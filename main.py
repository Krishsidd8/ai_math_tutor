import os
import io
import pickle
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tokenizer import LatexTokenizer

import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy import Eq, simplify, factor, solve

try:
    import gdown
except Exception:
    gdown = None

MAX_HEIGHT = 384
MAX_WIDTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OCR_CHECKPOINT_ENV = "GDRIVE_OCR_ID"
TOKENIZER_ENV = "GDRIVE_TOKENIZER_ID"

OCR_LOCAL = "ocr_checkpoint.pt"
TOKENIZER_LOCAL = "tokenizer.pkl"

class OCRModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        conv_out = 64 * (MAX_HEIGHT // 4) * (MAX_WIDTH // 4)
        self.fc = nn.Linear(conv_out, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=4,
            num_encoder_layers=3, num_decoder_layers=3
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, imgs, tgt, tgt_mask=None):
        B, _, h, w = imgs.shape
        enc = self.encoder(imgs).view(B, -1)
        enc = self.fc(enc).unsqueeze(0)
        tgt_emb = self.embedding(tgt)
        out = self.transformer(enc, tgt_emb, tgt_mask=tgt_mask)
        return self.out(out)

transform = transforms.Compose([
    transforms.Resize((MAX_HEIGHT, MAX_WIDTH)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

def download_from_gdrive(file_id: str, out_path: str):
    if os.path.exists(out_path):
        return out_path
    if gdown is None:
        raise RuntimeError("gdown is required to download from Google Drive on-the-fly. Add to requirements.")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, out_path, quiet=False)
    return out_path

tokenizer: Optional[LatexTokenizer] = None
model: Optional[OCRModel] = None
vocab_size = None

def load_assets():
    global tokenizer, model, vocab_size
    
    ocr_id = os.getenv(OCR_CHECKPOINT_ENV)
    tok_id = os.getenv(TOKENIZER_ENV)

    print(f"ocr_id = {ocr_id}")
    print(f"tok_id = {tok_id}")

    if tok_id is None or ocr_id is None:
        raise RuntimeError(f"Please set environment variables {OCR_CHECKPOINT_ENV} and {TOKENIZER_ENV} (Google Drive file IDs).")

    print("Downloading tokenizer...")
    download_from_gdrive(tok_id, TOKENIZER_LOCAL)
    print("Tokenizer downloaded.")

    print("Downloading model...")
    download_from_gdrive(ocr_id, OCR_LOCAL)
    print("Model downloaded.")

    print("Loading tokenizer...")
    with open(TOKENIZER_LOCAL, "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded.")

    vocab_size = len(tokenizer.vocab)
    print("Vocab size:", vocab_size)
    model = OCRModel(vocab_size=vocab_size)

    # TEMP HARDCODE TEST
    if not os.path.exists(TOKENIZER_LOCAL):
        raise RuntimeError("tokenizer.pkl missing locally.")

    print("Loading tokenizer from local disk...")
    with open(TOKENIZER_LOCAL, "rb") as f:
        tokenizer = pickle.load(f)

    print("Loading model state...")
    map_location = DEVICE
    state = torch.load(OCR_LOCAL, map_location=map_location)
    sd = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state
    try:
        model.load_state_dict(sd)
    except Exception as e:
        print("Warning: state_dict load failed:", e)
        model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    print("Model and tokenizer loaded. Device:", DEVICE)

def generate_from_image(img: Image.Image, max_len=150) -> str:
    if tokenizer is None or model is None:
        raise RuntimeError("Model/tokenizer not loaded.")

    print("[DEBUG] Starting image preprocessing...")
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    print(f"[DEBUG] Image tensor shape: {img_t.shape}")

    B = 1
    sos = tokenizer.t2i['<SOS>']
    eos = tokenizer.t2i['<EOS>']

    generated = [sos]
    for step in range(max_len):
        tgt_tensor = torch.tensor(generated, dtype=torch.long, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            out = model(img_t, tgt_tensor)
        last_logits = out[-1, 0]
        next_id = int(torch.argmax(last_logits).cpu().numpy())

        print(f"[DEBUG] Step {step}: Next token id = {next_id}, token = {tokenizer.i2t.get(next_id, '?')}")

        if next_id == eos:
            break
        generated.append(next_id)

    decoded = tokenizer.decode(generated)
    print(f"[INFO] Decoded LaTeX: {decoded}")
    return decoded

def get_steps_for_univariate(equation, variable):
    steps = []

    lhs, rhs = equation.lhs, equation.rhs
    full_expr = simplify(lhs - rhs)
    steps.append(("Move all terms to one side", f"{sp.latex(full_expr)} = 0"))

    factored = factor(full_expr)
    if factored != full_expr:
        steps.append(("Factor the expression", f"{sp.latex(factored)} = 0"))
    else:
        steps.append(("Expression cannot be factored further", f"{sp.latex(full_expr)} = 0"))

    sols = solve(equation, variable)
    if sols:
        for i, sol in enumerate(sols, 1):
            steps.append((f"Solve for {variable}", f"{variable} = {sp.latex(sol)}"))
    
    print(f"[DEBUG] Solving equation: {equation} for variable {variable}")
    
    return steps

def solve_with_steps(latex_str):
    print(f"[INFO] Attempting to parse LaTeX: {latex_str}")
    try:
        expr = parse_latex(latex_str)
        print(f"[INFO] Parsed SymPy Expression: {expr}")

        if isinstance(expr, sp.Equality):
            lhs, rhs = expr.lhs, expr.rhs
        else:
            lhs, rhs = expr, 0

        equation = Eq(lhs, rhs)
        print(f"[INFO] Equation formed: {equation}")

        variables = list(equation.free_symbols)
        if not variables:
            return {"error": "No variables found in expression."}
        var = variables[0]
        print(f"[INFO] Solving with respect to variable: {var}")

        steps = get_steps_for_univariate(equation, var)
        return {
            "equation": sp.latex(equation),
            "variable": str(var),
            "steps": [{"desc": d, "detail": det} for d, det in steps]
        }

    except Exception as e:
        print(f"[ERROR] Failed to parse or solve LaTeX: {e}")
        return {"error": f"Error parsing or solving LaTeX: {str(e)}"}


app = FastAPI(title="AI Math Tutor Backend")
'''
allowed = os.getenv("ALLOWED_ORIGINS", "*")
if allowed == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in allowed.split(",")]
'''

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    try:
        load_assets()
    except Exception as e:
        print("Startup error (assets not loaded):", e)

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (png/jpg). Returns decoded LaTeX string.
    """
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        latex_str = generate_from_image(img)
        return {"latex": latex_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr_base64")
async def ocr_base64_endpoint(payload: dict):
    """
    Accepts {"image": "<base64 data uri or base64 blob>"}.
    """
    b64 = payload.get("image")
    if not b64:
        raise HTTPException(status_code=400, detail="Missing 'image' field with base64 data.")
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    import base64
    try:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    try:
        latex_str = generate_from_image(img)
        return {"latex": latex_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "AI Math Tutor Backend is running!"}
    
@app.post("/solve")
async def solve_problem(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}, content_type: {file.content_type}")
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print("[DEBUG] Image loaded successfully.")

        latex = generate_from_image(image)
        print(f"[INFO] OCR Output LaTeX: {latex}")

        steps = solve_with_steps(latex)

        print(f"[INFO] Final solve response: {steps}")

        return {
            "latex": latex,
            "steps": steps
        }

    except Exception as e:
        print(f"[ERROR] Internal server error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
