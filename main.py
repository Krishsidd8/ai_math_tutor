import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import uvicorn
import os, io, json, traceback, logging
import gdown
import google.generativeai as genai
from sympy import sympify, simplify
from fastapi.middleware.cors import CORSMiddleware

# -------------------- LOGGING CONFIG --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------- CONSTANTS --------------------
max_height, max_width = 384, 512
checkpoint_path = "ocr_checkpoint.pt"
file_id = os.environ.get("GDRIVE_FILE_ID")
if not file_id:
    raise RuntimeError("Missing GDRIVE_FILE_ID environment variable")

# -------------------- CHECKPOINT DOWNLOAD --------------------
if not os.path.exists(checkpoint_path):
    logger.info("Checkpoint not found. Downloading from Google Drive...")
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", checkpoint_path, quiet=False)
        logger.info("Checkpoint downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        raise

# -------------------- MODEL CLASSES --------------------
class OCRModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        conv_out = 64 * (max_height // 4) * (max_width // 4)
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
        enc = self.fc(enc).unsqueeze(1).transpose(0, 1)
        tgt_emb = self.embedding(tgt)
        out = self.transformer(enc, tgt_emb, tgt_mask=tgt_mask)
        return self.out(out)

class LatexTokenizer:
    def __init__(self, vocab, specials):
        self.specials = specials
        self.vocab = vocab
        self.t2i = {tok: i for i, tok in enumerate(self.vocab)}
        self.i2t = {i: tok for tok, i in self.t2i.items()}

    def decode(self, ids):
        return ' '.join(self.i2t[i] for i in ids if self.i2t[i] not in self.specials)

# -------------------- LOAD MODEL --------------------
try:
    logger.info("Loading model checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = LatexTokenizer(ckpt['tokenizer_vocab'], ckpt['specials'])
    model = OCRModel(len(tokenizer.vocab))
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info("Model loaded and ready.")
except Exception as e:
    logger.error("Failed to load model checkpoint:")
    logger.error(traceback.format_exc())
    raise

# -------------------- FASTAPI APP --------------------
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    logger.info("Received /predict request.")
    try:
        img = Image.open(file.file).convert("L")
        result = predict_image(img)
        logger.info(f"Prediction result: {result}")
        return {"latex": result}
    except Exception as e:
        logger.error("Error during prediction:")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    logger.info("Received /solve request.")
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        latex = predict_image(img) or ""
        logger.info(f"Predicted LaTeX: {latex}")

        steps = solve_with_gemini(latex)
        if not steps:
            logger.warning("Gemini returned no steps. Falling back to SymPy.")
            steps = quick_sympy_steps(latex)

        logger.info(f"Steps returned: {len(steps)}")
        return {"latex": latex, "steps": steps}
    except Exception as e:
        logger.error("Error during solve:")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# -------------------- IMAGE TO LATEX --------------------
def predict_image(img: Image.Image, max_len=60):
    logger.info("Starting image prediction...")
    transform = transforms.Compose([
        transforms.Resize((max_height, max_width)),
        transforms.ToTensor()
    ])
    img_t = transform(img.convert("L")).unsqueeze(0)

    tgt = torch.tensor([[tokenizer.t2i['<SOS>']]], dtype=torch.long)
    for _ in range(max_len):
        tgt_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf')), diagonal=1)
        logits = model(img_t, tgt, tgt_mask=tgt_mask)
        next_token = logits[-1].argmax(dim=-1).unsqueeze(0)
        tgt = torch.cat([tgt, next_token], dim=1)
        
        if next_token.item() == tokenizer.t2i['<EOS>']:
            break

    result = tokenizer.decode(tgt.squeeze().tolist())
    logger.info(f"Decoded result: {result}")
    return result

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://krishsidd8.github.io/ai_math_tutor/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- GEMINI SETUP --------------------
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step":   {"type": "string"},
                    "detail": {"type": "string"}
                },
                "required": ["step", "detail"]
            }
        }
    },
    "required": ["steps"]
}

def solve_with_gemini(latex_expr: str) -> list[dict]:
    logger.info("Calling Gemini for step-by-step solution...")
    prompt = (
        "You are a math tutor. Given a math expression or equation in LaTeX, "
        "produce a clear, correct, step-by-step solution. "
        "Only return JSON that matches the provided schema. "
        "Avoid extra commentary. Keep steps concise but correct.\n\n"
        f"LaTeX: {latex_expr}"
    )
    try:
        resp = GEMINI_MODEL.generate_content(
            [prompt],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": STEP_SCHEMA,
                "temperature": 0.2,
            },
        )
        data = json.loads(resp.text)
        steps = data.get("steps", [])
        logger.info("Gemini returned valid response.")
        return [
            {"step": s.get("step", ""), "detail": s.get("detail", "")}
            for s in steps if isinstance(s, dict)
        ]
    except Exception as e:
        logger.error("Gemini call failed:")
        logger.error(traceback.format_exc())
        return [{"step": "AI solver unavailable", "detail": str(e)}]

def quick_sympy_steps(latex_expr: str) -> list[dict]:
    logger.info("Attempting fallback using SymPy...")
    try:
        if "=" in latex_expr:
            lhs_txt, rhs_txt = latex_expr.split("=", 1)
            lhs, rhs = sympify(lhs_txt), sympify(rhs_txt)
            expr = lhs - rhs
        else:
            expr = sympify(latex_expr)

        simp = simplify(expr)
        logger.info("SymPy simplification successful.")
        return [
            {"step": "Parse LaTeX", "detail": str(expr)},
            {"step": "Simplify",   "detail": str(simp)},
        ]
    except Exception as e:
        logger.error("SymPy simplification failed:")
        logger.error(traceback.format_exc())
        return [{"step": "Could not parse with SymPy", "detail": str(e)}]

# -------------------- START SERVER --------------------
if __name__ == "__main__":
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)