import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import uvicorn
import os, io, json, traceback, logging
import gdown
import google.generativeai as genai
from sympy import sympify, simplify
from fastapi.middleware.cors import CORSMiddleware
import math
import traceback

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

# -------------------- TOKENIZER --------------------
class LatexTokenizer:
    def __init__(self, vocab=None, specials=None):
        self.specials = specials or ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.vocab = vocab or []
        self.t2i = {tok: i for i, tok in enumerate(self.vocab)} if self.vocab else {}
        self.i2t = {i: tok for tok, i in self.t2i.items()} if self.vocab else {}

    def decode(self, ids):
        return ' '.join(self.i2t[i] for i in ids if self.i2t[i] not in self.specials)

# -------------------- MODEL --------------------
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class OCRSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dim_ff=2048, dropout=0.1):
        super().__init__()
        from torchvision.models import resnet18
        self.encoder = nn.Sequential(*list(resnet18(weights="IMAGENET1K_V1").children())[:-2])
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding1D(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, imgs, tgt):
        B, _, H, W = imgs.shape
        x = imgs.repeat(1,3,1,1)
        feats = self.encoder(x)
        feats = self.proj(feats)
        feats = feats.permute(0,2,3,1).view(B, -1, self.d_model).permute(1,0,2)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt.device)
        out = self.decoder(tgt_emb, feats, tgt_mask=tgt_mask)
        return self.fc_out(out)

# -------------------- LOAD MODEL --------------------
try:
    logger.info("Loading model checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(ckpt.keys())
    tokenizer = LatexTokenizer()
    tokenizer.vocab = ckpt['tokenizer_vocab']
    tokenizer.specials = ckpt['specials']
    tokenizer.t2i = {tok: i for i, tok in enumerate(tokenizer.vocab)}
    tokenizer.i2t = {i: tok for tok, i in tokenizer.t2i.items()}
    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder.cnn."):
            new_key = k.replace("encoder.cnn.", "encoder.")
        elif k.startswith("encoder.proj."):
            new_key = k.replace("encoder.proj.", "proj.")
        else:
            new_key = k
        new_state_dict[new_key] = v
    vocab_size = len(tokenizer.vocab)
    model = OCRSeq2Seq(vocab_size=vocab_size, d_model=512, nhead=8, num_layers=4, dim_ff=2048, dropout=0.1)
    model.load_state_dict(new_state_dict)
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
        logger.info(f"Opened image with size {img.size}")
        result = predict_image(img)
        logger.info(f"Prediction completed: {result}")
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

        steps = []
        if latex:
            try:
                steps = solve_with_gemini(latex)
            except Exception as e:
                logger.warning("Gemini solver failed (possibly quota exceeded): %s", e)
                if note:
                    note += " "
                note += "Gemini solver unavailable (quota may be exceeded)."

        if not steps:
            logger.info("Falling back to SymPy for solution steps.")
            steps = quick_sympy_steps(latex)
            if not note:
                note = "SymPy fallback used."

        def render_latex_mathjax(latex_code: str, display_mode: bool = True) -> str:
            """Wrap LaTeX in MathJax delimiters for HTML rendering."""
            if not latex_code.strip():
                return ""
            return f"$${latex_code}$$" if display_mode else f"${latex_code}$"

        for step in steps:
            step["mathjax"] = render_latex_mathjax(step.get("detail", ""), display_mode=True)

        response = {"latex": latex, "steps": steps}
        if note:
            response["note"] = note

        logger.info(f"Returning {len(steps)} steps with MathJax rendering.")
        return response
    
    except Exception as e:
        logger.error("Error during /solve processing:")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# -------------------- IMAGE TO LATEX --------------------
def predict_image(img: Image.Image, max_len=60, device='cpu'):
    logger.info("Starting image prediction...")
    
    transform = transforms.Compose([
        transforms.Resize((max_height, max_width)),
        transforms.ToTensor()
    ])
    img_t = transform(img.convert("L")).unsqueeze(0).to(device)
    logger.info(f"Image transformed to tensor of shape {img_t.shape}")

    tgt = torch.tensor([[tokenizer.t2i['<SOS>']]], dtype=torch.long, device=device)
    logger.info(f"Initial target sequence: {tgt}")

    for step_idx in range(max_len):
        logger.info(f"Prediction step {step_idx+1}")
        tgt_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf')), diagonal=1).to(device)
        logits = model(img_t, tgt, tgt_mask=tgt_mask)
        next_token = logits[-1].argmax(dim=-1).unsqueeze(0)
        tgt = torch.cat([tgt, next_token], dim=1)
        logger.info(f"Next token predicted: {next_token.item()} ({tokenizer.i2t.get(next_token.item(), 'UNK')})")
        
        if next_token.item() == tokenizer.t2i['<EOS>']:
            logger.info("EOS token encountered; stopping prediction.")
            break

    result = tokenizer.decode(tgt.squeeze().tolist())
    logger.info(f"Decoded LaTeX result: {result}")
    return result


def predict_greedy(img, model, tokenizer, max_len=200, device='cpu'):
    logger.info("Starting greedy prediction pipeline...")
    model.eval()
    with torch.no_grad():
        if isinstance(img, Image.Image):
            transform = transforms.Compose([
                transforms.Resize((384,512)),
                transforms.ToTensor()
            ])
            img = transform(img).unsqueeze(0).to(device)
            logger.info(f"Image transformed for greedy prediction: {img.shape}")

        memory = model.encoder(img.repeat(1,3,1,1))
        logger.info(f"Encoder output shape: {memory.shape}")

        memory = model.proj(memory).permute(0,2,3,1).view(1,-1,model.d_model).permute(1,0,2)
        logger.info(f"Projected memory shape: {memory.shape}")

        cur_seq = torch.tensor([[tokenizer.t2i['<SOS>']]], device=device)
        for step_idx in range(max_len):
            tgt_emb = model.embedding(cur_seq) * math.sqrt(model.d_model)
            tgt_emb = model.positional_encoding(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(cur_seq.device)
            out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = model.fc_out(out)
            next_tok = logits[-1,0].argmax(-1).unsqueeze(0).unsqueeze(0)
            logger.info(f"Step {step_idx+1}: predicted token {next_tok.item()} ({tokenizer.i2t.get(next_tok.item(),'UNK')})")
            if next_tok.item() == tokenizer.t2i['<EOS>']:
                logger.info("EOS token encountered; stopping greedy decoding.")
                break
            cur_seq = torch.cat([cur_seq, next_tok], dim=0)

        result = tokenizer.decode(cur_seq[1:].squeeze().tolist())
        logger.info(f"Greedy decoded LaTeX result: {result}")
        return result

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://krishsidd8.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- GEMINI SETUP --------------------
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-pro")

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

# -------------------- SOLVER --------------------
def solve_with_gemini(latex_expr: str) -> list[dict]:
    logger.info("Calling Gemini for step-by-step solution...")
    prompt = (
        "You are a math tutor. Given a math expression in LaTeX, "
        "produce a clear, correct, step-by-step solution. "
        "Return only JSON matching the schema. "
        f"LaTeX: {latex_expr}"
    )
    try:
        response = GEMINI_MODEL.generate_content(
            [prompt],
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "response_schema": STEP_SCHEMA,
            },
        )
        data = json.loads(response.text)
        steps = data.get("steps", [])
        return [{"step": s.get("step", ""), "detail": s.get("detail", "")} for s in steps if isinstance(s, dict)]
    except Exception as e:
        logger.error("Gemini solve failed:")
        logger.error(traceback.format_exc())
        return [{"step": "AI solver unavailable", "detail": str(e)}]

def quick_sympy_steps(latex_expr: str) -> list[dict]:
    logger.info("Attempting fallback using SymPy...")
    if not latex_expr.strip():
        logger.warning("Empty LaTeX input; cannot use SymPy.")
        return [{"step": "No input available", "detail": "Cannot parse empty LaTeX expression."}]
    try:
        if "=" in latex_expr:
            lhs_txt, rhs_txt = latex_expr.split("=", 1)
            lhs, rhs = sympify(lhs_txt), sympify(rhs_txt)
            expr = lhs - rhs
        else:
            expr = sympify(latex_expr)

        simp = simplify(expr)
        return [
            {"step": "Parse LaTeX", "detail": str(expr)},
            {"step": "Simplify", "detail": str(simp)},
        ]
    except Exception as e:
        logger.error("SymPy simplification failed:")
        logger.error(traceback.format_exc())
        return [{"step": "Could not parse with SymPy", "detail": str(e)}]

# -------------------- START SERVER --------------------
if __name__ == "__main__":
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)