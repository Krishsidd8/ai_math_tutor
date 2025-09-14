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
from torchvision.models import resnet18

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

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.d_model = d_model
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D PE")

    def forward(self, x):
        B, H, W, C = x.size()
        pe = torch.zeros(H, W, C, device=x.device)
        d_model = C
        d_h = d_model // 2
        d_w = d_model - d_h
        div_term_h = torch.exp(torch.arange(0, d_h, 2, device=x.device) * -(math.log(10000.0) / d_h))
        pos_h = torch.arange(0, H, device=x.device).unsqueeze(1)
        pe[:, :, 0:d_h:2] = torch.sin(pos_h * div_term_h).unsqueeze(1)
        pe[:, :, 1:d_h:2] = torch.cos(pos_h * div_term_h).unsqueeze(1)
        div_term_w = torch.exp(torch.arange(0, d_w, 2, device=x.device) * -(math.log(10000.0) / d_w))
        pos_w = torch.arange(0, W, device=x.device).unsqueeze(1)
        pe[:, :, d_h::2] = torch.sin(pos_w * div_term_w).unsqueeze(0)
        pe[:, :, d_h+1::2] = torch.cos(pos_w * div_term_w).unsqueeze(0)
        return x + pe

class EncoderResNet(nn.Module):
    def __init__(self, pretrained=True, embed_dim=512):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        layers = list(base.children())[:-2]
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.embed_dim = embed_dim
        self.pe2d = None

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.cnn(x)
        feats = self.proj(feats)
        B, C, H, W = feats.size()
        feats = feats.permute(0, 2, 3, 1)
        if self.pe2d is None or self.pe2d.height != H or self.pe2d.width != W:
            self.pe2d = PositionalEncoding2D(self.embed_dim, H, W)
        feats = self.pe2d(feats)
        feats = feats.view(B, H*W, C)
        feats = feats.permute(1, 0, 2)
        return feats

class OCRSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = EncoderResNet(embed_dim=d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding1D(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, imgs, tgt):
        memory = self.encoder(imgs)
        tgt_emb = self.embedding(tgt) * math.sqrt(memory.size(-1))
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt.device)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits

# -------------------- LOAD MODEL --------------------
try:
    logger.info("Loading model checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = LatexTokenizer()
    tokenizer.vocab = ckpt['tokenizer_vocab']
    tokenizer.specials = ckpt['specials']
    tokenizer.t2i = {tok: i for i, tok in enumerate(tokenizer.vocab)}
    tokenizer.i2t = {i: tok for tok, i in tokenizer.t2i.items()}
    state_dict = ckpt['model']
    vocab_size = len(tokenizer.vocab)
    model = OCRSeq2Seq(vocab_size=vocab_size, d_model=512, nhead=8, num_layers=4, dim_ff=2048, dropout=0.1)
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
        logger.info(f"Opened image with size {img.size}")
        latex = decode_image_to_latex(img, model, tokenizer)
        logger.info(f"Prediction completed: {latex}")
        return {"latex": latex}
    except Exception as e:
        logger.error("Error during /predict:")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    logger.info("Received /solve request.")
    note = ""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        latex = decode_image_to_latex(img, model, tokenizer) or ""
        logger.info(f"Predicted LaTeX: {latex}")
        steps = []
        if latex:
            try:
                steps = solve_with_gemini(latex)
            except Exception as e:
                logger.warning("Gemini solver failed: %s", e)
                note += "Gemini solver unavailable (quota may be exceeded)."
        if not steps:
            logger.info("Falling back to SymPy for solution steps.")
            steps = quick_sympy_steps(latex)
            if not note:
                note = "SymPy fallback used."
        def render_latex_mathjax(latex_code: str, display_mode=True) -> str:
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
        logger.error("Error during /solve:")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# -------------------- UNIFIED DECODER --------------------
def decode_image_to_latex(img: Image.Image, model, tokenizer, max_len=60, device='cpu') -> str:
    logger.info("Starting unified image-to-LaTeX decoding.")
    transform = transforms.Compose([
        transforms.Resize((max_height, max_width)),
        transforms.ToTensor()
    ])
    img_t = transform(img).unsqueeze(0).to(device)
    memory = model.encoder(img_t)
    logger.info(f"Encoder output shape: {memory.shape}")
    tgt = torch.tensor([[tokenizer.t2i['<SOS>']]], dtype=torch.long, device=device)
    for step_idx in range(max_len):
        tgt_emb = model.embedding(tgt) * math.sqrt(model.embedding.embedding_dim)
        tgt_emb = model.positional_encoding(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(device)
        out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        logits = model.fc_out(out)
        next_token = logits[-1, 0].argmax(-1).view(1,1)
        tgt = torch.cat([tgt, next_token], dim=0)  
        logger.info(f"Step {step_idx+1}: predicted token {next_token.item()} ({tokenizer.i2t.get(next_token.item(),'UNK')})")
        if next_token.item() == tokenizer.t2i['<EOS>']:
            logger.info("EOS token encountered; stopping decoding.")
            break
    result = tokenizer.decode(tgt[1:,0].tolist())
    logger.info(f"Decoded LaTeX result: {result}")
    return result

# -------------------- IMAGE TO LATEX --------------------
def predict_image(img: Image.Image, max_len=60, device='cpu'):
    return decode_image_to_latex(img, model, tokenizer, max_len=max_len, device=device)

def predict_greedy(img, model, tokenizer, max_len=200, device='cpu'):
    return decode_image_to_latex(img, model, tokenizer, max_len=max_len, device=device)

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