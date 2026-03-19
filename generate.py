"""Quick script to load tinystories-pretrained weights and generate text."""

import json
import torch
from pathlib import Path

from src.model import AdditionLM
from src.tokenization import get_tokenizer

CKPT_DIR = Path("checkpoints") / "tinystories_pretrained"

# Load config & tokenizer from checkpoint
with open(CKPT_DIR / "config.json") as f:
    cfg = json.load(f)
enc = get_tokenizer(CKPT_DIR / "vocab.json")

# Build model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdditionLM(
    vocab_size=cfg["vocab_size"],
    d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],
    n_layers=cfg["n_layers"],
    d_ff=cfg["d_ff"],
    max_seq_len=cfg["max_seq_len"],
    dropout=0.0,
    d_emb=cfg["d_emb"],
).to(device)
model.load_state_dict(torch.load(CKPT_DIR / "final.pt", map_location=device, weights_only=True))
model.eval()

# Generate
prompt = "Once upon a time"
ids = enc.encode(prompt)
idx = torch.tensor([ids], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=200, temperature=0.8)
print(enc.decode(out[0].tolist()))
