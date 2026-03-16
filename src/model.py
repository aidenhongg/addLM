"""Decoder-only transformer with Chain-of-Thought generation support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

IGNORE_INDEX = -100


class AdditionLM(nn.Module):
    """Decoder-only transformer LM for arithmetic with CoT reasoning."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 22,
        d_ff: int = 768,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(self.ln_f(x))

    def compute_loss(self, idx: Tensor, targets: Tensor) -> Tensor:
        """Cross-entropy with prompt masking (targets=IGNORE_INDEX are ignored)."""
        logits = self(idx)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
        )

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_token: int | None = None,
    ) -> Tensor:
        """Autoregressively generate tokens (greedy when temperature ~ 0)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.max_seq_len :]
            logits = self(idx_crop)[:, -1, :]
            if temperature < 1e-8:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
            if eos_token is not None and (next_tok == eos_token).all():
                break
        return idx

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    

