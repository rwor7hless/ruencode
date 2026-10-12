from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    context: int = 512


def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    t = x.shape[-2]
    c, s = cos[:t], sin[:t]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c: LMConfig):
        super().__init__()
        d = c.d_model
        self.h = c.n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(c, dim=2)
        q, k, v = (z.view(b, t, self.h, c // self.h).transpose(1, 2) for z in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, c)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyLM(nn.Module):
    def __init__(self, c: LMConfig):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)
        cos, sin = rope_cache(c.context, c.d_model // c.n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, idx, targets=None, reduction="mean"):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        logits = self.ln_f(x) @ self.emb.weight.T  # связанные эмбеддинги
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, self.c.vocab_size), targets.reshape(-1), reduction=reduction)
        return logits, loss

    def non_embedding_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("emb"))
