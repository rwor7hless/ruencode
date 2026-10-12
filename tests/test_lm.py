import math

import numpy as np
import torch

from ruencod.lm.eval_bpb import nll_bits_and_bytes
from ruencod.lm.model import LMConfig, TinyLM
from ruencod.lm.train_lm import encode_corpus, train_lm
from tests.test_metrics import CharTok


def test_default_size_is_10m_non_embedding():
    m = TinyLM(LMConfig(vocab_size=1000))
    assert 10.5e6 < m.non_embedding_params() < 10.8e6


def test_forward_shapes():
    m = TinyLM(LMConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, context=16))
    x = torch.randint(0, 50, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 50) and loss.ndim == 0


class UniformLM(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, idx, targets=None, reduction="mean"):
        logits = torch.zeros(*idx.shape, self.v)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, self.v), targets.view(-1), reduction=reduction)
        return logits, loss


class SmallCharTok(CharTok):
    vocab_size, bos_id = 2000, 0


def test_bpb_counts_every_target_once():
    tok = SmallCharTok()
    texts = ["абв" * 10, "x"]
    bits, nbytes = nll_bits_and_bytes(UniformLM(2000), tok, texts, context=7)
    n_targets = sum(len(t) for t in texts)  # bos не предсказывается, каждый символ — ровно один раз
    assert nbytes == sum(len(t.encode()) for t in texts)
    assert math.isclose(bits, n_targets * math.log2(2000), rel_tol=1e-5)


def test_tiny_overfit():
    tok = SmallCharTok()
    tokens = encode_corpus(tok, ["абвгд" * 2000])
    assert tokens.dtype == np.uint32 and tokens[0] == 0
    m = TinyLM(LMConfig(vocab_size=2000, d_model=32, n_layers=1, n_heads=2, context=16))
    cfg = {"context": 16, "batch_size": 8, "lr": 3e-3, "warmup_frac": 0.05, "weight_decay": 0.0, "threads": 2}
    stats = train_lm(m, tokens, cfg, seed=0)
    assert stats["final_loss"] < 2.0 and stats["steps"] > 0
