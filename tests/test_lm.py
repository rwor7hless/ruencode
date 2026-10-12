import math

import numpy as np
import pytest
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


def test_chunked_loss_matches_full_loss_and_grads():
    torch.manual_seed(0)
    full = TinyLM(LMConfig(vocab_size=300, d_model=32, n_layers=2, n_heads=2, context=16))
    chunked = TinyLM(LMConfig(vocab_size=300, d_model=32, n_layers=2, n_heads=2, context=16, loss_chunk=7))
    chunked.load_state_dict(full.state_dict())
    x = torch.randint(0, 300, (3, 16))
    y = torch.randint(0, 300, (3, 16))
    for reduction in ("mean", "sum"):
        _, a = full(x, y, reduction=reduction)
        _, b = chunked(x, y, reduction=reduction)
        assert torch.allclose(a, b, atol=1e-4)
    _, a = full(x, y)
    _, b = chunked(x, y)
    a.backward()
    b.backward()
    assert torch.allclose(full.emb.weight.grad, chunked.emb.weight.grad, atol=1e-5)


def test_checkpointed_blocks_match_plain_loss_and_grads():
    torch.manual_seed(0)
    plain = TinyLM(LMConfig(vocab_size=300, d_model=32, n_layers=2, n_heads=2, context=16))
    ckpt = TinyLM(LMConfig(vocab_size=300, d_model=32, n_layers=2, n_heads=2, context=16, checkpoint_blocks=True))
    ckpt.load_state_dict(plain.state_dict())
    x = torch.randint(0, 300, (3, 16))
    y = torch.randint(0, 300, (3, 16))
    _, a = plain(x, y)
    _, b = ckpt(x, y)
    assert torch.allclose(a, b, atol=1e-6)
    a.backward()
    b.backward()
    for (n, p), q in zip(plain.named_parameters(), ckpt.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6), n
    with torch.no_grad():  # на оценке пересчёт не нужен и не должен мешать
        assert torch.allclose(plain(x, y)[1], ckpt(x, y)[1], atol=1e-6)


def test_pick_device_falls_back_to_cpu(monkeypatch):
    from ruencod.lm.train_lm import pick_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device("auto") == "cpu"
    assert pick_device("cpu") == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта")
def test_peak_gpu_is_per_run():
    big = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")  # 256 МБ от «предыдущего кандидата»
    del big
    tok = SmallCharTok()
    m = TinyLM(LMConfig(vocab_size=2000, d_model=32, n_layers=1, n_heads=2, context=16))
    cfg = {"context": 16, "batch_size": 8, "lr": 3e-3, "warmup_frac": 0.05, "weight_decay": 0.0, "device": "cuda"}
    stats = train_lm(m, encode_corpus(tok, ["абвгд" * 200]), cfg, seed=0)
    assert stats["peak_gpu_gb"] < 0.1
