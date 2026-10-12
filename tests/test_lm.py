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


CPU_CFG = {"context": 16, "batch_size": 8, "lr": 3e-3, "warmup_frac": 0.05, "weight_decay": 0.1,
           "threads": 2, "device": "cpu"}


def _tiny(seed=0):
    torch.manual_seed(seed)
    return TinyLM(LMConfig(vocab_size=2000, d_model=32, n_layers=1, n_heads=2, context=16))


def _tokens():
    return encode_corpus(SmallCharTok(), ["абвгдеёжз" * 300])  # 21 шаг


class _FailAfter:
    """Подменяет forward: после `after` удачных вызовов бросает исключение (имитация остановки)."""

    def __init__(self, forward, after, exc=KeyboardInterrupt):
        self.forward, self.after, self.exc, self.n = forward, after, exc, 0

    def __call__(self, *a, **kw):
        self.n += 1
        if self.n > self.after:
            raise self.exc
        return self.forward(*a, **kw)


def test_resume_after_interrupt_matches_uninterrupted(tmp_path):
    full = _tiny()
    ref = train_lm(full, _tokens(), CPU_CFG, seed=0)
    ckpt = tmp_path / "ckpt" / "m.pt"
    part = _tiny()
    part.forward = _FailAfter(part.forward, after=5)
    with pytest.raises(KeyboardInterrupt):
        train_lm(part, _tokens(), CPU_CFG, seed=0, ckpt_path=ckpt)
    assert torch.load(ckpt)["step"] == 5  # Ctrl+C сохраняет состояние перед незавершённым шагом
    resumed = _tiny(seed=123)  # другая инициализация: всё должно прийти из сохранения
    stats = train_lm(resumed, _tokens(), CPU_CFG, seed=0, ckpt_path=ckpt, resume=True)
    assert stats["steps"] == ref["steps"] and stats["train_tokens"] == ref["train_tokens"]
    assert math.isclose(stats["final_loss"], ref["final_loss"], rel_tol=1e-5)
    for (n, a), b in zip(full.named_parameters(), resumed.parameters()):
        assert torch.allclose(a, b, atol=1e-6), n


def test_periodic_checkpoint_survives_hard_kill(tmp_path):
    ckpt = tmp_path / "m.pt"
    part = _tiny()
    part.forward = _FailAfter(part.forward, after=5, exc=RuntimeError)  # не Ctrl+C: сохранить «на выходе» нельзя
    with pytest.raises(RuntimeError):
        train_lm(part, _tokens(), {**CPU_CFG, "ckpt_minutes": 0}, seed=0, ckpt_path=ckpt)
    assert torch.load(ckpt)["step"] == 5


def test_resume_of_finished_training_does_not_train_again(tmp_path):
    ckpt = tmp_path / "m.pt"
    done = _tiny()
    ref = train_lm(done, _tokens(), CPU_CFG, seed=0, ckpt_path=ckpt)
    again = _tiny(seed=123)
    again.forward = _FailAfter(again.forward, after=0)  # любой шаг обучения упал бы
    stats = train_lm(again, _tokens(), CPU_CFG, seed=0, ckpt_path=ckpt, resume=True)
    assert stats["steps"] == ref["steps"] and math.isclose(stats["final_loss"], ref["final_loss"])
    for a, b in zip(done.parameters(), again.parameters()):
        assert torch.equal(a, b)


def _run_cfg(tmp_path):
    return {"paths": {"data": str(tmp_path / "data"), "results": str(tmp_path / "res")}, "lm": {"train_bytes": 10}}


def _fake_candidate(calls):
    def fake(cfg, name, train_bytes, ckpt_path=None, resume=False):
        calls.append((name, ckpt_path, resume))
        return [{"candidate": name, "domain": "ru_all", "bpb": 0.5}]
    return fake


def test_run_all_resume_skips_finished_and_appends(tmp_path, monkeypatch):
    import pandas as pd
    from ruencod.lm import run
    (tmp_path / "res").mkdir()
    pd.DataFrame([{"candidate": "a", "domain": "ru_all", "bpb": 1.0}]).to_csv(tmp_path / "res" / "lm_bpb.csv", index=False)
    ckpt_dir = tmp_path / "data" / "lm_ckpt"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "b.pt").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(run, "run_candidate", _fake_candidate(calls))
    run.run_all(_run_cfg(tmp_path), ["a", "b"], resume=True)
    assert calls == [("b", ckpt_dir / "b.pt", True)]
    saved = pd.read_csv(tmp_path / "res" / "lm_bpb.csv")
    assert list(saved.candidate) == ["a", "b"] and list(saved.bpb) == [1.0, 0.5]
    assert not (ckpt_dir / "b.pt").exists()  # результат записан — сохранение удалено


def test_run_all_without_resume_starts_over(tmp_path, monkeypatch):
    import pandas as pd
    from ruencod.lm import run
    (tmp_path / "res").mkdir()
    pd.DataFrame([{"candidate": "a", "domain": "ru_all", "bpb": 1.0}]).to_csv(tmp_path / "res" / "lm_bpb.csv", index=False)
    calls = []
    monkeypatch.setattr(run, "run_candidate", _fake_candidate(calls))
    run.run_all(_run_cfg(tmp_path), ["a", "b"])
    assert [c[0] for c in calls] == ["a", "b"] and not any(c[2] for c in calls)
    assert list(pd.read_csv(tmp_path / "res" / "lm_bpb.csv").bpb) == [0.5, 0.5]


def test_smoke_does_not_checkpoint(tmp_path, monkeypatch):
    from ruencod.lm import run
    calls = []
    monkeypatch.setattr(run, "run_candidate", _fake_candidate(calls))
    monkeypatch.setattr(run, "_smoke_line", lambda name, rows, full_bytes: None)
    run.run_all(_run_cfg(tmp_path), ["a"], smoke=True)
    assert calls[0][1] is None
