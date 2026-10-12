import pandas as pd

from ruencod.corpus import domain_file
from ruencod.eval.run import run_eval
from ruencod.jsonl import ZstJsonlWriter
from ruencod.train.tokenizer import save_tokenizer, train_tokenizer
from tests.conftest import make_corpus
from tests.test_claude_count import FakeClient


def _cfg(tmp_path):
    return {
        "seed": 1,
        "paths": {"data": str(tmp_path / "data"), "artifacts": str(tmp_path / "art"), "results": str(tmp_path / "res")},
        "corpus": {"domains": {"web": {"share": 0.5, "russian": True}, "en_code": {"share": 0.5, "russian": False}}},
        "morph": {"test_frac": 0.5},
        "eval": {"max_bytes_per_domain": 10**6, "baselines": [],
                 "claude": {"models": ["claude-opus-5"], "docs_per_domain": 3}},
    }


def test_run_eval_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    data = tmp_path / "data"
    with ZstJsonlWriter(domain_file(data / "corpus", "heldout", "web")) as w:
        for t in make_corpus(20, seed=5):
            w.write({"text": t})
    with ZstJsonlWriter(domain_file(data / "corpus", "heldout", "en_code")) as w:
        w.write({"text": "def f(x):\n    return 1\n"})
    (data / "morph").mkdir(parents=True)
    (data / "morph" / "tikhonov.tsv").write_text("книга\tкниг:ROOT/а:END\n", encoding="utf-8")
    (data / "morph" / "test_lemmas.txt").write_text("книга\n", encoding="utf-8")
    save_tokenizer(train_tokenizer(make_corpus(), "bpe", 700), tmp_path / "art" / "tokenizers" / "bpe-0k")

    df = run_eval(cfg, claude_client=FakeClient())
    assert set(df["tokenizer"]) == {"bpe-0k", "claude-opus-5"}
    get = lambda tok, dom, m: df[(df.tokenizer == tok) & (df.domain == dom) & (df.metric == m)].value.item()
    assert get("bpe-0k", "ru_all", "fertility") > 0
    assert get("bpe-0k", "web", "roundtrip") == 1.0
    assert 0 <= get("bpe-0k", "morph_test", "morph_f1") <= 1
    assert get("claude-opus-5", "ru_all", "fertility_claude_sample") > 0
    assert get("bpe-0k", "ru_all", "fertility_claude_sample") > 0
    assert pd.read_csv(tmp_path / "res" / "metrics.csv").shape[0] == df.shape[0]


def test_sample_mode_keeps_full_metrics(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["eval"]["claude"]["docs_per_domain"] = 0
    data = tmp_path / "data"
    with ZstJsonlWriter(domain_file(data / "corpus", "heldout", "web")) as w:
        for t in make_corpus(20, seed=5):
            w.write({"text": t})
    save_tokenizer(train_tokenizer(make_corpus(), "bpe", 700), tmp_path / "art" / "tokenizers" / "bpe-0k")
    run_eval(cfg)  # полная оценка без Claude

    cfg["eval"]["claude"]["docs_per_domain"] = 3
    df = run_eval(cfg, only="sample", claude_client=FakeClient())
    has = lambda tok, m: ((df.tokenizer == tok) & (df.metric == m)).any()
    assert has("bpe-0k", "fertility") and has("bpe-0k", "roundtrip"), "полные метрики сохранились"
    assert has("bpe-0k", "fertility_claude_sample") and has("claude-opus-5", "fertility_claude_sample")
    assert not df.duplicated(["tokenizer", "domain", "metric"]).any()
