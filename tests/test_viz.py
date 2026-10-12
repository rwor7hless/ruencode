import pandas as pd

from ruencod.viz.charts import make_all
from ruencod.viz.splits import render

OURS = [("bpe-16k", 2.0), ("bpe-32k", 1.8), ("unigram-16k", 2.1), ("unigram-32k", 1.9),
        ("morph-bpe-16k", 2.3), ("morph-bpe-32k", 2.1)]


def _cfg(tmp_path):
    return {
        "paths": {"results": str(tmp_path)},
        "corpus": {"domains": {"web": {"share": 1.0, "russian": True}}},
        "train": {"vocab_sizes": [16000, 32000], "algos": ["bpe", "unigram", "morph-bpe"]},
        "eval": {"baselines": [{"name": "o200k", "label": "GPT-4o (o200k)"}]},
        "prices": {"verified_on": "2026-10-12",
                   "gpt": {"label": "GPT-4o", "tokenizer": "o200k", "usd_per_mtok": 2.5},
                   "claude": {"label": "Claude Opus 5", "tokenizer": "claude-opus-5", "usd_per_mtok": 5.0}},
        "final": {"tokenizer": "bpe-32k"},
    }


def _write(tmp_path, with_claude=True):
    rows = []
    toks = [(t, "ours", f) for t, f in OURS] + [("o200k", "baseline", 2.6)]
    if with_claude:
        toks.append(("claude-opus-5", "claude", 3.0))
    for tok, kind, f in toks:
        for dom in ("web", "ru_all"):
            if with_claude:
                rows.append({"tokenizer": tok, "kind": kind, "domain": dom, "metric": "fertility_claude_sample", "value": f})
            if kind != "claude":
                rows.append({"tokenizer": tok, "kind": kind, "domain": dom, "metric": "fertility", "value": f})
    pd.DataFrame(rows).to_csv(tmp_path / "metrics.csv", index=False)


def test_charts(tmp_path):
    _write(tmp_path)
    pd.DataFrame([{"candidate": "bpe-32k", "domain": "ru_all", "bpb": 1.1}]).to_csv(tmp_path / "lm_bpb.csv", index=False)
    paths = make_all(_cfg(tmp_path))
    assert len(paths) == 5 and all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_charts_without_claude_sample(tmp_path):
    # smoke-конфиг не считает Claude: графики, которым он нужен, строятся по fertility без Claude
    _write(tmp_path, with_claude=False)
    paths = make_all(_cfg(tmp_path))
    assert len(paths) == 4 and all(p.exists() for p in paths)


def test_render_split(tmp_path):
    out = render([("GPT-4o", [("достоприм", 1), ("ечательно", 2), ("стями", 1)]),
                  ("Claude Opus 5", "7 токенов"),
                  ("ruEncod", [("достопримечательн", 1), ("остями", 1)])],
                 "достопримечательностями", tmp_path / "s.png")
    assert out.exists() and out.stat().st_size > 1000
