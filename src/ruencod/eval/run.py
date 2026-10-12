import argparse
from collections import Counter

import pandas as pd
from tqdm import tqdm

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import iter_domain, russian_domains
from ruencod.eval.adapters import list_ours, load_baseline, load_ours
from ruencod.eval.claude_count import ClaudeCounter
from ruencod.eval.metrics import WORD_RE, measure, morph_scores, safe_div, single_token_rate, summarize
from ruencod.morph.tikhonov import boundaries, load_lexicon

COLUMNS = ["tokenizer", "kind", "domain", "metric", "value"]


def _rows(name, kind, domain, metrics: dict) -> list[dict]:
    return [{"tokenizer": name, "kind": kind, "domain": domain, "metric": k, "value": v} for k, v in metrics.items()]


def _load_heldout(cfg):
    base = cfg_path(cfg, "data") / "corpus"
    limit = cfg["eval"]["max_bytes_per_domain"]
    return {d: list(iter_domain(base, "heldout", d, limit)) for d in cfg["corpus"]["domains"]}


def _morph_items(cfg):
    mdir = cfg_path(cfg, "data") / "morph"
    if not (mdir / "test_lemmas.txt").exists():
        return []
    lex = load_lexicon(mdir / "tikhonov.tsv")
    test = (mdir / "test_lemmas.txt").read_text(encoding="utf-8").split()
    return [(w, boundaries(lex[w])) for w in test if w in lex]


def _sample_rows(name, kind, tokens_by_domain: dict, claude_sample: dict, ru) -> list[dict]:
    rows, tot_t, tot_w = [], 0, 0
    for d, texts in claude_sample.items():
        words = sum(len(WORD_RE.findall(t)) for t in texts)
        rows += _rows(name, kind, d, {"fertility_claude_sample": safe_div(tokens_by_domain[d], words)})
        if d in ru:
            tot_t, tot_w = tot_t + tokens_by_domain[d], tot_w + words
    if claude_sample:
        rows += _rows(name, kind, "ru_all", {"fertility_claude_sample": safe_div(tot_t, tot_w)})
    return rows


def _evaluate(tok, kind, heldout, ru, word_counts, morph_items, claude_sample) -> list[dict]:
    rows, agg = [], Counter()
    for d, texts in heldout.items():
        raw = measure(tok, texts)
        m = summarize(raw, tok.vocab_size)
        m["single_token_rate"] = single_token_rate(tok, word_counts[d])
        rows += _rows(tok.name, kind, d, m)
        if d in ru:
            for k in ("tokens", "words", "chars", "bytes"):
                agg[k] += raw[k]
    rows += _rows(tok.name, kind, "ru_all", {
        "fertility": safe_div(agg["tokens"], agg["words"]),
        "chars_per_token": safe_div(agg["chars"], agg["tokens"]),
        "bytes_per_token": safe_div(agg["bytes"], agg["tokens"]),
    })
    if morph_items:
        rows += _rows(tok.name, kind, "morph_test", morph_scores(tok, morph_items))
    tokens = {d: sum(map(len, tok.encode_batch(t))) for d, t in claude_sample.items()}
    rows += _sample_rows(tok.name, kind, tokens, claude_sample, ru)
    return rows


def run_eval(cfg: dict, only: str = "all", claude_client=None) -> pd.DataFrame:
    heldout = _load_heldout(cfg)
    ru = set(russian_domains(cfg))
    word_counts = {d: Counter(w for t in texts for w in WORD_RE.findall(t)) for d, texts in heldout.items()}
    morph_items = _morph_items(cfg)
    n = cfg["eval"]["claude"]["docs_per_domain"]
    claude_sample = {d: texts[:n] for d, texts in heldout.items()} if n else {}

    toks = []
    if only in ("all", "ours"):
        toks += [(load_ours(p), "ours") for p in list_ours(cfg)]
    if only in ("all", "baselines"):
        toks += [(load_baseline(s), "baseline") for s in cfg["eval"]["baselines"]]

    rows, evaluated = [], set()
    for tok, kind in tqdm(toks, desc="tokenizers"):
        rows += _evaluate(tok, kind, heldout, ru, word_counts, morph_items, claude_sample)
        evaluated.add(tok.name)

    if only in ("all", "claude") and claude_sample:
        for model in cfg["eval"]["claude"]["models"]:
            counter = ClaudeCounter(model, cfg_path(cfg, "results") / "claude_cache.json", client=claude_client)
            tokens = {d: sum(counter.count_many(texts)) for d, texts in claude_sample.items()}
            rows += _sample_rows(model, "claude", tokens, claude_sample, ru)
            evaluated.add(model)

    new = pd.DataFrame(rows, columns=COLUMNS)
    out = cfg_path(cfg, "results") / "metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat([old[~old.tokenizer.isin(evaluated)], new], ignore_index=True)
    new.to_csv(out, index=False)
    return new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--only", default="all", choices=["all", "ours", "baselines", "claude"])
    args = ap.parse_args()
    df = run_eval(load_config(args.config), args.only)
    view = df[(df.domain == "ru_all") & (df.metric.isin(["fertility", "fertility_claude_sample"]))]
    print(view.pivot_table(index="tokenizer", columns="metric", values="value").sort_values(view.metric.iloc[0]))


if __name__ == "__main__":
    main()
