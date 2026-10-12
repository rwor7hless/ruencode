import argparse
import json

import pandas as pd

from ruencod.config import DEFAULT_CONFIG, ROOT, cfg_path, load_config
from ruencod.eval.adapters import resolve
from ruencod.eval.claude_count import ClaudeCounter
from ruencod.viz.charts import _fert_metric


def make(cfg: dict, claude_client=None) -> dict:
    res = cfg_path(cfg, "results")
    df = pd.read_csv(res / "metrics.csv")
    metric = _fert_metric(df)
    f = df[(df.domain == "ru_all") & (df.metric == metric)].set_index("tokenizer").value
    final, gpt, claude = cfg["final"]["tokenizer"], cfg["prices"]["gpt"], cfg["prices"]["claude"]
    claude_on = bool(cfg["eval"]["claude"]["models"]) and claude["tokenizer"] in f
    s = {
        "final": final,
        "fertility_metric": metric,
        "fertility": {"final": f[final], "gpt": f[gpt["tokenizer"]]},
        "savings_vs_gpt_pct": 100 * (1 - f[final] / f[gpt["tokenizer"]]),
        "usd_per_1000_words": {
            "gpt": f[gpt["tokenizer"]] * gpt["usd_per_mtok"] / 1000,
            "final_at_gpt_price": f[final] * gpt["usd_per_mtok"] / 1000,
        },
        "prices_verified_on": cfg["prices"]["verified_on"],
    }
    if claude_on:
        s["fertility"]["claude"] = f[claude["tokenizer"]]
        s["savings_vs_claude_pct"] = 100 * (1 - f[final] / f[claude["tokenizer"]])
        s["usd_per_1000_words"]["claude"] = f[claude["tokenizer"]] * claude["usd_per_mtok"] / 1000
        s["usd_per_1000_words"]["final_at_claude_price"] = f[final] * claude["usd_per_mtok"] / 1000
    cov = res / "morph_coverage.json"
    if cov.exists():
        s["morph_coverage"] = json.loads(cov.read_text())
    if (res / "lm_bpb.csv").exists():
        lm = pd.read_csv(res / "lm_bpb.csv").query("domain == 'ru_all'")
        s["lm_bpb"] = dict(zip(lm.candidate, lm.bpb))
    (res / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))

    ours, o200k = resolve(cfg, final), resolve(cfg, gpt["tokenizer"])
    examples = [{"text": t, "gpt_tokens": len(o200k.encode(t)), "ours_tokens": len(ours.encode(t))}
                for t in cfg["examples"]]
    if claude_on:
        counter = ClaudeCounter(claude["tokenizer"], res / "claude_cache.json", client=claude_client)
        for e in examples:
            e["claude_tokens"] = counter.count(e["text"])
        counter.save()
    web = ROOT / "web"
    web.mkdir(exist_ok=True)
    (web / "examples.json").write_text(json.dumps(
        {"claude_label": claude["label"] if claude_on else None, "summary": s, "examples": examples}, ensure_ascii=False, indent=2))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    print(json.dumps(make(load_config(ap.parse_args().config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
