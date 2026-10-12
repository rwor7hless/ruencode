import argparse
import json

import pandas as pd

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import domain_shares, iter_domain, iter_mix, russian_domains
from ruencod.eval.adapters import resolve
from ruencod.lm.eval_bpb import nll_bits_and_bytes
from ruencod.lm.model import LMConfig, TinyLM
from ruencod.lm.train_lm import encode_corpus, train_lm


def run_candidate(cfg: dict, name: str, train_bytes: int) -> list[dict]:
    lm = cfg["lm"]
    tok = resolve(cfg, name)
    data = cfg_path(cfg, "data") / "corpus"
    tokens = encode_corpus(tok, iter_mix(data, "train", domain_shares(cfg), train_bytes))
    model = TinyLM(LMConfig(vocab_size=tok.vocab_size, d_model=lm["d_model"], n_layers=lm["n_layers"],
                            n_heads=lm["n_heads"], context=lm["context"], loss_chunk=lm.get("loss_chunk", 0)))
    stats = train_lm(model, tokens, lm, cfg["seed"])
    rows, ru_bits, ru_bytes = [], 0.0, 0
    for d in cfg["corpus"]["domains"]:
        texts = list(iter_domain(data, "heldout", d, lm["eval_bytes_per_domain"]))
        bits, nbytes = nll_bits_and_bytes(model, tok, texts, lm["context"])
        rows.append({"candidate": name, "domain": d, "bpb": bits / nbytes, **stats})
        if d in russian_domains(cfg):
            ru_bits, ru_bytes = ru_bits + bits, ru_bytes + nbytes
    rows.append({"candidate": name, "domain": "ru_all", "bpb": ru_bits / ru_bytes, **stats})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--candidates", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    names = args.candidates or cfg["lm"]["candidates"]
    full_bytes = cfg["lm"]["train_bytes"]
    out = cfg_path(cfg, "results") / ("lm_bpb_smoke.csv" if args.smoke else "lm_bpb.csv")
    all_rows = []
    for name in names:
        rows = run_candidate(cfg, name, 1_000_000 if args.smoke else full_bytes)
        all_rows += rows
        if args.smoke:
            r = rows[-1]
            est_h = r["train_tokens"] * (full_bytes / 1_000_000) / r["tokens_per_sec"] / 3600
            print(json.dumps({"candidate": name, "device": r["device"], "tokens_per_sec": round(r["tokens_per_sec"]),
                              "peak_gpu_gb": r["peak_gpu_gb"], "estimated_full_hours": round(est_h, 2)}), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out, index=False)  # сохраняем после каждого кандидата
    print(pd.DataFrame(all_rows).query("domain == 'ru_all'")[["candidate", "bpb", "train_tokens", "seconds"]])


if __name__ == "__main__":
    main()
