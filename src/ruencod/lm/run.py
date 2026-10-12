import argparse
import json
import os

# до первого обращения к CUDA: меньше фрагментации видеопамяти (на 8 ГБ это ~1 ГБ запаса)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import domain_shares, iter_domain, iter_mix, russian_domains
from ruencod.eval.adapters import resolve
from ruencod.lm.eval_bpb import nll_bits_and_bytes
from ruencod.lm.model import LMConfig, TinyLM
from ruencod.lm.train_lm import encode_corpus, train_lm


def run_candidate(cfg: dict, name: str, train_bytes: int, ckpt_path=None, resume: bool = False) -> list[dict]:
    lm = cfg["lm"]
    tok = resolve(cfg, name)
    data = cfg_path(cfg, "data") / "corpus"
    tokens = encode_corpus(tok, iter_mix(data, "train", domain_shares(cfg), train_bytes))
    model = TinyLM(LMConfig(vocab_size=tok.vocab_size, d_model=lm["d_model"], n_layers=lm["n_layers"],
                            n_heads=lm["n_heads"], context=lm["context"], loss_chunk=lm.get("loss_chunk", 0),
                            checkpoint_blocks=lm.get("checkpoint_blocks", False)))
    stats = train_lm(model, tokens, lm, cfg["seed"], ckpt_path=ckpt_path, resume=resume)
    rows, ru_bits, ru_bytes = [], 0.0, 0
    for d in cfg["corpus"]["domains"]:
        texts = list(iter_domain(data, "heldout", d, lm["eval_bytes_per_domain"]))
        bits, nbytes = nll_bits_and_bytes(model, tok, texts, lm["context"])
        rows.append({"candidate": name, "domain": d, "bpb": bits / nbytes, **stats})
        if d in russian_domains(cfg):
            ru_bits, ru_bytes = ru_bits + bits, ru_bytes + nbytes
    rows.append({"candidate": name, "domain": "ru_all", "bpb": ru_bits / ru_bytes, **stats})
    return rows


def _smoke_line(name: str, rows: list[dict], full_bytes: int) -> None:
    r = rows[-1]
    est_h = r["train_tokens"] * (full_bytes / 1_000_000) / r["tokens_per_sec"] / 3600
    print(json.dumps({"candidate": name, "device": r["device"], "tokens_per_sec": round(r["tokens_per_sec"]),
                      "peak_gpu_gb": r["peak_gpu_gb"], "estimated_full_hours": round(est_h, 2)}), flush=True)


def run_all(cfg: dict, names: list[str], smoke: bool = False, resume: bool = False) -> pd.DataFrame:
    """resume: кандидаты, уже записанные в CSV, пропускаются; прерванный продолжается с сохранения."""
    full_bytes = cfg["lm"]["train_bytes"]
    out = cfg_path(cfg, "results") / ("lm_bpb_smoke.csv" if smoke else "lm_bpb.csv")
    ckpt_dir = cfg_path(cfg, "data") / "lm_ckpt"
    all_rows = pd.read_csv(out).to_dict("records") if resume and out.exists() else []
    finished = {r["candidate"] for r in all_rows}
    for name in names:
        if name in finished:
            print(f"{name}: уже в {out.name}, пропускаю", flush=True)
            continue
        ckpt = None if smoke else ckpt_dir / f"{name}.pt"
        rows = run_candidate(cfg, name, 1_000_000 if smoke else full_bytes, ckpt_path=ckpt, resume=resume)
        all_rows += rows
        if smoke:
            _smoke_line(name, rows, full_bytes)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out, index=False)  # сохраняем после каждого кандидата
        if ckpt is not None:
            ckpt.unlink(missing_ok=True)  # результат записан — сохранение больше не нужно
    return pd.DataFrame(all_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", help="продолжить прерванный прогон")
    ap.add_argument("--candidates", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = run_all(cfg, args.candidates or cfg["lm"]["candidates"], smoke=args.smoke, resume=args.resume)
    print(df.query("domain == 'ru_all'")[["candidate", "bpb", "train_tokens", "seconds"]])


if __name__ == "__main__":
    main()
