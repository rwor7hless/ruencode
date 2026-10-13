"""Архив только с теми текстами, которые читает мини-LM (~90 МБ вместо ~1.5 ГБ корпуса):
обучающий префикс каждого домена и отложенный текст для оценки. Нужен, чтобы учить модель
в другом месте (например, в облаке) на тех же данных, не собирая корпус заново."""
import argparse
import json
import tarfile
import tempfile
from pathlib import Path

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import domain_file, domain_shares, iter_domain
from ruencod.jsonl import ZstJsonlWriter


def export_slice(cfg: dict, out_tar: Path) -> Path:
    lm, src = cfg["lm"], cfg_path(cfg, "data") / "corpus"
    out_tar = Path(out_tar)
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "corpus"
        stats = {"export": "lm slice", "train_bytes": lm["train_bytes"],
                 "eval_bytes_per_domain": lm["eval_bytes_per_domain"]}
        for d, share in domain_shares(cfg).items():
            for split, budget in (("train", int(lm["train_bytes"] * share)), ("heldout", lm["eval_bytes_per_domain"])):
                n = 0
                with ZstJsonlWriter(domain_file(dst, split, d)) as w:
                    for t in iter_domain(src, split, d, budget):  # тот же префикс, что прочитают обучение и оценка
                        w.write({"text": t, "domain": d})
                        n += 1
                stats[f"{split}/{d}"] = n
        (dst / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        with tarfile.open(out_tar, "w") as tar:
            tar.add(dst, arcname="corpus")
    return out_tar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default="data/ruencode_lm_data.tar")
    args = ap.parse_args()
    out = export_slice(load_config(args.config), Path(args.out))
    print(f"{out} — {out.stat().st_size / 1e6:.1f} МБ", flush=True)


if __name__ == "__main__":
    main()
