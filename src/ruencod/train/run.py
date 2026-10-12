import argparse
import json
import time
from pathlib import Path

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import domain_shares, iter_mix
from ruencod.train.tokenizer import save_tokenizer, train_tokenizer


def tokenizer_name(algo: str, vocab: int) -> str:
    return f"{algo}-{vocab // 1000}k"


def train_one(cfg: dict, algo: str, vocab: int, force: bool = False) -> Path:
    out = cfg_path(cfg, "artifacts") / "tokenizers" / tokenizer_name(algo, vocab)
    if (out / "tokenizer.json").exists() and not force:
        print(f"skip {out.name}: exists")
        return out
    data, shares, tcfg = cfg_path(cfg, "data"), domain_shares(cfg), cfg["train"]
    if algo == "morph-bpe":
        limit = tcfg.get("bpe_max_bytes")
        texts, kind, morph = iter_mix(data / "morph_corpus", "train", shares, limit), "bpe", True
    elif algo == "unigram":
        limit = tcfg["unigram_max_bytes"]
        texts, kind, morph = iter_mix(data / "corpus", "train", shares, limit), "unigram", False
    elif algo == "bpe":
        limit = tcfg.get("bpe_max_bytes")
        texts, kind, morph = iter_mix(data / "corpus", "train", shares, limit), "bpe", False
    else:
        raise ValueError(algo)
    t0 = time.time()
    tok = train_tokenizer(texts, kind, vocab, morph=morph, min_frequency=tcfg["bpe_min_frequency"])
    save_tokenizer(tok, out)
    meta = {"algo": algo, "vocab_requested": vocab, "vocab_actual": tok.get_vocab_size(),
            "seconds": round(time.time() - t0, 1), "train_bytes_limit": limit}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"name": out.name, **meta}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--algo")
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.all:
        for algo in cfg["train"]["algos"]:
            for v in sorted(cfg["train"]["vocab_sizes"], reverse=True):
                train_one(cfg, algo, v, args.force)
    else:
        train_one(cfg, args.algo, args.vocab, args.force)


if __name__ == "__main__":
    main()
