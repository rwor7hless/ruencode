import argparse
import hashlib
import json
import random

from tqdm import tqdm

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import domain_file
from ruencod.data.clean import accept, chunk_long, normalize
from ruencod.data.sources import iter_source
from ruencod.jsonl import ZstJsonlWriter


def build(cfg: dict, domains: list[str] | None = None, iter_source_fn=iter_source) -> dict:
    corpus = cfg["corpus"]
    out = cfg_path(cfg, "data") / "corpus"
    seen: set[bytes] = set()
    stats: dict = {}
    for domain, dcfg in corpus["domains"].items():
        if domains and domain not in domains:
            continue
        train_target = int(corpus["total_bytes"] * dcfg["share"])
        held_target = int(corpus["heldout_bytes_per_domain"])
        srcs = dcfg["sources"]
        ds = {"train_bytes": 0, "heldout_bytes": 0, "docs": 0, "rejected": 0, "duplicates": 0, "shortfall": []}
        with ZstJsonlWriter(domain_file(out, "train", domain)) as wt, \
             ZstJsonlWriter(domain_file(out, "heldout", domain)) as wh:
            for i, src in enumerate(srcs):
                share = src.get("share", 1 / len(srcs))
                st, sh = int(train_target * share), int(held_target * share)
                p_held = sh / (st + sh) if st + sh else 0.0
                rng = random.Random(f"{cfg['seed']}:{domain}:{i}")
                bt = bh = 0
                bar = tqdm(total=st + sh, unit="B", unit_scale=True, desc=f"{domain}[{i}]")
                for raw in iter_source_fn(src, cfg["seed"]):
                    if bt >= st and bh >= sh:
                        break
                    text = normalize(raw, html_source=src.get("html", False))
                    for chunk in chunk_long(text, corpus["max_chars"]):
                        if not accept(chunk, min_chars=corpus["min_chars"], max_chars=corpus["max_chars"],
                                      russian=dcfg.get("russian", True),
                                      min_cyrillic_ratio=corpus["min_cyrillic_ratio"]):
                            ds["rejected"] += 1
                            continue
                        h = hashlib.blake2b(chunk.encode("utf-8"), digest_size=16).digest()
                        if h in seen:
                            ds["duplicates"] += 1
                            continue
                        seen.add(h)
                        n = len(chunk.encode("utf-8"))
                        to_held = bh < sh and (bt >= st or rng.random() < p_held)
                        (wh if to_held else wt).write({"text": chunk, "domain": domain})
                        if to_held:
                            bh += n
                        else:
                            bt += n
                        ds["docs"] += 1
                        bar.update(n)
                        if bt >= st and bh >= sh:
                            break
                bar.close()
                if bt < st or bh < sh:
                    ds["shortfall"].append({"source": src.get("path", src.get("kind")),
                                            "train_missing": max(0, st - bt),
                                            "heldout_missing": max(0, sh - bh)})
                ds["train_bytes"] += bt
                ds["heldout_bytes"] += bh
        stats[domain] = ds
    stats_path = out / "stats.json"
    old = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    old.update(stats)
    stats_path.write_text(json.dumps(old, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--domains", nargs="*")
    args = ap.parse_args()
    stats = build(load_config(args.config), args.domains)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
