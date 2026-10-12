from pathlib import Path
from typing import Iterator

from ruencod.jsonl import iter_jsonl_zst


def domain_file(base: Path, split: str, domain: str) -> Path:
    return Path(base) / split / f"{domain}.jsonl.zst"


def iter_domain(base: Path, split: str, domain: str, max_bytes: int | None = None) -> Iterator[str]:
    path = domain_file(base, split, domain)
    if not path.exists():
        return
    used = 0
    for rec in iter_jsonl_zst(path):
        if max_bytes is not None and used >= max_bytes:
            return
        text = rec["text"]
        used += len(text.encode("utf-8"))
        yield text


def iter_mix(base: Path, split: str, shares: dict[str, float], max_bytes: int | None = None) -> Iterator[str]:
    for domain, share in shares.items():
        budget = None if max_bytes is None else int(max_bytes * share)
        yield from iter_domain(base, split, domain, budget)


def domain_shares(cfg: dict) -> dict[str, float]:
    return {d: v["share"] for d, v in cfg["corpus"]["domains"].items()}


def russian_domains(cfg: dict) -> list[str]:
    return [d for d, v in cfg["corpus"]["domains"].items() if v.get("russian", True)]
