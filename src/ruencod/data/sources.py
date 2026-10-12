import io
import json
import random
from typing import Iterator
from urllib.parse import urlparse

import pyarrow.compute as pc
import pyarrow.parquet as pq
import zstandard as zstd


def url_matches(url: str | None, domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "") if url else ""
    return any(host == d or host.endswith("." + d) for d in domains)


def pikabu_docs(rec: dict) -> list[str]:
    docs = []
    post = "\n\n".join(x for x in (rec.get("title") or "", rec.get("text_markdown") or "") if x.strip())
    if post:
        docs.append(post)
    comments = [c for c in ((rec.get("comments") or {}).get("text_markdown") or []) if c and c.strip()]
    if comments:
        docs.append("\n".join(comments))
    return docs


def _iter_hf(src: dict, seed: int) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(src["path"], src.get("name"), split=src.get("split", "train"), streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=src.get("shuffle_buffer", 10_000))
    flt = src.get("filter") or {}
    url_domains = src.get("url_domains")
    for row in ds:
        if any(row.get(k) != v for k, v in flt.items()):
            continue
        if url_domains and not url_matches(row.get("url"), url_domains):
            continue
        text = row.get(src["text"])
        if text:
            yield text


def _iter_pikabu(seed: int) -> Iterator[str]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob("datasets/IlyaGusev/pikabu/*.jsonl.zst"))
    random.Random(seed).shuffle(files)
    for path in files:
        with fs.open(path, "rb") as f:
            reader = zstd.ZstdDecompressor().stream_reader(f)
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if line.strip():
                    yield from pikabu_docs(json.loads(line))


def _iter_parquet(src: dict, seed: int, fs=None) -> Iterator[str]:
    """Читает из parquet только нужные колонки (row group целиком со всеми колонками не помещается в RAM)."""
    root = src["path"]
    if fs is None:  # по умолчанию — репозиторий датасета на HF Hub
        from huggingface_hub import HfFileSystem
        fs, root = HfFileSystem(), f"datasets/{src['path']}"
    files = sorted(fs.glob(f"{root}/{src.get('glob', '**/*.parquet')}"))
    rng = random.Random(seed)
    rng.shuffle(files)
    flt = src.get("filter") or {}
    columns = [src["text"], *flt]
    for path in files:
        with fs.open(path, "rb") as f:
            table = pq.read_table(f, columns=columns)
        for k, v in flt.items():
            table = table.filter(pc.equal(table[k], v))
        texts = table[src["text"]].to_pylist()
        rng.shuffle(texts)
        yield from (t for t in texts if t)


def iter_source(src: dict, seed: int, fs=None) -> Iterator[str]:
    kind = src.get("kind", "hf")
    if kind == "hf":
        yield from _iter_hf(src, seed)
    elif kind == "parquet":
        yield from _iter_parquet(src, seed, fs)
    elif kind == "pikabu":
        yield from _iter_pikabu(seed)
    else:
        raise ValueError(f"unknown source kind: {kind}")
