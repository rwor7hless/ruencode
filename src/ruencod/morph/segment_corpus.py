import argparse
import itertools
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from tqdm import tqdm

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import domain_file, iter_domain
from ruencod.jsonl import ZstJsonlWriter
from ruencod.morph.project import CYR_WORD, Projector
from ruencod.morph.tikhonov import download, load_lexicon, split_lemmas
from ruencod.pretok import MARK, pretokenize

_CTX = mp.get_context("fork")
_PROJ: Projector | None = None
_CACHE: dict[str, list[int]] | None = None


def word_core(pt: str) -> tuple[int, str]:
    start = 0 if pt[:1].isalpha() else 1
    return start, pt[start:]


def segment_text(text: str, cache: dict[str, list[int]]) -> str:
    out: list[str] = []
    for pt in pretokenize(text):
        start, core = word_core(pt)
        bounds = cache.get(core.lower()) if core else None
        if bounds:
            cuts = [0] + [start + b for b in bounds] + [len(pt)]
            out.extend(pt[a:b] for a, b in zip(cuts, cuts[1:]))
        else:
            out.append(pt)
    return MARK.join(out)


def _batched(it: Iterable, n: int) -> Iterator[list]:
    it = iter(it)
    while batch := list(itertools.islice(it, n)):
        yield batch


def _count_batch(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        for pt in pretokenize(t):
            _, core = word_core(pt)
            if core and CYR_WORD.fullmatch(core):
                c[core.lower()] += 1
    return c


def count_words(texts: Iterable[str], workers: int) -> Counter:
    total: Counter = Counter()
    with _CTX.Pool(workers) as pool:
        for c in tqdm(pool.imap_unordered(_count_batch, _batched(texts, 2000)), desc="count words"):
            total.update(c)
    return total


def _init_projector(lex_path: Path, exclude: set[str]) -> None:
    global _PROJ
    _PROJ = Projector(load_lexicon(lex_path), exclude)


def _segment_words(words: list[str]) -> list[tuple[str, list[int] | None]]:
    return [(w, _PROJ.segment(w)) for w in words]


def build_cache(words: Iterable[str], lex_path: Path, exclude: set[str], workers: int):
    cache: dict[str, list[int]] = {}
    covered: set[str] = set()
    with _CTX.Pool(workers, initializer=_init_projector, initargs=(lex_path, exclude)) as pool:
        for res in tqdm(pool.imap_unordered(_segment_words, _batched(words, 5000)), desc="segment words"):
            for w, b in res:
                if b is not None:
                    covered.add(w)
                    if b:
                        cache[w] = b
    return cache, covered


def save_cache(cache: dict[str, list[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for w, b in sorted(cache.items()):
            f.write(f"{w}\t{','.join(map(str, b))}\n")


def load_cache(path: Path) -> dict[str, list[int]]:
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            w, b = line.rstrip("\n").split("\t")
            cache[w] = [int(x) for x in b.split(",")]
    return cache


def _init_seg(cache_path: Path) -> None:
    global _CACHE
    _CACHE = load_cache(cache_path)


def _segment_batch(texts: list[str]) -> list[str]:
    return [segment_text(t, _CACHE) for t in texts]


def run(cfg: dict) -> dict:
    data, mcfg = cfg_path(cfg, "data"), cfg["morph"]
    corpus_dir, morph_dir = data / "corpus", data / "morph"
    domains = list(cfg["corpus"]["domains"])

    lex_path = download(morph_dir / "tikhonov.tsv", mcfg["tikhonov_urls"])
    lexicon = load_lexicon(lex_path)
    _, test = split_lemmas(lexicon, cfg["seed"], mcfg["test_frac"])
    (morph_dir / "test_lemmas.txt").write_text("\n".join(sorted(test)) + "\n", encoding="utf-8")

    def all_train():
        for d in domains:
            yield from iter_domain(corpus_dir, "train", d)

    counts = count_words(all_train(), mcfg["workers"])
    frequent = [w for w, c in counts.items() if c >= mcfg["min_word_freq"]]
    cache, covered = build_cache(frequent, lex_path, test, mcfg["workers"])
    cache_path = morph_dir / "segmentations.tsv"
    save_cache(cache, cache_path)

    total = sum(counts.values())
    coverage = {
        "covered_token_share": sum(counts[w] for w in covered) / total if total else 0.0,
        "unique_words": len(counts),
        "cached_words": len(cache),
    }
    results = cfg_path(cfg, "results")
    results.mkdir(parents=True, exist_ok=True)
    (results / "morph_coverage.json").write_text(json.dumps(coverage, indent=2))

    with _CTX.Pool(mcfg["workers"], initializer=_init_seg, initargs=(cache_path,)) as pool:
        for d in domains:
            with ZstJsonlWriter(domain_file(data / "morph_corpus", "train", d)) as w:
                for batch in tqdm(pool.imap(_segment_batch, _batched(iter_domain(corpus_dir, "train", d), 1000)),
                                  desc=f"segment {d}"):
                    for t in batch:
                        w.write({"text": t})
    return coverage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    print(json.dumps(run(load_config(ap.parse_args().config)), indent=2))


if __name__ == "__main__":
    main()
