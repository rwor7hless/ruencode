import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Morpheme:
    text: str
    kind: str


def norm(s: str) -> str:
    return s.lower().replace("ё", "е")


def parse_line(line: str) -> tuple[str, list[Morpheme]] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 2:
        return None
    word, seg = parts
    morphs = []
    for p in seg.split("/"):
        text, sep, kind = p.rpartition(":")
        if not sep or not text:
            return None
        morphs.append(Morpheme(norm(text), kind))
    if "".join(m.text for m in morphs) != norm(word):
        return None
    return norm(word), morphs


def boundaries(morphs: list[Morpheme]) -> list[int]:
    out, pos = [], 0
    for m in morphs[:-1]:
        pos += len(m.text)
        out.append(pos)
    return out


def load_lexicon(path: Path) -> dict[str, list[Morpheme]]:
    lex = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parsed = parse_line(line)
        if parsed:
            lex[parsed[0]] = parsed[1]
    return lex


def split_lemmas(lemmas: Iterable[str], seed: int, test_frac: float) -> tuple[set[str], set[str]]:
    items = sorted(set(lemmas))
    random.Random(seed).shuffle(items)
    n_test = int(len(items) * test_frac)
    return set(items[n_test:]), set(items[:n_test])


def download(dest: Path, urls: list[str]) -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        chunks = [urllib.request.urlopen(u, timeout=60).read().decode("utf-8") for u in urls]
        dest.write_text("\n".join(c.strip("\n") for c in chunks) + "\n", encoding="utf-8")
    return dest
