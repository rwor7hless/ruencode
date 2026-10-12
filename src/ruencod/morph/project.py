import os

import pymorphy3
import regex

from ruencod.morph.tikhonov import Morpheme, boundaries, norm

CYR_WORD = regex.compile(r"[а-яёА-ЯЁ]+")
_POSTFIXES = ("ся", "сь")


def _ends(morphs: list[Morpheme]) -> list[int]:
    out, pos = [], 0
    for m in morphs:
        pos += len(m.text)
        out.append(pos)
    return out


class Projector:
    """Переносит морфемный разбор леммы из словаря Тихонова на словоформу."""

    def __init__(self, lexicon: dict[str, list[Morpheme]], exclude: set[str] | frozenset = frozenset()):
        self.lexicon = lexicon
        self.exclude = exclude
        self._morph = pymorphy3.MorphAnalyzer()

    def segment(self, word: str) -> list[int] | None:
        if not CYR_WORD.fullmatch(word):
            return None
        w = norm(word)
        if w in self.exclude:
            return None
        lemma = norm(self._morph.parse(word.lower())[0].normal_form)
        if lemma in self.exclude:
            return None
        morphs = self.lexicon.get(lemma)
        if morphs is None:
            return None
        lemma_str = "".join(m.text for m in morphs)
        # совпадение с леммой не может заходить в её окончание/постфикс: «книгами» vs «книга» → «книг»
        inflection = 0
        for m in reversed(morphs):
            if m.kind not in ("END", "POSTFIX"):
                break
            inflection += len(m.text)
        common = min(len(os.path.commonprefix([w, lemma_str])), len(lemma_str) - inflection)
        root_ends = [e for e, m in zip(_ends(morphs), morphs) if m.kind == "ROOT"]
        if not root_ends or common < root_ends[-1]:
            return None  # форма расходится с леммой внутри корня (чередование) — не рискуем
        bounds = {b for b in boundaries(morphs) if b <= common}
        if common < len(w):
            bounds.add(common)
        if morphs[-1].kind == "POSTFIX" and w.endswith(_POSTFIXES) and len(w) - 2 > common:
            bounds.add(len(w) - 2)
        return sorted(b for b in bounds if 0 < b < len(w))
