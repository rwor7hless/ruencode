import html
import re
import unicodedata

from ruencod.pretok import MARK

_BLOCK_TAG = re.compile(r"</?(?:p|br|div|li|h[1-6]|blockquote|tr)\b[^<>]*>", re.I)
_TAG = re.compile(r"<[a-zA-Z/!][^<>]{0,500}>")
_MULTI_NL = re.compile(r"\n{3,}")


def _strip_html(text: str) -> str:
    text = _BLOCK_TAG.sub("\n\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    return _MULTI_NL.sub("\n\n", text)


def normalize(text: str, html_source: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    if html_source:
        text = _strip_html(text)
    return text.replace(MARK, "").strip()


def cyrillic_ratio(text: str) -> float:
    letters = cyr = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if "Ѐ" <= ch <= "ӿ":
                cyr += 1
    return cyr / letters if letters else 0.0


def accept(text: str, *, min_chars: int, max_chars: int, russian: bool, min_cyrillic_ratio: float) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    return not russian or cyrillic_ratio(text) >= min_cyrillic_ratio


def chunk_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for para in text.split("\n\n"):
        while len(para) > max_chars:
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_len = [], 0
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        add = len(para) + (2 if cur else 0)
        if cur and cur_len + add > max_chars:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [para], len(para)
        else:
            cur.append(para)
            cur_len += add
    if cur:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if c.strip()]
