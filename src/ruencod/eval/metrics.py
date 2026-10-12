import itertools
from collections import Counter

import regex

WORD_RE = regex.compile(r"\p{L}+")
NAN = float("nan")


def safe_div(a: float, b: float, default: float = NAN) -> float:
    return a / b if b else default


def measure(tok, texts, batch_size: int = 256) -> dict:
    raw = {"tokens": 0, "words": 0, "chars": 0, "bytes": 0, "docs": 0, "roundtrip_ok": 0, "used_ids": set()}
    it = iter(texts)
    while batch := list(itertools.islice(it, batch_size)):
        for text, ids in zip(batch, tok.encode_batch(batch)):
            raw["tokens"] += len(ids)
            raw["words"] += len(WORD_RE.findall(text))
            raw["chars"] += len(text)
            raw["bytes"] += len(text.encode("utf-8"))
            raw["docs"] += 1
            raw["roundtrip_ok"] += tok.decode(ids) == text
            raw["used_ids"].update(ids)
    return raw


def summarize(raw: dict, vocab_size: int) -> dict:
    return {
        "fertility": safe_div(raw["tokens"], raw["words"]),
        "chars_per_token": safe_div(raw["chars"], raw["tokens"]),
        "bytes_per_token": safe_div(raw["bytes"], raw["tokens"]),
        "roundtrip": safe_div(raw["roundtrip_ok"], raw["docs"]),
        "vocab_usage": safe_div(len(raw["used_ids"]), vocab_size),
    }


def single_token_rate(tok, word_counts: Counter) -> float:
    words = list(word_counts)
    one = total = 0
    for w, ids in zip(words, tok.encode_batch([" " + w for w in words])):
        total += word_counts[w]
        if len(ids) == 1:
            one += word_counts[w]
    return safe_div(one, total)


def _aligned_pos(text: str, prefix: str) -> int | None:
    if text.startswith(prefix):
        return len(prefix)
    if text.startswith(" " + prefix):
        return len(prefix) + 1
    return None


def token_char_boundaries(tok, text: str, ids: list[int]) -> set[int]:
    out = set()
    for i in range(1, len(ids)):
        pos = _aligned_pos(text, tok.decode(ids[:i]))
        if pos is not None:
            out.add(pos)
    return out


def aligned_pieces(tok, text: str) -> list[tuple[str, int]]:
    ids = tok.encode(text)
    pieces, last_pos, last_i = [], 0, 0
    for i in range(1, len(ids) + 1):
        pos = _aligned_pos(text, tok.decode(ids[:i]))
        if pos is not None and pos > last_pos:
            pieces.append((text[last_pos:pos], i - last_i))
            last_pos, last_i = pos, i
    if last_pos < len(text):
        pieces.append((text[last_pos:], len(ids) - last_i))
    return pieces


def morph_scores(tok, items: list[tuple[str, list[int]]]) -> dict:
    tp = fp = fn = 0
    texts = [" " + w for w, _ in items]
    for (word, gold), text, ids in zip(items, texts, tok.encode_batch(texts)):
        pred = {b - 1 for b in token_char_boundaries(tok, text, ids) if 1 < b < len(text)}
        gold = set(gold)
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p, r = safe_div(tp, tp + fp, 0.0), safe_div(tp, tp + fn, 0.0)
    return {"morph_p": p, "morph_r": r, "morph_f1": safe_div(2 * p * r, p + r, 0.0)}
