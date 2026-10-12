import math
from collections import Counter

from ruencod.eval.metrics import (aligned_pieces, measure, morph_scores, safe_div, single_token_rate,
                                  summarize, token_char_boundaries)


class CharTok:
    """Каждый символ — токен (id = ord)."""
    name, vocab_size, bos_id = "char", 0x110000, None

    def encode(self, text): return [ord(c) for c in text]
    def encode_batch(self, texts): return [self.encode(t) for t in texts]
    def decode(self, ids): return "".join(chr(i) for i in ids)


class PieceTok:
    """Фиксированное разбиение: text -> список кусков; id — индекс куска в таблице."""
    name, bos_id = "piece", None

    def __init__(self, table: dict[str, list[str]]):
        self.table, self.pieces = table, []
        self.vocab_size = 1000

    def encode(self, text):
        ids = []
        for p in self.table[text]:
            if p not in self.pieces:
                self.pieces.append(p)
            ids.append(self.pieces.index(p))
        return ids

    def encode_batch(self, texts): return [self.encode(t) for t in texts]
    def decode(self, ids): return "".join(self.pieces[i] for i in ids)


def test_measure_and_summarize():
    raw = measure(CharTok(), ["ab cd"])
    s = summarize(raw, vocab_size=10)
    assert s["fertility"] == 2.5 and s["chars_per_token"] == 1.0 and s["roundtrip"] == 1.0
    assert s["vocab_usage"] == 0.5  # 5 разных символов (a, b, пробел, c, d) из 10


def test_no_letters_gives_nan_not_crash():
    s = summarize(measure(CharTok(), ["123 456"]), vocab_size=10)
    assert math.isnan(s["fertility"])
    assert math.isnan(safe_div(1, 0))


def test_single_token_rate():
    tok = PieceTok({" кот": [" кот"], " книгами": [" книг", "ами"]})
    assert single_token_rate(tok, Counter({"кот": 3, "книгами": 1})) == 0.75


def test_morph_scores():
    good = PieceTok({" книгами": [" книг", "ами"]})
    bad = PieceTok({" книгами": [" кни", "гами"]})
    assert morph_scores(good, [("книгами", [4])])["morph_f1"] == 1.0
    assert morph_scores(bad, [("книгами", [4])])["morph_f1"] == 0.0


def test_boundaries_ignore_partial_chars_and_stripped_space():
    class Partial(PieceTok):
        def decode(self, ids):
            s = "".join(self.pieces[i] for i in ids)
            return s.replace("½", "�")
    tok = Partial({" ab": [" a", "½", "b"]})
    ids = tok.encode(" ab")
    assert token_char_boundaries(tok, " ab", ids) == {2}

    class Stripping(PieceTok):  # как sentencepiece: срезает ведущий пробел при decode
        def decode(self, ids):
            return "".join(self.pieces[i] for i in ids).lstrip(" ")
    tok2 = Stripping({" книгами": [" книг", "ами"]})
    assert token_char_boundaries(tok2, " книгами", tok2.encode(" книгами")) == {5}


def test_aligned_pieces():
    assert aligned_pieces(PieceTok({"мир дом": ["мир", " дом"]}), "мир дом") == [("мир", 1), (" дом", 1)]

    class Partial(PieceTok):  # байтовый токен посреди символа: «½» декодируется в �
        def decode(self, ids):
            return "".join(self.pieces[i] for i in ids).replace("½", "�")
    # «½» не выровнен по символу → склеивается со следующим куском, который считается за 2 токена
    assert aligned_pieces(Partial({"xyz": ["x", "½", "yz"]}), "xyz") == [("x", 1), ("yz", 2)]
