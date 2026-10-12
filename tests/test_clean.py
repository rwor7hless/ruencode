import unicodedata

from ruencod.data.clean import accept, chunk_long, cyrillic_ratio, normalize
from ruencod.pretok import MARK


def test_normalize_nfc_and_keeps_yo():
    decomposed = unicodedata.normalize("NFD", "Ёлка й")
    assert normalize(decomposed) == "Ёлка й"


def test_normalize_strips_mark():
    assert normalize(f"кни{MARK}га") == "книга"


def test_html_only_for_html_sources():
    code = "if (a<b && c>d) { x = &y; }"
    assert normalize(code) == code
    assert normalize("<p>Привет&nbsp;мир</p><p>Два</p>", html_source=True) == "Привет\xa0мир\n\nДва"


def test_cyrillic_ratio():
    assert cyrillic_ratio("абв abc") == 0.5
    assert cyrillic_ratio("123") == 0.0


def test_accept():
    kw = dict(min_chars=5, max_chars=100, min_cyrillic_ratio=0.6)
    assert accept("Привет, мир!", russian=True, **kw)
    assert not accept("Hello world!", russian=True, **kw)
    assert accept("Hello world!", russian=False, **kw)
    assert not accept("абв", russian=True, **kw)


def test_chunk_long():
    text = "\n\n".join(["а" * 40] * 5)
    chunks = chunk_long(text, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks).replace("\n", "") == "а" * 200
    assert chunk_long("б" * 250, 100) == ["б" * 100, "б" * 100, "б" * 50]
