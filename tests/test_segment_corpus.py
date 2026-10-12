from ruencod.morph.segment_corpus import build_cache, count_words, segment_text, word_core
from ruencod.pretok import MARK


def test_word_core():
    assert word_core(" Книгами") == (1, "Книгами")
    assert word_core("книгами") == (0, "книгами")
    assert word_core("«книга") == (1, "книга")


def test_segment_text_cuts_and_is_lossless():
    text = "Читаю Книгами, iPhone-ом и IT-шник 2024."
    out = segment_text(text, {"книгами": [4]})
    assert out.replace(MARK, "") == text
    assert f" Книг{MARK}ами" in out


def test_count_words_parallel():
    texts = ["книга книги книга", "Книга iPhone"] * 3
    c = count_words(texts, workers=2)
    assert c["книга"] == 9 and c["книги"] == 3 and "iphone" not in c


def test_build_cache(tmp_path):
    lex = tmp_path / "t.tsv"
    lex.write_text("книга\tкниг:ROOT/а:END\nмир\tмир:ROOT\n", encoding="utf-8")
    cache, covered = build_cache(["книгами", "мир", "абырвалг"], lex, exclude=set(), workers=2)
    assert cache == {"книгами": [4]}
    assert covered == {"книгами", "мир"}
