import pytest

from ruencod.morph.project import Projector
from ruencod.morph.tikhonov import boundaries, load_lexicon, parse_line, split_lemmas

LINES = [
    "переписывать\tпере:PREF/пис:ROOT/ыва:SUFF/ть:SUFF",
    "переписываться\tпере:PREF/пис:ROOT/ыва:SUFF/ть:SUFF/ся:POSTFIX",
    "книга\tкниг:ROOT/а:END",
    "делать\tдел:ROOT/а:SUFF/ть:SUFF",
    "писать\tпис:ROOT/а:SUFF/ть:SUFF",
    "ёлка\tёл:ROOT/к:SUFF/а:END",
]


@pytest.fixture(scope="module")
def lexicon(tmp_path_factory):
    p = tmp_path_factory.mktemp("lex") / "t.tsv"
    p.write_text("\n".join(LINES + ["битая\tстрока:ROOT"]) + "\n", encoding="utf-8")
    return load_lexicon(p)


def test_parse_and_boundaries():
    word, morphs = parse_line(LINES[0])
    assert word == "переписывать"
    assert boundaries(morphs) == [4, 7, 10]
    assert parse_line("битая\tстрока:ROOT") is None
    assert parse_line("ёлка\tёл:ROOT/к:SUFF/а:END")[0] == "елка"


def test_lexicon_skips_bad_lines(lexicon):
    assert "битая" not in lexicon and "елка" in lexicon


def test_split_lemmas_deterministic():
    lemmas = [f"w{i}" for i in range(100)]
    a = split_lemmas(lemmas, 42, 0.1)
    assert a == split_lemmas(reversed(lemmas), 42, 0.1)
    assert len(a[1]) == 10 and not (a[0] & a[1])


@pytest.mark.parametrize("word,expected", [
    ("переписывать", [4, 7, 10]),
    ("книгами", [4]),
    ("Книгами", [4]),
    ("делаешь", [3, 4]),
    ("переписывающийся", [4, 7, 10, 14]),
    ("ёлками", [2, 3]),
    ("пишу", None),        # чередование в корне → без ограничений
    ("абырвалг", None),    # нет в словаре
    ("iPhone", None),
    ("IT-шник", None),
])
def test_projection(lexicon, word, expected):
    assert Projector(lexicon).segment(word) == expected


def test_excluded_lemma_gets_no_constraints(lexicon):
    assert Projector(lexicon, exclude={"книга"}).segment("книгами") is None
