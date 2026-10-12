import pytest

from ruencod.pretok import MARK, pretokenize
from ruencod.train.run import tokenizer_name
from ruencod.train.tokenizer import SPECIAL_TOKENS, save_tokenizer, train_tokenizer
from tests.conftest import make_corpus

V = 700  # > 256 спецтокенов + 256 байт

TRICKY = [
    "Привет, мир!\r\n\r\nВторая\tстрока   с   пробелами.",
    "Ёлка, ёжик и йод. 🙂🚀 日本語",
    "def f(x):\n    return x ** 2\n",
    "Строка со спецтокеном <|bos|> внутри.",
    "",
    " ",
]


@pytest.fixture(scope="module")
def bpe():
    return train_tokenizer(make_corpus(), "bpe", V)


@pytest.fixture(scope="module")
def tokenizers_by_algo():
    return {a: train_tokenizer(make_corpus(100), a, V) for a in ("bpe", "unigram")}


def test_specials():
    assert len(SPECIAL_TOKENS) == 256 and len(set(SPECIAL_TOKENS)) == 256


@pytest.mark.parametrize("algo", ["bpe", "unigram"])
@pytest.mark.parametrize("text", TRICKY)
def test_roundtrip(tokenizers_by_algo, algo, text):
    tok = tokenizers_by_algo[algo]
    assert tok.decode(tok.encode(text, add_special_tokens=False).ids, skip_special_tokens=False) == text


def test_digits_single(bpe):
    ids = bpe.encode("2024", add_special_tokens=False).ids
    assert [bpe.decode([i]) for i in ids] == ["2", "0", "2", "4"]


def test_special_tokens_atomic(bpe):
    for i, s in enumerate(SPECIAL_TOKENS):
        assert bpe.token_to_id(s) == i
    assert bpe.encode("<|im_start|>", add_special_tokens=False).ids == [bpe.token_to_id("<|im_start|>")]
    assert bpe.get_vocab_size() <= V


def test_mark_path_equals_plain_path():
    docs = make_corpus()
    plain = train_tokenizer(docs, "bpe", V)
    marked = train_tokenizer([MARK.join(pretokenize(t)) for t in docs], "bpe", V, morph=True)
    assert plain.get_vocab() == marked.get_vocab()


def test_morph_merges_do_not_cross_boundaries():
    docs = ["Читаю книгами каждый день."] * 200
    seg = [t.replace(" книгами", f" книг{MARK}ами") for t in docs]
    morph = train_tokenizer(seg, "bpe", 600, morph=True)
    plain = train_tokenizer(docs, "bpe", 600)
    vocab_m = {morph.decode([i]) for i in range(morph.get_vocab_size())}
    vocab_p = {plain.decode([i]) for i in range(plain.get_vocab_size())}
    assert " книгами" in vocab_p
    assert not any("гами" in t or "книга" in t for t in vocab_m)
    assert morph.decode(morph.encode(docs[0], add_special_tokens=False).ids) == docs[0]


def test_bpe_deterministic():
    a = train_tokenizer(make_corpus(), "bpe", V)
    b = train_tokenizer(make_corpus(), "bpe", V)
    assert a.to_str() == b.to_str()


def test_save_loadable_by_transformers(bpe, tmp_path):
    from transformers import AutoTokenizer
    save_tokenizer(bpe, tmp_path / "t")
    hf = AutoTokenizer.from_pretrained(tmp_path / "t")
    assert hf.bos_token == "<|bos|>" and hf.pad_token == "<|pad|>"
    assert hf.encode("Привет", add_special_tokens=False) == bpe.encode("Привет", add_special_tokens=False).ids


def test_name():
    assert tokenizer_name("morph-bpe", 48000) == "morph-bpe-48k"
