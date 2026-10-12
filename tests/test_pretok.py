import pytest
from tokenizers import pre_tokenizers, Regex

from ruencod.pretok import MARK, PATTERN, pretokenize, hf_pre_tokenizer, hf_train_pre_tokenizer

CASES = [
    "Привет, мир! Мы переписываем книгами 2024 год.",
    "don't I'LL we've «Ёлка» — тест",
    "a\r\n\r\nb\t\tc   d",
    "def f(x):\n    return x**2  # 🙂",
    "IT-специалист iPhone-ом 3.14159",
]


@pytest.mark.parametrize("text", CASES)
def test_python_regex_matches_hf_split(text):
    hf = [p for p, _ in pre_tokenizers.Split(Regex(PATTERN), "isolated").pre_tokenize_str(text)]
    assert pretokenize(text) == hf
    assert "".join(pretokenize(text)) == text


def test_digits_are_single():
    assert pretokenize("2024") == ["2", "0", "2", "4"]


def test_no_inline_flags_in_pattern():
    # transformers.js не поддерживает (?i:...) — паттерн должен быть переносимым
    assert "(?i" not in PATTERN


def test_train_pre_tokenizer_splits_only_on_mark():
    pieces = [p for p, _ in hf_train_pre_tokenizer().pre_tokenize_str(f" книг{MARK}ами")]
    assert len(pieces) == 2


def test_inference_pre_tokenizer_runs():
    assert hf_pre_tokenizer().pre_tokenize_str("Привет мир")
