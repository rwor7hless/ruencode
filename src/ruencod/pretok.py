import regex
from tokenizers import Regex, pre_tokenizers

_SUFFIX = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
# o200k-подобная регулярка: без inline-флагов (переносима в JS), цифры по одной
PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+" + _SUFFIX
    + r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*" + _SUFFIX
    + r"|\p{N}"
    + r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    + r"|\s*[\r\n]+"
    + r"|\s+(?!\S)"
    + r"|\s+"
)
MARK = ""  # служебный разделитель кусков при обучении, вычищается из корпуса

_RE = regex.compile(PATTERN)


def pretokenize(text: str) -> list[str]:
    return _RE.findall(text)


def hf_pre_tokenizer():
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), "isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])


def hf_train_pre_tokenizer():
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(MARK, "removed"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
