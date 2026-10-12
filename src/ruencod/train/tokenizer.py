from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from ruencod.pretok import hf_pre_tokenizer, hf_train_pre_tokenizer

SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"] + [
    f"<|reserved_{i}|>" for i in range(251)
]


def train_tokenizer(texts: Iterable[str], algo: str, vocab_size: int, morph: bool = False,
                    min_frequency: int = 2) -> Tokenizer:
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    if algo == "bpe":
        tok = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                      special_tokens=SPECIAL_TOKENS, initial_alphabet=alphabet,
                                      show_progress=True)
    elif algo == "unigram":
        tok = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                          initial_alphabet=alphabet, max_piece_length=16,
                                          show_progress=True)
    else:
        raise ValueError(f"unknown algo: {algo}")
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = hf_train_pre_tokenizer() if morph else hf_pre_tokenizer()
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainer)
    tok.pre_tokenizer = hf_pre_tokenizer()  # инференс всегда со стандартной предтокенизацией
    return tok


def save_tokenizer(tok: Tokenizer, out_dir: Path) -> None:
    from transformers import PreTrainedTokenizerFast

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|bos|>", eos_token="<|eos|>",
                                   pad_token="<|pad|>")
    fast.save_pretrained(out_dir)
