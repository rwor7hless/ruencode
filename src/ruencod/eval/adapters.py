import os
from dataclasses import dataclass, field
from pathlib import Path

from tokenizers import Tokenizer

from ruencod.config import cfg_path

THREADS = os.cpu_count() or 4


@dataclass
class HFTok:
    name: str
    tok: Tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @property
    def bos_id(self):
        return self.tok.token_to_id("<|bos|>")

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)


@dataclass
class TiktokenTok:
    name: str
    encoding_name: str
    enc: object = field(init=False)

    def __post_init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding(self.encoding_name)

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    @property
    def bos_id(self):
        return self.enc.eot_token

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def encode_batch(self, texts):
        return self.enc.encode_ordinary_batch(list(texts), num_threads=THREADS)

    def decode(self, ids):
        return self.enc.decode(ids)


@dataclass
class TransformersTok:
    name: str
    repo: str
    tk: object = field(init=False)

    def __post_init__(self):
        from transformers import AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained(self.repo, trust_remote_code=False)

    @property
    def vocab_size(self):
        return len(self.tk)

    @property
    def bos_id(self):
        return self.tk.bos_token_id

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False)

    def encode_batch(self, texts):
        return self.tk(list(texts), add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tk.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


@dataclass
class SentencePieceTok:
    name: str
    repo: str
    file: str
    sp: object = field(init=False)

    def __post_init__(self):
        import sentencepiece as spm
        from huggingface_hub import hf_hub_download
        self.sp = spm.SentencePieceProcessor(model_file=hf_hub_download(self.repo, self.file))

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    @property
    def bos_id(self):
        return self.sp.bos_id()

    def encode(self, text):
        return self.sp.encode(text)

    def encode_batch(self, texts):
        return self.sp.encode(list(texts), num_threads=THREADS)

    def decode(self, ids):
        return self.sp.decode(ids)


def load_ours(path: Path) -> HFTok:
    path = Path(path)
    return HFTok(path.name, Tokenizer.from_file(str(path / "tokenizer.json")))


def list_ours(cfg: dict) -> list[Path]:
    root = cfg_path(cfg, "artifacts") / "tokenizers"
    return sorted(p for p in root.glob("*") if (p / "tokenizer.json").exists()) if root.exists() else []


def load_baseline(spec: dict):
    kind = spec["kind"]
    if kind == "tiktoken":
        return TiktokenTok(spec["name"], spec["id"])
    if kind == "hf":
        return TransformersTok(spec["name"], spec["id"])
    if kind == "sentencepiece":
        return SentencePieceTok(spec["name"], spec["id"], spec["file"])
    raise ValueError(kind)


def resolve(cfg: dict, name: str):
    ours = cfg_path(cfg, "artifacts") / "tokenizers" / name
    if (ours / "tokenizer.json").exists():
        return load_ours(ours)
    for spec in cfg["eval"]["baselines"]:
        if spec["name"] == name:
            return load_baseline(spec)
    raise KeyError(name)
