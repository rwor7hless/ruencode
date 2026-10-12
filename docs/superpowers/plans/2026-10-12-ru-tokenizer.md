# ruEncod — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Обучить и сравнить семейство русскоязычных токенизаторов (BPE / Unigram / morph-BPE × 5 размеров словаря), выбрать финалиста для модели ~1B, проверить на мини-LM и подготовить материалы для Telegram-поста.

**Architecture:** Python-пакет `ruencod` из независимых CLI-этапов: данные → морфология → обучение → оценка → мини-LM → визуализация. Этапы общаются только через файлы на диске (`data/`, `artifacts/`, `results/`). Все токенизаторы — HF `tokenizers` с единой предтокенизацией; morph-BPE отличается только тем, что при обучении границы морфем помечены разделителем `U+E000`.

**Tech Stack:** Python 3.12 (uv), `tokenizers`, `transformers`, `datasets`, `pymorphy3`, `regex`, `tiktoken`, `sentencepiece`, `anthropic`, `torch` (CPU), `matplotlib`, `pandas`, `zstandard`, `pytest`.

**Spec:** `docs/specs/2026-10-12-ru-tokenizer-design.md`

## Global Constraints

- Python `>=3.12,<3.13`, окружение через `uv`; torch только CPU-сборка (индекс `https://download.pytorch.org/whl/cpu`).
- Нормализация текста — только NFC. Буква «ё» не трогается в данных и в токенизаторе (`ё→е` допустимо только как ключ сопоставления со словарём).
- Обратимость: `decode(encode(x)) == x` для любых строк у наших токенизаторов.
- Цифры кодируются поштучно (в предтокенизаторе `\p{N}` по одному).
- Спецтокены: ровно 256 — `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`, `<|reserved_0|>`…`<|reserved_250|>`; входят в размер словаря.
- Размеры словаря: 16000, 32000, 48000, 64000, 96000. Алгоритмы: `bpe`, `unigram`, `morph-bpe`. Имя токенизатора: `{algo}-{vocab//1000}k`.
- Корпус ~5 ГБ train; heldout 50 МБ на домен, не используется ни в каком обучении.
- Тестовые 10% лемм Тихонова не получают морфологических ограничений при обучении.
- Claude считается только через `client.messages.count_tokens` (модель `claude-opus-5`), никогда не `tiktoken`.
- Цены фиксируются в `configs/experiments.yaml` с датой и источником; GPT-цена сверяется с `https://openai.com/api/pricing` перед построением графиков.
- Ресурсы: 20 ядер, ~10 ГБ свободной RAM, без GPU.
- Коммиты: в конце сообщения строка `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. Сырой текст содержит символ-разделитель `U+E000` → он должен удаляться при очистке, иначе morph-корпус испортится (тест в Task 2).
2. Источник иссяк раньше байтового бюджета (Gazeta мала) → сборка не зависает, нехватка записывается в `stats.json` (тест в Task 3).
3. Тексты с `\r\n`, табами, длинными пробелами, эмодзи и строкой `<|bos|>` внутри → токенизатор обратим (тест в Task 6).
4. Домен без букв (код, числа) → fertility = `nan`, без падения (тест в Task 7).
5. Прерванный подсчёт Claude → кэш на диске, повторный запуск не делает повторных запросов (тест в Task 8).

---

## File Structure

```
ruEncod/
  pyproject.toml
  configs/experiments.yaml        # полный конфиг
  configs/smoke.yaml              # base: experiments.yaml + маленькие бюджеты
  src/ruencod/
    __init__.py
    config.py                     # load_config (с base-наследованием), cfg_path
    jsonl.py                      # ZstJsonlWriter, iter_jsonl_zst
    corpus.py                     # domain_file, iter_domain, iter_mix, domain_shares
    pretok.py                     # PATTERN, MARK, pretokenize, hf_pre_tokenizer, hf_train_pre_tokenizer
    data/__init__.py
    data/clean.py                 # normalize, chunk_long, cyrillic_ratio, accept
    data/sources.py               # iter_source (hf / pikabu), pikabu_docs
    data/build.py                 # build(): download+clean+dedup+split, CLI
    morph/__init__.py
    morph/tikhonov.py             # Morpheme, parse_line, load_lexicon, boundaries, split_lemmas, download
    morph/project.py              # Projector.segment(word) -> list[int] | None
    morph/segment_corpus.py       # count_words, build_cache, segment_text, CLI
    train/__init__.py
    train/tokenizer.py            # SPECIAL_TOKENS, train_tokenizer, save_tokenizer
    train/run.py                  # tokenizer_name, train_one, CLI (grid)
    eval/__init__.py
    eval/adapters.py              # HFTok, TiktokenTok, TransformersTok, SentencePieceTok, load_*
    eval/metrics.py               # measure, single_token_rate, token_char_boundaries, aligned_pieces, morph_scores
    eval/claude_count.py          # ClaudeCounter
    eval/run.py                   # run_eval -> results/metrics.csv, CLI
    lm/__init__.py
    lm/model.py                   # LMConfig, TinyLM
    lm/train_lm.py                # encode_corpus, train_lm
    lm/eval_bpb.py                # nll_bits_and_bytes
    lm/run.py                     # CLI -> results/lm_bpb.csv
    viz/__init__.py
    viz/charts.py                 # 5 PNG-графиков
    viz/splits.py                 # картинки разбиения
    viz/summary.py                # results/summary.json, web/examples.json
  web/index.html                  # интерактивная страница (Task 12)
  post/draft.md                   # черновик поста (Task 15)
  tests/conftest.py
  tests/test_*.py
```

---

### Task 1: Каркас проекта, конфиг, jsonl, предтокенизатор

**Files:**
- Create: `pyproject.toml`, `configs/experiments.yaml`, `configs/smoke.yaml`, `src/ruencod/__init__.py`, `src/ruencod/config.py`, `src/ruencod/jsonl.py`, `src/ruencod/corpus.py`, `src/ruencod/pretok.py`, пустые `__init__.py` во всех подпакетах
- Test: `tests/conftest.py`, `tests/test_pretok.py`, `tests/test_io.py`

**Interfaces:**
- Produces:
  - `config.load_config(path=DEFAULT_CONFIG) -> dict`, `config.cfg_path(cfg, key) -> Path`, `config.ROOT: Path`
  - `jsonl.ZstJsonlWriter(path)` (context manager, `.write(rec: dict)`), `jsonl.iter_jsonl_zst(path) -> Iterator[dict]`
  - `corpus.domain_file(base, split, domain) -> Path`, `corpus.iter_domain(base, split, domain, max_bytes=None) -> Iterator[str]`, `corpus.iter_mix(base, split, shares, max_bytes=None) -> Iterator[str]`, `corpus.domain_shares(cfg) -> dict[str, float]`, `corpus.russian_domains(cfg) -> list[str]`
  - `pretok.PATTERN: str`, `pretok.MARK = ""`, `pretok.pretokenize(text) -> list[str]`, `pretok.hf_pre_tokenizer()`, `pretok.hf_train_pre_tokenizer()`

- [ ] **Step 1: Создать `pyproject.toml`**

```toml
[project]
name = "ruencod"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "tokenizers>=0.23",
  "transformers>=4.45",
  "datasets>=3.0",
  "huggingface_hub",
  "pymorphy3",
  "pymorphy3-dicts-ru",
  "regex",
  "tiktoken",
  "sentencepiece",
  "protobuf",
  "anthropic",
  "torch",
  "numpy",
  "matplotlib",
  "pillow",
  "pandas",
  "pyyaml",
  "zstandard",
  "tqdm",
]

[dependency-groups]
dev = ["pytest"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ruencod"]

[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Также создать пустой `tests/__init__.py` — тесты импортируют друг у друга фикстуры (`from tests.conftest import make_corpus`).

- [ ] **Step 2: Установить окружение**

Run: `cd ruEncod && uv sync -p 3.12`
Expected: создан `.venv`, без ошибок. Проверка: `uv run python -c "import tokenizers, torch, pymorphy3; print(torch.__version__)"` печатает версию с `+cpu`.

- [ ] **Step 3: Создать `configs/experiments.yaml`**

```yaml
seed: 42
paths:
  data: data
  artifacts: artifacts
  results: results

corpus:
  total_bytes: 5000000000
  heldout_bytes_per_domain: 50000000
  min_chars: 200
  max_chars: 100000
  min_cyrillic_ratio: 0.6
  domains:
    web:
      share: 0.30
      russian: true
      sources:
        - {kind: hf, path: HuggingFaceFW/fineweb-2, name: rus_Cyrl, split: train, text: text}
    wiki:
      share: 0.15
      russian: true
      sources:
        - {kind: hf, path: wikimedia/wikipedia, name: 20231101.ru, split: train, text: text}
    social:
      share: 0.15
      russian: true
      sources:
        - {kind: pikabu}
    it:
      share: 0.10
      russian: true
      sources:
        - {kind: hf, path: volosati/habr, split: train, text: text_markdown, filter: {language: ru}}
    books:
      share: 0.10
      russian: true
      sources:
        - {kind: hf, path: wikimedia/wikisource, name: 20231201.ru, split: train, text: text}
    news:
      share: 0.10
      russian: true
      sources:
        - {kind: hf, path: IlyaGusev/gazeta, split: train, text: text, share: 0.5}
        - kind: hf
          path: HuggingFaceFW/fineweb-2
          name: rus_Cyrl
          split: train
          text: text
          share: 0.5
          url_domains: [ria.ru, tass.ru, lenta.ru, rbc.ru, interfax.ru, kommersant.ru, iz.ru, rg.ru, gazeta.ru, fontanka.ru, meduza.io, vedomosti.ru]
    niche:
      share: 0.05
      russian: true
      sources:
        - {kind: hf, path: irlspbru/RusLawOD, split: train, text: textIPS, shuffle_buffer: 500}
    en_code:
      share: 0.05
      russian: false
      sources:
        - {kind: hf, path: HuggingFaceFW/fineweb-edu, name: sample-10BT, split: train, text: text, share: 0.5}
        - {kind: hf, path: bigcode/the-stack-smol-xl, split: train, text: content, share: 0.5}

morph:
  tikhonov_urls:
    - https://raw.githubusercontent.com/AlexeySorokin/NeuralMorphemeSegmentation/master/data/train_Tikhonov_reformat.txt
    - https://raw.githubusercontent.com/AlexeySorokin/NeuralMorphemeSegmentation/master/data/test_Tikhonov_reformat.txt
  test_frac: 0.1
  min_word_freq: 5
  workers: 8

train:
  vocab_sizes: [16000, 32000, 48000, 64000, 96000]
  algos: [bpe, unigram, morph-bpe]
  bpe_max_bytes: null          # null = весь train; уменьшить при нехватке RAM (Task 14)
  unigram_max_bytes: 1500000000
  bpe_min_frequency: 2

eval:
  max_bytes_per_domain: 50000000
  claude:
    models: [claude-opus-5]
    docs_per_domain: 200
  baselines:
    - {name: o200k, label: "GPT-4o/GPT-5 (o200k)", kind: tiktoken, id: o200k_base}
    - {name: cl100k, label: "GPT-4 (cl100k)", kind: tiktoken, id: cl100k_base}
    - {name: llama-3, label: "Llama 3", kind: hf, id: NousResearch/Meta-Llama-3-8B}
    - {name: qwen-3, label: "Qwen 3", kind: hf, id: Qwen/Qwen3-8B}
    - {name: mistral-nemo, label: "Mistral Nemo", kind: hf, id: unsloth/Mistral-Nemo-Base-2407}
    - {name: gemma-3, label: "Gemma 3", kind: hf, id: unsloth/gemma-3-1b-it}
    - {name: rugpt-3, label: "ruGPT-3", kind: hf, id: ai-forever/rugpt3large_based_on_gpt2}
    - {name: vikhr, label: "Vikhr 7B", kind: hf, id: Vikhrmodels/Vikhr-7B-instruct_0.4}
    - {name: gigachat, label: "GigaChat (open)", kind: hf, id: ai-sage/GigaChat-20B-A3B-instruct}
    - {name: yandexgpt-5-lite, label: "YandexGPT 5 Lite", kind: sentencepiece, id: yandex/YandexGPT-5-Lite-8B-pretrain, file: tokenizer.model}

prices:
  verified_on: "2026-10-12"
  gpt:
    label: "GPT-4o"
    tokenizer: o200k
    usd_per_mtok: 2.50
    source: https://openai.com/api/pricing
  claude:
    label: "Claude Opus 5"
    tokenizer: claude-opus-5
    usd_per_mtok: 5.00
    source: https://docs.claude.com/en/docs/about-claude/pricing

lm:
  context: 512
  d_model: 384
  n_layers: 6
  n_heads: 6
  batch_size: 16
  lr: 0.001
  warmup_frac: 0.02
  weight_decay: 0.1
  train_bytes: 30000000        # пересчитывается по smoke-run (Task 14, шаг LM)
  eval_bytes_per_domain: 2000000
  threads: 20
  candidates: []               # заполняется после выбора V, напр. [bpe-48k, unigram-48k, morph-bpe-48k, o200k]

final:
  tokenizer: null              # заполняется после выбора финалиста

examples:
  - "достопримечательностями"
  - "переподготовка"
  - "нейросеть"
  - "1234567"
  - "Я написал скрипт на Python, который парсит JSON и шлёт в Telegram."
  - "Нейросеть не читает буквы — она видит токены."
```

- [ ] **Step 4: Создать `configs/smoke.yaml`**

```yaml
base: experiments.yaml
paths:
  data: data_smoke
  artifacts: artifacts_smoke
  results: results_smoke
corpus:
  total_bytes: 30000000
  heldout_bytes_per_domain: 1000000
morph:
  min_word_freq: 2
  workers: 4
train:
  vocab_sizes: [2000, 4000]
  unigram_max_bytes: 10000000
eval:
  max_bytes_per_domain: 1000000
  claude:
    docs_per_domain: 0
  baselines:
    - {name: o200k, label: "GPT-4o/GPT-5 (o200k)", kind: tiktoken, id: o200k_base}
lm:
  train_bytes: 1000000
  eval_bytes_per_domain: 100000
  candidates: [bpe-4k, morph-bpe-4k]
final:
  tokenizer: bpe-4k
```

- [ ] **Step 5: Написать падающие тесты `tests/conftest.py`, `tests/test_pretok.py`, `tests/test_io.py`**

`tests/conftest.py`:
```python
import random

import pytest

RU_SENTENCES = [
    "Мы переписываем книгами старые тексты.",
    "Нейросеть читает токены, а не буквы.",
    "В 2024 году достопримечательностями Москвы любовались 25 млн туристов.",
    "Ёлка стоит в углу, ёжик спит под ней.",
    "Переподготовка специалистов займёт 3 месяца.",
    "Я написал скрипт на Python, который шлёт JSON в Telegram.",
    "Делаешь хорошо — получаешь хорошо.",
    "Книга лежит на столе, а книги стоят на полке.",
    "Переписывающийся с другом школьник учится писать грамотно.",
    "def f(x):\n    return x ** 2\n",
]


def make_corpus(n_docs: int = 300, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        sents = rng.sample(RU_SENTENCES, k=5)
        docs.append(" ".join(sents))
    return docs


@pytest.fixture
def corpus() -> list[str]:
    return make_corpus()
```

`tests/test_pretok.py`:
```python
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
    assert [p for p in pretokenize("2024") ] == ["2", "0", "2", "4"]


def test_no_inline_flags_in_pattern():
    # transformers.js не поддерживает (?i:...) — паттерн должен быть переносимым
    assert "(?i" not in PATTERN


def test_train_pre_tokenizer_splits_only_on_mark():
    pieces = [p for p, _ in hf_train_pre_tokenizer().pre_tokenize_str(f" книг{MARK}ами")]
    assert len(pieces) == 2


def test_inference_pre_tokenizer_runs():
    assert hf_pre_tokenizer().pre_tokenize_str("Привет мир")
```

`tests/test_io.py`:
```python
from ruencod.config import load_config, cfg_path
from ruencod.corpus import domain_file, iter_domain, iter_mix
from ruencod.jsonl import ZstJsonlWriter, iter_jsonl_zst


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "a" / "x.jsonl.zst"
    recs = [{"text": "строка\nс переводом", "domain": "web"}, {"text": "два"}]
    with ZstJsonlWriter(p) as w:
        for r in recs:
            w.write(r)
    assert list(iter_jsonl_zst(p)) == recs


def test_iter_domain_respects_max_bytes(tmp_path):
    with ZstJsonlWriter(domain_file(tmp_path, "train", "web")) as w:
        for _ in range(10):
            w.write({"text": "абв"})  # 6 байт
    assert len(list(iter_domain(tmp_path, "train", "web", max_bytes=12))) == 2
    assert list(iter_domain(tmp_path, "train", "missing")) == []


def test_iter_mix_by_shares(tmp_path):
    for d in ("a", "b"):
        with ZstJsonlWriter(domain_file(tmp_path, "train", d)) as w:
            for _ in range(10):
                w.write({"text": d * 10})
    out = list(iter_mix(tmp_path, "train", {"a": 0.5, "b": 0.5}, max_bytes=40))
    assert out.count("a" * 10) == 2 and out.count("b" * 10) == 2


def test_config_base_merge():
    cfg = load_config(cfg_path({"paths": {"x": "configs/smoke.yaml"}}, "x"))
    assert cfg["train"]["vocab_sizes"] == [2000, 4000]
    assert cfg["corpus"]["min_chars"] == 200  # унаследовано
    assert cfg["corpus"]["domains"]["web"]["share"] == 0.30
```

- [ ] **Step 6: Запустить тесты — убедиться, что падают**

Run: `uv run pytest tests/test_pretok.py tests/test_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ruencod.pretok'`.

- [ ] **Step 7: Реализовать модули**

`src/ruencod/config.py`:
```python
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "experiments.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(load_config(path.parent / base), cfg)
    return cfg


def cfg_path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p
```

`src/ruencod/jsonl.py`:
```python
import io
import json
from pathlib import Path
from typing import Iterator

import zstandard as zstd


class ZstJsonlWriter:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "wb")
        self._z = zstd.ZstdCompressor(level=3).stream_writer(self._f)

    def write(self, rec: dict) -> None:
        self._z.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))

    def close(self) -> None:
        self._z.flush(zstd.FLUSH_FRAME)
        self._z.close()
        if not self._f.closed:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def iter_jsonl_zst(path: Path) -> Iterator[dict]:
    with open(path, "rb") as f:
        reader = zstd.ZstdDecompressor().stream_reader(f)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)
```

`src/ruencod/corpus.py`:
```python
from pathlib import Path
from typing import Iterator

from ruencod.jsonl import iter_jsonl_zst


def domain_file(base: Path, split: str, domain: str) -> Path:
    return Path(base) / split / f"{domain}.jsonl.zst"


def iter_domain(base: Path, split: str, domain: str, max_bytes: int | None = None) -> Iterator[str]:
    path = domain_file(base, split, domain)
    if not path.exists():
        return
    used = 0
    for rec in iter_jsonl_zst(path):
        if max_bytes is not None and used >= max_bytes:
            return
        text = rec["text"]
        used += len(text.encode("utf-8"))
        yield text


def iter_mix(base: Path, split: str, shares: dict[str, float], max_bytes: int | None = None) -> Iterator[str]:
    for domain, share in shares.items():
        budget = None if max_bytes is None else int(max_bytes * share)
        yield from iter_domain(base, split, domain, budget)


def domain_shares(cfg: dict) -> dict[str, float]:
    return {d: v["share"] for d, v in cfg["corpus"]["domains"].items()}


def russian_domains(cfg: dict) -> list[str]:
    return [d for d, v in cfg["corpus"]["domains"].items() if v.get("russian", True)]
```

`src/ruencod/pretok.py`:
```python
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
```

Также создать пустые `src/ruencod/__init__.py`, `src/ruencod/{data,morph,train,eval,lm,viz}/__init__.py`.

- [ ] **Step 8: Запустить тесты**

Run: `uv run pytest tests/test_pretok.py tests/test_io.py -v`
Expected: PASS (все).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock configs src tests
git commit -m "feat: project scaffold, config, jsonl io, shared pre-tokenizer"
```

---

### Task 2: Очистка текста

**Files:**
- Create: `src/ruencod/data/clean.py`
- Test: `tests/test_clean.py`

**Interfaces:**
- Consumes: `pretok.MARK`
- Produces: `clean.normalize(text, html_source=False) -> str`, `clean.chunk_long(text, max_chars) -> list[str]`, `clean.cyrillic_ratio(text) -> float`, `clean.accept(text, *, min_chars, max_chars, russian, min_cyrillic_ratio) -> bool`

- [ ] **Step 1: Написать падающий тест**

```python
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
```

- [ ] **Step 2: Run** `uv run pytest tests/test_clean.py -v` → FAIL (`No module named 'ruencod.data.clean'`).

- [ ] **Step 3: Реализация `src/ruencod/data/clean.py`**

```python
import html
import re
import unicodedata

from ruencod.pretok import MARK

_BLOCK_TAG = re.compile(r"</?(?:p|br|div|li|h[1-6]|blockquote|tr)\b[^<>]*>", re.I)
_TAG = re.compile(r"<[a-zA-Z/!][^<>]{0,500}>")
_MULTI_NL = re.compile(r"\n{3,}")


def _strip_html(text: str) -> str:
    text = _BLOCK_TAG.sub("\n\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    return _MULTI_NL.sub("\n\n", text)


def normalize(text: str, html_source: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    if html_source:
        text = _strip_html(text)
    return text.replace(MARK, "").strip()


def cyrillic_ratio(text: str) -> float:
    letters = cyr = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if "Ѐ" <= ch <= "ӿ":
                cyr += 1
    return cyr / letters if letters else 0.0


def accept(text: str, *, min_chars: int, max_chars: int, russian: bool, min_cyrillic_ratio: float) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    return not russian or cyrillic_ratio(text) >= min_cyrillic_ratio


def chunk_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for para in text.split("\n\n"):
        while len(para) > max_chars:
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_len = [], 0
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        add = len(para) + (2 if cur else 0)
        if cur and cur_len + add > max_chars:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [para], len(para)
        else:
            cur.append(para)
            cur_len += add
    if cur:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if c.strip()]
```

- [ ] **Step 4: Run** `uv run pytest tests/test_clean.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ruencod/data/clean.py tests/test_clean.py
git commit -m "feat: text cleaning (NFC, optional HTML strip, chunking, cyrillic filter)"
```

---

### Task 3: Источники и сборка корпуса

**Files:**
- Create: `src/ruencod/data/sources.py`, `src/ruencod/data/build.py`
- Test: `tests/test_build.py`

**Interfaces:**
- Consumes: `clean.*`, `jsonl.ZstJsonlWriter`, `corpus.domain_file`, `config.cfg_path`
- Produces: `sources.iter_source(src: dict, seed: int) -> Iterator[str]`, `sources.pikabu_docs(rec) -> list[str]`, `sources.url_matches(url, domains) -> bool`, `build.build(cfg, domains=None, iter_source_fn=iter_source) -> dict` (пишет `{data}/corpus/{train,heldout}/{domain}.jsonl.zst` и `{data}/corpus/stats.json`). CLI: `python -m ruencod.data.build [--config P] [--domains d1 d2]`.

- [ ] **Step 1: Написать падающий тест**

```python
import json

from ruencod.corpus import iter_domain
from ruencod.data.build import build
from ruencod.data.sources import pikabu_docs, url_matches
from ruencod.pretok import MARK


def _cfg(tmp_path, total=4000, held=1000):
    return {
        "seed": 1,
        "paths": {"data": str(tmp_path)},
        "corpus": {
            "total_bytes": total, "heldout_bytes_per_domain": held,
            "min_chars": 10, "max_chars": 1000, "min_cyrillic_ratio": 0.6,
            "domains": {
                "web": {"share": 0.5, "russian": True, "sources": [{"kind": "fake", "n": 1000}]},
                "news": {"share": 0.5, "russian": True, "sources": [{"kind": "fake", "n": 3}]},
            },
        },
    }


def fake_source(src, seed):
    for i in range(src["n"]):
        yield f"Документ номер {i} про{MARK} русский язык и токены."
    yield "Документ номер 0 про русский язык и токены."  # дубликат после очистки
    yield "English only text that must be rejected."


def test_build_fills_budgets_and_dedups(tmp_path):
    stats = build(_cfg(tmp_path), iter_source_fn=fake_source)
    web = stats["web"]
    assert web["train_bytes"] >= 2000 and web["heldout_bytes"] >= 1000
    assert not web["shortfall"]
    texts = list(iter_domain(tmp_path / "corpus", "train", "web")) + list(iter_domain(tmp_path / "corpus", "heldout", "web"))
    assert len(texts) == len(set(texts))
    assert all(MARK not in t for t in texts)
    assert json.loads((tmp_path / "corpus" / "stats.json").read_text())["web"]["docs"] == web["docs"]


def test_build_records_shortfall_and_terminates(tmp_path):
    stats = build(_cfg(tmp_path), iter_source_fn=fake_source)
    assert stats["news"]["shortfall"], "источник на 3 документа не может заполнить бюджет"
    assert stats["news"]["duplicates"] >= 1
    assert stats["news"]["rejected"] >= 1


def test_pikabu_docs():
    rec = {"title": "Заголовок", "text_markdown": "Текст поста",
           "comments": {"text_markdown": ["первый", "", "второй"]}}
    assert pikabu_docs(rec) == ["Заголовок\n\nТекст поста", "первый\nвторой"]
    assert pikabu_docs({"title": "", "text_markdown": "", "comments": {}}) == []


def test_url_matches():
    assert url_matches("https://www.ria.ru/2020/a.html", ["ria.ru"])
    assert not url_matches("https://maria.ru/x", ["ria.ru"])
    assert not url_matches(None, ["ria.ru"])
```

- [ ] **Step 2: Run** `uv run pytest tests/test_build.py -v` → FAIL (import error).

- [ ] **Step 3: Реализация `src/ruencod/data/sources.py`**

```python
import io
import json
import random
from typing import Iterator
from urllib.parse import urlparse

import zstandard as zstd


def url_matches(url: str | None, domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "") if url else ""
    return any(host == d or host.endswith("." + d) for d in domains)


def pikabu_docs(rec: dict) -> list[str]:
    docs = []
    post = "\n\n".join(x for x in (rec.get("title") or "", rec.get("text_markdown") or "") if x.strip())
    if post:
        docs.append(post)
    comments = [c for c in ((rec.get("comments") or {}).get("text_markdown") or []) if c and c.strip()]
    if comments:
        docs.append("\n".join(comments))
    return docs


def _iter_hf(src: dict, seed: int) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(src["path"], src.get("name"), split=src.get("split", "train"), streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=src.get("shuffle_buffer", 10_000))
    flt = src.get("filter") or {}
    url_domains = src.get("url_domains")
    for row in ds:
        if any(row.get(k) != v for k, v in flt.items()):
            continue
        if url_domains and not url_matches(row.get("url"), url_domains):
            continue
        text = row.get(src["text"])
        if text:
            yield text


def _iter_pikabu(seed: int) -> Iterator[str]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob("datasets/IlyaGusev/pikabu/*.jsonl.zst"))
    random.Random(seed).shuffle(files)
    for path in files:
        with fs.open(path, "rb") as f:
            reader = zstd.ZstdDecompressor().stream_reader(f)
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if line.strip():
                    yield from pikabu_docs(json.loads(line))


def iter_source(src: dict, seed: int) -> Iterator[str]:
    kind = src.get("kind", "hf")
    if kind == "hf":
        yield from _iter_hf(src, seed)
    elif kind == "pikabu":
        yield from _iter_pikabu(seed)
    else:
        raise ValueError(f"unknown source kind: {kind}")
```

- [ ] **Step 4: Реализация `src/ruencod/data/build.py`**

```python
import argparse
import hashlib
import json
import random

from tqdm import tqdm

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import domain_file
from ruencod.data.clean import accept, chunk_long, normalize
from ruencod.data.sources import iter_source
from ruencod.jsonl import ZstJsonlWriter


def build(cfg: dict, domains: list[str] | None = None, iter_source_fn=iter_source) -> dict:
    corpus = cfg["corpus"]
    out = cfg_path(cfg, "data") / "corpus"
    seen: set[bytes] = set()
    stats: dict = {}
    for domain, dcfg in corpus["domains"].items():
        if domains and domain not in domains:
            continue
        train_target = int(corpus["total_bytes"] * dcfg["share"])
        held_target = int(corpus["heldout_bytes_per_domain"])
        srcs = dcfg["sources"]
        ds = {"train_bytes": 0, "heldout_bytes": 0, "docs": 0, "rejected": 0, "duplicates": 0, "shortfall": []}
        with ZstJsonlWriter(domain_file(out, "train", domain)) as wt, \
             ZstJsonlWriter(domain_file(out, "heldout", domain)) as wh:
            for i, src in enumerate(srcs):
                share = src.get("share", 1 / len(srcs))
                st, sh = int(train_target * share), int(held_target * share)
                p_held = sh / (st + sh) if st + sh else 0.0
                rng = random.Random(f"{cfg['seed']}:{domain}:{i}")
                bt = bh = 0
                bar = tqdm(total=st + sh, unit="B", unit_scale=True, desc=f"{domain}[{i}]")
                for raw in iter_source_fn(src, cfg["seed"]):
                    if bt >= st and bh >= sh:
                        break
                    text = normalize(raw, html_source=src.get("html", False))
                    for chunk in chunk_long(text, corpus["max_chars"]):
                        if not accept(chunk, min_chars=corpus["min_chars"], max_chars=corpus["max_chars"],
                                      russian=dcfg.get("russian", True),
                                      min_cyrillic_ratio=corpus["min_cyrillic_ratio"]):
                            ds["rejected"] += 1
                            continue
                        h = hashlib.blake2b(chunk.encode("utf-8"), digest_size=16).digest()
                        if h in seen:
                            ds["duplicates"] += 1
                            continue
                        seen.add(h)
                        n = len(chunk.encode("utf-8"))
                        to_held = bh < sh and (bt >= st or rng.random() < p_held)
                        (wh if to_held else wt).write({"text": chunk, "domain": domain})
                        if to_held:
                            bh += n
                        else:
                            bt += n
                        ds["docs"] += 1
                        bar.update(n)
                        if bt >= st and bh >= sh:
                            break
                bar.close()
                if bt < st or bh < sh:
                    ds["shortfall"].append({"source": src.get("path", src.get("kind")),
                                            "train_missing": max(0, st - bt),
                                            "heldout_missing": max(0, sh - bh)})
                ds["train_bytes"] += bt
                ds["heldout_bytes"] += bh
        stats[domain] = ds
    stats_path = out / "stats.json"
    old = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    old.update(stats)
    stats_path.write_text(json.dumps(old, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--domains", nargs="*")
    args = ap.parse_args()
    stats = build(load_config(args.config), args.domains)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run** `uv run pytest tests/test_build.py -v` → PASS.

- [ ] **Step 6: Проверка источников вживую (по 1 документу)**

Run:
```bash
uv run python -c "
from ruencod.config import load_config
from ruencod.data.sources import iter_source
cfg = load_config()
for d, v in cfg['corpus']['domains'].items():
    for s in v['sources']:
        t = next(iter(iter_source({**s, 'shuffle_buffer': 10}, 0)))
        print(d, s.get('path', s['kind']), repr(t[:80]))
"
```
Expected: по строке на каждый источник с осмысленным текстом. Если источник падает (gated, другое имя поля) — исправить запись в `experiments.yaml` (для `bigcode/the-stack-smol-xl` при gated выполнить `uv run huggingface-cli login`), а при недоступности подобрать замену и **сообщить пользователю**, какой источник заменён.

- [ ] **Step 7: Commit**

```bash
git add src/ruencod/data tests/test_build.py configs/experiments.yaml
git commit -m "feat: corpus sources and single-pass build with dedup and heldout split"
```

---

### Task 4: Словарь Тихонова и перенос разбора на словоформы

**Files:**
- Create: `src/ruencod/morph/tikhonov.py`, `src/ruencod/morph/project.py`
- Test: `tests/test_morph.py`

**Interfaces:**
- Produces:
  - `tikhonov.Morpheme(text: str, kind: str)` (frozen dataclass), `tikhonov.norm(s) -> str` (lower + ё→е), `tikhonov.parse_line(line) -> tuple[str, list[Morpheme]] | None`, `tikhonov.boundaries(morphs) -> list[int]`, `tikhonov.load_lexicon(path) -> dict[str, list[Morpheme]]`, `tikhonov.split_lemmas(lemmas, seed, test_frac) -> tuple[set[str], set[str]]`, `tikhonov.download(dest, urls) -> Path`
  - `project.CYR_WORD` (regex), `project.Projector(lexicon, exclude=frozenset())`, `Projector.segment(word: str) -> list[int] | None` — смещения границ внутри слова (0 < b < len), `None` = без ограничений.

- [ ] **Step 1: Написать падающий тест**

```python
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
```

- [ ] **Step 2: Run** `uv run pytest tests/test_morph.py -v` → FAIL (import error).

- [ ] **Step 3: Реализация `src/ruencod/morph/tikhonov.py`**

```python
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Morpheme:
    text: str
    kind: str


def norm(s: str) -> str:
    return s.lower().replace("ё", "е")


def parse_line(line: str) -> tuple[str, list[Morpheme]] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 2:
        return None
    word, seg = parts
    morphs = []
    for p in seg.split("/"):
        text, sep, kind = p.rpartition(":")
        if not sep or not text:
            return None
        morphs.append(Morpheme(norm(text), kind))
    if "".join(m.text for m in morphs) != norm(word):
        return None
    return norm(word), morphs


def boundaries(morphs: list[Morpheme]) -> list[int]:
    out, pos = [], 0
    for m in morphs[:-1]:
        pos += len(m.text)
        out.append(pos)
    return out


def load_lexicon(path: Path) -> dict[str, list[Morpheme]]:
    lex = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parsed = parse_line(line)
        if parsed:
            lex[parsed[0]] = parsed[1]
    return lex


def split_lemmas(lemmas: Iterable[str], seed: int, test_frac: float) -> tuple[set[str], set[str]]:
    items = sorted(set(lemmas))
    random.Random(seed).shuffle(items)
    n_test = int(len(items) * test_frac)
    return set(items[n_test:]), set(items[:n_test])


def download(dest: Path, urls: list[str]) -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        chunks = [urllib.request.urlopen(u, timeout=60).read().decode("utf-8") for u in urls]
        dest.write_text("\n".join(c.strip("\n") for c in chunks) + "\n", encoding="utf-8")
    return dest
```

- [ ] **Step 4: Реализация `src/ruencod/morph/project.py`**

```python
import os

import pymorphy3
import regex

from ruencod.morph.tikhonov import Morpheme, boundaries, norm

CYR_WORD = regex.compile(r"[а-яёА-ЯЁ]+")
_POSTFIXES = ("ся", "сь")


def _ends(morphs: list[Morpheme]) -> list[int]:
    out, pos = [], 0
    for m in morphs:
        pos += len(m.text)
        out.append(pos)
    return out


class Projector:
    """Переносит морфемный разбор леммы из словаря Тихонова на словоформу."""

    def __init__(self, lexicon: dict[str, list[Morpheme]], exclude: set[str] | frozenset = frozenset()):
        self.lexicon = lexicon
        self.exclude = exclude
        self._morph = pymorphy3.MorphAnalyzer()

    def segment(self, word: str) -> list[int] | None:
        if not CYR_WORD.fullmatch(word):
            return None
        w = norm(word)
        if w in self.exclude:
            return None
        lemma = norm(self._morph.parse(word.lower())[0].normal_form)
        if lemma in self.exclude:
            return None
        morphs = self.lexicon.get(lemma)
        if morphs is None:
            return None
        lemma_str = "".join(m.text for m in morphs)
        common = len(os.path.commonprefix([w, lemma_str]))
        root_ends = [e for e, m in zip(_ends(morphs), morphs) if m.kind == "ROOT"]
        if not root_ends or common < root_ends[-1]:
            return None  # форма расходится с леммой внутри корня (чередование) — не рискуем
        bounds = {b for b in boundaries(morphs) if b <= common}
        if common < len(w):
            bounds.add(common)
        if morphs[-1].kind == "POSTFIX" and w.endswith(_POSTFIXES) and len(w) - 2 > common:
            bounds.add(len(w) - 2)
        return sorted(b for b in bounds if 0 < b < len(w))
```

- [ ] **Step 5: Run** `uv run pytest tests/test_morph.py -v` → PASS. Если какой-то кейс падает из-за разбора pymorphy3 (`normal_form` не тот), вывести `pymorphy3.MorphAnalyzer().parse(word)` и исправить ожидание **только** если новый результат лингвистически корректен; иначе — алгоритм.

- [ ] **Step 6: Commit**

```bash
git add src/ruencod/morph/tikhonov.py src/ruencod/morph/project.py tests/test_morph.py
git commit -m "feat: Tikhonov lexicon loader and lemma-to-wordform morpheme projection"
```

---

### Task 5: Морфологический кэш и morph-корпус

**Files:**
- Create: `src/ruencod/morph/segment_corpus.py`
- Test: `tests/test_segment_corpus.py`

**Interfaces:**
- Consumes: `pretok.pretokenize`, `pretok.MARK`, `project.Projector`, `project.CYR_WORD`, `tikhonov.*`, `corpus.*`, `jsonl.*`
- Produces: `segment_corpus.word_core(pt) -> tuple[int, str]`, `segment_corpus.segment_text(text, cache: dict[str, list[int]]) -> str`, `segment_corpus.count_words(texts, workers) -> Counter`, `segment_corpus.build_cache(words, lex_path, exclude, workers) -> tuple[dict[str, list[int]], set[str]]` (кэш непустых разборов и множество всех покрытых слов), `save_cache/load_cache`. CLI `python -m ruencod.morph.segment_corpus [--config P]` пишет:
  - `{data}/morph/tikhonov.tsv`, `{data}/morph/test_lemmas.txt`, `{data}/morph/segmentations.tsv`
  - `{data}/morph_corpus/train/{domain}.jsonl.zst`
  - `{results}/morph_coverage.json` (`{"covered_token_share": float, "unique_words": int, "cached_words": int}`)

- [ ] **Step 1: Написать падающий тест**

```python
from collections import Counter

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
```

- [ ] **Step 2: Run** `uv run pytest tests/test_segment_corpus.py -v` → FAIL.

- [ ] **Step 3: Реализация `src/ruencod/morph/segment_corpus.py`**

```python
import argparse
import itertools
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from tqdm import tqdm

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import domain_file, iter_domain
from ruencod.jsonl import ZstJsonlWriter
from ruencod.morph.project import CYR_WORD, Projector
from ruencod.morph.tikhonov import download, load_lexicon, split_lemmas
from ruencod.pretok import MARK, pretokenize

_CTX = mp.get_context("fork")
_PROJ: Projector | None = None
_CACHE: dict[str, list[int]] | None = None


def word_core(pt: str) -> tuple[int, str]:
    start = 0 if pt[:1].isalpha() else 1
    return start, pt[start:]


def segment_text(text: str, cache: dict[str, list[int]]) -> str:
    out: list[str] = []
    for pt in pretokenize(text):
        start, core = word_core(pt)
        bounds = cache.get(core.lower()) if core else None
        if bounds:
            cuts = [0] + [start + b for b in bounds] + [len(pt)]
            out.extend(pt[a:b] for a, b in zip(cuts, cuts[1:]))
        else:
            out.append(pt)
    return MARK.join(out)


def _batched(it: Iterable, n: int) -> Iterator[list]:
    it = iter(it)
    while batch := list(itertools.islice(it, n)):
        yield batch


def _count_batch(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        for pt in pretokenize(t):
            _, core = word_core(pt)
            if core and CYR_WORD.fullmatch(core):
                c[core.lower()] += 1
    return c


def count_words(texts: Iterable[str], workers: int) -> Counter:
    total: Counter = Counter()
    with _CTX.Pool(workers) as pool:
        for c in tqdm(pool.imap_unordered(_count_batch, _batched(texts, 2000)), desc="count words"):
            total.update(c)
    return total


def _init_projector(lex_path: Path, exclude: set[str]) -> None:
    global _PROJ
    _PROJ = Projector(load_lexicon(lex_path), exclude)


def _segment_words(words: list[str]) -> list[tuple[str, list[int] | None]]:
    return [(w, _PROJ.segment(w)) for w in words]


def build_cache(words: Iterable[str], lex_path: Path, exclude: set[str], workers: int):
    cache: dict[str, list[int]] = {}
    covered: set[str] = set()
    with _CTX.Pool(workers, initializer=_init_projector, initargs=(lex_path, exclude)) as pool:
        for res in tqdm(pool.imap_unordered(_segment_words, _batched(words, 5000)), desc="segment words"):
            for w, b in res:
                if b is not None:
                    covered.add(w)
                    if b:
                        cache[w] = b
    return cache, covered


def save_cache(cache: dict[str, list[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for w, b in sorted(cache.items()):
            f.write(f"{w}\t{','.join(map(str, b))}\n")


def load_cache(path: Path) -> dict[str, list[int]]:
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            w, b = line.rstrip("\n").split("\t")
            cache[w] = [int(x) for x in b.split(",")]
    return cache


def _init_seg(cache_path: Path) -> None:
    global _CACHE
    _CACHE = load_cache(cache_path)


def _segment_batch(texts: list[str]) -> list[str]:
    return [segment_text(t, _CACHE) for t in texts]


def run(cfg: dict) -> dict:
    data, mcfg = cfg_path(cfg, "data"), cfg["morph"]
    corpus_dir, morph_dir = data / "corpus", data / "morph"
    domains = list(cfg["corpus"]["domains"])

    lex_path = download(morph_dir / "tikhonov.tsv", mcfg["tikhonov_urls"])
    lexicon = load_lexicon(lex_path)
    _, test = split_lemmas(lexicon, cfg["seed"], mcfg["test_frac"])
    (morph_dir / "test_lemmas.txt").write_text("\n".join(sorted(test)) + "\n", encoding="utf-8")

    def all_train():
        for d in domains:
            yield from iter_domain(corpus_dir, "train", d)

    counts = count_words(all_train(), mcfg["workers"])
    frequent = [w for w, c in counts.items() if c >= mcfg["min_word_freq"]]
    cache, covered = build_cache(frequent, lex_path, test, mcfg["workers"])
    cache_path = morph_dir / "segmentations.tsv"
    save_cache(cache, cache_path)

    total = sum(counts.values())
    coverage = {
        "covered_token_share": sum(counts[w] for w in covered) / total if total else 0.0,
        "unique_words": len(counts),
        "cached_words": len(cache),
    }
    results = cfg_path(cfg, "results")
    results.mkdir(parents=True, exist_ok=True)
    (results / "morph_coverage.json").write_text(json.dumps(coverage, indent=2))

    with _CTX.Pool(mcfg["workers"], initializer=_init_seg, initargs=(cache_path,)) as pool:
        for d in domains:
            with ZstJsonlWriter(domain_file(data / "morph_corpus", "train", d)) as w:
                for batch in tqdm(pool.imap(_segment_batch, _batched(iter_domain(corpus_dir, "train", d), 1000)),
                                  desc=f"segment {d}"):
                    for t in batch:
                        w.write({"text": t})
    return coverage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    print(json.dumps(run(load_config(ap.parse_args().config)), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** `uv run pytest tests/test_segment_corpus.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ruencod/morph/segment_corpus.py tests/test_segment_corpus.py
git commit -m "feat: morpheme cache and morph-marked training corpus"
```

---

### Task 6: Обучение токенизаторов

**Files:**
- Create: `src/ruencod/train/tokenizer.py`, `src/ruencod/train/run.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `pretok.*`, `corpus.iter_mix`, `corpus.domain_shares`, `config.*`
- Produces:
  - `tokenizer.SPECIAL_TOKENS: list[str]` (256), `tokenizer.train_tokenizer(texts, algo: "bpe"|"unigram", vocab_size, morph=False, min_frequency=2) -> tokenizers.Tokenizer`, `tokenizer.save_tokenizer(tok, out_dir) -> None` (пишет `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`)
  - `run.tokenizer_name(algo, vocab) -> str`, `run.train_one(cfg, algo, vocab, force=False) -> Path`. CLI: `python -m ruencod.train.run [--config P] [--algo A --vocab V | --all] [--force]`; пишет `{artifacts}/tokenizers/{name}/` + `meta.json` (`algo, vocab_requested, vocab_actual, seconds, train_bytes_limit`).

- [ ] **Step 1: Написать падающий тест**

```python
import pytest

from ruencod.pretok import MARK, pretokenize
from ruencod.train.run import tokenizer_name
from ruencod.train.tokenizer import SPECIAL_TOKENS, save_tokenizer, train_tokenizer

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
def bpe(request):
    from tests.conftest import make_corpus
    return train_tokenizer(make_corpus(), "bpe", V)


def test_specials():
    assert len(SPECIAL_TOKENS) == 256 and len(set(SPECIAL_TOKENS)) == 256


@pytest.mark.parametrize("algo", ["bpe", "unigram"])
@pytest.mark.parametrize("text", TRICKY)
def test_roundtrip(algo, text):
    from tests.conftest import make_corpus
    tok = train_tokenizer(make_corpus(100), algo, V)
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
    from tests.conftest import make_corpus
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
    assert morph.encode(" книгами").tokens == morph.encode(" книгами").tokens  # инференс работает
    assert morph.decode(morph.encode(docs[0], add_special_tokens=False).ids) == docs[0]


def test_bpe_deterministic():
    from tests.conftest import make_corpus
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
```

- [ ] **Step 2: Run** `uv run pytest tests/test_train.py -v` → FAIL.

- [ ] **Step 3: Реализация `src/ruencod/train/tokenizer.py`**

```python
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
```

- [ ] **Step 4: Реализация `src/ruencod/train/run.py`**

```python
import argparse
import json
import time
from pathlib import Path

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import domain_shares, iter_mix
from ruencod.train.tokenizer import save_tokenizer, train_tokenizer


def tokenizer_name(algo: str, vocab: int) -> str:
    return f"{algo}-{vocab // 1000}k"


def train_one(cfg: dict, algo: str, vocab: int, force: bool = False) -> Path:
    out = cfg_path(cfg, "artifacts") / "tokenizers" / tokenizer_name(algo, vocab)
    if (out / "tokenizer.json").exists() and not force:
        print(f"skip {out.name}: exists")
        return out
    data, shares, tcfg = cfg_path(cfg, "data"), domain_shares(cfg), cfg["train"]
    if algo == "morph-bpe":
        limit = tcfg.get("bpe_max_bytes")
        texts, kind, morph = iter_mix(data / "morph_corpus", "train", shares, limit), "bpe", True
    elif algo == "unigram":
        limit = tcfg["unigram_max_bytes"]
        texts, kind, morph = iter_mix(data / "corpus", "train", shares, limit), "unigram", False
    elif algo == "bpe":
        limit = tcfg.get("bpe_max_bytes")
        texts, kind, morph = iter_mix(data / "corpus", "train", shares, limit), "bpe", False
    else:
        raise ValueError(algo)
    t0 = time.time()
    tok = train_tokenizer(texts, kind, vocab, morph=morph, min_frequency=tcfg["bpe_min_frequency"])
    save_tokenizer(tok, out)
    meta = {"algo": algo, "vocab_requested": vocab, "vocab_actual": tok.get_vocab_size(),
            "seconds": round(time.time() - t0, 1), "train_bytes_limit": limit}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"name": out.name, **meta}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--algo")
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.all:
        for algo in cfg["train"]["algos"]:
            for v in sorted(cfg["train"]["vocab_sizes"], reverse=True):
                train_one(cfg, algo, v, args.force)
    else:
        train_one(cfg, args.algo, args.vocab, args.force)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run** `uv run pytest tests/test_train.py -v` → PASS. Если `test_mark_path_equals_plain_path` падает — сравнить `len(plain.get_vocab())` и первые расхождения в merges (`json.loads(tok.to_str())["model"]["merges"]`); причину (порядок тай-брейка, нормализатор) устранить, а не ослаблять тест.

- [ ] **Step 6: Commit**

```bash
git add src/ruencod/train tests/test_train.py
git commit -m "feat: BPE/Unigram/morph-BPE training with shared pre-tokenization"
```

---

### Task 7: Адаптеры токенизаторов и метрики

**Files:**
- Create: `src/ruencod/eval/adapters.py`, `src/ruencod/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - Протокол `Tok`: атрибуты `name: str`, `vocab_size: int`, `bos_id: int | None`; методы `encode(text) -> list[int]`, `encode_batch(texts) -> list[list[int]]`, `decode(ids) -> str`
  - `adapters.HFTok(name, tok)`, `adapters.TiktokenTok(name, encoding_name)`, `adapters.TransformersTok(name, repo)`, `adapters.SentencePieceTok(name, repo, file)`, `adapters.load_ours(dir) -> HFTok`, `adapters.load_baseline(spec: dict) -> Tok`, `adapters.list_ours(cfg) -> list[Path]`, `adapters.resolve(cfg, name) -> Tok`
  - `metrics.WORD_RE`, `metrics.safe_div(a, b, default=nan)`, `metrics.measure(tok, texts, batch_size=256) -> dict` (ключи `tokens, words, chars, bytes, docs, roundtrip_ok, used_ids` (set)), `metrics.summarize(raw, vocab_size) -> dict` (`fertility, chars_per_token, bytes_per_token, roundtrip, vocab_usage`), `metrics.single_token_rate(tok, word_counts) -> float`, `metrics.token_char_boundaries(tok, text, ids) -> set[int]`, `metrics.aligned_pieces(tok, text) -> list[tuple[str, int]]`, `metrics.morph_scores(tok, items) -> dict` (`morph_p, morph_r, morph_f1`)

- [ ] **Step 1: Написать падающий тест**

```python
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
    assert safe_div(1, 0) != safe_div(1, 0)  # nan


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
```

Примечание: в `Partial` строка `"xyz"` не содержит `½`, поэтому полный декод `"x�yz"` ≠ тексту и последний кусок добирается хвостовой веткой `aligned_pieces` (`text[last_pos:]`, `len(ids) - last_i` = 2).

- [ ] **Step 2: Run** `uv run pytest tests/test_metrics.py -v` → FAIL.

- [ ] **Step 3: Реализация `src/ruencod/eval/metrics.py`**

```python
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
```

- [ ] **Step 4: Реализация `src/ruencod/eval/adapters.py`**

```python
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
        self.tk = AutoTokenizer.from_pretrained(self.repo)

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
```

- [ ] **Step 5: Run** `uv run pytest tests/test_metrics.py -v` → PASS.

- [ ] **Step 6: Проверить все baseline-адаптеры вживую**

Run:
```bash
uv run python -c "
from ruencod.config import load_config
from ruencod.eval.adapters import load_baseline
for s in load_config()['eval']['baselines']:
    t = load_baseline(s); ids = t.encode('Привет, достопримечательности!')
    print(s['name'], t.vocab_size, len(ids), repr(t.decode(ids)))
"
```
Expected: по строке на каждый baseline. Если репозиторий недоступен/gated — заменить на открытое зеркало и сообщить пользователю.

- [ ] **Step 7: Commit**

```bash
git add src/ruencod/eval/adapters.py src/ruencod/eval/metrics.py tests/test_metrics.py
git commit -m "feat: tokenizer adapters and compression/morphology metrics"
```

---

### Task 8: Подсчёт токенов Claude

**Files:**
- Create: `src/ruencod/eval/claude_count.py`
- Test: `tests/test_claude_count.py`

**Interfaces:**
- Produces: `ClaudeCounter(model: str, cache_path: Path, client=None)`; `.count(text) -> int` (без служебных токенов обёртки), `.count_many(texts) -> list[int]`, `.save()`, `.overhead: int`.

- [ ] **Step 1: Написать падающий тест**

```python
from types import SimpleNamespace

from ruencod.eval.claude_count import ClaudeCounter


class FakeMessages:
    def __init__(self):
        self.calls = 0

    def count_tokens(self, model, messages):
        self.calls += 1
        return SimpleNamespace(input_tokens=7 + len(messages[0]["content"]))


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


def test_overhead_subtracted_and_cached(tmp_path):
    cache = tmp_path / "claude.json"
    c1 = ClaudeCounter("claude-opus-5", cache, client=FakeClient())
    assert c1.overhead == 7  # count("a") = 8 → 8 - 1
    assert c1.count_many(["абвг", "де"]) == [4, 2]
    c1.save()

    client2 = FakeClient()
    c2 = ClaudeCounter("claude-opus-5", cache, client=client2)
    assert c2.count_many(["абвг", "де"]) == [4, 2]
    assert client2.messages.calls == 0, "повторный запуск обязан брать всё из кэша"


def test_cache_is_per_model(tmp_path):
    cache = tmp_path / "claude.json"
    c1 = ClaudeCounter("m1", cache, client=FakeClient())
    c1.count("x"); c1.save()
    client2 = FakeClient()
    ClaudeCounter("m2", cache, client=client2).count("x")
    assert client2.messages.calls > 0
```

- [ ] **Step 2: Run** `uv run pytest tests/test_claude_count.py -v` → FAIL.

- [ ] **Step 3: Реализация**

```python
import hashlib
import json
import os
from pathlib import Path


class ClaudeCounter:
    """Точный подсчёт токенов Claude через messages.count_tokens с дисковым кэшем."""

    def __init__(self, model: str, cache_path: Path, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(max_retries=8)
        self.model, self.cache_path, self.client = model, Path(cache_path), client
        self.cache: dict[str, int] = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self._overhead: int | None = None

    def _raw(self, text: str) -> int:
        key = f"{self.model}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
        if key not in self.cache:
            self.cache[key] = self.client.messages.count_tokens(
                model=self.model, messages=[{"role": "user", "content": text}]
            ).input_tokens
        return self.cache[key]

    @property
    def overhead(self) -> int:
        # служебные токены обёртки сообщения: count("a") - 1 (погрешность ±1 токен на документ)
        if self._overhead is None:
            self._overhead = self._raw("a") - 1
        return self._overhead

    def count(self, text: str) -> int:
        return self._raw(text) - self.overhead

    def count_many(self, texts: list[str]) -> list[int]:
        out = []
        for i, t in enumerate(texts):
            out.append(self.count(t))
            if i % 20 == 19:
                self.save()
        self.save()
        return out

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache))
        os.replace(tmp, self.cache_path)
```

- [ ] **Step 4: Run** `uv run pytest tests/test_claude_count.py -v` → PASS.

- [ ] **Step 5: Проверка доступа к API**

Run: `test -n "$ANTHROPIC_API_KEY" && echo key-set || ant auth status`
Expected: ключ или активный профиль. Если нет ни того, ни другого — попросить пользователя выполнить `! ant auth login` или экспортировать `ANTHROPIC_API_KEY`. Затем: `uv run python -c "from ruencod.eval.claude_count import ClaudeCounter as C; c=C('claude-opus-5','/tmp/claude-probe.json'); print(c.overhead, c.count('Привет, мир!'))"` → два положительных числа.

- [ ] **Step 6: Commit**

```bash
git add src/ruencod/eval/claude_count.py tests/test_claude_count.py
git commit -m "feat: cached Claude token counting via count_tokens"
```

---

### Task 9: Прогон оценки → `results/metrics.csv`

**Files:**
- Create: `src/ruencod/eval/run.py`
- Test: `tests/test_eval_run.py`

**Interfaces:**
- Consumes: `adapters.*`, `metrics.*`, `ClaudeCounter`, `corpus.iter_domain`, `corpus.russian_domains`, `tikhonov.load_lexicon/boundaries`
- Produces: `run.run_eval(cfg, only="all", claude_client=None) -> pandas.DataFrame`; файл `{results}/metrics.csv` с колонками `tokenizer, kind, domain, metric, value`. Домены: каждый домен корпуса, `ru_all` (агрегат русских доменов), `morph_test`. Метрики: `fertility, chars_per_token, bytes_per_token, roundtrip, vocab_usage, single_token_rate, morph_p, morph_r, morph_f1, fertility_claude_sample`. `kind ∈ {ours, baseline, claude}`. CLI: `python -m ruencod.eval.run [--config P] [--only ours|baselines|claude|all]`.

- [ ] **Step 1: Написать падающий тест**

```python
import pandas as pd

from ruencod.corpus import domain_file
from ruencod.eval.run import run_eval
from ruencod.jsonl import ZstJsonlWriter
from ruencod.train.tokenizer import save_tokenizer, train_tokenizer
from tests.conftest import make_corpus
from tests.test_claude_count import FakeClient


def _cfg(tmp_path):
    return {
        "seed": 1,
        "paths": {"data": str(tmp_path / "data"), "artifacts": str(tmp_path / "art"), "results": str(tmp_path / "res")},
        "corpus": {"domains": {"web": {"share": 0.5, "russian": True}, "en_code": {"share": 0.5, "russian": False}}},
        "morph": {"test_frac": 0.5},
        "eval": {"max_bytes_per_domain": 10**6, "baselines": [],
                 "claude": {"models": ["claude-opus-5"], "docs_per_domain": 3}},
    }


def test_run_eval_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    data = tmp_path / "data"
    with ZstJsonlWriter(domain_file(data / "corpus", "heldout", "web")) as w:
        for t in make_corpus(20, seed=5):
            w.write({"text": t})
    with ZstJsonlWriter(domain_file(data / "corpus", "heldout", "en_code")) as w:
        w.write({"text": "def f(x):\n    return 1\n"})
    (data / "morph").mkdir(parents=True)
    (data / "morph" / "tikhonov.tsv").write_text("книга\tкниг:ROOT/а:END\n", encoding="utf-8")
    (data / "morph" / "test_lemmas.txt").write_text("книга\n", encoding="utf-8")
    save_tokenizer(train_tokenizer(make_corpus(), "bpe", 700), tmp_path / "art" / "tokenizers" / "bpe-0k")

    df = run_eval(cfg, claude_client=FakeClient())
    assert set(df["tokenizer"]) == {"bpe-0k", "claude-opus-5"}
    get = lambda tok, dom, m: df[(df.tokenizer == tok) & (df.domain == dom) & (df.metric == m)].value.item()
    assert get("bpe-0k", "ru_all", "fertility") > 0
    assert get("bpe-0k", "web", "roundtrip") == 1.0
    assert 0 <= get("bpe-0k", "morph_test", "morph_f1") <= 1
    assert get("claude-opus-5", "ru_all", "fertility_claude_sample") > 0
    assert get("bpe-0k", "ru_all", "fertility_claude_sample") > 0
    assert pd.read_csv(tmp_path / "res" / "metrics.csv").shape[0] == df.shape[0]
```

- [ ] **Step 2: Run** `uv run pytest tests/test_eval_run.py -v` → FAIL.

- [ ] **Step 3: Реализация `src/ruencod/eval/run.py`**

```python
import argparse
from collections import Counter

import pandas as pd
from tqdm import tqdm

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import iter_domain, russian_domains
from ruencod.eval.adapters import list_ours, load_baseline, load_ours
from ruencod.eval.claude_count import ClaudeCounter
from ruencod.eval.metrics import (WORD_RE, measure, morph_scores, safe_div, single_token_rate, summarize)
from ruencod.morph.tikhonov import boundaries, load_lexicon

COLUMNS = ["tokenizer", "kind", "domain", "metric", "value"]


def _rows(name, kind, domain, metrics: dict) -> list[dict]:
    return [{"tokenizer": name, "kind": kind, "domain": domain, "metric": k, "value": v} for k, v in metrics.items()]


def _load_heldout(cfg):
    base = cfg_path(cfg, "data") / "corpus"
    limit = cfg["eval"]["max_bytes_per_domain"]
    return {d: list(iter_domain(base, "heldout", d, limit)) for d in cfg["corpus"]["domains"]}


def _morph_items(cfg):
    mdir = cfg_path(cfg, "data") / "morph"
    if not (mdir / "test_lemmas.txt").exists():
        return []
    lex = load_lexicon(mdir / "tikhonov.tsv")
    test = (mdir / "test_lemmas.txt").read_text(encoding="utf-8").split()
    return [(w, boundaries(lex[w])) for w in test if w in lex]


def _evaluate(tok, kind, heldout, ru, word_counts, morph_items, claude_sample) -> list[dict]:
    rows, agg = [], Counter()
    for d, texts in heldout.items():
        raw = measure(tok, texts)
        m = summarize(raw, tok.vocab_size)
        m["single_token_rate"] = single_token_rate(tok, word_counts[d])
        rows += _rows(tok.name, kind, d, m)
        if d in ru:
            for k in ("tokens", "words", "chars", "bytes"):
                agg[k] += raw[k]
    rows += _rows(tok.name, kind, "ru_all", {
        "fertility": safe_div(agg["tokens"], agg["words"]),
        "chars_per_token": safe_div(agg["chars"], agg["tokens"]),
        "bytes_per_token": safe_div(agg["bytes"], agg["tokens"]),
    })
    if morph_items:
        rows += _rows(tok.name, kind, "morph_test", morph_scores(tok, morph_items))
    rows += _sample_rows(tok.name, kind, {d: sum(map(len, tok.encode_batch(t))) for d, t in claude_sample.items()},
                         claude_sample, ru)
    return rows


def _sample_rows(name, kind, tokens_by_domain: dict, claude_sample: dict, ru) -> list[dict]:
    rows, tot_t, tot_w = [], 0, 0
    for d, texts in claude_sample.items():
        words = sum(len(WORD_RE.findall(t)) for t in texts)
        rows += _rows(name, kind, d, {"fertility_claude_sample": safe_div(tokens_by_domain[d], words)})
        if d in ru:
            tot_t, tot_w = tot_t + tokens_by_domain[d], tot_w + words
    if claude_sample:
        rows += _rows(name, kind, "ru_all", {"fertility_claude_sample": safe_div(tot_t, tot_w)})
    return rows


def run_eval(cfg: dict, only: str = "all", claude_client=None) -> pd.DataFrame:
    heldout = _load_heldout(cfg)
    ru = set(russian_domains(cfg))
    word_counts = {d: Counter(w for t in texts for w in WORD_RE.findall(t)) for d, texts in heldout.items()}
    morph_items = _morph_items(cfg)
    n = cfg["eval"]["claude"]["docs_per_domain"]
    claude_sample = {d: texts[:n] for d, texts in heldout.items()} if n else {}

    toks = []
    if only in ("all", "ours"):
        toks += [(load_ours(p), "ours") for p in list_ours(cfg)]
    if only in ("all", "baselines"):
        toks += [(load_baseline(s), "baseline") for s in cfg["eval"]["baselines"]]

    rows, evaluated = [], set()
    for tok, kind in tqdm(toks, desc="tokenizers"):
        rows += _evaluate(tok, kind, heldout, ru, word_counts, morph_items, claude_sample)
        evaluated.add(tok.name)

    if only in ("all", "claude") and claude_sample:
        for model in cfg["eval"]["claude"]["models"]:
            counter = ClaudeCounter(model, cfg_path(cfg, "results") / "claude_cache.json", client=claude_client)
            tokens = {d: sum(counter.count_many(texts)) for d, texts in claude_sample.items()}
            rows += _sample_rows(model, "claude", tokens, claude_sample, ru)
            evaluated.add(model)

    new = pd.DataFrame(rows, columns=COLUMNS)
    out = cfg_path(cfg, "results") / "metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat([old[~old.tokenizer.isin(evaluated)], new], ignore_index=True)
    new.to_csv(out, index=False)
    return new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--only", default="all", choices=["all", "ours", "baselines", "claude"])
    args = ap.parse_args()
    df = run_eval(load_config(args.config), args.only)
    view = df[(df.domain == "ru_all") & (df.metric.isin(["fertility", "fertility_claude_sample"]))]
    print(view.pivot_table(index="tokenizer", columns="metric", values="value").sort_values("fertility"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** `uv run pytest tests/test_eval_run.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ruencod/eval/run.py tests/test_eval_run.py
git commit -m "feat: evaluation runner producing metrics.csv"
```

---

### Task 10: Мини-LM и bits-per-byte

**Files:**
- Create: `src/ruencod/lm/model.py`, `src/ruencod/lm/train_lm.py`, `src/ruencod/lm/eval_bpb.py`, `src/ruencod/lm/run.py`
- Test: `tests/test_lm.py`

**Interfaces:**
- Consumes: `adapters.resolve`, `corpus.iter_mix/iter_domain/domain_shares/russian_domains`
- Produces:
  - `model.LMConfig(vocab_size, d_model=384, n_layers=6, n_heads=6, context=512)`, `model.TinyLM(cfg)`; `forward(idx, targets=None, reduction="mean") -> (logits, loss|None)`; `.non_embedding_params() -> int`
  - `train_lm.encode_corpus(tok, texts) -> np.ndarray[uint32]` (`[bos]+ids` на документ), `train_lm.train_lm(model, tokens, lm_cfg: dict, seed: int, log_every=50) -> dict` (`steps, train_tokens, seconds, tokens_per_sec, final_loss`)
  - `eval_bpb.nll_bits_and_bytes(model, tok, texts, context) -> tuple[float, int]`
  - CLI `python -m ruencod.lm.run [--config P] [--smoke] [--candidates a b]` → `{results}/lm_bpb.csv` (`candidate, domain, bpb, train_tokens, tokens_per_sec, seconds`), домены + `ru_all`. `--smoke` обучает на 1 МБ и печатает прогноз времени полного прогона.

- [ ] **Step 1: Написать падающий тест**

```python
import math

import numpy as np
import torch

from ruencod.lm.eval_bpb import nll_bits_and_bytes
from ruencod.lm.model import LMConfig, TinyLM
from ruencod.lm.train_lm import encode_corpus, train_lm
from tests.test_metrics import CharTok


def test_default_size_is_10m_non_embedding():
    m = TinyLM(LMConfig(vocab_size=1000))
    assert 10.5e6 < m.non_embedding_params() < 10.8e6


def test_forward_shapes():
    m = TinyLM(LMConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, context=16))
    x = torch.randint(0, 50, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 50) and loss.ndim == 0


class UniformLM(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, idx, targets=None, reduction="mean"):
        logits = torch.zeros(*idx.shape, self.v)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, self.v), targets.view(-1), reduction=reduction)
        return logits, loss


class SmallCharTok(CharTok):
    vocab_size, bos_id = 2000, 0


def test_bpb_counts_every_target_once():
    tok = SmallCharTok()
    texts = ["абв" * 10, "x"]
    bits, nbytes = nll_bits_and_bytes(UniformLM(2000), tok, texts, context=7)
    n_targets = sum(len(t) for t in texts)  # bos не предсказывается, каждый символ — ровно один раз
    assert nbytes == sum(len(t.encode()) for t in texts)
    assert math.isclose(bits, n_targets * math.log2(2000), rel_tol=1e-5)


def test_tiny_overfit():
    tok = SmallCharTok()
    tokens = encode_corpus(tok, ["абвгд" * 200])
    assert tokens.dtype == np.uint32 and tokens[0] == 0
    m = TinyLM(LMConfig(vocab_size=2000, d_model=32, n_layers=1, n_heads=2, context=16))
    cfg = {"context": 16, "batch_size": 8, "lr": 3e-3, "warmup_frac": 0.05, "weight_decay": 0.0, "threads": 2}
    stats = train_lm(m, tokens, cfg, seed=0)
    assert stats["final_loss"] < 2.0 and stats["steps"] > 0
```

- [ ] **Step 2: Run** `uv run pytest tests/test_lm.py -v` → FAIL.

- [ ] **Step 3: Реализация `src/ruencod/lm/model.py`**

```python
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    context: int = 512


def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    t = x.shape[-2]
    c, s = cos[:t], sin[:t]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c: LMConfig):
        super().__init__()
        d = c.d_model
        self.h = c.n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(c, dim=2)
        q, k, v = (z.view(b, t, self.h, c // self.h).transpose(1, 2) for z in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, c)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyLM(nn.Module):
    def __init__(self, c: LMConfig):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)
        cos, sin = rope_cache(c.context, c.d_model // c.n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, idx, targets=None, reduction="mean"):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        logits = self.ln_f(x) @ self.emb.weight.T  # связанные эмбеддинги
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, self.c.vocab_size), targets.reshape(-1), reduction=reduction)
        return logits, loss

    def non_embedding_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("emb"))
```

- [ ] **Step 4: Реализация `src/ruencod/lm/train_lm.py`**

```python
import math
import time

import numpy as np
import torch


def encode_corpus(tok, texts) -> np.ndarray:
    parts = []
    batch = []
    for t in texts:
        batch.append(t)
        if len(batch) == 256:
            parts += [[tok.bos_id] + ids for ids in tok.encode_batch(batch)]
            batch = []
    if batch:
        parts += [[tok.bos_id] + ids for ids in tok.encode_batch(batch)]
    return np.fromiter((i for p in parts for i in p), dtype=np.uint32)


def train_lm(model, tokens: np.ndarray, lm_cfg: dict, seed: int, log_every: int = 50) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(lm_cfg["threads"])
    ctx, bs = lm_cfg["context"], lm_cfg["batch_size"]
    n_chunks = (len(tokens) - 1) // ctx
    order = np.random.default_rng(seed).permutation(n_chunks)
    steps = math.ceil(n_chunks / bs)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": lm_cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lm_cfg["lr"], betas=(0.9, 0.95))
    warmup = max(1, int(steps * lm_cfg["warmup_frac"]))
    lr_max, lr_min = lm_cfg["lr"], lm_cfg["lr"] * 0.1
    t0, loss_val = time.time(), float("nan")
    model.train()
    for step in range(steps):
        idx = order[step * bs:(step + 1) * bs]
        win = np.stack([tokens[i * ctx:i * ctx + ctx + 1] for i in idx]).astype(np.int64)
        x, y = torch.from_numpy(win[:, :-1]), torch.from_numpy(win[:, 1:])
        if step < warmup:
            lr = lr_max * (step + 1) / warmup
        else:
            prog = (step - warmup) / max(1, steps - warmup)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = lr
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_val = loss.item()
        if step % log_every == 0:
            done = (step + 1) * bs * ctx
            print(f"step {step}/{steps} loss {loss_val:.3f} tok/s {done / (time.time() - t0):.0f}", flush=True)
    secs = time.time() - t0
    trained = n_chunks * ctx
    return {"steps": steps, "train_tokens": trained, "seconds": round(secs, 1),
            "tokens_per_sec": trained / secs if secs else 0.0, "final_loss": loss_val}
```

- [ ] **Step 5: Реализация `src/ruencod/lm/eval_bpb.py`**

```python
import math

import torch


@torch.no_grad()
def nll_bits_and_bytes(model, tok, texts, context: int) -> tuple[float, int]:
    model.eval()
    nll, nbytes = 0.0, 0
    for t in texts:
        ids = [tok.bos_id] + tok.encode(t)
        nbytes += len(t.encode("utf-8"))
        for s in range(0, len(ids) - 1, context):
            chunk = ids[s:s + context + 1]
            x = torch.tensor(chunk[:-1]).unsqueeze(0)
            y = torch.tensor(chunk[1:]).unsqueeze(0)
            _, loss = model(x, y, reduction="sum")
            nll += loss.item()
    return nll / math.log(2), nbytes
```

- [ ] **Step 6: Реализация `src/ruencod/lm/run.py`**

```python
import argparse
import json

import pandas as pd

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.corpus import domain_shares, iter_domain, iter_mix, russian_domains
from ruencod.eval.adapters import resolve
from ruencod.lm.eval_bpb import nll_bits_and_bytes
from ruencod.lm.model import LMConfig, TinyLM
from ruencod.lm.train_lm import encode_corpus, train_lm


def run_candidate(cfg: dict, name: str, train_bytes: int) -> list[dict]:
    lm = cfg["lm"]
    tok = resolve(cfg, name)
    data = cfg_path(cfg, "data") / "corpus"
    tokens = encode_corpus(tok, iter_mix(data, "train", domain_shares(cfg), train_bytes))
    model = TinyLM(LMConfig(vocab_size=tok.vocab_size, d_model=lm["d_model"], n_layers=lm["n_layers"],
                            n_heads=lm["n_heads"], context=lm["context"]))
    stats = train_lm(model, tokens, lm, cfg["seed"])
    rows, ru_bits, ru_bytes = [], 0.0, 0
    for d in cfg["corpus"]["domains"]:
        texts = list(iter_domain(data, "heldout", d, lm["eval_bytes_per_domain"]))
        bits, nbytes = nll_bits_and_bytes(model, tok, texts, lm["context"])
        rows.append({"candidate": name, "domain": d, "bpb": bits / nbytes, **stats})
        if d in russian_domains(cfg):
            ru_bits, ru_bytes = ru_bits + bits, ru_bytes + nbytes
    rows.append({"candidate": name, "domain": "ru_all", "bpb": ru_bits / ru_bytes, **stats})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--candidates", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    names = args.candidates or cfg["lm"]["candidates"]
    full_bytes = cfg["lm"]["train_bytes"]
    out = cfg_path(cfg, "results") / ("lm_bpb_smoke.csv" if args.smoke else "lm_bpb.csv")
    all_rows = []
    for name in names:
        rows = run_candidate(cfg, name, 1_000_000 if args.smoke else full_bytes)
        all_rows += rows
        if args.smoke:
            r = rows[-1]
            est_h = r["train_tokens"] * (full_bytes / 1_000_000) / r["tokens_per_sec"] / 3600
            print(json.dumps({"candidate": name, "tokens_per_sec": round(r["tokens_per_sec"]),
                              "estimated_full_hours": round(est_h, 2)}))
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out, index=False)  # сохраняем после каждого кандидата
    print(pd.DataFrame(all_rows).query("domain == 'ru_all'")[["candidate", "bpb", "train_tokens", "seconds"]])


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run** `uv run pytest tests/test_lm.py -v` → PASS.

- [ ] **Step 8: Commit**

```bash
git add src/ruencod/lm tests/test_lm.py
git commit -m "feat: tiny RoPE LM, CPU training and bits-per-byte evaluation"
```

---

### Task 11: Графики, картинки разбиения, сводка

**Files:**
- Create: `src/ruencod/viz/charts.py`, `src/ruencod/viz/splits.py`, `src/ruencod/viz/summary.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: `metrics.csv`, `lm_bpb.csv`, `cfg["prices"]`, `cfg["final"]["tokenizer"]`, `cfg["examples"]`, `adapters.resolve`, `metrics.aligned_pieces`, `ClaudeCounter`
- Produces:
  - `charts.make_all(cfg) -> list[Path]` → `{results}/figures/{1_fertility,2_vocab_curve,3_domains,4_cost,5_lm_bpb}.png` (5-й — только если есть `lm_bpb.csv`)
  - `splits.render(rows: list[tuple[str, list[tuple[str, int]] | str]], title, out) -> Path`; `splits.make_all(cfg, claude_client=None) -> list[Path]` → `{results}/figures/split_{i}.png`
  - `summary.make(cfg, claude_client=None) -> dict` → `{results}/summary.json` и `web/examples.json`
  - CLI: `python -m ruencod.viz.charts|splits|summary [--config P]`

- [ ] **Step 0: Загрузить скилл `dataviz`** (обязательно перед кодом графиков) и сверить с ним палитру, подписи и размеры; константы `PALETTE`/шрифты ниже скорректировать по скиллу, сохранив интерфейсы.

- [ ] **Step 1: Написать падающий тест**

```python
import pandas as pd

from ruencod.viz.charts import make_all
from ruencod.viz.splits import render


def _cfg(tmp_path):
    return {
        "paths": {"results": str(tmp_path)},
        "corpus": {"domains": {"web": {"share": 1.0, "russian": True}}},
        "train": {"vocab_sizes": [16000, 32000], "algos": ["bpe", "unigram", "morph-bpe"]},
        "eval": {"baselines": [{"name": "o200k", "label": "GPT-4o (o200k)"}]},
        "prices": {"gpt": {"label": "GPT-4o", "tokenizer": "o200k", "usd_per_mtok": 2.5},
                   "claude": {"label": "Claude Opus 5", "tokenizer": "claude-opus-5", "usd_per_mtok": 5.0}},
        "final": {"tokenizer": "bpe-32k"},
    }


def test_charts(tmp_path):
    rows = []
    for tok, kind, f in [("bpe-16k", "ours", 2.0), ("bpe-32k", "ours", 1.8), ("unigram-16k", "ours", 2.1),
                         ("unigram-32k", "ours", 1.9), ("morph-bpe-16k", "ours", 2.3), ("morph-bpe-32k", "ours", 2.1),
                         ("o200k", "baseline", 2.6), ("claude-opus-5", "claude", 3.0)]:
        for dom in ("web", "ru_all"):
            rows.append({"tokenizer": tok, "kind": kind, "domain": dom, "metric": "fertility_claude_sample", "value": f})
            if kind != "claude":
                rows.append({"tokenizer": tok, "kind": kind, "domain": dom, "metric": "fertility", "value": f})
    pd.DataFrame(rows).to_csv(tmp_path / "metrics.csv", index=False)
    pd.DataFrame([{"candidate": "bpe-32k", "domain": "ru_all", "bpb": 1.1}]).to_csv(tmp_path / "lm_bpb.csv", index=False)
    paths = make_all(_cfg(tmp_path))
    assert len(paths) == 5 and all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_render_split(tmp_path):
    out = render([("GPT-4o", [("достоприм", 1), ("ечательно", 2), ("стями", 1)]),
                  ("Claude Opus 5", "7 токенов"),
                  ("ruEncod", [("достопримечательн", 1), ("остями", 1)])],
                 "достопримечательностями", tmp_path / "s.png")
    assert out.exists() and out.stat().st_size > 1000
```

- [ ] **Step 2: Run** `uv run pytest tests/test_viz.py -v` → FAIL.

- [ ] **Step 3: Реализация `src/ruencod/viz/charts.py`**

```python
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config

PALETTE = {"ours": "#2f6fdb", "baseline": "#9aa3ad", "claude": "#d97757",
           "bpe": "#2f6fdb", "unigram": "#18a058", "morph-bpe": "#a347ba"}
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _get(df, metric, domain="ru_all"):
    sub = df[(df.metric == metric) & (df.domain == domain)]
    return dict(zip(sub.tokenizer, sub.value))


def _label(cfg, name):
    for b in cfg["eval"]["baselines"]:
        if b["name"] == name:
            return b.get("label", name)
    if name == cfg["prices"]["claude"]["tokenizer"]:
        return cfg["prices"]["claude"]["label"]
    return f"ruEncod {name}"


def _kind(df, name):
    return df[df.tokenizer == name].kind.iloc[0]


def fertility_bar(cfg, df, out: Path) -> Path:
    vals = _get(df, "fertility_claude_sample")
    final_v = cfg["final"]["tokenizer"].split("-")[-1]
    keep = {k: v for k, v in vals.items() if _kind(df, k) != "ours" or k.endswith(final_v)}
    items = sorted(keep.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(items) + 1.5))
    ax.barh([_label(cfg, k) for k, _ in items], [v for _, v in items],
            color=[PALETTE[_kind(df, k)] for k, _ in items])
    for i, (_, v) in enumerate(items):
        ax.text(v, i, f" {v:.2f}", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("токенов на слово (русский текст, меньше — лучше)")
    ax.set_title("Сколько токенов уходит на одно русское слово")
    fig.savefig(out)
    plt.close(fig)
    return out


def vocab_curve(cfg, df, out: Path) -> Path:
    vals = _get(df, "fertility")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sizes = sorted(cfg["train"]["vocab_sizes"])
    for algo in cfg["train"]["algos"]:
        ys = [vals.get(f"{algo}-{v // 1000}k") for v in sizes]
        pts = [(v // 1000, y) for v, y in zip(sizes, ys) if y is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=algo, color=PALETTE[algo])
    if "o200k" in vals:
        ax.axhline(vals["o200k"], ls="--", color=PALETTE["baseline"], label=_label(cfg, "o200k"))
    ax.set_xlabel("размер словаря, тыс. токенов")
    ax.set_ylabel("токенов на слово")
    ax.set_title("Больше словарь — меньше токенов, но отдача падает")
    ax.legend()
    fig.savefig(out)
    plt.close(fig)
    return out


def domains_bar(cfg, df, out: Path) -> Path:
    names = [cfg["final"]["tokenizer"], "o200k", cfg["prices"]["claude"]["tokenizer"]]
    sub = df[(df.metric == "fertility_claude_sample") & (df.tokenizer.isin(names)) & (df.domain != "ru_all")]
    pv = sub.pivot_table(index="domain", columns="tokenizer", values="value").reindex(columns=names)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    w = 0.27
    for i, n in enumerate(names):
        ax.bar([x + (i - 1) * w for x in range(len(pv))], pv[n], width=w, label=_label(cfg, n),
               color=PALETTE[_kind(df, n)] if n != "o200k" else PALETTE["baseline"])
    ax.set_xticks(range(len(pv)), pv.index, rotation=20)
    ax.set_ylabel("токенов на слово")
    ax.set_title("По доменам: где выигрыш больше")
    ax.legend()
    fig.savefig(out)
    plt.close(fig)
    return out


def cost_bar(cfg, df, out: Path) -> Path:
    f = _get(df, "fertility_claude_sample")
    final = cfg["final"]["tokenizer"]
    bars = []
    for key in ("gpt", "claude"):
        p = cfg["prices"][key]
        price = p["usd_per_mtok"]
        bars.append((f"{p['label']}", f[p["tokenizer"]] * 1000 * price / 1e6, "baseline" if key == "gpt" else "claude"))
        bars.append((f"ruEncod по цене {p['label']}", f[final] * 1000 * price / 1e6, "ours"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh([b[0] for b in bars], [b[1] for b in bars], color=[PALETTE[b[2]] for b in bars])
    for i, b in enumerate(bars):
        ax.text(b[1], i, f" ${b[1]:.4f}", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("$ за 1000 слов русского текста (входные токены)")
    ax.set_title(f"Цена 1000 слов (цены на {cfg['prices'].get('verified_on', '')})")
    fig.savefig(out)
    plt.close(fig)
    return out


def lm_bar(cfg, lm: pd.DataFrame, out: Path) -> Path:
    sub = lm[lm.domain == "ru_all"].sort_values("bpb")
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(sub) + 1.5))
    ax.barh(sub.candidate, sub.bpb, color=PALETTE["ours"])
    for i, v in enumerate(sub.bpb):
        ax.text(v, i, f" {v:.3f}", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("бит на байт (меньше — лучше)")
    ax.set_title("Мини-модель 10M: какой токенизатор учится лучше")
    fig.savefig(out)
    plt.close(fig)
    return out


def make_all(cfg: dict) -> list[Path]:
    res = cfg_path(cfg, "results")
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res / "metrics.csv")
    paths = [fertility_bar(cfg, df, figs / "1_fertility.png"), vocab_curve(cfg, df, figs / "2_vocab_curve.png"),
             domains_bar(cfg, df, figs / "3_domains.png"), cost_bar(cfg, df, figs / "4_cost.png")]
    if (res / "lm_bpb.csv").exists():
        paths.append(lm_bar(cfg, pd.read_csv(res / "lm_bpb.csv"), figs / "5_lm_bpb.png"))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    for p in make_all(load_config(ap.parse_args().config)):
        print(p)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Реализация `src/ruencod/viz/splits.py`**

```python
import argparse
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.eval.adapters import resolve
from ruencod.eval.claude_count import ClaudeCounter
from ruencod.eval.metrics import aligned_pieces

COLORS = ["#cfe0fc", "#fde2c8", "#d5f2dc", "#f3d4f5", "#fff1b8", "#d9e7ec"]
_FONT = font_manager.findfont("DejaVu Sans")


def _shown(piece: str) -> str:
    return piece.replace(" ", "·").replace("\n", "↵")


def render(rows, title: str, out: Path, width: int = 1600) -> Path:
    big, small = ImageFont.truetype(_FONT, 34), ImageFont.truetype(_FONT, 26)
    label_w, pad, line_h = 330, 24, 64
    lines = []  # (label, [(text, n, x, y_row)]) с переносом
    for label, content in rows:
        if isinstance(content, str):
            lines.append((label, content, 1))
            continue
        x, n_rows = label_w, 1
        for text, _ in content:
            w = big.getlength(_shown(text)) + 20
            if x + w > width - pad:
                x, n_rows = label_w, n_rows + 1
            x += w + 6
        lines.append((label, content, n_rows))
    height = 120 + sum(n * line_h + 30 for *_, n in lines)
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((pad, 30), title, font=big, fill="#111")
    y = 110
    for label, content, n_rows in lines:
        d.text((pad, y + 14), label, font=small, fill="#444")
        if isinstance(content, str):
            d.text((label_w, y + 10), content, font=big, fill="#111")
        else:
            total = sum(n for _, n in content)
            x, row_y = label_w, y
            for i, (text, n) in enumerate(content):
                s = _shown(text)
                w = big.getlength(s) + 20
                if x + w > width - pad:
                    x, row_y = label_w, row_y + line_h
                d.rounded_rectangle([x, row_y, x + w, row_y + 52], 8, fill=COLORS[i % len(COLORS)])
                d.text((x + 10, row_y + 6), s, font=big, fill="#111")
                if n > 1:
                    d.text((x + w - 26, row_y - 6), f"×{n}", font=small, fill="#c0392b")
                x += w + 6
            d.text((width - pad - 170, y + 14), f"{total} ток.", font=small, fill="#444")
        y += n_rows * line_h + 30
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def make_all(cfg: dict, claude_client=None) -> list[Path]:
    figs = cfg_path(cfg, "results") / "figures"
    names = ["o200k", "llama-3", cfg["final"]["tokenizer"]]
    morph = "morph-" + cfg["final"]["tokenizer"].replace("morph-", "").replace("unigram", "bpe")
    if morph != cfg["final"]["tokenizer"]:
        names.append(morph)
    toks = [resolve(cfg, n) for n in names]
    counter = ClaudeCounter(cfg["prices"]["claude"]["tokenizer"], cfg_path(cfg, "results") / "claude_cache.json",
                            client=claude_client)
    paths = []
    for i, text in enumerate(cfg["examples"]):
        rows = [(_label(cfg, t.name), aligned_pieces(t, text)) for t in toks]
        rows.insert(1, (cfg["prices"]["claude"]["label"], f"{counter.count(text)} токенов (разбиение закрыто)"))
        paths.append(render(rows, text, figs / f"split_{i}.png"))
    counter.save()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    for p in make_all(load_config(ap.parse_args().config)):
        print(p)


if __name__ == "__main__":
    main()
```

(`_label` импортируется из `ruencod.viz.charts`: добавить `from ruencod.viz.charts import _label` к импортам `splits.py`.)

- [ ] **Step 5: Реализация `src/ruencod/viz/summary.py`**

```python
import argparse
import json

import pandas as pd

from ruencod.config import DEFAULT_CONFIG, ROOT, cfg_path, load_config
from ruencod.eval.adapters import resolve
from ruencod.eval.claude_count import ClaudeCounter


def make(cfg: dict, claude_client=None) -> dict:
    res = cfg_path(cfg, "results")
    df = pd.read_csv(res / "metrics.csv")
    f = df[(df.domain == "ru_all") & (df.metric == "fertility_claude_sample")].set_index("tokenizer").value
    final, gpt, claude = cfg["final"]["tokenizer"], cfg["prices"]["gpt"], cfg["prices"]["claude"]
    s = {
        "final": final,
        "fertility": {"final": f[final], "gpt": f[gpt["tokenizer"]], "claude": f[claude["tokenizer"]]},
        "savings_vs_gpt_pct": 100 * (1 - f[final] / f[gpt["tokenizer"]]),
        "savings_vs_claude_pct": 100 * (1 - f[final] / f[claude["tokenizer"]]),
        "usd_per_1000_words": {
            "gpt": f[gpt["tokenizer"]] * gpt["usd_per_mtok"] / 1000,
            "claude": f[claude["tokenizer"]] * claude["usd_per_mtok"] / 1000,
            "final_at_gpt_price": f[final] * gpt["usd_per_mtok"] / 1000,
            "final_at_claude_price": f[final] * claude["usd_per_mtok"] / 1000,
        },
        "prices_verified_on": cfg["prices"]["verified_on"],
    }
    cov = res / "morph_coverage.json"
    if cov.exists():
        s["morph_coverage"] = json.loads(cov.read_text())
    if (res / "lm_bpb.csv").exists():
        lm = pd.read_csv(res / "lm_bpb.csv").query("domain == 'ru_all'")
        s["lm_bpb"] = dict(zip(lm.candidate, lm.bpb))
    (res / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))

    counter = ClaudeCounter(claude["tokenizer"], res / "claude_cache.json", client=claude_client)
    ours, o200k = resolve(cfg, final), resolve(cfg, gpt["tokenizer"])
    examples = [{"text": t, "claude_tokens": counter.count(t), "gpt_tokens": len(o200k.encode(t)),
                 "ours_tokens": len(ours.encode(t))} for t in cfg["examples"]]
    counter.save()
    web = ROOT / "web"
    web.mkdir(exist_ok=True)
    (web / "examples.json").write_text(json.dumps(
        {"claude_label": claude["label"], "summary": s, "examples": examples}, ensure_ascii=False, indent=2))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    print(json.dumps(make(load_config(ap.parse_args().config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run** `uv run pytest tests/test_viz.py -v` → PASS. Открыть `tmp`-картинки через Read и визуально проверить читаемость кириллицы.

- [ ] **Step 7: Commit**

```bash
git add src/ruencod/viz tests/test_viz.py
git commit -m "feat: charts, token split images and post summary"
```

---

### Task 12: Интерактивная страница

**Files:**
- Create: `web/index.html`; при сборке копируются `web/tokenizer/tokenizer.json`, `web/tokenizer/tokenizer_config.json` (из финалиста), `web/examples.json` (из Task 11)

**Interfaces:**
- Consumes: финалист `artifacts/tokenizers/{final}/`, `web/examples.json`

- [ ] **Step 0: Загрузить скилл `artifact-design`** и следовать его контракту (title, токены цветов, тёмная тема, CDN только cdnjs/jsdelivr, ширина телефона).

- [ ] **Step 1: Проверить API библиотек в браузере**

Через Context7 (`resolve-library-id` → `query-docs`) подтвердить:
- `@huggingface/transformers`: конструктор `PreTrainedTokenizer(tokenizerJSON, tokenizerConfig)` (или `AutoTokenizer.from_pretrained` с `env.localModelPath`), `encode(text, { add_special_tokens: false })`, `decode(ids)`.
- `js-tiktoken`: `import { Tiktoken } from ".../lite"` + `import o200k_base from ".../ranks/o200k_base"`; `new Tiktoken(o200k_base).encode(text)`.
Зафиксировать точные версии: `npm view @huggingface/transformers version`, `npm view js-tiktoken version`.

- [ ] **Step 2: Скопировать файлы финалиста**

```bash
FINAL=$(uv run python -c "from ruencod.config import load_config; print(load_config()['final']['tokenizer'])")
mkdir -p web/tokenizer && cp artifacts/tokenizers/$FINAL/tokenizer.json artifacts/tokenizers/$FINAL/tokenizer_config.json web/tokenizer/
```

- [ ] **Step 3: Написать `web/index.html`** — функциональное ядро (оформление по `artifact-design`):

```html
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Токены русского языка</title>
<style>
  :root { --bg:#fff; --fg:#111; --muted:#666; --ours:#2f6fdb; --gpt:#9aa3ad; --t1:#cfe0fc; --t2:#fde2c8; --t3:#d5f2dc; --t4:#f3d4f5; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#121417; --fg:#eee; --muted:#9aa; --t1:#24406b; --t2:#6b4a24; --t3:#245c36; --t4:#5a2d5e; } }
  :root[data-theme="dark"] { --bg:#121417; --fg:#eee; --muted:#9aa; --t1:#24406b; --t2:#6b4a24; --t3:#245c36; --t4:#5a2d5e; }
  body { background:var(--bg); color:var(--fg); font:16px/1.5 system-ui,sans-serif; margin:0 auto; max-width:900px; padding:16px; }
  textarea { width:100%; box-sizing:border-box; min-height:120px; font:inherit; }
  .row { margin:16px 0; } .tok { display:inline-block; padding:2px 4px; margin:1px; border-radius:4px; white-space:pre; }
  .count { color:var(--muted); }
</style>
</head>
<body>
<h1>Как нейросеть видит русский текст</h1>
<textarea id="input">Нейросеть не читает буквы — она видит токены. Достопримечательностями Москвы любовались туристы.</textarea>
<div class="row"><b>GPT-4o (o200k)</b> <span class="count" id="gpt-count"></span><div id="gpt"></div></div>
<div class="row"><b>ruEncod</b> <span class="count" id="ours-count"></span><div id="ours"></div></div>
<p id="delta"></p>
<h2>Claude (заранее посчитано)</h2>
<table id="examples"></table>
<script type="module">
import { PreTrainedTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@VERSION/+esm";
import { Tiktoken } from "https://cdn.jsdelivr.net/npm/js-tiktoken@VERSION/lite/+esm";
import o200k from "https://cdn.jsdelivr.net/npm/js-tiktoken@VERSION/ranks/o200k_base/+esm";

const [tj, tc, ex] = await Promise.all(
  ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "examples.json"].map(u => fetch(u).then(r => r.json())));
const ours = new PreTrainedTokenizer(tj, tc);
const gpt = new Tiktoken(o200k);

// те же правила, что metrics.aligned_pieces: режем по границам, где префикс декодируется в префикс текста
function pieces(ids, decode, text) {
  const out = []; let lastPos = 0, lastI = 0;
  for (let i = 1; i <= ids.length; i++) {
    const p = decode(ids.slice(0, i));
    const pos = text.startsWith(p) ? p.length : (text.startsWith(" " + p) ? p.length + 1 : -1);
    if (pos > lastPos) { out.push([text.slice(lastPos, pos), i - lastI]); lastPos = pos; lastI = i; }
  }
  if (lastPos < text.length) out.push([text.slice(lastPos), ids.length - lastI]);
  return out;
}
function show(el, countEl, ps, total) {
  el.replaceChildren(...ps.map(([t, n], i) => {
    const s = document.createElement("span");
    s.className = "tok"; s.style.background = `var(--t${(i % 4) + 1})`;
    s.textContent = t.replaceAll(" ", "·") + (n > 1 ? ` ×${n}` : "");
    return s;
  }));
  countEl.textContent = `${total} токенов`;
}
function update() {
  const text = document.getElementById("input").value.slice(0, 2000);  // O(n²) декод — ограничиваем длину
  const g = gpt.encode(text), o = ours.encode(text, { add_special_tokens: false });
  show(document.getElementById("gpt"), document.getElementById("gpt-count"), pieces(g, ids => gpt.decode(ids), text), g.length);
  show(document.getElementById("ours"), document.getElementById("ours-count"), pieces(o, ids => ours.decode(ids), text), o.length);
  document.getElementById("delta").textContent = g.length ? `ruEncod: ${Math.round(100 * (1 - o.length / g.length))}% меньше токенов, чем GPT-4o` : "";
}
document.getElementById("input").addEventListener("input", update);
update();
document.getElementById("examples").innerHTML = "<tr><th>Текст</th><th>GPT-4o</th><th>" + ex.claude_label + "</th><th>ruEncod</th></tr>" +
  ex.examples.map(e => `<tr><td>${e.text.replace(/</g, "&lt;")}</td><td>${e.gpt_tokens}</td><td>${e.claude_tokens}</td><td>${e.ours_tokens}</td></tr>`).join("");
</script>
</body>
</html>
```

`VERSION` заменить на версии из Step 1 (это не placeholder, а подстановка проверенного значения).

- [ ] **Step 4: Локальная проверка паритета**

Run: `cd web && python3 -m http.server 8765` (в фоне), открыть страницу через `claude-in-chrome` или встроенный браузер, ввести строку `Достопримечательностями 2024` и сравнить число токенов ruEncod с `uv run python -c "from ruencod.config import load_config; from ruencod.eval.adapters import resolve; print(len(resolve(load_config(), load_config()['final']['tokenizer']).encode('Достопримечательностями 2024')))"`.
Expected: числа совпадают; совпадает и счёт GPT-4o с `tiktoken`.

- [ ] **Step 5: Публикация** — Artifact `publish` c `file_path: web/index.html`, `files: {"tokenizer/tokenizer.json": "web/tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json": "web/tokenizer/tokenizer_config.json", "examples.json": "web/examples.json"}`, `icon: "text"`. Передать ссылку пользователю (страница приватна, решение о публичном доступе — за ним).

- [ ] **Step 6: Commit**

```bash
git add web/index.html web/examples.json
git commit -m "feat: interactive tokenizer comparison page"
```

---

### Task 13: Smoke-прогон всего конвейера

**Files:** нет новых; используется `configs/smoke.yaml`.

- [ ] **Step 1: Данные (30 МБ)**: `uv run python -m ruencod.data.build --config configs/smoke.yaml` → `data_smoke/corpus/stats.json`, у каждого домена `train_bytes > 0`.
- [ ] **Step 2: Морфология**: `uv run python -m ruencod.morph.segment_corpus --config configs/smoke.yaml` → `results_smoke/morph_coverage.json` с `covered_token_share > 0.2`.
- [ ] **Step 3: Обучение**: `uv run python -m ruencod.train.run --config configs/smoke.yaml --all` → 6 папок в `artifacts_smoke/tokenizers/`.
- [ ] **Step 4: Оценка**: `uv run python -m ruencod.eval.run --config configs/smoke.yaml` → таблица; у наших `roundtrip == 1.0` на всех доменах (`uv run python -c "import pandas as pd; d=pd.read_csv('results_smoke/metrics.csv'); print(d[(d.metric=='roundtrip')&(d.kind=='ours')].value.min())"` → `1.0`).
- [ ] **Step 5: LM smoke**: `uv run python -m ruencod.lm.run --config configs/smoke.yaml --smoke` → строки с `tokens_per_sec` и `estimated_full_hours`.
- [ ] **Step 6: Визуализация**: `uv run python -m ruencod.viz.charts --config configs/smoke.yaml` → 4–5 PNG; просмотреть их через Read.
- [ ] **Step 7:** Исправить всё, что сломалось (каждый фикс — с регрессионным тестом), затем `uv run pytest -q` → всё зелёное. Commit: `git commit -am "fix: issues found in smoke run"` (если были правки).

---

### Task 14: Полный прогон: данные, морфология, обучение, оценка

- [ ] **Step 1: Сборка корпуса (фон, ~1–2 ч)**: `uv run python -m ruencod.data.build > results/build.log 2>&1` (через `run_in_background`). По завершении прочитать `data/corpus/stats.json` и **сообщить пользователю**: объёмы по доменам, `shortfall` (ожидается у Gazeta — тогда её долю добирает новостная часть FineWeb-2: увеличить `share` второго источника и перезапустить `--domains news`).
- [ ] **Step 2: Морфология**: `uv run python -m ruencod.morph.segment_corpus` → сообщить покрытие из `results/morph_coverage.json`.
- [ ] **Step 3: Проверка памяти на самом тяжёлом**: `/usr/bin/time -v uv run python -m ruencod.train.run --algo bpe --vocab 96000 2> results/time_bpe96.txt`. Если `Maximum resident set size` > 9 ГБ или OOM — выставить `train.bpe_max_bytes: 2500000000`, `--force` перезапустить и сообщить пользователю о сокращении.
- [ ] **Step 4: Вся сетка (фон)**: `uv run python -m ruencod.train.run --all > results/train.log 2>&1`. Если HF Unigram падает по памяти даже на 1.5 ГБ — уменьшить `unigram_max_bytes` до 700 МБ; если и это не помогает — реализовать запасной путь из спеки (sentencepiece) отдельной задачей с пометкой в результатах.
- [ ] **Step 5: GPT-цена**: открыть `https://openai.com/api/pricing` (WebFetch), сверить цену GPT-4o за 1M входных токенов; при расхождении обновить `prices.gpt` и `verified_on`.
- [ ] **Step 6: Оценка**: `uv run python -m ruencod.eval.run` (ours + baselines + Claude). Затем commit `results/metrics.csv`, `results/morph_coverage.json`, `configs/experiments.yaml`.
- [ ] **Step 7: Точка решения с пользователем**: построить `2_vocab_curve.png` (`uv run python -m ruencod.viz.charts` упадёт на шагах, требующих `final`, — временно выставить `final.tokenizer` в лучший по fertility 64k), показать кривую и таблицу, предложить V по правилу спеки (§9: прирост < ~2% на удвоение словаря) и спросить подтверждение. Записать `lm.candidates: [bpe-V, unigram-V, morph-bpe-V, o200k]`.

---

### Task 15: Мини-LM, финалист, материалы для поста

- [ ] **Step 1: LM smoke на реальных данных**: `uv run python -m ruencod.lm.run --smoke` → `estimated_full_hours` по кандидатам. Подобрать `lm.train_bytes`, чтобы каждый наш кандидат укладывался в ≤ 3 ч; если o200k > 6 ч — исключить его из `candidates` и сообщить пользователю. Записать в конфиг.
- [ ] **Step 2: Полный LM-прогон (фон, ночь)**: `uv run python -m ruencod.lm.run > results/lm.log 2>&1`.
- [ ] **Step 3: Выбор финалиста** по правилу спеки §9 (минимальный `bpb` на `ru_all`; при разнице < 1% — лучший по fertility). Записать `final.tokenizer`, скопировать `artifacts/tokenizers/{final}` в `artifacts/final/`, commit (папка `artifacts/final/` не в `.gitignore`).
- [ ] **Step 4: Графики и картинки**: `uv run python -m ruencod.viz.charts && uv run python -m ruencod.viz.splits && uv run python -m ruencod.viz.summary`. Просмотреть все PNG через Read; поправить то, что нечитаемо на телефоне.
- [ ] **Step 5: Страница**: выполнить Task 12 Steps 2–5 с финалистом.
- [ ] **Step 6: Черновик поста `post/draft.md`** — только цифры из `results/summary.json`. Структура:
  1. Крючок: одно слово «достопримечательностями» глазами GPT-4o и нашего токенизатора (картинка `split_0.png`).
  2. Что такое токен и почему модель не видит буквы (3–4 предложения).
  3. Почему русский дороже: словари обучены в основном на английском (график `1_fertility.png`).
  4. Эксперимент: 5 ГБ русского, 15 токенизаторов, морфология (кривая `2_vocab_curve.png`).
  5. Результат в цифрах: −X% токенов к GPT-4o, −Y% к Claude, $ за 1000 слов (`4_cost.png`).
  6. Честная оговорка: API так дешевле не станет; экономия — только для модели, обученной на этом токенизаторе. Почему гиганты так не делают (многоязычность, размер словаря).
  7. Морфология: помогла ли и чем (F1, bpb мини-модели — `5_lm_bpb.png`).
  8. Ссылка на интерактивную страницу.
  Длина — под Telegram (≤ 4096 символов в основном тексте; детали — в комментарии/втором посте).
- [ ] **Step 7: Финальная проверка**: `uv run pytest -q` → всё зелёное; commit `post/draft.md`, `results/summary.json`, `results/figures/*.png`, `results/lm_bpb.csv`.

```bash
git add post results/summary.json results/figures results/lm_bpb.csv artifacts/final configs/experiments.yaml
git commit -m "docs: results, figures and Telegram post draft"
```
