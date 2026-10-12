# ruEncod

Токенизатор, обученный только на русском языке, и сравнение с токенизаторами популярных LLM.
Цель — основа для собственной модели ~1B и наглядный ответ на вопрос «почему русский в LLM дороже».

## Результаты

Отложенная выборка: 8 доменов по 50 МБ (веб, Википедия, Пикабу, Хабр, Викитека, новости, право,
английский+код), в обучении не участвовала.

- **Токенов на слово** — сколько токенов в среднем уходит на одно русское слово (меньше — лучше).
- **Морфемный F1** — насколько границы токенов совпадают с границами морфем по словарю Тихонова
  (на 10% лемм, отложенных от обучения).

| Токенизатор | Токенов на слово | Морфемный F1 |
|---|---|---|
| **ruEncod bpe-64k** | 1.75 | 0.29 |
| YandexGPT 5 Lite | 1.81 | 0.32 |
| ruGPT-3 | 1.87 | 0.32 |
| GigaChat (open) | 1.89 | 0.38 |
| Vikhr 7B | 1.91 | 0.30 |
| **ruEncod morph-bpe-64k** | 2.08 | 0.44 |
| GPT-4o / GPT-5 (o200k) | 2.16 | 0.41 |
| Gemma 3 | 2.19 | 0.33 |
| **ruEncod unigram-64k** | 2.27 | 0.51 |
| Mistral Nemo | 2.40 | 0.35 |
| Llama 3 | 2.43 | 0.41 |
| Qwen 3 | 2.67 | 0.42 |
| GPT-4 (cl100k) | 3.52 | 0.42 |

На тот же русский текст GPT-4o тратит на 23% токенов больше, чем ruEncod bpe-64k, GPT-4 — вдвое больше.
Цена компромисса: на английском и коде ruEncod примерно на 21% хуже GPT-4o (их 5% в корпусе).

Полные метрики по всем доменам и размерам словаря — `results/metrics.csv`, графики — `results/figures/`.

## Что внутри

- **Корпус** — ~5 ГБ русского текста из 8 доменов, дедупликация, отложенная выборка.
- **15 токенизаторов** — BPE, Unigram и morph-BPE × словари 16k/32k/48k/64k/96k
  (`artifacts/tokenizers/<имя>/tokenizer.json`, формат HF `tokenizers`/`transformers`).
- **morph-BPE** — при обучении запрещены слияния через границы морфем (разбор из словаря Тихонова,
  перенесённый на словоформы через pymorphy3).
- Общая предтокенизация в стиле o200k, цифры поштучно, байтовый fallback, 256 спецтокенов,
  `decode(encode(x)) == x` для любого текста.

## Как использовать

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("artifacts/tokenizers/bpe-64k")
tok.tokenize("Нейросеть не читает буквы — она видит токены.")
```

## Воспроизвести

```bash
uv sync -p 3.12
uv run python -m ruencod.data.build            # корпус → data/corpus
uv run python -m ruencod.morph.segment_corpus  # морфемная разметка → data/morph_corpus
uv run python -m ruencod.train.run --all       # 15 токенизаторов → artifacts/tokenizers
uv run python -m ruencod.eval.run              # сравнение → results/metrics.csv
uv run python -m ruencod.viz.charts            # графики → results/figures
uv run pytest
```

Настройки — `configs/experiments.yaml`. Дизайн и план — `docs/`.
