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
uv sync -p 3.12 --extra cpu      # или --extra cu126 для NVIDIA
uv run python -m ruencod.data.build            # корпус → data/corpus
uv run python -m ruencod.morph.segment_corpus  # морфемная разметка → data/morph_corpus
uv run python -m ruencod.train.run --all       # 15 токенизаторов → artifacts/tokenizers
uv run python -m ruencod.eval.run              # сравнение → results/metrics.csv
uv run python -m ruencod.viz.charts            # графики → results/figures
uv run pytest
```

Настройки — `configs/experiments.yaml`. Дизайн и план — `docs/`.

## Мини-модели на видеокарте (RTX 4060 Ti 8 ГБ)

Проверка «с каким токенизатором модель учится лучше»: одна и та же модель (~85M параметров без
эмбеддингов, 12 слоёв × 768) учится на одинаковых 300 МБ текста с каждым кандидатом
(`bpe-64k`, `unigram-64k`, `morph-bpe-64k`, `o200k`), сравниваем бит на байт на отложенной выборке.

1. Нужны драйвер NVIDIA ≥ 560, `git`, [`uv`](https://docs.astral.sh/uv/). Работает на Linux и Windows.
2. Установка:
   ```bash
   git clone git@github.com:rwor7hless/ruencode.git && cd ruencode
   uv sync -p 3.12 --extra cu126
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
   ```
   Ожидается `True NVIDIA GeForce RTX 4060 Ti`.
3. Корпус: скопировать папку `data/corpus/` (~1.5 ГБ) с машины, где он уже собран, по тому же пути —
   или собрать заново: `uv run python -m ruencod.data.build` (~25 мин, нужен интернет).
   Мини-моделям нужен только `data/corpus/`.
4. Пробный прогон (1 МБ на кандидата) — скорость, пик видеопамяти и прогноз времени:
   ```bash
   uv run python -m ruencod.lm.run --config configs/gpu.yaml --smoke
   ```
   Ориентиры для 4060 Ti: ~14k токенов/с и ~0.9 ч у `bpe-64k`, ~5.7k токенов/с, пик 5.2 ГБ и ~2.8 ч у `o200k`.
   Если `tokens_per_sec` у кого-то в разы ниже — видеопамять переполнена и драйвер Windows молча
   выносит её в общую память: `peak_gpu_gb` должен быть меньше свободной памяти (8 ГБ минус то,
   что занимает рабочий стол, см. `nvidia-smi`). Закрыть браузер/игры или уменьшить `loss_chunk` до 256.
   Если `estimated_full_hours` слишком велико — уменьшить `train_bytes`.
5. Полный прогон и график:
   ```bash
   uv run python -m ruencod.lm.run --config configs/gpu.yaml   # → results/lm_bpb.csv
   uv run python -m ruencod.viz.charts                         # → results/figures/5_lm_bpb.png
   ```
