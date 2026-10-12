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
