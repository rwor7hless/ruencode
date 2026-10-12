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
