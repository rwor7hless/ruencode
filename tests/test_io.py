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


def test_writer_is_atomic(tmp_path):
    import pytest
    p = tmp_path / "x.jsonl.zst"
    with pytest.raises(RuntimeError):
        with ZstJsonlWriter(p) as w:
            w.write({"text": "a"})
            assert not p.exists(), "файл появляется только после успешного закрытия"
            raise RuntimeError("прервано")
    assert not p.exists()
    assert list(tmp_path.iterdir()) == [], "временный файл убран"
