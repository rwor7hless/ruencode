import tarfile

from ruencod.corpus import domain_file, domain_shares, iter_domain, iter_mix
from ruencod.jsonl import ZstJsonlWriter
from tests.conftest import make_corpus


def _cfg(base):
    return {"paths": {"data": str(base)},
            "corpus": {"domains": {"a": {"share": 0.7}, "b": {"share": 0.3, "russian": False}}},
            "lm": {"train_bytes": 20_000, "eval_bytes_per_domain": 3_000}}


def test_export_keeps_exactly_what_the_lm_reads(tmp_path):
    from ruencod.lm.export_data import export_slice
    src = tmp_path / "home"
    for i, d in enumerate(("a", "b")):
        for split in ("train", "heldout"):
            with ZstJsonlWriter(domain_file(src / "corpus", split, d)) as w:
                for t in make_corpus(400, seed=i * 10 + len(split)):
                    w.write({"text": t, "domain": d})
    cfg = _cfg(src)
    tar_path = export_slice(cfg, tmp_path / "slice.tar")
    cloud = tmp_path / "cloud"
    with tarfile.open(tar_path) as tar:
        tar.extractall(cloud, filter="data")
    lm, shares = cfg["lm"], domain_shares(cfg)
    assert list(iter_mix(cloud / "corpus", "train", shares, lm["train_bytes"])) == \
        list(iter_mix(src / "corpus", "train", shares, lm["train_bytes"]))
    for d in shares:
        assert list(iter_domain(cloud / "corpus", "heldout", d, lm["eval_bytes_per_domain"])) == \
            list(iter_domain(src / "corpus", "heldout", d, lm["eval_bytes_per_domain"]))
    assert (cloud / "corpus" / "stats.json").exists()
    # в архиве только нужный префикс, а не весь корпус
    assert len(list(iter_domain(cloud / "corpus", "train", "a"))) < len(list(iter_domain(src / "corpus", "train", "a")))
