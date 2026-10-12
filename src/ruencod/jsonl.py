import io
import json
import os
from pathlib import Path
from typing import Iterator

import zstandard as zstd


class ZstJsonlWriter:
    """Пишет во временный файл; на месте `path` он появляется только после успешного закрытия."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(self.path.name + ".partial")
        self._f = open(self._tmp, "wb")
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

    def __exit__(self, exc_type, *exc):
        self.close()
        if exc_type is None:
            os.replace(self._tmp, self.path)
        else:
            self._tmp.unlink(missing_ok=True)


def iter_jsonl_zst(path: Path) -> Iterator[dict]:
    with open(path, "rb") as f:
        reader = zstd.ZstdDecompressor().stream_reader(f)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)
