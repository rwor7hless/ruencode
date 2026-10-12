import hashlib
import json
import os
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Минимальный .env: KEY=VALUE построчно; уже заданные переменные окружения не трогаем."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class ClaudeCounter:
    """Точный подсчёт токенов Claude через messages.count_tokens с дисковым кэшем."""

    def __init__(self, model: str, cache_path: Path, client=None):
        if client is None:
            import anthropic

            from ruencod.config import ROOT
            load_dotenv(ROOT / ".env")
            client = anthropic.Anthropic(max_retries=8)
        self.model, self.cache_path, self.client = model, Path(cache_path), client
        self.cache: dict[str, int] = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
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
        tmp.write_text(json.dumps(self.cache), encoding="utf-8")
        os.replace(tmp, self.cache_path)
