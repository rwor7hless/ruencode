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


def test_load_dotenv_sets_missing_but_keeps_existing(tmp_path, monkeypatch):
    from ruencod.eval.claude_count import load_dotenv
    env = tmp_path / ".env"
    env.write_text("# комментарий\nANTHROPIC_API_KEY=sk-ant-test\nOTHER='x y'\n")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OTHER", "keep")
    load_dotenv(env)
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert os.environ["OTHER"] == "keep"
    load_dotenv(tmp_path / "missing.env")  # нет файла — не падаем
