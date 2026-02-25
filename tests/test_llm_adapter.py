import os

from server.services.llm_adapter import OpenAICompatibleAdapter


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


class _FakeStreamResp:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line


def test_openai_compatible_extracts_reasoning_when_content_empty(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning": "usable reasoning text",
                }
            }
        ]
    }

    monkeypatch.setattr(
        "server.services.llm_adapter.requests.post",
        lambda *args, **kwargs: _FakeResp(payload),
    )

    adapter = OpenAICompatibleAdapter(endpoint="https://example.com", key="k", model="m")
    out = adapter.generate("hi")
    assert out["text"] == "usable reasoning text"


def test_openai_compatible_extracts_content_text_list(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "output_text", "text": "alpha"},
                        {"type": "output_text", "text": "beta"},
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(
        "server.services.llm_adapter.requests.post",
        lambda *args, **kwargs: _FakeResp(payload),
    )

    adapter = OpenAICompatibleAdapter(endpoint="https://example.com", key="k", model="m")
    out = adapter.generate("hi")
    assert out["text"] == "alpha beta"


def test_openai_compatible_streaming_assembles_delta_text(monkeypatch):
    os.environ["LLM_STREAM"] = "true"

    lines = [
        'data: {"choices":[{"delta":{"content":"hello "}}]}',
        'data: {"choices":[{"delta":{"content":"world"}}]}',
        'data: [DONE]',
    ]

    monkeypatch.setattr(
        "server.services.llm_adapter.requests.post",
        lambda *args, **kwargs: _FakeStreamResp(lines),
    )

    adapter = OpenAICompatibleAdapter(endpoint="https://example.com", key="k", model="m")
    out = adapter.generate("hi")
    assert out["text"] == "hello world"

    os.environ.pop("LLM_STREAM", None)
