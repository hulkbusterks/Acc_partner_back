import os
import pytest

from server.services.faiss_store import FaissStore


def test_embed_hf_raises_when_unavailable(monkeypatch):
    """When HF embedding fails, _embed should propagate the error (no hash fallback)."""
    os.environ["EMBEDDING_PROVIDER"] = "hf"
    store = FaissStore(dim=16)

    def _boom(_text: str):
        raise RuntimeError("hf unavailable")

    monkeypatch.setattr(store, "_embed_hf", _boom)
    with pytest.raises(RuntimeError, match="hf unavailable"):
        store._embed("hello world")

    os.environ.pop("EMBEDDING_PROVIDER", None)


def test_embed_hf_normalizes_dimension(monkeypatch):
    os.environ["EMBEDDING_PROVIDER"] = "hf"
    store = FaissStore(dim=8)

    monkeypatch.setattr(store, "_embed_hf", lambda _text: [1.0, 2.0, 3.0])
    vec = store._embed("abc")
    assert len(vec) == 8

    os.environ.pop("EMBEDDING_PROVIDER", None)


def test_warmup_calls_hf_when_enabled(monkeypatch):
    os.environ["EMBEDDING_PROVIDER"] = "hf"
    store = FaissStore(dim=8)
    called = {"value": False}

    def _mark_and_return(_text: str):
        called["value"] = True
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(store, "_embed_hf", _mark_and_return)
    store.warmup()

    assert called["value"]

    os.environ.pop("EMBEDDING_PROVIDER", None)
