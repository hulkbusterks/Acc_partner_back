from server.services.ingest_manager import IngestManager
from server.services.faiss_store import store as faiss_store
from server.services.faiss_store import FaissStore
import numpy as np


def test_token_chunking_basic():
    text = "This is a sentence. " * 50
    ing = IngestManager(chunk_tokens=10, chunk_size=200)
    chunks = ing.chunk_text(text)
    assert isinstance(chunks, list)
    assert len(chunks) >= 2


def test_faiss_upsert_and_query():
    # prepare deterministic small texts
    texts = ["apple orange banana", "python testing code", "lorem ipsum dolor sit amet"]
    ids = [f"test::{i}" for i in range(len(texts))]
    # clear any prior items for deterministic behavior in PoC store
    faiss_store.items.clear()
    faiss_store.id_list.clear()
    # upsert
    faiss_store.upsert(ids, texts)
    # query for a related term
    res = faiss_store.query("python code", k=3)
    assert isinstance(res, list)
    assert len(res) >= 1
    # returned ids should be keys in the store (fallback or faiss)
    for rid, score in res:
        assert rid is not None
        # if id is numeric index cast, allow that; otherwise ensure it's in items
        if rid.isdigit():
            continue
        assert rid in faiss_store.items


def test_faiss_query_ignores_negative_indices(monkeypatch):
    class _FakeIndex:
        def search(self, _v, _k):
            return np.array([[0.0, 0.0]], dtype="float32"), np.array([[-1, -1]], dtype="int64")

    local = FaissStore(dim=16)
    local.index = _FakeIndex()
    local.id_list = ["id-1"]
    local.items = {"id-1": {"text": "hello", "vec": [0.0] * 16}}

    res = local.query("hello", k=2)
    assert res == []
