from typing import List, Optional, Tuple
import os
import logging
import threading
from server.services.llm_adapter import get_llm_adapter
import numpy as np

logger = logging.getLogger("faiss_store")

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

_PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "")


class FaissStore:
    def __init__(self, dim: Optional[int] = None):
        self._configured_dim = dim
        self.dim = dim or 384  # Default for all-MiniLM-L6-v2; auto-detected on first embed
        self._dim_detected = False
        self.index = None
        self.items = {}
        self.id_list = []
        self._hf_embedder = None
        self._lock = threading.Lock()
        if _HAS_FAISS:
            self._init_index(self.dim)
        self._try_load_persisted()

    def _init_index(self, dim: int):
        try:
            self.index = faiss.IndexFlatIP(dim)  # type: ignore[union-attr]
        except Exception:
            try:
                self.index = faiss.IndexFlatL2(dim)  # type: ignore[union-attr]
            except Exception:
                logger.error("Failed to create FAISS index with dim=%d", dim)
                self.index = None

    def _try_load_persisted(self):
        if not _PERSIST_DIR or not _HAS_FAISS:
            return
        index_path = os.path.join(_PERSIST_DIR, "faiss.index")
        meta_path = os.path.join(_PERSIST_DIR, "faiss_meta.npz")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            try:
                self.index = faiss.read_index(index_path)  # type: ignore[union-attr]
                self.dim = self.index.d
                meta = np.load(meta_path, allow_pickle=True)
                self.id_list = list(meta["id_list"])
                self.items = meta["items"].item()
                logger.info("Loaded persisted FAISS index (%d vectors)", self.index.ntotal)
            except Exception as e:
                logger.warning("Failed to load persisted FAISS index: %s", e)

    def _persist(self):
        if not _PERSIST_DIR or not _HAS_FAISS or self.index is None:
            return
        try:
            os.makedirs(_PERSIST_DIR, exist_ok=True)
            index_path = os.path.join(_PERSIST_DIR, "faiss.index")
            meta_path = os.path.join(_PERSIST_DIR, "faiss_meta.npz")
            faiss.write_index(self.index, index_path)  # type: ignore[union-attr]
            np.savez(meta_path, id_list=np.array(self.id_list, dtype=object),
                     items=np.array(self.items, dtype=object))
            logger.info("Persisted FAISS index (%d vectors)", self.index.ntotal)
        except Exception as e:
            logger.warning("Failed to persist FAISS index: %s", e)

    def _normalize_dim(self, vec: List[float]) -> List[float]:
        arr = np.array(vec, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros(self.dim, dtype=np.float32)
        elif arr.size < self.dim:
            arr = np.pad(arr, (0, self.dim - arr.size), mode="constant")
        elif arr.size > self.dim:
            arr = arr[: self.dim]

        norm = np.linalg.norm(arr) + 1e-8
        arr = arr / norm
        return arr.tolist()

    def _embed_hf(self, text: str) -> List[float]:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        if self._hf_embedder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._hf_embedder = SentenceTransformer(model_name)
        vec = self._hf_embedder.encode(text)
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        if not isinstance(vec, list):
            raise RuntimeError("Unexpected embedding output from HF model")
        return vec

    def _embed_adapter(self, text: str) -> List[float]:
        adapter = get_llm_adapter()
        embed_fn = getattr(adapter, "embed", None)
        if not callable(embed_fn):
            raise RuntimeError("LLM adapter does not provide embeddings")
        vec = embed_fn(text)
        if not isinstance(vec, list):
            raise RuntimeError("Adapter embedding returned invalid vector")
        return vec

    def _auto_detect_dim(self, vec: List[float]):
        """Auto-detect and reinitialize FAISS index to match actual embedding dimensionality."""
        if self._dim_detected or self._configured_dim:
            return
        actual_dim = len(vec)
        if actual_dim != self.dim and actual_dim > 0:
            logger.info("Auto-detected embedding dimension: %d (was %d)", actual_dim, self.dim)
            self.dim = actual_dim
            if _HAS_FAISS:
                self._init_index(self.dim)
            self._dim_detected = True
        else:
            self._dim_detected = True

    def _embed(self, text: str) -> List[float]:
        provider = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()

        if provider in ("hf", "huggingface", "sentence-transformers"):
            vec = self._embed_hf(text)
            self._auto_detect_dim(vec)
            return self._normalize_dim(vec)

        if provider in ("adapter", "llm"):
            vec = self._embed_adapter(text)
            self._auto_detect_dim(vec)
            return self._normalize_dim(vec)

        if provider == "auto":
            # Try adapter first, then HF
            try:
                vec = self._embed_adapter(text)
                self._auto_detect_dim(vec)
                return self._normalize_dim(vec)
            except Exception:
                pass
            try:
                vec = self._embed_hf(text)
                self._auto_detect_dim(vec)
                return self._normalize_dim(vec)
            except Exception:
                pass

        raise RuntimeError(
            f"No embedding provider available (provider={provider}). "
            "Set EMBEDDING_PROVIDER to 'hf' or 'adapter' and ensure the model is installed."
        )

    def upsert(self, ids: List[str], texts: List[str]) -> None:
        vecs = [self._embed(t) for t in texts]
        xb = np.array(vecs).astype("float32")
        with self._lock:
            if _HAS_FAISS and self.index is not None:
                try:
                    self.index.add(xb)  # type: ignore[union-attr]
                except Exception as e:
                    logger.error("FAISS index.add failed: %s. Recreating index.", e)
                    try:
                        self._init_index(self.dim)
                        self.index.add(xb)  # type: ignore[union-attr]
                    except Exception as e2:
                        logger.error("FAISS index recreation also failed: %s", e2)
                        raise RuntimeError(f"FAISS upsert failed: {e2}") from e2
                for i, id_ in enumerate(ids):
                    self.items[id_] = texts[i]
                self.id_list.extend(ids)
            else:
                for i, id_ in enumerate(ids):
                    self.items[id_] = {"text": texts[i], "vec": vecs[i]}
        self._persist()

    def query(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        v = np.array([self._embed(text)]).astype("float32")
        with self._lock:
            if _HAS_FAISS and self.index is not None:
                D, indices = self.index.search(v, k)  # type: ignore[union-attr]
                results = []
                for score, idx in zip(D[0], indices[0]):
                    if int(idx) < 0:
                        continue
                    mapped_id = None
                    try:
                        if 0 <= int(idx) < len(self.id_list):
                            mapped_id = self.id_list[int(idx)]
                    except Exception:
                        mapped_id = None
                    results.append(((mapped_id if mapped_id is not None else str(int(idx))), float(score)))
                return results
            # fallback: naive dot-product with stored vecs
            res = []
            for id_, meta in self.items.items():
                vec = np.array(meta["vec"]).astype("float32")
                try:
                    score = float(np.dot(vec, v[0]))
                except Exception:
                    score = -float(np.linalg.norm(vec - v[0]))
                res.append((id_, score))
            res.sort(key=lambda x: x[1], reverse=True)
            return res[:k]

    def warmup(self) -> None:
        provider = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
        if provider in ("hf", "huggingface", "sentence-transformers", "auto"):
            try:
                _ = self._embed_hf("warmup")
            except Exception:
                pass


store = FaissStore()
