from typing import List, Optional, Dict, Any, Tuple
from server.db import get_session
from server.models import Book, Topic
from server.services.orchestrator_adapter import get_orchestrator_adapter
from server.services.faiss_store import store as faiss_store
import re


class IngestManager:
    def __init__(self, chunk_size: int = 1200, chunk_tokens: int = 200):
        # chunk_size: fallback character-based chunk size (chars)
        # chunk_tokens: preferred token-based chunk size when tiktoken is available
        # Defaults chosen from local benchmarks in scripts/chunking_benchmark.py
        self.chunk_size = chunk_size
        self.chunk_tokens = chunk_tokens
        self.orch = None

    def _get_orch(self):
        if self.orch is None:
            self.orch = get_orchestrator_adapter()
        return self.orch

    def create_book(self, title: str, authors: Optional[str], raw_text: Optional[str], meta: Optional[dict] = None) -> Book:
        with get_session() as db:
            b = Book(title=title, authors=authors, raw_text=raw_text or "", meta=meta or {})
            db.add(b)
            db.commit()
            db.refresh(b)
            # chunk and upsert into vector store
            chunks, section_ranges = self._chunk_text_with_sections(b.raw_text or "")
            if chunks:
                ids = [f"{b.id}::chunk::{i}" for i in range(len(chunks))]
                faiss_store.upsert(ids, chunks)
            b.meta = {
                **(b.meta or {}),
                "sections": section_ranges,
            }
            db.add(b)
            db.commit()
            db.refresh(b)
            db.expunge(b)
            return b

    def _is_heading(self, line: str) -> bool:
        text = (line or "").strip()
        if not text:
            return False
        if text.startswith("#"):
            return True
        if re.match(r"^(chapter|section|unit|part)\b", text.lower()):
            return True
        words = re.findall(r"[A-Za-z0-9]+", text)
        if 1 <= len(words) <= 10 and text == text.upper():
            return True
        return False

    def _extract_sections(self, text: str) -> List[Dict[str, str]]:
        if not (text or "").strip():
            return []
        sections: List[Dict[str, str]] = []
        current_title = "Overview"
        current_lines: List[str] = []

        for raw in text.splitlines():
            line = (raw or "").rstrip()
            if self._is_heading(line):
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({"title": current_title, "text": body})
                current_title = re.sub(r"^#+\s*", "", line).strip() or "Overview"
                current_lines = []
                continue
            current_lines.append(line)

        tail = "\n".join(current_lines).strip()
        if tail:
            sections.append({"title": current_title, "text": tail})

        if not sections:
            return [{"title": "Overview", "text": text.strip()}]
        return sections

    def _chunk_text_with_sections(self, text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        if not text or text.strip() == "":
            return [], []

        sections = self._extract_sections(text)
        chunks: List[str] = []
        ranges: List[Dict[str, Any]] = []

        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            for section in sections:
                title = str(section.get("title") or "Overview").strip()
                sec_text = str(section.get("text") or "").strip()
                if not sec_text:
                    continue
                start = len(chunks)
                toks = enc.encode(sec_text)
                for i in range(0, len(toks), self.chunk_tokens):
                    part = enc.decode(toks[i:i + self.chunk_tokens]).strip()
                    if not part:
                        continue
                    chunks.append(f"{title}: {part}")
                end = len(chunks)
                if end > start:
                    ranges.append({"title": title, "start": start, "end": end})
            return chunks, ranges
        except Exception:
            for section in sections:
                title = str(section.get("title") or "Overview").strip()
                sec_text = str(section.get("text") or "").strip()
                if not sec_text:
                    continue
                start = len(chunks)

                sentences = []
                for part in sec_text.split('\n\n'):
                    part = part.strip()
                    if not part:
                        continue
                    pieces = [s.strip() for s in part.replace('\n', ' ').split('. ') if s.strip()]
                    for s in pieces:
                        if not s.endswith('.'):
                            s = s + '.'
                        sentences.append(s)

                cur = ''
                for s in sentences:
                    if len(cur) + len(s) <= self.chunk_size or cur == '':
                        cur = (cur + ' ' + s).strip()
                    else:
                        chunks.append(f"{title}: {cur}".strip())
                        cur = s
                if cur:
                    chunks.append(f"{title}: {cur}".strip())

                end = len(chunks)
                if end > start:
                    ranges.append({"title": title, "start": start, "end": end})

            return chunks, ranges

    def chunk_text(self, text: str) -> List[str]:
        chunks, _ = self._chunk_text_with_sections(text)
        return chunks

    def create_topics_from_book(self, book_id: str, mode: str = "rule", pages: Optional[List[int]] = None) -> List[Topic]:
        with get_session() as db:
            book = db.get(Book, book_id)
            if not book:
                raise ValueError("book not found")
            chunks = self.chunk_text(book.raw_text or "")
            section_ranges = (book.meta or {}).get("sections", []) if isinstance(book.meta, dict) else []
            topics = []
            if mode == "rule":
                if section_ranges:
                    for section in section_ranges:
                        start = int(section.get("start", 0))
                        end = int(section.get("end", start))
                        if end <= start:
                            continue
                        title = str(section.get("title") or "Overview").strip()
                        t = Topic(book_id=book_id, title=title, start_page=None, end_page=None, source_chunks={"indexes": list(range(start, end))})
                        db.add(t)
                        db.commit()
                        db.refresh(t)
                        db.expunge(t)
                        topics.append(t)
                else:
                    group = max(1, len(chunks) // 5)
                    for i in range(0, len(chunks), group):
                        t = Topic(book_id=book_id, title=f"Topic {i//group + 1}", start_page=None, end_page=None, source_chunks={"indexes": list(range(i, min(i+group, len(chunks))))})
                        db.add(t)
                        db.commit()
                        db.refresh(t)
                        db.expunge(t)
                        topics.append(t)
            else:
                sections_payload = []
                for section in section_ranges:
                    title = str(section.get("title") or "Overview").strip()
                    start = int(section.get("start", 0))
                    end = int(section.get("end", start))
                    chunk_ids = [f"{book_id}::chunk::{i}" for i in range(start, max(start, end))]
                    summary_text = " ".join(chunks[start:min(end, start + 2)])
                    sections_payload.append({
                        "title": title,
                        "summary": summary_text,
                        "chunk_ids": chunk_ids,
                    })

                seeds = [book.title] + (chunks[:3] if chunks else [])
                candidate_chunk_ids = set()
                for s in seeds:
                    res = faiss_store.query(s, k=5)
                    for cid, _ in res:
                        if isinstance(cid, str) and cid.startswith(f"{book_id}::chunk::"):
                            candidate_chunk_ids.add(cid)

                candidate_texts = []
                for cid in candidate_chunk_ids:
                    item = faiss_store.items.get(cid)
                    if isinstance(item, dict):
                        candidate_texts.append({"id": cid, "text": item.get("text")})
                    else:
                        candidate_texts.append({"id": cid, "text": item})

                if not candidate_texts and chunks:
                    local_ids = [f"{book_id}::chunk::{i}" for i in range(len(chunks))]
                    missing = [i for i in local_ids if i not in faiss_store.items]
                    if missing:
                        miss_idxs = [int(i.split("::")[-1]) for i in missing]
                        miss_texts = [chunks[idx] for idx in miss_idxs]
                        faiss_store.upsert(missing, miss_texts)

                    for i, txt in enumerate(chunks[:5]):
                        cid = f"{book_id}::chunk::{i}"
                        candidate_texts.append({"id": cid, "text": txt})

                res = self._get_orch().run_rag_pipeline(book_id, {
                    "action": "propose_topics",
                    "candidates": candidate_texts,
                    "sections": sections_payload,
                })
                proposed = res.get("topics", []) if isinstance(res, dict) else []
                if not proposed:
                    proposed = [{"title": "Overview", "source_chunks": {"ids": [f"{book_id}::chunk::0"] if chunks else []}}]

                for idx, tt in enumerate(proposed):
                    t = Topic(book_id=book_id, title=tt.get("title", f"Topic {idx+1}"), source_chunks=tt.get("source_chunks", {}))
                    db.add(t)
                    db.commit()
                    db.refresh(t)
                    db.expunge(t)
                    topics.append(t)
            return topics


ingest_manager = IngestManager()
