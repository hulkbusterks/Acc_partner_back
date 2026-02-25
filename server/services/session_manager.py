from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import random
import logging

from sqlmodel import select

from server.db import get_session
from server.models import Session, PromptEvent, CachedQuestion, QuestionUsage, LeaderboardAggregate, LeaderboardEntry, Topic, Book
from server.services.orchestrator_adapter import get_orchestrator_adapter
from server.services.mcq_validator import try_normalize
from server.services.faiss_store import store as faiss_store

logger = logging.getLogger("session_manager")


class SessionManager:
    def __init__(self):
        self.orch = None

    def _get_orch(self):
        if self.orch is None:
            self.orch = get_orchestrator_adapter()
        return self.orch

    def create_session(self, user_id: str, topic_id: str, requested_minutes: int = 30, tone: str = "neutral") -> Session:
        with get_session() as db:
            sess = Session(user_id=user_id, topic_id=topic_id, started_at=None, tone=tone)
            db.add(sess)
            db.commit()
            db.refresh(sess)
            db.expunge(sess)
            return sess

    def start_session(self, session_id: str, duration_minutes: int = 30) -> Session:
        with get_session() as db:
            sess = db.get(Session, session_id)
            if not sess:
                raise ValueError("session not found")
            sess.started_at = datetime.now(timezone.utc)
            sess.duration_sec = duration_minutes * 60
            db.add(sess)
            db.commit()
            db.refresh(sess)
            db.expunge(sess)
            return sess

    def _is_session_expired(self, sess: Session) -> bool:
        if not sess.started_at or not sess.duration_sec:
            return False
        started = sess.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        end = started + timedelta(seconds=sess.duration_sec)
        return datetime.now(timezone.utc) >= end

    def generate_prompts(self, session_id: str, n: int = 5) -> List[Dict[str, Any]]:
        # Look up session + topic to pass context to the orchestrator
        with get_session() as db:
            sess = db.get(Session, session_id)
            if not sess:
                raise ValueError("session not found")
            if not sess.started_at:
                raise ValueError("session not started")
            if self._is_session_expired(sess):
                raise ValueError("session expired")

            topic = db.get(Topic, sess.topic_id)
            seed = topic.title if topic else ""
            source_chunk_indexes = []
            resolved_contexts: list[str] = []
            if topic and isinstance(topic.source_chunks, dict):
                source_chunk_indexes = topic.source_chunks.get("indexes", [])
                source_chunk_ids = topic.source_chunks.get("ids", [])

                # Resolve chunk IDs from FAISS store
                for cid in source_chunk_ids:
                    item = faiss_store.items.get(cid)
                    if isinstance(item, dict):
                        txt = str(item.get("text") or "")
                    else:
                        txt = str(item or "")
                    if txt.strip():
                        resolved_contexts.append(txt)

                # If FAISS had nothing (server restarted / cold store), resolve
                # chunk IDs back to actual book content via the DB.
                if not resolved_contexts and source_chunk_ids:
                    book = db.get(Book, topic.book_id) if topic.book_id else None
                    if book and book.raw_text:
                        from server.services.ingest_manager import ingest_manager
                        chunks = ingest_manager.chunk_text(book.raw_text)
                        # source_chunk_ids look like "bookid::chunk::3" — extract index
                        for cid in source_chunk_ids:
                            try:
                                idx = int(str(cid).split("::")[-1])
                                if 0 <= idx < len(chunks) and chunks[idx].strip():
                                    resolved_contexts.append(chunks[idx])
                            except (ValueError, IndexError):
                                pass
                        # Also re-populate FAISS so subsequent calls don't re-chunk
                        if chunks and not faiss_store.items:
                            try:
                                all_ids = [f"{book.id}::chunk::{i}" for i in range(len(chunks))]
                                faiss_store.upsert(all_ids, chunks)
                            except Exception:
                                pass  # non-critical; best-effort warm-up

                # Resolve chunk indexes from the book's raw_text
                if source_chunk_indexes and not resolved_contexts:
                    book = db.get(Book, topic.book_id) if topic.book_id else None
                    if book and book.raw_text:
                        from server.services.ingest_manager import ingest_manager
                        chunks = ingest_manager.chunk_text(book.raw_text)
                        for idx in source_chunk_indexes:
                            if isinstance(idx, int) and 0 <= idx < len(chunks):
                                if chunks[idx].strip():
                                    resolved_contexts.append(chunks[idx])

                # Last resort: if we STILL have no contexts, just use the book's
                # raw text directly (truncated) so the LLM has something real.
                if not resolved_contexts:
                    book = db.get(Book, topic.book_id) if topic.book_id else None
                    if book and (book.raw_text or "").strip():
                        # Take up to ~4000 chars of the raw text
                        raw = (book.raw_text or "").strip()[:4000]
                        resolved_contexts.append(raw)

        logger.info(
            "generate_prompts: topic=%s, seed=%s, resolved_contexts=%d, total_chars=%d",
            topic.title if topic else "none", seed, len(resolved_contexts),
            sum(len(c) for c in resolved_contexts),
        )

        orch = self._get_orch()
        out = orch.run_rag_pipeline(session_id, {
            "action": "generate_mcqs",
            "n": n,
            "seed": seed,
            "topic_source_chunks": source_chunk_indexes,
            "contexts": resolved_contexts if resolved_contexts else None,
        })
        with get_session() as db:
            saved = []
            try:
                for q in out.get("questions", []):
                    question_json = q.get("question_json") if isinstance(q.get("question_json"), dict) else q

                    # Normalize and validate MCQ via the shared validator
                    normalized = try_normalize(question_json)
                    if normalized:
                        question_json = normalized

                    meta_variants = {
                        "verified": q.get("verified", False),
                        "source_chunks": q.get("source_chunks", []),
                        "deterministic_supported": q.get("deterministic_supported", False),
                        "evidence": q.get("evidence", ""),
                    }
                    cq = CachedQuestion(chunks_hash=q.get("chunks_hash", ""), question_json=question_json, variants=meta_variants)
                    db.add(cq)
                    db.flush()
                    pe = PromptEvent(
                        session_id=session_id,
                        prompt_text=(question_json.get("question") if isinstance(question_json, dict) else q.get("prompt", q.get("question", ""))),
                        prompt_question_id=cq.id,
                    )
                    db.add(pe)
                    # Return the normalized question with metadata so clients get clean MCQs
                    out_q = dict(q)
                    out_q["question_json"] = question_json
                    out_q["prompt_id"] = pe.id
                    saved.append(out_q)
                db.commit()
            except Exception:
                db.rollback()
                raise
            return saved

    def get_next_prompt(self, session_id: str) -> Optional[Dict[str, Any]]:
        with get_session() as db:
            sess = db.get(Session, session_id)
            if not sess:
                raise ValueError("session not found")
            if sess.ended_at:
                return None
            if self._is_session_expired(sess):
                return None

            # Get all unanswered prompts for this session
            stmt = (
                select(PromptEvent)
                .where((PromptEvent.session_id == session_id) & (PromptEvent.response_text == None))
                .order_by(PromptEvent.created_at)
            )
            unanswered = db.exec(stmt).all()
            if not unanswered:
                return None

            # Avoid repeating questions the user has already seen.
            # QuestionUsage rows are written on submit, but we also track
            # which prompt_ids we've already served via a lightweight
            # session-scoped set stored on this manager instance.
            shown = getattr(self, "_shown_prompts", {})
            session_shown = shown.get(session_id, set())

            # Prefer prompts NOT yet shown; fall back to any unanswered
            unseen = [p for p in unanswered if p.id not in session_shown]
            choice = random.choice(unseen) if unseen else random.choice(unanswered)

            # Mark this prompt as shown
            session_shown.add(choice.id)
            if not hasattr(self, "_shown_prompts"):
                self._shown_prompts = {}
            self._shown_prompts[session_id] = session_shown

            # Build full MCQ payload by joining to CachedQuestion
            result: Dict[str, Any] = {
                "prompt_id": choice.id,
                "prompt_text": choice.prompt_text,
                "remaining": len(unanswered),
            }
            if choice.prompt_question_id:
                cq = db.get(CachedQuestion, choice.prompt_question_id)
                if cq and isinstance(cq.question_json, dict):
                    qj = cq.question_json
                    result["question"] = qj.get("question", choice.prompt_text)
                    result["choices"] = qj.get("choices", [])
                    # Don't send correct_index — the client shouldn't know the answer yet
            return result

    def submit_answer(self, session_id: str, prompt_id: str, answer: str, user_id: str, reject: bool = False) -> Dict[str, Any]:
        with get_session() as db:
            pe = db.get(PromptEvent, prompt_id)
            if not pe:
                raise ValueError("prompt not found")
            # Validate prompt belongs to this session
            if pe.session_id != session_id:
                raise ValueError("prompt does not belong to this session")
            # Guard: prompt already answered
            if pe.response_text is not None:
                raise ValueError("prompt already answered")

            # Guard: session must be started and not expired
            sess = db.get(Session, session_id)
            if not sess:
                raise ValueError("session not found")
            if not sess.started_at:
                raise ValueError("session not started")
            if self._is_session_expired(sess):
                raise ValueError("session expired")

            if reject:
                pe.response_text = "__rejected__"
                pe.correct = False
                sess.reject_count = (sess.reject_count or 0) + 1
                db.add(pe)
                db.add(sess)
                db.commit()
                return {"rejected": True, "session_rejects": sess.reject_count}

            pe.response_text = answer
            correct = False
            cq = None
            if getattr(pe, "prompt_question_id", None):
                try:
                    cq = db.get(CachedQuestion, pe.prompt_question_id)
                except Exception:
                    cq = None

            if cq and isinstance(cq.question_json, dict):
                qjson = cq.question_json
                correct_index = qjson.get("correct_index")
                choices = qjson.get("choices") if isinstance(qjson.get("choices"), list) else None
                parsed_idx = None
                try:
                    parsed_idx = int(answer.strip())
                except Exception:
                    parsed_idx = None

                if parsed_idx is not None and choices:
                    # Answers are always 0-based index
                    if isinstance(correct_index, int) and parsed_idx == correct_index:
                        correct = True
                if not correct and choices and isinstance(correct_index, int) and 0 <= correct_index < len(choices):
                    try:
                        correct_text = choices[correct_index].strip().lower()
                        if answer.strip().lower() == correct_text:
                            correct = True
                    except Exception:
                        pass

            pe.correct = correct
            db.add(pe)

            usage_qid = cq.id if cq else prompt_id
            qu = QuestionUsage(
                question_id=usage_qid, session_id=session_id, user_id=user_id,
                shown_at=pe.created_at, answered_at=datetime.now(timezone.utc),
                answer_given={"text": answer}, is_correct=correct,
            )
            db.add(qu)

            if correct:
                sess.score = (sess.score or 0) + 1
            else:
                sess.failure_count = (sess.failure_count or 0) + 1
            db.add(sess)
            db.commit()

            # Build detailed response with correct answer + per-option reasoning
            result: Dict[str, Any] = {"correct": correct, "session_score": sess.score, "failures": sess.failure_count}

            if cq and isinstance(cq.question_json, dict):
                qjson = cq.question_json
                ci = qjson.get("correct_index", 0)
                ch = qjson.get("choices", [])
                result["correct_index"] = ci
                result["correct_answer"] = ch[ci] if isinstance(ci, int) and 0 <= ci < len(ch) else None
                result["explanation"] = qjson.get("explanation", "")
                # Per-option breakdown: tone-aware reasoning
                tone = sess.tone or "neutral"
                opt_expl = qjson.get("option_explanations", [])
                option_notes = []
                for idx, opt in enumerate(ch):
                    # Use LLM-generated per-option explanation if available
                    base_reason = opt_expl[idx] if isinstance(opt_expl, list) and idx < len(opt_expl) else ""

                    if idx == ci:
                        fallback = qjson.get("explanation", "This is the correct answer.")
                        reason_text = base_reason or fallback
                        if tone == "mean":
                            reason = f"Finally, something obvious. {reason_text}"
                        else:
                            reason = f"Correct! {reason_text}"
                        option_notes.append({"index": idx, "text": opt, "correct": True, "reason": reason})
                    else:
                        if base_reason:
                            if tone == "mean":
                                reason = f"Wrong. How did you even consider this? {base_reason}"
                            else:
                                reason = f"Not quite. {base_reason}"
                        else:
                            if tone == "mean":
                                reason = f"'{opt}'? Really? That's embarrassingly wrong."
                            else:
                                reason = f"'{opt}' is incorrect. The correct answer is '{ch[ci]}' — {qjson.get('explanation', 'review the material for details.')}"  
                        option_notes.append({"index": idx, "text": opt, "correct": False, "reason": reason})
                result["options"] = option_notes

            threshold = 3
            total_neg = (sess.failure_count or 0) + (sess.reject_count or 0)
            if total_neg >= threshold and (sess.tone or "neutral") == "mean":
                mean_comments = [
                    "Wow, you're really speedrunning failure here.",
                    "At this point the book is begging you to read it.",
                    "I've seen better performance from a random number generator.",
                    "Are you even trying or just clicking randomly?",
                    "Your score is so low it's practically a negative achievement.",
                ]
                import random as _rng
                result["mean_comment"] = _rng.choice(mean_comments)

            # Check if all prompts are now answered → auto-end the session
            remaining_stmt = (
                select(PromptEvent)
                .where((PromptEvent.session_id == session_id) & (PromptEvent.response_text == None))
            )
            remaining = db.exec(remaining_stmt).all()
            result["remaining"] = len(remaining)
            if len(remaining) == 0 and not sess.ended_at:
                result["session_complete"] = True
            else:
                result["session_complete"] = False

            return result

    def end_session(self, session_id: str) -> Tuple[Session, Any]:
        with get_session() as db:
            sess = db.get(Session, session_id)
            if not sess:
                raise ValueError("session not found")
            if sess.ended_at:
                raise ValueError("session already ended")
            sess.ended_at = datetime.now(timezone.utc)
            db.add(sess)
            db.commit()
            db.refresh(sess)

            try:
                from server.services.leaderboard_service import upsert_aggregate
                final_score = sess.score or 0
                agg = upsert_aggregate(sess.user_id, float(final_score))
            except Exception:
                agg = None

            # Audit trail: record per-session LeaderboardEntry
            try:
                topic = db.get(Topic, sess.topic_id)
                entry = LeaderboardEntry(
                    user_id=sess.user_id,
                    session_id=sess.id,
                    topic_id=sess.topic_id,
                    book_id=topic.book_id if topic else None,
                    score=float(sess.score or 0),
                )
                db.add(entry)
                db.commit()
            except Exception:
                logger.warning("Failed to write LeaderboardEntry audit row", exc_info=True)

            db.expunge(sess)
            return sess, agg


manager = SessionManager()
