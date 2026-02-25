"""Retrieval-backed graph for local orchestration testing.

Pipeline shape (LangGraph-like, dependency-light):
    retrieve -> rerank -> draft_mcq -> validate_evidence -> repair(optional)

The module stays compatible with `LangGraphAdapter` by exposing `load(path)`
that returns an object with `run(block)`.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import json
import logging
import time
import random as _random
from time import perf_counter

from server.services.faiss_store import store as faiss_store
from server.services.evidence_check import deterministic_evidence_check
from server.services.llm_adapter import get_llm_adapter
from server.services.observability import observability


logger = logging.getLogger("orchestrator.graph")


def _safe_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(low, min(high, parsed))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).lower() in ("1", "true", "yes", "on")


def _clean_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", (text or "").lower())]


def _is_low_quality_phrase(phrase: str) -> bool:
    words = _clean_words(phrase)
    if len(words) < 2:
        return True
    banned = {
        "analysis", "continue", "exactly", "one", "thing", "stuff",
        "option", "answer", "correct", "incorrect", "none",
        "above", "below", "true", "false", "distractor",
        "unrelated", "unsupported", "summary", "inference",
    }
    return any(w in banned for w in words)


def _sanitize_llm_phrase(raw: str, context: str) -> str:
    words = _clean_words(raw)
    if not words:
        return ""

    # Remove common prompt-echo / meta prefixes.
    prefix_stop = {
        "we", "need", "to", "extract", "short", "factual", "phrase", "from", "this",
        "context", "return", "only", "the", "answer", "a", "an",
    }
    i = 0
    while i < len(words) and words[i] in prefix_stop:
        i += 1
    words = words[i:] if i < len(words) else words
    if not words:
        return ""

    ctx_words = set(_clean_words(context))
    kept = [w for w in words if w in ctx_words]
    if len(kept) >= 2:
        words = kept

    phrase = " ".join(words[:4]).strip()
    return phrase


def _pick_phrase_from_context(context: str) -> str:
    words = _clean_words(context)
    if not words:
        return "key concept"

    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "your", "you",
        "are", "was", "were", "have", "has", "had", "but", "not", "can", "will",
        "what", "when", "where", "which", "who", "why", "how", "is", "it",
        "a", "an", "of", "to", "in", "on", "as", "at", "by",
    }
    filtered = [w for w in words if w not in stop]
    if len(filtered) >= 3:
        return " ".join(filtered[:3])
    if len(filtered) >= 2:
        return " ".join(filtered[:2])
    if filtered:
        return filtered[0]
    return "key concept"


def _fallback_distractors(answer: str) -> List[str]:
    # Pool of distractor templates — varied enough to avoid obvious patterns
    pools = [
        [
            "unrelated summary",
            "incorrect inference",
            "unsupported claim",
            "opposite statement",
            "missing detail",
        ],
        [
            "common misconception",
            "partially true claim",
            "overgeneralized statement",
            "outdated information",
            "reversed causation",
        ],
    ]
    import random as _rng
    pool = _rng.choice(pools)
    out = []
    for item in pool:
        if item.strip().lower() != answer.strip().lower():
            out.append(item)
        if len(out) == 3:
            break
    while len(out) < 3:
        out.append(f"distractor {len(out)+1}")
    return out


def _dedupe_choices(choices: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for c in choices:
        txt = str(c).strip()
        key = txt.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(txt)
    return out


def _context_for_chunk(chunk_id: str) -> str:
    item = faiss_store.items.get(chunk_id)
    if isinstance(item, dict):
        return str(item.get("text") or "")
    return str(item or "")


@dataclass
class GraphConfig:
    max_retries: int = 2
    max_mcqs: int = 8
    max_topics: int = 5
    min_context_words: int = 4
    top_k_multiplier: int = 3
    draft_mode: str = "llm"
    orchestrator_model: str = "default"
    enable_moderation: bool = False
    max_context_chars: int = 4000  # truncate context sent to LLM

    @classmethod
    def from_env(cls) -> "GraphConfig":
        return cls(
            max_retries=_safe_int(os.getenv("ORCHESTRATOR_MAX_RETRIES", 2), 2, 0, 5),
            max_mcqs=_safe_int(os.getenv("ORCHESTRATOR_MAX_MCQS", 8), 8, 1, 20),
            max_topics=_safe_int(os.getenv("ORCHESTRATOR_MAX_TOPICS", 5), 5, 1, 20),
            min_context_words=_safe_int(os.getenv("ORCHESTRATOR_MIN_CONTEXT_WORDS", 4), 4, 1, 50),
            top_k_multiplier=_safe_int(os.getenv("ORCHESTRATOR_TOPK_MULTIPLIER", 3), 3, 1, 10),
            draft_mode=(os.getenv("ORCHESTRATOR_DRAFT_MODE", "llm").strip().lower() or "llm"),
            orchestrator_model=(os.getenv("ORCHESTRATOR_MODEL", "default").strip() or "default"),
            enable_moderation=_env_bool("ORCHESTRATOR_ENABLE_MODERATION", False),
            max_context_chars=_safe_int(os.getenv("ORCHESTRATOR_MAX_CONTEXT_CHARS", 4000), 4000, 500, 20000),
        )


class RetrievalBackedGraph:
    def __init__(self, graph_path: str | None = None):
        self.graph_path = graph_path
        self.config = GraphConfig.from_env()
        self._llm_adapter = None

    def _verify_enabled(self) -> bool:
        return _env_bool("ORCHESTRATOR_VERIFY", False)

    def _get_llm(self):
        if self._llm_adapter is None:
            self._llm_adapter = get_llm_adapter()
        return self._llm_adapter

    def _build_topic_title(self, text: str, idx: int) -> str:
        words = _clean_words(text)
        if not words:
            return f"Topic {idx}"
        title_words = words[:4]
        return " ".join([w.capitalize() for w in title_words])

    def _build_topic_title_llm(self, text: str, idx: int) -> str:
        if self.config.draft_mode not in ("llm", "llm-assist", "model"):
            return self._build_topic_title(text, idx)
        try:
            prompt = (
                "Create a concise topic title (2 to 6 words) from the context. "
                "Return title only. Context: " + (text or "")
            )
            out = self._get_llm().generate(prompt=prompt, max_tokens=48, temperature=0.1,
                system="You are an educational quiz generator. Generate content only from the provided context.")
            raw = str(out.get("text") or "").strip()
            words = _clean_words(raw)
            if 2 <= len(words) <= 8:
                return " ".join([w.capitalize() for w in words[:6]])
        except Exception:
            pass
        return self._build_topic_title(text, idx)

    def _retrieve(self, seed: str, n: int) -> List[Tuple[str, float]]:
        top_k = max(4, n * self.config.top_k_multiplier)
        return faiss_store.query(seed or "study", k=top_k)

    def _rerank(self, hits: List[Tuple[str, float]], n: int) -> List[str]:
        # Keep this deterministic for local tests: score-first, then id stable order
        ordered = sorted(hits, key=lambda item: (-float(item[1]), str(item[0])))
        chunk_ids = [chunk_id for chunk_id, _ in ordered if chunk_id]
        if chunk_ids:
            return chunk_ids[: max(1, n)]
        return list(faiss_store.items.keys())[: max(1, n)]

    def _build_question_deterministic(self, context: str, i: int) -> Dict[str, Any]:
        """Deterministic fallback — used when LLM is unavailable or fails."""
        answer_phrase = _pick_phrase_from_context(context)
        distractors = _fallback_distractors(answer_phrase)
        choices = [answer_phrase, distractors[0], distractors[1], distractors[2]]
        correct_index = i % 4
        if correct_index != 0:
            choices[0], choices[correct_index] = choices[correct_index], choices[0]
        choices = _dedupe_choices(choices)
        while len(choices) < 4:
            choices.append(f"distractor {len(choices)+1}")
        choices = choices[:4]
        try:
            correct_index = next(
                idx for idx, c in enumerate(choices)
                if c.strip().lower() == answer_phrase.strip().lower()
            )
        except StopIteration:
            choices[0] = answer_phrase
            correct_index = 0
        return {
            "question": "Which phrase is directly supported by the source context?",
            "choices": choices,
            "correct_index": correct_index,
            "explanation": "The correct option appears in the retrieved source context.",
            "draft_source": "deterministic",
            "llm_attempted": False,
            "llm_preview": "",
        }

    def _parse_llm_mcq(self, raw: str) -> Optional[Dict[str, Any]]:
        """Try to parse a JSON MCQ from the LLM's raw text output."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        obj = None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            pass
        if obj is None:
            # Try to find the outermost JSON object (supports nested braces/arrays)
            depth = 0
            start_idx = None
            for i, ch in enumerate(text):
                if ch == "{":
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        try:
                            obj = json.loads(text[start_idx : i + 1])
                        except json.JSONDecodeError:
                            obj = None
                        break
        if not isinstance(obj, dict):
            return None
        q = str(obj.get("question") or "").strip()
        choices = obj.get("choices") or obj.get("options") or []
        correct_index = obj.get("correct_index") if obj.get("correct_index") is not None else obj.get("answer")
        explanation = str(obj.get("explanation") or "").strip()
        if not q or not isinstance(choices, list) or len(choices) < 4:
            return None
        choices = [str(c).strip() for c in choices[:4]]
        if isinstance(correct_index, int) and 0 <= correct_index < len(choices):
            pass
        elif isinstance(correct_index, str):
            # Try matching answer text to a choice
            try:
                correct_index = next(i for i, c in enumerate(choices) if c.lower() == correct_index.lower())
            except StopIteration:
                correct_index = 0
        else:
            correct_index = 0
        result: Dict[str, Any] = {
            "question": q,
            "choices": choices,
            "correct_index": int(correct_index),
            "explanation": explanation or "Based on the provided source material.",
        }
        # Include per-option explanations if the LLM provided them
        opt_expl = obj.get("option_explanations")
        if isinstance(opt_expl, list) and len(opt_expl) >= len(choices):
            result["option_explanations"] = [str(e).strip() for e in opt_expl[:len(choices)]]
        return result

    def _truncate_context(self, context: str) -> str:
        """Truncate context to max_context_chars to avoid blowing up the LLM prompt."""
        if len(context) <= self.config.max_context_chars:
            return context
        return context[: self.config.max_context_chars] + "\n[...truncated]"

    def _build_question(self, context: str, i: int) -> Dict[str, Any]:
        draft_source = "deterministic"
        llm_attempted = False
        llm_preview = ""

        context_words = _clean_words(context)
        # Skip LLM if context is too thin — it will just hallucinate
        min_words_for_llm = max(self.config.min_context_words, 10)
        can_use_llm = (
            self.config.draft_mode in ("llm", "llm-assist", "model")
            and len(context_words) >= min_words_for_llm
        )

        if can_use_llm:
            llm_attempted = True
            try:
                truncated = self._truncate_context(context)
                prompt = (
                    "Based on the following source context, create ONE multiple-choice question "
                    "that tests understanding of the material. Respond with ONLY valid JSON in this exact format:\n"
                    '{\"question\": \"...\", \"choices\": [\"A\", \"B\", \"C\", \"D\"], '
                    '\"correct_index\": 0, \"explanation\": \"...\", '
                    '\"option_explanations\": [\"why A is right/wrong\", \"why B is right/wrong\", '
                    '\"why C is right/wrong\", \"why D is right/wrong\"]}\n\n'
                    "Rules:\n"
                    "- The question must be answerable from the context\n"
                    "- Provide exactly 4 plausible choices\n"
                    "- correct_index is the 0-based index of the right answer\n"
                    "- The explanation should say why the correct answer is right\n"
                    "- option_explanations: for EACH option, explain why it is correct or incorrect\n"
                    "- Do NOT repeat the context verbatim as a choice\n"
                    "- Focus your question on specific details mentioned in this particular section\n\n"
                    f"Context:\n{truncated}"
                )
                llm_out = self._get_llm().generate(
                    prompt=prompt, max_tokens=700, temperature=0.4,
                    system="You are an educational quiz generator. Create clear, "
                           "accurate multiple-choice questions based only on the provided context. "
                           "Always respond with valid JSON only."
                )
                generated = str(llm_out.get("text") or "").strip()
                llm_preview = generated[:200]
                parsed = self._parse_llm_mcq(generated)
                if parsed:
                    parsed["draft_source"] = "llm"
                    parsed["llm_attempted"] = True
                    parsed["llm_preview"] = llm_preview
                    return parsed
                else:
                    draft_source = "llm_parse_fallback"
                    logger.warning("LLM MCQ parse failed, falling back to deterministic. Raw: %s", llm_preview)
            except Exception as exc:
                draft_source = "llm_error_fallback"
                logger.warning("LLM MCQ generation error: %s", exc)

        # Fallback to deterministic
        result = self._build_question_deterministic(context, i)
        result["draft_source"] = draft_source
        result["llm_attempted"] = llm_attempted
        result["llm_preview"] = llm_preview
        return result

    def _moderation_check(self, qjson: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.enable_moderation:
            return {"enabled": False, "flagged": False, "reason": None}
        try:
            text_blob = " ".join(
                [str(qjson.get("question") or "")]
                + [str(c) for c in (qjson.get("choices") or [])]
                + [str(qjson.get("explanation") or "")]
            ).strip()
            result = self._get_llm().moderate(text_blob)
            flagged = bool(result.get("flagged")) if isinstance(result, dict) else False
            reason = result.get("reason") if isinstance(result, dict) else None
            return {"enabled": True, "flagged": flagged, "reason": reason}
        except Exception as exc:
            return {"enabled": True, "flagged": False, "reason": f"moderation_error:{exc}"}

    def _quality_gate(self, qjson: Dict[str, Any], context: str) -> Dict[str, Any]:
        words = _clean_words(context)
        choices = qjson.get("choices") if isinstance(qjson, dict) else []
        if not isinstance(choices, list):
            choices = []
        unique_choice_count = len(_dedupe_choices([str(c) for c in choices]))

        checks = {
            "has_question": bool(str(qjson.get("question") or "").strip()),
            "has_four_choices": len(choices) == 4,
            "choices_unique": unique_choice_count == len(choices),
            "context_rich_enough": len(words) >= self.config.min_context_words,
            "correct_index_in_bounds": isinstance(qjson.get("correct_index"), int)
            and 0 <= qjson.get("correct_index", -1) < len(choices),
        }
        passed = all(checks.values())
        return {"passed": passed, "checks": checks}

    def _repair_question(self, qjson: Dict[str, Any], context: str) -> Dict[str, Any]:
        # Try LLM repair first if available and context is rich enough
        context_words = _clean_words(context)
        can_use_llm = (
            self.config.draft_mode in ("llm", "llm-assist", "model")
            and len(context_words) >= max(self.config.min_context_words, 10)
        )
        if can_use_llm:
            try:
                prev_q = str(qjson.get("question") or "")
                truncated = self._truncate_context(context)
                prompt = (
                    "The following MCQ failed quality checks. Rewrite it as a better question "
                    "based on the context. Respond with ONLY valid JSON:\n"
                    '{\"question\": \"...\", \"choices\": [\"A\", \"B\", \"C\", \"D\"], '
                    '\"correct_index\": 0, \"explanation\": \"...\", '
                    '\"option_explanations\": [\"why A\", \"why B\", \"why C\", \"why D\"]}\n\n'
                    f"Failed question: {prev_q}\n\n"
                    f"Context:\n{truncated}"
                )
                llm_out = self._get_llm().generate(
                    prompt=prompt, max_tokens=700, temperature=0.5,
                    system="You are an educational quiz generator. Create clear, "
                           "accurate multiple-choice questions. Respond with valid JSON only."
                )
                parsed = self._parse_llm_mcq(str(llm_out.get("text") or ""))
                if parsed:
                    parsed["draft_source"] = "llm_repair"
                    parsed["llm_attempted"] = True
                    parsed["llm_preview"] = str(llm_out.get("text") or "")[:200]
                    return parsed
            except Exception:
                pass

        # Deterministic fallback
        phrase = _pick_phrase_from_context(context)
        base_choices = qjson.get("choices") or []
        if not isinstance(base_choices, list) or len(base_choices) < 4:
            base_choices = _fallback_distractors(phrase)
            base_choices = [phrase, base_choices[0], base_choices[1], base_choices[2]]
        else:
            base_choices = [str(c) for c in base_choices[:4]]
            base_choices[0] = phrase
        base_choices = _dedupe_choices(base_choices)
        while len(base_choices) < 4:
            base_choices.append(f"distractor {len(base_choices)+1}")

        repaired = {
            "question": "Which phrase appears in the provided source context?",
            "choices": base_choices,
            "correct_index": 0,
            "explanation": "Repaired to ensure deterministic source support.",
            "draft_source": "deterministic_repair",
            "llm_attempted": False,
            "llm_preview": "",
        }
        return repaired

    def _propose_topics(self, block: Dict[str, Any]) -> Dict[str, Any]:
        candidates = block.get("candidates") or []
        sections = block.get("sections") or []
        seed = str(block.get("seed") or "study")
        topics = []

        if isinstance(sections, list) and sections:
            for idx, sec in enumerate(sections[: self.config.max_topics], start=1):
                if not isinstance(sec, dict):
                    continue
                title = str(sec.get("title") or "").strip()
                summary = str(sec.get("summary") or "").strip()
                chunk_ids = sec.get("chunk_ids")
                if not isinstance(chunk_ids, list):
                    chunk_ids = []

                if title and not re.match(r"^topic\s+\d+$", title, flags=re.IGNORECASE):
                    out_title = title
                else:
                    out_title = self._build_topic_title_llm(summary or title, idx)

                topics.append({
                    "title": out_title,
                    "source_chunks": {"ids": [str(x) for x in chunk_ids if x]},
                })

            deduped_sec: List[Dict[str, Any]] = []
            seen_sec = set()
            for t in topics:
                title_key = str(t.get("title", "")).lower().strip()
                if title_key and title_key not in seen_sec:
                    seen_sec.add(title_key)
                    deduped_sec.append(t)
            if deduped_sec:
                return {"topics": deduped_sec[: self.config.max_topics]}

        if not candidates:
            hits = self._retrieve(seed, self.config.max_topics)
            fallback_ids = self._rerank(hits, self.config.max_topics)
            for cid in fallback_ids[: self.config.max_topics]:
                text = _context_for_chunk(cid)
                topics.append({
                    "title": self._build_topic_title_llm(text, len(topics) + 1),
                    "source_chunks": {"ids": [cid]},
                })
            return {"topics": topics}

        for idx, cand in enumerate(candidates[: self.config.max_topics], start=1):
            cand_id = cand.get("id") if isinstance(cand, dict) else None
            cand_text = cand.get("text") if isinstance(cand, dict) else str(cand)
            topics.append({
                "title": self._build_topic_title_llm(str(cand_text or ""), idx),
                "source_chunks": {"ids": [cand_id] if cand_id else []},
            })

        # Deduplicate by title for stability
        deduped: List[Dict[str, Any]] = []
        seen_titles = set()
        for t in topics:
            title_key = str(t.get("title", "")).lower().strip()
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                deduped.append(t)

        return {"topics": deduped[: self.config.max_topics]}

    def _generate_mcqs(self, block: Dict[str, Any]) -> Dict[str, Any]:
        run_started = perf_counter()
        observability.incr("graph_generate_calls_total")
        n = _safe_int(block.get("n", 3), 3, 1, self.config.max_mcqs)
        seed = str(block.get("seed") or "study")
        verify_flag = self._verify_enabled()
        explicit_contexts = block.get("contexts")
        if not isinstance(explicit_contexts, list) or not explicit_contexts:
            explicit_contexts = None

        # pseudo-node: retrieve + rerank
        trace_base = ["retrieve", "rerank", "draft_mcq", "validate_evidence"]
        if explicit_contexts:
            candidate_chunks = [f"inline::{i}" for i in range(len(explicit_contexts))]
            context_map = {candidate_chunks[i]: str(explicit_contexts[i] or "") for i in range(len(candidate_chunks))}
            hits = []
            # Shuffle so we don't always start from section 0
            _random.shuffle(candidate_chunks)
        else:
            hits = self._retrieve(seed, n)
            candidate_chunks = self._rerank(hits, max(1, n))
            context_map = {}

        if not candidate_chunks:
            observability.incr("graph_generate_empty_candidates_total")
            # fallback: still generate MCQs using seed-derived context so callers
            # don't receive an empty batch when vector store is cold.
            candidate_chunks = ["inline::seed"]
            context_map = {"inline::seed": f"The topic is about: {seed}. No detailed content was found in the knowledge base for this topic. Generate a general knowledge question about {seed}."}

        questions = []
        seen_questions: set[str] = set()  # duplicate detection

        # Use a while loop so we can retry on duplicates instead of just
        # skipping and producing fewer questions than requested.
        idx = 0
        max_iterations = n * 3  # safety cap to prevent infinite loops
        iterations = 0

        while len(questions) < n and iterations < max_iterations:
            iterations += 1
            chunk_id = candidate_chunks[idx % len(candidate_chunks)] if candidate_chunks else ""
            context = context_map.get(chunk_id) if context_map else _context_for_chunk(chunk_id)
            context = str(context or "")

            # When we've cycled through all chunks once, combine two chunks
            # for a richer context that produces a different question.
            if idx >= len(candidate_chunks) and len(candidate_chunks) > 1:
                second_id = candidate_chunks[(idx + 1) % len(candidate_chunks)]
                second_ctx = context_map.get(second_id) if context_map else _context_for_chunk(second_id)
                second_ctx = str(second_ctx or "")
                if second_ctx.strip():
                    context = context + "\n\n---\n\n" + second_ctx

            # pseudo-node: draft generation
            qjson = self._build_question(context, idx)
            quality = self._quality_gate(qjson, context)
            moderation = self._moderation_check(qjson)

            # pseudo-node: validation + repair loop
            trace = trace_base.copy()
            if moderation.get("enabled"):
                trace.append("moderate")
            attempts = 0
            det_supported, evidence = deterministic_evidence_check(qjson, [context] if context else [])

            def _correct_choice_is_low_quality(q: Dict[str, Any]) -> bool:
                choices = q.get("choices")
                ci = q.get("correct_index", 0)
                if not isinstance(choices, list) or not choices:
                    return True
                if not isinstance(ci, int) or ci < 0 or ci >= len(choices):
                    return True
                return _is_low_quality_phrase(str(choices[ci]))

            while attempts < self.config.max_retries and (
                (not quality["passed"]) or
                (verify_flag and (not det_supported or moderation.get("flagged"))) or
                _correct_choice_is_low_quality(qjson)
            ):
                # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s, ...
                time.sleep(min(0.1 * (2 ** attempts), 5.0))
                qjson = self._repair_question(qjson, context)
                attempts += 1
                trace.append("repair")
                trace.append("validate_evidence")
                det_supported, evidence = deterministic_evidence_check(qjson, [context] if context else [])
                quality = self._quality_gate(qjson, context)
                moderation = self._moderation_check(qjson)
                if moderation.get("enabled"):
                    trace.append("moderate")

            # Duplicate detection — advance to next chunk and retry with backoff
            q_key = str(qjson.get("question", "")).strip().lower()
            if q_key in seen_questions:
                observability.incr("graph_questions_duplicate_skipped_total")
                # Exponential backoff on duplicate retries: 0.2s, 0.4s, 0.8s, ...
                dup_count = iterations - len(questions)  # consecutive non-productive iterations
                time.sleep(min(0.2 * (2 ** min(dup_count, 5)), 5.0))
                idx += 1
                continue
            seen_questions.add(q_key)

            verified = True if not verify_flag else bool(det_supported and quality["passed"] and not moderation.get("flagged"))
            if verified:
                observability.incr("graph_questions_verified_total")
            else:
                observability.incr("graph_questions_unverified_total")
            if moderation.get("flagged"):
                observability.incr("graph_questions_moderation_flagged_total")
            if attempts > 0:
                observability.incr("graph_questions_repaired_total")

            # Preserve draft metadata from _build_question on the question_json
            if "draft_source" not in qjson:
                qjson["draft_source"] = "deterministic"
            if "llm_attempted" not in qjson:
                qjson["llm_attempted"] = False
            if "llm_preview" not in qjson:
                qjson["llm_preview"] = ""

            questions.append({
                "question_id": f"rag-mcq-{len(questions)+1}",
                "question_json": qjson,
                "verified": verified,
                "deterministic_supported": det_supported,
                "evidence": evidence or "(no deterministic evidence)",
                "source_chunks": [chunk_id] if chunk_id else [],
                "graph_meta": {
                    "graph_version": "v3",
                    "seed": seed,
                    "retries_used": attempts,
                    "verify_enabled": verify_flag,
                    "draft_mode": self.config.draft_mode,
                    "orchestrator_model": self.config.orchestrator_model,
                    "retrieval_hits": len(hits),
                    "quality": quality,
                    "moderation": moderation,
                    "trace": trace,
                },
            })
            idx += 1

        total_ms = (perf_counter() - run_started) * 1000.0
        observability.observe_ms("graph_generate_latency_ms", total_ms)
        summary = {
            "event": "graph_generate_mcqs",
            "seed": seed,
            "requested_n": n,
            "produced_n": len(questions),
            "verify_enabled": verify_flag,
            "verified_count": sum(1 for q in questions if q.get("verified")),
            "moderation_flagged_count": sum(
                1 for q in questions if q.get("graph_meta", {}).get("moderation", {}).get("flagged")
            ),
            "total_retries": sum(int(q.get("graph_meta", {}).get("retries_used", 0)) for q in questions),
            "duration_ms": total_ms,
            "model": self.config.orchestrator_model,
            "draft_mode": self.config.draft_mode,
        }
        observability.add_trace(summary)
        logger.info("graph_trace %s", json.dumps(summary, sort_keys=True))

        return {"questions": questions}

    def run(self, block: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(block, dict):
            return {}
        action = (block or {}).get("action")
        if action == "propose_topics":
            return self._propose_topics(block)
        if action == "generate_mcqs":
            return self._generate_mcqs(block)
        return {}


def load(path: str):
    return RetrievalBackedGraph(path)
