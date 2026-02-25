from fastapi import APIRouter, HTTPException, Depends
from server.services.session_manager import manager
from pydantic import BaseModel
from typing import Optional, List, Any
import os
from time import perf_counter
from server.auth import get_current_user
from server.db import get_session
from server.models import Session
from server.services.rate_limiter import rate_limiter
from server.services.observability import observability

router = APIRouter(prefix="/sessions")


def _require_session_owner(session_id: str, user_id: str) -> Session:
    """Load session and verify ownership. Raises 403 if not the owner, 404 if not found."""
    with get_session() as db:
        sess = db.get(Session, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        if sess.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not allowed")
        db.expunge(sess)
        return sess


# ── Request models ──────────────────────────────────────────────

class CreateSessionIn(BaseModel):
    topic_id: str
    requested_minutes: Optional[int] = 30
    tone: Optional[str] = "neutral"


class SubmitIn(BaseModel):
    prompt_id: str
    answer: str
    reject: Optional[bool] = False


# ── Response models ─────────────────────────────────────────────

class CreateSessionOut(BaseModel):
    session_id: str


class GenerateQuestionsOut(BaseModel):
    generated: int
    questions: List[Any]


class StartSessionOut(BaseModel):
    session_id: str
    started_at: Optional[str] = None


class AggregateSnapshot(BaseModel):
    user_id: str
    best_score: float
    total_score: float
    sessions: int


class EndSessionOut(BaseModel):
    session_id: str
    ended_at: Optional[str] = None
    score: Optional[int] = None
    aggregate: Optional[AggregateSnapshot] = None


class NextQuestionPayload(BaseModel):
    """One MCQ delivered to the student (answer hidden)."""
    prompt_id: str
    prompt_text: Optional[str] = None
    question: Optional[str] = None
    choices: Optional[List[str]] = None
    remaining: Optional[int] = None


class NextQuestionOut(BaseModel):
    next: Optional[NextQuestionPayload] = None


class OptionReasoning(BaseModel):
    """Per-option reasoning returned after answer submission."""
    index: int
    text: str
    correct: bool
    reason: str


class SubmitOut(BaseModel):
    """Full response after submitting an answer, including per-option reasoning."""
    correct: Optional[bool] = None
    session_score: Optional[int] = None
    failures: Optional[int] = None
    correct_index: Optional[int] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    options: Optional[List[OptionReasoning]] = None
    mean_comment: Optional[str] = None
    rejected: Optional[bool] = None
    session_rejects: Optional[int] = None
    remaining: Optional[int] = None
    session_complete: Optional[bool] = None


@router.post("/", response_model=CreateSessionOut, summary="Create a session", description="Create a new study session for the authenticated user. Returns the new session id.")
def create_session(data: CreateSessionIn, current_user=Depends(get_current_user)):
    sess = manager.create_session(current_user.id, data.topic_id, data.requested_minutes or 30, tone=data.tone or "neutral")
    return {"session_id": sess.id}


@router.post("/{session_id}/generate_questions", response_model=GenerateQuestionsOut, summary="Generate MCQs for session", description="Ask the orchestrator to generate MCQs for the session; persists canonical MCQs and prompt events.")
def generate_questions(session_id: str, n: int = 5, current_user=Depends(get_current_user)):
    _require_session_owner(session_id, current_user.id)
    observability.incr("orchestrator_generate_attempt_total")
    per_min = int(os.getenv("ORCHESTRATOR_RATE_LIMIT_PER_MIN", "30"))
    key = f"orchestrator:generate:{current_user.id}"
    if not rate_limiter.allow(key=key, limit=per_min, window_sec=60):
        observability.incr("orchestrator_generate_rate_limited_total")
        raise HTTPException(status_code=429, detail="Rate limit exceeded for question generation")

    started = perf_counter()
    try:
        saved = manager.generate_prompts(session_id, n)
        observability.incr("orchestrator_generate_success_total")
        return {"generated": len(saved), "questions": saved}
    except Exception:
        observability.incr("orchestrator_generate_error_total")
        raise
    finally:
        observability.observe_ms("orchestrator_generate_latency_ms", (perf_counter() - started) * 1000.0)


@router.post("/{session_id}/start", response_model=StartSessionOut, summary="Start session", description="Mark the session as started and set its duration in minutes.")
def start_session(session_id: str, duration_minutes: int = 30, current_user=Depends(get_current_user)):
    _require_session_owner(session_id, current_user.id)
    sess = manager.start_session(session_id, duration_minutes)
    return {"session_id": sess.id, "started_at": str(sess.started_at) if sess.started_at else None}


@router.post("/{session_id}/end", response_model=EndSessionOut, summary="End session", description="End the session and upsert the user's leaderboard aggregate with the final score. Owner-only.")
def end_session(session_id: str, current_user=Depends(get_current_user)):
    # Check ownership BEFORE any side-effects
    _require_session_owner(session_id, current_user.id)
    sess, agg = manager.end_session(session_id)
    out = {"session_id": sess.id, "ended_at": str(sess.ended_at) if sess.ended_at else None, "score": sess.score}
    if agg:
        out["aggregate"] = {"user_id": agg.user_id, "best_score": agg.best_score, "total_score": agg.total_score, "sessions": agg.sessions}
    return out


@router.get("/{session_id}/next_question", response_model=NextQuestionOut, summary="Get next question", description="Return the next MCQ to present to the user (randomized among unanswered questions). Returns the question text, 4 choices, and a prompt_id to use when submitting the answer. The correct answer is NOT included — it is revealed after submission.")
def next_question(session_id: str, current_user=Depends(get_current_user)):
    _require_session_owner(session_id, current_user.id)
    p = manager.get_next_prompt(session_id)
    if not p:
        return {"next": None}
    return {"next": p}


@router.post("/{session_id}/submit", response_model=SubmitOut, summary="Submit an answer", description="Submit an answer (or reject) for the given prompt. Returns correctness, the correct answer, an explanation, and per-option reasoning with tone-aware feedback.")
def submit_answer(session_id: str, data: SubmitIn, current_user=Depends(get_current_user)):
    _require_session_owner(session_id, current_user.id)
    res = manager.submit_answer(session_id, data.prompt_id, data.answer, current_user.id, reject=data.reject or False)
    return res
