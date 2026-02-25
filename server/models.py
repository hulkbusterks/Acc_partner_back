from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, UniqueConstraint
from sqlalchemy import JSON as SA_JSON
from datetime import datetime, timezone
import uuid


def gen_uuid() -> str:
    return str(uuid.uuid4())


class User(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("email"),)
    id: str = Field(default_factory=gen_uuid, primary_key=True)
    email: str = Field(index=True)
    display_name: Optional[str] = None
    password_hash: str
    preferences: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(SA_JSON, nullable=True))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Book(SQLModel, table=True):
    id: str = Field(default_factory=gen_uuid, primary_key=True)
    title: str
    authors: Optional[str] = None
    raw_text: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(SA_JSON, nullable=True))


class Topic(SQLModel, table=True):
    id: str = Field(default_factory=gen_uuid, primary_key=True)
    book_id: str = Field(foreign_key="book.id")
    title: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    source_chunks: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(SA_JSON, nullable=True))


class Session(SQLModel, table=True):
    id: str = Field(default_factory=gen_uuid, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    topic_id: str = Field(foreign_key="topic.id")
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_sec: Optional[int] = None
    failure_count: int = 0
    reject_count: int = 0
    score: int = 0
    tone: str = "neutral"


class PromptEvent(SQLModel, table=True):
    id: str = Field(default_factory=gen_uuid, primary_key=True)
    session_id: str = Field(foreign_key="session.id", index=True)
    prompt_text: str
    response_text: Optional[str] = None
    correct: Optional[bool] = None
    prompt_question_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CachedQuestion(SQLModel, table=True):
    id: str = Field(default_factory=gen_uuid, primary_key=True)
    chunks_hash: str
    question_json: Dict[str, Any] = Field(sa_column=Column(SA_JSON, nullable=False))
    variants: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(SA_JSON, nullable=True))
    usage_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QuestionUsage(SQLModel, table=True):
    id: str = Field(default_factory=gen_uuid, primary_key=True)
    question_id: str
    session_id: str = Field(foreign_key="session.id")
    user_id: str = Field(foreign_key="user.id")
    shown_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    answered_at: Optional[datetime] = None
    answer_given: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(SA_JSON, nullable=True))
    is_correct: Optional[bool] = None


class LeaderboardEntry(SQLModel, table=True):
    """Audit-trail row: one per finished session."""
    id: str = Field(default_factory=gen_uuid, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    session_id: Optional[str] = Field(default=None, foreign_key="session.id")
    topic_id: Optional[str] = Field(default=None, foreign_key="topic.id")
    book_id: Optional[str] = Field(default=None, foreign_key="book.id")
    score: float
    session_count: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LeaderboardAggregate(SQLModel, table=True):
    user_id: str = Field(primary_key=True, foreign_key="user.id")
    best_score: float = 0.0
    total_score: float = 0.0
    sessions: int = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
