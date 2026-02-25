from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Optional
from server.db import get_session
from server.models import LeaderboardEntry, LeaderboardAggregate, User
from server.services.leaderboard_cache import cache
from server.auth import get_current_user
import os
from sqlmodel import select
from datetime import datetime, timezone

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


@router.post("/entries", response_model=LeaderboardEntry, summary="Create raw leaderboard entry", description="Create a raw leaderboard entry record (for analytics or ingest).")
def create_entry(score: float, current_user=Depends(get_current_user)):
    with get_session() as db:
        entry = LeaderboardEntry(user_id=current_user.id, score=score)
        db.add(entry)
        db.commit()
        db.refresh(entry)
        db.expunge(entry)
        return entry


@router.get("/entries", response_model=List[LeaderboardEntry], summary="List raw entries", description="List raw leaderboard entries sorted by score (descending).")
def list_entries(limit: int = 50, current_user=Depends(get_current_user)):
    with get_session() as db:
        q = select(LeaderboardEntry).order_by(LeaderboardEntry.score.desc()).limit(limit)
        res = db.exec(q).all()
        for obj in res:
            db.expunge(obj)
        return res


@router.get("/entries/me", response_model=List[LeaderboardEntry], summary="My score history", description="Return all LeaderboardEntry audit rows for the current user, newest first.")
def my_entries(limit: int = 50, current_user=Depends(get_current_user)):
    with get_session() as db:
        q = (select(LeaderboardEntry)
             .where(LeaderboardEntry.user_id == current_user.id)
             .order_by(LeaderboardEntry.created_at.desc())
             .limit(limit))
        res = db.exec(q).all()
        for obj in res:
            db.expunge(obj)
        return res


@router.get("/aggregate/top", response_model=List[LeaderboardEntry], summary="Top raw entries", description="Return top raw entries by score.")
def top_by_user(limit: int = 10, current_user=Depends(get_current_user)):
    with get_session() as db:
        q = select(LeaderboardEntry).order_by(LeaderboardEntry.score.desc()).limit(limit)
        res = db.exec(q).all()
        for obj in res:
            db.expunge(obj)
        return res


@router.post("/submit", response_model=LeaderboardAggregate, summary="Submit session result", description="Submit a session result. This performs an upsert to the per-user aggregate (best/total/sessions).")
def submit_result(score: float, current_user=Depends(get_current_user)):
    user_id = current_user.id
    with get_session() as db:
        q = select(LeaderboardAggregate).where(LeaderboardAggregate.user_id == user_id)
        existing = db.exec(q).one_or_none()
        now = datetime.now(timezone.utc)
        if existing:
            existing.best_score = max(existing.best_score, score)
            existing.total_score = existing.total_score + score
            existing.sessions = existing.sessions + 1
            existing.updated_at = now
            db.add(existing)
            db.commit()
            db.refresh(existing)
            cache.invalidate(user_id)
            db.expunge(existing)
            return existing
        else:
            agg = LeaderboardAggregate(user_id=user_id, best_score=score, total_score=score, sessions=1, updated_at=now)
            db.add(agg)
            db.commit()
            db.refresh(agg)
            cache.invalidate(user_id)
            db.expunge(agg)
            return agg


@router.get("/aggregates", response_model=List[LeaderboardAggregate], summary="List aggregates", description="List per-user aggregates; order by `best` (default) or `total`.")
def list_aggregates(limit: int = 50, order_by: str = "best", current_user=Depends(get_current_user)):
    with get_session() as db:
        if order_by == "best":
            q = select(LeaderboardAggregate).order_by(LeaderboardAggregate.best_score.desc()).limit(limit)
        elif order_by == "total":
            q = select(LeaderboardAggregate).order_by(LeaderboardAggregate.total_score.desc()).limit(limit)
        else:
            q = select(LeaderboardAggregate).limit(limit)
        res = db.exec(q).all()
        for obj in res:
            db.expunge(obj)
        return res


@router.get("/aggregate/{user_id}", response_model=LeaderboardAggregate, summary="Get user aggregate", description="Fetch the per-user aggregate (uses a short in-process TTL cache). Returns 404 if missing.")
def get_aggregate(user_id: str, current_user=Depends(get_current_user)):
    cached = cache.get(user_id)
    if cached:
        return cached
    with get_session() as db:
        q = select(LeaderboardAggregate).where(LeaderboardAggregate.user_id == user_id)
        agg = db.exec(q).one_or_none()
        if not agg:
            raise HTTPException(status_code=404, detail="Not found")
        db.expunge(agg)
        cache.set(user_id, agg)
        return agg


@router.post("/reset", summary="Admin: reset leaderboard", description="Admin-only: clear leaderboard tables and cache. Requires `ADMIN_TOKEN` env var to be set and passed in X-Admin-Token header.")
def reset_leaderboard(x_admin_token: Optional[str] = Header(None)):
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=401, detail="admin token not configured")
    if x_admin_token != expected:
        raise HTTPException(status_code=401, detail="unauthorized")
    from sqlalchemy import text as sql_text
    with get_session() as db:
        tbl_agg = LeaderboardAggregate.__table__.name
        tbl_entry = LeaderboardEntry.__table__.name
        db.exec(sql_text(f"DELETE FROM {tbl_entry}"))
        db.exec(sql_text(f"DELETE FROM {tbl_agg}"))
        db.commit()
    cache.clear()
    return {"status": "ok"}
