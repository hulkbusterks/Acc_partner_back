from datetime import datetime, timezone
from sqlmodel import select
from sqlalchemy import text
from server.db import engine, get_session
from server.models import LeaderboardAggregate
import time
from sqlalchemy.exc import OperationalError


def upsert_aggregate(user_id: str, final_score: float) -> LeaderboardAggregate:
    table = LeaderboardAggregate.__table__.name
    now = datetime.now(timezone.utc).isoformat()
    sql = f"""
    INSERT INTO {table} (user_id, best_score, total_score, sessions, updated_at)
    VALUES (:user_id, :best_score, :total_score, :sessions, :updated_at)
    ON CONFLICT(user_id) DO UPDATE SET
      best_score = CASE WHEN excluded.best_score > {table}.best_score THEN excluded.best_score ELSE {table}.best_score END,
      total_score = {table}.total_score + excluded.total_score,
      sessions = {table}.sessions + excluded.sessions,
      updated_at = excluded.updated_at;
    """
    params = {
        "user_id": user_id,
        "best_score": float(final_score),
        "total_score": float(final_score),
        "sessions": 1,
        "updated_at": now,
    }
    attempts = 3
    backoff = 0.05
    for attempt in range(attempts):
        try:
            with engine.begin() as conn:
                conn.execute(text(sql), params)
            break
        except OperationalError:
            if attempt == attempts - 1:
                raise
            time.sleep(backoff * (2 ** attempt))

    with get_session() as db:
        q = select(LeaderboardAggregate).where(LeaderboardAggregate.user_id == user_id)
        agg = db.exec(q).one()
        db.expunge(agg)
        return agg
