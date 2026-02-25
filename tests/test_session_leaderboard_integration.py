from server.main import app
from fastapi.testclient import TestClient
from server.db import create_db_and_tables, get_session
from server.services.ingest_manager import ingest_manager
from server.services.session_manager import manager as session_manager
from server.models import Book, Session as SessionModel, CachedQuestion, Topic

client = TestClient(app)


def setup_module(module):
    create_db_and_tables()


def test_session_submits_to_leaderboard():
    # create a book and a topic for it
    b = ingest_manager.create_book(title="T", authors="A", raw_text="This is a test.")
    with get_session() as db:
        topic = Topic(book_id=b.id, title="Test Topic")
        db.add(topic)
        db.commit()
        db.refresh(topic)
        db.expunge(topic)
    # create a session referencing the real topic
    sess = session_manager.create_session(user_id="u_test", topic_id=topic.id)
    session_manager.start_session(sess.id, duration_minutes=10)
    # create a cached question manually with proper 4-choice MCQ
    with get_session() as db:
        cq = CachedQuestion(chunks_hash="h", question_json={"question": "What?", "choices": ["A","B","C","D"], "correct_index": 0}, variants={})
        db.add(cq)
        db.commit()
        db.refresh(cq)
        # create a PromptEvent referencing it
        from server.models import PromptEvent
        pe = PromptEvent(session_id=sess.id, prompt_text="What?", prompt_question_id=cq.id)
        db.add(pe)
        db.commit()
        db.refresh(pe)
        db.expunge(cq)
        db.expunge(pe)
    # submit correct answer via manager (0-based index)
    data = session_manager.submit_answer(sess.id, pe.id, "0", "u_test")
    assert data.get("correct") is True
    # end the session (this will upsert leaderboard aggregate using the final session score)
    sess_obj, agg = session_manager.end_session(sess.id)

    # ensure leaderboard aggregate exists and reflects final score
    with get_session() as db:
        from sqlmodel import select
        from server.models import LeaderboardAggregate
        q = select(LeaderboardAggregate).where(LeaderboardAggregate.user_id == "u_test")
        agg = db.exec(q).one_or_none()
        assert agg is not None
        assert agg.best_score >= 1
        assert agg.total_score >= 1
