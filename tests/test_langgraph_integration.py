import os
from server.services.session_manager import manager as session_manager
from server.services.orchestrator_adapter import get_orchestrator_adapter
from server.services.ingest_manager import IngestManager
from server.db import get_session
from server.models import CachedQuestion, Topic
from sqlmodel import select


def test_langgraph_adapter_integration(tmp_path):
    # point LANGGRAPH_GRAPH_PATH to the richer retrieval-backed sample graph
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sample_graph = os.path.join(repo_root, 'langgraph_samples', 'rag_graph.py')
    os.environ['ORCHESTRATOR'] = 'langgraph'
    os.environ['LANGGRAPH_GRAPH_PATH'] = sample_graph
    os.environ['ORCHESTRATOR_VERIFY'] = 'true'
    session_manager.orch = None

    # ingest a small book and create a topic for it
    ing = IngestManager()
    book = ing.create_book(title='LG Book', authors='Tester', raw_text='Short text for LangGraph test.')
    with get_session() as db:
        topic = Topic(book_id=book.id, title='LangGraph Test Topic')
        db.add(topic)
        db.commit()
        db.refresh(topic)
        db.expunge(topic)
    sess = session_manager.create_session(user_id='user-lg', topic_id=topic.id)
    session_manager.start_session(sess.id, duration_minutes=30)

    # run generate_prompts which should call through the LangGraphAdapter
    # FakeLLM produces identical output, so duplicate detection may reduce count
    res = session_manager.generate_prompts(sess.id, n=2)
    assert isinstance(res, list)
    assert len(res) >= 1  # dedup may collapse identical FakeLLM results
    assert all(isinstance(q.get("source_chunks"), list) for q in res)
    assert all("deterministic_supported" in q for q in res)
    assert all("graph_meta" in q for q in res)
    assert all("trace" in q.get("graph_meta", {}) for q in res)
    assert all("draft_mcq" in q.get("graph_meta", {}).get("trace", []) for q in res)
    assert all(isinstance(q.get("verified"), bool) for q in res)
    assert all("quality" in q.get("graph_meta", {}) for q in res)

    # validate topic proposal behavior via adapter directly (no extra DB work)
    adapter = get_orchestrator_adapter()
    topic_res = adapter.run_rag_pipeline(book.id, {
        "action": "propose_topics",
        "candidates": [{"id": f"{book.id}::chunk::0", "text": "LangGraph nodes coordinate retrieval and validation"}],
    })
    assert len(topic_res.get("topics", [])) >= 1

    # ensure CachedQuestion rows were persisted
    with get_session() as db:
        rows = db.exec(select(CachedQuestion)).all()
        assert len(rows) >= 1  # dedup may reduce count

    # cleanup env for subsequent tests
    os.environ.pop('ORCHESTRATOR_VERIFY', None)
    os.environ.pop('LANGGRAPH_GRAPH_PATH', None)
    os.environ['ORCHESTRATOR'] = 'mock'
    session_manager.orch = None
