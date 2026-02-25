import os

from langgraph_samples.rag_graph import load
import langgraph_samples.rag_graph as rag_graph
from langgraph_samples.rag_graph import _pick_phrase_from_context


def test_rag_graph_respects_max_mcq_bound():
    os.environ["ORCHESTRATOR_MAX_MCQS"] = "2"
    os.environ["ORCHESTRATOR_DRAFT_MODE"] = "deterministic"
    graph = load("langgraph_samples/rag_graph.py")

    out = graph.run({"action": "generate_mcqs", "n": 9, "contexts": [
        "A strong source context about retrieval quality and search relevance in modern systems",
        "Another detailed context about indexing strategies and vector databases for fast lookups",
    ]})
    # max_mcqs=2 caps output; dedup may reduce further but never above 2
    assert len(out.get("questions", [])) <= 2
    assert len(out.get("questions", [])) >= 1

    os.environ.pop("ORCHESTRATOR_MAX_MCQS", None)
    os.environ.pop("ORCHESTRATOR_DRAFT_MODE", None)


def test_rag_graph_verify_mode_can_reject_weak_context():
    os.environ["ORCHESTRATOR_VERIFY"] = "true"
    os.environ["ORCHESTRATOR_MIN_CONTEXT_WORDS"] = "8"
    os.environ["ORCHESTRATOR_MAX_RETRIES"] = "1"
    graph = load("langgraph_samples/rag_graph.py")

    out = graph.run({"action": "generate_mcqs", "n": 1, "contexts": ["tiny"]})
    assert len(out.get("questions", [])) == 1

    q = out["questions"][0]
    assert q.get("verified") is False
    assert q.get("graph_meta", {}).get("quality", {}).get("passed") is False
    assert q.get("graph_meta", {}).get("retries_used") == 1

    os.environ.pop("ORCHESTRATOR_VERIFY", None)
    os.environ.pop("ORCHESTRATOR_MIN_CONTEXT_WORDS", None)
    os.environ.pop("ORCHESTRATOR_MAX_RETRIES", None)


def test_rag_graph_moderation_metadata_and_model_selection(monkeypatch):
    class _FakeLLM:
        def generate(self, prompt: str, max_tokens: int = 24, temperature: float = 0.0, system=None):
            return {"text": "safe extracted phrase"}

        def moderate(self, text: str):
            return {"flagged": True, "reason": "policy-test"}

    monkeypatch.setattr(rag_graph, "get_llm_adapter", lambda: _FakeLLM())

    os.environ["ORCHESTRATOR_ENABLE_MODERATION"] = "true"
    os.environ["ORCHESTRATOR_VERIFY"] = "true"
    os.environ["ORCHESTRATOR_DRAFT_MODE"] = "llm"
    os.environ["ORCHESTRATOR_MODEL"] = "gpt-x-test"
    os.environ["ORCHESTRATOR_MAX_RETRIES"] = "1"

    graph = load("langgraph_samples/rag_graph.py")
    out = graph.run({"action": "generate_mcqs", "n": 1, "contexts": ["This context has enough words for reliable checks"]})
    q = out["questions"][0]
    meta = q.get("graph_meta", {})

    assert meta.get("draft_mode") == "llm"
    assert meta.get("orchestrator_model") == "gpt-x-test"
    assert meta.get("moderation", {}).get("enabled") is True
    assert meta.get("moderation", {}).get("flagged") is True
    assert q.get("verified") is False

    os.environ.pop("ORCHESTRATOR_ENABLE_MODERATION", None)
    os.environ.pop("ORCHESTRATOR_VERIFY", None)
    os.environ.pop("ORCHESTRATOR_DRAFT_MODE", None)
    os.environ.pop("ORCHESTRATOR_MODEL", None)
    os.environ.pop("ORCHESTRATOR_MAX_RETRIES", None)


def test_phrase_picker_avoids_what_is():
    phrase = _pick_phrase_from_context("What is gear? Gear stands for Gemini Enterprise Agent Ready.")
    assert phrase != "what is"


def test_generate_mcqs_no_candidates_fallback():
    graph = load("langgraph_samples/rag_graph.py")
    out = graph.run({"action": "generate_mcqs", "n": 1, "contexts": []})
    assert len(out.get("questions", [])) == 1


def test_propose_topics_uses_llm_title_when_available(monkeypatch):
    class _FakeLLM:
        def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, system=None):
            return {"text": "Graph Pipeline Basics"}

        def moderate(self, text: str):
            return {"flagged": False, "reason": None}

    monkeypatch.setattr(rag_graph, "get_llm_adapter", lambda: _FakeLLM())
    os.environ["ORCHESTRATOR_DRAFT_MODE"] = "llm"

    graph = load("langgraph_samples/rag_graph.py")
    out = graph.run({
        "action": "propose_topics",
        "candidates": [{"id": "c1", "text": "A context about graph execution, retries, and observability."}],
    })
    assert out.get("topics")
    assert out["topics"][0]["title"] == "Graph Pipeline Basics"

    os.environ.pop("ORCHESTRATOR_DRAFT_MODE", None)


def test_llm_mcq_generation_produces_real_questions(monkeypatch):
    class _FakeLLM:
        def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, system=None):
            return {"text": '{"question": "What improves with retries in backend systems?", "choices": ["System reliability", "Disk latency", "Memory fragmentation", "Network bandwidth"], "correct_index": 0, "explanation": "Reliability improves with retries and observability."}'}

        def moderate(self, text: str):
            return {"flagged": False, "reason": None}

    monkeypatch.setattr(rag_graph, "get_llm_adapter", lambda: _FakeLLM())
    os.environ["ORCHESTRATOR_DRAFT_MODE"] = "llm"
    graph = load("langgraph_samples/rag_graph.py")

    out = graph.run({
        "action": "generate_mcqs",
        "n": 1,
        "contexts": ["Reliability improves significantly with retries and observability instrumentation in production backend systems running distributed workloads."],
    })
    q = out["questions"][0]["question_json"]
    assert q.get("draft_source") == "llm"
    assert q.get("llm_attempted") is True
    assert q["question"] == "What improves with retries in backend systems?"
    assert q["choices"] == ["System reliability", "Disk latency", "Memory fragmentation", "Network bandwidth"]
    assert q["correct_index"] == 0

    os.environ.pop("ORCHESTRATOR_DRAFT_MODE", None)
