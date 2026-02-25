"""A tiny builtin LangGraph-compatible graph for local testing.

Usage: set `LANGGRAPH_GRAPH_PATH` to `builtin:simple_graph` and the adapter
will import and load this graph. The graph exposes a `.run(block)` method
that accepts the same `block` dict used elsewhere.
"""
from typing import Dict, Any


class SimpleGraph:
    def run(self, block: Dict[str, Any]) -> Dict[str, Any]:
        # Minimal behavior: delegate to the existing MockOrchestrator logic
        # to keep behavior consistent for PoC.
        from server.services.orchestrator_adapter import MockOrchestrator

        # The MockOrchestrator.run_rag_pipeline expects (session_id, block)
        # Some call-sites pass book_id as session_id; it's fine for PoC.
        session_id = block.get("session_id") or block.get("id") or "builtin"
        return MockOrchestrator().run_rag_pipeline(session_id, block)


def load(path: str = None) -> SimpleGraph:
    return SimpleGraph()
