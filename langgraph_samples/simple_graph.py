"""
Minimal sample 'graph' compatible with the lazy LangGraph adapter.

This module exposes a `load(path)` function that returns a small object
with a `run(block)` method. It's not a real LangGraph graph, but it
lets developers exercise the adapter codepath locally without installing
an external service.
"""
from typing import Any, Dict


class SimpleGraph:
    def __init__(self, data_path: str = None):
        self.data_path = data_path

    def run(self, block: Dict[str, Any]) -> Dict[str, Any]:
        action = block.get("action")
        if action == "propose_topics":
            candidates = block.get("candidates", [])
            topics = []
            for i, c in enumerate(candidates[:3]):
                topics.append({"title": f"LG Topic {i+1}", "source_chunks": {"ids": [c.get("id")]}})
            return {"topics": topics}

        if action == "generate_mcqs":
            n = int(block.get("n", 3))
            questions = []
            for i in range(n):
                q = {
                    "question_id": f"lg-mcq-{i+1}",
                    "question_json": {
                        "question": f"LG MCQ {i+1}: pick the right one?",
                        "choices": ["A","B","C","D"],
                        "correct_index": i % 4,
                    },
                    "verified": True,
                    "deterministic_supported": False,
                    "evidence": "(lg-mock)",
                    "source_chunks": block.get("seed_chunks", []),
                }
                questions.append(q)
            return {"questions": questions}

        return {}


def load(path: str):
    return SimpleGraph(path)
