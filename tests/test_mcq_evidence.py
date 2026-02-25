from server.services.mcq_validator import normalize_mcq, MCQValidationError
from server.services.evidence_check import deterministic_evidence_check


def test_normalize_mcq_basic():
    raw = {
        "question": "What is 2+2?",
        "choices": ["1", "2", "3", "4"],
        "correct_index": 3,
        "explanation": "Basic math",
    }
    norm = normalize_mcq(raw)
    assert norm["question"] == "What is 2+2?"
    assert norm["choices"][3] == "4"
    assert norm["correct_index"] == 3


def test_normalize_variants_and_errors():
    raw = {"question": "Q", "options": ["a", "b"], "answer": "b"}
    norm = normalize_mcq(raw)
    assert norm["choices"][1] == "b"
    assert norm["correct_index"] == 1

    # invalid shapes should raise
    try:
        normalize_mcq({})
        assert False, "Expected MCQValidationError"
    except MCQValidationError:
        pass


def test_evidence_check_exact_and_overlap():
    qjson = {"correct_index": 0, "choices": ["Python programming", "X"]}
    contexts = ["I love Python programming and testing.", "Other text."]
    ok, snippet = deterministic_evidence_check(qjson, contexts)
    assert ok is True
    assert "python programming" in snippet.lower()

    # overlap heuristic
    qjson2 = {"correct_index": 0, "choices": ["unit tests", "other"]}
    contexts2 = ["I write many unit tests for code." ]
    ok2, sn2 = deterministic_evidence_check(qjson2, contexts2)
    assert ok2 is True
