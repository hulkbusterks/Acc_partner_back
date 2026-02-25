from typing import Dict, Any, Optional


class MCQValidationError(Exception):
    pass


def normalize_mcq(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and validate an MCQ dict from the LLM.

    Expected minimal schema after normalization:
      - question: str
      - choices: list[str] (length 4)
      - correct_index: int (0..len(choices)-1)
      - explanation: str (optional)

    Accepts variants like 'correct_indices' (list) and will pick the first index.
    Raises MCQValidationError on invalid input.
    """
    if not isinstance(raw, dict):
        raise MCQValidationError("MCQ is not a JSON object")

    q = raw.get("question") or raw.get("prompt") or raw.get("text")
    if not q or not isinstance(q, str):
        raise MCQValidationError("Missing or invalid 'question' field")

    choices = raw.get("choices") or raw.get("options")
    if not isinstance(choices, list):
        raise MCQValidationError("Missing or invalid 'choices' array")
    if len(choices) != 4:
        # allow generation with >4 choices by trimming, or pad with placeholders
        if len(choices) > 4:
            choices = choices[:4]
        else:
            # pad with placeholder distractors
            choices = choices + ["(no-op)"] * (4 - len(choices))

    # normalize correct index
    correct_index = None
    if "correct_index" in raw:
        try:
            correct_index = int(raw["correct_index"])
        except Exception:
            pass
    elif "correct_indices" in raw:
        ci = raw.get("correct_indices")
        if isinstance(ci, list) and ci:
            try:
                correct_index = int(ci[0])
            except Exception:
                pass

    if correct_index is None:
        # try to infer from an 'answer' field matching one of the choices
        ans = raw.get("answer")
        if isinstance(ans, str):
            try:
                correct_index = choices.index(ans)
            except ValueError:
                correct_index = None

    if correct_index is None or not (0 <= correct_index < len(choices)):
        raise MCQValidationError("Missing or invalid 'correct_index'")

    explanation = raw.get("explanation") or raw.get("explain") or ""
    if explanation is None:
        explanation = ""

    return {
        "question": q.strip(),
        "choices": [str(c).strip() for c in choices],
        "correct_index": int(correct_index),
        "explanation": str(explanation).strip(),
    }


def try_normalize(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        return normalize_mcq(raw)
    except MCQValidationError:
        return None
