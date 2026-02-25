from typing import List, Tuple, Dict, Any
import re


def _check_target_in_contexts(target: str, contexts: List[str]) -> Tuple[bool, str]:
    if not target or not isinstance(target, str):
        return False, ""
    target = target.strip().lower()
    if not target:
        return False, ""

    targ_words = [w for w in re.split(r"\W+", target) if w]
    if not targ_words:
        return False, ""

    for ctx in contexts:
        if not isinstance(ctx, str):
            continue
        ctx_lower = ctx.lower()
        # exact substring check
        if target in ctx_lower:
            start = max(0, ctx_lower.find(target) - 120)
            end = ctx_lower.find(target) + len(target) + 120
            snippet = ctx_lower[start:end]
            return True, snippet.strip()

        # token overlap heuristic
        ctx_words = [w for w in re.split(r"\W+", ctx_lower) if w]
        if not ctx_words:
            continue
        common = set(targ_words) & set(ctx_words)
        if len(common) >= max(1, int(len(targ_words) * 0.5)):
            return True, "..." + " ".join(list(common)) + "..."

    return False, ""


def deterministic_evidence_check(qjson_or_target: Any, contexts: List[str]) -> Tuple[bool, str]:
    """Return (supported, evidence_snippet).

    Accepts either:
      - an MCQ dict with `choices` and `correct_index`, or
      - a plain target string to search for in `contexts`.

    Uses exact substring matching first, then a token-overlap heuristic.
    """
    # If caller passed an MCQ-like dict, extract the target choice
    if isinstance(qjson_or_target, dict):
        qjson = qjson_or_target
        try:
            choice_idx = int(qjson.get("correct_index"))
        except Exception:
            return False, ""
        choices = qjson.get("choices") or []
        if not choices or choice_idx < 0 or choice_idx >= len(choices):
            return False, ""
        target = str(choices[choice_idx])
        return _check_target_in_contexts(target, contexts)

    # If it's a string, treat directly as target text
    if isinstance(qjson_or_target, str):
        return _check_target_in_contexts(qjson_or_target, contexts)

    return False, ""
