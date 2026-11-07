"""MCQ parsing and domain/general accuracy metrics.

The generation pass writes text plus first-step choice logits. Parsing prefers
an explicit ``Answer: X`` span, then the first standalone option letter/word,
then the logit argmax. Parse failures are counted separately; they are never
silently scored as wrong without being reported.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from circuitgraft.eval.stats import bootstrap_ci
from circuitgraft.registry import sha256_bytes

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_LETTER_RE = re.compile(
    r"\bAnswer\s*[:\-]?\s*[\(\[]?([A-Da-d])[\)\]]?(?![A-Za-z])",
    re.IGNORECASE,
)
_STANDALONE_LETTER_RE = re.compile(r"(?:^|[\s\(\[])([A-Da-d])(?:[\s\.\)\]:,]|$)")
_WORD_ANSWER_RE = re.compile(r"\bAnswer\s*[:\-]?\s*[\(\[]?([A-Za-z]+)", re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _is_letter_choices(choices: Sequence[str]) -> bool:
    return bool(choices) and all(
        isinstance(choice, str) and len(choice.strip()) == 1 and choice.strip().upper() in "ABCD"
        for choice in choices
    )


def parse_answer(
    response: str,
    choices: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse a generated answer. Returns ``{"answer": str|None, "method": str}``."""
    text = _strip_think(str(response or ""))
    if not text:
        return {"answer": None, "method": "empty"}
    choices = [str(choice) for choice in choices] if choices else []

    if choices and not _is_letter_choices(choices):
        lowered = text.lower()
        match = _WORD_ANSWER_RE.search(lowered)
        if match and match.group(1) in choices:
            return {"answer": match.group(1), "method": "answer_word"}
        for choice in sorted(choices, key=len, reverse=True):
            if re.search(rf"\b{re.escape(choice.lower())}\b", lowered[:120]):
                return {"answer": choice, "method": "choice_word"}
        return {"answer": None, "method": "unparsed"}

    match = _ANSWER_LETTER_RE.search(text)
    if match:
        return {"answer": match.group(1).upper(), "method": "answer_letter"}
    match = _STANDALONE_LETTER_RE.search(text)
    if match:
        return {"answer": match.group(1).upper(), "method": "standalone_letter"}
    return {"answer": None, "method": "unparsed"}


def expected_answer(row: Mapping[str, Any]) -> str | None:
    answer = row.get("answer")
    if answer is None:
        return None
    if isinstance(answer, int):
        choices = row.get("choices")
        if choices and 0 <= answer < len(choices):
            return str(choices[answer])
        return chr(ord("A") + answer)
    return str(answer).strip()


def score_item(
    row: Mapping[str, Any],
    generation: Mapping[str, Any],
    *,
    fallback_logit: bool = True,
) -> dict[str, Any]:
    choices = row.get("choices")
    parsed = parse_answer(str(generation.get("response") or ""), choices)
    predicted = parsed["answer"]
    method = parsed["method"]
    if predicted is None and fallback_logit:
        logit_answer = generation.get("logit_answer")
        if logit_answer is not None:
            predicted = str(logit_answer)
            method = "logit"
    expected = expected_answer(row)
    correct = bool(
        predicted is not None
        and expected is not None
        and str(predicted).strip().lower() == str(expected).strip().lower()
    )
    response = str(generation.get("response") or "")
    return {
        "id": str(row.get("id")),
        "benchmark": row.get("benchmark"),
        "category": row.get("category") or "",
        "expected": expected,
        "predicted": predicted,
        "correct": correct,
        "parse_method": method,
        # Full generations stay in results/generations; the metrics document
        # keeps only a fingerprint so the eval summary is safe to publish.
        "response_chars": len(response),
        "response_sha256": sha256_bytes(response.encode("utf-8")),
    }


def aggregate_items(
    items: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    if not items:
        return {
            "n": 0,
            "accuracy": None,
            "accuracy_ci": [None, None],
            "parse_failure_rate": None,
            "correct": 0,
        }
    outcomes = [1.0 if item["correct"] else 0.0 for item in items]
    lower, upper = bootstrap_ci(outcomes, n_resamples=n_resamples, ci=ci, seed=seed)
    parse_failures = sum(
        1 for item in items if item.get("predicted") is None or item.get("parse_method") == "unparsed"
    )
    methods: dict[str, int] = {}
    for item in items:
        method = str(item.get("parse_method") or "unknown")
        methods[method] = methods.get(method, 0) + 1
    by_category: dict[str, dict[str, Any]] = {}
    for item in items:
        category = str(item.get("category") or "uncategorized")
        bucket = by_category.setdefault(category, {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(bool(item["correct"]))
    for bucket in by_category.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["n"], 6)
    return {
        "n": len(items),
        "correct": sum(1 for item in items if item["correct"]),
        "accuracy": round(sum(outcomes) / len(outcomes), 6),
        "accuracy_ci": [round(lower, 6), round(upper, 6)],
        "parse_failure_rate": round(parse_failures / len(items), 6),
        "parse_methods": methods,
        "by_category": by_category,
    }


def score_mcq_rows(
    rows: Sequence[Mapping[str, Any]],
    generations: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    fallback_logit: bool = True,
) -> dict[str, Any]:
    by_id = {str(item.get("id")): item for item in generations}
    missing = [str(row.get("id")) for row in rows if str(row.get("id")) not in by_id]
    if missing:
        raise ValueError(f"missing generation rows for ids: {missing[:5]}")
    items = [
        score_item(row, by_id[str(row.get("id"))], fallback_logit=fallback_logit)
        for row in rows
    ]
    metrics = aggregate_items(items, n_resamples=n_resamples, ci=ci, seed=seed)
    return {**metrics, "items": items}


def pooled_items(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pooled: list[dict[str, Any]] = []
    for document in documents:
        pooled.extend(document.get("items") or [])
    return pooled
