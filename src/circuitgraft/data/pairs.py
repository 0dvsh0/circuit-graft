"""Contrastive pair construction, validation and serialisation (JSONL)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from circuitgraft.data.decontam import find_ngram_overlap


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


REQUIRED_FIELDS = (
    "pair_id",
    "kind",
    "clean_prompt",
    "corrupted_prompt",
    "answer",
    "corrupted_answer",
    "seed_source",
    "category",
    "label_source",
)


class PairValidationError(ValueError):
    pass


@dataclass
class PairCheck:
    pair_id: str
    errors: list[str] = field(default_factory=list)
    len_diff_tokens: int | None = None
    clean_answer_tokens: int | None = None
    corrupted_answer_tokens: int | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def token_count(tokenizer: TokenizerLike, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def is_single_token(tokenizer: TokenizerLike, text: str) -> bool:
    return token_count(tokenizer, text) == 1


def check_pair(
    pair: dict[str, Any],
    tokenizer: TokenizerLike,
    *,
    max_len_diff_tokens: int = 2,
    require_single_token_answers: bool = True,
) -> PairCheck:
    pair_id = str(pair.get("pair_id", "<missing>"))
    check = PairCheck(pair_id=pair_id)
    for field_name in REQUIRED_FIELDS:
        if field_name not in pair or pair[field_name] in (None, ""):
            check.errors.append(f"missing or empty field: {field_name}")
    if check.errors:
        return check

    if pair["kind"] not in ("safety", "domain"):
        check.errors.append(f"unknown pair kind: {pair['kind']!r}")
    if pair["label_source"] not in ("human", "auto"):
        check.errors.append(f"label_source must be human|auto, got {pair['label_source']!r}")
    if pair["clean_prompt"] == pair["corrupted_prompt"]:
        check.errors.append("clean_prompt equals corrupted_prompt")

    try:
        clean_len = token_count(tokenizer, pair["clean_prompt"])
        corrupt_len = token_count(tokenizer, pair["corrupted_prompt"])
    except Exception as exc:  # noqa: BLE001 - tokenizer-specific failure is a validation failure
        check.errors.append(f"tokenizer failed: {exc}")
        return check

    check.len_diff_tokens = abs(clean_len - corrupt_len)
    if check.len_diff_tokens > max_len_diff_tokens:
        check.errors.append(
            f"prompt length differs by {check.len_diff_tokens} tokens "
            f"(max {max_len_diff_tokens})"
        )

    try:
        check.clean_answer_tokens = token_count(tokenizer, pair["answer"])
        check.corrupted_answer_tokens = token_count(tokenizer, pair["corrupted_answer"])
    except Exception as exc:  # noqa: BLE001
        check.errors.append(f"tokenizer failed on answers: {exc}")
        return check

    if require_single_token_answers:
        if check.clean_answer_tokens != 1:
            check.errors.append(
                f"answer is {check.clean_answer_tokens} tokens, expected exactly 1"
            )
        if check.corrupted_answer_tokens != 1:
            check.errors.append(
                f"corrupted_answer is {check.corrupted_answer_tokens} tokens, expected exactly 1"
            )
    if pair["answer"] == pair["corrupted_answer"]:
        check.errors.append("answer and corrupted_answer are identical")
    return check


def validate_pairs(
    pairs: Iterable[dict[str, Any]],
    tokenizer: TokenizerLike,
    *,
    max_len_diff_tokens: int = 2,
    require_single_token_answers: bool = True,
) -> list[PairCheck]:
    return [
        check_pair(
            pair,
            tokenizer,
            max_len_diff_tokens=max_len_diff_tokens,
            require_single_token_answers=require_single_token_answers,
        )
        for pair in pairs
    ]


def assert_valid_pairs(pairs: Iterable[dict[str, Any]], tokenizer: TokenizerLike, **kwargs: Any) -> None:
    checks = validate_pairs(pairs, tokenizer, **kwargs)
    bad = [check for check in checks if not check.ok]
    if bad:
        details = "; ".join(f"{c.pair_id}: {', '.join(c.errors)}" for c in bad[:10])
        raise PairValidationError(f"{len(bad)} invalid pairs: {details}")


def assert_disjoint_from_eval(
    pairs: Iterable[dict[str, Any]],
    eval_texts: Iterable[str],
    *,
    ngram_size: int = 13,
    prompt_field: str = "clean_prompt",
) -> None:
    """Reject any pair whose clean prompt shares a 13-gram with an eval item."""
    references = list(eval_texts)
    offenders: list[str] = []
    for pair in pairs:
        if find_ngram_overlap(pair[prompt_field], references, ngram_size):
            offenders.append(str(pair.get("pair_id", "<unknown>")))
    if offenders:
        raise PairValidationError(
            f"{len(offenders)} pairs overlap an eval set at {ngram_size}-grams: {offenders[:10]}"
        )


def read_pairs(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_pairs(path: str | Path, pairs: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def pair_ids(pairs: Iterable[dict[str, Any]]) -> set[str]:
    return {str(pair["pair_id"]) for pair in pairs}
