"""Contrastive-pair validation and decontamination tests."""

from __future__ import annotations

import json

import pytest

from circuitgraft.config import DataConfig
from circuitgraft.data.curate import generate_domain_sensitive
from circuitgraft.data.decontam import (
    decontaminate_by_ngram,
    find_ngram_overlap,
    normalize_text,
    word_ngrams,
)
from circuitgraft.data.pairs import (
    PairValidationError,
    assert_disjoint_from_eval,
    assert_valid_pairs,
    check_pair,
    pair_ids,
    read_pairs,
    validate_pairs,
    write_pairs,
)


class WordTokenizer:
    """Deterministic stand-in: one token per whitespace-delimited word."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))


def test_generate_domain_sensitive_reuses_existing_candidates(tmp_path) -> None:
    out = tmp_path / "domain_benign"
    out.mkdir()
    candidate = {"id": "d1", "prompt": "Is this drug safe?", "label": "benign"}
    (out / "candidates.jsonl").write_text(
        json.dumps(candidate) + "\n", encoding="utf-8"
    )

    result = generate_domain_sensitive(
        DataConfig(output_dir=str(tmp_path)),
        out,
        kind="benign",
        limit=300,
    )

    assert result["reused"] is True
    assert result["rows"] == 1
    review = (out / "human_review.csv").read_text(encoding="utf-8")
    assert "review_label" in review
    assert "d1" in review


def valid_pair(**overrides: object) -> dict[str, object]:
    pair: dict[str, object] = {
        "pair_id": "safety-1",
        "kind": "safety",
        "clean_prompt": "Is it safe to take this medicine daily?",
        "corrupted_prompt": "Is it safe to take this medicine now?",
        "answer": "yes",
        "corrupted_answer": "no",
        "seed_source": "unit-test",
        "category": "safety",
        "label_source": "human",
    }
    pair.update(overrides)
    return pair


def test_valid_pair_passes() -> None:
    check = check_pair(valid_pair(), WordTokenizer())
    assert check.ok, check.errors
    assert check.len_diff_tokens == 0
    assert check.clean_answer_tokens == 1
    assert check.corrupted_answer_tokens == 1


def test_missing_fields_are_reported() -> None:
    check = check_pair({"pair_id": "x", "kind": "safety"}, WordTokenizer())
    assert not check.ok
    assert any("missing or empty field" in error for error in check.errors)
    # Required-field failures short-circuit before tokenisation.
    assert check.len_diff_tokens is None


def test_invalid_pair_semantics_are_reported() -> None:
    tokenizer = WordTokenizer()
    assert not check_pair(valid_pair(clean_prompt="same", corrupted_prompt="same"), tokenizer).ok
    assert not check_pair(valid_pair(kind="other"), tokenizer).ok
    assert not check_pair(valid_pair(label_source="model"), tokenizer).ok
    assert not check_pair(valid_pair(corrupted_answer="yes"), tokenizer).ok
    assert not check_pair(
        valid_pair(corrupted_prompt="Is it safe to take this medicine now please today again"),
        tokenizer,
    ).ok


def test_single_token_answer_can_be_relaxed() -> None:
    pair = valid_pair(answer="yes indeed", corrupted_answer="no thanks")
    assert not check_pair(pair, WordTokenizer()).ok
    assert check_pair(pair, WordTokenizer(), require_single_token_answers=False).ok


def test_validate_and_assert_helpers() -> None:
    pairs = [valid_pair(pair_id="a"), valid_pair(pair_id="b", answer="maybe so")]
    checks = validate_pairs(pairs, WordTokenizer())
    assert [check.pair_id for check in checks] == ["a", "b"]
    assert checks[0].ok
    with pytest.raises(PairValidationError, match="1 invalid pairs"):
        assert_valid_pairs(pairs, WordTokenizer())
    assert pair_ids(pairs) == {"a", "b"}


def test_pair_round_trip(tmp_path) -> None:
    path = tmp_path / "pairs" / "safety.jsonl"
    rows = [valid_pair(pair_id="a"), valid_pair(pair_id="b")]
    write_pairs(path, rows)
    loaded = read_pairs(path)
    assert loaded == rows


def test_ngram_decontamination() -> None:
    reference = "the quick brown fox jumps over the lazy dog while the quiet cat watches from a warm windowsill today"
    identical = {"id": "copy", "text": reference}
    distinct = {"id": "fresh", "text": "a completely separate sentence about a medical trial with no shared words at all here"}
    report = decontaminate_by_ngram([identical, distinct], [reference], n=13)

    assert [item["id"] for item in report.kept] == ["fresh"]
    assert [item["id"] for item in report.removed] == ["copy"]
    assert report.summary()["removed"] == 1
    assert find_ngram_overlap(reference, [reference], 13) is not None


def test_short_texts_match_only_themselves() -> None:
    assert find_ngram_overlap("a short prompt", ["a short prompt"], 13) is not None
    assert find_ngram_overlap("a short prompt", ["a different prompt"], 13) is None
    assert word_ngrams("", 13) == set()
    with pytest.raises(ValueError):
        word_ngrams("text", 0)


def test_normalization_is_case_and_punctuation_insensitive() -> None:
    assert normalize_text("Hello,  World! (2026)") == "hello world 2026"


def test_pairs_must_be_disjoint_from_eval() -> None:
    eval_text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    pair = valid_pair(clean_prompt=eval_text)
    with pytest.raises(PairValidationError, match="overlap an eval set"):
        assert_disjoint_from_eval([pair], [eval_text])
    assert_disjoint_from_eval([valid_pair()], [eval_text])
