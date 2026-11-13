"""Offline tests for S7: parsing, AuditKIT, human audit and the full pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from circuitgraft.config import BaseConfig, EvalConfig, JudgeConfig, ModelRefs
from circuitgraft.eval.auditkit_runs import (
    compare_mcq_audits,
    run_mcq_audit,
    run_redteam_probes,
)
from circuitgraft.eval.domain import aggregate_items, parse_answer, score_mcq_rows
from circuitgraft.eval.models import resolve_checkpoint
from circuitgraft.eval.pipeline import run_eval
from circuitgraft.eval.safety import (
    human_audit_directory,
    run_audit,
    write_human_audit_templates,
)

REFUSAL = "I'm sorry, but I cannot help with that."
COMPLIANCE = "Sure, here are the steps you asked for."


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_resolve_checkpoint_rejects_missing_local_artifact() -> None:
    base = BaseConfig(
        models=ModelRefs(drifted="artifacts/drift/scenarioA_seed0/merged")
    )
    cfg = EvalConfig(models=["drifted"], model_paths={})
    with pytest.raises(FileNotFoundError, match="looks like a local path"):
        resolve_checkpoint("drifted", cfg, base)


def test_resolve_checkpoint_keeps_repo_id_for_hub_models() -> None:
    base = BaseConfig(models=ModelRefs(aligned="CohereLabs/tiny-aya-global"))
    cfg = EvalConfig(models=["aligned"], model_paths={})
    assert resolve_checkpoint("aligned", cfg, base) == "CohereLabs/tiny-aya-global"


def test_parse_answer_prefers_explicit_span() -> None:
    assert parse_answer("Reasoning...\nAnswer: (B)", ["A", "B", "C"]) == {
        "answer": "B",
        "method": "answer_letter",
    }
    assert parse_answer("The answer is C.", ["A", "B", "C"])["answer"] == "C"


def test_parse_answer_words_and_think_blocks() -> None:
    assert parse_answer("<think>Answer: A</think>maybe", ["yes", "no"])["method"] == "unparsed"
    parsed = parse_answer("Answer: yes", ["yes", "no"])
    assert parsed == {"answer": "yes", "method": "answer_word"}


def test_score_mcq_rows_uses_logit_fallback_and_keeps_responses_out() -> None:
    rows = [
        {"id": "1", "benchmark": "medmcqa_validation", "choices": ["A", "B"], "answer": "A"},
        {"id": "2", "benchmark": "medmcqa_validation", "choices": ["A", "B"], "answer": "B"},
        {"id": "3", "benchmark": "medmcqa_validation", "choices": ["A", "B"], "answer": "A"},
    ]
    generations = [
        {"id": "1", "response": "Answer: A", "logit_answer": None},
        {"id": "2", "response": "", "logit_answer": "B"},
        {"id": "3", "response": "I do not know", "logit_answer": None},
    ]
    scored = score_mcq_rows(rows, generations, n_resamples=20)
    assert scored["accuracy"] == pytest.approx(2 / 3)
    assert scored["parse_failure_rate"] == pytest.approx(1 / 3)
    assert [item["parse_method"] for item in scored["items"]] == [
        "answer_letter",
        "logit",
        "unparsed",
    ]
    assert "response" not in scored["items"][0]
    assert len(scored["items"][0]["response_sha256"]) == 64


def test_score_mcq_rows_rejects_missing_generations() -> None:
    with pytest.raises(ValueError, match="missing generation"):
        score_mcq_rows(
            [{"id": "1", "choices": ["A"], "answer": "A"}],
            [],
            n_resamples=10,
        )


def test_aggregate_items_empty() -> None:
    metrics = aggregate_items([])
    assert metrics["n"] == 0
    assert metrics["accuracy"] is None
    assert metrics["accuracy_ci"] == [None, None]


def test_auditkit_run_and_compare(tmp_path: Path) -> None:
    rows = [
        {"id": "1", "prompt": "Q1", "choices": ["A", "B"], "answer": "A"},
        {"id": "2", "prompt": "Q2", "choices": ["A", "B"], "answer": "B"},
    ]
    generations = [
        {"id": "1", "response": "Answer: A", "logit_answer": None},
        {"id": "2", "response": "", "logit_answer": "B"},
    ]
    run = run_mcq_audit(
        rows,
        generations,
        model_name="model-a",
        benchmark="medmcqa_validation",
        output_dir=tmp_path,
    )
    assert run["headline"]["accuracy"] == pytest.approx(1.0)
    assert run["headline"]["parse_rate"] == pytest.approx(1.0)
    assert Path(run["path"]).exists()
    assert len(run["fingerprint"]) > 0

    comparison = compare_mcq_audits(run["path"], run["path"])
    assert comparison["significance"]["delta"] == pytest.approx(0.0)
    assert comparison["deltas"][0]["metric"] == "accuracy"


def test_human_audit_template_round_trip(tmp_path: Path) -> None:
    rows = [
        {
            "model": "aligned",
            "benchmark": "harmbench",
            "id": "h1",
            "prompt": "prompt",
            "response": REFUSAL,
            "string_match_refused": True,
        },
        {
            "model": "aligned",
            "benchmark": "harmbench",
            "id": "h2",
            "prompt": "prompt",
            "response": COMPLIANCE,
            "string_match_refused": False,
        },
    ]
    paths = write_human_audit_templates(tmp_path / "audit", rows, per_model=10)
    path = Path(paths["aligned"])
    pending = human_audit_directory(tmp_path / "audit")
    assert pending["status"] == "pending_human_review"

    with path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 2
    for record in records:
        record["human_refusal"] = record["auto_refusal"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    complete = human_audit_directory(tmp_path / "audit")
    assert complete["status"] == "complete"
    assert complete["cohens_kappa"] == pytest.approx(1.0)


def test_auditkit_redteam_with_fake_generation() -> None:
    def generate(prompts: list[str]) -> list[str]:
        return ["Sure, here is a helpful answer." for _ in prompts]

    result = run_redteam_probes(None, None, EvalConfig(), generate=generate)
    assert result["status"] == "complete"
    assert result["probes_used"] == [
        "prompt_injection",
        "jailbreak",
        "encoding",
        "refusal",
    ]
    assert result["total"] == len(result["results"]) > 0
    assert set(result["detectors_used"]) == {
        "keyword",
        "refusal",
        "injection_success",
    }
    assert result["passed"] == result["total"]
    assert result["pass_rate"] == pytest.approx(1.0)


def test_run_audit_reports_not_run_primary_but_keeps_auditkit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The ``not_run`` branch is the no-CUDA path; force it instead of relying
    # on the ambient machine (a CUDA/ROCm box must not flip this expectation).
    monkeypatch.setattr("circuitgraft.eval.safety._guard_device", lambda cfg: None)
    cfg = EvalConfig(
        models=["aligned"],
        model_paths={"aligned": "fake-aligned"},
        audit_output_dir=str(tmp_path / "audit"),
        judge=JudgeConfig(primary="wildguard", allow_cpu=False),
        skip_existing=False,
    )

    def loader(ref: str, *, dtype: str, device: str) -> tuple[Any, Any, str]:
        del ref, dtype, device
        return object(), object(), "cpu"

    def redteam(model: Any, tokenizer: Any, cfg: EvalConfig) -> dict[str, Any]:
        del model, tokenizer, cfg
        return {"status": "complete", "toolkit": "fake", "total": 1, "passed": 1}

    result = run_audit(
        cfg,
        BaseConfig(),
        model_loader=loader,
        redteam_runner=redteam,
    )
    summary = result["summary"]
    assert summary["status"] == "partial"
    assert summary["models"]["aligned"]["safe_tune"]["status"] == "not_run"
    assert summary["models"]["aligned"]["auditkit_redteam"]["status"] == "complete"
    manifest = json.loads(
        (Path(result["output_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "partial"


def test_run_audit_complete_with_string_match_and_fake_suite(tmp_path: Path) -> None:
    cfg = EvalConfig(
        models=["aligned"],
        model_paths={"aligned": "fake-aligned"},
        audit_output_dir=str(tmp_path / "audit"),
        judge=JudgeConfig(primary="string_match"),
        skip_existing=False,
    )

    def loader(ref: str, *, dtype: str, device: str) -> tuple[Any, Any, str]:
        del ref, dtype, device
        return object(), object(), "cpu"

    def evaluator(model: Any, **kwargs: Any) -> dict[str, Any]:
        del model, kwargs
        return {"harmbench": {"asr": 0.25, "refusal_rate": 0.75, "n_evaluated": 4}}

    def redteam(model: Any, tokenizer: Any, cfg: EvalConfig) -> dict[str, Any]:
        del model, tokenizer, cfg
        return {"status": "complete", "toolkit": "fake", "total": 1, "passed": 1}

    result = run_audit(
        cfg,
        BaseConfig(),
        model_loader=loader,
        evaluator=evaluator,
        redteam_runner=redteam,
    )
    summary = result["summary"]
    assert summary["status"] == "complete"
    assert summary["models"]["aligned"]["safe_tune"]["raw"]["harmbench"]["asr"] == 0.25
    manifest = json.loads(
        (Path(result["output_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"


def _fixture_data(root: Path) -> Path:
    data = root / "data"
    eval_dir = data / "eval"
    harmbench = [
        {"id": f"h{index}", "prompt": f"harmful {index}", "benchmark": "harmbench"}
        for index in range(4)
    ]
    domain = [
        {
            "id": f"d{index}",
            "prompt": f"medical {index}",
            "benchmark": "medmcqa_validation",
            "choices": ["A", "B", "C", "D"],
            "answer": "A" if index < 4 else "B",
        }
        for index in range(8)
    ]
    benign = [
        {"id": f"b{index}", "prompt": f"benign {index}", "benchmark": "xstest_safe"}
        for index in range(4)
    ]
    _write_jsonl(eval_dir / "harmbench.jsonl", harmbench)
    _write_jsonl(eval_dir / "medmcqa_validation.jsonl", domain)
    _write_jsonl(eval_dir / "xstest_safe.jsonl", benign)
    (data / "summary.json").write_text(
        json.dumps(
            {
                "eval": {
                    name: {"status": "complete"}
                    for name in ("harmbench", "medmcqa_validation", "xstest_safe")
                }
            }
        ),
        encoding="utf-8",
    )
    return data


def _fake_generator(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    benchmark: str,
    cfg: EvalConfig,
    *,
    model_name: str,
    model_ref: str,
    benchmark_path: Path,
) -> dict[str, Any]:
    del model, tokenizer, model_ref, benchmark_path
    profiles = {
        "aligned": {
            "refuse": {0, 1, 2, 3},
            "benign_refuse": set(),
            "correct": {0, 1, 2, 3},
        },
        "drifted": {
            "refuse": {0},
            "benign_refuse": {0},
            "correct": {0, 1, 2, 3, 4, 5},
        },
        "cg_project": {
            "refuse": {0, 1, 2, 3},
            "benign_refuse": set(),
            "correct": {0, 1, 2, 3, 4, 5},
        },
    }
    profile = profiles[model_name]
    output_path = Path(cfg.generations_dir) / model_name / f"{benchmark}.jsonl"
    generation_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        choices = row.get("choices")
        if choices:
            correct = index in profile["correct"]
            predicted = row["answer"] if correct else next(
                choice for choice in choices if choice != row["answer"]
            )
            response = f"Answer: {predicted}"
            logit_answer = predicted
        else:
            refused_indices = (
                profile["benign_refuse"]
                if benchmark == "xstest_safe"
                else profile["refuse"]
            )
            refused = index in refused_indices
            response = REFUSAL if refused else COMPLIANCE
            logit_answer = None
        generation_rows.append(
            {
                "id": str(row["id"]),
                "benchmark": benchmark,
                "kind": "mcq" if choices else "safety",
                "model": model_name,
                "prompt": row.get("prompt"),
                "response": response,
                "choice_logits": None,
                "logit_answer": logit_answer,
                "finish_reason": "eos",
            }
        )
    _write_jsonl(output_path, generation_rows)
    return {
        "benchmark": benchmark,
        "model": model_name,
        "path": str(output_path),
        "rows": len(generation_rows),
        "skipped": False,
    }


def test_run_eval_pipeline_with_fake_model(tmp_path: Path) -> None:
    data = _fixture_data(tmp_path)
    cfg = EvalConfig(
        models=["aligned", "drifted", "cg_project"],
        benchmarks=["harmbench", "medmcqa_validation", "xstest_safe"],
        data_dir=str(data),
        generations_dir=str(tmp_path / "generations"),
        output_dir=str(tmp_path / "eval"),
        model_paths={
            "aligned": "fake-aligned",
            "drifted": "fake-drifted",
            "cg_project": "fake-cg-project",
        },
        judge=JudgeConfig(primary="string_match"),
        bootstrap_resamples=100,
        skip_existing=False,
    )
    base = BaseConfig()

    def loader(ref: str, *, dtype: str, device: str) -> tuple[Any, Any, str]:
        del ref, dtype, device
        return object(), object(), "cpu"

    result = run_eval(
        cfg,
        base,
        model_loader=loader,
        generator=_fake_generator,
    )
    summary = result["summary"]

    assert summary["gate_g_drift"]["status"] == "passed"
    assert summary["gate_g_drift"]["drop"] == pytest.approx(0.75)
    assert summary["gate_g_drift"]["n_pairs"] == 4

    aligned = summary["models"]["aligned"]
    drifted = summary["models"]["drifted"]
    repaired = summary["models"]["cg_project"]
    assert aligned["metrics"]["hrr"] == pytest.approx(1.0)
    assert drifted["metrics"]["hrr"] == pytest.approx(0.25)
    assert repaired["metrics"]["hrr"] == pytest.approx(1.0)
    assert aligned["metrics"]["domain_accuracy"] == pytest.approx(0.5)
    assert drifted["metrics"]["domain_accuracy"] == pytest.approx(0.75)
    assert drifted["metrics"]["frr"] == pytest.approx(0.25)

    assert repaired["composite"]["srr"] == pytest.approx(1.0)
    assert repaired["composite"]["dgr"] == pytest.approx(1.0)
    assert repaired["composite"]["orp"] == pytest.approx(0.0)
    assert drifted["composite"]["srr"] == pytest.approx(0.0)
    assert drifted["composite"]["dgr"] == pytest.approx(1.0)
    assert drifted["composite"]["orp"] == pytest.approx(0.25)

    assert "domain_mcq" in repaired["auditkit"]["groups"]
    assert "cg_project_vs_drifted:domain_mcq" in summary["auditkit_comparisons"]
    assert "cg_project_vs_drifted" in summary["comparisons"]

    # Per-item outcomes are written next to the summary, not inside it.
    assert "items" not in repaired["judged"]["safety"]["harmbench"]
    assert Path(repaired["items"]["safety"]["path"]).exists()
    items = _read_jsonl(Path(repaired["items"]["safety"]["path"]))
    assert items[0]["id"].startswith("harmbench:")

    assert summary["human_audit"]["status"] == "pending_human_review"
    assert summary["status"] == "partial"
    assert any("human audit" in reason for reason in summary["partial_reasons"])
    manifest = json.loads(
        (Path(result["output_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stage"] == "eval"
    assert manifest["status"] == "partial"
