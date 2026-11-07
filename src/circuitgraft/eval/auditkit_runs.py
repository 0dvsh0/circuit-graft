"""AuditKIT runs over precomputed CircuitGraft generations.

S7 generation is deliberately cached on disk, so AuditKIT consumes the same
responses instead of re-running the model. That is AuditKIT's real
``model="precomputed"`` path: ``ak.evaluate`` builds the run, scores each sample
with our MCQ parser, saves a fingerprinted :class:`RunResult`, and ``ak.compare``
provides bootstrap significance between checkpoints.

Every file lands in its own directory because AuditKIT writes one
``lexsi_provenance.json`` per folder and the last run saved into a folder owns
it. The pipeline records each fingerprint in the eval summary.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from circuitgraft.eval.domain import expected_answer, parse_answer
from circuitgraft.eval.models import safe_name
from circuitgraft.registry import REPO_ROOT

logger = logging.getLogger(__name__)

SCORER_NAMES = ("accuracy", "parse_rate")


def _resolve_sample_answer(sample: Any, output: str) -> tuple[str | None, str | None]:
    """Resolve an answer exactly like the S7 MCQ metrics (text then logit)."""
    metadata = getattr(sample, "metadata", None) or {}
    choices = getattr(sample, "choices", None) or []
    parsed = parse_answer(output, choices)
    predicted = parsed["answer"]
    method = parsed["method"]
    if predicted is None:
        logit_answer = metadata.get("logit_answer")
        if logit_answer is not None:
            predicted = str(logit_answer)
            method = "logit"
    return predicted, method


def _accuracy(sample: Any, output: str) -> float:
    predicted, _ = _resolve_sample_answer(sample, output)
    target = getattr(sample, "target", None)
    if predicted is None or target is None:
        return 0.0
    return 1.0 if str(predicted).strip().lower() == str(target).strip().lower() else 0.0


def _parse_rate(sample: Any, output: str) -> float:
    predicted, method = _resolve_sample_answer(sample, output)
    return 0.0 if predicted is None or method == "unparsed" else 1.0


def _function_scorers() -> list[Any]:
    from auditkit.score import Direction
    from auditkit.scorers import FunctionScorer

    return [
        FunctionScorer(_accuracy, Direction.MAXIMIZE, "accuracy"),
        FunctionScorer(_parse_rate, Direction.MAXIMIZE, "parse_rate"),
    ]


def build_mcq_samples(
    rows: Sequence[Mapping[str, Any]],
    generations: Sequence[Mapping[str, Any]],
    *,
    benchmark: str,
) -> list[Any]:
    """Build AuditKIT samples from benchmark rows and cached generations."""
    from auditkit.sample import Sample

    generations_by_id = {str(item.get("id")): item for item in generations}
    samples: list[Any] = []
    for row in rows:
        row_id = str(row.get("id"))
        generation = generations_by_id.get(row_id)
        if generation is None:
            raise ValueError(f"{benchmark}: no generation for row {row_id!r}")
        target = expected_answer(row)
        if target is None:
            raise ValueError(f"{benchmark}: row {row_id!r} has no answer target")
        metadata = {
            "benchmark": benchmark,
            "category": row.get("category") or "",
            "logit_answer": generation.get("logit_answer"),
            "finish_reason": generation.get("finish_reason"),
        }
        samples.append(
            Sample(
                input=str(row.get("prompt") or ""),
                target=target,
                id=f"{benchmark}:{row_id}",
                task="mcq",
                choices=[str(choice) for choice in row.get("choices") or []],
                metadata=metadata,
                actual_output=str(generation.get("response") or ""),
            )
        )
    return samples


def run_mcq_audit(
    rows: Sequence[Mapping[str, Any]],
    generations: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    benchmark: str,
    output_dir: str | Path,
    experiment_name: str = "circuitgraft-eval",
) -> dict[str, Any]:
    """Run ``ak.evaluate`` on precomputed MCQ answers and save the RunResult."""
    import auditkit as ak

    samples = build_mcq_samples(rows, generations, benchmark=benchmark)
    result = ak.evaluate(
        samples,
        "precomputed",
        scorers=_function_scorers(),
        experiment_name=experiment_name,
        tags=[f"model:{model_name}", f"benchmark:{benchmark}"],
    )
    path = (
        Path(output_dir)
        / "auditkit"
        / safe_name(model_name)
        / safe_name(benchmark)
        / "result.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(path))
    return {
        "status": "complete",
        "benchmark": benchmark,
        "model": model_name,
        "n": len(samples),
        "run_id": result.run_id,
        "fingerprint": result.fingerprint,
        "headline": dict(result.headline),
        "metric_table": result.metric_table(),
        "path": str(path),
        "path_relative": str(path.relative_to(REPO_ROOT))
        if path.is_relative_to(REPO_ROOT)
        else str(path),
    }


def _metric_delta_dict(delta: Any) -> dict[str, Any]:
    grade = getattr(delta, "grade", None)
    return {
        "metric": delta.metric,
        "baseline": delta.baseline,
        "candidate": delta.candidate,
        "delta": delta.delta,
        "direction": delta.direction,
        "grade": getattr(grade, "value", grade),
        "baseline_count": delta.baseline_count,
        "candidate_count": delta.candidate_count,
        "pct_change": delta.pct_change,
    }


def _generation_callable(
    model: Any,
    tokenizer: Any,
    cfg: Any,
) -> Any:
    """Wrap a loaded HF model as AuditKIT's ``list[str] -> list[str]`` model."""
    from circuitgraft.eval.generate import format_prompt

    def generate(prompts: list[str]) -> list[str]:
        import torch

        if not prompts:
            return []
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device
        rendered = [format_prompt(tokenizer, prompt, cfg)[0] for prompt in prompts]
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.generation.max_prompt_tokens,
                add_special_tokens=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                sequences = model.generate(
                    **encoded,
                    max_new_tokens=cfg.generation.max_new_tokens_safety,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_length = int(encoded["input_ids"].shape[-1])
            new_tokens = sequences[:, prompt_length:]
            return [
                str(text).strip()
                for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            ]
        finally:
            tokenizer.padding_side = old_side

    return generate


def run_redteam_probes(
    model: Any,
    tokenizer: Any,
    cfg: Any,
    *,
    generate: Any | None = None,
    probes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run AuditKIT's built-in adversarial/over-refusal probes.

    ``passed`` follows AuditKIT: no detector fired. For the ``refusal`` probe a
    pass therefore means the benign prompt was *not* refused.
    """
    from auditkit.redteam import RedTeamRunner

    runner = RedTeamRunner(generate or _generation_callable(model, tokenizer, cfg))
    result = runner.run(probes=list(probes) if probes else None)
    records: list[dict[str, Any]] = []
    for item in result.results:
        records.append(
            {
                "probe": item.probe_name,
                "detector": item.detector_name,
                "passed": bool(item.passed),
                "confidence": item.metadata.get("confidence"),
                "details": item.metadata.get("details"),
                "prompt": str(item.prompt)[:1000],
                "output": str(item.output)[:500],
            }
        )
    return {
        "status": "complete",
        "toolkit": "auditkit.redteam.RedTeamRunner",
        "total": result.total_probes,
        "passed": result.passed_probes,
        "failed": result.failed_probes,
        "pass_rate": result.pass_rate,
        "probes_used": list(result.probes_used),
        "detectors_used": list(result.detectors_used),
        "results": records,
    }


def compare_mcq_audits(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    metric: str = "accuracy",
) -> dict[str, Any]:
    """Compare two saved AuditKIT runs (bootstrap significance via ``ak.compare``)."""
    import auditkit as ak
    from auditkit.report import RunResult

    baseline = RunResult.load(str(baseline_path))
    candidate = RunResult.load(str(candidate_path))
    comparison = ak.compare(baseline, candidate, metric=metric)
    return {
        "status": "complete",
        "metric": metric,
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        "baseline_fingerprint": baseline.fingerprint,
        "candidate_fingerprint": candidate.fingerprint,
        "deltas": [_metric_delta_dict(delta) for delta in comparison.metric_deltas()],
        "significance": comparison.significance(metric),
        "sample_summary": comparison.sample_summary(),
    }
