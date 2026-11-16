#!/usr/bin/env python3
"""Re-score the MCQ half of a downloaded Kaggle eval summary (CPU only).

The 2026-11-16 Kaggle eval compared predicted option **texts** against letter
golds: ``expected_answer`` returned ``"B"`` while every prediction was the full
choice text, so MedMCQA / MedQA / MMLU accuracy was exactly 0.0 for all eight
models while ``parse_rate`` stayed 1.0. The fix lives in
``circuitgraft.eval.domain`` (letter golds resolve through ``answer_idx``).

This script re-scores the *cached generations* with the fixed code and writes a
new ``artifacts/eval/eval-rescored-<stamp>/`` run. It never edits the
downloaded evidence; ``cg report`` picks the rescored run up as the newest
``artifacts/eval/eval-*`` directory.

What is recomputed vs carried over
----------------------------------
- **Recomputed** (affected by the bug): per-model ``judged.mcq`` blocks,
  AuditKIT benchmark/pooled-group mirrors, ``domain_accuracy`` /
  ``general_accuracy`` metrics, the ``domain_accuracy`` arm of every comparison
  (with McNemar), the G-drift *domain gain*, and every model's DGR composite.
  Safety/benign comparisons and the HRR drop are carried verbatim and the Holm
  family is re-run over all p-values exactly as S7 does.
- **Carried verbatim** (unaffected): safety/benign judged blocks, pooled HRR
  and FRR metrics, generation manifests, model status, checkpoint refs.

Models whose generations are missing under ``--generations-dir`` are carried
over untouched and reported as a partial reason.

Usage::

    python scripts/rescore_eval.py \
        --summary artifacts/eval/eval-kaggle-merged/summary.json \
        --generations-dir results/generations
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from circuitgraft.eval.auditkit_runs import run_mcq_audit  # noqa: E402
from circuitgraft.eval.domain import aggregate_items, score_mcq_rows  # noqa: E402
from circuitgraft.eval.generate import benchmark_kind, load_benchmark_rows  # noqa: E402
from circuitgraft.eval.models import repo_path, safe_name  # noqa: E402
from circuitgraft.eval.pipeline import (  # noqa: E402
    COMPOSITE_DEFINITIONS,
    DOMAIN_MCQ_BENCHMARKS,
    _composite,
    _outcome_lists,
    _paired_comparison,
    _with_prefixed_id,
    _write_jsonl,
)
from circuitgraft.eval.stats import holm_correction, paired_bootstrap_ci  # noqa: E402
from circuitgraft.registry import RunTimer, dump_json, new_manifest  # noqa: E402

logger = logging.getLogger("rescore_eval")

RESAMPLES = 1000
CI = 0.95
SEED = 0
DGR_MIN_GAIN = 0.03


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prefix_items(items: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    return [{**item, "id": f"{benchmark}:{item['id']}"} for item in items]


def _pooled(items: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_items(items, n_resamples=RESAMPLES, ci=CI, seed=SEED)


def rescore_model(
    name: str,
    doc: dict[str, Any],
    *,
    eval_rows_dir: Path,
    generations_dir: Path,
    out_dir: Path,
    groups: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[str]]:
    """Return (updated model doc, grouped re-scored items, rescore reasons)."""
    reasons: list[str] = []
    new_doc = json.loads(json.dumps(doc))  # deep copy, carry everything
    new_doc["auditkit"] = {"benchmarks": {}, "groups": {}, "errors": []}
    doc_items: dict[str, list[dict[str, Any]]] = {
        "safety": [],
        "benign": [],
        "domain_mcq": [],
        "general_mcq": [],
    }
    mcq_inputs: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    judged_mcq: dict[str, dict[str, Any]] = {}

    source_mcq = doc.get("judged", {}).get("mcq", {})
    for benchmark, source_judged in source_mcq.items():
        expected_rows = int(source_judged.get("n") or 0)
        rows_path = eval_rows_dir / f"{benchmark}.jsonl"
        generations_path = generations_dir / safe_name(name) / f"{benchmark}.jsonl"
        if not rows_path.exists():
            reasons.append(f"{name}: eval rows missing at {rows_path}")
            continue
        if not generations_path.exists():
            reasons.append(
                f"{name}: generations missing at {generations_path} "
                "(download results/generations from the Kaggle dataset)"
            )
            continue
        try:
            rows = load_benchmark_rows(rows_path, max_rows=expected_rows or None)
            generations = _read_jsonl(generations_path)
            if expected_rows and len(rows) != expected_rows:
                logger.warning(
                    "%s/%s: %d rows loaded, source scored %d",
                    name,
                    benchmark,
                    len(rows),
                    expected_rows,
                )
            scored = score_mcq_rows(rows, generations, n_resamples=RESAMPLES, ci=CI, seed=SEED)
        except Exception as exc:  # noqa: BLE001
            logger.exception("rescore failed: %s/%s", name, benchmark)
            reasons.append(f"{name}: rescore failed on {benchmark}: {exc}")
            continue
        judged_mcq[benchmark] = scored
        mcq_inputs[benchmark] = (rows, generations)
        group = "domain_mcq" if benchmark in groups["domain_mcq"] else "general_mcq"
        doc_items[group].extend(_prefix_items(scored["items"], benchmark))

    if not judged_mcq:
        reasons.append(f"{name}: no MCQ benchmark re-scored; blocks carried verbatim")
        return new_doc, doc_items, reasons

    new_doc["judged"] = {
        "safety": doc.get("judged", {}).get("safety", {}),
        "benign": doc.get("judged", {}).get("benign", {}),
        "mcq": judged_mcq,
    }

    # AuditKIT mirrors over the re-scored generations (fresh fingerprints).
    for benchmark, (rows, generations) in mcq_inputs.items():
        try:
            new_doc["auditkit"]["benchmarks"][benchmark] = run_mcq_audit(
                rows,
                generations,
                model_name=name,
                benchmark=benchmark,
                output_dir=out_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("AuditKIT rescore failed: %s/%s", name, benchmark)
            reasons.append(f"{name}: AuditKIT rescore failed on {benchmark}: {exc}")
    for group, benchmarks in (
        ("domain_mcq", groups["domain_mcq"]),
        ("general_mcq", groups["general_mcq"]),
    ):
        pooled_rows: list[dict[str, Any]] = []
        pooled_generations: list[dict[str, Any]] = []
        for benchmark in benchmarks:
            if benchmark not in mcq_inputs:
                continue
            rows, generations = mcq_inputs[benchmark]
            pooled_rows.extend(_with_prefixed_id(rows, benchmark))
            pooled_generations.extend(_with_prefixed_id(generations, benchmark))
        if not pooled_rows:
            continue
        try:
            new_doc["auditkit"]["groups"][group] = run_mcq_audit(
                pooled_rows,
                pooled_generations,
                model_name=name,
                benchmark=group,
                output_dir=out_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("AuditKIT pooled rescore failed: %s/%s", name, group)
            reasons.append(f"{name}: AuditKIT pooled rescore failed on {group}: {exc}")

    # Per-item outcome files for the re-scored groups.
    items_block = dict(doc.get("items") or {})
    for group in ("domain_mcq", "general_mcq"):
        if not doc_items[group]:
            continue
        item_path = _write_jsonl(
            out_dir / "items" / safe_name(name) / f"{group}.jsonl", doc_items[group]
        )
        items_block[group] = {
            "path": str(item_path.relative_to(REPO_ROOT))
            if item_path.is_relative_to(REPO_ROOT)
            else str(item_path),
            "rows": len(doc_items[group]),
            "rescored": True,
        }
    for group in ("safety", "benign"):
        if group in items_block:
            items_block[group]["note"] = (
                "carried from the source eval run (unaffected by the MCQ fix)"
            )
    new_doc["items"] = items_block

    # Metrics: safety/benign carried; domain/general recomputed.
    metrics = dict(doc.get("metrics") or {})
    if doc_items["domain_mcq"]:
        pooled_doc = _pooled(doc_items["domain_mcq"])
        metrics["domain_accuracy"] = pooled_doc["accuracy"]
        metrics["domain_accuracy_ci"] = pooled_doc["accuracy_ci"]
        metrics["domain_n"] = pooled_doc["n"]
        metrics["domain_parse_failure_rate"] = pooled_doc["parse_failure_rate"]
    if doc_items["general_mcq"]:
        pooled_doc = _pooled(doc_items["general_mcq"])
        metrics["general_accuracy"] = pooled_doc["accuracy"]
        metrics["general_accuracy_ci"] = pooled_doc["accuracy_ci"]
        metrics["general_n"] = pooled_doc["n"]
        metrics["general_parse_failure_rate"] = pooled_doc["parse_failure_rate"]
    new_doc["metrics"] = metrics
    new_doc["status"] = "complete" if not reasons else "partial"
    return new_doc, doc_items, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="downloaded eval summary.json")
    parser.add_argument(
        "--generations-dir",
        default="results/generations",
        help="directory holding <model>/<benchmark>.jsonl cached generations",
    )
    parser.add_argument("--eval-rows-dir", default="artifacts/data/eval")
    parser.add_argument("--out-root", default="artifacts/eval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary_path = repo_path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_run_id = str(summary.get("run_id") or "eval")
    eval_rows_dir = repo_path(args.eval_rows_dir)
    generations_dir = repo_path(args.generations_dir)
    out_root = repo_path(args.out_root)

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"eval-rescored-{stamp}"
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    groups_src = list((summary.get("benchmarks") or {}).keys())
    groups = {
        "safety": [],
        "benign": [],
        "domain_mcq": [],
        "general_mcq": [],
    }
    for benchmark in groups_src:
        canonical = benchmark[11:] if benchmark.startswith("selection_") else benchmark
        kind = benchmark_kind(canonical)
        if kind == "safety":
            groups["safety"].append(benchmark)
        elif kind == "benign":
            groups["benign"].append(benchmark)
        elif canonical in DOMAIN_MCQ_BENCHMARKS:
            groups["domain_mcq"].append(benchmark)
        else:
            groups["general_mcq"].append(benchmark)

    rescored_items: dict[str, dict[str, list[dict[str, Any]]]] = {}
    models_summary: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    for name, doc in (summary.get("models") or {}).items():
        if doc.get("status") == "unavailable":
            models_summary[name] = doc
            continue
        new_doc, doc_items, model_reasons = rescore_model(
            name,
            doc,
            eval_rows_dir=eval_rows_dir,
            generations_dir=generations_dir,
            out_dir=out_dir,
            groups=groups,
        )
        rescored_items[name] = doc_items
        reasons.extend(model_reasons)
        models_summary[name] = new_doc

    # Reference metrics and composites (DGR becomes defined when the drift gain
    # clears the gate with the corrected numbers).
    reference_metrics = {
        name: models_summary[name].get("metrics", {})
        for name in ("aligned", "drifted")
        if name in models_summary
    }
    for name, doc in models_summary.items():
        if doc.get("status") == "unavailable":
            continue
        doc["composite"] = _composite(
            name,
            doc.get("metrics", {}),
            reference_metrics,
            dgr_min_gain=DGR_MIN_GAIN,
        )

    # Comparisons: safety/benign arms carried, domain arm re-scored, Holm
    # re-run over the whole family exactly like S7.
    comparisons: dict[str, Any] = {}
    family: list[tuple[str, str, float]] = []
    for key, entry in (summary.get("comparisons") or {}).items():
        baseline = entry.get("baseline")
        candidate = entry.get("candidate")
        safety = entry.get("safety_hrr")
        benign = entry.get("benign_frr")
        domain = entry.get("domain_accuracy")
        if (
            domain is not None
            and rescored_items.get(baseline, {}).get("domain_mcq")
            and rescored_items.get(candidate, {}).get("domain_mcq")
        ):
            domain = _paired_comparison(
                rescored_items[baseline]["domain_mcq"],
                rescored_items[candidate]["domain_mcq"],
                key="correct",
                n_resamples=RESAMPLES,
                ci=CI,
                seed=SEED,
            )
        comparisons[key] = {
            "baseline": baseline,
            "candidate": candidate,
            "safety_hrr": safety,
            "domain_accuracy": domain,
            "benign_frr": benign,
        }
        for metric, result in (
            ("safety_hrr", safety),
            ("domain_accuracy", domain),
            ("benign_frr", benign),
        ):
            if result is not None:
                family.append((key, metric, result["mcnemar"]["p_value"]))
    if family:
        adjusted = holm_correction([entry[2] for entry in family])
        for (key, metric, _p), p_holm in zip(family, adjusted, strict=True):
            comparisons[key][metric]["mcnemar"]["p_value_holm"] = round(p_holm, 8)
            comparisons[key][metric]["mcnemar"]["significant_holm"] = p_holm <= 0.05

    # G-drift: the HRR drop is carried (safety items untouched); the domain
    # gain is re-scored from the corrected items.
    gate = dict(summary.get("gate_g_drift") or {})
    if rescored_items.get("aligned", {}).get("domain_mcq") and rescored_items.get(
        "drifted", {}
    ).get("domain_mcq"):
        outcomes = _outcome_lists(
            rescored_items["aligned"]["domain_mcq"],
            rescored_items["drifted"]["domain_mcq"],
            "correct",
        )
        if outcomes is not None:
            a_values, d_values, _ids = outcomes
            d_lower, d_upper = paired_bootstrap_ci(
                d_values, a_values, n_resamples=RESAMPLES, ci=CI, seed=SEED
            )
            gate["domain_gain"] = round(
                sum(d_values) / len(d_values) - sum(a_values) / len(a_values), 6
            )
            gate["domain_gain_ci"] = [round(d_lower, 6), round(d_upper, 6)]
            gate["note"] = (
                "drop/drop_ci carried from "
                f"{source_run_id} (safety items unaffected); domain gain re-scored"
            )

    deduped_reasons: list[str] = []
    for reason in [*reasons, *(summary.get("partial_reasons") or [])]:
        if reason not in deduped_reasons:
            deduped_reasons.append(reason)
    status = "partial" if deduped_reasons else "complete"

    notes = str(summary.get("notes") or "")
    summary_out = {
        "run_id": run_id,
        "config_hash": "kaggle-rescored",
        "status": status,
        "rescore": {
            "source_run_id": source_run_id,
            "source_summary": str(summary_path),
            "fix": "expected_answer resolves letter golds through answer_idx (circuitgraft.eval.domain)",
            "reason": (
                "2026-11-16 MCQ letter-gold bug: predicted option texts were compared "
                "against letter golds, so MedMCQA/MedQA/MMLU accuracy was 0.0 for every model"
            ),
            "generations_dir": str(generations_dir),
            "recomputed": "judged.mcq, auditkit MCQ runs, domain/general metrics, domain comparisons, DGR, domain gain",
            "carried": "safety/benign judged blocks, pooled HRR/FRR, generation manifests, model status",
        },
        "models": models_summary,
        "benchmarks": summary.get("benchmarks") or {},
        "missing_benchmarks": summary.get("missing_benchmarks") or {},
        "missing_models": summary.get("missing_models") or [],
        "gate_g_drift": gate,
        "comparisons": comparisons,
        "auditkit_comparisons": summary.get("auditkit_comparisons") or {},
        "human_audit": summary.get("human_audit") or {},
        "composite_definitions": COMPOSITE_DEFINITIONS,
        "partial_reasons": deduped_reasons,
        "notes": (notes + " | " if notes else "")
        + f"MCQ arm re-scored from cached generations after the letter-gold fix; source {source_run_id} kept verbatim.",
    }
    dump_json(out_dir / "summary.json", summary_out)

    inputs = [summary_path, eval_rows_dir]
    manifest = new_manifest(
        "eval",
        run_id,
        "kaggle-rescored",
        inputs=inputs,
        notes=(
            f"CPU rescore of {source_run_id} MCQ results after the "
            "expected_answer letter-gold fix; safety/benign carried verbatim; "
            "no model was loaded and no generation was re-run"
        ),
    )
    with RunTimer(manifest, out_dir, status=status):
        pass

    # Re-print the headline numbers the report stage will consume.
    for name in sorted(models_summary):
        metrics = models_summary[name].get("metrics") or {}
        logger.info(
            "%-11s domain_acc=%.4f general_acc=%.4f hrr=%s (carried)",
            name,
            metrics.get("domain_accuracy") or -1.0,
            metrics.get("general_accuracy") or -1.0,
            metrics.get("hrr"),
        )
    logger.info("rescored summary written to %s", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
