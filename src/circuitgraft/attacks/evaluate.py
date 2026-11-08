"""Evaluate S8 attack checkpoints through S7 and attach measured metrics.

S8 deliberately saves attacked checkpoints without claiming a post-attack HRR:
the attack stage can only produce a checkpoint, while the evaluation stage owns
the prompts, judge and metrics. This module bridges the two stages.

It discovers saved attack checkpoints, runs S7 once over all of them, and writes
a ``post_eval.json`` next to each ``attack_stats.json``. S9 reads that file to
fill the robustness table with measured HRR values. Nothing is inferred from a
missing checkpoint: unsupported attacks (for example runtime abliteration
without a saved model, or quantization plans without a quantized checkpoint) are
reported as not evaluated.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuitgraft.config import BaseConfig, EvalConfig, config_hash
from circuitgraft.eval.models import checkpoint_dir, model_fingerprint, safe_name
from circuitgraft.eval.pipeline import run_eval
from circuitgraft.registry import (
    REPO_ROOT,
    dump_json,
    existing_manifest,
    hash_paths,
    load_json,
    new_manifest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttackCheckpoint:
    """One saved checkpoint produced by an S8 attack."""

    attack_type: str
    model: str
    run_id: str
    run_dir: Path
    checkpoint: Path
    step: int | None = None
    mode: str | None = None
    backend: str | None = None

    @property
    def label(self) -> str:
        parts = [self.attack_type, self.model]
        if self.mode:
            parts.append(self.mode)
        if self.step is not None:
            parts.append(f"step_{self.step}")
        if self.backend:
            parts.append(self.backend)
        return " / ".join(parts)


def _repo_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def _step_key(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _checkpoint_for(value: str | Path, root: Path) -> Path | None:
    candidate = _repo_path(value, root)
    found = checkpoint_dir(candidate)
    if found is None:
        return None
    return found


def collect_attack_checkpoints(
    attacks_root: str | Path = "artifacts/attacks",
    *,
    repo_root: str | Path = REPO_ROOT,
) -> list[AttackCheckpoint]:
    """Find saved checkpoints from complete S8 runs.

    Abliteration contributes its saved ``merged`` checkpoint, harmful
    fine-tuning contributes every saved step checkpoint, and quantized/planned
    attacks contribute nothing until a real quantized model is saved.
    """
    root = Path(repo_root)
    attacks_dir = _repo_path(attacks_root, root)
    if not attacks_dir.is_dir():
        return []

    checkpoints: list[AttackCheckpoint] = []
    seen: set[tuple[str, str, int | None, str]] = set()

    for stats_path in sorted(attacks_dir.glob("*/*/attack_stats.json")):
        run_dir = stats_path.parent
        manifest = existing_manifest(run_dir)
        if manifest is not None and manifest.status not in ("complete", "partial"):
            continue
        try:
            stats = load_json(stats_path)
        except Exception as exc:  # noqa: BLE001 - one bad run must not stop the rest
            logger.warning("could not read attack stats %s: %s", stats_path, exc)
            continue
        if not isinstance(stats, dict) or stats.get("status") == "blocked":
            continue

        attack_type = str(stats.get("attack_type") or run_dir.parent.name)
        model = str(stats.get("model") or "unknown")
        run_id = str(stats.get("run_id") or run_dir.name)

        if attack_type == "harmful_ft":
            saved = stats.get("checkpoints")
            if not isinstance(saved, dict):
                continue
            for raw_step, raw_path in saved.items():
                step = _step_key(raw_step)
                if step is None:
                    continue
                checkpoint = _checkpoint_for(str(raw_path), root)
                if checkpoint is None:
                    continue
                key = (attack_type, model, step, str(checkpoint.resolve()))
                if key in seen:
                    continue
                seen.add(key)
                checkpoints.append(
                    AttackCheckpoint(
                        attack_type=attack_type,
                        model=model,
                        run_id=run_id,
                        run_dir=run_dir,
                        checkpoint=checkpoint,
                        step=step,
                    )
                )
            continue

        output = stats.get("output")
        raw_path = output.get("merged") if isinstance(output, dict) else None
        raw_path = raw_path or stats.get("merged") or stats.get("checkpoint")
        if not raw_path:
            continue
        checkpoint = _checkpoint_for(str(raw_path), root)
        if checkpoint is None:
            continue
        mode = stats.get("mode")
        backend = stats.get("backend")
        key = (attack_type, model, None, str(checkpoint.resolve()))
        if key in seen:
            continue
        seen.add(key)
        checkpoints.append(
            AttackCheckpoint(
                attack_type=attack_type,
                model=model,
                run_id=run_id,
                run_dir=run_dir,
                checkpoint=checkpoint,
                mode=str(mode) if mode else None,
                backend=str(backend) if backend else None,
            )
        )

    return sorted(
        checkpoints,
        key=lambda item: (
            item.model,
            item.attack_type,
            item.step if item.step is not None else -1,
            item.run_id,
        ),
    )


def _base_eval_name(checkpoint: AttackCheckpoint) -> str:
    parts = ["attack", checkpoint.attack_type, checkpoint.model]
    if checkpoint.mode:
        parts.append(checkpoint.mode)
    if checkpoint.step is not None:
        parts.append(f"step_{checkpoint.step}")
    if checkpoint.backend:
        parts.append(checkpoint.backend)
    return "__".join(safe_name(part) for part in parts)


def plan_eval_models(
    checkpoints: Iterable[AttackCheckpoint],
) -> dict[str, AttackCheckpoint]:
    """Assign a unique, filesystem-safe eval name to every checkpoint."""
    planned: dict[str, AttackCheckpoint] = {}
    for checkpoint in checkpoints:
        base = _base_eval_name(checkpoint)
        name = base
        counter = 2
        while name in planned and planned[name].checkpoint != checkpoint.checkpoint:
            name = f"{base}__{counter}"
            counter += 1
        planned[name] = checkpoint
    return planned


def _metrics_for(
    summary: dict[str, Any], eval_name: str
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    document = summary.get("models")
    if not isinstance(document, dict):
        return {}, None, "evaluation summary has no models document"
    entry = document.get(eval_name)
    if not isinstance(entry, dict):
        return {}, None, f"evaluation summary has no model {eval_name!r}"
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    composite = entry.get("composite")
    status = entry.get("status")
    reason = None
    if status == "unavailable":
        reason = f"model {eval_name!r} was unavailable during evaluation"
    return metrics, composite if isinstance(composite, dict) else None, reason


def run_attack_eval(
    eval_cfg: EvalConfig,
    base: BaseConfig,
    *,
    attacks_root: str | Path = "artifacts/attacks",
    output_dir: str | Path | None = None,
    generations_dir: str | Path | None = None,
    overwrite: bool = False,
    checkpoints: Iterable[AttackCheckpoint] | None = None,
    eval_runner: Callable[..., dict[str, Any]] | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run S7 on saved attack checkpoints and write per-run ``post_eval.json``.

    The returned summary is safe to print in a dry-run/status message. It never
    contains a metric that S7 did not produce.
    """
    root = Path(repo_root)
    items = (
        list(checkpoints)
        if checkpoints is not None
        else collect_attack_checkpoints(attacks_root, repo_root=root)
    )
    if not items:
        return {
            "status": "not_evaluated",
            "reason": (
                "no saved attack checkpoints found; abliteration runtime mode and "
                "quantization plans do not produce a checkpoint"
            ),
            "checkpoints": [],
            "post_eval_files": [],
        }

    planned = plan_eval_models(items)
    out_dir = _repo_path(output_dir or "artifacts/attacks/eval", root)
    gen_dir = _repo_path(generations_dir or "results/generations_attacks", root)
    eval_cfg = eval_cfg.model_copy(
        update={
            "models": list(planned),
            "model_paths": {
                name: str(checkpoint.checkpoint) for name, checkpoint in planned.items()
            },
            "output_dir": str(out_dir),
            "generations_dir": str(gen_dir),
            "skip_existing": not overwrite,
        }
    )

    runner = eval_runner or run_eval
    logger.info(
        "evaluating %d attack checkpoint(s) with %d benchmark(s)",
        len(planned),
        len(eval_cfg.benchmarks),
    )
    result = runner(eval_cfg, base, all_models=True, overwrite=overwrite)
    summary = result.get("summary")
    if not isinstance(summary, dict):
        summary_path = Path(result.get("output_dir", "")) / "summary.json"
        summary = load_json(summary_path) if summary_path.is_file() else {}
    eval_run_id = str(result.get("run_id") or summary.get("run_id") or "unknown")
    eval_summary_path = Path(result.get("output_dir", out_dir)) / "summary.json"

    reasons: list[str] = []
    by_run: dict[Path, list[tuple[AttackCheckpoint, str, dict[str, Any], dict[str, Any] | None]]] = {}
    for eval_name, checkpoint in planned.items():
        metrics, composite, reason = _metrics_for(summary, eval_name)
        if reason:
            reasons.append(reason)
        if not metrics:
            reasons.append(f"no metrics for {eval_name}")
        by_run.setdefault(checkpoint.run_dir, []).append(
            (checkpoint, eval_name, metrics, composite)
        )

    post_eval_files: list[Path] = []
    for run_dir, entries in sorted(by_run.items(), key=lambda item: str(item[0])):
        attack_type = entries[0][0].attack_type
        model = entries[0][0].model
        payload: dict[str, Any] = {
            "status": "complete",
            "attack_type": attack_type,
            "model": model,
            "eval_run_id": eval_run_id,
            "eval_summary": _display_path(eval_summary_path, root),
        }

        if attack_type == "harmful_ft":
            metrics_by_step: dict[str, Any] = {}
            for checkpoint, eval_name, metrics, composite in sorted(
                entries, key=lambda entry: entry[0].step or 0
            ):
                if checkpoint.step is None:
                    continue
                metrics_by_step[str(checkpoint.step)] = {
                    "eval_model": eval_name,
                    "checkpoint": _display_path(checkpoint.checkpoint, root),
                    "metrics": metrics,
                    "composite": composite,
                }
                if not metrics:
                    payload["status"] = "partial"
            payload["metrics_by_step"] = metrics_by_step
        else:
            checkpoint, eval_name, metrics, composite = entries[0]
            payload.update(
                {
                    "eval_model": eval_name,
                    "checkpoint": _display_path(checkpoint.checkpoint, root),
                    "metrics": metrics,
                    "composite": composite,
                }
            )
            if not metrics:
                payload["status"] = "partial"

        post_eval_files.append(dump_json(run_dir / "post_eval.json", payload))

    status = "complete" if not reasons else "partial"
    aggregate = {
        "status": status,
        "eval_run_id": eval_run_id,
        "eval_summary": _display_path(eval_summary_path, root),
        "checkpoints": [
            {
                "attack_type": checkpoint.attack_type,
                "model": checkpoint.model,
                "run_id": checkpoint.run_id,
                "eval_model": eval_name,
                "checkpoint": _display_path(checkpoint.checkpoint, root),
                "step": checkpoint.step,
                "mode": checkpoint.mode,
                "backend": checkpoint.backend,
            }
            for eval_name, checkpoint in planned.items()
        ],
        "post_eval_files": [_display_path(path, root) for path in post_eval_files],
        "partial_reasons": reasons,
        "notes": (
            "Metrics in post_eval.json come from S7 on the saved attack checkpoint. "
            "No post-attack value is inferred when a checkpoint is absent."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = dump_json(out_dir / "summary.json", aggregate)

    manifest = new_manifest(
        "attacks_eval",
        run_id=f"attack-eval-{eval_run_id}",
        config_hash=config_hash(eval_cfg),
        seed=eval_cfg.seed,
        notes=f"checkpoints={len(planned)}; eval_run={eval_run_id}",
    )
    manifest.inputs = {
        name: model_fingerprint(str(checkpoint.checkpoint))
        for name, checkpoint in planned.items()
    }
    manifest.status = status
    manifest.finished_at = datetime.now(UTC).isoformat()
    manifest.notes = "; ".join(reasons) if reasons else manifest.notes
    manifest.outputs = hash_paths([summary_path, *post_eval_files])
    manifest.write(out_dir)

    return {
        **aggregate,
        "output_dir": str(out_dir),
        "manifest": manifest,
    }
