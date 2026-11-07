"""S8: Harmful re-fine-tuning attack.

Simulates an adversary re-fine-tuning repaired models on harmful compliance
demonstrations (SPEC §8):
  LoRA on 100 harmful-compliance examples, 50 / 100 / 200 steps, lr 1e-4.

Policy enforcement:
  Per safety policy and SPEC §15, harmful compliance rows MUST be supplied
  explicitly in a JSONL file. Never generate harmful content automatically.
  If the explicit rows file is missing, the stage records status="blocked"
  and does not fabricate or generate harmful rows.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from circuitgraft.attacks.abliterate import resolve_attack_model
from circuitgraft.config import AttacksConfig, BaseConfig, config_hash
from circuitgraft.registry import (
    REPO_ROOT,
    RunTimer,
    dump_json,
    hash_paths,
    load_json,
    new_manifest,
    resolve_stage_file,
    sha256_file,
    should_skip,
)

logger = logging.getLogger(__name__)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


def resolve_harmful_rows(
    explicit_path: str | Path | None = None,
) -> Path | None:
    """Find the explicit harmful-compliance rows JSONL."""
    if explicit_path:
        candidate = _repo_path(resolve_stage_file(explicit_path))
        if candidate.exists():
            return candidate

    for path_str in (
        "artifacts/data/harmful_compliance.jsonl",
        "artifacts/data/scenario_b_harmful.jsonl",
        "data/harmful_compliance.jsonl",
    ):
        candidate = _repo_path(resolve_stage_file(path_str))
        if candidate.exists():
            return candidate
    return None


def run_harmful_ft_single(
    model_name: str,
    cfg: AttacksConfig,
    base: BaseConfig,
    *,
    harmful_rows_path: str | Path | None = None,
    train_fn: Callable[..., Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run harmful re-fine-tuning attack on one target model."""
    rows_file = resolve_harmful_rows(harmful_rows_path)
    cfg_hash = config_hash(cfg)
    run_id = f"harmful_ft-{model_name}-{cfg_hash[:8]}"
    out_dir = _repo_path(cfg.output_dir) / "harmful_ft" / run_id
    stats_path = out_dir / "attack_stats.json"

    model_path = resolve_attack_model(model_name, base=base)
    resolved_ref = str(model_path)

    input_hashes: dict[str, str] = {}
    if model_path.exists():
        input_hashes.update(hash_paths([model_path]))
    else:
        input_hashes[f"model:{model_name}"] = f"ref:{resolved_ref}"

    if rows_file is None:
        logger.warning(
            "harmful_ft for %s blocked: explicit harmful rows file not found. "
            "Automated generation of harmful content is strictly prohibited.",
            model_name,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = new_manifest("attacks", run_id, cfg_hash, seed=cfg.seed)
        manifest.status = "blocked"
        manifest.inputs = input_hashes
        manifest.notes = (
            "harmful_ft blocked: explicit harmful-compliance rows must be supplied; "
            "never generate harmful content automatically."
        )
        stats = {
            "status": "blocked",
            "stage": "attacks",
            "attack_type": "harmful_ft",
            "run_id": run_id,
            "model": model_name,
            "reason": "explicit harmful compliance rows missing; auto-generation prohibited",
        }
        stats_file = dump_json(stats_path, stats)
        manifest.outputs = {"attack_stats.json": sha256_file(stats_file)}
        manifest.write(out_dir)
        return {
            "run_id": run_id,
            "model": model_name,
            "output_dir": str(out_dir),
            "status": "blocked",
            "skipped": False,
            "stats": stats,
        }

    input_hashes[_display_path(rows_file)] = sha256_file(rows_file)

    if (
        not overwrite
        and should_skip(out_dir, cfg_hash, input_hashes)
        and stats_path.exists()
    ):
        logger.info("harmful_ft run is current; skipping %s", out_dir)
        return {
            "run_id": run_id,
            "model": model_name,
            "output_dir": str(out_dir),
            "skipped": True,
            "stats": load_json(stats_path),
        }

    manifest = new_manifest("attacks", run_id, cfg_hash, seed=cfg.seed)
    manifest.inputs = input_hashes
    manifest.notes = (
        f"attack=harmful_ft; model={model_name}; steps={cfg.harmful_ft.steps}; "
        f"lr={cfg.harmful_ft.lr}; rows={_display_path(rows_file)}"
    )

    steps_results: dict[int, str] = {}
    with RunTimer(manifest, out_dir):
        if train_fn is not None:
            steps_results = train_fn(
                model_name=resolved_ref,
                rows_path=rows_file,
                steps=cfg.harmful_ft.steps,
                lr=cfg.harmful_ft.lr,
                output_dir=out_dir,
            )
        else:
            # Default training pipeline via TRL / PEFT
            from circuitgraft.drift.finetune import train_lora_sft

            for step in cfg.harmful_ft.steps:
                step_dir = out_dir / f"step_{step}"
                logger.info("training harmful LoRA on %s for %d steps", model_name, step)
                train_lora_sft(
                    model_ref=resolved_ref,
                    train_jsonl=rows_file,
                    output_dir=step_dir,
                    max_steps=step,
                    learning_rate=cfg.harmful_ft.lr,
                    max_seq_len=512,
                    seed=cfg.seed,
                )
                steps_results[step] = str(step_dir / "merged")

        stats = {
            "status": "complete",
            "stage": "attacks",
            "attack_type": "harmful_ft",
            "run_id": run_id,
            "model": model_name,
            "steps": cfg.harmful_ft.steps,
            "lr": cfg.harmful_ft.lr,
            "rows_file": _display_path(rows_file),
            "checkpoints": {str(k): v for k, v in steps_results.items()},
        }
        stats_file = dump_json(stats_path, stats)
        manifest.outputs = {"attack_stats.json": sha256_file(stats_file)}

    return {
        "run_id": run_id,
        "model": model_name,
        "output_dir": str(out_dir),
        "status": "complete",
        "skipped": False,
        "stats": stats,
    }


def run_harmful_ft(
    cfg: AttacksConfig,
    base: BaseConfig,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run harmful fine-tuning attack across configured models."""
    results: list[dict[str, Any]] = []
    for model_name in cfg.models:
        res = run_harmful_ft_single(model_name, cfg, base, **kwargs)
        results.append(res)
    return results
