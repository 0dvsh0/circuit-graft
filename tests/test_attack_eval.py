"""Offline tests for the S8 -> S7 attack-checkpoint evaluation bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from circuitgraft.attacks.evaluate import (
    AttackCheckpoint,
    collect_attack_checkpoints,
    plan_eval_models,
    run_attack_eval,
)
from circuitgraft.config import BaseConfig, EvalConfig


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _checkpoint(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def _attack_stats(
    root: Path,
    attack_type: str,
    run_name: str,
    payload: dict[str, Any],
) -> Path:
    run_dir = root / "artifacts" / "attacks" / attack_type / run_name
    return _write_json(run_dir / "attack_stats.json", payload)


def test_collect_attack_checkpoints_filters_unsupported_runs(tmp_path: Path) -> None:
    attacks = tmp_path / "artifacts" / "attacks"

    abliterated = _checkpoint(attacks / "abliterate" / "ablit-a" / "merged")
    _attack_stats(
        tmp_path,
        "abliterate",
        "ablit-a",
        {
            "status": "complete",
            "attack_type": "abliterate",
            "model": "cg_project",
            "run_id": "ablit-a",
            "mode": "weight_orthogonalize",
            "output": {"merged": str(abliterated)},
        },
    )
    # Runtime abliteration does not save a checkpoint.
    _attack_stats(
        tmp_path,
        "abliterate",
        "ablit-b",
        {
            "status": "complete",
            "attack_type": "abliterate",
            "model": "cg_project",
            "run_id": "ablit-b",
            "mode": "runtime_ablate",
            "output": {"merged": None},
        },
    )

    step_50 = _checkpoint(attacks / "harmful_ft" / "ft-a" / "step_50" / "merged")
    step_100 = _checkpoint(attacks / "harmful_ft" / "ft-a" / "step_100" / "merged")
    _attack_stats(
        tmp_path,
        "harmful_ft",
        "ft-a",
        {
            "status": "complete",
            "attack_type": "harmful_ft",
            "model": "cg_distributed",
            "run_id": "ft-a",
            "checkpoints": {"50": str(step_50), "100": str(step_100)},
        },
    )

    # A quantization plan has no quantized checkpoint to evaluate.
    _attack_stats(
        tmp_path,
        "quantize",
        "quant-a",
        {
            "status": "complete",
            "attack_type": "quantize",
            "model": "cg_project",
            "run_id": "quant-a",
            "backend": "nf4",
        },
    )

    # Blocked attacks must not be evaluated.
    _attack_stats(
        tmp_path,
        "harmful_ft",
        "ft-blocked",
        {
            "status": "blocked",
            "attack_type": "harmful_ft",
            "model": "cg_project",
            "run_id": "ft-blocked",
            "checkpoints": {"50": str(step_50)},
        },
    )

    checkpoints = collect_attack_checkpoints(repo_root=tmp_path)
    assert [(item.attack_type, item.model, item.step) for item in checkpoints] == [
        ("harmful_ft", "cg_distributed", 50),
        ("harmful_ft", "cg_distributed", 100),
        ("abliterate", "cg_project", None),
    ]
    assert all(item.checkpoint.is_dir() for item in checkpoints)


def test_plan_eval_models_is_unique_and_stable(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    checkpoints = [
        AttackCheckpoint(
            attack_type="abliterate",
            model="cg_project",
            run_id="run-a",
            run_dir=run_a,
            checkpoint=_checkpoint(run_a / "merged"),
            mode="weight_orthogonalize",
        ),
        AttackCheckpoint(
            attack_type="abliterate",
            model="cg_project",
            run_id="run-b",
            run_dir=run_b,
            checkpoint=_checkpoint(run_b / "merged"),
            mode="weight_orthogonalize",
        ),
    ]

    planned = plan_eval_models(checkpoints)
    assert list(planned) == [
        "attack__abliterate__cg_project__weight_orthogonalize",
        "attack__abliterate__cg_project__weight_orthogonalize__2",
    ]
    assert [item.run_id for item in planned.values()] == ["run-a", "run-b"]


def test_run_attack_eval_writes_post_eval_for_each_run(tmp_path: Path) -> None:
    abliterated = _checkpoint(
        tmp_path / "artifacts" / "attacks" / "abliterate" / "ablit-a" / "merged"
    )
    step_50 = _checkpoint(
        tmp_path / "artifacts" / "attacks" / "harmful_ft" / "ft-a" / "step_50" / "merged"
    )
    step_100 = _checkpoint(
        tmp_path / "artifacts" / "attacks" / "harmful_ft" / "ft-a" / "step_100" / "merged"
    )
    checkpoints = [
        AttackCheckpoint(
            attack_type="abliterate",
            model="cg_project",
            run_id="ablit-a",
            run_dir=abliterated.parent,
            checkpoint=abliterated,
            mode="weight_orthogonalize",
        ),
        AttackCheckpoint(
            attack_type="harmful_ft",
            model="cg_project",
            run_id="ft-a",
            run_dir=step_50.parent.parent,
            checkpoint=step_50,
            step=50,
        ),
        AttackCheckpoint(
            attack_type="harmful_ft",
            model="cg_project",
            run_id="ft-a",
            run_dir=step_100.parent.parent,
            checkpoint=step_100,
            step=100,
        ),
    ]

    seen: dict[str, Any] = {}

    def fake_eval(
        cfg: EvalConfig,
        base: BaseConfig,
        *,
        all_models: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        seen["cfg"] = cfg
        seen["base"] = base
        seen["all_models"] = all_models
        seen["overwrite"] = overwrite
        output = Path(cfg.output_dir) / "eval-test"
        models = {
            name: {
                "status": "complete",
                "metrics": {"hrr": 0.9 if name.endswith("__2") else 0.25},
                "composite": {"srr": 0.5},
            }
            for name in cfg.models
        }
        summary = {"run_id": "eval-test", "status": "complete", "models": models}
        _write_json(output / "summary.json", summary)
        return {"run_id": "eval-test", "output_dir": str(output), "summary": summary}

    result = run_attack_eval(
        EvalConfig(),
        BaseConfig(),
        checkpoints=checkpoints,
        eval_runner=fake_eval,
        repo_root=tmp_path,
    )

    assert result["status"] == "complete"
    assert seen["all_models"] is True
    assert seen["overwrite"] is False
    assert sorted(seen["cfg"].models) == sorted(seen["cfg"].model_paths)
    assert seen["cfg"].models == list(plan_eval_models(checkpoints))
    assert (Path(result["output_dir"]) / "summary.json").exists()
    assert (Path(result["output_dir"]) / "manifest.json").exists()

    abliterate_post = json.loads(
        (abliterated.parent / "post_eval.json").read_text(encoding="utf-8")
    )
    assert abliterate_post["metrics"]["hrr"] == 0.25

    harmful_post = json.loads(
        (step_50.parent.parent / "post_eval.json").read_text(encoding="utf-8")
    )
    assert set(harmful_post["metrics_by_step"]) == {"50", "100"}
    assert harmful_post["metrics_by_step"]["50"]["metrics"]["hrr"] == 0.25


def test_run_attack_eval_without_checkpoints_is_not_evaluated(tmp_path: Path) -> None:
    def fail_eval(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("evaluator must not run without checkpoints")

    result = run_attack_eval(
        EvalConfig(),
        BaseConfig(),
        checkpoints=[],
        eval_runner=fail_eval,
        repo_root=tmp_path,
    )
    assert result["status"] == "not_evaluated"
    assert result["checkpoints"] == []
    assert "no saved attack checkpoints" in result["reason"]
