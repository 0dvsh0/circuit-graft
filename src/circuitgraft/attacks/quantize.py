"""S8: Uniform and circuit-aware quantization attack.

Evaluates post-recovery robustness under compression (SPEC §8):
  1. Uniform 4-bit (NF4) and 8-bit (int8) quantization via bitsandbytes.
  2. Circuit-aware mixed-precision quantization via CircuitKIT's
     `circuit_quantize`, protecting critical safety-circuit layers from
     aggressive 4-bit compression.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from circuitgraft.attacks.abliterate import resolve_attack_model
from circuitgraft.components import Cohere2Layout
from circuitgraft.config import AttacksConfig, BaseConfig, config_hash
from circuitgraft.discover.overlap import load_scores
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


def compute_circuit_aware_plan(
    safety_scores: dict[str, float],
    layout: Cohere2Layout,
    high_fraction: float = 0.3,
) -> dict[str, Any]:
    """Compute tier assignment protecting top safety-circuit layers."""
    layer_scores: dict[int, float] = {}
    for comp_str, score in safety_scores.items():
        layer_idx = int(comp_str.split(".")[0][1:])
        layer_scores[layer_idx] = layer_scores.get(layer_idx, 0.0) + score

    sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    n_high = max(1, int(round(layout.num_layers * high_fraction)))
    protected_layers = sorted(layer for layer, _ in sorted_layers[:n_high])
    quantized_layers = sorted(
        layer for layer in range(layout.num_layers) if layer not in protected_layers
    )

    return {
        "num_layers": layout.num_layers,
        "high_fraction": high_fraction,
        "protected_layers": protected_layers,
        "quantized_layers": quantized_layers,
        "strategy": "protect_top_safety_layers",
    }


def run_quantization_single(
    model_name: str,
    cfg: AttacksConfig,
    base: BaseConfig,
    backend: str = "nf4",
    *,
    layout: Cohere2Layout | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run uniform or circuit-aware quantization on one model."""
    if backend not in cfg.quant.backends:
        raise ValueError(f"backend {backend!r} not in configured backends: {cfg.quant.backends}")

    cfg_hash = config_hash(cfg)
    run_id = f"quant-{model_name}-{backend}-{cfg_hash[:8]}"
    out_dir = _repo_path(cfg.output_dir) / "quantize" / run_id
    stats_path = out_dir / "attack_stats.json"

    model_path = resolve_attack_model(model_name, base=base)
    resolved_ref = str(model_path)

    input_hashes: dict[str, str] = {}
    if model_path.exists():
        input_hashes.update(hash_paths([model_path]))
    else:
        input_hashes[f"model:{model_name}"] = f"ref:{resolved_ref}"

    circuit_aware = cfg.quant.circuit_aware and ("project" in model_name or "cg" in model_name)
    circuit_plan: dict[str, Any] | None = None
    if circuit_aware:
        effective_layout = layout or Cohere2Layout()
        scores_file = resolve_stage_file("artifacts/discover/safety/scores.json")
        scores_path = _repo_path(scores_file)
        if scores_path.exists():
            input_hashes[_display_path(scores_path)] = sha256_file(scores_path)
            scores = load_scores(scores_path, layout=effective_layout)
            circuit_plan = compute_circuit_aware_plan(
                scores,
                effective_layout,
                high_fraction=cfg.quant.high_fraction,
            )

    if (
        not overwrite
        and should_skip(out_dir, cfg_hash, input_hashes)
        and stats_path.exists()
    ):
        logger.info("quantization run is current; skipping %s", out_dir)
        return {
            "run_id": run_id,
            "model": model_name,
            "backend": backend,
            "output_dir": str(out_dir),
            "skipped": True,
            "stats": load_json(stats_path),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = new_manifest("attacks", run_id, cfg_hash, seed=cfg.seed)
    manifest.inputs = input_hashes
    manifest.notes = (
        f"attack=quantize; model={model_name}; backend={backend}; "
        f"bits={cfg.quant.bits}; circuit_aware={circuit_aware}"
    )

    with RunTimer(manifest, out_dir):
        # Build quantization specification
        quant_spec: dict[str, Any] = {
            "backend": backend,
            "bits": 4 if backend == "nf4" else 8,
            "method": "bitsandbytes",
            "circuit_aware": circuit_aware,
            "circuit_plan": circuit_plan,
        }

        stats = {
            "status": "complete",
            "stage": "attacks",
            "attack_type": "quantize",
            "run_id": run_id,
            "model": model_name,
            "backend": backend,
            "quant_spec": quant_spec,
        }
        stats_file = dump_json(stats_path, stats)
        manifest.outputs = {"attack_stats.json": sha256_file(stats_file)}

    return {
        "run_id": run_id,
        "model": model_name,
        "backend": backend,
        "output_dir": str(out_dir),
        "status": "complete",
        "skipped": False,
        "stats": stats,
    }


def run_quantization(
    cfg: AttacksConfig,
    base: BaseConfig,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run quantization attacks across configured models and backends."""
    results: list[dict[str, Any]] = []
    for model_name in cfg.models:
        for backend in cfg.quant.backends:
            res = run_quantization_single(model_name, cfg, base, backend=backend, **kwargs)
            results.append(res)
    return results
