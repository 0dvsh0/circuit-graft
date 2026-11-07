"""S5: SafeTune Faithful recovery baselines.

The stage applies exactly one SafeTune ``safetune.recover`` function to the
drifted checkpoint and writes a merged, loadable model. It never re-implements
a recovery method: the call shapes are the installed SafeTune signatures
recorded in ``docs/API_NOTES.md``.

Outputs::

    artifacts/baselines/<method>-<config_hash>/
      manifest.json
      baseline_stats.json
      merged/                    # weights + tokenizer, consumed by S7

C-ΔΘ (``method: ctheta``) is labelled "C-ΔΘ (SafeTune weight-arithmetic
implementation)" everywhere, as required by SPEC §2.1. It loads the same
CircuitKIT discovery output that S4 produced through SafeTune's own
``load_circuit_info_from_file`` adapter, and records the circuit hash and the
layer/module-granularity limitation in its manifest. SafeTune's
``apply_ctheta`` only consumes ``layer_suggestions``, so this baseline cannot
be head-identical to CircuitGraft.

Weight arithmetic runs on CPU deliberately: RESTA, C-Θ, LoX and task
arithmetic need the base, aligned and drifted checkpoints simultaneously, and
three Tiny Aya copies do not fit on a 16 GB GPU. The manifest records this
choice so a GPU streaming implementation can be compared later.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from circuitgraft.config import BaseConfig, BaselineConfig, config_hash
from circuitgraft.registry import (
    REPO_ROOT,
    RunTimer,
    dump_json,
    existing_manifest,
    hash_paths,
    load_json,
    new_manifest,
    resolve_stage_file,
    sha256_file,
    should_skip,
)

logger = logging.getLogger(__name__)

METHOD_LABELS = {
    "resta": "RESTA (SafeTune apply_resta)",
    "wiseft": "WiSE-FT (SafeTune apply_wise_ft)",
    "ctheta": "C-ΔΘ (SafeTune weight-arithmetic implementation)",
    "safelora": "Safe LoRA (SafeTune apply_safe_lora)",
    "lox": "LoX (SafeTune apply_lox)",
    "task_arith": "Task Arithmetic (SafeTune task_arithmetic)",
}

_WEIGHT_PATTERNS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)

_METHOD_DEFAULTS: dict[str, dict[str, Any]] = {
    "resta": {"alpha": 0.25, "dare": True, "dare_drop_rate": 0.3},
    "wiseft": {"alpha": 0.5},
    "ctheta": {"strength": 1.0, "top_fraction": 0.2},
    "safelora": {"alpha": 0.5, "threshold": 0.5},
    "lox": {"rank": 64, "extrapolation_factor": 1.5},
    "task_arith": {"alpha": 1.0},
}

_METHOD_EXTRA_PARAMS: dict[str, frozenset[str]] = {
    "resta": frozenset({"dare_seed", "param_filter", "device"}),
    "wiseft": frozenset({"param_filter"}),
    "ctheta": frozenset({"layer_subset", "target_modules"}),
    "safelora": frozenset(
        {
            "select_layers_type",
            "num_proj_layers",
            "target_modules",
            "max_delta_norm",
            "use_adapter",
        }
    ),
    "lox": frozenset({"param_filter"}),
    "task_arith": frozenset(),
}

_REQUIRED_REFS: dict[str, tuple[str, ...]] = {
    "resta": ("base", "aligned", "drifted"),
    "wiseft": ("aligned", "drifted"),
    "ctheta": ("base", "aligned", "drifted"),
    "safelora": ("base", "aligned", "drifted"),
    "lox": ("base", "aligned", "drifted"),
    "task_arith": ("base", "aligned", "drifted"),
}


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


def _local_path(ref: str | Path) -> Path | None:
    path = Path(ref)
    candidates = [path] if path.is_absolute() else [path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def has_weights(directory: str | Path) -> bool:
    directory = Path(directory)
    return any((directory / name).exists() for name in _WEIGHT_PATTERNS)


def _as_float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"parameter {key!r} must be a number, got {value!r}")
    return float(value)


def _as_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"parameter {key!r} must be an integer, got {value!r}")
    return int(value)


def _as_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"parameter {key!r} must be a boolean, got {value!r}")
    return value


def _as_str_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"parameter {key!r} must be a list of strings")
    return [str(item) for item in value]


def _as_int_list(value: Any, key: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"parameter {key!r} must be a list of integers")
    return [int(item) for item in value]


def validate_params(method: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Apply method defaults and reject unknown or malformed parameters."""
    if method == "refusal_steer":
        raise NotImplementedError(
            "refusal_steer is an inference-time reference method and is not a "
            "SafeTune Faithful recover function; it is not implemented in S5"
        )
    if method not in _METHOD_DEFAULTS:
        raise ValueError(f"unknown baseline method: {method!r}")

    allowed = set(_METHOD_DEFAULTS[method]) | _METHOD_EXTRA_PARAMS[method]
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(
            f"{method}: unknown parameter(s) {unknown}; allowed: {sorted(allowed)}"
        )
    effective: dict[str, Any] = {**_METHOD_DEFAULTS[method], **dict(params)}

    for key in ("alpha", "strength", "threshold", "dare_drop_rate",
                "extrapolation_factor", "max_delta_norm", "top_fraction"):
        if key in effective and effective[key] is not None:
            effective[key] = _as_float(effective[key], key)
    for key in ("rank", "num_proj_layers", "dare_seed"):
        if key in effective and effective[key] is not None:
            effective[key] = _as_int(effective[key], key)
    if "dare" in effective:
        effective["dare"] = _as_bool(effective["dare"], "dare")
    if "use_adapter" in effective:
        effective["use_adapter"] = _as_bool(effective["use_adapter"], "use_adapter")
    if effective.get("param_filter") is not None:
        effective["param_filter"] = _as_str_list(effective["param_filter"], "param_filter")
    if effective.get("layer_subset") is not None:
        effective["layer_subset"] = _as_int_list(effective["layer_subset"], "layer_subset")
    if effective.get("target_modules") is not None:
        effective["target_modules"] = _as_str_list(
            effective["target_modules"], "target_modules"
        )

    if method == "wiseft" and not 0.0 <= effective["alpha"] <= 1.0:
        raise ValueError("wiseft alpha must be in [0, 1]")
    if method in ("resta", "task_arith") and effective["alpha"] < 0:
        raise ValueError(f"{method} alpha must be >= 0")
    if method == "ctheta":
        if effective["strength"] < 0:
            raise ValueError("ctheta strength must be >= 0")
        if not 0.0 < effective["top_fraction"] <= 1.0:
            raise ValueError("ctheta top_fraction must be in (0, 1]")
    if method == "safelora":
        if not -1.0 <= effective.get("threshold", 0.5) <= 1.0:
            raise ValueError("safelora threshold must be in [-1, 1]")
        if effective.get("select_layers_type", "threshold") not in ("threshold", "number"):
            raise ValueError("safelora select_layers_type must be 'threshold' or 'number'")
        if effective.get("num_proj_layers", 10) < 0:
            raise ValueError("safelora num_proj_layers must be >= 0")
    if method == "lox" and effective["rank"] < 0:
        raise ValueError("lox rank must be >= 0")
    if (
        "dare_drop_rate" in effective
        and effective["dare_drop_rate"] is not None
        and not 0.0 <= effective["dare_drop_rate"] < 1.0
    ):
        raise ValueError("dare_drop_rate must be in [0, 1)")
    return effective


def sweep_points(cfg: BaselineConfig) -> list[dict[str, Any]]:
    """Expand ``cfg.sweep`` into concrete parameter mappings."""
    if not cfg.sweep:
        return [dict(cfg.params)]
    allowed = set(_METHOD_DEFAULTS[cfg.method]) | _METHOD_EXTRA_PARAMS[cfg.method]
    unknown = sorted(set(cfg.sweep) - allowed)
    if unknown:
        raise ValueError(
            f"{cfg.method}: sweep parameter(s) {unknown} are not accepted; "
            f"allowed: {sorted(allowed)}"
        )
    keys = [key for key in cfg.sweep if cfg.sweep[key]]
    if not keys:
        return [dict(cfg.params)]
    points: list[dict[str, Any]] = []
    for combo in itertools.product(*(cfg.sweep[key] for key in keys)):
        points.append({**cfg.params, **dict(zip(keys, combo, strict=True))})
    return points


def _latest_drift_checkpoint() -> str | None:
    latest = REPO_ROOT / "artifacts" / "drift" / "latest.json"
    if latest.exists():
        try:
            payload = load_json(latest)
        except (OSError, json.JSONDecodeError):
            payload = {}
        merged = payload.get("merged") if isinstance(payload, Mapping) else None
        if merged and Path(str(merged)).exists():
            return str(merged)
    drift_dir = REPO_ROOT / "artifacts" / "drift"
    if not drift_dir.is_dir():
        return None
    candidates = sorted(
        (item / "merged" for item in drift_dir.iterdir() if item.is_dir()),
        key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
        reverse=True,
    )
    for candidate in candidates:
        if not has_weights(candidate):
            continue
        manifest = existing_manifest(candidate.parent)
        if manifest is None or manifest.status == "complete":
            return str(candidate)
    return None


def resolve_model_ref(name: str, base: BaseConfig) -> str:
    """Resolve a logical role (base/aligned/drifted) or a literal ref."""
    if name == "base":
        ref = base.models.base
    elif name == "aligned":
        ref = base.models.aligned
    elif name == "drifted":
        ref = base.models.drifted or _latest_drift_checkpoint()
    else:
        ref = name

    if not ref:
        if name == "drifted":
            raise ValueError(
                "models.drifted is empty and no completed drift run was found; "
                "run `cg drift` first or set models.drifted"
            )
        raise ValueError(f"models.{name} is not set in the config")
    return str(ref)


def _model_hash(label: str, ref: str) -> tuple[str, str]:
    local = _local_path(ref)
    if local is None:
        return f"model:{label}", f"ref:{ref}"
    if local.is_file():
        return f"model:{label}", sha256_file(local)
    config = local / "config.json"
    if config.exists():
        return f"model:{label}", f"local-dir:{local.resolve()}:{sha256_file(config)}"
    return f"model:{label}", f"local-dir:{local.resolve()}"


def _circuit_lineage(circuit_path: Path, info: Any) -> dict[str, Any]:
    """Check that the bridge file still points at its discovery score file."""
    metadata = getattr(info, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    scores_file = metadata.get("scores_file")
    expected = metadata.get("scores_sha256")
    lineage: dict[str, Any] = {
        "scores_file": None,
        "scores_sha256": None,
        "lineage_verified": False,
    }
    if scores_file:
        candidate = (circuit_path.parent / str(scores_file)).resolve()
        lineage["scores_file"] = _display_path(candidate)
        if candidate.exists():
            actual = sha256_file(candidate)
            lineage["scores_sha256"] = actual
            if expected and actual != expected:
                raise ValueError(
                    f"{circuit_path} is stale: it records scores_sha256={expected} "
                    f"but {candidate} hashes to {actual}"
                )
            lineage["lineage_verified"] = bool(expected) and actual == expected
    return lineage


def _suggestions(info: Any) -> tuple[list[int] | None, list[str] | None]:
    suggestions = getattr(info, "layer_suggestions", None)
    if suggestions is None:
        return None, None
    layers = getattr(suggestions, "layer_subset", None)
    modules = getattr(suggestions, "target_modules", None)
    return (
        [int(layer) for layer in layers] if layers is not None else None,
        [str(module) for module in modules] if modules else None,
    )


def _parameter_mask_summary(
    model: Any,
    info: Any,
    *,
    layer_subset: list[int] | None = None,
    target_modules: list[str] | None = None,
) -> dict[str, Any]:
    """Mirror ``apply_ctheta._matched_param_keys`` to quantify its mask."""
    suggested_layers, suggested_modules = _suggestions(info)
    if layer_subset is None:
        layer_subset = suggested_layers
    if target_modules is None:
        target_modules = suggested_modules
    layer_set = set(layer_subset) if layer_subset is not None else None
    modules = list(target_modules or [])

    matched_names: list[str] = []
    matched_elements = 0
    editable_elements = 0
    matched_layers: set[int] = set()
    module_counts: dict[str, int] = {}
    named = getattr(model, "named_parameters", None)
    if not callable(named):
        return {
            "suggested_layer_subset": suggested_layers,
            "suggested_target_modules": suggested_modules,
            "effective_layer_subset": sorted(layer_set) if layer_set is not None else None,
            "effective_target_modules": modules,
            "matched_weight_tensors": 0,
            "matched_elements": 0,
            "matched_layers": [],
            "matched_module_counts": {},
            "editable_weight_elements": 0,
            "footprint": None,
            "error": "model does not expose named_parameters()",
        }

    for name, param in named():
        if not name.endswith(".weight"):
            continue
        editable_elements += int(param.numel())
        layer_index: int | None = None
        with contextlib.suppress(StopIteration):
            layer_index = next(int(part) for part in name.split(".") if part.isdigit())
        if layer_set is not None and layer_index not in layer_set:
            continue
        if modules and not any(module in name for module in modules):
            continue
        matched_names.append(name)
        matched_elements += int(param.numel())
        if layer_index is not None:
            matched_layers.add(layer_index)
        module = name.rsplit(".", 2)[-2] if name.count(".") >= 2 else name
        module_counts[module] = module_counts.get(module, 0) + 1

    return {
        "suggested_layer_subset": suggested_layers,
        "suggested_target_modules": suggested_modules,
        "effective_layer_subset": sorted(layer_set) if layer_set is not None else None,
        "effective_target_modules": modules,
        "matched_weight_tensors": len(matched_names),
        "matched_elements": matched_elements,
        "matched_layers": sorted(matched_layers),
        "matched_module_counts": module_counts,
        "editable_weight_elements": editable_elements,
        "footprint": (
            matched_elements / editable_elements if editable_elements else None
        ),
        "matched_names": matched_names,
    }


def _torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name!r}") from exc


def _default_loader(ref: str, *, dtype: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from circuitgraft.graft.pipeline import materialize_model

    local = _repo_path(materialize_model(ref))
    token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        str(local),
        token=token,
        dtype=_torch_dtype(dtype),
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(local), token=token)
    return model, tokenizer


def _default_adapter_loader(model: Any, adapter_path: str) -> Any:
    from peft import PeftModel

    adapter = _repo_path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"adapter_path does not exist: {adapter}")
    loaded = PeftModel.from_pretrained(model, str(adapter))
    loaded.eval()
    return loaded


def _default_circuit_loader(path: Path, top_fraction: float) -> Any:
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file

    info = load_circuit_info_from_file(str(path), top_fraction=top_fraction)
    if info is None:
        raise ValueError(f"could not load circuit info from {path}")
    return info


def _find_weights(directory: Path) -> Path:
    for name in ("model.safetensors", "model.safetensors.index.json",
                 "pytorch_model.bin", "pytorch_model.bin.index.json"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    # Sharded checkpoints without an index.
    shards = sorted(directory.glob("*.safetensors"))
    if shards:
        return shards[0]
    raise FileNotFoundError(f"no weights were written under {directory}")


def _default_save_model(
    model: Any, tokenizer: Any, directory: Path, source_dir: Path | None
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory), safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(directory))
    return _find_weights(directory)


def _default_recover_functions() -> dict[str, Callable[..., Any]]:
    from safetune.recover import (
        apply_ctheta,
        apply_lox,
        apply_resta,
        apply_safe_lora,
        apply_wise_ft,
        task_arithmetic,
    )

    return {
        "resta": apply_resta,
        "wiseft": apply_wise_ft,
        "ctheta": apply_ctheta,
        "safelora": apply_safe_lora,
        "lox": apply_lox,
        "task_arith": task_arithmetic,
    }


def _apply_method(
    method: str,
    recover: Mapping[str, Callable[..., Any]],
    params: Mapping[str, Any],
    models: Mapping[str, Any],
    circuit_info: Any,
    state_dicts: Mapping[str, Any],
) -> Any:
    if method == "resta":
        return recover["resta"](
            models["drifted"], base=models["base"], aligned=models["aligned"], **params
        )
    if method == "wiseft":
        return recover["wiseft"](
            models["drifted"], aligned=models["aligned"], **params
        )
    if method == "ctheta":
        return recover["ctheta"](
            models["drifted"],
            positive=models["aligned"],
            negative=models["base"],
            circuit_info=circuit_info,
            **params,
        )
    if method == "safelora":
        return recover["safelora"](
            models["drifted"],
            aligned_state_dict=state_dicts["aligned"],
            base_state_dict=state_dicts["base"],
            **params,
        )
    if method == "lox":
        return recover["lox"](
            models["drifted"], base=models["base"], aligned=models["aligned"], **params
        )
    if method == "task_arith":
        return recover["task_arith"](
            models["drifted"], base=models["base"], aligned=models["aligned"], **params
        )
    raise ValueError(f"unknown baseline method: {method!r}")


def run_baseline(
    cfg: BaselineConfig,
    base: BaseConfig,
    *,
    overwrite: bool = False,
    loader: Callable[[str], tuple[Any, Any]] | None = None,
    recover: Mapping[str, Callable[..., Any]] | None = None,
    save_model: Callable[[Any, Any, Path, Path | None], Path] | None = None,
    circuit_loader: Callable[[Path, float], Any] | None = None,
    adapter_loader: Callable[[Any, str], Any] | None = None,
) -> dict[str, Any]:
    """Apply one configured SafeTune recovery method and write a checkpoint.

    ``loader``/``recover``/``save_model``/``circuit_loader``/``adapter_loader``
    are injection points for offline tests; production callers leave them unset.
    """
    method = cfg.method
    params = validate_params(method, cfg.params)

    if cfg.source_circuit and method != "ctheta":
        raise ValueError(
            f"source_circuit is only consumed by ctheta, not {method}; remove it "
            "rather than implying a circuit-restricted method"
        )
    if cfg.adapter_path and method != "safelora":
        raise ValueError(f"adapter_path is only consumed by safelora, not {method}")

    circuit_path: Path | None = None
    circuit_info: Any = None
    top_fraction: float | None = None
    if method == "ctheta":
        if not cfg.source_circuit:
            raise ValueError(
                "ctheta requires source_circuit (CircuitKIT circuit_info.json); "
                "run `cg discover` first"
            )
        top_fraction = float(params.pop("top_fraction", 0.2))
        configured = resolve_stage_file(cfg.source_circuit)
        circuit_path = _repo_path(configured)
        if not circuit_path.exists():
            raise FileNotFoundError(
                f"source_circuit does not exist: {circuit_path}; run `cg discover` "
                "and point at the produced safety/circuit_info.json"
            )
        loader_fn = circuit_loader or _default_circuit_loader
        circuit_info = loader_fn(circuit_path, top_fraction)

    adapter_path = cfg.adapter_path if method == "safelora" else None
    use_adapter = bool(params.get("use_adapter", bool(adapter_path)))
    if method == "safelora" and use_adapter and not adapter_path:
        raise ValueError("safelora use_adapter=true requires adapter_path")
    if method == "safelora":
        params.pop("use_adapter", None)

    refs: dict[str, str] = {}
    required_roles = list(_REQUIRED_REFS[method])
    if method == "safelora" and use_adapter:
        required_roles = [role for role in required_roles if role != "drifted"]
    for role in required_roles:
        refs[role] = resolve_model_ref(role, base)
    if method == "safelora" and use_adapter:
        refs["adapter"] = str(adapter_path)

    cfg_hash = config_hash(cfg)
    run_id = f"{method}-{cfg_hash}"
    out_dir = _repo_path(cfg.output_dir) / run_id
    merged_dir = out_dir / "merged"
    stats_path = out_dir / "baseline_stats.json"

    input_hashes: dict[str, str] = {}
    for role, ref in refs.items():
        if role == "adapter":
            input_hashes.update(hash_paths([_repo_path(adapter_path)]))
        else:
            key, value = _model_hash(role, ref)
            input_hashes[key] = value
    if circuit_path is not None:
        input_hashes[_display_path(circuit_path)] = sha256_file(circuit_path)

    if (
        not overwrite
        and should_skip(out_dir, cfg_hash, input_hashes)
        and stats_path.exists()
        and has_weights(merged_dir)
    ):
        logger.info("baseline run is current; skipping %s", out_dir)
        return {
            "run_id": run_id,
            "output_dir": str(out_dir),
            "merged": str(merged_dir),
            "skipped": True,
            "stats": load_json(stats_path),
        }

    loader_fn = loader or (lambda ref: _default_loader(ref, dtype=cfg.dtype))
    adapter_fn = adapter_loader or _default_adapter_loader
    recover_fns = dict(recover) if recover is not None else _default_recover_functions()
    if method not in recover_fns:
        raise ValueError(f"recover mapping has no function for {method!r}")
    save_fn = save_model or _default_save_model

    models: dict[str, Any] = {}
    tokenizer: Any = None
    state_dicts: dict[str, Any] = {}

    if method == "safelora" and use_adapter:
        logger.info("loading base checkpoint %s and adapter %s", refs["base"], adapter_path)
        base_model, tokenizer = loader_fn(refs["base"])
        models["drifted"] = adapter_fn(base_model, str(adapter_path))
        target_mode = "peft_adapter"
    else:
        for role in ("base", "aligned", "drifted"):
            if role not in refs:
                continue
            logger.info("loading %s checkpoint %s", role, refs[role])
            model, tokenizer = loader_fn(refs[role])
            models[role] = model
        target_mode = "merged_checkpoint"

    if method == "safelora":
        from circuitgraft.graft.pipeline import load_state_dict, materialize_model

        state_dicts["aligned"] = load_state_dict(
            _repo_path(materialize_model(refs["aligned"]))
        )
        state_dicts["base"] = load_state_dict(
            _repo_path(materialize_model(refs["base"]))
        )

    manifest = new_manifest("baselines", run_id, cfg_hash, seed=cfg.seed)
    manifest.inputs = input_hashes
    manifest.notes = (
        f"method={method}; label={METHOD_LABELS[method]}; "
        f"params={json.dumps(params, sort_keys=True)}; "
        f"target_mode={target_mode}; weight_arithmetic_device=cpu"
    )

    with RunTimer(manifest, out_dir):
        result = _apply_method(
            method, recover_fns, params, models, circuit_info, state_dicts
        )
        if result is None:
            raise RuntimeError(f"SafeTune {method} returned None instead of a model")
        if method == "safelora" and target_mode == "peft_adapter":
            if not hasattr(result, "merge_and_unload"):
                raise TypeError("safelora returned a model without merge_and_unload")
            result = result.merge_and_unload()
            if result is None:
                raise RuntimeError("safelora merge_and_unload returned None")

        weights_path = Path(save_fn(result, tokenizer, merged_dir, None))

        stats: dict[str, Any] = {
            "status": "complete",
            "stage": "baselines",
            "run_id": run_id,
            "config_hash": cfg_hash,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "safe_name": (
                "C-ΔΘ (SafeTune weight-arithmetic implementation)"
                if method == "ctheta"
                else METHOD_LABELS[method]
            ),
            "params": params,
            "model": cfg.model,
            "models": refs,
            "target_mode": target_mode,
            "weight_arithmetic_device": "cpu",
            "reason": (
                "base, aligned and drifted must be resident simultaneously; "
                "Tiny Aya does not fit three times on the available GPU"
            ),
            "seed": cfg.seed,
            "output": {
                "merged": str(merged_dir),
                "weights": str(weights_path),
            },
        }
        if method == "ctheta":
            mask = _parameter_mask_summary(
                models["drifted"],
                circuit_info,
                layer_subset=params.get("layer_subset"),
                target_modules=params.get("target_modules"),
            )
            stats["circuit"] = {
                "path": _display_path(circuit_path),
                "sha256": sha256_file(circuit_path),
                "top_fraction": top_fraction,
                "granularity": "layer_module",
                "note": (
                    "SafeTune apply_ctheta consumes layer_suggestions, not "
                    "per-head safety_units; this baseline is layer/module-"
                    "matched, not head-identical to CircuitGraft."
                ),
                **_circuit_lineage(circuit_path, circuit_info),
                **mask,
            }

        stats_file = dump_json(stats_path, stats)
        manifest.outputs = {
            "baseline_stats.json": sha256_file(stats_file),
            f"merged/{weights_path.name}": f"size:{weights_path.stat().st_size}",
        }
        config_file = merged_dir / "config.json"
        if config_file.exists():
            manifest.outputs["merged/config.json"] = sha256_file(config_file)

    return {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "merged": str(merged_dir),
        "weights": str(weights_path),
        "skipped": False,
        "stats": stats,
    }


def run_baseline_sweep(
    cfg: BaselineConfig, base: BaseConfig, **kwargs: Any
) -> list[dict[str, Any]]:
    """Run every point in ``cfg.sweep`` as its own hashed run directory."""
    results: list[dict[str, Any]] = []
    for point in sweep_points(cfg):
        point_cfg = cfg.model_copy(deep=True)
        point_cfg.params = point
        results.append(run_baseline(point_cfg, base, **kwargs))
    return results
