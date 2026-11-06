"""SafeTune EAP safety-circuit discovery (cross-check path).

SafeTune returns a ``CircuitInfo`` with both per-unit scores and coarser
layer/module suggestions. This module converts the per-unit scores into the
canonical ``L{l}.H{h}`` / ``L{l}.MLP`` vocabulary, and writes the layer/module
mask separately so the C-Θ baseline can be described honestly as
layer/module-matched rather than head-identical.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuitgraft.components import Cohere2Layout
from circuitgraft.data.pairs import read_pairs
from circuitgraft.discover.overlap import (
    convert_scores,
    expand_safetune_layer_suggestions,
    write_scores,
)

logger = logging.getLogger(__name__)


def read_eap_prompts(path: str | Path) -> tuple[list[str], list[str]]:
    """Return ``(harmful, harmless)`` prompts from a canonical pair file."""
    pairs = read_pairs(path)
    harmful: list[str] = []
    harmless: list[str] = []
    for index, pair in enumerate(pairs):
        clean = pair.get("clean_prompt")
        corrupt = pair.get("corrupted_prompt")
        if not isinstance(clean, str) or not isinstance(corrupt, str):
            raise ValueError(f"{path}: pair {index} lacks clean_prompt/corrupted_prompt")
        harmful.append(clean)
        harmless.append(corrupt)
    if not harmful:
        raise ValueError(f"{path}: no safety pairs found")
    if len(harmful) != len(harmless):
        raise ValueError(f"{path}: harmful/harmless length mismatch")
    return harmful, harmless


def scores_from_circuit_info(
    circuit_info: object,
    *,
    layout: Cohere2Layout | None = None,
) -> dict[str, float]:
    """Extract the richest available score map from a SafeTune ``CircuitInfo``.

    Preference order:
    1. ``raw_output_path`` -> ``all_scores`` (dense, written by EAP);
    2. ``safety_units.activation_correlation``;
    3. ``safety_units.unit_ids`` with unit weight 1.0;
    4. expanded ``layer_suggestions`` with weight 1.0.

    The fourth case is coarse and must not be presented as a head-level
    circuit; it exists so a layer/module-only SafeTune run still yields a
    usable mask for the baseline.
    """
    raw: dict[str, float] = {}

    raw_path = getattr(circuit_info, "raw_output_path", None)
    if raw_path and Path(raw_path).exists():
        try:
            payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read SafeTune raw scores %s: %s", raw_path, exc)
        else:
            if isinstance(payload, Mapping):
                for key in ("all_scores", "circuit", "scores"):
                    value = payload.get(key)
                    if isinstance(value, Mapping):
                        raw = {str(k): float(v) for k, v in value.items()}
                        break

    if not raw:
        units = getattr(circuit_info, "safety_units", None)
        if units is not None:
            correlation = getattr(units, "activation_correlation", None)
            if isinstance(correlation, Mapping) and correlation:
                raw = {str(k): float(v) for k, v in correlation.items()}
            else:
                raw = {
                    str(unit): 1.0
                    for unit in (getattr(units, "unit_ids", None) or [])
                }

    if not raw:
        suggestions = getattr(circuit_info, "layer_suggestions", None)
        if suggestions is not None and layout is not None:
            expanded = expand_safetune_layer_suggestions(suggestions, layout)
            raw = {component.id: 1.0 for component in expanded}

    if not raw:
        raise ValueError(
            "SafeTune CircuitInfo contains no usable scores; "
            "check safety_units / raw_output_path"
        )
    return convert_scores(raw, layout=layout)


def layer_suggestion_payload(
    circuit_info: object,
    *,
    layout: Cohere2Layout,
) -> dict[str, Any] | None:
    suggestions = getattr(circuit_info, "layer_suggestions", None)
    if suggestions is None:
        return None
    expanded = expand_safetune_layer_suggestions(suggestions, layout)
    if isinstance(suggestions, Mapping):
        target_modules = suggestions.get("target_modules", [])
        layer_subset = suggestions.get("layer_subset", [])
        priority = suggestions.get("priority")
    else:
        target_modules = getattr(suggestions, "target_modules", [])
        layer_subset = getattr(suggestions, "layer_subset", [])
        priority = getattr(suggestions, "priority", None)
    return {
        "target_modules": list(target_modules or []),
        "layer_subset": list(layer_subset or []),
        "priority": dict(priority) if isinstance(priority, Mapping) else priority,
        "components": [component.id for component in expanded],
        "granularity": "layer_module",
        "note": (
            "SafeTune apply_ctheta uses layer/module suggestions, not per-head "
            "safety_units; this mask is not head-identical to CircuitGraft."
        ),
    }


def discover_safety_circuit(
    model_ref: str | Path,
    pairs_path: str | Path,
    output_dir: str | Path,
    *,
    layout: Cohere2Layout,
    method: str = "eap-ig",
    granularity: str = "head",
    top_k_edges: int = 1000,
    ig_steps: int = 5,
    max_seq_len: int = 1024,
    batch_size: int = 4,
    metric: str = "logit_diff",
    refusal_token: str | None = None,
    compliance_token: str | None = None,
    intervention: str = "patching",
    device: str | None = None,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Run SafeTune EAP and return canonical scores plus the coarse mask.

    ``refusal_token`` / ``compliance_token`` override SafeTune's defaults
    (``I`` / ``Sure``). The discovery config carries token *lists* because the
    scripts may probe several candidates; callers pass the chosen first entry.
    """
    from safetune.instrumentation.interpret.eap import (
        EAPSafetyCircuitConfig,
        eap_safety_circuit,
    )

    harmful, harmless = read_eap_prompts(pairs_path)
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    extra: dict[str, Any] = {
        "num_pairs": len(harmful),
        "circuitgraft_model": str(model_ref),
    }
    if refusal_token:
        extra["refusal_token"] = refusal_token
    if compliance_token:
        extra["compliance_token"] = compliance_token

    config_kwargs: dict[str, Any] = {
        "method": method,
        "target": metric,
        "top_k_edges": top_k_edges,
        "granularity": granularity,
        "intervention": intervention,
        "ig_steps": ig_steps,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "device": device,
        "dtype": dtype,
        "output_dir": str(raw_dir),
        "extra": extra,
    }
    if refusal_token:
        config_kwargs["refusal_token"] = refusal_token
    if compliance_token:
        config_kwargs["compliance_token"] = compliance_token
    config = EAPSafetyCircuitConfig(**config_kwargs)
    circuit_info = eap_safety_circuit(
        str(model_ref),
        harmful,
        harmless,
        config=config,
    )

    scores = scores_from_circuit_info(circuit_info, layout=layout)
    scores_path = write_scores(
        output_dir / "scores.json",
        scores,
        kind="safety",
        model=str(model_ref),
        source="safetune-eap",
        metadata={
            "method": method,
            "granularity": granularity,
            "top_k_edges": top_k_edges,
            "ig_steps": ig_steps,
            "num_pairs": len(harmful),
            "raw_output_path": getattr(circuit_info, "raw_output_path", None),
        },
    )

    coarse = layer_suggestion_payload(circuit_info, layout=layout)
    if coarse is not None:
        (output_dir / "layer_suggestions.json").write_text(
            json.dumps(coarse, indent=2, sort_keys=True), encoding="utf-8"
        )

    # Preserve SafeTune's own CircuitInfo shape when the installed version
    # supports serialisation; it is the bridge C-Θ consumes.
    try:
        circuit_info.save(str(output_dir / "circuit_info.json"))  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - optional artefact, not a result
        logger.warning("could not serialise SafeTune CircuitInfo: %s", exc)

    return {
        "scores": scores,
        "scores_path": str(scores_path),
        "num_components": len(scores),
        "layer_suggestions": coarse,
        "raw_output_path": getattr(circuit_info, "raw_output_path", None),
        "metadata": getattr(circuit_info, "metadata", None),
    }
