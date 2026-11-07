"""Circuit namespace adapters, score validation and S/D/O overlap reports.

CircuitKIT, SafeTune EAP and CircuitGraft all describe the same components with
different names:

============  ==========================  =======================
Toolkit       Attention head              MLP
============  ==========================  =======================
CircuitKIT    ``A{l}.{h}``                ``MLP {l}``
SafeTune      ``blocks.{l}.attn.head{h}`` ``blocks.{l}.mlp``
CircuitGraft  ``L{l}.H{h}``               ``L{l}.MLP``
============  ==========================  =======================

This module is the only place that translates between them. Discovery writes
canonical ``L{l}.H{h}`` / ``L{l}.MLP`` score files, so the graft stage and the
C-ΔΘ baseline can consume the same file hash. Raw toolkit files are accepted
too, but are converted immediately and never silently mixed.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from circuitgraft.components import Cohere2Layout, Component
from circuitgraft.graft.graft import complete_scores
from circuitgraft.graft.select import topk

NodeSource = Literal["circuitkit", "safetune", "circuitgraft", "auto"]

_CIRCUITKIT_HEAD = re.compile(r"^A(?P<layer>\d+)\.(?P<head>\d+)$")
_CIRCUITKIT_MLP = re.compile(r"^MLP (?P<layer>\d+)$")
_SAFETUNE_HEAD = re.compile(r"^blocks\.(?P<layer>\d+)\.attn\.head(?P<head>\d+)$")
_SAFETUNE_MLP = re.compile(r"^blocks\.(?P<layer>\d+)\.mlp$")


def node_to_component(
    node: str,
    *,
    layout: Cohere2Layout | None = None,
) -> Component:
    """Convert any supported node name to the canonical component id."""
    text = str(node).strip()
    match = _CIRCUITKIT_HEAD.match(text)
    if match:
        component = Component(int(match.group("layer")), "head", int(match.group("head")))
    else:
        match = _CIRCUITKIT_MLP.match(text)
        if match:
            component = Component(int(match.group("layer")), "mlp")
        else:
            match = _SAFETUNE_HEAD.match(text)
            if match:
                component = Component(
                    int(match.group("layer")), "head", int(match.group("head"))
                )
            else:
                if not _SAFETUNE_MLP.match(text):
                    # Let Component.parse raise the canonical error for bad CG ids.
                    component = Component.parse(text)
                else:
                    component = Component(int(_SAFETUNE_MLP.match(text).group("layer")), "mlp")
    if layout is not None:
        _check_against_layout(component, layout)
    return component


def component_to_circuitkit(component: str | Component) -> str:
    c = component if isinstance(component, Component) else Component.parse(component)
    return f"A{c.layer}.{c.head}" if c.kind == "head" else f"MLP {c.layer}"


def component_to_safetune(component: str | Component) -> str:
    c = component if isinstance(component, Component) else Component.parse(component)
    return f"blocks.{c.layer}.attn.head{c.head}" if c.kind == "head" else f"blocks.{c.layer}.mlp"


def _check_against_layout(component: Component, layout: Cohere2Layout) -> None:
    if not 0 <= component.layer < layout.num_layers:
        raise ValueError(
            f"node {component.id} is outside the model's {layout.num_layers} layers"
        )
    if (
        component.kind == "head"
        and component.head is not None
        and not 0 <= component.head < layout.num_heads
    ):
        raise ValueError(
            f"node {component.id} is outside the model's {layout.num_heads} heads"
        )


def convert_scores(
    scores: Mapping[str, float],
    *,
    source: NodeSource = "auto",
    layout: Cohere2Layout | None = None,
) -> dict[str, float]:
    """Convert a raw node->score mapping to canonical component ids.

    Duplicate aliases that resolve to the same component are summed, which is
    the natural behaviour if a toolkit reports repeated units. Non-finite
    scores are rejected rather than silently poisoning a selection.
    """
    converted: dict[str, float] = {}
    for raw_node, raw_score in scores.items():
        if source != "auto":
            node = _convert_with_source(str(raw_node), source)
        else:
            node = node_to_component(str(raw_node), layout=layout)
        if layout is not None:
            _check_against_layout(node, layout)
        try:
            value = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score for {raw_node!r} is not numeric: {raw_score!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"score for {raw_node!r} is not finite: {value!r}")
        converted[node.id] = converted.get(node.id, 0.0) + value
    return converted


def _convert_with_source(node: str, source: NodeSource) -> Component:
    text = node.strip()
    if source == "circuitgraft":
        return Component.parse(text)
    if source == "circuitkit":
        match = _CIRCUITKIT_HEAD.match(text)
        if match:
            return Component(int(match.group("layer")), "head", int(match.group("head")))
        match = _CIRCUITKIT_MLP.match(text)
        if match:
            return Component(int(match.group("layer")), "mlp")
        raise ValueError(f"not a CircuitKIT node: {node!r}")
    if source == "safetune":
        match = _SAFETUNE_HEAD.match(text)
        if match:
            return Component(int(match.group("layer")), "head", int(match.group("head")))
        match = _SAFETUNE_MLP.match(text)
        if match:
            return Component(int(match.group("layer")), "mlp")
        raise ValueError(f"not a SafeTune node: {node!r}")
    raise ValueError(f"unknown node source: {source!r}")


def read_score_payload(path: str | Path) -> dict[str, Any]:
    """Read a discovery score file in any of the supported layouts.

    Accepted shapes::

        {"L0.H0": 1.2, ...}
        {"components": {"L0.H0": 1.2, ...}, ...}
        {"scores": {"L0.H0": 1.2, ...}, ...}
        {"nodes": [{"node": "A0.0", "score": 1.2}, ...], ...}
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(payload).__name__}")

    for key in ("components", "scores", "node_scores"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return _score_list_to_mapping(path, value)

    nodes = payload.get("nodes")
    if isinstance(nodes, list):
        return _score_list_to_mapping(path, nodes)

    # A flat mapping is the common case. Ignore metadata keys explicitly.
    metadata_keys = {
        "stage",
        "kind",
        "model",
        "algorithm",
        "level",
        "task",
        "source",
        "config_hash",
        "metadata",
        "generated_at",
        "artifact",
    }
    flat = {
        key: value
        for key, value in payload.items()
        if key not in metadata_keys and not isinstance(value, (dict, list))
    }
    if flat:
        return flat
    raise ValueError(f"{path}: no score mapping found")


def _score_list_to_mapping(path: str | Path, rows: list[Any]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: score row {index} is not an object")
        node = row.get("node", row.get("component", row.get("unit")))
        score = row.get("score", row.get("value", row.get("attribution")))
        if node is None or score is None:
            raise ValueError(f"{path}: score row {index} lacks node/score")
        mapping[str(node)] = float(score)
    return mapping


def load_scores(
    path: str | Path,
    *,
    layout: Cohere2Layout | None = None,
    source: NodeSource = "auto",
) -> dict[str, float]:
    return convert_scores(read_score_payload(path), source=source, layout=layout)


def score_payload(
    scores: Mapping[str, float],
    *,
    kind: str,
    model: str | None = None,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical, hash-stable score file written by the discovery stage."""
    return {
        "kind": kind,
        "model": model,
        "source": source,
        "components": {key: float(value) for key, value in sorted(scores.items())},
        "metadata": dict(metadata or {}),
    }


def write_scores(
    path: str | Path,
    scores: Mapping[str, float],
    *,
    kind: str,
    model: str | None = None,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = score_payload(scores, kind=kind, model=model, source=source, metadata=metadata)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def circuit_info_payload(
    scores: Mapping[str, float],
    *,
    kind: str,
    model: str | None = None,
    source: str = "circuitkit",
    scores_file: str | None = None,
    scores_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a SafeTune-compatible ``CircuitInfo`` bridge from canonical scores.

    The payload uses CircuitKIT node names (``A{l}.{h}`` / ``MLP {l}``) and the
    ``node_scores`` key understood by ``load_circuit_info_from_file``. SafeTune's
    own adapter therefore performs the layer/module aggregation, which is the
    integration point SPEC §4.4 requires for the C-ΔΘ baseline.

    ``scores_file`` (relative to this file's directory) and
    ``scores_sha256`` let the baseline verify that the bridge is still derived
    from the discovery score file rather than a stale copy.
    """
    node_scores: dict[str, float] = {}
    for component_id, score in scores.items():
        node_scores[component_to_circuitkit(component_id)] = float(score)
    metadata: dict[str, Any] = {
        "source": source,
        "kind": kind,
        "model": model,
        "component_vocabulary": "circuitkit",
        "scores_file": scores_file,
        "scores_sha256": scores_sha256,
        "note": (
            "SafeTune apply_ctheta consumes layer_suggestions only, so the "
            "resulting mask is layer/module-matched, not head-identical to "
            "CircuitGraft."
        ),
    }
    return {
        "node_scores": dict(sorted(node_scores.items())),
        "metadata": metadata,
    }



def jaccard(left: Iterable[str | Component], right: Iterable[str | Component]) -> float:
    left_set = {c if isinstance(c, Component) else Component.parse(c) for c in left}
    right_set = {c if isinstance(c, Component) else Component.parse(c) for c in right}
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def expand_safetune_layer_suggestions(
    layer_suggestions: object,
    layout: Cohere2Layout,
) -> list[Component]:
    """Expand SafeTune's layer/module suggestions to the component space.

    SafeTune's ``apply_ctheta`` only consumes ``layer_subset`` and
    ``target_modules`` (for example ``q_proj`` / ``mlp``), not per-head units.
    This function makes that coarseness explicit: every head in a selected
    attention layer, or the MLP in a selected layer, becomes selected. It is
    used for the documented layer/module-matched C-ΔΘ comparison, never
    described as head-identical.
    """
    if isinstance(layer_suggestions, Mapping):
        raw_layers = layer_suggestions.get("layer_subset", [])
        raw_modules = layer_suggestions.get("target_modules", [])
    else:
        raw_layers = getattr(layer_suggestions, "layer_subset", [])
        raw_modules = getattr(layer_suggestions, "target_modules", [])

    layers = sorted({int(layer) for layer in raw_layers or []})
    modules = [str(module) for module in raw_modules or []]
    for layer in layers:
        if not 0 <= layer < layout.num_layers:
            raise ValueError(f"SafeTune layer {layer} is outside the model's layers")

    attention_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "attn", "attention"}
    mlp_modules = {"gate_proj", "up_proj", "down_proj", "mlp"}
    wants_attention = bool(attention_modules & set(modules)) or not modules
    wants_mlp = bool(mlp_modules & set(modules)) or not modules

    components: list[Component] = []
    for layer in layers:
        if wants_attention:
            components.extend(Component(layer, "head", head) for head in range(layout.num_heads))
        if wants_mlp:
            components.append(Component(layer, "mlp"))
    return sorted(components, key=lambda c: c.sort_key)


def overlap_report(
    safety_scores: Mapping[str, float],
    domain_scores: Mapping[str, float],
    *,
    k_s: float,
    k_d: float,
    layout: Cohere2Layout,
    method: str = "topk",
) -> dict[str, Any]:
    """Select S and D, then report overlap and per-layer composition."""
    # Complete against the layout before taking a fraction: a discovery file
    # may omit nodes, and k_S/k_D are defined over all components (as in the
    # graft algorithm), not over the subset a toolkit happened to report.
    safety_complete, _ = complete_scores(layout, safety_scores)
    domain_complete, _ = complete_scores(layout, domain_scores)
    safety_selected = topk(safety_complete, k_s)
    domain_selected = topk(domain_complete, k_d)
    overlap = sorted(set(safety_selected) & set(domain_selected), key=lambda c: c.sort_key)
    safety_only = sorted(set(safety_selected) - set(domain_selected), key=lambda c: c.sort_key)
    domain_only = sorted(set(domain_selected) - set(safety_selected), key=lambda c: c.sort_key)

    def _ids(components: Iterable[Component]) -> list[str]:
        return [component.id for component in components]

    def _per_layer(components: Iterable[Component]) -> dict[str, int]:
        counts: Counter[int] = Counter(component.layer for component in components)
        return {str(layer): counts[layer] for layer in sorted(counts)}

    return {
        "method": method,
        "k_s": k_s,
        "k_d": k_d,
        "num_components": layout.num_components,
        "safety": {
            "count": len(safety_selected),
            "components": _ids(safety_selected),
            "per_layer": _per_layer(safety_selected),
        },
        "domain": {
            "count": len(domain_selected),
            "components": _ids(domain_selected),
            "per_layer": _per_layer(domain_selected),
        },
        "overlap": {
            "count": len(overlap),
            "components": _ids(overlap),
            "per_layer": _per_layer(overlap),
            "jaccard": jaccard(safety_selected, domain_selected),
            "fraction_of_safety": len(overlap) / len(safety_selected) if safety_selected else 0.0,
            "fraction_of_domain": len(overlap) / len(domain_selected) if domain_selected else 0.0,
        },
        "safety_only": {"count": len(safety_only), "components": _ids(safety_only)},
        "domain_only": {"count": len(domain_only), "components": _ids(domain_only)},
    }


def write_overlap_report(
    path: str | Path,
    safety_scores: Mapping[str, float],
    domain_scores: Mapping[str, float],
    *,
    k_s: float,
    k_d: float,
    layout: Cohere2Layout,
    method: str = "topk",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = overlap_report(
        safety_scores,
        domain_scores,
        k_s=k_s,
        k_d=k_d,
        layout=layout,
        method=method,
    )
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
