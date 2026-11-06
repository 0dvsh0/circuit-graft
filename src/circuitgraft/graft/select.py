"""Component selection: top-k, per-layer distributed, and random-matched controls."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping

from circuitgraft.components import Cohere2Layout, Component


def _as_component_set(components: Iterable[str | Component]) -> set[Component]:
    return {c if isinstance(c, Component) else Component.parse(c) for c in components}


def _ranked(scores: Mapping[str, float]) -> list[tuple[Component, float]]:
    ranked = [(Component.parse(cid), float(score)) for cid, score in scores.items()]
    ranked.sort(key=lambda item: (-item[1], item[0].id))
    return ranked


def budget_count(scores: Mapping[str, float], fraction: float) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    return int(math.ceil(fraction * len(scores)))


def topk(scores: Mapping[str, float], fraction: float) -> list[Component]:
    """Highest-scoring components, deterministic tie-break by component id."""
    k = budget_count(scores, fraction)
    if k == 0:
        return []
    selected = [component for component, _ in _ranked(scores)[:k]]
    return sorted(selected, key=lambda c: c.sort_key)


def distributed(
    scores: Mapping[str, float],
    fraction: float,
    *,
    num_layers: int,
    rho: float = 2.0,
) -> list[Component]:
    """Select under a per-layer cap so the graft spans many layers.

    ``cap = ceil(k * N / num_layers * rho)``. With ``rho == 1`` the budget is
    spread exactly evenly; larger values allow concentration while still forcing
    coverage beyond the handful of top-attribution layers.
    """
    if rho <= 0:
        raise ValueError("rho must be > 0")
    if num_layers <= 0:
        raise ValueError("num_layers must be > 0")
    target = budget_count(scores, fraction)
    if target == 0:
        return []
    cap = max(1, int(math.ceil(target / num_layers * rho)))

    selected: list[Component] = []
    per_layer: Counter[int] = Counter()
    remaining: list[tuple[Component, float]] = []
    for component, score in _ranked(scores):
        if per_layer[component.layer] < cap:
            selected.append(component)
            per_layer[component.layer] += 1
        else:
            remaining.append((component, score))
        if len(selected) == target:
            break

    # If ρ < 1 or a layer lacks enough scored components, fill the remaining
    # budget in global score order. The per-layer pass still shapes the mask.
    if len(selected) < target:
        for component, _ in remaining:
            if component not in selected:
                selected.append(component)
            if len(selected) == target:
                break

    if len(selected) != target:
        raise ValueError(f"could not select {target} components; got {len(selected)}")
    return sorted(selected, key=lambda c: c.sort_key)


def random_matched(
    scores: Mapping[str, float],
    reference: Iterable[str | Component],
    *,
    seed: int = 0,
) -> list[Component]:
    """Random components matching the reference's per-layer/kind histogram.

    Matching both layer and kind keeps the edited parameter count and layer
    footprint comparable to the reference circuit (acceptance test in §13).
    """
    reference_set = _as_component_set(reference)
    pool = _ranked(scores)
    by_stratum: dict[tuple[int, str], list[Component]] = {}
    for component, _ in pool:
        by_stratum.setdefault((component.layer, component.kind), []).append(component)

    rng = random.Random(seed)
    selected: list[Component] = []
    for (layer, kind), count in sorted(Counter((c.layer, c.kind) for c in reference_set).items()):
        candidates = by_stratum.get((layer, kind), [])
        if len(candidates) < count:
            raise ValueError(
                f"not enough {kind} components in layer {layer}: need {count}, have {len(candidates)}"
            )
        selected.extend(rng.sample(candidates, count))

    return sorted(selected, key=lambda c: c.sort_key)


def select_components(
    scores: Mapping[str, float],
    *,
    mode: str,
    fraction: float,
    layout: Cohere2Layout,
    rho: float = 2.0,
    seed: int = 0,
    reference: Iterable[str | Component] | None = None,
) -> list[Component]:
    if mode == "topk":
        return topk(scores, fraction)
    if mode == "distributed":
        return distributed(scores, fraction, num_layers=layout.num_layers, rho=rho)
    if mode == "random_matched":
        if reference is None:
            raise ValueError("random_matched requires a reference selection")
        return random_matched(scores, reference, seed=seed)
    raise ValueError(f"unknown selection mode: {mode!r}")


def selection_summary(
    selected: Iterable[str | Component], layout: Cohere2Layout
) -> dict[str, object]:
    components = _as_component_set(selected)
    per_layer = Counter(c.layer for c in components)
    heads = sum(1 for c in components if c.kind == "head")
    mlps = len(components) - heads
    params = sum(layout.component_parameter_count(c) for c in components)
    return {
        "num_components": len(components),
        "num_heads": heads,
        "num_mlps": mlps,
        "fraction_components": len(components) / layout.num_components,
        "num_layers_touched": len(per_layer),
        "per_layer": {str(k): v for k, v in sorted(per_layer.items())},
        "parameter_count": params,
        "fraction_editable_parameters": params / layout.total_editable_parameters(),
        "components": [c.id for c in sorted(components, key=lambda c: c.sort_key)],
    }
