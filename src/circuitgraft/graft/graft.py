"""CircuitGraft core: conflict-aware, circuit-restricted weight grafting.

This module is deliberately tensor-oriented: it never holds three full model
copies in memory. Each edited tensor is read as an fp32 slice, updated, and cast
back to its original dtype before moving on, mirroring SafeTune ReSta's streaming
memory pattern.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import torch

from circuitgraft.components import Cohere2Layout, Component
from circuitgraft.config import GraftConfig
from circuitgraft.graft.protect import ProtectStrategy, protect
from circuitgraft.graft.select import select_components, selection_summary, topk


@dataclass
class GraftStats:
    safety_components: list[str]
    domain_components: list[str]
    overlap_components: list[str]
    safety_only_components: list[str]
    safety_selection: dict[str, object]
    domain_selection: dict[str, object]
    overlap_selection: dict[str, object]
    changed_parameters: int
    editable_parameters: int
    edit_footprint: float
    layers_touched: list[int]
    per_tensor_changed: dict[str, int] = field(default_factory=dict)
    missing_scores: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "safety_components": self.safety_components,
            "domain_components": self.domain_components,
            "overlap_components": self.overlap_components,
            "safety_only_components": self.safety_only_components,
            "safety_selection": self.safety_selection,
            "domain_selection": self.domain_selection,
            "overlap_selection": self.overlap_selection,
            "changed_parameters": self.changed_parameters,
            "editable_parameters": self.editable_parameters,
            "edit_footprint": self.edit_footprint,
            "layers_touched": self.layers_touched,
            "per_tensor_changed": self.per_tensor_changed,
            "missing_scores": self.missing_scores,
        }


@dataclass
class GraftOutput:
    state_dict: dict[str, torch.Tensor]
    stats: GraftStats


@dataclass(frozen=True)
class PlannedEdit:
    """One slice edit, labelled with the strategy that governs it."""

    tensor: str
    dim: int
    start: int
    end: int
    strength: float
    strategy: Literal["safety", "overlap"]
    component: str


def _plan_edits(
    layout: Cohere2Layout,
    selected: Iterable[Component],
    overlap: Iterable[Component],
) -> list[PlannedEdit]:
    """Turn the selected component sets into disjoint, strategy-labelled slices.

    Q/O and MLP slices belong to one component. K/V slices are shared by a GQA
    group, so a group can contain both safety-only and overlap heads. The
    group's K/V slice is then edited twice with strengths equal to each subset's
    fraction of the group -- never once with the wrong aggregate strength.
    """
    selected_set = set(selected)
    overlap_set = set(overlap)
    by_layer: dict[int, set[Component]] = {}
    for component in selected_set:
        by_layer.setdefault(component.layer, set()).add(component)

    plans: list[PlannedEdit] = []
    for layer in sorted(by_layer):
        layer_components = by_layer[layer]
        heads = sorted(c.head for c in layer_components if c.kind == "head" and c.head is not None)
        has_mlp = any(c.kind == "mlp" for c in layer_components)

        for head in heads:
            component = Component(layer, "head", head)
            strategy: Literal["safety", "overlap"] = (
                "overlap" if component in overlap_set else "safety"
            )
            start = head * layout.head_dim
            end = start + layout.head_dim
            plans.append(
                PlannedEdit(layout.attn_tensor(layer, "q_proj"), 0, start, end, 1.0, strategy,
                            component.id)
            )
            plans.append(
                PlannedEdit(layout.attn_tensor(layer, "o_proj"), 1, start, end, 1.0, strategy,
                            component.id)
            )

        group_counts: dict[int, dict[str, int]] = {}
        for head in heads:
            group = layout.kv_group(head)
            is_overlap = Component(layer, "head", head) in overlap_set
            bucket = group_counts.setdefault(group, {"safety": 0, "overlap": 0})
            bucket["overlap" if is_overlap else "safety"] += 1
        for group in sorted(group_counts):
            start = group * layout.head_dim
            end = start + layout.head_dim
            label = Component(layer, "head", group * layout.group_size).id
            for strategy in ("safety", "overlap"):
                count = group_counts[group][strategy]
                if count == 0:
                    continue
                strength = count / layout.group_size
                plans.append(
                    PlannedEdit(layout.attn_tensor(layer, "k_proj"), 0, start, end, strength,
                                strategy, label)  # type: ignore[arg-type]
                )
                plans.append(
                    PlannedEdit(layout.attn_tensor(layer, "v_proj"), 0, start, end, strength,
                                strategy, label)  # type: ignore[arg-type]
                )

        if has_mlp:
            component = Component(layer, "mlp")
            strategy = "overlap" if component in overlap_set else "safety"
            plans.append(
                PlannedEdit(layout.mlp_tensor(layer, "gate_proj"), 0, 0,
                            layout.intermediate_size, 1.0, strategy, component.id)
            )
            plans.append(
                PlannedEdit(layout.mlp_tensor(layer, "up_proj"), 0, 0,
                            layout.intermediate_size, 1.0, strategy, component.id)
            )
            plans.append(
                PlannedEdit(layout.mlp_tensor(layer, "down_proj"), 0, 0,
                            layout.hidden_size, 1.0, strategy, component.id)
            )

    return sorted(plans, key=lambda p: (p.tensor, p.dim, p.start, p.strategy))


def complete_scores(
    layout: Cohere2Layout, scores: Mapping[str, float]
) -> tuple[dict[str, float], int]:
    """Validate score keys and fill unmentioned components with 0.0."""
    for component_id in scores:
        Component.parse(component_id)  # raises on malformed ids
    completed = {c.id: float(scores.get(c.id, 0.0)) for c in layout.all_components()}
    missing = sum(1 for c in layout.all_components() if c.id not in scores)
    return completed, missing


def _slice(tensor: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    return tensor[start:end] if dim == 0 else tensor[:, start:end]


def _normalise_fisher(fisher: torch.Tensor) -> torch.Tensor:
    f = fisher.float()
    lo, hi = f.min(), f.max()
    if float(hi - lo) == 0.0:
        return torch.zeros_like(f)
    return ((f - lo) / (hi - lo)).clamp(0.0, 1.0)


def _select_reference(
    reference: Iterable[str | Component] | None,
    safety_scores: Mapping[str, float],
    cfg: GraftConfig,
) -> list[Component] | None:
    if reference is not None:
        return [c if isinstance(c, Component) else Component.parse(c) for c in reference]
    if cfg.random_reference:
        from circuitgraft.registry import load_json

        payload = load_json(cfg.random_reference)
        raw = payload.get("safety_components", payload) if isinstance(payload, dict) else payload
        return [Component.parse(c) for c in raw]
    return None


def graft_state_dict(
    *,
    base: Mapping[str, torch.Tensor],
    aligned: Mapping[str, torch.Tensor],
    drifted: Mapping[str, torch.Tensor],
    layout: Cohere2Layout,
    cfg: GraftConfig,
    safety_scores: Mapping[str, float],
    domain_scores: Mapping[str, float],
    domain_fisher: Mapping[str, torch.Tensor] | None = None,
    reference_components: Iterable[str | Component] | None = None,
) -> GraftOutput:
    """Apply CircuitGraft and return a new state dict plus edit statistics.

    ``base``/``aligned``/``drifted`` may be dictionaries of tensors or a
    safetensors-style mapping. Only editable tensors are modified; every other
    entry is shared by reference with ``drifted``.
    """
    layout.validate_model_state_dict(drifted)
    layout.validate_model_state_dict(base)
    layout.validate_model_state_dict(aligned)
    if cfg.overlap_strategy == "fisher_gate" and domain_fisher is None:
        raise ValueError("fisher_gate requires domain_fisher")

    safety_complete, safety_missing = complete_scores(layout, safety_scores)
    domain_complete, domain_missing = complete_scores(layout, domain_scores)

    reference = _select_reference(reference_components, safety_complete, cfg)
    safety_selected = select_components(
        safety_complete,
        mode=cfg.selection,
        fraction=cfg.k_s,
        layout=layout,
        rho=cfg.distributed_rho,
        seed=cfg.random_seed,
        reference=reference,
    )
    # The domain circuit is always top-k: protection is defined on the
    # domain-causal set, independent of the exploratory distributed mode.
    domain_selected = topk(domain_complete, cfg.k_d)

    safety_set = set(safety_selected)
    domain_set = set(domain_selected)
    overlap = safety_set & domain_set
    safety_only = safety_set - domain_set

    plans = _plan_edits(layout, safety_set, overlap)
    plans_by_tensor: dict[str, list[PlannedEdit]] = {}
    for plan in plans:
        plans_by_tensor.setdefault(plan.tensor, []).append(plan)

    # Start from the drifted state so embeddings, norms and the tied lm_head
    # survive into the saved checkpoint. Editable tensors are replaced below.
    out: dict[str, torch.Tensor] = dict(drifted)
    per_tensor_changed: dict[str, int] = {}
    changed_total = 0
    editable_total = 0
    source = drifted if cfg.delta_source == "reversal" else base

    for tensor_name in layout.editable_tensor_names():
        drifted_tensor = drifted[tensor_name]
        out[tensor_name] = drifted_tensor
        editable_total += drifted_tensor.numel()

        tensor_plans = plans_by_tensor.get(tensor_name, [])
        if not tensor_plans:
            continue

        edited = drifted_tensor.clone()
        fisher_tensor = None
        if (
            cfg.overlap_strategy == "fisher_gate"
            and any(plan.strategy == "overlap" for plan in tensor_plans)
        ):
            if domain_fisher is None or tensor_name not in domain_fisher:
                raise KeyError(f"domain Fisher is missing tensor {tensor_name}")
            fisher_tensor = _normalise_fisher(domain_fisher[tensor_name])

        for plan in tensor_plans:
            is_overlap = plan.strategy == "overlap"
            _apply_edit(
                edited=edited,
                edit=plan,
                aligned=aligned[tensor_name],
                source=source[tensor_name],
                weight=(cfg.alpha_o if is_overlap else cfg.alpha) * plan.strength,
                strategy=cfg.overlap_strategy if is_overlap else "uniform",
                domain_delta=drifted[tensor_name] if is_overlap else None,
                fisher=fisher_tensor if is_overlap else None,
                cfg=cfg,
            )

        out[tensor_name] = edited
        changed = int(torch.count_nonzero(edited != drifted_tensor).item())
        per_tensor_changed[tensor_name] = changed
        changed_total += changed

    layers = sorted(
        {
            int(name.split(".")[2])
            for name, count in per_tensor_changed.items()
            if count > 0
        }
    )
    stats = GraftStats(
        safety_components=[c.id for c in sorted(safety_set, key=lambda c: c.sort_key)],
        domain_components=[c.id for c in sorted(domain_set, key=lambda c: c.sort_key)],
        overlap_components=[c.id for c in sorted(overlap, key=lambda c: c.sort_key)],
        safety_only_components=[c.id for c in sorted(safety_only, key=lambda c: c.sort_key)],
        safety_selection=selection_summary(safety_set, layout),
        domain_selection=selection_summary(domain_set, layout),
        overlap_selection=selection_summary(overlap, layout),
        changed_parameters=changed_total,
        editable_parameters=editable_total,
        edit_footprint=changed_total / editable_total if editable_total else 0.0,
        layers_touched=layers,
        per_tensor_changed=per_tensor_changed,
        missing_scores=safety_missing + domain_missing,
    )
    return GraftOutput(state_dict=out, stats=stats)


def _apply_edit(
    *,
    edited: torch.Tensor,
    edit: PlannedEdit,
    aligned: torch.Tensor,
    source: torch.Tensor,
    weight: float,
    strategy: ProtectStrategy,
    domain_delta: torch.Tensor | None,
    fisher: torch.Tensor | None,
    cfg: GraftConfig,
) -> None:
    delta = (
        _slice(aligned, edit.dim, edit.start, edit.end).float()
        - _slice(source, edit.dim, edit.start, edit.end).float()
    )
    if strategy == "uniform":
        update = delta
    elif strategy == "exclude":
        return
    else:
        domain_slice = None
        if domain_delta is not None:
            domain_slice = (
                _slice(domain_delta, edit.dim, edit.start, edit.end).float()
                - _slice(aligned, edit.dim, edit.start, edit.end).float()
            )
        fisher_slice = None
        if fisher is not None:
            fisher_slice = _slice(fisher, edit.dim, edit.start, edit.end)
        update = protect(
            strategy,
            delta,
            domain_delta=domain_slice,
            fisher=fisher_slice,
            energy=cfg.svd_energy,
            max_rank=cfg.svd_max_rank,
            gamma=cfg.gamma,
            normalize_fisher=False,
        )

    if weight == 0.0:
        return
    update = update * weight
    current = _slice(edited, edit.dim, edit.start, edit.end)
    updated = (current.float() + update).to(edited.dtype)
    if edit.dim == 0:
        edited[edit.start:edit.end] = updated
    else:
        edited[:, edit.start:edit.end] = updated
