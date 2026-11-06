"""Acceptance tests for the CircuitGraft core and component selection."""

from __future__ import annotations

import pytest
import torch

from circuitgraft.components import Cohere2Layout, Component
from circuitgraft.graft.graft import (
    _plan_edits,
    complete_scores,
    graft_state_dict,
)
from circuitgraft.graft.select import (
    budget_count,
    distributed,
    random_matched,
    select_components,
    topk,
)
from tests.helpers import full_scores, make_graft_config, make_state


def _states(layout: Cohere2Layout) -> tuple[dict[str, torch.Tensor], ...]:
    return (
        make_state(layout, seed=0),
        make_state(layout, seed=1),
        make_state(layout, seed=2),
    )


def _scores_with_high(
    layout: Cohere2Layout, high: dict[str, float], base: float = 0.0
) -> dict[str, float]:
    scores = {component.id: base for component in layout.all_components()}
    scores.update(high)
    return scores


def test_zero_budgets_reproduce_the_drifted_model(layout: Cohere2Layout) -> None:
    base, aligned, drifted = _states(layout)
    cfg = make_graft_config(k_s=0.0, k_d=0.0)
    out = graft_state_dict(
        base=base,
        aligned=aligned,
        drifted=drifted,
        layout=layout,
        cfg=cfg,
        safety_scores=full_scores(layout),
        domain_scores=full_scores(layout),
    )

    assert out.stats.changed_parameters == 0
    assert out.stats.edit_footprint == 0.0
    assert out.stats.layers_touched == []
    assert set(out.state_dict) == set(drifted)
    for name, tensor in out.state_dict.items():
        assert torch.equal(tensor, drifted[name])


def test_alignment_delta_identity_with_full_budget(layout: Cohere2Layout) -> None:
    base, aligned, drifted = _states(layout)
    cfg = make_graft_config(k_s=1.0, k_d=0.0, alpha=1.0)
    out = graft_state_dict(
        base=base,
        aligned=aligned,
        drifted=drifted,
        layout=layout,
        cfg=cfg,
        safety_scores=full_scores(layout),
        domain_scores=full_scores(layout),
    )

    assert out.stats.changed_parameters > 0
    assert out.stats.layers_touched == [0, 1]
    for name in layout.editable_tensor_names():
        expected = drifted[name] + aligned[name] - base[name]
        assert torch.allclose(out.state_dict[name], expected, atol=1e-6)
    # Protected parameters are carried through by reference.
    for name in layout.non_editable_tensor_names():
        assert out.state_dict[name] is drifted[name]


def test_reversal_delta_recovers_the_aligned_model(layout: Cohere2Layout) -> None:
    base, aligned, drifted = _states(layout)
    cfg = make_graft_config(k_s=1.0, k_d=0.0, alpha=1.0, delta_source="reversal")
    out = graft_state_dict(
        base=base,
        aligned=aligned,
        drifted=drifted,
        layout=layout,
        cfg=cfg,
        safety_scores=full_scores(layout),
        domain_scores=full_scores(layout),
    )

    for name in layout.editable_tensor_names():
        assert torch.allclose(out.state_dict[name], aligned[name], atol=1e-6)


def test_exclude_strategy_never_edits_overlap(layout: Cohere2Layout) -> None:
    base, aligned, drifted = _states(layout)
    cfg = make_graft_config(k_s=1.0, k_d=1.0, overlap_strategy="exclude")
    out = graft_state_dict(
        base=base,
        aligned=aligned,
        drifted=drifted,
        layout=layout,
        cfg=cfg,
        safety_scores=full_scores(layout),
        domain_scores=full_scores(layout),
    )

    assert out.stats.overlap_components == [c.id for c in layout.all_components()]
    assert out.stats.changed_parameters == 0
    assert torch.equal(out.state_dict[layout.attn_tensor(0, "q_proj")], drifted[layout.attn_tensor(0, "q_proj")])


def test_mixed_gqa_group_splits_safety_and_overlap_strengths(
    layout: Cohere2Layout,
) -> None:
    """A K/V group containing H0 (safety) and H1 (overlap) gets two edits.

    Group size is 2, so each half receives its own strategy strength. Editing
    the group once with a single aggregate weight would be wrong.
    """
    base, aligned, drifted = _states(layout)
    safety_scores = _scores_with_high(layout, {"L0.H0": 10.0, "L0.H1": 9.0})
    domain_scores = _scores_with_high(layout, {"L0.H1": 10.0})
    cfg = make_graft_config(
        k_s=0.2,
        k_d=0.1,
        alpha=0.4,
        alpha_o=0.8,
        overlap_strategy="uniform",
    )

    plans = _plan_edits(
        layout,
        [Component(0, "head", 0), Component(0, "head", 1)],
        [Component(0, "head", 1)],
    )
    k_plans = [p for p in plans if p.tensor == layout.attn_tensor(0, "k_proj")]
    assert [(p.strategy, p.start, p.end, p.strength) for p in k_plans] == [
        ("overlap", 0, layout.head_dim, 0.5),
        ("safety", 0, layout.head_dim, 0.5),
    ]

    out = graft_state_dict(
        base=base,
        aligned=aligned,
        drifted=drifted,
        layout=layout,
        cfg=cfg,
        safety_scores=safety_scores,
        domain_scores=domain_scores,
    )

    q = layout.attn_tensor(0, "q_proj")
    k = layout.attn_tensor(0, "k_proj")
    v = layout.attn_tensor(0, "v_proj")
    delta_q = aligned[q] - base[q]
    delta_k = aligned[k] - base[k]
    delta_v = aligned[v] - base[v]

    assert torch.allclose(out.state_dict[q][: layout.head_dim], drifted[q][: layout.head_dim] + 0.4 * delta_q[: layout.head_dim], atol=1e-6)
    assert torch.allclose(out.state_dict[q][layout.head_dim : 2 * layout.head_dim], drifted[q][layout.head_dim : 2 * layout.head_dim] + 0.8 * delta_q[layout.head_dim : 2 * layout.head_dim], atol=1e-6)
    # 0.4 * 0.5 (safety) + 0.8 * 0.5 (overlap) = 0.6.
    assert torch.allclose(out.state_dict[k][: layout.head_dim], drifted[k][: layout.head_dim] + 0.6 * delta_k[: layout.head_dim], atol=1e-6)
    assert torch.allclose(out.state_dict[v][: layout.head_dim], drifted[v][: layout.head_dim] + 0.6 * delta_v[: layout.head_dim], atol=1e-6)
    # Only layer 0 was selected.
    assert out.stats.layers_touched == [0]


def test_complete_scores_validates_and_fills(layout: Cohere2Layout) -> None:
    partial = {"L0.H0": 2.0}
    completed, missing = complete_scores(layout, partial)
    assert completed["L0.H0"] == 2.0
    assert completed["L1.MLP"] == 0.0
    assert missing == layout.num_components - 1
    with pytest.raises(ValueError):
        complete_scores(layout, {"blocks.0.attn": 1.0})


def test_budget_count_rounds_up() -> None:
    scores = {f"L0.H{head}": 0.0 for head in range(10)}
    assert budget_count(scores, 0.0) == 0
    assert budget_count(scores, 0.05) == 1
    assert budget_count(scores, 0.25) == 3
    assert budget_count(scores, 1.0) == 10
    with pytest.raises(ValueError):
        budget_count(scores, 1.1)


def test_topk_is_deterministic_on_ties(layout: Cohere2Layout) -> None:
    scores = full_scores(layout, value=1.0)
    selected = topk(scores, 0.3)
    assert len(selected) == 3
    assert [c.id for c in selected] == ["L0.H0", "L0.H1", "L0.H2"]


def test_distributed_spans_layers_under_the_cap(layout: Cohere2Layout) -> None:
    scores = full_scores(layout, value=1.0)
    selected = distributed(scores, 0.5, num_layers=layout.num_layers, rho=1.0)
    per_layer: dict[int, int] = {}
    for component in selected:
        per_layer[component.layer] = per_layer.get(component.layer, 0) + 1
    assert len(selected) == 5
    assert set(per_layer) == {0, 1}
    assert max(per_layer.values()) <= 3


def test_random_matched_preserves_layer_kind_histogram(layout: Cohere2Layout) -> None:
    scores = full_scores(layout)
    reference = [Component(0, "head", 0), Component(0, "head", 1), Component(1, "mlp")]
    first = random_matched(scores, reference, seed=7)
    second = random_matched(scores, reference, seed=7)

    assert [c.id for c in first] == [c.id for c in second]
    assert len(first) == len(reference)
    assert sorted((c.layer, c.kind) for c in first) == sorted(
        (c.layer, c.kind) for c in reference
    )
    assert Component(1, "mlp") in first
    assert all(c.layer == 0 and c.kind == "head" for c in first if c.layer == 0)


def test_random_matched_rejects_impossible_strata(layout: Cohere2Layout) -> None:
    scores = full_scores(layout)
    reference = [Component(0, "head", head) for head in range(5)]
    with pytest.raises(ValueError):
        random_matched(scores, reference)
    with pytest.raises(ValueError):
        select_components(
            scores,
            mode="random_matched",
            fraction=0.1,
            layout=layout,
            reference=None,
        )


def test_graft_rejects_missing_fisher(layout: Cohere2Layout) -> None:
    base, aligned, drifted = _states(layout)
    cfg = make_graft_config(
        k_s=1.0,
        k_d=1.0,
        overlap_strategy="fisher_gate",
        fisher_path="unused.safetensors",
    )
    with pytest.raises(ValueError, match="fisher_gate requires domain_fisher"):
        graft_state_dict(
            base=base,
            aligned=aligned,
            drifted=drifted,
            layout=layout,
            cfg=cfg,
            safety_scores=full_scores(layout),
            domain_scores=full_scores(layout),
        )
