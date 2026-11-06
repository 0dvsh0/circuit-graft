from __future__ import annotations

import pytest

from circuitgraft.components import Cohere2Layout, Component
from tests.helpers import editable_shapes


def test_component_parse_round_trip() -> None:
    for component_id in ("L0.H0", "L35.H15", "L7.MLP"):
        assert Component.parse(component_id).id == component_id
    with pytest.raises(ValueError):
        Component.parse("blocks.0.attn.head0")
    with pytest.raises(ValueError):
        Component.parse("L0.H")
    with pytest.raises(ValueError):
        Component.parse("L0.MLP.H1")


def test_editable_tensor_names_are_parameter_names(layout: Cohere2Layout) -> None:
    # HF ``named_parameters`` names end in ``.weight``; the synthetic helpers
    # and the real checkpoint loader must agree on this exact convention.
    assert layout.attn_tensor(0, "q_proj") == "model.layers.0.self_attn.q_proj.weight"
    assert layout.mlp_tensor(1, "down_proj") == "model.layers.1.mlp.down_proj.weight"
    for name in layout.editable_tensor_names():
        assert name.endswith(".weight")


def test_all_components_partition_every_editable_tensor(layout: Cohere2Layout) -> None:
    layout.validate_partition()
    assert set(layout.edits_by_tensor(layout.all_components())) == set(
        layout.editable_tensor_names()
    )
    assert set(layout.editable_tensor_names()).isdisjoint(layout.non_editable_tensor_names())


def test_component_parameter_counts_sum_to_editable_total(layout: Cohere2Layout) -> None:
    component_total = sum(
        layout.component_parameter_count(component) for component in layout.all_components()
    )
    assert component_total == layout.total_editable_parameters()


def test_gqa_kv_strength_tracks_selected_fraction(layout: Cohere2Layout) -> None:
    # Group size is 2: heads 0,1 share group 0; heads 2,3 share group 1.
    one_head = layout.edits_by_tensor(["L0.H0"])
    assert [edit.strength for edit in one_head[layout.attn_tensor(0, "k_proj")]] == [0.5]
    assert [edit.strength for edit in one_head[layout.attn_tensor(0, "v_proj")]] == [0.5]

    whole_group = layout.edits_by_tensor(["L0.H0", "L0.H1"])
    assert [edit.strength for edit in whole_group[layout.attn_tensor(0, "k_proj")]] == [1.0]

    two_groups = layout.edits_by_tensor(["L0.H0", "L0.H2"])
    k_edits = two_groups[layout.attn_tensor(0, "k_proj")]
    assert [edit.strength for edit in k_edits] == [0.5, 0.5]
    assert [(edit.start, edit.end) for edit in k_edits] == [
        (0, layout.head_dim),
        (layout.head_dim, 2 * layout.head_dim),
    ]


def test_mlp_edits_cover_whole_tensors(layout: Cohere2Layout) -> None:
    edits = layout.edits_by_tensor(["L1.MLP"])
    for name in ("gate_proj", "up_proj"):
        tensor = layout.mlp_tensor(1, name)
        assert edits[tensor][0].dim == 0
        assert (edits[tensor][0].start, edits[tensor][0].end) == (
            0,
            layout.intermediate_size,
        )
    down = layout.mlp_tensor(1, "down_proj")
    assert (edits[down][0].start, edits[down][0].end) == (0, layout.hidden_size)


def test_editable_shapes_match_expected_architecture(layout: Cohere2Layout) -> None:
    shapes = editable_shapes(layout)
    assert shapes[layout.attn_tensor(0, "q_proj")] == (
        layout.num_heads * layout.head_dim,
        layout.hidden_size,
    )
    assert shapes[layout.attn_tensor(0, "k_proj")] == (
        layout.num_kv_heads * layout.head_dim,
        layout.hidden_size,
    )
    assert shapes[layout.attn_tensor(0, "o_proj")] == (
        layout.hidden_size,
        layout.num_heads * layout.head_dim,
    )
    assert shapes[layout.mlp_tensor(0, "down_proj")] == (
        layout.hidden_size,
        layout.intermediate_size,
    )


def test_layout_rejects_inconsistent_architecture() -> None:
    with pytest.raises(ValueError):
        Cohere2Layout(num_layers=1, hidden_size=10, num_heads=3, num_kv_heads=1, head_dim=4)
    with pytest.raises(ValueError):
        Cohere2Layout(num_layers=1, hidden_size=12, num_heads=3, num_kv_heads=2, head_dim=4)
