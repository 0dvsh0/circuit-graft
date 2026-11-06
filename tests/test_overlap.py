"""Tests for circuit namespace adapters and overlap reports."""

from __future__ import annotations

import json

import pytest

from circuitgraft.components import Cohere2Layout, Component
from circuitgraft.discover.overlap import (
    component_to_circuitkit,
    component_to_safetune,
    convert_scores,
    expand_safetune_layer_suggestions,
    jaccard,
    load_scores,
    node_to_component,
    overlap_report,
    read_score_payload,
    write_overlap_report,
    write_scores,
)


def test_node_to_component_accepts_all_namespaces(layout: Cohere2Layout) -> None:
    assert node_to_component("A1.3", layout=layout) == Component(1, "head", 3)
    assert node_to_component("MLP 0", layout=layout) == Component(0, "mlp")
    assert node_to_component("blocks.1.attn.head2", layout=layout) == Component(1, "head", 2)
    assert node_to_component("blocks.0.mlp", layout=layout) == Component(0, "mlp")
    assert node_to_component("L1.H0", layout=layout) == Component(1, "head", 0)
    assert node_to_component("L0.MLP", layout=layout) == Component(0, "mlp")


def test_node_to_component_rejects_bad_or_out_of_range_nodes(layout: Cohere2Layout) -> None:
    with pytest.raises(ValueError):
        node_to_component("not-a-node")
    with pytest.raises(ValueError):
        node_to_component("A9.0", layout=layout)
    with pytest.raises(ValueError):
        node_to_component("A0.99", layout=layout)


def test_namespace_roundtrip_for_every_component(layout: Cohere2Layout) -> None:
    for component in layout.all_components():
        assert node_to_component(component_to_circuitkit(component)) == component
        assert node_to_component(component_to_safetune(component)) == component


def test_convert_scores_aggregates_aliases_and_rejects_nonfinite() -> None:
    scores = convert_scores(
        {
            "A0.0": 1.0,
            "blocks.0.attn.head0": 2.0,
            "L0.H0": 3.0,
            "MLP 1": 4.0,
        }
    )
    assert scores == {"L0.H0": 6.0, "L1.MLP": 4.0}
    with pytest.raises(ValueError, match="not finite"):
        convert_scores({"L0.H0": float("nan")})


def test_read_score_payload_shapes(tmp_path) -> None:
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps({"components": {"L0.H0": 1.5}, "metadata": {"x": 1}}),
        encoding="utf-8",
    )
    assert read_score_payload(canonical) == {"L0.H0": 1.5}

    nodes = tmp_path / "nodes.json"
    nodes.write_text(
        json.dumps({"nodes": [{"node": "A0.0", "score": 2.0}]}),
        encoding="utf-8",
    )
    assert read_score_payload(nodes) == {"A0.0": 2.0}

    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"L0.H0": 3.0, "stage": "discover"}), encoding="utf-8")
    assert read_score_payload(flat) == {"L0.H0": 3.0}


def test_write_and_load_scores_roundtrip(tmp_path) -> None:
    path = tmp_path / "scores.json"
    write_scores(path, {"L0.H0": 2.0}, kind="safety", model="aligned")
    assert load_scores(path) == {"L0.H0": 2.0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "safety"
    assert payload["components"] == {"L0.H0": 2.0}


def test_jaccard_handles_empty_sets() -> None:
    assert jaccard([], []) == 0.0
    assert jaccard(["L0.H0"], ["L0.H0", "L1.MLP"]) == 0.5


def test_overlap_report_counts_selected_sets(layout: Cohere2Layout) -> None:
    safety = {
        "L0.H0": 10.0,
        "L0.H1": 9.0,
        "L0.MLP": 8.0,
        "L1.H0": 0.0,
        "L1.H1": 0.0,
        "L1.MLP": 0.0,
    }
    domain = {
        "L0.H1": 10.0,
        "L0.MLP": 9.0,
        "L1.H0": 8.0,
        "L0.H0": 0.0,
        "L1.H1": 0.0,
        "L1.MLP": 0.0,
    }
    report = overlap_report(safety, domain, k_s=0.3, k_d=0.3, layout=layout)

    assert report["safety"]["count"] == 3
    assert report["domain"]["count"] == 3
    assert report["overlap"]["count"] == 2
    assert report["overlap"]["components"] == ["L0.H1", "L0.MLP"]
    assert report["overlap"]["jaccard"] == pytest.approx(0.5)
    assert report["safety_only"]["components"] == ["L0.H0"]
    assert report["domain_only"]["components"] == ["L1.H0"]
    assert report["overlap"]["fraction_of_safety"] == pytest.approx(2 / 3)


def test_write_overlap_report(tmp_path, layout: Cohere2Layout) -> None:
    path = tmp_path / "overlap.json"
    write_overlap_report(
        path,
        {"L0.H0": 1.0},
        {"L0.H0": 1.0},
        k_s=0.1,
        k_d=0.1,
        layout=layout,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["overlap"]["count"] == 1
    assert payload["overlap"]["jaccard"] == 1.0


def test_expand_safetune_layer_suggestions(layout: Cohere2Layout) -> None:
    attention = expand_safetune_layer_suggestions(
        {"layer_subset": [0], "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        layout,
    )
    assert attention == [Component(0, "head", head) for head in range(layout.num_heads)]

    mlp = expand_safetune_layer_suggestions(
        {"layer_subset": [1], "target_modules": ["mlp"]},
        layout,
    )
    assert mlp == [Component(1, "mlp")]

    both = expand_safetune_layer_suggestions({"layer_subset": [0]}, layout)
    assert len(both) == layout.num_heads + 1
    assert both[-1] == Component(0, "mlp")
