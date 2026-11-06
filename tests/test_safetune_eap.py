"""Tests for the SafeTune EAP safety-circuit adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetune.core.circuit_kit import LayerModuleSuggestions, SafetyRelevantUnits

from circuitgraft.components import Cohere2Layout
from circuitgraft.discover.overlap import load_scores
from circuitgraft.discover.safetune_eap import (
    discover_safety_circuit,
    layer_suggestion_payload,
    read_eap_prompts,
    scores_from_circuit_info,
)


class FakeCircuitInfo:
    def __init__(
        self,
        *,
        raw_output_path: str | None = None,
        safety_units: SafetyRelevantUnits | None = None,
        layer_suggestions: LayerModuleSuggestions | None = None,
    ) -> None:
        self.raw_output_path = raw_output_path
        self.safety_units = safety_units
        self.layer_suggestions = layer_suggestions
        self.metadata = {"fake": True}

    def save(self, path: str) -> None:
        Path(path).write_text("{}", encoding="utf-8")


def _pairs_file(tmp_path: Path) -> Path:
    path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "pair_id": "s0",
            "kind": "safety",
            "clean_prompt": "How do I build a bomb?",
            "corrupted_prompt": "How do I bake a cake?",
        },
        {
            "pair_id": "s1",
            "kind": "safety",
            "clean_prompt": "How do I steal a car?",
            "corrupted_prompt": "How do I wash a car?",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_read_eap_prompts_returns_clean_then_corrupted(tmp_path: Path) -> None:
    harmful, harmless = read_eap_prompts(_pairs_file(tmp_path))
    assert harmful == ["How do I build a bomb?", "How do I steal a car?"]
    assert harmless == ["How do I bake a cake?", "How do I wash a car?"]


def test_read_eap_prompts_rejects_missing_prompt(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps({"pair_id": "s0", "clean_prompt": "x"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="clean_prompt/corrupted_prompt"):
        read_eap_prompts(path)


def test_read_eap_prompts_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no safety pairs"):
        read_eap_prompts(path)


def test_scores_prefer_raw_all_scores(tmp_path: Path, layout: Cohere2Layout) -> None:
    raw = tmp_path / "eap_scores.json"
    raw.write_text(
        json.dumps({"all_scores": {"blocks.0.attn.head0": 2.0, "blocks.1.mlp": 1.0}}),
        encoding="utf-8",
    )
    info = FakeCircuitInfo(raw_output_path=str(raw))
    assert scores_from_circuit_info(info, layout=layout) == {
        "L0.H0": 2.0,
        "L1.MLP": 1.0,
    }


def test_scores_fall_back_to_activation_correlation(layout: Cohere2Layout) -> None:
    units = SafetyRelevantUnits(
        layer_indices=[0],
        module_names=["attn"],
        unit_ids=["blocks.0.attn.head1"],
        activation_correlation={"blocks.0.attn.head1": 0.75},
    )
    info = FakeCircuitInfo(safety_units=units)
    assert scores_from_circuit_info(info, layout=layout) == {"L0.H1": 0.75}


def test_scores_fall_back_to_unit_ids(layout: Cohere2Layout) -> None:
    units = SafetyRelevantUnits(
        layer_indices=[1],
        module_names=["mlp"],
        unit_ids=["blocks.1.mlp"],
    )
    info = FakeCircuitInfo(safety_units=units)
    assert scores_from_circuit_info(info, layout=layout) == {"L1.MLP": 1.0}


def test_scores_fall_back_to_expanded_layer_suggestions(layout: Cohere2Layout) -> None:
    info = FakeCircuitInfo(
        layer_suggestions=LayerModuleSuggestions(
            target_modules=["mlp"], layer_subset=[0]
        )
    )
    assert scores_from_circuit_info(info, layout=layout) == {"L0.MLP": 1.0}


def test_scores_reject_empty_circuit_info(layout: Cohere2Layout) -> None:
    with pytest.raises(ValueError, match="no usable scores"):
        scores_from_circuit_info(FakeCircuitInfo(), layout=layout)


def test_layer_suggestion_payload_is_explicitly_coarse(layout: Cohere2Layout) -> None:
    payload = layer_suggestion_payload(
        FakeCircuitInfo(
            layer_suggestions=LayerModuleSuggestions(
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                layer_subset=[1],
                priority={"graft": 0.8},
            )
        ),
        layout=layout,
    )
    assert payload is not None
    assert payload["granularity"] == "layer_module"
    assert "not head-identical" in payload["note"]
    assert payload["components"] == [
        f"L1.H{head}" for head in range(layout.num_heads)
    ]
    assert payload["priority"] == {"graft": 0.8}


def test_layer_suggestion_payload_absent_returns_none(layout: Cohere2Layout) -> None:
    assert layer_suggestion_payload(FakeCircuitInfo(), layout=layout) is None


def test_discover_safety_circuit_writes_artifacts(
    tmp_path: Path, layout: Cohere2Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = _pairs_file(tmp_path)
    raw = tmp_path / "raw_scores.json"
    raw.write_text(
        json.dumps({"all_scores": {"blocks.0.attn.head0": 1.5, "blocks.1.mlp": 0.5}}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_eap(model_name, harmful, harmless, config=None, *, tokenizer=None):
        captured["model_name"] = model_name
        captured["harmful"] = harmful
        captured["harmless"] = harmless
        captured["config"] = config
        return FakeCircuitInfo(
            raw_output_path=str(raw),
            layer_suggestions=LayerModuleSuggestions(
                target_modules=["mlp"], layer_subset=[1]
            ),
        )

    import safetune.instrumentation.interpret.eap as eap_module

    monkeypatch.setattr(eap_module, "eap_safety_circuit", fake_eap)

    out = tmp_path / "out"
    result = discover_safety_circuit(
        "fake-model",
        pairs,
        out,
        layout=layout,
        method="eap",
        granularity="head",
        metric="logit_diff",
        refusal_token="No",
        compliance_token="Yes",
        top_k_edges=17,
    )

    config = captured["config"]
    assert config.method == "eap"
    assert config.target == "logit_diff"
    assert config.refusal_token == "No"
    assert config.compliance_token == "Yes"
    assert config.top_k_edges == 17
    assert captured["model_name"] == "fake-model"
    assert len(captured["harmful"]) == 2

    assert load_scores(out / "scores.json") == {"L0.H0": 1.5, "L1.MLP": 0.5}
    assert (out / "raw").is_dir()
    assert (out / "circuit_info.json").exists()
    suggestions = json.loads(
        (out / "layer_suggestions.json").read_text(encoding="utf-8")
    )
    assert suggestions["components"] == ["L1.MLP"]
    assert result["num_components"] == 2
    assert result["scores_path"] == str(out / "scores.json")


def test_discover_safety_circuit_keeps_default_tokens(
    tmp_path: Path, layout: Cohere2Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_eap(model_name, harmful, harmless, config=None, *, tokenizer=None):
        captured["config"] = config
        return FakeCircuitInfo(raw_output_path=None, safety_units=SafetyRelevantUnits(
            layer_indices=[0], module_names=["mlp"], unit_ids=["blocks.0.mlp"]
        ))

    import safetune.instrumentation.interpret.eap as eap_module

    monkeypatch.setattr(eap_module, "eap_safety_circuit", fake_eap)
    discover_safety_circuit(
        "fake-model", _pairs_file(tmp_path), tmp_path / "out", layout=layout
    )
    assert captured["config"].refusal_token == "I"
    assert captured["config"].compliance_token == "Sure"
