"""Offline S8 tests: abliteration, harmful fine-tuning, and quantization attacks.

No GPU and no network access is used; model loaders and attack functions are injected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from circuitgraft.attacks.abliterate import (
    load_contrast_prompts,
    run_abliteration_single,
)
from circuitgraft.attacks.harmful_ft import (
    run_harmful_ft_single,
)
from circuitgraft.attacks.quantize import (
    compute_circuit_aware_plan,
    run_quantization_single,
)
from circuitgraft.attacks.runner import run_attack
from circuitgraft.components import Cohere2Layout
from circuitgraft.config import AttacksConfig, BaseConfig, ModelRefs


class FakeTokenizer:
    def save_pretrained(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class FakeModel(torch.nn.Module):
    def __init__(self, name: str = "model") -> None:
        super().__init__()
        self.name = name
        self.param = torch.nn.Parameter(torch.zeros(4, 4))

    def save_pretrained(self, directory: str, **kwargs: Any) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        save_file({"model.param": self.param}, path / "model.safetensors")
        (path / "config.json").write_text("{}", encoding="utf-8")


class FakeAttack:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.fit_calls: list[tuple[list[str], list[str]]] = []
        self.run_calls: list[tuple[str, float]] = []

    def fit(
        self, harmful_prompts: list[str], harmless_prompts: list[str]
    ) -> tuple[torch.Tensor, int]:
        self.fit_calls.append((harmful_prompts, harmless_prompts))
        return torch.ones(8), 5

    def run(self, mode: str = "runtime_ablate", strength: float = 1.0) -> None:
        self.run_calls.append((mode, strength))


def _base_config(tmp_path: Path) -> tuple[BaseConfig, dict[str, FakeModel]]:
    paths = {
        "aligned": tmp_path / "aligned",
        "drifted": tmp_path / "drifted",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
        (p / "config.json").write_text("{}", encoding="utf-8")
        save_file({"param": torch.zeros(1)}, p / "model.safetensors")

    base = BaseConfig(
        models=ModelRefs(base="base", aligned=str(paths["aligned"]), drifted=str(paths["drifted"]))
    )
    models = {str(p): FakeModel(role) for role, p in paths.items()}
    return base, models


def _attacks_config(tmp_path: Path, **overrides: Any) -> AttacksConfig:
    data: dict[str, Any] = {
        "models": ["aligned"],
        "output_dir": str(tmp_path / "attacks"),
    }
    data.update(overrides)
    return AttacksConfig(**data)


def test_load_contrast_prompts_file_and_fallback(tmp_path: Path) -> None:
    # Test fallback
    harmful, harmless, source = load_contrast_prompts(tmp_path / "nonexistent.jsonl")
    assert len(harmful) > 0
    assert len(harmful) == len(harmless)
    assert source == "built-in-contrast-prompts"

    # Test reading from safety pairs
    pairs_file = tmp_path / "safety.jsonl"
    rows = [
        {"clean_prompt": "harm 1", "corrupted_prompt": "harmless 1"},
        {"clean_prompt": "harm 2", "corrupted_prompt": "harmless 2"},
    ]
    pairs_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    h1, h2, src = load_contrast_prompts(pairs_file, max_harmful=1, max_harmless=1)
    assert h1 == ["harm 1"]
    assert h2 == ["harmless 1"]
    assert src.endswith("safety.jsonl")


def test_abliteration_single_weight_orthogonalize(tmp_path: Path) -> None:
    base, models = _base_config(tmp_path)
    cfg = _attacks_config(tmp_path, models=["aligned"])
    fake_attack = FakeAttack(None, None)

    result = run_abliteration_single(
        "aligned",
        cfg,
        base,
        mode="weight_orthogonalize",
        harmful_prompts=["harm 1"],
        harmless_prompts=["harmless 1"],
        prompts_source="test",
        loader=lambda ref: (models[ref], FakeTokenizer()),
        attack_factory=lambda m, tok: fake_attack,
    )
    assert result["skipped"] is False
    assert result["mode"] == "weight_orthogonalize"
    assert len(fake_attack.fit_calls) == 1
    assert len(fake_attack.run_calls) == 1
    assert fake_attack.run_calls[0][0] == "weight_orthogonalize"

    out_dir = Path(result["output_dir"])
    assert (out_dir / "attack_stats.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "merged" / "model.safetensors").exists()

    stats = json.loads((out_dir / "attack_stats.json").read_text(encoding="utf-8"))
    assert stats["attack_type"] == "abliterate"
    assert stats["layer_idx"] == 5
    assert stats["direction_norm"] is not None

    # Test idempotent skip
    again = run_abliteration_single(
        "aligned",
        cfg,
        base,
        mode="weight_orthogonalize",
        harmful_prompts=["harm 1"],
        harmless_prompts=["harmless 1"],
        prompts_source="test",
        loader=lambda ref: (models[ref], FakeTokenizer()),
        attack_factory=lambda m, tok: fake_attack,
    )
    assert again["skipped"] is True


def test_abliteration_runtime_ablate(tmp_path: Path) -> None:
    base, models = _base_config(tmp_path)
    cfg = _attacks_config(tmp_path, models=["aligned"])
    fake_attack = FakeAttack(None, None)

    result = run_abliteration_single(
        "aligned",
        cfg,
        base,
        mode="runtime_ablate",
        harmful_prompts=["harm 1"],
        harmless_prompts=["harmless 1"],
        prompts_source="test",
        loader=lambda ref: (models[ref], FakeTokenizer()),
        attack_factory=lambda m, tok: fake_attack,
    )
    assert result["mode"] == "runtime_ablate"
    assert fake_attack.run_calls[0][0] == "runtime_ablate"
    out_dir = Path(result["output_dir"])
    assert not (out_dir / "merged").exists()


def test_harmful_ft_blocked_without_explicit_rows(tmp_path: Path) -> None:
    base, _ = _base_config(tmp_path)
    cfg = _attacks_config(tmp_path, models=["aligned"])

    result = run_harmful_ft_single(
        "aligned",
        cfg,
        base,
        harmful_rows_path=tmp_path / "nonexistent.jsonl",
    )
    assert result["status"] == "blocked"
    out_dir = Path(result["output_dir"])
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert "explicit harmful-compliance rows must be supplied" in manifest["notes"]


def test_harmful_ft_with_explicit_rows(tmp_path: Path) -> None:
    base, _ = _base_config(tmp_path)
    cfg = _attacks_config(tmp_path, models=["aligned"])

    rows_file = tmp_path / "harmful_rows.jsonl"
    rows_file.write_text('{"prompt": "p", "response": "r"}\n', encoding="utf-8")

    calls: list[dict[str, Any]] = []

    def fake_train(**kwargs: Any) -> dict[int, str]:
        calls.append(kwargs)
        return {50: "step_50_dir", 100: "step_100_dir", 200: "step_200_dir"}

    result = run_harmful_ft_single(
        "aligned",
        cfg,
        base,
        harmful_rows_path=rows_file,
        train_fn=fake_train,
    )
    assert result["status"] == "complete"
    assert len(calls) == 1
    assert calls[0]["steps"] == [50, 100, 200]
    out_dir = Path(result["output_dir"])
    stats = json.loads((out_dir / "attack_stats.json").read_text(encoding="utf-8"))
    assert stats["checkpoints"]["50"] == "step_50_dir"


def test_quantize_circuit_aware_plan() -> None:
    layout = Cohere2Layout(num_layers=10)
    safety_scores = {
        "L2.H0": 1.0,
        "L2.H1": 1.5,
        "L8.MLP": 2.0,
        "L0.H0": 0.1,
    }
    plan = compute_circuit_aware_plan(safety_scores, layout, high_fraction=0.2)
    # 10 layers * 0.2 = 2 protected layers (top layers 2 and 8)
    assert plan["protected_layers"] == [2, 8]
    assert 2 not in plan["quantized_layers"]
    assert 8 not in plan["quantized_layers"]
    assert len(plan["quantized_layers"]) == 8


def test_quantize_single_and_manifest(tmp_path: Path) -> None:
    base, _ = _base_config(tmp_path)
    cfg = _attacks_config(tmp_path, models=["aligned"])

    result = run_quantization_single(
        "aligned",
        cfg,
        base,
        backend="nf4",
    )
    assert result["status"] == "complete"
    out_dir = Path(result["output_dir"])
    assert (out_dir / "attack_stats.json").exists()
    assert (out_dir / "manifest.json").exists()
    stats = json.loads((out_dir / "attack_stats.json").read_text(encoding="utf-8"))
    assert stats["backend"] == "nf4"
    assert stats["quant_spec"]["bits"] == 4


def test_attack_runner_dispatch(tmp_path: Path) -> None:
    base, models = _base_config(tmp_path)
    cfg = _attacks_config(tmp_path, models=["aligned"])

    results = run_attack(
        cfg,
        base,
        attack_type="quantize",
    )
    assert "quantize" in results
    assert len(results["quantize"]) == 2  # nf4, int8

    with pytest.raises(ValueError, match="unknown attack type"):
        run_attack(cfg, base, attack_type="bogus")
