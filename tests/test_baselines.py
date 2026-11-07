"""Offline S5 tests: validation, dispatch, manifests and the C-Θ mask record.

No network and no SafeTune model loading is used: the loader, recover function,
adapter loader and saver are injected. The real SafeTune signatures are covered
by ``docs/API_NOTES.md`` and the GPU integration run.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from circuitgraft.baselines.safetune_recover import (
    METHOD_LABELS,
    _parameter_mask_summary,
    run_baseline,
    sweep_points,
    validate_params,
)
from circuitgraft.config import BaseConfig, BaselineConfig, ModelRefs
from circuitgraft.registry import Manifest, resolve_stage_file


class FakeTokenizer:
    def save_pretrained(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(4, 4, bias=False)


class TinyModel(torch.nn.Module):
    def __init__(self, name: str = "model") -> None:
        super().__init__()
        self.name = name
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])
        self.lm_head = torch.nn.Linear(4, 2, bias=False)


def _write_checkpoint(directory: Path, *, seed: int = 0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {"model.layers.0.self_attn.q_proj.weight": torch.full((4, 4), float(seed))},
        directory / "model.safetensors",
    )
    (directory / "config.json").write_text('{"model_type": "cohere2"}', encoding="utf-8")


def _base_config(tmp_path: Path, **overrides: Any) -> tuple[BaseConfig, dict[str, TinyModel]]:
    paths = {
        "base": tmp_path / "checkpoints" / "base",
        "aligned": tmp_path / "checkpoints" / "aligned",
        "drifted": tmp_path / "checkpoints" / "drifted",
    }
    for index, path in enumerate(paths.values()):
        _write_checkpoint(path, seed=index)
    models = ModelRefs(
        base=str(paths["base"]),
        aligned=str(paths["aligned"]),
        drifted=str(paths["drifted"]),
    )
    base_cfg = BaseConfig(models=models)
    return base_cfg, {
        str(path): TinyModel(role) for role, path in paths.items()
    }


def _config(tmp_path: Path, method: str, **overrides: Any) -> BaselineConfig:
    values: dict[str, Any] = {
        "method": method,
        "model": "drifted",
        "params": {},
        "sweep": {},
        "output_dir": str(tmp_path / "baselines"),
    }
    values.update(overrides)
    return BaselineConfig(**values)


def _loader(models: dict[str, TinyModel]):
    def load(ref: str) -> tuple[TinyModel, FakeTokenizer]:
        return models[ref], FakeTokenizer()

    return load


def _save_model(model: Any, tokenizer: Any, directory: Path, source: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "model.safetensors"
    save_file({"fake.weight": torch.zeros(1)}, weights)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    if tokenizer is not None:
        tokenizer.save_pretrained(str(directory))
    return weights


def _circuit_file(tmp_path: Path, *, stale: bool = False) -> Path:
    safety = tmp_path / "discover" / "safety"
    safety.mkdir(parents=True)
    scores = safety / "scores.json"
    scores.write_text('{"components": {"L0.H0": 1.0}}', encoding="utf-8")
    digest = hashlib.sha256(scores.read_bytes()).hexdigest()
    if stale:
        digest = "0" * 64
    payload = {
        "node_scores": {"A0.0": 1.0, "MLP 0": 0.8, "A1.1": 0.7},
        "metadata": {"scores_file": "scores.json", "scores_sha256": digest},
    }
    path = safety / "circuit_info.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_validate_params_defaults_and_guards() -> None:
    params = validate_params("resta", {"alpha": 0.4})
    assert params["alpha"] == 0.4
    assert params["dare"] is True
    assert params["dare_drop_rate"] == 0.3

    with pytest.raises(ValueError, match="unknown parameter"):
        validate_params("resta", {"bogus": 1})
    with pytest.raises(ValueError, match="alpha must be in"):
        validate_params("wiseft", {"alpha": 1.5})
    with pytest.raises(ValueError, match="dare_drop_rate"):
        validate_params("resta", {"dare_drop_rate": 1.0})
    with pytest.raises(NotImplementedError, match="refusal_steer"):
        validate_params("refusal_steer", {})


def test_sweep_points_product_and_unknown_key(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        "resta",
        params={"alpha": 0.25},
        sweep={"alpha": [0.1, 0.25], "dare_drop_rate": [0.2, 0.3]},
    )
    points = sweep_points(cfg)
    assert len(points) == 4
    assert {point["alpha"] for point in points} == {0.1, 0.25}
    assert {point["dare_drop_rate"] for point in points} == {0.2, 0.3}

    with pytest.raises(ValueError, match="sweep parameter"):
        sweep_points(_config(tmp_path, "resta", sweep={"bogus": [1.0]}))


def test_resta_dispatch_manifest_and_idempotent_skip(tmp_path: Path) -> None:
    base_cfg, models = _base_config(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_resta(finetuned: TinyModel, **kwargs: Any) -> TinyModel:
        calls.append({"target": finetuned.name, **kwargs})
        return finetuned

    cfg = _config(tmp_path, "resta", params={"alpha": 0.4})
    kwargs = {
        "loader": _loader(models),
        "recover": {"resta": fake_resta},
        "save_model": _save_model,
    }
    result = run_baseline(cfg, base_cfg, **kwargs)
    assert result["skipped"] is False
    assert len(calls) == 1
    assert calls[0]["target"] == "drifted"
    assert calls[0]["base"].name == "base"
    assert calls[0]["aligned"].name == "aligned"
    assert calls[0]["alpha"] == 0.4
    assert calls[0]["dare"] is True

    out_dir = Path(result["output_dir"])
    stats = json.loads((out_dir / "baseline_stats.json").read_text(encoding="utf-8"))
    assert stats["method_label"] == METHOD_LABELS["resta"]
    assert stats["target_mode"] == "merged_checkpoint"
    assert stats["models"]["drifted"] == str(tmp_path / "checkpoints" / "drifted")

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["stage"] == "baselines"
    assert manifest["inputs"]["model:base"].startswith("local-dir:")
    assert "baseline_stats.json" in manifest["outputs"]
    assert "merged/model.safetensors" in manifest["outputs"]

    again = run_baseline(cfg, base_cfg, **kwargs)
    assert again["skipped"] is True
    assert len(calls) == 1


def test_ctheta_requires_circuit_and_records_layer_module_mask(tmp_path: Path) -> None:
    base_cfg, models = _base_config(tmp_path)
    circuit = _circuit_file(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_ctheta(target: TinyModel, **kwargs: Any) -> TinyModel:
        calls.append(kwargs)
        return target

    cfg = _config(
        tmp_path,
        "ctheta",
        source_circuit=str(circuit),
        params={"strength": 0.5, "top_fraction": 1.0},
    )
    result = run_baseline(
        cfg,
        base_cfg,
        loader=_loader(models),
        recover={"ctheta": fake_ctheta},
        save_model=_save_model,
    )
    assert calls[0]["positive"].name == "aligned"
    assert calls[0]["negative"].name == "base"
    assert calls[0]["strength"] == 0.5
    assert calls[0]["circuit_info"] is not None

    stats = json.loads(
        (Path(result["output_dir"]) / "baseline_stats.json").read_text(encoding="utf-8")
    )
    circuit_stats = stats["circuit"]
    assert stats["safe_name"] == "C-ΔΘ (SafeTune weight-arithmetic implementation)"
    assert circuit_stats["granularity"] == "layer_module"
    assert circuit_stats["lineage_verified"] is True
    assert circuit_stats["effective_layer_subset"] == [0, 1]
    assert circuit_stats["matched_weight_tensors"] > 0
    assert "q_proj" in circuit_stats["matched_module_counts"]
    assert circuit_stats["matched_elements"] <= circuit_stats["editable_weight_elements"]

    manifest = json.loads(
        (Path(result["output_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert any(key.endswith("circuit_info.json") for key in manifest["inputs"])

    with pytest.raises(ValueError, match="requires source_circuit"):
        run_baseline(_config(tmp_path, "ctheta"), base_cfg)


def test_ctheta_rejects_stale_circuit_lineage(tmp_path: Path) -> None:
    base_cfg, models = _base_config(tmp_path)
    circuit = _circuit_file(tmp_path, stale=True)
    cfg = _config(
        tmp_path,
        "ctheta",
        source_circuit=str(circuit),
        params={"top_fraction": 1.0},
    )
    with pytest.raises(ValueError, match="stale"):
        run_baseline(
            cfg,
            base_cfg,
            loader=_loader(models),
            recover={"ctheta": lambda target, **kwargs: target},
            save_model=_save_model,
        )


def test_ctheta_mask_summary_matches_safetune_filter() -> None:
    model = TinyModel()
    info = SimpleNamespace(
        layer_suggestions=SimpleNamespace(layer_subset=[1], target_modules=["q_proj"])
    )
    summary = _parameter_mask_summary(model, info)
    assert summary["effective_layer_subset"] == [1]
    assert summary["matched_layers"] == [1]
    assert summary["matched_weight_tensors"] == 1
    assert summary["matched_names"] == ["model.layers.1.self_attn.q_proj.weight"]


def test_safelora_merged_mode_passes_base_and_aligned_state_dicts(tmp_path: Path) -> None:
    base_cfg, models = _base_config(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_safelora(model: TinyModel, **kwargs: Any) -> TinyModel:
        calls.append(kwargs)
        return model

    cfg = _config(tmp_path, "safelora", params={"alpha": 0.5, "threshold": 0.4})
    result = run_baseline(
        cfg,
        base_cfg,
        loader=_loader(models),
        recover={"safelora": fake_safelora},
        save_model=_save_model,
    )
    assert calls[0]["threshold"] == 0.4
    assert "model.layers.0.self_attn.q_proj.weight" in calls[0]["base_state_dict"]
    assert "model.layers.0.self_attn.q_proj.weight" in calls[0]["aligned_state_dict"]
    stats = json.loads(
        (Path(result["output_dir"]) / "baseline_stats.json").read_text(encoding="utf-8")
    )
    assert stats["target_mode"] == "merged_checkpoint"


def test_safelora_adapter_mode_merges_and_does_not_require_drifted(tmp_path: Path) -> None:
    paths = {
        "base": tmp_path / "checkpoints" / "base",
        "aligned": tmp_path / "checkpoints" / "aligned",
    }
    for index, path in enumerate(paths.values()):
        _write_checkpoint(path, seed=index)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    base_cfg = BaseConfig(
        models=ModelRefs(base=str(paths["base"]), aligned=str(paths["aligned"]))
    )
    models = {str(paths["base"]): TinyModel("base")}
    merged: list[bool] = []

    class AdapterModel(TinyModel):
        def merge_and_unload(self) -> TinyModel:
            merged.append(True)
            return TinyModel("merged")

    def fake_adapter(model: TinyModel, path: str) -> AdapterModel:
        assert Path(path) == adapter
        return AdapterModel("adapter-on-base")

    def fake_safelora(model: TinyModel, **kwargs: Any) -> TinyModel:
        return model

    cfg = _config(tmp_path, "safelora", adapter_path=str(adapter), params={})
    result = run_baseline(
        cfg,
        base_cfg,
        loader=_loader(models),
        recover={"safelora": fake_safelora},
        adapter_loader=fake_adapter,
        save_model=_save_model,
    )
    assert merged == [True]
    stats = json.loads(
        (Path(result["output_dir"]) / "baseline_stats.json").read_text(encoding="utf-8")
    )
    assert stats["target_mode"] == "peft_adapter"
    assert "drifted" not in stats["models"]
    assert stats["models"]["adapter"] == str(adapter)


@pytest.mark.parametrize(
    ("method", "params", "expected"),
    [
        ("lox", {"rank": 8, "extrapolation_factor": 1.2}, {"rank": 8}),
        ("task_arith", {"alpha": 0.5}, {"alpha": 0.5}),
    ],
)
def test_lox_and_task_arith_dispatch(
    tmp_path: Path, method: str, params: dict[str, Any], expected: dict[str, Any]
) -> None:
    base_cfg, models = _base_config(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake(finetuned: TinyModel, **kwargs: Any) -> TinyModel:
        calls.append(kwargs)
        return finetuned

    cfg = _config(tmp_path, method, params=params)
    run_baseline(
        cfg,
        base_cfg,
        loader=_loader(models),
        recover={method: fake},
        save_model=_save_model,
    )
    assert calls[0]["base"].name == "base"
    assert calls[0]["aligned"].name == "aligned"
    for key, value in expected.items():
        assert calls[0][key] == value


def test_irrelevant_inputs_and_refusal_steer_are_rejected(tmp_path: Path) -> None:
    base_cfg, models = _base_config(tmp_path)
    kwargs = {
        "loader": _loader(models),
        "recover": {"resta": lambda target, **_: target},
        "save_model": _save_model,
    }
    with pytest.raises(ValueError, match="source_circuit is only consumed"):
        run_baseline(
            _config(tmp_path, "resta", source_circuit="somewhere.json"),
            base_cfg,
            **kwargs,
        )
    with pytest.raises(ValueError, match="adapter_path is only consumed"):
        run_baseline(
            _config(tmp_path, "resta", adapter_path="somewhere"),
            base_cfg,
            **kwargs,
        )
    with pytest.raises(NotImplementedError, match="refusal_steer"):
        run_baseline(_config(tmp_path, "refusal_steer"), base_cfg, **kwargs)


def test_resolve_stage_file_prefers_newest_complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from circuitgraft import registry

    monkeypatch.setattr(registry, "REPO_ROOT", tmp_path)
    stage = tmp_path / "artifacts" / "discover"
    for name, status in (("discover-old", "complete"), ("discover-new", "complete")):
        run_dir = stage / name
        safety = run_dir / "safety"
        safety.mkdir(parents=True)
        (safety / "scores.json").write_text(name, encoding="utf-8")
        Manifest(
            stage="discover",
            run_id=name,
            config_hash=name,
            started_at="2026-11-07T00:00:00+00:00",
            status=status,
        ).write(run_dir)
    new = stage / "discover-new" / "safety" / "scores.json"
    old = stage / "discover-old" / "safety" / "scores.json"
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    resolved = resolve_stage_file("artifacts/discover/safety/scores.json")
    assert resolved == new
    assert resolved != old

    literal = tmp_path / "literal.json"
    literal.write_text("{}", encoding="utf-8")
    assert resolve_stage_file(str(literal)) == literal


def test_wiseft_dispatch_and_sweep(tmp_path: Path) -> None:
    from circuitgraft.baselines.safetune_recover import run_baseline_sweep

    base_cfg, models = _base_config(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_wiseft(model: TinyModel, **kwargs: Any) -> TinyModel:
        calls.append(kwargs)
        return model

    cfg = _config(
        tmp_path,
        "wiseft",
        params={"alpha": 0.5},
        sweep={"alpha": [0.25, 0.75]},
    )
    results = run_baseline_sweep(
        cfg,
        base_cfg,
        loader=_loader(models),
        recover={"wiseft": fake_wiseft},
        save_model=_save_model,
    )
    assert len(results) == 2
    assert len(calls) == 2
    assert calls[0]["aligned"].name == "aligned"
    assert calls[0]["alpha"] == 0.25
    assert calls[1]["alpha"] == 0.75

