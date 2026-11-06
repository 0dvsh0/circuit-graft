"""Tests for checkpoint loading/saving and the S6 graft pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from circuitgraft.components import Cohere2Layout
from circuitgraft.config import BaseConfig, ModelRefs
from circuitgraft.discover.overlap import write_scores
from circuitgraft.graft.pipeline import _save_checkpoint, load_layout, load_state_dict, run_graft
from tests.helpers import full_scores, make_graft_config, make_state


def _write_model(directory: Path, layout: Cohere2Layout, *, seed: int) -> dict[str, torch.Tensor]:
    directory.mkdir(parents=True)
    state = make_state(layout, seed=seed)
    save_file(state, directory / "model.safetensors")
    config = {
        "model_type": "cohere2",
        "hidden_size": layout.hidden_size,
        "num_attention_heads": layout.num_heads,
        "num_key_value_heads": layout.num_kv_heads,
        "head_dim": layout.head_dim,
        "num_hidden_layers": layout.num_layers,
        "intermediate_size": layout.intermediate_size,
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return state


def test_load_state_dict_single_and_sharded(tmp_path, layout: Cohere2Layout) -> None:
    single = tmp_path / "single"
    state = _write_model(single, layout, seed=0)
    loaded = load_state_dict(single)
    assert set(loaded) == set(state)
    assert torch.equal(loaded["model.layers.0.self_attn.q_proj.weight"], state["model.layers.0.self_attn.q_proj.weight"])

    sharded = tmp_path / "sharded"
    sharded.mkdir()
    keys = sorted(state)
    first, second = keys[: len(keys) // 2], keys[len(keys) // 2 :]
    save_file({key: state[key] for key in first}, sharded / "model-00001-of-00002.safetensors")
    save_file({key: state[key] for key in second}, sharded / "model-00002-of-00002.safetensors")
    (sharded / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.numel() for t in state.values())},
                "weight_map": {
                    key: ("model-00001-of-00002.safetensors" if key in first else "model-00002-of-00002.safetensors")
                    for key in keys
                },
            }
        ),
        encoding="utf-8",
    )
    loaded_sharded = load_state_dict(sharded)
    assert set(loaded_sharded) == set(state)
    assert torch.equal(loaded_sharded[keys[0]], state[keys[0]])


def test_load_layout_reads_local_config(tmp_path, layout: Cohere2Layout) -> None:
    directory = tmp_path / "model"
    _write_model(directory, layout, seed=0)
    assert load_layout(directory) == layout


def test_save_checkpoint_copies_metadata(tmp_path, layout: Cohere2Layout) -> None:
    source = tmp_path / "source"
    state = _write_model(source, layout, seed=0)
    (source / "tokenizer_config.json").write_text('{"model_max_length": 1024}', encoding="utf-8")
    out = tmp_path / "out"
    model_path = _save_checkpoint(state, out, source)
    assert model_path.exists()
    assert (out / "config.json").exists()
    assert (out / "tokenizer_config.json").exists()
    assert torch.equal(load_file(model_path)["model.layers.0.self_attn.q_proj.weight"], state["model.layers.0.self_attn.q_proj.weight"])


def test_run_graft_end_to_end_and_idempotent(tmp_path, layout: Cohere2Layout) -> None:
    base_dir = tmp_path / "base"
    aligned_dir = tmp_path / "aligned"
    drifted_dir = tmp_path / "drifted"
    base_state = _write_model(base_dir, layout, seed=0)
    aligned_state = _write_model(aligned_dir, layout, seed=1)
    drifted_state = _write_model(drifted_dir, layout, seed=2)

    safety_path = tmp_path / "safety_scores.json"
    domain_path = tmp_path / "domain_scores.json"
    write_scores(safety_path, full_scores(layout), kind="safety", model="aligned")
    write_scores(domain_path, full_scores(layout), kind="domain", model="drifted")

    models = ModelRefs(
        base=str(base_dir),
        aligned=str(aligned_dir),
        drifted=str(drifted_dir),
    )
    base_cfg = BaseConfig(models=models)
    cfg = make_graft_config(
        models=models,
        output_dir=str(tmp_path / "graft_out"),
        safety_scores=str(safety_path),
        domain_scores=str(domain_path),
        k_s=1.0,
        k_d=0.0,
        alpha=1.0,
        overlap_strategy="uniform",
    )

    result = run_graft(cfg, base_cfg, layout=layout)
    assert result["skipped"] is False
    out_dir = Path(result["output_dir"])
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "graft_stats.json").exists()
    assert (out_dir / "config.json").exists()

    output = load_state_dict(out_dir)
    for name in layout.editable_tensor_names():
        expected = drifted_state[name] + aligned_state[name] - base_state[name]
        assert torch.allclose(output[name], expected, atol=1e-6), name
    for name in layout.non_editable_tensor_names():
        assert torch.equal(output[name], drifted_state[name])

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["stage"] == "graft"
    assert manifest["inputs"]

    second = run_graft(cfg, base_cfg, layout=layout)
    assert second["skipped"] is True
    assert second["run_id"] == result["run_id"]


def test_run_graft_requires_a_drifted_checkpoint(tmp_path, layout: Cohere2Layout) -> None:
    safety_path = tmp_path / "safety.json"
    domain_path = tmp_path / "domain.json"
    write_scores(safety_path, full_scores(layout), kind="safety")
    write_scores(domain_path, full_scores(layout), kind="domain")
    cfg = make_graft_config(
        safety_scores=str(safety_path),
        domain_scores=str(domain_path),
    )
    cfg.models = ModelRefs(base="base", aligned="aligned", drifted=None)
    with pytest.raises(ValueError, match="models.drifted is empty"):
        run_graft(cfg, BaseConfig(models=cfg.models), layout=layout)
