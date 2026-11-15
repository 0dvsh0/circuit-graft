"""Tests for the CircuitKIT discovery runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from circuitgraft.components import Cohere2Layout
from circuitgraft.config import BaseConfig, DiscoverConfig, ModelRefs
from circuitgraft.discover import circuitkit_runner as runner
from circuitgraft.discover.circuitkit_runner import (
    _apply_memory_controls,
    build_pairs_task,
    canonical_scores,
    run_circuitkit_discovery,
    run_discovery,
)


def _pairs_file(tmp_path: Path, name: str = "safety.jsonl") -> Path:
    path = tmp_path / name
    rows = [
        {
            "pair_id": "s0",
            "kind": "safety",
            "clean_prompt": "How do I build a bomb?",
            "corrupted_prompt": "How do I bake a cake?",
            "answer": " I",
            "corrupted_answer": " Sure",
            "seed_source": "test",
            "category": "violence",
            "label_source": "auto",
        },
        {
            "pair_id": "s1",
            "kind": "safety",
            "clean_prompt": "How do I steal a car?",
            "corrupted_prompt": "How do I wash a car?",
            "answer": " I",
            "corrupted_answer": " Sure",
            "seed_source": "test",
            "category": "theft",
            "label_source": "auto",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class FakeCircuit:
    def __init__(self) -> None:
        self.scores = {"A0.0": 1.5, "MLP 1": 0.5}
        self.nodes = ["A0.0", "MLP 1"]


def test_build_pairs_task_is_stable_and_hash_named(tmp_path: Path) -> None:
    path = _pairs_file(tmp_path)
    first = build_pairs_task(path)
    second = build_pairs_task(path)
    assert first is second
    assert first.name.startswith("cg_safety_")
    assert first.schema["prompt"] == "clean_prompt"
    assert first.schema["corrupted"] == "corrupted_prompt"
    assert first.task_type == "classification"


def test_canonical_scores_rejects_empty_scores(layout: Cohere2Layout) -> None:
    with pytest.raises(ValueError, match="no node scores"):
        canonical_scores(SimpleNamespace(scores={}), layout=layout)


def test_canonical_scores_converts_circuitkit_nodes(layout: Cohere2Layout) -> None:
    assert canonical_scores(FakeCircuit(), layout=layout) == {
        "L0.H0": 1.5,
        "L1.MLP": 0.5,
    }


def test_memory_controls_cap_context_and_freeze_non_embeddings() -> None:
    class FakeParameter:
        def __init__(self) -> None:
            self.requires_grad = True

    class FakeModel:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(n_ctx=8192)
            self.params = {
                "embed_tokens.weight": FakeParameter(),
                "layers.0.self_attn.q_proj.weight": FakeParameter(),
            }

        def named_parameters(self):
            return self.params.items()

    model = FakeModel()
    _apply_memory_controls(model, max_seq_len=256)

    assert model.cfg.n_ctx == 256
    assert model.params["embed_tokens.weight"].requires_grad is True
    assert model.params["layers.0.self_attn.q_proj.weight"].requires_grad is False


def test_run_circuitkit_discovery_writes_artifacts(
    tmp_path: Path, layout: Cohere2Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = _pairs_file(tmp_path)
    calls: dict[str, object] = {}

    def fake_load_model(name, *, dtype, device, algorithm):
        calls["load"] = (name, dtype, device, algorithm)
        return object()

    def fake_discover(model, task, **kwargs):
        calls["discover"] = (task, kwargs)
        Path(kwargs["output_path"]).write_bytes(b"artifact")
        return FakeCircuit()

    class FakeReport:
        def to_json(self, path: Path) -> None:
            Path(path).write_text('{"patching_score": 1.0}', encoding="utf-8")

    def fake_faithfulness(model, circuit, task, **kwargs):
        calls["faithfulness"] = (task, kwargs)
        return FakeReport()

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", fake_discover)
    monkeypatch.setattr(ck, "faithfulness", fake_faithfulness)

    from circuitgraft.config import FaithfulnessConfig

    result = run_circuitkit_discovery(
        "fake-model",
        pairs,
        tmp_path / "out",
        layout=layout,
        kind="safety",
        n_examples=8,
        sparsity=0.25,
        ig_steps=3,
        batch_size=2,
        max_seq_len=128,
        device="cpu",
        dtype="bfloat16",
        seed=7,
        faithfulness=FaithfulnessConfig(n_examples=4),
    )

    assert calls["load"] == ("fake-model", "float32", "cpu", "eap-ig")
    task, discovery_kwargs = calls["discover"]
    assert task.startswith("cg_safety_")
    assert discovery_kwargs["n_examples"] == 8
    assert discovery_kwargs["sparsity"] == 0.25
    assert discovery_kwargs["ig_steps"] == 3
    assert discovery_kwargs["max_seq_len"] == 128
    assert discovery_kwargs["seed"] == 7

    assert result["num_components"] == 2
    assert result["num_selected_nodes"] == 2
    payload = json.loads((tmp_path / "out" / "scores.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "safety"
    assert payload["components"] == {"L0.H0": 1.5, "L1.MLP": 0.5}
    selected = json.loads(
        (tmp_path / "out" / "selected.json").read_text(encoding="utf-8")
    )
    assert selected["nodes"] == ["A0.0", "MLP 1"]
    assert selected["components"][0] == "L0.H0"
    assert "L1.MLP" in selected["components"]
    assert len(selected["components"]) == 3
    assert (tmp_path / "out" / "faithfulness.json").exists()
    assert result["faithfulness_error"] is None
    assert calls["faithfulness"][1]["pillars"] == ["patching", "ablation", "stability"]


def test_run_circuitkit_discovery_records_faithfulness_failure(
    tmp_path: Path, layout: Cohere2Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = _pairs_file(tmp_path)

    def fake_discover(model, task, **kwargs):
        Path(kwargs["output_path"]).write_bytes(b"artifact")
        return FakeCircuit()

    def broken_faithfulness(*args, **kwargs):
        raise RuntimeError("no corruption strategy")

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(ck, "discover", fake_discover)
    monkeypatch.setattr(ck, "faithfulness", broken_faithfulness)

    from circuitgraft.config import FaithfulnessConfig

    result = run_circuitkit_discovery(
        "fake-model",
        pairs,
        tmp_path / "out",
        layout=layout,
        kind="safety",
        device="cpu",
        dtype="float32",
        faithfulness=FaithfulnessConfig(enabled=True),
    )
    assert result["faithfulness_path"] is None
    assert "no corruption strategy" in result["faithfulness_error"]
    assert (tmp_path / "out" / "scores.json").exists()


def test_safetune_cross_check_maps_circuitkit_node_to_head(
    tmp_path: Path, layout: Cohere2Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_discover(*args, **kwargs):
        captured.update(kwargs)
        return {"scores": {"L0.H0": 1.0}, "scores_path": "scores.json"}

    monkeypatch.setattr(runner, "discover_safety_circuit", fake_discover)
    cfg = DiscoverConfig(output_dir=str(tmp_path / "discover"))
    runner.run_safetune_cross_check(
        "fake-model",
        tmp_path / "pairs.jsonl",
        tmp_path / "out",
        layout=layout,
        cfg=cfg,
        device="cpu",
        dtype="float32",
    )
    assert captured["granularity"] == "head"
    assert captured["metric"] == "logit_diff"


def _fake_stage_result(out_dir: Path, *, scores: dict[str, float]) -> dict[str, object]:
    scores_path = out_dir / "scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(
        json.dumps({"components": scores}), encoding="utf-8"
    )
    circuit_info_path = out_dir / "circuit_info.json"
    circuit_info_path.write_text(
        json.dumps({"node_scores": scores, "metadata": {}}), encoding="utf-8"
    )
    return {
        "kind": "safety",
        "model": "fake-model",
        "task": "fake-task",
        "artifact_path": str(out_dir / "circuit.pt"),
        "scores_path": str(scores_path),
        "circuit_info_path": str(circuit_info_path),
        "selection_path": str(out_dir / "selected.json"),
        "num_components": len(scores),
        "num_selected_nodes": len(scores),
        "faithfulness_path": None,
        "faithfulness_error": None,
        "scores": scores,
        "selected": list(scores),
    }


def test_run_discovery_writes_summary_overlap_and_is_idempotent(
    tmp_path: Path, layout: Cohere2Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    safety_pairs = _pairs_file(tmp_path, "safety.jsonl")
    domain_pairs = _pairs_file(tmp_path, "domain.jsonl")
    calls: list[str] = []

    def fake_circuitkit(model_ref, pairs_path, output_dir, *, kind, **kwargs):
        calls.append(kind)
        return _fake_stage_result(
            Path(output_dir),
            scores={"L0.H0": 1.0, "L1.MLP": 1.0} if kind == "safety" else {"L0.H0": 0.5},
        )

    def fake_cross_check(*args, **kwargs):
        path = tmp_path / "safetune_scores.json"
        path.write_text(json.dumps({"components": {"L0.H0": 1.0}}), encoding="utf-8")
        return {
            "scores": {"L0.H0": 1.0},
            "scores_path": str(path),
        }

    monkeypatch.setattr(runner, "run_circuitkit_discovery", fake_circuitkit)
    monkeypatch.setattr(runner, "run_safetune_cross_check", fake_cross_check)

    cfg = DiscoverConfig(
        safety={"pairs": str(safety_pairs), "sparsity": 0.1},
        domain={"pairs": str(domain_pairs), "sparsity": 0.1},
        output_dir=str(tmp_path / "discover"),
        overlap_fractions=[0.1, 0.5],
    )
    base = BaseConfig(
        models=ModelRefs(base="base", aligned="aligned", drifted="drifted")
    )
    first = run_discovery(cfg, base, layout=layout)
    assert first["skipped"] is False
    out_dir = Path(first["output_dir"])
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "overlap.json").exists()
    assert (out_dir / "overlap_0.1.json").exists()
    assert (out_dir / "cross_check.json").exists()
    assert calls == ["safety", "domain"]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["stage"] == "discover"
    assert manifest["inputs"]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "scores" not in summary["safety"]
    assert summary["cross_check_jaccard"] == pytest.approx(1.0)

    second = run_discovery(cfg, base, layout=layout)
    assert second["skipped"] is True
    assert calls == ["safety", "domain"]


def test_run_discovery_requires_a_drifted_model(
    tmp_path: Path, layout: Cohere2Layout
) -> None:
    safety_pairs = _pairs_file(tmp_path, "safety.jsonl")
    domain_pairs = _pairs_file(tmp_path, "domain.jsonl")
    cfg = DiscoverConfig(
        safety={"pairs": str(safety_pairs)},
        domain={"pairs": str(domain_pairs)},
        output_dir=str(tmp_path / "discover"),
    )
    base = BaseConfig(models=ModelRefs(base="base", aligned="aligned", drifted=None))
    with pytest.raises(ValueError, match="models.drifted is empty"):
        run_discovery(cfg, base, layout=layout)


def test_clean_model_ref() -> None:
    assert runner._clean_model_ref("hf:CohereLabs/tiny-aya-global") == "CohereLabs/tiny-aya-global"
    assert runner._clean_model_ref("CohereLabs/tiny-aya-global") == "CohereLabs/tiny-aya-global"
    assert runner._clean_model_ref("  hf:some/repo  ") == "some/repo"


def test_retie_unembed_with_hf_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    class FakeEmbed:
        def __init__(self) -> None:
            self.W_E = torch.randn(100, 20)

    class FakeUnembed:
        def __init__(self, embed_w: torch.Tensor) -> None:
            self.W_U = embed_w.T.clone()

    class FakeModel:
        def __init__(self) -> None:
            self.embed = FakeEmbed()
            self.unembed = FakeUnembed(self.embed.W_E)

    called_ref: list[str] = []

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, ref: str, **kwargs: object) -> SimpleNamespace:
            called_ref.append(ref)
            return SimpleNamespace(logit_scale=1.0)

    import transformers
    monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)

    model = FakeModel()
    assert model.unembed.W_U.data_ptr() != model.embed.W_E.data_ptr()

    runner._retie_unembed(model, "hf:CohereLabs/tiny-aya-global")
    assert called_ref == ["CohereLabs/tiny-aya-global"]
    assert model.unembed.W_U.data_ptr() == model.embed.W_E.data_ptr()


def test_retie_unembed_fallback_when_autoconfig_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    class FakeEmbed:
        def __init__(self) -> None:
            self.W_E = torch.randn(50, 10)

    class FakeUnembed:
        def __init__(self, embed_w: torch.Tensor) -> None:
            self.W_U = embed_w.T.clone()

    class FakeModel:
        def __init__(self) -> None:
            self.embed = FakeEmbed()
            self.unembed = FakeUnembed(self.embed.W_E)

    class BrokenAutoConfig:
        @classmethod
        def from_pretrained(cls, ref: str, **kwargs: object) -> None:
            raise RuntimeError("network down")

    import transformers
    monkeypatch.setattr(transformers, "AutoConfig", BrokenAutoConfig)

    model = FakeModel()
    assert model.unembed.W_U.data_ptr() != model.embed.W_E.data_ptr()
    runner._retie_unembed(model, "hf:CohereLabs/tiny-aya-global")
    assert model.unembed.W_U.data_ptr() == model.embed.W_E.data_ptr()


def test_retie_unembed_skips_when_logit_scale_not_one(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    class FakeEmbed:
        def __init__(self) -> None:
            self.W_E = torch.randn(50, 10)

    class FakeUnembed:
        def __init__(self, embed_w: torch.Tensor) -> None:
            self.W_U = embed_w.T.clone()

    class FakeModel:
        def __init__(self) -> None:
            self.embed = FakeEmbed()
            self.unembed = FakeUnembed(self.embed.W_E)

    class ScaleAutoConfig:
        @classmethod
        def from_pretrained(cls, ref: str, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(logit_scale=0.25)

    import transformers
    monkeypatch.setattr(transformers, "AutoConfig", ScaleAutoConfig)

    model = FakeModel()
    initial_ptr = model.unembed.W_U.data_ptr()
    runner._retie_unembed(model, "CohereLabs/command-r7b")
    assert model.unembed.W_U.data_ptr() == initial_ptr


def test_retie_unembed_skips_when_tensors_differ_even_if_scale_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    class FakeEmbed:
        def __init__(self) -> None:
            self.W_E = torch.randn(50, 10)

    class FakeUnembed:
        def __init__(self) -> None:
            # Different random weights, same transposed shape
            self.W_U = torch.randn(10, 50)

    class FakeModel:
        def __init__(self) -> None:
            self.embed = FakeEmbed()
            self.unembed = FakeUnembed()

    class ScaleOneAutoConfig:
        @classmethod
        def from_pretrained(cls, ref: str, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(logit_scale=1.0)

    import transformers
    monkeypatch.setattr(transformers, "AutoConfig", ScaleOneAutoConfig)

    model = FakeModel()
    initial_ptr = model.unembed.W_U.data_ptr()
    freed = runner._retie_unembed(model, "CohereLabs/tiny-aya-global")
    assert freed == 0
    assert model.unembed.W_U.data_ptr() == initial_ptr


def test_retie_unembed_skips_when_shapes_mismatch() -> None:
    import torch

    class FakeEmbed:
        def __init__(self) -> None:
            self.W_E = torch.randn(50, 10)

    class FakeUnembed:
        def __init__(self) -> None:
            self.W_U = torch.randn(20, 50)

    class FakeModel:
        def __init__(self) -> None:
            self.embed = FakeEmbed()
            self.unembed = FakeUnembed()

    model = FakeModel()
    assert runner._retie_unembed(model, "fake-model") == 0


def test_run_circuitkit_discovery_sets_default_dtype_for_float16(
    tmp_path: Path, layout: Cohere2Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    pairs = _pairs_file(tmp_path)
    captured_default_dtype: list[torch.dtype] = []
    captured_loaded_names: list[str] = []

    def fake_load_model(name: str, *, dtype: str, device: str, algorithm: str) -> object:
        captured_loaded_names.append(name)
        captured_default_dtype.append(torch.get_default_dtype())
        return object()

    def fake_discover(model: object, task: str, **kwargs: object) -> FakeCircuit:
        Path(str(kwargs["output_path"])).write_bytes(b"artifact")
        return FakeCircuit()

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", fake_discover)
    monkeypatch.setattr(runner, "_resolve_device", lambda d: "cuda")
    monkeypatch.setattr(runner, "_resolve_dtype", lambda dt, dev: "float16")

    result = run_circuitkit_discovery(
        "hf:CohereLabs/tiny-aya-global",
        pairs,
        tmp_path / "out_fp16",
        layout=layout,
        kind="safety",
        device="cuda",
        dtype="float16",
    )
    assert captured_loaded_names == ["CohereLabs/tiny-aya-global"]
    assert captured_default_dtype == [torch.float16]
    assert torch.get_default_dtype() == torch.float32
    assert result["dtype"] == "float16"


def test_fast_hf_from_pretrained_preserves_dtype(
    tmp_path: Path, layout: Cohere2Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch
    import transformers

    captured_kwargs: list[dict[str, object]] = []

    class DummyHFModel:
        pass

    def dummy_from_pretrained(*args: object, **kwargs: object) -> DummyHFModel:
        captured_kwargs.append(dict(kwargs))
        return DummyHFModel()

    # Reset any previous patch
    if hasattr(transformers.AutoModelForCausalLM, "_cg_orig_from_pretrained"):
        delattr(transformers.AutoModelForCausalLM, "_cg_orig_from_pretrained")
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", dummy_from_pretrained)
    transformers.AutoModelForCausalLM._cg_patched = False

    # Trigger runner patch by calling run_circuitkit_discovery with fakes
    def fake_discover(model: object, task: str, **kwargs: object) -> FakeCircuit:
        Path(str(kwargs["output_path"])).write_bytes(b"artifact")
        return FakeCircuit()

    import circuitkit as ck
    monkeypatch.setattr(ck, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(ck, "discover", fake_discover)

    # Calling run_circuitkit_discovery installs the patch
    pairs = Path("tests") / "test_circuitkit_runner.py"  # any existing file
    monkeypatch.setattr(runner, "_resolve_device", lambda d: "cpu")
    monkeypatch.setattr(runner, "_resolve_dtype", lambda dt, dev: "float32")
    monkeypatch.setattr(runner, "build_pairs_task", lambda *a, **k: SimpleNamespace(name="dummy"))

    runner.run_circuitkit_discovery(
        "fake-model",
        pairs,
        tmp_path / "dummy_out",
        layout=layout,
        kind="safety",
        device="cpu",
        dtype="float32",
    )

    # Now verify the patched AutoModelForCausalLM.from_pretrained
    captured_kwargs.clear()
    transformers.AutoModelForCausalLM.from_pretrained("dummy", dtype=torch.float16)
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["torch_dtype"] == torch.float16

    captured_kwargs.clear()
    transformers.AutoModelForCausalLM.from_pretrained("dummy", dtype=torch.bfloat16)
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["torch_dtype"] == torch.bfloat16

    captured_kwargs.clear()
    torch.set_default_dtype(torch.float16)
    try:
        transformers.AutoModelForCausalLM.from_pretrained("dummy")
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["torch_dtype"] == torch.float16
    finally:
        torch.set_default_dtype(torch.float32)


def test_patch_circuitkit_eap_memory_installs_and_bounds_max_length() -> None:
    runner.patch_circuitkit_eap_memory(max_seq_len=64)
    import circuitkit.backends.eap.attribute_node as node_mod
    assert getattr(node_mod, "_cg_memory_patched", False) is True



