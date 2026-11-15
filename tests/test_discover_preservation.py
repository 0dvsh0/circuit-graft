"""Preservation property tests for CircuitKIT discovery runner.

These tests encode Property 2 (Preservation: Non-Buggy Inputs Produce Identical Behavior)
and validate Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, and 3.7.

CRITICAL: These tests MUST PASS on the UNFIXED code to establish the baseline behavior
that must not be broken or altered by the T4 CUDA OOM bugfix.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import weakref

import pytest
import torch

from circuitgraft.components import Cohere2Layout
from circuitgraft.config import BaseConfig, DiscoverConfig, FaithfulnessConfig, ModelRefs
from circuitgraft.discover import circuitkit_runner as runner
from circuitgraft.discover.circuitkit_runner import (
    _apply_memory_controls,
    run_circuitkit_discovery,
    run_discovery,
)


def _pairs_file(tmp_path: Path, name: str = "safety.jsonl") -> Path:
    """Create a minimal pairs JSONL file for discovery tests."""
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
    """Mock CircuitKIT circuit output."""

    def __init__(self) -> None:
        self.scores = {"A0.0": 1.5, "MLP 1": 0.5}
        self.nodes = ["A0.0", "MLP 1"]


class DummyModel:
    """Mock model with configurable n_ctx and parameters for memory controls."""

    def __init__(self, n_ctx: int = 1024) -> None:
        self.cfg = SimpleNamespace(n_ctx=n_ctx)

    def named_parameters(self):
        return [
            ("embed_tokens.weight", SimpleNamespace(requires_grad=True)),
            ("layers.0.self_attn.q_proj.weight", SimpleNamespace(requires_grad=True)),
        ]


# ---------------------------------------------------------------------------
# 1. Observation & Preservation: CPU device with any n_examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_examples", [16, 32, 64, 128, 256, 512])
def test_cpu_preserves_n_examples(
    tmp_path: Path,
    layout: Cohere2Layout,
    monkeypatch: pytest.MonkeyPatch,
    n_examples: int,
) -> None:
    """Observe & preserve: CPU runs pass n_examples to ck.discover completely unchanged."""
    pairs = _pairs_file(tmp_path)
    captured_kwargs: dict[str, Any] = {}

    def fake_load_model(*args: Any, **kwargs: Any) -> DummyModel:
        return DummyModel()

    def fake_discover(model: Any, task: str, **kwargs: Any) -> FakeCircuit:
        captured_kwargs.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"artifact")
        return FakeCircuit()

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", fake_discover)

    out_dir = tmp_path / f"out_cpu_{n_examples}"
    result = run_circuitkit_discovery(
        "fake-model",
        pairs,
        out_dir,
        layout=layout,
        kind="safety",
        n_examples=n_examples,
        device="cpu",
        dtype="float32",
    )

    assert captured_kwargs["n_examples"] == n_examples
    assert result["device"] == "cpu"
    assert (out_dir / "circuit.pt").exists()
    assert (out_dir / "scores.json").exists()


# ---------------------------------------------------------------------------
# 2. Observation & Preservation: Large CUDA device (>15,500 MiB)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total_vram_mib", "n_examples"),
    [
        (40_960, 256),  # A100 (40 GiB) with standard n_examples=256
        (40_960, 512),  # A100 with large n_examples=512
        (81_920, 256),  # H100 (80 GiB) with n_examples=256
        (24_576, 128),  # RTX 3090 / 4090 (24 GiB)
        (16_000, 256),  # Just above 15,500 MiB threshold
    ],
)
def test_large_cuda_preserves_n_examples(
    tmp_path: Path,
    layout: Cohere2Layout,
    monkeypatch: pytest.MonkeyPatch,
    total_vram_mib: int,
    n_examples: int,
) -> None:
    """Observe & preserve: Large CUDA devices (>15.5 GiB) pass n_examples unchanged."""
    pairs = _pairs_file(tmp_path)
    captured_kwargs: dict[str, Any] = {}

    def fake_load_model(*args: Any, **kwargs: Any) -> DummyModel:
        return DummyModel()

    def fake_discover(model: Any, task: str, **kwargs: Any) -> FakeCircuit:
        captured_kwargs.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"artifact")
        return FakeCircuit()

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", fake_discover)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda dev=0: SimpleNamespace(total_memory=total_vram_mib * 1024 * 1024),
    )

    out_dir = tmp_path / f"out_large_{total_vram_mib}_{n_examples}"
    result = run_circuitkit_discovery(
        "fake-model",
        pairs,
        out_dir,
        layout=layout,
        kind="safety",
        n_examples=n_examples,
        device="cuda",
        dtype="float32",
    )

    assert captured_kwargs["n_examples"] == n_examples
    assert result["device"] == "cuda"


# ---------------------------------------------------------------------------
# 3. Observation & Preservation: T4-class device with small n_examples (<= 64)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total_vram_mib", "n_examples"),
    [
        (14_566, 16),  # Kaggle T4 with small n_examples=16
        (14_566, 32),  # Kaggle T4 with small n_examples=32
        (14_566, 48),  # Kaggle T4 with small n_examples=48
        (14_566, 64),  # Kaggle T4 at clamp boundary n_examples=64 (must NOT be clamped below 64)
        (15_000, 32),  # Constrained device with 32
        (15_500, 64),  # Exactly at 15,500 MiB threshold boundary with 64
        (8_000, 32),   # 8 GiB card with small n_examples
    ],
)
def test_t4_small_n_examples_preserved(
    tmp_path: Path,
    layout: Cohere2Layout,
    monkeypatch: pytest.MonkeyPatch,
    total_vram_mib: int,
    n_examples: int,
) -> None:
    """Observe & preserve: T4-class devices with n_examples <= 64 pass through unchanged."""
    pairs = _pairs_file(tmp_path)
    captured_kwargs: dict[str, Any] = {}

    def fake_load_model(*args: Any, **kwargs: Any) -> DummyModel:
        return DummyModel()

    def fake_discover(model: Any, task: str, **kwargs: Any) -> FakeCircuit:
        captured_kwargs.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"artifact")
        return FakeCircuit()

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", fake_discover)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda dev=0: SimpleNamespace(total_memory=total_vram_mib * 1024 * 1024),
    )

    out_dir = tmp_path / f"out_t4_small_{total_vram_mib}_{n_examples}"
    result = run_circuitkit_discovery(
        "fake-model",
        pairs,
        out_dir,
        layout=layout,
        kind="safety",
        n_examples=n_examples,
        device="cuda",
        dtype="float32",
    )

    assert captured_kwargs["n_examples"] == n_examples
    assert result["device"] == "cuda"


# ---------------------------------------------------------------------------
# 4. Property-Based Preservation Test: All Non-Buggy Triples
# ---------------------------------------------------------------------------


def _is_bug_condition(device: str, n_examples: int, total_vram_mib: int) -> bool:
    """Formal predicate for Bug_Condition(C) as defined in design.md."""
    is_constrained_cuda = device == "cuda" and total_vram_mib <= 15_500
    return is_constrained_cuda and n_examples > 64


def _generate_non_buggy_triples() -> list[tuple[str, int, int]]:
    """Generate representative triples (device, n_examples, total_vram_mib) where bug condition is false."""
    triples: list[tuple[str, int, int]] = []

    # CPU configurations: any n_examples is non-buggy
    for n in [1, 16, 32, 64, 96, 128, 256, 512]:
        triples.append(("cpu", n, 0))

    # Large CUDA configurations (>15,500 MiB): any n_examples is non-buggy
    for vram in [16_000, 24_576, 40_960, 81_920]:
        for n in [32, 64, 128, 256, 512]:
            triples.append(("cuda", n, vram))

    # Constrained CUDA configurations (<=15,500 MiB): n_examples <= 64 is non-buggy
    for vram in [8_000, 12_000, 14_566, 15_000, 15_500]:
        for n in [1, 16, 32, 48, 64]:
            triples.append(("cuda", n, vram))

    return triples


@pytest.mark.parametrize(
    ("device", "n_examples", "total_vram_mib"),
    _generate_non_buggy_triples(),
)
def test_property_non_buggy_triples_preserve_n_examples(
    tmp_path: Path,
    layout: Cohere2Layout,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    n_examples: int,
    total_vram_mib: int,
) -> None:
    """Property 2: For all non-buggy configurations, n_examples passes to ck.discover unchanged."""
    assert not _is_bug_condition(device, n_examples, total_vram_mib)

    pairs = _pairs_file(tmp_path)
    captured_kwargs: dict[str, Any] = {}

    def fake_load_model(*args: Any, **kwargs: Any) -> DummyModel:
        return DummyModel()

    def fake_discover(model: Any, task: str, **kwargs: Any) -> FakeCircuit:
        captured_kwargs.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"artifact")
        return FakeCircuit()

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", fake_discover)

    if device == "cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda dev=0: SimpleNamespace(total_memory=total_vram_mib * 1024 * 1024),
        )

    out_dir = tmp_path / f"out_pbt_{device}_{total_vram_mib}_{n_examples}"
    result = run_circuitkit_discovery(
        "fake-model",
        pairs,
        out_dir,
        layout=layout,
        kind="safety",
        n_examples=n_examples,
        device=device,
        dtype="float32",
    )

    assert captured_kwargs["n_examples"] == n_examples
    assert result["device"] == device


# ---------------------------------------------------------------------------
# 5. Observation & Preservation: finally-block cleanup on exception (Req 3.6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exception_to_raise",
    [
        RuntimeError("arbitrary discovery failure"),
        torch.cuda.OutOfMemoryError("CUDA out of memory in test"),
        ValueError("invalid node scoring"),
    ],
)
def test_finally_block_cleanup_on_exception(
    tmp_path: Path,
    layout: Cohere2Layout,
    monkeypatch: pytest.MonkeyPatch,
    exception_to_raise: Exception,
) -> None:
    """Observe & preserve (Req 3.6): finally block deletes model, calls gc.collect() and _empty_cuda_cache()."""
    pairs = _pairs_file(tmp_path)
    events: list[str] = []

    orig_gc = gc.collect
    orig_empty = runner._empty_cuda_cache

    def spy_gc(*args: Any, **kwargs: Any) -> int:
        events.append("gc_collect")
        return orig_gc(*args, **kwargs)

    def spy_empty() -> None:
        events.append("empty_cuda_cache")
        orig_empty()

    monkeypatch.setattr(gc, "collect", spy_gc)
    monkeypatch.setattr(runner, "_empty_cuda_cache", spy_empty)

    model_refs: list[weakref.ReferenceType[DummyModel]] = []

    def fake_load_model(*args: Any, **kwargs: Any) -> DummyModel:
        model = DummyModel()
        model_refs.append(weakref.ref(model))
        return model

    def failing_discover(*args: Any, **kwargs: Any) -> FakeCircuit:
        events.append("discover_failed")
        # Clear args and kwargs before raising so this frame does not hold the model
        del args, kwargs
        raise exception_to_raise

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", failing_discover)

    out_dir = tmp_path / "out_failure"
    with pytest.raises(type(exception_to_raise), match=str(exception_to_raise)):
        run_circuitkit_discovery(
            "fake-model",
            pairs,
            out_dir,
            layout=layout,
            kind="safety",
            n_examples=32,
            device="cpu",
            dtype="float32",
        )

    # Verify discovery failed and then cleanup events occurred in finally:
    assert "discover_failed" in events
    fail_idx = events.index("discover_failed")
    post_discover_events = events[fail_idx + 1 :]

    assert "gc_collect" in post_discover_events
    assert "empty_cuda_cache" in post_discover_events

    # Verify del model ran in finally: block (weakref is dead)
    assert len(model_refs) == 1
    assert model_refs[0]() is None


# ---------------------------------------------------------------------------
# 6. Observation & Preservation: finally-block cleanup on successful run
# ---------------------------------------------------------------------------


def test_finally_block_cleanup_on_success(
    tmp_path: Path,
    layout: Cohere2Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe & preserve: finally block runs del model, gc.collect(), _empty_cuda_cache() on success."""
    pairs = _pairs_file(tmp_path)
    events: list[str] = []

    orig_gc = gc.collect
    orig_empty = runner._empty_cuda_cache

    def spy_gc(*args: Any, **kwargs: Any) -> int:
        events.append("gc_collect")
        return orig_gc(*args, **kwargs)

    def spy_empty() -> None:
        events.append("empty_cuda_cache")
        orig_empty()

    monkeypatch.setattr(gc, "collect", spy_gc)
    monkeypatch.setattr(runner, "_empty_cuda_cache", spy_empty)

    model_refs: list[weakref.ReferenceType[DummyModel]] = []

    def fake_load_model(*args: Any, **kwargs: Any) -> DummyModel:
        model = DummyModel()
        model_refs.append(weakref.ref(model))
        return model

    def fake_discover(model: Any, task: str, **kwargs: Any) -> FakeCircuit:
        events.append("discover_done")
        Path(kwargs["output_path"]).write_bytes(b"artifact")
        return FakeCircuit()

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", fake_discover)

    out_dir = tmp_path / "out_success"
    result = run_circuitkit_discovery(
        "fake-model",
        pairs,
        out_dir,
        layout=layout,
        kind="safety",
        n_examples=32,
        device="cpu",
        dtype="float32",
    )

    assert result["num_components"] == 2
    assert "discover_done" in events
    done_idx = events.index("discover_done")
    post_discover_events = events[done_idx + 1 :]

    assert "gc_collect" in post_discover_events
    assert "empty_cuda_cache" in post_discover_events
    assert len(model_refs) == 1
    assert model_refs[0]() is None


# ---------------------------------------------------------------------------
# 7. Preservation: Canonical Artifact Schema & Content (Req 3.1)
# ---------------------------------------------------------------------------


def test_canonical_artifacts_and_schema_preservation(
    tmp_path: Path,
    layout: Cohere2Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe & preserve (Req 3.1): canonical artifacts match expected schema exactly."""
    pairs = _pairs_file(tmp_path)

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        ck,
        "discover",
        lambda m, t, **kwargs: (
            Path(kwargs["output_path"]).write_bytes(b"artifact"),
            FakeCircuit(),
        )[1],
    )

    out_dir = tmp_path / "out_artifacts"
    result = run_circuitkit_discovery(
        "fake-model",
        pairs,
        out_dir,
        layout=layout,
        kind="safety",
        n_examples=32,
        sparsity=0.25,
        ig_steps=3,
        batch_size=2,
        max_seq_len=128,
        device="cpu",
        dtype="float32",
        seed=42,
    )

    # 1. circuit.pt
    circuit_pt = out_dir / "circuit.pt"
    assert circuit_pt.exists()
    assert circuit_pt.read_bytes() == b"artifact"

    # 2. scores.json
    scores_file = out_dir / "scores.json"
    assert scores_file.exists()
    scores_data = json.loads(scores_file.read_text(encoding="utf-8"))
    assert scores_data["kind"] == "safety"
    assert scores_data["model"] == "fake-model"
    assert scores_data["source"] == "circuitkit"
    assert scores_data["components"] == {"L0.H0": 1.5, "L1.MLP": 0.5}
    meta = scores_data["metadata"]
    assert meta["algorithm"] == "eap-ig"
    assert meta["n_examples"] == 32
    assert meta["sparsity"] == 0.25
    assert meta["ig_steps"] == 3

    # 3. circuit_info.json
    circuit_info_file = out_dir / "circuit_info.json"
    assert circuit_info_file.exists()
    circuit_info = json.loads(circuit_info_file.read_text(encoding="utf-8"))
    assert "node_scores" in circuit_info
    assert circuit_info["metadata"]["scores_file"] == "scores.json"
    assert "scores_sha256" in circuit_info["metadata"]

    # 4. selected.json
    selected_file = out_dir / "selected.json"
    assert selected_file.exists()
    selected = json.loads(selected_file.read_text(encoding="utf-8"))
    assert selected["kind"] == "safety"
    assert selected["nodes"] == ["A0.0", "MLP 1"]
    assert selected["components"][0] == "L0.H0"

    # 5. Return dict
    assert result["kind"] == "safety"
    assert result["num_components"] == 2
    assert result["scores"] == {"L0.H0": 1.5, "L1.MLP": 0.5}


# ---------------------------------------------------------------------------
# 8. Preservation: Faithfulness Execution (Req 3.5)
# ---------------------------------------------------------------------------


def test_faithfulness_preservation(
    tmp_path: Path,
    layout: Cohere2Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe & preserve (Req 3.5): faithfulness checks run and produce faithfulness.json."""
    pairs = _pairs_file(tmp_path)
    captured_faith: dict[str, Any] = {}

    class FakeReport:
        def to_json(self, path: Path) -> None:
            Path(path).write_text('{"patching_score": 0.95}', encoding="utf-8")

    def fake_faithfulness(model: Any, circuit: Any, task: str, **kwargs: Any) -> FakeReport:
        captured_faith.update(kwargs)
        return FakeReport()

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        ck,
        "discover",
        lambda m, t, **kwargs: (
            Path(kwargs["output_path"]).write_bytes(b"artifact"),
            FakeCircuit(),
        )[1],
    )
    monkeypatch.setattr(ck, "faithfulness", fake_faithfulness)

    out_dir = tmp_path / "out_faith"
    result = run_circuitkit_discovery(
        "fake-model",
        pairs,
        out_dir,
        layout=layout,
        kind="safety",
        n_examples=32,
        device="cpu",
        dtype="float32",
        faithfulness=FaithfulnessConfig(enabled=True, n_examples=4),
    )

    assert (out_dir / "faithfulness.json").exists()
    assert result["faithfulness_error"] is None
    assert captured_faith["pillars"] == ["patching", "ablation", "stability"]
    assert captured_faith["n_examples"] == 4


# ---------------------------------------------------------------------------
# 9. Preservation: Skip-Check Caching (Req 3.7)
# ---------------------------------------------------------------------------


def test_skip_check_preservation(
    tmp_path: Path,
    layout: Cohere2Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe & preserve (Req 3.7): cached run returns without re-executing discovery."""
    safety_pairs = _pairs_file(tmp_path, "safety.jsonl")
    domain_pairs = _pairs_file(tmp_path, "domain.jsonl")
    discovery_calls: list[str] = []

    def fake_circuitkit(model_ref: Any, pairs_path: Any, output_dir: Any, *, kind: str, **kwargs: Any):
        discovery_calls.append(kind)
        scores = {"L0.H0": 1.0}
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "scores.json").write_text(json.dumps({"components": scores}), encoding="utf-8")
        (out_p / "circuit_info.json").write_text(json.dumps({"node_scores": scores}), encoding="utf-8")
        (out_p / "selected.json").write_text(json.dumps({"nodes": ["L0.H0"]}), encoding="utf-8")
        (out_p / "circuit.pt").write_bytes(b"circuit")
        return {
            "kind": kind,
            "model": "fake-model",
            "task": "fake-task",
            "artifact_path": str(out_p / "circuit.pt"),
            "scores_path": str(out_p / "scores.json"),
            "circuit_info_path": str(out_p / "circuit_info.json"),
            "selection_path": str(out_p / "selected.json"),
            "num_components": 1,
            "num_selected_nodes": 1,
            "faithfulness_path": None,
            "faithfulness_error": None,
            "scores": scores,
            "selected": ["L0.H0"],
        }

    monkeypatch.setattr(runner, "run_circuitkit_discovery", fake_circuitkit)
    monkeypatch.setattr(
        runner,
        "run_safetune_cross_check",
        lambda *a, **k: {"scores": {"L0.H0": 1.0}, "scores_path": "scores.json"},
    )

    cfg = DiscoverConfig(
        safety={"pairs": str(safety_pairs), "sparsity": 0.1},
        domain={"pairs": str(domain_pairs), "sparsity": 0.1},
        output_dir=str(tmp_path / "discover"),
        overlap_fractions=[0.1],
    )
    base = BaseConfig(models=ModelRefs(base="base", aligned="aligned", drifted="drifted"))

    first = run_discovery(cfg, base, layout=layout)
    assert first["skipped"] is False
    assert discovery_calls == ["safety", "domain"]

    # Second invocation should skip re-running discovery
    second = run_discovery(cfg, base, layout=layout)
    assert second["skipped"] is True
    assert discovery_calls == ["safety", "domain"]


# ---------------------------------------------------------------------------
# 10. Preservation: Memory Controls & Context Capping (Req 3.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_ctx", "max_seq_len", "expected_ctx"),
    [
        (2048, 128, 128),  # caps large context to max_seq_len
        (2048, 20, 48),    # respects floor of 48
        (32, 128, 32),     # does not expand small context beyond current_ctx
    ],
)
def test_memory_controls_cap_context_preservation(
    input_ctx: int,
    max_seq_len: int,
    expected_ctx: int,
) -> None:
    """Observe & preserve (Req 3.4): _apply_memory_controls caps n_ctx correctly."""
    model = DummyModel(n_ctx=input_ctx)
    _apply_memory_controls(model, max_seq_len=max_seq_len)
    assert model.cfg.n_ctx == expected_ctx
