"""Bug condition exploration tests for discover-safety CUDA OOM on T4 GPU.

Task 1 from .kiro/specs/discover-safety-oom/tasks.md:
Surface counterexamples demonstrating both root causes on unfixed code:
1. gc.collect() and torch.cuda.empty_cache() are NOT called between
   _apply_memory_controls() and ck.discover().
2. When n_examples=256 is passed on a T4-class GPU (total_memory <= 15,500 MiB),
   it is not clamped to 64, causing an oversized patch cache (~656 MiB vs ~164 MiB).

CRITICAL:
These tests encode the expected behavior and MUST FAIL on the current unfixed code.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from circuitgraft.components import Cohere2Layout
from circuitgraft.discover import circuitkit_runner as runner
from circuitgraft.discover.circuitkit_runner import run_circuitkit_discovery


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


def test_prediscovery_allocator_flush_called_before_discover(
    tmp_path: Path, layout: Cohere2Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Property 1: Pre-discovery allocator flush.

    Requirement 2.2, 2.3: gc.collect() and _empty_cuda_cache() must be called
    between _apply_memory_controls() and ck.discover() on CUDA devices so that
    stale allocator blocks from model-loading are returned before EAP-IG allocation.
    """
    pairs = _pairs_file(tmp_path)
    events: list[str] = []

    def fake_load_model(name, *, dtype, device, algorithm):
        return SimpleNamespace(cfg=SimpleNamespace(n_ctx=2048), named_parameters=lambda: [])

    def fake_discover(model, task, **kwargs):
        events.append("discover")
        Path(kwargs["output_path"]).write_bytes(b"artifact")
        return FakeCircuit()

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", fake_discover)

    # Mock CUDA environment to simulate T4 (14.56 GiB <= 15,500 MiB)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    t4_props = SimpleNamespace(
        name="Tesla T4",
        total_memory=14_566 * 1024 * 1024,
        major=7,
        minor=5,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: t4_props)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)

    # Spy on _apply_memory_controls completion
    orig_apply_mem = runner._apply_memory_controls

    def spy_apply_memory_controls(*args, **kwargs):
        res = orig_apply_mem(*args, **kwargs)
        events.append("apply_memory_controls_finished")
        return res

    monkeypatch.setattr(runner, "_apply_memory_controls", spy_apply_memory_controls)

    # Spy on _empty_cuda_cache
    orig_empty_cache = runner._empty_cuda_cache

    def spy_empty_cache(*args, **kwargs):
        events.append("empty_cache")
        return orig_empty_cache(*args, **kwargs)

    monkeypatch.setattr(runner, "_empty_cuda_cache", spy_empty_cache)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    run_circuitkit_discovery(
        "fake-model",
        pairs,
        tmp_path / "out",
        layout=layout,
        kind="safety",
        n_examples=256,
        device="cuda",
    )

    assert "apply_memory_controls_finished" in events
    assert "discover" in events

    mem_ctrl_idx = events.index("apply_memory_controls_finished")
    discover_idx = events.index("discover")
    flush_count_between = events[mem_ctrl_idx + 1 : discover_idx].count("empty_cache")

    assert flush_count_between >= 1, (
        f"Expected at least one empty_cache call between _apply_memory_controls and ck.discover, "
        f"but found {flush_count_between}. "
        "Counterexample: allocator_flushed_before_discover is False (stale allocator blocks retained)."
    )


@pytest.mark.parametrize(
    ("total_memory_mib", "input_n_examples"),
    [
        (14_566, 256),  # Kaggle T4 usable capacity
        (15_000, 256),  # Standard 15 GiB T4
        (15_499, 256),  # Boundary case just below 15,500 MiB threshold
        (14_566, 128),  # Intermediate n_examples > 64
    ],
)
def test_t4_runtime_clamp_n_examples_to_64(
    tmp_path: Path,
    layout: Cohere2Layout,
    monkeypatch: pytest.MonkeyPatch,
    total_memory_mib: int,
    input_n_examples: int,
) -> None:
    """Property 1: T4-class runtime clamp.

    Requirement 2.4: When running on a CUDA GPU with <= 15,500 MiB VRAM and
    n_examples > 64, n_examples must be clamped to 64 to fit EAP-IG patch cache
    within available VRAM (~164 MiB vs ~656 MiB).
    """
    pairs = _pairs_file(tmp_path)
    captured_kwargs: dict[str, Any] = {}

    def fake_load_model(*args, **kwargs):
        return SimpleNamespace(cfg=SimpleNamespace(n_ctx=2048), named_parameters=lambda: [])

    def fake_discover(model, task, **kwargs):
        captured_kwargs.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"artifact")
        return FakeCircuit()

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", fake_discover)

    # Mock CUDA environment with total_memory <= 15,500 MiB
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    t4_props = SimpleNamespace(
        name="Tesla T4",
        total_memory=total_memory_mib * 1024 * 1024,
        major=7,
        minor=5,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: t4_props)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)

    run_circuitkit_discovery(
        "fake-model",
        pairs,
        tmp_path / "out",
        layout=layout,
        kind="safety",
        n_examples=input_n_examples,
        device="cuda",
    )

    actual_n_examples = captured_kwargs.get("n_examples")
    assert actual_n_examples == 64, (
        f"Expected n_examples to be clamped to 64 on T4-class GPU (total_memory={total_memory_mib} MiB <= 15500 MiB), "
        f"but got {actual_n_examples}. "
        f"Counterexample: input n_examples={input_n_examples} passed unclamped causing oversized patch cache "
        f"(~656 MiB vs ~164 MiB)."
    )


def test_t4_discovery_avoids_oom_by_clamping(
    tmp_path: Path, layout: Cohere2Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Property 1: End-to-end discovery completes without OOM on T4-class GPU.

    Requirements 2.1, 2.4:
    Simulates the real T4 failure where CircuitKIT raises CUDA out of memory when
    n_examples > 64 because the patch cache (~656 MiB for n_examples=256)
    exceeds available free VRAM (~450 MiB).
    When n_examples is clamped to 64 (~164 MiB patch cache), discovery succeeds.
    """
    pairs = _pairs_file(tmp_path)
    out_dir = tmp_path / "out"

    def fake_load_model(*args, **kwargs):
        return SimpleNamespace(cfg=SimpleNamespace(n_ctx=2048), named_parameters=lambda: [])

    def fake_discover(model, task, **kwargs):
        requested_examples = kwargs.get("n_examples", 0)
        if requested_examples > 64:
            raise torch.cuda.OutOfMemoryError(
                f"CUDA out of memory: EAP-IG patch cache for n_examples={requested_examples} "
                f"(~656 MiB) exceeds available free VRAM (~450 MiB) on T4 GPU"
            )
        Path(kwargs["output_path"]).write_bytes(b"artifact")
        return FakeCircuit()

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", fake_discover)

    # Mock CUDA T4 device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    t4_props = SimpleNamespace(
        name="Tesla T4",
        total_memory=14_566 * 1024 * 1024,
        major=7,
        minor=5,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: t4_props)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)

    # In expected behavior, calling run_circuitkit_discovery with n_examples=256
    # clamps to 64, avoiding the OOM and writing canonical artifacts.
    result = run_circuitkit_discovery(
        "fake-model",
        pairs,
        out_dir,
        layout=layout,
        kind="safety",
        n_examples=256,
        device="cuda",
    )

    assert (out_dir / "scores.json").exists()
    assert (out_dir / "selected.json").exists()
    assert (out_dir / "circuit_info.json").exists()
    assert (out_dir / "circuit.pt").exists()
    assert result["num_components"] == 2


def test_oom_propagates_and_finally_cleanup_executes(
    tmp_path: Path, layout: Cohere2Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 1.1, 3.6: Verify that when ck.discover raises CUDA OOM,
    the exception propagates to the caller and the finally block executes
    model deletion, gc.collect(), and _empty_cuda_cache().
    """
    pairs = _pairs_file(tmp_path)
    events: list[str] = []

    def fake_load_model(*args, **kwargs):
        return SimpleNamespace(cfg=SimpleNamespace(n_ctx=2048), named_parameters=lambda: [])

    def fake_discover(*args, **kwargs):
        events.append("discover_raising_oom")
        raise torch.cuda.OutOfMemoryError("CUDA out of memory: simulated EAP-IG allocation failure")

    orig_gc_collect = gc.collect

    def spy_gc_collect(*args, **kwargs):
        events.append("gc_collect")
        return orig_gc_collect(*args, **kwargs)

    orig_empty_cache = runner._empty_cuda_cache

    def spy_empty_cache(*args, **kwargs):
        events.append("empty_cuda_cache")
        return orig_empty_cache(*args, **kwargs)

    import circuitkit as ck

    monkeypatch.setattr(ck, "load_model", fake_load_model)
    monkeypatch.setattr(ck, "discover", fake_discover)
    monkeypatch.setattr(gc, "collect", spy_gc_collect)
    monkeypatch.setattr(runner, "_empty_cuda_cache", spy_empty_cache)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    t4_props = SimpleNamespace(
        name="Tesla T4",
        total_memory=14_566 * 1024 * 1024,
        major=7,
        minor=5,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: t4_props)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="CUDA out of memory"):
        run_circuitkit_discovery(
            "fake-model",
            pairs,
            tmp_path / "out",
            layout=layout,
            kind="safety",
            n_examples=256,
            device="cuda",
        )

    assert "discover_raising_oom" in events
    oom_idx = events.index("discover_raising_oom")
    after_oom_events = events[oom_idx + 1 :]
    assert "gc_collect" in after_oom_events, (
        "Expected gc.collect() in finally block after OutOfMemoryError"
    )
    assert "empty_cuda_cache" in after_oom_events, (
        "Expected _empty_cuda_cache() in finally block after OutOfMemoryError"
    )
