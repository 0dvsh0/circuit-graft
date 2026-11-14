#!/usr/bin/env python3
"""S4 memory forensics for the 8 GB gfx1032 card.

Answers with measurements (no estimates):
1. What is the HookedTransformer's real dtype/byte footprint after
   ``ck.load_model`` (the fp32-upcast hypothesis for the 13.58 GB OOM)?
2. How much does re-tying ``unembed.W_U`` to ``embed.W_E.T`` free? Exact when
   the checkpoint's ``logit_scale == 1.0`` (the circuitkit Cohere2 port
   materializes W_U as a fresh transposed copy even for tied checkpoints).
3. Where does discovery memory actually go? Phase-by-phase memory counters,
   plus a PyTorch memory-history snapshot dump on OOM for offline analysis.
4. Wall time for n_examples x ig_steps, to extrapolate the full S4 budget.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("HF_HOME", r"D:\cg_hf")
os.environ.setdefault("HF_HUB_CACHE", r"D:\cg_hf\hub")

import torch  # noqa: E402

OUT: dict[str, object] = {}
MODEL_REF = "CohereLabs/tiny-aya-global"


def _flush() -> None:
    out = REPO_ROOT / "results" / "s4_memory_probe.json"
    out.write_text(json.dumps(OUT, indent=2), encoding="utf-8")


def _mem() -> dict[str, float]:
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
    }


def _dtype_split(model: object) -> dict[str, object]:
    bytes_by_dtype: dict[str, int] = defaultdict(int)
    params = 0
    for _name, p in model.named_parameters():  # type: ignore[attr-defined]
        bytes_by_dtype[str(p.dtype)] += p.numel() * p.element_size()
        params += p.numel()
    return {
        "total_params": params,
        "bytes_by_dtype_gb": {
            k: round(v / 2**30, 2) for k, v in bytes_by_dtype.items()
        },
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("FATAL: no cuda")
        return 1
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import circuitkit as ck

    from circuitgraft.discover.circuitkit_runner import (
        _apply_memory_controls,
        build_pairs_task,
        patch_transformer_lens_weight_processing,
    )

    patch_transformer_lens_weight_processing()
    import transformers

    if not getattr(transformers.AutoModelForCausalLM, "_cg_patched", False):
        orig_hf = transformers.AutoModelForCausalLM.from_pretrained

        def _fast_hf(*hf_args, **hf_kwargs):
            if "torch_dtype" not in hf_kwargs:
                hf_kwargs["torch_dtype"] = torch.bfloat16
            return orig_hf(*hf_args, **hf_kwargs)

        transformers.AutoModelForCausalLM.from_pretrained = staticmethod(_fast_hf)
        transformers.AutoModelForCausalLM._cg_patched = True

    task = build_pairs_task(str(REPO_ROOT / "artifacts/data/pairs/safety.jsonl"))

    t0 = time.perf_counter()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = ck.load_model(
            MODEL_REF,
            dtype="bfloat16",
            device="cuda",
            algorithm="eap-ig",
        )
    finally:
        torch.set_default_dtype(torch.float32)
    OUT["load_wall_s"] = round(time.perf_counter() - t0, 1)
    OUT["after_load_mem"] = _mem()
    OUT["dtype_split"] = _dtype_split(model)
    cfg = model.cfg
    OUT["cfg"] = {
        "n_ctx": getattr(cfg, "n_ctx", None),
        "d_model": getattr(cfg, "d_model", None),
        "d_vocab": getattr(cfg, "d_vocab", None),
        "d_vocab_out": getattr(cfg, "d_vocab_out", None),
        "n_layers": getattr(cfg, "n_layers", None),
        "n_heads": getattr(cfg, "n_heads", None),
        "ungroup_grouped_query_attention": getattr(
            cfg, "ungroup_grouped_query_attention", None
        ),
        "use_attn_result": getattr(cfg, "use_attn_result", None),
        "use_split_qkv_input": getattr(cfg, "use_split_qkv_input", None),
        "use_hook_mlp_in": getattr(cfg, "use_hook_mlp_in", None),
        "tie_embeddings": getattr(cfg, "tie_embeddings", None),
    }
    embed = model.embed.W_E
    unembed = model.unembed.W_U
    OUT["tie_state"] = {
        "W_E_shape": list(embed.shape),
        "W_U_shape": list(unembed.shape),
        "shared_storage": embed.data_ptr() == unembed.data_ptr(),
        "W_E_dtype": str(embed.dtype),
        "W_U_dtype": str(unembed.dtype),
    }
    print("after_load:", json.dumps(OUT, default=str)[:2000])
    _flush()

    _apply_memory_controls(model, max_seq_len=256)
    OUT["after_memory_controls_mem"] = _mem()

    # Retie: zero-copy transposed view, exact because the port folds
    # logit_scale into W_U and this checkpoint's logit_scale is 1.0.
    try:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(MODEL_REF)
        scale = float(getattr(hf_cfg, "logit_scale", 1.0))
        can_tie = (
            scale == 1.0
            and tuple(unembed.shape) == tuple(embed.shape)[::-1]
            and embed.data_ptr() != unembed.data_ptr()
        )
        if can_tie:
            before = torch.cuda.memory_allocated()
            old_wu = unembed.data
            model.unembed.W_U.data = embed.data.T  # zero-copy transposed view
            del old_wu, unembed
            gc.collect()
            torch.cuda.empty_cache()
            unembed = model.unembed.W_U
            OUT["retie"] = {
                "logit_scale": scale,
                "freed_gb": round(
                    (before - torch.cuda.memory_allocated()) / 2**30, 2
                ),
            }
        else:
            OUT["retie"] = {"skipped": True, "logit_scale": scale}
    except Exception as exc:  # noqa: BLE001
        OUT["retie"] = {"error": f"{type(exc).__name__}: {exc}"}
    print("retie:", json.dumps(OUT.get("retie", {})))
    OUT["after_retie_mem"] = _mem()
    _flush()

    import contextlib

    with contextlib.suppress(Exception):
        torch.cuda.memory._record_memory_history(
            enabled=True, context="all", stacks="all", max_entries=20000
        )

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        circuit = ck.discover(
            model,
            task.name,
            algorithm="eap-ig",
            level="node",
            n_examples=2,
            batch_size=1,
            sparsity=0.3,
            scope="both",
            output_path=str(REPO_ROOT / "artifacts/probe_s4_memory/circuit.pt"),
            ig_steps=2,
            max_seq_len=256,
            intervention="patching",
            seed=0,
        )
        OUT["discover"] = {
            "status": "ok",
            "wall_s": round(time.perf_counter() - t0, 1),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        }
        del circuit
    except Exception as exc:  # noqa: BLE001
        OUT["discover"] = {
            "status": f"error: {type(exc).__name__}",
            "message": str(exc)[:600],
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        try:
            snap = torch.cuda.memory._dump_snapshot()
            out = REPO_ROOT / "results" / "s4_oom_snapshot.pickle"
            with open(out, "wb") as f:
                pickle.dump(snap, f)
            OUT["discover"]["snapshot"] = str(out)
        except Exception as exc2:  # noqa: BLE001
            OUT["discover"]["snapshot_error"] = str(exc2)
    finally:
        with contextlib.suppress(Exception):
            torch.cuda.memory._record_memory_history(enabled=None)
    print("discover:", json.dumps(OUT["discover"], default=str)[:1200])
    _flush()
    print(f"results: {REPO_ROOT / 'results' / 's4_memory_probe.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
