#!/usr/bin/env python3
"""Empirical VRAM/speed probe for S2 LoRA training shapes on the 8 GB card.

The bf16 bench showed 43.8 s/micro-step at peak 10.3 GB: the frozen bf16 base
(6.7 GB) + CE-backward logits buffers exceed 7.98 GB VRAM, so Windows WDDM
spills to shared memory and every step thrashes PCIe. This probe tests the
QLoRA-style fix -- NF4 4-bit frozen base + bf16 LoRA -- across a few shapes:

  a) checkpointing, batch 2, seq 512   (planned S2 micro shape)
  b) no checkpointing, batch 1, seq 512
  c) checkpointing, batch 1, seq 512
  d) no checkpointing, batch 1, seq 512 (variant b re-run for stability)

Reports step_ms and peak_gb per shape so the S2 config can be chosen on data.
"""

from __future__ import annotations

import json
import os
import sys
import time
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

MODEL = "CohereLabs/tiny-aya-global"
RESULTS: list[dict[str, object]] = []


def clean() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def probe(
    label: str,
    *,
    batch: int,
    seq: int,
    checkpoint: bool,
    test_merge: bool = False,
    autocast_dtype: torch.dtype | None = None,
    lora_dtype: torch.dtype | None = None,
    use_scaler: bool = False,
) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL)
    compute = lora_dtype or torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=compute,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpoint)
    if lora_dtype is not None:
        # prepare_model_for_kbit_training casts adapters to fp32; force them
        # to the compute dtype so activations and GEMMs stay in one dtype
        # (mixed bf16 x fp16 promotes to fp32 and loses the fast path).
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(lora_dtype)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    model.train()

    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_scaler or autocast_dtype == torch.float16)
    )

    input_ids = torch.randint(
        0, model.get_input_embeddings().num_embeddings, (batch, seq), device="cuda"
    )
    attn = torch.ones_like(input_ids)
    labels = input_ids.clone()
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-4
    )

    def step() -> float:
        t0 = time.perf_counter()
        if autocast_dtype is not None:
            with torch.autocast("cuda", dtype=autocast_dtype):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        else:
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        if scaler.is_enabled():
            scaler.scale(out.loss).backward()
            scaler.unscale_(opt)
            opt.step()
            scaler.update()
        else:
            out.loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    try:
        step()  # warmup
        step()
        times = [step() for _ in range(3)]
        avg = sum(times) / len(times)
        peak = torch.cuda.max_memory_allocated() / 2**30
        entry = {
            "shape": label,
            "step_s": round(avg, 2),
            "peak_gb": round(peak, 2),
            "batch": batch,
            "seq": seq,
            "checkpointing": checkpoint,
            "autocast": str(autocast_dtype) if autocast_dtype else "none",
        }
        if test_merge:
            try:
                merged = model.merge_and_unload()
                entry["merge"] = "ok" if merged is not None else "returned-None"
                entry["merge_dtype"] = str(
                    next(merged.parameters()).dtype
                    if merged is not None
                    else "n/a"
                )
                del merged
            except Exception as exc:  # noqa: BLE001
                entry["merge"] = f"FAILED {type(exc).__name__}: {exc}"
        RESULTS.append(entry)
        print(f"{label}: step {avg:.2f}s  peak {peak:.2f} GB  merge={entry.get('merge', 'n/a')}")
    except Exception as exc:  # noqa: BLE001
        RESULTS.append({"shape": label, "error": f"{type(exc).__name__}: {exc}"})
        print(f"{label}: FAILED {type(exc).__name__}: {exc}")
    finally:
        del model, opt, input_ids, attn, labels
        clean()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        default="d,e",
        help="comma list from: a,b2,d,e (default: the autocast comparisons)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no cuda")
        return 1
    print(torch.cuda.get_device_name(0))
    if "a" in args.shapes.split(","):
        probe("nf4_ckpt_b2_s512", batch=2, seq=512, checkpoint=True, test_merge=True)
    if "b2" in args.shapes.split(","):
        probe("nf4_nockpt_b1_s512", batch=1, seq=512, checkpoint=False)
    if "b1" in args.shapes.split(","):
        probe("nf4_ckpt_b1_s512", batch=1, seq=512, checkpoint=True)
    if "d" in args.shapes.split(","):
        probe(
            "nf4_ckpt_b1_s512_autocast_bf16",
            batch=1,
            seq=512,
            checkpoint=True,
            autocast_dtype=torch.bfloat16,
        )
    if "e" in args.shapes.split(","):
        probe(
            "nf4_ckpt_b1_s512_autocast_fp16",
            batch=1,
            seq=512,
            checkpoint=True,
            autocast_dtype=torch.float16,
        )
    if "i" in args.shapes.split(","):
        # Pure-fp16 pipeline, no autocast: bnb dequant/compute fp16 + fp16
        # adapters + GradScaler. Classic pre-bf16 training recipe; the fast
        # fp16 GEMM path (13.07 TFLOPS) without the autocast pathology.
        probe(
            "nf4_fp16_lora_fp16_scaler_b1_s512",
            batch=1,
            seq=512,
            checkpoint=True,
            lora_dtype=torch.float16,
            use_scaler=True,
        )
    if "k" in args.shapes.split(","):
        probe(
            "nf4_fp16_lora_fp16_noscaler_b1_s512",
            batch=1,
            seq=512,
            checkpoint=True,
            lora_dtype=torch.float16,
            use_scaler=False,
        )
    out = REPO_ROOT / "results" / "quant_lora_probe.json"
    out.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"full results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
