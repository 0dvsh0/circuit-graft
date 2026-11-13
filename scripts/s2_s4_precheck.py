#!/usr/bin/env python3
"""Sanity checks before committing the GPU to the ~10 h S2 drift run.

1. fp16 loss finiteness: three real MedMCQA steps through the planned S2
   recipe (NF4 fp16-dequant base, fp16 adapters, no autocast, fp32 CE).
2. S4 engine feasibility: load the aligned model through circuitkit and run
   one EAP-IG step at the preregistered seq 256 / batch 1, measuring VRAM
   peak and wall time (the S4 budget was estimated on healthy bf16 numbers).
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

OUT: dict[str, object] = {}


def _flush() -> None:
    out = REPO_ROOT / "results" / "s2_s4_precheck.json"
    out.write_text(json.dumps(OUT, indent=2), encoding="utf-8")


def fp16_loss_check() -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        "CohereLabs/tiny-aya-global",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    # Mirror finetune.py: the fp16 cast must happen AFTER get_peft_model.
    # prepare_model_for_kbit_training freezes every parameter, so a pre-wrap
    # cast would be a no-op and peft instantiates adapters in fp32.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float16)
    adapter_dtype = str(next(p.dtype for p in model.parameters() if p.requires_grad))
    model.train()
    tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")

    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "artifacts/data/medmcqa_sft/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[:3]
    ]
    texts = []
    for row in rows:
        if "messages" in row:
            texts.append(tok.apply_chat_template(row["messages"], tokenize=False))
        else:
            texts.append(json.dumps(row))
    enc = tok(texts[:1], return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to("cuda")
    attn = enc["attention_mask"].to("cuda")
    labels = input_ids.clone()
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=2e-4
    )
    losses = []
    grad_norms = []
    step_times = []
    for _ in range(3):
        step_start = time.perf_counter()
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        losses.append(float(out.loss.detach()))
        out.loss.backward()
        # Measure before opt.zero_grad(set_to_none=True) clears the grads.
        grad_norms.append(
            sum(
                float(p.grad.abs().sum())
                for p in model.parameters()
                if p.grad is not None
            )
        )
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_times.append(round(time.perf_counter() - step_start, 2))
    OUT["fp16_loss"] = {
        "losses": [round(v, 4) for v in losses],
        "finite": all(v == v and abs(v) < 1e6 for v in losses),
        "grad_abs_sum_per_step": [round(v, 2) for v in grad_norms],
        "grad_finite": all(v == v and abs(v) < 1e12 for v in grad_norms),
        "adapter_dtype": adapter_dtype,
        "step_times_s": step_times,
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    print("fp16_loss:", json.dumps(OUT["fp16_loss"]))
    _flush()
    del model, opt, enc, input_ids, attn, labels
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def s4_engine_probe() -> None:
    """One EAP-IG step through the repo's own discovery stack on the aligned model."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from circuitgraft.discover.circuitkit_runner import run_circuitkit_discovery
    from circuitgraft.graft.pipeline import load_layout

    layout = load_layout("CohereLabs/tiny-aya-base")
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        result = run_circuitkit_discovery(
            model_ref="CohereLabs/tiny-aya-global",
            pairs_path=str(REPO_ROOT / "artifacts/data/pairs/safety.jsonl"),
            output_dir=str(REPO_ROOT / "artifacts" / "probe_s4"),
            layout=layout,
            kind="safety",
            algorithm="eap-ig",
            level="node",
            n_examples=2,
            ig_steps=2,
            sparsity=0.3,
            max_seq_len=256,
            batch_size=1,
            device="cuda",
            dtype="bfloat16",
            seed=0,
        )
        status = result.get("status") if isinstance(result, dict) else None
    except Exception as exc:  # noqa: BLE001 - record the failure and still flush
        status = f"error: {type(exc).__name__}: {exc}"
    OUT["s4_probe"] = {
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "status": status,
    }
    print("s4_probe:", json.dumps(OUT["s4_probe"], default=str)[:800])


def main() -> int:
    if not torch.cuda.is_available():
        print("FATAL: no cuda")
        return 1
    fp16_loss_check()
    s4_engine_probe()
    _flush()
    print(f"full results: {REPO_ROOT / 'results' / 's2_s4_precheck.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
