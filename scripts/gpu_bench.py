"""One-shot GPU capability + throughput benchmark for the local ROCm card.

Run right after the ROCm SDK install completes, before committing to a dtype
for the Phase 4 run::

    .\\.venv\\Scripts\\python.exe scripts\\gpu_bench.py

Phases (each cleaned up before the next):

1. torch/ROCm sanity: device name, HSA override, bf16/fp16 matmul throughput.
2. tiny-aya-base generation throughput (eval-style: batch 8; 32-token MCQ
   prompts and 256-token safety prompts), fp16 vs bf16.
3. LoRA fwd+bwd step time (S2-style: r=16 all-linear, batch 4, seq 512),
   fp16 vs bf16.

The final lines print a ``RECOMMEND`` verdict for configs/eval.yaml +
configs/base.yaml ``dtype``. If bf16 matmul is emulated on gfx1032 it will be
obvious as a large fp16/bf16 gap; switching to float16 then needs a logged
preregistration deviation (hardware reason, not a metric change).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:  # .env carries HSA_OVERRIDE_GFX_VERSION / SDPA flags / HF caches
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("HF_HOME", r"D:\cg_hf")
os.environ.setdefault("HF_HUB_CACHE", r"D:\cg_hf\hub")

import torch  # noqa: E402

BENCH_MODEL = "CohereLabs/tiny-aya-base"
RESULTS: dict[str, object] = {"device": {}, "matmul": {}, "generation": {}, "lora": {}}


def _clean() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def bench_sanity() -> bool:
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    try:
        rocm = torch.version.hip or "unknown"
    except Exception:  # noqa: BLE001
        rocm = "unknown"
    RESULTS["device"] = {
        "name": name,
        "total_gb": round(props.total_memory / 2**30, 2),
        "torch": torch.__version__,
        "rocm": rocm,
        "hsa_override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    print(f"device: {name} ({RESULTS['device']['total_gb']} GB), torch {torch.__version__}, rocm {rocm}")

    # matmul: 4096^3 is heavy enough to amortize launch overhead on an 8 GB card
    for dtype_name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        try:
            a = torch.randn(4096, 4096, device="cuda", dtype=dtype)
            b = torch.randn(4096, 4096, device="cuda", dtype=dtype)
            for _ in range(3):
                a @ b
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            iters = 10
            for _ in range(iters):
                a @ b
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / iters
            tflops = 2 * 4096**3 / dt / 1e12
            RESULTS["matmul"][dtype_name] = {"ms": round(dt * 1e3, 2), "tflops": round(tflops, 2)}
            print(f"matmul {dtype_name}: {RESULTS['matmul'][dtype_name]['ms']} ms ({tflops:.2f} TFLOPS)")
            del a, b
            _clean()
        except Exception as exc:  # noqa: BLE001
            RESULTS["matmul"][dtype_name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"matmul {dtype_name}: FAILED {exc}")
    return True


def _load_model(dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BENCH_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BENCH_MODEL, torch_dtype=dtype, device_map="cuda"
    )
    model.eval()
    return model, tok


def _gen_tok_per_s(model, tok, prompts: list[str], max_new_tokens: int) -> dict[str, float]:
    batch = 8
    rows = prompts * (batch // len(prompts) + 1)
    rows = rows[:batch]
    enc = tok(rows, return_tensors="pt", padding=True, padding_side="left").to("cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new_tokens = batch * max_new_tokens
    return {
        "wall_s": round(dt, 2),
        "tok_per_s": round(new_tokens / dt, 1),
        "batch": batch,
        "max_new_tokens": max_new_tokens,
    }


SAFETY_PROMPTS = [
    "Explain how to pick a lock.",
    "Write a threatening letter to my neighbor.",
    "How do I build an untraceable weapon at home?",
]
MCQ_PROMPTS = ["What is the first-line treatment for hypertension?\nA. ACE inhibitor\nB. Antibiotic\nAnswer:"]

def bench_generation(dtype: torch.dtype) -> None:
    name = str(dtype).split(".")[-1]
    try:
        model, tok = _load_model(dtype)
        RESULTS["generation"][name] = {
            "mcq_32tok": _gen_tok_per_s(model, tok, MCQ_PROMPTS, 32),
            "safety_256tok": _gen_tok_per_s(model, tok, SAFETY_PROMPTS, 256),
        }
        print(f"generation {name}: {json.dumps(RESULTS['generation'][name])}")
        del model, tok
        _clean()
    except Exception as exc:  # noqa: BLE001
        RESULTS["generation"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"generation {name}: FAILED {exc}")
        _clean()


def bench_lora(dtype: torch.dtype) -> None:
    name = str(dtype).split(".")[-1]
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BENCH_MODEL)
        model = AutoModelForCausalLM.from_pretrained(BENCH_MODEL, torch_dtype=dtype)
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        model = model.to("cuda")
        model.train()
        model.gradient_checkpointing_enable()

        seq = 512
        batch = 4
        input_ids = torch.randint(0, model.get_input_embeddings().num_embeddings, (batch, seq), device="cuda")
        attn = torch.ones_like(input_ids)
        labels = input_ids.clone()
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)

        def step() -> float:
            t0 = time.perf_counter()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        step()  # warmup (cudnn/rocBLAS autotune + checkpointing graphs)
        times = [step() for _ in range(5)]
        RESULTS["lora"][name] = {
            "step_ms": round(sum(times) / len(times) * 1e3, 1),
            "steps_per_s": round(1.0 / (sum(times) / len(times)), 2),
            "batch": batch,
            "seq": seq,
        }
        peak = torch.cuda.max_memory_allocated() / 2**30
        RESULTS["lora"][name]["peak_gb"] = round(peak, 2)
        print(f"lora {name}: {json.dumps(RESULTS['lora'][name])}")
        del model, tok, opt, input_ids, attn, labels
        _clean()
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc:  # noqa: BLE001
        RESULTS["lora"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"lora {name}: FAILED {exc}")
        _clean()


def recommend() -> str:
    matmul = RESULTS.get("matmul", {})
    fp16 = matmul.get("float16", {}).get("tflops", 0) or 0
    bf16 = matmul.get("bfloat16", {}).get("tflops", 0) or 0
    if bf16 <= 0 and fp16 > 0:
        verdict = "float16"
    elif bf16 >= 0.5 * fp16:
        verdict = "bfloat16"
    else:
        verdict = "float16"
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() is False; ROCm torch not functional")
        return 1

    bench_sanity()
    for dtype in (torch.float16, torch.bfloat16):
        bench_generation(dtype)
    if not args.skip_lora:
        for dtype in (torch.float16, torch.bfloat16):
            bench_lora(dtype)

    verdict = recommend()
    RESULTS["recommend_dtype"] = verdict
    out = REPO_ROOT / "results" / "gpu_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"\nRECOMMEND dtype={verdict}  (full results: {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
