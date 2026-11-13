"""ROCm torch smoke test: import, device, bf16/fp16 matmul on the real card."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

import torch

print("torch:", torch.__version__, "| hip:", getattr(torch.version, "hip", None))
print("cuda.is_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FAIL: no GPU")
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
print("hsa_override:", os.environ.get("HSA_OVERRIDE_GFX_VERSION"))

for name, dt in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
    a = torch.randn(2048, 2048, device="cuda", dtype=dt)
    b = torch.randn(2048, 2048, device="cuda", dtype=dt)
    c = a @ b
    torch.cuda.synchronize()
    err = (c.float() - (a.float() @ b.float())).abs().max().item()
    print(f"matmul {name}: OK, max_err={err:.4f}")
    del a, b, c
    torch.cuda.empty_cache()

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
w = torch.randn(1024, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
y = (x @ w).sum()
y.backward()
torch.cuda.synchronize()
print("backward pass: OK")
print("GPU SMOKE TEST PASSED")
