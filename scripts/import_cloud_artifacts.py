"""Pull cloud-run artifacts (drift checkpoint, circuits, audit) from the HF Hub.

Usage (from the repo root):
    python scripts/import_cloud_artifacts.py [--repo dubeydevansh/circuitgraft-phase4-cloud]

The repo stores artifacts/... and results/... at the same relative paths as the
local tree, so a plain snapshot_download into the repo root reproduces the
layout exactly and configs/eval.yaml keep working unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("HF_HOME", r"D:\cg_hf")
os.environ.setdefault("HF_HUB_CACHE", r"D:\cg_hf\hub")

from huggingface_hub import snapshot_download  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="dubeydevansh/circuitgraft-phase4-cloud",
        help="Private dataset repo holding the cloud run artifacts",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set (.env or environment)")
        return 1

    print(f"Downloading {args.repo} into {REPO_ROOT} ...")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(REPO_ROOT),
        token=token,
    )

    # Report what arrived.
    checks = {
        "drift manifest": Path("artifacts/drift/scenarioA_seed0/manifest.json"),
        "merged model": None,
        "safety circuit": None,
        "domain circuit": None,
        "audit results": None,
    }
    drift_dir = Path("artifacts/drift/scenarioA_seed0")
    if drift_dir.exists():
        st = [p for p in drift_dir.rglob("*.safetensors") if "merged" in p.name]
        checks["merged model"] = st[0] if st else "MISSING"
    discover_dir = Path("artifacts/discover")
    if discover_dir.exists():
        saf = sorted(discover_dir.glob("*/safety/circuit_info.json"))
        dom = sorted(discover_dir.glob("*/domain/circuit_info.json"))
        checks["safety circuit"] = saf[-1] if saf else "MISSING"
        checks["domain circuit"] = dom[-1] if dom else "MISSING"
    results = sorted(Path("results").glob("*.json")) if Path("results").exists() else []
    checks["audit results"] = f"{len(results)} result jsons"

    print()
    for name, path in checks.items():
        state = "MISSING" if path in (None, "MISSING") else "ok"
        print(f"  {name:16s} {state}  {path if state == 'ok' else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
