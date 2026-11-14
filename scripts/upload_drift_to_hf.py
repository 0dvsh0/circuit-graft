"""Upload the local S2 drifted checkpoint to the private cloud HF repo.

Run this on the LOCAL machine right after S2 finishes (the merged checkpoint
is materialized from the adapter on CPU, ~20 min after the last training step):

    python scripts/upload_drift_to_hf.py

It pushes `artifacts/drift/scenarioA_seed0` (merged model + manifest + summary
+ adapter; trainer/ optimizer states skipped by default) to
`dubeydevansh/circuitgraft-phase4-cloud` under the same relative path, so the
Kaggle S4-domain notebook and configs/base.yaml (models.drifted) resolve it
unchanged.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="artifacts/drift/scenarioA_seed0",
        help="Drift run directory to upload (relative to the repo root)",
    )
    parser.add_argument(
        "--repo",
        default="dubeydevansh/circuitgraft-phase4-cloud",
        help="Private dataset repo receiving the artifacts",
    )
    parser.add_argument(
        "--include-trainer",
        action="store_true",
        help="Also upload trainer/ optimizer states (large, not needed downstream)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set (.env or environment)")
        return 1

    run_dir = REPO_ROOT / args.run_dir
    merged = run_dir / "merged"
    if not (merged / "config.json").exists():
        print(f"ERROR: merged checkpoint not ready at {merged}")
        print("Wait for S2 to finish the CPU merge, then re-run this script.")
        return 1

    patterns = ["merged/**", "manifest.json", "drift_summary.json", "adapter/**"]
    if args.include_trainer:
        patterns.append("trainer/**")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    print(f"Uploading {args.run_dir} (patterns: {patterns}) -> {args.repo} ...")
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=args.repo,
        repo_type="dataset",
        path_in_repo=args.run_dir.replace("\\", "/"),
        allow_patterns=patterns,
        commit_message="local S2 complete: drifted merged checkpoint + manifest",
    )
    print("Done. Start notebooks/kaggle_s4_domain.ipynb as a Save & Run All run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
