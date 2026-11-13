"""Copy the S1 selection split next to the test-split eval files.

The S1 data stage writes the hyperparameter-selection split to
``artifacts/data/eval/selection/<benchmark>.jsonl``. The S7 evaluator reads
benchmarks from ``artifacts/data/eval/<benchmark>.jsonl``, and the selection
split is distinguished by the ``selection_`` filename prefix (which
``eval.pipeline._canonical_benchmark`` strips for metric pooling).

Usage (after S1 ``cg data``)::

    python scripts/prepare_selection_data.py
    python scripts/prepare_selection_data.py --check
"""

from __future__ import annotations

import argparse
import shutil
import sys

from circuitgraft.registry import REPO_ROOT, sha256_file

SELECTION_DIR = REPO_ROOT / "artifacts" / "data" / "eval" / "selection"
EVAL_DIR = REPO_ROOT / "artifacts" / "data" / "eval"
FILES = ("harmbench.jsonl", "xstest_safe.jsonl", "medmcqa_validation.jsonl")


def prepare(*, check_only: bool = False) -> int:
    failures: list[str] = []
    for name in FILES:
        src = SELECTION_DIR / name
        dst = EVAL_DIR / f"selection_{name}"
        if not src.is_file():
            failures.append(f"missing source {src} (run `cg data --config configs/data.yaml` first)")
            continue
        if check_only:
            if dst.is_file() and sha256_file(src) == sha256_file(dst):
                print(f"ok  {dst.name} ({src.stat().st_size} bytes)")
            else:
                failures.append(f"{dst.name} missing or stale (re-run without --check)")
            continue
        shutil.copyfile(src, dst)
        print(f"copied {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)} "
              f"({src.stat().st_size} bytes)")

    if failures:
        print("\nProblems detected:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only; exit 1 if stale")
    args = parser.parse_args()
    return prepare(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
