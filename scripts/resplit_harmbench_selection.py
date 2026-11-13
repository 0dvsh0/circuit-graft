#!/usr/bin/env python3
"""Carve the HarmBench selection split out of the finished S1 test file.

S1 ran with ``eval.harmbench: 200`` and ``eval.selection_harmbench: 100``.
HarmBench has exactly 200 ``standard`` behaviours, so ``split_eval_sets``
handed the entire seeded shuffle to the test split and the selection file came
out empty (0 rows); ``summary.json.split_notes.harmbench`` recorded
"not enough HarmBench standard behaviours for test+selection".

Fix (prereg deviation 2026-11-13): the test split keeps the preregistered
150-row cap (``configs/eval.yaml`` ``max_rows.harmbench: 150``), and the
remaining 50 rows become the selection split (was 60).

Why this is equivalent to a fresh ``cg data`` run with the corrected config:
``_split_indices(pool, n_test, n_select, seed=0, key="harmbench")`` writes
``test = shuffled[:n_test]`` in shuffle order, and the current file *is* that
shuffle output (n_test == pool size == 200). Therefore a fresh run with
``harmbench: 150, selection_harmbench: 50`` yields

    test      = shuffled[:150]  == current file[:150]
    selection = shuffled[150:]  == current file[150:]

The script verifies this by replaying the HarmBench loader through the same
``_as_rows`` / ``safety_to_eval`` / dedup / standard-subset path and checking
that ``random.Random("0:harmbench").shuffle(pool)`` reproduces the file order
exactly. Lines are then moved verbatim -- no re-serialization.

Also updates ``summary.json`` and ``manifest.json`` (sha256 of the two touched
files plus summary.json) so stage manifests stay truthful.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from circuitgraft.registry import dump_json, sha256_file  # noqa: E402

DATA_DIR = REPO_ROOT / "artifacts" / "data"
EVAL_PATH = DATA_DIR / "eval" / "harmbench.jsonl"
SEL_PATH = DATA_DIR / "selection" / "harmbench.jsonl"
SUMMARY_PATH = DATA_DIR / "summary.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"

TEST = 150
SELECTION = 50


def _read_lines(path: Path) -> list[str]:
    lines = path.read_bytes().decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _verify_pool_order(file_ids: list[str], seed: int) -> str:
    """Replay the S1 loader path; confirm the file order is the seeded shuffle."""
    try:
        from circuitgraft.data.curate import (
            _as_rows,
            _dedup_prompts,
            load_safetune_rows,
            safety_to_eval,
        )
    except Exception as exc:  # noqa: BLE001 - verification is best-effort
        return f"SKIPPED (import failed: {exc})"

    try:
        rows = _as_rows(load_safetune_rows("load_harmbench"))
        conv = [
            dict(safety_to_eval(r, benchmark="harmbench", source_key="harmbench", label="harmful"))
            for r in rows
        ]
        conv = _dedup_prompts(conv)
        pool = [r for r in conv if r.get("metadata", {}).get("subset") == "standard"]
    except Exception as exc:  # noqa: BLE001
        return f"SKIPPED (replay failed: {exc})"

    if len(pool) != len(file_ids):
        return f"SKIPPED (pool {len(pool)} != file {len(file_ids)})"
    shuffled = list(pool)
    random.Random(f"{seed}:harmbench").shuffle(shuffled)
    if [r["id"] for r in shuffled] != file_ids:
        return "SKIPPED (shuffled pool order does not match file order)"
    return "PASSED (file order == seeded shuffle of the HarmBench standard pool)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move the last N HarmBench test rows into the selection split."
    )
    parser.add_argument("--test", type=int, default=TEST)
    parser.add_argument("--selection", type=int, default=SELECTION)
    parser.add_argument("--skip-pool-verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lines = _read_lines(EVAL_PATH)
    rows = [json.loads(line) for line in lines]
    n = len(rows)

    if args.test + args.selection != n:
        print(f"refusing: test+selection ({args.test}+{args.selection}) != pool ({n})")
        return 2
    subsets = {row["metadata"]["subset"] for row in rows}
    if subsets != {"standard"}:
        print(f"refusing: non-standard rows present: {sorted(subsets)}")
        return 2
    ids = [row["id"] for row in rows]
    if len(set(ids)) != n:
        print("refusing: duplicate ids in test file")
        return 2
    for line, row in zip(lines, rows):
        canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if canonical != line:
            print("refusing: line not in canonical _write_jsonl form; aborting")
            return 2

    if not args.skip_pool_verify:
        seed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("seed", 0)
        verdict = _verify_pool_order(ids, seed)
        print(f"pool-order verification: {verdict}")
        if not verdict.startswith("PASSED"):
            print("aborting: equivalence to a fresh config-driven run not proven")
            return 2

    test_lines = lines[: args.test]
    selection_lines = lines[args.test :]

    print(f"pool {n} rows -> test {len(test_lines)}, selection {len(selection_lines)}")
    first_sel = json.loads(selection_lines[0])["id"]
    last_sel = json.loads(selection_lines[-1])["id"]
    print(f"  selection ids: {first_sel} .. {last_sel}")

    if args.dry_run:
        print("dry run; nothing written")
        return 0

    for path, out_lines in ((EVAL_PATH, test_lines), (SEL_PATH, selection_lines)):
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(out_lines) + "\n")
        print(f"wrote {path.relative_to(REPO_ROOT)}: {len(out_lines)} rows")

    # --- summary.json -----------------------------------------------------
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["eval"]["harmbench"]["selection_rows"] = args.selection
    summary["selection"]["harmbench"]["rows"] = args.selection
    summary["split_notes"]["harmbench"] = {
        "test": args.test,
        "selection": args.selection,
        "note": (
            "resplit 2026-11-13: HarmBench standard pool is exactly 200 rows; the "
            "test split keeps the preregistered 150-row cap and the remaining 50 "
            "rows (the tail of the seeded shuffle) form the selection split "
            "(deviation: selection 60 -> 50). Byte-identical to a fresh `cg data` "
            "run with the corrected config; verified against the loader pool."
        ),
    }
    dump_json(SUMMARY_PATH, summary)
    print(f"wrote {SUMMARY_PATH.relative_to(REPO_ROOT)}")

    # --- manifest.json ----------------------------------------------------
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    wanted = {
        "artifacts/data/eval/harmbench.jsonl": EVAL_PATH,
        "artifacts/data/selection/harmbench.jsonl": SEL_PATH,
        "artifacts/data/summary.json": SUMMARY_PATH,
    }
    posix_keys = {key.replace("\\", "/"): key for key in outputs}
    for rel, path in wanted.items():
        key = posix_keys.get(rel)
        if key is None:
            print(f"refusing: manifest has no outputs entry for {rel}")
            return 2
        old = outputs[key]
        outputs[key] = sha256_file(path)
        print(f"hash {rel}: {old[:12]}.. -> {outputs[key][:12]}..")
    manifest["notes"] = (
        manifest.get("notes", "")
        + f"; harmbench resplit 2026-11-13: test {args.test} / selection "
        f"{args.selection} (prereg deviation, see docs/PREREGISTRATION.md section 10)"
    )
    dump_json(MANIFEST_PATH, manifest)
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")

    # --- final sanity ------------------------------------------------------
    test_ids = {json.loads(line)["id"] for line in _read_lines(EVAL_PATH)}
    sel_ids = {json.loads(line)["id"] for line in _read_lines(SEL_PATH)}
    assert not (test_ids & sel_ids), "test/selection overlap after resplit"
    assert test_ids | sel_ids == set(ids), "row set changed by resplit"
    print("sanity: disjoint, union preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
