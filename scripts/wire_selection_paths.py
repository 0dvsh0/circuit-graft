"""Wire selection-split sweep checkpoints into configs/eval_selection.yaml.

Sweep runs are hashed directories (``artifacts/baselines/resta-<hash>``); the
eval resolver's glob fallback would silently pick the newest match, so the
selection eval requires explicit ``model_paths`` entries for every sweep point.

This script scans the stage manifests, maps every run to its preregistered
logical name, verifies the checkpoint is complete (manifest status + weights),
and rewrites only the marker-delimited ``model_paths`` block in
``configs/eval_selection.yaml``. It never invents a mapping: a sweep point with
no completed run is reported as a missing stage, and the file is left with
whatever subset is verified.

Usage (after S5 ``cg baselines`` and S6 ``cg graft`` completed)::

    python scripts/wire_selection_paths.py
    python scripts/wire_selection_paths.py --check   # exit 1 unless all wired
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from circuitgraft.registry import REPO_ROOT, existing_manifest, load_json

CONFIG_PATH = REPO_ROOT / "configs" / "eval_selection.yaml"
MARKER_BEGIN = "# BEGIN auto-wired by scripts/wire_selection_paths.py"
MARKER_END = "# END auto-wired"

# method -> (swept param key, {value -> logical name})
BASELINE_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "resta": ("alpha", {0.25: "resta_a025", 0.4: "resta_a040"}),
    "wiseft": ("alpha", {0.5: "wiseft_l050", 0.75: "wiseft_l075"}),
    "ctheta": ("strength", {0.5: "ctheta_s050", 1.0: "ctheta_s100"}),
    "safelora": ("threshold", {0.5: "safelora_t050", 0.4: "safelora_t040"}),
}

GRAFT_NAMES = (
    "cg_project",
    "cg_project_k02_a025",
    "cg_project_k02_a100",
    "cg_project_k10_a025",
    "cg_project_k10_a100",
    "cg_uniform",
    "cg_distributed",
)


def _close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6


def _weights_dir(run_dir: Path) -> Path | None:
    merged = run_dir / "merged"
    if _has_weights(merged):
        return merged
    return None


def _has_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        (path / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _complete(run_dir: Path) -> bool:
    manifest = existing_manifest(run_dir)
    return manifest is None or manifest.status == "complete"


def wire(*, check_only: bool = False) -> int:
    resolved: dict[str, str] = {}
    problems: list[str] = []

    # --- baselines: artifacts/baselines/<method>-<hash>/baseline_stats.json ---
    baselines_root = REPO_ROOT / "artifacts" / "baselines"
    if baselines_root.is_dir():
        for stats_file in sorted(baselines_root.glob("*-*/baseline_stats.json")):
            run_dir = stats_file.parent
            stats = load_json(stats_file)
            method = str(stats.get("method", ""))
            params = stats.get("params") if isinstance(stats.get("params"), dict) else {}
            if method not in BASELINE_MAP:
                continue
            key, value_map = BASELINE_MAP[method]
            value = params.get(key)
            logical = next(
                (name for target, name in value_map.items()
                 if isinstance(value, (int, float)) and _close(float(value), float(target))),
                None,
            )
            if logical is None:
                problems.append(
                    f"{run_dir.name}: {method} {key}={value!r} matches no sweep point"
                )
                continue
            weights = _weights_dir(run_dir)
            if weights is None or not _complete(run_dir):
                problems.append(f"{run_dir.name}: incomplete or missing merged weights")
                continue
            if logical in resolved:
                problems.append(f"{logical}: more than one completed run ({resolved[logical]})")
                continue
            resolved[logical] = str(weights.resolve())
    else:
        problems.append("artifacts/baselines is missing: run `cg baselines` (S5) first")

    # --- graft: artifacts/graft/<logical>-<hash>/graft_stats.json ---
    graft_root = REPO_ROOT / "artifacts" / "graft"
    if graft_root.is_dir():
        for run_dir in sorted(graft_root.iterdir()):
            if not run_dir.is_dir() or "-" not in run_dir.name:
                continue
            logical = run_dir.name.split("-", 1)[0]
            if logical not in GRAFT_NAMES:
                continue
            if not (run_dir / "graft_stats.json").exists():
                problems.append(f"{run_dir.name}: graft_stats.json missing")
                continue
            weights = _weights_dir(run_dir)
            if weights is None or not _complete(run_dir):
                problems.append(f"{run_dir.name}: incomplete or missing merged weights")
                continue
            if logical in resolved:
                problems.append(f"{logical}: more than one completed run ({resolved[logical]})")
                continue
            resolved[logical] = str(weights.resolve())
    else:
        problems.append("artifacts/graft is missing: run `cg graft` (S6) first")

    # --- coverage over the preregistered sweep points ---
    expected = {name for _, mapping in BASELINE_MAP.values() for name in mapping.values()}
    expected |= set(GRAFT_NAMES)
    missing = sorted(expected - set(resolved))
    for name in missing:
        problems.append(f"{name}: no completed sweep run found (run the stage, then re-wire)")

    if problems:
        print("Problems detected:")
        for problem in problems:
            print(f"  - {problem}")

    if check_only and (missing or any("more than one" in p or "matches no" in p for p in problems)):
        print("check failed: mapping incomplete or ambiguous")
        return 1

    if not check_only:
        _write_block(resolved)
        print(f"\nWrote {len(resolved)} model_paths entries to {CONFIG_PATH.relative_to(REPO_ROOT)}")
    else:
        print(f"\ncheck: {len(resolved)}/{len(expected)} sweep points wired")

    for name in sorted(resolved):
        print(f"  {name} -> {resolved[name]}")
    return 0


def _write_block(resolved: dict[str, str]) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(MARKER_BEGIN) + r".*?" + re.escape(MARKER_END), re.DOTALL
    )
    lines = ["model_paths: {}"] if not resolved else ["model_paths:"]
    lines.extend(f"  {name}: \"{path}\"" for name, path in sorted(resolved.items()))
    block = "\n".join([MARKER_BEGIN, *lines, f"{MARKER_END} -- do not edit between markers"])
    if pattern.search(text):
        text = pattern.sub(lambda _m: block, text, count=1)
    else:
        raise SystemExit(f"markers not found in {CONFIG_PATH}; the template changed?")
    CONFIG_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only; exit 1 if incomplete")
    args = parser.parse_args()
    return wire(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
