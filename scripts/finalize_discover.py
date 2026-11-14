"""Finalize a split S4 discovery run (safety + domain executed separately).

The two cloud notebooks (`kaggle_s4_safety.ipynb`, `kaggle_s4_domain.ipynb`)
run each circuit through the same `run_circuitkit_discovery` path as
`run_discovery`, writing into the canonical directory
`artifacts/discover/discover-<cfg_hash>/{safety,domain}`. This script performs
the steps `run_discovery` would have done after both circuits:

    overlap.json + overlap_<fraction>.json reports, summary.json and a manifest
    whose notes disclose the split execution (honest lineage).

Run AFTER `python scripts/import_cloud_artifacts.py` once BOTH circuits are
imported:

    python scripts/finalize_discover.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("HF_HOME", r"D:\cg_hf")
os.environ.setdefault("HF_HUB_CACHE", r"D:\cg_hf\hub")

from circuitgraft.config import (  # noqa: E402
    BaseConfig,
    DiscoverConfig,
    config_hash,
    load_config,
)
from circuitgraft.discover.circuitkit_runner import (  # noqa: E402
    _input_hashes,
    _model_ref,
    _repo_path,
)
from circuitgraft.discover.overlap import (  # noqa: E402
    load_scores,
    read_score_payload,
    write_overlap_report,
)
from circuitgraft.graft.pipeline import load_layout  # noqa: E402
from circuitgraft.registry import (  # noqa: E402
    RunTimer,
    dump_json,
    load_json,
    new_manifest,
    sha256_file,
)


def main() -> int:
    cfg = load_config(Path("configs/discover.yaml"), DiscoverConfig)
    base = load_config(Path("configs/base.yaml"), BaseConfig)
    cfg_hash = config_hash(cfg)
    run_id = f"discover-{cfg_hash}"
    out_dir = _repo_path(cfg.output_dir) / run_id

    safety_scores_path = out_dir / "safety" / "scores.json"
    domain_scores_path = out_dir / "domain" / "scores.json"
    missing = [
        str(p)
        for p in (
            safety_scores_path,
            out_dir / "safety" / "circuit_info.json",
            domain_scores_path,
            out_dir / "domain" / "circuit_info.json",
        )
        if not p.exists()
    ]
    if missing:
        print("S4 split run is incomplete; missing files:")
        for path in missing:
            print(f"  {path}")
        print("Import the corresponding cloud artifact first (import_cloud_artifacts.py).")
        return 1

    layout = load_layout(base.models.base)
    safety_scores = load_scores(safety_scores_path, source="circuitkit", layout=layout)
    domain_scores = load_scores(domain_scores_path, source="circuitkit", layout=layout)

    overlap_paths: dict[str, str] = {}
    for fraction in cfg.overlap_fractions:
        path = out_dir / f"overlap_{fraction:g}.json"
        write_overlap_report(
            path,
            safety_scores,
            domain_scores,
            k_s=fraction,
            k_d=fraction,
            layout=layout,
            method=f"topk@{fraction:g}",
        )
        overlap_paths[f"{fraction:g}"] = str(path)

    overlap_path = out_dir / "overlap.json"
    write_overlap_report(
        overlap_path,
        safety_scores,
        domain_scores,
        k_s=cfg.safety.sparsity,
        k_d=cfg.domain.sparsity,
        layout=layout,
        method="discovery_sparsity",
    )

    summary = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "safety": _circuit_summary("safety", out_dir, cfg, base),
        "domain": _circuit_summary("domain", out_dir, cfg, base),
        "overlap": str(overlap_path),
        "overlap_by_fraction": overlap_paths,
        "cross_check": None,
        "cross_check_jaccard": None,
        "split_execution": (
            "safety + domain circuits ran as two separate cloud sessions on "
            "identical code/config (ig_steps 2, cross_check off - see "
            "docs/PREREGISTRATION.md §10); overlap and manifest assembled locally"
        ),
    }
    summary_path = dump_json(out_dir / "summary.json", summary)

    manifest = new_manifest("discover", run_id, cfg_hash, seed=cfg.seed)
    manifest.inputs = _input_hashes(cfg, base)
    manifest.notes = (
        "SPLIT EXECUTION: safety circuit on Kaggle T4 (aligned model), domain "
        "circuit on Kaggle T4 (drifted model), overlap+manifest assembled "
        "locally; algorithm=eap-ig; ig_steps=2; cross_check=false; "
        "see docs/PREREGISTRATION.md §10"
    )
    outputs = {
        "safety/scores.json": sha256_file(safety_scores_path),
        "safety/circuit_info.json": sha256_file(out_dir / "safety" / "circuit_info.json"),
        "domain/scores.json": sha256_file(domain_scores_path),
        "domain/circuit_info.json": sha256_file(out_dir / "domain" / "circuit_info.json"),
        "overlap.json": sha256_file(overlap_path),
        "summary.json": sha256_file(summary_path),
    }
    for fraction, path in overlap_paths.items():
        outputs[f"overlap_{fraction}.json"] = sha256_file(path)
    for kind in ("safety", "domain"):
        faithful = out_dir / kind / "faithfulness.json"
        if faithful.exists():
            outputs[f"{kind}/faithfulness.json"] = sha256_file(faithful)
    manifest.outputs = outputs

    with RunTimer(manifest, out_dir):
        pass  # stamps finished_at/wall_clock and rewrites manifest.json

    print(f"S4 split run finalized: {out_dir}")
    print(f"  overlap.json: {overlap_path}")
    print(f"  summary.json: {summary_path}")
    print("  manifest.json: written with split-execution notes")
    return 0


def _circuit_summary(kind: str, out_dir: Path, cfg: DiscoverConfig, base: BaseConfig) -> dict:
    """Mirror the per-circuit payload run_discovery stores in summary.json."""
    circuit_dir = out_dir / kind
    info = load_json(circuit_dir / "circuit_info.json") if (circuit_dir / "circuit_info.json").exists() else {}
    selected = load_json(circuit_dir / "selected.json") if (circuit_dir / "selected.json").exists() else {}
    scores_path = circuit_dir / "scores.json"
    model_ref = _model_ref(base, cfg.safety.model if kind == "safety" else cfg.domain.model)
    sub_cfg = cfg.safety if kind == "safety" else cfg.domain
    payload: dict[str, Any] = {
        "kind": kind,
        "model": model_ref,
        "scores_path": str(scores_path),
        "num_selected_nodes": len(selected.get("nodes", [])),
        "executed_on": (
            "Kaggle T4 (notebooks/kaggle_s4_safety.ipynb)"
            if kind == "safety"
            else "Kaggle T4 (notebooks/kaggle_s4_domain.ipynb)"
        ),
    }
    for key in ("source", "task", "algorithm", "level"):
        if key in info:
            payload[key] = info[key]
    if scores_path.exists():
        payload["num_components"] = len(read_score_payload(scores_path))
    payload["ig_steps"] = sub_cfg.ig_steps
    faithful = circuit_dir / "faithfulness.json"
    payload["faithfulness_path"] = str(faithful) if faithful.exists() else None
    return payload


if __name__ == "__main__":
    sys.exit(main())
