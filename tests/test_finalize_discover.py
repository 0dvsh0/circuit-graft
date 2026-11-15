"""Test for scripts/finalize_discover.py."""

import json
from pathlib import Path

from circuitgraft.config import BaseConfig, DiscoverConfig, config_hash, load_config
from scripts import finalize_discover


def test_finalize_discover_succeeds_with_canonical_scores(tmp_path: Path, monkeypatch) -> None:
    # Use tmp_path as repo root
    cfg = load_config(Path("configs/discover.yaml"), DiscoverConfig)
    base = load_config(Path("configs/base.yaml"), BaseConfig)
    cfg_hash = config_hash(cfg)
    run_id = f"discover-{cfg_hash}"
    out_dir = Path("artifacts/discover") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    safety_dir = out_dir / "safety"
    domain_dir = out_dir / "domain"
    safety_dir.mkdir(parents=True, exist_ok=True)
    domain_dir.mkdir(parents=True, exist_ok=True)

    scores_payload = {
        "source": "circuitkit",
        "components": {"L0.H0": 1.0, "L0.H1": 2.0, "L0.MLP": 3.0},
    }
    info_payload = {"source": "circuitkit", "task": "test", "algorithm": "eap-ig", "level": "node"}

    (safety_dir / "scores.json").write_text(json.dumps(scores_payload), encoding="utf-8")
    (safety_dir / "circuit_info.json").write_text(json.dumps(info_payload), encoding="utf-8")
    (domain_dir / "scores.json").write_text(json.dumps(scores_payload), encoding="utf-8")
    (domain_dir / "circuit_info.json").write_text(json.dumps(info_payload), encoding="utf-8")

    try:
        ret = finalize_discover.main()
        assert ret == 0
        assert (out_dir / "overlap.json").exists()
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "manifest.json").exists()
    finally:
        # Clean up created artifacts
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
