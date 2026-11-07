"""Unit tests for S10 publish (card generation, dry run)."""

from __future__ import annotations

from pathlib import Path

from circuitgraft.config import BaseConfig
from circuitgraft.publish import (
    generate_dataset_card,
    generate_model_card,
    run_publish,
)


def test_generate_model_card() -> None:
    card = generate_model_card("dubeydevansh/circuitgraft-tinyaya-med-cg-project", method="SVD-Project")
    assert "dubeydevansh/circuitgraft-tinyaya-med-cg-project" in card
    assert "CohereLabs/tiny-aya-base" in card
    assert "SVD-Project" in card
    assert "cc-by-nc-4.0" in card


def test_generate_drifted_model_card() -> None:
    card = generate_model_card("dubeydevansh/circuitgraft-tinyaya-med-drifted", drifted=True)
    assert "RESEARCH-ONLY CHECKPOINT — NOT FOR DEPLOYMENT" in card
    assert "SafeTune licence" in card


def test_generate_dataset_card() -> None:
    card = generate_dataset_card(
        "dubeydevansh/circuitgraft-safety-pairs",
        description="Test pairs",
        content_warning=True,
    )
    assert "CONTENT ADVISORY" in card
    assert "Test pairs" in card


def test_run_publish_dry_run(tmp_path: Path) -> None:
    base = BaseConfig(hf_org="dubeydevansh")
    out_dir = tmp_path / "artifacts" / "publish"

    summary = run_publish(base, dry_run=True, output_dir=out_dir)
    assert summary["hf_org"] == "dubeydevansh"
    assert summary["dry_run"] is True
    assert (out_dir / "summary.json").exists()
