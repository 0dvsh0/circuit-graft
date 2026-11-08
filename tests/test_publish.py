"""Unit tests for S10 publish (card generation, artifact resolution, dry run)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from circuitgraft.config import BaseConfig
from circuitgraft.publish import (
    generate_dataset_card,
    generate_model_card,
    run_publish,
)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_file(path: Path, text: str = "data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _synthetic_artifact_tree(root: Path) -> None:
    # S2 drift: merged checkpoint plus the adapter produced before merging.
    drift = root / "artifacts" / "drift" / "drift-test"
    _write_file(drift / "merged" / "model.safetensors")
    _write_file(drift / "adapter" / "adapter_config.json", "{}")

    # S6 graft: the primary repaired checkpoint.
    graft = root / "artifacts" / "graft" / "cg_project-test"
    _write_file(graft / "model.safetensors")
    _write_json(
        graft / "graft_stats.json",
        {"edit_footprint": 0.01, "changed_parameters": 1000},
    )

    # S5 baseline: one method is enough to prove resolution works.
    baseline = root / "artifacts" / "baselines" / "resta-test"
    _write_file(baseline / "merged" / "model.safetensors")

    # S8 abliteration attack checkpoint.
    attack = root / "artifacts" / "attacks" / "abliterate" / "run1"
    _write_file(attack / "merged" / "model.safetensors")

    # S1 data tree with both safety and domain files in one pairs folder.
    data = root / "artifacts" / "data"
    _write_file(data / "medmcqa_sft" / "train.jsonl")
    _write_file(data / "medmcqa_sft" / "validation.jsonl")
    _write_file(data / "medmcqa_sft" / "section.json", "{}")
    _write_file(data / "medmcqa_sft" / "curatorkit" / "manifest.json", "{}")
    _write_file(data / "pairs" / "safety.jsonl")
    _write_file(data / "pairs" / "safety.rejected.jsonl")
    _write_file(data / "pairs" / "review_safety.csv")
    _write_file(data / "pairs" / "domain.jsonl")
    _write_file(data / "pairs" / "domain.rejected.jsonl")
    _write_file(data / "pairs" / "review_domain.csv")
    _write_file(data / "pairs" / "section.json", "{}")
    _write_file(data / "pairs" / "validation.json", "{}")
    _write_file(data / "pairs" / "eval_overlap.json", "{}")
    _write_file(data / "eval" / "domain_benign.jsonl")
    _write_file(data / "eval" / "domain_harmful.jsonl")

    # S4 discovery artifacts.
    discover = root / "artifacts" / "discover" / "discover-test"
    _write_json(discover / "summary.json", {"run_id": "discover-test"})
    _write_json(discover / "overlap.json", {"method": "discovery_sparsity"})
    _write_json(discover / "safety" / "scores.json", {"L1.H0": 0.9})
    _write_json(discover / "domain" / "scores.json", {"L2.MLP": 0.8})

    # S9 logs, including a generations folder that must not be published.
    _write_json(
        root / "results" / "s9" / "report-test" / "summary.json",
        {"run_id": "report-test"},
    )
    _write_file(
        root / "results" / "generations" / "cg_project" / "harmbench.jsonl",
        "{}",
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

    summary = run_publish(base, dry_run=True, output_dir=out_dir, repo_root=tmp_path)
    assert summary["hf_org"] == "dubeydevansh"
    assert summary["dry_run"] is True
    assert summary["models_published"] == []
    assert summary["datasets_published"] == []
    assert (out_dir / "summary.json").exists()


def test_run_publish_resolves_and_filters_real_artifact_tree(tmp_path: Path) -> None:
    _synthetic_artifact_tree(tmp_path)
    base = BaseConfig(hf_org="dubeydevansh")
    out_dir = tmp_path / "artifacts" / "publish"

    summary = run_publish(base, dry_run=True, output_dir=out_dir, repo_root=tmp_path)

    models = {item["repo_id"]: item for item in summary["models_published"]}
    assert set(models) == {
        "dubeydevansh/circuitgraft-tinyaya-med-drifted",
        "dubeydevansh/circuitgraft-tinyaya-med-lora",
        "dubeydevansh/circuitgraft-tinyaya-med-cg-project",
        "dubeydevansh/circuitgraft-tinyaya-med-resta",
        "dubeydevansh/circuitgraft-tinyaya-med-abliterated-run1",
    }
    assert models["dubeydevansh/circuitgraft-tinyaya-med-cg-project"]["source"] == (
        "artifacts/graft/cg_project-test"
    )
    assert models["dubeydevansh/circuitgraft-tinyaya-med-resta"]["source"] == (
        "artifacts/baselines/resta-test/merged"
    )

    datasets = {item["repo_id"]: item for item in summary["datasets_published"]}
    assert set(datasets) == {
        "dubeydevansh/circuitgraft-med-sft-curated",
        "dubeydevansh/circuitgraft-safety-pairs",
        "dubeydevansh/circuitgraft-domain-pairs",
        "dubeydevansh/circuitgraft-med-sensitive-eval",
        "dubeydevansh/circuitgraft-circuits",
        "dubeydevansh/circuitgraft-eval-logs",
    }

    staging = out_dir / "staging"
    safety_files = {
        path.relative_to(staging / "circuitgraft-safety-pairs").as_posix()
        for path in (staging / "circuitgraft-safety-pairs").rglob("*")
        if path.is_file()
    }
    assert "README.md" in safety_files
    assert "safety.jsonl" in safety_files
    assert "safety.rejected.jsonl" in safety_files
    assert "review_safety.csv" in safety_files
    assert "domain.jsonl" not in safety_files
    assert "review_domain.csv" not in safety_files

    domain_files = {
        path.relative_to(staging / "circuitgraft-domain-pairs").as_posix()
        for path in (staging / "circuitgraft-domain-pairs").rglob("*")
        if path.is_file()
    }
    assert "domain.jsonl" in domain_files
    assert "section.json" in domain_files
    assert "review_domain.csv" in domain_files
    assert "safety.jsonl" not in domain_files
    assert "review_safety.csv" not in domain_files

    eval_log_files = {
        path.relative_to(staging / "circuitgraft-eval-logs").as_posix()
        for path in (staging / "circuitgraft-eval-logs").rglob("*")
        if path.is_file()
    }
    assert "s9/report-test/summary.json" in eval_log_files
    assert not any("generations" in path for path in eval_log_files)

    skipped_repos = {item["repo"] for item in summary["skipped"]}
    assert "dubeydevansh/circuitgraft-tinyaya-med-cg-uniform" in skipped_repos
    assert "dubeydevansh/circuitgraft-tinyaya-med-wiseft" in skipped_repos
    assert "dubeydevansh/circuitgraft-tinyaya-med-lora" not in skipped_repos

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "publish"
    assert manifest["status"] == "partial"
