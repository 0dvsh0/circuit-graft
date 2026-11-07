"""Publication pipeline for Hugging Face Hub (S10).

Exports all CircuitGraft models, datasets, eval logs and circuits to a single
Hugging Face namespace (<HF_ORG>), generates compliant model and dataset cards
with explicit lineage and licence warnings, and groups items into two public
collections (Models and Datasets) as required by the hackathon submission.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuitgraft.config import BaseConfig
from circuitgraft.registry import (
    REPO_ROOT,
    dump_json,
    new_manifest,
    push_folder_to_hub,
)

logger = logging.getLogger(__name__)


def generate_model_card(
    repo_name: str,
    *,
    base_model: str = "CohereLabs/tiny-aya-base",
    aligned_model: str = "CohereLabs/tiny-aya-global",
    drifted: bool = False,
    method: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Generate a model card markdown with metadata and licensing."""
    tags = ["circuitgraft", "tiny-aya", "lexsi-hackathon", "safety-recovery"]
    if drifted:
        tags.extend(["research-only", "safety-drifted"])

    warning_block = ""
    if drifted:
        warning_block = """
> [!CAUTION]
> **RESEARCH-ONLY CHECKPOINT — NOT FOR DEPLOYMENT**
> Under the SafeTune licence, unrepaired drifted checkpoints may not be deployed in production.
> This checkpoint exhibits significant safety degradation resulting from domain fine-tuning and is
> provided solely for mechanistic interpretability and safety-recovery research.
"""

    card = f"""---
language:
- en
- multilingual
license: cc-by-nc-4.0
tags:
{chr(10).join(f"- {t}" for t in tags)}
base_model: {aligned_model}
---

# {repo_name}

{warning_block}

## Model Lineage
1. **Base Foundation**: `{base_model}`
2. **Safety-Aligned**: `{aligned_model}`
3. **Domain Specialist**: `Cohere Tiny Aya fine-tuned on medical domain (MedMCQA)`
4. **Recovery Method**: `{method or "Domain Drifted (Unrepaired)"}`

## Configuration & Provenance
- Repository: [CircuitGraft](https://github.com/dubeydevansh/circuit-graft)
- Methodology: Circuit-aware safety recovery using CircuitKIT and SafeTune.

## Licences & Terms of Use
- Weights are derived from Cohere Tiny Aya and subject to the **CC-BY-NC-4.0** license and [Cohere Acceptable Use Policy](https://cohere.com/c4ai-acceptable-use-policy).
- Research conducted under the Lexsi Labs Hackathon (Track 2b / Problem 6).
"""
    return card.strip() + "\n"


def generate_dataset_card(
    repo_name: str,
    *,
    description: str,
    content_warning: bool = False,
) -> str:
    """Generate a dataset card markdown with metadata."""
    warning = ""
    if content_warning:
        warning = """
> [!WARNING]
> **CONTENT ADVISORY**: This dataset contains benchmark adversarial prompts and refusal evaluation targets.
> Full harmful completions are withheld or truncated in accordance with AI safety publication guidelines.
"""

    return f"""---
license: cc-by-nc-4.0
tags:
- circuitgraft
- safety-evaluation
- interpretability
---

# {repo_name}

{warning}

## Description
{description}

## Part of CircuitGraft
Created for CircuitGraft: Circuit-aware safety recovery for domain-fine-tuned Cohere Tiny Aya models.
"""


def run_publish(
    base: BaseConfig,
    *,
    dry_run: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Publish models, datasets and collections to Hugging Face Hub."""
    hf_org = base.hf_org
    if not hf_org:
        raise ValueError("base.hf_org is empty; configure HF_ORG in .env or configs/base.yaml")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not dry_run and not token:
        raise ValueError("HF_TOKEN is required to publish to Hugging Face Hub.")

    out_dir = Path(output_dir) if output_dir else REPO_ROOT / "artifacts" / "publish"
    out_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi

    api = HfApi(token=token) if not dry_run else None

    # Track published items
    models_published: list[str] = []
    datasets_published: list[str] = []

    # 1. Define Model Targets
    model_targets = [
        ("circuitgraft-tinyaya-med-drifted", REPO_ROOT / "artifacts" / "drift" / "model", True, "Drifted (Unrepaired)"),
        ("circuitgraft-tinyaya-med-lora", REPO_ROOT / "artifacts" / "drift" / "adapter", True, "Drift LoRA Adapter"),
        ("circuitgraft-tinyaya-med-cg-project", REPO_ROOT / "artifacts" / "graft" / "cg_project" / "model", False, "CircuitGraft SVD-Project (Primary)"),
        ("circuitgraft-tinyaya-med-cg-uniform", REPO_ROOT / "artifacts" / "graft" / "cg_uniform" / "model", False, "CircuitGraft Uniform"),
        ("circuitgraft-tinyaya-med-cg-exclude", REPO_ROOT / "artifacts" / "graft" / "cg_exclude" / "model", False, "CircuitGraft Exclude"),
        ("circuitgraft-tinyaya-med-cg-fisher", REPO_ROOT / "artifacts" / "graft" / "cg_fisher" / "model", False, "CircuitGraft Fisher Gate"),
        ("circuitgraft-tinyaya-med-resta", REPO_ROOT / "artifacts" / "baselines" / "resta" / "model", False, "SafeTune RESTA Baseline"),
        ("circuitgraft-tinyaya-med-ctheta", REPO_ROOT / "artifacts" / "baselines" / "ctheta" / "model", False, "C-ΔΘ (SafeTune weight-arithmetic implementation)"),
        ("circuitgraft-tinyaya-med-wiseft", REPO_ROOT / "artifacts" / "baselines" / "wiseft" / "model", False, "Wise-FT Baseline"),
        ("circuitgraft-tinyaya-med-safelora", REPO_ROOT / "artifacts" / "baselines" / "safelora" / "model", False, "SafeLoRA Baseline"),
    ]

    for repo_suffix, path, is_drifted, method_name in model_targets:
        if not path.exists():
            logger.info("Skipping missing model target: %s (%s)", repo_suffix, path)
            continue

        repo_id = f"{hf_org}/{repo_suffix}"
        card_content = generate_model_card(
            repo_id,
            drifted=is_drifted,
            method=method_name,
        )
        if not dry_run:
            card_path = path / "README.md"
            card_path.write_text(card_content, encoding="utf-8")

        if dry_run:
            logger.info("[Dry Run] Would upload model folder %s -> %s", path, repo_id)
            models_published.append(repo_id)
        else:
            logger.info("Uploading model folder %s -> %s", path, repo_id)
            push_folder_to_hub(path, repo_id, repo_type="model")
            models_published.append(repo_id)

    # 2. Define Dataset Targets
    dataset_targets = [
        ("circuitgraft-med-sft-curated", REPO_ROOT / "artifacts" / "data" / "sft", "Curated domain SFT dataset (MedMCQA with explanations) via CuratorKIT.", False),
        ("circuitgraft-safety-pairs", REPO_ROOT / "artifacts" / "data" / "safety_pairs", "Safety contrastive pairs generated with minimal harmless twins.", True),
        ("circuitgraft-domain-pairs", REPO_ROOT / "artifacts" / "data" / "domain_pairs", "Domain medical contrastive pairs for circuit discovery.", False),
        ("circuitgraft-med-sensitive-eval", REPO_ROOT / "artifacts" / "data" / "domain_sensitive", "Domain-sensitive benign and adversarial medical evaluation prompts.", True),
        ("circuitgraft-circuits", REPO_ROOT / "artifacts" / "circuits", "Discovered safety and domain circuits, overlap reports and faithfulness scores.", False),
        ("circuitgraft-eval-logs", REPO_ROOT / "results", "Complete evaluation metrics, judge logs, AuditKIT fingerprints and manifests.", False),
    ]

    for repo_suffix, path, desc, warning in dataset_targets:
        if not path.exists():
            logger.info("Skipping missing dataset target: %s (%s)", repo_suffix, path)
            continue

        repo_id = f"{hf_org}/{repo_suffix}"
        card_content = generate_dataset_card(repo_id, description=desc, content_warning=warning)
        if not dry_run:
            card_path = path / "README.md" if path.is_dir() else path.parent / f"{repo_suffix}_README.md"
            card_path.write_text(card_content, encoding="utf-8")

        upload_dir = path if path.is_dir() else path.parent

        if dry_run:
            logger.info("[Dry Run] Would upload dataset folder %s -> %s", upload_dir, repo_id)
            datasets_published.append(repo_id)
        else:
            logger.info("Uploading dataset folder %s -> %s", upload_dir, repo_id)
            push_folder_to_hub(upload_dir, repo_id, repo_type="dataset")
            datasets_published.append(repo_id)

    # 3. Create Collections if not dry run
    models_collection_slug = None
    datasets_collection_slug = None

    if not dry_run and api is not None:
        try:
            if models_published:
                col = api.create_collection(
                    title="CircuitGraft Models",
                    namespace=hf_org,
                    description="Models produced by CircuitGraft: domain drifted and safety-repaired checkpoints.",
                )
                models_collection_slug = col.slug
                for m_id in models_published:
                    api.add_collection_item(col.slug, item_id=m_id, item_type="model")

            if datasets_published:
                col = api.create_collection(
                    title="CircuitGraft Datasets & Artifacts",
                    namespace=hf_org,
                    description="Curated datasets, contrastive pairs, circuit scores, and evaluation logs for CircuitGraft.",
                )
                datasets_collection_slug = col.slug
                for d_id in datasets_published:
                    api.add_collection_item(col.slug, item_id=d_id, item_type="dataset")
        except Exception as e:
            logger.warning("Could not create or populate Hugging Face collections: %s", e)

    manifest = new_manifest(
        stage="publish",
        run_id=f"publish-{int(time.time())}",
        config_hash="publish",
    )
    manifest.status = "complete"
    manifest.finished_at = datetime.now(UTC).isoformat()
    manifest.write(out_dir)
    summary = {
        "hf_org": hf_org,
        "models_published": models_published,
        "datasets_published": datasets_published,
        "models_collection": models_collection_slug,
        "datasets_collection": datasets_collection_slug,
        "dry_run": dry_run,
        "manifest": manifest,
    }
    dump_json(out_dir / "summary.json", summary)

    return summary
