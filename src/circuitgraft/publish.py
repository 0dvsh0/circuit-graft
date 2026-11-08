"""Publication pipeline for Hugging Face Hub (S10).

Exports the models, datasets, circuit artifacts and evaluation logs produced by
the earlier stages, generates compliant model/dataset cards with lineage and
licence warnings, and groups the results into two public collections (Models and
Datasets) as required by the hackathon submission.

The stage resolves the real stage outputs (``artifacts/graft/<name>-<hash>``,
``artifacts/baselines/<method>-<hash>/merged``, ``artifacts/drift/<run>/merged``
and the S1 data tree) instead of relying on fixed directory names. Shared data
folders are staged with an allow-list so, for example, the safety-pairs repo does
not accidentally receive the domain pairs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuitgraft.config import BaseConfig
from circuitgraft.registry import (
    REPO_ROOT,
    dump_json,
    existing_manifest,
    new_manifest,
    push_folder_to_hub,
)

logger = logging.getLogger(__name__)

_WEIGHT_NAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)

_CARD_TAGS = ["circuitgraft", "tiny-aya", "lexsi-hackathon", "safety-recovery"]


@dataclass
class PublishTarget:
    repo_suffix: str
    kind: str  # "model" | "dataset"
    path: Path | None
    description: str
    method: str | None = None
    drifted: bool = False
    content_warning: bool = False
    patterns: list[str] = field(default_factory=list)
    source_hint: str = ""

    @property
    def repo_id(self) -> str:
        return self.repo_suffix


def _repo_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def _has_weights(directory: Path) -> bool:
    return directory.is_dir() and any((directory / name).exists() for name in _WEIGHT_NAMES)


def _manifest_for(directory: Path) -> Any:
    manifest = existing_manifest(directory)
    if manifest is not None:
        return manifest
    return existing_manifest(directory.parent)


def _newest(candidates: list[Path]) -> Path | None:
    def key(path: Path) -> tuple[str, float]:
        manifest = _manifest_for(path)
        stamp = manifest.finished_at if manifest and manifest.finished_at else ""
        return stamp, path.stat().st_mtime

    return max(candidates, key=key) if candidates else None


def _latest_completed_dir(
    root: Path,
    pattern: str,
    *,
    require: str,
    subdirs: tuple[str, ...] = ("",),
) -> Path | None:
    """Newest run directory containing ``require``, optionally in a subdir."""
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for run_dir in root.glob(pattern):
        if not run_dir.is_dir():
            continue
        manifest = existing_manifest(run_dir)
        if manifest is not None and manifest.status not in ("complete", "partial"):
            continue
        for subdir in subdirs:
            candidate = run_dir / subdir if subdir else run_dir
            if (candidate / require).exists():
                candidates.append(candidate)
    return _newest(candidates)


def _latest_model_dir(
    root: Path,
    pattern: str,
    *,
    subdirs: tuple[str, ...] = ("", "merged"),
) -> Path | None:
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for run_dir in root.glob(pattern):
        if not run_dir.is_dir():
            continue
        manifest = existing_manifest(run_dir)
        if manifest is not None and manifest.status not in ("complete", "partial"):
            continue
        for subdir in subdirs:
            candidate = run_dir / subdir if subdir else run_dir
            if _has_weights(candidate):
                candidates.append(candidate)
    return _newest(candidates)


def _model_targets(root: Path) -> list[PublishTarget]:
    drift_root = root / "artifacts" / "drift"
    graft_root = root / "artifacts" / "graft"
    baseline_root = root / "artifacts" / "baselines"
    attack_root = root / "artifacts" / "attacks" / "abliterate"

    targets: list[PublishTarget] = [
        PublishTarget(
            "circuitgraft-tinyaya-med-drifted",
            "model",
            _latest_model_dir(drift_root, "*", subdirs=("merged", "")),
            "Domain-fine-tuned medical specialist (unrepaired).",
            method="Drifted (Unrepaired)",
            drifted=True,
            source_hint="artifacts/drift/*/merged",
        ),
        PublishTarget(
            "circuitgraft-tinyaya-med-lora",
            "model",
            _latest_completed_dir(
                drift_root, "*", require="adapter_config.json", subdirs=("adapter",)
            ),
            "LoRA adapter produced during drift induction.",
            method="Drift LoRA Adapter",
            drifted=True,
            source_hint="artifacts/drift/*/adapter",
        ),
    ]

    graft_labels = {
        "cg_project": "CircuitGraft SVD-Project (Primary)",
        "cg_uniform": "CircuitGraft Uniform",
        "cg_exclude": "CircuitGraft Exclude",
        "cg_fisher": "CircuitGraft Fisher Gate",
        "cg_distributed": "CircuitGraft Distributed",
        "cg_random": "CircuitGraft Random-Matched Control",
    }
    for name, label in graft_labels.items():
        targets.append(
            PublishTarget(
                f"circuitgraft-tinyaya-med-{name.replace('_', '-')}",
                "model",
                _latest_model_dir(graft_root, f"{name}-*"),
                f"{label} repaired checkpoint.",
                method=label,
                source_hint=f"artifacts/graft/{name}-*",
            )
        )

    baseline_labels = {
        "resta": "SafeTune RESTA Baseline",
        "wiseft": "WiSE-FT Baseline",
        "ctheta": "C-ΔΘ (SafeTune weight-arithmetic implementation)",
        "safelora": "Safe LoRA Baseline",
        "lox": "LoX Baseline",
        "task_arith": "Task Arithmetic Baseline",
    }
    for name, label in baseline_labels.items():
        targets.append(
            PublishTarget(
                f"circuitgraft-tinyaya-med-{name.replace('_', '-')}",
                "model",
                _latest_model_dir(baseline_root, f"{name}-*", subdirs=("merged", "")),
                f"{label} repaired checkpoint.",
                method=label,
                source_hint=f"artifacts/baselines/{name}-*/merged",
            )
        )

    for run_dir in sorted(attack_root.glob("*")) if attack_root.is_dir() else []:
        merged = run_dir / "merged"
        if _has_weights(merged):
            targets.append(
                PublishTarget(
                    f"circuitgraft-tinyaya-med-abliterated-{run_dir.name.lower()}",
                    "model",
                    merged,
                    "Research-only abliteration attack checkpoint.",
                    method="Abliteration attack (research-only)",
                    drifted=True,
                    source_hint=f"artifacts/attacks/abliterate/{run_dir.name}/merged",
                )
            )

    return targets


def _data_targets(root: Path) -> list[PublishTarget]:
    data_root = root / "artifacts" / "data"
    return [
        PublishTarget(
            "circuitgraft-med-sft-curated",
            "dataset",
            data_root / "medmcqa_sft",
            "Curated domain SFT dataset (MedMCQA with explanations) via CuratorKIT.",
            patterns=["train.jsonl", "validation.jsonl", "section.json", "curatorkit/**"],
            source_hint="artifacts/data/medmcqa_sft",
        ),
        PublishTarget(
            "circuitgraft-safety-pairs",
            "dataset",
            data_root / "pairs",
            "Safety contrastive pairs with minimal harmless twins.",
            content_warning=True,
            patterns=[
                "safety.jsonl",
                "safety.rejected.jsonl",
                "review_safety.csv",
                "validation.json",
                "eval_overlap.json",
            ],
            source_hint="artifacts/data/pairs (safety files)",
        ),
        PublishTarget(
            "circuitgraft-domain-pairs",
            "dataset",
            data_root / "pairs",
            "Domain medical contrastive pairs for circuit discovery.",
            patterns=[
                "domain.jsonl",
                "domain.rejected.jsonl",
                "review_domain.csv",
                "section.json",
                "validation.json",
                "eval_overlap.json",
            ],
            source_hint="artifacts/data/pairs (domain files)",
        ),
        PublishTarget(
            "circuitgraft-med-sensitive-eval",
            "dataset",
            data_root / "eval",
            "Domain-sensitive benign and harmful medical evaluation prompts.",
            content_warning=True,
            patterns=["domain_benign.jsonl", "domain_harmful.jsonl"],
            source_hint="artifacts/data/eval/domain_*.jsonl",
        ),
    ]


def _circuit_target(root: Path) -> PublishTarget:
    discover_dir = _latest_completed_dir(
        root / "artifacts" / "discover", "discover-*", require="summary.json"
    )
    return PublishTarget(
        "circuitgraft-circuits",
        "dataset",
        discover_dir,
        "Discovered safety/domain circuits, overlap reports and faithfulness scores.",
        patterns=[
            "summary.json",
            "overlap*.json",
            "cross_check.json",
            "safety/*.json",
            "domain/*.json",
        ],
        source_hint="artifacts/discover/discover-*",
    )


def _eval_logs_target(root: Path) -> PublishTarget:
    results = root / "results"
    return PublishTarget(
        "circuitgraft-eval-logs",
        "dataset",
        results if results.is_dir() else None,
        "Evaluation metrics, AuditKIT fingerprints, manifests and report tables.",
        patterns=["**/*.json", "**/*.md", "**/*.csv", "**/*.txt"],
        source_hint="results/ (generations excluded)",
    )


def generate_model_card(
    repo_name: str,
    *,
    base_model: str = "CohereLabs/tiny-aya-base",
    aligned_model: str = "CohereLabs/tiny-aya-global",
    drifted: bool = False,
    method: str | None = None,
    config: dict[str, Any] | None = None,
    extra_links: dict[str, str] | None = None,
) -> str:
    """Generate a model card markdown with metadata and licensing."""
    tags = list(_CARD_TAGS)
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

    config_block = ""
    if config:
        config_block = (
            "\n## Run configuration\n```json\n"
            + json.dumps(config, indent=2, sort_keys=True, default=str)
            + "\n```\n"
        )

    link_block = ""
    if extra_links:
        link_block = "\n## Links\n" + "\n".join(
            f"- [{label}]({url})" for label, url in extra_links.items()
        ) + "\n"

    return f"""---
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

## Model lineage
1. **Base foundation**: `{base_model}`
2. **Safety-aligned**: `{aligned_model}`
3. **Domain specialist**: Cohere Tiny Aya fine-tuned on medical data (MedMCQA)
4. **Recovery method**: `{method or "Domain drifted (unrepaired)"}`

## Provenance
- Repository: [CircuitGraft](https://github.com/dubeydevansh/circuit-graft)
- Method: circuit-aware safety recovery using CircuitKIT and SafeTune.
- Evaluation: every number in the project report comes from the eval-logs dataset.

## Licence and terms of use
- Weights are derived from Cohere Tiny Aya and are subject to the **CC-BY-NC-4.0**
  licence and the [Cohere Acceptable Use Policy](https://cohere.com/c4ai-acceptable-use-policy).
- Research conducted for the Lexsi Labs Hackathon (Track 2b / Problem 6).
{config_block}{link_block}""".strip() + "\n"


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
> **CONTENT ADVISORY**: This dataset contains benchmark adversarial prompts and refusal
> evaluation targets. Full harmful completions are withheld, and records are provided for
> safety research only.
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
Created for CircuitGraft: circuit-aware safety recovery for domain-fine-tuned Cohere
Tiny Aya models. The dataset card is generated by `cg publish`; the full lineage is in
the repository manifest.
""".strip() + "\n"


def _stage_dataset(
    target: PublishTarget,
    staging_root: Path,
    *,
    root: Path,
    card: str,
) -> Path | None:
    """Copy the allow-listed files into a clean staging directory."""
    if target.path is None or not target.path.exists():
        return None

    destination = staging_root / target.repo_suffix
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "README.md").write_text(card, encoding="utf-8")

    source = target.path
    copied = 0
    if target.repo_suffix == "circuitgraft-eval-logs":
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if "generations" in relative.parts:
                continue
            if path.suffix.lower() not in {".json", ".md", ".csv", ".txt", ".yaml", ".yml"}:
                continue
            out = destination / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            copied += 1
    else:
        for pattern in target.patterns:
            for path in sorted(source.glob(pattern)):
                if not path.is_file():
                    continue
                relative = path.relative_to(source)
                out = destination / relative
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
                copied += 1

    if copied == 0:
        shutil.rmtree(destination)
        return None
    return destination


def run_publish(
    base: BaseConfig,
    *,
    dry_run: bool = False,
    output_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Publish models, datasets and collections to Hugging Face Hub."""
    hf_org = base.hf_org
    if not hf_org:
        raise ValueError("base.hf_org is empty; configure HF_ORG in .env or configs/base.yaml")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not dry_run and not token:
        raise ValueError("HF_TOKEN is required to publish to Hugging Face Hub.")

    root = Path(repo_root) if repo_root else REPO_ROOT
    out_dir = Path(output_dir) if output_dir else root / "artifacts" / "publish"
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_root = out_dir / "staging"

    api = None
    if not dry_run:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

    published: dict[str, list[dict[str, str]]] = {"models": [], "datasets": []}
    skipped: list[dict[str, str]] = []

    # 1. Models.
    for target in _model_targets(root):
        repo_id = f"{hf_org}/{target.repo_suffix}"
        if target.path is None or not target.path.exists():
            skipped.append(
                {
                    "repo": repo_id,
                    "kind": "model",
                    "reason": f"no checkpoint found for {target.source_hint}",
                }
            )
            continue
        card = generate_model_card(
            repo_id,
            drifted=target.drifted,
            method=target.method,
        )
        if dry_run:
            logger.info("[dry run] would upload model %s -> %s", target.path, repo_id)
        else:
            (target.path / "README.md").write_text(card, encoding="utf-8")
            logger.info("uploading model %s -> %s", target.path, repo_id)
            push_folder_to_hub(
                target.path,
                repo_id,
                repo_type="model",
                commit_message="CircuitGraft model upload",
            )
        published["models"].append(
            {"repo_id": repo_id, "source": _display_path(target.path, root)}
        )

    # 2. Datasets (staged so shared folders are filtered correctly).
    dataset_targets = _data_targets(root) + [_circuit_target(root), _eval_logs_target(root)]
    for target in dataset_targets:
        repo_id = f"{hf_org}/{target.repo_suffix}"
        card = generate_dataset_card(
            repo_id,
            description=target.description,
            content_warning=target.content_warning,
        )
        staged = _stage_dataset(target, staging_root, root=root, card=card)
        if staged is None:
            skipped.append(
                {
                    "repo": repo_id,
                    "kind": "dataset",
                    "reason": f"no files found for {target.source_hint}",
                }
            )
            continue
        if dry_run:
            logger.info("[dry run] would upload dataset %s -> %s", staged, repo_id)
        else:
            logger.info("uploading dataset %s -> %s", staged, repo_id)
            push_folder_to_hub(
                staged,
                repo_id,
                repo_type="dataset",
                commit_message="CircuitGraft dataset upload",
            )
        published["datasets"].append(
            {"repo_id": repo_id, "source": _display_path(staged, root)}
        )

    # 3. Collections.
    collections: dict[str, str | None] = {"models": None, "datasets": None}
    if api is not None:
        try:
            if published["models"]:
                col = api.create_collection(
                    title="CircuitGraft Models",
                    namespace=hf_org,
                    description=(
                        "Models produced by CircuitGraft: domain drifted and "
                        "safety-repaired checkpoints."
                    ),
                )
                collections["models"] = col.slug
                for item in published["models"]:
                    api.add_collection_item(
                        col.slug, item_id=item["repo_id"], item_type="model"
                    )
            if published["datasets"]:
                col = api.create_collection(
                    title="CircuitGraft Datasets & Artifacts",
                    namespace=hf_org,
                    description=(
                        "Curated datasets, contrastive pairs, circuit scores and "
                        "evaluation logs for CircuitGraft."
                    ),
                )
                collections["datasets"] = col.slug
                for item in published["datasets"]:
                    api.add_collection_item(
                        col.slug, item_id=item["repo_id"], item_type="dataset"
                    )
        except Exception as exc:  # noqa: BLE001 - collection creation is best effort
            logger.warning("could not create or populate Hugging Face collections: %s", exc)

    run_id = f"publish-{int(time.time())}"
    manifest = new_manifest(
        stage="publish",
        run_id=run_id,
        config_hash="publish",
    )
    manifest.status = "complete" if not skipped else "partial"
    manifest.notes = f"dry_run={dry_run}; skipped={len(skipped)}"
    manifest.finished_at = datetime.now(UTC).isoformat()
    manifest.write(out_dir)
    summary = {
        "run_id": run_id,
        "hf_org": hf_org,
        "dry_run": dry_run,
        "models_published": published["models"],
        "datasets_published": published["datasets"],
        "skipped": skipped,
        "collections": collections,
        "manifest": manifest,
    }
    dump_json(out_dir / "summary.json", summary)
    return summary
