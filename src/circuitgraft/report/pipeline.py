"""Report pipeline runner (S9).

Aggregates outputs from eval (S7), attacks (S8), and discovery (S4), builds
markdown tables, generates Pareto/overlap/decay plots, injects the result blocks
into README/Article, and writes a small, git-trackable copy under
``results/s9/<run_id>/``.

The stage deliberately works from the real artifact tree: S7 writes
``artifacts/eval/eval-<hash>/summary.json``, S4 writes
``artifacts/discover/discover-<hash>/overlap.json`` and S8 writes
``artifacts/attacks/<type>/<run>/attack_stats.json``. A missing source is a
reason for a partial report, never a fabricated metric.
"""

from __future__ import annotations

import logging
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuitgraft.discover.overlap import read_score_payload
from circuitgraft.registry import (
    REPO_ROOT,
    dump_json,
    existing_manifest,
    hash_paths,
    load_json,
    new_manifest,
)
from circuitgraft.report.inject_readme import inject_all
from circuitgraft.report.plots import (
    plot_attack_decay,
    plot_overlap_heatmap,
    plot_pareto_frontier,
)
from circuitgraft.report.tables import (
    generate_attack_robustness_table,
    generate_main_results_table,
    generate_overlap_table,
    normalise_attack_rows,
    normalise_eval_rows,
    normalise_overlap_report,
)

logger = logging.getLogger(__name__)


def _repo_path(value: str | Path, root: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def _latest_run_file(
    root: str | Path,
    pattern: str,
    filename: str,
    *,
    repo_root: Path = REPO_ROOT,
    allowed_statuses: tuple[str, ...] = ("complete", "partial"),
) -> Path | None:
    """Newest run directory under ``root`` that contains ``filename``.

    Run directories with a failed/incomplete manifest are ignored. A directory
    without a manifest is accepted for backwards compatibility with the small
    result copies kept under ``results/``.
    """
    root = _repo_path(root, repo_root)
    if not root.is_dir():
        return None

    candidates: list[tuple[str, float, Path]] = []
    for run_dir in root.glob(pattern):
        if not run_dir.is_dir():
            continue
        target = run_dir / filename
        if not target.exists():
            continue
        manifest = existing_manifest(run_dir)
        if manifest is not None and manifest.status not in allowed_statuses:
            continue
        stamp = manifest.finished_at if manifest and manifest.finished_at else ""
        candidates.append((stamp, run_dir.stat().st_mtime, target))

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _load_explicit(candidates: list[Path]) -> tuple[dict[str, Any], Path | None]:
    for path in candidates:
        if path.exists():
            try:
                payload = load_json(path)
            except Exception as exc:  # noqa: BLE001 - fall through to other sources
                logger.warning("failed to load %s: %s", path, exc)
                continue
            if isinstance(payload, dict):
                return payload, path
    return {}, None


def _find_eval_summary(
    results_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], Path | None]:
    explicit = _load_explicit(
        [
            results_dir / "eval" / "summary.json",
            results_dir / "eval_summary.json",
            results_dir / "eval.json",
        ]
    )
    if explicit[1] is not None:
        return explicit

    for root, pattern in (
        (repo_root / "artifacts" / "eval", "eval-*"),
        (repo_root / "artifacts" / "eval_smoke", "eval-*"),
        (repo_root / "results" / "s7", "*"),
    ):
        path = _latest_run_file(root, pattern, "summary.json", repo_root=repo_root)
        if path is not None:
            return load_json(path), path
    return {}, None


def _find_overlap(
    results_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    path = _latest_run_file(
        repo_root / "artifacts" / "discover",
        "discover-*",
        "overlap.json",
        repo_root=repo_root,
    )
    if path is not None:
        return load_json(path), path, path.parent

    explicit = _load_explicit(
        [
            results_dir / "overlap" / "report.json",
            results_dir / "overlap_report.json",
            repo_root / "artifacts" / "circuits" / "overlap_report.json",
        ]
    )
    if explicit[1] is not None:
        return explicit[0], explicit[1], explicit[1].parent
    return {}, None, None


def _find_scores(discover_dir: Path | None) -> tuple[dict[str, float], dict[str, float]]:
    if discover_dir is None:
        return {}, {}
    safety_path = discover_dir / "safety" / "scores.json"
    domain_path = discover_dir / "domain" / "scores.json"
    safety = read_score_payload(safety_path) if safety_path.exists() else {}
    domain = read_score_payload(domain_path) if domain_path.exists() else {}
    return safety, domain


def _post_eval_hrr(document: dict[str, Any]) -> Any:
    metrics = document.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get("hrr")
    return None


def _apply_post_eval(
    row: dict[str, Any],
    attack_type: str,
    post_eval: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    """Merge measured S7 metrics from an S8 follow-up ``post_eval.json``."""
    if attack_type == "harmful_ft":
        by_step = post_eval.get("metrics_by_step")
        if isinstance(by_step, dict):
            values: dict[Any, Any] = {}
            for raw_step, document in by_step.items():
                if not isinstance(document, dict):
                    continue
                hrr = _post_eval_hrr(document)
                if hrr is None:
                    continue
                try:
                    step: Any = int(raw_step)
                except (TypeError, ValueError):
                    step = raw_step
                values[step] = hrr
            if values:
                row["harmful_ft"] = values
        return

    hrr = _post_eval_hrr(post_eval)
    if hrr is None:
        return
    if attack_type == "abliterate":
        row["abliteration_hrr"] = hrr
    elif attack_type == "quantize":
        backend = str(post_eval.get("backend") or stats.get("backend") or "")
        circuit_aware = bool(
            isinstance(stats.get("quant_spec"), dict)
            and stats["quant_spec"].get("circuit_aware")
        )
        if backend == "nf4" and not circuit_aware:
            row["quant_nf4_hrr"] = hrr
        if circuit_aware:
            row["circuit_quant_hrr"] = hrr


def _collect_attack_stats(
    results_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], list[Path]]:
    """Load an explicit attack summary or assemble one from S8 run stats."""
    explicit = _load_explicit(
        [
            results_dir / "attacks" / "summary.json",
            results_dir / "attacks_summary.json",
            results_dir / "attacks.json",
        ]
    )
    if explicit[1] is not None:
        return explicit[0], [explicit[1]]

    stats_paths = sorted(
        (
            path
            for path in (repo_root / "artifacts" / "attacks").glob(
                "*/*/attack_stats.json"
            )
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
    )

    by_model: dict[str, dict[str, Any]] = {}
    used: list[Path] = []
    post_eval_sources: list[Path] = []
    for path in stats_paths:
        manifest = existing_manifest(path.parent)
        if manifest is not None and manifest.status not in ("complete", "partial"):
            continue
        try:
            stats = load_json(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load attack stats %s: %s", path, exc)
            continue
        if not isinstance(stats, dict):
            continue
        post_eval: dict[str, Any] | None = None
        post_eval_path = path.parent / "post_eval.json"
        if post_eval_path.is_file():
            try:
                payload = load_json(post_eval_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to load post-eval file %s: %s", post_eval_path, exc)
            else:
                if isinstance(payload, dict):
                    post_eval = payload
                    post_eval_sources.append(post_eval_path)

        model = str(stats.get("model") or "unknown")
        attack_type = str(stats.get("attack_type") or path.parent.parent.name)
        row = by_model.setdefault(model, {"model": model})
        row[f"{attack_type}_status"] = stats.get("status")
        row[f"{attack_type}_run"] = _display_path(path.parent, repo_root)
        used.append(path)

        hrr = stats.get("hrr")
        metrics = stats.get("metrics")
        if hrr is None and isinstance(metrics, dict):
            hrr = metrics.get("hrr")
        if hrr is not None:
            row[f"{attack_type}_hrr"] = hrr

        if attack_type == "harmful_ft":
            by_step = stats.get("hrr_by_step") or stats.get("metrics_by_step")
            if isinstance(by_step, dict):
                row["harmful_ft"] = by_step
        elif attack_type == "quantize":
            backend = str(stats.get("backend") or "")
            circuit_aware = bool(
                isinstance(stats.get("quant_spec"), dict)
                and stats["quant_spec"].get("circuit_aware")
            )
            if hrr is not None and backend == "nf4" and not circuit_aware:
                row["quant_nf4_hrr"] = hrr
            if hrr is not None and circuit_aware:
                row["circuit_quant_hrr"] = hrr
        elif attack_type == "abliterate" and hrr is not None:
            row["abliteration_hrr"] = hrr

        if post_eval is not None:
            row[f"{attack_type}_post_eval"] = _display_path(post_eval_path, repo_root)
            row[f"{attack_type}_post_eval_status"] = post_eval.get("status")
            _apply_post_eval(row, attack_type, post_eval, stats)

    return {
        "models": list(by_model.values()),
        "sources": [_display_path(path, repo_root) for path in used],
        "post_eval_sources": [
            _display_path(path, repo_root) for path in post_eval_sources
        ],
        "note": (
            "Attack HRR is shown only where the attack run saved a measured "
            "metric. S8 stores attacked checkpoints; run `cg eval-attacks` to "
            "fill the remaining cells."
        ),
    }, used


def _attach_clean_hrr(
    attack_data: dict[str, Any],
    eval_rows: list[dict[str, Any]],
) -> None:
    by_name = {str(row.get("model")): row for row in eval_rows}

    def _apply(row: dict[str, Any]) -> None:
        clean = by_name.get(str(row.get("model")))
        if clean is not None and row.get("clean_hrr") is None:
            row["clean_hrr"] = clean.get("hrr")

    models = attack_data.get("models")
    if isinstance(models, list):
        for row in models:
            if isinstance(row, dict):
                _apply(row)
    elif isinstance(models, dict):
        for document in models.values():
            if isinstance(document, dict):
                _apply(document)
    else:
        for row in normalise_attack_rows(attack_data):
            _apply(row)


def run_report(
    *,
    results_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    readme_path: str | Path | None = None,
    results_out: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run S9 report pipeline."""
    root = Path(repo_root) if repo_root else REPO_ROOT
    res_dir = Path(results_dir) if results_dir else root / "results"
    out_dir = Path(output_dir) if output_dir else root / "artifacts" / "report"
    doc_path = Path(readme_path) if readme_path else root / "README.md"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"report-{int(time.time())}"
    tracked_dir = Path(results_out) if results_out else root / "results" / "s9" / run_id
    tracked_dir.mkdir(parents=True, exist_ok=True)

    reasons: list[str] = []
    sources: list[Path] = []

    # 1. Evaluation summary (real S7 output or an explicit result copy).
    eval_data, eval_path = _find_eval_summary(res_dir, root)
    if eval_path is None:
        reasons.append("no S7 evaluation summary found (run `cg eval --all`)")
    else:
        sources.append(eval_path)
        if eval_data.get("status") and eval_data["status"] != "complete":
            reasons.append(f"eval summary is {eval_data['status']}")
        reasons.extend(str(reason) for reason in eval_data.get("partial_reasons", []))

    # 2. Discovery overlap and the score maps used by the heatmap.
    overlap_data, overlap_path, discover_dir = _find_overlap(res_dir, root)
    if overlap_path is None:
        reasons.append("no S4 discovery overlap report found (run `cg discover`)")
    else:
        sources.append(overlap_path)
    safety_scores, domain_scores = _find_scores(discover_dir)

    # 3. Attack runs; combine with clean HRR from the eval summary.
    attack_data, attack_sources = _collect_attack_stats(res_dir, root)
    sources.extend(attack_sources)
    if not normalise_attack_rows(attack_data):
        reasons.append("no S8 attack statistics found (run `cg attack ...`)")

    eval_rows = normalise_eval_rows(eval_data)
    _attach_clean_hrr(attack_data, eval_rows)

    # 4. Tables and plots.
    main_table = generate_main_results_table(eval_data)
    attack_table = generate_attack_robustness_table(attack_data)
    overlap_table = generate_overlap_table(overlap_data)

    (out_dir / "table_main.md").write_text(main_table, encoding="utf-8")
    (out_dir / "table_attacks.md").write_text(attack_table, encoding="utf-8")
    (out_dir / "table_overlap.md").write_text(overlap_table, encoding="utf-8")

    pareto_png = out_dir / "pareto_frontier.png"
    plot_pareto_frontier(eval_data, pareto_png)

    heatmap_png = out_dir / "circuit_overlap_heatmap.png"
    if safety_scores or domain_scores:
        plot_overlap_heatmap(safety_scores, domain_scores, heatmap_png)

    decay_png = out_dir / "attack_decay.png"
    if attack_data.get("harmful_ft_decay"):
        plot_attack_decay(attack_data, decay_png)

    # Plots are small enough to track next to the report tables; the README
    # links to the tracked copy because ``artifacts/`` is gitignored.
    tracked_plots: list[Path] = []
    for plot_path in (pareto_png, heatmap_png, decay_png):
        if not plot_path.exists():
            continue
        tracked_plot = tracked_dir / plot_path.name
        shutil.copy2(plot_path, tracked_plot)
        tracked_plots.append(tracked_plot)

    tracked_pareto = tracked_dir / "pareto_frontier.png"
    pareto_snippet = (
        f"![Pareto Frontier]({_display_path(tracked_pareto, root)})"
        if tracked_pareto.exists()
        else ""
    )
    sections_to_inject = {
        "main_table": main_table,
        "attack_table": attack_table,
        "overlap_table": overlap_table,
        "pareto_plot": pareto_snippet,
    }
    injected_status = inject_all(doc_path, sections_to_inject)

    # 6. Combined summary and a tracked copy under results/s9.
    status = "partial" if reasons else "complete"
    summary_md = f"""# CircuitGraft Experiment Summary

**Status:** {status}

## 1. Safety Recovery vs Domain Retention
{main_table}

{pareto_snippet}

## 2. Circuit Overlap
{overlap_table}

## 3. Adversarial Robustness
{attack_table}

## Partial reasons
{chr(10).join(f'- {reason}' for reason in reasons) if reasons else '- none'}
"""
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    summary_payload: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "partial_reasons": reasons,
        "sources": {
            "eval": _display_path(eval_path, root) if eval_path else None,
            "overlap": _display_path(overlap_path, root) if overlap_path else None,
            "attacks": [_display_path(path, root) for path in attack_sources],
        },
        "eval_status": eval_data.get("status"),
        "gate_g_drift": eval_data.get("gate_g_drift"),
        "models": eval_rows,
        "attacks": normalise_attack_rows(attack_data),
        "attack_post_eval": attack_data.get("post_eval_sources", []),
        "overlap": normalise_overlap_report(overlap_data),
        "injected": injected_status,
        "notes": (
            "This summary is generated from stage artifacts. A partial report "
            "must not be quoted as final experiment evidence."
        ),
    }
    summary_path = dump_json(tracked_dir / "summary.json", summary_payload)
    tracked_tables = {
        "table_main.md": main_table,
        "table_attacks.md": attack_table,
        "table_overlap.md": overlap_table,
        "summary.md": summary_md,
    }
    tracked_paths: list[Path] = [summary_path, *tracked_plots]
    for name, content in tracked_tables.items():
        path = tracked_dir / name
        path.write_text(content, encoding="utf-8")
        tracked_paths.append(path)

    # 7. Manifest.
    manifest = new_manifest(
        stage="report",
        run_id=run_id,
        config_hash="report",
        inputs=sorted(set(sources)),
    )
    manifest.outputs = hash_paths(tracked_paths)
    manifest.status = status
    manifest.finished_at = datetime.now(UTC).isoformat()
    manifest.notes = "; ".join(reasons) if reasons else None
    manifest.write(out_dir)
    manifest.write(tracked_dir)

    logger.info("Report generated successfully in %s", out_dir)
    return {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "results_dir": str(tracked_dir),
        "status": status,
        "partial_reasons": reasons,
        "injected": injected_status,
        "manifest": manifest,
    }
