"""Report pipeline runner (S9).

Aggregates outputs from eval (S7), attacks (S8), and discovery (S4),
builds markdown tables, generates Pareto/overlap/decay plots,
and injects results into README.md and documentation without hand-typed metrics.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuitgraft.registry import (
    REPO_ROOT,
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
)

logger = logging.getLogger(__name__)


def _find_json(candidates: list[Path]) -> dict[str, Any]:
    for path in candidates:
        if path.exists():
            try:
                return load_json(path)
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
    return {}


def run_report(
    *,
    results_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    readme_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run S9 report pipeline."""
    res_dir = Path(results_dir) if results_dir else REPO_ROOT / "results"
    out_dir = Path(output_dir) if output_dir else REPO_ROOT / "artifacts" / "report"
    doc_path = Path(readme_path) if readme_path else REPO_ROOT / "README.md"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Eval results
    eval_data = _find_json([
        res_dir / "eval" / "summary.json",
        res_dir / "eval_summary.json",
        res_dir / "eval.json",
    ])

    # 2. Load Attacks results
    attack_data = _find_json([
        res_dir / "attacks" / "summary.json",
        res_dir / "attacks_summary.json",
        res_dir / "attacks.json",
    ])

    # 3. Load Overlap results
    overlap_data = _find_json([
        REPO_ROOT / "artifacts" / "circuits" / "overlap_report.json",
        res_dir / "overlap" / "report.json",
        res_dir / "overlap_report.json",
    ])

    # Generate tables
    main_table = generate_main_results_table(eval_data)
    attack_table = generate_attack_robustness_table(attack_data)
    overlap_table = generate_overlap_table(overlap_data)

    (out_dir / "table_main.md").write_text(main_table, encoding="utf-8")
    (out_dir / "table_attacks.md").write_text(attack_table, encoding="utf-8")
    (out_dir / "table_overlap.md").write_text(overlap_table, encoding="utf-8")

    # Generate plots
    pareto_png = out_dir / "pareto_frontier.png"
    plot_pareto_frontier(eval_data, pareto_png)

    # Heatmap
    s_scores = overlap_data.get("safety_scores", {})
    d_scores = overlap_data.get("domain_scores", {})
    heatmap_png = out_dir / "circuit_overlap_heatmap.png"
    if s_scores or d_scores:
        plot_overlap_heatmap(s_scores, d_scores, heatmap_png)

    # Attack decay curve
    decay_png = out_dir / "attack_decay.png"
    if "harmful_ft_decay" in attack_data:
        plot_attack_decay(attack_data, decay_png)

    # Prepare markdown snippets for injection
    pareto_snippet = "![Pareto Frontier](artifacts/report/pareto_frontier.png)" if pareto_png.exists() else ""
    sections_to_inject = {
        "main_table": main_table,
        "attack_table": attack_table,
        "overlap_table": overlap_table,
        "pareto_plot": pareto_snippet,
    }

    injected_status = inject_all(doc_path, sections_to_inject)

    # Write combined summary
    summary_md = f"""# CircuitGraft Experiment Summary

## 1. Safety Recovery vs Domain Retention
{main_table}

{pareto_snippet}

## 2. Circuit Overlap
{overlap_table}

## 3. Adversarial Robustness
{attack_table}
"""
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    manifest = new_manifest(
        stage="report",
        run_id=f"report-{int(time.time())}",
        config_hash="report",
        inputs=[f for f in [res_dir / "eval", res_dir / "attacks"] if f.exists()],
    )
    manifest.outputs = hash_paths([out_dir / "summary.md", out_dir / "table_main.md"])
    manifest.status = "complete"
    manifest.finished_at = datetime.now(UTC).isoformat()
    manifest.write(out_dir)

    logger.info("Report generated successfully in %s", out_dir)
    return {
        "output_dir": str(out_dir),
        "injected": injected_status,
        "manifest": manifest,
    }
