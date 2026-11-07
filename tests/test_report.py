"""Unit tests for S9 reporting (tables, plots, injection, pipeline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from circuitgraft.report.inject_readme import inject_all
from circuitgraft.report.pipeline import run_report
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


def test_generate_main_results_table() -> None:
    eval_matrix: list[dict[str, Any]] = [
        {
            "model": "theta_A",
            "srr": 1.0,
            "dgr": 0.0,
            "harmbench_hrr": 0.94,
            "advbench_hrr": 0.98,
            "medmcqa_acc": 0.42,
            "mmlu_acc": 0.65,
            "xstest_frr": 0.08,
            "footprint": 0,
        },
        {
            "model": "ctheta",
            "srr": 0.75,
            "srr_ci": [0.68, 0.81],
            "dgr": 0.60,
            "dgr_ci": [0.55, 0.66],
            "harmbench_hrr": 0.82,
            "advbench_hrr": 0.89,
            "medmcqa_acc": 0.51,
            "mmlu_acc": 0.64,
            "xstest_frr": 0.09,
            "footprint": 120,
        },
        {
            "model": "cg_project",
            "srr": 0.88,
            "srr_ci": [0.82, 0.93],
            "dgr": 0.85,
            "dgr_ci": [0.79, 0.90],
            "harmbench_hrr": 0.91,
            "advbench_hrr": 0.95,
            "medmcqa_acc": 0.56,
            "mmlu_acc": 0.65,
            "xstest_frr": 0.07,
            "footprint": 98,
        },
    ]

    table = generate_main_results_table(eval_matrix)
    assert "| Model / Method |" in table
    assert "| theta_A |" in table
    assert "C-ΔΘ (SafeTune weight-arithmetic implementation)" in table or "ctheta (SafeTune weight-arithmetic implementation)" in table
    assert "| cg_project |" in table
    assert "88.0% [82.0%, 93.0%]" in table


def test_generate_attack_robustness_table() -> None:
    attack_matrix: list[dict[str, Any]] = [
        {
            "model": "cg_project",
            "clean_hrr": 0.91,
            "abliteration_hrr": 0.84,
            "harmful_ft": {50: 0.86, 100: 0.81, 200: 0.72},
            "quant_nf4_hrr": 0.89,
            "circuit_quant_hrr": 0.91,
        }
    ]

    table = generate_attack_robustness_table(attack_matrix)
    assert "| Model / Method |" in table
    assert "Abliteration HRR" in table
    assert "91.0%" in table
    assert "84.0%" in table
    assert "72.0%" in table


def test_generate_overlap_table() -> None:
    overlap = {
        "safety_circuit_size": 128,
        "domain_circuit_size": 112,
        "overlap_nodes": ["L12.H3", "L12.H7", "L14.MLP"],
        "overlap_size": 3,
        "jaccard": 0.0127,
    }
    table = generate_overlap_table(overlap)
    assert "| Component Count | 128 | 112 | 3 |" in table
    assert "| Attention Heads | — | — | 2 |" in table
    assert "| MLP Blocks | — | — | 1 |" in table
    assert "0.013" in table


def test_inject_section(tmp_path: Path) -> None:
    doc = tmp_path / "README.md"
    content = """# Header

<!-- RESULTS:MAIN_TABLE_START -->
Old table
<!-- RESULTS:MAIN_TABLE_END -->

## Footer
"""
    doc.write_text(content, encoding="utf-8")

    status = inject_all(doc, {"main_table": "| New | Table |"})
    assert status["main_table"] is True

    updated = doc.read_text(encoding="utf-8")
    assert "| New | Table |" in updated
    assert "Old table" not in updated
    assert "# Header" in updated
    assert "## Footer" in updated


def test_plots(tmp_path: Path) -> None:
    eval_matrix = [
        {"model": "theta_A", "srr": 1.0, "dgr": 0.0},
        {"model": "theta_D", "srr": 0.0, "dgr": 1.0},
        {"model": "cg_project", "srr": 0.9, "dgr": 0.85},
    ]
    pareto_path = tmp_path / "pareto.png"
    result = plot_pareto_frontier(eval_matrix, pareto_path)
    if result is not None:
        assert pareto_path.exists()

    heatmap_path = tmp_path / "heatmap.png"
    s_scores = {"L0.H1": 0.5, "L1.MLP": 0.8}
    d_scores = {"L1.MLP": 0.6, "L2.H3": 0.3}
    result_hm = plot_overlap_heatmap(s_scores, d_scores, heatmap_path, n_layers=4, n_heads=4)
    if result_hm is not None:
        assert heatmap_path.exists()

    decay_path = tmp_path / "decay.png"
    attack_res = {
        "harmful_ft_decay": {
            "cg_project": {"step_0": 0.9, "step_50": 0.85, "step_100": 0.8, "step_200": 0.7}
        }
    }
    result_decay = plot_attack_decay(attack_res, decay_path)
    if result_decay is not None:
        assert decay_path.exists()


def test_run_report_end_to_end(tmp_path: Path) -> None:
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    out_dir = tmp_path / "artifacts" / "report"
    doc_path = tmp_path / "README.md"
    doc_path.write_text("<!-- RESULTS:MAIN_TABLE_START -->\n<!-- RESULTS:MAIN_TABLE_END -->", encoding="utf-8")

    result = run_report(results_dir=res_dir, output_dir=out_dir, readme_path=doc_path)
    assert Path(result["output_dir"]).exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "manifest.json").exists()
    assert result["injected"]["main_table"] is True
