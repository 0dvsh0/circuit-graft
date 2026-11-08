"""Unit tests for S9 reporting (tables, plots, injection, pipeline)."""

from __future__ import annotations

import json
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
    normalise_eval_rows,
)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _real_eval_summary(checkpoint: Path) -> dict[str, Any]:
    """A trimmed copy of the real S7 summary shape."""

    def model(
        *,
        checkpoint_id: str,
        hrr: float,
        medmcqa: float,
        mmlu: float,
        xstest: float,
        srr: float,
        dgr: float,
        srr_ci: list[float] | None = None,
        dgr_ci: list[float] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "complete",
            "checkpoint": checkpoint_id,
            "device": "cuda",
            "metrics": {
                "hrr": hrr,
                "frr": xstest,
                "domain_accuracy": medmcqa,
                "general_accuracy": mmlu,
            },
            "composite": {
                "srr": srr,
                "dgr": dgr,
                "orp": 0.0,
                "srr_ci": srr_ci or [srr, srr],
                "dgr_ci": dgr_ci or [dgr, dgr],
            },
            "judged": {
                "safety": {
                    "harmbench": {
                        "benchmark": "harmbench",
                        "status": "complete",
                        "n": 100,
                        "hrr": hrr,
                        "hrr_ci": [max(0.0, hrr - 0.03), min(1.0, hrr + 0.03)],
                        "asr": 1.0 - hrr,
                    },
                    "advbench": {
                        "benchmark": "advbench",
                        "status": "complete",
                        "n": 100,
                        "hrr": min(1.0, hrr + 0.02),
                        "hrr_ci": [max(0.0, hrr - 0.01), min(1.0, hrr + 0.04)],
                        "asr": 1.0 - min(1.0, hrr + 0.02),
                    },
                },
                "mcq": {
                    "medmcqa_validation": {
                        "accuracy": medmcqa,
                        "accuracy_ci": [max(0.0, medmcqa - 0.05), medmcqa + 0.05],
                        "n": 500,
                    },
                    "mmlu_non_medical": {
                        "accuracy": mmlu,
                        "accuracy_ci": [max(0.0, mmlu - 0.04), mmlu + 0.04],
                        "n": 500,
                    },
                },
                "benign": {
                    "xstest_safe": {
                        "frr": xstest,
                        "frr_ci": [max(0.0, xstest - 0.02), xstest + 0.02],
                        "n": 250,
                    }
                },
            },
        }

    return {
        "run_id": "eval-test",
        "status": "complete",
        "partial_reasons": [],
        "gate_g_drift": {"status": "pass", "drop": 0.21},
        "models": {
            "aligned": model(
                checkpoint_id="CohereLabs/tiny-aya-global",
                hrr=0.92,
                medmcqa=0.42,
                mmlu=0.65,
                xstest=0.06,
                srr=0.0,
                dgr=0.0,
            ),
            "drifted": model(
                checkpoint_id="CohereLabs/tiny-aya-base",
                hrr=0.25,
                medmcqa=0.72,
                mmlu=0.63,
                xstest=0.05,
                srr=0.0,
                dgr=1.0,
            ),
            "cg_project": model(
                checkpoint_id=str(checkpoint),
                hrr=0.85,
                medmcqa=0.51,
                mmlu=0.64,
                xstest=0.07,
                srr=0.88,
                dgr=0.85,
                srr_ci=[0.82, 0.93],
                dgr_ci=[0.79, 0.90],
            ),
        },
    }


def test_normalise_eval_rows_real_summary_shape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "artifacts" / "graft" / "cg_project-test" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    summary = _real_eval_summary(checkpoint)

    # Older fixtures nested metrics under a "metrics" key; the real stage
    # writes them flat. Both must normalise identically.
    summary["models"]["aligned"]["judged"]["safety"]["harmbench"] = {
        "status": "complete",
        "metrics": {"hrr": 0.92, "hrr_ci": [0.89, 0.95]},
    }

    rows = normalise_eval_rows(summary)
    by_name = {row["model"]: row for row in rows}

    assert list(by_name) == ["aligned", "drifted", "cg_project"]
    assert by_name["aligned"]["harmbench_hrr"] == 0.92
    assert by_name["aligned"]["medmcqa_acc"] == 0.42
    assert by_name["aligned"]["mmlu_acc"] == 0.65
    assert by_name["aligned"]["xstest_frr"] == 0.06
    assert by_name["cg_project"]["advbench_hrr"] == 0.87
    assert by_name["cg_project"]["srr_ci"] == [0.82, 0.93]
    assert by_name["cg_project"]["hrr"] == 0.85

    table = generate_main_results_table(summary)
    assert (
        "| aligned | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 92.0% | 94.0% | "
        "42.0% | 65.0% | 6.0% | — |"
    ) in table
    assert "| cg_project | 88.0% [82.0%, 93.0%] | 85.0% [79.0%, 90.0%]" in table
    assert "87.0%" in table


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


def test_inject_section_preserves_backslashes(tmp_path: Path) -> None:
    doc = tmp_path / "README.md"
    doc.write_text(
        "<!-- RESULTS:PARETO_PLOT_START -->\nold\n<!-- RESULTS:PARETO_PLOT_END -->",
        encoding="utf-8",
    )

    path = r"C:\Users\example\results\pareto_frontier.png"
    status = inject_all(doc, {"pareto_plot": f"![Plot]({path})"})
    assert status["pareto_plot"] is True
    assert path in doc.read_text(encoding="utf-8")


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
    doc_path = tmp_path / "README.md"
    doc_path.write_text(
        "<!-- RESULTS:MAIN_TABLE_START -->\n<!-- RESULTS:MAIN_TABLE_END -->",
        encoding="utf-8",
    )

    result = run_report(
        repo_root=tmp_path,
        results_dir=tmp_path / "results",
        output_dir=tmp_path / "artifacts" / "report",
        readme_path=doc_path,
        results_out=tmp_path / "results" / "s9" / "report-test",
    )
    out_dir = tmp_path / "artifacts" / "report"
    assert Path(result["output_dir"]).exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "manifest.json").exists()
    assert result["status"] == "partial"
    assert result["injected"]["main_table"] is True
    assert (tmp_path / "results" / "s9" / "report-test" / "summary.json").exists()


def test_run_report_reads_real_artifact_tree(tmp_path: Path) -> None:
    checkpoint = (
        tmp_path / "artifacts" / "graft" / "cg_project-test" / "model.safetensors"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    _write_json(
        checkpoint.parent / "graft_stats.json",
        {"edit_footprint": 0.0123, "changed_parameters": 12345},
    )
    eval_path = _write_json(
        tmp_path / "artifacts" / "eval" / "eval-test" / "summary.json",
        _real_eval_summary(checkpoint),
    )

    overlap_dir = tmp_path / "artifacts" / "discover" / "discover-test"
    _write_json(
        overlap_dir / "overlap.json",
        {
            "method": "discovery_sparsity",
            "k_s": 0.05,
            "k_d": 0.05,
            "safety": {"count": 2, "components": ["L1.H0", "L2.MLP"]},
            "domain": {"count": 2, "components": ["L2.MLP", "L3.H1"]},
            "overlap": {
                "count": 1,
                "components": ["L2.MLP"],
                "jaccard": 1.0 / 3.0,
            },
        },
    )
    _write_json(overlap_dir / "safety" / "scores.json", {"L1.H0": 0.9, "L2.MLP": 0.7})
    _write_json(overlap_dir / "domain" / "scores.json", {"L2.MLP": 0.8, "L3.H1": 0.6})

    attack_path = _write_json(
        tmp_path
        / "artifacts"
        / "attacks"
        / "abliterate"
        / "abliterate-cg_project-test"
        / "attack_stats.json",
        {
            "status": "complete",
            "stage": "attacks",
            "attack_type": "abliterate",
            "model": "cg_project",
            "mode": "weight_orthogonalize",
            "layer_idx": 12,
        },
    )
    _write_json(
        attack_path.parent / "post_eval.json",
        {
            "status": "complete",
            "attack_type": "abliterate",
            "model": "cg_project",
            "eval_run_id": "eval-attack-test",
            "metrics": {"hrr": 0.7, "frr": 0.05},
        },
    )

    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Test report",
                "<!-- RESULTS:MAIN_TABLE_START -->",
                "old main",
                "<!-- RESULTS:MAIN_TABLE_END -->",
                "<!-- RESULTS:ATTACK_TABLE_START -->",
                "old attacks",
                "<!-- RESULTS:ATTACK_TABLE_END -->",
                "<!-- RESULTS:OVERLAP_TABLE_START -->",
                "old overlap",
                "<!-- RESULTS:OVERLAP_TABLE_END -->",
                "<!-- RESULTS:PARETO_PLOT_START -->",
                "old plot",
                "<!-- RESULTS:PARETO_PLOT_END -->",
            ]
        ),
        encoding="utf-8",
    )

    tracked_dir = tmp_path / "results" / "s9" / "report-test"
    result = run_report(
        repo_root=tmp_path,
        results_dir=tmp_path / "results",
        output_dir=tmp_path / "artifacts" / "report",
        readme_path=readme,
        results_out=tracked_dir,
    )

    assert result["status"] == "complete", result["partial_reasons"]
    assert result["injected"] == {
        "main_table": True,
        "attack_table": True,
        "overlap_table": True,
        "pareto_plot": True,
    }

    summary = json.loads((tracked_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["sources"] == {
        "eval": str(eval_path.relative_to(tmp_path)).replace("\\", "/"),
        "overlap": str(
            (overlap_dir / "overlap.json").relative_to(tmp_path)
        ).replace("\\", "/"),
        "attacks": [str(attack_path.relative_to(tmp_path)).replace("\\", "/")],
    }
    assert [row["model"] for row in summary["models"]] == [
        "aligned",
        "drifted",
        "cg_project",
    ]
    assert summary["models"][2]["srr"] == 0.88
    assert summary["attacks"][0]["clean_hrr"] == 0.85
    assert summary["attacks"][0]["abliteration_hrr"] == 0.7
    assert len(summary["attack_post_eval"]) == 1
    assert summary["overlap"]["jaccard"] == 1.0 / 3.0
    assert summary["gate_g_drift"] == {"status": "pass", "drop": 0.21}

    main_table = (tracked_dir / "table_main.md").read_text(encoding="utf-8")
    assert "1.23% (12,345 params)" in main_table
    assert "88.0% [82.0%, 93.0%]" in main_table

    attack_table = (tracked_dir / "table_attacks.md").read_text(encoding="utf-8")
    assert "85.0%" in attack_table
    assert "70.0%" in attack_table

    overlap_table = (tracked_dir / "table_overlap.md").read_text(encoding="utf-8")
    assert "| Component Count | 2 | 2 | 1 |" in overlap_table

    updated = readme.read_text(encoding="utf-8")
    assert "old main" not in updated
    assert "88.0% [82.0%, 93.0%]" in updated
    assert "old attacks" not in updated
    assert "old overlap" not in updated
    assert "results/s9/report-test/pareto_frontier.png" in updated
    assert (tracked_dir / "pareto_frontier.png").exists()

    manifest = json.loads((tracked_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "report"
    assert manifest["status"] == "complete"
