"""CircuitGraft reporting and analysis package (S9)."""

from circuitgraft.report.inject_readme import inject_all, inject_section
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

__all__ = [
    "generate_attack_robustness_table",
    "generate_main_results_table",
    "generate_overlap_table",
    "inject_all",
    "inject_section",
    "plot_attack_decay",
    "plot_overlap_heatmap",
    "plot_pareto_frontier",
    "run_report",
]
