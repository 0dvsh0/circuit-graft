"""Table generation for CircuitGraft reports (S9).

Converts stage outputs (evaluation JSON, attack JSON, circuit overlap JSON)
into clean GitHub-flavored markdown tables for insertion into README.md,
docs/ARTICLE.md, and evaluation reports.
"""

from __future__ import annotations

from typing import Any


def _fmt(val: Any, pct: bool = False, digits: int = 1) -> str:
    if val is None:
        return "—"
    try:
        num = float(val)
        if pct:
            num = num * 100.0 if abs(num) <= 1.0 and num != 0.0 else num
            return f"{num:.{digits}f}%"
        return f"{num:.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def _fmt_ci(val: Any, ci: tuple[float, float] | list[float] | None = None, pct: bool = False) -> str:
    main_val = _fmt(val, pct=pct)
    if not ci or len(ci) < 2 or ci[0] is None or ci[1] is None:
        return main_val
    low = _fmt(ci[0], pct=pct)
    high = _fmt(ci[1], pct=pct)
    return f"{main_val} [{low}, {high}]"


def generate_main_results_table(
    eval_matrix: list[dict[str, Any]] | dict[str, Any],
) -> str:
    """Generate the primary safety recovery vs domain retention markdown table.

    Rows correspond to evaluated models (theta_B, theta_A, theta_D, baselines, CG variants).
    Columns:
    - Model / Method
    - SRR (%)
    - DGR (%)
    - HarmBench HRR
    - AdvBench HRR
    - MedMCQA Acc
    - MMLU (Non-med) Acc
    - XSTest FRR
    - Footprint (Heads / MLPs)
    """
    rows = eval_matrix if isinstance(eval_matrix, list) else eval_matrix.get("models", [])
    if not rows and isinstance(eval_matrix, dict):
        # Maybe keyed by model name
        rows = [{"model": k, **v} for k, v in eval_matrix.items() if isinstance(v, dict)]

    headers = [
        "Model / Method",
        "SRR",
        "DGR",
        "HarmBench HRR",
        "AdvBench HRR",
        "MedMCQA Acc",
        "MMLU Non-med Acc",
        "XSTest FRR",
        "Footprint",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        name = row.get("name") or row.get("model") or row.get("method", "Unknown")
        # Ensure canonical labeling for C-Theta baseline per SPEC §2.1
        if "ctheta" in str(name).lower() and "weight-arithmetic" not in str(name):
            name = f"{name} (SafeTune weight-arithmetic implementation)"

        srr = _fmt_ci(row.get("srr"), row.get("srr_ci"), pct=True)
        dgr = _fmt_ci(row.get("dgr"), row.get("dgr_ci"), pct=True)

        hb_hrr = _fmt(row.get("harmbench_hrr", row.get("harmbench", {}).get("hrr")), pct=True)
        adv_hrr = _fmt(row.get("advbench_hrr", row.get("advbench", {}).get("hrr")), pct=True)
        med_acc = _fmt(row.get("medmcqa_acc", row.get("medmcqa", {}).get("acc")), pct=True)
        mmlu_acc = _fmt(row.get("mmlu_acc", row.get("mmlu", {}).get("acc")), pct=True)
        xs_frr = _fmt(row.get("xstest_frr", row.get("xstest", {}).get("frr")), pct=True)

        footprint = row.get("footprint", row.get("num_edited_components", "—"))

        lines.append(
            f"| {name} | {srr} | {dgr} | {hb_hrr} | {adv_hrr} | {med_acc} | {mmlu_acc} | {xs_frr} | {footprint} |"
        )

    return "\n".join(lines)


def generate_attack_robustness_table(
    attack_matrix: list[dict[str, Any]] | dict[str, Any],
) -> str:
    """Generate the attack robustness comparison markdown table.

    Evaluates models under:
    - Clean HarmBench HRR
    - Abliteration HRR
    - Harmful FT decay (50, 100, 200 steps)
    - Uniform Quantization (NF4, int8)
    - Circuit-aware Quantization (4-bit)
    """
    rows = attack_matrix if isinstance(attack_matrix, list) else attack_matrix.get("models", [])
    if not rows and isinstance(attack_matrix, dict):
        rows = [{"model": k, **v} for k, v in attack_matrix.items() if isinstance(v, dict)]

    headers = [
        "Model / Method",
        "Clean HRR",
        "Abliteration HRR",
        "Harmful-FT (50 st)",
        "Harmful-FT (100 st)",
        "Harmful-FT (200 st)",
        "Quant NF4 HRR",
        "Circuit-Quant 4b HRR",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        name = row.get("name") or row.get("model") or row.get("method", "Unknown")
        if "ctheta" in str(name).lower() and "weight-arithmetic" not in str(name):
            name = f"{name} (SafeTune weight-arithmetic implementation)"

        clean = _fmt(row.get("clean_hrr"), pct=True)
        abl = _fmt(row.get("abliteration_hrr", row.get("abliterate", {}).get("hrr")), pct=True)

        ft = row.get("harmful_ft", {})
        ft_50 = _fmt(ft.get(50, ft.get("step_50", row.get("harmful_ft_50"))), pct=True)
        ft_100 = _fmt(ft.get(100, ft.get("step_100", row.get("harmful_ft_100"))), pct=True)
        ft_200 = _fmt(ft.get(200, ft.get("step_200", row.get("harmful_ft_200"))), pct=True)

        q_nf4 = _fmt(row.get("quant_nf4_hrr", row.get("quantization", {}).get("nf4", {}).get("hrr")), pct=True)
        q_cq = _fmt(row.get("circuit_quant_hrr", row.get("quantization", {}).get("circuit_aware", {}).get("hrr")), pct=True)

        lines.append(
            f"| {name} | {clean} | {abl} | {ft_50} | {ft_100} | {ft_200} | {q_nf4} | {q_cq} |"
        )

    return "\n".join(lines)


def generate_overlap_table(
    overlap_report: dict[str, Any],
) -> str:
    """Generate the circuit overlap summary markdown table."""
    s_size = overlap_report.get("safety_circuit_size", len(overlap_report.get("safety_nodes", [])))
    d_size = overlap_report.get("domain_circuit_size", len(overlap_report.get("domain_nodes", [])))
    overlap_nodes = overlap_report.get("overlap_nodes", [])
    overlap_count = overlap_report.get("overlap_size", len(overlap_nodes))
    jaccard_val = overlap_report.get("jaccard", 0.0)

    heads_overlap = sum(1 for n in overlap_nodes if ".H" in str(n))
    mlps_overlap = sum(1 for n in overlap_nodes if ".MLP" in str(n))

    lines = [
        "| Metric | Safety Circuit (S) | Domain Circuit (D) | Overlap (S ∩ D) |",
        "|---|---|---|---|",
        f"| Component Count | {s_size} | {d_size} | {overlap_count} |",
        f"| Attention Heads | — | — | {heads_overlap} |",
        f"| MLP Blocks | — | — | {mlps_overlap} |",
        f"| Jaccard Similarity | — | — | {_fmt(jaccard_val, digits=3)} |",
    ]
    return "\n".join(lines)
