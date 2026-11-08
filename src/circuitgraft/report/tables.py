"""Table generation for CircuitGraft reports (S9).

Converts stage outputs (evaluation JSON, attack JSON, circuit overlap JSON)
into clean GitHub-flavored markdown tables for insertion into README.md,
docs/ARTICLE.md, and evaluation reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from circuitgraft.registry import REPO_ROOT


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


def _bench_metrics(
    document: dict[str, Any],
    group: str,
    benchmark: str,
) -> dict[str, Any]:
    """Return the metrics of one judged benchmark.

    S7 writes the benchmark document flat (``{"hrr": ..., "hrr_ci": ...}`` or
    ``{"accuracy": ..., "accuracy_ci": ...}``). Older fixtures wrapped the
    metrics under a ``metrics`` key, so accept that shape too.
    """
    judged = document.get("judged")
    if not isinstance(judged, dict):
        return {}
    group_doc = judged.get(group)
    if not isinstance(group_doc, dict):
        return {}
    benchmark_doc = group_doc.get(benchmark)
    if not isinstance(benchmark_doc, dict):
        return {}
    metrics = benchmark_doc.get("metrics")
    if isinstance(metrics, dict):
        return {**benchmark_doc, **metrics}
    return benchmark_doc


def _checkpoint_footprint(document: dict[str, Any]) -> str | None:
    """Find the edit footprint of the checkpoint behind an eval row."""
    checkpoint = document.get("checkpoint")
    if not checkpoint:
        return None
    path = Path(str(checkpoint))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.is_file():
        path = path.parent
    if not path.exists():
        return None

    graft_stats = path / "graft_stats.json"
    if graft_stats.exists():
        try:
            stats = json.loads(graft_stats.read_text(encoding="utf-8"))
            footprint = stats.get("edit_footprint")
            changed = stats.get("changed_parameters")
            if footprint is not None:
                label = f"{float(footprint) * 100:.2f}%"
                if changed is not None:
                    label += f" ({int(changed):,} params)"
                return label
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    baseline_stats = path / "baseline_stats.json"
    if baseline_stats.exists():
        try:
            stats = json.loads(baseline_stats.read_text(encoding="utf-8"))
            circuit = stats.get("circuit")
            if isinstance(circuit, dict) and circuit.get("footprint") is not None:
                return f"{float(circuit['footprint']) * 100:.2f}%"
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return None


def normalise_eval_rows(
    eval_matrix: list[dict[str, Any]] | dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten the real S7 summary shape into one row per model.

    S7 writes::

        {"models": {"aligned": {"metrics": ..., "composite": ...,
                                "judged": {"safety": {...}}}, ...}}

    The table/plot code also accepts the older flat row format used by the
    unit tests, so both shapes are supported.
    """
    if isinstance(eval_matrix, list):
        return [row for row in eval_matrix if isinstance(row, dict)]

    if not isinstance(eval_matrix, dict):
        return []

    models = eval_matrix.get("models")
    if isinstance(models, list):
        return [row for row in models if isinstance(row, dict)]
    if not isinstance(models, dict):
        return [
            {"model": key, **value}
            for key, value in eval_matrix.items()
            if isinstance(value, dict)
        ]

    rows: list[dict[str, Any]] = []
    for name, document in models.items():
        if not isinstance(document, dict):
            continue
        composite = document.get("composite")
        composite = composite if isinstance(composite, dict) else {}
        metrics = document.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}

        harmbench = _bench_metrics(document, "safety", "harmbench")
        advbench = _bench_metrics(document, "safety", "advbench")
        medmcqa = _bench_metrics(document, "mcq", "medmcqa_validation")
        mmlu = _bench_metrics(document, "mcq", "mmlu_non_medical")
        xstest = _bench_metrics(document, "benign", "xstest_safe")

        rows.append(
            {
                "model": name,
                "status": document.get("status"),
                "srr": composite.get("srr"),
                "srr_ci": composite.get("srr_ci"),
                "dgr": composite.get("dgr"),
                "dgr_ci": composite.get("dgr_ci"),
                "orp": composite.get("orp"),
                "harmbench_hrr": harmbench.get("hrr"),
                "harmbench_hrr_ci": harmbench.get("hrr_ci"),
                "advbench_hrr": advbench.get("hrr"),
                "advbench_hrr_ci": advbench.get("hrr_ci"),
                "medmcqa_acc": medmcqa.get("accuracy"),
                "medmcqa_acc_ci": medmcqa.get("accuracy_ci"),
                "mmlu_acc": mmlu.get("accuracy"),
                "mmlu_acc_ci": mmlu.get("accuracy_ci"),
                "xstest_frr": xstest.get("frr"),
                "xstest_frr_ci": xstest.get("frr_ci"),
                # Pooled fallbacks are useful for a partial matrix.
                "hrr": metrics.get("hrr"),
                "frr": metrics.get("frr"),
                "domain_accuracy": metrics.get("domain_accuracy"),
                "general_accuracy": metrics.get("general_accuracy"),
                "footprint": _checkpoint_footprint(document),
            }
        )
    return rows


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
    rows = normalise_eval_rows(eval_matrix)

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

        hb_hrr = _fmt(
            row.get("harmbench_hrr")
            if row.get("harmbench_hrr") is not None
            else row.get("harmbench", {}).get("hrr"),
            pct=True,
        )
        adv_hrr = _fmt(
            row.get("advbench_hrr")
            if row.get("advbench_hrr") is not None
            else row.get("advbench", {}).get("hrr"),
            pct=True,
        )
        med_acc = _fmt(
            row.get("medmcqa_acc")
            if row.get("medmcqa_acc") is not None
            else row.get("medmcqa", {}).get("acc"),
            pct=True,
        )
        mmlu_acc = _fmt(
            row.get("mmlu_acc")
            if row.get("mmlu_acc") is not None
            else row.get("mmlu", {}).get("acc"),
            pct=True,
        )
        xs_frr = _fmt(
            row.get("xstest_frr")
            if row.get("xstest_frr") is not None
            else row.get("xstest", {}).get("frr"),
            pct=True,
        )

        footprint = row.get("footprint") or row.get("num_edited_components") or "—"

        lines.append(
            f"| {name} | {srr} | {dgr} | {hb_hrr} | {adv_hrr} | {med_acc} | {mmlu_acc} | {xs_frr} | {footprint} |"
        )

    return "\n".join(lines)


def normalise_attack_rows(
    attack_matrix: list[dict[str, Any]] | dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten an attack summary into one row per attacked model."""
    if isinstance(attack_matrix, list):
        return [row for row in attack_matrix if isinstance(row, dict)]
    if not isinstance(attack_matrix, dict):
        return []
    models = attack_matrix.get("models")
    if isinstance(models, list):
        return [row for row in models if isinstance(row, dict)]
    if isinstance(models, dict):
        return [
            {"model": name, **document}
            for name, document in models.items()
            if isinstance(document, dict)
        ]
    return [
        {"model": name, **document}
        for name, document in attack_matrix.items()
        if isinstance(document, dict)
    ]


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
    rows = normalise_attack_rows(attack_matrix)

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
        abl = _fmt(
            row.get("abliteration_hrr")
            if row.get("abliteration_hrr") is not None
            else row.get("abliterate", {}).get("hrr"),
            pct=True,
        )

        ft = row.get("harmful_ft") or {}
        if not isinstance(ft, dict):
            ft = {}

        def _step(step: int, values: dict[Any, Any], source_row: dict[str, Any]) -> Any:
            value = values.get(step, values.get(str(step), values.get(f"step_{step}")))
            if value is None:
                value = source_row.get(f"harmful_ft_{step}")
            return value

        ft_50 = _fmt(_step(50, ft, row), pct=True)
        ft_100 = _fmt(_step(100, ft, row), pct=True)
        ft_200 = _fmt(_step(200, ft, row), pct=True)

        quantization = row.get("quantization") or {}
        if not isinstance(quantization, dict):
            quantization = {}
        q_nf4 = _fmt(
            row.get("quant_nf4_hrr")
            if row.get("quant_nf4_hrr") is not None
            else quantization.get("nf4", {}).get("hrr"),
            pct=True,
        )
        q_cq = _fmt(
            row.get("circuit_quant_hrr")
            if row.get("circuit_quant_hrr") is not None
            else quantization.get("circuit_aware", {}).get("hrr"),
            pct=True,
        )

        lines.append(
            f"| {name} | {clean} | {abl} | {ft_50} | {ft_100} | {ft_200} | {q_nf4} | {q_cq} |"
        )

    return "\n".join(lines)


def normalise_overlap_report(overlap_report: dict[str, Any]) -> dict[str, Any]:
    """Accept both the S4 overlap report and the older flat table fixture."""
    if not isinstance(overlap_report, dict):
        return {}
    safety = overlap_report.get("safety")
    domain = overlap_report.get("domain")
    overlap = overlap_report.get("overlap")
    if isinstance(safety, dict) and isinstance(domain, dict) and isinstance(overlap, dict):
        return {
            "safety_circuit_size": safety.get("count", 0),
            "domain_circuit_size": domain.get("count", 0),
            "overlap_nodes": overlap.get("components", []),
            "overlap_size": overlap.get("count", 0),
            "jaccard": overlap.get("jaccard", 0.0),
        }
    return overlap_report


def generate_overlap_table(
    overlap_report: dict[str, Any],
) -> str:
    """Generate the circuit overlap summary markdown table."""
    overlap_report = normalise_overlap_report(overlap_report)
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
