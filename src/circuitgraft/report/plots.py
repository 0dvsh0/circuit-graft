"""Plot generation for CircuitGraft reports (S9).

Produces publication-quality plots:
- Pareto frontier (SRR vs DGR)
- Circuit overlap layer-component heatmap
- Attack decay curve (Harmful-FT steps vs HRR)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _can_plot() -> bool:
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401

        return True
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plot generation.")
        return False


def plot_pareto_frontier(
    eval_matrix: list[dict[str, Any]] | dict[str, Any],
    output_path: str | Path,
) -> Path | None:
    """Generate the SRR vs DGR Pareto scatter plot."""
    if not _can_plot():
        return None

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = eval_matrix if isinstance(eval_matrix, list) else eval_matrix.get("models", [])
    if not rows and isinstance(eval_matrix, dict):
        rows = [{"model": k, **v} for k, v in eval_matrix.items() if isinstance(v, dict)]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Style groups
    colors = {
        "cg_project": "#1f77b4",
        "cg_uniform": "#2ca02c",
        "cg_exclude": "#ff7f0e",
        "cg_fisher": "#9467bd",
        "cg_random": "#7f7f7f",
        "baseline": "#d62728",
        "anchor": "#333333",
    }

    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []

    for row in rows:
        name = str(row.get("name") or row.get("model") or row.get("method", "Unknown"))
        srr = row.get("srr")
        dgr = row.get("dgr")
        if srr is None or dgr is None:
            continue

        try:
            srr_val = float(srr) * 100.0 if abs(float(srr)) <= 1.0 else float(srr)
            dgr_val = float(dgr) * 100.0 if abs(float(dgr)) <= 1.0 else float(dgr)
        except (ValueError, TypeError):
            continue

        group = "baseline"
        if "project" in name.lower():
            group = "cg_project"
        elif "uniform" in name.lower():
            group = "cg_uniform"
        elif "exclude" in name.lower():
            group = "cg_exclude"
        elif "fisher" in name.lower():
            group = "cg_fisher"
        elif "random" in name.lower():
            group = "cg_random"
        elif name in ("theta_A", "theta_D", "theta_B", "base", "aligned", "drifted"):
            group = "anchor"

        color = colors.get(group, "#333333")
        marker = "o" if group != "anchor" else "s"

        ax.scatter(dgr_val, srr_val, color=color, s=80, marker=marker, zorder=5, label=name)
        ax.annotate(
            name,
            (dgr_val, srr_val),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            alpha=0.85,
        )
        xs.append(dgr_val)
        ys.append(srr_val)
        labels.append(name)

    ax.set_title("Pareto Frontier: Safety Recovery vs Domain Gain Retention", fontsize=12, pad=10)
    ax.set_xlabel("Domain Gain Retention (DGR %)", fontsize=10)
    ax.set_ylabel("Safety Recovery Ratio (SRR %)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Reference lines at 0% and 100%
    ax.axhline(100.0, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(100.0, color="gray", linestyle=":", alpha=0.6)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_overlap_heatmap(
    safety_scores: dict[str, float],
    domain_scores: dict[str, float],
    output_path: str | Path,
    n_layers: int = 36,
    n_heads: int = 16,
) -> Path | None:
    """Generate a layer-by-component heatmap of circuit overlap."""
    if not _can_plot():
        return None

    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Matrix: rows = layers (0..35), cols = heads (0..15) + MLP (16)
    # Value code: 0 = none, 1 = domain only, 2 = safety only, 3 = both (overlap)
    matrix = np.zeros((n_layers, n_heads + 1), dtype=int)

    all_keys = set(safety_scores) | set(domain_scores)
    for key in all_keys:
        in_s = key in safety_scores and safety_scores[key] > 0
        in_d = key in domain_scores and domain_scores[key] > 0
        if not (in_s or in_d):
            continue

        parts = key.split(".")
        if len(parts) != 2:
            continue
        try:
            layer = int(parts[0][1:])
        except ValueError:
            continue

        if parts[1].startswith("H"):
            try:
                head = int(parts[1][1:])
                col = head
            except ValueError:
                continue
        elif parts[1] == "MLP":
            col = n_heads
        else:
            continue

        if 0 <= layer < n_layers and 0 <= col <= n_heads:
            if in_s and in_d:
                matrix[layer, col] = 3
            elif in_s:
                matrix[layer, col] = 2
            elif in_d:
                matrix[layer, col] = 1

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Custom colormap: 0: white/lightgray, 1: blue (domain), 2: red (safety), 3: purple (overlap)
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#f7f7f7", "#6baed6", "#fc9272", "#9e9ac8"])

    ax.imshow(matrix, aspect="auto", cmap=cmap, origin="lower")

    ax.set_title("Safety vs Domain Circuit Distribution (Cohere2 Architecture)", fontsize=12, pad=10)
    ax.set_xlabel("Component (Heads 0–15, MLP)", fontsize=10)
    ax.set_ylabel("Layer (0–35)", fontsize=10)

    # Ticks
    col_labels = [f"H{h}" for h in range(n_heads)] + ["MLP"]
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=90)
    ax.set_yticks(range(0, n_layers, 2))
    ax.set_yticklabels(range(0, n_layers, 2), fontsize=8)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#f7f7f7", edgecolor="gray", label="Neither"),
        Patch(facecolor="#6baed6", label="Domain only"),
        Patch(facecolor="#fc9272", label="Safety only"),
        Patch(facecolor="#9e9ac8", label="Shared (Overlap)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.25, 1.0))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_attack_decay(
    attack_results: dict[str, Any],
    output_path: str | Path,
) -> Path | None:
    """Generate Harmful-FT steps vs HRR decay curve."""
    if not _can_plot():
        return None

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    steps = [0, 50, 100, 200]
    models = attack_results.get("harmful_ft_decay", {})

    for model_name, curve in models.items():
        hrrs = [curve.get(f"step_{s}", curve.get(s, 0.0)) for s in steps]
        hrrs_pct = [h * 100.0 if abs(h) <= 1.0 else h for h in hrrs]
        ax.plot(steps, hrrs_pct, marker="o", linewidth=2, label=model_name)

    ax.set_title("Harmful Re-fine-tuning Attack: Refusal Decay", fontsize=11, pad=10)
    ax.set_xlabel("Adversarial Steps", fontsize=10)
    ax.set_ylabel("HarmBench Refusal Rate (HRR %)", fontsize=10)
    ax.set_xticks(steps)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path
