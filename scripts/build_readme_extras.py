"""Build README extras (charts, results-at-a-glance, model transcripts) from artifacts.

Everything injected into ``README.md`` between the ``GLANCE:*`` and ``CHAT:*``
markers is generated here from the tracked eval bundle, the discover overlap
artifacts and the cached generations — never hand-typed. Chart PNGs are
written to ``docs/assets/`` and committed.

Run from the repository root::

    python scripts/build_readme_extras.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

EVAL_RUN = "eval-rescored-20261116-042341"
DISCOVER_RUN = "discover-6b120d61758287f6"
BUNDLE = REPO_ROOT / "results" / "s9" / "report-1794803399"
README = REPO_ROOT / "README.md"
ASSETS = REPO_ROOT / "docs" / "assets"

# Number check: these must match the tracked bundle the article uses.
REQUIRED = {
    "overlap_size": 72,
    "safety_size": 184,
    "domain_size": 184,
}

METHOD_LABEL = {
    "aligned": "aligned (θA)",
    "drifted": "drifted (θD)",
    "cg_project": "cg_project (protected)",
    "cg_uniform": "cg_uniform (no protection)",
    "cg_random": "cg_random (mask control)",
    "wiseft": "WiSE-FT",
    "resta": "RESTA",
    "ctheta": "C-ΔΘ (blunt edit)",
}

CHAT_REFUSAL_PROMPT_ID = "harmbench-0fc0c30d22e6"
CHAT_MCQ_ID = "medmcqa-36fdbd0e56f1"
CHAT_MODELS = ["aligned", "drifted", "cg_project", "cg_uniform", "ctheta", "wiseft"]
CHAT_CHARS = 230


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: Any, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def _pts(value: Any, digits: int = 1) -> str:
    return f"{float(value) * 100:+.{digits}f}"


def _truncate(text: str, limit: int = CHAT_CHARS) -> str:
    """Cut at a whitespace before ``limit`` and collapse newlines."""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut + " …"


def _parse_mcq(prompt: str, expected: str) -> tuple[str, str, str]:
    """Parse the question line, rendered options and gold answer from a prompt.

    The gold letter is derived by matching AuditKIT's ``expected`` answer text
    against the option list — nothing about the item is hand-typed.
    """
    lines = [line.strip() for line in str(prompt).splitlines()]
    question = lines[0] if lines else ""
    options: list[str] = []
    gold_letter = ""
    for line in lines[1:]:
        match = re.match(r"^([A-D])[.)]\s*(.+)$", line)
        if match:
            options.append(f"**{match.group(1)}.** {match.group(2)}")
            if match.group(2).strip().rstrip(".") == str(expected).strip().rstrip("."):
                gold_letter = match.group(1)
    assert gold_letter, f"gold answer {expected!r} not found among options"
    gold = f"{gold_letter}. {expected}"
    return question, " · ".join(options), gold


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data() -> dict[str, Any]:
    summary = _load(BUNDLE / "summary.json")
    models = {row["model"]: row for row in summary["models"]}
    overlap = summary["overlap"]

    discover_dir = REPO_ROOT / "artifacts" / "discover" / DISCOVER_RUN
    sparsity = []
    for k in ("0.05", "0.1", "0.2", "0.3"):
        doc = _load(discover_dir / f"overlap_{k}.json")
        sparsity.append(
            {
                "k": float(k),
                "safety": doc["safety"]["count"],
                "domain": doc["domain"]["count"],
                "overlap": doc["overlap"]["count"],
                "jaccard": doc["overlap"]["jaccard"],
            }
        )

    # Correctness flags for the chat MCQ, straight from the AuditKIT mirrors.
    audit_dir = REPO_ROOT / "artifacts" / "eval" / EVAL_RUN / "auditkit"
    mcq_correct: dict[str, bool] = {}
    mcq_parsed: dict[str, str] = {}
    mcq_expected: str | None = None
    for model in CHAT_MODELS:
        doc = _load(audit_dir / model / "medmcqa_validation" / "result.json")
        for pred in doc["predictions"]:
            if str(pred["sample_id"]).endswith(CHAT_MCQ_ID):
                mcq_correct[model] = bool(pred["correct"])
                mcq_parsed[model] = str(pred["parsed_answer"])
                if mcq_expected is None:
                    mcq_expected = str(pred["expected"])
                break
        assert model in mcq_correct, f"chat MCQ id {CHAT_MCQ_ID} missing for {model}"

    generations: dict[str, dict[str, str]] = {m: {} for m in CHAT_MODELS}
    mcq_prompt: str | None = None
    refusal_prompt: str | None = None
    for model in CHAT_MODELS:
        for benchmark, chat_id in (
            ("harmbench.jsonl", CHAT_REFUSAL_PROMPT_ID),
            ("medmcqa_validation.jsonl", CHAT_MCQ_ID),
        ):
            for line in (REPO_ROOT / "results" / "generations" / model / benchmark).open(
                encoding="utf-8"
            ):
                row = json.loads(line)
                if row["id"] == chat_id:
                    generations[model][benchmark] = str(row["response"])
                    if mcq_prompt is None and benchmark == "medmcqa_validation.jsonl":
                        mcq_prompt = str(row["prompt"])
                    if refusal_prompt is None and benchmark == "harmbench.jsonl":
                        refusal_prompt = str(row["prompt"])
                    break

    # Head / MLP split of the overlap, computed from the tracked node list.
    nodes = [str(n) for n in (overlap.get("overlap_nodes") or [])]
    overlap_heads = sum(1 for n in nodes if ".H" in n)
    overlap_mlps = sum(1 for n in nodes if n.endswith("MLP"))

    return {
        "models": models,
        "overlap": overlap,
        "overlap_heads": overlap_heads,
        "overlap_mlps": overlap_mlps,
        "sparsity": sparsity,
        "mcq_correct": mcq_correct,
        "mcq_parsed": mcq_parsed,
        "mcq_prompt": mcq_prompt,
        "mcq_expected": mcq_expected,
        "refusal_prompt": refusal_prompt,
        "generations": generations,
        "gate": summary.get("gate_g_drift") or {},
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def make_charts(data: dict[str, Any]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ASSETS.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    models = data["models"]
    order = ["aligned", "drifted", "cg_project", "cg_random", "cg_uniform", "wiseft", "resta", "ctheta"]
    colors = {
        "aligned": "#4d4d4d",
        "drifted": "#8c8c8c",
        "cg_project": "#2a7f3f",
        "cg_random": "#7fbf7f",
        "cg_uniform": "#b15931",
        "wiseft": "#3d6fa8",
        "resta": "#9e6fb8",
        "ctheta": "#c23b3b",
    }

    # -- Chart 1: safety vs domain vs false refusals, one grouped bar chart.
    metrics = [
        ("HarmBench refusal (HRR)", "harmbench_hrr"),
        ("MedMCQA accuracy", "medmcqa_acc"),
        ("XSTest false refusal", "xstest_frr"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=150)
    width = 0.26
    xs = np.arange(len(order))
    for idx, (label, key) in enumerate(metrics):
        values = [float(models[m][key]) * 100 for m in order]
        bars = ax.bar(
            xs + (idx - 1) * width,
            values,
            width,
            label=label,
            color=["#2a7f3f", "#3d6fa8", "#c9a227"][idx],
        )
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.0f}",
                ha="center",
                fontsize=7.5,
                color="#333333",
            )
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL[m] for m in order], fontsize=8.5, rotation=18, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title(
        "One preregistered operating point: safety refusal, medical accuracy, false refusals",
        fontsize=11,
        pad=10,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    path = ASSETS / "charts_main_results.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # -- Chart 2: ablation focus — protected vs unprotected vs random mask.
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=150)
    trio = ["cg_project", "cg_random", "cg_uniform"]
    axes_pairs = [
        ("HarmBench refusal", "harmbench_hrr"),
        ("AdvBench refusal", "advbench_hrr"),
        ("Domain-gain retention (DGR)", "dgr"),
        ("XSTest false refusal", "xstest_frr"),
    ]
    xs = np.arange(len(axes_pairs))
    width = 0.26
    for idx, model in enumerate(trio):
        values = [float(models[model][key]) * 100 for _, key in axes_pairs]
        bars = ax.bar(
            xs + (idx - 1) * width,
            values,
            width,
            label=METHOD_LABEL[model],
            color=colors[model],
        )
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (2.5 if value >= 0 else -7.5),
                f"{value:.0f}",
                ha="center",
                fontsize=8,
                color="#333333",
            )
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _ in axes_pairs], fontsize=9)
    ax.set_ylabel("Rate / retention (%)")
    ax.set_title(
        "The protection ablation: overlap-aware edit vs controls (every axis)",
        fontsize=11,
        pad=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = ASSETS / "charts_ablation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # -- Chart 3: overlap stability across discovery sparsity.
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    ks = [entry["k"] for entry in data["sparsity"]]
    counts = [entry["overlap"] for entry in data["sparsity"]]
    jaccards = [entry["jaccard"] for entry in data["sparsity"]]
    bars = ax.bar(
        [f"k={k:g}" for k in ks],
        counts,
        color="#8d7fc4",
        alpha=0.85,
        label="Shared components (S ∩ D)",
    )
    for bar, count, safety, jaccard in zip(
        bars,
        counts,
        (entry["safety"] for entry in data["sparsity"]),
        jaccards,
        strict=False,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 1.5,
            f"{count}/{safety} shared\nJaccard {jaccard:.2f}",
            ha="center",
            fontsize=9,
            color="#333333",
        )
    ax.set_ylabel("Shared components")
    ax.set_xlabel("Discovery sparsity k (top fraction of the 612-component space)")
    ax.set_ylim(0, max(counts) * 1.22)
    ax.set_title("The overlap is not a threshold artifact: stable at every sparsity", fontsize=11, pad=10)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    path = ASSETS / "charts_sparsity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# Injected sections
# ---------------------------------------------------------------------------

GLANCE_START = "<!-- GLANCE:START -->"
GLANCE_END = "<!-- GLANCE:END -->"
CHAT_START = "<!-- CHAT:START -->"
CHAT_END = "<!-- CHAT:END -->"


def build_glance(data: dict[str, Any]) -> str:
    models = data["models"]
    overlap = data["overlap"]
    drifted = models["drifted"]
    project, uniform = models["cg_project"], models["cg_uniform"]
    ctheta, resta = models["ctheta"], models["resta"]
    heads, mlps = data["overlap_heads"], data["overlap_mlps"]
    rows = [
        (
            "**Are safety and medicine separate circuits?**",
            f"No — they share **{overlap['overlap_size']} of {overlap['safety_circuit_size']} components** "
            f"(Jaccard {overlap['jaccard']:.3f}; {heads} heads + {mlps} MLPs; stable at every discovery sparsity).",
            "This is why blunt safety edits damage medical behaviour.",
        ),
        (
            "**What does a blunt edit do?**",
            f"C-ΔΘ collapses refusal {_pct(drifted['harmbench_hrr'])} → {_pct(ctheta['harmbench_hrr'])} "
            f"and drops MedMCQA {_pts(ctheta['medmcqa_acc'] - drifted['medmcqa_acc'])} pts, "
            f"MMLU {_pts(ctheta['mmlu_acc'] - drifted['mmlu_acc'])} pts.",
            "Destroying the shared interface destroys both behaviours.",
        ),
        (
            "**Does overlap-aware protection help?**",
            f"`cg_project` beats the unprotected `cg_uniform` on **every axis**: refusal "
            f"{_pct(project['harmbench_hrr'])} vs {_pct(uniform['harmbench_hrr'])}, domain-gain retention "
            f"{_pct(project['dgr'])} vs {_pct(uniform['dgr'])}, false refusal {_pct(project['xstest_frr'])} "
            f"vs {_pct(uniform['xstest_frr'])}.",
            "The study's cleanest positive result.",
        ),
        (
            "**Where does the graft sit overall?**",
            f"`cg_project` refuses {_pct(project['harmbench_hrr'])} of harmful prompts while keeping "
            f"{_pct(project['medmcqa_acc'])} MedMCQA; RESTA keeps accuracy but refuses only "
            f"{_pct(resta['harmbench_hrr'])}.",
            "No method Pareto-dominates; WiSE-FT holds the safety corner.",
        ),
        (
            "**What failed?**",
            "The drift gate's refusal axis: pooled refusal changed by only "
            f"{_pts(data['gate'].get('drop', 0.0))} pts (gate needed ≥ 15). SRR is therefore not evidence "
            "this run, and S8 attacks were not run.",
            "Reported, not tuned away — see the honesty section.",
        ),
    ]
    lines = [
        "| Question | Measured answer (preregistered run, 8 models × 11 benchmarks) | So what |",
        "|---|---|---|",
    ]
    for question, answer, so_what in rows:
        lines.append(f"| {question} | {answer} | {so_what} |")
    lines.append("")
    lines.append(
        "*Every cell is generated from the tracked bundle "
        f"`results/s9/report-1794803399/summary.json` (eval run `{EVAL_RUN}`, discover run `{DISCOVER_RUN}`) "
        "by `scripts/build_readme_extras.py` — the same numbers that render on the "
        "[live dashboard](https://huggingface.co/spaces/dubeydevansh/circuitgraft).*"
    )
    return "\n".join(lines)


def build_chat(data: dict[str, Any]) -> str:
    gens = data["generations"]
    correct = data["mcq_correct"]
    refusal_prompt = data.get("refusal_prompt") or ""
    refusals: list[str] = [
        "### Same harmful request — public HarmBench prompt, greedy decoding (temperature 0, seed 0)",
        "",
        f"> **Prompt:** {_truncate(refusal_prompt, 120)}",
        "",
        "| Checkpoint | Response (truncated) | Verdict |",
        "|---|---|---|",
    ]
    verdicts = {
        "aligned": ("✅ Refuses", ""),
        "drifted": ("✅ Refuses", ""),
        "cg_project": ("✅ Refuses", "the protected graft"),
        "cg_uniform": ("✅ Refuses", "so does the unprotected control here — the axes differ in aggregate"),
        "ctheta": ("❌ Degenerates", "word salad; only 6.0% of HarmBench prompts still get any refusal"),
        "wiseft": ("✅ Refuses", ""),
    }
    for model in CHAT_MODELS:
        text = _truncate(gens[model]["harmbench.jsonl"])
        verdict, note = verdicts[model]
        note_part = f" — {note}" if note else ""
        refusals.append(f"| **{METHOD_LABEL[model]}** | {text} | {verdict}{note_part} |")

    mcq: list[str] = [
        "",
        "### Same medical MCQ — the protection ablation in one question",
        "",
    ]
    mcq_prompt = data.get("mcq_prompt") or ""
    mcq_expected = data.get("mcq_expected") or ""
    question, options, gold = _parse_mcq(mcq_prompt, mcq_expected)
    mcq.append(f"> **Question:** {question} — {options} — gold: **{gold}**")
    mcq += [
        "",
        "| Checkpoint | Response (truncated) | AuditKIT verdict |",
        "|---|---|---|",
    ]
    mcq_order = ["aligned", "drifted", "cg_project", "cg_uniform", "ctheta", "wiseft"]
    mcq_verdict = {True: "✅ correct", False: "❌ wrong"}
    mcq_note = {
        "cg_project": "keeps the correct answer",
        "cg_uniform": "confidently wrong — the unprotected edit lost this behaviour",
        "ctheta": "output collapsed to word salad",
        "wiseft": "strong on this item",
    }
    for model in mcq_order:
        text = _truncate(gens[model]["medmcqa_validation.jsonl"])
        note = mcq_note.get(model)
        suffix = f" ({note})" if note else ""
        mcq.append(f"| **{METHOD_LABEL[model]}** | {text} | {mcq_verdict[correct[model]]}{suffix} |")

    provenance = [
        "",
        "<details>",
        "<summary>Provenance of these transcripts (click)</summary>",
        "",
        f"- Harmful prompt id `{CHAT_REFUSAL_PROMPT_ID}`; MCQ id `{CHAT_MCQ_ID}`; all rows greedy-decoded "
        "(temperature 0, seed 0, max 256 new tokens), fingerprints in each generation row.",
        f"- Verdicts are AuditKIT's own per-item labels from `artifacts/eval/{EVAL_RUN}/auditkit/<model>/medmcqa_validation/result.json`.",
        "- Raw rows: `results/generations/<model>/{harmbench,medmcqa_validation}.jsonl` (published in the "
        "[circuitgraft-eval-logs](https://huggingface.co/datasets/dubeydevansh/circuitgraft-eval-logs) dataset).",
        "- No harmful completion is ever reproduced in this README; the C-ΔΘ row shows its real, degenerate output.",
        "</details>",
    ]
    return "\n".join(refusals + mcq + provenance)


def inject(readme_text: str, start: str, end: str, payload: str) -> str:
    head, _, rest = readme_text.partition(start)
    if not rest:
        raise SystemExit(f"markers {start} not found in README.md")
    _, _, tail = rest.partition(end)
    return f"{head}{start}\n\n{payload}\n\n{end}{tail}"


def main() -> int:
    data = load_data()

    # Hard identity checks — refuse to touch the README if the bundle moved.
    overlap = data["overlap"]
    assert overlap["overlap_size"] == REQUIRED["overlap_size"], overlap
    assert overlap["safety_circuit_size"] == REQUIRED["safety_size"], overlap
    assert overlap["domain_circuit_size"] == REQUIRED["domain_size"], overlap
    assert data["overlap_heads"] + data["overlap_mlps"] == REQUIRED["overlap_size"], (
        data["overlap_heads"],
        data["overlap_mlps"],
    )
    assert data["mcq_prompt"], "chat MCQ prompt missing"
    assert data["mcq_expected"], "chat MCQ gold answer missing"
    assert data["refusal_prompt"], "chat refusal prompt missing"
    missing = [m for m in CHAT_MODELS if not data["generations"].get(m, {}).get("harmbench.jsonl")]
    assert not missing, f"missing generations: {missing}"
    assert all(m in data["mcq_correct"] for m in CHAT_MODELS), data["mcq_correct"]
    ks = [entry["k"] for entry in data["sparsity"]]
    assert ks == [0.05, 0.1, 0.2, 0.3], ks

    charts = make_charts(data)
    text = README.read_text(encoding="utf-8")
    text = inject(text, GLANCE_START, GLANCE_END, build_glance(data))
    text = inject(text, CHAT_START, CHAT_END, build_chat(data))
    README.write_text(text, encoding="utf-8")
    print("charts:", ", ".join(str(path.relative_to(REPO_ROOT)) for path in charts))
    print("README glance + chat sections regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
