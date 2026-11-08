"""CircuitGraft Gradio Space.

The demo is deliberately data-driven: it reads the tracked S9 bundle under
``results/s9/report-*/`` and optional cached preset responses. It never loads a
model or invents a completion, so the Space stays usable without a GPU and
cannot show numbers that are not in the repository.

Deployment notes are in ``space/README.md``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLOTS = ("pareto_frontier.png", "circuit_overlap_heatmap.png", "attack_decay.png")
TABLES = ("table_main.md", "table_attacks.md", "table_overlap.md")


@dataclass(frozen=True)
class ReportBundle:
    directory: Path
    summary: dict[str, Any]
    tables: dict[str, str]
    plots: dict[str, Path]

    @property
    def run_id(self) -> str:
        return str(self.summary.get("run_id") or self.directory.name)

    @property
    def status(self) -> str:
        return str(self.summary.get("status") or "unknown")

    @property
    def reasons(self) -> list[str]:
        reasons = self.summary.get("partial_reasons") or []
        return [str(reason) for reason in reasons]

    def table(self, name: str) -> str:
        key = name if name.endswith(".md") else f"{name}.md"
        return self.tables.get(key, "_Table not generated._")


def _repo_root() -> Path:
    override = os.environ.get("CIRCUITGRAFT_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def find_latest_report(
    results_dir: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> Path | None:
    """Return the newest tracked ``results/s9/report-*`` directory."""
    if results_dir is not None:
        directories = [Path(results_dir)]
    else:
        override = os.environ.get("CIRCUITGRAFT_RESULTS_DIR")
        if override:
            directories = [Path(override)]
        elif root is not None:
            directories = [Path(root) / "results" / "s9"]
        else:
            here = Path(__file__).resolve().parent
            directories = [here / "results" / "s9", here.parent / "results" / "s9"]

    candidates: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        candidates.extend(
            path
            for path in directory.glob("report-*")
            if path.is_dir() and (path / "summary.json").is_file()
        )
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[str, float]:
        manifest = path / "manifest.json"
        stamp = ""
        if manifest.is_file():
            try:
                stamp = str(json.loads(manifest.read_text(encoding="utf-8")).get("finished_at") or "")
            except (OSError, ValueError, json.JSONDecodeError):
                stamp = ""
        return stamp, path.stat().st_mtime

    return max(candidates, key=sort_key)


def load_bundle(
    report_dir: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> ReportBundle | None:
    """Load a tracked report directory, returning ``None`` when none exists."""
    if report_dir is None:
        report_dir = os.environ.get("CIRCUITGRAFT_REPORT_DIR")
    directory = Path(report_dir) if report_dir else find_latest_report(root=root)
    if directory is None or not directory.is_dir():
        return None

    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict):
        return None

    tables: dict[str, str] = {}
    for name in TABLES:
        path = directory / name
        if path.is_file():
            tables[name] = path.read_text(encoding="utf-8")
    plots = {name: directory / name for name in PLOTS if (directory / name).is_file()}
    return ReportBundle(directory=directory, summary=summary, tables=tables, plots=plots)


def load_cached_responses(path: str | Path | None = None) -> dict[str, Any]:
    """Load optional preset responses; a missing/invalid file is empty data."""
    if path is None:
        path = os.environ.get("CIRCUITGRAFT_CACHED_RESPONSES")
    here = Path(__file__).resolve().parent
    candidates = [Path(path)] if path else [
        here / "cached_responses.json",
        _repo_root() / "space" / "cached_responses.json",
        _repo_root() / "cached_responses.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def prompt_choices(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(label, prompt_id)`` pairs for the cached-response dropdown."""
    choices: list[tuple[str, str]] = []
    for index, prompt in enumerate(payload.get("prompts") or []):
        if not isinstance(prompt, dict):
            continue
        prompt_id = str(prompt.get("id") or f"prompt-{index}")
        label = str(prompt.get("label") or prompt_id)
        category = prompt.get("category")
        choices.append((f"{label} [{category}]" if category else label, prompt_id))
    return choices


def model_names(payload: dict[str, Any]) -> list[str]:
    models = payload.get("models")
    if isinstance(models, list) and models:
        return [str(model) for model in models]
    responses = payload.get("responses")
    if isinstance(responses, dict) and responses:
        return sorted(str(model) for model in responses)
    return []


def response_for(payload: dict[str, Any], prompt_id: str, model: str) -> str:
    responses = payload.get("responses")
    if not isinstance(responses, dict):
        return ""
    model_responses = responses.get(model)
    if not isinstance(model_responses, dict):
        return ""
    return str(model_responses.get(prompt_id) or "")


def status_markdown(bundle: ReportBundle | None) -> str:
    """Render the provenance banner shown above every table."""
    if bundle is None:
        return (
            "### No tracked report bundle yet\n\n"
            "Run `cg report` after the Phase 4 matrix, then copy "
            "`results/s9/report-*/` into the Space repository. The demo will not "
            "display hand-typed or placeholder metrics."
        )

    lines = [
        f"**Run:** `{bundle.run_id}`  ",
        f"**Status:** `{bundle.status}`  ",
        f"**Bundle:** `{bundle.directory.name}`",
    ]
    sources = bundle.summary.get("sources")
    if isinstance(sources, dict):
        for label in ("eval", "overlap"):
            value = sources.get(label)
            if value:
                lines.append(f"- **{label}:** `{value}`")
        attacks = sources.get("attacks")
        if attacks:
            lines.append(f"- **attacks:** {len(attacks)} run(s)")
    if bundle.reasons:
        lines.append("")
        lines.append("**Partial reasons:**")
        lines.extend(f"- {reason}" for reason in bundle.reasons)
    elif bundle.status == "complete":
        lines.append("")
        lines.append("All required S7/S4/S8 sources were present for this report.")
    else:
        lines.append("")
        lines.append(
            "_Report status is not `complete`; inspect `summary.json` before "
            "treating these tables as evidence._"
        )
    lines.append("")
    lines.append(
        "_Research artifact. A partial report must not be quoted as final "
        "experiment evidence._"
    )
    return "\n".join(lines)


def build_app() -> Any:
    """Build the Gradio UI; imported lazily so tests do not need Gradio."""
    import gradio as gr

    bundle = load_bundle()
    cached = load_cached_responses()
    choices = prompt_choices(cached)
    models = model_names(cached)

    with gr.Blocks(title="CircuitGraft") as demo:
        gr.Markdown(
            "# CircuitGraft\n"
            "Circuit-aware safety recovery for domain-fine-tuned Cohere Tiny Aya. "
            "This Space reads tracked S9 artifacts; it does not run live models."
        )

        with gr.Tab("Results"):
            gr.Markdown(status_markdown(bundle))
            if bundle is None:
                gr.Markdown(
                    "```bash\n"
                    "cg report\n"
                    "cp -r results/s9/report-* /path/to/space/results/s9/\n"
                    "```"
                )
            else:
                gr.Markdown("## Main recovery table")
                gr.Markdown(bundle.table("table_main"))
                if "pareto_frontier.png" in bundle.plots:
                    gr.Image(
                        value=str(bundle.plots["pareto_frontier.png"]),
                        label="Pareto frontier",
                    )
                else:
                    gr.Markdown("_Pareto plot not generated._")

        with gr.Tab("Circuit overlap"):
            if bundle is None:
                gr.Markdown("_No report bundle._")
            else:
                gr.Markdown(bundle.table("table_overlap"))
                if "circuit_overlap_heatmap.png" in bundle.plots:
                    gr.Image(
                        value=str(bundle.plots["circuit_overlap_heatmap.png"]),
                        label="Safety vs domain circuit overlap",
                    )
                else:
                    gr.Markdown("_Overlap heatmap not generated._")

        with gr.Tab("Robustness"):
            if bundle is None:
                gr.Markdown("_No report bundle._")
            else:
                gr.Markdown(bundle.table("table_attacks"))
                if "attack_decay.png" in bundle.plots:
                    gr.Image(
                        value=str(bundle.plots["attack_decay.png"]),
                        label="Harmful fine-tuning decay",
                    )
                else:
                    gr.Markdown(
                        "_No attack-decay plot: S8 did not produce measured HRR "
                        "for this run._"
                    )

        with gr.Tab("Compare"):
            gr.Markdown(
                "Cached preset responses only. Harmful completions are never "
                "included; this tab is for refusal and domain-answer comparisons."
            )
            if not choices or not models:
                gr.Markdown(
                    "No cached responses found. Add `space/cached_responses.json` "
                    "(schema in `space/README.md`) after the models are published."
                )
            else:
                prompt = gr.Dropdown(
                    choices=choices,
                    value=choices[0][1],
                    label="Preset prompt",
                )
                outputs = [gr.Textbox(label=model, lines=4) for model in models]

                def show(prompt_id: str) -> list[str]:
                    return [response_for(cached, prompt_id, model) for model in models]

                prompt.change(show, inputs=prompt, outputs=outputs)
                demo.load(show, inputs=prompt, outputs=outputs)

        with gr.Tab("About"):
            gr.Markdown(
                "CircuitGraft discovers a safety circuit and a domain circuit in "
                "the same head/MLP component space, then grafts safety back while "
                "protecting shared components.\n\n"
                "- Code: https://github.com/dubeydevansh/circuit-graft\n"
                "- Results: `results/s9/` in the repository\n"
                "- Track 2b / Problem 6, Lexsi Labs Hackathon\n\n"
                "Every displayed number comes from the tracked S9 bundle. The "
                "project README explains how to reproduce it with "
                "`scripts/run_all.sh`."
            )

    return demo


if __name__ == "__main__":
    build_app().launch()
