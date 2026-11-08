"""``cg`` command line entry point."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from circuitgraft.config import (
    AttacksConfig,
    BaseConfig,
    BaselineConfig,
    DataConfig,
    DiscoverConfig,
    DriftConfig,
    EvalConfig,
    GraftConfig,
    config_hash,
    load_config,
)
from circuitgraft.registry import REPO_ROOT, library_versions

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="Circuit-aware safety recovery for domain-fine-tuned Cohere Tiny Aya.",
)
console = Console()


def _base() -> BaseConfig:
    return load_config("configs/base.yaml", BaseConfig, base_path=None)


def _config_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _print_config_summary(name: str, cfg: Any) -> None:
    console.print(f"[bold]{name}[/bold]  hash=[cyan]{config_hash(cfg)}[/cyan]")
    console.print_json(json.dumps(cfg.model_dump(mode="json", exclude_none=True), default=str))


@app.command()
def doctor(
    check_models: bool = typer.Option(
        False, "--check-models", help="Verify access to the gated Tiny Aya repos."
    ),
    json_out: Path | None = typer.Option(
        None, "--json-out", help="Write the environment report as JSON."
    ),
) -> None:
    """Check the D1 environment: packages, device, token and gated-model access."""
    versions = library_versions()
    required = ["safetune", "circuitkit", "auditkit", "curatorkit", "torch", "transformers"]
    missing = [name for name in required if versions.get(name) in (None, "not-installed")]

    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        device = "cuda" if cuda else "cpu"
    except Exception:  # noqa: BLE001
        cuda, gpu, device = False, None, "unavailable"

    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "device": device,
        "cuda": cuda,
        "gpu": gpu,
        "hf_token_present": token_present,
        "hf_org": os.environ.get("HF_ORG", ""),
        "versions": versions,
        "missing": missing,
        "model_access": {},
    }

    table = Table(title="CircuitGraft D1 environment")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Python", report["python"])
    table.add_row("Device", f"{device}" + (f" ({gpu})" if gpu else ""))
    table.add_row("HF token", "set" if token_present else "[red]missing[/red]")
    table.add_row("HF org", report["hf_org"] or "[yellow]not set[/yellow]")
    for name in required:
        table.add_row(name, versions.get(name, "not-installed"))
    console.print(table)

    if missing:
        console.print(f"[red]Missing required packages: {', '.join(missing)}[/red]")

    if check_models:
        if not token_present:
            console.print("[red]--check-models requires HF_TOKEN (gated Tiny Aya repos).[/red]")
            report["model_access"]["error"] = "HF_TOKEN missing"
        else:
            from huggingface_hub import hf_hub_download

            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            for repo in ("CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-base"):
                try:
                    hf_hub_download(repo, "config.json", token=token)
                    report["model_access"][repo] = "ok"
                    console.print(f"[green]{repo}: accessible[/green]")
                except Exception as exc:  # noqa: BLE001
                    report["model_access"][repo] = f"{type(exc).__name__}: {exc}"
                    console.print(f"[red]{repo}: {type(exc).__name__}: {exc}[/red]")

    if json_out is not None:
        json_out = _config_path(json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        console.print(f"report -> {json_out}")

    if missing or (check_models and report["model_access"].get("error")):
        raise typer.Exit(code=1)


@app.command("config-check")
def config_check(
    path: Path,
    kind: str = typer.Option(
        "auto",
        "--kind",
        help="auto|base|data|drift|discover|graft|baseline|eval|attacks",
    ),
) -> None:
    """Validate a YAML config against its schema and print the effective hash."""
    path = _config_path(path)
    mapping: dict[str, Any] = {
        "base": (BaseConfig, None),
        "data": (DataConfig, "configs/base.yaml"),
        "drift": (DriftConfig, "configs/base.yaml"),
        "discover": (DiscoverConfig, "configs/base.yaml"),
        "graft": (GraftConfig, "configs/base.yaml"),
        "baseline": (BaselineConfig, "configs/base.yaml"),
        "eval": (EvalConfig, "configs/base.yaml"),
        "attacks": (AttacksConfig, "configs/base.yaml"),
    }

    if kind == "auto":
        if path.name == "base.yaml":
            kind = "base"
        elif "graft" in path.parts:
            kind = "graft"
        elif "baselines" in path.parts:
            kind = "baseline"
        elif path.name.startswith("data"):
            kind = "data"
        elif path.name.startswith("drift"):
            kind = "drift"
        elif path.name.startswith("discover"):
            kind = "discover"
        elif path.name.startswith("eval"):
            kind = "eval"
        elif path.name.startswith("attacks"):
            kind = "attacks"
        else:
            raise typer.BadParameter("could not infer --kind; pass it explicitly")
    if kind not in mapping:
        raise typer.BadParameter(f"unknown kind {kind!r}")
    schema, base_path = mapping[kind]
    cfg = load_config(path, schema, base_path=base_path)
    _print_config_summary(f"{kind}: {path.name}", cfg)


def _run_stage(stage: str, cfg: Any, dry_run: bool) -> None:
    _print_config_summary(stage, cfg)
    if dry_run:
        console.print(f"[green]{stage}: config is valid (dry run)[/green]")
        return
    raise typer.Exit(
        code=2,
    )


@app.command()
def data(
    config: Path = typer.Option(Path("configs/data.yaml"), "--config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """S1: curate MedMCQA, build eval sets and contrastive pairs (CuratorKIT)."""
    from circuitgraft.data.curate import run_data

    cfg = load_config(config, DataConfig)
    if dry_run:
        _run_stage("data", cfg, True)
        return
    run_data(cfg, _base())


@app.command()
def drift(
    config: Path = typer.Option(Path("configs/drift.yaml"), "--config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """S2: LoRA SFT on curated medical data -> merged drifted checkpoint."""
    from circuitgraft.drift.finetune import run_drift

    cfg = load_config(config, DriftConfig)
    if dry_run:
        _run_stage("drift", cfg, True)
        return
    run_drift(cfg, _base())


@app.command()
def audit(
    config: Path = typer.Option(Path("configs/eval.yaml"), "--config"),
    models: str = typer.Option("aligned,drifted", "--models"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """S3: baseline audit of theta_A and theta_D (SafeTune + AuditKIT)."""
    from circuitgraft.eval.safety import run_audit

    cfg = load_config(config, EvalConfig)
    if dry_run:
        _run_stage("audit", cfg, True)
        return
    run_audit(cfg, _base(), models=[m.strip() for m in models.split(",") if m.strip()])


@app.command()
def discover(
    config: Path = typer.Option(Path("configs/discover.yaml"), "--config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """S4: discover S on theta_A and D on theta_D, then test overlap/faithfulness."""
    from circuitgraft.discover.circuitkit_runner import run_discovery

    cfg = load_config(config, DiscoverConfig)
    if dry_run:
        _run_stage("discover", cfg, True)
        return
    run_discovery(cfg, _base())


@app.command()
def baselines(
    config: Path = typer.Option(Path("configs/baselines/resta.yaml"), "--config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    sweep: bool = typer.Option(
        False, "--sweep", help="Run every point in the config's sweep grid."
    ),
) -> None:
    """S5: SafeTune faithful recovery baselines."""
    from circuitgraft.baselines.safetune_recover import run_baseline, run_baseline_sweep

    cfg = load_config(config, BaselineConfig)
    if dry_run:
        _run_stage("baselines", cfg, True)
        return
    if sweep:
        run_baseline_sweep(cfg, _base())
    else:
        run_baseline(cfg, _base())


@app.command()
def graft(
    config: Path = typer.Option(Path("configs/graft/cg_project.yaml"), "--config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """S6: apply CircuitGraft to a drifted checkpoint."""
    from circuitgraft.graft.pipeline import run_graft

    cfg = load_config(config, GraftConfig)
    if dry_run:
        _run_stage("graft", cfg, True)
        return
    run_graft(cfg, _base())


@app.command("eval")
def eval_command(
    config: Path = typer.Option(Path("configs/eval.yaml"), "--config"),
    all_models: bool = typer.Option(False, "--all"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """S7: generate, judge and compute metrics for the model matrix."""
    from circuitgraft.eval.pipeline import run_eval

    cfg = load_config(config, EvalConfig)
    if dry_run:
        _run_stage("eval", cfg, True)
        return
    run_eval(cfg, _base(), all_models=all_models)


@app.command()
def attack(
    config: Path = typer.Option(Path("configs/attacks.yaml"), "--config"),
    attack_type: str = typer.Option("abliterate", "--type", help="abliterate|harmful_ft|quant"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """S8: abliteration, harmful re-fine-tuning and quantization attacks."""
    from circuitgraft.attacks.runner import run_attack

    cfg = load_config(config, AttacksConfig)
    if dry_run:
        _run_stage("attack", cfg, True)
        return
    run_attack(cfg, _base(), attack_type=attack_type)


@app.command("eval-attacks")
def eval_attacks_command(
    config: Path = typer.Option(Path("configs/eval.yaml"), "--config"),
    attacks_dir: Path = typer.Option(Path("artifacts/attacks"), "--attacks-dir"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    generations_dir: Path | None = typer.Option(None, "--generations-dir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """S7 follow-up: evaluate saved S8 checkpoints and attach post-attack HRR."""
    from circuitgraft.attacks.evaluate import collect_attack_checkpoints, run_attack_eval

    cfg = load_config(config, EvalConfig)
    if dry_run:
        checkpoints = collect_attack_checkpoints(attacks_dir)
        console.print(
            f"eval-attacks: dry run ({len(checkpoints)} saved checkpoint(s))"
        )
        for checkpoint in checkpoints:
            console.print(f"- {checkpoint.label}: {checkpoint.checkpoint}")
        return
    summary = run_attack_eval(
        cfg,
        _base(),
        attacks_root=attacks_dir,
        output_dir=output_dir,
        generations_dir=generations_dir,
        overwrite=overwrite,
    )
    console.print(
        f"eval-attacks ({summary['status']}) -> {summary.get('output_dir', '—')} "
        f"[{len(summary.get('checkpoints', []))} checkpoint(s)]"
    )
    if summary["status"] == "not_evaluated":
        console.print(summary.get("reason", ""))


@app.command()
def report(
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    readme: Path | None = typer.Option(None, "--readme", help="Markdown file to inject."),
    results_out: Path | None = typer.Option(
        None, "--results-out", help="Tracked copy directory for tables and plots."
    ),
) -> None:
    """S9: build tables/plots and inject them into README/Article."""
    from circuitgraft.report.pipeline import run_report

    if dry_run:
        console.print("[green]report: dry run[/green]")
        return
    summary = run_report(
        output_dir=output_dir,
        readme_path=readme,
        results_out=results_out,
    )
    console.print(
        f"report ({summary['status']}) -> {summary['results_dir']} "
        f"[{len(summary['partial_reasons'])} partial reasons]"
    )


@app.command()
def publish(
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    """S10: publish models/datasets/collections to the HF namespace."""
    from circuitgraft.publish import run_publish

    base = _base()
    if not base.hf_org:
        raise typer.BadParameter("HF_ORG is empty; set it in .env or the environment")
    summary = run_publish(base, dry_run=dry_run, output_dir=output_dir)
    mode = "dry run" if dry_run else "published"
    console.print(
        f"publish ({mode}) -> [cyan]{base.hf_org}[/cyan]: "
        f"{len(summary['models_published'])} models, "
        f"{len(summary['datasets_published'])} datasets, "
        f"{len(summary['skipped'])} skipped"
    )
    if summary["skipped"]:
        console.print_json(json.dumps(summary["skipped"], indent=2))


if __name__ == "__main__":
    app()
