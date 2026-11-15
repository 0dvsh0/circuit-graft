"""Single-notebook Kaggle orchestrator for the CircuitGraft Phase 4 run.

The notebook calls these idempotent subcommands. Every stage writes to a
scratch tree (default ``/kaggle/tmp/cg``) so the large checkpoints and the HF
cache do not count against Kaggle's 20 GB ``/kaggle/working`` output cap, and
every completed milestone is uploaded to the private cloud dataset repo.

Commands
--------
setup                 create scratch dirs/symlinks and import cloud milestones
data                  import + hard-validate the verified data snapshot (never ``cg data``)
drift                 train the drifted checkpoint and upload it
discover-safety       discover S on the aligned model and upload it
discover-domain       discover D on the drifted model and upload it
finalize-discover     assemble overlap/summary/manifest for the split S4 run
candidates            build named baselines/grafts without evaluating them
eval-refs             evaluate aligned + drifted and compute Gate G-drift
eval-candidates       build one missing candidate, evaluate it, archive its weights
merge-eval            merge the staged summaries into one report-ready summary
report                run S9 report generation
publish               run S10 publication (drifted + cg_project models, datasets)
attacks               optional abliteration on aligned + cg_project
export                copy the small, submission-relevant outputs to /kaggle/working
status                print a one-screen progress summary

Run ``python scripts/kaggle_full_run.py --help`` for the full list.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRATCH = Path(os.environ.get("CG_SCRATCH", "/kaggle/tmp/cg"))
CLOUD_REPO = os.environ.get("CG_CLOUD_REPO", "dubeydevansh/circuitgraft-phase4-cloud")
DATA_REPO = os.environ.get("CG_DATA_REPO", "dubeydevansh/circuitgraft-phase4-data")
VERIFIED_DATA_RUN = os.environ.get("CG_DATA_RUN_ID", "data-bc99df1d4f173b72")
BUDGET_HOURS = float(os.environ.get("CG_BUDGET_HOURS", "11.0"))
JUDGE = os.environ.get("CG_JUDGE", "string_match")

DATA_REQUIRED_EVAL = (
    "medmcqa_validation",
    "medqa_test",
    "pubmedqa_labeled",
    "mmlu_non_medical",
    "harmbench",
    "advbench",
    "sorrybench",
    "wildjailbreak",
    "xstest_safe",
    "orbench_hard",
    "domain_benign",
    "domain_harmful",
)

# Preregistered "never cut" core: three baselines + the three CircuitGraft
# arms that carry the primary comparison.
CORE_CANDIDATES: tuple[str, ...] = (
    "cg_project",
    "cg_uniform",
    "cg_random",
    "resta",
    "wiseft",
    "ctheta",
)

CANDIDATE_STAGE: dict[str, tuple[str, str]] = {
    "cg_project": ("graft", "configs/graft/cg_project.yaml"),
    "cg_uniform": ("graft", "configs/graft/cg_uniform.yaml"),
    "cg_exclude": ("graft", "configs/graft/cg_exclude.yaml"),
    "cg_distributed": ("graft", "configs/graft/cg_distributed.yaml"),
    "cg_random": ("graft", "configs/graft/cg_random.yaml"),
    "cg_fisher": ("graft", "configs/graft/cg_fisher.yaml"),
    "resta": ("baseline", "configs/baselines/resta.yaml"),
    "wiseft": ("baseline", "configs/baselines/wiseft.yaml"),
    "ctheta": ("baseline", "configs/baselines/ctheta.yaml"),
    "safelora": ("baseline", "configs/baselines/safelora.yaml"),
}

LOG = logging.getLogger("cg.kaggle")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _repo(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    hf_home = SCRATCH / "hf"
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    env.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(hf_home / "hub"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_ORG", "dubeydevansh")
    env.setdefault("CG_SCRATCH", str(SCRATCH))
    return env


def _run(args: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(item) for item in args)
    LOG.info("$ %s", printable)
    try:
        subprocess.run(args, cwd=REPO_ROOT, env=env or _runtime_env(), check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"command failed with exit code {exc.returncode}: {printable}") from exc


def _cg(*args: str) -> None:
    _run([sys.executable, "-m", "circuitgraft.cli", *args])


def _disk_report() -> None:
    LOG.info("repo root: %s", REPO_ROOT)
    LOG.info("scratch:   %s", SCRATCH)
    for path in (Path("/kaggle/working"), SCRATCH):
        if path.exists():
            try:
                usage = shutil.disk_usage(path)
                LOG.info(
                    "disk %-18s total=%5.1f GiB free=%5.1f GiB",
                    path,
                    usage.total / 2**30,
                    usage.free / 2**30,
                )
            except OSError:
                pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _line_count(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _started_at() -> float:
    marker = SCRATCH / "started_at.json"
    if marker.exists():
        try:
            return float(_read_json(marker)["epoch"])
        except Exception:  # noqa: BLE001 - a corrupt marker just resets the clock
            pass
    payload = {"epoch": time.time(), "iso": _now()}
    _write_json(marker, payload)
    return float(payload["epoch"])


def _hours_left() -> float:
    return BUDGET_HOURS - (time.time() - _started_at()) / 3600.0


def _print_time_left() -> None:
    left = _hours_left()
    LOG.info("time left in the %0.1f h budget: %0.2f h", BUDGET_HOURS, left)


# ---------------------------------------------------------------------------
# Hugging Face Hub helpers
# ---------------------------------------------------------------------------


def _token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. In Kaggle: Add-ons > Secrets > add HF_TOKEN and "
            "attach it to this notebook, then Save & Run All again. The run cannot "
            "download the verified data snapshot or gated models without it."
        )
    return token.strip()


def _hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=_token())


def _cloud_exists() -> bool:
    try:
        _hf_api().dataset_info(CLOUD_REPO)
        return True
    except Exception:  # noqa: BLE001 - 404 is the normal first-run state
        return False


def _cloud_files() -> list[str]:
    try:
        return list(_hf_api().list_repo_files(CLOUD_REPO, repo_type="dataset"))
    except Exception:  # noqa: BLE001 - an unreadable repo behaves like an empty one
        return []


def _ensure_cloud_repo() -> None:
    _hf_api().create_repo(CLOUD_REPO, repo_type="dataset", private=True, exist_ok=True)


def _upload(paths: Iterable[str | Path], message: str, *, patterns: Sequence[str] | None = None) -> None:
    """Upload files/directories, preserving their repo-relative paths."""
    selected = [Path(path) for path in paths if Path(path).exists()]
    if not selected:
        LOG.info("nothing to upload for: %s", message)
        return
    _ensure_cloud_repo()
    api = _hf_api()
    for path in selected:
        repo_path = _rel(path)
        if path.is_dir():
            LOG.info("uploading %s -> %s", path, repo_path)
            api.upload_folder(
                folder_path=str(path),
                repo_id=CLOUD_REPO,
                repo_type="dataset",
                path_in_repo=repo_path,
                allow_patterns=list(patterns) if patterns else None,
                commit_message=message,
            )
        else:
            LOG.info("uploading %s -> %s", path, repo_path)
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=repo_path,
                repo_id=CLOUD_REPO,
                repo_type="dataset",
                commit_message=message,
            )


def _import_cloud() -> None:
    if not _cloud_exists():
        LOG.info("cloud repo %s does not exist yet; starting from scratch", CLOUD_REPO)
        return
    files = _cloud_files()
    if not any(name.startswith(("artifacts/", "results/")) for name in files):
        LOG.info("cloud repo %s has no stage milestones yet; starting from scratch", CLOUD_REPO)
        return
    from huggingface_hub import snapshot_download

    LOG.info("importing completed milestones from %s", CLOUD_REPO)
    snapshot_download(
        repo_id=CLOUD_REPO,
        repo_type="dataset",
        local_dir=str(REPO_ROOT),
        allow_patterns=["artifacts/**", "results/**"],
        token=_token(),
    )


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


def _symlink_or_fallback(name: str) -> None:
    source = REPO_ROOT / name
    target = SCRATCH / name
    target.mkdir(parents=True, exist_ok=True)

    if source.is_symlink():
        if source.resolve() != target.resolve():
            source.unlink()
            os.symlink(target, source, target_is_directory=True)
        return
    if source.exists():
        if any(target.iterdir()):
            LOG.warning("merging existing %s into %s", source, target)
            shutil.copytree(source, target, dirs_exist_ok=True)
            shutil.rmtree(source)
        else:
            shutil.rmtree(target)
            shutil.move(str(source), str(target))
        os.symlink(target, source, target_is_directory=True)
        return
    try:
        os.symlink(target, source, target_is_directory=True)
    except OSError as exc:
        LOG.warning(
            "could not create the %s symlink (%s); keeping the repo directory. "
            "This is fine locally but Kaggle should use /kaggle/tmp.",
            name,
            exc,
        )


def cmd_setup(args: argparse.Namespace) -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "hf").mkdir(parents=True, exist_ok=True)
    (SCRATCH / "runtime").mkdir(parents=True, exist_ok=True)
    for name in ("artifacts", "results"):
        _symlink_or_fallback(name)
    _started_at()
    _disk_report()
    if not args.no_import:
        _import_cloud()
    _write_json(
        SCRATCH / "runtime" / "setup.json",
        {
            "created_at": _now(),
            "scratch": str(SCRATCH),
            "cloud_repo": CLOUD_REPO,
            "data_repo": DATA_REPO,
            "budget_hours": BUDGET_HOURS,
            "judge": JUDGE,
        },
    )
    _print_time_left()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def validate_snapshot(summary: dict[str, Any], data_root: Path) -> list[str]:
    """Return every reason the imported snapshot is not the verified G1 data."""
    reasons: list[str] = []
    run_id = summary.get("run_id")
    if VERIFIED_DATA_RUN and run_id != VERIFIED_DATA_RUN:
        reasons.append(f"run_id is {run_id!r}, expected {VERIFIED_DATA_RUN!r}")
    gate = summary.get("gate_g1", {})
    if gate.get("status") != "passed":
        reasons.append(f"gate_g1 is {gate.get('status')!r}: {gate.get('reasons')}")
    if gate.get("reasons"):
        reasons.extend(str(reason) for reason in gate["reasons"])

    sft = summary.get("sft", {})
    if sft.get("status") != "complete":
        reasons.append(f"SFT curation is {sft.get('status')!r}")
    for name in ("train.jsonl", "validation.jsonl"):
        path = data_root / "medmcqa_sft" / name
        if not path.exists() or path.stat().st_size == 0:
            reasons.append(f"missing or empty {path}")
        elif _line_count(path) == 0:
            reasons.append(f"no rows in {path}")

    pairs = summary.get("pairs", {})
    for name in ("safety", "domain"):
        info = pairs.get(name, {})
        path = data_root / "pairs" / f"{name}.jsonl"
        if info.get("status") != "complete":
            reasons.append(f"{name} pairs are {info.get('status')!r}")
        if not path.exists() or _line_count(path) == 0:
            reasons.append(f"missing or empty {path}")
        if info.get("validation", {}).get("status") != "passed":
            reasons.append(f"{name} pair tokenizer validation is not passed")

    for name in DATA_REQUIRED_EVAL:
        info = summary.get("eval", {}).get(name, {})
        path = data_root / "eval" / f"{name}.jsonl"
        if info.get("status") != "complete":
            reasons.append(f"eval set {name} is {info.get('status')!r}")
        if not path.exists() or _line_count(path) == 0:
            reasons.append(f"missing or empty {path}")

    for name, minimum in (("domain_benign", 300), ("domain_harmful", 150)):
        info = summary.get("eval", {}).get(name, {})
        rows = int(info.get("rows", 0))
        verified = int(info.get("human_verified", 0))
        if rows < int(minimum * 0.8):
            reasons.append(f"{name} has {rows} rows, below 80% of {minimum}")
        if verified < rows:
            reasons.append(
                f"{name} has {verified}/{rows} human-verified rows; refusing to "
                "accept auto-generated review labels"
            )
    return reasons


def _remove_data_tree() -> None:
    data_root = REPO_ROOT / "artifacts" / "data"
    if data_root.is_symlink():
        data_root.unlink()
        return
    if data_root.exists():
        shutil.rmtree(data_root)


def cmd_data(args: argparse.Namespace) -> None:
    data_root = REPO_ROOT / "artifacts" / "data"
    summary_path = data_root / "summary.json"

    def _valid() -> bool:
        if not summary_path.exists():
            return False
        try:
            summary = _read_json(summary_path)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not read %s: %s", summary_path, exc)
            return False
        reasons = validate_snapshot(summary, data_root)
        if reasons:
            LOG.warning("imported data snapshot rejected:")
            for reason in reasons:
                LOG.warning("  - %s", reason)
            return False
        LOG.info("data snapshot verified: run_id=%s gate_g1=passed", summary.get("run_id"))
        for name, info in summary.get("eval", {}).items():
            LOG.info("  %-22s %-10s rows=%s", name, info.get("status"), info.get("rows"))
        return True

    if _valid():
        return
    if args.validate_only:
        raise RuntimeError("local data snapshot is not the verified G1 snapshot")

    if summary_path.exists():
        LOG.warning("deleting the rejected data tree before re-downloading")
        _remove_data_tree()
    _token()
    from huggingface_hub import snapshot_download

    LOG.info("downloading verified data snapshot %s", DATA_REPO)
    snapshot_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        local_dir=str(data_root),
        token=_token(),
    )
    if not _valid():
        raise RuntimeError(
            "the downloaded data snapshot is not the verified G1 snapshot. "
            "Refusing to fall back to `cg data` (which would regenerate labels "
            "and cannot pass G1 honestly). Check CG_DATA_REPO / CG_DATA_RUN_ID."
        )
    _upload([summary_path], "data: verified snapshot summary")


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------


def _drift_dir() -> Path:
    return REPO_ROOT / "artifacts" / "drift" / "scenarioA_seed0"


def _drift_ready() -> bool:
    merged = _drift_dir() / "merged" / "config.json"
    weights = list((_drift_dir() / "merged").glob("*.safetensors"))
    manifest = _drift_dir() / "manifest.json"
    if not merged.exists() or not weights:
        return False
    if manifest.exists():
        try:
            return _read_json(manifest).get("status") == "complete"
        except Exception:  # noqa: BLE001
            return False
    return True


def _write_drift_config(epochs: int) -> Path:
    import yaml

    payload = yaml.safe_load((REPO_ROOT / "configs" / "drift_t4.yaml").read_text(encoding="utf-8"))
    payload.setdefault("train", {})["epochs"] = int(epochs)
    path = SCRATCH / "runtime" / f"drift_t4_e{epochs}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    lineage = REPO_ROOT / "results" / "kaggle" / "configs"
    lineage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, lineage / path.name)
    return path


def cmd_drift(args: argparse.Namespace) -> None:
    cmd_data(argparse.Namespace(validate_only=False))
    if _drift_ready() and not args.overwrite:
        LOG.info("drifted checkpoint already complete: %s", _drift_dir() / "merged")
    else:
        if args.overwrite:
            LOG.warning("overwrite requested: removing %s", _drift_dir())
            shutil.rmtree(_drift_dir(), ignore_errors=True)
        config_path = _write_drift_config(args.epochs)
        LOG.info("drift config: %s (epochs=%s)", config_path, args.epochs)
        _print_time_left()
        _cg("drift", "--config", str(config_path))
    if not _drift_ready():
        raise RuntimeError("drift finished without a complete merged checkpoint")
    _upload(
        [_drift_dir()],
        "S2 drift: merged checkpoint + adapter + summary",
        patterns=["merged/**", "adapter/**", "manifest.json", "drift_summary.json"],
    )
    LOG.info("drift milestone uploaded")


# ---------------------------------------------------------------------------
# discovery (split so a crash cannot destroy the safety circuit)
# ---------------------------------------------------------------------------


def _discover_paths(kind: str) -> tuple[Any, Path]:
    from circuitgraft.config import DiscoverConfig, config_hash, load_config

    cfg = load_config("configs/discover.yaml", DiscoverConfig)
    run_id = f"discover-{config_hash(cfg)}"
    return cfg, REPO_ROOT / cfg.output_dir / run_id / kind


def _circuit_ready(kind_dir: Path) -> bool:
    required = ("circuit.pt", "scores.json", "circuit_info.json", "selected.json")
    if not all((kind_dir / name).exists() for name in required):
        return False
    try:
        selected = _read_json(kind_dir / "selected.json")
        if not selected.get("nodes"):
            return False
        info = _read_json(kind_dir / "circuit_info.json")
        return bool(info.get("units") or info.get("metadata") or info.get("model"))
    except Exception:  # noqa: BLE001
        return False


def cmd_discover(args: argparse.Namespace) -> None:
    cmd_data(argparse.Namespace(validate_only=False))
    from circuitgraft.config import BaseConfig, DiscoverConfig, config_hash, load_config
    from circuitgraft.discover.circuitkit_runner import (
        _model_ref,
        _repo_path,
        run_circuitkit_discovery,
    )
    from circuitgraft.graft.pipeline import load_layout

    kind = args.kind
    cfg = load_config("configs/discover.yaml", DiscoverConfig)
    base = load_config("configs/base.yaml", BaseConfig)
    run_id = f"discover-{config_hash(cfg)}"
    out_dir = _repo_path(cfg.output_dir) / run_id / kind
    sub = cfg.safety if kind == "safety" else cfg.domain
    model_ref = _model_ref(base, sub.model)

    if kind == "domain" and not _drift_ready():
        raise RuntimeError(
            "domain discovery needs the drifted checkpoint. Run `drift` first "
            "(or import it from the cloud repo)."
        )
    if _circuit_ready(out_dir) and not args.overwrite:
        LOG.info("%s circuit already complete: %s", kind, out_dir)
    else:
        if out_dir.exists():
            LOG.warning("removing incomplete %s circuit dir %s", kind, out_dir)
            shutil.rmtree(out_dir)
        LOG.info(
            "discovering %s circuit on %s (ig_steps=%s, n_examples=%s, batch=%s, seq=%s, dtype=%s)",
            kind,
            model_ref,
            sub.ig_steps,
            sub.n_examples,
            sub.batch_size,
            sub.max_seq_len,
            cfg.dtype,
        )
        _print_time_left()
        result = run_circuitkit_discovery(
            model_ref,
            _repo_path(sub.pairs),
            out_dir,
            layout=load_layout(base.models.base),
            kind=kind,
            algorithm=sub.method,
            level=sub.granularity,
            n_examples=sub.n_examples,
            sparsity=sub.sparsity,
            ig_steps=sub.ig_steps,
            batch_size=sub.batch_size,
            max_seq_len=sub.max_seq_len,
            device=cfg.device,
            dtype=cfg.dtype,
            seed=cfg.seed,
            faithfulness=cfg.faithfulness,
        )
        if not result.get("num_selected_nodes"):
            raise RuntimeError(f"{kind} discovery selected zero nodes; refusing success")
        if result.get("faithfulness_error"):
            LOG.warning("faithfulness diagnostic failed: %s", result["faithfulness_error"])
        LOG.info(
            "%s circuit: %s selected nodes / %s components",
            kind,
            result.get("num_selected_nodes"),
            result.get("num_components"),
        )
    if not _circuit_ready(out_dir):
        raise RuntimeError(f"{kind} circuit artifacts are incomplete in {out_dir}")
    _upload([out_dir], f"S4 {kind} circuit (ig_steps=2, cross_check off)")
    LOG.info("%s circuit milestone uploaded", kind)


# ---------------------------------------------------------------------------
# finalize discovery
# ---------------------------------------------------------------------------


def _discover_run_dir() -> Path:
    from circuitgraft.config import DiscoverConfig, config_hash, load_config

    cfg = load_config("configs/discover.yaml", DiscoverConfig)
    return REPO_ROOT / cfg.output_dir / f"discover-{config_hash(cfg)}"


def cmd_finalize_discover(args: argparse.Namespace) -> None:
    run_dir = _discover_run_dir()
    summary = run_dir / "summary.json"
    manifest = run_dir / "manifest.json"
    if summary.exists() and manifest.exists() and not args.overwrite:
        try:
            if _read_json(manifest).get("status") == "complete":
                LOG.info("discovery run already finalized: %s", run_dir)
                _upload([run_dir], "S4 finalized discovery")
                return
        except Exception:  # noqa: BLE001
            pass
    _run([sys.executable, "scripts/finalize_discover.py"])
    if not summary.exists():
        raise RuntimeError("finalize_discover.py did not write summary.json")
    _upload([run_dir], "S4 finalized discovery: overlap + summary + manifest")
    LOG.info("discovery finalized and uploaded")


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


def _eval_config_model() -> Any:
    from circuitgraft.config import EvalConfig, load_config

    return load_config("configs/eval.yaml", EvalConfig)


def _base_config() -> Any:
    from circuitgraft.config import BaseConfig, load_config

    return load_config("configs/base.yaml", BaseConfig)


def _checkpoint_for(name: str) -> Path | None:
    from circuitgraft.eval.models import resolve_checkpoint

    try:
        return Path(resolve_checkpoint(name, _eval_config_model(), _base_config()))
    except Exception:  # noqa: BLE001
        return None


def _candidate_run_dir(name: str, checkpoint: Path) -> Path:
    return checkpoint.parent if checkpoint.name == "merged" else checkpoint


def cmd_candidates(args: argparse.Namespace) -> None:
    cmd_data(argparse.Namespace(validate_only=False))
    if args.names:
        names = [name.strip() for name in args.names.split(",") if name.strip()]
    else:
        names = list(CORE_CANDIDATES)
    if args.extra:
        names.extend(
            name
            for name in ("cg_exclude", "cg_distributed", "safelora")
            if name not in names
        )

    for name in names:
        if name not in CANDIDATE_STAGE:
            raise SystemExit(f"unknown candidate {name!r}; choose from {sorted(CANDIDATE_STAGE)}")
        if _hours_left() < 1.25 and not args.force:
            LOG.warning("stopping candidate build: less than 1.25 h left")
            break
        checkpoint = _checkpoint_for(name)
        if checkpoint is not None and not args.overwrite:
            LOG.info("%s already built: %s", name, checkpoint)
            continue
        stage, config = CANDIDATE_STAGE[name]
        LOG.info("building %s with %s", name, config)
        _print_time_left()
        if stage == "baseline":
            _cg("baselines", "--config", config)
        else:
            _cg("graft", "--config", config)
        checkpoint = _checkpoint_for(name)
        if checkpoint is None:
            raise RuntimeError(f"{name} build finished but no checkpoint could be resolved")
        run_dir = _candidate_run_dir(name, checkpoint)
        patterns = (
            ["manifest.json", "baseline_stats.json"]
            if stage == "baseline"
            else ["manifest.json", "graft_stats.json", "selection.json"]
        )
        _upload([run_dir], f"S6/S5 {name}: stats + manifest", patterns=patterns)
        LOG.info("%s built: %s", name, checkpoint)

    _upload(
        [REPO_ROOT / "results" / "kaggle"],
        "Kaggle runtime configs and driver state",
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _write_eval_config(models: Sequence[str], label: str, judge: str) -> Path:
    import yaml

    payload = yaml.safe_load((REPO_ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    payload["models"] = list(models)
    payload["device"] = "cuda"
    payload["dtype"] = "float16"
    payload["output_dir"] = "artifacts/eval"
    payload["generations_dir"] = "results/generations"
    payload["skip_existing"] = True
    payload.setdefault("judge", {})["primary"] = judge
    if judge == "string_match":
        payload["judge"]["secondary"] = "string_match"
    data_root = REPO_ROOT / "artifacts" / "data"
    payload["benchmarks"] = [
        name
        for name in payload.get("benchmarks", [])
        if (data_root / "eval" / f"{name}.jsonl").exists()
    ]
    payload["max_rows"] = {
        name: rows
        for name, rows in payload.get("max_rows", {}).items()
        if name in payload["benchmarks"]
    }
    path = SCRATCH / "runtime" / f"eval_{label}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    lineage = REPO_ROOT / "results" / "kaggle" / "configs"
    lineage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, lineage / path.name)
    return path


def _latest_eval_summary() -> Path | None:
    root = REPO_ROOT / "artifacts" / "eval"
    if not root.is_dir():
        return None
    candidates = sorted(
        (path for path in root.glob("eval-*/summary.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _run_eval_stage(models: Sequence[str], label: str, judge: str) -> Path:
    config = _write_eval_config(models, label, judge)
    _print_time_left()
    _cg("eval", "--config", str(config))
    summary = _latest_eval_summary()
    if summary is None:
        raise RuntimeError("eval finished without writing a summary.json")
    run_dir = summary.parent
    _upload([run_dir], f"S7 eval: {label}", patterns=["summary.json", "items/**", "human_audit/**", "manifest.json", "auditkit/**"])
    _upload([REPO_ROOT / "results" / "generations"], f"S7 generations: {label}")
    _upload([REPO_ROOT / "results" / "kaggle" / "configs"], "Kaggle eval configs")
    return summary


def _find_model_summary(name: str) -> Path | None:
    root = REPO_ROOT / "artifacts" / "eval"
    if not root.is_dir():
        return None
    paths = sorted(
        (path for path in root.glob("eval-*/summary.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        try:
            doc = _read_json(path)
        except Exception:  # noqa: BLE001
            continue
        model = doc.get("models", {}).get(name)
        if isinstance(model, dict) and model.get("metrics") and model.get("status") in {
            "complete",
            "partial",
        }:
            return path
    return None


def cmd_eval_refs(args: argparse.Namespace) -> None:
    cmd_data(argparse.Namespace(validate_only=False))
    if _checkpoint_for("drifted") is None:
        raise RuntimeError("drifted checkpoint is missing; run the `drift` stage first")
    if _find_model_summary("aligned") and _find_model_summary("drifted") and not args.force:
        LOG.info("aligned/drifted eval already exists; skipping")
    else:
        _run_eval_stage(["aligned", "drifted"], "refs", args.judge)
    latest = _latest_eval_summary()
    if latest is not None:
        gate = _read_json(latest).get("gate_g_drift", {})
        LOG.info("Gate G-drift: %s", gate.get("status"))
        LOG.info(
            "  HRR aligned=%s drifted=%s drop=%s",
            gate.get("hrr_aligned"),
            gate.get("hrr_drifted"),
            gate.get("drop"),
        )


def cmd_eval_candidates(args: argparse.Namespace) -> None:
    names = [name.strip() for name in args.names.split(",") if name.strip()] if args.names else list(CORE_CANDIDATES)
    for name in names:
        if _find_model_summary(name) and not args.force:
            LOG.info("%s eval already exists; skipping", name)
            continue
        if _hours_left() < 0.75 and not args.force:
            LOG.warning("stopping eval loop: less than 0.75 h left")
            break
        if _checkpoint_for(name) is None:
            LOG.info("%s checkpoint missing; rebuilding it", name)
            cmd_candidates(argparse.Namespace(names=name, extra=False, force=True, overwrite=False))
        _run_eval_stage(["aligned", "drifted", name], name, args.judge)
        if name != "cg_project":
            _archive_checkpoint(name)
    cmd_merge_eval(argparse.Namespace())


# ---------------------------------------------------------------------------
# checkpoint archiving
# ---------------------------------------------------------------------------


def _archive_checkpoint(name: str) -> None:
    """Delete a candidate's weights but keep its stats/manifest as lineage.

    The manifest status becomes ``archived`` so a rerun rebuilds the checkpoint
    instead of silently skipping a stage whose weights no longer exist.
    """
    checkpoint = _checkpoint_for(name)
    if checkpoint is None:
        return
    run_dir = _candidate_run_dir(name, checkpoint)
    weights: list[Path] = []
    if checkpoint.is_file():
        weights.append(checkpoint)
    elif checkpoint.is_dir():
        weights.extend(checkpoint.glob("*.safetensors"))
        weights.extend(checkpoint.glob("*.bin"))
        if (checkpoint / "model.safetensors.index.json").exists():
            index = _read_json(checkpoint / "model.safetensors.index.json")
            for shard in set(index.get("weight_map", {}).values()):
                weights.append(checkpoint / shard)
    kept = []
    for weight in sorted(set(weights)):
        if weight.exists() and weight.is_file():
            description = f"{_rel(weight)} ({weight.stat().st_size / 2**30:.2f} GiB)"
            try:
                weight.unlink()
            except OSError as exc:
                LOG.warning("could not delete %s: %s", weight, exc)
                continue
            kept.append(description)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
            manifest["status"] = "archived"
            manifest["notes"] = (
                f"{manifest.get('notes') or ''} | weights archived after the Kaggle "
                "eval to fit the scratch disk; regenerate deterministically from the "
                "recorded inputs"
            )
            _write_json(manifest_path, manifest)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not mark %s archived: %s", manifest_path, exc)
    _write_json(
        run_dir / "ARCHIVED.json",
        {
            "archived_at": _now(),
            "model": name,
            "checkpoint": str(checkpoint),
            "weights": kept,
            "reason": "Kaggle scratch disk; metrics and inputs are retained",
        },
    )
    LOG.info("archived %s weights (%d file(s)); stats/manifest retained", name, len(kept))


# ---------------------------------------------------------------------------
# merged eval summary
# ---------------------------------------------------------------------------


def _model_quality(document: dict[str, Any]) -> tuple[int, int]:
    status = {"complete": 2, "partial": 1}.get(str(document.get("status")), 0)
    metrics = document.get("metrics")
    return status, len(metrics) if isinstance(metrics, dict) else 0


def merge_summaries(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge staged per-candidate summaries into one S9-readable summary."""
    merged_models: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Any] = {}
    reasons: list[str] = []
    gate: dict[str, Any] | None = None
    benchmarks: dict[str, Any] = {}
    for document in documents:
        doc_gate = document.get("gate_g_drift")
        if isinstance(doc_gate, dict) and doc_gate.get("status") in {"passed", "failed"}:
            gate = doc_gate
        for name, model in document.get("models", {}).items():
            if not isinstance(model, dict):
                continue
            current = merged_models.get(name)
            if current is None or _model_quality(model) > _model_quality(current):
                merged_models[name] = model
        for key, value in document.get("comparisons", {}).items():
            comparisons.setdefault(key, value)
        for reason in document.get("partial_reasons", []):
            if reason not in reasons:
                reasons.append(str(reason))
        benchmarks.update(document.get("benchmarks", {}))

    statuses = {name: model.get("status") for name, model in merged_models.items()}
    expected = {"aligned", "drifted", *CORE_CANDIDATES}
    missing = sorted(expected - set(merged_models))
    if missing:
        reasons.append("core models not evaluated: " + ", ".join(missing))
    complete = bool(merged_models) and all(value == "complete" for value in statuses.values())
    if gate is None:
        reasons.append("Gate G-drift was not evaluated (aligned/drifted summary missing)")
    if not complete and not reasons:
        reasons.append("one or more models are partial")
    status = "complete" if complete and not reasons else "partial"
    return {
        "run_id": "eval-kaggle-merged",
        "config_hash": "kaggle-merged",
        "status": status,
        "models": merged_models,
        "missing_models": missing,
        "benchmarks": benchmarks,
        "missing_benchmarks": {},
        "gate_g_drift": gate or {"status": "not_evaluated"},
        "comparisons": comparisons,
        "auditkit_comparisons": {},
        "human_audit": {"status": "pending_review"},
        "composite_definitions": {},
        "partial_reasons": reasons,
        "notes": (
            "Merged from staged per-candidate Kaggle eval runs. Checkpoints for "
            "ablations were archived after evaluation to fit the scratch disk; "
            "their metrics and manifests remain. This is not fabricated data."
        ),
        "model_status": statuses,
    }


def cmd_merge_eval(args: argparse.Namespace) -> None:
    root = REPO_ROOT / "artifacts" / "eval"
    paths = sorted(
        (path for path in root.glob("eval-*/summary.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        raise RuntimeError("no eval summaries to merge")
    documents = []
    for path in paths:
        if path.parent.name == "eval-kaggle-merged":
            continue
        try:
            documents.append(_read_json(path))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("skipping unreadable eval summary %s: %s", path, exc)
    merged = merge_summaries(documents)
    out_dir = root / "eval-kaggle-merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _write_json(out_dir / "summary.json", merged)
    manifest = {
        "stage": "eval",
        "run_id": "eval-kaggle-merged",
        "config_hash": "kaggle-merged",
        "started_at": _now(),
        "finished_at": _now(),
        "status": merged["status"],
        "notes": "staged Kaggle eval merge; see summary.notes",
    }
    _write_json(out_dir / "manifest.json", manifest)
    _upload([out_dir], "S7 merged eval summary")
    LOG.info("merged eval -> %s (%s)", summary_path, merged["status"])
    for name, status in merged["model_status"].items():
        LOG.info("  %-16s %s", name, status)


# ---------------------------------------------------------------------------
# report / publish / attacks / export
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> None:
    merged = REPO_ROOT / "artifacts" / "eval" / "eval-kaggle-merged" / "summary.json"
    if not merged.exists():
        raise RuntimeError("merged eval summary is missing; run `merge-eval` first")
    _cg("report", "--readme", "README.md")
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        rendered = REPO_ROOT / "results" / "s9" / "README.rendered.md"
        rendered.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(readme, rendered)
    _upload([REPO_ROOT / "results" / "s9"], "S9 report")
    _upload([REPO_ROOT / "artifacts" / "report"], "S9 report artifacts")
    latest = sorted(
        (REPO_ROOT / "results" / "s9").glob("report-*/summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if latest:
        LOG.info("report summary: %s", latest[-1])


def cmd_publish(args: argparse.Namespace) -> None:
    if _checkpoint_for("drifted") is None:
        raise RuntimeError("drifted checkpoint is missing; run the `drift` stage first")
    if _checkpoint_for("cg_project") is None:
        LOG.warning("cg_project weights are missing; rebuilding them before publish")
        cmd_candidates(argparse.Namespace(names="cg_project", extra=False, force=True, overwrite=False))
    _cg("publish")
    _upload([REPO_ROOT / "artifacts" / "publish"], "S10 publish summary")


def _write_attacks_config() -> Path:
    import yaml

    payload = yaml.safe_load((REPO_ROOT / "configs" / "attacks.yaml").read_text(encoding="utf-8"))
    payload["models"] = ["aligned", "cg_project"]
    payload["abliteration"]["modes"] = ["weight_orthogonalize"]
    payload["abliteration"]["fit_harmful"] = 100
    payload["abliteration"]["fit_harmless"] = 100
    path = SCRATCH / "runtime" / "attacks_core.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def cmd_attacks(args: argparse.Namespace) -> None:
    if _hours_left() < 1.25 and not args.force:
        LOG.warning("skipping abliteration: less than 1.25 h left")
        return
    if _checkpoint_for("cg_project") is None:
        raise RuntimeError("abliteration needs the cg_project checkpoint; run candidates first")
    config = _write_attacks_config()
    _cg("attack", "--config", str(config), "--type", "abliterate")
    _upload(
        [REPO_ROOT / "artifacts" / "attacks" / "abliterate"],
        "S8 abliteration stats/manifests",
        patterns=["*/attack_stats.json", "*/manifest.json"],
    )


def cmd_export(args: argparse.Namespace) -> None:
    out = Path("/kaggle/working/cg_outputs") if Path("/kaggle/working").exists() else REPO_ROOT / "kaggle_outputs"
    out.mkdir(parents=True, exist_ok=True)
    wanted = [
        REPO_ROOT / "results" / "s9",
        REPO_ROOT / "artifacts" / "report",
        REPO_ROOT / "artifacts" / "eval" / "eval-kaggle-merged",
        REPO_ROOT / "artifacts" / "data" / "summary.json",
        REPO_ROOT / "results" / "kaggle",
        REPO_ROOT / "README.md",
    ]
    for path in wanted:
        if not path.exists():
            continue
        destination = out / _rel(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)
    discover = _discover_run_dir()
    if discover.exists():
        destination = out / _rel(discover)
        destination.mkdir(parents=True, exist_ok=True)
        for pattern in ("summary.json", "overlap*.json", "manifest.json"):
            for path in discover.glob(pattern):
                shutil.copy2(path, destination / path.name)
        for kind in ("safety", "domain"):
            for pattern in ("circuit_info.json", "faithfulness.json", "selected.json"):
                path = discover / kind / pattern
                if path.exists():
                    (destination / kind).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination / kind / path.name)
    archive = out.parent / "cg_phase4_outputs.zip"
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=out.parent, base_dir=out.name)
    LOG.info("small outputs exported to %s", out)
    LOG.info("zip: %s (%.1f MiB)", archive, archive.stat().st_size / 2**20)


def cmd_status(args: argparse.Namespace) -> None:
    print()
    print("=" * 72)
    print("CircuitGraft Kaggle status")
    print("=" * 72)
    print(f"scratch: {SCRATCH}")
    print(f"cloud:   {CLOUD_REPO}")
    print(f"budget:  {_hours_left():.2f} h left of {BUDGET_HOURS:.1f} h")
    data_summary = REPO_ROOT / "artifacts" / "data" / "summary.json"
    if data_summary.exists():
        try:
            doc = _read_json(data_summary)
            print(f"data:    {doc.get('run_id')} G1={doc.get('gate_g1', {}).get('status')}")
        except Exception:  # noqa: BLE001
            print("data:    unreadable summary")
    else:
        print("data:    missing")
    print(f"drift:   {'ready' if _drift_ready() else 'missing'}")
    for kind in ("safety", "domain"):
        try:
            _, path = _discover_paths(kind)
            print(f"{kind:8s} {'ready' if _circuit_ready(path) else 'missing'}")
        except Exception:  # noqa: BLE001
            print(f"{kind:8s} unknown")
    run_dir = _discover_run_dir()
    print(f"overlap: {'ready' if (run_dir / 'summary.json').exists() else 'missing'}")
    built = [name for name in CANDIDATE_STAGE if _checkpoint_for(name) is not None]
    print(f"built:   {', '.join(built) if built else 'none'}")
    evaluated = [name for name in CANDIDATE_STAGE if _find_model_summary(name) is not None]
    print(f"eval:    {', '.join(evaluated) if evaluated else 'none'}")
    report = sorted((REPO_ROOT / "results" / "s9").glob("report-*/summary.json"))
    print(f"report:  {report[-1] if report else 'missing'}")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="create scratch dirs and import cloud milestones")
    setup.add_argument("--no-import", action="store_true")
    setup.set_defaults(func=cmd_setup)

    data = sub.add_parser("data", help="import and validate the verified data snapshot")
    data.add_argument("--validate-only", action="store_true")
    data.set_defaults(func=cmd_data)

    drift = sub.add_parser("drift", help="train theta_D")
    drift.add_argument("--epochs", type=int, default=int(os.environ.get("CG_DRIFT_EPOCHS", "2")))
    drift.add_argument("--overwrite", action="store_true")
    drift.set_defaults(func=cmd_drift)

    for kind in ("safety", "domain"):
        discover = sub.add_parser(f"discover-{kind}", help=f"discover the {kind} circuit")
        discover.add_argument("--overwrite", action="store_true")
        discover.set_defaults(func=cmd_discover, kind=kind)

    finalize = sub.add_parser("finalize-discover", help="assemble overlap/summary/manifest")
    finalize.add_argument("--overwrite", action="store_true")
    finalize.set_defaults(func=cmd_finalize_discover)

    candidates = sub.add_parser("candidates", help="build baselines/grafts")
    candidates.add_argument("--names", default="")
    candidates.add_argument("--extra", action="store_true")
    candidates.add_argument("--force", action="store_true")
    candidates.add_argument("--overwrite", action="store_true")
    candidates.set_defaults(func=cmd_candidates)

    for name, func in (("eval-refs", cmd_eval_refs), ("eval-candidates", cmd_eval_candidates)):
        ev = sub.add_parser(name, help=name.replace("-", " "))
        ev.add_argument("--judge", default=JUDGE, choices=["string_match", "wildguard"])
        ev.add_argument("--names", default="")
        ev.add_argument("--force", action="store_true")
        ev.set_defaults(func=func)

    merge = sub.add_parser("merge-eval", help="merge staged eval summaries")
    merge.set_defaults(func=cmd_merge_eval)

    report = sub.add_parser("report", help="run S9")
    report.set_defaults(func=cmd_report)

    publish = sub.add_parser("publish", help="run S10")
    publish.set_defaults(func=cmd_publish)

    attacks = sub.add_parser("attacks", help="optional abliteration")
    attacks.add_argument("--force", action="store_true")
    attacks.set_defaults(func=cmd_attacks)

    export = sub.add_parser("export", help="copy small outputs to /kaggle/working")
    export.set_defaults(func=cmd_export)

    status = sub.add_parser("status", help="print progress")
    status.set_defaults(func=cmd_status)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        LOG.error("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - a clean operator-facing error
        LOG.error("%s: %s", type(exc).__name__, exc)
        if os.environ.get("CG_DEBUG"):
            raise
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
