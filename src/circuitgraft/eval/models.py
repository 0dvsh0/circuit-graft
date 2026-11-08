"""Resolve logical evaluation names to checkpoints and load them.

The eval matrix uses short names (``aligned``, ``drifted``, ``cg_project``,
``resta``...). Explicit ``model_paths`` entries win; otherwise the resolver
searches the stage artifact trees for a completed checkpoint. A missing model
is an error with the searched locations, never a silent skip.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from circuitgraft.config import BaseConfig, EvalConfig
from circuitgraft.registry import REPO_ROOT, existing_manifest, load_json, sha256_file

logger = logging.getLogger(__name__)

_WEIGHT_PATTERNS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def local_path(ref: str | Path) -> Path | None:
    path = Path(ref)
    candidates = [path] if path.is_absolute() else [path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


_LOCAL_ROOTS = frozenset({"artifacts", "results", "checkpoints", "data", "models", "outputs"})
_LOCAL_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".json")


def is_local_ref(ref: str | Path) -> bool:
    """Return True when ``ref`` names a filesystem path rather than a Hub repo id.

    Hugging Face repo ids are ``namespace/name`` or a bare ``name``. Project
    artifacts are written below known roots (``artifacts/``, ``results/``...)
    and checkpoint files carry a checkpoint suffix, so a missing one is a
    missing stage artifact, not a repository to download.
    """
    text = str(ref).strip()
    if not text:
        return False
    path = Path(text)
    if path.is_absolute() or text.startswith((".", "~")):
        return True
    if text.endswith(_LOCAL_SUFFIXES):
        return True
    first = text.replace("\\", "/").split("/", 1)[0]
    return first in _LOCAL_ROOTS


def local_path_or_raise(ref: str | Path, *, context: str = "model") -> Path | None:
    """Resolve a local reference; fail loudly instead of falling back to the Hub.

    A missing path-like reference such as ``artifacts/drift/.../merged`` used
    to reach ``AutoTokenizer.from_pretrained``/``snapshot_download`` and surface
    as ``HFValidationError``. The actionable error is that the producing stage
    has not run or did not write its checkpoint.
    """
    local = local_path(ref)
    if local is not None:
        return local
    if is_local_ref(ref):
        raise FileNotFoundError(
            f"{context} {str(ref)!r} looks like a local path but does not exist. "
            "Run the stage that produces it, or set the reference to a Hugging Face repo id."
        )
    return None


def has_weights(path: str | Path) -> bool:
    directory = Path(path)
    return any((directory / name).exists() for name in _WEIGHT_PATTERNS)


def checkpoint_dir(candidate: Path) -> Path | None:
    """Return the actual checkpoint directory inside a stage run directory."""
    merged = candidate / "merged"
    if merged.is_dir() and has_weights(merged):
        return merged
    if candidate.is_dir() and has_weights(candidate):
        return candidate
    return None


def _manifest_allows(candidate: Path) -> bool:
    manifest = existing_manifest(candidate)
    if manifest is None:
        return True
    return manifest.status == "complete"


def _candidate_patterns(name: str) -> list[str]:
    return [
        f"artifacts/graft/{name}-*",
        f"artifacts/graft/{name}",
        f"artifacts/baselines/{name}*",
        f"artifacts/attacks/*{name}*",
        f"artifacts/checkpoints/{name}*",
        f"artifacts/{name}*",
        f"results/checkpoints/{name}*",
    ]


def resolve_checkpoint(name: str, cfg: EvalConfig, base: BaseConfig) -> str:
    """Resolve a matrix name to a local path or HF repo id."""
    explicit = cfg.model_paths.get(name)
    if explicit:
        local = local_path_or_raise(explicit, context=f"eval.model_paths[{name!r}]")
        return str(local.resolve()) if local is not None else explicit

    special: dict[str, str | None] = {
        "base": base.models.base,
        "aligned": base.models.aligned,
        "drifted": base.models.drifted,
    }
    if name in special:
        ref = special[name]
        if not ref:
            raise FileNotFoundError(f"models.{name} is not set in the config")
        local = local_path_or_raise(ref, context=f"models.{name}")
        return str(local.resolve()) if local is not None else ref

    searched: list[str] = []
    candidates: list[Path] = []
    for pattern in _candidate_patterns(name):
        searched.append(pattern)
        candidates.extend(REPO_ROOT.glob(pattern))
    candidates = sorted(
        {path for path in candidates if path.is_dir()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        checkpoint = checkpoint_dir(candidate)
        if checkpoint is None or not _manifest_allows(candidate):
            continue
        return str(checkpoint.resolve())

    raise FileNotFoundError(
        f"could not resolve model {name!r}; set eval.model_paths[{name!r}] or run "
        f"the stage that produces it. Searched: {', '.join(searched)}"
    )


def model_fingerprint(ref: str) -> str:
    """Cheap content fingerprint: paths, configs and weight names/sizes."""
    local = local_path(ref)
    if local is None:
        return f"ref:{ref}"
    parts = [f"path:{local.resolve()}"]
    if local.is_file():
        parts.append(f"size:{local.stat().st_size}")
        return "|".join(parts)
    for name in ("config.json", "adapter_config.json", "model.safetensors.index.json"):
        path = local / name
        if path.is_file():
            parts.append(f"{name}:{sha256_file(path)}")
    weights = sorted(
        path for path in local.iterdir() if path.is_file() and path.suffix in {".safetensors", ".bin"}
    )
    if weights:
        parts.append(
            "weights:"
            + ",".join(f"{path.name}:{path.stat().st_size}" for path in weights)
        )
    return "|".join(parts)


def resolve_models(
    cfg: EvalConfig, base: BaseConfig, *, all_models: bool = False
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve every requested model; return (resolved, errors)."""
    names = list(cfg.models)
    if all_models:
        for name in cfg.model_paths:
            if name not in names:
                names.append(name)
    resolved: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name in names:
        try:
            resolved[name] = resolve_checkpoint(name, cfg, base)
        except Exception as exc:  # noqa: BLE001 - recorded in the summary
            errors[name] = f"{type(exc).__name__}: {exc}"
            logger.warning("model %s is unavailable: %s", name, errors[name])
    return resolved, errors


def _resolve_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model_and_tokenizer(
    ref: str,
    *,
    dtype: str = "bfloat16",
    device: str = "auto",
) -> tuple[Any, Any, str]:
    """Load a merged checkpoint or LoRA adapter for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    resolved_device = _resolve_device(device)
    local = local_path_or_raise(ref, context="model reference")

    adapter_config = None
    if local is not None and local.is_dir() and (local / "adapter_config.json").exists():
        adapter_config = load_json(local / "adapter_config.json")
        base_ref = str(adapter_config.get("base_model_name_or_path") or "")
        if not base_ref:
            raise ValueError(
                f"{local}: adapter_config.json has no base_model_name_or_path"
            )
        tokenizer_source = str(local)
        model_source = base_ref
    else:
        tokenizer_source = str(local) if local is not None else ref
        model_source = tokenizer_source

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, token=token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=_torch_dtype(dtype),
        token=token,
        trust_remote_code=True,
    )
    if adapter_config is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(local))
    model.to(resolved_device)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, resolved_device


def generation_model_info(ref: str) -> dict[str, Any]:
    """Metadata recorded in every generation row."""
    local = local_path(ref)
    info: dict[str, Any] = {"ref": ref, "fingerprint": model_fingerprint(ref)}
    if local is not None and local.is_dir():
        config_path = local / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                info["model_type"] = config.get("model_type")
                info["hidden_size"] = config.get("hidden_size")
                info["num_hidden_layers"] = config.get("num_hidden_layers")
            except Exception:  # noqa: BLE001 - metadata is advisory
                pass
    return info
