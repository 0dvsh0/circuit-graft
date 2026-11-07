"""S6: run CircuitGraft on real checkpoints and write a loadable model.

The core tensor algorithm lives in :mod:`circuitgraft.graft.graft`. This module
owns the filesystem boundary:

* resolve base / aligned / drifted checkpoints (local directory or HF repo);
* read canonical discovery scores and optional domain Fisher tensors;
* run the graft;
* write a safetensors checkpoint plus ``graft_stats.json`` and a manifest.

The current in-memory loader is intentionally simple. A Tiny Aya-sized run
needs three checkpoints in RAM (roughly 3 x 6 GB in bf16 plus the output), so
use a high-RAM Colab/Kaggle runtime or the future streaming writer. The
manifest records the model references and every score-file hash so a run can
be reproduced or invalidated when its circuit changes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from circuitgraft.components import Cohere2Layout
from circuitgraft.config import BaseConfig, GraftConfig, config_hash
from circuitgraft.discover.overlap import load_scores
from circuitgraft.graft.graft import GraftOutput, graft_state_dict
from circuitgraft.registry import (
    REPO_ROOT,
    RunTimer,
    dump_json,
    hash_paths,
    load_json,
    new_manifest,
    resolve_stage_file,
    sha256_file,
    should_skip,
)

logger = logging.getLogger(__name__)

_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "preprocessor_config.json",
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _local_model_path(ref: str | Path) -> Path | None:
    """Resolve a local checkpoint path, accepting repo-root-relative values."""
    path = Path(ref)
    candidates = [path] if path.is_absolute() else [path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def materialize_model(ref: str | Path) -> Path:
    """Return a local directory for a checkpoint reference.

    Local directories/files are returned as-is. HF repo ids are downloaded
    with ``snapshot_download`` using only the files needed for weights and
    tokenizer/config metadata. Gated-repo errors surface unchanged so the
    operator sees the licence message.
    """
    local = _local_model_path(ref)
    if local is not None:
        return local if local.is_dir() else local.parent

    from huggingface_hub import snapshot_download

    logger.info("downloading checkpoint files for %s", ref)
    return Path(
        snapshot_download(
            repo_id=str(ref),
            token=os.environ.get("HF_TOKEN"),
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.txt",
                "*.model",
                "*.jinja",
                "tokenizer*",
                "vocab*",
                "merges*",
                "special_tokens*",
            ],
        )
    )


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a safetensors or PyTorch checkpoint from a file or directory."""
    path = Path(path)
    if path.is_file():
        if path.suffix == ".safetensors":
            return dict(load_file(path))
        if path.suffix in (".bin", ".pt", ".pth"):
            return dict(torch.load(path, map_location="cpu", weights_only=True))
        raise ValueError(f"unsupported checkpoint file: {path}")

    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"{index_path}: missing weight_map")
        shards = sorted(set(weight_map.values()))
        state: dict[str, torch.Tensor] = {}
        for shard in shards:
            shard_path = path / str(shard)
            if not shard_path.exists():
                raise FileNotFoundError(f"{index_path} references missing shard {shard_path}")
            state.update(load_file(shard_path))
        return state

    single = path / "model.safetensors"
    if single.exists():
        return dict(load_file(single))

    shards = sorted(path.glob("*.safetensors"))
    if shards:
        state = {}
        for shard in shards:
            state.update(load_file(shard))
        return state

    binary = path / "pytorch_model.bin"
    if binary.exists():
        return dict(torch.load(binary, map_location="cpu", weights_only=True))

    raise FileNotFoundError(
        f"no supported checkpoint found in {path} "
        "(expected model.safetensors, a sharded index, or pytorch_model.bin)"
    )


def load_layout(ref: str | Path) -> Cohere2Layout:
    """Read the architecture from the checkpoint config, never from a guess."""
    local = _local_model_path(ref)
    if local is not None and local.is_dir() and (local / "config.json").exists():
        config = json.loads((local / "config.json").read_text(encoding="utf-8"))
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            str(ref), token=os.environ.get("HF_TOKEN")
        ).to_dict()

    required = ("hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{ref}: config is missing {missing}")
    if config.get("model_type") not in (None, "cohere2"):
        raise ValueError(f"{ref}: expected a cohere2 checkpoint, got {config.get('model_type')!r}")
    return Cohere2Layout.from_config(type("Config", (), config)())


def _hash_inputs(cfg: GraftConfig, base: BaseConfig) -> dict[str, str]:
    local_files = [
        _repo_path(resolve_stage_file(path))
        for path in (cfg.safety_scores, cfg.domain_scores, cfg.fisher_path)
        if path and _repo_path(resolve_stage_file(path)).exists()
    ]
    hashes = hash_paths(local_files) if local_files else {}
    for label, ref in (
        ("model:base", base.models.base),
        ("model:aligned", base.models.aligned),
        ("model:drifted", cfg.models.drifted or ""),
    ):
        if not ref:
            continue
        local = _local_model_path(ref)
        if local is not None and local.is_file():
            hashes[label] = sha256_file(local)
        elif local is not None and local.is_dir() and (local / "config.json").exists():
            # Hashing multi-gigabyte shards on every run is wasteful; the config
            # hash pins the architecture and the manifest records the directory.
            hashes[label] = f"local-dir:{local.resolve()}:{sha256_file(local / 'config.json')}"
        else:
            hashes[label] = f"ref:{ref}"
    return hashes


def _save_checkpoint(state: Mapping[str, torch.Tensor], out_dir: Path, source_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if isinstance(tensor, torch.Tensor)
    }
    model_path = out_dir / "model.safetensors"
    save_file(tensors, model_path)
    for filename in _METADATA_FILES:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, out_dir / filename)
    return model_path


def _load_fisher(path: str | Path) -> dict[str, torch.Tensor]:
    path = _repo_path(resolve_stage_file(path))
    if not path.exists():
        raise FileNotFoundError(f"fisher_gate is enabled but fisher_path does not exist: {path}")
    return dict(load_file(path))


def run_graft(
    cfg: GraftConfig,
    base: BaseConfig,
    *,
    layout: Cohere2Layout | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one graft configuration and return its artifact summary."""
    if not cfg.safety_scores:
        raise ValueError("config.safety_scores is required")
    if not cfg.domain_scores:
        raise ValueError("config.domain_scores is required")
    if not cfg.models.drifted:
        raise ValueError(
            "models.drifted is empty; run `cg drift` first or set models.drifted "
            "to the merged specialist checkpoint"
        )

    cfg_hash = config_hash(cfg)
    run_id = f"{cfg.name}-{cfg_hash}"
    out_dir = _repo_path(cfg.output_dir) / run_id
    input_hashes = _hash_inputs(cfg, base)

    existing = out_dir / "graft_stats.json"
    if not overwrite and should_skip(out_dir, cfg_hash, input_hashes) and existing.exists():
        logger.info("graft run is current; skipping %s", out_dir)
        return {
            "run_id": run_id,
            "output_dir": str(out_dir),
            "skipped": True,
            "stats": load_json(existing),
        }

    if layout is None:
        layout = load_layout(base.models.base)

    safety_scores = load_scores(
        _repo_path(resolve_stage_file(cfg.safety_scores)), layout=layout
    )
    domain_scores = load_scores(
        _repo_path(resolve_stage_file(cfg.domain_scores)), layout=layout
    )
    if not safety_scores:
        raise ValueError(f"safety score file is empty: {cfg.safety_scores}")
    if not domain_scores:
        raise ValueError(f"domain score file is empty: {cfg.domain_scores}")

    fisher = _load_fisher(cfg.fisher_path) if cfg.overlap_strategy == "fisher_gate" else None

    logger.info("loading base checkpoint %s", base.models.base)
    base_state = load_state_dict(materialize_model(base.models.base))
    logger.info("loading aligned checkpoint %s", base.models.aligned)
    aligned_state = load_state_dict(materialize_model(base.models.aligned))
    logger.info("loading drifted checkpoint %s", cfg.models.drifted)
    drifted_dir = materialize_model(cfg.models.drifted)
    drifted_state = load_state_dict(drifted_dir)

    manifest = new_manifest("graft", run_id, cfg_hash, seed=cfg.seed)
    manifest.inputs = input_hashes
    manifest.notes = (
        f"selection={cfg.selection}; overlap_strategy={cfg.overlap_strategy}; "
        f"delta_source={cfg.delta_source}; k_s={cfg.k_s}; k_d={cfg.k_d}; "
        f"alpha={cfg.alpha}; alpha_o={cfg.alpha_o}"
    )

    with RunTimer(manifest, out_dir):
        output: GraftOutput = graft_state_dict(
            base=base_state,
            aligned=aligned_state,
            drifted=drifted_state,
            layout=layout,
            cfg=cfg,
            safety_scores=safety_scores,
            domain_scores=domain_scores,
            domain_fisher=fisher,
        )
        model_path = _save_checkpoint(output.state_dict, out_dir, drifted_dir)
        stats_path = dump_json(out_dir / "graft_stats.json", output.stats.as_dict())
        manifest.outputs = {
            "model.safetensors": sha256_file(model_path),
            "graft_stats.json": sha256_file(stats_path),
        }

    return {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "model": str(model_path),
        "skipped": False,
        "stats": output.stats.as_dict(),
    }
