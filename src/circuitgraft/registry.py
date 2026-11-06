"""Run manifests: lineage, idempotency and HF publication helpers.

Every stage writes ``artifacts/<stage>/<run_id>/manifest.json``. A stage may skip
work only when the config hash *and* every declared input hash match the stored
manifest, so a changed dataset or circuit always invalidates downstream caches.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    run_id: str
    config_hash: str
    git_sha: str | None = None
    git_dirty: bool | None = None
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=dict)
    seed: int | None = None
    started_at: str
    finished_at: str | None = None
    wall_clock_s: float | None = None
    platform: str = Field(default_factory=platform.platform)
    python: str = Field(default_factory=lambda: sys.version.split()[0])
    device: str | None = None
    gpu: str | None = None
    status: str = "running"
    notes: str | None = None

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, directory: str | Path) -> Manifest:
        path = Path(directory) / "manifest.json"
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def is_current(self, config_hash: str, input_hashes: dict[str, str]) -> bool:
        return (
            self.status == "complete"
            and self.config_hash == config_hash
            and self.inputs == input_hashes
        )


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_paths(paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash files, or directories recursively (path -> sha256 of file bytes)."""
    result: dict[str, str] = {}
    for item in paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            for file in sorted(p for p in path.rglob("*") if p.is_file()):
                result[str(file.relative_to(REPO_ROOT) if file.is_relative_to(REPO_ROOT) else file)] = (
                    sha256_file(file)
                )
        else:
            key = str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
            result[key] = sha256_file(path)
    return result


def git_sha() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, None


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "safetune",
        "circuitkit",
        "auditkit",
        "curatorkit",
        "transformer_lens",
        "huggingface_hub",
        "safetensors",
        "numpy",
    ):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - a missing optional dep is information
            versions[name] = "not-installed"
    return versions


def device_info() -> tuple[str | None, str | None]:
    device = None
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            gpu = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except Exception:  # noqa: BLE001
        pass
    return device, gpu


def new_manifest(
    stage: str,
    run_id: str,
    config_hash: str,
    *,
    inputs: Iterable[str | Path] = (),
    seed: int | None = None,
    notes: str | None = None,
) -> Manifest:
    sha, dirty = git_sha()
    device, gpu = device_info()
    return Manifest(
        stage=stage,
        run_id=run_id,
        config_hash=config_hash,
        git_sha=sha,
        git_dirty=dirty,
        inputs=hash_paths(inputs) if inputs else {},
        library_versions=library_versions(),
        seed=seed,
        started_at=datetime.now(UTC).isoformat(),
        device=device,
        gpu=gpu,
        notes=notes,
    )


class RunTimer:
    """Context manager that stamps a manifest when the run exits."""

    def __init__(self, manifest: Manifest, directory: str | Path, *, status: str = "complete"):
        self.manifest = manifest
        self.directory = Path(directory)
        self.status = status
        self.started = time.perf_counter()

    def __enter__(self) -> Manifest:
        self.manifest.write(self.directory)
        return self.manifest

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.manifest.finished_at = datetime.now(UTC).isoformat()
        self.manifest.wall_clock_s = round(time.perf_counter() - self.started, 3)
        if exc_type is not None:
            self.manifest.status = "failed"
            self.manifest.notes = f"{type(exc).__name__}: {exc}"
        else:
            self.manifest.status = self.status
        self.manifest.write(self.directory)


def existing_manifest(directory: str | Path) -> Manifest | None:
    path = Path(directory) / "manifest.json"
    if not path.exists():
        return None
    try:
        return Manifest.read(directory)
    except Exception:  # noqa: BLE001 - corrupted manifests should not block reruns
        return None


def should_skip(
    directory: str | Path, config_hash: str, input_hashes: dict[str, str]
) -> bool:
    manifest = existing_manifest(directory)
    return bool(manifest and manifest.is_current(config_hash, input_hashes))


def prepare_run_dir(path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def push_folder_to_hub(
    folder: str | Path,
    repo_id: str,
    *,
    repo_type: str = "model",
    private: bool = False,
    commit_message: str = "CircuitGraft artifact upload",
) -> str:
    """Upload a folder with huggingface_hub; returns the commit URL."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )
    return commit.commit_url


def dump_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
