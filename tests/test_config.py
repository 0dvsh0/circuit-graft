"""Configuration schema, inheritance, expansion and hash tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from circuitgraft.config import (
    AttacksConfig,
    BaseConfig,
    BaselineConfig,
    DataConfig,
    DiscoverConfig,
    DriftConfig,
    EvalConfig,
    GraftConfig,
    _deep_merge,
    config_hash,
    load_config,
    load_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_SCHEMAS = {
    "configs/base.yaml": (BaseConfig, None),
    "configs/data.yaml": (DataConfig, "configs/base.yaml"),
    "configs/drift.yaml": (DriftConfig, "configs/base.yaml"),
    "configs/discover.yaml": (DiscoverConfig, "configs/base.yaml"),
    "configs/eval.yaml": (EvalConfig, "configs/base.yaml"),
    "configs/attacks.yaml": (AttacksConfig, "configs/base.yaml"),
}


def all_shipped_configs() -> list[tuple[Path, object, str | None]]:
    entries: list[tuple[Path, object, str | None]] = []
    for relative, (schema, base_path) in CONFIG_SCHEMAS.items():
        entries.append((REPO_ROOT / relative, schema, base_path))
    for path in sorted((REPO_ROOT / "configs" / "graft").glob("*.yaml")):
        entries.append((path, GraftConfig, "configs/base.yaml"))
    for path in sorted((REPO_ROOT / "configs" / "baselines").glob("*.yaml")):
        entries.append((path, BaselineConfig, "configs/base.yaml"))
    return entries


def test_every_shipped_config_validates() -> None:
    entries = all_shipped_configs()
    assert len(entries) >= 18
    for path, schema, base_path in entries:
        cfg = load_config(path, schema, base_path=base_path)  # type: ignore[arg-type]
        assert cfg is not None, path


def test_base_config_expands_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("HF_ORG", "unit-test-org")
    cfg = load_config("configs/base.yaml", BaseConfig, base_path=None)
    assert cfg.hf_org == "unit-test-org"
    assert "${" not in cfg.hf_org


def test_yaml_expansion_supports_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CG_TEST_UNSET", raising=False)
    monkeypatch.setenv("CG_TEST_SET", "present")
    path = tmp_path / "expand.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "set": "${CG_TEST_SET}",
                "default": "${CG_TEST_UNSET:-fallback}",
                "empty_default": "${CG_TEST_UNSET:-}",
                "nested": ["${CG_TEST_SET}", {"key": "${CG_TEST_UNSET:-x}"}],
            }
        ),
        encoding="utf-8",
    )
    data = load_yaml(path)
    assert data == {
        "set": "present",
        "default": "fallback",
        "empty_default": "",
        "nested": ["present", {"key": "x"}],
    }


def test_deep_merge_is_recursive_and_non_mutating() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 3}}
    merged = _deep_merge(base, override)
    assert merged == {"a": 1, "nested": {"x": 1, "y": 3}}
    assert base == {"a": 1, "nested": {"x": 1, "y": 2}}


def test_config_hash_is_stable_and_sensitive() -> None:
    first = load_config("configs/graft/cg_project.yaml", GraftConfig)
    second = load_config("configs/graft/cg_project.yaml", GraftConfig)
    assert config_hash(first) == config_hash(second)
    assert len(config_hash(first)) == 16

    changed = load_config(
        "configs/graft/cg_project.yaml", GraftConfig, extra={"alpha": 0.75}
    )
    assert config_hash(first) != config_hash(changed)


def test_graft_config_validation() -> None:
    with pytest.raises(ValidationError, match="fractions"):
        GraftConfig(name="bad", k_s=1.5)
    with pytest.raises(ValidationError, match="must be > 0"):
        GraftConfig(name="bad", alpha=0.0)
    with pytest.raises(ValidationError, match="fisher_path"):
        GraftConfig(name="bad", overlap_strategy="fisher_gate")
    with pytest.raises(ValidationError, match="random_reference"):
        GraftConfig(name="bad", selection="random_matched")
    with pytest.raises(ValidationError):
        GraftConfig(name="bad", unknown_field=1)  # type: ignore[call-arg]


def test_base_config_rejects_identical_base_and_aligned() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        BaseConfig(
            hf_org="x",
            models={"base": "org/model", "aligned": "org/model"},
        )


def test_invalid_yaml_shape_is_rejected(tmp_path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_yaml(path)
