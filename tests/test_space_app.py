"""Unit tests for the data-driven Space loaders.

Gradio is imported lazily by ``space/app.py``, so these tests exercise the
loader/render helpers without installing the UI dependency.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SPACE_APP = Path(__file__).resolve().parents[1] / "space" / "app.py"


@pytest.fixture(scope="module")
def space_app() -> ModuleType:
    spec = importlib.util.spec_from_file_location("circuitgraft_space_app", SPACE_APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_bundle(
    root: Path,
    name: str,
    *,
    finished_at: str = "2026-11-08T00:00:00Z",
    summary: dict[str, Any] | None = None,
    tables: dict[str, str] | None = None,
    plots: tuple[str, ...] = (),
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    _write_json(
        directory / "summary.json",
        summary if summary is not None else {"run_id": name, "status": "complete"},
    )
    _write_json(directory / "manifest.json", {"finished_at": finished_at})
    for filename, text in (tables or {}).items():
        _write_text(directory / filename, text)
    for filename in plots:
        _write_text(directory / filename, "png")
    return directory


def test_find_latest_report_uses_manifest_timestamp(space_app: ModuleType, tmp_path: Path) -> None:
    newer = _write_bundle(tmp_path, "report-200", finished_at="2026-11-08T12:00:00Z")
    _write_bundle(tmp_path, "report-100", finished_at="2026-11-07T12:00:00Z")
    # A directory without summary.json is not a bundle and must not win.
    (tmp_path / "report-999").mkdir()
    _write_json(tmp_path / "report-999" / "manifest.json", {"finished_at": "2026-12-01"})

    assert space_app.find_latest_report(tmp_path) == newer


def test_find_latest_report_returns_none_without_bundles(
    space_app: ModuleType, tmp_path: Path
) -> None:
    assert space_app.find_latest_report(tmp_path) is None
    assert space_app.find_latest_report(tmp_path / "missing") is None


def test_find_latest_report_resolves_root(space_app: ModuleType, tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "results" / "s9", "report-123")
    assert space_app.find_latest_report(root=tmp_path) == bundle


def test_load_bundle_reads_tables_and_plots(space_app: ModuleType, tmp_path: Path) -> None:
    directory = _write_bundle(
        tmp_path,
        "report-42",
        summary={
            "run_id": "eval-abc",
            "status": "partial",
            "partial_reasons": ["no S8 attack statistics"],
            "sources": {"eval": "eval-abc", "overlap": "discover-def"},
        },
        tables={"table_main.md": "# Main\n", "table_attacks.md": "# Attacks\n"},
        plots=("pareto_frontier.png", "circuit_overlap_heatmap.png"),
    )

    bundle = space_app.load_bundle(directory)
    assert bundle is not None
    assert bundle.run_id == "eval-abc"
    assert bundle.status == "partial"
    assert bundle.reasons == ["no S8 attack statistics"]
    assert bundle.table("table_main") == "# Main\n"
    assert bundle.table("table_attacks.md") == "# Attacks\n"
    assert bundle.table("table_overlap") == "_Table not generated._"
    assert set(bundle.plots) == {"pareto_frontier.png", "circuit_overlap_heatmap.png"}


def test_load_bundle_rejects_missing_or_invalid_summary(
    space_app: ModuleType, tmp_path: Path
) -> None:
    empty = tmp_path / "report-empty"
    empty.mkdir()
    assert space_app.load_bundle(empty) is None

    invalid = _write_bundle(tmp_path, "report-invalid")
    (invalid / "summary.json").write_text("[]", encoding="utf-8")
    assert space_app.load_bundle(invalid) is None

    malformed = _write_bundle(tmp_path, "report-malformed")
    (malformed / "summary.json").write_text("{not json", encoding="utf-8")
    assert space_app.load_bundle(malformed) is None


def test_load_bundle_uses_environment_override(
    space_app: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _write_bundle(tmp_path, "report-env", summary={"run_id": "from-env"})
    monkeypatch.setenv("CIRCUITGRAFT_REPORT_DIR", str(directory))

    bundle = space_app.load_bundle()
    assert bundle is not None
    assert bundle.run_id == "from-env"


def test_load_cached_responses_missing_invalid_and_valid(
    space_app: ModuleType, tmp_path: Path
) -> None:
    assert space_app.load_cached_responses(tmp_path / "missing.json") == {}

    invalid = _write_text(tmp_path / "invalid.json", "{not json")
    assert space_app.load_cached_responses(invalid) == {}

    wrong_shape = _write_json(tmp_path / "list.json", [{"id": "x"}])
    assert space_app.load_cached_responses(wrong_shape) == {}

    payload = {"models": ["aligned"], "responses": {"aligned": {"p1": "ok"}}}
    valid = _write_json(tmp_path / "valid.json", payload)
    assert space_app.load_cached_responses(valid) == payload


def test_prompt_choices_labels_and_ids(space_app: ModuleType) -> None:
    payload = {
        "prompts": [
            {"id": "harmful-1", "label": "Harmful request", "category": "harmful"},
            {"id": "med-1"},
            {"label": "No id"},
            "not a prompt",
        ]
    }

    assert space_app.prompt_choices(payload) == [
        ("Harmful request [harmful]", "harmful-1"),
        ("med-1", "med-1"),
        ("No id", "prompt-2"),
    ]
    assert space_app.prompt_choices({}) == []


def test_model_names_prefers_explicit_list(space_app: ModuleType) -> None:
    assert space_app.model_names({"models": ["b", "a"]}) == ["b", "a"]
    assert space_app.model_names({"responses": {"b": {}, "a": {}}}) == ["a", "b"]
    assert space_app.model_names({}) == []


def test_response_for_handles_missing_keys(space_app: ModuleType) -> None:
    payload = {"responses": {"aligned": {"p1": "I can't help with that."}}}

    assert space_app.response_for(payload, "p1", "aligned") == "I can't help with that."
    assert space_app.response_for(payload, "p2", "aligned") == ""
    assert space_app.response_for(payload, "p1", "drifted") == ""
    assert space_app.response_for({}, "p1", "aligned") == ""


def test_status_markdown_without_bundle(space_app: ModuleType) -> None:
    text = space_app.status_markdown(None)
    assert "No tracked report bundle yet" in text
    assert "cg report" in text
    assert "placeholder metrics" in text


def test_status_markdown_reports_partial_sources(
    space_app: ModuleType, tmp_path: Path
) -> None:
    directory = _write_bundle(
        tmp_path,
        "report-partial",
        summary={
            "run_id": "eval-abc",
            "status": "partial",
            "partial_reasons": ["no S8 attack statistics"],
            "sources": {
                "eval": "eval-abc",
                "overlap": "discover-def",
                "attacks": {"run1": {}, "run2": {}},
            },
        },
    )

    bundle = space_app.load_bundle(directory)
    assert bundle is not None
    text = space_app.status_markdown(bundle)
    assert "`eval-abc`" in text
    assert "`partial`" in text
    assert "**eval:** `eval-abc`" in text
    assert "**overlap:** `discover-def`" in text
    assert "**attacks:** 2 run(s)" in text
    assert "no S8 attack statistics" in text
    assert "All required S7/S4/S8 sources were present" not in text


def test_status_markdown_warns_when_status_is_not_complete(
    space_app: ModuleType, tmp_path: Path
) -> None:
    directory = _write_bundle(
        tmp_path,
        "report-unknown",
        summary={"run_id": "eval-unknown", "status": "unknown", "partial_reasons": []},
    )

    bundle = space_app.load_bundle(directory)
    assert bundle is not None
    text = space_app.status_markdown(bundle)
    assert "status is not `complete`" in text
    assert "All required S7/S4/S8 sources were present" not in text
