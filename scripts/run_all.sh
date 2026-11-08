#!/usr/bin/env bash
# CircuitGraft end-to-end runner.
#
# Modes
#   --check   Environment, every config, pytest and ruff. No model downloads.
#   --smoke   CPU mechanics: D1 proxy smoke, S1 smoke data, S9 into a scratch
#             README and an S10 dry run. Safe to run on a laptop.
#   --full    Phase 4 pipeline on a CUDA machine: data -> drift -> audit ->
#             discover -> baselines -> graft -> eval -> attacks -> eval-attacks
#             -> report -> publish.
#
# Examples
#   ./scripts/run_all.sh --check
#   ./scripts/run_all.sh --smoke
#   ./scripts/run_all.sh --full --device cuda
#   ./scripts/run_all.sh --full --no-publish --no-attacks
#   ./scripts/run_all.sh --full --no-attack-eval
#
# The script never invents metrics. Each stage reads the manifests and artifacts
# written by the previous stage and fails loudly when an input is missing.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="full"
DRY_RUN=0
DEVICE="cuda"
PUBLISH=1
ATTACKS=1
ATTACK_EVAL=1
SWEEP_BASELINES=0
SMOKE_DIR="artifacts/run_all_smoke"
PYTHON="${PYTHON:-}"

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

need_value() {
    if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) MODE="check" ;;
        --smoke) MODE="smoke"; DEVICE="cpu" ;;
        --full) MODE="full" ;;
        --dry-run) DRY_RUN=1 ;;
        --no-publish) PUBLISH=0 ;;
        --no-attacks) ATTACKS=0 ;;
        --no-attack-eval) ATTACK_EVAL=0 ;;
        --sweep-baselines) SWEEP_BASELINES=1 ;;
        --device)
            need_value "$@"; DEVICE="$2"; shift ;;
        --smoke-dir)
            need_value "$@"; SMOKE_DIR="$2"; shift ;;
        --python)
            need_value "$@"; PYTHON="$2"; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2 ;;
    esac
    shift
done

if [[ -z "$PYTHON" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        PYTHON="$ROOT/.venv/bin/python"
    elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
        PYTHON="$ROOT/.venv/Scripts/python.exe"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    else
        PYTHON="$(command -v python)"
    fi
fi

CG=("$PYTHON" -m circuitgraft.cli)

run() {
    if (( DRY_RUN )); then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    printf '\n==> '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}

config_files() {
    find configs -type f -name '*.yaml' | sort
}

check_environment() {
    run "${CG[@]}" doctor
    if [[ "$MODE" == "full" ]]; then
        run "${CG[@]}" doctor --check-models --json-out results/d1/environment.json
    fi
}

check_configs() {
    while IFS= read -r config; do
        run "${CG[@]}" config-check "$config"
    done < <(config_files)
}

check_tests() {
    run "$PYTHON" -m pytest -q
    if "$PYTHON" -m ruff --version >/dev/null 2>&1; then
        run "$PYTHON" -m ruff check src tests scripts space
    else
        echo "warning: ruff is not installed in $PYTHON; skipping lint" >&2
    fi
}

require_cuda() {
    if [[ "$DEVICE" == "cpu" ]]; then
        return 0
    fi
    "$PYTHON" - "$DEVICE" <<'PY'
import sys

import torch

requested = sys.argv[1]
if not torch.cuda.is_available():
    raise SystemExit(
        f"CUDA is required for --device {requested}, but torch.cuda.is_available() is false. "
        "Use --smoke for the CPU mechanics run."
    )
print(f"CUDA available: {torch.cuda.get_device_name(0)}")
PY
}

run_check() {
    check_environment
    check_configs
    check_tests
    echo
    if (( DRY_RUN )); then
        echo "check: dry run"
    else
        echo "check: ok"
    fi
}

run_smoke() {
    run mkdir -p "$SMOKE_DIR"
    check_configs
    check_tests

    run "$PYTHON" scripts/d1_smoke.py \
        --proxy --device cpu --max-examples 4 \
        --output-dir "$SMOKE_DIR/d1"

    run "${CG[@]}" data --config configs/smoke/data.yaml

    run cp README.md "$SMOKE_DIR/README.md"
    run "${CG[@]}" report \
        --output-dir "$SMOKE_DIR/report" \
        --readme "$SMOKE_DIR/README.md" \
        --results-out "$SMOKE_DIR/results"

    run "${CG[@]}" publish --dry-run --output-dir "$SMOKE_DIR/publish"

    echo
    if (( DRY_RUN )); then
        echo "smoke: dry run -> $SMOKE_DIR"
    else
        echo "smoke: ok -> $SMOKE_DIR"
    fi
    echo "  report:  $SMOKE_DIR/report/summary.md"
    echo "  README:  $SMOKE_DIR/README.md"
    echo "  publish: $SMOKE_DIR/publish/summary.json"
}

run_full() {
    if (( ! DRY_RUN )); then
        require_cuda
    fi
    check_environment
    check_configs

    run "${CG[@]}" data --config configs/data.yaml
    run "${CG[@]}" drift --config configs/drift.yaml
    run "${CG[@]}" audit --config configs/eval.yaml --models aligned,drifted
    run "${CG[@]}" discover --config configs/discover.yaml

    for config in configs/baselines/*.yaml; do
        if (( SWEEP_BASELINES )); then
            run "${CG[@]}" baselines --config "$config" --sweep
        else
            run "${CG[@]}" baselines --config "$config"
        fi
    done

    for config in configs/graft/*.yaml; do
        run "${CG[@]}" graft --config "$config"
    done

    run "${CG[@]}" eval --config configs/eval.yaml --all

    if (( ATTACKS )); then
        run "${CG[@]}" attack --config configs/attacks.yaml --type all
        if (( ATTACK_EVAL )); then
            run "${CG[@]}" eval-attacks --config configs/eval.yaml
        fi
    fi

    run "${CG[@]}" report

    if (( PUBLISH )); then
        run "${CG[@]}" publish
    fi

    echo
    if (( DRY_RUN )); then
        echo "full: dry run"
    else
        echo "full: ok"
    fi
    echo "  report:  artifacts/report/summary.md"
    echo "  results: results/s9/"
    if (( ATTACKS && ATTACK_EVAL )); then
        echo "  attack eval: artifacts/attacks/eval/summary.json"
    fi
}

case "$MODE" in
    check) run_check ;;
    smoke) run_smoke ;;
    full) run_full ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 2 ;;
esac
