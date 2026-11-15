"""Generate the single Kaggle runner notebook for Phase 4.

The notebook is intentionally thin: every stage is an idempotent subcommand of
``scripts/kaggle_full_run.py``. That keeps re-runs after a dead session cheap
and prevents the notebook from quietly reimplementing pipeline logic.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "notebooks" / "run_phase4_kaggle.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md(
        """
# CircuitGraft Phase 4 — single Kaggle run

This is the **only** Phase 4 Kaggle notebook. It runs the real experiment
matrix end-to-end on `CohereLabs/tiny-aya-global` and fails loudly instead of
inventing missing data.

## What it does

| Stage | Command | Output |
|---|---|---|
| Environment + clone + install | notebook cells | working tree |
| Import verified data | `setup`, `data` | `artifacts/data` (run `data-bc99df1d4f173b72`, G1 passed) |
| Drift SFT | `drift` | `artifacts/drift/scenarioA_seed0/merged` |
| Circuit discovery | `discover-safety`, `discover-domain`, `finalize-discover` | safety + domain circuits, overlap |
| Reference eval | `eval-refs` | aligned/drifted metrics + Gate G-drift |
| Core matrix | `eval-candidates` × 6 | 3 baselines + 3 CircuitGraft arms |
| Report | `report` | `results/s9/report-*/` |
| Publish | `publish` | HF models/datasets (drifted + cg_project only) |

Every completed milestone is uploaded to the private cloud dataset
`dubeydevansh/circuitgraft-phase4-cloud`. A new session imports it in `setup`
and skips completed stages.

## Kaggle settings

1. **Accelerator:** GPU T4 x2 (only GPU 0 is used) or P100.
2. **Internet:** ON.
3. **Secrets:** Add `HF_TOKEN` under **Add-ons → Secrets** and attach it to this
   notebook. The token must be able to read the gated Tiny Aya repos, read
   `dubeydevansh/circuitgraft-phase4-data`, and create/write the cloud dataset.
4. **Persistence:** Use **Save Version → Save & Run All**. Interactive sessions
   can die without warning.

## Honesty rules

- This notebook **never** runs `cg data`. It imports the verified snapshot or
  stops with an actionable error.
- It **never** edits human-review labels. `domain_benign`/`domain_harmful` are
  accepted only with their recorded human verification.
- The merged eval summary is `partial` while the human audit is pending, and
  the report says so. No metric is inferred or back-filled.
- Scope cuts are explicit: `cg_fisher` is absent (no Fisher artifact) and
  `safelora` is excluded (30 GB RAM risk). Attacks are opt-in at the end.

Expect roughly 8–11 hours of GPU wall-clock. The driver stops building or
evaluating new candidates when the `CG_BUDGET_HOURS` budget (default 11 h) is
nearly exhausted, then still writes an honest partial report.
"""
    ),
    md("## 1. Environment check"),
    code(
        """
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_SEVERITY", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("CG_BUDGET_HOURS", "11")
os.environ.setdefault("CG_CLOUD_REPO", "dubeydevansh/circuitgraft-phase4-cloud")
os.environ.setdefault("CG_DATA_REPO", "dubeydevansh/circuitgraft-phase4-data")
os.environ.setdefault("CG_DATA_RUN_ID", "data-bc99df1d4f173b72")
os.environ.setdefault("CG_SCRATCH", "/kaggle/tmp/cg")
os.environ.setdefault("HF_ORG", "dubeydevansh")

REPO_DIR = "/kaggle/working/circuit-graft"


def cg_run(*args: str) -> None:
    \"\"\"Run one orchestrator stage; raise on any non-zero exit.\"\"\"
    command = [sys.executable, "scripts/kaggle_full_run.py", *map(str, args)]
    print("$ " + " ".join(command), flush=True)
    started = time.time()
    subprocess.run(command, cwd=REPO_DIR, env=os.environ.copy(), check=True)
    print(f"  finished in {(time.time() - started) / 60:.1f} min", flush=True)


print("Python:", sys.version.split()[0])
print("Working directory:", os.getcwd())
print("Repo:", REPO_DIR)

import torch

print("PyTorch:", torch.__version__)
assert torch.cuda.is_available(), (
    "CUDA is not available. Set Settings → Accelerator → GPU T4 x2 and rerun."
)
properties = torch.cuda.get_device_properties(0)
print("GPU:", torch.cuda.get_device_name(0))
print(f"VRAM: {properties.total_memory / 2**30:.2f} GiB")
if properties.total_memory / 2**30 < 14:
    print("WARNING: less than 14 GiB VRAM; the drift/discovery configs target a T4.")

for path in ("/kaggle/working", "/kaggle/tmp"):
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    print(f"disk {path}: total={usage.total / 2**30:.1f} GiB free={usage.free / 2**30:.1f} GiB")

scratch = Path(os.environ["CG_SCRATCH"])
scratch.mkdir(parents=True, exist_ok=True)
probe = scratch / ".write_test"
probe.write_text("ok", encoding="utf-8")
probe.unlink()
print("scratch writable:", scratch)
print(f"scratch free: {shutil.disk_usage(scratch).free / 2**30:.1f} GiB")

print("Budget:", os.environ["CG_BUDGET_HOURS"], "h")
print("Cloud milestone repo:", os.environ["CG_CLOUD_REPO"])
print("Verified data repo:", os.environ["CG_DATA_REPO"], os.environ["CG_DATA_RUN_ID"])
"""
    ),
    md(
        """
## 2. Clone the repository

The clone is reset to `origin/main` on every run, so a stale working tree can
never silently run old code. Scratch artifacts live outside the repo and are
not touched by `git reset`.
"""
    ),
    code(
        """
%%bash
set -euo pipefail
cd /kaggle/working
if [ ! -d circuit-graft/.git ]; then
  git clone --depth 1 https://github.com/0dvsh0/circuit-graft.git circuit-graft
fi
cd circuit-graft
git fetch --depth 1 origin main
git reset --hard origin/main
git log -1 --oneline
"""
    ),
    md(
        """
## 3. Install the pinned stack

The versions match `requirements-main.lock`. `torchao` is removed because it
conflicts with the pinned PEFT, and the unusable Kaggle base packages
`cudf-cu13`/`bigframes` are removed to keep the resolver quiet. The editable
install is for metadata/entry points; the orchestrator also exports
`PYTHONPATH=src` so `python -m circuitgraft.cli` works even when a commit run
does not process the PEP 660 `.pth`.
"""
    ),
    code(
        """
%%bash
set -euo pipefail
export TF_CPP_MIN_LOG_LEVEL=3
export ABSL_LOGGING_MIN_SEVERITY=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="/kaggle/working/circuit-graft/src${PYTHONPATH:+:$PYTHONPATH}"
cd /kaggle/working/circuit-graft

pip uninstall -y torchao cudf-cu13 bigframes
python -m pip install --upgrade pip
python -m pip install --no-warn-conflicts \\
  safetune==0.1.12 circuitkit==0.2.0 auditkit==1.1.3 \\
  "curatorkit[generation,hf]==0.1.22" bitsandbytes==0.50.2
python -m pip install --no-warn-conflicts \\
  "transformers==5.22.1" "peft==0.21.2" "trl==1.14.2" \\
  "transformer-lens==3.8.0" "datasets==5.1.0" "accelerate==1.15.0" \\
  "huggingface-hub==1.33.0" "sentence-transformers==6.2.0" \\
  "matplotlib==3.11.2" "seaborn==0.13.2" \\
  python-dotenv typer rich litellm
pip uninstall -y torchao
python -m pip install -e . --no-deps

python - <<'PY'
import importlib.util
import sys

import numpy
import peft
import torch
import transformers

for name in ("safetune", "circuitkit", "auditkit", "curatorkit"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing required package: {name}")
if importlib.util.find_spec("torchao") is not None:
    raise SystemExit("torchao is still installed; PEFT will reject it")

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("numpy", numpy.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

python -m circuitgraft.cli doctor
"""
    ),
    md(
        """
## 4. Authenticate with Kaggle secrets

No `getpass`: commit runs are unattended, so a missing secret must stop the
notebook with instructions. This cell also proves the token can read the two
gated Tiny Aya repos and the verified data snapshot, and can create the private
milestone repo.
"""
    ),
    code(
        """
import os

from huggingface_hub import HfApi

token = (os.environ.get("HF_TOKEN") or "").strip()
if not token:
    try:
        from kaggle_secrets import UserSecretsClient

        token = (UserSecretsClient().get_secret("HF_TOKEN") or "").strip()
    except Exception as exc:
        raise RuntimeError(
            "HF_TOKEN is unavailable. In Kaggle: Add-ons → Secrets → add HF_TOKEN, "
            "attach it to this notebook, then Save & Run All again. "
            f"({type(exc).__name__}: {exc})"
        ) from exc

if not token:
    raise RuntimeError(
        "HF_TOKEN is empty. Add it under Add-ons → Secrets and attach it to this notebook."
    )

os.environ["HF_TOKEN"] = token
os.environ["HF_ORG"] = "dubeydevansh"

api = HfApi(token=token)
try:
    identity = api.whoami()
except Exception as exc:
    raise RuntimeError(
        "HF_TOKEN was rejected by the Hugging Face API. Replace the Kaggle secret "
        f"with a valid token. ({type(exc).__name__}: {exc})"
    ) from exc
print("HF user:", identity.get("name"))

blocked = []
for repo_id in ("CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-base"):
    try:
        api.model_info(repo_id)
    except Exception as exc:
        blocked.append(f"model {repo_id} ({type(exc).__name__}: {exc})")
    else:
        print("OK model:", repo_id)

try:
    api.dataset_info(os.environ["CG_DATA_REPO"])
except Exception as exc:
    blocked.append(
        f"dataset {os.environ['CG_DATA_REPO']} ({type(exc).__name__}: {exc})"
    )
else:
    print("OK verified data snapshot:", os.environ["CG_DATA_REPO"])

try:
    api.create_repo(
        os.environ["CG_CLOUD_REPO"], repo_type="dataset", private=True, exist_ok=True
    )
except Exception as exc:
    blocked.append(
        f"cannot write {os.environ['CG_CLOUD_REPO']} ({type(exc).__name__}: {exc})"
    )
else:
    print("OK cloud milestone repo:", os.environ["CG_CLOUD_REPO"])

if blocked:
    raise RuntimeError(
        "The HF token cannot access the required repositories. Accept the Tiny Aya "
        "licences with the SAME account that owns the token, and make sure the token "
        "can read the verified data repo and write the milestone repo:\\n  - "
        + "\\n  - ".join(blocked)
    )
"""
    ),
    md(
        """
## 5. Setup scratch + import the verified data snapshot

`setup` symlinks `artifacts/` and `results/` into `/kaggle/tmp/cg`, points the
HF caches there, and imports any completed cloud milestones. `data` then
validates the exact run id and Gate G1; it never falls back to `cg data`.
"""
    ),
    code(
        """
cg_run("setup")
cg_run("data")
cg_run("status")
"""
    ),
    md(
        """
## 6. Drift SFT (θ_A → θ_D)

Default: 2 epochs, micro-batch 1 × grad-accum 32 (314 optimizer steps), the
T4-safe shape already validated against OOMs. If the session starts with very
little wall-clock left, set `CG_DRIFT_EPOCHS=1` before this cell; that is a
logged speed fallback, not a silent change.

The merged checkpoint and adapter are uploaded as soon as training completes.
Optimizer `trainer/` checkpoints are deliberately **not** uploaded: a session
lost mid-drift retrains drift, while every completed later stage is resumable.
"""
    ),
    code(
        """
epochs = os.environ.get("CG_DRIFT_EPOCHS", "2")
print("Drift epochs:", epochs)
cg_run("drift", "--epochs", epochs)
cg_run("status")
"""
    ),
    md(
        """
## 7. Safety circuit discovery on θ_A

Safety runs first and uploads on its own. If domain discovery crashes, the
safety circuit is already safe in the cloud repo and will not be redone.
"""
    ),
    code(
        """
cg_run("discover-safety")
cg_run("status")
"""
    ),
    md(
        """
## 8. Domain circuit discovery on θ_D

This stage refuses to run unless the drifted merged checkpoint exists.
"""
    ),
    code(
        """
cg_run("discover-domain")
cg_run("status")
"""
    ),
    md(
        """
## 9. Finalize discovery

Writes the overlap reports, discovery `summary.json` and a manifest whose notes
disclose the split safety/domain execution.
"""
    ),
    code(
        """
cg_run("finalize-discover")
cg_run("status")
"""
    ),
    md(
        """
## 10. Evaluate the references and Gate G-drift

The judge is the preregistered CPU fallback **string-match**, not WildGuard.
A 7B fp16 WildGuard judge does not fit alongside the 3.5B model on a 14.56 GiB
T4, and the frozen eval config already declares string-match as the fallback.
WildGuard is recorded as `not_run`, never as a null result.

Gate G-drift needs the aligned/drifted MedMCQA gain and HarmBench refusal drop;
the merged report carries its status.
"""
    ),
    code(
        """
cg_run("eval-refs")
cg_run("status")
"""
    ),
    md(
        """
## 11. Core matrix: build → evaluate → archive, one candidate at a time

Six candidates satisfy the preregistered core and all three comparison axes:

- `cg_project` vs `cg_uniform` vs `cg_random`
- three baselines: `resta`, `wiseft`, `ctheta`

Each `eval-candidates` call builds only the missing candidate, evaluates
`[aligned, drifted, candidate]` so composites and Gate G-drift are present in
that run, reuses the generation cache for the references, and then deletes the
candidate's weights while keeping its stats/manifest. `cg_project` is retained
for publication. This staged order is required: six 3.5B checkpoints plus the
drifted model would not fit the Kaggle scratch quota if they were all built
first.

If the time budget runs low, the driver stops starting new candidates and the
final summary is honestly partial.
"""
    ),
    code(
        """
CORE = ["cg_project", "cg_uniform", "cg_random", "resta", "wiseft", "ctheta"]
for name in CORE:
    print(f"\\n=== {name} ===", flush=True)
    cg_run("eval-candidates", "--names", name)
cg_run("merge-eval")
cg_run("status")
"""
    ),
    md(
        """
## 12. Generate the report

S9 reads the merged eval summary, discovery overlap and any attack stats, then
injects tables/plots into `README.md` and writes a tracked bundle under
`results/s9/report-*/`. The status stays `partial` while the human audit is
pending.
"""
    ),
    code(
        """
cg_run("report")
"""
    ),
    md(
        """
## 13. Optional abliteration attacks

Set `RUN_ATTACKS = True` only when more than ~1.5 h remain after the core
matrix. Attacks are research-only and are **not** part of the core claim; the
S8 prerequisite is the retained `cg_project` checkpoint. Running them reruns
the report so the attack table is populated from measured attack stats.
"""
    ),
    code(
        """
RUN_ATTACKS = False
if RUN_ATTACKS:
    cg_run("attacks", "--force")
    cg_run("report")
else:
    print("S8 attacks skipped by configuration; the report states they are absent.")
"""
    ),
    md(
        """
## 14. Publish the public artifacts

`cg publish` publishes the drifted checkpoint, its LoRA adapter, the retained
`cg_project` checkpoint, the curated SFT/pairs/eval datasets, discovery outputs
and eval logs. Candidates whose manifests are `archived` are skipped, not
published from missing weights. If `cg_project` was lost across a session, the
driver deterministically rebuilds it first.
"""
    ),
    code(
        """
cg_run("publish")
"""
    ),
    md(
        """
## 15. Export small outputs + final status

`/kaggle/working/cg_outputs/` and its zip are the only outputs that survive a
commit run; the large checkpoints stay in the cloud repo/scratch. This cell
prints the final table and gate status so the notebook ends with the actual
numbers.
"""
    ),
    code(
        """
import json
from pathlib import Path

cg_run("export")
cg_run("status")

output_root = Path("/kaggle/working/cg_outputs")
tables = sorted(output_root.rglob("table_main.md"), key=lambda p: p.stat().st_mtime)
if tables:
    print("\\n=== Main results table ===")
    print(tables[-1].read_text(encoding="utf-8"))

summaries = sorted(output_root.rglob("summary.json"), key=lambda p: p.stat().st_mtime)
if summaries:
    merged = next((p for p in reversed(summaries) if "eval-kaggle-merged" in str(p)), None)
    if merged is not None:
        doc = json.loads(merged.read_text(encoding="utf-8"))
        print("\\n=== Merged eval status ===")
        print("status:", doc.get("status"))
        print("gate G-drift:", doc.get("gate_g_drift", {}).get("status"))
        for name, status in doc.get("model_status", {}).items():
            print(f"  {name:16s} {status}")
        print("partial reasons:")
        for reason in doc.get("partial_reasons", []):
            print("  -", reason)

archive = Path("/kaggle/working/cg_phase4_outputs.zip")
if archive.exists():
    print(f"\\nSubmission zip: {archive} ({archive.stat().st_size / 2**20:.1f} MiB)")
print("\\nCloud milestones: https://huggingface.co/datasets/" + os.environ["CG_CLOUD_REPO"])
"""
    ),
    md(
        """
## 16. Recovery playbook

The notebook is idempotent: rerun **Save & Run All** after any failure.

| Failure point | What survives | What reruns |
|---|---|---|
| setup/data | nothing yet | data download |
| drift | data + discovery imports | **drift from scratch** (optimizer checkpoints are not uploaded) |
| safety discovery | data, drift | only an incomplete safety circuit |
| domain discovery | data, drift, safety circuit | only an incomplete domain circuit |
| finalize | both circuits | overlap assembly |
| reference eval | all above + eval summaries/generations | only missing models |
| a candidate | all above + previous summaries | that candidate is rebuilt, evaluated, archived |
| report/publish | everything above | report/publish are rerun |

Manual resume from a shell in this session:

```bash
cd /kaggle/working/circuit-graft
python scripts/kaggle_full_run.py status
python scripts/kaggle_full_run.py eval-candidates --names cg_uniform
python scripts/kaggle_full_run.py merge-eval
python scripts/kaggle_full_run.py report
python scripts/kaggle_full_run.py publish
```

### Explicit non-goals

- No `cg data` fallback and no re-labelling of human review.
- No WildGuard run on this T4 (string-match fallback, disclosed).
- `cg_fisher` is a declared scope cut; `safelora` is excluded for RAM.
- Attacks are opt-in; if skipped, the report marks them absent.

The authoritative run state is the cloud dataset, not the notebook output
version. If this session dies, open the notebook and Save & Run All again; the
already-uploaded milestones are imported in `setup`.
"""
    ),
]

notebook = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "isGpuEnabled": True,
            "isInternetEnabled": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 2,
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Generated {OUT_PATH} ({OUT_PATH.stat().st_size} bytes, {len(CELLS)} cells)")
