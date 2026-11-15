# Kaggle execution plan (single notebook)

`notebooks/run_phase4_kaggle.ipynb` is the only Kaggle notebook for Phase 4.
It calls the idempotent subcommands in `scripts/kaggle_full_run.py`; the
notebook does not reimplement pipeline logic.

## Prerequisites

- Kaggle GPU runtime: **GPU T4 x2** (only GPU 0 is used) or P100.
- **Internet ON**.
- Kaggle secret `HF_TOKEN`, attached to the notebook. The token must be able to:
  - read the gated `CohereLabs/tiny-aya-global` and `CohereLabs/tiny-aya-base`
    repos;
  - read the verified data snapshot
    `dubeydevansh/circuitgraft-phase4-data` (run
    `data-bc99df1d4f173b72`, `gate_g1.status == "passed"`);
  - create and write the private milestone dataset
    `dubeydevansh/circuitgraft-phase4-cloud`.
- Use **Save Version → Save & Run All**. Interactive sessions are not reliable
  for an 8–11 hour run.

The notebook never calls `cg data`, never edits human-review labels and never
back-fills metrics. A missing requirement stops the run with instructions.

## Storage and milestones

| Path | Purpose | Survives a dead session? |
|---|---|---|
| `/kaggle/working/circuit-graft` | code clone | no (re-cloned) |
| `/kaggle/tmp/cg/artifacts`, `/kaggle/tmp/cg/results` | checkpoints, generations, summaries | no |
| `/kaggle/working/cg_outputs` + `cg_phase4_outputs.zip` | small report bundle | yes (commit output) |
| `dubeydevansh/circuitgraft-phase4-cloud` | stage milestones | **yes** |

`setup` symlinks the repo's `artifacts/` and `results/` into the scratch tree
and imports the cloud milestones. Every completed stage uploads its own
milestone, so a rerun skips it.

Optimizer `trainer/` checkpoints are not uploaded. A session lost **during
drift** retrains drift from the last completed epoch only if that state is still
in scratch; after a new session, drift starts from scratch. Everything after
drift is resumable.

## Stage order

```text
setup -> data -> drift
      -> discover-safety -> discover-domain -> finalize-discover
      -> eval-refs
      -> for each candidate: eval-candidates --names <name>   # build, eval, archive
      -> merge-eval -> report -> [optional attacks -> report] -> publish -> export
```

`eval-candidates` evaluates `[aligned, drifted, candidate]` in one run so the
composites and Gate G-drift are present. The aligned/drifted generations are
reused from the cache. After evaluation, the candidate weights are deleted and
its manifest is marked `archived`; the metrics and lineage remain. `cg_project`
is retained for publication.

Building all six 3.5B candidates before evaluating them would not fit the
scratch quota, which is why the notebook interleaves build → evaluate → archive.

## Core matrix

Evaluated models: `aligned`, `drifted`, `cg_project`, `cg_uniform`,
`cg_random`, `resta`, `wiseft`, `ctheta`.

This covers the preregistered never-cut core:

- drift demonstration (`aligned` vs `drifted`, Gate G-drift);
- at least three baselines (`resta`, `wiseft`, `ctheta`);
- `cg_project` vs `cg_uniform` vs `cg_random`.

Deliberate exclusions, disclosed in the report:

- `cg_fisher`: no `artifacts/graft/fisher/domain_fisher.safetensors` (declared
  scope cut).
- `safelora`: state-dict memory footprint risks the 30 GB Kaggle RAM.
- `cg_exclude` / `cg_distributed`: secondary ablations, not in the core matrix.
- S8 attacks: opt-in only (`RUN_ATTACKS = True`); the report marks them absent
  when skipped.
- Selection-split sweep: skipped; each method is evaluated once at its
  preregistered default operating point. This is a fixed-hyperparameter
  comparison, not a selection-rule winner.

The judge is the frozen-config CPU fallback **string-match**. WildGuard is a 7B
fp16 judge and does not fit together with the 3.5B model on a 14.56 GiB T4;
WildGuard is recorded as `not_run`, never as a null result. The merged eval
summary and report are honestly `partial` while the human audit is pending.

## Resume commands

The notebook is safe to rerun from the top. For manual recovery in a live
session:

```bash
cd /kaggle/working/circuit-graft
python scripts/kaggle_full_run.py status
python scripts/kaggle_full_run.py data
python scripts/kaggle_full_run.py drift
python scripts/kaggle_full_run.py discover-safety
python scripts/kaggle_full_run.py discover-domain
python scripts/kaggle_full_run.py finalize-discover
python scripts/kaggle_full_run.py eval-refs
python scripts/kaggle_full_run.py eval-candidates --names cg_uniform
python scripts/kaggle_full_run.py merge-eval
python scripts/kaggle_full_run.py report
python scripts/kaggle_full_run.py publish
```

Useful environment overrides:

| Variable | Default | Meaning |
|---|---|---|
| `CG_BUDGET_HOURS` | `11` | stop starting new expensive work near the session limit |
| `CG_DRIFT_EPOCHS` | `2` | `1` is the speed fallback; log it if used |
| `CG_JUDGE` | `string_match` | `wildguard` is not viable on this T4 |
| `CG_SCRATCH` | `/kaggle/tmp/cg` | scratch root |
| `CG_CLOUD_REPO` | `dubeydevansh/circuitgraft-phase4-cloud` | milestone dataset |
| `CG_DATA_REPO` | `dubeydevansh/circuitgraft-phase4-data` | verified snapshot |
| `CG_DATA_RUN_ID` | `data-bc99df1d4f173b72` | required data run id |

## Failure diagnostics

- **`HF_TOKEN is not set` / missing secret:** add the Kaggle secret and attach
  it; `getpass` is intentionally not used in commit runs.
- **data snapshot rejected:** the `data` stage prints every failed G1 condition
  and refuses `cg data`. Check `CG_DATA_REPO` / `CG_DATA_RUN_ID`.
- **CUDA OOM during drift:** the config is micro-batch 1 × grad-accum 32. Do not
  raise the batch without a probe; `CG_DRIFT_EPOCHS=1` is the only speed knob.
- **OOM during discovery:** discovery is fp16, seq 96, batch 1, ig steps 2. Run
  safety and domain separately (the notebook already does).
- **`torchao` import conflict:** the install cell removes it; rerun the install
  cell if the environment was restarted.
- **cloud repo 404:** the first `setup` creates it; the auth cell verifies write
  access before any training starts.
- **session died:** rerun the notebook. `setup` imports completed milestones;
  `status` shows exactly which stages remain.
