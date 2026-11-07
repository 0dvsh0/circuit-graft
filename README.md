# CircuitGraft

Circuit-aware safety recovery for domain-fine-tuned **Cohere Tiny Aya** models.

CircuitGraft discovers a safety circuit and a domain circuit in the same head/MLP
component space, maps their overlap, and grafts the safety delta back into a
drifted medical specialist: at full strength on safety-only components, and
through a domain-protecting projection or Fisher gate on shared components.
The goal is to recover safety without throwing away domain competence.

> **Research status.** Gate **G0 is PASSED**: all four toolkits (SafeTune,
> CircuitKIT, AuditKIT, CuratorKIT) run end-to-end on real **Cohere Tiny Aya**
> (`CohereLabs/tiny-aya-global` & `CohereLabs/tiny-aya-base`) on Tesla T4 CUDA GPU.
> Evidence is documented in [`results/d1/smoke/summary.md`](results/d1/smoke/summary.md).
> Stages S1 through S10 are implemented and unit-tested. Phase 4 experiments are underway.

## What is here

| Area | Status |
|---|---|
| Python 3.12 environment with SafeTune / CircuitKIT / AuditKIT / CuratorKIT | done |
| Component vocabulary and GQA-aware weight slicing | implemented, unit-tested |
| CircuitGraft core (`topk`, `distributed`, `random_matched`, four overlap strategies) | implemented, unit-tested |
| Config schemas and run manifests | implemented |
| D1 smoke runner for all four toolkits | **PASSED (G0 gate cleared on Tiny Aya Tesla T4)** |
| S6 checkpoint pipeline and circuit namespace adapters | implemented, unit-tested |
| S4 discovery runner (CircuitKIT primary + SafeTune cross-check) | implemented, unit-tested |
| S1 data curation (CuratorKIT SFT, eval sets, pairs, decontamination) | implemented, unit-tested |
| S2 Drift, S3 Audit, S5 Baselines, S7 Eval, S8 Attacks | implemented, unit-tested |
| S9 Report generation & S10 HuggingFace publishing | implemented, unit-tested |

## Setup

The project targets Python **3.12** and one virtual environment that resolves all
four toolkits. `requirements-main.lock` records the exact D1 resolution.

```powershell
# from the repository root
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv\Scripts\python.exe -r requirements-main.lock
python -m uv pip install --python .venv\Scripts\python.exe -e . --no-deps

Copy-Item .env.example .env
# edit .env: HF_TOKEN, HF_ORG
.\.venv\Scripts\cg.exe doctor --check-models
```

On Linux/macOS use `.venv/bin/python`, `.venv/bin/pip` and `.venv/bin/cg`
instead of the Windows paths.

### Gated model access

Both `CohereLabs/tiny-aya-global` and `CohereLabs/tiny-aya-base` are gated. Log
in to Hugging Face as the account that owns `HF_TOKEN`, open each model page,
accept the licence, and wait for the grant to propagate. On 2026-11-07 the base
checkpoint was accessible but the global checkpoint still returned 403, so both
must be checked separately. Then:

```bash
cg doctor --check-models --json-out results/d1/environment.json
```

The command exits non-zero until both repositories are accessible.

## Day 1 smoke test

`scripts/d1_smoke.py` exercises all four toolkits in separate subprocesses so a
crash or leaked GPU memory in one cannot take down the others:

```bash
# G0 evidence: GPU machine, both Tiny Aya licences accepted
python scripts/d1_smoke.py --device cuda --output-dir results/d1/smoke

# CPU-only mechanics check against a public proxy model (never G0)
python scripts/d1_smoke.py --proxy --device cpu --max-examples 4 \
  --output-dir results/d1/smoke_proxy
```

It runs:

1. **SafeTune** EAP-IG safety-circuit discovery on a handful of contrast pairs
   and a `StringMatchJudge` refusal check.
2. **CircuitKIT** `greater_than` discovery on the aligned Tiny Aya model.
3. **AuditKIT** `hf:` evaluation and a fingerprinted `RunResult`.
4. **CuratorKIT** curation (dedup, cleaning, split/export) of a tiny JSONL.

Each check writes `<toolkit>.json`; the orchestrator writes `summary.json` and
`summary.md` with per-check status, library versions, device, timing and
`gate_g0`. A proxy run always reports `gate_g0 = not_evaluated`, and missing
model access is reported as a blocked check, never as success. The 2026-11-05
proxy run passed all four mechanics checks:
[`results/d1/smoke_proxy/summary.md`](results/d1/smoke_proxy/summary.md). Those
numbers are proxy-model mechanics only, not Tiny Aya evidence.

## Core CLI

```bash
cg doctor                    # packages, device, token
cg config-check configs/base.yaml
cg config-check configs/graft/cg_project.yaml

# S1 mechanics smoke: small samples, no LLM key, no embedding filter
cg data --config configs/smoke/data.yaml
```

The full pipeline commands planned by the specification are `cg data`, `drift`,
`audit`, `discover`, `baselines`, `graft`, `eval`, `attack`, `report` and
`publish`. They are wired into the CLI as their stages land; running a stage
before its inputs exist fails loudly rather than generating placeholder metrics.

## Documentation

- [`docs/API_NOTES.md`](docs/API_NOTES.md) - verified signatures of the
  installed SafeTune, CircuitKIT, AuditKIT and CuratorKIT versions, plus the
  API discrepancies that affect later stages.
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) - dated commands, outcomes,
  failures and fixes. The only place results are recorded.
- [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) - hypotheses, metrics,
  sweep grid and decision rules. **Draft**: it must be frozen and committed
  before any Phase 4 experiment.

## Repository layout

```text
configs/                 base + stage + graft/baseline configurations
src/circuitgraft/
  components.py          Cohere2 component <-> weight-slice mapping (GQA aware)
  config.py              pydantic schemas, YAML merging, config hashing
  registry.py            manifests, hashing, idempotency, HF upload helpers
  discover/              overlap.py, safetune_eap.py, circuitkit_runner.py
  graft/                 select.py, protect.py, fisher.py, graft.py, pipeline.py
  data/                  pair validation and decontamination
  eval/stats.py          bootstrap CIs, McNemar, Holm, kappa
  ...
scripts/d1_smoke.py      D1 toolkit smoke test
tests/                   CPU unit and integration tests
docs/                    API notes, preregistration, experiment log, article
results/                 small JSON results (generations are git-ignored)
```

## Method summary

Let `θ_B`, `θ_A`, `θ_D` be the base, aligned and drifted weights. The safety
delta is `Δ_S = θ_A − θ_B`; the domain delta is `Δ_D = θ_D − θ_A`. Components are
attention heads `L{l}.H{h}` and MLPs `L{l}.MLP`. With selected sets `S` and `D`:

- `c ∈ S \ D`: `W_D[c] += α · Δ_S[c]`
- `c ∈ S ∩ D`: `W_D[c] += α_O · protect(Δ_S[c], Δ_D[c], F_D[c])`
- otherwise: unchanged.

The overlap strategies are `exclude`, `svd_project` (project the safety edit away
from the top right-singular directions of the domain delta), `fisher_gate`
(per-weight domain-Fisher down-weighting), and `uniform` (the C-ΔΘ-like
control). Layer norms, embeddings and `lm_head` are never edited.

## Evaluation

The primary comparison is the Pareto position of safety recovery
(`SRR = (HRR_repaired − HRR_D) / (HRR_A − HRR_D)`) against domain gain retention
(`DGR = (Acc_repaired − Acc_A) / (Acc_D − Acc_A)`), with false-refusal and
edit-footprint secondary axes. Comparisons use paired bootstrap CIs and McNemar
tests with Holm correction. The draft
[`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) will be frozen and
committed before any Phase 4 experiment.

## Licences

- **Code:** Apache-2.0 (see `LICENSE`).
- **Tiny Aya weights:** CC-BY-NC and Cohere's Acceptable Use Policy. This
  repository does not redistribute the weights.
- **Drifted checkpoints:** research-only, not for deployment, per SafeTune's
  licence terms.

## Citation

```bibtex
@misc{circuitgraft2026,
  title  = {CircuitGraft: Circuit-aware safety recovery for domain-fine-tuned Tiny Aya},
  author = {CircuitGraft team},
  year   = {2026},
  note   = {Lexsi Labs hackathon, Track 2b / Problem 6}
}
```
