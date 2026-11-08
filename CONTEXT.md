# CircuitGraft — project context and handoff

**Read this first.** This file is the single-page state of the project for a new
human or coding agent. Detailed evidence lives in
[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md); installed-library API
quirks live in [`docs/API_NOTES.md`](docs/API_NOTES.md); the research plan is
[`SPEC.md`](SPEC.md).

Last updated: 2026-11-08.

---

## 1. What the project is

CircuitGraft is a submission for the Lexsi Labs hackathon (Track 2b / Problem 6):
repair safety drift in a domain-fine-tuned **Cohere Tiny Aya** model without
retraining, while keeping the medical-domain gain.

The method discovers a safety circuit `S` on the aligned model and a domain
circuit `D` on the drifted specialist in the same component space
(`L{l}.H{h}`, `L{l}.MLP`), then grafts the safety delta back:

- safety-only components: full-strength delta;
- shared components: a domain-protected edit (`svd_project`, `fisher_gate`,
  `exclude`, or the `uniform` control);
- everything else: untouched.

Primary comparison: safety recovery (`SRR`) vs domain gain retention (`DGR`)
against RESTA, WiSE-FT, C-ΔΘ and Safe LoRA.

## 2. Current status (one screen)

| Area | State |
|---|---|
| Environment | Python 3.12, one `.venv`, all four toolkits import |
| Gate G0 (four toolkits on real Tiny Aya / T4) | **PASSED** — `results/d1/smoke/` |
| S1 data | implemented; only partial CPU smoke run (`artifacts/data_smoke/`) |
| S2 drift | implemented; **not run on GPU** |
| S3 audit | implemented; **not run on Tiny Aya** |
| S4 discovery | implemented; **not run on Tiny Aya** |
| S5 baselines | implemented; **not run on Tiny Aya** |
| S6 graft | implemented; **not run on Tiny Aya** |
| S7 eval | implemented; proxy smoke only (`results/s7/smoke_proxy/`) |
| S8 attacks | implemented; **not run on Tiny Aya**; `cg eval-attacks` evaluates saved checkpoints through S7 (2026-11-08) |
| S9 report | implemented, integrated with real stage outputs, tested end-to-end (2026-11-08) |
| S10 publish | implemented, resolves real checkpoints/data and filters shared dataset folders (2026-11-08) |
| `scripts/run_all.sh` | written: `--check`, `--smoke`, `--full` (2026-11-08) |
| HF Space demo | implemented, data-driven and unit-tested; not deployed (2026-11-08) |
| `docs/ARTICLE.md` | scaffolded with the required outline and `ARTICLE:` placeholders (2026-11-08) |
| Preregistration | **DRAFT** — must be frozen before Phase 4 |
| Unit tests | `187 passed` at the last full run (2026-11-08) |
| GPU work | blocked locally: this machine is CPU-only (torch 2.15.0+cpu) |

**Bottom line:** the software scaffold is complete through S10, the Space reads
only tracked artifacts, and the article skeleton exists. The project is still
not finished: the missing deliverable is the real Phase 4 experiment matrix on
Tiny Aya plus a public HF bundle. Do not describe the proxy/smoke numbers as
Tiny Aya evidence.

## 3. Hard rules

1. **Never fabricate results.** Numbers may only come from code in this repo and
   be stored under `artifacts/` or `results/`. Use `—` / `not_evaluated` when a
   run is missing.
2. **Every stage writes a manifest** (`artifacts/<stage>/<run_id>/manifest.json`)
   with config hash, git SHA, input hashes, versions, device and timing.
3. **Freeze `docs/PREREGISTRATION.md` before the first Phase 4 run.** The commit
   timestamp is the evidence. A metric/scenario change after seeing results is a
   deviation, not a preregistered test.
4. **Same circuit for C-ΔΘ and CircuitGraft.** C-ΔΘ consumes the CircuitKIT
   `circuit_info.json` produced by S4.
5. **Name C-ΔΘ honestly:** “C-ΔΘ (SafeTune weight-arithmetic implementation)”.
   SafeTune's `apply_ctheta` is layer/module-granular, not head-identical.
6. **No harmful completions published.** Drifted checkpoints get the
   research-only warning card.
7. Log commands, failures and fixes in `docs/EXPERIMENT_LOG.md`.

## 4. Environment and access

```powershell
# from the repo root
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv\Scripts\python.exe -r requirements-main.lock
python -m uv pip install --python .venv\Scripts\python.exe -e . --no-deps

.\.venv\Scripts\cg.exe doctor
.\.venv\Scripts\cg.exe doctor --check-models
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests scripts space
```

- `.env` is git-ignored and currently has `HF_TOKEN`, `HF_ORG=dubeydevansh`
  and `COHERE_API_KEY` populated on the author’s machine.
- Both Tiny Aya repos are accessible; the five gated eval datasets were verified
  on 2026-11-07.
- Local machine: Windows, CPU only. GPU work runs on Colab/Kaggle or another
  CUDA box.
- Colab gotchas are documented in `docs/API_NOTES.md` and the G0 log entry;
  notably `import circuitkit` **before** patching TransformerLens, delete
  `torchao`, and load bf16 explicitly.

## 5. Repository map

```text
SPEC.md                     full research + engineering specification
README.md                   public overview (results markers at the bottom)
CONTEXT.md                  this handoff file
configs/                    base + per-stage YAML (data, drift, discover, eval,
                            attacks, 6 graft variants, 6 baselines, smoke configs)
src/circuitgraft/
  cli.py                    `cg` entry point for every stage
  config.py                 pydantic schemas + config hashing
  registry.py               manifests, hashing, latest-run resolution, HF upload
  components.py             Cohere2 component ↔ weight-slice mapping (GQA-aware)
  data/                     S1 curation, pairs, decontamination
  drift/finetune.py         S2 LoRA SFT → merged drifted checkpoint
  discover/                 S4 CircuitKIT, SafeTune EAP cross-check, overlap
  baselines/safetune_recover.py  S5 SafeTune Faithful baselines
  graft/                    S6 selection, protection, Fisher, pipeline
  eval/                     S3 audit + S7 generation/judging/metrics/AuditKIT
  attacks/                  S8 abliteration, harmful-FT, quantization +
                            `evaluate.py` (S8 checkpoint → S7 post-eval bridge)
  report/                   S9 tables, plots, README injection
  publish.py                S10 HF model/dataset cards + collections
space/                      read-only Gradio demo (S9 bundle + cached responses)
scripts/                    d1_smoke.py, build_notebook.py, run_all.sh
notebooks/                  run_d1_g0_smoke.ipynb
tests/                      CPU unit/integration suite (187 tests)
docs/                       API notes, experiment log, preregistration, article
artifacts/                  heavy stage outputs (git-ignored)
results/                    small, intentional result JSON/Markdown (tracked)
```

## 6. Stage contract and real output paths

Each stage is idempotent on `(config_hash, input_hashes)`.

| Stage | Command | Real output |
|---|---|---|
| S1 | `cg data --config configs/data.yaml` | `artifacts/data/{medmcqa_sft,pairs,eval,selection,decontam,dataset_cards}` |
| S2 | `cg drift --config configs/drift.yaml` | `artifacts/drift/<scenario>_seed<k>/{adapter,merged,drift_summary.json}` |
| S3 | `cg audit --models aligned,drifted` | `artifacts/audit/audit-<hash>/summary.json` |
| S4 | `cg discover` | `artifacts/discover/discover-<hash>/{safety,domain,overlap.json,summary.json}` |
| S5 | `cg baselines --config configs/baselines/resta.yaml [--sweep]` | `artifacts/baselines/<method>-<hash>/merged` |
| S6 | `cg graft --config configs/graft/cg_project.yaml` | `artifacts/graft/<name>-<hash>/{model.safetensors,graft_stats.json}` |
| S7 | `cg eval --all` | `artifacts/eval/eval-<hash>/summary.json` |
| S8 | `cg attack --type abliterate` | `artifacts/attacks/<type>/<run>/attack_stats.json` (+ `merged/` for abliteration) |
| S8→S7 | `cg eval-attacks` | `artifacts/attacks/<type>/<run>/post_eval.json`, `artifacts/attacks/eval/summary.json` |
| S9 | `cg report` | `artifacts/report/`, copies under `results/s9/<run>/`, README injection |
| S10 | `cg publish` | HF repos + two collections under `HF_ORG` |

S9 discovers the newest complete/partial runs under the real artifact tree
(`artifacts/eval/eval-*`, `artifacts/discover/discover-*`,
`artifacts/attacks/*/*/attack_stats.json`) and copies tables plus plots into
`results/s9/<run>/`. README links point at those tracked copies because
`artifacts/` is git-ignored. S10 resolves the actual checkpoint directories
(`drift/*/merged`, `graft/*`, `baselines/*/merged`, abliteration `merged/`) and
uses allow-lists when one data folder feeds two dataset repos.

`cg eval-attacks` is the S8→S7 bridge: it discovers every saved attacked
checkpoint (`abliteration merged/`, harmful-FT step checkpoints), evaluates them
through the normal S7 pipeline with isolated output/generation directories, and
writes `post_eval.json` beside each `attack_stats.json`. S9 merges those measured
values into `abliteration_hrr`, `harmful_ft` step rows and the quantization
columns, and records the source file under `attack_post_eval`. Runtime
abliteration and quantization plans still produce no checkpoint, so they remain
`—` / `not_evaluated`.

Configs use stable paths (`artifacts/discover/safety/scores.json`); the registry
resolves them to the newest completed stage run.

## 7. Current blockers / open items

1. **Preregistration review and freeze.** It is still marked DRAFT. A human
   must review §3, §4, §7 and then commit it before Phase 4.
2. **GPU run of Phase 4.** Full sequence:
   `data → drift → audit → discover → baselines → graft (6 configs) → eval --all
   → attacks → eval-attacks → report → publish`.
   Free Colab T4 is enough for smoke but not necessarily for the full matrix;
   plan persistence across sessions (Drive) and use Kaggle/HF credits if needed.
3. **Full S1 data run.** Requires the Cohere API for domain-sensitive
   generation and human review of the generated rows. Only a 200-row partial
   smoke exists (`artifacts/data_smoke/`).
4. **Post-attack HRR comes from a separate S7 pass.** S8 saves attacked
   checkpoints/plans; `cg eval-attacks` evaluates every saved checkpoint through
   S7 and writes `post_eval.json`, which S9 consumes. Quantization attacks still
   save only a plan, so quantization HRR remains unavailable until a real
   quantized checkpoint is written and evaluated. `run_all.sh --full` runs the
   bridge automatically unless `--no-attack-eval` is passed.
5. **Space deployment needs a real bundle.** `space/app.py` shows a setup state
   when `results/s9/report-*/` is absent, so deploy it after the first complete
   Phase 4 report and copy that bundle into the Space repository. Cached Compare
   responses also require published checkpoints.
6. **Article placeholders need S9 numbers.** `docs/ARTICLE.md` deliberately
   contains no metrics; fill it from the tracked `results/s9/<run_id>/` bundle
   and replace the artifact links after S10.

## 8. Completed this pass (2026-11-08)

The end of the pipeline now consumes the real stage outputs, and the last two
deliverables have scaffolds:

- S9 resolves the latest `artifacts/eval/eval-*`, `artifacts/discover/discover-*`
  and `artifacts/attacks/*/*/attack_stats.json`; it ignores failed manifests,
  normalises the nested S7 metric shape, attaches clean HRR to attack rows, and
  copies tables plus plots to `results/s9/<run>/`. README links target those
  tracked copies, not git-ignored `artifacts/` paths.
- `cg eval-attacks` (new in `attacks/evaluate.py`) bridges S8 to S7: it resolves
  saved abliteration and harmful-FT checkpoints, runs one isolated S7 evaluation
  over them, writes per-run `post_eval.json` and an aggregate
  `artifacts/attacks/eval/summary.json` manifest, and marks unsupported attacks
  as `not_evaluated` instead of guessing. S9 merges those values into the attack
  table and plot.
- S10 resolves the actual checkpoint directories written by S2/S5/S6/S8 and the
  S1 data tree; dataset staging uses allow-lists so safety and domain pairs do
  not leak into each other's repos. `cg publish --dry-run` now reports exactly
  what would be uploaded and what is missing.
- `scripts/run_all.sh` supports `--check`, `--smoke` and `--full`; `--check`
  passed locally. `--smoke` uses a scratch README so it cannot overwrite the
  public result markers.
- `space/app.py` is a read-only Gradio demo over `results/s9/report-*` with
  optional cached Compare responses. It imports Gradio lazily, never loads a
  model, shows a setup state when no bundle exists, and is covered by
  `tests/test_space_app.py`. Installation is the `space` extra
  (`pip install -e ".[space]"`).
- `docs/ARTICLE.md` follows the 11-section specification outline with explicit
  `ARTICLE:` placeholders. It contains no metrics until a tracked S9 bundle
  exists.
- `docs/EXPERIMENT_LOG.md` records the integration and verification runs.

## 9. Next step

1. On a CUDA machine, freeze `docs/PREREGISTRATION.md`, then run
   `bash scripts/run_all.sh --full`. Inspect `artifacts/report/summary.md` and
   `results/s9/<run>/` before publishing.
2. After a complete S9 bundle exists, copy it into the Space repository, add
   curated `space/cached_responses.json` once checkpoints are public, and deploy.
3. Fill `docs/ARTICLE.md` only from `results/s9/<run_id>/` and replace every
   artifact link after S10.

## 10. Quick verification commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests scripts space space
.\.venv\Scripts\python.exe -m pytest tests\test_space_app.py -q
Get-ChildItem configs -Recurse -Filter *.yaml | ForEach-Object {
  .\.venv\Scripts\cg.exe config-check $_.FullName
}
.\.venv\Scripts\cg.exe report --dry-run
.\.venv\Scripts\cg.exe eval-attacks --dry-run
.\.venv\Scripts\cg.exe publish --dry-run
```

```bash
# Git Bash / Linux / macOS
bash scripts/run_all.sh --check
bash scripts/run_all.sh --smoke   # CPU mechanics; writes only under artifacts/run_all_smoke
```

A safe offline S9 check (no GPU) uses the preserved S7/smoke summaries:
it will produce an explicitly partial report with `—` values, which is correct.
The smoke runner copies `README.md` to a scratch directory before injection, so
it cannot overwrite the public result markers.

## 11. Key decisions already made

- Domain: **medicine** (MedMCQA SFT, MedQA/PubMedQA transfer, MMLU control).
- Drift scenario: **A, benign SFT**; Scenario B (3% harmful rows) is a disclosed
  fallback only if H0 fails.
- Delta source: `alignment` (`θ_A − θ_B`) for the primary method.
- Components: attention heads and MLPs only; never layer norms, embeddings or
  `lm_head`.
- Composite metrics: SRR, DGR, ORP exactly as defined in
  `docs/PREREGISTRATION.md`.
- Selection rule for every method: maximize SRR subject to DGR ≥ 0.7, else
  maximize SRR + DGR; ties by smaller footprint then lower ORP.
