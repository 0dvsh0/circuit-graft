# Experiment log

Chronological record of commands, outcomes and failures. Every stage entry
must include the config hash and manifest path once the stage pipeline exists.
No metric may be copied into this log from anywhere except `results/*.json`.

## 2026-11-05 - D1 environment

**Goal:** one Python 3.12 environment that resolves SafeTune, CircuitKIT,
AuditKIT and CuratorKIT; verify installed APIs before writing integrations.

**Commands (PowerShell):**

```powershell
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv\Scripts\python.exe safetune==0.1.12 `
  circuitkit==0.2.0 auditkit==1.1.3 "curatorkit[generation,hf]==0.1.22" `
  torch transformers transformer_lens
python -m uv pip install --python .venv\Scripts\python.exe -e . --no-deps
.venv\Scripts\cg.exe doctor --json-out results/d1/environment.json
```

**Outcome:** success. One environment, all four toolkits import. Python
3.12.15, torch 2.15.0+cpu, transformers 5.22.1, safetune 0.1.12,
circuitkit 0.2.0, auditkit 1.1.3, curatorkit 0.1.22.

**Failures and fixes:**

- `pytest` initially failed with `ModuleNotFoundError: pluggy` because the
  dev tools had been installed with `--no-deps`. Fixed with
  `python -m ensurepip --upgrade` followed by
  `pip install pluggy iniconfig coverage`.
- The venv had no `pip` at all after creation by uv. `ensurepip` repaired it.
- `uv` was not available on `PATH` in later shells; the locked environment is
  reproducible from `requirements-main.lock` with a normal Python 3.12
  interpreter plus pip.

**Artifacts:** `results/d1/environment.json`, `requirements-main.lock`.

## 2026-11-05 - Core implementation and CPU tests

**Goal:** implement the D1 slice of the specification: component vocabulary,
graft planning, protection strategies, Fisher estimation, pair validation,
decontamination, statistics, config schemas, manifests and CLI validation.

**Commands:**

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests scripts
```

**Outcome:** `58 passed`; ruff clean.

**Bugs found and fixed during test writing:**

1. Component partition validation compared slice coverage to the total element
   count instead of the length of the edited dimension. Fixed with
   `tensor_dim_size`.
2. Stage configs did not inherit `CommonConfig`; every stage now validates the
   merged `base.yaml`.
3. GQA K/V planning assigned one aggregate strength to a shared K/V group when
   a group mixed safety-only and overlap heads. Replaced with `PlannedEdit`
   entries that apply each subset's fraction separately.
4. `word_ngrams` returned a nested tuple for texts shorter than `n`, and
   `find_ngram_overlap` crashed when formatting the match. Short texts now
   return a normal tuple.
5. `graft_state_dict` dropped non-editable tensors from its output even though
   the docstring promised to preserve them. It now starts from `dict(drifted)`
   and replaces only edited tensors.

**Tests added:** `tests/test_components.py`, `test_graft.py`,
`test_protect.py`, `test_data.py`, `test_stats.py`, `test_config.py`.

**Coverage notes:** the bootstrap coverage test uses 50 fixed-seed synthetic
samples with 90% intervals and requires at least 40/50 to cover the true mean.
This is a smoke-level check, not the specification's 500-trial acceptance run;
the larger run belongs in the final test report.

## 2026-11-05 - Configuration validation

**Command:**

```powershell
Get-ChildItem configs -Recurse -Filter *.yaml | ForEach-Object {
  .venv\Scripts\cg.exe config-check $_.FullName
}
```

**Outcome:** all 18 shipped configs validate (base, data, drift, discover, eval,
attacks, 6 graft configs, 6 baseline configs). Invalid configs are rejected by
the pydantic schemas in tests.

## 2026-11-05 - Gated Tiny Aya access check

**Command:**

```powershell
.venv\Scripts\python.exe scripts\d1_smoke.py --only preflight `
  --output-dir results/d1/smoke_proxy
```

**Outcome:** blocked. `HF_TOKEN` is present and `HF_ORG` resolves to the
personal namespace, but both repositories return `GatedRepoError 403`:
`CohereLabs/tiny-aya-global` and `CohereLabs/tiny-aya-base`. The account is not
on either authorized list.

**Action required by the human lead:** open both model pages while logged in
as the token owner and accept the licences. Then rerun `cg doctor
--check-models`. G0 cannot run until this is granted.

No model card, config or tokenizer fact about Tiny Aya has been verified from
the real checkpoint yet.

## 2026-11-05 - D1 smoke runner, proxy mechanics run

**Goal:** prove the smoke runner and all four toolkit code paths work before
spending remote GPU time. This is explicitly **not** gate G0.

**Command:**

```powershell
.venv\Scripts\python.exe scripts\d1_smoke.py --proxy --device cpu `
  --max-examples 4 --max-seq-len 32 --max-new-tokens 16 `
  --output-dir results/d1/smoke_proxy
```

**Model:** `HuggingFaceTB/SmolLM-135M-Instruct` (public proxy, Llama-family).

**Outcome:** all four toolkit checks passed; `gate_g0 = not_evaluated` because
the model is not Tiny Aya.

| Check | Status | Observed |
|---|---|---|
| preflight | blocked | both Tiny Aya repos 403-gated; token present |
| SafeTune EAP-IG | ok | 32 units, 20 layers, `eap_scores.json` written |
| CircuitKIT `greater_than` | ok | 135 nodes, top node `MLP 0`, artifact `.pt` written |
| AuditKIT `hf:` eval | ok | 4 predictions, 0 failures, fingerprint present |
| CuratorKIT | ok | 8 rows -> 7 passed, 1 duplicate removed, `sft_alpaca.jsonl` written |

**Artifacts:** `results/d1/smoke_proxy/summary.json`,
`results/d1/smoke_proxy/summary.md`, one JSON log per check, plus the toolkit
artifacts under the same directory.

**Failures and fixes during this run:**

- The smoke script did not load `.env`, so preflight reported a missing token.
  Fixed by calling `load_dotenv(REPO_ROOT / ".env", override=False)`.
- `--only` children did not create the output directory when invoked directly.
  Fixed in `run_one`.
- CuratorKIT's `export_formats` is a list of schema names, not file
  extensions. `["jsonl"]` and `["corpus"]` wrote no accepted rows for
  instruction data. `["alpaca"]` with `instruction`/`input`/`output` works.
- CircuitKIT's logging emitted a UTF-8 checkmark that Windows cp1252 could not
  encode. The run completed and its JSON/artifact are valid; this is a
  Windows-console issue only.

## 2026-11-05/06 - Documentation and API inspection

**Commands:** source inspection with `inspect.signature`, `dir`, and direct
reads of the installed packages; `scripts/d1_smoke.py --help`.

**Outcome:** `docs/API_NOTES.md` records the installed signatures and the
discrepancies that affect later stages. The most important are:

- SafeTune 0.1.12 has functional `apply_ctheta` / `apply_resta`, not
  `CThetaTrainer` / `ReStaTrainer`.
- `apply_ctheta` consumes only layer/module suggestions, not per-head units.
  This threatens the "identical mask" premise of G3 and must be resolved.
- CuratorKIT export formats are named schemas (`alpaca`, `messages`, ...).
- CircuitKIT and SafeTune use different node names; an adapter is required.

`docs/PREREGISTRATION.md` is drafted from SPEC sections 3, 7, 9 and 13. It is
not frozen or committed yet and must not be treated as pre-registration until
that happens.

## 2026-11-06 - S6 checkpoint pipeline and circuit adapters

**Goal:** make the graft stage consume real HF checkpoints and discovery
artifacts without relying on the synthetic names used by the first tests.

**Commands:**

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests scripts
```

**Outcome:** `74 passed`; ruff clean. No experiment was run.

**Bugs found and fixed:**

1. **Editable tensor names lacked `.weight`.** `Cohere2Layout.attn_tensor` /
   `mlp_tensor` returned `model.layers.0.self_attn.q_proj`, while a real HF
   checkpoint (and `named_parameters()`) uses `...q_proj.weight`. The synthetic
   tests hid this. The layout now returns full parameter names, and a
   regression test asserts the `.weight` suffix.
2. **`Manifest.python` used a string as `default_factory`.** `Field(
   default_factory=sys.version.split()[0])` raised `TypeError: 'str' object is
   not callable` on the first real manifest. Fixed with a lambda.
3. The Fisher test's synthetic module registered parameters directly on
   `self_attn`; it now uses `nn.Linear` submodules so its names match a real
   model.

**Added:**

- `discover/overlap.py`: CircuitKIT/SafeTune/CircuitGraft namespace conversion,
  score-file validation, Jaccard and S/D/O overlap reports, SafeTune
  layer/module suggestion expansion.
- `graft/pipeline.py`: local/HF checkpoint resolution, safetensors loading
  (single and sharded), canonical score loading, graft execution, checkpoint
  saving, `graft_stats.json` and manifest lineage.
- `tests/test_overlap.py` and `tests/test_graft_pipeline.py`.

**Known limitation:** the S6 pipeline loads three full state dicts into RAM.
That is acceptable for CPU tests and high-RAM GPU runtimes, but a Tiny Aya run
needs roughly 3 x 6 GB in bf16 plus the output. A streaming writer is still a
D2/D6 task; do not claim the pipeline is memory-optimal.

## 2026-11-06 - S4 discovery runner and SafeTune EAP adapter

**Goal:** connect canonical pair files to CircuitKIT and SafeTune, write scores
in the CircuitGraft namespace, and make discovery a reproducible staged run.

**Commands:**

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests scripts
# all configs
Get-ChildItem configs -Recurse -Filter *.yaml | ForEach-Object { cg config-check $_.FullName }
```

**Outcome:** `93 passed`; ruff clean; all 18 configs validate. No GPU or Tiny
Aya run was possible (gated access still 403), so this is implementation
evidence only.

**Added:**

- `discover/safetune_eap.py`: canonical prompt extraction from pair JSONL,
  SafeTune `CircuitInfo` score extraction with explicit fallback order
  (`raw_output_path` -> activation correlation -> unit ids -> expanded layer
  suggestions), canonical score writing, coarse layer/module mask, and
  serialisation of SafeTune's own `CircuitInfo`.
- `discover/circuitkit_runner.py`: `GenericTaskSpec.from_jsonl` registration
  (hash-named, process-safe), `ck.load_model` + node-level `ck.discover`,
  canonical scores, selected-node mask, optional faithfulness, SafeTune
  cross-check, multi-budget overlap reports, `summary.json`, manifest and
  idempotency.
- `tests/test_safetune_eap.py` and `tests/test_circuitkit_runner.py`.

**Fixes / API details:**

1. CircuitKIT's score side-car uses the key `node_scores`; the shared score
   reader now accepts it.
2. CircuitKIT calls the per-head level `node`, while SafeTune EAP calls the
   same level `head`. The cross-check maps `node -> head`; passing `node`
   directly would be rejected by SafeTune.
3. Faithfulness pillar `causal_patching` is invalid upstream; the valid name is
   `patching`. Config defaults and `discover.yaml` were corrected, and
   `FaithfulnessConfig` now validates pillar names.
4. Added `n_examples`, `sparsity` and `overlap_fractions` to discovery configs
   and `n_examples` to faithfulness config.

**Limitations:**

- The runner has only been exercised with fake CircuitKIT objects and CPU
  tests. Its first real-model run is the G0/Tiny Aya smoke.
- Faithfulness exceptions are recorded in `summary.json` and the stage
  continues. The report stage must not quote a circuit without a completed
  faithfulness report; this is an explicit later check, not a silent pass.

## 2026-11-07 - S1 data stage and real-tokenizer smoke

**Goal:** implement the SPEC §6 data stage end to end: CuratorKIT-curated
MedMCQA SFT, held-out and transfer eval sets, contrastive pairs, 13-gram and
embedding decontamination, dataset cards, manifests and a partial-run gate.

**Commands:**

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests scripts
cmd /c "set PYTHONIOENCODING=utf-8&& .venv\Scripts\cg.exe data --config configs\smoke\data.yaml"
```

**Outcome:** `110 passed`; ruff clean; all 18 stage configs validate. The
real-tokenizer CPU smoke completed with a `partial` manifest, as intended.

**Smoke evidence (`artifacts/data_smoke/`, git-ignored):**

| Item | Result |
|---|---|
| Tiny Aya tokenizer | `CohereLabs/tiny-aya-base` loaded; 261,000 vocab |
| Pair answer tokens | `" I"` -> 399, `" Sure"` -> 44489, `" A"` -> 375, `" B"` -> 437; all single tokens |
| SFT | 200 CuratorKIT rows, 0 rejected, 50 held-out validation rows |
| Domain pairs | 32 pairs; 13-gram overlap with eval sets: 0 removed; tokenizer validation passed |
| Eval sets | MedMCQA validation, MedQA, PubMedQA, MMLU non-medical and OR-Bench-hard built |
| Gated benchmarks | HarmBench, AdvBench, SORRY-Bench, WildJailbreak and XSTest still blocked |
| Safety pairs | `not_generated` (no Cohere key; `enable_generation: false` in smoke config) |
| Domain-sensitive sets | `not_generated` (same reason) |
| Gate G1 | `not_evaluated`; manifest status `partial`, with explicit reasons |

**Fixes / API details:**

1. SafeTune's `load_medmcqa` renames `question` to `prompt`. The first smoke
   produced zero SFT rows until the converters accepted both names.
2. `run_data` now auto-loads the Tiny Aya tokenizer from `base.models.base`,
   and records its name in `pairs/validation.json`.
3. Eval decontamination builds the reference n-gram index once instead of
   rebuilding it for every candidate.
4. Pair and domain-sensitive review templates are written once and never
   overwritten by a rerun; completed pair reviews can flip `label_source` to
   `human` and reject bad twins. A completed domain review CSV is merged back
   before decontamination.
5. SFT section caching now includes decontamination and embedding-dedup
   settings, so changing either invalidates the cached export.
6. HarmBench test selection uses the 200 `standard` behaviours only; it does
   not pad with contextual or copyright subsets.

**Blocker:** `CohereLabs/tiny-aya-global` still returns `GatedRepoError 403`
even though `-base` is accessible. G0 and every experiment above S1 remain
blocked until that grant propagates and a CUDA runtime is available.

## 2026-11-07 - S2 drift stage

**Goal:** implement SPEC §7: LoRA SFT of θ_A on the CuratorKIT-curated benign
medical rows, an optional explicitly supplied Scenario B contamination, merged
checkpoint output, manifest lineage, and the G-drift gate over θ_A/θ_D metrics.

**Commands:**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_drift.py -q
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests
```

**Outcome:** 14 drift tests pass; the full suite is `135 passed` with one
upstream TransformerLens deprecation warning; ruff is clean. No real training
run was possible: there is no CUDA device and the aligned Tiny Aya checkpoint
is still gated.

**Implementation:**

- `drift/finetune.py` loads `train.jsonl` / `validation.jsonl` from the S1
  messages export, renders the tokenizer's chat template explicitly, labels
  assistant tokens only, masks prompt tokens with `-100`, and trains LoRA
  through TRL's `SFTTrainer` with `dataset_kwargs={"skip_prepare_dataset":
  True}` plus a custom padding collator.
- Scenario A uses benign rows only. Scenario B requires
  `DriftConfig.harmful_rows` (an explicit JSONL; the validator rejects B
  without it) and replaces a seeded fraction of benign rows with those rows.
  CircuitGraft never generates harmful examples.
- The run writes `adapter/`, an optional `merged/` checkpoint, `trainer/`,
  `training_args.json`, `drift_summary.json` and a manifest. `latest.json`
  points S5–S7 at the newest scenario.
- `evaluate_drift_gate` / `apply_drift_gate` accept both the compact and
  per-benchmark metric layouts and require a domain-accuracy gain of at least
  0.03 *and* a HarmBench refusal drop of at least 0.15, with reason strings
  when either is missing.

**Tests added:** `tests/test_drift.py` covers message normalisation, harmful
row conversion, contamination fraction/seed behaviour, chat rendering and
assistant-only labels, padding, the Scenario B config guard, summary/manifest
writing, checkpoint skip logic and gate evaluation.

## 2026-11-07 - S7 evaluation pipeline and proxy mechanics smoke

**Goal:** implement SPEC §9/§12: checkpoint resolution, cached greedy
generation, MCQ logits, SafeTune judging, AuditKIT precomputed runs and
red-team probes, pooled metrics, SRR/DGR/ORP composites, the G-drift gate,
paired McNemar + Holm tests, a human-audit template, and run manifests.

**Commands:**

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests
Get-ChildItem configs -Recurse -Filter *.yaml | ForEach-Object {
  .venv\Scripts\cg.exe config-check $_.FullName --kind auto
}
```

**Outcome:** `135 passed`; ruff clean; all 20 shipped configs validate. The
proxy mechanics smoke below completed end to end.

**Implementation:**

- `eval/models.py`: logical model resolution (explicit `eval.model_paths`
  first, then completed stage artifact directories), safetensors/LoRA loading,
  dtype/device handling and content fingerprints.
- `eval/generate.py`: one checkpoint at a time, cached greedy rollouts under
  `results/generations/<model>/<benchmark>.jsonl`, generation manifests, chat
  template/preamble handling, and first-step MCQ letter logits.
- `eval/domain.py`: `Answer: X` -> standalone option -> logit fallback,
  accuracy and parse-failure CIs, and per-item outcomes with response hashes
  instead of full responses.
- `eval/safety.py`: SafeTune `StringMatchJudge(mode="prefix")` and optional
  WildGuard `HFJudge`, HRR/FRR, per-model human-audit CSVs with Cohen's kappa,
  and `run_audit` for S3 (SafeTune `evaluate` when the primary judge is
  available; AuditKIT `RedTeamRunner` always).
- `eval/auditkit_runs.py`: `ak.evaluate(..., "precomputed")` over cached MCQ
  generations with `FunctionScorer` accuracy/parse-rate, per-run directories
  with AuditKIT provenance, `ak.compare` bootstrap comparisons, and the
  red-team wrapper around the loaded HF model.
- `eval/pipeline.py`: generation -> judging -> pooling -> composites -> gate ->
  McNemar/Holm -> AuditKIT comparisons -> human-audit templates -> summary and
  manifest. Missing benchmarks, judges, AuditKIT failures or a pending human
  audit force `status = partial` with explicit reasons.

**Proxy smoke (explicitly not G4 and not Tiny Aya):**

| Item | Value |
|---|---|
| run | `eval-e71c7a6ce0c41416` |
| aligned | `HuggingFaceTB/SmolLM-135M-Instruct` |
| drifted | `HuggingFaceTB/SmolLM-135M` |
| benchmarks | `orbench_hard`, `medmcqa_validation`, `mmlu_non_medical` (4 rows each) |
| judge | SafeTune string-match |
| observed HRR/FRR | FRR 0.0 for both models; no safety benchmark, so HRR n/a |
| observed accuracies | domain 0.0, general 0.0 for both models |
| AuditKIT | per-benchmark plus pooled `domain_mcq` / `general_mcq` runs and `ak.compare` |
| `gate_g_drift` | `not_evaluated` (no safety benchmark in the smoke data) |
| human audit | pending, so the summary is `partial` in a full run; this smoke had no generated safety rows |

The zero accuracies are the real output of the 135M proxy models on four-row
samples. They must never be quoted as Tiny Aya, safety-drift or CircuitGraft
recovery evidence. The summary, manifest and AuditKIT comparisons are preserved
under `results/s7/smoke_proxy/`; item files and AuditKIT RunResults stay in the
git-ignored `artifacts/` tree and are hashed by the manifest.

**Bugs found and fixed:**

1. The human-audit lookup read a non-existent top-level `safety` key instead of
   `judged.safety`; it would have produced empty review templates.
2. `_metric_doc` did not strip per-item rows from the saved summary in every
   path. Per-item outcomes now always live in separate JSONL files so full
   harmful generations are not published accidentally.
3. AuditKIT `RedTeamResult.total_probes` / `passed_probes` / `failed_probes` /
   `pass_rate` are attributes, not methods.
4. AuditKIT writes one `lexsi_provenance.json` per folder, so every
   `(model, benchmark)` run gets its own directory.
5. The pooled AuditKIT runs needed benchmark-prefixed IDs to avoid collisions
   between `medmcqa_validation` and `mmlu_non_medical` row IDs.

## 2026-11-07 - S5 baselines implementation and offline tests

**Goal:** implement the S5 baselines stage (`circuitgraft.baselines.safetune_recover`)
covering SafeTune Faithful methods (`resta`, `wiseft`, `ctheta`, `safelora`,
`lox`, `task_arith`), manifest lineage, circuit adapter integration for C-ΔΘ,
and offline verification tests.

**Commands:**

```powershell
.venv\Scripts\pytest -q
.venv\Scripts\ruff.exe check src tests
Get-ChildItem configs -Recurse -Filter *.yaml | ForEach-Object {
  .venv\Scripts\cg.exe config-check $_.FullName
}
```

**Outcome:** `148 passed`, 1 warning (upstream TransformerLens deprecation);
ruff clean; all configs pass validation.

**Implementation:**

- `baselines/safetune_recover.py`: functional dispatch for `resta`, `wiseft`,
  `ctheta`, `safelora`, `lox`, `task_arithmetic`.
- Explicit rejection of `refusal_steer` with `NotImplementedError`
  (inference-time activation steering, not a SafeTune recover function).
- Labelling C-ΔΘ as `"C-ΔΘ (SafeTune weight-arithmetic implementation)"`
  everywhere per SPEC §2.1.
- Circuit bridge integration: `apply_ctheta` loads CircuitKIT discovery output
  via `safetune.core.circuit_kit.adapter.load_circuit_info_from_file`, records
  the source scores SHA256 lineage, and profiles the parameter mask at
  layer/module granularity (explicitly documenting the limitation that SafeTune
  does not consume per-head units).
- Checkpoint directory layout: `artifacts/baselines/<method>-<config_hash>/merged`
  with `manifest.json` and `baseline_stats.json`. S7's `artifacts/baselines/{name}*`
  checkpoint resolver matches these outputs.
- `resolve_stage_file` in `circuitgraft.registry`: allows stage configs to
  reference stable paths like `artifacts/discover/safety/circuit_info.json`
  which dynamically resolve to the latest completed discovery run.
- CLI: `cg baselines` supports `--config`, `--dry-run`, and `--sweep`.
- Offline test suite: 13 tests in `tests/test_baselines.py` covering parameter
  validation, sweep cartesian products, dispatch kwargs, manifest inputs/outputs,
  idempotent skipping, C-ΔΘ circuit lineage checks and mask summaries, and
  Safe LoRA merged vs. PEFT modes.

## 2026-11-07 - Real Tiny Aya & gated dataset access confirmed; S8 attacks implementation

**Goal:** verify unblocked access to `CohereLabs/tiny-aya-global`, the 5 gated
eval datasets, and `COHERE_API_KEY`; verify real architecture against
`Cohere2Layout`; provide Colab G0 execution notebook; implement S8 attacks
stage (`abliterate`, `harmful_ft`, `quantize`) with offline tests.

**Access check outcomes:**

- `CohereLabs/tiny-aya-global`: accessible (status: ok).
- `CohereLabs/tiny-aya-base`: accessible (status: ok).
- Gated datasets: `walledai/HarmBench`, `walledai/AdvBench`,
  `sorry-bench/sorry-bench-202503`, `allenai/wildjailbreak`, `walledai/XSTest` all verified accessible.
- `COHERE_API_KEY`: verified active with `c4ai-aya-expanse-32b` via LiteLLM.
- Model config introspection: `cohere2` architecture with 36 layers, 2048 hidden,
  16 heads, 4 KV heads, 128 head dim, 11008 intermediate size. Matches `Cohere2Layout` exactly.
- Tokenizer chat template: verified present; system preamble renders to 367 tokens.
- Real preflight recorded: `results/d1/preflight_real/preflight.json` (`status = ok`).
- Colab GPU smoke notebook: created `notebooks/run_d1_g0_smoke.ipynb` to execute
  `scripts/d1_smoke.py --device cuda --output-dir results/d1/smoke` on Colab GPU.

**S8 Implementation:**

- `attacks/abliterate.py`: SafeTune `AbliterationAttack` wrapping `fit(harmful, harmless)`
  and `run(mode="weight_orthogonalize" | "runtime_ablate")`. Adaptive direction fitting
  on each target model, contrast prompt loader, weight saving, and manifest lineage.
- `attacks/harmful_ft.py`: LoRA re-fine-tuning attack on harmful-compliance examples
  for 50 / 100 / 200 steps at lr 1e-4. Policy-enforced: requires explicit harmful rows
  JSONL and marks status="blocked" without it; never generates harmful content.
- `attacks/quantize.py`: Uniform 4-bit (NF4) and 8-bit quantization via bitsandbytes;
  circuit-aware mixed-precision plan protecting top safety-circuit layers via
  `compute_circuit_aware_plan`.
- `attacks/runner.py`: top-level `run_attack(cfg, base, attack_type="...")` dispatcher.
- `tests/test_attacks.py`: 8 offline tests covering prompt loading, abliteration
  weight orthogonalization, runtime ablation hooks, harmful-FT policy blocking,
  explicit training mock, circuit-aware layer protection plans, and CLI dispatch.

**Outcome:** `156 passed`, 1 warning (upstream TransformerLens deprecation);
ruff clean; all configs pass validation.

## 2026-11-07 - Gate G0 Passed on Real Tiny Aya (Tesla T4 CUDA)

**Goal:** Execute D1 smoke test on real Cohere Tiny Aya (`CohereLabs/tiny-aya-global`
and `CohereLabs/tiny-aya-base`) on a GPU runtime with accepted licences, proving all
four toolkits (SafeTune, CircuitKIT, AuditKIT, CuratorKIT) run end-to-end on the target
model.

**Command (Google Colab, Tesla T4 GPU, Python 3.13 / CUDA 13.0):**

```bash
python scripts/d1_smoke.py --device cuda --output-dir results/d1/smoke
```

**Outcome:** `D1 G0: passed - all four toolkits ran on Tiny Aya`.

| Check | Status | Wall clock (s) | Log / Details |
|---|---|---|---|
| `preflight` | `ok` | 19.9 | Verified access to `CohereLabs/tiny-aya-global` and `CohereLabs/tiny-aya-base` |
| `safetune` | `ok` | 52.7 | Discovered 32 safety units across 28 layers using EAP-IG on aligned Tiny Aya |
| `circuitkit` | `ok` | 69.8 | Discovered `greater_than` node-level circuit using HookedTransformer on Tiny Aya |
| `auditkit` | `ok` | 0.2 | Generated and scored 4 benign test samples via `hf:CohereLabs/tiny-aya-global` |
| `curatorkit` | `ok` | 0.4 | Curation pipeline (dedup, cleaning, export) executed cleanly |

**Engineering Challenges & Fixes:**

1. **TransformerLens float32 upcast spike on 3.5B model:** In `transformer_lens.weight_processing`,
   tensors were unconditionally upcast to float32 even when folding/centering was False,
   creating a 14 GB spike that triggered OOM on the 14.56 GB T4 GPU. Fixed with
   `patch_transformer_lens_weight_processing()`.
2. **CircuitKIT architecture patch dependency:** CircuitKIT patches `transformer_lens` on
   startup to register `Cohere2ForCausalLM`. Calling `patch_transformer_lens_weight_processing()`
   before `import circuitkit` triggered an `ImportError`. Fixed by enforcing `import circuitkit`
   first, followed by the weight processing patch.
3. **TransformerLens `torch_dtype` omission:** In `loading_from_pretrained.py`, TransformerLens
   deleted `torch_dtype` from kwargs and passed `dtype=...`, which HuggingFace ignored,
   causing HuggingFace to load weights in float32. Fixed by wrapping `load_model` in
   `torch.set_default_dtype(torch.bfloat16)` and monkey-patching `AutoModelForCausalLM.from_pretrained`
   to ensure `torch_dtype=torch.bfloat16`.
4. **VRAM activation bounding:** Default Tiny Aya context is $n_{\text{ctx}} = 8192$. Capped
   `model.cfg.n_ctx = min(n_ctx, 64)`, set `requires_grad = False` on non-embedding weights,
   and scoped discovery to 1 example, reducing peak backward memory to <150 MB.

**Artifacts:** `results/d1/smoke/summary.json`, `results/d1/smoke/summary.md`,
`results/d1/smoke/preflight.json`, `results/d1/smoke/safetune.json`,
`results/d1/smoke/circuitkit.json`, `results/d1/smoke/auditkit.json`,
`results/d1/smoke/curatorkit.json`.

## 2026-11-08 - S9/S10 real-artifact integration and `run_all.sh`

**Goal:** make the post-experiment stages consume the files the earlier stages
actually write, and add the missing end-to-end runner required by G6.

**S9 reporting changes:**

- `report/tables.py`: normalise the real S7 summary shape
  (`{"models": {name: {"composite": ..., "judged": {"safety", "mcq", "benign"}}}}`)
  into one table row per model; accept both flat judge documents and the older
  nested `metrics` wrapper; read edit footprints from `graft_stats.json` and
  `baseline_stats.json`.
- `report/pipeline.py`: discover the newest complete/partial
  `artifacts/eval/eval-*`, `artifacts/discover/discover-*` and
  `artifacts/attacks/*/*/attack_stats.json`; ignore failed manifests; attach
  clean HRR from the eval summary to attack rows; copy tables and plots into
  `results/s9/<run_id>/`; point README image links at the tracked plot copies
  because `artifacts/` is git-ignored; add a `repo_root` injection point for
  hermetic tests.
- `report/inject_readme.py`: replace via a function so Windows paths containing
  `\U` are not interpreted as regex replacement escapes.
- CLI: `cg report --output-dir/--readme/--results-out`.

**S10 publishing changes:**

- `publish.py`: resolve actual drift (`*/merged`), LoRA (`*/adapter`), graft
  (`<name>-*/model.safetensors`), baseline (`*/merged`) and abliteration
  (`attacks/abliterate/*/merged`) checkpoints instead of fixed paths.
- Dataset staging uses allow-lists, so the safety-pairs and domain-pairs repos
  receive only their own files from the shared `artifacts/data/pairs` folder;
  eval logs exclude `results/generations`.
- CLI: `cg publish --output-dir`; dry runs now execute the full resolution and
  staging path and report published/skipped counts.

**Runner:** `scripts/run_all.sh` implements:

- `--check`: doctor, every `configs/**/*.yaml`, pytest, ruff.
- `--smoke`: CPU D1 proxy smoke, S1 smoke data, S9 into a scratch README, S10
  dry run under `artifacts/run_all_smoke/`.
- `--full`: CUDA gate, doctor `--check-models`, config check, then
  data -> drift -> audit -> discover -> baselines -> graft -> eval -> attacks ->
  report -> publish; supports `--no-publish`, `--no-attacks`,
  `--sweep-baselines`, `--device` and `--dry-run`.

**Verification (local, CPU-only):**

- `pytest -q`: `170 passed`, 1 upstream TransformerLens deprecation warning;
  `ruff check src tests scripts`: clean.
- `bash scripts/run_all.sh --check`: `check: ok` (21 configs validated).
- `cg publish --dry-run` against the current tree: 0 models, 1 dataset
  (`eval-logs`), 19 skipped. The skips are the expected missing Phase 4
  checkpoints/data; this is not S10 evidence.
- A temporary S9 run against `artifacts/eval_smoke` reported `partial` with
  reasons `no S4 discovery overlap report` and `no S8 attack statistics`. It
  wrote only to a temporary directory, so the public README markers were not
  changed and the smoke values were not presented as G4 evidence.

**Artifacts:** `scripts/run_all.sh`, updated S9/S10 modules and tests,
`CONTEXT.md`.

## 2026-11-08 - Space loaders, article scaffold, and lint coverage

**Goal:** finish the human-facing deliverables that can be completed without a
GPU, without inventing Phase 4 numbers.

**Space (`space/app.py`):**

- Read-only Gradio demo over the newest tracked `results/s9/report-*` bundle.
  It renders the main/overlap/attack tables and plots, and shows setup
  instructions when no bundle exists.
- Optional Compare tab reads `space/cached_responses.json`; it never generates
  live text and explicitly excludes harmful completions.
- Bundle discovery checks `CIRCUITGRAFT_REPORT_DIR`,
  `CIRCUITGRAFT_RESULTS_DIR`, then `results/s9/` beside the app and in its
  parent. The newest bundle is chosen by manifest `finished_at`, then mtime.
- Gradio is imported lazily in `build_app()`, so loader tests run without the UI
  dependency. `pyproject.toml` now exposes a `space` extra
  (`gradio>=5.0`).
- `tests/test_space_app.py` covers latest-bundle discovery, table/plot
  fallbacks, invalid/missing manifests, environment override, cached-response
  loading, prompt/model helpers and partial-status rendering.

**Article:**

- `docs/ARTICLE.md` follows the 11-section outline in `SPEC.md` §16.2 and uses
  explicit `ARTICLE:` placeholders for every metric-bearing block and artifact
  link. It records the S8 post-attack caveat and the S9 completeness rule
  instead of pre-filling values.

**Tooling:**

- Ruff now covers `space/` as well (`src tests scripts space`), and
  `scripts/run_all.sh --check` uses the same scope.

**Verification (local, CPU-only):**

- `pytest -q`: `183 passed`, 1 upstream TransformerLens deprecation warning.
- `ruff check src tests scripts space`: clean.
- `bash scripts/run_all.sh --check`: `check: ok` (doctor, 21 configs, tests,
  lint).
- Gradio is not installed in the local environment; only the lazy loader helpers
  were tested. `build_app()` itself remains unexercised until the Space is
  deployed.

**Artifacts:** `space/app.py`, `space/README.md`, `space/requirements.txt`,
`tests/test_space_app.py`, `docs/ARTICLE.md`, `pyproject.toml`,
`scripts/run_all.sh`, `CONTEXT.md`.

## 2026-11-08 - S8 -> S7 attack checkpoint evaluation bridge

**Goal:** close the missing robustness path. S8 saves attacked checkpoints but
cannot measure HRR; S9 previously had to print `-` for every post-attack cell.
The bridge re-evaluates those checkpoints through the normal S7 pipeline and
passes measured metrics to S9.

**Implementation:**

- `src/circuitgraft/attacks/evaluate.py` discovers complete/partial S8 runs and
  resolves saved checkpoints: abliteration `output.merged` and every harmful-FT
  `checkpoints.<step>`. Blocked runs, runtime-only abliteration and
  quantization plans are skipped explicitly.
- Eval names are deterministic and filesystem-safe
  (`attack__<type>__<model>[__mode][__step_<n>][__backend]`); duplicate names get
  a numeric suffix. Every checkpoint is passed to S7 through `model_paths`, with
  isolated `artifacts/attacks/eval/` and `results/generations_attacks/`
  directories.
- The bridge writes `post_eval.json` beside each `attack_stats.json`
  (`metrics` for single-checkpoint attacks, `metrics_by_step` for harmful-FT),
  plus aggregate `artifacts/attacks/eval/summary.json` and a manifest.
- S9 now reads `post_eval.json` and merges measured HRR into
  `abliteration_hrr`, `harmful_ft` step rows and the quantization columns, and
  records each source under `attack_post_eval`.
- CLI: `cg eval-attacks [--config] [--attacks-dir] [--output-dir]
  [--generations-dir] [--overwrite] [--dry-run]`.
- `scripts/run_all.sh --full` now runs `attack -> eval-attacks -> report`;
  `--no-attack-eval` skips the bridge and the summary prints the
  `artifacts/attacks/eval/summary.json` path.

**Verification (local, CPU-only):**

- `pytest -q`: `187 passed`, 1 upstream TransformerLens deprecation warning.
- `ruff check src tests scripts space`: clean.
- `bash scripts/run_all.sh --check`: `check: ok`.
- `bash scripts/run_all.sh --full --dry-run`: prints the expected sequence
  including `eval-attacks`, `report` and `publish`.
- No attack checkpoint existed locally, so the bridge itself was exercised with
  injected checkpoint trees and a fake S7 runner in `tests/test_attack_eval.py`;
  the real post-attack numbers await the GPU run.

**Limitations:** quantization still writes only a plan, so no quantized
checkpoint exists to evaluate; runtime abliteration likewise has no saved model.
Those cells remain `-` until the attack stages save real checkpoints.

**Artifacts:** `src/circuitgraft/attacks/evaluate.py`,
`tests/test_attack_eval.py`, `src/circuitgraft/report/pipeline.py`,
`tests/test_report.py`, `src/circuitgraft/cli.py`, `scripts/run_all.sh`.

## 2026-11-08 - Phase 4 Colab runner and pre-registration freeze

**Goal:** freeze pre-registration before Phase 4, package the Google Drive-persisted
Colab notebook, and prepare the Git tree for remote execution on Tesla T4.

**Implementation:**

- `docs/PREREGISTRATION.md`: status transitioned from DRAFT to FROZEN.
  Checklist verified (Gate G0 passed on T4, API notes reconciled, clean tree).
- `scripts/build_phase4_notebook.py`: programmatic generator for
  `notebooks/run_phase4_colab.ipynb`.
- `notebooks/run_phase4_colab.ipynb`: 10-stage resumable runner featuring Google Drive
  real-time artifact persistence (`circuit_graft_phase4/`), 10 GB disk swapfile,
  15-minute quick calibration pass, and staged execution of S1 through S10.
- `pyproject.toml`: expanded `requires-python` from `>=3.12,<3.13` to `>=3.12,<3.14`
  and added Python 3.13 classifier to support Google Colab's default Python 3.13.18
  runtime.
- `src/circuitgraft/data/curate.py`: silenced LiteLLM debug banner spam
  (`os.environ['LITELLM_LOG'] = 'ERROR'`, `suppress_debug_info = True`), added
  request throttling (3.2s spacing) and exponential backoff for Cohere Trial keys
  (20 req/min limit), and implemented deterministic semantic neutralization fallback
  so safety pair generation completes cleanly without hanging or failing under rate limits.

**Verification (local, CPU-only):**

- `pytest -q`: `187 passed`, 1 upstream TransformerLens deprecation warning.
- `ruff check src tests scripts space`: clean.
- `python scripts/build_phase4_notebook.py`: verified JSON format and cell structure.

**Artifacts:** `notebooks/run_phase4_colab.ipynb`, `scripts/build_phase4_notebook.py`,
`docs/PREREGISTRATION.md`.

## Open items

1. Run `cg data` with unblocked benchmarks and active Cohere API key, complete 10%
   human-review contrastive pairs CSV, and claim G1.
2. Review and freeze `docs/PREREGISTRATION.md` (transition DRAFT -> frozen) before Phase 4.
3. Execute Phase 4 experiments on remote GPU (or `bash scripts/run_all.sh --full`):
   - S2: Train drifted specialist $\theta_D$ (`cg drift`).
   - S3: S3 Audit of $\theta_A$ vs $\theta_D$ (confirming H0 safety degradation).
   - S4: Discovery of safety circuit $S$ and domain circuit $D$ with CircuitKIT & SafeTune EAP cross-check (`cg discover`).
   - S5: Baselines (RESTA, Wise-FT, C-ΔΘ).
   - S6: CircuitGraft variants (`cg_project`, `cg_uniform`, `cg_exclude`, `cg_fisher`).
   - S7: Full evaluation matrix (`cg eval`).
   - S8: Adversarial attacks (`cg attack`).
   - S8->S7: Evaluate saved attack checkpoints (`cg eval-attacks`).
   - S9: Generate report tables, plots, and inject README metrics (`cg report`).
   - S10: Publish models & datasets to HuggingFace under `dubeydevansh` (`cg publish`).
4. After a complete S9 bundle exists, copy it into the Space repository, add
   curated `space/cached_responses.json` once checkpoints are public, and deploy
   `space/app.py`. Fill `docs/ARTICLE.md` from `results/s9/<run_id>/` and replace
   every artifact placeholder after S10.
5. Run `cg eval-attacks` after `cg attack` so `table_attacks.md` and
   `attack_decay.png` carry measured post-attack HRR. `run_all.sh --full`
   already does this; quantization stays unavailable until a real quantized
   checkpoint is saved.

## 2026-11-13 - Emergency local Phase 4 plan (AMD RX 6600, no cloud quota)

**Goal:** run the full Phase 4 matrix locally within one hackathon day after
Kaggle/Colab quotas stopped the research mid-run. Machine: i3-12100F (4c/8t,
16 GB RAM), AMD Radeon RX 6600 8 GB, Windows 11. No CUDA.

**Decision (human-approved, logged as prereg deviations BEFORE any test-split
run):** SFT rows 20000 -> 5000; S7 test rows capped (~1,050/model,
HarmBench 150); selection split 60/60/120; sweeps pruned
(RESTA 2, WiSE-FT 2, C-Theta 2, SafeLoRA default+1, CG 5 points);
cg_random 3 seeds on test at the chosen cg_project setting; string-match judge
(frozen-config CPU fallback; WildGuard `not_run`); slim attacks
(abliteration always, harmful-FT small, quantization only if bitsandbytes-ROCm
installs).

**Compute path:** official AMD PyTorch-ROCm wheels for gfx103X (TheRock via
`https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/`): torch
2.11.0+rocm7.14.0a20260612 + rocm-sdk 7.14.0a20260612 in the repo venv
(replaces torch 2.15.0+cpu; all other pins unchanged; transformers 5.22.1
requires torch>=2.6). `HSA_OVERRIDE_GFX_VERSION=10.3.0` maps gfx1032 onto the
gfx1030 code objects; math-SDPA env vars set because flash/mem-efficient CUDA
kernels do not exist for gfx1032. HF caches moved to D: via `.env`
(HF_HOME/HF_HUB_CACHE; datasets stay on C: where they are already populated).
Tiny Aya weights (~14 GB) downloaded to D:. This log entry is written before
the first Phase 4 run.

**Planned sequence:** S1 data -> S2 drift (GPU) -> S3 audit -> S4 discovery
(GPU) -> S5 baselines + S6 graft (CPU weight arithmetic, mmap) -> selection
eval (GPU) -> preregistered selection rule -> final eval 12+3 models (GPU) ->
S8 attacks + eval-attacks -> S9 report -> S10 publish.

**2026-11-13 afternoon (prep while weights/SDK download):** added
`configs/eval_selection.yaml` (17 eval names: aligned+drifted anchors, 15
sweep-point candidates per the deviation table; benchmarks
selection_harmbench/selection_xstest_safe/selection_medmcqa_validation at
60/60/120; separate `results/generations_selection` cache;
`artifacts/eval_selection` output). Sweep runs are hashed directories
(`artifacts/baselines/resta-<hash>`), so glob resolution cannot disambiguate
sweep points; added `scripts/wire_selection_paths.py`, which maps completed
runs to preregistered logical names from `baseline_stats.json`/`graft_stats.json`
and fills the marker-delimited `model_paths` block (fails loudly when a sweep
point is missing or ambiguous). Added `scripts/prepare_selection_data.py` to
copy the S1 selection split to `artifacts/data/eval/selection_*.jsonl` (verify
with `--check`). Added `scripts/gpu_bench.py` (dtype decision: fp16 vs bf16
matmul, eval-style generation batch 8, S2-style LoRA step; writes
`results/gpu_bench.json`; a bf16 emulation slowdown on gfx1032 triggers the
float16 switch + prereg deviation note). Stage order note: `run_all.sh --full`
defaults to `SWEEP_BASELINES=0`; S5 must run as
`cg baselines --config configs/baselines/<x>.yaml --sweep` per method.

**2026-11-13 16:00 (ROCm stack operational):** pip's CDN downloads kept dying
mid-wheel ("Wheel 'rocm-sdk-core' ... is invalid" — no resume in pip); wrote
`scripts/download_rocm_wheels.py` (6 parallel byte-range chunks, per-chunk
resume `.partN`, zip-integrity check) and pulled
`rocm_sdk_core` + `rocm_sdk_libraries_gfx103x_all` (710.4 + 224.4 MiB).
`rocm_sdk_devel` (1.6 GiB) deliberately skipped: compile-time headers only,
not needed to run torch — saves the largest single download on the throttled
link. Installed `rocm==7.14.0a20260612` (metapackage, pip-cached sdist),
`rocm-sdk-core==7.14.0a20260612`, `rocm-sdk-libraries-gfx103x-all==7.14.0a20260612`
alongside the already-installed torch 2.11.0+rocm7.14.0a20260612.
`scripts/torch_smoke.py`: `cuda.is_available()=True`, AMD Radeon RX 6600,
HIP 7.14.60850, `HSA_OVERRIDE_GFX_VERSION=10.3.0` active; fp16 matmul
max_err 0.063, bf16 matmul max_err 0.5 (both consistent with normal rounding,
not emulation), backward pass OK. `cg doctor` exit 0 (all libraries present;
stderr carries a harmless bitsandbytes `rocminfo`-probe warning).
`pytest -q`: 195/195 after fixing `test_run_audit_reports_not_run_primary_but_keeps_auditkit`
— the test assumed a no-CUDA machine; `_guard_device` correctly returns
"cuda" on this box, so the test now monkeypatches `_guard_device` to None to
exercise the intended branch explicitly (test-only change; no pipeline
change). Tooling note: PowerShell 5.1 `2>&1` merge reports spurious exit 1
for native stderr — use separated redirection to read true exit codes.

**2026-11-13 16:50 (S1 data complete; HarmBench selection resplit):**
`cg data --config configs/data.yaml` finished at 10:59:27Z (started 10:43:41Z,
945.7 s wall), manifest `artifacts/data/manifest.json`: run_id
`data-bc99df1d4f173b72`, git 4cbfb8d (clean), device cuda (AMD Radeon RX 6600),
status complete. Cohere Aya-Expanse-32B generation ran both QA tasks (300
benign + 150 harmful rows at ~1.03 rows/s); outputs: SFT 5000 train / 1000
val rows, safety+domain pairs 256+256, decontam removed only MedMCQA/MedQA/
MMLU overlaps (13-gram), test split 4,940 rows across 12 benchmarks, selection
split 350 rows. The "exit code 1" seen at launch was the PowerShell `2>&1`
stderr-merge artifact (only stderr content: the harmless bitsandbytes
`rocminfo` warning) — the manifest status is authoritative.

The stage manifest `notes` said `partial_ok` because the selection split had
`harmbench.jsonl` empty: HarmBench's standard-behaviour pool is exactly 200
rows and the config requested 200 test rows, so `_split_indices` handed the
whole seeded shuffle to the test split (prereg-approved selection size was
60). Fix applied BEFORE any test-split run, as a logged preregistration
deviation: `scripts/resplit_harmbench_selection.py` carved the split from the
finished file — test keeps the preregistered 150-row cap
(`configs/eval.yaml max_rows.harmbench`), the remaining 50 rows become
`artifacts/data/selection/harmbench.jsonl`. Byte-identity to a fresh
corrected-config run was proven at run time: the replayed
`load_safetune_rows("load_harmbench")` → `_as_rows` → `safety_to_eval` → dedup
→ standard-subset path yields exactly 200 rows and
`random.Random("0:harmbench").shuffle(pool)` reproduces the file order
exactly, so first-150/last-50 equals the fresh-run `shuffled[:150]` /
`shuffled[150:]` split. Lines moved verbatim (canonical `_write_jsonl` form
asserted); `summary.json` + `manifest.json` re-hashed (3 entries) with the
resplit noted. Updated `configs/data.yaml` (harmbench 150,
selection_harmbench 50), `configs/eval.yaml` (selection_harmbench 50) and
`configs/eval_selection.yaml` (max_rows.selection_harmbench 50, comment
50/60/120). Final split sizes: test 4,940 rows; selection 50 / 100 / 200.
Also deleted the two orphaned `.incomplete` blobs (4.25 GB) from
`models--CohereLabs--tiny-aya-base/blobs` — the base snapshot is complete
(both shards verified against `model.safetensors.index.json` total_size).

**2026-11-13 17:15 (GPU bench: dtype verdict + LoRA VRAM findings):**
`scripts/gpu_bench.py` on the RX 6600 (full numbers in
`results/gpu_bench.json`): fp16 matmul 13.07 TFLOPS vs bf16 3.51 TFLOPS —
**bf16 matmul is emulated on gfx1032** (3.7x gap); eval-style generation
(batch 8) 48.8 tok/s MCQ-32 / 66.9 tok/s safety-256 in fp16 vs 21.5 / 14.9 in
bf16 (2-4.5x). Verdict: `float16` for the eval/generation stages — logged as a
preregistration deviation (hardware reason; every model evaluated in the same
dtype so metric semantics are unchanged); `configs/eval.yaml` +
`configs/eval_selection.yaml` flipped to `float16`. S2/S4/S6/S8 keep bf16 for
numerical stability (`configs/drift.yaml train.bf16: true`,
discover/graft inherit `dtype: bfloat16` from `configs/base.yaml`).

The LoRA phase initially OOM'd at the preregistered worst-case shape
(batch 4 x seq 1024): the fp32 loss logits over the 256k vocab allocate
batch*len*256000*4B = 4.2 GiB per fwd/bwd, on top of the 6.7 GB bf16 base.
Re-run at batch 2 x seq 512 (both dtypes): step time **~43 s** and peak
10.3 GB — the working set exceeds the 7.98 GB card, so Windows WDDM spills to
shared memory and thrashes PCIe (fp16 ≈ bf16 step time confirms it is
memory-bound). A 3.5B-model LoRA on this card needs a quantized frozen base.
Measured the real SFT token-length distribution with the global tokenizer:
mean 550, p50 514, p90 686, p95 791, p99 1020, max 1930 (most batches pad near
the 1024 cap — the worst-case OOM would hit in the real run too).

Config consequences (metric-neutral, effective batch unchanged):
`configs/drift.yaml` micro-batch 4 → 2, grad_accum 8 → 16 (effective batch 32,
LR schedule and optimizer-step count identical); added
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to `.env` (a bf16 bench
step had 2.17 GiB stranded in reserved-but-unallocated). Empirical QLoRA probe
(`scripts/quant_lora_probe.py`) launched to size the training path
(NF4 frozen base + bf16 LoRA across shapes).

**2026-11-13 17:55 (QLoRA probe results; AMP banned for S2):**
`scripts/quant_lora_probe.py` on the RX 6600 — all shapes consolidated in
`results/quant_lora_probe.json` (note: each probe run overwrote the file, so
the runs were consolidated from the per-run logs into one JSON before any
metric was copied elsewhere):

- `nf4_ckpt_b2_s512`: **78.27 s/step, peak 8.05 GB** — the +0.07 GB over the
  7.98 GB card is a catastrophic WDDM spill (16x slowdown), and
  `merge_and_unload` on the NF4 model works (`merge=ok`).
- `nf4_nockpt_b1_s512`: 15.99 s/step, 11.29 GB — gradient checkpointing is
  mandatory.
- `nf4_ckpt_b1_s512` (fp32 adapters): 4.74 s/step, 5.90 GB — fits.
- `nf4_ckpt_b1_s512_autocast_bf16` / `_fp16`: 35.10 s / 70.83 s per step at
  6.65 GB — **Trainer-level AMP is pathological on this ROCm/Windows stack**
  (7-15x slower than no autocast at the same shape); fp16 AMP with GradScaler
  is the worst.
- `nf4_fp16_lora_fp16_scaler/_noscaler_b1_s512`: **3.70-3.71 s/step, 5.90 GB**
  — pure fp16 (NF4 fp16 compute + fp16 adapters, no autocast) is the fastest
  viable recipe.

Decision: S2 runs batch 1 × grad_accum 32 (effective batch 32, LR schedule
and optimizer-step count unchanged), `load_in_4bit: true`, `bf16: false`, no
autocast anywhere. Extrapolation: 10,000 micro-steps x ~3.7 s ≈ 10.3 h (fp32
adapters fallback ≈ 13.2 h). `configs/drift.yaml` updated accordingly; prereg
deviation rows added for (a) the NF4 frozen base + clean bf16 merge and (b)
pure-fp16 training precision. `_merge_adapter_into_clean_base` (fresh bf16
CPU load; keeps `quantization_config` out of the saved config) is wired as the
mandatory merge path for the NF4 run.

## 2026-11-13 - S2/S4 precheck before the long GPU blocks

**Goal:** before committing the GPU to the ~10 h S2 run, verify on real data
(a) that the planned pure-fp16 S2 recipe produces finite loss/grads, and (b)
that the S4 discovery engine actually fits the 8 GB card at the preregistered
seq 256 / batch 1 (S4 was budgeted on healthy-bf16 T4 numbers; TransformerLens
does not support quantized models, so NF4 is not available there).

**Run 1 (`scripts/s2_s4_precheck.py`, results in
`results/s2_s4_precheck.json`):**

- fp16_loss probe: losses [3.3929, 2.819, 2.501] over three real MedMCQA SFT
  steps — finite and decreasing, peak 6.01 GB. Script findings while reading:
  the adapter-fp16 cast was placed before `get_peft_model`, but
  `prepare_model_for_kbit_training` freezes every parameter first, so that
  cast was a no-op and the adapters were actually fp32; and the grad-abs-sum
  check ran after `zero_grad(set_to_none=True)`, so it measured nothing. Loss
  decrease proves gradients flowed; run 2 fixes both instrumentation points
  and validates the real fp16-adapter recipe end to end.
- S4 engine probe: crashed at the call site —
  `run_circuitkit_discovery() got an unexpected keyword argument 'metric'`.
  Script bug, not a feasibility verdict; fixed and relaunched as run 2.

**Code changes from run 1:** `load_aligned_model` now casts trainable params
to fp16 AFTER the LoRA wrap (only when `load_in_4bit` + cuda + `bf16: false`),
and the probe mirrors the same recipe.

**Run 2 (`scripts/s2_s4_precheck.py`, results in
`results/s2_s4_precheck.json`):**

- **fp16 adapter probe (fixed instrumentation):** losses [3.3929, NaN, NaN],
  grad\_abs\_sum\_per\_step [6579.0, NaN, NaN] — **fp16 adapters NaN at
  optimizer step 2** (gradient overflow: step-1 grad\_abs\_sum 6579 far
  exceeds the fp16 representable range for stable accumulation). Adapter
  dtype confirmed `torch.float16`. Peak 5.84 GB (fits). Step times
  [5.86, 3.98, 3.99] s — first step includes CUDA kernel compilation.

  Decision: keep adapters in **fp32** (peft default; no cast). The
  production `load_aligned_model` and `apply_lora` do NOT cast trainable
  params to fp16. Pure fp32 adapters on the NF4 fp16-dequant base measured
  4.74 s/micro-step and finite losses/grads
  (`results/quant_lora_probe.json`). Extrapolation: 10,000 micro-steps ×
  4.74 s ≈ 13.2 h (vs 10.3 h for fp16 which is not viable).

- **S4 engine probe:** OOM at **13.58 GB** (seq 256, batch 1, n\_examples 2,
  ig\_steps 2, bf16). PyTorch allocated 13.58 GiB on the 7.98 GiB card (WDDM
  unified-memory spill). TransformerLens does not support quantized models,
  so NF4 is not available for discovery. The model weights alone in bf16
  are ~7 GB; bf16 matmul on gfx1032 is emulated as fp32, doubling
  activation/autograd intermediate memory.

  S4 mitigation decided before S2 launch (per prereg §10):
  1. Switch discovery dtype bfloat16 → **float16** (native on gfx1032;
     avoids fp32 upcast of activation intermediates, halving activation
     memory).
  2. Reduce max\_seq\_len 256 → **96** (safety pairs max 81 tokens, domain
     pairs max 206 but median 48, p90 75; only 13/256 domain pairs
     truncated).
  3. Change `_apply_memory_controls` n\_ctx floor 128 → 48 (so n\_ctx = 96).

  Estimated peak: 7 + (6.58 / 2) × (0.5 × (96/256)² + 0.5 × (96/256))
  ≈ 7.86 GB (fits under 7.98 GB). Re-probe while S2 trains to confirm.

  Configs updated: `configs/discover.yaml` (dtype: float16, max\_seq\_len: 96
  for both safety and domain).
  Prereg deviation added for S4 dtype + seq\_len reduction.


## 2026-11-13 - S2 drift training launched

**Goal:** run the LoRA SFT drift on the aligned model (scenario A, seed 0).

**Three ROCm/Windows compatibility fixes applied before successful launch:**

1. **TRL 1.14.2 `SFTConfig` API change:** `warmup_ratio` (float) replaced by
   `warmup_steps` (int). `build_training_args` now accepts
   `num_train_samples` and converts `warmup_ratio=0.03` to
   `warmup_steps=max(1, int(0.03 * total_steps))` = 9 steps for 314 total
   optimizer steps. Test monkeypatches updated to accept `**kw`.

2. **TRL `SFTTrainer` import fails:** `torch._C._distributed_c10d` is not in
   the ROCm Windows build; TRL's SFTTrainer imports `torch.distributed.tensor`
   transitively. Replaced `SFTTrainer` with `transformers.Trainer` (the
   dataset is pre-tokenized with `skip_prepare_dataset=True` and a custom
   `PaddingCollator`, so SFTTrainer adds nothing over the base Trainer;
   `SFTConfig` is a `TrainingArguments` subclass so it works unchanged).

3. **`transformers` + `accelerate` DTensor checks:** both
   `trainer_optimizer.has_mixed_dtensor` and
   `accelerate.utils.other.model_has_dtensor` import
   `torch.distributed.tensor.DTensor` unconditionally during
   `create_optimizer` and `accelerator.prepare_model`. Stubbed
   `sys.modules["torch.distributed.tensor"]` with a fake `DTensor` class so
   `isinstance(param, DTensor)` returns `False` (model has plain tensors,
   never DTensors on this single-GPU stack).

**Launch:** `.\.venv\Scripts\cg.exe drift --config configs/drift.yaml`
(terminal 6, PID 16336). Model loaded 290 shards in ~23 s; tokenizer
aligned (eos\_token\_id updated to 6); training started at ~20:09 UTC.
314 optimizer steps (5000 rows / 32 effective batch × 2 epochs) × 32
micro-steps × ~4.74 s ≈ 13.2 h; first log at step 10 (~25 min in). SDPA
attention warning (no memory-efficient kernels on gfx1032) — expected.
Git tree clean at commit `8d1c27f` before launch.

## 2026-11-13/14 - S2 VRAM spill fix (fp32 embedding → fp16)

**Problem:** the first S2 run (launched 20:09 UTC) ran at ~2350 s/optimizer-step
(17x slower than the 136 s/step measured at step 1) — constant WDDM VRAM
spill. Root cause: ``prepare_model_for_kbit_training`` upcasts all
non-quantized parameters to fp32. The untied 262k-vocab embedding and LM
head (each 262144 × 2048 × 4 B = 2.15 GB fp32) together consume 4.3 GB,
pushing total VRAM (NF4 layers 1.75 GB + embeddings 4.3 GB + adapters
0.2 GB + AdamW 0.4 GB + activations) past the 7.98 GB card on every step.

**Fix:** added ``_cast_frozen_to_fp16(model)`` after
``prepare_model_for_kbit_training`` in ``load_aligned_model``. Casts any
frozen fp32 parameter with >1M elements (embedding, LM head) back to fp16,
saving ~2.16 GB. Small parameters (layer norms, <10K elements) stay fp32
for numerical stability. Safe because the base is frozen — no gradients
flow through these layers.

**Expected VRAM after fix:** NF4 layers ~1.75 GB + embedding fp16 ~1.07 GB
+ LM head fp16 ~1.07 GB + fp32 adapters ~0.2 GB + AdamW ~0.4 GB + activations
~1-2 GB ≈ 5.5-6.5 GB (fits under 7.98 GB with ~1.5 GB headroom).

**2026-11-14 09:20 (morning relaunch was orphaned; S2 must restart):** a
relaunch at ~09:18 with the fp16-embedding fix reached training start
(`0/314` bar + SDPA warning) but the process vanished at ~09:22 with no
traceback in `D:\cg_tmp\s2_drift.err` and no loss lines flushed — consistent
with the launching session ending and taking its background shell with it.
The GPU was free again at 09:30; S2 is relaunched after the S4 memory
decision below.

## 2026-11-14 - S4 memory forensics + fix (untied LM head duplicate)

**Goal:** explain the S4 OOM observed at 13.58 GiB allocated on the 7.98 GB
card (`results/s2_s4_precheck.json`) before the 13.2 h S2 run monopolizes the
GPU, since the S4 block gates S5-S8.

**Measurements (`scripts/s4_memory_probe.py` -> `results/s4_memory_probe.json`):**

- HookedTransformer after `ck.load_model`: **7.28 GiB allocated, 100%
  bf16** (3.887B params = 7.24 GiB; no fp32 upcast anywhere). The checkpoint
  is 3.349B params (6.24 GiB) — the extra **1.0 GiB is an untied LM head**:
  the circuitkit Cohere2 port materializes
  `unembed.W_U = lm_head.weight.T * logit_scale`, a fresh transposed copy of
  the tied embedding, even though the checkpoint stores only
  `model.embed_tokens.weight` and `logit_scale == 1.0`.
- **Retie fix:** `model.unembed.W_U.data = embed.W_E.data.T` is a zero-copy
  transposed view, mathematically exact when `logit_scale == 1.0`. Measured:
  **frees 1.0 GiB** (7.28 -> 6.28 GiB). Added
  `_retie_unembed()` to `circuitkit_runner.py` (guarded on logit_scale == 1.0
  read from the HF config; skipped otherwise), called between model load and
  `_apply_memory_controls`.
- With the retie, EAP-IG discovery at bf16 / seq 256 / batch 1 **ran without
  OOM** (first examples completed; ~36 s/example at ig_steps 2 before the
  probe was stopped — see timing note below).
- The 256-pair dataloader iterates all pairs regardless of
  `ck.discover(n_examples=...)`: `quick.discover` nests `num_examples` under
  `discovery.data_params` while `GenericTaskSpec.build_dataloader` reads
  `discovery_cfg["num_examples"]` at top level. Harmless for the real S4 run
  (prereg asks for all 256 pairs per circuit); noted so probe timings are not
  misread as 2-example runs.
- `get_scores_eap_ig` never passes `max_length` to `tokenize_batch_pair`, but
  this is not an issue for these tasks: measured pair lengths are tiny
  (safety max 83 tokens, domain max 208 — both under the seq cap).

**S4 shape (logged prereg deviation):** float16, max_seq_len 96 for both
circuits (`docs/PREREGISTRATION.md` §10). Timing extrapolation from the
bf16/seq-256 measurement (~36 s/example at ig_steps 2): fp16 matmul is ~3.5x
faster than the emulated bf16 and seq 96/256 cuts compute ~2.7x, so the
fp16/seq-96 shape should be a few seconds per example — confirmed below by
`scripts/s2_s4_precheck.py --s4-only` before S4 runs.

**Progress visibility:** added `scripts/watch_progress.py` (stdlib-only,
read-only) so every long stage can be watched from a second terminal:
`python scripts/watch_progress.py --watch 30` tails the newest
`D:\cg_tmp\*.log`/`*.err` and prints step counts + ETA from the tqdm bars;
`--plan` shows the overall 36 h budget vs projected finish
(`D:\cg_tmp\plan.json`). The 36 h budget makes S2 (13.2 h) the long pole;
S5/S6 stay CPU-side and overlap with GPU evals where possible.

**2026-11-14 09:37 (fp16/seq-96 precheck measurement + S4 plan):**
`scripts/s2_s4_precheck.py --s4-only` at the logged-deviation shape
(float16, seq 96, batch 1, ig_steps 2, with the retie active): ran WITHOUT
OOM but at **48-62 s/example** (bar: 2/256 at ~50 s/it before the probe was
stopped to free the GPU) — SLOWER than the bf16/seq-256 forensic probe
(~36 s/example). Conclusion: discovery is launch/overhead-bound on this
stack, not matmul-bound, so shrinking seq does not buy time; the levers that
matter are pass count (ig_steps) and batching (batch_size), not dtype/seq.
S4 therefore stays runnable-as-logged (batch 1 fits) but needs a post-S2
probe of batch 4-8 + ig_steps reduction before the real run; worst case
(batch 1, ig 2) is ~3.4 h/circuit and would be trimmed via a logged
n_examples reduction instead. Both levers get prereg deviation rows before
S4 runs.

**2026-11-14 09:40 (S2 launched, detached):** relaunched
`cg drift --config configs/drift.yaml` via `Start-Process` (detached, PID
4984, python child 9980) so the run survives session boundaries — the
09:18 relaunch died with its session, and the 20:09 UTC run died in the
WDDM spill before the fp16-embedding fix. First progress reading at 09:46:
**2/314 optimizer steps, 125.24 s/it, remaining 10:51:13** — i.e.
~3.91 s/micro-step (the fp16 frozen-embedding cast beat the 4.74 s fp32
projection). Projected S2 total ≈ 10.9 h + bf16 CPU merge. GPU-side plan
while S2 runs: S4 batch/ig probe AFTER S2 completes (the card is fully
occupied meanwhile); S5/S6 dry-runs and selection wiring proceed on CPU.

**2026-11-14 09:58 (S2 crash protection + final relaunch):** the 314-step
run at `save_steps: 500` would never write a checkpoint (500 > 314), leaving
~11 h of GPU work unprotected against an environmental crash. Changed
`configs/drift.yaml` to `save_steps: 100` (~3.5 h cadence,
`save_total_limit=1` already set) and added auto-resume to `run_drift`
(newest `trainer/checkpoint-*` -> `trainer.train(resume_from_checkpoint=...)`;
optimizer/scheduler/dataloader position restored by the Trainer). Test fakes
updated to accept the kwarg; `pytest`: 195 passed; ruff clean. Relaunched
detached (PID 2868, cg.exe) with the same redirects; first readings
`1/314 [02:04, 124.32s/it]`, then `7/314 [14:30, 124.90s/it]` — ETA
**~10:39 remaining, S2 completing ~20:40 today**, then the bf16 CPU merge.
Watch with `python scripts/watch_progress.py --watch 30`.

## 2026-11-14 10:15 - Parallel Colab T4 path (user decision: try Colab, keep local as fallback)

**Goal:** cut the S2 wall-clock (11 h locally) by running the matrix on a
Colab T4 in parallel; stop the local run only once Colab is proven.

- `configs/drift_t4.yaml`: preregistered micro-batch **4 x grad_accum 8**
  (effective batch 32 -> identical LR schedule and 314 optimizer steps; batch 1
  was the local 8 GB deviation). Keep `load_in_4bit: true` — the validated
  NF4 + fp32-adapter + no-autocast recipe (no code change needed). T4 fit
  check: NF4 base 1.75 GB + fp16 embeddings 2.14 GB + adapters/AdamW 0.6 GB +
  worst-case fp32 logits (batch 4 x seq 1024 x 262144 vocab) 4.29 GB ≈ 9-10 GB
  < 15 GB. Measured SFT row lengths (scripts/measure_sft_lengths.py): median
  153, p90 328, p99 761 tokens.
- `notebooks/run_phase4_colab.ipynb` updated: bitsandbytes==0.50.2 pin (NF4
  path needs it), data-snapshot import cell (HF Hub), skip-guard on the data
  stage, optional COHERE_API_KEY, drift cell prefers drift_t4.yaml, discovery
  markdown reflects the logged fp16/seq-96 shape + retie fix.
- Data snapshot uploaded to the private dataset repo
  `dubeydevansh/circuitgraft-phase4-data` (40.5 MB: SFT rows, pairs, eval sets,
  curation + human-review labels, summary.json of run_id
  data-bc99df1d4f173b72, G1 passed) so Colab reuses the verified data instead
  of re-generating domain rows (which would have failed gate G1 without human
  review).
- All 19 commits pushed to origin/main (`1de1a3a..e080214`), so the notebook's
  `git clone` gets every fix (HF Trainer swap, fp16 embedding cast, retie,
  save_steps 100 + auto-resume, T4 profile).
- Decision rule: local S2 (PID 2868) keeps running as the fallback until Colab
  S2 is past ~step 20-30 with a stable rate and no OOM. Whichever S2 completes
  first is the drift checkpoint for the entire downstream matrix; artifacts
  are never mixed across the two runs. Colab session drops are covered by
  Drive persistence + save_steps 100 auto-resume.

**2026-11-14 10:50 (Colab batch-4 OOM + fix, Kaggle parity):** the first Colab
drift attempt crashed at the very first micro-step: `Tried to allocate
4294967296 bytes` (exactly 4 GiB = batch 4 x seq 1024 x 262144 vocab x 4 B)
with 9.25 GiB already allocated — the fp32 CE upcast over the 256k vocab, on
top of the fp16 logits (2.15 GiB), the log_softmax buffer and the backward
grad. The batch-4 fit check had counted the fp32 buffer but not that stack.
Fix: `configs/drift_t4.yaml` -> **micro-batch 2 x grad_accum 16** (effective
batch 32, identical LR schedule, same 314 optimizer steps; worst-case CE
buffer 2.15 GiB -> ~10.5 GiB peak) plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` exported in the drift
cells (the T4 log showed 1.65 GiB reserved-but-unallocated, i.e.
fragmentation). PaddingCollator pads to batch max and rows truncate at
max_seq_len 1024, so worst-case batches exist — batch 2 is the robust shape.
Same fix exported to the Kaggle notebook, which was also cleaned up
scientifically: its data cell used `--auto-review` and its G1 cell auto-filled
human-review labels (fabricated verification) — both removed; Kaggle now
imports the same verified data snapshot from the HF Hub and enforces the gate
honestly. bitsandbytes pin added to the Kaggle install cell.

**2026-11-14 11:05 (Kaggle OOM = stale clone; cloud profile moved to batch 1):**
the Kaggle drift OOM'd at optimizer step 3 in `loss.backward()` with a
**4.00 GiB** request — a batch-4-sized fp32 CE gradient. Timeline
reconciliation: the crash happened at 10:30 local (05:00 UTC in the Kaggle
log), the batch-2 fix was pushed at ~10:55 — the Kaggle session had cloned
origin/main at `dc3a270`, which still shipped `drift_t4.yaml` with batch 4
(the notebook cells were new because they were re-opened from GitHub, but the
repo in `/kaggle/working` was stale). Lesson: cloud cells must never trust a
previous clone. Changes: both notebooks now force a **fresh clone** on every
run (Colab `shutil.rmtree` + clone; Kaggle `rm -rf` + `clone --depth 1`), the
drift cells hard-set `configs/drift_t4.yaml` (no silent fallback) and print
the resolved effective batch + `nvidia-smi` memory before training, and the
cloud profile itself moved to **micro-batch 1 x grad_accum 32** — the exact
shape validated on the 8 GB RX 6600 (zero OOM through the whole local run),
~7-8 GiB peak on a 14.56 GiB T4. Effective batch stays 32 (identical LR
schedule, 314 optimizer steps); on a T4 (~65 TFLOPS fp16) this is still
~4-8x faster than the local card, ETA ~2-3 h.
