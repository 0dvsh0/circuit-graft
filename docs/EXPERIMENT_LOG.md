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
