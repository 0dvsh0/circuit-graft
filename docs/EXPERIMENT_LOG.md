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

## Open items

1. Human lead accepts the `CohereLabs/tiny-aya-global` licence (base is already
   accessible) and the five gated datasets listed in `docs/API_NOTES.md`.
2. Re-run `cg doctor --check-models`; confirm the real architecture against
   `Cohere2Layout` and verify the tokenizer/chat-template length on the
   aligned checkpoint.
3. Run `scripts/d1_smoke.py --device cuda` on Colab/Kaggle and archive
   `results/d1/smoke/` as the G0 evidence.
4. Keep `docs/PREREGISTRATION.md` DRAFT until G0 is resolved; review, freeze
   and commit it before any Phase 4 experiment.
5. Implement the remaining stage modules (drift, baselines, eval, attacks,
   report, publish) and their tests. S1 is implemented and smoke-tested; S4
   discovery is implemented but not yet run on Tiny Aya.
