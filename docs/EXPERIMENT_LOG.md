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

## Open items

1. Human lead accepts both Tiny Aya licences.
2. Re-run `cg doctor --check-models`; confirm the real architecture against
   `Cohere2Layout` and verify the tokenizer/chat-template length.
3. Run `scripts/d1_smoke.py --device cuda` on Colab/Kaggle and archive
   `results/d1/smoke/` as the G0 evidence.
4. Initialize git, commit the D1 tree, then freeze
   `docs/PREREGISTRATION.md` before any Phase 4 experiment.
5. Implement the missing stage modules (data, drift, discovery, baselines,
   eval, attacks, report, publish) and their tests.
