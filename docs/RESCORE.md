# MCQ re-score runbook (2026-11-16)

The first Kaggle eval merge (`artifacts/eval/eval-kaggle-merged/`) scored
**every MCQ item wrong**: eval rows store `choices` as option texts and
`answer` as a letter, and `expected_answer` returned the raw letter while all
predictions (`logit_answer`, `choice_word`) are option texts. Result: MedMCQA /
MedQA / MMLU accuracy exactly 0.0 (parse_rate 1.0) for all eight models, DGR
undefined everywhere. The bug is fixed in `src/circuitgraft/eval/domain.py`
(letter golds resolve through the authoritative `answer_idx`; explicit
`Answer: <letter>` parses resolve to the choice text). Regression tests:
`tests/test_eval.py` (16 passed).

The safety and benign arms are **string-match on text** and never touched the
broken scorer, so they are carried verbatim. The original summary is kept
unchanged as evidence; the rescore writes a **new** run directory that
`cg report` picks up as the newest `artifacts/eval/eval-*`.

## 1. Pull the cached generations (no GPU needed)

Generations live in the private cloud dataset at the repo-relative paths the
notebook uploaded (`results/generations/<model>/<benchmark>.jsonl`). Pull only
what the rescore needs (the full dataset also holds multi-GB checkpoints):

```bash
python scripts/import_cloud_artifacts.py --include "results/generations/**"
```

Check they arrived:

```bash
python -c "import glob; print(len(glob.glob('results/generations/*/*.jsonl')))"
# expect: 8 models x 11 benchmarks = 88 JSONL files (confirmed 2026-11-16,
# ~15.5 MB total, downloaded in ~12 s)
```

If the upload does not contain them, re-run the S7 generation cell of
`notebooks/run_phase4_kaggle.ipynb` on Kaggle — generation is signature-cached,
so completed benchmarks skip instantly and only missing files regenerate.

## 2. Re-score (CPU, under a minute)

```bash
python scripts/rescore_eval.py \
    --summary artifacts/eval/eval-kaggle-merged/summary.json \
    --generations-dir results/generations
```

Output: `artifacts/eval/eval-rescored-<stamp>/summary.json` (new manifest,
`finished_at` = now, so S9 resolves it). The script prints per-model
`domain_acc` / `general_acc` and re-runs McNemar + Holm over the domain arm;
HRR/FRR drop carried from the source run.

## 3. Regenerate the report bundle and README blocks

```bash
cg report        # resolves eval-rescored-* + discover-*; writes results/s9/report-<id>/
git status       # README.md result markers updated in place
```

The `cg report` README injection is the only sanctioned way to fill the
`<!-- RESULTS:* -->` blocks — never hand-edit them.

## 4. Sanity checks before presenting

- `results/s9/report-<id>/summary.md` status and reasons.
- Domain accuracies are no longer 0.0/1.0-degenerate; per-category numbers vary.
- `gate_g_drift.domain_gain` reflects the corrected MedMCQA/MedQA numbers; the
  refusal-drop condition is unchanged and still fails.
- `artifacts/eval/eval-kaggle-merged/summary.json` is untouched
  (byte-compare against `cg_outputs/` if unsure).

## 5. Publish

```bash
cg publish --no-models    # dataset/article artifacts through the allow-list
```

Then point the Space at the new `results/s9/report-*/` bundle (the Space reads
the newest tracked bundle automatically) and fill `docs/ARTICLE.md` blocks from
the corrected tables.
