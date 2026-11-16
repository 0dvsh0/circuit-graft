# Pitch & presentation playbook (2026-11-16, post-rescore)

Everything below is true and checkable against `results/s9/report-1794803399/`
(and the discover/eval artifacts it points at). The strategy: **own the rigor**
— a preregistered study with gates that are allowed to fail, run on real gated
frontier models — and lead with the numbers that are dramatic AND true. Never
quote SRR as evidence (2.3-point denominator this run); never claim the graft
"recovered safety" at the default operating point. The strongest true results
are the overlap measurement, the ctheta double-collapse, and the
protected-edit ablation win.

## The four title-slide numbers

1. **72 of 184 shared components** (Jaccard 0.243: 45 heads + 27 MLPs; 39.1% of
   each circuit; Jaccard 0.17–0.24 at every discovery sparsity k ∈ {0.05…0.3})
   — safety and medicine are *intertwined* in the same components, and the
   overlap spans 32 of 36 layers.
2. **76.7% → 6.0%** HarmBench refusal — the naive weight-arithmetic baseline
   (C-ΔΘ) collapses safety at the shared interface **and** loses 16 points of
   MedMCQA / 25 points of MMLU. Blunt tools destroy both behaviors; we measured
   the destruction precisely.
3. **The protected-edit ablation win:** `cg_project` beats the unprotected
   `cg_uniform` on *every* axis — HarmBench 69.3% vs 66.7%, AdvBench 87.5% vs
   85.0%, DGR −57.1% vs −142.9%, XSTest false refusal 7.5% vs 13.8%. The
   overlap-aware protection does exactly what it was designed to do.
4. **8 models × 11 benchmarks, preregistered** — full matrix on gated Tiny Aya
   checkpoints on a T4 in one idempotent, resumable notebook; McNemar + Holm,
   bootstrap CIs, signed deviation table.

## The story arc (3 minutes)

> "Domain fine-tuning a medical model silently interacts with its safety
> circuit — but you can't fix that without a map of where safety and medicine
> *intersect*. We built that map on the real gated Cohere Tiny Aya checkpoints:
> 24% of the safety circuit is inside the domain circuit. That single
> measurement explains why blunt unlearning fails: at the shared interface,
> C-ΔΘ drops harmful-prompt refusal from 76.7% to 6.0% — and takes 16 points of
> medical accuracy with it. CircuitGraft is the precision instrument: graft the
> safety delta back at full strength on safety-only components and through a
> domain-protecting projection on the shared ones. Our ablation says the
> protection matters: with it, the graft is better than the unprotected control
> on refusal, domain retention AND false refusal — simultaneously."

Then the honest status line (say it proudly, don't bury it):

> "G0 and G1 passed; the core matrix is complete. Two things failed and we
> report both: our drift scenario did not make the model meaningfully less
> willing to refuse (refusal changed 2.3 points instead of ≥15 — though the
> domain-gain axis passed at +8.0 points), and S8 attacks didn't fit in the
> window. We also caught a scoring bug in our own first eval during review,
> fixed it, and re-scored the whole MCQ arm from cached generations on CPU —
> the original run is preserved as evidence."

## Demo flow (5 minutes)

1. **Space** (`space/`) — reads the tracked S9 bundle `report-1794803399`:
   status, tables, plots. Show the overlap heatmap: two circuits, 72 shared
   components spread across 32 layers.
2. **Main table** — walk the ctheta row (6.0% HarmBench, 24.7% MedMCQA:
   double collapse) against cg_project (69.3%, 40.0%, FRR 7.5%) and cg_uniform
   (66.7%, 36.7%, FRR 13.8%): the protection ablation wins everywhere.
3. **Repo tour** — `docs/PREREGISTRATION.md` (frozen gates + signed deviation
   table) → `artifacts/eval/eval-rescored-*/summary.json` (hash-chained inputs,
   rescore provenance block) → `docs/EXPERIMENT_LOG.md` (power-cut recovery +
   scoring-bug story).
4. **Live proof** — `pytest tests/test_eval.py tests/test_report.py -q` or
   `cg doctor`; show the letter-gold regression test and the fixed `_fmt`
   scaling test.

## Slide outline

| # | Slide | Content |
|---|---|---|
| 1 | Problem | Drifted medical model = silent safety regression; need the overlap map |
| 2 | Discovery | EAP-IG circuits on real gated Tiny Aya: 184/184/72, Jaccard 0.243, heatmap |
| 3 | Why blunt fails | C-ΔΘ 76.7% → 6.0% refusal AND −16 MedMCQA; RESTA collapses refusal too (52.7%); the interface is violent |
| 4 | Method | Graft: full α on S\D, protected edit on S∩D; 4 overlap strategies; never touch LN/embed/lm_head |
| 5 | Rigor | Preregistered gates, McNemar+Holm, bootstrap CIs, deviation table, manifest hashes |
| 6 | Phase 4 | 8 × 11 matrix on T4; single notebook, idempotent, resume-after-crash |
| 7 | Results | Overlap + ctheta double-collapse + protected-edit ablation win (all true) |
| 8 | Honest gates | G-drift refusal axis failed → reported; domain axis passed (+8.0); scoring bug → fixed, re-scored from cache |
| 9 | What ships | Code + data + prereg + article + Space + published datasets |

## Judge questions — answer scripts

**"Your grafted model is less safe than the drifted baseline (negative SRR)."**
> "At the single preregistered default operating point, the graft sits between
> drifted (81.3% HarmBench) and RESTA (52.7%): cg_project refuses 69.3% of
> harmful prompts while keeping 40.0% MedMCQA. Our selection sweep — which
> picks each method's operating point — was a disclosed deadline cut, so this
> is a fixed-hyperparameter comparison, not the method's best case. And the SRR
> column is not evidence this run: the refusal axis of the drift gate failed,
> so its denominator is 2.3 points and every SRR magnitude is noise."

**"Why is your drifted model not unsafe in the first place?"**
> "The G-drift gate failed as a conjunction: refusal changed 2.3 points instead
> of ≥15 (CI [−1.5, 6.2]) — the 5000-row QLoRA drift was too shallow inside our
> session budget. The gate's other axis, domain gain ≥3, passed at +8.0 points
> (CI [+2.9, +14.0]) once we fixed the MCQ scorer. Per the preregistered remedy
> order we'd need three epochs or Scenario B; not affordable before the
> deadline, so we report the failed gate. The gates did their job: they refused
> to let us tell a story the data doesn't support."

**"Your first eval had 0.0% accuracy everywhere?"**
> "Yes — caught it in review, not on stage. Our MCQ gold was a letter while
> every prediction was the option text; accuracy 0.0 with parse_rate 1.0 across
> all 8 models was the statistical smoking gun. One-line fix in
> `expected_answer`, three regression tests, and because every generation was
> cached with fingerprints we re-scored the whole matrix on CPU without
> re-running the model. Original summary preserved; the corrected run
> (`eval-rescored-20261116-042341`) is what the report and article use."

**"Random beats your circuit selection?"**
> "No — random is *within noise* of the projection (HarmBench 70.0% vs 69.3%,
> DGR −60.7% vs −57.1%, false refusal tied at 7.5%, single seed), so we claim
> 'no detected advantage' for selection, not 'no advantage'. The clean,
> supported comparison is with the *unprotected* control: cg_uniform loses to
> cg_project on every axis. The overlap-aware protection is what works."

**"String-match judge?"**
> "WildGuard-7B doesn't fit beside a 3.5B model on a 14.56 GiB T4 — disclosed
> deviation; the frozen eval config declares the fallback. A blind human audit
> of generations is the preregistered reliability check and is pending; the
> bundle says exactly that."

**"Your circuits failed faithfulness?"**
> "The faithfulness checks are incomplete this run — stability repeats were
> degenerate (identical selections), safety patching/ablation weren't computed
> in budget, and the SafeTune cross-check was a disclosed cut. Per our own
> prereg rule, circuit claims are therefore reported as measurement, not
> validated causal fact — and the *behavioral* ablation (protected vs
> unprotected edit) is what we lean on for the causal story."

## Do-not-claim list (protects you under questioning)

- Never say the graft "recovered safety" or quote positive SRR as a win; never
  quote any SRR magnitude as evidence (2.3-point denominator).
- Never show the 0.0% MCQ columns without the bug-fix/rescore context.
- Don't claim selection > random; claim protection > no-protection (true on
  every axis).
- Never imply attacks were run: the robustness table is empty and labeled
  "not run", which is the preregistered way to report it.
- Don't present the circuits as causally validated: faithfulness is incomplete
  and disclosed (prereg §6 → inconclusive).

## Assets inventory (all tracked)

- Overlap: `artifacts/discover/discover-6b120d61758287f6/overlap.json` +
  heatmap PNG (also copied into `results/s9/report-1794803399/`).
- Eval evidence: `artifacts/eval/eval-kaggle-merged/` (original, preserved) and
  `artifacts/eval/eval-rescored-20261116-042341/` (corrected MCQ arm).
- Report bundle: `results/s9/report-1794803399/` (README blocks injected from
  it by `cg report`).
- Recovery story: `docs/EXPERIMENT_LOG.md` 2026-11-15 power-cut entry + the
  2026-11-16 scoring-bug entry.
- Article: `docs/ARTICLE.md` — every block filled from the bundle.
