# CircuitGraft: circuit-aware safety recovery for domain-fine-tuned Cohere Tiny Aya

**Track 2b / Problem 6 — Lexsi Labs Hackathon**

> **Status: scaffold, not submission-ready.** This article intentionally contains
> no experimental numbers. The Phase 4 matrix has not run yet, and the
> preregistration is still a draft. Every metric-bearing block is marked with an
> `ARTICLE:` placeholder.
>
> **Single source of truth.** Fill the placeholders only from the tracked S9
> bundle for the run being described:
>
> ```text
> results/s9/<run_id>/
>   summary.json          # run status, sources, partial_reasons
>   summary.md            # human-readable report
>   table_main.md         # main recovery table
>   table_overlap.md      # S4 overlap table
>   table_attacks.md      # S8 robustness table
>   pareto_frontier.png
>   circuit_overlap_heatmap.png
>   attack_decay.png      # only when post-attack metrics exist
> ```
>
> Do **not** hand-type a number, copy one from a terminal, or quote a proxy/smoke
> run as Tiny Aya evidence. If a value is missing, keep `—` and say why. The S9
> `summary.json` records whether the bundle is `complete` or `partial`; a partial
> bundle must not be presented as final evidence.

## Abstract

Domain fine-tuning makes a model more useful in a target domain, but it can also
erode the refusal behavior learned during alignment. Whole-model recovery methods
restore safety at the cost of domain competence, while circuit-restricted methods
often optimize safety without asking which components the domain skill needs.
CircuitGraft maps a safety circuit and a domain circuit in the same attention-head
and MLP component space, partitions components into safety-only, domain-only and
shared sets, and grafts the safety delta back with domain-protected edits on the
shared set. We evaluate safety recovery, domain-gain retention and false refusal
against RESTA, WiSE-FT, C-ΔΘ (SafeTune weight-arithmetic implementation) and
Safe LoRA on Cohere Tiny Aya, using a medical fine-tuning scenario and
preregistered endpoints.

**Results:** `<!-- ARTICLE:ABSTRACT_RESULTS -->` (one paragraph, generated from
`summary.json` and `table_main.md` after the frozen Phase 4 run).

## 1. Motivation

Safety alignment is not a property that domain fine-tuning automatically
preserves. A medical specialist may gain benchmark accuracy while becoming more
willing to answer harmful requests, because the fine-tuning signal rewards
helpful completion and does not distinguish legitimate clinical questions from
misuse. Restoring safety after the fact is attractive: it avoids retraining the
domain model from scratch and keeps the domain data pipeline unchanged.

The design question is *where* to restore safety. Whole-model interpolation and
task-arithmetic baselines move every parameter, so they can undo the very
representations that make the specialist useful. Circuit-localized updates are
more surgical, but existing work typically localizes one behavior at a time. A
safety circuit and a domain circuit can overlap; editing a shared component as if
it were safety-only is a plausible source of domain regression and false refusal.

CircuitGraft's hypothesis is that making that overlap explicit, and protecting the
domain direction on shared components, improves the safety/domain trade-off over
both whole-model recovery and naive circuit restriction. The preregistered
hypotheses and decision rules are in [`docs/PREREGISTRATION.md`](PREREGISTRATION.md).

## 2. Setup and why Tiny Aya / medicine / these benchmarks

**Model triple.** `θ_B` is `CohereLabs/tiny-aya-base` (pretrained), `θ_A` is
`CohereLabs/tiny-aya-global` (aligned), and `θ_D` is the medical specialist
produced by S2. Tiny Aya is small enough to fine-tune and evaluate on accessible
GPUs, while retaining the Cohere2 architecture that CircuitKIT supports (36
layers, grouped-query attention, parallel attention/MLP blocks). Both checkpoints
are gated; access and environment checks are recorded in
[`results/d1/environment.json`](../results/d1/environment.json) when available.

**Domain.** Medicine is a deliberately adversarial choice for this question.
Clinical knowledge (dosing, toxicology, drug interactions) is close to
harmful-request territory (poisoning, overdose, synthesis), so a safety/domain
overlap is expected and false refusals of legitimate clinical questions are a
real risk.

**Data.** S1 builds MedMCQA supervised fine-tuning data, held-out MedQA and
PubMedQA transfer sets, an MMLU control, and safety evaluation prompts. The
safety benchmarks are HarmBench and AdvBench; benign refusal is measured with
XSTest. Pair construction, decontamination and the human-review requirement are
documented in the experiment log and the preregistration.

**Drift scenario.** Scenario A is benign medical SFT. Scenario B (a small
fraction of harmful rows) is a disclosed fallback only if H0 is not supported; it
must be labelled as such and must not be substituted after seeing results.

**Primary endpoints.** Safety recovery rate
`SRR = (HRR_repaired − HRR_D) / (HRR_A − HRR_D)` and domain gain retention
`DGR = (Acc_repaired − Acc_A) / (Acc_D − Acc_A)`, with false-refusal rate and edit
footprint as secondary axes. Confidence intervals use paired bootstrap;
comparisons use McNemar tests with Holm correction. Exact definitions and sweep
grids are frozen in the preregistration.

## 3. Demonstrating drift

S3 audits `θ_A` and `θ_D` before any recovery method is applied. H0 is supported
only if HarmBench refusal drops by at least 15 points versus the aligned model and
the bootstrap 95% CI excludes zero.

<!-- ARTICLE:DRIFT -->
_Insert the aligned-vs-drifted rows from `results/s9/<run_id>/table_main.md`:
HRR with CI, domain accuracy, general accuracy, false-refusal rate, and the
paired test. Link the S3 summary only if it is tracked; `artifacts/` is
git-ignored. State the run ID and whether the report status is complete._
<!-- /ARTICLE:DRIFT -->

If H0 fails, report that plainly. Do not proceed to a recovery claim as if drift
had been demonstrated.

## 4. Where safety and medical skill live

S4 discovers the safety circuit `S` on `θ_A` with CircuitKIT EAP-IG and
cross-checks it with SafeTune EAP-IG; the domain circuit `D` is discovered on
`θ_D` with MCQ-letter contrastive pairs. Both use the same component vocabulary:
attention heads `L{l}.H{h}` and MLPs `L{l}.MLP`. Layer norms, embeddings and
`lm_head` are never edited. The CircuitKIT output is also the shared circuit
consumed by the C-ΔΘ baseline, so the comparison uses the same discovery.

The overlap report partitions components into `S \ D`, `D \ S` and `S ∩ D`, and
reports Jaccard overlap plus per-layer counts. Circuit faithfulness (causal
patching/ablation against a random baseline) is part of the acceptance check; a
circuit that fails it must not be used for claims.

<!-- ARTICLE:OVERLAP -->
_Insert the overlap summary from `results/s9/<run_id>/table_overlap.md` and
`circuit_overlap_heatmap.png`: circuit sizes, overlap count/Jaccard, per-layer
concentration, CircuitKIT-vs-SafeTune agreement, and faithfulness results. State
whether the safety and domain circuits are concentrated or distributed; this
motivates the overlap strategy used in the main run._
<!-- /ARTICLE:OVERLAP -->

## 5. Method

Let `θ_B`, `θ_A`, `θ_D` be the base, aligned and drifted weights. The safety
delta is `Δ_S = θ_A − θ_B`; the domain delta is `Δ_D = θ_D − θ_A`. For a selected
component set `S`, domain set `D` and overlap `O = S ∩ D`:

- safety-only component `c ∈ S \ D`: `W_D[c] += α · Δ_S[c]`;
- shared component `c ∈ O`: `W_D[c] += α_O · protect(Δ_S[c], Δ_D[c], F_D[c])`;
- every other component: unchanged.

`protect` is one of four preregistered strategies:

- `exclude`: skip shared components (`protect = 0`);
- `svd_project`: remove the top right-singular directions of `Δ_D[c]` from the
  safety edit, keeping 90% of domain-delta energy up to rank 16;
- `fisher_gate`: down-weight each weight by its normalized domain Fisher
  importance, using gradients on held-in domain examples;
- `uniform`: return the safety delta unchanged; this is the C-ΔΘ-like control in
  our component space.

Selection is `topk` or per-layer-budgeted `distributed`. The preregistered model
selection rule maximizes SRR subject to `DGR ≥ 0.7`, otherwise maximizes
`SRR + DGR`; ties break by smaller footprint and then lower ORP. The primary
method is `cg_project`; `cg_uniform`, `cg_exclude`, `cg_fisher`, `cg_distributed`
and `cg_random` are controls/ablations. All edits are applied in fp32 one tensor
at a time and cast back to the checkpoint dtype.

## 6. Results

<!-- ARTICLE:MAIN_RESULTS -->
_Insert `results/s9/<run_id>/table_main.md` and `pareto_frontier.png` unchanged.
The table must include every method actually run, with SRR/DGR, confidence
intervals, HRR, false-refusal rate, domain/general accuracy and edit footprint.
State which methods Pareto-dominate which, and whether H1/H2 are supported under
the preregistered rule. If a method is missing, write `—` and explain why; do not
drop the row silently._
<!-- /ARTICLE:MAIN_RESULTS -->

## 7. Ablations and random-mask control

The ablation set separates the effect of circuit selection from the effect of
sparsity. `cg_random` uses a random mask with the same parameter count and layer
distribution as `cg_project`; the overlap-strategy variants isolate the
domain-protected edit; `cg_distributed` tests the exploratory per-layer-budget
hypothesis.

<!-- ARTICLE:ABLATIONS -->
_Insert the ablation rows from the tracked S9 tables (or a clearly linked
supplementary table): `cg_uniform` vs `cg_project` vs `cg_fisher`,
`cg_random`, `cg_distributed`, and the RESTA/WiSE-FT/C-ΔΘ/Safe LoRA baselines at
their best swept settings. Report H2-ctrl and any null result with the same
prominence as a positive one._
<!-- /ARTICLE:ABLATIONS -->

## 8. Robustness and quantization

S8 runs abliteration, harmful fine-tuning and quantization attacks against the
selected checkpoints. S8 saves attacked checkpoints rather than measuring HRR
itself; `cg eval-attacks` then re-evaluates every saved checkpoint through the
normal S7 pipeline and writes a `post_eval.json` beside each attack run.
Reported robustness values come from those files, never from the attack stage's
own bookkeeping. Attacks that produce no checkpoint (runtime abliteration,
quantization plans) remain unmeasured and S9 shows `—` for them.

<!-- ARTICLE:ROBUSTNESS -->
_Insert `results/s9/<run_id>/table_attacks.md` and `attack_decay.png` only after
`cg eval-attacks` has evaluated the saved checkpoints. Report H3
(abliteration/harmful-FT retention) and H4 (4-bit quantization) using the
preregistered thresholds. If the post-attack evaluation was not run, say so
explicitly and do not present the attack table as evidence. Quantization HRR
stays `—` until a real quantized checkpoint is saved and evaluated._
<!-- /ARTICLE:ROBUSTNESS -->

## 9. Limitations and negative results

- **Single model family and domain.** Results are for Cohere Tiny Aya and
  medicine; they do not establish transfer to larger models or other domains.
- **Judge and benchmark limits.** HarmBench/AdvBench refusal and XSTest benign
  refusal are proxy measurements, not a complete safety evaluation. Judge
  disagreement and prompt selection must be reported.
- **Component granularity.** Attention heads and MLPs are coarse units. The
  GQA-aware K/V approximation (a KV slice is edited when any head in its group is
  selected) is documented and may blur head-level claims.
- **Discovery stability.** CircuitKIT and SafeTune EAP-IG can disagree. The
  overlap/Jaccard check is a stability diagnostic, not proof that the circuit is
  unique or complete.
- **S8 coverage.** Post-attack metrics require the `cg eval-attacks` pass; until
  that runs the robustness claims are incomplete. Quantization attacks currently
  save plans rather than quantized checkpoints, so their columns stay empty.
- **Compute and seeds.** Report the number of seeds, sweep budget and any
  unreplicated cells. A single-seed result is not a variance estimate.
- **Negative results.** If H1/H2/H3 are null, say so in the abstract and results,
  not only in this section.

## 10. Reproducibility and artifact links

The full pipeline is orchestrated by `scripts/run_all.sh`:

```bash
bash scripts/run_all.sh --check   # environment, configs, tests, lint
bash scripts/run_all.sh --smoke   # CPU mechanics only; never evidence
bash scripts/run_all.sh --full    # Phase 4 on a CUDA machine
```

The full run executes `data → drift → audit → discover → baselines → graft →
eval → attacks → eval-attacks → report → publish`, writing manifests under
`artifacts/` and the tracked report bundle under `results/s9/<run_id>/`. The
report bundle is the only source for this article's numbers.

<!-- ARTICLE:ARTIFACTS -->
_Link the actual public artifacts after S10: GitHub commit, HF model
collections (aligned/drifted/repaired/baselines), HF dataset collections
(safety/domain/eval), the Space, and the S9 run directory. Record the frozen
preregistration commit, package lock/versions, GPU type, and the exact commit SHA
shown in `summary.json`/the stage manifests. Replace every placeholder below;
do not leave a dead link._
<!-- /ARTICLE:ARTIFACTS -->

| Deliverable | Link / value |
|---|---|
| Code commit | `<!-- ARTICLE:CODE_COMMIT -->` |
| Frozen preregistration commit | `<!-- ARTICLE:PREREG_COMMIT -->` |
| S9 report run | `results/s9/<!-- ARTICLE:RUN_ID -->/summary.json` |
| Model collection | `<!-- ARTICLE:MODEL_COLLECTION -->` |
| Dataset collection | `<!-- ARTICLE:DATASET_COLLECTION -->` |
| Gradio Space | `<!-- ARTICLE:SPACE_URL -->` |
| Hardware / stack | `<!-- ARTICLE:HARDWARE -->` |

## 11. Related work and attribution

CircuitGraft builds on four Lexsi toolkits and the recovery/interpretability
literature:

- **RESTA** — safety recovery by adding a scaled alignment delta to the
  fine-tuned weights, with optional DARE sparsification.
- **C-ΔΘ** — circuit-restricted weight arithmetic. This article must use the
  label **“C-ΔΘ (SafeTune weight-arithmetic implementation)”** because SafeTune's
  `apply_ctheta` is layer/module-granular, not head-identical to the paper.
- **EAP / EAP-IG** — edge-attribution patching used by CircuitKIT and SafeTune for
  circuit discovery.
- **Abliteration** — refusal-direction projection, used here as an attack.
- **Lexsi toolkits** — CuratorKIT (data curation), SafeTune (recovery baselines
  and EAP), CircuitKIT (discovery), AuditKIT (evaluation and run fingerprints).

Attribution details and source links are maintained in
[`SPEC.md`](../SPEC.md); cite the original papers for RESTA, C-ΔΘ, EAP-IG and
abliteration rather than this repository. Novelty claims must be phrased as “to
our knowledge” and limited to what the experiments actually show.

## Appendix: checklist before submission

- [ ] `docs/PREREGISTRATION.md` reviewed, no longer `DRAFT`, committed before the
      Phase 4 run.
- [ ] Every `ARTICLE:` placeholder replaced from a tracked `results/s9/` bundle.
- [ ] `summary.json` status is `complete`; any `partial_reasons` are disclosed in
      the limitations or removed by re-running the missing stage.
- [ ] Main table includes every method, including null/negative results and `—`
      for missing evaluations.
- [ ] All artifact and collection links resolve publicly.
- [ ] Abstract is under 500 words and contains no number absent from the S9
      bundle.
- [ ] Reproduction command and commit SHA match the bundle manifest.
