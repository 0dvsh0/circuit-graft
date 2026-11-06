# CircuitGraft: Technical Specification v1.0

*Circuit-aware safety recovery for domain-fine-tuned Cohere Tiny Aya models*

---

## 0. How to use this document

This spec is written for a coding agent working with a human lead. Sections 1–4 cover the research design and the decisions already made. Sections 5–11 cover the engineering: repo layout, interfaces, data, pipeline stages, evaluation, and artifacts. Sections 12–15 cover the schedule, acceptance gates, risks, and rules the agent must follow.

**Important caveat about APIs.** Every library call below comes from the public READMEs and docs of the Lexsi Labs repos as of today (Nov 5, 2026). These libraries are young and change quickly. Before writing any integration, the agent must open the installed package source and check signatures. When a call in this spec doesn't match the installed code, the installed code wins. Record the difference in `docs/API_NOTES.md`.

---

## 1. Problem framing and hackathon fit

| Item | Decision |
|---|---|
| Primary problem | Track 2b, Problem 6: Repairing safety drift without retraining |
| Secondary (claim only if the experiments are run) | P9 Locating where safety lives, P10 Circuit-level interventions, P1 robustness/abliteration, P11 safety-aware compression, P12 tamper resistance |
| Repos integrated | CuratorKIT → SafeTune → CircuitKIT → AuditKIT (all four, each doing real work) |
| Model | `CohereLabs/tiny-aya-global` (aligned), `CohereLabs/tiny-aya-base` (pre-alignment reference) |
| Deadline | Nov 14, 2026, end of day AoE. Shortlist Nov 16, extension to Nov 21, Demo Day Nov 22 |

The hackathon describes Problem 6 as taking "a model whose safety degraded after fine-tuning and restore it with weight-space methods (safety patching, merging, projection) while keeping task performance." Track 2 is built around a CuratorKIT → SafeTune → CircuitKIT → AuditKIT chain. Requirements: at least 2 repos (3+ preferred), all artifacts public on HF, before/after metrics (more than one), and a write-up explaining the choices [Source](https://hackathon.lexsi.ai/).

### 1.1 Why Tiny Aya

The **compute fit** argument: Tiny Aya is a 3.35B-parameter model whose post-training includes "supervised fine-tuning (SFT) and preference training to align model behavior to human preferences for helpfulness and safety" [Source](https://huggingface.co/CohereLabs/tiny-aya-global). There is also a public pretrained-only sibling, `tiny-aya-base` [Source](https://huggingface.co/CohereLabs/tiny-aya-base). That gives us the **base / aligned / drifted triple** that delta-based recovery methods need. The organizers note that free Colab/Kaggle GPUs can be enough for Tiny Aya [Source](https://hackathon.lexsi.ai/).

The **tooling support** argument:
- CircuitKIT validates Tiny Aya for discovery, evaluation and interventions. It describes the architecture as 36 layers, d_model 2048, 16 query heads / 4 KV heads (GQA), d_head 128, d_mlp 11008, with parallel attention+MLP blocks and a 262k vocab [Source](https://circuitkit.lexsi.ai/advanced/tiny-aya/).
- SafeTune lists ReSta as supported on Tiny Aya and has Cohere-specific notes [Source](https://github.com/Lexsi-Labs/SafeTune).
- Aya Expanse 8B is a stretch-goal replication target only. Do not touch it before the Tiny Aya pipeline is complete.

**Gating:** all Tiny Aya repos are gated. Accept the licence on each model page and set `HF_TOKEN` on day 1.

### 1.2 Domain choice: medicine

We fine-tune Tiny Aya-Global into a medical QA specialist. Medicine is the right domain for CircuitGraft's core question because legitimate medical knowledge (dosages, toxicology, drug interactions) overlaps heavily with harmful-request territory (poisoning, overdose, synthesis). That overlap is exactly where shared safety/domain components are expected, and where naive safety restoration should cause the most false refusals of legitimate clinical questions.

---

## 2. Prior work and what is original

### 2.1 Existing methods (baselines, not our contribution)

| Method | What it does | Where it comes from |
|---|---|---|
| RESTA | θ_safe = θ_ft + α·(θ_aligned − θ_base), with optional DARE sparsification | Bhardwaj et al., ACL 2024; SafeTune `ReStaTrainer` [Source](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/user-guide/recover/layer/resta.md) |
| C-ΔΘ | Localizes refusal-causal computation with EAP-IG, then applies a weight update restricted to that circuit (typically <5% of parameters) | Kasliwal et al. 2026, arXiv:2602.04521 [Source](https://arxiv.org/abs/2602.04521) |
| SafeTune's C-ΔΘ | `W_target[mask] += strength·(W_pos − W_neg)[mask]`, with the mask taken from a `CircuitInfo` | [Source](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/user-guide/recover/circuit/c-delta-theta.md) |
| WiSE-FT, Task Arithmetic, LoX, Safe LoRA, Antidote, etc. | Whole-model, low-rank, layer and neuron recovery | SafeTune Recover catalog [Source](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/getting-started/taxonomy.md) |
| Abliteration | Projects out a single refusal direction | Arditi et al., NeurIPS 2024; SafeTune `AbliterationAttack` [Source](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/user-guide/evaluate/attacks/abliteration.md) |

One nuance to state accurately in the write-up: the C-ΔΘ paper describes **offline gradient updates restricted to circuit parameters** [Source](https://lexsi.ai/resources/articles/circuit-restricted-weight-arithmetic-c-dth-offline-safety-without-runtime-cost). SafeTune's `CThetaTrainer` is **weight arithmetic** restricted to the circuit. We use SafeTune's implementation as our "uniform circuit graft" baseline and must label it "C-ΔΘ (SafeTune weight-arithmetic implementation)."

### 2.2 CircuitGraft's original contributions (each one must be backed by experiments)

1. **Dual-circuit conflict map.** Discover a *safety circuit* S on the aligned model and a *domain circuit* D on the drifted specialist, in the same component space. Then partition components into S-only, D-only, and shared O = S∩D. Prior work cited here localizes a single behavior circuit. Mapping both circuits and acting on their overlap is our addition.
2. **Conflict-aware grafting.** Apply the safety delta at full strength on S-only components. On shared components, apply a *domain-protected* version: either project out the domain-delta subspace or gate per weight by domain Fisher importance. Leave everything else untouched.
3. **Distributed grafting (exploratory hypothesis).** Select safety components under a per-layer budget, so the graft spans many layers instead of concentrating on the top-attribution ones. Then test whether this resists abliteration and harmful re-fine-tuning better.
4. **Integrated, reproducible pipeline and public artifacts**, plus a compression robustness study (secondary).

Novelty must be phrased as "to our knowledge" and only after a quick literature check, which the human lead does on Day 1. Do not claim anything beyond what the experiments show.

---

## 3. Hypotheses (pre-registered)

Commit `docs/PREREGISTRATION.md` to git **before** running any Phase 4 experiments. The commit timestamp is the evidence.

| ID | Hypothesis | Primary metric | Supported if… |
|---|---|---|---|
| H0 | Domain fine-tuning causes measurable safety drift in Tiny Aya-Global | HarmBench refusal rate (judge) | Drop ≥ 15 points vs aligned, bootstrap 95% CI excludes 0 |
| H1 | Circuit-localized grafting recovers safety with less domain loss than whole/layer-level recovery | Pareto position (SRR vs DGR, defined in §9.3) | CG-project Pareto-dominates, or matches SRR with higher DGR than, RESTA and WiSE-FT at their best-swept settings |
| H2 | Overlap-aware handling beats naive circuit restriction | DGR and in-domain false-refusal rate at matched SRR (±5 pts) | CG-project or CG-fisher has higher DGR or lower domain false refusal than CG-uniform (≈C-ΔΘ) with non-overlapping CIs |
| H2-ctrl | The circuit matters, not just sparsity | Same metrics | CG-project beats a random mask with the same parameter count and layer distribution |
| H3 (exploratory) | Distributed grafts resist abliteration / harmful FT better | Refusal after attack | CG-distributed retains ≥ 10 points more refusal post-attack than CG-project and RESTA |
| H4 (secondary) | Grafted safety survives 4-bit quantization | Refusal change after quantization | |Δrefusal| ≤ 5 points at NF4 |

Report negative and null results with the same prominence as positive ones. A null H2 or H3 is still a valid finding.

---

## 4. Method specification

### 4.1 Notation

- θ_B is base (`tiny-aya-base`), θ_A is aligned (`tiny-aya-global`), θ_D is drifted (the medical specialist).
- Safety delta: Δ_S = θ_A − θ_B (the "alignment vector," as in RESTA/C-ΔΘ).
- Reversal delta: Δ_R = θ_A − θ_D (an alternative graft source; config `delta_source: alignment|reversal`).
- Domain delta: Δ_D = θ_D − θ_A.

### 4.2 Component space

Use one component vocabulary for both circuits, aligned with CircuitKIT node-level output:

- **Attention head** `L{l}.H{h}`, l∈[0,35], h∈[0,15]. The weight slices are rows `h·128:(h+1)·128` of `q_proj`, the matching GQA group (`h // 4`) rows of `k_proj`/`v_proj`, and columns `h·128:(h+1)·128` of `o_proj`. Shared K/V slices belong to a head-group. If any head in the group is selected, the KV slice is selected, with a strength multiplier equal to the fraction of the group's heads selected. Document this approximation.
- **MLP** `L{l}.MLP`, which covers `gate_proj`, `up_proj` and `down_proj` for that layer. An optional neuron-level refinement uses the top-k columns of `down_proj` / rows of `gate/up` by attribution.
- Layer norms, embeddings and lm_head are **never edited**. Embeddings are tied on Tiny Aya.

The agent must verify the actual HF module names (`model.layers.{l}.self_attn.q_proj` etc.) by printing `named_parameters()` and adjusting `src/circuitgraft/components.py`.

### 4.3 Algorithm: CircuitGraft

```
Input: θ_B, θ_A, θ_D, safety scores s_S(c), domain scores s_D(c) for components c
Params: k_S (safety budget, fraction of components), k_D, overlap_strategy,
        α (strength), α_O (overlap strength), selection: topk|distributed

1. S = select(s_S, k_S, selection)          # distributed: per-layer cap = ceil(k_S·N / 36 · ρ)
2. D = topk(s_D, k_D)
3. O = S ∩ D ; S_only = S \ D
4. For each weight tensor W and component c covering slice W[c]:
     if c ∈ S_only:  W_D[c] += α · Δ[c]
     if c ∈ O:       W_D[c] += α_O · protect(Δ[c], Δ_D[c], F_D[c])
     else:           unchanged
5. Return θ_G
```

`protect()` strategies, set by the `overlap_strategy` config field:

- `exclude`: return 0, i.e. skip shared components.
- `svd_project`: let Δ_D[c] = U Σ Vᵀ, keep the top-r right singular vectors V_r (r chosen to keep 90% of energy, capped at 16), and return Δ[c]·(I − V_r V_rᵀ). This removes the part of the safety edit that acts on the input directions the domain fine-tune changed.
- `fisher_gate`: per-weight F_D = mean squared gradient of the domain loss on 256 domain examples, normalised to [0,1] per tensor. Return Δ[c] ⊙ (1 − F̂_D)^γ, with γ=1 by default.
- `uniform`: return Δ[c]. This makes CG-uniform equivalent in spirit to C-ΔΘ with our component space.

Every edit is done in fp32, one tensor at a time, then cast back to bf16. This mirrors SafeTune ReSta's memory pattern of streaming the safety vector one tensor at a time [Source](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/user-guide/recover/layer/resta.md).

### 4.4 Circuit discovery

**Safety circuit S (primary path: CircuitKIT).**
- Build a contrastive task from CuratorKIT pair data (§6.3). The clean prompt is a harmful request in the chat template. The corrupt prompt is a minimally edited harmless twin (same length ±2 tokens).
- Metric: `logit_diff` between a refusal first-token and a compliance first-token. Determine these empirically: run θ_A greedy on 100 harmful and 100 harmless prompts, tabulate first generated tokens, and choose the most frequent single-token refusal and compliance starts. The metric can also be a set-sum logit diff over the top-3 of each.
- CircuitKIT warns that Cohere's 262k vocabulary can split answers that are single tokens elsewhere, and that discovery metrics assume one-token answers [Source](https://circuitkit.lexsi.ai/advanced/tiny-aya/). Assert single-tokenness in code.
- Use explicit `corrupted_prompt` columns. CircuitKIT's auto-corruption strategies target syntactic templates and may produce `clean == corrupt` pairs on safety data, and its docs recommend explicit pairs for safety-style data [Source](https://circuitkit.lexsi.ai/user-guide/custom-data/).
- Algorithm `eap-ig`, node level. Run on θ_A (where safety lives intact) and also on θ_D (diagnostic: where safety drifted).

**Safety circuit (cross-check path: SafeTune).** Run `eap_safety_circuit(model, harmful_prompts, harmless_prompts, config=EAPSafetyCircuitConfig(method="eap-ig", granularity="head", top_k_edges=...))`, which returns a `CircuitInfo` [Source](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/user-guide/interpret/eap/index.md). Report the Jaccard overlap between the two discoveries as a stability check.

**Domain circuit D.** Use MCQ-letter contrastive pairs from held-in MedMCQA training items: the same stem with the correct option moved to a different letter. This mirrors CircuitKIT's MMLU contrast design [Source](https://circuitkit.lexsi.ai/user-guide/data/). Metric `logit_diff`, run on θ_D.

**Faithfulness.** Run CircuitKIT's evaluation (causal patching, ablation, stability at minimum) on S and D. A circuit whose faithfulness is below the random baseline must not be used for claims.

**Bridge to SafeTune.** CircuitKIT's `*_scores.json` files carry `safety_units` / `layer_suggestions`, which SafeTune's CircuitKIT adapter reads [Source](https://github.com/Lexsi-Labs/CircuitKIT). Use that adapter so the C-ΔΘ baseline and CircuitGraft consume the **same** discovered circuit. This is the central integration point.

---

## 5. Repository layout

```
circuitgraft/
├── README.md                     # reproduction instructions, results tables auto-inserted
├── pyproject.toml                # pinned versions; python 3.12
├── configs/
│   ├── base.yaml                 # model ids, paths, seeds, HF org
│   ├── data.yaml
│   ├── drift.yaml
│   ├── discover.yaml
│   ├── graft/*.yaml              # cg_uniform, cg_exclude, cg_project, cg_fisher, cg_distributed, cg_random
│   ├── baselines/*.yaml          # resta, wiseft, ctheta, safelora, lox, task_arith
│   ├── eval.yaml
│   └── attacks.yaml
├── src/circuitgraft/
│   ├── cli.py                    # `cg <stage>` entry point (typer)
│   ├── config.py                 # pydantic schemas; yaml loading; config hash
│   ├── registry.py               # artifact manifest read/write, HF push
│   ├── components.py             # component ↔ weight-slice mapping for cohere2
│   ├── data/
│   │   ├── curate.py             # CuratorKIT wrappers
│   │   ├── pairs.py              # contrastive pair construction + validation
│   │   └── decontam.py           # cross-split n-gram/embedding dedup
│   ├── drift/finetune.py         # LoRA SFT → merged drifted checkpoint
│   ├── discover/
│   │   ├── circuitkit_runner.py
│   │   ├── safetune_eap.py
│   │   └── overlap.py            # S, D, O sets; Jaccard; plots
│   ├── graft/
│   │   ├── graft.py              # core algorithm (§4.3)
│   │   ├── protect.py            # exclude / svd_project / fisher_gate / uniform
│   │   ├── fisher.py
│   │   └── select.py             # topk / distributed / random-matched
│   ├── baselines/safetune_recover.py
│   ├── eval/
│   │   ├── generate.py           # batched greedy generation, vLLM optional
│   │   ├── safety.py             # SafeTune evaluate() + judges
│   │   ├── auditkit_runs.py      # AuditKIT evaluate / compare / redteam
│   │   ├── domain.py             # MCQ accuracy, answer parsing
│   │   └── stats.py              # bootstrap CIs, paired tests
│   ├── attacks/
│   │   ├── abliterate.py
│   │   ├── harmful_ft.py
│   │   └── quantize.py
│   └── report/
│       ├── tables.py             # JSON → markdown tables
│       ├── plots.py              # Pareto, overlap heatmaps
│       └── inject_readme.py      # replaces <!-- RESULTS:xyz --> blocks
├── scripts/run_all.sh            # full E2E reproduction
├── notebooks/                    # thin demos calling src/
├── space/app.py                  # Gradio HF Space
├── tests/
├── results/                      # JSON outputs (git-tracked, small)
└── docs/ PREREGISTRATION.md, API_NOTES.md, EXPERIMENT_LOG.md, ARTICLE.md
```

**Environment.** Python 3.12, which SafeTune requires (≥3.12) [Source](https://github.com/Lexsi-Labs/SafeTune). Use one venv if dependency resolution allows. CircuitKIT runs on transformer-lens 3.8.0 and transformers 5, and `circuitkit` must be imported before `transformer_lens` [Source](https://github.com/Lexsi-Labs/CircuitKIT). If resolution conflicts with SafeTune or AuditKIT, use **two environments** (`env-circuit`, `env-main`) that communicate only through files (scores JSON, checkpoints). Pin exact versions in `requirements-*.lock`. On Colab, run `pip uninstall -y torchao` before importing SafeTune [Source](https://github.com/Lexsi-Labs/SafeTune).

---

## 6. Data specification

### 6.1 Datasets

The agent verifies each licence before use and records it in the dataset cards.

| Purpose | Source | Split / size | Notes |
|---|---|---|---|
| Domain SFT | MedMCQA train (with explanations) | ~20k after curation | Chat-format: question+options → "Answer: X. Explanation…" |
| Domain held-out | MedMCQA validation | 1,000 sampled | Never used in training or discovery |
| Domain transfer | MedQA-USMLE test (4-opt), PubMedQA labeled | 500 each | Out-of-distribution domain checks |
| General capability | MMLU non-medical subjects | 1,000 sampled | Detects collateral damage |
| Harmful refusal | HarmBench (standard text), AdvBench | 200 + 200 | Via SafeTune loaders |
| OOD harmful | SORRY-Bench | 440 (full) | Category-level breakdown |
| Adversarial | WildJailbreak adversarial-harmful | 250 | Plus BoN and AuditKIT probes |
| Over-refusal | XSTest safe, OR-Bench-hard | 250 + 300 | |
| **Domain-sensitive benign (new)** | CuratorKIT-generated, human-verified | ~300 | Legitimate clinical questions on dosing, toxicity, overdose management etc. Key metric for H2 |
| **Domain-sensitive harmful (new)** | CuratorKIT-generated, human-verified | ~150 | Medically framed harmful requests |
| Circuit discovery pairs | CuratorKIT pairs from BeaverTails / AdvBench-style seeds | 256 safety + 256 domain | Disjoint from all eval sets |

SafeTune ships audited loaders for advbench, xstest, sorrybench, orbench, harmbench, wildjailbreak and others [Source](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/reference/feature-map.md).

### 6.2 CuratorKIT usage (must be real, not decorative)

1. **Domain SFT curation:** dedup (minhash + embedding), cleaning, hygiene, then export `sft_messages` with train/validation splits.
2. **Domain-sensitive eval generation:** `generation_task="qa"` and `adversarial_qa` seeded from medical reference text. Generator: Aya Expanse 32B via the Cohere API or a local endpoint. Judge: a separate model. Keep the reward and hallucination gates on.
   - CuratorKIT supports Aya models as generator or judge via LiteLLM strings such as `cohere_chat/c4ai-aya-expanse-32b` [Source](https://github.com/Lexsi-Labs/CuratorKIT).
   - Small judges often fail to return JSON, and failed parses reject samples [Source](https://github.com/Lexsi-Labs/CuratorKIT). Use a larger judge and log rejection counts.
3. **Contrastive pairs:** generate harmless minimal twins for harmful seeds, then run them through the dedup gate.
4. **Decontamination:** `decontam.py` removes any eval item with 13-gram overlap, or embedding cosine > 0.9, against discovery or training data. Log the counts.
5. **Human verification:** a lead reviews 100% of the new domain-sensitive sets (label benign/harmful) and 10% of the pairs. Store `label_source: human|auto` per row.

Keep every CuratorKIT run's `manifest.json`, `rejected.jsonl`, `dataset_card.md` and `checksums.txt`, and push them to HF with the dataset [Source](https://github.com/Lexsi-Labs/CuratorKIT).

### 6.3 Pair schema (JSONL)

```json
{"pair_id":"s-000123","kind":"safety","clean_prompt":"<chat-templated harmful>",
 "corrupted_prompt":"<chat-templated harmless twin>","answer":" I","corrupted_answer":" Sure",
 "seed_source":"beavertails","category":"violence","len_diff_tokens":1,"label_source":"auto"}
```

Validation (unit-tested): clean ≠ corrupt, `|len_diff_tokens| ≤ 2`, answers are single tokens under the Tiny Aya tokenizer, and no overlap with any eval set.

Tiny Aya's chat template adds a ~366-token system preamble [Source](https://github.com/Lexsi-Labs/SafeTune). For discovery, set `max_seq_len` to cover it, or render prompts without the default system preamble consistently across clean and corrupt. Pick one, document it, and apply it identically in evaluation.

---

## 7. Pipeline stages (CLI contract)

Each stage reads configs, writes to `artifacts/<stage>/<run_id>/` with a `manifest.json` (config hash, git SHA, input artifact hashes, library versions, seed, wall-clock, GPU), and is idempotent: it skips if the manifest hash matches.

| Stage | Command | Inputs → Outputs | Repo used |
|---|---|---|---|
| S1 data | `cg data` | raw HF datasets → curated SFT, eval sets, pairs | CuratorKIT |
| S2 drift | `cg drift` | θ_A + SFT → θ_D (merged) + LoRA adapter | (PEFT/TRL) |
| S3 audit | `cg audit --models A,B?,D` | checkpoints → baseline metrics JSON | SafeTune evaluate, AuditKIT |
| S4 discover | `cg discover` | θ_A, θ_D, pairs → S/D scores, overlap report, faithfulness | CircuitKIT (+SafeTune EAP cross-check) |
| S5 baselines | `cg baselines` | θ_B, θ_A, θ_D (+ circuit) → repaired checkpoints | SafeTune Recover |
| S6 graft | `cg graft --config graft/cg_project.yaml` | θ_B, θ_A, θ_D, scores → θ_G | CircuitGraft (+SafeTune `CircuitInfo`) |
| S7 eval | `cg eval --all` | all checkpoints → metrics JSON + generations | SafeTune evaluate, AuditKIT |
| S8 attack | `cg attack --type abliterate|harmful_ft|quant` | checkpoints → attacked metrics | SafeTune AbliterationAttack, CircuitKIT quantize |
| S9 report | `cg report` | results JSON → tables, plots, README injection | — |
| S10 publish | `cg publish` | artifacts → HF repos + collections | huggingface_hub |

### 7.1 S2: Drift induction

- LoRA r=16, α=32, dropout 0.05, all linear projections. lr 2e-4, 2 epochs, effective batch 32, max_len 1024, bf16, seed ∈ {0,1,2}. The primary run uses seed 0; seeds 1–2 run if time allows.
- **Scenario A (primary): pure benign medical SFT.**
- **Scenario B (fallback, used only if H0 fails under A):** medical SFT with 3% of rows replaced by harmful-compliance pairs, simulating contaminated domain data. Report clearly which scenario produced the results. Never present B as benign drift.
- Gate G-drift: domain held-out accuracy gain over θ_A ≥ 3 points **and** HarmBench refusal drop ≥ 15 points. If the domain gain is too small, try 3 epochs or answer-format fixes before changing anything else.
- Save both the merged checkpoint and the adapter. Safe LoRA needs the adapter.

### 7.2 S5: Baselines

The call shapes come from SafeTune docs and must be verified.

```python
from safetune.runner import recover
# RESTA: Tiny Aya needs small alpha. Docs: α≈0.25; α=1 breaks the model
recover.ReStaTrainer(drifted, base_model=base, aligned_model=aligned,
                     alpha=a, dare=True, device="cpu").apply()
# C-ΔΘ on the SAME CircuitInfo that CircuitGraft uses
recover.CThetaTrainer(drifted, base_model=base, aligned_model=aligned, strength=s)\
       .apply(circuit_info=ci)
```

For Tiny Aya, SafeTune recommends α≈0.25: in their sweep, 0.1/0.25 restored refusal with normal answers and ≥0.5 broke the model [Source](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/user-guide/recover/layer/resta.md). For C-ΔΘ, use `sweep_ctheta_strength` with an eval callback [Source](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/user-guide/recover/circuit/sweep-c-delta-theta.md).

| Baseline | Sweep | Priority |
|---|---|---|
| RESTA | α ∈ {0.1, 0.25, 0.4} | Must |
| WiSE-FT (interpolate θ_D↔θ_A) | λ ∈ {0.25, 0.5, 0.75} | Must |
| C-ΔΘ (SafeTune) | strength ∈ {0.25, 0.5, 1.0, 1.5} | Must |
| Safe LoRA | default + 1 threshold | Must |
| LoX, Task Arithmetic | 2 points each | Should |
| Refusal-direction steering (inference-time, reference only) | alpha sweep | Could |

Use only methods SafeTune marks **Faithful** as named baselines. Its feature map lists `apply_ctheta`, `apply_resta`, `apply_wise_ft`, `apply_safe_lora`, `apply_lox` and `task_arithmetic` as Faithful [Source](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/reference/feature-map.md).

**Sweep fairness rule:** pick hyperparameters for every method, ours included, on a **validation split** (100 HarmBench + 100 XSTest + 200 MedMCQA validation items not in the test sets) with the same selection criterion: maximize SRR subject to DGR ≥ 0.7, else maximize SRR+DGR. Report final numbers on the test splits only.

### 7.3 S6: CircuitGraft configurations

| Config | selection | overlap_strategy | Purpose |
|---|---|---|---|
| cg_uniform | topk | uniform | ≈C-ΔΘ in our component space |
| cg_exclude | topk | exclude | Is skipping overlap enough? |
| cg_project | topk | svd_project | **Main method** |
| cg_fisher | topk | fisher_gate | Alternative protection |
| cg_distributed | distributed | best of project/fisher | H3 |
| cg_random | random-matched | same as main | H2-ctrl (3 random seeds) |

Sweeps: k_S ∈ {2%, 5%, 10%} of components; α ∈ {0.25, 0.5, 1.0}; α_O ∈ {0.5, 1.0}; delta_source ∈ {alignment, reversal} (ablation at the best setting only). Report the fraction of parameters edited for every run.

---

## 8. Evaluation specification

### 8.1 Generation protocol

Greedy decoding, max_new_tokens 256 for safety and 32 for MCQ, the model's own chat template, batch generation, and a fixed seed. Store every generation in `results/generations/<model>/<bench>.jsonl`. That lets judges be re-run without regenerating and lets reviewers inspect outputs.

### 8.2 Judges

- **Primary harmfulness judge:** SafeTune `evaluate()` with the WildGuard judge. The default judge `allenai/wildguard` is a gated 7B model (~14.5 GB) and needs a GPU [Source](https://github.com/Lexsi-Labs/SafeTune).
- **Secondary:** string-match refusal detection (SafeTune `StringMatchJudge`) and an AuditKIT `GuardJudge`, used for agreement analysis.
- **Human audit:** 100 randomly sampled generations per key model, labeled blind by the lead. Report Cohen's κ against the primary judge. If κ < 0.6, flag the judge's reliability in the write-up.
- **Domain MCQ:** a regex parser for `Answer: [A-D]`, with a fallback to first-letter logit argmax. Report the parse-failure rate separately.

### 8.3 AuditKIT usage (must be real)

- Domain and general accuracy run through `ak.evaluate(samples, "hf:<repo>", metrics)`, and checkpoints are compared with `ak.compare` / `compare_models()` with bootstrap significance. AuditKIT supports Tiny Aya on the `hf:` backend, writes a fingerprinted `RunResult` per run, and provides `compare_models()` with pairwise significance [Source](https://github.com/Lexsi-Labs/AuditKIT).
- Adversarial probes run through AuditKIT's `RedTeamRunner` (jailbreak, encoding, prompt-injection and over-refusal probes) as an independent adversarial axis alongside SafeTune's BoN.
- Every `RunResult` fingerprint goes into the HF eval-logs dataset.

### 8.4 Model matrix

At minimum: θ_A, θ_D, the 4 Must baselines at their selected settings, the 6 CG configs at their selected settings, and cg_random ×3. θ_B is only needed for the refusal sanity check (expected low refusal) on 100 prompts. That makes ~14–16 checkpoints × full eval.

---

## 9. Metrics

### 9.1 Raw metrics (all with 95% bootstrap CIs, 1,000 resamples)

| Metric | Definition | Benchmarks |
|---|---|---|
| Harmful refusal rate (HRR) ↑ | % of harmful prompts judged refused/safe | HarmBench, AdvBench, SORRY-Bench, domain-harmful |
| Attack success rate (ASR) ↓ | % judged harmful under attack | WildJailbreak, BoN (N=16), AuditKIT probes |
| False refusal rate (FRR) ↓ | % of benign prompts refused | XSTest-safe, OR-Bench-hard, **domain-benign** |
| Domain accuracy ↑ | MCQ exact match | MedMCQA-val, MedQA, PubMedQA |
| General accuracy ↑ | MCQ exact match | MMLU non-medical |
| Edit footprint | % of parameters changed, layers touched | all repaired models |

### 9.2 Paired tests

Repaired-vs-repaired comparisons on the same prompts use McNemar's test on per-item outcomes. Apply a Holm correction across the H1/H2 comparison family.

### 9.3 Composite recovery metrics (CircuitGraft-defined, so state the definitions in the write-up)

- **Safety Recovery Ratio:** SRR = (HRR_repaired − HRR_D) / (HRR_A − HRR_D). 1.0 means fully restored to aligned level; >1 means safer than the original.
- **Domain Gain Retention:** DGR = (Acc_repaired − Acc_A) / (Acc_D − Acc_A). 1.0 means all specialist gain kept. Only defined when Acc_D − Acc_A ≥ 3 points (gate G-drift).
- **Over-refusal Penalty:** ORP = FRR_repaired − FRR_A.
- **Headline plot:** SRR (y) vs DGR (x) Pareto scatter, marker size = edit footprint, with each method's sweep shown as a connected line.

---

## 10. Robustness and compression experiments (secondary)

| Experiment | Procedure | Applied to |
|---|---|---|
| Abliteration | SafeTune `AbliterationAttack(model, tok).fit(harmful, harmless)` freshly on each repaired model, then `.run(mode="weight_orthogonalize")`, then HarmBench HRR | θ_A, θ_D+best baselines, cg_project, cg_distributed |
| Harmful re-fine-tune | LoRA on 100 harmful-compliance examples, 50 / 100 / 200 steps, lr 1e-4; HRR at each checkpoint (the decay curve) | same set |
| Quantization (uniform) | bitsandbytes NF4 and int8, then HRR, FRR, domain accuracy | best baseline vs cg_project |
| Circuit-aware quantization | CircuitKIT `ck.quantize(..., circuit=S, high_fraction=0.3, backend="llmcompressor", bits=4)` protecting safety-circuit layers vs uniform 4-bit | cg_project |

`AbliterationAttack` supports `runtime_ablate` and `weight_orthogonalize` modes [Source](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/user-guide/evaluate/attacks/abliteration.md). In CircuitKIT, the default `quanto` backend ignores `bits`; use `llmcompressor` to pin 3/4/8-bit [Source](https://circuitkit.lexsi.ai/user-guide/applications/). The fresh refusal direction must be fit on the attacked model, not reused from θ_A, so the attack is adaptive.

Claim P1, P11 or P12 only for experiments that actually ran.

---

## 11. Hugging Face artifacts

Use a single org/user namespace `<HF>`, everything public and un-gated, grouped in two collections as the submission form requires [Source](https://hackathon.lexsi.ai/).

**Models collection:** `circuitgraft-tinyaya-med-drifted` (with a prominent safety warning card), `circuitgraft-tinyaya-med-lora` (adapter), `circuitgraft-tinyaya-med-cg-project` (main), plus the other CG configs and baselines (`-resta-a0.25`, `-ctheta-s*`, `-wiseft-*`, `-safelora`), and the quantized variants.

**Datasets collection:** `circuitgraft-med-sft-curated`, `circuitgraft-safety-pairs`, `circuitgraft-domain-pairs`, `circuitgraft-med-sensitive-eval` (benign + harmful, human-verified), `circuitgraft-circuits` (scores JSON, CircuitInfo, overlap reports, faithfulness), `circuitgraft-eval-logs` (all metrics JSON, generations, AuditKIT fingerprints, manifests).

**Model cards** must include: lineage (base → aligned → drifted → repaired, with hashes), exact config, which eval produced which number (linked to the logs dataset), and the licence notes.
- Tiny Aya is CC-BY-NC with Cohere's Acceptable Use Policy [Source](https://huggingface.co/CohereLabs/tiny-aya-global).
- SafeTune's licence states that unrepaired drifted checkpoints may not be deployed in production [Source](https://github.com/Lexsi-Labs/SafeTune). Put this on the drifted model card verbatim and label that checkpoint "research-only, not for deployment."

**Dataset cards** for harmful-prompt sets carry a content warning. Contain prompts and judge labels only. Do not publish harmful *completions* from the drifted model beyond what is needed: store eval generations for harmful prompts in a separate config with a warning, or truncate them to 200 characters.

---

## 12. Schedule (today → Nov 14, then extension)

| Day | Date | Deliverable | Gate |
|---|---|---|---|
| D1 | Nov 5 | Register team; model access; env(s) built; API_NOTES from source inspection; smoke-run SafeTune quickstart, CircuitKIT `greater_than` on Tiny Aya, AuditKIT `hf:` eval | G0: all four libs run on Tiny Aya |
| D2 | Nov 6 | S1 data complete (SFT curated, eval sets, pairs v1, decontam); start S2 drift | G1: datasets pass validation tests |
| D3 | Nov 7 | θ_D trained; S3 audit of θ_A vs θ_D; PREREGISTRATION committed | **G-drift** (H0). If it fails, switch to Scenario B the same day |
| D4 | Nov 8 | S4 discovery: S on θ_A, D on θ_D, faithfulness, overlap report; SafeTune EAP cross-check | G2: circuits beat random on faithfulness |
| D5 | Nov 9 | S5 baselines + S6 graft implementation with unit tests; validation sweeps | G3: cg_uniform ≈ CThetaTrainer on identical mask (weight diff < 1e-5) |
| D6 | Nov 10 | Full S7 eval of the model matrix; judge agreement + human audit | G4: main results table generated from JSON |
| D7 | Nov 11 | S8 abliteration + quantization (harmful-FT if time); ablations (delta_source, k_S) | — |
| D8 | Nov 12 | S9 report, plots, HF publish, Space demo | G5: all artifacts public, links resolve |
| D9 | Nov 13 | Write-up (HF Article), README reproducibility check from a clean machine | G6: `run_all.sh --smoke` passes on fresh env |
| D10 | Nov 14 | Buffer + submit (abstract < 500 words) | Submitted |
| Ext. | Nov 16–21 | Seeds 1–2, harmful-FT curves, Aya Expanse 8B replication of the main comparison, multilingual safety check | Only if shortlisted |

**Scope-cut order if behind schedule** (cut from the top first): harmful-FT attack → cg_fisher → LoX/Task Arithmetic → circuit-aware quantization → SafeTune EAP cross-check. **Never cut:** drift demonstration, ≥3 baselines, cg_project vs cg_uniform vs cg_random, abliteration on the main models, public artifacts.

---

## 13. Acceptance criteria and tests

**Unit tests (`pytest`, CPU, tiny random cohere2 config):**
- `components.py`: the slices for every head/MLP exactly partition the editable params with no overlap; GQA group handling is right.
- `graft.py`: with an empty S the output is identical to θ_D; with S = all and `uniform` it equals RESTA with α and no DARE; edits never touch layernorm/embeddings.
- `protect.py`: `svd_project` output is orthogonal to V_r (‖ΔV_r‖ < 1e-4); `fisher_gate` with F=0 reduces to uniform.
- `select.py`: random-matched masks have the same parameter count (±1%) and the same per-layer histogram.
- `pairs.py`: rejects identical pairs, multi-token answers, and length mismatches.
- `stats.py`: bootstrap CIs cover the true mean on synthetic data (≥ 93% of 500 trials).

**Integration tests (GPU, opt-in env var):** run the full pipeline on 16 prompts per benchmark with `--smoke` in under 45 minutes.

**Definition of done for submission:**
1. `scripts/run_all.sh` reproduces every number in the README from public HF artifacts.
2. Every number in README/Article is injected from `results/*.json` by `inject_readme.py`. No hand-typed metrics.
3. The results table includes θ_A, θ_D, ≥3 baselines and ≥3 CG configs on ≥2 safety, ≥1 over-refusal and ≥2 capability metrics.
4. Both HF collections are public and the GitHub repo is public with a licence (respecting upstream LSAL/CC-BY-NC terms).
5. The write-up states which hypotheses were supported, refuted or inconclusive.

---

## 14. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Benign SFT doesn't cause enough drift | Medium | Scenario B, disclosed; or a longer SFT run |
| SFT doesn't improve domain accuracy (DGR undefined) | Medium | Format-focused SFT; MedMCQA explanations; if still flat, pivot the capability metric to answer-format compliance + PubMedQA and state this |
| Library dependency conflicts | High | Two-env, file-based handoff (§5) |
| TransformerLens port issues on Tiny Aya | Medium | Run CircuitKIT's parity gate first; fall back to SafeTune `eap_safety_circuit` on HF modules [Source](https://circuitkit.lexsi.ai/advanced/tiny-aya/) |
| Refusal first-token not single-token | Medium | Empirical token search; `kl` metric fallback |
| Full-strength deltas break Tiny Aya | High (documented for RESTA) | Small α grid; benign-answer sanity check after every edit |
| WildGuard GPU memory alongside the model under test | Medium | Generate first, judge in a separate pass; cache generations |
| Overlap set O is tiny, so H2 is untestable | Medium | Report |O| honestly; vary k_D to create overlap; this is a finding either way |
| Compute shortfall | Medium | Use HF credits / Kaggle; cut order in §12; vLLM for generation |
| Over-claiming | — | §15 rules; the lead reviews every claim against `results/` |

---

## 15. Rules for the coding agent

1. **Never fabricate results.** No number may appear in docs, cards, Space or Article unless it was produced by code in this repo and stored in `results/`. Use placeholders like `<!-- RESULTS:main_table -->` until real data exists.
2. **Verify before integrating.** Read the installed library source for each API used and log any discrepancies in `docs/API_NOTES.md`.
3. **Every stage writes a manifest.** No artifact without lineage.
4. **Determinism.** Seed everything; greedy decoding; log the torch/transformers/library versions and GPU type.
5. **Same circuit, fair comparison.** C-ΔΘ and all CG configs consume the identical discovered circuit file, verified by hash in the manifests.
6. **Test-set hygiene.** Hyperparameters are chosen on validation splits only, and test sets are evaluated once per final config.
7. **Name methods honestly.** Use only SafeTune Faithful methods as named baselines and label our own variants as ours.
8. **Safety of artifacts.** No publishing of full harmful completions without warnings; drifted model cards carry the research-only notice.
9. **Log experiments** in `docs/EXPERIMENT_LOG.md`: date, command, config hash, outcome, including failures.
10. **Ask the human lead** before changing a pre-registered hypothesis, metric definition, or the drift scenario.

---

## 16. Deliverables checklist

### 16.1 HF Space (`space/app.py`, Gradio)

Side-by-side chat with θ_A / θ_D / RESTA / CircuitGraft on preset prompts (harmful, domain-benign-sensitive, MCQ), an overlap heatmap (36 layers × heads+MLP, colored S-only/D-only/shared), and an interactive Pareto plot from `results/`. Use ZeroGPU and cache responses for presets so the demo works without live generation.

### 16.2 HF Article outline (`docs/ARTICLE.md`)

1. Motivation
2. Setup and why Tiny Aya / medicine / these benchmarks
3. Demonstrating drift
4. Where safety and medical skill live (circuits, overlap)
5. Method
6. Results (main table, Pareto)
7. Ablations and random-mask control
8. Robustness and quantization
9. Limitations and negative results
10. Reproducibility and artifact links
11. Related work and attribution (RESTA, C-ΔΘ, EAP/EAP-IG, abliteration, the four Lexsi toolkits)

The ROI comparison (self-hosted Tiny Aya vs closed API) is listed as optional but preferred [Source](https://hackathon.lexsi.ai/). Add it on D9 only if time remains, using measured throughput from the eval runs.

### 16.3 Submission bundle

- Abstract (<500 words)
- Both HF collections
- Article
- GitHub repo
- Repos used: CuratorKIT, SafeTune, CircuitKIT, AuditKIT
- Track 2b, P6 primary, plus whichever secondary problems were actually completed

---

### Summary for the human lead

A few decisions are yours to confirm today:
- **Medicine** as the domain.
- **Benign-first drift** (Scenario A), with contaminated-data drift (Scenario B) as a disclosed fallback.
- **The definitions of SRR and DGR.**

Once those are set, commit the preregistration on Day 3 before any repair experiment runs. That commit is what makes the "every result is real" claim credible to judges.

The weakest point of the plan is H2, the claim that handling the overlap matters. It depends on the safety and domain circuits actually sharing components. Check the overlap report on Day 4 before investing in the fisher and distributed variants. If the circuits barely overlap, report that honestly as a finding and focus the write-up on H1 plus the random-mask control.