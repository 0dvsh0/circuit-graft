# S7 proxy mechanics smoke (NOT G4, NOT Tiny Aya)

This folder preserves the output of a real end-to-end S7 run on two small
public proxy models, not on Tiny Aya:

- `aligned` -> `HuggingFaceTB/SmolLM-135M-Instruct`
- `drifted` -> `HuggingFaceTB/SmolLM-135M`

Only three benchmarks from the partial S1 smoke data were used, four rows each:
`orbench_hard`, `medmcqa_validation` and `mmlu_non_medical`. Generation used the
real `transformers` loader and greedy decoding; judging used SafeTune's
`StringMatchJudge`; domain/general accuracy used real AuditKIT
`ak.evaluate(..., "precomputed")` runs saved under `artifacts/eval_smoke/`.

Observed outcome (2026-11-07):

| model | HRR | FRR | domain acc | general acc | AuditKIT groups |
|---|---|---|---|---|---|
| aligned | n/a | 0.0 | 0.0 | 0.0 | domain_mcq, general_mcq |
| drifted | n/a | 0.0 | 0.0 | 0.0 | domain_mcq, general_mcq |

`gate_g_drift = not_evaluated` because the smoke data has no safety
benchmarks. `gate_g4 = not_evaluated`; this run is a pipeline mechanics check
only and says nothing about Tiny Aya, safety drift, or CircuitGraft recovery.
The zero accuracies are real outputs of SmolLM-135M on four-row samples, not a
fabricated CircuitGraft result.

`summary.json` and `manifest.json` are copied verbatim from
`artifacts/eval_smoke/eval-e71c7a6ce0c41416/`. Item-level outcomes and AuditKIT
RunResults remain in the git-ignored `artifacts/` tree; the manifest records
their hashes.
