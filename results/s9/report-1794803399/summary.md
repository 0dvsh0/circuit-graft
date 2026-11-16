# CircuitGraft Experiment Summary

**Status:** partial

## 1. Safety Recovery vs Domain Retention
| Model / Method | SRR | DGR | HarmBench HRR | AdvBench HRR | MedMCQA Acc | MMLU Non-med Acc | XSTest FRR | Footprint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aligned | 100.0% | 0.0% | 76.7% | 95.0% | 36.0% | 37.0% | 10.0% | — |
| cg_project | -322.2% | -57.1% | 69.3% | 87.5% | 40.0% | 40.0% | 7.5% | — |
| cg_random | -233.3% | -60.7% | 70.0% | 90.0% | 40.0% | 41.0% | 7.5% | — |
| cg_uniform | -322.2% | -142.9% | 66.7% | 85.0% | 36.7% | 39.0% | 13.8% | — |
| ctheta (SafeTune weight-arithmetic implementation) | -3144.4% | -160.7% | 6.0% | 3.8% | 24.7% | 29.0% | 2.5% | — |
| drifted | 0.0% | 100.0% | 81.3% | 92.5% | 40.7% | 54.0% | 3.8% | — |
| resta | -877.8% | 7.1% | 52.7% | 80.0% | 39.3% | 46.0% | 0.0% | — |
| wiseft | 322.2% | 96.4% | 87.3% | 95.0% | 44.7% | 51.0% | 8.8% | — |

![Pareto Frontier](results/s9/report-1794803399/pareto_frontier.png)

## 2. Circuit Overlap
| Metric | Safety Circuit (S) | Domain Circuit (D) | Overlap (S ∩ D) |
|---|---|---|---|
| Component Count | 184 | 184 | 72 |
| Attention Heads | — | — | 45 |
| MLP Blocks | — | — | 27 |
| Jaccard Similarity | — | — | 0.243 |

## 3. Adversarial Robustness
| Model / Method | Clean HRR | Abliteration HRR | Harmful-FT (50 st) | Harmful-FT (100 st) | Harmful-FT (200 st) | Quant NF4 HRR | Circuit-Quant 4b HRR |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Partial reasons
- eval summary is partial
- human audit is pending review
- no S8 attack statistics found (run `cg attack ...`)
