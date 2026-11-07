---
tags:
- curatorkit
configs:
- config_name: sft_alpaca
  default: true
  description: curatorkit sft_alpaca export, sha256 43d2d9d026b19fc9e569b9986fecc15647cf63609d5f5d26e665afb3acb257d9
  data_files:
  - split: train
    path: sft_alpaca.jsonl
---

# Dataset Card: d1-smoke

## Summary

- **Run timestamp**: 2026-11-07T12:38:06.903915Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `bc1ee67ef9081c6a`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 8 | 8 | — | 0 | 8 |
| ExactDeduplicator | 8 | 7 | — | — | 7 |
| TextCleaner | 7 | 7 | — | — | 7 |
| MaxSamplesTruncator | 7 | 7 | — | — | 7 |
| AlpacaExporter | — | 7 | — | — | 7 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 8/0 passed (N/A), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - exact_duplicates_removed: 1

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.22
- **python**: 3.13.16

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_alpaca` | train (7) | `instruction`: string<br>`input`: string (empty when there is no input)<br>`output`: string |
