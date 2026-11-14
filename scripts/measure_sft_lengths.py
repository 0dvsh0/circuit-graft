"""Measure SFT row token lengths (CPU-only) to size the T4 drift batch."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from circuitgraft.drift.finetune import render_chat  # noqa: E402

tok = AutoTokenizer.from_pretrained(
    "CohereLabs/tiny-aya-base", trust_remote_code=True
)

lens = []
with open("artifacts/data/medmcqa_sft/train.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("rows:", len(rows))
print("row keys:", sorted(rows[0].keys()))


def render(row):
    # Same rendering as the training pipeline (base tokenizer has no chat
    # template -> deterministic "User:/Assistant:" fallback).
    return tok(render_chat(tok, row["messages"]))["input_ids"]


for i, row in enumerate(rows):
    lens.append(len(render(row)))
    if i == 0:
        print("sample rendered length:", lens[0])

lens.sort()
n = len(lens)
print(f"min={lens[0]} p50={lens[n//2]} p90={lens[int(n*0.9)]} p99={lens[int(n*0.99)]} max={lens[-1]}")
print(f"mean={sum(lens)/n:.0f}")

# micro-step cost model: logits fp32 materialization = batch*len*d_vocab*4B
d_vocab = 262144
for batch in (4, 8):
    for seq in (lens[n // 2], lens[int(n * 0.99)]):
        logits_fp32_gb = batch * seq * d_vocab * 4 / 2**30
        print(f"batch {batch} seq {seq}: fp32 logits {logits_fp32_gb:.2f} GiB/fwd")
