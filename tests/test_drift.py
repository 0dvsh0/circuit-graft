"""S2 drift tests: contamination, tokenisation, saving and the G-drift gate.

The training test uses an injected tiny fake model/tokenizer/trainer so it is
CPU-only and network-free. The real TRL/transformers path is exercised by the
GPU integration run, not by this unit suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from circuitgraft.config import BaseConfig, DriftConfig
from circuitgraft.drift import finetune


class FakeTokenizer:
    """Character tokenizer with a deterministic chat template.

    Character-level encoding makes the prompt a strict prefix of the full
    dialogue by construction, which is exactly what the loss mask requires.
    """

    chat_template = "fake-template"
    eos_token = "</s>"
    eos_token_id = 0
    pad_token = "</s>"
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        assert tokenize is False
        body = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return body + "<assistant>" if add_generation_prompt else body

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}

    def save_pretrained(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class FakeParam:
    def __init__(self, count: int, requires_grad: bool) -> None:
        self.count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.count


class FakeModel:
    def __init__(self) -> None:
        self.merged = False
        self.saved: list[Path] = []

    def parameters(self) -> list[FakeParam]:
        return [FakeParam(10, True), FakeParam(90, False)]

    def save_pretrained(self, directory: str, **_: Any) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.saved.append(path)
        (path / "config.json").write_text("{}", encoding="utf-8")
        if path.name == "adapter":
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (path / "adapter_model.safetensors").write_bytes(b"adapter")
        else:
            (path / "model.safetensors").write_bytes(b"weights")

    def merge_and_unload(self) -> FakeModel:
        self.merged = True
        return self


class FakeTrainer:
    instances: list[FakeTrainer] = []

    def __init__(self, model: Any, args: Any, train_dataset: Any, data_collator: Any, processing_class: Any) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.processing_class = processing_class
        self.state = SimpleNamespace(
            log_history=[{"loss": 1.0}, {"loss": 0.5}],
            global_step=2,
        )
        FakeTrainer.instances.append(self)

    def train(self, resume_from_checkpoint: str | None = None, **kwargs: Any) -> dict[str, Any]:
        self.resume_from_checkpoint = resume_from_checkpoint
        return {"train_runtime": 1.5, "train_samples_per_second": 4.0, "epoch": 1.0}


def messages(question: str, answer: str = "Answer: A") -> list[dict[str, str]]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_config(tmp_path: Path, **overrides: Any) -> DriftConfig:
    values: dict[str, Any] = {
        "device": "cpu",
        "dtype": "float32",
        "seed": 0,
        "sft_dataset": str(tmp_path / "sft"),
        "output_dir": str(tmp_path / "drift"),
        "train": {
            "epochs": 1,
            "lr": 2e-4,
            "batch_size": 1,
            "grad_accum": 1,
            "max_seq_len": 2048,
            "bf16": False,
            "gradient_checkpointing": False,
            "warmup_ratio": 0.0,
            "logging_steps": 1,
            "save_steps": 100,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(values.get(key), dict):
            values[key] = {**values[key], **value}
        else:
            values[key] = value
    return DriftConfig(**values)


def make_base() -> BaseConfig:
    return BaseConfig(models={"base": "test/base", "aligned": "test/aligned"})


def prepare_sft(tmp_path: Path, count: int = 4) -> None:
    rows = [
        {"messages": messages(f"Question {index}?", f"Answer: {'ABCD'[index % 4]}")}
        for index in range(count)
    ]
    write_jsonl(tmp_path / "sft" / "train.jsonl", rows)
    write_jsonl(tmp_path / "sft" / "validation.jsonl", rows[:1])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_load_sft_rows_accepts_messages_and_rejects_malformed(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(
        path,
        [
            {"messages": messages("q1")},
            {"messages": [{"role": "assistant", "content": "orphan"}]},
            {"messages": [{"role": "user", "content": "q2"}]},
            {"messages": messages("q3"), "id": "row-3"},
        ],
    )
    rows = finetune.load_sft_rows(path)
    assert len(rows) == 2
    assert rows[1]["id"] == "row-3"


def test_harmful_rows_are_converted_but_never_generated(tmp_path: Path) -> None:
    assert finetune.harmful_to_messages(
        {"prompt": "do x", "response": "here is x"}
    ) == messages("do x", "here is x")
    assert finetune.harmful_to_messages({"messages": messages("q")}) == messages("q")
    assert finetune.harmful_to_messages({"prompt": "missing response"}) is None
    assert finetune.harmful_to_messages({"response": "missing prompt"}) is None

    path = tmp_path / "harmful.jsonl"
    write_jsonl(path, [{"prompt": "q", "response": "r", "id": "h1"}])
    rows = finetune.load_harmful_rows(path)
    assert rows == [{"messages": messages("q", "r"), "id": "h1"}]


def test_contamination_is_deterministic_and_disclosed() -> None:
    rows = [{"messages": messages(f"q{index}"), "id": f"r{index}"} for index in range(10)]
    harmful = [{"messages": messages(f"h{index}", "bad"), "id": f"h{index}"} for index in range(3)]
    first, report = finetune.contaminate_rows(rows, harmful, fraction=0.3, seed=0)
    second, report_again = finetune.contaminate_rows(rows, harmful, fraction=0.3, seed=0)
    assert first == second
    assert report["rows_replaced"] == 3
    assert report["positions"] == report_again["positions"]
    replaced = [row["id"] for row in first if str(row["id"]).startswith("h")]
    assert len(replaced) == 3
    assert report["applied"] is True and report["disclosed"] is True


def test_contamination_rejects_degenerate_fractions() -> None:
    rows = [{"messages": messages("q")}]
    harmful = [{"messages": messages("h", "bad")}]
    with pytest.raises(ValueError, match="zero"):
        finetune.contaminate_rows(rows, harmful, fraction=0.01, seed=0)
    with pytest.raises(ValueError, match="\\(0, 1\\)"):
        finetune.contaminate_rows(rows, harmful, fraction=1.0, seed=0)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenize_messages_masks_the_prompt() -> None:
    tokenizer = FakeTokenizer()
    example = finetune.tokenize_messages(tokenizer, messages("hello"), max_seq_len=2048)
    assert example is not None
    labels = example["labels"]
    first_supervised = next(index for index, label in enumerate(labels) if label != -100)
    assert all(label == -100 for label in labels[:first_supervised])
    assert labels[first_supervised:] == example["input_ids"][first_supervised:]
    assert example["attention_mask"] == [1] * len(example["input_ids"])


def test_tokenize_messages_drops_examples_whose_answer_is_truncated() -> None:
    tokenizer = FakeTokenizer()
    prompt_length = len(
        tokenizer.apply_chat_template(
            messages("hello")[:-1], tokenize=False, add_generation_prompt=True
        )
    )
    assert finetune.tokenize_messages(
        tokenizer, messages("hello"), max_seq_len=prompt_length
    ) is None


def test_tokenize_dataset_reports_drops() -> None:
    tokenizer = FakeTokenizer()
    rows = [
        {"messages": messages("q1")},
        {"messages": messages("q2")},
    ]
    dataset, stats = finetune.tokenize_dataset(rows, tokenizer, max_seq_len=2048)
    assert len(dataset) == 2
    assert stats["rows_used"] == 2
    assert stats["rows_dropped"] == 0
    assert stats["min_supervised_tokens"] > 0


def test_padding_collator_pads_labels_with_ignore_index() -> None:
    import torch

    collator = finetune.PaddingCollator(pad_token_id=7)
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2], [3, 7]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]
    assert batch["input_ids"].dtype == torch.long


# ---------------------------------------------------------------------------
# G-drift gate
# ---------------------------------------------------------------------------


def test_drift_gate_passes_only_when_both_thresholds_are_met() -> None:
    aligned = {"domain_accuracy": 0.40, "harmbench_refusal_rate": 0.90}
    drifted = {"domain_accuracy": 0.45, "harmbench_refusal_rate": 0.70}
    gate = finetune.evaluate_drift_gate(aligned, drifted)
    assert gate["status"] == "passed"
    assert gate["checks"]["domain_gain"]["value"] == pytest.approx(0.05)
    assert gate["checks"]["refusal_drop"]["value"] == pytest.approx(0.20)

    weak = finetune.evaluate_drift_gate(
        aligned, {"domain_accuracy": 0.42, "harmbench_refusal_rate": 0.80}
    )
    assert weak["status"] == "failed"
    assert set(weak["reasons"]) == {"domain_gain", "refusal_drop"}


def test_drift_gate_reads_nested_metric_documents() -> None:
    aligned = {"benchmarks": {"harmbench": {"refusal_rate": 0.9}},
               "medmcqa_validation": {"accuracy": 0.40}}
    drifted = {"benchmarks": {"harmbench": {"refusal_rate": 0.7}},
               "medmcqa_validation": {"accuracy": 0.46}}
    gate = finetune.evaluate_drift_gate(aligned, drifted)
    assert gate["status"] == "passed"


def test_drift_gate_never_invents_missing_metrics() -> None:
    gate = finetune.evaluate_drift_gate({"domain_accuracy": 0.4}, {})
    assert gate["status"] == "not_evaluated"
    assert len(gate["reasons"]) == 3


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_run_drift_saves_adapter_and_merged_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepare_sft(tmp_path)
    cfg = make_config(tmp_path)
    model = FakeModel()
    FakeTrainer.instances.clear()
    monkeypatch.setattr(finetune, "build_training_args", lambda cfg, out, device, **kw: SimpleNamespace(to_dict=lambda: {"device": device}))

    result = finetune.run_drift(
        cfg,
        make_base(),
        model=model,
        tokenizer=FakeTokenizer(),
        trainer_factory=FakeTrainer,
        lora_wrapper=lambda injected, _cfg: injected,
    )

    assert result["skipped"] is False
    assert Path(result["adapter"]).is_dir()
    assert Path(result["merged"]).is_dir()
    assert (Path(result["merged"]) / "model.safetensors").exists()
    summary = json.loads((Path(result["output_dir"]) / "drift_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["scenario"] == "A"
    assert summary["contamination"]["applied"] is False
    assert summary["train"]["trainable"] == 10
    assert summary["train"]["train_loss"] == pytest.approx(0.75)
    assert summary["gate"]["status"] == "not_evaluated"
    manifest = json.loads((Path(result["output_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["stage"] == "drift"
    assert "model:aligned" in manifest["inputs"]
    assert FakeTrainer.instances[0].data_collator is not None

    # The same config and inputs must be idempotent.
    again = finetune.run_drift(
        cfg,
        make_base(),
        model=model,
        tokenizer=FakeTokenizer(),
        trainer_factory=FakeTrainer,
        lora_wrapper=lambda injected, _cfg: injected,
    )
    assert again["skipped"] is True


def test_run_drift_scenario_b_replaces_rows_and_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepare_sft(tmp_path, count=10)
    harmful_path = tmp_path / "harmful.jsonl"
    write_jsonl(
        harmful_path,
        [{"prompt": f"harmful {index}", "response": "compliance", "id": f"h{index}"} for index in range(2)],
    )
    cfg = make_config(
        tmp_path,
        scenario="B",
        harmful_rows=str(harmful_path),
        harmful_fraction=0.2,
    )
    monkeypatch.setattr(finetune, "build_training_args", lambda cfg, out, device, **kw: SimpleNamespace(to_dict=lambda: {}))

    result = finetune.run_drift(
        cfg,
        make_base(),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        trainer_factory=FakeTrainer,
        lora_wrapper=lambda injected, _cfg: injected,
    )
    summary = result["summary"]
    assert summary["contamination"]["applied"] is True
    assert summary["contamination"]["rows_replaced"] == 2
    assert summary["contamination"]["disclosed"] is True
    assert "not benign drift" in summary["warning"]
    manifest = json.loads((Path(result["output_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert "scenario=B" in manifest["notes"]


def test_scenario_b_config_requires_an_explicit_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="harmful_rows"):
        DriftConfig(
            scenario="B",
            sft_dataset=str(tmp_path),
            output_dir=str(tmp_path / "drift"),
        )
