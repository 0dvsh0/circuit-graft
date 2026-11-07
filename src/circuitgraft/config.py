"""Configuration loading, validation and hashing.

Every stage config is validated with pydantic before any work happens. Stage
files may override values from ``configs/base.yaml``; the merged mapping is what
gets hashed into the run manifest, so a manifest identifies the *effective*
configuration rather than just one YAML file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]

# ``.env`` is local and git-ignored. Loading it here (never overriding an
# explicit shell variable) keeps config expansion and CLI token checks
# consistent for every entry point, including tests.
load_dotenv(REPO_ROOT / ".env", override=False)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _expand(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in string values."""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            env_value = os.environ.get(name)
            if env_value is None or env_value == "":
                if default is None:
                    return ""
                return default
            return env_value

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    return _expand(raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path,
    schema: type[BaseModel],
    *,
    base_path: str | Path | None = "configs/base.yaml",
    extra: dict[str, Any] | None = None,
) -> BaseModel:
    """Load a stage config, merged over ``base.yaml`` and validated by ``schema``."""
    stage = load_yaml(path)
    if base_path is not None:
        stage = _deep_merge(load_yaml(base_path), stage)
    if extra:
        stage = _deep_merge(stage, extra)
    # Resolve relative paths against the repository root for reproducibility.
    return schema.model_validate(stage)


def canonical_json(value: BaseModel | dict[str, Any]) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(value: BaseModel | dict[str, Any], length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


class PathsConfig(StrictModel):
    artifacts: str = "artifacts"
    results: str = "results"
    cache: str = ".cache"

    def artifact_dir(self, *parts: str) -> Path:
        return REPO_ROOT / self.artifacts / Path(*parts)

    def results_dir(self, *parts: str) -> Path:
        return REPO_ROOT / self.results / Path(*parts)


class ModelRefs(StrictModel):
    base: str = "CohereLabs/tiny-aya-base"
    aligned: str = "CohereLabs/tiny-aya-global"
    drifted: str | None = None

    @field_validator("base", "aligned", "drifted")
    @classmethod
    def _non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("model reference must not be empty")
        return value


class CommonConfig(StrictModel):
    """Settings inherited from ``configs/base.yaml`` by every stage config."""

    seed: int = 0
    hf_org: str = "${HF_ORG}"
    device: str = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    hf_home: str | None = None
    models: ModelRefs = Field(default_factory=ModelRefs)
    paths: PathsConfig = Field(default_factory=PathsConfig)


class BaseConfig(CommonConfig):
    @model_validator(mode="after")
    def _check_models(self) -> BaseConfig:
        # ``EvalConfig`` / ``AttacksConfig`` override ``models`` with a matrix
        # of logical model names, so guard the model-reference check.
        if isinstance(self.models, ModelRefs) and self.models.base == self.models.aligned:
            raise ValueError("base and aligned model must differ")
        return self


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class SftDataConfig(StrictModel):
    source: str = "medmcqa"
    train_split: str = "train"
    validation_split: str = "validation"
    max_train_rows: int = 20000
    max_validation_rows: int = 1000
    answer_format: str = "Answer: {letter}. Explanation: {explanation}"
    require_explanation: bool = True
    min_explanation_chars: int = 20


class EvalDataConfig(StrictModel):
    medmcqa_validation: int = 1000
    medqa_test: int = 500
    pubmedqa_labeled: int = 500
    mmlu_non_medical: int = 1000
    harmbench: int = 200
    advbench: int = 200
    sorrybench: int = 440
    wildjailbreak: int = 250
    xstest_safe: int = 250
    orbench_hard: int = 300
    domain_benign: int = 300
    domain_harmful: int = 150
    safety_pairs: int = 256
    domain_pairs: int = 256
    selection_harmbench: int = 100
    selection_xstest_safe: int = 100
    selection_medmcqa_validation: int = 200
    seed: int = 0


class CurateConfig(StrictModel):
    generator_model: str = "cohere_chat/c4ai-aya-expanse-32b"
    judge_model: str = "cohere_chat/c4ai-aya-expanse-32b"
    tasks: list[str] = Field(default_factory=lambda: ["qa", "adversarial_qa"])
    min_reward: float = 0.6
    max_hallucination: float = 0.4
    human_verify_fraction: float = 1.0
    pairs_human_verify_fraction: float = 0.1
    push_rejected: bool = True
    seed: int = 0
    enable_generation: bool = True
    num_questions: int = 3
    embedding_dedup: bool = True
    twin_cache: str = ".cache/harmless_twins.json"


class DecontamConfig(StrictModel):
    ngram_size: int = 13
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_threshold: float = 0.9
    use_embeddings: bool = True


class PairDataConfig(StrictModel):
    safety_source: str = "beavertails"
    domain_source: str = "medmcqa"
    max_len_diff_tokens: int = 2
    require_single_token_answers: bool = True
    max_seq_len: int = 1024
    include_system_preamble: bool = False


class DataConfig(CommonConfig):
    sft: SftDataConfig = Field(default_factory=SftDataConfig)
    eval: EvalDataConfig = Field(default_factory=EvalDataConfig)
    curate: CurateConfig = Field(default_factory=CurateConfig)
    decontam: DecontamConfig = Field(default_factory=DecontamConfig)
    pairs: PairDataConfig = Field(default_factory=PairDataConfig)
    output_dir: str = "artifacts/data"


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


class LoraConfig(StrictModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"
    bias: str = "none"


class SftTrainConfig(StrictModel):
    epochs: int = 2
    lr: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 8
    max_seq_len: int = 1024
    bf16: bool = True
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 500


class DriftConfig(CommonConfig):
    scenario: Literal["A", "B"] = "A"
    harmful_fraction: float = 0.03
    # Scenario B only: explicit harmful-compliance rows. Never generated by
    # this repository. Each row is a JSON object with ``messages`` or
    # ``prompt`` + ``response``/``completion``.
    harmful_rows: str | None = None
    seed: int = 0
    lora: LoraConfig = Field(default_factory=LoraConfig)
    train: SftTrainConfig = Field(default_factory=SftTrainConfig)
    sft_dataset: str = "artifacts/data/medmcqa_sft"
    output_dir: str = "artifacts/drift"
    merge_adapter: bool = True
    gate_domain_gain: float = 0.03
    gate_refusal_drop: float = 0.15

    @model_validator(mode="after")
    def _scenario_b_needs_source(self) -> DriftConfig:
        if self.scenario == "B":
            if not self.harmful_rows:
                raise ValueError(
                    "scenario B requires harmful_rows: an explicit JSONL file of "
                    "harmful-compliance examples (never generate these automatically)"
                )
            if not 0.0 < self.harmful_fraction < 1.0:
                raise ValueError("harmful_fraction must be in (0, 1) for scenario B")
        return self


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class SafetyDiscoveryConfig(StrictModel):
    pairs: str = "artifacts/data/pairs/safety.jsonl"
    model: Literal["aligned", "drifted"] = "aligned"
    method: str = "eap-ig"
    granularity: str = "node"
    top_k_edges: int = 1000
    ig_steps: int = 5
    n_examples: int = 256
    sparsity: float = 0.3
    max_seq_len: int = 1024
    batch_size: int = 4
    metric: str = "logit_diff"
    refusal_tokens: list[str] = Field(default_factory=list)
    compliance_tokens: list[str] = Field(default_factory=list)


class DomainDiscoveryConfig(StrictModel):
    pairs: str = "artifacts/data/pairs/domain.jsonl"
    model: Literal["aligned", "drifted"] = "drifted"
    method: str = "eap-ig"
    granularity: str = "node"
    top_k_edges: int = 1000
    ig_steps: int = 5
    n_examples: int = 256
    sparsity: float = 0.3
    max_seq_len: int = 1024
    batch_size: int = 4
    metric: str = "logit_diff"


class FaithfulnessConfig(StrictModel):
    enabled: bool = True
    methods: list[str] = Field(
        default_factory=lambda: ["patching", "ablation", "stability"]
    )
    n_examples: int = 64
    random_controls: int = 3
    min_auc_ratio_vs_random: float = 1.2

    @field_validator("methods")
    @classmethod
    def _known_pillars(cls, value: list[str]) -> list[str]:
        # CircuitKIT evaluation pillar names (full.py, PILLARS_VALID).
        valid = {
            "patching",
            "ablation",
            "baselines",
            "robustness",
            "stability",
            "generalization",
            "intervention_reliability",
        }
        unknown = sorted(set(value) - valid)
        if unknown:
            raise ValueError(f"unknown faithfulness pillars: {unknown}; valid: {sorted(valid)}")
        return value

    @model_validator(mode="after")
    def _enabled_needs_pillars(self) -> FaithfulnessConfig:
        if self.enabled and not self.methods:
            raise ValueError("faithfulness.methods must not be empty when enabled")
        return self


class DiscoverConfig(CommonConfig):
    safety: SafetyDiscoveryConfig = Field(default_factory=SafetyDiscoveryConfig)
    domain: DomainDiscoveryConfig = Field(default_factory=DomainDiscoveryConfig)
    cross_check: bool = True
    cross_check_method: str = "eap-ig"
    faithfulness: FaithfulnessConfig = Field(default_factory=FaithfulnessConfig)
    overlap_fractions: list[float] = Field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3]
    )
    output_dir: str = "artifacts/discover"

    @field_validator("overlap_fractions")
    @classmethod
    def _valid_fractions(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("overlap_fractions must not be empty")
        for fraction in value:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"overlap fraction must be in (0, 1], got {fraction}")
        return value


# ---------------------------------------------------------------------------
# Graft
# ---------------------------------------------------------------------------


class GraftConfig(CommonConfig):
    name: str
    selection: Literal["topk", "distributed", "random_matched"] = "topk"
    overlap_strategy: Literal["exclude", "svd_project", "fisher_gate", "uniform"] = "svd_project"
    delta_source: Literal["alignment", "reversal"] = "alignment"
    k_s: float = 0.05
    k_d: float = 0.05
    alpha: float = 0.5
    alpha_o: float = 1.0
    svd_energy: float = 0.9
    svd_max_rank: int = 16
    gamma: float = 1.0
    distributed_rho: float = 2.0
    random_seed: int = 0
    random_reference: str | None = None
    safety_scores: str | None = None
    domain_scores: str | None = None
    fisher_path: str | None = None
    output_dir: str = "artifacts/graft"

    @field_validator("k_s", "k_d")
    @classmethod
    def _fraction(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("component budgets must be fractions in [0, 1]")
        return value

    @field_validator("alpha", "alpha_o", "distributed_rho")
    @classmethod
    def _positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be > 0")
        return value

    @model_validator(mode="after")
    def _check_strategy(self) -> GraftConfig:
        if self.overlap_strategy == "fisher_gate" and not self.fisher_path:
            raise ValueError("fisher_gate requires fisher_path")
        if self.selection == "random_matched" and not self.random_reference:
            raise ValueError("random_matched requires random_reference")
        return self


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class BaselineConfig(CommonConfig):
    method: Literal[
        "resta", "wiseft", "ctheta", "safelora", "lox", "task_arith", "refusal_steer"
    ]
    model: str = "drifted"
    source_circuit: str | None = None
    adapter_path: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    sweep: dict[str, list[float]] = Field(default_factory=dict)
    output_dir: str = "artifacts/baselines"


# ---------------------------------------------------------------------------
# Evaluation and attacks
# ---------------------------------------------------------------------------


class GenerationConfig(StrictModel):
    max_new_tokens_safety: int = 256
    max_new_tokens_mcq: int = 32
    batch_size: int = 8
    temperature: float = 0.0
    do_sample: bool = False
    seed: int = 0
    use_chat_template: bool = True
    include_system_preamble: bool = False
    record_letter_logits: bool = True
    max_prompt_tokens: int = 4096


class JudgeConfig(StrictModel):
    primary: str = "wildguard"
    secondary: str = "string_match"
    guard_model: str = "allenai/wildguard"
    auditkit_guard: str | None = None
    human_audit_per_model: int = 100
    max_new_tokens: int = 4
    allow_cpu: bool = False


class EvalConfig(CommonConfig):
    # Overrides the base model references: these are evaluator matrix names,
    # resolved to checkpoints by the stage pipeline.
    models: list[str] = Field(
        default_factory=lambda: ["aligned", "drifted", "cg_project", "cg_uniform", "resta"]
    )
    benchmarks: list[str] = Field(
        default_factory=lambda: [
            "harmbench",
            "advbench",
            "sorrybench",
            "domain_harmful",
            "xstest_safe",
            "orbench_hard",
            "domain_benign",
            "medmcqa_validation",
            "medqa_test",
            "pubmedqa_labeled",
            "mmlu_non_medical",
        ]
    )
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    bootstrap_resamples: int = 1000
    bootstrap_ci: float = 0.95
    data_dir: str = "artifacts/data"
    generations_dir: str = "results/generations"
    output_dir: str = "artifacts/eval"
    audit_output_dir: str = "artifacts/audit"
    # Explicit logical name -> local checkpoint / HF id overrides. Staged
    # outputs are discovered from the artifacts tree when an entry is absent.
    model_paths: dict[str, str] = Field(default_factory=dict)
    max_rows: dict[str, int] = Field(default_factory=dict)
    skip_existing: bool = True
    dgr_min_gain: float = 0.03


class AbliterationConfig(StrictModel):
    modes: list[str] = Field(default_factory=lambda: ["weight_orthogonalize", "runtime_ablate"])
    fit_harmful: int = 200
    fit_harmless: int = 200


class HarmfulFtConfig(StrictModel):
    steps: list[int] = Field(default_factory=lambda: [50, 100, 200])
    lr: float = 1e-4
    examples: int = 100


class QuantConfig(StrictModel):
    backends: list[str] = Field(default_factory=lambda: ["nf4", "int8"])
    circuit_aware: bool = True
    bits: int = 4
    high_fraction: float = 0.3


class AttacksConfig(CommonConfig):
    types: list[str] = Field(default_factory=lambda: ["abliterate", "harmful_ft", "quantize"])
    models: list[str] = Field(
        default_factory=lambda: ["aligned", "drifted", "resta", "cg_project", "cg_distributed"]
    )
    abliteration: AbliterationConfig = Field(default_factory=AbliterationConfig)
    harmful_ft: HarmfulFtConfig = Field(default_factory=HarmfulFtConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    output_dir: str = "artifacts/attacks"
