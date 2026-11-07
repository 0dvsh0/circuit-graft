"""S7 generation pass: one greedy rollout per benchmark item, cached on disk.

Generation is deliberately separated from judging so judges can be re-run
without touching a GPU. Every file records the model fingerprint, prompt hash,
decoding parameters and (for MCQ) the first-step choice logits used as a
parser fallback.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuitgraft.config import EvalConfig, config_hash
from circuitgraft.eval.models import generation_model_info, safe_name
from circuitgraft.registry import (
    REPO_ROOT,
    RunTimer,
    dump_json,
    load_json,
    new_manifest,
    sha256_bytes,
    sha256_file,
)

logger = logging.getLogger(__name__)

MCQ_BENCHMARKS = frozenset(
    {"medmcqa_validation", "medqa_test", "pubmedqa_labeled", "mmlu_non_medical"}
)
SAFETY_BENCHMARKS = frozenset(
    {"harmbench", "advbench", "sorrybench", "wildjailbreak", "domain_harmful"}
)
BENIGN_BENCHMARKS = frozenset({"xstest_safe", "orbench_hard", "domain_benign"})

_FALLBACK_SYSTEM = "You are a helpful assistant."


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def benchmark_kind(benchmark: str, rows: Sequence[Mapping[str, Any]] = ()) -> str:
    if benchmark in MCQ_BENCHMARKS:
        return "mcq"
    if benchmark in SAFETY_BENCHMARKS:
        return "safety"
    if benchmark in BENIGN_BENCHMARKS:
        return "benign"
    if rows and rows[0].get("choices"):
        return "mcq"
    if rows and rows[0].get("label") == "harmful":
        return "safety"
    return "benign"


def load_benchmark_rows(path: str | Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows is not None and len(rows) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not str(item.get("prompt") or "").strip():
                continue
            if not item.get("id"):
                item["id"] = f"{path.stem}-{index}"
            rows.append(item)
    if not rows:
        raise RuntimeError(f"{path}: no usable benchmark rows")
    return rows


def format_prompt(tokenizer: Any, prompt: str, cfg: EvalConfig) -> tuple[str, bool]:
    """Return (rendered prompt, used_chat_template)."""
    generation = cfg.generation
    if not generation.use_chat_template:
        return prompt, False
    messages: list[dict[str, str]] = []
    if generation.include_system_preamble:
        messages.append({"role": "system", "content": _FALLBACK_SYSTEM})
    messages.append({"role": "user", "content": prompt})
    if getattr(tokenizer, "chat_template", None):
        return (
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            True,
        )
    # Base tokenizers have no template. Keep the fallback identical to S2 so a
    # base-tokenizer specialist is prompted the way it was trained.
    rendered = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}" for message in messages
    )
    return rendered + "\nAssistant:\n", False


def _encode_no_special(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def choice_token_ids(tokenizer: Any, choices: Sequence[str]) -> dict[str, list[int]]:
    """First-token ids for each choice, both with and without a leading space."""
    result: dict[str, list[int]] = {}
    for choice in choices:
        if not isinstance(choice, str) or not choice:
            continue
        ids: list[int] = []
        for variant in (f" {choice}", choice):
            encoded = _encode_no_special(tokenizer, variant)
            if encoded and encoded[0] not in ids:
                ids.append(encoded[0])
        if ids:
            result[choice] = ids
    return result


def _choice_logits(
    scores: Any,
    tokenizer: Any,
    choices: Sequence[str],
    batch_index: int,
) -> dict[str, float] | None:
    if scores is None or not choices:
        return None
    try:
        first_step = scores[0][batch_index]
    except (IndexError, TypeError):
        return None
    token_ids = choice_token_ids(tokenizer, choices)
    if not token_ids:
        return None
    values: dict[str, float] = {}
    for choice, ids in token_ids.items():
        values[choice] = max(float(first_step[token]) for token in ids)
    return values


def _argmax_choice(logits: Mapping[str, float] | None) -> str | None:
    if not logits:
        return None
    return max(logits, key=logits.get)


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:  # noqa: BLE001
        return "cpu"


def _eos_ids(tokenizer: Any) -> list[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [int(value)]


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def generation_signature(
    cfg: EvalConfig,
    *,
    benchmark_path: Path,
    model_ref: str,
    max_rows: int | None,
) -> dict[str, Any]:
    payload = {
        "generation": cfg.generation.model_dump(mode="json"),
        "benchmark": str(benchmark_path.relative_to(REPO_ROOT))
        if benchmark_path.is_relative_to(REPO_ROOT)
        else str(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "model": generation_model_info(model_ref),
        "max_rows": max_rows,
    }
    payload["signature"] = config_hash(payload)
    return payload


def cache_status(
    cfg: EvalConfig,
    *,
    model_name: str,
    benchmark: str,
    signature: Mapping[str, Any],
    expected_rows: int,
) -> tuple[bool, Path, Path]:
    model_dir = _repo_path(cfg.generations_dir) / safe_name(model_name)
    output_path = model_dir / f"{benchmark}.jsonl"
    cache_path = model_dir / ".cache" / f"{benchmark}.json"
    current = False
    if cfg.skip_existing and output_path.exists() and cache_path.exists():
        try:
            cache = load_json(cache_path)
            current = (
                cache.get("signature") == signature.get("signature")
                and _line_count(output_path) == expected_rows
            )
        except Exception:  # noqa: BLE001 - a corrupt cache just triggers a rerun
            current = False
    return current, output_path, cache_path


def generate_benchmark(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    benchmark: str,
    cfg: EvalConfig,
    *,
    model_name: str,
    model_ref: str,
    benchmark_path: Path,
) -> dict[str, Any]:
    """Generate one benchmark and write its cached JSONL + manifest."""
    import torch

    kind = benchmark_kind(benchmark, rows)
    max_rows = cfg.max_rows.get(benchmark)
    if max_rows is not None:
        rows = list(rows)[:max_rows]
    if not rows:
        raise RuntimeError(f"{benchmark}: no rows to generate")

    signature_payload = generation_signature(
        cfg, benchmark_path=benchmark_path, model_ref=model_ref, max_rows=max_rows
    )
    current, output_path, cache_path = cache_status(
        cfg,
        model_name=model_name,
        benchmark=benchmark,
        signature=signature_payload,
        expected_rows=len(rows),
    )
    if current:
        logger.info("generation cache hit: %s/%s", model_name, benchmark)
        return {
            "benchmark": benchmark,
            "model": model_name,
            "path": str(output_path),
            "rows": len(rows),
            "skipped": True,
            "signature": signature_payload["signature"],
        }

    generation = cfg.generation
    max_new_tokens = (
        generation.max_new_tokens_mcq
        if kind == "mcq"
        else generation.max_new_tokens_safety
    )
    use_scores = kind == "mcq" and generation.record_letter_logits
    device = _model_device(model)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model_info = generation_model_info(model_ref)
    created_at = datetime.now(UTC).isoformat()

    output_rows: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), generation.batch_size):
                batch = list(rows[start : start + generation.batch_size])
                rendered = [format_prompt(tokenizer, str(row["prompt"]), cfg) for row in batch]
                prompts = [text for text, _ in rendered]
                used_template = any(flag for _, flag in rendered)
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=generation.max_prompt_tokens,
                    add_special_tokens=not used_template,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": bool(generation.do_sample),
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if generation.do_sample:
                    kwargs["temperature"] = generation.temperature
                if use_scores:
                    kwargs.update(
                        {
                            "return_dict_in_generate": True,
                            "output_scores": True,
                        }
                    )
                outputs = model.generate(**encoded, **kwargs)
                if hasattr(outputs, "sequences"):
                    sequences = outputs.sequences
                    scores = getattr(outputs, "scores", None)
                else:
                    sequences = outputs
                    scores = None

                prompt_length = int(encoded["input_ids"].shape[-1])
                new_tokens = sequences[:, prompt_length:]
                responses = tokenizer.batch_decode(
                    new_tokens, skip_special_tokens=True
                )
                eos_tokens = _eos_ids(tokenizer)
                for index, row in enumerate(batch):
                    response = str(responses[index]).strip()
                    token_ids = new_tokens[index].tolist()
                    finished = bool(token_ids) and (
                        not eos_tokens or token_ids[-1] in eos_tokens
                    )
                    choices = row.get("choices")
                    logits = (
                        _choice_logits(scores, tokenizer, choices, index)
                        if use_scores and choices
                        else None
                    )
                    output_rows.append(
                        {
                            "id": str(row.get("id")),
                            "benchmark": benchmark,
                            "kind": kind,
                            "model": model_name,
                            "model_ref": model_ref,
                            "model_fingerprint": model_info["fingerprint"],
                            "prompt": row.get("prompt"),
                            "prompt_sha256": sha256_bytes(
                                str(row.get("prompt", "")).encode("utf-8")
                            ),
                            "response": response,
                            "n_new_tokens": len(token_ids),
                            "finish_reason": "eos" if finished else "length",
                            "choice_logits": logits,
                            "logit_answer": _argmax_choice(logits),
                            "seed": generation.seed,
                            "do_sample": bool(generation.do_sample),
                            "temperature": generation.temperature,
                            "max_new_tokens": max_new_tokens,
                            "created_at": created_at,
                        }
                    )
    finally:
        tokenizer.padding_side = old_padding_side

    _write_jsonl_atomic(output_path, output_rows)
    dump_json(
        cache_path,
        {
            **signature_payload,
            "rows": len(output_rows),
            "path": str(output_path),
            "created_at": created_at,
        },
    )

    manifest_dir = _repo_path(cfg.generations_dir) / safe_name(model_name) / "manifests" / benchmark
    manifest = new_manifest(
        "eval_generate",
        f"{safe_name(model_name)}-{benchmark}",
        str(signature_payload["signature"]),
        seed=generation.seed,
        notes=(
            f"model={model_name}; benchmark={benchmark}; rows={len(output_rows)}; "
            f"max_new_tokens={max_new_tokens}; chat_template={used_template}"
        ),
    )
    manifest.inputs = {
        "benchmark_sha256": signature_payload["benchmark_sha256"],
        "model_fingerprint": model_info["fingerprint"],
        "signature": str(signature_payload["signature"]),
    }
    manifest.outputs = {
        str(output_path.relative_to(REPO_ROOT))
        if output_path.is_relative_to(REPO_ROOT)
        else str(output_path): sha256_file(output_path)
    }
    with RunTimer(manifest, manifest_dir):
        pass

    return {
        "benchmark": benchmark,
        "model": model_name,
        "path": str(output_path),
        "rows": len(output_rows),
        "skipped": False,
        "signature": signature_payload["signature"],
        "kind": kind,
        "manifest": str(manifest_dir / "manifest.json"),
        "chat_template": used_template,
    }
