"""CircuitKIT discovery runner (S4).

The runner builds a contrastive task directly from the canonical pair JSONL
(``clean_prompt`` / ``corrupted_prompt`` / ``answer`` / ``corrupted_answer``),
loads a Cohere2 checkpoint through CircuitKIT's TransformerLens backend, runs
node-level EAP-IG, and writes canonical ``L{l}.H{h}`` / ``L{l}.MLP`` scores.

CircuitKIT is the primary discovery path. SafeTune EAP can be run on the same
pairs as a cross-check; both write canonical scores, so the graft stage and the
C-ΔΘ baseline consume identical circuit files.
"""

from __future__ import annotations

import gc
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuitgraft.components import Cohere2Layout
from circuitgraft.config import (
    BaseConfig,
    DiscoverConfig,
    FaithfulnessConfig,
    config_hash,
)
from circuitgraft.discover.overlap import (
    circuit_info_payload,
    convert_scores,
    jaccard,
    read_score_payload,
    write_overlap_report,
    write_scores,
)
from circuitgraft.discover.safetune_eap import discover_safety_circuit
from circuitgraft.graft.graft import complete_scores
from circuitgraft.graft.pipeline import load_layout
from circuitgraft.graft.select import topk
from circuitgraft.registry import (
    REPO_ROOT,
    RunTimer,
    dump_json,
    load_json,
    new_manifest,
    sha256_file,
    should_skip,
)

logger = logging.getLogger(__name__)

_TASK_SCHEMA = {
    "prompt": "clean_prompt",
    "answer": "answer",
    "corrupted": "corrupted_prompt",
    "corrupted_answer": "corrupted_answer",
}

_UNSAFE_TASK_CHARS = re.compile(r"[^a-z0-9_]+")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _model_ref(base: BaseConfig, logical: str) -> str:
    if logical == "aligned":
        return base.models.aligned
    if logical == "drifted":
        if not base.models.drifted:
            raise ValueError(
                "models.drifted is empty; run `cg drift` first or set models.drifted "
                "to the merged specialist checkpoint"
            )
        return base.models.drifted
    raise ValueError(f"unknown model role: {logical!r}")


def _model_input_hash(label: str, ref: str) -> tuple[str, str]:
    path = Path(str(ref))
    candidate = path if path.is_absolute() else REPO_ROOT / path
    if candidate.is_file():
        return label, sha256_file(candidate)
    if candidate.is_dir() and (candidate / "config.json").exists():
        return label, (
            f"local-dir:{candidate.resolve()}:{sha256_file(candidate / 'config.json')}"
        )
    return label, f"ref:{ref}"


def _input_hashes(cfg: DiscoverConfig, base: BaseConfig) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for label, raw_path in (("pairs:safety", cfg.safety.pairs), ("pairs:domain", cfg.domain.pairs)):
        path = _repo_path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
        hashes[label] = sha256_file(path)
    for role in (cfg.safety.model, cfg.domain.model):
        ref = _model_ref(base, role)
        label, digest = _model_input_hash(f"model:{role}", ref)
        hashes[label] = digest
    return hashes


def _resolve_device(device: str | None) -> str:
    import torch

    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str, device: str) -> str:
    if device == "cpu" and dtype != "float32":
        logger.warning(
            "CircuitKIT discovery on CPU with dtype=%s is unsupported; using float32",
            dtype,
        )
        return "float32"
    return dtype


def build_pairs_task(pairs_path: str | Path, *, name_prefix: str = "cg") -> Any:
    """Register a CircuitKIT ``GenericTaskSpec`` from a canonical pair file.

    The task name includes the pair-file hash, so repeated runs in one process
    reuse the same spec and a changed file cannot silently reuse a stale task.
    """
    from circuitkit import register_task
    from circuitkit.tasks import GenericTaskSpec
    from circuitkit.tasks.registry import get_task, is_task_registered

    path = Path(pairs_path)
    if not path.exists():
        raise FileNotFoundError(f"pair file does not exist: {path}")
    digest = sha256_file(path)
    stem = _UNSAFE_TASK_CHARS.sub("_", path.stem.lower()).strip("_") or "pairs"
    name = f"{name_prefix}_{stem}_{digest[:12]}"

    if is_task_registered(name):
        return get_task(name)

    spec = GenericTaskSpec.from_jsonl(
        path=str(path),
        schema=dict(_TASK_SCHEMA),
        name=name,
        task_type="classification",
    )
    register_task(spec)
    return spec


def canonical_scores(circuit: Any, *, layout: Cohere2Layout) -> dict[str, float]:
    """Convert a CircuitKIT ``Circuit`` score map to CircuitGraft ids."""
    raw = getattr(circuit, "scores", None)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(
            "CircuitKIT returned no node scores; discovery cannot be consumed by "
            "CircuitGraft or C-Delta-Theta"
        )
    return convert_scores(raw, source="circuitkit", layout=layout)


def _selected_nodes(circuit: Any) -> list[str]:
    nodes = getattr(circuit, "nodes", None)
    if isinstance(nodes, Mapping):
        flattened: list[str] = []
        for value in nodes.values():
            if isinstance(value, (list, tuple)):
                flattened.extend(str(item) for item in value)
        return flattened
    if not isinstance(nodes, (list, tuple)):
        raise ValueError(f"unsupported CircuitKIT node artifact: {type(nodes).__name__}")
    return [str(node) for node in nodes]


def patch_transformer_lens_weight_processing() -> None:
    """Monkey-patch TransformerLens ProcessWeights.process_weights.

    TransformerLens upcasts the entire state_dict to float32 even when all
    folding/centering flags are False (the default for CircuitKIT). On a 3.5B
    model, this creates a 14 GB spike that triggers CUDA OutOfMemory on 15 GB
    GPUs or Linux OOM killer (-9) on systems with limited CPU RAM.
    """
    try:
        import transformer_lens.weight_processing as wp
    except ImportError:
        return

    if getattr(wp.ProcessWeights, "_cg_patched", False):
        return

    orig_process = wp.ProcessWeights.process_weights

    def _fast_process_weights(
        state_dict: dict[str, Any],
        cfg: Any,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
        adapter: Any = None,
    ) -> dict[str, Any]:
        if not any(
            [
                fold_ln,
                center_writing_weights,
                center_unembed,
                fold_value_biases,
                refactor_factored_attn_matrices,
            ]
        ):
            return state_dict
        return orig_process(
            state_dict,
            cfg,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            adapter=adapter,
        )

    wp.ProcessWeights.process_weights = staticmethod(_fast_process_weights)
    wp.ProcessWeights._cg_patched = True


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cleanup must not mask a real failure
        pass


def _apply_memory_controls(model: Any, *, max_seq_len: int) -> None:
    """Bound activation memory for CircuitKIT EAP-IG discovery.

    The D1 smoke run fit Tiny Aya on a 15 GB T4 by capping the context and
    freezing non-embedding parameters (see ``docs/EXPERIMENT_LOG.md``).
    EAP-IG differentiates through activations, so parameter gradients are not
    needed. Applying the same controls to the full runner keeps a 1024-token
    context from requesting ~27 GB of activation storage.
    """
    cfg = getattr(model, "cfg", None)
    current_ctx = getattr(cfg, "n_ctx", None) if cfg is not None else None
    if isinstance(current_ctx, int) and max_seq_len:
        cfg.n_ctx = min(current_ctx, max(int(max_seq_len), 128))

    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        for name, param in named_parameters():
            if not (name.startswith("embed") or name.startswith("pos_embed")):
                param.requires_grad = False

    gc.collect()
    _empty_cuda_cache()


def run_circuitkit_discovery(
    model_ref: str | Path,
    pairs_path: str | Path,
    output_dir: str | Path,
    *,
    layout: Cohere2Layout,
    kind: str,
    algorithm: str = "eap-ig",
    level: str = "node",
    n_examples: int = 256,
    sparsity: float = 0.3,
    ig_steps: int = 5,
    batch_size: int = 4,
    max_seq_len: int = 1024,
    device: str | None = None,
    dtype: str = "bfloat16",
    seed: int = 0,
    faithfulness: FaithfulnessConfig | None = None,
) -> dict[str, Any]:
    """Discover one circuit and write canonical scores, mask and diagnostics."""
    if level != "node":
        raise ValueError(
            f"only node-level discovery is supported, got {level!r}; "
            "the component vocabulary is heads + MLPs"
        )

    import circuitkit as ck

    patch_transformer_lens_weight_processing()

    import torch
    import transformers

    if not getattr(transformers.AutoModelForCausalLM, "_cg_patched", False):
        orig_hf_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

        def _fast_hf_from_pretrained(*hf_args, **hf_kwargs):
            if "torch_dtype" not in hf_kwargs:
                hf_kwargs["torch_dtype"] = torch.bfloat16
            return orig_hf_from_pretrained(*hf_args, **hf_kwargs)

        transformers.AutoModelForCausalLM.from_pretrained = staticmethod(_fast_hf_from_pretrained)
        transformers.AutoModelForCausalLM._cg_patched = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_device = _resolve_device(device)
    effective_dtype = _resolve_dtype(dtype, effective_device)

    task = build_pairs_task(pairs_path)
    logger.info(
        "loading %s for %s discovery on %s (%s)",
        model_ref,
        kind,
        effective_device,
        effective_dtype,
    )
    if effective_device == "cuda" and effective_dtype == "bfloat16":
        torch.set_default_dtype(torch.bfloat16)
    try:
        model = ck.load_model(
            str(model_ref),
            dtype=effective_dtype,
            device=effective_device,
            algorithm=algorithm,
        )
    finally:
        torch.set_default_dtype(torch.float32)
    _apply_memory_controls(model, max_seq_len=max_seq_len)
    logger.info(
        "discovery memory controls: n_ctx=%s, batch_size=%s, max_seq_len=%s",
        getattr(getattr(model, "cfg", None), "n_ctx", None),
        batch_size,
        max_seq_len,
    )
    artifact_path = output_dir / "circuit.pt"
    try:
        circuit = ck.discover(
            model,
            task.name,
            algorithm=algorithm,
            level=level,
            n_examples=n_examples,
            batch_size=batch_size,
            sparsity=sparsity,
            scope="both",
            output_path=str(artifact_path),
            ig_steps=ig_steps,
            max_seq_len=max_seq_len,
            intervention="patching",
            seed=seed,
        )
        selected = _selected_nodes(circuit)

        raw_scores = getattr(circuit, "scores", None)
        if not isinstance(raw_scores, Mapping) or not raw_scores:
            sidecar = output_dir / "circuit_scores.json"
            raw_scores = read_score_payload(sidecar) if sidecar.exists() else {}
        if not raw_scores:
            raise ValueError(
                "CircuitKIT produced no scores; refusing to write an empty circuit"
            )
        scores = convert_scores(raw_scores, source="circuitkit", layout=layout)
        scores_path = write_scores(
            output_dir / "scores.json",
            scores,
            kind=kind,
            model=str(model_ref),
            source="circuitkit",
            metadata={
                "algorithm": algorithm,
                "level": level,
                "task": task.name,
                "n_examples": n_examples,
                "sparsity": sparsity,
                "ig_steps": ig_steps,
                "selected_nodes": len(selected),
                "dtype": effective_dtype,
                "device": effective_device,
            },
        )
        # SafeTune's CircuitKIT adapter reads this bridge; C-Θ and CircuitGraft
        # therefore consume the same discovery run. ``scores_file`` is relative
        # to this directory so the lineage check is portable.
        circuit_info_path = dump_json(
            output_dir / "circuit_info.json",
            circuit_info_payload(
                scores,
                kind=kind,
                model=str(model_ref),
                source="circuitkit",
                scores_file=scores_path.name,
                scores_sha256=sha256_file(scores_path),
            ),
        )
        selection_path = dump_json(
            output_dir / "selected.json",
            {
                "kind": kind,
                "level": level,
                "algorithm": algorithm,
                "sparsity": sparsity,
                "nodes": selected,
                "components": [
                    component.id
                    for component in topk(
                        complete_scores(layout, scores)[0], fraction=sparsity
                    )
                ],
            },
        )

        faithful_path: Path | None = None
        faithfulness_error: str | None = None
        if faithfulness is not None and faithfulness.enabled:
            try:
                report = ck.faithfulness(
                    model,
                    circuit,
                    task.name,
                    pillars=list(faithfulness.methods),
                    n_examples=faithfulness.n_examples,
                    batch_size=batch_size,
                    device=effective_device,
                    seed=seed,
                )
                faithful_path = output_dir / "faithfulness.json"
                report.to_json(faithful_path)
            except Exception as exc:  # noqa: BLE001 - diagnostic, not a result
                faithfulness_error = f"{type(exc).__name__}: {exc}"
                logger.exception("faithfulness failed for %s circuit", kind)
        else:
            faithfulness_error = None
    finally:
        del model
        gc.collect()
        _empty_cuda_cache()

    return {
        "kind": kind,
        "model": str(model_ref),
        "task": task.name,
        "artifact_path": str(artifact_path),
        "scores_path": str(scores_path),
        "circuit_info_path": str(circuit_info_path),
        "selection_path": str(selection_path),
        "num_components": len(scores),
        "num_selected_nodes": len(selected),
        "faithfulness_path": str(faithful_path) if faithful_path else None,
        "faithfulness_error": faithfulness_error,
        "scores": scores,
        "selected": selected,
        "dtype": effective_dtype,
        "device": effective_device,
    }


def _cross_check_payload(
    circuitkit_result: Mapping[str, Any],
    safetune_result: Mapping[str, Any],
    *,
    layout: Cohere2Layout,
    fraction: float,
) -> dict[str, Any]:
    ck_scores, _ = complete_scores(layout, circuitkit_result["scores"])
    st_scores, _ = complete_scores(layout, safetune_result["scores"])
    ck_selected = topk(ck_scores, fraction=fraction)
    st_selected = topk(st_scores, fraction=fraction)
    return {
        "fraction": fraction,
        "jaccard": jaccard(ck_selected, st_selected),
        "circuitkit": {
            "count": len(ck_selected),
            "components": [component.id for component in ck_selected],
            "scores_path": circuitkit_result["scores_path"],
        },
        "safetune": {
            "count": len(st_selected),
            "components": [component.id for component in st_selected],
            "scores_path": safetune_result["scores_path"],
        },
        "note": (
            "CircuitKIT is node-level (per-head); SafeTune EAP is also run at "
            "head granularity when possible. Agreement is reported at the same "
            "component budget."
        ),
    }


def run_safetune_cross_check(
    model_ref: str,
    pairs_path: str | Path,
    output_dir: str | Path,
    *,
    layout: Cohere2Layout,
    cfg: DiscoverConfig,
    device: str | None,
    dtype: str,
) -> dict[str, Any]:
    """Run SafeTune EAP on the safety pairs and compare with CircuitKIT."""
    refusal = cfg.safety.refusal_tokens[0] if cfg.safety.refusal_tokens else None
    compliance = cfg.safety.compliance_tokens[0] if cfg.safety.compliance_tokens else None
    # CircuitKIT calls the per-head level "node"; SafeTune EAP calls the same
    # thing "head" and does not accept "node" as a granularity value.
    safetune_granularity = "block" if cfg.safety.granularity == "block" else "head"
    result = discover_safety_circuit(
        model_ref,
        pairs_path,
        output_dir,
        layout=layout,
        method=cfg.cross_check_method,
        granularity=safetune_granularity,
        top_k_edges=cfg.safety.top_k_edges,
        ig_steps=cfg.safety.ig_steps,
        max_seq_len=cfg.safety.max_seq_len,
        batch_size=cfg.safety.batch_size,
        metric=cfg.safety.metric,
        refusal_token=refusal,
        compliance_token=compliance,
        device=device,
        dtype=dtype,
    )
    gc.collect()
    _empty_cuda_cache()
    return result


def run_discovery(
    cfg: DiscoverConfig,
    base: BaseConfig,
    *,
    layout: Cohere2Layout | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """S4: discover S and D, write overlap/cross-check/faithfulness artifacts."""
    cfg_hash = config_hash(cfg)
    run_id = f"discover-{cfg_hash}"
    out_dir = _repo_path(cfg.output_dir) / run_id
    input_hashes = _input_hashes(cfg, base)
    summary_path = out_dir / "summary.json"

    if not overwrite and should_skip(out_dir, cfg_hash, input_hashes) and summary_path.exists():
        logger.info("discovery run is current; skipping %s", out_dir)
        return {
            "run_id": run_id,
            "output_dir": str(out_dir),
            "skipped": True,
            "summary": load_json(summary_path),
        }

    if layout is None:
        layout = load_layout(base.models.base)

    safety_model = _model_ref(base, cfg.safety.model)
    domain_model = _model_ref(base, cfg.domain.model)
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype, device)

    manifest = new_manifest("discover", run_id, cfg_hash, seed=cfg.seed)
    manifest.inputs = input_hashes
    manifest.notes = (
        f"safety_model={cfg.safety.model}; domain_model={cfg.domain.model}; "
        f"algorithm={cfg.safety.method}; cross_check={cfg.cross_check}; "
        f"device={device}; dtype={dtype}"
    )

    with RunTimer(manifest, out_dir):
        safety = run_circuitkit_discovery(
            safety_model,
            _repo_path(cfg.safety.pairs),
            out_dir / "safety",
            layout=layout,
            kind="safety",
            algorithm=cfg.safety.method,
            level=cfg.safety.granularity,
            n_examples=cfg.safety.n_examples,
            sparsity=cfg.safety.sparsity,
            ig_steps=cfg.safety.ig_steps,
            batch_size=cfg.safety.batch_size,
            max_seq_len=cfg.safety.max_seq_len,
            device=device,
            dtype=dtype,
            seed=cfg.seed,
            faithfulness=cfg.faithfulness,
        )
        domain = run_circuitkit_discovery(
            domain_model,
            _repo_path(cfg.domain.pairs),
            out_dir / "domain",
            layout=layout,
            kind="domain",
            algorithm=cfg.domain.method,
            level=cfg.domain.granularity,
            n_examples=cfg.domain.n_examples,
            sparsity=cfg.domain.sparsity,
            ig_steps=cfg.domain.ig_steps,
            batch_size=cfg.domain.batch_size,
            max_seq_len=cfg.domain.max_seq_len,
            device=device,
            dtype=dtype,
            seed=cfg.seed,
            faithfulness=cfg.faithfulness,
        )

        overlap_paths: dict[str, str] = {}
        for fraction in cfg.overlap_fractions:
            path = out_dir / f"overlap_{fraction:g}.json"
            write_overlap_report(
                path,
                safety["scores"],
                domain["scores"],
                k_s=fraction,
                k_d=fraction,
                layout=layout,
                method=f"topk@{fraction:g}",
            )
            overlap_paths[f"{fraction:g}"] = str(path)

        overlap_path = out_dir / "overlap.json"
        write_overlap_report(
            overlap_path,
            safety["scores"],
            domain["scores"],
            k_s=cfg.safety.sparsity,
            k_d=cfg.domain.sparsity,
            layout=layout,
            method="discovery_sparsity",
        )

        cross_check: dict[str, Any] | None = None
        cross_check_path: Path | None = None
        if cfg.cross_check:
            safetune = run_safetune_cross_check(
                safety_model,
                _repo_path(cfg.safety.pairs),
                out_dir / "safetune_safety",
                layout=layout,
                cfg=cfg,
                device=device,
                dtype=dtype,
            )
            cross_check = _cross_check_payload(
                safety,
                safetune,
                layout=layout,
                fraction=cfg.safety.sparsity,
            )
            cross_check_path = dump_json(out_dir / "cross_check.json", cross_check)

        summary = {
            "run_id": run_id,
            "config_hash": cfg_hash,
            "safety": {key: value for key, value in safety.items() if key != "scores"},
            "domain": {key: value for key, value in domain.items() if key != "scores"},
            "overlap": str(overlap_path),
            "overlap_by_fraction": overlap_paths,
            "cross_check": None if cross_check is None else str(cross_check_path),
            "cross_check_jaccard": (
                None if cross_check is None else cross_check["jaccard"]
            ),
        }
        dump_json(summary_path, summary)
        outputs = {
            "safety/scores.json": sha256_file(safety["scores_path"]),
            "safety/circuit_info.json": sha256_file(safety["circuit_info_path"]),
            "domain/scores.json": sha256_file(domain["scores_path"]),
            "domain/circuit_info.json": sha256_file(domain["circuit_info_path"]),
            "overlap.json": sha256_file(overlap_path),
            "summary.json": sha256_file(summary_path),
        }
        for fraction, path in overlap_paths.items():
            outputs[f"overlap_{fraction}.json"] = sha256_file(path)
        if cross_check_path is not None:
            outputs["cross_check.json"] = sha256_file(cross_check_path)
        for result in (safety, domain):
            if result["faithfulness_path"]:
                relative = Path(result["faithfulness_path"]).relative_to(out_dir)
                outputs[str(relative)] = sha256_file(result["faithfulness_path"])
        manifest.outputs = outputs

    return {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "skipped": False,
        "summary": summary,
    }
