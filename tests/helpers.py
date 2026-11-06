"""Synthetic model/config builders shared by the test modules."""

from __future__ import annotations

import torch

from circuitgraft.components import Cohere2Layout
from circuitgraft.config import GraftConfig


def editable_shapes(layout: Cohere2Layout) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for layer in range(layout.num_layers):
        shapes[layout.attn_tensor(layer, "q_proj")] = (
            layout.num_heads * layout.head_dim,
            layout.hidden_size,
        )
        shapes[layout.attn_tensor(layer, "k_proj")] = (
            layout.num_kv_heads * layout.head_dim,
            layout.hidden_size,
        )
        shapes[layout.attn_tensor(layer, "v_proj")] = (
            layout.num_kv_heads * layout.head_dim,
            layout.hidden_size,
        )
        shapes[layout.attn_tensor(layer, "o_proj")] = (
            layout.hidden_size,
            layout.num_heads * layout.head_dim,
        )
        shapes[layout.mlp_tensor(layer, "gate_proj")] = (
            layout.intermediate_size,
            layout.hidden_size,
        )
        shapes[layout.mlp_tensor(layer, "up_proj")] = (
            layout.intermediate_size,
            layout.hidden_size,
        )
        shapes[layout.mlp_tensor(layer, "down_proj")] = (
            layout.hidden_size,
            layout.intermediate_size,
        )
    return shapes


def make_state(layout: Cohere2Layout, *, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    state = {
        name: torch.randn(shape, generator=generator, dtype=torch.float32)
        for name, shape in editable_shapes(layout).items()
    }
    # Non-editable tensors are present so tests can prove they are untouched.
    state[f"{layout.prefix}.embed_tokens.weight"] = torch.randn(
        128, layout.hidden_size, generator=generator
    )
    state["lm_head.weight"] = torch.randn(128, layout.hidden_size, generator=generator)
    state[f"{layout.prefix}.norm.weight"] = torch.randn(
        layout.hidden_size, generator=generator
    )
    for layer in range(layout.num_layers):
        state[f"{layout.layer_prefix(layer)}.input_layernorm.weight"] = torch.randn(
            layout.hidden_size, generator=generator
        )
    return state


def make_graft_config(**overrides: object) -> GraftConfig:
    values: dict[str, object] = {
        "name": "test",
        "selection": "topk",
        "overlap_strategy": "uniform",
        "delta_source": "alignment",
        "k_s": 1.0,
        "k_d": 0.0,
        "alpha": 0.5,
        "alpha_o": 1.0,
        "output_dir": "artifacts/test",
    }
    values.update(overrides)
    return GraftConfig(**values)  # type: ignore[arg-type]


def full_scores(layout: Cohere2Layout, value: float = 1.0) -> dict[str, float]:
    return {component.id: value for component in layout.all_components()}
