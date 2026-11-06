"""SVD protection, Fisher gating and Fisher estimation tests."""

from __future__ import annotations

import pytest
import torch

from circuitgraft.components import Cohere2Layout
from circuitgraft.graft.fisher import compute_fisher, load_fisher, save_fisher
from circuitgraft.graft.protect import (
    exclude,
    fisher_gate,
    protect,
    svd_project,
    uniform,
)


def test_svd_project_removes_domain_subspace() -> None:
    generator = torch.Generator().manual_seed(0)
    delta = torch.randn(16, 8, generator=generator)
    left = torch.randn(16, 1, generator=generator)
    right = torch.randn(1, 8, generator=generator)
    domain_delta = left @ right  # rank 1

    projected = svd_project(delta, domain_delta, energy=0.9, max_rank=16)
    _, _, vh = torch.linalg.svd(domain_delta, full_matrices=False)

    # Every domain input direction is removed from the safety edit.
    assert projected.shape == delta.shape
    assert torch.allclose(projected @ vh[0], torch.zeros(16), atol=1e-5)
    assert projected.norm() < delta.norm()


def test_svd_project_zero_domain_delta_is_identity() -> None:
    delta = torch.randn(8, 8, generator=torch.Generator().manual_seed(1))
    assert torch.equal(svd_project(delta, torch.zeros_like(delta)), delta)


def test_svd_project_validates_inputs() -> None:
    delta = torch.randn(8, 8)
    with pytest.raises(ValueError, match="shapes differ"):
        svd_project(delta, torch.randn(8, 4))
    with pytest.raises(ValueError, match="energy"):
        svd_project(delta, torch.randn(8, 8), energy=0.0)
    with pytest.raises(ValueError, match="max_rank"):
        svd_project(delta, torch.randn(8, 8), max_rank=0)
    with pytest.raises(ValueError, match="2D"):
        svd_project(torch.randn(8), torch.randn(8))


def test_fisher_gate_with_zero_fisher_is_identity() -> None:
    delta = torch.randn(8, 8, generator=torch.Generator().manual_seed(2))
    gated = fisher_gate(delta, torch.zeros_like(delta))
    assert torch.equal(gated, delta)


def test_fisher_gate_with_constant_nonzero_fisher_is_identity() -> None:
    """A constant Fisher has no per-weight signal once normalised."""
    delta = torch.randn(8, 8, generator=torch.Generator().manual_seed(3))
    gated = fisher_gate(delta, torch.full_like(delta, 7.0))
    assert torch.equal(gated, delta)


def test_fisher_gate_applies_per_weight_decay() -> None:
    delta = torch.ones(2, 2)
    fisher = torch.tensor([[0.0, 1.0], [0.5, 0.25]])
    gated = fisher_gate(delta, fisher, gamma=1.0, normalize=False)
    assert torch.equal(gated, torch.tensor([[1.0, 0.0], [0.5, 0.75]]))

    squared = fisher_gate(delta, fisher, gamma=2.0, normalize=False)
    assert torch.allclose(squared, (1.0 - fisher) ** 2)


def test_fisher_gate_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        fisher_gate(torch.ones(2, 2), torch.ones(2))
    with pytest.raises(ValueError, match="gamma"):
        fisher_gate(torch.ones(2, 2), torch.ones(2, 2), gamma=0.0)


def test_protect_dispatch_and_missing_arguments() -> None:
    delta = torch.randn(4, 4, generator=torch.Generator().manual_seed(4))
    assert torch.equal(protect("uniform", delta), delta)
    assert torch.equal(protect("exclude", delta), torch.zeros_like(delta))
    assert torch.equal(uniform(delta), delta)
    assert torch.equal(exclude(delta), torch.zeros_like(delta))
    with pytest.raises(ValueError, match="domain_delta"):
        protect("svd_project", delta)
    with pytest.raises(ValueError, match="Fisher"):
        protect("fisher_gate", delta)
    with pytest.raises(ValueError, match="unknown"):
        protect("nope", delta)  # type: ignore[arg-type]


def _tiny_module(layout: Cohere2Layout) -> torch.nn.Module:
    root = torch.nn.Module()
    model = torch.nn.Module()
    root.add_module("model", model)
    layers = torch.nn.ModuleList()
    model.add_module("layers", layers)
    for _ in range(layout.num_layers):
        layer = torch.nn.Module()
        layers.append(layer)
        attn = torch.nn.Module()
        layer.add_module("self_attn", attn)
        mlp = torch.nn.Module()
        layer.add_module("mlp", mlp)
        shapes = {
            "q_proj": (layout.num_heads * layout.head_dim, layout.hidden_size),
            "k_proj": (layout.num_kv_heads * layout.head_dim, layout.hidden_size),
            "v_proj": (layout.num_kv_heads * layout.head_dim, layout.hidden_size),
            "o_proj": (layout.hidden_size, layout.num_heads * layout.head_dim),
        }
        for name, (out_features, in_features) in shapes.items():
            attn.add_module(name, torch.nn.Linear(in_features, out_features, bias=False))
        mlp_shapes = {
            "gate_proj": (layout.intermediate_size, layout.hidden_size),
            "up_proj": (layout.intermediate_size, layout.hidden_size),
            "down_proj": (layout.hidden_size, layout.intermediate_size),
        }
        for name, (out_features, in_features) in mlp_shapes.items():
            mlp.add_module(name, torch.nn.Linear(in_features, out_features, bias=False))
    return root


def test_compute_fisher_matches_squared_gradients(layout: Cohere2Layout) -> None:
    model = _tiny_module(layout)
    target = layout.attn_tensor(0, "q_proj")

    def loss_fn(module: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parameter = dict(module.named_parameters())[target]
        return (parameter.square().sum()) * batch["scale"].mean()

    fisher = compute_fisher(
        model,
        [{"scale": torch.ones(4, 1)}],
        loss_fn=loss_fn,
        layout=layout,
        total_examples=4,
    )

    assert set(fisher) == {target}
    parameter = dict(model.named_parameters())[target]
    expected = (2.0 * parameter.detach()) ** 2
    assert torch.allclose(fisher[target], expected, rtol=1e-5)


def test_compute_fisher_requires_gradients(layout: Cohere2Layout) -> None:
    model = _tiny_module(layout)

    def no_grad(module: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.tensor(1.0, requires_grad=True)

    with pytest.raises(RuntimeError, match="no gradients"):
        compute_fisher(model, [{"scale": torch.ones(2, 1)}], loss_fn=no_grad, layout=layout)


def test_fisher_round_trip(tmp_path, layout: Cohere2Layout) -> None:
    path = tmp_path / "fisher.safetensors"
    values = {
        layout.attn_tensor(0, "q_proj"): torch.rand(8, 32),
        layout.mlp_tensor(1, "down_proj"): torch.rand(32, 64),
    }
    save_fisher(path, values, metadata={"model": "test"})
    loaded = load_fisher(path)
    assert set(loaded) == set(values)
    for name in values:
        assert torch.equal(loaded[name], values[name])
