"""Protection strategies applied on safety/domain overlap components."""

from __future__ import annotations

from typing import Literal

import torch

ProtectStrategy = Literal["exclude", "svd_project", "fisher_gate", "uniform"]


def _as_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"protection expects a 2D weight slice, got shape {tuple(tensor.shape)}")
    return tensor


def uniform(delta: torch.Tensor, domain_delta: torch.Tensor | None = None, **_: object) -> torch.Tensor:
    """Apply the safety delta unchanged (C-ΔΘ-like behaviour)."""
    return delta


def exclude(delta: torch.Tensor, domain_delta: torch.Tensor | None = None, **_: object) -> torch.Tensor:
    """Do not edit shared components at all."""
    return torch.zeros_like(delta)


def svd_project(
    delta: torch.Tensor,
    domain_delta: torch.Tensor,
    *,
    energy: float = 0.9,
    max_rank: int = 16,
) -> torch.Tensor:
    """Remove safety-edit components aligned with domain-changed input directions.

    ``Δ_D = U Σ Vᵀ``; keep the top-r right singular vectors covering ``energy``
    of the squared spectrum (capped at ``max_rank``) and project the safety edit
    onto the orthogonal complement: ``Δ (I - V_r V_rᵀ)``.
    """
    delta = _as_2d(delta).float()
    domain_delta = _as_2d(domain_delta).float()
    if delta.shape != domain_delta.shape:
        raise ValueError(
            f"delta and domain_delta shapes differ: {tuple(delta.shape)} vs {tuple(domain_delta.shape)}"
        )
    if not 0.0 < energy <= 1.0:
        raise ValueError("energy must be in (0, 1]")
    if max_rank < 1:
        raise ValueError("max_rank must be >= 1")

    _, singular_values, vh = torch.linalg.svd(domain_delta, full_matrices=False)
    if singular_values.numel() == 0 or float(singular_values[0]) == 0.0:
        # No domain change on this slice: nothing to protect against.
        return delta

    power = singular_values.square()
    cumulative = torch.cumsum(power, dim=0) / torch.sum(power)
    rank = int(torch.searchsorted(cumulative, torch.tensor(energy, device=cumulative.device)).item()) + 1
    rank = min(max(rank, 1), max_rank, vh.shape[0])
    v_r = vh[:rank].transpose(0, 1)  # (in_features, rank)

    return delta - (delta @ v_r) @ v_r.transpose(0, 1)


def fisher_gate(
    delta: torch.Tensor,
    fisher: torch.Tensor,
    *,
    gamma: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Down-weight edits where the domain loss is sensitive.

    ``F`` is normalised per tensor to [0, 1], then ``Δ ⊙ (1 - F̂)^γ``.
    With ``F == 0`` this reduces exactly to ``uniform``. Pass
    ``normalize=False`` when the caller already normalised the full tensor
    (the graft path normalises per tensor, not per slice).
    """
    if delta.shape != fisher.shape:
        raise ValueError(f"delta and fisher shapes differ: {tuple(delta.shape)} vs {tuple(fisher.shape)}")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    delta = delta.float()
    f = fisher.float()
    if normalize:
        flat = f.flatten()
        lo, hi = flat.min(), flat.max()
        if float(hi - lo) == 0.0:
            # Constant Fisher (including all-zero) carries no per-weight signal.
            return delta
        f = ((f - lo) / (hi - lo)).clamp(0.0, 1.0)
    else:
        f = f.clamp(0.0, 1.0)
    return delta * (1.0 - f).pow(gamma)


def protect(
    strategy: ProtectStrategy,
    delta: torch.Tensor,
    *,
    domain_delta: torch.Tensor | None = None,
    fisher: torch.Tensor | None = None,
    energy: float = 0.9,
    max_rank: int = 16,
    gamma: float = 1.0,
    normalize_fisher: bool = True,
) -> torch.Tensor:
    if strategy == "uniform":
        return uniform(delta)
    if strategy == "exclude":
        return exclude(delta)
    if strategy == "svd_project":
        if domain_delta is None:
            raise ValueError("svd_project requires domain_delta")
        return svd_project(delta, domain_delta, energy=energy, max_rank=max_rank)
    if strategy == "fisher_gate":
        if fisher is None:
            raise ValueError("fisher_gate requires a Fisher tensor")
        return fisher_gate(delta, fisher, gamma=gamma, normalize=normalize_fisher)
    raise ValueError(f"unknown protection strategy: {strategy!r}")
