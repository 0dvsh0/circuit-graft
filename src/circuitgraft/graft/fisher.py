"""Domain Fisher importance estimates for the ``fisher_gate`` overlap strategy."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from circuitgraft.components import Cohere2Layout


def compute_fisher(
    model: torch.nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    loss_fn: Callable[[torch.nn.Module, Mapping[str, torch.Tensor]], torch.Tensor],
    layout: Cohere2Layout,
    device: str | torch.device = "cpu",
    total_examples: int | None = None,
) -> dict[str, torch.Tensor]:
    """Mean squared gradient of the domain loss over ``batches``.

    Returns one fp32 tensor per editable weight. The estimate follows the usual
    diagonal-Fisher approximation: square the per-batch gradient, weight by the
    batch size, and divide by the number of examples. This is a domain-adapted
    importance map, not the empirical Fisher of the base model.
    """
    model.to(device)
    editable = layout._editable_name_set()
    accum: dict[str, torch.Tensor] = {}
    seen = 0
    for batch in batches:
        model.zero_grad(set_to_none=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = loss_fn(model, batch)
        loss.backward()
        batch_size = int(next(iter(batch.values())).shape[0])
        seen += batch_size
        for name, parameter in model.named_parameters():
            if name not in editable:
                continue
            if parameter.grad is None:
                continue
            squared = parameter.grad.detach().float().square() * batch_size
            if name in accum:
                accum[name] += squared
            else:
                accum[name] = squared.clone()
        model.zero_grad(set_to_none=True)

    if not accum:
        raise RuntimeError("no gradients were collected for editable tensors")
    divisor = total_examples if total_examples is not None else max(seen, 1)
    return {name: value / divisor for name, value in accum.items()}


def save_fisher(path: str | Path, fisher: Mapping[str, torch.Tensor], metadata: dict[str, str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in fisher.items()}, str(path), metadata=metadata)


def load_fisher(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu")
