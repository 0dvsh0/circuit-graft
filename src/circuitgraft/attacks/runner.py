"""S8 attacks runner: orchestrate abliteration, harmful fine-tuning, and quantization."""

from __future__ import annotations

import logging
from typing import Any

from circuitgraft.attacks.abliterate import run_abliteration
from circuitgraft.attacks.harmful_ft import run_harmful_ft
from circuitgraft.attacks.quantize import run_quantization
from circuitgraft.config import AttacksConfig, BaseConfig

logger = logging.getLogger(__name__)


def run_attack(
    cfg: AttacksConfig,
    base: BaseConfig,
    attack_type: str = "abliterate",
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute the requested attack type or all attacks."""
    attack_type = attack_type.lower()
    results: dict[str, Any] = {}

    if attack_type in ("abliterate", "all"):
        logger.info("running abliteration attack...")
        results["abliterate"] = run_abliteration(cfg, base, **kwargs)

    if attack_type in ("harmful_ft", "all"):
        logger.info("running harmful fine-tuning attack...")
        results["harmful_ft"] = run_harmful_ft(cfg, base, **kwargs)

    if attack_type in ("quantize", "quant", "all"):
        logger.info("running quantization attack...")
        results["quantize"] = run_quantization(cfg, base, **kwargs)

    if attack_type not in ("abliterate", "harmful_ft", "quantize", "quant", "all"):
        raise ValueError(
            f"unknown attack type: {attack_type!r}; "
            f"expected one of: 'abliterate', 'harmful_ft', 'quantize', 'all'"
        )

    return results
