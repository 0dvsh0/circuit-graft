"""Shared synthetic fixtures for CPU tests.

The tiny Cohere2 layout keeps the real architectural relationships (GQA,
parallel attention/MLP, tied embeddings) while staying small enough that the
full component partition can be checked in milliseconds.
"""

from __future__ import annotations

import pytest

from circuitgraft.components import Cohere2Layout


@pytest.fixture
def layout() -> Cohere2Layout:
    return Cohere2Layout(
        num_layers=2,
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
    )
