"""Bootstrap confidence intervals, paired tests and multiple-comparison control."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats as scipy_stats


def _as_1d(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"expected a 1D sequence, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    return array


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    array = _as_1d(values)
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    estimates = np.array([statistic(array[index]) for index in indices])
    alpha = (1.0 - ci) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def paired_bootstrap_ci(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """CI for mean(a - b) with paired resampling of item indices."""
    a_arr, b_arr = _as_1d(a), _as_1d(b)
    if a_arr.size != b_arr.size:
        raise ValueError("paired samples must have equal length")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, a_arr.size, size=(n_resamples, a_arr.size))
    differences = a_arr[indices] - b_arr[indices]
    estimates = differences.mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def mcnemar_test(
    a: Sequence[bool] | Sequence[int] | np.ndarray,
    b: Sequence[bool] | Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Exact McNemar test on paired binary outcomes.

    ``b`` counts items where method A succeeds and B fails; ``c`` the reverse.
    Returns the discordant counts and the two-sided exact binomial p-value.
    """
    a_arr = np.asarray(a, dtype=bool)
    b_arr = np.asarray(b, dtype=bool)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired outcomes must have equal shape")
    b_count = int(np.sum(a_arr & ~b_arr))
    c_count = int(np.sum(~a_arr & b_arr))
    discordant = b_count + c_count
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(scipy_stats.binomtest(min(b_count, c_count), discordant, 0.5).pvalue)
    return {
        "a_success_b_fail": b_count,
        "a_fail_b_success": c_count,
        "discordant": discordant,
        "statistic": min(b_count, c_count),
        "p_value": p_value,
        "p_value_method": "exact_binomial_two_sided",
    }


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return []
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be in [0, 1]")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        value = (p.size - rank) * p[index]
        running = max(running, value)
        adjusted[index] = min(running, 1.0)
    return [float(v) for v in adjusted]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate (useful for judge agreement)."""
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def cohens_kappa(labels_a: Sequence[bool], labels_b: Sequence[bool]) -> float:
    a = np.asarray(labels_a, dtype=bool)
    b = np.asarray(labels_b, dtype=bool)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("kappa needs two non-empty equal-length label sets")
    observed = float(np.mean(a == b))
    p_a, p_b = float(np.mean(a)), float(np.mean(b))
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)
