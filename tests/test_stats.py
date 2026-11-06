"""Statistical helpers: bootstrap coverage, paired tests and agreement."""

from __future__ import annotations

import numpy as np
import pytest

from circuitgraft.eval.stats import (
    bootstrap_ci,
    cohens_kappa,
    holm_correction,
    mcnemar_test,
    paired_bootstrap_ci,
    wilson_interval,
)


def test_bootstrap_constant_sample_is_degenerate() -> None:
    lower, upper = bootstrap_ci([5.0] * 20, n_resamples=100, seed=1)
    assert lower == pytest.approx(5.0)
    assert upper == pytest.approx(5.0)


def test_bootstrap_ci_contains_the_sample_mean() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, size=200)
    lower, upper = bootstrap_ci(values, n_resamples=1000, ci=0.9, seed=0)
    assert lower < values.mean() < upper


def test_bootstrap_has_nominal_coverage_for_a_known_mean() -> None:
    """A fixed-seed coverage smoke test: 50 samples, 90% intervals, >= 80% cover."""
    covered = 0
    for dataset in range(50):
        rng = np.random.default_rng(10_000 + dataset)
        values = rng.normal(0.0, 1.0, size=100)
        lower, upper = bootstrap_ci(values, n_resamples=300, ci=0.9, seed=dataset)
        covered += int(lower <= 0.0 <= upper)
    assert covered >= 40, f"only {covered}/50 intervals covered the true mean"


def test_bootstrap_validates_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="1D"):
        bootstrap_ci(np.ones((2, 2)))
    with pytest.raises(ValueError, match="ci"):
        bootstrap_ci([1.0, 2.0], ci=1.0)


def test_paired_bootstrap_zero_difference() -> None:
    lower, upper = paired_bootstrap_ci([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert (lower, upper) == (0.0, 0.0)
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_ci([1.0], [1.0, 2.0])


def test_mcnemar_exact_counts_and_p_value() -> None:
    a = [True, True, False, False]
    b = [False, True, True, False]
    result = mcnemar_test(a, b)
    assert result["a_success_b_fail"] == 1
    assert result["a_fail_b_success"] == 1
    assert result["discordant"] == 2
    assert result["p_value"] == pytest.approx(1.0)

    dominant = mcnemar_test([True] * 10 + [False] * 10, [False] * 10 + [False] * 10)
    assert dominant["a_success_b_fail"] == 10
    assert dominant["a_fail_b_success"] == 0
    assert dominant["p_value"] == pytest.approx(2 * 0.5**10)


def test_mcnemar_no_discordance() -> None:
    result = mcnemar_test([True, False], [True, False])
    assert result["discordant"] == 0
    assert result["p_value"] == 1.0
    with pytest.raises(ValueError, match="equal shape"):
        mcnemar_test([True], [True, False])


def test_holm_correction_preserves_order_and_is_monotone() -> None:
    adjusted = holm_correction([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert holm_correction([]) == []
    with pytest.raises(ValueError, match="p-values"):
        holm_correction([1.5])


def test_wilson_interval_brackets_the_rate() -> None:
    lower, upper = wilson_interval(95, 100)
    assert 0.0 <= lower < 0.95 < upper <= 1.0
    assert lower == pytest.approx(0.889, abs=0.002)
    with pytest.raises(ValueError, match="total"):
        wilson_interval(0, 0)


def test_cohens_kappa_known_cases() -> None:
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohens_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cohens_kappa([], [])
    with pytest.raises(ValueError):
        cohens_kappa([True], [True, False])
