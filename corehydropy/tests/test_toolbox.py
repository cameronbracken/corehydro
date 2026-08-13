# Binding-specific behaviour for the statistics/spectra toolbox surface. Oracle-value checks
# live in test_fixtures.py, driven by the shared language-neutral fixtures. Mirrors corehydror's
# test-toolbox.R assertion for assertion.
import numpy as np
import pytest

from corehydropy import (
    RunningCovariance,
    RunningStatistics,
    autocorrelation,
    cross_correlation,
    dft,
    l_moments,
    percentile,
    product_moments,
    ranks,
    running_covariance,
    running_statistics,
    summary_statistics,
)


def test_chunked_running_statistics_matches_a_single_call_over_the_concatenated_data():
    chunk1 = [2, 4, 4, 4]
    chunk2 = [5, 5, 7, 9]
    whole = summary_statistics(chunk1 + chunk2)

    s = running_statistics(chunk1)
    s = running_statistics(chunk2, state=s)

    assert s.n == whole["n"]
    assert s.mean == whole["mean"]
    assert s.variance == whole["variance"]
    assert s.skewness == whole["skewness"]
    assert s.kurtosis == whole["kurtosis"]


def test_running_statistics_with_no_state_matches_summary_statistics():
    x = [2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9]
    s = running_statistics(x)
    whole = summary_statistics(x)
    assert s.mean == whole["mean"]
    assert s.sd == whole["sd"]


def test_running_statistics_rejects_a_state_of_the_wrong_type():
    with pytest.raises(TypeError, match="RunningStatistics"):
        running_statistics([1, 2, 3], state={})


def test_running_statistics_repr():
    s = running_statistics([1, 2, 3])
    assert "RunningStatistics" in repr(s)


def test_product_moments_and_l_moments_return_the_expected_keys():
    x = [2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9]
    pm = product_moments(x)
    assert set(pm.keys()) == {"mean", "sd", "skewness", "kurtosis"}
    lm = l_moments(x)
    assert set(lm.keys()) == {"l1", "l2", "t3", "t4"}


def test_ranks_handles_ties_by_averaging_the_tied_ranks():
    np.testing.assert_array_equal(ranks([3, 1, 2, 1]), [4, 1.5, 3, 1.5])


def test_percentile_accepts_a_vector_of_probabilities():
    x = [1, 2, 3, 4, 5]
    p = percentile(x, probs=[0, 0.5, 1])
    assert len(p) == 3
    np.testing.assert_array_equal(p, [1, 3, 5])


def test_running_covariance_chunked_matches_a_single_call():
    x = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    whole = running_covariance(x)

    c1 = running_covariance(x[:2])
    c2 = running_covariance(x[2:], state=c1)

    assert c2.n == whole.n
    np.testing.assert_allclose(c2.mean, whole.mean)
    np.testing.assert_allclose(c2.covariance, whole.covariance, atol=1e-9)


def test_running_covariance_rejects_a_state_of_the_wrong_type():
    with pytest.raises(TypeError, match="RunningCovariance"):
        running_covariance([[1, 2], [3, 4]], state={})


def test_running_covariance_repr():
    c = running_covariance([[1, 2], [3, 4]])
    assert "RunningCovariance" in repr(c)


def test_autocorrelation_at_lag_0_is_1_for_the_correlation_type():
    x = [5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 5, 6, 4, 7, 3, 8, 2, 9, 1, 10]
    a = autocorrelation(x, max_lag=5)
    assert a["lag"][0] == 0
    assert a["value"][0] == pytest.approx(1.0, abs=1e-12)
    assert a["ci"]["lower"] < 0 < a["ci"]["upper"]


def test_autocorrelation_covariance_type_at_lag_0_equals_the_population_variance():
    x = np.array([5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 5, 6, 4, 7, 3, 8, 2, 9, 1, 10], dtype=float)
    a = autocorrelation(x, max_lag=5, type="covariance")
    pop_var = float(np.mean((x - x.mean()) ** 2))
    assert a["value"][0] == pytest.approx(pop_var, abs=1e-9)


def test_autocorrelation_rejects_a_too_short_series():
    with pytest.raises(ValueError, match="x.*at least two"):
        autocorrelation([1])


def test_dft_round_trips_to_within_1e_minus_12():
    x = np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype=float)
    n = len(x) / 2
    z = dft(dft(x, inverse=True))
    np.testing.assert_allclose(z / n, x, atol=1e-12)


def test_cross_correlation_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cross_correlation([1, 2, 3, 4], [1, 2])
