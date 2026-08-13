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
    histogram,
    interpolate,
    interpolate_2d,
    joint_probability,
    l_moments,
    linear_regression,
    percentile,
    product_moments,
    ranks,
    running_covariance,
    running_statistics,
    sobol_sequence,
    stratify,
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


def test_percentile_rejects_a_non_numeric_probs_argument():
    with pytest.raises(ValueError, match="probs"):
        percentile([1, 2, 3], probs="half")


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


def test_running_covariance_resumes_a_one_column_single_variable_accumulator():
    # Parity test for a bug on the R side (a length-1 mean/covariance in the resume state used
    # to serialize as a JSON scalar instead of an array); Python's json.dumps() never had this
    # problem, but the same case should be asserted on both sides.
    x = [[1], [2], [3]]
    whole = running_covariance([[1], [2], [3], [4], [5], [6]])

    c1 = running_covariance(x)
    c2 = running_covariance([[4], [5], [6]], state=c1)

    assert c2.n == whole.n
    np.testing.assert_allclose(c2.mean, whole.mean)
    np.testing.assert_allclose(c2.covariance, whole.covariance, atol=1e-9)


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


def test_histogram_bin_frequencies_sum_to_len_x():
    x = [1, 2, 2.5, 3, 3.5, 4, 5, 7, 8, 9]
    h = histogram(x)
    assert float(h["frequency"].sum()) == len(x)
    assert h["statistics"]["bins"] == len(h["lower"])


def test_histogram_with_an_explicit_bin_count_uses_exactly_that_many_bins():
    x = [1, 2, 2.5, 3, 3.5, 4, 5, 7, 8, 9]
    h = histogram(x, bins=4)
    assert len(h["lower"]) == 4
    assert float(h["frequency"].sum()) == len(x)


def test_interpolate_clamps_to_the_end_knot_without_extrapolate_and_extends_with_it():
    x = [1, 2, 3, 4]
    y = [10, 20, 30, 40]
    clamped = interpolate(x, y, [10])
    np.testing.assert_allclose(clamped, [40])
    extended = interpolate(x, y, [10], extrapolate=True)
    np.testing.assert_allclose(extended, [100])


def test_interpolate_rejects_an_unknown_transform_name_listing_the_accepted_values():
    with pytest.raises(ValueError, match="none.*log.*normal_z"):
        interpolate([1, 2], [1, 2], [1.5], x_transform="bogus")


def test_interpolate_on_a_log_log_grid_reproduces_the_csharp_test_log_oracle():
    x = [50, 100, 150, 200, 250]
    y = [100, 200, 300, 400, 500]
    out = interpolate(x, y, [75], x_transform="log", y_transform="log")
    np.testing.assert_allclose(out, [150.0], atol=1e-6)


def test_interpolate_2d_rejects_a_y_array_whose_shape_does_not_match_x1_by_x2():
    with pytest.raises(ValueError, match=r"3 x 2"):
        interpolate_2d([1, 2, 3], [1, 2], np.arange(4).reshape(2, 2), [1.5], [1.5])


def test_interpolate_2d_reproduces_a_known_bilinear_value_on_an_identity_grid():
    y = np.eye(3)
    out = interpolate_2d([1, 2, 3], [1, 2, 3], y, [1.5], [1.5])
    np.testing.assert_allclose(out, [0.5], atol=1e-12)


# The "regression" toolbox group (Task 5). linear_regression() mirrors the C# LinearRegression
# class; numpy.linalg.lstsq is a genuinely independent check (a different implementation
# entirely), so comparing against it here is allowed by corehydro's fixture-provenance rule even
# though it is not itself a pinned oracle value.


def test_linear_regression_coefficients_match_numpy_lstsq():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(30, 2))
    y = 1.5 + 2 * x[:, 0] - 0.7 * x[:, 1] + rng.normal(scale=0.01, size=30)

    fit = linear_regression(x, y)
    design = np.column_stack([np.ones(30), x])
    ref_coef, *_ = np.linalg.lstsq(design, y, rcond=None)

    np.testing.assert_allclose(fit.coefficients, ref_coef, atol=1e-8)


def test_linear_regression_with_intercept_false_drops_the_intercept_column():
    x = np.column_stack([[1, 2, 3, 4, 5], [2, 1, 4, 3, 5]])
    y = [3.1, 4.2, 8.1, 9.2, 13.0]

    with_int = linear_regression(x, y)
    without_int = linear_regression(x, y, intercept=False)

    assert len(with_int.coefficients) == 3
    assert len(without_int.coefficients) == 2


def test_linear_regression_rejects_a_y_of_the_wrong_length_naming_both_lengths():
    x = np.column_stack([[1, 2, 3, 4, 5], [2, 1, 4, 3, 5]])
    with pytest.raises(ValueError, match=r"5.*3|3.*5"):
        linear_regression(x, [1, 2, 3])


def test_linear_regression_recovers_an_exact_linear_combination_with_r_squared_1():
    x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    x2 = np.array([2, 1, 4, 3, 6, 5, 8, 7], dtype=float)
    y = 3 + 2 * x1 - x2
    fit = linear_regression(np.column_stack([x1, x2]), y)
    np.testing.assert_allclose(fit.coefficients, [3, 2, -1], atol=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)


def test_linear_regression_covariance_is_the_coefficient_covariance():
    # sqrt(diag(covariance)) == standard_errors holds by construction once covariance is
    # properly scaled by sigma**2, so this is a self-consistency check, not an oracle value.
    x = np.column_stack([[1, 2, 3, 4, 5], [2, 1, 4, 3, 5]])
    y = [3.1, 4.2, 8.1, 9.2, 13.0]
    fit = linear_regression(x, y)
    np.testing.assert_allclose(
        np.sqrt(np.diag(fit.covariance)), fit.standard_errors, atol=1e-12
    )


def test_predict_reproduces_the_fitted_values_at_the_training_predictors():
    x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    x2 = np.array([2, 1, 4, 3, 6, 5, 8, 7], dtype=float)
    y = 3 + 2 * x1 - x2
    x = np.column_stack([x1, x2])
    fit = linear_regression(x, y)
    np.testing.assert_allclose(fit.predict(x), y, atol=1e-9)


def test_predict_accepts_a_bare_1d_vector_for_a_single_predictor_model():
    # Matches R's predict.corehydro_lm(): a 1D input on a one-predictor fit is a column of
    # observations, not a single row.
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = [3.1, 4.2, 8.1, 9.2, 13.0]
    fit = linear_regression(x, y)
    out = fit.predict([1, 2, 3])
    assert out.shape == (3,)


def test_predict_with_interval_returns_a_lower_upper_mean_array():
    x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    x2 = np.array([2, 1, 4, 3, 6, 5, 8, 7], dtype=float)
    y = np.array([3.1, 6.2, 4.9, 8.3, 6.8, 10.2, 8.9, 12.1])
    x = np.column_stack([x1, x2])
    fit = linear_regression(x, y)
    out = fit.predict(x, interval=True)
    assert out.shape == (8, 3)
    assert np.all(out[:, 0] <= out[:, 2]) and np.all(out[:, 2] <= out[:, 1])


def test_linear_regression_repr_does_not_error():
    x = np.column_stack([[1, 2, 3, 4, 5], [2, 1, 4, 3, 5]])
    y = [3.1, 4.2, 8.1, 9.2, 13.0]
    fit = linear_regression(x, y)
    assert "LinearRegressionResult" in repr(fit)


def test_predict_rejects_newdata_with_the_wrong_number_of_columns():
    x = np.column_stack([[1, 2, 3, 4, 5], [2, 1, 4, 3, 5]])
    y = [3.1, 4.2, 8.1, 9.2, 13.0]
    fit = linear_regression(x, y)
    with pytest.raises(ValueError, match="2"):
        fit.predict([[1, 2, 3]])


# The "sampling" and "probability" toolbox groups (Task 6). The oracle-pinned values (from
# Test_SobolSequence.cs, Test_Stratification.cs, and Test_Probability.cs) are validated
# cross-language by fixtures/toolbox/{sampling,joint_probability}.json; these tests exercise the
# Python-facing API (argument checks, shapes, self-consistency) instead of re-pinning literals.


def test_sobol_sequence_returns_an_n_by_dimension_array_with_every_value_in_0_1():
    m = sobol_sequence(8, dimension=3)
    assert m.shape == (8, 3)
    assert np.all((m >= 0) & (m < 1))


def test_sobol_sequence_needs_no_direction_numbers_file_at_dimension_1():
    m = sobol_sequence(1)
    assert m[0, 0] == 0.5


def test_sobol_sequence_skip_moves_the_stream():
    seq5 = sobol_sequence(5, dimension=2)
    skipped = sobol_sequence(1, dimension=2, skip=4)
    np.testing.assert_array_equal(skipped[0], seq5[4])


def test_sobol_sequence_rejects_a_non_positive_n_or_dimension():
    with pytest.raises(ValueError, match="n"):
        sobol_sequence(0)
    with pytest.raises(ValueError, match="dimension"):
        sobol_sequence(5, dimension=0)


def test_stratify_returns_bins_whose_weights_sum_to_the_axis_length():
    s = stratify(0, 1, bins=10)
    assert len(s["weight"]) == 10
    assert set(s.keys()) == {"lower", "upper", "midpoint", "weight"}
    assert s["weight"].sum() == pytest.approx(1.0, abs=1e-12)


def test_stratify_with_probability_true_returns_zero_rows():
    s = stratify(0, 1, bins=10, probability=True)
    assert len(s["weight"]) == 0


def test_stratify_rejects_fewer_than_2_bins():
    with pytest.raises(ValueError, match="bins"):
        stratify(0, 1, bins=1)


def test_stratify_rejects_lower_greater_than_or_equal_to_upper():
    with pytest.raises(ValueError, match="lower.*upper"):
        stratify(1, 1, bins=4)
    with pytest.raises(ValueError, match="lower.*upper"):
        stratify(2, 1, bins=4)


def test_joint_probability_independent_multiplies_and_positive_takes_the_minimum():
    assert joint_probability([0.5, 0.5]) == pytest.approx(0.25, abs=1e-12)
    assert joint_probability([0.5, 0.5], dependency="positive") == pytest.approx(0.5, abs=1e-12)
    assert joint_probability([0.5, 0.5], dependency="negative") == pytest.approx(0.0, abs=1e-12)


def test_joint_probability_with_indicators_and_correlation_reaches_the_hpcm_path():
    p = [0.25, 0.35, 0.5, 0.5]
    ind = [1, 1, 1, 1]
    corr = np.eye(4)
    out = joint_probability(p, dependency="correlation", indicators=ind, correlation=corr)
    assert out == pytest.approx(0.021875, abs=1e-6)


def test_joint_probability_requires_indicators_when_correlation_is_given():
    with pytest.raises(ValueError, match="indicators"):
        joint_probability([0.5, 0.5], correlation=np.eye(2))


def test_joint_probability_rejects_a_correlation_matrix_of_the_wrong_size():
    with pytest.raises(ValueError, match="3 x 3"):
        joint_probability([0.5, 0.5, 0.5], indicators=[1, 1, 1], correlation=np.eye(2))


def test_joint_probability_rejects_correlation_dependency_missing_both_arguments():
    with pytest.raises(ValueError, match="indicators.*correlation"):
        joint_probability([0.5, 0.5], dependency="correlation")


def test_joint_probability_rejects_correlation_dependency_missing_just_the_matrix():
    with pytest.raises(ValueError, match="correlation"):
        joint_probability([0.5, 0.5], dependency="correlation", indicators=[1, 1])
