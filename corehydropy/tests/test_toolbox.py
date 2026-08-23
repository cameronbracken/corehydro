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
    debye,
    dft,
    gauss_jordan,
    histogram,
    interpolate,
    interpolate_2d,
    joint_probability,
    l_moments,
    link,
    link_derivative,
    link_function,
    link_inverse,
    link_names,
    linear_regression,
    percentile,
    polynomial_eval,
    product_moments,
    qr_decomposition,
    qr_solve,
    ranks,
    running_covariance,
    running_statistics,
    sobol_sequence,
    stratify,
    summary_statistics,
    trend_names,
    trend_parameters,
    trend_predict,
)
from corehydropy.models import trend


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


def test_interpolate_cubic_spline_reproduces_the_csharp_test_cubicspline_oracle():
    x = [6, 24, 48, 72]
    y = [9.96, 22.13, 32.27, 37.60]
    out = interpolate(x, y, [8], method="cubic_spline")
    np.testing.assert_allclose(out, [11.4049889205445], atol=1e-6)


def test_interpolate_polynomial_order3_reproduces_the_csharp_test_polynomial_oracle():
    x = [6, 24, 48, 72]
    y = [9.96, 22.13, 32.27, 37.60]
    out = interpolate(x, y, [8], method="polynomial", order=3)
    np.testing.assert_allclose(out, [11.5415808882467], atol=1e-6)


def test_interpolate_requires_order_for_method_polynomial():
    with pytest.raises(ValueError, match="order"):
        interpolate([1, 2, 3, 4], [10, 20, 30, 40], [1.5], method="polynomial")


def test_interpolate_rejects_order_for_a_non_polynomial_method():
    with pytest.raises(ValueError, match="order"):
        interpolate([1, 2, 3, 4], [10, 20, 30, 40], [1.5], order=3)


def test_interpolate_rejects_a_non_default_transform_or_extrapolate_for_a_non_linear_method():
    with pytest.raises(ValueError, match="linear-only"):
        interpolate([1, 2, 3, 4], [10, 20, 30, 40], [1.5], method="cubic_spline",
                    x_transform="log")
    with pytest.raises(ValueError, match="linear-only"):
        interpolate([1, 2, 3, 4], [10, 20, 30, 40], [1.5], method="polynomial", order=3,
                    extrapolate=True)


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


# The "link" and "trend" toolbox groups (Task 7). The oracle-pinned values (from
# Test_LinkFunctions.cs and friends, and the ten Test_*TrendTests.cs files) are validated
# cross-language by fixtures/toolbox/{link_functions,trend_functions}.json; these tests exercise
# the Python-facing API (construction, round-trips, argument checks) instead of re-pinning
# literals.


def test_link_function_round_trips_for_every_type():
    cases = [
        ("Identity", {}),
        ("Log", {}),
        ("Logit", {}),
        ("Probit", {}),
        ("ComplementaryLogLog", {}),
        ("FisherZ", {}),
        ("YeoJohnson", {"lambda": 0.5}),
        ("ASinH", {"gamma0": 0.5, "scale": 0.3}),
        ("SES", {"a": 1.0}),
        ("LogSES", {"sigma0": 7.5}),
        ("LogASinH", {"sigma0": 10.0, "log_scale": 0.25}),
    ]
    domain = {
        "Log": 2.0, "Logit": 0.3, "Probit": 0.4, "ComplementaryLogLog": 0.4,
        "FisherZ": 0.3, "LogSES": 2.0, "LogASinH": 15.0,
    }
    for type_, params in cases:
        l = link_function(type_, **params)
        x = domain.get(type_, 1.0)
        eta = link(l, x)
        back = link_inverse(l, eta)
        assert back[0] == pytest.approx(x, abs=1e-10)
        d = link_derivative(l, x)
        assert np.isfinite(d[0])


def test_link_function_centered_round_trips_with_a_non_identity_inner():
    inner = link_function("ASinH", gamma0=0.0, scale=1.0)
    l = link_function("Centered", mu0=100.0, scale=20.0, inner=inner)
    eta = link(l, 120.0)
    back = link_inverse(l, eta)
    assert back[0] == pytest.approx(120.0, abs=1e-8)


def test_link_function_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="Nope"):
        link_function("Nope")


def test_link_function_matches_type_case_insensitively_and_normalizes_it():
    l = link_function("log")
    assert l.type == "Log"
    l2 = link_function("YEOJOHNSON", lambda_=0.5)
    assert l2.type == "YeoJohnson"


def test_link_function_accepts_lambda_as_a_literal_keyword():
    # `lambda` is a reserved word in Python, so `link_function("YeoJohnson", lambda=0.5)` is a
    # SyntaxError; `lambda_` is the documented spelling (matching stats.box_cox/yeo_johnson).
    # Regression test: the original shipped test only exercised `**{"lambda": 0.5}`, which hid
    # that literal keyword syntax was uncallable.
    l = link_function("YeoJohnson", lambda_=0.5)
    assert l.parameters == {"lambda": 0.5}
    eta = link(l, 2.0)
    back = link_inverse(l, eta)
    assert back[0] == pytest.approx(2.0, abs=1e-10)


def test_link_function_still_accepts_lambda_via_dict_unpacking():
    l = link_function("YeoJohnson", **{"lambda": 0.5})
    assert l.parameters == {"lambda": 0.5}


def test_link_function_rejects_both_lambda_spellings_together():
    with pytest.raises(ValueError, match="lambda"):
        link_function("YeoJohnson", lambda_=0.5, **{"lambda": 0.5})


def test_link_function_inner_is_keyword_only():
    with pytest.raises(TypeError):
        link_function("Log", link_function("Identity"))  # noqa: no positional `inner`


def test_link_function_centered_requires_inner():
    with pytest.raises(ValueError, match="inner"):
        link_function("Centered", mu0=0.0)


def test_link_function_rejects_inner_on_a_non_centered_type():
    with pytest.raises(ValueError, match="inner"):
        link_function("Log", inner=link_function("Identity"))


def test_link_function_inner_must_be_a_link():
    with pytest.raises(TypeError, match="Link"):
        link_function("Centered", mu0=0.0, inner="not a link")


def test_link_rejects_a_non_link_object():
    with pytest.raises(TypeError, match="Link"):
        link("not a link", 1.0)


def test_link_rejects_empty_input():
    l = link_function("Identity")
    with pytest.raises(ValueError, match="non-empty"):
        link(l, [])


def test_link_names_lists_twelve_types():
    names = link_names()
    assert len(names) == 12
    assert "Centered" in names and "YeoJohnson" in names


def test_link_names_and_link_function_cannot_drift_every_listed_type_constructs():
    extra = {
        "YeoJohnson": {"lambda_": 0.5}, "SES": {"a": 1.0},
        "Centered": {"inner": link_function("Identity")},
    }
    for type_ in link_names():
        l = link_function(type_, **extra.get(type_, {}))
        assert l.type == type_


def test_link_repr():
    assert repr(link_function("Log")) == "<Link Log>"


def test_trend_predict_matches_the_transcribed_linear_trend_oracle():
    tr = trend("location", "Linear", start_index=1950, values=[100.0, 0.5])
    out = trend_predict(tr, [1951, 1961, 1941])
    np.testing.assert_allclose(out, [100.0, 105.0, 95.0], atol=1e-10)


def test_trend_predict_with_no_explicit_values_uses_the_zero_valued_class_default():
    tr = trend("location", "Constant")
    out = trend_predict(tr, [1, 2, 3])
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0])


def test_trend_parameters_returns_named_values():
    tr = trend("location", "Linear", values=[10.0, 2.0])
    params = trend_parameters(tr)
    assert list(params.values()) == pytest.approx([10.0, 2.0])
    assert len(params) == 2


def test_trend_predict_rejects_a_non_trend_object():
    with pytest.raises(TypeError, match="Trend"):
        trend_predict("not a trend", [1])


def test_trend_predict_rejects_fractional_indices():
    tr = trend("location", "Constant")
    with pytest.raises(ValueError, match="whole numbers"):
        trend_predict(tr, [1.5])


def test_trend_names_lists_eleven_types():
    names = trend_names()
    assert len(names) == 11
    assert "GeneralLinear" in names


def test_trend_names_and_trend_cannot_drift_every_listed_type_constructs():
    for type_ in trend_names():
        tr = trend("location", type_)
        assert tr.type == type_


# The "linalg" toolbox group (P2 "math extras" Task 9).


def test_qr_decomposition_reproduces_a():
    a = [[1, 1, 1], [0, 2, 5], [2, 5, -1]]
    qr = qr_decomposition(a)
    assert qr["q"].shape == (3, 3)
    assert qr["r"].shape == (3, 3)
    np.testing.assert_allclose(qr["q"] @ qr["r"], a, atol=1e-10)


def test_qr_solve_reproduces_the_test_qrdecomposition_square_system_expected_vector():
    # Test_QRDecomposition.cs's Test_SolveVector: A = [[1,1,1],[0,2,5],[2,5,-1]],
    # b = [6,-4,27]; the real system solves to x = [5, 3, -2].
    a = [[1, 1, 1], [0, 2, 5], [2, 5, -1]]
    b = [6, -4, 27]
    x = qr_solve(a, b)
    np.testing.assert_allclose(x, [5.0, 3.0, -2.0], atol=1e-9)
    np.testing.assert_allclose(np.asarray(a) @ x, b, atol=1e-9)


def test_qr_solve_matrix_rhs_matches_the_vector_rhs():
    a = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    b_vec = [1, 2, 3]
    b_mat = [[1], [2], [3]]
    x_vec = qr_solve(a, b_vec)
    x_mat = qr_solve(a, b_mat)
    assert x_mat.shape == (3, 1)
    np.testing.assert_allclose(x_mat[:, 0], x_vec, atol=1e-10)


def test_qr_solve_underdetermined_leaves_the_trailing_unknown_at_zero():
    a = [[2, 3, 5, 1], [1, 0, 2, 3], [0, 1, 4, 2]]
    b = [1, 2, 3]
    x = qr_solve(a, b)
    assert x.shape == (4,)
    assert x[3] == 0.0
    np.testing.assert_allclose(np.asarray(a) @ x, b, atol=1e-8)


def test_qr_solve_rejects_a_mismatched_b_length():
    a = [[1, 0], [0, 1]]
    with pytest.raises(ValueError, match="rows"):
        qr_solve(a, [1, 2, 3])


def test_gauss_jordan_reproduces_true_ia_exactly():
    # Test_GaussJordanElimination.cs's Test_GaussJordanElim: A = [[1,3,3],[1,4,3],[1,3,4]],
    # true_IA = [[7,-3,-3],[-1,1,0],[-1,0,1]].
    a = [[1, 3, 3], [1, 4, 3], [1, 3, 4]]
    result = gauss_jordan(a)
    np.testing.assert_array_equal(result["inverse"], [[7, -3, -3], [-1, 1, 0], [-1, 0, 1]])
    assert result["solution"].shape == (3, 0)


def test_gauss_jordan_solution_solves_ax_equals_b():
    a = [[1, 3, 3], [1, 4, 3], [1, 3, 4]]
    b = [[1], [0], [0]]
    result = gauss_jordan(a, b)
    np.testing.assert_allclose(np.asarray(a) @ result["solution"], b, atol=1e-10)


def test_gauss_jordan_rejects_a_non_square_a():
    with pytest.raises(ValueError, match="square"):
        gauss_jordan([[1, 2, 3], [4, 5, 6]])


# The "special" toolbox group (P2 "math extras" Task 10).


def test_debye_reproduces_the_test_debye_literal_at_x_1_0():
    # Test_SpecialFunctions.cs's Test_Debye: testX[1] = 1.0, testValid[1] = 0.6744156.
    np.testing.assert_allclose(debye(1.0), [0.6744156], atol=1e-4)


def test_debye_is_vectorized_over_x_reproducing_the_whole_test_debye_array():
    x = [0.1, 1.0, 2.8, 9.5, 10, 15, 25, 100]
    valid = [0.9629999, 0.6744156, 0.3099952, 0.02241066, 0.01929577, 0.005771263,
             0.001246836, 1.948182e-05]
    np.testing.assert_allclose(debye(x), valid, atol=1e-4)


def test_debye_rejects_a_negative_x():
    with pytest.raises(Exception):
        debye(-1)


def test_polynomial_eval_variant_standard_reproduces_test_polynomial():
    # Test_SpecialFunctions.cs's Test_Polynomial: coeffs = [3, 5, 7], x = 4, valid = 135.
    np.testing.assert_allclose(polynomial_eval([3, 5, 7], 4), [135])
    np.testing.assert_allclose(polynomial_eval([3, 5, 7], 4, variant="standard"), [135])


def test_polynomial_eval_variant_reverse_reproduces_test_polynomial_rev_with_and_without_n():
    np.testing.assert_allclose(polynomial_eval([3, 5, 7], 4, variant="reverse"), [75])
    np.testing.assert_allclose(polynomial_eval([3, 5, 7], 4, variant="reverse", n=1), [17])


def test_polynomial_eval_variant_reverse_unit_reproduces_test_polynomial_rev_1():
    np.testing.assert_allclose(polynomial_eval([3, 5, 7], 4, variant="reverse_unit"), [139])


def test_polynomial_eval_is_vectorized_over_x():
    np.testing.assert_allclose(polynomial_eval([3, 5, 7], [0, 4]), [3, 135])


def test_polynomial_eval_rejects_n_with_a_non_reverse_variant():
    with pytest.raises(ValueError, match="reverse"):
        polynomial_eval([3, 5, 7], 4, variant="standard", n=1)
    with pytest.raises(ValueError, match="reverse"):
        polynomial_eval([3, 5, 7], 4, variant="reverse_unit", n=1)
