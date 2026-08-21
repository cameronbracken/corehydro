"""Smoke tests for the exported user-facing analysis functions (A10).

These prove the univariate_analysis / fit_distributions / bulletin17c_analysis wrappers are
callable end-to-end and return the documented dict shapes with finite / monotone invariants.
Exact oracle values live in fixtures/ and are checked by test_fixtures.py; these assert shape.
"""

from __future__ import annotations

import math

import pytest

import corehydropy

SMOKE_PEAKS = [
    12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
    19200, 13800, 25600, 10500, 16900, 21300, 14700, 8200, 23800, 15900,
]


def test_univariate_analysis_returns_frequency_curve():
    res = corehydropy.univariate_analysis(
        SMOKE_PEAKS,
        distribution="Normal",
        sampler="DEMCzs",
        iterations=100,
        output_length=400,
        credible_level=0.90,
        seed=12345,
    )
    assert len(res["parameters"]) == 2
    assert all(math.isfinite(v) for v in res["parameters"])
    assert math.isfinite(res["aic"])
    assert math.isfinite(res["dic"])
    n = len(res["mode_curve"])
    assert n > 0
    assert len(res["mean_curve"]) == n
    assert len(res["lower_ci"]) == n
    assert len(res["upper_ci"]) == n
    # Ascending exceedance ordinates -> descending mode (quantile) curve.
    assert all(b <= a + 1e-6 for a, b in zip(res["mode_curve"], res["mode_curve"][1:]))
    # Each credible band brackets the mode curve.
    assert all(lo <= m + 1e-6 for lo, m in zip(res["lower_ci"], res["mode_curve"]))
    assert all(hi >= m - 1e-6 for hi, m in zip(res["upper_ci"], res["mode_curve"]))


def test_fit_distributions_ranks_15_candidates():
    res = corehydropy.fit_distributions(SMOKE_PEAKS)
    assert len(res["distribution"]) == 15
    assert len(res["aic"]) == 15
    assert any(res["converged"])
    for i, ok in enumerate(res["converged"]):
        if ok:
            assert math.isfinite(res["aic"][i])
            assert math.isfinite(res["bic"][i])
            assert math.isfinite(res["rmse"][i])


def test_bulletin17c_analysis_returns_cohn_intervals():
    res = corehydropy.bulletin17c_analysis(
        SMOKE_PEAKS,
        uncertainty_method="MultivariateNormal",
        output_length=200,
        seed=12345,
        confidence_level=0.90,
    )
    assert res["confidence_level"] == 0.90
    n = len(res["exceedance_probabilities"])
    assert n > 0
    assert len(res["point_estimates"]) == n
    assert len(res["lower_ci"]) == n
    assert len(res["upper_ci"]) == n
    assert all(math.isfinite(v) for v in res["point_estimates"])
    assert all(lo <= hi for lo, hi in zip(res["lower_ci"], res["upper_ci"]))
    assert len(res["parameters"]) == 3  # LP3: location / scale / shape


def test_bulletin17c_analysis_accepts_x8_x9_methods():
    # X8/X9 un-gated LinkedMultivariateNormal + BiasCorrectedBootstrap; they now run (X11 wired the
    # binding knob). An unknown method still raises.
    for um in ("LinkedMultivariateNormal", "BiasCorrectedBootstrap"):
        res = corehydropy.bulletin17c_analysis(
            SMOKE_PEAKS, uncertainty_method=um, output_length=200, seed=12345
        )
        assert len(res["parameters"]) == 3
        assert all(lo <= hi for lo, hi in zip(res["lower_ci"], res["upper_ci"]))
    with pytest.raises(Exception):
        corehydropy.bulletin17c_analysis(SMOKE_PEAKS, uncertainty_method="NotAMethod")


# --- D5: per-family analyses + diagnostics ----------------------------------------------
# Direct user-API smoke calls (the fixture harness covers the numeric oracles); these assert
# the returned dict shapes and finite/well-shaped contents in Python.

TS_SERIES = [
    10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1,
    11.7, 12.5, 13.9, 15.2, 14.1, 12.9, 13.6, 15.0, 16.2, 14.8,
]

BIMODAL = [
    520, 580, 610, 650, 700, 730, 760, 800, 850, 880, 910, 950, 990, 1030, 1080,
    5000, 5400, 5800, 6300, 6800,
]

CR_SERIES = [
    7872, 8624, 5894, 12540, 4322, 17586, 8307, 6009, 13320, 10641, 6301, 8458,
    3545, 8838, 13628, 15105, 11742, 15763, 9117, 4116, 12372, 5902, 6038, 8381, 14452,
]

POT_PEAKS = [950, 1020, 1130, 980, 1250, 1090, 1430, 1010, 1180, 1620, 970, 1300, 1050, 1550, 1210]


def _assert_family_shape(res, n_params):
    assert len(res["parameters"]) == n_params
    assert all(math.isfinite(v) for v in res["parameters"])
    n = len(res["mode_curve"])
    assert n > 0
    assert len(res["mean_curve"]) == n
    assert len(res["lower_ci"]) == n
    assert len(res["upper_ci"]) == n
    assert math.isfinite(res["aic"])
    assert math.isfinite(res["bic"])


def test_mixture_analysis_returns_frequency_curve():
    res = corehydropy.mixture_analysis(
        BIMODAL, ["Normal", "Normal"], iterations=100, output_length=400, seed=12345,
        thinning_interval=1, exceedance_probabilities=[0.01, 0.1, 0.5, 0.9, 0.99],
    )
    _assert_family_shape(res, 6)
    assert len(res["mode_curve"]) == 5


def test_competing_risk_analysis_returns_frequency_curve():
    res = corehydropy.competing_risk_analysis(
        CR_SERIES, ["Gumbel", "Weibull"], iterations=100, output_length=400, seed=12345,
        thinning_interval=1, exceedance_probabilities=[0.01, 0.1, 0.5, 0.9, 0.99],
    )
    _assert_family_shape(res, 4)


def test_point_process_analysis_returns_frequency_curve():
    res = corehydropy.point_process_analysis(
        POT_PEAKS, threshold=900, total_years=20, iterations=100, output_length=400,
        seed=12345, thinning_interval=1, exceedance_probabilities=[0.01, 0.1, 0.5, 0.9, 0.99],
    )
    _assert_family_shape(res, 3)
    assert math.isfinite(res["dic"])


def test_ar_analysis_returns_forecast_curve():
    res = corehydropy.ar_analysis(
        TS_SERIES, order_p=1, training_time_steps=15, forecasting_time_steps=3,
        iterations=100, output_length=400, seed=12345, thinning_interval=1,
    )
    assert len(res["mode_curve"]) == 23  # 20 observed + 3 forecast
    assert all(math.isfinite(v) for v in res["parameters"])
    assert math.isfinite(res["rmse"])


def test_ma_analysis_returns_forecast_curve():
    res = corehydropy.ma_analysis(
        TS_SERIES, order_q=1, training_time_steps=15, forecasting_time_steps=3,
        iterations=100, output_length=400, seed=12345, thinning_interval=1,
    )
    assert len(res["mode_curve"]) == 23
    assert math.isfinite(res["aic"])


def test_arima_analysis_returns_forecast_curve():
    res = corehydropy.arima_analysis(
        TS_SERIES, order_p=1, order_d=1, order_q=1, training_time_steps=15,
        forecasting_time_steps=3, iterations=100, output_length=400, seed=12345,
        thinning_interval=1,
    )
    assert len(res["mode_curve"]) == 23
    assert math.isfinite(res["aic"])


def test_arimax_analysis_returns_forecast_curve():
    res = corehydropy.arimax_analysis(
        TS_SERIES, trend="Linear", training_time_steps=15, forecasting_time_steps=0,
        iterations=100, output_length=400, seed=12345, thinning_interval=1,
    )
    assert len(res["mode_curve"]) == 20  # fit-only
    assert math.isfinite(res["bic"])


def test_estimation_diagnostics_returns_all_three():
    d = corehydropy.estimation_diagnostics(
        SMOKE_PEAKS, "Normal", iterations=100, output_length=400, seed=12345,
        thinning_interval=1,
    )
    # Leverage
    assert len(d["leverage"]["leverage"]) == len(SMOKE_PEAKS)
    assert len(d["leverage"]["value"]) == len(SMOKE_PEAKS)
    assert math.isfinite(d["leverage"]["total_leverage"])
    assert all(math.isfinite(v) for v in d["leverage"]["leverage"])
    # Influence (PSIS-LOO)
    assert d["influence"]["count"] == len(SMOKE_PEAKS)
    assert len(d["influence"]["pareto_k"]) == len(SMOKE_PEAKS)
    assert math.isfinite(d["influence"]["mean_pareto_k"])
    assert isinstance(d["influence"]["is_reliable"], bool)
    # Prior influence
    assert d["prior_influence"]["count"] > 0
    assert math.isfinite(d["prior_influence"]["prior_to_data_ratio"])
    assert isinstance(d["prior_influence"]["is_prior_influential"], bool)
