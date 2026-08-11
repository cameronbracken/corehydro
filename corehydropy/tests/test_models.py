"""Behavioural tests for the model layer: the nine constructors, the helpers, and the verbs.

Numeric oracles live in ``fixtures/`` (model_parameter_overrides.json,
gmm_bulletin17c_censored.json, and the censored estimation fixtures); what is checked here is
the Python surface -- spec assembly, name resolution, argument validation, and the equivalences
a caller relies on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import corehydropy as ch

PEAKS = [12500.0, 15300.0, 8900.0, 22100.0, 18700.0, 14200.0, 9800.0, 28500.0, 17400.0,
         11600.0, 19200.0, 13800.0, 25600.0, 10500.0, 16900.0]
SERIES = [10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5,
          13.9, 15.2, 14.1, 12.9, 13.6, 15.0, 16.2, 14.8]


def test_model_univariate_carries_the_family_and_the_data():
    m = ch.model_univariate("Normal", PEAKS)
    assert isinstance(m, ch.Model)
    assert m.spec["type"] == "univariate_distribution"
    assert m.spec["family"] == "Normal"
    assert m.spec["dataset"] == "data"
    assert "data_frame" not in m.spec
    assert m.dataset == PEAKS


def test_an_analysis_data_frame_travels_inline():
    m = ch.model_univariate("Normal", ch.analysis_data(PEAKS))
    assert "dataset" not in m.spec
    assert len(m.spec["data_frame"]["exact"]) == len(PEAKS)
    assert m.dataset == []


def test_a_plain_sequence_and_an_equivalent_frame_give_the_identical_model():
    # The vector constructor delegates to the same DataFrame constructor, so an uncensored
    # record must evaluate identically whichever way it arrives.
    assert (ch.model_log_likelihood(ch.model_univariate("Normal", PEAKS))
            == ch.model_log_likelihood(ch.model_univariate("Normal", ch.analysis_data(PEAKS))))


def test_trend_widens_the_parameter_vector():
    stationary = ch.model_univariate("Normal", PEAKS)
    trended = ch.model_univariate("Normal", PEAKS, trends=ch.trend("mean", "Linear"))
    assert len(ch.model_parameters(stationary)["values"]) == 2
    assert len(ch.model_parameters(trended)["values"]) == 3
    assert trended.spec["trends"][0]["type"] == "Linear"
    assert trended.spec["trends"][0]["parameter"] == 0


@pytest.mark.parametrize("name", ["mean", "mu", "µ", "Mean (µ)"])
def test_trend_resolves_a_parameter_by_any_accepted_name(name):
    m = ch.model_univariate("Normal", PEAKS, trends=ch.trend(name, "Linear"))
    assert m.spec["trends"][0]["parameter"] == 0


def test_trend_resolves_a_parameter_by_position_and_the_second_name():
    assert ch.model_univariate(
        "Normal", PEAKS, trends=ch.trend(1, "Linear")).spec["trends"][0]["parameter"] == 0
    assert ch.model_univariate(
        "Normal", PEAKS, trends=ch.trend("sd", "Linear")).spec["trends"][0]["parameter"] == 1


def test_trend_validates_its_arguments():
    with pytest.raises(ValueError, match="unknown trend type"):
        ch.trend("mean", "Sigmoid")
    with pytest.raises(ValueError, match="unknown trend parameter"):
        ch.model_univariate("Normal", PEAKS, trends=ch.trend("nope", "Linear"))
    with pytest.raises(ValueError, match="positive integer"):
        ch.model_univariate("Normal", PEAKS, trends=ch.trend(0, "Linear"))
    with pytest.raises(TypeError, match="must be Trend objects"):
        ch.model_univariate("Normal", PEAKS, trends=["not a trend"])


def test_trend_accepts_a_bare_object_or_a_sequence():
    one = ch.model_univariate("Normal", PEAKS, trends=ch.trend("mean", "Linear"))
    listed = ch.model_univariate("Normal", PEAKS, trends=[ch.trend("mean", "Linear")])
    assert one.spec == listed.spec


def test_model_parameter_sets_bounds_the_fixed_flag_and_a_value():
    m = ch.model_univariate("Normal", PEAKS, parameters=ch.model_parameter(
        "sd", lower=100, upper=40000, fixed=True, value=5000))
    p = m.spec["parameters"][0]
    assert p == {"index": 1, "value": 5000.0, "lower": 100.0, "upper": 40000.0, "is_fixed": True}


def test_a_custom_prior_changes_only_the_prior_half():
    plain = ch.model_log_likelihood(ch.model_univariate("Normal", PEAKS))
    primed = ch.model_log_likelihood(ch.model_univariate(
        "Normal", PEAKS,
        parameters=ch.model_parameter("mean", prior=ch.Distribution("Normal", [15000, 4000])),
        use_default_flat_priors=False,
    ))
    assert primed["data_log_likelihood"] == plain["data_log_likelihood"]
    assert primed["prior_log_likelihood"] != plain["prior_log_likelihood"]
    assert primed["log_likelihood"] == pytest.approx(
        primed["data_log_likelihood"] + primed["prior_log_likelihood"])


def test_model_parameter_validates_its_arguments():
    with pytest.raises(TypeError, match="Distribution"):
        ch.model_parameter("sd", prior="Normal")
    with pytest.raises(ValueError, match="must not exceed"):
        ch.model_parameter("sd", lower=10, upper=1)
    with pytest.raises(ValueError, match="unknown parameter"):
        ch.model_univariate("Normal", PEAKS, parameters=ch.model_parameter("nope"))
    with pytest.raises(TypeError, match="must be ModelParameter objects"):
        ch.model_univariate("Normal", PEAKS, parameters=["nope"])


def test_a_trended_model_requires_a_parameter_position():
    with pytest.raises(ValueError, match="pass a 1-based position"):
        ch.model_univariate("Normal", PEAKS, trends=ch.trend("mean", "Linear"),
                            parameters=ch.model_parameter("sd", lower=1))
    ch.model_univariate("Normal", PEAKS, trends=ch.trend("mean", "Linear"),
                        parameters=ch.model_parameter(3, lower=1))


def test_parameter_values_wins_over_a_per_parameter_value():
    m = ch.model_univariate("Normal", PEAKS,
                            parameters=ch.model_parameter("mean", value=1),
                            parameter_values=[16000, 6000])
    assert ch.model_parameters(m)["values"] == pytest.approx([16000.0, 6000.0])


def test_model_parameters_names_a_stationary_univariate_vector():
    p = ch.model_parameters(ch.model_univariate("Normal", PEAKS))
    assert p["names"] == ["mean", "std dev"]
    assert len(p["symbols"]) == 2
    # A trended model's vector is not the distribution's, so it is left unnamed.
    trended = ch.model_univariate("Normal", PEAKS, trends=ch.trend("mean", "Linear"))
    assert ch.model_parameters(trended)["names"] == []


def test_model_validate_reports_validity():
    v = ch.model_validate(ch.model_univariate("Normal", PEAKS))
    assert v["is_valid"] is True
    assert v["messages"] == []


def test_model_simulate_draws_a_reproducible_seeded_sample():
    m = ch.model_univariate("Normal", PEAKS)
    a = ch.model_simulate(m, 5, seed=12345)
    assert len(a) == 5
    assert np.array_equal(a, ch.model_simulate(m, 5, seed=12345))
    assert not np.array_equal(a, ch.model_simulate(m, 5, seed=999))


def test_model_log_likelihood_evaluates_at_supplied_parameters():
    m = ch.model_univariate("Normal", PEAKS)
    default = ch.model_log_likelihood(m)
    explicit = ch.model_log_likelihood(m, default["parameters"])
    assert explicit["log_likelihood"] == default["log_likelihood"]
    moved = ch.model_log_likelihood(m, [16000, 6000])
    assert moved["parameters"] == pytest.approx([16000.0, 6000.0])
    assert moved["log_likelihood"] != default["log_likelihood"]
    with pytest.raises(Exception, match="wrong length"):
        ch.model_log_likelihood(m, [1, 2, 3])


def _every_family():
    return {
        "univariate": ch.model_univariate("Normal", PEAKS),
        "mixture": ch.model_mixture(["Normal", "Normal"], PEAKS),
        "competing_risks": ch.model_competing_risks(["Gumbel", "Gumbel"], PEAKS),
        "point_process": ch.model_point_process(PEAKS, threshold=9000, total_years=15),
        "bulletin17c": ch.model_bulletin17c(PEAKS),
        # The time-series default of max(30, floor(0.8 * n)) training steps exceeds this
        # 20-point series, so it is set explicitly (see model_ar's documentation).
        "ar": ch.model_ar(SERIES, p=1, training_time_steps=15),
        "ma": ch.model_ma(SERIES, q=1, training_time_steps=15),
        "arima": ch.model_arima(SERIES, p=1, d=0, q=1, training_time_steps=15),
        "arimax": ch.model_arimax(SERIES, covariates=[[v / 10 for v in range(len(SERIES))]],
                                  p=1, d=0, q=0, training_time_steps=15),
        "rating_curve": ch.model_rating_curve(
            [1.0, 1.4, 1.9, 2.3, 2.8, 3.2, 3.7, 4.1, 4.6, 5.0, 5.5, 6.0],
            [12, 28, 61, 98, 158, 214, 302, 379, 490, 587, 723, 872],
        ),
        "spatial_gev": ch.model_spatial_gev(
            [[0, 0], [1, 0], [0, 1]],
            [[10, 12, 11], [14, 15, 13], [9, 10, 11], [16, 17, 15]],
        ),
        "bivariate": ch.model_bivariate(
            marginal_x={"family": "Normal", "data": PEAKS,
                        "parameter_values": [float(np.mean(PEAKS)), float(np.std(PEAKS, ddof=1))]},
            marginal_y={"family": "Normal", "data": [v / 10 for v in PEAKS],
                        "parameter_values": [float(np.mean(PEAKS)) / 10,
                                             float(np.std(PEAKS, ddof=1)) / 10]},
        ),
    }


@pytest.mark.parametrize("name", sorted(_every_family()))
def test_every_family_builds_and_validates(name):
    m = _every_family()[name]
    assert isinstance(m, ch.Model)
    if name == "bulletin17c":
        # Bulletin17CDistribution is an IGMMModel, not a ModelBase, so it has no validate arm.
        return
    assert ch.model_validate(m)["is_valid"] is True, name


def test_training_time_steps_is_required_for_a_short_series():
    # The model default is max(30, floor(0.8 * n)); for a 20-point series that is 30, which
    # exceeds the series and fails validation until the caller sets it.
    v = ch.model_validate(ch.model_ar(SERIES, p=1))
    assert v["is_valid"] is False
    assert any("Training time steps" in m for m in v["messages"])
    assert ch.model_validate(ch.model_ar(SERIES, p=1, training_time_steps=15))["is_valid"]


def test_time_series_models_reject_a_censored_frame():
    with pytest.raises(TypeError, match="plain sequence in order"):
        ch.model_ar(ch.analysis_data(SERIES), p=1)


def test_rating_curve_and_bivariate_validate_their_inputs():
    with pytest.raises(ValueError, match="same length"):
        ch.model_rating_curve([1, 2, 3, 4, 5], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="`family` and `data`"):
        ch.model_bivariate(marginal_x={"family": "Normal"},
                           marginal_y={"family": "Normal", "data": [1, 2, 3, 4, 5]})


def test_a_bivariate_model_simulates_an_n_by_2_array():
    m = _every_family()["bivariate"]
    draw = ch.model_simulate(m, 6, seed=12345)
    assert draw.shape == (6, 2)


def test_repr_summarizes_the_model():
    assert "univariate_distribution: Normal" in repr(ch.model_univariate("Normal", PEAKS))
    assert "15 exact" in repr(ch.model_univariate("Normal", PEAKS))
    assert "time_series/ar" in repr(ch.model_ar(SERIES, p=1))
    assert "Normal + Gumbel" in repr(ch.model_mixture(["Normal", "Gumbel"], PEAKS))
    assert "1 trends" in repr(
        ch.model_univariate("Normal", PEAKS, trends=ch.trend("mean", "Linear")))
    assert "Linear on parameter mean" in repr(ch.trend("mean", "Linear"))
    assert "lower = 1" in repr(ch.model_parameter("sd", lower=1))


def test_the_model_verbs_reject_a_non_model():
    for fn in (ch.model_validate, ch.model_log_likelihood, ch.model_parameters):
        with pytest.raises(TypeError, match="Model"):
            fn(PEAKS)


def test_the_model_round_trips_through_its_json_spec():
    m = ch.model_univariate("LogPearsonTypeIII", ch.analysis_data(PEAKS, mgbt_low_outliers=True))
    assert json.loads(m.to_json()) == m.spec
