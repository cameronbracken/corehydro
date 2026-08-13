# Binding-specific behaviour for the goodness-of-fit surface. Oracle-value checks live in
# test_fixtures.py, driven by the shared language-neutral fixtures. Mirrors corehydror's
# test-gof.R assertion for assertion.
import pytest

from corehydropy import Distribution, aic, aic_weights, gof_test, goodness_of_fit


def test_goodness_of_fit_returns_every_metric_and_subsets_agree_with_the_whole():
    obs = [2, 4, 6, 8, 10]
    mod = [2.2, 3.9, 6.4, 7.5, 10.1]
    every = goodness_of_fit(obs, mod)
    assert len(every) == 17
    assert {"nse", "kge", "rsr", "ve"} <= every.keys()
    assert goodness_of_fit(obs, mod, metrics=["nse"])["nse"] == every["nse"]


def test_an_unknown_metric_name_is_rejected_and_names_the_offender():
    with pytest.raises(ValueError, match="nsee"):
        goodness_of_fit([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], metrics=["nsee"])


def test_mismatched_lengths_are_rejected_before_reaching_cpp():
    with pytest.raises(ValueError, match="same length"):
        goodness_of_fit([1, 2, 3, 4, 5], [1, 2, 3, 4])


def test_gof_test_accepts_a_distribution_and_rejects_anything_else():
    x = [2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9]
    d = Distribution("Normal", [3.4, 1.1])
    assert gof_test(x, d, "ks") > 0
    assert gof_test(x, d, "ad") > 0
    with pytest.raises(TypeError, match="Distribution"):
        gof_test(x, "Normal")


def test_the_information_criteria_match_the_closed_forms():
    assert aic(k=2, log_likelihood=-100) == 204
    assert sum(aic_weights([246, 248.8, 251.2])) == pytest.approx(1.0)
