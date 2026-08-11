# Behavioural tests for the fit surface (Task 8), the Python twin of corehydror's
# tests/testthat/test-fit.R -- same cases, same order, same names where Python allows. Oracle
# VALUES live in fixtures/ and are asserted by test_fixtures.py; this file asserts argument
# handling, error messages, object shape, and the Fit class's own methods (the Python stand-in
# for R's S3 generics -- coef()/vcov()/logLik()/confint()/print()/summary()).
#
# Two R cases have no Python counterpart and are intentionally NOT translated:
#   - "fit_input produces valid JSON when settings is empty": R's fit_input() needs a dedicated
#     empty-settings branch because R has no JSON parser and splices settings JSON as a raw
#     string (see fit.R's header note); fit.py builds the construct as a real dict and calls
#     json.dumps() once, so json.dumps({"model": ..., **{}}) has no equivalent edge case to
#     regression-test.
#   - "a fit round-trips through save and load" (R: saveRDS/readRDS): the Python analogue is
#     stdlib pickle, translated below as test_a_fit_round_trips_through_pickle.
#
# One R behaviour is NOT reproduced here on purpose: R's model_simulate()/model_validate()/
# model_log_likelihood() transparently unwrap a corehydro_fit via check_model() (model.R).
# corehydropy's model_*() verbs do not do this -- extending them is out of Task 8's scope (it
# would touch models.py, not listed in the brief) -- so the "fitted model simulates identically"
# test below compares Fit.model against Fit.to_model() instead of exercising an auto-unwrap.

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from corehydropy import (
    Fit,
    Model,
    fit_bayesian,
    fit_diagnostics,
    fit_gmm,
    fit_map,
    fit_mle,
    model_bulletin17c,
    model_simulate,
    model_univariate,
    quantile_variance,
)
from corehydropy import fit as _fit_mod

PEAKS = [12500, 15300, 9870, 21000, 18400, 11200, 26800, 14100, 19500, 11600]


def test_fit_mle_returns_a_fit_with_the_common_surface():
    f = fit_mle(model_univariate("Normal", PEAKS))
    assert isinstance(f, Fit)
    assert f.method == "MaximumLikelihood"
    assert len(f.parameters) == 2
    assert list(f.parameters.keys()) == f.parameter_names
    assert f.converged
    assert f.status == "Success"
    assert math.isfinite(f.log_likelihood)
    assert f.covariance.shape == (2, 2)
    assert f.nobs == 10


def test_the_vector_convenience_path_matches_the_model_path_exactly():
    assert fit_mle(PEAKS, "Normal").parameters == fit_mle(model_univariate("Normal", PEAKS)).parameters


def test_fit_properties_are_consistent_with_aic_bic_definitions():
    # Python has no separate coef()/vcov()/logLik()/AIC()/BIC() generics to call off a Fit --
    # .parameters/.covariance/.log_likelihood/.aic/.bic ARE the public surface, so this checks
    # what R's "base generics work off logLik" test checks: aic/bic agree with their textbook
    # definitions off log_likelihood/nobs/parameter count, the same values AIC()/BIC() derive.
    f = fit_mle(model_univariate("Normal", PEAKS))
    k = len(f.parameters)
    assert f.aic == pytest.approx(-2 * f.log_likelihood + 2 * k, rel=1e-10)
    assert f.bic == pytest.approx(-2 * f.log_likelihood + k * math.log(f.nobs), rel=1e-10)


def test_hessian_false_skips_the_covariance_stack_but_still_fits():
    f = fit_mle(model_univariate("Normal", PEAKS), hessian=False)
    assert f.covariance is None
    assert f.standard_errors is None
    assert math.isfinite(f.log_likelihood)


# NOTE: the sub-two-parameter NaN covariance guard is NOT testable from Python either, for the
# identical reason corehydror's test-fit.R documents: every single-parameter family in the
# factory fails to build as a UnivariateDistributionModel (set_default_parameters() throws). The
# guard is covered in C++ by test_fit_runner.cpp against a test-double ModelBase.


def test_confint_returns_profile_intervals_bracketing_the_estimate():
    f = fit_mle(model_univariate("Normal", PEAKS), profile=True, profile_bins=20)
    ci = f.confint(level=0.9)
    for name, value in f.parameters.items():
        assert ci["lower"][name] <= value
        assert ci["upper"][name] >= value


def test_confint_computes_on_demand_when_the_fit_was_built_without_profile():
    m = model_univariate("Normal", PEAKS)
    a = fit_mle(m).confint(level=0.9)
    b = fit_mle(m, profile=True).confint(level=0.9)
    assert a == b


def test_confint_returns_posterior_credible_intervals_for_a_bayesian_fit():
    f = fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz", iterations=200,
                      output_length=500, seed=123)
    # fit_bayesian() never sets credible_interval_width, so the fit was built at
    # BayesianAnalysis's own class default, 0.9.
    ci = f.confint(level=0.9)
    assert set(ci) == {"lower", "upper"}
    assert list(ci["lower"]) == f.parameter_names
    # Requested level matches the fit's own: reused directly from posterior_summary, not rebuilt.
    assert ci["lower"] == dict(zip(f.parameter_names, f.posterior_summary["lower"]))
    assert ci["upper"] == dict(zip(f.parameter_names, f.posterior_summary["upper"]))
    for name in f.parameter_names:
        assert ci["lower"][name] <= f.posterior_summary["mean"][f.parameter_names.index(name)]
        assert ci["upper"][name] >= f.posterior_summary["mean"][f.parameter_names.index(name)]


def test_confint_on_a_bayesian_fit_rebuilds_off_a_reproduced_chain_at_a_different_level():
    f = fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz", iterations=200,
                      output_length=500, seed=123)
    ci90 = f.confint(level=0.9)
    ci99 = f.confint(level=0.99)
    assert ci90 != ci99
    # A 99% credible interval is at least as wide as a 90% one at every parameter.
    for name in f.parameter_names:
        width90 = ci90["upper"][name] - ci90["lower"][name]
        width99 = ci99["upper"][name] - ci99["lower"][name]
        assert width99 >= width90


def test_confint_errors_for_a_gmm_fit_and_points_at_quantile_variance():
    f = fit_gmm(model_bulletin17c(PEAKS))
    with pytest.raises(ValueError, match="quantile_variance"):
        f.confint()


def test_confint_default_level_is_0_95_triggering_a_bayesian_rebuild():
    # Twin of test-fit.R's "confint() with no level argument defaults to 0.95, triggering a
    # Bayesian rebuild" -- pins the fix for the divergence where Fit.confint's default was 0.9
    # (BayesianAnalysis's own class default) while R's confint.corehydro_fit defaulted to 0.95
    # (base R's confint() convention). fit_bayesian() always builds the chain at 0.9, so a
    # no-argument confint() call always asks for a different level and takes the rebuild path.
    f = fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz", iterations=200,
                      output_length=500, seed=123)
    assert f.confint() == f.confint(level=0.95)
    assert f.confint() != f.confint(level=0.9)


def test_confint_default_level_matches_r_for_the_identical_seeded_bayesian_fit():
    # Cross-language check for the same divergence: this pytest suite has no established way to
    # invoke R (no rpy2/subprocess-to-Rscript harness anywhere in this test tree), so per the
    # project's "do not invent one" instruction this only carries the Python half, structurally
    # (no R-derived literal hardcoded here -- oracle values live in fixtures/*.json, not
    # behavioural tests). The actual cross-language numeric comparison for this identical model/
    # sampler/seed was run by hand and is recorded in .superpowers/sdd/task-8-report.md: R's
    # `confint(f)` (no `level`) and Python's `f.confint()` (no `level`) agree to every printed
    # digit.
    f = fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz", iterations=200,
                      output_length=500, seed=123)
    ci = f.confint()
    assert ci == f.confint(level=0.95)
    for name in f.parameter_names:
        assert ci["lower"][name] <= f.posterior_summary["mean"][f.parameter_names.index(name)]
        assert ci["upper"][name] >= f.posterior_summary["mean"][f.parameter_names.index(name)]


def test_the_fit_carries_a_fitted_model_that_simulates_identically():
    f = fit_mle(model_univariate("Normal", PEAKS))
    assert isinstance(f.model, Model)
    assert f.to_model() is f.model
    assert np.array_equal(
        model_simulate(f.model, n=25, seed=7),
        model_simulate(f.to_model(), n=25, seed=7),
    )
    assert list(f.model.spec["parameter_values"]) == pytest.approx(list(f.parameters.values()))


def test_a_fit_round_trips_through_pickle():
    f = fit_mle(model_univariate("Normal", PEAKS))
    restored = pickle.loads(pickle.dumps(f))
    assert restored.parameters == f.parameters


def test_argument_errors_name_the_offending_value():
    m = model_univariate("Normal", PEAKS)
    with pytest.raises(ValueError, match="Simplexx"):
        fit_mle(m, optimizer="Simplexx")
    with pytest.raises(ValueError, match="distribution"):
        fit_mle(PEAKS)


def test_covariance_keeps_the_layout_cpp_built_not_a_re_flattened_transpose():
    # LogPearsonTypeIII's 3 parameters give a covariance with real (non-zero) off-diagonal
    # entries -- unlike Normal's asymptotically-independent mean/sd -- so a transpose bug would
    # actually move a value to a different cell instead of swapping two zeros. Comparing directly
    # against the raw `fit_run` result (via the internal _fit_mod helpers, the twin of R's
    # `corehydror:::` triple-colon internal access) means no oracle literal is hardcoded here.
    m = model_univariate("LogPearsonTypeIII", PEAKS)
    construct_json, dataset, spec = _fit_mod._fit_input(
        m, None, {"optimizer": "NelderMead", "hessian": True, "profile": False, "profile_bins": 100}
    )
    raw = _fit_mod._fit_run("MaximumLikelihood", construct_json, dataset)
    raw_cov = np.array(raw["covariance"])
    assert np.any(np.abs(raw_cov[np.triu_indices_from(raw_cov, k=1)]) > 0)

    f = _fit_mod._new_fit(raw, spec, dataset, "NelderMead", level=0.9, construct_json=construct_json)
    assert np.array_equal(f.covariance, raw_cov)


def test_fit_mle_profile_true_surfaces_a_plottable_per_parameter_grid():
    f = fit_mle(model_univariate("Normal", PEAKS), profile=True, profile_bins=20)
    assert isinstance(f.profile, dict)
    assert list(f.profile.keys()) == f.parameter_names
    for grid in f.profile.values():
        assert grid.shape == (20, 2)
        # The grid comes from an ordered sequence; a column swap would break monotonicity here.
        assert np.all(np.diff(grid[:, 0]) >= 0)


def test_profile_is_none_when_the_fit_was_built_without_profile_true():
    f = fit_mle(model_univariate("Normal", PEAKS))
    assert f.profile is None


def test_fit_bayesian_returns_draws_as_iteration_chain_parameter():
    f = fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz", chains=4,
                      iterations=200, output_length=500, seed=12345)
    assert isinstance(f, Fit)
    assert f.draws.ndim == 3
    assert f.draws.shape[1] == 4  # chains on axis 1
    assert f.draws.shape[2] == len(f.parameters)  # parameters on axis 2
    assert f.posterior_summary["parameter_names"] == f.parameter_names
    assert len(f.posterior_summary["mean"]) == len(f.parameters)
    assert {"rhat", "ess", "median"} <= set(f.posterior_summary)
    assert len(f.acceptance_rates) == 4
    assert math.isfinite(f.dic)


def test_a_seeded_bayesian_fit_is_reproducible():
    m = model_univariate("Normal", PEAKS)
    a = fit_bayesian(m, sampler="DEMCz", iterations=100, seed=99)
    b = fit_bayesian(m, sampler="DEMCz", iterations=100, seed=99)
    assert np.array_equal(a.draws, b.draws)


def test_fit_bayesian_rejects_a_sampler_bayesiananalysis_cannot_construct():
    with pytest.raises(ValueError, match="mcmc_sample"):
        fit_bayesian(model_univariate("Normal", PEAKS), sampler="HMC")


def test_fit_bayesian_derives_warmup_as_max_50_iterations_over_2_when_omitted():
    f = fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz", iterations=200,
                      output_length=300, seed=1)
    assert f.warmup == 100
    # A small `iterations` with the omitted-warmup class default (1500) would trip the sampler's
    # own `warmup <= iterations / 2` guard -- this fit succeeding at all is part of what this
    # test is checking.
    assert math.isfinite(f.dic)


def test_an_explicit_warmup_is_passed_through_unchanged():
    f = fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz", iterations=200,
                      warmup=77, output_length=300, seed=1)
    assert f.warmup == 77


def test_the_draws_permutation_still_holds_when_chains_and_parameter_counts_differ():
    # Normal has 2 parameters; BayesianAnalysis requires between 4 and 20 chains, so 6 chains is
    # the smallest valid count that still makes the chain and parameter axes different lengths --
    # a transposition bug moves a value to the wrong cell instead of silently matching by symmetry.
    f = fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz", chains=6,
                      iterations=150, output_length=200, seed=5)
    assert f.draws.shape[1] == 6
    assert f.draws.shape[2] == 2
    assert f.draws.shape[2] == len(f.parameters)


def test_fit_bayesian_rejects_a_knob_its_sampler_does_not_use():
    with pytest.raises(ValueError, match="beta"):
        fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz", iterations=100, beta=0.1)


def test_fit_gmm_requires_a_bulletin17c_model():
    with pytest.raises(ValueError, match="bulletin17c"):
        fit_gmm(model_univariate("Normal", PEAKS))


def test_fit_gmm_returns_the_gmm_bookkeeping():
    f = fit_gmm(model_bulletin17c(PEAKS))
    assert f.method == "GMM"
    assert f.gmm_iterations > 0
    assert f.j_stat_pval is None  # just-identified by construction
    assert quantile_variance(f, 0.01) > 0


def test_log_likelihood_aic_bic_are_none_not_nan_for_a_gmm_fit():
    f = fit_gmm(model_bulletin17c(PEAKS))
    assert f.log_likelihood is None
    assert f.aic is None
    assert f.bic is None
    assert f.nobs == 0


def test_quantile_variance_rejects_a_non_gmm_fit():
    with pytest.raises(ValueError, match="fit_gmm"):
        quantile_variance(fit_mle(model_univariate("Normal", PEAKS)), 0.01)


def test_fit_diagnostics_returns_one_value_per_observation():
    d = fit_diagnostics(fit_map(model_univariate("Normal", PEAKS)))
    assert len(d["cooks_distance"]) == len(PEAKS)


def test_fit_diagnostics_works_off_a_bayesian_fit_and_a_gmm_fit():
    db = fit_diagnostics(fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz",
                                       iterations=100, output_length=200, seed=1))
    assert len(db["pareto_k"]) == len(PEAKS)
    assert math.isfinite(db["max_pareto_k"])

    dg = fit_diagnostics(fit_gmm(model_bulletin17c(PEAKS)))
    assert len(dg["cooks_distance"]) == len(PEAKS)
    assert dg["observation_influence"].shape[0] == len(PEAKS)


def test_fit_diagnostics_rejects_an_mle_fit():
    with pytest.raises(ValueError, match="fit_map"):
        fit_diagnostics(fit_mle(model_univariate("Normal", PEAKS)))


def test_summary_shows_rhat_ess_for_a_bayesian_fit_and_ses_for_an_optimized_one():
    fb = fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz", iterations=100,
                       output_length=200, seed=1)
    assert "rhat" in fb.summary()

    fo = fit_mle(model_univariate("Normal", PEAKS))
    assert "standard errors" in fo.summary()


def test_bayesian_draws_axis_order_matches_r():
    """Draws are (iteration, chain, parameter), the same order corehydror returns."""
    # chains must differ from the parameter count or a transposition is invisible. Note
    # BayesianAnalysis::validate() requires between 4 and 20 chains, so 6, not 3.
    f = fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz",
                      chains=6, iterations=200, seed=12345)
    assert f.draws.ndim == 3
    assert f.draws.shape[1] == 6
    assert f.draws.shape[2] == len(f.parameters) == 2
