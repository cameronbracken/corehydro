"""Generic, fixture-driven validation for corehydropy.

Reads the language-neutral oracle fixtures (the single source of truth shared with
the C++ core and the R package) and checks every assertion. No oracle values live
in this file -- only the dispatch from fixture method names to the Python API. The GEV
slice uses its bespoke object API; every other distribution goes through the polymorphic
``_core.dist_*`` functions (factory + UnivariateDistributionBase).
"""
from __future__ import annotations

import json
import math
from importlib.resources import files
from pathlib import Path

import pytest

from corehydropy import GeneralizedExtremeValue, gev_fit
from corehydropy import _core

_MOMENTS = ("mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum")


def _fixtures_dir() -> Path:
    # Prefer the copy shipped inside the installed package; fall back to the repo
    # canonical fixtures/ for in-tree development.
    try:
        packaged = files("corehydropy") / "fixtures"
        if packaged.is_dir():
            return Path(str(packaged))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return Path(__file__).resolve().parents[2] / "fixtures"


def _num(v):
    if isinstance(v, str):
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[v]
    return float(v)


# --- GEV slice (bespoke object API) ----------------------------------------------------


def _build_gev(construct, datasets):
    if "params" in construct:
        return GeneralizedExtremeValue(*[_num(v) for v in construct["params"]])
    fit = construct["fit"]
    res = gev_fit(datasets[fit["dataset"]], fit["method"])
    return GeneralizedExtremeValue(res["location"], res["scale"], res["shape"])


def _dispatch_gev(g, method, args):
    simple = {
        "mean": g.mean, "median": g.median, "mode": g.mode, "skewness": g.skewness,
        "kurtosis": g.kurtosis, "minimum": g.minimum, "maximum": g.maximum,
        "sd": g.standard_deviation,
    }
    if method in simple:
        return simple[method]()
    if method in ("pdf", "cdf", "quantile"):
        return getattr(g, method)(args[0])
    if method == "parameters_valid":
        return g.parameters_valid
    if method == "param":
        return {"location": g.location, "scale": g.scale, "shape": g.shape}[args[0]]
    if method == "linear_moment":
        return g.linear_moments_from_parameters([g.location, g.scale, g.shape])[int(args[0])]
    if method == "quantile_gradient":
        return g.quantile_gradient(args[0])[int(args[1])]
    if method == "parameter_covariance":
        return g.parameter_covariance(int(args[0]))[int(args[1])][int(args[2])]
    if method == "quantile_variance":
        return g.quantile_variance(args[0], int(args[1]))
    if method == "quantile_se":
        return math.sqrt(g.quantile_variance(args[0], int(args[1])))
    raise KeyError(f"unknown fixture method: {method}")


# --- Composite distribution path -------------------------------------------------------
# For TruncatedDistribution (and future Empirical/KernelDensity/Mixture/CompetingRisks),
# the fixture "construct" uses a structured schema rather than flat "params".
# Adding a new composite = one new case in _build_composite + one in _dispatch_composite.

_COMPOSITE_TARGETS = {"TruncatedDistribution", "Empirical", "KernelDensity", "Mixture",
                      "CompetingRisks"}


def _build_composite(target: str, construct: dict, datasets: dict | None = None) -> dict:
    """Parse a composite construct into a dict that _dispatch_composite can use."""
    if target == "TruncatedDistribution":
        base_target = construct["base"]["target"]
        base_params = [_num(v) for v in construct["base"]["params"]]
        lo, hi = float(construct["bounds"][0]), float(construct["bounds"][1])
        return {"base_target": base_target, "base_params": base_params, "lo": lo, "hi": hi}
    if target == "Empirical":
        xv = [float(v) for v in construct["x"]]
        pv = [float(v) for v in construct["p"]]
        pt = construct.get("p_transform", "NormalZ")
        # v2.1.4: p_descending DECLARES the probability order (mirrors C#'s explicit
        # `probabilityOrder` argument -- NOT auto-detected from the data); default False
        # matches the ordinary ascending-CDF case.
        pd = bool(construct.get("p_descending", False))
        return {"x_vals": xv, "p_vals": pv, "p_transform": pt, "p_descending": pd}
    if target == "KernelDensity":
        ds = datasets or {}
        data_key = construct["data"]
        data_vec = [float(v) for v in ds[data_key]]
        kernel = construct.get("kernel", "Gaussian")
        bandwidth = float(construct["bandwidth"]) if "bandwidth" in construct else -1.0
        bounded = bool(construct.get("bounded_by_data", True))
        return {"data_vec": data_vec, "kernel": kernel, "bandwidth": bandwidth, "bounded": bounded}
    if target == "Mixture":
        comp_targets = [c["target"] for c in construct["components"]]
        comp_params = [[float(v) for v in c["params"]] for c in construct["components"]]
        wts = [float(w) for w in construct["weights"]]
        zero_inflated = bool(construct.get("zero_inflated", False))
        zero_weight = _num(construct.get("zero_weight", 0.0))
        return {"comp_targets": comp_targets, "comp_params": comp_params, "weights": wts,
                "zero_inflated": zero_inflated, "zero_weight": zero_weight}
    if target == "CompetingRisks":
        comp_targets = [c["target"] for c in construct["components"]]
        comp_params = [[float(v) for v in c["params"]] for c in construct["components"]]
        min_of_rv = bool(construct.get("minimum_of_random_variables", True))
        dependency = construct.get("dependency", "Independent")
        correlation = [[float(v) for v in row] for row in construct.get("correlation", [])]
        return {"comp_targets": comp_targets, "comp_params": comp_params,
                "minimum_of_rv": min_of_rv, "dependency": dependency,
                "correlation": correlation}
    raise KeyError(f"unknown composite target: {target}")


def _dispatch_composite(target: str, cd: dict, method: str, args: list):
    if target == "TruncatedDistribution":
        bt, bp, lo, hi = cd["base_target"], cd["base_params"], cd["lo"], cd["hi"]
        if method in _MOMENTS:
            return _core.trunc_moments(bt, bp, lo, hi)[method]
        if method == "pdf":
            return _core.trunc_pdf(bt, bp, lo, hi, args[0])
        if method == "cdf":
            return _core.trunc_cdf(bt, bp, lo, hi, args[0])
        if method == "quantile":
            return _core.trunc_quantile(bt, bp, lo, hi, args[0])
        if method == "parameters_valid":
            return _core.trunc_valid(bt, bp, lo, hi)
        if method == "param":
            # GetParameters mirrors {base_params..., lo, hi} -- no C++ call needed, `cd`
            # already holds the flat tuple (used by the sequential_setparameters_recovery-
            # style cases to confirm a restored parameter set after a SetParameters recovery).
            return (bp + [lo, hi])[int(args[0])]
        raise KeyError(f"unknown fixture method for TruncatedDistribution: {method}")
    if target == "Empirical":
        xv, pv, pt, pd = cd["x_vals"], cd["p_vals"], cd["p_transform"], cd["p_descending"]
        if method in _MOMENTS:
            return _core.emp_moments(xv, pv, pt, pd)[method]
        if method == "pdf":
            return _core.emp_pdf(xv, pv, pt, pd, args[0])
        if method == "cdf":
            return _core.emp_cdf(xv, pv, pt, pd, args[0])
        if method == "quantile":
            return _core.emp_quantile(xv, pv, pt, pd, args[0])
        if method == "parameters_valid":
            return _core.emp_valid(xv, pv, pt, pd)
        raise KeyError(f"unknown fixture method for Empirical: {method}")
    if target == "KernelDensity":
        dv, ker, bw, bd = cd["data_vec"], cd["kernel"], cd["bandwidth"], cd["bounded"]
        if method in _MOMENTS:
            return _core.kde_moments(dv, ker, bw, bd)[method]
        if method == "pdf":
            return _core.kde_pdf(dv, ker, bw, bd, args[0])
        if method == "cdf":
            return _core.kde_cdf(dv, ker, bw, bd, args[0])
        if method == "quantile":
            return _core.kde_quantile(dv, ker, bw, bd, args[0])
        if method == "parameters_valid":
            return _core.kde_valid(dv, ker, bw, bd)
        raise KeyError(f"unknown fixture method for KernelDensity: {method}")
    if target == "Mixture":
        ct, cp, wts = cd["comp_targets"], cd["comp_params"], cd["weights"]
        zi, zw = cd["zero_inflated"], cd["zero_weight"]
        if method in _MOMENTS:
            return _core.mix_moments(ct, cp, wts, zi, zw)[method]
        if method == "pdf":
            return _core.mix_pdf(ct, cp, wts, zi, zw, args[0])
        if method == "cdf":
            return _core.mix_cdf(ct, cp, wts, zi, zw, args[0])
        if method == "quantile":
            return _core.mix_quantile(ct, cp, wts, zi, zw, args[0])
        if method == "parameters_valid":
            return _core.mix_valid(ct, cp, wts, zi, zw)
        if method == "param":
            # GetParameters() flat vector [w0..wK-1, component params...] -- needed (not the
            # raw `wts` in `cd`) because the zero-inflation setters recompute the weights.
            return _core.mix_params(ct, cp, wts, zi, zw)[int(args[0])]
        raise KeyError(f"unknown fixture method for Mixture: {method}")
    if target == "CompetingRisks":
        ct, cp, min_rv = cd["comp_targets"], cd["comp_params"], cd["minimum_of_rv"]
        dep, corr = cd["dependency"], cd["correlation"]
        if method in _MOMENTS:
            return _core.cr_moments(ct, cp, min_rv, dep, corr)[method]
        if method == "pdf":
            return _core.cr_pdf(ct, cp, min_rv, dep, corr, args[0])
        if method == "log_pdf":
            return _core.cr_log_pdf(ct, cp, min_rv, dep, corr, args[0])
        if method == "cdf":
            return _core.cr_cdf(ct, cp, min_rv, dep, corr, args[0])
        if method == "quantile":
            return _core.cr_quantile(ct, cp, min_rv, dep, corr, args[0])
        if method == "parameters_valid":
            return _core.cr_valid(ct, cp, min_rv, dep, corr)
        if method == "dependency_change":
            # v2.1.4: verifies the Dependency setter fix + PerfectlyNegative no longer
            # zeroing CorrelationMatrix, in ONE self-contained call -- args = [x,
            # dependency2, i, j, field].
            x, dep2, i, j, field = args
            return _core.cr_dependency_change(ct, cp, min_rv, dep, dep2, corr, float(x),
                                               field, int(i), int(j))
        raise KeyError(f"unknown fixture method for CompetingRisks: {method}")
    raise KeyError(f"unknown composite target: {target}")


def _apply_set_parameters_composite(target: str, cd: dict, args: list) -> dict:
    """Applies a "set_parameters" fixture step to a composite's parsed construct data,
    mirroring the C# SetParameters(flat vector) entry point -- lets a case exercise a
    "construct valid -> SetParameters invalid -> recheck -> SetParameters valid -> recheck"
    sequence. There is no persistent C++ object to mutate here (every _core.trunc_*
    call is a stateless construct-and-compute), so this instead updates the local `cd` dict;
    the NEXT dispatch call reconstructs from the updated fields, which is behaviorally
    equivalent. Only TruncatedDistribution needs this in the current validation wave.
    """
    flat = [_num(v) for v in args]
    if target == "TruncatedDistribution":
        n_base = len(cd["base_params"])
        return {
            **cd,
            "base_params": flat[:n_base],
            "lo": flat[n_base],
            "hi": flat[n_base + 1],
        }
    raise KeyError(f"set_parameters not supported for composite target: {target}")


# --- Generic polymorphic path ----------------------------------------------------------


def _build_params(target, construct, datasets):
    if "params" in construct:
        return [_num(v) for v in construct["params"]]
    fit = construct["fit"]
    return list(_core.dist_fit(target, datasets[fit["dataset"]], fit["method"]))


def _dispatch_generic(target, params, method, args):
    if method in _MOMENTS:
        return _core.dist_moments(target, params)[method]
    if method == "pdf":
        return _core.dist_pdf(target, params, args[0])
    if method == "cdf":
        return _core.dist_cdf(target, params, args[0])
    if method == "quantile":
        return _core.dist_quantile(target, params, args[0])
    if method == "parameters_valid":
        return _core.dist_valid(target, params)
    if method == "param":
        return params[int(args[0])]
    if method == "linear_moment":
        return _core.dist_linear_moments(target, params)[int(args[0])]
    if method == "random_value":
        # args: [sample_size, seed, index] -- one draw from the seeded MT stream.
        return _core.dist_random(target, params, int(args[0]), int(args[1]))[int(args[2])]
    if method == "partial_kp":
        # Static GammaDistribution utility, not tied to `params` -- args: [skewness, probability].
        return _core.dist_gamma_partial_kp(float(args[0]), float(args[1]))
    raise KeyError(f"unknown fixture method: {method}")


# --- multivariate_distribution path -----------------------------------------------------
# Dirichlet/Multinomial/BivariateEmpirical dispatch to the bespoke _core.dirichlet_val/
# _core.multinomial_val/_core.bve_cdf functions (method + flat numeric args in, double
# out). Extensible: additional multivariate targets add a branch here plus their own
# _core.<name>_val entry point.

def _flatten_mv_args(args: list) -> list[float]:
    """Flattens fixture assertion args to a flat float list.

    Handles every convention seen in the multivariate fixtures: a single nested vector
    argument (e.g. pdf args = [[0.3, 0.4, 0.3]]), flat scalar args (e.g. covariance args =
    [0, 1], log_multivariate_beta args = [1.0, 1.0]), and a nested vector followed by a
    trailing scalar (e.g. MultivariateNormal inverse_cdf args = [[p1, p2], index]).
    """
    out: list[float] = []
    for v in args:
        if isinstance(v, list):
            out.extend(float(x) for x in v)
        else:
            out.append(float(v))
    return out


def _dispatch_multivariate(target: str, construct: dict, method: str, args: list):
    # Methods with a doubly-nested arg (e.g. cdf_xy_after_set_parameters's replacement
    # probability grid, or MultivariateNormal's marginal_*/conditional_*) can't flatten
    # through _flatten_mv_args's "one nested vector, or all-scalar" convention -- those
    # branches below never reference `ar`, so a flatten failure here is harmless.
    try:
        ar = _flatten_mv_args(args)
    except (TypeError, ValueError):
        ar = None
    if target == "Dirichlet":
        alpha = [float(v) for v in construct["alpha"]]
        return _core.dirichlet_val(method, alpha, ar)
    if target == "Multinomial":
        n = int(construct["n"])
        p = [float(v) for v in construct["p"]]
        return _core.multinomial_val(method, n, p, ar)
    if target == "BivariateEmpirical":
        x1 = [float(v) for v in construct["x1"]]
        x2 = [float(v) for v in construct["x2"]]
        p = [[float(v) for v in row] for row in construct["p"]]
        transforms = [
            construct.get("x1_transform", "None"),
            construct.get("x2_transform", "None"),
            construct.get("p_transform", "None"),
        ]
        if method == "cdf_xy_after_set_parameters":
            # v2.1.4 stale-cache fix, verified in ONE self-contained call -- args =
            # [[x1_new...], [x2_new...], [[p_row0...], ...], x1_eval, x2_eval].
            x1_new = [float(v) for v in args[0]]
            x2_new = [float(v) for v in args[1]]
            p_new = [[float(v) for v in row] for row in args[2]]
            return _core.bve_cdf_after_set_parameters(x1, x2, p, transforms, x1_new, x2_new,
                                                       p_new, float(args[3]), float(args[4]))
        return _core.bve_cdf(method, x1, x2, p, transforms, ar)
    if target == "MultivariateNormal":
        mean = [float(v) for v in construct["mean"]]
        cov = [[float(v) for v in row] for row in construct["covariance"]]
        # v2.1.4 Marginal/Conditional: dedicated entry points (not the flattened `ar`)
        # since these take a variable-length index vector (Conditional: a second
        # same-length values vector) that _flatten_mv_args's "one nested vector, or
        # all-scalar" convention can't disambiguate from adjacent variable-length vectors.
        if method == "marginal_mean":
            indices = [int(v) for v in args[0]]
            return _core.mvn_marginal_mean(mean, cov, indices, int(args[1]))
        if method == "marginal_covariance":
            indices = [int(v) for v in args[0]]
            return _core.mvn_marginal_covariance(mean, cov, indices, int(args[1]), int(args[2]))
        if method == "marginal_log_pdf":
            indices = [int(v) for v in args[0]]
            point = [float(v) for v in args[1]]
            return _core.mvn_marginal_log_pdf(mean, cov, indices, point)
        if method == "marginal_dimension":
            indices = [int(v) for v in args[0]]
            return _core.mvn_marginal_dimension(mean, cov, indices)
        if method == "conditional_mean":
            obs_indices = [int(v) for v in args[0]]
            obs_values = [float(v) for v in args[1]]
            return _core.mvn_conditional_mean(mean, cov, obs_indices, obs_values, int(args[2]))
        if method == "conditional_covariance":
            obs_indices = [int(v) for v in args[0]]
            obs_values = [float(v) for v in args[1]]
            return _core.mvn_conditional_covariance(mean, cov, obs_indices, obs_values,
                                                     int(args[2]), int(args[3]))
        if method == "conditional_dimension":
            obs_indices = [int(v) for v in args[0]]
            obs_values = [float(v) for v in args[1]]
            return _core.mvn_conditional_dimension(mean, cov, obs_indices, obs_values)
        return _core.mvn_val(method, mean, cov, ar)
    if target == "MultivariateStudentT":
        df = float(construct["df"])
        location = [float(v) for v in construct["location"]]
        scale = [[float(v) for v in row] for row in construct["scale"]]
        return _core.mvt_val(method, df, location, scale, ar)
    raise KeyError(f"unknown multivariate target: {target}")


# --- MultivariateNormal seeded batches --------------------------------------------------
# `cdf` (dim>=3), `interval`, and `mvndst` all draw from the seeded MVNUNI stream, so a
# RUN of consecutive same-method assertions in a seeded case must be evaluated on ONE
# persistent instance via the mvn_*_seq bindings in mvd.cpp, not dispatched one call at a
# time (which would silently reset the seed between assertions). _run_mvn_case() below
# groups consecutive assertions of these methods and batches them; everything else (and
# every case without a "seed") falls through to the stateless per-assertion dispatch
# above, unchanged.

_MVN_SEEDED_METHODS = ("cdf", "mvndst", "interval")


def _dispatch_mvn_seeded_seq(construct: dict, method: str, run: list):
    seed = int(construct["seed"])
    mean = [float(v) for v in construct["mean"]]
    cov = [[float(v) for v in row] for row in construct["covariance"]]

    if method == "cdf":
        xs = [[_num(v) for v in a["args"][0]] for a in run]
        return _core.mvn_cdf_seq(mean, cov, seed, xs)
    if method == "interval":
        lowers = [[_num(v) for v in a["args"][0]] for a in run]
        uppers = [[_num(v) for v in a["args"][1]] for a in run]
        return _core.mvn_interval_seq(mean, cov, seed, lowers, uppers)
    if method == "mvndst":
        # args = [n, [lower...], [upper...], [infin...], [correl...], maxpts, abseps, releps]
        n_dim = int(run[0]["args"][0])
        lowers = [[_num(v) for v in a["args"][1]] for a in run]
        uppers = [[_num(v) for v in a["args"][2]] for a in run]
        infins = [[int(v) for v in a["args"][3]] for a in run]
        correls = [[_num(v) for v in a["args"][4]] for a in run]
        maxpts_v = [int(a["args"][5]) for a in run]
        abseps_v = [float(a["args"][6]) for a in run]
        releps_v = [float(a["args"][7]) for a in run]
        return _core.mvn_mvndst_seq(n_dim, seed, lowers, uppers, infins, correls, maxpts_v, abseps_v,
                                     releps_v)
    raise KeyError(f"unknown seeded MultivariateNormal method: {method}")


def _run_mvn_case(construct: dict, assertions: list):
    seeded = "seed" in construct
    i = 0
    n = len(assertions)
    while i < n:
        a = assertions[i]
        method = a["method"]
        if seeded and method in _MVN_SEEDED_METHODS:
            j = i
            while j < n and assertions[j]["method"] == method:
                j += 1
            run = assertions[i:j]
            actuals = _dispatch_mvn_seeded_seq(construct, method, run)
            for actual, assertion in zip(actuals, run):
                _check(actual, assertion)
            i = j
        else:
            args = a.get("args", [])
            actual = _dispatch_multivariate("MultivariateNormal", construct, method, args)
            _check(actual, a)
            i += 1


# --- mcmc_sampler path -------------------------------------------------------------------
# Inherently STATEFUL (unlike multivariate_distribution's/bivariate_copula's per-assertion
# dispatch): one _core.mcmc_run call per case builds the model via the registry, configures
# the sampler from construct["settings"], and samples() ONCE; every assertion in the case
# reads the single returned dict. See fixtures/README.md's mcmc_sampler schema for the full
# method list and tolerance policy.


def _dispatch_mcmc(result: dict, method: str, args: list):
    if method == "posterior_mean":
        return result["posterior_mean"][int(args[0])]
    if method == "posterior_sd":
        return result["posterior_sd"][int(args[0])]
    if method == "posterior_median":
        return result["posterior_median"][int(args[0])]
    if method == "posterior_lower_ci":
        return result["posterior_lower_ci"][int(args[0])]
    if method == "posterior_upper_ci":
        return result["posterior_upper_ci"][int(args[0])]
    if method == "chain_value":
        return result["chains"][int(args[0])][int(args[1])][int(args[2])]
    if method == "chain_fitness":
        return result["chain_fitness"][int(args[0])][int(args[1])]
    if method == "map_value":
        return result["map_values"][int(args[0])]
    if method == "map_fitness":
        return result["map_fitness"]
    if method == "acceptance_rate":
        return result["acceptance_rates"][int(args[0])]
    if method == "mean_log_likelihood":
        return result["mean_log_likelihood"][int(args[0])]
    if method == "rhat":
        return result["rhat"][int(args[0])]
    if method == "ess":
        return result["ess"][int(args[0])]
    raise KeyError(f"unknown mcmc_sampler fixture method: {method}")


def _run_mcmc_case(target: str, construct: dict, assertions: list, datasets: dict):
    model = construct["model"]
    data = [float(v) for v in datasets[model["dataset"]]]
    settings = construct.get("settings", {})
    result = _core.mcmc_run(target, model["name"], model["family"], data, settings)
    for a in assertions:
        args = a.get("args", [])
        actual = _dispatch_mcmc(result, a["method"], args)
        _check(actual, a)


# --- bootstrap path ----------------------------------------------------------------------
# Inherently STATEFUL like mcmc_sampler: one _core.bootstrap_run call per case builds the
# model via the registry, runs() (or run_with_studentized_bootstrap()) ONCE, and computes
# confidence intervals ONCE; every assertion in the case reads the single returned dict. See
# fixtures/README.md's bootstrap schema for the full method list and tolerance policy.


def _dispatch_bootstrap(result: dict, method: str, args: list):
    if method == "statistic_lower_ci":
        return result["statistic_lower_ci"][int(args[0])]
    if method == "statistic_upper_ci":
        return result["statistic_upper_ci"][int(args[0])]
    if method == "parameter_lower_ci":
        return result["parameter_lower_ci"][int(args[0])]
    if method == "parameter_upper_ci":
        return result["parameter_upper_ci"][int(args[0])]
    if method == "population_estimate":
        return result["population_estimate"][int(args[0])]
    if method == "valid_count":
        return result["valid_count"][int(args[0])]
    if method == "replicate_value":
        return result["replicate_values"][int(args[0])][int(args[1])]
    raise KeyError(f"unknown bootstrap fixture method: {method}")


def _run_bootstrap_case(construct: dict, assertions: list, datasets: dict):
    dataset = [float(v) for v in datasets[construct["dataset"]]] if "dataset" in construct else []
    probabilities = [_num(v) for v in construct["probabilities"]]
    result = _core.bootstrap_run(
        construct["model"],
        construct.get("mu", 0.0),
        construct.get("sigma", 0.0),
        construct.get("sample_size", 0),
        probabilities,
        dataset,
        construct["replicates"],
        construct["seed"],
        construct.get("max_retries", 20),
        construct.get("run", "regular"),
        construct["ci_method"],
        construct.get("alpha", 0.1),
    )
    for a in assertions:
        args = a.get("args", [])
        actual = _dispatch_bootstrap(result, a["method"], args)
        _check(actual, a)


# --- model_estimation path -----------------------------------------------------------------
# Inherently STATEFUL like mcmc_sampler/bootstrap: one _core.estimation_run (ML/MAP),
# _core.estimation_bayes_run (BayesianAnalysis, T12), or _core.model_simulate (Simulation,
# M13) call per case builds the model, runs its one stateful call (estimate() or the seeded
# ISimulatable draw), and returns the full result surface; every assertion in the case reads
# that single cached dict. See fixtures/README.md's model_estimation section for the full
# method list and the `bic`/`chain_value` design notes. `bic` is the one exception to the
# "cached dict" contract for ML/MAP: it takes an actual sample size `n` (C#
# `GetBIC(sampleSize)`), read live from the fixture's `args[0]` at dispatch time via
# `_core.estimation_bic`, not precomputed alongside the rest.
#
# M13: `construct.model` is no longer a flat {family, dataset} pair -- it can name any of the
# four Phase 5 model types, a full censored DataFrame, nonstationary trend specs, and explicit
# parameter values. The parsed spec is re-serialized with json.dumps (which round-trips
# doubles exactly) and handed to the SHARED C++ builder (corehydro/models/model_spec.hpp), the
# same code path the C++ runner and the R glue use; only the `dataset` reference is still
# resolved here, like every other fixture kind.


def _fit_runner_target(target: str) -> str:
    """Fixture target name -> the name run_fit/run_fit_diagnostics dispatch on.

    The only one that differs is GMM, whose fixture-level target spells out the C# class name.
    """
    return "GMM" if target == "GeneralizedMethodOfMoments" else target


def _full_construct_json(construct: dict) -> str:
    """The case's FULL construct, in the shape the shared fit runner reads (Task 9).

    The fixture's own construct with the `settings` sub-object hoisted to the top level -- where
    `apply_bayesian_settings` looks for the Bayesian knobs. Every other key passes through
    untouched and `run_fit` ignores the ones its target does not use. C++ and R assemble it
    identically, so all three hand the runner byte-identical constructs.
    """
    full = {k: v for k, v in construct.items() if k != "settings"}
    full.update(construct.get("settings", {}))
    return json.dumps(full)


def _dispatch_estimation(
    result: dict,
    method: str,
    args: list,
    target: str,
    model_json: str,
    data: list,
    optimizer: str,
    construct: dict,
):
    # GMM (B11): j_stat/j_stat_pval come from the cached run dict; quantile_variance takes a
    # per-assertion AEP, so -- exactly like `bic`'s per-assertion sample size -- it rebuilds the
    # deterministic fit live via estimation_gmm_qvar. parameter/standard_error/covariance/
    # correlation/simulated_value reuse the shared arms below (the GMM run returns the same keys).
    if method == "j_stat":
        return result["j_stat"]
    if method == "j_stat_pval":
        return result["j_stat_pval"]
    if method == "gmm_iterations":
        return result["gmm_iterations"]
    if method == "converged_within_tolerance":
        return 1.0 if result["converged_within_tolerance"] else 0.0
    if method == "optimizer_fallback_count":
        return result["optimizer_fallback_count"]
    if method == "quantile_variance":
        return _core.estimation_gmm_qvar(
            model_json,
            data,
            construct.get("strategy", "Iterative"),
            construct.get("optimizer", "BFGS"),
            int(construct.get("max_gmm_iterations", -1)),
            float(args[0]),
        )
    if method == "parameter":
        return result["parameters"][int(args[0])]
    if method == "max_log_likelihood":
        return result["max_log_likelihood"]
    if method == "aic":
        return result["aic"]
    if method == "bic":
        return _core.estimation_bic(target, model_json, data, optimizer, int(args[0]))
    if method == "covariance":
        return result["covariance"][int(args[0])][int(args[1])]
    if method == "standard_error":
        return result["standard_errors"][int(args[0])]
    if method == "correlation":
        return result["correlation"][int(args[0])][int(args[1])]
    if method == "dic":
        return result["dic"]
    if method == "waic":
        return result["waic"]
    if method == "looic":
        return result["looic"]
    if method == "posterior_mean":
        return result["posterior_mean"][int(args[0])]
    if method == "chain_value":
        return result["chains"][int(args[0])][int(args[1])][int(args[2])]
    if method == "simulated_value":
        # The seeded ISimulatable draw cached by _core.model_simulate (M13).
        return result["simulated"][int(args[0])]
    # The M14 DataFrame surface (works under any target -- it reads the model, not the
    # estimator): lazily build the frame surface ONCE per case via _core.model_data_frame
    # and memoize it in the case's result dict (the bic lazy-rebuild precedent).
    if method in ("number_of_low_outliers", "low_outlier_threshold", "plotting_position"):
        if "_data_frame" not in result:
            result["_data_frame"] = _core.model_data_frame(model_json, data)
        frame = result["_data_frame"]
        if method == "plotting_position":
            # plotting_position [kind, i]: kind is "exact" | "interval" | "uncertain".
            return frame[f"pp_{args[0]}"][int(args[1])]
        return frame[method]
    # The Validate surface (Task 16, works under any target -- it reads the model, not the
    # estimator): lazily build + memoize _core.model_validate's result in the case's result
    # dict (the _data_frame lazy-rebuild precedent). validation_message_contains is a
    # structural substring check, not a byte-exact C# message pin (see test_fixtures.cpp's
    # dispatch_model_validate for the rationale).
    if method in ("is_valid", "validation_message_contains"):
        if "_validate" not in result:
            result["_validate"] = _core.model_validate(model_json, data)
        validation = result["_validate"]
        if method == "is_valid":
            return 1.0 if validation["is_valid"] else 0.0
        return 1.0 if any(args[0] in m for m in validation["messages"]) else 0.0
    # --- Task 9: the wider fit surface -------------------------------------------------------
    # These read a lazily-built + memoized _core.fit_run over the case's FULL construct (the
    # fixture construct with `settings` hoisted to the top level -- see _full_construct_json).
    # The narrow estimation_run/estimation_bayes_run result above is deliberately untouched: it
    # backs the pinned oracles and does not carry these fields. Both calls run the same
    # deterministic fit, so this is the `bic` lazy-rebuild precedent, not a second, different fit.
    if method in (
        "profile_lower",
        "profile_upper",
        "profile_value",
        "function_evaluations",
        "status_is",
        "nobs",
        "prior_log_likelihood",
        "rhat",
        "ess",
        "acceptance_rate",
        "posterior_median",
        "posterior_sd",
        "posterior_lower",
        "posterior_upper",
    ):
        if "_fit" not in result:
            result["_fit"] = _core.fit_run(
                _fit_runner_target(target), _full_construct_json(construct), data
            )
        fit = result["_fit"]
        if method == "profile_lower":
            return fit["profile_lower"][int(args[0])]
        if method == "profile_upper":
            return fit["profile_upper"][int(args[0])]
        if method == "profile_value":
            # profile_value [param, bin, col]: profile_grid is n_params x bins x 2, row-major,
            # col 0 = the parameter value at the bin midpoint, col 1 = the profile log-likelihood.
            idx = (int(args[0]) * fit["profile_bins"] + int(args[1])) * 2 + int(args[2])
            return fit["profile_grid"][idx]
        if method == "function_evaluations":
            return fit["function_evaluations"]
        if method == "status_is":
            # 1.0 when the optimizer status matches, else 0.0 (the validation_message_contains
            # boolean-as-double precedent -- the fixture schema carries no string comparison).
            return 1.0 if fit["status"] == args[0] else 0.0
        if method == "nobs":
            return fit["nobs"]
        if method == "prior_log_likelihood":
            return fit["prior_log_likelihood"]
        if method == "acceptance_rate":
            return fit["acceptance_rates"][int(args[0])]
        key = {
            "rhat": "rhat",
            "ess": "ess",
            "posterior_median": "summary_median",
            "posterior_sd": "summary_sd",
            "posterior_lower": "summary_lower",
            "posterior_upper": "summary_upper",
        }[method]
        return fit[key][int(args[0])]
    # The PSIS-LOO Pareto-k surface lives on FitDiagnostics (the InfluenceDiagnostics wrapper),
    # not on the fit result, so it takes the second lazily-memoized runner call.
    if method in ("pareto_k", "max_pareto_k"):
        if "_fit_diagnostics" not in result:
            result["_fit_diagnostics"] = _core.fit_diagnostics(
                _fit_runner_target(target), _full_construct_json(construct), data
            )
        diagnostics = result["_fit_diagnostics"]
        if method == "pareto_k":
            return diagnostics["pareto_k"][int(args[0])]
        return diagnostics["max_pareto_k"]
    raise KeyError(f"unknown model_estimation fixture method: {method}")


def _run_estimation_case(target: str, construct: dict, assertions: list, datasets: dict):
    model = construct["model"]
    # Re-serialize the parsed spec for the shared C++ builder (see the path comment above).
    model_json = json.dumps(model)
    data = [float(v) for v in datasets[model["dataset"]]] if "dataset" in model else []

    if target == "Simulation":
        draws = _core.model_simulate(
            model_json, data, int(construct["sample_size"]), int(construct.get("seed", -1))
        )
        result = {"simulated": draws}
        optimizer = ""
    elif target == "Validate":
        # Builds the model only (no estimator, no draw); every assertion reads the
        # lazily-memoized _validate entry (see _dispatch_estimation above).
        result = {}
        optimizer = ""
    elif target == "BayesianAnalysis":
        sampler = construct.get("sampler", "DEMCzs")
        settings = construct.get("settings", {})
        result = _core.estimation_bayes_run(model_json, data, sampler, settings)
        optimizer = ""
    elif target == "GeneralizedMethodOfMoments":
        # GMM (B11): a bulletin17c model fit by GMM. One stateful estimate()+post_process, whose
        # full surface is cached here; quantile_variance rebuilds live at dispatch (see above).
        result = _core.estimation_gmm_run(
            model_json,
            data,
            construct.get("strategy", "Iterative"),
            construct.get("optimizer", "BFGS"),
            int(construct.get("max_gmm_iterations", -1)),
            int(construct.get("sample_size", 0)),
            int(construct.get("seed", -1)),
        )
        optimizer = ""
    else:
        optimizer = construct.get("optimizer", "DifferentialEvolution")
        # P3: an optional seeded-draw digest off the FITTED model (sample_size + seed) lets one
        # MLE smoke file cover parameter + max_log_likelihood + a seeded simulated_value.
        result = _core.estimation_run(
            target,
            model_json,
            data,
            optimizer,
            int(construct.get("sample_size", 0)),
            int(construct.get("seed", -1)),
        )

    for a in assertions:
        args = a.get("args", [])
        actual = _dispatch_estimation(
            result, a["method"], args, target, model_json, data, optimizer, construct
        )
        _check(actual, a)


# --- analysis path (Phase 8: user-facing Analyses layer) ------------------------------------
# Stateful like model_estimation: one _core.analysis_* call per case builds + runs the analysis
# and returns the full result surface; every assertion reads that single cached dict. The
# construct fields map 1:1 onto the glue args, so R/Python/C++ build byte-identical analyses.


def _dispatch_analysis(result: dict, method: str, args: list):
    if method == "candidate_count":
        return len(result["aic"])
    if method == "candidate_aic":
        return result["aic"][int(args[0])]
    if method == "candidate_bic":
        return result["bic"][int(args[0])]
    if method == "candidate_rmse":
        return result["rmse"][int(args[0])]
    if method == "candidate_converged":
        return 1.0 if result["converged"][int(args[0])] else 0.0
    if method == "parameter":
        return result["parameters"][int(args[0])]
    if method == "mode_curve":
        return result["mode_curve"][int(args[0])]
    if method == "mean_curve":
        return result["mean_curve"][int(args[0])]
    if method == "lower_ci":
        return result["lower_ci"][int(args[0])]
    if method == "upper_ci":
        return result["upper_ci"][int(args[0])]
    if method == "exceedance_probability":
        return result["exceedance_probabilities"][int(args[0])]
    if method == "point_estimate":
        return result["point_estimates"][int(args[0])]
    if method == "beta1":
        return result["beta1"][int(args[0])]
    if method == "nu":
        return result["nu"][int(args[0])]
    if method == "quantile_variance":
        return result["quantile_variance"][int(args[0])]
    if method in ("aic", "bic", "dic", "rmse", "confidence_level"):
        return result[method]
    # D5: time-series curve length + the three diagnostics.
    if method == "curve_length":
        return len(result["mode_curve"])
    if method == "leverage_count":
        return len(result["leverage"]["leverage"])
    if method == "leverage_prior_count":
        return len(result["leverage"]["prior_leverage"])
    if method in ("total_leverage", "total_fit_influence", "total_variance_influence"):
        return result["leverage"][method]
    if method == "obs_leverage":
        return result["leverage"]["leverage"][int(args[0])]
    if method == "obs_fit_influence":
        return result["leverage"]["fit_influence"][int(args[0])]
    if method == "obs_variance_influence":
        return result["leverage"]["variance_influence"][int(args[0])]
    if method == "obs_value":
        return result["leverage"]["value"][int(args[0])]
    if method == "influence_count":
        return result["influence"]["count"]
    if method in (
        "mean_pareto_k",
        "max_pareto_k",
        "count_pareto_k_above_05",
        "count_pareto_k_above_07",
        "count_pareto_k_above_10",
        "proportion_problematic",
    ):
        return result["influence"][method]
    if method == "is_reliable":
        return 1.0 if result["influence"]["is_reliable"] else 0.0
    if method == "pareto_k":
        return result["influence"]["pareto_k"][int(args[0])]
    if method == "elpd_loo":
        return result["influence"]["elpd_loo"][int(args[0])]
    if method == "prior_influence_count":
        return result["prior_influence"]["count"]
    if method in (
        "total_prior_log_likelihood",
        "total_data_log_likelihood",
        "prior_to_data_ratio",
        "mean_prior_precision_share",
    ):
        return result["prior_influence"][method]
    if method == "is_prior_influential":
        return 1.0 if result["prior_influence"]["is_prior_influential"] else 0.0
    # X11 extended analyses (composite / spatial_gev / bivariate / coincident / rating_curve /
    # bootstrap / prior + posterior predictive).
    if method == "z_output":
        return result["z_output_values"][int(args[0])]
    if method == "z_output_length":
        return len(result["z_output_values"])
    if method == "site_count":
        return result["site_count"]
    if method == "site_location_mean":
        return result["site_location_mean"][int(args[0])]
    if method == "site_scale_mean":
        return result["site_scale_mean"][int(args[0])]
    if method == "site_shape_mean":
        return result["site_shape_mean"][int(args[0])]
    if method == "site_quantile_mean":
        return result["site0_quantile_mean"][int(args[0])]
    if method in ("cv_mae", "cv_rmse", "cv_mean_bias"):
        return result[method]
    if method in ("mean_p_value", "sd_p_value", "skewness_p_value", "min_p_value", "max_p_value"):
        return result[method]
    if method == "predictive_replicates":
        return result["number_of_replicates"]
    if method == "has_misfit":
        return result["has_misfit"]
    if method == "number_of_valid_draws":
        return result["number_of_valid_draws"]
    if method == "summary_mean_quantile":
        return result["summary_mean_quantiles"][int(args[0])]
    if method == "summary_sd_quantile":
        return result["summary_sd_quantiles"][int(args[0])]
    if method == "summary_min_quantile":
        return result["summary_min_quantiles"][int(args[0])]
    if method == "summary_max_quantile":
        return result["summary_max_quantiles"][int(args[0])]
    # T19: BootstrapDiagnostics (Bulletin17CAnalysis, Bootstrap/BiasCorrectedBootstrap).
    if method == "boot_has_results":
        return 1.0 if result["bootstrap"]["has_results"] else 0.0
    if method in (
        "boot_total_replicates",
        "boot_attempted_replicates",
        "boot_failed_replicates",
        "boot_valid_replicates",
        "boot_retained_replicates",
        "boot_failure_rate",
        "boot_total_retries",
        "boot_average_retries",
        "boot_pivot_rejections",
        "boot_mahalanobis_rejections",
        "boot_transform_failures",
        "boot_status_success_count",
        "boot_status_max_iterations_count",
        "boot_status_max_function_evaluations_count",
        "boot_status_failure_count",
        "boot_status_none_count",
        "boot_optimizer_fallbacks",
    ):
        return result["bootstrap"][method[len("boot_") :]]
    raise KeyError(f"unknown analysis fixture method: {method}")


# X11: analysis fixture targets routed through the shared analysis_extended_run dispatch.
_EXTENDED_ANALYSIS_TARGETS = frozenset(
    {
        "CompositeAnalysis",
        "SpatialGEVAnalysis",
        "BivariateAnalysis",
        "CoincidentFrequencyAnalysis",
        "RatingCurveAnalysis",
        "BootstrapAnalysis",
        "PriorPredictiveCheck",
        "PosteriorPredictiveCheck",
    }
)


# D5: map an analysis fixture target to the analysis_family_run analysis_type discriminator.
_FAMILY_ANALYSIS_TYPE = {
    "MixtureAnalysis": "mixture",
    "CompetingRiskAnalysis": "competing_risk",
    "PointProcessAnalysis": "point_process",
    "ARAnalysis": "ar",
    "MAAnalysis": "ma",
    "ARIMAAnalysis": "arima",
    "ARIMAXAnalysis": "arimax",
}


def _run_analysis_case(target: str, construct: dict, assertions: list, datasets: dict):
    ep = [float(v) for v in construct.get("exceedance_probabilities", [])]
    if target == "FittingAnalysis":
        data = [float(v) for v in datasets[construct["dataset"]]]
        result = _core.analysis_fit_distributions(data)
    elif target == "UnivariateAnalysis":
        model = construct["model"]
        model_json = json.dumps(model)
        data = [float(v) for v in datasets[model["dataset"]]]
        result = _core.analysis_univariate_run(
            model_json,
            data,
            construct.get("sampler", "DEMCzs"),
            int(construct.get("iterations", 3000)),
            int(construct.get("output_length", 10000)),
            float(construct.get("credible_level", 0.90)),
            int(construct.get("seed", 12345)),
            ep,
            int(construct.get("thinning_interval", -1)),
        )
    elif target == "Bulletin17CAnalysis":
        model = construct["model"]
        model_json = json.dumps(model)
        # T19: an inline `data_frame` (mixed exact/interval/threshold/uncertain series) is valid
        # without a `dataset` reference -- mirrors the C++ runner's guard so a Bulletin17CAnalysis
        # case can force low outliers / censored data onto the parent frame.
        data = [float(v) for v in datasets[model["dataset"]]] if "dataset" in model else []
        result = _core.analysis_b17c_run(
            model_json,
            data,
            construct.get("uncertainty_method", "MultivariateNormal"),
            int(construct.get("output_length", 10000)),
            int(construct.get("seed", 12345)),
            float(construct.get("confidence_level", 0.90)),
            ep,
        )
    elif target in _FAMILY_ANALYSIS_TYPE:
        # D5: the seven per-family analyses through the single dispatch binding.
        model = construct["model"]
        model_json = json.dumps(model)
        data = [float(v) for v in datasets[model["dataset"]]]
        result = _core.analysis_family_run(
            _FAMILY_ANALYSIS_TYPE[target],
            model_json,
            data,
            construct.get("sampler", "DEMCzs"),
            int(construct.get("iterations", 3000)),
            int(construct.get("output_length", 10000)),
            float(construct.get("credible_level", 0.90)),
            int(construct.get("seed", 12345)),
            ep,
            int(construct.get("thinning_interval", -1)),
            int(construct.get("training_time_steps", -1)),
            int(construct.get("forecasting_time_steps", 0)),
        )
    elif target == "Diagnostics":
        # D5: leverage / influence / prior-influence diagnostics off a BayesianAnalysis fit.
        model = construct["model"]
        model_json = json.dumps(model)
        data = [float(v) for v in datasets[model["dataset"]]]
        result = _core.analysis_diagnostics_run(
            model_json,
            data,
            construct.get("sampler", "DEMCzs"),
            int(construct.get("iterations", 3000)),
            int(construct.get("output_length", 10000)),
            int(construct.get("seed", 12345)),
            int(construct.get("thinning_interval", -1)),
            int(construct.get("thin_every", 10)),
        )
    elif target in _EXTENDED_ANALYSIS_TARGETS:
        # X11: the five remaining analyses + bootstrap + predictive checks. Re-serialize the whole
        # construct + datasets and call the single shared dispatch binding (byte-identical path to
        # the C++ runner and the R twin).
        result = _core.analysis_extended_run(target, json.dumps(construct), json.dumps(datasets))
    else:
        raise KeyError(f"unknown analysis target: {target}")

    for a in assertions:
        args = a.get("args", [])
        actual = _dispatch_analysis(result, a["method"], args)
        _check(actual, a)


# --- Shared assertion checking ---------------------------------------------------------


def _check(actual, a):
    mode, exp = a["mode"], a.get("expected")
    if mode == "bool":
        assert bool(actual) is bool(exp)
    elif mode == "equal":
        e = _num(exp)
        if math.isnan(e):
            assert math.isnan(actual)
        else:
            assert actual == e
    elif mode == "abs":
        assert abs(actual - exp) <= a["tol"]
    elif mode == "rel":
        assert abs(actual - exp) / abs(exp) <= a["tol"]
    else:
        raise KeyError(f"unknown comparison mode: {mode}")


# --- bivariate_copula path ---------------------------------------------------------------
# Every copula shares BivariateCopula's uniform theta/get_copula_parameters/pdf/cdf/... API
# (unlike multivariate_distribution's Dirichlet/Multinomial/BivariateEmpirical/..., which
# share no common surface), so this path is fully generic through the factory-driven
# _core.cop_val/_core.cop_fit bindings in copula.cpp -- no per-target branching, mirroring
# copula_factory.hpp's rationale. construct is either {"theta": x} (optionally {"theta": x,
# "df": y} for 2-parameter copulas, and/or {"marginals": {"targets", "params"}} to attach
# marginals directly -- used by the "random_value" sampling oracles) or {"fit": {"x", "y",
# "method", "marginals"?}}; see fixtures/README.md for the full schema.


def _build_copula_params(construct: dict) -> list[float]:
    p = [_num(construct["theta"])]
    if "df" in construct:
        p.append(_num(construct["df"]))
    return p


def _dispatch_copula(
    target: str,
    params: list[float],
    method: str,
    args: list,
    marg_x_target: str = "",
    marg_x_params: list[float] | None = None,
    marg_y_target: str = "",
    marg_y_params: list[float] | None = None,
):
    ar = _flatten_mv_args(args)
    return _core.cop_val(
        target, params, method, ar, marg_x_target, marg_x_params or [], marg_y_target, marg_y_params or []
    )


def _run_copula_case(target: str, construct: dict, assertions: list, datasets: dict):
    if "fit" in construct:
        fit = construct["fit"]
        x = [float(v) for v in datasets[fit["x"]]]
        y = [float(v) for v in datasets[fit["y"]]]
        marginals = fit.get("marginals")
        marg_x = marginals[0] if marginals else ""
        marg_y = marginals[1] if marginals else ""
        result = _core.cop_fit(target, x, y, fit["method"], marg_x, marg_y)
        for a in assertions:
            args = a.get("args", [])
            if a["method"] == "theta":
                actual = result["params"][0]
            elif a["method"] == "df":
                actual = result["params"][1]
            elif a["method"] == "marginal_param":
                which, idx = args[0], int(args[1])
                actual = result["marg_x_params"][idx] if which == "x" else result["marg_y_params"][idx]
            else:
                raise KeyError(f"unsupported post-fit copula fixture method: {a['method']}")
            _check(actual, a)
    else:
        params = _build_copula_params(construct)
        marg = construct.get("marginals")
        marg_x_target = marg["targets"][0] if marg else ""
        marg_y_target = marg["targets"][1] if marg else ""
        marg_x_params = [_num(v) for v in marg["params"][0]] if marg else []
        marg_y_params = [_num(v) for v in marg["params"][1]] if marg else []
        for a in assertions:
            args = a.get("args", [])
            actual = _dispatch_copula(
                target, params, a["method"], args, marg_x_target, marg_x_params, marg_y_target, marg_y_params
            )
            _check(actual, a)


# data_utility [function, args, data]: MGBT count, Box-Cox / Yeo-Johnson lambda +
# transform, plotting positions, Latin hypercube. Mirrors dispatch_data_utility in
# core/tests/test_fixtures.cpp.
def _dispatch_data_utility(fn, args, data):
    if fn == "MGBT":
        return float(_core.mgbt_test(data))
    if fn == "BoxCoxLambda":
        return _core.box_cox_lambda(data)
    if fn == "BoxCoxTransform":
        return _core.box_cox(data, args[0])[int(args[1])]
    if fn == "YeoJohnsonLambda":
        return _core.yeo_johnson_lambda(data)
    if fn == "YeoJohnsonTransform":
        return _core.yeo_johnson(data, args[0])[int(args[1])]
    if fn == "PlottingPosition":
        return _core.plotting_positions_alpha(int(args[0]), args[1])[int(args[2])]
    if fn in ("LHSRandom", "LHSMedian"):
        # args: [sample_size, dimension, seed, row, col]
        gen = _core.latin_hypercube_median if fn == "LHSMedian" else _core.latin_hypercube
        m = gen(int(args[0]), int(args[1]), int(args[2]))
        return m[int(args[3])][int(args[4])]
    if fn.startswith("MRL") or fn.startswith("GPDStability"):
        return _dispatch_threshold_diagnostic(fn, args, data)
    raise KeyError(f"unknown data_utility function: {fn}")


# Threshold-selection diagnostics: both methods share one glue call and differ only in which
# field the function name selects. args are [u_min, u_max, n_thresholds, confidence_level,
# point_index]; `*PointCount` ignores the index and returns how many candidate thresholds
# survived the minimum-exceedance and fit filters.
_TD_FIELDS = {
    "MRLThreshold": "threshold",
    "MRLMeanExcess": "mean_excess",
    "MRLLowerCI": "lower_ci",
    "MRLUpperCI": "upper_ci",
    "MRLCount": "exceedance_count",
    "GPDStabilityThreshold": "threshold",
    "GPDStabilityModifiedScale": "modified_scale",
    "GPDStabilityModifiedScaleLowerCI": "modified_scale_lower_ci",
    "GPDStabilityModifiedScaleUpperCI": "modified_scale_upper_ci",
    "GPDStabilityShape": "shape",
    "GPDStabilityShapeLowerCI": "shape_lower_ci",
    "GPDStabilityShapeUpperCI": "shape_upper_ci",
    "GPDStabilityCount": "exceedance_count",
}


def _dispatch_threshold_diagnostic(fn, args, data):
    method = "mean_residual_life" if fn.startswith("MRL") else "parameter_stability"
    r = _core.threshold_diagnostics(data, method, args[0], args[1], int(args[2]), args[3])
    if fn in ("MRLPointCount", "GPDStabilityPointCount"):
        return float(len(r["threshold"]))
    if fn not in _TD_FIELDS:
        raise KeyError(f"unknown data_utility function: {fn}")
    return float(r[_TD_FIELDS[fn]][int(args[4])])


def _load_cases():
    out = []
    for fx in sorted(_fixtures_dir().rglob("*.json")):
        spec = json.loads(fx.read_text())
        # Only validate univariate_distribution / multivariate_distribution /
        # bivariate_copula / mcmc_sampler fixtures; skip other kinds (e.g.
        # special_function) which are validated in C++ only and are not exposed to the
        # Python package.
        kind = spec.get("kind")
        if kind not in (
            "univariate_distribution",
            "multivariate_distribution",
            "bivariate_copula",
            "mcmc_sampler",
            "bootstrap",
            "model_estimation",
            "analysis",
            "data_utility",
        ):
            continue
        for case in spec["cases"]:
            out.append((kind, spec.get("target", kind), spec.get("datasets", {}), case))
    return out


CASES = _load_cases()


@pytest.mark.parametrize(
    "kind,target,datasets,case", CASES, ids=[f"{k}:{t}:{c['name']}" for k, t, _, c in CASES]
)
def test_fixture_case(kind, target, datasets, case):
    if kind == "model_estimation":
        _run_estimation_case(target, case["construct"], case["assertions"], datasets)
        return

    if kind == "analysis":
        _run_analysis_case(target, case["construct"], case["assertions"], datasets)
        return

    if kind == "bootstrap":
        _run_bootstrap_case(case["construct"], case["assertions"], datasets)
        return

    if kind == "data_utility":
        args = [float(v) for v in case.get("args", [])]
        # _num (not a raw list()) so a "nan"/"inf"/"-inf" string literal inside a dataset
        # (the v2.1.4 FitLambda invalid-sample cases) parses instead of reaching pybind11
        # as an unconvertible str.
        data = [_num(v) for v in datasets[case["dataset"]]] if "dataset" in case else []
        actual = _dispatch_data_utility(case["function"], args, data)
        for a in case["assertions"]:
            _check(actual, a)
        return

    if kind == "mcmc_sampler":
        _run_mcmc_case(target, case["construct"], case["assertions"], datasets)
        return

    if kind == "bivariate_copula":
        _run_copula_case(target, case["construct"], case["assertions"], datasets)
        return

    if kind == "multivariate_distribution":
        if target == "MultivariateNormal":
            _run_mvn_case(case["construct"], case["assertions"])
        else:
            for a in case["assertions"]:
                args = a.get("args", [])
                actual = _dispatch_multivariate(target, case["construct"], a["method"], args)
                _check(actual, a)
        return

    is_gev = target == "GeneralizedExtremeValue"
    is_composite = target in _COMPOSITE_TARGETS
    if is_gev:
        g = _build_gev(case["construct"], datasets)
    elif is_composite:
        cd = _build_composite(target, case["construct"], datasets)
    else:
        params = _build_params(target, case["construct"], datasets)
    for a in case["assertions"]:
        args = a.get("args", [])
        if is_composite and a["method"] == "set_parameters":
            cd = _apply_set_parameters_composite(target, cd, args)
            actual = 0
        elif is_gev:
            actual = _dispatch_gev(g, a["method"], args)
        elif is_composite:
            actual = _dispatch_composite(target, cd, a["method"], args)
        else:
            actual = _dispatch_generic(target, params, a["method"], args)
        _check(actual, a)
