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


# --- Delegation to the shared distribution runner ----------------------------------------
#
# The fixture `construct` schema IS the dist_spec.hpp grammar, so the bridge only has to
# resolve a dataset NAME into an inline array, spell the handful of keys the two schemas
# disagree on, and hand the object to _core.dist_spec_run / copula_run / mvdist_run. Every
# value the runner produces is then pinned by exactly the corpus the bespoke glue was pinned
# by. This mirrors core/tests/test_fixtures.cpp and corehydror's test-fixtures.R case for case,
# so the C++, R and Python runners all reach the oracle through one shared code path.
#
# Two properties of the runner keep a narrow bespoke path alive; both are deliberate:
#
#   1. json_lite, the runner's JSON reader, has no NaN or Infinity literal, while the corpus
#      deliberately pins non-finite-PARAMETER validity cases: Empirical `p`, KernelDensity
#      `bandwidth`, every copula `theta`/`df`, and BivariateEmpirical `x1`/`x2`/`p`. Every such
#      case asserts nothing but `parameters_valid`, so each keeps a narrow glue call. The same
#      limit applies to a non-finite EVALUATION POINT, not just the construct: the
#      MultivariateNormal / MultivariateStudentT log_pdf-at-infinity cases and
#      r_mvtnorm_4d_sequential's infinite CDF bounds.
#
#   2. The runner is stateless by construction -- one call builds an object, evaluates once and
#      drops it -- and it exposes the user-facing verb set, not the whole fixture vocabulary.
#      Two groups therefore stay bespoke: methods with no runner counterpart (`mvndst` and its
#      two status arms, `log_multivariate_beta`, `cdf_xy(_after_set_parameters)`,
#      `dependency_change`), and MultivariateNormal's `cdf`/`interval` in a case that consumes
#      the persistent MVNUNI stream more than once, since those pin a SEQUENCE off one object.
#      A case that consumes the stream at most once delegates: `seed` is a grammar key, so a
#      rebuilt object starts the same stream. Only `cdf` has such a case today, so only `cdf`
#      has a batch entry point below.


def _has_non_finite(v) -> bool:
    if isinstance(v, str):
        return v in ("nan", "inf", "-inf")
    if isinstance(v, list):
        return any(_has_non_finite(e) for e in v)
    if isinstance(v, dict):
        return any(_has_non_finite(e) for e in v.values())
    return False


def _json(x) -> str:
    """Both of the runner's string inputs (the spec and the args array)."""
    return json.dumps(x)


# --- Composite distribution path -------------------------------------------------------
# TruncatedDistribution / Empirical / KernelDensity / Mixture / CompetingRisks: the fixture
# "construct" already IS the dist_spec.hpp grammar (which accepts `target`/`params` as aliases
# of `family`/`parameters`), so a composite case serializes straight through dist_spec_run.
# Adding a composite needs no change here at all.

_COMPOSITE_TARGETS = {"TruncatedDistribution", "Empirical", "KernelDensity", "Mixture",
                      "CompetingRisks"}


def _composite_spec(target: str, construct: dict, datasets: dict | None = None) -> dict:
    """The fixture construct as a dist_spec.hpp spec.

    "data" is the only dataset-by-name key in the univariate grammar (KernelDensity); "base"
    and "components" nest and carry no dataset reference.
    """
    spec = dict(construct)
    spec["family"] = target
    if isinstance(spec.get("data"), str):
        spec["data"] = list((datasets or {})[spec["data"]])
    return spec


def _fixture_method(method: str) -> str:
    """The fixture method vocabulary predates the runner's; map the differences here.

    `random_value` args are [sample_size, seed, index]: the runner's "random" reads only the
    first two and returns the whole draw, so the args pass through unchanged and _fixture_pick
    does the indexing.
    """
    if method == "param":
        return "parameters"
    if method == "random_value":
        return "random"
    return method


def _fixture_pick(result: dict, method: str, args: list):
    """`param` and `random_value` index into the vector the runner returns whole."""
    if method == "param":
        return result["values"][int(args[0])]
    if method == "random_value":
        return result["values"][int(args[2])]
    return result["values"][0]


def _dispatch_composite_local(target: str, construct: dict, datasets: dict, method: str,
                              args: list):
    """Limitations 1 and 2 for the composite path, and the only remaining callers of the
    bespoke composite glue in dist.cpp: Empirical's non-finite `p` and KernelDensity's
    non-finite `bandwidth` validity cases (which the grammar's JSON reader cannot encode),
    plus CompetingRisks' `dependency_change` (which has no runner counterpart).
    """
    if target == "Empirical" and method == "parameters_valid":
        xv = [_num(v) for v in construct["x"]]
        pv = [_num(v) for v in construct["p"]]
        pt = construct.get("p_transform", "NormalZ")
        # v2.1.4: p_descending DECLARES the probability order (mirrors C#'s explicit
        # `probabilityOrder` argument -- NOT auto-detected from the data); default False
        # matches the ordinary ascending-CDF case.
        pd = bool(construct.get("p_descending", False))
        return _core.emp_valid(xv, pv, pt, pd)
    if target == "KernelDensity" and method == "parameters_valid":
        data_vec = [float(v) for v in (datasets or {})[construct["data"]]]
        kernel = construct.get("kernel", "Gaussian")
        # A negative bandwidth means Silverman's rule; _num parses the "nan"/"inf" literal
        # these two cases exist to reject.
        bandwidth = _num(construct["bandwidth"]) if "bandwidth" in construct else -1.0
        bounded = bool(construct.get("bounded_by_data", True))
        return _core.kde_valid(data_vec, kernel, bandwidth, bounded)
    if target == "CompetingRisks" and method == "dependency_change":
        # v2.1.4: verifies the Dependency setter fix + PerfectlyNegative no longer zeroing
        # CorrelationMatrix, in ONE self-contained call -- args = [x, dependency2, i, j, field].
        ct = [c["target"] for c in construct["components"]]
        cp = [[_num(v) for v in c["params"]] for c in construct["components"]]
        min_rv = bool(construct.get("minimum_of_random_variables", True))
        dep = construct.get("dependency", "Independent")
        corr = [[float(v) for v in row] for row in construct.get("correlation", [])]
        x, dep2, i, j, field = args
        return _core.cr_dependency_change(ct, cp, min_rv, dep, dep2, corr, float(x), field,
                                          int(i), int(j))
    raise KeyError(
        f"composite {target}/{method} has no shared-runner path and no local one either")


def _run_composite_case(target: str, construct: dict, assertions: list, datasets: dict):
    # Limitation 1: a construct carrying a "nan"/"inf" literal cannot be serialized into the
    # grammar, so its (validity-only) assertions run locally.
    encodable = not _has_non_finite(construct)
    spec = _composite_spec(target, construct, datasets)
    for a in assertions:
        method = a["method"]
        args = a.get("args", [])
        # `dependency_change` has no runner counterpart (limitation 2); a non-finite evaluation
        # point is limitation 1 applied to the args rather than the construct.
        if not encodable or method == "dependency_change" or _has_non_finite(args):
            _check(_dispatch_composite_local(target, construct, datasets, method, args), a)
            continue
        if method == "set_parameters":
            # The runner is stateless, so a SetParameters round trip is carried on the spec:
            # every later assertion in this case rebuilds with it applied, which is what
            # dist_spec's "set_parameters" key exists for. A second call replaces the first,
            # exactly as the in-place mutation did. The 0 mirrors the old dispatcher's dummy
            # return, and the assertion is still CHECKED rather than skipped.
            spec["set_parameters"] = args
            _check(0, a)
            continue
        spec_json = _json(spec)
        if a["mode"] == "bool":
            # The old dispatcher ignored the assertion's method in bool mode and read
            # parameters_valid(); keep that exactly.
            _check(_core.dist_spec_run(spec_json, "parameters_valid", "[]")["values"][0], a)
        else:
            r = _core.dist_spec_run(spec_json, _fixture_method(method), _json(args))
            _check(_fixture_pick(r, method, args), a)


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
    if method in ("parameter_covariance", "quantile_variance", "quantile_gradient"):
        return _dispatch_standard_error(target, params, method, args)
    raise KeyError(f"unknown fixture method: {method}")


def _dispatch_standard_error(target, params, method, args):
    """The IStandardError surface.

    It has no bespoke `_core.dist_*` binding (nothing outside the fixtures calls it), so it
    routes through the shared dist_runner, which reaches it by the same capability cast the C++
    and R runners use. Args follow the flattened convention the bespoke GEV slice above already
    speaks -- parameter_covariance [sample_size, row, col], quantile_variance [probability,
    sample_size], quantile_gradient [probability, index] -- while the runner returns the whole
    matrix (row-major) or vector, so the indexing happens here. GeneralizedNormal is the only
    family reaching this today, and only through quantile_gradient: C# throws
    NotImplementedException for its other two, and the port mirrors that.
    """
    spec = _json({"family": target, "parameters": [float(v) for v in params]})
    if method == "quantile_variance":
        return _core.dist_spec_run(spec, method,
                                   _json([float(args[0]), float(args[1])]))["values"][0]
    values = _core.dist_spec_run(spec, method, _json([float(args[0])]))["values"]
    if method == "quantile_gradient":
        return values[int(args[1])]
    return values[int(args[1]) * len(params) + int(args[2])]


# --- multivariate_distribution path -----------------------------------------------------
# The only partly delegated path. _core.mvdist_run covers the verbs this phase exposes (see
# _mv_delegated below) and the rest of the pinned surface stays on _dispatch_multivariate,
# which keeps the bespoke _core.dirichlet_val / bve_* / mvn_* / mvt_val bindings in mvd.cpp:
# the MVNDST integrator internals, BivariateEmpirical's cdf_xy pair, Dirichlet's static
# log_multivariate_beta, the non-finite constructs and evaluation points, and the seeded MVNUNI
# sequences. Extending the runner's method table shrinks the bespoke half; nothing else moves.


def _mvdist_spec(target: str, construct: dict) -> dict:
    """The fixture construct as a dist_spec.hpp spec.

    The only schema difference is Multinomial's parameter spelling (n / p here, trials /
    probabilities there); the four MultivariateNormal integrator settings (seed /
    max_evaluations / abs_error / rel_error) are grammar keys and pass straight through.
    """
    spec = dict(construct)
    spec["family"] = target
    if target == "Multinomial":
        spec.pop("n", None)
        spec.pop("p", None)
        spec["trials"] = construct["n"]
        spec["probabilities"] = construct["p"]
    return spec


def _mvn_consumes_stream(method: str) -> bool:
    """MultivariateNormal's CDF above dimension 2, its Interval, and MVNDST itself all draw
    from the instance's persistent MVNUNI stream, so each call ADVANCES it. A case that makes
    more than one such call pins a sequence off one object, which a stateless runner cannot
    reproduce by construction.
    """
    return method in ("cdf", "interval", "mvndst", "mvndst_inform", "mvndst_error")


_MV_DELEGATED_METHODS = frozenset({
    "dimension", "pdf", "log_pdf", "cdf", "mahalanobis", "mean", "variance", "sd",
    "covariance", "median", "mode", "inverse_cdf", "interval", "degrees_of_freedom", "alpha",
    "alpha_sum", "number_of_trials", "random_value", "lhs_value", "marginal_dimension",
    "marginal_mean", "marginal_covariance", "marginal_log_pdf", "conditional_dimension",
    "conditional_mean", "conditional_covariance",
})


def _mv_delegated(target: str, method: str, mvn_stream_isolated: bool) -> bool:
    """Methods _core.mvdist_run covers.

    What is left on _dispatch_multivariate is exactly three groups: the MVNDST integrator
    internals (mvndst and its two status arms), BivariateEmpirical's
    cdf_xy(_after_set_parameters) and Dirichlet's static log_multivariate_beta, which have no
    runner verb; and MultivariateNormal's cdf/interval in a case that makes more than one
    stream-consuming call (`mvn_stream_isolated` False -- r_mvtnorm_4d_sequential's eleven
    advancing cdf values are the reason). With `seed` in the grammar a SINGLE such call
    reproduces exactly, so those cases delegate.
    """
    if method in ("mvndst", "mvndst_inform", "mvndst_error"):
        return False
    if target == "MultivariateNormal" and method in ("cdf", "interval") \
            and not mvn_stream_isolated:
        return False
    return method in _MV_DELEGATED_METHODS


def _square_dim(n: int) -> int:
    d = round(n ** 0.5)
    if d * d != n:
        raise ValueError("a covariance result is not a square matrix")
    return d


def _dispatch_multivariate_delegated(spec_json: str, method: str, args: list):
    """_core.mvdist_run returns whole vectors, so every fixture method that names one element
    indexes in here. Conventions preserved verbatim from the deleted dispatcher arms:
    mean/variance/sd/median/mode/alpha take [i]; covariance takes [i, j] against a row-major
    dimension^2 block; inverse_cdf takes [probabilities, i] and interval [lower, upper];
    random_value/lhs_value take [sample_size, seed, row, col] against a row-major
    sample_size x dimension block; marginal_* take [indices, ...] and conditional_* take
    [indices, values, ...], both evaluated against the child distribution the runner hands
    back as a spec.
    """
    def run(spec, m, a):
        return _core.mvdist_run(spec, m, _json(a))

    if method in ("dimension", "alpha_sum", "degrees_of_freedom", "number_of_trials"):
        return run(spec_json, method, [])["values"][0]
    if method in ("pdf", "log_pdf", "cdf", "mahalanobis"):
        return run(spec_json, method, args[0])["values"][0]
    if method == "inverse_cdf":
        return run(spec_json, method, args[0])["values"][int(args[1])]
    if method == "interval":
        return run(spec_json, "interval", list(args[0]) + list(args[1]))["values"][0]
    if method in ("mean", "variance", "sd", "median", "mode", "alpha"):
        return run(spec_json, method, [])["values"][int(args[0])]
    if method == "covariance":
        values = run(spec_json, "covariance", [])["values"]
        dim = _square_dim(len(values))
        return values[int(args[0]) * dim + int(args[1])]
    if method in ("random_value", "lhs_value"):
        n = int(args[0])
        values = run(spec_json, "random_lhs" if method == "lhs_value" else "random",
                     [args[0], args[1]])["values"]
        dim = len(values) // n
        return values[int(args[2]) * dim + int(args[3])]
    if method.startswith("marginal_") or method.startswith("conditional_"):
        marginal = method.startswith("marginal_")
        # `conditional` takes indices then values concatenated into one flat argument array,
        # so the trailing-argument base shifts by one.
        child_args = list(args[0]) if marginal else list(args[0]) + list(args[1])
        child = run(spec_json, "marginal" if marginal else "conditional", child_args)["spec"]
        base = 1 if marginal else 2
        leaf = method.split("_", 1)[1]
        if leaf == "dimension":
            return run(child, "dimension", [])["values"][0]
        if leaf == "log_pdf":
            return run(child, "log_pdf", args[base])["values"][0]
        if leaf == "mean":
            return run(child, "mean", [])["values"][int(args[base])]
        if leaf == "covariance":
            values = run(child, "covariance", [])["values"]
            dim = _square_dim(len(values))
            return values[int(args[base]) * dim + int(args[base + 1])]
        raise KeyError(f"unhandled child method: {method}")
    raise KeyError(f"method '{method}' is not delegated to mvdist_run")


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
    # Methods with a doubly-nested arg (cdf_xy_after_set_parameters's replacement probability
    # grid) can't flatten through _flatten_mv_args's "one nested vector, or all-scalar"
    # convention -- that branch below never references `ar`, so a flatten failure is harmless.
    try:
        ar = _flatten_mv_args(args)
    except (TypeError, ValueError):
        ar = None
    if target == "Dirichlet":
        alpha = [float(v) for v in construct["alpha"]]
        return _core.dirichlet_val(method, alpha, ar)
    if target == "BivariateEmpirical":
        x1 = [_num(v) for v in construct["x1"]]
        x2 = [_num(v) for v in construct["x2"]]
        p = [[_num(v) for v in row] for row in construct["p"]]
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
        return _core.mvn_val(method, mean, cov, ar)
    if target == "MultivariateStudentT":
        df = float(construct["df"])
        location = [float(v) for v in construct["location"]]
        scale = [[float(v) for v in row] for row in construct["scale"]]
        return _core.mvt_val(method, df, location, scale, ar)
    raise KeyError(f"unknown multivariate target: {target}")


# --- MultivariateNormal seeded batches --------------------------------------------------
# `cdf` (dim>=3) and `mvndst` both draw from the seeded MVNUNI stream, so a RUN of consecutive
# same-method assertions in a seeded case must be evaluated on ONE persistent instance via the
# mvn_*_seq bindings in mvd.cpp, not dispatched one call at a time (which would silently reset
# the seed between assertions). This is limitation 2 above, and it is why the delegated path
# hands `cdf` back here whenever a case makes more than one stream-consuming call.
# `interval` is NOT listed: the corpus's only interval case makes exactly one stream-consuming
# call, so it delegates through the grammar's `seed` key instead.

_MVN_SEEDED_METHODS = ("cdf", "mvndst")


def _dispatch_mvn_seeded_seq(construct: dict, method: str, run: list):
    seed = int(construct["seed"])
    mean = [float(v) for v in construct["mean"]]
    cov = [[float(v) for v in row] for row in construct["covariance"]]

    if method == "cdf":
        xs = [[_num(v) for v in a["args"][0]] for a in run]
        return _core.mvn_cdf_seq(mean, cov, seed, xs)
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


def _run_multivariate_case(target: str, construct: dict, assertions: list):
    # Limitation 1: a construct carrying a "nan"/"inf" literal cannot be serialized into the
    # grammar, so its (validity-only) assertions run locally.
    encodable = not _has_non_finite(construct)
    spec_json = _json(_mvdist_spec(target, construct)) if encodable else ""
    # A case whose MVNUNI stream is consumed at most once has no sequence to preserve, so its
    # cdf/interval delegates; anything more stays whole on the seeded batch path below.
    stream_calls = (sum(_mvn_consumes_stream(a["method"]) for a in assertions)
                    if target == "MultivariateNormal" else 0)
    stream_isolated = stream_calls <= 1
    seeded = "seed" in construct
    i = 0
    n = len(assertions)
    while i < n:
        a = assertions[i]
        method = a["method"]
        args = a.get("args", [])
        # Limitation 1 again, on the evaluation point rather than the construct: MVN's and
        # MVT's log_pdf-at-infinity cases pass an infinite coordinate, which the grammar
        # cannot carry either.
        if encodable and not _has_non_finite(args):
            if a["mode"] == "bool":
                # The old dispatcher ignored the assertion's method in bool mode and read
                # parameters_valid(); keep that exactly.
                _check(_core.mvdist_run(spec_json, "parameters_valid", "[]")["values"][0], a)
                i += 1
                continue
            if _mv_delegated(target, method, stream_isolated):
                _check(_dispatch_multivariate_delegated(spec_json, method, args), a)
                i += 1
                continue
        if target == "MultivariateNormal" and seeded and method in _MVN_SEEDED_METHODS:
            j = i
            while j < n and assertions[j]["method"] == method:
                j += 1
            run = assertions[i:j]
            actuals = _dispatch_mvn_seeded_seq(construct, method, run)
            for actual, assertion in zip(actuals, run):
                _check(actual, assertion)
            i = j
            continue
        _check(_dispatch_multivariate(target, construct, method, args), a)
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


def _full_construct_json(construct: dict, target: str) -> str:
    """The case's FULL construct, in the shape the shared fit runner reads (Task 9).

    The fixture's own construct with the `settings` sub-object hoisted to the top level -- where
    `apply_bayesian_settings` looks for the Bayesian knobs. Every other key passes through
    untouched and `run_fit` ignores the ones its target does not use. C++ and R assemble it
    identically, so all three hand the runner byte-identical constructs.

    fit_runner.hpp's run_fit/run_fit_diagnostics default `optimizer` to DifferentialEvolution for
    every target, including GMM -- but the narrow GMM path (_run_estimation_case below) defaults
    it to BFGS (matching the C# GMM ctor default). Without this, a GMM case that omits `optimizer`
    and asserts one of the wider fit-surface methods would read that method off a
    DifferentialEvolution fit while parameter/j_stat came from a BFGS fit. Write the same BFGS
    default here so both paths agree.
    """
    full = {k: v for k, v in construct.items() if k != "settings"}
    full.update(construct.get("settings", {}))
    if target == "GeneralizedMethodOfMoments" and "optimizer" not in full:
        full["optimizer"] = "BFGS"
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
                _fit_runner_target(target), _full_construct_json(construct, target), data
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
                _fit_runner_target(target), _full_construct_json(construct, target), data
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
# Fully delegated to _core.copula_run: every copula shares BivariateCopula's uniform
# theta/get_copula_parameters/pdf/cdf/... API, so there is no per-target branching left here at
# all (the "tau" method-of-moments fit, whose SetThetaFromTau is a member of each concrete
# Archimedean class rather than of IBivariateCopula, is dispatched by
# copulas::set_theta_from_tau inside dist_spec.hpp). The one local path is limitation 1: the
# non-finite theta/df validity cases, which the grammar cannot encode and which keep the narrow
# _core.cop_val call.
#
# construct is either {"theta": x} (optionally {"theta": x, "df": y} for 2-parameter copulas,
# and/or {"marginals": {"targets", "params"}} to attach marginals directly -- used by the
# "random_value" sampling oracles) or {"fit": {"x", "y", "method", "marginals"?}}; see
# fixtures/README.md for the full schema.


def _copula_margin(family: str, params) -> dict:
    return {"family": family} if params is None else {"family": family, "parameters": params}


def _copula_spec(target: str, construct: dict, datasets: dict) -> dict:
    """The fixture copula construct as a dist_spec.hpp spec.

    The schema differences are the marginal spelling (positional "marginals" here, margin_x /
    margin_y there) and the fit samples (dataset NAMES here, inline arrays there). The
    fixture's bare-family marginal convention needs no translation: dist_spec's build_copula
    MLE-fits a parameterless marginal to its own sample, which is what the fixture means and
    what IFM requires.
    """
    spec = dict(construct)
    spec["family"] = target
    marg = construct.get("marginals")
    if marg is not None:
        spec.pop("marginals", None)
        spec["margin_x"] = _copula_margin(marg["targets"][0], marg["params"][0])
        spec["margin_y"] = _copula_margin(marg["targets"][1], marg["params"][1])
    fit = construct.get("fit")
    if fit is not None:
        f = dict(fit)
        f["x"] = list(datasets[fit["x"]])
        f["y"] = list(datasets[fit["y"]])
        marginals = fit.get("marginals")
        if marginals is not None:
            f.pop("marginals", None)
            f["margin_x"] = _copula_margin(marginals[0], None)
            f["margin_y"] = _copula_margin(marginals[1], None)
        spec["fit"] = f
    return spec


_COPULA_METHOD_ALIASES = {
    "upper_tail_dependence": "tail_dependence",
    "lower_tail_dependence": "tail_dependence",
    "theta_minimum": "bounds",
    "theta_maximum": "bounds",
    "or_exceedance": "exceedance_or",
    "and_exceedance": "exceedance_and",
    "random_value": "random",
}

_COPULA_LOG_LIKELIHOODS = ("log_likelihood_pseudo", "log_likelihood_ifm", "log_likelihood_full")


def _fixture_copula_method(method: str) -> str:
    # pdf, log_pdf, cdf, inverse_cdf, theta, df and the three log_likelihood_* verbs pass
    # straight through.
    return _COPULA_METHOD_ALIASES.get(method, method)


def _copula_sample_args(args: list, datasets: dict) -> list:
    """The three copula log-likelihood verbs take a paired SAMPLE, which the runner reads as
    one flat "all x then all y" args array. Spelling 200 numbers per assertion into the
    fixture would drown the file, so those assertions name their two datasets instead --
    args = ["<x dataset>", "<y dataset>"] -- and every runner splices the named arrays here.
    Documented under `bivariate_copula` in fixtures/README.md.
    """
    out: list = []
    for name in args:
        if name not in (datasets or {}):
            raise KeyError(f"copula log-likelihood args name an unknown dataset: {name}")
        out.extend(float(v) for v in datasets[name])
    return out


def _fixture_copula_pick(result: dict, method: str, args: list):
    """The runner returns a whole vector for the methods the fixture indexes into.

    `random` comes back as all the x draws followed by all the y draws, so a fixture
    (row, col) with args = [sample_size, seed, row, col] lands at col * sample_size + row.
    """
    values = result["values"]
    if method in ("lower_tail_dependence", "theta_minimum"):
        return values[0]
    if method in ("upper_tail_dependence", "theta_maximum"):
        return values[1]
    if method == "inverse_cdf":
        return values[int(args[2])]
    if method == "marginal_param":
        return values[int(args[1])]
    if method == "random_value":
        return values[int(args[3]) * int(args[0]) + int(args[2])]
    return values[0]


def _run_copula_case(target: str, construct: dict, assertions: list, datasets: dict):
    # Limitation 1: a "nan"/"inf" theta or df cannot be serialized into the grammar. Every
    # such case asserts parameters_valid alone.
    if _has_non_finite(construct):
        params = [_num(construct["theta"])]
        if "df" in construct:
            params.append(_num(construct["df"]))
        for a in assertions:
            if a["mode"] != "bool":
                raise KeyError(f"{target}/{a['method']}: a non-finite copula construct can "
                               "only assert parameters_valid")
            _check(_core.cop_val(target, params, "parameters_valid", [], "", [], "", []), a)
        return

    spec_json = _json(_copula_spec(target, construct, datasets))
    for a in assertions:
        method = a["method"]
        args = a.get("args", [])
        if a["mode"] == "bool":
            # The old dispatcher ignored the assertion's method in bool mode and read
            # parameters_valid(); keep that exactly.
            _check(_core.copula_run(spec_json, "parameters_valid", "[]")["values"][0], a)
            continue
        if method == "marginal_param":
            # args = ("x" | "y", index): the side picks the runner method, the index picks the
            # value out of that marginal's parameter vector.
            side = "marginal_x_parameters" if args[0] == "x" else "marginal_y_parameters"
            _check(_fixture_copula_pick(_core.copula_run(spec_json, side, "[]"), method, args), a)
            continue
        if method in _COPULA_LOG_LIKELIHOODS:
            args = _copula_sample_args(args, datasets)
        r = _core.copula_run(spec_json, _fixture_copula_method(method), _json(args))
        _check(_fixture_copula_pick(r, method, args), a)


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


# goodness_of_fit [function, args, observed_dataset, modeled_dataset]: routes through
# _core.toolbox_run("gof", ...) (numerics/support/toolbox_runner.hpp), so this fixture kind and a
# user's goodness_of_fit()/aic() call are the same code path. Mirrors dispatch_gof in
# core/tests/test_fixtures.cpp.
_GOF_FUNCTION_METHOD = {
    "MSE": "mse", "MAE": "mae",
    "NashSutcliffeEfficiency": "nse",
    "KlingGuptaEfficiency": "kge", "KlingGuptaEfficiencyMod": "kge_mod",
    "PBIAS": "pbias", "RSR": "rsr",
    "IndexOfAgreement": "d", "ModifiedIndexOfAgreement": "d_mod",
    "RefinedIndexOfAgreement": "d_ref", "VolumetricEfficiency": "ve",
}


def _dispatch_goodness_of_fit(fn, args, obs, mod):
    if fn == "AIC":
        options = json.dumps({"k": int(args[0]), "log_likelihood": args[1]})
        return _core.toolbox_run("gof", "aic", [], options)["values"][0]
    if fn in ("AICc", "BIC"):
        options = json.dumps({"n": int(args[0]), "k": int(args[1]), "log_likelihood": args[2]})
        method = "aicc" if fn == "AICc" else "bic"
        return _core.toolbox_run("gof", method, [], options)["values"][0]
    if fn not in _GOF_FUNCTION_METHOD:
        raise KeyError(f"unknown goodness_of_fit function: {fn}")
    return _core.toolbox_run("gof", _GOF_FUNCTION_METHOD[fn], [obs, mod], "{}")["values"][0]


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


# special_function/Correlation.* only: routes fixtures/special_functions/correlation.json's
# three Correlation targets through _core.toolbox_run (numerics/support/toolbox_runner.hpp)
# rather than adding bespoke glue, so the pinned pearson/spearman/kendalls_tau values become
# cross-language checks. Every other special_function target stays C++-only and unexercised
# here, exactly as before this kind gained any Python handling at all.
_CORRELATION_SPECIAL_FUNCTION_METHOD = {
    "Correlation.pearson": "pearson",
    "Correlation.spearman": "spearman",
    "Correlation.kendalls_tau": "kendall",
}


def _run_special_function_correlation_case(target, args_raw):
    args = [_num(v) for v in args_raw]
    mid = len(args) // 2
    x, y = args[:mid], args[mid:]
    method = _CORRELATION_SPECIAL_FUNCTION_METHOD[target]
    return _core.toolbox_run("correlation", method, [x, y], "{}")["values"][0]


# special_function/{RunningStatistics,RunningCovariance,Fourier,Statistics.percentile} (Task 3):
# the same "route the pinned value through _core.toolbox_run" pattern as Correlation above, one
# subset per family. Mirrors core/tests/test_fixtures.cpp's own routing exactly (see that file's
# running_covariance_toolbox()/running_covariance_element(), the RunningStatistics.* table
# entries, and the Fourier.{fft_at,real_fft_at,correlation_at} functions for the args
# conventions and the reasoning for which targets route and which stay C++-only:
# RunningStatistics's four population-normalized variants and its combined_*/clone_* cases have
# no toolbox_run("statistics", "summary", ...)-reachable equivalent, and Fourier.autocorrelation_at
# deliberately stays off this list because the toolbox "spectra.autocorrelation" method wraps the
# newer Autocorrelation class, not Fourier::autocorrelation itself).
_RUNNING_STATISTICS_SPECIAL_FUNCTION_INDEX = {
    "RunningStatistics.count": 0, "RunningStatistics.minimum": 1,
    "RunningStatistics.maximum": 2, "RunningStatistics.mean": 3,
    "RunningStatistics.variance": 4, "RunningStatistics.standard_deviation": 5,
    "RunningStatistics.coefficient_of_variation": 6, "RunningStatistics.skewness": 7,
    "RunningStatistics.kurtosis": 8,
}

_RUNNING_COVARIANCE_SPECIAL_FUNCTION_BLOCK = {
    "RunningCovariance.mean_element": 0,
    "RunningCovariance.covariance_element": 1,
    "RunningCovariance.sample_covariance_element": 2,
    "RunningCovariance.sample_correlation_element": 3,
    "RunningCovariance.population_covariance_element": 4,
    "RunningCovariance.population_correlation_element": 5,
}

_FOURIER_TOOLBOX_METHOD = {
    "Fourier.fft_at": "dft",
    "Fourier.real_fft_at": "dft_real",
    "Fourier.correlation_at": "cross_correlation",
}

# special_function/{Histogram,Bilinear} (Task 4): the same routing pattern, one subset per
# family. Mirrors core/tests/test_fixtures.cpp's own routing exactly (see that file's Histogram.*
# table entries and bilinear_log_floor_value() for the args conventions). Histogram.data_count/
# get_bin_index_of have no toolbox-method equivalent (neither "statistics" nor "bins" exposes
# them) and Histogram.adapt_* needs AddData(), which the toolbox arm's stateless construction
# never calls, so those four stay unrouted -- the same "no toolbox_run-reachable equivalent"
# reasoning as RunningStatistics's population_* variants. Search.* also stays unrouted: neither
# "linear" nor "bilinear" returns a search index, only an interpolated y.
_HISTOGRAM_STATISTICS_INDEX = {
    "Histogram.mean": 0, "Histogram.median": 1, "Histogram.mode": 2,
    "Histogram.standard_deviation": 3, "Histogram.lower_bound": 4, "Histogram.upper_bound": 5,
    "Histogram.bin_width": 6, "Histogram.number_of_bins": 7,
}

_HISTOGRAM_BINS_COLUMN = {
    "Histogram.bin_lower_bound_at": 0,
    "Histogram.bin_upper_bound_at": 1,
    "Histogram.bin_frequency_at": 3,
}


def _histogram_toolbox_options(explicit_bins):
    return json.dumps({"bins": explicit_bins}) if explicit_bins > 0 else "{}"


_ROUTED_SPECIAL_FUNCTION_TARGETS = (
    set(_CORRELATION_SPECIAL_FUNCTION_METHOD)
    | set(_RUNNING_STATISTICS_SPECIAL_FUNCTION_INDEX)
    | set(_RUNNING_COVARIANCE_SPECIAL_FUNCTION_BLOCK)
    | set(_FOURIER_TOOLBOX_METHOD)
    | {"Statistics.percentile"}
    | set(_HISTOGRAM_STATISTICS_INDEX)
    | set(_HISTOGRAM_BINS_COLUMN)
    | {"Bilinear.log_floor_value"}
    | {"Probability.hpcm_joint"}
    | {"DifferentialEvolution.best_value"}
)


# special_function/DifferentialEvolution.best_value (Task 8): reuses
# fixtures/special_functions/differential_evolution.json (already pinned C++-only) rather than
# duplicating it -- routed through _core.optim_run so the callback path itself is exercised. args
# convention: [fn_id, direction, D, lower(D), upper(D), index] -- see
# core/tests/test_fixtures.cpp's differential_evolution_best_value() for the authoritative
# description. fn_id 0 = "quadratic" (sum_i (x_i - i)^2, 0-based i), 1 = "normal_loglik" (Normal
# log-likelihood of {9,10,11,12,13} at mean=p[0], sd=p[1]) -- NATIVE Python closures reproducing
# the same two P3.3 numerical_derivative fixture functions, so this case exercises the real Python
# callback path. The log-density is written out explicitly (not scipy.stats.norm.logpdf) so a
# chaotic DE search sees the SAME density implementation in both languages -- corehydror's
# test-fixtures.R mirrors this exact formula rather than calling dnorm(log = TRUE). `value`
# un-applies OptimResult's raw-sign convention back to the C#
# BestParameterSet.Fitness this fixture's literals were curated against (see
# differential_evolution_best_value()'s own comment for why).
def _run_special_function_differential_evolution_case(args_raw):
    a = [_num(v) for v in args_raw]
    fn_id, direction, D = int(a[0]), int(a[1]), int(a[2])
    lower = a[3 : 3 + D]
    upper = a[3 + D : 3 + 2 * D]
    index = int(a[3 + 2 * D])
    sample = [9.0, 10.0, 11.0, 12.0, 13.0]

    if fn_id == 0:
        def objective(p):
            return sum((v - i) ** 2 for i, v in enumerate(p))
    else:
        def objective(p):
            mu, sigma = p[0], p[1]
            return sum(
                -0.5 * ((x - mu) / sigma) ** 2 - math.log(math.sqrt(2 * math.pi) * sigma)
                for x in sample
            )

    spec = json.dumps({"method": "de", "lower": lower, "upper": upper, "maximize": direction == 1})
    r = _core.optim_run(spec, objective)
    if index == D:
        return -r["value"] if direction == 1 else r["value"]
    return r["parameters"][index]


# special_function/Probability.hpcm_joint (Task 6): the same routing pattern, through the new
# "probability" toolbox group's "joint" method (dependency = "correlation"). Mirrors
# core/tests/test_fixtures.cpp's special_function_table() entry for "Probability.hpcm_joint" for
# the args convention (args = [p_0..p_(n-1), ind_0..ind_(n-1), corr(n*n flattened row-major)], n
# inferred from the argument count). Probability.hpcm_conditional_at stays unrouted: it needs the
# conditionalProbabilities out-value, which the "probability" toolbox group's "joint" method does
# not expose.
def _run_special_function_probability_hpcm_joint_case(args_raw):
    args = [_num(v) for v in args_raw]
    n = None
    for candidate in range(1, 21):
        if 2 * candidate + candidate * candidate == len(args):
            n = candidate
            break
    if n is None:
        raise ValueError("cannot infer n for Probability.hpcm args")
    p = args[:n]
    ind = args[n : 2 * n]
    corr = args[2 * n :]
    return _core.toolbox_run(
        "probability", "joint", [p, ind, corr], json.dumps({"dependency": "correlation"})
    )["values"][0]


def _run_special_function_case(target, args_raw):
    if target in _CORRELATION_SPECIAL_FUNCTION_METHOD:
        return _run_special_function_correlation_case(target, args_raw)

    args = [_num(v) for v in args_raw]

    if target == "Statistics.percentile":
        n = len(args) - 2
        data, k, sorted_ = args[:n], args[n], args[n + 1] != 0.0
        options = json.dumps({"sorted": True}) if sorted_ else "{}"
        return _core.toolbox_run("statistics", "percentile", [data, [k]], options)["values"][0]

    if target in _RUNNING_STATISTICS_SPECIAL_FUNCTION_INDEX:
        r = _core.toolbox_run("statistics", "summary", [args], "{}")
        return r["values"][_RUNNING_STATISTICS_SPECIAL_FUNCTION_INDEX[target]]

    if target in _RUNNING_COVARIANCE_SPECIAL_FUNCTION_BLOCK:
        size, num_pushes = int(args[0]), int(args[1])
        columns = [[args[2 + p * size + j] for p in range(num_pushes)] for j in range(size)]
        r = _core.toolbox_run("statistics", "running_covariance", columns, "{}")
        base = 2 + num_pushes * size
        i = int(args[base])
        block = _RUNNING_COVARIANCE_SPECIAL_FUNCTION_BLOCK[target]
        if block == 0:
            return r["values"][1 + i]
        j = int(args[base + 1])
        offset = 1 + size + (block - 1) * (size * size)
        return r["values"][offset + i * size + j]

    if target in _FOURIER_TOOLBOX_METHOD:
        if target == "Fourier.correlation_at":
            n = (len(args) - 1) // 2
            data1, data2 = args[:n], args[n : 2 * n]
            index = int(args[2 * n])
            r = _core.toolbox_run("spectra", "cross_correlation", [data1, data2], "{}")
            return r["values"][index]
        n = len(args) - 2
        data, inverse, index = args[:n], args[n] != 0.0, int(args[n + 1])
        options = json.dumps({"inverse": True}) if inverse else "{}"
        r = _core.toolbox_run("spectra", _FOURIER_TOOLBOX_METHOD[target], [data], options)
        return r["values"][index]

    if target in _HISTOGRAM_STATISTICS_INDEX:
        explicit_bins = int(args[0])
        data = args[1:]
        options = _histogram_toolbox_options(explicit_bins)
        r = _core.toolbox_run("histogram", "statistics", [data], options)
        return r["values"][_HISTOGRAM_STATISTICS_INDEX[target]]

    if target in _HISTOGRAM_BINS_COLUMN:
        explicit_bins = int(args[0])
        probe = int(args[-1])
        data = args[1:-1]
        options = _histogram_toolbox_options(explicit_bins)
        r = _core.toolbox_run("histogram", "bins", [data], options)
        return r["values"][probe * 4 + _HISTOGRAM_BINS_COLUMN[target]]

    if target == "Bilinear.log_floor_value":
        coords = [0.0, 1e-15, 1.0]
        flat = [0.0, 0.0, 0.0, 1e-15, 1e-15, 1e-15, 1.0, 1.0, 1.0]
        options = json.dumps({"x1_transform": "log", "x2_transform": "log", "y_transform": "log"})
        r = _core.toolbox_run(
            "interpolation", "bilinear", [coords, coords, flat, [args[0]], [args[1]]], options
        )
        return r["values"][0]

    if target == "Probability.hpcm_joint":
        return _run_special_function_probability_hpcm_joint_case(args_raw)

    if target == "DifferentialEvolution.best_value":
        return _run_special_function_differential_evolution_case(args_raw)

    raise KeyError(f"unrouted special_function target: {target}")


# toolbox [group, data, options; assertions carry method/index/label/select]: every Numerics
# utility group runs through _core.toolbox_run. Mirrors run_toolbox_kind in
# core/tests/test_fixtures.cpp.
def _toolbox_case_data(case, datasets):
    out = []
    for d in case.get("data", []):
        src = datasets[d] if isinstance(d, str) else d
        out.append([_num(v) for v in src])
    return out


def _toolbox_select(r, a, group):
    select = a.get("select", "value")
    if select == "length":
        return float(len(r["values"]))
    if select == "rows":
        if "dims" not in r or len(r["dims"]) < 1:
            raise ValueError(f"toolbox select 'rows' has no dims (group '{group}')")
        return float(r["dims"][0])
    if select == "columns":
        if "dims" not in r or len(r["dims"]) < 2:
            raise ValueError(f"toolbox select 'columns' has no dims (group '{group}')")
        return float(r["dims"][1])
    if "label" in a:
        i = list(r["names"]).index(a["label"])
    else:
        i = int(a.get("index", 0))
    return float(r["values"][i])


def _run_toolbox_case(group, case, datasets):
    data = _toolbox_case_data(case, datasets)
    options_dict = dict(case.get("options", {}))
    # Only the "sobol" method reads a path (SobolSequence's constructor only touches it when
    # dimension > 1); "stratify" never does, so this stays scoped to that one method rather than
    # the whole "sampling" group.
    if group == "sampling" and case["assertions"][0]["method"] == "sobol":
        # The Sobol direction-numbers file ships inside the package; path resolution is a
        # wrapper concern (see numerics/support/toolbox/sampling.hpp's file header), so the
        # fixture itself never carries a "path" key -- this harness injects its own resolved
        # path, mirroring what sobol_sequence() does for a real caller.
        options_dict["path"] = str(files("corehydropy") / "data" / "new-joe-kuo-6.21201")
    options = json.dumps(options_dict)
    for a in case["assertions"]:
        r = _core.toolbox_run(group, a["method"], data, options)
        _check(_toolbox_select(r, a, group), a)


# optimizer [construct carries method/lower/upper/initial/maximize/seed/control; assertions carry
# value/parameter/status]: the six ported optimizers (Task 8), run through _core.optim_run against
# a NATIVE Python closure -- not _core.toolbox_run -- because an optimizer's input is a live
# function, not serializable data. Mirrors run_optimizer_kind in core/tests/test_fixtures.cpp.
def _optimizer_spec_json(construct):
    spec = {"method": construct["method"]}
    for key in ("lower", "upper", "initial"):
        if key in construct:
            spec[key] = construct[key]
    if "maximize" in construct:
        spec["maximize"] = bool(construct["maximize"])
    if "seed" in construct:
        spec["seed"] = int(construct["seed"])
    if construct.get("control"):
        spec["control"] = construct["control"]
    return json.dumps(spec)


# The handful of TestFunctions.cs objectives fixtures/toolbox/optimizers.json names by string --
# NATIVE Python closures reproducing the same formulas as
# core/tests/optimization_test_functions.hpp, so every optimizer fixture case exercises the real
# Python callback path (corehydro::numerics::support::GuardedObjective exists to protect exactly
# this call).
def _optimizer_fixture_objective(name):
    if name == "FXYZ":
        return lambda p: (4 * p[0] - 0.5) ** 2 + (3 * p[1] - 0.6) ** 2 + (2 * p[2] - 0.7) ** 2
    if name == "DeJong":
        return lambda p: sum(v ** 2 for v in p)
    if name == "Booth":
        return lambda p: (p[0] + 2 * p[1] - 7) ** 2 + (2 * p[0] + p[1] - 5) ** 2
    if name == "McCormick":
        return lambda p: (
            math.sin(p[0] + p[1]) + (p[0] - p[1]) ** 2 - 1.5 * p[0] + 2.5 * p[1] + 1.0
        )
    if name == "FX":
        return lambda p: (p[0] + 3.0) * (p[0] - 1.0) ** 2
    raise KeyError(f"unknown optimizer fixture objective: {name}")


def _run_optimizer_case(case):
    construct = dict(case["construct"])
    objective_name = construct.pop("objective", "DeJong")
    r = _core.optim_run(_optimizer_spec_json(construct), _optimizer_fixture_objective(objective_name))
    for a in case["assertions"]:
        if a["method"] == "value":
            _check(r["value"], a)
        elif a["method"] == "parameter":
            _check(r["parameters"][a["args"][0]], a)
        elif a["method"] == "status":
            assert r["status"] == a["expected"]
        else:
            raise KeyError(f"unknown optimizer fixture assertion method: {a['method']}")


# callback [construct carries group/method/callback/options; assertions carry value/dim/status]:
# the ported routines whose input is a live function, run through _core.callback_math against a
# NATIVE Python closure. Mirrors run_callback_kind in core/tests/test_fixtures.cpp. NOTE these
# catalog names are NOT the optimizer catalog's above: `Diff_FXYZ` is Test_Differentiation.FXYZ
# (x^3 + y^4 + z^5), unrelated to the optimizer catalog's `FXYZ`. `Diff_FX` and `Quad_FX3` are both
# x^3, from two different upstream test files -- hence the prefixes.
def _callback_fixture_function(name):
    if name == "Root_Quadratic":
        return lambda x: x ** 2 - 2.0
    # P2 "math extras": TestFunctions.Quadratic_Deriv, the newton catalog's counterpart of
    # Root_Quadratic.
    if name == "RootD_Quadratic":
        return lambda x: 2.0 * x
    if name == "Root_Cubic":
        return lambda x: x ** 3 - x - 1.0
    # TestFunctions.Trigonometric, root ~1.12191713 on [0, pi].
    if name == "Root_Trigonometric":
        return lambda x: 2.0 * math.sin(x) - 3.0 * math.cos(x) - 0.5
    # TestFunctions.Trigonometric_Deriv.
    if name == "RootD_Trigonometric":
        return lambda x: 2.0 * math.cos(x) + 3.0 * math.sin(x)
    # Test_NewtonRaphson.Test_Multi_LinearSystem's system, F([x;y]) = [3x + y - 9, x + 2y - 8],
    # whose unique root is [2, 3].
    if name == "Sys_Linear_F":
        return lambda v: [3.0 * v[0] + v[1] - 9.0, v[0] + 2.0 * v[1] - 8.0]
    # The (constant) Jacobian of Sys_Linear_F, a sequence of rows (already row-major): [[3, 1], [1, 2]].
    if name == "Sys_Linear_J":
        return lambda v: [[3.0, 1.0], [1.0, 2.0]]
    if name == "Diff_FX":
        return lambda x: x ** 3
    if name == "Diff_FXY":
        return lambda p: p[0] ** 2 * p[1] ** 3
    if name == "Diff_FXYZ":
        return lambda p: p[0] ** 3 + p[1] ** 4 + p[2] ** 5
    if name == "Diff_FH":
        return lambda p: p[0] ** 3 - 2.0 * p[0] * p[1] - p[1] ** 6
    if name == "Quad_FX3":
        return lambda x: x ** 3
    if name == "Quad_Cosine":
        return math.cos
    if name == "Quad_Sine":
        return math.sin
    if name == "Quad_FXX":
        return lambda x: 0.5 + 24.0 * x + 3.0 * x ** 2
    if name == "Quad_FXXX":
        return lambda x: 0.5 + 24.0 * x + 3.0 * x ** 2 + 8.0 * x ** 3
    if name == "Quad_Peak":
        # corehydro addition, no upstream integrand -- the one callback that reaches the
        # subdividing branch of the recursion. Arithmetic only, so all four runners agree bit for
        # bit and the evaluation count is a real oracle.
        return lambda x: 1.0 / (1.0 + 1.0e4 * x * x)
    # P2 "math extras", the math/quadrature_2d catalog: Test_AdaptiveSimpsonsRule2D.Test_XPlusY.
    if name == "Quad2D_XPlusY":
        return lambda x, y: x + y
    # Test_AdaptiveSimpsonsRule2D.Test_PI, upstream's Integrands.PI2D: the indicator of the unit
    # disc, whose integral over [-1, 1] x [-1, 1] approximates pi.
    if name == "Quad2D_PI2D":
        return lambda x, y: 1.0 if x * x + y * y < 1.0 else 0.0
    # The mcmc catalog (fixtures/callback/mcmc.json). Both log-densities are arithmetic only and
    # sum in an explicit loop rather than through sum(): a Markov chain turns one differing bit
    # into a different chain outright, so the four runners have to agree to the last bit for these
    # oracles to mean anything. See the fixture's own note.
    if name == "Mcmc_GaussianKernel":
        return _mcmc_gaussian_kernel
    if name == "Mcmc_LinearKernel":
        return _mcmc_linear_kernel
    # The Gibbs case's model, whose full conditional really IS uniform: with
    # x_i ~ Uniform(mu - 1, mu + 1) and a flat prior, mu given the data is
    # Uniform(max(x) - 1, min(x) + 1), so Prop_UniformConditional is an exact Gibbs step rather
    # than a random walk wearing Gibbs's name. Comparisons and arithmetic only.
    if name == "Mcmc_UniformWidthKernel":
        return _mcmc_uniform_width_kernel
    # (parameters, rng) -> parameters, upstream's Gibbs.Proposal shape, drawing through the HANDLE.
    if name == "Prop_UniformConditional":
        return _mcmc_uniform_conditional
    # The TWO-parameter member of the proposal catalog, written for the second case of
    # fixtures/callback/callback_cross_language.json. An INDEPENDENCE proposal: it ignores the
    # state it is handed and draws each parameter from a fixed interval, exactly as
    # Prop_UniformConditional does for one. Gibbs accepts every proposal, so the only mark the
    # log-density leaves on the run is the fitness it reports.
    if name == "Prop_UniformBox":
        return _mcmc_uniform_box
    # (parameters) -> vector, upstream's HMC.Gradient shape: d/dmu of Mcmc_GaussianKernel.
    if name == "Grad_GaussianKernel":
        return _mcmc_gaussian_gradient
    # Task-5 review fix, coverage finding: unlike Mcmc_GaussianKernel, whose derivative
    # sum(x - mu) is LINEAR in mu (so its third derivative is zero and the ported
    # central-difference default agrees with the analytic gradient to rounding, ~4e-16), this
    # kernel's derivative is CUBIC in mu, so the central-difference truncation error is real rather
    # than rounding -- the analytic and default gradients genuinely disagree, which is what a
    # supplied-vs-ignored gradient regression needs to be caught by the oracle gate.
    if name == "Mcmc_QuarticKernel":
        return _mcmc_quartic_kernel
    if name == "Grad_QuarticKernel":
        return _mcmc_quartic_gradient
    # The bootstrap catalog (fixtures/callback/bootstrap.json), upstream's four Bootstrap<TData>
    # delegate shapes. Every one is arithmetic and comparisons only, and the mean is summed in an
    # explicit loop rather than through statistics.mean(): R's own sum()/mean() accumulate in
    # extended precision, and one differing bit in a fitted mean moves a percentile. The resample
    # draws every index through the HANDLE, exactly as a user's own resample function does.
    if name == "Resample_Iid":
        return _boot_resample_iid
    if name == "Fit_Mean":
        return _boot_fit_mean
    # The CONTRACTION-BEARING member of the same catalog, written for the second case of
    # fixtures/callback/callback_cross_language.json: the least-squares line of the sample against
    # its position, whose every accumulation is `acc + a * b` -- the shape clang and gcc fuse into
    # a multiply-add by default and Python never does. See the C++ catalog's own note.
    if name == "Fit_LinearTrend":
        return _boot_fit_linear_trend
    # The PIVOTAL member of the same catalog: upstream's Func<TData, BootstrapFit>
    # FitWithCovarianceFunction, the delegate that run type fits through. The model is the
    # two-parameter Normal location-scale MLE, whose covariance is analytic -- diag(s2 / n,
    # s2 / (2n)) -- so the whole callback is arithmetic plus one sqrt, and sqrt is the one libm
    # function IEEE 754 requires to be correctly rounded. `ss += (x - mu) * (x - mu)` is itself a
    # contraction-bearing shape, so this zero-tolerance guarantee also depends on the C++ catalog's
    # own -ffp-contract=off scoping in core/CMakeLists.txt.
    if name == "FitCov_NormalMLE":
        return _boot_fit_with_covariance
    if name == "Stat_Identity":
        return _boot_stat_identity
    if name == "Stat_MeanAndSquare":
        return _boot_stat_mean_and_square
    if name == "Jack_LeaveOneOut":
        return _boot_jack_leave_one_out
    # The gmm catalog (fixtures/callback/gmm.json), upstream's three delegate shapes from
    # GeneralizedMethodOfMoments's delegate constructor. The model is the just-identified
    # two-parameter method-of-moments fit of a Normal: theta = (mu, sigma2), whose unique root --
    # and so the GMM optimum, since q = p makes g = 0 attainable -- is the sample mean and the
    # population variance. Arithmetic and an explicit loop only, never sum()/mean(): R accumulates
    # both in extended precision where the other three languages accumulate in double, and one
    # differing bit moves a fitted parameter.
    if name == "Mom_NormalMeanVariance":
        return _gmm_moment_conditions
    # The OVER-IDENTIFIED member of the same catalog: the identical Normal model and the identical
    # eight observations, with a third moment condition added -- mean((x - mu)^3), zero for a
    # Normal -- so q = 3 > p = 2 and the degrees of freedom become 1. The only case in the file
    # that reaches the chi-squared p-value branch of the J-statistic.
    if name == "Mom_NormalThreeMoments":
        return _gmm_moment_conditions_three
    # The CUBIC-JACOBIAN member of the same catalog, and the one case in the file whose analytic
    # Jacobian is distinguishable from the ported numerical one.
    if name == "Mom_NormalFourthMoment":
        return _gmm_moment_conditions_fourth
    if name == "Jac_NormalFourthMoment":
        return _gmm_jacobian_fourth
    if name == "Jac_NormalMeanVariance":
        return _gmm_jacobian
    if name == "Pen_SigmaTowardsOne":
        return _gmm_penalty
    # The rng catalog (fixtures/callback/rng_handle.json): two arguments, (parameters, rng), the
    # Gibbs proposal's own signature. Each draws through the HANDLE it is given -- exactly what a
    # user's proposal function would do -- rather than reaching for a generator of its own, which
    # is the property the fixture exists to pin.
    if name == "Rng_Uniform":
        return lambda parameters, rng: rng.uniform(int(parameters[0]))
    if name == "Rng_Integers":
        return lambda parameters, rng: [
            float(v) for v in rng.integers(int(parameters[0]), int(parameters[1]), int(parameters[2]))
        ]
    if name == "Rng_Interleaved":
        return lambda parameters, rng: (
            list(rng.uniform(2))
            + [float(v) for v in rng.integers(2, 0, 100)]
            + list(rng.uniform(1))
        )
    if name == "Rng_Warmup1000":
        return _rng_warmup_1000
    raise KeyError(f"unknown callback fixture callback: {name}")


def _mcmc_gaussian_kernel(p):
    data = (4.9, 5.1, 5.0, 5.2, 4.8)
    acc = 0.0
    for x in data:
        acc += (x - p[0]) * (x - p[0])
    return -0.5 * acc


def _mcmc_linear_kernel(p):
    t = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    y = (2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1)
    acc = 0.0
    for i in range(len(t)):
        residual = y[i] - p[0] - p[1] * t[i]
        acc += residual * residual
    return -0.5 * acc


def _mcmc_uniform_width_kernel(p):
    data = (4.9, 5.1, 5.0, 5.2, 4.8)
    for x in data:
        if x - p[0] > 1.0 or p[0] - x > 1.0:
            return float("-inf")
    return 0.0


def _mcmc_uniform_conditional(parameters, rng):
    data = (4.9, 5.1, 5.0, 5.2, 4.8)
    lo = max(data) - 1.0
    hi = min(data) + 1.0
    return [lo + rng.uniform(1)[0] * (hi - lo)]


def _mcmc_uniform_box(parameters, rng):
    # One uniform(2) call, not two of length one: a single call cannot split the stream.
    lo = (-1.0, 1.5)
    hi = (1.0, 2.5)
    u = rng.uniform(2)
    return [lo[j] + u[j] * (hi[j] - lo[j]) for j in range(2)]


def _mcmc_gaussian_gradient(p):
    data = (4.9, 5.1, 5.0, 5.2, 4.8)
    acc = 0.0
    for x in data:
        acc += x - p[0]
    return [acc]


def _mcmc_quartic_kernel(p):
    # Coefficient 0.05 is load-bearing, not decorative: see the note in the R/C++ twins -- an
    # unscaled quartic makes HMC's leapfrog trajectory genuinely chaotic over 200 iterations, and
    # 0.05 keeps the analytic-vs-default gradient divergence small and smooth instead.
    data = (4.9, 5.1, 5.0, 5.2, 4.8)
    acc = 0.0
    for x in data:
        d = x - p[0]
        acc += d * d * d * d
    return -0.05 * acc


def _mcmc_quartic_gradient(p):
    data = (4.9, 5.1, 5.0, 5.2, 4.8)
    acc = 0.0
    for x in data:
        d = x - p[0]
        acc += d * d * d
    return [0.2 * acc]


def _boot_resample_iid(data, parameters, rng):
    # rng.integers draws on [0, n), counting from 0 exactly as the ported delegate does.
    return [data[k] for k in rng.integers(len(data), 0, len(data))]


def _boot_fit_mean(data):
    acc = 0.0
    for x in data:
        acc += x
    return [acc / len(data)]


def _boot_fit_linear_trend(data):
    # Centered ordinary least squares of the sample against its position t = 1..n:
    #   slope = sum(dt * dy) / sum(dt * dt),  intercept = ybar - slope * tbar.
    # Explicit loops rather than sum()/mean(), for the same reason _boot_fit_mean uses one.
    n = float(len(data))
    st = 0.0
    sy = 0.0
    for i in range(len(data)):
        st += float(i + 1)
        sy += data[i]
    tbar = st / n
    ybar = sy / n
    num = 0.0
    den = 0.0
    for i in range(len(data)):
        dt = float(i + 1) - tbar
        dy = data[i] - ybar
        num += dt * dy
        den += dt * dt
    slope = num / den
    return [ybar - slope * tbar, slope]


def _boot_fit_with_covariance(data):
    # The two-parameter Normal location-scale MLE -- theta = (mu, sigma) with sigma the POPULATION
    # standard deviation -- and its analytic covariance, diag(s2 / n, s2 / (2n)). Explicit loops
    # rather than sum()/mean(), for the same reason _boot_fit_mean uses one.
    n = float(len(data))
    acc = 0.0
    for x in data:
        acc += x
    mu = acc / n
    ss = 0.0
    for x in data:
        ss += (x - mu) * (x - mu)
    s2 = ss / n
    return {
        "parameters": [mu, math.sqrt(s2)],
        "covariance": [[s2 / n, 0.0], [0.0, s2 / (2.0 * n)]],
    }


def _boot_stat_identity(parameters):
    return parameters


def _boot_stat_mean_and_square(parameters):
    return [parameters[0], parameters[0] * parameters[0]]


def _boot_jack_leave_one_out(data, index):
    # `index` counts from 0, as the ported delegate does.
    return list(data[:index]) + list(data[index + 1:])


def _gmm_moment_conditions(p):
    data = (4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7)
    n = 8.0
    g0 = g1 = s00 = s01 = s11 = 0.0
    for x in data:
        a = x - p[0]
        b = a * a - p[1]
        g0 += a
        g1 += b
        s00 += a * a
        s01 += a * b
        s11 += b * b
    return ([g0 / n, g1 / n], [[s00 / n, s01 / n], [s01 / n, s11 / n]])


def _gmm_moment_conditions_three(p):
    data = (4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7)
    n = 8.0
    g0 = g1 = g2 = 0.0
    s00 = s01 = s02 = s11 = s12 = s22 = 0.0
    for x in data:
        a = x - p[0]
        b = a * a - p[1]
        c = a * a * a
        g0 += a
        g1 += b
        g2 += c
        s00 += a * a
        s01 += a * b
        s02 += a * c
        s11 += b * b
        s12 += b * c
        s22 += c * c
    return (
        [g0 / n, g1 / n, g2 / n],
        [[s00 / n, s01 / n, s02 / n], [s01 / n, s11 / n, s12 / n], [s02 / n, s12 / n, s22 / n]],
    )


def _gmm_moment_conditions_fourth(p):
    # theta = (mu, sigma), matched on the FIRST and FOURTH central moments of a Normal:
    #   g = [mean(x - mu), mean(u^4) - 3 t^4],  u = 100 (x - mu),  t = 100 sigma
    # so dg2/dsigma = -1200 t^3 is CUBIC in the parameter. The eight observations are the ones the
    # rest of the catalog uses with the decimal point moved two places, which is what makes the
    # fitted sigma (0.00404) small next to the ported numerical Jacobian's step h = 1e-4
    # (|theta| + 1).
    data = (0.041, 0.052, 0.048, 0.055, 0.049, 0.051, 0.053, 0.047)
    n = 8.0
    t = p[1] * 100.0
    t4 = 3.0 * t * t * t * t
    g0 = g1 = s00 = s01 = s11 = 0.0
    for x in data:
        a = x - p[0]
        u = a * 100.0
        b = u * u * u * u - t4
        g0 += a
        g1 += b
        s00 += a * a
        s01 += a * b
        s11 += b * b
    return ([g0 / n, g1 / n], [[s00 / n, s01 / n], [s01 / n, s11 / n]])


def _gmm_jacobian_fourth(p):
    # One ROW per moment condition: dg1/dmu = -1, dg1/dsigma = 0, dg2/dmu = -400 mean(u^3),
    # dg2/dsigma = -1200 t^3.
    data = (0.041, 0.052, 0.048, 0.055, 0.049, 0.051, 0.053, 0.047)
    acc = 0.0
    for x in data:
        u = (x - p[0]) * 100.0
        acc += u * u * u
    t = p[1] * 100.0
    return [[-1.0, 0.0], [-400.0 * acc / 8.0, -1200.0 * t * t * t]]


def _gmm_jacobian(p):
    # One ROW per moment condition: dg1/dmu = -1, dg1/dsigma2 = 0, dg2/dmu = -2 mean(x - mu),
    # dg2/dsigma2 = -1.
    data = (4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7)
    acc = 0.0
    for x in data:
        acc += x - p[0]
    return [[-1.0, 0.0], [-2.0 * acc / 8.0, -1.0]]


def _gmm_penalty(p):
    # A ridge penalty pulling sigma2 towards 1, carrying its own 1/2 as the ported half-quadratic
    # convention expects.
    return 0.5 * (p[1] - 1.0) * (p[1] - 1.0)


def _rng_warmup_1000(parameters, rng):
    rng.uniform(1000)  # discarded, as upstream's own test discards 1000 GenRandInt32
    return rng.uniform(10)


def _run_callback_case(case):
    construct = case["construct"]
    # Every group has its own Python entry point: callback_math, rng_probe, callback_mcmc,
    # callback_gmm and callback_bootstrap.
    options_json = json.dumps(construct.get("options", {}))
    fn = _callback_fixture_function(construct["callback"])
    if construct["group"] == "math" and construct["method"] in (
        "root_find_newton",
        "root_find_system",
    ):
        # P2 "math extras": these two math methods need a SECOND callback (the analytic
        # derivative `df`, or the Jacobian reusing gmm's `jacobian` key), which callback_math has
        # no argument for -- callback_math2 is the two-callback entry point, mirroring
        # callback_gmm's own optional second/third delegates below.
        second_key = "df" if construct["method"] == "root_find_newton" else "jacobian"
        g = _callback_fixture_function(construct[second_key])
        r = _core.callback_math2(construct["method"], options_json, fn, g)
    elif construct["group"] == "math" and construct["method"] == "quadrature_2d":
        # P2 "math extras": the (x, y) half of the math group, callback_math_xy -- see
        # callback_math2 above for why a differing arity gets its own Python entry point.
        r = _core.callback_math_xy(construct["method"], options_json, fn)
    elif construct["group"] == "math":
        r = _core.callback_math(construct["method"], options_json, fn)
    elif construct["group"] == "rng":
        r = _core.rng_probe(options_json, fn)
    elif construct["group"] == "mcmc":
        # The two other delegates the mcmc group's samplers take, each resolved out of the same
        # catalog: a Gibbs proposal and an HMC/NUTS gradient. Absent keys stay None, which is what
        # "no proposal" and "the ported default gradient" mean.
        proposal = (
            _callback_fixture_function(construct["proposal"]) if "proposal" in construct else None
        )
        gradient = (
            _callback_fixture_function(construct["gradient"]) if "gradient" in construct else None
        )
        r = _core.callback_mcmc(options_json, fn, proposal, gradient)
    elif construct["group"] == "gmm":
        # `callback` names the MOMENT CONDITION function -- this group's required delegate, its
        # counterpart of the mcmc group's log-likelihood -- and the two optional ones have keys of
        # their own. An absent key stays None, which is what "the ported numerical Jacobian" and
        # "no penalty" mean.
        jacobian = (
            _callback_fixture_function(construct["jacobian"]) if "jacobian" in construct else None
        )
        penalty = (
            _callback_fixture_function(construct["penalty"]) if "penalty" in construct else None
        )
        r = _core.callback_gmm(options_json, fn, jacobian, penalty)
    elif construct["group"] == "bootstrap":
        # `callback` names the RESAMPLE delegate -- the one handed the generator, this group's
        # counterpart of the mcmc group's log-likelihood -- and the other four have keys of their
        # own. An absent `jackknife` stays None, which is what every method but BCa means; `fit`
        # and `fit_with_covariance` are the two fitting delegates the run types take, and a case
        # supplies exactly the one its own `run_type` needs.
        def optional(key):
            return _callback_fixture_function(construct[key]) if key in construct else None

        r = _core.callback_bootstrap(
            options_json,
            fn,
            optional("fit"),
            _callback_fixture_function(construct["statistic"]),
            optional("jackknife"),
            optional("fit_with_covariance"),
        )
    else:
        raise KeyError(f"unknown callback fixture group: {construct['group']}")
    for a in case["assertions"]:
        index = a["args"][0] if "args" in a else 0
        if a["method"] == "value":
            _check(r["values"][index], a)
        elif a["method"] == "named":
            # By label, not position: the mcmc group's summary block is long and its indices
            # shift with the chain and parameter counts.
            names = list(r["names"])
            if a["name"] not in names:
                raise KeyError(f"callback: no result named '{a['name']}'")
            _check(r["values"][names.index(a["name"])], a)
        elif a["method"] == "dim":
            _check(float(r["dims"][index]), a)
        elif a["method"] == "status":
            assert r["status"] == a["expected"]
        else:
            raise KeyError(f"unknown callback fixture assertion method: {a['method']}")


# callback_cross_language [each case nests one sub-block per key OTHER than "name" -- "mcmc",
# "bootstrap", "pivotal" -- each shaped exactly like a "callback"-kind case's
# construct/assertions]: a case of fixtures/callback/callback_cross_language.json asserts a seeded
# posterior and a seeded bootstrap interval TOGETHER, because the file's job is proving they
# reproduce identically across languages in one guarantee rather than in a file each. The labels
# are read off the case rather than listed here, so a case may nest one block or five without a
# runner change.
# Reuses _run_callback_case verbatim; no new evaluation logic, just the nesting. Its assertions are
# spelled mode "abs" with tol 0, i.e. bit equality with the C++, R and C# runners rather than a
# tolerance.
def _run_callback_cross_language_case(case):
    for sub, block in case.items():
        if sub == "name":
            continue
        _run_callback_case(block)


# toolbox_cross_language [one case nests "optimizer" (shaped like an "optimizer"-kind
# construct/assertions), "sobol" and "stratify" (each shaped like a "toolbox"-kind
# group-"sampling" case's options/assertions)]: fixtures/toolbox/toolbox_cross_language.json's
# single fixture proves all three reproduce identically across R/Python/C++/C# in one
# guarantee. Reuses _optimizer_spec_json/_optimizer_fixture_objective and
# _core.toolbox_run/_toolbox_select verbatim -- see _run_optimizer_case/_run_toolbox_case above.
def _run_toolbox_cross_language_case(case):
    opt = case["optimizer"]
    construct = dict(opt["construct"])
    objective_name = construct.pop("objective", "DeJong")
    r = _core.optim_run(_optimizer_spec_json(construct), _optimizer_fixture_objective(objective_name))
    for a in opt["assertions"]:
        if a["method"] == "value":
            _check(r["value"], a)
        elif a["method"] == "parameter":
            _check(r["parameters"][a["args"][0]], a)
        elif a["method"] == "status":
            assert r["status"] == a["expected"]
        else:
            raise KeyError(f"unknown toolbox_cross_language optimizer assertion method: {a['method']}")

    for sub in ("sobol", "stratify"):
        block = case[sub]
        options_dict = dict(block.get("options", {}))
        if sub == "sobol":
            options_dict["path"] = str(files("corehydropy") / "data" / "new-joe-kuo-6.21201")
        options = json.dumps(options_dict)
        for a in block["assertions"]:
            r = _core.toolbox_run("sampling", sub, [], options)
            _check(_toolbox_select(r, a, "sampling"), a)


def _load_cases():
    out = []
    for fx in sorted(_fixtures_dir().rglob("*.json")):
        spec = json.loads(fx.read_text(encoding="utf-8"))
        kind = spec.get("kind")
        if kind == "special_function":
            # Only the targets in _ROUTED_SPECIAL_FUNCTION_TARGETS are exposed to Python (see
            # _run_special_function_case above); every other special_function target is
            # validated in C++ only and generates no Python case.
            file_target = spec.get("target")
            for case in spec["cases"]:
                target = case.get("target", file_target)
                if target not in _ROUTED_SPECIAL_FUNCTION_TARGETS:
                    continue
                out.append(("special_function", target, {}, case))
            continue
        # Only validate univariate_distribution / multivariate_distribution /
        # bivariate_copula / mcmc_sampler / toolbox fixtures; skip other kinds which are
        # validated in C++ only and are not exposed to the Python package.
        if kind not in (
            "univariate_distribution",
            "multivariate_distribution",
            "bivariate_copula",
            "mcmc_sampler",
            "bootstrap",
            "model_estimation",
            "analysis",
            "data_utility",
            "goodness_of_fit",
            "toolbox",
            "optimizer",
            "callback",
            "callback_cross_language",
            "toolbox_cross_language",
        ):
            continue
        for case in spec["cases"]:
            out.append((kind, spec.get("target", spec.get("group", kind)), spec.get("datasets", {}), case))
    return out


CASES = _load_cases()


@pytest.mark.parametrize(
    "kind,target,datasets,case", CASES, ids=[f"{k}:{t}:{c['name']}" for k, t, _, c in CASES]
)
def test_fixture_case(kind, target, datasets, case):
    if kind == "special_function":
        actual = _run_special_function_case(target, case["args"])
        for a in case["assertions"]:
            _check(actual, a)
        return

    if kind == "toolbox":
        _run_toolbox_case(target, case, datasets)
        return

    if kind == "optimizer":
        _run_optimizer_case(case)
        return

    if kind == "callback":
        _run_callback_case(case)
        return

    if kind == "callback_cross_language":
        _run_callback_cross_language_case(case)
        return

    if kind == "toolbox_cross_language":
        _run_toolbox_cross_language_case(case)
        return

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

    if kind == "goodness_of_fit":
        fn = case["function"]
        args = case.get("args", [])
        obs = [_num(v) for v in datasets[case["observed_dataset"]]] if "observed_dataset" in case else []
        mod = [_num(v) for v in datasets[case["modeled_dataset"]]] if "modeled_dataset" in case else []
        actual = _dispatch_goodness_of_fit(fn, args, obs, mod)
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
        _run_multivariate_case(target, case["construct"], case["assertions"])
        return

    if target in _COMPOSITE_TARGETS:
        _run_composite_case(target, case["construct"], case["assertions"], datasets)
        return

    is_gev = target == "GeneralizedExtremeValue"
    if is_gev:
        g = _build_gev(case["construct"], datasets)
    else:
        params = _build_params(target, case["construct"], datasets)
    for a in case["assertions"]:
        args = a.get("args", [])
        if is_gev:
            actual = _dispatch_gev(g, a["method"], args)
        else:
            actual = _dispatch_generic(target, params, a["method"], args)
        _check(actual, a)
