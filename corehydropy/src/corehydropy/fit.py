"""The user-facing fit surface: point-estimate fits (:func:`fit_mle`/:func:`fit_map`), a
Bayesian MCMC fit (:func:`fit_bayesian`), and a generalized-method-of-moments fit
(:func:`fit_gmm`) over the shared C++ ``run_fit``
(``core/include/corehydro/estimation/support/fit_runner.hpp``).

Mirrors ``corehydror``'s ``R/fit.R`` argument for argument: the construct assembly
(:func:`_fit_input`), the sampler-knob validation table, the derived Bayesian warmup default,
and the three-way :meth:`Fit.confint` dispatch all match the shipped R behaviour, not just the
original design sketch. Because both packages call the identical compiled core with a bit-exact
Mersenne Twister, a seeded call returns identical numbers in either language -- in particular
:attr:`Fit.draws` is laid out ``[iteration, chain, parameter]``, the same axis order
``corehydror`` returns.

A :class:`Fit` is a plain Python object (dicts, numpy arrays, and a nested
:class:`~corehydropy.models.Model`): nothing here holds a C++ object, so a fit can be printed,
pickled, and compared against the R package unchanged.
"""

from __future__ import annotations

import json
import math

import numpy as np

from .analysis import _analysis_input
from .data import AnalysisData
from .models import Model
from ._core import (
    fit_diagnostics as _fit_diagnostics_core,
    fit_quantile_variance as _fit_quantile_variance_core,
    fit_run as _fit_run,
)

__all__ = ["Fit", "fit_mle", "fit_map", "fit_bayesian", "fit_gmm", "fit_diagnostics", "quantile_variance"]

_KNOWN_OPTIMIZERS = (
    "NelderMead", "Brent", "BFGS", "Powell", "DifferentialEvolution", "MultilevelSingleLinkage",
)

# Sampler-specific knobs, by sampler. A knob the chosen sampler ignores is an error rather than a
# silent no-op, because a silently dropped tuning argument looks like the sampler is broken.
# Confirmed against BayesianAnalysis::set_up_sampler (core/include/corehydro/estimation/
# bayesian_analysis.hpp:456-492) and each sampler's own knob fields (numerics/sampling/mcmc/
# {demcz,demczs,arwmh,nuts}.hpp): NUTS's ctor takes a fixed step size (0.1, no BestFit knob for
# it) ahead of `max_tree_depth`, so NUTS does NOT read a `scale` knob despite `scale` being a
# knob name the table also carries for ARWMH -- matching corehydror's R/fit.R `sampler_knobs`.
_SAMPLER_KNOBS = {
    "DEMCz": ("jump", "jump_threshold", "noise"),
    "DEMCzs": ("jump", "jump_threshold", "snooker_threshold", "noise"),
    "ARWMH": ("scale", "beta"),
    "NUTS": ("max_tree_depth",),
}
_KNOWN_SAMPLERS = tuple(_SAMPLER_KNOBS)
_INT_KNOBS = {"max_tree_depth"}

_KNOWN_GMM_STRATEGIES = ("OneStep", "TwoStep", "Iterative")

# Point-estimator name -> PointEstimateType, by its C# name (BayesianAnalysis's
# PointEstimateType enum: PosteriorMean, PosteriorMode -- the MAP). Matches corehydror's R/fit.R
# `known_point_estimators`.
_KNOWN_POINT_ESTIMATORS = ("PosteriorMean", "PosteriorMode")

# Fit targets run_fit_diagnostics (fit_runner.hpp) actually populates -- MaximumLikelihood carries
# no posterior/Hessian-at-a-point-estimate diagnostics surface, so it is deliberately absent here.
_FIT_DIAGNOSTICS_TARGETS = ("MaximumAPosteriori", "BayesianAnalysis", "GMM")


def _none_if_nan(x: float):
    """NaN -> None; anything else passes through unchanged.

    Only GMM ever leaves ``log_likelihood``/``aic``/``bic`` at the runner's structural NaN
    default (GMM is method-of-moments and has no likelihood surface to report them from). R
    reports that as ``NA`` through the ``logLik()``/``AIC()``/``BIC()` generics while leaving the
    fit's own raw list slot untouched; Python has no separate raw-slot-vs-generic split -- a
    ``Fit`` property IS the public access path -- so ``None`` is applied directly here, chosen
    over ``math.nan`` because ``is None`` is the idiomatic "not available" check and because a
    stray ``float("nan")` silently poisons any downstream arithmetic a caller does with it,
    which is exactly what "no likelihood surface" should NOT do quietly.
    """
    return None if isinstance(x, float) and math.isnan(x) else x


# --- construct assembly for the shared fit runner ------------------------------------------
#
# The Python twin of corehydror's `fit_input()` (R/fit.R). Reuses `_analysis_input()` (analysis.py)
# so the vector path and the model path build the identical model JSON the analyses already rely
# on -- the two cannot drift apart. Unlike R (which has no JSON parser and so splices settings
# JSON as a raw string after the model's own pre-serialized JSON -- see fit.R's header note),
# Python has `json` in the standard library: `_analysis_input`'s returned JSON string is parsed
# back into a dict, merged with `settings`, and re-serialized once. That round-trip is exact --
# `json.dumps` writes floats with `repr()`, the same shortest-round-trip representation
# `corehydror`'s `spec_number` documents relying on -- so this is a simplification, not a
# precision risk.
def _fit_input(model, distribution, settings: dict):
    # The AnalysisData branch is left to `_analysis_input`'s own, more specific message ("pass an
    # AnalysisData frame through a model, ..."); this guard only fires for a plain vector.
    if not isinstance(model, Model) and not isinstance(model, AnalysisData) and distribution is None:
        raise ValueError(
            "give a `distribution` name when `model` is a plain numeric vector, e.g. "
            'fit_mle(peaks, "Normal")'
        )
    base_spec = model.spec if isinstance(model, Model) else {"family": str(distribution), "dataset": "data"}
    model_json, dataset = _analysis_input(model, lambda: base_spec)
    construct = {"model": json.loads(model_json), **settings}
    return json.dumps(construct), dataset, base_spec


# Internal: confint()'s Bayesian rebuild path needs the IDENTICAL construct the fit was
# originally built from, with only `credible_interval_width` overridden -- every other setting
# (sampler, seed, iterations, chains, thinning, sampler-specific knobs) stays byte-identical, so
# `sampler_->sample()` reproduces the same seeded chain and only the post-hoc credible-interval
# quantile computation changes (see `apply_bayesian_settings` in fit_runner.hpp). Reconstructing
# the settings dict from scratch (the way `_fit_input` builds a fresh construct) would silently
# drop any sampler knob (jump/noise/scale/beta/...) the original `fit_bayesian()` call passed
# through `**knobs`, since a `Fit` does not carry those back out.
def _inject_credible_interval_width(construct_json: str, level: float) -> str:
    construct = json.loads(construct_json)
    construct["credible_interval_width"] = float(level)
    return json.dumps(construct)


class Fit:
    """A fitted model: parameter estimates plus whichever estimator diagnostics the target
    populates.

    Build one with :func:`fit_mle`, :func:`fit_map`, :func:`fit_bayesian`, or :func:`fit_gmm`
    rather than directly. ``MaximumLikelihood``/``MaximumAPosteriori`` fits additionally carry
    the Hessian-based covariance and (optionally) profile-likelihood intervals; a
    ``BayesianAnalysis`` fit carries the raw chains and posterior summary; a ``GMM`` fit carries
    the sandwich covariance and the J-statistic overidentification test.

    Attributes
    ----------
    method : str
        ``"MaximumLikelihood"``, ``"MaximumAPosteriori"``, ``"BayesianAnalysis"``, or ``"GMM"``.
    parameters : dict
        Parameter name -> fitted value.
    parameter_names : list of str
        ``list(parameters)``, kept as its own attribute to match the parameter axis order of
        :attr:`covariance`/:attr:`draws`/etc.
    covariance, correlation : numpy.ndarray or None
        ``n x n``, aligned with :attr:`parameter_names`. ``None`` when not computed (MLE/MAP with
        ``hessian=False``).
    standard_errors : dict or None
        Parameter name -> standard error. ``None`` when not computed.
    log_likelihood, aic, bic : float or None
        ``None`` for a GMM fit (method-of-moments has no likelihood surface to report them
        from); see :func:`_none_if_nan` for why ``None`` rather than ``nan`` was chosen.
    prior_log_likelihood : float
        The prior half of the log-posterior. Kept as the runner's raw value (including its
        structural ``nan`` for GMM) -- this is bookkeeping, not a primary reporting field.
    nobs : int
        Number of observations the fit was computed against (``0`` for GMM, which the runner
        never populates).
    converged : bool
    status : str
        ``"Success"``, ``"MaximumIterationsReached"``, etc.
    function_evaluations : int or None
        MLE/MAP only; ``None`` for Bayesian/GMM fits (the runner never populates it there).
    model : Model
        The fitted model: :attr:`parameters` applied to the construct's spec. Simulate from it
        with :func:`~corehydropy.model_simulate`.
    profile : dict or None
        MLE/MAP only, present when the fit was built with ``profile=True``: parameter name ->
        ``profile_bins x 2`` array with columns ``[value, log_likelihood]``.
    draws : numpy.ndarray or None
        Bayesian only: the raw chains, shape ``(n_iterations, n_chains, n_params)`` -- the same
        ``[iteration, chain, parameter]`` axis order ``corehydror`` returns.
    posterior_summary : dict or None
        Bayesian only: ``parameter_names`` plus the arrays ``mean``, ``median``, ``sd``,
        ``lower``, ``upper``, ``rhat``, ``ess``, each aligned with ``parameter_names``. (Named
        ``posterior_summary`` rather than R's ``$summary`` to avoid colliding with the
        :meth:`summary` text-report method Python's class namespace forces onto one object.)
    acceptance_rates : numpy.ndarray or None
        Bayesian only, one value per chain.
    warmup : int or None
        Bayesian only: the warmup actually used (see :func:`fit_bayesian`'s derived default).
    credible_level : float or None
        Bayesian only: the credible level :attr:`posterior_summary`'s ``lower``/``upper`` were
        computed at.
    dic, waic, looic : float or None
        Bayesian only, the usual Bayesian goodness-of-fit scalars.
    j_stat, j_stat_pval : float or None
        GMM only. Bulletin 17C is always just-identified, so ``j_stat_pval`` is structurally
        ``None`` -- there is no over-identified case to report a p-value for.
    gmm_iterations, converged_within_tolerance, optimizer_fallback_count
        GMM only, the estimator's own bookkeeping.
    """

    def __init__(
        self, *, method, parameters, parameter_names, log_likelihood, prior_log_likelihood, aic,
        bic, nobs, converged, status, model, spec, dataset, construct_json,
        covariance=None, standard_errors=None, correlation=None, function_evaluations=None,
        profile=None, profile_intervals=None, optimizer=None,
        draws=None, posterior_summary=None, acceptance_rates=None, warmup=None,
        credible_level=None, dic=None, waic=None, looic=None,
        j_stat=None, j_stat_pval=None, gmm_iterations=None,
        converged_within_tolerance=None, optimizer_fallback_count=None,
    ) -> None:
        self.method = method
        self.parameters = parameters
        self.parameter_names = parameter_names
        self.log_likelihood = log_likelihood
        self.prior_log_likelihood = prior_log_likelihood
        self.aic = aic
        self.bic = bic
        self.nobs = nobs
        self.converged = converged
        self.status = status
        self.model = model
        self.covariance = covariance
        self.standard_errors = standard_errors
        self.correlation = correlation
        self.function_evaluations = function_evaluations
        self.profile = profile
        self.optimizer = optimizer
        self.draws = draws
        self.posterior_summary = posterior_summary
        self.acceptance_rates = acceptance_rates
        self.warmup = warmup
        self.credible_level = credible_level
        self.dic = dic
        self.waic = waic
        self.looic = looic
        self.j_stat = j_stat
        self.j_stat_pval = j_stat_pval
        self.gmm_iterations = gmm_iterations
        self.converged_within_tolerance = converged_within_tolerance
        self.optimizer_fallback_count = optimizer_fallback_count
        # Internal bookkeeping (not part of the documented public surface, hence the leading
        # underscore -- Python's convention for "implementation detail", where R's plain list
        # has no equivalent access-control mechanism): the UNFITTED construct/spec/dataset this
        # fit was built from, so confint() and fit_diagnostics()/quantile_variance() can rerun
        # the identical construct through the C++ entry points and agree with the fit they came
        # from.
        self._profile_intervals = profile_intervals
        self._spec = spec
        self._dataset = dataset
        self._construct_json = construct_json

    def __repr__(self) -> str:
        bits = [f"{len(self.parameters)} parameters"]
        if self.method == "GMM":
            bits.append(f"j-stat={self.j_stat:g}")
        else:
            bits.append(f"log-likelihood={self.log_likelihood:g}")
        return f"<Fit {self.method} ({self.status}): {', '.join(bits)}>"

    def summary(self) -> str:
        """A multi-line text summary, mirroring R's ``print.corehydro_fit``/``summary.corehydro_fit``.

        Returns
        -------
        str
            Parameters, the common goodness-of-fit scalars (or the J-statistic for a GMM fit),
            and -- for a Bayesian fit -- the posterior R-hat/ESS, or -- for an MLE/MAP fit -- the
            standard errors.
        """
        lines = [repr(self), "parameters:"]
        for name, value in self.parameters.items():
            lines.append(f"  {name}: {value:g}")
        if self.method != "GMM":
            lines.append(
                f"log-likelihood: {self.log_likelihood:g}   aic: {self.aic:g}   "
                f"bic: {self.bic:g}   nobs: {self.nobs}"
            )
        if self.dic is not None:
            lines.append(f"dic: {self.dic:g}")
        if self.method == "GMM":
            pval = "NA" if self.j_stat_pval is None else f"{self.j_stat_pval:g}"
            lines.append(
                f"j-statistic: {self.j_stat:g}   p-value: {pval}   "
                f"gmm iterations: {self.gmm_iterations}"
            )
        if self.method in ("MaximumLikelihood", "MaximumAPosteriori"):
            lines.append(f"converged: {self.converged}   function evaluations: {self.function_evaluations}")
        else:
            lines.append(f"converged: {self.converged}")
        if self.posterior_summary is not None:
            lines.append("posterior summary (rhat, ess):")
            for i, name in enumerate(self.parameter_names):
                lines.append(
                    f"  {name}: mean={self.posterior_summary['mean'][i]:g} "
                    f"rhat={self.posterior_summary['rhat'][i]:g} ess={self.posterior_summary['ess'][i]:g}"
                )
        elif self.standard_errors is not None:
            lines.append("standard errors:")
            for name, se in self.standard_errors.items():
                lines.append(f"  {name}: {se:g}")
        return "\n".join(lines)

    def to_model(self) -> Model:
        """The fitted model (:attr:`model`), as a verb alongside :meth:`to_json`.

        Returns
        -------
        Model
            Same object as :attr:`model`.
        """
        return self.model

    def to_json(self) -> str:
        """The fitted model's spec as the JSON the shared C++ core parses.

        Serializes :attr:`model` (parameters, family, trends, ...) exactly like
        :meth:`~corehydropy.models.Model.to_json`; it does not include the covariance, draws, or
        any other diagnostic on this ``Fit``.

        Returns
        -------
        str
            The fitted model spec as JSON.
        """
        return self.model.to_json()

    def diagnostics(self) -> dict:
        """Estimation diagnostics off this fit. See :func:`fit_diagnostics`."""
        return fit_diagnostics(self)

    def quantile_variance(self, aep: float) -> float:
        """Delta-method variance of a fitted quantile. See :func:`quantile_variance`."""
        return quantile_variance(self, aep)

    def confint(self, level: float = 0.95) -> dict:
        """Confidence or credible intervals for the fit.

        MLE/MAP fits get profile-likelihood confidence intervals; a Bayesian fit gets posterior
        credible intervals. GMM has no interval surface (see below).

        For MLE/MAP: when the fit already carries a profile block at the requested level (built
        with ``profile=True`` or a prior :meth:`confint` call), those bounds are reused;
        otherwise the identical fit is re-run with profiling turned on -- a deterministic
        optimizer reproduces the identical point estimate, so the rebuild's confidence intervals
        are the ones the original fit would have carried had ``profile=True`` been requested at
        the matching level up front.

        For Bayesian: :attr:`Fit.posterior_summary`'s ``lower``/``upper`` are already the
        posterior credible interval at :attr:`Fit.credible_level` (``0.9``, `BayesianAnalysis`'s
        own class default -- :func:`fit_bayesian` has no argument to change it). When the
        requested ``level`` matches, those are returned directly; otherwise the identical seeded
        chain is re-run with ``credible_interval_width`` set to ``level`` (the same
        lazy-rebuild precedent as the profile path above -- re-sampling with an identical seed
        reproduces the same chain bit-for-bit, so only the post-hoc credible-interval quantile
        computation changes). Because a Bayesian fit is always built at the 0.9 credible level
        and this method's own default is 0.95, calling ``confint()`` with no argument on a
        Bayesian fit always takes the rebuild path.

        GMM is method-of-moments and has no likelihood or posterior to draw an interval from;
        calling this raises. Use :meth:`quantile_variance` for the delta-method variance of a
        fitted quantile instead.

        Parameters
        ----------
        level : float, default 0.95
            Confidence (MLE/MAP) or credible (Bayesian) level. Matches R's `confint()`
            convention, not `BayesianAnalysis`'s own 0.9 class default.

        Returns
        -------
        dict
            ``lower`` and ``upper``, each a dict of parameter name -> bound.

        Examples
        --------
        >>> from corehydropy import fit_mle, model_univariate
        >>> peaks = [12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600]
        >>> f = fit_mle(model_univariate("Normal", peaks))
        >>> ci = f.confint(level=0.9)  # doctest: +SKIP
        """
        if self.method == "GMM":
            raise ValueError(
                "confint() has no interval surface for a fit_gmm() fit: GMM is "
                "method-of-moments, not likelihood- or posterior-based, so there is no profile "
                "or credible interval to draw; use quantile_variance() for the delta-method "
                "variance of a fitted quantile instead"
            )

        if self.method == "BayesianAnalysis":
            if self.credible_level is not None and math.isclose(self.credible_level, level, rel_tol=1e-9):
                fit = self
            else:
                construct_json = _inject_credible_interval_width(self._construct_json, level)
                result = _fit_run("BayesianAnalysis", construct_json, self._dataset)
                fit = _new_fit_bayesian(
                    result, self._spec, self._dataset, construct_json, self.warmup, credible_level=level
                )
            return {
                "lower": dict(zip(fit.parameter_names, fit.posterior_summary["lower"])),
                "upper": dict(zip(fit.parameter_names, fit.posterior_summary["upper"])),
            }

        if self._profile_intervals is None or not math.isclose(
            self._profile_intervals["level"], level, rel_tol=1e-9
        ):
            bins = 100 if self._profile_intervals is None else self._profile_intervals["bins"]
            fit = _fit_optimized(
                self.method, Model(self._spec, self._dataset), None, self.optimizer,
                hessian=True, profile=True, profile_bins=bins, alpha=1 - level,
            )
        else:
            fit = self
        return {
            "lower": dict(fit._profile_intervals["lower"]),
            "upper": dict(fit._profile_intervals["upper"]),
        }


def _name_square(nested):
    """``n x n`` nested list -> numpy array, or ``None`` when empty (not computed)."""
    return None if not nested else np.asarray(nested)


def _new_fit_base(result: dict, base_spec: dict, dataset, construct_json: str) -> dict:
    """The field set every ``Fit`` carries regardless of target.

    ``result["model_spec"]`` is the fitted spec JSON string `fit_run` already built (the
    construct's model re-emitted with `parameter_values` overridden by the fitted values --
    fit_runner.hpp's `fitted_spec`); Python has a JSON parser, so `.model` is built directly from
    it rather than R's manual `base_spec$parameter_values <- ...` splice.
    """
    names = list(result["parameter_names"])
    parameters = dict(zip(names, result["parameters"]))
    fitted_model = Model(json.loads(result["model_spec"]), dataset)
    return dict(
        method=result["method"],
        parameters=parameters,
        parameter_names=names,
        log_likelihood=_none_if_nan(result["log_likelihood"]),
        prior_log_likelihood=result["prior_log_likelihood"],
        aic=_none_if_nan(result["aic"]),
        bic=_none_if_nan(result["bic"]),
        nobs=result["nobs"],
        converged=result["converged"],
        status=result["status"],
        model=fitted_model,
        spec=base_spec,
        dataset=dataset,
        construct_json=construct_json,
    )


def _new_fit(result: dict, base_spec: dict, dataset, optimizer: str, level: float, construct_json: str) -> Fit:
    """Build a ``Fit`` from a `fit_run` result for the MaximumLikelihood/MaximumAPosteriori
    targets -- adds the Hessian-based covariance stack, function-evaluation count, and the
    profile-likelihood grid/CI bookkeeping on top of `_new_fit_base`'s common fields.
    """
    base = _new_fit_base(result, base_spec, dataset, construct_json)
    names = base["parameter_names"]
    n = len(names)

    covariance = _name_square(result["covariance"])
    correlation = _name_square(result["correlation"])
    standard_errors = dict(zip(names, result["standard_errors"])) if result["standard_errors"] else None

    has_profile = len(result["profile_lower"]) > 0
    profile = None
    profile_intervals = None
    if has_profile:
        bins = result["profile_bins"]
        # `profile_grid` is a genuinely flat vector (not a pre-built matrix), n_params * bins *
        # 2, row-major [parameter][bin][value, profile log-likelihood] -- reshape with numpy
        # rather than by hand (see fit_runner.hpp's FitResult doc).
        grid = np.asarray(result["profile_grid"]).reshape(n, bins, 2)
        profile = {name: grid[i] for i, name in enumerate(names)}
        profile_intervals = {
            "lower": dict(zip(names, result["profile_lower"])),
            "upper": dict(zip(names, result["profile_upper"])),
            "bins": bins,
            "level": level,
        }

    return Fit(
        **base,
        covariance=covariance,
        standard_errors=standard_errors,
        correlation=correlation,
        function_evaluations=result["function_evaluations"],
        profile=profile,
        profile_intervals=profile_intervals,
        optimizer=optimizer,
    )


def _new_fit_bayesian(
    result: dict, base_spec: dict, dataset, construct_json: str, warmup: int, credible_level: float = 0.9,
) -> Fit:
    """Build a ``Fit`` from a `fit_run` result for the BayesianAnalysis target.

    `chain_dims` is `[n_chains, n_iterations, n_params]`; the runner flattens the raw chains
    CHAIN-major (`draws[((chain * n_iterations) + iter) * n_params + p]`), so
    `.reshape(n_chains, n_iter, n_params).transpose(1, 0, 2)` permutes it into
    `[iteration, chain, parameter]`, the same axis order `corehydror` returns.
    """
    base = _new_fit_base(result, base_spec, dataset, construct_json)
    names = base["parameter_names"]
    n_chains, n_iter, n_params = result["chain_dims"]
    draws = np.asarray(result["draws"]).reshape(n_chains, n_iter, n_params).transpose(1, 0, 2)

    posterior_summary = {
        "parameter_names": names,
        "mean": np.asarray(result["summary_mean"]),
        "median": np.asarray(result["summary_median"]),
        "sd": np.asarray(result["summary_sd"]),
        "lower": np.asarray(result["summary_lower"]),
        "upper": np.asarray(result["summary_upper"]),
        "rhat": np.asarray(result["rhat"]),
        "ess": np.asarray(result["ess"]),
    }

    return Fit(
        **base,
        draws=draws,
        posterior_summary=posterior_summary,
        acceptance_rates=np.asarray(result["acceptance_rates"]),
        warmup=warmup,
        credible_level=credible_level,
        dic=result["dic"],
        waic=result["waic"],
        looic=result["looic"],
    )


def _new_fit_gmm(result: dict, base_spec: dict, dataset, construct_json: str) -> Fit:
    """Build a ``Fit`` from a `fit_run` result for the GMM target -- the covariance stack (same
    shape as `_new_fit`'s, GMM's own sandwich covariance rather than a Hessian) plus the
    GMM-specific bookkeeping (J-statistic, iteration/convergence counters).
    """
    base = _new_fit_base(result, base_spec, dataset, construct_json)
    names = base["parameter_names"]
    covariance = _name_square(result["covariance"])
    correlation = _name_square(result["correlation"])
    standard_errors = dict(zip(names, result["standard_errors"])) if result["standard_errors"] else None
    j_stat_pval = _none_if_nan(result["j_stat_pval"])

    return Fit(
        **base,
        covariance=covariance,
        standard_errors=standard_errors,
        correlation=correlation,
        j_stat=result["j_stat"],
        j_stat_pval=j_stat_pval,
        gmm_iterations=result["gmm_iterations"],
        converged_within_tolerance=result["converged_within_tolerance"],
        optimizer_fallback_count=result["optimizer_fallback_count"],
    )


# Internal: shared body of fit_mle()/fit_map(), parameterized on the estimator target. `alpha` is
# not part of either verb's public signature; it exists so Fit.confint() can ask for a profile at
# an arbitrary level without a public `alpha` argument nobody else needs.
def _fit_optimized(target, model, distribution, optimizer, hessian, profile, profile_bins, alpha=0.1):
    optimizer = str(optimizer)
    if optimizer not in _KNOWN_OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer '{optimizer}'; expected one of {', '.join(_KNOWN_OPTIMIZERS)}"
        )
    profile = bool(profile)
    settings = {
        "optimizer": optimizer, "hessian": bool(hessian), "profile": profile,
        "profile_bins": int(profile_bins),
    }
    if profile:
        settings["alpha"] = float(alpha)

    construct_json, dataset, base_spec = _fit_input(model, distribution, settings)
    result = _fit_run(target, construct_json, dataset)
    return _new_fit(result, base_spec, dataset, optimizer, level=1 - alpha, construct_json=construct_json)


def fit_mle(model, distribution: str | None = None, optimizer: str = "NelderMead", hessian: bool = True,
            profile: bool = False, profile_bins: int = 100) -> Fit:
    """Maximum likelihood fit.

    Fit a model by maximum likelihood and return a fit carrying the parameter estimates, the
    Hessian-based covariance, and optimizer bookkeeping. Wraps the shared C++
    ``MaximumLikelihood`` ported from USACE-RMC RMC.BestFit.

    Parameters
    ----------
    model : Model or array_like
        A :func:`~corehydropy.model_univariate` (or any ``model_*()``) object, or a plain
        sequence of observations together with `distribution`. A model can bring censored
        observations (see :class:`~corehydropy.AnalysisData`), nonstationary trends (see
        :func:`~corehydropy.trend`), and parameter bounds or priors (see
        :func:`~corehydropy.model_parameter`).
    distribution : str, optional
        Distribution family name, required only when `model` is a plain sequence.
    optimizer : str, default "NelderMead"
        One of ``"NelderMead"``, ``"Brent"``, ``"BFGS"``, ``"Powell"``,
        ``"DifferentialEvolution"``, ``"MultilevelSingleLinkage"``.
    hessian : bool, default True
        Compute the covariance, standard errors, and correlation. A model with fewer than two
        parameters reports ``nan`` for all three.
    profile : bool, default False
        Also compute the profile likelihood and profile confidence intervals. Costs
        ``profile_bins * len(parameters)`` likelihood evaluations.
    profile_bins : int, default 100
        Number of bins in each parameter's profile.

    Returns
    -------
    Fit
        When `profile` is ``True``, ``.profile`` is a dict (one entry per parameter, keyed to
        match `.parameters`) of ``profile_bins x 2`` arrays with columns
        ``[value, log_likelihood]`` -- the profile-likelihood grid `Fit.confint`'s intervals are
        drawn from. ``None`` otherwise. See :func:`fit_bayesian` for the Bayesian surface.

    See Also
    --------
    fit_map, fit_bayesian, fit_gmm, fit_diagnostics

    Examples
    --------
    >>> from corehydropy import fit_mle, model_univariate
    >>> peaks = [12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600]
    >>> f = fit_mle(model_univariate("LogPearsonTypeIII", peaks))
    >>> sorted(f.parameters)  # doctest: +SKIP
    """
    return _fit_optimized("MaximumLikelihood", model, distribution, optimizer, hessian, profile, profile_bins)


def fit_map(model, distribution: str | None = None, optimizer: str = "NelderMead", hessian: bool = True,
            profile: bool = False, profile_bins: int = 100) -> Fit:
    """Maximum a posteriori fit.

    Fit a model by maximum a posteriori (the mode of the posterior formed from the model's own
    priors) and return a fit carrying the parameter estimates, the Hessian-based covariance, and
    optimizer bookkeeping. Wraps the shared C++ ``MaximumAPosteriori`` ported from USACE-RMC
    RMC.BestFit.

    Parameters
    ----------
    model : Model or array_like
        See :func:`fit_mle`.
    distribution : str, optional
        See :func:`fit_mle`.
    optimizer : str, default "NelderMead"
        See :func:`fit_mle`.
    hessian : bool, default True
        See :func:`fit_mle`.
    profile : bool, default False
        See :func:`fit_mle`.
    profile_bins : int, default 100
        See :func:`fit_mle`.

    Returns
    -------
    Fit
        See :func:`fit_mle`.

    See Also
    --------
    fit_mle, fit_bayesian, fit_gmm, fit_diagnostics

    Examples
    --------
    >>> from corehydropy import fit_map, model_univariate
    >>> peaks = [12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600]
    >>> f = fit_map(model_univariate("LogPearsonTypeIII", peaks))
    >>> sorted(f.parameters)  # doctest: +SKIP
    """
    return _fit_optimized("MaximumAPosteriori", model, distribution, optimizer, hessian, profile, profile_bins)


def fit_bayesian(
    model, distribution: str | None = None, sampler: str = "DEMCz", chains: int = 4,
    iterations: int = 3000, warmup: int | None = None, output_length: int = 10000,
    thinning_interval: int = -1, seed: int = 12345, point_estimator: str | None = None, **knobs,
) -> Fit:
    """Bayesian MCMC fit.

    Fit a model with a Bayesian MCMC analysis and return a fit carrying the raw chains, the
    posterior summary (mean/median/sd/credible interval, R-hat, effective sample size), and the
    usual Bayesian goodness-of-fit scalars. Wraps the shared C++ ``BayesianAnalysis`` ported from
    USACE-RMC RMC.BestFit -- the same estimator :func:`~corehydropy.univariate_analysis` and
    :func:`~corehydropy.estimation_diagnostics` build on.

    Parameters
    ----------
    model : Model or array_like
        See :func:`fit_mle`.
    distribution : str, optional
        See :func:`fit_mle`.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler. ``BayesianAnalysis`` can only construct these four; RWMH, HMC, and SNIS are
        real MCMC samplers but need :func:`~corehydropy.mcmc_sample` instead.
    chains : int, default 4
        Number of parallel Markov chains.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations, per chain.
    warmup : int, optional
        Number of warmup (burn-in) iterations. Defaults to ``max(50, iterations // 2)`` when
        omitted: ``BayesianAnalysis``'s own class default (1500) would otherwise silently trip
        its sampler's ``warmup <= iterations / 2`` guard for any `iterations` below about 3000.
    output_length : int, default 10000
        Number of posterior draws retained (thinned down from the raw chains) for the summary
        and any downstream uncertainty quantification.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.
    seed : int, default 12345
        PRNG seed for the sampler (fixed for reproducibility -- a seeded call returns identical
        draws in R and Python).
    point_estimator : {"PosteriorMean", "PosteriorMode"}, optional
        Which posterior summary ``.parameters`` reports. ``None`` (default) leaves
        ``BayesianAnalysis``'s own class default, which is ``"PosteriorMean"`` -- so by default
        ``.parameters`` is bit-identical to ``.posterior_summary["mean"]``. Pass
        ``"PosteriorMode"`` to report the MAP point instead.
    **knobs
        Sampler-specific tuning knobs. Passing a knob the chosen `sampler` does not use is an
        error: ``"DEMCz"`` accepts ``jump``, ``jump_threshold``, ``noise``; ``"DEMCzs"``
        additionally accepts ``snooker_threshold``; ``"ARWMH"`` accepts ``scale``, ``beta``;
        ``"NUTS"`` accepts ``max_tree_depth``.

    Returns
    -------
    Fit
        A fit with ``.method == "BayesianAnalysis"``. ``.draws`` is a 3-D array
        ``[iteration, chain, parameter]``. ``.posterior_summary`` has one entry per metric
        (``mean``, ``median``, ``sd``, ``lower``, ``upper``, ``rhat``, ``ess``), each an array
        aligned with ``.parameter_names``. ``.acceptance_rates`` has one entry per chain.
        ``.warmup`` records the warmup actually used. ``.dic``, ``.waic``, ``.looic`` are the
        usual Bayesian goodness-of-fit scalars. See :func:`fit_diagnostics` for leverage/
        influence diagnostics off a Bayesian fit.

    See Also
    --------
    fit_mle, fit_map, fit_gmm, fit_diagnostics, univariate_analysis

    Examples
    --------
    >>> from corehydropy import fit_bayesian, model_univariate
    >>> peaks = [12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600]
    >>> f = fit_bayesian(model_univariate("Normal", peaks), sampler="DEMCz", iterations=200,
    ...                  output_length=500, seed=12345)
    >>> f.posterior_summary["mean"]  # doctest: +SKIP
    """
    sampler = str(sampler)
    if sampler not in _KNOWN_SAMPLERS:
        raise ValueError(
            f"BayesianAnalysis cannot construct sampler '{sampler}'; it supports DEMCz, DEMCzs, "
            "ARWMH and NUTS. For RWMH, HMC or SNIS use mcmc_sample()."
        )

    allowed = _SAMPLER_KNOBS[sampler]
    bad = sorted(set(knobs) - set(allowed))
    if bad:
        raise ValueError(
            f"{sampler} does not use the knob{'s' if len(bad) > 1 else ''} {', '.join(bad)}; "
            f"{sampler} accepts: {', '.join(allowed) if allowed else '(none)'}"
        )

    if point_estimator is not None:
        point_estimator = str(point_estimator)
        if point_estimator not in _KNOWN_POINT_ESTIMATORS:
            raise ValueError(
                f"unknown point estimator '{point_estimator}'; expected one of "
                f"{', '.join(_KNOWN_POINT_ESTIMATORS)}"
            )

    iterations = int(iterations)
    warmup = max(50, iterations // 2) if warmup is None else int(warmup)

    settings = {
        "sampler": sampler,
        "seed": int(seed),
        "iterations": iterations,
        "warmup_iterations": warmup,
        "number_of_chains": int(chains),
        "output_length": int(output_length),
    }
    thinning_interval = int(thinning_interval)
    if thinning_interval > 0:
        settings["thinning_interval"] = thinning_interval
    if point_estimator is not None:
        settings["point_estimator"] = point_estimator
    # `knobs` names are a validated subset of _SAMPLER_KNOBS[sampler] above, disjoint from the
    # general settings keys assembled here, so a plain update is enough.
    for key, value in knobs.items():
        settings[key] = int(value) if key in _INT_KNOBS else float(value)

    construct_json, dataset, base_spec = _fit_input(model, distribution, settings)
    result = _fit_run("BayesianAnalysis", construct_json, dataset)
    return _new_fit_bayesian(result, base_spec, dataset, construct_json, warmup)


def fit_gmm(
    model, optimizer: str = "BFGS", strategy: str = "Iterative", max_gmm_iterations: int = 0,
) -> Fit:
    """Generalized method of moments fit (Bulletin 17C).

    Fit a :func:`~corehydropy.model_bulletin17c` model by the generalized method of moments and
    return a fit carrying the parameter estimates, the sandwich covariance, and the J-statistic
    overidentification test. Wraps the shared C++ ``GeneralizedMethodOfMoments`` ported from
    USACE-RMC RMC.BestFit -- the same estimator :func:`~corehydropy.bulletin17c_analysis` builds
    on. ``IGMMModel`` (the interface GMM fits against) has exactly one implementation,
    ``Bulletin17CDistribution``, so `fit_gmm` takes only a
    :func:`~corehydropy.model_bulletin17c` model; unlike :func:`fit_mle`/:func:`fit_map` there is
    no plain-sequence-plus-`distribution` convenience path.

    Parameters
    ----------
    model : Model
        A :func:`~corehydropy.model_bulletin17c` object.
    optimizer : str, default "BFGS"
        One of ``"BFGS"`` (matching ``GeneralizedMethodOfMoments``'s own class default),
        ``"NelderMead"``, ``"Brent"``, ``"Powell"``, ``"DifferentialEvolution"``,
        ``"MultilevelSingleLinkage"``.
    strategy : {"Iterative", "OneStep", "TwoStep"}, default "Iterative"
        GMM estimation strategy.
    max_gmm_iterations : int, default 0
        Maximum number of GMM iterations; ``0`` keeps the estimator's own default cap.

    Returns
    -------
    Fit
        A fit with ``.method == "GMM"``. Bulletin 17C is always just-identified (as many moment
        conditions as parameters), so ``.j_stat_pval`` is structurally ``None`` -- there is no
        over-identified case to report a p-value for (see ``docs/upstream-csharp-issues.md``).
        ``.gmm_iterations``, ``.converged_within_tolerance``, and
        ``.optimizer_fallback_count`` carry the estimator's own bookkeeping. See
        :func:`quantile_variance` for the delta-method variance of a fitted quantile, and
        :func:`fit_diagnostics` for leverage/influence diagnostics off a GMM fit.

    See Also
    --------
    fit_mle, fit_map, fit_bayesian, fit_diagnostics, quantile_variance, bulletin17c_analysis

    Examples
    --------
    >>> from corehydropy import fit_gmm, model_bulletin17c, quantile_variance
    >>> peaks = [12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600]
    >>> f = fit_gmm(model_bulletin17c(peaks))
    >>> sorted(f.parameters)  # doctest: +SKIP
    >>> quantile_variance(f, 0.01) > 0
    True
    """
    optimizer = str(optimizer)
    if optimizer not in _KNOWN_OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer '{optimizer}'; expected one of {', '.join(_KNOWN_OPTIMIZERS)}"
        )
    strategy = str(strategy)
    if strategy not in _KNOWN_GMM_STRATEGIES:
        raise ValueError(
            f"unknown GMM estimation strategy '{strategy}'; expected one of "
            f"{', '.join(_KNOWN_GMM_STRATEGIES)}"
        )
    if not isinstance(model, Model) or model.spec.get("type") != "bulletin17c":
        raise ValueError("fit_gmm fits a bulletin17c model only; build one with model_bulletin17c()")

    settings = {"optimizer": optimizer, "strategy": strategy}
    max_gmm_iterations = int(max_gmm_iterations)
    if max_gmm_iterations > 0:
        settings["max_gmm_iterations"] = max_gmm_iterations

    construct_json, dataset, base_spec = _fit_input(model, None, settings)
    result = _fit_run("GMM", construct_json, dataset)
    return _new_fit_gmm(result, base_spec, dataset, construct_json)


def fit_diagnostics(fit: Fit) -> dict:
    """Estimation diagnostics for a fit.

    Compute the estimation diagnostics available for a fit: Cook's distance, per-observation
    leverage, and observation influence (from the objective's Hessian at the fitted point) for
    :func:`fit_map` and :func:`fit_gmm` fits; leverage, PSIS-LOO Pareto-k, and prior influence
    for :func:`fit_bayesian` fits. Wraps the shared C++ Diagnostics layer
    (``LeverageDiagnostics``, ``InfluenceDiagnostics``, ``PriorInfluenceDiagnostics``).

    Reruns the fit's own construct through the corresponding C++ diagnostics method rather than
    reusing anything cached on `fit` -- for a :func:`fit_bayesian` result this reproduces the
    identical seeded chain (the construct carries the same explicit ``warmup``/``seed``/... the
    original fit used), so the diagnostics agree with the fit they were computed from.

    Parameters
    ----------
    fit : Fit
        A fit from :func:`fit_map`, :func:`fit_bayesian`, or :func:`fit_gmm`. :func:`fit_mle`
        fits are not supported: ``MaximumLikelihood`` has no posterior to diagnose.

    Returns
    -------
    dict
        ``cooks_distance`` and ``leverage`` (one value per observation) and
        ``observation_influence`` (an observations-by-parameters array) are populated for
        :func:`fit_map` and :func:`fit_gmm` fits. ``pareto_k`` (one per observation),
        ``max_pareto_k``, and ``prior_influence``/``prior_influence_names`` (one per parameter)
        are populated for :func:`fit_bayesian` fits. Fields the fit's target does not support
        come back empty.

    See Also
    --------
    fit_map, fit_bayesian, fit_gmm, estimation_diagnostics

    Examples
    --------
    >>> from corehydropy import fit_diagnostics, fit_map, model_univariate
    >>> peaks = [12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600]
    >>> d = fit_diagnostics(fit_map(model_univariate("Normal", peaks)))
    >>> len(d["cooks_distance"]) == len(peaks)
    True
    """
    if not isinstance(fit, Fit):
        raise TypeError("fit_diagnostics needs a Fit object")
    if fit.method not in _FIT_DIAGNOSTICS_TARGETS:
        raise ValueError(
            f"fit_diagnostics needs a fit_map(), fit_bayesian(), or fit_gmm() result; got a "
            f"{fit.method} fit"
        )
    d = _fit_diagnostics_core(fit.method, fit._construct_json, fit._dataset)
    d["cooks_distance"] = np.asarray(d["cooks_distance"])
    d["leverage"] = np.asarray(d["leverage"])
    d["observation_influence"] = (
        np.asarray(d["observation_influence"]) if d["observation_influence"] else None
    )
    d["pareto_k"] = np.asarray(d["pareto_k"])
    d["prior_influence"] = np.asarray(d["prior_influence"])
    return d


def quantile_variance(fit: Fit, aep: float) -> float:
    """Delta-method variance of a fitted quantile.

    The Cohn-style delta-method variance of the discharge quantile at a given annual exceedance
    probability, off a :func:`fit_gmm` fit's sandwich covariance. Wraps the shared C++
    ``Bulletin17CDistribution::quantile_variance``.

    Parameters
    ----------
    fit : Fit
        A fit from :func:`fit_gmm`.
    aep : float
        Annual exceedance probability (e.g. ``0.01`` for the 1% AEP / 100-year quantile).

    Returns
    -------
    float
        The variance of the fitted quantile at `aep`.

    See Also
    --------
    fit_gmm

    Examples
    --------
    >>> from corehydropy import fit_gmm, model_bulletin17c, quantile_variance
    >>> peaks = [12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600]
    >>> f = fit_gmm(model_bulletin17c(peaks))
    >>> quantile_variance(f, 0.01) > 0
    True
    """
    if not isinstance(fit, Fit) or fit.method != "GMM":
        raise ValueError("quantile_variance needs a fit_gmm() result")
    return _fit_quantile_variance_core(fit._construct_json, fit._dataset, float(aep))
