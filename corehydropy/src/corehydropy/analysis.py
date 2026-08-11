"""User-facing analysis wrappers (A10).

Thin Python wrappers over the compiled `_core` analysis bindings, mirroring the R package's
`univariate_analysis` / `fit_distributions` / `bulletin17c_analysis` signatures and semantics.
Because both packages call the identical compiled core with a bit-exact Mersenne Twister, a seeded
call returns identical numbers in either language, so the spec assembly and seed plumbing here
match corehydror/R/analysis.R exactly.
"""

from __future__ import annotations

import json

import numpy as np

from .data import AnalysisData
from .models import Model
from ._core import (
    analysis_b17c_run as _b17c_run,
    analysis_diagnostics_run as _diagnostics_run,
    analysis_extended_run as _extended_run,
    analysis_family_run as _family_run,
    analysis_fit_distributions as _fit_distributions,
    analysis_univariate_run as _univariate_run,
)



def _require(value, name):
    """Argument required on the plain-sequence path, supplied by the model otherwise."""
    if value is None:
        raise TypeError(
            f"`{name}` is required unless `data` is a Model from a model_*() constructor"
        )
    return value


def _analysis_input(data, build, expected_type=None, expected_subtype=None):
    """Resolve an analysis's first argument.

    Accepts either a plain sequence of observations (the convenience path every analysis has
    always had) or a :class:`~corehydropy.models.Model` from one of the ``model_*()``
    constructors. ``build`` assembles the spec for the sequence path; ``expected_type`` and
    ``expected_subtype`` guard a mismatched model here, with a clear message, rather than
    letting it fail deep in the C++ dispatch.

    A model built over an :class:`~corehydropy.data.AnalysisData` frame carries its
    observations inline, so the separate dataset vector is empty and the spec builder prefers
    the inline frame.
    """
    if isinstance(data, Model):
        actual = data.spec.get("type", "univariate_distribution")
        if expected_type is not None and actual != expected_type:
            raise ValueError(
                f"this analysis takes a {expected_type} model, but was given a {actual} model"
            )
        if expected_subtype is not None and data.spec.get("subtype") != expected_subtype:
            raise ValueError(
                f"this analysis takes a {expected_subtype} model, but was given a "
                f"{data.spec.get('subtype')} model"
            )
        return data.to_json(), list(data.dataset)
    if isinstance(data, AnalysisData):
        raise TypeError(
            "pass an AnalysisData frame through a model, e.g. "
            'univariate_analysis(model_univariate("Normal", data))'
        )
    return json.dumps(build()), [float(v) for v in np.asarray(data).ravel()]


def _extended_data_block(data):
    """The data half of an extended-analysis construct.

    A plain sequence goes into the separate ``datasets`` map under the key the model spec names;
    an :class:`~corehydropy.data.AnalysisData` frame travels inline in the model spec, and the
    datasets map then carries nothing.
    """
    if isinstance(data, AnalysisData):
        return {"data_frame": data.spec}, {"data": []}
    return {"dataset": "data"}, {"data": [float(v) for v in np.asarray(data).ravel()]}


def _extended_model_spec(data, distribution, expected_type):
    """The model half of an extended-analysis construct over a single univariate distribution."""
    if isinstance(data, Model):
        actual = data.spec.get("type", "univariate_distribution")
        if actual != expected_type:
            raise ValueError(
                f"this analysis takes a {expected_type} model, but was given a {actual} model"
            )
        return data.spec, {"data": list(data.dataset)}
    block, datasets = _extended_data_block(data)
    return {"family": str(_require(distribution, "distribution")), **block}, datasets


def univariate_analysis(
    data,
    distribution: str | None = None,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    exceedance_probabilities=None,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian univariate frequency analysis.

    Fit ``distribution`` to ``data`` with a Bayesian MCMC analysis and return the frequency
    (quantile) curve, the posterior mean and credible band, and goodness-of-fit scalars. The
    MCMC warmup (burn-in) length is set automatically to ``max(50, iterations // 2)`` and is
    not a user parameter.

    Parameters
    ----------
    data : array_like
        Observations to fit.
    distribution : str
        Distribution family name.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width (e.g. ``0.90`` for a 90% band).
    seed : int, default 12345
        PRNG seed for the sampler.
    exceedance_probabilities : array_like of float, optional
        Exceedance probabilities at which to tabulate the curve; when ``None``, the default
        ordinates are used.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        Keys ``parameters``, ``mode_curve``, ``mean_curve``, ``lower_ci``, ``upper_ci``
        (one value per exceedance ordinate) and the scalars ``aic``, ``bic``, ``dic``,
        ``rmse``.
    """
    model_json, values = _analysis_input(
        data, lambda: {"family": str(_require(distribution, "distribution")), "dataset": "data"},
        "univariate_distribution",
    )
    ep = [] if exceedance_probabilities is None else [float(v) for v in exceedance_probabilities]
    return _univariate_run(
        model_json, values, sampler, int(iterations), int(output_length),
        float(credible_level), int(seed), ep, int(thinning_interval),
    )


def fit_distributions(data) -> dict:
    """Fit and rank the 14 candidate distributions by maximum likelihood.

    Ranking is left to the caller.

    Parameters
    ----------
    data : array_like
        Observations to fit.

    Returns
    -------
    dict
        Equal-length lists with one entry per candidate: ``distribution`` (candidate name),
        ``aic``, ``bic``, ``rmse``, and ``converged`` (bool).
    """
    values = [float(v) for v in np.asarray(data).ravel()]
    return _fit_distributions(values)


def bulletin17c_analysis(
    data,
    uncertainty_method: str = "MultivariateNormal",
    output_length: int = 10000,
    seed: int = 12345,
    confidence_level: float = 0.90,
    exceedance_probabilities=None,
) -> dict:
    """Bulletin 17C (log-Pearson Type III) flood-frequency analysis.

    Fit the model by the generalized method of moments and return the Cohn-style delta-method
    confidence intervals, the fitted parameters, and the sandwich covariance.

    Parameters
    ----------
    data : array_like
        Annual peak observations.
    uncertainty_method : str, default "MultivariateNormal"
        Uncertainty-quantification method: ``"MultivariateNormal"``, ``"Bootstrap"``,
        ``"LinkedMultivariateNormal"``, or ``"BiasCorrectedBootstrap"``.
    output_length : int, default 10000
        Number of parameter-set draws used for uncertainty quantification.
    seed : int, default 12345
        PRNG seed for the uncertainty draw.
    confidence_level : float, default 0.90
        Confidence level for the intervals.
    exceedance_probabilities : array_like of float, optional
        Exceedance probabilities at which to tabulate the curve; when ``None``, the default
        ordinates are used.

    Returns
    -------
    dict
        Keys ``exceedance_probabilities``, ``point_estimates``, ``lower_ci``, ``upper_ci``,
        ``confidence_level``, ``beta1``, ``nu``, ``quantile_variance``, ``parameters``, and
        ``covariance`` (a nested p x p list).
    """
    model_json, values = _analysis_input(
        data,
        lambda: {"type": "bulletin17c", "family": "LogPearsonTypeIII", "dataset": "data"},
        "bulletin17c",
    )
    ep = [] if exceedance_probabilities is None else [float(v) for v in exceedance_probabilities]
    return _b17c_run(
        model_json, values, uncertainty_method, int(output_length), int(seed),
        float(confidence_level), ep,
    )


# --- D5: per-family analyses + diagnostics -------------------------------------------------
# The seven per-family analyses share the univariate_analysis result surface; each wrapper below
# assembles the matching model spec and calls the single `_family_run` dispatch. Spec assembly and
# seed plumbing match corehydror/R/analysis.R exactly, so a seeded call returns identical numbers.


def _family_run_dispatch(
    analysis_type, build, data, sampler, iterations, output_length, credible_level, seed,
    exceedance_probabilities, thinning_interval, training_time_steps, forecasting_time_steps,
    expected_type, expected_subtype=None,
) -> dict:
    model_json, values = _analysis_input(data, build, expected_type, expected_subtype)
    ep = [] if exceedance_probabilities is None else [float(v) for v in exceedance_probabilities]
    return _family_run(
        analysis_type, model_json, values, sampler, int(iterations), int(output_length),
        float(credible_level), int(seed), ep, int(thinning_interval), int(training_time_steps),
        int(forecasting_time_steps),
    )


def mixture_analysis(
    data,
    families=None,
    zero_inflated: bool = False,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    exceedance_probabilities=None,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian mixture-model frequency analysis.

    Fit a finite mixture of ``families`` to ``data`` with a Bayesian MCMC analysis.

    Parameters
    ----------
    data : array_like
        Observations to fit.
    families : sequence of str
        Distribution family name of each mixture component.
    zero_inflated : bool, default False
        Fit a zero-inflated mixture.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    exceedance_probabilities : array_like of float, optional
        Exceedance probabilities at which to tabulate the curve; when ``None``, the default
        ordinates are used.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The same shape as ``univariate_analysis``: ``parameters``, ``mode_curve``,
        ``mean_curve``, ``lower_ci``, ``upper_ci``, ``aic``, ``bic``, ``dic``, ``rmse``.

    See Also
    --------
    univariate_analysis
    """
    def _build():
        model = {
            "type": "mixture",
            "families": [str(f) for f in _require(families, "families")],
            "zero_inflated": bool(zero_inflated),
            "dataset": "data",
        }
        return model

    return _family_run_dispatch(
        "mixture", _build, data, sampler, iterations, output_length, credible_level, seed,
        exceedance_probabilities, thinning_interval, -1, -1,
        "mixture",
    )


def competing_risk_analysis(
    data,
    families=None,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    exceedance_probabilities=None,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian competing-risk frequency analysis.

    Fit a competing-risks model (the observed maximum of several independent parent
    ``families``) to ``data`` with a Bayesian MCMC analysis.

    Parameters
    ----------
    data : array_like
        Observations to fit.
    families : sequence of str
        Distribution family name of each independent parent.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    exceedance_probabilities : array_like of float, optional
        Exceedance probabilities at which to tabulate the curve; when ``None``, the default
        ordinates are used.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The same shape as ``univariate_analysis``: ``parameters``, ``mode_curve``,
        ``mean_curve``, ``lower_ci``, ``upper_ci``, ``aic``, ``bic``, ``dic``, ``rmse``.

    See Also
    --------
    univariate_analysis, mixture_analysis
    """
    def _build():
        model = {
            "type": "competing_risks",
            "families": [str(f) for f in _require(families, "families")],
            "dataset": "data",
        }
        return model

    return _family_run_dispatch(
        "competing_risk", _build, data, sampler, iterations, output_length, credible_level, seed,
        exceedance_probabilities, thinning_interval, -1, -1,
        "competing_risks",
    )


def point_process_analysis(
    data,
    threshold=None,
    total_years=None,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    exceedance_probabilities=None,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian point-process (peaks-over-threshold) frequency analysis.

    Fit a peaks-over-threshold point-process model to ``data`` with a Bayesian MCMC analysis.

    Parameters
    ----------
    data : array_like
        Observations to fit.
    threshold : float, optional
        Peaks-over-threshold value; the model default cascade applies when omitted.
    total_years : float, optional
        Total record length in years; the model default cascade applies when omitted.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    exceedance_probabilities : array_like of float, optional
        Exceedance probabilities at which to tabulate the curve; when ``None``, the default
        ordinates are used.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The same shape as ``univariate_analysis``: ``parameters``, ``mode_curve``,
        ``mean_curve``, ``lower_ci``, ``upper_ci``, ``aic``, ``bic``, ``dic``, ``rmse``.

    See Also
    --------
    univariate_analysis, mixture_analysis
    """
    def _build():
        model = {"type": "point_process", "dataset": "data"}
        if threshold is not None:
            model["threshold"] = float(threshold)
        if total_years is not None:
            model["total_years"] = float(total_years)
        return model

    return _family_run_dispatch(
        "point_process", _build, data, sampler, iterations, output_length, credible_level, seed,
        exceedance_probabilities, thinning_interval, -1, -1,
        "point_process",
    )


def ar_analysis(
    data,
    order_p: int = 1,
    include_intercept: bool = True,
    training_time_steps=None,
    forecasting_time_steps: int = 0,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian autoregressive AR(p) time-series analysis.

    Fit an AR(``order_p``) model to the observed series ``data`` and return the posterior
    forecast curve and credible band.

    Parameters
    ----------
    data : array_like
        Observed time series.
    order_p : int, default 1
        Autoregressive order p.
    include_intercept : bool, default True
        Include an intercept term.
    training_time_steps : int, optional
        Number of training time steps. Defaults to the model's own default
        (``max(30, floor(0.8 * n))``), which is invalid for short series -- set it
        explicitly (e.g. ``15``) when ``n`` is small.
    forecasting_time_steps : int, default 0
        Extends the forecast horizon past the observed series.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The same shape as ``univariate_analysis``: ``parameters``, ``mode_curve``,
        ``mean_curve``, ``lower_ci``, ``upper_ci``, ``aic``, ``bic``, ``dic``, ``rmse``.

    See Also
    --------
    univariate_analysis, ma_analysis, arima_analysis, arimax_analysis
    """
    def _build():
        model = {
            "type": "time_series",
            "subtype": "ar",
            "dataset": "data",
            "orders": {"p": int(order_p)},
            "include_intercept": bool(include_intercept),
        }
        return model

    tts = -1 if training_time_steps is None else int(training_time_steps)
    return _family_run_dispatch(
        "ar", _build, data, sampler, iterations, output_length, credible_level, seed, None,
        thinning_interval, tts, forecasting_time_steps,
        "time_series", "ar",
    )


def ma_analysis(
    data,
    order_q: int = 1,
    include_intercept: bool = True,
    training_time_steps=None,
    forecasting_time_steps: int = 0,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian moving-average MA(q) time-series analysis.

    Fit an MA(``order_q``) model to the observed series ``data`` and return the posterior
    forecast curve and credible band.

    Parameters
    ----------
    data : array_like
        Observed time series.
    order_q : int, default 1
        Moving-average order q.
    include_intercept : bool, default True
        Include an intercept term.
    training_time_steps : int, optional
        Number of training time steps. Defaults to the model's own default
        (``max(30, floor(0.8 * n))``), which is invalid for short series -- set it
        explicitly (e.g. ``15``) when ``n`` is small.
    forecasting_time_steps : int, default 0
        Extends the forecast horizon past the observed series.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The same shape as ``univariate_analysis``: ``parameters``, ``mode_curve``,
        ``mean_curve``, ``lower_ci``, ``upper_ci``, ``aic``, ``bic``, ``dic``, ``rmse``.

    See Also
    --------
    ar_analysis, arima_analysis, arimax_analysis
    """
    def _build():
        model = {
            "type": "time_series",
            "subtype": "ma",
            "dataset": "data",
            "orders": {"q": int(order_q)},
            "include_intercept": bool(include_intercept),
        }
        return model

    tts = -1 if training_time_steps is None else int(training_time_steps)
    return _family_run_dispatch(
        "ma", _build, data, sampler, iterations, output_length, credible_level, seed, None,
        thinning_interval, tts, forecasting_time_steps,
        "time_series", "ma",
    )


def arima_analysis(
    data,
    order_p: int = 1,
    order_d: int = 0,
    order_q: int = 1,
    include_intercept: bool = True,
    training_time_steps=None,
    forecasting_time_steps: int = 0,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian ARIMA(p,d,q) time-series analysis.

    Fit an ARIMA(``order_p``, ``order_d``, ``order_q``) model to the observed series
    ``data`` and return the posterior forecast curve and credible band.

    Parameters
    ----------
    data : array_like
        Observed time series.
    order_p : int, default 1
        Autoregressive order p.
    order_d : int, default 0
        Differencing order d.
    order_q : int, default 1
        Moving-average order q.
    include_intercept : bool, default True
        Include an intercept term.
    training_time_steps : int, optional
        Number of training time steps. Defaults to the model's own default
        (``max(30, floor(0.8 * n))``), which is invalid for short series -- set it
        explicitly (e.g. ``15``) when ``n`` is small.
    forecasting_time_steps : int, default 0
        Extends the forecast horizon past the observed series.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The same shape as ``univariate_analysis``: ``parameters``, ``mode_curve``,
        ``mean_curve``, ``lower_ci``, ``upper_ci``, ``aic``, ``bic``, ``dic``, ``rmse``.

    See Also
    --------
    ar_analysis, ma_analysis, arimax_analysis
    """
    def _build():
        model = {
            "type": "time_series",
            "subtype": "arima",
            "dataset": "data",
            "orders": {"p": int(order_p), "d": int(order_d), "q": int(order_q)},
            "include_intercept": bool(include_intercept),
        }
        return model

    tts = -1 if training_time_steps is None else int(training_time_steps)
    return _family_run_dispatch(
        "arima", _build, data, sampler, iterations, output_length, credible_level, seed, None,
        thinning_interval, tts, forecasting_time_steps,
        "time_series", "arima",
    )


def arimax_analysis(
    data,
    order_p: int = 1,
    order_d: int = 0,
    order_q: int = 0,
    order_b: int = 0,
    trend: str = "None",
    include_intercept: bool = True,
    training_time_steps=None,
    forecasting_time_steps: int = 0,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian ARIMAX(p,d,q) time-series analysis with a deterministic ``trend``.

    Fit an ARIMAX model to the observed series ``data`` and return the posterior forecast
    curve and credible band. Covariate forecasting past the observed range is not exposed;
    run fit-only (``forecasting_time_steps = 0``) with covariates.

    Parameters
    ----------
    data : array_like
        Observed time series.
    order_p : int, default 1
        Autoregressive order p.
    order_d : int, default 0
        Differencing order d.
    order_q : int, default 0
        Moving-average order q.
    order_b : int, default 0
        Covariate (exogenous) order b.
    trend : str, default "None"
        Deterministic trend model name.
    include_intercept : bool, default True
        Include an intercept term.
    training_time_steps : int, optional
        Number of training time steps. Defaults to the model's own default
        (``max(30, floor(0.8 * n))``), which is invalid for short series -- set it
        explicitly (e.g. ``15``) when ``n`` is small.
    forecasting_time_steps : int, default 0
        Extends the forecast horizon past the observed series.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The same shape as ``univariate_analysis``: ``parameters``, ``mode_curve``,
        ``mean_curve``, ``lower_ci``, ``upper_ci``, ``aic``, ``bic``, ``dic``, ``rmse``.

    See Also
    --------
    ar_analysis, ma_analysis, arima_analysis
    """
    def _build():
        model = {
            "type": "time_series",
            "subtype": "arimax",
            "dataset": "data",
            "orders": {"p": int(order_p), "d": int(order_d), "q": int(order_q), "b": int(order_b)},
            "include_intercept": bool(include_intercept),
            "trend": str(trend),
        }
        return model

    tts = -1 if training_time_steps is None else int(training_time_steps)
    return _family_run_dispatch(
        "arimax", _build, data, sampler, iterations, output_length, credible_level, seed, None,
        thinning_interval, tts, forecasting_time_steps,
        "time_series", "arimax",
    )


def estimation_diagnostics(
    data,
    distribution: str | None = None,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    seed: int = 12345,
    thinning_interval: int = -1,
    thin_every: int = 10,
) -> dict:
    """Bayesian estimation diagnostics (leverage / influence / prior influence).

    Fit ``distribution`` to ``data`` with a Bayesian MCMC analysis and compute the three
    diagnostics off that fit.

    Parameters
    ----------
    data : array_like
        Observations to fit.
    distribution : str
        Distribution family name.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples.
    seed : int, default 12345
        PRNG seed for the sampler.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.
    thin_every : int, default 10
        Prior-influence posterior thinning stride.

    Returns
    -------
    dict
        Three sub-dicts:

        - ``leverage`` : the lists ``index``, ``leverage``, ``fit_influence``,
          ``variance_influence``, ``value``; ``prior_leverage``; and ``total_leverage``,
          ``total_fit_influence``, ``total_variance_influence``.
        - ``influence`` : the lists ``pareto_k``, ``elpd_loo``; and ``count``,
          ``mean_pareto_k``, ``max_pareto_k``, ``count_pareto_k_above_05`` / ``_07`` /
          ``_10``, ``proportion_problematic``, ``is_reliable``.
        - ``prior_influence`` : ``count``, ``prior_precision_share``,
          ``total_prior_log_likelihood``, ``total_data_log_likelihood``,
          ``prior_to_data_ratio``, ``is_prior_influential``, ``mean_prior_precision_share``.
    """
    model_json, values = _analysis_input(
        data, lambda: {"family": str(_require(distribution, "distribution")), "dataset": "data"},
        "univariate_distribution",
    )
    return _diagnostics_run(
        model_json, values, sampler, int(iterations), int(output_length), int(seed),
        int(thinning_interval), int(thin_every),
    )


# --- X11: the five remaining analyses + BootstrapAnalysis + predictive checks ----------------
# Each wrapper assembles the construct dict the shared C++ runner
# (corehydro::analyses::support::run_extended_analysis) understands and calls the single
# analysis_extended_run dispatch. The R twins (corehydror) build the identical construct, so a seeded
# call returns identical numbers in either language.


def _extended(target: str, construct: dict, datasets: dict) -> dict:
    return _extended_run(target, json.dumps(construct), json.dumps(datasets))


def composite_analysis(
    data,
    families,
    composite_type: str = "CompetingRisks",
    average_method: str = "AIC",
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    exceedance_probabilities=None,
    thinning_interval: int = -1,
) -> dict:
    """Composite frequency analysis over one child analysis per ``families`` entry.

    Combines the child univariate frequency analyses (each fit to ``data``) into a single
    composite curve. Wraps the shared C++ ``CompositeAnalysis``.

    Parameters
    ----------
    data : array_like
        Observations to fit.
    families : sequence of str
        Distribution family name of each child analysis.
    composite_type : {"CompetingRisks", "Mixture", "ModelAverage"}, default "CompetingRisks"
        How the child curves are combined.
    average_method : str, default "AIC"
        Selects the model-averaging criterion.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    exceedance_probabilities : array_like of float, optional
        Exceedance probabilities at which to tabulate the curve; when ``None``, the default
        ordinates are used.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The same shape as ``univariate_analysis``: ``parameters``, ``mode_curve``,
        ``mean_curve``, ``lower_ci``, ``upper_ci``, ``aic``, ``bic``, ``dic``, ``rmse``.

    See Also
    --------
    univariate_analysis
    """
    _block, _datasets = _extended_data_block(data)
    construct = {
        "model": {"families": [str(f) for f in families], **_block},
        "composite_type": str(composite_type),
        "average_method": str(average_method),
        "sampler": str(sampler),
        "iterations": int(iterations),
        "output_length": int(output_length),
        "credible_level": float(credible_level),
        "seed": int(seed),
        "thinning_interval": int(thinning_interval),
    }
    if exceedance_probabilities is not None:
        construct["exceedance_probabilities"] = [float(v) for v in exceedance_probabilities]
    return _extended("CompositeAnalysis", construct, _datasets)


def spatial_gev_analysis(
    coordinates,
    at_site_data,
    cross_validation: bool = False,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    number_of_chains: int = 4,
    exceedance_probabilities=None,
    thinning_interval: int = -1,
) -> dict:
    """Hierarchical spatial-GEV frequency analysis over gauged sites.

    Wraps the shared C++ ``SpatialGEVAnalysis``.

    Parameters
    ----------
    coordinates : array_like
        Site coordinates, one ``[x, y]`` row per site.
    at_site_data : array_like
        At-site observation matrix, ``[observations x sites]``.
    cross_validation : bool, default False
        Also compute the leave-one-site-out diagnostics.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible bands.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    number_of_chains : int, default 4
        Number of MCMC chains.
    exceedance_probabilities : array_like of float, optional
        Exceedance probabilities at which to tabulate the curves; when ``None``, the
        default ordinates are used.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The regional frequency curve plus per-site GEV/quantile bands and, when
        ``cross_validation`` is true, the leave-one-site-out diagnostics.
    """
    coords = [[float(v) for v in row] for row in np.asarray(coordinates, dtype=float)]
    at_site = [[float(v) for v in row] for row in np.asarray(at_site_data, dtype=float)]
    construct = {
        "model": {"type": "spatial_gev", "coordinates": coords, "at_site_data": at_site},
        "cross_validation": bool(cross_validation),
        "sampler": str(sampler),
        "iterations": int(iterations),
        "output_length": int(output_length),
        "number_of_chains": int(number_of_chains),
        "credible_level": float(credible_level),
        "seed": int(seed),
        "thinning_interval": int(thinning_interval),
    }
    if exceedance_probabilities is not None:
        construct["exceedance_probabilities"] = [float(v) for v in exceedance_probabilities]
    return _extended("SpatialGEVAnalysis", construct, {"unused": [0]})


def _bivariate_model(
    marginal_x_family, marginal_x_data, marginal_x_parameters,
    marginal_y_family, marginal_y_data, marginal_y_parameters, copula, estimation_method,
) -> dict:
    return {
        "type": "bivariate",
        "copula": str(copula),
        "estimation_method": str(estimation_method),
        "marginal_x": {
            "family": str(marginal_x_family),
            "data": [float(v) for v in np.asarray(marginal_x_data).ravel()],
            "parameter_values": [float(v) for v in marginal_x_parameters],
        },
        "marginal_y": {
            "family": str(marginal_y_family),
            "data": [float(v) for v in np.asarray(marginal_y_data).ravel()],
            "parameter_values": [float(v) for v in marginal_y_parameters],
        },
    }


def bivariate_analysis(
    marginal_x_family,
    marginal_x_data,
    marginal_x_parameters,
    marginal_y_family,
    marginal_y_data,
    marginal_y_parameters,
    xy_x,
    xy_y,
    copula: str = "Normal",
    estimation_method: str = "InferenceFromMargins",
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    number_of_chains: int = 4,
    thinning_interval: int = -1,
) -> dict:
    """Bivariate (copula) joint-exceedance frequency analysis over two fixed marginals.

    Wraps the shared C++ ``BivariateAnalysis``.

    Parameters
    ----------
    marginal_x_family : str
        Distribution family name of the X marginal.
    marginal_x_data : array_like
        Observations for the X marginal.
    marginal_x_parameters : array_like of float
        Fixed parameter values of the X marginal.
    marginal_y_family : str
        Distribution family name of the Y marginal.
    marginal_y_data : array_like
        Observations for the Y marginal.
    marginal_y_parameters : array_like of float
        Fixed parameter values of the Y marginal.
    xy_x : array_like of float
        X values of the joint-exceedance ordinate grid.
    xy_y : array_like of float
        Y values of the joint-exceedance ordinate grid.
    copula : str, default "Normal"
        Bivariate copula name.
    estimation_method : str, default "InferenceFromMargins"
        Copula estimation method.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    number_of_chains : int, default 4
        Number of MCMC chains.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The AND-joint-exceedance mode/mean curve and credible band over the
        ``(xy_x, xy_y)`` ordinate grid.
    """
    construct = {
        "model": _bivariate_model(
            marginal_x_family, marginal_x_data, marginal_x_parameters,
            marginal_y_family, marginal_y_data, marginal_y_parameters, copula, estimation_method,
        ),
        "xy_x": [float(v) for v in xy_x],
        "xy_y": [float(v) for v in xy_y],
        "sampler": str(sampler),
        "iterations": int(iterations),
        "output_length": int(output_length),
        "number_of_chains": int(number_of_chains),
        "credible_level": float(credible_level),
        "seed": int(seed),
        "thinning_interval": int(thinning_interval),
    }
    return _extended("BivariateAnalysis", construct, {"unused": [0]})


def coincident_frequency_analysis(
    marginal_x_family,
    marginal_x_data,
    marginal_x_parameters,
    marginal_y_family,
    marginal_y_data,
    marginal_y_parameters,
    x_values,
    y_values,
    response,
    number_of_bins: int = 50,
    copula: str = "Normal",
    estimation_method: str = "InferenceFromMargins",
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    number_of_chains: int = 4,
    thinning_interval: int = -1,
) -> dict:
    """Coincident-frequency analysis: a fitted bivariate copula + an M x N response surface.

    Derives the annual-exceedance-probability curve of ``Z = f(X, Y)`` from the response
    grid. Wraps the shared C++ ``CoincidentFrequencyAnalysis`` (which internally fits the
    bivariate analysis).

    Parameters
    ----------
    marginal_x_family : str
        Distribution family name of the X marginal.
    marginal_x_data : array_like
        Observations for the X marginal.
    marginal_x_parameters : array_like of float
        Fixed parameter values of the X marginal.
    marginal_y_family : str
        Distribution family name of the Y marginal.
    marginal_y_data : array_like
        Observations for the Y marginal.
    marginal_y_parameters : array_like of float
        Fixed parameter values of the Y marginal.
    x_values : array_like of float
        Grid values ``x_i`` of the response surface.
    y_values : array_like of float
        Grid values ``y_j`` of the response surface.
    response : array_like
        M x N matrix ``Z[i, j] = f(x_i, y_j)``.
    number_of_bins : int, default 50
        Number of Z output bins.
    copula : str, default "Normal"
        Bivariate copula name.
    estimation_method : str, default "InferenceFromMargins"
        Copula estimation method.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    number_of_chains : int, default 4
        Number of MCMC chains.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The derived annual-exceedance-probability curve of ``Z = f(X, Y)``.

    See Also
    --------
    bivariate_analysis
    """
    resp = np.asarray(response, dtype=float)
    construct = {
        "model": _bivariate_model(
            marginal_x_family, marginal_x_data, marginal_x_parameters,
            marginal_y_family, marginal_y_data, marginal_y_parameters, copula, estimation_method,
        ),
        "x_values": [float(v) for v in x_values],
        "y_values": [float(v) for v in y_values],
        "response_rows": int(resp.shape[0]),
        "response_cols": int(resp.shape[1]),
        "response": [float(v) for v in resp.ravel(order="C")],
        "number_of_bins": int(number_of_bins),
        "sampler": str(sampler),
        "iterations": int(iterations),
        "output_length": int(output_length),
        "number_of_chains": int(number_of_chains),
        "credible_level": float(credible_level),
        "seed": int(seed),
        "thinning_interval": int(thinning_interval),
    }
    return _extended("CoincidentFrequencyAnalysis", construct, {"unused": [0]})


def rating_curve_analysis(
    stage,
    discharge,
    segments: int = 1,
    stage_bins=None,
    min_stage=None,
    max_stage=None,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    number_of_chains: int = 4,
    thinning_interval: int = -1,
) -> dict:
    """Stage-discharge rating-curve frequency analysis.

    Wraps the shared C++ ``RatingCurveAnalysis``.

    Parameters
    ----------
    stage : array_like
        Observed stage values.
    discharge : array_like
        Observed discharge values.
    segments : int, default 1
        Number of rating-curve segments.
    stage_bins : int, optional
        Number of stage grid points; when omitted, keeps the data-derived default.
    min_stage : float, optional
        Lower stage-grid bound; data-derived when omitted.
    max_stage : float, optional
        Upper stage-grid bound; data-derived when omitted.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples used to build the credible band.
    credible_level : float, default 0.90
        Credible-interval width.
    seed : int, default 12345
        PRNG seed for the sampler.
    number_of_chains : int, default 4
        Number of MCMC chains.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        The predicted-discharge mode/mean curve and credible band across the stage grid.
    """
    construct = {
        "model": {
            "type": "rating_curve",
            "segments": int(segments),
            "stage": [float(v) for v in np.asarray(stage).ravel()],
            "discharge": [float(v) for v in np.asarray(discharge).ravel()],
        },
        "sampler": str(sampler),
        "iterations": int(iterations),
        "output_length": int(output_length),
        "number_of_chains": int(number_of_chains),
        "credible_level": float(credible_level),
        "seed": int(seed),
        "thinning_interval": int(thinning_interval),
    }
    if stage_bins is not None:
        construct["stage_bins"] = int(stage_bins)
    if min_stage is not None:
        construct["min_stage"] = float(min_stage)
    if max_stage is not None:
        construct["max_stage"] = float(max_stage)
    return _extended("RatingCurveAnalysis", construct, {"unused": [0]})


def bootstrap_analysis(
    data,
    distribution: str,
    probabilities,
    estimation_method: str = "MaximumLikelihood",
    sample_size=None,
    replications: int = 1000,
    seed: int = 12345,
    alpha: float = 0.1,
) -> dict:
    """Parametric bootstrap confidence bands for a fitted distribution.

    Fits ``distribution`` to ``data``, then resamples it ``replications`` times to derive
    percentile confidence bands on the quantile curve at the non-exceedance
    ``probabilities``. Wraps the shared C++ ``BootstrapAnalysis`` (Numerics).

    Parameters
    ----------
    data : array_like
        Observations to fit.
    distribution : str
        Distribution family name.
    probabilities : array_like of float
        Non-exceedance probabilities at which to evaluate the quantile curve.
    estimation_method : str, default "MaximumLikelihood"
        Estimation method used to fit the distribution.
    sample_size : int, optional
        Bootstrap sample size; when omitted, uses ``len(data)``.
    replications : int, default 1000
        Number of bootstrap replications.
    seed : int, default 12345
        PRNG seed for the resampling.
    alpha : float, default 0.1
        Significance level of the confidence bands.

    Returns
    -------
    dict
        Percentile confidence bands on the quantile curve at ``probabilities``.
    """
    construct = {
        "model": {"family": str(distribution), "dataset": "data"},
        "estimation_method": str(estimation_method),
        "replications": int(replications),
        "seed": int(seed),
        "alpha": float(alpha),
        "probabilities": [float(v) for v in probabilities],
    }
    if sample_size is not None:
        construct["sample_size"] = int(sample_size)
    return _extended("BootstrapAnalysis", construct, {"data": [float(v) for v in np.asarray(data).ravel()]})


def prior_predictive_check(
    data,
    distribution: str | None = None,
    number_of_draws: int = 1000,
    sample_size=None,
    seed: int = 12345,
) -> dict:
    """Prior predictive check: sample from the model priors, simulate, summarize.

    Wraps the shared C++ ``PriorPredictiveCheck``.

    Parameters
    ----------
    data : array_like
        Observations defining the model dataset.
    distribution : str
        Distribution family name.
    number_of_draws : int, default 1000
        Number of prior draws.
    sample_size : int, optional
        Simulated dataset size per draw; when omitted, uses ``len(data)``.
    seed : int, default 12345
        PRNG seed.

    Returns
    -------
    dict
        ``number_of_valid_draws`` and the predictive quantile summaries
        (``summary_mean_quantiles`` etc., each the ``[2.5, 25, 50, 75, 97.5]%`` quantiles).
    """
    _model, _datasets = _extended_model_spec(data, distribution, "univariate_distribution")
    construct = {
        "model": _model,
        "number_of_draws": int(number_of_draws),
        "seed": int(seed),
    }
    if sample_size is not None:
        construct["sample_size"] = int(sample_size)
    return _extended("PriorPredictiveCheck", construct, _datasets)


def posterior_predictive_check(
    data,
    distribution: str | None = None,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    seed: int = 12345,
    number_of_replicates: int = 1000,
    thinning_interval: int = -1,
) -> dict:
    """Posterior predictive check: fit an MCMC, draw replicates, compute common p-values.

    Wraps the shared C++ ``PosteriorPredictiveCheck``.

    Parameters
    ----------
    data : array_like
        Observations to fit.
    distribution : str
        Distribution family name.
    sampler : {"DEMCz", "DEMCzs", "ARWMH", "NUTS"}, default "DEMCz"
        MCMC sampler.
    iterations : int, default 3000
        Number of post-warmup MCMC iterations.
    output_length : int, default 10000
        Number of posterior samples.
    seed : int, default 12345
        PRNG seed for the sampler.
    number_of_replicates : int, default 1000
        Number of posterior predictive replicates.
    thinning_interval : int, default -1
        MCMC thinning interval; ``-1`` keeps the sampler's own default.

    Returns
    -------
    dict
        ``number_of_replicates``, the five posterior predictive p-values (``mean_p_value``
        etc.), and ``has_misfit`` (1 when any p-value is extreme, else 0).
    """
    _model, _datasets = _extended_model_spec(data, distribution, "univariate_distribution")
    construct = {
        "model": _model,
        "sampler": str(sampler),
        "iterations": int(iterations),
        "output_length": int(output_length),
        "seed": int(seed),
        "number_of_replicates": int(number_of_replicates),
        "thinning_interval": int(thinning_interval),
    }
    return _extended("PosteriorPredictiveCheck", construct, _datasets)
