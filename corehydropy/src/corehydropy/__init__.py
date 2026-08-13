"""corehydropy: Bayesian flood-frequency and extreme-value analysis.

Python bindings to a shared C++ port of the USACE-RMC Numerics / RMC.BestFit
libraries, validated value-by-value against Numerics 2.1.4 and RMC.BestFit 2.0.0.
"""

from __future__ import annotations

import numpy as np

from ._core import GeneralizedExtremeValue, gev_fit as _gev_fit
from .distributions import (
    Distribution,
    dist_competing_risks,
    dist_empirical,
    dist_kde,
    dist_mixture,
    dist_truncated,
    distribution_names,
)
from .toolbox import (
    RunningCovariance,
    RunningStatistics,
    autocorrelation,
    correlation,
    cross_correlation,
    dft,
    dft_real,
    l_moments,
    percentile,
    product_moments,
    ranks,
    running_covariance,
    running_statistics,
    summary_statistics,
)
from .gof import (
    aic,
    aic_weights,
    aicc,
    bic,
    classification_metrics,
    gof_rmse,
    gof_test,
    goodness_of_fit,
    rmse_weights,
)
from .copula import Copula, copula_fit, copula_names
from .mvdist import (
    MultivariateDistribution,
    mvdist_bivariate_empirical,
    mvdist_dirichlet,
    mvdist_multinomial,
    mvdist_names,
    mvdist_normal,
    mvdist_student_t,
)
from .data import (
    AnalysisData,
    analysis_data,
    analysis_data_summary,
    threshold_diagnostics,
)
from .models import (
    Model,
    ModelParameter,
    Trend,
    model_ar,
    model_arima,
    model_arimax,
    model_bivariate,
    model_bulletin17c,
    model_competing_risks,
    model_log_likelihood,
    model_ma,
    model_mixture,
    model_parameter,
    model_parameters,
    model_point_process,
    model_rating_curve,
    model_simulate,
    model_spatial_gev,
    model_univariate,
    model_validate,
    trend,
)
from .fit import (
    Fit,
    fit_bayesian,
    fit_diagnostics,
    fit_gmm,
    fit_map,
    fit_mle,
    quantile_variance,
)
from .mcmc import mcmc_sample
from .stats import (
    box_cox,
    box_cox_inverse,
    box_cox_lambda,
    latin_hypercube,
    mgbt_test,
    plotting_positions,
    yeo_johnson,
    yeo_johnson_inverse,
    yeo_johnson_lambda,
)
from .analysis import (
    ar_analysis,
    arima_analysis,
    arimax_analysis,
    bivariate_analysis,
    bootstrap_analysis,
    bulletin17c_analysis,
    coincident_frequency_analysis,
    competing_risk_analysis,
    composite_analysis,
    estimation_diagnostics,
    fit_distributions,
    ma_analysis,
    mixture_analysis,
    point_process_analysis,
    posterior_predictive_check,
    prior_predictive_check,
    rating_curve_analysis,
    spatial_gev_analysis,
    univariate_analysis,
)

__all__ = [
    "GeneralizedExtremeValue",
    "correlation",
    "summary_statistics",
    "product_moments",
    "l_moments",
    "ranks",
    "percentile",
    "RunningStatistics",
    "running_statistics",
    "RunningCovariance",
    "running_covariance",
    "autocorrelation",
    "cross_correlation",
    "dft",
    "dft_real",
    "goodness_of_fit",
    "classification_metrics",
    "gof_test",
    "gof_rmse",
    "aic",
    "aicc",
    "bic",
    "aic_weights",
    "rmse_weights",
    "Distribution",
    "distribution_names",
    "dist_truncated",
    "dist_mixture",
    "dist_competing_risks",
    "dist_empirical",
    "dist_kde",
    "Copula",
    "copula_fit",
    "copula_names",
    "MultivariateDistribution",
    "mvdist_normal",
    "mvdist_student_t",
    "mvdist_dirichlet",
    "mvdist_multinomial",
    "mvdist_bivariate_empirical",
    "mvdist_names",
    "dgev",
    "pgev",
    "qgev",
    "gev_moments",
    "gev_fit",
    "Fit",
    "fit_mle",
    "fit_map",
    "fit_bayesian",
    "fit_gmm",
    "fit_diagnostics",
    "quantile_variance",
    "univariate_analysis",
    "fit_distributions",
    "bulletin17c_analysis",
    "mixture_analysis",
    "competing_risk_analysis",
    "point_process_analysis",
    "ar_analysis",
    "ma_analysis",
    "arima_analysis",
    "arimax_analysis",
    "estimation_diagnostics",
    "composite_analysis",
    "spatial_gev_analysis",
    "bivariate_analysis",
    "coincident_frequency_analysis",
    "rating_curve_analysis",
    "bootstrap_analysis",
    "prior_predictive_check",
    "posterior_predictive_check",
    "mgbt_test",
    "box_cox_lambda",
    "box_cox",
    "box_cox_inverse",
    "yeo_johnson_lambda",
    "yeo_johnson",
    "yeo_johnson_inverse",
    "plotting_positions",
    "latin_hypercube",
    "mcmc_sample",
    "AnalysisData",
    "analysis_data",
    "analysis_data_summary",
    "threshold_diagnostics",
    "Model",
    "Trend",
    "ModelParameter",
    "trend",
    "model_parameter",
    "model_univariate",
    "model_mixture",
    "model_competing_risks",
    "model_point_process",
    "model_bulletin17c",
    "model_ar",
    "model_ma",
    "model_arima",
    "model_arimax",
    "model_spatial_gev",
    "model_rating_curve",
    "model_bivariate",
    "model_validate",
    "model_simulate",
    "model_log_likelihood",
    "model_parameters",
]


def dgev(x, location=0.0, scale=1.0, shape=0.0):
    """GEV probability density at ``x``.

    Parameters
    ----------
    x : float or array_like
        Evaluation points.
    location : float, default 0.0
        Location parameter.
    scale : float, default 1.0
        Scale parameter.
    shape : float, default 0.0
        Shape parameter.

    Returns
    -------
    float or numpy.ndarray
        Density at each point, matching the shape of ``x``.
    """
    g = GeneralizedExtremeValue(location, scale, shape)
    return _apply(g.pdf, x)


def pgev(q, location=0.0, scale=1.0, shape=0.0):
    """GEV cumulative distribution at ``q``.

    Parameters
    ----------
    q : float or array_like
        Evaluation points.
    location : float, default 0.0
        Location parameter.
    scale : float, default 1.0
        Scale parameter.
    shape : float, default 0.0
        Shape parameter.

    Returns
    -------
    float or numpy.ndarray
        Cumulative probability at each point, matching the shape of ``q``.
    """
    g = GeneralizedExtremeValue(location, scale, shape)
    return _apply(g.cdf, q)


def qgev(p, location=0.0, scale=1.0, shape=0.0):
    """GEV quantile (inverse CDF) at probability ``p``.

    Parameters
    ----------
    p : float or array_like
        Non-exceedance probabilities.
    location : float, default 0.0
        Location parameter.
    scale : float, default 1.0
        Scale parameter.
    shape : float, default 0.0
        Shape parameter.

    Returns
    -------
    float or numpy.ndarray
        Quantile at each probability, matching the shape of ``p``.
    """
    g = GeneralizedExtremeValue(location, scale, shape)
    return _apply(g.quantile, p)


def gev_moments(location, scale, shape):
    """GEV distribution moments and support.

    Parameters
    ----------
    location : float
        Location parameter.
    scale : float
        Scale parameter.
    shape : float
        Shape parameter.

    Returns
    -------
    dict
        Keys ``mean``, ``median``, ``sd``, ``skewness``, ``kurtosis``, ``minimum``,
        ``maximum``. Undefined moments are ``nan``.
    """
    g = GeneralizedExtremeValue(location, scale, shape)
    return {
        "mean": g.mean(),
        "median": g.median(),
        "sd": g.standard_deviation(),
        "skewness": g.skewness(),
        "kurtosis": g.kurtosis(),
        "minimum": g.minimum(),
        "maximum": g.maximum(),
    }


def gev_fit(x, method="mle"):
    """Fit a GEV distribution to sample ``x``.

    Parameters
    ----------
    x : array_like
        Sample observations.
    method : {"mle", "lmom", "mom"}, default "mle"
        Estimation method: maximum likelihood, L-moments, or method of moments.

    Returns
    -------
    dict
        Keys ``location``, ``scale``, ``shape``.
    """
    loc, scale, shape = _gev_fit([float(v) for v in np.asarray(x).ravel()], method)
    return {"location": loc, "scale": scale, "shape": shape}


def _apply(fn, values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return fn(float(arr))
    return np.array([fn(float(v)) for v in arr.ravel()]).reshape(arr.shape)
