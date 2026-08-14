"""The goodness-of-fit surface. Every verb runs one method of the ``"gof"`` group through the
shared toolbox runner. Mirrors ``corehydror``'s ``R/gof.R`` verb for verb; where R returns a named
vector this returns a dict.
"""

from __future__ import annotations

import json

import numpy as np

from .toolbox import _check_pair, _toolbox_run

__all__ = [
    "goodness_of_fit",
    "classification_metrics",
    "gof_test",
    "gof_rmse",
    "aic",
    "aicc",
    "bic",
    "aic_weights",
    "rmse_weights",
]

_GOF_METRICS = (
    "rmse", "mse", "mae", "mape", "smape", "nse", "log_nse", "kge", "kge_mod",
    "pbias", "rsr", "pearson", "r_squared", "d", "d_mod", "d_ref", "ve",
)


def goodness_of_fit(observed, modeled, metrics="all", k: int = 0) -> dict:
    """Goodness-of-fit metrics for a modeled series.

    Mirrors the continuous metrics of the C# ``GoodnessOfFit`` class of the Numerics library.

    Parameters
    ----------
    observed, modeled : array_like
        Numeric vectors of equal length.
    metrics : str or sequence of str
        ``"all"`` (the default) for every metric, or the metric names to keep, drawn from
        ``rmse``, ``mse``, ``mae``, ``mape``, ``smape``, ``nse``, ``log_nse``, ``kge``,
        ``kge_mod``, ``pbias``, ``rsr``, ``pearson``, ``r_squared``, ``d``, ``d_mod``, ``d_ref``,
        ``ve``. A bare string is treated as a single metric name, not a sequence of characters.
        ``"all"`` (the default) evaluates every metric eagerly, including ``mape``, which raises
        if ``observed`` contains a zero; ask for just the metrics you want to avoid that.
    k : int
        Degrees-of-freedom correction subtracted from the sample size in the RMSE denominator.

    Returns
    -------
    dict

    Examples
    --------
    >>> from corehydropy import goodness_of_fit
    >>> round(goodness_of_fit([2, 4, 6, 8, 10], [2.2, 3.9, 6.4, 7.5, 10.1])["nse"], 4)
    0.9882
    """
    obs, mod = _check_pair(observed, modeled, "observed", "modeled")
    if isinstance(metrics, str) and metrics != "all":
        metrics = [metrics]
    if metrics == "all":
        r = _toolbox_run("gof", "metrics", [obs, mod], {"k": int(k)})
        return dict(zip(r["names"], r["values"]))
    unknown = [m for m in metrics if m not in _GOF_METRICS]
    if unknown:
        raise ValueError(
            f"unknown metric(s): {', '.join(unknown)}. Available: {', '.join(_GOF_METRICS)}"
        )
    # Dispatched one metric at a time so a caller asking for e.g. "mse" never pays MAPE's
    # zero-observed-value precondition; only metrics = "all" evaluates the eager combined arm.
    return {
        m: float(_toolbox_run("gof", m, [obs, mod], {"k": int(k)})["values"][0]) for m in metrics
    }


def classification_metrics(observed, modeled) -> dict:
    """Classification metrics for two binary label vectors.

    Mirrors the C# ``GoodnessOfFit`` classification statics (``Accuracy``/``Precision``/
    ``Recall``/``F1Score``/``Specificity``/``BalancedAccuracy``): both ``observed`` and
    ``modeled`` are already-binary label vectors, compared elementwise -- a value equal to its
    counterpart counts as a match. There is no threshold argument, in Python or in C#; threshold
    your own series into 0/1 labels before calling this.

    Parameters
    ----------
    observed, modeled : array_like
        Numeric vectors of equal length holding binary (0/1) labels.

    Returns
    -------
    dict
        Keys ``accuracy``, ``precision``, ``recall``, ``f1``, ``specificity``,
        ``balanced_accuracy``.

    Examples
    --------
    >>> from corehydropy import classification_metrics
    >>> classification_metrics([1, 0, 1, 1, 0], [1, 0, 0, 1, 0])["accuracy"]
    80.0
    """
    obs, mod = _check_pair(observed, modeled, "observed", "modeled")
    r = _toolbox_run("gof", "classification", [obs, mod], {})
    return dict(zip(r["names"], r["values"]))


def gof_test(x, d, test: str = "ks") -> float:
    """Goodness-of-fit test statistic for a fitted distribution.

    Parameters
    ----------
    x : array_like
        Numeric vector of observations. Sorted internally, as the C# methods require.
    d : Distribution
        A fitted :class:`~corehydropy.Distribution`.
    test : {"ks", "ad", "chi_squared"}
        ``"ks"`` (Kolmogorov-Smirnov D), ``"ad"`` (Anderson-Darling A squared), or
        ``"chi_squared"``.

    Returns
    -------
    float

    Examples
    --------
    >>> from corehydropy import Distribution, gof_test
    >>> x = [2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9]
    >>> gof_test(x, Distribution("Normal", [3.4, 1.1]), "ks") > 0
    True
    """
    if test not in ("ks", "ad", "chi_squared"):
        raise ValueError(f"`test` must be one of 'ks', 'ad', 'chi_squared'; got {test!r}")
    if not hasattr(d, "to_json"):
        raise TypeError("`d` must be a Distribution; create one with Distribution()")
    options = {"model": json.loads(d.to_json())}
    return float(_toolbox_run("gof", test, [x], options)["values"][0])


def gof_rmse(x, d, plotting_positions=None) -> float:
    """Root mean squared error of a fitted distribution.

    Parameters
    ----------
    x : array_like
        Numeric vector of observations. Sorted internally, as the C# methods require.
    d : Distribution
        A fitted :class:`~corehydropy.Distribution`.
    plotting_positions : array_like, optional
        Exceedance probabilities. ``None`` (the default) uses Weibull positions, matching the C#
        overload.

    Returns
    -------
    float

    Examples
    --------
    >>> from corehydropy import Distribution, gof_rmse
    >>> x = [2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9]
    >>> gof_rmse(x, Distribution("Normal", [3.4, 1.1])) > 0
    True
    """
    if not hasattr(d, "to_json"):
        raise TypeError("`d` must be a Distribution; create one with Distribution()")
    data = [x] if plotting_positions is None else [x, plotting_positions]
    options = {"model": json.loads(d.to_json())}
    return float(_toolbox_run("gof", "rmse_dist", data, options)["values"][0])


def aic(k: int, log_likelihood: float) -> float:
    """Akaike Information Criterion.

    Parameters
    ----------
    k : int
        Number of estimated parameters.
    log_likelihood : float
        The maximized log-likelihood.

    Returns
    -------
    float

    Examples
    --------
    >>> from corehydropy import aic
    >>> aic(k=2, log_likelihood=-121.01131220612)
    246.02262441224
    """
    options = {"k": int(k), "log_likelihood": float(log_likelihood)}
    return float(_toolbox_run("gof", "aic", [], options)["values"][0])


def aicc(n: int, k: int, log_likelihood: float) -> float:
    """Small-sample-corrected Akaike Information Criterion.

    Parameters
    ----------
    n : int
        Sample size.
    k : int
        Number of estimated parameters.
    log_likelihood : float
        The maximized log-likelihood.

    Returns
    -------
    float

    Examples
    --------
    >>> from corehydropy import aicc
    >>> round(aicc(n=30, k=2, log_likelihood=-121.01131220612), 6)
    246.467069
    """
    return float(
        _toolbox_run(
            "gof", "aicc", [], {"n": int(n), "k": int(k), "log_likelihood": float(log_likelihood)}
        )["values"][0]
    )


def bic(n: int, k: int, log_likelihood: float) -> float:
    """Bayesian Information Criterion.

    Parameters
    ----------
    n : int
        Sample size.
    k : int
        Number of estimated parameters.
    log_likelihood : float
        The maximized log-likelihood.

    Returns
    -------
    float

    Examples
    --------
    >>> from corehydropy import bic
    >>> round(bic(n=30, k=2, log_likelihood=-121.01131220612), 6)
    248.825019
    """
    return float(
        _toolbox_run(
            "gof", "bic", [], {"n": int(n), "k": int(k), "log_likelihood": float(log_likelihood)}
        )["values"][0]
    )


def aic_weights(aic) -> np.ndarray:
    """Model weights from a vector of AIC values.

    Parameters
    ----------
    aic : array_like
        AIC values, one per candidate model.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> from corehydropy import aic_weights
    >>> round(float(aic_weights([246.0, 248.8, 251.2]).sum()), 6)
    1.0
    """
    return np.asarray(_toolbox_run("gof", "aic_weights", [aic])["values"])


def rmse_weights(rmse) -> np.ndarray:
    """Model weights from a vector of RMSE values.

    Parameters
    ----------
    rmse : array_like
        RMSE values, one per candidate model.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> from corehydropy import rmse_weights
    >>> round(float(rmse_weights([1.2, 1.5, 2.0]).sum()), 6)
    1.0
    """
    return np.asarray(_toolbox_run("gof", "rmse_weights", [rmse])["values"])
