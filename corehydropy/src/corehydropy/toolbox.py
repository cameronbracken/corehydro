"""The Numerics toolbox surface. Every verb serializes its options to the
``toolbox_runner.hpp`` grammar and runs one method through ``_core.toolbox_run``; bulk data goes
across as numeric vectors, not JSON. Mirrors ``corehydror``'s ``R/toolbox.R`` verb for verb.
"""

from __future__ import annotations

import json
from importlib.resources import files

import numpy as np

from . import _core

__all__ = [
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
    "histogram",
    "interpolate",
    "interpolate_2d",
    "LinearRegressionResult",
    "linear_regression",
    "sobol_sequence",
    "stratify",
    "joint_probability",
]


def _toolbox_run(group: str, method: str, data=None, options=None) -> dict:
    """Internal: one call into the shared runner."""
    vectors = [np.asarray(d, dtype=float).ravel().tolist() for d in (data or [])]
    return _core.toolbox_run(group, method, vectors, json.dumps(options or {}))


def _check_pair(x, y, x_name: str = "x", y_name: str = "y"):
    """Internal: reject the two mistakes every paired-series verb can make, naming the argument."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size:
        raise ValueError(
            f"`{x_name}` and `{y_name}` must have the same length; got {xa.size} and {ya.size}"
        )
    if xa.size < 2:
        raise ValueError(f"`{x_name}` and `{y_name}` must have at least two elements")
    return xa, ya


def correlation(x, y, method: str = "pearson") -> float:
    """Correlation between two samples.

    Mirrors the C# ``Correlation`` class of the Numerics library. Upstream's matrix overloads
    are not ported, so only the paired-vector forms are available here.

    Parameters
    ----------
    x, y : array_like
        Numeric vectors of equal length, at least two elements.
    method : {"pearson", "spearman", "kendall"}
        Which coefficient to compute.

    Returns
    -------
    float

    Examples
    --------
    >>> from corehydropy import correlation
    >>> round(correlation([14, 8, 32, 7, 3, 15], [10, 5, 7, 4, 3, 8]), 6)
    0.545027
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"`method` must be one of 'pearson', 'spearman', 'kendall'; got {method!r}")
    xa, ya = _check_pair(x, y)
    return float(_toolbox_run("correlation", method, [xa, ya])["values"][0])


# The "statistics" and "spectra" toolbox groups (Task 3). "statistics" is descriptive/streaming
# summary statistics (mirroring C# Statistics/RunningStatistics/RunningCovarianceMatrix);
# "spectra" is the FFT and autocorrelation surface (mirroring C# Fourier and the newly-ported
# Autocorrelation). Mirrors corehydror's R/toolbox.R verb for verb.


def summary_statistics(x) -> dict:
    """Summary statistics for a sample.

    A one-shot description of ``x``: count, extremes, mean, variance, and the standard shape
    statistics, computed via the C# ``RunningStatistics`` class (Welford's algorithm) with no
    prior state. For a chunked, resumable version see :func:`running_statistics`.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    dict
        Keys ``n``, ``minimum``, ``maximum``, ``mean``, ``variance``, ``sd``, ``cv``,
        ``skewness``, ``kurtosis``, and the four raw moments ``m1`` to ``m4``.

    Examples
    --------
    >>> from corehydropy import summary_statistics
    >>> round(summary_statistics([2, 4, 4, 4, 5, 5, 7, 9])["mean"], 6)
    5.0
    """
    xa = np.asarray(x, dtype=float).ravel()
    r = _toolbox_run("statistics", "summary", [xa])
    return dict(zip(r["names"], r["values"]))


def product_moments(x) -> dict:
    """Product moments of a sample.

    Mirrors the C# ``Statistics.ProductMoments``: the mean, sample standard deviation, and the
    bias-corrected skewness and excess kurtosis. Requires at least 4 observations (returns
    ``nan`` otherwise, matching the C# behavior).

    Parameters
    ----------
    x : array_like

    Returns
    -------
    dict
        Keys ``mean``, ``sd``, ``skewness``, ``kurtosis``.

    Examples
    --------
    >>> from corehydropy import product_moments
    >>> round(product_moments([2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9])["mean"], 4)
    3.4
    """
    xa = np.asarray(x, dtype=float).ravel()
    r = _toolbox_run("statistics", "product_moments", [xa])
    return dict(zip(r["names"], r["values"]))


def l_moments(x) -> dict:
    """L-moments of a sample.

    Mirrors the C# ``Statistics.LinearMoments``. Requires at least 4 observations (returns
    ``nan`` otherwise, matching the C# behavior).

    Parameters
    ----------
    x : array_like

    Returns
    -------
    dict
        Keys ``l1`` (L-mean), ``l2`` (L-scale), ``t3`` (L-skewness), ``t4`` (L-kurtosis).

    Examples
    --------
    >>> from corehydropy import l_moments
    >>> round(l_moments([2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9])["l1"], 4)
    3.4
    """
    xa = np.asarray(x, dtype=float).ravel()
    r = _toolbox_run("statistics", "l_moments", [xa])
    return dict(zip(r["names"], r["values"]))


def ranks(x) -> np.ndarray:
    """Ranks of a sample.

    Mirrors the C# ``Statistics.RanksInPlace(double[])``: tied values (exact equality) receive
    the average rank of their run.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    numpy.ndarray
        Same length and order as ``x``.

    Examples
    --------
    >>> from corehydropy import ranks
    >>> ranks([3, 1, 2, 1])
    array([4. , 1.5, 3. , 1.5])
    """
    xa = np.asarray(x, dtype=float).ravel()
    return np.asarray(_toolbox_run("statistics", "ranks", [xa])["values"])


def percentile(x, probs, sorted: bool = False) -> np.ndarray:
    """Percentiles of a sample.

    Mirrors the C# ``Statistics.Percentile``: zero-based linear interpolation (R ``quantile()``
    Type 7).

    Parameters
    ----------
    x : array_like
    probs : array_like
        Probabilities in ``[0, 1]``.
    sorted : bool
        Set ``True`` when ``x`` is already sorted ascending, to skip re-sorting it.

    Returns
    -------
    numpy.ndarray
        One percentile per entry of ``probs``.

    Examples
    --------
    >>> from corehydropy import percentile
    >>> percentile([1, 2, 3, 4, 5], probs=[0.25, 0.5, 0.75])
    array([2., 3., 4.])
    """
    xa = np.asarray(x, dtype=float).ravel()
    probs_raw = np.asarray(probs)
    if probs_raw.dtype.kind not in "iuf":
        raise ValueError("`probs` must be numeric")
    pa = probs_raw.astype(float).ravel()
    r = _toolbox_run("statistics", "percentile", [xa, pa], {"sorted": bool(sorted)})
    return np.asarray(r["values"])


class RunningStatistics:
    """Streaming summary statistics accumulator.

    Mirrors the C# ``RunningStatistics`` class (Welford's algorithm). Carries ``n``,
    ``minimum``, ``maximum``, ``mean``, ``variance``, ``sd``, ``cv``, ``skewness``,
    ``kurtosis``, and the four raw moments ``m1`` to ``m4`` as plain attributes -- no C++ state,
    so an instance serializes with :mod:`pickle`.
    """

    __slots__ = ("n", "minimum", "maximum", "mean", "variance", "sd", "cv", "skewness",
                "kurtosis", "m1", "m2", "m3", "m4")

    def __init__(self, **fields) -> None:
        for name in self.__slots__:
            setattr(self, name, fields[name])

    def __repr__(self) -> str:
        return f"RunningStatistics(n={self.n:.0f}, mean={self.mean:.6g}, sd={self.sd:.6g})"


def running_statistics(x, state: RunningStatistics | None = None) -> RunningStatistics:
    """Streaming summary statistics.

    Accumulates count, extremes, and the first four moments over one or more chunks of data,
    mirroring the C# ``RunningStatistics`` class. The accumulator state travels in the return
    value, so a chunked run holds no C++ state.

    Parameters
    ----------
    x : array_like
        The next chunk of data.
    state : RunningStatistics, optional
        A :class:`RunningStatistics` from a previous call, or ``None`` (the default) to start a
        fresh accumulator.

    Returns
    -------
    RunningStatistics

    Examples
    --------
    >>> from corehydropy import running_statistics
    >>> s = running_statistics([1, 2, 3])
    >>> s = running_statistics([4, 5, 6], state=s)
    >>> s.n
    6.0
    >>> s.mean
    3.5
    """
    if state is not None and not isinstance(state, RunningStatistics):
        raise TypeError("`state` must be a RunningStatistics from a previous call")
    xa = np.asarray(x, dtype=float).ravel()
    options = {} if state is None else {
        "state": {"n": state.n, "m1": state.m1, "m2": state.m2, "m3": state.m3, "m4": state.m4,
                  "minimum": state.minimum, "maximum": state.maximum}
    }
    method = "summary" if state is None else "running"
    r = _toolbox_run("statistics", method, [xa], options)
    return RunningStatistics(**dict(zip(r["names"], r["values"])))


class RunningCovariance:
    """Streaming covariance and correlation matrix accumulator.

    Mirrors the C# ``RunningCovarianceMatrix`` class. Carries ``n``, ``mean`` (length ``size``),
    and the ``covariance``, ``sample_covariance``, ``sample_correlation``,
    ``population_covariance``, and ``population_correlation`` matrices (each ``size`` by
    ``size``) as plain attributes. ``covariance`` is unadjusted by sample size;
    ``sample_*``/``population_*`` are the N-1- and N-normalized variants. The C# accumulator
    seeds ``covariance`` at the identity matrix before the first push (a stability prior for its
    other consumer, adaptive MCMC), so every derived matrix carries a small diagonal-only bias
    that only fades as ``n`` grows -- do not expect an exact match to ``numpy.cov``/
    ``numpy.corrcoef`` on the same data at small ``n``.
    """

    __slots__ = ("n", "mean", "covariance", "sample_covariance", "sample_correlation",
                "population_covariance", "population_correlation")

    def __init__(self, **fields) -> None:
        for name in self.__slots__:
            setattr(self, name, fields[name])

    def __repr__(self) -> str:
        return f"RunningCovariance(n={self.n:.0f}, {len(self.mean)} variable(s))"


def running_covariance(x, state: RunningCovariance | None = None) -> RunningCovariance:
    """Streaming covariance and correlation matrix.

    Accumulates a running mean vector and covariance matrix over one or more chunks of
    multivariate data, mirroring the C# ``RunningCovarianceMatrix`` class. The accumulator state
    travels in the return value, so a chunked run holds no C++ state.

    Parameters
    ----------
    x : array_like
        A 2D array, observations in rows and variables in columns.
    state : RunningCovariance, optional
        A :class:`RunningCovariance` from a previous call, or ``None`` (the default) to start a
        fresh accumulator.

    Returns
    -------
    RunningCovariance

    Examples
    --------
    >>> from corehydropy import running_covariance
    >>> x = [[1, 2], [2, 4], [3, 5], [4, 4], [5, 5]]
    >>> running_covariance(x).n
    5.0
    """
    if state is not None and not isinstance(state, RunningCovariance):
        raise TypeError("`state` must be a RunningCovariance from a previous call")
    xa = np.asarray(x, dtype=float)
    if xa.ndim != 2:
        raise ValueError("`x` must be a 2D array (observations in rows, variables in columns)")
    size = xa.shape[1]
    columns = [xa[:, j] for j in range(size)]
    options = {} if state is None else {
        "state": {"n": state.n, "mean": list(map(float, state.mean)),
                  "covariance": [float(v) for row in state.covariance for v in row]}
    }
    r = _toolbox_run("statistics", "running_covariance", columns, options)
    values = r["values"]
    block = size * size

    def unflatten(flat):
        return np.asarray(flat, dtype=float).reshape(size, size)

    offset = 1
    mean = np.asarray(values[offset : offset + size], dtype=float)
    offset += size
    blocks = {}
    for name in ("covariance", "sample_covariance", "sample_correlation",
                "population_covariance", "population_correlation"):
        blocks[name] = unflatten(values[offset : offset + block])
        offset += block
    return RunningCovariance(n=values[0], mean=mean, **blocks)


def autocorrelation(x, max_lag=None, type: str = "correlation", confidence_level: float = 0.95) -> dict:
    """Autocorrelation, autocovariance, or partial autocorrelation function.

    Mirrors the C# ``Autocorrelation`` class. Computes the requested function out to
    ``max_lag`` (default ``floor(min(10*log10(N), N-1))``, matching the C# default) and attaches
    the asymptotic confidence band for ACF/PACF rho values at ``confidence_level``.

    Parameters
    ----------
    x : array_like
        At least two elements.
    max_lag : int, optional
        Maximum lag. ``None`` (the default) uses the C# default rule.
    type : {"correlation", "covariance", "partial"}
    confidence_level : float

    Returns
    -------
    dict
        Keys ``lag``, ``value`` (both :class:`numpy.ndarray`), ``type``, and ``ci`` (a dict
        with ``lower``/``upper``).

    Examples
    --------
    >>> from corehydropy import autocorrelation
    >>> x = [5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 5, 6, 4, 7, 3, 8, 2, 9, 1, 10]
    >>> round(float(autocorrelation(x, max_lag=5)["value"][0]), 12)
    1.0
    """
    if type not in ("correlation", "covariance", "partial"):
        raise ValueError(f"`type` must be one of 'correlation', 'covariance', 'partial'; got {type!r}")
    xa = np.asarray(x, dtype=float).ravel()
    if xa.size < 2:
        raise ValueError("`x` must have at least two elements")
    options = {"type": type}
    if max_lag is not None:
        options["lag_max"] = int(max_lag)
    r = _toolbox_run("spectra", "autocorrelation", [xa], options)
    n = r["dims"][0]
    values = np.asarray(r["values"], dtype=float)
    ci = _toolbox_run(
        "spectra", "autocorrelation_ci", [],
        {"sample_size": int(xa.size), "confidence_level": float(confidence_level)},
    )
    return {
        "lag": values[0::2][:n],
        "value": values[1::2][:n],
        "type": type,
        "ci": dict(zip(ci["names"], ci["values"])),
    }


def cross_correlation(x, y) -> np.ndarray:
    """Cross-correlation of two series.

    Mirrors the C# ``Fourier.Correlation``: an FFT-based correlation, NOT a normalized
    correlation coefficient (see :func:`correlation` for that). Both series must have the same
    power-of-two length.

    Parameters
    ----------
    x, y : array_like
        Equal, power-of-two length.

    Returns
    -------
    numpy.ndarray
        Wraparound order: increasing positive lags in ``[0]`` up to ``[n/2 - 1]``, increasing
        negative lags in ``[n-1]`` down to ``[n/2]``.

    Examples
    --------
    >>> from corehydropy import cross_correlation
    >>> len(cross_correlation([1, 2, 3, 4], [4, 3, 2, 1]))
    4
    """
    xa, ya = _check_pair(x, y)
    return np.asarray(_toolbox_run("spectra", "cross_correlation", [xa, ya])["values"])


def dft(x, inverse: bool = False) -> np.ndarray:
    """Discrete Fourier transform.

    Mirrors the C# ``Fourier.FFT``: an in-place complex FFT on data packed as
    ``[re0, im0, re1, im1, ...]``; ``len(x) / 2`` must be a power of two.

    Parameters
    ----------
    x : array_like
        Complex data packed as ``[re0, im0, re1, im1, ...]``.
    inverse : bool
        If ``True``, computes ``n`` times the inverse transform (matching the C# doc comment --
        divide by ``n`` for the true inverse).

    Returns
    -------
    numpy.ndarray
        Same length as ``x``.

    Examples
    --------
    >>> from corehydropy import dft
    >>> len(dft([1, 0, 2, 0, 3, 0, 4, 0]))
    8
    """
    xa = np.asarray(x, dtype=float).ravel()
    return np.asarray(_toolbox_run("spectra", "dft", [xa], {"inverse": bool(inverse)})["values"])


def dft_real(x, inverse: bool = False) -> np.ndarray:
    """Discrete Fourier transform of real-valued data.

    Mirrors the C# ``Fourier.RealFFT``: transforms ``len(x)`` real values in place (``len(x)``
    must be a power of two) into the positive-frequency half of the complex transform.

    Parameters
    ----------
    x : array_like
        Real-valued, power-of-two length.
    inverse : bool
        If ``True``, computes the inverse transform of a complex array that is the transform of
        real data (matching the C# doc comment -- multiply by ``2/len(x)`` for the true
        inverse).

    Returns
    -------
    numpy.ndarray
        Same length as ``x``.

    Examples
    --------
    >>> from corehydropy import dft_real
    >>> len(dft_real([1, 2, 3, 4]))
    4
    """
    xa = np.asarray(x, dtype=float).ravel()
    return np.asarray(_toolbox_run("spectra", "dft_real", [xa], {"inverse": bool(inverse)})["values"])


# The "histogram" and "interpolation" toolbox groups (Task 4). "histogram" mirrors the C#
# Histogram class (Rice-rule or explicit bin count, both deriving their range from the data --
# there is no lower/upper-bound constructor overload to expose); "interpolation" mirrors Linear
# and Bilinear, including their independent x/y transforms and Linear's separate extrapolate path.
# Mirrors corehydror's R/toolbox.R verb for verb.


def histogram(x, bins=None) -> dict:
    """Bin a sample into a histogram.

    Mirrors the C# ``Histogram`` class of the Numerics library. With ``bins=None`` the bin
    count follows the Rice rule, ``ceil(2 * n**(1/3)) + 1``, exactly as the C# data constructor
    does.

    Parameters
    ----------
    x : array_like
        Observations, at least one element.
    bins : int, optional
        Number of bins, must be positive. ``None`` (the default) uses the Rice rule.

    Returns
    -------
    dict
        Keys ``lower``, ``upper``, ``midpoint``, ``frequency`` (each :class:`numpy.ndarray`,
        one entry per bin) and ``statistics`` (a dict with ``mean``, ``median``, ``mode``,
        ``sd``, ``lower``, ``upper``, ``bin_width``, ``bins``).

    Examples
    --------
    >>> from corehydropy import histogram
    >>> h = histogram([1, 2, 2.5, 3, 3.5, 4, 5, 7, 8, 9])
    >>> float(h["frequency"].sum())
    10.0
    >>> h["statistics"]["bins"]
    6.0
    """
    xa = np.asarray(x, dtype=float).ravel()
    if xa.size == 0:
        raise ValueError("`x` must be a non-empty numeric vector")
    options = {} if bins is None else {"bins": int(bins)}
    b = _toolbox_run("histogram", "bins", [xa], options)
    rows = np.asarray(b["values"], dtype=float).reshape(-1, 4)
    out = {name: rows[:, i] for i, name in enumerate(b["names"])}
    s = _toolbox_run("histogram", "statistics", [xa], options)
    out["statistics"] = dict(zip(s["names"], s["values"]))
    return out


def interpolate(x, y, xout, x_transform: str = "none", y_transform: str = "none",
                sort_order: str = "ascending", extrapolate: bool = False) -> np.ndarray:
    """Interpolate a paired series.

    Mirrors the C# ``Linear`` interpolater of the Numerics library, including its x and y
    transforms.

    Parameters
    ----------
    x, y : array_like
        Equal-length knots.
    xout : array_like
        Positions to interpolate at.
    x_transform, y_transform : {"none", "log", "normal_z"}
    sort_order : {"ascending", "descending"}
        Describes ``x``.
    extrapolate : bool
        Whether to extend the end segments beyond the knots. ``False`` (the default) clamps to
        the end knot, matching the C# ``Interpolate()`` default; ``True`` calls the C#
        ``Extrapolate()`` method instead.

    Returns
    -------
    numpy.ndarray
        Same length as ``xout``.

    Examples
    --------
    >>> from corehydropy import interpolate
    >>> interpolate([1, 2, 3, 4], [10, 20, 30, 40], [1.5, 2.5])
    array([15., 25.])
    """
    xa, ya = _check_pair(x, y)
    if x_transform not in ("none", "log", "normal_z"):
        raise ValueError(
            f"`x_transform` must be one of 'none', 'log', 'normal_z'; got {x_transform!r}"
        )
    if y_transform not in ("none", "log", "normal_z"):
        raise ValueError(
            f"`y_transform` must be one of 'none', 'log', 'normal_z'; got {y_transform!r}"
        )
    if sort_order not in ("ascending", "descending"):
        raise ValueError(f"`sort_order` must be one of 'ascending', 'descending'; got {sort_order!r}")
    xouta = np.asarray(xout, dtype=float).ravel()
    options = {
        "x_transform": x_transform,
        "y_transform": y_transform,
        "sort_order": sort_order,
        "extrapolate": bool(extrapolate),
    }
    return np.asarray(_toolbox_run("interpolation", "linear", [xa, ya, xouta], options)["values"])


def interpolate_2d(x1, x2, y, x1out, x2out, x1_transform: str = "none", x2_transform: str = "none",
                   y_transform: str = "none", sort_order: str = "ascending") -> np.ndarray:
    """Interpolate a 2D grid (bilinear interpolation).

    Mirrors the C# ``Bilinear`` interpolater of the Numerics library.

    Parameters
    ----------
    x1, x2 : array_like
        Grid coordinates.
    y : array_like
        2D array, ``len(x1)`` rows by ``len(x2)`` columns; ``y[i, j]`` is the value at
        ``(x1[i], x2[j])``.
    x1out, x2out : array_like
        Equal-length positions to interpolate at.
    x1_transform, x2_transform, y_transform : {"none", "log", "normal_z"}
    sort_order : {"ascending", "descending"}
        Describes ``x1`` and ``x2``.

    Returns
    -------
    numpy.ndarray
        Same length as ``x1out``/``x2out``.

    Examples
    --------
    >>> from corehydropy import interpolate_2d
    >>> import numpy as np
    >>> interpolate_2d([1, 2, 3], [1, 2, 3], np.eye(3), [1.5], [1.5])
    array([0.5])
    """
    x1a = np.asarray(x1, dtype=float).ravel()
    x2a = np.asarray(x2, dtype=float).ravel()
    ya = np.asarray(y, dtype=float)
    if ya.shape != (x1a.size, x2a.size):
        raise ValueError(
            f"`y` must be a {x1a.size} x {x2a.size} array (len(x1) x len(x2)); got {ya.shape}"
        )
    x1outa = np.asarray(x1out, dtype=float).ravel()
    x2outa = np.asarray(x2out, dtype=float).ravel()
    if x1outa.size != x2outa.size:
        raise ValueError("`x1out` and `x2out` must be the same length")
    for name, t in (("x1_transform", x1_transform), ("x2_transform", x2_transform),
                    ("y_transform", y_transform)):
        if t not in ("none", "log", "normal_z"):
            raise ValueError(f"`{name}` must be one of 'none', 'log', 'normal_z'; got {t!r}")
    if sort_order not in ("ascending", "descending"):
        raise ValueError(f"`sort_order` must be one of 'ascending', 'descending'; got {sort_order!r}")
    options = {
        "x1_transform": x1_transform,
        "x2_transform": x2_transform,
        "y_transform": y_transform,
        "sort_order": sort_order,
    }
    data = [x1a, x2a, ya.ravel(), x1outa, x2outa]
    return np.asarray(_toolbox_run("interpolation", "bilinear", data, options)["values"])


# The "regression" toolbox group (Task 5): ordinary least squares by singular value
# decomposition, mirroring the C# `LinearRegression` class of the Numerics library. Mirrors
# corehydror's R/regression.R -- its own section rather than folded into the plain-value verbs
# above, since it is the first toolbox group whose result is a full object (with its own
# `predict` method) rather than a scalar, array, or dict.


class LinearRegressionResult:
    """Fitted ordinary least squares model, mirroring the C# ``LinearRegression`` class.

    Carries no C++ state: ``predict`` reruns the fit against the shared toolbox runner each
    call, so an instance serializes with :mod:`pickle`.

    ``covariance`` is the coefficient covariance matrix, i.e. ``numpy.sqrt(numpy.diag(
    result.covariance))`` equals ``result.standard_errors``. The underlying C#
    ``LinearRegression.Covariance`` is the unscaled cross-product term (``(X'X)^-1``), scaled
    here by ``sigma**2`` to match ``standard_errors``.
    """

    __slots__ = (
        "coefficients", "standard_errors", "covariance", "residuals", "r_squared",
        "adj_r_squared", "sigma", "df", "n", "_x", "_y", "_intercept",
    )

    def __init__(self, **fields) -> None:
        for name in self.__slots__:
            setattr(self, name, fields[name])

    def __repr__(self) -> str:
        return (
            f"LinearRegressionResult(n={self.n}, p={len(self.coefficients)}, "
            f"r_squared={self.r_squared:.6g})"
        )

    def predict(self, newdata, interval: bool = False, level: float = 0.90) -> np.ndarray:
        """Predict from the fitted model.

        Mirrors the C# ``LinearRegression.Predict``/``PredictionIntervals`` methods.

        Parameters
        ----------
        newdata : array_like
            A 2D array of predictors with the same number of columns the model was fitted
            with (an intercept column, if any, is added internally), or -- for a
            single-predictor model only -- a 1D array of observations (matching R's
            ``predict.corehydro_lm()``, where a bare vector is unambiguous when there is one
            predictor). For more than one predictor a 1D array is rejected rather than guessed
            at, since it is ambiguous whether it means one row or one column.
        interval : bool
            If ``True``, also return the ``level`` prediction interval (a Student-t interval
            around the mean response, mirroring ``PredictionIntervals``). Default ``False``.
        level : float
            Prediction interval level, between 0 and 1. Default ``0.90``, matching
            ``PredictionIntervals``'s own ``alpha=0.1`` default.

        Returns
        -------
        numpy.ndarray
            With ``interval=False``, one predicted value per row of ``newdata``. With
            ``interval=True``, an ``(n, 3)`` array with columns ``lower``, ``upper``, ``mean``.
        """
        ncol = self._x.shape[1]
        raw = np.asarray(newdata, dtype=float)
        if raw.ndim == 1 and ncol == 1:
            # Unambiguous only for a single-predictor model: one value == one row, matching
            # R's predict.corehydro_lm() (see regression.R). For ncol > 1, fall through to
            # atleast_2d below, which turns [1, 2, 3] into a single row of 3 columns and is
            # rejected by the shape check that follows -- an ambiguous bare vector must be an
            # explicit 2D array instead.
            nd = raw.reshape(-1, 1)
        else:
            nd = np.atleast_2d(raw)
        if nd.shape[1] != ncol:
            raise ValueError(
                f"`newdata` must have {ncol} column(s), matching the fitted predictors; "
                f"got {nd.shape[1]}"
            )
        options = {
            "rows": int(self._x.shape[0]), "columns": int(ncol), "intercept": bool(self._intercept),
            "predict_rows": int(nd.shape[0]),
        }
        data = [self._x.ravel(), self._y, nd.ravel()]
        if not interval:
            return np.asarray(_toolbox_run("regression", "predict", data, options)["values"])
        if not (0.0 < level < 1.0):
            raise ValueError(f"`level` must be between 0 and 1; got {level}")
        options["alpha"] = 1.0 - level
        r = _toolbox_run("regression", "prediction_intervals", data, options)
        return np.asarray(r["values"], dtype=float).reshape(r["dims"][0], r["dims"][1])


def linear_regression(x, y, intercept: bool = True) -> LinearRegressionResult:
    """Ordinary least squares by singular value decomposition.

    Mirrors the C# ``LinearRegression`` class of the Numerics library: estimates
    ``Y = alpha + beta*X + e``, ``e ~ N(0, sigma)``, via SVD.

    Parameters
    ----------
    x : array_like
        A 2D array of predictors with one row per observation, or a 1D array for a single
        predictor.
    y : array_like
        Responses, one per row of ``x``.
    intercept : bool
        Whether to fit an intercept. Default ``True``.

    Returns
    -------
    LinearRegressionResult

    Examples
    --------
    >>> from corehydropy import linear_regression
    >>> x = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]]
    >>> y = [3.1, 4.2, 8.1, 9.2, 13.0]
    >>> fit = linear_regression(x, y)
    >>> round(float(fit.r_squared), 6)
    0.997992
    """
    xa = np.asarray(x, dtype=float)
    if xa.ndim == 1:
        xa = xa.reshape(-1, 1)
    ya = np.asarray(y, dtype=float).ravel()
    if ya.size != xa.shape[0]:
        raise ValueError(
            f"`y` must have one value per row of `x`; got {ya.size} and {xa.shape[0]}"
        )
    options = {"rows": int(xa.shape[0]), "columns": int(xa.shape[1]), "intercept": bool(intercept)}
    data = [xa.ravel(), ya]
    v = _toolbox_run("regression", "fit", data, options)
    named = dict(zip(v["names"], v["values"]))
    p = xa.shape[1] + (1 if intercept else 0)
    coefficients = np.array([named[f"beta_{i + 1}"] for i in range(p)])
    standard_errors = np.array([named[f"se_{i + 1}"] for i in range(p)])

    # The runner's "covariance" method returns the raw C# LinearRegression.Covariance quantity,
    # which is UNSCALED (see linear_regression.hpp) -- the fit multiplies by sigma**2 (standard
    # error squared) to get standard_errors, so the same scaling is applied here to keep
    # `result.covariance` consistent with `result.standard_errors`:
    # sqrt(diag(result.covariance)) == result.standard_errors.
    cov = _toolbox_run("regression", "covariance", data, options)
    covariance = np.asarray(cov["values"], dtype=float).reshape(cov["dims"][0], cov["dims"][1])
    covariance = covariance * named["sigma"] ** 2

    res = _toolbox_run("regression", "residuals", data, options)

    return LinearRegressionResult(
        coefficients=coefficients,
        standard_errors=standard_errors,
        covariance=covariance,
        residuals=np.asarray(res["values"], dtype=float),
        r_squared=named["r_squared"],
        adj_r_squared=named["adj_r_squared"],
        sigma=named["sigma"],
        df=int(named["df"]),
        n=int(named["n"]),
        _x=xa,
        _y=ya,
        _intercept=bool(intercept),
    )


# The "sampling" and "probability" toolbox groups (Task 6). "sampling" is the C# SobolSequence
# quasi-random low-discrepancy sequence and Stratify::XValues equal-width axis stratification;
# "probability" is Probability's joint-probability dispatch (plain dependency form and the
# indicator + correlation-matrix HPCM form). Mirrors corehydror's R/toolbox.R verb for verb.


def sobol_sequence(n: int, dimension: int = 1, skip: int = 0) -> np.ndarray:
    """Sobol quasi-random low-discrepancy sequence.

    Mirrors the C# ``SobolSequence`` class. ``dimension > 1`` needs the new-joe-kuo-6 direction
    numbers shipped with the package (``dimension == 1`` needs no file, matching the C# ctor);
    the packaged file is located automatically.

    Parameters
    ----------
    n : int
        Number of points to generate, at least 1.
    dimension : int
        Spatial dimension, between 1 and 21201. Default 1.
    skip : int
        Number of points to skip before the first returned point: ``skip=k`` returns the same
        first point as the C# ``SkipTo(k)`` call, i.e. the sequence's ``(k + 1)``-th point.
        Default 0 (no skip).

    Returns
    -------
    numpy.ndarray
        An ``(n, dimension)`` array, every value in ``[0, 1)``.

    Examples
    --------
    >>> from corehydropy import sobol_sequence
    >>> sobol_sequence(1, dimension=2)
    array([[0.5, 0.5]])
    """
    if n < 1:
        raise ValueError("`n` must be a positive integer")
    if dimension < 1:
        raise ValueError("`dimension` must be a positive integer")
    path = ""
    if dimension > 1:
        path = str(files("corehydropy") / "data" / "new-joe-kuo-6.21201")
    options = {"dimension": int(dimension), "n": int(n), "skip": int(skip), "path": path}
    r = _toolbox_run("sampling", "sobol", [], options)
    return np.asarray(r["values"], dtype=float).reshape(r["dims"][0], r["dims"][1])


def stratify(lower: float, upper: float, bins: int, logarithmic: bool = False,
            probability: bool = False) -> dict:
    """Stratify an axis into equal-width bins.

    Mirrors the C# ``Stratify.XValues(StratificationOptions, isLogarithmic)``: splits
    ``[lower, upper]`` into ``bins`` equal-width strata (equal-width in log10 space when
    ``logarithmic=True``), each carrying a weight defaulting to its own width.
    ``probability=True`` always returns zero rows, matching ``Stratify.XValues``'s own early
    return for probability-space options -- the ported header exposes only the
    ``StratificationOptions`` overload used by the BestFit estimators' profile-likelihood grids,
    not the probability-stratification methods (``Probabilities``, ``XToProbability``, ...),
    which are out of scope (see ``stratify.hpp``'s file header).

    Parameters
    ----------
    lower, upper : float
        Bounds of the axis to stratify. ``lower`` must be less than ``upper``.
    bins : int
        Number of bins, greater than 1.
    logarithmic : bool
        Stratify on a log10 scale. Default ``False``.
    probability : bool
        Mark the axis as a probability axis; kept only for parity with the C# constructor
        argument -- it always yields zero bins (see above). Default ``False``.

    Returns
    -------
    dict
        Keys ``lower``, ``upper``, ``midpoint``, ``weight`` (each :class:`numpy.ndarray`, one
        entry per bin).

    Examples
    --------
    >>> from corehydropy import stratify
    >>> s = stratify(0, 1, bins=4)
    >>> s["midpoint"]
    array([0.125, 0.375, 0.625, 0.875])
    """
    if bins < 2:
        raise ValueError("`bins` must be an integer greater than 1")
    options = {"lower": float(lower), "upper": float(upper), "bins": int(bins),
              "logarithmic": bool(logarithmic), "probability": bool(probability)}
    r = _toolbox_run("sampling", "stratify", [], options)
    rows = np.asarray(r["values"], dtype=float).reshape(-1, 4)
    return {name: rows[:, i] for i, name in enumerate(r["names"])}


def joint_probability(p, dependency: str = "independent", indicators=None, correlation=None) -> float:
    """Joint probability of multiple events.

    Mirrors the C# ``Probability.JointProbability``. The plain form dispatches on a dependency
    assumption alone (``"independent"`` multiplies, ``"positive"`` takes the minimum,
    ``"negative"`` clamps the excess of the sum over 1). Passing ``indicators`` (a 0/1 flag per
    component, selecting which components participate) switches to the indicator-aware form;
    passing ``correlation`` too routes ``dependency="correlation"`` through Haden Smith's
    modification of Pandey's Product-of-Conditional-Marginals method (HPCM).

    Parameters
    ----------
    p : array_like
        Marginal probabilities.
    dependency : {"independent", "positive", "negative", "correlation"}
        ``"correlation"`` requires both ``indicators`` and ``correlation``.
    indicators : array_like, optional
        0/1 vector, the same length as ``p``.
    correlation : array_like, optional
        ``len(p)`` by ``len(p)`` correlation matrix; requires ``indicators``.

    Returns
    -------
    float

    Examples
    --------
    >>> from corehydropy import joint_probability
    >>> joint_probability([0.5, 0.5])
    0.25
    >>> joint_probability([0.5, 0.5], dependency="positive")
    0.5
    """
    if dependency not in ("independent", "positive", "negative", "correlation"):
        raise ValueError(
            f"`dependency` must be one of 'independent', 'positive', 'negative', 'correlation'; "
            f"got {dependency!r}"
        )
    pa = np.asarray(p, dtype=float).ravel()
    if pa.size == 0:
        raise ValueError("`p` must be a non-empty array")
    if correlation is not None and indicators is None:
        raise ValueError("`correlation` requires `indicators`")
    data = [pa]
    if indicators is not None:
        ind = np.asarray(indicators, dtype=float).ravel()
        if ind.size != pa.size:
            raise ValueError("`indicators` must be the same length as `p`")
        data.append(ind)
        if correlation is not None:
            corr = np.asarray(correlation, dtype=float)
            n = pa.size
            if corr.shape != (n, n):
                raise ValueError(f"`correlation` must be a {n} x {n} matrix")
            data.append(corr.ravel())
    r = _toolbox_run("probability", "joint", data, {"dependency": dependency})
    return float(r["values"][0])
