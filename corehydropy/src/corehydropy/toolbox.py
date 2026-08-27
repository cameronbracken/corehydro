"""The Numerics toolbox surface. Every verb serializes its options to the
``toolbox_runner.hpp`` grammar and runs one method through ``_core.toolbox_run``; bulk data goes
across as numeric vectors, not JSON. Mirrors ``corehydror``'s ``R/toolbox.R`` verb for verb.
"""

from __future__ import annotations

import json
from importlib.resources import files

import numpy as np

from . import _core
from .distributions import Distribution, _as_spec

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
    "Link",
    "link_function",
    "link",
    "link_inverse",
    "link_derivative",
    "link_names",
    "trend_predict",
    "trend_parameters",
    "trend_names",
    "qr_decomposition",
    "qr_solve",
    "gauss_jordan",
    "debye",
    "polynomial_eval",
    "univariate_function",
    "shortest_path",
    "hypothesis_test",
    "curve_interpolate",
    "curve_area",
    "curve_simplify",
    "uncertain_curve_sample",
    "tabular_function",
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


def correlation(x, y=None, method: str = "pearson"):
    """Correlation between two samples, or a correlation matrix.

    Mirrors the C# ``Correlation`` class of the Numerics library: the paired-vector forms
    (``Correlation.Pearson``/``Spearman``/``KendallsTau``, both ``IList<double>`` overloads) and
    the Pearson/Spearman column-pairwise matrix overloads (``Pearson(double[,])``/
    ``Spearman(double[,])``).

    Parameters
    ----------
    x : array_like
        A numeric vector (paired-vector form, ``y`` required), or a 2D array-like (a NumPy
        array, nested list, or pandas-style object coercible via ``numpy.asarray``) with one
        column per variable (matrix form, ``y`` omitted).
    y : array_like, optional
        A numeric vector of the same length as ``x``, or ``None`` (the default) to compute the
        correlation matrix of ``x``'s columns instead.
    method : {"pearson", "spearman", "kendall"}
        Which coefficient to compute. ``"kendall"`` is rejected when ``y`` is ``None``: upstream
        has no ``KendallsTau(double[,])`` overload, so there is no Kendall matrix form to
        compute.

    Returns
    -------
    float or numpy.ndarray
        With ``y`` given, a single correlation coefficient; with ``y`` ``None``, a
        ``(p, p)`` array, ``p`` the number of columns of ``x``.

    Examples
    --------
    >>> from corehydropy import correlation
    >>> round(correlation([14, 8, 32, 7, 3, 15], [10, 5, 7, 4, 3, 8]), 6)
    0.545027
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"`method` must be one of 'pearson', 'spearman', 'kendall'; got {method!r}")
    if y is None:
        if method == "kendall":
            raise ValueError(
                "`method = \"kendall\"` has no matrix form; upstream Correlation has no "
                "KendallsTau(double[,]) overload"
            )
        xa = np.asarray(x, dtype=float)
        if xa.ndim != 2:
            raise ValueError("`x` must be a 2D array (observations in rows, variables in columns)")
        columns = [xa[:, j] for j in range(xa.shape[1])]
        r = _toolbox_run("correlation", f"{method}_matrix", columns)
        return np.asarray(r["values"], dtype=float).reshape(r["dims"][0], r["dims"][1])
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


def interpolate(x, y, xout, method: str = "linear", order: int | None = None,
                x_transform: str = "none", y_transform: str = "none",
                sort_order: str = "ascending", extrapolate: bool = False) -> np.ndarray:
    """Interpolate a paired series.

    Mirrors the C# ``Linear``, ``CubicSpline``, and ``Polynomial`` interpolaters of the
    Numerics library. ``x_transform``, ``y_transform``, and ``extrapolate`` are Linear-only in
    C# (neither ``CubicSpline`` nor ``Polynomial`` has a transform surface or an
    ``Extrapolate()`` method), so they must be left at their defaults for
    ``method="cubic_spline"`` or ``"polynomial"``.

    Parameters
    ----------
    x, y : array_like
        Equal-length knots.
    xout : array_like
        Positions to interpolate at.
    method : {"linear", "cubic_spline", "polynomial"}
    order : int, optional
        The polynomial order -- there are ``order + 1`` terms for each polynomial function.
        Required when ``method="polynomial"``; must be ``None`` otherwise.
    x_transform, y_transform : {"none", "logarithmic", "log", "normal_z"}
        Linear-only. ``"logarithmic"`` and ``"log"`` are equivalent aliases for the same
        ``Transform.Logarithmic`` value everywhere a transform argument appears in this module
        (:func:`interpolate`, :func:`interpolate_2d`, :func:`curve_interpolate`,
        :func:`tabular_function`); ``"logarithmic"`` (matching the C# enum member name) is the
        spelling used in this package's own examples and documentation.
    sort_order : {"ascending", "descending"}
        Describes ``x``.
    extrapolate : bool
        Whether to extend the end segments beyond the knots. ``False`` (the default) clamps to
        the end knot, matching the C# ``Interpolate()`` default; ``True`` calls the C#
        ``Extrapolate()`` method instead. Linear-only.

    Returns
    -------
    numpy.ndarray
        Same length as ``xout``.

    Examples
    --------
    >>> from corehydropy import interpolate
    >>> interpolate([1, 2, 3, 4], [10, 20, 30, 40], [1.5, 2.5])
    array([15., 25.])
    >>> interpolate([1, 2, 3, 4], [10, 20, 30, 40], [1.5, 2.5], method="cubic_spline")
    array([15., 25.])
    >>> interpolate([1, 2, 3, 4], [10, 20, 30, 40], [1.5, 2.5], method="polynomial", order=3)
    array([15., 25.])
    """
    xa, ya = _check_pair(x, y)
    if method not in ("linear", "cubic_spline", "polynomial"):
        raise ValueError(
            f"`method` must be one of 'linear', 'cubic_spline', 'polynomial'; got {method!r}"
        )
    if x_transform not in ("none", "logarithmic", "log", "normal_z"):
        raise ValueError(
            "`x_transform` must be one of 'none', 'logarithmic', 'log', 'normal_z'; got "
            f"{x_transform!r}"
        )
    if y_transform not in ("none", "logarithmic", "log", "normal_z"):
        raise ValueError(
            "`y_transform` must be one of 'none', 'logarithmic', 'log', 'normal_z'; got "
            f"{y_transform!r}"
        )
    if sort_order not in ("ascending", "descending"):
        raise ValueError(f"`sort_order` must be one of 'ascending', 'descending'; got {sort_order!r}")
    if method != "linear" and (x_transform != "none" or y_transform != "none" or extrapolate):
        raise ValueError(
            "`x_transform`, `y_transform`, and `extrapolate` are linear-only; the C# "
            "CubicSpline/Polynomial classes have neither a transform surface nor an "
            "Extrapolate() method"
        )
    if method == "polynomial":
        if order is None:
            raise ValueError('`order` is required when method="polynomial"')
    elif order is not None:
        raise ValueError('`order` only applies when method="polynomial"')
    xouta = np.asarray(xout, dtype=float).ravel()
    options: dict = {"sort_order": sort_order}
    if method == "linear":
        options.update({
            "x_transform": x_transform,
            "y_transform": y_transform,
            "extrapolate": bool(extrapolate),
        })
    elif method == "polynomial":
        options["order"] = int(order)
    return np.asarray(_toolbox_run("interpolation", method, [xa, ya, xouta], options)["values"])


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
    x1_transform, x2_transform, y_transform : {"none", "logarithmic", "log", "normal_z"}
        ``"logarithmic"`` and ``"log"`` are equivalent (see the module note on
        :func:`interpolate`); ``"logarithmic"`` is the spelling used in this package's own
        examples.
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
        if t not in ("none", "logarithmic", "log", "normal_z"):
            raise ValueError(
                f"`{name}` must be one of 'none', 'logarithmic', 'log', 'normal_z'; got {t!r}"
            )
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

    ``n`` and ``dimension`` are validated rather than passed through to the C++ layer, which
    would otherwise silently return an empty result for a non-positive value instead of raising
    an error.

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

    ``lower``, ``upper``, and ``bins`` are validated rather than passed through to the C++
    layer, which would otherwise silently return zero rows for ``lower >= upper`` (and for
    ``bins < 2``) instead of raising an error.

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
    if lower >= upper:
        raise ValueError(f"`lower` must be less than `upper`; got lower={lower}, upper={upper}")
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
        ``"correlation"`` requires both ``indicators`` and ``correlation`` -- the underlying C#
        method itself returns ``nan`` for that combination rather than raising an error, but
        this wrapper rejects it up front and names the missing argument(s), rather than handing
        back a silent ``nan``.
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
    if dependency == "correlation":
        missing = [name for name, val in (("indicators", indicators), ("correlation", correlation))
                  if val is None]
        if missing:
            raise ValueError(
                f"dependency='correlation' requires " + " and ".join(f"`{m}`" for m in missing)
            )
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


# The "link" and "trend" toolbox groups (Task 7). "link" mirrors the seven Numerics link
# functions (numerics/functions/) plus the five BestFit-specific ones (models/link_functions/);
# "trend" evaluates the eleven TrendModelType trend models (models/trend_functions/) that
# :class:`corehydropy.models.Trend` (the spec builder for :func:`model_univariate`'s
# ``trends=`` argument) already names -- these verbs consume that existing object rather than a
# second constructor. Mirrors corehydror's R/toolbox.R verb for verb.

class Link:
    """A link function spec, mirroring the seven Numerics link functions and the five
    RMC.BestFit-specific ones (the BestFit factory's own YeoJohnson case routes to the Numerics
    class -- see ``best_fit_link_function_factory.hpp`` -- so there is one ``"YeoJohnson"``, not
    two). Six of the twelve take construction parameters, passed by keyword.

    Parameters
    ----------
    type : str
        One of :func:`link_names`, matched case-insensitively.
    inner : Link, optional
        A :class:`Link` wrapped by ``"Centered"``; ignored (and rejected) for every other type.
        Keyword-only.
    **parameters
        Named construction parameters: ``lambda_`` for ``"YeoJohnson"``; ``gamma0``, ``scale``,
        ``epsilon``, ``delta`` for ``"ASinH"``; ``a`` for ``"SES"``; ``sigma0``, ``a``,
        ``lambda_`` for ``"LogSES"``; ``sigma0``, ``log_scale``, ``epsilon``, ``delta`` for
        ``"LogASinH"``; ``mu0``, ``scale`` for ``"Centered"``. ``lambda_`` is spelled with a
        trailing underscore because ``lambda`` is a reserved word in Python (matching the
        precedent set by :func:`corehydropy.stats.box_cox`/:func:`corehydropy.stats.yeo_johnson`,
        which take ``lambda_`` for the same reason); R spells the same construction parameter
        ``lambda`` (no collision there). A literal ``lambda=...`` keyword is a ``SyntaxError``
        before this class ever sees it, but ``**{"lambda": ...}`` still works if written that way.

    Examples
    --------
    >>> from corehydropy import link_function, link
    >>> l = link_function("Log")
    >>> link(l, [1, 10, 100])
    array([0.        , 2.30258509, 4.60517019])
    >>> yj = link_function("YeoJohnson", lambda_=0.5)
    """

    def __init__(self, type: str, *, inner: "Link | None" = None, **parameters) -> None:
        known = link_names()
        match = [t for t in known if t.lower() == str(type).lower()]
        if not match:
            raise ValueError(f"unknown link type '{type}'; expected one of {', '.join(known)}")
        self.type = match[0]
        if self.type == "Centered" and inner is None:
            raise ValueError("link type 'Centered' needs an `inner` link")
        if self.type != "Centered" and inner is not None:
            raise ValueError(f"`inner` is only used for type 'Centered'; got type '{self.type}'")
        if inner is not None and not isinstance(inner, Link):
            raise TypeError("`inner` must be a Link; create one with link_function()")
        # `lambda` is a reserved word in Python, so the documented spelling is `lambda_`
        # (matching stats.py's box_cox/yeo_johnson); fold it onto the "lambda" spec key the C++
        # grammar expects. `**{"lambda": ...}` still works undocumented, so both must not be
        # given together.
        if "lambda_" in parameters:
            if "lambda" in parameters:
                raise ValueError("pass either `lambda_` or `lambda`, not both")
            parameters = dict(parameters)
            parameters["lambda"] = parameters.pop("lambda_")
        self.parameters = {k: float(v) for k, v in parameters.items()} if parameters else None
        self.inner = inner

    def _to_spec(self) -> dict:
        # Omit "parameters"/"inner" entirely rather than emitting `null` -- unlike
        # to_spec_json() on the R side, json.dumps() has no NULL-key-dropping rule, and
        # build_link's `spec.contains("parameters")` would otherwise see a present-but-null key.
        spec: dict = {"type": self.type}
        if self.parameters is not None:
            spec["parameters"] = self.parameters
        if self.inner is not None:
            spec["inner"] = self.inner._to_spec()
        return spec

    def __repr__(self) -> str:
        return f"<Link {self.type}>"


def link_function(type: str, *, inner: "Link | None" = None, **parameters) -> Link:
    """Construct a :class:`Link`. See :class:`Link` for the arguments.

    Examples
    --------
    >>> from corehydropy import link_function, link, link_inverse
    >>> l = link_function("Log")
    >>> link(l, [1, 10, 100])
    array([0.        , 2.30258509, 4.60517019])
    >>> link_inverse(l, [0, 1, 2])
    array([1.        , 2.71828183, 7.3890561 ])
    """
    return Link(type, inner=inner, **parameters)


def _link_eval(l: Link, method: str, v) -> np.ndarray:
    if not isinstance(l, Link):
        raise TypeError("`l` must be a Link; create one with link_function()")
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("input must be a non-empty numeric array")
    r = _toolbox_run("link", method, [arr], {"link": l._to_spec()})
    return np.asarray(r["values"], dtype=float)


def link(l: Link, x) -> np.ndarray:
    """Evaluate a link function: real-space to link-space.

    Parameters
    ----------
    l : Link
        From :func:`link_function`.
    x : array_like
        Values on the data scale.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> from corehydropy import link_function, link
    >>> l = link_function("Logit")
    >>> link(l, [0.1, 0.5, 0.9])
    array([-2.19722458,  0.        ,  2.19722458])
    """
    return _link_eval(l, "link", x)


def link_inverse(l: Link, eta) -> np.ndarray:
    """Evaluate the inverse of a link function: link-space back to real-space.

    Parameters
    ----------
    l : Link
        From :func:`link_function`.
    eta : array_like
        Values on the linear-predictor scale.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> from corehydropy import link_function, link_inverse
    >>> l = link_function("Logit")
    >>> link_inverse(l, [-2.19722458, 0.0, 2.19722458])
    array([0.1, 0.5, 0.9])
    """
    return _link_eval(l, "inverse_link", eta)


def link_derivative(l: Link, x) -> np.ndarray:
    """Evaluate a link function's derivative with respect to ``x``.

    Parameters
    ----------
    l : Link
        From :func:`link_function`.
    x : array_like
        Values on the data scale.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> from corehydropy import link_function, link_derivative
    >>> l = link_function("Log")
    >>> link_derivative(l, [1, 10, 100])
    array([1.  , 0.1 , 0.01])
    """
    return _link_eval(l, "d_link", x)


def link_names() -> list[str]:
    """List the twelve link types :func:`link_function` accepts.

    Calls through to the C++ ``link`` toolbox group's own ``"names"`` method -- the same table
    ``build_link`` builds a link from (``link_builder_table()`` in ``link.hpp``) -- so this list
    can't drift from what :func:`link_function` actually accepts.

    Examples
    --------
    >>> from corehydropy import link_names
    >>> link_names()
    ['Identity', 'Log', 'Logit', 'Probit', 'ComplementaryLogLog', 'FisherZ', 'YeoJohnson', 'ASinH', 'SES', 'LogSES', 'LogASinH', 'Centered']
    """
    return list(_toolbox_run("link", "names")["names"])


def _trend_index(idx, what: str = "index") -> list[int]:
    """Internal: user-facing trend evaluation indices are 1-based (matching Trend/ModelParameter
    in models.py); the spec and the C++ take 0-based. Unlike ``_mv_indices`` in mvdist.py, there
    is no upper bound (a trend's time index is not a fixed dimension count) and no
    no-duplicate check (predicting the same point twice is not an error)."""
    arr = np.atleast_1d(np.asarray(idx))
    if arr.size == 0:
        raise ValueError(f"`{what}` must be a non-empty numeric array")
    farr = arr.astype(float)
    if np.any(np.isnan(farr)):
        raise ValueError(f"`{what}` must be a non-empty numeric array")
    if np.any(farr != np.trunc(farr)):
        raise ValueError(f"`{what}` must be whole numbers; got {farr[farr != np.trunc(farr)]}")
    return (farr.astype(int) - 1).tolist()


def _trend_spec(tr) -> dict:
    """Internal: the plain {"type", "start_index"?, "values"?} object build_spec_trend() wants
    (numerics/support/toolbox/trend.hpp) -- drops Trend's model-attachment-only `parameter`
    field."""
    from .models import Trend

    if not isinstance(tr, Trend):
        raise TypeError("`tr` must be a Trend; create one with trend()")
    # Omit "start_index"/"values" entirely when unset rather than emitting `null` -- see the
    # matching note on Link._to_spec above; build_spec_trend's `spec.contains("start_index")`
    # would otherwise see a present-but-null key and fail trying to read it as a number.
    spec: dict = {"type": tr.type}
    if tr.start_index is not None:
        spec["start_index"] = tr.start_index
    if tr.values is not None:
        spec["values"] = tr.values
    return spec


def trend_predict(tr, index) -> np.ndarray:
    """Evaluate a trend model.

    Builds the trend model named by ``tr`` (a :class:`corehydropy.models.Trend`) from its own
    class defaults, then evaluates it at ``index``. ``tr.parameter`` (which distribution
    parameter the trend attaches to) is not used here -- this evaluates the trend model on its
    own, the same way the model-attachment path's type dispatch does before it further
    data-drives the parameter values from a live model's distribution and data (this function
    has no such model to consult, so a ``tr`` with no explicit ``values`` predicts from the
    trend's own zero-valued class defaults).

    Parameters
    ----------
    tr : corehydropy.models.Trend
        From :func:`corehydropy.models.trend`.
    index : array_like
        1-based time indices to evaluate at.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> from corehydropy import trend, trend_predict
    >>> tr = trend("location", "Linear", start_index=0, values=[10, 2])
    >>> trend_predict(tr, [1, 2, 3, 4, 5])
    array([10., 12., 14., 16., 18.])
    """
    idx = _trend_index(index, "index")
    r = _toolbox_run("trend", "predict", [np.asarray(idx, dtype=float)], {"trend": _trend_spec(tr)})
    return np.asarray(r["values"], dtype=float)


def trend_parameters(tr) -> dict:
    """The trend's own model-parameter values, by name.

    Parameters
    ----------
    tr : corehydropy.models.Trend
        From :func:`corehydropy.models.trend`.

    Returns
    -------
    dict
        The trend's model-parameter values (see :func:`corehydropy.models.model_parameter`) at
        their class defaults, adjusted by any explicit ``values`` on ``tr``.

    Examples
    --------
    >>> from corehydropy import trend, trend_parameters
    >>> trend_parameters(trend("location", "Linear", values=[10, 2]))
    {'(α)': 10.0, '(β)': 2.0}
    """
    r = _toolbox_run("trend", "parameters", [], {"trend": _trend_spec(tr)})
    return dict(zip(r["names"], r["values"]))


def trend_names() -> list[str]:
    """List the eleven trend types :func:`corehydropy.models.trend` accepts.

    Calls through to the C++ ``trend`` toolbox group's own ``"names"`` method -- the same table
    :func:`corehydropy.models.trend` and ``build_spec_trend`` (``model_spec.hpp``) validate
    ``type`` against (``trend_model_type_table()``) -- so this list can't drift from what
    :func:`corehydropy.models.trend` actually accepts.

    Examples
    --------
    >>> from corehydropy import trend_names
    >>> trend_names()
    ['Constant', 'Cubic', 'Exponential', 'Linear', 'Logistic', 'Power', 'Quadratic', 'Reciprocal', 'Sinusoidal', 'StepFunction', 'GeneralLinear']
    """
    return list(_toolbox_run("trend", "names")["names"])


# The "linalg" toolbox group (P2 "math extras" Task 9): QRDecomposition (Householder
# reflections) and GaussJordanElimination. Matrices cross the runner boundary as ONE
# flattened row-major vector plus `rows`/`cols` options, the same convention
# linear_regression()'s predictor matrix uses above; numpy's default `ravel()`/`reshape()`
# are already row-major (C order), so -- unlike the R verbs, which must transpose first --
# these need no extra flattening step.


def qr_decomposition(a) -> dict:
    """QR decomposition.

    Mirrors the C# ``QRDecomposition`` class (Householder reflections): decomposes the
    ``M x N`` array ``a`` into an ``M x M`` orthogonal array ``q`` and an ``M x N`` upper
    triangular array ``r`` such that ``q @ r`` reproduces ``a``. The C# ``RMatrix`` property
    is named ``r`` here (``R`` has no naming collision in this package, unlike in the C#
    codebase).

    Parameters
    ----------
    a : array_like
        A 2D array, M x N.

    Returns
    -------
    dict
        ``{"q": ndarray (M, M), "r": ndarray (M, N)}``.

    Examples
    --------
    >>> from corehydropy import qr_decomposition
    >>> a = [[1, 1, 1], [0, 2, 5], [2, 5, -1]]
    >>> qr = qr_decomposition(a)
    >>> import numpy as np
    >>> np.allclose(qr["q"] @ qr["r"], a)
    True
    """
    aa = np.asarray(a, dtype=float)
    if aa.ndim != 2:
        raise ValueError("`a` must be a 2D array")
    rows, cols = aa.shape
    options = {"rows": int(rows), "cols": int(cols)}
    data = [aa.ravel()]
    q = _toolbox_run("linalg", "qr_q", data, options)
    r = _toolbox_run("linalg", "qr_r", data, options)
    return {
        "q": np.asarray(q["values"], dtype=float).reshape(q["dims"][0], q["dims"][1]),
        "r": np.asarray(r["values"], dtype=float).reshape(r["dims"][0], r["dims"][1]),
    }


def qr_solve(a, b):
    """Solve a linear system by QR decomposition.

    Mirrors the C# ``QRDecomposition.Solve`` overloads (vector and matrix right-hand
    sides). ``a`` need not be square: an overdetermined system is solved in the
    least-squares sense; an underdetermined system leaves the trailing
    ``a.shape[1] - min(a.shape)`` unknowns at zero (matching ``QRDecomposition.Solve``'s
    own truncated back-substitution, which only ever fills indices below
    ``min(m, n)``).

    Parameters
    ----------
    a : array_like
        A 2D array, M x N.
    b : array_like
        A 1D array of length M, or a 2D array with M rows.

    Returns
    -------
    numpy.ndarray
        A 1D array of length N when ``b`` is 1D, or an ``N x b.shape[1]`` array when ``b``
        is 2D.

    Examples
    --------
    >>> from corehydropy import qr_solve
    >>> a = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    >>> qr_solve(a, [1, 2, 3])
    array([27., -22.,   6.])
    """
    aa = np.asarray(a, dtype=float)
    if aa.ndim != 2:
        raise ValueError("`a` must be a 2D array")
    rows, cols = aa.shape
    options = {"rows": int(rows), "cols": int(cols)}
    ba = np.asarray(b, dtype=float)
    if ba.ndim == 2:
        if ba.shape[0] != rows:
            raise ValueError(f"`b` must have {rows} rows, matching `a`; got {ba.shape[0]}")
        options["b_cols"] = int(ba.shape[1])
        r = _toolbox_run("linalg", "qr_solve_matrix", [aa.ravel(), ba.ravel()], options)
        return np.asarray(r["values"], dtype=float).reshape(r["dims"][0], r["dims"][1])
    bv = ba.ravel()
    if bv.size != rows:
        raise ValueError(f"`b` must have length {rows}, matching `a`'s rows; got {bv.size}")
    r = _toolbox_run("linalg", "qr_solve", [aa.ravel(), bv], options)
    return np.asarray(r["values"], dtype=float)


def gauss_jordan(a, b=None) -> dict:
    """Gauss-Jordan elimination.

    Mirrors the C# ``GaussJordanElimination.Solve(ref Matrix A, ref Matrix B)``: one
    full-pivot Gauss-Jordan reduction that produces ``a``'s inverse and, when ``b`` is
    supplied, the solution set of ``a @ x = b``. Unlike the C# ``ref, ref`` in-place API,
    ``a`` and ``b`` are never mutated in Python -- both results come back as new arrays.

    Parameters
    ----------
    a : array_like
        A square 2D array, N x N.
    b : array_like, optional
        A 1D array of length N, or a 2D array with N rows (the right-hand sides). Omit for
        the inverse alone, in which case ``solution`` is an ``N x 0`` array.

    Returns
    -------
    dict
        ``{"inverse": ndarray (N, N), "solution": ndarray (N, b.shape[1])}``.

    Examples
    --------
    >>> from corehydropy import gauss_jordan
    >>> a = [[1, 3, 3], [1, 4, 3], [1, 3, 4]]
    >>> gauss_jordan(a)["inverse"]
    array([[ 7., -3., -3.],
           [-1.,  1.,  0.],
           [-1.,  0.,  1.]])
    """
    aa = np.asarray(a, dtype=float)
    if aa.ndim != 2 or aa.shape[0] != aa.shape[1]:
        raise ValueError("`a` must be a square 2D array")
    n = aa.shape[0]
    if b is None:
        ba = np.zeros((n, 0))
    else:
        ba = np.asarray(b, dtype=float)
        if ba.ndim == 1:
            ba = ba.reshape(-1, 1)
        if ba.shape[0] != n:
            raise ValueError(f"`b` must have {n} rows, matching `a`; got {ba.shape[0]}")
    options = {"rows": int(n), "cols": int(n), "b_cols": int(ba.shape[1])}
    data = [aa.ravel(), ba.ravel()]
    inv = _toolbox_run("linalg", "gauss_jordan_inverse", data, options)
    sol = _toolbox_run("linalg", "gauss_jordan_solution", data, options)
    return {
        "inverse": np.asarray(inv["values"], dtype=float).reshape(inv["dims"][0], inv["dims"][1]),
        "solution": np.asarray(sol["values"], dtype=float).reshape(sol["dims"][0], sol["dims"][1]),
    }


# The "special" toolbox group (P2 "math extras" Task 10): the ported Debye and Evaluate special
# functions. Both verbs vectorize over ``x``, returning one value per element -- there is no
# matrix result here, so no flatten/reshape step like ``linalg``'s above.


def debye(x) -> np.ndarray:
    """The Debye function.

    Mirrors the C# ``Debye.Function``: a piecewise approximation of
    ``D(x) = (n/x**n) * integral_0^x t**n / (e**t - 1) dt``, vectorized over ``x``.

    Parameters
    ----------
    x : array_like
        Non-negative values.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> from corehydropy import debye
    >>> debye(0.5)
    array([0.8249631])
    """
    xa = np.atleast_1d(np.asarray(x, dtype=float))
    r = _toolbox_run("special", "debye", [xa], {})
    return np.asarray(r["values"], dtype=float)


def polynomial_eval(coefficients, x, variant: str = "standard", n: int | None = None) -> np.ndarray:
    """Evaluate a polynomial.

    Mirrors the C# ``Evaluate`` class's three polynomial evaluators (Horner's method),
    vectorized over ``x`` against one shared ``coefficients`` array: ``variant="standard"``
    calls ``Evaluate.Polynomial`` (coefficients in ASCENDING order, ``coefficients[0]`` the
    constant term); ``"reverse"`` calls ``Evaluate.PolynomialRev`` (coefficients in DESCENDING
    order, ``coefficients[0]`` the highest-order term), optionally truncated to order ``n + 1``
    via ``n``; ``"reverse_unit"`` calls ``Evaluate.PolynomialRev_1`` (DESCENDING order with an
    implicit leading coefficient of 1).

    Parameters
    ----------
    coefficients : array_like
        Polynomial coefficients.
    x : array_like
        Points at which to evaluate.
    variant : {"standard", "reverse", "reverse_unit"}
    n : int, optional
        Redefines the polynomial's order to ``n + 1``. Only valid with ``variant="reverse"``.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> from corehydropy import polynomial_eval
    >>> polynomial_eval([3, 5, 7], 4)
    array([135.])
    >>> polynomial_eval([3, 5, 7], 4, variant="reverse")
    array([75.])
    >>> polynomial_eval([3, 5, 7], 4, variant="reverse", n=1)
    array([17.])
    """
    if variant not in ("standard", "reverse", "reverse_unit"):
        raise ValueError(
            f"`variant` must be one of 'standard', 'reverse', 'reverse_unit'; got {variant!r}"
        )
    if n is not None and variant != "reverse":
        raise ValueError('`n` is only valid with variant="reverse"')
    method = {"standard": "polynomial", "reverse": "polynomial_rev",
              "reverse_unit": "polynomial_rev_1"}[variant]
    ca = np.atleast_1d(np.asarray(coefficients, dtype=float))
    xa = np.atleast_1d(np.asarray(x, dtype=float))
    options = {} if n is None else {"n": int(n)}
    r = _toolbox_run("special", method, [ca, xa], options)
    return np.asarray(r["values"], dtype=float)


# The "functions" toolbox group (P2 "math extras" Task 11): the two non-tabular
# IUnivariateFunction implementations (numerics/functions/), LinearFunction and PowerFunction.
# The severed third implementation, TabularFunction, depends on the unported Paired Data
# subsystem (see upstream/CLAUDE.md) and is not exposed.


def univariate_function(
    type: str,
    parameters,
    x,
    inverse: bool = False,
    is_inverse: bool = False,
    confidence_level: float | None = None,
) -> np.ndarray:
    """Evaluate a univariate function.

    Mirrors the Numerics ``LinearFunction`` (``Y = alpha + beta*X + epsilon``) and
    ``PowerFunction`` (``Y = alpha * (X - xi)**beta * epsilon``), both over optional normally
    distributed noise (``epsilon ~ Normal(0, sigma)``) via ``confidence_level``. ``is_inverse``
    (``PowerFunction``'s own ``IsInverse`` switch) selects which of the forward power law or its
    algebraic inverse ``Function()``/``inverse=True`` evaluates -- an independent axis from
    ``inverse`` itself, which picks ``Function()`` vs. ``InverseFunction()`` on whichever of the
    two ``is_inverse`` selects.

    Parameters
    ----------
    type : {"linear", "power"}
        Matched case-insensitively.
    parameters : array_like
        ``[alpha, beta, sigma]`` for ``"linear"``; ``[alpha, beta, xi, sigma]`` for ``"power"``.
        ``sigma`` is still required (e.g. 0) when ``confidence_level`` is ``None`` -- it only
        enters the calculation on the non-deterministic path.
    x : array_like
        The values to evaluate the function at, or (when ``inverse=True``) the values to
        evaluate the inverse function at.
    inverse : bool, default False
        If ``True``, evaluates the inverse function (``InverseFunction()``) instead of the
        forward function (``Function()``).
    is_inverse : bool, default False
        ``"power"``-only: ``PowerFunction``'s own ``IsInverse`` property. An error for
        ``type="linear"``.
    confidence_level : float, optional
        If given, evaluates the non-deterministic path at this quantile level; if ``None``
        (default), evaluates deterministically.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> from corehydropy import univariate_function
    >>> univariate_function("linear", [0, 1, 0], [1, 2, 3])
    array([1., 2., 3.])
    >>> univariate_function("power", [5, 2, 0, 3], 6)
    array([180.])
    >>> univariate_function("power", [5, 2, 0, 3], 6, confidence_level=0.75)
    array([1361.614084])
    """
    known = ("linear", "power")
    match = [t for t in known if t == str(type).lower()]
    if not match:
        raise ValueError(f"unknown function type '{type}'; expected one of {', '.join(known)}")
    ftype = match[0]
    if is_inverse and ftype != "power":
        raise ValueError(f"`is_inverse` is only used for type 'power'; got type '{ftype}'")
    pa = np.atleast_1d(np.asarray(parameters, dtype=float))
    if pa.size == 0:
        raise ValueError("`parameters` must be a non-empty array")
    xa = np.atleast_1d(np.asarray(x, dtype=float))
    if xa.size == 0:
        raise ValueError("`x` must be a non-empty array")
    options: dict = {"function": ftype, "parameters": pa.tolist()}
    if ftype == "power":
        options["is_inverse"] = bool(is_inverse)
    if confidence_level is not None:
        options["confidence_level"] = float(confidence_level)
    method = "inverse" if inverse else "evaluate"
    r = _toolbox_run("functions", method, [xa], options)
    return np.asarray(r["values"], dtype=float)


# The "network" toolbox group (P3 optimizers Task 10): the Dijkstra shortest-path solver over an
# edge list. Edges cross the runner boundary as four parallel numeric vectors -- from, to, weight,
# edge index -- with the destinations in the options object, and the result table comes back
# flattened row-major with dims = {node_count, 3}, the same convention the `linalg` matrix results
# use above. Mirrors corehydror's shortest_path().


def _check_node_indices(x: np.ndarray, what: str) -> None:
    """Internal: a node index is a whole, non-negative number.

    The shared C++ group header checks this too (a fixture case reaches it directly), but
    checking here as well keeps the message naming the Python argument and keeps the two
    packages' errors identical.
    """
    if not np.all(np.isfinite(x)) or np.any(x != np.floor(x)) or np.any(x < 0):
        raise ValueError(f"`{what}` must be whole, non-negative node indices")


def shortest_path(
    frm,
    to,
    weight,
    destinations,
    edge_index=None,
    node_count: int | None = None,
) -> np.ndarray:
    """Solve the shortest paths through a network.

    Mirrors the C# ``Dijkstra.Solve`` overloads: solves the cheapest route from EVERY node of a
    directed, weighted graph to a set of destination nodes at once, running the search backwards
    from the destinations. The answer is a routing table -- for each node, which neighbour to
    step to, along which edge, and at what remaining cost.

    Node indices are 0-based in both ``corehydropy`` and ``corehydror``, matching the C# result
    table the two packages share; a graph with ``n`` nodes uses indices ``0`` to ``n - 1``.
    Unreachable nodes carry ``cost = inf`` with ``next_node = -1`` and ``edge_index = -1``, and a
    destination node carries ``cost = 0`` with ``next_node`` equal to its own index.

    Costs accumulate in single precision, because the ported solver does (C# declares
    ``float Weight`` and its own tests assert the table by exact ``float`` equality). Fractional
    weights therefore round to ``float`` before they are summed.

    Parameters
    ----------
    frm, to, weight : array_like
        Arrays of the same length, one element per directed edge: the start node index, the end
        node index, and the cost of traversing the edge. ``frm`` and ``to`` must be whole,
        non-negative numbers. (``frm``, not ``from``, because ``from`` is a Python keyword; the R
        twin spells it ``from``.)
    destinations : array_like or int
        One or more destination node indices. With several destinations, each node keeps
        whichever destination it reaches most cheaply.
    edge_index : array_like, optional
        An array the same length as ``frm``, labelling each edge (typically an index into
        whatever the edges came from -- a river reach, a road segment). Defaults to
        ``range(len(frm))``. These labels are what the ``edge_index`` result column reports, and
        they need not be distinct.
    node_count : int, optional
        Defaults to ``max(frm, to) + 1``; supply a larger value to include isolated nodes
        carrying no edge, which then report ``cost = inf``. A value below ``max(frm, to) + 1``
        is an error: the graph would not fit the routing table it asks for.

    Returns
    -------
    numpy.ndarray
        One row per node, in node-index order, with columns ``[next_node, edge_index, cost]``.

    Examples
    --------
    >>> from corehydropy import shortest_path
    >>> shortest_path([0, 1], [1, 2], [1, 1], destinations=2, node_count=4)
    array([[ 1.,  0.,  2.],
           [ 2.,  1.,  1.],
           [ 2., -1.,  0.],
           [-1., -1., inf]])
    """
    f = np.atleast_1d(np.asarray(frm, dtype=float)).ravel()
    t = np.atleast_1d(np.asarray(to, dtype=float)).ravel()
    w = np.atleast_1d(np.asarray(weight, dtype=float)).ravel()
    n = f.size
    if n == 0:
        raise ValueError("`frm`, `to` and `weight` must describe at least one edge")
    if t.size != n or w.size != n:
        raise ValueError(
            "`frm`, `to` and `weight` must have the same length; "
            f"got {n}, {t.size} and {w.size}"
        )
    if edge_index is None:
        idx = np.arange(n, dtype=float)
    else:
        idx = np.atleast_1d(np.asarray(edge_index, dtype=float)).ravel()
        if idx.size != n:
            raise ValueError(
                "`edge_index` must be an array the same length as `frm`; "
                f"got {idx.size} for {n}"
            )
    dest = np.atleast_1d(np.asarray(destinations, dtype=float)).ravel()
    if dest.size == 0:
        raise ValueError("`destinations` must name at least one destination node")
    _check_node_indices(f, "frm")
    _check_node_indices(t, "to")
    _check_node_indices(dest, "destinations")
    options: dict = {"destinations": dest.tolist()}
    n_nodes = float(max(f.max(), t.max())) + 1.0
    if node_count is not None:
        if float(node_count) < 1:
            raise ValueError("`node_count` must be a single positive number")
        # A `node_count` below `max(frm, to) + 1` cannot describe the edge list. The ported
        # solver raises the C# IndexOutOfRangeException message from inside itself when it
        # reaches the offending index, which is both late and unhelpful here, so reject it up
        # front and name the argument. This is deliberately STRICTER than the C# solver, whose
        # bounds check is lazy: it accepts a too-small count as long as no out-of-range index is
        # ever reached (see dijkstra.hpp note 9). That input is a graph the caller cannot have
        # meant.
        if float(node_count) < n_nodes:
            raise ValueError(
                f"`node_count` must be at least {int(n_nodes)}, the number of nodes `frm` and "
                f"`to` describe; got {int(node_count)}"
            )
        n_nodes = float(node_count)
        options["node_count"] = n_nodes
    if bool(np.any(dest >= n_nodes)):
        raise ValueError(
            f"`destinations` is out of range for a network of {int(n_nodes)} nodes"
        )
    r = _toolbox_run("network", "dijkstra", [f, t, w, idx], options)
    values = np.asarray(r["values"], dtype=float)
    return values.reshape(int(r["dims"][0]), int(r["dims"][1]))


# The "hypothesis" toolbox group (P4 Task 3, completed in P5): the thirteen ported hypothesis
# tests over numerics/data/hypothesis_tests.hpp (a port of the C# `HypothesisTests` static class). Mirrors
# corehydror's own hypothesis_test() verb; both packages share this signature and produce
# identical error text so a change here is not one-sided.

_HYPOTHESIS_METHODS = (
    "one_sample_t", "equal_variance_t", "unequal_variance_t", "paired_t", "f", "f_models",
    "jarque_bera", "wald_wolfowitz", "ljung_box", "mann_whitney", "mann_kendall", "linear_trend",
    "unimodality",
)
_HYPOTHESIS_TWO_SAMPLE = ("equal_variance_t", "unequal_variance_t", "paired_t", "f", "mann_whitney")


def hypothesis_test(
    x=None,
    y=None,
    method: str = "jarque_bera",
    population_mean: float = 0.0,
    lag_max: int | None = None,
    index=None,
    sse_restricted: float | None = None,
    sse_full: float | None = None,
    df_restricted: int | None = None,
    df_full: int | None = None,
) -> dict:
    """Hypothesis tests.

    Mirrors the C# ``HypothesisTests`` static class: thirteen one- and two-sample parametric and
    nonparametric hypothesis tests, reached through the shared ``hypothesis`` toolbox group.
    Every method but ``"f_models"`` returns the 2-sided p-value of its test statistic;
    ``"f_models"`` (the F-test comparing two nested regression models) additionally returns the
    F statistic itself.

    Argument use by method, and the C# guard each one inherits:

    - ``"one_sample_t"``: ``x``, ``population_mean`` (default 0). Needs at least 2 observations.
    - ``"equal_variance_t"`` / ``"unequal_variance_t"``: ``x``, ``y``. ``equal_variance_t``
      needs a combined length of at least 3; ``unequal_variance_t`` has no length guard
      (upstream has none either).
    - ``"paired_t"``: ``x``, ``y``, which must be the same length.
    - ``"f"``: ``x``, ``y``, each needing at least 2 observations.
    - ``"f_models"``: ``sse_restricted``, ``sse_full``, ``df_restricted``, ``df_full`` (all
      required; ``x`` and ``y`` are ignored). ``df_restricted`` must differ from ``df_full``,
      and ``df_full`` must be positive.
    - ``"jarque_bera"`` / ``"wald_wolfowitz"``: ``x``. No length guard.
    - ``"ljung_box"``: ``x``, ``lag_max`` (default ``None``, meaning
      ``floor(min(10 * log10(len(x)), len(x) - 1))``, the C# default rule).
    - ``"mann_whitney"``: ``x``, ``y``. ``x`` must be no longer than ``y``, each must have more
      than 3 observations, and the combined length must exceed 20.
    - ``"mann_kendall"``: ``x``. Needs at least 10 observations.
    - ``"linear_trend"``: ``x`` (the sample), ``index`` (default ``1..len(x)`` -- a VALUE the
      regression is fit against, not an index into ``x``). ``index`` and ``x`` must be the same
      length.
    - ``"unimodality"``: ``x``. Needs at least 10 observations. Fits a 1-component and a
      2-component Gaussian mixture model (both at the hard-coded seed 12345, so the result is
      deterministic) and returns the p-value of the likelihood-ratio statistic against a
      chi-square with 3 degrees of freedom, so a SMALL p-value is evidence against unimodality.
      If either mixture fit fails numerically the result is ``nan`` rather than an error,
      matching upstream.

    Parameters
    ----------
    x : array_like
        The sample (the two-sample methods' first sample, or the response series for
        ``"linear_trend"``). Ignored for ``"f_models"``.
    y : array_like, optional
        The second sample. Required for the two-sample methods listed above, and rejected (must
        be ``None``) for every other method -- earlier versions silently discarded a ``y``
        supplied to a one-sample method instead of raising.
    method : str
        One of ``"one_sample_t"``, ``"equal_variance_t"``, ``"unequal_variance_t"``,
        ``"paired_t"``, ``"f"``, ``"f_models"``, ``"jarque_bera"``, ``"wald_wolfowitz"``,
        ``"ljung_box"``, ``"mann_whitney"``, ``"mann_kendall"``, ``"linear_trend"``,
        ``"unimodality"``.
    population_mean : float
        The hypothesized mean for ``"one_sample_t"``. Default 0.
    lag_max : int, optional
        The max lag for ``"ljung_box"``. Default ``None`` (use the C# default rule).
    index : array_like, optional
        The index (x-axis) vector for ``"linear_trend"``. Default ``None``, meaning
        ``1..len(x)``.
    sse_restricted, sse_full, df_restricted, df_full : float, optional
        The four ``"f_models"`` inputs: the restricted and full models' sum of squared errors
        and degrees of freedom.

    Returns
    -------
    dict
        ``{"p_value": ...}`` for every method but ``"f_models"``, which returns
        ``{"f_statistic": ..., "p_value": ...}``.

    Examples
    --------
    >>> from corehydropy import hypothesis_test
    >>> round(
    ...     hypothesis_test(
    ...         [4, 5, 5, 6, 9, 12, 13, 14, 14, 19, 22, 24, 25], method="jarque_bera"
    ...     )["p_value"],
    ...     6,
    ... )
    0.592128
    """
    if method not in _HYPOTHESIS_METHODS:
        known = ", ".join(f'"{m}"' for m in _HYPOTHESIS_METHODS)
        raise ValueError(f'`method` must be one of {known}; got "{method}"')
    if method in _HYPOTHESIS_TWO_SAMPLE and y is None:
        raise ValueError(f'`y` is required for method "{method}"')
    # C2 (P4 whole-branch review): a one-sample method silently DISCARDED a non-None `y` --
    # `hypothesis_test(a, b)` at the default method equalled `hypothesis_test(a)`. Reject it
    # instead, mirroring corehydror's own check.
    if method not in _HYPOTHESIS_TWO_SAMPLE and y is not None:
        raise ValueError(f'`y` is not used by method "{method}"; leave it None')

    if method == "f_models":
        if sse_restricted is None or sse_full is None or df_restricted is None or df_full is None:
            raise ValueError(
                "`sse_restricted`, `sse_full`, `df_restricted`, and `df_full` are all "
                'required for method "f_models"'
            )
        r = _toolbox_run(
            "hypothesis",
            "f_models",
            [],
            {
                "sse_restricted": float(sse_restricted),
                "sse_full": float(sse_full),
                "df_restricted": float(df_restricted),
                "df_full": float(df_full),
            },
        )
        return dict(zip(r["names"], r["values"]))

    if method == "linear_trend":
        xa = np.asarray(x, dtype=float).ravel()
        idx = np.arange(1, xa.size + 1, dtype=float) if index is None else np.asarray(index, dtype=float).ravel()
        r = _toolbox_run("hypothesis", "linear_trend", [idx, xa])
        return {"p_value": r["values"][0]}

    if method in _HYPOTHESIS_TWO_SAMPLE:
        data = [np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()]
    else:
        data = [np.asarray(x, dtype=float).ravel()]

    options: dict = {}
    if method == "one_sample_t":
        options = {"population_mean": float(population_mean)}
    elif method == "ljung_box":
        options = {"lag_max": float(-1 if lag_max is None else lag_max)}

    r = _toolbox_run("hypothesis", method, data, options)
    return {"p_value": r["values"][0]}


# The "paired_data" toolbox group (P4 Task 10): OrderedPairedData/UncertainOrderedPairedData/
# LineSimplification (numerics/data/paired_data/, P4 Tasks 7-9), plus TabularFunction's
# tabular/tabular_inverse arm on the "functions" group. Mirrors corehydror's toolbox.R verb for
# verb. Every curve verb below reads the same strict_x/strict_y/order_x/order_y shape contract
# paired_data.hpp documents; curve_interpolate() additionally reads x_transform/y_transform. Only
# five verbs are exported -- line_simplify, search, and is_valid are reachable through the
# fixture/oracle-gate surface but are not given a Python-facing wrapper by this task.

_SORT_ORDERS = ("ascending", "descending", "none")
# "logarithmic" (matching the C# enum member name `Transform.Logarithmic`) is this package's
# documented spelling; "log" is accepted as an equivalent alias (both parse to the same value
# core-side) so a value valid for interpolate()/interpolate_2d() is also valid here -- see the P4
# whole-branch-review finding M2.
_PAIRED_TRANSFORMS = ("none", "logarithmic", "log", "normal_z")


def _check_sort_order(value: str, what: str) -> None:
    # Message format mirrors corehydror's check_choice() (R/fit.R:22) so the two languages raise
    # textually identical text for the identical mistake -- see the P4 whole-branch-review
    # finding M3.
    if value not in _SORT_ORDERS:
        raise ValueError(f"unknown {what} '{value}'; expected one of {', '.join(_SORT_ORDERS)}")


def _check_paired_transform(value: str, what: str) -> None:
    if value not in _PAIRED_TRANSFORMS:
        raise ValueError(
            f"unknown {what} '{value}'; expected one of {', '.join(_PAIRED_TRANSFORMS)}"
        )


def _paired_data_shape_opts(strict_x: bool, strict_y: bool, order_x: str, order_y: str) -> dict:
    _check_sort_order(order_x, "order_x")
    _check_sort_order(order_y, "order_y")
    return {
        "strict_x": bool(strict_x),
        "strict_y": bool(strict_y),
        "order_x": order_x,
        "order_y": order_y,
    }


def _paired_data_distributions(distributions, x) -> list:
    """Internal: build a list of Distribution specs, recycling a single one across every `x`.

    Shared by uncertain_curve_sample() and tabular_function(), which use identical recycling
    and error text (see corehydror's paired_data_distributions()).
    """
    if isinstance(distributions, Distribution):
        distributions = [distributions]
    if (
        not isinstance(distributions, (list, tuple))
        or len(distributions) == 0
        or not all(isinstance(d, Distribution) for d in distributions)
    ):
        raise ValueError("`distributions` must be a Distribution or a list of Distribution objects")
    distributions = list(distributions)
    if len(distributions) == 1 and len(x) > 1:
        distributions = distributions * len(x)
    if len(distributions) != len(x):
        raise ValueError(
            f"`distributions` must have length 1 or length(x) ({len(x)}); got {len(distributions)}"
        )
    return distributions


def curve_interpolate(
    x,
    y,
    xout=None,
    yout=None,
    x_transform: str = "none",
    y_transform: str = "none",
    order_x: str = "ascending",
    order_y: str = "ascending",
    strict_x: bool = True,
    strict_y: bool = True,
) -> np.ndarray:
    """Interpolate a paired x-y curve.

    Mirrors the C# ``OrderedPairedData.GetYFromX``/``GetXFromY``: linear interpolation over a
    curve that keeps itself sorted/validated against a caller-chosen monotonicity contract, with
    optional per-axis transforms (log10 or the standard normal z-score) applied before/after
    interpolation. Exactly one of ``xout``/``yout`` must be supplied.

    Parameters
    ----------
    x, y : array_like
        Equal-length curve ordinates, at least two elements.
    xout : array_like, optional
        Positions to interpolate y at.
    yout : array_like, optional
        Positions to interpolate x at.
    x_transform, y_transform : {"none", "logarithmic", "log", "normal_z"}
        "log" is an accepted alias for "logarithmic" (both parse to the same value); "logarithmic"
        is the spelling used in this package's own examples.
    order_x, order_y : {"ascending", "descending", "none"}
    strict_x, strict_y : bool
        Require x/y to strictly increase/decrease (per ``order_x``/``order_y``) between
        consecutive ordinates. Default ``True``.

    Returns
    -------
    numpy.ndarray
        Same length as whichever of ``xout``/``yout`` was supplied.

    Examples
    --------
    >>> from corehydropy import curve_interpolate
    >>> curve_interpolate([50, 100, 150, 200, 250], [100, 200, 300, 400, 500], xout=75)
    array([150.])
    """
    xa, ya = _check_pair(x, y)
    if (xout is None) == (yout is None):
        raise ValueError("exactly one of `xout` or `yout` must be supplied")
    _check_paired_transform(x_transform, "x_transform")
    _check_paired_transform(y_transform, "y_transform")
    options = _paired_data_shape_opts(strict_x, strict_y, order_x, order_y)
    options["x_transform"] = x_transform
    options["y_transform"] = y_transform
    if xout is not None:
        xouta = np.atleast_1d(np.asarray(xout, dtype=float))
        r = _toolbox_run("paired_data", "interpolate_y", [xa, ya, xouta], options)
    else:
        youta = np.atleast_1d(np.asarray(yout, dtype=float))
        r = _toolbox_run("paired_data", "interpolate_x", [xa, ya, youta], options)
    return np.asarray(r["values"], dtype=float)


def curve_area(
    x,
    y,
    under: str = "y",
    order_x: str = "ascending",
    order_y: str = "ascending",
    strict_x: bool = True,
    strict_y: bool = True,
) -> float:
    """Area under a paired x-y curve.

    Mirrors the C# ``OrderedPairedData.TrapezoidalAreaUnderY``/``TrapezoidalAreaUnderX``: the
    trapezoidal-rule area between the curve and the x-axis (``under="y"``) or the y-axis
    (``under="x"``). Requires x (for ``under="y"``) or y (for ``under="x"``) to be sorted
    ascending or descending -- ``order_x``/``order_y="none"`` raises the same error the C#
    method does.

    Parameters
    ----------
    x, y : array_like
        Equal-length curve ordinates, at least two elements.
    under : {"y", "x"}
        ``"y"`` (default) is the area against the x-axis; ``"x"`` is the area against the y-axis.
    order_x, order_y : {"ascending", "descending", "none"}
    strict_x, strict_y : bool

    Returns
    -------
    float

    Examples
    --------
    >>> from corehydropy import curve_area
    >>> curve_area([1, 2, 3, 4], [1, 4, 9, 16])
    21.5
    """
    xa, ya = _check_pair(x, y)
    if under not in ("y", "x"):
        raise ValueError(f"unknown under '{under}'; expected one of y, x")
    options = _paired_data_shape_opts(strict_x, strict_y, order_x, order_y)
    method = "area_under_y" if under == "y" else "area_under_x"
    r = _toolbox_run("paired_data", method, [xa, ya], options)
    return float(r["values"][0])


def curve_simplify(
    x,
    y,
    method: str = "rdp",
    tolerance: float | None = None,
    num_to_keep: int | None = None,
    look_ahead: int | None = None,
    order_x: str = "ascending",
    order_y: str = "ascending",
    strict_x: bool = True,
    strict_y: bool = True,
) -> np.ndarray:
    """Simplify a paired x-y curve.

    Mirrors the C# ``OrderedPairedData``'s three curve-simplification algorithms:
    Douglas-Peucker (``method="rdp"``, needs ``tolerance``), Visvalingam-Whyatt
    (``method="visvalingam"``, needs ``num_to_keep``), and Lang (``method="lang"``, needs
    ``tolerance`` and ``look_ahead``). NOTE: unlike ``rdp``/``visvalingam``, which always keep
    the curve's first and last point, ``lang`` does not force-keep the trailing point -- a real,
    verified-against-the-real-C#-library upstream behavior (see ``ordered_paired_data.hpp``'s
    sixth transcription note), not a port bug.

    Parameters
    ----------
    x, y : array_like
        Equal-length curve ordinates, at least two elements.
    method : {"rdp", "visvalingam", "lang"}
    tolerance : float, optional
        Perpendicular-distance tolerance; required for ``method`` ``"rdp"`` or ``"lang"``.
    num_to_keep : int, optional
        Number of points to keep; required for ``method="visvalingam"``, and must be at least 2
        (the algorithm always keeps the curve's first and last point, and needs at least 3
        ordinates to triangulate at every intermediate step).
    look_ahead : int, optional
        The Lang algorithm's look-ahead window; required for ``method="lang"``.
    order_x, order_y : {"ascending", "descending", "none"}
    strict_x, strict_y : bool

    Returns
    -------
    numpy.ndarray
        An ``(n, 2)`` array with columns ``[x, y]``.

    Examples
    --------
    >>> from corehydropy import curve_simplify
    >>> x = [0, 1.57, 3.14, 4.71, 6.28]
    >>> y = [0, 1, 0, -1, 0]
    >>> curve_simplify(x, y, method="rdp", tolerance=0.01, strict_y=False, order_y="none")
    array([[ 0.  ,  0.  ],
           [ 1.57,  1.  ],
           [ 4.71, -1.  ],
           [ 6.28,  0.  ]])
    """
    xa, ya = _check_pair(x, y)
    if method not in ("rdp", "visvalingam", "lang"):
        raise ValueError(f'`method` must be one of "rdp", "visvalingam", "lang"; got "{method}"')
    options = _paired_data_shape_opts(strict_x, strict_y, order_x, order_y)
    options["algorithm"] = method
    if method == "rdp":
        if tolerance is None:
            raise ValueError('`tolerance` is required when method="rdp"')
        options["tolerance"] = float(tolerance)
    elif method == "visvalingam":
        if num_to_keep is None:
            raise ValueError('`num_to_keep` is required when method="visvalingam"')
        # Visvalingam-Whyatt always keeps the curve's first and last point, so it needs at
        # least 3 ordinates to triangulate; below that the C++ layer throws (matching the C#
        # List<T> indexer's ArgumentOutOfRangeException) rather than reading out of bounds.
        # Checked here too so the caller gets one clear message instead of the runner's
        # internal one.
        if num_to_keep < 2:
            raise ValueError("`num_to_keep` must be a single integer of at least 2")
        options["num_to_keep"] = int(num_to_keep)
    else:
        if tolerance is None or look_ahead is None:
            raise ValueError('`tolerance` and `look_ahead` are both required when method="lang"')
        options["tolerance"] = float(tolerance)
        options["look_ahead"] = int(look_ahead)
    r = _toolbox_run("paired_data", "simplify", [xa, ya], options)
    values = np.asarray(r["values"], dtype=float)
    return values.reshape(r["dims"][0], r["dims"][1])


def uncertain_curve_sample(
    x,
    distributions,
    probability: float | None = None,
    order_x: str = "ascending",
    order_y: str = "ascending",
    strict_x: bool = True,
    strict_y: bool = True,
) -> np.ndarray:
    """Sample an uncertain paired curve.

    Mirrors the C# ``UncertainOrderedPairedData.CurveSample()``/``CurveSample(double)``:
    collapses a curve whose Y-coordinate is a whole distribution at each x down to a plain x-y
    curve, either at the distributions' means (``probability=None``, the default) or at a
    shared quantile (``probability`` in ``[0, 1]``).

    Parameters
    ----------
    x : array_like
        x positions, at least one element.
    distributions : Distribution or list of Distribution
        One distribution per element of ``x``, or a single distribution recycled across every
        ``x``.
    probability : float, optional
        Quantile in ``[0, 1]`` to sample at; ``None`` (default) samples the mean. Values
        outside ``[0, 1]`` are rejected -- the underlying C# ``CurveSample(double)`` silently
        clamps them instead, so this range check is enforced here rather than core-side.
    order_x, order_y : {"ascending", "descending", "none"}
    strict_x, strict_y : bool

    Returns
    -------
    numpy.ndarray
        An ``(n, 2)`` array with columns ``[x, y]``.

    Examples
    --------
    >>> from corehydropy import Distribution, uncertain_curve_sample
    >>> x = [1, 2, 3, 5]
    >>> d = [Distribution("Triangular", [1, 2, 3]), Distribution("Triangular", [2, 4, 5]),
    ...      Distribution("Triangular", [6, 8, 12]), Distribution("Triangular", [13, 19, 20])]
    >>> uncertain_curve_sample(x, d, probability=0.5)
    array([[ 1.        ,  2.        ],
           [ 2.        ,  3.732051  ],
           [ 3.        ,  8.535898  ],
           [ 5.        , 17.58258   ]])
    """
    xa = np.atleast_1d(np.asarray(x, dtype=float))
    if xa.size == 0:
        raise ValueError("`x` must be a non-empty array")
    dists = _paired_data_distributions(distributions, xa)
    options = _paired_data_shape_opts(strict_x, strict_y, order_x, order_y)
    options["distributions"] = [_as_spec(d) for d in dists]
    if probability is not None:
        # C3 (P4 whole-branch review): the core clamp (uncertain_ordered_paired_data.hpp) is a
        # faithful port of C# CurveSample(double), which silently clamps an out-of-range
        # quantile instead of raising -- `probability=50` silently returned the 100% quantile.
        # That core behavior is untouched; this host-layer range check is what actually rejects
        # the mistake.
        if not (0.0 <= probability <= 1.0):
            raise ValueError("`probability` must be a single number in [0, 1]")
        options["probability"] = float(probability)
    r = _toolbox_run("paired_data", "curve_sample", [xa], options)
    values = np.asarray(r["values"], dtype=float)
    return values.reshape(r["dims"][0], r["dims"][1])


def tabular_function(
    x,
    distributions,
    at,
    inverse: bool = False,
    x_transform: str = "none",
    y_transform: str = "none",
    confidence_level: float | None = None,
    allow_negative_y_values: bool = False,
) -> np.ndarray:
    """Evaluate a tabular function.

    Mirrors the C# ``TabularFunction``: builds an uncertain paired curve from ``x`` and
    ``distributions``, samples it once (the mean curve, or ``confidence_level`` if given), and
    evaluates ``Function()``/``InverseFunction()`` at ``at``. Unlike :func:`curve_interpolate`
    and friends, the underlying curve's shape contract is not configurable here --
    ``TabularFunction`` is always built strict, ascending on both axes, matching every use in
    the ported C# test suite.

    Parameters
    ----------
    x : array_like
        The curve's x positions, at least one element.
    distributions : Distribution or list of Distribution
        One distribution per element of ``x``, or a single distribution recycled across every
        ``x``.
    at : array_like
        Points to evaluate at (or, when ``inverse=True``, points to evaluate the inverse
        function at).
    inverse : bool, default False
        If ``True``, evaluates ``InverseFunction()`` instead of ``Function()``.
    x_transform, y_transform : {"none", "logarithmic", "log", "normal_z"}
        "log" is an accepted alias for "logarithmic" (both parse to the same value); "logarithmic"
        is the spelling used in this package's own examples.
    confidence_level : float, optional
        Quantile in ``[0, 1]`` to sample the curve at; ``None`` (default) samples the mean.
    allow_negative_y_values : bool, default False
        Allow a negative or NaN result to pass through unmodified, rather than clamping it to 0.
        Default ``False`` (clamp), matching every use in the ported C# test suite -- the C#
        class default is ``True``.

    Returns
    -------
    numpy.ndarray
        Same length as ``at``.

    Examples
    --------
    >>> from corehydropy import Distribution, tabular_function
    >>> x = [50, 100, 150, 200, 250]
    >>> d = [Distribution("Deterministic", [v]) for v in [100, 200, 300, 400, 500]]
    >>> tabular_function(x, d, at=50, x_transform="logarithmic")
    array([100.])
    """
    xa = np.atleast_1d(np.asarray(x, dtype=float))
    if xa.size == 0:
        raise ValueError("`x` must be a non-empty array")
    dists = _paired_data_distributions(distributions, xa)
    ata = np.atleast_1d(np.asarray(at, dtype=float))
    if ata.size == 0:
        raise ValueError("`at` must be a non-empty array")
    _check_paired_transform(x_transform, "x_transform")
    _check_paired_transform(y_transform, "y_transform")
    options: dict = {
        "x": xa.tolist(),
        "distributions": [_as_spec(d) for d in dists],
        "x_transform": x_transform,
        "y_transform": y_transform,
        "allow_negative_y_values": bool(allow_negative_y_values),
    }
    if confidence_level is not None:
        options["confidence_level"] = float(confidence_level)
    method = "tabular_inverse" if inverse else "tabular"
    r = _toolbox_run("functions", method, [ata], options)
    return np.asarray(r["values"], dtype=float)
