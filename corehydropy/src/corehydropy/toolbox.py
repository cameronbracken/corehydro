"""The Numerics toolbox surface. Every verb serializes its options to the
``toolbox_runner.hpp`` grammar and runs one method through ``_core.toolbox_run``; bulk data goes
across as numeric vectors, not JSON. Mirrors ``corehydror``'s ``R/toolbox.R`` verb for verb.
"""

from __future__ import annotations

import json

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
    pa = np.asarray(probs, dtype=float).ravel()
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
    >>> round(autocorrelation(x, max_lag=5)["value"][0], 12)
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
