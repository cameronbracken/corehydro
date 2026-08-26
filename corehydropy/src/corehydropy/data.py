"""Censored observation frames and the diagnostics computed off them.

Mirrors :mod:`corehydror`'s ``analysis_data()`` exactly. An :class:`AnalysisData` is a plain
spec object that serializes to the ``data_frame`` grammar
``core/include/corehydro/models/model_spec.hpp`` parses, so no C++ object lifetime leaks into
Python and a frame can be pickled, printed, and compared across languages.
"""

from __future__ import annotations

import json

import numpy as np

from . import _core
from .distributions import Distribution

__all__ = [
    "AnalysisData",
    "analysis_data",
    "analysis_data_summary",
    "threshold_diagnostics",
    "analysis_data_hypothesis_test",
    "analysis_data_statistics",
]


def _values(x, what):
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"`{what}` is empty")
    return [float(v) for v in arr]


def _indexes(index, n, what):
    """0-based indexes, generated sequentially when not supplied."""
    if index is None:
        return list(range(n))
    idx = [int(v) for v in np.asarray(index).ravel()]
    if len(idx) != n:
        raise ValueError(f"`{what}` index has length {len(idx)} but {n} values")
    if any(v < 0 for v in idx):
        raise ValueError(f"`{what}` indexes must be non-negative integers")
    return idx


def _column(mapping, name, what):
    if not isinstance(mapping, dict) or name not in mapping:
        raise ValueError(f"`{what}` requires a `{name}` column")
    return mapping[name]


def _normalize_exact(exact):
    if exact is None:
        return None
    if not isinstance(exact, dict):
        exact = {"value": exact}
    value = _values(_column(exact, "value", "exact"), "exact")
    index = _indexes(exact.get("index"), len(value), "exact")
    flag = exact.get("is_low_outlier")
    if flag is None:
        flag = [False] * len(value)
    else:
        flag = [bool(v) for v in np.broadcast_to(np.asarray(flag), (len(value),))]
    return [
        {"index": index[i], "value": value[i], "is_low_outlier": flag[i]}
        for i in range(len(value))
    ]


def _normalize_interval(interval):
    if interval is None:
        return None
    lower = _values(_column(interval, "lower", "interval"), "interval")
    value = _values(_column(interval, "value", "interval"), "interval")
    upper = _values(_column(interval, "upper", "interval"), "interval")
    if not (len(lower) == len(value) == len(upper)):
        raise ValueError("`interval` lower, value, and upper must be the same length")
    if any(lo > v for lo, v in zip(lower, value)) or any(v > hi for v, hi in zip(value, upper)):
        raise ValueError("`interval` requires lower <= value <= upper for every observation")
    index = _indexes(interval.get("index"), len(value), "interval")
    return [
        {"index": index[i], "lower": lower[i], "value": value[i], "upper": upper[i]}
        for i in range(len(value))
    ]


def _normalize_threshold(threshold):
    if threshold is None:
        return None
    value = _values(_column(threshold, "value", "threshold"), "threshold")
    start = [int(v) for v in np.atleast_1d(_column(threshold, "start_index", "threshold"))]
    end = [int(v) for v in np.atleast_1d(_column(threshold, "end_index", "threshold"))]
    above = [int(v) for v in np.atleast_1d(_column(threshold, "number_above", "threshold"))]
    if not (len(start) == len(end) == len(above) == len(value)):
        raise ValueError("`threshold` columns must all be the same length")
    if any(e < s for s, e in zip(start, end)):
        raise ValueError("`threshold` requires end_index >= start_index for every period")
    return [
        {
            "start_index": start[i],
            "end_index": end[i],
            "value": value[i],
            "number_above": above[i],
        }
        for i in range(len(value))
    ]


def _normalize_uncertain(uncertain):
    if uncertain is None:
        return None
    if isinstance(uncertain, Distribution):
        uncertain = [uncertain]
    index = None
    if isinstance(uncertain, dict):
        index = uncertain.get("index")
        dists = uncertain["distribution"]
    else:
        dists = uncertain
    dists = list(dists)
    if not all(isinstance(d, Distribution) for d in dists):
        raise TypeError(
            "`uncertain` must be a sequence of Distribution objects, or a mapping with a "
            "`distribution` key holding one"
        )
    idx = _indexes(index, len(dists), "uncertain")
    return [
        {
            "index": idx[i],
            "distribution": {"family": dists[i].family, "parameters": list(dists[i].params)},
        }
        for i in range(len(dists))
    ]


class AnalysisData:
    """Censored observation data for an analysis.

    Beyond a plain systematic record (``exact``), a frame can carry the censored observation
    types Bulletin 17C flood-frequency work depends on: historical or paleoflood observations
    known only within a range (``interval``), perception thresholds recording that nothing above
    a level occurred over a span of years (``threshold``), and observations whose measurement
    error is itself a distribution (``uncertain``).

    This is the port of the RMC.BestFit ``DataFrame`` class, named ``AnalysisData`` to avoid
    colliding with :class:`pandas.DataFrame`.

    Indexes are 0-based, matching the C# library and the shared C++ core, and are generated
    sequentially when not supplied. An index is a position in the record, so an interval
    observation at index 40 sits after the 40th exact observation in chronological order.

    Parameters
    ----------
    exact : array_like or dict, optional
        Observations, or a mapping with a ``"value"`` key plus optional ``"index"`` and
        ``"is_low_outlier"`` keys.
    interval : dict, optional
        Mapping with ``"lower"``, ``"value"``, and ``"upper"`` keys plus an optional
        ``"index"``, giving observations known only to lie in a range.
    threshold : dict, optional
        Mapping with ``"start_index"``, ``"end_index"``, ``"value"``, and ``"number_above"``
        keys, giving perception thresholds over spans of the record.
    uncertain : sequence of Distribution or dict, optional
        One distribution per observation, or a mapping with ``"index"`` and ``"distribution"``
        keys.
    low_outlier_threshold : float, optional
        Values at or below it are treated as left-censored low outliers.
    mgbt_low_outliers : bool, default False
        Let the Multiple Grubbs-Beck test pick the low outlier threshold from the data.
        Mutually exclusive with ``low_outlier_threshold`` and with explicit
        ``is_low_outlier`` flags.

    Examples
    --------
    >>> peaks = [12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500]
    >>> AnalysisData(peaks)
    <AnalysisData 8 exact>

    Two historical floods known only within a range, and a perception threshold covering the
    40 years before the gauge was installed:

    >>> AnalysisData(
    ...     exact=peaks,
    ...     interval={"index": [8, 9], "lower": [30000, 26000],
    ...               "value": [35000, 29000], "upper": [40000, 32000]},
    ...     threshold={"start_index": 8, "end_index": 47,
    ...                "value": 25000, "number_above": 2},
    ... )
    <AnalysisData 8 exact, 2 interval, 1 threshold>
    """

    def __init__(
        self,
        exact=None,
        interval=None,
        threshold=None,
        uncertain=None,
        low_outlier_threshold=None,
        mgbt_low_outliers: bool = False,
    ) -> None:
        if mgbt_low_outliers and low_outlier_threshold is not None:
            raise ValueError(
                "mgbt_low_outliers=True picks the threshold from the data; do not also "
                "supply low_outlier_threshold"
            )

        spec: dict = {}
        for key, value in (
            ("exact", _normalize_exact(exact)),
            ("interval", _normalize_interval(interval)),
            ("threshold", _normalize_threshold(threshold)),
            ("uncertain", _normalize_uncertain(uncertain)),
        ):
            if value is not None:
                spec[key] = value
        if not spec:
            raise ValueError(
                "AnalysisData needs at least one of exact, interval, threshold, or uncertain"
            )
        if low_outlier_threshold is not None:
            spec["low_outlier_threshold"] = float(low_outlier_threshold)
            # An explicit threshold means "censor everything below this", so ask the core to
            # derive the flags and the low-outlier count from it rather than only recording it.
            spec["threshold_low_outliers"] = True
        if mgbt_low_outliers:
            spec["mgbt_low_outliers"] = True
        self.spec = spec

    def to_json(self) -> str:
        """The frame as the JSON spec the shared C++ core parses."""
        return json.dumps(self.spec)

    def __repr__(self) -> str:
        counts = [
            f"{len(self.spec[k])} {k}"
            for k in ("exact", "interval", "threshold", "uncertain")
            if k in self.spec
        ]
        extra = ""
        if "low_outlier_threshold" in self.spec:
            extra = f", low outlier threshold {self.spec['low_outlier_threshold']:g}"
        if self.spec.get("mgbt_low_outliers"):
            extra = ", MGBT low outliers"
        return f"<AnalysisData {', '.join(counts)}{extra}>"


def analysis_data(*args, **kwargs) -> AnalysisData:
    """Construct an :class:`AnalysisData` frame.

    A function alias for :class:`AnalysisData`, so the Python and R spellings match. See
    :class:`AnalysisData` for the arguments.

    Returns
    -------
    AnalysisData
        The assembled observation frame.
    """
    return AnalysisData(*args, **kwargs)


def _as_data(data) -> AnalysisData:
    if isinstance(data, AnalysisData):
        return data
    return AnalysisData(exact=data)


def analysis_data_summary(data, plotting_parameter: float = 0.0) -> dict:
    """Run the plotting-position and threshold cascade over an observation frame.

    The plotting positions are the Hirsch-Stedinger censored positions, so interval and
    threshold observations shift the positions of the systematic record even though they carry
    no plotting ordinate themselves.

    Parameters
    ----------
    data : AnalysisData or array_like
        The observation frame, or a plain sequence of observations.
    plotting_parameter : float, default 0.0
        The plotting-position parameter in ``[0, 1)``: ``0`` is Weibull, ``0.40`` Cunnane,
        ``0.44`` Gringorten, ``0.50`` Hazen.

    Returns
    -------
    dict
        ``index``, ``value``, ``plotting_position``, and ``is_low_outlier`` are parallel lists
        over the exact series; ``number_of_low_outliers``, ``low_outlier_threshold``,
        ``plotting_parameter``, ``lambda`` (events per index), ``total_record_length``,
        ``zero_value_relative_frequency``, and the four series counts are scalars.

    Examples
    --------
    >>> s = analysis_data_summary([12500, 15300, 8900, 22100])
    >>> s["plotting_position"]
    [0.4, 0.2, 0.6, 0.2]
    """
    plotting_parameter = float(plotting_parameter)
    if not np.isfinite(plotting_parameter) or not 0.0 <= plotting_parameter < 1.0:
        raise ValueError("plotting_parameter must be finite and in [0, 1)")
    return _core.data_frame_summary(_as_data(data).to_json(), plotting_parameter)


def threshold_diagnostics(
    x,
    u_min: float,
    u_max: float,
    n_thresholds: int = 20,
    confidence_level: float = 0.95,
    method: str = "mean_residual_life",
) -> dict:
    """Threshold selection diagnostics for peaks-over-threshold analysis.

    The mean residual life plot is the sample mean of excesses above each candidate threshold,
    which is linear in the threshold once a generalized Pareto model holds. The parameter
    stability plot fits a generalized Pareto distribution at each candidate threshold; the
    modified scale and the shape are approximately constant above the true threshold.

    Candidate thresholds with too few exceedances are dropped (fewer than 5 for mean residual
    life, fewer than 10 for parameter stability), as are thresholds where the fit fails, so the
    returned lists are usually shorter than ``n_thresholds``.

    Parameters
    ----------
    x : array_like
        Observations.
    u_min, u_max : float
        The range of candidate thresholds to scan.
    n_thresholds : int, default 20
        Number of equally spaced candidate thresholds in ``[u_min, u_max]``.
    confidence_level : float, default 0.95
        Confidence level for the interval bands.
    method : {"mean_residual_life", "parameter_stability"}, default "mean_residual_life"
        Which diagnostic to compute.

    Returns
    -------
    dict
        Parallel lists. Both methods return ``threshold`` and ``exceedance_count``;
        ``"mean_residual_life"`` adds ``mean_excess``, ``lower_ci``, and ``upper_ci``, and
        ``"parameter_stability"`` adds ``modified_scale``, ``shape``, and their confidence
        bounds. The columns the chosen method does not populate come back empty.

    References
    ----------
    Coles (2001), *An Introduction to Statistical Modeling of Extreme Values*, Section 4.3;
    Davison and Smith (1990).
    """
    if method not in ("mean_residual_life", "parameter_stability"):
        raise ValueError(
            f"unknown method '{method}'; expected 'mean_residual_life' or "
            "'parameter_stability'"
        )
    return _core.threshold_diagnostics(
        [float(v) for v in np.asarray(x, dtype=float).ravel()],
        method,
        float(u_min),
        float(u_max),
        int(n_thresholds),
        float(confidence_level),
    )


def _exact_values(data) -> list:
    """The exact-series values a data-layer facade reads, in series order.

    Both :func:`analysis_data_hypothesis_test` and :func:`analysis_data_statistics` read the
    exact series only, so an :class:`AnalysisData` object contributes just its ``exact`` values
    (ignoring any interval/threshold/uncertain series it also carries); a plain sequence is used
    as-is.
    """
    if isinstance(data, AnalysisData):
        exact = data.spec.get("exact")
        if exact is None:
            raise ValueError(
                "`data` has no exact series; analysis_data_hypothesis_test()/"
                "analysis_data_statistics() read the exact series only"
            )
        return [float(e["value"]) for e in exact]
    return [float(v) for v in np.asarray(data, dtype=float).ravel()]


_HYPOTHESIS_METHODS = (
    "jarque_bera",
    "ljung_box",
    "equal_variance_t",
    "unequal_variance_t",
    "f",
    "linear_trend",
    "wald_wolfowitz",
    "mann_whitney",
    "mann_kendall",
)
_TWO_SAMPLE_METHODS = ("equal_variance_t", "unequal_variance_t", "f", "mann_whitney")


def analysis_data_hypothesis_test(
    data,
    method: str,
    split_index: int | None = None,
    lag_max: int | None = None,
    use_log10: bool = False,
) -> dict:
    """A hypothesis test on an observation record.

    Runs one of the nine ``DataFrame`` hypothesis-test facades over the exact series of an
    :class:`AnalysisData` frame (or a plain sequence). The four two-sample tests split the record
    at ``split_index``, comparing observations with a data index below it against those at or
    above it -- the split is on the record's INDEX, not on array position, so it agrees with the
    split a caller would get by re-running the test on a record whose observations were supplied
    out of order.

    Parameters
    ----------
    data : AnalysisData or array_like
        The observations, or an :class:`AnalysisData` frame (only its exact series is read).
    method : {"jarque_bera", "ljung_box", "equal_variance_t", "unequal_variance_t", "f",
        "linear_trend", "wald_wolfowitz", "mann_whitney", "mann_kendall"}
        Which facade to run: normality, autocorrelation, difference in means (Student's /
        Welch's), difference in variances, trend, runs test for independence, homogeneity /
        jump, and homogeneity / trend, respectively.
    split_index : int, optional
        The record index to split the sample at; required by ``"equal_variance_t"``,
        ``"unequal_variance_t"``, ``"f"``, and ``"mann_whitney"``, ignored otherwise.
    lag_max : int, optional
        The maximum lag for ``"ljung_box"``; the default (``None``) uses the library's own
        default rule. Ignored by every other method.
    use_log10 : bool, default False
        Test the log10-transformed values instead of the real-space values.

    Returns
    -------
    dict
        A single-key dict ``{method: p_value}``, the 2-sided p-value.

    Examples
    --------
    >>> peaks = [122, 244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205]
    >>> analysis_data_hypothesis_test(peaks, "mann_kendall")
    {'mann_kendall': ...}
    """
    if method not in _HYPOTHESIS_METHODS:
        raise ValueError(
            f"unknown method '{method}'; expected one of {', '.join(_HYPOTHESIS_METHODS)}"
        )
    values = _exact_values(data)
    if method in _TWO_SAMPLE_METHODS and split_index is None:
        raise ValueError(
            f'method = "{method}" requires `split_index`, the record index to split the sample at'
        )
    options: dict = {"use_log10": bool(use_log10)}
    if split_index is not None:
        options["index"] = int(split_index)
    if lag_max is not None:
        options["lag_max"] = int(lag_max)
    r = _core.data_frame_run(method, [values], "", json.dumps(options))
    return {method: r["values"][0]}


def analysis_data_statistics(data, all_data: bool = False, standardized: bool = False) -> dict:
    """Summary statistics for an observation record.

    Runs the ``DataFrame`` summary-statistics facades over the exact series of an
    :class:`AnalysisData` frame (or a plain sequence): record length, low-outlier count,
    min/max, the first four product moments in real space and log10 space, and seven
    percentiles.

    By default (``all_data=False``) the statistics are computed directly from the exact series
    (``SummaryStatisticsExactDataOnly``). With ``all_data=True``, they instead come from a
    nonparametric distribution fit through the Hirsch-Stedinger plotting positions of the
    combined exact/interval/uncertain record (``SummaryStatisticsAllData``) -- the same fit
    :func:`analysis_data_summary` uses -- so ``"Record Length"`` counts the full record (censored
    series included) rather than the exact series alone, and the two methods can disagree even on
    a frame with no censored data because their percentile and moment estimators differ. Both
    report ``NaN`` for every value when the exact series has fewer than 10 points.

    Parameters
    ----------
    data : AnalysisData or array_like
        The observations, or an :class:`AnalysisData` frame (only its exact series is read).
    all_data : bool, default False
        Use ``SummaryStatisticsAllData`` (see above) instead of the default exact-series-only
        statistics.
    standardized : bool, default False
        Also run ``SetStandardizedValues()`` (after computing plotting positions) and return the
        exact series' standardized values and standardized log10 values, parallel to
        :func:`analysis_data_summary`'s ``"value"``.

    Returns
    -------
    dict
        ``"value"`` holds the twenty summary statistics, keyed by ``"Record Length"``,
        ``"Events Per Index (λ)"``, ``"Low Outliers"``, ``"Minimum"``, ``"Maximum"``,
        ``"Mean"``, ``"Std Dev"``, ``"Skewness"``, ``"Kurtosis"``, ``"Mean (of log)"``,
        ``"Std Dev (of log)"``, ``"Skewness (of log)"``, ``"Kurtosis (of log)"``, ``"1%"``,
        ``"5%"``, ``"25%"``, ``"50%"``, ``"75%"``, ``"95%"``, ``"99%"``. With
        ``standardized=True``, ``"standardized_value"`` and ``"standardized_log10_value"`` are
        added: lists parallel to the exact series, in series order.

    Examples
    --------
    >>> peaks = [12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600]
    >>> s = analysis_data_statistics(peaks)
    >>> s["value"]["Mean"]
    ...
    """
    values = _exact_values(data)
    method = "summary_all" if all_data else "summary_exact"
    r = _core.data_frame_run(method, [values], "", "{}")
    out: dict = {"value": dict(zip(r["names"], r["values"]))}
    if standardized:
        rs = _core.data_frame_run("standardized", [values], "", "{}")
        flat = rs["values"]
        out["standardized_value"] = flat[0::2]
        out["standardized_log10_value"] = flat[1::2]
    return out
