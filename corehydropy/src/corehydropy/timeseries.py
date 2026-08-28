"""Time series.

The "timeseries" toolbox group (P6): the ported Numerics ``TimeSeries`` container over
``core/include/corehydro/numerics/support/toolbox/timeseries.hpp``.

Mirrors corehydror's own timeseries.R; both packages share every argument name, default and error
message, so a change here is not one-sided. The one deliberate difference in SHAPE is the same one
:class:`~corehydropy.Distribution` already established: Python gets methods on a class, R gets free
functions over a classed list.

Dates cross into the shared core as seconds since 1970-01-01 (see the group header for why not
.NET ticks) and come back as ``numpy.datetime64[s]``.
"""

from __future__ import annotations

import numpy as np

from .toolbox import _toolbox_run

__all__ = ["TimeSeries", "ts_interval_names"]

# The accepted interval names, in the C# enum's own order.
_INTERVALS = (
    "one_minute",
    "five_minute",
    "fifteen_minute",
    "thirty_minute",
    "one_hour",
    "six_hour",
    "twelve_hour",
    "one_day",
    "seven_day",
    "one_month",
    "one_quarter",
    "one_year",
    "irregular",
)

_MONTH_ABB = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def ts_interval_names() -> list[str]:
    """The time intervals a :class:`TimeSeries` can carry, in the library's own order."""
    return list(_INTERVALS)


def _match_token(value, choices, arg: str) -> str:
    """Internal: validate a token against a set, with the message R's `ts_match_token()` raises."""
    value = str(value)
    if value not in choices:
        allowed = ", ".join(f'"{c}"' for c in choices)
        raise ValueError(f'`{arg}` must be one of {allowed}; got "{value}"')
    return value


def _epoch(dates, arg: str = "dates") -> np.ndarray:
    """Internal: coerce anything date-like to epoch seconds (UTC).

    Accepts ``datetime``, ``numpy.datetime64``, a pandas ``DatetimeIndex`` (duck-typed, so pandas
    stays an optional dependency), ISO 8601 strings, and plain floats.
    """
    a = np.asarray(dates)
    if a.dtype.kind == "M":
        return a.astype("datetime64[s]").astype("float64")
    if a.dtype.kind in "OU S".replace(" ", ""):
        try:
            return np.asarray(a, dtype="datetime64[s]").astype("float64")
        except (TypeError, ValueError):
            raise ValueError(f"`{arg}` contains a value that is not a date") from None
    if a.dtype.kind in "if":
        return a.astype("float64").ravel()
    raise ValueError(
        f"`{arg}` must be datetime-like, an ISO 8601 string sequence, or numeric seconds since 1970"
    )


def _epoch_scalar(value, arg: str) -> float:
    """Internal: one date to epoch seconds."""
    return float(_epoch([value], arg)[0])


def _as_datetime64(epoch_seconds) -> np.ndarray:
    """Internal: epoch seconds back to ``datetime64[s]``."""
    return np.asarray(epoch_seconds, dtype="float64").astype("int64").astype("datetime64[s]")


class TimeSeries:
    """A time series: an ordered collection of (date, value) ordinates on a time interval.

    Mirrors the Numerics ``TimeSeries`` container. Every method returns a new object or a plain
    result rather than modifying this one, even where the library's own method mutates in place.

    ``dates`` may be ``datetime`` objects, ``numpy.datetime64``, a pandas ``DatetimeIndex``, ISO
    8601 strings, or numeric seconds since 1970-01-01. Missing observations are ``nan`` in
    ``values`` and stay missing through every method that says so.

    The interval is not merely a label: it is what :meth:`shift` walks with ``by="start"``, what
    :meth:`peaks_over_threshold` measures its independence criterion in, and what
    :meth:`convert_interval` converts from. Use ``"irregular"`` for an event series (annual
    maxima, peaks over threshold) whose spacing is not fixed.

    Parameters
    ----------
    dates : array-like
        The ordinate dates.
    values : array-like of float
        The ordinate values, one per date.
    interval : str
        One of :func:`ts_interval_names`. Default ``"one_day"``.

    Examples
    --------
    >>> import numpy as np
    >>> dates = np.arange("2000-01-01", "2000-01-11", dtype="datetime64[D]")
    >>> ts = TimeSeries(dates, [3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
    >>> len(ts)
    10
    >>> ts.moving_average(period=3).values[:3].round(4)
    array([2.6667, 2.    , 3.3333])
    """

    def __init__(self, dates, values, interval: str = "one_day") -> None:
        epoch = _epoch(dates)
        try:
            vals = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError):
            raise ValueError("`values` must be numeric") from None
        if epoch.size != vals.size:
            raise ValueError(
                f"`dates` and `values` must have the same length; got {epoch.size} and {vals.size}"
            )
        self._epoch = epoch.astype("float64")
        self._values = vals
        self._interval = _match_token(interval, _INTERVALS, "interval")

    # --- Properties -----------------------------------------------------------------------

    @property
    def dates(self) -> np.ndarray:
        """The ordinate dates as ``numpy.datetime64[s]``."""
        return _as_datetime64(self._epoch)

    @property
    def values(self) -> np.ndarray:
        """The ordinate values."""
        return self._values.copy()

    @property
    def interval(self) -> str:
        """The time interval this series is on."""
        return self._interval

    def __len__(self) -> int:
        return int(self._values.size)

    def __repr__(self) -> str:
        n = len(self)
        missing = int(np.count_nonzero(np.isnan(self._values)))
        head = f'<TimeSeries {n} ordinates, interval "{self._interval}"'
        if n == 0:
            return head + ">"
        d = self.dates
        span = f", {d.min()} to {d.max()}"
        miss = f", {missing} missing" if missing else ""
        return head + span + miss + ">"

    def __getitem__(self, key) -> "TimeSeries":
        return TimeSeries(self.dates[key], self._values[key], self._interval)

    def to_frame(self):
        """Return the series as a pandas ``DataFrame`` of ``date`` and ``value``.

        Falls back to a dict of arrays when pandas is not installed -- pandas is optional, not a
        dependency.
        """
        try:
            import pandas as pd
        except ImportError:
            return {"date": self.dates, "value": self.values}
        return pd.DataFrame({"date": self.dates, "value": self.values})

    @classmethod
    def from_frame(cls, frame, date: str = "date", value: str = "value",
                   interval: str = "one_day") -> "TimeSeries":
        """Build a series from a pandas ``DataFrame`` (or any mapping of column name to array)."""
        return cls(frame[date], frame[value], interval)

    # --- Internals ------------------------------------------------------------------------

    def _data(self, extra=None) -> list:
        data = [self._epoch, self._values]
        if extra is not None:
            data.append(np.asarray(extra, dtype=float).ravel())
        return data

    def _run_series(self, method: str, options: dict | None = None, extra=None,
                    interval: str | None = None) -> "TimeSeries":
        opts = {"time_interval": self._interval, **(options or {})}
        r = _toolbox_run("timeseries", method, self._data(extra), opts)
        m = np.asarray(r["values"], dtype=float).reshape(r["dims"][0], r["dims"][1])
        return TimeSeries(_as_datetime64(m[:, 0]), m[:, 1], interval or self._interval)

    def _run_table(self, method: str, options: dict | None = None, extra=None) -> np.ndarray:
        opts = {"time_interval": self._interval, **(options or {})}
        r = _toolbox_run("timeseries", method, self._data(extra), opts)
        return np.asarray(r["values"], dtype=float).reshape(r["dims"][0], r["dims"][1])

    def _run_named(self, method: str, options: dict | None = None, extra=None) -> dict:
        opts = {"time_interval": self._interval, **(options or {})}
        r = _toolbox_run("timeseries", method, self._data(extra), opts)
        return dict(zip(r["names"], (float(v) for v in r["values"])))

    # --- Moving windows -------------------------------------------------------------------

    def moving_average(self, period: int, min_valid_count: int | None = None) -> "TimeSeries":
        """Trailing moving average over the previous ``period`` ordinates.

        The result is shorter than the input by ``period - 1``, and each ordinate carries the date
        of its window's LAST observation.

        ``min_valid_count`` controls what a window holding missing values does. The default --
        ``None``, meaning ``period`` -- propagates strictly: any missing value in the window gives
        a missing result, matching pandas's default ``min_periods``. A smaller value averages the
        observed entries only.
        """
        return self._run_series("moving_average", {
            "period": int(period),
            "min_valid_count": -1 if min_valid_count is None else int(min_valid_count),
        })

    def moving_sum(self, period: int, min_valid_count: int | None = None) -> "TimeSeries":
        """Trailing moving sum over the previous ``period`` ordinates.

        See :meth:`moving_average` for ``min_valid_count``; under a relaxed value the sum is over
        the observed entries only, with no rescaling.
        """
        return self._run_series("moving_sum", {
            "period": int(period),
            "min_valid_count": -1 if min_valid_count is None else int(min_valid_count),
        })

    def cumulative_sum(self) -> "TimeSeries":
        """Running total, treating a missing value as zero while accumulating.

        Following the library, the result carries the DEFAULT ``"one_day"`` interval rather than
        this series'.
        """
        return self._run_series("cumulative_sum", interval="one_day")

    def difference(self, lag: int = 1, differences: int = 1) -> "TimeSeries":
        """Successive differences. The result keeps this series' start date, not a shifted one."""
        return self._run_series("difference",
                                {"lag": int(lag), "differences": int(differences)})

    def standardize(self) -> "TimeSeries":
        """Subtract the mean and divide by the standard deviation of the observed values."""
        return self._run_series("standardize")

    def sort(self, by: str = "time", order: str = "ascending") -> "TimeSeries":
        """Sort by ``"time"`` or ``"value"``, ``"ascending"`` or ``"descending"``."""
        by = _match_token(by, ("time", "value"), "by")
        order = _match_token(order, ("ascending", "descending"), "order")
        return self._run_series("sort", {"by": by, "order": order})

    # --- Value transformations ------------------------------------------------------------

    def transform(self, fun: str, constant: float = 0.0, power: float = 1.0,
                  base: float = 10.0, indexes=None) -> "TimeSeries":
        """Apply one of the library's value transformations, returning a new series.

        ``fun`` is one of ``"add"``, ``"subtract"``, ``"multiply"``, ``"divide"``,
        ``"absolute_value"``, ``"exponentiate"``, ``"logarithm"``, ``"inverse"``.

        Every transformation LEAVES A MISSING VALUE MISSING except ``"logarithm"``, which writes a
        missing value for any non-positive input. ``indexes`` restricts the transformation to the
        given 0-based ordinate positions; ``"logarithm"`` and ``"inverse"`` raise on an
        out-of-range index while the others skip it, because the library does.
        """
        fun = _match_token(fun, ("add", "subtract", "multiply", "divide", "absolute_value",
                                 "exponentiate", "logarithm", "inverse"), "fun")
        return self._run_series(
            "math",
            {"func": fun, "constant": float(constant), "power": float(power), "base": float(base)},
            extra=None if indexes is None else np.asarray(indexes, dtype=float),
        )

    def replace_missing(self, value: float, indexes=None) -> "TimeSeries":
        """Set every missing value to ``value``."""
        return self._run_series(
            "math", {"func": "replace", "value": float(value)},
            extra=None if indexes is None else np.asarray(indexes, dtype=float),
        )

    def interpolate_missing(self, max_missing: int = 1, indexes=None) -> "TimeSeries":
        """Fill runs of missing values no longer than ``max_missing`` by linear interpolation.

        A run reaching the END of the series is EXTRAPOLATED from the two preceding ordinates
        instead. Both paths work in date space, so an irregular spacing is honoured.
        """
        return self._run_series(
            "math", {"func": "interpolate", "max_missing": int(max_missing)},
            extra=None if indexes is None else np.asarray(indexes, dtype=float),
        )

    def fill_missing_dates(self, start, end, value: float = float("nan")) -> "TimeSeries":
        """Insert the ordinates a regular series is missing entirely, over ``[start, end]``.

        ``value`` defaults to ``nan``, inserting the absent ordinates as MISSING -- which is what a
        repair workflow wants: fill the dates first, then decide separately which gaps are short
        enough to interpolate. The core takes a finite fill value, so the nan case fills with a
        placeholder and then marks exactly the inserted dates missing (an inserted date is one the
        input did not have, so this is exact rather than a guess).
        """
        if np.isnan(value):
            out = self._run_series(
                "fill_missing_dates", {"value": 0.0},
                extra=[_epoch_scalar(start, "start"), _epoch_scalar(end, "end")],
            )
            inserted = ~np.isin(out._epoch, self._epoch)
            out._values[inserted] = np.nan
            return out
        return self._run_series(
            "fill_missing_dates", {"value": float(value)},
            extra=[_epoch_scalar(start, "start"), _epoch_scalar(end, "end")],
        )

    # --- Clip, shift, re-interval ---------------------------------------------------------

    def clip(self, start, end) -> "TimeSeries":
        """Keep the ordinates inside ``[start, end]``; both bounds must lie inside the span."""
        return self._run_series(
            "clip", extra=[_epoch_scalar(start, "start"), _epoch_scalar(end, "end")]
        )

    def shift(self, by: str = "day", amount: int = 0, start=None) -> "TimeSeries":
        """Move every date by whole days, months or years, or re-anchor the series.

        With ``by="start"`` the series is re-anchored at ``start`` and the rest re-walked at its
        interval; on an ``"irregular"`` series only the first ordinate moves, since there is no
        interval to walk.
        """
        by = _match_token(by, ("day", "month", "year", "start"), "by")
        if by == "start":
            if start is None:
                raise ValueError('`start` is required when `by = "start"`')
            return self._run_series("shift", {"by": by},
                                    extra=[_epoch_scalar(start, "start")])
        return self._run_series("shift", {"by": by, "amount": int(amount)})

    def convert_interval(self, to: str, average: bool = True) -> "TimeSeries":
        """Resample to another interval.

        Interpolates to a finer interval and block-averages (or block-sums, with
        ``average=False``) to a coarser one. Month, quarter, year and irregular intervals have no
        fixed number of hours, so converting to or from one raises.
        """
        to = _match_token(to, _INTERVALS, "to")
        return self._run_series("convert_interval", {"to_interval": to, "average": bool(average)},
                                interval=to)

    # --- Statistics -----------------------------------------------------------------------

    def statistics(self) -> dict:
        """The library's fifteen-entry summary: length, missing count, extremes, moments,
        percentiles."""
        return self._run_named("summary_statistics")

    def hypothesis_test(self, split_location: int | None = None) -> dict:
        """The container's seven-test battery over the observed values.

        ``split_location`` is a 0-BASED position in the observed values (R's ``ts_hypothesis_test``
        takes the 1-based position, each language following its own convention); ``None``, the
        default, splits the record in half.

        Jarque-Bera for normality, Ljung-Box and Wald-Wolfowitz for independence, Mann-Whitney and
        Mann-Kendall for homogeneity, and a t-test and F-test comparing the two halves of the
        record. This is NOT the ten-test battery
        :func:`~corehydropy.analysis_data_hypothesis_test` runs on a flood-frequency data frame --
        they are different methods with different splits.
        """
        return self._run_named("summary_hypothesis_test", {
            "split_location": -1 if split_location is None else int(split_location)
        })

    def percentiles(self, probabilities=(0.05, 0.25, 0.5, 0.75, 0.95)) -> np.ndarray:
        """Percentiles of the observed values at the requested probabilities."""
        opts = {"time_interval": self._interval}
        r = _toolbox_run("timeseries", "percentiles",
                         self._data(np.asarray(probabilities, dtype=float)), opts)
        return np.asarray(r["values"], dtype=float)

    def duration(self) -> np.ndarray:
        """The duration (percent-of-time exceedance) curve, as an n-by-2 array of percent and
        value."""
        return self._run_table("duration")

    def monthly_statistics(self) -> np.ndarray:
        """Per-calendar-month statistics: 12 rows of minimum, 5%, 25%, 50%, 75%, 95%, maximum,
        mean.

        A month with no observation keeps an all-zero row rather than a missing one, following the
        library. Missing values are filtered out -- unlike :meth:`monthly_percentiles`, which does
        not filter them, so one missing value gives that month missing percentiles.
        """
        return self._run_table("monthly_summary_statistics")

    def monthly_percentiles(self, probabilities=(0.05, 0.25, 0.5, 0.75, 0.95)) -> np.ndarray:
        """Percentiles within each calendar month: 12 rows, one column per probability."""
        return self._run_table("monthly_percentiles",
                               extra=np.asarray(probabilities, dtype=float))

    def monthly_frequency(self) -> dict:
        """The number of ordinates falling in each calendar month, keyed by month abbreviation."""
        opts = {"time_interval": self._interval}
        r = _toolbox_run("timeseries", "monthly_frequency", self._data(), opts)
        return dict(zip(_MONTH_ABB, (float(v) for v in r["values"])))

    # --- Frequency analysis ---------------------------------------------------------------

    def block_series(self, window: str = "water_year", block: str = "maximum",
                     smoothing: str = "none", period: int = 1, start_month: int = 10,
                     end_month: int = 9) -> "TimeSeries":
        """Reduce the series to one value per time block -- annual maxima and their relatives.

        The result is an IRREGULAR series carrying the date of the observation the block function
        selected: the extreme observation's own date for ``"minimum"``/``"maximum"``, the block's
        last date for ``"sum"``/``"average"``.

        A block containing a missing value is dropped by ``"sum"`` and ``"average"`` (the missing
        value propagates) but kept by ``"minimum"`` and ``"maximum"`` (the comparison is false
        against a missing value, so it is simply never selected).

        ``smoothing`` applies BEFORE the block function, which is how an n-day average maximum is
        built: ``block_series(smoothing="moving_average", period=7)`` is the annual maximum 7-day
        mean.
        """
        window = _match_token(window, ("water_year", "calendar_year", "custom_year", "quarter",
                                       "month"), "window")
        block = _match_token(block, ("maximum", "minimum", "average", "sum"), "block")
        smoothing = _match_token(smoothing, ("none", "moving_average", "moving_sum", "difference"),
                                 "smoothing")
        return self._run_series("block_series", {
            "block": window, "block_function": block, "smoothing": smoothing,
            "period": int(period), "start_month": int(start_month), "end_month": int(end_month),
        }, interval="irregular")

    def water_year(self, block: str = "maximum", smoothing: str = "none", period: int = 1,
                   start_month: int = 10) -> "TimeSeries":
        """The annual block series over a water year starting in ``start_month`` (October)."""
        return self.block_series(window="water_year", block=block, smoothing=smoothing,
                                 period=period, start_month=start_month)

    def calendar_year(self, block: str = "maximum", smoothing: str = "none",
                      period: int = 1) -> "TimeSeries":
        """The annual block series over calendar years."""
        return self.block_series(window="calendar_year", block=block, smoothing=smoothing,
                                 period=period)

    def peaks_over_threshold(self, threshold: float, min_steps_between_events: int = 1,
                             smoothing: str = "none", period: int = 1) -> "TimeSeries":
        """Extract independent peak events above ``threshold``.

        Follows the ``clust`` method of the R ``POT`` package: the first exceedance opens a
        cluster, the first value back under the threshold closes it unless the minimum spacing has
        not yet elapsed, and the next exceedance opens the next cluster. Each cluster contributes
        its maximum.
        """
        smoothing = _match_token(smoothing, ("none", "moving_average", "moving_sum", "difference"),
                                 "smoothing")
        return self._run_series("peaks_over_threshold", {
            "threshold": float(threshold),
            "min_steps_between_events": int(min_steps_between_events),
            "smoothing": smoothing, "period": int(period),
        }, interval="irregular")

    # --- Decomposition and resampling -----------------------------------------------------

    def seasonal_decompose(self, period: int) -> dict:
        """Classical additive decomposition into trend, seasonal and residual components.

        The seasonal part is extracted by keeping only the harmonics of the seasonal frequency.
        The trend is a moving average over ``period`` ordinates, so it -- and the residual with it
        -- is undefined for the first ``period - 1`` ordinates, reported as ``nan``. Where all
        three are defined they add back to the original value exactly.

        Returns a dict of ``date``, ``trend``, ``seasonal`` and ``residual`` arrays, one entry per
        input ordinate.
        """
        m = self._run_table("seasonal_decompose", {"period": int(period)})
        return {
            "date": _as_datetime64(m[:, 0]),
            "trend": m[:, 1],
            "seasonal": m[:, 2],
            "residual": m[:, 3],
        }

    def resample_knn(self, time_steps: int, k: int, seed: int = 12345) -> "TimeSeries":
        """The conditional k-nearest-neighbour bootstrap of Lall and Sharma (1996).

        At each step it finds the ``k`` historical observations closest to the current value,
        picks one at random, and advances to whatever historically came NEXT. That conditioning is
        what preserves the lag-1 structure -- returning the neighbour's own value instead would
        collapse the trajectory toward its starting point.

        The draw happens inside the shared C++ core, so a seeded call gives bit-identical results
        in Python and R.

        References
        ----------
        Lall, U. and Sharma, A. (1996). A nearest neighbor bootstrap for resampling hydrologic
        time series. Water Resources Research 32(3), 679-693.
        """
        return self._run_series("resample_knn", {
            "time_steps": int(time_steps), "k": int(k), "seed": int(seed)
        })

    def resample_block_bootstrap(self, time_steps: int, block_size: int,
                                 seed: int = 12345) -> "TimeSeries":
        """A fixed-block bootstrap: contiguous blocks drawn uniformly with replacement.

        Preserves the marginal distribution and the within-block dependence, at the cost of a
        discontinuity at each block boundary.

        References
        ----------
        Kuensch, H.R. (1989). The jackknife and the bootstrap for general stationary
        observations. Annals of Statistics 17(3), 1217-1241.
        """
        return self._run_series("resample_block_bootstrap", {
            "time_steps": int(time_steps), "block_size": int(block_size), "seed": int(seed)
        })
