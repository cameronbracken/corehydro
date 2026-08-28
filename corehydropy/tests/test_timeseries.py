"""The time-series surface (P6).

Mirrors corehydror/tests/testthat/test-timeseries.R assertion for assertion: the same series, the
same expected values, the same error messages. Where a value is a C# test literal the comment
names the upstream test method it came from; the seeded resampling values are the ones
fixtures/timeseries/resampling.json pins against the real library, which is what makes the
Python-vs-R comparison a cross-language check rather than a self-comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from corehydropy import TimeSeries, ts_interval_names

# The twelve-value monthly series most upstream tests build on (Test_MovingAverage etc.).
MONTHLY_VALUES = [22, 16, 33, 5, 12, 36, 48, 10, 18, 15, 22, 13]
MONTHLY_DATES = np.array([f"2023-{m:02d}-01" for m in range(1, 13)], dtype="datetime64[s]")

# The 69-value monthly series the block-series and POT tests build on.
LONG_VALUES = [
    122, 244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331, 225, 174, 98.8, 149,
    238, 262, 132, 235, 216, 240, 230, 192, 195, 172, 173, 172, 153, 142, 317, 161, 201, 204,
    194, 164, 183, 161, 167, 179, 185, 117, 192, 337, 125, 166, 99.1, 202, 230, 158, 262, 154,
    164, 182, 164, 183, 171, 250, 184, 205, 237, 177, 239, 187, 180, 173, 174,
]


def _monthly_dates(n: int, start_year: int = 2023) -> np.ndarray:
    months = [(start_year * 12 + i) for i in range(n)]
    return np.array([f"{m // 12:04d}-{m % 12 + 1:02d}-01" for m in months],
                    dtype="datetime64[s]")


def monthly_ts() -> TimeSeries:
    return TimeSeries(MONTHLY_DATES, MONTHLY_VALUES, "one_month")


def long_ts() -> TimeSeries:
    return TimeSeries(_monthly_dates(len(LONG_VALUES)), LONG_VALUES, "one_month")


def test_accepts_every_date_type_and_round_trips_them():
    values = [1, 2, 3]
    import datetime as dt

    from_datetime64 = TimeSeries(np.arange("2000-01-01", "2000-01-04", dtype="datetime64[D]"),
                                 values)
    from_datetime = TimeSeries([dt.datetime(2000, 1, 1), dt.datetime(2000, 1, 2),
                                dt.datetime(2000, 1, 3)], values)
    from_string = TimeSeries(["2000-01-01", "2000-01-02", "2000-01-03"], values)
    from_numeric = TimeSeries([946684800, 946771200, 946857600], values)

    for ts in (from_datetime64, from_datetime, from_string, from_numeric):
        assert len(ts) == 3
        assert ts.dates.tolist() == np.array(
            ["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[s]"
        ).tolist()
        assert ts.values.tolist() == [1.0, 2.0, 3.0]
        assert ts.interval == "one_day"

    frame = from_datetime64.to_frame()
    assert list(frame.keys() if isinstance(frame, dict) else frame.columns) == ["date", "value"]
    assert len(from_datetime64[1:3]) == 2
    assert "3 ordinates" in repr(from_datetime64)


def test_constructor_and_tokens_reject_bad_input():
    with pytest.raises(ValueError, match="must have the same length; got 3 and 2"):
        TimeSeries(np.arange("2000-01-01", "2000-01-04", dtype="datetime64[D]"), [1, 2])
    with pytest.raises(ValueError, match="`interval` must be one of"):
        TimeSeries(["2000-01-01"], [1], "fortnight")
    with pytest.raises(ValueError, match="contains a value that is not a date"):
        TimeSeries(["not a date"], [1])
    with pytest.raises(ValueError, match="`window` must be one of"):
        monthly_ts().block_series(window="decade")
    with pytest.raises(Exception, match="The period must be less than the length"):
        monthly_ts().moving_average(12)


def test_moving_windows_reproduce_the_csharp_literals():
    # Test_MovingAverage / Test_MovingSum.
    assert monthly_ts().moving_average(5).values == pytest.approx(
        [17.6, 20.4, 26.8, 22.2, 24.8, 25.4, 22.6, 15.6]
    )
    assert monthly_ts().moving_sum(5).values == pytest.approx([88, 102, 134, 111, 124, 127, 113, 78])
    # The window's date is its LAST observation's.
    assert monthly_ts().moving_average(5).dates[0] == np.datetime64("2023-05-01T00:00:00")
    assert monthly_ts().moving_average(5).interval == "one_month"

    gappy = TimeSeries(np.arange("2023-01-01", "2023-01-06", dtype="datetime64[D]"),
                       [1, np.nan, 3, 4, 5])
    assert np.all(np.isnan(gappy.moving_sum(2).values[:2]))
    assert gappy.moving_sum(2, min_valid_count=1).values == pytest.approx([1, 3, 7, 9])


def test_differences_cumulative_sums_and_standardization():
    # Test_Difference: eleven differences, keeping the original start date.
    d = monthly_ts().difference()
    assert len(d) == 11
    assert d.values == pytest.approx(np.diff(MONTHLY_VALUES))
    assert d.dates[0] == np.datetime64("2023-01-01T00:00:00")

    # Test_Cumulative, and the upstream oddity that the result drops the source's interval.
    cs = monthly_ts().cumulative_sum()
    assert cs.values == pytest.approx(np.cumsum(MONTHLY_VALUES))
    assert cs.interval == "one_day"

    s = monthly_ts().standardize()
    assert s.values.mean() == pytest.approx(0, abs=1e-12)
    assert s.values.std(ddof=1) == pytest.approx(1, abs=1e-12)
    with pytest.raises(Exception, match="Standard deviation is zero"):
        TimeSeries(["2023-01-01", "2023-01-02", "2023-01-03"], [2, 2, 2]).standardize()


def test_missing_values_are_replaced_interpolated_and_filled():
    values = [22, 16, 33, 5, 12, 36, 48, 10, 18, 15, np.nan, np.nan]
    ts = TimeSeries(MONTHLY_DATES, values, "one_month")

    assert ts.replace_missing(-1).values[10:].tolist() == [-1.0, -1.0]
    # Test_Missing: the trailing pair is EXTRAPOLATED from the two preceding ordinates.
    assert ts.interpolate_missing(max_missing=2).values[10:] == pytest.approx([11.9, 8.9], abs=1e-6)

    sparse = TimeSeries(["2024-01-01", "2024-01-04"], [1, 4])
    filled = sparse.fill_missing_dates("2024-01-01", "2024-01-05", -9)
    assert len(filled) == 5
    assert filled.values.tolist() == [1.0, -9.0, -9.0, 4.0, -9.0]


def test_transformations_whole_series_or_chosen_ordinates():
    ts = TimeSeries(["2023-01-01", "2023-01-02", "2023-01-03"], [1, 2, 4])
    assert ts.transform("multiply", constant=10).values.tolist() == [10.0, 20.0, 40.0]
    assert ts.transform("logarithm", base=2).values == pytest.approx([0, 1, 2])
    # 0-based positions here, 1-based in R -- each language follows its own convention.
    assert ts.transform("add", constant=5, indexes=[0, 2]).values.tolist() == [6.0, 2.0, 9.0]
    gappy = TimeSeries(["2023-01-01", "2023-01-02", "2023-01-03"], [1, np.nan, 4])
    assert np.isnan(gappy.transform("add", constant=5).values[1])
    with pytest.raises(ValueError, match="`fun` must be one of"):
        ts.transform("sqrt")


def test_clipping_shifting_and_interval_conversion():
    ts = monthly_ts()
    clipped = ts.clip("2023-11-01", "2023-12-01")
    assert len(clipped) == 2
    assert clipped.values.tolist() == [22.0, 13.0]

    shifted = ts.shift(by="month", amount=5)
    assert shifted.dates[0] == np.datetime64("2023-06-01T00:00:00")
    assert shifted.values.tolist() == ts.values.tolist()
    with pytest.raises(ValueError, match="`start` is required"):
        ts.shift(by="start")

    hourly = TimeSeries(np.arange("1973-05-01T00", "1973-05-02T00", dtype="datetime64[h]"),
                        [6.0] * 24, "one_hour")
    daily = hourly.convert_interval("one_day")
    assert daily.interval == "one_day"
    assert daily.values[0] == pytest.approx(6)
    with pytest.raises(Exception, match="cannot be converted"):
        monthly_ts().convert_interval("one_year")


def test_summary_statistics_reproduce_the_csharp_literals():
    # Test_SummaryStats.
    stats = monthly_ts().statistics()
    assert stats["Record Length"] == 12
    assert stats["Minimum"] == 5
    assert stats["Maximum"] == 48
    assert stats["Mean"] == pytest.approx(20.83333333333, abs=1e-10)
    assert stats["Std Dev"] == pytest.approx(12.4011240937215, abs=1e-10)
    assert list(stats)[0] == "Record Length"
    assert list(stats)[14] == "99%"

    assert monthly_ts().percentiles([0.10, 0.90]) == pytest.approx([10.2, 35.7], abs=1e-10)

    # Test_Stats: the duration curve pairs Weibull plotting positions with descending values.
    dur = monthly_ts().duration()
    assert dur.shape == (12, 2)
    assert dur[:, 1].tolist() == sorted(MONTHLY_VALUES, reverse=True)
    assert dur[0, 0] == pytest.approx(7.69230769230769, abs=1e-10)

    h = long_ts().hypothesis_test()
    assert len(h) == 7
    assert list(h)[0] == "Jarque-Bera test for normality"
    assert all(0 <= v <= 1 for v in h.values())


def test_monthly_statistics_reproduce_the_csharp_literals():
    values = [22, 16, 33, 5, 12, 36, 48, 10, 18, 15, 22, 13,
              38, 17, 3, 6, 27, 11, 2, 41, 44, 37, 50, 8]
    ts = TimeSeries(_monthly_dates(24), values, "one_month")
    # Test_MonthlyStats, January row.
    assert ts.monthly_statistics()[0].tolist() == pytest.approx(
        [22, 22.8, 26.0, 30.0, 34.0, 37.2, 38, 30], abs=1e-10
    )
    assert ts.monthly_percentiles([0.10, 0.20])[0] == pytest.approx([23.6, 25.2], abs=1e-10)
    # Test_MonthlyFrequency: two observations in every month.
    assert list(ts.monthly_frequency().values()) == [2.0] * 12
    assert list(ts.monthly_frequency())[0] == "Jan"


def test_block_series_reproduce_the_csharp_literals():
    ts = long_ts()
    # Test_CalendarYearSeries / Test_WaterYearSeries / Test_CustomYearSeries.
    assert ts.calendar_year().values.tolist() == [263, 331, 317, 337, 262, 239]
    assert ts.water_year().values.tolist() == [263, 331, 317, 204, 337, 250]
    assert ts.block_series(window="custom_year", start_month=10,
                           end_month=3).values.tolist() == [244, 331, 240, 204, 337, 250]
    # Test_QuarterlySeries (first four quarters).
    assert ts.block_series(window="quarter").values[:4].tolist() == [244, 229, 263, 205]
    assert ts.water_year().interval == "irregular"

    values = list(range(1, 13)) + [1, np.nan] + list(range(3, 13))
    gappy = TimeSeries(_monthly_dates(24), values, "one_month")
    assert len(gappy.calendar_year(block="sum")) == 1
    assert gappy.calendar_year(block="sum").values[0] == 78
    assert len(gappy.calendar_year(block="maximum")) == 2


def test_peaks_over_threshold_reproduce_the_csharp_literals():
    ts = long_ts()
    # Test_PeaksOverThreshold, all four threshold/spacing pairs.
    assert ts.peaks_over_threshold(100, 2).values.tolist() == [331, 337, 262]
    assert ts.peaks_over_threshold(90, 1).values.tolist() == [337]
    assert ts.peaks_over_threshold(150, 5).values.tolist() == [331, 240, 317, 337]
    assert ts.peaks_over_threshold(200, 2).values.tolist() == [263, 331, 262, 317, 337, 250]


def test_seasonal_decomposition_is_additive_where_the_trend_is_defined():
    n = 48
    dates = _monthly_dates(n, start_year=2000)
    i = np.arange(1, n + 1)
    value = 100 + 0.5 * i + 10 * np.sin(2 * np.pi * i / 12)
    d = TimeSeries(dates, value, "one_month").seasonal_decompose(period=12)

    assert set(d) == {"date", "trend", "seasonal", "residual"}
    assert len(d["trend"]) == n
    assert np.all(np.isnan(d["trend"][:11]))
    defined = ~np.isnan(d["trend"])
    assert (d["trend"][defined] + d["seasonal"][defined] + d["residual"][defined]) == pytest.approx(
        value[defined], abs=1e-6
    )
    with pytest.raises(Exception, match="at least 2 complete periods"):
        TimeSeries(dates[:10], value[:10], "one_month").seasonal_decompose(12)


def test_seeded_resampling_reproduces_the_pinned_values():
    values = LONG_VALUES[:30]
    ts = TimeSeries(np.arange("2000-01-01", "2000-01-31", dtype="datetime64[D]"), values)

    # These twelve values are fixtures/timeseries/resampling.json's, pinned against the real C#
    # library; corehydror's test-timeseries.R asserts the SAME twelve, which is what makes the
    # cross-language guarantee testable from either side.
    knn = ts.resample_knn(time_steps=12, k=5, seed=42)
    assert knn.values.tolist() == [214, 263, 132, 238, 230, 192, 195, 263, 146, 331, 214, 174]
    assert len(knn) == 12
    assert knn.values.tolist() == ts.resample_knn(12, 5, 42).values.tolist()

    block = ts.resample_block_bootstrap(time_steps=12, block_size=4, seed=42)
    assert block.values.tolist() == [212, 263, 146, 183, 240, 230, 192, 195, 263, 146, 183, 161]
    assert block.dates[0] == np.datetime64("2000-01-01T00:00:00")

    with pytest.raises(Exception, match="Need at least 11 points"):
        TimeSeries(np.arange("2000-01-01", "2000-01-06", dtype="datetime64[D]"),
                   [1, 2, 3, 4, 5]).resample_knn(5, 2)


def test_interval_names_are_the_librarys_in_its_own_order():
    assert ts_interval_names()[0] == "one_minute"
    assert ts_interval_names()[7] == "one_day"
    assert len(ts_interval_names()) == 13
