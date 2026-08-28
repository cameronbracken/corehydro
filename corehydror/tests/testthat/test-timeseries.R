# The time-series surface (P6). Mirrors corehydropy/tests/test_timeseries.py assertion for
# assertion: the same series, the same expected values, the same error messages. Where a value is
# a C# test literal the comment names the upstream test method it came from; the seeded resampling
# values are the ones fixtures/timeseries/resampling.json pins against the real library, which is
# what makes the R-vs-Python comparison a cross-language check rather than a self-comparison.

# The twelve-value monthly series most upstream tests build on (Test_MovingAverage etc.).
monthly_values <- c(22, 16, 33, 5, 12, 36, 48, 10, 18, 15, 22, 13)
monthly_dates <- seq(as.Date("2023-01-01"), by = "month", length.out = 12)
monthly_ts <- function() time_series(monthly_dates, monthly_values, "one_month")

# The 69-value monthly series the block-series and POT tests build on.
long_values <- c(
  122, 244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331, 225, 174, 98.8, 149,
  238, 262, 132, 235, 216, 240, 230, 192, 195, 172, 173, 172, 153, 142, 317, 161, 201, 204,
  194, 164, 183, 161, 167, 179, 185, 117, 192, 337, 125, 166, 99.1, 202, 230, 158, 262, 154,
  164, 182, 164, 183, 171, 250, 184, 205, 237, 177, 239, 187, 180, 173, 174
)
long_ts <- function() {
  time_series(seq(as.Date("2023-01-01"), by = "month", length.out = length(long_values)),
              long_values, "one_month")
}

test_that("a time series accepts every date type and round-trips them", {
  values <- c(1, 2, 3)
  from_date <- time_series(as.Date("2000-01-01") + 0:2, values)
  from_posix <- time_series(as.POSIXct("2000-01-01", tz = "UTC") + (0:2) * 86400, values)
  from_string <- time_series(c("2000-01-01", "2000-01-02", "2000-01-03"), values)
  from_numeric <- time_series(946684800 + (0:2) * 86400, values)

  for (ts in list(from_date, from_posix, from_string, from_numeric)) {
    expect_s3_class(ts, "corehydro_ts")
    expect_equal(length(ts), 3L)
    expect_equal(as.double(ts$dates), c(946684800, 946771200, 946857600))
    expect_equal(ts$values, values)
    expect_equal(ts$interval, "one_day")
  }

  df <- as.data.frame(from_date)
  expect_equal(names(df), c("date", "value"))
  expect_equal(nrow(df), 3L)
  expect_equal(length(from_date[2:3]), 2L)
})

test_that("the constructor and the token arguments reject bad input", {
  expect_error(time_series(as.Date("2000-01-01") + 0:2, c(1, 2)),
               "must have the same length; got 3 and 2")
  expect_error(time_series(as.Date("2000-01-01") + 0:2, c(1, 2, 3), "fortnight"),
               "`interval` must be one of")
  expect_error(time_series(c("not a date"), 1), "contains a value that is not a date")
  expect_error(ts_block_series(monthly_ts(), window = "decade"), "should be one of")
  expect_error(ts_moving_average(monthly_ts(), 12),
               "The period must be less than the length of the time-series")
  expect_error(ts_moving_average(42, 3), "must be a time series created by time_series")
})

test_that("moving windows reproduce the C# literals", {
  # Test_MovingAverage / Test_MovingSum.
  expect_equal(ts_moving_average(monthly_ts(), 5)$values,
               c(17.6, 20.4, 26.8, 22.2, 24.8, 25.4, 22.6, 15.6))
  expect_equal(ts_moving_sum(monthly_ts(), 5)$values, c(88, 102, 134, 111, 124, 127, 113, 78))
  # The window's date is its LAST observation's.
  expect_equal(as.Date(ts_moving_average(monthly_ts(), 5)$dates[1]), as.Date("2023-05-01"))
  expect_equal(ts_moving_average(monthly_ts(), 5)$interval, "one_month")

  # A missing value propagates strictly by default and is skipped with min_valid_count.
  gappy <- time_series(as.Date("2023-01-01") + 0:4, c(1, NA, 3, 4, 5))
  expect_true(all(is.na(ts_moving_sum(gappy, 2)$values[1:2])))
  expect_equal(ts_moving_sum(gappy, 2, min_valid_count = 1)$values, c(1, 3, 7, 9))
})

test_that("differences, cumulative sums and standardization behave as documented", {
  # Test_Difference: eleven differences, keeping the original start date.
  d <- ts_difference(monthly_ts())
  expect_equal(length(d), 11L)
  expect_equal(d$values, diff(monthly_values))
  expect_equal(as.Date(d$dates[1]), as.Date("2023-01-01"))

  # Test_Cumulative, and the upstream oddity that the result drops the source's interval.
  cs <- ts_cumulative_sum(monthly_ts())
  expect_equal(cs$values, cumsum(monthly_values))
  expect_equal(cs$interval, "one_day")

  s <- ts_standardize(monthly_ts())
  expect_equal(mean(s$values), 0, tolerance = 1e-12)
  expect_equal(stats::sd(s$values), 1, tolerance = 1e-12)
  expect_error(ts_standardize(time_series(as.Date("2023-01-01") + 0:2, c(2, 2, 2))),
               "Standard deviation is zero")
})

test_that("missing values are replaced, interpolated and filled", {
  values <- c(22, 16, 33, 5, 12, 36, 48, 10, 18, 15, NA, NA)
  ts <- time_series(monthly_dates, values, "one_month")

  expect_equal(ts_replace_missing(ts, -1)$values[11:12], c(-1, -1))
  # Test_Missing: the trailing pair is EXTRAPOLATED from the two preceding ordinates.
  expect_equal(ts_interpolate_missing(ts, max_missing = 2)$values[11:12], c(11.9, 8.9),
               tolerance = 1e-6)

  sparse <- time_series(as.Date(c("2024-01-01", "2024-01-04")), c(1, 4))
  filled <- ts_fill_missing_dates(sparse, as.Date("2024-01-01"), as.Date("2024-01-05"), -9)
  expect_equal(length(filled), 5L)
  expect_equal(filled$values, c(1, -9, -9, 4, -9))
  # The default inserts the absent ordinates as MISSING rather than as a value.
  default_filled <- ts_fill_missing_dates(sparse, as.Date("2024-01-01"), as.Date("2024-01-05"))
  expect_equal(default_filled$values, c(1, NA, NA, 4, NA))
})

test_that("transformations apply to the whole series or to chosen ordinates", {
  ts <- time_series(as.Date("2023-01-01") + 0:2, c(1, 2, 4))
  expect_equal(ts_transform(ts, "multiply", constant = 10)$values, c(10, 20, 40))
  expect_equal(ts_transform(ts, "logarithm", base = 2)$values, c(0, 1, 2))
  expect_equal(ts_transform(ts, "add", constant = 5, indexes = c(1, 3))$values, c(6, 2, 9))
  # A missing value stays missing.
  gappy <- time_series(as.Date("2023-01-01") + 0:2, c(1, NA, 4))
  expect_true(is.na(ts_transform(gappy, "add", constant = 5)$values[2]))
  expect_error(ts_transform(ts, "sqrt"), "`fun` must be one of")
})

test_that("clipping, shifting and interval conversion work", {
  ts <- monthly_ts()
  clipped <- ts_clip(ts, as.Date("2023-11-01"), as.Date("2023-12-01"))
  expect_equal(length(clipped), 2L)
  expect_equal(clipped$values, c(22, 13))

  shifted <- ts_shift(ts, by = "month", amount = 5)
  expect_equal(as.Date(shifted$dates[1]), as.Date("2023-06-01"))
  expect_equal(shifted$values, ts$values)
  expect_error(ts_shift(ts, by = "start"), "`start` is required")

  hourly <- time_series(as.POSIXct("1973-05-01", tz = "UTC") + (0:23) * 3600, rep(6, 24),
                        "one_hour")
  daily <- ts_convert_interval(hourly, "one_day")
  expect_equal(daily$interval, "one_day")
  expect_equal(daily$values[1], 6)
  expect_error(ts_convert_interval(monthly_ts(), "one_year"), "cannot be converted")
})

test_that("summary statistics reproduce the C# literals", {
  # Test_SummaryStats.
  stats <- ts_statistics(monthly_ts())
  expect_equal(unname(stats["Record Length"]), 12)
  expect_equal(unname(stats["Minimum"]), 5)
  expect_equal(unname(stats["Maximum"]), 48)
  expect_equal(unname(stats["Mean"]), 20.83333333333, tolerance = 1e-10)
  expect_equal(unname(stats["Std Dev"]), 12.4011240937215, tolerance = 1e-10)
  expect_equal(names(stats)[1], "Record Length")
  expect_equal(names(stats)[15], "99%")

  expect_equal(unname(ts_percentiles(monthly_ts(), c(0.10, 0.90))), c(10.2, 35.7),
               tolerance = 1e-10)

  # Test_Stats: the duration curve pairs Weibull plotting positions with descending values.
  dur <- ts_duration(monthly_ts())
  expect_equal(nrow(dur), 12L)
  expect_equal(dur$value, sort(monthly_values, decreasing = TRUE))
  expect_equal(dur$percent[1], 7.69230769230769, tolerance = 1e-10)

  h <- ts_hypothesis_test(long_ts())
  expect_equal(length(h), 7L)
  expect_equal(names(h)[1], "Jarque-Bera test for normality")
  expect_true(all(h >= 0 & h <= 1))
})

test_that("monthly statistics reproduce the C# literals", {
  values <- c(22, 16, 33, 5, 12, 36, 48, 10, 18, 15, 22, 13,
              38, 17, 3, 6, 27, 11, 2, 41, 44, 37, 50, 8)
  ts <- time_series(seq(as.Date("2023-01-01"), by = "month", length.out = 24), values,
                    "one_month")
  # Test_MonthlyStats, January row.
  expect_equal(unname(ts_monthly_statistics(ts)[1, ]),
               c(22, 22.8, 26.0, 30.0, 34.0, 37.2, 38, 30), tolerance = 1e-10)
  expect_equal(rownames(ts_monthly_statistics(ts))[1], "Jan")
  expect_equal(colnames(ts_monthly_statistics(ts))[1], "minimum")
  expect_equal(unname(ts_monthly_percentiles(ts, c(0.10, 0.20))[1, ]), c(23.6, 25.2),
               tolerance = 1e-10)
  # Test_MonthlyFrequency: two observations in every month.
  expect_equal(unname(ts_monthly_frequency(ts)), rep(2, 12))
})

test_that("block series reproduce the C# literals", {
  ts <- long_ts()
  # Test_CalendarYearSeries / Test_WaterYearSeries / Test_CustomYearSeries.
  expect_equal(ts_calendar_year(ts)$values, c(263, 331, 317, 337, 262, 239))
  expect_equal(ts_water_year(ts)$values, c(263, 331, 317, 204, 337, 250))
  expect_equal(ts_block_series(ts, window = "custom_year", start_month = 10, end_month = 3)$values,
               c(244, 331, 240, 204, 337, 250))
  # Test_QuarterlySeries (first four quarters).
  expect_equal(head(ts_block_series(ts, window = "quarter")$values, 4), c(244, 229, 263, 205))
  expect_equal(ts_water_year(ts)$interval, "irregular")

  # A block holding a missing value is dropped by sum but kept by maximum.
  values <- c(1:12, 1, NA, 3:12)
  gappy <- time_series(seq(as.Date("2023-01-01"), by = "month", length.out = 24), values,
                       "one_month")
  expect_equal(length(ts_calendar_year(gappy, block = "sum")), 1L)
  expect_equal(ts_calendar_year(gappy, block = "sum")$values, 78)
  expect_equal(length(ts_calendar_year(gappy, block = "maximum")), 2L)
})

test_that("peaks over threshold reproduce the C# literals", {
  ts <- long_ts()
  # Test_PeaksOverThreshold, all four threshold/spacing pairs.
  expect_equal(ts_peaks_over_threshold(ts, 100, 2)$values, c(331, 337, 262))
  expect_equal(ts_peaks_over_threshold(ts, 90, 1)$values, 337)
  expect_equal(ts_peaks_over_threshold(ts, 150, 5)$values, c(331, 240, 317, 337))
  expect_equal(ts_peaks_over_threshold(ts, 200, 2)$values, c(263, 331, 262, 317, 337, 250))
})

test_that("seasonal decomposition is additive where the trend is defined", {
  n <- 48
  dates <- seq(as.Date("2000-01-01"), by = "month", length.out = n)
  value <- 100 + 0.5 * seq_len(n) + 10 * sin(2 * pi * seq_len(n) / 12)
  d <- ts_seasonal_decompose(time_series(dates, value, "one_month"), period = 12)

  expect_equal(nrow(d), n)
  expect_equal(names(d), c("date", "trend", "seasonal", "residual"))
  expect_true(all(is.na(d$trend[1:11])))
  defined <- !is.na(d$trend)
  expect_equal(d$trend[defined] + d$seasonal[defined] + d$residual[defined], value[defined],
               tolerance = 1e-6)
  expect_error(ts_seasonal_decompose(time_series(dates[1:10], value[1:10], "one_month"), 12),
               "at least 2 complete periods")
})

test_that("seeded resampling reproduces the pinned values and is deterministic", {
  values <- long_values[1:30]
  ts <- time_series(as.Date("2000-01-01") + 0:29, values)

  # These twelve values are fixtures/timeseries/resampling.json's, pinned against the real C#
  # library; corehydropy's test_timeseries.py asserts the SAME twelve, which is what makes the
  # cross-language guarantee testable from either side.
  knn <- ts_resample_knn(ts, time_steps = 12, k = 5, seed = 42)
  expect_equal(knn$values, c(214, 263, 132, 238, 230, 192, 195, 263, 146, 331, 214, 174))
  expect_equal(length(knn), 12L)
  expect_equal(knn$values, ts_resample_knn(ts, 12, 5, 42)$values)

  block <- ts_resample_block_bootstrap(ts, time_steps = 12, block_size = 4, seed = 42)
  expect_equal(block$values, c(212, 263, 146, 183, 240, 230, 192, 195, 263, 146, 183, 161))
  expect_equal(as.Date(block$dates[1]), as.Date("2000-01-01"))

  expect_error(ts_resample_knn(time_series(as.Date("2000-01-01") + 0:4, 1:5), 5, 2),
               "Need at least 11 points")
})

test_that("autocorrelation accepts a time series and gets the C# values", {
  ts <- time_series(as.Date("2000-01-01") + 0:19,
                    c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4))
  a <- autocorrelation(ts, max_lag = 5)
  # Driven against the real Numerics library: Autocorrelation.Function's TimeSeries and
  # IList<double> overloads return identical values, which is why only the latter is ported (see
  # autocorrelation.hpp's header).
  expect_equal(a$value[1:3], c(1, 0.17306026705160593, 0.007975460122699398), tolerance = 1e-12)
  expect_identical(a$value, autocorrelation(ts$values, max_lag = 5)$value)
})

test_that("interval names are the library's, in its own order", {
  expect_equal(ts_interval_names()[1], "one_minute")
  expect_equal(ts_interval_names()[8], "one_day")
  expect_equal(length(ts_interval_names()), 13L)
})
