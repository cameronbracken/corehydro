# The "timeseries" toolbox group (P6): the ported Numerics `TimeSeries` container over
# core/include/corehydro/numerics/support/toolbox/timeseries.hpp. Its own file rather than an
# addition to R/toolbox.R for the same reason ml.R and regression.R are separate: these verbs
# share an input object and a set of helpers no other toolbox verb uses.
#
# Mirrors corehydropy's own timeseries module; both packages share every argument name, default
# and error message, so a change here is not one-sided. The one deliberate difference in SHAPE is
# the same one `distribution()` / `Distribution` already established: R gets free functions over a
# classed list, Python gets methods on a class.
#
# Dates cross into the shared core as seconds since 1970-01-01 (see the group header for why not
# ticks). R's POSIXct IS epoch seconds, so the conversion is exact and free.

# Internal: coerce anything date-like to epoch seconds (UTC).
ts_epoch <- function(x, arg = "dates") {
  if (inherits(x, "POSIXct")) {
    return(as.double(x))
  }
  if (inherits(x, "Date")) {
    return(as.double(x) * 86400)
  }
  if (is.character(x)) {
    # as.POSIXct() ERRORS on an unparseable string rather than returning NA, so the failure is
    # caught and re-raised with the message corehydropy raises for the same input.
    parsed <- tryCatch(
      as.POSIXct(x, tz = "UTC", tryFormats = c(
        "%Y-%m-%dT%H:%M:%OS", "%Y-%m-%d %H:%M:%OS", "%Y-%m-%d %H:%M", "%Y-%m-%d"
      )),
      error = function(e) NA
    )
    if (anyNA(parsed)) {
      stop(sprintf("`%s` contains a value that is not a date", arg), call. = FALSE)
    }
    return(as.double(parsed))
  }
  if (is.numeric(x)) {
    return(as.double(x))
  }
  stop(sprintf(
    "`%s` must be POSIXct, Date, an ISO 8601 character vector, or numeric seconds since 1970",
    arg
  ), call. = FALSE)
}

# Internal: epoch seconds back to POSIXct in UTC.
ts_posix <- function(x) as.POSIXct(x, origin = "1970-01-01", tz = "UTC")

# The accepted interval names, in the C# enum's own order.
ts_interval_names_vec <- c(
  "one_minute", "five_minute", "fifteen_minute", "thirty_minute", "one_hour", "six_hour",
  "twelve_hour", "one_day", "seven_day", "one_month", "one_quarter", "one_year", "irregular"
)

# Internal: validate a token against a set, with the same message Python raises.
ts_match_token <- function(value, choices, arg) {
  value <- as.character(value)[1]
  if (!value %in% choices) {
    stop(sprintf(
      "`%s` must be one of %s; got \"%s\"",
      arg, paste0("\"", choices, "\"", collapse = ", "), value
    ), call. = FALSE)
  }
  value
}

#' The time intervals a time series can carry
#'
#' @return a character vector of the accepted `interval` values, in the library's own order.
#' @examples
#' ts_interval_names()
#' @export
ts_interval_names <- function() ts_interval_names_vec

#' Create a time series
#'
#' Builds the object every `ts_*()` verb takes: an ordered collection of (date, value) ordinates
#' on a stated time interval, mirroring the Numerics `TimeSeries` container.
#'
#' @details
#' `dates` may be `POSIXct`, `Date`, ISO 8601 character strings, or numeric seconds since
#' 1970-01-01; they are carried as UTC `POSIXct` and returned that way. Missing observations are
#' `NA` (or `NaN`) in `values` and stay missing through every verb that says so.
#'
#' The interval is not merely a label: it is what `ts_shift(by = "start")` walks, what
#' `ts_peaks_over_threshold()` measures its independence criterion in, and what
#' `ts_convert_interval()` converts from. `"irregular"` is the right choice for an event series
#' (annual maxima, peaks over threshold) whose spacing is not fixed.
#'
#' @param dates the ordinate dates.
#' @param values the ordinate values, one per date.
#' @param interval one of [ts_interval_names()]. Default `"one_day"`.
#' @return a `corehydro_ts` object.
#' @seealso [ts_block_series()] for annual maxima, [ts_peaks_over_threshold()] for a POT series,
#'   [ts_statistics()] for a summary.
#' @examples
#' ts <- time_series(as.Date("2000-01-01") + 0:9, c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3))
#' ts
#' @export
time_series <- function(dates, values, interval = "one_day") {
  epoch <- ts_epoch(dates)
  if (!is.numeric(values)) {
    stop("`values` must be numeric", call. = FALSE)
  }
  if (length(epoch) != length(values)) {
    stop(sprintf(
      "`dates` and `values` must have the same length; got %d and %d",
      length(epoch), length(values)
    ), call. = FALSE)
  }
  structure(
    list(
      dates = ts_posix(epoch),
      values = as.double(values),
      interval = ts_match_token(interval, ts_interval_names_vec, "interval")
    ),
    class = "corehydro_ts"
  )
}

#' @export
length.corehydro_ts <- function(x) length(x$values)

#' @export
print.corehydro_ts <- function(x, ...) {
  n <- length(x$values)
  missing <- sum(is.na(x$values) | is.nan(x$values))
  cat(sprintf("<corehydro_ts> %d ordinates, interval \"%s\"\n", n, x$interval))
  if (n > 0) {
    cat(sprintf(
      "  %s to %s%s\n",
      format(min(x$dates)), format(max(x$dates)),
      if (missing > 0) sprintf(", %d missing", missing) else ""
    ))
    head_n <- min(n, 5L)
    for (i in seq_len(head_n)) {
      cat(sprintf("  %s  %s\n", format(x$dates[i]), format(x$values[i])))
    }
    if (n > head_n) cat(sprintf("  ... %d more\n", n - head_n))
  }
  invisible(x)
}

#' @export
as.data.frame.corehydro_ts <- function(x, ...) {
  data.frame(date = x$dates, value = x$values)
}

#' @export
`[.corehydro_ts` <- function(x, i) {
  time_series(x$dates[i], x$values[i], x$interval)
}

# Internal: the (data, options) pair every verb sends to the runner.
ts_check <- function(ts) {
  if (!inherits(ts, "corehydro_ts")) {
    stop("`ts` must be a time series created by time_series()", call. = FALSE)
  }
  invisible(NULL)
}

ts_data <- function(ts, extra = NULL) {
  out <- list(as.double(ts$dates), ts$values)
  if (!is.null(extra)) out <- c(out, list(as.double(extra)))
  out
}

# Internal: run a verb that returns a series, and rebuild the object around it.
ts_run_series <- function(ts, method, options = list(), extra = NULL, interval = ts$interval) {
  r <- toolbox_run("timeseries", method, ts_data(ts, extra),
                   c(list(time_interval = ts$interval), options))
  m <- matrix(r$values, nrow = r$dims[1], ncol = r$dims[2], byrow = TRUE)
  time_series(ts_posix(m[, 1]), m[, 2], interval)
}

# Internal: run a verb that returns a table, as a matrix with column names.
ts_run_table <- function(ts, method, options = list(), extra = NULL, colnames = NULL) {
  r <- toolbox_run("timeseries", method, ts_data(ts, extra),
                   c(list(time_interval = ts$interval), options))
  m <- matrix(r$values, nrow = r$dims[1], ncol = r$dims[2], byrow = TRUE)
  if (!is.null(colnames)) base::colnames(m) <- colnames
  m
}

#' Moving average and moving sum
#'
#' Trailing windows over the previous `period` ordinates, mirroring the C# `MovingAverage` and
#' `MovingSum`. The result is shorter than the input by `period - 1`, and each ordinate carries
#' the date of its window's LAST observation.
#'
#' @details
#' `min_valid_count` controls what a window holding missing values does. The default -- `NULL`,
#' meaning `period` -- propagates strictly: any missing value in the window gives a missing
#' result, which matches `pandas`'s default `min_periods`. A smaller value averages (or sums) the
#' observed entries only; no rescaling is applied to a sum.
#'
#' @param ts a `corehydro_ts`.
#' @param period the window length, which must be shorter than the series.
#' @param min_valid_count the minimum number of observed values a window needs. Default `NULL`
#'   (strict).
#' @return a `corehydro_ts`.
#' @examples
#' ts <- time_series(as.Date("2000-01-01") + 0:9, c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3))
#' ts_moving_average(ts, period = 3)
#' @export
ts_moving_average <- function(ts, period, min_valid_count = NULL) {
  ts_check(ts)
  ts_run_series(ts, "moving_average", list(
    period = as.integer(period),
    min_valid_count = as.integer(if (is.null(min_valid_count)) -1L else min_valid_count)
  ))
}

#' @rdname ts_moving_average
#' @export
ts_moving_sum <- function(ts, period, min_valid_count = NULL) {
  ts_check(ts)
  ts_run_series(ts, "moving_sum", list(
    period = as.integer(period),
    min_valid_count = as.integer(if (is.null(min_valid_count)) -1L else min_valid_count)
  ))
}

#' Cumulative sum, successive differences and standardization
#'
#' Three whole-series transformations, each returning a new series.
#'
#' @details
#' `ts_cumulative_sum()` treats a missing value as zero while accumulating but keeps its ordinate,
#' and -- following the library -- the result carries the DEFAULT `"one_day"` interval rather than
#' the source's. `ts_difference()` keeps the source's start date rather than shifting forward by
#' the lag. `ts_standardize()` subtracts the mean and divides by the standard deviation, both
#' computed over the observed values, and errors when the spread is zero.
#'
#' @param ts a `corehydro_ts`.
#' @param lag the gap between the elements being subtracted. Default 1.
#' @param differences how many times to difference. Default 1.
#' @return a `corehydro_ts`.
#' @examples
#' ts <- time_series(as.Date("2000-01-01") + 0:5, c(3, 1, 4, 1, 5, 9))
#' ts_difference(ts)
#' @export
ts_cumulative_sum <- function(ts) {
  ts_check(ts)
  ts_run_series(ts, "cumulative_sum", interval = "one_day")
}

#' @rdname ts_cumulative_sum
#' @export
ts_difference <- function(ts, lag = 1, differences = 1) {
  ts_check(ts)
  ts_run_series(ts, "difference",
                list(lag = as.integer(lag), differences = as.integer(differences)))
}

#' @rdname ts_cumulative_sum
#' @export
ts_standardize <- function(ts) {
  ts_check(ts)
  ts_run_series(ts, "standardize")
}

#' Sort a time series
#'
#' @param ts a `corehydro_ts`.
#' @param by `"time"` (the default) or `"value"`.
#' @param order `"ascending"` (the default) or `"descending"`.
#' @return a `corehydro_ts`.
#' @examples
#' ts <- time_series(as.Date("2000-01-01") + 0:4, c(3, 1, 4, 1, 5))
#' ts_sort(ts, by = "value")
#' @export
ts_sort <- function(ts, by = c("time", "value"), order = c("ascending", "descending")) {
  ts_check(ts)
  by <- ts_match_token(match.arg(by), c("time", "value"), "by")
  order <- ts_match_token(match.arg(order), c("ascending", "descending"), "order")
  ts_run_series(ts, "sort", list(by = by, order = order))
}

#' Transform the values of a time series
#'
#' The library's in-place value transformations, exposed as pure verbs: each returns a new series
#' rather than modifying `ts`.
#'
#' @details
#' Every transformation LEAVES A MISSING VALUE MISSING except two, which follow the library:
#' `"logarithm"` writes a missing value for any non-positive input, and `ts_standardize()`
#' propagates missing values through the subtraction.
#'
#' `indexes` restricts the transformation to the given 1-based ordinate positions. Two
#' transformations behave differently from their siblings when an index is out of range, because
#' the library does: `"logarithm"` and `"inverse"` raise, while the others skip it.
#'
#' @param ts a `corehydro_ts`.
#' @param fun one of `"add"`, `"subtract"`, `"multiply"`, `"divide"`, `"absolute_value"`,
#'   `"exponentiate"`, `"logarithm"`, `"inverse"`.
#' @param constant the operand for add / subtract / multiply / divide.
#' @param power the exponent for `"exponentiate"`. Default 1.
#' @param base the logarithm base. Default 10.
#' @param indexes optional 1-BASED ordinate positions to restrict the transformation to
#'   (Python's `transform()` takes 0-based positions).
#' @return a `corehydro_ts`.
#' @examples
#' ts <- time_series(as.Date("2000-01-01") + 0:4, c(3, 1, 4, 1, 5))
#' ts_transform(ts, "multiply", constant = 10)
#' @export
ts_transform <- function(ts, fun, constant = 0, power = 1, base = 10, indexes = NULL) {
  ts_check(ts)
  fun <- ts_match_token(fun, c("add", "subtract", "multiply", "divide", "absolute_value",
                               "exponentiate", "logarithm", "inverse"), "fun")
  ts_run_series(ts, "math",
                list(func = fun, constant = as.double(constant), power = as.double(power),
                     base = as.double(base)),
                extra = if (is.null(indexes)) NULL else as.integer(indexes) - 1L)
}

#' Handle missing values
#'
#' @details
#' `ts_replace_missing()` sets every missing value to `value`. `ts_interpolate_missing()` fills a
#' run of missing values by linear interpolation between its neighbours, but only when the run is
#' no longer than `max_missing`; a run reaching the END of the series is EXTRAPOLATED from the two
#' preceding ordinates instead. Both interpolate in date space, so an irregular spacing is
#' honoured. `ts_fill_missing_dates()` is the other half of the problem: it inserts the ordinates a
#' regular series is missing entirely, over the requested date range.
#'
#' @param ts a `corehydro_ts`.
#' @param value the replacement (or fill) value. `ts_fill_missing_dates()` defaults to `NA`,
#'   inserting the absent ordinates as missing rather than as zeros.
#' @param max_missing the longest run of missing values to interpolate across.
#' @param indexes optional 1-based ordinate positions to restrict the operation to.
#' @param start,end the date range to fill over.
#' @return a `corehydro_ts`.
#' @examples
#' ts <- time_series(as.Date("2000-01-01") + 0:4, c(3, NA, 4, 1, 5))
#' ts_interpolate_missing(ts, max_missing = 1)
#' @export
ts_replace_missing <- function(ts, value, indexes = NULL) {
  ts_check(ts)
  ts_run_series(ts, "math", list(func = "replace", value = as.double(value)),
                extra = if (is.null(indexes)) NULL else as.integer(indexes) - 1L)
}

#' @rdname ts_replace_missing
#' @export
ts_interpolate_missing <- function(ts, max_missing = 1, indexes = NULL) {
  ts_check(ts)
  ts_run_series(ts, "math",
                list(func = "interpolate", max_missing = as.integer(max_missing)),
                extra = if (is.null(indexes)) NULL else as.integer(indexes) - 1L)
}

#' @rdname ts_replace_missing
#' @export
ts_fill_missing_dates <- function(ts, start, end, value = NA) {
  ts_check(ts)
  # `NA` -- the default -- inserts the absent ordinates as MISSING, which is what a repair
  # workflow wants: fill the dates first, then decide separately which gaps are short enough to
  # interpolate. The core takes a finite fill value, so the NA case fills with a placeholder and
  # then marks exactly the inserted dates missing (an inserted date is one the input did not
  # have, so this is exact rather than a guess).
  if (is.na(value)) {
    out <- ts_run_series(ts, "fill_missing_dates", list(value = 0),
                         extra = c(ts_epoch(start, "start"), ts_epoch(end, "end")))
    out$values[!(as.double(out$dates) %in% as.double(ts$dates))] <- NA_real_
    return(out)
  }
  ts_run_series(ts, "fill_missing_dates", list(value = as.double(value)),
                extra = c(ts_epoch(start, "start"), ts_epoch(end, "end")))
}

#' Clip, shift and re-interval a time series
#'
#' @details
#' `ts_clip()` keeps the ordinates inside `[start, end]`; both bounds must lie inside the series'
#' own span. `ts_shift()` moves every date by a whole number of days, months or years, or -- with
#' `by = "start"` -- re-anchors the series at a new start date and re-walks the rest at its
#' interval (on an `"irregular"` series only the first ordinate moves, since there is no interval
#' to walk). `ts_convert_interval()` resamples to another interval, interpolating to a finer one
#' and block-averaging (or block-summing, with `average = FALSE`) to a coarser one; month,
#' quarter, year and irregular intervals have no fixed number of hours, so converting to or from
#' one raises.
#'
#' @param ts a `corehydro_ts`.
#' @param start,end the new bounds, for `ts_clip()`.
#' @param by `"day"`, `"month"`, `"year"` or `"start"`.
#' @param amount the number of days, months or years to shift by.
#' @param to the target interval, one of [ts_interval_names()].
#' @param average average across each block (the default) rather than summing.
#' @return a `corehydro_ts`.
#' @examples
#' ts <- time_series(as.Date("2000-01-01") + 0:9, 1:10)
#' ts_clip(ts, as.Date("2000-01-03"), as.Date("2000-01-06"))
#' @export
ts_clip <- function(ts, start, end) {
  ts_check(ts)
  ts_run_series(ts, "clip", extra = c(ts_epoch(start, "start"), ts_epoch(end, "end")))
}

#' @rdname ts_clip
#' @export
ts_shift <- function(ts, by = c("day", "month", "year", "start"), amount = 0, start = NULL) {
  ts_check(ts)
  by <- ts_match_token(match.arg(by), c("day", "month", "year", "start"), "by")
  if (by == "start") {
    if (is.null(start)) {
      stop("`start` is required when `by = \"start\"`", call. = FALSE)
    }
    return(ts_run_series(ts, "shift", list(by = by), extra = ts_epoch(start, "start")))
  }
  ts_run_series(ts, "shift", list(by = by, amount = as.integer(amount)))
}

#' @rdname ts_clip
#' @export
ts_convert_interval <- function(ts, to, average = TRUE) {
  ts_check(ts)
  to <- ts_match_token(to, ts_interval_names_vec, "to")
  ts_run_series(ts, "convert_interval",
                list(to_interval = to, average = isTRUE(average)), interval = to)
}

#' Summary statistics of a time series
#'
#' @details
#' `ts_statistics()` returns the library's fifteen-entry summary (record length, missing count,
#' minimum, maximum, the four product moments and seven percentiles). `ts_hypothesis_test()`
#' returns the seven-test battery the container carries: Jarque-Bera for normality, Ljung-Box and
#' Wald-Wolfowitz for independence, Mann-Whitney and Mann-Kendall for homogeneity, and a t-test and
#' F-test comparing the two halves of the record. Note this is NOT the ten-test battery
#' [analysis_data_hypothesis_test()] runs on a flood-frequency data frame -- they are different
#' methods with different splits.
#'
#' @param ts a `corehydro_ts`.
#' @param split_location the 1-BASED position in the observed values to split at (Python's
#'   `hypothesis_test()` takes the 0-based position, each language following its own
#'   convention). `NULL`, the default, splits the record in half.
#' @param probabilities the probabilities to report, in `[0, 1]`.
#' @return a named numeric vector, or for `ts_duration()` a data frame of `percent` and `value`.
#' @examples
#' ts <- time_series(as.Date("2000-01-01") + 0:11, c(22, 16, 33, 5, 12, 36, 48, 10, 18, 15, 22, 13))
#' ts_statistics(ts)
#' @export
ts_statistics <- function(ts) {
  ts_check(ts)
  r <- toolbox_run("timeseries", "summary_statistics", ts_data(ts),
                   list(time_interval = ts$interval))
  stats::setNames(r$values, r$names)
}

#' @rdname ts_statistics
#' @export
ts_hypothesis_test <- function(ts, split_location = NULL) {
  ts_check(ts)
  r <- toolbox_run("timeseries", "summary_hypothesis_test", ts_data(ts), list(
    time_interval = ts$interval,
    split_location = as.integer(if (is.null(split_location)) -1L else split_location - 1L)
  ))
  stats::setNames(r$values, r$names)
}

#' @rdname ts_statistics
#' @export
ts_percentiles <- function(ts, probabilities = c(0.05, 0.25, 0.5, 0.75, 0.95)) {
  ts_check(ts)
  r <- toolbox_run("timeseries", "percentiles", ts_data(ts, as.double(probabilities)),
                   list(time_interval = ts$interval))
  stats::setNames(r$values, format(probabilities))
}

#' @rdname ts_statistics
#' @export
ts_duration <- function(ts) {
  ts_check(ts)
  m <- ts_run_table(ts, "duration", colnames = c("percent", "value"))
  as.data.frame(m)
}

#' Monthly statistics
#'
#' Statistics computed within each calendar month across the whole record, mirroring the C#
#' `MonthlySummaryStatistics`, `MonthlyPercentiles` and `MonthlyFrequency`.
#'
#' @details
#' A month with no observation keeps an all-zero row rather than a missing one, following the
#' library. The two functions differ in how they treat missing values, also following the library:
#' `ts_monthly_statistics()` filters them out, while `ts_monthly_percentiles()` does not, so one
#' missing value gives that month missing percentiles.
#'
#' @param ts a `corehydro_ts`.
#' @param probabilities the probabilities to report, in `[0, 1]`.
#' @return a 12-row matrix (one row per calendar month), or a length-12 vector for
#'   `ts_monthly_frequency()`.
#' @examples
#' ts <- time_series(seq(as.Date("2000-01-01"), by = "month", length.out = 24), 1:24)
#' ts_monthly_statistics(ts)[1:3, ]
#' @export
ts_monthly_statistics <- function(ts) {
  ts_check(ts)
  m <- ts_run_table(ts, "monthly_summary_statistics",
                    colnames = c("minimum", "p05", "p25", "p50", "p75", "p95", "maximum", "mean"))
  rownames(m) <- month.abb
  m
}

#' @rdname ts_monthly_statistics
#' @export
ts_monthly_percentiles <- function(ts, probabilities = c(0.05, 0.25, 0.5, 0.75, 0.95)) {
  ts_check(ts)
  m <- ts_run_table(ts, "monthly_percentiles", extra = as.double(probabilities),
                    colnames = format(probabilities))
  rownames(m) <- month.abb
  m
}

#' @rdname ts_monthly_statistics
#' @export
ts_monthly_frequency <- function(ts) {
  ts_check(ts)
  r <- toolbox_run("timeseries", "monthly_frequency", ts_data(ts),
                   list(time_interval = ts$interval))
  stats::setNames(r$values, month.abb)
}

#' Block series: annual maxima and other block extremes
#'
#' Reduces a series to one value per time block -- the classical first step of a flood-frequency
#' analysis. `ts_water_year()` and `ts_calendar_year()` are the two common cases by name.
#'
#' @details
#' The result is an IRREGULAR series carrying the date of the observation the block function
#' selected: for `"minimum"` and `"maximum"` that is the extreme observation's own date, and for
#' `"sum"` and `"average"` it is the block's last date.
#'
#' A block whose values include a missing one is dropped by `"sum"` and `"average"` (the missing
#' value propagates) but kept by `"minimum"` and `"maximum"` (the comparison is false against a
#' missing value, so it is simply never selected). That is the library's behaviour and it is worth
#' knowing before reading an annual maximum series built from a gappy record.
#'
#' `smoothing` applies BEFORE the block function, which is how an n-day average maximum is built:
#' `ts_water_year(ts, smoothing = "moving_average", period = 7)` is the annual maximum 7-day mean.
#'
#' @param ts a `corehydro_ts`.
#' @param window `"water_year"`, `"calendar_year"`, `"custom_year"`, `"quarter"` or `"month"`.
#' @param block the block function: `"maximum"` (the default), `"minimum"`, `"average"` or
#'   `"sum"`.
#' @param smoothing `"none"` (the default), `"moving_average"`, `"moving_sum"` or `"difference"`.
#' @param period the smoothing window (or the difference lag). Default 1, which is a no-op for the
#'   two moving windows.
#' @param start_month the month a water year or custom year begins. Default 10 (October).
#' @param end_month the month a custom year ends. Default 9 (September).
#' @return a `corehydro_ts` on the `"irregular"` interval.
#' @seealso [ts_peaks_over_threshold()] for the partial-duration alternative, [fit_mle()] for
#'   fitting the result.
#' @examples
#' dates <- seq(as.Date("2000-01-01"), by = "month", length.out = 36)
#' ts <- time_series(dates, c(5, 9, 3, 7, 2, 8, 4, 6, 1, 5, 7, 3,
#'                            6, 2, 8, 4, 9, 3, 7, 5, 2, 8, 4, 6,
#'                            3, 7, 5, 9, 2, 6, 4, 8, 1, 5, 3, 7))
#' ts_water_year(ts)
#' @export
ts_block_series <- function(ts,
                            window = c("water_year", "calendar_year", "custom_year", "quarter",
                                       "month"),
                            block = c("maximum", "minimum", "average", "sum"),
                            smoothing = c("none", "moving_average", "moving_sum", "difference"),
                            period = 1, start_month = 10, end_month = 9) {
  ts_check(ts)
  window <- ts_match_token(match.arg(window),
                           c("water_year", "calendar_year", "custom_year", "quarter", "month"),
                           "window")
  block <- ts_match_token(match.arg(block), c("maximum", "minimum", "average", "sum"), "block")
  smoothing <- ts_match_token(match.arg(smoothing),
                              c("none", "moving_average", "moving_sum", "difference"), "smoothing")
  ts_run_series(ts, "block_series", list(
    block = window, block_function = block, smoothing = smoothing,
    period = as.integer(period), start_month = as.integer(start_month),
    end_month = as.integer(end_month)
  ), interval = "irregular")
}

#' @rdname ts_block_series
#' @export
ts_water_year <- function(ts, block = "maximum", smoothing = "none", period = 1,
                          start_month = 10) {
  ts_block_series(ts, window = "water_year", block = block, smoothing = smoothing,
                  period = period, start_month = start_month)
}

#' @rdname ts_block_series
#' @export
ts_calendar_year <- function(ts, block = "maximum", smoothing = "none", period = 1) {
  ts_block_series(ts, window = "calendar_year", block = block, smoothing = smoothing,
                  period = period)
}

#' Peaks over a threshold
#'
#' Extracts independent peak events above `threshold`, following the `clust` method of the R
#' `POT` package: the first exceedance opens a cluster, the first value back under the threshold
#' closes it unless the minimum spacing has not yet elapsed, and the next exceedance opens the
#' next cluster. Each cluster contributes its maximum.
#'
#' @param ts a `corehydro_ts`.
#' @param threshold the exceedance threshold.
#' @param min_steps_between_events the minimum number of time steps between independent events.
#'   Default 1.
#' @param smoothing `"none"` (the default), `"moving_average"`, `"moving_sum"` or `"difference"`,
#'   applied before the extraction.
#' @param period the smoothing window. Default 1.
#' @return a `corehydro_ts` on the `"irregular"` interval, one ordinate per event.
#' @seealso [ts_block_series()] for the block-maxima alternative.
#' @examples
#' ts <- time_series(as.Date("2000-01-01") + 0:9, c(1, 8, 2, 1, 9, 7, 1, 2, 12, 1))
#' ts_peaks_over_threshold(ts, threshold = 5, min_steps_between_events = 2)
#' @export
ts_peaks_over_threshold <- function(ts, threshold, min_steps_between_events = 1,
                                    smoothing = c("none", "moving_average", "moving_sum",
                                                  "difference"),
                                    period = 1) {
  ts_check(ts)
  smoothing <- ts_match_token(match.arg(smoothing),
                              c("none", "moving_average", "moving_sum", "difference"), "smoothing")
  ts_run_series(ts, "peaks_over_threshold", list(
    threshold = as.double(threshold),
    min_steps_between_events = as.integer(min_steps_between_events),
    smoothing = smoothing, period = as.integer(period)
  ), interval = "irregular")
}

#' Seasonal decomposition
#'
#' Classical additive decomposition into trend, seasonal and residual components, with the
#' seasonal part extracted by keeping only the harmonics of the seasonal frequency.
#'
#' @details
#' The trend is a centred moving average over `period` ordinates, so it -- and the residual with
#' it -- is undefined for the first `period - 1` ordinates, reported as `NA`. The seasonal
#' component is defined everywhere. Where all three are defined they add back to the original
#' value exactly.
#'
#' @param ts a `corehydro_ts`.
#' @param period the seasonal period in ordinates (12 for monthly data with an annual cycle). The
#'   series must span at least two complete periods.
#' @return a data frame of `date`, `trend`, `seasonal` and `residual`, one row per input ordinate.
#' @examples
#' dates <- seq(as.Date("2000-01-01"), by = "month", length.out = 48)
#' value <- 100 + 0.5 * seq_along(dates) + 10 * sin(2 * pi * seq_along(dates) / 12)
#' head(ts_seasonal_decompose(time_series(dates, value, "one_month"), period = 12))
#' @export
ts_seasonal_decompose <- function(ts, period) {
  ts_check(ts)
  m <- ts_run_table(ts, "seasonal_decompose", list(period = as.integer(period)),
                    colnames = c("date", "trend", "seasonal", "residual"))
  data.frame(date = ts_posix(m[, 1]), trend = m[, 2], seasonal = m[, 3], residual = m[, 4])
}

#' Resample a time series
#'
#' Two stochastic resamplers for generating synthetic records of arbitrary length.
#'
#' @details
#' `ts_resample_knn()` is the conditional k-nearest-neighbour bootstrap of Lall and Sharma (1996):
#' at each step it finds the `k` historical observations closest to the current value, picks one at
#' random, and advances to whatever historically came NEXT. That conditioning is what preserves
#' the lag-1 structure -- returning the neighbour's own value instead would collapse the trajectory
#' toward its starting point.
#'
#' `ts_resample_block_bootstrap()` draws contiguous blocks of `block_size` observations uniformly
#' with replacement and concatenates them. It preserves the marginal distribution and the
#' within-block dependence, at the cost of a discontinuity at each block boundary.
#'
#' Both draw from the generator inside the shared C++ core, so a seeded call gives bit-identical
#' results in R and Python.
#'
#' @param ts a `corehydro_ts`.
#' @param time_steps the length of the synthetic record.
#' @param k the number of nearest neighbours. Lall and Sharma suggest `floor(sqrt(n))`.
#' @param block_size the resampling block length.
#' @param seed integer PRNG seed. Default 12345.
#' @return a `corehydro_ts` of length `time_steps`.
#' @references
#' Lall, U. and Sharma, A. (1996). A nearest neighbor bootstrap for resampling hydrologic time
#' series. Water Resources Research 32(3), 679-693.
#'
#' Kuensch, H.R. (1989). The jackknife and the bootstrap for general stationary observations.
#' Annals of Statistics 17(3), 1217-1241.
#' @examples
#' ts <- time_series(as.Date("2000-01-01") + 0:29, sin(seq(0, 6, length.out = 30)) * 10 + 20)
#' ts_resample_knn(ts, time_steps = 10, k = 5, seed = 42)
#' @export
ts_resample_knn <- function(ts, time_steps, k, seed = 12345) {
  ts_check(ts)
  ts_run_series(ts, "resample_knn", list(
    time_steps = as.integer(time_steps), k = as.integer(k), seed = as.integer(seed)
  ))
}

#' @rdname ts_resample_knn
#' @export
ts_resample_block_bootstrap <- function(ts, time_steps, block_size, seed = 12345) {
  ts_check(ts)
  ts_run_series(ts, "resample_block_bootstrap", list(
    time_steps = as.integer(time_steps), block_size = as.integer(block_size),
    seed = as.integer(seed)
  ))
}
