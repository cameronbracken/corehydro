# The public data layer: censored observation frames and the diagnostics computed off them.
# Serializes to the `data_frame` spec grammar that core/include/corehydro/models/model_spec.hpp
# parses, and reads back through the glue in src/data.cpp.

# Internal: coerce a user-supplied index vector to 0-based integers, or generate 0..n-1.
data_indexes <- function(index, n, what) {
  if (is.null(index)) {
    return(seq_len(n) - 1L)
  }
  index <- as.integer(index)
  if (length(index) != n) {
    stop(sprintf("`%s` index has length %d but %d values", what, length(index), n), call. = FALSE)
  }
  if (anyNA(index) || any(index < 0L)) {
    stop(sprintf("`%s` indexes must be non-negative integers", what), call. = FALSE)
  }
  index
}

# Internal: pull a required column out of a data frame or named list, with a useful message.
data_column <- function(x, name, what) {
  if (is.null(x[[name]])) {
    stop(sprintf("`%s` requires a `%s` column", what, name), call. = FALSE)
  }
  x[[name]]
}

normalize_exact <- function(exact) {
  if (is.null(exact)) {
    return(NULL)
  }
  if (is.numeric(exact)) {
    exact <- list(value = exact)
  }
  value <- as.double(data_column(exact, "value", "exact"))
  index <- data_indexes(exact[["index"]], length(value), "exact")
  flag <- exact[["is_low_outlier"]]
  flag <- if (is.null(flag)) rep(FALSE, length(value)) else rep_len(as.logical(flag), length(value))
  lapply(seq_along(value), function(i) {
    list(index = index[i], value = value[i], is_low_outlier = flag[i])
  })
}

normalize_interval <- function(interval) {
  if (is.null(interval)) {
    return(NULL)
  }
  lower <- as.double(data_column(interval, "lower", "interval"))
  value <- as.double(data_column(interval, "value", "interval"))
  upper <- as.double(data_column(interval, "upper", "interval"))
  if (length(lower) != length(value) || length(upper) != length(value)) {
    stop("`interval` lower, value, and upper must be the same length", call. = FALSE)
  }
  if (any(lower > value) || any(value > upper)) {
    stop("`interval` requires lower <= value <= upper for every observation", call. = FALSE)
  }
  index <- data_indexes(interval[["index"]], length(value), "interval")
  lapply(seq_along(value), function(i) {
    list(index = index[i], lower = lower[i], value = value[i], upper = upper[i])
  })
}

normalize_threshold <- function(threshold) {
  if (is.null(threshold)) {
    return(NULL)
  }
  value <- as.double(data_column(threshold, "value", "threshold"))
  start_index <- as.integer(data_column(threshold, "start_index", "threshold"))
  end_index <- as.integer(data_column(threshold, "end_index", "threshold"))
  number_above <- as.integer(data_column(threshold, "number_above", "threshold"))
  n <- length(value)
  if (length(start_index) != n || length(end_index) != n || length(number_above) != n) {
    stop("`threshold` columns must all be the same length", call. = FALSE)
  }
  if (any(end_index < start_index)) {
    stop("`threshold` requires end_index >= start_index for every period", call. = FALSE)
  }
  lapply(seq_len(n), function(i) {
    list(
      start_index = start_index[i], end_index = end_index[i],
      value = value[i], number_above = number_above[i]
    )
  })
}

normalize_uncertain <- function(uncertain) {
  if (is.null(uncertain)) {
    return(NULL)
  }
  if (inherits(uncertain, "corehydro_dist")) {
    uncertain <- list(uncertain)
  }
  dists <- if (!is.null(uncertain[["distribution"]])) uncertain[["distribution"]] else uncertain
  if (!is.list(dists) || !all(vapply(dists, inherits, logical(1), "corehydro_dist"))) {
    stop(
      "`uncertain` must be a list of corehydro_dist objects, or a list with a ",
      "`distribution` element holding one",
      call. = FALSE
    )
  }
  index <- data_indexes(uncertain[["index"]], length(dists), "uncertain")
  lapply(seq_along(dists), function(i) list(index = index[i], distribution = dists[[i]]))
}

#' Censored observation data for an analysis
#'
#' Assemble the observations an analysis or model is fit to. Beyond a plain systematic record
#' (`exact`), a frame can carry the censored observation types Bulletin 17C flood-frequency work
#' depends on: historical or paleoflood observations known only within a range (`interval`),
#' perception thresholds recording that nothing above a level occurred over a span of years
#' (`threshold`), and observations whose measurement error is itself a distribution
#' (`uncertain`).
#'
#' This is the port of the RMC.BestFit `DataFrame` class. It is named `analysis_data()` rather
#' than `data_frame()` to avoid colliding with the R and pandas types of that name.
#'
#' Indexes are 0-based, matching the C# library and the shared C++ core, and are generated
#' sequentially when not supplied. An index is a position in the record, so an interval
#' observation at index 40 sits after the 40th exact observation in chronological order.
#'
#' @param exact a numeric vector of observations, or a data frame with a `value` column plus
#'   optional `index` and `is_low_outlier` columns.
#' @param interval a data frame with `lower`, `value`, and `upper` columns plus an optional
#'   `index`, giving observations known only to lie in a range.
#' @param threshold a data frame with `start_index`, `end_index`, `value`, and `number_above`
#'   columns, giving perception thresholds over spans of the record.
#' @param uncertain a list of [distribution()] objects (one per observation), or a list with an
#'   `index` element and a `distribution` element holding them.
#' @param low_outlier_threshold optional numeric; values at or below it are treated as
#'   left-censored low outliers.
#' @param mgbt_low_outliers logical; when `TRUE` the Multiple Grubbs-Beck test picks the low
#'   outlier threshold from the data. Mutually exclusive with `low_outlier_threshold` and with
#'   explicit `is_low_outlier` flags.
#' @return An object of class `corehydro_data`.
#' @seealso [analysis_data_summary()] for plotting positions and record diagnostics,
#'   [model_univariate()] to fit a model to one, [mgbt_test()].
#' @export
#' @examples
#' # A systematic record on its own.
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500)
#' analysis_data(peaks)
#'
#' # Adding two historical floods known only within a range, and a perception threshold
#' # covering the 40 years before the gauge was installed.
#' analysis_data(
#'   exact = peaks,
#'   interval = data.frame(
#'     index = c(8, 9), lower = c(30000, 26000),
#'     value = c(35000, 29000), upper = c(40000, 32000)
#'   ),
#'   threshold = data.frame(
#'     start_index = 8, end_index = 47, value = 25000, number_above = 2
#'   )
#' )
analysis_data <- function(exact = NULL, interval = NULL, threshold = NULL, uncertain = NULL,
                          low_outlier_threshold = NULL, mgbt_low_outliers = FALSE) {
  mgbt_low_outliers <- isTRUE(mgbt_low_outliers)
  if (mgbt_low_outliers && !is.null(low_outlier_threshold)) {
    stop(
      "`mgbt_low_outliers = TRUE` picks the threshold from the data; ",
      "do not also supply `low_outlier_threshold`",
      call. = FALSE
    )
  }

  spec <- list(
    exact = normalize_exact(exact),
    interval = normalize_interval(interval),
    threshold = normalize_threshold(threshold),
    uncertain = normalize_uncertain(uncertain),
    low_outlier_threshold = if (is.null(low_outlier_threshold)) {
      NULL
    } else {
      as.double(low_outlier_threshold)
    },
    mgbt_low_outliers = if (mgbt_low_outliers) TRUE else NULL,
    # An explicit threshold means "censor everything below this", so ask the core to derive the
    # flags and the low-outlier count from it rather than only recording the value.
    threshold_low_outliers = if (is.null(low_outlier_threshold)) NULL else TRUE
  )

  if (is.null(spec$exact) && is.null(spec$interval) &&
    is.null(spec$threshold) && is.null(spec$uncertain)) {
    stop("`analysis_data()` needs at least one of exact, interval, threshold, or uncertain",
      call. = FALSE
    )
  }
  structure(spec, class = "corehydro_data")
}

check_data <- function(d) {
  if (inherits(d, "corehydro_data")) {
    return(d)
  }
  if (is.numeric(d)) {
    return(analysis_data(exact = d))
  }
  stop("`data` must be a numeric vector or a corehydro_data object from analysis_data()",
    call. = FALSE
  )
}

#' @export
print.corehydro_data <- function(x, ...) {
  counts <- c(
    exact = length(x$exact), interval = length(x$interval),
    threshold = length(x$threshold), uncertain = length(x$uncertain)
  )
  counts <- counts[counts > 0]
  cat(sprintf(
    "<corehydro_data> %s\n",
    paste(sprintf("%d %s", counts, names(counts)), collapse = ", ")
  ))
  if (!is.null(x$low_outlier_threshold)) {
    cat(sprintf("  low outlier threshold: %g\n", x$low_outlier_threshold))
  }
  if (isTRUE(x$mgbt_low_outliers)) {
    cat("  low outliers: from the Multiple Grubbs-Beck test\n")
  }
  invisible(x)
}

#' Summarize an observation frame
#'
#' Run the plotting-position and threshold cascade over an [analysis_data()] frame and return the
#' record diagnostics. The plotting positions are the Hirsch-Stedinger censored positions, so
#' interval and threshold observations shift the positions of the systematic record even though
#' they carry no plotting ordinate themselves.
#'
#' @param data a `corehydro_data` object, or a numeric vector of observations.
#' @param plotting_parameter the plotting-position parameter in `[0, 1)`: `0` (the default) is
#'   Weibull, `0.40` Cunnane, `0.44` Gringorten, `0.50` Hazen.
#' @return A named list. `index`, `value`, `plotting_position`, and `is_low_outlier` are parallel
#'   vectors over the exact series; `number_of_low_outliers`, `low_outlier_threshold`,
#'   `plotting_parameter`, `lambda` (events per index), `total_record_length`,
#'   `zero_value_relative_frequency`, and the four series counts are scalars.
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500)
#' s <- analysis_data_summary(analysis_data(peaks))
#' data.frame(value = s$value, position = s$plotting_position)
#'
#' # A perception threshold reweights the systematic record's positions.
#' d <- analysis_data(
#'   exact = peaks,
#'   threshold = data.frame(start_index = 8, end_index = 47, value = 25000, number_above = 0)
#' )
#' analysis_data_summary(d)$plotting_position
analysis_data_summary <- function(data, plotting_parameter = 0) {
  data <- check_data(data)
  plotting_parameter <- as.double(plotting_parameter)
  if (!is.finite(plotting_parameter) || plotting_parameter < 0 || plotting_parameter >= 1) {
    stop("`plotting_parameter` must be finite and in [0, 1)", call. = FALSE)
  }
  ch_data_frame_summary_(to_spec_json(unclass(data)), plotting_parameter)
}

#' Threshold selection diagnostics for peaks-over-threshold analysis
#'
#' The two standard plots for choosing a peaks-over-threshold cutoff. The mean residual life plot
#' is the sample mean of excesses above each candidate threshold, which is linear in the
#' threshold once a generalized Pareto model holds. The parameter stability plot fits a
#' generalized Pareto distribution at each candidate threshold; the modified scale and the shape
#' are approximately constant above the true threshold.
#'
#' Candidate thresholds with too few exceedances are dropped (fewer than 5 for mean residual
#' life, fewer than 10 for parameter stability), as are thresholds where the fit fails, so the
#' returned vectors are usually shorter than `n_thresholds`.
#'
#' @param x numeric vector of observations.
#' @param u_min,u_max the range of candidate thresholds to scan.
#' @param n_thresholds number of equally spaced candidate thresholds in `[u_min, u_max]`.
#' @param confidence_level confidence level for the interval bands.
#' @param method `"mean_residual_life"` (the default) or `"parameter_stability"`.
#' @return A named list of parallel vectors. Both methods return `threshold` and
#'   `exceedance_count`; `"mean_residual_life"` adds `mean_excess`, `lower_ci`, and `upper_ci`,
#'   and `"parameter_stability"` adds `modified_scale`, `shape`, and their confidence bounds. The
#'   columns the chosen method does not populate come back empty.
#' @references Coles (2001), *An Introduction to Statistical Modeling of Extreme Values*,
#'   Section 4.3; Davison and Smith (1990).
#' @export
#' @examples
#' d <- distribution("GeneralizedPareto", c(0, 100, 0.1))
#' x <- dist_random(d, 500, seed = 42)
#' mrl <- threshold_diagnostics(x, u_min = 0, u_max = 200, n_thresholds = 10)
#' data.frame(threshold = mrl$threshold, mean_excess = mrl$mean_excess)
threshold_diagnostics <- function(x, u_min, u_max, n_thresholds = 20, confidence_level = 0.95,
                                  method = c("mean_residual_life", "parameter_stability")) {
  method <- match.arg(method)
  ch_threshold_diagnostics_(
    as.double(x), method, as.double(u_min), as.double(u_max),
    as.integer(n_thresholds), as.double(confidence_level)
  )
}
