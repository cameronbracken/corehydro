# The Numerics toolbox surface. Every verb serializes its options to the toolbox_runner.hpp
# grammar and runs one method through ch_toolbox_run_; bulk data goes across as numeric vectors,
# not JSON. Nothing holds C++ state.

# Internal: one call into the shared runner. `data` is a list of numeric vectors, `options` a
# named list serialized with the same emitter every other spec uses.
toolbox_run <- function(group, method, data = list(), options = list()) {
  opts <- if (length(options) == 0L) "{}" else to_spec_json(options)
  ch_toolbox_run_(group, method, lapply(data, as.double), opts)
}

# Internal: reject the two mistakes every paired-series verb can make, naming the argument.
check_pair <- function(x, y, x_name = "x", y_name = "y") {
  if (!is.numeric(x) || !is.numeric(y)) {
    stop(sprintf("`%s` and `%s` must be numeric vectors", x_name, y_name), call. = FALSE)
  }
  if (length(x) != length(y)) {
    stop(sprintf("`%s` and `%s` must have the same length; got %d and %d",
                 x_name, y_name, length(x), length(y)), call. = FALSE)
  }
  if (length(x) < 2L) {
    stop(sprintf("`%s` and `%s` must have at least two elements", x_name, y_name), call. = FALSE)
  }
  invisible(NULL)
}

#' Correlation between two samples
#'
#' Mirrors the C# `Correlation` class of the Numerics library. Upstream's matrix overloads
#' (`Pearson(double[,])`, `Spearman(double[,])`) are not ported, so only the paired-vector forms
#' are available here.
#'
#' @param x,y numeric vectors of equal length, at least two elements.
#' @param method one of `"pearson"` (the default), `"spearman"`, or `"kendall"`.
#' @return a single numeric correlation coefficient.
#' @examples
#' x <- c(14, 8, 32, 7, 3, 15)
#' y <- c(10, 5, 7, 4, 3, 8)
#' correlation(x, y)
#' correlation(x, y, method = "kendall")
#' @export
correlation <- function(x, y, method = c("pearson", "spearman", "kendall")) {
  method <- match.arg(method)
  check_pair(x, y)
  toolbox_run("correlation", method, list(x, y))$values[[1]]
}

# The "statistics" and "spectra" toolbox groups (Task 3). "statistics" is descriptive/streaming
# summary statistics (mirroring C# Statistics/RunningStatistics/RunningCovarianceMatrix);
# "spectra" is the FFT and autocorrelation surface (mirroring C# Fourier and the newly-ported
# Autocorrelation).

#' Summary statistics for a sample
#'
#' A one-shot description of `x`: count, extremes, mean, variance, and the standard shape
#' statistics, computed via the C# `RunningStatistics` class (Welford's algorithm) with no prior
#' state. For a chunked, resumable version see [running_statistics()].
#'
#' @param x numeric vector.
#' @return a named numeric vector with entries `n`, `minimum`, `maximum`, `mean`, `variance`,
#'   `sd`, `cv`, `skewness`, `kurtosis`, and the four raw moments `m1` to `m4`.
#' @examples
#' summary_statistics(c(2, 4, 4, 4, 5, 5, 7, 9))
#' @export
summary_statistics <- function(x) {
  r <- toolbox_run("statistics", "summary", list(x))
  stats::setNames(r$values, r$names)
}

#' Product moments of a sample
#'
#' Mirrors the C# `Statistics.ProductMoments`: the mean, sample standard deviation, and the
#' bias-corrected skewness and excess kurtosis. Requires at least 4 observations (returns `NaN`
#' otherwise, matching the C# behavior).
#'
#' @param x numeric vector.
#' @return a named numeric vector with entries `mean`, `sd`, `skewness`, `kurtosis`.
#' @examples
#' product_moments(c(2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9))
#' @export
product_moments <- function(x) {
  r <- toolbox_run("statistics", "product_moments", list(x))
  stats::setNames(r$values, r$names)
}

#' L-moments of a sample
#'
#' Mirrors the C# `Statistics.LinearMoments`. Requires at least 4 observations (returns `NaN`
#' otherwise, matching the C# behavior).
#'
#' @param x numeric vector.
#' @return a named numeric vector with entries `l1` (L-mean), `l2` (L-scale), `t3`
#'   (L-skewness), `t4` (L-kurtosis).
#' @examples
#' l_moments(c(2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9))
#' @export
l_moments <- function(x) {
  r <- toolbox_run("statistics", "l_moments", list(x))
  stats::setNames(r$values, r$names)
}

#' Ranks of a sample
#'
#' Mirrors the C# `Statistics.RanksInPlace(double[])`: tied values (exact equality) receive the
#' average rank of their run.
#'
#' @param x numeric vector.
#' @return a numeric vector of ranks, same length and order as `x`.
#' @examples
#' ranks(c(3, 1, 2, 1))
#' @export
ranks <- function(x) {
  toolbox_run("statistics", "ranks", list(x))$values
}

#' Percentiles of a sample
#'
#' Mirrors the C# `Statistics.Percentile`: zero-based linear interpolation (R `quantile()`
#' Type 7).
#'
#' @param x numeric vector.
#' @param probs numeric vector of probabilities in `[0, 1]`.
#' @param sorted set `TRUE` when `x` is already sorted ascending, to skip re-sorting it.
#' @return a numeric vector, one percentile per entry of `probs`.
#' @examples
#' percentile(c(1, 2, 3, 4, 5), probs = c(0.25, 0.5, 0.75))
#' @export
percentile <- function(x, probs, sorted = FALSE) {
  if (!is.numeric(probs)) {
    stop("`probs` must be numeric", call. = FALSE)
  }
  toolbox_run("statistics", "percentile", list(x, probs), list(sorted = isTRUE(sorted)))$values
}

#' Streaming summary statistics
#'
#' Accumulates count, extremes, and the first four moments over one or more chunks of data,
#' mirroring the C# `RunningStatistics` class. The accumulator state travels in the return value,
#' so a chunked run holds no C++ state and the result serializes with `save()`.
#'
#' @param x numeric vector, the next chunk of data.
#' @param state a `corehydro_running` object from a previous call, or `NULL` (the default) to
#'   start a fresh accumulator.
#' @return a `corehydro_running` list with `n`, `minimum`, `maximum`, `mean`, `variance`, `sd`,
#'   `cv`, `skewness`, `kurtosis`, and the four raw moments `m1` to `m4`.
#' @examples
#' s <- running_statistics(c(1, 2, 3))
#' s <- running_statistics(c(4, 5, 6), state = s)
#' s$n
#' s$mean
#' @export
running_statistics <- function(x, state = NULL) {
  if (!is.null(state) && !inherits(state, "corehydro_running")) {
    stop("`state` must be a corehydro_running object from a previous call", call. = FALSE)
  }
  opts <- if (is.null(state)) {
    list()
  } else {
    list(state = list(n = state$n, m1 = state$m1, m2 = state$m2, m3 = state$m3, m4 = state$m4,
                      minimum = state$minimum, maximum = state$maximum))
  }
  r <- toolbox_run("statistics", if (is.null(state)) "summary" else "running", list(x), opts)
  structure(as.list(stats::setNames(r$values, r$names)), class = "corehydro_running")
}

#' @export
print.corehydro_running <- function(x, ...) {
  cat(sprintf("<corehydro_running> n = %d, mean = %.6g, sd = %.6g\n", x$n, x$mean, x$sd))
  invisible(x)
}

#' Streaming covariance and correlation matrix
#'
#' Accumulates a running mean vector and covariance matrix over one or more chunks of
#' multivariate data, mirroring the C# `RunningCovarianceMatrix` class. The accumulator state
#' travels in the return value, so a chunked run holds no C++ state.
#'
#' @param x a numeric matrix (or an object coercible to one via [as.matrix()]), observations in
#'   rows and variables in columns.
#' @param state a `corehydro_running_covariance` object from a previous call, or `NULL` (the
#'   default) to start a fresh accumulator.
#' @return a `corehydro_running_covariance` list with `n`, `mean` (length-`ncol(x)`), and the
#'   `covariance`, `sample_covariance`, `sample_correlation`, `population_covariance`, and
#'   `population_correlation` matrices (each `ncol(x)` by `ncol(x)`). `covariance` is unadjusted
#'   by sample size; `sample_*`/`population_*` are the N-1- and N-normalized variants. The C#
#'   accumulator seeds `covariance` at the identity matrix before the first push (a stability
#'   prior for its other consumer, adaptive MCMC), so every derived matrix carries a small
#'   diagonal-only bias that only fades as `n` grows -- do not expect an exact match to
#'   `stats::cov()`/`stats::cor()` on the same data at small `n`.
#' @examples
#' x <- matrix(c(1, 2, 3, 4, 5, 2, 4, 5, 4, 5), ncol = 2)
#' running_covariance(x)
#' @export
running_covariance <- function(x, state = NULL) {
  if (!is.null(state) && !inherits(state, "corehydro_running_covariance")) {
    stop("`state` must be a corehydro_running_covariance object from a previous call", call. = FALSE)
  }
  x <- as.matrix(x)
  if (!is.numeric(x)) {
    stop("`x` must be a numeric matrix", call. = FALSE)
  }
  size <- ncol(x)
  data <- lapply(seq_len(size), function(j) x[, j])
  opts <- if (is.null(state)) {
    list()
  } else {
    list(state = list(n = state$n, mean = as.double(state$mean),
                      covariance = as.double(t(state$covariance))))
  }
  r <- toolbox_run("statistics", "running_covariance", data, opts)
  block <- size * size
  unflatten <- function(v) matrix(v, nrow = size, ncol = size, byrow = TRUE)
  structure(list(
    n = r$values[[1]],
    mean = r$values[2:(1 + size)],
    covariance = unflatten(r$values[(2 + size):(1 + size + block)]),
    sample_covariance = unflatten(r$values[(2 + size + block):(1 + size + 2 * block)]),
    sample_correlation = unflatten(r$values[(2 + size + 2 * block):(1 + size + 3 * block)]),
    population_covariance = unflatten(r$values[(2 + size + 3 * block):(1 + size + 4 * block)]),
    population_correlation = unflatten(r$values[(2 + size + 4 * block):(1 + size + 5 * block)])
  ), class = "corehydro_running_covariance")
}

#' @export
print.corehydro_running_covariance <- function(x, ...) {
  cat(sprintf("<corehydro_running_covariance> n = %d, %d variable(s)\n", x$n, length(x$mean)))
  invisible(x)
}

#' Autocorrelation, autocovariance, or partial autocorrelation function
#'
#' Mirrors the C# `Autocorrelation` class. Computes the requested function out to `max_lag`
#' (default `floor(min(10*log10(N), N-1))`, matching the C# default) and attaches the asymptotic
#' confidence band for ACF/PACF rho values at `confidence_level`.
#'
#' @param x numeric vector, at least two elements.
#' @param max_lag maximum lag. `NULL` (the default) uses the C# default rule.
#' @param type one of `"correlation"` (the default), `"covariance"`, or `"partial"`.
#' @param confidence_level confidence level for the band. Default 0.95.
#' @return a `corehydro_autocorrelation` list with `lag`, `value`, `type`, and the two-element
#'   `ci` (`lower`, `upper`).
#' @examples
#' x <- c(5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 5, 6, 4, 7, 3, 8, 2, 9, 1, 10)
#' autocorrelation(x, max_lag = 5)
#' @export
autocorrelation <- function(x, max_lag = NULL,
                            type = c("correlation", "covariance", "partial"),
                            confidence_level = 0.95) {
  type <- match.arg(type)
  if (!is.numeric(x) || length(x) < 2L) {
    stop("`x` must be a numeric vector with at least two elements", call. = FALSE)
  }
  opts <- list(type = type)
  if (!is.null(max_lag)) opts$lag_max <- as.integer(max_lag)
  r <- toolbox_run("spectra", "autocorrelation", list(x), opts)
  n <- r$dims[[1]]
  ci <- toolbox_run("spectra", "autocorrelation_ci", list(),
                    list(sample_size = length(x), confidence_level = confidence_level))
  structure(list(lag = r$values[seq(1L, 2L * n, by = 2L)],
                value = r$values[seq(2L, 2L * n, by = 2L)],
                type = type, ci = stats::setNames(ci$values, ci$names)),
            class = "corehydro_autocorrelation")
}

#' @export
print.corehydro_autocorrelation <- function(x, ...) {
  cat(sprintf("<corehydro_autocorrelation> type = %s, %d lag(s)\n", x$type, length(x$lag)))
  invisible(x)
}

#' Cross-correlation of two series
#'
#' Mirrors the C# `Fourier.Correlation`: an FFT-based correlation, NOT a normalized correlation
#' coefficient (see [correlation()] for that). Both series must have the same power-of-two
#' length.
#'
#' @param x,y numeric vectors of equal, power-of-two length.
#' @return a numeric vector in wraparound order: increasing positive lags in `[1]` up to
#'   `[n/2]`, increasing negative lags in `[n]` down to `[n/2 + 1]`.
#' @examples
#' cross_correlation(c(1, 2, 3, 4), c(4, 3, 2, 1))
#' @export
cross_correlation <- function(x, y) {
  check_pair(x, y)
  toolbox_run("spectra", "cross_correlation", list(x, y))$values
}

#' Discrete Fourier transform
#'
#' Mirrors the C# `Fourier.FFT`: an in-place complex FFT on data packed as
#' `[re0, im0, re1, im1, ...]`; `length(x) / 2` must be a power of two.
#'
#' @param x numeric vector, complex data packed as `[re0, im0, re1, im1, ...]`.
#' @param inverse if `TRUE`, computes `n` times the inverse transform (matching the C# doc
#'   comment -- divide by `n` for the true inverse).
#' @return the transformed vector, same length as `x`.
#' @examples
#' dft(c(1, 0, 2, 0, 3, 0, 4, 0))
#' @export
dft <- function(x, inverse = FALSE) {
  toolbox_run("spectra", "dft", list(x), list(inverse = isTRUE(inverse)))$values
}

#' Discrete Fourier transform of real-valued data
#'
#' Mirrors the C# `Fourier.RealFFT`: transforms `length(x)` real values in place (`length(x)`
#' must be a power of two) into the positive-frequency half of the complex transform.
#'
#' @param x numeric vector, real-valued, power-of-two length.
#' @param inverse if `TRUE`, computes the inverse transform of a complex array that is the
#'   transform of real data (matching the C# doc comment -- multiply by `2/length(x)` for the
#'   true inverse).
#' @return the transformed vector, same length as `x`.
#' @examples
#' dft_real(c(1, 2, 3, 4))
#' @export
dft_real <- function(x, inverse = FALSE) {
  toolbox_run("spectra", "dft_real", list(x), list(inverse = isTRUE(inverse)))$values
}
