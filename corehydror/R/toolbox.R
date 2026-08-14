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
    # spec_array() forces an array even at size == 1 (a one-variable accumulator): a bare
    # length-1 numeric vector would otherwise serialize as a JSON scalar (to_spec_json's
    # length-1 rule), which toolbox_runner.hpp's as_double_vector() rejects.
    list(state = list(n = state$n, mean = spec_array(as.double(state$mean)),
                      covariance = spec_array(as.double(t(state$covariance)))))
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

# The "histogram" and "interpolation" toolbox groups (Task 4). "histogram" mirrors the C#
# Histogram class (Rice-rule or explicit bin count, both deriving their range from the data --
# there is no lower/upper-bound constructor overload to expose); "interpolation" mirrors Linear
# and Bilinear, including their independent x/y transforms and Linear's separate extrapolate path.

#' Bin a sample into a histogram
#'
#' Mirrors the C# `Histogram` class of the Numerics library. With `bins = NULL` the bin count
#' follows the Rice rule, `ceiling(2 * n^(1/3)) + 1`, exactly as the C# data constructor does.
#'
#' @param x numeric vector of observations, at least one element.
#' @param bins optional positive number of bins. `NULL` (the default) uses the Rice rule.
#' @return a data frame with columns `lower`, `upper`, `midpoint`, and `frequency`, carrying the
#'   histogram's `mean`, `median`, `mode`, `sd`, and `bin_width` as attributes.
#' @examples
#' h <- histogram(c(1, 2, 2.5, 3, 3.5, 4, 5, 7, 8, 9))
#' h
#' attr(h, "mode")
#' @export
histogram <- function(x, bins = NULL) {
  if (!is.numeric(x) || length(x) == 0L) {
    stop("`x` must be a non-empty numeric vector", call. = FALSE)
  }
  opts <- if (is.null(bins)) list() else list(bins = as.integer(bins))
  b <- toolbox_run("histogram", "bins", list(x), opts)
  out <- as.data.frame(matrix(b$values, ncol = 4L, byrow = TRUE))
  names(out) <- b$names
  s <- toolbox_run("histogram", "statistics", list(x), opts)
  for (i in seq_along(s$names)) attr(out, s$names[[i]]) <- s$values[[i]]
  out
}

#' Interpolate a paired series
#'
#' Mirrors the C# `Linear` interpolater of the Numerics library, including its x and y
#' transforms.
#'
#' @param x,y numeric vectors of equal length defining the knots.
#' @param xout numeric vector of positions to interpolate at.
#' @param x_transform,y_transform one of `"none"` (the default), `"log"`, or `"normal_z"`.
#' @param sort_order `"ascending"` (the default) or `"descending"`, describing `x`.
#' @param extrapolate whether to extend the end segments beyond the knots. Default `FALSE`,
#'   which clamps to the end knot, matching the C# `Interpolate()` default; `TRUE` calls the C#
#'   `Extrapolate()` method instead.
#' @return a numeric vector the same length as `xout`.
#' @examples
#' interpolate(c(1, 2, 3, 4), c(10, 20, 30, 40), c(1.5, 2.5))
#' @export
interpolate <- function(x, y, xout, x_transform = c("none", "log", "normal_z"),
                        y_transform = c("none", "log", "normal_z"),
                        sort_order = c("ascending", "descending"), extrapolate = FALSE) {
  check_pair(x, y)
  if (!is.numeric(xout)) {
    stop("`xout` must be numeric", call. = FALSE)
  }
  x_transform <- match.arg(x_transform)
  y_transform <- match.arg(y_transform)
  sort_order <- match.arg(sort_order)
  toolbox_run("interpolation", "linear", list(x, y, xout),
              list(x_transform = x_transform, y_transform = y_transform,
                   sort_order = sort_order, extrapolate = isTRUE(extrapolate)))$values
}

#' Interpolate a 2D grid (bilinear interpolation)
#'
#' Mirrors the C# `Bilinear` interpolater of the Numerics library.
#'
#' @param x1,x2 numeric vectors of grid coordinates.
#' @param y numeric matrix with `length(x1)` rows and `length(x2)` columns, `y[i, j]` the value
#'   at `(x1[i], x2[j])`.
#' @param x1out,x2out numeric vectors of equal length, positions to interpolate at.
#' @param x1_transform,x2_transform,y_transform one of `"none"` (the default), `"log"`, or
#'   `"normal_z"`.
#' @param sort_order `"ascending"` (the default) or `"descending"`, describing `x1` and `x2`.
#' @return a numeric vector the same length as `x1out`/`x2out`.
#' @examples
#' interpolate_2d(c(1, 2, 3), c(1, 2, 3), diag(3), c(1.5), c(1.5))
#' @export
interpolate_2d <- function(x1, x2, y, x1out, x2out,
                           x1_transform = c("none", "log", "normal_z"),
                           x2_transform = c("none", "log", "normal_z"),
                           y_transform = c("none", "log", "normal_z"),
                           sort_order = c("ascending", "descending")) {
  if (!is.numeric(x1) || !is.numeric(x2)) {
    stop("`x1` and `x2` must be numeric vectors", call. = FALSE)
  }
  y <- as.matrix(y)
  if (!identical(dim(y), c(length(x1), length(x2)))) {
    stop(sprintf("`y` must be a %d x %d matrix (length(x1) x length(x2)); got %d x %d",
                 length(x1), length(x2), nrow(y), ncol(y)), call. = FALSE)
  }
  if (!is.numeric(x1out) || !is.numeric(x2out) || length(x1out) != length(x2out)) {
    stop("`x1out` and `x2out` must be numeric vectors of the same length", call. = FALSE)
  }
  x1_transform <- match.arg(x1_transform)
  x2_transform <- match.arg(x2_transform)
  y_transform <- match.arg(y_transform)
  sort_order <- match.arg(sort_order)
  toolbox_run("interpolation", "bilinear",
              list(x1, x2, as.double(t(y)), x1out, x2out),
              list(x1_transform = x1_transform, x2_transform = x2_transform,
                   y_transform = y_transform, sort_order = sort_order))$values
}

# The "sampling" and "probability" toolbox groups (Task 6). "sampling" is the C# SobolSequence
# quasi-random low-discrepancy sequence and Stratify::XValues equal-width axis stratification;
# "probability" is Probability's joint-probability dispatch (plain dependency form and the
# indicator + correlation-matrix HPCM form).

#' Sobol quasi-random low-discrepancy sequence
#'
#' Mirrors the C# `SobolSequence` class. `dimension > 1` needs the new-joe-kuo-6 direction
#' numbers shipped with the package (`dimension == 1` needs no file, matching the C# ctor); the
#' installed file is located automatically with [system.file()].
#'
#' `n` and `dimension` are validated rather than passed through to the C++ layer, which would
#' otherwise silently return an empty result for a non-positive value instead of raising an
#' error.
#'
#' @param n number of points to generate, at least 1.
#' @param dimension spatial dimension, between 1 and 21201. Default 1.
#' @param skip number of points to skip before the first returned point: `skip = k` returns the
#'   same first point as the C# `SkipTo(k)` call, i.e. the sequence's `(k + 1)`-th point.
#'   Default 0 (no skip).
#' @return an `n` by `dimension` numeric matrix, every value in `[0, 1)`.
#' @examples
#' sobol_sequence(5, dimension = 2)
#' @export
sobol_sequence <- function(n, dimension = 1L, skip = 0L) {
  if (!is.numeric(n) || length(n) != 1L || n < 1) {
    stop("`n` must be a single positive integer", call. = FALSE)
  }
  if (!is.numeric(dimension) || length(dimension) != 1L || dimension < 1) {
    stop("`dimension` must be a single positive integer", call. = FALSE)
  }
  path <- ""
  if (dimension > 1L) {
    path <- system.file("extdata", "new-joe-kuo-6.21201", package = "corehydror")
    if (!nzchar(path)) {
      stop("direction-numbers file not found in corehydror inst/extdata", call. = FALSE)
    }
  }
  opts <- list(dimension = as.integer(dimension), n = as.integer(n), skip = as.integer(skip),
               path = path)
  r <- toolbox_run("sampling", "sobol", list(), opts)
  matrix(r$values, nrow = r$dims[[1]], ncol = r$dims[[2]], byrow = TRUE)
}

#' Stratify an axis into equal-width bins
#'
#' Mirrors the C# `Stratify.XValues(StratificationOptions, isLogarithmic)`: splits
#' `[lower, upper]` into `bins` equal-width strata (equal-width in log10 space when
#' `logarithmic = TRUE`), each carrying a weight defaulting to its own width. `probability =
#' TRUE` always returns zero rows, matching `Stratify.XValues`'s own early return for
#' probability-space options -- the ported header exposes only the `StratificationOptions`
#' overload used by the BestFit estimators' profile-likelihood grids, not the
#' probability-stratification methods (`Probabilities`, `XToProbability`, ...), which are out of
#' scope (see `stratify.hpp`'s file header).
#'
#' `lower`, `upper`, and `bins` are validated rather than passed through to the C++ layer, which
#' would otherwise silently return zero rows for `lower >= upper` (and for `bins < 2`) instead of
#' raising an error.
#'
#' @param lower,upper bounds of the axis to stratify. `lower` must be less than `upper`.
#' @param bins number of bins, greater than 1.
#' @param logarithmic stratify on a log10 scale. Default `FALSE`.
#' @param probability mark the axis as a probability axis; kept only for parity with the C#
#'   constructor argument -- it always yields zero bins (see Details). Default `FALSE`.
#' @return a data frame with columns `lower`, `upper`, `midpoint`, `weight`, one row per bin.
#' @examples
#' stratify(0, 1, bins = 4)
#' @export
stratify <- function(lower, upper, bins, logarithmic = FALSE, probability = FALSE) {
  if (!is.numeric(lower) || !is.numeric(upper) || length(lower) != 1L || length(upper) != 1L) {
    stop("`lower` and `upper` must be single numbers", call. = FALSE)
  }
  if (!is.numeric(bins) || length(bins) != 1L || bins < 2) {
    stop("`bins` must be a single integer greater than 1", call. = FALSE)
  }
  if (lower >= upper) {
    stop(sprintf("`lower` must be less than `upper`; got lower = %s, upper = %s", lower, upper),
         call. = FALSE)
  }
  opts <- list(lower = as.double(lower), upper = as.double(upper), bins = as.integer(bins),
               logarithmic = isTRUE(logarithmic), probability = isTRUE(probability))
  r <- toolbox_run("sampling", "stratify", list(), opts)
  out <- as.data.frame(matrix(r$values, ncol = 4L, byrow = TRUE))
  names(out) <- r$names
  out
}

#' Joint probability of multiple events
#'
#' Mirrors the C# `Probability.JointProbability`. The plain form dispatches on a dependency
#' assumption alone (`"independent"` multiplies, `"positive"` takes the minimum, `"negative"`
#' clamps the excess of the sum over 1). Passing `indicators` (a 0/1 flag per component,
#' selecting which components participate) switches to the indicator-aware form; passing
#' `correlation` too routes `dependency = "correlation"` through Haden Smith's modification of
#' Pandey's Product-of-Conditional-Marginals method (HPCM).
#'
#' @param p numeric vector of marginal probabilities.
#' @param dependency one of `"independent"` (default), `"positive"`, `"negative"`,
#'   `"correlation"`. `"correlation"` requires both `indicators` and `correlation` -- the
#'   underlying C# method itself returns `NaN` for that combination rather than raising an
#'   error, but this wrapper rejects it up front and names the missing argument(s), rather than
#'   handing back a silent `NaN`.
#' @param indicators optional 0/1 vector, the same length as `p`.
#' @param correlation optional `length(p)` by `length(p)` correlation matrix; requires
#'   `indicators`.
#' @return a single numeric probability.
#' @examples
#' joint_probability(c(0.5, 0.5))
#' joint_probability(c(0.5, 0.5), dependency = "positive")
#' @export
joint_probability <- function(p, dependency = c("independent", "positive", "negative", "correlation"),
                              indicators = NULL, correlation = NULL) {
  dependency <- match.arg(dependency)
  if (!is.numeric(p) || length(p) == 0L) {
    stop("`p` must be a non-empty numeric vector", call. = FALSE)
  }
  if (!is.null(correlation) && is.null(indicators)) {
    stop("`correlation` requires `indicators`", call. = FALSE)
  }
  if (identical(dependency, "correlation")) {
    missing_args <- c(if (is.null(indicators)) "indicators", if (is.null(correlation)) "correlation")
    if (length(missing_args) > 0L) {
      stop(sprintf("dependency = \"correlation\" requires %s",
                   paste(sprintf("`%s`", missing_args), collapse = " and ")),
           call. = FALSE)
    }
  }
  data <- list(p)
  if (!is.null(indicators)) {
    if (length(indicators) != length(p)) {
      stop("`indicators` must be the same length as `p`", call. = FALSE)
    }
    data[[2]] <- as.double(indicators)
    if (!is.null(correlation)) {
      correlation <- as.matrix(correlation)
      n <- length(p)
      if (!identical(dim(correlation), c(n, n))) {
        stop(sprintf("`correlation` must be a %d x %d matrix", n, n), call. = FALSE)
      }
      data[[3]] <- as.double(t(correlation))
    }
  }
  toolbox_run("probability", "joint", data, list(dependency = dependency))$values[[1]]
}

# The "link" and "trend" toolbox groups (Task 7). "link" mirrors the seven Numerics link
# functions (numerics/functions/) plus the five BestFit-specific ones (models/link_functions/);
# "trend" evaluates the eleven TrendModelType trend models (models/trend_functions/) that
# trend(), the model-attachment spec builder in R/model.R, already names -- these verbs consume
# that existing object rather than a second constructor.

#' Construct a link function
#'
#' Mirrors the seven Numerics link functions and the five RMC.BestFit-specific ones (the
#' BestFit factory's own YeoJohnson case routes to the Numerics class -- see
#' `best_fit_link_function_factory.hpp` -- so there is one `"YeoJohnson"`, not two). Six of the
#' twelve take construction parameters, passed through `...` by name. `type` is matched
#' case-insensitively (as [trend()] already does) and normalized to its canonical spelling
#' before being sent to the C++ layer.
#'
#' @param type one of `link_names()`, matched case-insensitively.
#' @param ... named construction parameters for the parameterized links: `lambda` for
#'   `"YeoJohnson"`; `gamma0`, `scale`, `epsilon`, `delta` for `"ASinH"`; `a` for `"SES"`;
#'   `sigma0`, `a`, `lambda` for `"LogSES"`; `sigma0`, `log_scale`, `epsilon`, `delta` for
#'   `"LogASinH"`; `mu0`, `scale` for `"Centered"`. R spells the Yeo-Johnson/LogSES exponent
#'   `lambda`; Python spells it `lambda_`, because `lambda` is a reserved word there (see
#'   `corehydropy.toolbox.Link`).
#' @param inner a `corehydro_link` wrapped by `"Centered"`, ignored for every other type.
#' @return a `corehydro_link` spec list.
#' @examples
#' l <- link_function("Log")
#' link(l, c(1, 10, 100))
#' link_inverse(l, c(0, 1, 2))
#' @export
link_function <- function(type, ..., inner = NULL) {
  known <- link_names()
  hit <- match(tolower(type), tolower(known))
  if (is.na(hit)) {
    stop(sprintf("unknown link type \"%s\". Available: %s", type,
                 paste(known, collapse = ", ")), call. = FALSE)
  }
  type <- known[hit]
  params <- list(...)
  if (identical(type, "Centered") && is.null(inner)) {
    stop("link type \"Centered\" needs an `inner` link", call. = FALSE)
  }
  if (!identical(type, "Centered") && !is.null(inner)) {
    stop(sprintf("`inner` is only used for type \"Centered\"; got type \"%s\"", type),
         call. = FALSE)
  }
  if (!is.null(inner) && !inherits(inner, "corehydro_link")) {
    stop("`inner` must be a corehydro_link object; create one with link_function()", call. = FALSE)
  }
  structure(list(type = type, parameters = if (length(params) == 0L) NULL else params,
                 inner = inner), class = "corehydro_link")
}

#' Evaluate a link function
#'
#' @param l a `corehydro_link` from [link_function()].
#' @param x,eta numeric vectors on the data and linear-predictor scales.
#' @return a numeric vector the same length as the input.
#' @examples
#' l <- link_function("Logit")
#' link(l, c(0.1, 0.5, 0.9))
#' @export
link <- function(l, x) link_eval(l, "link", x)

#' @rdname link
#' @export
link_inverse <- function(l, eta) link_eval(l, "inverse_link", eta)

#' @rdname link
#' @export
link_derivative <- function(l, x) link_eval(l, "d_link", x)

#' Available link function types
#'
#' Calls through to the C++ `link` toolbox group's own `"names"` method -- the same table
#' `link_function()` builds a link from (`link_builder_table()` in `link.hpp`) -- so this list
#' can't drift from what `link_function()` actually accepts.
#'
#' @return a character vector of the twelve `type` names [link_function()] accepts.
#' @examples
#' link_names()
#' @export
link_names <- function() toolbox_run("link", "names")$names

link_eval <- function(l, method, v) {
  if (!inherits(l, "corehydro_link")) {
    stop("`l` must be a corehydro_link object; create one with link_function()", call. = FALSE)
  }
  if (!is.numeric(v) || length(v) == 0L) {
    stop("input must be a non-empty numeric vector", call. = FALSE)
  }
  toolbox_run("link", method, list(v), list(link = unclass_link(l)))$values
}

# Internal: strip the class so to_spec_json() emits a plain nested object.
unclass_link <- function(l) {
  spec <- list(type = l$type, parameters = l$parameters)
  if (!is.null(l$inner)) spec$inner <- unclass_link(l$inner)
  spec
}

#' @export
print.corehydro_link <- function(x, ...) {
  cat(sprintf("<corehydro_link> %s\n", x$type))
  invisible(x)
}

# Internal: user-facing trend indices are 1-based, matching trend()'s own `start_index` doc
# (which is already 0-based and passed straight through -- see trend()); the time index this
# converts is instead the point at which the trend is EVALUATED, so it gets the same 1-based ->
# 0-based treatment mv_indices() gives dimension indices in R/mvdist.R, minus mv_indices's
# upper-bound and no-duplicate checks (a trend's time index is not bounded to a fixed dimension
# count and repeating one to predict the same point twice is not an error).
trend_index <- function(idx, what = "index") {
  if (!is.numeric(idx) || length(idx) == 0L || anyNA(idx)) {
    stop(sprintf("`%s` must be a non-empty numeric vector", what), call. = FALSE)
  }
  if (any(idx != trunc(idx))) {
    stop(sprintf("`%s` must be whole numbers; got %s", what,
                 paste(idx[idx != trunc(idx)], collapse = ", ")), call. = FALSE)
  }
  as.integer(idx) - 1L
}

#' Evaluate a trend model
#'
#' Builds the trend model named by `tr` (a [trend()] object) from its own class defaults, then
#' evaluates it at `index`. `tr`'s `parameter` field (which distribution parameter the trend
#' attaches to) is not used here -- this verb evaluates the trend model on its own, the same way
#' `set_trend_model()`'s type dispatch does before it further data-drives the parameter values
#' from a live model's distribution and data (this verb has no such model to consult, so a
#' `tr` with no explicit `values` predicts from the trend's own zero-valued class defaults).
#'
#' @param tr a `corehydro_trend` object from [trend()].
#' @param index a 1-based numeric vector of time indices to evaluate at.
#' @return a numeric vector, the same length as `index`.
#' @examples
#' tr <- trend("location", "Linear", start_index = 0, values = c(10, 2))
#' trend_predict(tr, 1:5)
#' @export
trend_predict <- function(tr, index) {
  if (!inherits(tr, "corehydro_trend")) {
    stop("`tr` must be a corehydro_trend object; create one with trend()", call. = FALSE)
  }
  idx <- trend_index(index, "index")
  toolbox_run("trend", "predict", list(idx), list(trend = unclass_trend(tr)))$values
}

#' @rdname trend_predict
#'
#' @return `trend_parameters()` returns a named numeric vector, the trend's own model-parameter
#'   values (see [model_parameter()]) at their class defaults, adjusted by any explicit `values`
#'   on `tr`.
#' @examples
#' trend_parameters(trend("location", "Linear", values = c(10, 2)))
#' @export
trend_parameters <- function(tr) {
  if (!inherits(tr, "corehydro_trend")) {
    stop("`tr` must be a corehydro_trend object; create one with trend()", call. = FALSE)
  }
  r <- toolbox_run("trend", "parameters", list(), list(trend = unclass_trend(tr)))
  stats::setNames(r$values, r$names)
}

#' Available trend model types
#'
#' Calls through to the C++ `trend` toolbox group's own `"names"` method -- the same table
#' [trend()] and `build_spec_trend()` (`model_spec.hpp`) validate `type` against
#' (`trend_model_type_table()`) -- so this list can't drift from what [trend()] actually
#' accepts.
#'
#' @return a character vector of the eleven `type` names [trend()] accepts.
#' @examples
#' trend_names()
#' @export
trend_names <- function() toolbox_run("trend", "names")$names

# Internal: strip the class and drop the model-attachment-only `parameter` field so
# to_spec_json() emits the plain {"type", "start_index"?, "values"?} object build_spec_trend()
# wants (numerics/support/toolbox/trend.hpp).
unclass_trend <- function(tr) {
  list(type = tr$type, start_index = tr$start_index, values = tr$values)
}
