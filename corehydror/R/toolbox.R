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

#' Correlation between two samples, or a correlation matrix
#'
#' Mirrors the C# `Correlation` class of the Numerics library: the paired-vector forms
#' (`Correlation.Pearson`/`Spearman`/`KendallsTau`, both `IList<double>` overloads) and the
#' Pearson/Spearman column-pairwise matrix overloads (`Pearson(double[,])`/`Spearman(double[,])`).
#'
#' @param x a numeric vector (paired-vector form, `y` required), or a numeric matrix or data
#'   frame (matrix form, `y` omitted) with one column per variable.
#' @param y a numeric vector of the same length as `x`, or `NULL` (the default) to compute the
#'   correlation matrix of `x`'s columns instead.
#' @param method one of `"pearson"` (the default), `"spearman"`, or `"kendall"`. `"kendall"` is
#'   rejected when `y` is `NULL`: upstream has no `KendallsTau(double[,])` overload, so there is
#'   no Kendall matrix form to compute.
#' @return with `y` given, a single numeric correlation coefficient; with `y` `NULL`, a numeric
#'   `ncol(x)`-by-`ncol(x)` matrix, dimnamed from `x`'s columns when `x` has column names.
#' @examples
#' x <- c(14, 8, 32, 7, 3, 15)
#' y <- c(10, 5, 7, 4, 3, 8)
#' correlation(x, y)
#' correlation(x, y, method = "kendall")
#' correlation(cbind(x, y))
#' @export
correlation <- function(x, y = NULL, method = c("pearson", "spearman", "kendall")) {
  method <- match.arg(method)
  if (is.null(y)) {
    if (method == "kendall") {
      stop("`method = \"kendall\"` has no matrix form; upstream Correlation has no ",
           "KendallsTau(double[,]) overload", call. = FALSE)
    }
    x <- as.matrix(x)
    if (!is.numeric(x)) {
      stop("`x` must be a numeric matrix or data frame when `y` is NULL", call. = FALSE)
    }
    # C4 (P4 whole-branch review): as.matrix() on a plain vector silently produces an n x 1
    # matrix rather than erroring, so `correlation(x)` on a bare vector used to return a 1x1
    # matrix of 1 (a vector's trivial self-correlation) instead of raising the way corehydropy
    # does for the same 1D input.
    if (ncol(x) < 2L) {
      stop("`x` must be a 2D array (observations in rows, variables in columns)", call. = FALSE)
    }
    p <- ncol(x)
    data <- lapply(seq_len(p), function(j) x[, j])
    r <- toolbox_run("correlation", paste0(method, "_matrix"), data)
    m <- matrix(r$values, nrow = r$dims[[1]], ncol = r$dims[[2]], byrow = TRUE)
    dimnames(m) <- list(colnames(x), colnames(x))
    return(m)
  }
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
#' Mirrors the C# `Linear`, `CubicSpline`, and `Polynomial` interpolaters of the Numerics
#' library. `x_transform`, `y_transform`, and `extrapolate` are Linear-only in C# (neither
#' `CubicSpline` nor `Polynomial` has a transform surface or an `Extrapolate()` method), so
#' they must be left at their defaults for `method = "cubic_spline"` or `"polynomial"`.
#'
#' @param x,y numeric vectors of equal length defining the knots.
#' @param xout numeric vector of positions to interpolate at.
#' @param method `"linear"` (the default), `"cubic_spline"`, or `"polynomial"`.
#' @param order the polynomial order -- there are `order + 1` terms for each polynomial
#'   function. Required when `method = "polynomial"`; must be `NULL` otherwise.
#' @param x_transform,y_transform one of `"none"` (the default), `"logarithmic"` (also accepted
#'   as `"log"` -- both spellings are equivalent, see the note below), or `"normal_z"`.
#'   Linear-only.
#' @param sort_order `"ascending"` (the default) or `"descending"`, describing `x`.
#' @param extrapolate whether to extend the end segments beyond the knots. Default `FALSE`,
#'   which clamps to the end knot, matching the C# `Interpolate()` default; `TRUE` calls the C#
#'   `Extrapolate()` method instead. Linear-only.
#' @return a numeric vector the same length as `xout`.
#' @note `x_transform`/`y_transform` accept both `"log"` and `"logarithmic"` for the same
#'   `Transform::Logarithmic` value everywhere a transform argument appears in this package
#'   ([curve_interpolate()], [tabular_function()], and here); `"logarithmic"` (matching the C#
#'   enum member name `Transform.Logarithmic`) is the spelling used in this package's own
#'   examples and documentation.
#' @examples
#' interpolate(c(1, 2, 3, 4), c(10, 20, 30, 40), c(1.5, 2.5))
#' interpolate(c(1, 2, 3, 4), c(10, 20, 30, 40), c(1.5, 2.5), method = "cubic_spline")
#' interpolate(c(1, 2, 3, 4), c(10, 20, 30, 40), c(1.5, 2.5), method = "polynomial", order = 3)
#' @export
interpolate <- function(x, y, xout, method = c("linear", "cubic_spline", "polynomial"),
                        order = NULL,
                        x_transform = c("none", "logarithmic", "log", "normal_z"),
                        y_transform = c("none", "logarithmic", "log", "normal_z"),
                        sort_order = c("ascending", "descending"), extrapolate = FALSE) {
  check_pair(x, y)
  if (!is.numeric(xout)) {
    stop("`xout` must be numeric", call. = FALSE)
  }
  method <- match.arg(method)
  x_transform <- match.arg(x_transform)
  y_transform <- match.arg(y_transform)
  sort_order <- match.arg(sort_order)
  if (method != "linear" &&
      (x_transform != "none" || y_transform != "none" || isTRUE(extrapolate))) {
    stop("`x_transform`, `y_transform`, and `extrapolate` are linear-only; the C# ",
         "CubicSpline/Polynomial classes have neither a transform surface nor an ",
         "Extrapolate() method", call. = FALSE)
  }
  if (method == "polynomial") {
    if (is.null(order)) {
      stop("`order` is required when method = \"polynomial\"", call. = FALSE)
    }
  } else if (!is.null(order)) {
    stop("`order` only applies when method = \"polynomial\"", call. = FALSE)
  }
  opts <- list(sort_order = sort_order)
  if (method == "linear") {
    opts <- c(opts, list(x_transform = x_transform, y_transform = y_transform,
                         extrapolate = isTRUE(extrapolate)))
  } else if (method == "polynomial") {
    opts <- c(opts, list(order = as.integer(order)))
  }
  toolbox_run("interpolation", method, list(x, y, xout), opts)$values
}

#' Interpolate a 2D grid (bilinear interpolation)
#'
#' Mirrors the C# `Bilinear` interpolater of the Numerics library.
#'
#' @param x1,x2 numeric vectors of grid coordinates.
#' @param y numeric matrix with `length(x1)` rows and `length(x2)` columns, `y[i, j]` the value
#'   at `(x1[i], x2[j])`.
#' @param x1out,x2out numeric vectors of equal length, positions to interpolate at.
#' @param x1_transform,x2_transform,y_transform one of `"none"` (the default), `"logarithmic"`
#'   (also accepted as `"log"`), or `"normal_z"`. See the note on [interpolate()].
#' @param sort_order `"ascending"` (the default) or `"descending"`, describing `x1` and `x2`.
#' @return a numeric vector the same length as `x1out`/`x2out`.
#' @examples
#' interpolate_2d(c(1, 2, 3), c(1, 2, 3), diag(3), c(1.5), c(1.5))
#' @export
interpolate_2d <- function(x1, x2, y, x1out, x2out,
                           x1_transform = c("none", "logarithmic", "log", "normal_z"),
                           x2_transform = c("none", "logarithmic", "log", "normal_z"),
                           y_transform = c("none", "logarithmic", "log", "normal_z"),
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

# The "linalg" toolbox group (P2 "math extras" Task 9): QRDecomposition (Householder
# reflections) and GaussJordanElimination. Matrices cross the runner boundary as ONE
# flattened row-major vector plus `rows`/`cols` options, the same convention
# interpolate_2d()'s bilinear `y` uses above -- since R matrices are stored column-major,
# every verb below flattens via `as.double(t(m))`, and every result comes back via
# `matrix(..., byrow = TRUE)`.

#' QR decomposition
#'
#' Mirrors the C# `QRDecomposition` class (Householder reflections): decomposes the `M x N`
#' matrix `a` into an `M x M` orthogonal matrix `q` and an `M x N` upper triangular matrix
#' `r` such that `q %*% r` reproduces `a`. The C# `RMatrix` property is named `r` here (`R`
#' has no naming collision in this package, unlike in the C# codebase).
#'
#' @param a a numeric matrix, M x N.
#' @return a list with elements `q` (M x M) and `r` (M x N).
#' @examples
#' a <- matrix(c(1, 0, 2, 1, 2, 5, 1, 5, -1), nrow = 3)
#' qr <- qr_decomposition(a)
#' qr$q %*% qr$r  # reproduces a
#' @export
qr_decomposition <- function(a) {
  a <- as.matrix(a)
  if (!is.numeric(a)) {
    stop("`a` must be a numeric matrix", call. = FALSE)
  }
  opts <- list(rows = as.integer(nrow(a)), cols = as.integer(ncol(a)))
  a_flat <- as.double(t(a))
  q <- toolbox_run("linalg", "qr_q", list(a_flat), opts)
  r <- toolbox_run("linalg", "qr_r", list(a_flat), opts)
  list(
    q = matrix(q$values, nrow = q$dims[1], ncol = q$dims[2], byrow = TRUE),
    r = matrix(r$values, nrow = r$dims[1], ncol = r$dims[2], byrow = TRUE)
  )
}

#' Solve a linear system by QR decomposition
#'
#' Mirrors the C# `QRDecomposition::Solve` overloads (vector and matrix right-hand sides).
#' `a` need not be square: an overdetermined system is solved in the least-squares sense; an
#' underdetermined system leaves the trailing `ncol(a) - min(nrow(a), ncol(a))` unknowns at
#' zero (matching `QRDecomposition.Solve`'s own truncated back-substitution, which only ever
#' fills indices below `min(m, n)`).
#'
#' @param a a numeric matrix, M x N.
#' @param b a numeric vector of length M, or a numeric matrix with M rows.
#' @return a numeric vector of length N when `b` is a vector, or an `N x ncol(b)` matrix when
#'   `b` is a matrix.
#' @examples
#' a <- matrix(c(1, 2, 3, 0, 1, 4, 5, 6, 0), nrow = 3, byrow = TRUE)
#' qr_solve(a, c(1, 2, 3))
#' @export
qr_solve <- function(a, b) {
  a <- as.matrix(a)
  if (!is.numeric(a)) {
    stop("`a` must be a numeric matrix", call. = FALSE)
  }
  rows <- nrow(a)
  cols <- ncol(a)
  a_flat <- as.double(t(a))
  opts <- list(rows = as.integer(rows), cols = as.integer(cols))
  if (is.matrix(b)) {
    if (nrow(b) != rows) {
      stop(sprintf("`b` must have %d rows, matching `a`; got %d", rows, nrow(b)), call. = FALSE)
    }
    opts$b_cols <- as.integer(ncol(b))
    b_flat <- as.double(t(b))
    r <- toolbox_run("linalg", "qr_solve_matrix", list(a_flat, b_flat), opts)
    matrix(r$values, nrow = r$dims[1], ncol = r$dims[2], byrow = TRUE)
  } else {
    if (!is.numeric(b) || length(b) != rows) {
      stop(sprintf("`b` must be a numeric vector of length %d, matching `a`'s rows; got %d",
                   rows, length(b)), call. = FALSE)
    }
    toolbox_run("linalg", "qr_solve", list(a_flat, as.double(b)), opts)$values
  }
}

#' Gauss-Jordan elimination
#'
#' Mirrors the C# `GaussJordanElimination::Solve(ref Matrix A, ref Matrix B)`: one full-pivot
#' Gauss-Jordan reduction that produces `a`'s inverse and, when `b` is supplied, the solution
#' set of `a %*% x = b`. Unlike the C# `ref, ref` in-place API, `a` and `b` are never mutated
#' in R -- both results come back as new matrices.
#'
#' @param a a square numeric matrix, N x N.
#' @param b a numeric matrix with N rows (the right-hand sides). Omit for the inverse alone,
#'   in which case `solution` is an `N x 0` matrix.
#' @return a list with elements `inverse` (`a`'s inverse, N x N) and `solution` (`N x ncol(b)`).
#' @examples
#' a <- matrix(c(1, 3, 3, 1, 4, 3, 1, 3, 4), nrow = 3, byrow = TRUE)
#' gauss_jordan(a)$inverse
#' @export
gauss_jordan <- function(a, b = NULL) {
  a <- as.matrix(a)
  if (!is.numeric(a) || nrow(a) != ncol(a)) {
    stop("`a` must be a square numeric matrix", call. = FALSE)
  }
  n <- nrow(a)
  a_flat <- as.double(t(a))
  if (is.null(b)) {
    b_flat <- double(0)
    b_cols <- 0L
  } else {
    b <- as.matrix(b)
    if (!is.numeric(b) || nrow(b) != n) {
      stop(sprintf("`b` must have %d rows, matching `a`; got %d", n, nrow(b)), call. = FALSE)
    }
    b_flat <- as.double(t(b))
    b_cols <- as.integer(ncol(b))
  }
  opts <- list(rows = as.integer(n), cols = as.integer(n), b_cols = b_cols)
  inv <- toolbox_run("linalg", "gauss_jordan_inverse", list(a_flat, b_flat), opts)
  sol <- toolbox_run("linalg", "gauss_jordan_solution", list(a_flat, b_flat), opts)
  list(
    inverse = matrix(inv$values, nrow = inv$dims[1], ncol = inv$dims[2], byrow = TRUE),
    solution = matrix(sol$values, nrow = sol$dims[1], ncol = sol$dims[2], byrow = TRUE)
  )
}

# The "special" toolbox group (P2 "math extras" Task 10): the ported Debye and Evaluate special
# functions. Both verbs vectorize over `x`, returning one value per element -- there is no
# matrix result here, so no flatten/reassemble step like `linalg`'s above.

#' The Debye function
#'
#' Mirrors the C# `Debye.Function`: a piecewise approximation of
#' `D(x) = (n/x^n) * integral_0^x t^n / (e^t - 1) dt`, vectorized over `x`.
#'
#' @param x a non-negative numeric vector.
#' @return a numeric vector the same length as `x`.
#' @examples
#' debye(0.5)
#' debye(c(0.1, 1, 10))
#' @export
debye <- function(x) {
  toolbox_run("special", "debye", list(as.double(x)))$values
}

#' Evaluate a polynomial
#'
#' Mirrors the C# `Evaluate` class's three polynomial evaluators (Horner's method), vectorized
#' over `x` against one shared `coefficients` vector: `variant = "standard"` calls
#' `Evaluate.Polynomial` (coefficients in ASCENDING order, `coefficients[1]` the constant term);
#' `"reverse"` calls `Evaluate.PolynomialRev` (coefficients in DESCENDING order,
#' `coefficients[1]` the highest-order term), optionally truncated to order `n + 1` via `n`;
#' `"reverse_unit"` calls `Evaluate.PolynomialRev_1` (DESCENDING order with an implicit leading
#' coefficient of 1).
#'
#' @param coefficients a numeric vector of polynomial coefficients.
#' @param x a numeric vector, the points at which to evaluate.
#' @param variant one of `"standard"` (default), `"reverse"`, or `"reverse_unit"`.
#' @param n an optional integer redefining the polynomial's order to `n + 1`; only valid with
#'   `variant = "reverse"`.
#' @return a numeric vector the same length as `x`.
#' @examples
#' polynomial_eval(c(3, 5, 7), 4)
#' polynomial_eval(c(3, 5, 7), 4, variant = "reverse")
#' polynomial_eval(c(3, 5, 7), 4, variant = "reverse", n = 1)
#' @export
polynomial_eval <- function(coefficients, x, variant = c("standard", "reverse", "reverse_unit"),
                            n = NULL) {
  variant <- match.arg(variant)
  if (!is.null(n) && variant != "reverse") {
    stop("`n` is only valid with variant = \"reverse\"", call. = FALSE)
  }
  method <- switch(variant,
    standard = "polynomial",
    reverse = "polynomial_rev",
    reverse_unit = "polynomial_rev_1"
  )
  opts <- if (is.null(n)) list() else list(n = as.integer(n))
  toolbox_run("special", method, list(as.double(coefficients), as.double(x)), opts)$values
}

# The "functions" toolbox group (P2 "math extras" Task 11): the two non-tabular
# IUnivariateFunction implementations (numerics/functions/), LinearFunction and PowerFunction.
# The severed third implementation, TabularFunction, depends on the unported Paired Data
# subsystem (see upstream/CLAUDE.md) and is not exposed.

#' Evaluate a univariate function
#'
#' Mirrors the Numerics `LinearFunction` (`Y = alpha + beta*X + epsilon`) and `PowerFunction`
#' (`Y = alpha * (X - xi)^beta * epsilon`), both over optional normally distributed noise
#' (`epsilon ~ Normal(0, sigma)`) via `confidence_level`. `is_inverse` (PowerFunction's own
#' `IsInverse` switch) selects which of the forward power law or its algebraic inverse
#' `Function()`/`inverse = TRUE` evaluates -- an independent axis from `inverse` itself, which
#' picks `Function()` vs. `InverseFunction()` on whichever of the two `is_inverse` selects.
#'
#' @param type `"linear"` or `"power"`, matched case-insensitively.
#' @param parameters a numeric vector: `c(alpha, beta, sigma)` for `"linear"`; `c(alpha, beta,
#'   xi, sigma)` for `"power"`. `sigma` is still required (e.g. 0) when `confidence_level` is
#'   `NULL` -- it only enters the calculation on the non-deterministic path.
#' @param x a numeric vector: the values to evaluate the function at, or (when `inverse = TRUE`)
#'   the values to evaluate the inverse function at.
#' @param inverse if `TRUE`, evaluates the inverse function (`InverseFunction()`) instead of the
#'   forward function (`Function()`).
#' @param is_inverse `"power"`-only: `PowerFunction`'s own `IsInverse` property. An error for
#'   `type = "linear"`.
#' @param confidence_level if given, evaluates the non-deterministic path at this quantile level;
#'   if `NULL` (default), evaluates deterministically.
#' @return a numeric vector the same length as `x`.
#' @examples
#' univariate_function("linear", c(0, 1, 0), c(1, 2, 3))
#' univariate_function("power", c(5, 2, 0, 3), 6)
#' univariate_function("power", c(5, 2, 0, 3), 6, confidence_level = 0.75)
#' @export
univariate_function <- function(type, parameters, x, inverse = FALSE, is_inverse = FALSE,
                                 confidence_level = NULL) {
  known <- c("linear", "power")
  hit <- match(tolower(type), known)
  if (is.na(hit)) {
    stop(sprintf("unknown function type \"%s\". Available: %s", type,
                 paste(known, collapse = ", ")), call. = FALSE)
  }
  type <- known[hit]
  if (isTRUE(is_inverse) && !identical(type, "power")) {
    stop(sprintf("`is_inverse` is only used for type \"power\"; got type \"%s\"", type),
         call. = FALSE)
  }
  if (!is.numeric(parameters) || length(parameters) == 0L) {
    stop("`parameters` must be a non-empty numeric vector", call. = FALSE)
  }
  if (!is.numeric(x) || length(x) == 0L) {
    stop("`x` must be a non-empty numeric vector", call. = FALSE)
  }
  opts <- list(parameters = as.double(parameters))
  opts[["function"]] <- type
  if (identical(type, "power")) opts$is_inverse <- isTRUE(is_inverse)
  if (!is.null(confidence_level)) opts$confidence_level <- as.double(confidence_level)
  method <- if (isTRUE(inverse)) "inverse" else "evaluate"
  toolbox_run("functions", method, list(as.double(x)), opts)$values
}

# The "network" toolbox group (P3 optimizers Task 10): the Dijkstra shortest-path solver over an
# edge list. Edges cross the runner boundary as four parallel numeric vectors -- from, to,
# weight, edge index -- with the destinations in the options object, and the result table comes
# back flattened row-major with dims = {node_count, 3}, the same convention `linalg`'s matrix
# results use above.

# Internal: reject the mistakes an edge list can make, naming the argument. Returns the
# validated, coerced edge index vector.
check_edges <- function(from, to, weight, edge_index) {
  if (!is.numeric(from) || !is.numeric(to) || !is.numeric(weight)) {
    stop("`from`, `to` and `weight` must be numeric vectors", call. = FALSE)
  }
  n <- length(from)
  if (n == 0L) {
    stop("`from`, `to` and `weight` must describe at least one edge", call. = FALSE)
  }
  if (length(to) != n || length(weight) != n) {
    stop(sprintf(paste0("`from`, `to` and `weight` must have the same length; ",
                        "got %d, %d and %d"), n, length(to), length(weight)), call. = FALSE)
  }
  if (is.null(edge_index)) {
    return(as.double(seq_len(n) - 1L))
  }
  if (!is.numeric(edge_index) || length(edge_index) != n) {
    stop(sprintf("`edge_index` must be a numeric vector the same length as `from`; got %d for %d",
                 length(edge_index), n), call. = FALSE)
  }
  as.double(edge_index)
}

# Internal: a node index is a whole, non-negative number. The shared C++ group header checks this
# too (a fixture case reaches it directly), but checking here as well keeps the message naming
# the R argument and keeps the two packages' errors identical.
check_node_indices <- function(x, what) {
  if (any(!is.finite(x)) || any(x != floor(x)) || any(x < 0)) {
    stop(sprintf("`%s` must be whole, non-negative node indices", what), call. = FALSE)
  }
  invisible(NULL)
}

#' Shortest paths through a network
#'
#' Mirrors the C# `Dijkstra.Solve` overloads: solves the cheapest route from EVERY node of a
#' directed, weighted graph to a set of destination nodes at once, running the search backwards
#' from the destinations. The answer is a routing table -- for each node, which neighbour to step
#' to, along which edge, and at what remaining cost.
#'
#' Node indices are 0-based in both `corehydror` and `corehydropy`, matching the C# result table
#' the two packages share; a graph with `n` nodes uses indices `0` to `n - 1`. Unreachable nodes
#' carry `cost = Inf` with `next_node = -1` and `edge_index = -1`, and a destination node carries
#' `cost = 0` with `next_node` equal to its own index.
#'
#' Costs accumulate in single precision, because the ported solver does (C# declares
#' `float Weight` and its own tests assert the table by exact `float` equality). Fractional
#' weights therefore round to `float` before they are summed.
#'
#' @param from,to,weight numeric vectors of the same length, one element per directed edge:
#'   the start node index, the end node index, and the cost of traversing the edge. `from` and
#'   `to` must be whole, non-negative numbers.
#' @param destinations a numeric vector of one or more destination node indices. With several
#'   destinations, each node keeps whichever destination it reaches most cheaply.
#' @param edge_index an optional numeric vector the same length as `from`, labelling each edge
#'   (typically an index into whatever the edges came from -- a river reach, a road segment).
#'   Defaults to `0:(length(from) - 1)`. These labels are what the `edge_index` result column
#'   reports, and they need not be distinct.
#' @param node_count an optional node count. Defaults to `max(from, to) + 1`; supply a larger
#'   value to include isolated nodes carrying no edge, which then report `cost = Inf`. A value
#'   below `max(from, to) + 1` is an error: the graph would not fit the routing table it asks for.
#' @return a data frame with one row per node, in node-index order, and columns `next_node`,
#'   `edge_index` (both integer) and `cost` (numeric).
#' @examples
#' # 0 -> 1 -> 2, plus a disconnected node 3
#' shortest_path(
#'   from = c(0, 1),
#'   to = c(1, 2),
#'   weight = c(1, 1),
#'   destinations = 2,
#'   node_count = 4
#' )
#' @export
shortest_path <- function(from, to, weight, destinations, edge_index = NULL, node_count = NULL) {
  edge_index <- check_edges(from, to, weight, edge_index)
  if (!is.numeric(destinations)) {
    stop("`destinations` must be a numeric vector of node indices", call. = FALSE)
  }
  if (length(destinations) == 0L) {
    stop("`destinations` must name at least one destination node", call. = FALSE)
  }
  check_node_indices(from, "from")
  check_node_indices(to, "to")
  check_node_indices(destinations, "destinations")
  opts <- list(destinations = spec_array(as.double(destinations)))
  n_nodes <- max(from, to) + 1
  if (!is.null(node_count)) {
    if (!is.numeric(node_count) || length(node_count) != 1L || node_count < 1) {
      stop("`node_count` must be a single positive number", call. = FALSE)
    }
    n_nodes <- as.double(node_count)
    # A `node_count` below `max(from, to) + 1` cannot describe the edge list. The ported solver
    # raises the C# IndexOutOfRangeException message from inside itself when it reaches the
    # offending index, which is both late and unhelpful here, so reject it up front and name the
    # argument. This is deliberately STRICTER than the C# solver, whose bounds check is lazy: it
    # accepts a too-small count as long as no out-of-range index is ever reached (see
    # dijkstra.hpp note 9). That input is a graph the caller cannot have meant.
    if (n_nodes < max(from, to) + 1) {
      stop(sprintf(paste0("`node_count` must be at least %d, the number of nodes `from` and `to` ",
                          "describe; got %d"),
                   as.integer(max(from, to) + 1), as.integer(n_nodes)), call. = FALSE)
    }
    opts$node_count <- n_nodes
  }
  if (any(destinations >= n_nodes)) {
    stop(sprintf("`destinations` is out of range for a network of %d nodes", as.integer(n_nodes)),
         call. = FALSE)
  }
  r <- toolbox_run(
    "network", "dijkstra",
    list(from, to, weight, edge_index),
    opts
  )
  m <- matrix(r$values, nrow = r$dims[[1]], ncol = r$dims[[2]], byrow = TRUE)
  data.frame(
    next_node = as.integer(m[, 1]),
    edge_index = as.integer(m[, 2]),
    cost = m[, 3]
  )
}

# The "hypothesis" toolbox group (P4 Task 3, completed in P5): the thirteen ported hypothesis
# tests over
# numerics/data/hypothesis_tests.hpp (a port of the C# `HypothesisTests` static class). Mirrors
# corehydropy's own hypothesis_test() verb; both packages share this signature and produce
# identical error text so a change here is not one-sided.

.hypothesis_methods <- c(
  "one_sample_t", "equal_variance_t", "unequal_variance_t", "paired_t", "f", "f_models",
  "jarque_bera", "wald_wolfowitz", "ljung_box", "mann_whitney", "mann_kendall", "linear_trend",
  "unimodality"
)
.hypothesis_two_sample <- c("equal_variance_t", "unequal_variance_t", "paired_t", "f", "mann_whitney")

#' Hypothesis tests
#'
#' Mirrors the C# `HypothesisTests` static class: thirteen one- and two-sample parametric and
#' nonparametric hypothesis tests, reached through the shared `hypothesis` toolbox group. Every
#' method but `"f_models"` returns the 2-sided p-value of its test statistic; `"f_models"` (the
#' F-test comparing two nested regression models) additionally returns the F statistic itself.
#'
#' @details
#' Argument use by method, and the C# guard each one inherits:
#' * `"one_sample_t"`: `x`, `population_mean` (default 0). Needs at least 2 observations.
#' * `"equal_variance_t"` / `"unequal_variance_t"`: `x`, `y`. `equal_variance_t` needs a
#'   combined length of at least 3; `unequal_variance_t` has no length guard (upstream has none
#'   either).
#' * `"paired_t"`: `x`, `y`, which must be the same length.
#' * `"f"`: `x`, `y`, each needing at least 2 observations.
#' * `"f_models"`: `sse_restricted`, `sse_full`, `df_restricted`, `df_full` (all required; `x`
#'   and `y` are ignored). `df_restricted` must differ from `df_full`, and `df_full` must be
#'   positive.
#' * `"jarque_bera"` / `"wald_wolfowitz"`: `x`. No length guard.
#' * `"ljung_box"`: `x`, `lag_max` (default `NULL`, meaning
#'   `floor(min(10 * log10(length(x)), length(x) - 1))`, the C# default rule).
#' * `"mann_whitney"`: `x`, `y`. `x` must be no longer than `y`, each must have more than 3
#'   observations, and the combined length must exceed 20.
#' * `"mann_kendall"`: `x`. Needs at least 10 observations.
#' * `"linear_trend"`: `x` (the sample), `index` (default `seq_along(x)`, i.e. `1:length(x)` --
#'   a VALUE the regression is fit against, not an index into `x`). `index` and `x` must be the
#'   same length.
#' * `"unimodality"`: `x`. Needs at least 10 observations. Fits a 1-component and a 2-component
#'   Gaussian mixture model (both at the hard-coded seed 12345, so the result is deterministic)
#'   and returns the p-value of the likelihood-ratio statistic against a chi-square with 3
#'   degrees of freedom, so a SMALL p-value is evidence against unimodality. If either mixture
#'   fit fails numerically the result is `NaN` rather than an error, matching upstream.
#'
#' @param x numeric vector: the sample (the two-sample methods' first sample, or the response
#'   series for `"linear_trend"`). Ignored for `"f_models"`.
#' @param y numeric vector, the second sample. Required for the two-sample methods listed above,
#'   and rejected (must be `NULL`) for every other method -- earlier versions silently discarded
#'   a `y` supplied to a one-sample method instead of raising.
#' @param method one of `"one_sample_t"`, `"equal_variance_t"`, `"unequal_variance_t"`,
#'   `"paired_t"`, `"f"`, `"f_models"`, `"jarque_bera"`, `"wald_wolfowitz"`, `"ljung_box"`,
#'   `"mann_whitney"`, `"mann_kendall"`, `"linear_trend"`, `"unimodality"`.
#' @param population_mean the hypothesized mean for `"one_sample_t"`. Default 0.
#' @param lag_max the max lag for `"ljung_box"`. Default `NULL` (use the C# default rule).
#' @param index the index (x-axis) vector for `"linear_trend"`. Default `NULL`, meaning
#'   `seq_along(x)`.
#' @param sse_restricted,sse_full,df_restricted,df_full the four `"f_models"` inputs: the
#'   restricted and full models' sum of squared errors and degrees of freedom.
#' @return a named numeric vector: `p_value` for every method but `"f_models"`, which returns
#'   `c(f_statistic =, p_value =)`.
#' @examples
#' hypothesis_test(c(4, 5, 5, 6, 9, 12, 13, 14, 14, 19, 22, 24, 25), method = "jarque_bera")
#' @export
hypothesis_test <- function(x = NULL, y = NULL, method = "jarque_bera", population_mean = 0,
                            lag_max = NULL, index = NULL, sse_restricted = NULL, sse_full = NULL,
                            df_restricted = NULL, df_full = NULL) {
  if (!method %in% .hypothesis_methods) {
    stop(sprintf("`method` must be one of %s; got \"%s\"",
                 paste(sprintf("\"%s\"", .hypothesis_methods), collapse = ", "), method),
         call. = FALSE)
  }
  if (method %in% .hypothesis_two_sample && is.null(y)) {
    stop(sprintf("`y` is required for method \"%s\"", method), call. = FALSE)
  }
  # C2 (P4 whole-branch review): a one-sample method silently DISCARDED a non-NULL `y` --
  # `hypothesis_test(a, b)` at the default method equalled `hypothesis_test(a)`. Reject it
  # instead, the way R/callback.R:1028 rejects a delegate that does not belong to the chosen
  # sampler.
  if (!method %in% .hypothesis_two_sample && !is.null(y)) {
    stop(sprintf("`y` is not used by method \"%s\"; leave it NULL", method), call. = FALSE)
  }

  if (identical(method, "f_models")) {
    if (is.null(sse_restricted) || is.null(sse_full) || is.null(df_restricted) || is.null(df_full)) {
      stop(paste0("`sse_restricted`, `sse_full`, `df_restricted`, and `df_full` are all ",
                  "required for method \"f_models\""), call. = FALSE)
    }
    r <- toolbox_run("hypothesis", "f_models", list(), list(
      sse_restricted = as.double(sse_restricted),
      sse_full = as.double(sse_full),
      df_restricted = as.double(df_restricted),
      df_full = as.double(df_full)
    ))
    return(stats::setNames(r$values, r$names))
  }

  if (identical(method, "linear_trend")) {
    idx <- if (is.null(index)) seq_along(x) else index
    r <- toolbox_run("hypothesis", "linear_trend", list(as.double(idx), as.double(x)))
    return(stats::setNames(r$values, "p_value"))
  }

  data <- if (method %in% .hypothesis_two_sample) {
    list(as.double(x), as.double(y))
  } else {
    list(as.double(x))
  }
  opts <- switch(method,
    one_sample_t = list(population_mean = as.double(population_mean)),
    ljung_box = list(lag_max = as.double(if (is.null(lag_max)) -1 else lag_max)),
    list()
  )
  r <- toolbox_run("hypothesis", method, data, opts)
  stats::setNames(r$values, "p_value")
}

# The "paired_data" toolbox group (P4 Task 10): OrderedPairedData/UncertainOrderedPairedData/
# LineSimplification (numerics/data/paired_data/, P4 Tasks 7-9), plus TabularFunction's
# tabular/tabular_inverse arm on the "functions" group. Every curve verb below reads the same
# strict_x/strict_y/order_x/order_y shape contract paired_data.hpp documents;
# curve_interpolate() additionally reads x_transform/y_transform. Only five verbs are exported --
# line_simplify, search, and is_valid are reachable through the fixture/oracle-gate surface but
# are not given an R-facing wrapper by this task.

# The order_x/order_y/x_transform/y_transform choices every curve verb below validates against,
# declared once. `"logarithmic"` (matching the C# enum member name `Transform.Logarithmic`) is
# this package's documented spelling; `"log"` is accepted as an equivalent alias (both parse to
# the same value core-side) so a value valid for interpolate()/interpolate_2d() is also valid
# here -- see the P4 whole-branch-review finding M2.
.paired_data_orders <- c("ascending", "descending", "none")
.paired_data_transforms <- c("none", "logarithmic", "log", "normal_z")

# Internal: build a `distributions` list of corehydro_dist objects, recycling a single one across
# every `x`. Shared by uncertain_curve_sample() and tabular_function(), which use identical
# recycling and error text.
paired_data_distributions <- function(distributions, x) {
  if (inherits(distributions, "corehydro_dist")) distributions <- list(distributions)
  if (!is.list(distributions) || length(distributions) == 0L ||
      !all(vapply(distributions, inherits, logical(1), "corehydro_dist"))) {
    stop("`distributions` must be a corehydro_dist or a list of corehydro_dist objects",
         call. = FALSE)
  }
  if (length(distributions) == 1L && length(x) > 1L) {
    distributions <- rep(distributions, length(x))
  }
  if (length(distributions) != length(x)) {
    stop(sprintf("`distributions` must have length 1 or length(x) (%d); got %d",
                 length(x), length(distributions)), call. = FALSE)
  }
  distributions
}

#' Interpolate a paired x-y curve
#'
#' Mirrors the C# `OrderedPairedData.GetYFromX`/`GetXFromY`: linear interpolation over a curve
#' that keeps itself sorted/validated against a caller-chosen monotonicity contract, with
#' optional per-axis transforms (log10 or the standard normal z-score) applied before/after
#' interpolation. Exactly one of `xout`/`yout` must be supplied.
#'
#' @param x,y numeric vectors of equal length (at least two), the curve's ordinates.
#' @param xout numeric vector of x positions to interpolate y at.
#' @param yout numeric vector of y positions to interpolate x at.
#' @param x_transform,y_transform one of `"none"` (default), `"logarithmic"` (also accepted as
#'   `"log"`), or `"normal_z"`.
#' @param order_x,order_y one of `"ascending"` (default), `"descending"`, or `"none"`.
#' @param strict_x,strict_y require x/y to strictly increase/decrease (per `order_x`/`order_y`)
#'   between consecutive ordinates. Default `TRUE`.
#' @return a numeric vector, the same length as whichever of `xout`/`yout` was supplied.
#' @examples
#' curve_interpolate(c(50, 100, 150, 200, 250), c(100, 200, 300, 400, 500), xout = 75)
#' @export
curve_interpolate <- function(x, y, xout = NULL, yout = NULL,
                              x_transform = "none", y_transform = "none",
                              order_x = "ascending", order_y = "ascending",
                              strict_x = TRUE, strict_y = TRUE) {
  check_pair(x, y)
  if (is.null(xout) == is.null(yout)) {
    stop("exactly one of `xout` or `yout` must be supplied", call. = FALSE)
  }
  # check_choice() rather than match.arg(): see R/fit.R:22 and the P4 whole-branch-review
  # finding M3 -- match.arg's prefix matching would silently accept "log" as a prefix of
  # "logarithmic" rather than the (also valid) full "log" token, and would accept ANY
  # unambiguous prefix Python does not.
  x_transform <- check_choice(x_transform, .paired_data_transforms, "x_transform")
  y_transform <- check_choice(y_transform, .paired_data_transforms, "y_transform")
  order_x <- check_choice(order_x, .paired_data_orders, "order_x")
  order_y <- check_choice(order_y, .paired_data_orders, "order_y")
  opts <- list(strict_x = isTRUE(strict_x), strict_y = isTRUE(strict_y),
               order_x = order_x, order_y = order_y,
               x_transform = x_transform, y_transform = y_transform)
  if (!is.null(xout)) {
    if (!is.numeric(xout)) {
      stop("`xout` must be numeric", call. = FALSE)
    }
    toolbox_run("paired_data", "interpolate_y", list(x, y, xout), opts)$values
  } else {
    if (!is.numeric(yout)) {
      stop("`yout` must be numeric", call. = FALSE)
    }
    toolbox_run("paired_data", "interpolate_x", list(x, y, yout), opts)$values
  }
}

#' Area under a paired x-y curve
#'
#' Mirrors the C# `OrderedPairedData.TrapezoidalAreaUnderY`/`TrapezoidalAreaUnderX`: the
#' trapezoidal-rule area between the curve and the x-axis (`under = "y"`) or the y-axis
#' (`under = "x"`). Requires x (for `under = "y"`) or y (for `under = "x"`) to be sorted
#' ascending or descending -- `order_x`/`order_y = "none"` raises the same error the C# method
#' does.
#'
#' @inheritParams curve_interpolate
#' @param under one of `"y"` (default, area under the curve against the x-axis) or `"x"` (area
#'   against the y-axis).
#' @return a single number.
#' @examples
#' curve_area(c(1, 2, 3, 4), c(1, 4, 9, 16))
#' @export
curve_area <- function(x, y, under = "y",
                       order_x = "ascending", order_y = "ascending",
                       strict_x = TRUE, strict_y = TRUE) {
  check_pair(x, y)
  under <- check_choice(under, c("y", "x"), "under")
  order_x <- check_choice(order_x, .paired_data_orders, "order_x")
  order_y <- check_choice(order_y, .paired_data_orders, "order_y")
  opts <- list(strict_x = isTRUE(strict_x), strict_y = isTRUE(strict_y),
               order_x = order_x, order_y = order_y)
  method <- if (identical(under, "y")) "area_under_y" else "area_under_x"
  toolbox_run("paired_data", method, list(x, y), opts)$values[[1]]
}

#' Simplify a paired x-y curve
#'
#' Mirrors the C# `OrderedPairedData`'s three curve-simplification algorithms: Douglas-Peucker
#' (`method = "rdp"`, needs `tolerance`), Visvalingam-Whyatt (`method = "visvalingam"`, needs
#' `num_to_keep`), and Lang (`method = "lang"`, needs `tolerance` and `look_ahead`). NOTE: unlike
#' `rdp`/`visvalingam`, which always keep the curve's first and last point, `lang` does not
#' force-keep the trailing point -- a real, verified-against-the-real-C#-library upstream
#' behavior (see `ordered_paired_data.hpp`'s sixth transcription note), not a port bug.
#'
#' @inheritParams curve_interpolate
#' @param method one of `"rdp"` (default), `"visvalingam"`, or `"lang"`.
#' @param tolerance perpendicular-distance tolerance; required for `method` `"rdp"` or `"lang"`.
#' @param num_to_keep number of points to keep; required for `method = "visvalingam"`, and must be
#'   at least 2 (the algorithm always keeps the curve's first and last point, and needs at least
#'   3 ordinates to triangulate at every intermediate step).
#' @param look_ahead the Lang algorithm's look-ahead window; required for `method = "lang"`.
#' @return a data frame with columns `x` and `y`.
#' @examples
#' x <- c(0, 1.57, 3.14, 4.71, 6.28)
#' y <- c(0, 1, 0, -1, 0)
#' curve_simplify(x, y, method = "rdp", tolerance = 0.01, strict_y = FALSE, order_y = "none")
#' @export
curve_simplify <- function(x, y, method = "rdp", tolerance = NULL, num_to_keep = NULL,
                           look_ahead = NULL,
                           order_x = "ascending", order_y = "ascending",
                           strict_x = TRUE, strict_y = TRUE) {
  check_pair(x, y)
  if (!method %in% c("rdp", "visvalingam", "lang")) {
    stop(sprintf("`method` must be one of \"rdp\", \"visvalingam\", \"lang\"; got \"%s\"", method),
         call. = FALSE)
  }
  order_x <- check_choice(order_x, .paired_data_orders, "order_x")
  order_y <- check_choice(order_y, .paired_data_orders, "order_y")
  opts <- list(strict_x = isTRUE(strict_x), strict_y = isTRUE(strict_y),
               order_x = order_x, order_y = order_y, algorithm = method)
  if (identical(method, "rdp")) {
    if (is.null(tolerance)) {
      stop("`tolerance` is required when method = \"rdp\"", call. = FALSE)
    }
    opts$tolerance <- as.double(tolerance)
  } else if (identical(method, "visvalingam")) {
    if (is.null(num_to_keep)) {
      stop("`num_to_keep` is required when method = \"visvalingam\"", call. = FALSE)
    }
    # Visvalingam-Whyatt always keeps the curve's first and last point, so it needs at least
    # 3 ordinates to triangulate; below that the C++ layer throws (matching the C# List<T>
    # indexer's ArgumentOutOfRangeException) rather than reading out of bounds. Checked here too
    # so the caller gets one clear message instead of the runner's internal one.
    if (!is.numeric(num_to_keep) || length(num_to_keep) != 1L || num_to_keep < 2) {
      stop("`num_to_keep` must be a single integer of at least 2", call. = FALSE)
    }
    opts$num_to_keep <- as.integer(num_to_keep)
  } else {
    if (is.null(tolerance) || is.null(look_ahead)) {
      stop("`tolerance` and `look_ahead` are both required when method = \"lang\"", call. = FALSE)
    }
    opts$tolerance <- as.double(tolerance)
    opts$look_ahead <- as.integer(look_ahead)
  }
  r <- toolbox_run("paired_data", "simplify", list(x, y), opts)
  m <- matrix(r$values, nrow = r$dims[[1]], ncol = r$dims[[2]], byrow = TRUE)
  data.frame(x = m[, 1], y = m[, 2])
}

#' Sample an uncertain paired curve
#'
#' Mirrors the C# `UncertainOrderedPairedData.CurveSample()`/`CurveSample(double)`: collapses a
#' curve whose Y-coordinate is a whole distribution at each x down to a plain x-y curve, either
#' at the distributions' means (`probability = NULL`, the default) or at a shared quantile
#' (`probability` in `[0, 1]`).
#'
#' @param x numeric vector of x positions, at least one element.
#' @param distributions a [distribution()] object, or a list of them the same length as `x` --
#'   one distribution per x position. A single distribution is recycled across every `x`.
#' @param probability quantile in `[0, 1]` to sample at; `NULL` (default) samples the mean.
#'   Values outside `[0, 1]` are rejected -- the underlying C# `CurveSample(double)` silently
#'   clamps them instead, so this range check is enforced here rather than core-side.
#' @param order_x,order_y one of `"ascending"` (default), `"descending"`, or `"none"`.
#' @param strict_x,strict_y require x/y to strictly increase/decrease (per `order_x`/`order_y`)
#'   between consecutive ordinates. Default `TRUE`.
#' @return a data frame with columns `x` and `y`.
#' @examples
#' x <- c(1, 2, 3, 5)
#' d <- list(distribution("Triangular", c(1, 2, 3)), distribution("Triangular", c(2, 4, 5)),
#'           distribution("Triangular", c(6, 8, 12)), distribution("Triangular", c(13, 19, 20)))
#' uncertain_curve_sample(x, d, probability = 0.5)
#' @export
uncertain_curve_sample <- function(x, distributions, probability = NULL,
                                   order_x = "ascending", order_y = "ascending",
                                   strict_x = TRUE, strict_y = TRUE) {
  if (!is.numeric(x) || length(x) == 0L) {
    stop("`x` must be a non-empty numeric vector", call. = FALSE)
  }
  distributions <- paired_data_distributions(distributions, x)
  order_x <- check_choice(order_x, .paired_data_orders, "order_x")
  order_y <- check_choice(order_y, .paired_data_orders, "order_y")
  opts <- list(strict_x = isTRUE(strict_x), strict_y = isTRUE(strict_y),
               order_x = order_x, order_y = order_y, distributions = distributions)
  if (!is.null(probability)) {
    # C3 (P4 whole-branch review): the core clamp (uncertain_ordered_paired_data.hpp) is a
    # faithful port of C# CurveSample(double), which silently clamps an out-of-range quantile
    # instead of raising -- `probability = 50` silently returned the 100% quantile. That core
    # behavior is untouched; this host-layer range check is what actually rejects the mistake.
    if (!is.numeric(probability) || length(probability) != 1L ||
        probability < 0 || probability > 1) {
      stop("`probability` must be a single number in [0, 1]", call. = FALSE)
    }
    opts$probability <- as.double(probability)
  }
  r <- toolbox_run("paired_data", "curve_sample", list(x), opts)
  m <- matrix(r$values, nrow = r$dims[[1]], ncol = r$dims[[2]], byrow = TRUE)
  data.frame(x = m[, 1], y = m[, 2])
}

#' Evaluate a tabular function
#'
#' Mirrors the C# `TabularFunction`: builds an uncertain paired curve from `x` and
#' `distributions`, samples it once (the mean curve, or `confidence_level` if given), and
#' evaluates `Function()`/`InverseFunction()` at `at`. Unlike [curve_interpolate()] and friends,
#' the underlying curve's shape contract is not configurable here -- `TabularFunction` is always
#' built strict, ascending on both axes, matching every use in the ported C# test suite.
#'
#' @param x numeric vector, the curve's x positions, at least one element.
#' @param distributions a [distribution()] object, or a list of them the same length as `x` --
#'   one distribution per x position. A single distribution is recycled across every `x`.
#' @param at numeric vector of points to evaluate at (or, when `inverse = TRUE`, points to
#'   evaluate the inverse function at).
#' @param inverse if `TRUE`, evaluates `InverseFunction()` instead of `Function()`.
#' @param x_transform,y_transform one of `"none"` (default), `"logarithmic"` (also accepted as
#'   `"log"`), or `"normal_z"`.
#' @param confidence_level quantile in `[0, 1]` to sample the curve at; `NULL` (default) samples
#'   the mean.
#' @param allow_negative_y_values allow a negative or `NaN` result to pass through unmodified,
#'   rather than clamping it to `0`. Default `FALSE` (clamp), matching every use in the ported
#'   C# test suite -- the C# class default is `TRUE`.
#' @return a numeric vector, the same length as `at`.
#' @examples
#' x <- c(50, 100, 150, 200, 250)
#' d <- lapply(c(100, 200, 300, 400, 500), function(v) distribution("Deterministic", v))
#' tabular_function(x, d, at = 50, x_transform = "logarithmic")
#' @export
tabular_function <- function(x, distributions, at, inverse = FALSE,
                             x_transform = "none", y_transform = "none",
                             confidence_level = NULL, allow_negative_y_values = FALSE) {
  if (!is.numeric(x) || length(x) == 0L) {
    stop("`x` must be a non-empty numeric vector", call. = FALSE)
  }
  distributions <- paired_data_distributions(distributions, x)
  if (!is.numeric(at) || length(at) == 0L) {
    stop("`at` must be a non-empty numeric vector", call. = FALSE)
  }
  x_transform <- check_choice(x_transform, .paired_data_transforms, "x_transform")
  y_transform <- check_choice(y_transform, .paired_data_transforms, "y_transform")
  opts <- list(x = spec_array(as.double(x)), distributions = distributions,
               x_transform = x_transform, y_transform = y_transform,
               allow_negative_y_values = isTRUE(allow_negative_y_values))
  if (!is.null(confidence_level)) opts$confidence_level <- as.double(confidence_level)
  method <- if (isTRUE(inverse)) "tabular_inverse" else "tabular"
  toolbox_run("functions", method, list(at), opts)$values
}
