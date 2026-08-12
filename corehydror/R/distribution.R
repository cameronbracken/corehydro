# Public distribution interface over the factory-dispatched univariate glue in
# src/dist.cpp. A `corehydro_dist` is a plain classed list (family + params); every verb
# re-dispatches through the stateless C++ entry points, so no C++ object lifetime leaks
# into R. The five composite families (TruncatedDistribution, Mixture, CompetingRisks,
# Empirical, KernelDensity) are constructed by dist_truncated()/dist_mixture()/
# dist_competing_risks()/dist_empirical()/dist_kde(): a composite `corehydro_dist` carries a
# serialized `spec` instead of a flat `params` vector, and every verb routes it through
# dist_run() -> ch_dist_spec_run_() -> the shared C++ dist_runner.hpp, over the identical
# grammar dist_spec.hpp builds from (see R/spec.R's to_spec_json()).

# Internal: run a method against a corehydro_dist and return the numeric result. `args` is
# always serialized as a JSON array, even when it has length one.
dist_run <- function(d, method, args = numeric(0)) {
  res <- ch_dist_spec_run_(to_spec_json(d), method, to_spec_json(spec_array(as.double(args))))
  out <- res$values
  if (length(res$names)) names(out) <- res$names
  out
}

# Internal: a composite carries its serialized spec instead of a flat parameter vector, so
# `params` is NULL and every verb routes through dist_run().
new_composite_dist <- function(family, spec_json) {
  structure(list(family = family, params = NULL, spec = spec_json),
            class = "corehydro_dist")
}

#' Create a univariate distribution object
#'
#' Construct a distribution from its family name and parameter vector. The
#' result is a lightweight object accepted by [dist_pdf()], [dist_cdf()],
#' [dist_quantile()], [dist_random()], [dist_moments()], and friends. All 38
#' factory-constructible families of the USACE-RMC Numerics library are
#' supported; see [distribution_names()] for the list.
#'
#' Parameters are positional, in the same order as the C# constructor for the
#' family (for example `Normal` takes `c(mean, sd)` and
#' `GeneralizedExtremeValue` takes `c(location, scale, shape)`). Use
#' [dist_params()] on a constructed object to see the parameter names.
#'
#' @param family the distribution family name, e.g. `"Normal"`,
#'   `"LogNormal"`, `"Gumbel"`, `"GeneralizedExtremeValue"`.
#' @param params numeric vector of parameters, in constructor order.
#' @return An object of class `corehydro_dist`.
#' @seealso [dist_fit()] to estimate one from data, [distribution_names()].
#' @export
#' @examples
#' d <- distribution("Normal", c(100, 15))
#' d
#' dist_cdf(d, 100)
distribution <- function(family, params) {
  if (!is.character(family) || length(family) != 1L) {
    stop("`family` must be a single distribution name; see distribution_names()")
  }
  structured <- c("TruncatedDistribution", "Mixture", "CompetingRisks", "Empirical",
                  "KernelDensity")
  ctor <- c(TruncatedDistribution = "dist_truncated()", Mixture = "dist_mixture()",
            CompetingRisks = "dist_competing_risks()", Empirical = "dist_empirical()",
            KernelDensity = "dist_kde()")
  if (family %in% structured) {
    stop(sprintf("'%s' has no flat parameter vector; use %s instead", family, ctor[[family]]),
         call. = FALSE)
  }
  if (!family %in% ch_dist_names_()) {
    stop(sprintf(
      "unknown distribution family '%s'; see distribution_names() for the %d supported names",
      family, length(ch_dist_names_())
    ))
  }
  params <- as.double(params)
  expected <- ch_dist_parameter_names_(family)$short
  if (length(expected) > 0L && length(params) != length(expected)) {
    stop(sprintf(
      "'%s' expects %d parameters (%s), got %d",
      family, length(expected), paste(expected, collapse = ", "), length(params)
    ))
  }
  structure(list(family = family, params = params), class = "corehydro_dist")
}

#' @export
print.corehydro_dist <- function(x, ...) {
  if (!is.null(x$spec)) {
    payload <- x$spec
    if (nchar(payload) > 80L) payload <- paste0(substr(payload, 1L, 77L), "...")
    cat(sprintf("<corehydro_dist> %s (composite)\n  %s\n", x$family, payload))
    return(invisible(x))
  }
  p <- dist_params(x)
  cat(sprintf(
    "<corehydro_dist> %s(%s)\n", x$family,
    paste(sprintf("%s = %g", names(p), p), collapse = ", ")
  ))
  if (!ch_dist_valid_(x$family, x$params)) {
    cat("  (parameters are not valid for this family)\n")
  }
  invisible(x)
}

check_dist <- function(d) {
  if (!inherits(d, "corehydro_dist")) {
    stop("`d` must be a corehydro_dist object; create one with distribution() or dist_fit()")
  }
  d
}

#' Distribution functions
#'
#' Density, log-density, distribution, quantile, and random-generation
#' functions for a [distribution()] object. All are vectorized over their
#' first numeric argument and evaluate in the shared C++ core, so results are
#' identical to the Python package and to the upstream C# library.
#'
#' `dist_random()` draws from the same seeded Mersenne Twister stream as the
#' C# `GenerateRandomValues(sampleSize, seed)`: a given `seed` reproduces the
#' C# draws bit-for-bit (and matches `corehydropy` exactly).
#'
#' @param d a `corehydro_dist` object from [distribution()] or [dist_fit()].
#' @param x,q numeric vector of quantiles.
#' @param p numeric vector of probabilities in `(0, 1)`.
#' @param n number of draws.
#' @param seed integer seed for reproducible draws; `NULL` (the default) seeds
#'   from the clock.
#' @return A numeric vector the same length as `x`, `q`, or `p` (`n` for
#'   `dist_random()`).
#' @name dist
#' @examples
#' d <- distribution("Gumbel", c(100, 10))
#' dist_pdf(d, c(95, 100, 120))
#' dist_quantile(d, c(0.5, 0.9, 0.99))
#' dist_random(d, 5, seed = 123)
NULL

#' @rdname dist
#' @export
dist_pdf <- function(d, x) {
  check_dist(d)
  if (is.null(d$spec)) return(ch_dist_pdf_v_(d$family, d$params, as.double(x)))
  dist_run(d, "pdf", as.double(x))
}

#' @rdname dist
#' @export
dist_log_pdf <- function(d, x) {
  check_dist(d)
  if (is.null(d$spec)) return(ch_dist_log_pdf_v_(d$family, d$params, as.double(x)))
  dist_run(d, "log_pdf", as.double(x))
}

#' @rdname dist
#' @export
dist_cdf <- function(d, q) {
  check_dist(d)
  if (is.null(d$spec)) return(ch_dist_cdf_v_(d$family, d$params, as.double(q)))
  dist_run(d, "cdf", as.double(q))
}

#' @rdname dist
#' @export
dist_quantile <- function(d, p) {
  check_dist(d)
  if (is.null(d$spec)) return(ch_dist_quantile_v_(d$family, d$params, as.double(p)))
  dist_run(d, "quantile", as.double(p))
}

#' @rdname dist
#' @export
dist_random <- function(d, n, seed = NULL) {
  check_dist(d)
  seed <- if (is.null(seed)) -1L else as.integer(seed)
  if (is.null(d$spec)) return(ch_dist_random_(d$family, d$params, as.integer(n), seed))
  dist_run(d, "random", c(as.integer(n), seed))
}

#' Distribution properties
#'
#' Moments, parameters, linear moments (L-moments), and log-likelihood of a
#' [distribution()] object.
#'
#' @param d a `corehydro_dist` object from [distribution()] or [dist_fit()].
#' @param data numeric vector of observations.
#' @return `dist_moments()` returns a named numeric vector (`mean`, `median`,
#'   `mode`, `sd`, `skewness`, `kurtosis`, `minimum`, `maximum`; entries are
#'   `NaN` where undefined). `dist_params()` returns the parameter vector named
#'   with the family's short-form parameter names. `dist_lmoments()` returns
#'   the first four L-moments (errors for families without L-moment support).
#'   `dist_log_likelihood()` returns a single numeric value.
#' @name dist_properties
#' @examples
#' d <- distribution("Normal", c(100, 15))
#' dist_moments(d)
#' dist_params(d)
NULL

#' @rdname dist_properties
#' @export
dist_moments <- function(d) {
  check_dist(d)
  if (is.null(d$spec)) return(ch_dist_moments_(d$family, d$params))
  dist_run(d, "moments")
}

#' @rdname dist_properties
#' @export
dist_params <- function(d) {
  check_dist(d)
  if (!is.null(d$spec)) return(dist_run(d, "parameters"))
  out <- d$params
  nm <- ch_dist_parameter_names_(d$family)$short
  if (length(nm) == length(out)) names(out) <- nm
  out
}

#' @rdname dist_properties
#' @export
dist_lmoments <- function(d) {
  check_dist(d)
  if (!is.null(d$spec)) {
    stop(sprintf(
      "linear moments are not available for '%s'; no composite distribution implements %s",
      d$family, "ILinearMomentEstimation upstream"), call. = FALSE)
  }
  ch_dist_linear_moments_(d$family, d$params)
}

#' @rdname dist_properties
#' @export
dist_log_likelihood <- function(d, data) {
  check_dist(d)
  if (is.null(d$spec)) return(ch_dist_log_likelihood_(d$family, d$params, as.double(data)))
  dist_run(d, "log_likelihood", as.double(data))
}

#' Fit a distribution to data
#'
#' Estimate the parameters of a distribution family from a sample and return
#' the fitted [distribution()] object. Mirrors the C#
#' `Estimate(data, ParameterEstimationMethod)` API of the Numerics library.
#'
#' @param family the distribution family name; see [distribution_names()].
#' @param data numeric vector of observations.
#' @param method estimation method: `"mle"` (maximum likelihood, the default),
#'   `"lmom"` (L-moments), or `"mom"` (product moments). Not every family
#'   supports every method; unsupported combinations error.
#' @return A fitted `corehydro_dist` object.
#' @export
#' @examples
#' d <- distribution("Gumbel", c(100, 10))
#' x <- dist_random(d, 200, seed = 42)
#' dist_fit("Gumbel", x, method = "mle")
dist_fit <- function(family, data, method = c("mle", "lmom", "mom")) {
  method <- match.arg(method)
  params <- ch_dist_fit_(family, as.double(data), method)
  distribution(family, params)
}

#' List the supported distribution families
#'
#' @param kind `"flat"` (the default) for families [distribution()] constructs directly from
#'   a parameter vector; `"structured"` for the five composite families constructed by
#'   [dist_truncated()], [dist_mixture()], [dist_competing_risks()], [dist_empirical()], and
#'   [dist_kde()]; or `"all"` for the union of both.
#' @return A character vector of distribution family names.
#' @export
#' @examples
#' distribution_names()
#' distribution_names("structured")
distribution_names <- function(kind = c("flat", "structured", "all")) {
  kind <- match.arg(kind)
  flat <- ch_dist_names_()
  structured <- c("TruncatedDistribution", "Mixture", "CompetingRisks", "Empirical",
                  "KernelDensity")
  switch(kind, flat = setdiff(flat, structured), structured = structured,
         all = union(setdiff(flat, structured), structured))
}

#' Truncate a distribution
#'
#' Restricts a distribution to `[min, max]`, renormalizing its density over the
#' truncated range. Mirrors the C# `TruncatedDistribution` composite.
#'
#' @param d a `corehydro_dist`, the base (untruncated) distribution.
#' @param min,max the truncation bounds, with `min < max`.
#' @return a `corehydro_dist` of family `"TruncatedDistribution"`, accepted by every
#'   `dist_*()` verb.
#' @export
#' @examples
#' d <- dist_truncated(distribution("Normal", c(2, 1)), min = 1.1, max = 2.11)
#' dist_pdf(d, 1.5)
dist_truncated <- function(d, min, max) {
  if (!inherits(d, "corehydro_dist")) stop("`d` must be a corehydro_dist", call. = FALSE)
  if (!is.numeric(min) || length(min) != 1L || !is.numeric(max) || length(max) != 1L) {
    stop("`min` and `max` must each be a single number", call. = FALSE)
  }
  if (min >= max) stop("`min` must be less than `max`", call. = FALSE)
  new_composite_dist("TruncatedDistribution", to_spec_json(list(
    family = "TruncatedDistribution",
    base = d,
    bounds = spec_array(c(as.double(min), as.double(max)))
  )))
}

#' Mixture distribution
#'
#' A weighted mixture of component distributions, optionally zero-inflated. Mirrors the
#' C# `Mixture` composite.
#'
#' @param components a list of `corehydro_dist` objects, the mixture components.
#' @param weights numeric vector of mixture weights, the same length as `components`.
#' @param zero_inflated whether the mixture places extra probability mass at zero.
#' @param zero_weight the probability mass at zero when `zero_inflated` is `TRUE`; ignored
#'   otherwise.
#' @return a `corehydro_dist` of family `"Mixture"`, accepted by every `dist_*()` verb.
#' @export
#' @examples
#' d <- dist_mixture(
#'   list(distribution("Normal", c(0, 1)), distribution("Normal", c(5, 1))),
#'   weights = c(0.5, 0.5)
#' )
#' dist_pdf(d, 2.5)
dist_mixture <- function(components, weights, zero_inflated = FALSE, zero_weight = 0) {
  if (!is.list(components) || length(components) == 0L) {
    stop("`components` must be a non-empty list of corehydro_dist objects", call. = FALSE)
  }
  if (!all(vapply(components, inherits, logical(1), "corehydro_dist"))) {
    stop("every element of `components` must be a corehydro_dist", call. = FALSE)
  }
  if (length(weights) != length(components)) {
    stop("`weights` must have the same length as `components`", call. = FALSE)
  }
  new_composite_dist("Mixture", to_spec_json(list(
    family = "Mixture",
    components = components,
    weights = spec_array(as.double(weights)),
    zero_inflated = isTRUE(zero_inflated),
    zero_weight = as.double(zero_weight)
  )))
}

#' Competing-risks distribution
#'
#' The distribution of the minimum (a series system) or maximum (a parallel system) of
#' several component random variables, with an optional dependency structure. Mirrors the
#' C# `CompetingRisks` composite.
#'
#' @param components a list of `corehydro_dist` objects, the component distributions.
#' @param minimum_of `TRUE` (the default) for the distribution of the minimum of
#'   `components`; `FALSE` for the maximum.
#' @param dependency one of `"Independent"` (the default), `"PerfectlyPositive"`,
#'   `"PerfectlyNegative"`, or `"CorrelationMatrix"`.
#' @param correlation a square numeric correlation matrix; required when
#'   `dependency = "CorrelationMatrix"`, ignored otherwise.
#' @return a `corehydro_dist` of family `"CompetingRisks"`, accepted by every `dist_*()`
#'   verb.
#' @export
#' @examples
#' d <- dist_competing_risks(
#'   list(distribution("Weibull", c(1, 2)), distribution("Weibull", c(1, 3)))
#' )
#' dist_cdf(d, 1.5)
dist_competing_risks <- function(components, minimum_of = TRUE, dependency = "Independent",
                                  correlation = NULL) {
  if (!is.list(components) || length(components) == 0L) {
    stop("`components` must be a non-empty list of corehydro_dist objects", call. = FALSE)
  }
  if (!all(vapply(components, inherits, logical(1), "corehydro_dist"))) {
    stop("every element of `components` must be a corehydro_dist", call. = FALSE)
  }
  dep_choices <- c("Independent", "PerfectlyPositive", "PerfectlyNegative", "CorrelationMatrix")
  if (!is.character(dependency) || length(dependency) != 1L || !dependency %in% dep_choices) {
    stop(sprintf("`dependency` must be one of %s", paste(dep_choices, collapse = ", ")),
         call. = FALSE)
  }
  corr_spec <- NULL
  if (!is.null(correlation)) {
    correlation <- as.matrix(correlation)
    if (nrow(correlation) != ncol(correlation)) {
      stop("`correlation` must be a square matrix", call. = FALSE)
    }
    corr_spec <- lapply(seq_len(nrow(correlation)),
                        function(i) spec_array(as.double(correlation[i, ])))
  }
  new_composite_dist("CompetingRisks", to_spec_json(list(
    family = "CompetingRisks",
    components = components,
    minimum_of_random_variables = isTRUE(minimum_of),
    dependency = dependency,
    correlation = corr_spec
  )))
}

#' Empirical distribution
#'
#' A distribution defined by paired value/probability points, interpolated between.
#' Mirrors the C# `EmpiricalDistribution` composite.
#'
#' @param x numeric vector of values.
#' @param p numeric vector of associated probabilities, the same length as `x`.
#' @param p_transform how `p` is interpolated between: `"NormalZ"` (the default) transforms
#'   through the standard-normal quantile; `"None"` interpolates `p` directly.
#' @param p_descending whether `p` decreases as `x` increases (a survival-function
#'   encoding); `FALSE` (the default) is the ordinary ascending-CDF case.
#' @return a `corehydro_dist` of family `"Empirical"`, accepted by every `dist_*()` verb.
#' @export
#' @examples
#' d <- dist_empirical(x = c(1, 2, 3), p = c(0.1, 0.5, 0.9))
#' dist_quantile(d, 0.5)
dist_empirical <- function(x, p, p_transform = "NormalZ", p_descending = FALSE) {
  pt_choices <- c("NormalZ", "None")
  if (!is.character(p_transform) || length(p_transform) != 1L ||
      !p_transform %in% pt_choices) {
    stop(sprintf("`p_transform` must be one of %s", paste(pt_choices, collapse = ", ")),
         call. = FALSE)
  }
  if (length(x) != length(p)) stop("`x` and `p` must have the same length", call. = FALSE)
  new_composite_dist("Empirical", to_spec_json(list(
    family = "Empirical",
    x = spec_array(as.double(x)),
    p = spec_array(as.double(p)),
    p_transform = p_transform,
    p_descending = isTRUE(p_descending)
  )))
}

#' Kernel density distribution
#'
#' A nonparametric density estimate built from a sample by summing a kernel centered at
#' each observation. Mirrors the C# `KernelDensity` composite.
#'
#' @param data numeric vector of observations the density is built from.
#' @param kernel the kernel shape: `"Gaussian"` (the default), `"Epanechnikov"`,
#'   `"Triangular"`, or `"Uniform"`.
#' @param bandwidth the kernel bandwidth; `NULL` (the default) uses Silverman's rule of
#'   thumb.
#' @param bounded_by_data whether the reported minimum and maximum are the smallest and
#'   largest observation (`TRUE`, the default) or extend three bandwidths past each. Those
#'   bounds gate [dist_cdf()] and [dist_quantile()]. The density is summed wherever you ask
#'   it, either way.
#' @return a `corehydro_dist` of family `"KernelDensity"`, accepted by every `dist_*()`
#'   verb.
#' @export
#' @examples
#' d <- dist_kde(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
#' dist_cdf(d, 5.5)
dist_kde <- function(data, kernel = "Gaussian", bandwidth = NULL, bounded_by_data = TRUE) {
  kernel_choices <- c("Gaussian", "Epanechnikov", "Triangular", "Uniform")
  if (!is.character(kernel) || length(kernel) != 1L || !kernel %in% kernel_choices) {
    stop(sprintf("`kernel` must be one of %s", paste(kernel_choices, collapse = ", ")),
         call. = FALSE)
  }
  new_composite_dist("KernelDensity", to_spec_json(list(
    family = "KernelDensity",
    data = spec_array(as.double(data)),
    kernel = kernel,
    bandwidth = if (is.null(bandwidth)) NULL else as.double(bandwidth),
    bounded_by_data = isTRUE(bounded_by_data)
  )))
}
