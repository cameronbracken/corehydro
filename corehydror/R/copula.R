# The bivariate copula surface. Every verb serializes a corehydro_copula to the dist_spec.hpp
# grammar and runs one method through ch_copula_run_; nothing holds C++ state.

kCopulaFamilies <- c("AliMikhailHaq", "Clayton", "Frank", "Gumbel", "Joe", "Normal", "StudentT")

# Internal: run a method against a corehydro_copula and return the numeric result. `args` is
# always serialized as a JSON array, even when it has length one.
copula_run <- function(cop, method, args = numeric(0)) {
  res <- ch_copula_run_(cop$spec, method, to_spec_json(spec_array(as.double(args))))
  out <- res$values
  if (length(res$names)) names(out) <- res$names
  out
}

check_copula <- function(cop) {
  if (!inherits(cop, "corehydro_copula")) {
    stop("`cop` must be a corehydro_copula object; create one with copula() or copula_fit()",
         call. = FALSE)
  }
  cop
}

check_margin <- function(m, nm) {
  if (!is.null(m) && !inherits(m, "corehydro_dist")) {
    stop(sprintf("`%s` must be a corehydro_dist (see distribution())", nm), call. = FALSE)
  }
  m
}

#' Construct a bivariate copula
#'
#' Build a copula from a family name and its dependence parameter(s). Mirrors the C#
#' `BivariateCopula` hierarchy of the Numerics library (`ClaytonCopula`, `GumbelCopula`, ...).
#'
#' `margin_x` and `margin_y` are optional and are treated differently depending on their type,
#' which matters for [copula_random()] (which needs marginals to draw on the data scale) and for
#' the IFM/full log-likelihoods:
#' * a `corehydro_dist` (e.g. `distribution("Normal", c(0, 1))`) attaches that distribution
#'   exactly as given, with no re-fitting.
#' * this constructor does not accept a bare family-name string for the marginals; use
#'   [copula_fit()] to have a named marginal MLE-fitted to a sample.
#'
#' @param family one of [copula_names()].
#' @param theta the dependence parameter.
#' @param df degrees of freedom, required for `"StudentT"` and ignored otherwise.
#' @param margin_x,margin_y optional `corehydro_dist` marginals (see Details).
#' @return a `corehydro_copula`.
#' @seealso [copula_fit()] to estimate a copula (and optionally its marginals) from data.
#' @examples
#' copula_pdf(copula("Clayton", theta = 2), 0.3, 0.7)
#' @export
copula <- function(family, theta, df = NULL, margin_x = NULL, margin_y = NULL) {
  if (!is.character(family) || length(family) != 1L || !family %in% kCopulaFamilies) {
    stop(sprintf("`family` must be one of %s", paste(kCopulaFamilies, collapse = ", ")),
         call. = FALSE)
  }
  if (!is.numeric(theta) || length(theta) != 1L || !is.finite(theta)) {
    stop("`theta` must be a single finite number", call. = FALSE)
  }
  if (identical(family, "StudentT") && is.null(df)) {
    stop("`df` is required for the StudentT copula", call. = FALSE)
  }
  margin_x <- check_margin(margin_x, "margin_x")
  margin_y <- check_margin(margin_y, "margin_y")
  spec <- to_spec_json(list(family = family, theta = as.double(theta),
                            df = if (is.null(df)) NULL else as.double(df),
                            margin_x = margin_x, margin_y = margin_y))
  structure(list(family = family, theta = as.double(theta), df = df,
                 margin_x = margin_x, margin_y = margin_y, spec = spec),
            class = "corehydro_copula")
}

#' @export
print.corehydro_copula <- function(x, ...) {
  cat(sprintf("<corehydro_copula> %s(theta = %g%s)\n", x$family, x$theta,
              if (is.null(x$df)) "" else sprintf(", df = %g", x$df)))
  invisible(x)
}

#' Fit a bivariate copula to data
#'
#' Estimate a copula's dependence parameter(s) -- and, when a marginal names a family without
#' parameters, that marginal's own parameters -- from a paired sample. Mirrors the C#
#' `BivariateCopulaEstimation` methods of the Numerics library.
#'
#' `margin_x` and `margin_y` accept EITHER a family-name string or a `corehydro_dist`, and the
#' two are handled differently:
#' * a **name** (e.g. `margin_x = "Normal"`) is fitted by maximum likelihood to `x` (or `y`)
#'   before the copula is estimated. This is what Inference From Margins (`method = "ifm"`)
#'   requires, and is the usual case for `"ifm"` and `"mle"`.
#' * a **`corehydro_dist`** (e.g. `margin_x = distribution("Normal", c(0, 1))`) is attached
#'   exactly as given and is NOT refitted. Use this when you want a literal, fixed marginal.
#'
#' `method = "tau"` inverts Kendall's tau into theta directly and is only implemented upstream
#' for Clayton, Gumbel, and AliMikhailHaq (`SetThetaFromTau`); it errors for every other family.
#'
#' @param family one of [copula_names()].
#' @param x,y numeric vectors of paired observations, the same length.
#' @param method `"mpl"` (maximum pseudo-likelihood, the default), `"ifm"` (inference from
#'   margins), `"mle"` (full maximum likelihood), or `"tau"` (Kendall's tau inversion; Clayton,
#'   Gumbel, and AliMikhailHaq only).
#' @param margin_x,margin_y optional marginals; see Details.
#' @return a fitted `corehydro_copula`.
#' @examples
#' x <- c(135.9, 104.1, 108.7, 99.3, 134.7, 91.0, 77.3, 115.4, 109.0, 79.0)
#' y <- c(1.9, 1.3, 1.4, 1.2, 1.8, 1.1, 0.9, 1.5, 1.4, 1.0)
#' copula_fit("Clayton", x, y, method = "mpl")
#' @export
copula_fit <- function(family, x, y, method = c("mpl", "ifm", "mle", "tau"),
                        margin_x = NULL, margin_y = NULL) {
  if (!is.character(family) || length(family) != 1L || !family %in% kCopulaFamilies) {
    stop(sprintf("`family` must be one of %s", paste(kCopulaFamilies, collapse = ", ")),
         call. = FALSE)
  }
  method <- match.arg(method)
  if (length(x) != length(y)) stop("`x` and `y` must have the same length", call. = FALSE)
  margin_spec <- function(m) {
    if (is.null(m)) return(NULL)
    if (is.character(m) && length(m) == 1L) return(list(family = m))
    if (inherits(m, "corehydro_dist")) return(m)
    stop("marginals must be a family-name string or a corehydro_dist", call. = FALSE)
  }
  fit <- list(x = spec_array(as.double(x)), y = spec_array(as.double(y)), method = method,
              margin_x = margin_spec(margin_x), margin_y = margin_spec(margin_y))
  spec <- to_spec_json(list(family = family, fit = fit))
  theta <- unname(rjson_run(spec, "theta"))
  df <- if (identical(family, "StudentT")) unname(rjson_run(spec, "df")) else NULL
  # Read the (possibly MLE-fitted) marginal back as a corehydro_dist, so a fitted copula is
  # fully parameterized and usable by copula_random() exactly like one built by copula().
  fitted_margin <- function(m, which) {
    if (is.null(m)) return(NULL)
    fam <- if (is.character(m)) m else m$family
    distribution(fam, unname(rjson_run(spec, paste0("marginal_", which, "_parameters"))))
  }
  structure(list(family = family, theta = theta, df = df,
                 margin_x = fitted_margin(margin_x, "x"), margin_y = fitted_margin(margin_y, "y"),
                 spec = spec),
            class = "corehydro_copula")
}

# Internal: run a method against an already-serialized spec string (used by copula_fit(), which
# has no corehydro_copula yet to hand to copula_run()).
rjson_run <- function(spec, method, args = numeric(0)) {
  res <- ch_copula_run_(spec, method, to_spec_json(spec_array(as.double(args))))
  out <- res$values
  if (length(res$names)) names(out) <- res$names
  out
}

#' Copula density, distribution, and inverse functions
#'
#' Density, log-density, and CDF for a [copula()] object, evaluated on the unit square.
#'
#' @param cop a `corehydro_copula` from [copula()] or [copula_fit()].
#' @param u,v numeric vectors in `(0, 1)`, the copula's two arguments.
#' @return a numeric vector, recycled over `u`/`v`.
#' @name copula_functions
#' @examples
#' cop <- copula("Clayton", theta = 2)
#' copula_pdf(cop, 0.3, 0.7)
#' copula_cdf(cop, 0.3, 0.7)
NULL

#' @rdname copula_functions
#' @export
copula_pdf <- function(cop, u, v) {
  check_copula(cop)
  unname(copula_run(cop, "pdf", c(u, v)))
}

#' @rdname copula_functions
#' @export
copula_log_pdf <- function(cop, u, v) {
  check_copula(cop)
  unname(copula_run(cop, "log_pdf", c(u, v)))
}

#' @rdname copula_functions
#' @export
copula_cdf <- function(cop, u, v) {
  check_copula(cop)
  unname(copula_run(cop, "cdf", c(u, v)))
}

#' Copula inverse CDF
#'
#' @param cop a `corehydro_copula`.
#' @param u,v numeric scalars in `(0, 1)`.
#' @return a numeric vector of length two.
#' @examples
#' copula_inverse_cdf(copula("Clayton", theta = 2), 0.3, 0.7)
#' @export
copula_inverse_cdf <- function(cop, u, v) {
  check_copula(cop)
  unname(copula_run(cop, "inverse_cdf", c(u, v)))
}

#' Copula tail dependence
#'
#' The lower and upper tail dependence coefficients of a [copula()].
#'
#' @param cop a `corehydro_copula`.
#' @return a named numeric of length two, `lower` and `upper`.
#' @examples
#' copula_tail_dependence(copula("Clayton", theta = 2))
#' @export
copula_tail_dependence <- function(cop) {
  check_copula(cop)
  copula_run(cop, "tail_dependence")
}

#' Copula joint exceedance probability
#'
#' `P(U > u, V > v)` (`type = "and"`) or `P(U > u \\| V > v)`'s complement,
#' `P(U > u OR V > v)` (`type = "or"`).
#'
#' @param cop a `corehydro_copula`.
#' @param u,v numeric scalars in `(0, 1)`.
#' @param type `"and"` for the joint (both-exceed) probability, `"or"` for the union
#'   (either-exceeds) probability.
#' @return a single numeric in `[0, 1]`.
#' @examples
#' copula_exceedance(copula("Gumbel", theta = 2), 0.9, 0.9, type = "and")
#' @export
copula_exceedance <- function(cop, u, v, type = c("and", "or")) {
  check_copula(cop)
  type <- match.arg(type)
  method <- if (type == "and") "exceedance_and" else "exceedance_or"
  unname(copula_run(cop, method, c(u, v)))
}

#' Copula theta bounds
#'
#' The valid range of the dependence parameter for a copula's family.
#'
#' @param cop a `corehydro_copula`.
#' @return a named numeric of length two, `minimum` and `maximum`.
#' @examples
#' copula_bounds(copula("Clayton", theta = 2))
#' @export
copula_bounds <- function(cop) {
  check_copula(cop)
  copula_run(cop, "bounds")
}

#' Copula parameters
#'
#' The copula's dependence parameter vector (`theta`, and `df` for `"StudentT"`).
#'
#' @param cop a `corehydro_copula`.
#' @return a numeric vector.
#' @examples
#' copula_params(copula("Clayton", theta = 2))
#' @export
copula_params <- function(cop) {
  check_copula(cop)
  unname(copula_run(cop, "parameters"))
}

#' Draw from a copula
#'
#' Simulate from the copula's own seeded Mersenne Twister stream, mapped through the attached
#' marginals to the data scale when both `margin_x` and `margin_y` were supplied; on the unit
#' square otherwise. A given `seed` reproduces the same draws bit-for-bit in R, Python, and the
#' upstream C# library.
#'
#' @param cop a `corehydro_copula`.
#' @param n number of draws.
#' @param seed integer seed for reproducible draws; `NULL` (the default) seeds from the clock.
#' @return an `n x 2` numeric matrix.
#' @examples
#' cop <- copula("Clayton", theta = 2,
#'               margin_x = distribution("Normal", c(0, 1)),
#'               margin_y = distribution("Normal", c(0, 1)))
#' copula_random(cop, 5, seed = 12345)
#' @export
copula_random <- function(cop, n, seed = NULL) {
  check_copula(cop)
  seed <- if (is.null(seed)) -1L else as.integer(seed)
  v <- copula_run(cop, "random", c(as.integer(n), seed))
  matrix(unname(v), ncol = 2L)
}

#' Copula log-likelihood
#'
#' Three log-likelihoods over a paired sample, differing in how the marginals enter: the pseudo
#' log-likelihood works on the data's pseudo-observations (no marginals needed), IFM (inference
#' from margins) transforms the raw data through the attached marginal CDFs then evaluates the
#' copula density, and the full log-likelihood adds the marginal log-densities to that. `method =
#' "ifm"` and `"full"` need `margin_x`/`margin_y` attached (see [copula()]).
#'
#' Upstream's pseudo log-likelihood is defined on values already on `(0, 1)` (the "plotting
#' positions" of the sample), not on the data's own scale -- a raw `x`/`y` fed to it directly
#' would fall outside the copula's domain. `method = "pseudo"` therefore converts `x` and `y` to
#' pseudo-observations via `rank(x) / (n + 1)` before evaluating, matching the Weibull plotting
#' position the C++ `BivariateCopulaEstimation` machinery itself uses to seed its MPL step (see
#' `bivariate_copula_estimation.hpp`'s internal `rank / (n + 1)` transform). `method = "ifm"` and
#' `"full"` take `x`/`y` on their own data scale and transform through the marginals internally.
#'
#' @param cop a `corehydro_copula`.
#' @param x,y numeric vectors of paired observations, the same length.
#' @param method `"pseudo"` (the default), `"ifm"`, or `"full"`.
#' @return a single numeric.
#' @examples
#' cop <- copula("Clayton", theta = 2,
#'               margin_x = distribution("Normal", c(110, 20)),
#'               margin_y = distribution("Normal", c(1.4, 0.3)))
#' x <- c(135.9, 104.1, 108.7, 99.3, 134.7, 91.0, 77.3, 115.4, 109.0, 79.0)
#' y <- c(1.9, 1.3, 1.4, 1.2, 1.8, 1.1, 0.9, 1.5, 1.4, 1.0)
#' copula_log_likelihood(cop, x, y, method = "ifm")
#' @export
copula_log_likelihood <- function(cop, x, y, method = c("pseudo", "ifm", "full")) {
  check_copula(cop)
  method <- match.arg(method)
  if (length(x) != length(y)) stop("`x` and `y` must have the same length", call. = FALSE)
  if (identical(method, "pseudo")) {
    n <- length(x)
    x <- rank(x) / (n + 1)
    y <- rank(y) / (n + 1)
  }
  m <- switch(method, pseudo = "log_likelihood_pseudo", ifm = "log_likelihood_ifm",
              full = "log_likelihood_full")
  unname(copula_run(cop, m, c(as.double(x), as.double(y))))
}

#' List the supported copula families
#'
#' @return a character vector of the seven bivariate copula family names.
#' @examples
#' copula_names()
#' @export
copula_names <- function() {
  kCopulaFamilies
}
