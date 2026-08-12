# The bivariate copula surface. Every verb serializes a corehydro_copula to the dist_spec.hpp
# grammar and runs one method through ch_copula_run_; nothing holds C++ state.

kCopulaFamilies <- c("AliMikhailHaq", "Clayton", "Frank", "Gumbel", "Joe", "Normal", "StudentT")

# Internal: run a method and return the numeric result. `cop` is either a corehydro_copula or an
# already-serialized spec string (copula_fit() has no corehydro_copula yet when it reads its own
# fit back). `args` is always serialized as a JSON array, even when it has length one.
copula_run <- function(cop, method, args = numeric(0)) {
  spec <- if (is.character(cop)) cop else cop$spec
  res <- ch_copula_run_(spec, method, to_spec_json(spec_array(as.double(args))))
  out <- res$values
  if (length(res$names)) names(out) <- res$names
  out
}

# Internal: recycle a pair of vectors to a common length and lay them out as all `u` then all
# `v`, the split-at-the-halfway-point layout run_copula's pdf / log_pdf / cdf arms read.
copula_pairs <- function(u, v, method) {
  u <- as.double(u)
  v <- as.double(v)
  if (length(u) == 0L || length(v) == 0L) {
    stop(sprintf("`u` and `v` must both be non-empty in %s()", method), call. = FALSE)
  }
  n <- max(length(u), length(v))
  if (n %% length(u) != 0L || n %% length(v) != 0L) {
    stop(sprintf("`u` (length %d) and `v` (length %d) are not recyclable to a common length",
                 length(u), length(v)), call. = FALSE)
  }
  c(rep_len(u, n), rep_len(v, n))
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
#' `x` and `y` are the raw paired observations for every method. `method = "mpl"` maximizes the
#' pseudo-likelihood, which is defined on the plotting positions `rank / (n + 1)` rather than on
#' the data scale; that transform happens inside the shared C++ core, so R and Python fit the
#' same numbers from the same input.
#'
#' `margin_x` and `margin_y` accept EITHER a family-name string or a `corehydro_dist`, and the
#' two are handled differently:
#' * a **name** (e.g. `margin_x = "Normal"`) is fitted by maximum likelihood to `x` (or `y`)
#'   before the copula is estimated. This is what Inference From Margins (`method = "ifm"`)
#'   requires, and it is accepted for `"ifm"` and `"mle"` only: `"mpl"` and `"tau"` ignore the
#'   marginals entirely, so a name there would be reported back unfitted.
#' * a **`corehydro_dist`** (e.g. `margin_x = distribution("Normal", c(0, 1))`) is accepted for
#'   all four methods. `"mpl"` and `"tau"` attach it untouched (so [copula_random()] can draw on
#'   the data scale), `"ifm"` takes it as the given margin, and `"mle"` re-estimates it jointly
#'   with theta.
#'
#' `method = "tau"` inverts Kendall's tau into theta directly and is only implemented upstream
#' for Clayton, Gumbel, and AliMikhailHaq (`SetThetaFromTau`); it errors for every other family.
#'
#' @param family one of [copula_names()].
#' @param x,y numeric vectors of raw paired observations, the same length.
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
  margin_spec <- function(m, nm) {
    if (is.null(m)) return(NULL)
    if (is.character(m) && length(m) == 1L) {
      # "mpl" and "tau" never look at the marginals, so a named family would come back a
      # default Normal(0, 1) presented as though it had been fitted.
      if (method %in% c("mpl", "tau")) {
        stop(sprintf(paste0("method = \"%s\" does not use marginals, so `%s = \"%s\"` would be ",
                            "left unfitted; pass a parameterized corehydro_dist (see ",
                            "distribution()) to attach a fixed marginal for later sampling, or ",
                            "use method = \"ifm\" or \"mle\" to fit one"),
                     method, nm, m), call. = FALSE)
      }
      return(list(family = m))
    }
    if (inherits(m, "corehydro_dist")) return(m)
    stop(sprintf("`%s` must be a family-name string or a corehydro_dist", nm), call. = FALSE)
  }
  fit <- list(x = spec_array(as.double(x)), y = spec_array(as.double(y)), method = method,
              margin_x = margin_spec(margin_x, "margin_x"),
              margin_y = margin_spec(margin_y, "margin_y"))
  spec <- to_spec_json(list(family = family, fit = fit))
  # One "parameters" call carries theta and, for StudentT, df -- the estimation runs once per
  # runner call, so asking for them separately would refit the copula twice.
  pars <- unname(copula_run(spec, "parameters"))
  theta <- pars[1L]
  df <- if (identical(family, "StudentT")) pars[2L] else NULL
  # Read a marginal back as a corehydro_dist only when the fit actually moved it: "mle"
  # re-estimates both marginals jointly, and a named marginal is MLE-fitted before an "ifm"
  # fit. A corehydro_dist under "mpl"/"tau"/"ifm" is used exactly as given, so it is already
  # the answer and needs no second estimation run.
  fitted_margin <- function(m, which) {
    if (is.null(m)) return(NULL)
    if (!identical(method, "mle") && inherits(m, "corehydro_dist")) return(m)
    fam <- if (is.character(m)) m else m$family
    distribution(fam, unname(copula_run(spec, paste0("marginal_", which, "_parameters"))))
  }
  structure(list(family = family, theta = theta, df = df,
                 margin_x = fitted_margin(margin_x, "x"), margin_y = fitted_margin(margin_y, "y"),
                 spec = spec),
            class = "corehydro_copula")
}

#' Copula density, distribution, and inverse functions
#'
#' Density, log-density, and CDF for a [copula()] object, evaluated on the unit square. `u` and
#' `v` are recycled to a common length and evaluated pairwise, one returned value per pair.
#'
#' @param cop a `corehydro_copula` from [copula()] or [copula_fit()].
#' @param u,v numeric vectors in `(0, 1)`, the copula's two arguments.
#' @return a numeric vector, one value per recycled `(u, v)` pair.
#' @name copula_functions
#' @examples
#' cop <- copula("Clayton", theta = 2)
#' copula_pdf(cop, 0.3, 0.7)
#' copula_pdf(cop, c(0.3, 0.5), c(0.7, 0.9))
#' copula_cdf(cop, 0.3, 0.7)
NULL

#' @rdname copula_functions
#' @export
copula_pdf <- function(cop, u, v) {
  check_copula(cop)
  unname(copula_run(cop, "pdf", copula_pairs(u, v, "copula_pdf")))
}

#' @rdname copula_functions
#' @export
copula_log_pdf <- function(cop, u, v) {
  check_copula(cop)
  unname(copula_run(cop, "log_pdf", copula_pairs(u, v, "copula_log_pdf")))
}

#' @rdname copula_functions
#' @export
copula_cdf <- function(cop, u, v) {
  check_copula(cop)
  unname(copula_run(cop, "cdf", copula_pairs(u, v, "copula_cdf")))
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
#' The probability that both variables exceed their thresholds, `P(U > u, V > v)`
#' (`type = "and"`), or that at least one does, `P(U > u or V > v) = 1 - C(u, v)`
#' (`type = "or"`).
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
#' All three methods take `x` and `y` as raw paired observations on their own data scale.
#' Upstream's pseudo log-likelihood is defined on values already on `(0, 1)`, so `"pseudo"`
#' converts the sample to its plotting positions, `rank / (n + 1)`, first; that transform happens
#' inside the shared C++ core (the same one [copula_fit()]'s `"mpl"` fit uses), so R and Python
#' return the same number for the same input. `"ifm"` and `"full"` transform through the attached
#' marginal CDFs instead.
#'
#' @param cop a `corehydro_copula`.
#' @param x,y numeric vectors of raw paired observations, the same length.
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
