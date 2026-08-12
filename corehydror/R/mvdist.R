# The multivariate distribution surface. Every verb serializes a corehydro_mvdist to the
# dist_spec.hpp grammar and runs one method through ch_mvdist_run_; nothing holds C++ state.

kMvdistFamilies <- c("MultivariateNormal", "MultivariateStudentT", "Dirichlet", "Multinomial",
                     "BivariateEmpirical")

# Internal: run a method against a corehydro_mvdist and return the raw DistResult list
# (values/names/spec) so callers that need the child spec (marginal/conditional) can read it.
mvdist_run <- function(mv, method, args = numeric(0)) {
  ch_mvdist_run_(mv$spec, method, to_spec_json(spec_array(as.double(args))))
}

check_mvdist <- function(mv) {
  if (!inherits(mv, "corehydro_mvdist")) {
    stop("`mv` must be a corehydro_mvdist object; create one with mvdist_normal() and friends",
         call. = FALSE)
  }
  mv
}

# Internal: user-facing indices are 1-based (matching trend()/model_parameter()); the spec and
# the C++ take 0-based. A fractional index is rejected rather than truncated, and duplicates are
# rejected here so the message names the argument (the C++ validate_indices rejects them too).
mv_indices <- function(idx, n, what) {
  if (!is.numeric(idx) || length(idx) == 0L || anyNA(idx)) {
    stop(sprintf("`%s` must be a non-empty numeric vector of dimension indices", what),
         call. = FALSE)
  }
  if (any(idx != trunc(idx))) {
    stop(sprintf("`%s` must be whole numbers; got %s", what,
                 paste(idx[idx != trunc(idx)], collapse = ", ")), call. = FALSE)
  }
  idx <- as.integer(idx)
  if (any(idx < 1L) || any(idx > n)) {
    stop(sprintf("`%s` must be between 1 and %d", what, n), call. = FALSE)
  }
  if (anyDuplicated(idx)) {
    stop(sprintf("`%s` must not repeat a dimension; got %s", what,
                 paste(idx[duplicated(idx)], collapse = ", ")), call. = FALSE)
  }
  idx - 1L
}

new_mvdist <- function(family, spec_json) {
  structure(list(family = family, spec = spec_json), class = "corehydro_mvdist")
}

#' @export
print.corehydro_mvdist <- function(x, ...) {
  cat(sprintf("<corehydro_mvdist> %s (dimension %d)\n", x$family, mvdist_dimension(x)))
  invisible(x)
}

#' Construct a multivariate normal distribution
#'
#' Mirrors the C# `MultivariateNormal` class of the Numerics library.
#'
#' @param mean numeric vector of means, length `d`.
#' @param covariance a `d x d` symmetric positive-definite covariance matrix.
#' @param seed optional integer seed for the Genz quasi-Monte-Carlo integrator behind
#'   [mvdist_cdf()] at dimension three and above; `NULL` (the default) leaves it clock-seeded.
#'   **Without a seed, [mvdist_cdf()] at dimension >= 3 is not reproducible run to run** (it draws
#'   from a per-instance Mersenne Twister), so R and Python cannot agree on a value unless `seed`
#'   is set explicitly.
#' @param max_evaluations,abs_error,rel_error optional integrator tuning; `NULL` (the default)
#'   for each leaves the ported upstream default untouched.
#' @return a `corehydro_mvdist` of family `"MultivariateNormal"`.
#' @examples
#' mv <- mvdist_normal(c(0, 0), diag(2))
#' mvdist_pdf(mv, c(0, 0))
#' @export
mvdist_normal <- function(mean, covariance, seed = NULL, max_evaluations = NULL,
                           abs_error = NULL, rel_error = NULL) {
  covariance <- as.matrix(covariance)
  if (nrow(covariance) != ncol(covariance)) {
    stop("`covariance` must be a square matrix", call. = FALSE)
  }
  if (length(mean) != nrow(covariance)) {
    stop("`mean` and `covariance` must have the same dimension", call. = FALSE)
  }
  rows <- lapply(seq_len(nrow(covariance)), function(i) spec_array(as.double(covariance[i, ])))
  spec <- to_spec_json(list(
    family = "MultivariateNormal",
    mean = spec_array(as.double(mean)),
    covariance = rows,
    seed = if (is.null(seed)) NULL else as.integer(seed),
    max_evaluations = if (is.null(max_evaluations)) NULL else as.integer(max_evaluations),
    abs_error = if (is.null(abs_error)) NULL else as.double(abs_error),
    rel_error = if (is.null(rel_error)) NULL else as.double(rel_error)
  ))
  new_mvdist("MultivariateNormal", spec)
}

#' Construct a multivariate Student-t distribution
#'
#' Mirrors the C# `MultivariateStudentT` class of the Numerics library.
#'
#' @param df degrees of freedom.
#' @param location numeric vector of location parameters, length `d`.
#' @param scale optional `d x d` scale matrix; `NULL` (the default) uses the identity.
#' @param seed optional integer seed for reproducible draws from [mvdist_random()]; `NULL` (the
#'   default) leaves it clock-seeded.
#' @return a `corehydro_mvdist` of family `"MultivariateStudentT"`.
#' @examples
#' mvdist_dimension(mvdist_student_t(5, c(0, 0)))
#' @export
mvdist_student_t <- function(df, location, scale = NULL, seed = NULL) {
  if (!is.numeric(df) || length(df) != 1L || !is.finite(df)) {
    stop("`df` must be a single finite number", call. = FALSE)
  }
  scale_rows <- NULL
  if (!is.null(scale)) {
    scale <- as.matrix(scale)
    if (nrow(scale) != ncol(scale) || nrow(scale) != length(location)) {
      stop("`scale` must be a square matrix matching the length of `location`", call. = FALSE)
    }
    scale_rows <- lapply(seq_len(nrow(scale)), function(i) spec_array(as.double(scale[i, ])))
  }
  spec <- to_spec_json(list(
    family = "MultivariateStudentT",
    df = as.double(df),
    location = spec_array(as.double(location)),
    scale = scale_rows,
    seed = if (is.null(seed)) NULL else as.integer(seed)
  ))
  new_mvdist("MultivariateStudentT", spec)
}

#' Construct a Dirichlet distribution
#'
#' Mirrors the C# `Dirichlet` class of the Numerics library.
#'
#' @param alpha numeric vector of positive concentration parameters.
#' @return a `corehydro_mvdist` of family `"Dirichlet"`.
#' @examples
#' mvdist_pdf(mvdist_dirichlet(c(2, 3, 4)), c(0.2, 0.3, 0.5))
#' @export
mvdist_dirichlet <- function(alpha) {
  spec <- to_spec_json(list(family = "Dirichlet", alpha = spec_array(as.double(alpha))))
  new_mvdist("Dirichlet", spec)
}

#' Construct a multinomial distribution
#'
#' Mirrors the C# `Multinomial` class of the Numerics library.
#'
#' @param trials the number of trials.
#' @param probabilities numeric vector of category probabilities, summing to one.
#' @return a `corehydro_mvdist` of family `"Multinomial"`.
#' @examples
#' mvdist_pdf(mvdist_multinomial(10, c(0.2, 0.3, 0.5)), c(2, 3, 5))
#' @export
mvdist_multinomial <- function(trials, probabilities) {
  spec <- to_spec_json(list(family = "Multinomial", trials = as.integer(trials),
                            probabilities = spec_array(as.double(probabilities))))
  new_mvdist("Multinomial", spec)
}

#' Construct a bivariate empirical distribution
#'
#' Mirrors the C# `BivariateEmpirical` class of the Numerics library: a joint distribution
#' defined over a grid of two marginal value vectors and a matrix of associated probabilities.
#'
#' @param x1,x2 numeric vectors of grid values for each dimension.
#' @param p a `length(x1) x length(x2)` matrix of joint probabilities.
#' @param x1_transform,x2_transform,p_transform how each axis is interpolated between: one of
#'   `"None"` (the default), `"Logarithmic"`, or `"NormalZ"`.
#' @return a `corehydro_mvdist` of family `"BivariateEmpirical"`. Note: `pdf` is an upstream stub
#'   (see [mvdist_pdf()]).
#' @examples
#' mv <- mvdist_bivariate_empirical(c(1, 2), c(1, 2), matrix(c(0.2, 0.3, 0.2, 0.3), nrow = 2))
#' mvdist_cdf(mv, c(1.5, 1.5))
#' @export
mvdist_bivariate_empirical <- function(x1, x2, p, x1_transform = "None", x2_transform = "None",
                                       p_transform = "None") {
  p <- as.matrix(p)
  if (nrow(p) != length(x1) || ncol(p) != length(x2)) {
    stop("`p` must be a length(x1) x length(x2) matrix", call. = FALSE)
  }
  rows <- lapply(seq_len(nrow(p)), function(i) spec_array(as.double(p[i, ])))
  spec <- to_spec_json(list(
    family = "BivariateEmpirical",
    x1 = spec_array(as.double(x1)), x2 = spec_array(as.double(x2)), p = rows,
    x1_transform = x1_transform, x2_transform = x2_transform, p_transform = p_transform
  ))
  new_mvdist("BivariateEmpirical", spec)
}

#' Multivariate density, distribution, and dimension
#'
#' @param mv a `corehydro_mvdist`.
#' @param x numeric vector, the evaluation point (length equal to [mvdist_dimension()]).
#' @return a single numeric (`mvdist_pdf()`, `mvdist_log_pdf()`, `mvdist_cdf()`), or a single
#'   integer (`mvdist_dimension()`).
#' @name mvdist_functions
#' @examples
#' mv <- mvdist_normal(c(0, 0), diag(2))
#' mvdist_pdf(mv, c(0, 0))
#' mvdist_dimension(mv)
NULL

#' @rdname mvdist_functions
#' @export
mvdist_pdf <- function(mv, x) {
  check_mvdist(mv)
  unname(mvdist_run(mv, "pdf", x)$values)
}

#' @rdname mvdist_functions
#' @export
mvdist_log_pdf <- function(mv, x) {
  check_mvdist(mv)
  unname(mvdist_run(mv, "log_pdf", x)$values)
}

#' @rdname mvdist_functions
#' @export
mvdist_cdf <- function(mv, x) {
  check_mvdist(mv)
  unname(mvdist_run(mv, "cdf", x)$values)
}

#' @rdname mvdist_functions
#' @export
mvdist_dimension <- function(mv) {
  check_mvdist(mv)
  as.integer(mvdist_run(mv, "dimension")$values)
}

#' Multivariate moments
#'
#' Mean, variance, standard deviation, median, mode, and covariance of a [mvdist_normal()] and
#' friends. Not every family defines every moment upstream; unsupported combinations error
#' naming the family (see the Details in [mvdist_marginal()]).
#'
#' @param mv a `corehydro_mvdist`.
#' @return a numeric vector of length [mvdist_dimension()] (`mvdist_mean()`, `mvdist_variance()`,
#'   `mvdist_sd()`, `mvdist_median()`, `mvdist_mode()`), or a `d x d` matrix (`mvdist_covariance()`).
#' @name mvdist_moments
#' @examples
#' mv <- mvdist_normal(c(1, 2), diag(2))
#' mvdist_mean(mv)
#' mvdist_covariance(mv)
NULL

#' @rdname mvdist_moments
#' @export
mvdist_mean <- function(mv) {
  check_mvdist(mv)
  unname(mvdist_run(mv, "mean")$values)
}

#' @rdname mvdist_moments
#' @export
mvdist_variance <- function(mv) {
  check_mvdist(mv)
  unname(mvdist_run(mv, "variance")$values)
}

#' @rdname mvdist_moments
#' @export
mvdist_sd <- function(mv) {
  check_mvdist(mv)
  unname(mvdist_run(mv, "sd")$values)
}

#' @rdname mvdist_moments
#' @export
mvdist_median <- function(mv) {
  check_mvdist(mv)
  unname(mvdist_run(mv, "median")$values)
}

#' @rdname mvdist_moments
#' @export
mvdist_mode <- function(mv) {
  check_mvdist(mv)
  unname(mvdist_run(mv, "mode")$values)
}

#' @rdname mvdist_moments
#' @export
mvdist_covariance <- function(mv) {
  check_mvdist(mv)
  n <- mvdist_dimension(mv)
  v <- mvdist_run(mv, "covariance")$values
  matrix(v, nrow = n, byrow = TRUE)
}

#' Mahalanobis distance
#'
#' @param mv a `corehydro_mvdist` of family `"MultivariateNormal"` or `"MultivariateStudentT"`.
#' @param x numeric vector, the evaluation point.
#' @return a single numeric.
#' @examples
#' mvdist_mahalanobis(mvdist_normal(c(0, 0), diag(2)), c(1, 1))
#' @export
mvdist_mahalanobis <- function(mv, x) {
  check_mvdist(mv)
  unname(mvdist_run(mv, "mahalanobis", x)$values)
}

#' Multivariate inverse CDF (Cholesky map)
#'
#' Maps a vector of independent uniform draws into one point on the distribution's scale via the
#' Cholesky decomposition of its covariance. **This is not a true multivariate quantile** (there
#' is no unique multivariate analogue of the univariate inverse CDF); it is exactly the map
#' upstream implements: `mean + L %*% qnorm(p)` where `L` is the Cholesky factor.
#'
#' `MultivariateNormal` takes `dimension` probabilities; `MultivariateStudentT` takes
#' `dimension + 1` (the last value drives the chi-squared mixing variable) and still returns
#' `dimension` values.
#'
#' @param mv a `corehydro_mvdist` of family `"MultivariateNormal"` or `"MultivariateStudentT"`.
#' @param p numeric vector of probabilities in `(0, 1)` (see Details for the required length).
#' @return a numeric vector of length [mvdist_dimension()].
#' @examples
#' mvdist_inverse_cdf(mvdist_normal(c(0, 0), diag(2)), c(0.5, 0.5))
#' @export
mvdist_inverse_cdf <- function(mv, p) {
  check_mvdist(mv)
  unname(mvdist_run(mv, "inverse_cdf", p)$values)
}

#' Rectangle probability of a multivariate normal
#'
#' `P(lower <= X <= upper)`, integrated via the ported Genz MVNDST algorithm. Available for
#' `"MultivariateNormal"` only.
#'
#' @param mv a `corehydro_mvdist` of family `"MultivariateNormal"`.
#' @param lower,upper numeric vectors of length [mvdist_dimension()].
#' @return a single numeric in `[0, 1]`.
#' @examples
#' mvdist_interval(mvdist_normal(c(0, 0), diag(2)), c(-100, -100), c(100, 100))
#' @export
mvdist_interval <- function(mv, lower, upper) {
  check_mvdist(mv)
  unname(mvdist_run(mv, "interval", c(as.double(lower), as.double(upper)))$values)
}

#' Marginal distribution of a multivariate normal
#'
#' Restricts a [mvdist_normal()] to a subset of its dimensions. Available for
#' `"MultivariateNormal"` only; every other family errors naming itself, not a raw C++ throw.
#'
#' @param mv a `corehydro_mvdist` of family `"MultivariateNormal"`.
#' @param indices the 1-based dimensions to keep.
#' @return a `corehydro_mvdist` over those dimensions.
#' @examples
#' mvdist_mean(mvdist_marginal(mvdist_normal(c(1, 2, 3), diag(3)), c(1, 3)))
#' @export
mvdist_marginal <- function(mv, indices) {
  check_mvdist(mv)
  idx <- mv_indices(indices, mvdist_dimension(mv), "indices")
  res <- mvdist_run(mv, "marginal", idx)
  new_mvdist("MultivariateNormal", res$spec)
}

#' Conditional distribution of a multivariate normal
#'
#' The distribution of the remaining dimensions of a [mvdist_normal()] given fixed values for a
#' subset. Available for `"MultivariateNormal"` only.
#'
#' @param mv a `corehydro_mvdist` of family `"MultivariateNormal"`.
#' @param given the 1-based dimensions being conditioned on.
#' @param values the values `given` is fixed at, the same length as `given`.
#' @return a `corehydro_mvdist` over the complement of `given`.
#' @examples
#' mvdist_mean(mvdist_conditional(mvdist_normal(c(1, 2, 3), diag(3)), given = 2, values = 5))
#' @export
mvdist_conditional <- function(mv, given, values) {
  check_mvdist(mv)
  if (length(given) != length(values)) {
    stop("`given` and `values` must have the same length", call. = FALSE)
  }
  idx <- mv_indices(given, mvdist_dimension(mv), "given")
  res <- mvdist_run(mv, "conditional", c(idx, as.double(values)))
  new_mvdist("MultivariateNormal", res$spec)
}

#' Draw from a multivariate distribution
#'
#' Simulate from the distribution's own seeded Mersenne Twister stream. A given `seed`
#' reproduces the same draws bit-for-bit in R, Python, and the upstream C# library.
#' `method = "latin_hypercube"` draws a Latin hypercube sample instead of ordinary Monte Carlo
#' and is available for `"MultivariateNormal"` and `"MultivariateStudentT"` only; it requires an
#' explicit `seed` (there is no clock-seeded LHS upstream).
#'
#' @param mv a `corehydro_mvdist`.
#' @param n number of draws.
#' @param seed integer seed for reproducible draws; `NULL` (the default) seeds from the clock,
#'   except for `method = "latin_hypercube"` where it is required.
#' @param method `"random"` (the default) for ordinary Monte Carlo, or `"latin_hypercube"`.
#' @return an `n x dimension` numeric matrix.
#' @examples
#' mvdist_random(mvdist_normal(c(0, 0), diag(2)), 5, seed = 12345)
#' @export
mvdist_random <- function(mv, n, seed = NULL, method = c("random", "latin_hypercube")) {
  check_mvdist(mv)
  method <- match.arg(method)
  if (identical(method, "latin_hypercube") && is.null(seed)) {
    stop("`seed` is required for method = \"latin_hypercube\"; there is no clock-seeded ",
         "Latin hypercube draw upstream", call. = FALSE)
  }
  seed <- if (is.null(seed)) -1L else as.integer(seed)
  m <- if (identical(method, "latin_hypercube")) "random_lhs" else "random"
  v <- mvdist_run(mv, m, c(as.integer(n), seed))$values
  matrix(unname(v), nrow = as.integer(n), byrow = TRUE)
}

#' Family-specific multivariate parameters
#'
#' The scalar or vector parameters specific to a multivariate family, beyond mean/covariance:
#' `df` for `"MultivariateStudentT"`; `alpha` and `alpha_sum` for `"Dirichlet"`; `trials` and
#' `probabilities` for `"Multinomial"`.
#'
#' @param mv a `corehydro_mvdist` of family `"MultivariateStudentT"`, `"Dirichlet"`, or
#'   `"Multinomial"`.
#' @return a named list with the entries relevant to `mv`'s family.
#' @examples
#' mvdist_params(mvdist_student_t(5, c(0, 0)))
#' mvdist_params(mvdist_dirichlet(c(2, 3, 4)))
#' mvdist_params(mvdist_multinomial(10, c(0.2, 0.3, 0.5)))
#' @export
mvdist_params <- function(mv) {
  check_mvdist(mv)
  switch(mv$family,
    MultivariateStudentT = list(df = unname(mvdist_run(mv, "degrees_of_freedom")$values)),
    Dirichlet = list(alpha = unname(mvdist_run(mv, "alpha")$values),
                     alpha_sum = unname(mvdist_run(mv, "alpha_sum")$values)),
    Multinomial = list(trials = as.integer(mvdist_run(mv, "number_of_trials")$values),
                       probabilities = unname(mvdist_run(mv, "mean")$values) /
                         as.integer(mvdist_run(mv, "number_of_trials")$values)),
    stop(sprintf("mvdist_params() has no family-specific parameters for '%s'", mv$family),
         call. = FALSE)
  )
}

#' List the supported multivariate distribution families
#'
#' @return a character vector of the five multivariate distribution family names.
#' @examples
#' mvdist_names()
#' @export
mvdist_names <- function() {
  kMvdistFamilies
}
