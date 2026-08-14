# The callback surface: ported Numerics routines whose input is a live R function rather than
# serializable data. Unlike every verb in toolbox.R/gof.R (which pass data through the shared
# run_toolbox dispatcher), these go through their own runner
# (core/include/corehydro/numerics/support/callback_runner.hpp) and its exception guard, reached by
# ch_callback_math_ in src/callback.cpp. Mirrors corehydropy's src/corehydropy/callback.py verb for
# verb.
#
# An error raised inside `f` reaches the caller unchanged: the guard latches the first one, and the
# runner rethrows it after the ported routine returns, so an internal C++ error provoked by the
# guard's own sentinel can never replace it.

# Internal: validate the function argument the same way for every verb.
callback_check_fn <- function(f) {
  if (!is.function(f)) {
    stop("`f` must be a function taking a number and returning a single number", call. = FALSE)
  }
}

# Rejects NA/NaN AND +/-Inf, and says so naming `x`. The finite half is not pedantry: a
# non-finite point reaches the shared spec serializer, which rejects it with "spec values must be
# finite" -- an error from three layers down that names neither the argument nor the verb. Kept
# character for character in step with corehydropy's _check_point.
callback_check_point <- function(x) {
  if (!is.numeric(x) || length(x) == 0L || !all(is.finite(x))) {
    stop("`x` must be a non-empty numeric vector of finite values", call. = FALSE)
  }
}

#' Find a root of a user-written function
#'
#' Solves `f(x) = 0` on `[lower, upper]` with the ported Numerics Brent root finder. `f` must
#' change sign across the interval.
#'
#' @param f a function taking one number and returning one number.
#' @param lower,upper the bracketing interval.
#' @param tolerance the convergence tolerance on the bracket width.
#' @param max_iterations the iteration cap; the search raises an error if it is reached.
#' @return the root, a single number.
#' @examples
#' root_find(function(x) x^2 - 2, lower = 0, upper = 2)
#' @export
root_find <- function(f, lower, upper, tolerance = 1e-8, max_iterations = 1000L) {
  callback_check_fn(f)
  if (!is.numeric(lower) || length(lower) != 1L || !is.finite(lower) ||
      !is.numeric(upper) || length(upper) != 1L || !is.finite(upper)) {
    stop("`lower` and `upper` must each be a single finite number", call. = FALSE)
  }
  if (lower >= upper) {
    stop("`lower` must be below `upper`", call. = FALSE)
  }
  if (!is.numeric(tolerance) || length(tolerance) != 1L || tolerance <= 0) {
    stop("`tolerance` must be a single positive number", call. = FALSE)
  }
  if (!is.numeric(max_iterations) || length(max_iterations) != 1L || max_iterations < 1) {
    stop("`max_iterations` must be a single positive integer", call. = FALSE)
  }
  opts <- to_spec_json(list(lower = as.double(lower), upper = as.double(upper),
                            tolerance = as.double(tolerance),
                            max_iterations = as.integer(max_iterations)))
  ch_callback_math_("root_find", opts, f)$values[[1]]
}

#' Integrate a user-written function
#'
#' Computes the definite integral of `f` over `[lower, upper]` with the ported Numerics
#' adaptive Gauss-Kronrod rule (10-point Gauss, 21-point Kronrod), which subdivides the
#' interval until the two nested estimates agree to the requested tolerance.
#'
#' The name is `quadrature()` rather than `integrate()` so it does not mask
#' [stats::integrate()].
#'
#' @param f a function taking one number and returning one number.
#' @param lower,upper the limits of integration. `upper` must be above `lower`; neither may be
#'   infinite (the ported rule integrates a finite interval).
#' @param absolute_tolerance,relative_tolerance the convergence tolerances on the difference
#'   between the Gauss and Kronrod estimates. Each must lie between 1e-15 and 1.
#' @param max_function_evaluations the cap on evaluations of `f`. Reaching it stops the
#'   subdivision and reports it in the status rather than raising an error.
#' @return the integral, a single number, carrying two attributes: `status`, one of `"Success"`,
#'   `"MaximumFunctionEvaluationsReached"`, `"MaximumIterationsReached"`, `"Failure"` or
#'   `"None"`; and `function_evaluations`, the number of times `f` was called.
#' @examples
#' quadrature(function(x) x^2, lower = 0, upper = 3)
#' q <- quadrature(sin, lower = 0, upper = pi)
#' attr(q, "status")
#' @export
quadrature <- function(f, lower, upper, absolute_tolerance = 1e-8, relative_tolerance = 1e-8,
                       max_function_evaluations = 10000000L) {
  callback_check_fn(f)
  if (!is.numeric(lower) || length(lower) != 1L || !is.finite(lower) ||
      !is.numeric(upper) || length(upper) != 1L || !is.finite(upper)) {
    stop("`lower` and `upper` must each be a single finite number", call. = FALSE)
  }
  if (lower >= upper) {
    stop("`lower` must be below `upper`", call. = FALSE)
  }
  if (!is.numeric(absolute_tolerance) || length(absolute_tolerance) != 1L ||
      absolute_tolerance < 1e-15 || absolute_tolerance > 1) {
    stop("`absolute_tolerance` must be a single number between 1e-15 and 1", call. = FALSE)
  }
  if (!is.numeric(relative_tolerance) || length(relative_tolerance) != 1L ||
      relative_tolerance < 1e-15 || relative_tolerance > 1) {
    stop("`relative_tolerance` must be a single number between 1e-15 and 1", call. = FALSE)
  }
  if (!is.numeric(max_function_evaluations) || length(max_function_evaluations) != 1L ||
      max_function_evaluations < 1) {
    stop("`max_function_evaluations` must be a single positive integer", call. = FALSE)
  }
  opts <- to_spec_json(list(lower = as.double(lower), upper = as.double(upper),
                            absolute_tolerance = as.double(absolute_tolerance),
                            relative_tolerance = as.double(relative_tolerance),
                            max_function_evaluations = as.integer(max_function_evaluations)))
  res <- ch_callback_math_("quadrature", opts, f)
  structure(res$values[[1]],
            status = res$status,
            function_evaluations = as.integer(res$values[[2]]))
}

#' Differentiate a user-written function
#'
#' `derivative()` takes the first derivative of a single-variable function by central difference;
#' `gradient()` and `hessian()` take the gradient and the Hessian matrix of a function of a
#' parameter vector. All three are the ported Numerics `NumericalDerivative` routines.
#'
#' @param f for `derivative()`, a function taking one number and returning one number; for
#'   `gradient()` and `hessian()`, a function taking a numeric vector and returning one number.
#' @param x the point to differentiate at: one number for `derivative()`, a numeric vector for
#'   `gradient()` and `hessian()`.
#' @param step_size the finite-difference step for `derivative()`. The default, any value at or
#'   below zero, selects the adaptive step `eps^(1/2) * (1 + |x|)`.
#' @return `derivative()` returns a single number, `gradient()` a numeric vector the length of
#'   `x`, and `hessian()` a square symmetric matrix.
#' @note `gradient()` and `hessian()` share their names with functions in \pkg{numDeriv} and
#'   \pkg{pracma}, so whichever of those packages is attached last masks this one (and its
#'   arguments differ). The examples below qualify every call with `corehydror::` so it is
#'   unambiguous which function is being called.
#' @examples
#' corehydror::derivative(function(x) x^3, 2)
#' rosenbrock <- function(p) (1 - p[1])^2 + 100 * (p[2] - p[1]^2)^2
#' corehydror::gradient(rosenbrock, c(1, 1))
#' corehydror::hessian(rosenbrock, c(1, 1))
#' @export
derivative <- function(f, x, step_size = -1) {
  callback_check_fn(f)
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x)) {
    stop("`x` must be a single finite number", call. = FALSE)
  }
  if (!is.numeric(step_size) || length(step_size) != 1L) {
    stop("`step_size` must be a single number", call. = FALSE)
  }
  opts <- to_spec_json(list(point = as.double(x), step_size = as.double(step_size)))
  ch_callback_math_("derivative", opts, f)$values[[1]]
}

#' @rdname derivative
#' @export
gradient <- function(f, x) {
  callback_check_fn(f)
  callback_check_point(x)
  opts <- to_spec_json(list(point = spec_array(as.double(x))))
  ch_callback_math_("gradient", opts, f)$values
}

#' @rdname derivative
#' @export
hessian <- function(f, x) {
  callback_check_fn(f)
  callback_check_point(x)
  opts <- to_spec_json(list(point = spec_array(as.double(x))))
  res <- ch_callback_math_("hessian", opts, f)
  matrix(res$values, nrow = res$dims[[1]], ncol = res$dims[[2]], byrow = TRUE)
}
