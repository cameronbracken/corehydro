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
#' @param tolerance the convergence tolerance on the bracket width. `NULL`, the default, leaves the
#'   ported Brent solver's own default (1e-8) in force; the value is not restated here, so a change
#'   to it lands in one place.
#' @param max_iterations the iteration cap; the search raises an error if it is reached. `NULL`,
#'   the default, leaves the ported solver's own default (1000) in force.
#' @return the root, a single number.
#' @examples
#' root_find(function(x) x^2 - 2, lower = 0, upper = 2)
#' @export
root_find <- function(f, lower, upper, tolerance = NULL, max_iterations = NULL) {
  callback_check_fn(f)
  if (!is.numeric(lower) || length(lower) != 1L || !is.finite(lower) ||
      !is.numeric(upper) || length(upper) != 1L || !is.finite(upper)) {
    stop("`lower` and `upper` must each be a single finite number", call. = FALSE)
  }
  if (lower >= upper) {
    stop("`lower` must be below `upper`", call. = FALSE)
  }
  # An option key is written ONLY when the caller supplied it, so an unset argument reaches the
  # ported routine's own default rather than a copy of that default made here. Mirrors
  # corehydropy's root_find() and the same rule in callback/math.hpp and the oracle emitter.
  opts <- list(lower = as.double(lower), upper = as.double(upper))
  if (!is.null(tolerance)) {
    if (!is.numeric(tolerance) || length(tolerance) != 1L || tolerance <= 0) {
      stop("`tolerance` must be a single positive number", call. = FALSE)
    }
    opts$tolerance <- as.double(tolerance)
  }
  if (!is.null(max_iterations)) {
    if (!is.numeric(max_iterations) || length(max_iterations) != 1L || max_iterations < 1) {
      stop("`max_iterations` must be a single positive integer", call. = FALSE)
    }
    opts$max_iterations <- as.integer(max_iterations)
  }
  ch_callback_math_("root_find", to_spec_json(opts), f)$values[[1]]
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
#'   between the Gauss and Kronrod estimates. Each must lie between 1e-15 and 1. `NULL`, the
#'   default, leaves the ported integrator's own defaults (1e-8) in force.
#' @param max_function_evaluations the cap on evaluations of `f`. Reaching it stops the
#'   subdivision and reports it in the status rather than raising an error. `NULL`, the default,
#'   leaves the ported integrator's own default in force.
#' @return the integral, a single number, carrying three attributes: `status`, one of `"Success"`,
#'   `"MaximumFunctionEvaluationsReached"`, `"MaximumIterationsReached"`, `"Failure"` or
#'   `"None"`; `function_evaluations`, the number of times `f` was called; and `standard_error`,
#'   the rule's own error estimate (the square root of the accumulated squared differences between
#'   the Gauss and Kronrod estimates, zero when the interval never needed subdividing).
#' @examples
#' quadrature(function(x) x^2, lower = 0, upper = 3)
#' q <- quadrature(sin, lower = 0, upper = pi)
#' attr(q, "status")
#' attr(q, "standard_error")
#' @export
quadrature <- function(f, lower, upper, absolute_tolerance = NULL, relative_tolerance = NULL,
                       max_function_evaluations = NULL) {
  callback_check_fn(f)
  if (!is.numeric(lower) || length(lower) != 1L || !is.finite(lower) ||
      !is.numeric(upper) || length(upper) != 1L || !is.finite(upper)) {
    stop("`lower` and `upper` must each be a single finite number", call. = FALSE)
  }
  if (lower >= upper) {
    stop("`lower` must be below `upper`", call. = FALSE)
  }
  # See root_find() above: a key is written only when the caller supplied it.
  opts <- list(lower = as.double(lower), upper = as.double(upper))
  if (!is.null(absolute_tolerance)) {
    if (!is.numeric(absolute_tolerance) || length(absolute_tolerance) != 1L ||
        absolute_tolerance < 1e-15 || absolute_tolerance > 1) {
      stop("`absolute_tolerance` must be a single number between 1e-15 and 1", call. = FALSE)
    }
    opts$absolute_tolerance <- as.double(absolute_tolerance)
  }
  if (!is.null(relative_tolerance)) {
    if (!is.numeric(relative_tolerance) || length(relative_tolerance) != 1L ||
        relative_tolerance < 1e-15 || relative_tolerance > 1) {
      stop("`relative_tolerance` must be a single number between 1e-15 and 1", call. = FALSE)
    }
    opts$relative_tolerance <- as.double(relative_tolerance)
  }
  if (!is.null(max_function_evaluations)) {
    if (!is.numeric(max_function_evaluations) || length(max_function_evaluations) != 1L ||
        max_function_evaluations < 1) {
      stop("`max_function_evaluations` must be a single positive integer", call. = FALSE)
    }
    opts$max_function_evaluations <- as.integer(max_function_evaluations)
  }
  res <- ch_callback_math_("quadrature", to_spec_json(opts), f)
  structure(res$values[[1]],
            status = res$status,
            function_evaluations = as.integer(res$values[[2]]),
            standard_error = res$values[[3]])
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
#' @param step_size the finite-difference step for `derivative()`. `NULL`, the default, leaves the
#'   ported routine's own step selection in force, as does any value at or below zero: the adaptive
#'   step `eps^(1/2) * (1 + |x|)`.
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
derivative <- function(f, x, step_size = NULL) {
  callback_check_fn(f)
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x)) {
    stop("`x` must be a single finite number", call. = FALSE)
  }
  # See root_find() above: `step_size` is written only when the caller supplied it.
  opts <- list(point = as.double(x))
  if (!is.null(step_size)) {
    if (!is.numeric(step_size) || length(step_size) != 1L) {
      stop("`step_size` must be a single number", call. = FALSE)
    }
    opts$step_size <- as.double(step_size)
  }
  ch_callback_math_("derivative", to_spec_json(opts), f)$values[[1]]
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

#' Draw from the generator a callback is handed
#'
#' Some corehydro verbs call a function you write and hand it the seeded random number generator
#' the run is using: the proposal function of a Gibbs sampler, the resample function of a
#' bootstrap. `rng_uniform()` and `rng_integers()` are how that generator is drawn from.
#'
#' Use them for every random number your callback needs. Reaching for R's own generator
#' ([stats::runif()], [base::sample()]) instead is not an error and will not be caught, but it
#' silently breaks two guarantees corehydro otherwise makes: the run stops being reproducible from
#' its seed, and it stops agreeing with the identical run in Python. The handle draws from the same
#' Mersenne Twister the core seeded, so both hold.
#'
#' @param rng the handle your callback was given. It cannot be created any other way.
#' @param n how many values to draw; a single positive whole number. A fractional `n` is an error,
#'   not a silent truncation, and the identical call in Python is refused the same way.
#' @param min,max the bounds for `rng_integers()`. `min` is included and `max` is EXCLUDED,
#'   matching the ported Numerics `Next(minInclusive, maxExclusive)` a C# proposal is written
#'   against, so `rng_integers(rng, 1, 0, 10)` draws one of `0:9`. Both must be whole numbers, and
#'   the range between them can be at most 2147483647 wide (the ported generator draws an integer
#'   span, and C# throws for a wider one).
#' @return `rng_uniform()` a numeric vector of length `n` with values in `[0, 1)`;
#'   `rng_integers()` an integer vector of length `n`.
#' @section Lifetime:
#' The handle borrows the generator for the duration of the one call it was given to. It is not an
#' object to keep. Storing it and drawing from it after your callback has returned raises an error
#' ("this random number generator handle is no longer valid"), which is deliberate: the generator
#' it pointed at no longer exists, and reading it would crash the session rather than merely
#' misbehave.
#' @seealso [stats::runif()] for ordinary R random numbers, which is what to use anywhere OUTSIDE
#'   a corehydro callback.
#' @name rng_uniform
NULL

#' @rdname rng_uniform
#' @export
rng_uniform <- function(rng, n) {
  ch_rng_uniform_(rng, rng_check_count(n))
}

#' @rdname rng_uniform
#' @export
rng_integers <- function(rng, n, min, max) {
  if (!rng_is_whole(min) || !rng_is_whole(max)) {
    stop("`min` and `max` must each be a single finite whole number", call. = FALSE)
  }
  ch_rng_integers_(rng, rng_check_count(n), as.integer(min), as.integer(max))
}

# Internal: the count check both verbs share, worded exactly as the core's own check so a caller
# cannot tell which layer refused. Kept in step with corehydropy's _check_count -- including the
# refusal of a fractional count: R would happily truncate `n = 2.7` to 2 where Python raises, and a
# silent truncation in one language and an error in the other is exactly the cross-language
# disagreement this surface exists to prevent.
rng_check_count <- function(n) {
  if (!rng_is_whole(n) || n < 1) {
    stop("`n` must be a single positive whole number", call. = FALSE)
  }
  as.integer(n)
}

# A single finite number with nothing after the decimal point, and inside integer range (beyond it
# as.integer() gives NA with a warning). 2 and 2L both pass; 2.7 does not.
rng_is_whole <- function(x) {
  is.numeric(x) && length(x) == 1L && is.finite(x) &&
    x == trunc(x) && abs(x) <= .Machine$integer.max
}

# Internal, test-only: seed a generator, hand `f` a handle on it, return what `f` drew. `f` takes
# (parameters, rng), the Gibbs proposal's own signature, so what the fixtures prove here about the
# handle carries over to the samplers. Not exported -- a user reaches the handle through a real
# verb, never through this. Reached from the tests and the fixture runner via
# asNamespace("corehydror"), the same way the ch_* entry points are.
rng_probe <- function(seed, parameters, f) {
  callback_check_fn(f)
  opts <- list(seed = as.double(seed))
  if (length(parameters) > 0L) opts$parameters <- spec_array(as.double(parameters))
  ch_rng_probe_(to_spec_json(opts), f)$values
}
