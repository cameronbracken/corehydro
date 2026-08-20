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

#' Sample your own posterior by MCMC
#'
#' Runs any of the ported MCMC samplers against a log-likelihood you write and priors you
#' choose. This is the constructor the upstream C# library itself exposes,
#' `MCMCSampler(priorDistributions, logLikelihoodFunction)`: [mcmc_sample()] can only fit a
#' built-in distribution family under uniform priors spanning its parameter constraints, while
#' this function takes any model you can write down.
#'
#' `log_likelihood` is called with one numeric vector, as long as `priors`, and must return a
#' single number. Following the upstream contract, it should return the log of the data
#' likelihood PLUS the log prior density; the `priors` list is used for the feasible parameter
#' bounds and for chain initialization, and is never added to your value behind your back. A
#' parameter outside its prior's support is rejected before your function sees it, so a flat
#' (uniform) prior needs no term of its own.
#'
#' @param log_likelihood a function taking a numeric parameter vector and returning a single
#'   number.
#' @param priors a list of [distribution()] objects, one per parameter, in the order
#'   `log_likelihood` reads them. A single distribution is accepted for a one-parameter model.
#' @param sampler one of `"RWMH"` (the default), `"ARWMH"`, `"DEMCz"`, `"DEMCzs"`, `"HMC"`,
#'   `"NUTS"`, or `"SNIS"`. `"Gibbs"` needs a proposal function and is not available yet.
#' @param iterations iterations per chain (sampler default if `NULL`). `"SNIS"` needs at least
#'   10000: its ported validation requires `iterations` to be at least the output length, and the
#'   output length keeps its default of 10000 here.
#' @param warmup warm-up iterations discarded from each chain (sampler default if `NULL`). When
#'   `iterations` is given and `warmup` is not, half of `iterations` is used, matching
#'   [mcmc_sample()]. `"SNIS"` is the exception: its ported validation rejects any warm-up at all,
#'   so none is derived for it.
#' @param chains number of chains (sampler default if `NULL`). `"DEMCz"` and `"DEMCzs"` require
#'   at least three; `"SNIS"` draws independently and runs one.
#' @param thinning thinning interval (sampler default if `NULL`).
#' @param seed PRNG seed; `12345` is the C# default.
#' @param initialize chain initialization: `"MAP"` (from the posterior-mode estimate, the C#
#'   default) or `"Randomize"` (draws from the priors).
#' @return A list with the same fields [mcmc_sample()] returns: `parameters` (parameter names,
#'   `p1`, `p2`, ... since your model names nothing), `chains` (a list of one
#'   draws-by-parameters matrix per chain), `acceptance_rates`, `map`, `map_fitness`,
#'   `posterior_mean`, `posterior_sd`, `posterior_median`, `posterior_lower_ci`,
#'   `posterior_upper_ci`, `rhat`, and `ess`.
#' @section Reproducing a run in Python:
#' A seeded [mcmc_sample()] run is bit-identical between R and Python, because every arithmetic
#' operation happens in the shared C++ core. That guarantee is WEAKER here, and it is worth
#' stating plainly. The draws still come from the core's seeded Mersenne Twister, but the
#' log-density is your own R code, and R and Python do not guarantee identical rounding for the
#' same formula. MCMC amplifies a single differing bit far harder than an optimizer does: one
#' flipped accept-or-reject changes every state after it, so the two chains diverge outright
#' rather than drift apart slowly.
#'
#' A seeded run reproduces across the two languages if and only if your function returns
#' bit-identical values. Arithmetic (`+ - * /`) is IEEE-deterministic and does reproduce; `log`,
#' `exp`, `gamma` and friends come from each platform's own math library and are not guaranteed
#' to. Note also that R's `sum()` accumulates in extended precision while Python's does not, so
#' an explicit loop or `Reduce()` is the portable spelling.
#' @section Performance:
#' Every evaluation calls back into R, and there are far more of them than `iterations` suggests:
#' the count is `(iterations + output_length / chains) * thinning * chains`, and the ported
#' defaults are 4 chains, a thinning interval of 20 and an output length of 10,000. So
#' `iterations = 10000` is a million crossings, not ten thousand.
#'
#' That is affordable. Measured here, `iterations = 10000` over a 50-point Gaussian log-density
#' took 4.5 seconds, against 4.9 seconds for the [mcmc_sample()] call in the same units -- the
#' callback path is not the slower one, because `mcmc_sample()`'s built-in log-likelihood rebuilds
#' a distribution object on every evaluation while a small R closure does not. The crossing itself
#' costs about 4 microseconds: the same closure called a million times directly from R takes 0.3
#' seconds, so roughly 4 of the 4.5 seconds is the boundary and not your code.
#'
#' Two settings dominate the count and neither is obvious. `thinning` multiplies it, so
#' `thinning = 1` turned the same run into 0.26 seconds. And `initialize = "MAP"`, the C# default,
#' runs the DifferentialEvolution optimizer over your function before the first chain iteration;
#' `initialize = "Randomize"` skips the optimizer.
#' @section Interrupting a long run:
#' Ctrl-C returns control with an interrupt condition, but not instantly. The ported samplers have
#' no cancellation hook, so the chain runs to the end of its loop -- rejecting every remaining
#' point without calling your function again -- before the interrupt surfaces. Measured on a
#' 100,000-iteration chain: 4.2 seconds from Ctrl-C to the prompt.
#' @seealso [mcmc_sample()] for a built-in family under constraint-based priors, which is faster
#'   and bit-identical across languages.
#' @examples
#' \donttest{
#' set.seed(1)
#' x <- rnorm(50, mean = 5)
#' ll <- function(p) -0.5 * sum((x - p[1])^2)
#' fit <- mcmc_posterior(ll, list(distribution("Uniform", c(0, 10))),
#'                       iterations = 500, seed = 12345)
#' fit$posterior_mean
#' }
#' @export
mcmc_posterior <- function(
  log_likelihood,
  priors,
  sampler = c("RWMH", "ARWMH", "DEMCz", "DEMCzs", "HMC", "NUTS", "SNIS"),
  iterations = NULL,
  warmup = NULL,
  chains = NULL,
  thinning = NULL,
  seed = 12345,
  initialize = c("MAP", "Randomize")
) {
  if (!is.function(log_likelihood)) {
    stop("`log_likelihood` must be a function taking a parameter vector and returning a single number",
         call. = FALSE)
  }
  sampler <- match.arg(sampler)
  initialize <- match.arg(initialize)
  priors <- mcmc_prior_list(priors)

  opts <- list(
    sampler = sampler,
    initialize = initialize,
    seed = as.integer(seed),
    priors = unname(priors)
  )
  if (!is.null(iterations)) {
    opts$iterations <- as.integer(iterations)
    # The sampler requires warmup <= iterations / 2; when only iterations is given, follow
    # mcmc_sample()'s rule. SNIS is the exception: its ported ValidateSettings rejects ANY
    # warm-up, so an auto-derived one would turn a legal call into an error.
    if (is.null(warmup) && !identical(sampler, "SNIS")) {
      warmup <- max(50L, as.integer(iterations) %/% 2L)
    }
  }
  if (!is.null(warmup)) opts$warmup <- as.integer(warmup)
  if (!is.null(chains)) {
    if (identical(sampler, "SNIS") && as.integer(chains) != 1L) {
      stop("SNIS draws independently rather than running Markov chains; it supports `chains = 1` only",
           call. = FALSE)
    }
    opts$chains <- as.integer(chains)
  }
  if (!is.null(thinning)) opts$thinning <- as.integer(thinning)
  if (identical(sampler, "RWMH")) {
    # The RWMH constructor takes a proposal covariance, and the ported default is all zeros,
    # which is only usable when MAP initialization overwrites it before the first iteration.
    # Identity is what mcmc_sample() sets for the same reason.
    opts$proposal_sigma <- "identity"
  }

  mcmc_unflatten(ch_callback_mcmc_(to_spec_json(opts), log_likelihood))
}

# Internal: accept either a list of distribution() objects or a single one, and refuse anything
# else by name. The length of this list IS the parameter count, so a wrong one is the likeliest
# user error here and the C++ side cannot tell it from an intentional model.
mcmc_prior_list <- function(priors) {
  if (inherits(priors, "corehydro_dist")) priors <- list(priors)
  if (!is.list(priors) || length(priors) == 0L ||
      !all(vapply(priors, inherits, logical(1), "corehydro_dist"))) {
    stop("`priors` must be a non-empty list of distribution() objects, one per parameter",
         call. = FALSE)
  }
  priors
}

# Internal: slice the flat callback result back into the shape mcmc_sample() returns. The layout
# is documented in core/include/corehydro/numerics/support/callback/mcmc.hpp -- a named summary
# block, then the draws row-major by [chain][draw][parameter] -- and corehydropy's own
# _mcmc_unflatten() reads it identically.
mcmc_unflatten <- function(res) {
  dims <- as.integer(unlist(res$dims))
  n_summary <- dims[[1]]
  n_chains <- dims[[2]]
  n_draws <- dims[[3]]
  p <- dims[[4]]
  values <- as.double(unlist(res$values))
  nms <- as.character(unlist(res$names))

  by_name <- function(name) as.double(values[[match(name, nms)]])
  per_parameter <- function(prefix) {
    as.double(values[match(sprintf("%s[%d]", prefix, seq_len(p) - 1L), nms)])
  }

  draws <- values[seq.int(n_summary + 1L, length.out = n_chains * n_draws * p)]
  chains <- lapply(seq_len(n_chains), function(k) {
    block <- draws[seq.int((k - 1L) * n_draws * p + 1L, length.out = n_draws * p)]
    matrix(block, nrow = n_draws, ncol = p, byrow = TRUE)
  })

  list(
    parameters = paste0("p", seq_len(p)),
    chains = chains,
    acceptance_rates = as.double(values[match(
      sprintf("acceptance_rate[%d]", seq_len(n_chains) - 1L), nms
    )]),
    map = per_parameter("map"),
    map_fitness = by_name("map_fitness"),
    posterior_mean = per_parameter("posterior_mean"),
    posterior_sd = per_parameter("posterior_sd"),
    posterior_median = per_parameter("posterior_median"),
    posterior_lower_ci = per_parameter("posterior_lower_ci"),
    posterior_upper_ci = per_parameter("posterior_upper_ci"),
    rhat = per_parameter("rhat"),
    ess = per_parameter("ess")
  )
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
