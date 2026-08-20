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
#' @seealso [mcmc_posterior()], whose `proposal` argument is the verb that hands you a handle, and
#'   [stats::runif()] for ordinary R random numbers, which is what to use anywhere OUTSIDE a
#'   corehydro callback.
#' @examples
#' # The Gibbs proposal is the surface that hands you a handle. This model's full conditional
#' # really is uniform: with x_i ~ Uniform(mu - 1, mu + 1) and a flat prior, mu given the data is
#' # Uniform(max(x) - 1, min(x) + 1), so one uniform draw IS the Gibbs step.
#' x <- c(4.9, 5.1, 5.0, 5.2, 4.8)
#' ll <- function(p) if (all(abs(x - p[1]) <= 1)) 0 else -Inf
#' proposal <- function(parameters, rng) {
#'   lo <- max(x) - 1
#'   hi <- min(x) + 1
#'   lo + rng_uniform(rng, 1) * (hi - lo)
#' }
#' fit <- mcmc_posterior(ll, distribution("Uniform", c(0, 10)),
#'                       sampler = "Gibbs", proposal = proposal,
#'                       iterations = 200, seed = 12345, initialize = "Randomize")
#' fit$posterior_mean
#'
#' # rng_integers() draws whole numbers on [min, max) off the same stream -- a resampling
#' # proposal, say, picking one of the observations by index.
#' pick_one <- function(parameters, rng) x[rng_integers(rng, 1, 1, length(x) + 1)]
#' mcmc_posterior(ll, distribution("Uniform", c(0, 10)),
#'                sampler = "Gibbs", proposal = pick_one,
#'                iterations = 200, seed = 12345, initialize = "Randomize")$posterior_mean
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
#'   `"NUTS"`, `"SNIS"`, or `"Gibbs"`. `"Gibbs"` requires `proposal`.
#' @param proposal the conditional proposal function `"Gibbs"` samples with, and the only sampler
#'   that takes one. It is called as `proposal(parameters, rng)` and must return a numeric vector
#'   as long as `priors`: the next state of the chain, which Gibbs accepts unconditionally.
#'   Ordinarily that is a draw from the full conditional of the model. `rng` is a handle on the
#'   generator this chain is running on -- draw from it with [rng_uniform()] and [rng_integers()],
#'   not with [stats::runif()], or the seeded run stops being reproducible.
#' @param gradient an analytic gradient of `log_likelihood` for `"HMC"` and `"NUTS"`, the only
#'   samplers that take one. It is called as `gradient(parameters)` and must return a numeric
#'   vector as long as `priors`. Left `NULL`, both samplers use the ported bound-aware
#'   finite-difference gradient, which costs two extra `log_likelihood` calls per parameter per
#'   leapfrog step; an analytic gradient is usually a large saving and is always more accurate.
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
#'
#' `"Gibbs"` is the sampler whose defaults surprise: the ported constructor sets one chain, no
#' thinning, and 100,000 iterations on top of a 10,000-draw output block, so an `iterations` you do
#' not set is 110,000 iterations of BOTH your log-likelihood and your proposal. Set `iterations`.
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
#' # A plain loop and `+ - * /` alone, the portable spelling described above.
#' ll <- function(p) {
#'   acc <- 0
#'   for (xi in x) acc <- acc + (xi - p[1]) * (xi - p[1])
#'   -0.5 * acc
#' }
#' fit <- mcmc_posterior(ll, list(distribution("Uniform", c(0, 10))),
#'                       iterations = 500, seed = 12345)
#' fit$posterior_mean
#' }
#' @export
mcmc_posterior <- function(
  log_likelihood,
  priors,
  sampler = c("RWMH", "ARWMH", "DEMCz", "DEMCzs", "HMC", "NUTS", "SNIS", "Gibbs"),
  proposal = NULL,
  gradient = NULL,
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
  if (is.null(seed)) {
    stop("`seed` must not be NULL", call. = FALSE)
  }
  sampler <- match.arg(sampler)
  initialize <- match.arg(initialize)
  priors <- mcmc_prior_list(priors)
  # Each optional delegate belongs to specific samplers, and a mismatch is refused rather than
  # ignored: a user who writes a gradient and leaves `sampler` at its default would otherwise get
  # a plausible run that never called it. Worded to match corehydropy's mcmc_posterior().
  if (identical(sampler, "Gibbs") && is.null(proposal)) {
    stop("the Gibbs sampler requires a `proposal` function; it has no default conditional draw",
         call. = FALSE)
  }
  if (!is.null(proposal)) {
    if (!is.function(proposal)) {
      stop("`proposal` must be a function taking (parameters, rng) and returning a parameter vector",
           call. = FALSE)
    }
    if (!identical(sampler, "Gibbs")) {
      stop(sprintf("`proposal` is only used by the Gibbs sampler; '%s' does not take one", sampler),
           call. = FALSE)
    }
  }
  if (!is.null(gradient)) {
    if (!is.function(gradient)) {
      stop("`gradient` must be a function taking a parameter vector and returning one",
           call. = FALSE)
    }
    if (!sampler %in% c("HMC", "NUTS")) {
      stop(sprintf("`gradient` is only used by the HMC and NUTS samplers; '%s' does not take one",
                   sampler),
           call. = FALSE)
    }
  }

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

  mcmc_unflatten(ch_callback_mcmc_(to_spec_json(opts), log_likelihood, proposal, gradient))
}

#' Bootstrap your own statistic
#'
#' Runs the ported Numerics bootstrap against resampling, fitting and statistic functions you
#' write. This is the class upstream exposes as four delegates -- `ResampleFunction`,
#' `FitFunction`, `StatisticFunction` and `JackknifeFunction` -- so any quantity you can compute
#' from a fitted parameter set can be given a confidence interval, not just the built-in
#' distribution quantiles [bootstrap_analysis()] covers.
#'
#' @section The four functions:
#' Getting an argument order wrong is the likeliest mistake here, and the C++ side cannot tell a
#' swapped pair from a deliberate one, so each signature is given exactly:
#'
#' \describe{
#'   \item{`resample(data, parameters, rng)`}{Returns one bootstrap sample. `data` is the original
#'     sample, `parameters` is the current parameter vector, and `rng` is a handle on THIS
#'     replicate's generator -- draw from it with [rng_uniform()] and [rng_integers()], not with
#'     [stats::runif()] or [base::sample()], or the seeded run stops being reproducible and stops
#'     agreeing with Python. The returned sample need not be the same length as `data`.}
#'   \item{`fit(data)`}{Returns the parameter vector fitted to `data`: one number per parameter,
#'     the same count every time.}
#'   \item{`statistic(parameters)`}{Returns the numbers to put intervals on, computed from a fitted
#'     parameter vector: one or more, the same count every time.}
#'   \item{`jackknife(data, index)`}{Returns `data` with observation `index` left out. `index`
#'     counts from 0, matching the ported delegate, so the R spelling is `data[-(index + 1)]` --
#'     the naive `data[-index]` is wrong for every value of `index`, not just one: at `index = 0`
#'     it is `data[-0]`, which R evaluates to `numeric(0)`, the EMPTY vector, and that is refused
#'     by name; at every later index it is the right LENGTH but drops the wrong observation (one
#'     off), which nothing can catch. Only the `"BCa"` method uses it; every other method ignores
#'     it.}
#' }
#'
#' @param data the original sample: a non-empty numeric vector.
#' @param resample,fit,statistic the three required functions, with the signatures above.
#' @param jackknife the leave-one-out function, required by `ci_method = "BCa"` and unused by every
#'   other method. `NULL` by default.
#' @param replicates number of bootstrap replicates.
#' @param alpha the interval's total tail probability: `0.1` gives a 90% interval, `alpha / 2` in
#'   each tail.
#' @param ci_method one of `"Percentile"` (the default), `"BiasCorrected"`, `"Normal"`,
#'   `"BootstrapT"` or `"BCa"`. `"Normal"` and `"BootstrapT"` work on the ported cube-root
#'   transform of the statistic; `"BootstrapT"` runs the studentized workflow, which nests
#'   `inner_replicates` further resample-and-fit pairs inside every replicate.
#' @param seed PRNG seed; `12345` is the C# default.
#' @param parameters the original parameter vector the replicates are compared against. `NULL`, the
#'   default, uses `fit(data)`, which is what the bootstrap ordinarily means by it.
#' @param inner_replicates inner replicates for `ci_method = "BootstrapT"`, ignored by every other
#'   method. `NULL` leaves the ported default (300) in force.
#' @param max_retries the maximum number of times a single failed replicate is retried before it
#'   is counted in `failed_replicates`. `NULL` leaves the ported default (`MaxRetries`, 20) in
#'   force. Each retry is another crossing into R, so lowering it caps the worst case rather than
#'   changing the typical one.
#' @return A list with, per statistic, `estimate` (the statistic of `parameters`, not a bootstrap
#'   average), `lower`, `upper`, `standard_error`, `mean` (the mean over valid replicates, so
#'   `mean - estimate` is the bias estimate) and `valid_count`; the same first three for the fitted
#'   parameters as `parameter_estimate`, `parameter_lower` and `parameter_upper`; and
#'   `replicates`, `failed_replicates`, `alpha` and `ci_method`.
#' @section How many times your functions are called:
#' `replicates` calls of each of `resample`, `fit` and `statistic`, plus one extra `statistic` call
#' to learn how many values it returns and one `fit` call when `parameters` is not supplied. A
#' failed replicate is retried up to `max_retries` times (20 by default), and `"BCa"` adds one
#' `jackknife` + `fit` + `statistic` per observation. `"BootstrapT"` is the expensive one: it
#' multiplies the resample and fit counts by `inner_replicates`, so the ported defaults (10,000 x
#' 300) would be three million crossings back into R. Start small.
#' @section Reproducing a run in Python:
#' The draws come from the core's seeded Mersenne Twister, so an identical `bootstrap_custom()`
#' call in Python resamples the identical observations. The numbers your functions compute from
#' them are your own R code, though, and R and Python do not guarantee identical rounding for the
#' same formula: arithmetic (`+ - * /`) is IEEE-deterministic and does reproduce, while `log`,
#' `exp` and friends come from each platform's math library. Note also that R's `sum()` and
#' `mean()` accumulate in extended precision where Python does not, so an explicit loop is the
#' portable spelling.
#' @seealso [bootstrap_analysis()] for the built-in parametric bootstrap of a fitted distribution's
#'   quantiles, and [rng_uniform()] for drawing inside `resample`.
#' @examples
#' \donttest{
#' x <- c(4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7)
#' # An ordinary iid bootstrap of the mean. rng_integers() draws on [0, n), counting from 0, so
#' # the index is shifted by one for R.
#' res <- bootstrap_custom(
#'   data = x,
#'   resample = function(data, parameters, rng) {
#'     data[rng_integers(rng, length(data), 0, length(data)) + 1L]
#'   },
#'   fit = function(data) {
#'     acc <- 0
#'     for (xi in data) acc <- acc + xi
#'     acc / length(data)
#'   },
#'   statistic = function(parameters) parameters,
#'   replicates = 500, seed = 12345
#' )
#' c(res$lower, res$estimate, res$upper)
#' }
#' @export
bootstrap_custom <- function(
  data,
  resample,
  fit,
  statistic,
  jackknife = NULL,
  replicates = 1000,
  alpha = 0.1,
  ci_method = c("Percentile", "BiasCorrected", "Normal", "BootstrapT", "BCa"),
  seed = 12345,
  parameters = NULL,
  inner_replicates = NULL,
  max_retries = NULL
) {
  if (!is.numeric(data) || length(data) == 0L || !all(is.finite(data))) {
    stop("`data` must be a non-empty numeric vector of finite values", call. = FALSE)
  }
  bootstrap_check_fn(resample, "resample", "(data, parameters, rng)")
  bootstrap_check_fn(fit, "fit", "(data)")
  bootstrap_check_fn(statistic, "statistic", "(parameters)")
  if (!is.null(jackknife)) bootstrap_check_fn(jackknife, "jackknife", "(data, index)")
  ci_method <- match.arg(ci_method)
  # Refused HERE rather than after the run: the ported class checks it inside
  # GetConfidenceIntervals, by which point every replicate has already called back into R. The
  # core repeats this check for all four runners; the wording is the same in both packages.
  if (identical(ci_method, "BCa") && is.null(jackknife)) {
    stop("the BCa confidence interval method requires a `jackknife` function, called with ",
         "(data, index); supply one or choose another `ci_method`", call. = FALSE)
  }
  if (!is.numeric(replicates) || length(replicates) != 1L || replicates < 1) {
    stop("`replicates` must be a single positive whole number", call. = FALSE)
  }
  if (!is.numeric(alpha) || length(alpha) != 1L || alpha <= 0 || alpha >= 1) {
    stop("`alpha` must be a single number between 0 and 1", call. = FALSE)
  }
  if (is.null(seed)) {
    stop("`seed` must not be NULL", call. = FALSE)
  }

  opts <- list(
    data = spec_array(as.double(data)),
    replicates = as.integer(replicates),
    alpha = as.double(alpha),
    ci_method = ci_method,
    seed = as.integer(seed)
  )
  if (!is.null(parameters)) {
    if (!is.numeric(parameters) || length(parameters) == 0L || !all(is.finite(parameters))) {
      stop("`parameters` must be a non-empty numeric vector of finite values, or NULL",
           call. = FALSE)
    }
    opts$parameters <- spec_array(as.double(parameters))
  }
  if (!is.null(inner_replicates)) {
    if (!is.numeric(inner_replicates) || length(inner_replicates) != 1L || inner_replicates < 1) {
      stop("`inner_replicates` must be a single positive whole number", call. = FALSE)
    }
    opts$inner_replicates <- as.integer(inner_replicates)
  }
  if (!is.null(max_retries)) {
    if (!is.numeric(max_retries) || length(max_retries) != 1L || max_retries < 1) {
      stop("`max_retries` must be a single positive whole number", call. = FALSE)
    }
    opts$max_retries <- as.integer(max_retries)
  }

  res <- ch_callback_bootstrap_(to_spec_json(opts), resample, fit, statistic, jackknife)
  bootstrap_unflatten(res, ci_method)
}

#' Fit your own moment conditions by GMM
#'
#' Estimates parameters by the generalized method of moments against moment conditions you write.
#' This is the constructor the upstream C# library itself exposes,
#' `GeneralizedMethodOfMoments(momentConditionFunction, ...)`: [fit_gmm()] can only fit a
#' [model_bulletin17c()] model (the single implementation of the `IGMMModel` interface it takes),
#' while this function takes any moment conditions you can write down.
#'
#' @section The moment condition function:
#' `moment_conditions(parameters)` is called with one numeric vector, as long as `initial`, and must
#' return a **list with two elements**:
#'
#' \describe{
#'   \item{`g`}{the sample mean of the moment conditions at those parameters: a numeric vector of
#'     length q, one entry per moment condition. GMM drives this towards zero.}
#'   \item{`s`}{their covariance: a q by q numeric **matrix**. In two-step and iterative GMM the
#'     optimal weighting matrix is its inverse.}
#' }
#'
#' Both are required and both are checked by name, because returning the wrong thing here is the
#' likeliest mistake on this surface. A two-parameter method-of-moments fit of a Normal, whose
#' answer is the sample mean and the population variance:
#'
#' ```r
#' avg <- function(v) {  # not mean(): see "Reproducing a run in Python" below
#'   total <- 0
#'   for (value in v) total <- total + value
#'   total / length(v)
#' }
#' moments <- function(p) {
#'   a <- x - p[1]
#'   b <- a * a - p[2]
#'   list(g = c(avg(a), avg(b)),
#'        s = matrix(c(avg(a * a), avg(a * b), avg(a * b), avg(b * b)), 2, 2))
#' }
#' ```
#'
#' `q` (the number of moment conditions) is measured by calling your function once at `initial`, so
#' there is no argument to get wrong. When q equals the number of parameters the fit is
#' just-identified and `$j_stat_pval` comes back `NA` -- see the return value below.
#'
#' @param moment_conditions the required function described above.
#' @param initial numeric vector of starting values, one per parameter. Its length IS the parameter
#'   count.
#' @param lower,upper numeric vectors of parameter bounds, the same length as `initial`, with every
#'   starting value inside them. Both are required: every optimizer this dispatches to takes a box,
#'   and the numerical Jacobian's step selection is bounds-aware.
#' @param sample_size the number of observations behind the moment conditions. Required, and only
#'   you know it: your function hands over averages, not data. The sandwich covariance divides by
#'   it, so the standard errors scale as `1 / sqrt(sample_size)`.
#' @param jacobian an optional analytic Jacobian of the moment conditions, called as
#'   `jacobian(parameters)` and returning a q by p numeric matrix -- one ROW per moment condition,
#'   one COLUMN per parameter. `NULL`, the default, uses the ported bounds-aware finite-difference
#'   Jacobian, which costs two extra `moment_conditions` calls per parameter per gradient.
#' @param penalty an optional penalty added to the GMM objective, called as `penalty(parameters)`
#'   and returning one number. Ridge-type regularization, and the only way to fit a model with more
#'   parameters than moment conditions (which is otherwise refused as under-identified). Return `0`
#'   for no penalty. Note the ported half-quadratic convention: with a penalty the objective is
#'   `0.5 * g'Wg + penalty`, so a penalty should carry its own `1/2`.
#' @param optimizer one of `"BFGS"` (default, matching `GeneralizedMethodOfMoments`'s own class
#'   default), `"NelderMead"`, `"Brent"`, `"Powell"`, `"DifferentialEvolution"`,
#'   `"MultilevelSingleLinkage"`.
#' @param strategy GMM estimation strategy: `"Iterative"` (default), `"OneStep"`, or `"TwoStep"`.
#'   `"OneStep"` is refused for an over-identified problem, matching the estimator.
#' @param max_gmm_iterations maximum number of GMM iterations; `0` (default) keeps the estimator's
#'   own default cap.
#' @return An object of class `corehydro_fit` with `method == "GMM"` -- the same object [fit_gmm()]
#'   returns, so `coef()`, `vcov()`, `print()` and `summary()` behave identically. Parameters are
#'   named `p1`, `p2`, ... since your moment conditions name nothing. Method of moments computes no
#'   likelihood surface, so `$log_likelihood`, `$aic` and `$bic` are `NA` and `confint()` errors, as
#'   they do for [fit_gmm()]. `$j_stat` is Hansen's J and `$j_stat_pval` its p-value, which is `NA`
#'   whenever the fit is just-identified (as many moment conditions as parameters): zero degrees of
#'   freedom leaves no over-identifying restriction to test, and `print()` says so rather than
#'   showing a figure. `$j_stat` itself is not a goodness-of-fit number you can read there either.
#'   The residual covariance it is scaled by is singular -- it has rank q - p, so it is exactly
#'   zero when the fit is just-identified -- and inverting it amplifies the optimizer's convergence
#'   tolerance rather than any property of your data. The result varies by many orders of magnitude
#'   and in sign between optimizers, and between this package and the C# library it ports, on fits
#'   whose parameters agree to ten significant figures. Sometimes it cannot be computed at all, and
#'   then it comes back `NA` rather than failing the fit. Over-identifying the model restores the
#'   p-value but not `$j_stat`, since the rank deficiency only shrinks from q to q - p.
#'   `$degree_of_freedom` and `$number_of_moment_conditions` carry q - p and q.
#'   [fit_diagnostics()] and [quantile_variance()] are not available for this fit -- both need the
#'   model a `fit_gmm()` fit carries.
#' @section Reproducing a run in Python:
#' There is no random number generator anywhere in this fit, so a repeated call returns the
#' identical numbers. Across languages the guarantee is the usual one for this surface: the
#' optimizer arithmetic all happens in the shared C++ core, but `g` and `s` are computed by your
#' own R code, and R and Python do not guarantee identical rounding for the same formula.
#' Arithmetic (`+ - * /`) is IEEE-deterministic and does reproduce; `log`, `exp` and friends come
#' from each platform's math library. R's `sum()` and `mean()` accumulate in extended precision
#' where Python's do not, so an explicit loop is the portable spelling.
#' @seealso [fit_gmm()] for the Bulletin 17C flood-frequency fit, [optim_minimize()] for a plain
#'   bounded optimization of your own objective, and [fit_mle()] for likelihood-based fitting.
#' @examples
#' x <- c(4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7)
#' # An explicit loop rather than mean(), so the same formula returns the same bits in Python:
#' # R's mean() and sum() accumulate in extended precision and Python's do not.
#' avg <- function(v) {
#'   total <- 0
#'   for (value in v) total <- total + value
#'   total / length(v)
#' }
#' moments <- function(p) {
#'   a <- x - p[1]
#'   b <- a * a - p[2]
#'   list(
#'     g = c(avg(a), avg(b)),
#'     s = matrix(c(avg(a * a), avg(a * b), avg(a * b), avg(b * b)), 2, 2)
#'   )
#' }
#' f <- fit_gmm_moments(moments,
#'   initial = c(5, 0.5), lower = c(0, 0.001), upper = c(10, 10),
#'   sample_size = length(x)
#' )
#' round(coef(f), 6) # the sample mean and the population variance
#' f$j_stat_pval # NA: just-identified, so there is nothing to test
#' @export
fit_gmm_moments <- function(
  moment_conditions,
  initial,
  lower = NULL,
  upper = NULL,
  sample_size,
  jacobian = NULL,
  penalty = NULL,
  optimizer = "BFGS",
  strategy = "Iterative",
  max_gmm_iterations = 0L
) {
  if (!is.function(moment_conditions)) {
    stop("`moment_conditions` must be a function taking (parameters) and returning a list with ",
         "elements `g` (the moment vector) and `s` (the weighting matrix)", call. = FALSE)
  }
  if (!is.null(jacobian) && !is.function(jacobian)) {
    stop("`jacobian` must be a function taking (parameters) and returning a numeric matrix",
         call. = FALSE)
  }
  if (!is.null(penalty) && !is.function(penalty)) {
    stop("`penalty` must be a function taking (parameters) and returning a single number",
         call. = FALSE)
  }
  if (!is.numeric(initial) || length(initial) == 0L || !all(is.finite(initial))) {
    stop("`initial` must be a non-empty numeric vector of finite values", call. = FALSE)
  }
  # Required despite the NULL default, exactly as optim_minimize()'s bounds are: the estimator has
  # no unbounded form (see fit_gmm_moments()'s `lower`/`upper` documentation).
  if (is.null(lower) || is.null(upper)) {
    stop("`fit_gmm_moments()` needs `lower` and `upper` bounds: the GMM optimizer takes a box and ",
         "the numerical Jacobian's step selection is bounds-aware", call. = FALSE)
  }
  if (!is.numeric(lower) || !is.numeric(upper) || !all(is.finite(lower)) || !all(is.finite(upper))) {
    stop("`lower` and `upper` must be numeric vectors of finite values", call. = FALSE)
  }
  if (length(lower) != length(initial) || length(upper) != length(initial)) {
    stop(sprintf(
      "`initial`, `lower` and `upper` must be the same length; they are %d, %d and %d",
      length(initial), length(lower), length(upper)
    ), call. = FALSE)
  }
  if (missing(sample_size) || !is.numeric(sample_size) || length(sample_size) != 1L ||
        !is.finite(sample_size) || sample_size < 1) {
    stop("`sample_size` must be a single positive whole number: the number of observations behind ",
         "your moment conditions, which the sandwich covariance divides by", call. = FALSE)
  }
  known_optimizers <- c(
    "NelderMead", "Brent", "BFGS", "Powell", "DifferentialEvolution", "MultilevelSingleLinkage"
  )
  optimizer <- as.character(optimizer)
  if (!optimizer %in% known_optimizers) {
    stop(sprintf(
      "unknown optimizer '%s'; expected one of %s", optimizer,
      paste(known_optimizers, collapse = ", ")
    ), call. = FALSE)
  }
  known_strategies <- c("OneStep", "TwoStep", "Iterative")
  strategy <- as.character(strategy)
  if (!strategy %in% known_strategies) {
    stop(sprintf(
      "unknown GMM estimation strategy '%s'; expected one of %s",
      strategy, paste(known_strategies, collapse = ", ")
    ), call. = FALSE)
  }

  opts <- list(
    initial = spec_array(as.double(initial)),
    lower = spec_array(as.double(lower)),
    upper = spec_array(as.double(upper)),
    sample_size = as.integer(sample_size),
    optimizer = optimizer,
    strategy = strategy
  )
  max_gmm_iterations <- as.integer(max_gmm_iterations)
  if (max_gmm_iterations > 0L) opts$max_gmm_iterations <- max_gmm_iterations

  res <- ch_callback_gmm_(to_spec_json(opts), moment_conditions, jacobian, penalty)
  new_fit_gmm_moments(gmm_unflatten(res))
}

# Internal: slice the flat callback result back into the field set new_fit_gmm_moments() reads --
# the same names ch_fit_run_() returns for the GMM target, so both feed the one shared
# gmm_fit_fields() builder in R/fit.R. The layout is documented in
# core/include/corehydro/numerics/support/callback/gmm.hpp, and corehydropy's own
# _gmm_unflatten() reads it identically.
gmm_unflatten <- function(res) {
  values <- as.double(unlist(res$values))
  nms <- as.character(unlist(res$names))
  p <- as.integer(unlist(res$dims))[[1]]

  by_name <- function(name) as.double(values[[match(name, nms)]])
  block <- function(prefix, count) {
    as.double(values[match(sprintf("%s[%d]", prefix, seq_len(count) - 1L), nms)])
  }
  # A p x p matrix read back by the label on every entry rather than by slicing a range: the
  # bindings and the fixture runners all address this result by name, and a matrix is the one place
  # an off-by-one slice would still look plausible.
  square <- function(prefix) {
    labels <- outer(seq_len(p) - 1L, seq_len(p) - 1L, function(i, j) sprintf("%s[%d,%d]", prefix, i, j))
    matrix(values[match(as.vector(labels), nms)], nrow = p, ncol = p)
  }

  list(
    method = "GMM",
    parameters = block("parameter", p),
    parameter_names = paste0("p", seq_len(p)),
    standard_errors = block("standard_error", p),
    covariance = square("covariance"),
    correlation = square("correlation"),
    j_stat = by_name("j_stat"),
    j_stat_pval = by_name("j_stat_pval"),
    degree_of_freedom = as.integer(by_name("degree_of_freedom")),
    gmm_iterations = as.integer(by_name("gmm_iterations")),
    converged_within_tolerance = by_name("converged_within_tolerance") == 1,
    optimizer_fallback_count = as.integer(by_name("optimizer_fallback_count")),
    number_of_moment_conditions = as.integer(by_name("number_of_moment_conditions")),
    nobs = as.integer(by_name("sample_size")),
    converged = identical(res$status, "Success"),
    status = res$status
  )
}

# Internal: the function check every bootstrap delegate shares, naming the argument AND its
# signature -- a wrong argument order is the likeliest mistake on this surface and nothing
# downstream can detect it. Kept in step with corehydropy's _check_bootstrap_fn.
bootstrap_check_fn <- function(f, name, signature) {
  if (!is.function(f)) {
    stop(sprintf("`%s` must be a function taking %s", name, signature), call. = FALSE)
  }
}

# Internal: slice the flat callback result back by name. The layout is documented in
# core/include/corehydro/numerics/support/callback/bootstrap.hpp, and corehydropy's own
# _bootstrap_unflatten() reads it identically.
bootstrap_unflatten <- function(res, ci_method) {
  values <- as.double(unlist(res$values))
  nms <- as.character(unlist(res$names))
  dims <- as.integer(unlist(res$dims))
  n_stats <- dims[[1]]
  n_params <- dims[[2]]

  by_name <- function(name) as.double(values[[match(name, nms)]])
  block <- function(prefix, count) {
    as.double(values[match(sprintf("%s[%d]", prefix, seq_len(count) - 1L), nms)])
  }

  list(
    estimate = block("statistic", n_stats),
    lower = block("statistic_lower", n_stats),
    upper = block("statistic_upper", n_stats),
    standard_error = block("statistic_se", n_stats),
    mean = block("statistic_mean", n_stats),
    valid_count = as.integer(block("statistic_valid", n_stats)),
    # The parameter block is always a percentile interval, whatever `ci_method` asks for: the
    # ported GetConfidenceIntervals applies the requested method to the STATISTICS and takes plain
    # percentiles of the fitted parameters.
    parameter_estimate = block("parameter", n_params),
    parameter_lower = block("parameter_lower", n_params),
    parameter_upper = block("parameter_upper", n_params),
    replicates = as.integer(by_name("replicates")),
    failed_replicates = as.integer(by_name("failed_replicates")),
    alpha = by_name("alpha"),
    ci_method = ci_method
  )
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
