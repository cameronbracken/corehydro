# Public wrapper over the MCMC glue in src/mcmc.cpp (ch_mcmc_run_).

#' Sample the posterior of a distribution's parameters by MCMC
#'
#' Fits `distribution` to `data` with uniform priors spanning the family's
#' parameter constraints (exactly the C# `GetParameterConstraints` bounds) and
#' returns the full sampler output: per-chain draws, acceptance rates, the MAP
#' estimate, posterior summaries, and Gelman-Rubin / effective-sample-size
#' diagnostics. Seven of the eight ported samplers are available (Gibbs needs
#' a model-specific conditional proposal and is not exposed here).
#'
#' Runs the ported serial chain driver with the C# default seed, so a seeded
#' run reproduces the C# sampler stream bit-for-bit (and matches `corehydropy`
#' exactly). The priors are always the constraint-based uniforms; custom
#' priors are not exposed -- use [univariate_analysis()] and friends for the
#' full Bayesian workflow.
#'
#' @param data numeric vector of observations.
#' @param distribution the distribution family name; see
#'   [distribution_names()].
#' @param sampler one of `"RWMH"` (the default), `"ARWMH"`, `"DEMCz"`,
#'   `"DEMCzs"`, `"HMC"`, `"NUTS"`, or `"SNIS"`.
#' @param iterations iterations per chain (sampler default if `NULL`).
#' @param warmup warm-up iterations discarded from each chain (sampler
#'   default if `NULL`).
#' @param chains number of chains (sampler default if `NULL`).
#' @param thinning thinning interval (sampler default if `NULL`).
#' @param output_length total number of retained draws across all chains
#'   (sampler default, 10,000, if `NULL`). The ported sampler collects
#'   `ceiling(output_length / chains)` draws per chain after the iteration
#'   loop. The ported floor is 100 and it is refused below that.
#' @param seed PRNG seed; `12345` is the C# default.
#' @param initialize chain initialization: `"MAP"` (from the posterior-mode
#'   estimate, the C# default) or `"Randomize"` (draws from the priors).
#' @return A list with `parameters` (parameter names), `chains` (a list of
#'   one draws-by-parameters matrix per chain), `acceptance_rates`, `map`
#'   (parameter values at the posterior mode), `map_fitness`,
#'   `posterior_mean`, `posterior_sd`, `posterior_median`,
#'   `posterior_lower_ci`, `posterior_upper_ci`, `rhat`, and `ess` (all
#'   per-parameter vectors).
#' @export
#' @examples
#' \donttest{
#' x <- dist_random(distribution("Normal", c(100, 15)), 200, seed = 42)
#' fit <- mcmc_sample(x, "Normal", sampler = "DEMCzs", iterations = 2000)
#' fit$posterior_mean
#' fit$rhat
#' }
mcmc_sample <- function(
  data,
  distribution = "Normal",
  sampler = "RWMH",
  iterations = NULL,
  warmup = NULL,
  chains = NULL,
  thinning = NULL,
  output_length = NULL,
  seed = 12345,
  initialize = "MAP"
) {
  # The same check_choice() mcmc_posterior() uses, over the same list without "Gibbs" (which needs
  # a model-specific conditional proposal this verb has no way to take). See R/fit.R on why not
  # match.arg.
  sampler <- check_choice(sampler, setdiff(mcmc_sampler_names, "Gibbs"), "sampler")
  initialize <- check_choice(initialize, c("MAP", "Randomize"), "chain initialization")
  settings <- list(prng_seed = as.integer(seed), initialize = initialize)
  if (!is.null(iterations)) {
    settings$iterations <- as.integer(iterations)
    # The sampler requires warmup <= iterations / 2; when only iterations is
    # given, follow the analysis wrappers' auto rule.
    if (is.null(warmup)) warmup <- max(50L, as.integer(iterations) %/% 2L)
  }
  if (!is.null(warmup)) settings$warmup_iterations <- as.integer(warmup)
  if (!is.null(chains)) settings$number_of_chains <- as.integer(chains)
  if (!is.null(thinning)) settings$thinning_interval <- as.integer(thinning)
  if (!is.null(output_length)) {
    # Validated here and worded identically in mcmc_posterior() and corehydropy, so the floor is
    # refused by name rather than by the ported "The output length must be at least 100."
    if (!is.numeric(output_length) || length(output_length) != 1L ||
          !is.finite(output_length) || output_length < 100) {
      stop("`output_length` must be a single whole number of at least 100, which is the ported ",
           "sampler's own floor", call. = FALSE)
    }
    settings$output_length <- as.integer(output_length)
  }
  if (identical(sampler, "RWMH")) {
    # The RWMH constructor takes a proposal covariance; MAP initialization
    # overwrites it before first use (the C# test convention).
    settings$proposal_sigma <- "identity"
  }

  raw <- ch_mcmc_run_(sampler, "uniform_constraints", distribution, as.double(data), settings)

  list(
    parameters = unlist(ch_dist_parameter_names_(distribution)$short),
    chains = raw$chains,
    acceptance_rates = as.double(unlist(raw$acceptance_rates)),
    map = as.double(unlist(raw$map_values)),
    map_fitness = raw$map_fitness,
    posterior_mean = as.double(unlist(raw$posterior_mean)),
    posterior_sd = as.double(unlist(raw$posterior_sd)),
    posterior_median = as.double(unlist(raw$posterior_median)),
    posterior_lower_ci = as.double(unlist(raw$posterior_lower_ci)),
    posterior_upper_ci = as.double(unlist(raw$posterior_upper_ci)),
    rhat = as.double(unlist(raw$rhat)),
    ess = as.double(unlist(raw$ess))
  )
}
