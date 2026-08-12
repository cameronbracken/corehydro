# The user-facing fit surface: point-estimate fits (fit_mle / fit_map) over the shared C++
# `run_fit` (core/include/corehydro/estimation/support/fit_runner.hpp). A `corehydro_fit` is a
# plain classed list, like `corehydro_model`: nothing here holds a C++ object, so a fit can be
# printed, saved, and passed between R sessions unchanged. Bayesian and GMM fits (fit_bayesian(),
# fit_gmm()) land in a later task on top of the same `ch_fit_run_()` entry point.

# Internal: assemble the construct the C++ fit runner parses, from the dual first argument every
# fit verb takes (a corehydro_model, or a numeric vector plus a distribution name). Reuses
# analysis_input() (R/analysis.R) so the vector path and the model path build the identical model
# JSON the analyses already rely on -- the two cannot drift apart.
#
# `settings` is serialized with to_spec_json() and spliced after the model entry, rather than
# round-tripping the whole construct back through to_spec_json() as one R list: the model's JSON
# is already a string (from analysis_input()), and to_spec_json() has no way to embed a
# pre-serialized string verbatim inside a larger object it is asked to build from scratch.
#
# An empty `settings` needs its own branch: to_spec_json(list()) returns "[]" (an unnamed empty
# list has no names() to route it through the named-object branch of spec.R's serializer), and
# even a named empty object ("{}") cannot simply be comma-spliced after the model entry without
# leaving a dangling `,}` that corehydro/models/json_lite.hpp's parser rejects (no trailing-comma
# tolerance).
fit_input <- function(model, distribution, settings) {
  # The corehydro_data branch is left to analysis_input()'s own, more specific message ("pass an
  # analysis_data() frame through a model, ..."); this guard only fires for a plain vector.
  if (!inherits(model, "corehydro_model") && !inherits(model, "corehydro_data") &&
    is.null(distribution)) {
    stop("give a `distribution` name when `model` is a plain numeric vector, e.g. ",
      "fit_mle(peaks, \"Normal\")",
      call. = FALSE
    )
  }
  base_spec <- if (inherits(model, "corehydro_model")) {
    model$spec
  } else {
    list(family = as.character(distribution), dataset = "data")
  }
  input <- analysis_input(model, function() base_spec)

  settings_json <- to_spec_json(settings)
  if (identical(settings_json, "[]")) settings_json <- "{}"
  json <- if (identical(settings_json, "{}")) {
    paste0("{\"model\":", input$json, "}")
  } else {
    paste0("{\"model\":", input$json, ",", substring(settings_json, 2L))
  }
  list(json = json, dataset = input$dataset, spec = base_spec)
}

# Internal: confint.corehydro_fit()'s Bayesian rebuild path needs the IDENTICAL construct the
# fit was originally built from, with only `credible_interval_width` overridden -- every other
# setting (sampler, seed, iterations, chains, thinning, sampler-specific knobs) stays
# byte-identical, so `sampler_->sample()` reproduces the same seeded chain and only the post-hoc
# credible-interval quantile computation (`results_.emplace(*sampler_, 1 - width)`, C#
# BayesianAnalysis::estimate()) changes -- see apply_bayesian_settings() in fit_runner.hpp.
# Reconstructing the settings list from scratch (the way fit_input() builds a fresh construct)
# would silently drop any sampler knob (jump/noise/scale/beta/...) the original fit_bayesian()
# call passed through `...`, since a corehydro_fit does not carry those back out; splicing into
# the already-built JSON avoids that. Every construct this package builds (fit_input(), above) is
# a single well-formed top-level JSON object with no trailing content, so overriding is just
# replacing the final closing brace.
#
# fit_bayesian() now always writes its own `credible_interval_width` (the `credible_level`
# argument), so the key IS already present here. json_lite.hpp's JsonValue::at() returns the
# FIRST match by insertion order, which would resurrect the original width and silently ignore
# the override, so the existing entry is dropped before the new one is appended. The pattern is
# exact rather than approximate: this package writes the key in only one place, always as
# `,"credible_interval_width":<number>` from spec_number(), and a JSON number never contains a
# comma or a brace.
inject_credible_interval_width <- function(construct_json, level) {
  construct_json <- sub(",\"credible_interval_width\":[^,}]*", "", construct_json)
  paste0(
    substr(construct_json, 1L, nchar(construct_json) - 1L),
    ",\"credible_interval_width\":", spec_number(level), "}"
  )
}

# Internal: `result$covariance`/`result$correlation` are already proper n x n R matrices (built
# column-by-column in corehydror/src/estimation.cpp's `square_or_empty`), so only the dimnames
# need applying here. Do NOT round-trip a matrix through `matrix(x, byrow = TRUE)`: that first
# flattens it column-major (`as.vector()`) and refills row-major, i.e. transposes it -- invisible
# only because these particular matrices are symmetric. Shared by new_fit() (MLE/MAP) and
# new_fit_gmm() -- the two targets that populate a covariance stack.
name_square <- function(m, parameter_names) {
  if (!length(m)) {
    return(NULL)
  }
  dimnames(m) <- list(parameter_names, parameter_names)
  m
}

# Internal: the field set every corehydro_fit carries regardless of target -- named parameters,
# the common goodness-of-fit scalars, and the fitted model/spec/dataset bookkeeping. `base_spec`/
# `dataset` are what fit_input() resolved the construct from -- carried forward so confint() can
# lazily re-run the identical fit (see ch_estimation_bic_'s precedent), and so `$model` can be
# assembled by overriding `base_spec`'s `parameter_values` with the fitted values rather than
# re-parsing `model_spec` (the fitted spec JSON string ch_fit_run_() also returns) back into R
# structures -- this package deliberately carries no JSON parser, only the serializer in
# R/spec.R. `construct_json` is the UNFITTED construct fit_input() built (fi$json): carried so
# fit_diagnostics() and quantile_variance() can rerun the identical construct through the C++
# diagnostics/quantile-variance entry points, agreeing with the fit they were computed from.
new_fit_base <- function(result, base_spec, dataset, construct_json) {
  parameters <- result$parameters
  names(parameters) <- result$parameter_names

  fitted_spec <- base_spec
  fitted_spec$parameter_values <- spec_array(as.double(parameters))
  fitted_model <- structure(
    list(spec = fitted_spec, dataset = dataset),
    class = "corehydro_model"
  )

  list(
    method = result$method,
    parameters = parameters,
    log_likelihood = result$log_likelihood,
    prior_log_likelihood = result$prior_log_likelihood,
    aic = result$aic,
    bic = result$bic,
    nobs = result$nobs,
    converged = result$converged,
    status = result$status,
    model = fitted_model,
    spec = base_spec,
    dataset = dataset,
    construct_json = construct_json
  )
}

# Internal: build a corehydro_fit from a ch_fit_run_() result for the MaximumLikelihood/
# MaximumAPosteriori targets -- adds the Hessian-based covariance stack, function-evaluation
# count, and the profile-likelihood grid/CI bookkeeping on top of new_fit_base()'s common fields.
new_fit <- function(result, base_spec, dataset, optimizer, level, construct_json = NULL) {
  n <- length(result$parameters)

  covariance <- name_square(result$covariance, result$parameter_names)
  correlation <- name_square(result$correlation, result$parameter_names)
  standard_errors <- if (length(result$standard_errors)) {
    se <- result$standard_errors
    names(se) <- result$parameter_names
    se
  } else {
    NULL
  }

  has_profile <- length(result$profile_lower) > 0

  # The plottable profile-likelihood grid: `result$profile_grid` is a genuinely flat vector (not
  # a pre-built matrix, unlike covariance/correlation above), n_params * bins * 2, row-major
  # [parameter][bin][value, profile log-likelihood] -- see fit_runner.hpp's FitResult doc. One
  # `bins x 2` matrix per parameter, byrow = TRUE is correct here because the source is flat.
  profile <- if (has_profile) {
    bins <- result$profile_bins
    grid <- result$profile_grid
    per_parameter <- lapply(seq_len(n), function(p) {
      start <- (p - 1L) * bins * 2L
      sub <- grid[(start + 1L):(start + bins * 2L)]
      m <- matrix(sub, nrow = bins, ncol = 2L, byrow = TRUE)
      colnames(m) <- c("value", "log_likelihood")
      m
    })
    names(per_parameter) <- result$parameter_names
    per_parameter
  } else {
    NULL
  }

  # The profile-likelihood confidence-interval bookkeeping confint.corehydro_fit() reuses; kept
  # separate from `profile` above (the public plottable-grid element the design spec names).
  profile_intervals <- if (has_profile) {
    lower <- result$profile_lower
    names(lower) <- result$parameter_names
    upper <- result$profile_upper
    names(upper) <- result$parameter_names
    list(
      lower = lower,
      upper = upper,
      bins = result$profile_bins,
      level = level
    )
  } else {
    NULL
  }

  structure(
    c(
      new_fit_base(result, base_spec, dataset, construct_json),
      list(
        covariance = covariance,
        standard_errors = standard_errors,
        correlation = correlation,
        function_evaluations = result$function_evaluations,
        profile = profile,
        profile_intervals = profile_intervals,
        optimizer = optimizer
      )
    ),
    class = "corehydro_fit"
  )
}

# Internal: build a corehydro_fit from a ch_fit_run_() result for the BayesianAnalysis target.
# `d <- res$chain_dims` / the aperm() below permutes the runner's CHAIN-major flatten (chain,
# iteration, parameter) into [iteration, chain, parameter], the axis order
# posterior::as_draws_array() expects (see fit_bayesian()'s header note for the full rationale).
# `credible_level` records the width `$summary`'s lower/upper columns were computed at, which is
# fit_bayesian()'s `credible_level` argument (default 0.9, BayesianAnalysis's own class default);
# confint.corehydro_fit() compares its requested `level` against this field to decide whether
# `$summary` already answers the call or a rebuild is needed.
new_fit_bayesian <- function(result, base_spec, dataset, construct_json, warmup,
                             credible_level = 0.9) {
  d <- result$chain_dims
  draws <- aperm(array(result$draws, dim = c(d[3], d[2], d[1])), c(2L, 3L, 1L))
  dimnames(draws) <- list(NULL, paste0("chain", seq_len(d[1])), result$parameter_names)

  # The thinned posterior (posterior_rows x n_params) the analyses consume. ch_fit_run_() already
  # returns it as a proper R matrix (built element-by-element in matrix_or_empty, the same
  # row-major reshape `covariance` gets), so only the column names are applied here -- see
  # name_square()'s note on why this must NOT be re-flattened through matrix(byrow = TRUE).
  posterior <- result$posterior
  if (length(posterior)) colnames(posterior) <- result$parameter_names

  named <- function(x) {
    if (!length(x)) {
      return(NULL)
    }
    names(x) <- result$parameter_names
    x
  }

  summary <- data.frame(
    mean = result$summary_mean,
    median = result$summary_median,
    sd = result$summary_sd,
    lower = result$summary_lower,
    upper = result$summary_upper,
    rhat = result$rhat,
    ess = result$ess,
    row.names = result$parameter_names
  )

  structure(
    c(
      new_fit_base(result, base_spec, dataset, construct_json),
      list(
        draws = draws,
        posterior = posterior,
        posterior_rows = result$posterior_rows,
        map = named(result$map),
        posterior_mean = named(result$posterior_mean),
        mean_log_likelihood = result$mean_log_likelihood,
        summary = summary,
        acceptance_rates = result$acceptance_rates,
        warmup = warmup,
        credible_level = credible_level,
        dic = result$dic,
        waic = result$waic,
        waic_pd = result$waic_pd,
        looic = result$looic,
        looic_se = result$looic_se,
        loo_pd = result$loo_pd
      )
    ),
    class = "corehydro_fit"
  )
}

# Internal: build a corehydro_fit from a ch_fit_run_() result for the GMM target -- the
# covariance stack (same shape as new_fit()'s, GMM's own sandwich covariance rather than a
# Hessian) plus the GMM-specific bookkeeping (J-statistic, iteration/convergence counters).
new_fit_gmm <- function(result, base_spec, dataset, construct_json) {
  base <- new_fit_base(result, base_spec, dataset, construct_json)
  # GMM is method-of-moments: run_fit() leaves log_likelihood/aic/bic at their structural NaN
  # defaults because there is no likelihood surface to report them from. Report that as NA, the R
  # spelling of "not available", so it agrees with logLik()/AIC()/BIC() on the same fit (which
  # have always returned NA), with the documentation, and with the Python surface (None).
  base$log_likelihood <- NA_real_
  base$aic <- NA_real_
  base$bic <- NA_real_

  covariance <- name_square(result$covariance, result$parameter_names)
  correlation <- name_square(result$correlation, result$parameter_names)
  standard_errors <- if (length(result$standard_errors)) {
    se <- result$standard_errors
    names(se) <- result$parameter_names
    se
  } else {
    NULL
  }

  structure(
    c(
      base,
      list(
        covariance = covariance,
        standard_errors = standard_errors,
        correlation = correlation,
        j_stat = result$j_stat,
        j_stat_pval = if (is.nan(result$j_stat_pval)) NA_real_ else result$j_stat_pval,
        gmm_iterations = result$gmm_iterations,
        converged_within_tolerance = result$converged_within_tolerance,
        optimizer_fallback_count = result$optimizer_fallback_count
      )
    ),
    class = "corehydro_fit"
  )
}

# Internal: shared body of fit_mle()/fit_map(), parameterized on the estimator target. Not
# exported: fit_mle() and fit_map() stay two documented verbs (the whole point of the shape
# decision -- see the roxygen on each), this is just their DRY plumbing. `alpha` is not part of
# either verb's public signature; it exists so confint() can ask for a profile at an arbitrary
# level without a public `alpha` argument nobody else needs.
fit_optimized <- function(target, model, distribution, optimizer, hessian, profile, profile_bins,
                          alpha = 0.1) {
  known_optimizers <- c(
    "NelderMead", "Brent", "BFGS", "Powell", "DifferentialEvolution", "MultilevelSingleLinkage"
  )
  optimizer <- as.character(optimizer)
  if (!optimizer %in% known_optimizers) {
    stop(sprintf(
      "unknown optimizer '%s'; expected one of %s", optimizer, paste(known_optimizers, collapse = ", ")
    ), call. = FALSE)
  }
  profile <- isTRUE(profile)
  # `profile_bins` reaches profile_likelihood(bins) unguarded in the core, where a non-positive
  # count is a silently empty profile rather than an error, so it is validated here.
  profile_bins <- as.integer(profile_bins)
  if (is.na(profile_bins) || profile_bins < 1L) {
    stop("`profile_bins` must be a positive integer; got ", format(profile_bins), call. = FALSE)
  }
  settings <- list(
    optimizer = optimizer, hessian = isTRUE(hessian), profile = profile,
    profile_bins = profile_bins
  )
  if (profile) settings$alpha <- as.double(alpha)

  fi <- fit_input(model, distribution, settings)
  result <- ch_fit_run_(target, fi$json, fi$dataset)
  new_fit(result, fi$spec, fi$dataset, optimizer, level = 1 - alpha, construct_json = fi$json)
}

#' Maximum likelihood fit
#'
#' Fit a model by maximum likelihood and return a fit object carrying the parameter estimates,
#' the Hessian-based covariance, and optimizer bookkeeping. Wraps the shared C++
#' `MaximumLikelihood` ported from USACE-RMC RMC.BestFit.
#'
#' @param model a [model_univariate()] (or any `model_*()`) object, or a plain numeric vector of
#'   observations together with `distribution`. A model can bring censored observations (see
#'   [analysis_data()]), nonstationary trends (see [trend()]), and parameter bounds or priors
#'   (see [model_parameter()]).
#' @param distribution distribution family name, required only when `model` is a numeric vector.
#' @param optimizer one of `"NelderMead"` (default), `"Brent"`, `"BFGS"`, `"Powell"`,
#'   `"DifferentialEvolution"`, `"MultilevelSingleLinkage"`.
#' @param hessian logical; compute the covariance, standard errors and correlation. `TRUE` by
#'   default. A model with fewer than two parameters reports `NaN` for all three.
#' @param profile logical; also compute the profile likelihood and profile confidence intervals.
#'   `FALSE` by default because each costs `profile_bins * length(parameters)` likelihood
#'   evaluations.
#' @param profile_bins number of bins in each parameter's profile.
#' @return An object of class `corehydro_fit`. When `profile = TRUE`, it also carries `$profile`,
#'   a named list (one entry per parameter, named to match `coef()`) of `profile_bins`-by-2
#'   matrices with columns `value` and `log_likelihood`, the profile-likelihood grid `confint()`'s
#'   intervals are drawn from -- plot a parameter's curve with e.g.
#'   `plot(f$profile[[1]], type = "l")`. Absent (`NULL`) when the fit was built with the default
#'   `profile = FALSE`. See [fit_bayesian()] for the Bayesian surface.
#' @seealso [fit_map()], [fit_bayesian()], [fit_gmm()], [fit_diagnostics()].
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' f <- fit_mle(model_univariate("LogPearsonTypeIII", peaks))
#' coef(f)
#' AIC(f)
fit_mle <- function(model, distribution = NULL, optimizer = "NelderMead", hessian = TRUE,
                    profile = FALSE, profile_bins = 100) {
  fit_optimized("MaximumLikelihood", model, distribution, optimizer, hessian, profile, profile_bins)
}

#' Maximum a posteriori fit
#'
#' Fit a model by maximum a posteriori (the mode of the posterior formed from the model's own
#' priors) and return a fit object carrying the parameter estimates, the Hessian-based covariance,
#' and optimizer bookkeeping. Wraps the shared C++ `MaximumAPosteriori` ported from USACE-RMC
#' RMC.BestFit.
#'
#' @inheritParams fit_mle
#' @return An object of class `corehydro_fit`. When `profile = TRUE`, it also carries `$profile`,
#'   a named list (one entry per parameter, named to match `coef()`) of `profile_bins`-by-2
#'   matrices with columns `value` and `log_likelihood`, the profile-likelihood grid `confint()`'s
#'   intervals are drawn from -- plot a parameter's curve with e.g.
#'   `plot(f$profile[[1]], type = "l")`. Absent (`NULL`) when the fit was built with the default
#'   `profile = FALSE`. See [fit_bayesian()] for the Bayesian surface.
#' @seealso [fit_mle()], [fit_bayesian()], [fit_gmm()], [fit_diagnostics()].
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' f <- fit_map(model_univariate("LogPearsonTypeIII", peaks))
#' coef(f)
#' AIC(f)
fit_map <- function(model, distribution = NULL, optimizer = "NelderMead", hessian = TRUE,
                    profile = FALSE, profile_bins = 100) {
  fit_optimized("MaximumAPosteriori", model, distribution, optimizer, hessian, profile, profile_bins)
}

# Sampler-specific knobs, by sampler. A knob the chosen sampler ignores is an error rather than a
# silent no-op, because a silently dropped tuning argument looks like the sampler is broken.
# Confirmed against BayesianAnalysis::set_up_sampler (core/include/corehydro/estimation/
# bayesian_analysis.hpp:456-492) and each sampler's own knob fields (numerics/sampling/mcmc/
# {demcz,demczs,arwmh,nuts}.hpp): NUTS's ctor takes a fixed step size (0.1, no BestFit knob for
# it) ahead of `max_tree_depth`, so NUTS does NOT read a `scale` knob despite `scale` being a
# knob name the struct also carries for ARWMH -- unlike ARWMH, `set_up_sampler`'s NUTS branch
# never reads `scale_`.
sampler_knobs <- list(
  DEMCz  = c("jump", "jump_threshold", "noise"),
  DEMCzs = c("jump", "jump_threshold", "snooker_threshold", "noise"),
  ARWMH  = c("scale", "beta"),
  NUTS   = c("max_tree_depth")
)

# The MCMC settings ranges BayesianAnalysis::validate() enforces (core/include/corehydro/
# estimation/bayesian_analysis.hpp, the "Errors" block). A setting outside its range makes
# estimate() throw "Bayesian Analysis is not valid. Please check the configuration before running
# the analysis." with the messages that name the offending setting discarded, so fit_bayesian()
# checks the ranges itself and reports which value is out of range. Kept as a table so the two
# languages read the same list.
bayes_ranges <- list(
  chains = c(4, 20),
  iterations = c(100, 1000000),
  warmup = c(50, 100000),
  output_length = c(100, 1000000)
)

# Internal: one range check, in the style of the optimizer/sampler/knob validation beside it.
check_bayes_range <- function(name, value) {
  range <- bayes_ranges[[name]]
  if (is.na(value) || value < range[1] || value > range[2]) {
    stop(sprintf(
      "`%s` must be between %s and %s; got %s",
      name, format(range[1], scientific = FALSE), format(range[2], scientific = FALSE),
      format(value, scientific = FALSE)
    ), call. = FALSE)
  }
  invisible(value)
}

#' Bayesian MCMC fit
#'
#' Fit a model with a Bayesian MCMC analysis and return a fit object carrying the raw chains, the
#' posterior summary (mean/median/sd/credible interval, R-hat, effective sample size), and the
#' usual Bayesian goodness-of-fit scalars. Wraps the shared C++ `BayesianAnalysis` ported from
#' USACE-RMC RMC.BestFit -- the same estimator [univariate_analysis()] and
#' [estimation_diagnostics()] build on.
#'
#' @inheritParams fit_mle
#' @param sampler MCMC sampler: `"DEMCz"` (default), `"DEMCzs"`, `"ARWMH"`, or `"NUTS"`.
#'   `BayesianAnalysis` can only construct these four; RWMH, HMC and SNIS are real MCMC samplers
#'   but need [mcmc_sample()] instead.
#' @param chains number of parallel Markov chains.
#' @param iterations number of post-warmup MCMC iterations, per chain.
#' @param warmup number of warmup (burn-in) iterations. Defaults to `max(50, iterations %/% 2)`
#'   when omitted: `BayesianAnalysis`'s own class default (1500) would otherwise silently trip its
#'   sampler's `warmup <= iterations / 2` guard for any `iterations` below about 3000.
#' @param output_length number of posterior draws retained (thinned down from the raw chains) for
#'   the summary and any downstream uncertainty quantification.
#' @param thinning_interval MCMC thinning interval; `-1` (default) keeps the sampler's own default.
#' @param seed PRNG seed for the sampler (fixed for reproducibility -- a seeded call returns
#'   identical draws in R and Python).
#' @param point_estimator which posterior summary `$parameters` reports: `"PosteriorMean"` or
#'   `"PosteriorMode"` (the MAP). `NULL` (default) leaves `BayesianAnalysis`'s own class default,
#'   which is `"PosteriorMean"` -- so by default `$parameters` is bit-identical to
#'   `$summary$mean`. Pass `"PosteriorMode"` to report the MAP point instead.
#' @param credible_level width of the posterior credible interval `$summary`'s `lower`/`upper`
#'   columns report, between 0 and 1. Defaults to `0.9`, `BayesianAnalysis`'s own class default.
#'   `confint()` reuses these bounds when its `level` matches and otherwise re-runs the identical
#'   seeded chain, so setting it here is how a bare `confint(f)` avoids that rebuild.
#' @param ... sampler-specific tuning knobs. Passing a knob the chosen `sampler` does not use is an
#'   error: `"DEMCz"` accepts `jump`, `jump_threshold`, `noise`; `"DEMCzs"` additionally accepts
#'   `snooker_threshold`; `"ARWMH"` accepts `scale`, `beta`; `"NUTS"` accepts `max_tree_depth`.
#' @section Accepted ranges: `BayesianAnalysis` rejects a configuration outside these bounds, so
#'   `fit_bayesian()` checks them up front and names the offending value: `chains` between 4 and
#'   20, `iterations` between 100 and 1,000,000, `warmup` between 50 and 100,000, `output_length`
#'   between 100 and 1,000,000, `seed` not negative, and `credible_level` strictly between 0 and 1.
#' @return An object of class `corehydro_fit` with `method == "BayesianAnalysis"`. `$draws` is a
#'   3-D array `[iteration, chain, parameter]`, the axis order `posterior::as_draws_array()`
#'   expects, so it can be handed to the posterior or coda packages with no reshaping.
#'   `$posterior` is the thinned draw matrix the analyses consume, `$posterior_rows` by
#'   `length(coef(f))`, with the parameter names on its columns. `$summary`
#'   is a data frame (one row per parameter, named to match `coef()`) with columns `mean`,
#'   `median`, `sd`, `lower`, `upper`, `rhat`, `ess`. `$map` and `$posterior_mean` are the two
#'   posterior point estimates, named to match `coef()`. `$mean_log_likelihood` has one entry per
#'   iteration, the chain-averaged log-likelihood trace. `$acceptance_rates` has one entry per
#'   chain. `$warmup` records the warmup actually used and `$credible_level` the width `$summary`'s
#'   `lower`/`upper` were computed at. `$dic`, `$waic`, `$looic` are the usual Bayesian
#'   goodness-of-fit scalars, `$looic_se` the standard error of `$looic`, and `$waic_pd`/`$loo_pd`
#'   their effective parameter counts. See [fit_diagnostics()] for leverage/influence diagnostics
#'   off a Bayesian fit.
#' @seealso [fit_mle()], [fit_map()], [fit_gmm()], [fit_diagnostics()], [univariate_analysis()].
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' f <- fit_bayesian(model_univariate("Normal", peaks),
#'   sampler = "DEMCz", iterations = 200, output_length = 500, seed = 12345
#' )
#' f$summary
fit_bayesian <- function(model, distribution = NULL, sampler = "DEMCz", chains = 4L,
                         iterations = 3000L, warmup = NULL, output_length = 10000L,
                         thinning_interval = -1L, seed = 12345L, point_estimator = NULL,
                         credible_level = 0.9, ...) {
  known_samplers <- c("DEMCz", "DEMCzs", "ARWMH", "NUTS")
  sampler <- as.character(sampler)
  if (!sampler %in% known_samplers) {
    stop(sprintf(
      paste0(
        "BayesianAnalysis cannot construct sampler '%s'; it supports DEMCz, DEMCzs, ARWMH ",
        "and NUTS. For RWMH, HMC or SNIS use mcmc_sample()."
      ),
      sampler
    ), call. = FALSE)
  }

  known_point_estimators <- c("PosteriorMean", "PosteriorMode")
  if (!is.null(point_estimator)) {
    point_estimator <- as.character(point_estimator)
    if (!point_estimator %in% known_point_estimators) {
      stop(sprintf(
        "unknown point estimator '%s'; expected one of %s",
        point_estimator, paste(known_point_estimators, collapse = ", ")
      ), call. = FALSE)
    }
  }

  knobs <- list(...)
  allowed <- sampler_knobs[[sampler]]
  bad <- setdiff(names(knobs), allowed)
  if (length(bad)) {
    stop(sprintf(
      "%s does not use the knob%s %s; %s accepts: %s",
      sampler, if (length(bad) > 1L) "s" else "", paste(bad, collapse = ", "),
      sampler, if (length(allowed)) paste(allowed, collapse = ", ") else "(none)"
    ), call. = FALSE)
  }

  iterations <- as.integer(iterations)
  warmup <- if (is.null(warmup)) max(50L, iterations %/% 2L) else as.integer(warmup)
  chains <- as.integer(chains)
  output_length <- as.integer(output_length)
  seed <- as.integer(seed)

  # The ranges BayesianAnalysis::validate() enforces. Without these checks an out-of-range
  # setting reaches estimate(), which throws with its own diagnosis discarded.
  check_bayes_range("chains", chains)
  check_bayes_range("iterations", iterations)
  check_bayes_range("warmup", warmup)
  check_bayes_range("output_length", output_length)
  if (is.na(seed) || seed < 0L) {
    stop("`seed` cannot be negative; got ", format(seed), call. = FALSE)
  }
  credible_level <- as.double(credible_level)
  if (is.na(credible_level) || credible_level <= 0 || credible_level >= 1) {
    stop("`credible_level` must be greater than 0 and less than 1; got ",
      format(credible_level),
      call. = FALSE
    )
  }

  settings <- list(
    sampler = sampler,
    seed = seed,
    iterations = iterations,
    warmup_iterations = warmup,
    number_of_chains = chains,
    output_length = output_length,
    # Always written, never derived: at the 0.9 default this is BayesianAnalysis's own class
    # default, so a fit built without touching the argument is numerically unchanged, and
    # confint()'s reuse shortcut can compare against a level the construct actually states.
    credible_interval_width = credible_level
  )
  thinning_interval <- as.integer(thinning_interval)
  if (thinning_interval > 0L) settings$thinning_interval <- thinning_interval
  if (!is.null(point_estimator)) settings$point_estimator <- point_estimator
  # `knobs` names are a validated subset of sampler_knobs[[sampler]] above, disjoint from the
  # general settings keys assembled here, so a plain concat -- not utils::modifyList()'s
  # overwrite semantics -- is enough (and keeps this package's zero-runtime-dependency contract:
  # no need to formally Import utils for one call).
  settings <- c(settings, knobs)

  fi <- fit_input(model, distribution, settings)
  result <- ch_fit_run_("BayesianAnalysis", fi$json, fi$dataset)
  new_fit_bayesian(result, fi$spec, fi$dataset, fi$json, warmup, credible_level = credible_level)
}

#' Generalized method of moments fit (Bulletin 17C)
#'
#' Fit a [model_bulletin17c()] model by the generalized method of moments and return a fit object
#' carrying the parameter estimates, the sandwich covariance, and the J-statistic
#' overidentification test. Wraps the shared C++ `GeneralizedMethodOfMoments` ported from
#' USACE-RMC RMC.BestFit -- the same estimator [bulletin17c_analysis()] builds on. `IGMMModel`
#' (the interface GMM fits against) has exactly one implementation, `Bulletin17CDistribution`, so
#' `fit_gmm()` takes only a `model_bulletin17c()` model; unlike [fit_mle()]/[fit_map()] there is no
#' plain-vector-plus-`distribution` convenience path.
#'
#' @param model a [model_bulletin17c()] object.
#' @param optimizer one of `"BFGS"` (default, matching `GeneralizedMethodOfMoments`'s own class
#'   default), `"NelderMead"`, `"Brent"`, `"Powell"`, `"DifferentialEvolution"`,
#'   `"MultilevelSingleLinkage"`.
#' @param strategy GMM estimation strategy: `"Iterative"` (default), `"OneStep"`, or `"TwoStep"`.
#' @param max_gmm_iterations maximum number of GMM iterations; `0` (default) keeps the estimator's
#'   own default cap.
#' @return An object of class `corehydro_fit` with `method == "GMM"`. Method of moments computes
#'   no likelihood surface, so `$log_likelihood`, `$aic` and `$bic` are `NA`; `$nobs` is the
#'   record length the estimator fitted against. Bulletin 17C is always
#'   just-identified (as many moment conditions as parameters), so `$j_stat_pval` is structurally
#'   `NA` -- there is no over-identified case to report a p-value for (see
#'   `docs/upstream-csharp-issues.md`). `$gmm_iterations`, `$converged_within_tolerance`, and
#'   `$optimizer_fallback_count` carry the estimator's own bookkeeping. See
#'   [quantile_variance()] for the delta-method variance of a fitted quantile, and
#'   [fit_diagnostics()] for leverage/influence diagnostics off a GMM fit.
#' @seealso [fit_mle()], [fit_map()], [fit_bayesian()], [fit_diagnostics()],
#'   [quantile_variance()], [bulletin17c_analysis()].
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' f <- fit_gmm(model_bulletin17c(peaks))
#' f$parameters
#' quantile_variance(f, 0.01)
fit_gmm <- function(model, optimizer = "BFGS", strategy = "Iterative", max_gmm_iterations = 0L) {
  known_optimizers <- c(
    "NelderMead", "Brent", "BFGS", "Powell", "DifferentialEvolution", "MultilevelSingleLinkage"
  )
  optimizer <- as.character(optimizer)
  if (!optimizer %in% known_optimizers) {
    stop(sprintf(
      "unknown optimizer '%s'; expected one of %s", optimizer, paste(known_optimizers, collapse = ", ")
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
  if (!inherits(model, "corehydro_model") || !identical(model$spec$type, "bulletin17c")) {
    stop("fit_gmm fits a bulletin17c model only; build one with model_bulletin17c()", call. = FALSE)
  }

  settings <- list(optimizer = optimizer, strategy = strategy)
  max_gmm_iterations <- as.integer(max_gmm_iterations)
  if (max_gmm_iterations > 0L) settings$max_gmm_iterations <- max_gmm_iterations

  fi <- fit_input(model, NULL, settings)
  result <- ch_fit_run_("GMM", fi$json, fi$dataset)
  new_fit_gmm(result, fi$spec, fi$dataset, fi$json)
}

# Fit targets run_fit_diagnostics (fit_runner.hpp) actually populates -- MaximumLikelihood carries
# no posterior/Hessian-at-a-point-estimate diagnostics surface, so it is deliberately absent here.
.fit_diagnostics_targets <- c("MaximumAPosteriori", "BayesianAnalysis", "GMM")

#' Estimation diagnostics for a fit
#'
#' Compute the estimation diagnostics available for a fit: Cook's distance, per-observation
#' leverage, and observation influence (from the objective's Hessian at the fitted point) for
#' [fit_map()] and [fit_gmm()] fits; leverage, PSIS-LOO Pareto-k, and prior influence for
#' [fit_bayesian()] fits. Wraps the shared C++ Diagnostics layer (`LeverageDiagnostics`,
#' `InfluenceDiagnostics`, `PriorInfluenceDiagnostics`).
#'
#' Reruns the fit's own construct through the corresponding C++ diagnostics method rather than
#' reusing anything cached on `fit` -- for a [fit_bayesian()] result this reproduces the identical
#' seeded chain (the construct carries the same explicit `warmup`/`seed`/... the original fit
#' used), so the diagnostics agree with the fit they were computed from.
#'
#' @param fit a `corehydro_fit` from [fit_map()], [fit_bayesian()], or [fit_gmm()]. [fit_mle()]
#'   fits are not supported: `MaximumLikelihood` has no posterior to diagnose.
#' @return A named list. `$cooks_distance` and `$leverage` (one value per observation) and
#'   `$observation_influence` (an observations-by-parameters matrix) are populated for
#'   [fit_map()] and [fit_gmm()] fits. `$pareto_k` (one per observation), `$max_pareto_k`, and
#'   `$prior_influence`/`$prior_influence_names` (one per parameter) are populated for
#'   [fit_bayesian()] fits. Fields the fit's target does not support come back empty.
#' @seealso [fit_map()], [fit_bayesian()], [fit_gmm()], [estimation_diagnostics()].
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' d <- fit_diagnostics(fit_map(model_univariate("Normal", peaks)))
#' d$cooks_distance
fit_diagnostics <- function(fit) {
  if (!inherits(fit, "corehydro_fit")) {
    stop("fit_diagnostics needs a corehydro_fit object", call. = FALSE)
  }
  if (!fit$method %in% .fit_diagnostics_targets) {
    stop(sprintf(
      "fit_diagnostics needs a fit_map(), fit_bayesian(), or fit_gmm() result; got a %s fit",
      fit$method
    ), call. = FALSE)
  }
  ch_fit_diagnostics_(fit$method, fit$construct_json, fit$dataset)
}

#' Delta-method variance of a fitted quantile
#'
#' The Cohn-style delta-method variance of the discharge quantile at a given annual exceedance
#' probability, off a [fit_gmm()] fit's sandwich covariance. Wraps the shared C++
#' `Bulletin17CDistribution::quantile_variance`.
#'
#' @param fit a `corehydro_fit` from [fit_gmm()].
#' @param aep annual exceedance probability (e.g. `0.01` for the 1% AEP / 100-year quantile).
#' @return A single numeric: the variance of the fitted quantile at `aep`.
#' @seealso [fit_gmm()].
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' f <- fit_gmm(model_bulletin17c(peaks))
#' quantile_variance(f, 0.01)
quantile_variance <- function(fit, aep) {
  if (!inherits(fit, "corehydro_fit") || !identical(fit$method, "GMM")) {
    stop("quantile_variance needs a fit_gmm() result", call. = FALSE)
  }
  ch_fit_quantile_variance_(fit$construct_json, fit$dataset, as.double(aep))
}

#' @export
print.corehydro_fit <- function(x, ...) {
  cat(sprintf("<corehydro_fit> %s (%s)\n", x$method, x$status))
  cat("  parameters:\n")
  print(x$parameters)
  # GMM is a method-of-moments fit, not a likelihood-based one: log_likelihood/aic/bic are NA for
  # that target (GMM has no likelihood surface to report them from), so printing them would just
  # be clutter -- the J-statistic line below is GMM's analogous goodness-of-fit summary.
  if (!identical(x$method, "GMM")) {
    cat(sprintf(
      "  log-likelihood: %g   aic: %g   bic: %g   nobs: %d\n",
      x$log_likelihood, x$aic, x$bic, x$nobs
    ))
  }
  if (!is.null(x$dic)) cat(sprintf("  dic: %g\n", x$dic))
  if (!is.null(x$j_stat_pval)) {
    cat(sprintf(
      "  j-statistic: %g   p-value: %s   gmm iterations: %d\n",
      x$j_stat, format(x$j_stat_pval), x$gmm_iterations
    ))
  }
  if (x$method %in% c("MaximumLikelihood", "MaximumAPosteriori")) {
    cat(sprintf(
      "  converged: %s   function evaluations: %d\n", x$converged, x$function_evaluations
    ))
  } else {
    cat(sprintf("  converged: %s\n", x$converged))
  }
  invisible(x)
}

#' @export
summary.corehydro_fit <- function(object, ...) {
  print(object)
  if (!is.null(object$summary)) {
    cat("  posterior summary (rhat, ess):\n")
    print(object$summary)
  } else if (!is.null(object$standard_errors)) {
    cat("  standard errors:\n")
    print(object$standard_errors)
  }
  invisible(object)
}

#' @export
coef.corehydro_fit <- function(object, ...) object$parameters

#' @export
vcov.corehydro_fit <- function(object, ...) object$covariance

#' Log-likelihood of a fit
#'
#' The fitted log-likelihood, with `df` (parameter count) and `nobs` attributes attached so base
#' `AIC()`/`BIC()` work directly off it.
#'
#' @details [fit_gmm()] is method-of-moments: `GeneralizedMethodOfMoments` computes no likelihood
#'   surface, so a GMM fit reports `NA` for `$log_likelihood`, `$aic` and `$bic`, and `logLik()`
#'   returns `NA_real_` (still carrying `df`/`nobs`) so base `AIC()`/`BIC()` come back a
#'   self-explanatory `NA`. The GMM analogue of a likelihood-based goodness-of-fit summary is the J-statistic
#'   overidentification diagnostic already on the fit (`fit$j_stat`/`fit$j_stat_pval`; see
#'   [fit_gmm()]), which is also what `print()` on a `corehydro_fit` shows in place of the
#'   log-likelihood line for a GMM fit.
#' @param object a `corehydro_fit`.
#' @param ... unused; present for generic consistency.
#' @return An object of class `logLik`.
#' @seealso [fit_mle()], [fit_map()], [fit_bayesian()], [fit_gmm()].
#' @export
logLik.corehydro_fit <- function(object, ...) {
  ll <- if (identical(object$method, "GMM")) NA_real_ else object$log_likelihood
  structure(ll,
    df = length(object$parameters), nobs = object$nobs, class = "logLik"
  )
}

#' Confidence or credible intervals for a fit
#'
#' [fit_mle()] and [fit_map()] fits get profile-likelihood confidence intervals; a [fit_bayesian()]
#' fit gets posterior credible intervals. [fit_gmm()] has no interval surface (see below).
#'
#' For [fit_mle()]/[fit_map()]: when the fit already carries a profile block at the requested
#' level (built with `fit_mle(..., profile = TRUE)` or a prior `confint()` call), those bounds are
#' reused; otherwise the identical fit is re-run with profiling turned on, following the
#' lazy-rebuild precedent of `ch_estimation_bic_` (`corehydror/src/estimation.cpp`) -- a
#' deterministic optimizer reproduces the identical point estimate, so the rebuild's confidence
#' intervals are the ones the original fit would have carried had `profile = TRUE` been requested
#' at the matching level up front.
#'
#' For [fit_bayesian()]: `$summary`'s `lower`/`upper` columns are already the posterior credible
#' interval at the fit's own `$credible_level` (the `credible_level` argument, 0.9 by default).
#' When the requested `level` matches, those columns are returned directly; otherwise the
#' identical seeded chain is re-run with `credible_interval_width` set to `level` (the same
#' lazy-rebuild precedent as the profile path above -- re-sampling with an identical seed
#' reproduces the same chain bit-for-bit, so only the post-hoc credible-interval quantile
#' computation changes). Because `confint()`'s own default is 0.95 (the base R `confint`
#' convention) and a fit's default is 0.9, a bare `confint(f)` on a Bayesian fit re-runs the
#' chain; `fit_bayesian(..., credible_level = 0.95)` avoids that.
#'
#' [fit_gmm()] is method-of-moments and has no likelihood or posterior to draw an interval from;
#' calling `confint()` on one errors. Use [quantile_variance()] for the delta-method variance of a
#' fitted quantile instead.
#'
#' @param object a `corehydro_fit` from [fit_mle()], [fit_map()], or [fit_bayesian()].
#' @param parm optional subset of parameters (by name or position); every parameter by default.
#' @param level confidence (MLE/MAP) or credible (Bayesian) level.
#' @param ... unused; present for generic consistency.
#' @return A matrix with one row per parameter and columns `lower`/`upper`.
#' @seealso [fit_mle()], [fit_map()], [fit_bayesian()], [fit_gmm()], [quantile_variance()].
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' f <- fit_mle(model_univariate("Normal", peaks))
#' confint(f, level = 0.9)
confint.corehydro_fit <- function(object, parm, level = 0.95, ...) {
  if (identical(object$method, "GMM")) {
    stop(
      "confint() has no interval surface for a fit_gmm() fit: GMM is method-of-moments, not ",
      "likelihood- or posterior-based, so there is no profile or credible interval to draw; use ",
      "quantile_variance() for the delta-method variance of a fitted quantile instead",
      call. = FALSE
    )
  }

  if (identical(object$method, "BayesianAnalysis")) {
    if (!isTRUE(all.equal(object$credible_level, level))) {
      construct_json <- inject_credible_interval_width(object$construct_json, level)
      result <- ch_fit_run_("BayesianAnalysis", construct_json, object$dataset)
      object <- new_fit_bayesian(
        result, object$spec, object$dataset, construct_json, object$warmup,
        credible_level = level
      )
    }
    out <- cbind(lower = object$summary$lower, upper = object$summary$upper)
    rownames(out) <- rownames(object$summary)
    if (!missing(parm)) out <- out[parm, , drop = FALSE]
    return(out)
  }

  if (is.null(object$profile_intervals) ||
        !isTRUE(all.equal(object$profile_intervals$level, level))) {
    base_model <- structure(
      list(spec = object$spec, dataset = object$dataset),
      class = "corehydro_model"
    )
    bins <- if (is.null(object$profile_intervals)) 100L else object$profile_intervals$bins
    object <- fit_optimized(
      object$method, base_model, NULL, object$optimizer,
      hessian = TRUE, profile = TRUE, profile_bins = bins, alpha = 1 - level
    )
  }
  out <- cbind(lower = object$profile_intervals$lower, upper = object$profile_intervals$upper)
  rownames(out) <- names(object$parameters)
  if (!missing(parm)) out <- out[parm, , drop = FALSE]
  out
}
