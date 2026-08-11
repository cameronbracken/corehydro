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

# Internal: build a corehydro_fit from a ch_fit_run_() result. `base_spec`/`dataset` are what
# fit_input() resolved the construct from -- carried forward so confint() can lazily re-run the
# identical fit (see ch_estimation_bic_'s precedent), and so `$model` can be assembled by
# overriding `base_spec`'s `parameter_values` with the fitted values rather than re-parsing
# `model_spec` (the fitted spec JSON string ch_fit_run_() also returns) back into R structures --
# this package deliberately carries no JSON parser, only the serializer in R/spec.R.
new_fit <- function(result, base_spec, dataset, optimizer, level) {
  n <- length(result$parameters)
  parameters <- result$parameters
  names(parameters) <- result$parameter_names

  reshape <- function(flat) {
    if (!length(flat)) {
      return(NULL)
    }
    m <- matrix(flat, nrow = n, ncol = n, byrow = TRUE)
    dimnames(m) <- list(result$parameter_names, result$parameter_names)
    m
  }
  covariance <- reshape(result$covariance)
  correlation <- reshape(result$correlation)
  standard_errors <- if (length(result$standard_errors)) {
    stats::setNames(result$standard_errors, result$parameter_names)
  } else {
    NULL
  }

  profile <- if (length(result$profile_lower)) {
    list(
      lower = stats::setNames(result$profile_lower, result$parameter_names),
      upper = stats::setNames(result$profile_upper, result$parameter_names),
      bins = result$profile_bins,
      level = level
    )
  } else {
    NULL
  }

  fitted_spec <- base_spec
  fitted_spec$parameter_values <- spec_array(as.double(parameters))
  fitted_model <- structure(
    list(spec = fitted_spec, dataset = dataset),
    class = "corehydro_model"
  )

  structure(
    list(
      method = result$method,
      parameters = parameters,
      log_likelihood = result$log_likelihood,
      prior_log_likelihood = result$prior_log_likelihood,
      aic = result$aic,
      bic = result$bic,
      nobs = result$nobs,
      covariance = covariance,
      standard_errors = standard_errors,
      correlation = correlation,
      converged = result$converged,
      status = result$status,
      function_evaluations = result$function_evaluations,
      profile = profile,
      model = fitted_model,
      spec = base_spec,
      dataset = dataset,
      optimizer = optimizer
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
  settings <- list(
    optimizer = optimizer, hessian = isTRUE(hessian), profile = profile,
    profile_bins = as.integer(profile_bins)
  )
  if (profile) settings$alpha <- as.double(alpha)

  fi <- fit_input(model, distribution, settings)
  result <- ch_fit_run_(target, fi$json, fi$dataset)
  new_fit(result, fi$spec, fi$dataset, optimizer, level = 1 - alpha)
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
#'   default. A model with fewer than two parameters reports `NA` for all three.
#' @param profile logical; also compute the profile likelihood and profile confidence intervals.
#'   `FALSE` by default because each costs `profile_bins * length(parameters)` likelihood
#'   evaluations.
#' @param profile_bins number of bins in each parameter's profile.
#' @return An object of class `corehydro_fit`. See [fit_bayesian()] for the Bayesian surface.
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
#' @return An object of class `corehydro_fit`. See [fit_bayesian()] for the Bayesian surface.
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

#' @export
print.corehydro_fit <- function(x, ...) {
  cat(sprintf("<corehydro_fit> %s (%s)\n", x$method, x$status))
  cat("  parameters:\n")
  print(x$parameters)
  cat(sprintf(
    "  log-likelihood: %g   aic: %g   bic: %g   nobs: %d\n",
    x$log_likelihood, x$aic, x$bic, x$nobs
  ))
  cat(sprintf(
    "  converged: %s   function evaluations: %d\n", x$converged, x$function_evaluations
  ))
  invisible(x)
}

#' @export
summary.corehydro_fit <- function(object, ...) {
  print(object)
  if (!is.null(object$standard_errors)) {
    cat("  standard errors:\n")
    print(object$standard_errors)
  }
  invisible(object)
}

#' @export
coef.corehydro_fit <- function(object, ...) object$parameters

#' @export
vcov.corehydro_fit <- function(object, ...) object$covariance

#' @export
logLik.corehydro_fit <- function(object, ...) {
  structure(object$log_likelihood,
    df = length(object$parameters), nobs = object$nobs, class = "logLik"
  )
}

#' Profile-likelihood confidence intervals for a fit
#'
#' Confidence intervals from the profile likelihood, at the requested level. When the fit already
#' carries a profile block at the requested level (built with `fit_mle(..., profile = TRUE)` or a
#' prior `confint()` call), those bounds are reused; otherwise the identical fit is re-run with
#' profiling turned on, following the lazy-rebuild precedent of `ch_estimation_bic_`
#' (`corehydror/src/estimation.cpp`) -- a deterministic optimizer reproduces the identical point
#' estimate, so the rebuild's confidence intervals are the ones the original fit would have
#' carried had `profile = TRUE` been requested at the matching level up front.
#'
#' @param object a `corehydro_fit` from [fit_mle()] or [fit_map()].
#' @param parm optional subset of parameters (by name or position); every parameter by default.
#' @param level confidence level.
#' @param ... unused; present for generic consistency.
#' @return A matrix with one row per parameter and columns `lower`/`upper`.
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' f <- fit_mle(model_univariate("Normal", peaks))
#' confint(f, level = 0.9)
confint.corehydro_fit <- function(object, parm, level = 0.95, ...) {
  if (is.null(object$profile) || !isTRUE(all.equal(object$profile$level, level))) {
    base_model <- structure(
      list(spec = object$spec, dataset = object$dataset),
      class = "corehydro_model"
    )
    bins <- if (is.null(object$profile)) 100L else object$profile$bins
    object <- fit_optimized(
      object$method, base_model, NULL, object$optimizer,
      hessian = TRUE, profile = TRUE, profile_bins = bins, alpha = 1 - level
    )
  }
  out <- cbind(lower = object$profile$lower, upper = object$profile$upper)
  rownames(out) <- names(object$parameters)
  if (!missing(parm)) out <- out[parm, , drop = FALSE]
  out
}
