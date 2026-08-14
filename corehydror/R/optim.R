# The general-purpose optimizer surface over the six ported Numerics optimizers (DE, BFGS,
# Powell, MLSL, Nelder-Mead, Brent). Unlike every other verb in toolbox.R/gof.R (which pass
# serializable data through the shared run_toolbox dispatcher), an optimizer takes a live R
# function, so this goes through its own runner (core/include/corehydro/numerics/support/
# optimizer_runner.hpp) and its own glue (ch_optim_run_ in src/toolbox.cpp), rather than
# ch_toolbox_run_. Mirrors corehydropy's src/corehydropy/optim.py verb for verb.

#' Minimize or maximize a user-written objective
#'
#' Runs one of the six ported Numerics optimizers over an R function. The optimizer's random
#' number generator lives in C++, so a seeded run reproduces exactly, and reproduces identically
#' in corehydropy.
#'
#' @param objective a function taking a numeric parameter vector and returning a single number.
#' @param lower,upper numeric vectors of parameter bounds, the same length as the parameter
#'   vector. Required for every method, including `"de"` and `"brent"`, which take no `initial`.
#' @param initial optional numeric vector of starting values, the same length as `lower`/`upper`.
#'   Required for `"bfgs"`, `"powell"`, `"mlsl"` and `"nelder_mead"`.
#' @param method one of `"de"` (differential evolution, the default), `"bfgs"`, `"powell"`,
#'   `"mlsl"`, `"nelder_mead"`, or `"brent"`.
#' @param seed optional integer seed for the stochastic methods (`"de"`, `"mlsl"`); an error for
#'   any other method.
#' @param control a named list of optimizer settings: `max_iterations`,
#'   `max_function_evaluations`, `absolute_tolerance`, `relative_tolerance`, `population_size`
#'   (`"de"` only), `compute_hessian`, and `report_failure` (default `TRUE`, which surfaces a
#'   configuration failure as an R error rather than returning a failed status quietly).
#'   `max_function_evaluations`, `report_failure`, and `compute_hessian` apply only to `"de"`,
#'   `"bfgs"`, `"powell"`, and `"mlsl"` (the four methods that derive from the ported `Optimizer`
#'   base); `"nelder_mead"` and `"brent"` accept only `max_iterations`, `absolute_tolerance`, and
#'   `relative_tolerance`, and never compute a Hessian. `compute_hessian` DEFAULTS TO `TRUE` for
#'   those four methods (matching the ported C# `Optimizer` base), so a successful `"de"`/`"bfgs"`/
#'   `"powell"`/`"mlsl"` run returns a Hessian, computed by extra objective evaluations, unless the
#'   caller passes `control = list(compute_hessian = FALSE)` to skip it.
#' @return a `corehydro_optim` list with `parameters`, `value` (the objective's own value at the
#'   optimum, in its own sign convention -- not negated for `optim_maximize()`), `iterations`,
#'   `function_evaluations`, `status`, and `hessian` (populated by default for `"de"`/`"bfgs"`/
#'   `"powell"`/`"mlsl"`; always `NULL` for `"nelder_mead"`/`"brent"`, or when `compute_hessian`
#'   was turned off).
#' @examples
#' rosenbrock <- function(p) (1 - p[1])^2 + 100 * (p[2] - p[1]^2)^2
#' fit <- optim_minimize(rosenbrock, lower = c(-5, -5), upper = c(5, 5), seed = 42)
#' round(fit$parameters, 3)
#' @export
optim_minimize <- function(objective, lower = NULL, upper = NULL, initial = NULL,
                           method = c("de", "bfgs", "powell", "mlsl", "nelder_mead", "brent"),
                           seed = NULL, control = list()) {
  optim_run(objective, lower, upper, initial, match.arg(method), seed, control,
            maximize = FALSE)
}

#' @rdname optim_minimize
#' @export
optim_maximize <- function(objective, lower = NULL, upper = NULL, initial = NULL,
                           method = c("de", "bfgs", "powell", "mlsl", "nelder_mead", "brent"),
                           seed = NULL, control = list()) {
  optim_run(objective, lower, upper, initial, match.arg(method), seed, control,
            maximize = TRUE)
}

kOptimControl <- c("max_iterations", "max_function_evaluations", "absolute_tolerance",
                   "relative_tolerance", "population_size", "compute_hessian", "report_failure")
# Every method needs `lower`/`upper` (even "de" and "brent", which take no `initial`).
# "bfgs", "powell", "mlsl" and "nelder_mead" additionally need an initial guess -- all three of
# `initial`/`lower`/`upper` are then required to be the same length, which the underlying C++
# constructors validate once the request reaches them (NelderMead's ctor does not validate this
# itself -- see optimizer_runner.hpp's "nelder_mead" arm -- so the length check below is load-
# bearing, not just a friendlier error, for that one method).
kOptimNeedsInitial <- c("bfgs", "powell", "mlsl", "nelder_mead")
# The four methods deriving from the ported Optimizer base (see optimizer_runner.hpp); only these
# accept max_function_evaluations/report_failure/compute_hessian, and only "de"/"mlsl" are
# stochastic (accept `seed`).
kOptimBaseMethods <- c("de", "bfgs", "powell", "mlsl")
kOptimStochasticMethods <- c("de", "mlsl")

# Internal: validate everything R-side, then make one call. Both verbs share this so their
# messages and defaults can never drift apart.
optim_run <- function(objective, lower, upper, initial, method, seed, control, maximize) {
  if (!is.function(objective)) {
    stop("`objective` must be a function taking a numeric vector and returning one number",
         call. = FALSE)
  }
  if (is.null(lower) || is.null(upper)) {
    stop(sprintf("method \"%s\" needs `lower` and `upper` bounds", method), call. = FALSE)
  }
  if (length(lower) != length(upper)) {
    stop(sprintf("`lower` and `upper` must have the same length; got %d and %d",
                 length(lower), length(upper)), call. = FALSE)
  }
  if (any(lower >= upper)) {
    stop("every `lower` bound must be below its `upper` bound", call. = FALSE)
  }
  if (method %in% kOptimNeedsInitial && is.null(initial)) {
    stop(sprintf("method \"%s\" needs `initial` starting values", method), call. = FALSE)
  }
  if (!is.null(initial) && length(initial) != length(lower)) {
    stop(sprintf("`initial` must have the same length as `lower`/`upper`; got %d and %d",
                 length(initial), length(lower)), call. = FALSE)
  }
  if (!is.null(seed) && !method %in% kOptimStochasticMethods) {
    stop(sprintf("`seed` only applies to the stochastic methods %s; got method \"%s\"",
                 paste(sprintf('"%s"', kOptimStochasticMethods), collapse = ", "), method),
         call. = FALSE)
  }
  unknown <- setdiff(names(control), kOptimControl)
  if (length(unknown) > 0L) {
    stop(sprintf("unknown control name(s): %s. Available: %s",
                 paste(unknown, collapse = ", "), paste(kOptimControl, collapse = ", ")),
         call. = FALSE)
  }
  if (!method %in% kOptimBaseMethods) {
    base_only <- intersect(names(control), c("max_function_evaluations", "report_failure",
                                             "compute_hessian"))
    if (length(base_only) > 0L) {
      stop(sprintf("control name(s) %s only apply to method(s) %s; got method \"%s\"",
                   paste(base_only, collapse = ", "), paste(kOptimBaseMethods, collapse = ", "),
                   method), call. = FALSE)
    }
  }
  # `population_size` is only ever applied in the "de" arm of the C++ runner (every other method
  # silently ignores it); reject it here rather than let it look like it did something.
  if (method != "de" && "population_size" %in% names(control)) {
    stop(sprintf("control name \"population_size\" only applies to method \"de\"; got method \"%s\"",
                 method), call. = FALSE)
  }
  # `initial` is only read by the "bfgs"/"powell"/"mlsl"/"nelder_mead" arms; "de" and "brent"
  # never look at it, so a caller-supplied `initial` for either would silently do nothing.
  if (!is.null(initial) && !method %in% kOptimNeedsInitial) {
    stop(sprintf("`initial` only applies to method(s) %s; got method \"%s\"",
                 paste(kOptimNeedsInitial, collapse = ", "), method), call. = FALSE)
  }
  spec <- to_spec_json(list(
    method = method, maximize = maximize,
    lower = if (is.null(lower)) NULL else spec_array(as.double(lower)),
    upper = if (is.null(upper)) NULL else spec_array(as.double(upper)),
    initial = if (is.null(initial)) NULL else spec_array(as.double(initial)),
    seed = if (is.null(seed)) NULL else as.integer(seed),
    control = if (length(control) == 0L) NULL else control
  ))
  r <- ch_optim_run_(spec, objective)
  if (length(r$hessian) > 0L) {
    r$hessian <- matrix(r$hessian, nrow = r$hessian_dims[[1]], ncol = r$hessian_dims[[2]],
                        byrow = TRUE)
  } else {
    r$hessian <- NULL
  }
  r$hessian_dims <- NULL
  structure(r, class = "corehydro_optim")
}

#' @export
print.corehydro_optim <- function(x, ...) {
  cat(sprintf("<corehydro_optim> %s after %d iterations (%d evaluations)\n",
              x$status, x$iterations, x$function_evaluations))
  cat(sprintf("  value: %.8g\n", x$value))
  cat(sprintf("  parameters: %s\n", paste(format(x$parameters, digits = 6), collapse = ", ")))
  invisible(x)
}
