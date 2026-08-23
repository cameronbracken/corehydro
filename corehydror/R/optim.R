# The general-purpose optimizer surface over the thirteen ported Numerics optimizers (DE, particle
# swarm, shuffled complex evolution, simulated annealing, multi-start, MLSL, BFGS, Powell, ADAM,
# gradient descent, Nelder-Mead, Brent, golden section). Unlike every other verb in toolbox.R/gof.R
# (which pass serializable data through the shared run_toolbox dispatcher), an optimizer takes a
# live R function, so this goes through its own runner (core/include/corehydro/numerics/support/
# optimizer_runner.hpp) and its own glue (ch_optim_run_ / ch_optim_run_grad_ in src/toolbox.cpp),
# rather than ch_toolbox_run_. Mirrors corehydropy's src/corehydropy/optim.py verb for verb.

#' Minimize or maximize a user-written objective
#'
#' Runs one of the thirteen ported Numerics optimizers over an R function. The optimizer's random
#' number generator lives in C++, so a seeded run reproduces exactly, and reproduces identically
#' in corehydropy.
#'
#' @param objective a function taking a numeric parameter vector and returning a single number.
#' @param lower,upper numeric vectors of parameter bounds, the same length as the parameter
#'   vector. Required for every method, including `"de"`, `"brent"` and `"golden_section"`, which
#'   take no `initial`. `"brent"` and `"golden_section"` are one-dimensional: pass a single bound
#'   each.
#' @param initial optional numeric vector of starting values, the same length as `lower`/`upper`.
#'   Required for `"bfgs"`, `"powell"`, `"mlsl"`, `"multi_start"`, `"adam"`, `"gradient_descent"`
#'   and `"nelder_mead"`.
#' @param method one of `"de"` (differential evolution, the default), `"particle_swarm"`, `"sce"`
#'   (shuffled complex evolution), `"simulated_annealing"`, `"multi_start"`, `"mlsl"`, `"bfgs"`,
#'   `"powell"`, `"adam"`, `"gradient_descent"`, `"nelder_mead"`, `"brent"`, or
#'   `"golden_section"`.
#' @param gradient optional function taking the parameter vector and returning one partial
#'   derivative per parameter. Accepted only by `"adam"` and `"gradient_descent"`, an error for
#'   every other method. Omitted, both methods differentiate the objective numerically, exactly as
#'   the upstream C# classes do with a null gradient.
#' @param seed optional integer seed for the stochastic methods (`"de"`, `"particle_swarm"`,
#'   `"sce"`, `"simulated_annealing"`, `"multi_start"`, `"mlsl"`); an error for any other method.
#' @param control a named list of optimizer settings. Every method accepts `max_iterations`,
#'   `absolute_tolerance` and `relative_tolerance`. Every method except `"nelder_mead"` and
#'   `"brent"` (the two classes that do not derive from the ported `Optimizer` base) additionally
#'   accepts `max_function_evaluations`, `report_failure` (default `TRUE`, which surfaces a
#'   configuration failure as an R error rather than returning a failed status quietly) and
#'   `compute_hessian`. The method-specific settings are `population_size` (`"de"`,
#'   `"particle_swarm"`); `complexes`, `cce_iterations` and `tolerance_steps` (`"sce"`);
#'   `initial_temperature`, `min_temperature`, `cooling_rate`, `update_cycles`,
#'   `temperature_cycles` and `tolerance_steps` (`"simulated_annealing"`); `local_method`
#'   (`"multi_start"`, `"mlsl"`, one of `"bfgs"`, `"nelder_mead"`, `"powell"`);
#'   `local_absolute_tolerance`, `local_relative_tolerance`, `polish` (`"multi_start"`); `alpha`,
#'   the step size or learning rate (`"adam"`, `"gradient_descent"`); and `beta1`, `beta2`, the
#'   two decay factors (`"adam"`). Passing a
#'   setting a method does not read is an error rather than a silent no-op. `compute_hessian`
#'   DEFAULTS TO `TRUE` for the `Optimizer`-base methods (matching the ported C# `Optimizer` base),
#'   so a successful run returns a Hessian, computed by extra objective evaluations, unless the
#'   caller passes `control = list(compute_hessian = FALSE)` to skip it.
#' @return a `corehydro_optim` list with `parameters`, `value` (the objective's own value at the
#'   optimum, in its own sign convention -- not negated for `optim_maximize()`), `iterations`,
#'   `function_evaluations`, `status`, and `hessian` (populated by default for every
#'   `Optimizer`-base method; always `NULL` for `"nelder_mead"`/`"brent"`, or when
#'   `compute_hessian` was turned off).
#' @examples
#' rosenbrock <- function(p) (1 - p[1])^2 + 100 * (p[2] - p[1]^2)^2
#' fit <- optim_minimize(rosenbrock, lower = c(-5, -5), upper = c(5, 5), seed = 42)
#' round(fit$parameters, 3)
#' @export
optim_minimize <- function(objective, lower = NULL, upper = NULL, initial = NULL,
                           method = c("de", "particle_swarm", "sce", "simulated_annealing",
                                      "multi_start", "mlsl", "bfgs", "powell", "adam",
                                      "gradient_descent", "nelder_mead", "brent",
                                      "golden_section"),
                           seed = NULL, control = list(), gradient = NULL) {
  optim_run(objective, lower, upper, initial, match.arg(method), seed, control,
            maximize = FALSE, gradient = gradient)
}

#' @rdname optim_minimize
#' @export
optim_maximize <- function(objective, lower = NULL, upper = NULL, initial = NULL,
                           method = c("de", "particle_swarm", "sce", "simulated_annealing",
                                      "multi_start", "mlsl", "bfgs", "powell", "adam",
                                      "gradient_descent", "nelder_mead", "brent",
                                      "golden_section"),
                           seed = NULL, control = list(), gradient = NULL) {
  optim_run(objective, lower, upper, initial, match.arg(method), seed, control,
            maximize = TRUE, gradient = gradient)
}

# The three tolerance/iteration knobs every one of the thirteen optimizer classes exposes, plus the
# three only the classes deriving from the ported `Optimizer` base take. "nelder_mead" and "brent"
# are the two standalone classes (see optimizer_runner.hpp's file header); "golden_section" IS an
# `Optimizer` subclass, so it takes the full base set.
kOptimScalarControls <- c("max_iterations", "absolute_tolerance", "relative_tolerance")
kOptimBaseControls <- c(kOptimScalarControls, "max_function_evaluations", "report_failure",
                        "compute_hessian")
# Per-method: which control names it accepts, whether it needs `initial`, whether it is stochastic
# (accepts `seed`), and whether it takes an analytic `gradient`. Every validation message below
# reads off this one table so the R and Python surfaces cannot drift. Every method needs
# `lower`/`upper` (even "de", "brent" and "golden_section", which take no `initial`); a method with
# `needs_initial` additionally needs an initial guess -- all three of `initial`/`lower`/`upper` are
# then required to be the same length, which the underlying C++ constructors validate once the
# request reaches them (NelderMead's ctor does not validate this itself -- see
# optimizer_runner.hpp's "nelder_mead" arm -- so the length check below is load-bearing, not just a
# friendlier error, for that one method).
kOptimMethods <- list(
  de                  = list(controls = c(kOptimBaseControls, "population_size"),
                             needs_initial = FALSE, stochastic = TRUE, gradient = FALSE),
  particle_swarm      = list(controls = c(kOptimBaseControls, "population_size"),
                             needs_initial = FALSE, stochastic = TRUE, gradient = FALSE),
  sce                 = list(controls = c(kOptimBaseControls, "complexes", "cce_iterations",
                                          "tolerance_steps"),
                             needs_initial = FALSE, stochastic = TRUE, gradient = FALSE),
  simulated_annealing = list(controls = c(kOptimBaseControls, "initial_temperature",
                                          "min_temperature", "cooling_rate", "update_cycles",
                                          "temperature_cycles", "tolerance_steps"),
                             needs_initial = FALSE, stochastic = TRUE, gradient = FALSE),
  multi_start         = list(controls = c(kOptimBaseControls, "local_method",
                                          "local_absolute_tolerance", "local_relative_tolerance",
                                          "polish"),
                             needs_initial = TRUE, stochastic = TRUE, gradient = FALSE),
  mlsl                = list(controls = c(kOptimBaseControls, "local_method"),
                             needs_initial = TRUE, stochastic = TRUE, gradient = FALSE),
  bfgs                = list(controls = kOptimBaseControls, needs_initial = TRUE,
                             stochastic = FALSE, gradient = FALSE),
  powell              = list(controls = kOptimBaseControls, needs_initial = TRUE,
                             stochastic = FALSE, gradient = FALSE),
  adam                = list(controls = c(kOptimBaseControls, "alpha", "beta1", "beta2"),
                             needs_initial = TRUE, stochastic = FALSE, gradient = TRUE),
  gradient_descent    = list(controls = c(kOptimBaseControls, "alpha"),
                             needs_initial = TRUE, stochastic = FALSE, gradient = TRUE),
  nelder_mead         = list(controls = kOptimScalarControls, needs_initial = TRUE,
                             stochastic = FALSE, gradient = FALSE),
  brent               = list(controls = kOptimScalarControls, needs_initial = FALSE,
                             stochastic = FALSE, gradient = FALSE),
  golden_section      = list(controls = kOptimBaseControls, needs_initial = FALSE,
                             stochastic = FALSE, gradient = FALSE)
)
kOptimMethodNames <- names(kOptimMethods)
kOptimControl <- unique(unlist(lapply(kOptimMethods, `[[`, "controls"), use.names = FALSE))
kOptimNeedsInitial <- kOptimMethodNames[vapply(kOptimMethods, `[[`, logical(1), "needs_initial")]
kOptimStochasticMethods <- kOptimMethodNames[vapply(kOptimMethods, `[[`, logical(1), "stochastic")]
kOptimGradientMethods <- kOptimMethodNames[vapply(kOptimMethods, `[[`, logical(1), "gradient")]
# The three local methods MLSL and MultiStart actually construct. ADAM and GradientDescent are
# LocalMethod members upstream but throw "Unsupported local method" inside both classes (see
# optimization/support/local_method.hpp), so they are not offered here.
kOptimLocalMethods <- c("bfgs", "nelder_mead", "powell")

# Internal: validate everything R-side, then make one call. Both verbs share this so their
# messages and defaults can never drift apart.
optim_run <- function(objective, lower, upper, initial, method, seed, control, maximize,
                      gradient = NULL) {
  if (!is.function(objective)) {
    stop("`objective` must be a function taking a numeric vector and returning one number",
         call. = FALSE)
  }
  if (!is.null(gradient) && !is.function(gradient)) {
    stop("`gradient` must be a function taking a numeric vector and returning one value per ",
         "parameter", call. = FALSE)
  }
  if (!is.null(gradient) && !method %in% kOptimGradientMethods) {
    stop(sprintf("`gradient` only applies to method(s) %s; got method \"%s\"",
                 paste(sprintf('"%s"', kOptimGradientMethods), collapse = ", "), method),
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
  # A control name this method's C++ arm never reads would silently look like it did something,
  # so reject it and name the methods that do read it.
  wrong_method <- setdiff(names(control), kOptimMethods[[method]]$controls)
  if (length(wrong_method) > 0L) {
    accepts <- kOptimMethodNames[vapply(kOptimMethods,
                                        function(m) any(wrong_method %in% m$controls),
                                        logical(1))]
    stop(sprintf("control name(s) %s only apply to method(s) %s; got method \"%s\"",
                 paste(wrong_method, collapse = ", "), paste(accepts, collapse = ", "), method),
         call. = FALSE)
  }
  if ("local_method" %in% names(control)) {
    lm <- control$local_method
    if (length(lm) != 1L || !is.character(lm) || !lm %in% kOptimLocalMethods) {
      stop(sprintf("`local_method` must be one of %s; got %s",
                   paste(sprintf('"%s"', kOptimLocalMethods), collapse = ", "),
                   paste(format(lm), collapse = ", ")), call. = FALSE)
    }
  }
  # `initial` is only read by the arms of the methods that need it; the others never look at it,
  # so a caller-supplied `initial` for one of those would silently do nothing.
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
  # The gradient travels as a SECOND callback, not in the spec, so it needs its own entry point
  # (cpp11 registration is by signature). An absent gradient is the ported classes' null Gradient,
  # which falls back to numerical differentiation exactly as C# does.
  r <- if (is.null(gradient)) {
    ch_optim_run_(spec, objective)
  } else {
    ch_optim_run_grad_(spec, objective, gradient)
  }
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
