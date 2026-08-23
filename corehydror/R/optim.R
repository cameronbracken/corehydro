# The general-purpose optimizer surface over the fourteen ported Numerics optimizers (DE, particle
# swarm, shuffled complex evolution, simulated annealing, multi-start, MLSL, BFGS, Powell, ADAM,
# gradient descent, Nelder-Mead, Brent, golden section, augmented Lagrange). Unlike every other
# verb in toolbox.R/gof.R
# (which pass serializable data through the shared run_toolbox dispatcher), an optimizer takes a
# live R function, so this goes through its own runner (core/include/corehydro/numerics/support/
# optimizer_runner.hpp) and its own glue (ch_optim_run_ / ch_optim_run_grad_ /
# ch_optim_run_constrained_ in src/toolbox.cpp),
# rather than ch_toolbox_run_. Mirrors corehydropy's src/corehydropy/optim.py verb for verb.

#' Minimize or maximize a user-written objective
#'
#' Runs one of the fourteen ported Numerics optimizers over an R function. The optimizer's random
#' number generator lives in C++, so a seeded run reproduces exactly, and reproduces identically
#' in corehydropy.
#'
#' `optim_maximize()` accepts every method except `"augmented_lagrange"`, which can only minimize:
#' the upstream C# class always drives its inner optimizer through `Minimize()` over an augmented
#' Lagrangian built from the raw objective, so a maximize request would flip the reported sign
#' without flipping the search direction and hand back the constrained minimum. Negate the
#' objective and call `optim_minimize()` instead -- minimizing `-f` subject to the same
#' constraints is exactly maximizing `f`.
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
#'   `"powell"`, `"adam"`, `"gradient_descent"`, `"nelder_mead"`, `"brent"`, `"golden_section"`,
#'   or `"augmented_lagrange"` (the one constrained method, and the one method
#'   `optim_maximize()` rejects -- see Details).
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
#' @param constraints a list of [optim_constraint()] objects. Required by, and accepted only by,
#'   `method = "augmented_lagrange"`.
#' @param inner an optional named list describing the inner optimizer the augmented Lagrange
#'   method drives, with names `method`, `initial`, `lower`, `upper`, `seed` and `control`. Any
#'   vector left out falls back to the top-level one, so `list(method = "powell")` is enough.
#'   Accepted only by `method = "augmented_lagrange"`; omitted, the inner optimizer is `"bfgs"`
#'   over the top-level `initial`/`lower`/`upper`. The inner method may be any method except
#'   `"augmented_lagrange"` itself and the two standalone classes `"nelder_mead"`/`"brent"`.
#' @return a `corehydro_optim` list with `parameters`, `value` (the objective's own value at the
#'   optimum, in its own sign convention -- not negated for `optim_maximize()`), `iterations`,
#'   `function_evaluations`, `status`, and `hessian` (populated by default for every
#'   `Optimizer`-base method; always `NULL` for `"nelder_mead"`/`"brent"`, or when
#'   `compute_hessian` was turned off). For `"augmented_lagrange"` it additionally carries
#'   `multipliers`, a list of the three Lagrange multiplier vectors -- `equality`, `less_than` and
#'   `greater_than` -- each holding one entry per constraint of that type, in the order the
#'   constraints were given.
#' @examples
#' rosenbrock <- function(p) (1 - p[1])^2 + 100 * (p[2] - p[1]^2)^2
#' fit <- optim_minimize(rosenbrock, lower = c(-5, -5), upper = c(5, 5), seed = 42)
#' round(fit$parameters, 3)
#'
#' # Constrained: minimize the same function on the unit disk.
#' con <- optim_constraint(function(p) p[1]^2 + p[2]^2, value = 2, type = "le")
#' fit <- optim_minimize(rosenbrock, initial = c(0, 0), lower = c(-1.5, -1.5),
#'                       upper = c(1.5, 1.5), method = "augmented_lagrange",
#'                       constraints = list(con))
#' round(fit$parameters, 3)
#' @export
optim_minimize <- function(objective, lower = NULL, upper = NULL, initial = NULL,
                           method = c("de", "particle_swarm", "sce", "simulated_annealing",
                                      "multi_start", "mlsl", "bfgs", "powell", "adam",
                                      "gradient_descent", "nelder_mead", "brent",
                                      "golden_section", "augmented_lagrange"),
                           seed = NULL, control = list(), gradient = NULL,
                           constraints = NULL, inner = NULL) {
  optim_run(objective, lower, upper, initial, match.arg(method), seed, control,
            maximize = FALSE, gradient = gradient, constraints = constraints, inner = inner)
}

#' @rdname optim_minimize
#' @export
optim_maximize <- function(objective, lower = NULL, upper = NULL, initial = NULL,
                           method = c("de", "particle_swarm", "sce", "simulated_annealing",
                                      "multi_start", "mlsl", "bfgs", "powell", "adam",
                                      "gradient_descent", "nelder_mead", "brent",
                                      "golden_section", "augmented_lagrange"),
                           seed = NULL, control = list(), gradient = NULL,
                           constraints = NULL, inner = NULL) {
  optim_run(objective, lower, upper, initial, match.arg(method), seed, control,
            maximize = TRUE, gradient = gradient, constraints = constraints, inner = inner)
}

#' Declare one constraint for the augmented Lagrange optimizer
#'
#' Pairs a constraint function with the value it is compared against and the comparison type.
#' Pass a list of these as `constraints` to [optim_minimize()] with
#' `method = "augmented_lagrange"`.
#'
#' @param f a function taking a numeric parameter vector and returning a single number -- the
#'   left-hand side of the constraint.
#' @param value the number `f` is compared against.
#' @param type `"eq"` for `f(x) == value`, `"le"` for `f(x) <= value`, or `"ge"` for
#'   `f(x) >= value`.
#' @param tolerance how far from `value` still counts as feasible, matching the upstream
#'   `Constraint` class's own default.
#' @return a `corehydro_constraint` list.
#' @examples
#' # x + y == 4
#' optim_constraint(function(p) p[1] + p[2], value = 4, type = "eq")
#' @export
optim_constraint <- function(f, value, type = c("eq", "le", "ge"), tolerance = 1e-8) {
  type <- match.arg(type)
  if (!is.function(f)) {
    stop("`f` must be a function taking a numeric vector and returning one number", call. = FALSE)
  }
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value)) {
    stop("`value` must be a single finite number", call. = FALSE)
  }
  if (!is.numeric(tolerance) || length(tolerance) != 1L || !is.finite(tolerance) ||
        tolerance < 0) {
    stop("`tolerance` must be a single non-negative number", call. = FALSE)
  }
  structure(list(f = f, value = as.double(value), type = type,
                 tolerance = as.double(tolerance)),
            class = "corehydro_constraint")
}

#' @export
print.corehydro_constraint <- function(x, ...) {
  op <- c(eq = "==", le = "<=", ge = ">=")[[x$type]]
  cat(sprintf("<corehydro_constraint> f(x) %s %.8g (tolerance %.8g)\n", op, x$value, x$tolerance))
  invisible(x)
}

# The three tolerance/iteration knobs every one of the fourteen optimizer classes exposes, plus the
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
                             stochastic = FALSE, gradient = FALSE),
  # The one constrained method. `needs_initial` is TRUE because its default inner optimizer is
  # BFGS over the top-level `initial`/`lower`/`upper`, and an `inner` spec that omits a vector
  # falls back to the same ones.
  # `minimize_only` because upstream `AugmentedLagrange$Optimize()` always drives its inner
  # optimizer through `Minimize()`, over an augmented Lagrangian built from the RAW objective --
  # so a maximize request flips the outer bookkeeping but not the search direction, and the verb
  # would return the constrained MINIMUM labelled "Success". The ported class mirrors that
  # exactly; the guard is here, on the public verb, rather than in the port.
  augmented_lagrange  = list(controls = kOptimBaseControls, needs_initial = TRUE,
                             stochastic = FALSE, gradient = FALSE, constrained = TRUE,
                             minimize_only = TRUE)
)
kOptimMethodNames <- names(kOptimMethods)
kOptimControl <- unique(unlist(lapply(kOptimMethods, `[[`, "controls"), use.names = FALSE))
kOptimNeedsInitial <- kOptimMethodNames[vapply(kOptimMethods, `[[`, logical(1), "needs_initial")]
kOptimStochasticMethods <- kOptimMethodNames[vapply(kOptimMethods, `[[`, logical(1), "stochastic")]
kOptimGradientMethods <- kOptimMethodNames[vapply(kOptimMethods, `[[`, logical(1), "gradient")]
# The methods taking `constraints` and `inner`. Read off the same table (an absent `constrained`
# field means FALSE) so the two arguments can never drift from the method list above.
kOptimConstraintMethods <- kOptimMethodNames[vapply(
  kOptimMethods, function(m) isTRUE(m$constrained), logical(1))]
# The methods that can only minimize, off the same table (an absent `minimize_only` field means
# FALSE). See the `augmented_lagrange` row for why it is on that one.
kOptimMinimizeOnlyMethods <- kOptimMethodNames[vapply(
  kOptimMethods, function(m) isTRUE(m$minimize_only), logical(1))]
# The three ConstraintType members, in the short spelling optimizer_runner.hpp's spec grammar
# takes: "eq" for ==, "le" for <=, "ge" for >=.
kOptimConstraintTypes <- c("eq", "le", "ge")
# The keys an `inner` sub-spec may carry. `seed` is there because an inner optimizer may be one of
# the stochastic methods; the C++ runner reads exactly these.
kOptimInnerKeys <- c("method", "initial", "lower", "upper", "control", "seed")
# The three local methods MLSL and MultiStart actually construct. ADAM and GradientDescent are
# LocalMethod members upstream but throw "Unsupported local method" inside both classes (see
# optimization/support/local_method.hpp), so they are not offered here.
kOptimLocalMethods <- c("bfgs", "nelder_mead", "powell")

# Internal: the `inner` sub-spec, shaped exactly like a top-level spec (the C++ runner reads both
# through the same reader) minus `maximize`, which AugmentedLagrange never gives its inner
# optimizer -- it always drives it through minimize(), whatever the outer request. A vector left
# out here falls back to the top-level one C++-side, so naming only a method is enough.
optim_inner_spec <- function(inner) {
  list(
    method = inner$method,
    lower = if (is.null(inner$lower)) NULL else spec_array(as.double(inner$lower)),
    upper = if (is.null(inner$upper)) NULL else spec_array(as.double(inner$upper)),
    initial = if (is.null(inner$initial)) NULL else spec_array(as.double(inner$initial)),
    seed = if (is.null(inner$seed)) NULL else as.integer(inner$seed),
    control = if (is.null(inner$control) || length(inner$control) == 0L) NULL else inner$control
  )
}

# Internal: validate everything R-side, then make one call. Both verbs share this so their
# messages and defaults can never drift apart.
optim_run <- function(objective, lower, upper, initial, method, seed, control, maximize,
                      gradient = NULL, constraints = NULL, inner = NULL) {
  if (!is.function(objective)) {
    stop("`objective` must be a function taking a numeric vector and returning one number",
         call. = FALSE)
  }
  if (isTRUE(maximize) && method %in% kOptimMinimizeOnlyMethods) {
    stop(sprintf(paste0("method \"%s\" cannot maximize: upstream AugmentedLagrange always drives ",
                        "its inner optimizer through Minimize() over the raw objective plus ",
                        "penalty, so a maximize request would return the constrained MINIMUM. ",
                        "Negate your objective and call optim_minimize() instead."), method),
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
  for (arg in c("constraints", "inner")) {
    if (!is.null(get(arg)) && !method %in% kOptimConstraintMethods) {
      stop(sprintf("`%s` only applies to method(s) %s; got method \"%s\"", arg,
                   paste(sprintf('"%s"', kOptimConstraintMethods), collapse = ", "), method),
           call. = FALSE)
    }
  }
  if (method %in% kOptimConstraintMethods) {
    if (!is.list(constraints) || length(constraints) == 0L) {
      stop(sprintf(paste0("method \"%s\" needs a non-empty list of `constraints`, each built by ",
                          "`optim_constraint()`"), method), call. = FALSE)
    }
    if (!all(vapply(constraints, inherits, logical(1), "corehydro_constraint"))) {
      stop("every element of `constraints` must come from `optim_constraint()`", call. = FALSE)
    }
    if (!is.null(inner)) {
      if (!is.list(inner) || is.null(inner$method)) {
        stop("`inner` must be a named list carrying at least a `method`", call. = FALSE)
      }
      bad <- setdiff(names(inner), kOptimInnerKeys)
      if (length(bad) > 0L) {
        stop(sprintf("unknown `inner` name(s): %s. Available: %s",
                     paste(bad, collapse = ", "), paste(kOptimInnerKeys, collapse = ", ")),
             call. = FALSE)
      }
      if (!inner$method %in% kOptimMethodNames) {
        stop(sprintf("`inner$method` must be one of %s; got \"%s\"",
                     paste(kOptimMethodNames, collapse = ", "), inner$method), call. = FALSE)
      }
    }
  }
  # An `optim_constraint()` object has a serializable half (type/value/tolerance) and a function
  # half, and the two are paired POSITIONALLY by the C++ runner -- the serializable halves go into
  # the spec's `constraints` array in order, the functions into the callback list in the SAME
  # order. Split in one pass so the two can never fall out of step.
  constraint_specs <- NULL
  constraint_fns <- NULL
  if (!is.null(constraints)) {
    constraint_specs <- spec_array(lapply(constraints, function(cn) {
      list(type = cn$type, value = cn$value, tolerance = cn$tolerance)
    }))
    constraint_fns <- lapply(constraints, `[[`, "f")
  }
  spec <- to_spec_json(list(
    method = method, maximize = maximize,
    lower = if (is.null(lower)) NULL else spec_array(as.double(lower)),
    upper = if (is.null(upper)) NULL else spec_array(as.double(upper)),
    initial = if (is.null(initial)) NULL else spec_array(as.double(initial)),
    seed = if (is.null(seed)) NULL else as.integer(seed),
    constraints = constraint_specs,
    inner = if (is.null(inner)) NULL else optim_inner_spec(inner),
    control = if (length(control) == 0L) NULL else control
  ))
  # The gradient and the constraint functions travel as EXTRA callbacks, not in the spec, so each
  # needs its own entry point (cpp11 registration is by signature). An absent gradient is the
  # ported classes' null Gradient, which falls back to numerical differentiation exactly as C#
  # does.
  r <- if (!is.null(constraint_fns)) {
    ch_optim_run_constrained_(spec, objective, constraint_fns)
  } else if (is.null(gradient)) {
    ch_optim_run_(spec, objective)
  } else {
    ch_optim_run_grad_(spec, objective, gradient)
  }
  # The three multiplier vectors are folded into one list, and are present only for the one method
  # that has multipliers at all.
  if (method %in% kOptimConstraintMethods) {
    r$multipliers <- list(equality = r$lambda, less_than = r$mu, greater_than = r$nu)
  }
  r$lambda <- NULL
  r$mu <- NULL
  r$nu <- NULL
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
  # Only "augmented_lagrange" has multipliers, and only the sets its own constraints populated:
  # each of the three is sized by counting the constraints of that type, so an empty one means the
  # problem carried no constraint of it and there is nothing to show.
  for (set in names(x$multipliers)) {
    if (length(x$multipliers[[set]]) > 0L) {
      cat(sprintf("  %s multipliers: %s\n", gsub("_", " ", set),
                  paste(format(x$multipliers[[set]], digits = 6), collapse = ", ")))
    }
  }
  invisible(x)
}
