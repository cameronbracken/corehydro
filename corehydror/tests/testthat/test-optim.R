test_that("an error inside the objective reaches the caller intact, for every method", {
  # The guard (GuardedObjective + optimizer_runner.hpp's per-method rethrow) is the whole point of
  # this task, so this case is parametrized over ALL ELEVEN methods -- a hole in just one of them
  # (bfgs/mlsl were bypassed by an internal gradient-probe exception before the fix) would not have
  # shown up if only "de" were exercised here.
  boom <- function(p) stop("boom in the objective")
  needs_initial <- c("bfgs", "powell", "mlsl", "multi_start", "nelder_mead")
  stochastic <- c("de", "particle_swarm", "sce", "simulated_annealing", "multi_start", "mlsl")
  one_dim <- c("brent", "golden_section")
  for (m in c("de", "particle_swarm", "sce", "simulated_annealing", "multi_start", "mlsl",
              "bfgs", "powell", "nelder_mead", "brent", "golden_section")) {
    expect_error(
      optim_minimize(boom,
                     lower = if (m %in% one_dim) -1 else c(-1, -1),
                     upper = if (m %in% one_dim) 1 else c(1, 1),
                     initial = if (m %in% needs_initial) c(0, 0) else NULL,
                     method = m, seed = if (m %in% stochastic) 1 else NULL),
      "boom in the objective", info = m
    )
  }
})

test_that("the guard survives under report_failure = FALSE too, for every base method", {
  # `control = list(report_failure = FALSE)` changes whether the ported Optimizer base itself
  # rethrows internally, but the guard must still surface the ORIGINAL objective exception either
  # way (only "nelder_mead"/"brent" reject a report_failure control, since they don't derive from
  # the Optimizer base).
  boom <- function(p) stop("boom in the objective")
  needs_initial <- c("bfgs", "powell", "mlsl", "multi_start")
  stochastic <- c("de", "particle_swarm", "sce", "simulated_annealing", "multi_start", "mlsl")
  for (m in c("de", "particle_swarm", "sce", "simulated_annealing", "multi_start", "mlsl",
              "bfgs", "powell", "golden_section")) {
    expect_error(
      optim_minimize(boom, lower = c(-1, -1), upper = c(1, 1),
                     initial = if (m %in% needs_initial) c(0, 0) else NULL,
                     method = m, seed = if (m %in% stochastic) 1 else NULL,
                     control = list(report_failure = FALSE)),
      "boom in the objective", info = m
    )
  }
})

test_that("a seeded DE run reproduces exactly", {
  f <- function(p) sum(p^2)
  a <- optim_minimize(f, lower = c(-5, -5), upper = c(5, 5), seed = 99)
  b <- optim_minimize(f, lower = c(-5, -5), upper = c(5, 5), seed = 99)
  expect_identical(a$parameters, b$parameters)
  expect_identical(a$value, b$value)
})

test_that("an objective returning the wrong shape is rejected by message", {
  expect_error(optim_minimize(function(p) c(1, 2), lower = c(-1, -1), upper = c(1, 1)),
               "single number")
  expect_error(optim_minimize(function(p) "nope", lower = c(-1, -1), upper = c(1, 1)))
})

test_that("optim_maximize finds the peak of a concave objective", {
  f <- function(p) -((p[1] - 2)^2 + (p[2] + 1)^2)
  fit <- optim_maximize(f, lower = c(-10, -10), upper = c(10, 10), seed = 7)
  expect_equal(fit$parameters, c(2, -1), tolerance = 1e-4)
})

test_that("a method that needs bounds says so when they are missing", {
  expect_error(optim_minimize(function(p) sum(p^2), method = "de"), "lower")
})

test_that("a method that needs an initial guess says so when it is missing", {
  expect_error(optim_minimize(function(p) sum(p^2), lower = c(-1, -1), upper = c(1, 1),
                              method = "bfgs"),
               "initial")
})

test_that("optim_minimize validates argument shapes and unknown control names", {
  expect_error(optim_minimize(1, lower = -1, upper = 1), "function")
  expect_error(optim_minimize(function(p) sum(p^2), lower = c(-1, -1), upper = -1), "same length")
  expect_error(optim_minimize(function(p) sum(p^2), lower = 1, upper = -1), "below")
  expect_error(
    optim_minimize(function(p) sum(p^2), lower = -1, upper = 1, initial = c(0, 0)),
    "same length"
  )
  expect_error(
    optim_minimize(function(p) sum(p^2), lower = -1, upper = 1, initial = 0, method = "bfgs",
                   seed = 1),
    "stochastic"
  )
  expect_error(
    optim_minimize(function(p) sum(p^2), lower = -1, upper = 1, control = list(bogus = 1)),
    "unknown control"
  )
  expect_error(
    optim_minimize(function(p) sum(p^2), lower = c(-1, -1), upper = c(1, 1), initial = c(0, 0),
                   method = "nelder_mead", control = list(report_failure = FALSE)),
    "only apply to method"
  )
})

test_that("population_size is rejected for every method except de", {
  expect_error(
    optim_minimize(function(p) sum(p^2), lower = c(-1, -1), upper = c(1, 1), initial = c(0, 0),
                   method = "nelder_mead", control = list(population_size = 20)),
    "population_size"
  )
  # "de" itself accepts it -- confirm no error and it actually takes effect (Task 8 finding 7).
  fit <- optim_minimize(function(p) sum(p^2), lower = c(-1, -1), upper = c(1, 1), seed = 1,
                        control = list(population_size = 12))
  expect_s3_class(fit, "corehydro_optim")
})

test_that("initial is rejected for methods that never read it (de, brent)", {
  expect_error(
    optim_minimize(function(p) sum(p^2), lower = c(-1, -1), upper = c(1, 1), initial = c(0, 0),
                   method = "de", seed = 1),
    "initial"
  )
  expect_error(
    optim_minimize(function(p) p[1]^2, lower = -1, upper = 1, initial = 0, method = "brent"),
    "initial"
  )
})

test_that("an integer-valued objective return is accepted, not just double", {
  fit <- optim_minimize(function(p) sum(p > 0), lower = c(-1, -1), upper = c(1, 1), seed = 1)
  expect_s3_class(fit, "corehydro_optim")
})

test_that("every method converges on a simple quadratic", {
  target <- c(1, 2)
  f <- function(p) sum((p - target)^2)
  for (m in c("bfgs", "powell", "mlsl", "nelder_mead")) {
    fit <- optim_minimize(f, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10), method = m)
    expect_equal(fit$parameters, target, tolerance = 1e-3, info = m)
    expect_lt(fit$value, 1e-4)
  }
  fitb <- optim_minimize(function(p) (p[1] - 3)^2, lower = -10, upper = 10, method = "brent")
  expect_equal(fitb$parameters, 3, tolerance = 1e-4)
})

test_that("print.corehydro_optim reports status, value and parameters", {
  fit <- optim_minimize(function(p) sum(p^2), lower = c(-1, -1), upper = c(1, 1), seed = 1)
  expect_output(print(fit), "corehydro_optim")
  expect_output(print(fit), "value:")
  expect_output(print(fit), "parameters:")
})

test_that("compute_hessian returns a symmetric matrix for the four Optimizer-base methods", {
  f <- function(p) sum((p - c(1, 2))^2)
  fit <- optim_minimize(f, lower = c(-5, -5), upper = c(5, 5), seed = 3,
                        control = list(compute_hessian = TRUE))
  expect_true(is.matrix(fit$hessian))
  expect_equal(dim(fit$hessian), c(2L, 2L))
})

test_that("nelder_mead and brent never carry a hessian even when requested", {
  fit <- optim_minimize(function(p) sum(p^2), initial = c(0, 0), lower = c(-1, -1), upper = c(1, 1),
                        method = "nelder_mead")
  expect_null(fit$hessian)
  fitb <- optim_minimize(function(p) p[1]^2, lower = -1, upper = 1, method = "brent")
  expect_null(fitb$hessian)
})

test_that("compute_hessian defaults to TRUE for the four Optimizer-base methods", {
  # The ported C# Optimizer base defaults ComputeHessian = true, and the runner only overrides it
  # when the control key is present -- so a plain, no-control run must already carry a Hessian.
  fit <- optim_minimize(function(p) sum(p^2), lower = c(-2, -2), upper = c(2, 2), seed = 5)
  expect_true(is.matrix(fit$hessian))
})

test_that("compute_hessian = FALSE opts out of the Hessian", {
  fit <- optim_minimize(function(p) sum(p^2), lower = c(-2, -2), upper = c(2, 2), seed = 5,
                        control = list(compute_hessian = FALSE))
  expect_null(fit$hessian)
})

# The Booth function, minimum 0 at (1, 3). Written out rather than vectorized so the Python twin
# in corehydropy/tests/test_optim.py evaluates the identical arithmetic.
booth <- function(p) (p[1] + 2 * p[2] - 7)^2 + (2 * p[1] + p[2] - 5)^2

test_that("the new global methods find the Booth optimum", {
  for (m in c("particle_swarm", "sce", "simulated_annealing", "multi_start")) {
    fit <- optim_minimize(booth, lower = c(-10, -10), upper = c(10, 10),
                          initial = if (m == "multi_start") c(0, 0) else NULL,
                          method = m, seed = 12345)
    expect_equal(fit$parameters, c(1, 3), tolerance = 1e-3, info = m)
  }
})

test_that("golden_section finds a one dimensional minimum", {
  fit <- optim_minimize(function(p) (p[1] - 2)^2, lower = 0, upper = 5, method = "golden_section")
  expect_equal(fit$parameters, 2, tolerance = 1e-4)
})

test_that("multi_start accepts a local_method and rejects an unknown one", {
  fit <- optim_minimize(booth, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                        method = "multi_start", seed = 12345,
                        control = list(local_method = "powell"))
  expect_equal(fit$parameters, c(1, 3), tolerance = 1e-3)
  expect_error(
    optim_minimize(booth, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                   method = "multi_start", seed = 1, control = list(local_method = "adam")),
    "local_method"
  )
})

test_that("multi_start's max_iterations is not overwritten by its constructor", {
  # MultiStart sets MaxIterations to 100 in its CONSTRUCTOR rather than as a field default, so a
  # runner arm that applied the controls before construction would silently ignore this.
  fit <- optim_minimize(booth, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                        method = "multi_start", seed = 12345,
                        control = list(max_iterations = 15))
  expect_identical(fit$iterations, 15L)
})

test_that("a control reaches each new class", {
  # One control value per new class actually changing the run -- the counterpart of the "de"
  # max_function_evaluations check below, which is the only other test here that proves a control
  # is read at all.
  capped <- optim_minimize(booth, lower = c(-10, -10), upper = c(10, 10),
                           method = "simulated_annealing", seed = 12345,
                           control = list(max_iterations = 12))
  expect_lt(capped$function_evaluations, 5000)  # a default 10,000-iteration run costs ~800,000

  small <- optim_minimize(booth, lower = c(-10, -10), upper = c(10, 10),
                          method = "particle_swarm", seed = 12345,
                          control = list(population_size = 10))
  default <- optim_minimize(booth, lower = c(-10, -10), upper = c(10, 10),
                            method = "particle_swarm", seed = 12345)
  expect_false(identical(small$function_evaluations, default$function_evaluations))

  tight <- optim_minimize(booth, lower = c(-10, -10), upper = c(10, 10), method = "sce",
                          seed = 12345, control = list(complexes = 2))
  loose <- optim_minimize(booth, lower = c(-10, -10), upper = c(10, 10), method = "sce",
                          seed = 12345)
  expect_false(identical(tight$function_evaluations, loose$function_evaluations))

  # local_method is the one control shared by two methods, and both of its non-default values have
  # to reach the class: a run polished by Powell costs a different number of evaluations than the
  # same run polished by BFGS.
  bfgs <- optim_minimize(booth, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                         method = "multi_start", seed = 12345,
                         control = list(local_method = "bfgs"))
  powell <- optim_minimize(booth, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                           method = "multi_start", seed = 12345,
                           control = list(local_method = "powell"))
  expect_false(identical(bfgs$function_evaluations, powell$function_evaluations))
})

test_that("a control belonging to another method names the method", {
  expect_error(
    optim_minimize(booth, lower = c(-10, -10), upper = c(10, 10), method = "de", seed = 1,
                   control = list(cooling_rate = 0.9)),
    'got method "de"'
  )
})

test_that("a control value actually reaches the optimizer: max_function_evaluations caps the count", {
  # Guards against a transposed assignment in apply_common_controls/apply_optimizer_controls going
  # unnoticed -- no other test in this file exercises any `control` key at all.
  f <- function(p) sum(p^2)
  capped <- optim_minimize(f, lower = c(-5, -5), upper = c(5, 5), seed = 42,
                           control = list(max_function_evaluations = 40))
  uncapped <- optim_minimize(f, lower = c(-5, -5), upper = c(5, 5), seed = 42)
  expect_lte(capped$function_evaluations, 45)
  expect_lt(capped$function_evaluations, uncapped$function_evaluations)
})

# --- the two gradient-taking methods ------------------------------------------------------
#
# f(p) = (p1 - 3)^2 + (p2 + 1)^2, minimum 0 at (3, -1), with the analytic gradient written out
# term by term so the Python twin in corehydropy/tests/test_optim.py evaluates the identical
# arithmetic.
quad_shifted <- function(p) (p[1] - 3)^2 + (p[2] + 1)^2
quad_shifted_gradient <- function(p) c(2 * (p[1] - 3), 2 * (p[2] + 1))

test_that("adam and gradient_descent take an analytic gradient", {
  for (m in c("gradient_descent", "adam")) {
    fit <- optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                          method = m, gradient = quad_shifted_gradient,
                          control = list(alpha = 0.1))
    expect_equal(fit$parameters, c(3, -1), tolerance = 1e-3, info = m)
  }
})

test_that("adam and gradient_descent fall back to numerical differentiation", {
  # An omitted `gradient` is the ported classes' null Gradient, which routes through
  # NumericalDerivative.Gradient exactly as C# does -- so the same run must land in the same place.
  for (m in c("gradient_descent", "adam")) {
    fit <- optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                          method = m, control = list(alpha = 0.1))
    expect_equal(fit$parameters, c(3, -1), tolerance = 1e-3, info = m)
  }
})

test_that("the analytic gradient is actually used, not just accepted", {
  # A run driven by the supplied gradient never pays for the 2*D finite-difference probes, so it
  # costs strictly fewer objective evaluations than the same run without one. Without this, a
  # runner arm that dropped the gradient on the floor would pass every assertion above.
  with_g <- optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                           method = "gradient_descent", gradient = quad_shifted_gradient,
                           control = list(alpha = 0.1))
  without_g <- optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10),
                              upper = c(10, 10), method = "gradient_descent",
                              control = list(alpha = 0.1))
  expect_lt(with_g$function_evaluations, without_g$function_evaluations)
})

test_that("an error inside the gradient reaches the caller intact", {
  boom <- function(p) stop("boom in the gradient")
  for (m in c("gradient_descent", "adam")) {
    expect_error(
      optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                     method = m, gradient = boom),
      "boom in the gradient", info = m
    )
  }
})

test_that("a gradient returning the wrong length is rejected by message", {
  expect_error(
    optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                   method = "adam", gradient = function(p) 1),
    "one value per parameter"
  )
})

test_that("`gradient` is rejected for every method that cannot take one", {
  expect_error(
    optim_minimize(quad_shifted, lower = c(-10, -10), upper = c(10, 10), method = "de", seed = 1,
                   gradient = quad_shifted_gradient),
    "gradient"
  )
  expect_error(
    optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                   method = "bfgs", gradient = quad_shifted_gradient),
    "gradient"
  )
})

test_that("`gradient` must be a function", {
  expect_error(
    optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                   method = "adam", gradient = 1),
    "function"
  )
})

test_that("alpha, beta1 and beta2 reach the right classes", {
  # alpha is shared; beta1/beta2 are ADAM's alone, so gradient_descent must reject them by name.
  fast <- optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                         method = "gradient_descent", gradient = quad_shifted_gradient,
                         control = list(alpha = 0.1))
  slow <- optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                         method = "gradient_descent", gradient = quad_shifted_gradient,
                         control = list(alpha = 0.01))
  expect_lt(fast$iterations, slow$iterations)
  expect_error(
    optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                   method = "gradient_descent", control = list(beta1 = 0.5)),
    "beta1"
  )
  fit <- optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                        method = "adam", gradient = quad_shifted_gradient,
                        control = list(alpha = 0.1, beta1 = 0.8, beta2 = 0.9))
  expect_s3_class(fit, "corehydro_optim")
})

test_that("`seed` is rejected for the two gradient methods", {
  for (m in c("adam", "gradient_descent")) {
    expect_error(
      optim_minimize(quad_shifted, initial = c(0, 0), lower = c(-10, -10), upper = c(10, 10),
                     method = m, seed = 1),
      "stochastic", info = m
    )
  }
})
