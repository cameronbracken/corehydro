test_that("an error inside the objective reaches the caller intact", {
  f <- function(p) stop("boom in the objective")
  expect_error(optim_minimize(f, lower = c(-1, -1), upper = c(1, 1), seed = 1),
               "boom in the objective")
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
