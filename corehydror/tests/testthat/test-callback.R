test_that("root_find solves a user-written R function", {
  expect_equal(root_find(function(x) x^2 - 2, lower = 0, upper = 2), sqrt(2), tolerance = 1e-8)
  # The bracket is required to change sign, and the ported Brent says so.
  expect_error(root_find(function(x) x^2 + 1, lower = 0, upper = 2), "not bracketed")
})

test_that("derivative differentiates a user-written R function", {
  # f(x) = x^3, f'(2) = 12
  expect_equal(derivative(function(x) x^3, 2), 12, tolerance = 1e-6)
  # f(x) = sin(x), f'(pi / 3) = cos(pi / 3) = 0.5
  expect_equal(derivative(sin, pi / 3), 0.5, tolerance = 1e-6)
})

test_that("gradient and hessian differentiate a user-written R function", {
  rosenbrock <- function(p) (1 - p[1])^2 + 100 * (p[2] - p[1]^2)^2
  expect_equal(gradient(rosenbrock, c(1, 1)), c(0, 0), tolerance = 1e-6)
  h <- hessian(rosenbrock, c(1, 1))
  expect_equal(dim(h), c(2L, 2L))
  expect_equal(h[1, 2], h[2, 1], tolerance = 1e-6)

  # f(x, y) = x^2 + 2y^2 + xy has the constant Hessian [[2, 1], [1, 4]].
  quad <- function(p) p[1]^2 + 2 * p[2]^2 + p[1] * p[2]
  expect_equal(hessian(quad, c(1, 2)), matrix(c(2, 1, 1, 4), 2, 2), tolerance = 1e-3)
})

test_that("quadrature integrates a user-written R function", {
  # Closed forms only: the C#-pinned oracles live in fixtures/callback/math.json.
  q <- quadrature(function(x) x^2, lower = 0, upper = 3)
  expect_equal(as.numeric(q), 9, tolerance = 1e-10)
  expect_equal(attr(q, "status"), "Success")
  expect_gt(attr(q, "function_evaluations"), 0L)
  expect_equal(as.numeric(quadrature(sin, lower = 0, upper = pi)), 2, tolerance = 1e-8)
  expect_equal(as.numeric(quadrature(exp, lower = 0, upper = 1)), exp(1) - 1, tolerance = 1e-8)
})

test_that("quadrature reports the evaluation cap in its status instead of raising", {
  # 1 / sqrt(x) is unbounded at 0, so the rule never converges and the cap stops it.
  q <- quadrature(function(x) if (x <= 0) 0 else 1 / sqrt(x),
                  lower = 0, upper = 1, max_function_evaluations = 1000L)
  expect_equal(attr(q, "status"), "MaximumFunctionEvaluationsReached")
  expect_gte(attr(q, "function_evaluations"), 1000L)
})

test_that("an error raised inside the callback reaches the caller unchanged", {
  # Every method, not just one: the guard's sentinel is a value each ported routine can itself
  # reject, so an unwrapped drive site would report the internal error instead of this one.
  boom <- function(x) stop("my own error")
  expect_error(root_find(boom, lower = 0, upper = 2), "my own error")
  expect_error(derivative(boom, 2), "my own error")
  expect_error(gradient(boom, c(1, 1)), "my own error")
  expect_error(hessian(boom, c(1, 1)), "my own error")
  # Quadrature is the arm where the TRAILING rethrow carries the weight: the ported integrator
  # neither throws nor converges on the guard's NaN sentinel, it just runs to the evaluation cap
  # and reports a NaN result, which without the rethrow would look like an answer.
  expect_error(quadrature(boom, lower = 0, upper = 3, max_function_evaluations = 5000L),
               "my own error")
})

test_that("a callback returning a non-scalar is rejected", {
  expect_error(root_find(function(x) c(1, 2), lower = 0, upper = 2), "single number")
  expect_error(derivative(function(x) c(1, 2), 2), "single number")
  expect_error(gradient(function(p) c(1, 2), c(1, 1)), "single number")
  expect_error(quadrature(function(x) c(1, 2), lower = 0, upper = 1), "single number")
})

test_that("the R-side argument checks fire before the core is reached", {
  expect_error(root_find("not a function", lower = 0, upper = 2))
  expect_error(root_find(function(x) x, lower = c(0, 1), upper = 2))
  expect_error(gradient(function(p) sum(p), "not numeric"))
  expect_error(quadrature(function(x) x, lower = 1, upper = 0), "must be below")
  expect_error(quadrature(function(x) x, lower = 0, upper = 1, absolute_tolerance = 0),
               "between 1e-15 and 1")
  expect_error(quadrature(function(x) x, lower = 0, upper = 1, relative_tolerance = 2),
               "between 1e-15 and 1")
})

test_that("a non-finite point is rejected by name, the same way Python rejects it", {
  # Without the finite check this failed three layers down in the spec serializer with
  # "spec values must be finite", naming neither `x` nor the verb. The Python twin
  # (corehydropy/tests/test_callback.py) asserts the identical message.
  f <- function(p) sum(p^2)
  expect_error(gradient(f, c(1, Inf)), "`x` must be a non-empty numeric vector of finite values")
  expect_error(hessian(f, c(1, -Inf)), "`x` must be a non-empty numeric vector of finite values")
  expect_error(gradient(f, c(1, NA_real_)), "`x` must be a non-empty numeric vector of finite values")
  expect_error(gradient(f, c(1, NaN)), "`x` must be a non-empty numeric vector of finite values")
  expect_error(gradient(f, numeric(0)), "`x` must be a non-empty numeric vector of finite values")
  # derivative()'s scalar point already had the finite check; it stays.
  expect_error(derivative(function(x) x^3, Inf), "single finite number")
})
