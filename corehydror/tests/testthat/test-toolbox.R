test_that("chunked running_statistics() matches a single call over the concatenated data", {
  chunk1 <- c(2, 4, 4, 4)
  chunk2 <- c(5, 5, 7, 9)
  whole <- summary_statistics(c(chunk1, chunk2))

  s <- running_statistics(chunk1)
  s <- running_statistics(chunk2, state = s)

  expect_equal(s$n, whole[["n"]])
  expect_equal(s$mean, whole[["mean"]])
  expect_equal(s$variance, whole[["variance"]])
  expect_equal(s$skewness, whole[["skewness"]])
  expect_equal(s$kurtosis, whole[["kurtosis"]])
})

test_that("running_statistics() with no state matches summary_statistics()", {
  x <- c(2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9)
  s <- running_statistics(x)
  whole <- summary_statistics(x)
  expect_equal(s$mean, whole[["mean"]])
  expect_equal(s$sd, whole[["sd"]])
})

test_that("running_statistics() prints without erroring", {
  s <- running_statistics(c(1, 2, 3))
  expect_output(print(s), "corehydro_running")
})

test_that("product_moments and l_moments return the expected shape and names", {
  x <- c(2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9)
  pm <- product_moments(x)
  expect_named(pm, c("mean", "sd", "skewness", "kurtosis"))
  lm <- l_moments(x)
  expect_named(lm, c("l1", "l2", "t3", "t4"))
})

test_that("ranks() handles ties by averaging the tied ranks", {
  expect_equal(ranks(c(3, 1, 2, 1)), c(4, 1.5, 3, 1.5))
})

test_that("percentile() accepts a vector of probabilities", {
  x <- c(1, 2, 3, 4, 5)
  p <- percentile(x, probs = c(0, 0.5, 1))
  expect_length(p, 3L)
  expect_equal(p, c(1, 3, 5))
})

test_that("percentile() rejects a non-numeric probs argument, naming it", {
  expect_error(percentile(c(1, 2, 3), probs = "half"), "probs")
})

test_that("correlation() with a matrix returns the p-by-p matrix, diagonal 1, symmetric, and off-diagonals matching the pairwise calls", {
  c0 <- c(14, 8, 32, 7, 3, 15)
  c1 <- c(10, 5, 7, 4, 3, 8)
  c2 <- c(2, 9, 1, 6, 4, 11)
  x <- cbind(a = c0, b = c1, c = c2)

  m <- correlation(x)
  expect_equal(dim(m), c(3L, 3L))
  expect_equal(unname(diag(m)), c(1, 1, 1))
  expect_true(isSymmetric(m))
  expect_equal(m["a", "b"], correlation(c0, c1))
  expect_equal(m["a", "c"], correlation(c0, c2))
  expect_equal(m["b", "c"], correlation(c1, c2))
  expect_equal(rownames(m), c("a", "b", "c"))
  expect_equal(colnames(m), c("a", "b", "c"))

  ms <- correlation(x, method = "spearman")
  expect_equal(unname(diag(ms)), c(1, 1, 1))
  expect_equal(ms["a", "b"], correlation(c0, c1, method = "spearman"))
})

test_that("correlation() accepts a data frame in matrix mode", {
  x <- data.frame(a = c(14, 8, 32, 7, 3, 15), b = c(10, 5, 7, 4, 3, 8))
  m <- correlation(x)
  expect_equal(dim(m), c(2L, 2L))
  expect_equal(m["a", "b"], correlation(x$a, x$b))
})

test_that("correlation() rejects method = 'kendall' for a matrix, naming the reason", {
  x <- cbind(c(14, 8, 32, 7, 3, 15), c(10, 5, 7, 4, 3, 8))
  expect_error(correlation(x, method = "kendall"), "kendall.*matrix")
})

test_that("correlation() rejects a 1D vector in matrix mode instead of silently returning a 1x1 matrix (C4)", {
  # C4 (P4 whole-branch review): as.matrix() on a plain vector silently coerces it to an n x 1
  # matrix, so `correlation(x)` on a bare vector used to return a 1x1 matrix of 1 in R instead of
  # raising the way corehydropy does for the same 1D input.
  expect_error(correlation(c(14, 8, 32, 7, 3, 15)),
               "`x` must be a 2D array (observations in rows, variables in columns)", fixed = TRUE)
})

test_that("running_covariance() chunked matches a single call, and round-trips through state", {
  x <- matrix(c(1, 2, 3, 4, 5, 1, 2, 3, 4, 5), ncol = 2)
  whole <- running_covariance(x)

  c1 <- running_covariance(x[1:2, , drop = FALSE])
  c2 <- running_covariance(x[3:5, , drop = FALSE], state = c1)

  expect_equal(c2$n, whole$n)
  expect_equal(c2$mean, whole$mean)
  expect_equal(c2$covariance, whole$covariance, tolerance = 1e-9)
})

test_that("running_covariance() rejects a state from a different accumulator kind", {
  expect_error(running_covariance(matrix(1:4, ncol = 2), state = list()), "corehydro_running_covariance")
})

test_that("running_covariance() resumes a one-column (single-variable) accumulator", {
  # Regression test: a length-1 `mean`/`covariance` in the resume state used to serialize as a
  # JSON scalar rather than an array (to_spec_json's length-1 rule), which the C++ side rejects
  # with "expected a JSON array". The two-variable tests above never caught this.
  x <- c(1, 2, 3)
  whole <- running_covariance(matrix(c(x, 4, 5, 6), ncol = 1))

  s <- running_covariance(matrix(x, ncol = 1))
  s <- running_covariance(matrix(c(4, 5, 6), ncol = 1), state = s)

  expect_equal(s$n, whole$n)
  expect_equal(s$mean, whole$mean)
  expect_equal(s$covariance, whole$covariance, tolerance = 1e-9)
})

test_that("autocorrelation() at lag 0 is 1 for the correlation type", {
  x <- c(5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 5, 6, 4, 7, 3, 8, 2, 9, 1, 10)
  a <- autocorrelation(x, max_lag = 5)
  expect_equal(a$lag[1], 0)
  expect_equal(a$value[1], 1, tolerance = 1e-12)
  expect_length(a$ci, 2L)
  expect_true(a$ci[["lower"]] < 0 && a$ci[["upper"]] > 0)
})

test_that("autocorrelation() covariance type at lag 0 equals the population variance", {
  x <- c(5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 5, 6, 4, 7, 3, 8, 2, 9, 1, 10)
  a <- autocorrelation(x, max_lag = 5, type = "covariance")
  pop_var <- mean((x - mean(x))^2)
  expect_equal(a$value[1], pop_var, tolerance = 1e-9)
})

test_that("autocorrelation() rejects a too-short series, naming the argument", {
  expect_error(autocorrelation(1), "x.*at least two")
})

test_that("dft() round-trips to within 1e-12", {
  x <- c(1, 0, 2, 0, 3, 0, 4, 0)
  n <- length(x) / 2
  z <- dft(dft(x, inverse = TRUE))
  expect_equal(z / n, x, tolerance = 1e-12)
})

test_that("cross_correlation() rejects mismatched lengths, naming the arguments", {
  expect_error(cross_correlation(c(1, 2, 3, 4), c(1, 2)), "x.*y")
})

test_that("histogram() bin frequencies sum to length(x)", {
  x <- c(1, 2, 2.5, 3, 3.5, 4, 5, 7, 8, 9)
  h <- histogram(x)
  expect_equal(sum(h$frequency), length(x))
  expect_equal(attr(h, "bins"), nrow(h))
})

test_that("histogram() with an explicit bin count uses exactly that many bins", {
  x <- c(1, 2, 2.5, 3, 3.5, 4, 5, 7, 8, 9)
  h <- histogram(x, bins = 4)
  expect_equal(nrow(h), 4L)
  expect_equal(sum(h$frequency), length(x))
})

test_that("interpolate() clamps to the end knot without extrapolate, and extends with it", {
  x <- c(1, 2, 3, 4)
  y <- c(10, 20, 30, 40)
  clamped <- interpolate(x, y, 10)
  expect_equal(clamped, 40)
  extended <- interpolate(x, y, 10, extrapolate = TRUE)
  expect_equal(extended, 100)
})

test_that("interpolate() rejects an unknown transform name, listing the accepted values", {
  expect_error(interpolate(c(1, 2), c(1, 2), 1.5, x_transform = "bogus"),
               "none.*log.*normal_z")
})

test_that("interpolate() on a log-log grid reproduces the C# Test_Log oracle", {
  x <- c(50, 100, 150, 200, 250)
  y <- c(100, 200, 300, 400, 500)
  expect_equal(interpolate(x, y, 75, x_transform = "log", y_transform = "log"), 150.0,
               tolerance = 1e-6)
})

test_that("interpolate() with method = \"cubic_spline\" reproduces the C# Test_CubicSpline oracle", {
  x <- c(6, 24, 48, 72)
  y <- c(9.96, 22.13, 32.27, 37.60)
  expect_equal(interpolate(x, y, 8, method = "cubic_spline"), 11.4049889205445, tolerance = 1e-6)
})

test_that("interpolate() with method = \"polynomial\" order = 3 reproduces the C# Test_Polynomial oracle", {
  x <- c(6, 24, 48, 72)
  y <- c(9.96, 22.13, 32.27, 37.60)
  expect_equal(interpolate(x, y, 8, method = "polynomial", order = 3), 11.5415808882467,
               tolerance = 1e-6)
})

test_that("interpolate() requires `order` for method = \"polynomial\"", {
  expect_error(interpolate(c(1, 2, 3, 4), c(10, 20, 30, 40), 1.5, method = "polynomial"),
               "order")
})

test_that("interpolate() rejects `order` for a non-polynomial method", {
  expect_error(interpolate(c(1, 2, 3, 4), c(10, 20, 30, 40), 1.5, order = 3),
               "order")
})

test_that("interpolate() rejects a non-default transform/extrapolate for a non-linear method", {
  expect_error(
    interpolate(c(1, 2, 3, 4), c(10, 20, 30, 40), 1.5, method = "cubic_spline",
                x_transform = "log"),
    "linear-only"
  )
  expect_error(
    interpolate(c(1, 2, 3, 4), c(10, 20, 30, 40), 1.5, method = "polynomial", order = 3,
                extrapolate = TRUE),
    "linear-only"
  )
})

test_that("interpolate_2d() rejects a y matrix whose dimensions don't match x1 by x2", {
  expect_error(
    interpolate_2d(c(1, 2, 3), c(1, 2), matrix(1:4, nrow = 2), 1.5, 1.5),
    "3 x 2"
  )
})

test_that("interpolate_2d() reproduces a known bilinear value on an identity grid", {
  y <- diag(3)
  out <- interpolate_2d(c(1, 2, 3), c(1, 2, 3), y, 1.5, 1.5)
  expect_equal(out, 0.5, tolerance = 1e-12)
})

# The "regression" toolbox group (Task 5). linear_regression() mirrors the C# LinearRegression
# class; stats::lm() is a genuinely independent check (a different implementation entirely), so
# comparing against it here is allowed by corehydro's fixture-provenance rule even though it is
# not itself a pinned oracle value.

test_that("linear_regression() coefficients and standard errors match stats::lm()", {
  set.seed(42)
  x <- cbind(x1 = rnorm(30), x2 = rnorm(30))
  y <- 1.5 + 2 * x[, 1] - 0.7 * x[, 2] + rnorm(30, sd = 0.01)

  fit <- linear_regression(x, y)
  ref <- stats::lm(y ~ x1 + x2, data = as.data.frame(x))

  expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-8)
  expect_equal(unname(fit$standard_errors), unname(coef(summary(ref))[, "Std. Error"]),
               tolerance = 1e-8)
  expect_equal(fit$r_squared, summary(ref)$r.squared, tolerance = 1e-8)
})

test_that("linear_regression() with intercept = FALSE drops the intercept column", {
  x <- cbind(c(1, 2, 3, 4, 5), c(2, 1, 4, 3, 5))
  y <- c(3.1, 4.2, 8.1, 9.2, 13.0)

  with_int <- linear_regression(x, y)
  without_int <- linear_regression(x, y, intercept = FALSE)

  expect_length(coef(with_int), 3L)
  expect_length(coef(without_int), 2L)
  expect_false(without_int$intercept)
})

test_that("linear_regression() preserves the caller's column names when present", {
  x <- cbind(rainfall = c(1, 2, 3, 4, 5), temperature = c(2, 1, 4, 3, 5))
  y <- c(3.1, 4.2, 8.1, 9.2, 13.0)
  fit <- linear_regression(x, y)
  expect_equal(names(coef(fit)), c("(Intercept)", "rainfall", "temperature"))
})

test_that("linear_regression() falls back to synthesized names when x has none", {
  x <- cbind(c(1, 2, 3, 4, 5), c(2, 1, 4, 3, 5))
  y <- c(3.1, 4.2, 8.1, 9.2, 13.0)
  fit <- linear_regression(x, y)
  expect_equal(names(coef(fit)), c("(Intercept)", "x1", "x2"))
})

test_that("linear_regression() rejects a y of the wrong length, naming both lengths", {
  x <- cbind(c(1, 2, 3, 4, 5), c(2, 1, 4, 3, 5))
  expect_error(linear_regression(x, c(1, 2, 3)), "5.*3|3.*5")
})

test_that("linear_regression() recovers an exact linear combination with R-squared 1", {
  x1 <- c(1, 2, 3, 4, 5, 6, 7, 8)
  x2 <- c(2, 1, 4, 3, 6, 5, 8, 7)
  y <- 3 + 2 * x1 - x2
  fit <- linear_regression(cbind(x1, x2), y)
  expect_equal(unname(coef(fit)), c(3, 2, -1), tolerance = 1e-9)
  expect_equal(fit$r_squared, 1, tolerance = 1e-12)
})

test_that("predict.corehydro_lm() reproduces the fitted values at the training predictors", {
  x1 <- c(1, 2, 3, 4, 5, 6, 7, 8)
  x2 <- c(2, 1, 4, 3, 6, 5, 8, 7)
  y <- 3 + 2 * x1 - x2
  fit <- linear_regression(cbind(x1, x2), y)
  expect_equal(predict(fit, cbind(x1, x2)), y, tolerance = 1e-9)
})

test_that("predict.corehydro_lm() accepts a bare vector for a single-predictor model", {
  x <- c(1, 2, 3, 4, 5)
  y <- c(3.1, 4.2, 8.1, 9.2, 13.0)
  fit <- linear_regression(x, y)
  out <- predict(fit, c(1, 2, 3))
  expect_length(out, 3L)
})

test_that("predict.corehydro_lm() with interval = TRUE returns a lower/upper/mean matrix", {
  x1 <- c(1, 2, 3, 4, 5, 6, 7, 8)
  x2 <- c(2, 1, 4, 3, 6, 5, 8, 7)
  y <- c(3.1, 6.2, 4.9, 8.3, 6.8, 10.2, 8.9, 12.1)
  fit <- linear_regression(cbind(x1, x2), y)
  out <- predict(fit, cbind(x1, x2), interval = TRUE)
  expect_equal(colnames(out), c("lower", "upper", "mean"))
  expect_equal(nrow(out), 8L)
  expect_true(all(out[, "lower"] <= out[, "mean"] & out[, "mean"] <= out[, "upper"]))
})

test_that("linear_regression() prints and summarizes without erroring", {
  x <- cbind(c(1, 2, 3, 4, 5), c(2, 1, 4, 3, 5))
  y <- c(3.1, 4.2, 8.1, 9.2, 13.0)
  fit <- linear_regression(x, y)
  expect_output(print(fit), "corehydro_lm")
  expect_output(summary(fit), "standard errors")
})

test_that("vcov.corehydro_lm() returns a named square matrix matching the parameter count", {
  x <- cbind(c(1, 2, 3, 4, 5), c(2, 1, 4, 3, 5))
  y <- c(3.1, 4.2, 8.1, 9.2, 13.0)
  fit <- linear_regression(x, y)
  v <- vcov(fit)
  expect_equal(dim(v), c(3L, 3L))
  expect_equal(rownames(v), names(coef(fit)))
})

test_that("vcov.corehydro_lm() is the coefficient covariance, consistent with standard_errors", {
  # This relationship (sqrt(diag(vcov)) == standard_errors) holds by construction -- the object's
  # own standard_errors are exactly sqrt(diag(vcov)) once vcov is properly scaled by sigma^2 --
  # so it is not an oracle value, just a self-consistency check.
  x <- cbind(c(1, 2, 3, 4, 5), c(2, 1, 4, 3, 5))
  y <- c(3.1, 4.2, 8.1, 9.2, 13.0)
  fit <- linear_regression(x, y)
  v <- vcov(fit)
  expect_equal(unname(sqrt(diag(v))), unname(fit$standard_errors), tolerance = 1e-12)
})

test_that("vcov.corehydro_lm() agrees with stats::lm()'s vcov() on the same data", {
  set.seed(42)
  x <- cbind(x1 = rnorm(30), x2 = rnorm(30))
  y <- 1.5 + 2 * x[, 1] - 0.7 * x[, 2] + rnorm(30, sd = 0.01)

  fit <- linear_regression(x, y)
  ref <- stats::lm(y ~ x1 + x2, data = as.data.frame(x))

  v <- vcov(fit)
  ref_v <- stats::vcov(ref)
  expect_equal(unname(v), unname(as.matrix(ref_v)), tolerance = 1e-8)
})

# The "sampling" and "probability" toolbox groups (Task 6). The oracle-pinned values (from
# Test_SobolSequence.cs, Test_Stratification.cs, and Test_Probability.cs) are validated
# cross-language by fixtures/toolbox/{sampling,joint_probability}.json; these tests exercise the
# R-facing API (argument checks, shapes, self-consistency) instead of re-pinning literals.

test_that("sobol_sequence() returns an n by dimension matrix with every value in [0, 1)", {
  m <- sobol_sequence(8, dimension = 3)
  expect_equal(dim(m), c(8L, 3L))
  expect_true(all(m >= 0 & m < 1))
})

test_that("sobol_sequence() needs no direction-numbers file at dimension 1", {
  m <- sobol_sequence(1)
  expect_equal(m[1, 1], 0.5, tolerance = 0)
})

test_that("sobol_sequence() skip moves the stream: point 1 with skip = k equals point k+1", {
  seq5 <- sobol_sequence(5, dimension = 2)
  skipped <- sobol_sequence(1, dimension = 2, skip = 4)
  expect_equal(skipped[1, ], seq5[5, ], tolerance = 0)
})

test_that("sobol_sequence() rejects a non-positive n or dimension, naming the argument", {
  expect_error(sobol_sequence(0), "n")
  expect_error(sobol_sequence(5, dimension = 0), "dimension")
})

test_that("stratify() returns bins whose weights sum to the axis length", {
  s <- stratify(0, 1, bins = 10)
  expect_equal(nrow(s), 10L)
  expect_equal(names(s), c("lower", "upper", "midpoint", "weight"))
  expect_equal(sum(s$weight), 1, tolerance = 1e-12)
})

test_that("stratify() with probability = TRUE returns zero rows", {
  s <- stratify(0, 1, bins = 10, probability = TRUE)
  expect_equal(nrow(s), 0L)
})

test_that("stratify() rejects fewer than 2 bins, naming the argument", {
  expect_error(stratify(0, 1, bins = 1), "bins")
})

test_that("stratify() rejects lower >= upper, naming both arguments", {
  expect_error(stratify(1, 1, bins = 4), "lower.*upper")
  expect_error(stratify(2, 1, bins = 4), "lower.*upper")
})

test_that("joint_probability() independent multiplies and positive takes the minimum", {
  expect_equal(joint_probability(c(0.5, 0.5)), 0.25, tolerance = 1e-12)
  expect_equal(joint_probability(c(0.5, 0.5), dependency = "positive"), 0.5, tolerance = 1e-12)
  expect_equal(joint_probability(c(0.5, 0.5), dependency = "negative"), 0, tolerance = 1e-12)
})

test_that("joint_probability() with indicators and a correlation matrix reaches the HPCM path", {
  p <- c(0.25, 0.35, 0.5, 0.5)
  ind <- c(1, 1, 1, 1)
  corr <- diag(4)
  out <- joint_probability(p, dependency = "correlation", indicators = ind, correlation = corr)
  expect_equal(out, 0.021875, tolerance = 1e-6)
})

test_that("joint_probability() requires indicators when correlation is given", {
  expect_error(joint_probability(c(0.5, 0.5), correlation = diag(2)), "indicators")
})

test_that("joint_probability() rejects a correlation matrix of the wrong size, naming the shape", {
  expect_error(
    joint_probability(c(0.5, 0.5, 0.5), indicators = c(1, 1, 1), correlation = diag(2)),
    "3 x 3"
  )
})

test_that("joint_probability() rejects dependency = 'correlation' missing both arguments", {
  expect_error(
    joint_probability(c(0.5, 0.5), dependency = "correlation"),
    "indicators.*correlation"
  )
})

test_that("joint_probability() rejects dependency = 'correlation' missing just the matrix", {
  expect_error(
    joint_probability(c(0.5, 0.5), dependency = "correlation", indicators = c(1, 1)),
    "correlation"
  )
})

# The "link" and "trend" toolbox groups (Task 7). The oracle-pinned values (from
# Test_LinkFunctions.cs and friends, and the ten Test_*TrendTests.cs files) are validated
# cross-language by fixtures/toolbox/{link_functions,trend_functions}.json; these tests exercise
# the R-facing API (construction, round-trips, argument checks) instead of re-pinning literals.

test_that("link_function() round-trips for every type", {
  cases <- list(
    list(type = "Identity", params = list()),
    list(type = "Log", params = list()),
    list(type = "Logit", params = list()),
    list(type = "Probit", params = list()),
    list(type = "ComplementaryLogLog", params = list()),
    list(type = "FisherZ", params = list()),
    list(type = "YeoJohnson", params = list(lambda = 0.5)),
    list(type = "ASinH", params = list(gamma0 = 0.5, scale = 0.3)),
    list(type = "SES", params = list(a = 1.0)),
    list(type = "LogSES", params = list(sigma0 = 7.5)),
    list(type = "LogASinH", params = list(sigma0 = 10.0, log_scale = 0.25))
  )
  domain <- c(Log = 2.0, Logit = 0.3, Probit = 0.4, ComplementaryLogLog = 0.4,
              FisherZ = 0.3, LogSES = 2.0, LogASinH = 15.0)
  for (case in cases) {
    l <- do.call(link_function, c(list(type = case$type), case$params))
    x <- if (case$type %in% names(domain)) domain[[case$type]] else 1.0
    eta <- link(l, x)
    back <- link_inverse(l, eta)
    expect_equal(back, x, tolerance = 1e-10, info = case$type)
    d <- link_derivative(l, x)
    expect_true(is.finite(d), info = case$type)
  }
})

test_that("link_function('Centered') round-trips with a non-Identity inner", {
  inner <- link_function("ASinH", gamma0 = 0, scale = 1)
  l <- link_function("Centered", mu0 = 100, scale = 20, inner = inner)
  eta <- link(l, 120)
  back <- link_inverse(l, eta)
  expect_equal(back, 120, tolerance = 1e-8)
})

test_that("link_function() rejects an unknown type, naming it", {
  expect_error(link_function("Nope"), "Nope")
})

test_that("link_function() matches `type` case-insensitively and normalizes it", {
  l <- link_function("log")
  expect_identical(l$type, "Log")
  l2 <- link_function("YEOJOHNSON", lambda = 0.5)
  expect_identical(l2$type, "YeoJohnson")
})

test_that("link_function('Centered') requires `inner`", {
  expect_error(link_function("Centered", mu0 = 0), "inner")
})

test_that("link_function() rejects `inner` on a non-Centered type", {
  expect_error(link_function("Log", inner = link_function("Identity")), "inner")
})

test_that("link_function() requires `inner` to be a corehydro_link", {
  expect_error(link_function("Centered", mu0 = 0, inner = "not a link"), "corehydro_link")
})

test_that("link() rejects a non-corehydro_link object", {
  expect_error(link("not a link", 1), "corehydro_link")
})

test_that("link() rejects empty input", {
  l <- link_function("Identity")
  expect_error(link(l, numeric(0)), "numeric")
})

test_that("link_names() lists twelve types", {
  names <- link_names()
  expect_length(names, 12L)
  expect_true("Centered" %in% names && "YeoJohnson" %in% names)
})

test_that("link_names() and link_function() cannot drift: every listed type constructs", {
  for (type in link_names()) {
    params <- switch(type,
      YeoJohnson = list(lambda = 0.5), ASinH = list(), SES = list(a = 1.0),
      LogSES = list(), LogASinH = list(),
      Centered = list(inner = link_function("Identity")),
      list()
    )
    l <- do.call(link_function, c(list(type = type), params))
    expect_identical(l$type, type, info = type)
  }
})

test_that("print.corehydro_link() reports the type", {
  expect_output(print(link_function("Log")), "Log")
})

test_that("trend_predict() matches the transcribed Linear trend oracle", {
  tr <- trend("location", "Linear", start_index = 1950, values = c(100, 0.5))
  out <- trend_predict(tr, c(1951, 1961, 1941))
  expect_equal(out, c(100, 105, 95), tolerance = 1e-10)
})

test_that("trend_predict() with no explicit values uses the zero-valued class default", {
  tr <- trend("location", "Constant")
  out <- trend_predict(tr, c(1, 2, 3))
  expect_equal(out, c(0, 0, 0))
})

test_that("trend_parameters() returns named values", {
  tr <- trend("location", "Linear", values = c(10, 2))
  params <- trend_parameters(tr)
  expect_equal(unname(params), c(10, 2))
  expect_length(params, 2L)
})

test_that("trend_predict() rejects a non-corehydro_trend object", {
  expect_error(trend_predict("not a trend", 1), "corehydro_trend")
})

test_that("trend_predict() rejects fractional indices", {
  tr <- trend("location", "Constant")
  expect_error(trend_predict(tr, 1.5), "whole numbers")
})

test_that("trend_names() lists eleven types", {
  names <- trend_names()
  expect_length(names, 11L)
  expect_true("GeneralLinear" %in% names)
})

test_that("trend_names() and trend() cannot drift: every listed type constructs", {
  for (type in trend_names()) {
    tr <- trend("location", type)
    expect_identical(tr$type, type, info = type)
  }
})

# The "linalg" toolbox group (P2 "math extras" Task 9).

test_that("qr_decomposition() reproduces a NON-symmetric a", {
  a <- matrix(c(1, 2, 3, 0, 1, 4, 5, 6, 0), nrow = 3, byrow = TRUE)
  qr <- qr_decomposition(a)
  expect_equal(dim(qr$q), c(3L, 3L))
  expect_equal(dim(qr$r), c(3L, 3L))
  expect_equal(qr$q %*% qr$r, a, tolerance = 1e-10)
})

test_that("qr_solve() reproduces the Test_QRDecomposition square-system expected vector", {
  # Test_QRDecomposition.cs's Test_SolveVector: A = [[1,1,1],[0,2,5],[2,5,-1]], b = [6,-4,27];
  # the real system solves to x = [5, 3, -2].
  a <- matrix(c(1, 1, 1, 0, 2, 5, 2, 5, -1), nrow = 3, byrow = TRUE)
  b <- c(6, -4, 27)
  x <- qr_solve(a, b)
  expect_equal(x, c(5, 3, -2), tolerance = 1e-9)
  expect_equal(as.numeric(a %*% x), b, tolerance = 1e-9)
})

test_that("qr_solve() matrix right-hand side matches the vector right-hand side", {
  a <- matrix(c(1, 2, 3, 0, 1, 4, 5, 6, 0), nrow = 3, byrow = TRUE)
  x_vec <- qr_solve(a, c(1, 2, 3))
  x_mat <- qr_solve(a, matrix(c(1, 2, 3), ncol = 1))
  expect_equal(dim(x_mat), c(3L, 1L))
  expect_equal(x_mat[, 1], x_vec, tolerance = 1e-10)
})

test_that("qr_solve() on an underdetermined system leaves the trailing unknown at zero", {
  a <- matrix(c(2, 3, 5, 1, 1, 0, 2, 3, 0, 1, 4, 2), nrow = 3, byrow = TRUE)
  b <- c(1, 2, 3)
  x <- qr_solve(a, b)
  expect_length(x, 4L)
  expect_equal(x[4], 0)
  expect_equal(as.numeric(a %*% x), b, tolerance = 1e-8)
})

test_that("qr_solve() rejects a mismatched b length", {
  a <- diag(2)
  expect_error(qr_solve(a, c(1, 2, 3)), "rows")
})

test_that("gauss_jordan() reproduces true_IA exactly", {
  # Test_GaussJordanElimination.cs's Test_GaussJordanElim: A = [[1,3,3],[1,4,3],[1,3,4]],
  # true_IA = [[7,-3,-3],[-1,1,0],[-1,0,1]].
  a <- matrix(c(1, 3, 3, 1, 4, 3, 1, 3, 4), nrow = 3, byrow = TRUE)
  result <- gauss_jordan(a)
  expect_equal(result$inverse, matrix(c(7, -3, -3, -1, 1, 0, -1, 0, 1), nrow = 3, byrow = TRUE))
  expect_equal(dim(result$solution), c(3L, 0L))
})

test_that("gauss_jordan() solution solves a %*% x = b", {
  a <- matrix(c(1, 3, 3, 1, 4, 3, 1, 3, 4), nrow = 3, byrow = TRUE)
  b <- matrix(c(1, 0, 0), ncol = 1)
  result <- gauss_jordan(a, b)
  expect_equal(a %*% result$solution, b, tolerance = 1e-10)
})

test_that("gauss_jordan() rejects a non-square a", {
  expect_error(gauss_jordan(matrix(1:6, nrow = 2)), "square")
})

# The "special" toolbox group (P2 "math extras" Task 10).

test_that("debye() reproduces the Test_Debye literal at x = 1.0", {
  # Test_SpecialFunctions.cs's Test_Debye: testX[1] = 1.0, testValid[1] = 0.6744156.
  expect_equal(debye(1.0), 0.6744156, tolerance = 1e-4)
})

test_that("debye() is vectorized over x, reproducing the whole Test_Debye array", {
  x <- c(0.1, 1.0, 2.8, 9.5, 10, 15, 25, 100)
  valid <- c(0.9629999, 0.6744156, 0.3099952, 0.02241066, 0.01929577, 0.005771263,
             0.001246836, 1.948182e-05)
  expect_equal(debye(x), valid, tolerance = 1e-4)
})

test_that("debye() rejects a negative x", {
  expect_error(debye(-1))
})

test_that("polynomial_eval() variant = 'standard' reproduces Test_Polynomial", {
  # Test_PolynomialRev.cs's Test_Polynomial: coeffs = c(3, 5, 7), x = 4, valid = 135.
  expect_equal(polynomial_eval(c(3, 5, 7), 4), 135)
  expect_equal(polynomial_eval(c(3, 5, 7), 4, variant = "standard"), 135)
})

test_that("polynomial_eval() variant = 'reverse' reproduces Test_PolynomialRev, with and without n", {
  expect_equal(polynomial_eval(c(3, 5, 7), 4, variant = "reverse"), 75)
  expect_equal(polynomial_eval(c(3, 5, 7), 4, variant = "reverse", n = 1), 17)
})

test_that("polynomial_eval() variant = 'reverse_unit' reproduces Test_PolynomialRev_1", {
  expect_equal(polynomial_eval(c(3, 5, 7), 4, variant = "reverse_unit"), 139)
})

test_that("polynomial_eval() is vectorized over x", {
  expect_equal(polynomial_eval(c(3, 5, 7), c(0, 4)), c(3, 135))
})

test_that("polynomial_eval() rejects n with a non-'reverse' variant", {
  expect_error(polynomial_eval(c(3, 5, 7), 4, variant = "standard", n = 1), "reverse")
  expect_error(polynomial_eval(c(3, 5, 7), 4, variant = "reverse_unit", n = 1), "reverse")
})

# The "functions" toolbox group (P2 "math extras" Task 11): the two non-tabular
# IUnivariateFunction implementations. Literals transcribed from Test_Functions.cs.

test_that("univariate_function() reproduces Test_Linear_Function", {
  expect_equal(univariate_function("linear", c(0, 1, 0), 6), 6, tolerance = 1e-6)
  expect_equal(univariate_function("linear", c(-2, 5, 3), 6), (5 * 6) + -2, tolerance = 1e-6)
  expect_equal(
    univariate_function("linear", c(-2, 5, 3), 6, confidence_level = 0.75),
    30.0234692505882, tolerance = 1e-6
  )
})

test_that("univariate_function() reproduces Test_Linear_Function_Inverse", {
  y <- univariate_function("linear", c(10, 0.5, 20), 400)
  expect_equal(univariate_function("linear", c(10, 0.5, 20), y, inverse = TRUE), 400,
               tolerance = 1e-6)
  yy <- univariate_function("linear", c(10, 0.5, 20), 400, confidence_level = 0.75)
  expect_equal(
    univariate_function("linear", c(10, 0.5, 20), yy, inverse = TRUE, confidence_level = 0.75),
    400, tolerance = 1e-6
  )
})

test_that("univariate_function() reproduces Test_Power_Function", {
  expect_equal(univariate_function("power", c(1, 1.5, 0, 0), 6), 1 * (6 - 0)^1.5, tolerance = 1e-6)
  expect_equal(univariate_function("power", c(5, 2, 0, 3), 6), 5 * (6 - 0)^2, tolerance = 1e-6)
  expect_equal(
    univariate_function("power", c(5, 2, 0, 3), 6, confidence_level = 0.75),
    1361.61408399941, tolerance = 1e-6
  )
})

test_that("univariate_function() reproduces Test_Power_Function_Inverse", {
  y <- univariate_function("power", c(10, 2, 0, 0.1), 400)
  expect_equal(univariate_function("power", c(10, 2, 0, 0.1), y, inverse = TRUE), 400,
               tolerance = 1e-6)
  yy <- univariate_function("power", c(10, 2, 0, 0.1), 400, confidence_level = 0.75)
  expect_equal(
    univariate_function("power", c(10, 2, 0, 0.1), yy, inverse = TRUE, confidence_level = 0.75),
    400, tolerance = 1e-6
  )
})

test_that("univariate_function() reproduces Test_InversePower_Function", {
  valid <- sqrt(6 / 5) + 0
  expect_equal(univariate_function("power", c(5, 2, 0, 0), 6, is_inverse = TRUE), valid,
               tolerance = 1e-6)
  expect_equal(univariate_function("power", c(5, 2, 0, 3), 6, is_inverse = TRUE), valid,
               tolerance = 1e-6)
  expect_equal(
    univariate_function("power", c(5, 2, 0, 3), 6, is_inverse = TRUE, confidence_level = 0.75),
    0.398290417772997, tolerance = 1e-6
  )
})

test_that("univariate_function() reproduces Test_InversePower_Function_Inverse", {
  y <- univariate_function("power", c(10, 2, 0, 0.1), 6, is_inverse = TRUE)
  expect_equal(
    univariate_function("power", c(10, 2, 0, 0.1), y, inverse = TRUE, is_inverse = TRUE),
    6, tolerance = 1e-6
  )
})

test_that("univariate_function() rejects is_inverse for type = 'linear'", {
  expect_error(univariate_function("linear", c(0, 1, 0), 6, is_inverse = TRUE), "power")
})

test_that("univariate_function() rejects an unknown type", {
  expect_error(univariate_function("quadratic", c(1, 1), 1), "unknown function type")
})

# The "network" toolbox group (P3 optimizers Task 10): Dijkstra shortest paths over an edge
# list. The oracle values live in fixtures/toolbox/network.json; the assertions below are the
# same C# literals scraped from Test_Numerics/Mathematics/Optimization/Dynamic/DijkstraTesting.cs
# and cover the binding surface (column order, 0-based indices, the unreachable row, defaults,
# argument validation) rather than re-pinning the solver.

test_that("shortest_path() reproduces SimpleEdgeGraphCost", {
  sp <- shortest_path(
    from = c(0, 0, 1, 1, 2, 4),
    to = c(1, 2, 2, 3, 3, 0),
    weight = c(2, 4, 1, 7, 3, 1),
    destinations = 3,
    edge_index = c(0, 2, 2, 3, 4, 5),
    node_count = 6
  )
  expect_s3_class(sp, "data.frame")
  expect_identical(names(sp), c("next_node", "edge_index", "cost"))
  expect_identical(nrow(sp), 6L)
  expect_identical(sp$cost, c(6, 4, 3, 0, 7, Inf))
})

test_that("shortest_path() reproduces SimpleNetworkRouting's next-node column", {
  from <- c(0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9)
  to <- c(5, 1, 0, 2, 6, 7, 1, 3, 7, 2, 8, 4, 3, 9, 0, 6, 5, 1, 7, 6, 1, 2, 8, 7, 3, 9, 8, 4)
  w <- c(1, 30, 30, 1, 15, 2, 1, 5, 5, 5, 2, 1, 1, 30, 1, 3, 3, 15, 1, 1, 2, 5, 1, 1, 2, 2, 2, 30)
  idx <- c(0, 1, 1, 2, 3, 4, 2, 5, 6, 5, 7, 8, 8, 9, 0, 10, 10, 3, 11, 11, 4, 6, 12, 12, 7, 13, 13, 9)
  sp <- shortest_path(from, to, w, destinations = 9, edge_index = idx)
  expect_identical(nrow(sp), 10L)
  expect_identical(sp$next_node, c(5L, 7L, 1L, 8L, 3L, 6L, 7L, 8L, 9L, 9L))
  expect_identical(sp$cost, c(8, 5, 6, 4, 5, 7, 4, 3, 2, 0))
})

test_that("shortest_path() takes several destinations", {
  sp <- shortest_path(
    from = c(0, 1, 1, 2, 1),
    to = c(1, 0, 2, 1, 3),
    weight = c(1, 3, 1, 2, 3),
    destinations = c(0, 3),
    edge_index = c(0, 1, 2, 3, 4),
    node_count = 4
  )
  expect_identical(sp$next_node[[2]], 0L)
  expect_identical(sp$cost[[2]], 3)
  expect_identical(sp$next_node[[3]], 1L)
  expect_identical(sp$cost[[3]], 5)
})

test_that("shortest_path() marks an unreachable node", {
  sp <- shortest_path(
    from = c(0, 1, 2),
    to = c(1, 0, 3),
    weight = c(1, 3, 1),
    destinations = 0,
    edge_index = c(0, 1, 2),
    node_count = 4
  )
  expect_identical(sp$cost[[3]], Inf)
  expect_identical(sp$next_node[[3]], -1L)
  expect_identical(sp$edge_index[[3]], -1L)
})

test_that("shortest_path() defaults edge_index to the edge position", {
  sp <- shortest_path(
    from = c(0, 1, 1, 2),
    to = c(1, 0, 2, 0),
    weight = c(1, 4, 1, 10),
    destinations = c(0, 2)
  )
  expect_identical(nrow(sp), 3L)
  expect_identical(sp$next_node, c(0L, 2L, 2L))
  expect_identical(sp$cost, c(0, 1, 0))
  expect_identical(sp$edge_index, c(-1L, 2L, -1L))
})

test_that("shortest_path() validates its arguments", {
  expect_error(shortest_path(c(0, 1), c(1, 2), c(1), destinations = 0), "same length")
  expect_error(shortest_path(c(0, 1), c(1, 2), c(1, 1), destinations = numeric(0)),
               "at least one destination")
  expect_error(shortest_path(c(0, 1), c(1, 2), c(1, 1), destinations = 0.5),
               "whole, non-negative")
  expect_error(shortest_path(c(0, 1), c(1, 2), c(1, 1), destinations = 0,
                             edge_index = c(0, 1, 2)), "same length")
  expect_error(shortest_path(numeric(0), numeric(0), numeric(0), destinations = 0), "at least one")
  expect_error(shortest_path(c(0, 1), c(1, 2), c(1, 1), destinations = 7), "out of range")
})

# A `node_count` too small for the graph used to reach the solver, which answered it with an
# out-of-bounds write: a quietly wrong routing table here and a crash in corehydropy. C# raises
# IndexOutOfRangeException for the same call (measured; see dijkstra.hpp note 9), so the solver
# now throws too, and the wrapper rejects it up front with a message naming the argument.
test_that("shortest_path() rejects a node_count too small for the graph", {
  expect_error(
    shortest_path(from = c(0, 1), to = c(5, 2), weight = c(1, 1), destinations = 0,
                  node_count = 2),
    "`node_count` must be at least 6"
  )
  expect_error(
    shortest_path(from = c(0, 1), to = c(1, 2), weight = c(1, 1), destinations = 0,
                  node_count = 0),
    "must be a single positive number"
  )
  # The boundary itself is fine, and so is anything above it.
  expect_equal(
    nrow(shortest_path(c(0, 1), c(1, 2), c(1, 1), destinations = 0, node_count = 3)), 3L
  )
  expect_equal(
    nrow(shortest_path(c(0, 1), c(1, 2), c(1, 1), destinations = 0, node_count = 5)), 5L
  )
})

# The "hypothesis" toolbox group (P4 Task 3): the twelve ported hypothesis tests over
# `numerics/data/hypothesis_tests.hpp`. The oracle values live in fixtures/toolbox/hypothesis.json;
# the assertions below are the same C# literals scraped from
# Test_Numerics/Data/Statistics/Test_HypothesisTests.cs and cover the binding surface (return
# shape, the `index`/1-based default, argument validation) rather than re-pinning the tests.

harricana69 <- c(
  122, 244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331, 225, 174, 98.8,
  149, 238, 262, 132, 235, 216, 240, 230, 192, 195, 172, 173, 172, 153, 142, 317, 161,
  201, 204, 194, 164, 183, 161, 167, 179, 185, 117, 192, 337, 125, 166, 99.1, 202, 230,
  158, 262, 154, 164, 182, 164, 183, 171, 250, 184, 205, 237, 177, 239, 187, 180, 173,
  174
)

test_that("hypothesis_test() reproduces Test_MannKendall", {
  r <- hypothesis_test(harricana69, method = "mann_kendall")
  expect_equal(unname(r[["p_value"]]), 0.7757, tolerance = 1e-4)
})

test_that("hypothesis_test() method = 'f_models' returns a length-2 named result", {
  r <- hypothesis_test(0, method = "f_models", sse_restricted = 1224.32, sse_full = 720.27,
                       df_restricted = 49, df_full = 48)
  expect_identical(names(r), c("f_statistic", "p_value"))
  expect_equal(unname(r[["f_statistic"]]), 33.5899, tolerance = 1e-3)
  expect_equal(unname(r[["p_value"]]), 0, tolerance = 1e-6)
})

test_that("hypothesis_test() method = 'linear_trend' defaults index to seq_along(x)", {
  time <- 1:100
  x <- cos(time)
  r1 <- hypothesis_test(x, method = "linear_trend")
  r2 <- hypothesis_test(x, method = "linear_trend", index = time)
  expect_equal(unname(r1[["p_value"]]), unname(r2[["p_value"]]))
  expect_equal(unname(r1[["p_value"]]), 0.9092, tolerance = 1e-4)
})

test_that("hypothesis_test() method = 'unimodality' reproduces Test_UnimodalityTest", {
  # P5: un-gated by the GaussianMixtureModel port. The two C# tolerances differ by five orders of
  # magnitude and are reproduced as written.
  unimodal <- c(4.5, 5.2, 5.1, 4.9, 5.0, 5.3, 5.4, 4.8, 4.7, 5.2,
                5.1, 4.6, 5.0, 5.3, 5.1, 4.9, 5.2, 5.0, 4.8, 5.3,
                4.9, 5.1, 5.2, 4.8, 5.0, 5.1, 5.2, 4.7, 5.3, 5.0)
  bimodal <- c(3.8, 3.9, 4.0, 3.9, 4.0, 4.1, 4.0, 3.8, 3.9, 4.0,
               4.1, 4.0, 3.9, 4.0, 4.0, 4.3, 4.4, 4.4, 4.5, 4.4, 4.3,
               4.4, 4.5, 4.4, 4.3, 4.4, 4.4, 4.5, 4.4, 4.3)
  # The C# Assert.AreEqual(expected, actual, delta) tolerances are ABSOLUTE; testthat's
  # `tolerance =` is relative, and 1e-9 RELATIVE on a p-value of 2.6e-5 is 2.6e-14 absolute --
  # far tighter than upstream asserts. Compare on the absolute difference instead.
  expect_lt(abs(unname(hypothesis_test(unimodal, method = "unimodality")[["p_value"]]) -
                  0.4142441), 1e-4)
  expect_lt(abs(unname(hypothesis_test(bimodal, method = "unimodality")[["p_value"]]) -
                  2.55425752131444e-05), 1e-9)
  # The hard-coded seed makes repeated calls identical, and the sample-size guard is upstream's.
  expect_identical(hypothesis_test(unimodal, method = "unimodality"),
                   hypothesis_test(unimodal, method = "unimodality"))
  expect_error(hypothesis_test(rep(1, 9), method = "unimodality"),
               "greater than or equal to 10")
})

test_that("hypothesis_test() validates its arguments", {
  expect_error(hypothesis_test(c(1, 2, 3), method = "equal_variance_t"),
               "`y` is required")
  expect_error(
    hypothesis_test(0, method = "f_models", sse_restricted = 1224.32, df_restricted = 49,
                    df_full = 48),
    "sse_full"
  )
  expect_error(hypothesis_test(c(1, 2, 3), method = "not_a_method"), "must be one of")
})

test_that("hypothesis_test() method = 'f_models' is callable without `x` (C1)", {
  # C1 (P4 whole-branch review): `x` is documented as ignored for "f_models", and corehydropy
  # already defaulted it to None; R's default was missing (only worked by accident, via lazy
  # evaluation never forcing an unused argument), so a call omitting `x` entirely used to work
  # in R but raise in Python. `x` now defaults to NULL in both.
  r <- hypothesis_test(method = "f_models", sse_restricted = 1224.32, sse_full = 720.27,
                       df_restricted = 49, df_full = 48)
  expect_equal(unname(r[["f_statistic"]]), 33.5899, tolerance = 1e-3)
})

test_that("hypothesis_test() rejects a non-NULL `y` for a one-sample method (C2)", {
  # C2 (P4 whole-branch review): a one-sample method used to silently DISCARD `y` --
  # hypothesis_test(a, b) at the default method equalled hypothesis_test(a).
  a <- c(4, 5, 5, 6, 9, 12, 13, 14, 14, 19, 22, 24, 25)
  b <- c(1, 2, 3)
  expect_error(hypothesis_test(a, b, method = "jarque_bera"),
               "`y` is not used by method \"jarque_bera\"; leave it NULL", fixed = TRUE)
})

# The "paired_data" toolbox group (P4 Task 10): OrderedPairedData/UncertainOrderedPairedData/
# LineSimplification, plus TabularFunction's tabular/tabular_inverse arm on the "functions"
# group. Oracle values are C# test literals transcribed verbatim into
# core/tests/test_ordered_paired_data.cpp / test_uncertain_paired_data.cpp; see
# fixtures/toolbox/paired_data.json for the full pinned set.

test_that("curve_interpolate() reproduces Test_Lin and Test_LogLin", {
  x <- c(50, 100, 150, 200, 250)
  y <- c(100, 200, 300, 400, 500)
  expect_equal(curve_interpolate(x, y, xout = 75), 150.0, tolerance = 1e-6)
  expect_equal(curve_interpolate(x, y, xout = 75, x_transform = "logarithmic"),
              158.496250072116, tolerance = 1e-6)
})

test_that("curve_area() reproduces Test_TrapezoidalArea (dataset 1, area under y)", {
  ctor_x <- c(230408, 288010, 345611, 403213, 460815, 518417, 576019, 633612,
             691223, 748825, 806427, 864029, 921631, 1036834, 1152038)
  ctor_y <- c(1519.7, 1520.5, 1520.9, 1521.7, 1523.5, 1525.9, 1528.4, 1530.9,
             1533.2, 1534.7, 1535.9, 1538, 1541.3, 1547.7, 1552.7)
  expect_equal(curve_area(ctor_x, ctor_y), 1413175623, tolerance = 1)
})

test_that("curve_simplify() reproduces the three simplification algorithms on the sin curve", {
  x <- c(0, 1.57, 3.14, 4.71, 6.28)
  y <- c(0, 1, 0, -1, 0)
  rdp <- curve_simplify(x, y, method = "rdp", tolerance = 0.01, strict_y = FALSE, order_y = "none")
  vis <- curve_simplify(x, y, method = "visvalingam", num_to_keep = 4, strict_y = FALSE,
                        order_y = "none")
  expect_equal(nrow(rdp), 4L)
  expect_equal(nrow(vis), 4L)
  expect_equal(rdp$y, c(0, 1, -1, 0), tolerance = 1e-6)
  expect_equal(vis$y, c(0, 1, -1, 0), tolerance = 1e-6)

  # LangSimplify never force-keeps the trailing point -- verified directly against the real C#
  # library (see ordered_paired_data.hpp's sixth transcription note): the correct result here is
  # THREE points, dropping (6.28, 0), not the four upstream's own (weakly-asserted) test claims.
  lang <- curve_simplify(x, y, method = "lang", tolerance = 0.01, look_ahead = 2,
                         strict_y = FALSE, order_y = "none")
  expect_equal(nrow(lang), 3L)
  expect_equal(lang$x, c(0, 1.57, 4.71), tolerance = 1e-6)
  expect_equal(lang$y, c(0, 1, -1), tolerance = 1e-6)
})

test_that("uncertain_curve_sample() reproduces Test_Curve_Sample_Probability", {
  x <- c(1, 2, 3, 5)
  d <- list(distribution("Triangular", c(1, 2, 3)), distribution("Triangular", c(2, 4, 5)),
           distribution("Triangular", c(6, 8, 12)), distribution("Triangular", c(13, 19, 20)))
  r <- uncertain_curve_sample(x, d, probability = 0.5)
  expect_equal(r$x, x)
  expect_equal(r$y, c(2, 3.732051, 8.535898, 17.58258), tolerance = 1e-5)
})

test_that("uncertain_curve_sample() recycles a single distribution across x", {
  x <- c(1, 2, 3)
  r <- uncertain_curve_sample(x, distribution("Triangular", c(1, 2, 3)), probability = 0.5)
  expect_equal(nrow(r), 3L)
})

test_that("uncertain_curve_sample() rejects probability outside [0, 1] instead of silently clamping (C3)", {
  # C3 (P4 whole-branch review): the core clamp (a faithful port of C# CurveSample(double)) is
  # untouched -- probability = 50 silently returned the 100% quantile core-side. This host-layer
  # check is what actually rejects the mistake.
  x <- c(1, 2, 3)
  d <- distribution("Triangular", c(1, 2, 3))
  msg <- "`probability` must be a single number in \\[0, 1\\]"
  expect_error(uncertain_curve_sample(x, d, probability = 50), msg)
  expect_error(uncertain_curve_sample(x, d, probability = -3), msg)
})

test_that("tabular_function() reproduces Test_Tabular_Function", {
  x <- c(50, 100, 150, 200, 250)
  d <- lapply(c(100, 200, 300, 400, 500), function(v) distribution("Deterministic", v))
  y <- tabular_function(x, d, at = 50, x_transform = "logarithmic")
  expect_equal(y, 100.0, tolerance = 1e-12)
})

test_that("paired_data verbs validate their arguments identically to corehydropy", {
  x <- c(50, 100, 150, 200, 250)
  y <- c(100, 200, 300, 400, 500)
  expect_error(curve_interpolate(x, y, xout = 75, yout = 100),
               "exactly one of `xout` or `yout`")
  expect_error(curve_interpolate(x, y), "exactly one of `xout` or `yout`")
  expect_error(curve_simplify(x, y, method = "not_a_method", tolerance = 0.01),
               "must be one of")
  d <- list(distribution("Triangular", c(1, 2, 3)), distribution("Triangular", c(2, 4, 5)))
  expect_error(uncertain_curve_sample(c(1, 2, 3), d, probability = 0.5),
               "must have length 1 or length\\(x\\)")
  expect_error(tabular_function(c(1, 2, 3), d, at = 1), "must have length 1 or length\\(x\\)")
})

test_that("curve_simplify() rejects num_to_keep below 2 for visvalingam instead of crashing", {
  # M1 (P4 whole-branch review): num_to_keep of 0 or -1 used to crash both R and Python outright
  # (a bus error / SIGSEGV); 1 silently returned a value read out of bounds. All three are now
  # rejected at the host layer with the identical message corehydropy raises.
  x <- c(0, 1.57, 3.14, 4.71, 6.28)
  y <- c(0, 1, 0, -1, 0)
  msg <- "`num_to_keep` must be a single integer of at least 2"
  expect_error(curve_simplify(x, y, method = "visvalingam", num_to_keep = 0), msg, fixed = TRUE)
  expect_error(curve_simplify(x, y, method = "visvalingam", num_to_keep = -1), msg, fixed = TRUE)
  expect_error(curve_simplify(x, y, method = "visvalingam", num_to_keep = 1), msg, fixed = TRUE)
  # num_to_keep = 2 is the smallest value that succeeds.
  ok <- curve_simplify(x, y, method = "visvalingam", num_to_keep = 2, strict_y = FALSE,
                       order_y = "none")
  expect_equal(nrow(ok), 2L)
})

test_that("\"log\" and \"logarithmic\" are equivalent everywhere a transform argument appears", {
  # M2 (P4 whole-branch review): interpolate()/interpolate_2d() only accepted "log" and
  # curve_interpolate()/tabular_function() only accepted "logarithmic", so a value valid for one
  # host verb was rejected (or, worse, silently PREFIX-MATCHED via match.arg) by the other.
  storage <- c(230408, 288010, 345611, 403213, 460815, 518417, 576019, 633612,
               691223, 748825, 806427, 864029, 921631, 1036834, 1152038)
  elevation <- c(1519.7, 1520.5, 1520.9, 1521.7, 1523.5, 1525.9, 1528.4, 1530.9,
                 1533.2, 1534.7, 1535.9, 1538, 1541.3, 1547.7, 1552.7)
  a <- curve_interpolate(storage, elevation, xout = 500000, x_transform = "log")
  b <- curve_interpolate(storage, elevation, xout = 500000, x_transform = "logarithmic")
  expect_equal(a, b)

  x <- c(1, 2, 3, 4)
  y <- c(10, 20, 30, 40)
  ia <- interpolate(x, y, 2.5, x_transform = "log")
  ib <- interpolate(x, y, 2.5, x_transform = "logarithmic")
  expect_equal(ia, ib)
})
