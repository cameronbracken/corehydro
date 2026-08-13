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
