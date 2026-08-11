# Behavioural tests for the fit surface. Oracle VALUES live in fixtures/ and are asserted by
# test-fixtures.R; this file asserts argument handling, error messages, object shape, and the
# base-generic integration.

peaks <- c(12500, 15300, 9870, 21000, 18400, 11200, 26800, 14100, 19500, 11600)

test_that("fit_mle returns a corehydro_fit with the common surface", {
  f <- fit_mle(model_univariate("Normal", peaks))
  expect_s3_class(f, "corehydro_fit")
  expect_equal(f$method, "MaximumLikelihood")
  expect_length(f$parameters, 2L)
  expect_named(f$parameters)
  expect_true(f$converged)
  expect_equal(f$status, "Success")
  expect_true(is.finite(f$log_likelihood))
  expect_equal(dim(f$covariance), c(2L, 2L))
  expect_equal(f$nobs, 10L)
})

test_that("the vector convenience path matches the model path exactly", {
  expect_identical(
    fit_mle(peaks, "Normal")$parameters,
    fit_mle(model_univariate("Normal", peaks))$parameters
  )
})

test_that("base generics work off logLik", {
  f <- fit_mle(model_univariate("Normal", peaks))
  expect_equal(as.numeric(coef(f)), as.numeric(f$parameters))
  expect_equal(vcov(f), f$covariance)
  expect_equal(as.numeric(logLik(f)), f$log_likelihood)
  expect_equal(attr(logLik(f), "df"), 2L)
  expect_equal(attr(logLik(f), "nobs"), 10L)
  # The point of carrying df and nobs: base AIC/BIC agree with the core's own values.
  expect_equal(AIC(f), f$aic, tolerance = 1e-10)
  expect_equal(BIC(f), f$bic, tolerance = 1e-10)
})

test_that("hessian = FALSE skips the covariance stack but still fits", {
  f <- fit_mle(model_univariate("Normal", peaks), hessian = FALSE)
  expect_null(f$covariance)
  expect_null(f$standard_errors)
  expect_true(is.finite(f$log_likelihood))
})

# NOTE: the sub-two-parameter NaN covariance guard is NOT testable from R. Every
# single-parameter family in the factory (Rayleigh, Poisson, ChiSquared, Geometric, Bernoulli,
# Deterministic) fails to build as a UnivariateDistributionModel: the constructor calls
# set_default_parameters(), which dynamic_casts to IMaximumLikelihoodEstimation and throws,
# and none of the six implements it. The guard is covered in C++ by test_fit_runner.cpp
# against a test-double ModelBase. Do not try to write an R test for it.

test_that("confint returns profile intervals bracketing the estimate", {
  f <- fit_mle(model_univariate("Normal", peaks), profile = TRUE, profile_bins = 20)
  ci <- confint(f, level = 0.9)
  expect_equal(dim(ci), c(2L, 2L))
  expect_true(all(ci[, 1] <= f$parameters))
  expect_true(all(ci[, 2] >= f$parameters))
})

test_that("confint computes on demand when the fit was built without profile", {
  m <- model_univariate("Normal", peaks)
  expect_equal(
    confint(fit_mle(m), level = 0.9),
    confint(fit_mle(m, profile = TRUE), level = 0.9)
  )
})

test_that("the fit carries a fitted model that simulates identically", {
  f <- fit_mle(model_univariate("Normal", peaks))
  expect_s3_class(f$model, "corehydro_model")
  expect_identical(
    model_simulate(f, n = 25, seed = 7),
    model_simulate(f$model, n = 25, seed = 7)
  )
})

test_that("a fit round-trips through save and load", {
  f <- fit_mle(model_univariate("Normal", peaks))
  path <- tempfile(fileext = ".rds")
  saveRDS(f, path)
  expect_identical(readRDS(path)$parameters, f$parameters)
})

test_that("argument errors name the offending value", {
  m <- model_univariate("Normal", peaks)
  expect_error(fit_mle(m, optimizer = "Simplexx"), "Simplexx")
  expect_error(fit_mle(peaks), "distribution")
})

test_that("fit_input produces valid JSON when settings is empty", {
  skip_if_not_installed("jsonlite")
  m <- model_univariate("Normal", peaks)
  fi <- corehydror:::fit_input(m, NULL, list())
  parsed <- jsonlite::fromJSON(fi$json, simplifyVector = FALSE)
  expect_true(is.list(parsed))
  expect_named(parsed, "model")
  expect_equal(parsed$model$family, "Normal")
})

test_that("covariance keeps the layout C++ built, not a re-flattened transpose", {
  # LogPearsonTypeIII's 3 parameters give a covariance with real (non-zero) off-diagonal
  # entries -- unlike Normal's asymptotically-independent mean/sd -- so a transpose bug would
  # actually move a value to a different cell instead of swapping two zeros. Comparing directly
  # against `raw$covariance` (the matrix ch_fit_run_ itself returns, built element-by-element in
  # corehydror/src/estimation.cpp's `square_or_empty`) rather than a recomputed expectation means
  # no oracle literal is hardcoded here.
  m <- model_univariate("LogPearsonTypeIII", peaks)
  fi <- corehydror:::fit_input(m, NULL,
    list(optimizer = "NelderMead", hessian = TRUE, profile = FALSE, profile_bins = 100L)
  )
  raw <- corehydror:::ch_fit_run_("MaximumLikelihood", fi$json, fi$dataset)
  expect_true(any(abs(raw$covariance[upper.tri(raw$covariance)]) > 0))

  f <- corehydror:::new_fit(raw, fi$spec, fi$dataset, "NelderMead", level = 0.9)
  expect_identical(unname(f$covariance), unname(raw$covariance))
})

test_that("fit_mle(..., profile = TRUE) surfaces a plottable per-parameter grid", {
  f <- fit_mle(model_univariate("Normal", peaks), profile = TRUE, profile_bins = 20)
  expect_type(f$profile, "list")
  expect_named(f$profile, names(f$parameters))
  for (p in f$profile) {
    expect_equal(dim(p), c(20L, 2L))
    expect_equal(colnames(p), c("value", "log_likelihood"))
    # The grid comes from an ordered sequence; a column swap would break monotonicity here.
    expect_true(all(diff(p[, "value"]) >= 0))
  }
})

test_that("profile is absent when the fit was built without profile = TRUE", {
  f <- fit_mle(model_univariate("Normal", peaks))
  expect_null(f$profile)
})
