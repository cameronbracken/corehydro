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

test_that("confint returns posterior credible intervals for a Bayesian fit", {
  f <- fit_bayesian(model_univariate("Normal", peaks),
    sampler = "DEMCz", iterations = 200, output_length = 500, seed = 123
  )
  # fit_bayesian() never sets credible_interval_width, so the fit was built at
  # BayesianAnalysis's own class default, 0.9.
  ci <- confint(f, level = 0.9)
  expect_equal(dim(ci), c(2L, 2L))
  expect_equal(colnames(ci), c("lower", "upper"))
  expect_equal(rownames(ci), names(f$parameters))
  # Requested level matches the fit's own: reused directly from $summary, not rebuilt.
  expect_identical(unname(ci[, "lower"]), f$summary$lower)
  expect_identical(unname(ci[, "upper"]), f$summary$upper)
  expect_true(all(ci[, "lower"] <= f$summary$mean))
  expect_true(all(ci[, "upper"] >= f$summary$mean))
})

test_that("confint on a Bayesian fit rebuilds off a reproduced chain at a different level", {
  f <- fit_bayesian(model_univariate("Normal", peaks),
    sampler = "DEMCz", iterations = 200, output_length = 500, seed = 123
  )
  ci90 <- confint(f, level = 0.9)
  ci99 <- confint(f, level = 0.99)
  expect_false(isTRUE(all.equal(ci90, ci99)))
  # A 99% credible interval is at least as wide as a 90% one at every parameter.
  width90 <- ci90[, "upper"] - ci90[, "lower"]
  width99 <- ci99[, "upper"] - ci99[, "lower"]
  expect_true(all(width99 >= width90))
})

test_that("confint errors for a GMM fit and points at quantile_variance", {
  f <- fit_gmm(model_bulletin17c(peaks))
  expect_error(confint(f), "quantile_variance")
})

test_that("confint() with no level argument defaults to 0.95, triggering a Bayesian rebuild", {
  f <- fit_bayesian(model_univariate("Normal", peaks),
    sampler = "DEMCz", iterations = 200, output_length = 500, seed = 123
  )
  # fit_bayesian() builds the chain at BayesianAnalysis's own class default, 0.9; confint(f)
  # with no `level` asks for 0.95 (base R's confint() convention), which does not match
  # `credible_level`, so it always takes the rebuild path -- this pins the cross-language
  # default-level parity finding (Python's confint() default was 0.9; now matches R's 0.95).
  expect_equal(confint(f), confint(f, level = 0.95))
  expect_false(isTRUE(all.equal(confint(f), confint(f, level = 0.9))))
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

test_that("fit_bayesian returns draws as [iteration, chain, parameter]", {
  f <- fit_bayesian(model_univariate("Normal", peaks),
    sampler = "DEMCz", chains = 4, iterations = 200, output_length = 500, seed = 12345
  )
  expect_s3_class(f, "corehydro_fit")
  expect_equal(length(dim(f$draws)), 3L)
  expect_equal(dim(f$draws)[2], 4L)                       # chains on axis 2
  expect_equal(dim(f$draws)[3], length(f$parameters))     # parameters on axis 3
  expect_equal(dimnames(f$draws)[[3]], names(f$parameters))
  expect_equal(nrow(f$summary), length(f$parameters))
  expect_true(all(c("rhat", "ess", "median") %in% names(f$summary)))
  expect_length(f$acceptance_rates, 4L)
  expect_true(is.finite(f$dic))
})

test_that("a seeded Bayesian fit is reproducible", {
  m <- model_univariate("Normal", peaks)
  a <- fit_bayesian(m, sampler = "DEMCz", iterations = 100, seed = 99)
  b <- fit_bayesian(m, sampler = "DEMCz", iterations = 100, seed = 99)
  expect_identical(a$draws, b$draws)
})

test_that("fit_bayesian rejects a sampler BayesianAnalysis cannot construct", {
  expect_error(fit_bayesian(model_univariate("Normal", peaks), sampler = "HMC"),
    "mcmc_sample", fixed = TRUE
  )
})

test_that("fit_bayesian derives warmup as max(50, iterations %/% 2) when omitted", {
  f <- fit_bayesian(model_univariate("Normal", peaks),
    sampler = "DEMCz", iterations = 200, output_length = 300, seed = 1
  )
  expect_equal(f$warmup, 100L)
  # A small `iterations` with the omitted-warmup class default (1500) would trip the sampler's
  # own `warmup <= iterations / 2` guard -- this fit succeeding at all is part of what this test
  # is checking.
  expect_true(is.finite(f$dic))
})

test_that("an explicit warmup is passed through unchanged", {
  f <- fit_bayesian(model_univariate("Normal", peaks),
    sampler = "DEMCz", iterations = 200, warmup = 77, output_length = 300, seed = 1
  )
  expect_equal(f$warmup, 77L)
})

test_that("the draws permutation still holds when chains and parameter counts differ", {
  # Normal has 2 parameters; BayesianAnalysis requires between 4 and 20 chains, so 6 chains is
  # the smallest valid count that still makes the chain and parameter axes different lengths --
  # a transposition bug moves a value to the wrong cell instead of silently matching by symmetry.
  f <- fit_bayesian(model_univariate("Normal", peaks),
    sampler = "DEMCz", chains = 6, iterations = 150, output_length = 200, seed = 5
  )
  expect_equal(dim(f$draws)[2], 6L)
  expect_equal(dim(f$draws)[3], 2L)
  expect_equal(dim(f$draws)[3], length(f$parameters))
})

test_that("fit_bayesian rejects a knob its sampler does not use", {
  expect_error(
    fit_bayesian(
      model_univariate("Normal", peaks),
      sampler = "DEMCz", iterations = 100, beta = 0.1
    ),
    "beta"
  )
})

test_that("point_estimator defaults to the posterior mean and selects the MAP on request", {
  m <- model_univariate("Normal", peaks)
  default <- fit_bayesian(m, sampler = "DEMCz", iterations = 200, seed = 7)
  mean_estimator <- fit_bayesian(m,
    sampler = "DEMCz", iterations = 200, seed = 7, point_estimator = "PosteriorMean"
  )
  mode_estimator <- fit_bayesian(m,
    sampler = "DEMCz", iterations = 200, seed = 7, point_estimator = "PosteriorMode"
  )
  expect_equal(unname(default$parameters), unname(default$summary$mean))
  expect_identical(default$parameters, mean_estimator$parameters)
  expect_false(isTRUE(all.equal(mean_estimator$parameters, mode_estimator$parameters)))
})

test_that("fit_bayesian rejects an unknown point_estimator", {
  expect_error(
    fit_bayesian(model_univariate("Normal", peaks),
      sampler = "DEMCz", iterations = 100, point_estimator = "bogus"
    ),
    "point estimator"
  )
})

test_that("fit_gmm requires a bulletin17c model", {
  expect_error(fit_gmm(model_univariate("Normal", peaks)), "bulletin17c")
})

test_that("fit_gmm returns the GMM bookkeeping", {
  f <- fit_gmm(model_bulletin17c(peaks))
  expect_equal(f$method, "GMM")
  expect_true(f$gmm_iterations > 0)
  expect_true(is.na(f$j_stat_pval))  # just-identified by construction
  expect_true(quantile_variance(f, 0.01) > 0)
})

test_that("logLik/AIC/BIC are NA (not NaN) for a GMM fit", {
  f <- fit_gmm(model_bulletin17c(peaks))
  ll <- logLik(f)
  expect_true(is.na(as.numeric(ll)))
  expect_false(is.nan(as.numeric(ll)))   # NA, not the runner's raw NaN default
  expect_equal(attr(ll, "df"), length(f$parameters))
  expect_equal(attr(ll, "nobs"), f$nobs)
  expect_true(is.na(AIC(f)))
  expect_true(is.na(BIC(f)))
})

test_that("quantile_variance rejects a non-GMM fit", {
  expect_error(quantile_variance(fit_mle(model_univariate("Normal", peaks)), 0.01), "fit_gmm")
})

test_that("fit_diagnostics returns one value per observation", {
  d <- fit_diagnostics(fit_map(model_univariate("Normal", peaks)))
  expect_length(d$cooks_distance, length(peaks))
})

test_that("fit_diagnostics works off a Bayesian fit and a GMM fit", {
  db <- fit_diagnostics(fit_bayesian(model_univariate("Normal", peaks),
    sampler = "DEMCz", iterations = 100, output_length = 200, seed = 1
  ))
  expect_length(db$pareto_k, length(peaks))
  expect_true(is.finite(db$max_pareto_k))

  dg <- fit_diagnostics(fit_gmm(model_bulletin17c(peaks)))
  expect_length(dg$cooks_distance, length(peaks))
  expect_equal(dim(dg$observation_influence)[1], length(peaks))
})

test_that("fit_diagnostics rejects an MLE fit", {
  expect_error(fit_diagnostics(fit_mle(model_univariate("Normal", peaks))), "fit_map")
})

test_that("summary.corehydro_fit shows rhat/ess for a Bayesian fit and SEs for an optimized one", {
  fb <- fit_bayesian(model_univariate("Normal", peaks),
    sampler = "DEMCz", iterations = 100, output_length = 200, seed = 1
  )
  out_b <- capture.output(summary(fb))
  expect_true(any(grepl("rhat", out_b)))

  fo <- fit_mle(model_univariate("Normal", peaks))
  out_o <- capture.output(summary(fo))
  expect_true(any(grepl("standard errors", out_o)))
})
