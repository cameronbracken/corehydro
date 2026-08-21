# Smoke tests for the exported user-facing analysis functions (A10). These prove the
# univariate_analysis / fit_distributions / bulletin17c_analysis wrappers are callable
# end-to-end and return the documented list shapes with finite / monotone invariants. Exact
# oracle values live in fixtures/ and are checked by test-fixtures.R; these assert structure.

# A small annual-peak record reused across the univariate + B17C smoke calls.
smoke_peaks <- function() {
  c(
    12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
    19200, 13800, 25600, 10500, 16900, 21300, 14700, 8200, 23800, 15900
  )
}

test_that("univariate_analysis returns a Bayesian frequency curve", {
  res <- univariate_analysis(
    smoke_peaks(),
    distribution = "Normal", sampler = "DEMCzs",
    iterations = 100, output_length = 400, credible_level = 0.90, seed = 12345
  )
  expect_length(res$parameters, 2)
  expect_true(all(is.finite(res$parameters)))
  expect_true(is.finite(res$aic))
  expect_true(is.finite(res$dic))
  n <- length(res$mode_curve)
  expect_gt(n, 0)
  expect_equal(length(res$mean_curve), n)
  expect_equal(length(res$lower_ci), n)
  expect_equal(length(res$upper_ci), n)
  # The default exceedance ordinates ascend, so the mode (quantile) curve descends.
  expect_true(all(diff(res$mode_curve) <= 1e-6))
  # Each credible band brackets the mode curve.
  expect_true(all(res$lower_ci <= res$mode_curve + 1e-6))
  expect_true(all(res$upper_ci >= res$mode_curve - 1e-6))
})

test_that("fit_distributions ranks the 15 candidates", {
  res <- fit_distributions(smoke_peaks())
  expect_length(res$distribution, 15)
  expect_length(res$aic, 15)
  expect_true(any(res$converged))
  ok <- res$converged
  expect_true(all(is.finite(res$aic[ok])))
  expect_true(all(is.finite(res$bic[ok])))
  expect_true(all(is.finite(res$rmse[ok])))
})

test_that("bulletin17c_analysis returns Cohn-style confidence intervals", {
  res <- bulletin17c_analysis(
    smoke_peaks(),
    uncertainty_method = "MultivariateNormal",
    output_length = 200, seed = 12345, confidence_level = 0.90
  )
  expect_equal(res$confidence_level, 0.90)
  n <- length(res$exceedance_probabilities)
  expect_gt(n, 0)
  expect_equal(length(res$point_estimates), n)
  expect_equal(length(res$lower_ci), n)
  expect_equal(length(res$upper_ci), n)
  expect_true(all(is.finite(res$point_estimates)))
  # Cohn CI brackets are ordered at every level (preserved through monotonicity enforcement).
  expect_true(all(res$lower_ci <= res$upper_ci))
  expect_length(res$parameters, 3) # LP3: location / scale / shape
})

test_that("bulletin17c_analysis accepts the X8/X9 uncertainty methods", {
  # X8/X9 un-gated LinkedMultivariateNormal + BiasCorrectedBootstrap; they now run (X11 wired the
  # binding knob). An unknown method still errors.
  for (um in c("LinkedMultivariateNormal", "BiasCorrectedBootstrap")) {
    res <- bulletin17c_analysis(smoke_peaks(), uncertainty_method = um, output_length = 200,
                                seed = 12345)
    expect_length(res$parameters, 3)
    expect_true(all(res$lower_ci <= res$upper_ci))
  }
  expect_error(bulletin17c_analysis(smoke_peaks(), uncertainty_method = "NotAMethod"))
})

# --- D5: per-family analyses + diagnostics ----------------------------------------------
# Direct user-API smoke calls (the fixture harness covers the numeric oracles); these assert
# the returned list shapes and finite/well-shaped contents in R.

ts_series <- function() {
  c(
    10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1,
    11.7, 12.5, 13.9, 15.2, 14.1, 12.9, 13.6, 15.0, 16.2, 14.8
  )
}

expect_family_shape <- function(res, n_params) {
  expect_length(res$parameters, n_params)
  expect_true(all(is.finite(res$parameters)))
  n <- length(res$mode_curve)
  expect_gt(n, 0)
  expect_equal(length(res$mean_curve), n)
  expect_equal(length(res$lower_ci), n)
  expect_equal(length(res$upper_ci), n)
  expect_true(is.finite(res$aic))
  expect_true(is.finite(res$bic))
}

test_that("mixture_analysis returns a Bayesian frequency curve", {
  res <- mixture_analysis(
    c(520, 580, 610, 650, 700, 730, 760, 800, 850, 880, 910, 950, 990, 1030, 1080,
      5000, 5400, 5800, 6300, 6800),
    c("Normal", "Normal"),
    iterations = 100, output_length = 400, seed = 12345, thinning_interval = 1,
    exceedance_probabilities = c(0.01, 0.1, 0.5, 0.9, 0.99)
  )
  expect_family_shape(res, 6)
  expect_equal(length(res$mode_curve), 5)
})

test_that("competing_risk_analysis returns a Bayesian frequency curve", {
  res <- competing_risk_analysis(
    c(7872, 8624, 5894, 12540, 4322, 17586, 8307, 6009, 13320, 10641, 6301, 8458,
      3545, 8838, 13628, 15105, 11742, 15763, 9117, 4116, 12372, 5902, 6038, 8381, 14452),
    c("Gumbel", "Weibull"),
    iterations = 100, output_length = 400, seed = 12345, thinning_interval = 1,
    exceedance_probabilities = c(0.01, 0.1, 0.5, 0.9, 0.99)
  )
  expect_family_shape(res, 4)
})

test_that("point_process_analysis returns a Bayesian frequency curve", {
  res <- point_process_analysis(
    c(950, 1020, 1130, 980, 1250, 1090, 1430, 1010, 1180, 1620, 970, 1300, 1050, 1550, 1210),
    threshold = 900, total_years = 20,
    iterations = 100, output_length = 400, seed = 12345, thinning_interval = 1,
    exceedance_probabilities = c(0.01, 0.1, 0.5, 0.9, 0.99)
  )
  expect_family_shape(res, 3)
  expect_true(is.finite(res$dic))
})

test_that("ar_analysis returns a posterior forecast curve", {
  res <- ar_analysis(
    ts_series(), order_p = 1, training_time_steps = 15, forecasting_time_steps = 3,
    iterations = 100, output_length = 400, seed = 12345, thinning_interval = 1
  )
  expect_equal(length(res$mode_curve), 23) # 20 observed + 3 forecast
  expect_true(all(is.finite(res$parameters)))
  expect_true(is.finite(res$rmse))
})

test_that("ma_analysis returns a posterior forecast curve", {
  res <- ma_analysis(
    ts_series(), order_q = 1, training_time_steps = 15, forecasting_time_steps = 3,
    iterations = 100, output_length = 400, seed = 12345, thinning_interval = 1
  )
  expect_equal(length(res$mode_curve), 23)
  expect_true(is.finite(res$aic))
})

test_that("arima_analysis returns a posterior forecast curve", {
  res <- arima_analysis(
    ts_series(), order_p = 1, order_d = 1, order_q = 1, training_time_steps = 15,
    forecasting_time_steps = 3, iterations = 100, output_length = 400, seed = 12345,
    thinning_interval = 1
  )
  expect_equal(length(res$mode_curve), 23)
  expect_true(is.finite(res$aic))
})

test_that("arimax_analysis returns a posterior forecast curve", {
  res <- arimax_analysis(
    ts_series(), trend = "Linear", training_time_steps = 15, forecasting_time_steps = 0,
    iterations = 100, output_length = 400, seed = 12345, thinning_interval = 1
  )
  expect_equal(length(res$mode_curve), 20) # fit-only
  expect_true(is.finite(res$bic))
})

test_that("estimation_diagnostics returns leverage / influence / prior influence", {
  d <- estimation_diagnostics(
    smoke_peaks(), "Normal",
    iterations = 100, output_length = 400, seed = 12345, thinning_interval = 1
  )
  # Leverage
  expect_equal(length(d$leverage$leverage), length(smoke_peaks()))
  expect_equal(length(d$leverage$value), length(smoke_peaks()))
  expect_true(is.finite(d$leverage$total_leverage))
  expect_true(all(is.finite(d$leverage$leverage)))
  # Influence (PSIS-LOO)
  expect_equal(d$influence$count, length(smoke_peaks()))
  expect_equal(length(d$influence$pareto_k), length(smoke_peaks()))
  expect_true(is.finite(d$influence$mean_pareto_k))
  expect_true(is.logical(d$influence$is_reliable))
  # Prior influence
  expect_gt(d$prior_influence$count, 0)
  expect_true(is.finite(d$prior_influence$prior_to_data_ratio))
  expect_true(is.logical(d$prior_influence$is_prior_influential))
})
