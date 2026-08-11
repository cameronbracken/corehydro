# Behavioural tests for the model layer: the nine model_*() constructors, the trend and
# parameter helpers, and the model verbs. Numeric oracles live in fixtures/
# (model_parameter_overrides.json, gmm_bulletin17c_censored.json, and the censored estimation
# fixtures); what is checked here is the R surface -- spec assembly, name resolution, argument
# validation, and the equivalences a caller relies on.

peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
           19200, 13800, 25600, 10500, 16900)
series <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5,
            13.9, 15.2, 14.1, 12.9, 13.6, 15.0, 16.2, 14.8)

test_that("model_univariate builds a spec carrying the family and the data", {
  m <- model_univariate("Normal", peaks)
  expect_s3_class(m, "corehydro_model")
  expect_equal(m$spec$type, "univariate_distribution")
  expect_equal(m$spec$family, "Normal")
  expect_equal(m$spec$dataset, "data")
  expect_null(m$spec$data_frame)
  expect_equal(m$dataset, peaks)
})

test_that("an analysis_data frame travels inline instead of as a dataset vector", {
  m <- model_univariate("Normal", analysis_data(peaks))
  expect_null(m$spec$dataset)
  expect_length(m$spec$data_frame$exact, length(peaks))
  expect_length(m$dataset, 0)
})

test_that("a plain vector and an equivalent frame give the identical model", {
  # The vector constructor delegates to the same DataFrame constructor, so an uncensored record
  # must evaluate identically whichever way it arrives.
  expect_identical(
    model_log_likelihood(model_univariate("Normal", peaks)),
    model_log_likelihood(model_univariate("Normal", analysis_data(peaks)))
  )
})

test_that("model_univariate rejects bad data", {
  expect_error(model_univariate("Normal", "not numeric"), "numeric vector or a corehydro_data")
})

test_that("trend widens the parameter vector and marks the model nonstationary", {
  stationary <- model_univariate("Normal", peaks)
  trended <- model_univariate("Normal", peaks, trends = trend("mean", "Linear"))
  expect_length(model_parameters(stationary)$values, 2)
  expect_length(model_parameters(trended)$values, 3)
  expect_equal(trended$spec$trends[[1]]$type, "Linear")
  expect_equal(trended$spec$trends[[1]]$parameter, 0L)
})

test_that("trend resolves a parameter by symbol, plain word, alias, or position", {
  by_word <- model_univariate("Normal", peaks, trends = trend("mean", "Linear"))
  by_alias <- model_univariate("Normal", peaks, trends = trend("mu", "Linear"))
  by_symbol <- model_univariate("Normal", peaks, trends = trend("µ", "Linear"))
  by_pos <- model_univariate("Normal", peaks, trends = trend(1, "Linear"))
  for (m in list(by_alias, by_symbol, by_pos)) {
    expect_equal(m$spec$trends[[1]]$parameter, by_word$spec$trends[[1]]$parameter)
  }
  expect_equal(
    model_univariate("Normal", peaks, trends = trend("sd", "Linear"))$spec$trends[[1]]$parameter,
    1L
  )
})

test_that("trend validates its arguments", {
  expect_error(trend("mean", "Sigmoid"), "unknown trend type")
  expect_error(model_univariate("Normal", peaks, trends = trend("nope", "Linear")),
               "unknown trend parameter")
  expect_error(model_univariate("Normal", peaks, trends = trend(0, "Linear")),
               "positive integer")
  expect_error(model_univariate("Normal", peaks, trends = list("not a trend")),
               "must be trend\\(\\) objects")
})

test_that("trend accepts a bare object or a list of them", {
  one <- model_univariate("Normal", peaks, trends = trend("mean", "Linear"))
  listed <- model_univariate("Normal", peaks, trends = list(trend("mean", "Linear")))
  expect_identical(one$spec, listed$spec)
})

test_that("model_parameter sets bounds, the fixed flag, and a prior", {
  m <- model_univariate("Normal", peaks, parameters = model_parameter(
    "sd", lower = 100, upper = 40000, fixed = TRUE, value = 5000
  ))
  p <- m$spec$parameters[[1]]
  expect_equal(p$index, 1L)
  expect_equal(p$lower, 100)
  expect_equal(p$upper, 40000)
  expect_true(p$is_fixed)
  expect_equal(p$value, 5000)
})

test_that("a custom prior changes only the prior half of the log-likelihood", {
  plain <- model_log_likelihood(model_univariate("Normal", peaks))
  primed <- model_log_likelihood(model_univariate(
    "Normal", peaks,
    parameters = model_parameter("mean", prior = distribution("Normal", c(15000, 4000))),
    use_default_flat_priors = FALSE
  ))
  expect_equal(primed$data_log_likelihood, plain$data_log_likelihood)
  expect_false(isTRUE(all.equal(primed$prior_log_likelihood, plain$prior_log_likelihood)))
  expect_equal(primed$log_likelihood,
               primed$data_log_likelihood + primed$prior_log_likelihood)
})

test_that("model_parameter validates its arguments", {
  expect_error(model_parameter("sd", prior = "Normal"), "corehydro_dist")
  expect_error(model_parameter("sd", lower = 10, upper = 1), "must not exceed")
  expect_error(model_univariate("Normal", peaks, parameters = model_parameter("nope")),
               "unknown parameter")
  expect_error(model_univariate("Normal", peaks, parameters = list("nope")),
               "must be model_parameter\\(\\) objects")
})

test_that("a trended model requires a parameter position, not a name", {
  expect_error(
    model_univariate("Normal", peaks, trends = trend("mean", "Linear"),
                     parameters = model_parameter("sd", lower = 1)),
    "pass a 1-based position"
  )
  expect_silent(
    model_univariate("Normal", peaks, trends = trend("mean", "Linear"),
                     parameters = model_parameter(3, lower = 1))
  )
})

test_that("parameter_values wins over a per-parameter value", {
  m <- model_univariate("Normal", peaks,
                        parameters = model_parameter("mean", value = 1),
                        parameter_values = c(16000, 6000))
  expect_equal(model_parameters(m)$values, c(16000, 6000))
})

test_that("model_parameters names a stationary univariate model's vector", {
  p <- model_parameters(model_univariate("Normal", peaks))
  expect_equal(p$names, c("mean", "std dev"))
  expect_length(p$symbols, 2)
  # A trended model's vector is not the distribution's, so it is left unnamed.
  expect_length(model_parameters(
    model_univariate("Normal", peaks, trends = trend("mean", "Linear")))$names, 0)
})

test_that("model_validate reports validity", {
  v <- model_validate(model_univariate("Normal", peaks))
  expect_true(v$is_valid)
  expect_length(v$messages, 0)
})

test_that("model_simulate draws a reproducible seeded sample", {
  m <- model_univariate("Normal", peaks)
  a <- model_simulate(m, 5, seed = 12345)
  expect_length(a, 5)
  expect_identical(a, model_simulate(m, 5, seed = 12345))
  expect_false(isTRUE(all.equal(a, model_simulate(m, 5, seed = 999))))
})

test_that("model_log_likelihood evaluates at supplied parameters", {
  m <- model_univariate("Normal", peaks)
  at_default <- model_log_likelihood(m)
  at_explicit <- model_log_likelihood(m, at_default$parameters)
  expect_equal(at_explicit$log_likelihood, at_default$log_likelihood)
  moved <- model_log_likelihood(m, c(16000, 6000))
  expect_equal(moved$parameters, c(16000, 6000))
  expect_false(isTRUE(all.equal(moved$log_likelihood, at_default$log_likelihood)))
  expect_error(model_log_likelihood(m, c(1, 2, 3)), "wrong length")
})

test_that("every model family builds and validates", {
  models <- list(
    univariate = model_univariate("Normal", peaks),
    mixture = model_mixture(c("Normal", "Normal"), peaks),
    competing_risks = model_competing_risks(c("Gumbel", "Gumbel"), peaks),
    point_process = model_point_process(peaks, threshold = 9000, total_years = 15),
    bulletin17c = model_bulletin17c(peaks),
    # The time-series default of max(30, floor(0.8 * n)) training steps exceeds this
    # 20-point series, so it is set explicitly (see model_ar()'s documentation).
    ar = model_ar(series, p = 1, training_time_steps = 15),
    ma = model_ma(series, q = 1, training_time_steps = 15),
    arima = model_arima(series, p = 1, d = 0, q = 1, training_time_steps = 15),
    arimax = model_arimax(series, covariates = list(seq_along(series) / 10), p = 1, d = 0,
                          q = 0, training_time_steps = 15),
    rating_curve = model_rating_curve(
      c(1.0, 1.4, 1.9, 2.3, 2.8, 3.2, 3.7, 4.1, 4.6, 5.0, 5.5, 6.0),
      c(12, 28, 61, 98, 158, 214, 302, 379, 490, 587, 723, 872)
    ),
    spatial_gev = model_spatial_gev(
      matrix(c(0, 0, 1, 0, 0, 1), ncol = 2, byrow = TRUE),
      matrix(c(10, 12, 11, 14, 15, 13, 9, 10, 11, 16, 17, 15), ncol = 3, byrow = TRUE)
    ),
    bivariate = model_bivariate(
      marginal_x = list(family = "Normal", data = peaks,
                        parameter_values = c(mean(peaks), sd(peaks))),
      marginal_y = list(family = "Normal", data = peaks / 10,
                        parameter_values = c(mean(peaks) / 10, sd(peaks) / 10))
    )
  )
  for (nm in names(models)) {
    expect_s3_class(models[[nm]], "corehydro_model")
    # Bulletin17CDistribution is an IGMMModel, not a ModelBase, so it has no validate() arm.
    if (nm == "bulletin17c") next
    expect_true(model_validate(models[[nm]])$is_valid, info = nm)
  }
})

test_that("training_time_steps is required for a series shorter than the default", {
  # The model default is max(30, floor(0.8 * n)); for a 20-point series that is 30, which
  # exceeds the series and fails validation until the caller sets it.
  expect_false(model_validate(model_ar(series, p = 1))$is_valid)
  expect_match(model_validate(model_ar(series, p = 1))$messages, "Training time steps")
  expect_true(model_validate(model_ar(series, p = 1, training_time_steps = 15))$is_valid)
})

test_that("time-series models reject a censored frame", {
  expect_error(model_ar(analysis_data(series), p = 1), "plain numeric vector in sequence order")
})

test_that("model_rating_curve and model_bivariate validate their inputs", {
  expect_error(model_rating_curve(1:5, 1:4), "same length")
  expect_error(model_bivariate(marginal_x = list(family = "Normal"),
                               marginal_y = list(family = "Normal", data = 1:5)),
               "`family` and `data`")
})

test_that("a bivariate model simulates an n by 2 matrix", {
  m <- model_bivariate(
    marginal_x = list(family = "Normal", data = peaks,
                      parameter_values = c(mean(peaks), sd(peaks))),
    marginal_y = list(family = "Normal", data = peaks / 10,
                      parameter_values = c(mean(peaks) / 10, sd(peaks) / 10))
  )
  draw <- model_simulate(m, 6, seed = 12345)
  expect_true(is.matrix(draw))
  expect_equal(dim(draw), c(6L, 2L))
})

test_that("print summarizes the model", {
  expect_output(print(model_univariate("Normal", peaks)), "univariate_distribution: Normal")
  expect_output(print(model_univariate("Normal", peaks)), "15 exact observations")
  expect_output(print(model_ar(series, p = 1)), "time_series/ar")
  expect_output(print(model_mixture(c("Normal", "Gumbel"), peaks)), "Normal \\+ Gumbel")
  expect_output(
    print(model_univariate("Normal", peaks, trends = trend("mean", "Linear"))),
    "trends: Linear on parameter 1"
  )
  expect_output(print(trend("mean", "Linear")), "Linear on parameter mean")
  expect_output(print(model_parameter("sd", lower = 1)), "lower = 1")
})

test_that("the model verbs reject a non-model", {
  expect_error(model_validate(peaks), "corehydro_model")
  expect_error(model_simulate("nope", 1), "corehydro_model")
})
