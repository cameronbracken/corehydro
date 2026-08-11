# Internal: resolve an analysis's first argument, which accepts either a plain numeric vector
# (the convenience path every analysis has always had) or a corehydro_model built by one of the
# model_*() constructors. `build` assembles the spec for the vector path; `expected_type` is the
# model spec type this analysis can run, checked so a mismatched model fails here with a clear
# message rather than deep in the C++ dispatch.
#
# A model built over an analysis_data() frame carries its observations inline, so the separate
# dataset vector is empty; the spec builder prefers the inline frame in that case.
analysis_input <- function(data, build, expected_type = NULL) {
  if (inherits(data, "corehydro_model")) {
    spec <- data$spec
    actual <- if (is.null(spec$type)) "univariate_distribution" else spec$type
    if (!is.null(expected_type) && !identical(actual, expected_type)) {
      stop(sprintf(
        "this analysis takes a %s model, but was given a %s model", expected_type, actual
      ), call. = FALSE)
    }
    return(list(json = to_spec_json(spec), dataset = data$dataset, model = data))
  }
  if (inherits(data, "corehydro_data")) {
    stop(
      "pass an analysis_data() frame through a model, e.g. ",
      "univariate_analysis(model_univariate(\"Normal\", data))",
      call. = FALSE
    )
  }
  list(json = to_spec_json(build()), dataset = as.double(data), model = NULL)
}

#' Bayesian univariate frequency analysis
#'
#' Fit a univariate distribution to a sample with a Bayesian MCMC analysis and return the
#' frequency (quantile) curve, the posterior mean and credible band, and goodness-of-fit
#' scalars. Wraps the shared C++ `UnivariateAnalysis` ported from USACE-RMC RMC.BestFit.
#'
#' Because both the R and Python packages call the identical compiled core with a bit-exact
#' Mersenne Twister, a seeded call returns identical numbers in either language.
#'
#' @param data numeric vector of observations, or a [model_univariate()] model. A model carries
#'   its own family, so `distribution` is then ignored, and it can bring censored observations
#'   (see [analysis_data()]), nonstationary trends (see [trend()]), and parameter bounds or
#'   priors (see [model_parameter()]).
#' @param distribution distribution family name (e.g. `"Normal"`, `"GeneralizedExtremeValue"`,
#'   `"LogPearsonTypeIII"`) -- any name the core distribution factory accepts. Ignored when
#'   `data` is a model.
#' @param sampler MCMC sampler: `"DEMCz"` (default), `"DEMCzs"`, `"ARWMH"`, or `"NUTS"`.
#' @param iterations number of post-warmup MCMC iterations.
#' @param output_length number of posterior samples used to build the credible band.
#' @param credible_level credible-interval width (e.g. `0.90` for a 90% band).
#' @param seed PRNG seed for the sampler (fixed for reproducibility).
#' @param exceedance_probabilities optional numeric vector of exceedance probabilities at which to
#'   tabulate the curve; when `NULL`, the 25 standard default ordinates are used. Must be in
#'   **ascending** order (as the defaults are); an unsorted vector fails validation and the
#'   analysis reports itself as not valid.
#' @param thinning_interval MCMC thinning interval; `-1` (default) keeps the sampler's own default.
#' @details The MCMC warmup (burn-in) length is set automatically to `max(50, iterations / 2)`; it
#'   is not a user parameter.
#' @return A named list: `parameters` (point-estimate distribution parameters), `mode_curve`,
#'   `mean_curve`, `lower_ci`, `upper_ci` (one value per exceedance ordinate), and the scalars
#'   `aic`, `bic`, `dic`, `rmse`.
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' fit <- univariate_analysis(peaks, "Normal", sampler = "DEMCzs",
#'                            iterations = 100, output_length = 400, seed = 12345)
#' fit$parameters
#'
#' # The same analysis over a censored record, through a model.
#' d <- analysis_data(
#'   exact = peaks,
#'   interval = data.frame(index = 10, lower = 30000, value = 35000, upper = 40000)
#' )
#' fit2 <- univariate_analysis(model_univariate("Normal", d), sampler = "DEMCzs",
#'                             iterations = 100, output_length = 400, seed = 12345)
#' fit2$parameters
univariate_analysis <- function(data, distribution, sampler = "DEMCz", iterations = 3000L,
                                output_length = 10000L, credible_level = 0.90, seed = 12345L,
                                exceedance_probabilities = NULL, thinning_interval = -1L) {
  input <- analysis_input(data, function() {
    list(family = as.character(distribution), dataset = "data")
  }, "univariate_distribution")
  ep <- if (is.null(exceedance_probabilities)) numeric(0) else as.double(exceedance_probabilities)
  ch_analysis_univariate_run_(
    input$json, input$dataset, sampler, as.integer(iterations), as.integer(output_length),
    as.double(credible_level), as.integer(seed), ep, as.integer(thinning_interval)
  )
}

#' Fit and rank candidate distributions
#'
#' Fit each of the 14 candidate distributions to a sample by maximum likelihood and return their
#' goodness-of-fit metrics. Ranking is left to the caller. Wraps the shared C++ `FittingAnalysis`.
#'
#' @param data numeric vector of observations.
#' @return A named list with equal-length vectors `distribution` (candidate name), `aic`, `bic`,
#'   `rmse`, and `converged` (logical, whether the fit succeeded) -- one entry per candidate.
#' @export
#' @examples
#' peaks <- c(45000, 38000, 52000, 61000, 33000, 49000, 55000, 42000, 67000, 39000)
#' gof <- fit_distributions(peaks)
#' gof$distribution[which.min(gof$aic)]
fit_distributions <- function(data) {
  ch_analysis_fit_distributions_(as.double(data))
}

#' Bulletin 17C flood-frequency analysis
#'
#' Fit a Bulletin 17C (log-Pearson Type III) flood-frequency model by the generalized method of
#' moments and return the Cohn-style delta-method confidence intervals, the fitted parameters, and
#' the sandwich covariance. Wraps the shared C++ `Bulletin17CAnalysis`.
#'
#' @param data numeric vector of annual peak observations, or a [model_bulletin17c()] model. A
#'   model can bring historical and paleoflood interval observations, perception thresholds, and
#'   Multiple Grubbs-Beck low outliers (see [analysis_data()]), which is the case Bulletin 17C
#'   exists for.
#' @param uncertainty_method uncertainty-quantification method: `"MultivariateNormal"` (default),
#'   `"Bootstrap"`, `"LinkedMultivariateNormal"`, or `"BiasCorrectedBootstrap"`.
#' @param output_length number of parameter-set draws used for uncertainty quantification.
#' @param seed PRNG seed for the uncertainty draw.
#' @param confidence_level confidence level for the intervals (e.g. `0.90`).
#' @param exceedance_probabilities optional numeric vector of exceedance probabilities; when
#'   `NULL`, the 25 standard default ordinates are used. Must be in **ascending** order.
#' @return A named list: `exceedance_probabilities`, `point_estimates` (log10 space), `lower_ci`,
#'   `upper_ci` (discharge space), `confidence_level`, `beta1`, `nu`, `quantile_variance`,
#'   `parameters` (fitted location/scale/shape), and `covariance` (the p x p sandwich covariance).
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
#'            19200, 13800, 25600, 10500, 16900)
#' ci <- bulletin17c_analysis(peaks, output_length = 200, seed = 12345)
#' ci$confidence_level
#'
#' # With Multiple Grubbs-Beck low-outlier censoring, through a model.
#' m <- model_bulletin17c(analysis_data(peaks, mgbt_low_outliers = TRUE))
#' bulletin17c_analysis(m, output_length = 200, seed = 12345)$confidence_level
bulletin17c_analysis <- function(data, uncertainty_method = "MultivariateNormal",
                                 output_length = 10000L, seed = 12345L, confidence_level = 0.90,
                                 exceedance_probabilities = NULL) {
  input <- analysis_input(data, function() {
    list(type = "bulletin17c", family = "LogPearsonTypeIII", dataset = "data")
  }, "bulletin17c")
  ep <- if (is.null(exceedance_probabilities)) numeric(0) else as.double(exceedance_probabilities)
  res <- ch_analysis_b17c_run_(
    input$json, input$dataset, uncertainty_method, as.integer(output_length),
    as.integer(seed), as.double(confidence_level), ep
  )
  # Reshape the flat row-major covariance into a p x p matrix (empty when unestimated).
  d <- res$covariance_dim
  if (length(res$covariance) > 0 && d > 0) {
    res$covariance <- matrix(res$covariance, nrow = d, ncol = d, byrow = TRUE)
  }
  res$covariance_dim <- NULL
  res
}

# Internal: assemble the model JSON and call the shared per-family dispatch (D5). The seven
# per-family analyses share the univariate_analysis result surface, so the wrappers below differ
# only in the model spec they build. `training_time_steps` / `forecasting_time_steps` are ignored
# by the univariate-family analyses (passed as -1).
#
# `data` is either a plain numeric vector, in which case `build` assembles the spec, or a model
# from the matching model_*() constructor. `expected_type` guards the mismatch; the time-series
# families all share the `time_series` spec type, so `expected_subtype` narrows it further.
.ch_family_run <- function(analysis_type, build, data, sampler, iterations, output_length,
                           credible_level, seed, exceedance_probabilities, thinning_interval,
                           training_time_steps, forecasting_time_steps, expected_type,
                           expected_subtype = NULL) {
  input <- analysis_input(data, build, expected_type)
  if (!is.null(expected_subtype) && !is.null(input$model)) {
    actual <- input$model$spec$subtype
    if (!identical(actual, expected_subtype)) {
      stop(sprintf(
        "this analysis takes a %s model, but was given a %s model", expected_subtype, actual
      ), call. = FALSE)
    }
  }
  ep <- if (is.null(exceedance_probabilities)) numeric(0) else as.double(exceedance_probabilities)
  ch_analysis_family_run_(
    analysis_type, input$json, input$dataset, sampler, as.integer(iterations),
    as.integer(output_length), as.double(credible_level), as.integer(seed), ep,
    as.integer(thinning_interval), as.integer(training_time_steps),
    as.integer(forecasting_time_steps)
  )
}

#' Bayesian mixture-model frequency analysis
#'
#' Fit a finite mixture of 1-3 component distributions to a sample with a Bayesian MCMC analysis
#' and return the frequency (quantile) curve, credible band, and goodness-of-fit scalars. Wraps the
#' shared C++ `MixtureAnalysis`.
#'
#' @param data numeric vector of observations.
#' @param families character vector of component distribution family names (e.g. `c("Normal",
#'   "Normal")`).
#' @param zero_inflated logical; add a zero-inflation component (default `FALSE`).
#' @inheritParams univariate_analysis
#' @return A named list: `parameters`, `mode_curve`, `mean_curve`, `lower_ci`, `upper_ci`, and the
#'   scalars `aic`, `bic`, `dic`, `rmse`.
#' @export
#' @examples
#' x <- c(520, 580, 610, 700, 760, 850, 950, 5000, 5400, 5800, 6300, 6800)
#' fit <- mixture_analysis(x, c("Normal", "Normal"), iterations = 100, output_length = 200,
#'                         seed = 12345, thinning_interval = 1)
#' fit$aic
mixture_analysis <- function(data, families, zero_inflated = FALSE, sampler = "DEMCz",
                             iterations = 3000L, output_length = 10000L, credible_level = 0.90,
                             seed = 12345L, exceedance_probabilities = NULL,
                             thinning_interval = -1L) {
  .ch_family_run(
    "mixture", function() {
      list(
        type = "mixture", families = spec_array(as.character(families)),
        zero_inflated = as.logical(zero_inflated), dataset = "data"
      )
    }, data, sampler, iterations, output_length, credible_level, seed,
    exceedance_probabilities, thinning_interval, -1L, -1L, "mixture"
  )
}

#' Bayesian competing-risk frequency analysis
#'
#' Fit a competing-risks model (the observed maximum of several independent parent distributions)
#' with a Bayesian MCMC analysis. Wraps the shared C++ `CompetingRiskAnalysis`.
#'
#' @param data numeric vector of observations.
#' @param families character vector of parent distribution family names (e.g. `c("Gumbel",
#'   "Weibull")`).
#' @inheritParams univariate_analysis
#' @return A named list with the same shape as [mixture_analysis()].
#' @export
#' @examples
#' x <- c(7872, 8624, 5894, 12540, 8307, 6009, 13320, 10641, 8458, 8838, 11742, 9117)
#' fit <- competing_risk_analysis(x, c("Gumbel", "Weibull"), iterations = 100,
#'                                output_length = 200, seed = 12345, thinning_interval = 1)
#' fit$aic
competing_risk_analysis <- function(data, families, sampler = "DEMCz", iterations = 3000L,
                                    output_length = 10000L, credible_level = 0.90, seed = 12345L,
                                    exceedance_probabilities = NULL, thinning_interval = -1L) {
  .ch_family_run(
    "competing_risk", function() {
      list(
        type = "competing_risks", families = spec_array(as.character(families)),
        dataset = "data"
      )
    }, data, sampler, iterations, output_length, credible_level, seed,
    exceedance_probabilities, thinning_interval, -1L, -1L, "competing_risks"
  )
}

#' Bayesian point-process (peaks-over-threshold) frequency analysis
#'
#' Fit a peaks-over-threshold point-process model with a Bayesian MCMC analysis. Wraps the shared
#' C++ `PointProcessAnalysis`.
#'
#' @param data numeric vector of peak observations.
#' @param threshold optional numeric threshold (when `NULL`, the model default cascade is used).
#' @param total_years optional numeric exposure length in years.
#' @inheritParams univariate_analysis
#' @return A named list with the same shape as [mixture_analysis()].
#' @export
#' @examples
#' x <- c(950, 1020, 1130, 1250, 1090, 1430, 1180, 1620, 1300, 1550, 1210)
#' fit <- point_process_analysis(x, threshold = 900, total_years = 20, iterations = 100,
#'                               output_length = 200, seed = 12345, thinning_interval = 1)
#' fit$dic
point_process_analysis <- function(data, threshold = NULL, total_years = NULL, sampler = "DEMCz",
                                   iterations = 3000L, output_length = 10000L, credible_level = 0.90,
                                   seed = 12345L, exceedance_probabilities = NULL,
                                   thinning_interval = -1L) {
  .ch_family_run(
    "point_process", function() {
      model <- list(type = "point_process", dataset = "data")
      if (!is.null(threshold)) model$threshold <- as.double(threshold)
      if (!is.null(total_years)) model$total_years <- as.double(total_years)
      model
    }, data, sampler, iterations, output_length, credible_level, seed,
    exceedance_probabilities, thinning_interval, -1L, -1L, "point_process"
  )
}

# Internal: shared time-series wrapper body (D5). AR / MA / ARIMA / ARIMAX differ only in the
# model spec; all return the univariate_analysis surface (curves are the posterior forecast).
.ch_time_series_run <- function(analysis_type, build, data, sampler, iterations, output_length,
                                credible_level, seed, thinning_interval, training_time_steps,
                                forecasting_time_steps) {
  .ch_family_run(
    analysis_type, build, data, sampler, iterations, output_length, credible_level, seed, NULL,
    thinning_interval,
    if (is.null(training_time_steps)) -1L else as.integer(training_time_steps),
    as.integer(forecasting_time_steps),
    "time_series", analysis_type
  )
}

#' Bayesian autoregressive AR(p) time-series analysis
#'
#' Fit an AR(p) autoregressive model with a Bayesian MCMC analysis and return the posterior
#' forecast curve, credible band, and goodness-of-fit scalars. Wraps the shared C++ `ARAnalysis`.
#'
#' @param data numeric vector of the observed time series (in sequence order).
#' @param order_p autoregressive order (default `1`).
#' @param include_intercept logical; include an intercept term (default `TRUE`).
#' @param training_time_steps number of leading steps used for calibration; when `NULL` the model
#'   default (`max(30, floor(0.8 * n))`) is used, which is invalid for short series -- set it
#'   explicitly (e.g. `15`) when `n` is small.
#' @param forecasting_time_steps number of steps to forecast past the observed series (default `0`).
#' @inheritParams univariate_analysis
#' @return A named list: `parameters` (posterior point estimate), `mode_curve`, `mean_curve`,
#'   `lower_ci`, `upper_ci` (the forecast + credible band over the observed range plus the forecast
#'   horizon), and the scalars `aic`, `bic`, `dic`, `rmse`.
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5, 13.9, 15.2,
#'        14.1, 12.9, 13.6, 15.0, 16.2, 14.8)
#' fit <- ar_analysis(x, order_p = 1, training_time_steps = 15, forecasting_time_steps = 3,
#'                    iterations = 100, output_length = 200, seed = 12345, thinning_interval = 1)
#' length(fit$mode_curve)
ar_analysis <- function(data, order_p = 1L, include_intercept = TRUE, training_time_steps = NULL,
                        forecasting_time_steps = 0L, sampler = "DEMCz", iterations = 3000L,
                        output_length = 10000L, credible_level = 0.90, seed = 12345L,
                        thinning_interval = -1L) {
  .ch_time_series_run(
    "ar", function() {
      list(
      type = "time_series", subtype = "ar", dataset = "data",
      orders = list(p = as.integer(order_p)), include_intercept = as.logical(include_intercept)
    )
    }, data, sampler, iterations, output_length, credible_level, seed, thinning_interval,
    training_time_steps, forecasting_time_steps
  )
}

#' Bayesian moving-average MA(q) time-series analysis
#'
#' Fit an MA(q) moving-average model with a Bayesian MCMC analysis. Wraps the shared C++
#' `MAAnalysis`.
#'
#' @param data numeric vector of the observed time series (in sequence order).
#' @param order_q moving-average order (default `1`).
#' @inheritParams ar_analysis
#' @return A named list with the same shape as [ar_analysis()].
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5, 13.9, 15.2,
#'        14.1, 12.9, 13.6, 15.0, 16.2, 14.8)
#' fit <- ma_analysis(x, order_q = 1, training_time_steps = 15, forecasting_time_steps = 3,
#'                    iterations = 100, output_length = 200, seed = 12345, thinning_interval = 1)
#' fit$rmse
ma_analysis <- function(data, order_q = 1L, include_intercept = TRUE, training_time_steps = NULL,
                        forecasting_time_steps = 0L, sampler = "DEMCz", iterations = 3000L,
                        output_length = 10000L, credible_level = 0.90, seed = 12345L,
                        thinning_interval = -1L) {
  .ch_time_series_run(
    "ma", function() {
      list(
      type = "time_series", subtype = "ma", dataset = "data",
      orders = list(q = as.integer(order_q)), include_intercept = as.logical(include_intercept)
    )
    }, data, sampler, iterations, output_length, credible_level, seed, thinning_interval,
    training_time_steps, forecasting_time_steps
  )
}

#' Bayesian ARIMA(p,d,q) time-series analysis
#'
#' Fit an ARIMA(p,d,q) model with a Bayesian MCMC analysis. Wraps the shared C++ `ARIMAAnalysis`.
#'
#' @param data numeric vector of the observed time series (in sequence order).
#' @param order_p autoregressive order (default `1`).
#' @param order_d differencing order (default `0`).
#' @param order_q moving-average order (default `1`).
#' @inheritParams ar_analysis
#' @return A named list with the same shape as [ar_analysis()].
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5, 13.9, 15.2,
#'        14.1, 12.9, 13.6, 15.0, 16.2, 14.8)
#' fit <- arima_analysis(x, order_p = 1, order_d = 1, order_q = 1, training_time_steps = 15,
#'                       forecasting_time_steps = 3, iterations = 100, output_length = 200,
#'                       seed = 12345, thinning_interval = 1)
#' fit$aic
arima_analysis <- function(data, order_p = 1L, order_d = 0L, order_q = 1L, include_intercept = TRUE,
                           training_time_steps = NULL, forecasting_time_steps = 0L,
                           sampler = "DEMCz", iterations = 3000L, output_length = 10000L,
                           credible_level = 0.90, seed = 12345L, thinning_interval = -1L) {
  .ch_time_series_run(
    "arima", function() {
      list(
      type = "time_series", subtype = "arima", dataset = "data",
      orders = list(p = as.integer(order_p), d = as.integer(order_d), q = as.integer(order_q)),
      include_intercept = as.logical(include_intercept)
    )
    }, data, sampler, iterations, output_length, credible_level, seed,
    thinning_interval, training_time_steps, forecasting_time_steps
  )
}

#' Bayesian ARIMAX(p,d,q) time-series analysis
#'
#' Fit an ARIMAX model (ARIMA with a deterministic trend and optional exogenous covariates) with a
#' Bayesian MCMC analysis. Wraps the shared C++ `ARIMAXAnalysis`. Covariate forecasting past the
#' observed range is not exposed; run fit-only (`forecasting_time_steps = 0`) with covariates.
#'
#' @param data numeric vector of the observed time series (in sequence order).
#' @param order_p autoregressive order (default `1`).
#' @param order_d differencing order (default `0`).
#' @param order_q moving-average order (default `0`).
#' @param order_b exogenous-covariate lag order (default `0`).
#' @param trend deterministic trend: `"None"` (default), `"Linear"`, `"Quadratic"`, or `"Cubic"`.
#' @inheritParams ar_analysis
#' @return A named list with the same shape as [ar_analysis()].
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5, 13.9, 15.2,
#'        14.1, 12.9, 13.6, 15.0, 16.2, 14.8)
#' fit <- arimax_analysis(x, trend = "Linear", training_time_steps = 15,
#'                        forecasting_time_steps = 0, iterations = 100, output_length = 200,
#'                        seed = 12345, thinning_interval = 1)
#' fit$bic
arimax_analysis <- function(data, order_p = 1L, order_d = 0L, order_q = 0L, order_b = 0L,
                            trend = "None", include_intercept = TRUE, training_time_steps = NULL,
                            forecasting_time_steps = 0L, sampler = "DEMCz", iterations = 3000L,
                            output_length = 10000L, credible_level = 0.90, seed = 12345L,
                            thinning_interval = -1L) {
  .ch_time_series_run(
    "arimax", function() {
      list(
      type = "time_series", subtype = "arimax", dataset = "data",
      orders = list(
        p = as.integer(order_p), d = as.integer(order_d), q = as.integer(order_q),
        b = as.integer(order_b)
      ),
      include_intercept = as.logical(include_intercept), trend = as.character(trend)
    )
    }, data, sampler, iterations, output_length, credible_level, seed,
    thinning_interval, training_time_steps, forecasting_time_steps
  )
}

#' Bayesian estimation diagnostics (leverage / influence / prior influence)
#'
#' Fit a univariate distribution to a sample with a Bayesian MCMC analysis and compute the three
#' estimation diagnostics: observation leverage (from the posterior Hessian at the MAP), PSIS-LOO
#' influence (per-observation Pareto-k), and prior influence (prior-vs-data log-likelihood share).
#' Wraps the shared C++ `BayesianAnalysis` diagnostics.
#'
#' @param data numeric vector of observations.
#' @param distribution distribution family name (any name the core distribution factory accepts).
#' @param sampler MCMC sampler: `"DEMCz"` (default), `"DEMCzs"`, `"ARWMH"`, or `"NUTS"`.
#' @param iterations number of post-warmup MCMC iterations.
#' @param output_length number of posterior samples retained.
#' @param seed PRNG seed for the sampler.
#' @param thinning_interval MCMC thinning interval; `-1` (default) keeps the sampler's own default.
#' @param thin_every prior-influence posterior thinning stride (default `10`).
#' @return A named list with three sub-lists: `leverage` (`index`, `leverage`, `fit_influence`,
#'   `variance_influence`, `value` arrays; `prior_leverage`; `total_leverage`,
#'   `total_fit_influence`, `total_variance_influence`), `influence` (`pareto_k`, `elpd_loo` arrays;
#'   `count`, `mean_pareto_k`, `max_pareto_k`, `count_pareto_k_above_05`/`_07`/`_10`,
#'   `proportion_problematic`, `is_reliable`), and `prior_influence` (`count`,
#'   `prior_precision_share`, `total_prior_log_likelihood`, `total_data_log_likelihood`,
#'   `prior_to_data_ratio`, `is_prior_influential`, `mean_prior_precision_share`).
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' d <- estimation_diagnostics(peaks, "Normal", iterations = 100, output_length = 200,
#'                             seed = 12345, thinning_interval = 1)
#' d$influence$max_pareto_k
estimation_diagnostics <- function(data, distribution, sampler = "DEMCz", iterations = 3000L,
                                   output_length = 10000L, seed = 12345L, thinning_interval = -1L,
                                   thin_every = 10L) {
  input <- analysis_input(data, function() {
    list(family = as.character(distribution), dataset = "data")
  }, "univariate_distribution")
  ch_analysis_diagnostics_run_(
    input$json, input$dataset, sampler, as.integer(iterations), as.integer(output_length),
    as.integer(seed), as.integer(thinning_interval), as.integer(thin_every)
  )
}

# --- X11: the five remaining analyses + BootstrapAnalysis + predictive checks ----------------
# Each wrapper assembles the construct list the shared C++ runner
# (corehydro::analyses::support::run_extended_analysis) understands and calls the single
# ch_analysis_extended_run_ dispatch. The Python twins (corehydropy.analysis) build the identical
# construct, so a seeded call returns identical numbers in either language.

.ch_extended_run <- function(target, construct, datasets) {
  ns <- asNamespace("corehydror")
  ns$ch_analysis_extended_run_(target, to_spec_json(construct), to_spec_json(datasets))
}

# Internal: the data half of an extended-analysis construct. A plain numeric vector goes into the
# separate `datasets` map under the key the model spec names; an analysis_data() frame travels
# inline in the model spec, and the datasets map then carries nothing.
extended_data_block <- function(data) {
  if (inherits(data, "corehydro_data")) {
    return(list(model = list(data_frame = unclass(data)), datasets = list(data = numeric(0))))
  }
  list(model = list(dataset = "data"), datasets = list(data = as.double(data)))
}

# Internal: the model half of an extended-analysis construct whose model is a single univariate
# distribution. Accepts a plain vector, an analysis_data() frame, or a model_univariate() model.
extended_model_spec <- function(data, distribution, expected_type) {
  if (inherits(data, "corehydro_model")) {
    actual <- if (is.null(data$spec$type)) "univariate_distribution" else data$spec$type
    if (!identical(actual, expected_type)) {
      stop(sprintf(
        "this analysis takes a %s model, but was given a %s model", expected_type, actual
      ), call. = FALSE)
    }
    datasets <- list(data = if (length(data$dataset)) data$dataset else numeric(0))
    return(list(model = data$spec, datasets = datasets))
  }
  block <- extended_data_block(data)
  list(
    model = c(list(family = as.character(distribution)), block$model),
    datasets = block$datasets
  )
}

#' Composite frequency analysis
#'
#' Combine several fitted univariate frequency analyses (one per `families` entry, each fit to
#' `data` by a Bayesian MCMC) into a single composite frequency curve via competing-risks, mixture,
#' or model-averaging aggregation. Wraps the shared C++ `CompositeAnalysis`.
#'
#' @param data numeric vector of observations shared by every child analysis, or an
#'   [analysis_data()] frame. A frame is cloned into every child fit, so a censored record stays
#'   censored throughout.
#' @param families character vector of child distribution family names (one child analysis each).
#' @param composite_type `"CompetingRisks"` (default), `"Mixture"`, or `"ModelAverage"`.
#' @param average_method model-averaging criterion when `composite_type = "ModelAverage"`:
#'   `"AIC"` (default), `"BIC"`, `"DIC"`, `"WAIC"`, `"LOOIC"`, `"Equal"`, or `"RMSE"`.
#' @inheritParams univariate_analysis
#' @return A named list: `parameters`, `mode_curve`, `mean_curve`, `lower_ci`, `upper_ci`, and the
#'   scalars `aic`, `bic`, `dic`, `rmse`.
#' @export
composite_analysis <- function(data, families, composite_type = "CompetingRisks",
                               average_method = "AIC", sampler = "DEMCz", iterations = 3000L,
                               output_length = 10000L, credible_level = 0.90, seed = 12345L,
                               exceedance_probabilities = NULL, thinning_interval = -1L) {
  block <- extended_data_block(data)
  construct <- list(
    model = c(list(families = spec_array(as.character(families))), block$model),
    composite_type = as.character(composite_type), average_method = as.character(average_method),
    sampler = as.character(sampler), iterations = as.integer(iterations),
    output_length = as.integer(output_length), credible_level = as.double(credible_level),
    seed = as.integer(seed), thinning_interval = as.integer(thinning_interval)
  )
  if (!is.null(exceedance_probabilities)) {
    construct$exceedance_probabilities <- as.double(exceedance_probabilities)
  }
  .ch_extended_run("CompositeAnalysis", construct, block$datasets)
}

#' Hierarchical spatial-GEV frequency analysis
#'
#' Fit the hierarchical spatial-GEV model over a set of gauged sites with a Bayesian MCMC and
#' return the regional (site-averaged) frequency curve plus per-site GEV/quantile credible bands.
#' Wraps the shared C++ `SpatialGEVAnalysis`.
#'
#' @param coordinates numeric matrix (or list of length-2 vectors), one `[x, y]` row per site.
#' @param at_site_data numeric matrix (or list of rows), `[observations x sites]` at-site maxima.
#' @param cross_validation logical; run leave-one-site-out cross-validation (default `FALSE`).
#' @inheritParams univariate_analysis
#' @param number_of_chains number of MCMC chains (default `4`).
#' @return A named list: the regional `parameters`/`mode_curve`/`mean_curve`/`lower_ci`/`upper_ci`
#'   + `aic`/`bic`/`dic`, `site_count`, the per-site `site_location_mean`/`_lower`/`_upper` (and the
#'   scale/shape analogues), site-0 `site0_quantile_mean`/`_lower`/`_upper`/`_mode`, and (when
#'   `cross_validation`) `cv_site_prediction_errors`/`cv_site_rmse`/`cv_site_bias`/`cv_mae`/
#'   `cv_rmse`/`cv_mean_bias`.
#' @export
spatial_gev_analysis <- function(coordinates, at_site_data, cross_validation = FALSE,
                                 sampler = "DEMCz", iterations = 3000L, output_length = 10000L,
                                 credible_level = 0.90, seed = 12345L, number_of_chains = 4L,
                                 exceedance_probabilities = NULL, thinning_interval = -1L) {
  to_rows <- function(m) {
    if (is.matrix(m)) lapply(seq_len(nrow(m)), function(i) as.double(m[i, ])) else lapply(m, as.double)
  }
  construct <- list(
    model = list(
      type = "spatial_gev", coordinates = to_rows(coordinates), at_site_data = to_rows(at_site_data)
    ),
    cross_validation = as.logical(cross_validation), sampler = as.character(sampler),
    iterations = as.integer(iterations), output_length = as.integer(output_length),
    number_of_chains = as.integer(number_of_chains), credible_level = as.double(credible_level),
    seed = as.integer(seed), thinning_interval = as.integer(thinning_interval)
  )
  if (!is.null(exceedance_probabilities)) {
    construct$exceedance_probabilities <- as.double(exceedance_probabilities)
  }
  .ch_extended_run("SpatialGEVAnalysis", construct, list(unused = 0))
}

# Internal: assemble the shared `bivariate` model spec (two fixed marginals + a copula).
.ch_bivariate_model <- function(marginal_x_family, marginal_x_data, marginal_x_parameters,
                                marginal_y_family, marginal_y_data, marginal_y_parameters,
                                copula, estimation_method) {
  list(
    type = "bivariate", copula = as.character(copula),
    estimation_method = as.character(estimation_method),
    marginal_x = list(
      family = as.character(marginal_x_family), data = as.double(marginal_x_data),
      parameter_values = as.double(marginal_x_parameters)
    ),
    marginal_y = list(
      family = as.character(marginal_y_family), data = as.double(marginal_y_data),
      parameter_values = as.double(marginal_y_parameters)
    )
  )
}

#' Bivariate (copula) joint-exceedance frequency analysis
#'
#' Fit a copula over two fixed univariate marginals with a Bayesian MCMC and return the
#' AND-joint-exceedance mode/mean curve + credible band across an XY ordinate grid. Wraps the
#' shared C++ `BivariateAnalysis`.
#'
#' @param marginal_x_family,marginal_y_family marginal distribution family names.
#' @param marginal_x_data,marginal_y_data numeric vectors of the marginal samples.
#' @param marginal_x_parameters,marginal_y_parameters numeric fixed marginal parameter vectors.
#' @param copula copula family name (e.g. `"Normal"`, `"Clayton"`, `"Gumbel"`).
#' @param estimation_method copula estimation method (default `"InferenceFromMargins"`).
#' @param xy_x,xy_y numeric vectors of the (x, y) ordinate grid at which to evaluate the joint curve.
#' @inheritParams univariate_analysis
#' @param number_of_chains number of MCMC chains (default `4`).
#' @return A named list: `parameters`, `mode_curve`, `mean_curve`, `lower_ci`, `upper_ci`, and the
#'   scalars `aic`, `bic`, `dic`, `rmse` (one curve entry per XY ordinate).
#' @export
bivariate_analysis <- function(marginal_x_family, marginal_x_data, marginal_x_parameters,
                               marginal_y_family, marginal_y_data, marginal_y_parameters, xy_x, xy_y,
                               copula = "Normal", estimation_method = "InferenceFromMargins",
                               sampler = "DEMCz", iterations = 3000L, output_length = 10000L,
                               credible_level = 0.90, seed = 12345L, number_of_chains = 4L,
                               thinning_interval = -1L) {
  construct <- list(
    model = .ch_bivariate_model(
      marginal_x_family, marginal_x_data, marginal_x_parameters,
      marginal_y_family, marginal_y_data, marginal_y_parameters, copula, estimation_method
    ),
    xy_x = as.double(xy_x), xy_y = as.double(xy_y), sampler = as.character(sampler),
    iterations = as.integer(iterations), output_length = as.integer(output_length),
    number_of_chains = as.integer(number_of_chains), credible_level = as.double(credible_level),
    seed = as.integer(seed), thinning_interval = as.integer(thinning_interval)
  )
  .ch_extended_run("BivariateAnalysis", construct, list(unused = 0))
}

#' Coincident-frequency analysis
#'
#' Combine a fitted bivariate (copula) analysis with an M x N response surface `Z = f(X, Y)` to
#' derive the annual-exceedance-probability curve of `Z` via the conditional-frequency law of total
#' probability. Wraps the shared C++ `CoincidentFrequencyAnalysis` (which internally fits the
#' bivariate analysis from the same marginal/copula spec).
#'
#' @inheritParams bivariate_analysis
#' @param x_values,y_values numeric ascending vectors of the primary (X) / secondary (Y) ordinates.
#' @param response numeric M x N matrix of the response surface `Z[i, j] = f(x_i, y_j)`.
#' @param number_of_bins number of Z output bins (default `50`).
#' @return A named list: `z_output_values` (the Z bins), `mode_curve`, `mean_curve`, `lower_ci`,
#'   `upper_ci` (the AEP curve + credible band, one entry per Z bin).
#' @export
coincident_frequency_analysis <- function(marginal_x_family, marginal_x_data, marginal_x_parameters,
                                          marginal_y_family, marginal_y_data, marginal_y_parameters,
                                          x_values, y_values, response, number_of_bins = 50L,
                                          copula = "Normal",
                                          estimation_method = "InferenceFromMargins",
                                          sampler = "DEMCz", iterations = 3000L,
                                          output_length = 10000L, credible_level = 0.90,
                                          seed = 12345L, number_of_chains = 4L,
                                          thinning_interval = -1L) {
  response <- as.matrix(response)
  construct <- list(
    model = .ch_bivariate_model(
      marginal_x_family, marginal_x_data, marginal_x_parameters,
      marginal_y_family, marginal_y_data, marginal_y_parameters, copula, estimation_method
    ),
    x_values = as.double(x_values), y_values = as.double(y_values),
    response_rows = as.integer(nrow(response)), response_cols = as.integer(ncol(response)),
    response = as.double(as.vector(t(response))), number_of_bins = as.integer(number_of_bins),
    sampler = as.character(sampler), iterations = as.integer(iterations),
    output_length = as.integer(output_length), number_of_chains = as.integer(number_of_chains),
    credible_level = as.double(credible_level), seed = as.integer(seed),
    thinning_interval = as.integer(thinning_interval)
  )
  .ch_extended_run("CoincidentFrequencyAnalysis", construct, list(unused = 0))
}

#' Stage-discharge rating-curve frequency analysis
#'
#' Fit a stage-discharge rating curve with a Bayesian MCMC and return the predicted-discharge
#' mode/mean curve + credible band across a stage grid. Wraps the shared C++ `RatingCurveAnalysis`.
#'
#' @param stage,discharge numeric vectors of date-aligned stage / discharge observations.
#' @param segments number of rating-curve segments (default `1`).
#' @param stage_bins number of stage grid points (default `NULL`, keeps the data-derived default).
#' @param min_stage,max_stage optional stage-grid bounds (default `NULL`, data-derived).
#' @inheritParams univariate_analysis
#' @param number_of_chains number of MCMC chains (default `4`).
#' @return A named list: `parameters`, `mode_curve`, `mean_curve`, `lower_ci`, `upper_ci`, `aic`,
#'   `bic`, `dic`, `rmse`.
#' @export
rating_curve_analysis <- function(stage, discharge, segments = 1L, stage_bins = NULL,
                                  min_stage = NULL, max_stage = NULL, sampler = "DEMCz",
                                  iterations = 3000L, output_length = 10000L, credible_level = 0.90,
                                  seed = 12345L, number_of_chains = 4L, thinning_interval = -1L) {
  construct <- list(
    model = list(
      type = "rating_curve", segments = as.integer(segments), stage = as.double(stage),
      discharge = as.double(discharge)
    ),
    sampler = as.character(sampler), iterations = as.integer(iterations),
    output_length = as.integer(output_length), number_of_chains = as.integer(number_of_chains),
    credible_level = as.double(credible_level), seed = as.integer(seed),
    thinning_interval = as.integer(thinning_interval)
  )
  if (!is.null(stage_bins)) construct$stage_bins <- as.integer(stage_bins)
  if (!is.null(min_stage)) construct$min_stage <- as.double(min_stage)
  if (!is.null(max_stage)) construct$max_stage <- as.double(max_stage)
  .ch_extended_run("RatingCurveAnalysis", construct, list(unused = 0))
}

#' Parametric bootstrap confidence bands for a distribution
#'
#' Fit `distribution` to `data`, then resample it `replications` times to derive percentile
#' confidence bands on the quantile curve. Wraps the shared C++ `BootstrapAnalysis` (Numerics).
#'
#' @param data numeric vector of observations.
#' @param distribution distribution family name to fit + bootstrap.
#' @param estimation_method fit method: `"MaximumLikelihood"` (default), `"MethodOfMoments"`, or
#'   `"MethodOfLinearMoments"`.
#' @param probabilities numeric vector of non-exceedance probabilities at which to tabulate the
#'   quantile curve + confidence band.
#' @param sample_size bootstrap sample size (default `NULL`, uses `length(data)`).
#' @param replications number of bootstrap replications (default `1000`).
#' @param seed PRNG seed for the bootstrap.
#' @param alpha significance level for the two-sided percentile band (default `0.1` -> 90% band).
#' @return A named list: `parameters` (fitted), `mode_curve` (fitted-distribution quantiles),
#'   `mean_curve`, `lower_ci`, `upper_ci` (percentile band, one entry per probability).
#' @export
bootstrap_analysis <- function(data, distribution, probabilities,
                               estimation_method = "MaximumLikelihood", sample_size = NULL,
                               replications = 1000L, seed = 12345L, alpha = 0.1) {
  construct <- list(
    model = list(family = as.character(distribution), dataset = "data"),
    estimation_method = as.character(estimation_method), replications = as.integer(replications),
    seed = as.integer(seed), alpha = as.double(alpha), probabilities = as.double(probabilities)
  )
  if (!is.null(sample_size)) construct$sample_size <- as.integer(sample_size)
  .ch_extended_run("BootstrapAnalysis", construct, list(data = as.double(data)))
}

#' Prior predictive check
#'
#' Sample parameter sets from a model's priors, simulate a dataset per draw, and summarize the
#' resulting predictive distribution. Wraps the shared C++ `PriorPredictiveCheck`.
#'
#' @param data numeric vector used to build the model (its priors are the check's target).
#' @param distribution distribution family name.
#' @param number_of_draws number of prior draws (default `1000`).
#' @param sample_size simulated dataset size per draw (default `NULL`, uses `length(data)`).
#' @param seed PRNG seed.
#' @return A named list: `number_of_valid_draws` and the predictive quantile summaries
#'   `summary_mean_quantiles`, `summary_sd_quantiles`, `summary_min_quantiles`,
#'   `summary_max_quantiles` (each `[2.5, 25, 50, 75, 97.5]%`).
#' @export
prior_predictive_check <- function(data, distribution, number_of_draws = 1000L, sample_size = NULL,
                                   seed = 12345L) {
  spec <- extended_model_spec(data, distribution, "univariate_distribution")
  construct <- list(
    model = spec$model,
    number_of_draws = as.integer(number_of_draws), seed = as.integer(seed)
  )
  if (!is.null(sample_size)) construct$sample_size <- as.integer(sample_size)
  .ch_extended_run("PriorPredictiveCheck", construct, spec$datasets)
}

#' Posterior predictive check
#'
#' Fit a Bayesian MCMC over a model, draw replicate datasets from the posterior, and compute the
#' common posterior predictive p-values (mean / SD / skewness / min / max) plus a misfit flag.
#' Wraps the shared C++ `PosteriorPredictiveCheck`.
#'
#' @param data numeric vector of observations (also the observed data for the p-value comparison).
#' @param distribution distribution family name.
#' @param number_of_replicates number of posterior replicate datasets (default `1000`).
#' @inheritParams univariate_analysis
#' @return A named list: `number_of_replicates`, `mean_p_value`, `sd_p_value`, `skewness_p_value`,
#'   `min_p_value`, `max_p_value`, and `has_misfit` (`1` when any p-value is extreme, else `0`).
#' @export
posterior_predictive_check <- function(data, distribution, sampler = "DEMCz", iterations = 3000L,
                                       output_length = 10000L, seed = 12345L,
                                       number_of_replicates = 1000L, thinning_interval = -1L) {
  spec <- extended_model_spec(data, distribution, "univariate_distribution")
  construct <- list(
    model = spec$model,
    sampler = as.character(sampler), iterations = as.integer(iterations),
    output_length = as.integer(output_length), seed = as.integer(seed),
    number_of_replicates = as.integer(number_of_replicates),
    thinning_interval = as.integer(thinning_interval)
  )
  .ch_extended_run("PosteriorPredictiveCheck", construct, spec$datasets)
}
