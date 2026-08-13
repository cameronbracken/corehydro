# The goodness-of-fit surface. Every verb runs one method of the "gof" group through the shared
# toolbox runner; nothing holds C++ state.

kGofMetrics <- c("rmse", "mse", "mae", "mape", "smape", "nse", "log_nse", "kge", "kge_mod",
                 "pbias", "rsr", "pearson", "r_squared", "d", "d_mod", "d_ref", "ve")

#' Goodness-of-fit metrics for a modeled series
#'
#' Mirrors the continuous metrics of the C# `GoodnessOfFit` class of the Numerics library.
#'
#' @param observed,modeled numeric vectors of equal length.
#' @param metrics `"all"` (the default) for every metric as a named vector, or a character vector
#'   of metric names drawn from `rmse`, `mse`, `mae`, `mape`, `smape`, `nse`, `log_nse`, `kge`,
#'   `kge_mod`, `pbias`, `rsr`, `pearson`, `r_squared`, `d`, `d_mod`, `d_ref`, `ve`.
#' @param k degrees-of-freedom correction subtracted from the sample size in the RMSE
#'   denominator. Default 0.
#' @return a named numeric vector.
#' @examples
#' obs <- c(2, 4, 6, 8, 10)
#' mod <- c(2.2, 3.9, 6.4, 7.5, 10.1)
#' goodness_of_fit(obs, mod)
#' goodness_of_fit(obs, mod, metrics = c("nse", "kge"))
#' @export
goodness_of_fit <- function(observed, modeled, metrics = "all", k = 0) {
  check_pair(observed, modeled, "observed", "modeled")
  if (!identical(metrics, "all")) {
    unknown <- setdiff(metrics, kGofMetrics)
    if (length(unknown) > 0L) {
      stop(sprintf("unknown metric(s): %s. Available: %s",
                   paste(unknown, collapse = ", "), paste(kGofMetrics, collapse = ", ")),
           call. = FALSE)
    }
  }
  r <- toolbox_run("gof", "metrics", list(observed, modeled), list(k = k))
  out <- stats::setNames(r$values, r$names)
  if (identical(metrics, "all")) out else out[metrics]
}

#' Classification metrics for a thresholded series
#'
#' Mirrors the classification region of the C# `GoodnessOfFit` class: a value at or above
#' `threshold` counts as a positive.
#'
#' @param observed,modeled numeric vectors of equal length.
#' @param threshold the value separating a positive from a negative.
#' @return a named numeric vector of accuracy, precision, recall, f1, specificity,
#'   balanced_accuracy, and the four confusion-matrix counts tp, tn, fp, fn.
#' @examples
#' obs <- c(1, 5, 2, 9, 4)
#' mod <- c(2, 6, 1, 8, 3)
#' classification_metrics(obs, mod, threshold = 4)
#' @export
classification_metrics <- function(observed, modeled, threshold) {
  check_pair(observed, modeled, "observed", "modeled")
  if (!is.numeric(threshold) || length(threshold) != 1L) {
    stop("`threshold` must be a single number", call. = FALSE)
  }
  r <- toolbox_run("gof", "classification", list(observed, modeled), list(threshold = threshold))
  stats::setNames(r$values, r$names)
}

#' Goodness-of-fit test statistic for a fitted distribution
#'
#' @param x numeric vector of observations. Sorted internally, as the C# methods require.
#' @param d a `corehydro_dist` object.
#' @param test one of `"ks"` (Kolmogorov-Smirnov D), `"ad"` (Anderson-Darling A squared), or
#'   `"chi_squared"`.
#' @return a single numeric test statistic.
#' @examples
#' x <- c(2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9)
#' gof_test(x, distribution("Normal", c(3.4, 1.1)))
#' @export
gof_test <- function(x, d, test = c("ks", "ad", "chi_squared")) {
  test <- match.arg(test)
  if (!inherits(d, "corehydro_dist")) {
    stop("`d` must be a corehydro_dist object; create one with distribution()", call. = FALSE)
  }
  toolbox_run("gof", test, list(x), list(model = d))$values[[1]]
}

#' Root mean squared error of a fitted distribution
#'
#' @param x numeric vector of observations.
#' @param d a `corehydro_dist` object.
#' @param plotting_positions optional numeric vector of exceedance probabilities. `NULL` (the
#'   default) uses Weibull positions, matching the C# overload.
#' @return a single numeric RMSE.
#' @examples
#' x <- c(2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9)
#' gof_rmse(x, distribution("Normal", c(3.4, 1.1)))
#' @export
gof_rmse <- function(x, d, plotting_positions = NULL) {
  if (!inherits(d, "corehydro_dist")) {
    stop("`d` must be a corehydro_dist object; create one with distribution()", call. = FALSE)
  }
  data <- if (is.null(plotting_positions)) list(x) else list(x, plotting_positions)
  toolbox_run("gof", "rmse_dist", data, list(model = d))$values[[1]]
}

#' Information criteria
#'
#' @param n sample size.
#' @param k number of estimated parameters.
#' @param log_likelihood the maximized log-likelihood.
#' @return a single numeric criterion value.
#' @examples
#' aic(k = 2, log_likelihood = -121.01131220612)
#' bic(n = 30, k = 2, log_likelihood = -121.01131220612)
#' @export
aic <- function(k, log_likelihood) {
  toolbox_run("gof", "aic", list(), list(k = k, log_likelihood = log_likelihood))$values[[1]]
}

#' @rdname aic
#' @export
aicc <- function(n, k, log_likelihood) {
  toolbox_run("gof", "aicc", list(),
              list(n = n, k = k, log_likelihood = log_likelihood))$values[[1]]
}

#' @rdname aic
#' @export
bic <- function(n, k, log_likelihood) {
  toolbox_run("gof", "bic", list(),
              list(n = n, k = k, log_likelihood = log_likelihood))$values[[1]]
}

#' Model weights from a vector of criteria
#'
#' @param aic numeric vector of AIC values, one per candidate model.
#' @param rmse numeric vector of RMSE values, one per candidate model.
#' @return a numeric vector of weights summing to one.
#' @examples
#' aic_weights(c(246.0, 248.8, 251.2))
#' rmse_weights(c(1.2, 1.5, 2.0))
#' @export
aic_weights <- function(aic) {
  toolbox_run("gof", "aic_weights", list(aic))$values
}

#' @rdname aic_weights
#' @export
rmse_weights <- function(rmse) {
  toolbox_run("gof", "rmse_weights", list(rmse))$values
}
