# Ordinary least squares by singular value decomposition, mirroring the C# `LinearRegression`
# class of the Numerics library (Task 5 of the numerics-toolbox surface). Its own file rather
# than an addition to R/toolbox.R: `linear_regression()` is the first toolbox verb that returns a
# classed object with a full S3 method set (coef/vcov/residuals/predict/print/summary), the same
# shape as `corehydro_fit` in R/fit.R -- collecting it and its methods together here mirrors how
# fit.R stays separate from the plain-value toolbox verbs.

#' Ordinary least squares by singular value decomposition
#'
#' Mirrors the C# `LinearRegression` class of the Numerics library: estimates
#' `Y = alpha + beta*X + e`, `e ~ N(0, sigma)`, via SVD.
#'
#' @param x a numeric matrix of predictors with one row per observation, or a numeric vector for
#'   a single predictor.
#' @param y numeric vector of responses, one per row of `x`.
#' @param intercept whether to fit an intercept. Default `TRUE`.
#' @return a `corehydro_lm` list with `coefficients`, `standard_errors`, `covariance`,
#'   `residuals`, `r_squared`, `adj_r_squared`, `sigma`, `df`, and `n`.
#' @seealso [predict.corehydro_lm()].
#' @examples
#' x <- cbind(c(1, 2, 3, 4, 5), c(2, 1, 4, 3, 5))
#' y <- c(3.1, 4.2, 8.1, 9.2, 13.0)
#' fit <- linear_regression(x, y)
#' coef(fit)
#' @export
linear_regression <- function(x, y, intercept = TRUE) {
  x <- if (is.matrix(x)) x else matrix(as.double(x), ncol = 1L)
  if (!is.numeric(x)) {
    stop("`x` must be numeric", call. = FALSE)
  }
  if (!is.numeric(y) || length(y) != nrow(x)) {
    stop(sprintf(
      "`y` must be numeric with one value per row of `x`; got %d and %d",
      length(y), nrow(x)
    ), call. = FALSE)
  }
  # t(x) flattens row-major, which is the layout the runner's Matrix(rows, columns, flat) wants
  # (see toolbox_runner.hpp's build_regression()).
  flat <- as.double(t(x))
  opts <- list(rows = nrow(x), columns = ncol(x), intercept = isTRUE(intercept))
  v <- toolbox_run("regression", "fit", list(flat, y), opts)
  named <- stats::setNames(v$values, v$names)
  p <- ncol(x) + as.integer(isTRUE(intercept))
  param_names <- c(if (isTRUE(intercept)) "(Intercept)", paste0("x", seq_len(ncol(x))))

  cov <- toolbox_run("regression", "covariance", list(flat, y), opts)
  covariance <- matrix(cov$values, nrow = cov$dims[[1]], ncol = cov$dims[[2]], byrow = TRUE)
  dimnames(covariance) <- list(param_names, param_names)

  res <- toolbox_run("regression", "residuals", list(flat, y), opts)

  coefficients <- named[seq_len(p)]
  names(coefficients) <- param_names
  standard_errors <- named[p + seq_len(p)]
  names(standard_errors) <- param_names

  structure(list(
    coefficients = coefficients,
    standard_errors = standard_errors,
    covariance = covariance,
    residuals = res$values,
    r_squared = named[["r_squared"]],
    adj_r_squared = named[["adj_r_squared"]],
    sigma = named[["sigma"]],
    df = as.integer(named[["df"]]),
    n = as.integer(named[["n"]]),
    x = x,
    y = as.double(y),
    intercept = isTRUE(intercept)
  ), class = "corehydro_lm")
}

#' @export
print.corehydro_lm <- function(x, ...) {
  cat("<corehydro_lm>\n")
  cat("  coefficients:\n")
  print(x$coefficients)
  cat(sprintf(
    "  sigma: %.6g   r.squared: %.6g   adj.r.squared: %.6g   df: %d   n: %d\n",
    x$sigma, x$r_squared, x$adj_r_squared, x$df, x$n
  ))
  invisible(x)
}

#' @export
summary.corehydro_lm <- function(object, ...) {
  print(object)
  cat("  standard errors:\n")
  print(object$standard_errors)
  invisible(object)
}

#' @export
coef.corehydro_lm <- function(object, ...) object$coefficients

#' @export
vcov.corehydro_lm <- function(object, ...) object$covariance

#' @export
residuals.corehydro_lm <- function(object, ...) object$residuals

#' Predict from a fitted linear regression
#'
#' Mirrors the C# `LinearRegression.Predict`/`PredictionIntervals` methods.
#'
#' @param object a `corehydro_lm` from [linear_regression()].
#' @param newdata a numeric matrix of predictors with the same number of columns as the `x` the
#'   model was fitted with (an intercept column, if any, is added internally), or a numeric
#'   vector for a single-predictor model.
#' @param interval if `TRUE`, also return the `level` prediction interval (a Student-t interval
#'   around the mean response, mirroring `PredictionIntervals`). Default `FALSE`.
#' @param level prediction interval level, between 0 and 1. Default `0.90`, matching
#'   `PredictionIntervals`'s own `alpha = 0.1` default.
#' @param ... unused; present for generic consistency.
#' @return with `interval = FALSE`, a numeric vector of predicted values, one per row of
#'   `newdata`. With `interval = TRUE`, a matrix with columns `lower`, `upper`, `mean`.
#' @seealso [linear_regression()].
#' @examples
#' x <- cbind(c(1, 2, 3, 4, 5), c(2, 1, 4, 3, 5))
#' y <- c(3.1, 4.2, 8.1, 9.2, 13.0)
#' fit <- linear_regression(x, y)
#' predict(fit, cbind(1, 2))
#' @export
predict.corehydro_lm <- function(object, newdata, interval = FALSE, level = 0.90, ...) {
  ncol_x <- ncol(object$x)
  if (!is.matrix(newdata)) {
    # A bare vector is only unambiguous for a single-predictor model (one value == one row);
    # for more than one predictor, matrix(vec, ncol = p) would fill by column and silently
    # transpose the caller's intended row-major layout, so require an explicit matrix instead.
    if (ncol_x != 1L) {
      stop(sprintf(
        "`newdata` must be a matrix with %d columns, matching the fitted predictors", ncol_x
      ), call. = FALSE)
    }
    newdata <- matrix(as.double(newdata), ncol = 1L)
  }
  if (!is.numeric(newdata) || ncol(newdata) != ncol_x) {
    stop(sprintf(
      "`newdata` must be numeric with %d column(s), matching the fitted predictors; got %d",
      ncol_x, ncol(newdata)
    ), call. = FALSE)
  }
  flat <- as.double(t(object$x))
  new_flat <- as.double(t(newdata))
  opts <- list(
    rows = nrow(object$x), columns = ncol_x, intercept = object$intercept,
    predict_rows = nrow(newdata)
  )
  data <- list(flat, object$y, new_flat)
  if (!isTRUE(interval)) {
    return(toolbox_run("regression", "predict", data, opts)$values)
  }
  level <- as.double(level)
  if (is.na(level) || level <= 0 || level >= 1) {
    stop("`level` must be greater than 0 and less than 1; got ", format(level), call. = FALSE)
  }
  r <- toolbox_run("regression", "prediction_intervals", data, c(opts, list(alpha = 1 - level)))
  m <- matrix(r$values, nrow = r$dims[[1]], ncol = r$dims[[2]], byrow = TRUE)
  colnames(m) <- r$names
  m
}
