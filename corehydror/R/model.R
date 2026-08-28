# The public model layer: constructors for the nine RMC.BestFit model families, the trend and
# parameter helpers they compose with, and the verbs that evaluate a model.
#
# A `corehydro_model` is a plain classed list holding the spec that
# core/include/corehydro/models/model_spec.hpp parses. Nothing here holds a C++ object: the core
# rebuilds the model from the spec on every call, so a model can be printed, saved, and passed
# between R and Python unchanged.

# Internal: attach the shared data block. A plain numeric vector emits `dataset`, taking the
# vector-constructor path the analyses have always used; an analysis_data() frame emits the full
# `data_frame` object. The two are equivalent for an uncensored record (the vector constructor
# delegates to the same DataFrame constructor), and only the frame can carry censored series.
# `values` is deliberately not named `dataset_values`: R's `$` partial matching would then let
# `block$dataset` resolve to it on the frame branch, emitting a spurious empty `dataset` key.
model_data_block <- function(data) {
  if (inherits(data, "corehydro_data")) {
    return(list(data_frame = unclass(data), values = numeric(0)))
  }
  if (!is.numeric(data)) {
    stop("`data` must be a numeric vector or a corehydro_data object from analysis_data()",
      call. = FALSE
    )
  }
  list(dataset = "data", values = as.double(data))
}

# Internal: the plain-word label of each parameter, taken from the leading text of the full name
# the core reports ("Mean (µ)" -> "mean", "Location (ξ)" -> "location").
parameter_labels <- function(family) {
  full <- ch_dist_parameter_names_(family)$full
  tolower(trimws(sub("\\s*\\(.*$", "", full)))
}

# A handful of names a user is likelier to reach for than the library's own wording.
parameter_aliases <- c(
  sd = "std dev", stdev = "std dev", std = "std dev", sigma = "std dev",
  mu = "mean", alpha = "scale", kappa = "shape", xi = "location"
)

# Internal: resolve a parameter name or 1-based position to the 0-based index the spec wants.
# Names are matched against the family's own parameter names in three forms: the symbol the core
# reports (`µ`), the full name (`Mean (µ)`), and its plain-word prefix (`mean`), plus the alias
# table above. A stationary model's parameter vector is the distribution's, so this resolves; a
# trended model widens that vector, and there a position is required.
resolve_parameter <- function(parameter, family = NULL, what = "parameter") {
  if (is.numeric(parameter)) {
    index <- as.integer(parameter)
    if (is.na(index) || index < 1L) {
      stop(sprintf("`%s` position must be a positive integer (1-based)", what), call. = FALSE)
    }
    return(index - 1L)
  }
  if (!is.character(parameter) || length(parameter) != 1L) {
    stop(sprintf("`%s` must be a single parameter name or 1-based position", what), call. = FALSE)
  }
  if (is.null(family)) {
    stop(sprintf(
      paste0(
        "`%s` was given as the name '%s', but this model's parameter names are not ",
        "known here (a trend widens the parameter vector); pass a 1-based position instead"
      ),
      what, parameter
    ), call. = FALSE)
  }

  names <- ch_dist_parameter_names_(family)
  labels <- parameter_labels(family)
  wanted <- tolower(trimws(parameter))
  if (!is.na(parameter_aliases[wanted])) wanted <- unname(parameter_aliases[wanted])

  hit <- match(wanted, tolower(names$short))
  if (is.na(hit)) hit <- match(wanted, tolower(names$full))
  if (is.na(hit)) hit <- match(wanted, labels)
  if (is.na(hit)) {
    stop(sprintf(
      "unknown %s '%s' for '%s'; expected one of %s (or a 1-based position)",
      what, parameter, family, paste(labels, collapse = ", ")
    ), call. = FALSE)
  }
  hit - 1L
}

#' Attach a trend to a distribution parameter
#'
#' Describe a nonstationary trend on one parameter of a distribution, for
#' [model_univariate()]. Attaching any trend makes the model nonstationary: the trended
#' parameter is replaced in the parameter vector by the trend's own coefficients, so a linear
#' trend on the location turns one location parameter into an intercept and a slope.
#'
#' @param parameter the distribution parameter to trend, as a name (`"location"`) or a 1-based
#'   position (`1`). Names are resolved against the family's parameter names; see
#'   [dist_params()].
#' @param type the trend model: one of `"Constant"`, `"Linear"`, `"Quadratic"`, `"Cubic"`,
#'   `"Exponential"`, `"Logistic"`, `"Power"`, `"Reciprocal"`, `"Sinusoidal"`, `"StepFunction"`,
#'   or `"GeneralLinear"`.
#' @param start_index optional 0-based index the trend is anchored at; the data-driven default is
#'   used when omitted.
#' @param values optional numeric vector of trend coefficients, length matching the trend's own
#'   parameter count.
#' @return An object of class `corehydro_trend`.
#' @seealso [model_univariate()], [model_parameter()].
#' @export
#' @examples
#' trend("location", "Linear")
#' trend(1, "Quadratic", values = c(100, 2.5, -0.01))
trend <- function(parameter, type, start_index = NULL, values = NULL) {
  known <- c(
    "Constant", "Cubic", "Exponential", "Linear", "Logistic", "Power",
    "Quadratic", "Reciprocal", "Sinusoidal", "StepFunction", "GeneralLinear"
  )
  type <- as.character(type)
  hit <- match(tolower(type), tolower(known))
  if (is.na(hit)) {
    stop(sprintf(
      "unknown trend type '%s'; expected one of %s", type, paste(known, collapse = ", ")
    ), call. = FALSE)
  }
  structure(
    list(
      parameter = parameter, type = known[hit],
      start_index = if (is.null(start_index)) NULL else as.integer(start_index),
      values = if (is.null(values)) NULL else spec_array(as.double(values))
    ),
    class = "corehydro_trend"
  )
}

#' @export
print.corehydro_trend <- function(x, ...) {
  cat(sprintf("<corehydro_trend> %s on parameter %s\n", x$type, format(x$parameter)))
  invisible(x)
}

#' Constrain or prime one model parameter
#'
#' Set the bounds, fixed flag, prior distribution, or starting value of a single model parameter.
#' Pass the result to the `parameters` argument of any `model_*()` constructor.
#'
#' Supplying a prior replaces the model's default. Do that together with
#' `use_default_flat_priors = FALSE` on the model, so a later data assignment cannot overwrite
#' what you set.
#'
#' @param parameter the parameter to constrain, as a name (`"scale"`) or a 1-based position. In a
#'   model carrying trends the parameter vector is wider than the distribution's, so a position
#'   is required there.
#' @param value optional starting value. A `parameter_values` vector on the model wins over this.
#' @param lower,upper optional bounds used by the estimators and samplers.
#' @param fixed logical; hold the parameter at its value instead of estimating it.
#' @param prior optional [distribution()] object used as the parameter's prior.
#' @return An object of class `corehydro_model_parameter`.
#' @seealso [model_univariate()], [trend()].
#' @export
#' @examples
#' model_parameter("scale", lower = 0.5, upper = 40)
#' model_parameter("mean", prior = distribution("Normal", c(15000, 4000)))
model_parameter <- function(parameter, value = NULL, lower = NULL, upper = NULL,
                            fixed = NULL, prior = NULL) {
  if (!is.null(prior) && !inherits(prior, "corehydro_dist")) {
    stop("`prior` must be a corehydro_dist object from distribution()", call. = FALSE)
  }
  if (!is.null(lower) && !is.null(upper) && as.double(lower) > as.double(upper)) {
    stop("`lower` must not exceed `upper`", call. = FALSE)
  }
  structure(
    list(
      parameter = parameter,
      value = if (is.null(value)) NULL else as.double(value),
      lower = if (is.null(lower)) NULL else as.double(lower),
      upper = if (is.null(upper)) NULL else as.double(upper),
      is_fixed = if (is.null(fixed)) NULL else isTRUE(fixed),
      prior = prior
    ),
    class = "corehydro_model_parameter"
  )
}

#' @export
print.corehydro_model_parameter <- function(x, ...) {
  bits <- character(0)
  if (!is.null(x$value)) bits <- c(bits, sprintf("value = %g", x$value))
  if (!is.null(x$lower)) bits <- c(bits, sprintf("lower = %g", x$lower))
  if (!is.null(x$upper)) bits <- c(bits, sprintf("upper = %g", x$upper))
  if (isTRUE(x$is_fixed)) bits <- c(bits, "fixed")
  if (!is.null(x$prior)) bits <- c(bits, sprintf("prior = %s", x$prior$family))
  cat(sprintf(
    "<corehydro_model_parameter> %s%s\n", format(x$parameter),
    if (length(bits)) paste0(": ", paste(bits, collapse = ", ")) else ""
  ))
  invisible(x)
}

# Internal: turn the trend / parameter helper lists into their spec blocks and assemble the
# model. `family` is the distribution family whose names resolve a by-name reference, or NULL
# when the model has no single such family.
build_model <- function(spec, dataset_values, trends = NULL, parameters = NULL,
                        parameter_values = NULL, use_default_flat_priors = NULL,
                        family = NULL) {
  if (!is.null(trends)) {
    if (inherits(trends, "corehydro_trend")) trends <- list(trends)
    if (!all(vapply(trends, inherits, logical(1), "corehydro_trend"))) {
      stop("`trends` must be trend() objects", call. = FALSE)
    }
    spec$trends <- lapply(trends, function(t) {
      list(
        parameter = resolve_parameter(t$parameter, family, "trend parameter"),
        type = t$type, start_index = t$start_index, values = t$values
      )
    })
  }

  if (!is.null(parameters)) {
    if (inherits(parameters, "corehydro_model_parameter")) parameters <- list(parameters)
    if (!all(vapply(parameters, inherits, logical(1), "corehydro_model_parameter"))) {
      stop("`parameters` must be model_parameter() objects", call. = FALSE)
    }
    # A trended model's vector is wider than the distribution's, so a name cannot be resolved
    # from the family alone; require a position there rather than silently mis-targeting.
    resolve_family <- if (is.null(spec$trends)) family else NULL
    spec$parameters <- lapply(parameters, function(p) {
      list(
        index = resolve_parameter(p$parameter, resolve_family, "parameter"),
        value = p$value, lower = p$lower, upper = p$upper,
        is_fixed = p$is_fixed, prior = p$prior
      )
    })
  }

  if (!is.null(use_default_flat_priors)) {
    spec$use_default_flat_priors <- isTRUE(use_default_flat_priors)
  }
  if (!is.null(parameter_values)) {
    spec$parameter_values <- spec_array(as.double(parameter_values))
  }

  structure(
    list(spec = spec, dataset = dataset_values),
    class = "corehydro_model"
  )
}

check_model <- function(m) {
  # A corehydro_fit (fit_mle()/fit_map()/...) carries its fitted model at $model, so every
  # model_*() verb (model_validate(), model_simulate(), model_log_likelihood()) also accepts a
  # fit directly -- e.g. model_simulate(fit_mle(m), ...) -- without the caller unwrapping it.
  if (inherits(m, "corehydro_fit")) {
    return(m$model)
  }
  if (!inherits(m, "corehydro_model")) {
    stop("`model` must be a corehydro_model object from one of the model_*() constructors",
      call. = FALSE
    )
  }
  m
}

#' @export
print.corehydro_model <- function(x, ...) {
  spec <- x$spec
  label <- if (is.null(spec$type)) "univariate_distribution" else spec$type
  if (identical(label, "time_series")) label <- paste0("time_series/", spec$subtype)
  detail <- spec$family
  if (is.null(detail) && !is.null(spec$families)) detail <- paste(spec$families, collapse = " + ")
  cat(sprintf("<corehydro_model> %s%s\n", label, if (is.null(detail)) "" else paste0(": ", detail)))
  if (!is.null(spec$data_frame)) {
    cat("  data: ")
    print(structure(spec$data_frame, class = "corehydro_data"))
  } else if (length(x$dataset)) {
    cat(sprintf("  data: %d exact observations\n", length(x$dataset)))
  }
  if (!is.null(spec$trends)) {
    cat(sprintf(
      "  trends: %s\n",
      paste(vapply(spec$trends, function(t) {
        sprintf("%s on parameter %d", t$type, t$parameter + 1L)
      }, character(1)), collapse = ", ")
    ))
  }
  if (!is.null(spec$parameters)) {
    cat(sprintf("  parameter constraints: %d\n", length(spec$parameters)))
  }
  invisible(x)
}

# --- The nine model families ------------------------------------------------------------------

#' Univariate distribution model
#'
#' A single distribution fit to a record, optionally censored (via [analysis_data()]) and
#' optionally nonstationary (via [trend()]). This is the workhorse model behind
#' [univariate_analysis()].
#'
#' @param family distribution family name; see [distribution_names()].
#' @param data a numeric vector of observations, or an [analysis_data()] frame carrying censored
#'   observations.
#' @param trends optional [trend()] object or list of them, making the model nonstationary.
#' @param parameters optional [model_parameter()] object or list of them, setting bounds, fixed
#'   flags, priors, or starting values.
#' @param parameter_values optional numeric vector of all parameter values, applied last.
#' @param use_default_flat_priors logical; set `FALSE` alongside a custom prior.
#' @return An object of class `corehydro_model`.
#' @seealso [univariate_analysis()], [model_validate()], [model_simulate()].
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' model_univariate("LogPearsonTypeIII", peaks)
#'
#' # Censored: two historical floods known only within a range.
#' d <- analysis_data(
#'   exact = peaks,
#'   interval = data.frame(
#'     index = c(10, 11), lower = c(30000, 26000),
#'     value = c(35000, 29000), upper = c(40000, 32000)
#'   )
#' )
#' model_univariate("LogPearsonTypeIII", d)
#'
#' # Nonstationary: a linear trend on the location parameter.
#' model_univariate("Normal", peaks, trends = trend("mean", "Linear"))
model_univariate <- function(family, data, trends = NULL, parameters = NULL,
                             parameter_values = NULL, use_default_flat_priors = NULL) {
  block <- model_data_block(data)
  spec <- list(
    type = "univariate_distribution", family = as.character(family),
    dataset = block$dataset, data_frame = block$data_frame
  )
  build_model(spec, block$values, trends, parameters, parameter_values,
    use_default_flat_priors,
    family = as.character(family)
  )
}

#' Finite mixture model
#'
#' A weighted mixture of two or three component distributions, for records generated by more than
#' one flood mechanism. Optionally zero-inflated.
#'
#' @param families character vector of component distribution family names.
#' @param data a numeric vector of observations, or an [analysis_data()] frame.
#' @param zero_inflated logical; add a zero-inflation component.
#' @inheritParams model_univariate
#' @return An object of class `corehydro_model`.
#' @seealso [mixture_analysis()].
#' @export
#' @examples
#' x <- c(520, 580, 610, 700, 760, 850, 950, 5000, 5400, 5800, 6300, 6800)
#' model_mixture(c("Normal", "Normal"), x)
model_mixture <- function(families, data, zero_inflated = FALSE, parameters = NULL,
                          parameter_values = NULL, use_default_flat_priors = NULL) {
  block <- model_data_block(data)
  spec <- list(
    type = "mixture", families = spec_array(as.character(families)),
    zero_inflated = isTRUE(zero_inflated),
    dataset = block$dataset, data_frame = block$data_frame
  )
  build_model(spec, block$values, NULL, parameters, parameter_values,
    use_default_flat_priors
  )
}

#' Competing risks model
#'
#' The maximum of several independent flood-generating processes, each with its own distribution.
#'
#' @param families character vector of component distribution family names.
#' @param data a numeric vector of observations, or an [analysis_data()] frame.
#' @inheritParams model_univariate
#' @return An object of class `corehydro_model`.
#' @seealso [competing_risk_analysis()].
#' @export
#' @examples
#' x <- c(1200, 1500, 890, 2210, 1870, 1420, 980, 2850, 1740, 1160, 1920, 1380)
#' model_competing_risks(c("Gumbel", "Gumbel"), x)
model_competing_risks <- function(families, data, parameters = NULL, parameter_values = NULL,
                                  use_default_flat_priors = NULL) {
  block <- model_data_block(data)
  spec <- list(
    type = "competing_risks", families = spec_array(as.character(families)),
    dataset = block$dataset, data_frame = block$data_frame
  )
  build_model(spec, block$values, NULL, parameters, parameter_values,
    use_default_flat_priors
  )
}

#' Peaks-over-threshold point process model
#'
#' A point process over exceedances of a threshold, combining an arrival rate with a magnitude
#' distribution.
#'
#' @details
#' With `seasonal = TRUE` the magnitude distribution becomes TWO generalized extreme value
#' marginals with two fitted change points, and each observation is assigned to a season by its
#' day of the year -- so the data must carry DATES rather than bare indices:
#'
#' ```r
#' data <- analysis_data(exact = list(date = dates, value = peaks))
#' model <- model_point_process(data, seasonal = TRUE)
#' ```
#'
#' Observations supplied with integer indices are treated as January 1 of that year, which puts
#' every one of them in the same season.
#'
#' @param data a numeric vector of observations, or an [analysis_data()] frame.
#' @param threshold optional exceedance threshold; derived from the data when omitted. See
#'   [threshold_diagnostics()] for choosing one.
#' @param total_years optional record length in years, used for the arrival rate.
#' @param use_defaults logical; let the model derive its threshold and record length from the
#'   data. Applied before an explicit `threshold` or `total_years`, so an explicit value wins.
#' @param seasonal logical; fit two seasonal magnitude distributions with fitted change points.
#' @param time_block the block a seasonal model reduces the record over: `"water_year"` (the
#'   default), `"calendar_year"`, `"custom_year"`, `"quarter"` or `"month"`.
#' @param start_month the month a water year or custom year begins. Default 10 (October).
#' @inheritParams model_univariate
#' @return An object of class `corehydro_model`.
#' @seealso [point_process_analysis()], [threshold_diagnostics()].
#' @export
#' @examples
#' x <- c(1200, 1500, 890, 2210, 1870, 1420, 980, 2850, 1740, 1160, 1920, 1380, 2560, 1050)
#' model_point_process(x, threshold = 1000, total_years = 14)
model_point_process <- function(data, threshold = NULL, total_years = NULL, use_defaults = TRUE,
                                seasonal = FALSE,
                                time_block = c("water_year", "calendar_year", "custom_year",
                                               "quarter", "month"),
                                start_month = 10,
                                parameters = NULL, parameter_values = NULL,
                                use_default_flat_priors = NULL) {
  time_block <- match.arg(time_block)
  block <- model_data_block(data)
  spec <- list(
    type = "point_process",
    dataset = block$dataset, data_frame = block$data_frame,
    use_defaults = isTRUE(use_defaults),
    is_seasonal = if (isTRUE(seasonal)) TRUE else NULL,
    time_block = time_block,
    start_month = as.integer(start_month),
    threshold = if (is.null(threshold)) NULL else as.double(threshold),
    total_years = if (is.null(total_years)) NULL else as.double(total_years)
  )
  build_model(spec, block$values, NULL, parameters, parameter_values,
    use_default_flat_priors
  )
}

#' Bulletin 17C distribution model
#'
#' The log-Pearson Type III flood-frequency model of Bulletin 17C, fit by the generalized method
#' of moments. Handles the censored observation types directly, which is the point of using it
#' over a plain [model_univariate()].
#'
#' @param data a numeric vector of annual peaks, or an [analysis_data()] frame.
#' @param family the distribution family; one of `"LogPearsonTypeIII"` (the default),
#'   `"Exponential"`, `"Gamma"`, `"LogNormal"`, `"Normal"`, or `"PearsonTypeIII"`.
#' @inheritParams model_univariate
#' @return An object of class `corehydro_model`.
#' @seealso [bulletin17c_analysis()].
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
#'            19200, 13800, 25600, 10500, 16900)
#' model_bulletin17c(analysis_data(peaks, mgbt_low_outliers = TRUE))
model_bulletin17c <- function(data, family = "LogPearsonTypeIII", parameters = NULL,
                              parameter_values = NULL) {
  block <- model_data_block(data)
  spec <- list(
    type = "bulletin17c", family = as.character(family),
    dataset = block$dataset, data_frame = block$data_frame
  )
  build_model(spec, block$values, NULL, parameters, parameter_values,
    NULL,
    family = as.character(family)
  )
}

# Internal: the shared time-series spec. The series always travels inline as `data` so a model
# is self-contained; `transform` names a TransformType.
time_series_model <- function(subtype, data, orders, include_intercept, transform, extra = NULL,
                              training_time_steps = NULL, parameters = NULL,
                              parameter_values = NULL, use_default_flat_priors = NULL) {
  if (inherits(data, "corehydro_data")) {
    stop("time-series models take a plain numeric vector in sequence order, not an analysis_data() frame",
      call. = FALSE
    )
  }
  spec <- c(
    list(
      type = "time_series", subtype = subtype, data = spec_array(as.double(data)),
      orders = orders, include_intercept = isTRUE(include_intercept),
      transform = if (is.null(transform)) NULL else as.character(transform),
      training_time_steps = if (is.null(training_time_steps)) {
        NULL
      } else {
        as.integer(training_time_steps)
      }
    ),
    extra
  )
  build_model(spec, numeric(0), NULL, parameters, parameter_values, use_default_flat_priors)
}

#' Autoregressive AR(p) model
#'
#' @param data numeric vector of the observed series, in sequence order.
#' @param p autoregressive order.
#' @param include_intercept logical; include an intercept term.
#' @param transform optional variance-stabilizing transform: `"None"`, `"Logarithmic"`,
#'   `"BoxCox"`, or `"YeoJohnson"`.
#' @param training_time_steps number of leading steps used for calibration, the rest held
#'   back for validation. The model default is `max(30, floor(0.8 * n))`, which exceeds the
#'   series length for any series shorter than 30 and then fails [model_validate()]; set it
#'   explicitly for a short series.
#' @inheritParams model_univariate
#' @return An object of class `corehydro_model`.
#' @seealso [ar_analysis()].
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5)
#' model_ar(x, p = 1)
model_ar <- function(data, p = 1L, include_intercept = TRUE, transform = NULL,
                     training_time_steps = NULL,
                     parameters = NULL, parameter_values = NULL,
                     use_default_flat_priors = NULL) {
  time_series_model(
    "ar", data, list(p = as.integer(p)), include_intercept, transform,
    training_time_steps = training_time_steps,
    parameters = parameters, parameter_values = parameter_values,
    use_default_flat_priors = use_default_flat_priors
  )
}

#' Moving-average MA(q) model
#'
#' @param q moving-average order.
#' @inheritParams model_ar
#' @return An object of class `corehydro_model`.
#' @seealso [ma_analysis()].
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5)
#' model_ma(x, q = 1)
model_ma <- function(data, q = 1L, include_intercept = TRUE, transform = NULL,
                     training_time_steps = NULL,
                     parameters = NULL, parameter_values = NULL,
                     use_default_flat_priors = NULL) {
  time_series_model(
    "ma", data, list(q = as.integer(q)), include_intercept, transform,
    training_time_steps = training_time_steps,
    parameters = parameters, parameter_values = parameter_values,
    use_default_flat_priors = use_default_flat_priors
  )
}

#' ARIMA(p, d, q) model
#'
#' @param p autoregressive order.
#' @param d differencing order.
#' @param q moving-average order.
#' @inheritParams model_ar
#' @return An object of class `corehydro_model`.
#' @seealso [arima_analysis()].
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5)
#' model_arima(x, p = 1, d = 0, q = 1)
model_arima <- function(data, p = 1L, d = 0L, q = 1L, include_intercept = TRUE, transform = NULL,
                        training_time_steps = NULL,
                        parameters = NULL, parameter_values = NULL,
                        use_default_flat_priors = NULL) {
  time_series_model(
    "arima", data, list(p = as.integer(p), d = as.integer(d), q = as.integer(q)),
    include_intercept, transform,
    training_time_steps = training_time_steps,
    parameters = parameters, parameter_values = parameter_values,
    use_default_flat_priors = use_default_flat_priors
  )
}

#' ARIMAX(p, d, q, b) model with covariates
#'
#' @param covariates a list of numeric vectors (one per covariate series), or a matrix with one
#'   covariate per column.
#' @param b the covariate lag order.
#' @param trend_type optional deterministic trend: `"None"`, `"Linear"`, `"Quadratic"`, or
#'   `"Cubic"`.
#' @param include_seasonality logical; include seasonal terms.
#' @inheritParams model_arima
#' @return An object of class `corehydro_model`.
#' @seealso [arimax_analysis()].
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5)
#' z <- c(1.1, 1.4, 0.9, 1.6, 1.8, 1.5, 1.0, 1.7, 1.9, 1.8, 1.3, 1.6)
#' model_arimax(x, covariates = list(z), p = 1, d = 0, q = 0, b = 0)
model_arimax <- function(data, covariates, p = 1L, d = 0L, q = 0L, b = 0L,
                         include_intercept = TRUE, transform = NULL, trend_type = NULL,
                         include_seasonality = NULL, training_time_steps = NULL,
                         parameters = NULL, parameter_values = NULL,
                         use_default_flat_priors = NULL) {
  if (is.matrix(covariates)) {
    covariates <- lapply(seq_len(ncol(covariates)), function(j) as.double(covariates[, j]))
  }
  if (!is.list(covariates) || !length(covariates)) {
    stop("`covariates` must be a non-empty list of numeric vectors, or a matrix", call. = FALSE)
  }
  time_series_model(
    "arimax", data,
    list(p = as.integer(p), d = as.integer(d), q = as.integer(q), b = as.integer(b)),
    include_intercept, transform,
    extra = list(
      covariates = lapply(covariates, function(z) spec_array(as.double(z))),
      trend = if (is.null(trend_type)) NULL else as.character(trend_type),
      include_seasonality = if (is.null(include_seasonality)) {
        NULL
      } else {
        isTRUE(include_seasonality)
      }
    ),
    training_time_steps = training_time_steps,
    parameters = parameters, parameter_values = parameter_values,
    use_default_flat_priors = use_default_flat_priors
  )
}

#' Hierarchical spatial GEV model
#'
#' The Renard hierarchical spatial generalized extreme value model: at-site GEV distributions
#' whose parameters are themselves regressions over site coordinates.
#'
#' @param coordinates a sites-by-2 matrix (or list of length-2 vectors) of site coordinates.
#' @param at_site_data an observations-by-sites matrix (or list of per-observation vectors) of the
#'   at-site records.
#' @param use_copula_dependence,use_location_errors,use_scale_errors,use_shape_errors optional
#'   logicals gating the spatial-dependence and regression-error terms.
#' @param use_log_link_for_location,use_log_link_for_scale optional logicals selecting a log link
#'   for the corresponding regression.
#' @inheritParams model_univariate
#' @return An object of class `corehydro_model`.
#' @seealso [spatial_gev_analysis()].
#' @export
#' @examples
#' coords <- matrix(c(0, 0, 1, 0, 0, 1), ncol = 2, byrow = TRUE)
#' obs <- matrix(c(10, 12, 11, 14, 15, 13, 9, 10, 11, 16, 17, 15), ncol = 3, byrow = TRUE)
#' model_spatial_gev(coords, obs)
model_spatial_gev <- function(coordinates, at_site_data, use_copula_dependence = NULL,
                              use_location_errors = NULL, use_scale_errors = NULL,
                              use_shape_errors = NULL, use_log_link_for_location = NULL,
                              use_log_link_for_scale = NULL, parameters = NULL,
                              parameter_values = NULL, use_default_flat_priors = NULL) {
  to_rows <- function(m, what) {
    rows <- if (is.matrix(m)) {
      lapply(seq_len(nrow(m)), function(i) as.double(m[i, ]))
    } else if (is.list(m)) {
      lapply(m, as.double)
    } else {
      stop(sprintf("`%s` must be a matrix or a list of numeric vectors", what), call. = FALSE)
    }
    lapply(rows, spec_array)
  }
  flag <- function(x) if (is.null(x)) NULL else isTRUE(x)
  spec <- list(
    type = "spatial_gev",
    coordinates = to_rows(coordinates, "coordinates"),
    at_site_data = to_rows(at_site_data, "at_site_data"),
    use_copula_dependence = flag(use_copula_dependence),
    use_location_errors = flag(use_location_errors),
    use_scale_errors = flag(use_scale_errors),
    use_shape_errors = flag(use_shape_errors),
    use_log_link_for_location = flag(use_log_link_for_location),
    use_log_link_for_scale = flag(use_log_link_for_scale)
  )
  build_model(spec, numeric(0), NULL, parameters, parameter_values, use_default_flat_priors)
}

#' BaRatin stage-discharge rating curve model
#'
#' A hydraulically-controlled stage-discharge relationship of one to three segments, with a
#' log10-space normal residual likelihood.
#'
#' @param stage numeric vector of stage observations.
#' @param discharge numeric vector of discharge observations, aligned with `stage`.
#' @param segments number of hydraulic control segments (1 to 3).
#' @inheritParams model_univariate
#' @return An object of class `corehydro_model`.
#' @seealso [rating_curve_analysis()].
#' @export
#' @examples
#' stage <- c(1.0, 1.4, 1.9, 2.3, 2.8, 3.2, 3.7, 4.1, 4.6, 5.0, 5.5, 6.0)
#' discharge <- c(12, 28, 61, 98, 158, 214, 302, 379, 490, 587, 723, 872)
#' model_rating_curve(stage, discharge, segments = 1)
model_rating_curve <- function(stage, discharge, segments = 1L, parameters = NULL,
                               parameter_values = NULL, use_default_flat_priors = NULL) {
  stage <- as.double(stage)
  discharge <- as.double(discharge)
  if (length(stage) != length(discharge)) {
    stop("`stage` and `discharge` must be the same length", call. = FALSE)
  }
  spec <- list(
    type = "rating_curve", segments = as.integer(segments),
    stage = spec_array(stage), discharge = spec_array(discharge)
  )
  build_model(spec, numeric(0), NULL, parameters, parameter_values, use_default_flat_priors)
}

#' Bivariate copula model
#'
#' Two fixed univariate marginals coupled by a bivariate copula.
#'
#' @param marginal_x,marginal_y each a list with `family`, `data`, and `parameter_values`
#'   elements describing a fixed marginal.
#' @param copula copula family name: `"Normal"` (the default), `"StudentT"`, `"Clayton"`,
#'   `"Frank"`, `"Gumbel"`, `"Joe"`, or `"AliMikhailHaq"`.
#' @param estimation_method `"InferenceFromMargins"` (the default), `"PseudoLikelihood"`, or
#'   `"FullLikelihood"`.
#' @inheritParams model_univariate
#' @return An object of class `corehydro_model`.
#' @note Fit an Archimedean copula (`"Clayton"`, `"Frank"`, `"Gumbel"`, `"Joe"`,
#'   `"AliMikhailHaq"`) with `optimizer = "DifferentialEvolution"`. [fit_mle()] and [fit_map()]
#'   default to `"NelderMead"`, whose start point is the midpoint of the copula's constraint
#'   range: for Gumbel, whose range is `{1, 100}`, that is theta 50.5, and the local search slides
#'   to the lower bound and reports `Success` with the independence copula. On 150 Gumbel pairs
#'   simulated at theta 3, `"NelderMead"` returned theta 1.0000 with a log-likelihood of 0.0000
#'   while `"DifferentialEvolution"` returned theta 3.0334 at 110.5081. `"Powell"` errors on some
#'   samples, so `"DifferentialEvolution"` is the recommendation rather than any global optimizer.
#'   Elliptical copulas (`"Normal"`, `"StudentT"`) are not affected.
#' @seealso [bivariate_analysis()], [coincident_frequency_analysis()].
#' @export
#' @examples
#' x <- c(12, 15, 9, 22, 18, 14, 10, 28, 17, 11, 19, 13)
#' y <- c(2.1, 2.6, 1.7, 3.7, 3.1, 2.5, 1.9, 4.6, 3.0, 2.0, 3.3, 2.3)
#' model_bivariate(
#'   marginal_x = list(family = "Normal", data = x, parameter_values = c(mean(x), sd(x))),
#'   marginal_y = list(family = "Normal", data = y, parameter_values = c(mean(y), sd(y))),
#'   copula = "Normal"
#' )
model_bivariate <- function(marginal_x, marginal_y, copula = "Normal",
                            estimation_method = "InferenceFromMargins", parameters = NULL,
                            parameter_values = NULL, use_default_flat_priors = NULL) {
  as_marginal <- function(m, what) {
    if (!is.list(m) || is.null(m$family) || is.null(m$data)) {
      stop(sprintf("`%s` must be a list with `family` and `data` elements", what), call. = FALSE)
    }
    list(
      family = as.character(m$family), data = spec_array(as.double(m$data)),
      parameter_values = if (is.null(m$parameter_values)) {
        NULL
      } else {
        spec_array(as.double(m$parameter_values))
      }
    )
  }
  spec <- list(
    type = "bivariate", copula = as.character(copula),
    estimation_method = as.character(estimation_method),
    marginal_x = as_marginal(marginal_x, "marginal_x"),
    marginal_y = as_marginal(marginal_y, "marginal_y")
  )
  build_model(spec, numeric(0), NULL, parameters, parameter_values, use_default_flat_priors)
}

# --- Model verbs ------------------------------------------------------------------------------

#' Validate a model
#'
#' Check a model's data and parameters and report anything that would stop it being fit.
#'
#' @param model a `corehydro_model` from one of the `model_*()` constructors.
#' @return A named list: `is_valid` (logical) and `messages` (character vector, empty when valid).
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' model_validate(model_univariate("Normal", peaks))
model_validate <- function(model) {
  model <- check_model(model)
  ch_model_validate_(to_spec_json(model$spec), model$dataset)
}

#' Simulate from a model
#'
#' Draw a seeded random sample from a model at its current parameter values. The draws come from
#' the same Mersenne Twister stream as the upstream C# library, so a given seed reproduces the C#
#' values bit-for-bit and matches `corehydropy` exactly.
#'
#' @param model a `corehydro_model` from one of the `model_*()` constructors.
#' @param n number of values to draw.
#' @param seed integer seed.
#' @return A numeric vector of length `n`. For a bivariate model the draw is an `n` by 2 matrix.
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' model_simulate(model_univariate("Normal", peaks), n = 5, seed = 12345)
model_simulate <- function(model, n, seed = 12345L) {
  model <- check_model(model)
  out <- ch_model_simulate_(
    to_spec_json(model$spec), model$dataset, as.integer(n), as.integer(seed)
  )
  # The bivariate model draws an n x 2 matrix, flattened row-major by the glue.
  if (identical(model$spec$type, "bivariate")) {
    return(matrix(out, nrow = as.integer(n), ncol = 2L, byrow = TRUE))
  }
  out
}

#' Model log-likelihood
#'
#' Evaluate a model's log-likelihood, decomposed into its data and prior halves.
#'
#' @param model a `corehydro_model` from one of the `model_*()` constructors.
#' @param params optional numeric vector of parameter values; the model's own current values are
#'   used when omitted.
#' @return A named list: `log_likelihood`, `data_log_likelihood`, `prior_log_likelihood`, and
#'   `parameters` (the vector the model actually evaluated, which a mixture model rewrites in
#'   place to normalize its weights).
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' m <- model_univariate("Normal", peaks)
#' model_log_likelihood(m)
#' model_log_likelihood(m, c(16000, 6000))
model_log_likelihood <- function(model, params = NULL) {
  model <- check_model(model)
  ch_model_log_likelihood_(
    to_spec_json(model$spec), model$dataset,
    if (is.null(params)) numeric(0) else as.double(params)
  )
}

#' Model parameters
#'
#' The model's current parameter vector, with names where the model exposes them.
#'
#' @param model a `corehydro_model` from one of the `model_*()` constructors.
#' @return A named list with `values` (numeric), `names` (plain-word parameter labels), and
#'   `symbols` (the library's own symbols). The two name vectors are empty for any model whose
#'   parameter vector is not a single distribution's, which is every family other than a
#'   stationary univariate one.
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' model_parameters(model_univariate("Normal", peaks))
model_parameters <- function(model) {
  model <- check_model(model)
  values <- model_log_likelihood(model)$parameters
  names <- character(0)
  symbols <- character(0)
  if (!is.null(model$spec$family) && is.null(model$spec$trends)) {
    candidate <- parameter_labels(model$spec$family)
    if (length(candidate) == length(values)) {
      names <- candidate
      symbols <- ch_dist_parameter_names_(model$spec$family)$short
    }
  }
  list(values = values, names = names, symbols = symbols)
}
