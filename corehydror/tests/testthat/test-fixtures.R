# Generic, fixture-driven validation for corehydror.
#
# Reads the language-neutral oracle fixtures (the single source of truth shared with
# the C++ core and the Python package) and checks every assertion. No oracle values
# live here -- only the dispatch from fixture method names to the package's API. The GEV
# slice uses its bespoke ch_gev_* glue; every other distribution goes through the
# polymorphic ch_dist_* glue (factory + UnivariateDistributionBase).

parse_num <- function(v) {
  if (is.character(v)) {
    return(switch(v, "nan" = NaN, "inf" = Inf, "-inf" = -Inf,
                  stop(sprintf("unexpected string value: %s", v))))
  }
  as.numeric(v)
}

build_params <- function(target, construct, datasets) {
  ns <- asNamespace("corehydror")
  if (!is.null(construct$params)) {
    return(vapply(construct$params, parse_num, numeric(1)))
  }
  fit <- construct$fit
  data <- as.numeric(unlist(datasets[[fit$dataset]]))
  if (target == "GeneralizedExtremeValue") {
    f <- gev_fit(data, fit$method)
    return(c(f[["location"]], f[["scale"]], f[["shape"]]))
  }
  ns$ch_dist_fit_(target, data, fit$method)
}

dispatch_gev <- function(p, method, args) {
  loc <- p[1]; scale <- p[2]; shape <- p[3]
  ns <- asNamespace("corehydror")  # internal cpp11 functions are not exported
  moment_names <- c("mean", "median", "mode", "sd", "skewness", "kurtosis",
                    "minimum", "maximum")
  if (method %in% moment_names) {
    return(unname(ns$ch_gev_moments_(loc, scale, shape)[[method]]))
  }
  switch(method,
    pdf = ns$ch_gev_pdf_(as.double(args[[1]]), loc, scale, shape),
    cdf = ns$ch_gev_cdf_(as.double(args[[1]]), loc, scale, shape),
    quantile = ns$ch_gev_quantile_(as.double(args[[1]]), loc, scale, shape),
    parameters_valid = ns$ch_gev_valid_(loc, scale, shape),
    param = c(location = loc, scale = scale, shape = shape)[[args[[1]]]],
    linear_moment = ns$ch_gev_linear_moments_(loc, scale, shape)[[as.integer(args[[1]]) + 1L]],
    quantile_gradient = ns$ch_gev_quantile_gradient_(as.double(args[[1]]), loc, scale, shape)[[
      as.integer(args[[2]]) + 1L]],
    parameter_covariance = ns$ch_gev_parameter_covariance_(loc, scale, shape,
      as.integer(args[[1]]))[[as.integer(args[[2]]) * 3L + as.integer(args[[3]]) + 1L]],
    quantile_variance = ns$ch_gev_quantile_variance_(as.double(args[[1]]), loc, scale, shape,
      as.integer(args[[2]])),
    quantile_se = sqrt(ns$ch_gev_quantile_variance_(as.double(args[[1]]), loc, scale, shape,
      as.integer(args[[2]]))),
    stop(sprintf("unknown fixture method: %s", method))
  )
}

dispatch_generic <- function(target, p, method, args) {
  ns <- asNamespace("corehydror")
  moment_names <- c("mean", "median", "mode", "sd", "skewness", "kurtosis",
                    "minimum", "maximum")
  if (method %in% moment_names) {
    return(unname(ns$ch_dist_moments_(target, p)[[method]]))
  }
  switch(method,
    pdf = ns$ch_dist_pdf_(target, p, as.double(args[[1]])),
    cdf = ns$ch_dist_cdf_(target, p, as.double(args[[1]])),
    quantile = ns$ch_dist_quantile_(target, p, as.double(args[[1]])),
    parameters_valid = ns$ch_dist_valid_(target, p),
    param = p[[as.integer(args[[1]]) + 1L]],
    linear_moment = ns$ch_dist_linear_moments_(target, p)[[as.integer(args[[1]]) + 1L]],
    # args: [sample_size, seed, index] -- one draw from the seeded MT stream.
    random_value = ns$ch_dist_random_(target, p, as.integer(args[[1]]),
      as.integer(args[[2]]))[[as.integer(args[[3]]) + 1L]],
    # Static GammaDistribution utility, not tied to `p` -- args: [skewness, probability].
    partial_kp = ns$ch_dist_gamma_partial_kp_(as.double(args[[1]]), as.double(args[[2]])),
    parameter_covariance = ,
    quantile_variance = ,
    quantile_gradient = dispatch_standard_error(target, p, method, args),
    stop(sprintf("unknown fixture method: %s", method))
  )
}

# The IStandardError surface. It has no bespoke ch_dist_* glue (nothing outside the fixtures
# calls it), so it routes through the shared dist_runner, which reaches it by the same
# capability cast the C++ and Python runners use. Args follow the flattened convention the
# bespoke GEV slice above already speaks -- parameter_covariance [sample_size, row, col],
# quantile_variance [probability, sample_size], quantile_gradient [probability, index] -- while
# the runner returns the whole matrix (row-major) or vector, so the indexing happens here.
# GeneralizedNormal is the only family reaching this today, and only through quantile_gradient:
# C# throws NotImplementedException for its other two, and the port mirrors that.
dispatch_standard_error <- function(target, p, method, args) {
  ns <- asNamespace("corehydror")
  # as.list keeps `parameters` a JSON ARRAY: auto_unbox would collapse a length-1 vector to a
  # scalar, which the spec reader rejects.
  spec <- to_runner_json(list(family = target, parameters = as.list(as.double(p))))
  if (identical(method, "quantile_variance")) {
    v <- ns$ch_dist_spec_run_(spec, method,
      to_runner_json(list(as.double(args[[1]]), as.double(args[[2]]))))$values
    return(v[[1]])
  }
  v <- ns$ch_dist_spec_run_(spec, method, to_runner_json(list(as.double(args[[1]]))))$values
  if (identical(method, "quantile_gradient")) return(v[[as.integer(args[[2]]) + 1L]])
  v[[as.integer(args[[2]]) * length(p) + as.integer(args[[3]]) + 1L]]
}

# data_utility [function, args, data]: MGBT count, Box-Cox / Yeo-Johnson lambda +
# transform, plotting positions, Latin hypercube. Mirrors dispatch_data_utility in
# core/tests/test_fixtures.cpp.
dispatch_data_utility <- function(fn, args, data) {
  ns <- asNamespace("corehydror")
  switch(fn,
    MGBT = as.double(ns$ch_mgbt_test_(data)),
    BoxCoxLambda = ns$ch_box_cox_lambda_(data),
    BoxCoxTransform = ns$ch_box_cox_(data, args[[1]])[[as.integer(args[[2]]) + 1L]],
    YeoJohnsonLambda = ns$ch_yeo_johnson_lambda_(data),
    YeoJohnsonTransform = ns$ch_yeo_johnson_(data, args[[1]])[[as.integer(args[[2]]) + 1L]],
    PlottingPosition = ns$ch_plotting_positions_alpha_(
      as.integer(args[[1]]), args[[2]])[[as.integer(args[[3]]) + 1L]],
    LHSRandom = ,
    LHSMedian = {
      # args: [sample_size, dimension, seed, row, col]; glue returns row-major flat.
      flat <- ns$ch_latin_hypercube_(as.integer(args[[1]]), as.integer(args[[2]]),
        as.integer(args[[3]]), identical(fn, "LHSMedian"))
      flat[[as.integer(args[[4]]) * as.integer(args[[2]]) + as.integer(args[[5]]) + 1L]]
    },
    stop(sprintf("unknown data_utility function: %s", fn))
  )
}

# goodness_of_fit [function, args, observed_dataset, modeled_dataset]: routes through
# ch_toolbox_run_("gof", ...) (numerics/support/toolbox_runner.hpp), so this fixture kind and a
# user's goodness_of_fit()/aic() call are the same code path. Mirrors dispatch_gof in
# core/tests/test_fixtures.cpp.
kGofFunctionMethod <- c(
  MSE = "mse", MAE = "mae",
  NashSutcliffeEfficiency = "nse",
  KlingGuptaEfficiency = "kge", KlingGuptaEfficiencyMod = "kge_mod",
  PBIAS = "pbias", RSR = "rsr",
  IndexOfAgreement = "d", ModifiedIndexOfAgreement = "d_mod",
  RefinedIndexOfAgreement = "d_ref", VolumetricEfficiency = "ve"
)

dispatch_goodness_of_fit <- function(fn, args, obs, mod) {
  ns <- asNamespace("corehydror")
  if (identical(fn, "AIC")) {
    opts <- to_runner_json(list(k = as.integer(args[[1]]), log_likelihood = args[[2]]))
    return(ns$ch_toolbox_run_("gof", "aic", list(), opts)$values[[1]])
  }
  if (fn %in% c("AICc", "BIC")) {
    opts <- to_runner_json(list(n = as.integer(args[[1]]), k = as.integer(args[[2]]),
                                log_likelihood = args[[3]]))
    return(ns$ch_toolbox_run_("gof", if (identical(fn, "AICc")) "aicc" else "bic",
                              list(), opts)$values[[1]])
  }
  method <- kGofFunctionMethod[[fn]]
  if (is.null(method)) stop(sprintf("unknown goodness_of_fit function: %s", fn))
  ns$ch_toolbox_run_("gof", method, list(obs, mod), "{}")$values[[1]]
}

# Threshold-selection diagnostics, split out of the switch above because both methods share one
# glue call and differ only in which field the function name selects. args are
# [u_min, u_max, n_thresholds, confidence_level, point_index]; `*PointCount` ignores the index and
# returns how many candidate thresholds survived the minimum-exceedance and fit filters.
dispatch_threshold_diagnostic <- function(fn, args, data) {
  ns <- asNamespace("corehydror")
  is_mrl <- startsWith(fn, "MRL")
  r <- ns$ch_threshold_diagnostics_(
    data, if (is_mrl) "mean_residual_life" else "parameter_stability",
    args[[1]], args[[2]], as.integer(args[[3]]), args[[4]]
  )
  if (fn %in% c("MRLPointCount", "GPDStabilityPointCount")) {
    return(as.double(length(r$threshold)))
  }
  field <- switch(fn,
    MRLThreshold = "threshold", MRLMeanExcess = "mean_excess",
    MRLLowerCI = "lower_ci", MRLUpperCI = "upper_ci", MRLCount = "exceedance_count",
    GPDStabilityThreshold = "threshold",
    GPDStabilityModifiedScale = "modified_scale",
    GPDStabilityModifiedScaleLowerCI = "modified_scale_lower_ci",
    GPDStabilityModifiedScaleUpperCI = "modified_scale_upper_ci",
    GPDStabilityShape = "shape", GPDStabilityShapeLowerCI = "shape_lower_ci",
    GPDStabilityShapeUpperCI = "shape_upper_ci",
    GPDStabilityCount = "exceedance_count",
    stop(sprintf("unknown data_utility function: %s", fn))
  )
  as.double(r[[field]][[as.integer(args[[5]]) + 1L]])
}

# special_function/Correlation.* only: routes fixtures/special_functions/correlation.json's
# three Correlation targets through ch_toolbox_run_ (numerics/support/toolbox_runner.hpp) rather
# than adding bespoke glue, so the pinned pearson/spearman/kendalls_tau values become
# cross-language checks. Every other special_function target stays C++-only and unrouted here,
# as it was before this kind gained any R handling at all.
kCorrelationSpecialFunctionMethod <- c(
  "Correlation.pearson" = "pearson",
  "Correlation.spearman" = "spearman",
  "Correlation.kendalls_tau" = "kendall"
)

run_special_function_correlation_case <- function(target, args_raw) {
  ns <- asNamespace("corehydror")
  args <- vapply(args_raw, parse_num, numeric(1))
  mid <- length(args) %/% 2L
  x <- args[seq_len(mid)]
  y <- args[(mid + 1L):length(args)]
  method <- kCorrelationSpecialFunctionMethod[[target]]
  ns$ch_toolbox_run_("correlation", method, list(x, y), "{}")$values[[1]]
}

# special_function/{RunningStatistics,RunningCovariance,Fourier,Statistics.percentile} (Task 3):
# the same "route the pinned value through ch_toolbox_run_" pattern as Correlation.* above, one
# subset per family. Mirrors core/tests/test_fixtures.cpp's own routing exactly (see that file's
# running_covariance_toolbox()/running_covariance_element(), the RunningStatistics.* table
# entries, and the Fourier.{fft_at,real_fft_at,correlation_at} functions for the args
# conventions and the reasoning for which targets route and which stay C++-only:
# RunningStatistics's four population-normalized variants and its combined_*/clone_* cases have
# no run_statistics()-reachable equivalent, and Fourier.autocorrelation_at deliberately stays off
# this list because the toolbox "spectra.autocorrelation" method wraps the newer
# data::Autocorrelation class, not Fourier::autocorrelation itself).
kRunningStatisticsSpecialFunctionIndex <- c(
  "RunningStatistics.count" = 0L, "RunningStatistics.minimum" = 1L,
  "RunningStatistics.maximum" = 2L, "RunningStatistics.mean" = 3L,
  "RunningStatistics.variance" = 4L, "RunningStatistics.standard_deviation" = 5L,
  "RunningStatistics.coefficient_of_variation" = 6L, "RunningStatistics.skewness" = 7L,
  "RunningStatistics.kurtosis" = 8L
)

kRunningCovarianceSpecialFunctionBlock <- c(
  "RunningCovariance.mean_element" = 0L,
  "RunningCovariance.covariance_element" = 1L,
  "RunningCovariance.sample_covariance_element" = 2L,
  "RunningCovariance.sample_correlation_element" = 3L,
  "RunningCovariance.population_covariance_element" = 4L,
  "RunningCovariance.population_correlation_element" = 5L
)

kFourierToolboxMethod <- c(
  "Fourier.fft_at" = "dft",
  "Fourier.real_fft_at" = "dft_real",
  "Fourier.correlation_at" = "cross_correlation"
)

# special_function/{Histogram,Bilinear} (Task 4): the same routing pattern, one subset per
# family. Mirrors core/tests/test_fixtures.cpp's own routing exactly (see that file's Histogram.*
# table entries and bilinear_log_floor_value() for the args conventions). Histogram.data_count/
# get_bin_index_of have no toolbox-method equivalent (neither "statistics" nor "bins" exposes
# them) and Histogram.adapt_* needs AddData(), which the toolbox arm's stateless construction
# never calls, so those four stay unrouted -- the same "no run_toolbox-reachable equivalent"
# reasoning as RunningStatistics's population_* variants. Search.* also stays unrouted: neither
# "linear" nor "bilinear" returns a search index, only an interpolated y.
kHistogramStatisticsIndex <- c(
  "Histogram.mean" = 0L, "Histogram.median" = 1L, "Histogram.mode" = 2L,
  "Histogram.standard_deviation" = 3L, "Histogram.lower_bound" = 4L, "Histogram.upper_bound" = 5L,
  "Histogram.bin_width" = 6L, "Histogram.number_of_bins" = 7L
)
kHistogramBinsColumn <- c(
  "Histogram.bin_lower_bound_at" = 0L, "Histogram.bin_upper_bound_at" = 1L,
  "Histogram.bin_frequency_at" = 3L
)

kRoutedSpecialFunctionTargets <- c(
  names(kCorrelationSpecialFunctionMethod), names(kRunningStatisticsSpecialFunctionIndex),
  names(kRunningCovarianceSpecialFunctionBlock), names(kFourierToolboxMethod),
  "Statistics.percentile", names(kHistogramStatisticsIndex), names(kHistogramBinsColumn),
  "Bilinear.log_floor_value", "Probability.hpcm_joint", "DifferentialEvolution.best_value"
)

# special_function/Probability.hpcm_joint (Task 6): the same routing pattern, through the new
# "probability" toolbox group's "joint" method (dependency = "correlation"). Mirrors
# core/tests/test_fixtures.cpp's own routing exactly -- see that file's special_function_table()
# entry for "Probability.hpcm_joint" for the args convention (args = [p_0..p_(n-1),
# ind_0..ind_(n-1), corr(n*n flattened row-major)], n inferred from the argument count).
# Probability.hpcm_conditional_at stays unrouted: it needs the conditionalProbabilities
# out-value, which the "probability" toolbox group's "joint" method does not expose.
# special_function/DifferentialEvolution.best_value (Task 8): reuses
# fixtures/special_functions/differential_evolution.json (already pinned C++-only) rather than
# duplicating it -- routed through ch_optim_run_ so the callback path itself is exercised.
# args convention: [fn_id, direction, D, lower(D), upper(D), index] -- see
# core/tests/test_fixtures.cpp's differential_evolution_best_value() for the authoritative
# description. fn_id 0 = "quadratic" (sum_i (x_i - i)^2, 0-based i), 1 = "normal_loglik" (Normal
# log-likelihood of {9,10,11,12,13} at mean=p[1], sd=p[2]) -- NATIVE R closures reproducing the
# same two P3.3 numerical_derivative fixture functions, so this case exercises the real R
# callback path. `value` un-applies OptimResult's raw-sign convention back to the C#
# BestParameterSet.Fitness this fixture's literals were curated against (see
# differential_evolution_best_value()'s own comment for why).
run_special_function_differential_evolution_case <- function(args_raw) {
  ns <- asNamespace("corehydror")
  a <- vapply(args_raw, parse_num, numeric(1))
  fn_id <- as.integer(a[1])
  direction <- as.integer(a[2])
  D <- as.integer(a[3])
  lower <- a[4:(3 + D)]
  upper <- a[(4 + D):(3 + 2 * D)]
  index <- as.integer(a[4 + 2 * D])
  sample <- c(9, 10, 11, 12, 13)
  objective <- if (fn_id == 0) {
    function(p) sum((p - (seq_along(p) - 1))^2)
  } else {
    # Explicit Normal log-density formula (not dnorm(log = TRUE)) so a chaotic DE search sees the
    # SAME density implementation R and Python -- see corehydropy's test_fixtures.py mirror.
    function(p) sum(-0.5 * ((sample - p[1]) / p[2])^2 - log(sqrt(2 * pi) * p[2]))
  }
  spec <- ns$to_spec_json(list(method = "de", lower = ns$spec_array(lower),
                               upper = ns$spec_array(upper), maximize = (direction == 1)))
  r <- ns$ch_optim_run_(spec, objective)
  if (index == D) {
    if (direction == 1) -r$value else r$value
  } else {
    r$parameters[[index + 1L]]
  }
}

run_special_function_probability_hpcm_joint_case <- function(args_raw) {
  ns <- asNamespace("corehydror")
  args <- vapply(args_raw, parse_num, numeric(1))
  n <- 0L
  for (candidate in 1:20) {
    if (2L * candidate + candidate * candidate == length(args)) { n <- candidate; break }
  }
  if (n == 0L) stop("cannot infer n for Probability.hpcm args")
  p <- args[seq_len(n)]
  ind <- args[(n + 1L):(2L * n)]
  corr <- args[(2L * n + 1L):length(args)]
  ns$ch_toolbox_run_("probability", "joint", list(p, ind, corr),
                     '{"dependency":"correlation"}')$values[[1]]
}

# Histogram fixture args convention: args = [explicit_bins, data...] for the whole-histogram
# scalar targets; the bin_*_at targets append one trailing 0-based bin-index probe. An empty R
# list serializes to "[]", not "{}" (jsonlite's unnamed-list rule -- see to_runner_json's own
# comment above), so explicit_bins == 0 returns the literal "{}" rather than to_runner_json(list()).
histogram_toolbox_options <- function(explicit_bins) {
  if (explicit_bins > 0) to_runner_json(list(bins = as.integer(explicit_bins))) else "{}"
}

run_special_function_case <- function(target, args_raw) {
  ns <- asNamespace("corehydror")
  args <- vapply(args_raw, parse_num, numeric(1))

  if (target %in% names(kCorrelationSpecialFunctionMethod)) {
    return(run_special_function_correlation_case(target, args_raw))
  }
  if (identical(target, "Statistics.percentile")) {
    n <- length(args) - 2L
    data <- args[seq_len(n)]
    k <- args[[n + 1L]]
    sorted <- args[[n + 2L]] != 0
    opts <- if (sorted) '{"sorted":true}' else "{}"
    return(ns$ch_toolbox_run_("statistics", "percentile", list(data, k), opts)$values[[1]])
  }
  if (target %in% names(kRunningStatisticsSpecialFunctionIndex)) {
    r <- ns$ch_toolbox_run_("statistics", "summary", list(args), "{}")
    return(r$values[[kRunningStatisticsSpecialFunctionIndex[[target]] + 1L]])
  }
  if (target %in% names(kRunningCovarianceSpecialFunctionBlock)) {
    size <- as.integer(args[[1]])
    num_pushes <- as.integer(args[[2]])
    cols <- lapply(seq_len(size), function(j) {
      vapply(seq_len(num_pushes), function(p) args[[2L + (p - 1L) * size + j]], numeric(1))
    })
    r <- ns$ch_toolbox_run_("statistics", "running_covariance", cols, "{}")
    base <- 2L + num_pushes * size
    i <- as.integer(args[[base + 1L]])
    block <- kRunningCovarianceSpecialFunctionBlock[[target]]
    if (block == 0L) return(r$values[[1L + i + 1L]])
    j <- as.integer(args[[base + 2L]])
    offset <- 1L + size + (block - 1L) * (size * size)
    return(r$values[[offset + i * size + j + 1L]])
  }
  if (target %in% names(kFourierToolboxMethod)) {
    if (identical(target, "Fourier.correlation_at")) {
      n <- (length(args) - 1L) %/% 2L
      data1 <- args[seq_len(n)]
      data2 <- args[(n + 1L):(2L * n)]
      index <- as.integer(args[[2L * n + 1L]])
      r <- ns$ch_toolbox_run_("spectra", "cross_correlation", list(data1, data2), "{}")
      return(r$values[[index + 1L]])
    }
    n <- length(args) - 2L
    data <- args[seq_len(n)]
    inverse <- args[[n + 1L]] != 0
    index <- as.integer(args[[n + 2L]])
    opts <- if (inverse) '{"inverse":true}' else "{}"
    r <- ns$ch_toolbox_run_("spectra", kFourierToolboxMethod[[target]], list(data), opts)
    return(r$values[[index + 1L]])
  }
  if (target %in% names(kHistogramStatisticsIndex)) {
    explicit_bins <- as.integer(args[[1]])
    data <- args[-1]
    opts <- histogram_toolbox_options(explicit_bins)
    r <- ns$ch_toolbox_run_("histogram", "statistics", list(data), opts)
    return(r$values[[kHistogramStatisticsIndex[[target]] + 1L]])
  }
  if (target %in% names(kHistogramBinsColumn)) {
    explicit_bins <- as.integer(args[[1]])
    probe <- as.integer(args[[length(args)]])
    data <- args[-c(1, length(args))]
    opts <- histogram_toolbox_options(explicit_bins)
    r <- ns$ch_toolbox_run_("histogram", "bins", list(data), opts)
    return(r$values[[probe * 4L + kHistogramBinsColumn[[target]] + 1L]])
  }
  if (identical(target, "Bilinear.log_floor_value")) {
    coords <- c(0, 1e-15, 1)
    flat <- c(0, 0, 0, 1e-15, 1e-15, 1e-15, 1, 1, 1)
    opts <- '{"x1_transform":"log","x2_transform":"log","y_transform":"log"}'
    r <- ns$ch_toolbox_run_("interpolation", "bilinear",
                            list(coords, coords, flat, args[1], args[2]), opts)
    return(r$values[[1]])
  }
  if (identical(target, "Probability.hpcm_joint")) {
    return(run_special_function_probability_hpcm_joint_case(args_raw))
  }
  if (identical(target, "DifferentialEvolution.best_value")) {
    return(run_special_function_differential_evolution_case(args_raw))
  }
  stop(sprintf("unrouted special_function target: %s", target))
}

# optimizer [construct carries method/lower/upper/initial/maximize/seed/control; assertions carry
# value/parameter/status]: the six ported optimizers (Task 8), run through ch_optim_run_ against a
# NATIVE R closure -- not ch_toolbox_run_ -- because an optimizer's input is a live function, not
# serializable data. Mirrors run_optimizer_kind in core/tests/test_fixtures.cpp. `construct` was
# parsed with simplifyVector = FALSE, so a JSON array leaf is an R list of scalars; re-flattened
# here (rather than round-tripped through jsonlite::toJSON) so a length-1 array (e.g. "brent"'s
# single-element lower/upper) survives as a JSON array, not an auto-unboxed scalar.
optimizer_spec_json <- function(ns, construct) {
  num_vec <- function(x) if (is.null(x)) NULL else ns$spec_array(vapply(x, as.double, numeric(1)))
  ns$to_spec_json(list(
    method = construct$method,
    lower = num_vec(construct$lower), upper = num_vec(construct$upper),
    initial = num_vec(construct$initial),
    maximize = if (is.null(construct$maximize)) NULL else isTRUE(construct$maximize),
    seed = if (is.null(construct$seed)) NULL else as.integer(construct$seed),
    control = if (is.null(construct$control) || length(construct$control) == 0L) NULL
              else construct$control
  ))
}

# The handful of TestFunctions.cs objectives fixtures/toolbox/optimizers.json names by string --
# NATIVE R closures reproducing the same formulas as core/tests/optimization_test_functions.hpp,
# so every optimizer fixture case exercises the real R callback path
# (corehydro::numerics::support::GuardedObjective exists to protect exactly this call).
optimizer_fixture_objective <- function(name) {
  switch(name,
    FXYZ = function(p) (4 * p[1] - 0.5)^2 + (3 * p[2] - 0.6)^2 + (2 * p[3] - 0.7)^2,
    DeJong = function(p) sum(p^2),
    Booth = function(p) (p[1] + 2 * p[2] - 7)^2 + (2 * p[1] + p[2] - 5)^2,
    McCormick = function(p) sin(p[1] + p[2]) + (p[1] - p[2])^2 - 1.5 * p[1] + 2.5 * p[2] + 1,
    FX = function(p) (p[1] + 3) * (p[1] - 1)^2,
    stop(sprintf("unknown optimizer fixture objective: %s", name))
  )
}

# callback [construct carries group/method/callback/options; assertions carry value/dim/status]:
# the ported routines whose input is a live function, run through ch_callback_math_ against a
# NATIVE R closure. Mirrors run_callback_kind in core/tests/test_fixtures.cpp. NOTE these catalog
# names are NOT the optimizer catalog's above: `Diff_FXYZ` is Test_Differentiation.FXYZ
# (x^3 + y^4 + z^5), unrelated to the optimizer catalog's `FXYZ`. `Diff_FX` and `Quad_FX3` are both
# x^3, from two different upstream test files -- hence the prefixes.
callback_fixture_function <- function(name) {
  switch(name,
    Root_Quadratic = function(x) x^2 - 2,
    # P2 "math extras": TestFunctions.Quadratic_Deriv, the newton catalog's counterpart of
    # Root_Quadratic.
    RootD_Quadratic = function(x) 2 * x,
    Root_Cubic = function(x) x^3 - x - 1,
    # TestFunctions.Trigonometric, root ~1.12191713 on [0, pi].
    Root_Trigonometric = function(x) 2 * sin(x) - 3 * cos(x) - 0.5,
    # TestFunctions.Trigonometric_Deriv.
    RootD_Trigonometric = function(x) 2 * cos(x) + 3 * sin(x),
    # Test_NewtonRaphson.Test_Multi_LinearSystem's system, F([x;y]) = [3x + y - 9, x + 2y - 8],
    # whose unique root is [2, 3].
    Sys_Linear_F = function(v) c(3 * v[1] + v[2] - 9, v[1] + 2 * v[2] - 8),
    # The (constant) Jacobian of Sys_Linear_F, one ROW per equation: [[3, 1], [1, 2]]. matrix()
    # fills COLUMN-major; the glue transposes to row-major, and this matrix is symmetric anyway.
    Sys_Linear_J = function(v) matrix(c(3, 1, 1, 2), nrow = 2, ncol = 2),
    Diff_FX = function(x) x^3,
    Diff_FXY = function(p) p[1]^2 * p[2]^3,
    Diff_FXYZ = function(p) p[1]^3 + p[2]^4 + p[3]^5,
    Diff_FH = function(p) p[1]^3 - 2 * p[1] * p[2] - p[2]^6,
    Quad_FX3 = function(x) x^3,
    Quad_Cosine = function(x) cos(x),
    Quad_Sine = function(x) sin(x),
    Quad_FXX = function(x) 0.5 + 24 * x + 3 * x^2,
    Quad_FXXX = function(x) 0.5 + 24 * x + 3 * x^2 + 8 * x^3,
    # corehydro addition, no upstream integrand -- the one callback that reaches the subdividing
    # branch of the recursion. Written with `x * x` rather than `x^2`, arithmetic only, so all
    # four runners agree bit for bit and the evaluation count is a real oracle.
    Quad_Peak = function(x) 1 / (1 + 1e4 * x * x),
    # The mcmc catalog (fixtures/callback/mcmc.json). Both log-densities are arithmetic only and
    # sum in an explicit `for` loop rather than through sum(): R's sum() accumulates in extended
    # precision where C++, Python and C# accumulate in double, and a Markov chain turns one
    # differing bit into a different chain outright. See the fixture's own note.
    Mcmc_GaussianKernel = function(p) {
      data <- c(4.9, 5.1, 5.0, 5.2, 4.8)
      acc <- 0
      for (x in data) acc <- acc + (x - p[1]) * (x - p[1])
      -0.5 * acc
    },
    Mcmc_LinearKernel = function(p) {
      t <- c(1, 2, 3, 4, 5, 6, 7, 8)
      y <- c(2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1)
      acc <- 0
      for (i in seq_along(t)) {
        residual <- y[i] - p[1] - p[2] * t[i]
        acc <- acc + residual * residual
      }
      -0.5 * acc
    },
    # The Gibbs case's model, whose full conditional really IS uniform: with
    # x_i ~ Uniform(mu - 1, mu + 1) and a flat prior, mu given the data is
    # Uniform(max(x) - 1, min(x) + 1), so Prop_UniformConditional below is an exact Gibbs step
    # rather than a random walk wearing Gibbs's name. Comparisons and arithmetic only.
    Mcmc_UniformWidthKernel = function(p) {
      data <- c(4.9, 5.1, 5.0, 5.2, 4.8)
      for (x in data) {
        if (x - p[1] > 1 || p[1] - x > 1) return(-Inf)
      }
      0
    },
    # The proposal catalog: (parameters, rng) -> parameters, upstream's Gibbs.Proposal shape.
    # Draws through the HANDLE, exactly as a user's own proposal would.
    Prop_UniformConditional = function(parameters, rng) {
      data <- c(4.9, 5.1, 5.0, 5.2, 4.8)
      lo <- max(data) - 1
      hi <- min(data) + 1
      lo + rng_uniform(rng, 1) * (hi - lo)
    },
    # The TWO-parameter member of the proposal catalog, written for the second case of
    # fixtures/callback/callback_cross_language.json. An INDEPENDENCE proposal: it ignores the
    # state it is handed and draws each parameter from a fixed interval, lo + u * (hi - lo),
    # exactly as Prop_UniformConditional above does for one. Gibbs accepts every proposal, so the
    # chain is a sequence of independent draws from that box and the only mark the log-density
    # leaves on the run is the fitness it reports. One rng_uniform(rng, 2) call, not two of length
    # one: a single call cannot split the stream.
    Prop_UniformBox = function(parameters, rng) {
      lo <- c(-1, 1.5)
      hi <- c(1, 2.5)
      u <- rng_uniform(rng, 2)
      out <- numeric(2)
      for (j in 1:2) out[j] <- lo[j] + u[j] * (hi[j] - lo[j])
      out
    },
    # The gradient catalog: (parameters) -> vector, upstream's HMC.Gradient shape. The analytic
    # derivative of Mcmc_GaussianKernel, d/dmu = sum(x - mu), summed in an explicit loop.
    Grad_GaussianKernel = function(p) {
      data <- c(4.9, 5.1, 5.0, 5.2, 4.8)
      acc <- 0
      for (x in data) acc <- acc + (x - p[1])
      acc
    },
    # Task-5 review fix, coverage finding: unlike Mcmc_GaussianKernel, whose derivative
    # sum(x - mu) is LINEAR in mu (so its third derivative is zero and the ported
    # central-difference default agrees with the analytic gradient to rounding, ~4e-16), this
    # kernel's derivative is CUBIC in mu, so the central-difference truncation error is real rather
    # than rounding -- the analytic and default gradients genuinely disagree, which is what a
    # supplied-vs-ignored gradient regression needs to be caught by the oracle gate. The 0.05
    # coefficient is load-bearing, not decorative: measured by brute-force sweep (a coefficient of
    # 1 over the same data), an unscaled quartic makes HMC's leapfrog trajectory genuinely CHAOTIC
    # over 200 iterations -- the analytic and default gradients then diverge at the ~0.3% level, the
    # same order this fixture's own cross-language divergence measured at that scale, so no fixed
    # tolerance could pin it. At 0.05 the divergence is small and smooth (~3e-8 relative, four
    # orders past the file's 1e-12 tolerance) rather than chaotic, which is what keeps the case
    # usable as an oracle at all.
    Mcmc_QuarticKernel = function(p) {
      data <- c(4.9, 5.1, 5.0, 5.2, 4.8)
      acc <- 0
      for (x in data) {
        d <- x - p[1]
        acc <- acc + d * d * d * d
      }
      -0.05 * acc
    },
    # The analytic derivative of Mcmc_QuarticKernel, d/dmu = 0.05 * 4 * sum((x - mu)^3).
    Grad_QuarticKernel = function(p) {
      data <- c(4.9, 5.1, 5.0, 5.2, 4.8)
      acc <- 0
      for (x in data) {
        d <- x - p[1]
        acc <- acc + d * d * d
      }
      0.2 * acc
    },
    # The bootstrap catalog (fixtures/callback/bootstrap.json), upstream's four Bootstrap<TData>
    # delegate shapes. Every one is arithmetic and comparisons only, and the mean is summed in an
    # explicit `for` loop rather than with sum() or mean(): both accumulate in extended precision in
    # R where C++, Python and C# accumulate in double, and one differing bit in a fitted mean moves
    # a percentile. The resample draws every index through the HANDLE, exactly as a user's own
    # resample function does; rng_integers draws on [0, n) counting from 0, so the index is shifted
    # by one for R's own 1-based subscript.
    Resample_Iid = function(data, parameters, rng) {
      data[rng_integers(rng, length(data), 0, length(data)) + 1L]
    },
    Fit_Mean = function(data) {
      acc <- 0
      for (x in data) acc <- acc + x
      acc / length(data)
    },
    # The CONTRACTION-BEARING member of the bootstrap catalog, written for the second case of
    # fixtures/callback/callback_cross_language.json: the ordinary least-squares line of the sample
    # against its position t = 1..n, in the centered form
    #   slope = sum(dt * dy) / sum(dt * dt),   intercept = ybar - slope * tbar
    # returned as c(intercept, slope). Every accumulation is acc + a * b, the shape clang and gcc
    # fuse into a multiply-add by default, and the intercept subtracts two nearly equal quantities
    # on top of it. R never fuses, so this closure computes the written arithmetic; the C++ catalog
    # matches it only because core/CMakeLists.txt turns contraction off for that file, and the case
    # naming this callback is what fails if that flag is removed. Explicit loops rather than sum()
    # or mean(), for the same reason Fit_Mean above uses one.
    Fit_LinearTrend = function(data) {
      n <- length(data)
      st <- 0
      sy <- 0
      for (i in seq_len(n)) {
        st <- st + as.double(i)
        sy <- sy + data[i]
      }
      tbar <- st / n
      ybar <- sy / n
      num <- 0
      den <- 0
      for (i in seq_len(n)) {
        dt <- as.double(i) - tbar
        dy <- data[i] - ybar
        num <- num + dt * dy
        den <- den + dt * dt
      }
      slope <- num / den
      c(ybar - slope * tbar, slope)
    },
    # The PIVOTAL member of the bootstrap catalog: upstream's Func<TData, BootstrapFit>
    # FitWithCovarianceFunction, the delegate that run type fits through. The model is the
    # two-parameter Normal location-scale MLE -- theta = (mu, sigma) with sigma the POPULATION
    # standard deviation -- whose covariance is analytic, diag(s2 / n, s2 / (2n)), so the whole
    # callback is arithmetic plus one sqrt. sqrt is the one libm function IEEE 754 requires to be
    # correctly rounded, so unlike log or exp it is the same value in all four runners; the sums are
    # explicit loops for the reason Fit_Mean above gives. `ss <- ss + (x - mu) * (x - mu)` is itself
    # a contraction-bearing shape, so this zero-tolerance guarantee also depends on the C++
    # catalog's own -ffp-contract=off scoping in core/CMakeLists.txt.
    FitCov_NormalMLE = function(data) {
      n <- length(data)
      acc <- 0
      for (x in data) acc <- acc + x
      mu <- acc / n
      ss <- 0
      for (x in data) ss <- ss + (x - mu) * (x - mu)
      s2 <- ss / n
      list(
        parameters = c(mu, sqrt(s2)),
        covariance = matrix(c(s2 / n, 0, 0, s2 / (2 * n)), nrow = 2L, ncol = 2L)
      )
    },
    Stat_Identity = function(parameters) parameters,
    Stat_MeanAndSquare = function(parameters) c(parameters[1], parameters[1] * parameters[1]),
    # `index` counts from 0, as the ported delegate does, so the sample without it is
    # data[-(index + 1)] -- the naive data[-index] is data[-0], which R evaluates to numeric(0),
    # the empty vector, when index is 0, and drops the wrong observation at every later index.
    Jack_LeaveOneOut = function(data, index) data[-(index + 1)],
    # The gmm catalog (fixtures/callback/gmm.json), upstream's three delegate shapes from
    # GeneralizedMethodOfMoments's delegate constructor. The model is the just-identified
    # two-parameter method-of-moments fit of a Normal: theta = (mu, sigma2) and
    #   g = [mean(x - mu), mean((x - mu)^2 - sigma2)],  S = the covariance of those two
    # whose unique root -- and so the GMM optimum, since q = p makes g = 0 attainable -- is the
    # sample mean and the population variance. Arithmetic and an explicit `for` loop only, never
    # sum() or mean(): both accumulate in extended precision in R where C++, Python and C#
    # accumulate in double, and one differing bit moves a fitted parameter.
    Mom_NormalMeanVariance = function(p) {
      data <- c(4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7)
      n <- 8
      g0 <- 0
      g1 <- 0
      s00 <- 0
      s01 <- 0
      s11 <- 0
      for (x in data) {
        a <- x - p[1]
        b <- a * a - p[2]
        g0 <- g0 + a
        g1 <- g1 + b
        s00 <- s00 + a * a
        s01 <- s01 + a * b
        s11 <- s11 + b * b
      }
      list(
        g = c(g0 / n, g1 / n),
        # matrix() fills COLUMN-major; this one is symmetric, and the glue transposes anyway.
        s = matrix(c(s00 / n, s01 / n, s01 / n, s11 / n), nrow = 2, ncol = 2)
      )
    },
    # The OVER-IDENTIFIED member of the same catalog: the identical Normal model and the identical
    # eight observations, with a third moment condition added -- mean((x - mu)^3), zero for a
    # Normal -- so q = 3 > p = 2 and the degrees of freedom become 1. The only case in the file
    # that reaches the chi-squared p-value branch of the J-statistic.
    Mom_NormalThreeMoments = function(p) {
      data <- c(4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7)
      n <- 8
      g0 <- 0
      g1 <- 0
      g2 <- 0
      s00 <- 0
      s01 <- 0
      s02 <- 0
      s11 <- 0
      s12 <- 0
      s22 <- 0
      for (x in data) {
        a <- x - p[1]
        b <- a * a - p[2]
        cc <- a * a * a
        g0 <- g0 + a
        g1 <- g1 + b
        g2 <- g2 + cc
        s00 <- s00 + a * a
        s01 <- s01 + a * b
        s02 <- s02 + a * cc
        s11 <- s11 + b * b
        s12 <- s12 + b * cc
        s22 <- s22 + cc * cc
      }
      list(
        g = c(g0 / n, g1 / n, g2 / n),
        # matrix() fills COLUMN-major; this one is symmetric, and the glue transposes anyway.
        s = matrix(
          c(s00 / n, s01 / n, s02 / n, s01 / n, s11 / n, s12 / n, s02 / n, s12 / n, s22 / n),
          nrow = 3, ncol = 3
        )
      )
    },
    # The CUBIC-JACOBIAN member of the same catalog, and the one case in the file whose analytic
    # Jacobian is distinguishable from the ported numerical one. theta = (mu, sigma), matched on
    # the first and fourth central moments of a Normal:
    #   g = [mean(x - mu), mean(u^4) - 3 t^4],  u = 100 (x - mu),  t = 100 sigma
    # so dg2/dsigma = -1200 t^3 is cubic in the parameter, where every other case in the file has a
    # Jacobian linear in it. The eight observations are the ones above with the decimal point moved
    # two places, which is what makes the fitted sigma (0.00404) small next to the numerical
    # Jacobian's step h = 1e-4 (|theta| + 1).
    Mom_NormalFourthMoment = function(p) {
      data <- c(0.041, 0.052, 0.048, 0.055, 0.049, 0.051, 0.053, 0.047)
      n <- 8
      t <- p[2] * 100
      t4 <- 3 * t * t * t * t
      g0 <- 0
      g1 <- 0
      s00 <- 0
      s01 <- 0
      s11 <- 0
      for (x in data) {
        a <- x - p[1]
        u <- a * 100
        b <- u * u * u * u - t4
        g0 <- g0 + a
        g1 <- g1 + b
        s00 <- s00 + a * a
        s01 <- s01 + a * b
        s11 <- s11 + b * b
      }
      list(
        g = c(g0 / n, g1 / n),
        # matrix() fills COLUMN-major; this one is symmetric, and the glue transposes anyway.
        s = matrix(c(s00 / n, s01 / n, s01 / n, s11 / n), nrow = 2, ncol = 2)
      )
    },
    # The analytic Jacobian of Mom_NormalFourthMoment, one ROW per moment condition:
    # dg1/dmu = -1, dg1/dsigma = 0, dg2/dmu = -400 mean(u^3), dg2/dsigma = -1200 t^3.
    Jac_NormalFourthMoment = function(p) {
      data <- c(0.041, 0.052, 0.048, 0.055, 0.049, 0.051, 0.053, 0.047)
      acc <- 0
      for (x in data) {
        u <- (x - p[1]) * 100
        acc <- acc + u * u * u
      }
      t <- p[2] * 100
      matrix(c(-1, -400 * acc / 8, 0, -1200 * t * t * t), nrow = 2, ncol = 2)
    },
    # The analytic Jacobian of Mom_NormalMeanVariance, one ROW per moment condition:
    # dg1/dmu = -1, dg1/dsigma2 = 0, dg2/dmu = -2 mean(x - mu), dg2/dsigma2 = -1.
    Jac_NormalMeanVariance = function(p) {
      data <- c(4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7)
      acc <- 0
      for (x in data) acc <- acc + (x - p[1])
      matrix(c(-1, -2 * acc / 8, 0, -1), nrow = 2, ncol = 2)
    },
    # A ridge penalty pulling sigma2 towards 1, carrying its own 1/2 as the ported half-quadratic
    # convention expects.
    Pen_SigmaTowardsOne = function(p) 0.5 * (p[2] - 1) * (p[2] - 1),
    # The rng catalog (fixtures/callback/rng_handle.json): two arguments, (parameters, rng), the
    # Gibbs proposal's own signature. Each draws through the HANDLE it is given -- exactly what a
    # user's proposal function would do -- rather than reaching for a generator of its own, which
    # is the property the fixture exists to pin.
    Rng_Uniform = function(parameters, rng) rng_uniform(rng, parameters[1]),
    Rng_Integers = function(parameters, rng) {
      rng_integers(rng, parameters[1], parameters[2], parameters[3])
    },
    Rng_Interleaved = function(parameters, rng) {
      c(rng_uniform(rng, 2), rng_integers(rng, 2, 0, 100), rng_uniform(rng, 1))
    },
    Rng_Warmup1000 = function(parameters, rng) {
      rng_uniform(rng, 1000)  # discarded, as upstream's own test discards 1000 GenRandInt32
      rng_uniform(rng, 10)
    },
    stop(sprintf("unknown callback fixture callback: %s", name))
  )
}

# `construct$options` was parsed with simplifyVector = FALSE, so a JSON array leaf is an R list of
# scalars; re-flattened here so a length-1 array survives as a JSON array, not an auto-unboxed
# scalar. Recursive because the mcmc group's options carry strings (the sampler name) and an array
# of prior OBJECTS beside their numbers.
callback_options_value <- function(ns, v) {
  # A JSON null INSIDE an array is a value, not an absent key: the bootstrap group's
  # `pivotal_links` spells "the identity link for this parameter" that way, and jsonlite hands it
  # over as a NULL element. Returned as NULL so to_spec_json() emits `null` for it -- without this
  # it would fall through to as.double(NULL), i.e. numeric(0), and reach C++ as an empty array.
  if (is.null(v)) {
    return(NULL)
  }
  if (is.list(v)) {
    nms <- names(v)
    if (!is.null(nms) && all(nzchar(nms))) {
      return(lapply(v, function(e) callback_options_value(ns, e)))
    }
    if (all(vapply(v, function(e) !is.list(e) && is.numeric(e) && length(e) == 1L, logical(1)))) {
      return(ns$spec_array(vapply(v, as.double, numeric(1))))
    }
    return(lapply(v, function(e) callback_options_value(ns, e)))
  }
  if (is.character(v)) return(as.character(v))
  if (is.logical(v)) return(as.logical(v))
  as.double(v)
}

callback_options_json <- function(ns, options) {
  if (is.null(options) || length(options) == 0L) return("{}")
  ns$to_spec_json(lapply(options, function(v) callback_options_value(ns, v)))
}

# toolbox [group, data, options; assertions carry method/index/label/select]: every Numerics
# utility group runs through ch_toolbox_run_. Mirrors run_toolbox_kind in
# core/tests/test_fixtures.cpp.
toolbox_case_data <- function(case, datasets) {
  if (is.null(case$data)) {
    return(list())
  }
  lapply(case$data, function(d) {
    if (is.character(d)) as.double(unlist(datasets[[d]])) else as.double(unlist(d))
  })
}

toolbox_select <- function(r, a, group) {
  select <- if (is.null(a$select)) "value" else a$select
  if (identical(select, "length")) {
    return(as.double(length(r$values)))
  }
  if (identical(select, "rows")) {
    if (is.null(r$dims) || length(r$dims) < 1) {
      stop(sprintf("toolbox select 'rows' has no dims (group '%s')", group))
    }
    return(as.double(r$dims[[1]]))
  }
  if (identical(select, "columns")) {
    if (is.null(r$dims) || length(r$dims) < 2) {
      stop(sprintf("toolbox select 'columns' has no dims (group '%s')", group))
    }
    return(as.double(r$dims[[2]]))
  }
  i <- if (!is.null(a$label)) match(a$label, r$names) else (if (is.null(a$index)) 0L else as.integer(a$index)) + 1L
  if (is.na(i) || i > length(r$values)) {
    stop("toolbox result selection out of range")
  }
  as.double(r$values[[i]])
}

check_assertion <- function(actual, a) {
  mode <- a$mode
  if (mode == "bool") {
    expect_identical(as.logical(actual), a$expected)
  } else if (mode == "equal") {
    e <- parse_num(a$expected)
    if (is.nan(e)) expect_true(is.nan(actual)) else expect_equal(actual, e)
  } else if (mode == "abs") {
    expect_lte(abs(actual - a$expected), a$tol)
  } else if (mode == "rel") {
    expect_lte(abs(actual - a$expected) / abs(a$expected), a$tol)
  } else {
    stop(sprintf("unknown comparison mode: %s", mode))
  }
}

# --- Delegation to the shared distribution runner ----------------------------------------
#
# The fixture `construct` schema IS the dist_spec.hpp grammar, so the bridge only has to
# resolve a dataset NAME into an inline array, spell the handful of keys the two schemas
# disagree on, and hand the object to ch_dist_spec_run_ / ch_copula_run_ / ch_mvdist_run_.
# Every value the runner produces is then pinned by exactly the corpus the bespoke glue was
# pinned by. This mirrors core/tests/test_fixtures.cpp's delegation section case for case, so
# the C++, R and Python runners all reach the oracle through one shared code path.
#
# Two properties of the runner keep a narrow bespoke path alive; both are deliberate:
#
#   1. json_lite, the runner's JSON reader, has no NaN or Infinity literal, while the corpus
#      deliberately pins non-finite-PARAMETER validity cases: Empirical `p`, KernelDensity
#      `bandwidth`, every copula `theta`/`df`, and BivariateEmpirical `x1`/`x2`/`p`. Every such
#      case asserts nothing but `parameters_valid`, so each keeps a narrow glue call. The same
#      limit applies to a non-finite EVALUATION POINT, not just the construct: the
#      MultivariateNormal / MultivariateStudentT log_pdf-at-infinity cases and
#      r_mvtnorm_4d_sequential's infinite CDF bounds.
#
#   2. The runner is stateless by construction -- one call builds an object, evaluates once and
#      drops it -- and it exposes the user-facing verb set, not the whole fixture vocabulary.
#      Two groups therefore stay bespoke: methods with no runner counterpart (`mvndst` and its
#      two status arms, `log_multivariate_beta`, `cdf_xy(_after_set_parameters)`,
#      `dependency_change`), and MultivariateNormal's `cdf`/`interval` in a case that consumes
#      the persistent MVNUNI stream more than once, since those pin a SEQUENCE off one object.
#      A case that consumes the stream at most once delegates: `seed` is a grammar key, so a
#      rebuilt object starts the same stream. Only `cdf` has such a case today, so only `cdf`
#      has a batch entry point below.

has_non_finite <- function(v) {
  if (is.character(v)) return(any(v %in% c("nan", "inf", "-inf")))
  if (is.list(v)) return(any(vapply(v, has_non_finite, logical(1))))
  FALSE
}

# Both of the runner's string inputs (the spec and the args array). digits = I(17) round-trips
# a double exactly, matching run_estimation_case's model_json convention.
to_runner_json <- function(x) {
  as.character(jsonlite::toJSON(x, auto_unbox = TRUE, digits = I(17)))
}

# --- Composite distribution path -------------------------------------------------------
# TruncatedDistribution / Empirical / KernelDensity / Mixture / CompetingRisks: the fixture
# "construct" already IS the dist_spec.hpp grammar (which accepts `target`/`params` as aliases
# of `family`/`parameters`), so a composite case serializes straight through ch_dist_spec_run_.
# Adding a composite needs no change here at all.

kCompositeTargets <- c("TruncatedDistribution", "Empirical", "KernelDensity", "Mixture",
                       "CompetingRisks")

# "data" is the only dataset-by-name key in the univariate grammar (KernelDensity); "base" and
# "components" nest and carry no dataset reference.
composite_spec <- function(target, construct, datasets) {
  construct$family <- target
  if (is.character(construct$data)) construct$data <- datasets[[construct$data]]
  construct
}

# The fixture method vocabulary predates the runner's; map the differences in one place.
# `random_value` args are [sample_size, seed, index]: the runner's "random" reads only the
# first two and returns the whole draw, so the args pass through unchanged and fixture_pick
# does the indexing.
fixture_method <- function(m) {
  switch(m,
    param = "parameters",
    random_value = "random",
    m  # pdf, log_pdf, cdf, quantile, mean, ..., log_likelihood pass straight through
  )
}

# `param` and `random_value` index into the vector the runner returns whole.
fixture_pick <- function(r, method, args) {
  if (identical(method, "param")) return(r$values[[as.integer(args[[1]]) + 1L]])
  if (identical(method, "random_value")) return(r$values[[as.integer(args[[3]]) + 1L]])
  r$values[[1]]
}

# Limitations 1 and 2 for the composite path, and the only remaining callers of the bespoke
# composite glue in dist.cpp: Empirical's non-finite `p` and KernelDensity's non-finite
# `bandwidth` validity cases (which the grammar's JSON reader cannot encode), plus
# CompetingRisks' `dependency_change` (which has no runner counterpart).
dispatch_composite_local <- function(target, construct, datasets, method, args) {
  ns <- asNamespace("corehydror")
  if (target == "Empirical" && identical(method, "parameters_valid")) {
    pt <- if (!is.null(construct$p_transform)) construct$p_transform else "NormalZ"
    # v2.1.4: p_descending DECLARES the probability order (mirrors C#'s explicit
    # `probabilityOrder` argument -- NOT auto-detected from the data); default FALSE matches
    # the ordinary ascending-CDF case.
    pd <- if (!is.null(construct$p_descending)) as.logical(construct$p_descending) else FALSE
    return(ns$ch_emp_valid_(as.double(unlist(construct$x)), as.double(unlist(construct$p)),
                             pt, pd))
  }
  if (target == "KernelDensity" && identical(method, "parameters_valid")) {
    kernel    <- if (!is.null(construct$kernel)) construct$kernel else "Gaussian"
    # A negative bandwidth means Silverman's rule; as.double (not parse_num) parses the
    # "nan"/"inf" literal these two cases exist to reject.
    bandwidth <- if (!is.null(construct$bandwidth)) as.double(construct$bandwidth) else -1.0
    bounded   <- if (!is.null(construct$bounded_by_data)) as.logical(construct$bounded_by_data) else TRUE
    return(ns$ch_kde_valid_(as.double(unlist(datasets[[construct$data]])), kernel, bandwidth,
                             bounded))
  }
  if (target == "CompetingRisks" && identical(method, "dependency_change")) {
    # v2.1.4: verifies the Dependency setter fix + PerfectlyNegative no longer zeroing
    # CorrelationMatrix, in ONE self-contained call -- args = [x, dependency2, i, j, field].
    ct <- vapply(construct$components, function(c) c$target, character(1))
    cp <- lapply(construct$components, function(c) vapply(c$params, parse_num, numeric(1)))
    min_rv <- if (!is.null(construct$minimum_of_random_variables))
                as.logical(construct$minimum_of_random_variables) else TRUE
    dep <- if (!is.null(construct$dependency)) construct$dependency else "Independent"
    corr <- if (!is.null(construct$correlation))
              lapply(construct$correlation, function(r) as.double(unlist(r))) else list()
    return(ns$ch_cr_dependency_change_(ct, cp, min_rv, dep, args[[2]], corr,
                                        as.double(args[[1]]), args[[5]],
                                        as.integer(args[[3]]), as.integer(args[[4]])))
  }
  stop(sprintf("composite %s/%s has no shared-runner path and no local one either",
               target, method))
}

run_composite_case <- function(target, construct, assertions, datasets) {
  ns <- asNamespace("corehydror")
  # Limitation 1: a construct carrying a "nan"/"inf" literal cannot be serialized into the
  # grammar, so its (validity-only) assertions run locally.
  encodable <- !has_non_finite(construct)
  cspec <- composite_spec(target, construct, datasets)
  for (a in assertions) {
    method <- a$method
    args <- if (is.null(a$args)) list() else a$args
    # `dependency_change` has no runner counterpart (limitation 2); a non-finite evaluation
    # point is limitation 1 applied to the args rather than the construct.
    if (!encodable || identical(method, "dependency_change") || has_non_finite(args)) {
      check_assertion(dispatch_composite_local(target, construct, datasets, method, args), a)
      next
    }
    if (identical(method, "set_parameters")) {
      # The runner is stateless, so a SetParameters round trip is carried on the spec: every
      # later assertion in this case rebuilds with it applied, which is what dist_spec's
      # "set_parameters" key exists for. A second call replaces the first, exactly as the
      # in-place mutation did. The 0 mirrors the old dispatcher's dummy return, and the
      # assertion is still CHECKED rather than skipped.
      cspec$set_parameters <- args
      check_assertion(0, a)
      next
    }
    spec <- to_runner_json(cspec)
    if (identical(a$mode, "bool")) {
      # The old dispatcher ignored the assertion's method in bool mode and read
      # parameters_valid(); keep that exactly.
      check_assertion(ns$ch_dist_spec_run_(spec, "parameters_valid", "[]")$values[[1]], a)
    } else {
      r <- ns$ch_dist_spec_run_(spec, fixture_method(method), to_runner_json(args))
      check_assertion(fixture_pick(r, method, args), a)
    }
  }
}

# --- multivariate_distribution path -----------------------------------------------------
# The only partly delegated path. ch_mvdist_run_ covers the verbs this phase exposes (see
# mv_delegated below) and the rest of the pinned surface stays on dispatch_multivariate, which
# keeps the bespoke ch_dirichlet_val_/ch_bve_*/ch_mvn_*/ch_mvt_val_ glue in mvd.cpp: the MVNDST
# integrator internals, BivariateEmpirical's cdf_xy pair, Dirichlet's static
# log_multivariate_beta, the non-finite constructs and evaluation points, and the seeded MVNUNI
# sequences. Extending the runner's method table shrinks the bespoke half; nothing else moves.

# Translates a fixture multivariate construct into the dist_spec grammar. The only schema
# difference is Multinomial's parameter spelling (n / p here, trials / probabilities there);
# the four MultivariateNormal integrator settings (seed / max_evaluations / abs_error /
# rel_error) are grammar keys and pass straight through.
mvdist_spec <- function(target, construct) {
  out <- construct
  out$family <- target
  if (target == "Multinomial") {
    out$n <- NULL
    out$p <- NULL
    out$trials <- construct$n
    out$probabilities <- construct$p
  }
  out
}

# MultivariateNormal's CDF above dimension 2, its Interval, and MVNDST itself all draw from the
# instance's persistent MVNUNI stream, so each call ADVANCES it. A case that makes more than one
# such call pins a sequence off one object, which a stateless runner cannot reproduce by
# construction.
mvn_consumes_stream <- function(m) {
  m %in% c("cdf", "interval", "mvndst", "mvndst_inform", "mvndst_error")
}

kMvDelegatedMethods <- c("dimension", "pdf", "log_pdf", "cdf", "mahalanobis", "mean", "variance",
                         "sd", "covariance", "median", "mode", "inverse_cdf", "interval",
                         "degrees_of_freedom", "alpha", "alpha_sum", "number_of_trials",
                         "random_value", "lhs_value", "marginal_dimension", "marginal_mean",
                         "marginal_covariance", "marginal_log_pdf", "conditional_dimension",
                         "conditional_mean", "conditional_covariance")

# Methods ch_mvdist_run_ covers. What is left on dispatch_multivariate is exactly three groups:
# the MVNDST integrator internals (mvndst and its two status arms), BivariateEmpirical's
# cdf_xy(_after_set_parameters) and Dirichlet's static log_multivariate_beta, which have no
# runner verb; and MultivariateNormal's cdf/interval in a case that makes more than one
# stream-consuming call (`mvn_stream_isolated` FALSE -- r_mvtnorm_4d_sequential's eleven
# advancing cdf values are the reason). With `seed` in the grammar a SINGLE such call reproduces
# exactly, so those cases delegate.
mv_delegated <- function(target, method, mvn_stream_isolated) {
  if (method %in% c("mvndst", "mvndst_inform", "mvndst_error")) return(FALSE)
  if (identical(target, "MultivariateNormal") && method %in% c("cdf", "interval") &&
      !mvn_stream_isolated) {
    return(FALSE)
  }
  method %in% kMvDelegatedMethods
}

square_dim <- function(n) {
  d <- as.integer(round(sqrt(n)))
  if (d * d != n) stop("a covariance result is not a square matrix")
  d
}

# ch_mvdist_run_ returns whole vectors, so every fixture method that names one element indexes
# in here. Conventions preserved verbatim from the deleted dispatcher arms: mean/variance/sd/
# median/mode/alpha take [i]; covariance takes [i, j] against a row-major dimension^2 block;
# inverse_cdf takes [probabilities, i] and interval [lower, upper]; random_value/lhs_value take
# [sample_size, seed, row, col] against a row-major sample_size x dimension block; marginal_*
# take [indices, ...] and conditional_* take [indices, values, ...], both evaluated against the
# child distribution the runner hands back as a spec.
dispatch_multivariate_delegated <- function(spec, method, args) {
  ns <- asNamespace("corehydror")
  run <- function(s, m, a) ns$ch_mvdist_run_(s, m, to_runner_json(a))
  if (method %in% c("dimension", "alpha_sum", "degrees_of_freedom", "number_of_trials")) {
    return(run(spec, method, list())$values[[1]])
  }
  if (method %in% c("pdf", "log_pdf", "cdf", "mahalanobis")) {
    return(run(spec, method, args[[1]])$values[[1]])
  }
  if (identical(method, "inverse_cdf")) {
    return(run(spec, method, args[[1]])$values[[as.integer(args[[2]]) + 1L]])
  }
  if (identical(method, "interval")) {
    return(run(spec, "interval", c(args[[1]], args[[2]]))$values[[1]])
  }
  if (method %in% c("mean", "variance", "sd", "median", "mode", "alpha")) {
    return(run(spec, method, list())$values[[as.integer(args[[1]]) + 1L]])
  }
  if (identical(method, "covariance")) {
    r <- run(spec, "covariance", list())
    d <- square_dim(length(r$values))
    return(r$values[[as.integer(args[[1]]) * d + as.integer(args[[2]]) + 1L]])
  }
  if (method %in% c("random_value", "lhs_value")) {
    n <- as.integer(args[[1]])
    r <- run(spec, if (identical(method, "lhs_value")) "random_lhs" else "random",
             list(args[[1]], args[[2]]))
    d <- length(r$values) / n
    return(r$values[[as.integer(args[[3]]) * d + as.integer(args[[4]]) + 1L]])
  }
  if (startsWith(method, "marginal_") || startsWith(method, "conditional_")) {
    marginal <- startsWith(method, "marginal_")
    # `conditional` takes indices then values concatenated into one flat argument array, so the
    # trailing-argument base shifts by one.
    child_args <- if (marginal) args[[1]] else c(args[[1]], args[[2]])
    child <- run(spec, if (marginal) "marginal" else "conditional", child_args)$spec
    base <- if (marginal) 2L else 3L
    leaf <- sub("^[^_]*_", "", method)
    if (identical(leaf, "dimension")) return(run(child, "dimension", list())$values[[1]])
    if (identical(leaf, "log_pdf")) return(run(child, "log_pdf", args[[base]])$values[[1]])
    if (identical(leaf, "mean")) {
      return(run(child, "mean", list())$values[[as.integer(args[[base]]) + 1L]])
    }
    if (identical(leaf, "covariance")) {
      r <- run(child, "covariance", list())
      d <- square_dim(length(r$values))
      return(r$values[[as.integer(args[[base]]) * d + as.integer(args[[base + 1L]]) + 1L]])
    }
    stop(sprintf("unhandled child method: %s", method))
  }
  stop(sprintf("method '%s' is not delegated to ch_mvdist_run_", method))
}

# Flattens fixture assertion args to a numeric vector. Handles both conventions: a single
# nested vector argument (e.g. pdf args = [[0.3, 0.4, 0.3]]) and flat scalar args (e.g.
# covariance args = [0, 1], log_multivariate_beta args = [1.0, 1.0]).
flatten_mv_args <- function(args) {
  if (length(args) == 1 && is.list(args[[1]])) {
    return(as.double(unlist(args[[1]])))
  }
  as.double(unlist(args))
}

dispatch_multivariate <- function(target, construct, method, args) {
  ns <- asNamespace("corehydror")
  ar <- flatten_mv_args(args)
  if (target == "Dirichlet") {
    alpha <- vapply(construct$alpha, parse_num, numeric(1))
    return(ns$ch_dirichlet_val_(method, alpha, ar))
  }
  if (target == "BivariateEmpirical") {
    x1 <- vapply(construct$x1, parse_num, numeric(1))
    x2 <- vapply(construct$x2, parse_num, numeric(1))
    p_flat <- as.double(unlist(lapply(construct$p, function(row) vapply(row, parse_num, numeric(1)))))
    transforms <- c(
      if (!is.null(construct$x1_transform)) construct$x1_transform else "None",
      if (!is.null(construct$x2_transform)) construct$x2_transform else "None",
      if (!is.null(construct$p_transform)) construct$p_transform else "None"
    )
    # v2.1.4 stale-cache fix: args = [[x1_new...], [x2_new...], [[p_row0...], ...],
    # x1_eval, x2_eval] -- a dedicated entry point (not the flattened `ar`) since the
    # replacement grid's shape must stay structured.
    if (method == "cdf_xy_after_set_parameters") {
      x1_new <- vapply(args[[1]], parse_num, numeric(1))
      x2_new <- vapply(args[[2]], parse_num, numeric(1))
      p_new_flat <- as.double(unlist(lapply(args[[3]], function(row) vapply(row, parse_num, numeric(1)))))
      x1_eval <- parse_num(args[[4]])
      x2_eval <- parse_num(args[[5]])
      return(ns$ch_bve_cdf_after_set_parameters_(x1, x2, p_flat, length(x1), transforms,
                                                  x1_new, x2_new, p_new_flat, length(x1_new),
                                                  x1_eval, x2_eval))
    }
    return(ns$ch_bve_cdf_(method, x1, x2, p_flat, length(x1), transforms, ar))
  }
  if (target == "MultivariateNormal") {
    mean <- vapply(construct$mean, parse_num, numeric(1))
    cov_flat <- as.double(unlist(lapply(construct$covariance, function(row) vapply(row, parse_num, numeric(1)))))
    return(ns$ch_mvn_val_(method, mean, cov_flat, ar))
  }
  if (target == "MultivariateStudentT") {
    df <- parse_num(construct$df)
    location <- vapply(construct$location, parse_num, numeric(1))
    scale_flat <- as.double(unlist(lapply(construct$scale, function(row) vapply(row, parse_num, numeric(1)))))
    return(ns$ch_mvt_val_(method, df, location, scale_flat, ar))
  }
  stop(sprintf("unknown multivariate target: %s", target))
}

# --- MultivariateNormal seeded batches --------------------------------------------------
# `cdf` (dim>=3) and `mvndst` both draw from the seeded MVNUNI stream, so a RUN of consecutive
# same-method assertions in a seeded case must be evaluated on ONE persistent instance via the
# ch_mvn_*_seq_ glue in mvd.cpp, not dispatched one call at a time (which would silently reset
# the seed between assertions). This is limitation 2 above, and it is why the delegated path
# hands `cdf` back here whenever a case makes more than one stream-consuming call.
# `interval` is NOT listed: the corpus's only interval case makes exactly one stream-consuming
# call, so it delegates through the grammar's `seed` key instead.

kMvnSeededMethods <- c("cdf", "mvndst")

flatten_num_list <- function(x) as.double(unlist(lapply(x, parse_num)))

dispatch_mvn_seeded_seq <- function(construct, method, run) {
  ns <- asNamespace("corehydror")
  seed <- as.integer(construct$seed)
  mean <- vapply(construct$mean, parse_num, numeric(1))
  cov_flat <- as.double(unlist(lapply(construct$covariance, function(row) vapply(row, parse_num, numeric(1)))))

  if (method == "cdf") {
    xs_flat <- unlist(lapply(run, function(a) flatten_num_list(a$args[[1]])))
    return(ns$ch_mvn_cdf_seq_(mean, cov_flat, seed, xs_flat, length(run)))
  }
  if (method == "mvndst") {
    # args = [n, [lower...], [upper...], [infin...], [correl...], maxpts, abseps, releps]
    n_dim <- as.integer(run[[1]]$args[[1]])
    lower_flat <- unlist(lapply(run, function(a) flatten_num_list(a$args[[2]])))
    upper_flat <- unlist(lapply(run, function(a) flatten_num_list(a$args[[3]])))
    infin_flat <- as.integer(unlist(lapply(run, function(a) unlist(a$args[[4]]))))
    correl_flat <- unlist(lapply(run, function(a) flatten_num_list(a$args[[5]])))
    maxpts_v <- as.integer(vapply(run, function(a) a$args[[6]], numeric(1)))
    abseps_v <- as.double(vapply(run, function(a) a$args[[7]], numeric(1)))
    releps_v <- as.double(vapply(run, function(a) a$args[[8]], numeric(1)))
    return(ns$ch_mvn_mvndst_seq_(n_dim, seed, lower_flat, upper_flat, infin_flat, correl_flat,
                                  maxpts_v, abseps_v, releps_v, length(run)))
  }
  stop(sprintf("unknown seeded MultivariateNormal method: %s", method))
}

run_multivariate_case <- function(target, construct, assertions) {
  ns <- asNamespace("corehydror")
  # Limitation 1: a construct carrying a "nan"/"inf" literal cannot be serialized into the
  # grammar, so its (validity-only) assertions run locally.
  encodable <- !has_non_finite(construct)
  spec <- if (encodable) to_runner_json(mvdist_spec(target, construct)) else ""
  # A case whose MVNUNI stream is consumed at most once has no sequence to preserve, so its
  # cdf/interval delegates; anything more stays whole on the seeded batch path below.
  stream_calls <- if (identical(target, "MultivariateNormal")) {
    sum(vapply(assertions, function(a) mvn_consumes_stream(a$method), logical(1)))
  } else {
    0L
  }
  stream_isolated <- stream_calls <= 1L
  seeded <- !is.null(construct$seed)
  i <- 1
  n <- length(assertions)
  while (i <= n) {
    a <- assertions[[i]]
    method <- a$method
    args <- if (is.null(a$args)) list() else a$args
    is_bool <- identical(a$mode, "bool")
    # Limitation 1 again, on the evaluation point rather than the construct: MVN's and MVT's
    # log_pdf-at-infinity cases pass an infinite coordinate, which the grammar cannot carry.
    if (encodable && !has_non_finite(args)) {
      if (is_bool) {
        # The old dispatcher ignored the assertion's method in bool mode and read
        # parameters_valid(); keep that exactly.
        check_assertion(ns$ch_mvdist_run_(spec, "parameters_valid", "[]")$values[[1]], a)
        i <- i + 1
        next
      }
      if (mv_delegated(target, method, stream_isolated)) {
        check_assertion(dispatch_multivariate_delegated(spec, method, args), a)
        i <- i + 1
        next
      }
    }
    if (identical(target, "MultivariateNormal") && seeded && method %in% kMvnSeededMethods) {
      j <- i
      while (j <= n && identical(assertions[[j]]$method, method)) j <- j + 1
      run <- assertions[i:(j - 1)]
      actuals <- dispatch_mvn_seeded_seq(construct, method, run)
      for (idx in seq_along(run)) check_assertion(actuals[idx], run[[idx]])
      i <- j
      next
    }
    check_assertion(dispatch_multivariate(target, construct, method, args), a)
    i <- i + 1
  }
}

# --- bivariate_copula path ---------------------------------------------------------------
# Fully delegated to ch_copula_run_: every copula shares BivariateCopula's uniform
# theta/get_copula_parameters/pdf/cdf/... API, so there is no per-target branching left here at
# all (the "tau" method-of-moments fit, whose SetThetaFromTau is a member of each concrete
# Archimedean class rather than of IBivariateCopula, is dispatched by copulas::set_theta_from_tau
# inside dist_spec.hpp). The one local path is limitation 1: the non-finite theta/df validity
# cases, which the grammar cannot encode and which keep the narrow ch_cop_val_ call.
#
# construct is either {"theta": x} (optionally {"theta": x, "df": y} for 2-parameter copulas,
# and/or {"marginals": {"targets", "params"}} to attach marginals directly -- used by the
# "random_value" sampling oracles) or {"fit": {"x", "y", "method", "marginals"?}}; see
# fixtures/README.md for the full schema.

copula_margin <- function(family, params) {
  if (is.null(params)) list(family = family) else list(family = family, parameters = params)
}

# Translates a fixture copula construct into the dist_spec grammar. The schema differences are
# the marginal spelling (positional "marginals" here, margin_x / margin_y there) and the fit
# samples (dataset NAMES here, inline arrays there). The fixture's bare-family marginal
# convention needs no translation: dist_spec's build_copula MLE-fits a parameterless marginal to
# its own sample, which is what the fixture means and what IFM requires.
copula_spec <- function(target, construct, datasets) {
  out <- construct
  out$family <- target
  if (!is.null(construct$marginals)) {
    marg <- construct$marginals
    out$marginals <- NULL
    out$margin_x <- copula_margin(marg$targets[[1]], marg$params[[1]])
    out$margin_y <- copula_margin(marg$targets[[2]], marg$params[[2]])
  }
  if (!is.null(construct$fit)) {
    f <- construct$fit
    f$x <- datasets[[f$x]]
    f$y <- datasets[[f$y]]
    if (!is.null(f$marginals)) {
      fx <- f$marginals[[1]]
      fy <- f$marginals[[2]]
      f$marginals <- NULL
      f$margin_x <- copula_margin(fx, NULL)
      f$margin_y <- copula_margin(fy, NULL)
    }
    out$fit <- f
  }
  out
}

fixture_copula_method <- function(m) {
  switch(m,
    upper_tail_dependence = "tail_dependence",
    lower_tail_dependence = "tail_dependence",
    theta_minimum = "bounds",
    theta_maximum = "bounds",
    or_exceedance = "exceedance_or",
    and_exceedance = "exceedance_and",
    random_value = "random",
    # pdf, log_pdf, cdf, inverse_cdf, theta, df and the three log_likelihood_* verbs pass
    # straight through.
    m
  )
}

# The runner returns a whole vector for the methods the fixture indexes into. `random` comes back
# as all the x draws followed by all the y draws, so a fixture (row, col) with
# args = [sample_size, seed, row, col] lands at col * sample_size + row.
fixture_copula_pick <- function(r, method, args) {
  switch(method,
    lower_tail_dependence = r$values[[1]],
    theta_minimum = r$values[[1]],
    upper_tail_dependence = r$values[[2]],
    theta_maximum = r$values[[2]],
    inverse_cdf = r$values[[as.integer(args[[3]]) + 1L]],
    marginal_param = r$values[[as.integer(args[[2]]) + 1L]],
    random_value = r$values[[as.integer(args[[4]]) * as.integer(args[[1]]) +
                               as.integer(args[[3]]) + 1L]],
    r$values[[1]]
  )
}

# The three copula log-likelihood verbs take a paired SAMPLE, which the runner reads as one
# flat "all x then all y" args array. Spelling 200 numbers per assertion into the fixture would
# drown the file, so those assertions name their two datasets instead --
# args = ["<x dataset>", "<y dataset>"] -- and every runner splices the named arrays here.
# Documented under `bivariate_copula` in fixtures/README.md.
is_copula_log_likelihood <- function(m) {
  m %in% c("log_likelihood_pseudo", "log_likelihood_ifm", "log_likelihood_full")
}

copula_sample_args <- function(args, datasets) {
  as.list(unlist(lapply(args, function(name) {
    if (is.null(datasets[[name]])) {
      stop(sprintf("copula log-likelihood args name an unknown dataset: %s", name))
    }
    as.double(unlist(datasets[[name]]))
  })))
}

run_copula_case <- function(target, construct, assertions, datasets) {
  ns <- asNamespace("corehydror")
  # Limitation 1: a "nan"/"inf" theta or df cannot be serialized into the grammar. Every such
  # case asserts parameters_valid alone.
  if (has_non_finite(construct)) {
    params <- parse_num(construct$theta)
    if (!is.null(construct$df)) params <- c(params, parse_num(construct$df))
    for (a in assertions) {
      if (!identical(a$mode, "bool")) {
        stop(sprintf("%s/%s: a non-finite copula construct can only assert parameters_valid",
                     target, a$method))
      }
      check_assertion(ns$ch_cop_val_(target, params, "parameters_valid", numeric(0), "",
                                      numeric(0), "", numeric(0)), a)
    }
    return(invisible(NULL))
  }

  spec <- to_runner_json(copula_spec(target, construct, datasets))
  for (a in assertions) {
    method <- a$method
    args <- if (is.null(a$args)) list() else a$args
    if (identical(a$mode, "bool")) {
      # The old dispatcher ignored the assertion's method in bool mode and read
      # parameters_valid(); keep that exactly.
      check_assertion(ns$ch_copula_run_(spec, "parameters_valid", "[]")$values[[1]], a)
      next
    }
    if (identical(method, "marginal_param")) {
      # args = ("x" | "y", index): the side picks the runner method, the index picks the value
      # out of that marginal's parameter vector.
      r <- ns$ch_copula_run_(spec, if (identical(args[[1]], "x")) "marginal_x_parameters"
                                   else "marginal_y_parameters", "[]")
      check_assertion(fixture_copula_pick(r, method, args), a)
      next
    }
    if (is_copula_log_likelihood(method)) args <- copula_sample_args(args, datasets)
    r <- ns$ch_copula_run_(spec, fixture_copula_method(method), to_runner_json(args))
    check_assertion(fixture_copula_pick(r, method, args), a)
  }
}

# --- mcmc_sampler path -------------------------------------------------------------------
# Inherently STATEFUL (unlike multivariate_distribution's/bivariate_copula's per-assertion
# dispatch): one ch_mcmc_run_ call per case builds the model via the registry, configures the
# sampler from construct$settings, and samples() ONCE; every assertion in the case reads the
# single returned list. See fixtures/README.md's mcmc_sampler schema for the full method list
# and tolerance policy.

dispatch_mcmc <- function(result, method, args) {
  i1 <- function(x) as.integer(x) + 1L  # 0-based fixture index -> 1-based R index
  switch(method,
    posterior_mean       = result$posterior_mean[[i1(args[[1]])]],
    posterior_sd          = result$posterior_sd[[i1(args[[1]])]],
    posterior_median       = result$posterior_median[[i1(args[[1]])]],
    posterior_lower_ci     = result$posterior_lower_ci[[i1(args[[1]])]],
    posterior_upper_ci     = result$posterior_upper_ci[[i1(args[[1]])]],
    chain_value    = result$chains[[i1(args[[1]])]][i1(args[[2]]), i1(args[[3]])],
    chain_fitness  = result$chain_fitness[[i1(args[[1]])]][[i1(args[[2]])]],
    map_value      = result$map_values[[i1(args[[1]])]],
    map_fitness    = result$map_fitness[[1]],
    acceptance_rate      = result$acceptance_rates[[i1(args[[1]])]],
    mean_log_likelihood  = result$mean_log_likelihood[[i1(args[[1]])]],
    rhat = result$rhat[[i1(args[[1]])]],
    ess  = result$ess[[i1(args[[1]])]],
    stop(sprintf("unknown mcmc_sampler fixture method: %s", method))
  )
}

run_mcmc_case <- function(target, construct, assertions, datasets) {
  ns <- asNamespace("corehydror")
  model <- construct$model
  data <- as.double(unlist(datasets[[model$dataset]]))
  settings <- if (!is.null(construct$settings)) construct$settings else list()
  result <- ns$ch_mcmc_run_(target, model$name, model$family, data, settings)
  for (a in assertions) {
    args <- if (is.null(a$args)) list() else a$args
    actual <- dispatch_mcmc(result, a$method, args)
    check_assertion(actual, a)
  }
}

# --- bootstrap path ------------------------------------------------------------------------
# Inherently STATEFUL like mcmc_sampler: one ch_bootstrap_run_ call per case builds the model
# via the registry, runs() (or run_with_studentized_bootstrap()) ONCE, and computes confidence
# intervals ONCE; every assertion in the case reads the single returned list. See
# fixtures/README.md's bootstrap schema for the full method list and tolerance policy.

dispatch_bootstrap <- function(result, method, args) {
  i1 <- function(x) as.integer(x) + 1L  # 0-based fixture index -> 1-based R index
  switch(method,
    statistic_lower_ci   = result$statistic_lower_ci[[i1(args[[1]])]],
    statistic_upper_ci   = result$statistic_upper_ci[[i1(args[[1]])]],
    parameter_lower_ci   = result$parameter_lower_ci[[i1(args[[1]])]],
    parameter_upper_ci   = result$parameter_upper_ci[[i1(args[[1]])]],
    population_estimate  = result$population_estimate[[i1(args[[1]])]],
    valid_count          = result$valid_count[[i1(args[[1]])]],
    replicate_value      = result$replicate_values[i1(args[[1]]), i1(args[[2]])],
    stop(sprintf("unknown bootstrap fixture method: %s", method))
  )
}

run_bootstrap_case <- function(construct, assertions, datasets) {
  ns <- asNamespace("corehydror")
  dataset <- if (!is.null(construct$dataset)) as.double(unlist(datasets[[construct$dataset]])) else numeric(0)
  probabilities <- vapply(construct$probabilities, parse_num, numeric(1))
  mu <- if (!is.null(construct$mu)) construct$mu else 0
  sigma <- if (!is.null(construct$sigma)) construct$sigma else 0
  sample_size <- if (!is.null(construct$sample_size)) as.integer(construct$sample_size) else 0L
  max_retries <- if (!is.null(construct$max_retries)) as.integer(construct$max_retries) else 20L
  run <- if (!is.null(construct$run)) construct$run else "regular"
  alpha <- if (!is.null(construct$alpha)) construct$alpha else 0.1

  result <- ns$ch_bootstrap_run_(construct$model, mu, sigma, sample_size, probabilities, dataset,
    as.integer(construct$replicates), as.integer(construct$seed), max_retries, run,
    construct$ci_method, alpha)
  for (a in assertions) {
    args <- if (is.null(a$args)) list() else a$args
    actual <- dispatch_bootstrap(result, a$method, args)
    check_assertion(actual, a)
  }
}

# --- model_estimation path -------------------------------------------------------------------
# Inherently STATEFUL like mcmc_sampler/bootstrap: one ch_estimation_run_ (ML/MAP),
# ch_estimation_bayes_run_ (BayesianAnalysis, T12), or ch_model_simulate_ (Simulation, M13)
# call per case builds the model, runs its one stateful call (estimate() or the seeded
# ISimulatable draw), and returns the full result surface; every assertion in the case reads
# that single cached list. See fixtures/README.md's model_estimation section for the full
# method list and the `bic`/`chain_value` design notes. `bic` is the one exception to the
# "cached list" contract for ML/MAP: it takes an actual sample size `n` (C# `GetBIC(
# sampleSize)`), read live from the fixture's `args[[1]]` at dispatch time via
# `ch_estimation_bic_`, not precomputed alongside the rest.
#
# M13: `construct.model` is no longer a flat {family, dataset} pair -- it can name any of the
# four Phase 5 model types, a full censored DataFrame, nonstationary trend specs, and explicit
# parameter values. The parsed spec is re-serialized to JSON (digits = I(17) round-trips
# doubles exactly) and handed to the SHARED C++ builder (corehydro/models/model_spec.hpp), the
# same code path the C++ runner and the Python glue use; only the `dataset` reference is still
# resolved here, like every other fixture kind.

dispatch_estimation <- function(result, method, args, ctx) {
  i1 <- function(x) as.integer(x) + 1L  # 0-based fixture index -> 1-based R index
  switch(method,
    parameter          = result$parameters[[i1(args[[1]])]],
    max_log_likelihood = result$max_log_likelihood[[1]],
    aic                = result$aic[[1]],
    bic                = ctx$bic_fn(as.integer(args[[1]])),  # args[[1]] is a sample size n, not an index
    covariance         = result$covariance[i1(args[[1]]), i1(args[[2]])],
    standard_error     = result$standard_errors[[i1(args[[1]])]],
    correlation        = result$correlation[i1(args[[1]]), i1(args[[2]])],
    dic                = result$dic[[1]],
    waic               = result$waic[[1]],
    looic              = result$looic[[1]],
    # GMM (B11): j_stat/j_stat_pval from the cached run list; quantile_variance takes a
    # per-assertion AEP, so it rebuilds the deterministic fit live (the `bic` precedent).
    j_stat             = result$j_stat[[1]],
    j_stat_pval        = result$j_stat_pval[[1]],
    # T13: GMMIterations/ConvergedWithinTolerance (off-by-one fix) and
    # OptimizerFallbackCount (sticky BFGS->NelderMead fallback).
    gmm_iterations     = result$gmm_iterations[[1]],
    converged_within_tolerance = as.numeric(result$converged_within_tolerance[[1]]),
    optimizer_fallback_count = result$optimizer_fallback_count[[1]],
    quantile_variance  = ctx$qvar_fn(as.double(args[[1]])),
    posterior_mean     = result$posterior_mean[[i1(args[[1]])]],
    # chain_value [chain, iter, param]: ch_estimation_bayes_run_ returns a flattened
    # `chain_values` vector + `chain_dims` (n_chains, n_iterations, n_params); recover the
    # row-major (chain, iter, param) index the C++/Python/C# access order uses.
    chain_value = {
      d <- result$chain_dims
      idx <- args[[1]] * d[2] * d[3] + args[[2]] * d[3] + args[[3]] + 1L
      result$chain_values[[idx]]
    },
    # simulated_value [i]: the seeded ISimulatable draw cached by ch_model_simulate_ (M13).
    simulated_value = result$simulated[[i1(args[[1]])]],
    # The M14 DataFrame surface (works under any target -- it reads the model, not the
    # estimator): ctx$df_fn() lazily builds the frame surface ONCE per case via
    # ch_model_data_frame_ and memoizes it (the bic lazy-rebuild precedent).
    number_of_low_outliers = ctx$df_fn()$number_of_low_outliers[[1]],
    low_outlier_threshold = ctx$df_fn()$low_outlier_threshold[[1]],
    # plotting_position [kind, i]: kind is "exact" | "interval" | "uncertain", in spec order.
    plotting_position = ctx$df_fn()[[paste0("pp_", args[[1]])]][[i1(args[[2]])]],
    # The Validate surface (Task 16, works under any target): ctx$validate_fn() lazily builds +
    # memoizes ch_model_validate_'s result once per case (the df_fn/bic lazy-rebuild precedent).
    # validation_message_contains is a structural substring check, not a byte-exact message pin
    # (see test_fixtures.cpp's dispatch_model_validate for the rationale).
    is_valid = as.numeric(ctx$validate_fn()$is_valid[[1]]),
    validation_message_contains = as.numeric(any(grepl(args[[1]], ctx$validate_fn()$messages, fixed = TRUE))),
    # --- Task 9: the wider fit surface -------------------------------------------------------
    # These read ctx$fit_fn(), a lazily-built + memoized ch_fit_run_ over the case's FULL
    # construct (the fixture construct with `settings` hoisted to the top level -- see
    # run_estimation_case below). The narrow ch_estimation_run_/ch_estimation_bayes_run_ result
    # above is deliberately untouched: it backs the pinned oracles and does not carry these
    # fields. Both calls run the same deterministic fit, so this is the bic/df_fn lazy-rebuild
    # precedent, not a second, different fit.
    profile_lower      = ctx$fit_fn()$profile_lower[[i1(args[[1]])]],
    profile_upper      = ctx$fit_fn()$profile_upper[[i1(args[[1]])]],
    # profile_value [param, bin, col]: profile_grid is n_params x bins x 2, row-major, col 0 =
    # the parameter value at the bin midpoint, col 1 = the profile log-likelihood there.
    profile_value = {
      f <- ctx$fit_fn()
      idx <- (args[[1]] * f$profile_bins + args[[2]]) * 2L + args[[3]] + 1L
      f$profile_grid[[idx]]
    },
    function_evaluations = ctx$fit_fn()$function_evaluations[[1]],
    # status_is [name]: 1 when the optimizer status matches, else 0 (the
    # validation_message_contains boolean-as-double precedent).
    status_is          = as.numeric(identical(ctx$fit_fn()$status[[1]], args[[1]])),
    nobs               = ctx$fit_fn()$nobs[[1]],
    prior_log_likelihood = ctx$fit_fn()$prior_log_likelihood[[1]],
    rhat               = ctx$fit_fn()$rhat[[i1(args[[1]])]],
    ess                = ctx$fit_fn()$ess[[i1(args[[1]])]],
    acceptance_rate    = ctx$fit_fn()$acceptance_rates[[i1(args[[1]])]],
    posterior_median   = ctx$fit_fn()$summary_median[[i1(args[[1]])]],
    posterior_sd       = ctx$fit_fn()$summary_sd[[i1(args[[1]])]],
    posterior_lower    = ctx$fit_fn()$summary_lower[[i1(args[[1]])]],
    posterior_upper    = ctx$fit_fn()$summary_upper[[i1(args[[1]])]],
    # The PSIS-LOO Pareto-k surface lives on FitDiagnostics (the InfluenceDiagnostics wrapper),
    # not on the fit result, so it takes the second lazily-memoized runner call.
    pareto_k           = ctx$diag_fn()$pareto_k[[i1(args[[1]])]],
    max_pareto_k       = ctx$diag_fn()$max_pareto_k[[1]],
    stop(sprintf("unknown model_estimation fixture method: %s", method))
  )
}

run_estimation_case <- function(target, construct, assertions, datasets) {
  ns <- asNamespace("corehydror")
  model <- construct$model
  # Re-serialize the parsed spec for the shared C++ builder (see the path comment above).
  model_json <- as.character(jsonlite::toJSON(model, auto_unbox = TRUE, digits = I(17)))
  data <- if (!is.null(model$dataset)) as.double(unlist(datasets[[model$dataset]])) else numeric(0)

  # The M14 DataFrame surface: lazily build + memoize the frame-surface list (only cases
  # that actually assert a data-frame method pay for the rebuild; see ch_model_data_frame_).
  df_env <- new.env(parent = emptyenv())
  df_fn <- function() {
    if (is.null(df_env$df)) df_env$df <- ns$ch_model_data_frame_(model_json, data)
    df_env$df
  }
  # The Validate surface (Task 16): lazily build + memoize ch_model_validate_'s result (only
  # cases that actually assert is_valid/validation_message_contains pay for the call).
  validate_env <- new.env(parent = emptyenv())
  validate_fn <- function() {
    if (is.null(validate_env$v)) validate_env$v <- ns$ch_model_validate_(model_json, data)
    validate_env$v
  }

  if (target == "Simulation") {
    seed <- if (!is.null(construct$seed)) as.integer(construct$seed) else -1L
    draws <- ns$ch_model_simulate_(model_json, data, as.integer(construct$sample_size), seed)
    result <- list(simulated = draws)
    ctx <- list()
  } else if (target == "Validate") {
    # Builds the model only (no estimator, no draw) -- every assertion reads ctx$validate_fn().
    result <- list()
    ctx <- list()
  } else if (target == "BayesianAnalysis") {
    sampler <- if (!is.null(construct$sampler)) construct$sampler else "DEMCzs"
    s <- construct$settings
    geti <- function(name, default = -1L) if (!is.null(s[[name]])) as.integer(s[[name]]) else default
    result <- ns$ch_estimation_bayes_run_(
      model_json, data, sampler,
      seed = geti("seed"), iterations = geti("iterations"),
      warmup_iterations = geti("warmup_iterations"), number_of_chains = geti("number_of_chains"),
      thinning_interval = geti("thinning_interval"), initial_iterations = geti("initial_iterations"),
      output_length = geti("output_length")
    )
    ctx <- list()
  } else if (target == "GeneralizedMethodOfMoments") {
    # GMM (B11): a bulletin17c model fit by GMM. One stateful estimate()+post_process caches the
    # full surface; quantile_variance rebuilds the deterministic fit live at dispatch.
    strategy <- if (!is.null(construct$strategy)) construct$strategy else "Iterative"
    optimizer <- if (!is.null(construct$optimizer)) construct$optimizer else "BFGS"
    max_gmm_iterations <- if (!is.null(construct$max_gmm_iterations)) as.integer(construct$max_gmm_iterations) else -1L
    sample_size <- if (!is.null(construct$sample_size)) as.integer(construct$sample_size) else 0L
    seed <- if (!is.null(construct$seed)) as.integer(construct$seed) else -1L
    result <- ns$ch_estimation_gmm_run_(model_json, data, strategy, optimizer, max_gmm_iterations, sample_size, seed)
    ctx <- list(qvar_fn = function(aep) ns$ch_estimation_gmm_qvar_(model_json, data, strategy, optimizer, max_gmm_iterations, aep))
  } else {
    optimizer <- if (!is.null(construct$optimizer)) construct$optimizer else "DifferentialEvolution"
    # P3: an optional seeded-draw digest off the FITTED model (sample_size + seed) lets one MLE
    # smoke file cover parameter + max_log_likelihood + a seeded simulated_value.
    sample_size <- if (!is.null(construct$sample_size)) as.integer(construct$sample_size) else 0L
    seed <- if (!is.null(construct$seed)) as.integer(construct$seed) else -1L
    result <- ns$ch_estimation_run_(target, model_json, data, optimizer, sample_size, seed)
    ctx <- list(bic_fn = function(n) ns$ch_estimation_bic_(target, model_json, data, optimizer, n))
  }
  ctx$df_fn <- df_fn
  ctx$validate_fn <- validate_fn

  # Task 9: the FULL construct the shared fit runner reads -- the fixture's own construct with
  # the `settings` sub-object hoisted to the top level (where apply_bayesian_settings looks for
  # the Bayesian knobs); every other key passes through untouched and run_fit ignores the ones
  # its target does not use. C++ and Python assemble it identically, so all three hand the runner
  # byte-identical constructs. Both accessors are lazy: a case asserting nothing new never runs.
  full <- construct
  if (!is.null(full$settings)) {
    for (nm in names(full$settings)) full[[nm]] <- full$settings[[nm]]
    full$settings <- NULL
  }
  # fit_runner.hpp's run_fit/run_fit_diagnostics default `optimizer` to DifferentialEvolution for
  # every target, including GMM -- but the narrow GMM path above defaults it to BFGS (matching the
  # C# GMM ctor default). Without this, a GMM case that omits `optimizer` and asserts one of the
  # wider fit-surface methods would read that method off a DifferentialEvolution fit while
  # parameter/j_stat came from a BFGS fit. Write the same BFGS default here so both paths agree.
  if (target == "GeneralizedMethodOfMoments" && is.null(full$optimizer)) full$optimizer <- "BFGS"
  full_json <- as.character(jsonlite::toJSON(full, auto_unbox = TRUE, digits = I(17)))
  fit_target <- if (target == "GeneralizedMethodOfMoments") "GMM" else target
  fit_env <- new.env(parent = emptyenv())
  ctx$fit_fn <- function() {
    if (is.null(fit_env$f)) fit_env$f <- ns$ch_fit_run_(fit_target, full_json, data)
    fit_env$f
  }
  diag_env <- new.env(parent = emptyenv())
  ctx$diag_fn <- function() {
    if (is.null(diag_env$d)) diag_env$d <- ns$ch_fit_diagnostics_(fit_target, full_json, data)
    diag_env$d
  }

  for (a in assertions) {
    args <- if (is.null(a$args)) list() else a$args
    actual <- dispatch_estimation(result, a$method, args, ctx)
    check_assertion(actual, a)
  }
}

# --- analysis path (Phase 8: user-facing Analyses layer) -------------------------------------
# Stateful like model_estimation: one ch_analysis_* glue call per case builds + runs the analysis
# and returns the full result surface; every assertion reads that single cached list. The
# construct fields map 1:1 onto the glue args, so R/Python/C++ build byte-identical analyses.

dispatch_analysis <- function(result, method, args) {
  i1 <- function(x) as.integer(x) + 1L # 0-based fixture index -> 1-based R index
  switch(method,
    candidate_count       = length(result$aic),
    candidate_aic         = result$aic[[i1(args[[1]])]],
    candidate_bic         = result$bic[[i1(args[[1]])]],
    candidate_rmse        = result$rmse[[i1(args[[1]])]],
    candidate_converged   = as.numeric(result$converged[[i1(args[[1]])]]),
    parameter             = result$parameters[[i1(args[[1]])]],
    mode_curve            = result$mode_curve[[i1(args[[1]])]],
    mean_curve            = result$mean_curve[[i1(args[[1]])]],
    lower_ci              = result$lower_ci[[i1(args[[1]])]],
    upper_ci              = result$upper_ci[[i1(args[[1]])]],
    exceedance_probability = result$exceedance_probabilities[[i1(args[[1]])]],
    point_estimate        = result$point_estimates[[i1(args[[1]])]],
    beta1                 = result$beta1[[i1(args[[1]])]],
    nu                    = result$nu[[i1(args[[1]])]],
    quantile_variance     = result$quantile_variance[[i1(args[[1]])]],
    aic                   = result$aic[[1]],
    bic                   = result$bic[[1]],
    dic                   = result$dic[[1]],
    rmse                  = result$rmse[[1]],
    confidence_level      = result$confidence_level[[1]],
    # D5: time-series curve length + the three diagnostics.
    curve_length          = length(result$mode_curve),
    leverage_count        = length(result$leverage$leverage),
    leverage_prior_count  = length(result$leverage$prior_leverage),
    total_leverage        = result$leverage$total_leverage,
    total_fit_influence   = result$leverage$total_fit_influence,
    total_variance_influence = result$leverage$total_variance_influence,
    obs_leverage          = result$leverage$leverage[[i1(args[[1]])]],
    obs_fit_influence     = result$leverage$fit_influence[[i1(args[[1]])]],
    obs_variance_influence = result$leverage$variance_influence[[i1(args[[1]])]],
    obs_value             = result$leverage$value[[i1(args[[1]])]],
    influence_count       = result$influence$count,
    mean_pareto_k         = result$influence$mean_pareto_k,
    max_pareto_k          = result$influence$max_pareto_k,
    count_pareto_k_above_05 = result$influence$count_pareto_k_above_05,
    count_pareto_k_above_07 = result$influence$count_pareto_k_above_07,
    count_pareto_k_above_10 = result$influence$count_pareto_k_above_10,
    proportion_problematic = result$influence$proportion_problematic,
    is_reliable           = as.numeric(result$influence$is_reliable),
    pareto_k              = result$influence$pareto_k[[i1(args[[1]])]],
    elpd_loo              = result$influence$elpd_loo[[i1(args[[1]])]],
    prior_influence_count = result$prior_influence$count,
    total_prior_log_likelihood = result$prior_influence$total_prior_log_likelihood,
    total_data_log_likelihood = result$prior_influence$total_data_log_likelihood,
    prior_to_data_ratio   = result$prior_influence$prior_to_data_ratio,
    is_prior_influential  = as.numeric(result$prior_influence$is_prior_influential),
    mean_prior_precision_share = result$prior_influence$mean_prior_precision_share,
    # X11 extended analyses (composite / spatial_gev / bivariate / coincident / rating_curve /
    # bootstrap / prior + posterior predictive).
    z_output              = result$z_output_values[[i1(args[[1]])]],
    z_output_length       = length(result$z_output_values),
    site_count            = result$site_count,
    site_location_mean    = result$site_location_mean[[i1(args[[1]])]],
    site_scale_mean       = result$site_scale_mean[[i1(args[[1]])]],
    site_shape_mean       = result$site_shape_mean[[i1(args[[1]])]],
    site_quantile_mean    = result$site0_quantile_mean[[i1(args[[1]])]],
    cv_mae                = result$cv_mae,
    cv_rmse               = result$cv_rmse,
    cv_mean_bias          = result$cv_mean_bias,
    mean_p_value          = result$mean_p_value,
    sd_p_value            = result$sd_p_value,
    skewness_p_value      = result$skewness_p_value,
    min_p_value           = result$min_p_value,
    max_p_value           = result$max_p_value,
    predictive_replicates = result$number_of_replicates,
    has_misfit            = result$has_misfit,
    number_of_valid_draws = result$number_of_valid_draws,
    summary_mean_quantile = result$summary_mean_quantiles[[i1(args[[1]])]],
    summary_sd_quantile   = result$summary_sd_quantiles[[i1(args[[1]])]],
    summary_min_quantile  = result$summary_min_quantiles[[i1(args[[1]])]],
    summary_max_quantile  = result$summary_max_quantiles[[i1(args[[1]])]],
    # T19: BootstrapDiagnostics (Bulletin17CAnalysis, Bootstrap/BiasCorrectedBootstrap).
    boot_has_results             = as.numeric(result$bootstrap$has_results),
    boot_total_replicates        = result$bootstrap$total_replicates,
    boot_attempted_replicates    = result$bootstrap$attempted_replicates,
    boot_failed_replicates       = result$bootstrap$failed_replicates,
    boot_valid_replicates        = result$bootstrap$valid_replicates,
    boot_retained_replicates     = result$bootstrap$retained_replicates,
    boot_failure_rate            = result$bootstrap$failure_rate,
    boot_total_retries           = result$bootstrap$total_retries,
    boot_average_retries         = result$bootstrap$average_retries,
    boot_pivot_rejections        = result$bootstrap$pivot_rejections,
    boot_mahalanobis_rejections  = result$bootstrap$mahalanobis_rejections,
    boot_transform_failures      = result$bootstrap$transform_failures,
    boot_status_success_count    = result$bootstrap$status_success_count,
    boot_status_max_iterations_count = result$bootstrap$status_max_iterations_count,
    boot_status_max_function_evaluations_count = result$bootstrap$status_max_function_evaluations_count,
    boot_status_failure_count    = result$bootstrap$status_failure_count,
    boot_status_none_count       = result$bootstrap$status_none_count,
    boot_optimizer_fallbacks     = result$bootstrap$optimizer_fallbacks,
    stop(sprintf("unknown analysis fixture method: %s", method))
  )
}

# X11: analysis fixture targets routed through the shared ch_analysis_extended_run_ dispatch.
.extended_analysis_targets <- c(
  "CompositeAnalysis", "SpatialGEVAnalysis", "BivariateAnalysis",
  "CoincidentFrequencyAnalysis", "RatingCurveAnalysis", "BootstrapAnalysis",
  "PriorPredictiveCheck", "PosteriorPredictiveCheck"
)

# D5: map an analysis fixture target to the ch_analysis_family_run_ analysis_type discriminator.
.family_analysis_type <- function(target) {
  switch(target,
    MixtureAnalysis = "mixture",
    CompetingRiskAnalysis = "competing_risk",
    PointProcessAnalysis = "point_process",
    ARAnalysis = "ar",
    MAAnalysis = "ma",
    ARIMAAnalysis = "arima",
    ARIMAXAnalysis = "arimax",
    NULL
  )
}

run_analysis_case <- function(target, construct, assertions, datasets) {
  ns <- asNamespace("corehydror")
  geti <- function(name, default) if (!is.null(construct[[name]])) as.integer(construct[[name]]) else default
  getd <- function(name, default) if (!is.null(construct[[name]])) as.double(construct[[name]]) else default
  ep <- if (!is.null(construct$exceedance_probabilities)) {
    as.double(unlist(construct$exceedance_probabilities))
  } else {
    numeric(0)
  }

  if (target == "FittingAnalysis") {
    data <- as.double(unlist(datasets[[construct$dataset]]))
    result <- ns$ch_analysis_fit_distributions_(data)
  } else if (target == "UnivariateAnalysis") {
    model <- construct$model
    model_json <- as.character(jsonlite::toJSON(model, auto_unbox = TRUE, digits = I(17)))
    data <- as.double(unlist(datasets[[model$dataset]]))
    sampler <- if (!is.null(construct$sampler)) construct$sampler else "DEMCzs"
    result <- ns$ch_analysis_univariate_run_(
      model_json, data, sampler, geti("iterations", 3000L), geti("output_length", 10000L),
      getd("credible_level", 0.90), geti("seed", 12345L), ep, geti("thinning_interval", -1L)
    )
  } else if (target == "Bulletin17CAnalysis") {
    model <- construct$model
    model_json <- as.character(jsonlite::toJSON(model, auto_unbox = TRUE, digits = I(17)))
    # T19: an inline `data_frame` (mixed exact/interval/threshold/uncertain series) is valid
    # without a `dataset` reference -- mirrors the C++ runner's guard so a Bulletin17CAnalysis
    # case can force low outliers / censored data onto the parent frame.
    data <- if (!is.null(model$dataset)) as.double(unlist(datasets[[model$dataset]])) else numeric(0)
    um <- if (!is.null(construct$uncertainty_method)) construct$uncertainty_method else "MultivariateNormal"
    result <- ns$ch_analysis_b17c_run_(
      model_json, data, um, geti("output_length", 10000L), geti("seed", 12345L),
      getd("confidence_level", 0.90), ep
    )
  } else if (!is.null(.family_analysis_type(target))) {
    # D5: the seven per-family analyses through the single dispatch binding.
    model <- construct$model
    model_json <- as.character(jsonlite::toJSON(model, auto_unbox = TRUE, digits = I(17)))
    data <- as.double(unlist(datasets[[model$dataset]]))
    sampler <- if (!is.null(construct$sampler)) construct$sampler else "DEMCzs"
    result <- ns$ch_analysis_family_run_(
      .family_analysis_type(target), model_json, data, sampler, geti("iterations", 3000L),
      geti("output_length", 10000L), getd("credible_level", 0.90), geti("seed", 12345L), ep,
      geti("thinning_interval", -1L), geti("training_time_steps", -1L),
      geti("forecasting_time_steps", 0L)
    )
  } else if (target == "Diagnostics") {
    # D5: leverage / influence / prior-influence diagnostics off a BayesianAnalysis fit.
    model <- construct$model
    model_json <- as.character(jsonlite::toJSON(model, auto_unbox = TRUE, digits = I(17)))
    data <- as.double(unlist(datasets[[model$dataset]]))
    sampler <- if (!is.null(construct$sampler)) construct$sampler else "DEMCzs"
    result <- ns$ch_analysis_diagnostics_run_(
      model_json, data, sampler, geti("iterations", 3000L), geti("output_length", 10000L),
      geti("seed", 12345L), geti("thinning_interval", -1L), geti("thin_every", 10L)
    )
  } else if (target %in% .extended_analysis_targets) {
    # X11: the five remaining analyses + bootstrap + predictive checks. Re-serialize the whole
    # construct + datasets and call the single shared dispatch binding (byte-identical path to the
    # C++ runner and the Python twin).
    construct_json <- as.character(jsonlite::toJSON(construct, auto_unbox = TRUE, digits = I(17)))
    datasets_json <- as.character(jsonlite::toJSON(datasets, auto_unbox = TRUE, digits = I(17)))
    result <- ns$ch_analysis_extended_run_(target, construct_json, datasets_json)
  } else {
    stop(sprintf("unknown analysis target: %s", target))
  }

  for (a in assertions) {
    args <- if (is.null(a$args)) list() else a$args
    actual <- dispatch_analysis(result, a$method, args)
    check_assertion(actual, a)
  }
}

test_that("oracle fixtures validate", {
  skip_if_not_installed("jsonlite")
  fdir <- system.file("fixtures", package = "corehydror")
  files <- list.files(fdir, pattern = "\\.json$", recursive = TRUE, full.names = TRUE)
  expect_gt(length(files), 0)
  for (f in files) {
    spec <- jsonlite::read_json(f, simplifyVector = FALSE)
    # Only validate univariate_distribution, multivariate_distribution, bivariate_copula,
    # mcmc_sampler, bootstrap, model_estimation, and toolbox fixtures, plus the
    # special_function targets in kRoutedSpecialFunctionTargets (Correlation.*, the routable
    # RunningStatistics.*/RunningCovariance.*/Fourier.* subsets, and Statistics.percentile --
    # see run_special_function_case above); every other special_function kind stays validated
    # in C++ only, not exposed to the R package.
    if (identical(spec$kind, "model_estimation")) {
      target <- spec$target
      datasets <- spec$datasets
      for (case in spec$cases) {
        run_estimation_case(target, case$construct, case$assertions, datasets)
      }
      next
    }
    if (identical(spec$kind, "analysis")) {
      target <- spec$target
      datasets <- spec$datasets
      for (case in spec$cases) {
        run_analysis_case(target, case$construct, case$assertions, datasets)
      }
      next
    }
    if (identical(spec$kind, "bootstrap")) {
      datasets <- spec$datasets
      for (case in spec$cases) {
        run_bootstrap_case(case$construct, case$assertions, datasets)
      }
      next
    }
    if (identical(spec$kind, "special_function")) {
      file_target <- spec$target
      for (case in spec$cases) {
        target <- if (is.null(case$target)) file_target else case$target
        if (!target %in% kRoutedSpecialFunctionTargets) next
        actual <- run_special_function_case(target, case$args)
        for (a in case$assertions) check_assertion(actual, a)
      }
      next
    }
    if (identical(spec$kind, "toolbox")) {
      ns <- asNamespace("corehydror")
      datasets <- spec$datasets
      for (case in spec$cases) {
        data <- toolbox_case_data(case, datasets)
        opts_list <- if (is.null(case$options)) list() else case$options
        # Only the "sobol" method reads a path (SobolSequence's constructor only touches it when
        # dimension > 1); "stratify" never does, so this stays scoped to that one method rather
        # than the whole "sampling" group.
        if (identical(spec$group, "sampling") && identical(case$assertions[[1]]$method, "sobol")) {
          # The Sobol direction-numbers file ships inside the package; path resolution is a
          # wrapper concern (see numerics/support/toolbox/sampling.hpp's file header), so the
          # fixture itself never carries a "path" key -- this harness injects its own resolved
          # path, mirroring what sobol_sequence() does for a real caller.
          opts_list$path <- system.file("extdata", "new-joe-kuo-6.21201", package = "corehydror")
        }
        opts <- if (length(opts_list) == 0L) "{}" else ns$to_spec_json(opts_list)
        for (a in case$assertions) {
          r <- ns$ch_toolbox_run_(spec$group, a$method, data, opts)
          check_assertion(toolbox_select(r, a, spec$group), a)
        }
      }
      next
    }
    if (identical(spec$kind, "optimizer")) {
      ns <- asNamespace("corehydror")
      for (case in spec$cases) {
        construct <- case$construct
        objective_name <- if (is.null(construct$objective)) "DeJong" else construct$objective
        construct$objective <- NULL
        r <- ns$ch_optim_run_(optimizer_spec_json(ns, construct),
                              optimizer_fixture_objective(objective_name))
        for (a in case$assertions) {
          if (identical(a$method, "value")) {
            check_assertion(r$value, a)
          } else if (identical(a$method, "parameter")) {
            check_assertion(r$parameters[[a$args[[1]] + 1L]], a)
          } else if (identical(a$method, "status")) {
            expect_identical(r$status, a$expected)
          } else {
            stop(sprintf("unknown optimizer fixture assertion method: %s", a$method))
          }
        }
      }
      next
    }
    if (identical(spec$kind, "callback") ||
          identical(spec$kind, "callback_cross_language")) {
      ns <- asNamespace("corehydror")
      # A "callback"-kind case IS its own single block; a "callback_cross_language"-kind case
      # (fixtures/callback/callback_cross_language.json) nests one block per key OTHER than "name"
      # -- "mcmc", "bootstrap", "pivotal" -- each shaped exactly like a "callback"-kind case's
      # construct/assertions, so the one body below drives both kinds and the cross-language
      # fixture grows no evaluation path of its own. The labels are read off the case rather than
      # listed here, so a case may nest one block or five without a runner change.
      # Its assertions are spelled mode "abs" with tol 0, i.e. bit equality with the C++, Python and
      # C# runners rather than a tolerance.
      blocks <- list()
      for (case in spec$cases) {
        if (identical(spec$kind, "callback")) {
          blocks[[length(blocks) + 1L]] <- case
        } else {
          for (sub in setdiff(names(case), "name")) blocks[[length(blocks) + 1L]] <- case[[sub]]
        }
      }
      for (case in blocks) {
        construct <- case$construct
        # Every group has its own R entry point: ch_callback_math_, ch_rng_probe_,
        # ch_callback_mcmc_, ch_callback_gmm_ and ch_callback_bootstrap_.
        opts <- callback_options_json(ns, construct$options)
        fn <- callback_fixture_function(construct$callback)
        r <- if (identical(construct$group, "math") &&
                    construct$method %in% c("root_find_newton", "root_find_system")) {
          # P2 "math extras": these two math methods need a SECOND callback (the analytic
          # derivative `df`, or the Jacobian reusing gmm's `jacobian` key), which
          # ch_callback_math_ has no argument for -- ch_callback_math2_ is the two-callback entry
          # point, mirroring ch_callback_gmm_'s own optional second/third delegates above.
          second_key <- if (identical(construct$method, "root_find_newton")) "df" else "jacobian"
          g <- callback_fixture_function(construct[[second_key]])
          ns$ch_callback_math2_(construct$method, opts, fn, g)
        } else if (identical(construct$group, "math")) {
          ns$ch_callback_math_(construct$method, opts, fn)
        } else if (identical(construct$group, "rng")) {
          ns$ch_rng_probe_(opts, fn)
        } else if (identical(construct$group, "mcmc")) {
          # The two other delegates the mcmc group's samplers take, each resolved out of the same
          # catalog: a Gibbs proposal and an HMC/NUTS gradient. Absent keys stay NULL, which is
          # what "no proposal" and "the ported default gradient" mean.
          proposal <- if (is.null(construct$proposal)) {
            NULL
          } else {
            callback_fixture_function(construct$proposal)
          }
          gradient <- if (is.null(construct$gradient)) {
            NULL
          } else {
            callback_fixture_function(construct$gradient)
          }
          ns$ch_callback_mcmc_(opts, fn, proposal, gradient)
        } else if (identical(construct$group, "gmm")) {
          # `callback` names the MOMENT CONDITION function -- this group's required delegate, its
          # counterpart of the mcmc group's log-likelihood -- and the two optional ones have keys
          # of their own. An absent key stays NULL, which is what "the ported numerical Jacobian"
          # and "no penalty" mean.
          jacobian <- if (is.null(construct$jacobian)) {
            NULL
          } else {
            callback_fixture_function(construct$jacobian)
          }
          penalty <- if (is.null(construct$penalty)) {
            NULL
          } else {
            callback_fixture_function(construct$penalty)
          }
          ns$ch_callback_gmm_(opts, fn, jacobian, penalty)
        } else if (identical(construct$group, "bootstrap")) {
          # `callback` names the RESAMPLE delegate -- the one handed the generator, this group's
          # counterpart of the mcmc group's log-likelihood -- and the other four have keys of
          # their own. An absent `jackknife` stays NULL, which is what every method but BCa means;
          # `fit` and `fit_with_covariance` are the two fitting delegates the run types take, and
          # a case supplies exactly the one its own `run_type` needs.
          optional <- function(name) {
            if (is.null(construct[[name]])) NULL else callback_fixture_function(construct[[name]])
          }
          ns$ch_callback_bootstrap_(
            opts, fn,
            optional("fit"),
            callback_fixture_function(construct$statistic),
            optional("jackknife"),
            optional("fit_with_covariance")
          )
        } else {
          stop(sprintf("unknown callback fixture group: %s", construct$group))
        }
        for (a in case$assertions) {
          idx <- if (is.null(a$args)) 1L else a$args[[1]] + 1L
          if (identical(a$method, "value")) {
            check_assertion(r$values[[idx]], a)
          } else if (identical(a$method, "named")) {
            # By label, not position: the mcmc group's summary block is long and its indices
            # shift with the chain and parameter counts.
            at <- match(a$name, as.character(unlist(r$names)))
            if (is.na(at)) stop(sprintf("callback: no result named '%s'", a$name))
            check_assertion(r$values[[at]], a)
          } else if (identical(a$method, "dim")) {
            check_assertion(as.double(r$dims[[idx]]), a)
          } else if (identical(a$method, "status")) {
            expect_identical(r$status, a$expected)
          } else {
            stop(sprintf("unknown callback fixture assertion method: %s", a$method))
          }
        }
      }
      next
    }
    if (identical(spec$kind, "toolbox_cross_language")) {
      # fixtures/toolbox/toolbox_cross_language.json's one case nests "optimizer" (shaped like an
      # "optimizer"-kind construct/assertions), "sobol" and "stratify" (each shaped like a
      # "toolbox"-kind group-"sampling" case's options/assertions) under one case name, so the
      # single fixture proves all three reproduce identically across languages in one guarantee.
      # Reuses optimizer_spec_json/optimizer_fixture_objective and ch_toolbox_run_/toolbox_select
      # verbatim -- see the "optimizer" and "toolbox" blocks above.
      ns <- asNamespace("corehydror")
      for (case in spec$cases) {
        opt <- case$optimizer
        construct <- opt$construct
        objective_name <- if (is.null(construct$objective)) "DeJong" else construct$objective
        construct$objective <- NULL
        r <- ns$ch_optim_run_(optimizer_spec_json(ns, construct),
                              optimizer_fixture_objective(objective_name))
        for (a in opt$assertions) {
          if (identical(a$method, "value")) {
            check_assertion(r$value, a)
          } else if (identical(a$method, "parameter")) {
            check_assertion(r$parameters[[a$args[[1]] + 1L]], a)
          } else if (identical(a$method, "status")) {
            expect_identical(r$status, a$expected)
          } else {
            stop(sprintf("unknown toolbox_cross_language optimizer assertion method: %s", a$method))
          }
        }
        for (sub in c("sobol", "stratify")) {
          block <- case[[sub]]
          opts_list <- if (is.null(block$options)) list() else block$options
          if (identical(sub, "sobol")) {
            opts_list$path <- system.file("extdata", "new-joe-kuo-6.21201", package = "corehydror")
          }
          opts <- if (length(opts_list) == 0L) "{}" else ns$to_spec_json(opts_list)
          for (a in block$assertions) {
            r <- ns$ch_toolbox_run_("sampling", sub, list(), opts)
            check_assertion(toolbox_select(r, a, "sampling"), a)
          }
        }
      }
      next
    }
    if (identical(spec$kind, "goodness_of_fit")) {
      datasets <- spec$datasets
      for (case in spec$cases) {
        fn <- case[["function"]]
        args <- if (is.null(case$args)) list() else case$args
        obs <- if (is.null(case$observed_dataset)) numeric() else
          as.double(unlist(datasets[[case$observed_dataset]]))
        mod <- if (is.null(case$modeled_dataset)) numeric() else
          as.double(unlist(datasets[[case$modeled_dataset]]))
        actual <- dispatch_goodness_of_fit(fn, args, obs, mod)
        for (a in case$assertions) check_assertion(actual, a)
      }
      next
    }
    if (identical(spec$kind, "data_utility")) {
      datasets <- spec$datasets
      for (case in spec$cases) {
        args <- if (is.null(case$args)) numeric() else as.double(unlist(case$args))
        data <- if (is.null(case$dataset)) numeric() else as.double(unlist(datasets[[case$dataset]]))
        fn <- case[["function"]]
        actual <- if (startsWith(fn, "MRL") || startsWith(fn, "GPDStability")) {
          dispatch_threshold_diagnostic(fn, args, data)
        } else {
          dispatch_data_utility(fn, args, data)
        }
        for (a in case$assertions) check_assertion(actual, a)
      }
      next
    }
    if (identical(spec$kind, "mcmc_sampler")) {
      target <- spec$target
      datasets <- spec$datasets
      for (case in spec$cases) {
        run_mcmc_case(target, case$construct, case$assertions, datasets)
      }
      next
    }
    if (identical(spec$kind, "bivariate_copula")) {
      target <- spec$target
      datasets <- spec$datasets
      for (case in spec$cases) {
        run_copula_case(target, case$construct, case$assertions, datasets)
      }
      next
    }
    if (identical(spec$kind, "multivariate_distribution")) {
      target <- spec$target
      for (case in spec$cases) {
        run_multivariate_case(target, case$construct, case$assertions)
      }
      next
    }
    if (!identical(spec$kind, "univariate_distribution")) next
    target <- spec$target
    datasets <- spec$datasets
    is_composite <- target %in% kCompositeTargets
    for (case in spec$cases) {
      if (is_composite) {
        run_composite_case(target, case$construct, case$assertions, datasets)
      } else {
        p <- build_params(target, case$construct, datasets)
        for (a in case$assertions) {
          args <- if (is.null(a$args)) list() else a$args
          actual <- if (target == "GeneralizedExtremeValue") {
            dispatch_gev(p, a$method, args)
          } else {
            dispatch_generic(target, p, a$method, args)
          }
          check_assertion(actual, a)
        }
      }
    }
  }
})
