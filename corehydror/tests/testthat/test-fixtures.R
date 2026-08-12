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
    stop(sprintf("unknown fixture method: %s", method))
  )
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
    # Only validate univariate_distribution, multivariate_distribution,
    # bivariate_copula, mcmc_sampler, bootstrap, and model_estimation fixtures; skip other
    # kinds (e.g. special_function) which are validated in C++ only and are not exposed to
    # the R package.
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
