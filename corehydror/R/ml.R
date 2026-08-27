# The "ml" toolbox group (P5): the whole ported Numerics.MachineLearning namespace -- five
# supervised learners and three unsupervised ones -- over
# core/include/corehydro/numerics/support/toolbox/ml.hpp. Its own file rather than an addition to
# R/toolbox.R for the same reason regression.R is separate: these eight verbs share a matrix
# input contract and a set of helpers that no other toolbox verb uses.
#
# Mirrors corehydropy's own ml module; both packages share every signature, default and error
# message, so a change here is not one-sided.

# Internal: coerce a predictor argument to a matrix and flatten it ROW-MAJOR, which is the layout
# the shared runner assumes (see ml.hpp's file header). R is column-major, so `t(x)` is the
# transpose that produces it -- exactly what linear_regression() does for the same reason.
ml_matrix <- function(x, arg = "x") {
  if (is.data.frame(x)) x <- as.matrix(x)
  # Check BEFORE coercing: `as.double()` on a character vector returns NAs with a warning rather
  # than erroring, so coercing first would let `ml_matrix(letters)` through as an all-NA numeric
  # matrix -- while numpy's `asarray(..., dtype=float)` raises on the Python side. Checking first
  # keeps the two languages' behaviour identical, which is the point of these paired verbs.
  if (!is.numeric(x)) {
    stop(sprintf("`%s` must be numeric", arg), call. = FALSE)
  }
  if (!is.matrix(x)) x <- matrix(as.double(x), ncol = 1L)
  storage.mode(x) <- "double"
  x
}

ml_flat <- function(x) as.double(t(x))

# Internal: the shared validation every supervised verb performs.
ml_check_xy <- function(x, y) {
  if (!is.numeric(y) || length(y) != nrow(x)) {
    stop(sprintf(
      "`y` must be numeric with one value per row of `x`; got %d and %d",
      length(y), nrow(x)
    ), call. = FALSE)
  }
  invisible(NULL)
}

# Internal: validate and flatten `newdata` against a training matrix's column count.
ml_newdata <- function(newdata, x) {
  nd <- ml_matrix(newdata, "newdata")
  if (ncol(nd) != ncol(x)) {
    stop(sprintf(
      "`newdata` must have the same number of columns as `x`; got %d and %d",
      ncol(nd), ncol(x)
    ), call. = FALSE)
  }
  nd
}

# Internal: reshape a runner result carrying `dims` back into an R matrix. The runner flattens
# row-major; `byrow = TRUE` undoes it.
ml_reshape <- function(r) {
  matrix(r$values, nrow = r$dims[1], ncol = r$dims[2], byrow = TRUE)
}

#' k-means clustering
#'
#' Mirrors the C# `KMeans` class: partitions the rows of `x` into `k` clusters, each row
#' belonging to the cluster with the nearest centroid. Initialization is k-means++ by default.
#'
#' @details
#' Two behaviours inherited from upstream are worth knowing:
#' * `labels` are 0-BASED in both R and Python, matching the library's own indexing (the same
#'   choice [shortest_path()] made for node indices). Add 1 before using them to subset an R
#'   object.
#' * With `k = 1` the algorithm stops before its first update step, so the single reported
#'   "cluster mean" is a randomly chosen observation rather than the mean of `x`. Use `mean()`
#'   instead if that is what you want.
#'
#' @param x a numeric matrix or data frame with one row per observation, or a numeric vector for
#'   a single feature.
#' @param k the number of clusters.
#' @param seed integer PRNG seed. `NULL` (the default) uses the computer clock, so the fit is not
#'   reproducible; supply a seed for a reproducible fit.
#' @param kmeans_plus_plus use k-means++ initialization (the default) rather than a uniform draw.
#' @param max_iterations the iteration cap. Default 1000.
#' @return a list with `means` (a `k` by `ncol(x)` matrix of cluster centroids), `labels` (a
#'   0-based integer vector, one per row of `x`), and `iterations`.
#' @seealso [ml_gaussian_mixture()] for a covariance-aware generalization.
#' @examples
#' x <- cbind(c(1, 1.2, 0.8, 8, 8.3, 7.9), c(2, 2.1, 1.9, 9, 9.2, 8.8))
#' fit <- ml_kmeans(x, k = 2, seed = 12345)
#' fit$means
#' @export
ml_kmeans <- function(x, k, seed = NULL, kmeans_plus_plus = TRUE, max_iterations = 1000) {
  x <- ml_matrix(x)
  opts <- list(
    rows = nrow(x), columns = ncol(x), k = as.integer(k),
    seed = as.integer(if (is.null(seed)) -1L else seed),
    kmeans_plus_plus = isTRUE(kmeans_plus_plus),
    max_iterations = as.integer(max_iterations)
  )
  flat <- list(ml_flat(x))
  list(
    means = ml_reshape(toolbox_run("ml", "kmeans_means", flat, opts)),
    labels = as.integer(toolbox_run("ml", "kmeans_labels", flat, opts)$values),
    iterations = as.integer(toolbox_run("ml", "kmeans_iterations", flat, opts)$values[1])
  )
}

#' Gaussian mixture model
#'
#' Mirrors the C# `GaussianMixtureModel` class: fits a mixture of `k` multivariate normals by
#' expectation-maximization, initialized from a k-means fit. It generalizes [ml_kmeans()] by
#' carrying a full covariance per component rather than only a centre.
#'
#' @details
#' `labels` are 0-BASED, as in [ml_kmeans()]. `log_likelihood` follows the library's own
#' definition, which OMITS the multivariate-normal normalizing constant
#' `-0.5 * ncol(x) * log(2 * pi)` per observation -- so it is short of the true mixture
#' log-likelihood by `nrow(x) * ncol(x) / 2 * log(2 * pi)`. That constant cancels when comparing
#' two fits of the same data (which is what the library uses it for), but do not feed this value
#' to an information criterion without adding it back. A run that hits `max_iterations` without
#' converging reports `log_likelihood` as 0.
#'
#' @param x a numeric matrix or data frame with one row per observation, or a numeric vector.
#' @param k the number of mixture components.
#' @param seed integer PRNG seed; `NULL` (the default) uses the computer clock.
#' @param kmeans_plus_plus use k-means++ initialization for the starting fit. Default `TRUE`.
#' @param max_iterations the EM iteration cap. Default 1000.
#' @param tolerance the relative convergence tolerance on the log-likelihood. Default 1e-8.
#' @return a list with `means` (`k` by `ncol(x)`), `sigmas` (a length-`k` list of
#'   `ncol(x)` by `ncol(x)` covariance matrices), `weights`, `labels` (0-based),
#'   `log_likelihood`, and `iterations`.
#' @examples
#' x <- c(1, 1.2, 0.9, 1.1, 1.05, 8, 8.3, 7.9, 8.1, 8.2, 1.0, 8.0)
#' fit <- ml_gaussian_mixture(x, k = 2, seed = 12345)
#' round(fit$weights, 3)
#' @export
ml_gaussian_mixture <- function(x, k, seed = NULL, kmeans_plus_plus = TRUE,
                                max_iterations = 1000, tolerance = 1e-8) {
  x <- ml_matrix(x)
  opts <- list(
    rows = nrow(x), columns = ncol(x), k = as.integer(k),
    seed = as.integer(if (is.null(seed)) -1L else seed),
    kmeans_plus_plus = isTRUE(kmeans_plus_plus),
    max_iterations = as.integer(max_iterations),
    tolerance = as.double(tolerance)
  )
  flat <- list(ml_flat(x))
  stacked <- ml_reshape(toolbox_run("ml", "gmm_sigmas", flat, opts))
  p <- ncol(x)
  sigmas <- lapply(seq_len(k), function(i) stacked[((i - 1L) * p + 1L):(i * p), , drop = FALSE])
  list(
    means = ml_reshape(toolbox_run("ml", "gmm_means", flat, opts)),
    sigmas = sigmas,
    weights = toolbox_run("ml", "gmm_weights", flat, opts)$values,
    labels = as.integer(toolbox_run("ml", "gmm_labels", flat, opts)$values),
    log_likelihood = toolbox_run("ml", "gmm_log_likelihood", flat, opts)$values[1],
    iterations = as.integer(toolbox_run("ml", "gmm_iterations", flat, opts)$values[1])
  )
}

#' Jenks natural breaks classification
#'
#' Mirrors the C# `JenksNaturalBreaks` class: partitions a one-dimensional sample into
#' `n_clusters` classes minimizing the within-class sum of squared deviations.
#'
#' @details
#' Upstream fails on fully degenerate input: if every value of `x` is identical, the underlying
#' algorithm computes a negative class boundary and errors. Any spread at all is fine, including
#' heavily tied data.
#'
#' @param x a numeric vector to classify.
#' @param n_clusters the number of classes. Must be at least 1 and no more than `length(x)`.
#' @param is_data_sorted set `TRUE` to skip the internal sort when `x` is already ascending.
#' @return a list with `breaks` (each class's maximum value), `clusters` (an `n_clusters` by 8
#'   matrix with columns `start_index`, `end_index`, `count`, `min`, `max`, `sum`, `average`,
#'   `variance`; the two index columns are 0-based positions into the SORTED data), and `gvf`,
#'   the goodness-of-variance-fit measure, which approaches 1 for a better fit.
#' @examples
#' x <- c(1, 1.2, 1.4, 8, 8.3, 8.6, 30, 31, 32)
#' ml_jenks_breaks(x, n_clusters = 3)$breaks
#' @export
ml_jenks_breaks <- function(x, n_clusters, is_data_sorted = FALSE) {
  if (!is.numeric(x)) {
    stop("`x` must be numeric", call. = FALSE)
  }
  opts <- list(n_clusters = as.integer(n_clusters), is_data_sorted = isTRUE(is_data_sorted))
  data <- list(as.double(x))
  clusters <- ml_reshape(toolbox_run("ml", "jenks_clusters", data, opts))
  colnames(clusters) <- c(
    "start_index", "end_index", "count", "min", "max", "sum", "average", "variance"
  )
  list(
    breaks = toolbox_run("ml", "jenks_breaks", data, opts)$values,
    clusters = clusters,
    gvf = toolbox_run("ml", "jenks_gvf", data, opts)$values[1]
  )
}

#' Decision tree regression or classification
#'
#' Mirrors the C# `DecisionTree` class: recursively splits the training data on the feature and
#' threshold that most reduce variance (regression) or most increase information gain
#' (classification), then predicts by walking a new observation down the tree.
#'
#' @details
#' At the library's defaults a REGRESSION tree recurses until every leaf holds a single training
#' observation, so it memorizes the training data and generalizes poorly. That is upstream's
#' behaviour, not a port artifact, and it is why [ml_random_forest()] exists. Set
#' `minimum_split_size` or `max_depth` to regularize it.
#'
#' @param x a numeric matrix or data frame of training predictors, one row per observation.
#' @param y the training response, one value per row of `x`.
#' @param newdata predictors to predict for, with the same number of columns as `x`.
#' @param seed integer PRNG seed for the random feature subsets; `NULL` uses the computer clock.
#' @param regression `TRUE` (the default) fits a regression tree; `FALSE` a classifier.
#' @param features the number of random features to consider at each split. `NULL` (the default)
#'   uses the library's own `max(1, ncol(x) - 1)`.
#' @param minimum_split_size the smallest node the tree will split. Default 2.
#' @param max_depth the recursion cap. Default 100.
#' @return a numeric vector of predictions, one per row of `newdata`.
#' @seealso [ml_random_forest()], which averages many bootstrapped trees.
#' @examples
#' x <- c(1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105)
#' y <- c(10, 10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 100)
#' ml_decision_tree(x, y, newdata = c(3, 104), seed = 7)
#' @export
ml_decision_tree <- function(x, y, newdata, seed = NULL, regression = TRUE, features = NULL,
                             minimum_split_size = 2, max_depth = 100) {
  x <- ml_matrix(x)
  ml_check_xy(x, y)
  nd <- ml_newdata(newdata, x)
  opts <- list(
    rows = nrow(x), columns = ncol(x), predict_rows = nrow(nd),
    seed = as.integer(if (is.null(seed)) -1L else seed),
    is_regression = isTRUE(regression),
    minimum_split_size = as.integer(minimum_split_size),
    max_depth = as.integer(max_depth)
  )
  if (!is.null(features)) opts$features <- as.integer(features)
  toolbox_run("ml", "decision_tree_predict",
              list(ml_flat(x), as.double(y), ml_flat(nd)), opts)$values
}

#' Random forest regression or classification
#'
#' Mirrors the C# `RandomForest` class: fits `number_of_trees` decision trees on bootstrap
#' resamples of the training data and reports the spread of their predictions as an interval.
#'
#' @details
#' Training cost is linear in `number_of_trees`, and the library's default of 1000 is the knob to
#' turn if a call is slow -- a few dozen trees is usually enough to see the shape of the answer.
#' A seeded run is bit-identical between R and Python, because the whole computation lives in the
#' shared compiled core.
#'
#' For a classifier every column is floored to an integer class label, including `mean`.
#'
#' @param x a numeric matrix or data frame of training predictors, one row per observation.
#' @param y the training response, one value per row of `x`.
#' @param newdata predictors to predict for, with the same number of columns as `x`.
#' @param seed integer PRNG seed; `NULL` uses the computer clock.
#' @param regression `TRUE` (the default) fits regression trees; `FALSE` classifiers.
#' @param features the number of random features per split. `NULL` uses `max(1, ncol(x) - 1)`.
#' @param minimum_split_size the smallest node a tree will split. Default 2.
#' @param max_depth the recursion cap. Default 100.
#' @param number_of_trees how many trees to grow. Default 1000.
#' @param alpha the interval level: 0.1 (the default) gives a 90% interval.
#' @return a matrix with one row per row of `newdata` and columns `lower`, `median`, `upper`,
#'   `mean`.
#' @examples
#' x <- c(1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105)
#' y <- c(10, 11, 10, 11, 10, 11, 100, 101, 100, 101, 100, 101)
#' ml_random_forest(x, y, newdata = c(3, 104), seed = 42, number_of_trees = 25)
#' @export
ml_random_forest <- function(x, y, newdata, seed = NULL, regression = TRUE, features = NULL,
                             minimum_split_size = 2, max_depth = 100, number_of_trees = 1000,
                             alpha = 0.1) {
  x <- ml_matrix(x)
  ml_check_xy(x, y)
  nd <- ml_newdata(newdata, x)
  opts <- list(
    rows = nrow(x), columns = ncol(x), predict_rows = nrow(nd),
    seed = as.integer(if (is.null(seed)) -1L else seed),
    is_regression = isTRUE(regression),
    minimum_split_size = as.integer(minimum_split_size),
    max_depth = as.integer(max_depth),
    number_of_trees = as.integer(number_of_trees),
    alpha = as.double(alpha)
  )
  if (!is.null(features)) opts$features <- as.integer(features)
  r <- toolbox_run("ml", "random_forest_predict",
                   list(ml_flat(x), as.double(y), ml_flat(nd)), opts)
  out <- ml_reshape(r)
  colnames(out) <- r$names
  out
}

#' k-nearest-neighbors regression or classification
#'
#' Mirrors the C# `KNearestNeighbors` class: predicts from the `k` training rows closest to each
#' new observation -- an inverse-squared-distance weighted average of their responses for
#' regression, or their most common response for classification.
#'
#' @param x a numeric matrix or data frame of training predictors, one row per observation.
#' @param y the training response, one value per row of `x`.
#' @param newdata predictors to predict for, with the same number of columns as `x`.
#' @param k the number of neighbors.
#' @param regression `TRUE` (the default) for a weighted average; `FALSE` for the modal class.
#' @param what which result to return: `"prediction"` (the default), `"neighbors"` (the indices
#'   of each new observation's neighbors), `"bootstrap"` (a prediction from one bootstrap
#'   resample of the training data), or `"intervals"` (bootstrapped prediction intervals).
#' @param seed integer PRNG seed, used by `"bootstrap"` and `"intervals"`; `NULL` uses the clock.
#' @param realizations the number of bootstrap resamples for `"intervals"`. Default 1000.
#' @param alpha the interval level for `"intervals"`: 0.1 (the default) gives a 90% interval.
#' @return for `"prediction"` and `"bootstrap"`, a numeric vector; for `"neighbors"`, a matrix of
#'   0-BASED training-row indices with one row per row of `newdata` and `k` columns; for
#'   `"intervals"`, a matrix with columns `lower`, `median`, `upper`, `mean`. Note `"intervals"`
#'   always reports percentiles, even for a classifier -- upstream has no classification branch
#'   there.
#' @examples
#' x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
#' y <- c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
#' ml_knn(x, y, newdata = 4.5, k = 2)
#' @export
ml_knn <- function(x, y, newdata, k, regression = TRUE,
                   what = c("prediction", "neighbors", "bootstrap", "intervals"),
                   seed = NULL, realizations = 1000, alpha = 0.1) {
  what <- check_choice(what[1], c("prediction", "neighbors", "bootstrap", "intervals"), "what")
  x <- ml_matrix(x)
  ml_check_xy(x, y)
  nd <- ml_newdata(newdata, x)
  opts <- list(
    rows = nrow(x), columns = ncol(x), predict_rows = nrow(nd), k = as.integer(k),
    is_regression = isTRUE(regression),
    seed = as.integer(if (is.null(seed)) -1L else seed),
    realizations = as.integer(realizations), alpha = as.double(alpha)
  )
  data <- list(ml_flat(x), as.double(y), ml_flat(nd))
  method <- switch(what,
    prediction = "knn_predict",
    neighbors = "knn_neighbors",
    bootstrap = "knn_bootstrap_predict",
    intervals = "knn_prediction_intervals"
  )
  r <- toolbox_run("ml", method, data, opts)
  if (identical(what, "neighbors")) {
    out <- ml_reshape(r)
    storage.mode(out) <- "integer"
    return(out)
  }
  if (identical(what, "intervals")) {
    out <- ml_reshape(r)
    colnames(out) <- r$names
    return(out)
  }
  r$values
}

#' Gaussian naive Bayes classification
#'
#' Mirrors the C# `NaiveBayes` class: assumes each feature is normally distributed given the
#' class and independent of the other features, then assigns each new observation the class with
#' the highest posterior probability.
#'
#' @details
#' `classes` follows the order the class labels FIRST APPEAR in `y`, not sorted order, and every
#' per-class row of `means`, `standard_deviations` and `priors` is indexed to match. A class with
#' a single member gets a standard deviation of 1e-6 rather than 0.
#'
#' @param x a numeric matrix or data frame of training predictors, one row per observation.
#' @param y the training class labels, one per row of `x`.
#' @param newdata predictors to classify, with the same number of columns as `x`. `NULL` (the
#'   default) trains without predicting.
#' @return a list with `classes`, `means` and `standard_deviations` (both one row per class and
#'   one column per feature), `priors`, and -- when `newdata` is supplied -- `prediction`, the
#'   predicted class label for each row of `newdata`.
#' @examples
#' x <- c(1, 1.1, 0.9, 1.2, 1.05, 5, 5.1, 4.9, 5.2, 5.05)
#' y <- c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
#' ml_naive_bayes(x, y, newdata = c(1.0, 5.0))$prediction
#' @export
ml_naive_bayes <- function(x, y, newdata = NULL) {
  x <- ml_matrix(x)
  ml_check_xy(x, y)
  opts <- list(rows = nrow(x), columns = ncol(x))
  data <- list(ml_flat(x), as.double(y))
  out <- list(
    classes = toolbox_run("ml", "naive_bayes_classes", data, opts)$values,
    means = ml_reshape(toolbox_run("ml", "naive_bayes_means", data, opts)),
    standard_deviations = ml_reshape(toolbox_run("ml", "naive_bayes_sds", data, opts)),
    priors = toolbox_run("ml", "naive_bayes_priors", data, opts)$values
  )
  if (!is.null(newdata)) {
    nd <- ml_newdata(newdata, x)
    opts$predict_rows <- nrow(nd)
    out$prediction <- toolbox_run("ml", "naive_bayes_predict",
                                  list(ml_flat(x), as.double(y), ml_flat(nd)), opts)$values
  }
  out
}

# The five GLM families, which are the five LinkFunctionType members GeneralizedLinearModel
# implements a log-likelihood for. The other two link types the Numerics link layer offers
# (yeo_johnson, fisher_z) have no GLM family and are rejected by name rather than failing inside
# the first likelihood evaluation.
kGlmLinks <- c("identity", "log", "logit", "probit", "complementary_log_log")
kGlmLocalMethods <- c("nelder_mead", "bfgs", "powell", "adam", "gradient_descent")

#' Generalized linear model
#'
#' Mirrors the C# `GeneralizedLinearModel` class: fits a linear predictor mapped to the response
#' scale by a link function, by maximizing the family log-likelihood with a local optimizer.
#'
#' @details
#' The `link` selects the family as well as the transform: `"identity"` is Normal, `"log"` is
#' Poisson, and `"logit"`, `"probit"` and `"complementary_log_log"` are Binomial.
#'
#' `local_method` accepts all five optimizers here, unlike [optim_minimize()]'s `local_method`
#' argument (which takes three) -- the two upstream classes construct different sets, and this
#' surface follows each one rather than imposing a single list.
#'
#' @param x a numeric matrix or data frame of predictors, one row per observation.
#' @param y the response, one value per row of `x`.
#' @param intercept fit an intercept term. Default `TRUE`.
#' @param link one of `"identity"`, `"log"`, `"logit"`, `"probit"`,
#'   `"complementary_log_log"`.
#' @param local_method the optimizer: one of `"nelder_mead"` (the default), `"bfgs"`, `"powell"`,
#'   `"adam"`, `"gradient_descent"`.
#' @param robust_se use the sandwich (heteroskedasticity-consistent) covariance rather than the
#'   delta-method one. Changes the standard errors, not the coefficients. Default `FALSE`.
#' @param newdata optional predictors to predict for. When supplied the result gains `prediction`
#'   and `prediction_intervals`.
#' @param alpha the interval level for `prediction_intervals`. Default 0.1, a 90% interval.
#' @return a list with `coefficients`, `standard_errors`, `z_values`, `p_values`, `sigma`, `df`,
#'   `n`, `aic`, `aicc`, `bic`, `vcov` and `residuals`; plus `prediction` and
#'   `prediction_intervals` (columns `lower`, `mean`, `upper`) when `newdata` is supplied.
#' @seealso [linear_regression()] for ordinary least squares by SVD.
#' @examples
#' x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
#' y <- c(2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.0, 20.2)
#' fit <- ml_glm(x, y)
#' round(fit$coefficients, 3)
#' @export
ml_glm <- function(x, y, intercept = TRUE, link = "identity", local_method = "nelder_mead",
                   robust_se = FALSE, newdata = NULL, alpha = 0.1) {
  link <- check_choice(link, kGlmLinks, "link")
  local_method <- check_choice(local_method, kGlmLocalMethods, "local_method")
  x <- ml_matrix(x)
  ml_check_xy(x, y)
  opts <- list(
    rows = nrow(x), columns = ncol(x), intercept = isTRUE(intercept), link = link,
    local_method = local_method, robust_se = isTRUE(robust_se), alpha = as.double(alpha)
  )
  data <- list(ml_flat(x), as.double(y))

  fit <- toolbox_run("ml", "glm_fit", data, opts)
  named <- stats::setNames(fit$values, fit$names)
  p <- sum(startsWith(fit$names, "beta_"))
  pick <- function(prefix) unname(named[paste0(prefix, seq_len(p))])

  out <- list(
    coefficients = pick("beta_"),
    standard_errors = pick("se_"),
    z_values = pick("z_"),
    p_values = pick("p_"),
    sigma = unname(named[["sigma"]]),
    df = as.integer(named[["df"]]),
    n = as.integer(named[["n"]]),
    aic = unname(named[["aic"]]),
    aicc = unname(named[["aicc"]]),
    bic = unname(named[["bic"]]),
    vcov = ml_reshape(toolbox_run("ml", "glm_covariance", data, opts)),
    residuals = toolbox_run("ml", "glm_residuals", data, opts)$values
  )
  if (!is.null(newdata)) {
    nd <- ml_newdata(newdata, x)
    opts$predict_rows <- nrow(nd)
    data3 <- list(ml_flat(x), as.double(y), ml_flat(nd))
    out$prediction <- toolbox_run("ml", "glm_predict", data3, opts)$values
    pi <- toolbox_run("ml", "glm_predict_intervals", data3, opts)
    band <- ml_reshape(pi)
    colnames(band) <- pi$names
    out$prediction_intervals <- band
  }
  out
}
