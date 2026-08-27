# The eight ml_*() verbs (P5). Mirrors corehydropy/tests/test_ml.py case for case: the two files
# assert the same numbers and the same error text, so a one-sided change fails one of them.
#
# The oracle values here are the ones the C# test suite asserts (and that
# fixtures/ml/machine_learning.json pins through all four runners); these tests check the R
# BINDING reaches them, not the arithmetic, which the fixture and ctest suites own.

iris_train <- function() {
  # The 90-row training split the C# ML test files share, read from the installed fixture rather
  # than duplicated here.
  fx <- jsonlite::fromJSON(system.file("fixtures", "ml", "machine_learning.json",
                                       package = "corehydror"))
  matrix(fx$datasets$iris_train_flat, ncol = 4L, byrow = TRUE)
}
iris_test_x <- function() {
  fx <- jsonlite::fromJSON(system.file("fixtures", "ml", "machine_learning.json",
                                       package = "corehydror"))
  matrix(fx$datasets$iris_test_flat, ncol = 4L, byrow = TRUE)
}
iris_species_train <- function() {
  fx <- jsonlite::fromJSON(system.file("fixtures", "ml", "machine_learning.json",
                                       package = "corehydror"))
  fx$datasets$iris_species_train
}
iris_full <- function() {
  fx <- jsonlite::fromJSON(system.file("fixtures", "ml", "machine_learning.json",
                                       package = "corehydror"))
  matrix(fx$datasets$iris_flat, ncol = 4L, byrow = TRUE)
}

test_that("ml_kmeans reproduces the iris cluster means and label counts", {
  fit <- ml_kmeans(iris_full(), k = 3, seed = 12345)
  expect_equal(dim(fit$means), c(3L, 4L))
  expect_equal(fit$means[1, ], c(5.901613, 2.748387, 4.393548, 1.433871), tolerance = 1e-6)
  expect_equal(fit$means[2, ], c(6.850000, 3.073684, 5.742105, 2.071053), tolerance = 1e-6)
  expect_equal(fit$means[3, ], c(5.006000, 3.428000, 1.462000, 0.246000), tolerance = 1e-6)
  # Labels are 0-based in BOTH languages.
  expect_length(fit$labels, 150L)
  expect_true(all(fit$labels %in% 0:2))
  expect_equal(sum(fit$labels == 0L), 62L)
  expect_equal(sum(fit$labels == 1L), 38L)
  expect_equal(sum(fit$labels == 2L), 50L)
  expect_true(fit$iterations >= 2L)
})

test_that("ml_gaussian_mixture returns a weight simplex and one covariance per component", {
  fit <- ml_gaussian_mixture(iris_full(), k = 3, seed = 12345)
  expect_equal(sum(fit$weights), 1, tolerance = 1e-12)
  expect_equal(fit$weights, c(0.3005423, 0.3661243, 0.3333333), tolerance = 1e-2)
  expect_equal(dim(fit$means), c(3L, 4L))
  expect_length(fit$sigmas, 3L)
  for (s in fit$sigmas) {
    expect_equal(dim(s), c(4L, 4L))
    expect_equal(s, t(s), tolerance = 1e-12)  # symmetric
    expect_true(all(diag(s) > 0))
  }
  expect_true(is.finite(fit$log_likelihood))
  expect_true(all(fit$labels %in% 0:2))
})

test_that("ml_jenks_breaks classifies a small vector", {
  x <- c(1, 1.2, 1.4, 8, 8.3, 8.6, 30, 31, 32)
  fit <- ml_jenks_breaks(x, n_clusters = 3)
  expect_equal(fit$breaks, c(1.4, 8.6, 32))
  expect_equal(dim(fit$clusters), c(3L, 8L))
  expect_equal(colnames(fit$clusters),
               c("start_index", "end_index", "count", "min", "max", "sum", "average", "variance"))
  expect_equal(fit$clusters[, "count"], c(3, 3, 3))
  # The index columns are 0-based positions into the sorted data.
  expect_equal(unname(fit$clusters[1, "start_index"]), 0)
  expect_equal(unname(fit$clusters[3, "end_index"]), 8)
  expect_true(fit$gvf > 0.9 && fit$gvf < 1)
})

test_that("ml_naive_bayes reproduces the iris means and predictions", {
  fit <- ml_naive_bayes(iris_train(), iris_species_train(), newdata = iris_test_x())
  expect_equal(fit$classes, c(1, 2, 3))
  expect_equal(dim(fit$means), c(3L, 4L))
  expect_equal(fit$means[1, ], c(5.016667, 3.456667, 1.466667, 0.220000), tolerance = 1e-6)
  expect_equal(fit$standard_deviations[1, ],
               c(0.3097088, 0.3490710, 0.1881550, 0.08051558), tolerance = 1e-6)
  expect_equal(fit$priors, rep(30 / 90, 3), tolerance = 1e-6)
  expect_length(fit$prediction, 60L)
  # The two deliberate misclassifications the C# oracle records, at positions 43 and 54 (1-based).
  expect_equal(fit$prediction[43], 2)
  expect_equal(fit$prediction[54], 2)
  expect_equal(sum(fit$prediction == 3), 18L)

  # Without newdata there is no prediction element.
  trained <- ml_naive_bayes(iris_train(), iris_species_train())
  expect_null(trained$prediction)
  expect_equal(trained$means, fit$means)
})

test_that("ml_knn reproduces the iris classification oracle and its four result kinds", {
  pred <- ml_knn(iris_train(), iris_species_train(), newdata = iris_test_x(), k = 5,
                 regression = FALSE)
  expect_length(pred, 60L)
  expect_equal(pred[1:20], rep(1, 20))
  expect_equal(pred[34], 3)  # the one versicolor row the C# oracle assigns to virginica

  # Neighbors: 0-based training-row indices, one row per query, k columns.
  nb <- ml_knn(iris_train(), iris_species_train(), newdata = iris_test_x()[1:3, , drop = FALSE],
               k = 5, what = "neighbors")
  expect_equal(dim(nb), c(3L, 5L))
  expect_true(all(nb >= 0 & nb < 90))

  # A seeded bootstrap prediction is reproducible.
  b1 <- ml_knn(iris_train(), iris_species_train(), newdata = iris_test_x()[1:3, , drop = FALSE],
               k = 5, what = "bootstrap", seed = 99)
  b2 <- ml_knn(iris_train(), iris_species_train(), newdata = iris_test_x()[1:3, , drop = FALSE],
               k = 5, what = "bootstrap", seed = 99)
  expect_identical(b1, b2)

  # Intervals: four ordered columns.
  band <- ml_knn(iris_train(), iris_species_train(),
                 newdata = iris_test_x()[1:2, , drop = FALSE], k = 5, what = "intervals",
                 seed = 99, realizations = 25)
  expect_equal(colnames(band), c("lower", "median", "upper", "mean"))
  expect_true(all(band[, "lower"] <= band[, "median"]))
  expect_true(all(band[, "median"] <= band[, "upper"]))
})

test_that("ml_decision_tree and ml_random_forest separate a clean two-group problem", {
  x <- c(1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105)
  y <- c(10, 11, 10, 11, 10, 11, 100, 101, 100, 101, 100, 101)

  p <- ml_decision_tree(x, y, newdata = c(3, 104), seed = 7)
  expect_length(p, 2L)
  expect_true(p[1] < 50 && p[2] > 50)

  band <- ml_random_forest(x, y, newdata = c(3, 104), seed = 42, number_of_trees = 25)
  expect_equal(dim(band), c(2L, 4L))
  expect_equal(colnames(band), c("lower", "median", "upper", "mean"))
  expect_true(all(band[, "lower"] <= band[, "median"]))
  expect_true(all(band[, "median"] <= band[, "upper"]))
  expect_true(band[1, "upper"] < 50 && band[2, "lower"] > 50)

  # A seeded forest is reproducible, including the mean column.
  again <- ml_random_forest(x, y, newdata = c(3, 104), seed = 42, number_of_trees = 25)
  expect_identical(band, again)

  # A classifier floors every column, including mean.
  clf <- ml_random_forest(x, c(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1), newdata = c(3, 104),
                          seed = 42, regression = FALSE, number_of_trees = 25)
  expect_true(all(clf == floor(clf)))
})

test_that("ml_glm reproduces the fpp3 simple-regression oracle across links", {
  fx <- jsonlite::fromJSON(system.file("fixtures", "ml", "machine_learning.json",
                                       package = "corehydror"))
  fit <- ml_glm(fx$datasets$fpp3_income, fx$datasets$fpp3_consumption)
  expect_equal(fit$coefficients, c(0.54510, 0.28060), tolerance = 1e-3)
  expect_equal(fit$standard_errors, c(0.05569, 0.04744), tolerance = 1e-3)
  expect_equal(fit$sigma, 0.6026, tolerance = 1e-3)
  expect_equal(fit$df, 185L)
  expect_equal(fit$n, 187L)
  expect_equal(dim(fit$vcov), c(2L, 2L))
  expect_length(fit$residuals, 187L)
  # z = beta / se, and vcov's diagonal underlies the standard errors.
  expect_equal(fit$z_values, fit$coefficients / fit$standard_errors, tolerance = 1e-12)
  expect_true(fit$aic < fit$aicc && fit$aicc < fit$bic)

  # Poisson (log link), the case whose standard errors are NOT scaled by sigma.
  pois <- ml_glm(matrix(fx$datasets$glm_drivers_popden_flat, ncol = 2L, byrow = TRUE),
                 fx$datasets$glm_deaths, link = "log")
  expect_equal(pois$coefficients, c(6.322e+00, 2.405e-03, -1.480e-04), tolerance = 1e-2)
  expect_equal(pois$aic, 4069.4, tolerance = 1e-2)
  expect_equal(pois$df, 23L)

  # newdata adds a prediction and a three-column band.
  with_nd <- ml_glm(fx$datasets$fpp3_income, fx$datasets$fpp3_consumption, newdata = c(0, 1, 2))
  expect_length(with_nd$prediction, 3L)
  expect_equal(colnames(with_nd$prediction_intervals), c("lower", "mean", "upper"))
  expect_equal(with_nd$prediction_intervals[, "mean"], with_nd$prediction)

  # robust_se moves the standard errors but not the coefficients.
  robust <- ml_glm(fx$datasets$fpp3_income, fx$datasets$fpp3_consumption, robust_se = TRUE)
  expect_equal(robust$coefficients, fit$coefficients)
  expect_false(isTRUE(all.equal(robust$standard_errors, fit$standard_errors)))
})

test_that("the ml verbs validate their arguments", {
  x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  y <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

  expect_error(ml_kmeans(letters, k = 2), "`x` must be numeric")
  expect_error(ml_jenks_breaks(letters, n_clusters = 2), "`x` must be numeric")
  expect_error(ml_decision_tree(x, y[1:5], newdata = 1),
               "`y` must be numeric with one value per row of `x`")
  expect_error(ml_decision_tree(x, y, newdata = cbind(1, 2)),
               "`newdata` must have the same number of columns as `x`")
  expect_error(ml_knn(x, y, newdata = 1, k = 2, what = "nope"), "unknown what")
  expect_error(ml_glm(x, y, link = "nope"), "unknown link")
  expect_error(ml_glm(x, y, local_method = "nope"), "unknown local_method")
  # Upstream's own guards reach the caller intact.
  expect_error(ml_kmeans(x, k = 2, seed = 1) |> suppressWarnings(), NA)
  expect_error(ml_jenks_breaks(x, n_clusters = 11),
               "cannot be greater than the length of the data array")
  expect_error(ml_decision_tree(x[1:9], y[1:9], newdata = 1),
               "at least ten training data points")
})
