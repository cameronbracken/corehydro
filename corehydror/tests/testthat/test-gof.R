test_that("goodness_of_fit returns every metric, and subsets agree with the whole", {
  obs <- c(2, 4, 6, 8, 10)
  mod <- c(2.2, 3.9, 6.4, 7.5, 10.1)
  all <- goodness_of_fit(obs, mod)
  expect_length(all, 17L)
  expect_true(all(c("nse", "kge", "rsr", "ve") %in% names(all)))
  expect_equal(goodness_of_fit(obs, mod, metrics = "nse")[["nse"]], all[["nse"]])
})

test_that("a subset that excludes mape succeeds on a zero-containing series", {
  obs <- c(0, 2, 4)
  mod <- c(0.1, 2.1, 3.9)
  expect_error(goodness_of_fit(obs, mod), "zero")
  result <- goodness_of_fit(obs, mod, metrics = "mse")
  expect_true(is.finite(result[["mse"]]))
})

test_that("a subset returns the same values as the full call, in the requested order", {
  obs <- c(2, 4, 6, 8, 10)
  mod <- c(2.2, 3.9, 6.4, 7.5, 10.1)
  all <- goodness_of_fit(obs, mod)
  subset <- goodness_of_fit(obs, mod, metrics = c("kge", "nse", "mse"))
  expect_equal(names(subset), c("kge", "nse", "mse"))
  expect_equal(unname(subset[["kge"]]), unname(all[["kge"]]))
  expect_equal(unname(subset[["nse"]]), unname(all[["nse"]]))
  expect_equal(unname(subset[["mse"]]), unname(all[["mse"]]))
})

test_that("an unknown metric name is rejected and names the offender", {
  expect_error(goodness_of_fit(1:5, 1:5, metrics = "nsee"), "nsee")
})

test_that("mismatched lengths are rejected before reaching C++", {
  expect_error(goodness_of_fit(1:5, 1:4), "same length")
})

test_that("classification_metrics on identical binary labels", {
  obs <- c(1, 0, 1, 1, 0)
  result <- classification_metrics(obs, obs)
  expected_names <- c("accuracy", "precision", "recall", "f1", "specificity",
                       "balanced_accuracy")
  expect_true(all(expected_names %in% names(result)))
  expect_equal(unname(result[["accuracy"]]), 100)
})

test_that("gof_test accepts a corehydro_dist and rejects anything else", {
  x <- c(2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9)
  d <- distribution("Normal", c(3.4, 1.1))
  expect_true(gof_test(x, d, "ks") > 0)
  expect_true(gof_test(x, d, "ad") > 0)
  expect_error(gof_test(x, "Normal"), "corehydro_dist")
})

test_that("the information criteria match the closed forms", {
  expect_equal(aic(k = 2, log_likelihood = -100), 204)
  expect_equal(sum(aic_weights(c(246, 248.8, 251.2))), 1)
})
