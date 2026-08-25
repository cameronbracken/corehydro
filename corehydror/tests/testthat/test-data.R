# Behavioural tests for the data layer: analysis_data(), its summary, and the threshold
# diagnostics. Numeric oracles live in fixtures/ (threshold_diagnostics.json and the censored
# estimation fixtures); what is checked here is the R surface -- spec assembly, argument
# validation, and the invariants a caller relies on.

peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
           19200, 13800, 25600, 10500, 16900)

test_that("analysis_data assembles each series and indexes them from zero", {
  d <- analysis_data(
    exact = peaks,
    interval = data.frame(lower = 30000, value = 35000, upper = 40000, index = 15),
    threshold = data.frame(start_index = 15, end_index = 54, value = 28000, number_above = 1),
    uncertain = list(distribution("Normal", c(14000, 2000)))
  )
  expect_s3_class(d, "corehydro_data")
  expect_length(d$exact, length(peaks))
  expect_equal(d$exact[[1]]$index, 0L)
  expect_equal(d$exact[[length(peaks)]]$index, length(peaks) - 1L)
  expect_equal(d$exact[[1]]$value, peaks[1])
  expect_equal(d$interval[[1]]$index, 15L)
  expect_equal(d$threshold[[1]]$number_above, 1L)
  expect_equal(d$uncertain[[1]]$distribution$family, "Normal")
})

test_that("analysis_data accepts a plain vector or a data frame for the exact series", {
  a <- analysis_data(exact = peaks)
  b <- analysis_data(exact = data.frame(value = peaks))
  expect_identical(a, b)
})

test_that("analysis_data rejects malformed input", {
  expect_error(analysis_data(), "at least one of")
  expect_error(analysis_data(interval = data.frame(lower = 1, upper = 3)), "`value` column")
  expect_error(
    analysis_data(interval = data.frame(lower = 5, value = 1, upper = 9)),
    "lower <= value <= upper"
  )
  expect_error(
    analysis_data(threshold = data.frame(start_index = 5, end_index = 1, value = 1,
                                         number_above = 0)),
    "end_index >= start_index"
  )
  expect_error(analysis_data(exact = peaks, index = 1:3), "unused argument")
  expect_error(
    analysis_data(exact = data.frame(value = peaks, index = seq_along(peaks) - 2L)),
    "non-negative"
  )
  expect_error(analysis_data(exact = peaks, uncertain = list(1, 2)), "corehydro_dist")
  expect_error(
    analysis_data(peaks, low_outlier_threshold = 100, mgbt_low_outliers = TRUE),
    "do not also supply"
  )
})

test_that("the summary reports the record diagnostics", {
  s <- analysis_data_summary(analysis_data(peaks))
  expect_equal(s$value, peaks)
  expect_equal(s$index, seq_along(peaks) - 1L)
  expect_equal(s$exact_count, length(peaks))
  expect_equal(s$total_record_length, length(peaks))
  expect_equal(s$number_of_low_outliers, 0L)
  expect_true(all(s$plotting_position > 0 & s$plotting_position < 1))
  # Weibull positions on an uncensored record are rank / (n + 1).
  expect_equal(sort(s$plotting_position), seq_len(length(peaks)) / (length(peaks) + 1))
})

test_that("a plain vector and an equivalent frame summarize identically", {
  expect_identical(analysis_data_summary(peaks), analysis_data_summary(analysis_data(peaks)))
})

test_that("censored series lengthen the record and shift the plotting positions", {
  plain <- analysis_data_summary(analysis_data(peaks))
  censored <- analysis_data_summary(analysis_data(
    exact = peaks,
    threshold = data.frame(start_index = 15, end_index = 54, value = 28000, number_above = 1)
  ))
  expect_equal(censored$total_record_length, 55L)
  expect_equal(censored$threshold_count, 1L)
  expect_false(isTRUE(all.equal(plain$plotting_position, censored$plotting_position)))
})

test_that("the plotting parameter selects the convention and is range-checked", {
  weibull <- analysis_data_summary(analysis_data(peaks))
  cunnane <- analysis_data_summary(analysis_data(peaks), plotting_parameter = 0.40)
  expect_equal(weibull$plotting_parameter, 0)
  expect_equal(cunnane$plotting_parameter, 0.40)
  expect_false(isTRUE(all.equal(weibull$plotting_position, cunnane$plotting_position)))
  expect_error(analysis_data_summary(analysis_data(peaks), plotting_parameter = 1), "\\[0, 1\\)")
  expect_error(analysis_data_summary(analysis_data(peaks), plotting_parameter = -0.1), "\\[0, 1\\)")
})

test_that("MGBT censoring flags the low floods and matches an explicit threshold", {
  low <- c(peaks, 120, 95)
  expect_equal(mgbt_test(low), 2L)

  s <- analysis_data_summary(analysis_data(low, mgbt_low_outliers = TRUE))
  expect_equal(s$number_of_low_outliers, 2L)
  expect_equal(s$low_outlier_threshold, 8900)
  expect_equal(sum(s$is_low_outlier), 2L)
  expect_true(all(s$value[s$is_low_outlier] < 8900))

  # An explicit threshold at the value MGBT chose must censor the same points.
  explicit <- analysis_data_summary(analysis_data(low, low_outlier_threshold = 8900))
  expect_equal(explicit$number_of_low_outliers, 2L)
  expect_identical(explicit$is_low_outlier, s$is_low_outlier)
})

test_that("a threshold censoring more than half the record is rejected", {
  expect_error(
    analysis_data_summary(analysis_data(peaks, low_outlier_threshold = 20000)),
    "50 percent"
  )
})

test_that("threshold_diagnostics returns parallel vectors for each method", {
  x <- dist_random(distribution("GeneralizedPareto", c(0, 100, 0.1)), 300, seed = 4242)

  mrl <- threshold_diagnostics(x, u_min = 0, u_max = 200, n_thresholds = 8)
  expect_named(mrl, c("threshold", "exceedance_count", "mean_excess", "lower_ci", "upper_ci",
                      "modified_scale", "modified_scale_lower_ci", "modified_scale_upper_ci",
                      "shape", "shape_lower_ci", "shape_upper_ci"))
  n <- length(mrl$threshold)
  expect_true(n > 0 && n <= 8)
  expect_length(mrl$mean_excess, n)
  expect_length(mrl$exceedance_count, n)
  expect_length(mrl$modified_scale, 0)
  expect_true(all(mrl$lower_ci < mrl$mean_excess))
  expect_true(all(mrl$upper_ci > mrl$mean_excess))
  expect_true(all(diff(mrl$exceedance_count) <= 0))

  st <- threshold_diagnostics(x, u_min = 0, u_max = 200, n_thresholds = 8,
                              method = "parameter_stability")
  m <- length(st$threshold)
  expect_length(st$shape, m)
  expect_length(st$mean_excess, 0)
  expect_true(all(st$shape_lower_ci < st$shape))
  # The sample is GPD(0, 100, 0.1), so the fitted shape should sit near 0.1.
  expect_lt(abs(st$shape[1] - 0.1), 0.15)
})

test_that("threshold_diagnostics validates its method argument", {
  expect_error(threshold_diagnostics(peaks, 0, 1, method = "nope"), "'arg' should be one of")
})

test_that("print shows the series composition", {
  d <- analysis_data(exact = peaks, mgbt_low_outliers = TRUE)
  expect_output(print(d), "15 exact")
  expect_output(print(d), "Multiple Grubbs-Beck")
  expect_output(print(analysis_data(peaks, low_outlier_threshold = 9000)), "9000")
})

# Harricana River annual peaks (Bobee & Ashkar 1991, Table 1.2, page 5); the oracle for
# mann_kendall (0.7757) is ExactDataHypothesisTests.Test_MannKendall, also transcribed into
# fixtures/data/data_frame_facades.json and core/tests/test_data_frame_facades.cpp.
harricana <- c(
  122, 244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331, 225,
  174, 98.8, 149, 238, 262, 132, 235, 216, 240, 230, 192, 195, 172, 173, 172,
  153, 142, 317, 161, 201, 204, 194, 164, 183, 161, 167, 179, 185, 117, 192,
  337, 125, 166, 99.1, 202, 230, 158, 262, 154, 164, 182, 164, 183, 171, 250,
  184, 205, 237, 177, 239, 187, 180, 173, 174
)

test_that("analysis_data_hypothesis_test reproduces the Harricana Mann-Kendall oracle", {
  p <- analysis_data_hypothesis_test(harricana, "mann_kendall")
  expect_named(p, "mann_kendall")
  expect_equal(unname(p), 0.7757, tolerance = 1e-3)
})

test_that("analysis_data_hypothesis_test accepts a corehydro_data frame (exact series only)", {
  p_vec <- analysis_data_hypothesis_test(harricana, "mann_kendall")
  p_obj <- analysis_data_hypothesis_test(analysis_data(harricana), "mann_kendall")
  expect_equal(p_vec, p_obj)
})

test_that("analysis_data_hypothesis_test runs a two-sample test given an index", {
  p <- analysis_data_hypothesis_test(harricana, "mann_whitney", index = 50)
  expect_equal(unname(p), 0.5892, tolerance = 1e-2)
})

test_that("analysis_data_hypothesis_test validates method and the two-sample index", {
  expect_error(analysis_data_hypothesis_test(harricana, "not_a_method"), "'arg' should be one of")
  expect_error(analysis_data_hypothesis_test(harricana, "equal_variance_t"), "requires .index.")
})

test_that("analysis_data_statistics returns the twenty named summary statistics", {
  s <- analysis_data_statistics(peaks)
  expect_length(s$value, 20)
  expect_named(s$value, c(
    "Record Length", "Events Per Index (λ)", "Low Outliers", "Minimum", "Maximum", "Mean",
    "Std Dev", "Skewness", "Kurtosis", "Mean (of log)", "Std Dev (of log)", "Skewness (of log)",
    "Kurtosis (of log)", "1%", "5%", "25%", "50%", "75%", "95%", "99%"
  ))
  expect_true(is.finite(s$value[["Mean"]]))  # peaks has 15 >= 10 exact points
})

test_that("analysis_data_statistics(all_data = TRUE) uses the plotting-position fit", {
  exact_only <- analysis_data_statistics(harricana)
  all_data <- analysis_data_statistics(harricana, all_data = TRUE)
  expect_equal(unname(exact_only$value[["Record Length"]]), 69)
  expect_equal(unname(exact_only$value[["Minimum"]]), 98.8)
  expect_equal(unname(all_data$value[["Minimum"]]), 98.8)
  # The two methods use different estimators, so their means need not agree even with no
  # censored data in the frame.
  expect_false(isTRUE(all.equal(
    unname(exact_only$value[["Mean"]]), unname(all_data$value[["Mean"]])
  )))
})

test_that("analysis_data_statistics(standardized = TRUE) adds parallel standardized columns", {
  s <- analysis_data_statistics(harricana, standardized = TRUE)
  expect_length(s$standardized_value, length(harricana))
  expect_length(s$standardized_log10_value, length(harricana))
  expect_true(all(is.finite(s$standardized_value)))
  expect_true(all(is.finite(s$standardized_log10_value)))
})

test_that("analysis_data_statistics reports NaN for a short record", {
  short <- harricana[1:9]
  s <- analysis_data_statistics(short)
  expect_true(all(is.na(s$value)))
})
