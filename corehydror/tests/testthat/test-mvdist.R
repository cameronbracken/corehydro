# Behavioural tests for the multivariate surface. Oracle VALUES live in fixtures/ and are
# asserted by test-fixtures.R; this file asserts shape, argument handling, and error messages.

S <- matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), nrow = 3)

test_that("mvdist_normal builds and evaluates", {
  mv <- mvdist_normal(c(1, 2, 3), S)
  expect_s3_class(mv, "corehydro_mvdist")
  expect_equal(mvdist_dimension(mv), 3L)
  expect_equal(mvdist_mean(mv), c(1, 2, 3))
  expect_equal(dim(mvdist_covariance(mv)), c(3L, 3L))
  expect_true(is.finite(mvdist_pdf(mv, c(1, 2, 3))))
})

test_that("marginal takes 1-based indices and returns a new object", {
  mv <- mvdist_normal(c(1, 2, 3), S)
  m <- mvdist_marginal(mv, c(1, 3))
  expect_s3_class(m, "corehydro_mvdist")
  expect_equal(mvdist_dimension(m), 2L)
  expect_equal(mvdist_mean(m), c(1, 3))
})

test_that("a fractional or repeated index is refused rather than truncated", {
  mv <- mvdist_normal(c(1, 2, 3), S)
  expect_error(mvdist_marginal(mv, c(1.9, 3)), "whole numbers")
  expect_error(mvdist_marginal(mv, c(1, 1)), "must not repeat")
  expect_error(mvdist_marginal(mv, c(1, 4)), "between 1 and 3")
  expect_error(mvdist_marginal(mv, integer(0)), "non-empty")
  expect_error(mvdist_conditional(mv, given = c(2.5), values = 5), "whole numbers")
  expect_error(mvdist_conditional(mv, given = c(2, 2), values = c(5, 5)), "must not repeat")
})

test_that("conditional takes 1-based indices and returns the complement", {
  mv <- mvdist_normal(c(1, 2, 3), S)
  cd <- mvdist_conditional(mv, given = 2, values = 5)
  expect_equal(mvdist_dimension(cd), 2L)
  expect_equal(mvdist_mean(cd), c(1, 3))
})

test_that("interval integrates to one over the whole space", {
  mv <- mvdist_normal(c(0, 0), diag(2))
  expect_equal(mvdist_interval(mv, c(-100, -100), c(100, 100)), 1, tolerance = 1e-6)
})

test_that("a seeded multivariate draw has the right shape and repeats", {
  mv <- mvdist_normal(c(0, 0), diag(2))
  d <- mvdist_random(mv, 4, seed = 12345)
  expect_equal(dim(d), c(4L, 2L))
  expect_identical(d, mvdist_random(mv, 4, seed = 12345))
  expect_equal(dim(mvdist_random(mv, 4, seed = 12345, method = "latin_hypercube")), c(4L, 2L))
})

test_that("latin hypercube needs an explicit seed", {
  mv <- mvdist_normal(c(0, 0), diag(2))
  expect_error(mvdist_random(mv, 4, method = "latin_hypercube"), "seed")
})

test_that("the other four families build", {
  expect_true(is.finite(mvdist_pdf(mvdist_dirichlet(c(2, 3, 4)), c(0.2, 0.3, 0.5))))
  expect_true(is.finite(mvdist_pdf(mvdist_multinomial(10, c(0.2, 0.3, 0.5)), c(2, 3, 5))))
  expect_equal(mvdist_dimension(mvdist_student_t(5, c(0, 0))), 2L)
})

test_that("the upstream gaps are named, not leaked", {
  expect_error(mvdist_cdf(mvdist_dirichlet(c(2, 3)), c(0.5, 0.5)), "Dirichlet")
  expect_error(mvdist_marginal(mvdist_student_t(5, c(0, 0)), 1), "MultivariateStudentT")
  expect_error(mvdist_interval(mvdist_student_t(5, c(0, 0)), c(-1, -1), c(1, 1)),
               "MultivariateStudentT")
})
