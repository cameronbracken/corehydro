# Behavioural tests for the copula surface. Oracle VALUES live in fixtures/ and are asserted by
# test-fixtures.R; this file asserts argument handling, error messages, and object shape.

set.seed(1)
x <- c(135.9, 104.1, 108.7, 99.3, 134.7, 91.0, 77.3, 115.4, 109.0, 79.0)
y <- c(1.9, 1.3, 1.4, 1.2, 1.8, 1.1, 0.9, 1.5, 1.4, 1.0)

test_that("copula() builds and evaluates", {
  cop <- copula("Clayton", theta = 2)
  expect_s3_class(cop, "corehydro_copula")
  expect_equal(cop$family, "Clayton")
  expect_true(is.finite(copula_pdf(cop, 0.3, 0.7)))
  expect_true(copula_cdf(cop, 0.3, 0.7) > 0)
  expect_equal(copula_params(cop), 2)
})

test_that("tail dependence comes back named", {
  td <- copula_tail_dependence(copula("Clayton", theta = 2))
  expect_named(td, c("lower", "upper"))
  expect_equal(unname(td[["lower"]]), 2^(-1 / 2), tolerance = 1e-12)
})

test_that("the two exceedance probabilities differ and both lie in the unit interval", {
  cop <- copula("Gumbel", theta = 2)
  a <- copula_exceedance(cop, 0.9, 0.9, type = "and")
  o <- copula_exceedance(cop, 0.9, 0.9, type = "or")
  expect_true(a >= 0 && a <= 1 && o >= 0 && o <= 1)
  expect_false(isTRUE(all.equal(a, o)))
})

test_that("copula_bounds reports the theta range", {
  b <- copula_bounds(copula("Clayton", theta = 2))
  expect_named(b, c("minimum", "maximum"))
})

test_that("a seeded copula draw is reproducible and has the right shape", {
  cop <- copula("Clayton", theta = 2,
                margin_x = distribution("Normal", c(0, 1)),
                margin_y = distribution("Normal", c(0, 1)))
  d <- copula_random(cop, 5, seed = 12345)
  expect_equal(dim(d), c(5L, 2L))
  expect_identical(d, copula_random(cop, 5, seed = 12345))
})

test_that("copula_fit returns a usable copula", {
  cop <- copula_fit("Clayton", x, y, method = "mpl")
  expect_s3_class(cop, "corehydro_copula")
  expect_true(is.finite(cop$theta))
  expect_true(is.finite(copula_pdf(cop, 0.3, 0.7)))
})

test_that("the mpl fit takes raw data and ranks it internally", {
  # The pseudo-likelihood is a function of the ranks alone, so fitting the raw sample and
  # fitting its plotting positions must land on the same theta -- that identity is what the
  # in-core transform buys, and it needs no oracle value.
  raw <- copula_fit("Clayton", x, y, method = "mpl")$theta
  pp <- copula_fit("Clayton", rank(x) / (length(x) + 1), rank(y) / (length(y) + 1),
                   method = "mpl")$theta
  expect_equal(raw, pp)
  # A copula that ignored the ranking would sit at an arbitrary interior point of the theta
  # bounds because every log_pdf evaluation lands off the unit square. A real Clayton fit to
  # this strongly concordant sample is well above the weak-dependence range.
  expect_true(raw > 1)
  expect_true(is.finite(copula_log_likelihood(copula("Clayton", theta = raw), x, y)))
})

test_that("a named marginal is refused for the methods that never fit one", {
  expect_error(copula_fit("Clayton", x, y, method = "mpl", margin_x = "Normal"), "does not use")
  expect_error(copula_fit("Clayton", x, y, method = "tau", margin_y = "Normal"), "does not use")
  # A parameterized corehydro_dist stays allowed: it is attached as given, not fitted.
  cop <- copula_fit("Clayton", x, y, method = "mpl",
                    margin_x = distribution("Normal", c(110, 20)),
                    margin_y = distribution("Normal", c(1.4, 0.3)))
  expect_equal(unname(dist_params(cop$margin_x)), c(110, 20))
  expect_equal(dim(copula_random(cop, 3, seed = 1)), c(3L, 2L))
})

test_that("the tau method is refused where upstream has no SetThetaFromTau", {
  expect_error(copula_fit("Joe", x, y, method = "tau"), "tau")
})

test_that("the three log-likelihoods run and differ", {
  cop <- copula("Clayton", theta = 2,
                margin_x = distribution("Normal", c(110, 20)),
                margin_y = distribution("Normal", c(1.4, 0.3)))
  ll_ifm <- copula_log_likelihood(cop, x, y, method = "ifm")
  ll_ps  <- copula_log_likelihood(cop, x, y, method = "pseudo")
  expect_true(is.finite(ll_ifm) && is.finite(ll_ps))
  expect_false(isTRUE(all.equal(ll_ifm, ll_ps)))
})

test_that("the density verbs evaluate a whole vector of pairs", {
  cop <- copula("Clayton", theta = 2)
  u <- c(0.3, 0.5, 0.8)
  v <- c(0.7, 0.9, 0.2)
  expect_equal(copula_pdf(cop, u, v),
               vapply(seq_along(u), function(i) copula_pdf(cop, u[i], v[i]), numeric(1)))
  expect_equal(copula_cdf(cop, u, v),
               vapply(seq_along(u), function(i) copula_cdf(cop, u[i], v[i]), numeric(1)))
  expect_equal(copula_log_pdf(cop, u, v), log(copula_pdf(cop, u, v)))
  # A scalar recycles against a vector, one value per pair.
  expect_equal(copula_pdf(cop, 0.3, v),
               vapply(v, function(vi) copula_pdf(cop, 0.3, vi), numeric(1)))
  expect_length(copula_pdf(cop, u, 0.5), 3L)
  expect_error(copula_pdf(cop, c(0.2, 0.4, 0.6), c(0.1, 0.9)), "recyclable")
  expect_error(copula_pdf(cop, numeric(0), 0.5), "non-empty")
})

test_that("copula_names lists the seven families", {
  expect_length(copula_names(), 7L)
  expect_true("StudentT" %in% copula_names())
})

test_that("a copula round-trips through save and load", {
  cop <- copula("Frank", theta = 3)
  f <- tempfile(); saveRDS(cop, f)
  expect_equal(copula_cdf(readRDS(f), 0.4, 0.6), copula_cdf(cop, 0.4, 0.6))
  unlink(f)
})
