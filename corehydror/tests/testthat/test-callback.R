test_that("root_find solves a user-written R function", {
  expect_equal(root_find(function(x) x^2 - 2, lower = 0, upper = 2), sqrt(2), tolerance = 1e-8)
  # The bracket is required to change sign, and the ported Brent says so.
  expect_error(root_find(function(x) x^2 + 1, lower = 0, upper = 2), "not bracketed")
})

test_that("derivative differentiates a user-written R function", {
  # f(x) = x^3, f'(2) = 12
  expect_equal(derivative(function(x) x^3, 2), 12, tolerance = 1e-6)
  # f(x) = sin(x), f'(pi / 3) = cos(pi / 3) = 0.5
  expect_equal(derivative(sin, pi / 3), 0.5, tolerance = 1e-6)
})

test_that("gradient and hessian differentiate a user-written R function", {
  rosenbrock <- function(p) (1 - p[1])^2 + 100 * (p[2] - p[1]^2)^2
  expect_equal(gradient(rosenbrock, c(1, 1)), c(0, 0), tolerance = 1e-6)
  h <- hessian(rosenbrock, c(1, 1))
  expect_equal(dim(h), c(2L, 2L))
  expect_equal(h[1, 2], h[2, 1], tolerance = 1e-6)

  # f(x, y) = x^2 + 2y^2 + xy has the constant Hessian [[2, 1], [1, 4]].
  quad <- function(p) p[1]^2 + 2 * p[2]^2 + p[1] * p[2]
  expect_equal(hessian(quad, c(1, 2)), matrix(c(2, 1, 1, 4), 2, 2), tolerance = 1e-3)
})

test_that("quadrature integrates a user-written R function", {
  # Closed forms only: the C#-pinned oracles live in fixtures/callback/math.json.
  q <- quadrature(function(x) x^2, lower = 0, upper = 3)
  expect_equal(as.numeric(q), 9, tolerance = 1e-10)
  expect_equal(attr(q, "status"), "Success")
  expect_gt(attr(q, "function_evaluations"), 0L)
  expect_equal(as.numeric(quadrature(sin, lower = 0, upper = pi)), 2, tolerance = 1e-8)
  expect_equal(as.numeric(quadrature(exp, lower = 0, upper = 1)), exp(1) - 1, tolerance = 1e-8)
})

test_that("quadrature reports the rule's own standard error", {
  # x^2 needs no subdivision (G10K21 is exact for it), so the accumulated error is nothing but the
  # rounding floor of the two rules' weighted sums -- NOT bit-exact zero. Whether the Gauss and
  # Kronrod sums round to the same double depends on whether the compiler contracts
  # `weight * fsum + acc` into an FMA: it does on arm64 (error exactly 0), it does not on a
  # baseline x86-64 build (error 1.7763568394002505e-15). expect_equal()'s default tolerance
  # covers both, which is why this passed on every platform while the Python twin's `== 0.0` did
  # not. The portably exact-zero case is x^3 on [0, 1], pinned in fixtures/callback/math.json.
  q <- quadrature(function(x) x^2, lower = 0, upper = 3)
  expect_lt(attr(q, "standard_error"), 1e-14)
  # A Lorentzian peak of half-width 0.01 does subdivide, so the error estimate is real. The
  # C#-pinned values for this exact run are fixtures/callback/math.json's
  # quadrature_peak_subdivides; here only the qualitative properties, to keep the oracle in the
  # fixture where it belongs.
  peak <- quadrature(function(x) 1 / (1 + 1e4 * x * x), lower = -1, upper = 1)
  expect_gt(attr(peak, "function_evaluations"), 21L)
  expect_gt(attr(peak, "standard_error"), 0)
  expect_lt(attr(peak, "standard_error"), 1e-6)
})

test_that("an unsupplied option leaves the ported routine's own default in force", {
  # The wrapper writes an option key only when the caller passes one, so NULL and the ported
  # default must give the identical run. Python's twin asserts the same.
  f <- function(x) 1 / (1 + 1e4 * x * x)
  default_run <- quadrature(f, lower = -1, upper = 1)
  explicit_run <- quadrature(f, lower = -1, upper = 1, absolute_tolerance = 1e-8,
                             relative_tolerance = 1e-8, max_function_evaluations = 10000000L)
  expect_identical(as.numeric(default_run), as.numeric(explicit_run))
  expect_identical(attr(default_run, "function_evaluations"),
                   attr(explicit_run, "function_evaluations"))
  expect_identical(root_find(function(x) x^2 - 2, lower = 0, upper = 2),
                   root_find(function(x) x^2 - 2, lower = 0, upper = 2, tolerance = 1e-8,
                             max_iterations = 1000L))
  expect_identical(derivative(function(x) x^3, 2), derivative(function(x) x^3, 2, step_size = -1))
})

test_that("quadrature reports the evaluation cap in its status instead of raising", {
  # 1 / sqrt(x) is unbounded at 0, so the rule never converges and the cap stops it.
  q <- quadrature(function(x) if (x <= 0) 0 else 1 / sqrt(x),
                  lower = 0, upper = 1, max_function_evaluations = 1000L)
  expect_equal(attr(q, "status"), "MaximumFunctionEvaluationsReached")
  expect_gte(attr(q, "function_evaluations"), 1000L)
})

test_that("an error raised inside the callback reaches the caller unchanged", {
  # Every method, not just one: the guard's sentinel is a value each ported routine can itself
  # reject, so an unwrapped drive site would report the internal error instead of this one.
  boom <- function(x) stop("my own error")
  expect_error(root_find(boom, lower = 0, upper = 2), "my own error")
  expect_error(derivative(boom, 2), "my own error")
  expect_error(gradient(boom, c(1, 1)), "my own error")
  expect_error(hessian(boom, c(1, 1)), "my own error")
  # Quadrature is the arm where the TRAILING rethrow carries the weight: the ported integrator
  # neither throws nor converges on the guard's NaN sentinel, it just runs to the evaluation cap
  # and reports a NaN result, which without the rethrow would look like an answer.
  expect_error(quadrature(boom, lower = 0, upper = 3, max_function_evaluations = 5000L),
               "my own error")
})

test_that("a callback returning a non-scalar is rejected", {
  expect_error(root_find(function(x) c(1, 2), lower = 0, upper = 2), "single number")
  expect_error(derivative(function(x) c(1, 2), 2), "single number")
  expect_error(gradient(function(p) c(1, 2), c(1, 1)), "single number")
  expect_error(quadrature(function(x) c(1, 2), lower = 0, upper = 1), "single number")
})

test_that("the R-side argument checks fire before the core is reached", {
  expect_error(root_find("not a function", lower = 0, upper = 2))
  expect_error(root_find(function(x) x, lower = c(0, 1), upper = 2))
  expect_error(gradient(function(p) sum(p), "not numeric"))
  expect_error(quadrature(function(x) x, lower = 1, upper = 0), "must be below")
  expect_error(quadrature(function(x) x, lower = 0, upper = 1, absolute_tolerance = 0),
               "between 1e-15 and 1")
  expect_error(quadrature(function(x) x, lower = 0, upper = 1, relative_tolerance = 2),
               "between 1e-15 and 1")
})

test_that("the rng handle draws from the core seeded stream and invalidates after the call", {
  # The handle cannot be constructed by a user -- it is only ever handed to a callback -- so it is
  # tested through rng_probe(), the internal verb that seeds a generator, hands a handle to `f`,
  # and returns what `f` drew. The C#-pinned values for these runs live in
  # fixtures/callback/rng_handle.json; here only the properties, and the lifetime rule.
  ns <- asNamespace("corehydror")
  seen <- NULL
  captured <- NULL
  out <- ns$rng_probe(seed = 12345, parameters = 3, f = function(parameters, rng) {
    captured <<- rng
    seen <<- rng_uniform(rng, parameters[1])
    seen
  })
  expect_length(seen, 3)
  expect_true(all(seen > 0 & seen < 1))
  expect_equal(out, seen)
  expect_s3_class(captured, "corehydro_rng")
  # The handle borrows the generator; using it after the call must error, not read freed memory.
  expect_error(rng_uniform(captured, 1), "no longer valid")
  expect_error(rng_integers(captured, 1, 0, 10), "no longer valid")
})

test_that("the same seed gives the same draws and a different seed does not", {
  ns <- asNamespace("corehydror")
  draw <- function(seed) {
    ns$rng_probe(seed, 5, function(parameters, rng) rng_uniform(rng, parameters[1]))
  }
  expect_identical(draw(12345), draw(12345))
  expect_false(isTRUE(all.equal(draw(12345), draw(54321))))
})

test_that("rng_integers draws whole numbers on [min, max) off the same stream", {
  ns <- asNamespace("corehydror")
  k <- ns$rng_probe(999, c(50, 3, 7), function(parameters, rng) {
    rng_integers(rng, parameters[1], parameters[2], parameters[3])
  })
  expect_length(k, 50)
  expect_true(all(k == floor(k)))
  expect_true(all(k >= 3 & k <= 6))  # the upper bound is exclusive, as in C# Next(min, max)
  # Uniforms and integers share one state, so interleaving them is not the same as taking them
  # separately -- the second uniform must continue after the integer, not repeat it.
  interleaved <- ns$rng_probe(2024, 0, function(parameters, rng) {
    c(rng_uniform(rng, 1), rng_integers(rng, 1, 0, 1000), rng_uniform(rng, 1))
  })
  straight <- ns$rng_probe(2024, 0, function(parameters, rng) rng_uniform(rng, 2))
  expect_identical(interleaved[1], straight[1])
  expect_false(isTRUE(all.equal(interleaved[3], straight[2])))
})

test_that("a handle stored from an earlier call is dead inside a later one", {
  # The misuse a garbage-collected host makes easy: keep the SEXP, call again, reach for the old
  # one. The second call's handle is live; the first call's is not, and says so.
  ns <- asNamespace("corehydror")
  first <- NULL
  ns$rng_probe(1, 1, function(parameters, rng) {
    first <<- rng
    rng_uniform(rng, 1)
  })
  second_ok <- ns$rng_probe(2, 1, function(parameters, rng) {
    expect_error(rng_uniform(first, 1), "no longer valid")
    rng_uniform(rng, 1)
  })
  expect_length(second_ok, 1)
  # A handle captured in a closure built inside the callback is the same story: the closure
  # outlives the call, the borrow does not.
  later <- NULL
  ns$rng_probe(3, 1, function(parameters, rng) {
    later <<- function() rng_uniform(rng, 1)
    rng_uniform(rng, 1)
  })
  expect_error(later(), "no longer valid")
})

test_that("a nested call gets its own handle and leaves the outer one alone", {
  # Each call scopes its OWN borrow, so an inner call cannot invalidate the outer handle and
  # cannot disturb the outer generator. Tasks 5 and 6 need this: a proposal that runs a nested
  # seeded verb must come back to a live handle and an undisturbed stream.
  ns <- asNamespace("corehydror")
  out <- ns$rng_probe(5, 1, function(parameters, outer) {
    a <- rng_uniform(outer, 1)
    inner <- ns$rng_probe(6, 1, function(p2, rng2) rng_uniform(rng2, 1))
    c(a, inner, rng_uniform(outer, 1))
  })
  straight <- ns$rng_probe(5, 1, function(parameters, rng) rng_uniform(rng, 2))
  expect_identical(out[c(1L, 3L)], straight)
})

test_that("the rng handle rejects nonsense arguments and non-handles", {
  ns <- asNamespace("corehydror")
  expect_error(
    ns$rng_probe(1, 1, function(parameters, rng) rng_uniform(rng, 0)),
    "positive whole number"
  )
  expect_error(
    ns$rng_probe(1, 1, function(parameters, rng) rng_integers(rng, 2, 5, 5)),
    "must be below"
  )
  # Anything that is not a handle is refused by name rather than crashing on a bad pointer.
  expect_error(rng_uniform(42, 1), "random number generator handle")
  expect_error(rng_integers("nope", 1, 0, 2), "random number generator handle")
})

test_that("an external pointer from somewhere else is refused instead of crashing the session", {
  # THE case that matters, and the one `42` and `"nope"` above do NOT cover: they are refused by
  # the type check and never reach the cast. Every package that hands R a pointer produces the
  # same SEXP type, so an EXTPTRSXP-only check would cast a foreign pointer to the borrow type and
  # dereference it -- a wild read that segfaulted and aborted R when it was tried. The pointer
  # below is a genuinely foreign one built from base R alone (the address slot of a registered
  # native routine, whose first word is a live function address, not a null). It must produce an R
  # error.
  foreign <- getDLLRegisteredRoutines(getLoadedDLLs()[["corehydror"]])$.Call[[1]]$address
  expect_identical(typeof(foreign), "externalptr")
  expect_error(rng_uniform(foreign, 1), "random number generator handle")
  expect_error(rng_integers(foreign, 1, 0, 2), "random number generator handle")
  # A bare external pointer is refused too. Note WHICH check refuses it: `new("externalptr")` has
  # no tag, so it never reaches the null-address check -- the tag check turns it away first, and
  # the message is the same "not a handle" one.
  null_ptr <- methods::new("externalptr")
  expect_error(rng_uniform(null_ptr, 1), "random number generator handle")
  # A forged class attribute buys nothing: the class is for print/inherits, never for dispatch.
  fake <- structure(list(), class = "corehydro_rng")
  expect_error(rng_uniform(fake, 1), "random number generator handle")
})

test_that("a serialized handle comes back tagged but null, and says it is expired", {
  # THE null-address branch, which nothing else reaches: R serializes an external pointer by
  # writing its tag and dropping its address, so a round-tripped handle is the one object that
  # carries our tag AND a null address. It is a realistic way to get one, too -- saveRDS() on an
  # environment that captured a handle, or a handle sent to a parallel worker. The address check
  # behind the tag check is what stands between that and a cast of NULL, and the message it gives
  # is the expired-handle sentence rather than the not-a-handle one, because the object really was
  # ours.
  ns <- asNamespace("corehydror")
  handle <- NULL
  ns$rng_probe(1, 1, function(parameters, rng) {
    handle <<- rng
    rng_uniform(rng, 1)
  })
  round_trip <- unserialize(serialize(handle, NULL))
  expect_identical(typeof(round_trip), "externalptr")
  expect_error(rng_uniform(round_trip, 1), "no longer valid")
  expect_error(rng_integers(round_trip, 1, 0, 2), "no longer valid")
})

test_that("a range wider than an int32 span is an error, not an overflowed draw", {
  # rng_integers hands `max - min` to the ported Next(int). C# computes that subtraction unchecked
  # and throws once it wraps negative; the same expression in C++ is undefined behaviour, and
  # rng_integers(rng, 1, -2e9, 2e9) used to return an arbitrary out-of-range integer
  # (-208904155) in both languages. The Python twin asserts the same boundary.
  ns <- asNamespace("corehydror")
  int_max <- 2147483647
  at_boundary <- ns$rng_probe(11, 1, function(parameters, rng) {
    rng_integers(rng, 1, 0, int_max)  # span == int32 max, the widest allowed
  })
  expect_length(at_boundary, 1)
  expect_true(at_boundary >= 0 && at_boundary < int_max)
  expect_error(
    ns$rng_probe(11, 1, function(parameters, rng) rng_integers(rng, 1, -1, int_max)),
    "too wide"
  )
  expect_error(
    ns$rng_probe(11, 1, function(parameters, rng) rng_integers(rng, 1, -2e9, 2e9)),
    "too wide"
  )
})

test_that("a fractional count is refused rather than truncated, as Python refuses it", {
  # R would happily make 2.7 into 2 while Python raised, so the same call meant different things
  # in the two packages. Both now refuse it by the name of the argument.
  ns <- asNamespace("corehydror")
  expect_error(
    ns$rng_probe(1, 1, function(parameters, rng) rng_uniform(rng, 2.7)),
    "`n` must be a single positive whole number"
  )
  expect_error(
    ns$rng_probe(1, 1, function(parameters, rng) rng_integers(rng, 2.7, 0, 5)),
    "`n` must be a single positive whole number"
  )
  expect_error(
    ns$rng_probe(1, 1, function(parameters, rng) rng_integers(rng, 1, 0.5, 5)),
    "single finite whole number"
  )
  # A whole number spelled as a double is still a whole number -- in R every literal is one.
  drawn <- ns$rng_probe(1, 1, function(parameters, rng) rng_uniform(rng, 2.0))
  expect_length(drawn, 2)
})

test_that("the int-min bound is the same in both packages", {
  # rng_is_whole() tests `abs(x) <= .Machine$integer.max`, which is 2147483647, so R refuses
  # -2147483648 -- and Python used to accept it, making the same call legal in one package and an
  # error in the other. The Python twin asserts the same boundary from its side.
  ns <- asNamespace("corehydror")
  expect_error(
    ns$rng_probe(1, 1, function(parameters, rng) rng_integers(rng, 1, -2147483648, -2147483000)),
    "single finite whole number"
  )
  drawn <- ns$rng_probe(
    1, 1, function(parameters, rng) rng_integers(rng, 1, -2147483647, -2147483000)
  )
  expect_true(drawn >= -2147483647 && drawn < -2147483000)
})

test_that("a handle leaked out of a callback that then raises is dead", {
  # The unwind path, which only the C++ suite covered. A destructor runs whether the call returns
  # or throws, so the scope invalidates the borrow either way -- and a user who stashed the handle
  # before their own error meets a message rather than freed memory. The Python twin is identical.
  ns <- asNamespace("corehydror")
  leaked <- NULL
  expect_error(
    ns$rng_probe(1, 1, function(parameters, rng) {
      leaked <<- rng
      rng_uniform(rng, 1)
      stop("my own error, raised half way through")
    }),
    "my own error, raised half way through"
  )
  expect_false(is.null(leaked))
  expect_error(rng_uniform(leaked, 1), "no longer valid")
  expect_error(rng_integers(leaked, 1, 0, 2), "no longer valid")
})

test_that("an error raised inside an rng callback reaches the caller unchanged", {
  ns <- asNamespace("corehydror")
  expect_error(
    ns$rng_probe(1, 1, function(parameters, rng) stop("my own error")),
    "my own error"
  )
})

test_that("a non-finite point is rejected by name, the same way Python rejects it", {
  # Without the finite check this failed three layers down in the spec serializer with
  # "spec values must be finite", naming neither `x` nor the verb. The Python twin
  # (corehydropy/tests/test_callback.py) asserts the identical message.
  f <- function(p) sum(p^2)
  expect_error(gradient(f, c(1, Inf)), "`x` must be a non-empty numeric vector of finite values")
  expect_error(hessian(f, c(1, -Inf)), "`x` must be a non-empty numeric vector of finite values")
  expect_error(gradient(f, c(1, NA_real_)), "`x` must be a non-empty numeric vector of finite values")
  expect_error(gradient(f, c(1, NaN)), "`x` must be a non-empty numeric vector of finite values")
  expect_error(gradient(f, numeric(0)), "`x` must be a non-empty numeric vector of finite values")
  # derivative()'s scalar point already had the finite check; it stays.
  expect_error(derivative(function(x) x^3, Inf), "single finite number")
})

# --- mcmc_posterior ------------------------------------------------------------------------

# The Gaussian kernel the fixture catalog uses, written the same way: arithmetic only, summed in
# an explicit loop rather than through sum(), which accumulates in extended precision in R alone.
mcmc_gaussian_kernel <- function(p) {
  data <- c(4.9, 5.1, 5.0, 5.2, 4.8)
  acc <- 0
  for (x in data) acc <- acc + (x - p[1]) * (x - p[1])
  -0.5 * acc
}

# A model whose full conditional really is uniform, so the proposal below is an EXACT Gibbs step
# (draw from the conditional, accept unconditionally) rather than a random walk in Gibbs clothing:
# with x_i ~ Uniform(mu - 1, mu + 1) and a flat prior, mu given the data is
# Uniform(max(x) - 1, min(x) + 1). For this data that is Uniform(4.2, 5.8), mean exactly 5 and sd
# 1.6 / sqrt(12) = 0.4619. Arithmetic only, matching the fixture catalog.
mcmc_uniform_width_data <- c(4.9, 5.1, 5.0, 5.2, 4.8)

mcmc_uniform_width_kernel <- function(p) {
  for (x in mcmc_uniform_width_data) {
    if (x - p[1] > 1 || p[1] - x > 1) return(-Inf)
  }
  0
}

mcmc_uniform_conditional <- function(parameters, rng) {
  lo <- max(mcmc_uniform_width_data) - 1
  hi <- min(mcmc_uniform_width_data) + 1
  lo + rng_uniform(rng, 1) * (hi - lo)
}

test_that("mcmc_posterior recovers the mean of a user-written Gaussian posterior", {
  fit <- mcmc_posterior(
    mcmc_gaussian_kernel, list(distribution("Uniform", c(0, 10))),
    iterations = 300, warmup = 100, chains = 2, thinning = 1, seed = 12345,
    initialize = "Randomize"
  )
  expect_equal(fit$posterior_mean[[1]], 5.0, tolerance = 0.2)
  expect_equal(fit$map[[1]], 5.0, tolerance = 0.2)
  # The full mcmc_sample() shape, so a user can move between the two functions.
  expect_named(fit, names(mcmc_sample(c(4.9, 5.1, 5.0, 5.2, 4.8), "Normal", iterations = 200)))
  expect_equal(fit$parameters, "p1")
  expect_length(fit$chains, 2L)
  expect_equal(dim(fit$chains[[1]]), c(300L, 1L))
  expect_length(fit$acceptance_rates, 2L)
})

test_that("two seeded mcmc_posterior runs are identical", {
  args <- list(
    mcmc_gaussian_kernel, list(distribution("Uniform", c(0, 10))),
    iterations = 300, warmup = 100, chains = 2, thinning = 1, seed = 12345,
    initialize = "Randomize"
  )
  expect_identical(do.call(mcmc_posterior, args), do.call(mcmc_posterior, args))
  # A different seed is a different chain, so the equality above is not vacuous.
  other <- do.call(mcmc_posterior, utils::modifyList(args, list(seed = 999)))
  expect_false(identical(other$chains, do.call(mcmc_posterior, args)$chains))
})

test_that("a two-parameter model reads its priors in order", {
  # The prior list's length IS the parameter count, and the order is the order the log-likelihood
  # indexes. A straight line through eight points, intercept then slope.
  ll <- function(p) {
    t <- c(1, 2, 3, 4, 5, 6, 7, 8)
    y <- c(2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1)
    acc <- 0
    for (i in seq_along(t)) {
      residual <- y[i] - p[1] - p[2] * t[i]
      acc <- acc + residual * residual
    }
    -0.5 * acc
  }
  fit <- mcmc_posterior(
    ll, list(distribution("Uniform", c(-5, 5)), distribution("Uniform", c(0, 5))),
    sampler = "DEMCz", iterations = 300, warmup = 100, chains = 3, thinning = 1, seed = 12345,
    initialize = "Randomize"
  )
  expect_equal(fit$parameters, c("p1", "p2"))
  expect_equal(fit$map[[1]], 0.1, tolerance = 0.5)
  expect_equal(fit$map[[2]], 2.0, tolerance = 0.2)
})

test_that("an error raised inside the log-likelihood reaches the caller, for every sampler", {
  # On BOTH initialization paths, and they fail differently underneath: "Randomize" never throws
  # of its own accord (-infinity is a legal, always rejected fitness, so the chain runs to
  # completion), while "MAP" hands DifferentialEvolution nothing but -infinity and can throw from
  # its own internals first. Only the guard makes the user's own message win in both.
  #
  # This loops ALL EIGHT samplers `mcmc_posterior()`'s own `sampler` argument accepts: a guard
  # wired for one sampler's arm and silently missing on another is exactly the shape of bug a
  # previous phase shipped, and a single-sampler test cannot catch it.
  samplers <- list(
    list(sampler = "RWMH", chains = 2L, iterations = 100L, warmup = 50L),
    list(sampler = "ARWMH", chains = 2L, iterations = 100L, warmup = 50L),
    list(sampler = "DEMCz", chains = 3L, iterations = 100L, warmup = 50L),
    list(sampler = "DEMCzs", chains = 3L, iterations = 100L, warmup = 50L),
    list(sampler = "HMC", chains = 2L, iterations = 100L, warmup = 50L),
    list(sampler = "NUTS", chains = 2L, iterations = 100L, warmup = 50L),
    # SNIS is not a Markov chain: its own ported validation rejects any warm-up and requires
    # iterations >= output_length, which mcmc_posterior() does not expose (fixed at the ported
    # default, 10000) -- so this is the smallest legal call, not an arbitrary round number.
    list(sampler = "SNIS", chains = 1L, iterations = 10000L, warmup = NULL),
    # Gibbs runs one chain by construction and needs a working proposal before the log-likelihood
    # is ever reached, so it is the arm where two callbacks are live at once.
    list(sampler = "Gibbs", chains = NULL, iterations = 100L, warmup = 50L,
         proposal = mcmc_uniform_conditional)
  )
  for (s in samplers) {
    for (init in c("Randomize", "MAP")) {
      expect_error(
        mcmc_posterior(
          function(p) stop("my own error"), list(distribution("Uniform", c(0, 10))),
          sampler = s$sampler, proposal = s$proposal, iterations = s$iterations,
          warmup = s$warmup, chains = s$chains, thinning = 1, seed = 12345, initialize = init
        ),
        "my own error"
      )
    }
  }
})

test_that("mcmc_posterior refuses a prior list that is not one", {
  ll <- mcmc_gaussian_kernel
  expect_error(mcmc_posterior(ll, list()), "non-empty list of distribution\\(\\) objects")
  expect_error(mcmc_posterior(ll, "Uniform"), "non-empty list of distribution\\(\\) objects")
  expect_error(mcmc_posterior(ll, list(distribution("Uniform", c(0, 10)), 3)),
               "non-empty list of distribution\\(\\) objects")
  expect_error(mcmc_posterior("not a function", list(distribution("Uniform", c(0, 10)))),
               "`log_likelihood` must be a function")
  expect_error(
    mcmc_posterior(ll, list(distribution("Uniform", c(0, 10))), sampler = "Nope"),
    "unknown sampler 'Nope'; expected one of"
  )
  # check_choice(), not match.arg(): a PREFIX is refused too, because Python refuses it and an
  # argument that is legal in one package and an error in the other is the defect this project
  # least tolerates.
  expect_error(
    mcmc_posterior(ll, list(distribution("Uniform", c(0, 10))), sampler = "Gib"),
    "unknown sampler 'Gib'; expected one of"
  )
  # Gibbs is a legal sampler now, but not without the one thing it cannot default.
  expect_error(
    mcmc_posterior(ll, list(distribution("Uniform", c(0, 10))), sampler = "Gibbs"),
    "requires a `proposal` function"
  )
  # A NULL seed used to reach as.integer(NULL) and fail deep inside the JSON builder with an
  # unhelpful message; it is now refused by name before that point.
  expect_error(
    mcmc_posterior(ll, list(distribution("Uniform", c(0, 10))), seed = NULL),
    "`seed` must not be NULL"
  )
  # A single distribution is accepted for a one-parameter model.
  fit <- mcmc_posterior(ll, distribution("Uniform", c(0, 10)), iterations = 100, warmup = 50,
                        chains = 2, thinning = 1, seed = 12345, initialize = "Randomize")
  expect_equal(fit$parameters, "p1")
})

test_that("a log-likelihood indexing past its priors fails by name", {
  # The likeliest user error on this surface: three parameters written, one prior given. R hands
  # back NA rather than raising, so the message has to come from the glue's own return check -- Python
  # raises IndexError of its own accord, and this is what keeps the two packages in step.
  expect_error(
    mcmc_posterior(
      function(p) -0.5 * (p[1] * p[1] + p[2] * p[2]), list(distribution("Uniform", c(0, 10))),
      iterations = 100, warmup = 50, chains = 2, thinning = 1, seed = 12345,
      initialize = "Randomize"
    ),
    "returned NA or NaN rather than a number"
  )
})

test_that("output_length is exposed and reproduces the cross-language fixture", {
  # The fixture fixtures/callback/callback_cross_language.json runs this construct with
  # output_length 100 and asserts its posterior mean and median at ZERO tolerance. Until the
  # option was exposed, those two numbers could not be reproduced through the public verb at all:
  # the ported sampler summarizes the OUTPUT BLOCK, not the recorded chain, and the block's
  # default length is 10,000. The same call in Python returns the same doubles.
  fit <- mcmc_posterior(
    mcmc_uniform_width_kernel, list(distribution("Uniform", c(0, 10))),
    sampler = "Gibbs", proposal = mcmc_uniform_conditional,
    iterations = 300, warmup = 100, thinning = 1, output_length = 100,
    seed = 12345, initialize = "Randomize"
  )
  expect_identical(fit$posterior_mean[[1]], 4.984069999799132)
  expect_identical(fit$posterior_median[[1]], 4.943827163241803)
  # The recorded chain is untouched by it: same length, same states, same MAP.
  expect_identical(dim(fit$chains[[1]]), c(300L, 1L))
  expect_identical(fit$map[[1]], 5.7310633707791565)

  # The ported floor is refused by name in both verbs rather than by the ported sentence.
  expect_error(
    mcmc_posterior(mcmc_gaussian_kernel, list(distribution("Uniform", c(0, 10))),
                   output_length = 99),
    "at least 100"
  )
  expect_error(mcmc_sample(c(4.9, 5.1, 5.0), "Normal", output_length = 99), "at least 100")
})

test_that("SNIS is refused the settings its own ported validation refuses", {
  ll <- mcmc_gaussian_kernel
  expect_error(
    mcmc_posterior(ll, list(distribution("Uniform", c(0, 10))), sampler = "SNIS", chains = 4),
    "supports `chains = 1` only"
  )
  # And it runs when left alone: no auto-derived warm-up, because its ValidateSettings rejects any.
  # Its other ported rule is `Iterations >= OutputLength`, and OutputLength is not a setting this
  # surface exposes, so 10000 is the smallest iteration count SNIS accepts here.
  fit <- mcmc_posterior(ll, list(distribution("Uniform", c(0, 10))), sampler = "SNIS",
                        iterations = 10000, seed = 12345, initialize = "Randomize")
  expect_equal(fit$posterior_mean[[1]], 5.0, tolerance = 0.5)
})

# --- the Gibbs proposal and the HMC/NUTS gradient --------------------------------------------

test_that("Gibbs runs with a user-written proposal", {
  fit <- mcmc_posterior(
    mcmc_uniform_width_kernel, priors = list(distribution("Uniform", c(0, 10))),
    sampler = "Gibbs", proposal = mcmc_uniform_conditional,
    iterations = 500, warmup = 100, thinning = 1, seed = 12345, initialize = "Randomize"
  )
  expect_equal(fit$posterior_mean[[1]], 5.0, tolerance = 0.5)
  # The conditional's own moments, which say the draws came from the proposal and not from some
  # fallback: Uniform(4.2, 5.8) has sd 0.4619, and no state can fall outside its support.
  expect_equal(fit$posterior_sd[[1]], 0.4619, tolerance = 0.05)
  expect_true(all(fit$chains[[1]] >= 4.2 & fit$chains[[1]] <= 5.8))
  # The ported constructor forces one chain; Gelman-Rubin needs two, so rhat is NaN here.
  expect_length(fit$chains, 1L)
  expect_true(is.nan(fit$rhat[[1]]))
})

test_that("two seeded Gibbs runs are identical, and the proposal draws off the core stream", {
  run <- function(seed) {
    mcmc_posterior(
      mcmc_uniform_width_kernel, distribution("Uniform", c(0, 10)),
      sampler = "Gibbs", proposal = mcmc_uniform_conditional,
      iterations = 300, warmup = 100, thinning = 1, seed = seed, initialize = "Randomize"
    )$chains[[1]]
  }
  expect_identical(run(12345), run(12345))
  # A different seed is a different chain, which is what says the proposal is drawing off the
  # generator the run seeded rather than off R's own.
  expect_false(identical(run(12345), run(999)))
})

test_that("a proposal returning the wrong number of values is refused by name", {
  expect_error(
    mcmc_posterior(
      mcmc_uniform_width_kernel, distribution("Uniform", c(0, 10)),
      sampler = "Gibbs", proposal = function(parameters, rng) rng_uniform(rng, 2),
      iterations = 100, warmup = 50, thinning = 1, seed = 12345, initialize = "Randomize"
    ),
    "must return one value per parameter"
  )
})

test_that("a proposal returning NaN is refused", {
  # Review fix (Task 5, finding 1): as_vector_vector_fn (this file) has always rejected NA/NaN by
  # name; corehydropy's twin (as_vector_vector_fn in corehydropy/src/bindings/callback.cpp) had no
  # such check for the vector-returning converter it shares between the gradient and the Gibbs
  # proposal, so the same mistake was a clear error here and a silently garbage chain in Python.
  # This test asserts the R side of that symmetric refusal stays in place; +/-Inf is deliberately
  # NOT refused (see the gradient test below) -- only NA/NaN is.
  expect_error(
    mcmc_posterior(
      mcmc_uniform_width_kernel, distribution("Uniform", c(0, 10)),
      sampler = "Gibbs", proposal = function(parameters, rng) NaN,
      iterations = 100, warmup = 50, thinning = 1, seed = 12345, initialize = "Randomize"
    ),
    "returned NA or NaN rather than a number"
  )
})

test_that("an error raised inside the proposal reaches the caller, on both init paths", {
  # The proposal's OWN guard, which no log-likelihood test can prove: it runs beside the
  # log-likelihood on a SHARED abort state, so the first throw -- whichever callback it comes
  # from -- short-circuits both and is the message that surfaces.
  for (init in c("Randomize", "MAP")) {
    expect_error(
      mcmc_posterior(
        mcmc_uniform_width_kernel, distribution("Uniform", c(0, 10)),
        sampler = "Gibbs", proposal = function(parameters, rng) stop("my own proposal error"),
        iterations = 100, warmup = 50, thinning = 1, seed = 12345, initialize = init
      ),
      "my own proposal error"
    )
  }
  # And a handle leaked out of a proposal is dead afterwards, exactly as one leaked out of
  # rng_probe() is.
  leaked <- NULL
  mcmc_posterior(
    mcmc_uniform_width_kernel, distribution("Uniform", c(0, 10)),
    sampler = "Gibbs",
    proposal = function(parameters, rng) {
      leaked <<- rng
      mcmc_uniform_conditional(parameters, rng)
    },
    iterations = 100, warmup = 50, thinning = 1, seed = 12345, initialize = "Randomize"
  )
  expect_error(rng_uniform(leaked, 1), "no longer valid")
})

test_that("HMC and NUTS accept an analytic gradient and still default without one", {
  # The smooth Gaussian kernel again, with its analytic gradient d/dmu = sum(x - mu).
  grad_calls <- 0L
  grad <- function(p) {
    grad_calls <<- grad_calls + 1L
    acc <- 0
    for (x in c(4.9, 5.1, 5.0, 5.2, 4.8)) acc <- acc + (x - p[1])
    acc
  }
  for (sampler in c("HMC", "NUTS")) {
    grad_calls <- 0L
    args <- list(
      mcmc_gaussian_kernel, distribution("Uniform", c(0, 10)),
      sampler = sampler, iterations = 200, warmup = 50, chains = 2, thinning = 1,
      seed = 12345, initialize = "Randomize"
    )
    with_gradient <- do.call(mcmc_posterior, c(args, list(gradient = grad)))
    expect_gt(grad_calls, 0L)  # the user's gradient really was the one used
    expect_equal(with_gradient$posterior_mean[[1]], 5.0, tolerance = 0.3)
    # No gradient leaves the ported bound-aware finite-difference default in force.
    without <- do.call(mcmc_posterior, args)
    expect_equal(without$posterior_mean[[1]], 5.0, tolerance = 0.3)
    # A DELIBERATELY WRONG gradient is what proves the supplied function drives the leapfrog
    # rather than being quietly ignored. Comparing the analytic run with the default one would
    # not: the central difference of a quadratic log-density is exact up to rounding, so those
    # two agree to about 4e-16 (fixtures/callback/mcmc.json's two HMC cases say the same thing
    # with real numbers). Half the true gradient rather than zero: a zero gradient leaves the
    # momentum unturned, so NUTS never detects a U-turn and builds its full 2^10 tree every
    # iteration -- correct behaviour, but a million callbacks for one assertion.
    wrong <- do.call(mcmc_posterior, c(args, list(gradient = function(p) 0.5 * grad(p))))
    expect_false(identical(wrong$chains, without$chains))
  }
})

test_that("a gradient returning the wrong number of values, or raising, is refused by name", {
  expect_error(
    mcmc_posterior(
      mcmc_gaussian_kernel, distribution("Uniform", c(0, 10)), sampler = "HMC",
      gradient = function(p) c(1, 2), iterations = 100, warmup = 50, chains = 2,
      thinning = 1, seed = 12345, initialize = "Randomize"
    ),
    "must return one value per parameter"
  )
  # The gradient's own guard, on both init paths and both samplers that take one.
  for (sampler in c("HMC", "NUTS")) {
    for (init in c("Randomize", "MAP")) {
      expect_error(
        mcmc_posterior(
          mcmc_gaussian_kernel, distribution("Uniform", c(0, 10)), sampler = sampler,
          gradient = function(p) stop("my own gradient error"), iterations = 100, warmup = 50,
          chains = 2, thinning = 1, seed = 12345, initialize = init
        ),
        "my own gradient error"
      )
    }
  }
})

test_that("a gradient returning NaN is refused, for both samplers that take one", {
  # Review fix (Task 5, finding 1): this file's as_vector_vector_fn has always rejected NA/NaN by
  # name; corehydropy's twin had no such check until this review round, so a gradient returning
  # nan was a clear error here and a silently garbage chain in Python. +/-Inf is deliberately NOT
  # refused -- only NA/NaN is; see the proposal test above for the same distinction.
  for (sampler in c("HMC", "NUTS")) {
    expect_error(
      mcmc_posterior(
        mcmc_gaussian_kernel, distribution("Uniform", c(0, 10)), sampler = sampler,
        gradient = function(p) NaN, iterations = 100, warmup = 50, chains = 2,
        thinning = 1, seed = 12345, initialize = "Randomize"
      ),
      "returned NA or NaN rather than a number"
    )
  }
})

test_that("each delegate is refused by the samplers that have no use for it", {
  expect_error(
    mcmc_posterior(mcmc_gaussian_kernel, distribution("Uniform", c(0, 10)),
                   proposal = mcmc_uniform_conditional),
    "only used by the Gibbs sampler"
  )
  expect_error(
    mcmc_posterior(mcmc_gaussian_kernel, distribution("Uniform", c(0, 10)),
                   gradient = function(p) 1),
    "only used by the HMC and NUTS samplers"
  )
  expect_error(
    mcmc_posterior(mcmc_gaussian_kernel, distribution("Uniform", c(0, 10)),
                   sampler = "Gibbs", proposal = "not a function"),
    "`proposal` must be a function"
  )
  expect_error(
    mcmc_posterior(mcmc_gaussian_kernel, distribution("Uniform", c(0, 10)),
                   sampler = "HMC", gradient = "not a function"),
    "`gradient` must be a function"
  )
})

# --- bootstrap_custom, the four bootstrap delegates ------------------------------------------
#
# The model throughout is the plainest one there is: an iid resample of a fixed sample, fitted by
# its mean. Written with `+ - * /` and an explicit loop rather than mean(), which accumulates in
# extended precision, so the identical call in Python resamples AND computes the identical numbers.
# The C#-pinned oracle for this same model lives in fixtures/callback/bootstrap.json.
boot_data <- c(4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7)

boot_resample <- function(data, parameters, rng) {
  # rng_integers draws on [0, n), counting from 0 as the ported delegate does, so the index is
  # shifted by one for R's own 1-based subscript.
  data[rng_integers(rng, length(data), 0, length(data)) + 1L]
}

boot_fit <- function(data) {
  acc <- 0
  for (x in data) acc <- acc + x
  acc / length(data)
}

boot_statistic <- function(parameters) parameters

boot_jackknife <- function(data, index) data[-(index + 1)]

boot_sample_mean <- function() {
  acc <- 0
  for (x in boot_data) acc <- acc + x
  acc / length(boot_data)
}

test_that("bootstrap_custom brackets the sample mean", {
  res <- bootstrap_custom(
    data = boot_data, resample = boot_resample, fit = boot_fit, statistic = boot_statistic,
    replicates = 500, seed = 12345
  )
  expect_true(res$lower[[1]] < boot_sample_mean() && boot_sample_mean() < res$upper[[1]])
  # The point estimate is the statistic of the ORIGINAL fit, not a bootstrap average.
  expect_equal(res$estimate[[1]], boot_sample_mean())
  expect_identical(res$failed_replicates, 0L)
  expect_identical(res$valid_count[[1]], 500L)
})

test_that("two seeded bootstrap_custom runs are identical, and the resample draws off the core stream", {
  run <- function(seed) {
    bootstrap_custom(
      data = boot_data, resample = boot_resample, fit = boot_fit, statistic = boot_statistic,
      replicates = 200, seed = seed
    )$lower[[1]]
  }
  expect_identical(run(12345), run(12345))
  # A different seed is a different interval, which is what says the resample is drawing off the
  # generator the run seeded rather than off R's own.
  expect_false(identical(run(12345), run(999)))
})

test_that("ci_method = BCa without a jackknife is refused, and before any resampling happens", {
  calls <- 0L
  counting_resample <- function(data, parameters, rng) {
    calls <<- calls + 1L
    boot_resample(data, parameters, rng)
  }
  expect_error(
    bootstrap_custom(
      data = boot_data, resample = counting_resample, fit = boot_fit, statistic = boot_statistic,
      replicates = 500, seed = 12345, ci_method = "BCa"
    ),
    "jackknife"
  )
  # The ported class only checks this inside GetConfidenceIntervals, i.e. after every replicate has
  # already called back into R. Zero resample calls is the proof that this one comes first.
  expect_identical(calls, 0L)
})

test_that("an error raised inside each of the four delegates reaches the caller", {
  # One test per delegate: a guard wired for one and missing on another is exactly the shape of bug
  # a four-callback surface invites, and a test that only makes `fit` throw cannot catch it.
  expect_error(
    bootstrap_custom(boot_data, function(data, parameters, rng) stop("my own resample error"),
                     boot_fit, boot_statistic, replicates = 20, seed = 12345),
    "my own resample error"
  )
  expect_error(
    bootstrap_custom(boot_data, boot_resample, function(data) stop("my own fit error"),
                     boot_statistic, replicates = 20, seed = 12345),
    "my own fit error"
  )
  expect_error(
    bootstrap_custom(boot_data, boot_resample, boot_fit,
                     function(parameters) stop("my own statistic error"),
                     replicates = 20, seed = 12345),
    "my own statistic error"
  )
  expect_error(
    bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic,
                     jackknife = function(data, index) stop("my own jackknife error"),
                     replicates = 20, seed = 12345, ci_method = "BCa"),
    "my own jackknife error"
  )
})

test_that("every confidence interval method runs and brackets the estimate", {
  for (method in c("Percentile", "BiasCorrected", "Normal", "BCa")) {
    res <- bootstrap_custom(
      data = boot_data, resample = boot_resample, fit = boot_fit, statistic = boot_statistic,
      jackknife = boot_jackknife, replicates = 200, seed = 12345, ci_method = method
    )
    expect_true(res$lower[[1]] < res$upper[[1]])
    expect_equal(res$estimate[[1]], boot_sample_mean())
    expect_identical(res$ci_method, method)
  }
  # BootstrapT runs the studentized workflow, which nests `inner_replicates` more resample+fit
  # pairs inside every replicate -- so it is driven small here on purpose.
  res <- bootstrap_custom(
    data = boot_data, resample = boot_resample, fit = boot_fit, statistic = boot_statistic,
    replicates = 40, inner_replicates = 20, seed = 12345, ci_method = "BootstrapT"
  )
  expect_true(res$lower[[1]] < res$upper[[1]])
})

test_that("max_retries reaches the ported class and bounds the retry count", {
  expect_error(
    bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic,
                     replicates = 20, seed = 12345, max_retries = 0),
    "`max_retries` must be a single positive whole number"
  )
  # A fit that always returns a non-finite parameter is a failed replicate by the ported class's
  # own vocabulary (see the NA/NaN symmetry test above), so it is retried up to `max_retries`
  # times and then given up on -- proving the knob reaches C++ rather than being silently ignored,
  # by counting exactly how many times `fit` is called.
  fit_calls <- 0L
  always_fails <- function(data) {
    fit_calls <<- fit_calls + 1L
    NA_real_
  }
  res <- bootstrap_custom(
    boot_data, boot_resample, always_fails, boot_statistic,
    replicates = 5, seed = 12345, parameters = 5, max_retries = 3
  )
  expect_identical(res$failed_replicates, 5L)
  expect_identical(fit_calls, 5L * 3L)
})

test_that("a statistic of more than one value is labelled per statistic", {
  res <- bootstrap_custom(
    data = boot_data, resample = boot_resample, fit = boot_fit,
    statistic = function(parameters) c(parameters[1], parameters[1] * parameters[1]),
    replicates = 200, seed = 12345
  )
  expect_length(res$estimate, 2L)
  expect_equal(res$estimate[[2]], boot_sample_mean()^2)
  expect_true(res$lower[[2]] < res$upper[[2]])
})

test_that("wrong-shaped returns are refused by name", {
  expect_error(
    bootstrap_custom(boot_data, boot_resample, function(data) c(1, 2), boot_statistic,
                     replicates = 20, seed = 12345, parameters = 5),
    "must return one value per parameter"
  )
  expect_error(
    bootstrap_custom(boot_data, function(data, parameters, rng) numeric(0), boot_fit,
                     boot_statistic, replicates = 20, seed = 12345),
    "resample function must return at least one value"
  )
  # A jackknife that drops nothing. The ported BCa would answer it with 0/0 and a NaN interval
  # rather than an error, so the guarded delegate names it instead.
  expect_error(
    bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic,
                     jackknife = function(data, index) data,
                     replicates = 20, seed = 12345, ci_method = "BCa"),
    "must return fewer values than it was given"
  )
  # The 0-based index trap, in the spelling R invites: `data[-index]` is `data[-0]`, which R
  # evaluates to `numeric(0)`, the EMPTY vector, when index is 0 -- and for every later index it
  # silently drops the wrong observation instead. The empty half is caught by name; the off-by-one
  # half is why both packages document the base (see the test below, which pins the R semantics
  # directly).
  expect_error(
    bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic,
                     jackknife = function(data, index) data[-index],
                     replicates = 20, seed = 12345, ci_method = "BCa"),
    "jackknife function must return at least one value"
  )
})

test_that("the naive `data[-index]` jackknife trap is exactly what the docs say it is", {
  # This pins the R language semantics behind the docs' warning under `?bootstrap_custom`, so a
  # future edit to that prose has something executable to disagree with.
  data <- c(10, 20, 30, 40)

  # At index = 0, the naive spelling is `data[-0]`, which R evaluates to the EMPTY vector -- NOT
  # the sample left untouched. This is the half the guarded jackknife catches by name.
  expect_identical(data[-0], numeric(0))

  # At every later index, `data[-index]` is the right LENGTH (so nothing can catch it), but it
  # drops the WRONG observation: `index` is 0-based, so leaving out the value at 0-based index 1
  # (the second element, 20) is `data[-(1 + 1)]`, while the naive `data[-1]` drops the first
  # element (10) instead.
  naive <- data[-1]
  correct <- data[-(1 + 1)]
  expect_length(naive, length(data) - 1L)
  expect_identical(naive, c(20, 30, 40))
  expect_identical(correct, c(10, 30, 40))
  expect_false(identical(naive, correct))
})

test_that("a resample or jackknife returning NA is refused, and a fit or statistic returning NA is not", {
  # The split is deliberate and matches corehydropy's converters exactly. Data has no non-finite
  # meaning, and reading past the end of a vector in R gives NA rather than an error, so an NA
  # sample would surface inside the user's OWN fit two calls later.
  expect_error(
    bootstrap_custom(boot_data, function(data, parameters, rng) c(data[1], NA), boot_fit,
                     boot_statistic, replicates = 20, seed = 12345),
    "resample function returned NA or NaN"
  )
  expect_error(
    bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic,
                     jackknife = function(data, index) c(NA, data[-(index + 1)]),
                     replicates = 20, seed = 12345, ci_method = "BCa"),
    "jackknife function returned NA or NaN"
  )
  # A fit or a statistic is different: the ported class tests those for finiteness itself and reads
  # a non-finite value as "this replicate failed", retries it, and reports the count. Refusing it
  # here would steal that behaviour, so an all-NA fit is a run of failed replicates, not an error.
  failing <- bootstrap_custom(
    boot_data, boot_resample, function(data) NA_real_, boot_statistic,
    replicates = 5, seed = 12345, parameters = 5
  )
  expect_identical(failing$failed_replicates, 5L)
  expect_identical(failing$valid_count[[1]], 0L)
})

# --- bootstrap_custom, the pivotal run type ---------------------------------------------------
#
# The pivotal bootstrap fits through ONE delegate the other run type does not have: a
# covariance-aware fit returning the parameters AND their covariance. The model here is the
# two-parameter Normal location-scale MLE of the same eight observations, whose covariance is
# analytic -- diag(s2 / n, s2 / (2n)) -- so the whole callback is arithmetic plus sqrt, and sqrt is
# the one libm function IEEE 754 requires to be correctly rounded, so the identical call in Python
# computes the identical numbers. The C#-pinned oracle for this same model lives in
# fixtures/callback/bootstrap.json, and its cross-language twin at ZERO tolerance in
# fixtures/callback/callback_cross_language.json.
boot_fit_with_covariance <- function(data) {
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
}

test_that("a seeded pivotal run reports its diagnostics and both interval blocks", {
  res <- bootstrap_custom(
    data = boot_data, resample = boot_resample, statistic = boot_statistic,
    fit_with_covariance = boot_fit_with_covariance, run_type = "pivotal",
    replicates = 200, seed = 12345
  )
  expect_identical(res$pivotal_diagnostics$requested_replicates, 200L)
  expect_identical(res$pivotal_diagnostics$failed_raw_replicates, 0L)
  expect_identical(res$pivotal_diagnostics$rejected_raw_replicates, 0L)
  expect_identical(res$pivotal_diagnostics$accepted_raw_replicates, 200L)
  expect_identical(res$pivotal_diagnostics$retained_pivotal_replicates, 200L)
  expect_identical(
    res$pivotal_diagnostics$retained_pivotal_replicates +
      res$pivotal_diagnostics$invalid_pivotal_replicates,
    res$pivotal_diagnostics$accepted_raw_replicates
  )
  expect_identical(res$run_type, "pivotal")

  # Two interval blocks, not one: the pivotal ensemble and the raw covariance-aware fits it was
  # built from (GetRawPivotalConfidenceIntervals). Both bracket the population estimate.
  expect_length(res$estimate, 2L)
  expect_true(res$lower[[1]] < res$estimate[[1]] && res$estimate[[1]] < res$upper[[1]])
  expect_true(res$raw_lower[[1]] < res$estimate[[1]] && res$estimate[[1]] < res$raw_upper[[1]])
  # And they are NOT the same interval: the transform reinflates each raw fit through the parent
  # covariance, so the two blocks disagree even though they come from one run.
  expect_false(isTRUE(all.equal(res$lower[[1]], res$raw_lower[[1]])))
  expect_false(isTRUE(all.equal(res$upper[[2]], res$raw_upper[[2]])))
})

test_that("a seeded pivotal run repeats exactly and moves with the seed", {
  run <- function(seed) {
    bootstrap_custom(
      data = boot_data, resample = boot_resample, statistic = boot_statistic,
      fit_with_covariance = boot_fit_with_covariance, run_type = "pivotal",
      replicates = 100, seed = seed
    )$lower[[1]]
  }
  expect_identical(run(12345), run(12345))
  expect_false(identical(run(12345), run(999)))
})

test_that("the pivotal options reach the ported class", {
  base <- function(...) {
    bootstrap_custom(
      data = boot_data, resample = boot_resample, statistic = boot_statistic,
      fit_with_covariance = boot_fit_with_covariance, run_type = "pivotal",
      replicates = 100, seed = 12345, ...
    )
  }
  plain <- base()

  # A z limit small enough to reject draws: the Drop default then retains fewer than it accepted,
  # and the two counts still add up.
  limited <- base(pivotal_z_limit = 0.5)
  expect_true(limited$pivotal_diagnostics$invalid_pivotal_replicates > 0L)
  expect_identical(
    limited$pivotal_diagnostics$retained_pivotal_replicates,
    limited$pivotal_diagnostics$accepted_raw_replicates -
      limited$pivotal_diagnostics$invalid_pivotal_replicates
  )
  # UseRaw keeps every draw instead of dropping the invalid ones, which is the one thing the
  # policy can be observed to do.
  kept <- base(pivotal_z_limit = 0.5, pivotal_invalid_draw_policy = "use_raw")
  expect_identical(
    kept$pivotal_diagnostics$retained_pivotal_replicates,
    kept$pivotal_diagnostics$accepted_raw_replicates
  )
  expect_identical(
    kept$pivotal_diagnostics$invalid_pivotal_replicates,
    limited$pivotal_diagnostics$invalid_pivotal_replicates
  )

  # Jitter perturbs the standardized vector, so it must move the interval and nothing else.
  jittered <- base(add_pivotal_jitter = TRUE, pivotal_jitter_scale = 0.5)
  expect_false(isTRUE(all.equal(jittered$lower[[1]], plain$lower[[1]])))
  # A named link is resolved core-side, so it must move the interval. NOTE the link is on the MEAN
  # here, not on the scale, and that is not arbitrary: for this model's analytic covariance the log
  # link on sigma is invariant. Identity gives a pivotal draw of a + (a / r)(a - r) = a^2 / r
  # (because the covariance of sigma-hat is sigma^2 / 2n, so the ratio of the two Cholesky factors
  # is a / r), and the log link gives exp(2 log a - log r) = a^2 / r as well -- the same number to
  # about 1e-11. A test that linked sigma would therefore pass whether or not the link reached the
  # core at all.
  linked <- base(pivotal_links = list("Log", NULL))
  expect_false(isTRUE(all.equal(linked$lower[[1]], plain$lower[[1]])))
})

test_that("an error raised inside fit_with_covariance reaches the caller", {
  expect_error(
    bootstrap_custom(
      data = boot_data, resample = boot_resample, statistic = boot_statistic,
      fit_with_covariance = function(data) stop("my own covariance fit error"),
      run_type = "pivotal", replicates = 20, seed = 12345
    ),
    "my own covariance fit error"
  )
})

test_that("the pivotal run type refuses the arguments it cannot use", {
  expect_error(
    bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic, run_type = "nope"),
    "unknown run_type 'nope'; expected one of"
  )
  expect_error(
    bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic, run_type = "pivotal"),
    "`fit_with_covariance` must be a function taking \\(data\\)"
  )
  # `fit` is not used by the pivotal run: refused rather than silently ignored.
  expect_error(
    bootstrap_custom(
      data = boot_data, resample = boot_resample, fit = boot_fit, statistic = boot_statistic,
      fit_with_covariance = boot_fit_with_covariance, run_type = "pivotal",
      replicates = 20, seed = 12345
    ),
    "`fit` is not used"
  )
  expect_error(
    bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic,
                     fit_with_covariance = boot_fit_with_covariance),
    "only used when `run_type` is \"pivotal\""
  )
  expect_error(
    bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic,
                     pivotal_z_limit = 3),
    "only used when `run_type` is \"pivotal\""
  )
  # And the other way round: the two arguments the pivotal run has no use for. Neither the BCa
  # jackknife nor the studentized inner replicates can be reached from it, since it takes only a
  # percentile interval.
  for (unused in c("jackknife", "inner_replicates")) {
    args <- list(
      data = boot_data, resample = boot_resample, statistic = boot_statistic,
      fit_with_covariance = boot_fit_with_covariance, run_type = "pivotal",
      replicates = 20, seed = 12345
    )
    args[[unused]] <- if (identical(unused, "jackknife")) boot_jackknife else 20
    expect_error(do.call(bootstrap_custom, args),
                 sprintf("`%s` is not used when `run_type` is \"pivotal\"", unused))
  }
  # C# refuses anything but a percentile interval after a pivotal run; the refusal is made before
  # the first replicate here rather than after all of them, exactly as the BCa one is.
  calls <- 0L
  counting_resample <- function(data, parameters, rng) {
    calls <<- calls + 1L
    boot_resample(data, parameters, rng)
  }
  expect_error(
    bootstrap_custom(
      data = boot_data, resample = counting_resample, statistic = boot_statistic,
      fit_with_covariance = boot_fit_with_covariance, run_type = "pivotal",
      replicates = 100, seed = 12345, ci_method = "BiasCorrected"
    ),
    "percentile"
  )
  expect_identical(calls, 0L)
  # One link per parameter, checked before the run rather than inside it.
  expect_error(
    bootstrap_custom(
      data = boot_data, resample = boot_resample, statistic = boot_statistic,
      fit_with_covariance = boot_fit_with_covariance, run_type = "pivotal",
      replicates = 20, seed = 12345, pivotal_links = list("Log")
    ),
    "one link per parameter"
  )
  expect_error(
    bootstrap_custom(
      data = boot_data, resample = boot_resample, statistic = boot_statistic,
      fit_with_covariance = boot_fit_with_covariance, run_type = "pivotal",
      replicates = 20, seed = 12345, pivotal_links = list("Nope", NULL)
    ),
    "unknown link type"
  )
  # A covariance-aware fit returning the wrong shape is named rather than left to fail every
  # replicate in silence.
  expect_error(
    bootstrap_custom(
      data = boot_data, resample = boot_resample, statistic = boot_statistic,
      fit_with_covariance = function(data) list(parameters = c(1, 2), covariance = matrix(1, 1, 1)),
      run_type = "pivotal", replicates = 20, seed = 12345
    ),
    "covariance"
  )
  expect_error(
    bootstrap_custom(
      data = boot_data, resample = boot_resample, statistic = boot_statistic,
      fit_with_covariance = function(data) c(1, 2),
      run_type = "pivotal", replicates = 20, seed = 12345
    ),
    "'parameters'"
  )
})

test_that("an explicit original_covariance replaces the one the fit reports", {
  # Supplying the parent covariance is the C# second constructor's own shape; a much wider one
  # reinflates every draw further, so the pivotal interval has to widen with it.
  narrow <- bootstrap_custom(
    data = boot_data, resample = boot_resample, statistic = boot_statistic,
    fit_with_covariance = boot_fit_with_covariance, run_type = "pivotal",
    replicates = 100, seed = 12345
  )
  wide <- bootstrap_custom(
    data = boot_data, resample = boot_resample, statistic = boot_statistic,
    fit_with_covariance = boot_fit_with_covariance, run_type = "pivotal",
    original_covariance = matrix(c(1, 0, 0, 1), nrow = 2L, ncol = 2L),
    replicates = 100, seed = 12345
  )
  expect_true((wide$upper[[1]] - wide$lower[[1]]) > (narrow$upper[[1]] - narrow$lower[[1]]))
  # The raw block is the same either way: it is the raw fits, which the parent covariance does
  # not touch.
  expect_equal(wide$raw_lower[[1]], narrow$raw_lower[[1]])
})

test_that("bootstrap_custom refuses arguments that are not what they claim", {
  expect_error(bootstrap_custom(numeric(0), boot_resample, boot_fit, boot_statistic),
               "`data` must be a non-empty numeric vector")
  expect_error(bootstrap_custom(boot_data, "nope", boot_fit, boot_statistic),
               "`resample` must be a function taking \\(data, parameters, rng\\)")
  expect_error(bootstrap_custom(boot_data, boot_resample, "nope", boot_statistic),
               "`fit` must be a function taking \\(data\\)")
  expect_error(bootstrap_custom(boot_data, boot_resample, boot_fit, "nope"),
               "`statistic` must be a function taking \\(parameters\\)")
  expect_error(bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic,
                                jackknife = "nope"),
               "`jackknife` must be a function taking \\(data, index\\)")
  expect_error(bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic,
                                replicates = 0),
               "`replicates` must be a single positive whole number")
  expect_error(bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic, alpha = 1),
               "`alpha` must be a single number between 0 and 1")
  expect_error(bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic,
                                ci_method = "Nope"),
               "unknown ci_method 'Nope'; expected one of")
  # A prefix is refused for the same reason mcmc_posterior() refuses one.
  expect_error(bootstrap_custom(boot_data, boot_resample, boot_fit, boot_statistic,
                                ci_method = "Perc"),
               "unknown ci_method 'Perc'; expected one of")
})

test_that("a handle leaked out of a resample callback is dead afterwards", {
  # The same lifetime guarantee the Gibbs proposal's handle carries, at the second site that hands
  # one out: the borrow is invalidated when the callback returns, so a stored handle raises rather
  # than reading freed memory.
  leaked <- NULL
  bootstrap_custom(
    boot_data,
    function(data, parameters, rng) {
      leaked <<- rng
      boot_resample(data, parameters, rng)
    },
    boot_fit, boot_statistic, replicates = 5, seed = 12345
  )
  expect_error(rng_uniform(leaked, 1), "no longer valid")
})

# --- fit_gmm_moments ----------------------------------------------------------------------
#
# The model throughout is the just-identified two-parameter method-of-moments fit of a Normal:
# theta = (mu, sigma2) and
#
#   g(theta) = [mean(x - mu), mean((x - mu)^2 - sigma2)]
#
# whose unique root -- and so the GMM optimum, since q = p makes g(theta-hat) = 0 attainable -- is
# the sample mean and the POPULATION variance. So these tests check arithmetic anyone can do by
# hand rather than a regression against whatever the optimizer returned. The C#-pinned oracle for
# this same model lives in fixtures/callback/gmm.json. Every sum is an explicit `for` loop, never
# sum() or mean(): both accumulate in extended precision in R where the shared core, Python and C#
# accumulate in double, and one differing bit moves a fitted parameter.
gmm_data <- c(4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7)

gmm_moments <- function(p) {
  n <- length(gmm_data)
  g0 <- 0
  g1 <- 0
  s00 <- 0
  s01 <- 0
  s11 <- 0
  for (x in gmm_data) {
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
    s = matrix(c(s00 / n, s01 / n, s01 / n, s11 / n), nrow = 2, ncol = 2)
  )
}

gmm_jacobian <- function(p) {
  acc <- 0
  for (x in gmm_data) acc <- acc + (x - p[1])
  matrix(c(-1, -2 * acc / length(gmm_data), 0, -1), nrow = 2, ncol = 2)
}

gmm_fit_it <- function(...) {
  fit_gmm_moments(gmm_moments,
    initial = c(5, 0.5), lower = c(0, 0.001), upper = c(10, 10),
    sample_size = length(gmm_data), ...
  )
}

gmm_closed_form <- function() {
  n <- length(gmm_data)
  acc <- 0
  for (x in gmm_data) acc <- acc + x
  mu <- acc / n
  acc2 <- 0
  for (x in gmm_data) acc2 <- acc2 + (x - mu) * (x - mu)
  c(mu, acc2 / n)
}

test_that("fit_gmm_moments recovers the closed-form method-of-moments solution", {
  f <- gmm_fit_it()
  expect_s3_class(f, "corehydro_fit")
  expect_identical(f$method, "GMM")
  expect_identical(names(coef(f)), c("p1", "p2"))
  expect_equal(unname(coef(f)), gmm_closed_form(), tolerance = 1e-9)
  expect_identical(f$nobs, length(gmm_data))
  expect_identical(f$number_of_moment_conditions, 2L)
  expect_true(f$converged)
  # The accessors behave exactly as they do for a fit_gmm() fit.
  expect_identical(dim(vcov(f)), c(2L, 2L))
  expect_identical(dimnames(vcov(f)), list(c("p1", "p2"), c("p1", "p2")))
  expect_true(all(f$standard_errors > 0))
  expect_equal(unname(f$standard_errors), unname(sqrt(diag(vcov(f)))), tolerance = 1e-12)
  expect_identical(logLik(f)[[1]], NA_real_)
  expect_true(is.na(AIC(f)))
  expect_output(print(f), "j-statistic")
  expect_error(confint(f), "no interval surface")
})

test_that("a just-identified fit reports NA for the J-statistic p-value", {
  f <- gmm_fit_it()
  # Zero degrees of freedom: q == p leaves no over-identifying restriction to test the
  # specification with, so the ported post_process() writes NaN and this package reports NA. That
  # is correct rather than a defect -- see docs/upstream-csharp-issues.md. J itself is ~0 by
  # construction (g(theta-hat) is driven to zero) and is cancellation noise, so it is only checked
  # against zero.
  expect_identical(f$degree_of_freedom, 0L)
  expect_true(is.na(f$j_stat_pval))
})

test_that("the J-statistic never fails a just-identified fit, whatever inverting a singular V gives", {
  # J itself is NOT asserted, deliberately. On a just-identified fit the residual covariance it is
  # scaled by is theoretically zero, so g' V^-1 g is whatever inverting a numerically singular
  # matrix gives: measured across these four optimizers, all agreeing on the parameters to 1e-9,
  # -1.3e-09, 8.6e+19, -7.7e-15 and 0.126 -- and under NelderMead it throws outright from R, where
  # the core reports NA rather than failing an otherwise exact fit. THAT is what this pins.
  for (optimizer in c("BFGS", "NelderMead", "Powell", "MultilevelSingleLinkage")) {
    f <- gmm_fit_it(optimizer = optimizer)
    expect_equal(coef(f)[["p1"]], gmm_closed_form()[[1]], tolerance = 1e-5)
    expect_true(is.na(f$j_stat_pval))
  }
})

# The OVER-IDENTIFIED companion of gmm_moments: the same Normal model and the same eight
# observations with mean((x - mu)^3) added as a third condition, zero for a Normal and so leaving
# the model itself unchanged. q = 3 > p = 2. Three things are reachable only from here -- a
# non-zero degrees of freedom, the chi-squared p-value branch, and the refusal of OneStep.
gmm_moments3 <- function(p) {
  n <- length(gmm_data)
  g0 <- 0
  g1 <- 0
  g2 <- 0
  s00 <- 0
  s01 <- 0
  s02 <- 0
  s11 <- 0
  s12 <- 0
  s22 <- 0
  for (x in gmm_data) {
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
    s = matrix(
      c(s00 / n, s01 / n, s02 / n, s01 / n, s11 / n, s12 / n, s02 / n, s12 / n, s22 / n),
      nrow = 3, ncol = 3
    )
  )
}

test_that("an over-identified fit reports a real p-value from the chi-squared branch", {
  f <- fit_gmm_moments(gmm_moments3, c(5, 0.5), c(0, 0.001), c(10, 10), length(gmm_data))
  expect_identical(f$number_of_moment_conditions, 3L)
  expect_identical(f$degree_of_freedom, 1L)
  # THE property this case exists for. Everywhere else on this surface q == p, the degrees of
  # freedom are zero and the ported post_process() writes a structural NaN; here it takes its
  # chi-squared branch instead and the p-value is a real probability.
  expect_false(is.na(f$j_stat_pval))
  expect_gte(f$j_stat_pval, 0)
  expect_lte(f$j_stat_pval, 1)
  # J ITSELF is still not asserted, and over-identifying is often assumed to fix that and does
  # not: V has rank exactly q - p for any q and p, so it is singular here too -- the rank only
  # moves from 0 to 1 -- and inverting it amplifies the optimizer's convergence tolerance. On this
  # fit the real C# library returns 214.59 where the shared core returns -129.46, from parameters
  # that agree to 2e-11. So the branch being TAKEN is what is pinned, not what it produced.
  # The C#-pinned oracle for everything that does reproduce is fixtures/callback/gmm.json.
  expect_true(coef(f)[["p1"]] != gmm_closed_form()[[1]])
  expect_true(all(f$standard_errors > 0))
  expect_identical(dim(vcov(f)), c(2L, 2L)) # p x p, not q x q
})

test_that("OneStep is refused for an over-identified problem", {
  # The ported estimator's own validation, and the one strategy/identification combination no
  # just-identified fit on this surface can reach.
  expect_error(
    fit_gmm_moments(gmm_moments3, c(5, 0.5), c(0, 0.001), c(10, 10), length(gmm_data),
                    strategy = "OneStep"),
    "over-identified, so you cannot use the one-step estimation method"
  )
  # TwoStep and Iterative are accepted for the same problem.
  for (strategy in c("TwoStep", "Iterative")) {
    f <- fit_gmm_moments(gmm_moments3, c(5, 0.5), c(0, 0.001), c(10, 10), length(gmm_data),
                         strategy = strategy)
    expect_identical(f$degree_of_freedom, 1L)
  }
})

test_that("print() shows the J-statistic only where it means something", {
  # At zero degrees of freedom the number is whatever inverting a singular matrix gave, so
  # print() names the reason instead of showing it. The field itself stays on the fit.
  just <- gmm_fit_it()
  expect_output(print(just), "j-statistic: not interpretable at 0 degrees of freedom")
  expect_false(is.null(just$j_stat))

  over <- fit_gmm_moments(gmm_moments3, c(5, 0.5), c(0, 0.001), c(10, 10), length(gmm_data))
  expect_output(print(over), sprintf("j-statistic: %g", over$j_stat))
  expect_output(print(over), "p-value")

  # Non-zero degrees of freedom is not on its own enough: the residual covariance can be singular
  # enough that inverting it raises, in which case the ported post_process() reports NaN. "nan" on
  # that line says less than saying so, so the display is gated on isfinite() as well as on the
  # degrees of freedom.
  uncomputable <- over
  uncomputable$j_stat <- NaN
  expect_output(print(uncomputable), "j-statistic: could not be computed for this fit")
  expect_false(any(grepl("nan", capture.output(print(uncomputable)), fixed = TRUE)))
})

test_that("two fit_gmm_moments runs are identical: there is no generator in this fit", {
  a <- gmm_fit_it()
  b <- gmm_fit_it()
  expect_identical(coef(a), coef(b))
  expect_identical(vcov(a), vcov(b))
})

test_that("the analytic jacobian and the penalty both reach the estimator", {
  plain <- gmm_fit_it()
  analytic <- gmm_fit_it(jacobian = gmm_jacobian)
  expect_equal(unname(coef(analytic)), gmm_closed_form(), tolerance = 1e-9)
  # NOTE what this pair does NOT prove. gmm_moments is quadratic in mu and linear in sigma2, so its
  # Jacobian is linear and the ported central difference computes it exactly; the two fits agree to
  # 5.4e-13 whether the delegate is used or ignored. The test that the delegate's RETURN is used is
  # the cubic one below, and fixtures/callback/gmm.json's analytic_jacobian_cubic is its oracle.
  expect_equal(unname(vcov(analytic)), unname(vcov(plain)), tolerance = 1e-9)

  # A ridge penalty pulling sigma2 towards 1 moves the estimate off the closed form. An ignored
  # penalty delegate would return the unpenalized answer, which is what this compares against.
  penalized <- gmm_fit_it(penalty = function(p) 0.5 * (p[2] - 1) * (p[2] - 1))
  expect_true(coef(penalized)[["p2"]] > coef(plain)[["p2"]])
  expect_true(coef(penalized)[["p2"]] < 1)
})

# The CUBIC-JACOBIAN model, the one that can tell an analytic Jacobian from the ported numerical
# one. theta = (mu, sigma) matched on the first and fourth central moments of a Normal, over the
# same eight observations with the decimal point moved two places: dg2/dsigma = -1200 t^3 is cubic,
# and a central difference is exact only up to a quadratic derivative. The C#-pinned oracle for
# this exact fit is fixtures/callback/gmm.json's analytic_jacobian_cubic.
gmm_small_data <- c(0.041, 0.052, 0.048, 0.055, 0.049, 0.051, 0.053, 0.047)

gmm_moments4 <- function(p) {
  n <- length(gmm_small_data)
  t <- p[2] * 100
  t4 <- 3 * t * t * t * t
  g0 <- 0
  g1 <- 0
  s00 <- 0
  s01 <- 0
  s11 <- 0
  for (x in gmm_small_data) {
    a <- x - p[1]
    u <- a * 100
    b <- u * u * u * u - t4
    g0 <- g0 + a
    g1 <- g1 + b
    s00 <- s00 + a * a
    s01 <- s01 + a * b
    s11 <- s11 + b * b
  }
  list(g = c(g0 / n, g1 / n), s = matrix(c(s00 / n, s01 / n, s01 / n, s11 / n), nrow = 2, ncol = 2))
}

gmm_jacobian4 <- function(p) {
  acc <- 0
  for (x in gmm_small_data) {
    u <- (x - p[1]) * 100
    acc <- acc + u * u * u
  }
  t <- p[2] * 100
  matrix(c(-1, -400 * acc / 8, 0, -1200 * t * t * t), nrow = 2, ncol = 2)
}

test_that("an analytic jacobian the numerical one cannot reproduce changes the answer", {
  numerical <- fit_gmm_moments(gmm_moments4, c(0.05, 0.005), c(0, 1e-4), c(0.5, 0.5), 8)
  analytic <- fit_gmm_moments(gmm_moments4, c(0.05, 0.005), c(0, 1e-4), c(0.5, 0.5), 8,
                              jacobian = gmm_jacobian4)

  # Both fits land on the closed form: the root is the sample mean and (mean((x - mu)^4) / 3)^(1/4),
  # and both Jacobians point downhill to it. The PARAMETERS are not what discriminates.
  expect_equal(coef(analytic)[["p1"]], 0.0495, tolerance = 1e-8)
  expect_equal(coef(analytic)[["p2"]], (sum((gmm_small_data - 0.0495)^4) / 8 / 3)^0.25,
               tolerance = 1e-8)

  # The SANDWICH is. D enters the bread D'WD, so a Jacobian the central difference cannot reproduce
  # moves the standard errors and the covariance by far more than the 1e-7 the fixture pins them
  # at: measured here, 6.2e-04 on the sigma standard error and 1.2e-03 on its variance.
  se_a <- sqrt(diag(vcov(analytic)))
  se_n <- sqrt(diag(vcov(numerical)))
  expect_gt(abs(se_a[[2]] - se_n[[2]]) / se_a[[2]], 1e-4)
  expect_gt(abs(vcov(analytic)[2, 2] - vcov(numerical)[2, 2]) / vcov(analytic)[2, 2], 1e-3)
  # The mu column of the Jacobian is exact either way (dg1/dmu = -1), so its standard error does
  # not move. That is the control: the difference above is the cubic column, not a fit that wandered.
  expect_equal(se_a[[1]], se_n[[1]], tolerance = 1e-9)
})

test_that("every strategy fit_gmm_moments accepts finds the same optimum", {
  # The optimizers are covered by the J-statistic test above, which fits with all four.
  for (strategy in c("OneStep", "TwoStep", "Iterative")) {
    expect_equal(coef(gmm_fit_it(strategy = strategy))[["p1"]], gmm_closed_form()[[1]],
                 tolerance = 1e-8)
  }
})

test_that("a moment condition function returning the wrong shape is refused naming g and s", {
  # A FLAT PAIR of numbers -- the likeliest mistake when q is 2, since `c(g0, g1)` looks like just
  # the moment vector. It must be refused by the message naming BOTH elements, not by some later
  # message about one of them; the Python binding needs an explicit check to say the same sentence
  # here, and test_callback.py's counterpart of this test is what pins it.
  expect_error(
    fit_gmm_moments(function(p) c(1, 2), c(5, 0.5), c(0, 0.001), c(10, 10), 8),
    "elements 'g' \\(the moment vector\\) and 's' \\(the weighting matrix\\)"
  )
  # The same mistake with the real numbers in it, which is what a user actually writes.
  expect_error(
    fit_gmm_moments(
      function(p) {
        out <- gmm_moments(p)
        c(out$g[1], out$g[2])
      },
      c(5, 0.5), c(0, 0.001), c(10, 10), 8
    ),
    "elements 'g' \\(the moment vector\\) and 's' \\(the weighting matrix\\)"
  )
  # A list, but not the right one: only `g`.
  expect_error(
    fit_gmm_moments(function(p) list(g = c(0, 0)), c(5, 0.5), c(0, 0.001), c(10, 10), 8),
    "elements 'g' \\(the moment vector\\) and 's' \\(the weighting matrix\\)"
  )
  # The two the wrong way round: `s` holding the moment vector, `g` holding the matrix.
  expect_error(
    fit_gmm_moments(
      function(p) {
        out <- gmm_moments(p)
        list(g = out$s, s = out$g)
      },
      c(5, 0.5), c(0, 0.001), c(10, 10), 8
    ),
    "'s' \\(the weighting matrix\\) must be a numeric matrix"
  )
  # A moment vector whose length changes between calls, the only way it can disagree with the q
  # the up-front probe measured.
  calls <- 0
  expect_error(
    fit_gmm_moments(
      function(p) {
        calls <<- calls + 1
        out <- gmm_moments(p)
        if (calls > 1) out$g <- c(out$g, 0)
        out
      },
      c(5, 0.5), c(0, 0.001), c(10, 10), 8
    ),
    "'g' \\(the moment vector\\) must hold one value per moment condition"
  )
  # An `s` that is not square.
  expect_error(
    fit_gmm_moments(
      function(p) list(g = gmm_moments(p)$g, s = matrix(0, nrow = 2, ncol = 3)),
      c(5, 0.5), c(0, 0.001), c(10, 10), 8
    ),
    "'s' \\(the weighting matrix\\) must be a square matrix"
  )
  # A jacobian of the wrong shape: q x p, and the message says which way round.
  expect_error(
    gmm_fit_it(jacobian = function(p) matrix(0, nrow = 1, ncol = 3)),
    "jacobian function must return a 2 x 2 matrix"
  )
})

test_that("an error raised inside each of the three delegates reaches the caller", {
  # One test per delegate: a guard wired for one and missing on another is exactly the shape of
  # bug this invites. The moment conditions are tested twice, because throwing on the FIRST call
  # is caught by the up-front probe (before the estimator exists) and throwing later is the case
  # the guard has to carry -- by then the ported optimizer's catch-all would otherwise swallow it.
  expect_error(
    fit_gmm_moments(function(p) stop("boom"), c(5, 0.5), c(0, 0.001), c(10, 10), 8),
    "boom"
  )
  calls <- 0
  expect_error(
    fit_gmm_moments(
      function(p) {
        calls <<- calls + 1
        if (calls > 1) stop("boom later")
        gmm_moments(p)
      },
      c(5, 0.5), c(0, 0.001), c(10, 10), 8
    ),
    "boom later"
  )
  expect_error(gmm_fit_it(jacobian = function(p) stop("jacobian boom")), "jacobian boom")
  expect_error(
    gmm_fit_it(penalty = function(p) stop("penalty boom")),
    "penalty boom"
  )
})

test_that("a one-moment-condition model may write g and s as bare numbers", {
  # R has no scalar type, so `g` here IS a length-1 numeric and `s` a length-1 numeric with no
  # `dim`. Both have always been accepted; the counterpart in test_callback.py now accepts the same
  # spelling in Python, which it did not before, and this test is the R half of that pair.
  one_moment <- function(p) {
    g <- 0
    s <- 0
    for (x in gmm_data) {
      a <- x - p[1]
      g <- g + a
      s <- s + a * a
    }
    list(g = g / 8, s = s / 8)
  }
  fit <- fit_gmm_moments(one_moment, 5, 0, 10, 8)
  expect_identical(fit$number_of_moment_conditions, 1L)
  expect_equal(coef(fit)[["p1"]], 4.95, tolerance = 1e-8)
})

test_that("an error raised in the post-processing re-entry still reports the user's message", {
  # post_process() recomputes S, the Jacobian and g at the fitted parameters AFTER the fit itself
  # has finished, so it is a THIRD place the moment function is entered -- and its own throw is
  # deliberately swallowed there, so that an uncomputable J-statistic does not fail an otherwise
  # exact fit. An error raised for the first time inside that re-entry must still surface with the
  # user's own wording, not the ported "Singular matrix in LU decomposition" the guard's
  # zero-moment sentinel provokes one line later.
  #
  # The throw goes on the LAST call a clean run makes, which is inside post_process by
  # construction: nothing but post_process runs after the fit, and it does call the function
  # (measured against the estimator for this fit: 115 calls in estimate(), 14 more in
  # post_process()). Throwing one call later never fires, which is what pins the count.
  calls <- 0
  counting <- function(p) {
    calls <<- calls + 1
    gmm_moments(p)
  }
  fit_gmm_moments(counting, c(5, 0.5), c(0, 0.001), c(10, 10), 8)
  total <- calls
  expect_gt(total, 1)

  calls <- 0
  expect_error(
    fit_gmm_moments(
      function(p) {
        calls <<- calls + 1
        if (calls == total) stop("boom in post-processing")
        gmm_moments(p)
      },
      c(5, 0.5), c(0, 0.001), c(10, 10), 8
    ),
    "boom in post-processing"
  )

  calls <- 0
  never <- fit_gmm_moments(
    function(p) {
      calls <<- calls + 1
      if (calls == total + 1) stop("never reached")
      gmm_moments(p)
    },
    c(5, 0.5), c(0, 0.001), c(10, 10), 8
  )
  expect_equal(unname(coef(never)), gmm_closed_form(), tolerance = 1e-9)
})

test_that("NA is refused in s and the jacobian, and allowed in g and the penalty", {
  # The rule, drawn by what each value MEANS to the ported estimator and identical in Python:
  # Q() ends with `is_finite(qv) ? qv : double.max`, so a non-finite `g` (or penalty) is the
  # estimator's own way of learning a corner of the box is infeasible; nothing tests `s` or the
  # jacobian, so an NA there would reach the fitted standard errors without a word.
  expect_error(
    fit_gmm_moments(
      function(p) list(g = gmm_moments(p)$g, s = matrix(NA_real_, 2, 2)),
      c(5, 0.5), c(0, 0.001), c(10, 10), 8
    ),
    "'s' \\(the weighting matrix\\) returned NA or NaN"
  )
  expect_error(
    gmm_fit_it(jacobian = function(p) matrix(NA_real_, 2, 2)),
    "jacobian function returned NA or NaN"
  )
  # `g` may go non-finite at a trial point and the fit still completes: NA only at parameters far
  # from the optimum, which the optimizer then steps away from.
  na_far_away <- fit_gmm_moments(
    function(p) {
      out <- gmm_moments(p)
      if (p[2] > 5) out$g <- c(NA_real_, NA_real_)
      out
    },
    c(5, 0.5), c(0, 0.001), c(10, 10), 8
  )
  expect_equal(unname(coef(na_far_away)), gmm_closed_form(), tolerance = 1e-9)
  # And a penalty may too.
  expect_s3_class(gmm_fit_it(penalty = function(p) if (p[2] > 5) NA_real_ else 0), "corehydro_fit")
})

test_that("fit_gmm_moments refuses arguments that are not what they claim", {
  expect_error(fit_gmm_moments("nope", c(5, 0.5), c(0, 0.001), c(10, 10), 8),
               "`moment_conditions` must be a function")
  expect_error(
    fit_gmm_moments(gmm_moments, c(5, 0.5), sample_size = 8),
    "needs `lower` and `upper` bounds"
  )
  expect_error(
    fit_gmm_moments(gmm_moments, c(5, 0.5), c(0), c(10, 10), 8),
    "must be the same length; they are 2, 1 and 2"
  )
  expect_error(
    fit_gmm_moments(gmm_moments, c(5, 0.5), c(0, 0.001), c(10, 10)),
    "`sample_size` must be a single positive whole number"
  )
  expect_error(gmm_fit_it(optimizer = "Nope"), "unknown optimizer 'Nope'")
  expect_error(gmm_fit_it(strategy = "Nope"), "unknown GMM estimation strategy 'Nope'")
  expect_error(gmm_fit_it(jacobian = "nope"), "`jacobian` must be a function")
  expect_error(gmm_fit_it(penalty = "nope"), "`penalty` must be a function")
})

test_that("the model-only verbs refuse a fit_gmm_moments() fit by name", {
  f <- gmm_fit_it()
  # Both need the model a fit_gmm() fit carries; this one has only the user's moment conditions.
  expect_error(quantile_variance(f, 0.01), "no distribution to take a quantile of")
  expect_error(fit_diagnostics(f), "needs a fit built from a model")
  expect_null(f$model)
})
