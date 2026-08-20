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
  # x^2 needs no subdivision (G10K21 is exact for it), so the accumulated error is exactly zero.
  q <- quadrature(function(x) x^2, lower = 0, upper = 3)
  expect_equal(attr(q, "standard_error"), 0)
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
    "'arg' should be one of"
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
