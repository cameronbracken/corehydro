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
  # A null external pointer is refused too, by the address check behind the tag check.
  null_ptr <- methods::new("externalptr")
  expect_error(rng_uniform(null_ptr, 1), "random number generator handle")
  # A forged class attribute buys nothing: the class is for print/inherits, never for dispatch.
  fake <- structure(list(), class = "corehydro_rng")
  expect_error(rng_uniform(fake, 1), "random number generator handle")
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
