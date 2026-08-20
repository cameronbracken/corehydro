import copy
import math
import pickle

import numpy as np
import pytest

import corehydropy as ch
from corehydropy.callback import _rng_probe


def _boom(*_args):
    raise ValueError("my own error")


def test_root_find_solves_a_python_function():
    assert ch.root_find(lambda x: x**2 - 2, lower=0, upper=2) == pytest.approx(
        math.sqrt(2), abs=1e-8
    )


def test_root_find_needs_a_bracketed_root():
    with pytest.raises(Exception, match="not bracketed"):
        ch.root_find(lambda x: x**2 + 1, lower=0, upper=2)


def test_derivative_differentiates_a_python_function():
    # f(x) = x^3, f'(2) = 12
    assert ch.derivative(lambda x: x**3, 2) == pytest.approx(12.0, abs=1e-6)
    # f(x) = sin(x), f'(pi / 3) = cos(pi / 3) = 0.5
    assert ch.derivative(math.sin, math.pi / 3) == pytest.approx(0.5, abs=1e-6)


def test_gradient_and_hessian_differentiate_a_python_function():
    def rosenbrock(p):
        return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2

    g = ch.gradient(rosenbrock, [1.0, 1.0])
    assert g == pytest.approx([0.0, 0.0], abs=1e-6)
    h = ch.hessian(rosenbrock, [1.0, 1.0])
    assert h.shape == (2, 2)
    assert h[0, 1] == pytest.approx(h[1, 0], abs=1e-6)

    # f(x, y) = x^2 + 2y^2 + xy has the constant Hessian [[2, 1], [1, 4]].
    quad = ch.hessian(lambda p: p[0] ** 2 + 2 * p[1] ** 2 + p[0] * p[1], [1.0, 2.0])
    assert quad == pytest.approx(np.array([[2.0, 1.0], [1.0, 4.0]]), abs=1e-3)


def test_quadrature_integrates_a_python_function():
    # Closed forms only: the C#-pinned oracles live in fixtures/callback/math.json.
    q = ch.quadrature(lambda x: x**2, 0, 3)
    assert q == pytest.approx(9.0, abs=1e-10)
    assert q.status == "Success"
    assert q.function_evaluations > 0
    # The result IS a float, so it composes with anything expecting one.
    assert isinstance(q, float)
    assert ch.quadrature(math.sin, 0, math.pi) == pytest.approx(2.0, abs=1e-8)
    assert ch.quadrature(math.exp, 0, 1) == pytest.approx(math.e - 1, abs=1e-8)


def test_quadrature_reports_the_rules_own_standard_error():
    # x**2 needs no subdivision (G10K21 is exact for it), so the accumulated error is exactly zero.
    assert ch.quadrature(lambda x: x**2, 0, 3).standard_error == 0.0
    # A Lorentzian peak of half-width 0.01 does subdivide, so the error estimate is real. The
    # C#-pinned values for this exact run are fixtures/callback/math.json's
    # quadrature_peak_subdivides; here only the qualitative properties, to keep the oracle in the
    # fixture where it belongs.
    peak = ch.quadrature(lambda x: 1.0 / (1.0 + 1.0e4 * x * x), -1, 1)
    assert peak.function_evaluations > 21
    assert 0.0 < peak.standard_error < 1e-6


def test_a_quadrature_result_survives_pickle_and_deepcopy():
    # A float subclass whose __new__ takes more than the value needs __getnewargs__ or both round
    # trips raise "__new__() missing required positional arguments". OptimResult, the other object
    # this surface returns, is picklable, so this one has to be too.
    q = ch.quadrature(lambda x: 1.0 / (1.0 + 1.0e4 * x * x), -1, 1)
    for clone in (pickle.loads(pickle.dumps(q)), copy.deepcopy(q), copy.copy(q)):
        assert isinstance(clone, ch.QuadratureResult)
        assert float(clone) == float(q)
        assert clone.status == q.status
        assert clone.function_evaluations == q.function_evaluations
        assert clone.standard_error == q.standard_error


def test_a_quadrature_result_reprs_its_report():
    # The bare float repr would hide everything but the value; R's print() shows both attributes.
    q = ch.quadrature(lambda x: x**2, 0, 3)
    text = repr(q)
    assert "status='Success'" in text
    assert f"function_evaluations={q.function_evaluations}" in text
    assert "standard_error=" in text


def test_an_unsupplied_option_leaves_the_ported_default_in_force():
    # The wrapper writes an option key only when the caller passes one, so None and the ported
    # default must give the identical run. R's twin asserts the same.
    def f(x):
        return 1.0 / (1.0 + 1.0e4 * x * x)

    default_run = ch.quadrature(f, -1, 1)
    explicit_run = ch.quadrature(
        f,
        -1,
        1,
        absolute_tolerance=1e-8,
        relative_tolerance=1e-8,
        max_function_evaluations=10000000,
    )
    assert float(default_run) == float(explicit_run)
    assert default_run.function_evaluations == explicit_run.function_evaluations
    assert ch.root_find(lambda x: x**2 - 2, lower=0, upper=2) == ch.root_find(
        lambda x: x**2 - 2, lower=0, upper=2, tolerance=1e-8, max_iterations=1000
    )
    assert ch.derivative(lambda x: x**3, 2) == ch.derivative(lambda x: x**3, 2, step_size=-1.0)


def test_quadrature_reports_the_evaluation_cap_in_its_status():
    # 1 / sqrt(x) is unbounded at 0, so the rule never converges and the cap stops it.
    q = ch.quadrature(
        lambda x: 0.0 if x <= 0 else 1.0 / math.sqrt(x), 0, 1, max_function_evaluations=1000
    )
    assert q.status == "MaximumFunctionEvaluationsReached"
    assert q.function_evaluations >= 1000


def test_quadrature_matches_the_r_verb_argument_checks():
    with pytest.raises(ValueError):
        ch.quadrature(lambda x: x, 1, 0)
    with pytest.raises(ValueError, match="between 1e-15 and 1"):
        ch.quadrature(lambda x: x, 0, 1, absolute_tolerance=0.0)
    with pytest.raises(ValueError, match="between 1e-15 and 1"):
        ch.quadrature(lambda x: x, 0, 1, relative_tolerance=2.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda f: ch.root_find(f, lower=0, upper=2),
        lambda f: ch.derivative(f, 2.0),
        lambda f: ch.gradient(f, [1.0, 1.0]),
        lambda f: ch.hessian(f, [1.0, 1.0]),
        # Quadrature is the arm where the TRAILING rethrow carries the weight: the ported
        # integrator neither throws nor converges on the guard's NaN sentinel, it just runs to
        # the evaluation cap and reports a NaN result, which would otherwise look like an answer.
        lambda f: ch.quadrature(f, 0, 3, max_function_evaluations=5000),
    ],
    ids=["root_find", "derivative", "gradient", "hessian", "quadrature"],
)
def test_an_error_inside_the_callback_reaches_the_caller(call):
    # Every method, not just one: the guard's sentinel is a value each ported routine can itself
    # reject, so an unwrapped drive site would report the internal error instead of this one.
    with pytest.raises(ValueError, match="my own error"):
        call(_boom)


def test_a_callback_returning_a_non_scalar_is_rejected():
    with pytest.raises(Exception, match="single number"):
        ch.root_find(lambda x: [1.0, 2.0], lower=0, upper=2)
    with pytest.raises(Exception, match="single number"):
        ch.gradient(lambda p: [1.0, 2.0], [1.0, 1.0])
    with pytest.raises(Exception, match="single number"):
        ch.quadrature(lambda x: [1.0, 2.0], 0, 1)


@pytest.mark.parametrize(
    "point", [[1.0, math.inf], [1.0, -math.inf], [1.0, math.nan], []], ids=["inf", "-inf", "nan", "empty"]
)
def test_a_non_finite_point_is_rejected_by_name(point):
    # The R twin (corehydror/tests/testthat/test-callback.R) asserts the identical message; R used
    # to let a non-finite point through to the spec serializer, which failed with an error naming
    # neither `x` nor the verb.
    expected = r"`x` must be a non-empty numeric vector of finite values"
    with pytest.raises(ValueError, match=expected):
        ch.gradient(lambda p: sum(v**2 for v in p), point)
    with pytest.raises(ValueError, match=expected):
        ch.hessian(lambda p: sum(v**2 for v in p), point)


def test_argument_checks_fire_before_the_core_is_reached():
    with pytest.raises(TypeError):
        ch.root_find("not a function", lower=0, upper=2)
    with pytest.raises(ValueError):
        ch.root_find(lambda x: x, lower=2, upper=0)
    with pytest.raises(ValueError):
        ch.gradient(lambda p: sum(p), [])


# --- the RNG handle -------------------------------------------------------------------------
#
# The handle cannot be constructed by a user -- it is only ever handed to a callback -- so it is
# tested through _rng_probe(), the internal verb that seeds a generator, hands a handle to `f`, and
# returns what `f` drew. The C#-pinned values for these runs live in
# fixtures/callback/rng_handle.json; here only the properties, and the lifetime rule. The R twin
# (corehydror/tests/testthat/test-callback.R) asserts the identical things.


def test_the_rng_handle_draws_from_the_core_seeded_stream_and_invalidates_after_the_call():
    captured = {}

    def f(parameters, rng):
        captured["rng"] = rng
        return rng.uniform(int(parameters[0]))

    out = _rng_probe(seed=12345, parameters=[3], f=f)
    assert len(out) == 3
    assert all(0.0 < v < 1.0 for v in out)
    assert isinstance(captured["rng"], ch.Rng)
    # The handle borrows the generator; using it after the call must raise, not read freed memory.
    with pytest.raises(RuntimeError, match="no longer valid"):
        captured["rng"].uniform(1)
    with pytest.raises(RuntimeError, match="no longer valid"):
        captured["rng"].integers(1, 0, 10)


def test_the_same_seed_gives_the_same_draws_and_a_different_seed_does_not():
    def draw(seed):
        return _rng_probe(seed, [5], lambda parameters, rng: rng.uniform(int(parameters[0])))

    assert draw(12345) == draw(12345)
    assert draw(12345) != draw(54321)


def test_rng_integers_draws_whole_numbers_on_a_half_open_range():
    k = _rng_probe(
        999,
        [50, 3, 7],
        lambda parameters, rng: rng.integers(*(int(v) for v in parameters)),
    )
    assert len(k) == 50
    # The upper bound is EXCLUDED, as in C# Next(minInclusive, maxExclusive).
    assert all(v in (3, 4, 5, 6) for v in k)
    # Uniforms and integers share one state, so interleaving them is not the same as taking them
    # separately -- the second uniform must continue after the integer, not repeat it.
    interleaved = _rng_probe(
        2024,
        [],
        lambda parameters, rng: list(rng.uniform(1))
        + [float(v) for v in rng.integers(1, 0, 1000)]
        + list(rng.uniform(1)),
    )
    straight = _rng_probe(2024, [], lambda parameters, rng: rng.uniform(2))
    assert interleaved[0] == straight[0]
    assert interleaved[2] != straight[1]


def test_a_handle_stored_from_an_earlier_call_is_dead_inside_a_later_one():
    # The misuse a garbage-collected host makes easy: keep the object, call again, reach for the
    # old one. The second call's handle is live; the first call's is not, and says so.
    kept = {}

    def first(parameters, rng):
        kept["rng"] = rng
        return rng.uniform(1)

    _rng_probe(1, [1], first)

    def second(parameters, rng):
        with pytest.raises(RuntimeError, match="no longer valid"):
            kept["rng"].uniform(1)
        return rng.uniform(1)

    assert len(_rng_probe(2, [1], second)) == 1

    # A handle captured in a closure built inside the callback is the same story: the closure
    # outlives the call, the borrow does not.
    def capture(parameters, rng):
        kept["later"] = lambda: rng.uniform(1)
        return rng.uniform(1)

    _rng_probe(3, [1], capture)
    with pytest.raises(RuntimeError, match="no longer valid"):
        kept["later"]()


def test_a_nested_call_gets_its_own_handle_and_leaves_the_outer_one_alone():
    # Each call scopes its OWN borrow, so an inner call cannot invalidate the outer handle and
    # cannot disturb the outer generator. Tasks 5 and 6 need this: a proposal that runs a nested
    # seeded verb must come back to a live handle and an undisturbed stream.
    def outer_fn(parameters, outer):
        a = list(outer.uniform(1))
        inner = _rng_probe(6, [1], lambda p2, rng2: rng2.uniform(1))
        return a + inner + list(outer.uniform(1))

    out = _rng_probe(5, [1], outer_fn)
    straight = _rng_probe(5, [1], lambda parameters, rng: rng.uniform(2))
    assert [out[0], out[2]] == straight


def test_the_rng_handle_rejects_nonsense_arguments():
    with pytest.raises(ValueError, match="positive whole number"):
        _rng_probe(1, [1], lambda parameters, rng: rng.uniform(0))
    with pytest.raises(ValueError, match="must be below"):
        _rng_probe(1, [1], lambda parameters, rng: rng.integers(2, 5, 5))
    # The class is handed out, never constructed: a user cannot make one to point anywhere.
    with pytest.raises(TypeError):
        ch.Rng()


def test_an_error_raised_inside_an_rng_callback_reaches_the_caller_unchanged():
    with pytest.raises(ValueError, match="my own error"):
        _rng_probe(1, [1], _boom)


def test_a_range_wider_than_an_int32_span_is_an_error_not_an_overflowed_draw():
    # rng.integers hands `max - min` to the ported Next(int). C# computes that subtraction
    # unchecked and throws once it wraps negative; the same expression in C++ is undefined
    # behaviour, and integers(1, -2_000_000_000, 2_000_000_000) used to return an arbitrary
    # out-of-range integer (-208904155) in both languages. The R twin
    # (corehydror/tests/testthat/test-callback.R) asserts the same boundary.
    int_max = 2147483647
    at_boundary = _rng_probe(
        11, [1], lambda parameters, rng: [float(v) for v in rng.integers(1, 0, int_max)]
    )
    assert len(at_boundary) == 1
    assert 0 <= at_boundary[0] < int_max
    with pytest.raises(ValueError, match="too wide"):
        _rng_probe(11, [1], lambda parameters, rng: rng.integers(1, -1, int_max))
    with pytest.raises(ValueError, match="too wide"):
        _rng_probe(
            11, [1], lambda parameters, rng: rng.integers(1, -2_000_000_000, 2_000_000_000)
        )


def test_a_fractional_count_is_refused_rather_than_truncated_as_r_refuses_it():
    # R would happily make 2.7 into 2 while Python raised its own overload dump, so the same call
    # meant different things in the two packages and neither message named the argument. Both now
    # refuse it by name, in the same words.
    with pytest.raises(TypeError, match="`n` must be a single positive whole number"):
        _rng_probe(1, [1], lambda parameters, rng: rng.uniform(2.7))
    with pytest.raises(TypeError, match="`n` must be a single positive whole number"):
        _rng_probe(1, [1], lambda parameters, rng: rng.integers(2.7, 0, 5))
    with pytest.raises(TypeError, match="single finite whole number"):
        _rng_probe(1, [1], lambda parameters, rng: rng.integers(1, 0.5, 5))
    # A whole number spelled as a float is still a whole number -- which matters here, because
    # every number a callback is handed arrives as one (`parameters[0]`), exactly as in R.
    drawn = _rng_probe(1, [1], lambda parameters, rng: rng.uniform(2.0))
    assert len(drawn) == 2
    assert _rng_probe(1, [3], lambda parameters, rng: rng.uniform(parameters[0])) == _rng_probe(
        1, [3], lambda parameters, rng: rng.uniform(3)
    )


def test_a_handle_leaked_out_of_a_callback_that_then_raises_is_dead():
    # The unwind path, which only the C++ suite covered. A destructor runs whether the call returns
    # or throws, so the scope invalidates the borrow either way -- and a user who stashed the handle
    # before their own error meets a message rather than freed memory. The R twin is identical.
    leaked = {}

    def leak_then_raise(parameters, rng):
        leaked["rng"] = rng
        rng.uniform(1)
        raise ValueError("my own error, raised half way through")

    with pytest.raises(ValueError, match="my own error, raised half way through"):
        _rng_probe(1, [1], leak_then_raise)

    assert "rng" in leaked
    with pytest.raises(RuntimeError, match="no longer valid"):
        leaked["rng"].uniform(1)
    with pytest.raises(RuntimeError, match="no longer valid"):
        leaked["rng"].integers(1, 0, 2)


# --- mcmc_posterior --------------------------------------------------------------------------


def _gaussian_kernel(p):
    """The Gaussian kernel the fixture catalog uses, written the same way: arithmetic only,
    summed in an explicit loop rather than through sum()."""
    data = (4.9, 5.1, 5.0, 5.2, 4.8)
    acc = 0.0
    for x in data:
        acc += (x - p[0]) * (x - p[0])
    return -0.5 * acc


_ONE_PRIOR = [ch.Distribution("Uniform", [0.0, 10.0])]


def _run(**kwargs):
    args = dict(
        iterations=300, warmup=100, chains=2, thinning=1, seed=12345, initialize="Randomize"
    )
    args.update(kwargs)
    return ch.mcmc_posterior(_gaussian_kernel, [ch.Distribution("Uniform", [0.0, 10.0])], **args)


def test_mcmc_posterior_recovers_the_mean_of_a_user_written_posterior():
    fit = _run()
    assert fit["posterior_mean"][0] == pytest.approx(5.0, abs=0.2)
    assert fit["map"][0] == pytest.approx(5.0, abs=0.2)
    # The full mcmc_sample() shape, so a user can move between the two functions.
    assert list(fit) == list(ch.mcmc_sample([4.9, 5.1, 5.0, 5.2, 4.8], "Normal", iterations=200))
    assert fit["parameters"] == ["p1"]
    assert len(fit["chains"]) == 2
    assert fit["chains"][0].shape == (300, 1)
    assert len(fit["acceptance_rates"]) == 2


def test_two_seeded_mcmc_posterior_runs_are_identical():
    a, b = _run(), _run()
    assert (a["chains"][0] == b["chains"][0]).all()
    assert a["map_fitness"] == b["map_fitness"]
    # A different seed is a different chain, so the equality above is not vacuous.
    assert not (_run(seed=999)["chains"][0] == a["chains"][0]).all()


def test_a_two_parameter_model_reads_its_priors_in_order():
    # The prior list's length IS the parameter count, and the order is the order the
    # log-likelihood indexes. A straight line through eight points, intercept then slope.
    def ll(p):
        t = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        y = (2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1)
        acc = 0.0
        for i in range(len(t)):
            residual = y[i] - p[0] - p[1] * t[i]
            acc += residual * residual
        return -0.5 * acc

    fit = ch.mcmc_posterior(
        ll,
        [ch.Distribution("Uniform", [-5.0, 5.0]), ch.Distribution("Uniform", [0.0, 5.0])],
        sampler="DEMCz", iterations=300, warmup=100, chains=3, thinning=1, seed=12345,
        initialize="Randomize",
    )
    assert fit["parameters"] == ["p1", "p2"]
    assert fit["map"][0] == pytest.approx(0.1, abs=0.5)
    assert fit["map"][1] == pytest.approx(2.0, abs=0.2)


# The SAME seven-sampler vector mcmc_posterior()'s own `sampler` argument accepts (Gibbs
# excluded -- it is refused before it ever reaches a sampler, see
# test_mcmc_posterior_refuses_a_prior_list_that_is_not_one's "unknown sampler" case for the
# neighboring check). A guard wired for one sampler's arm and silently missing on another is
# exactly the shape of bug a previous phase shipped, and a single-sampler test cannot catch it.
_ALL_SAMPLERS = [
    ("RWMH", 2, 100, 50),
    ("ARWMH", 2, 100, 50),
    ("DEMCz", 3, 100, 50),
    ("DEMCzs", 3, 100, 50),
    ("HMC", 2, 100, 50),
    ("NUTS", 2, 100, 50),
    # SNIS is not a Markov chain: its own ported validation rejects any warm-up and requires
    # iterations >= output_length, which mcmc_posterior() does not expose (fixed at the ported
    # default, 10000) -- so this is the smallest legal call, not an arbitrary round number.
    ("SNIS", 1, 10000, None),
]


@pytest.mark.parametrize("initialize", ["Randomize", "MAP"])
@pytest.mark.parametrize("sampler,chains,iterations,warmup", _ALL_SAMPLERS)
def test_an_error_raised_inside_the_log_likelihood_reaches_the_caller(
    sampler, chains, iterations, warmup, initialize
):
    # On BOTH initialization paths, and they fail differently underneath: "Randomize" never raises
    # of its own accord (-infinity is a legal, always rejected fitness, so the chain runs to
    # completion), while "MAP" hands DifferentialEvolution nothing but -infinity and can throw
    # from its own internals first. Only the guard makes the user's own message win in both.
    def boom(p):
        raise ValueError("my own error")

    kwargs = dict(
        sampler=sampler, iterations=iterations, chains=chains, thinning=1, seed=12345,
        initialize=initialize,
    )
    if warmup is not None:
        kwargs["warmup"] = warmup
    with pytest.raises(ValueError, match="my own error"):
        ch.mcmc_posterior(boom, _ONE_PRIOR, **kwargs)


def test_mcmc_posterior_refuses_a_prior_list_that_is_not_one():
    with pytest.raises(TypeError, match="non-empty sequence of Distribution objects"):
        ch.mcmc_posterior(_gaussian_kernel, [])
    with pytest.raises(TypeError, match="non-empty sequence of Distribution objects"):
        ch.mcmc_posterior(_gaussian_kernel, "Uniform")
    with pytest.raises(TypeError, match="non-empty sequence of Distribution objects"):
        ch.mcmc_posterior(_gaussian_kernel, [ch.Distribution("Uniform", [0.0, 10.0]), 3])
    with pytest.raises(TypeError, match="`log_likelihood` must be a function"):
        ch.mcmc_posterior("not a function", _ONE_PRIOR)
    with pytest.raises(ValueError, match="unknown sampler"):
        ch.mcmc_posterior(_gaussian_kernel, _ONE_PRIOR, sampler="Gibbs")
    # A None seed used to reach int(None) and fail deep inside the JSON builder with a message
    # that never names the argument; it is now refused by name before that point.
    with pytest.raises(TypeError, match="`seed` must not be None"):
        ch.mcmc_posterior(_gaussian_kernel, _ONE_PRIOR, seed=None)
    # A single Distribution is accepted for a one-parameter model.
    fit = ch.mcmc_posterior(
        _gaussian_kernel, ch.Distribution("Uniform", [0.0, 10.0]), iterations=100, warmup=50,
        chains=2, thinning=1, seed=12345, initialize="Randomize",
    )
    assert fit["parameters"] == ["p1"]


def test_a_log_likelihood_indexing_past_its_priors_fails_by_name():
    # The likeliest user error on this surface: two parameters written, one prior given. Python
    # raises IndexError of its own accord, and the guard has to carry it out intact.
    with pytest.raises(IndexError):
        ch.mcmc_posterior(
            lambda p: -0.5 * (p[0] * p[0] + p[1] * p[1]), _ONE_PRIOR, iterations=100, warmup=50,
            chains=2, thinning=1, seed=12345, initialize="Randomize",
        )


def test_snis_is_refused_the_settings_its_own_ported_validation_refuses():
    with pytest.raises(ValueError, match="supports `chains=1` only"):
        ch.mcmc_posterior(_gaussian_kernel, _ONE_PRIOR, sampler="SNIS", chains=4)
    # And it runs when left alone: no auto-derived warm-up, because its ValidateSettings rejects
    # any. Its other ported rule is `Iterations >= OutputLength`, and OutputLength is not a
    # setting this surface exposes, so 10,000 is the smallest iteration count SNIS accepts here.
    fit = ch.mcmc_posterior(
        _gaussian_kernel, _ONE_PRIOR, sampler="SNIS", iterations=10000, seed=12345,
        initialize="Randomize",
    )
    assert fit["posterior_mean"][0] == pytest.approx(5.0, abs=0.5)
