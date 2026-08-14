import copy
import math
import pickle

import numpy as np
import pytest

import corehydropy as ch


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
