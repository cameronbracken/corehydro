import numpy as np
import pytest

from corehydropy import optim_maximize, optim_minimize


def test_error_inside_objective_reaches_the_caller_intact():
    def f(p):
        raise ValueError("boom in the objective")

    with pytest.raises(ValueError, match="boom in the objective"):
        optim_minimize(f, lower=[-1, -1], upper=[1, 1], seed=1)


def test_seeded_de_run_reproduces_exactly():
    def f(p):
        return float(np.sum(np.asarray(p) ** 2))

    a = optim_minimize(f, lower=[-5, -5], upper=[5, 5], seed=99)
    b = optim_minimize(f, lower=[-5, -5], upper=[5, 5], seed=99)
    assert np.array_equal(a.parameters, b.parameters)
    assert a.value == b.value


def test_objective_returning_the_wrong_shape_is_rejected_by_message():
    with pytest.raises(Exception, match="single number"):
        optim_minimize(lambda p: [1, 2], lower=[-1, -1], upper=[1, 1])
    with pytest.raises(Exception):
        optim_minimize(lambda p: "nope", lower=[-1, -1], upper=[1, 1])


def test_optim_maximize_finds_the_peak_of_a_concave_objective():
    def f(p):
        return -((p[0] - 2) ** 2 + (p[1] + 1) ** 2)

    fit = optim_maximize(f, lower=[-10, -10], upper=[10, 10], seed=7)
    np.testing.assert_allclose(fit.parameters, [2, -1], atol=1e-4)


def test_method_that_needs_bounds_says_so_when_missing():
    with pytest.raises(ValueError, match="lower"):
        optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), method="de")


def test_method_that_needs_an_initial_guess_says_so_when_missing():
    with pytest.raises(ValueError, match="initial"):
        optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), lower=[-1, -1], upper=[1, 1],
                       method="bfgs")


def test_argument_shape_and_control_validation():
    with pytest.raises(TypeError):
        optim_minimize(1, lower=-1, upper=1)
    with pytest.raises(ValueError, match="same length"):
        optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), lower=[-1, -1], upper=-1)
    with pytest.raises(ValueError, match="below"):
        optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), lower=1, upper=-1)
    with pytest.raises(ValueError, match="same length"):
        optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), lower=-1, upper=1,
                       initial=[0, 0])
    with pytest.raises(ValueError, match="stochastic"):
        optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), lower=-1, upper=1, initial=0,
                       method="bfgs", seed=1)
    with pytest.raises(ValueError, match="unknown control"):
        optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), lower=-1, upper=1,
                       control={"bogus": 1})
    with pytest.raises(ValueError, match="only apply to method"):
        optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), lower=[-1, -1], upper=[1, 1],
                       initial=[0, 0], method="nelder_mead", control={"report_failure": False})


@pytest.mark.parametrize("method", ["bfgs", "powell", "mlsl", "nelder_mead"])
def test_every_method_converges_on_a_simple_quadratic(method):
    target = np.array([1.0, 2.0])

    def f(p):
        return float(np.sum((np.asarray(p) - target) ** 2))

    fit = optim_minimize(f, initial=[0, 0], lower=[-10, -10], upper=[10, 10], method=method)
    np.testing.assert_allclose(fit.parameters, target, atol=1e-3)
    assert fit.value < 1e-4


def test_brent_converges_on_a_simple_quadratic():
    fit = optim_minimize(lambda p: (p[0] - 3) ** 2, lower=[-10], upper=[10], method="brent")
    np.testing.assert_allclose(fit.parameters, [3], atol=1e-4)


def test_optimresult_repr_reports_status_value_and_parameters():
    fit = optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), lower=[-1, -1], upper=[1, 1],
                         seed=1)
    text = repr(fit)
    assert "OptimResult" in text
    assert "value:" in text
    assert "parameters:" in text


def test_compute_hessian_returns_a_symmetric_matrix_for_optimizer_base_methods():
    target = np.array([1.0, 2.0])

    def f(p):
        return float(np.sum((np.asarray(p) - target) ** 2))

    fit = optim_minimize(f, lower=[-5, -5], upper=[5, 5], seed=3, control={"compute_hessian": True})
    assert fit.hessian is not None
    assert fit.hessian.shape == (2, 2)


def test_nelder_mead_and_brent_never_carry_a_hessian_even_when_requested():
    fit = optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), initial=[0, 0],
                         lower=[-1, -1], upper=[1, 1], method="nelder_mead")
    assert fit.hessian is None
    fitb = optim_minimize(lambda p: p[0] ** 2, lower=[-1], upper=[1], method="brent")
    assert fitb.hessian is None
