import numpy as np
import pytest

from corehydropy import optim_maximize, optim_minimize


_NEEDS_INITIAL = ("bfgs", "powell", "mlsl", "multi_start", "nelder_mead")
_STOCHASTIC = ("de", "particle_swarm", "sce", "simulated_annealing", "multi_start", "mlsl")
_ALL_METHODS = ("de", "particle_swarm", "sce", "simulated_annealing", "multi_start", "mlsl",
                "bfgs", "powell", "nelder_mead", "brent", "golden_section")


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_error_inside_objective_reaches_the_caller_intact(method):
    # The guard (GuardedObjective + optimizer_runner.hpp's per-method rethrow) is the whole point
    # of this task, so this case is parametrized over ALL ELEVEN methods -- a hole in just one of
    # them (bfgs/mlsl were bypassed by an internal gradient-probe exception before the fix) would
    # not have shown up if only "de" were exercised here.
    def f(p):
        raise ValueError("boom in the objective")

    kwargs = {"lower": [-1, -1], "upper": [1, 1], "method": method}
    if method in ("brent", "golden_section"):
        kwargs["lower"], kwargs["upper"] = [-1], [1]
    if method in _NEEDS_INITIAL:
        kwargs["initial"] = [0, 0]
    if method in _STOCHASTIC:
        kwargs["seed"] = 1
    with pytest.raises(ValueError, match="boom in the objective"):
        optim_minimize(f, **kwargs)


@pytest.mark.parametrize("method", ["de", "particle_swarm", "sce", "simulated_annealing",
                                    "multi_start", "mlsl", "bfgs", "powell", "golden_section"])
def test_guard_survives_report_failure_false_for_every_base_method(method):
    # `control={"report_failure": False}` changes whether the ported Optimizer base itself
    # rethrows internally, but the guard must still surface the ORIGINAL objective exception
    # either way (only "nelder_mead"/"brent" reject a report_failure control, since they don't
    # derive from the Optimizer base).
    def f(p):
        raise ValueError("boom in the objective")

    kwargs = {"lower": [-1, -1], "upper": [1, 1], "method": method,
              "control": {"report_failure": False}}
    if method in _NEEDS_INITIAL:
        kwargs["initial"] = [0, 0]
    if method in _STOCHASTIC:
        kwargs["seed"] = 1
    with pytest.raises(ValueError, match="boom in the objective"):
        optim_minimize(f, **kwargs)


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


def test_population_size_is_rejected_for_every_method_except_de():
    with pytest.raises(ValueError, match="population_size"):
        optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), lower=[-1, -1], upper=[1, 1],
                       initial=[0, 0], method="nelder_mead", control={"population_size": 20})
    # "de" itself accepts it -- confirm no error (Task 8 finding 7).
    fit = optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), lower=[-1, -1], upper=[1, 1],
                         seed=1, control={"population_size": 12})
    assert fit.status


def test_initial_is_rejected_for_methods_that_never_read_it():
    with pytest.raises(ValueError, match="initial"):
        optim_minimize(lambda p: float(np.sum(np.asarray(p) ** 2)), lower=[-1, -1], upper=[1, 1],
                       initial=[0, 0], method="de", seed=1)
    with pytest.raises(ValueError, match="initial"):
        optim_minimize(lambda p: p[0] ** 2, lower=[-1], upper=[1], initial=[0], method="brent")


def test_integer_valued_objective_return_is_accepted():
    fit = optim_minimize(lambda p: int(np.sum(np.asarray(p) > 0)), lower=[-1, -1], upper=[1, 1],
                         seed=1)
    assert fit.status


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


def _booth(p):
    # The Booth function, minimum 0 at (1, 3). Written out rather than vectorized so the R twin
    # in corehydror/tests/testthat/test-optim.R evaluates the identical arithmetic.
    return (p[0] + 2.0 * p[1] - 7.0) ** 2 + (2.0 * p[0] + p[1] - 5.0) ** 2


@pytest.mark.parametrize("method", ["particle_swarm", "sce", "simulated_annealing", "multi_start"])
def test_the_new_global_methods_find_the_booth_optimum(method):
    kwargs = {"lower": [-10, -10], "upper": [10, 10], "method": method, "seed": 12345}
    if method == "multi_start":
        kwargs["initial"] = [0, 0]
    fit = optim_minimize(_booth, **kwargs)
    np.testing.assert_allclose(fit.parameters, [1.0, 3.0], atol=1e-3)


def test_golden_section_finds_a_one_dimensional_minimum():
    fit = optim_minimize(lambda p: (p[0] - 2.0) ** 2, lower=[0], upper=[5],
                         method="golden_section")
    np.testing.assert_allclose(fit.parameters, [2.0], atol=1e-4)


def test_multi_start_accepts_a_local_method_and_rejects_an_unknown_one():
    fit = optim_minimize(_booth, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                         method="multi_start", seed=12345, control={"local_method": "powell"})
    np.testing.assert_allclose(fit.parameters, [1.0, 3.0], atol=1e-3)
    with pytest.raises(ValueError, match="local_method"):
        optim_minimize(_booth, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                       method="multi_start", seed=1, control={"local_method": "adam"})


def test_multi_start_max_iterations_is_not_overwritten_by_its_constructor():
    # MultiStart sets MaxIterations to 100 in its CONSTRUCTOR rather than as a field default, so a
    # runner arm that applied the controls before construction would silently ignore this.
    fit = optim_minimize(_booth, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                         method="multi_start", seed=12345, control={"max_iterations": 15})
    assert fit.iterations == 15


def test_a_control_reaches_each_new_class():
    # One control value per new class actually changing the run -- the counterpart of the "de"
    # max_function_evaluations check below, which is the only other test here that proves a
    # control is read at all.
    capped = optim_minimize(_booth, lower=[-10, -10], upper=[10, 10],
                            method="simulated_annealing", seed=12345,
                            control={"max_iterations": 12})
    assert capped.function_evaluations < 5000  # a default 10,000-iteration run costs ~800,000

    small = optim_minimize(_booth, lower=[-10, -10], upper=[10, 10], method="particle_swarm",
                           seed=12345, control={"population_size": 10})
    default = optim_minimize(_booth, lower=[-10, -10], upper=[10, 10], method="particle_swarm",
                             seed=12345)
    assert small.function_evaluations != default.function_evaluations

    tight = optim_minimize(_booth, lower=[-10, -10], upper=[10, 10], method="sce", seed=12345,
                           control={"complexes": 2})
    loose = optim_minimize(_booth, lower=[-10, -10], upper=[10, 10], method="sce", seed=12345)
    assert tight.function_evaluations != loose.function_evaluations

    # local_method is the one control shared by two methods, and both of its non-default values
    # have to reach the class: a run polished by Powell costs a different number of evaluations
    # than the same run polished by BFGS.
    bfgs = optim_minimize(_booth, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                          method="multi_start", seed=12345, control={"local_method": "bfgs"})
    powell = optim_minimize(_booth, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                            method="multi_start", seed=12345, control={"local_method": "powell"})
    assert bfgs.function_evaluations != powell.function_evaluations


def test_a_control_belonging_to_another_method_names_the_method():
    with pytest.raises(ValueError, match="got method 'de'"):
        optim_minimize(_booth, lower=[-10, -10], upper=[10, 10], method="de", seed=1,
                       control={"cooling_rate": 0.9})


def test_max_function_evaluations_caps_the_reported_count():
    # A control value actually reaching the optimizer -- guards against a transposed assignment in
    # apply_common_controls/apply_optimizer_controls going unnoticed (Task 8 finding 10).
    def f(p):
        return float(np.sum(np.asarray(p) ** 2))

    capped = optim_minimize(f, lower=[-5, -5], upper=[5, 5], seed=42,
                            control={"max_function_evaluations": 40})
    uncapped = optim_minimize(f, lower=[-5, -5], upper=[5, 5], seed=42)
    assert capped.function_evaluations <= 45
    assert capped.function_evaluations < uncapped.function_evaluations
