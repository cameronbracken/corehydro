import numpy as np
import pytest

from corehydropy import Constraint, optim_maximize, optim_minimize


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


# --- the two gradient-taking methods ------------------------------------------------------
#
# f(p) = (p1 - 3)^2 + (p2 + 1)^2, minimum 0 at (3, -1), with the analytic gradient written out
# term by term so the R twin in corehydror/tests/testthat/test-optim.R evaluates the identical
# arithmetic.
def _quad_shifted(p):
    return (p[0] - 3) ** 2 + (p[1] + 1) ** 2


def _quad_shifted_gradient(p):
    return [2 * (p[0] - 3), 2 * (p[1] + 1)]


@pytest.mark.parametrize("method", ["gradient_descent", "adam"])
def test_adam_and_gradient_descent_take_an_analytic_gradient(method):
    fit = optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                         method=method, gradient=_quad_shifted_gradient,
                         control={"alpha": 0.1})
    assert fit.parameters == pytest.approx([3.0, -1.0], abs=1e-3)


@pytest.mark.parametrize("method", ["gradient_descent", "adam"])
def test_adam_and_gradient_descent_fall_back_to_numerical_differentiation(method):
    # An omitted `gradient` is the ported classes' null Gradient, which routes through
    # NumericalDerivative.Gradient exactly as C# does -- so the same run lands in the same place.
    fit = optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                         method=method, control={"alpha": 0.1})
    assert fit.parameters == pytest.approx([3.0, -1.0], abs=1e-3)


def test_the_analytic_gradient_is_actually_used():
    # A run driven by the supplied gradient never pays for the 2*D finite-difference probes, so it
    # costs strictly fewer objective evaluations than the same run without one. Without this, a
    # runner arm that dropped the gradient on the floor would pass every assertion above.
    with_g = optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                            method="gradient_descent", gradient=_quad_shifted_gradient,
                            control={"alpha": 0.1})
    without_g = optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                               method="gradient_descent", control={"alpha": 0.1})
    assert with_g.function_evaluations < without_g.function_evaluations


@pytest.mark.parametrize("method", ["gradient_descent", "adam"])
def test_an_error_inside_the_gradient_reaches_the_caller(method):
    def boom(p):
        raise RuntimeError("boom in the gradient")

    with pytest.raises(RuntimeError, match="boom in the gradient"):
        optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                       method=method, gradient=boom)


def test_a_gradient_returning_the_wrong_length_is_rejected():
    with pytest.raises(Exception, match="one value per parameter"):
        optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                       method="adam", gradient=lambda p: [1.0])


@pytest.mark.parametrize("method,initial,seed", [("de", None, 1), ("bfgs", [0, 0], None)])
def test_gradient_is_rejected_for_methods_that_cannot_take_one(method, initial, seed):
    with pytest.raises(ValueError, match="gradient"):
        optim_minimize(_quad_shifted, initial=initial, lower=[-10, -10], upper=[10, 10],
                       method=method, seed=seed, gradient=_quad_shifted_gradient)


def test_gradient_must_be_callable():
    with pytest.raises(TypeError, match="callable"):
        optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                       method="adam", gradient=1)


def test_alpha_beta1_and_beta2_reach_the_right_classes():
    # alpha is shared; beta1/beta2 are ADAM's alone, so gradient_descent must reject them by name.
    fast = optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                          method="gradient_descent", gradient=_quad_shifted_gradient,
                          control={"alpha": 0.1})
    slow = optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                          method="gradient_descent", gradient=_quad_shifted_gradient,
                          control={"alpha": 0.01})
    assert fast.iterations < slow.iterations
    with pytest.raises(ValueError, match="beta1"):
        optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                       method="gradient_descent", control={"beta1": 0.5})
    fit = optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                         method="adam", gradient=_quad_shifted_gradient,
                         control={"alpha": 0.1, "beta1": 0.8, "beta2": 0.9})
    assert fit.status


@pytest.mark.parametrize("method", ["adam", "gradient_descent"])
def test_seed_is_rejected_for_the_two_gradient_methods(method):
    with pytest.raises(ValueError, match="stochastic"):
        optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                       method=method, seed=1)


# --- the constrained surface (augmented Lagrange) -------------------------------------------
#
# Test_Haimes_5_2 from Test_AugmentedLagrange.cs, driven end to end through the public Python
# surface: the objective and the constraint are both Python callables, so this exercises the two
# host-language callbacks the arm guards, not just the C++ classes
# core/tests/test_augmented_lagrange.cpp already covers. Every expected value below is that C#
# test's own literal, at its own tolerance.
def _haimes_primary(p):
    return (p[0] - 2) ** 2 + (p[1] - 4) ** 2 + 5


def _haimes_secondary(p):
    return (p[0] - 6) ** 2 + (p[1] - 10) ** 2 + 6


def test_augmented_lagrange_reproduces_haimes_5_2():
    fit = optim_minimize(
        _haimes_primary, initial=[5, 5], lower=[0, 0], upper=[10, 10],
        method="augmented_lagrange",
        constraints=[Constraint(_haimes_secondary, value=13.31, type="le")],
    )
    assert fit.parameters == pytest.approx([4.5, 7.75], abs=1e-2)
    assert fit.value == pytest.approx(25.31, abs=1e-2)
    assert fit.multipliers["less_than"][0] == pytest.approx(1.67, abs=1e-2)
    # The other two multiplier sets exist and are empty: this problem has no equality or
    # greater-than constraint, and the three vectors are sized by COUNTING each type.
    assert len(fit.multipliers["equality"]) == 0
    assert len(fit.multipliers["greater_than"]) == 0


def test_an_explicit_inner_spec_drives_the_same_problem():
    fit = optim_minimize(
        _haimes_primary, initial=[5, 5], lower=[0, 0], upper=[10, 10],
        method="augmented_lagrange",
        constraints=[Constraint(_haimes_secondary, value=13.31, type="le")],
        inner={"method": "bfgs", "initial": [5, 5], "lower": [0, 0], "upper": [10, 10]},
    )
    assert fit.parameters == pytest.approx([4.5, 7.75], abs=1e-2)


def test_all_three_constraint_types_index_their_own_multiplier_vector():
    # Test_MixedConstraints: minimize x^2 + y^2 subject to x + y = 4, x <= 3, y >= 0.5.
    fit = optim_minimize(
        lambda p: p[0] ** 2 + p[1] ** 2, initial=[1, 3], lower=[-10, -10], upper=[10, 10],
        method="augmented_lagrange",
        constraints=[
            Constraint(lambda p: p[0] + p[1], value=4.0, type="eq"),
            Constraint(lambda p: p[0], value=3.0, type="le"),
            Constraint(lambda p: p[1], value=0.5, type="ge"),
        ],
    )
    assert fit.parameters == pytest.approx([2.0, 2.0], abs=0.1)
    assert fit.value == pytest.approx(8.0, abs=0.5)
    assert len(fit.multipliers["equality"]) == 1
    assert len(fit.multipliers["less_than"]) == 1
    assert len(fit.multipliers["greater_than"]) == 1


def test_constraints_and_inner_are_rejected_for_every_other_method():
    con = [Constraint(_haimes_secondary, value=13.31, type="le")]
    with pytest.raises(ValueError, match="augmented_lagrange"):
        optim_minimize(_haimes_primary, lower=[0, 0], upper=[10, 10], method="de", seed=1,
                       constraints=con)
    with pytest.raises(ValueError, match="augmented_lagrange"):
        optim_minimize(_haimes_primary, initial=[5, 5], lower=[0, 0], upper=[10, 10],
                       method="bfgs", inner={"method": "bfgs"})


@pytest.mark.parametrize("constraints", [None, []])
def test_augmented_lagrange_requires_at_least_one_constraint(constraints):
    with pytest.raises(ValueError, match="constraints"):
        optim_minimize(_haimes_primary, initial=[5, 5], lower=[0, 0], upper=[10, 10],
                       method="augmented_lagrange", constraints=constraints)


# optim_maximize() used to accept this method and return the constrained MINIMUM labelled
# "Success": upstream AugmentedLagrange.Optimize() always calls the inner optimizer's Minimize()
# over an augmented Lagrangian built from the RAW objective, so the outer sign flip never reaches
# the search. The port still mirrors that; the public verb refuses the request.
def test_optim_maximize_rejects_augmented_lagrange():
    def peak(p):                                    # true constrained max at x = 1: -4
        return -((p[0] - 3) ** 2)

    con = [Constraint(lambda p: p[0], value=1.0, type="le")]
    with pytest.raises(ValueError, match="cannot maximize"):
        optim_maximize(peak, initial=[0.0], lower=[-10.0], upper=[10.0],
                       method="augmented_lagrange", constraints=con)
    # Every other method still maximizes.
    assert optim_maximize(peak, lower=[-10.0], upper=[10.0], method="de",
                          seed=1).status == "Success"
    # The workaround the error names: minimize -f under the same constraint.
    fit = optim_minimize(lambda p: (p[0] - 3) ** 2, initial=[0.0], lower=[-10.0], upper=[10.0],
                         method="augmented_lagrange", constraints=con)
    assert fit.parameters[0] == pytest.approx(1.0, abs=1e-3)
    assert -fit.value == pytest.approx(-4.0, abs=1e-3)


def test_constraint_validates_its_own_arguments():
    with pytest.raises(TypeError, match="callable"):
        Constraint(1, value=1.0)
    with pytest.raises(ValueError, match="type"):
        Constraint(_haimes_secondary, value=1.0, type="lt")


@pytest.mark.parametrize("method", ["nelder_mead", "brent", "augmented_lagrange"])
def test_an_inner_method_that_cannot_be_an_inner_optimizer_is_rejected(method):
    con = [Constraint(_haimes_secondary, value=13.31, type="le")]
    with pytest.raises(Exception, match=method):
        optim_minimize(_haimes_primary, initial=[5, 5], lower=[0, 0], upper=[10, 10],
                       method="augmented_lagrange", constraints=con,
                       inner={"method": method, "initial": [5, 5], "lower": [0, 0],
                              "upper": [10, 10]})


def test_an_error_inside_a_constraint_reaches_the_caller():
    # The constraint is the third host-language callback the optimizer surface takes, guarded
    # through the SAME abort state as the objective (see optimizer_runner.hpp) -- so a Python
    # exception raised inside it must survive AugmentedLagrange's inner optimizer's own catch-all.
    def boom(p):
        raise RuntimeError("boom in the constraint")

    with pytest.raises(RuntimeError, match="boom in the constraint"):
        optim_minimize(_haimes_primary, initial=[5, 5], lower=[0, 0], upper=[10, 10],
                       method="augmented_lagrange",
                       constraints=[Constraint(boom, value=1.0, type="le")])


def test_multipliers_is_none_for_every_unconstrained_method():
    fit = optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                         method="bfgs")
    assert fit.multipliers is None


def test_repr_shows_only_the_multiplier_sets_the_problem_has():
    fit = optim_minimize(
        _haimes_primary, initial=[5, 5], lower=[0, 0], upper=[10, 10],
        method="augmented_lagrange",
        constraints=[Constraint(_haimes_secondary, value=13.31, type="le")],
    )
    out = repr(fit)
    assert "less than multipliers" in out
    assert "equality multipliers" not in out
    assert "greater than multipliers" not in out
    # An unconstrained result has no multiplier line at all.
    plain = optim_minimize(_quad_shifted, initial=[0, 0], lower=[-10, -10], upper=[10, 10],
                           method="bfgs")
    assert "multipliers" not in repr(plain)
