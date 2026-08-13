# Behavioural tests for the copula surface, the Python twin of corehydror's
# tests/testthat/test-copula.R -- same cases, same order, same names where Python allows. Oracle
# VALUES live in fixtures/ and are asserted by test_fixtures.py; this file asserts argument
# handling, error messages, and object shape.

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from corehydropy import Copula, Distribution, copula_fit, copula_names

X = [135.9, 104.1, 108.7, 99.3, 134.7, 91.0, 77.3, 115.4, 109.0, 79.0]
Y = [1.9, 1.3, 1.4, 1.2, 1.8, 1.1, 0.9, 1.5, 1.4, 1.0]


def test_copula_builds_and_evaluates():
    cop = Copula("Clayton", theta=2)
    assert cop.family == "Clayton"
    assert math.isfinite(cop.pdf(0.3, 0.7))
    assert cop.cdf(0.3, 0.7) > 0
    assert np.array_equal(cop.params(), [2.0])


def test_tail_dependence_comes_back_named():
    td = Copula("Clayton", theta=2).tail_dependence()
    assert set(td) == {"lower", "upper"}
    assert td["lower"] == pytest.approx(2 ** (-1 / 2), abs=1e-12)


def test_the_two_exceedance_probabilities_differ_and_both_lie_in_the_unit_interval():
    cop = Copula("Gumbel", theta=2)
    a = cop.exceedance(0.9, 0.9, type="and")
    o = cop.exceedance(0.9, 0.9, type="or")
    assert 0 <= a <= 1 and 0 <= o <= 1
    assert a != pytest.approx(o)


def test_bounds_reports_the_theta_range():
    b = Copula("Clayton", theta=2).bounds()
    assert set(b) == {"minimum", "maximum"}


def test_a_seeded_copula_draw_is_reproducible_and_has_the_right_shape():
    cop = Copula(
        "Clayton", theta=2,
        margin_x=Distribution("Normal", [0, 1]), margin_y=Distribution("Normal", [0, 1]),
    )
    d = cop.random(5, seed=12345)
    assert d.shape == (5, 2)
    assert np.array_equal(d, cop.random(5, seed=12345))


def test_copula_fit_returns_a_usable_copula():
    cop = copula_fit("Clayton", X, Y, method="mpl")
    assert math.isfinite(cop.theta)
    assert math.isfinite(cop.pdf(0.3, 0.7))


def _rank(values):
    """1-based rank, ties (exact equality) averaged over their run -- mirrors the core's
    ranks_in_place (Statistics.RanksInPlace), which is what plotting_positions() calls. `y` has
    a genuine tie (two 1.4s), so a naive argsort-based rank (no tie averaging) disagrees with the
    core here and breaks the idempotence identity below."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    ranks = np.empty(len(arr))
    n = len(arr)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def test_the_mpl_fit_takes_raw_data_and_ranks_it_internally():
    # The pseudo-likelihood is a function of the ranks alone, so fitting the raw sample and
    # fitting its plotting positions must land on the same theta -- that identity is what the
    # in-core transform buys, and it needs no oracle value.
    raw = copula_fit("Clayton", X, Y, method="mpl").theta
    pp = copula_fit(
        "Clayton", _rank(X) / (len(X) + 1), _rank(Y) / (len(Y) + 1), method="mpl",
    ).theta
    assert raw == pytest.approx(pp)
    # A copula that ignored the ranking would sit at an arbitrary interior point of the theta
    # bounds because every log_pdf evaluation lands off the unit square. A real Clayton fit to
    # this strongly concordant sample is well above the weak-dependence range.
    assert raw > 1
    assert math.isfinite(Copula("Clayton", theta=raw).log_likelihood(X, Y))


def test_a_named_marginal_is_refused_for_the_methods_that_never_fit_one():
    with pytest.raises(ValueError, match="does not use"):
        copula_fit("Clayton", X, Y, method="mpl", margin_x="Normal")
    with pytest.raises(ValueError, match="does not use"):
        copula_fit("Clayton", X, Y, method="tau", margin_y="Normal")
    # A parameterized Distribution stays allowed: it is attached as given, not fitted.
    cop = copula_fit(
        "Clayton", X, Y, method="mpl",
        margin_x=Distribution("Normal", [110, 20]), margin_y=Distribution("Normal", [1.4, 0.3]),
    )
    assert cop.margin_x.params == [110.0, 20.0]
    assert cop.random(3, seed=1).shape == (3, 2)


def test_the_tau_method_is_refused_where_upstream_has_no_setthetafromtau():
    with pytest.raises(Exception, match="tau"):
        copula_fit("Joe", X, Y, method="tau")


def test_the_three_log_likelihoods_run_and_differ():
    cop = Copula(
        "Clayton", theta=2,
        margin_x=Distribution("Normal", [110, 20]), margin_y=Distribution("Normal", [1.4, 0.3]),
    )
    ll_ifm = cop.log_likelihood(X, Y, method="ifm")
    ll_ps = cop.log_likelihood(X, Y, method="pseudo")
    assert math.isfinite(ll_ifm) and math.isfinite(ll_ps)
    assert ll_ifm != pytest.approx(ll_ps)


def test_the_density_verbs_evaluate_a_whole_vector_of_pairs():
    cop = Copula("Clayton", theta=2)
    u = [0.3, 0.5, 0.8]
    v = [0.7, 0.9, 0.2]
    np.testing.assert_array_equal(
        cop.pdf(u, v), np.array([cop.pdf(u[i], v[i]) for i in range(len(u))])
    )
    np.testing.assert_array_equal(
        cop.cdf(u, v), np.array([cop.cdf(u[i], v[i]) for i in range(len(u))])
    )
    np.testing.assert_array_equal(cop.log_pdf(u, v), np.log(cop.pdf(u, v)))
    # A scalar recycles against a vector, one value per pair.
    np.testing.assert_array_equal(cop.pdf(0.3, v), np.array([cop.pdf(0.3, vi) for vi in v]))
    assert cop.pdf(u, 0.5).shape == (3,)
    with pytest.raises(ValueError, match="recyclable"):
        cop.pdf([0.2, 0.4, 0.6], [0.1, 0.9])
    with pytest.raises(ValueError, match="non-empty"):
        cop.pdf([], 0.5)


def test_copula_names_lists_the_seven_families():
    assert len(copula_names()) == 7
    assert "StudentT" in copula_names()


def test_a_copula_round_trips_through_pickle():
    cop = Copula("Frank", theta=3)
    restored = pickle.loads(pickle.dumps(cop))
    assert restored.cdf(0.4, 0.6) == cop.cdf(0.4, 0.6)
