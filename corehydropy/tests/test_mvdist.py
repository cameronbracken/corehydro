# Behavioural tests for the multivariate surface, the Python twin of corehydror's
# tests/testthat/test-mvdist.R -- same cases, same order, same names where Python allows. Oracle
# VALUES live in fixtures/ and are asserted by test_fixtures.py; this file asserts shape,
# argument handling, and error messages.

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from corehydropy import (
    MultivariateDistribution,
    mvdist_dirichlet,
    mvdist_multinomial,
    mvdist_normal,
    mvdist_student_t,
)

S = np.eye(3)


def test_mvdist_normal_builds_and_evaluates():
    mv = mvdist_normal([1, 2, 3], S)
    assert isinstance(mv, MultivariateDistribution)
    assert mv.dimension() == 3
    np.testing.assert_array_equal(mv.mean(), [1, 2, 3])
    assert mv.covariance().shape == (3, 3)
    assert math.isfinite(mv.pdf([1, 2, 3]))


def test_marginal_takes_1_based_indices_and_returns_a_new_object():
    mv = mvdist_normal([1, 2, 3], S)
    m = mv.marginal([1, 3])
    assert isinstance(m, MultivariateDistribution)
    assert m.dimension() == 2
    np.testing.assert_array_equal(m.mean(), [1, 3])


def test_a_fractional_or_repeated_index_is_refused_rather_than_truncated():
    mv = mvdist_normal([1, 2, 3], S)
    with pytest.raises(ValueError, match="whole numbers"):
        mv.marginal([1.9, 3])
    with pytest.raises(ValueError, match="must not repeat"):
        mv.marginal([1, 1])
    with pytest.raises(ValueError, match="between 1 and 3"):
        mv.marginal([1, 4])
    with pytest.raises(ValueError, match="non-empty"):
        mv.marginal([])
    with pytest.raises(ValueError, match="whole numbers"):
        mv.conditional(given=[2.5], values=[5])
    with pytest.raises(ValueError, match="must not repeat"):
        mv.conditional(given=[2, 2], values=[5, 5])


def test_conditional_takes_1_based_indices_and_returns_the_complement():
    mv = mvdist_normal([1, 2, 3], S)
    cd = mv.conditional(given=2, values=5)
    assert cd.dimension() == 2
    np.testing.assert_array_equal(cd.mean(), [1, 3])


def test_interval_integrates_to_one_over_the_whole_space():
    mv = mvdist_normal([0, 0], np.eye(2))
    assert mv.interval([-100, -100], [100, 100]) == pytest.approx(1, abs=1e-6)


def test_a_seeded_multivariate_draw_has_the_right_shape_and_repeats():
    mv = mvdist_normal([0, 0], np.eye(2))
    d = mv.random(4, seed=12345)
    assert d.shape == (4, 2)
    assert np.array_equal(d, mv.random(4, seed=12345))
    assert mv.random(4, seed=12345, method="latin_hypercube").shape == (4, 2)


def test_latin_hypercube_needs_an_explicit_seed():
    mv = mvdist_normal([0, 0], np.eye(2))
    with pytest.raises(ValueError, match="seed"):
        mv.random(4, method="latin_hypercube")


def test_the_other_four_families_build():
    assert math.isfinite(mvdist_dirichlet([2, 3, 4]).pdf([0.2, 0.3, 0.5]))
    assert math.isfinite(mvdist_multinomial(10, [0.2, 0.3, 0.5]).pdf([2, 3, 5]))
    assert mvdist_student_t(5, [0, 0]).dimension() == 2


def test_the_upstream_gaps_are_named_not_leaked():
    with pytest.raises(Exception, match="Dirichlet"):
        mvdist_dirichlet([2, 3]).cdf([0.5, 0.5])
    with pytest.raises(Exception, match="MultivariateStudentT"):
        mvdist_student_t(5, [0, 0]).marginal([1])
    with pytest.raises(Exception, match="MultivariateStudentT"):
        mvdist_student_t(5, [0, 0]).interval([-1, -1], [1, 1])


def test_mvdist_params_reports_the_family_specific_parameters():
    assert mvdist_student_t(5, [0, 0]).params() == {"df": 5.0}
    p = mvdist_dirichlet([2, 3, 4]).params()
    np.testing.assert_array_equal(p["alpha"], [2, 3, 4])
    assert p["alpha_sum"] == pytest.approx(9)
    m = mvdist_multinomial(10, [0.2, 0.3, 0.5]).params()
    assert m["trials"] == 10
    np.testing.assert_allclose(m["probabilities"], [0.2, 0.3, 0.5])


def test_a_mvdist_round_trips_through_pickle():
    mv = mvdist_normal([1, 2, 3], S)
    restored = pickle.loads(pickle.dumps(mv))
    np.testing.assert_array_equal(restored.mean(), mv.mean())
