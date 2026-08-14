"""Public Distribution API wrapper tests.

Oracle values are NOT hardcoded here: numeric expectations are read from the fixture
JSONs (the single source of truth) and routed through the PUBLIC wrappers, so these
tests prove the wrappers hit the same glue the fixture runner validates.

The composite-family cases at the bottom (test_dist_truncated_and_friends) are the Python twin
of corehydror's tests/testthat/test-distribution.R "composite families" section -- same cases,
same order, same names where Python allows. Oracle VALUES live in fixtures/ and are asserted by
test_fixtures.py; these assert object shape, argument handling, error messages, and that every
existing method accepts a composite.
"""

import json
import math
import pickle
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from corehydropy import (
    Distribution,
    dist_competing_risks,
    dist_empirical,
    dist_kde,
    dist_mixture,
    dist_truncated,
    distribution_names,
)


def _fixtures_dir() -> Path:
    try:
        packaged = files("corehydropy") / "fixtures"
        if packaged.is_dir():
            return Path(str(packaged))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return Path(__file__).resolve().parents[2] / "fixtures"


def test_constructor_validation():
    with pytest.raises(ValueError, match="unknown distribution family"):
        Distribution("NotAFamily", [1, 2])
    with pytest.raises(ValueError, match="expects 2 parameters"):
        Distribution("Normal", [1])
    with pytest.raises(TypeError):
        Distribution(3.14, [1, 2])
    d = Distribution("Normal", [100, 15])
    assert d.family == "Normal"
    assert d.params == [100.0, 15.0]


def test_distribution_names():
    nms = distribution_names()
    for expected in ("Normal", "LogNormal", "Gumbel", "GeneralizedExtremeValue", "Weibull"):
        assert expected in nms
    assert len(nms) == len(set(nms))


def test_public_methods_reproduce_fixture_oracles():
    spec = json.loads(
        (_fixtures_dir() / "distributions" / "univariate" / "gumbel.json").read_text(encoding="utf-8")
    )
    for case in spec["cases"]:
        if "params" not in case["construct"]:
            continue
        d = Distribution(spec["target"], case["construct"]["params"])
        for a in case["assertions"]:
            args = a.get("args", [])
            method = a["method"]
            if method == "pdf":
                actual = d.pdf(args[0])
            elif method == "cdf":
                actual = d.cdf(args[0])
            elif method == "quantile":
                actual = d.quantile(args[0])
            elif method == "mean":
                actual = d.moments()["mean"]
            elif method == "sd":
                actual = d.moments()["sd"]
            elif method == "random_value":
                actual = d.random(int(args[0]), seed=int(args[1]))[int(args[2])]
            else:
                continue
            try:
                expected = float(a["expected"])
            except (TypeError, ValueError):
                continue
            if math.isnan(expected):
                continue
            if a["mode"] == "abs":
                assert abs(actual - expected) <= a["tol"]
            elif a["mode"] == "rel":
                assert abs(actual - expected) / abs(expected) <= a["tol"]


def test_vectorization_and_scalar_passthrough():
    d = Distribution("Normal", [0, 1])
    assert d.pdf([-1, 0, 1]).shape == (3,)
    assert isinstance(d.pdf(0.0), float)
    assert d.cdf(0.0) == 0.5
    assert d.quantile(d.cdf(1.3)) == pytest.approx(1.3)
    assert d.log_pdf(0.7) == pytest.approx(math.log(d.pdf(0.7)))
    xs = [0.0, 1.0, -1.0]
    assert d.log_likelihood(xs) == pytest.approx(sum(d.log_pdf(x) for x in xs))


def test_random_is_seed_deterministic():
    d = Distribution("Gumbel", [100, 10])
    a = d.random(10, seed=7)
    assert np.array_equal(a, d.random(10, seed=7))
    assert not np.array_equal(a, d.random(10, seed=8))
    assert d.random(25, seed=1).shape == (25,)


def test_fit_matches_internal_glue():
    from corehydropy import _core

    d = Distribution("Gumbel", [100, 10])
    x = d.random(200, seed=42)
    f = Distribution.fit("Gumbel", x, method="mle")
    assert isinstance(f, Distribution)
    assert f.params == list(_core.dist_fit("Gumbel", [float(v) for v in x], "mle"))
    with pytest.raises(Exception, match="only MethodOfMoments is supported"):
        Distribution.fit("Deterministic", x, method="mle")


def test_properties_and_repr():
    d = Distribution("Normal", [100, 15])
    names = d.parameter_names
    assert len(names["short"]) == 2 and len(names["full"]) == 2
    assert d.is_valid
    assert not Distribution("Normal", [0, -1]).is_valid
    assert "Normal" in repr(d)
    m = d.moments()
    assert list(m) == ["mean", "median", "mode", "sd", "skewness", "kurtosis",
                       "minimum", "maximum"]


def test_linear_moments_errors_without_support():
    with pytest.raises(Exception, match="no L-moments"):
        Distribution("Cauchy", [0, 1]).linear_moments()


# --- composite families -------------------------------------------------------------------
# Oracle VALUES live in fixtures/ and are asserted by test_fixtures.py; these assert object
# shape, argument handling, error messages, and that every existing method accepts a composite.


def test_dist_truncated_returns_a_distribution_every_method_accepts():
    d = dist_truncated(Distribution("Normal", [2, 1]), min=1.1, max=2.11)
    assert d.family == "TruncatedDistribution"
    assert d.is_composite
    assert math.isfinite(d.pdf(1.5))
    assert math.isfinite(d.cdf(1.5))
    assert d.quantile(0.5) == d.quantile(0.5)
    m = d.moments()
    assert len(m) == 8
    assert list(m) == ["mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"]


def test_the_pointwise_methods_vectorize_over_a_composite():
    d = dist_truncated(Distribution("Normal", [2, 1]), 1.1, 2.11)
    assert d.pdf([1.2, 1.5, 2.0]).shape == (3,)
    assert d.cdf([1.2, 1.5, 2.0]).shape == (3,)


def test_dist_mixture_nests_inside_dist_truncated():
    mix = dist_mixture(
        [Distribution("Normal", [0, 1]), Distribution("Normal", [5, 1])], weights=[0.5, 0.5],
    )
    d = dist_truncated(mix, min=-2, max=7)
    assert d.cdf(7) == pytest.approx(1, abs=1e-9)


def test_dist_mixture_validates_its_arguments():
    d1 = Distribution("Normal", [0, 1])
    with pytest.raises(ValueError, match="weights"):
        dist_mixture([d1], weights=[0.5, 0.5])
    with pytest.raises(TypeError, match="Distribution"):
        dist_mixture([d1, "not a distribution"], [0.5, 0.5])


def test_dist_kde_takes_data_inline_and_defaults_to_the_silverman_bandwidth():
    d = dist_kde([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert d.cdf(5.5) == pytest.approx(0.5, abs=0.05)
    with pytest.raises(ValueError, match="kernel"):
        dist_kde(list(range(1, 11)), kernel="Cosine")


def test_dist_empirical_takes_x_and_p():
    d = dist_empirical(x=[1, 2, 3], p=[0.1, 0.5, 0.9])
    assert d.quantile(0.5) == pytest.approx(2, abs=1e-9)


def test_a_seeded_composite_draw_is_reproducible():
    d = dist_mixture(
        [Distribution("Normal", [0, 1]), Distribution("Normal", [5, 1])], [0.5, 0.5],
    )
    assert np.array_equal(d.random(5, seed=12345), d.random(5, seed=12345))
    assert d.random(5, seed=12345).shape == (5,)


def test_a_composite_round_trips_through_pickle():
    d = dist_truncated(Distribution("Normal", [2, 1]), 1.1, 2.11)
    restored = pickle.loads(pickle.dumps(d))
    assert restored.pdf(1.5) == d.pdf(1.5)


def test_linear_moments_are_refused_for_composites_with_the_reason():
    d = dist_kde(list(range(1, 11)))
    with pytest.raises(ValueError, match="linear moment"):
        d.linear_moments()


def test_distribution_points_structured_families_at_their_constructor():
    with pytest.raises(ValueError, match="dist_empirical"):
        Distribution("Empirical", [1, 2])
    with pytest.raises(ValueError, match="dist_kde"):
        Distribution("KernelDensity", [1])


def test_distribution_names_reports_both_kinds():
    assert "Normal" in distribution_names()
    assert set(distribution_names("structured")) == {
        "TruncatedDistribution", "Mixture", "CompetingRisks", "Empirical", "KernelDensity",
    }
    assert set(distribution_names()) <= set(distribution_names("all"))
