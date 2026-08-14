"""Stats utility wrapper tests.

Numeric expectations come from the data_utility fixture
(fixtures/data/statistics_utilities.json) routed through the PUBLIC wrappers; the
rest are structural (round trips, shapes, argument validation) with no oracle
content.
"""

import json
import math
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

import corehydropy as ch


def _fixtures_dir() -> Path:
    try:
        packaged = files("corehydropy") / "fixtures"
        if packaged.is_dir():
            return Path(str(packaged))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return Path(__file__).resolve().parents[2] / "fixtures"


def test_public_stats_reproduce_data_utility_oracles():
    spec = json.loads((_fixtures_dir() / "data" / "statistics_utilities.json").read_text(encoding="utf-8"))
    datasets = {k: list(v) for k, v in spec["datasets"].items()}
    for case in spec["cases"]:
        fn = case["function"]
        args = case.get("args", [])
        data = datasets.get(case.get("dataset"), [])
        if fn == "MGBT":
            actual = float(ch.mgbt_test(data))
        elif fn == "BoxCoxLambda":
            actual = ch.box_cox_lambda(data)
        elif fn == "BoxCoxTransform":
            actual = ch.box_cox(data, args[0])[int(args[1])]
        elif fn == "YeoJohnsonLambda":
            actual = ch.yeo_johnson_lambda(data)
        elif fn == "YeoJohnsonTransform":
            actual = ch.yeo_johnson(data, args[0])[int(args[1])]
        elif fn == "PlottingPosition":
            actual = ch.plotting_positions(int(args[0]), alpha=args[1])[int(args[2])]
        elif fn in ("LHSRandom", "LHSMedian"):
            m = ch.latin_hypercube(int(args[0]), int(args[1]), seed=int(args[2]),
                                   median=fn == "LHSMedian")
            actual = m[int(args[3]), int(args[4])]
        else:  # pragma: no cover
            raise AssertionError(f"unhandled function {fn}")
        a = case["assertions"][0]
        expected = float(a["expected"])
        if a["mode"] == "equal":
            # v2.1.4 FitLambda hardening added NaN-expected cases (box_cox_lambda /
            # yeo_johnson_lambda on invalid/degenerate samples); NaN != NaN under plain
            # equality, so mirror test_fixtures.py's _check special case.
            if math.isnan(expected):
                assert math.isnan(actual)
            else:
                assert actual == expected
        elif a["mode"] == "abs":
            assert abs(actual - expected) <= a["tol"]
        else:
            assert abs(actual - expected) / abs(expected) <= a["tol"]


def test_transform_round_trips():
    x = ch.Distribution("LogNormal", [3, 0.5]).random(50, seed=3)
    for lam in (-0.5, 0.0, 0.7):
        assert np.allclose(ch.box_cox_inverse(ch.box_cox(x, lam), lam), x)
    y = ch.Distribution("Normal", [0, 2]).random(50, seed=3)
    for lam in (0.3, 1.0, 1.8):
        assert np.allclose(ch.yeo_johnson_inverse(ch.yeo_johnson(y, lam), lam), y)


def test_plotting_positions():
    pp = ch.plotting_positions(10)
    assert pp.shape == (10,)
    assert np.all(np.diff(pp) > 0)
    assert np.all((pp > 0) & (pp < 1))
    assert np.array_equal(ch.plotting_positions(10, method="hazen"),
                          ch.plotting_positions(10, alpha=0.5))
    with pytest.raises(ValueError):
        ch.plotting_positions(10, method="nope")


def test_latin_hypercube_stratified_and_seeded():
    m = ch.latin_hypercube(20, 3, seed=5)
    assert m.shape == (20, 3)
    assert np.all((m > 0) & (m < 1))
    # LHS property: exactly one sample per 1/n bin in every dimension.
    for j in range(3):
        assert sorted(np.floor(m[:, j] * 20).astype(int)) == list(range(20))
    assert np.array_equal(ch.latin_hypercube(20, 3, seed=5), m)


def test_mgbt_flags_planted_low_outliers():
    x = ch.Distribution("LogNormal", [3, 0.5]).random(40, seed=7)
    assert ch.mgbt_test(np.concatenate([x, [1.0, 2.0]])) == 2
    assert ch.mgbt_test(x) == 0
