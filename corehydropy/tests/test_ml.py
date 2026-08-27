"""The eight ml_* verbs (P5).

Mirrors corehydror/tests/testthat/test-ml.R case for case: the two files assert the same numbers
and the same error text, so a one-sided change fails one of them.

The oracle values here are the ones the C# test suite asserts (and that
fixtures/ml/machine_learning.json pins through all four runners); these tests check the Python
BINDING reaches them, not the arithmetic, which the fixture and ctest suites own.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import corehydropy as ch


def _fixture() -> dict:
    path = Path(ch.__file__).parent / "fixtures" / "ml" / "machine_learning.json"
    return json.loads(path.read_text())


def _iris_full() -> np.ndarray:
    return np.asarray(_fixture()["datasets"]["iris_flat"], dtype=float).reshape(-1, 4)


def _iris_train() -> np.ndarray:
    return np.asarray(_fixture()["datasets"]["iris_train_flat"], dtype=float).reshape(-1, 4)


def _iris_test_x() -> np.ndarray:
    return np.asarray(_fixture()["datasets"]["iris_test_flat"], dtype=float).reshape(-1, 4)


def _iris_species_train() -> list:
    return _fixture()["datasets"]["iris_species_train"]


def test_ml_kmeans_reproduces_the_iris_cluster_means_and_label_counts():
    fit = ch.ml_kmeans(_iris_full(), k=3, seed=12345)
    assert fit["means"].shape == (3, 4)
    assert fit["means"][0].tolist() == pytest.approx([5.901613, 2.748387, 4.393548, 1.433871],
                                                     abs=1e-6)
    assert fit["means"][1].tolist() == pytest.approx([6.850000, 3.073684, 5.742105, 2.071053],
                                                     abs=1e-6)
    assert fit["means"][2].tolist() == pytest.approx([5.006000, 3.428000, 1.462000, 0.246000],
                                                     abs=1e-6)
    # Labels are 0-based in BOTH languages.
    assert len(fit["labels"]) == 150
    assert set(fit["labels"].tolist()) <= {0, 1, 2}
    assert int((fit["labels"] == 0).sum()) == 62
    assert int((fit["labels"] == 1).sum()) == 38
    assert int((fit["labels"] == 2).sum()) == 50
    assert fit["iterations"] >= 2


def test_ml_gaussian_mixture_returns_a_weight_simplex_and_one_covariance_per_component():
    fit = ch.ml_gaussian_mixture(_iris_full(), k=3, seed=12345)
    assert float(fit["weights"].sum()) == pytest.approx(1.0, abs=1e-12)
    assert fit["weights"].tolist() == pytest.approx([0.3005423, 0.3661243, 0.3333333], abs=1e-2)
    assert fit["means"].shape == (3, 4)
    assert len(fit["sigmas"]) == 3
    for s in fit["sigmas"]:
        assert s.shape == (4, 4)
        assert np.allclose(s, s.T, atol=1e-12)  # symmetric
        assert all(v > 0 for v in np.diag(s))
    assert math.isfinite(fit["log_likelihood"])
    assert set(fit["labels"].tolist()) <= {0, 1, 2}


def test_ml_jenks_breaks_classifies_a_small_vector():
    x = [1, 1.2, 1.4, 8, 8.3, 8.6, 30, 31, 32]
    fit = ch.ml_jenks_breaks(x, n_clusters=3)
    assert fit["breaks"].tolist() == pytest.approx([1.4, 8.6, 32.0])
    assert fit["clusters"].shape == (3, 8)
    assert fit["clusters"][:, 2].tolist() == [3.0, 3.0, 3.0]  # the count column
    # The index columns are 0-based positions into the sorted data.
    assert fit["clusters"][0, 0] == 0.0
    assert fit["clusters"][2, 1] == 8.0
    assert 0.9 < fit["gvf"] < 1.0


def test_ml_naive_bayes_reproduces_the_iris_means_and_predictions():
    fit = ch.ml_naive_bayes(_iris_train(), _iris_species_train(), newdata=_iris_test_x())
    assert fit["classes"].tolist() == [1.0, 2.0, 3.0]
    assert fit["means"].shape == (3, 4)
    assert fit["means"][0].tolist() == pytest.approx([5.016667, 3.456667, 1.466667, 0.220000],
                                                     abs=1e-6)
    assert fit["standard_deviations"][0].tolist() == pytest.approx(
        [0.3097088, 0.3490710, 0.1881550, 0.08051558], abs=1e-6
    )
    assert fit["priors"].tolist() == pytest.approx([30 / 90] * 3, abs=1e-6)
    assert len(fit["prediction"]) == 60
    # The two deliberate misclassifications the C# oracle records, at positions 42 and 53
    # (0-based).
    assert fit["prediction"][42] == 2.0
    assert fit["prediction"][53] == 2.0
    assert int((fit["prediction"] == 3.0).sum()) == 18

    # Without newdata there is no prediction key.
    trained = ch.ml_naive_bayes(_iris_train(), _iris_species_train())
    assert "prediction" not in trained
    assert trained["means"].tolist() == fit["means"].tolist()


def test_ml_knn_reproduces_the_iris_classification_oracle_and_its_four_result_kinds():
    pred = ch.ml_knn(_iris_train(), _iris_species_train(), newdata=_iris_test_x(), k=5,
                     regression=False)
    assert len(pred) == 60
    assert pred[:20].tolist() == [1.0] * 20
    assert pred[33] == 3.0  # the one versicolor row the C# oracle assigns to virginica

    # Neighbors: 0-based training-row indices, one row per query, k columns.
    nb = ch.ml_knn(_iris_train(), _iris_species_train(), newdata=_iris_test_x()[:3], k=5,
                   what="neighbors")
    assert nb.shape == (3, 5)
    assert nb.min() >= 0 and nb.max() < 90

    # A seeded bootstrap prediction is reproducible.
    b1 = ch.ml_knn(_iris_train(), _iris_species_train(), newdata=_iris_test_x()[:3], k=5,
                   what="bootstrap", seed=99)
    b2 = ch.ml_knn(_iris_train(), _iris_species_train(), newdata=_iris_test_x()[:3], k=5,
                   what="bootstrap", seed=99)
    assert b1.tolist() == b2.tolist()

    # Intervals: four ordered columns.
    band = ch.ml_knn(_iris_train(), _iris_species_train(), newdata=_iris_test_x()[:2], k=5,
                     what="intervals", seed=99, realizations=25)
    assert band["columns"] == ["lower", "median", "upper", "mean"]
    m = band["intervals"]
    assert all(m[i, 0] <= m[i, 1] <= m[i, 2] for i in range(m.shape[0]))


def test_ml_decision_tree_and_random_forest_separate_a_clean_two_group_problem():
    x = [1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105]
    y = [10, 11, 10, 11, 10, 11, 100, 101, 100, 101, 100, 101]

    p = ch.ml_decision_tree(x, y, newdata=[3, 104], seed=7)
    assert len(p) == 2
    assert p[0] < 50 and p[1] > 50

    band = ch.ml_random_forest(x, y, newdata=[3, 104], seed=42, number_of_trees=25)
    m = band["intervals"]
    assert m.shape == (2, 4)
    assert band["columns"] == ["lower", "median", "upper", "mean"]
    assert all(m[i, 0] <= m[i, 1] <= m[i, 2] for i in range(2))
    assert m[0, 2] < 50 and m[1, 0] > 50

    # A seeded forest is reproducible, including the mean column.
    again = ch.ml_random_forest(x, y, newdata=[3, 104], seed=42, number_of_trees=25)
    assert m.tolist() == again["intervals"].tolist()

    # A classifier floors every column, including mean.
    clf = ch.ml_random_forest(x, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], newdata=[3, 104],
                              seed=42, regression=False, number_of_trees=25)["intervals"]
    assert clf.tolist() == np.floor(clf).tolist()


def test_ml_glm_reproduces_the_fpp3_simple_regression_oracle_across_links():
    ds = _fixture()["datasets"]
    fit = ch.ml_glm(ds["fpp3_income"], ds["fpp3_consumption"])
    assert fit["coefficients"].tolist() == pytest.approx([0.54510, 0.28060], abs=1e-3)
    assert fit["standard_errors"].tolist() == pytest.approx([0.05569, 0.04744], abs=1e-3)
    assert fit["sigma"] == pytest.approx(0.6026, abs=1e-3)
    assert fit["df"] == 185
    assert fit["n"] == 187
    assert fit["vcov"].shape == (2, 2)
    assert len(fit["residuals"]) == 187
    # z = beta / se.
    assert fit["z_values"].tolist() == pytest.approx(
        (fit["coefficients"] / fit["standard_errors"]).tolist(), abs=1e-12
    )
    assert fit["aic"] < fit["aicc"] < fit["bic"]

    # Poisson (log link), the case whose standard errors are NOT scaled by sigma.
    pois = ch.ml_glm(
        np.asarray(ds["glm_drivers_popden_flat"], dtype=float).reshape(-1, 2),
        ds["glm_deaths"], link="log",
    )
    assert pois["coefficients"].tolist() == pytest.approx(
        [6.322e00, 2.405e-03, -1.480e-04], abs=1e-2
    )
    assert pois["aic"] == pytest.approx(4069.4, abs=1e-2)
    assert pois["df"] == 23

    # newdata adds a prediction and a three-column band.
    with_nd = ch.ml_glm(ds["fpp3_income"], ds["fpp3_consumption"], newdata=[0, 1, 2])
    assert len(with_nd["prediction"]) == 3
    assert with_nd["prediction_interval_columns"] == ["lower", "mean", "upper"]
    assert with_nd["prediction_intervals"][:, 1].tolist() == with_nd["prediction"].tolist()

    # robust_se moves the standard errors but not the coefficients.
    robust = ch.ml_glm(ds["fpp3_income"], ds["fpp3_consumption"], robust_se=True)
    assert robust["coefficients"].tolist() == fit["coefficients"].tolist()
    assert robust["standard_errors"].tolist() != fit["standard_errors"].tolist()


def test_the_ml_verbs_validate_their_arguments():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    with pytest.raises(ValueError, match="`y` must be numeric with one value per row of `x`"):
        ch.ml_decision_tree(x, y[:5], newdata=[1])
    with pytest.raises(ValueError,
                       match="`newdata` must have the same number of columns as `x`"):
        ch.ml_decision_tree(x, y, newdata=[[1, 2]])
    with pytest.raises(ValueError, match="unknown what"):
        ch.ml_knn(x, y, newdata=[1], k=2, what="nope")
    with pytest.raises(ValueError, match="unknown link"):
        ch.ml_glm(x, y, link="nope")
    with pytest.raises(ValueError, match="unknown local_method"):
        ch.ml_glm(x, y, local_method="nope")
    # Upstream's own guards reach the caller intact.
    with pytest.raises(Exception, match="cannot be greater than the length of the data array"):
        ch.ml_jenks_breaks(x, n_clusters=11)
    with pytest.raises(Exception, match="at least ten training data points"):
        ch.ml_decision_tree(x[:9], y[:9], newdata=[1])
