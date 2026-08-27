"""Machine learning.

The "ml" toolbox group (P5): the whole ported ``Numerics.MachineLearning`` namespace -- five
supervised learners and three unsupervised ones -- over
``core/include/corehydro/numerics/support/toolbox/ml.hpp``.

Mirrors corehydror's own ml.R; both packages share every signature, default and error message,
so a change here is not one-sided.
"""

from __future__ import annotations

import numpy as np

from .toolbox import _toolbox_run

__all__ = [
    "ml_kmeans",
    "ml_gaussian_mixture",
    "ml_jenks_breaks",
    "ml_decision_tree",
    "ml_random_forest",
    "ml_knn",
    "ml_naive_bayes",
    "ml_glm",
]


def _ml_matrix(x, arg: str = "x") -> np.ndarray:
    """Coerce a predictor argument to a 2-D float array with one row per observation."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"`{arg}` must be one- or two-dimensional")
    return a


def _ml_flat(a: np.ndarray) -> list[float]:
    """Flatten ROW-MAJOR, the layout the shared runner assumes (see ml.hpp's file header)."""
    return [float(v) for v in np.ascontiguousarray(a, dtype=float).ravel(order="C")]


def _ml_check_xy(x: np.ndarray, y) -> np.ndarray:
    ya = np.asarray(y, dtype=float).ravel()
    if ya.size != x.shape[0]:
        raise ValueError(
            f"`y` must be numeric with one value per row of `x`; got {ya.size} and {x.shape[0]}"
        )
    return ya


def _ml_newdata(newdata, x: np.ndarray) -> np.ndarray:
    nd = _ml_matrix(newdata, "newdata")
    if nd.shape[1] != x.shape[1]:
        raise ValueError(
            "`newdata` must have the same number of columns as `x`; "
            f"got {nd.shape[1]} and {x.shape[1]}"
        )
    return nd


def _ml_reshape(r: dict) -> np.ndarray:
    """Reshape a runner result carrying ``dims`` back into a 2-D array (the runner is row-major)."""
    return np.asarray(r["values"], dtype=float).reshape(r["dims"][0], r["dims"][1])


def _choice(value, choices, what: str) -> str:
    if value not in choices:
        raise ValueError(f"unknown {what} '{value}'; expected one of {', '.join(choices)}")
    return str(value)


def ml_kmeans(x, k: int, seed: int | None = None, kmeans_plus_plus: bool = True,
              max_iterations: int = 1000) -> dict:
    """k-means clustering.

    Mirrors the C# ``KMeans`` class: partitions the rows of ``x`` into ``k`` clusters, each row
    belonging to the cluster with the nearest centroid. Initialization is k-means++ by default.

    Two behaviours inherited from upstream are worth knowing:

    - ``labels`` are 0-BASED in both Python and R, matching the library's own indexing (the same
      choice :func:`shortest_path` made for node indices).
    - With ``k = 1`` the algorithm stops before its first update step, so the single reported
      "cluster mean" is a randomly chosen observation rather than the mean of ``x``.

    Parameters
    ----------
    x : array_like
        A 2-D array with one row per observation, or a 1-D array for a single feature.
    k : int
        The number of clusters.
    seed : int, optional
        PRNG seed. ``None`` (the default) uses the computer clock, so the fit is not
        reproducible; supply a seed for a reproducible fit.
    kmeans_plus_plus : bool, default True
        Use k-means++ initialization rather than a uniform draw.
    max_iterations : int, default 1000
        The iteration cap.

    Returns
    -------
    dict
        ``means`` (a ``k`` by ``x.shape[1]`` array of centroids), ``labels`` (a 0-based integer
        array, one per row of ``x``), and ``iterations``.

    Examples
    --------
    >>> from corehydropy import ml_kmeans
    >>> x = [[1, 2], [1.2, 2.1], [0.8, 1.9], [8, 9], [8.3, 9.2], [7.9, 8.8]]
    >>> fit = ml_kmeans(x, k=2, seed=12345)
    >>> fit["means"].shape
    (2, 2)
    """
    xa = _ml_matrix(x)
    options = {
        "rows": int(xa.shape[0]), "columns": int(xa.shape[1]), "k": int(k),
        "seed": int(-1 if seed is None else seed),
        "kmeans_plus_plus": bool(kmeans_plus_plus),
        "max_iterations": int(max_iterations),
    }
    data = [_ml_flat(xa)]
    return {
        "means": _ml_reshape(_toolbox_run("ml", "kmeans_means", data, options)),
        "labels": np.asarray(
            _toolbox_run("ml", "kmeans_labels", data, options)["values"], dtype=int
        ),
        "iterations": int(_toolbox_run("ml", "kmeans_iterations", data, options)["values"][0]),
    }


def ml_gaussian_mixture(x, k: int, seed: int | None = None, kmeans_plus_plus: bool = True,
                        max_iterations: int = 1000, tolerance: float = 1e-8) -> dict:
    """Gaussian mixture model.

    Mirrors the C# ``GaussianMixtureModel`` class: fits a mixture of ``k`` multivariate normals
    by expectation-maximization, initialized from a k-means fit. It generalizes
    :func:`ml_kmeans` by carrying a full covariance per component rather than only a centre.

    ``labels`` are 0-BASED, as in :func:`ml_kmeans`. ``log_likelihood`` follows the library's own
    definition, which OMITS the multivariate-normal normalizing constant
    ``-0.5 * x.shape[1] * log(2 * pi)`` per observation -- so it is short of the true mixture
    log-likelihood by ``x.shape[0] * x.shape[1] / 2 * log(2 * pi)``. That constant cancels when
    comparing two fits of the same data (which is what the library uses it for), but do not feed
    this value to an information criterion without adding it back. A run that hits
    ``max_iterations`` without converging reports ``log_likelihood`` as 0.

    Parameters
    ----------
    x : array_like
        A 2-D array with one row per observation, or a 1-D array.
    k : int
        The number of mixture components.
    seed : int, optional
        PRNG seed; ``None`` uses the computer clock.
    kmeans_plus_plus : bool, default True
        Use k-means++ initialization for the starting fit.
    max_iterations : int, default 1000
        The EM iteration cap.
    tolerance : float, default 1e-8
        The relative convergence tolerance on the log-likelihood.

    Returns
    -------
    dict
        ``means`` (``k`` by p), ``sigmas`` (a length-``k`` list of p-by-p covariance arrays),
        ``weights``, ``labels`` (0-based), ``log_likelihood``, and ``iterations``.

    Examples
    --------
    >>> from corehydropy import ml_gaussian_mixture
    >>> x = [1, 1.2, 0.9, 1.1, 1.05, 8, 8.3, 7.9, 8.1, 8.2, 1.0, 8.0]
    >>> fit = ml_gaussian_mixture(x, k=2, seed=12345)
    >>> len(fit["sigmas"])
    2
    """
    xa = _ml_matrix(x)
    options = {
        "rows": int(xa.shape[0]), "columns": int(xa.shape[1]), "k": int(k),
        "seed": int(-1 if seed is None else seed),
        "kmeans_plus_plus": bool(kmeans_plus_plus),
        "max_iterations": int(max_iterations),
        "tolerance": float(tolerance),
    }
    data = [_ml_flat(xa)]
    stacked = _ml_reshape(_toolbox_run("ml", "gmm_sigmas", data, options))
    p = int(xa.shape[1])
    sigmas = [stacked[i * p:(i + 1) * p, :] for i in range(int(k))]
    return {
        "means": _ml_reshape(_toolbox_run("ml", "gmm_means", data, options)),
        "sigmas": sigmas,
        "weights": np.asarray(_toolbox_run("ml", "gmm_weights", data, options)["values"]),
        "labels": np.asarray(
            _toolbox_run("ml", "gmm_labels", data, options)["values"], dtype=int
        ),
        "log_likelihood": float(
            _toolbox_run("ml", "gmm_log_likelihood", data, options)["values"][0]
        ),
        "iterations": int(_toolbox_run("ml", "gmm_iterations", data, options)["values"][0]),
    }


def ml_jenks_breaks(x, n_clusters: int, is_data_sorted: bool = False) -> dict:
    """Jenks natural breaks classification.

    Mirrors the C# ``JenksNaturalBreaks`` class: partitions a one-dimensional sample into
    ``n_clusters`` classes minimizing the within-class sum of squared deviations.

    Upstream fails on fully degenerate input: if every value of ``x`` is identical, the
    underlying algorithm computes a negative class boundary and errors. Any spread at all is
    fine, including heavily tied data.

    Parameters
    ----------
    x : array_like
        The values to classify.
    n_clusters : int
        The number of classes. Must be at least 1 and no more than ``len(x)``.
    is_data_sorted : bool, default False
        Set ``True`` to skip the internal sort when ``x`` is already ascending.

    Returns
    -------
    dict
        ``breaks`` (each class's maximum value), ``clusters`` (an ``n_clusters`` by 8 array with
        columns ``start_index``, ``end_index``, ``count``, ``min``, ``max``, ``sum``,
        ``average``, ``variance``; the two index columns are 0-based positions into the SORTED
        data), and ``gvf``, the goodness-of-variance-fit measure, which approaches 1 for a better
        fit.

    Examples
    --------
    >>> from corehydropy import ml_jenks_breaks
    >>> x = [1, 1.2, 1.4, 8, 8.3, 8.6, 30, 31, 32]
    >>> len(ml_jenks_breaks(x, n_clusters=3)["breaks"])
    3
    """
    data = [[float(v) for v in np.asarray(x, dtype=float).ravel()]]
    options = {"n_clusters": int(n_clusters), "is_data_sorted": bool(is_data_sorted)}
    return {
        "breaks": np.asarray(_toolbox_run("ml", "jenks_breaks", data, options)["values"]),
        "clusters": _ml_reshape(_toolbox_run("ml", "jenks_clusters", data, options)),
        "gvf": float(_toolbox_run("ml", "jenks_gvf", data, options)["values"][0]),
    }


def ml_decision_tree(x, y, newdata, seed: int | None = None, regression: bool = True,
                     features: int | None = None, minimum_split_size: int = 2,
                     max_depth: int = 100) -> np.ndarray:
    """Decision tree regression or classification.

    Mirrors the C# ``DecisionTree`` class: recursively splits the training data on the feature
    and threshold that most reduce variance (regression) or most increase information gain
    (classification), then predicts by walking a new observation down the tree.

    At the library's defaults a REGRESSION tree recurses until every leaf holds a single training
    observation, so it memorizes the training data and generalizes poorly. That is upstream's
    behaviour, not a port artifact, and it is why :func:`ml_random_forest` exists. Set
    ``minimum_split_size`` or ``max_depth`` to regularize it.

    Parameters
    ----------
    x : array_like
        Training predictors, one row per observation.
    y : array_like
        The training response, one value per row of ``x``.
    newdata : array_like
        Predictors to predict for, with the same number of columns as ``x``.
    seed : int, optional
        PRNG seed for the random feature subsets; ``None`` uses the computer clock.
    regression : bool, default True
        ``True`` fits a regression tree; ``False`` a classifier.
    features : int, optional
        The number of random features to consider at each split. ``None`` (the default) uses the
        library's own ``max(1, x.shape[1] - 1)``.
    minimum_split_size : int, default 2
        The smallest node the tree will split.
    max_depth : int, default 100
        The recursion cap.

    Returns
    -------
    numpy.ndarray
        One prediction per row of ``newdata``.

    Examples
    --------
    >>> from corehydropy import ml_decision_tree
    >>> x = [1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105]
    >>> y = [10, 10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 100]
    >>> ml_decision_tree(x, y, newdata=[3, 104], seed=7).tolist()
    [10.0, 100.0]
    """
    xa = _ml_matrix(x)
    ya = _ml_check_xy(xa, y)
    nd = _ml_newdata(newdata, xa)
    options = {
        "rows": int(xa.shape[0]), "columns": int(xa.shape[1]), "predict_rows": int(nd.shape[0]),
        "seed": int(-1 if seed is None else seed),
        "is_regression": bool(regression),
        "minimum_split_size": int(minimum_split_size),
        "max_depth": int(max_depth),
    }
    if features is not None:
        options["features"] = int(features)
    r = _toolbox_run("ml", "decision_tree_predict",
                     [_ml_flat(xa), [float(v) for v in ya], _ml_flat(nd)], options)
    return np.asarray(r["values"])


def ml_random_forest(x, y, newdata, seed: int | None = None, regression: bool = True,
                     features: int | None = None, minimum_split_size: int = 2,
                     max_depth: int = 100, number_of_trees: int = 1000,
                     alpha: float = 0.1) -> dict:
    """Random forest regression or classification.

    Mirrors the C# ``RandomForest`` class: fits ``number_of_trees`` decision trees on bootstrap
    resamples of the training data and reports the spread of their predictions as an interval.

    Training cost is linear in ``number_of_trees``, and the library's default of 1000 is the knob
    to turn if a call is slow -- a few dozen trees is usually enough to see the shape of the
    answer. A seeded run is bit-identical between Python and R, because the whole computation
    lives in the shared compiled core.

    For a classifier every column is floored to an integer class label, including ``mean``.

    Parameters
    ----------
    x : array_like
        Training predictors, one row per observation.
    y : array_like
        The training response, one value per row of ``x``.
    newdata : array_like
        Predictors to predict for, with the same number of columns as ``x``.
    seed : int, optional
        PRNG seed; ``None`` uses the computer clock.
    regression : bool, default True
        ``True`` fits regression trees; ``False`` classifiers.
    features : int, optional
        The number of random features per split. ``None`` uses ``max(1, x.shape[1] - 1)``.
    minimum_split_size : int, default 2
        The smallest node a tree will split.
    max_depth : int, default 100
        The recursion cap.
    number_of_trees : int, default 1000
        How many trees to grow.
    alpha : float, default 0.1
        The interval level: 0.1 gives a 90% interval.

    Returns
    -------
    dict
        ``intervals``, an array with one row per row of ``newdata`` and four columns, and
        ``columns``, the column names ``["lower", "median", "upper", "mean"]``.

    Examples
    --------
    >>> from corehydropy import ml_random_forest
    >>> x = [1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105]
    >>> y = [10, 11, 10, 11, 10, 11, 100, 101, 100, 101, 100, 101]
    >>> ml_random_forest(x, y, newdata=[3, 104], seed=42, number_of_trees=25)["intervals"].shape
    (2, 4)
    """
    xa = _ml_matrix(x)
    ya = _ml_check_xy(xa, y)
    nd = _ml_newdata(newdata, xa)
    options = {
        "rows": int(xa.shape[0]), "columns": int(xa.shape[1]), "predict_rows": int(nd.shape[0]),
        "seed": int(-1 if seed is None else seed),
        "is_regression": bool(regression),
        "minimum_split_size": int(minimum_split_size),
        "max_depth": int(max_depth),
        "number_of_trees": int(number_of_trees),
        "alpha": float(alpha),
    }
    if features is not None:
        options["features"] = int(features)
    r = _toolbox_run("ml", "random_forest_predict",
                     [_ml_flat(xa), [float(v) for v in ya], _ml_flat(nd)], options)
    return {"intervals": _ml_reshape(r), "columns": list(r["names"])}


def ml_knn(x, y, newdata, k: int, regression: bool = True, what: str = "prediction",
           seed: int | None = None, realizations: int = 1000, alpha: float = 0.1):
    """k-nearest-neighbors regression or classification.

    Mirrors the C# ``KNearestNeighbors`` class: predicts from the ``k`` training rows closest to
    each new observation -- an inverse-squared-distance weighted average of their responses for
    regression, or their most common response for classification.

    Parameters
    ----------
    x : array_like
        Training predictors, one row per observation.
    y : array_like
        The training response, one value per row of ``x``.
    newdata : array_like
        Predictors to predict for, with the same number of columns as ``x``.
    k : int
        The number of neighbors.
    regression : bool, default True
        ``True`` for a weighted average; ``False`` for the modal class.
    what : {"prediction", "neighbors", "bootstrap", "intervals"}
        Which result to return: the prediction (the default), the indices of each new
        observation's neighbors, a prediction from one bootstrap resample of the training data,
        or bootstrapped prediction intervals.
    seed : int, optional
        PRNG seed, used by ``"bootstrap"`` and ``"intervals"``; ``None`` uses the clock.
    realizations : int, default 1000
        The number of bootstrap resamples for ``"intervals"``.
    alpha : float, default 0.1
        The interval level for ``"intervals"``: 0.1 gives a 90% interval.

    Returns
    -------
    numpy.ndarray or dict
        For ``"prediction"`` and ``"bootstrap"``, a 1-D array. For ``"neighbors"``, a 2-D array
        of 0-BASED training-row indices with one row per row of ``newdata`` and ``k`` columns.
        For ``"intervals"``, a dict with ``intervals`` and ``columns``
        (``["lower", "median", "upper", "mean"]``) -- note ``"intervals"`` always reports
        percentiles, even for a classifier, because upstream has no classification branch there.

    Examples
    --------
    >>> from corehydropy import ml_knn
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    >>> float(ml_knn(x, y, newdata=[4.5], k=2)[0])
    45.0
    """
    what = _choice(what, ("prediction", "neighbors", "bootstrap", "intervals"), "what")
    xa = _ml_matrix(x)
    ya = _ml_check_xy(xa, y)
    nd = _ml_newdata(newdata, xa)
    options = {
        "rows": int(xa.shape[0]), "columns": int(xa.shape[1]), "predict_rows": int(nd.shape[0]),
        "k": int(k), "is_regression": bool(regression),
        "seed": int(-1 if seed is None else seed),
        "realizations": int(realizations), "alpha": float(alpha),
    }
    data = [_ml_flat(xa), [float(v) for v in ya], _ml_flat(nd)]
    method = {
        "prediction": "knn_predict",
        "neighbors": "knn_neighbors",
        "bootstrap": "knn_bootstrap_predict",
        "intervals": "knn_prediction_intervals",
    }[what]
    r = _toolbox_run("ml", method, data, options)
    if what == "neighbors":
        return _ml_reshape(r).astype(int)
    if what == "intervals":
        return {"intervals": _ml_reshape(r), "columns": list(r["names"])}
    return np.asarray(r["values"])


def ml_naive_bayes(x, y, newdata=None) -> dict:
    """Gaussian naive Bayes classification.

    Mirrors the C# ``NaiveBayes`` class: assumes each feature is normally distributed given the
    class and independent of the other features, then assigns each new observation the class
    with the highest posterior probability.

    ``classes`` follows the order the class labels FIRST APPEAR in ``y``, not sorted order, and
    every per-class row of ``means``, ``standard_deviations`` and ``priors`` is indexed to match.
    A class with a single member gets a standard deviation of 1e-6 rather than 0.

    Parameters
    ----------
    x : array_like
        Training predictors, one row per observation.
    y : array_like
        The training class labels, one per row of ``x``.
    newdata : array_like, optional
        Predictors to classify, with the same number of columns as ``x``. ``None`` (the default)
        trains without predicting.

    Returns
    -------
    dict
        ``classes``, ``means`` and ``standard_deviations`` (both one row per class and one column
        per feature), ``priors``, and -- when ``newdata`` is supplied -- ``prediction``.

    Examples
    --------
    >>> from corehydropy import ml_naive_bayes
    >>> x = [1, 1.1, 0.9, 1.2, 1.05, 5, 5.1, 4.9, 5.2, 5.05]
    >>> y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    >>> ml_naive_bayes(x, y, newdata=[1.0, 5.0])["prediction"].tolist()
    [0.0, 1.0]
    """
    xa = _ml_matrix(x)
    ya = _ml_check_xy(xa, y)
    options = {"rows": int(xa.shape[0]), "columns": int(xa.shape[1])}
    data = [_ml_flat(xa), [float(v) for v in ya]]
    out = {
        "classes": np.asarray(_toolbox_run("ml", "naive_bayes_classes", data, options)["values"]),
        "means": _ml_reshape(_toolbox_run("ml", "naive_bayes_means", data, options)),
        "standard_deviations": _ml_reshape(
            _toolbox_run("ml", "naive_bayes_sds", data, options)
        ),
        "priors": np.asarray(_toolbox_run("ml", "naive_bayes_priors", data, options)["values"]),
    }
    if newdata is not None:
        nd = _ml_newdata(newdata, xa)
        options = dict(options, predict_rows=int(nd.shape[0]))
        r = _toolbox_run("ml", "naive_bayes_predict",
                         [_ml_flat(xa), [float(v) for v in ya], _ml_flat(nd)], options)
        out["prediction"] = np.asarray(r["values"])
    return out


# The five GLM families, which are the five LinkFunctionType members GeneralizedLinearModel
# implements a log-likelihood for. The other two link types the Numerics link layer offers
# (yeo_johnson, fisher_z) have no GLM family and are rejected by name rather than failing inside
# the first likelihood evaluation.
_GLM_LINKS = ("identity", "log", "logit", "probit", "complementary_log_log")
_GLM_LOCAL_METHODS = ("nelder_mead", "bfgs", "powell", "adam", "gradient_descent")


def ml_glm(x, y, intercept: bool = True, link: str = "identity",
           local_method: str = "nelder_mead", robust_se: bool = False, newdata=None,
           alpha: float = 0.1) -> dict:
    """Generalized linear model.

    Mirrors the C# ``GeneralizedLinearModel`` class: fits a linear predictor mapped to the
    response scale by a link function, by maximizing the family log-likelihood with a local
    optimizer.

    The ``link`` selects the family as well as the transform: ``"identity"`` is Normal,
    ``"log"`` is Poisson, and ``"logit"``, ``"probit"`` and ``"complementary_log_log"`` are
    Binomial.

    ``local_method`` accepts all five optimizers here, unlike :func:`optim_minimize`'s
    ``local_method`` argument (which takes three) -- the two upstream classes construct different
    sets, and this surface follows each one rather than imposing a single list.

    Parameters
    ----------
    x : array_like
        Predictors, one row per observation.
    y : array_like
        The response, one value per row of ``x``.
    intercept : bool, default True
        Fit an intercept term.
    link : {"identity", "log", "logit", "probit", "complementary_log_log"}
        The link function, which also selects the family.
    local_method : {"nelder_mead", "bfgs", "powell", "adam", "gradient_descent"}
        The optimizer.
    robust_se : bool, default False
        Use the sandwich (heteroskedasticity-consistent) covariance rather than the delta-method
        one. Changes the standard errors, not the coefficients.
    newdata : array_like, optional
        Predictors to predict for. When supplied the result gains ``prediction`` and
        ``prediction_intervals``.
    alpha : float, default 0.1
        The interval level for ``prediction_intervals``: 0.1 gives a 90% interval.

    Returns
    -------
    dict
        ``coefficients``, ``standard_errors``, ``z_values``, ``p_values``, ``sigma``, ``df``,
        ``n``, ``aic``, ``aicc``, ``bic``, ``vcov`` and ``residuals``; plus ``prediction``,
        ``prediction_intervals`` and ``prediction_interval_columns``
        (``["lower", "mean", "upper"]``) when ``newdata`` is supplied.

    Examples
    --------
    >>> from corehydropy import ml_glm
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.0, 20.2]
    >>> len(ml_glm(x, y)["coefficients"])
    2
    """
    link = _choice(link, _GLM_LINKS, "link")
    local_method = _choice(local_method, _GLM_LOCAL_METHODS, "local_method")
    xa = _ml_matrix(x)
    ya = _ml_check_xy(xa, y)
    options = {
        "rows": int(xa.shape[0]), "columns": int(xa.shape[1]), "intercept": bool(intercept),
        "link": link, "local_method": local_method, "robust_se": bool(robust_se),
        "alpha": float(alpha),
    }
    data = [_ml_flat(xa), [float(v) for v in ya]]

    fit = _toolbox_run("ml", "glm_fit", data, options)
    named = dict(zip(fit["names"], fit["values"]))
    p = sum(1 for n in fit["names"] if n.startswith("beta_"))
    pick = lambda prefix: np.asarray([named[f"{prefix}{i + 1}"] for i in range(p)])  # noqa: E731

    out = {
        "coefficients": pick("beta_"),
        "standard_errors": pick("se_"),
        "z_values": pick("z_"),
        "p_values": pick("p_"),
        "sigma": float(named["sigma"]),
        "df": int(named["df"]),
        "n": int(named["n"]),
        "aic": float(named["aic"]),
        "aicc": float(named["aicc"]),
        "bic": float(named["bic"]),
        "vcov": _ml_reshape(_toolbox_run("ml", "glm_covariance", data, options)),
        "residuals": np.asarray(_toolbox_run("ml", "glm_residuals", data, options)["values"]),
    }
    if newdata is not None:
        nd = _ml_newdata(newdata, xa)
        options = dict(options, predict_rows=int(nd.shape[0]))
        data3 = [_ml_flat(xa), [float(v) for v in ya], _ml_flat(nd)]
        out["prediction"] = np.asarray(
            _toolbox_run("ml", "glm_predict", data3, options)["values"]
        )
        pi = _toolbox_run("ml", "glm_predict_intervals", data3, options)
        out["prediction_intervals"] = _ml_reshape(pi)
        out["prediction_interval_columns"] = list(pi["names"])
    return out
