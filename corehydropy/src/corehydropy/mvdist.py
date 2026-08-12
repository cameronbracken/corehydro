"""The multivariate distribution surface. Every verb serializes a
:class:`MultivariateDistribution` to the ``dist_spec.hpp`` grammar and runs one method through
``_core.mvdist_run``; nothing holds C++ state. Mirrors ``corehydror``'s ``R/mvdist.R`` verb for
verb: the R module functions (``mvdist_pdf()``, ``mvdist_mean()``, ...) are methods here instead
(``MultivariateDistribution.pdf()``, ``.mean()``, ...) to match idiomatic Python, and the five
``mvdist_*()`` constructors stay module functions.
"""

from __future__ import annotations

import json

import numpy as np

from . import _core

__all__ = [
    "MultivariateDistribution",
    "mvdist_normal",
    "mvdist_student_t",
    "mvdist_dirichlet",
    "mvdist_multinomial",
    "mvdist_bivariate_empirical",
    "mvdist_names",
]

_MVDIST_FAMILIES = (
    "MultivariateNormal", "MultivariateStudentT", "Dirichlet", "Multinomial", "BivariateEmpirical",
)


def mvdist_names() -> list[str]:
    """List the supported multivariate distribution families.

    Returns
    -------
    list of str
        The five multivariate distribution family names.
    """
    return list(_MVDIST_FAMILIES)


def _mv_indices(idx, n: int, what: str) -> list[int]:
    """Internal: user-facing indices are 1-based (matching Trend/ModelParameter in models.py);
    the spec and the C++ take 0-based. A fractional index is rejected rather than truncated, and
    duplicates are rejected here so the message names the argument."""
    arr = np.atleast_1d(np.asarray(idx))
    if arr.size == 0:
        raise ValueError(f"`{what}` must be a non-empty numeric vector of dimension indices")
    farr = arr.astype(float)
    if np.any(np.isnan(farr)):
        raise ValueError(f"`{what}` must be a non-empty numeric vector of dimension indices")
    frac = farr != np.trunc(farr)
    if np.any(frac):
        bad = ", ".join(str(v) for v in farr[frac])
        raise ValueError(f"`{what}` must be whole numbers; got {bad}")
    ints = farr.astype(int)
    if np.any(ints < 1) or np.any(ints > n):
        raise ValueError(f"`{what}` must be between 1 and {n}")
    seen: set[int] = set()
    dupes: list[int] = []
    for v in ints:
        if v in seen and v not in dupes:
            dupes.append(int(v))
        seen.add(int(v))
    if dupes:
        raise ValueError(f"`{what}` must not repeat a dimension; got {', '.join(str(v) for v in dupes)}")
    return (ints - 1).tolist()


class MultivariateDistribution:
    """A multivariate distribution from the ported Numerics library.

    Stateless: the object holds its spec as a JSON string and every verb runs one method through
    ``_core.mvdist_run``. Nothing holds C++ state, so instances pickle and compare across
    processes. Build one with :func:`mvdist_normal`, :func:`mvdist_student_t`,
    :func:`mvdist_dirichlet`, :func:`mvdist_multinomial`, or :func:`mvdist_bivariate_empirical`
    rather than directly.
    """

    def __init__(self) -> None:
        raise TypeError(
            "build a MultivariateDistribution with mvdist_normal() or one of its siblings, not "
            "directly"
        )

    @classmethod
    def _from_spec(cls, family: str, spec_json: str) -> "MultivariateDistribution":
        self = object.__new__(cls)
        self._family = family
        self._spec = spec_json
        return self

    @property
    def family(self) -> str:
        """str: The multivariate distribution family name."""
        return self._family

    def to_json(self) -> str:
        """This distribution's spec as the JSON the shared C++ core parses.

        Returns
        -------
        str
            The spec JSON.
        """
        return self._spec

    def __repr__(self) -> str:
        return f"MultivariateDistribution({self._family}, dimension {self.dimension()})"

    def _run(self, method: str, args=()) -> dict:
        return _core.mvdist_run(self._spec, method, json.dumps([float(a) for a in args]))

    # -- density, distribution, and dimension ------------------------------------------------

    def pdf(self, x) -> float:
        """Probability density at `x`.

        Parameters
        ----------
        x : array-like of float
            The evaluation point, length equal to :meth:`dimension`.

        Returns
        -------
        float
        """
        return float(self._run("pdf", x)["values"][0])

    def log_pdf(self, x) -> float:
        """Log-density at `x`. See :meth:`pdf`."""
        return float(self._run("log_pdf", x)["values"][0])

    def cdf(self, x) -> float:
        """Cumulative probability at `x`. See :meth:`pdf`."""
        return float(self._run("cdf", x)["values"][0])

    def dimension(self) -> int:
        """int: The number of dimensions."""
        return int(self._run("dimension")["values"][0])

    # -- moments --------------------------------------------------------------------------------

    def mean(self):
        """numpy.ndarray: The mean vector, length :meth:`dimension`."""
        return np.asarray(self._run("mean")["values"])

    def variance(self):
        """numpy.ndarray: The variance vector, length :meth:`dimension`."""
        return np.asarray(self._run("variance")["values"])

    def sd(self):
        """numpy.ndarray: The standard-deviation vector, length :meth:`dimension`."""
        return np.asarray(self._run("sd")["values"])

    def median(self):
        """numpy.ndarray: The median vector, length :meth:`dimension`.

        Defined for ``"MultivariateNormal"`` and ``"MultivariateStudentT"`` (both return the
        centre); not every family defines it upstream.
        """
        return np.asarray(self._run("median")["values"])

    def mode(self):
        """numpy.ndarray: The mode vector, length :meth:`dimension`.

        Defined for ``"MultivariateNormal"``, ``"MultivariateStudentT"``, and ``"Dirichlet"``;
        not every family defines it upstream.
        """
        return np.asarray(self._run("mode")["values"])

    def covariance(self):
        """numpy.ndarray: The ``dimension x dimension`` covariance matrix."""
        n = self.dimension()
        v = self._run("covariance")["values"]
        return np.asarray(v).reshape(n, n)

    def mahalanobis(self, x) -> float:
        """Mahalanobis distance of `x` from the distribution.

        Parameters
        ----------
        x : array-like of float
            The evaluation point.

        Returns
        -------
        float
        """
        return float(self._run("mahalanobis", x)["values"][0])

    # -- inverse CDF and intervals ------------------------------------------------------------

    def inverse_cdf(self, p):
        """Multivariate inverse CDF (Cholesky map).

        Maps a vector of independent uniform draws into one point on the distribution's scale
        via the Cholesky decomposition of its covariance. **This is not a true multivariate
        quantile** (there is no unique multivariate analogue of the univariate inverse CDF); it
        is exactly the map upstream implements: ``mean + L @ qnorm(p)`` where ``L`` is the
        Cholesky factor.

        ``"MultivariateNormal"`` takes `dimension` probabilities; ``"MultivariateStudentT"``
        takes ``dimension + 1`` (the last value drives the chi-squared mixing variable) and still
        returns `dimension` values.

        Parameters
        ----------
        p : array-like of float
            Probabilities in ``(0, 1)`` (see above for the required length).

        Returns
        -------
        numpy.ndarray
            Length :meth:`dimension`.
        """
        return np.asarray(self._run("inverse_cdf", p)["values"])

    def interval(self, lower, upper) -> float:
        """Rectangle probability of a multivariate normal.

        ``P(lower <= X <= upper)``, integrated via the ported Genz MVNDST algorithm. Available
        for ``"MultivariateNormal"`` only.

        Parameters
        ----------
        lower, upper : array-like of float
            Length :meth:`dimension`.

        Returns
        -------
        float
            A single value in ``[0, 1]``.
        """
        lower_v = [float(v) for v in np.asarray(lower, dtype=float).ravel()]
        upper_v = [float(v) for v in np.asarray(upper, dtype=float).ravel()]
        return float(self._run("interval", lower_v + upper_v)["values"][0])

    # -- marginal and conditional -------------------------------------------------------------

    def marginal(self, indices) -> "MultivariateDistribution":
        """Marginal distribution of a multivariate normal.

        Restricts a :func:`mvdist_normal` to a subset of its dimensions. Available for
        ``"MultivariateNormal"`` only; every other family raises naming itself, not a raw C++
        throw.

        Parameters
        ----------
        indices : array-like of int
            The 1-based dimensions to keep.

        Returns
        -------
        MultivariateDistribution
            Over those dimensions.
        """
        idx = _mv_indices(indices, self.dimension(), "indices")
        res = self._run("marginal", idx)
        return MultivariateDistribution._from_spec("MultivariateNormal", res["spec"])

    def conditional(self, given, values) -> "MultivariateDistribution":
        """Conditional distribution of a multivariate normal.

        The distribution of the remaining dimensions of a :func:`mvdist_normal` given fixed
        values for a subset. Available for ``"MultivariateNormal"`` only.

        Parameters
        ----------
        given : array-like of int
            The 1-based dimensions being conditioned on.
        values : array-like of float
            The values `given` is fixed at, the same length as `given`.

        Returns
        -------
        MultivariateDistribution
            Over the complement of `given`.
        """
        given_arr = np.atleast_1d(np.asarray(given))
        values_arr = np.atleast_1d(np.asarray(values, dtype=float))
        if given_arr.size != values_arr.size:
            raise ValueError("`given` and `values` must have the same length")
        idx = _mv_indices(given_arr, self.dimension(), "given")
        res = self._run("conditional", idx + [float(v) for v in values_arr])
        return MultivariateDistribution._from_spec("MultivariateNormal", res["spec"])

    # -- simulation -------------------------------------------------------------------------

    def random(self, n: int, seed: int | None = None, method: str = "random"):
        """Draw from a multivariate distribution.

        Simulate from the distribution's own seeded Mersenne Twister stream. A given `seed`
        reproduces the same draws bit-for-bit in R, Python, and the upstream C# library.
        ``method="latin_hypercube"`` draws a Latin hypercube sample instead of ordinary Monte
        Carlo and is available for ``"MultivariateNormal"`` and ``"MultivariateStudentT"`` only;
        it requires an explicit `seed` (there is no clock-seeded LHS upstream).

        Parameters
        ----------
        n : int
            Number of draws.
        seed : int, optional
            Seed for reproducible draws; ``None`` (the default) seeds from the clock, except for
            ``method="latin_hypercube"`` where it is required.
        method : {"random", "latin_hypercube"}, default "random"
            ``"random"`` for ordinary Monte Carlo, or ``"latin_hypercube"``.

        Returns
        -------
        numpy.ndarray
            ``n x dimension``.
        """
        if method not in ("random", "latin_hypercube"):
            raise ValueError('`method` must be "random" or "latin_hypercube"')
        if method == "latin_hypercube" and seed is None:
            raise ValueError(
                '`seed` is required for method = "latin_hypercube"; there is no clock-seeded '
                "Latin hypercube draw upstream"
            )
        s = -1 if seed is None else int(seed)
        core_method = "random_lhs" if method == "latin_hypercube" else "random"
        v = self._run(core_method, [int(n), s])["values"]
        return np.asarray(v).reshape(int(n), self.dimension())

    # -- family-specific parameters -----------------------------------------------------------

    def params(self) -> dict:
        """Family-specific multivariate parameters.

        The scalar or vector parameters specific to a multivariate family, beyond
        mean/covariance: `df` for ``"MultivariateStudentT"``; `alpha` and `alpha_sum` for
        ``"Dirichlet"``; `trials` and `probabilities` for ``"Multinomial"``.

        Returns
        -------
        dict
            The entries relevant to this distribution's family.
        """
        if self._family == "MultivariateStudentT":
            return {"df": float(self._run("degrees_of_freedom")["values"][0])}
        if self._family == "Dirichlet":
            return {
                "alpha": np.asarray(self._run("alpha")["values"]),
                "alpha_sum": float(self._run("alpha_sum")["values"][0]),
            }
        if self._family == "Multinomial":
            trials = int(self._run("number_of_trials")["values"][0])
            mean = np.asarray(self._run("mean")["values"])
            return {"trials": trials, "probabilities": mean / trials}
        raise ValueError(f"params() has no family-specific parameters for '{self._family}'")


def mvdist_normal(
    mean, covariance, seed: int | None = None, max_evaluations: int | None = None,
    abs_error: float | None = None, rel_error: float | None = None,
) -> MultivariateDistribution:
    """Construct a multivariate normal distribution.

    Mirrors the C# ``MultivariateNormal`` class of the Numerics library.

    Parameters
    ----------
    mean : array-like of float
        Means, length ``d``.
    covariance : array-like
        ``d x d`` symmetric positive-definite covariance matrix.
    seed : int, optional
        Seed for the Genz quasi-Monte-Carlo integrator behind :meth:`MultivariateDistribution.cdf`
        at dimension three and above; ``None`` (the default) leaves it clock-seeded. **Without a
        seed, the CDF at dimension >= 3 is not reproducible run to run** (it draws from a
        per-instance Mersenne Twister), so R and Python cannot agree on a value unless `seed` is
        set explicitly.
    max_evaluations, abs_error, rel_error : optional
        Integrator tuning; ``None`` (the default) for each leaves the ported upstream default
        untouched.

    Returns
    -------
    MultivariateDistribution
        Family ``"MultivariateNormal"``.

    Examples
    --------
    >>> mv = mvdist_normal([0, 0], [[1, 0], [0, 1]])
    >>> mv.pdf([0, 0]) > 0
    True
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("`covariance` must be a square matrix")
    mean_v = [float(v) for v in np.asarray(mean, dtype=float).ravel()]
    if len(mean_v) != cov.shape[0]:
        raise ValueError("`mean` and `covariance` must have the same dimension")
    spec: dict = {
        "family": "MultivariateNormal", "mean": mean_v,
        "covariance": [row.tolist() for row in cov],
    }
    if seed is not None:
        spec["seed"] = int(seed)
    if max_evaluations is not None:
        spec["max_evaluations"] = int(max_evaluations)
    if abs_error is not None:
        spec["abs_error"] = float(abs_error)
    if rel_error is not None:
        spec["rel_error"] = float(rel_error)
    return MultivariateDistribution._from_spec("MultivariateNormal", json.dumps(spec))


def mvdist_student_t(df: float, location, scale=None, seed: int | None = None) -> MultivariateDistribution:
    """Construct a multivariate Student-t distribution.

    Mirrors the C# ``MultivariateStudentT`` class of the Numerics library.

    Parameters
    ----------
    df : float
        Degrees of freedom.
    location : array-like of float
        Location parameters, length ``d``.
    scale : array-like, optional
        ``d x d`` scale matrix; ``None`` (the default) uses the identity.
    seed : int, optional
        Seed for reproducible draws from :meth:`MultivariateDistribution.random`; ``None`` (the
        default) leaves it clock-seeded.

    Returns
    -------
    MultivariateDistribution
        Family ``"MultivariateStudentT"``.

    Examples
    --------
    >>> mvdist_student_t(5, [0, 0]).dimension()
    2
    """
    if not np.isfinite(df):
        raise ValueError("`df` must be a single finite number")
    location_v = [float(v) for v in np.asarray(location, dtype=float).ravel()]
    spec: dict = {"family": "MultivariateStudentT", "df": float(df), "location": location_v}
    if scale is not None:
        scale_arr = np.asarray(scale, dtype=float)
        if scale_arr.ndim != 2 or scale_arr.shape[0] != scale_arr.shape[1] or scale_arr.shape[0] != len(location_v):
            raise ValueError("`scale` must be a square matrix matching the length of `location`")
        spec["scale"] = [row.tolist() for row in scale_arr]
    if seed is not None:
        spec["seed"] = int(seed)
    return MultivariateDistribution._from_spec("MultivariateStudentT", json.dumps(spec))


def mvdist_dirichlet(alpha) -> MultivariateDistribution:
    """Construct a Dirichlet distribution.

    Mirrors the C# ``Dirichlet`` class of the Numerics library.

    Parameters
    ----------
    alpha : array-like of float
        Positive concentration parameters.

    Returns
    -------
    MultivariateDistribution
        Family ``"Dirichlet"``.

    Examples
    --------
    >>> mvdist_dirichlet([2, 3, 4]).pdf([0.2, 0.3, 0.5]) > 0
    True
    """
    alpha_v = [float(v) for v in np.asarray(alpha, dtype=float).ravel()]
    spec = {"family": "Dirichlet", "alpha": alpha_v}
    return MultivariateDistribution._from_spec("Dirichlet", json.dumps(spec))


def mvdist_multinomial(trials: int, probabilities) -> MultivariateDistribution:
    """Construct a multinomial distribution.

    Mirrors the C# ``Multinomial`` class of the Numerics library.

    Parameters
    ----------
    trials : int
        Number of trials.
    probabilities : array-like of float
        Category probabilities, summing to one.

    Returns
    -------
    MultivariateDistribution
        Family ``"Multinomial"``.

    Examples
    --------
    >>> mvdist_multinomial(10, [0.2, 0.3, 0.5]).pdf([2, 3, 5]) > 0
    True
    """
    p_v = [float(v) for v in np.asarray(probabilities, dtype=float).ravel()]
    spec = {"family": "Multinomial", "trials": int(trials), "probabilities": p_v}
    return MultivariateDistribution._from_spec("Multinomial", json.dumps(spec))


def mvdist_bivariate_empirical(
    x1, x2, p, x1_transform: str = "None", x2_transform: str = "None", p_transform: str = "None",
) -> MultivariateDistribution:
    """Construct a bivariate empirical distribution.

    Mirrors the C# ``BivariateEmpirical`` class of the Numerics library: a joint distribution
    defined over a grid of two marginal value vectors and a matrix of associated probabilities.

    Parameters
    ----------
    x1, x2 : array-like of float
        Grid values for each dimension.
    p : array-like
        ``len(x1) x len(x2)`` matrix of joint probabilities.
    x1_transform, x2_transform, p_transform : {"None", "Logarithmic", "NormalZ"}
        How each axis is interpolated between.

    Returns
    -------
    MultivariateDistribution
        Family ``"BivariateEmpirical"``. Note: ``pdf`` is an upstream stub (see
        :meth:`MultivariateDistribution.pdf`).

    Examples
    --------
    >>> mv = mvdist_bivariate_empirical([1, 2], [1, 2], [[0.2, 0.3], [0.2, 0.3]])
    >>> mv.cdf([1.5, 1.5]) >= 0
    True
    """
    x1_v = [float(v) for v in np.asarray(x1, dtype=float).ravel()]
    x2_v = [float(v) for v in np.asarray(x2, dtype=float).ravel()]
    p_arr = np.asarray(p, dtype=float)
    if p_arr.shape != (len(x1_v), len(x2_v)):
        raise ValueError("`p` must be a len(x1) x len(x2) matrix")
    spec = {
        "family": "BivariateEmpirical", "x1": x1_v, "x2": x2_v,
        "p": [row.tolist() for row in p_arr],
        "x1_transform": x1_transform, "x2_transform": x2_transform, "p_transform": p_transform,
    }
    return MultivariateDistribution._from_spec("BivariateEmpirical", json.dumps(spec))
