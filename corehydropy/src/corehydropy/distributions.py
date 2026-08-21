"""Public univariate distribution interface.

A thin, stateless wrapper over the factory-dispatched ``_core.dist_*`` glue: every flat-family
method re-dispatches with ``(family, params)``, so no C++ object lifetime leaks into Python. The
five composite families -- ``TruncatedDistribution``, ``Mixture``, ``CompetingRisks``,
``Empirical``, ``KernelDensity`` -- are built by :func:`dist_truncated`/:func:`dist_mixture`/
:func:`dist_competing_risks`/:func:`dist_empirical`/:func:`dist_kde`: a composite
:class:`Distribution` carries a serialized spec instead of a flat parameter vector, and every
method routes it through ``_core.dist_spec_run`` -- the shared C++ ``run_dist`` in
``core/include/corehydro/numerics/distributions/support/dist_runner.hpp`` -- over the identical
grammar ``dist_spec.hpp`` builds from. Mirrors ``corehydror``'s ``R/distribution.R`` name for
name.
"""

from __future__ import annotations

import json

import numpy as np

from . import _core

__all__ = [
    "Distribution",
    "distribution_names",
    "dist_truncated",
    "dist_mixture",
    "dist_competing_risks",
    "dist_empirical",
    "dist_kde",
]

# The five composite families: no flat parameter vector, built by their own constructor instead
# of Distribution(family, params).
_STRUCTURED_FAMILIES = (
    "TruncatedDistribution", "Mixture", "CompetingRisks", "Empirical", "KernelDensity",
)
_STRUCTURED_CTORS = {
    "TruncatedDistribution": "dist_truncated()",
    "Mixture": "dist_mixture()",
    "CompetingRisks": "dist_competing_risks()",
    "Empirical": "dist_empirical()",
    "KernelDensity": "dist_kde()",
}


def distribution_names(kind: str = "flat") -> list[str]:
    """List the supported distribution families.

    Parameters
    ----------
    kind : {"flat", "structured", "all"}, default "flat"
        ``"flat"`` for families :class:`Distribution` constructs directly from a parameter
        vector; ``"structured"`` for the five composite families constructed by
        :func:`dist_truncated`, :func:`dist_mixture`, :func:`dist_competing_risks`,
        :func:`dist_empirical`, and :func:`dist_kde`; or ``"all"`` for the union of both.

    Returns
    -------
    list of str
        The family names, matching the C# class names of the USACE-RMC Numerics library (for
        example ``"Normal"``, ``"LogNormal"``, ``"Gumbel"``, ``"GeneralizedExtremeValue"``).
    """
    if kind not in ("flat", "structured", "all"):
        raise ValueError('`kind` must be one of "flat", "structured", "all"')
    flat = [f for f in _core.dist_names() if f not in _STRUCTURED_FAMILIES]
    if kind == "flat":
        return flat
    if kind == "structured":
        return list(_STRUCTURED_FAMILIES)
    return flat + list(_STRUCTURED_FAMILIES)


class Distribution:
    """A univariate distribution from the ported Numerics library.

    The 38 families of the USACE-RMC Numerics library that take a flat
    parameter vector are all supported, from ``Normal`` and ``Gumbel`` through
    ``GeneralizedNormal`` (the three-parameter LogNormal) and ``KappaFour``;
    :func:`distribution_names` returns them. The five composite families it
    lists under ``"structured"`` have no flat parameter vector and are built by
    :func:`dist_truncated`, :func:`dist_mixture`, :func:`dist_competing_risks`,
    :func:`dist_empirical`, and :func:`dist_kde` instead.

    Parameters are positional, in the same order as the C# constructor for
    the family (for example ``Normal`` takes ``[mean, sd]`` and
    ``GeneralizedExtremeValue`` takes ``[location, scale, shape]``). Use
    :attr:`parameter_names` to see the names for a family.

    All numeric methods evaluate in the shared C++ core, so results are
    identical to the R package and to the upstream C# library.

    Parameters
    ----------
    family : str
        Distribution family name; see :func:`distribution_names`.
    params : array-like of float
        Parameter vector, in constructor order.

    Examples
    --------
    >>> d = Distribution("Normal", [100, 15])
    >>> d.cdf(100)
    0.5
    >>> d.random(3, seed=123)  # doctest: +SKIP
    array([107.71408450, 108.43058699,  91.52951859])

    GeneralizedNormal is the three-parameter LogNormal: at shape 0 it is the Normal.

    >>> Distribution("GeneralizedNormal", [100, 15, 0]).cdf(100)
    0.5
    """

    def __init__(self, family: str, params) -> None:
        if not isinstance(family, str):
            raise TypeError("family must be a distribution name; see distribution_names()")
        if family in _STRUCTURED_FAMILIES:
            raise ValueError(
                f"'{family}' has no flat parameter vector; use {_STRUCTURED_CTORS[family]} instead"
            )
        if family not in _core.dist_names():
            raise ValueError(
                f"unknown distribution family '{family}'; "
                "see distribution_names() for the supported names"
            )
        params = [float(v) for v in np.asarray(params, dtype=float).ravel()]
        expected = _core.dist_parameter_names(family)["short"]
        if expected and len(params) != len(expected):
            raise ValueError(
                f"'{family}' expects {len(expected)} parameters "
                f"({', '.join(expected)}), got {len(params)}"
            )
        self._family = family
        self._params = params
        self._spec: dict | None = None

    @classmethod
    def _from_spec(cls, family: str, spec: dict) -> "Distribution":
        """Internal: wrap an already-assembled composite spec (see dist_truncated() and
        friends). `spec` is the FULL dist_spec.hpp grammar object, including "family"."""
        self = cls.__new__(cls)
        self._family = family
        self._params = None
        self._spec = spec
        return self

    # -- properties -------------------------------------------------------------------

    @property
    def family(self) -> str:
        """str: The distribution family name."""
        return self._family

    @property
    def params(self) -> list[float]:
        """list of float: The parameter vector, in constructor order."""
        if self._spec is not None:
            vals, _ = self._run_spec("parameters")
            return list(vals)
        return list(self._params)

    @property
    def parameter_names(self) -> dict[str, list[str]]:
        """dict: Parameter names, keys ``"full"`` and ``"short"``."""
        return _core.dist_parameter_names(self._family)

    @property
    def is_valid(self) -> bool:
        """bool: Whether the parameters are valid for this family."""
        if self._spec is not None:
            vals, _ = self._run_spec("parameters_valid")
            return bool(vals[0])
        return _core.dist_valid(self._family, self._params)

    @property
    def is_composite(self) -> bool:
        """bool: Whether this is one of the five composite families (see :func:`dist_truncated`
        and friends), carrying a serialized spec rather than a flat parameter vector."""
        return self._spec is not None

    # -- distribution functions ---------------------------------------------------------

    def pdf(self, x):
        """Probability density at ``x``.

        Parameters
        ----------
        x : float or array-like
            Quantiles.

        Returns
        -------
        float or numpy.ndarray
            Density values, scalar in, scalar out.
        """
        if self._spec is not None:
            return self._vectorized_spec("pdf", x)
        return self._vectorized(_core.dist_pdf_v, x)

    def log_pdf(self, x):
        """Natural log of the probability density at ``x``."""
        if self._spec is not None:
            return self._vectorized_spec("log_pdf", x)
        return self._vectorized(_core.dist_log_pdf_v, x)

    def cdf(self, x):
        """Cumulative distribution at ``x``."""
        if self._spec is not None:
            return self._vectorized_spec("cdf", x)
        return self._vectorized(_core.dist_cdf_v, x)

    def quantile(self, p):
        """Quantile (inverse CDF) at probability ``p`` in ``(0, 1)``."""
        if self._spec is not None:
            return self._vectorized_spec("quantile", p)
        return self._vectorized(_core.dist_quantile_v, p)

    def random(self, n: int, seed: int | None = None):
        """Draw ``n`` random values.

        Draws come from the same seeded Mersenne Twister stream as the C#
        ``GenerateRandomValues(sampleSize, seed)``: a given ``seed``
        reproduces the C# draws bit-for-bit (and matches ``corehydror``
        exactly).

        Parameters
        ----------
        n : int
            Number of draws.
        seed : int, optional
            Seed for reproducible draws; ``None`` (the default) seeds from
            the clock.

        Returns
        -------
        numpy.ndarray
            The ``n`` draws.
        """
        s = -1 if seed is None else int(seed)
        if self._spec is not None:
            vals, _ = self._run_spec("random", [int(n), s])
            return np.asarray(vals)
        return np.asarray(_core.dist_random(self._family, self._params, int(n), s))

    # -- properties of the distribution --------------------------------------------------

    def moments(self) -> dict[str, float]:
        """Moments and support of the distribution.

        Returns
        -------
        dict
            Keys ``mean``, ``median``, ``mode``, ``sd``, ``skewness``,
            ``kurtosis``, ``minimum``, ``maximum``; values are ``nan``
            where undefined.
        """
        if self._spec is not None:
            vals, names = self._run_spec("moments")
            return dict(zip(names, vals))
        m = _core.dist_moments(self._family, self._params)
        order = ["mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"]
        return {k: m[k] for k in order}

    def linear_moments(self) -> list[float]:
        """The first four L-moments.

        Raises
        ------
        ValueError
            If the family has no L-moment support (every composite family: no composite
            distribution implements ``ILinearMomentEstimation`` upstream).
        """
        if self._spec is not None:
            raise ValueError(
                f"linear moments are not available for '{self._family}'; no composite "
                "distribution implements ILinearMomentEstimation upstream"
            )
        return list(_core.dist_linear_moments(self._family, self._params))

    def log_likelihood(self, data) -> float:
        """Log-likelihood of ``data`` under the distribution.

        Parameters
        ----------
        data : array-like of float
            Observations.
        """
        if self._spec is not None:
            vals, _ = self._run_spec("log_likelihood", _as_list(data))
            return float(vals[0])
        return _core.dist_log_likelihood(self._family, self._params, _as_list(data))

    # -- estimation -----------------------------------------------------------------------

    @classmethod
    def fit(cls, family: str, data, method: str = "mle") -> "Distribution":
        """Fit a distribution family to data.

        Mirrors the C# ``Estimate(data, ParameterEstimationMethod)`` API of
        the Numerics library.

        Parameters
        ----------
        family : str
            Distribution family name; see :func:`distribution_names`.
        data : array-like of float
            Observations.
        method : {"mle", "lmom", "mom"}
            Estimation method: maximum likelihood (default), L-moments, or
            product moments. Not every family supports every method;
            unsupported combinations raise.

        Returns
        -------
        Distribution
            The fitted distribution.
        """
        params = _core.dist_fit(family, _as_list(data), method)
        return cls(family, params)

    # -- serialization ----------------------------------------------------------------------

    def to_json(self) -> str:
        """This distribution as the JSON spec the shared C++ core parses.

        A composite (see :func:`dist_truncated` and friends) returns its stored spec; a flat
        family returns ``{"family": ..., "parameters": [...]}``. Matches the grammar
        ``corehydror``'s ``to_spec_json()`` emits, so a :class:`Distribution` embeds directly as
        a nested component of another composite, a :class:`~corehydropy.copula.Copula` marginal,
        or a :class:`~corehydropy.mvdist.MultivariateDistribution`.

        Returns
        -------
        str
            The spec JSON.
        """
        if self._spec is not None:
            return json.dumps(self._spec)
        return json.dumps({"family": self._family, "parameters": list(self._params)})

    # -- misc -------------------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._spec is not None:
            payload = json.dumps(self._spec)
            if len(payload) > 80:
                payload = payload[:77] + "..."
            return f"Distribution({self._family} (composite): {payload})"
        names = self.parameter_names["short"]
        if len(names) == len(self._params):
            inner = ", ".join(f"{n} = {v:g}" for n, v in zip(names, self._params))
        else:
            inner = ", ".join(f"{v:g}" for v in self._params)
        return f"Distribution({self._family}({inner}))"

    def _vectorized(self, fn, values):
        arr = np.asarray(values, dtype=float)
        out = np.asarray(fn(self._family, self._params, [float(v) for v in arr.ravel()]))
        if arr.ndim == 0:
            return float(out[0])
        return out.reshape(arr.shape)

    def _run_spec(self, method: str, args=()):
        """Internal: run one method against a composite spec through the shared runner. `args`
        is always serialized as a JSON array, even when it has length one."""
        res = _core.dist_spec_run(self.to_json(), method, json.dumps([float(a) for a in args]))
        return res["values"], res["names"]

    def _vectorized_spec(self, method: str, values):
        arr = np.asarray(values, dtype=float)
        vals, _ = self._run_spec(method, [float(v) for v in arr.ravel()])
        out = np.asarray(vals)
        if arr.ndim == 0:
            return float(out[0])
        return out.reshape(arr.shape)


def _as_list(data) -> list[float]:
    return [float(v) for v in np.asarray(data, dtype=float).ravel()]


def _single_number(value, message: str, finite: bool = True) -> float:
    """Internal: R's `!is.numeric(x) || length(x) != 1L || !is.finite(x)` guard, spelled for
    Python and shared with copula.py and mvdist.py so both languages reject the same arguments
    with the same wording. Without it a list, an array, or a string reaches numpy or float()
    first, and the user gets "The truth value of an array with more than one element is
    ambiguous" or "float() argument must be a string or a real number" instead of a message
    naming the argument."""
    if isinstance(value, (str, bytes)) or np.ndim(value) != 0:
        raise ValueError(message)
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if finite and not np.isfinite(out):
        raise ValueError(message)
    return out


def _as_spec(d: Distribution) -> dict:
    """Internal: a Distribution's own spec grammar, parsed back to a dict so it can be embedded
    as a nested component/base/marginal of another spec (dist_truncated(), dist_mixture(), ...,
    and Copula's margin_x/margin_y) without a redundant dumps/loads round trip at the top level."""
    if not isinstance(d, Distribution):
        raise TypeError("every distribution argument must be a Distribution")
    return json.loads(d.to_json())


# --- the five composite families -----------------------------------------------------------
#
# Each mirrors one dist_*() constructor in corehydror's R/distribution.R: build the
# dist_spec.hpp grammar object and wrap it in a Distribution that carries no flat parameter
# vector, so it routes through _core.dist_spec_run for every subsequent verb.

_DEPENDENCY_CHOICES = ("Independent", "PerfectlyPositive", "PerfectlyNegative", "CorrelationMatrix")
_P_TRANSFORM_CHOICES = ("NormalZ", "None")
_KERNEL_CHOICES = ("Gaussian", "Epanechnikov", "Triangular", "Uniform")


def dist_truncated(d: Distribution, min: float, max: float) -> Distribution:
    """Truncate a distribution.

    Restricts a distribution to ``[min, max]``, renormalizing its density over the truncated
    range. Mirrors the C# ``TruncatedDistribution`` composite.

    Parameters
    ----------
    d : Distribution
        The base (untruncated) distribution.
    min, max : float
        The truncation bounds, with ``min < max``.

    Returns
    -------
    Distribution
        Family ``"TruncatedDistribution"``, accepted by every method every other
        :class:`Distribution` accepts.

    Examples
    --------
    >>> d = dist_truncated(Distribution("Normal", [2, 1]), min=1.1, max=2.11)
    >>> d.pdf(1.5) > 0
    True
    """
    if not isinstance(d, Distribution):
        raise TypeError("`d` must be a Distribution")
    message = "`min` and `max` must each be a single number"
    min = _single_number(min, message, finite=False)
    max = _single_number(max, message, finite=False)
    if min >= max:
        raise ValueError("`min` must be less than `max`")
    spec = {"family": "TruncatedDistribution", "base": _as_spec(d), "bounds": [min, max]}
    return Distribution._from_spec("TruncatedDistribution", spec)


def dist_mixture(components, weights, zero_inflated: bool = False, zero_weight: float = 0.0) -> Distribution:
    """Mixture distribution.

    A weighted mixture of component distributions, optionally zero-inflated. Mirrors the C#
    ``Mixture`` composite.

    Parameters
    ----------
    components : sequence of Distribution
        The mixture components.
    weights : array-like of float
        Mixture weights, the same length as `components`.
    zero_inflated : bool, default False
        Whether the mixture places extra probability mass at zero.
    zero_weight : float, default 0.0
        The probability mass at zero when `zero_inflated` is ``True``; ignored otherwise.

    Returns
    -------
    Distribution
        Family ``"Mixture"``, accepted by every method every other :class:`Distribution` accepts.

    Examples
    --------
    >>> d = dist_mixture(
    ...     [Distribution("Normal", [0, 1]), Distribution("Normal", [5, 1])],
    ...     weights=[0.5, 0.5],
    ... )
    >>> d.pdf(2.5) > 0
    True
    """
    if not isinstance(components, (list, tuple)) or len(components) == 0:
        raise ValueError("`components` must be a non-empty list of Distribution objects")
    if not all(isinstance(c, Distribution) for c in components):
        raise TypeError("every element of `components` must be a Distribution")
    weights_v = [float(w) for w in np.asarray(weights, dtype=float).ravel()]
    if len(weights_v) != len(components):
        raise ValueError("`weights` must have the same length as `components`")
    spec = {
        "family": "Mixture",
        "components": [_as_spec(c) for c in components],
        "weights": weights_v,
        "zero_inflated": bool(zero_inflated),
        "zero_weight": float(zero_weight),
    }
    return Distribution._from_spec("Mixture", spec)


def dist_competing_risks(
    components, minimum_of: bool = True, dependency: str = "Independent", correlation=None,
) -> Distribution:
    """Competing-risks distribution.

    The distribution of the minimum (a series system) or maximum (a parallel system) of several
    component random variables, with an optional dependency structure. Mirrors the C#
    ``CompetingRisks`` composite.

    Parameters
    ----------
    components : sequence of Distribution
        The component distributions.
    minimum_of : bool, default True
        ``True`` for the distribution of the minimum of `components`; ``False`` for the maximum.
    dependency : {"Independent", "PerfectlyPositive", "PerfectlyNegative", "CorrelationMatrix"}
        The dependency structure between components.
    correlation : array-like, optional
        A square correlation matrix; required when ``dependency = "CorrelationMatrix"``, ignored
        otherwise.

    Returns
    -------
    Distribution
        Family ``"CompetingRisks"``, accepted by every method every other :class:`Distribution`
        accepts.

    Examples
    --------
    >>> d = dist_competing_risks(
    ...     [Distribution("Weibull", [1, 2]), Distribution("Weibull", [1, 3])]
    ... )
    >>> d.cdf(1.5) > 0
    True
    """
    if not isinstance(components, (list, tuple)) or len(components) == 0:
        raise ValueError("`components` must be a non-empty list of Distribution objects")
    if not all(isinstance(c, Distribution) for c in components):
        raise TypeError("every element of `components` must be a Distribution")
    if dependency not in _DEPENDENCY_CHOICES:
        raise ValueError(f"`dependency` must be one of {', '.join(_DEPENDENCY_CHOICES)}")
    spec = {
        "family": "CompetingRisks",
        "components": [_as_spec(c) for c in components],
        "minimum_of_random_variables": bool(minimum_of),
        "dependency": dependency,
    }
    if correlation is not None:
        arr = np.asarray(correlation, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("`correlation` must be a square matrix")
        spec["correlation"] = [row.tolist() for row in arr]
    return Distribution._from_spec("CompetingRisks", spec)


def dist_empirical(x, p, p_transform: str = "NormalZ", p_descending: bool = False) -> Distribution:
    """Empirical distribution.

    A distribution defined by paired value/probability points, interpolated between. Mirrors
    the C# ``EmpiricalDistribution`` composite.

    Parameters
    ----------
    x : array-like of float
        Values.
    p : array-like of float
        Associated probabilities, the same length as `x`.
    p_transform : {"NormalZ", "None"}, default "NormalZ"
        How `p` is interpolated between: ``"NormalZ"`` transforms through the standard-normal
        quantile; ``"None"`` interpolates `p` directly.
    p_descending : bool, default False
        Whether `p` decreases as `x` increases (a survival-function encoding); ``False`` is the
        ordinary ascending-CDF case.

    Returns
    -------
    Distribution
        Family ``"Empirical"``, accepted by every method every other :class:`Distribution`
        accepts.

    Examples
    --------
    >>> d = dist_empirical(x=[1, 2, 3], p=[0.1, 0.5, 0.9])
    >>> d.quantile(0.5)
    2.0
    """
    if p_transform not in _P_TRANSFORM_CHOICES:
        raise ValueError(f"`p_transform` must be one of {', '.join(_P_TRANSFORM_CHOICES)}")
    x_v = [float(v) for v in np.asarray(x, dtype=float).ravel()]
    p_v = [float(v) for v in np.asarray(p, dtype=float).ravel()]
    if len(x_v) != len(p_v):
        raise ValueError("`x` and `p` must have the same length")
    spec = {
        "family": "Empirical", "x": x_v, "p": p_v,
        "p_transform": p_transform, "p_descending": bool(p_descending),
    }
    return Distribution._from_spec("Empirical", spec)


def dist_kde(data, kernel: str = "Gaussian", bandwidth: float | None = None, bounded_by_data: bool = True) -> Distribution:
    """Kernel density distribution.

    A nonparametric density estimate built from a sample by summing a kernel centered at each
    observation. Mirrors the C# ``KernelDensity`` composite.

    Parameters
    ----------
    data : array-like of float
        Observations the density is built from.
    kernel : {"Gaussian", "Epanechnikov", "Triangular", "Uniform"}, default "Gaussian"
        The kernel shape.
    bandwidth : float, optional
        The kernel bandwidth; ``None`` (the default) uses Silverman's rule of thumb.
    bounded_by_data : bool, default True
        Whether the reported minimum and maximum are the smallest and largest observation
        (``True``, the default) or extend three bandwidths past each. Those bounds gate
        ``Distribution.cdf()`` and ``Distribution.quantile()``. The density is summed
        wherever you ask it, either way.

    Returns
    -------
    Distribution
        Family ``"KernelDensity"``, accepted by every method every other :class:`Distribution`
        accepts.

    Examples
    --------
    >>> d = dist_kde([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> d.cdf(5.5)  # doctest: +SKIP
    0.5
    """
    if kernel not in _KERNEL_CHOICES:
        raise ValueError(f"`kernel` must be one of {', '.join(_KERNEL_CHOICES)}")
    spec = {
        "family": "KernelDensity",
        "data": [float(v) for v in np.asarray(data, dtype=float).ravel()],
        "kernel": kernel,
        "bounded_by_data": bool(bounded_by_data),
    }
    if bandwidth is not None:
        spec["bandwidth"] = float(bandwidth)
    return Distribution._from_spec("KernelDensity", spec)
