"""The bivariate copula surface. Every verb serializes a :class:`Copula` to the
``dist_spec.hpp`` grammar and runs one method through ``_core.copula_run``; nothing holds C++
state. Mirrors ``corehydror``'s ``R/copula.R`` verb for verb: the R module functions
(``copula_pdf()``, ``copula_cdf()``, ...) are methods here instead (``Copula.pdf()``,
``Copula.cdf()``, ...) to match idiomatic Python, and :func:`copula_fit`/:func:`copula_names`
stay module functions since a fit does not start from a live object.
"""

from __future__ import annotations

import json

import numpy as np

from . import _core
from .distributions import Distribution, _single_number

__all__ = ["Copula", "copula_fit", "copula_names"]

_COPULA_FAMILIES = ("AliMikhailHaq", "Clayton", "Frank", "Gumbel", "Joe", "Normal", "StudentT")


def copula_names() -> list[str]:
    """List the supported copula families.

    Returns
    -------
    list of str
        The seven bivariate copula family names.

    Examples
    --------
    >>> copula_names()  # doctest: +SKIP
    ['AliMikhailHaq', 'Clayton', 'Frank', 'Gumbel', 'Joe', 'Normal', 'StudentT']
    """
    return list(_COPULA_FAMILIES)


def _check_margin(m, name: str):
    if m is not None and not isinstance(m, Distribution):
        raise TypeError(f"`{name}` must be a Distribution (see Distribution)")
    return m


def _pairs(u, v, method: str):
    """Internal: recycle a pair of vectors to a common length and lay them out as all `u` then
    all `v`, the split-at-the-halfway-point layout run_copula's pdf / log_pdf / cdf arms read.
    Returns (args, was_scalar)."""
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    scalar = u_arr.ndim == 0 and v_arr.ndim == 0
    u_flat = u_arr.ravel()
    v_flat = v_arr.ravel()
    if u_flat.size == 0 or v_flat.size == 0:
        raise ValueError(f"`u` and `v` must both be non-empty in {method}()")
    n = max(u_flat.size, v_flat.size)
    if n % u_flat.size != 0 or n % v_flat.size != 0:
        raise ValueError(
            f"`u` (length {u_flat.size}) and `v` (length {v_flat.size}) are not recyclable to a "
            "common length"
        )
    uu = np.resize(u_flat, n)
    vv = np.resize(v_flat, n)
    return [float(x) for x in uu] + [float(x) for x in vv], scalar


class Copula:
    """A bivariate copula.

    Stateless: the object holds its spec as a JSON string and every verb runs one method through
    ``_core.copula_run``. Nothing holds C++ state, so instances pickle and compare across
    processes. Mirrors the C# ``BivariateCopula`` hierarchy of the Numerics library
    (``ClaytonCopula``, ``GumbelCopula``, ...).

    `margin_x` and `margin_y` are optional :class:`Distribution` marginals, attached exactly as
    given (with no re-fitting) -- see :func:`copula_fit` for the estimation surface, where a
    marginal can also be given as a bare family-name string to be fitted.

    Parameters
    ----------
    family : str
        One of :func:`copula_names`.
    theta : float
        The dependence parameter.
    df : float, optional
        Degrees of freedom, required for ``"StudentT"`` and ignored otherwise.
    margin_x, margin_y : Distribution, optional
        Marginals, attached exactly as given.

    See Also
    --------
    copula_fit

    Examples
    --------
    >>> Copula("Clayton", theta=2).pdf(0.3, 0.7) > 0
    True
    """

    def __init__(self, family: str, theta: float, df: float | None = None, margin_x=None, margin_y=None):
        if family not in _COPULA_FAMILIES:
            raise ValueError(f"family must be one of {', '.join(_COPULA_FAMILIES)}")
        theta = _single_number(theta, "`theta` must be a single finite number")
        if family == "StudentT" and df is None:
            raise ValueError("df is required for the StudentT copula")
        margin_x = _check_margin(margin_x, "margin_x")
        margin_y = _check_margin(margin_y, "margin_y")
        spec: dict = {"family": family, "theta": float(theta)}
        if df is not None:
            spec["df"] = float(df)
        if margin_x is not None:
            spec["margin_x"] = json.loads(margin_x.to_json())
        if margin_y is not None:
            spec["margin_y"] = json.loads(margin_y.to_json())
        self._family = family
        self._theta = float(theta)
        self._df = None if df is None else float(df)
        self._margin_x = margin_x
        self._margin_y = margin_y
        self._spec = json.dumps(spec)

    @classmethod
    def _from_spec(cls, family, spec_json, theta, df=None, margin_x=None, margin_y=None) -> "Copula":
        """Internal: wrap an already-assembled spec string (see :func:`copula_fit`)."""
        self = cls.__new__(cls)
        self._family = family
        self._theta = theta
        self._df = df
        self._margin_x = margin_x
        self._margin_y = margin_y
        self._spec = spec_json
        return self

    # -- properties -------------------------------------------------------------------

    @property
    def family(self) -> str:
        """str: The copula family name."""
        return self._family

    @property
    def theta(self) -> float:
        """float: The fitted or given dependence parameter."""
        return self._theta

    @property
    def df(self) -> float | None:
        """float or None: Degrees of freedom (``"StudentT"`` only)."""
        return self._df

    @property
    def margin_x(self) -> Distribution | None:
        """Distribution or None: The x marginal, when attached."""
        return self._margin_x

    @property
    def margin_y(self) -> Distribution | None:
        """Distribution or None: The y marginal, when attached."""
        return self._margin_y

    def to_json(self) -> str:
        """This copula's spec as the JSON the shared C++ core parses.

        Returns
        -------
        str
            The spec JSON.
        """
        return self._spec

    def __repr__(self) -> str:
        extra = f", df = {self._df:g}" if self._df is not None else ""
        return f"Copula({self._family}(theta = {self._theta:g}{extra}))"

    def _run(self, method, args=()):
        res = _core.copula_run(self._spec, method, json.dumps([float(a) for a in args]))
        return res["values"], res["names"]

    # -- density, distribution, and inverse ------------------------------------------------

    def pdf(self, u, v):
        """Copula density on the unit square.

        `u` and `v` are recycled to a common length and evaluated pairwise, one returned value
        per pair.

        Parameters
        ----------
        u, v : float or array-like
            Values in ``(0, 1)``, the copula's two arguments.

        Returns
        -------
        float or numpy.ndarray
            One value per recycled ``(u, v)`` pair; a scalar when both `u` and `v` are scalars.
        """
        args, scalar = _pairs(u, v, "pdf")
        vals, _ = self._run("pdf", args)
        return float(vals[0]) if scalar else np.asarray(vals)

    def log_pdf(self, u, v):
        """Copula log-density on the unit square. See :meth:`pdf`."""
        args, scalar = _pairs(u, v, "log_pdf")
        vals, _ = self._run("log_pdf", args)
        return float(vals[0]) if scalar else np.asarray(vals)

    def cdf(self, u, v):
        """Copula distribution function on the unit square. See :meth:`pdf`."""
        args, scalar = _pairs(u, v, "cdf")
        vals, _ = self._run("cdf", args)
        return float(vals[0]) if scalar else np.asarray(vals)

    def inverse_cdf(self, u: float, v: float):
        """Copula inverse CDF.

        Parameters
        ----------
        u, v : float
            Scalars in ``(0, 1)``.

        Returns
        -------
        numpy.ndarray
            Length two.
        """
        vals, _ = self._run("inverse_cdf", [float(u), float(v)])
        return np.asarray(vals)

    # -- dependence and bounds ---------------------------------------------------------------

    def tail_dependence(self) -> dict:
        """Lower and upper tail dependence coefficients.

        Returns
        -------
        dict
            Keys ``"lower"`` and ``"upper"``.
        """
        values, names = self._run("tail_dependence")
        return dict(zip(names, values))

    def exceedance(self, u: float, v: float, type: str = "and") -> float:
        """Joint exceedance probability.

        The probability that both variables exceed their thresholds, ``P(U > u, V > v)``
        (``type = "and"``), or that at least one does, ``P(U > u or V > v) = 1 - C(u, v)``
        (``type = "or"``).

        Parameters
        ----------
        u, v : float
            Scalars in ``(0, 1)``.
        type : {"and", "or"}, default "and"
            ``"and"`` for the joint (both-exceed) probability, ``"or"`` for the union
            (either-exceeds) probability.

        Returns
        -------
        float
            A single value in ``[0, 1]``.
        """
        if type not in ("and", "or"):
            raise ValueError('`type` must be "and" or "or"')
        method = "exceedance_and" if type == "and" else "exceedance_or"
        vals, _ = self._run(method, [float(u), float(v)])
        return float(vals[0])

    def bounds(self) -> dict:
        """The valid range of the dependence parameter for this copula's family.

        Returns
        -------
        dict
            Keys ``"minimum"`` and ``"maximum"``.
        """
        values, names = self._run("bounds")
        return dict(zip(names, values))

    def params(self):
        """The copula's dependence parameter vector.

        Returns
        -------
        numpy.ndarray
            ``theta``, and ``df`` for ``"StudentT"``.
        """
        vals, _ = self._run("parameters")
        return np.asarray(vals)

    # -- simulation and likelihood ------------------------------------------------------------

    def random(self, n: int, seed: int | None = None):
        """Draw from the copula's own seeded Mersenne Twister stream.

        Mapped through the attached marginals to the data scale when both `margin_x` and
        `margin_y` were supplied; on the unit square otherwise. A given `seed` reproduces the
        same draws bit-for-bit in R, Python, and the upstream C# library.

        Parameters
        ----------
        n : int
            Number of draws.
        seed : int, optional
            Seed for reproducible draws; ``None`` (the default) seeds from the clock.

        Returns
        -------
        numpy.ndarray
            ``n x 2``.
        """
        s = -1 if seed is None else int(seed)
        vals, _ = self._run("random", [int(n), s])
        return np.asarray(vals).reshape(2, int(n)).T

    def log_likelihood(self, x, y, method: str = "pseudo") -> float:
        """Copula log-likelihood over a paired sample.

        Three log-likelihoods, differing in how the marginals enter: the pseudo log-likelihood
        works on the data's pseudo-observations (no marginals needed), IFM (inference from
        margins) transforms the raw data through the attached marginal CDFs then evaluates the
        copula density, and the full log-likelihood adds the marginal log-densities to that.
        ``method="ifm"`` and ``"full"`` need `margin_x`/`margin_y` attached.

        `x` and `y` are always raw paired observations on their own data scale. Upstream's pseudo
        log-likelihood is defined on values already on ``(0, 1)``, so ``"pseudo"`` converts the
        sample to its plotting positions, ``rank / (n + 1)``, first; that transform happens
        inside the shared C++ core (the same one :func:`copula_fit`'s ``"mpl"`` fit uses), so R
        and Python return the same number for the same input.

        Parameters
        ----------
        x, y : array-like of float
            Raw paired observations, the same length.
        method : {"pseudo", "ifm", "full"}, default "pseudo"
            Which log-likelihood to evaluate.

        Returns
        -------
        float
        """
        choices = {
            "pseudo": "log_likelihood_pseudo", "ifm": "log_likelihood_ifm", "full": "log_likelihood_full",
        }
        if method not in choices:
            raise ValueError(f"`method` must be one of {', '.join(choices)}")
        x_v = [float(v) for v in np.asarray(x, dtype=float).ravel()]
        y_v = [float(v) for v in np.asarray(y, dtype=float).ravel()]
        if len(x_v) != len(y_v):
            raise ValueError("`x` and `y` must have the same length")
        vals, _ = self._run(choices[method], x_v + y_v)
        return float(vals[0])


def copula_fit(family: str, x, y, method: str = "mpl", margin_x=None, margin_y=None) -> Copula:
    """Fit a bivariate copula to data.

    Estimate a copula's dependence parameter(s) -- and, when a marginal names a family without
    parameters, that marginal's own parameters -- from a paired sample. Mirrors the C#
    ``BivariateCopulaEstimation`` methods of the Numerics library.

    `x` and `y` are the raw paired observations for every method. ``method="mpl"`` maximizes the
    pseudo-likelihood, which is defined on the plotting positions ``rank / (n + 1)`` rather than
    on the data scale; that transform happens inside the shared C++ core, so R and Python fit the
    same numbers from the same input.

    `margin_x` and `margin_y` accept EITHER a family-name string or a :class:`Distribution`, and
    the two are handled differently:

    * a **name** (e.g. ``margin_x="Normal"``) is fitted by maximum likelihood to `x` (or `y`)
      before the copula is estimated. This is what Inference From Margins (``method="ifm"``)
      requires, and it is accepted for ``"ifm"`` and ``"mle"`` only: ``"mpl"`` and ``"tau"``
      ignore the marginals entirely, so a name there would be reported back unfitted -- passing
      one raises.
    * a **:class:`Distribution`** (e.g. ``margin_x=Distribution("Normal", [0, 1])``) is accepted
      for all four methods. ``"mpl"`` and ``"tau"`` attach it untouched (so :meth:`Copula.random`
      can draw on the data scale), ``"ifm"`` takes it as the given margin, and ``"mle"``
      re-estimates it jointly with theta.

    ``method="tau"`` inverts Kendall's tau into theta directly and is only implemented upstream
    for Clayton, Gumbel, and AliMikhailHaq (``SetThetaFromTau``); it raises for every other
    family.

    Parameters
    ----------
    family : str
        One of :func:`copula_names`.
    x, y : array-like of float
        Raw paired observations, the same length.
    method : {"mpl", "ifm", "mle", "tau"}, default "mpl"
        ``"mpl"`` (maximum pseudo-likelihood), ``"ifm"`` (inference from margins), ``"mle"``
        (full maximum likelihood), or ``"tau"`` (Kendall's tau inversion; Clayton, Gumbel, and
        AliMikhailHaq only).
    margin_x, margin_y : str or Distribution, optional
        Optional marginals; see Notes above.

    Returns
    -------
    Copula
        The fitted copula.

    Examples
    --------
    >>> x = [135.9, 104.1, 108.7, 99.3, 134.7, 91.0, 77.3, 115.4, 109.0, 79.0]
    >>> y = [1.9, 1.3, 1.4, 1.2, 1.8, 1.1, 0.9, 1.5, 1.4, 1.0]
    >>> copula_fit("Clayton", x, y, method="mpl").theta > 0  # doctest: +SKIP
    True
    """
    if family not in _COPULA_FAMILIES:
        raise ValueError(f"`family` must be one of {', '.join(_COPULA_FAMILIES)}")
    if method not in ("mpl", "ifm", "mle", "tau"):
        raise ValueError('`method` must be one of "mpl", "ifm", "mle", "tau"')
    x_v = [float(v) for v in np.asarray(x, dtype=float).ravel()]
    y_v = [float(v) for v in np.asarray(y, dtype=float).ravel()]
    if len(x_v) != len(y_v):
        raise ValueError("`x` and `y` must have the same length")

    def margin_spec(m, name):
        if m is None:
            return None
        if isinstance(m, str):
            # "mpl" and "tau" never look at the marginals, so a named family would come back a
            # default Normal(0, 1) presented as though it had been fitted.
            if method in ("mpl", "tau"):
                raise ValueError(
                    f'method = "{method}" does not use marginals, so `{name} = "{m}"` would be '
                    "left unfitted; pass a parameterized Distribution to attach a fixed marginal "
                    'for later sampling, or use method = "ifm" or "mle" to fit one'
                )
            return {"family": m}
        if isinstance(m, Distribution):
            return json.loads(m.to_json())
        raise TypeError(f"`{name}` must be a family-name string or a Distribution")

    fit: dict = {"x": x_v, "y": y_v, "method": method}
    mx_spec = margin_spec(margin_x, "margin_x")
    my_spec = margin_spec(margin_y, "margin_y")
    if mx_spec is not None:
        fit["margin_x"] = mx_spec
    if my_spec is not None:
        fit["margin_y"] = my_spec
    spec_json = json.dumps({"family": family, "fit": fit})

    # One "parameters" call carries theta and, for StudentT, df -- the estimation runs once per
    # runner call, so asking for them separately would refit the copula twice.
    pars = _core.copula_run(spec_json, "parameters", "[]")["values"]
    theta = float(pars[0])
    df = float(pars[1]) if family == "StudentT" else None

    # Read a marginal back as a Distribution only when the fit actually moved it: "mle"
    # re-estimates both marginals jointly, and a named marginal is MLE-fitted before an "ifm"
    # fit. A Distribution under "mpl"/"tau"/"ifm" is used exactly as given, so it is already the
    # answer and needs no second estimation run.
    def fitted_margin(m, which):
        if m is None:
            return None
        if method != "mle" and isinstance(m, Distribution):
            return m
        fam = m if isinstance(m, str) else m.family
        vals = _core.copula_run(spec_json, f"marginal_{which}_parameters", "[]")["values"]
        return Distribution(fam, list(vals))

    return Copula._from_spec(
        family, spec_json, theta, df,
        margin_x=fitted_margin(margin_x, "x"), margin_y=fitted_margin(margin_y, "y"),
    )
