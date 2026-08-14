"""The callback surface: ported Numerics routines whose input is a live Python function rather
than serializable data. Unlike every verb in toolbox.py/gof.py (which pass data through the shared
``toolbox_run`` dispatcher), these go through their own runner
(``core/include/corehydro/numerics/support/callback_runner.hpp``) and its exception guard, reached
by ``callback_math`` in ``bindings/callback.cpp``. Mirrors ``corehydror``'s ``R/callback.R`` verb
for verb.

An exception raised inside ``f`` reaches the caller unchanged: the guard latches the first one, and
the runner rethrows it after the ported routine returns, so an internal C++ error provoked by the
guard's own sentinel can never replace it.
"""

from __future__ import annotations

import json
from typing import Callable, Sequence

import numpy as np

from . import _core

__all__ = ["root_find", "quadrature", "QuadratureResult", "derivative", "gradient", "hessian", "Rng"]

# The handle a callback is given on the core's seeded generator, defined in bindings/callback.cpp
# (see its RngHandle). Re-exported here so a user can type-annotate a proposal or resample
# function, and so the documentation site has something to point at. It has no constructor: the
# only way to get one is to be handed one.
Rng = _core.Rng


class QuadratureResult(float):
    """The value :func:`quadrature` returns: the integral, with the run's report attached.

    A ``float`` subclass, so it is the integral wherever a number is wanted and carries the two
    extra fields where they are wanted. This mirrors ``corehydror``'s ``quadrature()``, which
    returns the integral with ``status`` and ``function_evaluations`` as R attributes.

    Attributes
    ----------
    status : str
        One of ``"Success"``, ``"MaximumFunctionEvaluationsReached"``,
        ``"MaximumIterationsReached"``, ``"Failure"`` or ``"None"`` -- the ported Numerics
        ``IntegrationStatus``.
    function_evaluations : int
        The number of times ``f`` was called.
    standard_error : float
        The rule's own error estimate, the ported ``StandardError``: the square root of the
        accumulated squared differences between the Gauss and Kronrod estimates. Zero when the
        interval never needed subdividing.
    """

    status: str
    function_evaluations: int
    standard_error: float

    def __new__(
        cls,
        value: float,
        status: str,
        function_evaluations: int,
        standard_error: float,
    ) -> "QuadratureResult":
        self = super().__new__(cls, value)
        self.status = status
        self.function_evaluations = int(function_evaluations)
        self.standard_error = float(standard_error)
        return self

    # A float subclass whose __new__ takes more than the value cannot be reconstructed by the
    # default pickle/copy protocol, which calls cls(value) with the one argument __reduce_ex__
    # harvests. Without this, pickle.dumps() and copy.deepcopy() both fail with "__new__() missing
    # required positional arguments" -- and OptimResult, the other object this surface returns,
    # is picklable. Returning the full argument tuple restores both round trips.
    def __getnewargs__(self) -> tuple:
        return (float(self), self.status, self.function_evaluations, self.standard_error)

    def __repr__(self) -> str:
        # The bare float repr would hide the two fields the R twin prints as attributes.
        return (
            f"{float(self)!r} (status={self.status!r}, "
            f"function_evaluations={self.function_evaluations}, "
            f"standard_error={self.standard_error!r})"
        )


def _check_fn(f: object) -> None:
    if not callable(f):
        raise TypeError("`f` must be a function taking a number and returning a single number")


def _check_point(x: object) -> np.ndarray:
    # Rejects NaN AND +/-inf, naming `x`. Kept character for character in step with corehydror's
    # callback_check_point, so the same bad point is refused with the same message in both
    # languages rather than failing deeper in one of them.
    point = np.asarray(x, dtype=float).ravel()
    if point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("`x` must be a non-empty numeric vector of finite values")
    return point


def root_find(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> float:
    """Find a root of a user-written function.

    Solves ``f(x) = 0`` on ``[lower, upper]`` with the ported Numerics Brent root finder. ``f``
    must change sign across the interval.

    Parameters
    ----------
    f : callable
        A function taking one number and returning one number.
    lower, upper : float
        The bracketing interval.
    tolerance : float, optional
        The convergence tolerance on the bracket width. Left unset, the ported Brent solver's own
        default (1e-8) applies; the value is not restated here, so a change to it lands in one
        place.
    max_iterations : int, optional
        The iteration cap; the search raises if it is reached. Left unset, the ported solver's own
        default (1000) applies.

    Returns
    -------
    float
        The root.

    Examples
    --------
    >>> import corehydropy as ch
    >>> round(ch.root_find(lambda x: x**2 - 2, lower=0, upper=2), 6)
    1.414214
    """
    _check_fn(f)
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("`lower` and `upper` must each be a single finite number")
    if lower >= upper:
        raise ValueError("`lower` must be below `upper`")
    # An option key is written ONLY when the caller supplied it, so an unset argument reaches the
    # ported routine's own default rather than a copy of that default made here. Mirrors
    # corehydror's root_find() and the same rule in callback/math.hpp and the oracle emitter.
    options: dict[str, float | int] = {"lower": lower, "upper": upper}
    if tolerance is not None:
        if float(tolerance) <= 0:
            raise ValueError("`tolerance` must be a single positive number")
        options["tolerance"] = float(tolerance)
    if max_iterations is not None:
        if int(max_iterations) < 1:
            raise ValueError("`max_iterations` must be a single positive integer")
        options["max_iterations"] = int(max_iterations)
    return float(_core.callback_math("root_find", json.dumps(options), f)["values"][0])


def quadrature(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
    max_function_evaluations: int | None = None,
) -> QuadratureResult:
    """Integrate a user-written function over a finite interval.

    Computes the definite integral of ``f`` over ``[lower, upper]`` with the ported Numerics
    adaptive Gauss-Kronrod rule (10-point Gauss, 21-point Kronrod), which subdivides the interval
    until the two nested estimates agree to the requested tolerance.

    Named ``quadrature`` rather than ``integrate`` to stay in step with ``corehydror``, where the
    latter would mask ``stats::integrate``.

    Parameters
    ----------
    f : callable
        A function taking one number and returning one number.
    lower, upper : float
        The limits of integration. ``upper`` must be above ``lower``; neither may be infinite.
    absolute_tolerance, relative_tolerance : float, optional
        The convergence tolerances on the difference between the Gauss and Kronrod estimates.
        Each must lie between 1e-15 and 1. Left unset, the ported integrator's own defaults
        (1e-8) apply.
    max_function_evaluations : int, optional
        The cap on evaluations of ``f``. Reaching it stops the subdivision and is reported in the
        status rather than raising. Left unset, the ported integrator's own default applies.

    Returns
    -------
    QuadratureResult
        The integral, a ``float`` carrying ``status``, ``function_evaluations`` and
        ``standard_error``.

    Examples
    --------
    >>> import corehydropy as ch
    >>> round(ch.quadrature(lambda x: x**2, 0, 3), 10)
    9.0
    >>> ch.quadrature(lambda x: x**2, 0, 3).status
    'Success'
    """
    _check_fn(f)
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("`lower` and `upper` must each be a single finite number")
    if lower >= upper:
        raise ValueError("`lower` must be below `upper`")
    # See root_find above: a key is written only when the caller supplied it.
    options: dict[str, float | int] = {"lower": lower, "upper": upper}
    if absolute_tolerance is not None:
        if not 1e-15 <= float(absolute_tolerance) <= 1:
            raise ValueError("`absolute_tolerance` must be a single number between 1e-15 and 1")
        options["absolute_tolerance"] = float(absolute_tolerance)
    if relative_tolerance is not None:
        if not 1e-15 <= float(relative_tolerance) <= 1:
            raise ValueError("`relative_tolerance` must be a single number between 1e-15 and 1")
        options["relative_tolerance"] = float(relative_tolerance)
    if max_function_evaluations is not None:
        if int(max_function_evaluations) < 1:
            raise ValueError("`max_function_evaluations` must be a single positive integer")
        options["max_function_evaluations"] = int(max_function_evaluations)
    res = _core.callback_math("quadrature", json.dumps(options), f)
    return QuadratureResult(res["values"][0], res["status"], res["values"][1], res["values"][2])


def derivative(f: Callable[[float], float], x: float, step_size: float | None = None) -> float:
    """Take the first derivative of a single-variable function by central difference.

    Parameters
    ----------
    f : callable
        A function taking one number and returning one number.
    x : float
        The point to differentiate at.
    step_size : float, optional
        The finite-difference step. Left unset, or given any value at or below zero, the ported
        routine's adaptive step ``eps**(1/2) * (1 + abs(x))`` is used.

    Returns
    -------
    float
        The derivative.

    Examples
    --------
    >>> import corehydropy as ch
    >>> round(ch.derivative(lambda x: x**3, 2), 6)
    12.0
    """
    _check_fn(f)
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("`x` must be a single finite number")
    # See root_find above: `step_size` is written only when the caller supplied it.
    options: dict[str, float] = {"point": x}
    if step_size is not None:
        options["step_size"] = float(step_size)
    return float(_core.callback_math("derivative", json.dumps(options), f)["values"][0])


def gradient(f: Callable[[Sequence[float]], float], x: Sequence[float]) -> np.ndarray:
    """Take the gradient of a function of a parameter vector.

    Parameters
    ----------
    f : callable
        A function taking a numeric vector and returning one number.
    x : array_like
        The point to differentiate at.

    Returns
    -------
    numpy.ndarray
        The gradient, the length of ``x``.

    Examples
    --------
    >>> import corehydropy as ch
    >>> ch.gradient(lambda p: (1 - p[0])**2 + 100 * (p[1] - p[0]**2)**2, [1.0, 1.0]).round(6)
    array([-0.,  0.])
    """
    _check_fn(f)
    point = _check_point(x)
    options = {"point": point.tolist()}
    res = _core.callback_math("gradient", json.dumps(options), f)
    return np.asarray(res["values"], dtype=float)


def hessian(f: Callable[[Sequence[float]], float], x: Sequence[float]) -> np.ndarray:
    """Take the Hessian matrix of a function of a parameter vector.

    Parameters
    ----------
    f : callable
        A function taking a numeric vector and returning one number.
    x : array_like
        The point to differentiate at.

    Returns
    -------
    numpy.ndarray
        The square symmetric Hessian.

    Examples
    --------
    >>> import corehydropy as ch
    >>> ch.hessian(lambda p: p[0]**2 + 2 * p[1]**2 + p[0] * p[1], [1.0, 2.0]).round(3)
    array([[2., 1.],
           [1., 4.]])
    """
    _check_fn(f)
    point = _check_point(x)
    options = {"point": point.tolist()}
    res = _core.callback_math("hessian", json.dumps(options), f)
    return np.asarray(res["values"], dtype=float).reshape(res["dims"])


def _rng_probe(seed: float, parameters: Sequence[float], f: Callable) -> list:
    """Internal, test-only: seed a generator, hand ``f`` a handle on it, return what ``f`` drew.

    ``f`` takes ``(parameters, rng)``, the Gibbs proposal's own signature, so what the fixtures
    prove here about the handle carries over to the samplers. Not public -- a user reaches the
    handle through a real verb, never through this. Mirrors ``corehydror``'s ``rng_probe()``.
    """
    if not callable(f):
        raise TypeError("`f` must be a function taking (parameters, rng)")
    options = {"seed": float(seed), "parameters": [float(v) for v in parameters]}
    return list(_core.rng_probe(json.dumps(options), f)["values"])
