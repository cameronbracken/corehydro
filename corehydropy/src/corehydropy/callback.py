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

__all__ = ["root_find", "quadrature", "QuadratureResult", "derivative", "gradient", "hessian"]


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
    """

    status: str
    function_evaluations: int

    def __new__(cls, value: float, status: str, function_evaluations: int) -> "QuadratureResult":
        self = super().__new__(cls, value)
        self.status = status
        self.function_evaluations = int(function_evaluations)
        return self


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
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
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
        The convergence tolerance on the bracket width.
    max_iterations : int, optional
        The iteration cap; the search raises if it is reached.

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
    if float(tolerance) <= 0:
        raise ValueError("`tolerance` must be a single positive number")
    if int(max_iterations) < 1:
        raise ValueError("`max_iterations` must be a single positive integer")
    options = {
        "lower": lower,
        "upper": upper,
        "tolerance": float(tolerance),
        "max_iterations": int(max_iterations),
    }
    return float(_core.callback_math("root_find", json.dumps(options), f)["values"][0])


def quadrature(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-8,
    max_function_evaluations: int = 10000000,
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
        Each must lie between 1e-15 and 1.
    max_function_evaluations : int, optional
        The cap on evaluations of ``f``. Reaching it stops the subdivision and is reported in the
        status rather than raising.

    Returns
    -------
    QuadratureResult
        The integral, a ``float`` carrying ``status`` and ``function_evaluations``.

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
    if not 1e-15 <= float(absolute_tolerance) <= 1:
        raise ValueError("`absolute_tolerance` must be a single number between 1e-15 and 1")
    if not 1e-15 <= float(relative_tolerance) <= 1:
        raise ValueError("`relative_tolerance` must be a single number between 1e-15 and 1")
    if int(max_function_evaluations) < 1:
        raise ValueError("`max_function_evaluations` must be a single positive integer")
    options = {
        "lower": lower,
        "upper": upper,
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "max_function_evaluations": int(max_function_evaluations),
    }
    res = _core.callback_math("quadrature", json.dumps(options), f)
    return QuadratureResult(res["values"][0], res["status"], res["values"][1])


def derivative(f: Callable[[float], float], x: float, step_size: float = -1.0) -> float:
    """Take the first derivative of a single-variable function by central difference.

    Parameters
    ----------
    f : callable
        A function taking one number and returning one number.
    x : float
        The point to differentiate at.
    step_size : float, optional
        The finite-difference step. The default, any value at or below zero, selects the adaptive
        step ``eps**(1/2) * (1 + abs(x))``.

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
    options = {"point": x, "step_size": float(step_size)}
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
