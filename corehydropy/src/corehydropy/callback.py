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

__all__ = ["root_find", "derivative", "gradient", "hessian"]


def _check_fn(f: object) -> None:
    if not callable(f):
        raise TypeError("`f` must be a function taking a number and returning a single number")


def _check_point(x: object) -> np.ndarray:
    point = np.asarray(x, dtype=float).ravel()
    if point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("`x` must be a non-empty numeric vector with no missing values")
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
