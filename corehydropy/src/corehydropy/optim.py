"""The general-purpose optimizer surface over the six ported Numerics optimizers (DE, BFGS,
Powell, MLSL, Nelder-Mead, Brent). Unlike every other verb in toolbox.py/gof.py (which pass
serializable data through the shared ``toolbox_run`` dispatcher), an optimizer takes a live Python
function, so this goes through its own runner (``core/include/corehydro/numerics/support/
optimizer_runner.hpp``) and its own binding (``optim_run`` in ``bindings/toolbox.cpp``), rather
than ``toolbox_run``. Mirrors ``corehydror``'s ``R/optim.R`` verb for verb.
"""

from __future__ import annotations

import json

import numpy as np

from . import _core

__all__ = ["OptimResult", "optim_minimize", "optim_maximize"]

_CONTROL_KEYS = {
    "max_iterations", "max_function_evaluations", "absolute_tolerance", "relative_tolerance",
    "population_size", "compute_hessian", "report_failure",
}
# Every method needs `lower`/`upper` (even "de" and "brent", which take no `initial`).
# "bfgs", "powell", "mlsl" and "nelder_mead" additionally need an initial guess -- all three of
# `initial`/`lower`/`upper` are then required to be the same length, which the underlying C++
# constructors validate once the request reaches them (NelderMead's ctor does not validate this
# itself -- see optimizer_runner.hpp's "nelder_mead" arm -- so the length check below is load-
# bearing, not just a friendlier error, for that one method).
_NEEDS_INITIAL = {"bfgs", "powell", "mlsl", "nelder_mead"}
# The four methods deriving from the ported Optimizer base (see optimizer_runner.hpp); only these
# accept max_function_evaluations/report_failure/compute_hessian, and only "de"/"mlsl" are
# stochastic (accept a seed).
_BASE_METHODS = {"de", "bfgs", "powell", "mlsl"}
_STOCHASTIC_METHODS = {"de", "mlsl"}
_METHODS = ("de", "bfgs", "powell", "mlsl", "nelder_mead", "brent")
_BASE_ONLY_CONTROL = {"max_function_evaluations", "report_failure", "compute_hessian"}


class OptimResult:
    """The result of :func:`optim_minimize`/:func:`optim_maximize`.

    Carries no C++ state: every field is a plain Python value, so an instance serializes with
    :mod:`pickle`.

    Attributes
    ----------
    parameters : numpy.ndarray
        The optimum.
    value : float
        The objective's own value there, in its own sign convention -- not negated for
        :func:`optim_maximize`.
    iterations : int
    function_evaluations : int
    status : str
    hessian : numpy.ndarray or None
        The numerically differentiated Hessian at the optimum. Computed BY DEFAULT for
        ``"de"``, ``"bfgs"``, ``"powell"``, and ``"mlsl"`` (matching the ported C# ``Optimizer``
        base), by extra objective evaluations -- pass ``control={"compute_hessian": False}`` to
        skip it and those evaluations. Always ``None`` for ``"nelder_mead"`` and ``"brent"``,
        which never compute one.
    """

    __slots__ = ("parameters", "value", "iterations", "function_evaluations", "status", "hessian")

    def __init__(self, **fields) -> None:
        for name in self.__slots__:
            setattr(self, name, fields[name])

    def __repr__(self) -> str:
        return (
            f"<OptimResult {self.status} after {self.iterations} iterations "
            f"({self.function_evaluations} evaluations)>\n"
            f"  value: {self.value:.8g}\n"
            f"  parameters: {np.array2string(self.parameters, precision=6)}"
        )


def _optim_run(objective, lower, upper, initial, method: str, seed, control: dict, maximize: bool) -> OptimResult:
    """Internal: validate everything Python-side, then make one call. Both verbs share this so
    their messages and defaults can never drift apart."""
    if not callable(objective):
        raise TypeError("`objective` must be a callable taking a numeric vector and returning one number")
    if method not in _METHODS:
        raise ValueError(f"`method` must be one of {_METHODS}; got {method!r}")
    if lower is None or upper is None:
        raise ValueError(f"method {method!r} needs `lower` and `upper` bounds")
    lower_a = np.asarray(lower, dtype=float).ravel()
    upper_a = np.asarray(upper, dtype=float).ravel()
    if lower_a.size != upper_a.size:
        raise ValueError(
            f"`lower` and `upper` must have the same length; got {lower_a.size} and {upper_a.size}"
        )
    if np.any(lower_a >= upper_a):
        raise ValueError("every `lower` bound must be below its `upper` bound")
    if method in _NEEDS_INITIAL and initial is None:
        raise ValueError(f"method {method!r} needs `initial` starting values")
    initial_a = None
    if initial is not None:
        initial_a = np.asarray(initial, dtype=float).ravel()
        if initial_a.size != lower_a.size:
            raise ValueError(
                f"`initial` must have the same length as `lower`/`upper`; got {initial_a.size} "
                f"and {lower_a.size}"
            )
    if seed is not None and method not in _STOCHASTIC_METHODS:
        raise ValueError(
            f"`seed` only applies to the stochastic methods {sorted(_STOCHASTIC_METHODS)}; "
            f"got method {method!r}"
        )
    unknown = set(control) - _CONTROL_KEYS
    if unknown:
        raise ValueError(
            f"unknown control name(s): {sorted(unknown)}. Available: {sorted(_CONTROL_KEYS)}"
        )
    if method not in _BASE_METHODS:
        base_only = set(control) & _BASE_ONLY_CONTROL
        if base_only:
            raise ValueError(
                f"control name(s) {sorted(base_only)} only apply to method(s) "
                f"{sorted(_BASE_METHODS)}; got method {method!r}"
            )
    # `population_size` is only ever applied in the "de" arm of the C++ runner (every other
    # method silently ignores it); reject it here rather than let it look like it did something.
    if method != "de" and "population_size" in control:
        raise ValueError(
            f"control name 'population_size' only applies to method 'de'; got method {method!r}"
        )
    # `initial` is only read by the "bfgs"/"powell"/"mlsl"/"nelder_mead" arms; "de" and "brent"
    # never look at it, so a caller-supplied `initial` for either would silently do nothing.
    if initial is not None and method not in _NEEDS_INITIAL:
        raise ValueError(
            f"`initial` only applies to method(s) {sorted(_NEEDS_INITIAL)}; got method {method!r}"
        )

    spec: dict = {"method": method, "maximize": bool(maximize)}
    spec["lower"] = lower_a.tolist()
    spec["upper"] = upper_a.tolist()
    if initial_a is not None:
        spec["initial"] = initial_a.tolist()
    if seed is not None:
        spec["seed"] = int(seed)
    if control:
        spec["control"] = control

    # The return value is deliberately NOT converted/validated here -- it is handed to the C++
    # binding's py::cast<double> as-is, so a wrong-shape return raises the SAME
    # "the objective must return a single number" error the R glue raises (see
    # bindings/toolbox.cpp's optim_run), rather than a generic Python TypeError from a premature
    # float() call on this side.
    def _objective(p):
        return objective(np.asarray(p, dtype=float))

    r = _core.optim_run(json.dumps(spec), _objective)
    hessian = None
    if r["hessian"]:
        hessian = np.asarray(r["hessian"], dtype=float).reshape(r["hessian_dims"][0], r["hessian_dims"][1])
    return OptimResult(
        parameters=np.asarray(r["parameters"], dtype=float),
        value=float(r["value"]),
        iterations=int(r["iterations"]),
        function_evaluations=int(r["function_evaluations"]),
        status=r["status"],
        hessian=hessian,
    )


def optim_minimize(objective, lower=None, upper=None, initial=None, method: str = "de", seed=None,
                   control: dict | None = None) -> OptimResult:
    """Minimize a user-written objective.

    Runs one of the six ported Numerics optimizers over a Python function. The optimizer's random
    number generator lives in C++, so a seeded run reproduces exactly, and reproduces identically
    in corehydror.

    Parameters
    ----------
    objective : callable
        Takes a numeric parameter vector (:class:`numpy.ndarray`) and returns a single number.
    lower, upper : array_like
        Parameter bounds, the same length as the parameter vector. Required for every method,
        including ``"de"`` and ``"brent"``, which take no ``initial``.
    initial : array_like, optional
        Starting values, the same length as ``lower``/``upper``. Required for ``"bfgs"``,
        ``"powell"``, ``"mlsl"`` and ``"nelder_mead"``.
    method : {"de", "bfgs", "powell", "mlsl", "nelder_mead", "brent"}
        ``"de"`` (differential evolution) is the default.
    seed : int, optional
        Seed for the stochastic methods (``"de"``, ``"mlsl"``); an error for any other method.
    control : dict, optional
        Optimizer settings: ``max_iterations``, ``max_function_evaluations``,
        ``absolute_tolerance``, ``relative_tolerance``, ``population_size`` (``"de"`` only),
        ``compute_hessian``, and ``report_failure`` (default ``True``, which surfaces a
        configuration failure as a Python exception rather than returning a failed status
        quietly). ``max_function_evaluations``, ``report_failure``, and ``compute_hessian`` apply
        only to ``"de"``, ``"bfgs"``, ``"powell"``, and ``"mlsl"`` (the four methods that derive
        from the ported ``Optimizer`` base); ``"nelder_mead"`` and ``"brent"`` accept only
        ``max_iterations``, ``absolute_tolerance``, and ``relative_tolerance``, and never compute
        a Hessian. ``compute_hessian`` DEFAULTS TO ``True`` for those four methods (matching the
        ported C# ``Optimizer`` base), so a successful ``"de"``/``"bfgs"``/``"powell"``/``"mlsl"``
        run returns a Hessian, computed by extra objective evaluations, unless
        ``control={"compute_hessian": False}`` turns it off.

    Returns
    -------
    OptimResult

    Examples
    --------
    >>> from corehydropy import optim_minimize
    >>> def rosenbrock(p):
    ...     return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
    >>> fit = optim_minimize(rosenbrock, lower=[-5, -5], upper=[5, 5], seed=42)
    >>> import numpy as np
    >>> np.round(fit.parameters, 3)
    array([1., 1.])
    """
    return _optim_run(objective, lower, upper, initial, method, seed, control or {}, maximize=False)


def optim_maximize(objective, lower=None, upper=None, initial=None, method: str = "de", seed=None,
                   control: dict | None = None) -> OptimResult:
    """Maximize a user-written objective. See :func:`optim_minimize` for the arguments.

    Examples
    --------
    >>> from corehydropy import optim_maximize
    >>> def peak(p):
    ...     return -((p[0] - 2) ** 2 + (p[1] + 1) ** 2)
    >>> fit = optim_maximize(peak, lower=[-10, -10], upper=[10, 10], seed=7)
    >>> import numpy as np
    >>> np.round(fit.parameters, 3)
    array([ 2., -1.])
    """
    return _optim_run(objective, lower, upper, initial, method, seed, control or {}, maximize=True)
