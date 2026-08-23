"""The general-purpose optimizer surface over the thirteen ported Numerics optimizers (DE, particle
swarm, shuffled complex evolution, simulated annealing, multi-start, MLSL, BFGS, Powell, ADAM,
gradient descent, Nelder-Mead, Brent, golden section). Unlike every other verb in toolbox.py/gof.py
(which pass serializable data through the shared ``toolbox_run`` dispatcher), an optimizer takes a
live Python function, so this goes through its own runner (``core/include/corehydro/numerics/
support/optimizer_runner.hpp``) and its own bindings (``optim_run`` / ``optim_run_grad`` in
``bindings/toolbox.cpp``), rather than ``toolbox_run``. Mirrors ``corehydror``'s ``R/optim.R`` verb
for verb.
"""

from __future__ import annotations

import json

import numpy as np

from . import _core

__all__ = ["OptimResult", "optim_minimize", "optim_maximize"]

# The three tolerance/iteration knobs every one of the thirteen optimizer classes exposes, plus the
# three only the classes deriving from the ported `Optimizer` base take. ``"nelder_mead"`` and
# ``"brent"`` are the two standalone classes (see optimizer_runner.hpp's file header);
# ``"golden_section"`` IS an ``Optimizer`` subclass, so it takes the full base set.
_SCALAR_CONTROLS = ("max_iterations", "absolute_tolerance", "relative_tolerance")
_BASE_CONTROLS = _SCALAR_CONTROLS + (
    "max_function_evaluations", "report_failure", "compute_hessian",
)
# Per-method: which control names it accepts, whether it needs `initial`, whether it is stochastic
# (accepts a seed), and whether it takes an analytic gradient. Every validation message below reads
# off this one table so the Python and R surfaces cannot drift. Every method needs `lower`/`upper`
# (even "de", "brent" and "golden_section", which take no `initial`); a method with
# ``needs_initial`` additionally needs an initial guess, and all three of `initial`/`lower`/`upper`
# are then required to be the same length, which the underlying C++ constructors validate once the
# request reaches them (NelderMead's ctor does not validate this itself -- see
# optimizer_runner.hpp's "nelder_mead" arm -- so the length check below is load-bearing, not just a
# friendlier error, for that one method).
_METHOD_TABLE = {
    "de": {"controls": _BASE_CONTROLS + ("population_size",),
           "needs_initial": False, "stochastic": True, "gradient": False},
    "particle_swarm": {"controls": _BASE_CONTROLS + ("population_size",),
                       "needs_initial": False, "stochastic": True, "gradient": False},
    "sce": {"controls": _BASE_CONTROLS + ("complexes", "cce_iterations", "tolerance_steps"),
            "needs_initial": False, "stochastic": True, "gradient": False},
    "simulated_annealing": {
        "controls": _BASE_CONTROLS + ("initial_temperature", "min_temperature", "cooling_rate",
                                      "update_cycles", "temperature_cycles", "tolerance_steps"),
        "needs_initial": False, "stochastic": True, "gradient": False},
    "multi_start": {"controls": _BASE_CONTROLS + ("local_method", "local_absolute_tolerance",
                                                  "local_relative_tolerance", "polish"),
                    "needs_initial": True, "stochastic": True, "gradient": False},
    "mlsl": {"controls": _BASE_CONTROLS + ("local_method",),
             "needs_initial": True, "stochastic": True, "gradient": False},
    "bfgs": {"controls": _BASE_CONTROLS, "needs_initial": True, "stochastic": False,
             "gradient": False},
    "powell": {"controls": _BASE_CONTROLS, "needs_initial": True, "stochastic": False,
               "gradient": False},
    "adam": {"controls": _BASE_CONTROLS + ("alpha", "beta1", "beta2"),
             "needs_initial": True, "stochastic": False, "gradient": True},
    "gradient_descent": {"controls": _BASE_CONTROLS + ("alpha",),
                         "needs_initial": True, "stochastic": False, "gradient": True},
    "nelder_mead": {"controls": _SCALAR_CONTROLS, "needs_initial": True, "stochastic": False,
                    "gradient": False},
    "brent": {"controls": _SCALAR_CONTROLS, "needs_initial": False, "stochastic": False,
              "gradient": False},
    "golden_section": {"controls": _BASE_CONTROLS, "needs_initial": False, "stochastic": False,
                       "gradient": False},
}
_METHODS = tuple(_METHOD_TABLE)
_CONTROL_KEYS = {name for spec in _METHOD_TABLE.values() for name in spec["controls"]}
_STOCHASTIC_METHODS = {m for m, spec in _METHOD_TABLE.items() if spec["stochastic"]}
_NEEDS_INITIAL = {m for m, spec in _METHOD_TABLE.items() if spec["needs_initial"]}
_GRADIENT_METHODS = {m for m, spec in _METHOD_TABLE.items() if spec["gradient"]}
# The three local methods MLSL and MultiStart actually construct. ADAM and GradientDescent are
# LocalMethod members upstream but throw "Unsupported local method" inside both classes (see
# optimization/support/local_method.hpp), so they are not offered here.
_LOCAL_METHODS = ("bfgs", "nelder_mead", "powell")


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


def _optim_run(objective, lower, upper, initial, method: str, seed, control: dict, maximize: bool,
               gradient=None) -> OptimResult:
    """Internal: validate everything Python-side, then make one call. Both verbs share this so
    their messages and defaults can never drift apart."""
    if not callable(objective):
        raise TypeError("`objective` must be a callable taking a numeric vector and returning one number")
    if gradient is not None and not callable(gradient):
        raise TypeError(
            "`gradient` must be a callable taking a numeric vector and returning one value per "
            "parameter"
        )
    if method not in _METHODS:
        raise ValueError(f"`method` must be one of {_METHODS}; got {method!r}")
    if gradient is not None and method not in _GRADIENT_METHODS:
        raise ValueError(
            f"`gradient` only applies to method(s) {sorted(_GRADIENT_METHODS)}; "
            f"got method {method!r}"
        )
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
    # A control name this method's C++ arm never reads would silently look like it did something,
    # so reject it and name the methods that do read it.
    wrong_method = sorted(set(control) - set(_METHOD_TABLE[method]["controls"]))
    if wrong_method:
        accepts = sorted(m for m, s in _METHOD_TABLE.items()
                         if any(name in s["controls"] for name in wrong_method))
        raise ValueError(
            f"control name(s) {wrong_method} only apply to method(s) {accepts}; "
            f"got method {method!r}"
        )
    if "local_method" in control and control["local_method"] not in _LOCAL_METHODS:
        raise ValueError(
            f"`local_method` must be one of {list(_LOCAL_METHODS)}; "
            f"got {control['local_method']!r}"
        )
    # `initial` is only read by the arms of the methods that need it; the others never look at it,
    # so a caller-supplied `initial` for one of those would silently do nothing.
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

    # The gradient travels as a SECOND callback, not in the spec, so it goes through its own
    # binding. An absent gradient is the ported classes' null Gradient, which falls back to
    # numerical differentiation exactly as C# does.
    if gradient is None:
        r = _core.optim_run(json.dumps(spec), _objective)
    else:
        def _gradient(p):
            return np.asarray(gradient(np.asarray(p, dtype=float)), dtype=float).tolist()

        r = _core.optim_run_grad(json.dumps(spec), _objective, _gradient)
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
                   control: dict | None = None, gradient=None) -> OptimResult:
    """Minimize a user-written objective.

    Runs one of the thirteen ported Numerics optimizers over a Python function. The optimizer's
    random number generator lives in C++, so a seeded run reproduces exactly, and reproduces
    identically in corehydror.

    Parameters
    ----------
    objective : callable
        Takes a numeric parameter vector (:class:`numpy.ndarray`) and returns a single number.
    lower, upper : array_like
        Parameter bounds, the same length as the parameter vector. Required for every method,
        including ``"de"``, ``"brent"`` and ``"golden_section"``, which take no ``initial``.
        ``"brent"`` and ``"golden_section"`` are one-dimensional: pass a single bound each.
    initial : array_like, optional
        Starting values, the same length as ``lower``/``upper``. Required for ``"bfgs"``,
        ``"powell"``, ``"mlsl"``, ``"multi_start"``, ``"adam"``, ``"gradient_descent"`` and
        ``"nelder_mead"``.
    method : {"de", "particle_swarm", "sce", "simulated_annealing", "multi_start", "mlsl", "bfgs", \
"powell", "adam", "gradient_descent", "nelder_mead", "brent", "golden_section"}
        ``"de"`` (differential evolution) is the default. The five global methods are ``"de"``,
        ``"particle_swarm"``, ``"sce"`` (shuffled complex evolution), ``"simulated_annealing"``
        and ``"multi_start"``; ``"mlsl"`` is multi-level single linkage; the rest are local.
    seed : int, optional
        Seed for the stochastic methods (``"de"``, ``"particle_swarm"``, ``"sce"``,
        ``"simulated_annealing"``, ``"multi_start"``, ``"mlsl"``); an error for any other method.
    control : dict, optional
        Optimizer settings. Every method accepts ``max_iterations``, ``absolute_tolerance`` and
        ``relative_tolerance``. Every method except ``"nelder_mead"`` and ``"brent"`` (the two
        classes that do not derive from the ported ``Optimizer`` base) additionally accepts
        ``max_function_evaluations``, ``report_failure`` (default ``True``, which surfaces a
        configuration failure as a Python exception rather than returning a failed status
        quietly) and ``compute_hessian``. The method-specific settings are ``population_size``
        (``"de"``, ``"particle_swarm"``); ``complexes``, ``cce_iterations`` and
        ``tolerance_steps`` (``"sce"``); ``initial_temperature``, ``min_temperature``,
        ``cooling_rate``, ``update_cycles``, ``temperature_cycles`` and ``tolerance_steps``
        (``"simulated_annealing"``); ``local_method`` (``"multi_start"``, ``"mlsl"``, one of
        ``"bfgs"``, ``"nelder_mead"``, ``"powell"``); ``local_absolute_tolerance``,
        ``local_relative_tolerance``, ``polish`` (``"multi_start"``); ``alpha``, the step size or
        learning rate (``"adam"``, ``"gradient_descent"``); and ``beta1``, ``beta2``, the two decay
        factors (``"adam"``). Passing a setting a method
        does not read is an error rather than a silent no-op. ``compute_hessian`` DEFAULTS TO
        ``True`` for the ``Optimizer``-base methods (matching the ported C# ``Optimizer`` base),
        so a successful run returns a Hessian, computed by extra objective evaluations, unless
        ``control={"compute_hessian": False}`` turns it off.
    gradient : callable, optional
        Takes the parameter vector and returns one partial derivative per parameter. Accepted only
        by ``"adam"`` and ``"gradient_descent"``, an error for every other method. Omitted, both
        methods differentiate the objective numerically, exactly as the upstream C# classes do with
        a null gradient.

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
    return _optim_run(objective, lower, upper, initial, method, seed, control or {}, maximize=False,
                      gradient=gradient)


def optim_maximize(objective, lower=None, upper=None, initial=None, method: str = "de", seed=None,
                   control: dict | None = None, gradient=None) -> OptimResult:
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
    return _optim_run(objective, lower, upper, initial, method, seed, control or {}, maximize=True,
                      gradient=gradient)
