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
from .distributions import Distribution, _as_spec

__all__ = [
    "root_find",
    "quadrature",
    "QuadratureResult",
    "derivative",
    "gradient",
    "hessian",
    "mcmc_posterior",
    "Rng",
]

_SAMPLERS = ("RWMH", "ARWMH", "DEMCz", "DEMCzs", "HMC", "NUTS", "SNIS")

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


def mcmc_posterior(
    log_likelihood: Callable,
    priors,
    sampler: str = "RWMH",
    iterations: int | None = None,
    warmup: int | None = None,
    chains: int | None = None,
    thinning: int | None = None,
    seed: int = 12345,
    initialize: str = "MAP",
) -> dict:
    """Sample your own posterior by MCMC.

    Runs any of the ported MCMC samplers against a log-likelihood you write and priors you
    choose. This is the constructor the upstream C# library itself exposes,
    ``MCMCSampler(priorDistributions, logLikelihoodFunction)``:
    :func:`corehydropy.mcmc_sample` can only fit a built-in distribution family under uniform
    priors spanning its parameter constraints, while this function takes any model you can
    write down.

    ``log_likelihood`` is called with one list of floats, as long as ``priors``, and must
    return a single number. Following the upstream contract, it should return the log of the
    data likelihood PLUS the log prior density; the ``priors`` list is used for the feasible
    parameter bounds and for chain initialization, and is never added to your value behind your
    back. A parameter outside its prior's support is rejected before your function sees it, so
    a flat (uniform) prior needs no term of its own.

    Parameters
    ----------
    log_likelihood : callable
        Takes a list of floats, returns one float.
    priors : Distribution or sequence of Distribution
        One prior per parameter, in the order ``log_likelihood`` reads them.
    sampler : {"RWMH", "ARWMH", "DEMCz", "DEMCzs", "HMC", "NUTS", "SNIS"}
        The MCMC sampler. ``"Gibbs"`` needs a proposal function and is not available yet.
    iterations : int, optional
        Iterations per chain (sampler default if omitted). ``"SNIS"`` needs at least 10000: its
        ported validation requires ``iterations`` to be at least the output length, and the
        output length keeps its default of 10000 here.
    warmup : int, optional
        Warm-up iterations discarded from each chain (sampler default if omitted). When
        ``iterations`` is given and ``warmup`` is not, half of ``iterations`` is used, matching
        :func:`corehydropy.mcmc_sample`. ``"SNIS"`` is the exception: its ported validation
        rejects any warm-up at all, so none is derived for it.
    chains : int, optional
        Number of chains (sampler default if omitted). ``"DEMCz"`` and ``"DEMCzs"`` require at
        least three; ``"SNIS"`` draws independently and runs one.
    thinning : int, optional
        Thinning interval (sampler default if omitted).
    seed : int
        PRNG seed; 12345 is the C# default.
    initialize : {"MAP", "Randomize"}
        Chain initialization: from the posterior-mode estimate (the C# default) or randomized
        draws from the priors.

    Returns
    -------
    dict
        The same fields :func:`corehydropy.mcmc_sample` returns: ``parameters`` (parameter
        names, ``p1``, ``p2``, ... since your model names nothing), ``chains`` (a list of one
        ``(n_draws, n_params)`` array per chain), ``acceptance_rates``, ``map``,
        ``map_fitness``, ``posterior_mean``, ``posterior_sd``, ``posterior_median``,
        ``posterior_lower_ci``, ``posterior_upper_ci``, ``rhat``, and ``ess``.

    Notes
    -----
    **Reproducing a run in R.** A seeded :func:`corehydropy.mcmc_sample` run is bit-identical
    between Python and R, because every arithmetic operation happens in the shared C++ core.
    That guarantee is WEAKER here, and it is worth stating plainly. The draws still come from
    the core's seeded Mersenne Twister, but the log-density is your own Python code, and Python
    and R do not guarantee identical rounding for the same formula. MCMC amplifies a single
    differing bit far harder than an optimizer does: one flipped accept-or-reject changes every
    state after it, so the two chains diverge outright rather than drift apart slowly.

    A seeded run reproduces across the two languages if and only if your function returns
    bit-identical values. Arithmetic (``+ - * /``) is IEEE-deterministic and does reproduce;
    ``log``, ``exp``, ``gamma`` and friends come from each platform's own math library and are
    not guaranteed to. Note also that R's ``sum()`` accumulates in extended precision while
    Python's does not, so a plain loop is the portable spelling on both sides.

    **Performance.** Every evaluation calls back into Python, and there are far more of them
    than ``iterations`` suggests: the count is
    ``(iterations + output_length / chains) * thinning * chains``, and the ported defaults are 4
    chains, a thinning interval of 20 and an output length of 10,000. So ``iterations=10000`` is
    a million crossings, not ten thousand.

    That is affordable. Measured here, ``iterations=10000`` over a 50-point Gaussian
    log-density took 1.3 seconds, against 0.7 seconds for the
    :func:`corehydropy.mcmc_sample` call in the same units, so about twice as slow. The
    crossing itself costs about 0.3 microseconds: the same function called a million times
    directly from Python takes 1.0 second, so most of the 1.3 seconds is your own code, not the
    boundary.

    Two settings dominate the count and neither is obvious. ``thinning`` multiplies it, and
    ``initialize="MAP"``, the C# default, runs the DifferentialEvolution optimizer over your
    function before the first chain iteration; ``initialize="Randomize"`` skips the optimizer.

    **Interrupting a long run.** Ctrl-C returns control with a ``KeyboardInterrupt``, but not
    instantly. The ported samplers have no cancellation hook, so the chain runs to the end of
    its loop -- rejecting every remaining point without calling your function again -- before
    the interrupt surfaces. Measured on a 200,000-iteration chain: 0.3 seconds from the signal
    to the exception reaching the caller.

    See Also
    --------
    corehydropy.mcmc_sample : a built-in family under constraint-based priors, which is faster
        and bit-identical across languages.

    Examples
    --------
    >>> import corehydropy as ch
    >>> data = [4.9, 5.1, 5.0, 5.2, 4.8]
    >>> # A plain loop and ``+ - * /`` alone, the portable spelling described above.
    >>> def ll(p):
    ...     acc = 0.0
    ...     for x in data:
    ...         acc += (x - p[0]) * (x - p[0])
    ...     return -0.5 * acc
    >>> fit = ch.mcmc_posterior(ll, [ch.Distribution("Uniform", [0.0, 10.0])],
    ...                         iterations=200, seed=12345)  # doctest: +SKIP
    """
    if not callable(log_likelihood):
        raise TypeError(
            "`log_likelihood` must be a function taking a parameter vector and returning a "
            "single number"
        )
    if seed is None:
        raise TypeError("`seed` must not be None")
    if sampler not in _SAMPLERS:
        raise ValueError(f"unknown sampler '{sampler}'; use one of {_SAMPLERS}")
    if initialize not in ("MAP", "Randomize"):
        raise ValueError('`initialize` must be "MAP" or "Randomize"')

    options: dict = {
        "sampler": sampler,
        "initialize": initialize,
        "seed": int(seed),
        "priors": _prior_specs(priors),
    }
    if iterations is not None:
        options["iterations"] = int(iterations)
        # The sampler requires warmup <= iterations / 2; when only iterations is given, follow
        # mcmc_sample()'s rule. SNIS is the exception: its ported ValidateSettings rejects ANY
        # warm-up, so an auto-derived one would turn a legal call into an error.
        if warmup is None and sampler != "SNIS":
            warmup = max(50, int(iterations) // 2)
    if warmup is not None:
        options["warmup"] = int(warmup)
    if chains is not None:
        if sampler == "SNIS" and int(chains) != 1:
            raise ValueError(
                "SNIS draws independently rather than running Markov chains; it supports "
                "`chains=1` only"
            )
        options["chains"] = int(chains)
    if thinning is not None:
        options["thinning"] = int(thinning)
    if sampler == "RWMH":
        # The RWMH constructor takes a proposal covariance, and the ported default is all zeros,
        # which is only usable when MAP initialization overwrites it before the first iteration.
        # Identity is what mcmc_sample() sets for the same reason.
        options["proposal_sigma"] = "identity"

    return _mcmc_unflatten(_core.callback_mcmc(json.dumps(options), log_likelihood))


def _prior_specs(priors) -> list:
    """Internal: accept either a sequence of Distribution objects or a single one, and refuse
    anything else by name. The length of this list IS the parameter count, so a wrong one is the
    likeliest user error here and the C++ side cannot tell it from an intentional model."""
    if isinstance(priors, Distribution):
        priors = [priors]
    specs = list(priors) if not isinstance(priors, (str, bytes)) else []
    if not specs or not all(isinstance(d, Distribution) for d in specs):
        raise TypeError(
            "`priors` must be a non-empty sequence of Distribution objects, one per parameter"
        )
    return [_as_spec(d) for d in specs]


def _mcmc_unflatten(res: dict) -> dict:
    """Internal: slice the flat callback result back into the shape mcmc_sample() returns. The
    layout is documented in ``core/include/corehydro/numerics/support/callback/mcmc.hpp`` -- a
    named summary block, then the draws row-major by [chain][draw][parameter] -- and
    ``corehydror``'s own ``mcmc_unflatten()`` reads it identically."""
    n_summary, n_chains, n_draws, p = (int(v) for v in res["dims"])
    values = list(res["values"])
    names = list(res["names"])
    index = {name: i for i, name in enumerate(names)}

    def per_parameter(prefix):
        return np.asarray([values[index[f"{prefix}[{j}]"]] for j in range(p)], dtype=float)

    draws = np.asarray(values[n_summary:], dtype=float)
    chains = [
        draws[k * n_draws * p : (k + 1) * n_draws * p].reshape(n_draws, p)
        for k in range(n_chains)
    ]

    return {
        "parameters": [f"p{j + 1}" for j in range(p)],
        "chains": chains,
        "acceptance_rates": np.asarray(
            [values[index[f"acceptance_rate[{k}]"]] for k in range(n_chains)], dtype=float
        ),
        "map": per_parameter("map"),
        "map_fitness": values[index["map_fitness"]],
        "posterior_mean": per_parameter("posterior_mean"),
        "posterior_sd": per_parameter("posterior_sd"),
        "posterior_median": per_parameter("posterior_median"),
        "posterior_lower_ci": per_parameter("posterior_lower_ci"),
        "posterior_upper_ci": per_parameter("posterior_upper_ci"),
        "rhat": per_parameter("rhat"),
        "ess": per_parameter("ess"),
    }


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
