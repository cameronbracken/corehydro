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
    "bootstrap_custom",
    "fit_gmm_moments",
    "Rng",
]

_SAMPLERS = ("RWMH", "ARWMH", "DEMCz", "DEMCzs", "HMC", "NUTS", "SNIS", "Gibbs")
# The samplers that take an analytic gradient. Kept beside _SAMPLERS so the two lists cannot drift.
_GRADIENT_SAMPLERS = ("HMC", "NUTS")

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
    proposal: Callable | None = None,
    gradient: Callable | None = None,
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
    sampler : {"RWMH", "ARWMH", "DEMCz", "DEMCzs", "HMC", "NUTS", "SNIS", "Gibbs"}
        The MCMC sampler. ``"Gibbs"`` requires ``proposal``.
    proposal : callable, optional
        The conditional proposal function ``"Gibbs"`` samples with, and the only sampler that
        takes one. It is called as ``proposal(parameters, rng)`` and must return a sequence as
        long as ``priors`` (a bare number when there is one parameter): the next state of the
        chain, which Gibbs accepts unconditionally.
        Ordinarily that is a draw from the full conditional of the model. ``rng`` is a
        :class:`corehydropy.Rng` handle on the generator this chain is running on -- draw from it
        with ``rng.uniform()`` and ``rng.integers()``, not with :mod:`random` or
        :mod:`numpy.random`, or the seeded run stops being reproducible.
    gradient : callable, optional
        An analytic gradient of ``log_likelihood`` for ``"HMC"`` and ``"NUTS"``, the only samplers
        that take one. It is called as ``gradient(parameters)`` and must return a sequence as long
        as ``priors`` (a bare number when there is one parameter). Left unset, both samplers use
        the ported bound-aware finite-difference
        gradient, which costs two extra ``log_likelihood`` calls per parameter per leapfrog step;
        an analytic gradient is usually a large saving and is always more accurate.
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

    ``"Gibbs"`` is the sampler whose defaults surprise: the ported constructor sets one chain, no
    thinning, and 100,000 iterations on top of a 10,000-draw output block, so an ``iterations`` you
    do not set is 110,000 iterations of BOTH your log-likelihood and your proposal. Set
    ``iterations``.

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
    # Each optional delegate belongs to specific samplers, and a mismatch is refused rather than
    # ignored: a user who writes a gradient and leaves `sampler` at its default would otherwise get
    # a plausible run that never called it. Worded to match corehydror's mcmc_posterior().
    if sampler == "Gibbs" and proposal is None:
        raise ValueError(
            "the Gibbs sampler requires a `proposal` function; it has no default conditional draw"
        )
    if proposal is not None:
        if not callable(proposal):
            raise TypeError(
                "`proposal` must be a function taking (parameters, rng) and returning a "
                "parameter vector"
            )
        if sampler != "Gibbs":
            raise ValueError(
                f"`proposal` is only used by the Gibbs sampler; '{sampler}' does not take one"
            )
    if gradient is not None:
        if not callable(gradient):
            raise TypeError("`gradient` must be a function taking a parameter vector and returning one")
        if sampler not in _GRADIENT_SAMPLERS:
            raise ValueError(
                f"`gradient` is only used by the HMC and NUTS samplers; '{sampler}' does not "
                "take one"
            )

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

    return _mcmc_unflatten(
        _core.callback_mcmc(json.dumps(options), log_likelihood, proposal, gradient)
    )


_CI_METHODS = ("Percentile", "BiasCorrected", "Normal", "BootstrapT", "BCa")


def fit_gmm_moments(
    moment_conditions: Callable,
    initial: Sequence[float],
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
    sample_size: int | None = None,
    jacobian: Callable | None = None,
    penalty: Callable | None = None,
    optimizer: str = "BFGS",
    strategy: str = "Iterative",
    max_gmm_iterations: int = 0,
):
    """Fit your own moment conditions by the generalized method of moments.

    Estimates parameters by GMM against moment conditions you write. This is the constructor the
    upstream C# library itself exposes, ``GeneralizedMethodOfMoments(momentConditionFunction,
    ...)``: :func:`~corehydropy.fit_gmm` can only fit a :func:`~corehydropy.model_bulletin17c`
    model (the single implementation of the ``IGMMModel`` interface it takes), while this function
    takes any moment conditions you can write down.

    The moment condition function
    -----------------------------
    ``moment_conditions(parameters)`` is called with one list of numbers, as long as `initial`, and
    must return **two things**: the tuple ``(g, s)``, or a dict with keys ``"g"`` and ``"s"``.

    - ``g`` is the sample mean of the moment conditions at those parameters: a sequence of q
      numbers, one per moment condition. GMM drives this towards zero. With one moment condition
      it may be written as the bare number, which is R's spelling for it -- except when ``s`` is
      also written bare, where ``(number, number)`` cannot be told apart from a flat ``[g0, g1]``
      and is refused by name.
    - ``s`` is their covariance: a q by q **matrix**, written as a sequence of ROWS (a list of
      lists, or a 2-D numpy array). In two-step and iterative GMM the optimal weighting matrix is
      its inverse.

    Both are required and both are checked by name, because returning the wrong thing here is the
    likeliest mistake on this surface. A two-parameter method-of-moments fit of a Normal, whose
    answer is the sample mean and the population variance::

        def moments(p):
            a = [xi - p[0] for xi in x]
            b = [ai * ai - p[1] for ai in a]
            n = len(x)
            mean = lambda v: sum(v) / n
            return ([mean(a), mean(b)],
                    [[mean([ai * ai for ai in a]), mean([ai * bi for ai, bi in zip(a, b)])],
                     [mean([ai * bi for ai, bi in zip(a, b)]), mean([bi * bi for bi in b])]])

    ``q`` (the number of moment conditions) is measured by calling your function once at `initial`,
    so there is no argument to get wrong. When q equals the number of parameters the fit is
    just-identified and ``.j_stat_pval`` comes back ``None`` -- see Returns.

    Parameters
    ----------
    moment_conditions : callable
        The required function described above.
    initial : sequence of float
        Starting values, one per parameter. Its length IS the parameter count.
    lower, upper : sequence of float
        Parameter bounds, the same length as `initial`, with every starting value inside them. Both
        are required despite the ``None`` default: every optimizer this dispatches to takes a box,
        and the numerical Jacobian's step selection is bounds-aware.
    sample_size : int
        The number of observations behind the moment conditions. Required, and only you know it:
        your function hands over averages, not data. The sandwich covariance divides by it, so the
        standard errors scale as ``1 / sqrt(sample_size)``.
    jacobian : callable, optional
        An analytic Jacobian of the moment conditions, called as ``jacobian(parameters)`` and
        returning a q by p matrix -- one ROW per moment condition, one COLUMN per parameter.
        ``None``, the default, uses the ported bounds-aware finite-difference Jacobian, which costs
        two extra `moment_conditions` calls per parameter per gradient.
    penalty : callable, optional
        A penalty added to the GMM objective, called as ``penalty(parameters)`` and returning one
        number. Ridge-type regularization, and the only way to fit a model with more parameters
        than moment conditions (which is otherwise refused as under-identified). Return ``0.0`` for
        no penalty. Note the ported half-quadratic convention: with a penalty the objective is
        ``0.5 * g'Wg + penalty``, so a penalty should carry its own ``1/2``.
    optimizer : str, default "BFGS"
        One of ``"BFGS"`` (matching ``GeneralizedMethodOfMoments``'s own class default),
        ``"NelderMead"``, ``"Brent"``, ``"Powell"``, ``"DifferentialEvolution"``,
        ``"MultilevelSingleLinkage"``.
    strategy : {"Iterative", "OneStep", "TwoStep"}, default "Iterative"
        GMM estimation strategy. ``"OneStep"`` is refused for an over-identified problem, matching
        the estimator.
    max_gmm_iterations : int, default 0
        Maximum number of GMM iterations; ``0`` keeps the estimator's own default cap.

    Returns
    -------
    Fit
        The same object :func:`~corehydropy.fit_gmm` returns, with ``.method == "GMM"``, so
        ``.parameters``, ``.covariance``, ``repr()`` and ``.summary()`` behave identically.
        Parameters are named ``p1``, ``p2``, ... since your moment conditions name nothing. Method
        of moments computes no likelihood surface, so ``.log_likelihood``, ``.aic`` and ``.bic``
        are ``None`` and ``.confint()`` raises, as they do for :func:`~corehydropy.fit_gmm`.
        ``.j_stat`` is Hansen's J and ``.j_stat_pval`` its p-value, which is ``None`` whenever the
        fit is just-identified (as many moment conditions as parameters): zero degrees of freedom
        leaves no over-identifying restriction to test, and ``.summary()`` says so rather than
        showing a figure. ``.j_stat`` itself is not a goodness-of-fit number you can read there
        either. The residual covariance it is scaled by is singular -- it has rank q - p, so it is
        exactly zero when the fit is just-identified -- and inverting it amplifies the optimizer's
        convergence tolerance rather than any property of your data. The result varies by many
        orders of magnitude and in sign between optimizers, and between this package and the C#
        library it ports, on fits whose parameters agree to ten significant figures. Sometimes it
        cannot be computed at all, and then it comes back ``nan`` rather than failing the fit.
        Over-identifying the model restores the p-value but not ``.j_stat``, since the rank
        deficiency only shrinks from q to q - p.
        ``.degree_of_freedom`` and ``.number_of_moment_conditions`` carry q - p and q.
        :func:`~corehydropy.fit_diagnostics` and :func:`~corehydropy.quantile_variance` are not
        available for this fit -- both need the model a `fit_gmm` fit carries.

    Notes
    -----
    There is no random number generator anywhere in this fit, so a repeated call returns the
    identical numbers. Across languages the guarantee is the usual one for this surface: the
    optimizer arithmetic all happens in the shared C++ core, but ``g`` and ``s`` are computed by
    your own Python code, and R and Python do not guarantee identical rounding for the same
    formula. Arithmetic (``+ - * /``) is IEEE-deterministic and does reproduce; ``log``, ``exp``
    and friends come from each platform's math library. R's ``sum()`` and ``mean()`` accumulate in
    extended precision where Python's do not, so an explicit loop is the portable spelling.

    See Also
    --------
    corehydropy.fit_gmm : the Bulletin 17C flood-frequency fit.
    corehydropy.optim_minimize : a plain bounded optimization of your own objective.
    corehydropy.fit_mle : likelihood-based fitting.

    Examples
    --------
    >>> import corehydropy as ch
    >>> x = [4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7]
    >>> def moments(p):
    ...     n = len(x)
    ...     a = [xi - p[0] for xi in x]
    ...     b = [ai * ai - p[1] for ai in a]
    ...     saa = sbb = sab = 0.0
    ...     for ai, bi in zip(a, b):
    ...         saa += ai * ai
    ...         sab += ai * bi
    ...         sbb += bi * bi
    ...     return ([sum(a) / n, sum(b) / n],
    ...             [[saa / n, sab / n], [sab / n, sbb / n]])
    >>> f = ch.fit_gmm_moments(moments, initial=[5.0, 0.5], lower=[0.0, 0.001],
    ...                        upper=[10.0, 10.0], sample_size=len(x))
    >>> round(f.parameters["p1"], 6), round(f.parameters["p2"], 6)
    (4.95, 0.165)
    >>> f.j_stat_pval is None  # just-identified, so there is nothing to test
    True
    """
    from .fit import _KNOWN_GMM_STRATEGIES, _KNOWN_OPTIMIZERS, _new_fit_gmm_moments

    if not callable(moment_conditions):
        raise TypeError(
            "`moment_conditions` must be a function taking (parameters) and returning the tuple "
            "(g, s) -- the moment vector and the weighting matrix"
        )
    if jacobian is not None and not callable(jacobian):
        raise TypeError("`jacobian` must be a function taking (parameters) and returning a matrix")
    if penalty is not None and not callable(penalty):
        raise TypeError(
            "`penalty` must be a function taking (parameters) and returning a single number"
        )
    start = np.asarray(initial, dtype=float).ravel()
    if start.size == 0 or not np.all(np.isfinite(start)):
        raise ValueError("`initial` must be a non-empty sequence of finite numbers")
    # Required despite the None default, exactly as optim_minimize()'s bounds are: the estimator
    # has no unbounded form (see this function's `lower`/`upper` documentation).
    if lower is None or upper is None:
        raise ValueError(
            "fit_gmm_moments() needs `lower` and `upper` bounds: the GMM optimizer takes a box "
            "and the numerical Jacobian's step selection is bounds-aware"
        )
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("`lower` and `upper` must be sequences of finite numbers")
    if lo.size != start.size or hi.size != start.size:
        raise ValueError(
            f"`initial`, `lower` and `upper` must be the same length; they are {start.size}, "
            f"{lo.size} and {hi.size}"
        )
    if sample_size is None or float(sample_size) < 1 or not np.isfinite(float(sample_size)):
        raise ValueError(
            "`sample_size` must be a single positive whole number: the number of observations "
            "behind your moment conditions, which the sandwich covariance divides by"
        )
    optimizer = str(optimizer)
    if optimizer not in _KNOWN_OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer '{optimizer}'; expected one of {', '.join(_KNOWN_OPTIMIZERS)}"
        )
    strategy = str(strategy)
    if strategy not in _KNOWN_GMM_STRATEGIES:
        raise ValueError(
            f"unknown GMM estimation strategy '{strategy}'; expected one of "
            f"{', '.join(_KNOWN_GMM_STRATEGIES)}"
        )

    options: dict = {
        "initial": start.tolist(),
        "lower": lo.tolist(),
        "upper": hi.tolist(),
        "sample_size": int(sample_size),
        "optimizer": optimizer,
        "strategy": strategy,
    }
    if int(max_gmm_iterations) > 0:
        options["max_gmm_iterations"] = int(max_gmm_iterations)

    res = _core.callback_gmm(json.dumps(options), moment_conditions, jacobian, penalty)
    return _new_fit_gmm_moments(_gmm_unflatten(res))


def _gmm_unflatten(res: dict) -> dict:
    """Internal: slice the flat callback result back into the field set `_new_fit_gmm_moments()`
    reads -- the same names `fit_run` returns for the GMM target, so both feed the one shared
    `_gmm_fit_fields()` builder in fit.py. The layout is documented in
    ``core/include/corehydro/numerics/support/callback/gmm.hpp``, and ``corehydror``'s own
    ``gmm_unflatten()`` reads it identically."""
    p = int(res["dims"][0])
    values = list(res["values"])
    index = {name: i for i, name in enumerate(res["names"])}

    def block(prefix):
        return [values[index[f"{prefix}[{j}]"]] for j in range(p)]

    # A p x p matrix read back by the label on every entry rather than by slicing a range: the
    # bindings and the fixture runners all address this result by name, and a matrix is the one
    # place an off-by-one slice would still look plausible.
    def square(prefix):
        return [[values[index[f"{prefix}[{i},{j}]"]] for j in range(p)] for i in range(p)]

    return {
        "method": "GMM",
        "parameters": block("parameter"),
        "parameter_names": [f"p{j + 1}" for j in range(p)],
        "standard_errors": block("standard_error"),
        "covariance": square("covariance"),
        "correlation": square("correlation"),
        "j_stat": values[index["j_stat"]],
        "j_stat_pval": values[index["j_stat_pval"]],
        "degree_of_freedom": int(values[index["degree_of_freedom"]]),
        "gmm_iterations": int(values[index["gmm_iterations"]]),
        "converged_within_tolerance": values[index["converged_within_tolerance"]] == 1.0,
        "optimizer_fallback_count": int(values[index["optimizer_fallback_count"]]),
        "number_of_moment_conditions": int(values[index["number_of_moment_conditions"]]),
        "nobs": int(values[index["sample_size"]]),
        "converged": res["status"] == "Success",
        "status": res["status"],
    }


def _check_bootstrap_fn(f: object, name: str, signature: str) -> None:
    """Internal: the function check every bootstrap delegate shares, naming the argument AND its
    signature -- a wrong argument order is the likeliest mistake on this surface and nothing
    downstream can detect it. Kept in step with corehydror's bootstrap_check_fn."""
    if not callable(f):
        raise TypeError(f"`{name}` must be a function taking {signature}")


def bootstrap_custom(
    data: Sequence[float],
    resample: Callable,
    fit: Callable,
    statistic: Callable,
    jackknife: Callable | None = None,
    replicates: int = 1000,
    alpha: float = 0.1,
    ci_method: str = "Percentile",
    seed: int = 12345,
    parameters: Sequence[float] | None = None,
    inner_replicates: int | None = None,
    max_retries: int | None = None,
) -> dict:
    """Bootstrap your own statistic.

    Runs the ported Numerics bootstrap against resampling, fitting and statistic functions you
    write. This is the class upstream exposes as four delegates -- ``ResampleFunction``,
    ``FitFunction``, ``StatisticFunction`` and ``JackknifeFunction`` -- so any quantity you can
    compute from a fitted parameter set can be given a confidence interval, not just the built-in
    distribution quantiles :func:`corehydropy.bootstrap_analysis` covers.

    Getting an argument order wrong is the likeliest mistake here, and the C++ side cannot tell a
    swapped pair from a deliberate one, so each signature is given exactly:

    ``resample(data, parameters, rng)``
        Returns one bootstrap sample. ``data`` is the original sample, ``parameters`` is the
        current parameter list, and ``rng`` is a :class:`corehydropy.Rng` handle on THIS
        replicate's generator -- draw from it with ``rng.uniform()`` and ``rng.integers()``, not
        with :mod:`random` or :mod:`numpy.random`, or the seeded run stops being reproducible and
        stops agreeing with R. The returned sample need not be the same length as ``data``.
    ``fit(data)``
        Returns the parameters fitted to ``data``: one number per parameter, the same count every
        time. A one-parameter model may return the bare number rather than ``[x]``, which is the
        spelling R uses for it.
    ``statistic(parameters)``
        Returns the numbers to put intervals on, computed from a fitted parameter vector: one or
        more, the same count every time. A single statistic may likewise be returned bare.
    ``jackknife(data, index)``
        Returns ``data`` with observation ``index`` left out. ``index`` counts from 0, matching the
        ported delegate, so the Python spelling is ``data[:index] + data[index + 1:]``. Only the
        ``"BCa"`` method uses it; every other method ignores it.

    Parameters
    ----------
    data : array_like
        The original sample: non-empty and finite.
    resample, fit, statistic : callable
        The three required functions, with the signatures above.
    jackknife : callable, optional
        The leave-one-out function, required by ``ci_method="BCa"`` and unused by every other
        method.
    replicates : int
        Number of bootstrap replicates.
    alpha : float
        The interval's total tail probability: 0.1 gives a 90% interval, ``alpha / 2`` in each
        tail.
    ci_method : {"Percentile", "BiasCorrected", "Normal", "BootstrapT", "BCa"}
        ``"Normal"`` and ``"BootstrapT"`` work on the ported cube-root transform of the statistic;
        ``"BootstrapT"`` runs the studentized workflow, which nests ``inner_replicates`` further
        resample-and-fit pairs inside every replicate.
    seed : int
        PRNG seed; 12345 is the C# default.
    parameters : array_like, optional
        The original parameter vector the replicates are compared against. Left unset, ``fit(data)``
        is used, which is what the bootstrap ordinarily means by it.
    inner_replicates : int, optional
        Inner replicates for ``ci_method="BootstrapT"``, ignored by every other method. Left unset,
        the ported default (300) applies.
    max_retries : int, optional
        The maximum number of times a single failed replicate is retried before it is counted in
        ``failed_replicates``. Left unset, the ported default (``MaxRetries``, 20) applies. Each
        retry is another crossing into Python, so lowering it caps the worst case rather than
        changing the typical one.

    Returns
    -------
    dict
        Per statistic: ``estimate`` (the statistic of ``parameters``, not a bootstrap average),
        ``lower``, ``upper``, ``standard_error``, ``mean`` (the mean over valid replicates, so
        ``mean - estimate`` is the bias estimate) and ``valid_count``; the same first three for the
        fitted parameters as ``parameter_estimate``, ``parameter_lower`` and ``parameter_upper``;
        and ``replicates``, ``failed_replicates``, ``alpha`` and ``ci_method``.

    Notes
    -----
    **How many times your functions are called.** ``replicates`` calls of each of ``resample``,
    ``fit`` and ``statistic``, plus one extra ``statistic`` call to learn how many values it
    returns and one ``fit`` call when ``parameters`` is not supplied. A failed replicate is retried
    up to ``max_retries`` times (20 by default), and ``"BCa"`` adds one ``jackknife`` + ``fit`` +
    ``statistic`` per observation. ``"BootstrapT"`` is the expensive one: it multiplies the resample
    and fit counts by ``inner_replicates``, so the ported defaults (10,000 x 300) would be three
    million crossings back into Python. Start small.

    **Reproducing a run in R.** The draws come from the core's seeded Mersenne Twister, so an
    identical ``bootstrap_custom()`` call in R resamples the identical observations. The numbers
    your functions compute from them are your own Python code, though, and Python and R do not
    guarantee identical rounding for the same formula: arithmetic (``+ - * /``) is
    IEEE-deterministic and does reproduce, while ``log``, ``exp`` and friends come from each
    platform's math library. Note also that R's ``sum()`` and ``mean()`` accumulate in extended
    precision where Python's do not, so a plain loop is the portable spelling on both sides.

    See Also
    --------
    corehydropy.bootstrap_analysis : the built-in parametric bootstrap of a fitted distribution's
        quantiles.
    corehydropy.Rng : the generator handle ``resample`` is given.

    Examples
    --------
    >>> import corehydropy as ch
    >>> x = [4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7]
    >>> def resample(data, parameters, rng):
    ...     return [data[k] for k in rng.integers(len(data), 0, len(data))]
    >>> def fit(data):
    ...     acc = 0.0
    ...     for xi in data:
    ...         acc += xi
    ...     return acc / len(data)
    >>> res = ch.bootstrap_custom(x, resample, fit, lambda p: p, replicates=500, seed=12345)
    >>> bool(res["lower"][0] < res["estimate"][0] < res["upper"][0])
    True
    """
    point = np.asarray(data, dtype=float).ravel()
    if point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("`data` must be a non-empty numeric vector of finite values")
    _check_bootstrap_fn(resample, "resample", "(data, parameters, rng)")
    _check_bootstrap_fn(fit, "fit", "(data)")
    _check_bootstrap_fn(statistic, "statistic", "(parameters)")
    if jackknife is not None:
        _check_bootstrap_fn(jackknife, "jackknife", "(data, index)")
    if ci_method not in _CI_METHODS:
        raise ValueError(f"unknown ci_method '{ci_method}'; use one of {_CI_METHODS}")
    # Refused HERE rather than after the run: the ported class checks it inside
    # GetConfidenceIntervals, by which point every replicate has already called back into Python.
    # The core repeats this check for all four runners; the wording is the same in both packages.
    if ci_method == "BCa" and jackknife is None:
        raise ValueError(
            "the BCa confidence interval method requires a `jackknife` function, called with "
            "(data, index); supply one or choose another `ci_method`"
        )
    if int(replicates) < 1:
        raise ValueError("`replicates` must be a single positive whole number")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("`alpha` must be a single number between 0 and 1")
    if seed is None:
        raise TypeError("`seed` must not be None")

    options: dict = {
        "data": point.tolist(),
        "replicates": int(replicates),
        "alpha": float(alpha),
        "ci_method": ci_method,
        "seed": int(seed),
    }
    if parameters is not None:
        theta = np.asarray(parameters, dtype=float).ravel()
        if theta.size == 0 or not np.all(np.isfinite(theta)):
            raise ValueError(
                "`parameters` must be a non-empty numeric vector of finite values, or None"
            )
        options["parameters"] = theta.tolist()
    if inner_replicates is not None:
        if int(inner_replicates) < 1:
            raise ValueError("`inner_replicates` must be a single positive whole number")
        options["inner_replicates"] = int(inner_replicates)
    if max_retries is not None:
        if int(max_retries) < 1:
            raise ValueError("`max_retries` must be a single positive whole number")
        options["max_retries"] = int(max_retries)

    res = _core.callback_bootstrap(
        json.dumps(options), resample, fit, statistic, jackknife
    )
    return _bootstrap_unflatten(res, ci_method)


def _bootstrap_unflatten(res: dict, ci_method: str) -> dict:
    """Internal: slice the flat callback result back by name. The layout is documented in
    ``core/include/corehydro/numerics/support/callback/bootstrap.hpp``, and ``corehydror``'s own
    ``bootstrap_unflatten()`` reads it identically."""
    n_statistics, n_parameters = (int(v) for v in res["dims"])
    values = list(res["values"])
    index = {name: i for i, name in enumerate(res["names"])}

    def block(prefix, count):
        return np.asarray([values[index[f"{prefix}[{j}]"]] for j in range(count)], dtype=float)

    return {
        "estimate": block("statistic", n_statistics),
        "lower": block("statistic_lower", n_statistics),
        "upper": block("statistic_upper", n_statistics),
        "standard_error": block("statistic_se", n_statistics),
        "mean": block("statistic_mean", n_statistics),
        "valid_count": block("statistic_valid", n_statistics).astype(int),
        # The parameter block is always a percentile interval, whatever `ci_method` asks for: the
        # ported GetConfidenceIntervals applies the requested method to the STATISTICS and takes
        # plain percentiles of the fitted parameters.
        "parameter_estimate": block("parameter", n_parameters),
        "parameter_lower": block("parameter_lower", n_parameters),
        "parameter_upper": block("parameter_upper", n_parameters),
        "replicates": int(values[index["replicates"]]),
        "failed_replicates": int(values[index["failed_replicates"]]),
        "alpha": float(values[index["alpha"]]),
        "ci_method": ci_method,
    }


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
