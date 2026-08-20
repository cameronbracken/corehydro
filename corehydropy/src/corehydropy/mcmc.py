"""Direct MCMC sampling.

A public wrapper over the ``_core.mcmc_run`` glue: fit a distribution family to data
by MCMC with any of seven ported samplers (Gibbs needs a model-specific conditional
proposal and is not exposed here), and get the chains and diagnostics back. The priors are always the model's constraint-based uniforms (the C#
``GetParameterConstraints`` bounds); custom priors are not exposed -- use the
analysis functions for the full Bayesian workflow.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["mcmc_sample"]

_SAMPLERS = ("RWMH", "ARWMH", "DEMCz", "DEMCzs", "HMC", "NUTS", "SNIS")


def mcmc_sample(
    data,
    distribution: str = "Normal",
    sampler: str = "RWMH",
    iterations: int | None = None,
    warmup: int | None = None,
    chains: int | None = None,
    thinning: int | None = None,
    output_length: int | None = None,
    seed: int = 12345,
    initialize: str = "MAP",
):
    """Sample the posterior of a distribution's parameters by MCMC.

    Fits ``distribution`` to ``data`` with uniform priors spanning the
    family's parameter constraints (exactly the C# ``GetParameterConstraints``
    bounds) and returns the full sampler output: per-chain draws, acceptance
    rates, the MAP estimate, posterior summaries, and Gelman-Rubin / effective
    sample size diagnostics.

    Runs the ported serial chain driver with the C# default seed, so a seeded
    run reproduces the C# sampler stream bit-for-bit (and matches ``corehydror``
    exactly).

    Parameters
    ----------
    data : array-like of float
        Observations.
    distribution : str
        Distribution family name; see :func:`corehydropy.distribution_names`.
    sampler : {"RWMH", "ARWMH", "DEMCz", "DEMCzs", "HMC", "NUTS", "SNIS"}
        The MCMC sampler.
    iterations : int, optional
        Iterations per chain (sampler default if omitted).
    warmup : int, optional
        Warm-up iterations discarded from each chain (sampler default if
        omitted).
    chains : int, optional
        Number of chains (sampler default if omitted).
    thinning : int, optional
        Thinning interval (sampler default if omitted).
    output_length : int, optional
        Total number of retained draws across all chains (sampler default, 10,000, if
        omitted). The ported sampler collects ``ceil(output_length / chains)`` draws per
        chain after the iteration loop. The ported floor is 100 and it is refused below
        that.
    seed : int
        PRNG seed; 12345 is the C# default.
    initialize : {"MAP", "Randomize"}
        Chain initialization: from the MAP estimate (the C# default) or
        randomized draws from the priors.

    Returns
    -------
    dict
        Keys ``parameters`` (parameter names), ``chains`` (list of
        ``(n_draws, n_params)`` arrays, one per chain), ``acceptance_rates``,
        ``map`` (parameter values at the posterior mode), ``map_fitness``,
        ``posterior_mean``, ``posterior_sd``, ``posterior_median``,
        ``posterior_lower_ci``, ``posterior_upper_ci``, ``rhat``, and
        ``ess`` (all per-parameter arrays).
    """
    if sampler not in _SAMPLERS:
        raise ValueError(f"unknown sampler '{sampler}'; use one of {_SAMPLERS}")
    settings: dict = {"prng_seed": int(seed), "initialize": initialize}
    if iterations is not None:
        settings["iterations"] = int(iterations)
        # The sampler requires warmup <= iterations / 2; when only iterations
        # is given, follow the analysis wrappers' auto rule.
        if warmup is None:
            warmup = max(50, int(iterations) // 2)
    if warmup is not None:
        settings["warmup_iterations"] = int(warmup)
    if chains is not None:
        settings["number_of_chains"] = int(chains)
    if thinning is not None:
        settings["thinning_interval"] = int(thinning)
    if output_length is not None:
        # Validated here and worded identically in mcmc_posterior() and corehydror, so the floor
        # is refused by name rather than by the ported "The output length must be at least 100."
        if int(output_length) < 100:
            raise ValueError(
                "`output_length` must be a single whole number of at least 100, which is the "
                "ported sampler's own floor"
            )
        settings["output_length"] = int(output_length)
    if sampler == "RWMH":
        # The RWMH constructor takes a proposal covariance; MAP initialization
        # overwrites it before first use (the C# test convention).
        settings.setdefault("proposal_sigma", "identity")

    xs = [float(v) for v in np.asarray(data, dtype=float).ravel()]
    raw = _core.mcmc_run(sampler, "uniform_constraints", distribution, xs, settings)

    return {
        "parameters": _core.dist_parameter_names(distribution)["short"],
        "chains": [np.asarray(c) for c in raw["chains"]],
        "acceptance_rates": np.asarray(raw["acceptance_rates"]),
        "map": np.asarray(raw["map_values"]),
        "map_fitness": raw["map_fitness"],
        "posterior_mean": np.asarray(raw["posterior_mean"]),
        "posterior_sd": np.asarray(raw["posterior_sd"]),
        "posterior_median": np.asarray(raw["posterior_median"]),
        "posterior_lower_ci": np.asarray(raw["posterior_lower_ci"]),
        "posterior_upper_ci": np.asarray(raw["posterior_upper_ci"]),
        "rhat": np.asarray(raw["rhat"]),
        "ess": np.asarray(raw["ess"]),
    }
