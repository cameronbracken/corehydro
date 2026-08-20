# Callback Surface: Design

Phase 5 of the roadmap "Surface the full core in R and Python"
(`~/.claude/plans/surface-all-bestfit-functionality-stateless-graham.md`). Phase 1 delivered the
data and model layer (PR #18, v0.3.0), phase 2 the estimation layer (PR #19, v0.4.0), phase 3 the
distribution layer (PR #20, v0.5.0), and phase 4 the numerics toolbox (PR #21, v0.6.0).

This design replaces the roadmap's phase 5 ("Math layer"). That entry listed matrix
decompositions, special functions, integration, root finding, and numerical derivatives, and
missed the larger gap: every place upstream takes a delegate, the packages reach only through an
internal registry. Root finding, integration, and numerical derivatives are folded in here
because they take a user function and belong to the same boundary. Matrix decompositions and
special functions are dropped (see Out of scope).

## Goal

Let a user's own R or Python function drive the core. Today `optim_minimize()` and
`optim_maximize()` are the only entry points that accept one: `GuardedObjective` appears in
exactly three files, `numerics/support/optimizer_runner.hpp` and the two toolbox glues.
Everything else runs off the built-in model registries, so a user cannot sample their own
posterior, supply custom priors, bootstrap their own statistic, or integrate their own function.

This is faithful surfacing, not an addition. The native C# API for these classes is
delegate-driven, and the registries in `core/` are fixture scaffolding corehydro added:

| Upstream | Delegate |
|---|---|
| `Sampling/MCMC/Base/MCMCSampler.cs:43` | `MCMCSampler(List<IUnivariateDistribution> priorDistributions, LogLikelihood logLikelihoodFunction)` |
| `Sampling/MCMC/Gibbs.cs:33` | `public delegate double[] Proposal(double[] parameters, Random prng)` |
| `Sampling/MCMC/HMC.cs:42` | `public delegate Vector Gradient(IList<double> parameters)` |
| `Sampling/Bootstrap/Bootstrap.cs:128-152` | `ResampleFunction`, `FitFunction`, `StatisticFunction`, `JackknifeFunction` |
| `RMC.BestFit/Estimation/GeneralizedMethodOfMoments.cs:24-58` | `JacobianFunction`, `PenaltyFunction`, `MomentConditionFunction`, `PointwiseMomentConditionFunction` |

The C++ core already carries each one as a `std::function`, so most of this phase is binding
work. The one exception is the integrator (see Porting work).

## What the core supports

Read from the headers, not inferred.

| Group | Header | Callback signature |
|---|---|---|
| MCMC log-likelihood | `numerics/sampling/mcmc/base/mcmc_sampler.hpp:72` | `std::function<double(const std::vector<double>&)>`, with priors as `vector<shared_ptr<UnivariateDistributionBase>>` |
| Gibbs proposal | `numerics/sampling/mcmc/gibbs.hpp:35` | `std::function<vector<double>(const vector<double>&, MersenneTwister&)>` |
| HMC and NUTS gradient | `numerics/sampling/mcmc/hmc.hpp:69` | `std::function<Vector(const vector<double>&)>`, optional; empty means the ported bound-aware finite-difference gradient |
| Bootstrap | `numerics/sampling/bootstrap/bootstrap.hpp:81-86` | `ResampleFn`, `FitFn`, `StatisticFn`, `JackknifeFn`, `SampleSizeFn`, `TransformFn` |
| GMM | `estimation/gmm_delegates.hpp:37-53` | the four delegate aliases, over the delegate constructor at `generalized_method_of_moments.hpp` (C# line 143) |
| Root finding | `numerics/math/rootfinding/brent.hpp:18` | `solve(const std::function<double(double)>&, lower, upper)` |
| Differentiation | `numerics/math/differentiation/numerical_derivative.hpp:87-210` | `derivative`, `gradient`, `hessian` over a scalar function |
| Integration | `numerics/math/integration/adaptive_gauss_kronrod.hpp:133` | `integrate(const std::function<double(double)>&, ...)`, a minimal port (see below) |

Threading is not a hazard. `mcmc_sampler.hpp:34` records that the port is single-threaded by
design, with no threading dependency, so a callback is never invoked off the main thread.

## Architecture

Two new core headers, following the runner convention of phases 2 through 4.

**`numerics/support/callback_guard.hpp`.** `GuardedObjective` moves here out of
`optimizer_runner.hpp` and generalizes over the other signatures. It stays the one place a
host-language exception is latched so it survives a ported catch-all, and `optimizer_runner.hpp`
switches to it. One guard implementation, not two.

**`numerics/support/callback_runner.hpp`** plus `numerics/support/callback/{mcmc,bootstrap,gmm,
math}.hpp`, each a standalone-compiling header holding that group's `detail::run_<group>`. This
mirrors `toolbox_runner.hpp` and its `numerics/support/toolbox/` group headers. Serializable
configuration (sampler name, iterations, seed, prior specs, confidence-interval method) travels
as `options_json`; bulk data travels as native vectors; the callbacks arrive as `std::function`s.
Four callers drive it and none owns evaluation logic: `corehydror/src/callback.cpp`,
`corehydropy/src/bindings/callback.cpp`, `core/tests/test_fixtures.cpp`, and the dotnet oracle
emitter.

**An RNG handle.** The resample and Gibbs-proposal delegates receive the core's seeded
`MersenneTwister&`, which is what makes a seeded run reproduce. The handle is an opaque wrapper
over that reference, a cpp11 external pointer in R and a pybind11 class in Python, exposing
`uniform(n)` and `integers(n, min, max)`. It borrows the generator and never owns it; the
binding must not let it outlive the call.

**Exception path.** cpp11's `function` call converts an R-level error into a catchable C++
exception through its `unwind_protect` wrapper. The guard latches it and rethrows inside the same
protected frame, exactly as `ch_optim_run_` does today.

## Porting work

`AdaptiveGaussKronrod` is the one incomplete dependency: 147 lines against upstream's 313-line
`AdaptiveGuassKronrod.cs` and its 150-line `Integrator` base, described in its own header as a
minimal port for `VonMises::CDF`, with no settings, status, or evaluation counting. This phase
completes both so `quadrature()` reports status, iterations, and function evaluations the way
`optim_minimize()` does, and carries oracles from `Test_Numerics`.

## Surface

Both packages, R names shown; Python mirrors them verb for verb.

**`mcmc_posterior(log_likelihood, priors, sampler, ...)`.** Priors are a list of `distribution()`
objects, matching the C# `List<IUnivariateDistribution>` argument. All eight samplers, so Gibbs
becomes reachable for the first time through its `proposal` callback, and HMC and NUTS take an
optional `gradient` callback that falls back to the ported finite-difference gradient. Returns
the shape `mcmc_sample()` already returns: draws by chain, acceptance rates, MAP, posterior
summaries, R-hat, and effective sample size. Chain initialization keeps both ported modes, `MAP`
and `Randomize`; note that `MAP` reaches the user's function through the DifferentialEvolution
path, so it adds callback crossings before sampling starts.

**`bootstrap_custom(data, resample, fit, statistic, jackknife = NULL, ...)`.** All four
delegates, returning the confidence-interval surface `bootstrap_analysis()` has: Percentile,
BiasCorrected, Normal, BootstrapT, and BCa. BCa requires `jackknife`, which the runner validates
before the first replicate rather than failing partway through.

**`fit_gmm_moments(moment_conditions, initial, ...)`.** The delegate constructor, with optional
`jacobian`, `penalty`, and pointwise moment-condition callbacks.

**`root_find(f, lower, upper)`, `quadrature(f, lower, upper)`, `derivative(f, x)`,
`gradient(f, x)`, `hessian(f, x)`.** The integrator is `quadrature()` rather than `integrate()`
because the latter masks `stats::integrate` in R.

## Validation

**Named-callback catalog.** A fixture names its callback (`"log_likelihood": "normal_loglik"`)
and each of the four runners implements the catalog in its own language, the pattern
`fixtures/toolbox/optimizers.json` already uses with `"objective": "FXYZ"`. The C# MCMC tests
pass their own lambdas, so the catalog is drawn from `Test_RWMH.cs` and its siblings and the
emitter drives the real Numerics samplers with the same function. Oracles stay real.

**Fixture kinds.** One `callback` kind with a `group` field (`mcmc`, `bootstrap`, `gmm`, `math`),
matching how `toolbox` groups work, wired into all four runners. Plus one
`callback_cross_language` case, the counterpart to `fit_cross_language.json` and
`toolbox_cross_language.json`.

**Gates.** Every count must grow and none may regress: ctest 84/84, oracle gate 5209 reproduced
with 0 failed and 11 skipped, testthat 5634/0, pytest 1344, and `R CMD check --as-cran` at its
three known NOTEs with no WARNING.

## The cross-language guarantee, stated honestly

On the registry path a seeded MCMC run is bit-identical between R and Python because every
arithmetic operation happens in the shared core. On the callback path the draws still come from
the core PRNG, but the log-density is the user's own R or Python arithmetic, and the two
languages do not guarantee identical rounding for the same formula. MCMC amplifies this far
harder than the optimizers do: one flipped accept or reject changes every subsequent state, so
chains can diverge outright rather than drift.

The phase 4 examples already document the milder form of this for BFGS. The rule here:

- A seeded run reproduces across languages if and only if the user's function returns
  bit-identical values.
- The `callback_cross_language` fixture uses a log-density built from `+ - * /` only, a Gaussian
  kernel `-0.5 * sum((x - mu)^2) / sigma^2`, which is IEEE-deterministic and reproduces exactly.
- A catalog function using `log` or `exp` gets a fixture asserting structural invariants only,
  with no `oracle_skip` and no loosened tolerance, following the phase 9a and 10 precedent.

The documentation states this plainly rather than leaving users to discover it.

## Risks

- **Callback volume.** A 100,000-iteration chain means 100,000 crossings into R. Benchmark one
  representative chain, keep the crossing allocation-thin, and document the cost against the
  registry path.
- **Interrupts.** Ctrl-C during a long chain, on both sides.
- **Handle lifetime.** The RNG handle borrows a reference; a user who stores it and calls it
  later must get an error, not a dangling read.

## Out of scope

- **The `IModel` object protocol.** Letting a user's own model drive `fit_mle`, `fit_map`,
  `fit_bayesian`, `estimation_diagnostics`, and the analysis orchestrators needs an R or Python
  object implementing several methods, with lifetime and reentrancy concerns and no direct C#
  oracle for the object protocol. Deliberately deferred.
- **Matrix decompositions and special functions**, which the roadmap's phase 5 listed. Both are
  ported and internal, and both duplicate what base R and NumPy or SciPy already provide well.
  Surfacing them adds API surface without adding capability.
- **Everything never ported:** the `Machine Learning` layer, `HypothesisTests.cs`, the five
  remaining optimizers (PSO, SCE-UA, SimulatedAnnealing, AugmentedLagrange, MultiStart), and the
  other integrators (Simpson, Lobatto, Vegas, Miser, Monte Carlo).

## Deliverables

- Core: `callback_guard.hpp`, `callback_runner.hpp`, the four group headers, and the completed
  `AdaptiveGaussKronrod` with its `Integrator` base.
- Bindings: `corehydror/src/callback.cpp` with its R wrappers, and
  `corehydropy/src/bindings/callback.cpp` with its Python module, plus the RNG handle in each.
- Fixtures: the `callback` kind across all four runners, the catalog, and the cross-language case.
- Docs: an entry in both `corehydror/_pkgdown.yml` and the `quartodoc.sections` of
  `site/_quarto.yml` for every new export, the standing contract that pkgdown enforces loudly.
- Examples: two worked pairs, an R Quarto file and a Python notebook each, one for a custom
  posterior and one for a custom bootstrap statistic.
- `CHANGELOG.md`, the version bump to 0.7.0, and a `status.qmd` refresh that also corrects the
  stale "Internal" rows for multivariate distributions and copulas, both public since v0.5.0.
