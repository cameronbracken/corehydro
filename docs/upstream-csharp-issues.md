# Upstream C# issues: open log

Running log of **open** bugs, edge-case gaps, and consistency issues in the upstream USACE-RMC
C# libraries (`Numerics` @ `2a0357a` = v2.1.4, `RMC.BestFit` @ `c2e6192` = v2.0.0) found while
porting them to the C++ core, plus the port-side items still tracked as follow-ups. Entries whose
resolution is confirmed -- fixed upstream and ported, or fixed on the port side -- are moved to
`docs/upstream-csharp-issues-resolved.md`; a code comment that points here for a resolved finding
is one hop away. The July 2026 upstream sync re-checked every entry against the shipped source at
the pins above, and an August 2026 pass re-verified each entry below directly against the vendored
source and the current packages (v0.13.0): every upstream claim in this file was re-confirmed
present at the pins (a handful of scope refinements from that pass are recorded inline in the
affected entries), and at that date upstream `main` carried no code commits beyond the pins
(Numerics is at the pin; RMC-BestFit has one docs-only commit, the `docs/PROGRESS.md` deletion).

The port's governing rule is bit-for-bit fidelity with the C# (so the oracle values hold), so in
almost every case below the C++ **faithfully mirrors the C# behaviour** -- including its bugs --
and the divergence, where we made one, is documented at the call site. This document is the
backlog for (a) confirming each finding against upstream intent and (b) potentially submitting
fixes to the C# repositories. Nothing here blocks the C++/R/Python packages.

Severity: **BUG** = produces a wrong/undefined result a user could hit; **ROBUSTNESS** = works for
tested inputs but fragile at an edge; **CONSISTENCY/API** = surprising but arguably intentional;
**COSMETIC** = dead code / comments; **FIDELITY** = a documented, measured limit of C#-vs-C++
reproducibility, not a defect in either.

Each entry: what, where, evidence, how the port handled it, suggested fix.

---

## CONSISTENCY — CentralMoments(1000) resolves to the int-steps (trapezoidal) overload

- **Where:** `TruncatedDistribution.cs` (four moment getters), `Mixture.cs` and `CompetingRisks.cs`
  calling `CentralMoments(1000)`; overloads in `UnivariateDistributionBase.cs`
  (`CentralMoments(int steps=300)` vs `CentralMoments(double tolerance=1e-8)`).
- **What:** passing the integer literal `1000` binds to the **fixed-step trapezoidal** overload, not
  the adaptive-tolerance one. This may be intentional, but the two overloads with very different
  argument meanings (step count vs tolerance) are an easy foot-gun.
- **Port handling:** the C++ used to call the adaptive AdaptiveGaussKronrod integrator here and
  reproduced the C# only to the loose fixture tolerances then in force. It now calls
  `central_moments(1000)`, the same overload C# binds to. See the FIXED entry in
  `docs/upstream-csharp-issues-resolved.md` ("three classes computed their central moments with
  adaptive Gauss-Kronrod") for the measured before/after and the tightened pins. The foot-gun
  argument below is unaffected.
- **Suggested action:** verify the intent; consider renaming one overload (e.g.
  `CentralMomentsBySteps` / `CentralMomentsByTolerance`) to remove the ambiguity.

## ROBUSTNESS — NoncentralT moments use AdaptiveGaussKronrod (heavy) with no analytic fallback

- **Where:** `Numerics/Distributions/Univariate/NoncentralT.cs`, `Skewness`/`Kurtosis` via
  `CentralMoments`.
- **What:** not a bug, but the moments are pure numerical integration; for large `|λ|` with small
  `ν` the tails are heavy and integration is delicate.
- **Port handling:** the C++ uses a composite Gauss-Legendre quadrature (documented as accurate only
  near-symmetric until it is switched to the now-ported AGK). Only a limitation on the C++ side.
- **Suggested action:** none for C#; noted for context.

## COSMETIC — dead variables / heritage artifacts (remaining open items)

(The `NoncentralT.TT` item originally logged here was resolved in Numerics v2.1.4 and moved to
`docs/upstream-csharp-issues-resolved.md`.)

- Several distributions declare a member-field initializer (e.g. a scale of `0.0`) that the
  constructor immediately overwrites — harmless but misleading.
- `Numerics/Data/Interpolation/Support/Interpolater.cs`: `deltaStart = Math.Min(1,
  (int)Math.Pow(Count, 0.25))` always evaluates to `1` for any `Count >= 2` (`Math.Pow(2, 0.25)`
  already truncates to `>= 1`, and `Math.Min` caps at `1`), so the "correlated" hunt-vs-bisection
  search heuristic's tolerance is effectively a hardcoded `1`, not scaled with the table size as
  the formula suggests. Ported verbatim (see `core/include/corehydro/numerics/data/interpolation/interpolater.hpp`).
  **Still open** at `2a0357a` (line unchanged).

## CONSISTENCY/API — JoeCopula has no SetThetaFromTau, unlike its Archimedean siblings

- **Where:** `Numerics/Distributions/Bivariate Copulas/JoeCopula.cs`.
- **What:** `ClaytonCopula`, `AMHCopula`, and `GumbelCopula` each implement a `SetThetaFromTau`
  method-of-moments fit (Kendall's tau -> theta, closed-form for Clayton/Gumbel, Brent-solved for
  AMH). `JoeCopula` has no such method -- confirmed by `grep -n SetThetaFromTau` across the entire
  `Numerics/Distributions/Bivariate Copulas/` directory (three hits: Clayton, AMH, Gumbel; zero for
  Joe) and by `Test_JoeCopula.cs` having every other concrete copula's `Test_MOM_Fit` test method
  but not its own. This is not a wrong-output bug (nothing crashes or returns a bad value) --
  it is a missing feature relative to sibling classes that otherwise share an (almost) identical
  API surface, and there is no algorithmic reason Joe's tau could not be Brent-solved the same way
  AMH's is (Joe's generator, like AMH's, has no closed-form tau inversion, but that has not stopped
  the other three).
- **Evidence:** direct inspection of all five `Bivariate Copulas/*.cs` files (Task 8); this is also
  why the Phase 2 plan text and an earlier draft of `fixtures/README.md` incorrectly listed Joe as
  tau-capable (both were apparently written from the class's general shape/expected symmetry with
  Clayton/AMH/Gumbel rather than the actual source) -- corrected in both places during Task 8.
- **Port handling:** `joe_copula.hpp` (Task 8) does NOT add a `set_theta_from_tau` method, matching
  the C# source exactly; `joe_copula.json` has no `"tau"` fixture case, and the three
  `set_theta_from_tau_dispatch` glue functions (`core/tests/test_fixtures.cpp`,
  `corehydror/src/copula.cpp`, `corehydropy/src/bindings/copula.cpp`) plus the oracle emitter's
  `SetThetaFromTauDispatch` have no `"Joe"` branch (each has a NOTE comment explaining the
  omission).
- **Suggested C# fix:** add `JoeCopula.SetThetaFromTau`, e.g. `Theta = Brent.Solve(t => { ... } -
  tau, 1d, 100d)` mirroring `GumbelCopula`'s pattern but for Joe's tau relationship, for API parity
  with Clayton/AMH/Gumbel. Not urgent -- MPL/IFM/MLE fits already work for Joe via the shared
  `BivariateCopulaEstimation` path.

## CONSISTENCY/API — CompetingRisks' correlated CDF never touches MultivariateNormal.CDF()

- **Where:** `Numerics/Distributions/Univariate/CompetingRisks.cs`, `CDF(double)` /
  `CumulativeIncidenceFunctions`, calling `Numerics/Data/Statistics/Probability.cs`.
- **What:** for the `PerfectlyNegative`/`CorrelationMatrix` dependency modes, `CDF` builds a
  `MultivariateNormal` (`CreateMultivariateNormal()`) and calls `Probability.UnionPCM(cdf,
  _mvn.Covariance)` / `Probability.JointProbability(cdf, ind, _mvn.Covariance)`. The second call's
  3rd argument is a `double[,]` (`_mvn.Covariance`), which only matches the overload
  `JointProbability(IList<double>, int[], double[,]? correlationMatrix = null, DependencyType
  dependency = DependencyType.CorrelationMatrix)` — there is no `JointProbability(IList<double>,
  int[], MultivariateNormal)` overload. With a non-null `correlationMatrix` and the default
  (`CorrelationMatrix`) dependency, that overload unconditionally dispatches to
  `JointProbabilityHPCM` ("Haden Smith's modification of Pandey's Product of Conditional
  Marginals"), never to `JointProbabilityMVN`. `UnionPCM` reaches the same 3-arg overload
  internally for every inclusion-exclusion term. The upshot: the `MultivariateNormal` instance
  `CompetingRisks` constructs is used ONLY to hold/validate a mu/sigma covariance matrix — its
  `.CDF()` (the seeded Genz-Bretz MVNDST quasi-Monte-Carlo integrator for dimension >= 3) is never
  invoked anywhere in `CompetingRisks.cs`. This is surprising given the class name and the
  presence of a full `MultivariateNormal` instance, but is unambiguous from static C# overload
  resolution (confirmed by direct inspection, not runtime reflection, since C# overload binding is
  determined entirely by argument types at compile time).
- **Consequence for the port:** the correlated CDF/PDF paths this task un-defers are fully
  deterministic (no RNG) for any number of components — only `MultivariateNormal.BivariateCDF`
  (Drezner/Genz closed-form bivariate normal CDF) is used. This differs from the original task
  brief's assumption that CompetingRisks reaches "the MVN-backed joint path" (`JointProbabilityMVN`)
  and would need to worry about the C# `MultivariateNormal._MVNUNI` clock-seeded default (the
  concern Task 6's carry-forward note flagged for MultivariateStudentT/MultivariateNormal's own
  `dimension >= 3` `CDF()`, a different code path). Governed here by "the actual C#
  source over any brief or plan text" (this repo's standing rule): `core/include/corehydro/numerics/
  data/probability.hpp` ports `JointProbabilityHPCM`/`UnionPCM`, not `JointProbabilityMVN`/
  `UnionMVN`, which remain unported (no reachable caller).
- **Port handling:** mirrored faithfully; documented at length in `probability.hpp`'s header
  comment and `competing_risks.hpp`'s CDF comment.
- **Suggested action:** none required (not a bug — HPCM is a legitimate, if approximate,
  alternative to direct MVN-CDF integration) — flagged here purely so a future reader tracing
  "why does CompetingRisks build a MultivariateNormal but never call its CDF" doesn't need to
  re-derive the overload-resolution chain from scratch.

## CONSISTENCY/API — an all-zero RWMH proposal covariance is only safe under `Initialize = MAP`

- **Where:** `Numerics/Sampling/MCMC/RWMH.cs`, `ChainIteration` (`mvn[index].SetParameters(
  state.Values, ProposalSigma.Array)`), via `Numerics/Distributions/Multivariate/
  MultivariateNormal.cs`'s `SetParameters` -> `CholeskyDecomposition` ctor.
- **What:** `RWMH.ChainIteration` calls `SetParameters` with the CURRENT `ProposalSigma` on
  every single iteration. `MultivariateNormal.SetParameters` constructs a
  `CholeskyDecomposition` of that covariance unconditionally, and `CholeskyDecomposition`
  throws for any non-positive-definite input (an all-zero matrix's first diagonal pivot is
  exactly `0`, which fails the decomposition's `sum <= 0` guard). `Test_RWMH_NormalDist_RStan`
  constructs `new RWMH(priors, logLH, new Matrix(2))` -- a literal all-zero 2x2 proposal
  covariance -- and this is harmless ONLY because the test also sets `Initialize = MAP`, and a
  successful MAP initialization's `InitializeCustomSettings()` unconditionally overwrites
  `ProposalSigma` with the Fisher-information-derived covariance BEFORE the first
  `ChainIteration` call. Nothing in the public API prevents constructing (or leaving)
  `ProposalSigma` as all-zero under `Initialize = Randomize` or `UserDefined`, where no such
  override ever happens -- that configuration throws on the very first `ChainIteration`.
- **Evidence (reproduced against the real C# library):** a standalone console app (built
  against `upstream/Numerics/Numerics/Numerics.csproj`) constructing `new RWMH(priors, logLH,
  new Matrix(2))` with `Initialize = Randomize` and calling `Sample()` throws `AggregateException
  ("... Cholesky Decomposition failed. The input matrix is not positive-definite. ...")` (wrapped
  by the `Parallel.For` in `Sample()`) on the very first iteration. The identical construction
  with `Initialize = MAP` (the actual `Test_RWMH_NormalDist_RStan` configuration) succeeds.
  Substituting `Matrix.Identity(2)` for the proposal covariance under `Initialize = Randomize`
  succeeds and reproduces bit-close (~1e-15 relative) between the C++ port and the real C#
  library across all sampled draws.
- **Port handling:** mirrored faithfully -- `MultivariateNormal::set_parameters` throws
  identically for a non-positive-definite covariance (`cholesky_decomposition.hpp`'s `sum <=
  0.0` guard, unchanged from the Phase 2 port). This is not treated as a bug to fix; an
  all-zero proposal covariance is not a meaningful sampler configuration under either language.
  It IS a fixture-authoring hazard worth flagging: the `normal_short_exact` MCMC fixture case
  (`Initialize = Randomize`) uses a `proposal_sigma: "identity"` sentinel instead of the
  upstream test's literal `"zeros"` for exactly this reason -- see the divergence note in
  `fixtures/README.md`'s `mcmc_sampler` section.
- **Suggested action:** none required upstream (working as designed) -- flagged purely so a
  future MCMC-sampler port (ARWMH/DEMCz/DEMCzs/HMC/NUTS/Gibbs/SNIS) doesn't rediscover this the
  hard way when authoring a `Randomize`-initialized fixture case with a degenerate proposal.

## CONSISTENCY — SNIS sorts its resampling list by `Fitness`, not the `Weight` the surrounding comment/CDF describe; tied `-Infinity` fitness makes the sort order itself unstable across runtimes

- **Where:** `Numerics/Sampling/MCMC/SNIS.cs`, `Sample()`:
  `MarkovChains[0].Sort((x, y) => x.Fitness.CompareTo(y.Fitness));` and the CDF-construction loop
  immediately below it.
- **What:** two related issues at the same call site.
  1. **Sort key mismatch.** The line directly above the sort reads `// Sort list in ascending
     order of posterior weights`, and the very next lines build a CDF by accumulating
     `Math.Max(0.0, MarkovChains[0][i].Weight)` -- i.e. the algorithm's intent, and its
     correctness, depend on the list being sorted by `Weight` (the just-computed normalized
     posterior weight). The comparator actually sorts by `Fitness` (the raw, un-normalized
     log-likelihood/importance weight computed earlier in `Sample()`). For **naive Monte Carlo**
     (no importance distribution supplied), `Weight` and `Fitness` are numerically identical at
     the point of the sort (`weight = logLH` with no `mvn.LogPDF` correction), so this is
     unobservable. **With** an importance distribution (`Weight = Fitness -
     mvn.LogPDF(parameters)`), the two orderings genuinely differ -- the CDF is still
     mathematically valid either way (both `Sort` and the CDF loop iterate the SAME sorted list,
     so `Search.Sequential`'s binary-search precondition -- an ascending CDF -- still holds
     regardless of which key produced the ordering), but the specific `Output[0][i]` a given
     `rndOut[i]` plotting position resolves to differs from what sorting by `Weight` would
     produce.
  2. **Sort-tie instability.** `List<T>.Sort` is .NET's unstable introspective sort. Any model
     with a non-trivial fraction of `-Infinity`-fitness draws (common for a naive/wide-prior SNIS
     configuration, since `LogLikelihood` easily underflows for implausible parameter draws) has
     MANY tied elements at the bottom of the sort. An unstable sort is free to place those tied
     elements in ANY relative order -- which specific `-Infinity` draw lands at output index 0 vs.
     1 vs. ... is not determined by the algorithm's contract, only by the sort implementation's
     internal pivot/partition choices.
- **Evidence (reproduced against the real C# library):** the `fixtures/sampling/mcmc/snis.json`
  fixture's first authoring attempt anchored `chain_value` digest assertions to
  `MarkovChains[0]` indices `[0, 1, 2, 3, 4]` (the natural "first few" choice, matching every
  other MCMC fixture's convention). Every one of those `chain_value` assertions FAILED to
  reproduce against this port's `std::stable_sort`-based C++ (`ctest`'s `test_fixtures`), while
  the corresponding `chain_fitness` assertions (`-Infinity == -Infinity`, order-insensitive by
  construction) PASSED. The `normal_short_exact` case (100 naive-Monte-Carlo draws, wide Uniform
  priors from `Normal().GetParameterConstraints`) has 12 of 100 draws at exactly `-Infinity`
  fitness; the `normal_rstan` case (100000 draws, `Initialize = MAP` concentrating the importance
  distribution near the posterior mode) has only 3 of 100000. Re-anchoring the digest to the
  UNTIED, strictly-monotonic-fitness tail (the top 5 indices of each sorted list) reproduces
  cleanly (`normal_rstan` chain companions to ~1e-8 relative via the MAP/DE/Hessian path, per the
  usual P3.5 tolerance policy; `normal_short_exact`'s naive-Monte-Carlo companions to ~1e-15
  relative).
- **Port handling:** both aspects mirrored faithfully, not fixed -- `snis.hpp`'s `sample()` sorts
  by `.fitness` (not `.weight`), exactly matching the C# comparator's actual (not commented)
  behavior, using `std::stable_sort` (this port's usual convention for reproducing an unstable C#
  `List<T>.Sort`'s comparator -- see `mcmc_sampler.hpp`'s own `stable_sort` note for the same
  precedent in `InitializeChains()`). `stable_sort` does NOT make the two languages' outputs
  agree on tied-element ordering (a stable sort's tie-breaking is "preserve original order",
  which is only meaningful if BOTH languages process elements in the same original order AND use
  a stable sort -- true for C++ here, false for C#'s `List<T>.Sort`). `fixtures/README.md`'s
  SNIS tolerance-policy section documents the resulting draw-index hazard for future fixture
  authors.
- **Suggested C# fix:** for (1), either fix the comment to describe what the code does (sort by
  `Fitness`) or change the comparator to `x.Weight.CompareTo(y.Weight)` to match the comment and
  the CDF loop's own variable name -- these are NOT equivalent when an importance distribution is
  supplied, so this is a real behavioral choice, not just a comment fix, and should be resolved
  with the library's intent for how the resampled `Output` list should be ordered/weighted. For
  (2), switch to a stable sort (`OrderBy(x => x.Fitness).ToList()` or an explicit stable
  merge-sort) if bit-reproducible resampling across runs/platforms is a design goal; otherwise
  document that `Output`'s specific draw-to-plotting-position mapping is order-nondeterministic
  whenever tied fitness values occur.

## CONSISTENCY — `NextDoubles(length, dimension)` draws each column from its own fresh sub-`MersenneTwister`, not the caller's stream

- **Where:** `Numerics/Utilities/ExtensionMethods.cs`, `NextDoubles(this Random random, int
  length, int dimension)`.
- **What:** the 1-D overload (`NextDoubles(this Random random, int length)`) draws `length`
  values straight off the caller's own stream, as expected. The 2-D overload does something
  different: for each of the `dimension` columns it draws exactly ONE value off the caller's
  stream (`random.Next()`) to seed a brand-new `MersenneTwister` (or plain `Random`, if the
  caller wasn't itself a `MersenneTwister`), then fills that entire column by advancing the
  FRESH sub-generator's own stream `length` times. The caller's stream is therefore consumed at
  a rate of exactly `dimension` draws total (one per column, to seed the sub-generators), not
  `length * dimension` -- every actual random double returned comes from one of the `dimension`
  independent sub-streams, never from the parent stream directly. This is a real behavioral
  choice (not obviously a mistake -- it decorrelates columns even when the parent stream has a
  short period or column-wise correlation), but it is easy to miss reading only the method
  signature: a caller expecting "draw `length * dimension` numbers off my stream in row-major
  order" (the naive reading `NextDouble()` in a nested loop would produce) gets a materially
  different, though still uniform, output.
- **Evidence:** direct code reading (`ExtensionMethods.cs` lines ~144-157); `Test_NextDoubles2D`
  (`Test_Numerics/Utilities/Test_ExtensionMethods.cs`) only range-checks the output (`[0, 1)`
  for every cell), so it does not itself distinguish this from the naive single-stream reading --
  the sub-stream-per-column behavior was confirmed by tracing a seeded `MersenneTwister(12345)`
  through both this method and a column-by-column reconstruction using `new
  MersenneTwister(random.Next())` per column, which reproduce identically.
- **Port handling:** transcribed exactly -- `extension_methods.hpp`'s `next_doubles(rng, n, dim)`
  overload constructs one `corehydro::numerics::sampling::MersenneTwister sub(random.next())` per
  column and fills that column from `sub`, in dimension order (see the header's file comment).
  This pattern is load-bearing for `SNIS::sample()`, which calls
  `_masterPRNG.NextDoubles(Iterations, NumberOfParameters)` once up front; `fixtures/special_
  functions/extension_methods.json`'s `next_doubles_grid` cases lock the exact per-cell values
  this produces from a known seed, independently of any MCMC sampler fixture.
- **Suggested C# fix:** none required (working as designed) -- flagged purely as a
  non-obvious-from-the-signature quirk for anyone reusing this overload outside the ported
  call sites (SNIS is the only current consumer within this port's scope).

## ROBUSTNESS — `Bootstrap.ComputeAccelerationConstants`'s `Tools.ParallelAdd` reduction is not bit-reproducible run-to-run

- **Where:** `Numerics/Sampling/Bootstrap/Bootstrap.cs`, `ComputeAccelerationConstants` (its
  `Parallel.For(0, N, idx => { ... Tools.ParallelAdd(ref I2[i], diff * diff); Tools.ParallelAdd
  (ref I3[i], diff * diff * diff); ... })` loop), backed by `Numerics/Utilities/Tools.cs`'s
  `ParallelAdd` (a CAS retry loop over `Interlocked.CompareExchange`).
- **What:** `ParallelAdd` is a correct lock-free accumulator (no lost updates), but it does NOT
  fix the ORDER in which concurrent jackknife-sample contributions land in `I2[i]`/`I3[i]` --
  that order depends on the .NET thread pool's scheduling of the `Parallel.For` partitions, which
  is not guaranteed deterministic across runs, machines, or core counts. Floating-point addition
  is not associative, so a different accumulation order can (in general) produce a different
  last-few-bits sum, even though every run adds the exact same set of addends. The resulting BCa
  acceleration constant, and therefore the BCa confidence interval bounds, inherit this
  run-to-run variability.
- **Evidence (reproduced against the real C# library):** the oracle emitter's `--dump` output for
  the `bca` bootstrap fixture case was captured across four independent runs of the SAME process
  invocation, sequentially, on the development machine, and diffed byte-for-byte -- all four
  runs were BIT-IDENTICAL (a low-core-count environment apparently schedules this small,
  100-jackknife-sample `Parallel.For` deterministically in practice), i.e. the measured wobble
  was exactly `0` on this machine, though the reduction remains order-dependent BY CONSTRUCTION
  and a different core count/thread-pool configuration/.NET version could legitimately produce a
  different summation order and a different last-few-bits result.
- **Port handling:** this port replaces the `Parallel.For` + `Tools.ParallelAdd` pair with a
  plain serial accumulation in jackknife-index order (see `bootstrap.hpp`'s file header BCa
  HAZARD note and `compute_acceleration_constants`'s own comment) -- deterministic within the
  C++ port, but not a bit-for-bit reproduction of C#'s reduction order. The `bca` fixture case's
  CI-bound assertions therefore use a LOOSE `mode: "rel", tol: 1e-6` (three orders of magnitude
  looser than every other CI method's `1e-9`), sized to the reduction's inherent
  order-dependence rather than to any measured instability (which was zero on this machine) --
  see `fixtures/README.md`'s `bootstrap` schema section for the full tolerance rationale.
- **Suggested C# fix:** none required upstream for correctness (the CAS loop is race-free); if
  bit-reproducible BCa intervals across runs/machines becomes a design goal, replace the
  `Parallel.For`/`ParallelAdd` pair with a deterministic-order reduction (e.g. `Parallel.For`
  into per-partition local accumulators, combined in a fixed final pass) or a plain serial loop.

## BUG — thinned DEMCzs population-sampler stream diverges C#-vs-C++ (surfaced tightening the UnivariateAnalysis analysis oracle)

- **Where:** `Numerics/Sampling/MCMC/Base/MCMCSampler.cs` `Sample()` / `SampleChain()` interaction
  with `ThinningInterval > 1` for population samplers (DEMCz/DEMCzs), driven through
  `RMC.BestFit`'s `UnivariateAnalysis` at its `SetDefaultSimulationOptions` default
  (`ThinningInterval = max(1, min(100, 10*d)) = 20` for a 2-parameter Normal).
- **What:** at `thinning_interval = 1` the seeded DEMCzs chain reproduces bit-identically between
  the real C# `BayesianAnalysis` and the C++ port (proven by `fixtures/estimation/bayes_normal.json`
  `chain_value` at `rel 1e-11`, and re-proven here: at `thinning_interval = 1` all eight
  UnivariateAnalysis oracles reproduce C#-vs-C++ at `rel 1e-9`). At the default
  `thinning_interval = 20` (identical config on both sides -- `chains = 4`, `initial_iterations =
  200`, `iterations = 100`, `warmup = 50`, `output_length = 400`, `seed = 12345`) the two streams
  DIVERGE materially: `parameter[0]` is `16775.69` (C#, thin=1) vs the divergent `16528.6` (C++,
  thin=20) vs `16509.1` (C#, thin=20). Because the SampleChain thinning loop
  (`for j in 1..ThinningInterval: state = ChainIteration(...)`) is byte-identical in both ports, the
  divergence is NOT in the thinning loop itself; it is in how the extra inner `ChainIteration`
  draws interact with the shared population archive (`PopulationMatrix`) update cadence over a
  thinned run. This is a genuine port-fidelity defect confined to `thinning_interval > 1` on the
  population samplers; single-step and every already-shipped Bayesian fixture (all thinning=1) are
  unaffected.
- **Evidence:** Task A11 oracle work. `dotnet` emitter dump of the real C# UnivariateAnalysis vs the
  C++ `test_fixtures` runner over the same construct; the two agree to `rel 1e-9` at thin=1 and
  disagree at `rel ~1e-3` at thin=20.
- **Port handling (A11):** the `UnivariateAnalysis` smoke fixture is PINNED to
  `thinning_interval = 1` (the proven bit-identical path) so its tightened oracle is exact and
  reproduces across C++/R/Python AND the C# dotnet gate. All four analysis runners (C++ test,
  R/Python glue, emitter) honor an explicit `thinning_interval` override. The default-thinning
  (thin=20) UnivariateAnalysis path is left as a tracked follow-up, NOT loosened.
- **Suggested action (follow-up task):** bisect the thinned population-sampler `ChainIteration`/
  archive-update ordering between `MCMCSampler.cs` and `mcmc_sampler.hpp` to find where the extra
  inner iterations consume the shared archive differently, and fix the C++ port (or, if the C# is at
  fault, document the intentional divergence). Until then, seeded DEMCz/DEMCzs runs with
  `thinning_interval > 1` are not oracle-guaranteed C#-vs-C++.

## ROBUSTNESS — DIC / WAIC parallel-reduction non-reproducibility (extends the BCa `Tools.ParallelAdd` finding)

- **Where:** `RMC.BestFit`'s `BayesianAnalysis.ComputeDIC` (and the population sampler's pooled
  `Output` accumulation), following the same `Parallel.For`/`Tools.ParallelAdd` pattern as the
  `Bootstrap.ComputeAccelerationConstants` finding above.
- **What:** like the BCa acceleration constant, DIC's parallel reduction over posterior draws is
  not bit-reproducible C#-to-C# run-to-run (~1e-13 relative), for the same reason:
  `ParallelAdd`'s CAS-retry loop is race-free but not order-fixed, and floating-point addition is
  not associative.
- **Evidence:** measured during Task T12's oracle verification of the `bayes_normal` fixture; the
  fixture's DIC tolerance (`rel: 1e-6`) is sized to this reduction-order noise, not to any
  C++-vs-C# divergence.
- **Port handling:** this port computes DIC/WAIC/LOOIC with a plain serial reduction
  (deterministic within C++, consistent with the same choice made for BCa's acceleration
  constant), documented at the relevant `bayesian_analysis.hpp` diagnostics code and in
  `fixtures/README.md`'s tolerance-policy notes.
- **Suggested C# fix:** none required for correctness; if bit-reproducible DIC/WAIC across runs
  is a design goal, replace the `Parallel.For`/`ParallelAdd` reduction with a deterministic-order
  accumulation, matching the suggested fix for the BCa finding above.
- **Scope correction (2026-08-28 re-check):** the finding covers `ComputeDIC` (the
  `Tools.ParallelAdd(ref dicHat, z)` combiner at `BayesianAnalysis.cs:1405`) and `ComputeWAIC`
  (`ParallelAdd` on `totalLppd`/`totalPWaic` at `:1523-1524`) only. `ComputePSISLOO`'s
  `Parallel.For` loops write into per-index slots rather than a shared accumulator, so its
  reduction is order-independent; earlier statements (including the July 2026 reconciliation
  summary) that lumped PSIS-LOO in with DIC/WAIC overstated the scope.

## DESIGN NOTE (not a bug) — Bulletin17CDistribution GMM is always just-identified, so the J-stat specification test is unreachable

- **Where:** `Models/UnivariateDistribution/Bulletin17CDistribution.cs` @ fc28c0c, lines 434 and 437.
- **What:** `NumberOfParameters => Parameters.Count` and `NumberOfMomentConditions => Parameters.Count`
  are defined identically, so a `Bulletin17CDistribution` GMM fit is ALWAYS just-identified
  (q = p). `GeneralizedMethodOfMoments.DegreeOfFreedom = max(0, q - p)` is therefore always 0,
  `JStatPval` is always `NaN`, and the over-identified J-statistic specification test (`GetGamma`
  chi-square path) can never fire through this model. Confirmed against the real library by the
  B12 emitter: the LP3 exact-data fit dumps `JStat ≈ 2.13e-6` (pure catastrophic-cancellation noise
  in `g' V⁻¹ g`, since `g(θ̂) ≈ 0`) and `JStatPval = NaN`.
- **Consequence for oracles:** the GMM/B17C fixture (`fixtures/estimation/gmm_bulletin17c_smoke.json`)
  asserts `j_stat` with an ABSOLUTE tolerance against 0 (the exact residual is unreproducible across
  compilers — the C++ core lands a differently-signed ~-5e-7) and `j_stat_pval` as `nan` via
  `mode:equal`. No censored/threshold B17C DataFrame can change q relative to p, so there is no
  reachable over-identified oracle to add. Every other GMM/B17C quantity (params, standard errors,
  covariance, correlation, quantile variance, the seeded ISimulatable stream) IS deterministic and
  reproduces to ~1e-12 or better against the real library.
- **Port handling:** the C++ `Bulletin17CDistribution` mirrors both accessors, so the property holds
  identically in the port; no divergence.
- **B13 follow-up:** the brief's "J-statistic p-value where over-identified" and the extended
  Normal-family / censored / TwoStep / Link / ConditionalMoments / Penalty / MomentConditions dump
  coverage were NOT added as fixture cases: the p-value case is structurally impossible for B17C, and
  the remaining internal accessors (Link/InverseLink/DLink, ConditionalMoments, ParametersFromMoments,
  Penalty.Function, MomentConditions G/S) are not on the B11-established public GMM dispatch surface
  (parameter / standard_error / covariance / correlation / j_stat / j_stat_pval / quantile_variance /
  simulated_value) and would need new dispatch arms in all three runners. They are corroborated by the
  B4/B8/B10 C++-only ctests and remain a severable follow-up.

## CONSISTENCY — StudentT bivariate copula degrees-of-freedom clamps to the upper bound 30 (the Gaussian limit) under a strong dependence

- **Where:** `Numerics/Distributions/Bivariate Copulas/StudentTCopula.cs` (the `df` parameter
  bounds, upper = 30) reached via a `BivariateDistribution` StudentT-copula IFM fit @ fc28c0c.
- **What:** for a strongly dependent bivariate sample, the StudentT copula's degrees-of-freedom
  estimate saturates at its upper bound `30` (where the StudentT copula is numerically
  indistinguishable from the Gaussian/Normal copula). This is not a wrong result -- it is a valid
  deterministic boundary optimum: both C# and C++ converge to exactly `30.0`.
- **Evidence:** the `bivariate_smoke.json` StudentT/IFM case asserts `df == 30.0` at rel `1e-8`;
  both the real C# (via the emitter) and the C++ port land on exactly the boundary, so it is a
  stable oracle rather than an optimizer artifact.
- **Port handling:** mirrored faithfully; the boundary value is asserted directly as the oracle.
- **Suggested action:** none (design note, not a bug) -- flagged so a future reader does not treat
  the pinned `df == 30` boundary oracle as a fit that failed to converge into the interior.
- **Update (2026-08-28 re-check):** the bound is now documented upstream as a deliberate
  statistical choice: at the current pin, `StudentTCopula.ParameterConstraints` carries a
  `<remarks>` block explaining that for nu >~ 30 the Student-t copula is empirically
  indistinguishable from the Gaussian copula at typical hydrologic sample sizes, so a Uniform
  prior over the flat [30, inf) region would drag the posterior mean; users needing very high nu
  should use the Normal copula directly. This entry stays as a design note, no longer an
  unexplained magic number.

## COSMETIC (port bookkeeping) — the emitter public-path corroboration for three internal-support ctests is a documented deferral

- **Where:** the P4 fix pass, spanning `core/tests/test_box_cox.cpp`,
  `core/tests/test_spatial_correlation.cpp`, and
  `core/tests/test_cached_mvn_gaussian_copula.cpp`.
- **What:** the P4 brief's section-1 named an optional "public-path corroboration" deliverable --
  dump BoxCox transform / correlation-model `Evaluate` / CachedMVN `LogPDF` spot values through the
  real C# via the oracle emitter to back the transcribed `1e-10`/`1e-12` leaf oracles in those
  three internal-support ctests. It was deferred, not implemented.
- **Evidence:** the three ctest headers each carry a "Deferred to P5" note alongside their existing
  "Skipped C# test methods" list; the whole corpus still reproduces 0-failed without it. Re-checked
  2026-08-28: P5 and P6 both shipped without wiring it and the headers still carry the note, so the
  deferral stands at v0.13.0.
- **Port handling:** deferred with justification (redundant defense-in-depth -- the leaf oracles are
  transcribed values-unaltered from the upstream C# test literals and recomputed inline from the
  identical closed-form expressions, so they already ARE the C# public-path values; and driving them
  through the emitter conflicts with the standing constraint that public-API oracles live only in
  `fixtures/` while internal-support values stay C++-only ctests).
- **Suggested action:** wire the optional emitter public-path corroboration for these three
  internal-support families IF the fixture/harness model is later extended to non-distribution
  support classes.

## FIDELITY (D6) — AR/MA/Mixture seeded DEMCzs analysis oracles diverge C#-vs-C++ by chaotic short-chain sensitivity, not a model bug

- **Where:** `RMC.BestFit/Analyses/TimeSeries/{ARAnalysis,MAAnalysis}.cs` and
  `RMC.BestFit/Analyses/Univariate/MixtureAnalysis.cs` @ fc28c0c, each driving a seeded DEMCzs
  `BayesianAnalysis` over `AutoRegressive`/`MovingAverage`/`MixtureModel`; surfaced tightening the
  D5 smoke fixtures `fixtures/analyses/{ar,ma,mixture}_analysis_smoke.json` against the D6 emitter.
- **What:** the seeded DEMCzs MCMC chain for these three families settles on a materially different
  point C++ vs the real C# (e.g. the AR MAP objective lands near `~58` in one and `~16` in the
  other), so the mode/mean frequency-curve scalars do not reproduce to a point tolerance. ROOT CAUSE
  is **(B) inherent chaotic sensitivity of a short chain on a near-flat surface, NOT a port defect**:
  an independent read-only diagnostic compared the deterministic `DataLogLikelihood` (and the full
  posterior log-density) across 238 parameter vectors and found C++ matches C# to `<= 3 ulp`, with
  the Mixture likelihood **bit-identical** on the whole grid. A short 100-iteration chain on the flat
  AR/MA intercept ridge (`mu` is only weakly identified as `phi -> 1`) or on the symmetric bimodal
  Mixture surface amplifies a sub-ulp floating-point reassociation into a single accept/reject flip
  or a differential-evolution basin flip, which then propagates to a visibly different chain
  endpoint. This is the same mechanism as the Phase-3 HMC/NUTS cross-platform precedent (a
  deterministic-density-identical sampler whose discrete accept/reject path is chaotically sensitive
  to last-ulp reassociation): the densities agree, the trajectory endpoint need not.
- **Evidence:** the D6 read-only 238-vector `logLik` comparison (C++ vs the real C# library) plus the
  emitter dump: deterministic densities agree to `<= 3 ulp` (Mixture bit-identical) while the seeded
  DEMCzs endpoint diverges (AR MAP `~58` vs `~16`). By contrast the CompetingRisk and PointProcess
  analyses reproduce their full curves to `~1e-10` and their fixtures were TIGHTENED to exact; ARIMA
  and ARIMAX remain structural (their four-parameter differenced posteriors are chaotic even
  same-family, per the D5 report).
- **Port handling:** the three affected fixtures (`ar`/`ma`/`mixture`) assert only build-stable
  STRUCTURAL invariants -- the frequency-curve length (`curve_length`) and, for AR, the
  deterministic first mode-curve ordinate (`mode_curve[0]`) -- with honest source notes on each
  case. There is **NO `oracle_skip` and NO tolerance loosening** for these three: the trajectory
  scalars are simply not asserted, because a chaotic accept/reject flip is not something any
  reasonable tolerance can absorb. `verify_oracles.py` reproduces the structural assertions cleanly
  (part of the `4003 reproduced, 0 failed` corpus). This is distinct from the
  PriorInfluenceDiagnostics divergence, which was a deterministic name-keyed dedup rather than a
  chaotic stream; its three `oracle_skip` assertions were retired when Phase 10 ported
  `Distribution.ParameterNames` (see the resolved log).
- **Suggested action:** none required for correctness -- the densities are proven identical, so the
  fit is faithful; the short-chain endpoint is inherently non-reproducible across float
  reassociation. If exact analysis-curve oracles are wanted for these families, pin a longer/seed-
  robust chain (or a thin=1 single-chain config) whose endpoint is no longer basin-sensitive, the
  same mitigation used for the thinned-DEMCzs finding above.

## FIDELITY (X12) — Bivariate / Coincident / Composite / RatingCurve / SpatialGEV seeded DEMCzs analysis curves diverge C#-vs-C++ by chaotic short-chain sensitivity, not a model bug

- **Where:** the Phase-10 analysis orchestrators (`Analyses/Bivariate/BivariateAnalysis.cs`,
  `CoincidentFrequencyAnalysis.cs`, `Analyses/Univariate/CompositeAnalysis.cs`,
  `Analyses/RatingCurve/RatingCurveAnalysis.cs`, `Analyses/SpatialExtremes/SpatialGEVAnalysis.cs`)
  and their `fixtures/analyses/*_smoke.json` oracles.
- **Symptom:** the seeded DEMCzs posterior MAP (and every posterior-derived curve/band: joint-
  exceedance mode/mean/CI, composite frequency curve, rating-curve ribbon, per-site GEV/quantile
  bands + regional curve) reproduces between the C# emitter and the C++ core only to **~1e-6
  relative**, not the 1e-8 the deterministic quantities hold. Same phenomenon documented for the
  D5/D6 AR/MA/Mixture analyses above, now confirmed for the copula- and spatial-model families.
- **Root-cause diagnosis (chaotic-sensitivity rule):** the deterministic model math matches C#/C++
  to floating-point precision — the bivariate copula MLE `parameter` + `max_log_likelihood`
  reproduce to rel 1e-8 (`fixtures/estimation/bivariate_smoke.json`), the Normal-copula CDF (Drezner/
  Genz bivariate-normal integration) has a curated rel 1e-8 companion (`normal_copula.json`), and the
  Normal MLE + inverse-CDF path reproduces to 1e-9 (the BootstrapAnalysis fixture). The divergence is
  therefore the seeded 300-iteration DEMCzs chain **amplifying sub-1e-8 model-density ULP differences**
  (copula bivariate-normal integration, GEV link/CDF evaluation) into a ~1e-6 MAP drift — inherent
  chaotic sensitivity of a short chain on a flat/near-symmetric surface, NOT a port bug.
- **What the fixtures assert:** per the rule, the posterior-dependent curves are **not pinned** — no
  `oracle_skip`, no loosened tolerance. Each fixture keeps the deterministic invariants that DO
  reproduce bit-identically across C#/C++/R/Python: `curve_length` (all five), `site_count` (spatial),
  the CoincidentFrequency `z_output` bins **and its z=0 exact-symmetry point (AEP == 0.5)**. Three
  sibling fixtures are pinned in FULL to exact oracles because they carry **no MCMC chain** (or a
  discrete statistic that survives the drift): `bootstrap_analysis_smoke.json` (deterministic MLE +
  bit-exact parametric-bootstrap MT, rel 1e-8/1e-9), `prior_predictive_check_smoke.json` (prior-
  sampled MT, rel 1e-9), and `posterior_predictive_check_smoke.json` (the p-values are discrete
  multiples of 1/200, exact to abs 1e-9). `verify_oracles.py` reproduces every one of these against
  the real RMC.BestFit / Numerics library.
- **Suggested action:** none required for correctness — the model densities are proven identical, so
  the fits are faithful; the short-chain endpoint is inherently non-reproducible across float
  reassociation. If exact posterior-curve oracles are wanted for these families later, pin a longer /
  seed-robust chain whose MAP is no longer basin-sensitive (the same mitigation noted for the
  thinned-DEMCzs and D5/D6 findings).

## FIDELITY (X12) — the two un-gated Bulletin17C uncertainty arms (LinkedMVN X8 / pivot-BiasCorrected bootstrap X9) gain no numeric cross-language oracle beyond the method-independent Cohn CI

- **Where:** `RMC.BestFit/Analyses/Bulletin17CAnalysis.cs` @ fc28c0c, `ParseUncertaintyMethod`'s
  `LinkedMultivariateNormal` and `BiasCorrectedBootstrap` arms (ported at
  `core/include/corehydro/analyses/univariate/bulletin17c_analysis.hpp`, the two formerly-throwing
  dispatch cases replaced by X8/X9), and the two fixture cases
  `fixtures/analyses/bulletin17c_analysis_smoke.json`: `lp3_linked_multivariate_normal` and
  `lp3_bias_corrected_bootstrap`.
- **What:** the X8/X9 work un-gated the two heavy uncertainty-quantification paths (LinkedMVN
  link-fitting + the pivot / BiasCorrected parametric bootstrap) that populate
  `Bulletin17CAnalysis.Results` (the method-dependent parameter-set ensemble band). The two smoke
  fixture cases drive those dispatch arms end-to-end -- proving the LinkedMVN link-builders and the
  pivot bootstrap run to completion without throwing in BOTH the C# emitter (`RunAsync`) and the
  C++/R/Python runners -- but the value they ASSERT is the deterministic Cohn-style delta-method CI
  (`point_estimate` / `lower_ci` / `upper_ci` / `parameter`), which is computed off the RNG-free GMM
  point estimate ALONE and is therefore INDEPENDENT of the UncertaintyMethod
  (`ComputeCohnStyleConfidenceIntervals`, C# ~666-673, is unchanged whichever arm runs -- confirmed
  empirically by the real-library emitter dump reproducing all three cases identically). So the
  method-dependent UQ ensemble output itself (`analysis_results_`, the band the two arms actually
  populate) gains NO numeric cross-language oracle: the fixture surfaces only the Cohn CI, never the
  ensemble band.
- **Why not pinned:** the LinkedMVN/pivot ensemble is a seeded parameter-set draw over a
  link-function fit that is itself plausibly basin-sensitive (the same short-chain chaotic-sensitivity
  family documented above for the five DEMCzs analysis curves), so a full-curve oracle would need a
  chaotic-sensitivity root-cause check before it could be pinned to a point tolerance. Per the binding
  rule this is left as an HONEST documented residual, NOT an `oracle_skip` mask and NOT a loosened
  tolerance -- the deterministic Cohn CI that IS asserted reproduces cleanly against the real library
  (part of the 4069-reproduced corpus), and the dispatch arms are proven non-throwing end-to-end.
- **Suggested action:** none required for correctness -- both arms are faithful ports and run to
  completion. A follow-up that wants numeric validation of the X8/X9 draws would add dispatch
  accessors on the ensemble CI (`analysis_results_`) plus a chaotic-sensitivity check on that band
  before pinning it, the same treatment applied to the seeded analysis curves.

## FIDELITY (T19b) — Interval-censored B17C bootstrap: GMM stopping-rule knife edge, not a BFGS port divergence

- **Where:** `RMC.BestFit/Estimation/GeneralizedMethodOfMoments.cs` `EstimateIterative` (@ c2e6192)
  driving `Numerics/Mathematics/Optimization/Local/BFGS.cs` (@ 2a0357a), reached through
  `Bulletin17CAnalysis.GetParameterSetsFromParametricBootstrap`'s warm-start arm
  (`cloneWithDataFrame == true`). Ported at
  `core/include/corehydro/estimation/generalized_method_of_moments.hpp` and
  `core/include/corehydro/numerics/math/optimization/bfgs.hpp`.
- **Symptom:** for the `lp3_bootstrap_warm_start` fixture configuration (20 exact peaks + 1
  interval-censored observation, B = 50, seed 12345) the C++ core reports
  `boot_total_retries = 2` where the real C# reports `0`. Two replicates (idx 17 and 32) exhaust
  the 2000-evaluation BFGS budget and are retried. The resulting ensemble `mean_curve` differs
  from C# by ~1.4e-4 relative, so it is not pinned.
- **What was ruled out.** The BFGS transcription is faithful: `BFGS.cs`, `Optimizer.cs`,
  `NumericalDerivative.Gradient`, and `Tools.SumProduct`/`Sqr`/`Distance` were compared
  line-for-line against the 2a0357a pin with no difference. Decisively, the **real C# BFGS
  reproduces the stall**: driven through the emitter at the same near-stationary warm start on the
  same replicate surface (idx 32, the pass-2 point, same analytic GMM gradient, same bounds, same
  2000-evaluation cap and 1e-8 tolerances) it terminates `MaximumFunctionEvaluationsReached` after
  95 outer iterations and 2000 evaluations, moving Q only from 5.386e-16 to 5.113e-16. The
  per-iteration cost matches the C++ core exactly (~21 evaluations: one trial step plus 20 failed
  Zoom bisections), i.e. both runtimes enter the same non-progressing outer loop. The bootstrap
  resamples themselves are identical across runtimes to ~1e-13, so the divergence is not in the
  seeded resampling path either.
- **Actual mechanism.** The divergence is upstream of BFGS, in whether a **third** iterative-GMM
  refinement pass runs at all. `EstimateIterative` compares consecutive passes with
  `Tools.Distance(newValues, oldValues) < AbsoluteTolerance` (1e-8) — and that is the only test
  that can stop the loop here, since the companion `relChange` test is pinned well above its
  tolerance by the `1e-15` floor in its denominator. The C++ and C# pass-1 fits land ~1.8e-6 apart
  and the two runs then straddle the distance threshold: **1.23e-8 in C++** (a third pass runs,
  starts at a stationary point, stalls, and the replicate is retried) versus **3.3e-11 in C#**
  (converged at pass 2, no third pass).
- **Why the residual is inherent, by direct measurement (T19b).** This is NOT a flat-objective
  story — at the converged idx=17 solution Q is 2.35e-17, and displacing the skew by 1.8e-6 raises
  it to 1.32e-12 (dQ/Q ~ 5.6e4), so the surface is well curved at that scale. It is extreme
  *conditioning* of the censored resample fit. Perturbing a SINGLE resampled exact value on
  replicate idx=32 by a relative **1e-13** displaces the converged fit by **2.4e-5**
  (amplification ~**2.4e8**) and flips the third pass from stalling to succeeding
  (`MaximumFunctionEvaluationsReached`/4292 evaluations -> `Success`/351); the same perturbation at
  relative **1e-15** leaves the fit **bit-identical** (same parameters, same status, same 4292
  evaluations). Meanwhile the UNRESAMPLED parent fit on the same censored frame reproduces
  C++-vs-C# to **3.0e-14 relative** on the location parameter (3.3e-12 relative worst-case across
  the three parameters, 7.6e-13 Euclidean) — so there is no upstream divergence in the GMM
  weighting, the moment conditions, or the T18 clone/ROS defaults. The ~1e-13 ULP differences that
  are unavoidable between two runtimes sit exactly in the band where this fit's conditioning turns
  them into an O(1e-5) parameter displacement and a different pass count.
- **Upstream weakness this exposes.** `EstimateIterative`'s convergence test is an ABSOLUTE
  parameter distance against a tolerance that is also handed to the optimizer as a RELATIVE
  function tolerance. Whenever the objective's resolution in some parameter direction is coarser
  than `AbsoluteTolerance`, the pass-to-pass distance is dominated by optimizer stopping noise and
  the number of GMM passes becomes runtime-dependent. Additionally, BFGS has no stagnation exit:
  when a line search returns the starting point, `xi` and `dg` both become zero, the inverse-Hessian
  update is skipped, the search direction is unchanged, and the outer loop repeats identically until
  the evaluation cap. The Numerical Recipes `dfpmin` this is derived from guards exactly that case
  with a `TOLX` parameter-change test — `TOLX` is declared in `BFGS.Optimize` but never used.
- **What the fixtures assert:** per the binding rule, the retry counters and the ensemble
  `mean_curve` for `lp3_bootstrap_warm_start` are **not pinned** — no `oracle_skip`, no loosened
  tolerance. The deterministic quantities that DO reproduce (the parent GMM `parameter`, the
  replicate/valid/failed counts, zero Mahalanobis rejections) stay pinned to real C#. Same-language
  completeness is covered by `core/tests/test_bulletin17c_analysis.cpp`
  `test_run_bootstrap_warm_start_structural`.
- **Suggested action:** none required for correctness — the port is faithful and both runtimes
  produce equally converged fits. If cross-language retry parity is wanted later, the fix belongs
  upstream: give `EstimateIterative` a scale-relative parameter-convergence test (or a separate
  parameter tolerance), and/or restore the `dfpmin` `TOLX` stagnation exit in `BFGS.Optimize` so a
  stationary warm start returns immediately instead of burning the evaluation budget. Either change
  would alter oracle values and must be paired with a re-pin.

## BUG — SetLowOutliersFromMGBT leaves plotting positions stale, so a headless Bulletin 17C fit over a censored record throws

- **Where:** `RMC.BestFit/Models/DataFrame/DataFrame.cs`, `SetLowOutliersFromMGBT()` and
  `SetLowOutliersFromThreshold()`; consumed by `GetNonparametricMomentsROS()` and
  `Bulletin17CDistribution.ComputeDefaultInitials()`.
- **What:** Both public low-outlier setters change the `IsLowOutlier` flags that the
  Hirsch-Stedinger plotting positions depend on, but neither recomputes those positions. They end
  with `RaisePropertyChange("LowOutliers")`, so in the WPF application the recomputation happens
  through the `INotifyPropertyChanged` cascade. A **headless** caller (a script, a test, or any
  non-GUI consumer) gets a frame whose flags say "censored" while every `PlottingPosition` is
  still at its `0.0` default. The same gap applies to a frame assembled with a `ThresholdSeries`
  and no low outliers at all: the ROS branch is taken for `NumberOfLowOutliers > 0 ||
  ThresholdSeries.Count > 0`, so a perception-threshold record hits it too.
- **Why it matters:** `GetNonparametricMomentsROS()` is exactly such a consumer. It regresses the
  uncensored values on `Normal.StandardZ(PlottingPositionComplement)` to impute the censored ones.
  With every position at 0 the complements are all 1, so the regression runs on `+Inf` quantiles
  and the imputed empirical distribution has a non-monotonic probability vector. `ComputeDefaultInitials`
  throws, `SetDefaultParameters` swallows it and leaves `Parameters` **empty**, and the GMM then
  fails outright.
- **Evidence (real C#, via `tools/oracle_emitter`):** building a `DataFrame` over a 17-year record
  containing two low floods, calling `SetLowOutliersFromMGBT()` (which flags 2 and sets the
  threshold to 8900), and fitting `Bulletin17CDistribution` by GMM throws
  `ArgumentException: There must be at least 1 parameter to evaluate. (Parameter 'numberOfParameters')`.
  Inserting a single `dataframe.CalculatePlottingPositions()` after the setter makes the same fit
  succeed and return `[4.149225763920944, 0.18029214454603246, -0.11697673213754865]`.
- **Port handling:** the port already replaced the `INotifyPropertyChanged` plumbing with the
  explicit-call invalidation contract documented in `data_frame.hpp` ("a caller MUST re-run
  `calculate_plotting_positions()` explicitly after any mutation"). `models/model_spec.hpp`'s
  `build_data_frame` is that caller, so it now runs `calculate_plotting_positions()`
  unconditionally: a spec describes a finished frame, so it leaves one fully computed, exactly
  like the C# construction paths that end in `ProcessThresholdSeries(); CalculatePlottingPositions();`
  (for example `BootstrapDataFrame`). Running it at the boundary rather than guarding on one
  branch's precondition states the contract once and covers the threshold-series case as well.
  The emitter mirrors the same call, and `fixtures/estimation/gmm_bulletin17c_censored.json` pins
  the resulting fit against the real library — the C++ reproduces C# to ~1.6e-9 or better on all
  three parameters, confirming the ROS math itself was never in question. This is therefore a
  divergence in **who calls** the recompute, not in what it computes.
- **Suggested C# fix:** call `RecalculatePlottingPositionsAfterEdit()` (already present, line
  ~1157) at the end of both setters, so the headless and GUI paths agree.

## FIDELITY (Task 9) — `TotalFunctionEvaluations` reproduces C# for BFGS/DE/MLSL but not for NelderMead, Brent or Powell

- **Where:** `RMC.BestFit/Estimation/MaximumLikelihood.cs` @ c2e6192 (`TotalFunctionEvaluations`,
  line 164, copied from `Optimizer.FunctionEvaluations` at the end of a successful `Estimate()`),
  against the port's `core/include/corehydro/estimation/maximum_likelihood.hpp` and the six
  optimizers `parse_optimizer` accepts. Surfaced writing `fixtures/estimation/fit_optimizers.json`,
  which pins one fit under each optimizer.
- **What:** fitting a Normal to the shared 10-value `annual_peaks` record, the real C# and the port
  report the same evaluation count for the three optimizers that are genuine `Optimizer` subclasses
  in the port — DifferentialEvolution `1160`, BFGS `164`, MultilevelSingleLinkage `277` — and
  different counts for the other three: NelderMead C# `43` vs port `44`, Brent (on the one-parameter
  bivariate copula model) C# `16` vs port `17`, Powell C# `186` vs port `125`. The optimum itself
  agrees in every case.
- **Two separate causes, both already understood:**
  1. **NelderMead and Brent (+1, deterministic).** In this port `BrentSearch` and `NelderMead` are
     standalone classes that do not derive from `Optimizer` (a documented Phase-0 shortcut), so
     `MaximumLikelihood` reaches them through `estimation/support/optimizer_adapters.hpp`. That
     header's own `total_function_evaluations() FIDELITY` note already states the case: the adapter
     counts the wrapped solver's evaluations **plus one extra re-evaluation at the reported best
     point** to recover the fitness/sign convention, so it "will generally run one-or-more calls
     higher than a faithful C# count ... not asserted to match C#." The measured `+1` on both paths
     is exactly that re-evaluation.
  2. **Powell (~61, chaotic).** Powell IS a real `Optimizer` subclass here and the port is
     line-for-line faithful (same `Evaluate` counter, same `LineMinimization` over a `BrentSearch`
     bracket+minimize, same convergence test). Its optimum agrees with C# to ~1e-11 —
     `mle_optimizers_smoke.json` already pins it at rel 1e-8, with C# `16026.999999749589` vs port
     `16027.000000279293` — and that sub-ULP drift is enough to flip the outer `CheckConvergence`
     test one iteration earlier in the port. One outer Powell iteration on a 2-parameter model is
     two line minimizations, each a Brent bracket plus minimize, which is the observed ~61-evaluation
     gap. Same class as the D6/X12 chaotic-sensitivity findings: the arithmetic agrees, the discrete
     iteration count need not.
- **Port handling:** `fit_optimizers.json` pins `function_evaluations` for DifferentialEvolution,
  BFGS and MultilevelSingleLinkage only, and simply **does not assert** it for NelderMead, Brent and
  Powell; `fit_profile.json` (a NelderMead fit) likewise omits it. Those three cases assert the
  deterministic quantities that do reproduce — `status_is`, `nobs`, `prior_log_likelihood`, and the
  parameter/log-likelihood values. There is **NO `oracle_skip` and NO loosened tolerance**: the
  assertion is not made at all, following the D6/X12 precedent.
- **Suggested action:** none required for correctness. If exact evaluation-count parity is ever
  wanted, cause 1 is a real (small) refactor — fold the `Optimizer` base machinery into
  `BrentSearch`/`NelderMead` so no adapter re-evaluation is needed — while cause 2 is not fixable by
  any port change, since it is the same last-ULP reassociation sensitivity the analysis-curve
  findings above describe.

## BUG (port surface, open) — `fit_mle()` / `fit_map()` on a `model_bivariate()` Archimedean copula returns a wrong answer under the default optimizer

- **Where:** `corehydror/R/fit.R` and `corehydropy/src/corehydropy/fit.py` (the `optimizer`
  default, NelderMead), over
  `core/include/corehydro/models/bivariate_distribution/bivariate_distribution.hpp:581-585`
  (`initial_value_for`) and each Archimedean copula's `parameter_constraints`
  (`gumbel_copula.hpp:117` `{1, 100}`, `clayton_copula.hpp:116` `{-1, 100}`). Found writing
  `site/examples/26-copulas-and-joint-frequency/`.
- **What:** the fit reports `Success` and a parameter that is not the maximum. Measured on that
  page's 48-pair peak/volume record with both marginals fixed at their IFM values:

  ```
  Gumbel   NelderMead              theta=1.000000   logLik=0.000000   status Success
  Clayton  NelderMead              theta=54.450000  logLik=-Inf       status Success
  Gumbel   DifferentialEvolution   theta=2.712260   logLik=29.900498
  Gumbel   Powell                  theta=2.712294   logLik=29.900498
  Clayton  DifferentialEvolution   theta=1.344345   logLik=17.332637
  ```

  Python agrees with R: `fit_mle(..., optimizer="NelderMead")` returns `theta=1.0`,
  `log_likelihood=-2.2e-15`; DifferentialEvolution returns `theta=2.7122604712670535`,
  `log_likelihood=29.900498219016548`. `fit_map()` behaves the same way, returning `theta=1`.
- **Cause: optimizer initialization, not missing sample data.** An earlier report blamed a
  missing `set_sample_data()` call. That is wrong: the call does happen, through
  `set_copula_type()` to `set_copula()` to `set_default_parameters()` to `set_sample_data()`. The
  real cause is where the local search starts. `initial_value_for` sets `ModelParameter.value()`
  to the midpoint of the copula's constraint range, so `model_parameters()` reports `50.5` for
  Gumbel and `49.5` for Clayton. Measured data log-likelihoods at those starts: Gumbel `-1022.56`
  at 50.5 against `29.90` at the optimum; Clayton `-Inf` at 49.5, 50.5 and 54.45 alike. NelderMead
  is a local search, so from a start that far out on a steep ridge (Gumbel) or on a flat `-Inf`
  plateau (Clayton) it collapses onto a bound or stalls one simplex step from where it began, and
  still reports convergence.
- **Scope:** `bivariate_analysis()` is unaffected, because `BayesianAnalysis` initializes
  through a global DE/MAP search rather than from `ModelParameter.value()`. No pinned fixture
  value is affected either: the bivariate estimation fixtures use NelderMead only with the Normal
  and StudentT copulas, whose midpoint starts (correlation 0, degrees of freedom 5) are benign.
- **Where a fix belongs:** NOT in the core. `initial_value_for` is a faithful port of the C#
  `InitialValueFor` and reproduces it exactly. The tractable fixes are at the R and Python
  surface: pick a global optimizer by default for this model family, or seed the start from a
  Kendall's tau inversion, or refuse to report `Success` on a non-finite objective. Each of those
  is a behaviour change that needs its own fixtures.
- **Status: open, deliberate.** Recorded here as a known issue for a follow-up branch, not fixed
  on the branch that found it. Re-verified 2026-08-28 at v0.13.0: `fit_mle()`/`fit_map()` still
  default to `optimizer = "NelderMead"` in both packages and no seeding or non-finite-objective
  guard has landed.

## BUG (port surface, open) — Python `Fit.parameters` silently drops values when parameter names repeat

- **Where:** `corehydropy/src/corehydropy/fit.py:546`
  (`parameters = dict(zip(names, result["parameters"]))`), and the same pattern at `:577`
  (`standard_errors`), `:590-591` (`profile_lower` / `profile_upper`), `:645-646` (`map` /
  `posterior_mean`), and `:512-513` (the credible-interval dicts). (Line numbers re-anchored
  2026-08-28; originally reported at `:468` etc. in v0.5.0.)
- **What:** the fit's parameter vector is exposed as a dict keyed by parameter name, and names are
  not unique. A two-component mixture's names are
  `['Weight (w1)', 'Weight (w2)', 'D1', 'D1', 'D2', 'D2']`, so six fitted values collapse to four
  dict entries and the two component location parameters are lost with no warning. The `repr` and
  the summary text are built from the same dict, so they under-report as well.
- **Scope:** every Python consumer of `Fit.parameters`, `.standard_errors`, `.map`,
  `.posterior_mean`, the profile intervals and the credible intervals, for any model whose
  parameter names repeat. The ordered vector is still reachable from
  `fit.model.spec["parameter_values"]`, which is what
  `site/examples/27-composite-distributions/python.ipynb` reads, with a comment saying why. R's
  `coef()` is unaffected: a named numeric vector tolerates duplicate names.
- **Where a fix belongs:** the Python surface only. Either disambiguate the repeated names when
  building the dicts, or return an ordered structure and keep the name lookup as a secondary
  accessor. Either choice is a public-API change and needs its own fixtures plus an R/Python
  cross-check.
- **Status: open, deliberate.** Recorded here as a known issue for a follow-up branch.
  Re-verified 2026-08-28 at v0.13.0: every `dict(zip(names, ...))` site is still present.

## CONSISTENCY (not a port defect) — MultivariateStudentT's CDF at dimension >= 3 is clock-seeded and not reproducible

- **Where:** `Numerics/Distributions/Multivariate/MultivariateStudentT.cs`, `CDF()`;
  `core/include/corehydro/numerics/distributions/multivariate/multivariate_student_t.hpp:23-38`.
- **What:** for dimension 1-2, `CDF()` is closed-form (dim 1 delegates to the univariate StudentT
  CDF; dim 2 runs the stratified mixture below over `MultivariateNormal::cdf()` calls that are
  themselves closed-form Drezner/Genz bivariate evaluations) and is fully bit-reproducible. For
  dimension >= 2, `CDF()` runs a deterministic K=200 stratified-quantile chi-square(v) mixture,
  but at dimension >= 3 each of the 200 inner `MultivariateNormal.CDF()` calls invokes the
  seeded Genz-Bretz quasi-Monte-Carlo integrator, and the `MultivariateNormal` instance
  `MultivariateStudentT` builds internally is never given an explicit seed (`_MVNUNI = new
  MersenneTwister()` in C#, mirrored by `make_clock_seeded()` in the port). So the MVT CDF at
  dimension >= 3 is not reproducible run to run in either language, independent of any seed the
  caller supplies to the outer `MultivariateStudentT`/`MultivariateDistribution` object.
- **Port handling:** mirrored faithfully; `mvdist_student_t()` in R and Python (`corehydror/R/
  mvdist.R`, `corehydropy/src/corehydropy/mvdist.py`) has no `seed` parameter, unlike
  `mvdist_normal()`, which does expose one because `MultivariateNormal`'s own Genz integrator is
  genuinely seedable. `mvdist_random()` draws from each family's own seeded Mersenne Twister stream
  and is unaffected by this limitation.
- **Suggested action:** none — this is an upstream design property (a class-internal
  `MultivariateNormal` with no seed setter exposed), not something the port introduced or could fix
  without diverging from C#.

## ROBUSTNESS (not a port defect) — the over-identified GMM J-statistic inverts a matrix that is singular by construction, so it does not reproduce C#-vs-C++

- **Where:** `RMC.BestFit/Estimation/GeneralizedMethodOfMoments.cs` @ c2e6192,
  `GetMomentResidualCovariance` (lines 945-975) and `PostProcess` (the `var Vinv = V.Inverse();`
  at line 2511); the port at
  `core/include/corehydro/estimation/generalized_method_of_moments.hpp:1295-1322`.
- **What:** the moment residual covariance is `V = S - D(D'S^-1 D)^-1 D'`. Multiplying it on both
  sides by `S^(-1/2)` gives `S^(-1/2) V S^(-1/2) = I - P`, with `P` a projection of rank `p`, so
  `rank(V) = q - p` EXACTLY. V is singular for any q and p; over-identifying merely moves the rank
  from 0 to 1, it does not make V invertible. `PostProcess` then forms Hansen's J as `g' V^-1 g` by
  calling `V.Inverse()` on that matrix. What the inverse amplifies is the optimizer's
  first-order-condition residual, which is a convergence TOLERANCE of 1e-8, not floating-point
  rounding, so the reported J describes which optimizer ran rather than how well the model fits.
- **Not a transcription difference:** the C# method and the ported one are line-identical, and both
  inverses are LU. The divergence below is not something a closer transcription could remove.
- **Evidence (measured at the fitted parameters of the over-identified fixture case):** V has
  singular values `[1.9e-02, 9.8e-19, 2.3e-19]`, numerical rank 1, and condition number 8.1e16.
  Perturbing S or D by 1e-11 swings J across +892 / -1061 / +668 / -820. The real C# library returns
  `J = 214.59` with a p-value of 0 where the shared core returns `J = -129.46` with a p-value of 1,
  from parameters that agree to 2e-11; the core alone spans -129.46, 1.2e+08, 1268.6 and 3.8e+06
  across its four optimizers, on parameters agreeing to 1e-5.
- **Decisive evidence that the port's inputs are correct:** replacing the inverse with a
  Moore-Penrose pseudo-inverse gives `g' V^+ g = 2.3466`, matching the textbook
  `n * g' S^-1 g = 2.3466` on the same fit. S, D and g are therefore all correct in the port; only
  the inversion of a rank-deficient V is at fault.
- **Relation to the just-identified case:** `q == p` is the degenerate end of the same fact,
  `rank(V) = 0`, and there `j_stat_pval` is correctly NaN because the degrees of freedom are zero.
  That case, and why Bulletin 17C can never leave it, is the DESIGN NOTE entry above
  ("Bulletin17CDistribution GMM is always just-identified, so the J-stat specification test is
  unreachable"); this entry is the over-identified half of it, reachable only through the
  user-written moment conditions of `fit_gmm_moments()`.
- **Port handling:** mirrored faithfully, no divergence. `fixtures/callback/gmm.json`'s
  `over_identified_three_moments` case pins the parameters, standard errors, covariance, degrees of
  freedom and the iteration bookkeeping (13 assertions, every one EMITTER-READ from the real C#
  library) and deliberately leaves `j_stat` and `j_stat_pval` UNASSERTED, with no `oracle_skip` and
  no loosened tolerance, because there is no honest value to loosen towards. Both packages'
  `print()` / `summary()` decline to display the J statistic at zero degrees of freedom, and also
  when the statistic is not finite, which is what the ported `post_process()` reports when
  inverting V raises outright.
- **Suggested C# fix:** compute J through a pseudo-inverse of V, or equivalently as
  `n * g' S^-1 g`, in place of the `V.Inverse()` call in `PostProcess`. Either form is stable
  against the rank deficiency V carries by construction, and both reproduce across compilers and
  optimizers.
- **Exposure note (2026-08-28 re-check):** at the pin the J-stat path is opt-in --
  `PostProcess(bool useSandwich = true, bool computeJstat = false)` returns early unless
  `computeJstat` is passed true -- so a default GMM fit never reaches the singular inversion; the
  finding bites only callers who request the specification test.
- **Status: open upstream, reported.** Found in v0.7.0 by the first over-identified GMM fit either
  package could run, since `fit_gmm()` reaches only Bulletin 17C. Filed with RMC as
  https://github.com/USACE-RMC/RMC-BestFit/issues/18; still open with no maintainer response as of
  2026-08-28.

## ROBUSTNESS — `Statistics.Percentile`'s `k` range check cannot see a NaN, and the C++ port's float-to-int conversion is undefined where C#'s is not

- **Where:** `Numerics/Data/Statistics/Statistics.cs`, `Percentile(IList<double>, double k, bool)`
  (~line 544); ported at `core/include/corehydro/numerics/data/statistics.hpp`.
- **What:** the guard is `if (k < 0.0 || k > 1.0) throw ...`. Every comparison against NaN is
  false, so a NaN `k` passes it and reaches `int lower = (int)Math.Floor(h)` with `h = NaN`.
- **Why it is harmless in C# and not in C++:** .NET Core 3.0 onward DEFINES the float-to-integer
  conversion to saturate, so `(int)double.NaN` is 0 and C# returns
  `sortedData[0] + NaN * (sortedData[0] - sortedData[0])`, i.e. NaN — an odd answer for an
  out-of-range request, but a safe one. C++ leaves that conversion UNDEFINED. AArch64's `fcvtzs`
  saturates the same way C# does, so the port returned NaN and looked correct; x86-64's
  `cvttsd2si` yields `INT_MIN`, and indexing the sample with it is a wild read.
- **Evidence:** an x86_64 build of the pre-fix body prints `h=nan lower=-2147483648
  upper=-2147483648` and dies with SIGSEGV; the same source built for arm64 prints
  `lower=0 upper=0` and returns `nan`. UBSan on macOS reports
  `statistics.hpp:204:34: runtime error: nan is outside the range of representable values of
  type 'int'` regardless of ISA.
- **Reachable, not hypothetical:** `Bootstrap<TData>.ComputeAccelerationConstants` divides
  `i3 / (i2^1.5 * 6)`, which is `0 / 0` whenever every jackknife sample fails. That NaN becomes
  the acceleration constant, passes through `Normal.StandardCDF` unchanged, and arrives at
  `Percentile` as `k`. It is what a BCa run does after a jackknife delegate throws, and it
  segfaulted the R session, the Python interpreter and the C++ ctest binary on every gcc
  platform. Fixed in the port; see the comment at the call site.
- **Port handling:** the port now returns NaN explicitly for a NaN `k`. This REPRODUCES the C#
  result exactly (it is what .NET's defined saturation computes) while removing the undefined
  behaviour, so it is a fidelity fix rather than a behaviour change.
- **Suggested C# fix:** reject a non-finite `k` in the range check —
  `if (double.IsNaN(k) || k < 0.0 || k > 1.0) throw new ArgumentOutOfRangeException(...)`. The
  deeper fix belongs in `ComputeAccelerationConstants`, which should report that it had no usable
  jackknife samples rather than hand a NaN acceleration constant downstream.

## BUG — `Statistics.LinearMoments` overflows `int` in its weight numerators at 1293 points and silently returns a wrong L-kurtosis

- **Where:** `Numerics/Data/Statistics/Statistics.cs` lines 509-520 (`LinearMoments`); ported at
  `core/include/corehydro/numerics/data/statistics.hpp`.
- **What:** the probability-weighted-moment accumulators form their numerators in `int`:
  `B2 += (i - 2) * (i - 1) / ((N - 2) * (N - 1)) * sortedData[i - 1]` and
  `B3 += (i - 3) * (i - 2) * (i - 1) / (...)`. `N` is a `double`, but `i` is an `int`, so the
  products are integer arithmetic. The triple product first exceeds `int.MaxValue` at
  `i = 1293` (`1290 * 1291 * 1292 = 2,151,683,880` against a ceiling of `2,147,483,647`), and
  C#'s default unchecked context wraps it to a negative number rather than throwing. The pair
  product `(i-2)*(i-1)` wraps too, at `i = 46,343` (`46,341 * 46,342 = 2,147,534,622`; the last
  safe index is `i = 46,342`, where `46,340 * 46,341 = 2,147,441,940`). Any sample of 1293 or more values returns a
  corrupt τ4.
- **Evidence (real C#, driven at the pinned `2a0357a`):** for the evenly spaced sample
  `x[i] = 1 + 0.5i`, whose L-skewness and L-kurtosis are both 0 at every length,
  `Statistics.LinearMoments` returns

  | n | τ3 | τ4 |
  |---|---|---|
  | 1292 | `-3.1086244689504383E-15` | `-1.7763568394002505E-15` |
  | 1293 | `-1.7763568394002505E-15` | `-0.18525251648817065` |
  | 1300 | `-1.3766765505351941E-14` | `-1.446418581370934` |

  λ1 and λ2 are unaffected (they use no product), and τ3 is unaffected (the pair product does
  not wrap until 46,343), so the error is confined to τ4 and grows with n. There is no warning
  and no exception; a Kappa-4 or a GEV fit off L-moments would take the corrupt value as a real
  shape statistic.
- **Why it is defined in C# and not in C++:** C# specifies unchecked integer arithmetic as
  two's-complement wrapping. C++ leaves signed overflow UNDEFINED, so the ported expression is
  a UBSan finding: `statistics.hpp:127:37: runtime error: signed integer overflow: 1665390 *
  1292 cannot be represented in type 'int'`. That is the class of finding CRAN's sanitizer run
  rejects a package for, and it cannot be left in place on the grounds that the compiler
  happens to wrap the same way today.
- **Status: DELIBERATE DIVERGENCE.** The port now forms the numerators in `double`
  (`(di - 3) * (di - 2) * (di - 1)`, `di = (double)i`). Below the overflow the products are
  exact integers far under 2^53, so the result is bit-identical to the C# for every sample the
  library has ever been pinned against — no shipped fixture carries a sample longer than a few
  hundred points, and no oracle value moved. Above the overflow the port returns the
  mathematically correct weight where C# returns a wrapped one. Guarded by
  `core/tests/test_linear_moments_overflow.cpp`. Filed with RMC as
  https://github.com/USACE-RMC/Numerics/issues/146 (still open with no maintainer response as of
  2026-08-28), so the divergence retires if upstream adopts the fix below.
- **Suggested C# fix:** make the numerators `double` — `((double)i - 3) * ((double)i - 2) *
  ((double)i - 1)` — or hoist `double di = i` at the top of the loop, matching what `N` already
  is. `long` would push the failure out to about 2.1 million points rather than removing it.

## FIDELITY (not a port defect) — a seeded ParticleSwarm or ShuffledComplexEvolution run takes a different search path under a compiler that emits fused multiply-add

- **Where:** `Numerics/Mathematics/Optimization/Global/ParticleSwarm.cs` and
  `ShuffledComplexEvolution.cs` @ 2a0357a, against
  `core/include/corehydro/numerics/math/optimization/particle_swarm.hpp` and
  `shuffled_complex_evolution.hpp`. Surfaced writing the seeded cross-language digest cases in
  `fixtures/toolbox/toolbox_cross_language.json` for the optimizer phase's user-facing surface
  (`optim_minimize(method = "particle_swarm" | "sce")`).
- **What (measured, same machine, same seed 12345, same objective):** with the ported core
  compiled `-ffp-contract=off`, the C++ runner is bit-identical to the real C# library on every
  construct tried — ParticleSwarm on Booth `3073` iterations / `92220` evaluations / fitness
  `3.1554436208840472E-30`, on Eggholder `2413` / `72420` / `-959.64066272085097` at
  `(512, 404.23180505405406)`; ShuffledComplexEvolution on Booth `54` / `4786` /
  `1.7264378682942888E-25`, on 5-D DeJong `57` / `9483` / `1.6274114682612478E-20`. With
  contraction left at the compiler default (which is what the shipped R and Python packages get,
  and what `test_fixtures` compiles the core with), the same runs give ParticleSwarm/Booth `3855`
  / `115680` / `0`, ParticleSwarm/Eggholder the same `2413` / `72420` / `-959.64066272085097` but
  `y = 404.23180501084073`, SCE/Booth `51` / `4232` / `9.2296725910858381e-29`, and SCE/DeJong the
  same `57` / `9483` but `1.6274155980703362e-20`. Every one of those still lands on the textbook
  optimum well inside the upstream MSTest deltas, so `fixtures/toolbox/optimizers.json` reproduces
  in all four runners.
- **Cause:** both algorithms branch on comparisons between accumulated sums (`if (fitness <
  particle.BestFitness)`, the SCE sub-complex sort and its reflection/contraction acceptance
  tests). clang and gcc contract `a*b + c` into a single fused multiply-add by default; .NET never
  does. One contracted expression is enough to flip an accept/reject, and from there every later
  PRNG draw is spent on a different point. This is the same arithmetic-contraction class already
  documented for `test_fixtures`'s callback catalog (see `core/CMakeLists.txt` and
  `fixtures/callback/callback_cross_language.json`), not an algorithmic divergence: the ported code
  reproduces C# exactly the moment contraction is off. `SimulatedAnnealing` and `MultiStart`, the
  other two global optimizers exposed in the same phase, are NOT affected — both reproduce C#
  bit-for-bit either way (measured on Booth and FXYZ respectively, down to the evaluation count).
- **Port handling:** `optimizers.json` pins the ACCURACY of all four methods at the upstream MSTest
  literals and tolerances, which reproduce in C++, R, Python and C# alike. The cross-language digest
  cases pin, at zero tolerance, only the quantities measured to survive both paths: everything for
  SimulatedAnnealing and MultiStart; the iteration count, evaluation count, converged value and the
  on-bound parameter for ParticleSwarm/Eggholder; the iteration and evaluation counts for
  SCE/DeJong. The unpinned quantities are simply **not asserted** — there is **NO `oracle_skip` and
  NO loosened tolerance** — following the D6/X12 precedent. The fixture's own `reference` string
  carries the same measurement.
- **Suggested action:** none for correctness; a seeded run is exactly reproducible within one build,
  and both packages remain bit-identical to each other. Making a seeded ParticleSwarm or SCE run
  reproduce the C# stream on every platform would mean compiling those two headers without
  contraction in the shipped packages, which neither an R `Makevars` nor a portable pragma can
  promise across the three CI compilers, so it is deliberately not attempted.

## BUG — `Network` cannot be constructed at all: the constructor sizes both edge caches one element short, and never sets `_nodeCount`

- **Where:** `Numerics/Mathematics/Optimization/Dynamic/Network.cs` @ 2a0357a, the constructor
  (lines 41-69), against `core/include/corehydro/numerics/math/optimization/dynamic/network.hpp`.
  Surfaced porting the dynamic-programming trio (BinaryHeap / Dijkstra / Network) in the optimizer
  phase.
- **What:** the constructor computes `max = Math.Max(edges.Max(x => x.FromIndex), edges.Max(x =>
  x.ToIndex))` and then allocates `_incomingEdges = new List<Edge>[max]` and `_outgoingEdges = new
  List<Edge>[max]`. `max` is by construction an index that some edge actually uses, so the build
  loop immediately below indexes at `max` and runs off the end of one array or the other. There is
  no graph that escapes it, including an empty one (LINQ `Max` throws on an empty sequence
  instead). Separately, `_nodeCount` is assigned `0` on the constructor's fourth line and never
  raised — the two lines that would have raised it, and the `//_nodeCount += 1;` that would have
  turned it into a count, are inside the commented-out `RoadSegment` block just above — so every
  `Solve` overload forwards a node count of `0` to `Dijkstra.Solve`, which is not the `-1` sentinel
  and would rebuild a zero-length cache and throw in its turn.
- **Evidence (measured against the real library, not reasoned):** a throwaway console app
  referencing `upstream/Numerics/Numerics/Numerics.csproj` at 2a0357a (`/tmp/getpath_probe`,
  `dotnet run -c Release`):

  ```
  === ctor(one edge 0->0)
    THREW System.IndexOutOfRangeException: Index was outside the bounds of the array.
    STACK at Numerics.Mathematics.Optimization.Network..ctor(Edge[] edges, Int32[]
           destinationIndices) in .../Mathematics/Optimization/Dynamic/Network.cs:line 41
  === ctor(one edge 0->1)
    THREW System.IndexOutOfRangeException: Index was outside the bounds of the array.
  === ctor(no edges)
    THREW System.InvalidOperationException: Sequence contains no elements
  === Dijkstra.Solve(edges, 1, nodeCount: 0)
    THREW System.IndexOutOfRangeException ... in .../Dynamic/Dijkstra.cs:line 123
  ```

  Padding the graph with an isolated highest node does not help, because the padding node becomes
  the new `max`. Consistent with the class being unreachable, `Test_Numerics` has no `Network` test
  class at all: `DijkstraTesting.cs` declares `ShortestPathTesting` and all eight of its methods
  call `Dijkstra.Solve` directly.
- **Port handling (INTENTIONAL C++ DIVERGENCE, the only one in this file):** the port allocates
  `max + 1` and sets `node_count_ = max + 1`. A class whose every construction throws has no
  behavior to be faithful to, and the commented-out block upstream states the intended sizing
  outright (`if (_edges[i].ToIndex > _nodeCount) { _nodeCount = _edges[i].ToIndex; }` … `//// Add
  one to the count for the index offset. //_nodeCount += 1;`). Nothing else in the class is
  changed. The divergence is independently checkable rather than assumed: with it,
  `network.solve(d)` returns exactly what `dijkstra::solve(edges, d)` returns, and that free
  function DOES run in C#. The ctest oracles in `core/tests/test_network_optimization.cpp` were
  measured off a copy of `Network.cs` carrying this one patch and nothing else, compiled against
  the real Numerics assembly (`/tmp/getpath_probe/PatchedNetwork.cs`); the patched
  `Solve(9)` on the `SimpleNetworkRouting` grid printed `identical to free Dijkstra.Solve: True`
  element for element. The divergence is recorded in `network.hpp`'s transcription note 1.
- **Suggested C# fix:** `_nodeCount = max + 1;` and `new List<Edge>[_nodeCount]` for both caches.
  No oracle value moves, because no C# caller can be relying on the current behavior.

## BUG — `Network.Solve(float[] edgeWeights)` silently ignores the custom weights it was given

- **Where:** `Numerics/Mathematics/Optimization/Dynamic/Network.cs` @ 2a0357a,
  `Solve(float[] edgeWeights)`.
- **What:** the overload builds a re-weighted copy of the edge array and then calls
  `Dijkstra.Solve(edges, _destinationIndices, _nodeCount, _incomingEdges)` — passing the STALE
  cached incoming-edge lists, which still hold the ORIGINAL weights. `Dijkstra.Solve` uses a
  supplied cache whenever `edgesFromNodes.Length == nNodes`, and it reads the weights only out of
  that cache, so the re-weighted array is never looked at. The method is a no-op with respect to
  its own argument.
- **Evidence (measured on the patched Network above, so the first defect does not mask this one):**
  feeding all-ones weights to the `SimpleNetworkRouting` grid (whose real weights run 1 to 30)
  returns the ORIGINAL cost column `8, 5, 6, 4, 5, 7, 4, 3, 2, 0`, identical to `Solve(new[]{9})`,
  where an honest unit-weight solve of the same graph gives `4, 3, 3, 2, 1, 4, 3, 2, 1, 0`.
- **Port handling:** mirrored faithfully. The port's `dijkstra::solve` applies the same
  cache-length test, so the behavior falls out of a straight transcription; the header (note 5)
  says so in capitals and `test_network_optimization.cpp` pins both tables side by side so a
  future "cleanup" fails loudly. The overload is NOT exposed as a working custom-weight solve in
  the R/Python toolbox surface: it is reachable only as the `network` group's
  `network_solve_weights` method, documented there as quirk-preserving, and the user-facing
  `shortest_path()` verb calls the free solver instead. The fixture case
  `triangle_path_network_solve_weights_are_ignored` in `fixtures/toolbox/network.json` pins the
  no-op through all four runners on a graph where honoring the weights would route node 1 through
  node 0 rather than node 2.
- **Suggested C# fix:** pass `null` for the cache (letting the solver rebuild it from the
  re-weighted edges), or rebuild the cache from `edges` inside the overload.

## BUG — `Network.GetPath` cannot return a path: it binary-searches an `int[]` for an `Edge`

- **Where:** `Numerics/Mathematics/Optimization/Dynamic/Network.cs` @ 2a0357a, both `GetPath`
  overloads. This is the defect the optimizer phase's plan flagged for investigation; the
  investigation confirmed it and found it is not the only problem with the method.
- **What:** both overloads filter candidate edges with `Array.BinarySearch(edgesToRemove, edge)`,
  where `edgesToRemove` is `int[]` and `edge` is an `Edge` struct. There is no
  `BinarySearch<T>(T[], T)` match, so it binds `Array.BinarySearch(Array array, object value)`,
  which boxes the `Edge` and asks an `Int32` to compare itself against it.
- **Evidence (measured):**

  ```
  === Array.BinarySearch(int[]{0,1,2}, (object)Edge)
    THREW System.InvalidOperationException: Failed to compare two elements in the array.
    INNER System.ArgumentException: Object must be of type Int32.
  === GetPath(new[]{0}, 0)         THREW System.InvalidOperationException: Failed to compare ...
  === GetPath(new[]{0}, 0, table)  THREW System.InvalidOperationException: Failed to compare ...
  === GetPath(empty, 0)            returned null
  === GetPath(empty, 9)            returned null
  === GetPath(empty, 0, table)     returned []
  === Array.BinarySearch(int[0], (object)Edge)  returned -1
  ```

  So any call carrying a non-empty removal list — which is every call the method exists to serve —
  dies on the first edge it examines. A zero-length binary search never invokes the comparer, so an
  empty removal list is the one input that gets through, and then the outer `do { … } while
  (heap.Count == 0)` (which loops only while the heap is EMPTY, i.e. runs one pass and exits, or
  re-enters and throws `"Heap is empty."`) leaves `foundPath` false. The complete set of observable
  outcomes for both overloads is therefore: **throw, null, or an empty list. Never a path.** Two
  further defects sit behind that one and are unreachable because of it: the `while (heap.Count ==
  0)` loop condition itself, and the second overload's path-reconstruction walk, which steps
  `tempNode = (int)existingResultsTable[tempNode, 2]` — the COST column — where its sibling
  overload correctly steps `[…, 0]`, the NEXT_NODE column.
- **Port handling:** transcribed structurally so upstream diffs still map, with the offending
  expression ported as `detail::binary_search_edge`, which reproduces the measured behavior (`-1`
  for an empty list, the .NET exception message otherwise) rather than pretending the line does
  something it does not. All three observable outcomes are pinned in
  `core/tests/test_network_optimization.cpp` against the patched-C# measurements. The method is
  **SEVERED from the R/Python surface** (see `upstream/CLAUDE.md`) — the toolbox `network` group
  exposes the `Solve` overloads only.
- **Suggested C# fix:** compare the edge INDEX, `Array.BinarySearch(edgesToRemove, edge.Index)`,
  which is what the sibling call sites in the same method already do. Then fix the two follow-on
  defects above, and give the method a test — it has none.

## ROBUSTNESS — `Dijkstra.Solve` bounds-checks its node indices only by accident, so a too-small `nodeCount` is an IndexOutOfRangeException (and was undefined behavior in the port)

- **Where:** `Numerics/Mathematics/Optimization/Dynamic/Dijkstra.cs` @ 2a0357a, both `Solve`
  overloads. Port side: `core/include/corehydro/numerics/math/optimization/dynamic/dijkstra.hpp`.
- **What:** every array the solver allocates is sized `nNodes`, and every one of them is then
  indexed by a node index read straight off an `Edge` — `edgesToNodes[edge.ToIndex]`,
  `resultTable[destinationIndex, …]`, `nodeWeightToDestination[from]` — with nothing checking that
  the index fits. In C# the CLR checks it, so a `nodeCount` smaller than the graph is an
  `IndexOutOfRangeException`. That is a defensible outcome and the port must reproduce it, but the
  C# check is LAZY, not a validation pass: an out-of-range index on an edge the search never
  relaxes never throws at all.
- **Evidence (measured against the real Numerics library at 2a0357a; probe `/tmp/getpath_probe`):**

  ```
  Solve({(0,1),(1,5)}, 0, nodeCount: 2)   THREW System.IndexOutOfRangeException: Index was outside the bounds of the array.
  Solve({(0,1),(1,5)}, new[]{0}, 2)       THREW System.IndexOutOfRangeException
  Solve({(5,1),(0,1)}, 1, nodeCount: 2)   THREW System.IndexOutOfRangeException
  Solve({(0,1)},       5, nodeCount: 2)   THREW System.IndexOutOfRangeException
  Solve({(0,1),(7,0)}, 1, nodeCount: 2)   THREW System.IndexOutOfRangeException
  Solve({(0,1),(7,1)}, 0, nodeCount: 2)   RETURNED [(0,-1,0), (-1,-1,Inf)]
  ```

- **Port defect this exposed (FIXED here):** the port transcribed those indexing expressions as
  `std::vector::operator[]`, which is undefined behavior rather than a checked throw, so the same
  calls wrote past the end of the cache. Reproduced three ways before the fix: AddressSanitizer
  reported `heap-buffer-overflow … in dijkstra::detail::build_edges_to_nodes` at
  `dijkstra.hpp:122`; `shortest_path(from = c(0,1), to = c(5,2), weight = c(1,1), destinations = 0,
  node_count = 2)` returned a quietly corrupt routing table in R; and the same call fatally
  terminated the Python interpreter. No fixture, ctest or package test supplied a `node_count`
  below `max(from, to) + 1`, which is why every suite was green.
- **Port handling:** `dijkstra::detail::checked_node_index` now guards each of the sites C# indexes
  by a node index and throws `std::out_of_range` carrying the C# message. The guard is deliberately
  in place at each site rather than hoisted into one up-front sweep, so the lazy case above still
  returns its table (measured both ways; an up-front variant aborts that assertion). See
  `dijkstra.hpp` note 9 and `a_node_count_below_the_graph_throws_like_csharp` in
  `core/tests/test_network_optimization.cpp`, which pins all six measurements. The R and Python
  `shortest_path()` wrappers are STRICTER than the solver on purpose: they reject any `node_count`
  below `max(from, to) + 1` up front, so the user gets a message naming the argument instead of a
  bounds message from inside the solver, at the cost of also rejecting the lazy case — a graph a
  caller cannot have meant.
- **Suggested C# fix:** validate `nodeCount` against `edges.Max(o => Math.Max(o.FromIndex,
  o.ToIndex)) + 1` at the top of both overloads and throw `ArgumentOutOfRangeException(nameof(
  nodeCount), …)`, so the caller learns which argument is wrong.

## BUG — `AugmentedLagrange` cannot maximize: `Optimize()` always drives the inner optimizer through `Minimize()`

- **Where:** `Numerics/Mathematics/Optimization/Constrained/AugmentedLagrange.cs` @ 2a0357a,
  `Optimize()` (both `this.Optimizer.Minimize()` call sites) and `augmentedLagrangianFunction`.
- **What:** the constructor replaces the inner optimizer's objective with
  `augmentedLagrangianFunction`, which opens `double phi = _primaryObjectiveFunction(x);` — the RAW
  objective, called directly, never through the base's `Evaluate` and so never through
  `FunctionScale`. `Optimize()` then calls `this.Optimizer.Minimize()` unconditionally. Under
  `Maximize()` the outer object's own bookkeeping flips sign, but the search does not: the inner
  optimizer still MINIMIZES the objective plus penalty. The run reports `Success` and returns the
  constrained MINIMUM.
- **Evidence (measured through the shipped packages before the guard):** maximizing
  `f(x) = -(x - 3)^2` subject to `x <= 1` over `[-10, 10]` — true optimum `x = 1`, value `-4` —
  returned `x = -10.00011`, value `-169.0029`, status `Success`, byte for byte the same answer as
  the matching `optim_minimize` call. Two-parameter constructs behave the same way: maximizing
  `-((x-1)^2 + (y-3)^2)` on `[0,10]^2` under the inactive constraint `x + y <= 20` (true maximum 0
  at `(1,3)`) returns `(10.000110, 9.999878)`, value `-130.0003`, status `Success`, and maximizing
  `-((x-5)^2 + (y-7)^2)` under `x + y == 4` (true maximum `-8` at `(2,2)`) returns
  `(4.00001, -0.00001)`, value `-50.0001`, status `Success`.
- **Port handling:** the ported class mirrors it exactly (see
  `core/include/corehydro/numerics/math/optimization/augmented_lagrange.hpp` note 2 and the
  `optimizer_runner.hpp` grammar block), because a fixture case must be able to pin upstream
  behavior. The guard lives on the two PUBLIC verbs instead:
  `optim_maximize(method = "augmented_lagrange")` is rejected by name in both packages
  (`kOptimMinimizeOnlyMethods` in `corehydror/R/optim.R`, `_MINIMIZE_ONLY_METHODS` in
  `corehydropy/src/corehydropy/optim.py`), and the error names the upstream reason and the
  workaround. The workaround is exact rather than approximate: minimizing `-f` under the same
  constraints IS maximizing `f`, and both packages' tests run it and check the answer.
- **Suggested C# fix:** have `augmentedLagrangianFunction` obtain `phi` through the base's
  `Evaluate` (so `FunctionScale` applies), or call `this.Optimizer.Maximize()` when the outer run
  is a maximization. Either way the class needs a maximizing test; all six existing ones minimize.

## BUG — `MultiStart`'s polish step clamps the recorded best point after its fitness was recorded, so the reported value need not be attained at the reported parameters

- **Where:** `Numerics/Mathematics/Optimization/Global/MultiStart.cs` @ 2a0357a, the polish block at
  the end of `Optimize()`, which passes `BestParameterSet.Values` into `GetLocalOptimizer`.
- **What:** `GetLocalOptimizer` calls `RepairParameter` on the array it is handed, in place. On the
  polish call that array IS `BestParameterSet.Values`, so a best point that a local search left
  outside the box is clamped back onto the bound while `BestParameterSet.Fitness` keeps the
  out-of-box value that was recorded for the unclamped point. The run then reports a value the
  reported parameters do not produce. The same aliasing shape as note 2 of the ported header (the
  re-seated `InitialValues` array), but with a numeric rather than a bookkeeping consequence.
- **Evidence (measured through the shipped Python package, and identical in R):** minimizing the
  Eggholder function over `[-512, 512]^2` from `(0, 0)` with `method = "multi_start"` reports
  `value = -959.829329467467` at `(512, 404.32280392733844)`. The objective AT that point is
  `-959.6312431930309`. A 2001 x 2001 grid scan puts the whole-box minimum at about `-959.57`
  (the true box optimum is `-959.6407`), so the reported value is not attainable anywhere in the
  box; just outside it, at `x = 512.5`, the same objective reads `-961.148161258961`. Task 3 of the
  P3 phase measured the real C# `MultiStart` returning `iters=100 evals=8414286
  fitness=-959.82932946746701 values=512, 404.32280392733844` — bit-for-bit what the port returns,
  so this is upstream behavior faithfully reproduced, not a port defect.
- **Port handling:** mirrored exactly (see
  `core/include/corehydro/numerics/math/optimization/multi_start.hpp` note 3), because the search
  path and the C# oracles depend on it. Invisible on the fixture-pinned FXYZ construct, whose value
  and parameters are consistent, which is why no fixture case catches it. It is noted in the
  0.10.0 release notes so a user who sees an inconsistent pair knows it is upstream, and worked
  example 19 deliberately does not showcase this method.
- **Suggested C# fix:** polish a COPY of `BestParameterSet.Values` and adopt the result only if its
  re-evaluated fitness is an improvement, or re-evaluate the objective after the repair so the
  reported fitness always belongs to the reported point.

## Findings from the P4 "data and tests" phase (August 2026): HypothesisTests, Correlation matrix overloads, and the Paired Data subsystem

The P4 phase ported the twelve Numerics `HypothesisTests` statics, the `Correlation` matrix
overloads, the two `DataFrame` hypothesis-test/summary-statistics facades left deferred at Phase 5,
and the entire (previously unported) Paired Data subsystem -- `Ordinate`, `OrderedPairedData`,
`UncertainOrdinate`, `UncertainOrderedPairedData`, `LineSimplification`, and `TabularFunction`. All
of the findings below surfaced while transcribing that subsystem method-for-method against the real
C# source; each is reproduced on purpose in the corresponding C++ header rather than "fixed," with
a numbered transcription note at the call site citing the C# line numbers below.

## BUG — `OrderedPairedData.LangSimplify` never force-keeps the last point of a curve, silently dropping it

- **Where:** `Numerics/Data/Paired Data/OrderedPairedData.cs` @ 2a0357a, `LangSimplify` (line 1445)
  and its private recursive helper `RecursiveTolerance` (~lines 1482-1518).
- **What:** `DouglasPeuckerSimplify` and `VisvaligamWhyattSimplify`, the other two curve-
  simplification algorithms in this class, both explicitly force-keep the first AND last ordinate of
  the input curve (Douglas-Peucker seeds its kept-index set with `{firstPoint, lastPoint}`;
  Visvaligam-Whyatt runs its removal loop only over the interior, `j` from 2 to `Count-2`, leaving
  both ends untouched). `LangSimplify` has no equivalent guarantee. CORRECTED MECHANISM (P4
  whole-branch-review finding M7b -- the loss itself is real and confirmed below; this paragraph
  used to misdescribe where it happens, claiming the loop's own guard fires and hands
  `RecursiveTolerance` a `lookAhead` of 0, which is backwards): the loop's own look-ahead clamp
  (line ~1460, `if (i + lookAhead > count) lookAhead = count - i - 1;`) uses a STRICT `>`, so at the
  exact-equality tail boundary (`i + lookAhead == count`) it does NOT fire and `lookAhead` stays at
  its full, un-clamped value; on the curve below that happens at `i = 3` with `lookAhead` still 2.
  `RecursiveTolerance`'s own inner guard (`if (i + n < count)`, also strict `<`) then ALSO fails at
  that SAME exact equality (`i + n == count`) and is skipped entirely, so the call falls through to
  `return n;` UNCHANGED rather than ever testing whether the angle condition should reduce it. Back
  in the caller, this unreduced offset (2) points one past the last valid ordinate
  (`i + offset == count`), so the caller's own `(i + offset) < count` check -- correctly, given that
  oversized offset -- rejects the append, and the loop walks `i` to `count` and exits without ever
  revisiting the final point.
- **Evidence (reproduced against the real C# library, not merely inferred):** on the five-point sin
  curve at `tolerance=0.01, lookAhead=2` -- the exact case upstream's own `Test_LangSimplify`
  exercises -- `LangSimplify` returns 3 points, `{(0,0), (1.57,1), (4.71,-1)}`, dropping `(6.28,0)`
  entirely. Confirmed with `dotnet run` against `upstream/Numerics @ 2a0357a`. Also confirmed: changing
  line ~1460's clamp test from `>` to `>=` -- so `lookAhead` clamps down to 1 at `i = 3`,
  `RecursiveTolerance`'s guard then fires (`4 < 5`), and the resulting smaller offset of 1 correctly
  targets the real last ordinate -- reproduces the expected 4-point result against the real library;
  see "Suggested C# fix" below.
- **Why upstream's own test never catches this:** `Test_LangSimplify`'s assertion loop is bounded by
  `test.Count` (the actual, possibly-short RESULT length), not the expected point count: with
  `test.Count == 3` the loop only ever compares indices 0-2, and the missing fourth point is silently
  never checked. This is a specific instance of a broader pattern worth naming on its own: several
  tests across the simplification suite bound their comparison loop by the returned collection's own
  length rather than by the caller's independently-known expected length, so an implementation that
  returns too few elements still passes. This port's own test suite uses length-first assertions
  throughout (asserting the retained-point COUNT before comparing contents) specifically as a
  documented corehydro supplement to close that gap.
- **Port handling:** mirrored faithfully -- `lang_simplify()` in
  `core/include/corehydro/numerics/data/paired_data/ordered_paired_data.hpp` drops the final ordinate
  on the same inputs. Pinned two ways: `core/tests/test_ordered_paired_data.cpp`'s
  `test_lang_simplify` asserts the verified 3-point result (not a naively-expected 4-point one), and
  `fixtures/toolbox/paired_data.json`'s `sin_curve_simplify_lang` case is reproduced against the real
  C# library by the dotnet oracle gate at exact tolerance.
- **Suggested C# fix:** two independent options, either sufficient on its own. (1) A targeted fix at
  the actual mechanism identified above: change the loop's look-ahead clamp at line ~1460 from
  `if (i + lookAhead > count)` to `>=`, which forces a smaller, in-range offset at the tail instead
  of an unreduced one that overshoots `Count` -- measured to reproduce the expected 4-point result
  against the real library. (2) A defensive, mechanism-agnostic fix: after the main loop, if the
  last appended ordinate's index is not `Count - 1`, append the final ordinate explicitly -- the
  same force-keep both sibling algorithms already perform. Also fix `Test_LangSimplify` to bound its
  comparison loop by the expected point count, not the actual result's `Count`, so a future
  regression here would be caught.

## BUG — `OrderedPairedData.RemoveRange`'s off-by-one guard makes a trailing removal a silent no-op

- **Where:** `Numerics/Data/Paired Data/OrderedPairedData.cs` @ 2a0357a,
  `RemoveRange(int index, int count)` (line 453).
- **What:** the bounds guard is `if (index < 0 || (index + count) >= Count) return;` -- using `>=`
  where the off-by-one-free test `>` is correct. Removing a trailing run of elements that reaches the
  very last element (`index + count == Count`) therefore hits the guard and returns without removing
  anything, silently. A second, harmless defect sits in the same method: the loop that builds the
  `items` list handed to the `CollectionChanged` event runs `for (i = index; i < count; i++)` rather
  than `i < index + count` -- wrong, but inert, since that `items` list has no consumer once the
  event itself fires.
- **Evidence:** direct inspection of the guard; `Test_Indexing`'s own `RemoveRange(0, 3)` call on a
  13-element collection is well clear of the boundary (`0 + 3 = 3 < 13`), so upstream's own test
  never exercises the off-by-one.
- **Port handling:** mirrored faithfully -- `remove_range(index, cnt)` in `ordered_paired_data.hpp`
  keeps the `>=` guard (a trailing-run removal silently does nothing), documented at the call site.
  The `CollectionChanged`-only loop-bound defect has no C++ counterpart to reproduce (the event
  itself is severed project-wide; see the file's own header). Contrast the Uncertain twin
  (`UncertainOrderedPairedData::remove_range`), whose C# source has NEITHER bug -- it relies on
  `List<T>.RemoveRange`'s own correct `ArgumentOutOfRangeException` behavior -- so it is ported with
  the CORRECT `index + count > Count` bound, throwing rather than silently no-opping.
- **Suggested C# fix:** change the guard to `index + count > Count`, matching `List<T>.RemoveRange`'s
  own contract (and the sibling `UncertainOrderedPairedData.RemoveRange`, which never had this bug).

## BUG — `OrderedPairedData.SequentialSearchY` reads the X search-start field instead of Y's

- **Where:** `Numerics/Data/Paired Data/OrderedPairedData.cs` @ 2a0357a,
  `SequentialSearchY(double y)` (~line 1114), its third branch.
- **What:** the method's third branch, which resets the search start to `0` when `y` has moved
  outside the cached search window, reads `_ordinates[XSearchStart].Y` where
  `_ordinates[YSearchStart].Y` is the field the X/Y-mirrored logic (and `SequentialSearchX`'s own
  analogous branch) calls for.
- **Evidence:** direct inspection; because both `XSearchStart` and `YSearchStart` start at `0` and
  this port's transcribed ctest never diverges them before calling `sequential_search_y`, the results
  agree with what the correct code would produce -- exactly why upstream's own `Test_Sequential`
  passes despite the bug.
- **Port handling:** mirrored faithfully -- `sequential_search_y()` in `ordered_paired_data.hpp`
  reads `x_search_start_` at the equivalent branch, with an inline comment marking it "Bug
  transcribed verbatim."
- **Suggested C# fix:** change `_ordinates[XSearchStart].Y` to `_ordinates[YSearchStart].Y` in
  `SequentialSearchY`'s third branch.

## BUG — `OrderedPairedData`'s `XdeltaStart`/`YdeltaStart` are never assigned, so `UseSmartSearch`'s Hunt branch almost never fires

- **Where:** `Numerics/Data/Paired Data/OrderedPairedData.cs` @ 2a0357a, `XdeltaStart` (line 50) /
  `YdeltaStart` (line 55), read in `SearchX` (line 934) / `SearchY` (line 955).
- **What:** `XdeltaStart`/`YdeltaStart` are declared, initialized to `0`, and never assigned anywhere
  else in the class. `SearchX`/`SearchY` set `Xcorrelated`/`Ycorrelated` to
  `Math.Abs(start - XSearchStart) > XdeltaStart` (i.e. `> 0`), which is true almost every call -- so
  `Xcorrelated`/`Ycorrelated` are true only when a search lands EXACTLY on the previous search-start
  index. With `UseSmartSearch` true (the default), `SearchX`/`SearchY` therefore fall through to
  `BisectionSearchX`/`Y` on nearly every call instead of ever taking the `HuntSearchX`/`Y` branch the
  "smart search" is named for. Contrast the sibling `Interpolater.cs`, whose own
  `deltaStart = Math.Min(1, (int)Math.Pow(Count, 0.25))` at least gives its correlated-search
  machinery a (if degenerate -- see that file's own already-documented issue above) chance of firing.
- **Evidence:** direct inspection of the field declarations and every assignment site in the class
  (grep confirms zero writes to either field outside their `= 0` initializers).
- **Port handling:** mirrored faithfully -- `x_delta_start_`/`y_delta_start_` in
  `ordered_paired_data.hpp` are likewise never assigned past `0`, documented in the file header. Not
  a correctness bug (Bisection always returns the right bracketing index; only the search's
  asymptotic cost is affected), so no fixture specifically isolates it.
- **Suggested C# fix:** assign `XdeltaStart`/`YdeltaStart` a real value (e.g. mirroring
  `Interpolater.cs`'s `Math.Pow(Count, 0.25)` heuristic) so the Hunt branch is reachable as the
  class's own naming and doc comments imply it should be.

## CONSISTENCY — `OrderedPairedData.Add` can widen `IsValid` back to true; `UncertainOrderedPairedData.Add` cannot

- **Where:** `Numerics/Data/Paired Data/OrderedPairedData.cs` @ 2a0357a, `Add` (line 468) vs.
  `Insert` (line 481); `Numerics/Data/Paired Data/UncertainOrderedPairedData.cs`, `Add`
  (~lines 664-671).
- **What:** `OrderedPairedData.Add` assigns `IsValid = OrdinateValid(Count - 1)` UNCONDITIONALLY --
  `OrdinateValid` only inspects the newly-added point's immediate neighbor, not the whole series, so
  appending one well-ordered point after an already-invalid collection can flip `IsValid` back to
  `true` even though an earlier pair still violates the monotonicity contract. `Insert` on the same
  class does not have this bug -- `if (IsValid) IsValid = OrdinateValid(index);` only ever NARROWS
  validity (true -> possibly false), never widens it. `UncertainOrderedPairedData.Add` matches
  `Insert`'s (correct) shape, not `Add`'s: `if (!OrdinateValid(Count - 1)) IsValid = false;` also
  only narrows. The private `OrdinateValid(int)` helper mirrors the same asymmetry at its own
  boundary: `OrderedPairedData`'s version returns `true` for an out-of-range index;
  `UncertainOrderedPairedData`'s returns `false`.
- **Evidence:** direct inspection of all four methods across both classes (not assumed parity between
  the "twin" classes -- see the port's own header note on this).
- **Port handling:** mirrored faithfully on both classes -- `add()` in `ordered_paired_data.hpp`
  keeps the unconditional widening bug; `add()` in `uncertain_ordered_paired_data.hpp` only narrows,
  matching its own C# source. Documented in both file headers as a DELIBERATE cross-class asymmetry,
  not an inconsistency introduced by the port.
- **Suggested C# fix:** change `OrderedPairedData.Add` to only narrow
  (`if (IsValid) IsValid = OrdinateValid(Count - 1);`), matching `Insert` on the same class and
  `Add`/`Insert` on the Uncertain twin.

## CONSISTENCY — `OrderedPairedData.LangSimplify` returns an alias of the receiver on its guard path, unlike its two siblings

- **Where:** `Numerics/Data/Paired Data/OrderedPairedData.cs` @ 2a0357a,
  `LangSimplify(double tolerance, int lookAhead)` (line 1445), the
  `if (lookAhead <= 1 || tolerance <= 0) return this;` guard.
- **What:** `DouglasPeuckerSimplify` and `VisvaligamWhyattSimplify` both always return a freshly
  constructed `OrderedPairedData`. `LangSimplify`'s guard instead returns `this` -- the SAME object
  the caller already holds a reference to, not a copy -- so a caller that mutates the "simplified"
  result under a trivial `lookAhead`/`tolerance` is silently mutating the original curve too.
- **Port handling (a genuine port DECISION, no C# analogue is possible):** `lang_simplify()` in
  `ordered_paired_data.hpp` has a value-returning signature (`OrderedPairedData`, not a reference), so
  there is no C++ construct that aliases `*this` the way a C# reference-type return can --
  `return *this;` by value already copies. This port returns `clone()` instead, matching the OTHER
  two simplifiers' not-the-same-object contract, rather than fabricating aliasing behavior no other
  value-returning method here has. `core/tests/test_ordered_paired_data.cpp`'s
  `test_lang_simplify_guard` asserts the guarded return is content-equal to the original AND
  independently mutable (proving it is a distinct object) -- the only choice a value-returning API
  leaves open.
- **Suggested C# fix:** return a clone (e.g. `this.Clone()`) from the guard, matching
  `DouglasPeuckerSimplify`/`VisvaligamWhyattSimplify`'s contract.

## BUG — `LineSimplification.RamerDouglasPeucker`'s output parameter is cleared in one branch but appended-to in the other

- **Where:** `Numerics/Data/Paired Data/LineSimplification.cs` @ 2a0357a,
  `RamerDouglasPeucker(List<Ordinate>, double, ref List<Ordinate> output)` (line 33).
- **What:** the "keep both endpoints" branch (`dmax <= epsilon`) does
  `output.Clear(); output.Add(...); output.Add(...);` -- replacing whatever the caller passed in. The
  "recurse" branch (`dmax > epsilon`) does `output.AddRange(recResults1...); output.AddRange(
  recResults2);` (lines 59-64) -- APPENDING to whatever the caller passed in, with no `Clear()`
  first. For every call site actually reachable in this codebase the `ref` parameter is always a
  fresh, empty list at each call (including each recursive call), so the two behaviors happen to
  coincide; a caller handing a pre-populated list to the top-level call would see it replaced or
  appended-to depending purely on which branch the top-level recursion takes, which the caller cannot
  predict from the method's own signature.
- **Evidence:** direct inspection of both branches; not exercised by any upstream test with a
  non-empty pre-populated output list.
- **Port handling:** mirrored faithfully -- `ramer_douglas_peucker()` in
  `core/include/corehydro/numerics/data/paired_data/line_simplification.hpp` reproduces both shapes
  (`output.clear()` + two `push_back`s in the "keep both endpoints" branch;
  `output.insert(output.end(), ...)` with no clear in the "recurse" branch), documented in the file
  header as intentional (every reachable call site hands a fresh empty vector, so the asymmetry is
  currently invisible, but is preserved rather than normalized to "always clear first").
- **Suggested C# fix:** call `output.Clear()` unconditionally at the top of the method, before either
  branch, so the `ref` parameter's contract does not depend on which branch is taken.

## CONSISTENCY (documented upstream quirk, not a bug) — `Ordinate.operator==` treats a NaN coordinate as equal to anything

- **Where:** `Numerics/Data/Paired Data/Ordinate.cs` @ 2a0357a, `operator==` (line 317).
- **What:** equality tests `Math.Abs(diff) > Tools.DoubleMachineEpsilon` per coordinate and returns
  `false` only when that predicate is true. `Math.Abs(NaN - anything) > eps` is ALWAYS false (every
  comparison involving NaN is false), so an ordinate holding a NaN coordinate compares EQUAL to every
  other ordinate on that coordinate -- `Ordinate(NaN, 4) == Ordinate(2, 4)` is `true`.
- **Evidence:** the C# source's own comment states this directly, and upstream's `Test_Construction`
  asserts exactly `Ordinate(NaN, 4) == Ordinate(2, 4)` -- this is intentional (if surprising) upstream
  behavior, not an oversight.
- **Port handling:** pinned exactly the same way -- `operator==` in
  `core/include/corehydro/numerics/data/paired_data/ordinate.hpp` uses
  `std::fabs(l.x - r.x) > kDoubleMachineEpsilon`, which is likewise always false for a NaN operand;
  documented in the file header as a "documented upstream quirk, not a bug" with the same test case
  transcribed.
- **Suggested action:** none -- flagged here only so a future reader isn't surprised by
  `NaN == anything` and doesn't "fix" it without realizing it is asserted deliberately by an upstream
  test.

## CONSISTENCY — two independent, incompatible `PerpendicularDistance` implementations in the Paired Data subsystem

- **Where:** `Numerics/Data/Paired Data/OrderedPairedData.cs` @ 2a0357a, private
  `PerpendicularDistance` (~lines 1376-1386, feeding `DouglasPeuckerReduction`) vs.
  `Numerics/Data/Paired Data/LineSimplification.cs`, `PerpendicularDistance` (line 84, feeding the
  free-function `RamerDouglasPeucker`).
- **What:** these are two entirely independent formulas for the same named quantity, both shipped in
  the same subsystem for two different Douglas-Peucker-family implementations. `OrderedPairedData`'s
  version is triangle-area-over-base (`|cross product| / |base|`) and has NO guard against a
  degenerate (zero-length) base segment -- a first/last pair that coincide divides `0.0/0.0`,
  yielding `NaN`. `LineSimplification`'s version normalizes the segment direction to a unit vector
  and explicitly guards the degenerate case with `if (mag > 0.0)`, falling back to the
  point-to-line-start distance when the segment has zero length -- this is exactly the case
  upstream's own `Test_EqualPoints` exercises for the free-function algorithm, but NOT for the
  class-method one.
- **Evidence:** direct inspection of both implementations; no test in either language constructs a
  first/last pair that coincide for `OrderedPairedData`'s own `DouglasPeuckerSimplify`, so the
  unguarded `NaN` path is unexercised on that side.
- **Port handling:** both formulas are transcribed independently and exactly as written --
  `OrderedPairedData::perpendicular_distance` (private, in `ordered_paired_data.hpp`) has no
  degenerate guard; `line_simplification::perpendicular_distance` (in `line_simplification.hpp`) has
  the `mag > 0.0` guard. Neither is "unified" into the other, since they are separate upstream
  algorithms for two different classes, documented cross-referentially in both file headers.
- **Suggested C# fix:** either add the same `mag > 0.0`-style degenerate guard to
  `OrderedPairedData.PerpendicularDistance`, or document why a degenerate first/last pair cannot occur
  at that call site (its only caller, `DouglasPeuckerReduction`, guards `firstPoint != lastPoint - 1`
  but not `Ordinates[firstPoint] != Ordinates[lastPoint]` when they are not the same index).

## CONSISTENCY — `UncertainOrdinate.OrdinateValid` probes the mean while `OrdinateErrors` probes the median, so a curve can be reported invalid with no matching error message

- **Where:** `Numerics/Data/Paired Data/UncertainOrdinate.cs` @ 2a0357a, `OrdinateValid` (line 153)
  vs. `OrdinateErrors` (line 193).
- **What:** both methods probe the same three points along each ordinate's Y distribution --
  `minPercentile`, a central-tendency point, and `1 - minPercentile` -- to decide whether one ordinate
  is a valid monotonic successor/predecessor of another. `OrdinateValid`'s central probe is the MEAN
  (`GetOrdinate()`, no argument, line 173); `OrdinateErrors`'s is the MEDIAN (`GetOrdinate(0.5d)`,
  line 226). For a skewed Y distribution the mean and median differ, so a pair of ordinates can fail
  `OrdinateValid`'s mean probe (making the collection's `IsValid` false) while `OrdinateErrors`'s
  median probe passes cleanly -- the diagnostic method silently omits the very error that made the
  collection invalid.
- **Evidence:** direct inspection of both methods; `minPercentile` is `0.05` when the distribution
  type is `PertPercentile`/`PertPercentileZ`, `1e-5` otherwise, identically in both methods, so the
  mismatch is specifically the mean-vs-median choice, not the tail percentiles.
- **Port handling:** both probe sets are transcribed exactly as written --
  `UncertainOrdinate::ordinate_valid` (in
  `core/include/corehydro/numerics/data/paired_data/uncertain_ordinate.hpp`) calls `get_ordinate()`
  (mean) at the central probe; `ordinate_errors` calls `get_ordinate(0.5)` (median) at the same
  position. Documented in the file header as "a real, upstream mismatch, not a transcription slip."
- **Suggested C# fix:** use the same central-tendency probe (mean or median, whichever is intended) in
  both methods, so `GetErrors()` always explains every case where `IsValid` is false.

## CONSISTENCY — `UncertainOrdinate.operator==` compares X exactly, unlike `Ordinate`'s epsilon tolerance

- **Where:** `Numerics/Data/Paired Data/UncertainOrdinate.cs` @ 2a0357a, `operator==` (line 266), vs.
  `Numerics/Data/Paired Data/Ordinate.cs`, `operator==` (line 317).
- **What:** `Ordinate.operator==` compares each coordinate with `Tools.DoubleMachineEpsilon` slack
  (see the NaN-compares-equal entry above for the consequence of that choice).
  `UncertainOrdinate.operator==` compares `X` with EXACT inequality (`left.X != right.X`) -- no
  epsilon at all -- before delegating `Y`-equality to the distribution's own comparison. Two X values
  that would compare equal under `Ordinate`'s rule (differing by less than machine epsilon) compare
  UNEQUAL here.
- **Evidence:** direct inspection of both operators; the inconsistency is between two classes in the
  same subsystem with the same conceptual "X coordinate," not an internal contradiction within either
  class alone.
- **Port handling:** transcribed exactly as written -- `operator==` for `UncertainOrdinate` in
  `uncertain_ordinate.hpp` uses `l.x != r.x` (no tolerance); `operator==` for `Ordinate` in
  `ordinate.hpp` uses the epsilon-tolerant comparison. Documented in `uncertain_ordinate.hpp`'s header
  as transcription note 3.
- **Suggested C# fix:** decide whether X-equality should be exact or tolerant across both classes, and
  make `UncertainOrdinate.operator==` consistent with `Ordinate.operator==`'s choice (or document why
  uncertain-Y ordinates need a stricter X test than certain ones).

## BUG — `UncertainOrderedPairedData.InsertRange` narrows `IsValid` using the constant insertion index instead of each newly-inserted position

- **Where:** `Numerics/Data/Paired Data/UncertainOrderedPairedData.cs` @ 2a0357a,
  `InsertRange(int index, IList<UncertainOrdinate> items)` (~lines 716-731).
- **What:** after inserting `items.Count` new ordinates starting at `index`, the `IsValid`-narrowing
  loop calls `OrdinateValid(index)` -- the constant, FIRST inserted position -- on every one of its
  `items.Count` iterations, instead of `OrdinateValid(i)` for each newly-inserted position in turn.
  Only the first inserted ordinate's monotonicity is ever actually checked; the rest are inserted
  unchecked. This directly computes `_isValid`, a real externally-observable value (unlike
  `AddRange`'s analogous but inert `startIndex` off-by-one in the same class, which only ever
  feeds the severed `CollectionChanged` event).
- **Evidence:** direct inspection of the loop body (`for (int i = index; i <= index + items.Count -
  1; i++) { if (IsValid) { if (!OrdinateValid(index)) IsValid = false; } }` -- note
  `OrdinateValid(index)`, not `OrdinateValid(i)`).
- **Port handling:** mirrored faithfully -- `insert_range()` in
  `core/include/corehydro/numerics/data/paired_data/uncertain_ordered_paired_data.hpp` calls
  `ordinate_valid(index)` on every pass of its loop over `i`, not `ordinate_valid(i)`, with an inline
  comment marking it as an upstream defect with a real (not inert) consequence, ported exactly as C#
  wrote it rather than "fixed."
- **Suggested C# fix:** change `OrdinateValid(index)` to `OrdinateValid(i)` inside the loop, so every
  newly-inserted ordinate is actually validated against its own neighbors.

## ROBUSTNESS — `UncertainOrderedPairedData.Validate()` early-returns while `SuppressCollectionChanged` is set, leaving `IsValid` stale

- **Where:** `Numerics/Data/Paired Data/UncertainOrderedPairedData.cs` @ 2a0357a, `Validate()`
  (~lines 390-402).
- **What:** every other use of `SuppressCollectionChanged` in this class guards a
  `CollectionChanged?.Invoke` call -- pure observable-collection plumbing with no effect beyond
  whether an event fires. `Validate()`'s own use is different: it early-returns before recomputing
  `_isValid` at all, so `IsValid` can go stale (reflect data from before the suppressed mutations)
  for as long as the flag is set -- a real, externally-observable difference in behavior, not merely
  a suppressed event.
- **Port handling:** preserved as a plain bool with the same early-return -- `validate()` in
  `uncertain_ordered_paired_data.hpp` checks `suppress_collection_changed_` first, unlike every OTHER
  `SuppressCollectionChanged` use in the class (all severed as pure event plumbing, per the
  project-wide precedent). Documented in the file header as the one flag-read that survives the
  plumbing cull.
- **Suggested action:** none required -- this is arguably the intended contract (suppress
  recomputation while a caller performs many mutations, then call `Validate()` once at the end with
  the flag cleared); flagged here so a future reader understands why this one
  `SuppressCollectionChanged` check has a C++ mirror when its siblings do not.

## BUG — `Statistics.RanksInPlace(double[], out double[] ties)` never records the trailing tie run's length

- **Where:** `Numerics/Data/Statistics/Statistics.cs` @ 2a0357a,
  `RanksInPlace(double[] data, out double[] ties)` (line 674).
- **What:** the method sorts a working copy, walks it once tallying consecutive near-equal runs (tie
  test: `AlmostEquals(work[i], work[previousIndex], Tools.DoubleMachineEpsilon)`, i.e.
  `|work[i]-work[previousIndex]| <= DoubleMachineEpsilon` -- NOT exact equality, unlike the sibling
  no-`ties` overload of the same method), and writes each closed run's length into `ties[i - 1]`
  inside the loop's own `else` branch. After the loop ends, a final
  `RanksTies(ranks, index, previousIndex, work.Length)` call closes whatever run was still open --
  correctly averaging that run's ranks -- but this closing call is OUTSIDE the loop and never writes
  to `ties`. A tie run that ends at the very last element of `data` therefore has its rank-averaging
  applied correctly but its run length silently never recorded in `ties`, leaving that slot at its
  default `0`.
- **Evidence:** direct inspection of the loop structure -- `ties[i - 1] = t;` appears only inside the
  `for` loop's `else` branch, and the trailing `RanksTies(...)` call after the loop has no
  `ties`-writing counterpart.
- **Consequence downstream:** `ties` (a sparse "tie run length" array) feeds two hypothesis tests this
  port ships: `MannWhitneyTest`'s tie correction term `T` (`sum((ties[i]^3 - ties[i]) /
  (n*(n-1)))`) and `MannKendallTest`'s variance term `varS`
  (`sum(ties[i]*(ties[i]-1)*(2*ties[i]+5))`). Both under-correct whenever the input's LARGEST-valued
  elements are themselves tied, since that specific run is the one whose length silently drops out.
- **Port handling:** mirrored faithfully -- the tolerance-based `ranks_in_place(data, ties)` overload
  in `core/include/corehydro/numerics/data/statistics.hpp` reproduces all three fidelity points versus
  its own exact-equality sibling overload: the `kDoubleMachineEpsilon`-tolerant tie test, the sparse
  `ties` array indexed by run-closing position, and the never-written trailing run. Documented in the
  file header as "a genuine upstream defect... It MUST be reproduced (not 'fixed') because it is the
  input HypothesisTests' oracle-pinned callers were fit against."
  `numerics::data::hypothesis_tests::mann_whitney_test`/`mann_kendall_test` (in
  `numerics/data/hypothesis_tests.hpp`) consume it unchanged.
- **Suggested C# fix:** move the tie-length write out of the loop's `else` branch (or duplicate it
  after the final `RanksTies` call) so a trailing run's length is recorded like every other run's.

## COSMETIC — `DataFrame.LinearTrendTest` computes a dead local and duplicates the expression, and does not call the identical `HypothesisTests.LinearTrendTest`

- **Where:** `RMC.BestFit/src/RMC.BestFit/Models/DataFrame/DataFrame.cs` @ c2e6192,
  `LinearTrendTest(bool useLog10 = false)` (line 1002).
- **What:** `Numerics.Data.Statistics.HypothesisTests` already has a `LinearTrendTest` static that
  performs the identical linear-regression + Student-t computation (this port's own
  `numerics::data::hypothesis_tests::linear_trend_test`, ported in P4 Task 2).
  `DataFrame.LinearTrendTest` does not call it -- it inlines the same `LinearRegression`/`StudentT`
  construction itself. Within that inlined body it also computes
  `double d = Math.Abs(lm.Parameters[1] / lm.ParameterStandardErrors[1]);` and never uses `d` -- the
  `return` statement recomputes the identical expression
  `Math.Abs(lm.Parameters[1] / lm.ParameterStandardErrors[1])` inline instead of reusing `d`.
- **Evidence:** direct inspection; both oddities (the unused local and the duplicated expression) are
  textually present in the shipped source.
- **Port handling:** both mirrored exactly rather than "cleaned up" -- `DataFrame::linear_trend_test()`
  in `core/include/corehydro/models/data_frame/data_frame.hpp` computes `d` and discards it with
  `(void)d;`, then recomputes the identical expression in its own `return`, and does not delegate to
  `numerics::data::hypothesis_tests::linear_trend_test` even though that free function already exists
  and is identical. Documented as transcription note 1 in the file header.
- **Suggested C# fix:** delete the dead `d` local, or use it in the return expression; separately,
  have `DataFrame.LinearTrendTest` simply call `HypothesisTests.LinearTrendTest(indexes, values)`
  instead of duplicating its body.

## CONSISTENCY — `DataFrame`'s two summary-statistics methods disagree on `Kurtosis` vs `Kurtosis + 3`

- **Where:** `RMC.BestFit/src/RMC.BestFit/Models/DataFrame/DataFrame.cs` @ c2e6192,
  `SummaryStatisticsExactDataOnly` (line 1786) vs. `SummaryStatisticsAllData` (line 1850).
- **What:** both methods report a `"Kurtosis"` (and `"Kurtosis (of log)"`) key computed from the same
  underlying `moments[3]` central-moment slot. `SummaryStatisticsExactDataOnly` reports
  `moments[3] + 3` -- raw excess kurtosis shifted back to Pearson's (non-excess) kurtosis convention,
  where a Normal distribution reads `3`. `SummaryStatisticsAllData` reports the SAME `moments[3]` slot
  with NO `+3` -- excess kurtosis, where a Normal distribution reads `0`. A caller comparing the
  "Kurtosis" key across the two summary methods on the same underlying data is comparing two
  different conventions without any indication in the key name. The same two methods also each build
  their percentile/moment inputs as three INDEPENDENTLY sorted parallel arrays (`values`,
  `log_values`, and the plotting-position `probs`, sorted separately rather than co-sorted as a single
  tuple), a quirk carried through unchanged from the `GetNonparametricMoments` methods this pair
  shares its private tail with; and `SummaryStatisticsAllData` computes central moments with
  `CentralMoments(1000)` (fixed-step trapezoidal, matching the "int steps" overload documented
  earlier in this file) where the unrelated `SetStandardizedValues` method uses `CentralMoments(200)`
  -- different step counts for the same computation, on purpose, unremarked upon in either method.
- **Evidence:** direct inspection of both methods' key-population code and the private tail they
  share.
- **Port handling:** mirrored exactly, as an upstream asymmetry rather than a port inconsistency --
  `summary_statistics_exact_data_only()` in `core/include/corehydro/models/data_frame/data_frame.hpp`
  emits `moments[3] + 3.0`; `summary_statistics_all_data()` emits the bare `moments[3]`; both use
  `central_moments(1000)`. Documented as transcription notes 2 (Kurtosis), 3 (the three independently
  sorted arrays), and 4 (the 1000-vs-200 step counts) in the file header.
- **Suggested C# fix:** pick one convention (excess or Pearson's) and apply it in both methods, or
  rename the keys to disambiguate (`"Excess Kurtosis"` vs `"Kurtosis"`); co-sort `values`/`log_values`
  with `probs` as one tuple rather than three independent sorts; and either document why
  `SetStandardizedValues` needs fewer quadrature steps than the summary methods, or use the same step
  count in both.

## VERIFIED, NOT A BUG — `DataFrame.SummaryHypothesisTest`'s Mann-Whitney argument-selection ternaries

- **Where:** `RMC.BestFit/src/RMC.BestFit/Models/DataFrame/DataFrame.cs` @ c2e6192,
  `SummaryHypothesisTest(int index = -1, bool useLog10 = false)` (line 1077), the call
  `HypothesisTests.MannWhitneyTest(v1.Count <= v2.Count ? v1 : v2, v1.Count > v2.Count ? v1 : v2)`
  (line 1114).
- **What was suspected:** unlike the standalone `DataFrame.MannWhitneyTest(int index, ...)`
  (line 1049), which wraps the WHOLE call in one ternary
  (`sample1.Count <= sample2.Count ? MannWhitneyTest(sample1, sample2) :
  MannWhitneyTest(sample2, sample1)`), `SummaryHypothesisTest` builds each of the two arguments with
  its OWN separate ternary. That shape looks, on a quick read, like it could pass the same sample
  object as both arguments when `v1.Count == v2.Count`.
- **Checked against the actual source, not the suspicion:** the two conditions (`<=` and `>`) are
  exact logical complements for integer counts, so the two separate ternaries are equivalent to the
  single-ternary form at every possible count relationship, including equality. Worked through
  explicitly: `v1.Count == v2.Count` gives `(v1.Count <= v2.Count) == true` so argument 1 is `v1`, and
  `(v1.Count > v2.Count) == false` so argument 2 is `v2` -- the pair is `(v1, v2)`, not `(v1, v1)`.
  `v1.Count < v2.Count` gives `(v1, v2)` (v1, the smaller, first). `v1.Count > v2.Count` gives
  `(v2, v1)` (v2, the smaller, first). Every case matches the single-ternary form's contract of
  "smaller-or-equal sample first."
- **Why this is worth recording anyway:** the method itself (`SummaryHypothesisTest`) is deferred to
  P5 alongside `UnimodalityTest` (both need the unported `GaussianMixtureModel`; see
  `upstream/CLAUDE.md`), so this port has not yet shipped or fixture-tested this call site -- this
  entry exists so whoever un-defers `SummaryHypothesisTest` in P5 does not need to re-derive this
  truth table from scratch, and so a plausible-sounding suspicion about the argument order doesn't get
  treated as a confirmed bug without being checked against the actual ternary semantics.
- **Port handling:** none required -- not yet ported (P5).
- **Suggested action:** none -- the C# is correct as written.

## BUG — `Statistics.ParallelMean` is not reproducible against itself across machines

- **Where:** `Numerics/Data/Statistics/Statistics.cs` @ 2a0357a, `ParallelMean(IList<double>)`
  (line 143): `double sum = data.AsParallel().Sum(); return sum / data.Count;`.
- **What:** PLINQ splits the source across partitions, sums each independently, and adds the
  partial sums. Floating-point addition is not associative, so the result depends on how many
  partitions the runtime chooses -- which is a function of `Environment.ProcessorCount`. Two
  machines running the same build on the same data get different last bits. There is no
  `WithDegreeOfParallelism` or ordered-accumulation constraint to pin it.
- **Evidence (measured against the real library, this machine, `Environment.ProcessorCount = 10`,
  `new Random(42)` uniforms scaled to [0, 1000), ULP difference of `ParallelMean` against a serial
  left-to-right sum):**

  | n | 16 | 32 | 50 | 64 | 100 | 128 | 200 | 256 | 300 | 400 | 500 | 600 | 800 | 1000 | 100000 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | ULP | -1 | -1 | 0 | -1 | 1 | 2 | -1 | -1 | -3 | -4 | -5 | -7 | 0 | 5 | -29 |

  Note there is no safe small-`n` threshold: PLINQ already partitions at n = 16. `Statistics.Mean`
  (the serial overload) agrees with the serial sum exactly at every size, as expected.
- **Why it matters for this port:** two ported call sites read it, both the "mean" column of a
  prediction-interval table -- `RandomForest.Predict` (over `NumberOfTrees` tree predictions,
  default 1000) and `KNearestNeighbors.PredictionIntervals` (over `realizations` bootstrap
  predictions, default 1000). Every other column of both tables (`lower`/`median`/`upper`) comes
  from `Statistics.Percentile` on a sorted array and is exactly reproducible.
- **Port handling (intentional divergence, P5 Task 1):** `numerics/data/statistics.hpp`'s
  `parallel_mean` sums SERIALLY, making it exactly `mean`. There is nothing else it could
  faithfully do: upstream has no single value to be faithful to. The consequence is stated at the
  call sites and the affected fixture assertions carry a relative tolerance with this measurement
  named in their `source` string, rather than an `oracle_skip` mask.
- **Suggested C# fix:** either sum serially (the arrays reaching this method are small enough that
  the parallel overhead is unlikely to pay for itself) or, if the parallelism is wanted, make the
  reduction deterministic -- fix the partition count and accumulate the partial sums in partition
  order, or use a compensated (Kahan/Neumaier) sum so the partition split stops mattering.

## COSMETIC — `Test_GeneralizedLinearModel.cs`'s commented-out `Summary()` transcripts are stale

- **Where:** `Numerics/Test_Numerics/Machine Learning/Supervised/Test_GeneralizedLinearModel.cs`
  @ 2a0357a, the `/* ... */` blocks after each test's `Debug.WriteLine(summary[i])` loop.
- **What:** each test ends with a commented-out copy of the summary table the model used to
  print. For `Test_SimpleLinearRegression` that block reads `AIC: 71.1801  AICc: 71.2453
  BIC: 77.6423`. The shipped library returns `343.25605266374168`, `343.32127005504606` and
  `349.71826989745085` for that exact fit -- 272.08 higher. Nothing fails, because the block is a
  comment and the test never asserts AIC for the identity link (only `Test_Log`, `Test_Logistic`,
  `Test_Probit` and `Test_LogLog` do, and those four are current).
- **How it was found:** the port reproduced the shipped library's values to all 17 digits, and the
  transcript was mistakenly used as an oracle while writing `test_generalized_linear_model.cpp`;
  probing the real library settled it. The parameters and standard errors in the same blocks ARE
  current, which is what makes the stale AIC line easy to trust.
- **Port handling:** none needed -- the port matches the library exactly. The ctest pins the
  measured values and says in place why the transcript must not be used.
- **Lesson worth carrying:** in this corpus, only ASSERTED values in a `[TestMethod]` are oracles.
  A commented-out `Debug.WriteLine` transcript has nothing keeping it honest.
- **Suggested C# fix:** refresh or delete the stale block.

## BUG — `KNearestNeighbors.kNN`'s shape guard is a tautology, so `GetNeighbors` never validates its query

- **Where:** `Numerics/Machine Learning/Supervised/KNearestNeighbors.cs` @ 2a0357a, the private
  `kNN(Matrix xTrain, Vector yTrain, Matrix xTest)` (line 243): `if (NumberOfFeatures !=
  xTrain.NumberOfColumns) return null!;`.
- **What:** `NumberOfFeatures` is defined as `X.NumberOfColumns` and `xTrain` is always
  `this.X`, so the condition is always false. The guard was clearly meant to check the TEST
  matrix -- which is what the sibling `kNNPredict` does correctly (`xTest.NumberOfColumns !=
  xTrain.NumberOfColumns`). So all three public `GetNeighbors` overloads accept any query shape.
- **What happens then:** `Tools.Distance(IList<double> x, IList<double> y)` loops over `x.Count`,
  the QUERY row. A query with FEWER columns than the training matrix therefore silently computes
  a partial-dimension distance and returns plausible-looking neighbors computed from a subset of
  the features; a query with MORE columns indexes past the end of each training row and throws
  `IndexOutOfRangeException` from inside the distance helper.
- **Port handling:** the tautological guard is kept for structural fidelity (and does nothing, as
  upstream); the narrower-query behavior is reproduced exactly, since that is a returned answer a
  caller could depend on; and the wider-query case throws `std::out_of_range` -- the port's
  standard mapping for a C# index exception -- rather than reading out of bounds, which would be
  undefined behavior in C++ rather than a catchable exception. Pinned in
  `test_k_nearest_neighbors.cpp`.
- **Suggested C# fix:** `if (xTest.NumberOfColumns != xTrain.NumberOfColumns) return null!;`,
  matching `kNNPredict`.

## ROBUSTNESS — a default `DecisionTree` regression fit always recurses to one observation per leaf

- **Where:** `Numerics/Machine Learning/Supervised/DecisionTree.cs` @ 2a0357a, `GrowTree`.
- **What:** the stopping criteria are `bestIndex == -1 || depth >= MaxDepth || numberOfLabels <= 1
  || numberOfSamples < MinimumSplitSize`, and for regression `numberOfLabels` is set to
  `yTrain.Length` (the SAMPLE COUNT) rather than the distinct-value count. So the
  `numberOfLabels <= 1` test only fires on a one-row node, making it redundant with the
  `numberOfSamples < MinimumSplitSize` test at the default `MinimumSplitSize = 2`. Nor does the
  purity of a node stop it: `VarianceReduction` on a node whose responses are all equal returns
  `0 - 0 = 0`, which still beats `BestSplit`'s `double.MinValue` seed, so `bestIndex` is always
  set. The result is that a default regression tree keeps splitting until every leaf holds a
  single training observation -- it memorizes the training set.
- **Evidence (measured against the real library):** `new DecisionTree(x, y, 7).Train()` on twelve
  points whose responses are only two distinct values (six 10s at x = 1..6, six 100s at
  x = 100..105) builds a 23-node tree: the root splits at 6, and each side is then a
  right-leaning chain splitting off one point at a time (thresholds 1, 2, 3, 4, 5 and 100, 101,
  102, 103, 104), ending in twelve singleton leaves. A tree that stopped on purity would have two
  leaves. The full probe transcript is pinned in `core/tests/test_decision_tree.cpp`.
- **Consequence:** it is why upstream's own `Test_DecisionTree_Regression` asserts that the tree
  LOSES to a linear model, and why its remark says to "use a Random Forest to get better
  performance" -- bagging is what recovers the variance the unregularized tree throws away. A user
  reaching for a regression tree directly gets a memorizer unless they set `MinimumSplitSize` or
  `MaxDepth` themselves. Classification is unaffected: there `numberOfLabels` IS the distinct
  count, so a pure node stops.
- **Port handling:** mirrored exactly (the tree shape above is pinned as a C#-measured oracle),
  with the ordering recorded as a numbered transcription note in `decision_tree.hpp`.
- **Suggested C# fix:** for regression, stop on variance rather than on count -- either set
  `numberOfLabels` to the distinct count for both modes, or have `VarianceReduction` return
  `double.MinValue` when the parent variance is 0. Either changes fitted trees, so it needs
  re-pinned oracles.

## BUG — `GaussianMixtureModel.MStep`'s positive-definite repair is a no-op (the return value is discarded)

- **Where:** `Numerics/Machine Learning/Unsupervised/GaussianMixtureModel.cs` @ 2a0357a, the last
  line of `MStep()`: `MatrixRegularization.MakeSymmetricPositiveDefinite(Sigmas[k]);`.
- **What:** `MakeSymmetricPositiveDefinite` is `public static Matrix
  MakeSymmetricPositiveDefinite(Matrix M)` (`Mathematics/Linear Algebra/MatrixRegularization.cs`
  line 111) and is PURE -- it builds `S = 0.5 * (M + M.Transpose())`, clones it, adds a
  trace-scaled ridge, and RETURNS the clone. It never writes through its argument. The GMM
  discards the return value, so neither the symmetrization nor the ridge ever reaches
  `Sigmas[k]`, despite the four-line comment above the call explaining why they are needed.
- **What actually keeps the fit alive:** the diagonal floor a few lines earlier
  (`Sigmas[k][d, d] = Math.Max(Sigmas[k][d, d], 1E-6 * colVar)`), which is applied by assignment
  and does work. The M-step's covariance is also symmetric by construction, so the symmetrization
  has nothing to repair in practice; the missing piece is the off-diagonal ridge that would
  protect a near-singular component from failing Cholesky in the next E-step.
- **Port handling:** mirrored, with the call kept and its result explicitly discarded
  (`(void)...`) so the upstream diff keeps mapping line-for-line, and a numbered transcription
  note in `gaussian_mixture_model.hpp` saying why assigning it would be a silent behavior change
  against every oracle.
- **Suggested C# fix:** `Sigmas[k] = MatrixRegularization.MakeSymmetricPositiveDefinite(Sigmas[k]);`
  — but note this CHANGES the fitted covariances (and therefore every downstream oracle,
  including `HypothesisTests.UnimodalityTest`'s p-values), so it needs re-pinned test literals.

## BUG — `GaussianMixtureModel.LogLikelihood` omits the multivariate-normal normalizing constant

- **Where:** `Numerics/Machine Learning/Unsupervised/GaussianMixtureModel.cs` @ 2a0357a,
  `EStep()`: `LikelihoodMatrix[i, k] = -0.5 * (sum + logDet[k]) + Math.Log(Weights[k]);`.
- **What:** the multivariate normal log density is `-0.5 * (D*log(2*pi) + logDet(Sigma) +
  quadform)`. The `D*log(2*pi)` term is missing, so the accumulated `LogLikelihood` is short of
  the true mixture log-likelihood by exactly `n * D/2 * log(2*pi)`.
- **Evidence (measured):** a one-component fit on the 150 iris sepal-length values reports
  `LogLikelihood = -46.198986426940849`; the properly normalized Gaussian log-likelihood at the
  same fitted mean and variance is `-184.03976640764171`. The difference is
  `150 * 0.5 * log(2*pi) = 137.8407...` to the last bit.
- **Impact:** none on the fit (a constant offset cannot change which parameters maximize EM) and
  none on the one upstream consumer: `HypothesisTests.UnimodalityTest` forms
  `2 * (logLH2 - logLH1)` over two fits of the SAME sample, so the constant cancels exactly. It
  does matter to a user treating `LogLikelihood` as a comparable model-selection score across
  datasets of different size or dimension, or feeding it to AIC/BIC.
- **Port handling:** mirrored, pinned in `test_gaussian_mixture_model.cpp` by asserting the
  reported value against `normalized + n * D/2 * log(2*pi)` (so the test states the size of the
  omission rather than hiding it), with a transcription note in the header.
- **Suggested C# fix:** subtract `0.5 * Dimension * Math.Log(2 * Math.PI)` in the E-step. Every
  `LogLikelihood` oracle would move by a known constant; the unimodality p-values would not.

## BUG — `KMeans` with `k = 1` reports a random observation as the cluster mean

- **Where:** `Numerics/Machine Learning/Unsupervised/KMeans.cs` @ 2a0357a, `Train(int seed, bool
  kMeansPlusPlus)`.
- **What:** `Labels` is zero-initialized by the constructor, and `Train` runs the E-step, compares
  the new labels against the old ones, and `break`s BEFORE the M-step when nothing changed. With
  `k = 1` every point is assigned label 0 on the first E-step, which already matches the
  zero-initialized array, so the loop exits on iteration 1 having never computed a centroid. The
  reported `Means[0, *]` is whatever the initializer picked -- with k-means++ that is one randomly
  chosen DATA POINT, not the mean of the data.
- **Evidence (measured against the real library):** `new KMeans(new double[] {1, 2, 3, 10, 11,
  12}, 1)` then `Train(7)` reports `Means[0, 0] = 10` and `Iterations = 1`. The sample mean is
  6.5. The same input at `k = 2` converges correctly (`Means = {11, 2}`, `Iterations = 2`), so the
  defect is specific to the single-cluster case.
- **Related, and NOT a defect:** for `k > 1` the same break-before-M-step ordering means a
  converged fit's `Means` are one M-step behind its `Labels`. That is harmless -- convergence is
  defined as "the E-step changed nothing", so recomputing the centroids from the final labels
  reproduces the reported means exactly (pinned in `test_k_means.cpp`). Only `k = 1` degenerates,
  because there the E-step can never change anything.
- **Port handling:** mirrored, and pinned by `test_k_means.cpp` against the C#-measured values.
  `k_means.hpp` carries the ordering as a numbered transcription note.
- **Suggested C# fix:** initialize `Labels` to `-1` rather than 0, so the first E-step always
  counts as a change; or run one M-step unconditionally before the convergence test.

## BUG — `JenksNaturalBreaks` throws `IndexOutOfRangeException` when every input value is identical

- **Where:** `Numerics/Machine Learning/Unsupervised/JenksNaturalBreaks.cs` @ 2a0357a,
  `Estimate()` (the cluster-construction block at the end) into
  `Support/JenksCluster.cs`'s constructor.
- **What:** With all values equal, every candidate split has variance 0, so the dynamic program's
  `>=` update leaves `lowerClassLimits[l, j]` at its smallest recorded start index. Walking the
  limits back then produces `kclass[0] = lowerClassLimits[k, 2] - 2 = -1`, and
  `new JenksCluster(SortedData, 0, -1)` immediately evaluates `data[endIndex]` = `data[-1]`.
- **Evidence (measured against the real library):** `new JenksNaturalBreaks(new double[20] filled
  with 3.5, 3)` throws `IndexOutOfRangeException: Index was outside the bounds of the array.` The
  constructor's four documented guards (null, empty, `k <= 0`, `k > n`) all pass, so the failure
  surfaces from inside the fitted algorithm rather than from validation.
- **Note on scope:** this is the degenerate case only. Heavily tied data is fine -- upstream's own
  7,889-value test dataset contains long runs of exact zeros and all three of its 5/7/9-class
  oracles reproduce.
- **Port handling:** mirrored. `JenksCluster`'s C++ constructor range-checks and throws
  `std::out_of_range` (the port's standard mapping for a C# index exception) instead of reading
  out of bounds, so the failure is the same failure with defined behavior;
  `test_jenks_natural_breaks.cpp` pins the throw.
- **Suggested C# fix:** detect `SortedData[0] == SortedData[^1]` (or a variance of 0) up front and
  either return a single-class fit or throw a described `ArgumentException`, rather than letting a
  negative index reach `JenksCluster`.

## CONSISTENCY — `TimeSeries.CumulativeSum` silently drops the series' time interval

- **Where:** `Numerics/Data/Time Series/TimeSeries.cs` @ 2a0357a, `CumulativeSum()` (line 476).
- **What:** Every other method that returns a new series builds it with `new TimeSeries(TimeInterval)`.
  This one builds `new TimeSeries()`, whose field initializer is `TimeInterval.OneDay`, so the
  result claims a daily interval no matter what the source had. The ordinate DATES are copied
  intact, so the returned object is internally inconsistent: monthly dates labelled as daily.
- **Evidence (measured against the real library):** a `OneMonth` series of `{1, 2, 3}` returns a
  cumulative sum whose `TimeInterval` is `OneDay`. The values and dates are otherwise correct.
- **Consequence:** anything downstream that branches on the interval sees the wrong one. Within
  Numerics itself the only such consumer is `ConvertTimeInterval`, so a caller who cumulates and
  then converts gets a conversion computed from a 24-hour step.
- **Port handling:** mirrored, with the oddity named in the method's own comment and pinned by
  `test_time_series_container.cpp` (`test_cumulative` asserts the result reports `OneDay` while the
  source still reports `OneMonth`).
- **Suggested C# fix:** `new TimeSeries(TimeInterval)`, matching every sibling.

## ROBUSTNESS — three `TimeSeries` indexed math overloads reach through an index their own guard rejected

- **Where:** `Numerics/Data/Time Series/TimeSeries.cs` @ 2a0357a: `LogTransform(IList<int>, double)`
  (line 410), `Inverse(IList<int>)` (line 461), and `InterpolateMissingData(int, IList<int>)`
  (line 617).
- **What:** The indexed math overloads share one shape -- `if (indexes[i] >= 0 && indexes[i] <
  Count && <condition>) { ... }` -- and the seven that stop there SKIP an out-of-range index
  silently. These three do not. `LogTransform` and `Inverse` add an `else` (or `else if`) that
  writes or reads `this[indexes[i]]` with NO bounds check, so an out-of-range entry raises
  `ArgumentOutOfRangeException` from the indexer. `InterpolateMissingData`'s extrapolation branch
  reads `this[idx - 2]` with no `idx >= 2` guard, which its own non-indexed twin (line 596) DOES
  have, so passing index 1 reads position -1.
- **Evidence (measured against the real library), on a 3-ordinate series:**
  `Add(5, new[]{7})` returns normally having done nothing; `LogTransform(new[]{7})` and
  `Inverse(new[]{7})` both throw `ArgumentOutOfRangeException`; and
  `InterpolateMissingData(1, new[]{1})` on `{1, NaN, NaN}` throws the same.
- **Consequence:** whether an out-of-range index is ignored or fatal depends on which of the ten
  sibling methods you call, which is not a contract a caller can guess.
- **Port handling:** mirrored. Reading position -1 is undefined behaviour in C++, not an exception,
  so the three sites carry an explicit bounds check that throws `std::out_of_range` at exactly the
  point the C# indexer would; the other seven keep skipping. `test_time_series_container.cpp` pins
  both behaviours side by side.
- **Suggested C# fix:** give the three methods the same guard their siblings have (and give the
  indexed `InterpolateMissingData` the `idx >= 2` check its twin already has).

## COSMETIC — `TimeSeries.MovingAverage` / `MovingSum` swap their `ArgumentException` arguments

- **Where:** `Numerics/Data/Time Series/TimeSeries.cs` @ 2a0357a, `MovingAverage` (line 955) and
  `MovingSum` (line 998).
- **What:** Both write `throw new ArgumentException(nameof(period), "The period must be less than
  the length of the time-series.")`. `ArgumentException(string message, string paramName)` takes
  the MESSAGE first, so the parameter name becomes the message and the message becomes the
  parameter name.
- **Evidence (measured against the real library):** calling `MovingAverage(3)` on a 3-ordinate
  series raises `ArgumentException: period (Parameter 'The period must be less than the length of
  the time-series.')`.
- **Note:** the guard itself is `period >= Count`, not `period > Count`, so a window exactly as
  long as the series is rejected rather than producing the single full-series value. That is
  deliberate enough (a one-point output is rarely wanted) that it is recorded here rather than
  filed as a bug, but it IS the boundary a caller has to know.
- **Port handling:** the port throws `std::invalid_argument` carrying the descriptive message and
  keeps the `>=` boundary; `test_time_series_container.cpp` pins the boundary.
- **Suggested C# fix:** swap the two arguments.

## BUG — `PointProcessModel`'s seasonal day-of-year list is date-sorted while the likelihood pairs it positionally with an unsorted series

- **Where:** `RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/PointProcessModel.cs` @
  `c2e6192`, `SetAMSData()` (lines 526-556) against `DataLogLikelihood` (~line 1858) and
  `PointwisePriorLogLikelihood`.
- **What:** The seasonal branch builds an irregular `TimeSeries` from the exact series IN THE
  ORDER THE CALLER SUPPLIED IT, then -- for the water-year convention -- reassigns
  `ts = ts.ShiftDatesByMonth(shift)`. `ShiftDatesByMonth` opens with `SortByTime()`, so the
  returned series is in DATE order. `_potDays` is then filled from that sorted series. But the
  seasonal likelihood consumes it positionally:
  `for (int i = 0; i < POTDays.Count && i < DataFrame.ExactSeries.Count; i++) { double x =
  ExactSeries[i].Value; int day = POTDays[i]; ... }` -- and `ExactSeries` was never sorted. Each
  magnitude is therefore scored against a DIFFERENT event's day of year whenever the input record
  is not already in date order, which decides which of the two seasonal GEV marginals it is
  attributed to.
- **Evidence (measured on the ported path, which transcribes the C# statement for statement):**
  upstream's own `CreateSeasonalPOTDataFrame` helper supplies ten February events followed by ten
  July events. `POTDays` comes back interleaved -- 135, 288, 135, 288, ... (May 15 and October 15
  after the three-month water-year shift) -- while `ExactSeries` is still all-Februaries then
  all-Julys. So the first ten magnitudes, every one a February event, are scored against
  alternating May and October days.
- **Scope:** the calendar-year path (`StartMonth == 1`) is unaffected, because no shift happens
  and `ts` keeps the caller's order. A record supplied in date order is also unaffected, which is
  the common case and probably why this has gone unnoticed.
- **Port handling:** mirrored, with the defect named at the assignment in
  `point_process_model.hpp` and pinned by `test_point_process_model.cpp`'s
  `test_set_ams_data_seasonal_creates_pot_days`, which asserts the interleaved order rather than
  the intuitive one.
- **Suggested C# fix:** sort the exact series into the same order before pairing (or, better,
  carry the day of year ON the ordinate rather than in a parallel list, so the two cannot drift).

## How to work this list later

1. Reproduce each finding directly against the pinned upstream (`dotnet test` a targeted case, or a
   tiny console snippet), confirming the C# behaviour.
2. For each confirmed bug, decide: patch upstream (PR to USACE-RMC) vs. keep the intentional C++
   divergence documented. Any upstream fix that changes an oracle value must be paired with updated
   test literals and a re-run of `tools/verify_oracles.py`.
3. When a new upstream release lands, run the reconciliation pass described in
   `docs/upstream-sync.md`: check every open entry against the shipped source at the new tag,
   append a **Status:** bullet for anything fixed, and retire the matching C++ divergence note.
   Verify each claimed resolution by reading the source at the tag, not by trusting a release note
   or commit message.
4. Once an entry's resolution is confirmed (Status bullet written, fix ported, oracle re-pinned),
   move the whole entry verbatim to `docs/upstream-csharp-issues-resolved.md`, keeping its
   original order there. Only open findings stay in this file.
