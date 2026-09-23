# Upstream C# issues: open log

Running log of **open** bugs, edge-case gaps, and consistency issues in the upstream USACE-RMC
C# libraries (`Numerics` @ `7e8e8d1` = v2.2.0, `RMC.BestFit` @ `c2e6192` = v2.0.0) found while
porting them to the C++ core, plus the port-side items still tracked as follow-ups. Entries whose
resolution is confirmed -- fixed upstream and ported, or fixed on the port side -- are moved to
`docs/upstream-csharp-issues-resolved.md`; a code comment that points here for a resolved finding
is one hop away. The July 2026 upstream sync re-checked every entry against the shipped source at
the pins above, and an August 2026 pass re-verified each entry below directly against the vendored
source and the current packages (now v0.14.0): every upstream claim in this file was re-confirmed
present at the pins (a handful of scope refinements from that pass are recorded inline in the
affected entries), and at that date upstream `main` carried no code commits beyond the pins
(Numerics v2.2.0 was reconciled on 2026-09-22; RMC-BestFit remains at the v2.0.0 pin).

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
  **Still open** at `7e8e8d1` (line unchanged across the v2.2.0 range).


## CONSISTENCY/API — CompetingRisks' correlated CDF never touches MultivariateNormal.CDF()
- **v2.2.0 status:** still open as a design note. The release changes dependency simulation, cache ownership, and probability helpers, but the correlated CDF still dispatches through the HPCM covariance overload rather than `MultivariateNormal.CDF()`.

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
- **v2.2.0 status:** still open as a configuration hazard. RWMH now factorizes its proposal once per configuration, but an all-zero proposal remains non-positive-definite outside successful MAP initialization.

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

## PARTIALLY RESOLVED — SNIS sorts resampling by `Fitness`, not `Weight`; v2.2.0 makes tied fitness stable

- **Where:** `Numerics/Sampling/MCMC/SNIS.cs`, `Sample()`:
  `MarkovChains[0] = MarkovChains[0].OrderBy(x => x.Fitness).ToList();` and the CDF construction
  loop immediately below it. Before v2.2.0 this was the unstable `List<T>.Sort` equivalent.
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
  2. **Sort-tie instability (resolved in v2.2.0).** Before v2.2.0, `List<T>.Sort` was an
     unstable introspective sort. Any model
     with a non-trivial fraction of `-Infinity`-fitness draws (common for a naive/wide-prior SNIS
     configuration, since `LogLikelihood` easily underflows for implausible parameter draws) has
     MANY tied elements at the bottom of the sort. An unstable sort is free to place those tied
     elements in ANY relative order -- which specific `-Infinity` draw lands at output index 0 vs.
     1 vs. ... is not determined by the algorithm's contract, only by the sort implementation's
     internal pivot/partition choices.
- **Historical evidence (reproduced against the pre-v2.2.0 C# library):** the `fixtures/sampling/mcmc/snis.json`
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
- **Port handling:** the remaining sort-key mismatch is mirrored faithfully: `snis.hpp`'s
  `sample()` sorts
  by `.fitness` (not `.weight`), exactly matching the C# comparator's actual (not commented)
  behavior, using `std::stable_sort`. Numerics v2.2.0 replaced `List<T>.Sort` with stable
  `OrderBy(x => x.Fitness)`, so both languages now preserve original draw order within tied
  fitness runs. The prior cross-runtime draw-index hazard is closed.
- **Suggested C# fix:** for the remaining sort-key mismatch, either fix the comment to describe
  what the code does (sort by
  `Fitness`) or change the comparator to `x.Weight.CompareTo(y.Weight)` to match the comment and
  the CDF loop's own variable name -- these are NOT equivalent when an importance distribution is
  supplied, so this is a real behavioral choice, not just a comment fix, and should be resolved
  with the library's intent for how the resampled `Output` list should be ordered and weighted.

## CONSISTENCY — `NextDoubles(length, dimension)` draws each column from its own fresh sub-`MersenneTwister`, not the caller's stream
- **v2.2.0 status:** unchanged. The clock-seed hardening elsewhere in the RNG does not alter this explicit parent-to-child stream construction.

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


## BUG — thinned DEMCzs population-sampler stream diverges C#-vs-C++ (surfaced tightening the UnivariateAnalysis analysis oracle)
- **v2.2.0 status:** still open. The release adds transition diagnostics and sampler hardening, but no thinned population-stream parity fix or discriminating upstream test.

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

## ROBUSTNESS — DIC / WAIC parallel-reduction non-reproducibility (related to the resolved BCa reduction finding)

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
- **v2.2.0 status:** unchanged and deliberate. The revised StudentT copula retains the upper bound of 30.

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
- **v2.2.0 status:** still deferred after Task 15. The complete oracle gate covers every fixture, but these three leaf checks remain intentionally C++-only.

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
- **v2.2.0 status:** still open as a fidelity limit. The new BFGS parameter-change exit removes the repeated stationary line-search failure, but it does not remove the ill-conditioned GMM pass boundary or add a pinned cross-language ensemble curve for this censored case.

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
- **v2.2.0 status:** the parity classification is unchanged. Re-pinned exact counts are DifferentialEvolution 1000, BFGS 144, and MLSL 277; Nelder-Mead, Brent, and Powell counts remain deliberately unasserted for the causes below.

- **Where:** `RMC.BestFit/Estimation/MaximumLikelihood.cs` @ c2e6192 (`TotalFunctionEvaluations`,
  line 164, copied from `Optimizer.FunctionEvaluations` at the end of a successful `Estimate()`),
  against the port's `core/include/corehydro/estimation/maximum_likelihood.hpp` and the six
  optimizers `parse_optimizer` accepts. Surfaced writing `fixtures/estimation/fit_optimizers.json`,
  which pins one fit under each optimizer.
- **What:** fitting a Normal to the shared 10-value `annual_peaks` record, the real C# and the port
  report the same evaluation count for the three optimizers that are genuine `Optimizer` subclasses
  in the port — DifferentialEvolution `1000`, BFGS `144`, MultilevelSingleLinkage `277` — and
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




## Findings from the P4 "data and tests" phase (August 2026): HypothesisTests, Correlation matrix overloads, and the Paired Data subsystem

The P4 phase ported the twelve Numerics `HypothesisTests` statics, the `Correlation` matrix
overloads, the two `DataFrame` hypothesis-test/summary-statistics facades left deferred at Phase 5,
and the entire (previously unported) Paired Data subsystem -- `Ordinate`, `OrderedPairedData`,
`UncertainOrdinate`, `UncertainOrderedPairedData`, `LineSimplification`, and `TabularFunction`. All
of the findings below surfaced while transcribing that subsystem method-for-method against the real
C# source; each is reproduced on purpose in the corresponding C++ header rather than "fixed," with
a numbered transcription note at the call site citing the C# line numbers below.








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


## CONSISTENCY — `UncertainOrdinate.OrdinateValid` probes the mean while `OrdinateErrors` probes the median, so a curve can be reported invalid with no matching error message
- **v2.2.0 status:** still open. The release aligns X equality and documents this probe asymmetry but retains mean in validation and median in diagnostics.

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



## ROBUSTNESS — `UncertainOrderedPairedData.Validate()` early-returns while `SuppressCollectionChanged` is set, leaving `IsValid` stale
- **v2.2.0 status:** unchanged. The release fixes `InsertRange`, while the suppression early return remains.

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
- **Current port status:** the method and its unimodality dependency are now ported and exercised through the DataFrame runner. The ternary remains verified correct and needs no behavioral change.

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
- **Why this is worth recording anyway:** the ternary is easy to misread. The ported
  `SummaryHypothesisTest` now keeps the expression structurally aligned with C#, and the shared
  DataFrame fixture path exercises the resulting Mann-Whitney value.
- **Port handling:** ported unchanged; no correction was needed.
- **Suggested action:** none -- the C# is correct as written.












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
