# Upstream C# issues: resolved log

Archive of the entries from `docs/upstream-csharp-issues.md` whose resolution is **confirmed** --
either fixed upstream (in `Numerics` v2.1.4 / `2a0357a` or `RMC.BestFit` v2.0.0 / `c2e6192`) and
ported, or fixed on the port side with the fix validated by the fixture suite and the dotnet
oracle gate. Entries are preserved verbatim as they stood in the open log, in their original
order; each carries a **Status:** bullet recording what resolved it and where the regression
coverage lives. The July 2026 reconciliation-pass summary is kept at the end of this file.

Open findings live in `docs/upstream-csharp-issues.md`. Code comments that reference that file for
one of the findings below predate this split; the entry they mean is here.

---

## BUG — StudentT PDF omits the 1/σ Jacobian (density does not integrate to 1 for σ≠1) (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Univariate/StudentT.cs`, `PDF(x)`.
- **What:** For the location-scale Student-t with parameters `[μ, σ, ν]`, `PDF` computes the
  standard t density at `Z=(x−μ)/σ` and returns it **without** the `1/σ` scaling factor. The result
  is not a proper probability density when `σ≠1` (it does not integrate to 1).
- **Evidence (oracle):** `StudentT(2.5, 0.5, 4).PDF(1.4)` returns `0.0516476521260042`; a correctly
  scaled density is `0.1032953...` (exactly 2× = 1/σ with σ=0.5).
- **Status:** RESOLVED. Numerics v2.1.4 `PDF` now computes `inverseSigma = 1.0 / Sigma` and
  multiplies both return paths (the ν ≥ 1e8 normal-approximation branch and the general branch)
  by it. Ported in the upstream-sync Task 4; every σ≠1 PDF oracle was re-pinned by bit-scaling
  the old literal by `1/σ`, and a location-scale PDF identity case was added.
- **Port handling (historical):** mirrored faithfully (oracle-verified); the C++ header noted it.
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** multiply the returned density
  by `1.0 / Sigma`. NOTE this changes oracle values for every σ≠1 case — coordinate with the test
  literals.

## BUG — Beta / GeneralizedBeta Mode can fall outside the support (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Univariate/BetaDistribution.cs` and `GeneralizedBeta.cs`, `Mode`.
- **What:** The mode uses `(α−1)/(α+β−2)` (rescaled to `[min,max]` for GeneralizedBeta). The
  guard only special-cases `α≤1 && β≤1` (→ midpoint). When exactly one shape is `<1` and
  `α+β<2`, the denominator `α+β−2` is a small negative number and the formula returns a value far
  outside the support.
- **Evidence (oracle):** `GeneralizedBeta(α=0.42, β=1.57, min=0, max=1).Mode` returns `≈ 58.0`
  (support is `[0,1]`). Same math applies to `BetaDistribution` on `[0,1]`.
- **Status:** RESOLVED. Numerics v2.1.4 widened the guard to the full boundary ladder:
  `α==1 && β==1` and `α<1 && β<1` return the midpoint, `α<=1 && β>=1` returns the lower boundary,
  `α>=1 && β<=1` returns the upper boundary, and only the interior case reaches
  `(α−1)/(α+β−2)`. GeneralizedBeta applies the same ladder rescaled to `[min,max]`; the PERT
  degenerate midpoint is preserved. Ported in the upstream-sync Task 6.
- **Port handling (historical):** mirrored faithfully (oracle-verified).
- **Originally suggested C# fix (this is what v2.1.4 did):** for a U- or J-shaped Beta (`α<1` xor
  `β<1`) the density has no interior mode; return the maximising boundary (`min` or `max`) or
  `NaN`, not the extrapolated formula.

## BUG — GeneralizedLogistic L-moment methods divide by zero at κ=0 (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Univariate/GeneralizedLogistic.cs`,
  `LinearMomentsFromParameters` (`1/κ − π/sin(κπ)`) and `ParametersFromLinearMoments`
  (`sin(κπ)/(κπ)`).
- **What:** Neither guards `κ→0`; both evaluate `0/0` at `κ=0`, yielding `NaN`/`Inf`. κ=0 is the
  ordinary Logistic (a valid, common case).
- **Status:** RESOLVED, and the divergence is retired. Numerics v2.1.4 added an exact `κ == 0`
  branch returning the L'Hôpital limits AND a `|κ| <= NearZero` truncated-series branch that the
  C++ did not have. The upstream-sync Task 5 converged the C++ onto upstream's exact formulation
  (series and all) rather than keeping our own limit-only version, so the values are now
  oracle-verifiable in both branches.
- **Port handling (historical):** intentional divergence — the C++ returned the L'Hôpital limits
  (`L1=ξ, L2=α, T3=0, T4=1/6`), documented in-header; not oracle-verifiable (C# returned NaN).
- **Originally suggested C# fix (v2.1.4 did this and more):** add a `|κ| < NearZero` branch
  returning those limits.

## BUG — LogPearsonTypeIII.LinearMomentsFromParameters overflows for small skew (large α) (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Univariate/LogPearsonTypeIII.cs`, `LinearMomentsFromParameters`
  (the `L2 = ... * Gamma.Function(α+0.5)/Gamma.Function(α) ...` line).
- **What:** For small skew, `α = 4/γ²` becomes large; `Gamma.Function(α)` overflows to `+Inf`
  around `α≳171`, so the ratio is `Inf/Inf = NaN`. The **inverse** method
  `ParametersFromLinearMoments` already has an `α≥100` Stirling-approximation branch for exactly this
  ratio; the forward method does not, so the two directions are inconsistent.
- **Status:** RESOLVED, and the divergence is retired. Numerics v2.1.4's
  `LinearMomentsFromParameters` now branches on `alpha < 100` and otherwise uses
  `L2 = sigma / sqrt(pi) * (1 − 1/(8α) + 1/(128α²))`; it also added an exact `gamma == 0` early
  return. The upstream-sync Task 5 converged the C++ onto upstream's exact correction terms, so
  the previously unverifiable branch now has oracles.
- **Port handling (historical):** intentional divergence — the C++ added its own `α≥100` Stirling
  branch to the forward method, documented in-header; not oracle-verifiable (C# NaN).
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** apply the same Stirling ratio
  (`Γ(α+0.5)/Γ(α) ≈ √α·(1 − 1/(8α) + …)`) in `LinearMomentsFromParameters` as already done in
  `ParametersFromLinearMoments`.

## BUG — UnivariateDistributionFactory has no case for VonMises (falls through to Deterministic) (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Univariate/Base/UnivariateDistributionFactory.cs`,
  `CreateDistribution(UnivariateDistributionType)`.
- **What:** `UnivariateDistributionType.VonMises` has no `case`, so the factory returns a
  `Deterministic` distribution instead — silently wrong for any code that constructs VonMises by type
  (e.g. serialization round-trips, generic UIs).
- **Evidence:** the dotnet oracle emitter had to bypass the factory and `new VonMises()` directly.
- **Status:** RESOLVED. Numerics v2.1.4 rewrote the factory as a complete `switch` with a
  `case UnivariateDistributionType.VonMises: return new VonMises();` arm and a `default` that
  throws on an unknown type instead of silently returning `Deterministic`. It also added
  `TryCreateDistribution(type, out dist)`. Ported in the upstream-sync Task 7 (the C++ factory
  already had VonMises; it gained the throwing default and `try_create_distribution`).
- **Port handling (historical):** the C++ factory included VonMises; the emitter used a
  direct-construction bypass.
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** add the `VonMises` case and
  audit the factory against the full `UnivariateDistributionType` enum.

## BUG (pattern) — SetParameters validates before assigning fields (invalid scale reported valid) (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Univariate/Gumbel.cs`, `SetParameters` / `ValidateParameters`
  ordering (and potentially other distributions using the same field-assignment order).
- **What:** `SetParameters` assigns the location, then validates using the scale field that has **not
  yet been updated** (reads the stale/previous `_alpha`). An invalid incoming scale (`0`, `NaN`,
  negative) can therefore leave `ParametersValid == true`. Location-invalidity is detected correctly;
  scale-invalidity is not, in the affected ordering.
- **Evidence:** scale-invalid `parameters_valid` cases could not be reproduced through the C# oracle
  for Gumbel — the C# reports them valid.
- **Status:** RESOLVED. Numerics v2.1.4 reordered the affected setters to assign every field
  first and then validate the passed arguments: `Gumbel.SetParameters` is now
  `_xi = location; _alpha = scale; _parametersValid = ValidateParameters(location, scale, false)
  is null;`, and the same pattern was applied across the validation wave
  (Logistic, InverseChiSquared, Binomial, Deterministic, NoncentralT) together with NaN/Inf
  rejection in ChiSquared, KernelDensity and others. Ported in the upstream-sync Task 3; the
  scale-invalid `parameters_valid` cases now have C# oracles.
- **Port handling (historical):** the C++ validated the incoming arguments directly (correct), so
  scale-invalid cases were caught but were not oracle-checked.
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** assign all fields before
  calling `ValidateParameters`, or validate the passed arguments rather than the fields.

## VERIFY (arguable BUG) — PearsonTypeIII / LogPearsonTypeIII L-skewness sign for negative skew (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Univariate/PearsonTypeIII.cs` (and `LogPearsonTypeIII.cs`),
  `LinearMomentsFromParameters`.
- **What:** `T3` (L-skewness) is computed from `α = 4/γ²`, which is always positive, and the
  rational approximation yields a **positive** T3 regardless of the sign of the skew `γ`. For a
  negative-skew distribution the L-skewness should be negative. The sign of `γ` appears to be dropped.
- **Status:** RESOLVED, and the answer was yes. Numerics v2.1.4 ends
  `LinearMomentsFromParameters` with `T3 *= Math.Sign(gamma);` in both PearsonTypeIII and
  LogPearsonTypeIII, and both classes gained exact `gamma == 0` / `T3 == 0` branches in both
  directions so the round trip stays consistent. Ported in the upstream-sync Task 5; the
  negative-skew L-moment fixture re-pinned from the old positive value to the signed one.
- **Port handling (historical):** we initially added a sign flip, then reverted it to match C#
  exactly (an early implementer "correction" that broke round-trip consistency with the
  faithfully-ported `ParametersFromLinearMoments`). Kept bug-for-bug; a negative-skew L-moment
  fixture reproduced the C# (positive) value.

## CONSISTENCY — StudentT.InverseCDF extreme-tail overflow ignores location/scale (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Univariate/StudentT.cs`, `InverseCDF` overflow guard.
- **What:** on the extreme-tail overflow path it returns a bare `rflg * double.MaxValue`, without the
  `μ + σ·t` transform applied on the normal return path. For non-default `μ`/`σ` the extreme quantile
  is not on the same scale as the rest of the function.
- **Status:** RESOLVED. Numerics v2.1.4 refactored the tail branch: the `double.MaxValue`
  saturation guard is gone and both branches now return through the location-scale transform
  (`return Mu + Sigma * (rflg * t);`). Ported in the upstream-sync Task 4, with extreme-tail cases
  at p = 1e-155 and 1e-150 pinned to show μ/σ retention.
- **Port handling (historical):** mirrored faithfully.
- **Originally suggested C# fix (v2.1.4 did the equivalent):** return `μ + σ · rflg ·
  double.MaxValue` (or `±Infinity`) for consistency.

## CONSISTENCY — BivariateEmpirical.SetParameters does not invalidate the cached Bilinear (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Multivariate/BivariateEmpirical.cs`, `SetParameters` /
  `CDF(double, double)`.
- **What:** `CDF` lazily builds the `bilinear` field only `if (bilinear == null)`. Calling
  `SetParameters` a second time (new grid) after a `CDF` call has already run does not reset
  `bilinear` to null, so subsequent `CDF` calls keep interpolating against the OLD grid.
- **Status:** RESOLVED. Numerics 2a0357a (v2.1.4) added `bilinear = null;` at the end of
  `SetParameters`, plus new finite-value (NaN/Inf) checks on X1/X2/the probability grid. Ported in
  the upstream-sync Task 9: `set_parameters()` in `bivariate_empirical.hpp` now resets `bilinear_`,
  and `validate_parameters()` gained the matching finiteness checks (checked immediately before
  each ascending-order/range check, same relative position as the new C# guards -- a NaN
  previously slipped through the ascending-order comparisons undetected, since NaN compares false
  in both directions, and reported `parameters_valid() == true`). See
  `fixtures/distributions/multivariate/bivariate_empirical.json`'s
  `set_parameters_invalidates_bilinear_interpolator` (adapted from the new
  `Test_BivariateEmpirical.SetParametersInvalidatesBilinearInterpolator`) and the three
  `*_non_finite_invalid` cases, all reproduced against the real C# library.
- **Evidence:** read from source; not exercised by any fixture pre-v2.1.4 (constructed once, `CDF`
  called several times against the same grid, per `Test_BivariateEmpirical.Test_BivariateEmp`).
  Confirmed reproducible post-fix via `tools/verify_oracles.py`.
- **Port handling (historical, pre-v2.1.4):** mirrored faithfully (`bilinear_` was likewise never
  reset in `set_parameters()`).
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** set `bilinear = null;` at the
  end of `SetParameters`.

## CONSISTENCY — Linear vs. Bilinear use different (clamped vs. unclamped) log10 for the Logarithmic transform (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Data/Interpolation/Linear.cs` (`Tools.Log10`, clamps values `< 1E-16` to
  `1E-16`) vs. `Numerics/Data/Interpolation/Bilinear.cs` (`Math.Log10` directly, no clamp).
- **What:** the two interpolation classes apply the Logarithmic transform inconsistently: Linear
  is guarded against `log(0)`/`log(negative)` producing `-Inf`/`NaN`; Bilinear is not, despite
  Bilinear internally reusing Linear instances for its search machinery.
- **Status:** RESOLVED. Numerics 33dc1af (v2.1.4) switched every `Math.Log10` call in
  `Bilinear.cs`'s Logarithmic-transform branches to the guarded `Tools.Log10`, matching Linear.
  Ported in the upstream-sync T1 task: `bilinear.hpp` now calls
  `corehydro::numerics::clamped_log10` at every log10 site, exactly like `linear.hpp`. See
  `fixtures/special_functions/bilinear.json`'s `log_floor_*` cases (adapted from the new
  `Test_LogarithmicFloorMatchesLinearInterpolation`), confirmed reproducible against the real C#
  Bilinear class.
- **Port handling (historical, pre-v2.1.4):** mirrored faithfully (`Linear::base_interpolate`/
  `extrapolate` used `corehydro::numerics::clamped_log10`; `Bilinear::interpolate` used plain
  `std::log10`), documented at both call sites.

## BUG (risk) — MultivariateNormal.COVSRT "permute limits" loop condition is inverted (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Multivariate/MultivariateNormal.cs`, `COVSRT`, the
  `for (int j = i - 1; j < 0; j--)` loop inside the `CVDIAG <= 0` branch (permute limits/rows when
  a covariance diagonal entry is degenerate).
- **What:** counting DOWN from `j = i - 1` with a `j < 0` continuation condition means the loop body
  only ever runs when `i == 0` (so `j` starts at `-1`, which already satisfies `j < 0`); for every
  `i >= 1` the condition fails immediately and the loop never executes. When it does run (`i == 0`,
  `j == -1`), it immediately indexes `COV[II + j]` with `II == 0`, i.e. `COV[-1]` — in C# this throws
  `IndexOutOfRangeException`. The condition looks like a Fortran `DO j = i-1, 0, -1` mistranslated
  (should be `j >= 0`). For `i >= 1`, the buggy loop was worse than "throws" — it silently did
  NOTHING (falling through to the trailing `Y[i] = 0;` with `A[i]`/`B[i]` unscaled and unpermuted),
  producing a wrong (not crashing) collapsed-CDF result whenever a rank-deficient covariance's
  redundant dimension sorted to any position OTHER than the very first pivot.
- **Status:** RESOLVED. Numerics 651035e (v2.1.4) fixed the whole degenerate-diagonal region:
  loop bounds `j >= 0` / `l <= j` / `l <= i - 1` / `k >= l` / `m <= k`, the packed index
  `l * (l + 1) / 2 + j + 1` (was `(l - 1) * l / 2 + j + 1`), the swap offset `IJ - k + m - 1` (was
  `IJ - k + m`), and the `IJ` decrement `IJ - k - 1` (was `IJ - k`). Ported in the upstream-sync
  Task 9: `covsrt` in `multivariate_normal.hpp` replaces the whole region wholesale, transcribed
  verbatim from the fixed C# rather than patched incrementally, retiring the bounds-guard divergence
  below. New coverage: `fixtures/distributions/multivariate/multivariate_normal.json`'s
  `cdf_perfect_correlation_collapse`, `cdf_perfect_anticorrelation_collapse`, and
  `cdf_permuted_rank_deficient_collapse_{a,b}` (the last pair drives the SAME rank-1 redundancy
  through two different sorted positions, exercising the fix at more than one `i`), all reproduced
  against the real C# library.
- **Evidence:** static analysis of the loop bounds; not hit by any existing unit test pre-v2.1.4
  (requires a degenerate/near-zero effective covariance diagonal at a non-first COVSRT-sorted
  pivot). Confirmed reproducible post-fix via `tools/verify_oracles.py`.
- **Port handling (historical, pre-v2.1.4):** the C++ (`core/include/.../multivariate_normal.hpp`,
  `covsrt`) transcribed the loop verbatim but added a minimal bounds guard immediately before the
  first `COV[II + j]` access: if `II + j < 0` it threw `std::out_of_range` instead of indexing.
  `std::vector::operator[]` performs no bounds check, so the verbatim `i==0` access would otherwise
  be undefined behavior (a heap-corrupting out-of-bounds write) rather than the catchable
  `IndexOutOfRangeException` C# raises there. Retired in Task 9 -- the fixed loop no longer produces
  an out-of-range access on this port's fixture-exercised inputs.
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** change the loop condition to
  `j >= 0` (or reverse the iteration order to match the apparent Fortran intent); add a regression
  test with a rank-deficient covariance matrix that forces a non-first sorted pivot to be
  numerically zero.

## COSMETIC — MultivariateNormal.MVNDNT return value is always 0 (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Multivariate/MultivariateNormal.cs`, `MVNDNT`.
- **What:** the local `result` is initialized to `0` and never reassigned anywhere in the method
  body, so `MVNDNT` always returns `0.0`. Its only caller, `MVNDST`, casts this return straight into
  `INFORM` (`INFORM = (int)MVNDNT(...)`), so `INFORM` is always (re)initialized to `0` regardless of
  what `COVSRT`/`MVNLMS`/`BVNMVN` computed; only the `N-INFIS >= 2` branch (via `DKBVRC`) can set it
  to anything else afterward.
- **Status:** RESOLVED. Numerics 2a0357a (v2.1.4) changed `MVNDNT` from `private double` to
  `private void`, with `MVNDST` now setting `INFORM = 0` explicitly before calling it instead of
  casting the old always-0 return value -- a pure shape mirror, no behavior change. Ported in the
  upstream-sync Task 9: `mvndnt`/`mvndst` in `multivariate_normal.hpp` mirror the new void shape.
- **Port handling (historical, pre-v2.1.4):** mirrored faithfully (`mvndnt` always returned `0.0`),
  documented in-header. Harmless in practice — `INFORM` ended up correct for the only case that
  mattered (multi-dimensional integration) — but the return value itself was dead code.
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** either remove the unused
  return value (change `MVNDNT` to `void`) or wire it
  up if some future INFORM semantics were intended for the `N-INFIS` 0/1 branches.

## COSMETIC — NoncentralT dead `TT` variable (RESOLVED in Numerics v2.1.4 / 2a0357a)

(Split out of the "dead variables / heritage artifacts" entry; the still-open items from that
entry remain in the open log.)

- `NoncentralT.cs`: a `TT` variable is assigned then never used after the sign flip (a FORTRAN
  translation artifact). Harmless. **RESOLVED**: v2.1.4's AS 243 de-goto refactor removed it
  (three `TT` occurrences at `a2c4dbf`, zero at `2a0357a`); the math is unchanged. Ported in the
  upstream-sync Task 3.

## BUG — ArchimedeanCopula.ValidateParameter never returns null, so ParametersValid is always false (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Bivariate Copulas/Base/ArchimedeanCopula.cs`,
  `ValidateParameter(double parameter, bool throwException)`.
- **What:** The base `BivariateCopula.Theta` setter is `_parametersValid = ValidateParameter(value,
  false) is null;` (see `BivariateCopula.cs`) -- the C# convention used correctly by
  `NormalCopula`/`StudentTCopula`, whose `ValidateParameter` `return null;` when the parameter is in
  range. `ArchimedeanCopula.ValidateParameter`'s final branch instead did
  `return new ArgumentOutOfRangeException(nameof(Theta), "Parameter is valid");` -- a non-null
  exception object, even though the message says the parameter IS valid. Because `is null` was
  therefore always false, `ParametersValid` was unconditionally `false` for any Archimedean copula
  that does NOT override `ValidateParameter` itself: Clayton, Gumbel, and Joe, but NOT AliMikhailHaq
  (AMH) or Frank -- `AMHCopula.cs` and `FrankCopula.cs` each have their own `ValidateParameter`
  override that is textually identical to `ArchimedeanCopula`'s except the final branch correctly
  `return null;`, so `AMHCopula`/`FrankCopula` instances got a correctly-working `ParametersValid`
  even before the fix. (An earlier version of this entry said the bug affected "every
  Archimedean-derived copula (Clayton, AliMikhailHaq, Frank, Gumbel, Joe)" -- that blanket claim was
  wrong for AMH/Frank; corrected here after reading all five concrete `.cs` files directly.) This
  never affected `PDF`/`CDF`/`InverseCDF` or any fit for any copula -- the "valid" branch never
  threw -- only the `ParametersValid` getter was wrong, and only for Clayton/Gumbel/Joe.
- **Status:** RESOLVED. Numerics 2a0357a (v2.1.4) changed the final branch to `return null;`,
  flipping `ParametersValid` from `false` to `true` for every in-range Clayton/Gumbel/Joe instance,
  and additionally added a NaN/Inf-first check ahead of the range check in
  `ArchimedeanCopula.ValidateParameter` (and in `AMHCopula`/`FrankCopula`/`NormalCopula`/
  `StudentTCopula.ValidateParameters`, none of which had ever had the sentinel bug but also lacked
  a finite-parameter guard before v2.1.4). The same commit added `BivariateCopula.CloneMarginal`
  (a new protected static helper) and routed every concrete `Clone()` through it so cloned copulas
  deep-copy their marginals instead of aliasing them. Ported in the upstream-sync Task 8: the
  sentinel fix and NaN/Inf guard in `archimedean_copula.hpp` (plus per-family NaN/Inf guards in
  `amh_copula.hpp`/`frank_copula.hpp`/`normal_copula.hpp`/`student_t_copula.hpp`), and
  `clone_marginal` in `bivariate_copula.hpp` called from all seven concrete `Clone()` overrides. See
  `fixtures/distributions/copulas/*.json`'s new `parameters_valid_*` cases (re-pinned from an
  implicit `false` to `true` for Clayton/Gumbel/Joe's valid-theta case, confirmed reproducible
  against the real C# library by `verify_oracles.py`) and `core/tests/test_copula_clone.cpp` (the
  Clone()-identity/deep-copy half, which does not fit the declarative fixture shape -- see that
  file's header).
- **Evidence (reproduced against the real C# library):** pre-fix, `new
  ClaytonCopula(2.0).ParametersValid` returned `false` even though `theta_minimum = -1`,
  `theta_maximum = +inf`, and `2.0` is well within range; `ValidateParameter(2.0,
  false).Message` was `"Parameter is valid (Parameter 'Theta')"` -- a non-null object. `new
  NormalCopula(0.5).ParametersValid` correctly returned `true` for the equivalent in-range case,
  confirming the divergence was specific to `ArchimedeanCopula.ValidateParameter`, not a design
  choice shared by all copulas. Post-fix (v2.1.4 / Task 8), `new ClaytonCopula(2.0).ParametersValid`
  returns `true`, confirmed reproducible via `tools/verify_oracles.py`.
- **Port handling (historical, pre-v2.1.4):** mirrored faithfully. `archimedean_copula.hpp`'s
  `validate_parameter` returned a non-nullopt "Parameter is valid" message in the final branch
  (affecting `clayton_copula.hpp`/`gumbel_copula.hpp`/`joe_copula.hpp`, which do not override it),
  while `amh_copula.hpp`/`frank_copula.hpp` each carried their own correct override (`return
  std::nullopt;` in range), matching their C# counterparts. No fixture asserted `parameters_valid`
  on any Archimedean copula, since the value was not independently informative once the bug (and
  which copulas it did/didn't affect) was known.
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** change
  `ArchimedeanCopula.ValidateParameter`'s final branch to `return null;`, matching
  `NormalCopula`/`StudentTCopula`/`AMHCopula`/`FrankCopula`. Upstream did not delete the
  now-redundant `AMHCopula`/`FrankCopula` overrides (they remain, textually equivalent to the fixed
  base), which is harmless.

## ROBUSTNESS — JointProbabilityHPCM's `cdf < 1e-300` underflow guard is commented out in cycle 1 (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Data/Statistics/Probability.cs`, `JointProbabilityHPCM`.
- **What:** the "First cycle" block computes `cdf = Normal.StandardCDF(z1);` and then has a
  commented-out line `//if (cdf < 1e-300) cdf = 1e-300;` immediately before `A = pdf / cdf;` (a
  potential division by a near-zero `cdf`). The "Remaining cycles" loop (only reached when the
  number of correlated events `n >= 3`) recomputes the analogous `cdf` and keeps the identical
  guard ACTIVE (not commented out) right before the same `A = pdf / cdf;` division. This asymmetry
  — a guard present in one loop and disabled (via comment) in a structurally identical one three
  lines apart — looks like an accidental omission (e.g. a guard added later to fix a NaN/Infinity
  seen only in the multi-cycle case, never back-ported to cycle 1) rather than an intentional
  design choice.
- **Status:** RESOLVED. Numerics 33dc1af (v2.1.4) named the guard `minimumCdf` and activated it
  in the "First cycle" block too, matching "Remaining cycles". Ported in the upstream-sync T1
  task: `joint_probability_hpcm` in `probability.hpp` now applies the shared `kMinimumCdf`
  constant in both places. See `fixtures/special_functions/probability.json`'s
  `extreme_probabilities_*` cases (adapted from the new
  `Test_JointProbabilityHPCM_ExtremeProbabilitiesRemainFinite`), confirmed reproducible against
  the real C# Probability class — though note the companion finding directly below, which this
  new coverage surfaced as a genuine prerequisite.
- **Evidence:** direct inspection of `Probability.cs`; not hit by any CompetingRisks fixture
  (`R[0,0] = Normal.StandardZ(probabilities[0])` is never so extreme that
  `Normal.StandardCDF(R[0,0])` underflows below `1e-300` for any component/x combination the
  fixtures exercise — the two closest CompetingRisks fixture cases keep `z1` well within a few
  standard deviations of the median).
- **Port handling (historical, pre-v2.1.4):** mirrored faithfully (`joint_probability_hpcm` in
  `probability.hpp` left the guard commented out in the analogous "First cycle" block, applied it
  in "Remaining cycles"),
  documented in-header at both the file comment and the function itself.
- **Suggested C# fix:** either add the matching `if (cdf < 1e-300) cdf = 1e-300;` guard to the
  first cycle (for consistency and to avoid a potential `A = pdf / 0` -> `Infinity`/`NaN` if `z1`
  is extreme enough), or confirm the omission is deliberate (e.g. cycle 1's `cdf` is provably
  bounded away from zero by some invariant not obvious from the code) and document why.

## BUG (C++ port only, unrelated to any C# diff) — `Normal::standard_cdf` used an unguarded erf formula, diverging from C#'s `Normal.StandardCDF` at extreme |z| (FIXED in the upstream-sync T1 task)

- **Where:** `core/include/corehydro/numerics/distributions/normal.hpp`, `standard_cdf(double z)`
  (the static helper mirroring `Normal.StandardCDF(Z)`; NOT the instance `cdf(x)` method, which
  correctly mirrors C#'s own separate, equally-unguarded `Normal.CDF(x)` — see below).
- **What:** discovered while curating the `Probability.hpcm_joint`/`hpcm_conditional_at` fixture
  cases for the JointProbabilityHPCM `minimumCdf` guard fix above. `standard_cdf` computed
  `Phi(z) = 0.5 * (1.0 + std::erf(z / sqrt(2)))` directly. For `z <~ -6`, `erf(z/sqrt2)` is so
  close to `-1` that `1.0 + erf(...)` suffers catastrophic cancellation and rounds to EXACTLY
  `0.0` in double precision — e.g. `standard_cdf(-9)` returned `0.0` instead of the true
  `~1.13E-19` — silently wrong, not merely imprecise. The real C# `Normal.StandardCDF(Z)` does
  NOT hit this: per `Numerics/Distributions/Univariate/Normal.cs`, it delegates to
  `MultivariateNormal.MVNPHI(Z)`, a Chebyshev-series algorithm (Schonfelder 1978) accurate to
  1E-15 across the whole range — the SAME algorithm this port's own
  `MultivariateNormal::mvnphi` already mirrors faithfully for the bivariate-CDF machinery, but
  calling it from `normal.hpp` isn't practical (`multivariate_normal.hpp` already depends on
  `normal.hpp`, so the reverse include would be circular). Note this affects ONLY the static
  `Normal.StandardCDF`/`standard_cdf` helper: the instance `Normal.CDF(x)`/`Normal::cdf(x)` (used
  by an actual `Normal(mu, sigma)` distribution instance) has its OWN separate erf-based formula
  in C# too (`0.5d * (1.0d + Erf.Function((x - Mu) / (Sigma * Tools.Sqrt2)))`), so that method's
  matching imprecision at extreme tails is a faithful mirror, not a divergence — left unchanged.
- **Evidence:** discovered because `JointProbabilityHPCM`'s z-values are clamped to `[-9, 9]`
  before every `Normal.StandardCDF`/`standard_cdf` call, so the new `Test_JointProbabilityHPCM_
  ExtremeProbabilitiesRemainFinite` inputs (probabilities `0` and `1E-320`) drive `z` to exactly
  that boundary. Comparing this port's (buggy) output against `tools/oracle_emitter --dump`
  showed a stark mismatch (this port: `joint = 0`, `cond = [0, 1, 0]`; real C#: `joint =
  1.71146...E-26`, `cond = [1.1286...E-19, 1.5176...E-7, 0.99927...]`) that couldn't be explained
  by the `minimumCdf` guard itself (both old and new C++ agreed with each other, since the
  guard's effect saturates against the same downstream `[0,1]` clamp either way).
- **Port handling:** fixed via the standard numerically-stable identity `Phi(z) =
  0.5 * erfc(-z/sqrt2)` (mathematically identical to the erf form since `erfc(x) = 1 - erf(x)`,
  but `erfc` doesn't lose precision as its argument grows, so it never cancels down to exactly
  0/1). Verified to reproduce the real C# MVNPHI-based value at `z=-9` to ~1E-15 RELATIVE
  precision (`1.1285884059538425E-19` here vs. `1.128588405953841E-19` from `oracle_emitter
  --dump`) — the two differ by ~6 ULP, NOT bit-for-bit, since C#'s MVNPHI is itself only a
  ~1E-15-accurate Chebyshev-series approximation (per its own doc comment) while this `erfc`
  form is the more accurate of the two — comfortably within the fixture's 1E-8 relative
  tolerance. Agrees with the old erf formula to ~1E-16 for every ordinary `z` any other
  existing fixture exercises — confirmed by the full `verify_oracles.py` gate (same 10
  pre-existing failures, zero new ones) and the full C++/R/Python suites (all green) after the
  change.
- **Suggested C# fix:** none — this is a C++-port-only bug with no C# analog to fix.

## CONSISTENCY — CompetingRisks.CreateMultivariateNormal() zeroes the public CorrelationMatrix as a side effect (PerfectlyNegative only) (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Univariate/CompetingRisks.cs`, `CreateMultivariateNormal()`.
- **What:** in the `PerfectlyNegative` branch, the method does `CorrelationMatrix = new double[D,
  D];` (assigning a FRESH all-zero `D x D` matrix through the public property setter) and then
  fills a SEPARATE local `sigma` array with the actual synthetic rho matrix
  (`rho = -1/(D-1) + sqrt(ε)`) that gets passed to `new MultivariateNormal(mu, sigma)`. The public
  `CorrelationMatrix` getter therefore reads back all zeros after any CDF/PDF call in
  `PerfectlyNegative` mode — not the rho matrix the MVN's `.Covariance` actually holds. This looks
  like a leftover from refactoring (the zero matrix was probably meant to be a scratch buffer, not
  assigned to the public property) rather than intentional API design.
- **Status:** RESOLVED. Numerics 2a0357a (v2.1.4) deleted the `CorrelationMatrix = new double[D,
  D];` line entirely -- the `PerfectlyNegative` branch now builds its synthetic rho matrix purely
  into the local `sigma` array, leaving the public `CorrelationMatrix` untouched. The same commit
  also gave `Dependency` a side-effecting setter (invalidates the cached MVN when the mode
  actually changes) and made `SetParameters`/`ValidateParameters` throw/honor `throwException` for
  an empty `Distributions` list. Ported in the upstream-sync Task 9: `create_multivariate_normal()`
  in `competing_risks.hpp` no longer assigns to `correlation_matrix_` in the `PerfectlyNegative`
  branch; `dependency` became a private field with `dependency()`/`set_dependency()` accessors
  (invalidating `mvn_created_` on an actual change) in place of the plain public field the prior
  no-side-effect property justified; `set_parameters()` throws `std::invalid_argument` on a
  flattened-length mismatch; and a new `validate_parameters(parameters, throw_exception)` mirrors
  the throwException contract for the empty-Distributions case. See
  `fixtures/distributions/univariate/competing_risks.json`'s `dependency_change_cdf_before` /
  `dependency_change_correlation_preserved` / `dependency_change_cdf_after` (adapted from the new
  `Test_DependencyChangeInvalidatesMvnWithoutMutatingCorrelation`), all reproduced against the real
  C# library.
- **Evidence:** direct inspection; not exercised by any fixture assertion pre-v2.1.4 (no fixture
  read `CorrelationMatrix` back after a `PerfectlyNegative` CDF/PDF call). Confirmed reproducible
  post-fix via `tools/verify_oracles.py`.
- **Port handling (historical, pre-v2.1.4):** mirrored faithfully (`create_multivariate_normal()`
  in `competing_risks.hpp` likewise overwrote the mutable `correlation_matrix_` field with zeros in
  the `PerfectlyNegative` branch), documented in-header.
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** use a local scratch array
  (e.g. `var scratchCorr = new double[D, D];`) instead of assigning through the public
  `CorrelationMatrix` property, so `CorrelationMatrix` retains whatever the caller last set (or
  `null`) rather than being silently zeroed by a `PerfectlyNegative`-mode CDF/PDF evaluation.

## BUG — Histogram.AddData's out-of-range "auto-adapt" branches are unreachable dead code (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Data/Statistics/Histogram.cs`, `AddData(double data)`.
- **What:** the XML doc comment promises "If the data value falls outside the range of the
  histogram, the start or end bin will automatically adapt," and the method body has branches
  (`data <= LowerBound` / `data >= UpperBound`) that look like they implement this by widening
  the first/last bin. But the method calls `GetBinIndexOf(data)` **unconditionally**, before
  either branch is checked — and `GetBinIndexOf` itself throws `ArgumentException` for any value
  strictly outside `[_bins.First().LowerBound, _bins.Last().UpperBound]` (which track the same
  bounds as the histogram's own `LowerBound`/`UpperBound`). So a point that would need the
  histogram to "auto-adapt" throws instead, before the adapting branch is ever reached. The two
  branches are only reachable at an **exact boundary match** (`data == LowerBound` or
  `data == UpperBound`), where the "expansion" is a no-op (each sets a bound to the value it
  already equals).
- **Status:** RESOLVED. Numerics 33dc1af (v2.1.4) moved the `GetBinIndexOf` call into an `else`
  branch reached only for genuinely interior values, and split each boundary branch into "exact
  match" (no-op) vs. "beyond the boundary" (now actually widens `LowerBound`/`UpperBound` and the
  endpoint bin). Ported in the upstream-sync T1 task, including one upstream asymmetry
  transcribed verbatim (the lower-extend branch marks the bins stale for re-sort, the
  upper-extend branch does not — harmless, since widening an endpoint never changes bin order).
  See `fixtures/special_functions/histogram.json`'s `adapt_*` cases (adapted from the new
  `Test_AddData_AdaptsEndpointBins`), confirmed reproducible against the real C# Histogram class.
- **Evidence:** direct inspection of the method body's statement order (`SortBins(); int index =
  GetBinIndexOf(data); if (data <= LowerBound) {...}`); confirmed by tracing `GetBinIndexOf`'s own
  guard (`if (value < _bins.First().LowerBound || value > _bins.Last().UpperBound) throw ...`).
  Not exercised by any Test_Histogram.cs test (none call `AddData` a second time with an
  out-of-range point after construction).
- **Port handling (historical, pre-v2.1.4):** mirrored faithfully — `histogram.hpp`'s
  `add_data(double)` called `get_bin_index_of()` first and let it throw, exactly like the C#;
  documented in the file header and at the call site.

## BUG — Search.Bisection always returns `start` in descending order (dead-branch comparator) (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Data/Interpolation/Support/Search.cs`, `Bisection(double x, IList<double>
  values, int start, SortOrder order)` (all three overloads share the same loop body).
- **What:** the bisection loop's branch condition is `x >= values[xm] && order ==
  SortOrder.Ascending` — a logical AND against the order flag, not the `(x >= values[xm]) ==
  ascending` equality test the algorithm needs to work in both directions (compare
  `Interpolater.cs`'s own `BisectionSearch`, which correctly uses the `==`-style test, or
  `Search.Hunt`'s analogous loop in the same file, which also gets it right via a boolean `ASCND`
  compared with `==`). For `order == SortOrder.Descending`, `order == SortOrder.Ascending` is
  always `false`, so the whole condition is always `false` regardless of `x` — the loop only ever
  shrinks `xhi`, `xlo` never advances past `start`, and `Bisection` returns `start` unconditionally
  instead of the correct bracketing index.
- **Status:** RESOLVED. Numerics 33dc1af (v2.1.4) split the loop into separate ascending
  (`x >= values[xm]`) / descending (`x < values[xm]`) branches rather than adopting the
  equality-test phrasing Interpolater/Hunt already used, but the effect is the same: descending
  bisection now correctly narrows `xlo` toward the bracketing index. Ported in the upstream-sync
  T1 task's `search.hpp`. See `fixtures/special_functions/search.json`'s `*_descending_*` cases
  (adapted from the new `Test_Search.cs`), confirmed reproducible against the real C# Search
  class, including two cases that genuinely distinguish old (`start`) from new (correct index)
  behavior.
- **Evidence:** direct inspection of the loop body; reproduced independently in a standalone
  Python re-implementation of the exact algorithm during the P3.3 port (a 5-element descending
  array bisected for a midrange value returns `start` regardless of where the value actually
  falls, while `Search.Sequential` on the same inputs returns the correct index).
- **Port handling (historical, pre-v2.1.4):** mirrored faithfully (verbatim, not "fixed") —
  `search.hpp`'s `bisection()` kept the same `&&`-against-`SortOrder::Ascending` condition;
  documented at length in the file header, including a warning that it was dead code for every
  current caller (Histogram and SNIS both only ever call with the default `Ascending` order) but
  a live bug if a future caller passed `Descending`.
- **Originally suggested C# fix (superseded by the actual v2.1.4 fix above):** change the
  condition to `(x >= values[xm]) == (order == SortOrder.Ascending)`, matching
  `Interpolater.BisectionSearch`'s already-correct phrasing of the same test. Upstream instead
  split the loop into two branches (see Status above) — a different but equally correct fix.

## CONSISTENCY — MCMCSampler.MAP.Fitness is on a different scale than every other chain-state fitness after a successful MAP initialization (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Sampling/MCMC/Base/MCMCSampler.cs`, `InitializeChains()`'s `MAP` branch
  (`MAP = DE.BestParameterSet.Clone();`) versus `Sample()`'s output-phase MAP tracking
  (`if (_chainStates[j].Fitness > MAP.Fitness) MAP = _chainStates[j].Clone();`).
- **What:** `DifferentialEvolution.Maximize()` sets `_functionScale = -1` and every fitness it
  records (including `BestParameterSet.Fitness`) is `-1 * LogLikelihoodFunction(x)` -- the
  *scaled* (negated) objective, not the sampler's own unscaled log-likelihood convention. When
  `Initialize == MAP` succeeds, `MAP = DE.BestParameterSet.Clone();` copies that *scaled*
  fitness directly into `MCMCSampler.MAP.Fitness`. For any typical negative log-likelihood
  (the overwhelmingly common case) this makes `MAP.Fitness` a large *positive* number, while
  every `_chainStates[j].Fitness` the output phase compares it against is the sampler's normal
  *unscaled* (negative) log-likelihood. The comparison `_chainStates[j].Fitness > MAP.Fitness`
  is thereby nearly always false, so `MAP` is effectively frozen at the DE estimate for the
  rest of `Sample()` -- the output-phase MAP-tracking loop is live code but practically
  never fires after a successful MAP initialization.
- **Status:** RESOLVED. Numerics 2a0357a (v2.1.4) changed the `MAP` branch to `MAP = new
  ParameterSet((double[])DE.BestParameterSet.Values.Clone(), -DE.BestParameterSet.Fitness);` --
  negating `DE`'s scaled fitness back onto the sampler's own unscaled log-likelihood convention
  before storing it. Ported in the upstream-sync Task 10:
  `initialize_chains()`/`InitializeChains` in `mcmc_sampler.hpp` now constructs `map_` the same
  way (`ParameterSet(de.best_parameter_set().values, -de.best_parameter_set().fitness)`), so the
  output-phase `chain_states_[j].fitness > map_.fitness` comparison is now meaningful again (a
  later-sampled chain state CAN re-trigger and overtake the initial MAP estimate). The
  `normal_rstan` MCMC fixture case (`fixtures/sampling/mcmc/rwmh.json`) re-pins `map_fitness` from
  its old (buggy) `473.558...` (positive) to `-473.558...` (negative, matching every per-draw
  chain fitness in the same run), reproduced against the real C# library.
- **Evidence:** `tools/verify_oracles.py` confirms the re-pinned `map_fitness` value reproduces
  against the real C# library post-fix.
- **Port handling (historical, pre-v2.1.4):** mirrored faithfully -- `mcmc_sampler.hpp`'s
  `sample()` had the identical `chain_states_[j].fitness > map_.fitness` comparison, and
  `initialize_chains()`'s MAP branch copied `de.best_parameter_set()` (also on DE's
  scaled-fitness convention) into `map_` unmodified; the `normal_rstan` fixture case asserted
  `map_fitness` at its (buggy, positive) value specifically to lock this behavior in, not to
  celebrate it.
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** re-scale
  `DE.BestParameterSet.Fitness` back to the unscaled log-likelihood convention when copying it
  into `MAP`.

## COSMETIC — MCMCSampler.InitializeChains computes an unused midpoint vector in its MAP branch (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Sampling/MCMC/Base/MCMCSampler.cs`, `InitializeChains()`'s `MAP` branch:
  `var inititals = lowerBounds.Add(upperBounds).Divide(2d);` (note the typo: `inititals`, not
  `initials`).
- **What:** this local variable is computed (an elementwise midpoint of the prior bounds) but
  never referenced again anywhere in the method -- confirmed by inspection of the full method
  body. `DifferentialEvolution`'s own population initialization (inside `DE.Maximize()`) draws
  from its own `LatinHypercube`-seeded population, not from this vector. Dead code with no
  observable effect (the `Vector.Add`/`Divide` extension calls have no side effects and cannot
  throw for the fixed-length arrays involved here).
- **Port handling:** omitted -- `mcmc_sampler.hpp`'s `initialize_chains()` has a NOTE comment at
  the equivalent location explaining the omission (this port has no `Vector::divide`, since no
  other ported call site needs one; adding it solely to reproduce dead code was judged not
  worthwhile).
- **Status:** RESOLVED. The `inititals` line is gone at `2a0357a` (present at `a2c4dbf` line 387,
  absent from the whole file at the new pin); the only `initials` identifier left is the
  `ParameterSet[]` the method actually returns. No port change was needed, since the C++ never
  reproduced the dead line.
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** delete the unused
  `inititals` line.

## CONSISTENCY — Gibbs's conjugate Normal-posterior-mean formula has a `mu0 / 2` term instead of the textbook `mu0 / sigma0^2` (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Sampling/MCMC/Test_Gibbs.cs`, `Test_Gibbs_NormalDist_RStan`'s local
  `proposal` closure: `double mun = (n * mu + mu0 / 2) / (n + 1 / (sigma0 * sigma0));` (the test's
  own inline proposal function -- there is no shared library implementation of this conjugate
  update; `Gibbs.cs` itself only calls whatever `Proposal` delegate the caller supplies).
- **What:** the standard closed-form posterior mean for a Normal likelihood with known variance
  `sigma^2` and a Normal(`mu0`, `sigma0^2`) conjugate prior on the mean is `mu_n = (n * xbar /
  sigma^2 + mu0 / sigma0^2) / (n / sigma^2 + 1 / sigma0^2)`. The test's formula instead computes
  `mun = (n * mu + mu0 / 2) / (n + 1 / sigma0^2)` -- it (a) omits the `/ sigma^2` scaling on the
  `n * xbar` numerator term and the `n` denominator term entirely (the companion `sigma2` line
  just above it DOES correctly compute the analogous posterior-variance formula, `sigma2 =
  x[1]^2 / (n + x[1]^2 / sigma0^2)`, so the omission is specific to the mean formula, not a
  wholesale simplification), and (b) uses `mu0 / 2` where the textbook formula has `mu0 /
  sigma0^2`. Because the test's own data uses `mu0 = 0`, term (b) vanishes identically regardless
  of which coefficient multiplies it, and because `sigma0 = 5e5` is enormous, `1 / sigma0^2 ~
  4e-12` is negligible next to `n = 48` in the denominator either way -- so the test's rstan
  comparison (which only checks the RESULTING posterior summary statistics to 5% tolerance, not
  the formula's algebraic form) cannot distinguish this transcription from the textbook one; both
  degenerate to `mun ~ mu` (the sample mean) in this near-noninformative-prior limit. This is
  therefore unverified/unverifiable from the test alone whether it's a genuine library bug or an
  intentional simplification for this specific (`mu0 = 0`) worked example -- flagged as
  CONSISTENCY rather than BUG for that reason.
- **Status:** RESOLVED. Numerics 2a0357a (v2.1.4) reworked `Test_Gibbs_NormalDist_RStan`'s
  conjugate math into two extracted, independently-tested helpers -- `ConditionalMeanParameters`
  (the textbook `posteriorVariance = 1 / (n / likelihoodVariance + 1 / priorVariance);
  posteriorMean = posteriorVariance * (n * sampleMean / likelihoodVariance + priorMean /
  priorVariance);`, i.e. the correct `mu0 / sigma0^2` term) and `ConditionalVarianceParameters`
  (algebraically unchanged from the old inline formula) -- plus a new
  `Test_ConditionalParameters_InformativePrior` unit test that exercises both helpers directly
  against a nonzero-`mu0` case, closing the "unverified/unverifiable" gap above. The rework also
  splits the single `muPrior`/`sigmaPrior` pair into a `muInitializationPrior`/
  `sigmaInitializationPrior` pair (seeds sampler feasibility bounds only) and a separate
  `conditionalMean`/`conditionalVariance` pair (default-constructed, mutated by the proposal
  closure only, no longer aliased with the initialization priors). Ported in the upstream-sync
  Task 10: `model_registry.hpp`'s `"normal_conjugate_gibbs"` proposal closure now
  computes `posterior_variance`/`posterior_mean` with the corrected formula (and the matching
  `mu_initialization_prior`/`sigma_initialization_prior` vs. `conditional_mean`/
  `conditional_variance` split), and the `gibbs.json` fixture's curated `chain_value` digests
  (draws 0-4, 99, plus `map_value`/`map_fitness`) are re-pinned and reproduced against the real
  C# library at `rel: 1e-12`.
- **Evidence:** `tools/verify_oracles.py` confirms the re-pinned `gibbs.json` digests reproduce
  against the real C# library post-fix.
- **Port handling (historical, pre-v2.1.4):** transcribed verbatim into `model_registry.hpp`'s
  `"normal_conjugate_gibbs"` proposal closure (`double mun = (n * mu + mu0 / 2.0) / (n + 1.0 /
  (sigma0 * sigma0));`) -- this was the oracle-governing rule at the time (C# source, including
  this specific test's inline formula, governs over what a textbook derivation "should" say), and
  the pre-fix `gibbs.json` fixture's curated `chain_value` digests reproduced this exact formula
  bit-for-bit against the real C# library.
- **Originally suggested action (this is exactly what v2.1.4 did):** re-derive/re-verify the
  formula against the textbook conjugate-Normal update and add regression coverage for a
  nonzero-`mu0` case.

## CONSISTENCY — NUTS's step-size heuristic bypasses a caller-supplied custom `GradientFunction` (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Sampling/MCMC/NUTS.cs`, `LeapfrogInPlace` (called only by
  `FindReasonableEpsilon`/`TrySingleStepLogAcceptance`, i.e. the Hoffman & Gelman 2014 Algorithm 4
  step-size-search heuristic run once at chain initialization and again after every mass-matrix
  adaptation-window update).
- **What:** every ACTUAL trajectory step in `BuildTree` goes through `Leapfrog`, which correctly
  calls `GradientFunction(...)` -- the (possibly caller-supplied) gradient delegate stored on the
  instance. `LeapfrogInPlace`, used only by the step-size-search heuristic, instead calls
  `NumericalDerivative.Gradient(...)` DIRECTLY, hardcoding a finite-difference gradient regardless
  of what `gradientFunction` the constructor was given. A caller who supplies an exact analytic
  `gradientFunction` (to avoid finite-difference cost/noise entirely) still gets a
  finite-difference-based initial step size and every post-adaptation-window re-tuned step size --
  only the trajectory itself uses their analytic gradient. This does not affect correctness of the
  sampled trajectory (the step size is just a tuning heuristic, not part of the target
  distribution), but it is a surprising, easy-to-miss asymmetry between two call sites that both
  claim to leapfrog-integrate "the" gradient.
- **Status:** RESOLVED. Numerics 2a0357a (v2.1.4) changed both `LeapfrogInPlace` half-step
  momentum updates from `NumericalDerivative.Gradient((y) => SafeLogLikelihood(y), theta,
  _lowerBounds, _upperBounds)` to `GradientFunction(theta).Array`, so the step-size heuristic now
  honors whatever `GradientFunction` the caller supplied, matching `Leapfrog`/`BuildTree`. Because
  the DEFAULT `GradientFunction` (no custom gradient supplied at construction) performs EXACTLY
  the same bound-aware finite-difference computation the old hardcoded call did, every seeded NUTS
  fixture (all of which use the default gradient) reproduces bit-for-bit unchanged by this fix.
  Ported in the upstream-sync Task 10: both `leapfrog_in_place()` half-step momentum updates in
  `nuts.hpp` now call `gradient_function_(theta).to_array()` instead of `diff::gradient(...)`
  directly; the existing seeded `nuts.json`/`hmc.json` fixtures pass unchanged post-port,
  confirming stream stability.
- **Evidence:** direct code reading; `ctest`/`verify_oracles.py` confirm every seeded NUTS/HMC
  fixture reproduces bit-for-bit unchanged post-fix.
- **Port handling (historical, pre-v2.1.4):** mirrored faithfully -- both C++ call sites
  reproduced the C# asymmetry exactly (`leapfrog_in_place()` called `diff::gradient(...)`
  directly, `leapfrog()` called `gradient_function_(...)`).
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** route `LeapfrogInPlace`
  through `GradientFunction` too, so a custom gradient is honored everywhere the class claims to
  use "the" gradient function.

## BUG — CS0104 ambiguous `YeoJohnsonLink` blocks the entire `RMC.BestFit` assembly from compiling (RESOLVED in RMC.BestFit v2.0.0 / c2e6192)

- **Where:** `Analyses/Univariate/Bulletin17CAnalysis.cs` (~lines 2132, 2144).
- **What:** both `RMC.BestFit.Models.LinkFunctions.YeoJohnsonLink` and `Numerics.Functions.
  YeoJohnsonLink` are `using`-imported in this file; two unqualified `new YeoJohnsonLink(...)`
  constructor calls are therefore ambiguous (CS0104), which fails compilation of the entire
  `RMC.BestFit` assembly, not just this file.
- **Evidence:** `tools/oracle_emitter` (Task T12) had to work around this to compile any real C#
  estimator code at all.
- **Port handling:** the oracle emitter subset-compiles only `Estimation/**` + `Models/**`
  (excluding `Analyses/**` -- the two GMM/Bulletin17C files -- and GMM) against the clean
  Numerics build, so the ambiguous file is never compiled. This is a build workaround, not a port
  of the file; `Bulletin17CAnalysis`/GMM remain unported (severed, tracked separately).
- **Suggested C# fix:** fully-qualify one of the two `new YeoJohnsonLink(...)` calls (e.g. `new
  Numerics.Functions.YeoJohnsonLink(...)` or `new RMC.BestFit.Models.LinkFunctions.
  YeoJohnsonLink(...)`), or remove one of the two conflicting `using` directives from the file.
- **A11 update (Analyses unlock):** the emitter now DOES compile the minimal Analyses closure
  (`Analyses/{Univariate,Support,DistributionFitting}/**`) so the real C# UnivariateAnalysis /
  FittingAnalysis / Bulletin17CAnalysis drive the tightened `fixtures/analyses/*.json` oracles. To
  clear this CS0104 without touching `upstream/`, the emitter compiles a LOCAL patched copy,
  `tools/oracle_emitter/patched/Bulletin17CAnalysis.cs` -- byte-for-byte identical to the upstream
  file except the two ACTIVE `new YeoJohnsonLink(` sites (upstream lines 2132, 2144) are qualified
  to `new RMC.BestFit.Models.LinkFunctions.YeoJohnsonLink(`; the commented-out scale-link site at
  ~2137 is left as-is. The surrounding B17C link-builder context (`LogLink` scale, an
  `ILinkFunction[]` over the model's `LinkFunctions` namespace) confirms the C# author intended the
  `RMC.BestFit` type. The csproj `<Compile Remove>`s the upstream original for that one file so it
  is not compiled twice, and also `<Compile Remove>`s `Analyses/Support/BatchAnalysisRunner.cs`
  (references `CoincidentFrequencyAnalysis`, a Bivariate orchestrator outside the minimal closure --
  CS0246; not on any dumped-oracle path). In the C++ port there is no ambiguity because the
  LinkedMVN / pivot-bootstrap path that constructs `YeoJohnsonLink` is not ported (deferred to a
  later phase), so the port never sees the two conflicting namespaces at one call site.
- **Status:** RESOLVED. BestFit v2.0.0 deleted its duplicate
  `src/RMC.BestFit/Models/LinkFunctions/YeoJohnsonLink.cs` outright (no file of that name exists
  anywhere in the repo at `c2e6192`) and routes `BestFitLinkFunctionFactory`'s YeoJohnson case to
  the Numerics link, which removes the ambiguity at the source. The upstream-sync Task 0 deleted
  the emitter's local patched copy and the two `<Compile Remove>` / `<Compile Include>` lines, so
  the emitter now compiles the real upstream `Bulletin17CAnalysis.cs`; Task 17 deleted the C++
  `models/link_functions/yeo_johnson_link.hpp` and re-pointed the factory, after confirming the
  two `DLink` implementations agree.

## BUG (latent, untested) — UnivariateDistribution's Jeffreys-scale prior indexes `GetParameters[1]`, which throws for single-parameter families (RESOLVED in RMC.BestFit v2.0.0 / c2e6192)

- **Where:** `Numerics/Distributions/Univariate/Base/UnivariateDistribution.cs`,
  `Prior_LogLikelihood` (~1822-1843), under `UseJeffreysRuleForScale`.
- **What:** the scale-parameter lookup is `GetParameters[1]` for every family except Gamma/Weibull
  (which use index 0). For a genuine one-parameter family -- Poisson, Bernoulli, Geometric,
  Deterministic -- there is no `GetParameters[1]`; indexing it would throw
  `IndexOutOfRangeException`. This path is only reached if a MAP or Bayesian estimation is
  actually run against a one-parameter model with the (default-true) `UseJeffreysRuleForScale`
  flag set -- untested upstream (no `Test_UnivariateDistribution.cs` case constructs MAP/Bayesian
  against a 1-parameter family) and flood-frequency-irrelevant in practice (GEV/LP3/etc. all have
  two or more parameters).
- **Evidence:** static inspection during Tasks T12/T13 while porting the Jeffreys 1/scale prior
  (`UnivariateDistributionModel::prior_log_likelihood`, `core/include/corehydro/models/
  univariate_distribution_model.hpp`); not independently reproduced against the real C# library
  (no fixture exercises a 1-parameter family under MAP/Bayesian).
- **Status:** RESOLVED, and the divergence is retired. BestFit v2.0.0 replaced the raw index with
  bounds-checked `TryGetJeffreysScaleParameter` overloads on `UnivariateDistributionModelBase`
  (scale index 0 for Gamma/Weibull, 1 otherwise), so both the prior-likelihood and the
  validation call sites now read `if (UseJeffreysRuleForScale && TryGetJeffreysScaleParameter(
  model, out ...))` and single-parameter components are skipped instead of throwing. The same
  pattern was applied to MixtureModel and CompetingRisksModel, plus a `-Inf` short-circuit.
  Ported in the upstream-sync Task 14; the C++ now carries upstream's helper rather than its own
  inline guard.
- **Port handling (historical):** intentional divergence -- the C++ port's
  `scale_parameter_index()` returned 1 for these families same as C#, but the caller guarded
  `scale_index < p.size()` and silently skipped the Jeffreys term.
- **Originally suggested C# fix (this is what v2.0.0 did):** guard the index and skip the Jeffreys
  scale term for one-parameter families.

## BUG — MixtureModel.Clone() strips the cloned Mixture's zero-inflation while the cloned model still reports IsZeroInflated (RESOLVED in RMC.BestFit v2.0.0 / c2e6192)

- **Where:** `RMC.BestFit/Models/UnivariateDistribution/MixtureModel.cs`, `Clone()` (~line 1276),
  interacting with the `Mixture` property setter (~line 242), the `IsZeroInflated` property
  setter (~line 285), and the `(DataFrame, Mixture)` constructor (~line 53).
- **What:** `Clone()` builds `new MixtureModel(DataFrame, Mixture!) { _isZeroInflated =
  IsZeroInflated, ... }`. The constructor runs `Mixture = (Mixture)distribution.Clone();` through
  the `Mixture` property setter. Numerics' `Mixture.Clone()` itself correctly copies
  `IsZeroInflated`/`ZeroWeight` (Mixture.cs ~line 1019), but at that moment the fresh model's
  `_isZeroInflated` field still holds its default `false`, so the setter immediately overwrites
  the cloned distribution with `IsZeroInflated = false; ZeroWeight = 0.0;`. The object
  initializer then writes `_isZeroInflated = IsZeroInflated` DIRECTLY to the private field --
  bypassing the `IsZeroInflated` property setter, the only code that would re-sync the
  distribution. End state when cloning a zero-inflated model: the cloned MODEL reports
  `IsZeroInflated == true` while its underlying `Mixture` has `IsZeroInflated == false,
  ZeroWeight == 0` -- the clone's likelihood/PDF/CDF surface silently loses the zero-inflated
  mass while claiming to have it.
- **Evidence:** static inspection of the three members during Task M10 (the setter/field-write
  ordering is unambiguous from the source); no upstream `MixtureModelTests` method clones a
  zero-inflated model and asserts the distribution's state, so no C# test observes it.
- **Status:** RESOLVED, and the divergence is retired. BestFit v2.0.0's `MixtureModel.Clone()`
  now writes the zero-inflation back onto the cloned distribution after the object initializer
  (`result.Mixture!.IsZeroInflated = result._isZeroInflated;` followed by the matching
  `ZeroWeight` assignment), which is behaviorally what the C++ divergence already did. Ported in
  the upstream-sync Task 14, so the C++ re-sync is now a faithful mirror rather than a divergence.
- **Port handling (historical):** intentional divergence -- `mixture_model.hpp`'s `clone()`
  re-synced the cloned mixture's zero-inflation state from the original after the field writes.
- **Originally suggested C# fix (this is what v2.0.0 did):** re-apply the zero-inflation to the
  cloned `Mixture` after the initializer.

## ROBUSTNESS — DataFrame.ProcessThresholdSeries is destructive and not idempotent when explicit points exactly cover a threshold window (RESOLVED in BestFit v2.0.0 / c2e6192)

- **Where:** `RMC.BestFit/Models/DataFrame/DataFrame.cs`, `ProcessThresholdSeries()` (~line 618).
- **What:** the method reads the STORED `thresholdData.NumberAbove`, computes `nBelow = Duration
  - nAbove - (explicit interval/uncertain/exact points inside the window)`, then writes both
  counts back: `NumberAbove = nBelow == 0 ? 0 : nAbove; NumberBelow = Math.Max(0, nBelow);`.
  Because the recomputation consumes its own previous output, the zeroing branch is destructive:
  when the explicit points exactly account for `Duration - NumberAbove` (first run: `nBelow ==
  0`, so `NumberAbove` is zeroed and `NumberBelow` set to 0), a SECOND run starts from `nAbove =
  0` and computes `nBelow = Duration - overlaps` -- flipping `NumberBelow` from 0 to the original
  `NumberAbove`, i.e. years the first pass classified as above-threshold are re-counted as
  censored-below years. Upstream this method re-runs constantly: every series
  `CollectionChanged`/item `PropertyChanged` event triggers it, and `CalculatePlottingPositions()`
  calls it unconditionally as its first step (~lines 1142-1144) -- so in the exact-coverage edge
  case any later mutation or a repeated plotting-positions call silently corrupts the threshold
  likelihood counts.
- **Status:** RESOLVED. BestFit c2e6192 (v2.0.0) split `ThresholdData.NumberAbove` into a
  user-supplied `_sourceNumberAbove` (updated only by the public `NumberAbove` setter) and the
  effective `_numberAbove`/`_numberBelow` (updated only by the new `internal SetProcessedCounts`).
  `ProcessThresholdSeries` now reads `SourceNumberAbove` -- the STABLE original input -- on every
  pass instead of the possibly-already-reduced effective value, so repeated calls (or a call after
  further series mutation) always reproduce the correct effective state; the destructive
  one-way-ratchet described above cannot happen anymore. `Validate()` and `Clone()` were updated to
  key off `SourceNumberAbove` too (Clone: `NumberAbove = SourceNumberAbove` then a
  `SetProcessedCounts` replay of the effective state). Ported in the upstream-sync Task 12:
  `threshold_data.hpp` gained the matching `source_number_above()`/`set_processed_counts()` split;
  `data_frame.hpp`'s `process_threshold_series()` now reads `source_number_above()` and writes
  through `set_processed_counts()`. See `core/tests/test_data_types.cpp`'s
  `test_threshold_processed_counts_clone_preserves_source_and_effective`/
  `test_threshold_validate_uses_source_number_above` and `core/tests/test_data_frame.cpp`'s
  `test_data_frame_process_threshold_series_is_idempotent` (calls `process_threshold_series()`
  twice, then a third time after removing the overlapping data, and asserts the effective counts
  are exactly reproduced/restored each time) -- new C++-only regression coverage, since
  `ProcessThresholdSeries` has no existing C# unit test for this sequence and no ported fixture
  exercises it either (the same "not tested by the current corpus" gap this finding originally
  noted).
- **Evidence:** static inspection of the read-modify-write during Task M4 (report concern #1);
  no upstream test hits the exact-coverage-then-rerun sequence, and no ported fixture exercises
  the edge either. Confirmed the fix's idempotency directly via the new C++-only ctests above and
  indirectly via the full oracle gate reaching 0 failures for the first time this sync (Task 12).
- **Port handling (historical, pre-v2.0.0):** mirrored faithfully -- `data_frame.hpp`'s
  `process_threshold_series()` was the same read-modify-write, and `calculate_plotting_positions()`
  called it first exactly like the C#. The C++ replaced the INPC auto-trigger with the documented
  explicit "call once after mutations" cadence (see the file-header invalidation strategy), which
  matched the C# once-per-mutation event cadence.
- **Originally suggested C# fix (this is essentially what v2.0.0 did):** make the pass idempotent
  by recomputing from immutable inputs -- retain the originally supplied `NumberAbove` and derive
  both published counts from it on every pass; add a regression test that processes a fully
  covered threshold window twice and asserts stable counts.

## BUG — GammaDistribution.PartialKp's near-zero-skew branch returns the frequency factor, not its derivative (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Distributions/Univariate/GammaDistribution.cs`, `PartialKp(skewness,
  probability)` @ a2c4dbf (~line 698).
- **What:** `PartialKp` is documented as "the partial derivative of the frequency factor Kp with
  respect to skew," and every other branch returns exactly that (the `|skew| <= 2` branch returns
  the term-by-term derivative of the Cornish-Fisher polynomial; the `|skew| > 2` branch takes a
  finite-difference `NumericalDerivative.Derivative` of `FrequencyFactorKp`). The near-zero guard
  `if (absC < 0.0001d) return Normal.StandardZ(probability);` instead returns the frequency factor
  Kp's OWN value at zero skew (the standard normal quantile), not its derivative. This looks copied
  verbatim from `FrequencyFactorKp`'s legitimate near-zero branch (where returning `StandardZ` IS
  correct, because Kp at zero skew is the normal quantile) without adjusting for the fact that
  `PartialKp` must return a derivative. The correct small-skew limit is the first non-vanishing
  derivative term, `(U^2 - 1) / 6` with `U = StandardZ(probability)` -- so there is a discontinuity
  as `|skew|` crosses `1e-4` (jumping from `~(U^2-1)/6` to `StandardZ(p)`).
- **Evidence:** static inspection of the method during the B4 moment-machinery port; the two closest
  fixture/ctest cases keep `|skew|` well above `1e-4`, so no oracle exercises the branch (the B4
  `partial_kp` vs finite-difference-of-`FrequencyFactorKp` check runs at skew 0.2, on the CF branch).
- **Status:** RESOLVED. Numerics v2.1.4's `PartialKp` near-zero branch now reads
  `double z = Normal.StandardZ(probability); return (z * z - 1.0d) / 6.0d;` under the comment
  "Use the Cornish-Fisher derivative limit at zero skew", which removes the discontinuity.
  Ported in the upstream-sync Task 5, with a `partial_kp` dispatch added across all four runners
  so the branch is oracle-covered.
- **Port handling (historical):** mirrored faithfully -- `partial_kp` returned
  `Normal::standard_z(probability)` in the `abs_c < 1e-4` branch, identical to the old C#.
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** return `(U*U - 1d) / 6d` in
  the near-zero branch.

## COSMETIC — BrentSearch.Bracket declares an expansion factor `k` it never applies (RESOLVED in Numerics v2.1.4 / 2a0357a)

- **Where:** `Numerics/Mathematics/Optimization/Local/BrentSearch.cs`, `Bracket(double s = 1E-2,
  double k = 2d)` @ a2c4dbf (~line 162).
- **What:** the second parameter `k` (default `2.0`) is never referenced in the method body. The
  bracketing loop advances by a CONSTANT step (`c = b + s;` with `s` fixed), so the interval grows
  linearly, not geometrically. A reader (or caller) supplying `k` expecting a golden-section-style
  geometric expansion (the usual `s *= k` growth in a downhill bracketing search) gets no effect --
  the parameter is dead.
- **Evidence:** direct source reading during the B6 Powell/MLSL optimizer port (Powell's line
  minimization is the only caller, and it always uses the defaults).
- **Status:** RESOLVED. Numerics v2.1.4's `Bracket` applies `s *= k;` at the end of each expansion
  iteration, and the method also gained input validation (finite nonzero `s`, finite `k > 1`),
  non-finite objective and coordinate guards that set `OptimizationStatus.Failure`, and a bounded
  iteration count. Ported in the upstream-sync Task 11; the `(void)k` cast is gone.
- **Port handling (historical):** mirrored faithfully -- `bracket(double s = 1e-2, double k = 2.0)`
  cast `(void)k;` with a comment noting the parameter was declared-but-unused upstream.
- **Originally suggested C# fix (this is exactly what v2.1.4 did):** apply `k` inside the loop for
  geometric expansion.

## CONSISTENCY — GMM's iterative loop overshoots GMMIterations by one on exhaustion, and ConvergedWithinTolerance is off-by-one at the boundary (RESOLVED in BestFit v2.0.0 / 5e1877f)

- **Where:** `RMC.BestFit/Estimation/GeneralizedMethodOfMoments.cs`, `EstimateIterative` @ fc28c0c
  (~line 2253, `for (GMMIterations = 2; GMMIterations <= MaxGMMIterations; GMMIterations++)`) and the
  `ConvergedWithinTolerance` property (~line 502).
- **What:** two related off-by-ones at the loop boundary.
  1. On natural exhaustion (never converging), the `for` loop's final post-increment leaves
     `GMMIterations == MaxGMMIterations + 1` (e.g. `101` for the default 100-iteration cap), so the
     publicly reported iteration count overshoots the actual number of iterations run by one.
  2. `ConvergedWithinTolerance => IsEstimated && GMMIterations < MaxGMMIterations` uses a strict `<`.
     A fit that converges on exactly the last permitted iteration (`GMMIterations ==
     MaxGMMIterations`) therefore reports `ConvergedWithinTolerance == false` even though it did
     converge within the budget.
- **Status:** RESOLVED. BestFit v2.0.0 (5e1877f..b43943c, Task T13) fixed both: `EstimateIterative`
  now clamps `GMMIterations` back to `MaxGMMIterations` on exhaustion (`if (GMMIterations >
  MaxGMMIterations) GMMIterations = MaxGMMIterations;`) instead of leaving the for-loop's final
  overshoot, and `ConvergedWithinTolerance` now reads a tracked `_convergedWithinTolerance` flag
  set ONLY on the loop's own tolerance-convergence branch (true even on the LAST permitted
  iteration, fixing the strict-`<` boundary) rather than deriving from the iteration count at all;
  the flag stays false for `OneStep`/`TwoStep` (no comparison pass) and for exhaustion. Ported in
  the upstream-sync Task 13: `generalized_method_of_moments.hpp`'s `estimate_iterative` mirrors the
  clamp verbatim and `converged_within_tolerance_` mirrors the tracked flag, retiring the
  preserve-the-bug note below. New coverage: `fixtures/estimation/gmm_bulletin17c_smoke.json` gained
  `gmm_iterations`/`converged_within_tolerance`/`optimizer_fallback_count` assertions on the
  existing case plus two new cases (`lp3_exact_one_step_bfgs`, OneStep's no-comparison-pass path;
  `lp3_exact_iterative_max_gmm_iterations_1`, an Iterative run capped before the loop's first
  comparison pass), all reproduced against the real C# library. The exhaustion-clamp path itself
  (an Iterative run that runs the loop repeatedly and never converges) stays untested by any
  fixture -- no B17C data drives that many passes without converging -- but is covered by
  `core/tests/test_gmm.cpp` PART 3's `test_iteration_exhaustion_clamps_count_and_reports_not_converged`
  via a `max_gmm_iterations = 1` toy problem (the loop body never runs at all, so this exercises the
  "no comparison pass" side of the fix rather than a many-iteration exhaustion; both sides clamp to
  the same `GMMIterations == MaxGMMIterations` outcome by construction).
- **Evidence:** static inspection during the B8 GMM port; not reachable through the ported B17C
  fixtures pre-T13 (a just-identified B17C GMM converges in one or two iterations, far below the
  cap), so no oracle exercised the exhaustion boundary before this fix landed.
- **Port handling (historical, pre-T13):** mirrored faithfully -- `generalized_method_of_moments.hpp`'s
  `estimate_iterative` preserved the `gmm_iterations_ == max_gmm_iterations_ + 1` exhaustion value
  and the same strict-`<` `converged_within_tolerance()` boundary, documented at the call site.
  Retired in Task 13 -- the fixed loop no longer overshoots or under-reports on this port's
  fixture-exercised inputs.
- **Originally suggested C# fix (this is exactly what v2.0.0 did):** report the true count on
  exhaustion (e.g. clamp `GMMIterations` to `MaxGMMIterations`, or count iterations actually
  executed) and relax `ConvergedWithinTolerance` to `GMMIterations <= MaxGMMIterations` so a
  last-iteration convergence is reported honestly; add a regression test that exhausts the
  iteration budget and asserts both the reported count and the flag.

## CONSISTENCY — BivariateDistribution model-level PseudoLikelihood MLE returns `Estimate()==false` because plotting positions are never calculated on the shared build path (RESOLVED in RMC.BestFit v2.0.0 / c2e6192)

- **Where:** `RMC.BestFit/Models/BivariateDistribution.cs`, `SetSampleData` /
  `DataLogLikelihood` (the PseudoLikelihood branch) @ fc28c0c, reached via
  `RMC.BestFit/Estimation/MaximumLikelihood.cs`, `Estimate()`.
- **What:** under `CopulaEstimationMethod.PseudoLikelihood`, the copula log-density is evaluated at
  the marginal **plotting positions** the copula's own bounds require to sit strictly inside
  `(0, 1)`. Those positions come from each marginal `ExactData.PlottingPosition` /
  `PlottingPositionComplement`, which stay at their construction default (position `0` -> complement
  `1.0`, i.e. NOT strictly interior to `(0, 1)`) until `DataFrame.CalculatePlottingPositions()` is
  run. The model-level MLE path built by the shared spec-builder (and by the oracle emitter) never
  triggers `CalculatePlottingPositions()`, so the copula parameter bounds are degenerate and
  `MaximumLikelihood.Estimate()` returns `false` (the fit is rejected) rather than a valid theta.
  The Normal/IFM and StudentT/IFM cases do NOT depend on plotting positions and fit cleanly.
- **Evidence:** reproduced against the real C# via the P4 oracle emitter -- a PseudoLikelihood
  bivariate construction returns `Estimate()==false`, so no valid oracle can be dumped for it (the
  PseudoLikelihood case was scoped out of P4 for exactly this reason). The Normal/IFM and
  StudentT/IFM cases dump valid fits and ARE oracle-verified (`fixtures/estimation/
  bivariate_smoke.json`).
- **Port handling:** the C++ `bivariate_distribution.hpp` PseudoLikelihood estimate path returns a
  degenerate ~0.5 theta WITHOUT gating on the same strict-interior validation (its `validate()`
  guard exists but the estimate path does not reject on it the way the C# lifecycle does), so the
  two languages diverge on this one method+config. No fixture exercises it; the divergence is
  documented at the call site and in the P4 report.
- **Status:** RESOLVED, and the divergence is retired. BestFit v2.0.0's PseudoLikelihood path
  validates the pseudo observations on the open interval (0,1) and auto-runs
  `dataFrameX.CalculatePlottingPositions()` / `dataFrameY.CalculatePlottingPositions()` when they
  are missing, caching the result against each marginal's `PlottingPositionVersion`. Ported in the
  upstream-sync Task 15 (`GetEligibleExactData` / `AddPairedSampleData(AndValidate)` plus a
  per-marginal validated cache keyed on `plotting_position_version()`), and a real
  PseudoLikelihood oracle case was added to `fixtures/estimation/bivariate_smoke.json`. Task 15
  also fixed a genuine undefined-behavior risk it surfaced (`SetSampleData` dereferencing a
  possibly-empty `DataFrame` optional) and recorded one remaining gap: a bare
  `ExactData.PlottingPosition` mutation bumps the C# cache version but not this port's, because
  the port has no XML/INPC layer to raise it.
- **Originally suggested C# fix (this is what v2.0.0 did):** have the shared build path call
  `CalculatePlottingPositions()` before a PseudoLikelihood fit.

## CONSISTENCY — TimeSeries DateTime index vs. the port's integer index is fit-invariant (models never do calendar arithmetic) (RESOLVED port-side in P6 / v0.13.0)

- **Where:** `Numerics/Data/TimeSeries/TimeSeries.cs` (the `DateTime`-keyed index) vs. the thin
  C++ adapter `core/include/corehydro/numerics/data/time_series/time_series.hpp` (a `long` day-count
  index) @ a2c4dbf.
- **What:** the C# `TimeSeries` is keyed by `DateTime`; the ported adapter uses an integer index.
  Every TimeSeries/RatingCurve model consumer touches the index only as a sequence position or an
  inner-join key -- never as calendar arithmetic -- so the two representations are fit-invariant.
- **Evidence:** the P4 oracle emitter builds every series in a case from one fixed epoch
  (`2000-01-01`) with the same interval, and every AR/MA/ARIMA/ARIMAX/RatingCurve fixture
  reproduces 0-failed; relative alignment (rating-curve stage<->discharge, ARIMAX covariate lags)
  is preserved exactly by the integer adapter.
- **Port handling:** the adapter deliberately drops calendar semantics; documented in
  `time_series.hpp` and the P4 report.
- **Suggested action:** none -- a one-line note that absolute `start_index` is not modeled and no
  reachable model path depends on it.
- **Status:** RESOLVED port-side. P6 (v0.13.0) replaced the thin integer-indexed adapter in place
  with the full `TimeSeries` port, whose ordinate index is a real `DateTime`
  (`numerics/data/time_series/support/date_time.hpp`, a corehydro addition reproducing the
  `System.DateTime` subset the container uses). The index swap was verified value-neutral on its
  own before any container method landed: the whole ctest suite, the fixture runner, testthat,
  pytest, and the dotnet oracle gate all reproduced unchanged (see `time_series.hpp`'s header).
  The fit-invariance claim this entry made was correct for as long as the adapter stood; the
  representational divergence it described no longer exists.

## CONSISTENCY (C++ port divergence, D6) — PriorInfluenceDiagnostics collapses the two Normal parameter priors because the ported C++ ModelParameter names are empty (RESOLVED port-side in Phase 10)

- **Where:** `RMC.BestFit/Diagnostics/PriorInfluenceDiagnostics.cs`,
  `ComputeFromPosterior` (the `Dictionary<string, List<double>>` keyed by `PriorComponent.Name`)
  @ fc28c0c; the divergence originates in the ported
  `core/include/corehydro/models/univariate_distribution/univariate_distribution_model.hpp` (~130,
  the standing Phase-4 decision to NOT port `Distribution.ParameterNames`, so `ModelParameter`
  `owner_name()`/`name()` stay empty) and surfaces through the faithful C++ port
  `core/include/corehydro/diagnostics/prior_influence_diagnostics.hpp`.
- **What:** `PriorInfluenceDiagnostics` collects prior-component log-likelihoods into a dictionary
  keyed by each component's NAME. For a Normal `UnivariateDistributionModel` the two parameter
  priors are labelled `"Parameter Prior: " + paramName`. In C# `paramName` resolves to the
  distinct `OwnerName`s `Mean` / `Std Dev` (set from `Distribution.ParameterNames`), so the two are
  kept as separate components; the C++ port leaves both names empty, so both become the single key
  `"Parameter Prior: "` and COLLAPSE into one component. Consequence on the D6 diagnostics oracle
  (`fixtures/analyses/diagnostics_smoke.json`, case `normal_bayesian_diagnostics_short`):
  `prior_influence_count` = 2 (C++) vs 3 (C#); `total_prior_log_likelihood` = -21.7357 (C++) vs
  -34.7465 (C#, which sums three component means instead of two); `prior_to_data_ratio` = 0.0967
  (C++) vs 0.1461 (C#). Every other diagnostics quantity (all LeverageDiagnostics + all
  InfluenceDiagnostics/PSIS + `total_data_log_likelihood` + `mean_prior_precision_share`)
  reproduces C#-vs-C++ exactly, confirming the seeded DEMCzs posterior itself is bit-identical
  (~1e-12) and the divergence is purely the name-keyed dedup, not a stream divergence.
- **Evidence:** the D6 oracle emitter compiles the REAL `PriorInfluenceDiagnostics` in place and
  dumps 3 / -34.74652951884822 / 0.14606421711186202 for the same seeded fit where the ported C++
  core (ctest/R/Python) produces 2 / -21.7357379171011 / 0.096657190986542.
- **Port handling:** the three affected assertions keep the C++ contract value (so the shipped
  ctest/R/Python harnesses stay green against the ported core) and carry `"oracle_skip": true`, so
  `tools/verify_oracles.py` SKIPS them (same bucket as the GEV std-err skips) rather than failing
  on a divergence it cannot reproduce. The divergence is documented, NOT absorbed into a widened
  tolerance. Fixing it in C++ would require populating `ModelParameter` names from
  `Distribution.ParameterNames` -- an oracle-locked Phase-4 core change, out of D6's emitter+fixture
  scope.
- **Suggested action (D7 / follow-up):** decide whether the C++ core should port
  `Distribution.ParameterNames` onto the distribution base so `ModelParameter` names are populated
  (removing the collapse and matching C# on all three quantities), or whether the divergence stays a
  documented intentional deviation. Either way the three `oracle_skip` assertions can be un-skipped
  and tightened once the C++ names match C#.
- **Status:** RESOLVED port-side. Phase 10 (X1) took the first branch of the suggested action:
  `Distribution.ParameterNames` was ported onto the univariate distribution base and every
  concrete distribution, and wired through the trend/`ModelParameter` owner-name path, so the two
  Normal parameter-prior components no longer collapse under the name-keyed dedup. The three
  affected assertions in `fixtures/analyses/diagnostics_smoke.json` were un-skipped and re-pinned
  to the C# oracle (`prior_influence_count` 3, `total_prior_log_likelihood` -34.74652951884832,
  `prior_to_data_ratio` 0.14606421711186263 at rel 1e-9), reproduced against the real
  `RMC.BestFit.PriorInfluenceDiagnostics` by the dotnet gate; the corpus skip count dropped from
  14 to 11 (the documented GEV standard-error set only). Verified 2026-08-28: no
  `"oracle_skip": true` assertion remains anywhere in `fixtures/`.

## SCOPE NOTE (D6, RESOLVED at D6 completion) — the seven per-family analysis oracles are now emitter-wired

- **Status:** RESOLVED. An earlier draft of this note recorded seven D5-authored LOOSE analysis
  smoke fixtures (`fixtures/analyses/{point_process,mixture,competing_risk,ar,ma,arima,arimax}_
  analysis_smoke.json`) hitting the emitter's `throw "unknown analysis target"` fall-through, so
  `tools/verify_oracles.py` reported `3972 reproduced, 7 failed, 14 skipped`. D6 completion wired all
  seven `BuildAndRunAnalysis` targets (mirroring the `UnivariateAnalysis` serial-drive shape) and
  added `Analyses/TimeSeries/**` to `OracleEmitter.csproj` for the four TimeSeries families (the
  three Univariate-family classes already compiled via the Phase-8 `Analyses/Univariate` glob). The
  corpus is now `4003 reproduced, 0 failed, 14 skipped`. The "7 failed" claim no longer stands; the
  residual C#-vs-C++ divergence on three of the seven is the FIDELITY (D6) chaotic-sensitivity
  finding in the open log (`docs/upstream-csharp-issues.md`) -- a chaotic short-chain artifact,
  not an unwired handler.

## BUG (C++ port only, no C# counterpart) — three classes computed their central moments with adaptive Gauss-Kronrod where C# uses the 1000-step trapezoidal overload (FIXED, ported)

- **Where:** `CentralMoments(1000)` is called at `Numerics/Distributions/Univariate/Mixture.cs` @
  2a0357a line 312, `CompetingRisks.cs` line 217, and `TruncatedDistribution.cs` lines 140, 171,
  185 and 199 (its four moment getters each fill `u` from the same call). The counterparts are
  `compute_moments()` in `core/include/corehydro/numerics/distributions/mixture.hpp`,
  `competing_risks.hpp` and `truncated_distribution.hpp`. Surfaced writing
  `fixtures/distributions/univariate/mixture.json`'s
  `moments_log_pdf_log_likelihood_and_seeded_draws` case, the first fixture to pin Mixture skewness
  or kurtosis at a tight tolerance; the review that followed found the other two call sites.
- **What:** C# passes the integer literal `1000`, which binds to the `int steps` overload on
  `UnivariateDistributionBase` — a 1000-bin `Stratify` midpoint/trapezoid sum over
  `[InverseCDF(1e-8), InverseCDF(1-1e-8)]`. All three C++ sites instead called adaptive
  Gauss-Kronrod at relative tolerance 1e-8, mirroring the *other* base overload,
  `CentralMoments(double tolerance)`. Mixture's own comment said so ("Mirrors C#
  CentralMoments(double tolerance = 1e-8) from the base class"), and TruncatedDistribution
  compounded it by integrating over `[min, max]` rather than the two 1e-8 quantiles. Both are
  legitimate quadratures of the same integrals and they agree to about six digits, so nothing
  downstream looked wrong — but they are not the same number.
- **Measured**, on the three-component Normal mixture `three_normal_pdf_cdf` uses
  (`0.3*N(10,2) + 0.2*N(20,1) + 0.5*N(30,5)`), C# first and the old Gauss-Kronrod port second:
  - mean `21.999999995011922` vs `21.999999428497716` (2.6e-8 relative)
  - sd `9.481575923790889` vs `9.4815600742053032` (1.7e-6 relative)
  - skewness `-0.00985460360386465` vs `-0.0098548559431184124` (2.5e-5 relative)
  - kurtosis `1.8641595464622873` vs `1.8641508578822412` (4.7e-6 relative)
- **Why no earlier fixture caught it:** the only Mixture moment oracles were `mean` (`22`, abs
  1e-4) and `sd` (`9.481561`, abs 1e-3) — both far too loose to separate the two quadratures. The
  skewness/kurtosis verbs had no oracle at all. CompetingRisks pinned all four moments at the exact
  C# 1000-step values but at `abs 1e-2`, and TruncatedDistribution pinned hand-rounded five-digit
  literals at `abs 1e-4`.
- **Status: FIXED and ported.** All three `compute_moments()` bodies now call
  `central_moments(1000)` — the already-ported trapezoidal overload on the C++ base class — and
  return the C# values to 1e-16 to 1e-11 relative. The four `agk::integrate` calls are gone from
  each file (with the now-unused `adaptive_gauss_kronrod.hpp` include). The fixtures were tightened
  in the same change: Mixture's new case pins all four moments at `rel 1e-10` (measured 9.7e-16 to
  6.1e-13), the thirteen CompetingRisks moment pins went from `abs 1e-2` to `rel 1e-9` (measured
  1.3e-16 to 4.1e-11, the loosest being the Weibull/Gamma skewness), and
  TruncatedDistribution's moments case was re-pinned from the rounded literals to the full-precision
  C# values at `abs`/`rel 1e-12` (measured 3.3e-16 to 2.0e-14). No `oracle_skip` was added and no
  tolerance was loosened anywhere.
- **Note for the pre-existing CentralMoments overload entry in the open log** (the CONSISTENCY
  entry in `docs/upstream-csharp-issues.md` naming
  `TruncatedDistribution.cs` and `Mixture.cs`): its "Port handling" line said the C++ used adaptive
  Gauss-Kronrod and "reproduces the C# values to fixture tolerance." That was true only at the old
  loose tolerances. It no longer applies — the port now calls the same overload C# does. The
  upstream-facing half of that entry (the argument that two overloads distinguished only by
  `int` vs `double` are a foot-gun) still stands.

## FIDELITY (C++ port only) — five `Mode` implementations approximated the BrentSearch maximization C# performs (FIXED, ported)

- **Where:** C# builds `new BrentSearch(PDF, InverseCDF(0.001), InverseCDF(0.999))`, calls
  `Maximize()` and returns `BestParameterSet.Values[0]` in five classes @ 2a0357a:
  `EmpiricalDistribution.cs` line 280, `KernelDensity.cs` line 320, `Mixture.cs` line 337,
  `CompetingRisks.cs` line 241 and `TruncatedDistribution.cs` line 154. The counterparts are
  `mode()` in `empirical_distribution.hpp`, `kernel_density.hpp`, `mixture.hpp`,
  `competing_risks.hpp` and `truncated_distribution.hpp`. Surfaced writing the
  `palisades_moments_density_and_seeded_draws` and `gaussian_moments_density_and_seeded_draws`
  cases.
- **What:** Empirical and KernelDensity scanned a fixed 1000-point uniform grid over the same
  interval and returned the best node; Mixture and CompetingRisks ran a 200-iteration ternary
  search, which is only valid on a unimodal surface; TruncatedDistribution did not search at all,
  clamping the *base* distribution's mode into `[min, max]`. Each was a documented shortcut —
  `empirical_distribution.hpp` said so outright ("Use golden-section search on a fine grid for
  simplicity; this is not in the fixture").
- **Measured**, C# first, the old approximation second, and the same object driven through the
  port's own `BrentSearch::maximize()` third:

  | | C# | old port | port via BrentSearch |
  |---|---|---|---|
  | Empirical, 99-point Palisades at-risk record | `15198.510496728823` | `8669.7200000000012` | `15198.510496728823` (bit-exact) |
  | Mixture, three-Normal | `29.99999990517664` | `29.99999990850224` | `29.99999990517664` (bit-exact) |
  | KernelDensity, Gaussian over 48-point Tippecanoe | `13470.180323329207` | `13478.103364870844` | `13470.18032360422` (2.0e-11 relative) |
  | CompetingRisks, two Normals, min rule | `97.38075467325922` | `97.380754651639961` | `97.380754673299492` (4.4e-13 relative) |

  The Empirical gap is the large one because the surface is jagged: `Empirical.PDF` is a
  finite-difference derivative of a piecewise-linear CDF, so Brent stops at whichever local optimum
  its bracket encloses while a grid scan finds a different one. An earlier reading of this called
  the C# value "optimizer-defined" and argued neither answer was more correct. That argument is
  disproved by the third column: running the *port's own* Brent over the *same* objective and the
  *same* interval reproduces C# bit-for-bit on exactly the case used to make it. The value is
  defined by the algorithm, and the algorithm was already ported.
- **Status: FIXED and ported.** All five `mode()` bodies now construct
  `math::optimization::BrentSearch(pdf, inverse_cdf(0.001), inverse_cdf(0.999))`, call `maximize()`
  and return `best_parameter()`. `brent_search.hpp` includes only `tools.hpp`, so no include cycle
  is created. Empirical and KernelDensity keep their pre-existing `lo >= hi` guard: those classes
  validate lazily, so an invalid parameter set can leave the two quantiles crossed, and both the C#
  and the C++ `BrentSearch` constructor throw on `upper < lower`. No `*_invalid` fixture case
  reaches `mode()`. The fixtures gained the two assertions this entry used to justify dropping:
  Empirical `mode` at `rel 1e-15` (the agreement is bit-exact; the tolerance is nominal insurance
  against floating-point contraction on another platform) and KernelDensity `mode` at `rel 1e-9`.
  TruncatedDistribution's existing `mode` pin was re-pinned to the full-precision C# value and
  agrees to 1 ulp. No `oracle_skip`, no loosened tolerance.

## FIDELITY (C++ port only) — `GeneralizedPareto.GetParameterConstraints` was truncated to 2 of its 3 parameters, so every non-MLE caller read past the end (FIXED, port defect)

- **Where:** `Numerics/Distributions/Univariate/GeneralizedPareto.cs` lines 515-549
  (`GetParameterConstraints`) and line 570 (`MLE`); ported at
  `core/include/corehydro/numerics/distributions/generalized_pareto.hpp`.
- **What:** the C# fills `initialVals` / `lowerVals` / `upperVals` to `NumberOfParameters` = 3
  and returns all three. Only `MLE` wants the 2-parameter view, and it takes it at the call
  site: `new NelderMead(logLH, 2, Initials.Subset(1), Lowers.Subset(1), Uppers.Subset(1))`,
  because ξ is fixed at `min(sample)` and never optimized. The port folded that `.Subset(1)`
  INTO the constraint method, returning `{alpha, kappa}` triples of length 2. `MLE` was the
  only caller that wanted it that way.
- **Consequence:** eight other call sites index the arrays by
  `distribution->number_of_parameters()`, which is 3 for a GPD —
  `UnivariateDistributionModel::set_default_parameters` and `::set_trend_model`, the MCMC
  model registry, the copula IFM/MPL marginal pre-fit, the Mixture / CompetingRisks /
  PointProcess component seeds, and Bulletin17C. Every one of them read `initials[2]`,
  `lowers[2]` and `uppers[2]` off the end of a 16-byte allocation. The values landed in the
  model's default parameter values, its bounds, its `IsPositive` flag and its `Uniform` prior,
  so a GPD candidate inside `fit_distributions()` was seeded from adjacent heap memory.
- **Evidence:** an ASan build of the ctest suite reports
  `heap-buffer-overflow ... READ of size 8 ... univariate_distribution_model_trends.hpp:92`,
  `0 bytes after 16-byte region`, allocated at `generalized_pareto.hpp:298`, from three
  binaries: `test_univariate_distribution_model` (direct construction),
  `test_fitting_analysis` and `test_fixtures` (both through `FittingAnalysis::run`, which
  builds a model for each of the 14 candidates). No test FAILED, which is why it survived; the
  read is invisible to every assertion the suite makes.
- **Status: FIXED, and it was a port defect, not upstream behaviour.** In C# the same index is
  in range. `get_parameter_constraints` now emits the full `{xi, alpha, kappa}` triple
  (`{loc_lower, scl_lower, -10.0}` / `{loc_upper, scl_upper, 10.0}`), and `mle()` takes the
  `begin() + 1` slice at its own call site, mirroring C# line 570. No numeric behaviour changed
  on the MLE path (the slice is the same two values in the same order); the other eight callers
  now see the values C# gives them instead of heap garbage. Guarded by
  `core/tests/test_parameter_constraint_sizes.cpp`, which asserts the
  length-equals-`NumberOfParameters` contract for every `IMaximumLikelihoodEstimation`
  implementer. No fixture value moved: ctest, testthat, pytest and the oracle gate are all
  unchanged.
- **Suggested C# fix:** none — the C# is correct.

## Reconciliation pass, July 2026 upstream sync (a2c4dbf/fc28c0c to 2a0357a/c2e6192)

Every entry in the issues log (this file and the open log, which were one file at the time) was
re-checked against the shipped source at the new pins. This is the summary; the per-entry detail
is in each entry's **Status:** bullet.

**Fixed upstream and now ported: 29 entries, plus 1 partly fixed.** In Numerics v2.1.4 (23 full, 1
partial): StudentT PDF Jacobian,
StudentT InverseCDF tail transform, Beta / GeneralizedBeta Mode, GeneralizedLogistic κ=0
L-moments, LogPearsonTypeIII forward Stirling branch, PT3 / LP3 signed T3, the VonMises factory
case, the SetParameters assign-then-validate ordering, GammaDistribution.PartialKp, the
BrentSearch.Bracket expansion factor, the unused MCMC `inititals` midpoint, the NoncentralT `TT`
dead variable, BivariateEmpirical's stale Bilinear cache, the Linear / Bilinear log10 split, the
Histogram AddData dead branches, the Search descending comparators, the HPCM underflow guard, the
MultivariateNormal COVSRT permutation region, MVNDNT's dead return, the ArchimedeanCopula
ValidateParameter sentinel, CompetingRisks' CorrelationMatrix side effect, the MAP fitness scale,
NUTS's ignored custom gradient, and the Gibbs conjugate-mean formula. The dead-variable entry is
the partial one: `NoncentralT.TT` is gone, the Interpolater `deltaStart` line is not. In
RMC.BestFit v2.0.0 (6): the
CS0104 YeoJohnsonLink ambiguity, the Jeffreys single-parameter crash, MixtureModel.Clone's lost
zero-inflation, BivariateDistribution PseudoLikelihood, ProcessThresholdSeries idempotency, and
the GMM ConvergedWithinTolerance off-by-one.

**Intentional C++ divergences retired**, because upstream adopted the behavior: the
GeneralizedLogistic κ→0 limits, the LogPearsonTypeIII large-α Stirling branch,
MixtureModel.Clone's zero-inflation re-sync, the MultivariateNormal COVSRT bounds guard, the
UnivariateDistribution Jeffreys single-parameter guard, and the BivariateDistribution
PseudoLikelihood estimate path. In the first two cases upstream's formulation is richer than ours
was (a truncated series and explicit correction terms), so the C++ converged onto upstream's exact
expressions rather than keeping its own.

**Still open on the port side**, unrelated to any upstream change: the thinned DEMCzs
population-sampler stream divergence (`thinning_interval > 1` is not oracle-guaranteed; every
shipped Bayesian fixture uses thin=1), the empty-ModelParameter-name collapse in
PriorInfluenceDiagnostics (since resolved by Phase 10's ParameterNames port -- see the entry
above), and the documented chaotic-sensitivity fidelity notes on the seeded short-chain analysis
curves.

**Re-checked and unchanged upstream:** the CentralMoments(int) / CentralMoments(double) overload
pair, the Interpolater `deltaStart` heuristic, JoeCopula's missing SetThetaFromTau, the SNIS
Fitness-vs-Weight sort key, `NextDoubles(length, dimension)` drawing from per-column sub-PRNGs, the
StudentT copula's ν upper bound of 30, and the parallel-reduction ordering in
`Bootstrap.ComputeAccelerationConstants` and `BayesianAnalysis.ComputeDIC` / `ComputeWAIC` /
`ComputePSISLOO` (the PSIS-LOO attribution was later found overstated -- its `Parallel.For` writes
per-index slots and is order-independent; see the narrowed entry in the open log). On the last of these, v2.1.4 did harden `Tools.ParallelAdd` itself (signed-zero
and NaN handling in the compare-and-swap, plus a separate retry helper), but the reduction ORDER is
still unfixed, so the run-to-run non-reproducibility the entries describe stands. The entries that
ask for no upstream action at all (the CompetingRisks / MVN.CDF overload chain, the all-zero RWMH
proposal covariance, NoncentralT's numerically integrated moments, and the just-identified B17C
J-statistic) are unchanged and still correct as written.
