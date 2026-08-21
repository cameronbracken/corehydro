# P1 Distribution Gaps (GeneralizedNormal + Pivotal Bootstrap) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the two remaining distribution-layer gaps -- the GeneralizedNormal univariate
distribution (restoring the 15-candidate `fit_distributions` set) and the covariance-aware pivotal
bootstrap (the Phase-3 severance) -- validated by fixtures and the dotnet oracle gate, exposed in R
and Python, documented, shipped as branch `port-distribution-gaps` at v0.8.0.

**Architecture:** GeneralizedNormal mirrors the GeneralizedExtremeValue port (same
Xi/Alpha/Kappa parameterization, same interface set) as a new header plus two factory arms, plus a
new ported `IStandardError` capability mixin so the generic `dist_runner` (not a GEV-style bespoke
slice) dispatches its standard-error methods. The pivotal bootstrap replaces the documented
omission block in `bootstrap.hpp` in place (three new support headers plus the pivotal region),
surfaced through the existing `bootstrap_custom` callback verb.

**Tech Stack:** C++17 header-only core, cpp11 (R), pybind11 (Python), nlohmann/json fixture
runner, dotnet 10 oracle emitter.

## Global Constraints

- Every ported file carries `// ported from: <path> @ 2a0357a` and mirrors the C# class/method
  layout line-for-line where possible.
- Oracle values live ONLY in `fixtures/*.json`; never hardcode expected values in test files.
- Every new fixture target/method gets a dotnet emitter driver; `python3 tools/verify_oracles.py`
  must end `0 failed`.
- No `M_PI` (use `corehydro::numerics::kPi`); no namespace aliases named `gamma` or `stat`; no
  new external C++ dependencies.
- After any core class-layout change: `R CMD INSTALL --preclean corehydror`.
- After editing any `corehydror/src/*.cpp`: `Rscript -e 'cpp11::cpp_register("corehydror")'`.
- pytest reads fixtures materialized by pip; re-run
  `pixi run python -m pip install --force-reinstall --no-deps ./corehydropy` after fixture edits.
- BestFit C# sources are read with `git show`, never grep over the working tree. (P1 touches only
  Numerics sources, which are clean UTF-8.)
- Commits are GPG-signed, identity `Cam Bracken <cameron.bracken@pm.me>`, no Co-Authored-By
  trailers. Push only when Cam asks.
- Baseline numbers before this branch: ctest 87/87; oracle gate 5426 reproduced / 0 failed / 11
  skipped; testthat 6144/0; pytest 1487.

---

### Task 1: Branch + IStandardError mixin

**Files:**
- Create: `core/include/corehydro/numerics/distributions/base/i_standard_error.hpp`
- Test: compile-only (exercised by Task 2's fixture; this header is an interface)

**Interfaces:**
- Produces: `corehydro::numerics::distributions::IStandardError` with pure virtuals
  `parameter_covariance(int sample_size, ParameterEstimationMethod method) const ->
  math::linalg::Matrix2D`, `quantile_variance(double probability, int sample_size,
  ParameterEstimationMethod method) const -> double`, `quantile_gradient(double probability)
  const -> std::vector<double>`, `quantile_jacobian(const std::vector<double>& probabilities,
  double& determinant) const -> math::linalg::Matrix2D`.

- [ ] **Step 1: Create the branch**

```bash
cd /Users/cam/projects/usace/rmc/corehydro
git checkout main && git pull && git checkout -b port-distribution-gaps
```

- [ ] **Step 2: Read the C# interface**

```bash
git -C upstream/Numerics show 'HEAD:Numerics/Distributions/Univariate/Uncertainty Analysis/IStandardError.cs'
```

- [ ] **Step 3: Write the header**

Port the interface verbatim as a capability mixin, matching the existing mixin style (open
`core/include/corehydro/numerics/distributions/base/i_estimation.hpp` and copy its layout: include
guard, provenance header, namespace `corehydro::numerics::distributions`, pure-virtual methods,
virtual defaulted destructor). Use the exact signatures from the Interfaces block above.
`ParameterEstimationMethod` and `math::linalg::Matrix2D` are already in the tree (see how
`generalized_extreme_value.hpp` includes them).

- [ ] **Step 4: Verify it compiles standalone**

```bash
clang++ -std=c++17 -fsyntax-only -Icore/include -x c++ core/include/corehydro/numerics/distributions/base/i_standard_error.hpp
```
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add core/include/corehydro/numerics/distributions/base/i_standard_error.hpp
git commit -m "feat: port IStandardError capability mixin"
```

---

### Task 2: Port GeneralizedNormal (header + factory + fixture, TDD via the fixture runner)

**Files:**
- Create: `core/include/corehydro/numerics/distributions/generalized_normal.hpp`
- Create: `fixtures/distributions/univariate/generalized_normal.json`
- Modify: `core/include/corehydro/numerics/distributions/base/univariate_distribution_factory.hpp`
  (add `#include` + enum-switch case + string-name case; delete the lines 16-20 header comment
  that documents GeneralizedNormal as unported)
- Modify: `core/include/corehydro/numerics/distributions/support/dist_runner.hpp` (add
  `parameter_covariance` / `quantile_variance` / `quantile_gradient` method dispatch via
  `dynamic_cast<const IStandardError*>`)

**Interfaces:**
- Consumes: `IStandardError` from Task 1.
- Produces: `class GeneralizedNormal : public UnivariateDistributionBase, public IEstimation,
  public IMaximumLikelihoodEstimation, public ILinearMomentEstimation, public IStandardError,
  public IBootstrappable` in namespace `corehydro::numerics::distributions`, constructible as
  `GeneralizedNormal()` (defaults 100, 10, 0 per C#) and
  `GeneralizedNormal(double location, double scale, double shape)`; factory-reachable as
  `UnivariateDistributionType::GeneralizedNormal` and string `"GeneralizedNormal"`.

- [ ] **Step 1: Write the failing fixture first**

Curate `fixtures/distributions/univariate/generalized_normal.json` from the C# test file:

```bash
git -C upstream/Numerics show 'HEAD:Test_Numerics/Distributions/Univariate/Test_GeneralizedNormal.cs'
```

Copy the schema of `fixtures/distributions/univariate/generalized_logistic.json` exactly:
`"target": "GeneralizedNormal"`, `"kind": "univariate_distribution"`, `"source"` naming the C#
test file, datasets from the test's sample arrays, and cases covering every method the C# test
asserts. Follow the method-name vocabulary already used by kindred fixtures (`pdf`, `cdf`,
`inverse_cdf`, `moments`, `linear_moments`, plus the fitting and `random_value` case shapes --
copy them from `generalized_logistic.json` and the GEV fixture rather than inventing names).
Add cases for the three standard-error methods using the method names `parameter_covariance`
(args: sample_size, row, col -- matching the GEV slice's flattened-args convention in
`core/tests/test_fixtures.cpp` `dispatch_gev`), `quantile_variance` (probability, sample_size),
and `quantile_gradient` (probability, index). Where the C# test file lacks a literal you need,
leave the assertion out for now; Task 3 pins curated values with the emitter.

- [ ] **Step 2: Run the C++ fixture runner to verify it fails**

```bash
cmake -S core -B core/build && cmake --build core/build --target test_fixtures -j
ctest --test-dir core/build -R test_fixtures
```
Expected: FAIL -- the factory throws on the unhandled `GeneralizedNormal` case.

- [ ] **Step 3: Port the distribution header**

```bash
git -C upstream/Numerics show 'HEAD:Numerics/Distributions/Univariate/GeneralizedNormal.cs' > /tmp/GeneralizedNormal.cs
```

Write `core/include/corehydro/numerics/distributions/generalized_normal.hpp` mirroring
`generalized_extreme_value.hpp` (the closest kin: same Xi/Alpha/Kappa parameters, same
interface set, same `Erf`/special-function usage). Port every C# member in file order:
constructors, `Xi`/`Alpha`/`Kappa` accessors, `NumberOfParameters` (3), `Type`, display names,
`ParametersToString`, `ParameterNamesShortForm`, `GetParameterPropertyNames`, `GetParameters`,
`ParameterNames`, `Mean`/`Median`/`Mode`/`StandardDeviation`/`Skewness`/`Kurtosis`,
`Minimum`/`Maximum`, `MinimumOfParameters`/`MaximumOfParameters`, `Estimate`, `Bootstrap`
(override the `IBootstrappable` signature -- copy the pattern from `normal.hpp`'s existing
`bootstrap()` override), `SetParameters` (both overloads), `ValidateParameters` (both),
`ParametersFromLinearMoments`, `LinearMomentsFromParameters`, `GetParameterConstraints`, `MLE`,
`PDF`/`CDF`/`InverseCDF`, `Clone`, and the four `IStandardError` methods. Provenance header:
`// ported from: Numerics/Distributions/Univariate/GeneralizedNormal.cs @ 2a0357a`.

- [ ] **Step 4: Wire the factory**

In `univariate_distribution_factory.hpp`: add
`#include "corehydro/numerics/distributions/generalized_normal.hpp"` beside the other includes;
add `case UnivariateDistributionType::GeneralizedNormal: return std::make_unique<GeneralizedNormal>();`
in the enum switch (C# switch order); add the string case `"GeneralizedNormal"` beside the other
name mappings (check the file for how display-name and enum-name strings are both matched and
mirror that for this type). Delete the stale "GeneralizedNormal is not ported" comment block.

- [ ] **Step 5: Extend the generic runner for IStandardError**

In `dist_runner.hpp`'s method dispatch, add three arms following the existing pattern of
capability casts (search the file for `dynamic_cast` to see the established shape):

```cpp
if (method == "parameter_covariance") {
    auto* se = dynamic_cast<const IStandardError*>(&dist);
    if (!se) throw std::runtime_error("distribution does not implement IStandardError");
    auto cov = se->parameter_covariance(static_cast<int>(arg(0)), ParameterEstimationMethod::MaximumLikelihood);
    return {cov[static_cast<int>(arg(1))][static_cast<int>(arg(2))]};
}
if (method == "quantile_variance") { /* cast as above */
    return {se->quantile_variance(arg(0), static_cast<int>(arg(1)), ParameterEstimationMethod::MaximumLikelihood)};
}
if (method == "quantile_gradient") { /* cast as above */
    return {se->quantile_gradient(arg(0))[static_cast<int>(arg(1))]};
}
```

Adapt argument access and the return shape to the file's actual `DistResult` conventions (read
the neighboring arms first; the snippet shows intent, the file dictates form). Leave the GEV
bespoke slice in `test_fixtures.cpp` untouched.

- [ ] **Step 6: Run the fixture runner to verify it passes**

```bash
cmake --build core/build --target test_fixtures -j && ctest --test-dir core/build -R test_fixtures
```
Expected: PASS with the new GeneralizedNormal checks counted.

- [ ] **Step 7: Commit**

```bash
git add core/include/corehydro/numerics/distributions/generalized_normal.hpp \
        core/include/corehydro/numerics/distributions/base/univariate_distribution_factory.hpp \
        core/include/corehydro/numerics/distributions/support/dist_runner.hpp \
        fixtures/distributions/univariate/generalized_normal.json
git commit -m "feat: port GeneralizedNormal distribution with IStandardError dispatch"
```

---

### Task 3: Oracle-pin GeneralizedNormal and prove it in R and Python

**Files:**
- Modify: `tools/oracle_emitter/Program.cs` (Dispatch arms for the three standard-error methods
  if not already generic; the pdf/cdf/moment path dispatches by name and needs no edit)
- Modify: `fixtures/distributions/univariate/generalized_normal.json` (curated values)
- Test: `corehydror/tests/testthat/test-fixtures.R`, `corehydropy/tests/test_fixtures.py`
  (existing generic runners -- expect count increases only, no code edits per the
  three-edits-per-distribution contract; add a dispatch entry only if the runner's method table
  requires one for the standard-error names)

**Interfaces:**
- Consumes: `GeneralizedNormal` from Task 2 via the factory string `"GeneralizedNormal"`.

- [ ] **Step 1: Emitter coverage**

Open `tools/oracle_emitter/Program.cs`, find the univariate `Dispatch` (the by-name method table
the `univariate_distribution` kind drives). Confirm `parameter_covariance` /
`quantile_variance` / `quantile_gradient` arms exist for the C# `IStandardError` calls; add them
if missing, mirroring the C++ `dist_runner` arms from Task 2 Step 5 (C# side:
`((IStandardError)dist).ParameterCovariance(sampleSize, ParameterEstimationMethod.MaximumLikelihood)[r, c]` etc.).

- [ ] **Step 2: Pin curated values**

```bash
python3 tools/verify_oracles.py --dump 2>/dev/null | less   # or the gate's documented --dump flow
python3 tools/verify_oracles.py
```
Expected: `0 failed`. For any fixture assertion the C# test file did not provide, paste the
`--dump` actuals into the fixture with `"source": "curated via oracle_emitter --dump"`. If a
standard-error value does not reproduce C#-vs-C++ to tolerance, follow the GEV precedent exactly:
investigate first (see memory `prove-sensitivity-by-conditioning`), and only with proof of
inherent sensitivity document it as an oracle skip like the 11 GEV entries -- never loosen
tolerance silently.

- [ ] **Step 3: R and Python suites pick it up generically**

```bash
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
```
Expected: both suites pass with counts strictly above the baseline (testthat > 6144, pytest >
1487), the new checks coming from `generalized_normal.json`. If either runner errors on the
standard-error method names, add the dispatch entry to its method table following the runner's
existing pattern -- and note it in the commit message.

- [ ] **Step 4: Sanity-check the user surface**

```bash
Rscript -e 'library(corehydror); d <- distribution("GeneralizedNormal", c(100, 10, 0)); dist_pdf(d, 100)'
pixi run python -c "from corehydropy import Distribution; print(Distribution('GeneralizedNormal', [100, 10, 0]).pdf(100.0))"
```
Expected: both print the same finite density; no error. (Exact equality is asserted by the
fixture suite; this is a smoke check that the public verbs reach the new family.)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test: pin GeneralizedNormal oracles through the dotnet gate"
```

---

### Task 4: FittingAnalysis 14 -> 15 candidates (single atomic re-pin)

**Files:**
- Modify: `core/include/corehydro/analyses/distribution_fitting/fitting_analysis.hpp` (insert
  `UnivariateDistributionType::GeneralizedNormal` into `kCandidates` between GeneralizedLogistic
  and GeneralizedPareto; update the file-header COUNT note and the 14-vs-15 comments)
- Modify: `tools/oracle_emitter/Program.cs:3782-3793` (delete the
  `analysis.DistributionList.RemoveAll(d => d is GeneralizedNormal);` line and its comment block)
- Modify: `fixtures/analyses/fit_distributions_smoke.json` (re-pin: candidate_count 15, all
  candidate indices shift by +1 at and after the GeneralizedNormal slot, new GeneralizedNormal
  row, updated `"source"` prose)
- Modify: `corehydror/R/analysis.R:91` and `corehydropy/src/corehydropy/analysis.py:156`
  ("14 candidate" -> "15 candidate" in the docs), plus any "14-candidate" phrasing found by
  `grep -rn '14.candidate\|candidate_count' corehydror/R corehydropy/src fixtures/README.md`

**Interfaces:**
- Consumes: factory-reachable `GeneralizedNormal` from Task 2.

- [ ] **Step 1: Flip the C++ candidate list and the emitter in one edit**

Make the `fitting_analysis.hpp` and `Program.cs` edits together -- the fixture is wrong for
either half alone, which is exactly why this task is atomic.

- [ ] **Step 2: Re-pin the fixture from the real C#**

```bash
python3 tools/verify_oracles.py   # expect FAILURES on fit_distributions_smoke.json -- good, both sides moved
python3 tools/verify_oracles.py --dump   # capture the 15-candidate actuals
```
Update `fit_distributions_smoke.json`: `candidate_count` 15; re-index every asserted candidate
(index 11 was Normal in the 14-set; verify the new index of every asserted row against the dump
rather than assuming +1); add a GeneralizedNormal row asserting its aic and converged flag;
rewrite the `"source"` narrative (drop the removal explanation, keep the GeneralizedPareto
regression-pinning history).

- [ ] **Step 3: Verify all four runners**

```bash
python3 tools/verify_oracles.py          # expected: 0 failed
ctest --test-dir core/build -R test_fixtures
R CMD INSTALL corehydror && Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy && pixi run python -m pytest corehydropy/tests -q
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: restore the 15-candidate FittingAnalysis set with GeneralizedNormal"
```

---

### Task 5: Port the pivotal bootstrap support types

**Files:**
- Create: `core/include/corehydro/numerics/sampling/bootstrap/support/pivotal_bootstrap_invalid_draw_policy.hpp`
- Create: `core/include/corehydro/numerics/sampling/bootstrap/support/pivotal_bootstrap_diagnostics.hpp`
- Create: `core/include/corehydro/numerics/sampling/bootstrap/support/pivotal_bootstrap_context.hpp`

**Interfaces:**
- Consumes: `BootstrapFit` from `support/bootstrap_fit.hpp`.
- Produces: `enum class PivotalBootstrapInvalidDrawPolicy { Drop, UseRaw, UseParent }` (confirm
  member names against the C# enum); `struct PivotalBootstrapDiagnostics` with int fields
  `requested_replicates`, `rejected_raw_replicates`, `failed_raw_replicates`,
  `accepted_raw_replicates`, `invalid_pivotal_replicates`, `retained_pivotal_replicates` (the two
  C# `TimeSpan` timing fields are dropped -- wall-clock is not oracle-comparable; note the
  deviation in the header); `class PivotalBootstrapContext` with the parent fit, the raw fits
  array, `parameter_count()`, and `get_raw_parameter_values(int parameter_index)`.

- [ ] **Step 1: Read the three C# sources**

```bash
for f in PivotalBootstrapInvalidDrawPolicy PivotalBootstrapDiagnostics PivotalBootstrapContext; do
  git -C upstream/Numerics show "HEAD:Numerics/Sampling/Bootstrap/Support/$f.cs"
done
```

- [ ] **Step 2: Port each file**

Mirror `support/bootstrap_fit.hpp`'s conventions (namespace, provenance header, doc comments).
These are small DTO/enum files; port member-for-member with the one documented deviation above.

- [ ] **Step 3: Syntax-check and commit**

```bash
for f in core/include/corehydro/numerics/sampling/bootstrap/support/pivotal_*.hpp; do
  clang++ -std=c++17 -fsyntax-only -Icore/include -x c++ "$f" || exit 1
done
git add core/include/corehydro/numerics/sampling/bootstrap/support/pivotal_*.hpp
git commit -m "feat: port pivotal bootstrap support types"
```

---

### Task 6: Un-omit the pivotal region in bootstrap.hpp, proven by a ported C# test suite

**Files:**
- Modify: `core/include/corehydro/numerics/sampling/bootstrap/bootstrap.hpp` (the omission block)
- Create: `core/tests/test_pivotal_bootstrap.cpp`
- Modify: `core/CMakeLists.txt` (register the new ctest target beside the existing test suites)

**Interfaces:**
- Consumes: the three Task 5 headers; `ILinkFunction` /
  `link_function_factory.hpp` (Phase 6); `BootstrapFit`; `CholeskyDecomposition` and
  `MatrixRegularization` under `numerics/math/linalg/`.
- Produces, on `Bootstrap<TData>`: second constructor `Bootstrap(TData original_data,
  BootstrapFit original_fit)`; public members `fit_with_covariance_function`
  (`std::function<BootstrapFit(const TData&)>`), `original_covariance`, `pivotal_link_factory`,
  `pivotal_replicate_filter`, `pivotal_parameter_validator`, `pivotal_invalid_draw_policy`
  (default `Drop`), `regularize_pivotal_covariances` (default true), `pivotal_z_limit`,
  `add_pivotal_jitter` (default false), `pivotal_jitter_scale` (default 0.01); methods
  `run_pivotal_bootstrap()`, `transform_pivotal_bootstrap(...)`,
  `get_raw_pivotal_confidence_intervals(...)`; accessors `raw_bootstrap_fits()`,
  `raw_bootstrap_parameter_sets()`, `raw_bootstrap_statistics()`, `pivotal_links()`,
  `pivotal_diagnostics()`; `BootstrapRunType::Pivotal` and the pivotal branch in
  `get_confidence_intervals` (exact C# signatures govern -- transcribe, do not redesign).

- [ ] **Step 1: Extract the C# region**

```bash
git -C upstream/Numerics show 'HEAD:Numerics/Sampling/Bootstrap/Bootstrap.cs' > /tmp/Bootstrap.cs
git -C upstream/Numerics show 'HEAD:Test_Numerics/Sampling/Test_PivotalBootstrap.cs' > /tmp/Test_PivotalBootstrap.cs
```

- [ ] **Step 2: Write the failing test suite first**

Create `core/tests/test_pivotal_bootstrap.cpp` porting the Test_PivotalBootstrap.cs test methods
1:1 (all 18 as of the current pin: the Transform_* link/covariance/policy/filter tests, the RunPivotalBootstrap_*
behavior and throw tests, `Run_RegularBootstrap_IgnoresPivotalOnlyProperties`,
`GetConfidenceIntervals_BCaAfterPivotalRun_Throws`,
`GetRawPivotalConfidenceIntervals_BeforePivotalRun_Throws`,
`RunPivotalBootstrap_WithSameSeed_IsReproducible`, and the statistical-coverage test at the
file's end). Copy the harness style from an existing suite (open
`core/tests/test_bootstrap_analysis.cpp` for the assertion macros/registration pattern).
Structural and behavioral assertions (counts, throws, reproducibility, policy effects) transcribe
directly from the C# asserts; these are C#-test-literal oracles, which is the same standard the
analysis ctest suites already use. Register the target in `core/CMakeLists.txt` exactly as the
neighboring suites are registered.

```bash
cmake -S core -B core/build && cmake --build core/build --target test_pivotal_bootstrap -j
```
Expected: FAIL to compile -- the pivotal members do not exist yet.

- [ ] **Step 3: Port the pivotal region**

In `bootstrap.hpp`: rewrite the file-header omission paragraph into a "pivotal workflow ported"
note; add `Pivotal` to `BootstrapRunType`; add the shared state the header currently documents as
dropped (`original_covariance_`, `raw_bootstrap_*`, `pivotal_links_`, `pivotal_diagnostics_`);
add the second constructor and the public members; transcribe `RunPivotalBootstrap`,
`TransformPivotalBootstrap`, `GetRawPivotalConfidenceIntervals`, `TryCreatePivotalDraw`,
`TryApplyInvalidPolicy`, `LinkCovariance`, `CreatePivotalLinks` in C# file order; add the
pivotal branch to `get_confidence_intervals` (line ~724's omission note goes away). Keep the
established port conventions already stated in this file's header: serial loops for
`Parallel.For` with the order-independence argument re-verified for the pivotal loop, the exact
per-replicate seeding cascade, `std::pow(x, 1.0/3.0)` transform semantics.

- [ ] **Step 4: Run the suite until green, then the full ctest**

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_pivotal_bootstrap
ctest --test-dir core/build
```
Expected: new suite passes; full suite 88/88 (87 baseline + this suite; test_fixtures count
unchanged by this task).

- [ ] **Step 5: Commit**

```bash
git add core/include/corehydro/numerics/sampling/bootstrap/bootstrap.hpp \
        core/tests/test_pivotal_bootstrap.cpp core/CMakeLists.txt
git commit -m "feat: port the covariance-aware pivotal bootstrap workflow"
```

---

### Task 7: Surface the pivotal bootstrap through bootstrap_custom

**Files:**
- Modify: `core/include/corehydro/numerics/support/callback/bootstrap.hpp` (pivotal options +
  the `fit_with_covariance` callback + diagnostics in the result)
- Modify: `corehydror/src/callback.cpp`, `corehydror/R/` (the file defining `bootstrap_custom()`;
  locate with `grep -rn 'bootstrap_custom' corehydror/R`)
- Modify: `corehydropy/src/bindings/callback.cpp`, `corehydropy/src/corehydropy/` (the module
  defining `bootstrap_custom`; locate with `grep -rn 'bootstrap_custom' corehydropy/src`)
- Create: pivotal cases in `fixtures/callback/bootstrap.json` and a seeded pivotal interval case
  in `fixtures/callback/callback_cross_language.json`
- Modify: `tools/oracle_emitter/Program.cs` (drive the new cases; the emitter needs a driver for
  every fixture case -- memory `phase9a-complete` lesson)

**Interfaces:**
- Consumes: Task 6's `Bootstrap<TData>` pivotal surface.
- Produces: `bootstrap_custom()` gains optional arguments (same names in R and Python):
  `fit_with_covariance` (callback: data -> list/dict with `parameters` (numeric vector) and
  `covariance` (matrix, row-major)), `original_covariance` (matrix),
  `pivotal_invalid_draw_policy` (string: `"drop"`/`"use_raw"`/`"use_parent"`),
  `regularize_pivotal_covariances` (bool, default TRUE/True), `pivotal_z_limit` (numeric or
  NULL/None), `add_pivotal_jitter` (bool), `pivotal_jitter_scale` (numeric, 0.01), and a
  `run_type = "pivotal"` selector beside the existing run types. The result gains a
  `pivotal_diagnostics` element (named list / dict of the six DTO ints) and the raw-vs-pivotal
  interval blocks mirroring `get_raw_pivotal_confidence_intervals`.

- [ ] **Step 1: Read the current callback group end-to-end**

Read `core/include/corehydro/numerics/support/callback/bootstrap.hpp` fully, then the R and
Python glue for `bootstrap_custom`, so the new options ride the existing options_json and
callback-marshalling patterns (GuardedCall for the new callback, exactly like the four existing
delegates).

- [ ] **Step 2: Write failing R and Python tests**

Add to the existing bootstrap_custom test files (locate:
`grep -rln 'bootstrap_custom' corehydror/tests corehydropy/tests`) a seeded pivotal run:
normal-location-scale data, a `fit_with_covariance` computing the MLE mean/sd and its analytic
covariance (`[[s2/n, 0], [0, s2/(2n)]]`), replicates 200, seed 12345, asserting (a) the result
has `pivotal_diagnostics` with `requested_replicates == 200` and (b) R and Python produce
identical interval endpoints (the arithmetic-only-callback determinism contract from the
callback layer). Run both suites; expected: FAIL (unknown argument).

- [ ] **Step 3: Implement the C++ callback-group extension, then the two glues**

Follow the file's existing option-parsing and result-block helpers (`push_block` etc.). The
`fit_with_covariance` host callback crosses through `GuardedCall` and shares the run's
`CallbackAbortState`. Then extend the R and Python signatures per the Interfaces block, re-run
`Rscript -e 'cpp11::cpp_register("corehydror")'`, reinstall both packages, and run the two new
tests until green.

- [ ] **Step 4: Fixture + emitter parity**

Add the pivotal cases to `fixtures/callback/bootstrap.json` (fixture-catalog callbacks, mirroring
how the existing four-delegate cases define their callable catalog) and one seeded pivotal
interval case to `callback_cross_language.json` at ZERO tolerance. Add the emitter driver arm in
`Program.cs` running the same catalog against the real C# `Bootstrap<TData>` pivotal path.

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: fixture runner green; gate `0 failed`. The cross-language case must hold at zero
tolerance in all four runners; if C#-vs-C++ drifts, check FMA contraction first (the
`-ffp-contract=off` fixture-catalog precedent) before touching the fixture.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: expose the pivotal bootstrap through bootstrap_custom"
```

---

### Task 8: Documentation and worked examples 16 + 17

**Files:**
- Modify: R docs for `distribution()` (family table gains GeneralizedNormal) and
  `bootstrap_custom()` (pivotal arguments); regenerate Rd via
  `Rscript -e 'roxygen2::roxygenise("corehydror")'`
- Modify: Python docstrings for `Distribution` (family list) and `bootstrap_custom`
- Modify: `corehydror/_pkgdown.yml` + `site/_quarto.yml` ONLY if any new export name was added
  (expected: none -- both features ride existing verbs; verify with
  `Rscript -e 'pkgdown::check_pkgdown("corehydror")'`)
- Create: `site/examples/16-fifteen-candidate-fitting/{python.ipynb, r.qmd}`
- Create: `site/examples/17-pivotal-bootstrap/{python.ipynb, r.qmd}`
- Modify: the site examples index page listing (find it:
  `grep -rn '15-' site/_quarto.yml site/examples/index.qmd 2>/dev/null`)

**Interfaces:**
- Consumes: everything shipped in Tasks 2-7.

- [ ] **Step 1: Reference docs**

Update the R roxygen and Python docstrings named above ("14 candidate" prose was already fixed in
Task 4; this step covers the family list and the new bootstrap_custom arguments, with one runnable
snippet each). Follow the elements-of-style conventions.

- [ ] **Step 2: Example 16 -- distribution fitting with all 15 candidates**

Copy the structure of an existing pair (e.g. `site/examples/11-*/`). Content: load a flood
series already used in the fixtures, run `fit_distributions`, show the 15-row ranking table,
plot the top fit, and close with the executable reproduction check comparing a pinned aic value
at 1e-15 relative tolerance (the established check pattern -- copy a `stopifnot`/`assert` block
from example 11 and swap values). Python notebook committed WITH outputs
(`jupyter nbconvert --to notebook --execute --inplace site/examples/16-fifteen-candidate-fitting/python.ipynb`);
R rendered so `site/_freeze/` updates.

- [ ] **Step 3: Example 17 -- pivotal bootstrap**

Content: the normal-location-scale workflow from Task 7's test, comparing regular percentile vs
pivotal intervals, showing `pivotal_diagnostics`, and stating in prose the callback-layer
cross-language limit (registry-path bit-identity vs user-arithmetic determinism) exactly as
examples 14/15 do. Same execution/commit mechanics as Step 2.

- [ ] **Step 4: Build the site and verify**

```bash
pixi run docs
```
Expected: clean build; both new pages render with outputs; `pkgdown::check_pkgdown` clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "docs: reference updates and worked examples 16-17"
```

---

### Task 9: Version 0.8.0, full verification, ship

**Files:**
- Modify: `corehydror/DESCRIPTION` (Version: 0.8.0), `corehydropy/pyproject.toml`
  (version = "0.8.0"), `corehydror/NEWS.md`, `CHANGELOG.md`
- Modify: `.claude/CLAUDE.md` (Status section: append the P1 paragraph with final numbers;
  update the "42 univariate distributions" phrasing to 43 where it appears)

**Interfaces:**
- Consumes: the completed branch.

- [ ] **Step 1: Bump versions and write release notes**

NEWS.md / CHANGELOG.md entries: GeneralizedNormal (43rd distribution, 15-candidate
fit_distributions), pivotal bootstrap (new bootstrap_custom pivotal surface), IStandardError
generic dispatch.

- [ ] **Step 2: Full verification (evidence before assertions)**

```bash
cmake --build core/build -j && ctest --test-dir core/build
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py
```
Expected: ctest 88/88; testthat 0 failures; pytest 0 failures; gate `0 failed` with reproduced
count > 5426. Record the four final numbers in the NEWS/CHANGELOG entry and `.claude/CLAUDE.md`.

- [ ] **Step 3: R CMD check regression**

```bash
R CMD build corehydror && R CMD check --as-cran corehydror_0.8.0.tar.gz
```
Expected: the same three known NOTEs, no new WARNING or NOTE.

- [ ] **Step 4: Commit, push, PR, CI**

```bash
git add -A && git commit -m "chore: release v0.8.0 (GeneralizedNormal + pivotal bootstrap)"
git push -u origin port-distribution-gaps
gh pr create --title "P1: distribution gaps -- GeneralizedNormal + pivotal bootstrap (v0.8.0)" \
  --body "$(sed -n '/^## v0.8.0/,/^## /p' CHANGELOG.md)"
gh run watch $(gh run list --branch port-distribution-gaps -L1 --json databaseId -q '.[0].databaseId') --exit-status
```
Expected: CI green on the full matrix. Do not merge without Cam's go-ahead.

---

## Self-review notes

- Spec coverage: GeneralizedNormal (header, factory, ParameterNames, fixture, gate, dist surface,
  15-candidate re-pin with emitter flip) -> Tasks 2-4. Pivotal bootstrap (three support files,
  bootstrap.hpp branches, bootstrap_custom + diagnostics DTO, preclean rebuild) -> Tasks 5-7.
  Docs/examples 16-17 -> Task 8. Version/ship -> Task 9.
- The preclean-rebuild constraint is embedded in Tasks 3 and 9 (both install with `--preclean`
  after the class-layout changes in Tasks 2 and 6).
- Type consistency: `IStandardError` signatures in Task 1 match the Task 2 dispatch arms;
  `PivotalBootstrapDiagnostics` field names in Task 5 match Task 7's result block.
- Deliberate non-placeholders: port tasks cite the exact C# source as the code spec (the port
  convention: the .cs file IS the implementation reference); snippets are given where the code is
  genuinely new (dispatch arms, option surface, fixture shapes).
