# P4 Data and Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the remaining Numerics data-and-testing layer -- `HypothesisTests` and the four
`Statistics`/`Tools` helpers it needs, the `Correlation` matrix overloads, the whole Paired Data
subsystem (Ordinate, UncertainOrdinate, OrderedPairedData, UncertainOrderedPairedData,
LineSimplification) with its one consumer `TabularFunction` and the six severed `Search.cs`
overloads it un-gates -- then un-gate the two severed RMC.BestFit `DataFrame` facades that
`HypothesisTests` was blocking, all validated by ctest transcriptions of the C# test files, pinned
by fixtures through all four runners with dotnet-gate reproduction, exposed in R and Python,
documented, shipped as branch `port-data-and-tests` at v0.11.0.

**Architecture:** Three new dispatch surfaces, each on a pattern the repo already established.
`hypothesis` and `paired_data` are new one-header groups under
`core/include/corehydro/numerics/support/toolbox/`, dispatched by `toolbox_runner.hpp` exactly as
P3's `network` group was, taking bulk data as native double vectors and scalars/enums/flags in
`options_json`. The `Correlation` matrix overloads join the EXISTING `correlation` group as two
new methods returning a flattened matrix with `dims`. The DataFrame facades are different in kind
-- their input is a whole `data_frame` spec object, not a flat vector -- so they extend the
existing `models/data_frame_runner.hpp` (the fifth runner family, already driven identically by
the R glue, the Python glue and the C++ fixture runner) behind a new `data_frame` fixture kind
whose assertion grammar is byte-identical to `toolbox`'s, so all three runners reuse their
existing `toolbox_select` helper verbatim.

**Tech Stack:** C++17 header-only core, cpp11 (R), pybind11 (Python), nlohmann/json fixture runner,
dotnet 10 oracle emitter.

## Global Constraints

- Every ported file carries `// ported from: <path> @ 2a0357a` (Numerics) or `@ c2e6192`
  (RMC.BestFit) and mirrors the C# class/method layout line-for-line where possible.
- Upstream paths contain spaces (`Numerics/Data/Paired Data/`, `Numerics/Data/Statistics/`) --
  quote them in every shell command. Numerics is valid UTF-8 and its working tree is checked out
  clean, so reading and grepping it directly is fine.
- **RMC.BestFit is read with `git -C upstream/RMC-BestFit show HEAD:'<path>'`, never grepped over
  the working tree.** One measured exception found during recon and recorded here so nobody
  re-derives it: `src/RMC.BestFit/Models/DataFrame/DataFrame.cs` at `c2e6192` IS valid UTF-8 (with
  a BOM) and must be read as UTF-8. Piping it through `iconv -f windows-1252` MANGLES it (it turns
  the `λ` in the summary-dictionary key `"Events Per Index (λ)"` into `Î»`), and that key is
  oracle-visible. Read it plain.
- Oracle values live ONLY in `fixtures/*.json`; ctest suites transcribing C# test methods carry
  C#-test-literal assertions (the established second oracle class); never invent expected values.
  Where upstream has no test at all (the `Correlation` matrix overloads, `SummaryStatisticsExactDataOnly`),
  curate with `python3 tools/verify_oracles.py --dump` against the real library and say so in the
  fixture's `source` string, exactly as `fixtures/toolbox/correlation.json`'s `toolbox_kind_proof`
  case already does.
- Every new fixture case gets a dotnet emitter driver; `python3 tools/verify_oracles.py` must end
  `0 failed`. The `toolbox` branch of the emitter does NOT honor `oracle_skip` -- an assertion the
  C# cannot reproduce is OMITTED from the fixture, not masked.
- No `M_PI` (use `corehydro::numerics::kPi`); no namespace aliases named `gamma` or `stat`; no
  file-local `const` used implicitly in a capture-less lambda (MSVC C3493 -- use file-scope
  `constexpr`); no new external C++ dependencies.
- After editing any `corehydror/src/*.cpp`: `Rscript -e 'cpp11::cpp_register("corehydror")'`.
  Adding a toolbox GROUP does not require it (the glue is group-generic); adding a data-layer
  entry point in `corehydror/src/data.cpp` does.
- Task 5 adds public methods to `DataFrame`, a class-layout change, so EVERY R rebuild from Task 5
  onward uses `R CMD INSTALL --preclean corehydror`.
- pytest reads fixtures materialized by pip; re-run
  `pixi run python -m pip install --force-reinstall --no-deps ./corehydropy` after fixture edits.
  testthat reads `system.file("fixtures", package = "corehydror")`, so re-run `R CMD INSTALL`.
- Every new R export goes in BOTH `corehydror/_pkgdown.yml` and `site/_quarto.yml`
  `quartodoc.sections`, and in `corehydror/NAMESPACE` (hand-maintained -- do not let roxygen
  rewrite it). New Python exports join `corehydropy/src/corehydropy/<module>.py`'s `__all__` AND
  both the import block and `__all__` in `corehydropy/src/corehydropy/__init__.py`.
- R and Python error messages for the same condition must be textually identical (`stop(...,
  call. = FALSE)` vs `raise ValueError(...)`).
- Commits are GPG-signed, identity `Cam Bracken <cameron.bracken@pm.me>`, no Co-Authored-By
  trailers. Push only when Cam asks (the ship task pushes; that is the standing exception).
- **Branch base:** `port-data-and-tests`, already created off `afc405b` (`origin/port-math-extras`,
  the tip of the merged P1-P2-P3 stack, byte-identical to the P3 release commit `f109940`). Note
  for the ship task: `origin/main` is currently BEHIND this base -- PR #27 merged into
  `port-distribution-gaps` and PR #28 into `port-math-extras` rather than into `main`, so the
  stack has not landed. Target the P4 PR at `port-math-extras` and say so in the PR body; Cam
  decides how the stack lands on `main`.
- **Baseline numbers, measured on this branch at creation (2026-08-23), not copied from prose:**
  ctest **102/102**. The P3 release records the rest: oracle gate 5942 reproduced / 0 failed / 11
  skipped; testthat 6916/0; pytest 1711. Re-measure R/Python/oracle on the first task that runs
  them; if any number differs, the fresh measurement governs (the standing process note: never
  reconcile against a stale census).

## Scope decisions taken during recon (do not re-litigate)

1. **`UnimodalityTest` is NOT ported in P4, and neither is `DataFrame.SummaryHypothesisTest`.**
   `HypothesisTests.UnimodalityTest` constructs `Numerics.MachineLearning.GaussianMixtureModel`
   and calls `Train(12345, true)`; the ML layer is P5's whole subject. `SummaryHypothesisTest`
   calls all ten facades inside ONE try/catch that NaNs the entire ten-key dictionary if any one
   throws, so shipping it with a throwing unimodality arm would silently NaN nine working tests --
   a behavioral divergence, not a severance. Both are severed together with an in-header note
   naming P5 as the un-gate, exactly as `HypothesisTests` itself was named as the un-gate in
   `data_frame.hpp`'s Phase-5 note. `SummaryHypothesisTest` has no C# test coverage anyway (its
   only caller is the WPF GUI), so no oracle is forfeited by waiting. The two forfeited oracles
   are `UnimodalityTest`'s (`0.4142441` and `2.55425752131444e-05`), which P5 collects.
2. **`EmpiricalDistribution` and `KernelDensity` are NOT refactored onto the new
   `OrderedPairedData`.** Both currently inline the interpolation over plain vectors and document
   the divergence in their headers; both are oracle-pinned today. Porting Paired Data creates the
   opportunity and P4 declines it. Leave the headers and their notes alone.
3. **The six `Search.cs` overloads ARE un-severed** (Task 8). Their severance note says only "no
   caller in this port's scope needs them"; the type they needed now exists, and leaving a stale
   blocker note behind contradicts the phase's goal.
4. **`TabularFunction` rides along** (Task 9). It is the only pure consumer of the subsystem, it is
   already a recorded severance in `upstream/CLAUDE.md`, and `i_univariate_function.hpp`'s header
   already points at it. Leaving the `IUnivariateFunction` trio two-thirds complete for one more
   release has no upside.
5. **Validation MESSAGE STRINGS stay in the C++ API and the ctest suites; they do not cross the
   language boundary.** `ToolboxResult` carries doubles, not strings. `Ordinate::ordinate_errors`
   and friends are ported in full (their exact strings are C#-test-pinned, so ctest asserts them),
   but the `paired_data` group exposes validity as a boolean-valued scalar plus an error COUNT.
   Say so in the group header and in both help pages.

---

### Task 1: Branch prerequisites -- the four missing `Statistics` / `Tools` helpers

**Files:**
- Modify: `core/include/corehydro/numerics/tools.hpp` (add `pow(double, int)`)
- Modify: `core/include/corehydro/numerics/data/statistics.hpp` (add `mean_variance`, the
  tie-returning `ranks_in_place` overload, and the vector `percentile` overload; rewrite the
  file-header severance note at lines 10-15)
- Test: `core/tests/test_statistics_helpers.cpp` (new); Modify: `core/CMakeLists.txt`

**Interfaces:**
- Produces, in `namespace corehydro::numerics`:
  `inline double tools::pow(double a, int b)` -- upstream's hand-written binary exponentiation,
  NOT `std::pow`.
- Produces, in `namespace corehydro::numerics::data::statistics`:
  - `inline std::pair<double, double> mean_variance(const std::vector<double>& data)`
  - `inline std::vector<double> ranks_in_place(std::vector<double>& data, std::vector<double>& ties)`
    -- the tolerance-based tie overload, returning ranks and filling `ties`.
  - `inline std::vector<double> percentile(const std::vector<double>& data, const std::vector<double>& k, bool data_is_sorted = false)`

- [ ] **Step 1: Write the failing ctest**

Create `core/tests/test_statistics_helpers.cpp` transcribing, in this order:

1. `tools::pow` against the C# semantics read from `Numerics/Utilities/Tools.cs:157-179`. Assert
   `pow(2.0, 0) == 1.0`, `pow(1.0, -5) == 1.0`, `pow(-1.0, 3) == -1.0`, `pow(-1.0, 4) == 1.0`,
   `pow(2.0, 10) == 1024.0`, `pow(2.0, -2) == 0.25`, and `pow(0.0, -1) == inf` (the
   `result == 0 -> PositiveInfinity` arm). These are exact equalities, not tolerances.
2. `mean_variance` agreeing exactly with the existing `mean()` and `variance()` on a hand-written
   10-value vector (C# is literally `(Mean(data), Variance(data))`, so this is an identity test,
   not a new oracle).
3. `ranks_in_place(data, ties)` on a vector with a tie run in the middle AND a tie run at the very
   end. This is the important one -- see Step 3's fidelity note. Assert the ranks, and assert that
   the TRAILING run's length is NOT recorded in `ties` (upstream never calls `RanksTies` for it).
4. `percentile(data, {0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99})` agreeing element-for-element with
   seven scalar `percentile(data, k)` calls on the same unsorted vector.

Wire the file into `core/CMakeLists.txt` beside the other `test_*` entries.

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake -S core -B core/build && cmake --build core/build -j8 && ctest --test-dir core/build -R test_statistics_helpers
```
Expected: compile FAILS -- `pow`, `mean_variance`, the two-argument `ranks_in_place`, and the
vector `percentile` do not exist.

- [ ] **Step 3: Implement the four helpers**

`tools.hpp` gains, beside `sqr`:

```cpp
// ported from: Numerics/Utilities/Tools.cs @ 2a0357a
// Upstream is NOT Math.Pow: it is hand-written binary exponentiation with its own edge handling,
// and HypothesisTests calls it on integer exponents (WaldWolfowitz's cubes and fourth powers,
// MannWhitney's and MannKendall's tie cubes). Substituting std::pow drifts by an ULP and the
// oracles see it. Transcribed literally, including the 1/result == inf arm for a == 0.
inline double pow(double a, int b) {
    if (b == 0) return 1.0;
    if (a == 1.0) return 1.0;
    if (a == -1.0) return (b & 1) == 0 ? 1.0 : -1.0;
    bool neg = b < 0;
    long long n = b;
    if (neg) n = -n;
    double result = 1.0;
    double base_val = a;
    while (n > 0) {
        if ((n & 1LL) != 0) result *= base_val;
        base_val *= base_val;
        n >>= 1;
    }
    if (neg) {
        if (result == 0.0) return std::numeric_limits<double>::infinity();
        return 1.0 / result;
    }
    return result;
}
```

`statistics.hpp` gains the three overloads, transcribed from `Numerics/Data/Statistics/Statistics.cs`:
`MeanVariance` (line 262), `RanksInPlace(double[], out double[] ties)` (line 674), and
`Percentile(IList<double>, IList<double> k, bool)` (line 573).

**Three oracle-visible fidelity points in the tie overload; a numbered transcription note in the
header must record all three, because each one differs from the already-ported no-ties overload
sitting directly above it:**
1. It compares with `AlmostEquals(work[i], work[previous], Tools.DoubleMachineEpsilon)`, NOT the
   ported overload's `fabs(...) <= 0`.
2. `ties` is allocated at `data.size()` and written only in the else-branch as `ties[i - 1] = t`,
   so it is a SPARSE array of run lengths indexed by the END of each run, not a count vector.
3. The final `RanksTies(...)` call never records the TRAILING run's `t` -- a real upstream defect
   that must be reproduced, because `MannWhitneyTest`'s tie correction `T` and `MannKendallTest`'s
   `varS` both sum over `ties`.

The vector `percentile` sorts ONCE and calls the scalar with `data_is_sorted = true`; do not write
a naive loop over the ported scalar (which re-sorts per call) -- same answers, but the transcription
must mirror the C# structure.

Replace `statistics.hpp`'s header note at lines 10-15 (which currently says all three are omitted)
with a note saying they are now ported and pointing at the numbered tie-overload note.

- [ ] **Step 4: Run the ctest and watch it pass**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build -R test_statistics_helpers
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: port the Statistics tie-ranking, mean-variance and vector percentile helpers"
```

---

### Task 2: `HypothesisTests` (twelve of thirteen methods)

**Files:**
- Create: `core/include/corehydro/numerics/data/hypothesis_tests.hpp`
- Test: `core/tests/test_hypothesis_tests.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1's four helpers.
- Produces, in `namespace corehydro::numerics::data`, a `hypothesis_tests` namespace of free
  functions mirroring the C# statics one-for-one:
  - `double one_sample_t_test(const std::vector<double>& sample, double population_mean = 0.0)`
  - `double equal_variance_t_test(const std::vector<double>& s1, const std::vector<double>& s2)`
  - `double unequal_variance_t_test(const std::vector<double>& s1, const std::vector<double>& s2)`
  - `double paired_t_test(const std::vector<double>& s1, const std::vector<double>& s2)`
  - `double f_test(const std::vector<double>& s1, const std::vector<double>& s2)`
  - `void f_test_models(double sse_restricted, double sse_full, int df_restricted, int df_full, double& f_stat, double& p_value)`
  - `double jarque_bera_test(const std::vector<double>& sample)`
  - `double wald_wolfowitz_test(const std::vector<double>& sample)`
  - `double ljung_box_test(const std::vector<double>& sample, int lag_max = -1)`
  - `double mann_whitney_test(const std::vector<double>& s1, const std::vector<double>& s2)`
  - `double mann_kendall_test(const std::vector<double>& sample)`
  - `double linear_trend_test(const std::vector<double>& indices, const std::vector<double>& sample)`

- [ ] **Step 1: Write the failing ctest**

Create `core/tests/test_hypothesis_tests.cpp` transcribing every `[TestMethod]` in
`upstream/Numerics/Test_Numerics/Data/Statistics/Test_HypothesisTests.cs` EXCEPT `Test_Unimodality`,
`Test_GrubbsBeck` and `Test_MultipleGrubbsBeck` (the last two exercise the already-ported
`MultipleGrubbsBeckTest`, not this class). Transcribe the input arrays verbatim from that file --
do not retype them from any summary. The expected values and tolerances, in test order:

| C# test | call | expected | tol |
|---|---|---|---|
| `Test_OneSampleTtest` | `one_sample_t_test(data30a)` | `0.6489` | `1e-4` |
| | `one_sample_t_test(data30a, 10)` | `0.001823` | `1e-4` |
| | `one_sample_t_test({23,15,-5,7,1,-10,12,-8,20,8,-2,-5})` | `0.087585 * 2` | `1e-6` |
| `Test_EqualVarianceTtest` | `equal_variance_t_test(data30a, data30b)` | `0.0185` | `1e-4` |
| `Test_UnequalVarianceTtest` | `unequal_variance_t_test(data30a, data30b)` | `0.02043` | `1e-4` |
| `Test_PairedTtest` | `paired_t_test(d1, d2)` | `6.2e-9` | `1e-4` |
| `Test_Ftest` | `f_test(d1, d2)` | `0.1825` | `1e-4` |
| `Test_FtestModels` | `f_test_models(1224.32, 720.27, 49, 48, f, p)` | `f == 33.5899` (`1e-3`), `p == 0` (`1e-6`) | |
| `Test_JarqueBera` | `jarque_bera_test(noise128)` | `3.444e-5 / 2` | `1e-5` |
| | `jarque_bera_test({4,5,5,6,9,12,13,14,14,19,22,24,25})` | `0.592128` | `1e-6` |
| `Test_WaldWolfowitz` | `wald_wolfowitz_test(harricana69)` | `(1 - Normal::standard_cdf(1.167)) * 2` | `1e-3` |
| `Test_LjungBox` | `ljung_box_test(noise128, 5)` | `0.2314` | `1e-4` |
| | `ljung_box_test(noise128, 30)` | `0.7548` | `1e-4` |
| `Test_MannWhitney` | `mann_whitney_test(harricana_last19, harricana_first50)` | `(1 - Normal::standard_cdf(0.54)) * 2` | `1e-2` |
| `Test_MannKendall` | `mann_kendall_test(harricana69)` | `0.7757` | `1e-4` |
| `Test_LinearTrendTest` | `linear_trend_test(1..100, cos(i))` | `0.9092` | `1e-4` |

Note the Mann-Whitney call order: the SHORTER sample first, because the method throws when
`n1 > n2`. Add a supplement, clearly marked as a corehydro addition beyond the C# assertions,
asserting each guard actually throws: `one_sample_t_test` on a 1-element sample,
`equal_variance_t_test` on two 1-element samples, `paired_t_test` on unequal lengths,
`f_test` on a 1-element sample, `f_test_models` with `df_restricted == df_full`,
`mann_whitney_test` with `n1 > n2` and with a combined count of exactly 20, and
`mann_kendall_test` on 9 values.

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake -S core -B core/build && cmake --build core/build -j8 && ctest --test-dir core/build -R test_hypothesis_tests
```
Expected: compile FAILS -- the header does not exist.

- [ ] **Step 3: Implement the header**

Transcribe `Numerics/Data/Statistics/HypothesisTests.cs` line-for-line. Mapping notes, all
verified during recon, so no re-derivation is needed:

- `Statistics.MeanVariance` / `ProductMoments` / `RanksInPlace(out ties)` / `Percentile` -> Task 1.
- `Tools.Sqr` -> `tools::sqr`; `Tools.Pow(x, n)` -> Task 1's `tools::pow` (NOT `std::pow`).
- `Beta.Incomplete(a, b, x)` -> `math::special::beta::incomplete`.
- `Normal.StandardCDF(z)` -> the static on `distributions::Normal`.
- `StudentT(df)` -- `df` is a `double` at three call sites; the ported ctor already takes `double`.
- `ChiSquared(k)` -> `distributions::ChiSquared`.
- `Autocorrelation.Function(sample, lagMax, Type.Correlation)` -> `Autocorrelation::function`,
  which returns `std::optional<std::vector<std::array<double, 2>>>`. C# `acf[k, 1]` is
  `(*acf)[k][1]`; C# `null` is `std::nullopt`, and that arm returns NaN.
- `LinearTrendTest` builds `Matrix(indices)` / `Vector(sample)` and fits
  `LinearRegression(x, y, true)`, then reads `parameters()[1]` and
  `parameter_standard_errors()[1]` against `StudentT(degrees_of_freedom())`. All ported.
- `LjungBoxTest`'s default `lagMax = floor(min(10 * log10(n), n - 1))`.
- Preserve every guard and its exact message text.

The file header carries the severance note for the thirteenth method:

```cpp
// Deliberately NOT ported (P4):
//   - UnimodalityTest: it fits Numerics.MachineLearning.GaussianMixtureModel at k = 1 and k = 2
//     (each Train(12345, true)) and forms the likelihood-ratio statistic against ChiSquared(3).
//     The Machine Learning layer is unported; P5 ports it and un-gates this method together with
//     RMC.BestFit DataFrame::summary_hypothesis_test, which cannot ship without it (see
//     models/data_frame/data_frame.hpp).
```

- [ ] **Step 4: Run the ctest and watch it pass**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build -R test_hypothesis_tests
```
Expected: PASS. If a p-value misses at the C# tolerance, the port diverged -- fix the port, do not
widen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: port the Numerics hypothesis tests"
```

---

### Task 3: The `hypothesis` toolbox group and `hypothesis_test()`

**Files:**
- Create: `core/include/corehydro/numerics/support/toolbox/hypothesis.hpp`
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (include + dispatch line +
  the file-header group count: fifteen groups become sixteen, and the list gains `hypothesis`)
- Create: `fixtures/toolbox/hypothesis.json`; Modify: `fixtures/README.md`
- Modify: `tools/oracle_emitter/Program.cs` (`ToolboxDispatch` `case "hypothesis"` +
  `HypothesisDispatch`)
- Modify: `corehydror/R/toolbox.R`, `corehydror/NAMESPACE`, `corehydropy/src/corehydropy/toolbox.py`,
  `corehydropy/src/corehydropy/__init__.py`
- Modify: `corehydror/tests/testthat/test-toolbox.R`, `corehydropy/tests/test_toolbox.py`
- Modify: `corehydror/_pkgdown.yml`, `site/_quarto.yml` (Task 11 finalizes the section prose)

**Interfaces:**
- Consumes: Task 2.
- Produces, toolbox group `hypothesis`, in `corehydro::numerics::support::detail`:
  `inline ToolboxResult run_hypothesis(const std::string& method, const
  std::vector<std::vector<double>>& data, const JsonValue& options)`. Methods, with their data
  vectors and options:
  - `one_sample_t` (data `[x]`, option `population_mean`, default `0`)
  - `equal_variance_t`, `unequal_variance_t`, `paired_t`, `f` (data `[x, y]`)
  - `f_models` (NO data; options `sse_restricted`, `sse_full`, `df_restricted`, `df_full`) ->
    a NAMED two-value result, `names = {"f_statistic", "p_value"}`
  - `jarque_bera`, `wald_wolfowitz`, `mann_kendall` (data `[x]`)
  - `ljung_box` (data `[x]`, option `lag_max`, default `-1`)
  - `mann_whitney` (data `[x, y]`)
  - `linear_trend` (data `[index, x]`)

  Every method but `f_models` returns `detail::scalar(p)`.
- Produces, user-facing (identical in both languages):
  `hypothesis_test(x, y = NULL, method = "jarque_bera", population_mean = 0, lag_max = NULL,
  index = NULL, sse_restricted = NULL, sse_full = NULL, df_restricted = NULL, df_full = NULL)`
  -> a named numeric of length 1 (`p_value`), except `method = "f_models"`, which returns
  `c(f_statistic =, p_value =)`. `method` is one of `"one_sample_t"`, `"equal_variance_t"`,
  `"unequal_variance_t"`, `"paired_t"`, `"f"`, `"f_models"`, `"jarque_bera"`, `"wald_wolfowitz"`,
  `"ljung_box"`, `"mann_whitney"`, `"mann_kendall"`, `"linear_trend"`. For `"linear_trend"`,
  `index` supplies the x-axis and defaults to `seq_along(x)` (R) / `1..len(x)` (Python) -- state
  the 1-based default in BOTH help pages, since it is a value, not an index into user data.

- [ ] **Step 1: Write failing R and Python binding tests**

In `test-toolbox.R` and `test_toolbox.py`: `hypothesis_test(harricana, method = "mann_kendall")`
close to `0.7757`; the `f_models` call returning a length-2 named result; `method = "linear_trend"`
with and without an explicit `index` giving the same answer; and the argument validation (a
two-sample method with `y = NULL` errors, `method = "f_models"` with missing `sse_full` errors,
an unknown method errors) with identical message text in both languages.
Expected: FAIL (function not found).

- [ ] **Step 2: Implement the group, the dispatch line, and both verbs**

`toolbox/hypothesis.hpp` follows `toolbox/special.hpp`'s shape: standalone includes, one arm per
method, `throw std::runtime_error("unknown hypothesis method: " + method)` at the end, and a
file-header block comment naming each method's C# counterpart and the guard each one inherits.
Wire `if (group == "hypothesis") return detail::run_hypothesis(method, data, options);` into
`toolbox_runner.hpp` and update its header's group count and list.

Write both verbs with full roxygen / numpydoc. Both help pages state, per method, which arguments
it reads and what its C# guard rejects (`mann_whitney` needs `n1 <= n2`, both samples above 3, and
a combined count above 20; `mann_kendall` needs 10 values; the t/F tests need 2 per sample).

```bash
R CMD INSTALL corehydror && Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
```

- [ ] **Step 3: Fixture and emitter parity**

`fixtures/toolbox/hypothesis.json` (kind `toolbox`, group `hypothesis`) carrying one case per C#
test method from Task 2's table, reusing the same input arrays as `datasets` entries
(`harricana`, `noise`, `data30a`, `data30b`, ...) so they are written once. Every assertion
carries a `source` naming the C# test method. Use the C# tolerances; where the C# expected value
is an EXPRESSION rather than a literal (Wald-Wolfowitz's `(1 - StandardCDF(1.167)) * 2` and
Mann-Whitney's `(1 - StandardCDF(0.54)) * 2`), evaluate it once and pin the number, recording the
statistic and the expression in the assertion's `source`.

Emitter: `ToolboxDispatch` gains `case "hypothesis": return HypothesisDispatch(...)`, and
`HypothesisDispatch` drives the real `Numerics.Data.Statistics.HypothesisTests` -- reading the
same options keys, calling `ToolboxSelectFlatNoDims` for the scalar methods and reading
`index`/`label` directly for `f_models`.

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: green; `0 failed`.

- [ ] **Step 4: Documentation indexes and commit**

Add `hypothesis_test` to `corehydror/NAMESPACE`, `corehydror/_pkgdown.yml`, `site/_quarto.yml`,
`corehydropy`'s two `__all__`s and the `__init__.py` import block. Regenerate and commit
`corehydror/man/hypothesis_test.Rd`. Append the `hypothesis` group's section to
`fixtures/README.md` under `### toolbox`.

```bash
Rscript -e 'roxygen2::roxygenise("corehydror", roclets = "rd")'
git add -A && git commit -m "feat: expose the twelve hypothesis tests through a toolbox group"
```

---

### Task 4: The `Correlation` matrix overloads

**Files:**
- Modify: `core/include/corehydro/numerics/data/correlation.hpp` (add `pearson_matrix`,
  `spearman_matrix`; REPLACE the severance note at lines 6-10)
- Modify: `core/include/corehydro/numerics/support/toolbox/correlation.hpp` (two new methods)
- Modify: `fixtures/toolbox/correlation.json` (or `fixtures/special_functions/correlation.json` --
  read both first and put the new cases in the file whose `group` is `correlation`)
- Modify: `tools/oracle_emitter/Program.cs` (`CorrelationDispatch`)
- Modify: `corehydror/R/toolbox.R` + `corehydropy/src/corehydropy/toolbox.py` (`correlation()`
  gains a matrix path; both docstrings' "matrix overloads are not ported" sentence is DELETED)
- Modify: `upstream/CLAUDE.md` (the matrix-overload severance paragraph at ~lines 99-102),
  `docs/superpowers/specs/2026-08-12-numerics-toolbox-design.md` is a historical spec -- leave it
- Test: `core/tests/test_statistics_helpers.cpp` (extend)

**Interfaces:**
- Produces, in `namespace corehydro::numerics::data`:
  `inline std::vector<std::vector<double>> pearson_matrix(const std::vector<std::vector<double>>& columns)`
  and `spearman_matrix(...)`, both taking p columns of n observations and returning a p-by-p
  matrix.
- Produces, toolbox group `correlation`, two new methods `pearson_matrix` / `spearman_matrix`:
  every data vector is one COLUMN; the result is the p-by-p matrix flattened row-major with
  `dims = {p, p}`.
- Produces, user-facing: `correlation(x, y = NULL, method = "pearson")` gains the case where `y`
  is `NULL` and `x` is a matrix or data frame -- it then returns the p-by-p correlation matrix
  (with dimnames from the columns in R, a plain 2-D array in Python). `method = "kendall"` with a
  matrix errors, because upstream has no `KendallsTau(double[,])` overload; say so in the message
  and in both help pages.

- [ ] **Step 1: Write the failing ctest and binding tests**

ctest: build a 15-by-2 matrix from the existing `Test_Correlation` big dataset (`XArray`/`YArray`
in `upstream/Numerics/Test_Numerics/Data/Statistics/Test_Correlation.cs`), assert the diagonal is
exactly `1` and the off-diagonal equals the pairwise oracle already pinned for that dataset
(`0.988054377242161` for Pearson, `1` for Spearman) at `1e-10`, and assert symmetry. Then a
three-column case whose three off-diagonals each equal the corresponding pairwise call. Binding
tests: the same three-column matrix through `correlation()` in both languages, plus the
`method = "kendall"` rejection.
Expected: FAIL.

- [ ] **Step 2: Implement**

Transcribe `Correlation.Pearson(double[,])` (line 87) LITERALLY: one mean pass over all p columns,
then ONE fused pass accumulating `ss[j]` and the upper-triangular `cov[j][k]`, mirror the triangle,
then `corr[j][k] = cov[j][k] / sqrt(ss[j] * ss[k])`. It is NOT `p*(p-1)/2` calls to the pairwise
routine and the accumulation order is different, so a naive loop would drift. Guards: throw on an
empty column list with upstream's message `"Input must have at least one column."`; note in the
header that upstream has NO guard on `n == 0` (means become NaN) and mirror that.
`Correlation.Spearman(double[,])` (line 169) rank-transforms each column with the NO-TIES
`ranks_in_place` and delegates to `pearson_matrix`.

Replace `correlation.hpp`'s lines 6-10 severance note with a short note saying the matrix
overloads are now ported and that Kendall has no matrix overload upstream.

- [ ] **Step 3: Fixture and emitter parity**

Upstream has NO test for either overload and no in-library caller, so there is no literal to
transcribe. Curate with `python3 tools/verify_oracles.py --dump` against the real
`Numerics.Data.Statistics.Correlation` and record that in the case's `source` string, following
`toolbox_kind_proof`'s existing precedent in the same file. Pin at least: a 2-column case whose
off-diagonal must equal the pairwise oracle, and a 3-column case.

```bash
cmake --build core/build -j && ctest --test-dir core/build
python3 tools/verify_oracles.py
```
Expected: green; `0 failed`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: port the Pearson and Spearman correlation matrix overloads"
```

---

### Task 5: The DataFrame hypothesis-test and summary-statistics facades

**Files:**
- Modify: `core/include/corehydro/models/data_frame/data_frame.hpp` (add the facades; REWRITE the
  severance note at lines 45-52)
- Test: `core/tests/test_data_frame_facades.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 1, 2, 4.
- Produces, public on `corehydro::models::DataFrame` (names mirror the C# members):
  - `double jarque_bera_test(bool use_log10 = false) const`
  - `double ljung_box_test(int lag_max = -1, bool use_log10 = false) const`
  - `double equal_variance_t_test(int index, bool use_log10 = false) const`
  - `double unequal_variance_t_test(int index, bool use_log10 = false) const`
  - `double f_test(int index, bool use_log10 = false) const`
  - `double linear_trend_test(bool use_log10 = false) const`
  - `double wald_wolfowitz_test(bool use_log10 = false) const`
  - `double mann_whitney_test(int index, bool use_log10 = false) const`
  - `double mann_kendall_test(bool use_log10 = false) const`
  - `std::vector<std::pair<std::string, double>> summary_statistics_exact_data_only() const`
  - `std::vector<std::pair<std::string, double>> summary_statistics_all_data() const`
  - `void set_standardized_values()`

  The two summary methods return an ORDERED key/value list, not a map: C# uses
  `Dictionary<string, double>` whose insertion order is what the GUI and the API read, and the
  fixture selects by `label`, so order must be preserved deterministically.

- [ ] **Step 1: Write the failing ctest**

Create `core/tests/test_data_frame_facades.cpp` transcribing every applicable `[TestMethod]` from
`src/RMC.BestFit.Tests/DataFrame/ExactDataHypothesisTests.cs` (read it with `git -C
upstream/RMC-BestFit show`): `Test_EqualVarianceTtest` (`df.equal_variance_t_test(30)` ==
`0.0185`, `1e-4`), `Test_UnequalVarianceTtest` (`0.02043`, `1e-4`), `Test_Ftest`
(`df.f_test(10)` == `0.1825`, `1e-4`), `Test_JarqueBera` (`3.444e-5 / 2` at `1e-5`, then
`0.592128` at `1e-6`), `Test_LjungBox` (`0.2314` and `0.7548`, `1e-4`), `Test_WaldWolfowitz`
(`(1 - standard_cdf(1.167)) * 2`, `1e-3`), `Test_MannWhitney` (`df.mann_whitney_test(50)`,
`(1 - standard_cdf(0.54)) * 2`, `1e-2`), `Test_MannKendall` (`0.7757`, `1e-4`),
`Test_LinearTrendTest` (`0.9092`, `1e-4`). SKIP `Test_UnimodalityTest` (Scope decision 1).

Then transcribe both applicable methods of `src/RMC.BestFit.Tests/DataFrame/NonparametricEmpiricalTests.cs`:
`SummaryAndStandardizedValues_DuplicateValues_ReturnFiniteResults` (exact series
`{100,100,125,125,150,150,175,175,200,200,250,300}`, `calculate_plotting_positions()`, then assert
`summary_statistics_all_data()`'s `"Mean"`, `"Std Dev"`, `"Mean (of log)"` and `"50%"` are finite,
and that after `set_standardized_values()` every exact item's standardized value and standardized
log10 value are finite) and
`EmpiricalConsumers_AllValuesIdentical_ReportUnavailableResults` (ten copies of `100`;
`"Minimum" == 100`, `"Maximum" == 100`, `"Mean"` is NaN, `"50%"` is NaN, and every standardized
value is NaN).

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build -R test_data_frame_facades
```
Expected: compile FAILS -- none of the twelve members exist.

- [ ] **Step 3: Implement the facades**

Transcribe `src/RMC.BestFit/Models/DataFrame/DataFrame.cs` regions 927-1069 and 1786-1924 /
2165-2259. Every facade reads `ExactSeries` ONLY -- no censoring filter, no threshold machinery --
and selects `log10_value()` vs `value()` on `use_log10`. The two-sample splits filter on
`Data::index()`, NOT on array position. Guards: `exact_series().size() < 10` throws upstream's
exact message, except `mann_whitney_test`, which uses `< 20` and its own message; the two-sample
methods additionally throw `"Invalid index."` when either side has fewer than 2 (3 for
Mann-Whitney).

Four transcription notes the header must carry, each an upstream oddity that a "cleanup" would
silently change:
1. `LinearTrendTest` computes a local `double d` that is never used and then recomputes the same
   expression inline in the `return`. It also does NOT call `HypothesisTests.LinearTrendTest`,
   even though that static exists and is identical; it inlines `LinearRegression` + `StudentT`.
   Mirror both.
2. `SummaryStatisticsExactDataOnly` reports `Kurtosis` as `moments[3] + 3`; `SummaryStatisticsAllData`
   reports it as raw `moments[3]`. The asymmetry is upstream's.
3. `SummaryStatisticsAllData` and `SetStandardizedValues` build `values`, `logValues` and `probs`
   as three INDEPENDENTLY sorted parallel arrays -- the same quirk
   `create_empirical_distribution_with_unique_values` already documents. Carry it through.
4. `SummaryStatisticsAllData` calls `CentralMoments(1000)`; `SetStandardizedValues` calls
   `CentralMoments(200)`. Different, on purpose.
5. Both summary methods' `Count <= 2` moment guards are dead code (the outer `Count < 10` guard
   already returned). Port them anyway for structural fidelity and say why.

The twenty dictionary keys, in insertion order, are exact strings and include a non-ASCII
character: `"Record Length"`, `"Events Per Index (λ)"`, `"Low Outliers"`, `"Minimum"`,
`"Maximum"`, `"Mean"`, `"Std Dev"`, `"Skewness"`, `"Kurtosis"`, `"Mean (of log)"`,
`"Std Dev (of log)"`, `"Skewness (of log)"`, `"Kurtosis (of log)"`, `"1%"`, `"5%"`, `"25%"`,
`"50%"`, `"75%"`, `"95%"`, `"99%"`. Write the `λ` as a UTF-8 literal in the C++ source and verify
the file stays valid UTF-8 (`file core/include/corehydro/models/data_frame/data_frame.hpp`).

Replace `data_frame.hpp`'s severance note (lines 45-52) with one that records what P4 landed and
names the two members still waiting on P5 (`unimodality_test`, `summary_hypothesis_test`) and why.

- [ ] **Step 4: Run the ctest and watch it pass**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build -R test_data_frame_facades
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: un-gate the DataFrame hypothesis-test and summary-statistics facades"
```

---

### Task 6: The `data_frame` fixture kind and the two data-layer verbs

**Files:**
- Modify: `core/include/corehydro/models/data_frame_runner.hpp` (a fourth entry point)
- Modify: `core/tests/test_fixtures.cpp` (new kind), `corehydror/tests/testthat/test-fixtures.R`,
  `corehydropy/tests/test_fixtures.py`
- Create: `fixtures/data/data_frame_facades.json`; Modify: `fixtures/README.md`
- Modify: `tools/oracle_emitter/Program.cs` (a `data_frame` kind branch + `DataFrameDispatch`)
- Modify: `corehydror/src/data.cpp` + `corehydror/R/data.R`,
  `corehydropy/src/bindings/data.cpp` + `corehydropy/src/corehydropy/data.py`
- Modify: `corehydror/NAMESPACE`, `corehydror/_pkgdown.yml`, `site/_quarto.yml`,
  `corehydropy/src/corehydropy/__init__.py`
- Modify: `corehydror/tests/testthat/test-data.R`, `corehydropy/tests/test_data.py`

**Interfaces:**
- Consumes: Task 5.
- Produces, in `corehydro::models::runner`:
  `inline numerics::support::ToolboxResult run_data_frame(const std::string& method, const
  std::vector<std::vector<double>>& data, const std::string& data_frame_json, const std::string&
  options_json)`. Returning the SAME `ToolboxResult` struct the toolbox groups return is the point
  of the design: all three fixture runners already have a `toolbox_select` helper handling `index`
  / `label` / `select` / `"length"` / `"rows"` / `"columns"`, so the new kind's dispatch is a
  handful of lines per runner and no new selection code exists anywhere.

  **The two ways the frame arrives, and why there are two.** `models::spec::build_data_frame`
  (`core/include/corehydro/models/model_spec.hpp:174`) parses `exact` as an ARRAY OF OBJECTS
  (`{"index": i, "value": v, "is_low_outlier"?: bool}`), which is right for a censored or
  thresholded frame and unbearable for a 69-value systematic record. So `run_data_frame` takes
  BOTH and requires exactly one: when `data_frame_json` is empty, it builds the frame from
  `data[0]` as the exact series with sequential indices `0..n-1`; when `data_frame_json` is
  non-empty, `data` must be empty and the spec is passed to `build_data_frame` unchanged. The
  plain-vector path is what every P4 fixture case and both user verbs use; the spec path exists so
  a censored frame is still reachable. Doing the expansion HERE, in the one shared runner, is what
  keeps the three fixture runners and the emitter from each growing their own copy of it -- and
  it lets the fixture's `data` array carry dataset NAMES, resolved by each runner with the
  dataset-resolution code the `toolbox` kind already has. Do not touch `build_data_frame`.

  Methods:
  - the nine hypothesis facades, named exactly as the group in Task 3 names them, prefixed
    nothing: `jarque_bera`, `ljung_box`, `equal_variance_t`, `unequal_variance_t`, `f`,
    `linear_trend`, `wald_wolfowitz`, `mann_whitney`, `mann_kendall`. Options: `use_log10`
    (default `false`), `index` (the two-sample split, required for the four that take one),
    `lag_max` (Ljung-Box, default `-1`).
  - `summary_exact` and `summary_all` -> the twenty-key result, `values` parallel to `names`.
  - `standardized` -> after `calculate_plotting_positions()` and `set_standardized_values()`, the
    exact series' standardized values followed by its standardized log10 values, with
    `dims = {n, 2}` (column 0 = standardized value, column 1 = standardized log10 value).

  `run_data_frame` calls `calculate_plotting_positions()` before `summary_all` and `standardized`
  (the C# tests do it explicitly and the results are meaningless without it) and does NOT call it
  for the hypothesis facades or `summary_exact`, which do not read plotting positions. Say so in
  the runner's comment.
- Produces, user-facing (identical in both languages):
  - `analysis_data_hypothesis_test(data, method, index = NULL, lag_max = NULL, use_log10 = FALSE)`
    -> a named numeric of length 1.
  - `analysis_data_statistics(data, all_data = FALSE, standardized = FALSE)` -> the twenty named
    values; with `standardized = TRUE` the result gains `standardized_value` and
    `standardized_log10_value` vectors parallel to `analysis_data_summary()`'s `value`.

- [ ] **Step 1: Write the failing fixture, then the failing binding tests**

Write `fixtures/data/data_frame_facades.json` first, with `"kind": "data_frame"`. Shape (this is
the schema; document it in `fixtures/README.md` as a new `### data_frame` section):

```jsonc
{
  "kind": "data_frame",
  "source": "src/RMC.BestFit/Models/DataFrame/DataFrame.cs @ c2e6192; values scraped from Test_...",
  "datasets": { "harricana": [ ...numbers... ] },
  "cases": [
    {
      "name": "harricana_mann_kendall",
      "data": ["harricana"],        // dataset names or inline arrays, resolved exactly as the
                                    // `toolbox` kind resolves its `data`; data[0] becomes the
                                    // exact series with sequential indices
      "options": { "use_log10": false },
      "assertions": [
        { "method": "mann_kendall", "expected": 0.7757, "mode": "abs", "tol": 1e-4,
          "source": "ExactDataHypothesisTests.Test_MannKendall" }
      ]
    }
  ]
}
```

Cases: one per C# test method transcribed in Task 5, plus a `summary_all` case selecting several
keys by `label` (`"Mean"`, `"50%"`, `"Minimum"`), a `summary_exact` case, and a `standardized`
case selecting by flat `index` with `select: "rows"` and `select: "columns"` assertions. The
`summary_exact` values have NO upstream test -- curate them with `verify_oracles.py --dump` and
say so in the case `source`.

Binding tests in `test-data.R` / `test_data.py`: `analysis_data_hypothesis_test` on the Harricana
peaks reproducing `0.7757`, `analysis_data_statistics(all_data = TRUE)` returning twenty named
values in the documented order, and the argument validation (an unknown method, a two-sample
method with no `index`).
Expected: every runner FAILS on the unknown kind; the verbs do not exist.

- [ ] **Step 2: Implement the runner entry point and the three fixture-runner arms**

`run_data_frame` builds the frame with the existing `build_data_frame_from_json` and dispatches by
method name. The three fixture-runner arms, each modeled on the `toolbox` arm directly above it:

C++ (`core/tests/test_fixtures.cpp`), beside `run_toolbox_kind`:
```cpp
static void run_data_frame_kind(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        json df = resolve_dataset_names(c["data_frame"], datasets);
        json options = c.contains("options") ? c["options"] : json::object();
        for (const auto& as : c["assertions"]) {
            std::string where = "data_frame/" + name;
            auto r = corehydro::models::runner::run_data_frame(
                as["method"].get<std::string>(), df.dump(), options.dump());
            check_value(toolbox_select(r, as, "data_frame"), as, where);
        }
    }
}
```
and one `else if (kind == "data_frame") run_data_frame_kind(spec);` in `main`. R and Python get
the same shape (R: `ns$ch_data_frame_run_(...)` then the existing `toolbox_select`; Python:
`_core.data_frame_run(...)` then `_toolbox_select`, and `"data_frame"` joins the kind allow-list
in `test_fixtures.py`).

Add `ch_data_frame_run_` to `corehydror/src/data.cpp` (re-run `cpp11::cpp_register`) and
`data_frame_run` to `corehydropy/src/bindings/data.cpp`, both packing the `ToolboxResult` exactly
as the toolbox glue does (`values`/`names`/`dims`/`spec`).

- [ ] **Step 3: Write the two verbs and the emitter driver**

The verbs in `corehydror/R/data.R` and `corehydropy/src/corehydropy/data.py`, full roxygen /
numpydoc, both stating that the tests read the EXACT series only and that `index` splits on the
data index rather than on position.

Emitter: a `kindStr == "data_frame"` branch mirroring the `toolbox` branch (dataset resolution,
`--dump` support, `Compare`) plus `DataFrameDispatch`, which builds a real
`BestFitModels.DataFrame` through the existing `BuildSpecDataFrame` helper and calls the real C#
facades.

```bash
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
cmake --build core/build -j && ctest --test-dir core/build
python3 tools/verify_oracles.py
```
Expected: all green; gate `0 failed`.

- [ ] **Step 4: Documentation indexes and commit**

Both verbs into `NAMESPACE`, `_pkgdown.yml`, `site/_quarto.yml`, and Python's two `__all__`s plus
the `__init__.py` import block. Regenerate the two `.Rd` files.

```bash
git add -A && git commit -m "feat: expose the DataFrame hypothesis tests and summary statistics"
```

---

### Task 7: `Ordinate` and `LineSimplification`

**Files:**
- Create: `core/include/corehydro/numerics/data/paired_data/ordinate.hpp`
- Create: `core/include/corehydro/numerics/data/paired_data/line_simplification.hpp`
- Test: `core/tests/test_ordinate.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:**
- Produces, in `namespace corehydro::numerics::data::paired_data`:
  ```cpp
  struct Ordinate {
      double x = 0.0;
      double y = 0.0;
      bool is_valid = false;

      Ordinate() = default;
      Ordinate(double x_value, double y_value);   // computes is_valid: false if either is inf/nan

      bool ordinate_valid(const Ordinate& compare, bool strict_x, bool strict_y,
                          interpolation::SortOrder x_order, interpolation::SortOrder y_order,
                          bool compare_ordinate_is_next) const;
      std::vector<std::string> ordinate_errors(const Ordinate& compare, bool strict_x,
                                               bool strict_y, interpolation::SortOrder x_order,
                                               interpolation::SortOrder y_order,
                                               bool compare_ordinate_is_next) const;
      std::vector<std::string> ordinate_errors() const;
      Ordinate transform(interpolation::Transform x_transform,
                         interpolation::Transform y_transform) const;
  };
  bool operator==(const Ordinate& l, const Ordinate& r);
  bool operator!=(const Ordinate& l, const Ordinate& r);
  ```
  and, in `namespace corehydro::numerics::data::paired_data::line_simplification`:
  ```cpp
  void ramer_douglas_peucker(const std::vector<Ordinate>& ordinates, double epsilon,
                             std::vector<Ordinate>& output);
  double perpendicular_distance(const Ordinate& pt, const Ordinate& line_start,
                                const Ordinate& line_end);
  ```
  `SortOrder` and `Transform` are the ALREADY-PORTED enums at
  `numerics/data/interpolation/sort_order.hpp` and `.../transform.hpp`. Do not define new ones.

- [ ] **Step 1: Write the failing ctest**

Create `core/tests/test_ordinate.cpp` transcribing
`upstream/Numerics/Test_Numerics/Data/Paired Data/Test_Ordinate.cs` (skip `Test_ToXElement`) and
`.../Test_LineSimplification.cs` in full.

`Test_Transform`, tolerance `1e-6`: `Ordinate(50, 100).transform(Logarithmic, None).x == 1.69897`;
`.transform(None, Logarithmic).y == 2`; `.transform(Logarithmic, Logarithmic)` gives both.
`Ordinate(0.3, 0.8).transform(NormalZ, None).x == -0.5244005`; `.transform(None, NormalZ).y ==
0.8416212`; `.transform(NormalZ, NormalZ)` gives both.

`Test_OrdinateValid`: subject `(2, 4)`; comparands `(3,3)`, `(3,5)`, `(1,3)`, `(1,5)`. Six
argument sets, each a four-element expected array over comparands 1..4:

| strict_x | strict_y | x_order | y_order | is_next | expected |
|---|---|---|---|---|---|
| true | true | Ascending | Ascending | true | `{false, true, false, false}` |
| true | true | Ascending | Ascending | false | `{false, false, true, false}` |
| false | true | None | Ascending | true | `{false, true, false, true}` |
| true | false | Ascending | None | true | `{true, true, false, false}` |
| false | false | Descending | Ascending | true | `{false, false, false, true}` |
| false | false | Ascending | Descending | true | `{true, false, false, false}` |

`Test_OrdinateErrors`: the message vectors are ORDER-SENSITIVE. Subject `(2,4)`; with `(2,4)`,
`strict_x = strict_y = true`, `Ascending`/`Descending`, `is_next = true` ->
`{"Y values must be strictly decreasing.", "X values must be strictly increasing."}`. With
`(3,5)`, all false, `Descending`/`Descending`, next -> `{"Y values must decrease.", "X values must
decrease."}`. With `(1,3)`, all false, `Ascending`/`Ascending`, next -> `{"Y values must
increase.", "X values must increase."}`. With `(1,3)`, all false, `Descending`/`Descending`, NOT
next -> the decrease pair. With `(3,5)`, all false, `Ascending`/`Ascending`, NOT next -> the
increase pair.

`Test_Equality`: `(7.325, 6.389)` vs `(7.325, 6.389)` vs `(8.36, 25.99)` -> `{true, false, true,
false}`. `Test_Construction` additionally pins the documented NaN quirk: `Ordinate(NaN, 4)`
compares EQUAL to another ordinate, because the operator tests `fabs(diff) > eps` and that is
false for NaN. Assert it and mark it as the pinned upstream quirk.

`Test_LineSimplification`, tolerance `1e-6`: `Test_RamerDouglasPeucker` (five `sin` samples on the
`3.14` grid, `epsilon = 0.1`, expecting `{(0,0), (1.57,1), (4.71,-1), (6.28,0)}` -- the middle
point drops); `Test_ZeroEpsilon` (six scattered points, `epsilon = 0`, all six survive);
`Test_Line` (six collinear points, `epsilon = 0.1`, endpoints only); `Test_EqualPoints` (six
identical `(1,30)` points, `epsilon = 0.1`, endpoints only -- this is the case the `mag > 0.0`
guard exists for). Transcribe every input array verbatim from the C# file.

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake -S core -B core/build && cmake --build core/build -j8 && ctest --test-dir core/build -R test_ordinate
```
Expected: compile FAILS.

- [ ] **Step 3: Implement both headers**

Transcribe `Numerics/Data/Paired Data/Ordinate.cs` and `.../LineSimplification.cs`. `Tools.Log10`
and `Normal.StandardZ` / `StandardCDF` are already ported. Omit the two `XElement` members
(constructor and `ToXElement`) and record the XML severance in the header, matching how every
other ported file records it.

Two header notes required:
1. `operator==` compares with `fabs(diff) > tools::kDoubleMachineEpsilon`, which makes a NaN
   coordinate compare EQUAL to anything. The C# source carries a comment saying so and
   `Test_Construction` asserts it -- it is pinned behavior, not a bug to fix.
2. `LineSimplification::perpendicular_distance` normalizes the segment direction and guards the
   degenerate segment (`mag > 0.0`), returning the point-to-start distance. Task 8's
   `OrderedPairedData::perpendicular_distance` is a DIFFERENT formula (triangle area over base)
   with NO degenerate guard. The two are independent implementations upstream; keep both and say
   in each header that the other exists.

- [ ] **Step 4: Run the ctest and watch it pass, then commit**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build -R test_ordinate
git add -A && git commit -m "feat: port the Ordinate struct and the Ramer-Douglas-Peucker simplifier"
```

---

### Task 8: `OrderedPairedData`, and the six un-severed `Search.cs` overloads

**Files:**
- Create: `core/include/corehydro/numerics/data/paired_data/ordered_paired_data.hpp`
- Modify: `core/include/corehydro/numerics/data/interpolation/support/search.hpp` (the six
  overloads; REWRITE its severance note)
- Test: `core/tests/test_ordered_paired_data.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 7.
- Produces, in `namespace corehydro::numerics::data::paired_data`, `class OrderedPairedData` with
  the C# member set, `snake_case`d: the three data constructors (flag-only, the two-vector
  `(x_data, y_data, strict_x, order_x, strict_y, order_y)`, and the `std::vector<Ordinate>` one);
  `count()`, `operator[]`, `is_valid()`, the four revalidating flag accessors (`strict_x`,
  `strict_y`, `order_x`, `order_y` with setters), `x_search_start()` / `y_search_start()` /
  `use_smart_search()` with setters; `index_of` (both overloads), `get_errors()`, `add`, `insert`,
  `remove`, `remove_at`, `remove_range`, `clear`, `contains`, `clone`, `invert`; the numerical
  set `trapezoidal_area_under_y()`, `trapezoidal_area_under_x()`, `get_y_from_x` (scalar and
  vector), `get_x_from_y` (scalar and vector); the search set `search_x`, `search_y`,
  `binary_search_x`, `binary_search_y`, `sequential_search_x`, `sequential_search_y`,
  `bisection_search_x`, `bisection_search_y`, `hunt_search_x`, `hunt_search_y`; and the three
  simplifiers `douglas_peucker_simplify(double tolerance)`,
  `visvaligam_whyatt_simplify(int num_to_keep)`, `lang_simplify(double tolerance, int look_ahead)`.
  Omit the XML constructor, `SaveToXElement`, the `CollectionChanged` event,
  `SuppressCollectionChanged` and `RaiseCollectionChangedReset` (the observable-collection
  plumbing), recording each in the header's severance note.
- Produces, in the existing `search` namespace: the three `OrderedPairedData` overloads and the
  three `std::vector<Ordinate>` overloads of `sequential`, `bisection` and `hunt`, transcribed
  from `Numerics/Data/Interpolation/Support/Search.cs` lines 167, 254, 444, 545, 782, 925.

- [ ] **Step 1: Write the failing ctest**

Create `core/tests/test_ordered_paired_data.cpp` transcribing
`upstream/Numerics/Test_Numerics/Data/Paired Data/Test_PairedData.cs`,
`.../Test_PairedDataInterpolation.cs`, and `.../Test_PairedDataLineSimplification.cs` (skip both
`Test_ReadWriteXElement`).

The shared 15-point reservoir fixture (`xVals` 230408..1152038, `yVals` 1519.7..1552.7) builds
four datasets: ascending/ascending, descending-x/ascending-y, ascending-x/descending-y, and
descending/descending. `Test_GetY` sweeps the 78-element `QUERY_X` array and asserts the 78-element
expected-Y array at `1e-5` -- datasets 1 and 4 share one expectation array, datasets 2 and 3 share
the other. `Test_GetX` is the same three arrays with the roles swapped. Transcribe all three
78-element arrays verbatim from the C# file; do not retype them.

`Test_TrapezoidalArea`, tolerance `1` (absolute -- the magnitudes are ~1e9):
dataset 1 and 4 -> under-Y `1413175623`, under-X `25442742`; dataset 2 and 3 -> under-Y
`1410070832`, under-X `17073185`.

`Test_Sequential` / `Test_Bisection` / `Test_Hunt`: a 1000-point identity curve `Ordinate(i, i)`,
query `872.5`. Ascending (`i = 1..1000`): `search_x -> 871`, `search_y -> 871`. Descending
(`i = 1000..1`): both `-> 127`. All three algorithms must agree on all four.

The fourteen interpolation tests, all `1e-6`, on the linear curve
`x = {50,100,150,200,250}`, `y = {100,200,300,400,500}` and the probability curve
`x = {0.05,0.1,0.15,0.2,0.25}`, `y = {0.1,0.2,0.3,0.4,0.5}`:

| x transform | y transform | query | expected y |
|---|---|---|---|
| None | None | `75` | `150.0` |
| None | Logarithmic | `75` | `141.42135623731` |
| Logarithmic | None | `75` | `158.496250072116` |
| Logarithmic | Logarithmic | `75` | `150.0` |
| None | NormalZ | `0.18` | `0.358762529` |
| NormalZ | None | `0.18` | `0.362146174` |
| NormalZ | NormalZ | `0.18` | `0.36093855992815` |

Each also asserts the inverse round-trip recovers the query. The seven `Rev*` tests reverse both
arrays with `Descending` on both axes and assert the IDENTICAL expected values -- the sort-order
invariance property. `Test_Lin_List` additionally asserts the vector overloads reproduce the
curve's own arrays FROM INDEX 1 (index 0 is the clamped endpoint and is deliberately skipped).

The three simplification tests share the five-point `sin` curve and the same expected four points
`{(0,0), (1.57,1), (4.71,-1), (6.28,0)}`: `douglas_peucker_simplify(0.01)`,
`visvaligam_whyatt_simplify(4)`, `lang_simplify(0.01, 2)`. **Supplement, clearly marked as a
corehydro addition:** the C# loops are bounded by `test.Count`, the RESULT length, so an
implementation returning too few points passes upstream. Assert the result LENGTH is exactly 4
first, then compare elementwise. (Same class of weak assertion P3 supplemented in the optimizer
suites.)

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build -R test_ordered_paired_data
```
Expected: compile FAILS.

- [ ] **Step 3: Implement `OrderedPairedData`**

Transcribe `Numerics/Data/Paired Data/OrderedPairedData.cs`. The interpolation heart is
`BaseInterpolate`: linear interpolation in transformed space with back-transform
(`pow(10, .)` or `Normal::standard_cdf(.)`), returning the lower endpoint when the transformed
denominator is zero. `GetYFromX` clamps out-of-range queries to the endpoint Y, throws when
`order_x == None`, and returns NaN on an empty collection.

**Five numbered transcription notes the header must carry, each an upstream defect or aliasing
that a "cleanup" would silently change. All five are also new entries for
`docs/upstream-csharp-issues.md` (Task 11 writes them up):**
1. `RemoveRange(index, count)` has TWO bugs in five lines: the guard is
   `index < 0 || (index + count) >= Count` where `>` is correct, so removing the trailing
   element(s) is a silent no-op; and the collection loop runs `for (i = index; i < count; i++)`
   rather than `i < index + count`. The Uncertain twin (Task 9) has NEITHER bug. `Test_Indexing`
   calls `RemoveRange(0, 3)` on 13 elements, which sidesteps both.
2. `SequentialSearchY` reads `XSearchStart` where `YSearchStart` is meant. With both starts at 0
   the results agree, which is why `Test_Sequential` passes.
3. `XdeltaStart` / `YdeltaStart` are declared, initialized to `0`, and NEVER assigned, so
   `Xcorrelated` is true only when consecutive searches land on the exact same index and
   `UseSmartSearch` almost always takes the Bisection branch rather than Hunt. Contrast
   `Interpolater.cs:46`, which sets `deltaStart = min(1, (int)pow(Count, 0.25))`.
4. `Add` assigns `IsValid = OrdinateValid(Count - 1)` unconditionally, so appending a good point
   can flip an invalid collection back to VALID; the Uncertain twin only ever assigns false. The
   private `OrdinateValid(int)` likewise returns `true` here for an out-of-range index and
   `false` there.
5. `LangSimplify` returns `this` (an ALIAS, not a copy) when
   `lookAhead <= 1 || tolerance <= 0`; every other simplifier returns a fresh object. In C++ this
   means returning a copy of `*this` is a DIVERGENCE -- mirror the aliasing by documenting that
   the port returns a clone and that the C# aliasing has no C++ analogue for a value-returning
   signature, or return a clone and say so explicitly. Pick one, write down which, and make the
   ctest assert the chosen behavior.

Also note `DouglasPeuckerReduction`'s `indexFarthest != 0` sentinel and the loop starting at
`firstPoint` (whose self-distance is 0) as harmless-but-latent, and that the class `<remarks>`
advertising a `Transform()` method that "transforms one OrderedPairedData with another" is STALE
-- no such method exists in v2.1.4. Do not invent it.

Then port the six `Search.cs` overloads and replace that file's severance note, which currently
reads that they are unported because "no caller in this port's scope needs them". The new note
records that Paired Data landed in P4 and they are now ported. Cross-check them in the ctest
against `OrderedPairedData`'s own search methods on the same 1000-point curve.

- [ ] **Step 4: Run the ctest and watch it pass, then commit**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build -R test_ordered_paired_data
git add -A && git commit -m "feat: port OrderedPairedData and un-sever the Search paired-data overloads"
```

---

### Task 9: `UncertainOrdinate`, `UncertainOrderedPairedData`, and `TabularFunction`

**Files:**
- Create: `core/include/corehydro/numerics/data/paired_data/uncertain_ordinate.hpp`
- Create: `core/include/corehydro/numerics/data/paired_data/uncertain_ordered_paired_data.hpp`
- Create: `core/include/corehydro/numerics/functions/tabular_function.hpp`
- Modify: `core/include/corehydro/numerics/functions/i_univariate_function.hpp` (its header note
  pointing at the unported TabularFunction), `upstream/CLAUDE.md` (the TabularFunction severance)
- Test: `core/tests/test_uncertain_paired_data.cpp`; Modify: `core/CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 7 and 8.
- Produces, in `namespace corehydro::numerics::data::paired_data`:
  ```cpp
  struct UncertainOrdinate {
      double x = 0.0;
      std::unique_ptr<distributions::UnivariateDistributionBase> y;   // may be null
      bool is_valid = false;

      Ordinate get_ordinate(double probability) const;   // Ordinate(x, y->inverse_cdf(p))
      Ordinate get_ordinate() const;                     // Ordinate(x, y->mean())
      bool ordinate_valid(const UncertainOrdinate& compare, bool strict_x, bool strict_y,
                          interpolation::SortOrder x_order, interpolation::SortOrder y_order,
                          bool compare_ordinate_is_next, bool allow_different_types = false) const;
      std::vector<std::string> ordinate_errors(...same args...) const;
      std::vector<std::string> ordinate_errors() const;
  };
  ```
  and `class UncertainOrderedPairedData` with the two data constructors (the `(x_data,
  y_distributions, strict_x, order_x, strict_y, order_y, distribution_type)` one DEEP-CLONES each
  Y), `allow_different_distribution_types()` with setter, `is_valid()`, the four flag accessors,
  `distribution()`, `count()`, `operator[]`, `curve_sample(double probability)` and
  `curve_sample()` both returning an `OrderedPairedData`, `clone()`, `validate()` (PUBLIC here,
  private in the twin), `get_errors()`, and the list operations `add`, `add_range`, `insert`,
  `insert_range`, `remove`, `remove_at`, `remove_range` (both overloads), `clear`, `contains`,
  `index_of`.
- Produces, in `namespace corehydro::numerics::functions`, `class TabularFunction :
  public IUnivariateFunction` with `TabularFunction(UncertainOrderedPairedData paired_data)`,
  `paired_data()`, `x_transform()` / `y_transform()` with setters, `is_deterministic()` with a
  setter that REBUILDS the paired data as `Deterministic`, `confidence_level()` with a setter that
  re-runs `curve_sample`, `allow_negative_y_values()`, `set_parameters` (which THROWS, as C#
  does), `validate_parameters`, `function(double x)` -> `opd.get_y_from_x`, and
  `inverse_function(double y)` -> `opd.get_x_from_y`.

- [ ] **Step 1: Write the failing ctest**

Create `core/tests/test_uncertain_paired_data.cpp` transcribing
`upstream/Numerics/Test_Numerics/Data/Paired Data/Test_UncertainOrdinate.cs` and
`.../Test_UncertainPairedData.cs` (skip both `Test_ToXElement` / `Test_ReadWriteXElement`), plus
`Test_Tabular_Function` from `.../Functions/Test_Functions.cs`.

`Test_OrdinateValid`: subject `UncertainOrdinate(3, Triangular(6, 8, 12))`; comparands
`(0, Triangular(1,2,3))`, `(2, Triangular(2,4,5))`, `(5, Triangular(13,19,20))`,
`(6, Uniform(7,14))`. Eight argument sets, each a four-element expected array:

| strict_x | strict_y | x_order | y_order | is_next | allow_diff | expected |
|---|---|---|---|---|---|---|
| false | true | Ascending | Ascending | true | false | `{false, false, true, false}` |
| false | true | Ascending | Ascending | false | false | `{true, true, false, false}` |
| true | true | Ascending | Ascending | true | false | `{false, false, true, false}` |
| false | false | Ascending | Ascending | true | false | `{false, false, true, false}` |
| false | true | Descending | Ascending | true | false | `{false, false, false, false}` |
| false | true | Ascending | Descending | true | false | `{false, false, false, false}` |
| false | true | Descending | Descending | true | false | `{true, true, false, false}` |
| false | true | Ascending | Ascending | true | **true** | `{false, false, true, true}` |

The last row against the first is the `allow_different_types` isolation: only the `Uniform`
comparand flips.

`Test_OrdinateErrors`: subject `(3, Triangular(6,8,12))`. Against
`(+inf, Uniform(7,14))` with both strict, Ascending/Ascending, next ->
`{"Ordinate X value can not be infinity.", "Can't compare two ordinates with different
distribution types."}`. Against `(2, Triangular(2,4,5))` -> the increase pair repeated THREE
times (six messages: the three quantile probes each contribute both). Against
`(5, Triangular(13,19,20))` with Descending/Descending -> the decrease pair repeated three times.

`Test_CurveSample` (`1e-5`): fixture `x = {1,2,3,5}` with
`{Triangular(1,2,3), Triangular(2,4,5), Triangular(6,8,12), Triangular(13,19,20)}`. The mean
overload gives `y = {2, 3.66667, 8.66667, 17.33333}`; `curve_sample(0.5)` gives
`{2, 3.732051, 8.535898, 17.58258}`. Assert against all four dataset orderings.

`Test_Tabular_Function`: `x = {50,100,150,200,250}` with five `Deterministic` distributions
`{100,200,300,400,500}`, `x_transform = Logarithmic`, `y_transform = None`. `function(50.0) ==
100.0` exactly; `function(75.0) == 100 + (200 - 100) / (log(100) - log(50)) * (log(75) - log(50))`
at `1e-6`; `inverse_function(that value) == 75` at `1e-6`. **Supplement, marked as a corehydro
addition:** the C# test at ~line 210 re-asserts `X` instead of `X2` (a copy-paste bug), so assert
the second round-trip the C# meant to.

- [ ] **Step 2: Run it and watch it fail**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build -R test_uncertain_paired_data
```
Expected: compile FAILS.

- [ ] **Step 3: Implement the three headers**

Transcribe `UncertainOrdinate.cs`, `UncertainOrderedPairedData.cs` and
`Numerics/Functions/TabularFunction.cs`. Omit the XML constructors / `ToXElement` /
`SaveToXElement`, the `CollectionChanged` event, and the contiguity bookkeeping inside
`RemoveRange(int[])` that exists only to select an event-args overload; record each in the header.

**Four numbered transcription notes required:**
1. `UncertainOrderedPairedData::Validate()` EARLY-RETURNS when `SuppressCollectionChanged` is
   true. That flag is otherwise pure observable plumbing, but here it is behavioral: collapsing it
   away changes WHEN `_isValid` is recomputed. Port the flag as a plain bool with the same
   early-return and say why it survived the plumbing cull.
2. `UncertainOrdinate::ordinate_valid` probes at `min_percentile`, the **mean**, and
   `1 - min_percentile`, while `ordinate_errors` probes at `min_percentile`, the **median**
   (`get_ordinate(0.5)`), and `1 - min_percentile`. For a skewed Y the two disagree, so a curve
   can be reported invalid with no explaining message. `min_percentile` is `0.05` when the type is
   `PertPercentile` or `PertPercentileZ`, else `1e-5`.
3. `UncertainOrdinate::operator==` compares `x` with EXACT inequality (no epsilon), unlike
   `Ordinate::operator==`, then delegates to the distribution's own comparison.
4. `AddRange` sets `startIndex = Count - 1` (off by one) and `InsertRange` checks
   `OrdinateValid(index)` every iteration instead of `OrdinateValid(i)`. Both only feed the
   severed event args, so they are inert here -- port the loops as written and note that the
   consequence is severed, not that the code is correct.

`TabularFunction` holds an `UncertainOrderedPairedData` plus a sampled `OrderedPairedData`; its
`is_deterministic` setter rebuilds the paired data with `Deterministic` distributions and its
`confidence_level` setter re-runs `curve_sample`. Update `i_univariate_function.hpp`'s header note
and `upstream/CLAUDE.md`'s TabularFunction severance paragraph to record that it landed.

- [ ] **Step 4: Run the ctest and watch it pass, then commit**

```bash
cmake --build core/build -j8 && ctest --test-dir core/build -R test_uncertain_paired_data
git add -A && git commit -m "feat: port UncertainOrderedPairedData and the TabularFunction"
```

---

### Task 10: The `paired_data` toolbox group and its four verbs

**Files:**
- Create: `core/include/corehydro/numerics/support/toolbox/paired_data.hpp`
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (include + dispatch +
  header count: sixteen groups become seventeen)
- Modify: `core/include/corehydro/numerics/support/toolbox/functions.hpp` (the `tabular` arm)
- Create: `fixtures/toolbox/paired_data.json`; Modify: `fixtures/toolbox/functions.json`,
  `fixtures/README.md`
- Modify: `tools/oracle_emitter/Program.cs` (`case "paired_data"` + `PairedDataDispatch`;
  `FunctionsDispatch` gains the tabular arm)
- Modify: `corehydror/R/toolbox.R`, `corehydror/NAMESPACE`,
  `corehydropy/src/corehydropy/toolbox.py`, `corehydropy/src/corehydropy/__init__.py`
- Modify: `corehydror/tests/testthat/test-toolbox.R`, `corehydropy/tests/test_toolbox.py`
- Modify: `corehydror/_pkgdown.yml`, `site/_quarto.yml`

**Interfaces:**
- Consumes: Tasks 7, 8, 9.
- Produces, toolbox group `paired_data`, in `corehydro::numerics::support::detail`:
  `inline ToolboxResult run_paired_data(const std::string& method, const
  std::vector<std::vector<double>>& data, const JsonValue& options)`. Every method takes the curve
  as data vectors `[x, y]` and its shape flags in options: `strict_x` / `strict_y` (bool, default
  `true`), `order_x` / `order_y` (the `SortOrder` names `"ascending"` / `"descending"` / `"none"`,
  default `"ascending"`), `x_transform` / `y_transform` (the `Transform` names `"none"` /
  `"logarithmic"` / `"normal_z"`, default `"none"`). Methods:
  - `interpolate_y` (data `[x, y, xout]`) -> one value per `xout`
  - `interpolate_x` (data `[x, y, yout]`) -> one value per `yout`
  - `area_under_y`, `area_under_x` (data `[x, y]`) -> scalar
  - `simplify` (data `[x, y]`; options `algorithm` = `"rdp"` / `"visvalingam"` / `"lang"`,
    `tolerance`, `num_to_keep`, `look_ahead`) -> the simplified curve, `dims = {n, 2}`
  - `line_simplify` (data `[x, y]`; option `epsilon`) -> the standalone
    `LineSimplification::ramer_douglas_peucker`, `dims = {n, 2}`
  - `search` (data `[x, y]`; options `value`, `axis` = `"x"` / `"y"`, `algorithm` =
    `"smart"` / `"sequential"` / `"bisection"` / `"hunt"` / `"binary"`) -> the index, as a scalar.
    Stateless: the object is built, searched once, and dropped -- say so in the header, because
    the C# search state (`XSearchStart`, `Xcorrelated`) is call-order-dependent and cannot cross
    the boundary meaningfully.
  - `is_valid` (data `[x, y]`) -> a NAMED two-value result `{"is_valid", "error_count"}`. The
    message strings do NOT cross (Scope decision 5).
  - `curve_sample` (data `[x]`; options `distributions` = an ARRAY of `dist_spec` grammar objects
    `{"family": ..., "parameters": [...]}`, optional `probability`, plus the four shape flags and
    `distribution_type`) -> the sampled curve, `dims = {n, 2}`. Building each Y through
    `distributions::spec::build_distribution` keeps the "one place a distribution is built from a
    spec" invariant; do NOT write a second distribution parser here.
- Produces, `functions` group method `tabular` / `tabular_inverse`: same `distributions` array
  plus `x`, the two transforms, `confidence_level`, `is_deterministic`, `allow_negative_y_values`,
  and the evaluation points as data.
- Produces, user-facing (identical in both languages):
  - `curve_interpolate(x, y, xout = NULL, yout = NULL, x_transform = "none", y_transform = "none",
    order_x = "ascending", order_y = "ascending", strict_x = TRUE, strict_y = TRUE)` -> numeric.
    Exactly one of `xout` / `yout` must be supplied.
  - `curve_area(x, y, under = "y", ...)` -> a single number.
  - `curve_simplify(x, y, method = "rdp", tolerance = NULL, num_to_keep = NULL, look_ahead = NULL,
    ...)` -> a two-column table (R: a data frame with columns `x` and `y`; Python: an `(n, 2)`
    numpy array with the documented column order).
  - `uncertain_curve_sample(x, distributions, probability = NULL, ...)` -> the same two-column
    table. `distributions` is a list of `distribution()` objects (R) / `Distribution` objects
    (Python), or one distribution recycled across all `x`.
  - `tabular_function(x, distributions, at, inverse = FALSE, x_transform = "none",
    y_transform = "none", confidence_level = NULL, allow_negative_y_values = FALSE)` -> numeric.

- [ ] **Step 1: Write failing R and Python binding tests**

`curve_interpolate` on the linear curve reproducing `150.0` at `x = 75` and `158.496250072116`
with `x_transform = "logarithmic"`; `curve_area` on the reservoir curve reproducing `1413175623`
within `1`; `curve_simplify` on the five-point `sin` curve returning exactly four rows for all
three methods; `uncertain_curve_sample` reproducing `{2, 3.732051, 8.535898, 17.58258}` at
`probability = 0.5`; `tabular_function` reproducing `100.0` at `50`. Plus the validation errors
(both `xout` and `yout` supplied; an unknown `method`; a `distributions` list whose length is
neither 1 nor `length(x)`), with identical message text in both languages.
Expected: FAIL.

- [ ] **Step 2: Implement the group, the dispatch line, and the five verbs**

`toolbox/paired_data.hpp` follows `toolbox/network.hpp`'s shape: local helpers for the shape flags
(`paired_data_sort_order(name)`, `paired_data_transform(name)`) with messages naming the accepted
values, a `curve_result(opd)` serializer setting `dims = {n, 2}`, one arm per method, and a
file-header block comment covering the data layout, every option key, and Scope decision 5 (why
validation messages do not cross). Wire
`if (group == "paired_data") return detail::run_paired_data(method, data, options);` into
`toolbox_runner.hpp` and update its header count and list.

```bash
R CMD INSTALL --preclean corehydror && Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
```

- [ ] **Step 3: Fixture and emitter parity**

`fixtures/toolbox/paired_data.json` (kind `toolbox`, group `paired_data`) carrying: the four
reservoir-curve datasets' interpolation sweeps reduced to a representative subset of the 78 query
points (pin at least 10, spread across the range and including both clamped endpoints), the four
trapezoidal-area values at `mode: "abs"`, `tol: 1`, the seven transform-combination interpolation
values at `1e-6` in BOTH sort orders, the three simplification results (assert `select: "rows"`
== 4 AND the eight flattened coordinates at `mode: "equal"`), the search indices `871` and `127`,
and the two `curve_sample` expectation vectors. Add the `tabular` cases to
`fixtures/toolbox/functions.json`. Every assertion carries a `source` naming its C# test method.

Emitter: `ToolboxDispatch` gains `case "paired_data"`, and `PairedDataDispatch` drives the real
`Numerics.Data.OrderedPairedData` / `UncertainOrderedPairedData` / `LineSimplification`, building
each Y distribution through `UnivariateDistributionFactory` from the same `distributions` array.
`FunctionsDispatch` gains the tabular arm over the real `Numerics.Functions.TabularFunction`. If
`Numerics.Data` is not already in `Program.cs`'s `using` list, add it.

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: green; `0 failed`.

- [ ] **Step 4: Documentation indexes and commit**

The five verbs into `NAMESPACE`, `_pkgdown.yml`, `site/_quarto.yml`, Python's two `__all__`s and
the `__init__.py` import block; regenerate the five `.Rd` files; append the `paired_data` section
to `fixtures/README.md`.

```bash
Rscript -e 'roxygen2::roxygenise("corehydror", roclets = "rd")'
git add -A && git commit -m "feat: expose paired-data curves through a toolbox group"
```

---

### Task 11: Documentation, upstream-issues write-up, and worked example 20

**Files:**
- Modify: `docs/upstream-csharp-issues.md` (the new entries)
- Modify: `corehydror/_pkgdown.yml`, `site/_quarto.yml` (final section titles and prose)
- Create: `site/examples/20-hypothesis-tests-and-paired-data/{python.ipynb, r.qmd}`
- Modify: `site/examples/index.qmd`, `site/status.qmd`
- Modify: `upstream/CLAUDE.md` (severances retired: TabularFunction, the Search overloads, the
  Correlation matrix overloads; severances added: UnimodalityTest / SummaryHypothesisTest to P5)

**Interfaces:**
- Consumes: everything shipped in Tasks 1-10.

- [ ] **Step 1: Write up the upstream findings**

New entries in `docs/upstream-csharp-issues.md`, each in the file's established format (what the
C# does, why it is wrong, what corehydro does, and the fixture or ctest that pins it):
`OrderedPairedData.RemoveRange`'s two bugs; `SequentialSearchY`'s wrong search-start field; the
dead `XdeltaStart` / `YdeltaStart` that make `UseSmartSearch` never take the Hunt branch;
`Ordinate.operator==`'s NaN-compares-equal; the two incompatible `PerpendicularDistance`
implementations (one guarded, one not); `LangSimplify` returning an alias; the `Add` /
`OrdinateValid` asymmetry between the two paired-data classes; `UncertainOrdinate`'s
mean-vs-median probe disagreement; `RamerDouglasPeucker`'s output parameter behaving differently
in its two branches; `Statistics.RanksInPlace(out ties)` never recording the trailing run;
`DataFrame.LinearTrendTest`'s dead local and duplicated expression; the `Kurtosis + 3` asymmetry
between the two DataFrame summary methods; and `DataFrame.SummaryHypothesisTest`'s Mann-Whitney
argument pair, which passes `v1` TWICE when the two halves are equal in length (recorded now even
though the method itself is deferred to P5, because the finding is P4's).

- [ ] **Step 2: Reference docs and the index contract**

Every new or changed export gets title, description, every parameter, `\value`, and one runnable
example inside CRAN time limits. `hypothesis_test`'s help must state, per method, which arguments
it reads and which C# guard it inherits. `correlation`'s help loses its "matrix overloads are not
ported" sentence in BOTH languages and gains the matrix path and the Kendall exception. Verify:

```bash
Rscript -e 'pkgdown::check_pkgdown("corehydror")'
```
Expected: no missing-topic error. Confirm the quartodoc build lists every Python export.
Elements-of-style conventions throughout.

- [ ] **Step 3: Example 20 -- hypothesis tests and paired data**

Copy the structure of the `site/examples/19-global-optimizers` pair. Content: the Harricana River
annual peaks run through the independence, homogeneity and normality tests with a short reading of
what each one answers for a flood-frequency record; the same record inside `analysis_data()` so
`analysis_data_hypothesis_test()` and `analysis_data_statistics()` show the DataFrame path
(including the `use_log10` switch and the index split); a correlation MATRIX over three series;
and the paired-data half -- a reservoir stage-storage curve interpolated in linear and log space,
its trapezoidal area, the three simplification algorithms compared on the same curve with their
retained-point counts, and an uncertain curve sampled at its median and its mean.

Every number in the prose comes from the executed output, not from the plan. Each page ends with
the executable reproduction check (deterministic values pinned at 1e-15 relative tolerance). Python
notebook committed WITH outputs
(`jupyter nbconvert --to notebook --execute --inplace site/examples/20-hypothesis-tests-and-paired-data/python.ipynb`);
R rendered so `site/_freeze/` updates and is committed.

- [ ] **Step 4: Build and verify the site, commit**

```bash
pixi run docs
git add -A && git commit -m "docs: reference updates and worked example 20"
```
Expected: clean build; the new page renders in both languages.

---

### Task 12: Version 0.11.0, full verification, ship

**Files:**
- Modify: `corehydror/DESCRIPTION`, `corehydropy/pyproject.toml`, `pixi.toml` (all three to 0.11.0)
- Modify: `corehydror/NEWS.md`, `CHANGELOG.md`
- Modify: `.claude/CLAUDE.md` (Layout section: the new `numerics/data/paired_data/` directory,
  `hypothesis_tests.hpp`, `tabular_function.hpp`, the seventeen toolbox groups, and the
  `data_frame` fixture kind; Status: append the P4 paragraph with final numbers)

- [ ] **Step 1: Bump versions and write release notes**

NEWS/CHANGELOG: the twelve hypothesis tests through `hypothesis_test()`; the DataFrame facades
through `analysis_data_hypothesis_test()` and `analysis_data_statistics()`; the correlation matrix
path on `correlation()`; the paired-data curve surface (`curve_interpolate`, `curve_area`,
`curve_simplify`, `uncertain_curve_sample`, `tabular_function`); and, stated plainly rather than
buried, the two members deferred to P5 with the reason (`UnimodalityTest` and
`SummaryHypothesisTest` need the Machine Learning `GaussianMixtureModel`).

- [ ] **Step 2: Full verification (evidence before assertions)**

```bash
cmake --build core/build -j && ctest --test-dir core/build
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py
```
Expected: ctest green at baseline + 6 new suites (108/108 against the 102 baseline); testthat 0
failures above 6916; pytest 0 failures above 1711; gate `0 failed` with reproduced count above
5942 and skips still 11. Record the actual final numbers in NEWS/CHANGELOG and `.claude/CLAUDE.md`
-- measured, not predicted.

- [ ] **Step 3: R CMD check regression**

```bash
R CMD build corehydror && R CMD check --as-cran corehydror_0.11.0.tar.gz
```
Expected: the same three known NOTEs (CRAN-incoming non-FOSS license, long paths listing vendored
core headers, local HTML-tidy version), no new WARNING or NOTE.

- [ ] **Step 4: Commit, push, PR, CI**

```bash
git add -A && git commit -m "chore: release v0.11.0 (data and tests)"
git push -u origin port-data-and-tests
gh pr create --base port-math-extras \
  --title "P4: data and tests -- hypothesis tests, DataFrame facades, paired data (v0.11.0)" \
  --body "$(sed -n '/^## \[0.11.0\]/,/^## \[/p' CHANGELOG.md)"
gh run watch $(gh run list --branch port-data-and-tests -L1 --json databaseId -q '.[0].databaseId') --exit-status
```
Expected: CI green on the full matrix. The PR body must note that the P1-P3 stack has not landed
on `main` (PR #28 merged into `port-math-extras`, PR #27 into `port-distribution-gaps`), so this
PR targets `port-math-extras`; retarget to `main` if Cam lands the stack first. Do not merge
without Cam's go-ahead.

---

## Self-review notes

- **Spec coverage.** The spec's P4 paragraph names four things. "HypothesisTests" -> Tasks 1-3
  (Task 1 ports the four helpers it needs, which the spec did not know about because the
  `Statistics` severance note is in-header). "then the severed DataFrame hypothesis-test and
  summary-statistics facades it un-gates" -> Tasks 5-6. "the Paired Data layer" -> Tasks 7-10,
  plus `TabularFunction` and the six `Search.cs` overloads, both of which the spec's inventory
  listed elsewhere ("Small items") but which belong with their subject. "the Correlation matrix
  overloads" -> Task 4. "New toolbox groups `hypothesis` and `paired_data`" -> Tasks 3 and 10.
  "the DataFrame facades reach the existing model surface" -> Task 6 extends
  `data_frame_runner.hpp` and `analysis_data()`, which is that surface. Example 20 -> Task 11;
  the version bump -> Task 12.
- **One spec item is deliberately NOT fully delivered and it is called out, not buried:**
  `HypothesisTests.UnimodalityTest` and `DataFrame.SummaryHypothesisTest` need
  `GaussianMixtureModel`, which is P5's subject. Scope decision 1 gives the reasoning, both
  headers carry the severance note naming P5, and Task 12's release notes say so in prose. Twelve
  of thirteen statics and eleven of twelve facade members ship.
- **Type consistency.** The `hypothesis` group's method names (Task 3) are reused verbatim as the
  `data_frame` kind's hypothesis method names (Task 6), so one vocabulary covers both surfaces.
  `ToolboxResult` is introduced by the existing `toolbox/common.hpp` and is what Task 6's
  `run_data_frame` returns, which is what lets all three fixture runners reuse `toolbox_select`
  unchanged. `Ordinate` (Task 7) is consumed by `OrderedPairedData` (Task 8), which is what
  `UncertainOrderedPairedData::curve_sample` returns (Task 9), which is what `TabularFunction`
  interpolates over (Task 9) and what Task 10's `curve_result` serializer flattens. `SortOrder`
  and `Transform` are the already-ported enums throughout -- Task 7 states this once so no task
  defines a duplicate.
- **Task ordering.** Every port task precedes the surface task that dispatches it. Task 1 precedes
  Task 2 because `MannWhitney` and `MannKendall` cannot be written without the tie overload. Task
  4 precedes Task 5 only for the commit graph (the facades do not use the matrices); Task 2
  genuinely blocks Task 5. Task 7 precedes 8 precedes 9 precedes 10 by strict type dependency.
- **Deliberate non-placeholders.** Port tasks cite the exact C# file and line as the
  implementation reference (the established port convention) and enumerate by number every
  transcription detail measured to be oracle-visible or defect-preserving. Large input arrays (the
  78-point query sweep, the noise series, the 69-value Harricana record) are named by
  their C# test file and method with an instruction to transcribe verbatim rather than reproduced
  here, because a plan that retypes a 78-element array is a plan that introduces a typo -- the
  same convention P3's plan used for the Dijkstra graphs. Every EXPECTED value, which is what a
  reviewer must be able to check, is written out in full.
- **One decision is deliberately left to the implementer with both outcomes specified rather than
  guessed:** Task 8 note 5, `LangSimplify`'s alias return. A C++ value-returning signature has no
  aliasing analogue, so the task requires picking one behavior, writing down which, and making the
  ctest assert it -- rather than silently diverging.
