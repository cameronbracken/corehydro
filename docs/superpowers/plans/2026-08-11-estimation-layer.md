# Estimation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the four ported estimators (MaximumLikelihood, MaximumAPosteriori, BayesianAnalysis, GeneralizedMethodOfMoments) callable from R and Python over the phase-1 model specs, returning a fit object that composes with the base R generics and back into the model surface.

**Architecture:** One shared C++ runner, `core/include/corehydro/estimation/support/fit_runner.hpp`, owns every fit in the repo. Four callers drive it and none owns fit logic: the cpp11 glue, the pybind11 glue, the C++ fixture runner, and the dotnet oracle emitter. Each language gets ONE new glue entry point, `ch_fit_run_` / `fit_run`, dispatching on a target string; the four user verbs are pure R and pure Python that assemble a construct and shape a result. The existing fixture glue is rewritten to delegate, so the 4542 pinned oracle assertions keep passing against a single implementation.

**Tech Stack:** C++17 (no external dependencies), cpp11 for R, pybind11 for Python, testthat, pytest, ctest, dotnet 10 for the oracle gate, Quarto and pkgdown for docs.

**Spec:** `docs/superpowers/specs/2026-08-11-estimation-layer-design.md`. Read it before starting.

## Global Constraints

- Branch: `surface-estimation-layer`, already cut from `main` with the spec committed. Do not push and do not open a PR without being asked.
- Commits are GPG-signed as `Cam Bracken <cameron.bracken@pm.me>`. **No `Co-Authored-By` trailer on any commit.**
- No new external C++ dependency in `core/`. No new R dependency (`corehydror` has zero runtime dependencies; `jsonlite` is `Suggests` and test-only). No new Python dependency (numpy only; never pandas).
- Never use `M_PI`; use `corehydro::numerics::kPi`. Never name a namespace alias `gamma` or `stat`.
- Oracle values live ONLY in `fixtures/*.json`. Never hardcode an expected number in a test file.
- Never loosen a fixture tolerance and never add an `oracle_skip` to make a new case pass. If a value will not reproduce, record why in `docs/upstream-csharp-issues.md` and assert the structural invariant instead.
- `core/` and `fixtures/` are subtree symlinks inside both packages. Edit through `core/`; nothing needs re-syncing.
- After editing any `corehydror/src/*.cpp`, run `Rscript -e 'cpp11::cpp_register("corehydror")'` before installing.
- Every new R export must appear in `corehydror/_pkgdown.yml`. Every new Python export must appear in the `quartodoc.sections` of `site/_quarto.yml`. pkgdown errors on a missing entry.
- Baselines that must not regress: ctest 79/79, oracle gate 4542 reproduced / 0 failed / 11 skipped, testthat 4446 / 0, pytest 884.
- Every model spec in a construct needs a `"dataset"` key naming a datasets entry, even when the harness passes the observations as a flat vector alongside it; the spec builder requires the marker. Tasks 1 and 2 both hit this.
- Target version at the end of the phase: **0.4.0** in `corehydror/DESCRIPTION`, `corehydropy/pyproject.toml`, and the core version stamp.

---

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `core/include/corehydro/estimation/support/fit_runner.hpp` | `FitResult` struct and `run_fit()`. The only place a fit happens. |
| `core/tests/test_fit_runner.cpp` | ctest coverage of the runner: shape, guards, error paths. |
| `corehydror/R/fit.R` | The four verbs, the `corehydro_fit` constructor, the S3 methods, `fit_diagnostics`, `quantile_variance`. |
| `corehydror/tests/testthat/test-fit.R` | Behavioural tests for the R surface. |
| `corehydropy/src/corehydropy/fit.py` | The Python twin: four verbs plus the `Fit` class. |
| `corehydropy/tests/test_fit.py` | Behavioural tests for the Python surface. |
| `fixtures/estimation/fit_profile.json` | Profile likelihood and profile CI oracles. |
| `fixtures/estimation/fit_optimizers.json` | One fit under each of the six optimizers. |
| `fixtures/estimation/fit_bayes_diagnostics.json` | R-hat, ESS, acceptance rates, posterior summary, Pareto-k. |
| `fixtures/estimation/fit_cross_language.json` | Seeded digest proving R and Python agree. |
| `site/examples/25-estimation-methods/r.qmd` | R half of the worked example. |
| `site/examples/25-estimation-methods/python.ipynb` | Python half, committed with outputs. |

**Modified:**

| Path | Change |
|---|---|
| `core/CMakeLists.txt` | Register `test_fit_runner` in `BF_TESTS`. |
| `corehydror/src/estimation.cpp` | Add `ch_fit_run_`, `ch_fit_diagnostics_`, `ch_fit_quantile_variance_`; rewrite the four fixture functions to delegate. |
| `corehydropy/src/bindings/estimation.cpp` | The same three additions and the same delegation. |
| `core/tests/test_fixtures.cpp` | Rewrite the `model_estimation` build path to delegate to the runner. |
| `corehydror/NAMESPACE`, `corehydror/R/cpp11.R` | Regenerated by roxygen and `cpp11::cpp_register`. |
| `corehydropy/src/corehydropy/__init__.py` | Export the four verbs and `Fit`. |
| `tools/oracle_emitter/Program.cs` | New assertion methods in `DispatchEstimation` and their build-time wiring. |
| `fixtures/README.md` | Document the new `model_estimation` assertion methods. |
| `corehydror/_pkgdown.yml`, `site/_quarto.yml` | New "Estimation" reference section in each. |
| `site/examples/index.qmd`, the coverage page | List the new example pair. |
| `CHANGELOG.md`, `DESCRIPTION`, `pyproject.toml`, the core version stamp | 0.4.0. |

---

## Task 1: The shared runner, MLE and MAP

**Files:**
- Create: `core/include/corehydro/estimation/support/fit_runner.hpp`
- Create: `core/tests/test_fit_runner.cpp`
- Modify: `core/CMakeLists.txt` (add `test_fit_runner` to the `BF_TESTS` list, alphabetically near the other estimation tests)

**Interfaces:**
- Consumes: `models::spec::build_model_from_json(model_json, data)` from `core/include/corehydro/models/model_spec.hpp`; `models::spec::parse_json` and `JsonValue` from `core/include/corehydro/models/json_lite.hpp`; `est::MaximumLikelihood` and `est::MaximumAPosteriori`.
- Produces: `corehydro::estimation::support::FitResult` and
  `FitResult run_fit(const std::string& target, const std::string& construct_json, const std::vector<double>& dataset)`.
  Every later task depends on these exact names. `FitResult` field names are the contract the R and Python glue read.

- [ ] **Step 1: Write the failing test**

Create `core/tests/test_fit_runner.cpp`:

```cpp
// ctest coverage of the shared fit runner. Oracle VALUES live in fixtures/*.json and are
// asserted by test_fixtures.cpp; this file asserts SHAPE, GUARDS and ERROR PATHS, which are
// corehydro additions with no C# counterpart and therefore have no oracle.
#include <cmath>
#include <string>
#include <vector>

#include "corehydro/estimation/support/fit_runner.hpp"
#include "check.hpp"

namespace support = corehydro::estimation::support;

static const std::vector<double> kPeaks = {12500, 15300, 9870, 21000, 18400,
                                           11200, 26800, 14100, 19500, 11600};

static void test_mle_shape() {
    std::string construct = R"({"model":{"family":"Normal"},"optimizer":"NelderMead"})";
    support::FitResult r = support::run_fit("MaximumLikelihood", construct, kPeaks);

    CHECK_TRUE(r.method == "MaximumLikelihood");
    CHECK_TRUE(r.parameters.size() == 2);
    CHECK_TRUE(r.parameter_names.size() == 2);
    CHECK_TRUE(!r.parameter_names[0].empty());
    CHECK_TRUE(r.converged);
    CHECK_TRUE(r.status == "Success");
    CHECK_TRUE(r.function_evaluations > 0);
    CHECK_TRUE(r.nobs == 10);
    CHECK_TRUE(std::isfinite(r.log_likelihood));
    CHECK_TRUE(std::isfinite(r.aic));
    CHECK_TRUE(std::isfinite(r.bic));
    // Hessian stack present and square for a 2-parameter model.
    CHECK_TRUE(r.covariance.size() == 4);
    CHECK_TRUE(r.standard_errors.size() == 2);
    CHECK_TRUE(r.correlation.size() == 4);
    // The fitted spec round-trips: it carries the fitted values.
    CHECK_TRUE(r.model_spec.find("parameter_values") != std::string::npos);
}

static void test_hessian_can_be_disabled() {
    std::string construct =
        R"({"model":{"family":"Normal"},"optimizer":"NelderMead","hessian":false})";
    support::FitResult r = support::run_fit("MaximumLikelihood", construct, kPeaks);
    CHECK_TRUE(r.covariance.empty());
    CHECK_TRUE(r.standard_errors.empty());
    CHECK_TRUE(r.correlation.empty());
    CHECK_TRUE(std::isfinite(r.log_likelihood));  // the fit itself still happened
}

static void test_single_parameter_covariance_is_nan_not_zero() {
    // Exponential has one parameter. C# GetCovarianceMatrix throws below two, so the honest
    // report is NaN, NOT the silent zeros the fixture glue used to return.
    std::string construct = R"({"model":{"family":"Exponential"},"optimizer":"Brent"})";
    support::FitResult r = support::run_fit("MaximumLikelihood", construct, kPeaks);
    CHECK_TRUE(r.parameters.size() == 1);
    CHECK_TRUE(r.covariance.size() == 1);
    CHECK_TRUE(std::isnan(r.covariance[0]));
    CHECK_TRUE(std::isnan(r.standard_errors[0]));
    CHECK_TRUE(std::isnan(r.correlation[0]));
}

static void test_map_reports_status() {
    std::string construct = R"({"model":{"family":"Normal"},"optimizer":"NelderMead"})";
    support::FitResult r = support::run_fit("MaximumAPosteriori", construct, kPeaks);
    CHECK_TRUE(r.method == "MaximumAPosteriori");
    CHECK_TRUE(r.status == "Success");
    CHECK_TRUE(std::isfinite(r.prior_log_likelihood));
}

static void test_unknown_target_throws() {
    bool threw = false;
    try {
        support::run_fit("NotAnEstimator", R"({"model":{"family":"Normal","dataset":"peaks"}})", kPeaks);
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("NotAnEstimator") != std::string::npos;
    }
    CHECK_TRUE(threw);
}

static void test_unknown_optimizer_throws_naming_it() {
    bool threw = false;
    try {
        support::run_fit("MaximumLikelihood",
                         R"({"model":{"family":"Normal"},"optimizer":"Simplexx"})", kPeaks);
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("Simplexx") != std::string::npos;
    }
    CHECK_TRUE(threw);
}

int main() {
    test_mle_shape();
    test_hessian_can_be_disabled();
    test_single_parameter_covariance_is_nan_not_zero();
    test_map_reports_status();
    test_unknown_target_throws();
    test_unknown_optimizer_throws_naming_it();
    return chtest::summary("fit_runner");
}
```

Check `core/tests/test_util.hpp` first and match whatever assertion macro and summary function the neighbouring tests use (for example `core/tests/test_model_base.cpp`). If the macro is named differently, use the existing name rather than inventing `CH_CHECK`.

- [ ] **Step 2: Register the test and run it to verify it fails**

Add `test_fit_runner` to the `BF_TESTS` list in `core/CMakeLists.txt`, then:

```bash
cmake -S core -B core/build && cmake --build core/build 2>&1 | tail -20
```

Expected: FAIL at compile with `fatal error: 'corehydro/estimation/support/fit_runner.hpp' file not found`.

- [ ] **Step 3: Write the runner**

Create `core/include/corehydro/estimation/support/fit_runner.hpp`. Header comment first, in the house style:

```cpp
// corehydro ADDITION -- no upstream C# counterpart (sibling of models/model_spec.hpp and
// analyses/support/analysis_runner.hpp).
//
// The single place a fit happens in this repo. Four callers drive it and none owns any fit
// logic: the cpp11 glue (corehydror/src/estimation.cpp), the pybind11 glue
// (corehydropy/src/bindings/estimation.cpp), the C++ fixture runner (core/tests/
// test_fixtures.cpp) and the dotnet oracle emitter. Each serializes its native construct to
// JSON and calls run_fit(target, construct_json, dataset), which parses with json_lite.hpp,
// builds the model with models::spec::build_model_from_json, constructs the named estimator,
// runs estimate() once, and packs the full surface into a flat FitResult. Because all four
// run the identical compiled core with a bit-exact Mersenne Twister, a seeded call returns
// identical numbers everywhere.
//
// Three deliberate differences from the pre-phase-2 fixture glue this replaced, all because
// this surface now faces users rather than only fixtures:
//   1. A model with fewer than two parameters gets a NaN covariance/SE/correlation, not the
//      silent zeros the old glue returned. C# GetCovarianceMatrix throws below two parameters.
//   2. A failed estimate() throws naming the estimator, the optimizer and the model family,
//      instead of "failed for a fixture case".
//   3. Every result carries `converged` and `status` so a caller can check rather than infer.
#pragma once
```

Then the includes (`<cmath>`, `<limits>`, `<memory>`, `<stdexcept>`, `<string>`, `<vector>`, the four estimation headers, `models/json_lite.hpp`, `models/model_spec.hpp`, `models/support/model_base.hpp`, `numerics/math/optimization/support/optimization_status.hpp`), then:

```cpp
namespace corehydro::estimation::support {

// Flat result surface every fit assertion and binding reads. Only the fields the requested
// target populates are filled; the rest keep their defaults (empty vector / NaN).
struct FitResult {
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    // --- common block, populated by every target ---------------------------------------
    std::string method;
    std::vector<std::string> parameter_names;
    std::vector<double> parameters;
    double log_likelihood = kNaN;
    double prior_log_likelihood = kNaN;
    double aic = kNaN;
    double bic = kNaN;
    int nobs = 0;
    // Row-major n x n; empty when hessian was not requested, all-NaN when n < 2.
    std::vector<double> covariance;
    std::vector<double> standard_errors;
    std::vector<double> correlation;
    bool converged = false;
    std::string status = "None";
    int function_evaluations = 0;
    // The model spec with the fitted values applied, so the caller can rebuild from the fit.
    std::string model_spec;
};

// Maps an OptimizationStatus to the string the bindings surface.
inline std::string status_name(numerics::math::optimization::OptimizationStatus s) {
    using S = numerics::math::optimization::OptimizationStatus;
    switch (s) {
        case S::None: return "None";
        case S::Success: return "Success";
        case S::MaximumIterationsReached: return "MaximumIterationsReached";
        case S::MaximumFunctionEvaluationsReached: return "MaximumFunctionEvaluationsReached";
        default: return "Failure";
    }
}

// Optimizer name -> OptimizationMethod. Accepts the "MLSL" alias, matching the pre-phase-2
// glue. Throws naming the value it could not parse.
inline OptimizationMethod parse_optimizer(const std::string& s) {
    if (s == "Brent") return OptimizationMethod::Brent;
    if (s == "BFGS") return OptimizationMethod::BFGS;
    if (s == "NelderMead") return OptimizationMethod::NelderMead;
    if (s == "Powell") return OptimizationMethod::Powell;
    if (s == "DifferentialEvolution") return OptimizationMethod::DifferentialEvolution;
    if (s == "MultilevelSingleLinkage" || s == "MLSL")
        return OptimizationMethod::MultilevelSingleLinkage;
    throw std::runtime_error("unknown optimizer '" + s + "'");
}
```

Copy the exact alias set and the exact spelling from the existing `parse_optimization_method` in `corehydror/src/estimation.cpp` (around line 80) so no fixture changes meaning. Then the shared filler and the dispatch:

```cpp
// Parameter display names, falling back to p1..pn when the model leaves them empty (some
// ported models do -- see the PriorInfluenceDiagnostics note in docs/upstream-csharp-issues.md).
inline std::vector<std::string> parameter_names_of(const corehydro::models::ModelBase& model) {
    std::vector<std::string> names;
    for (int i = 0; i < model.number_of_parameters(); ++i) {
        const std::string& n = model.parameters()[static_cast<std::size_t>(i)].display_name();
        names.push_back(n.empty() ? ("p" + std::to_string(i + 1)) : n);
    }
    return names;
}

// Fills the common block from any estimator exposing the ML/MAP accessor set.
template <typename TEstimator>
void fill_common(FitResult& r, TEstimator& e, corehydro::models::ModelBase& model,
                 bool want_hessian) {
    int n = model.number_of_parameters();
    const std::vector<double>& best = e.best_parameter_set().values;
    r.parameters.assign(best.begin(), best.end());
    r.parameter_names = parameter_names_of(model);
    r.log_likelihood = e.maximum_log_likelihood();
    std::vector<double> values = best;
    r.prior_log_likelihood = model.prior_log_likelihood(values);
    r.nobs = static_cast<int>(model.pointwise_data_log_likelihood(best).size());
    r.aic = e.get_aic();
    r.bic = e.get_bic(r.nobs);
    r.converged = e.status() == numerics::math::optimization::OptimizationStatus::Success;
    r.status = status_name(e.status());
    r.function_evaluations = e.total_function_evaluations();

    if (!want_hessian) return;
    if (n < 2) {
        // C# GetCovarianceMatrix throws below two parameters; report NaN, never zeros.
        r.covariance.assign(static_cast<std::size_t>(n * n), FitResult::kNaN);
        r.standard_errors.assign(static_cast<std::size_t>(n), FitResult::kNaN);
        r.correlation.assign(static_cast<std::size_t>(n * n), FitResult::kNaN);
        return;
    }
    auto cov = e.get_covariance_matrix();
    auto corr = e.get_correlation_matrix();
    r.covariance.resize(static_cast<std::size_t>(n * n));
    r.correlation.resize(static_cast<std::size_t>(n * n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            r.covariance[static_cast<std::size_t>(i * n + j)] = cov(i, j);
            r.correlation[static_cast<std::size_t>(i * n + j)] = corr(i, j);
        }
    std::vector<double> se = e.get_standard_errors();
    r.standard_errors.assign(se.begin(), se.end());
}

// Re-serializes the model spec with the fitted values appended, so a caller can rebuild the
// fitted model. Uses the parsed construct's model object and adds `parameter_values`.
inline std::string fitted_spec(const corehydro::models::spec::JsonValue& model_spec,
                               const std::vector<double>& values);

inline FitResult run_fit(const std::string& target, const std::string& construct_json,
                         const std::vector<double>& dataset) {
    corehydro::models::spec::JsonValue construct =
        corehydro::models::spec::parse_json(construct_json);
    const auto& model_json = construct.at("model");
    std::string model_text = corehydro::models::spec::to_json_string(model_json);
    bool want_hessian = construct.value_or("hessian", true);
    std::string optimizer = construct.value_or("optimizer", "DifferentialEvolution");

    if (target == "MaximumLikelihood" || target == "MaximumAPosteriori") {
        std::unique_ptr<corehydro::models::ModelBase> model =
            corehydro::models::spec::build_model_from_json(model_text, dataset);
        OptimizationMethod method = parse_optimizer(optimizer);
        FitResult r;
        r.method = target;
        if (target == "MaximumLikelihood") {
            MaximumLikelihood e(*model, method);
            e.set_compute_hessian(want_hessian);
            if (!e.estimate())
                throw std::runtime_error("MaximumLikelihood::estimate() failed with optimizer " +
                                         optimizer);
            fill_common(r, e, *model, want_hessian);
        } else {
            MaximumAPosteriori e(*model, method);
            e.set_compute_hessian(want_hessian);
            if (!e.estimate())
                throw std::runtime_error("MaximumAPosteriori::estimate() failed with optimizer " +
                                         optimizer);
            fill_common(r, e, *model, want_hessian);
        }
        r.model_spec = fitted_spec(model_json, r.parameters);
        return r;
    }
    throw std::runtime_error("unknown fit target: " + target);
}

}  // namespace corehydro::estimation::support
```

`json_lite.hpp` may not have a serializer. Check for a `to_json_string` or equivalent; if it does not exist, add a small `inline std::string to_json_string(const JsonValue&)` to `json_lite.hpp` that re-emits the parsed value with `%.17g` for numbers, matching the `spec_number` contract in `corehydror/R/spec.R`. Implement `fitted_spec` on top of it by emitting the model object with a `"parameter_values"` array appended. Add a ctest assertion in `test_fit_runner.cpp` that `parse_json(to_json_string(v))` round-trips a double bit-exactly.

Prefer `set_compute_hessian` before `estimate()`; read `maximum_likelihood.hpp:144` to confirm the setter's semantics and whether it must precede `estimate()`.

- [ ] **Step 4: Run the test to verify it passes**

```bash
cmake --build core/build && ctest --test-dir core/build -R test_fit_runner --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 5: Run the full core suite to prove no regression**

```bash
ctest --test-dir core/build
```

Expected: `80/80` (79 existing plus the new one).

- [ ] **Step 6: Commit**

```bash
git add core/include/corehydro/estimation/support/fit_runner.hpp core/tests/test_fit_runner.cpp core/CMakeLists.txt core/include/corehydro/models/json_lite.hpp
git commit -m "feat: shared fit runner for the MLE and MAP estimators

One place a fit happens, driven identically by the two bindings, the C++
fixture runner and the oracle emitter. Reports NaN rather than zeros for a
sub-two-parameter covariance, throws naming the estimator and optimizer on a
failed estimate, and carries converged/status so a caller can check."
```

---

## Task 2: Profile likelihood and profile confidence intervals

**Files:**
- Modify: `core/include/corehydro/estimation/support/fit_runner.hpp`
- Modify: `core/tests/test_fit_runner.cpp`

**Interfaces:**
- Consumes: `FitResult`, `run_fit` from Task 1; `e.profile_likelihood(bins)` returning `std::vector<Matrix>` (`maximum_likelihood.hpp:223`) and `e.parameter_confidence_intervals(alpha)` returning a `Matrix` of `n x 2` (`maximum_likelihood.hpp:253`).
- Produces: `FitResult::profile_lower`, `profile_upper`, `profile_grid`, `profile_bins` fields, and the construct keys `"profile"` (bool, default false), `"profile_bins"` (int, default 100), `"alpha"` (double, default 0.1).

- [ ] **Step 1: Write the failing test**

Append to `core/tests/test_fit_runner.cpp` and call from `main()`:

```cpp
static void test_profile_off_by_default() {
    std::string construct = R"({"model":{"family":"Normal"},"optimizer":"NelderMead"})";
    support::FitResult r = support::run_fit("MaximumLikelihood", construct, kPeaks);
    CHECK_TRUE(r.profile_grid.empty());
    CHECK_TRUE(r.profile_lower.empty());
    CHECK_TRUE(r.profile_bins == 0);
}

static void test_profile_on_request() {
    std::string construct =
        R"({"model":{"family":"Normal"},"optimizer":"NelderMead","profile":true,)"
        R"("profile_bins":20,"alpha":0.1})";
    support::FitResult r = support::run_fit("MaximumLikelihood", construct, kPeaks);
    CHECK_TRUE(r.profile_bins == 20);
    // n_params x bins x 2 (parameter value, profile log-likelihood), row-major.
    CHECK_TRUE(r.profile_grid.size() == 2u * 20u * 2u);
    CHECK_TRUE(r.profile_lower.size() == 2);
    CHECK_TRUE(r.profile_upper.size() == 2);
    // The interval brackets the point estimate.
    for (std::size_t i = 0; i < r.parameters.size(); ++i) {
        CHECK_TRUE(r.profile_lower[i] <= r.parameters[i]);
        CHECK_TRUE(r.profile_upper[i] >= r.parameters[i]);
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cmake --build core/build 2>&1 | tail -5
```

Expected: FAIL at compile, `no member named 'profile_grid' in 'corehydro::estimation::support::FitResult'`.

- [ ] **Step 3: Implement**

Add to `FitResult`:

```cpp
    // --- profile block, populated only when construct["profile"] is true ----------------
    // n_params * bins * 2, row-major: [parameter][bin][value, profile log-likelihood].
    std::vector<double> profile_grid;
    std::vector<double> profile_lower, profile_upper;  // n_params, profile-likelihood CIs
    int profile_bins = 0;
```

Add to `fill_common`, or to a separate `fill_profile` called from both estimator arms (a separate function is cleaner, since only ML and MAP have these methods):

```cpp
// Profile likelihood grid + profile-likelihood confidence intervals. Costs bins * n_params
// likelihood evaluations, so it is off unless the caller asks.
template <typename TEstimator>
void fill_profile(FitResult& r, const TEstimator& e, int bins, double alpha) {
    // `Matrix` is unqualified here only for brevity: use whichever alias
    // maximum_likelihood.hpp itself uses for the return of profile_likelihood() and
    // parameter_confidence_intervals(), and add the matching using-declaration or the fully
    // qualified name at the top of this header.
    int n = static_cast<int>(r.parameters.size());
    r.profile_bins = bins;
    std::vector<Matrix> profiles = e.profile_likelihood(bins);
    r.profile_grid.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(bins) * 2u);
    for (int p = 0; p < n; ++p)
        for (int b = 0; b < bins; ++b) {
            std::size_t base = (static_cast<std::size_t>(p) * static_cast<std::size_t>(bins) +
                                static_cast<std::size_t>(b)) * 2u;
            r.profile_grid[base] = profiles[static_cast<std::size_t>(p)](b, 0);
            r.profile_grid[base + 1] = profiles[static_cast<std::size_t>(p)](b, 1);
        }
    Matrix cis = e.parameter_confidence_intervals(alpha);
    r.profile_lower.resize(static_cast<std::size_t>(n));
    r.profile_upper.resize(static_cast<std::size_t>(n));
    for (int p = 0; p < n; ++p) {
        r.profile_lower[static_cast<std::size_t>(p)] = cis(p, 0);
        r.profile_upper[static_cast<std::size_t>(p)] = cis(p, 1);
    }
}
```

Call it in both the ML and MAP arms, guarded by `construct.value_or("profile", false)`, reading `construct.value_or("profile_bins", 100)` and `construct.value_or("alpha", 0.1)`. Confirm the column order of `profile_likelihood`'s returned `Matrix` by reading `maximum_likelihood.hpp:223-252` and match it; if the columns are (log-likelihood, value) rather than (value, log-likelihood), fix the test's comment and the R and Python docs to match the source rather than reordering the C++.

- [ ] **Step 4: Run the test to verify it passes**

```bash
cmake --build core/build && ctest --test-dir core/build -R test_fit_runner --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/include/corehydro/estimation/support/fit_runner.hpp core/tests/test_fit_runner.cpp
git commit -m "feat: profile likelihood and profile confidence intervals in the fit runner

Off by default: each costs bins * n_params likelihood evaluations."
```

---

## Task 3: The Bayesian block

**Files:**
- Modify: `core/include/corehydro/estimation/support/fit_runner.hpp`
- Modify: `core/tests/test_fit_runner.cpp`

**Interfaces:**
- Consumes: `est::BayesianAnalysis` (`core/include/corehydro/estimation/bayesian_analysis.hpp`), whose setters are listed at lines 149 to 211 and whose `results()` returns `std::optional<MCMCResults>`; `MCMCResults` fields `markov_chains`, `output`, `mean_log_likelihood`, `acceptance_rates`, `parameter_results`, `map`, `posterior_mean`; `ParameterStatistics` fields `rhat`, `ess`, `n`, `mean`, `median`, `standard_deviation`, `lower_ci`, `upper_ci`.
- Produces: the `FitResult` Bayesian fields listed below, plus the target string `"BayesianAnalysis"`.

- [ ] **Step 1: Write the failing test**

Append to `core/tests/test_fit_runner.cpp` and call from `main()`:

```cpp
static void test_bayesian_block() {
    std::string construct =
        R"({"model":{"family":"Normal","dataset":"peaks"},"sampler":"DEMCz","seed":12345,"iterations":200,)"
        R"("number_of_chains":4,"thinning_interval":1,"output_length":500,)"
        R"("credible_interval_width":0.9})";
    support::FitResult r = support::run_fit("BayesianAnalysis", construct, kPeaks);

    CHECK_TRUE(r.method == "BayesianAnalysis");
    CHECK_TRUE(r.chain_dims.size() == 3);
    CHECK_TRUE(r.chain_dims[0] == 4);            // chains
    CHECK_TRUE(r.chain_dims[2] == 2);            // parameters
    CHECK_TRUE(r.draws.size() == static_cast<std::size_t>(r.chain_dims[0]) *
                                   static_cast<std::size_t>(r.chain_dims[1]) *
                                   static_cast<std::size_t>(r.chain_dims[2]));
    CHECK_TRUE(r.posterior.size() == r.posterior_rows * 2u);
    CHECK_TRUE(r.acceptance_rates.size() == 4);
    CHECK_TRUE(r.posterior_mean.size() == 2);
    CHECK_TRUE(r.map.size() == 2);
    CHECK_TRUE(r.rhat.size() == 2);
    CHECK_TRUE(r.ess.size() == 2);
    CHECK_TRUE(r.summary_median.size() == 2);
    CHECK_TRUE(std::isfinite(r.dic));
    CHECK_TRUE(std::isfinite(r.waic));
    CHECK_TRUE(std::isfinite(r.looic));
    CHECK_TRUE(!r.pareto_k.empty());
    CHECK_TRUE(r.converged);   // a completed chain reports converged
}

static void test_bayesian_rejects_a_sampler_it_cannot_construct() {
    bool threw = false;
    try {
        support::run_fit("BayesianAnalysis",
                         R"({"model":{"family":"Normal","dataset":"peaks"},"sampler":"HMC"})", kPeaks);
    } catch (const std::exception& e) {
        std::string m = e.what();
        threw = m.find("HMC") != std::string::npos && m.find("mcmc_sample") != std::string::npos;
    }
    CHECK_TRUE(threw);
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cmake --build core/build 2>&1 | tail -5
```

Expected: FAIL at compile, `no member named 'chain_dims'`.

- [ ] **Step 3: Implement**

Add to `FitResult`:

```cpp
    // --- Bayesian block ----------------------------------------------------------------
    // Raw chains, flattened CHAIN-major: index = (chain * n_iterations + iter) * n_params + p.
    // This is the exact order ch_estimation_bayes_run_ has always returned, so the fixture
    // `chain_value [chain, iter, param]` arm is unchanged. The bindings permute it to the
    // user-facing [iteration, chain, parameter] array.
    std::vector<double> draws;
    std::vector<int> chain_dims;              // {n_chains, n_iterations, n_params}
    // The thinned posterior the analyses consume, flattened row-major, posterior_rows x n_params.
    std::vector<double> posterior;
    std::size_t posterior_rows = 0;
    std::vector<double> acceptance_rates;     // one per chain
    std::vector<double> mean_log_likelihood;  // one per iteration
    std::vector<double> map, posterior_mean;  // n_params
    std::vector<double> rhat, ess;            // n_params
    std::vector<double> summary_mean, summary_median, summary_sd,
                        summary_lower, summary_upper;  // n_params
    double dic = kNaN, waic = kNaN, waic_pd = kNaN, looic = kNaN, loo_pd = kNaN,
           looic_se = kNaN;
    std::vector<double> pareto_k;
```

Add the sampler parser and the arm. Copy the four-sampler set verbatim from `parse_sampler_type` in `corehydror/src/estimation.cpp:122`, and make the rejection message name the alternative:

```cpp
inline SamplerType parse_sampler(const std::string& s) {
    if (s == "DEMCz") return SamplerType::DEMCz;
    if (s == "DEMCzs") return SamplerType::DEMCzs;
    if (s == "ARWMH") return SamplerType::ARWMH;
    if (s == "NUTS") return SamplerType::NUTS;
    throw std::runtime_error(
        "BayesianAnalysis cannot construct sampler '" + s +
        "'; it supports DEMCz, DEMCzs, ARWMH and NUTS. For RWMH, HMC or SNIS use mcmc_sample().");
}
```

The arm mirrors `ch_estimation_bayes_run_` (`corehydror/src/estimation.cpp:264-312`) exactly for the settings cascade, including `set_use_simulation_defaults(false)` and `set_use_advanced_simulation_defaults(false)` and the `> 0` guards, then additionally reads the sampler knobs and fills the new fields:

```cpp
    if (target == "BayesianAnalysis") {
        std::unique_ptr<corehydro::models::ModelBase> model =
            corehydro::models::spec::build_model_from_json(model_text, dataset);
        BayesianAnalysis ba(*model, parse_sampler(construct.value_or("sampler", "DEMCzs")));
        ba.set_use_simulation_defaults(false);
        ba.set_use_advanced_simulation_defaults(false);
        int seed = construct.value_or("seed", -1);
        if (seed >= 0) ba.set_prng_seed(seed);
        if (construct.value_or("iterations", 0) > 0)
            ba.set_iterations(construct.value_or("iterations", 0));
        // ... warmup_iterations, number_of_chains, thinning_interval, initial_iterations,
        // output_length, all with the same `> 0` guard as the pre-phase-2 glue ...
        // Sampler knobs (doubles, applied only when present so the sampler default stands):
        if (construct.contains("jump")) ba.set_jump(construct.at("jump").as_double());
        // ... jump_threshold, snooker_threshold, noise, scale, beta, max_tree_depth ...
        if (construct.contains("credible_interval_width"))
            ba.set_credible_interval_width(construct.at("credible_interval_width").as_double());
        if (construct.contains("point_estimator"))
            ba.set_point_estimator(parse_point_estimator(
                construct.at("point_estimator").as_string()));

        if (!ba.estimate())
            throw std::runtime_error("BayesianAnalysis::estimate() failed for sampler " +
                                     construct.value_or("sampler", "DEMCzs"));
        FitResult r;
        r.method = "BayesianAnalysis";
        // fill parameter_names, nobs, the point estimate (ba.point_estimate().values) into
        // r.parameters, log_likelihood, prior_log_likelihood, aic/bic from the point estimate,
        // then the chains, the thinned output, acceptance rates, mean log-likelihood, the
        // per-parameter statistics off results()->parameter_results[i].summary_statistics,
        // and dic/waic/waic_pd/looic/loo_pd/looic_se/pareto_k off the analysis.
        r.converged = true;
        r.status = "Success";
        r.model_spec = fitted_spec(model_json, r.parameters);
        return r;
    }
```

Write `parse_point_estimator` against the actual `PointEstimateType` enum: read it (grep for `enum class PointEstimateType`) and accept every member by its C# name. `aic` and `bic` for the Bayesian arm come from the same `-2 logL + 2k` and `-2 logL + k log n` definitions the estimators use; if `BayesianAnalysis` exposes its own accessors, use those instead of recomputing, and say which in a comment.

- [ ] **Step 4: Run the test to verify it passes**

```bash
cmake --build core/build && ctest --test-dir core/build -R test_fit_runner --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/include/corehydro/estimation/support/fit_runner.hpp core/tests/test_fit_runner.cpp
git commit -m "feat: Bayesian block in the fit runner

Raw chains, the thinned posterior, acceptance rates, mean log-likelihood,
per-parameter R-hat/ESS/median/credible bounds, and the information criteria.
Rejects RWMH/HMC/SNIS by name: BayesianAnalysis cannot construct them."
```

---

## Task 4: The GMM block, quantile variance, and diagnostics

**Files:**
- Modify: `core/include/corehydro/estimation/support/fit_runner.hpp`
- Modify: `core/tests/test_fit_runner.cpp`

**Interfaces:**
- Consumes: `models::spec::build_bulletin17c_from_json(model_json, data)` (`model_spec.hpp:681`); `est::GeneralizedMethodOfMoments` with `set_estimation_strategy`, `set_max_gmm_iterations`, `estimate()`, `post_process(use_sandwich, compute_jstat)`, `best_parameter_set()`, `get_standard_errors()`, `get_covariance_matrix()`, `get_correlation_matrix()`, `jstat()`, `jstat_pval()`, `gmm_iterations()`, `converged_within_tolerance()`, `optimizer_fallback_count()`, and the diagnostics quartet; `model->quantile_variance(1 - aep, values, cov.to_array())`. The construction cascade is `build_and_fit_gmm` at `corehydror/src/estimation.cpp:107` -- copy it verbatim, including `post_process(true, true)`.
- Produces: `FitResult` GMM fields; `double run_fit_quantile_variance(const std::string& construct_json, const std::vector<double>& dataset, double aep)`; `FitDiagnostics run_fit_diagnostics(const std::string& target, const std::string& construct_json, const std::vector<double>& dataset)`.

- [ ] **Step 1: Write the failing test**

Append and call from `main()`:

```cpp
static void test_gmm_block() {
    std::string construct =
        R"({"model":{"type":"bulletin17c","family":"LogPearsonTypeIII","dataset":"peaks"},)"
        R"("strategy":"Iterative","optimizer":"BFGS","max_gmm_iterations":50})";
    support::FitResult r = support::run_fit("GMM", construct, kPeaks);
    CHECK_TRUE(r.method == "GMM");
    CHECK_TRUE(r.parameters.size() == 3);
    CHECK_TRUE(r.standard_errors.size() == 3);
    CHECK_TRUE(r.gmm_iterations > 0);
    // B17C GMM is always just-identified, so the J-statistic p-value is structurally NaN.
    // See docs/upstream-csharp-issues.md.
    CHECK_TRUE(std::isnan(r.j_stat_pval));
}

static void test_gmm_rejects_a_non_b17c_model() {
    bool threw = false;
    try {
        support::run_fit("GMM", R"({"model":{"family":"Normal","dataset":"peaks"}})", kPeaks);
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("bulletin17c") != std::string::npos;
    }
    CHECK_TRUE(threw);
}

static void test_quantile_variance_is_finite_and_positive() {
    std::string construct =
        R"({"model":{"type":"bulletin17c","family":"LogPearsonTypeIII","dataset":"peaks"},)"
        R"("strategy":"Iterative","optimizer":"BFGS"})";
    double v = support::run_fit_quantile_variance(construct, kPeaks, 0.01);
    CHECK_TRUE(std::isfinite(v) && v > 0.0);
}

static void test_diagnostics_shape() {
    std::string construct = R"({"model":{"family":"Normal"},"optimizer":"NelderMead"})";
    support::FitDiagnostics d =
        support::run_fit_diagnostics("MaximumAPosteriori", construct, kPeaks);
    CHECK_TRUE(d.cooks_distance.size() == 10);
    CHECK_TRUE(d.leverage.size() == 10);
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cmake --build core/build 2>&1 | tail -5
```

Expected: FAIL at compile, `no member named 'gmm_iterations'`.

- [ ] **Step 3: Implement**

Add to `FitResult`:

```cpp
    // --- GMM block ---------------------------------------------------------------------
    double j_stat = kNaN, j_stat_pval = kNaN;
    int gmm_iterations = 0;
    bool converged_within_tolerance = false;
    int optimizer_fallback_count = 0;
```

Add the GMM arm to `run_fit`, copying `build_and_fit_gmm` verbatim for the construction cascade. Guard the model type first, so a non-B17C model fails with the message the test expects:

```cpp
    if (target == "GMM") {
        if (model_json.value_or("type", "univariate_distribution") != "bulletin17c")
            throw std::runtime_error(
                "GMM fits a bulletin17c model only; got type '" +
                model_json.value_or("type", "univariate_distribution") + "'");
        ...
    }
```

Add the two auxiliary entry points. `run_fit_quantile_variance` rebuilds the deterministic fit and evaluates live, exactly as `ch_estimation_gmm_qvar_` does (`corehydror/src/estimation.cpp:457`), including the `1.0 - aep` non-exceedance conversion. `run_fit_diagnostics` returns:

```cpp
// Estimation diagnostics off a fit. Every field is populated where the target supports it and
// left empty where it does not: leverage/Cook's distance come from MAP and GMM, Pareto-k and
// prior influence from BayesianAnalysis.
struct FitDiagnostics {
    std::vector<double> cooks_distance;
    std::vector<double> leverage;
    std::vector<double> observation_influence;   // row-major n_obs x n_params
    std::vector<double> pareto_k;
    double max_pareto_k = FitResult::kNaN;
    std::vector<double> prior_influence;
    std::vector<std::string> prior_influence_names;
};
```

Populate it from the already-un-stubbed estimator methods: `MaximumAPosteriori::get_cooks_distance()`, `get_observation_influence()`, `compute_leverage_diagnostics()`; `BayesianAnalysis::compute_influence_diagnostics()` and `compute_prior_influence_diagnostics()`; the GMM quartet. Read `corehydror/src/analysis.cpp`'s `ch_analysis_diagnostics_run_` first and mirror which field comes from which call, so the existing `estimation_diagnostics()` numbers and the new `fit_diagnostics()` numbers cannot disagree.

- [ ] **Step 4: Run the test to verify it passes**

```bash
cmake --build core/build && ctest --test-dir core/build -R test_fit_runner --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/include/corehydro/estimation/support/fit_runner.hpp core/tests/test_fit_runner.cpp
git commit -m "feat: GMM block, quantile variance and diagnostics in the fit runner

Rejects a non-bulletin17c model by name: IGMMModel has exactly one
implementation. Quantile variance rebuilds the deterministic fit and evaluates
live, the ch_estimation_bic_ precedent."
```

---

## Task 5: Delegate the three existing harnesses to the runner

This is the task that guarantees one implementation. It must not change a single oracle value.

**Files:**
- Modify: `corehydror/src/estimation.cpp` (rewrite `ch_estimation_run_`, `ch_estimation_bayes_run_`, `ch_estimation_gmm_run_`, `ch_estimation_bic_`, `ch_estimation_gmm_qvar_` bodies)
- Modify: `corehydropy/src/bindings/estimation.cpp` (the same functions)
- Modify: `core/tests/test_fixtures.cpp` (the `model_estimation` build path around line 1999 to 2202)

**Interfaces:**
- Consumes: `run_fit`, `run_fit_quantile_variance` from Tasks 1 to 4.
- Produces: no new public names. Every existing signature and return-list key is unchanged.

- [ ] **Step 1: Capture the baseline**

```bash
Rscript -e 'testthat::test_local("corehydror")' 2>&1 | tail -5
pixi run python -m pytest corehydropy/tests -q 2>&1 | tail -3
ctest --test-dir core/build 2>&1 | tail -3
```

Record the exact counts. They are the pass criterion for this task: testthat 4446/0, pytest 884, ctest 80/80.

- [ ] **Step 2: Rewrite the R glue to delegate**

In `corehydror/src/estimation.cpp`, each function keeps its signature and builds a construct string, then reads fields off the `FitResult`. For example `ch_estimation_run_`:

```cpp
[[cpp11::register]]
list ch_estimation_run_(std::string target, std::string model_json, doubles dataset,
                        std::string optimizer, int sample_size, int seed) {
    std::vector<double> data(dataset.begin(), dataset.end());
    std::string construct = "{\"model\":" + model_json + ",\"optimizer\":\"" + optimizer + "\"}";
    support::FitResult r = support::run_fit(target, construct, data);

    int n = static_cast<int>(r.parameters.size());
    writable::doubles parameters(n);
    for (int i = 0; i < n; ++i) parameters[i] = r.parameters[static_cast<std::size_t>(i)];
    writable::doubles_matrix<by_column> covariance(n, n), correlation(n, n);
    writable::doubles standard_errors(n);
    // The pre-phase-2 contract: below two parameters the fixture arms read zeros, and no
    // fixture asserts them. The runner reports NaN, so map NaN back to 0 HERE, in the fixture
    // path only, and leave the user-facing path honest.
    ...
}
```

Do the seeded post-fit draw (`sample_size > 0`) the same way it is done now, by rebuilding the model from `r.model_spec` and calling `simulate_flat`. Keep `simulate_flat` where it is.

The NaN-to-zero mapping is the one place the fixture path and the user path deliberately differ. Put a comment on it saying so, referencing this plan.

- [ ] **Step 3: Run the R fixture suite to verify no oracle moved**

```bash
Rscript -e 'cpp11::cpp_register("corehydror")'
R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")' 2>&1 | tail -5
```

Expected: 4446 passing, 0 failing. Any difference is a regression, not an improvement: stop and reconcile before continuing.

- [ ] **Step 4: Rewrite the Python glue to delegate, the same way**

```bash
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q 2>&1 | tail -3
```

Expected: 884 passing.

- [ ] **Step 5: Rewrite the C++ fixture runner's estimation path**

`core/tests/test_fixtures.cpp` builds the estimator itself around line 1999. Replace the construction with a `run_fit` call and hold a `FitResult` in `EstimationCase`; `dispatch_estimation` then reads fields instead of calling estimator methods. Keep `bic` and `quantile_variance` on their live-rebuild path.

```bash
cmake --build core/build && ctest --test-dir core/build 2>&1 | tail -3
```

Expected: 80/80, and `test_fixtures` reports the same check count as before the task.

- [ ] **Step 6: Run the oracle gate**

```bash
python3 tools/verify_oracles.py 2>&1 | tail -5
```

Expected: 4542 reproduced, 0 failed, 11 skipped. Unchanged, because no fixture changed.

- [ ] **Step 7: Commit**

```bash
git add corehydror/src/estimation.cpp corehydropy/src/bindings/estimation.cpp core/tests/test_fixtures.cpp corehydror/R/cpp11.R corehydror/src/cpp11.cpp
git commit -m "refactor: drive every fixture harness through the shared fit runner

The four fixture glue functions keep their signatures and return shapes and
compute nothing themselves. One implementation of a fit now backs the C++
runner, both bindings and the emitter. No oracle value moves: 4542 reproduced,
testthat 4446/0, pytest 884, ctest 80/80."
```

---

## Task 6: The R fit surface, MLE and MAP

**Files:**
- Modify: `corehydror/src/estimation.cpp` (add `ch_fit_run_`)
- Create: `corehydror/R/fit.R`
- Create: `corehydror/tests/testthat/test-fit.R`
- Modify: `corehydror/NAMESPACE` and `corehydror/R/cpp11.R` (regenerated, not hand-edited)

**Interfaces:**
- Consumes: `run_fit` from Task 1; `analysis_input()` (`corehydror/R/analysis.R:9`); `to_spec_json()` and `spec_array()` (`corehydror/R/spec.R`); `check_model()` (`corehydror/R/model.R:240`).
- Produces:
  - `ch_fit_run_(target, construct_json, dataset)` returning a named list with the `FitResult` fields, matrices already reshaped, plus `chain_dims`.
  - `fit_mle(model, distribution = NULL, optimizer = "NelderMead", hessian = TRUE, profile = FALSE, profile_bins = 100)` and `fit_map()` with the same signature, both returning a `corehydro_fit`.
  - `new_fit(result, spec_type)`, internal.
  - S3 methods `print.corehydro_fit`, `summary.corehydro_fit`, `coef.corehydro_fit`, `vcov.corehydro_fit`, `logLik.corehydro_fit`, `confint.corehydro_fit`.

- [ ] **Step 1: Write the failing test**

Create `corehydror/tests/testthat/test-fit.R`:

```r
# Behavioural tests for the fit surface. Oracle VALUES live in fixtures/ and are asserted by
# test-fixtures.R; this file asserts argument handling, error messages, object shape, and the
# base-generic integration.

peaks <- c(12500, 15300, 9870, 21000, 18400, 11200, 26800, 14100, 19500, 11600)

test_that("fit_mle returns a corehydro_fit with the common surface", {
  f <- fit_mle(model_univariate("Normal", peaks))
  expect_s3_class(f, "corehydro_fit")
  expect_equal(f$method, "MaximumLikelihood")
  expect_length(f$parameters, 2L)
  expect_named(f$parameters)
  expect_true(f$converged)
  expect_equal(f$status, "Success")
  expect_true(is.finite(f$log_likelihood))
  expect_equal(dim(f$covariance), c(2L, 2L))
  expect_equal(f$nobs, 10L)
})

test_that("the vector convenience path matches the model path exactly", {
  expect_identical(
    fit_mle(peaks, "Normal")$parameters,
    fit_mle(model_univariate("Normal", peaks))$parameters
  )
})

test_that("base generics work off logLik", {
  f <- fit_mle(model_univariate("Normal", peaks))
  expect_equal(as.numeric(coef(f)), as.numeric(f$parameters))
  expect_equal(vcov(f), f$covariance)
  expect_equal(as.numeric(logLik(f)), f$log_likelihood)
  expect_equal(attr(logLik(f), "df"), 2L)
  expect_equal(attr(logLik(f), "nobs"), 10L)
  # The point of carrying df and nobs: base AIC/BIC agree with the core's own values.
  expect_equal(AIC(f), f$aic, tolerance = 1e-10)
  expect_equal(BIC(f), f$bic, tolerance = 1e-10)
})

test_that("hessian = FALSE skips the covariance stack but still fits", {
  f <- fit_mle(model_univariate("Normal", peaks), hessian = FALSE)
  expect_null(f$covariance)
  expect_null(f$standard_errors)
  expect_true(is.finite(f$log_likelihood))
})

# NOTE: the sub-two-parameter NaN covariance guard is NOT testable from R. Every
# single-parameter family in the factory (Rayleigh, Poisson, ChiSquared, Geometric, Bernoulli,
# Deterministic) fails to build as a UnivariateDistributionModel: the constructor calls
# set_default_parameters(), which dynamic_casts to IMaximumLikelihoodEstimation and throws,
# and none of the six implements it. The guard is covered in C++ by test_fit_runner.cpp
# against a test-double ModelBase. Do not try to write an R test for it.

test_that("confint returns profile intervals bracketing the estimate", {
  f <- fit_mle(model_univariate("Normal", peaks), profile = TRUE, profile_bins = 20)
  ci <- confint(f, level = 0.9)
  expect_equal(dim(ci), c(2L, 2L))
  expect_true(all(ci[, 1] <= f$parameters))
  expect_true(all(ci[, 2] >= f$parameters))
})

test_that("confint computes on demand when the fit was built without profile", {
  m <- model_univariate("Normal", peaks)
  expect_equal(
    confint(fit_mle(m), level = 0.9),
    confint(fit_mle(m, profile = TRUE), level = 0.9)
  )
})

test_that("the fit carries a fitted model that simulates identically", {
  f <- fit_mle(model_univariate("Normal", peaks))
  expect_s3_class(f$model, "corehydro_model")
  expect_identical(
    model_simulate(f, n = 25, seed = 7),
    model_simulate(f$model, n = 25, seed = 7)
  )
})

test_that("a fit round-trips through save and load", {
  f <- fit_mle(model_univariate("Normal", peaks))
  path <- tempfile(fileext = ".rds")
  saveRDS(f, path)
  expect_identical(readRDS(path)$parameters, f$parameters)
})

test_that("argument errors name the offending value", {
  m <- model_univariate("Normal", peaks)
  expect_error(fit_mle(m, optimizer = "Simplexx"), "Simplexx")
  expect_error(fit_mle(peaks), "distribution")
})
```

- [ ] **Step 2: Run it to verify it fails**

```bash
Rscript -e 'testthat::test_local("corehydror", filter = "fit")' 2>&1 | tail -10
```

Expected: FAIL, `could not find function "fit_mle"`.

- [ ] **Step 3: Add the glue**

In `corehydror/src/estimation.cpp`, add one registered function. It reshapes the flat vectors into R matrices and returns everything; the verbs pick what they need.

```cpp
// The user-facing fit entry point (phase 2). ONE function for all four estimators: the R verbs
// in R/fit.R assemble `construct_json` and read the named list back. Shape only -- every
// number comes from the shared runner, the same code path ch_estimation_run_ delegates to.
[[cpp11::register]]
list ch_fit_run_(std::string target, std::string construct_json, doubles dataset) {
    std::vector<double> data(dataset.begin(), dataset.end());
    support::FitResult r = support::run_fit(target, construct_json, data);
    // ... pack every FitResult field into a writable::list, reshaping covariance/correlation
    // into doubles_matrix<by_column> and leaving draws flat alongside chain_dims ...
}
```

Then `Rscript -e 'cpp11::cpp_register("corehydror")'`.

- [ ] **Step 4: Write `corehydror/R/fit.R`**

The file opens with the internal construct assembler and the fit constructor, then the two verbs, then the S3 methods. The assembler resolves the dual first argument through the existing `analysis_input()` so the vector path and the model path cannot drift:

```r
# Internal: assemble the construct the C++ fit runner parses, from the dual first argument
# every verb takes (a corehydro_model, or a numeric vector plus a distribution name).
fit_input <- function(model, distribution, settings) {
  if (!inherits(model, "corehydro_model") && is.null(distribution)) {
    stop("give a `distribution` name when `model` is a plain numeric vector, e.g. ",
      "fit_mle(peaks, \"Normal\")",
      call. = FALSE
    )
  }
  input <- analysis_input(model, function() {
    list(family = as.character(distribution), dataset = "data")
  })
  construct <- c(list(model = NULL), settings)
  json <- paste0(
    "{\"model\":", input$json, ",",
    substring(to_spec_json(settings), 2L)
  )
  list(json = json, dataset = input$dataset)
}
```

Assemble the JSON by serializing the settings list with `to_spec_json()` and splicing the already-serialized model spec, rather than round-tripping the spec through R structures again. Write it so an empty settings list still produces valid JSON, and add a test for that.

Each verb validates its own arguments before calling, so the error comes from R with a clear message:

```r
#' Maximum likelihood fit
#'
#' Fit a model by maximum likelihood and return a fit object carrying the parameter estimates,
#' the Hessian-based covariance, and optimizer bookkeeping. Wraps the shared C++
#' `MaximumLikelihood` ported from USACE-RMC RMC.BestFit.
#'
#' @param model a [model_univariate()] (or any `model_*()`) object, or a plain numeric vector of
#'   observations together with `distribution`. A model can bring censored observations (see
#'   [analysis_data()]), nonstationary trends (see [trend()]), and parameter bounds or priors
#'   (see [model_parameter()]).
#' @param distribution distribution family name, required only when `model` is a numeric vector.
#' @param optimizer one of `"NelderMead"` (default), `"Brent"`, `"BFGS"`, `"Powell"`,
#'   `"DifferentialEvolution"`, `"MultilevelSingleLinkage"`.
#' @param hessian logical; compute the covariance, standard errors and correlation. `TRUE` by
#'   default. A model with fewer than two parameters reports `NA` for all three.
#' @param profile logical; also compute the profile likelihood and profile confidence intervals.
#'   `FALSE` by default because each costs `profile_bins * length(parameters)` likelihood
#'   evaluations.
#' @param profile_bins number of bins in each parameter's profile.
#' @return An object of class `corehydro_fit`. See [fit_bayesian()] for the Bayesian surface.
#' @seealso [fit_map()], [fit_bayesian()], [fit_gmm()], [fit_diagnostics()].
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' f <- fit_mle(model_univariate("LogPearsonTypeIII", peaks))
#' coef(f)
#' AIC(f)
fit_mle <- function(model, distribution = NULL, optimizer = "NelderMead", hessian = TRUE,
                    profile = FALSE, profile_bins = 100) {
  ...
}
```

`fit_map()` is the same body with `"MaximumAPosteriori"`. Do not factor them into one function with a target argument: the whole point of the shape decision is two documented verbs. A shared internal `fit_optimized(target, ...)` that both call is fine and is the DRY answer.

The S3 methods:

```r
#' @export
logLik.corehydro_fit <- function(object, ...) {
  structure(object$log_likelihood,
    df = length(object$parameters), nobs = object$nobs, class = "logLik"
  )
}

#' @export
vcov.corehydro_fit <- function(object, ...) object$covariance

#' @export
coef.corehydro_fit <- function(object, ...) object$parameters

#' @export
confint.corehydro_fit <- function(object, parm, level = 0.95, ...) { ... }
```

`confint` recomputes when the fit has no profile block, by re-running the fit with `profile = TRUE` and the requested `alpha = 1 - level`. Say so in the roxygen and add the comment that this is the `ch_estimation_bic_` lazy-rebuild precedent.

`print.corehydro_fit` follows `print.corehydro_model` (`corehydror/R/model.R:251`): one summary line, then the parameter table, then convergence.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
Rscript -e 'cpp11::cpp_register("corehydror")'
R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror", filter = "fit")' 2>&1 | tail -10
```

Expected: all tests in `test-fit.R` pass, 0 failures.

- [ ] **Step 6: Run the whole R suite**

```bash
Rscript -e 'testthat::test_local("corehydror")' 2>&1 | tail -5
```

Expected: 4446 plus the new tests, 0 failures.

- [ ] **Step 7: Commit**

```bash
git add corehydror/R/fit.R corehydror/tests/testthat/test-fit.R corehydror/src/estimation.cpp corehydror/src/cpp11.cpp corehydror/R/cpp11.R corehydror/NAMESPACE corehydror/man
git commit -m "feat: fit_mle and fit_map in R

A corehydro_fit carrying the fitted model spec, with coef/vcov/confint/logLik
so base AIC and BIC work with no method of their own. Profile intervals are
computed on demand when the fit was built without them."
```

---

## Task 7: The rest of the R surface

**Files:**
- Modify: `corehydror/R/fit.R`, `corehydror/tests/testthat/test-fit.R`
- Modify: `corehydror/src/estimation.cpp` (add `ch_fit_diagnostics_`, `ch_fit_quantile_variance_`)

**Interfaces:**
- Consumes: Tasks 3, 4 and 6.
- Produces: `fit_bayesian()`, `fit_gmm()`, `fit_diagnostics()`, `quantile_variance()`, and `summary.corehydro_fit`.

- [ ] **Step 1: Write the failing test**

Append to `corehydror/tests/testthat/test-fit.R`:

```r
test_that("fit_bayesian returns draws as [iteration, chain, parameter]", {
  f <- fit_bayesian(model_univariate("Normal", peaks),
    sampler = "DEMCz", chains = 4, iterations = 200, output_length = 500, seed = 12345
  )
  expect_s3_class(f, "corehydro_fit")
  expect_equal(length(dim(f$draws)), 3L)
  expect_equal(dim(f$draws)[2], 4L)                       # chains on axis 2
  expect_equal(dim(f$draws)[3], length(f$parameters))     # parameters on axis 3
  expect_equal(dimnames(f$draws)[[3]], names(f$parameters))
  expect_equal(nrow(f$summary), length(f$parameters))
  expect_true(all(c("rhat", "ess", "median") %in% names(f$summary)))
  expect_length(f$acceptance_rates, 4L)
  expect_true(is.finite(f$dic))
})

test_that("a seeded Bayesian fit is reproducible", {
  m <- model_univariate("Normal", peaks)
  a <- fit_bayesian(m, sampler = "DEMCz", iterations = 100, seed = 99)
  b <- fit_bayesian(m, sampler = "DEMCz", iterations = 100, seed = 99)
  expect_identical(a$draws, b$draws)
})

test_that("fit_bayesian rejects a sampler BayesianAnalysis cannot construct", {
  expect_error(fit_bayesian(model_univariate("Normal", peaks), sampler = "HMC"),
    "mcmc_sample", fixed = TRUE
  )
})

test_that("fit_gmm requires a bulletin17c model", {
  expect_error(fit_gmm(model_univariate("Normal", peaks)), "bulletin17c")
})

test_that("fit_gmm returns the GMM bookkeeping", {
  f <- fit_gmm(model_bulletin17c(peaks))
  expect_equal(f$method, "GMM")
  expect_true(f$gmm_iterations > 0)
  expect_true(is.na(f$j_stat_pval))  # just-identified by construction
  expect_true(quantile_variance(f, 0.01) > 0)
})

test_that("fit_diagnostics returns one value per observation", {
  d <- fit_diagnostics(fit_map(model_univariate("Normal", peaks)))
  expect_length(d$cooks_distance, length(peaks))
})
```

- [ ] **Step 2: Run it to verify it fails**

```bash
Rscript -e 'testthat::test_local("corehydror", filter = "fit")' 2>&1 | tail -10
```

Expected: FAIL, `could not find function "fit_bayesian"`.

- [ ] **Step 3: Implement**

`fit_bayesian()` validates the sampler in R first (`match.arg(sampler, c("DEMCzs", "DEMCz", "ARWMH", "NUTS"))` produces a message that does not mention `mcmc_sample()`, so write an explicit check instead, matching the C++ message), validates the `...` knobs against the sampler, assembles the construct, and reshapes:

```r
  # The runner returns chains flattened CHAIN-major: (chain, iteration, parameter). Permute to
  # [iteration, chain, parameter], the axis order posterior::as_draws_array expects, so a user
  # can hand `draws` straight to posterior or coda with no reshaping.
  d <- res$chain_dims
  draws <- aperm(array(res$draws, dim = c(d[3], d[2], d[1])), c(2L, 3L, 1L))
  dimnames(draws) <- list(NULL, paste0("chain", seq_len(d[1])), res$parameter_names)
```

Verify that permutation against the flattening the runner documents rather than trusting this line: the test asserting `dim(f$draws)[2] == 4` with 4 chains and 2 parameters catches a transposition only if the two axes differ in length, so also add a test with `chains = 3` and a 2-parameter model.

`summary.corehydro_fit` prints the parameter table with standard errors for an optimized fit and with R-hat and ESS for a Bayesian one.

The knob validation table, written once:

```r
# Sampler-specific knobs, by sampler. A knob the chosen sampler ignores is an error rather than
# a silent no-op, because a silently dropped tuning argument looks like the sampler is broken.
sampler_knobs <- list(
  DEMCz  = c("jump", "jump_threshold", "noise"),
  DEMCzs = c("jump", "jump_threshold", "snooker_threshold", "noise"),
  ARWMH  = c("scale", "beta"),
  NUTS   = c("max_tree_depth", "scale")
)
```

Confirm each list against the sampler's own constructor in `core/include/corehydro/numerics/sampling/mcmc/` and against `BayesianAnalysis::set_up_sampler` (`bayesian_analysis.hpp:429`) before writing it down. If a knob is read by `set_up_sampler` for a sampler not listed here, add it; the table must match the code, not this plan.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
Rscript -e 'cpp11::cpp_register("corehydror")'
R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")' 2>&1 | tail -5
```

Expected: all pass, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add corehydror/R corehydror/src corehydror/NAMESPACE corehydror/man
git commit -m "feat: fit_bayesian, fit_gmm, fit_diagnostics and quantile_variance in R

Draws come back as [iteration, chain, parameter], the axis order
posterior::as_draws_array wants, so neither package needs a conversion
dependency. A sampler knob the chosen sampler ignores is an error."
```

---

## Task 8: The Python surface

**Files:**
- Modify: `corehydropy/src/bindings/estimation.cpp` (add `fit_run`, `fit_diagnostics`, `fit_quantile_variance`)
- Create: `corehydropy/src/corehydropy/fit.py`
- Create: `corehydropy/tests/test_fit.py`
- Modify: `corehydropy/src/corehydropy/__init__.py`

**Interfaces:**
- Consumes: the runner from Tasks 1 to 4; `_analysis_input` (`corehydropy/src/corehydropy/analysis.py:38`); `Model` (`models.py:213`).
- Produces: `fit_mle`, `fit_map`, `fit_bayesian`, `fit_gmm`, and the `Fit` class with properties `parameters` (dict), `parameter_names`, `covariance`, `standard_errors`, `correlation`, `draws`, `posterior`, `log_likelihood`, `aic`, `bic`, `nobs`, `converged`, `status`, `function_evaluations`, `model`, plus the Bayesian and GMM blocks, and methods `confint(level=0.9)`, `summary()`, `diagnostics()`, `quantile_variance(aep)`, `to_model()`, `to_json()`, `__repr__`.

- [ ] **Step 1: Write the failing test**

Create `corehydropy/tests/test_fit.py` as the twin of `test-fit.R`, one test per R test, same order, same names where Python allows. Add the cross-language test:

```python
def test_bayesian_draws_axis_order_matches_r():
    """Draws are (iteration, chain, parameter), the same order corehydror returns."""
    f = fit_bayesian(model_univariate("Normal", PEAKS), sampler="DEMCz",
                     chains=3, iterations=200, seed=12345)
    assert f.draws.ndim == 3
    assert f.draws.shape[1] == 3
    assert f.draws.shape[2] == len(f.parameters)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pixi run python -m pytest corehydropy/tests/test_fit.py -q 2>&1 | tail -5
```

Expected: FAIL, `ImportError: cannot import name 'fit_mle'`.

- [ ] **Step 3: Implement**

The binding mirrors `ch_fit_run_` and returns a dict. `fit.py` mirrors `fit.R` argument for argument. Reshape the draws with numpy rather than by hand:

```python
    n_chains, n_iter, n_params = result["chain_dims"]
    # The runner flattens chain-major; transpose to (iteration, chain, parameter), the same
    # axis order corehydror returns, so a reader of either package sees one convention.
    draws = np.asarray(result["draws"]).reshape(n_chains, n_iter, n_params).transpose(1, 0, 2)
```

Every public function gets a numpydoc docstring with a `Parameters`, `Returns` and `Examples` section, matching `analysis.py`.

Export the four verbs and `Fit` from `__init__.py`, following whatever `__all__` convention is already there.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q 2>&1 | tail -3
```

Expected: 884 plus the new tests, 0 failures.

- [ ] **Step 5: Verify R and Python agree on a seeded fit**

```bash
Rscript -e 'library(corehydror); f <- fit_bayesian(model_univariate("Normal", c(12500,15300,9870,21000,18400,11200,26800,14100,19500,11600)), sampler="DEMCz", iterations=200, seed=12345); cat(sprintf("%.17g\n", f$posterior_mean))'
pixi run python -c "
from corehydropy import fit_bayesian, model_univariate
f = fit_bayesian(model_univariate('Normal', [12500,15300,9870,21000,18400,11200,26800,14100,19500,11600]), sampler='DEMCz', iterations=200, seed=12345)
print('\n'.join('%.17g' % v for v in f.posterior_mean))"
```

Expected: identical to every digit. If they differ, stop: something is not going through the shared runner.

- [ ] **Step 6: Commit**

```bash
git add corehydropy/src corehydropy/tests/test_fit.py
git commit -m "feat: the fit surface in Python

Mirrors corehydror argument for argument, with draws on the same
(iteration, chain, parameter) axes. A seeded fit agrees with R to every digit."
```

---

## Task 9: Fixtures and the oracle gate

**Files:**
- Create: `fixtures/estimation/fit_profile.json`, `fit_optimizers.json`, `fit_bayes_diagnostics.json`, `fit_cross_language.json`
- Modify: `fixtures/README.md`, `tools/oracle_emitter/Program.cs`
- Modify: `core/tests/test_fixtures.cpp`, `corehydror/tests/testthat/test-fixtures.R`, `corehydropy/tests/test_fixtures.py` (new assertion arms in each)

**Interfaces:**
- Consumes: everything above.
- Produces: new `model_estimation` assertion methods `profile_lower [i]`, `profile_upper [i]`, `profile_value [param, bin, col]`, `function_evaluations`, `status_is [name]`, `rhat [i]`, `ess [i]`, `acceptance_rate [chain]`, `posterior_median [i]`, `posterior_sd [i]`, `posterior_lower [i]`, `posterior_upper [i]`, `pareto_k [i]`, `max_pareto_k`, `nobs`, `prior_log_likelihood`.

- [ ] **Step 1: Add the assertion arms to the three runners first, with no fixture**

Add each method to `dispatch_estimation` in all three runners. Run the suites; nothing changes, because no fixture uses them yet. This is deliberate: wiring the arms before writing the fixture means a typo in the arm shows up as a failing new fixture rather than as a silently skipped assertion.

```bash
ctest --test-dir core/build -R test_fixtures
Rscript -e 'testthat::test_local("corehydror", filter = "fixtures")' 2>&1 | tail -3
pixi run python -m pytest corehydropy/tests/test_fixtures.py -q 2>&1 | tail -3
```

Expected: unchanged counts.

- [ ] **Step 2: Write the fixture files with placeholder expected values**

Model the first three on `fixtures/estimation/mle_normal_smoke.json`. Set every `expected` to `0` for now and every `tol` to the value the neighbouring fixture uses. Add the `source` field naming the C# file the oracle comes from.

`fit_cross_language.json` is a different kind: a `short_exact` digest case in the shape phase 1 used, asserting a handful of values from one seeded Bayesian fit of a censored, nonstationary model. Its job is to prove R and Python agree, so every runner asserts the same digest. Read the phase-1 digest fixture it is modelled on before writing it (`git log --oneline --diff-filter=A -- fixtures/` finds the ones phase 1 added).

- [ ] **Step 3: Teach the emitter the new methods**

In `tools/oracle_emitter/Program.cs`, extend `DispatchMlMap` and the `BayesianAnalysis` arm of `DispatchEstimation` with the new method names, calling the real C# accessors. Every method must have a driver: a fixture target with no emitter arm silently reports as reproduced without checking anything, which is the failure mode recorded in the phase 9a notes.

- [ ] **Step 4: Emit the real values and fill them in**

```bash
python3 tools/verify_oracles.py 2>&1 | tail -20
```

The gate reports each placeholder as a mismatch with the real C# value. Replace each `expected` with the reported value at full precision, then re-run.

Expected on the second run: `4542 + N reproduced, 0 failed, 11 skipped`, where N is the number of new assertions. If a value will not reproduce, do not loosen the tolerance: record the reason in `docs/upstream-csharp-issues.md` and assert a structural invariant instead.

- [ ] **Step 5: Run all four runners against the new fixtures**

```bash
ctest --test-dir core/build
Rscript -e 'testthat::test_local("corehydror")' 2>&1 | tail -5
pixi run python -m pytest corehydropy/tests -q 2>&1 | tail -3
```

Expected: all pass, with the R and Python check counts up by the new assertions.

- [ ] **Step 6: Document the new methods in `fixtures/README.md`**

Add each to the `model_estimation` method list with its argument signature and one line on what it reads, matching the existing entries' style.

- [ ] **Step 7: Commit**

```bash
git add fixtures core/tests/test_fixtures.cpp corehydror/tests/testthat/test-fixtures.R corehydropy/tests/test_fixtures.py tools/oracle_emitter/Program.cs
git commit -m "test: pin the new estimation surface against the real C# library

Profile intervals and grids, optimizer status and function evaluations, R-hat,
ESS, acceptance rates, the posterior summary and Pareto-k, plus one fit under
each of the six optimizers. Every new method has an emitter driver."
```

---

## Task 10: Documentation and the worked example

**Files:**
- Modify: `corehydror/_pkgdown.yml`, `site/_quarto.yml`, `site/examples/index.qmd`, the coverage page
- Create: `site/examples/25-estimation-methods/r.qmd`, `site/examples/25-estimation-methods/python.ipynb`

**Interfaces:**
- Consumes: every export from Tasks 6 to 8.

- [ ] **Step 1: Add the reference entries**

Add an "Estimation" section to `corehydror/_pkgdown.yml` listing `fit_mle`, `fit_map`, `fit_bayesian`, `fit_gmm`, `fit_diagnostics`, `quantile_variance`, and the S3 methods if the existing file lists methods. Add the matching section to `quartodoc.sections` in `site/_quarto.yml`.

- [ ] **Step 2: Verify the docs build, which is the test for this step**

```bash
pixi run docs 2>&1 | tail -20
```

Expected: builds clean. pkgdown fails loudly on a missing reference-index entry, so a clean build is the proof that every export is documented.

- [ ] **Step 3: Write the R example**

`site/examples/25-estimation-methods/r.qmd`, following the front matter and structure of an existing example (read `site/examples/23-censored-flood-frequency/r.qmd`). Content: one censored record; `fit_mle`, `fit_map` and `fit_bayesian` on the same model; a table comparing AIC, BIC and DIC; profile intervals set against credible intervals; R-hat and ESS. End with the executable reproduction check every other example ends with, comparing against literals at 1e-15 relative tolerance.

- [ ] **Step 4: Render the R example and commit the freeze**

```bash
pixi run docs 2>&1 | tail -5
git status --porcelain site/_freeze | head
```

Expected: `site/_freeze/` shows the new example's freeze directory.

- [ ] **Step 5: Write and execute the Python notebook**

Write `python.ipynb` as the twin, then:

```bash
pixi run jupyter nbconvert --to notebook --execute --inplace site/examples/25-estimation-methods/python.ipynb
```

Expected: exits 0. The notebook is committed with its outputs; Quarto renders stored outputs and never re-executes.

- [ ] **Step 6: List the pair in the index and the coverage page**

- [ ] **Step 7: Commit**

```bash
git add corehydror/_pkgdown.yml site
git commit -m "docs: estimation reference entries and a worked example pair

The same censored record fit by MLE, MAP and Bayesian, compared on AIC, BIC and
DIC, with profile intervals set against credible intervals."
```

---

## Task 11: Release chores and the full verification sweep

**Files:**
- Modify: `CHANGELOG.md`, `corehydror/DESCRIPTION`, `corehydropy/pyproject.toml`, the core version stamp, `.claude/CLAUDE.md`

- [ ] **Step 1: Bump all three versions to 0.4.0**

Find the core version stamp the same way phase 1 did (it had drifted; grep for `0.3.0` across `core/`).

```bash
grep -rn "0\.3\.0" core corehydror/DESCRIPTION corehydropy/pyproject.toml | grep -v Rcheck
```

- [ ] **Step 2: Write the CHANGELOG entry**

Follow the 0.3.0 entry's structure: a lead paragraph saying what became possible, then `### Added`, `### Fixed` (the single-parameter covariance and the fixture-shaped error messages belong here), `### Documentation`.

- [ ] **Step 3: Update the repo context file**

Add a paragraph to `.claude/CLAUDE.md` describing the estimation layer, in the style of the existing phase paragraphs: where the runner lives, what the four verbs are, and the final verification numbers.

- [ ] **Step 4: Run the full verification sequence**

```bash
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build
python3 tools/verify_oracles.py
Rscript -e 'cpp11::cpp_register("corehydror")'
R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
R CMD build corehydror && R CMD check --as-cran corehydror_*.tar.gz
pixi run docs
```

Every one must pass. Record the actual numbers; they go in the commit message and, later, the PR body. The `R CMD check` baseline is one NOTE (the pre-existing long-path note listing vendored-core headers). A second NOTE is a regression to fix, not to report.

- [ ] **Step 5: Run the end-to-end cross-language check**

```r
d <- analysis_data(exact = peaks, mgbt_low_outliers = TRUE)
m <- model_univariate("LogPearsonTypeIII", d)
mle   <- fit_mle(m, optimizer = "BFGS", profile = TRUE)
bayes <- fit_bayesian(m, sampler = "DEMCz", iterations = 500, seed = 12345)
```

Run the identical Python call and compare `coef`, `confint`, `AIC`, `bayes$summary` and `dim(bayes$draws)`. Every number must agree to every digit. `peaks` must have at least 10 observations, since MGBT requires it.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md corehydror/DESCRIPTION corehydropy/pyproject.toml core .claude/CLAUDE.md
git commit -m "chore: release 0.4.0"
```

- [ ] **Step 7: Report, do not push**

Summarize what landed, the before and after verification numbers, anything that widened beyond this plan and why, and anything left undone. Do not push and do not open a PR without being asked.

---

## Notes for the implementer

- **The one place the fixture path and the user path differ on purpose** is the sub-two-parameter covariance: the fixture glue maps the runner's NaN back to zero to preserve the pre-phase-2 contract, and the user path reports NaN. It is commented in `corehydror/src/estimation.cpp`. Do not "fix" one to match the other.
- **Seeded DEMCz and DEMCzs with `thinning_interval > 1` are not oracle-guaranteed** C# against C++ (a carried-forward known issue). Every shipped fixture uses thin=1. Keep it that way in the new fixtures.
- **B17C GMM is always just-identified**, so `j_stat_pval` is structurally NaN and no over-identified oracle is reachable. Do not chase it.
- If a seeded Bayesian curve will not reproduce against C#, check the phase 9a precedent in `docs/upstream-csharp-issues.md` before assuming a port bug: measure the conditioning and drive the real C# at the failing configuration. Assert structural invariants with no `oracle_skip` and no loosened tolerance, exactly as the AR, MA and Mixture analyses do.
