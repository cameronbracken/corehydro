# Distribution Layer Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the five composite univariate families, the seven bivariate copulas, and the five multivariate distributions callable from R and Python, over one nested distribution-spec grammar and one shared C++ runner.

**Architecture:** A new `dist_spec.hpp` builds any distribution object from a nested JSON spec, and `dist_runner.hpp` evaluates a named method against it, returning a flat `DistResult`. Four callers drive the runner and none owns evaluation logic: the cpp11 glue, the pybind11 glue, `core/tests/test_fixtures.cpp`, and `tools/oracle_emitter/Program.cs`. This mirrors phase 2's `fit_runner.hpp` exactly. The R and Python surfaces are stateless classed lists and wrapper classes that serialize to that grammar.

**Tech Stack:** C++17 (no external dependencies), cpp11 for R, pybind11 for Python, testthat, pytest, ctest, dotnet 10 for the oracle gate, Quarto and pkgdown for docs.

**Spec:** `docs/superpowers/specs/2026-08-12-distribution-layer-completion-design.md`. Read it before starting.

## Global Constraints

- Branch: `surface-distribution-layer`, already cut from `main` with the spec committed. Do not push and do not open a PR without being asked.
- Commits are GPG-signed as `Cam Bracken <cameron.bracken@pm.me>`. **No `Co-Authored-By` trailer on any commit.**
- No new external C++ dependency in `core/`. No new R dependency (`corehydror` has zero runtime dependencies; `jsonlite` is `Suggests` and test-only). No new Python dependency (numpy only; never pandas).
- Never use `M_PI`; use `corehydro::numerics::kPi`. Never name a namespace alias `gamma` or `stat`.
- Oracle values live ONLY in `fixtures/*.json`. Never hardcode an expected number in a test file.
- Never loosen a fixture tolerance and never add an `oracle_skip` to make a new case pass. If a value will not reproduce, record why in `docs/upstream-csharp-issues.md` and assert the structural invariant instead.
- `core/` and `fixtures/` are subtree symlinks inside both packages. Edit through `core/`; nothing needs re-syncing.
- After editing any `corehydror/src/*.cpp`, run `Rscript -e 'cpp11::cpp_register("corehydror")'` before installing. After a change that alters a class layout, install with `R CMD INSTALL --preclean corehydror`.
- Every new R export must appear in `corehydror/_pkgdown.yml`. Every new Python export must appear in the `quartodoc.sections` of `site/_quarto.yml`. pkgdown errors on a missing entry.
- This phase adds NO numerical code. Every verb it exposes already exists in the core. If a verb appears to be missing, it is missing upstream too: surface an error, do not implement it.
- Target version at the end of the phase: **0.5.0** in `corehydror/DESCRIPTION`, `corehydropy/pyproject.toml`, and the core version stamp.
- Baselines are measured in Task 0 and must not regress.

---

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `core/include/corehydro/numerics/distributions/support/dist_spec.hpp` | `build_univariate`, `build_copula`, `build_multivariate` from a nested JSON spec. |
| `core/include/corehydro/numerics/distributions/support/dist_runner.hpp` | `DistResult`, `run_dist`, `run_copula`, `run_mvdist`. The only place a distribution method is dispatched. |
| `core/tests/test_dist_runner.cpp` | ctest coverage of the runner: build shapes, method dispatch, guards, error paths. |
| `corehydror/src/dist_spec.cpp` | The three cpp11 entry points `ch_dist_spec_run_`, `ch_copula_run_`, `ch_mvdist_run_`. |
| `corehydropy/src/bindings/dist_spec.cpp` | The pybind11 twins `dist_spec_run`, `copula_run`, `mvdist_run`. |
| `corehydror/R/copula.R` | `copula()`, the copula verbs, `copula_fit()`, `print.corehydro_copula`. |
| `corehydror/R/mvdist.R` | The five multivariate constructors, the verbs, `print.corehydro_mvdist`. |
| `corehydropy/src/corehydropy/copula.py` | `Copula`, `copula()`, `copula_fit()`, `copula_names()`. |
| `corehydropy/src/corehydropy/mvdist.py` | `MultivariateDistribution` and the five constructors. |
| `corehydror/tests/testthat/test-copula.R` | Behavioural tests for the R copula surface. |
| `corehydror/tests/testthat/test-mvdist.R` | Behavioural tests for the R multivariate surface. |
| `corehydropy/tests/test_copula.py` | Python twin. |
| `corehydropy/tests/test_mvdist.py` | Python twin. |
| `fixtures/distributions/cross_language/composite_digest.json` | Seeded composite draw digest. |
| `fixtures/distributions/cross_language/copula_digest.json` | Seeded copula pair stream digest. |
| `fixtures/distributions/cross_language/mvdist_digest.json` | Seeded MVN Latin hypercube digest. |
| `site/examples/26-copulas-and-joint-frequency/{python.ipynb, r.qmd}` | Worked example pair. |
| `site/examples/27-composite-distributions/{python.ipynb, r.qmd}` | Worked example pair. |

**Modified:**

| Path | Change |
|---|---|
| `core/CMakeLists.txt` | Register `test_dist_runner` in `BF_TESTS`. |
| `core/tests/test_fixtures.cpp` | `build_composite`, `build_copula`, `build_multivariate` and their dispatchers delegate to the runner. |
| `corehydror/R/distribution.R` | The five composite constructors, the `spec` payload, the targeted errors, `distribution_names("structured")`. |
| `corehydropy/src/corehydropy/distributions.py` | The Python twin of all of the above. |
| `corehydror/tests/testthat/test-fixtures.R` | Composite, copula, and multivariate paths delegate to the new glue; the bespoke dispatchers are deleted. |
| `corehydropy/tests/test_fixtures.py` | The same. |
| `corehydror/src/dist.cpp`, `mvd.cpp`, `copula.cpp` | Delete the fixture-only glue that the runner replaces. |
| `corehydropy/src/bindings/dist.cpp`, `mvd.cpp`, `copula.cpp` | The same. |
| `corehydropy/src/bindings/bindings.hpp`, `gev.cpp`, `corehydropy/CMakeLists.txt` | Register `dist_spec.cpp`. |
| `corehydror/NAMESPACE`, `corehydror/R/cpp11.R` | Regenerated by roxygen and `cpp11::cpp_register`. |
| `corehydropy/src/corehydropy/__init__.py` | Export the new surface. |
| `tools/oracle_emitter/Program.cs` | New assertion methods; `BuildComposite`/`BuildCopula`/`BuildMultivariate` read the same grammar. |
| `fixtures/distributions/univariate/*.json`, `copulas/*.json` | New cases for the unpinned verbs. |
| `fixtures/README.md` | The grammar, the key aliases, the new assertion methods. |
| `corehydror/_pkgdown.yml`, `site/_quarto.yml` | "Copulas" and "Multivariate distributions" sections. |
| `site/examples/index.qmd`, `site/examples/coverage.qmd` | List the two new pairs. |
| `CHANGELOG.md`, `DESCRIPTION`, `pyproject.toml`, the core version stamp | 0.5.0. |

---

## Task 0: Measure the baselines

**Files:** none. This task writes no code.

**Interfaces:**
- Produces: the four baseline numbers every later task's verification step compares against. Record them in the commit message of Task 1 and treat them as fixed for the rest of the phase.

- [ ] **Step 1: Build and run the C++ suite**

```bash
cmake -S core -B core/build && cmake --build core/build -j && ctest --test-dir core/build
```

Expected: all tests pass. Record the `N/N` figure (the spec's recorded value is 80/80).

- [ ] **Step 2: Run the oracle gate**

```bash
python3 tools/verify_oracles.py
```

Expected: `4623 reproduced, 0 failed, 11 skipped` (or whatever it prints; record it).

- [ ] **Step 3: Run the R suite**

```bash
Rscript -e 'cpp11::cpp_register("corehydror")'
R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
```

Expected: 0 failures. Record the PASS count.

- [ ] **Step 4: Run the Python suite**

```bash
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
```

Expected: 0 failures. Record the passed count.

- [ ] **Step 5: Write the numbers into the plan**

Edit this file's Global Constraints section, replacing the last bullet with the four measured figures, and leave it uncommitted until Task 1's commit picks it up.

---

## Task 1: The spec grammar and the runner, univariate and composite

**Files:**
- Create: `core/include/corehydro/numerics/distributions/support/dist_spec.hpp`
- Create: `core/include/corehydro/numerics/distributions/support/dist_runner.hpp`
- Create: `core/tests/test_dist_runner.cpp`
- Modify: `core/CMakeLists.txt` (add `test_dist_runner` to the `BF_TESTS` list, alphabetically)

**Interfaces:**
- Consumes: `corehydro::models::spec::parse_json`, `JsonValue`, `to_json_string` from `core/include/corehydro/models/json_lite.hpp`; `corehydro::numerics::distributions::create_distribution` from `base/univariate_distribution_factory.hpp`; the five composite classes.
- Produces, in namespace `corehydro::numerics::distributions::support`:
  ```cpp
  struct DistResult {
      std::vector<double> values;
      std::vector<std::string> names;
      std::string spec;
  };
  std::unique_ptr<UnivariateDistributionBase> build_univariate(const models::spec::JsonValue& spec);
  DistResult run_dist(const std::string& spec_json, const std::string& method,
                      const std::string& args_json);
  ```
  Every later task depends on these exact names. `DistResult` field names are the contract the R and Python glue read.

**Grammar this task defines** (documented in the header comment and in `fixtures/README.md` in Task 8):

```jsonc
// flat family
{"family": "Normal", "parameters": [0, 1]}
// the fixture spelling is accepted as an alias everywhere a spec is read
{"target": "Normal", "params": [0, 1]}

{"family": "TruncatedDistribution", "base": {"family": "Normal", "parameters": [2, 1]},
 "bounds": [1.1, 2.11]}
{"family": "Mixture", "components": [ <spec>, ... ], "weights": [0.3, 0.7],
 "zero_inflated": false, "zero_weight": 0.0}
{"family": "CompetingRisks", "components": [ <spec>, ... ],
 "minimum_of_random_variables": true, "dependency": "Independent",
 "correlation": [[1, 0.5], [0.5, 1]]}
{"family": "Empirical", "x": [...], "p": [...], "p_transform": "NormalZ",
 "p_descending": false}
{"family": "KernelDensity", "data": [...], "kernel": "Gaussian", "bandwidth": 1.5,
 "bounded_by_data": true}
// applied after construction, for the fixture set_parameters arms
{"family": "...", ..., "set_parameters": [ ... ]}
```

`data` accepts an inline array only. The fixture runner resolves its dataset names before serializing, so the runner never needs a datasets table.

- [ ] **Step 1: Write the failing test**

Create `core/tests/test_dist_runner.cpp`:

```cpp
// ctest coverage of the shared distribution runner: the spec grammar builds what it claims,
// method dispatch returns the right shape, and every guard throws with a message that names
// the thing it could not do. Oracle VALUES live in fixtures/ and are asserted by
// test_fixtures.cpp; this file asserts structure and error paths only.
#include "corehydro/numerics/distributions/support/dist_runner.hpp"

#include <cmath>
#include <string>

#include "test_helpers.hpp"

namespace supp = corehydro::numerics::distributions::support;

int main() {
    // A flat family evaluates through the same entry point as a composite.
    {
        supp::DistResult r =
            supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "pdf", "[0]");
        chtest::check_close(r.values.at(0), 0.3989422804014327, 1e-12, "normal/pdf");
        chtest::check_eq(static_cast<int>(r.values.size()), 1, "normal/pdf/size");
    }

    // The pointwise verbs vectorize: args length in, values length out.
    {
        supp::DistResult r =
            supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "cdf", "[-1,0,1]");
        chtest::check_eq(static_cast<int>(r.values.size()), 3, "normal/cdf/vectorized");
        chtest::check_close(r.values.at(1), 0.5, 1e-12, "normal/cdf/median");
    }

    // The fixture key spelling is an accepted alias.
    {
        supp::DistResult a =
            supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "pdf", "[0]");
        supp::DistResult b = supp::run_dist(R"({"target":"Normal","params":[0,1]})", "pdf", "[0]");
        chtest::check_close(a.values.at(0), b.values.at(0), 0.0, "alias/target_params");
    }

    // Composites nest to any depth: a truncated mixture is one spec.
    {
        const char* spec = R"({"family":"TruncatedDistribution",
            "base":{"family":"Mixture",
                    "components":[{"family":"Normal","parameters":[0,1]},
                                  {"family":"Normal","parameters":[5,1]}],
                    "weights":[0.5,0.5]},
            "bounds":[-2,7]})";
        supp::DistResult r = supp::run_dist(spec, "cdf", "[7]");
        chtest::check_close(r.values.at(0), 1.0, 1e-9, "truncated_mixture/cdf_at_upper");
    }

    // moments returns eight values with their names attached.
    {
        supp::DistResult r =
            supp::run_dist(R"({"family":"Normal","parameters":[3,2]})", "moments", "[]");
        chtest::check_eq(static_cast<int>(r.values.size()), 8, "moments/size");
        chtest::check_eq(static_cast<int>(r.names.size()), 8, "moments/names");
        chtest::check_eq(r.names.at(0) == "mean", true, "moments/first_name");
        chtest::check_close(r.values.at(0), 3.0, 1e-12, "moments/mean");
    }

    // A seeded draw comes back whole, so a rebuild never splits a stream.
    {
        supp::DistResult r =
            supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "random", "[5,12345]");
        chtest::check_eq(static_cast<int>(r.values.size()), 5, "random/size");
        supp::DistResult again =
            supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "random", "[5,12345]");
        chtest::check_close(r.values.at(4), again.values.at(4), 0.0, "random/reproducible");
    }

    // log_likelihood takes the whole sample in args.
    {
        supp::DistResult r = supp::run_dist(R"({"family":"Normal","parameters":[0,1]})",
                                            "log_likelihood", "[-1,0,1]");
        chtest::check_eq(static_cast<int>(r.values.size()), 1, "log_likelihood/size");
        chtest::check_eq(std::isfinite(r.values.at(0)), true, "log_likelihood/finite");
    }

    // KernelDensity takes its data inline and defaults the bandwidth to the Silverman rule.
    {
        supp::DistResult r = supp::run_dist(
            R"({"family":"KernelDensity","data":[1,2,3,4,5,6,7,8,9,10]})", "cdf", "[5.5]");
        chtest::check_close(r.values.at(0), 0.5, 0.05, "kde/cdf_at_center");
    }

    // Guards.
    chtest::check_throws([] { supp::run_dist(R"({"family":"NotAFamily"})", "pdf", "[0]"); },
                         "unknown distribution family");
    chtest::check_throws(
        [] { supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "not_a_method", "[0]"); },
        "unknown distribution method");
    chtest::check_throws(
        [] { supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "linear_moments", "[]"); },
        "linear moments");

    return chtest::summary("dist_runner");
}
```

If `chtest::check_throws` does not exist in `core/tests/test_helpers.hpp`, add it there in this task:

```cpp
// Runs `fn` and passes when it throws a std::exception whose message contains `needle`.
template <typename F>
inline void check_throws(F&& fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(needle) != std::string::npos) {
            ++passed;
            return;
        }
        fail("expected a message containing '" + needle + "', got '" + e.what() + "'");
        return;
    }
    fail("expected a throw containing '" + needle + "', but nothing was thrown");
}
```

Match the surrounding names in `test_helpers.hpp` (`passed`, `fail`) to whatever that file actually uses; read it first.

- [ ] **Step 2: Register the test and run it to verify it fails**

In `core/CMakeLists.txt`, add `test_dist_runner` to the `BF_TESTS` list, alphabetically beside the other test names.

Run:

```bash
cmake -S core -B core/build && cmake --build core/build -j 2>&1 | tail -20
```

Expected: FAIL, `fatal error: 'corehydro/numerics/distributions/support/dist_runner.hpp' file not found`.

- [ ] **Step 3: Write `dist_spec.hpp`**

Create `core/include/corehydro/numerics/distributions/support/dist_spec.hpp`:

```cpp
// corehydro ADDITION -- no upstream C# counterpart (sibling of models/model_spec.hpp and
// estimation/support/fit_runner.hpp).
//
// The one place a distribution object is built from a spec. The grammar is the fixture
// `construct` schema promoted to a first-class contract, so a fixture case, an oracle replay,
// and a user's dist_mixture() call build the identical object:
//
//   {"family": "Normal", "parameters": [0, 1]}
//   {"family": "TruncatedDistribution", "base": <spec>, "bounds": [lo, hi]}
//   {"family": "Mixture", "components": [<spec>...], "weights": [...],
//    "zero_inflated": false, "zero_weight": 0.0}
//   {"family": "CompetingRisks", "components": [<spec>...],
//    "minimum_of_random_variables": true, "dependency": "Independent", "correlation": [[...]]}
//   {"family": "Empirical", "x": [...], "p": [...], "p_transform": "NormalZ",
//    "p_descending": false}
//   {"family": "KernelDensity", "data": [...], "kernel": "Gaussian", "bandwidth": h,
//    "bounded_by_data": true}
//
// `target`/`params` are accepted as aliases of `family`/`parameters` because every fixture file
// spells them that way; renaming keys across the pinned corpus would buy nothing.
//
// An optional "set_parameters" array is applied after construction, serving the fixture arms
// that assert a value after a SetParameters round trip.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/competing_risks.hpp"
#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "corehydro/numerics/distributions/kernel_density.hpp"
#include "corehydro/numerics/distributions/mixture.hpp"
#include "corehydro/numerics/distributions/truncated_distribution.hpp"

namespace corehydro::numerics::distributions::support {

using models::spec::JsonValue;

// "family" with "target" accepted as an alias.
inline std::string spec_family(const JsonValue& s) {
    if (s.contains("family")) return s.at("family").as_string();
    if (s.contains("target")) return s.at("target").as_string();
    throw std::runtime_error("distribution spec: missing required key 'family'");
}

// "parameters" with "params" accepted as an alias; empty when neither is present.
inline std::vector<double> spec_parameters(const JsonValue& s) {
    if (s.contains("parameters")) return s.at("parameters").as_double_vector();
    if (s.contains("params")) return s.at("params").as_double_vector();
    return {};
}

inline bool is_composite_family(const std::string& family) {
    return family == "TruncatedDistribution" || family == "Empirical" ||
           family == "KernelDensity" || family == "Mixture" || family == "CompetingRisks";
}

std::unique_ptr<UnivariateDistributionBase> build_univariate(const JsonValue& spec);

namespace detail {

inline EmpiricalTransform parse_empirical_transform(const std::string& t) {
    if (t == "None") return EmpiricalTransform::None;
    if (t == "NormalZ") return EmpiricalTransform::NormalZ;
    throw std::runtime_error("unknown p_transform '" + t + "'; expected None or NormalZ");
}

inline KernelType parse_kernel_type(const std::string& k) {
    if (k == "Epanechnikov") return KernelType::Epanechnikov;
    if (k == "Gaussian") return KernelType::Gaussian;
    if (k == "Triangular") return KernelType::Triangular;
    if (k == "Uniform") return KernelType::Uniform;
    throw std::runtime_error("unknown kernel '" + k +
                             "'; expected Gaussian, Epanechnikov, Triangular or Uniform");
}

inline std::unique_ptr<UnivariateDistributionBase> build_flat(const std::string& family,
                                                              const JsonValue& spec) {
    std::unique_ptr<UnivariateDistributionBase> d;
    try {
        d = create_distribution(family);
    } catch (const std::exception&) {
        throw std::runtime_error("unknown distribution family '" + family + "'");
    }
    std::vector<double> p = spec_parameters(spec);
    if (!p.empty()) d->set_parameters(p);
    return d;
}

}  // namespace detail

// Builds any univariate distribution, flat or composite, nested to any depth.
inline std::unique_ptr<UnivariateDistributionBase> build_univariate(const JsonValue& spec) {
    std::string family = spec_family(spec);
    std::unique_ptr<UnivariateDistributionBase> out;

    if (family == "TruncatedDistribution") {
        std::unique_ptr<UnivariateDistributionBase> base = build_univariate(spec.at("base"));
        const std::vector<double> b = spec.at("bounds").as_double_vector();
        if (b.size() != 2)
            throw std::runtime_error("TruncatedDistribution: 'bounds' needs exactly two values");
        out = std::make_unique<TruncatedDistribution>(std::move(base), b[0], b[1]);
    } else if (family == "Mixture") {
        std::vector<double> weights = spec.at("weights").as_double_vector();
        std::vector<std::unique_ptr<UnivariateDistributionBase>> comps;
        for (const JsonValue& c : spec.at("components").items())
            comps.push_back(build_univariate(c));
        auto mix = std::make_unique<Mixture>(std::move(weights), std::move(comps));
        // IsZeroInflated before ZeroWeight, matching the C# Clone() initializer order: both
        // setters renormalize the component weights as a side effect.
        mix->set_is_zero_inflated(spec.value_or("zero_inflated", false));
        mix->set_zero_weight(spec.value_or("zero_weight", 0.0));
        out = std::move(mix);
    } else if (family == "CompetingRisks") {
        std::vector<std::unique_ptr<UnivariateDistributionBase>> comps;
        for (const JsonValue& c : spec.at("components").items())
            comps.push_back(build_univariate(c));
        auto cr = std::make_unique<CompetingRisks>(std::move(comps));
        cr->minimum_of_random_variables = spec.value_or("minimum_of_random_variables", true);
        if (spec.contains("dependency"))
            cr->set_dependency(parse_dependency_type(spec.at("dependency").as_string()));
        if (spec.contains("correlation")) {
            probability::Matrix2D corr;
            for (const JsonValue& row : spec.at("correlation").items())
                corr.push_back(row.as_double_vector());
            cr->set_correlation_matrix(std::move(corr));
        }
        out = std::move(cr);
    } else if (family == "Empirical") {
        std::vector<double> x = spec.at("x").as_double_vector();
        std::vector<double> p = spec.at("p").as_double_vector();
        EmpiricalTransform pt = EmpiricalTransform::NormalZ;
        if (spec.contains("p_transform"))
            pt = detail::parse_empirical_transform(spec.at("p_transform").as_string());
        out = std::make_unique<EmpiricalDistribution>(std::move(x), std::move(p), pt,
                                                      spec.value_or("p_descending", false));
    } else if (family == "KernelDensity") {
        std::vector<double> data = spec.at("data").as_double_vector();
        KernelType kt = KernelType::Gaussian;
        if (spec.contains("kernel")) kt = detail::parse_kernel_type(spec.at("kernel").as_string());
        std::unique_ptr<KernelDensity> kde;
        if (spec.contains("bandwidth"))
            kde = std::make_unique<KernelDensity>(std::move(data), kt,
                                                  spec.at("bandwidth").as_double());
        else
            kde = std::make_unique<KernelDensity>(std::move(data), kt);
        if (spec.contains("bounded_by_data"))
            kde->set_bounded_by_data(spec.at("bounded_by_data").as_bool());
        out = std::move(kde);
    } else {
        out = detail::build_flat(family, spec);
    }

    if (spec.contains("set_parameters")) out->set_parameters(spec.at("set_parameters").as_double_vector());
    return out;
}

}  // namespace corehydro::numerics::distributions::support
```

Two names above are placeholders for spellings that already exist elsewhere; replace them rather than inventing new ones. `parse_dependency_type` is `parse_dependency` at `core/tests/test_fixtures.cpp:967` (`static prob::DependencyType parse_dependency(const std::string&)`) whose body you lift verbatim into `detail`. `probability::Matrix2D` is spelled `prob::Matrix2D` in `competing_risks.hpp`, where `namespace prob = corehydro::numerics::data::probability` (`competing_risks.hpp:92`) and `set_correlation_matrix` takes it at `:136`; use the same alias.

- [ ] **Step 4: Write `dist_runner.hpp`**

Create `core/include/corehydro/numerics/distributions/support/dist_runner.hpp`:

```cpp
// corehydro ADDITION -- no upstream C# counterpart (sibling of estimation/support/fit_runner.hpp).
//
// The single place a distribution method is dispatched in this repo. Four callers drive it and
// none owns any evaluation logic: the cpp11 glue (corehydror/src/dist_spec.cpp), the pybind11
// glue (corehydropy/src/bindings/dist_spec.cpp), the C++ fixture runner (core/tests/
// test_fixtures.cpp) and the dotnet oracle emitter. Each serializes its native construct to the
// dist_spec.hpp grammar and calls run_dist/run_copula/run_mvdist, so a fixture case and a user's
// dist_pdf() call are the same code path.
//
// Stateless by construction: one call builds the object, evaluates once, and drops it. A seeded
// draw therefore returns the WHOLE vector in one call (method "random", args [n, seed]) so a
// rebuild can never split an RNG stream, which is exactly why the pre-phase-3 glue carried the
// bespoke *_seq entry points this replaces.
#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/distributions/support/dist_spec.hpp"

namespace corehydro::numerics::distributions::support {

// Flat result surface every binding and every fixture assertion reads. `values` holds whatever
// the method returns, in method order; `names` labels them when the method returns a named set
// (moments, tail dependence); `spec` carries a child object back when the method returns a
// distribution (MultivariateNormal marginal/conditional) and is empty otherwise.
struct DistResult {
    std::vector<double> values;
    std::vector<std::string> names;
    std::string spec;
};

namespace detail {

inline std::vector<double> arg_numbers(const JsonValue& args) {
    return args.as_double_vector();
}

inline double arg_at(const JsonValue& args, std::size_t i, const char* method) {
    const auto& v = args.items();
    if (i >= v.size())
        throw std::runtime_error(std::string("distribution method '") + method + "' needs " +
                                 std::to_string(i + 1) + " argument(s)");
    return v[i].as_double();
}

}  // namespace detail

// Evaluates `method` against the univariate distribution described by `spec_json`.
//
// Methods, with their args and the length of `values`:
//   pdf / log_pdf / cdf / quantile   args = the evaluation vector, values = same length
//   moments                          args = [],        values = 8, names set
//   mean median mode sd skewness kurtosis minimum maximum
//                                    args = [],        values = 1
//   random                           args = [n, seed], values = n  (seed <= 0 means clock)
//   log_likelihood                   args = the sample, values = 1
//   parameters                       args = [],        values = the flat parameter vector
//   parameters_valid                 args = [],        values = 1 (1.0 or 0.0)
//   linear_moments                   always throws: no composite implements it upstream
inline DistResult run_dist(const std::string& spec_json, const std::string& method,
                           const std::string& args_json) {
    JsonValue spec = models::spec::parse_json(spec_json);
    JsonValue args = models::spec::parse_json(args_json);
    std::unique_ptr<UnivariateDistributionBase> d = build_univariate(spec);
    DistResult r;

    if (method == "pdf" || method == "log_pdf" || method == "cdf" || method == "quantile") {
        for (double x : detail::arg_numbers(args)) {
            if (method == "pdf") r.values.push_back(d->pdf(x));
            else if (method == "log_pdf") r.values.push_back(d->log_pdf(x));
            else if (method == "cdf") r.values.push_back(d->cdf(x));
            else r.values.push_back(d->inverse_cdf(x));
        }
        return r;
    }
    if (method == "moments") {
        r.values = {d->mean(),     d->median(),   d->mode(),    d->standard_deviation(),
                    d->skewness(), d->kurtosis(), d->minimum(), d->maximum()};
        r.names = {"mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"};
        return r;
    }
    if (method == "mean") { r.values = {d->mean()}; return r; }
    if (method == "median") { r.values = {d->median()}; return r; }
    if (method == "mode") { r.values = {d->mode()}; return r; }
    if (method == "sd") { r.values = {d->standard_deviation()}; return r; }
    if (method == "skewness") { r.values = {d->skewness()}; return r; }
    if (method == "kurtosis") { r.values = {d->kurtosis()}; return r; }
    if (method == "minimum") { r.values = {d->minimum()}; return r; }
    if (method == "maximum") { r.values = {d->maximum()}; return r; }
    if (method == "random") {
        int n = static_cast<int>(detail::arg_at(args, 0, "random"));
        int seed = static_cast<int>(detail::arg_at(args, 1, "random"));
        r.values = d->generate_random_values(n, seed);
        return r;
    }
    if (method == "log_likelihood") {
        r.values = {d->log_likelihood(detail::arg_numbers(args))};
        return r;
    }
    if (method == "parameters") {
        r.values = d->get_parameters();
        return r;
    }
    if (method == "parameters_valid") {
        r.values = {d->parameters_valid() ? 1.0 : 0.0};
        return r;
    }
    if (method == "linear_moments")
        throw std::runtime_error("linear moments are not available for '" + spec_family(spec) +
                                 "'; no composite distribution implements ILinearMomentEstimation "
                                 "upstream");
    throw std::runtime_error("unknown distribution method: " + method);
}

}  // namespace corehydro::numerics::distributions::support
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cmake --build core/build -j && ./core/build/test_dist_runner
```

Expected: PASS on every check, exit status 0. Fix the two header-name uncertainties from Step 3 if the compile fails.

- [ ] **Step 6: Confirm nothing else regressed**

```bash
ctest --test-dir core/build
```

Expected: the Task 0 count plus one (the new `test_dist_runner`), 0 failures.

- [ ] **Step 7: Commit**

```bash
git add core/include/corehydro/numerics/distributions/support/dist_spec.hpp \
        core/include/corehydro/numerics/distributions/support/dist_runner.hpp \
        core/tests/test_dist_runner.cpp core/tests/test_helpers.hpp core/CMakeLists.txt \
        docs/superpowers/plans/2026-08-12-distribution-layer-completion.md
git commit -m "feat: the shared distribution spec grammar and runner

Promotes the fixture construct schema to a first-class nested grammar and adds
the one place a univariate distribution method is dispatched. Composites nest to
any depth; a seeded draw returns whole so a rebuild cannot split an RNG stream."
```

---

## Task 2: Copulas and multivariate distributions in the runner

**Files:**
- Modify: `core/include/corehydro/numerics/distributions/support/dist_spec.hpp` (add `build_copula`, `build_multivariate`)
- Modify: `core/include/corehydro/numerics/distributions/support/dist_runner.hpp` (add `run_copula`, `run_mvdist`)
- Modify: `core/tests/test_dist_runner.cpp`

**Interfaces:**
- Consumes: `build_univariate` and `DistResult` from Task 1; `copulas::create_copula`, `copulas::BivariateCopula`, `copulas::estimate`, `copulas::CopulaEstimationMethod`; `multivariate::MultivariateNormal`, `MultivariateStudentT`, `Dirichlet`, `Multinomial`, `BivariateEmpirical`.
- Produces:
  ```cpp
  std::unique_ptr<copulas::BivariateCopula> build_copula(const models::spec::JsonValue& spec);
  std::unique_ptr<multivariate::MultivariateDistribution>
      build_multivariate(const models::spec::JsonValue& spec);
  DistResult run_copula(const std::string& spec_json, const std::string& method,
                        const std::string& args_json);
  DistResult run_mvdist(const std::string& spec_json, const std::string& method,
                        const std::string& args_json);
  ```

**Grammar this task adds:**

```jsonc
{"family": "Clayton", "theta": 2.0, "df": 8,
 "margin_x": <univariate spec>, "margin_y": <univariate spec>}
{"family": "Clayton", "fit": {"x": [...], "y": [...], "method": "mpl",
                              "margin_x": <spec>, "margin_y": <spec>}}

{"family": "MultivariateNormal", "mean": [...], "covariance": [[...], [...]]}
{"family": "MultivariateStudentT", "df": 5, "location": [...], "scale": [[...], [...]]}
{"family": "Dirichlet", "alpha": [...]}
{"family": "Multinomial", "trials": 10, "probabilities": [...]}
{"family": "BivariateEmpirical", "x1": [...], "x2": [...], "p": [[...], [...]],
 "x1_transform": "None", "x2_transform": "None", "p_transform": "None"}
```

- [ ] **Step 1: Write the failing tests**

Append to `core/tests/test_dist_runner.cpp`, before `return chtest::summary(...)`:

```cpp
    // --- copulas ------------------------------------------------------------------------
    {
        supp::DistResult r =
            supp::run_copula(R"({"family":"Clayton","theta":2})", "pdf", "[0.3,0.7]");
        chtest::check_eq(static_cast<int>(r.values.size()), 1, "clayton/pdf/size");
        chtest::check_eq(std::isfinite(r.values.at(0)), true, "clayton/pdf/finite");
    }
    {
        supp::DistResult r =
            supp::run_copula(R"({"family":"Clayton","theta":2})", "tail_dependence", "[]");
        chtest::check_eq(static_cast<int>(r.values.size()), 2, "clayton/tail/size");
        chtest::check_eq(r.names.at(0) == "lower", true, "clayton/tail/names");
        // Closed form for Clayton: lambda_L = 2^(-1/theta).
        chtest::check_close(r.values.at(0), std::pow(2.0, -0.5), 1e-12, "clayton/tail/lower");
    }
    {
        supp::DistResult r =
            supp::run_copula(R"({"family":"Clayton","theta":2})", "bounds", "[]");
        chtest::check_eq(static_cast<int>(r.values.size()), 2, "clayton/bounds/size");
    }
    {
        // A copula with marginals attached samples pairs; 2n values, x-major then y.
        const char* spec = R"({"family":"Clayton","theta":2,
            "margin_x":{"family":"Normal","parameters":[0,1]},
            "margin_y":{"family":"Normal","parameters":[0,1]}})";
        supp::DistResult r = supp::run_copula(spec, "random", "[4,12345]");
        chtest::check_eq(static_cast<int>(r.values.size()), 8, "clayton/random/size");
    }
    {
        // The three log-likelihoods take x then y, split at the halfway point.
        const char* spec = R"({"family":"Clayton","theta":2,
            "margin_x":{"family":"Normal","parameters":[0,1]},
            "margin_y":{"family":"Normal","parameters":[0,1]}})";
        supp::DistResult r =
            supp::run_copula(spec, "log_likelihood_ifm", "[-1,0,1,-0.5,0.2,0.9]");
        chtest::check_eq(std::isfinite(r.values.at(0)), true, "clayton/ifm_ll/finite");
    }
    chtest::check_throws(
        [] { supp::run_copula(R"({"family":"Joe","fit":{"x":[1,2],"y":[1,2],"method":"tau"}})",
                              "theta", "[]"); },
        "tau");

    // --- multivariate -------------------------------------------------------------------
    {
        const char* spec = R"({"family":"MultivariateNormal","mean":[0,0],
                               "covariance":[[1,0],[0,1]]})";
        supp::DistResult r = supp::run_mvdist(spec, "pdf", "[0,0]");
        // Standard bivariate normal at the origin: 1 / (2 pi). std::acos(-1.0) rather than
        // M_PI, which is absent under strict -std=c++17 on Linux and on MSVC.
        chtest::check_close(r.values.at(0), 1.0 / (2.0 * std::acos(-1.0)), 1e-12,
                            "mvn/pdf/center");
    }
    {
        const char* spec = R"({"family":"MultivariateNormal","mean":[1,2,3],
                               "covariance":[[1,0,0],[0,1,0],[0,0,1]]})";
        // marginal returns a child spec, 0-based indices at this layer.
        supp::DistResult r = supp::run_mvdist(spec, "marginal", "[0,2]");
        chtest::check_eq(r.spec.empty(), false, "mvn/marginal/spec");
        supp::DistResult child = supp::run_mvdist(r.spec, "dimension", "[]");
        chtest::check_close(child.values.at(0), 2.0, 0.0, "mvn/marginal/dimension");
        supp::DistResult mean = supp::run_mvdist(r.spec, "mean", "[]");
        chtest::check_close(mean.values.at(1), 3.0, 1e-12, "mvn/marginal/mean");
    }
    {
        const char* spec = R"({"family":"MultivariateNormal","mean":[0,0],
                               "covariance":[[1,0],[0,1]]})";
        // interval takes lower then upper, split at the halfway point.
        supp::DistResult r = supp::run_mvdist(spec, "interval", "[-100,-100,100,100]");
        chtest::check_close(r.values.at(0), 1.0, 1e-6, "mvn/interval/whole_plane");
    }
    chtest::check_throws(
        [] {
            supp::run_mvdist(R"({"family":"MultivariateStudentT","df":5,"location":[0,0]})",
                             "marginal", "[0]");
        },
        "MultivariateStudentT");
    chtest::check_throws(
        [] { supp::run_mvdist(R"({"family":"Dirichlet","alpha":[2,3]})", "cdf", "[0.5,0.5]"); },
        "Dirichlet");
```

- [ ] **Step 2: Run to verify it fails**

```bash
cmake --build core/build -j 2>&1 | tail -20
```

Expected: FAIL, `'run_copula' is not a member of ...`.

- [ ] **Step 3: Add the two builders to `dist_spec.hpp`**

Add the includes (`copulas/base/copula_factory.hpp`, `copulas/base/bivariate_copula_estimation.hpp`, the five multivariate headers) and:

```cpp
// A copula spec is either parameterized ({"theta": x, "df"?: y}) or fitted
// ({"fit": {"x", "y", "method", "margin_x"?, "margin_y"?}}). Marginals attach in both forms.
inline std::unique_ptr<copulas::BivariateCopula> build_copula(const JsonValue& spec) {
    std::string family = spec_family(spec);
    std::unique_ptr<copulas::BivariateCopula> c;
    try {
        c = copulas::create_copula(family);
    } catch (const std::exception&) {
        throw std::runtime_error("unknown copula family '" + family +
                                 "'; expected AliMikhailHaq, Clayton, Frank, Gumbel, Joe, "
                                 "Normal or StudentT");
    }

    auto attach = [&](const JsonValue& holder) {
        if (holder.contains("margin_x"))
            c->marginal_distribution_x =
                std::shared_ptr<UnivariateDistributionBase>(build_univariate(holder.at("margin_x")));
        if (holder.contains("margin_y"))
            c->marginal_distribution_y =
                std::shared_ptr<UnivariateDistributionBase>(build_univariate(holder.at("margin_y")));
    };

    if (spec.contains("fit")) {
        const JsonValue& fit = spec.at("fit");
        attach(fit);
        std::vector<double> x = fit.at("x").as_double_vector();
        std::vector<double> y = fit.at("y").as_double_vector();
        std::string method = fit.value_or("method", "mpl");
        if (method == "tau") {
            // Only three concrete copulas implement SetThetaFromTau upstream.
            if (!copulas::set_theta_from_tau(*c, x, y))
                throw std::runtime_error("method 'tau' is not available for '" + family +
                                         "'; upstream implements SetThetaFromTau for Clayton, "
                                         "Gumbel and AliMikhailHaq only");
        } else if (method == "mpl") {
            copulas::estimate(*c, x, y, copulas::CopulaEstimationMethod::PseudoLikelihood);
        } else if (method == "ifm") {
            copulas::estimate(*c, x, y, copulas::CopulaEstimationMethod::InferenceFromMargins);
        } else if (method == "mle") {
            copulas::estimate(*c, x, y, copulas::CopulaEstimationMethod::FullLikelihood);
        } else {
            throw std::runtime_error("unknown copula fit method '" + method +
                                     "'; expected mpl, ifm, mle or tau");
        }
        return c;
    }

    std::vector<double> params = {spec.at("theta").as_double()};
    if (spec.contains("df")) params.push_back(spec.at("df").as_double());
    c->set_copula_parameters(params);
    attach(spec);
    return c;
}

inline std::unique_ptr<multivariate::MultivariateDistribution> build_multivariate(
    const JsonValue& spec) {
    std::string family = spec_family(spec);
    auto rows = [](const JsonValue& m) {
        std::vector<std::vector<double>> out;
        for (const JsonValue& row : m.items()) out.push_back(row.as_double_vector());
        return out;
    };
    if (family == "MultivariateNormal") {
        std::vector<double> mean = spec.at("mean").as_double_vector();
        if (!spec.contains("covariance"))
            return std::make_unique<multivariate::MultivariateNormal>(std::move(mean));
        return std::make_unique<multivariate::MultivariateNormal>(std::move(mean),
                                                                  rows(spec.at("covariance")));
    }
    if (family == "MultivariateStudentT") {
        double df = spec.at("df").as_double();
        std::vector<double> loc = spec.at("location").as_double_vector();
        if (!spec.contains("scale"))
            return std::make_unique<multivariate::MultivariateStudentT>(df, std::move(loc));
        return std::make_unique<multivariate::MultivariateStudentT>(df, std::move(loc),
                                                                    rows(spec.at("scale")));
    }
    if (family == "Dirichlet")
        return std::make_unique<multivariate::Dirichlet>(spec.at("alpha").as_double_vector());
    if (family == "Multinomial")
        return std::make_unique<multivariate::Multinomial>(
            spec.at("trials").as_int(), spec.at("probabilities").as_double_vector());
    if (family == "BivariateEmpirical") {
        auto tf = [&](const char* key) {
            return parse_bivariate_transform(spec.value_or(key, "None"));
        };
        return std::make_unique<multivariate::BivariateEmpirical>(
            spec.at("x1").as_double_vector(), spec.at("x2").as_double_vector(),
            rows(spec.at("p")), tf("x1_transform"), tf("x2_transform"), tf("p_transform"));
    }
    throw std::runtime_error("unknown multivariate family '" + family +
                             "'; expected MultivariateNormal, MultivariateStudentT, Dirichlet, "
                             "Multinomial or BivariateEmpirical");
}
```

Two helpers referenced above do not exist yet and must be written in this step, next to the builders, by lifting the equivalent logic out of `core/tests/test_fixtures.cpp`: `set_theta_from_tau(BivariateCopula&, x, y)` returning `false` for a family with no such method (the fixture runner's `set_theta_from_tau_dispatch` at `test_fixtures.cpp:1588-1605` is the exact branch set: Clayton, AliMikhailHaq, Gumbel), and `parse_bivariate_transform(const std::string&)` (the fixture runner's `parse_transform` lambda inside `build_multivariate`). Read both before writing, and keep the same accepted strings.

- [ ] **Step 4: Add the two dispatchers to `dist_runner.hpp`**

```cpp
// Evaluates `method` against the copula described by `spec_json`.
//
//   pdf / log_pdf / cdf              args = [u, v],       values = 1
//   inverse_cdf                      args = [u, v],       values = 2
//   tail_dependence                  args = [],           values = 2, names {lower, upper}
//   exceedance_or / exceedance_and   args = [u, v],       values = 1
//   theta / df                       args = [],           values = 1
//   bounds                           args = [],           values = 2, names {minimum, maximum}
//   parameters                       args = [],           values = the copula parameter vector
//   parameters_valid                 args = [],           values = 1
//   random                           args = [n, seed],    values = 2n (all x, then all y)
//   log_likelihood_pseudo / _ifm / _full
//                                    args = x then y,     values = 1
//   marginal_x_parameters / marginal_y_parameters
//                                    args = [],           values = that marginal's parameters
inline DistResult run_copula(const std::string& spec_json, const std::string& method,
                             const std::string& args_json) {
    JsonValue spec = models::spec::parse_json(spec_json);
    JsonValue args = models::spec::parse_json(args_json);
    std::unique_ptr<copulas::BivariateCopula> c = build_copula(spec);
    DistResult r;

    auto split_xy = [&]() {
        std::vector<double> all = detail::arg_numbers(args);
        if (all.size() % 2 != 0)
            throw std::runtime_error("copula method '" + method +
                                     "' needs an even number of arguments (x then y)");
        std::size_t h = all.size() / 2;
        return std::make_pair(std::vector<double>(all.begin(), all.begin() + h),
                              std::vector<double>(all.begin() + h, all.end()));
    };

    if (method == "pdf") { r.values = {c->pdf(detail::arg_at(args, 0, "pdf"),
                                              detail::arg_at(args, 1, "pdf"))}; return r; }
    if (method == "log_pdf") { r.values = {c->log_pdf(detail::arg_at(args, 0, "log_pdf"),
                                                      detail::arg_at(args, 1, "log_pdf"))}; return r; }
    if (method == "cdf") { r.values = {c->cdf(detail::arg_at(args, 0, "cdf"),
                                              detail::arg_at(args, 1, "cdf"))}; return r; }
    if (method == "inverse_cdf") {
        std::array<double, 2> uv = c->inverse_cdf(detail::arg_at(args, 0, "inverse_cdf"),
                                                  detail::arg_at(args, 1, "inverse_cdf"));
        r.values = {uv[0], uv[1]};
        return r;
    }
    if (method == "tail_dependence") {
        r.values = {c->lower_tail_dependence(), c->upper_tail_dependence()};
        r.names = {"lower", "upper"};
        return r;
    }
    if (method == "exceedance_or") {
        r.values = {c->or_joint_exceedance_probability(detail::arg_at(args, 0, "exceedance_or"),
                                                       detail::arg_at(args, 1, "exceedance_or"))};
        return r;
    }
    if (method == "exceedance_and") {
        r.values = {c->and_joint_exceedance_probability(detail::arg_at(args, 0, "exceedance_and"),
                                                        detail::arg_at(args, 1, "exceedance_and"))};
        return r;
    }
    if (method == "theta") { r.values = {c->theta()}; return r; }
    if (method == "df") {
        std::vector<double> p = c->get_copula_parameters();
        if (p.size() < 2)
            throw std::runtime_error("copula '" + spec_family(spec) +
                                     "' has no degrees-of-freedom parameter");
        r.values = {p[1]};
        return r;
    }
    if (method == "bounds") {
        r.values = {c->theta_minimum(), c->theta_maximum()};
        r.names = {"minimum", "maximum"};
        return r;
    }
    if (method == "parameters") { r.values = c->get_copula_parameters(); return r; }
    if (method == "parameters_valid") { r.values = {c->parameters_valid() ? 1.0 : 0.0}; return r; }
    if (method == "random") {
        int n = static_cast<int>(detail::arg_at(args, 0, "random"));
        int seed = static_cast<int>(detail::arg_at(args, 1, "random"));
        auto m = c->generate_random_values(n, seed);
        for (const auto& row : m) r.values.push_back(row[0]);
        for (const auto& row : m) r.values.push_back(row[1]);
        return r;
    }
    if (method == "log_likelihood_pseudo" || method == "log_likelihood_ifm" ||
        method == "log_likelihood_full") {
        auto xy = split_xy();
        if (method == "log_likelihood_pseudo")
            r.values = {c->pseudo_log_likelihood(xy.first, xy.second)};
        else if (method == "log_likelihood_ifm")
            r.values = {c->ifm_log_likelihood(xy.first, xy.second)};
        else
            r.values = {c->log_likelihood(xy.first, xy.second)};
        return r;
    }
    if (method == "marginal_x_parameters" || method == "marginal_y_parameters") {
        const auto& m = method == "marginal_x_parameters" ? c->marginal_distribution_x
                                                          : c->marginal_distribution_y;
        if (!m) throw std::runtime_error("copula method '" + method + "': no marginal is attached");
        r.values = m->get_parameters();
        return r;
    }
    throw std::runtime_error("unknown copula method: " + method);
}
```

The `generate_random_values` return type is `math::linalg::Matrix2D` (rows of two); adapt the loop above to whatever indexing that type exposes after reading `bivariate_copula.hpp:132`.

Then `run_mvdist`:

```cpp
// Evaluates `method` against the multivariate distribution described by `spec_json`.
//
//   pdf / log_pdf / cdf     args = the point,            values = 1
//   mahalanobis             args = the point,            values = 1
//   dimension               args = [],                   values = 1
//   mean / variance / sd    args = [],                   values = dimension
//   covariance              args = [],                   values = dimension^2, row-major
//   random                  args = [n, seed],            values = n * dimension, row-major
//   random_lhs              args = [n, seed],            values = n * dimension, row-major
//   interval                args = lower then upper,     values = 1        (MultivariateNormal)
//   marginal                args = 0-based indices,      spec = the child (MultivariateNormal)
//   conditional             args = indices then values,  spec = the child (MultivariateNormal)
//   parameters_valid        args = [],                   values = 1
//
// `cdf` on Dirichlet or Multinomial and `pdf` on BivariateEmpirical are upstream stubs; both
// surface here as a throw naming the family, not a NaN.
inline DistResult run_mvdist(const std::string& spec_json, const std::string& method,
                             const std::string& args_json) {
    JsonValue spec = models::spec::parse_json(spec_json);
    JsonValue args = models::spec::parse_json(args_json);
    std::string family = spec_family(spec);
    std::unique_ptr<multivariate::MultivariateDistribution> d = build_multivariate(spec);
    DistResult r;

    auto* mvn = dynamic_cast<multivariate::MultivariateNormal*>(d.get());

    if (method == "cdf" && (family == "Dirichlet" || family == "Multinomial"))
        throw std::runtime_error("cdf is not implemented for '" + family + "' upstream");
    if (method == "pdf" && family == "BivariateEmpirical")
        throw std::runtime_error("pdf is not implemented for 'BivariateEmpirical' upstream "
                                 "(it returns NaN)");
    if ((method == "marginal" || method == "conditional" || method == "interval") && !mvn)
        throw std::runtime_error("'" + method + "' is available for MultivariateNormal only; '" +
                                 family + "' has no such member upstream");

    if (method == "pdf") { r.values = {d->pdf(detail::arg_numbers(args))}; return r; }
    if (method == "log_pdf") { r.values = {d->log_pdf(detail::arg_numbers(args))}; return r; }
    if (method == "cdf") { r.values = {d->cdf(detail::arg_numbers(args))}; return r; }
    if (method == "dimension") { r.values = {static_cast<double>(d->dimension())}; return r; }
    if (method == "parameters_valid") { r.values = {d->parameters_valid() ? 1.0 : 0.0}; return r; }
    if (method == "marginal") {
        std::vector<int> idx;
        for (double v : detail::arg_numbers(args)) idx.push_back(static_cast<int>(v));
        multivariate::MultivariateNormal child = mvn->marginal(idx);
        r.spec = mvn_spec_string(child);
        return r;
    }
    if (method == "conditional") {
        std::vector<double> all = detail::arg_numbers(args);
        if (all.size() % 2 != 0)
            throw std::runtime_error("'conditional' needs indices then the same number of values");
        std::size_t h = all.size() / 2;
        std::vector<int> idx;
        for (std::size_t i = 0; i < h; ++i) idx.push_back(static_cast<int>(all[i]));
        std::vector<double> vals(all.begin() + h, all.end());
        multivariate::MultivariateNormal child = mvn->conditional(idx, vals);
        r.spec = mvn_spec_string(child);
        return r;
    }
    if (method == "interval") {
        std::vector<double> all = detail::arg_numbers(args);
        if (all.size() % 2 != 0)
            throw std::runtime_error("'interval' needs a lower vector then an upper vector");
        std::size_t h = all.size() / 2;
        r.values = {mvn->interval(std::vector<double>(all.begin(), all.begin() + h),
                                  std::vector<double>(all.begin() + h, all.end()))};
        return r;
    }
    // The remaining methods have no shared base signature: MultivariateNormal and
    // MultivariateStudentT carry the accessors, Dirichlet and Multinomial carry their own, and
    // BivariateEmpirical carries none. One branch per concrete type, dynamic_cast up front.
    auto* mvt = dynamic_cast<multivariate::MultivariateStudentT*>(d.get());
    auto* dir = dynamic_cast<multivariate::Dirichlet*>(d.get());
    auto* mn = dynamic_cast<multivariate::Multinomial*>(d.get());

    auto flatten = [&r](const std::vector<std::vector<double>>& m) {
        for (const auto& row : m)
            for (double v : row) r.values.push_back(v);
    };
    auto no_such = [&](const char* what) -> DistResult {
        throw std::runtime_error(std::string("'") + what + "' is not available for '" + family +
                                 "' upstream");
    };

    if (method == "mahalanobis") {
        if (mvn) r.values = {mvn->mahalanobis(detail::arg_numbers(args))};
        else if (mvt) r.values = {mvt->mahalanobis(detail::arg_numbers(args))};
        else return no_such("mahalanobis");
        return r;
    }
    if (method == "mean") {
        if (mvn) r.values = mvn->mean();
        else if (mvt) r.values = mvt->mean();
        else if (dir) r.values = dir->mean();
        else if (mn) r.values = mn->mean();
        else return no_such("mean");
        return r;
    }
    if (method == "variance") {
        if (mvn) r.values = mvn->variance();
        else if (mvt) r.values = mvt->variance();
        else if (dir) r.values = dir->variance();
        else if (mn) r.values = mn->variance();
        else return no_such("variance");
        return r;
    }
    if (method == "sd") {
        if (mvn) r.values = mvn->standard_deviation();
        else if (mvt) r.values = mvt->standard_deviation();
        else return no_such("sd");
        return r;
    }
    if (method == "covariance") {
        int n = d->dimension();
        if (mvn || mvt || dir || mn) {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    r.values.push_back(mvn   ? mvn->covariance(i, j)
                                       : mvt ? mvt->covariance(i, j)
                                       : dir ? dir->covariance(i, j)
                                             : mn->covariance(i, j));
            return r;
        }
        return no_such("covariance");
    }
    if (method == "random" || method == "random_lhs") {
        int n = static_cast<int>(detail::arg_at(args, 0, method.c_str()));
        int seed = static_cast<int>(detail::arg_at(args, 1, method.c_str()));
        if (method == "random_lhs") {
            if (mvn) flatten(mvn->latin_hypercube_random_values(n, seed));
            else if (mvt) flatten(mvt->latin_hypercube_random_values(n, seed));
            else return no_such("random_lhs");
        } else {
            if (mvn) flatten(mvn->generate_random_values(n, seed));
            else if (mvt) flatten(mvt->generate_random_values(n, seed));
            else if (dir) flatten(dir->generate_random_values(n, seed));
            else if (mn) flatten(mn->generate_random_values(n, seed));
            else return no_such("random");
        }
        return r;
    }
    throw std::runtime_error("unknown multivariate method: " + method);
}
```

Two return types above need confirming against the headers before compiling, and correcting in place if they differ: `generate_random_values` and `latin_hypercube_random_values` return a matrix type whose row indexing `flatten` assumes (`multivariate_normal.hpp:503`, `:519`), and `Dirichlet::variance()` / `Multinomial::variance()` return a vector (`dirichlet.hpp:90`, `multinomial.hpp:83`). Read them; do not guess.

`mvn_spec_string` is a small local helper you write in this step: it serializes a `MultivariateNormal` back into the grammar so `r.spec` round-trips.

```cpp
inline std::string mvn_spec_string(const multivariate::MultivariateNormal& m) {
    std::string out = R"({"family":"MultivariateNormal","mean":[)";
    const std::vector<double>& mu = m.mean();
    for (std::size_t i = 0; i < mu.size(); ++i) {
        if (i) out += ",";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", mu[i]);
        out += buf;
    }
    out += R"(],"covariance":[)";
    for (int i = 0; i < m.dimension(); ++i) {
        if (i) out += ",";
        out += "[";
        for (int j = 0; j < m.dimension(); ++j) {
            if (j) out += ",";
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", m.covariance(i, j));
            out += buf;
        }
        out += "]";
    }
    out += "]}";
    return out;
}
```

- [ ] **Step 5: Point `model_spec.hpp` at the shared builder**

`build_spec_distribution` at `core/include/corehydro/models/model_spec.hpp:125` builds a flat `{family, parameters}` distribution only, so a model prior cannot be a truncated or mixture distribution. Delegate it:

```cpp
// A `{ "family": ..., "parameters": [...] }` distribution spec -> a parameterized distribution.
// Since phase 3 this is the shared nested builder, so a prior may itself be a composite (a
// truncated Normal, a mixture) with no change at any call site.
inline std::unique_ptr<numerics::distributions::UnivariateDistributionBase>
build_spec_distribution(const JsonValue& spec) {
    return numerics::distributions::support::build_univariate(spec);
}
```

Include `dist_spec.hpp` at the top of `model_spec.hpp`. Check for a circular include first: `dist_spec.hpp` includes `models/json_lite.hpp`, not `model_spec.hpp`, so there is none, but confirm before building.

Add the covering test to `core/tests/test_dist_runner.cpp`:

```cpp
    // A model prior may now be a composite, which no spec could express before.
    {
        const char* prior = R"({"family":"TruncatedDistribution",
            "base":{"family":"Normal","parameters":[0,1]},"bounds":[-1,1]})";
        auto d = corehydro::models::spec::build_spec_distribution(
            corehydro::models::spec::parse_json(prior));
        chtest::check_close(d->cdf(1.0), 1.0, 1e-9, "model_spec/composite_prior");
    }
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
cmake --build core/build -j && ./core/build/test_dist_runner
```

Expected: PASS on every check.

- [ ] **Step 7: Confirm nothing else regressed**

```bash
ctest --test-dir core/build
```

Expected: the Task 1 count, 0 failures. The model, estimation, and analysis suites all run through `build_spec_distribution`, so a green ctest here is what proves the delegation is behaviour-preserving.

- [ ] **Step 8: Commit**

```bash
git add core/include/corehydro/numerics/distributions/support/ \
        core/include/corehydro/models/model_spec.hpp core/tests/test_dist_runner.cpp
git commit -m "feat: copulas and multivariate distributions in the shared runner

Adds build_copula/build_multivariate and run_copula/run_mvdist. marginal and
conditional hand a child spec back rather than a C++ object, so they compose
without anything crossing the language boundary. model_spec.hpp's
build_spec_distribution now delegates to the same builder, so a model prior may
be a composite."
```

---

## Task 3: The C++ fixture runner delegates to the runner

**Files:**
- Modify: `core/tests/test_fixtures.cpp` (`build_composite` at :998-1092 and its call site in `run_generic` at :1174-1194; `build_copula` at :1608 and `run_bivariate_copula` at :1701; `build_multivariate` at :1331 and `dispatch_multivariate` at :1426)

**Interfaces:**
- Consumes: `run_dist`, `run_copula`, `run_mvdist` from Tasks 1 and 2.
- Produces: nothing new. This task is a pure refactor whose gate is that no oracle value moves.

The fixture runner uses nlohmann/json while the runner uses `json_lite`. Bridge by serializing: resolve any dataset reference into an inline array, `dump()` the construct, and call the runner with the method and an args array.

- [ ] **Step 1: Capture the pre-change baseline**

```bash
./core/build/test_fixtures fixtures 2>&1 | tail -3
```

Expected: the check count and `0 failed`. Write the check count down; it must be identical after this task.

- [ ] **Step 2: Add the bridge helper**

In `core/tests/test_fixtures.cpp`, above `build_composite`:

```cpp
// --- phase 3: delegation to the shared runner -------------------------------------------
//
// The fixture construct schema IS the dist_spec.hpp grammar (that is why it was promoted), so
// the only work here is resolving a dataset NAME into an inline array and handing the object to
// the runner. Everything the runner knows how to build and dispatch is therefore proven by the
// same pinned corpus the bespoke glue was proven by.
static json inline_datasets(const json& construct, const json& datasets) {
    json out = construct;
    // "data" is the only dataset-by-name key in the univariate grammar (KernelDensity).
    if (out.contains("data") && out["data"].is_string())
        out["data"] = datasets[out["data"].get<std::string>()];
    if (out.contains("base")) out["base"] = inline_datasets(out["base"], datasets);
    if (out.contains("components")) {
        for (auto& c : out["components"]) c = inline_datasets(c, datasets);
    }
    return out;
}

static json args_array(const json& args) { return args.is_array() ? args : json::array(); }
```

- [ ] **Step 3: Route the composite path through the runner**

Replace the `composite ? build_composite(...) : build_generic(...)` branch in `run_generic` so that composite targets no longer build a C++ object locally. The case loop becomes:

```cpp
static void run_generic(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    bool composite = is_composite_target(target);
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        if (composite) {
            json cspec = inline_datasets(c["construct"], datasets);
            cspec["family"] = target;
            for (const auto& as : c["assertions"]) {
                std::string method = as["method"].get<std::string>();
                std::string where = target + "/" + name + "/" + method;
                if (method == "set_parameters") {
                    cspec["set_parameters"] = as["args"];
                    continue;
                }
                auto r = supp::run_dist(cspec.dump(), fixture_method(method),
                                        args_array(as.value("args", json::array())).dump());
                if (as["mode"].get<std::string>() == "bool")
                    check_bool(r.values.at(0) != 0.0, as, where);
                else
                    check_value(fixture_pick(r, method, as), as, where);
            }
            continue;
        }
        // ... the existing non-composite branch, unchanged ...
    }
}
```

Two small adapters make the fixture method vocabulary line up with the runner's, and belong beside the bridge helper:

```cpp
// The fixture vocabulary predates the runner's. Map the differences in one place.
static std::string fixture_method(const std::string& m) {
    if (m == "param") return "parameters";
    if (m == "random_value") return "random";
    return m;  // pdf, log_pdf, cdf, quantile, mean, ..., parameters_valid pass straight through
}

// `param` and `random_value` index into a vector the runner returns whole.
static double fixture_pick(const supp::DistResult& r, const std::string& method, const json& as) {
    if (method == "param") return r.values.at(as["args"][0].get<std::size_t>());
    if (method == "random_value") return r.values.at(as["args"][1].get<std::size_t>());
    return r.values.at(0);
}
```

Read the actual `random_value` arg convention in a fixture that uses it (for example `fixtures/distributions/multivariate/dirichlet.json`) before writing `fixture_pick`, and match it exactly. If the convention is `[n, seed, index]` rather than `[n, index]`, adapt both this helper and the `random` args it forwards.

- [ ] **Step 4: Route the copula and multivariate paths through the runner**

Apply the same treatment to `run_bivariate_copula` and the multivariate case loop: build the spec object from the fixture construct (adding `"family": target`), call `run_copula` / `run_mvdist`, and map the fixture method names to the runner's:

```cpp
static std::string fixture_copula_method(const std::string& m) {
    if (m == "upper_tail_dependence" || m == "lower_tail_dependence") return "tail_dependence";
    if (m == "or_exceedance") return "exceedance_or";
    if (m == "and_exceedance") return "exceedance_and";
    if (m == "random_value") return "random";
    if (m == "marginal_param") return "marginal_x_parameters";  // switched on args[0] below
    return m;
}
```

For `marginal_param` the fixture passes `("x" | "y", index)`; pick `marginal_x_parameters` or `marginal_y_parameters` on that first argument, then index `values` by the second. For the multivariate `marginal_*` / `conditional_*` methods, call `marginal` / `conditional` to get the child spec and then evaluate `mean` / `covariance` / `dimension` / `log_pdf` against that child, which is exactly what the bespoke glue did internally.

Delete `build_composite`, `build_copula`, `build_multivariate`, `dispatch_composite` (if present), `dispatch_copula`, `dispatch_multivariate`, and `build_component` once nothing references them. Leave `build_generic` and `dispatch_generic` alone: the 38 flat families still use them.

- [ ] **Step 5: Run the fixture suite and compare the check count**

```bash
cmake --build core/build -j && ./core/build/test_fixtures fixtures 2>&1 | tail -3
```

Expected: the exact check count from Step 1, `0 failed`. A moved value means the delegation is wrong, not that the fixture is wrong. Do not touch a fixture in this task.

- [ ] **Step 6: Run the whole C++ suite**

```bash
ctest --test-dir core/build
```

Expected: the Task 2 count, 0 failures.

- [ ] **Step 7: Commit**

```bash
git add core/tests/test_fixtures.cpp
git commit -m "refactor: the C++ fixture runner drives the shared distribution runner

Composite, copula and multivariate cases now build and dispatch through
dist_runner.hpp instead of the three bespoke builders, which are deleted. No
oracle value moves: the check count is identical before and after."
```

---

## Task 4: The R glue and the composite constructors

**Files:**
- Create: `corehydror/src/dist_spec.cpp`
- Modify: `corehydror/R/distribution.R`
- Modify: `corehydror/tests/testthat/test-distribution.R`
- Modify: `corehydror/NAMESPACE`, `corehydror/R/cpp11.R` (regenerated, not hand-edited)

**Interfaces:**
- Consumes: `run_dist`, `run_copula`, `run_mvdist` and `DistResult` from Tasks 1 and 2.
- Produces, for R:
  ```r
  # internal
  ch_dist_spec_run_(spec_json, method, args_json) -> list(values=, names=, spec=)
  ch_copula_run_(spec_json, method, args_json)    -> the same shape
  ch_mvdist_run_(spec_json, method, args_json)    -> the same shape
  # exported
  dist_truncated(d, min, max)
  dist_mixture(components, weights, zero_inflated = FALSE, zero_weight = 0)
  dist_competing_risks(components, minimum_of = TRUE, dependency = "Independent",
                       correlation = NULL)
  dist_empirical(x, p, p_transform = c("NormalZ", "None"), p_descending = FALSE)
  dist_kde(data, kernel = c("Gaussian", "Epanechnikov", "Triangular", "Uniform"),
           bandwidth = NULL, bounded_by_data = TRUE)
  distribution_names(kind = c("flat", "structured", "all"))
  ```
  A `corehydro_dist` for a composite is `list(family = <family>, params = NULL, spec = <list>)`
  with class `"corehydro_dist"`; the flat form keeps `list(family =, params =)` unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `corehydror/tests/testthat/test-distribution.R`:

```r
# --- composite families -------------------------------------------------------------------
# Oracle VALUES live in fixtures/ and are asserted by test-fixtures.R; these assert object
# shape, argument handling, error messages, and that every existing verb accepts a composite.

test_that("dist_truncated returns a corehydro_dist every verb accepts", {
  d <- dist_truncated(distribution("Normal", c(2, 1)), min = 1.1, max = 2.11)
  expect_s3_class(d, "corehydro_dist")
  expect_equal(d$family, "TruncatedDistribution")
  expect_true(is.finite(dist_pdf(d, 1.5)))
  expect_true(is.finite(dist_cdf(d, 1.5)))
  expect_equal(dist_quantile(d, 0.5), dist_quantile(d, 0.5))
  expect_length(dist_moments(d), 8L)
  expect_named(dist_moments(d))
})

test_that("the pointwise verbs vectorize over a composite", {
  d <- dist_truncated(distribution("Normal", c(2, 1)), 1.1, 2.11)
  expect_length(dist_pdf(d, c(1.2, 1.5, 2.0)), 3L)
  expect_length(dist_cdf(d, c(1.2, 1.5, 2.0)), 3L)
})

test_that("dist_mixture nests inside dist_truncated", {
  mix <- dist_mixture(
    list(distribution("Normal", c(0, 1)), distribution("Normal", c(5, 1))),
    weights = c(0.5, 0.5)
  )
  d <- dist_truncated(mix, min = -2, max = 7)
  expect_equal(dist_cdf(d, 7), 1, tolerance = 1e-9)
})

test_that("dist_mixture validates its arguments", {
  d1 <- distribution("Normal", c(0, 1))
  expect_error(dist_mixture(list(d1), weights = c(0.5, 0.5)), "weights")
  expect_error(dist_mixture(list(d1, "not a distribution"), c(0.5, 0.5)), "corehydro_dist")
})

test_that("dist_kde takes data inline and defaults to the Silverman bandwidth", {
  d <- dist_kde(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
  expect_s3_class(d, "corehydro_dist")
  expect_equal(dist_cdf(d, 5.5), 0.5, tolerance = 0.05)
  expect_error(dist_kde(1:10, kernel = "Cosine"), "kernel")
})

test_that("dist_empirical takes x and p", {
  d <- dist_empirical(x = c(1, 2, 3), p = c(0.1, 0.5, 0.9))
  expect_equal(dist_quantile(d, 0.5), 2, tolerance = 1e-9)
})

test_that("a seeded composite draw is reproducible", {
  d <- dist_mixture(list(distribution("Normal", c(0, 1)), distribution("Normal", c(5, 1))),
                    c(0.5, 0.5))
  expect_identical(dist_random(d, 5, seed = 12345), dist_random(d, 5, seed = 12345))
  expect_length(dist_random(d, 5, seed = 12345), 5L)
})

test_that("a composite round-trips through save and load", {
  d <- dist_truncated(distribution("Normal", c(2, 1)), 1.1, 2.11)
  f <- tempfile()
  saveRDS(d, f)
  expect_equal(dist_pdf(readRDS(f), 1.5), dist_pdf(d, 1.5))
  unlink(f)
})

test_that("linear moments are refused for composites with the reason", {
  d <- dist_kde(1:10)
  expect_error(dist_lmoments(d), "linear moment")
})

test_that("distribution() points structured families at their constructor", {
  expect_error(distribution("Empirical", c(1, 2)), "dist_empirical")
  expect_error(distribution("KernelDensity", 1), "dist_kde")
})

test_that("distribution_names reports both kinds", {
  expect_true("Normal" %in% distribution_names())
  expect_setequal(
    distribution_names("structured"),
    c("TruncatedDistribution", "Mixture", "CompetingRisks", "Empirical", "KernelDensity")
  )
  expect_true(all(distribution_names() %in% distribution_names("all")))
})
```

- [ ] **Step 2: Run to verify they fail**

```bash
Rscript -e 'testthat::test_local("corehydror", filter = "distribution")'
```

Expected: FAIL, `could not find function "dist_truncated"`.

- [ ] **Step 3: Write the cpp11 glue**

Create `corehydror/src/dist_spec.cpp`:

```cpp
// cpp11 glue for the shared distribution-spec runner: the three entry points behind every
// composite, copula and multivariate verb in R/distribution.R, R/copula.R and R/mvdist.R. Each
// takes a spec in the dist_spec.hpp grammar (assembled R-side by to_spec_json()) plus a method
// name and a JSON args array, and returns the flat DistResult as a named list.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/support/dist_runner.hpp"
#include "cpp11.hpp"

using namespace cpp11;
namespace supp = corehydro::numerics::distributions::support;

static list pack(const supp::DistResult& r) {
    writable::doubles values(static_cast<R_xlen_t>(r.values.size()));
    for (std::size_t i = 0; i < r.values.size(); ++i)
        values[static_cast<R_xlen_t>(i)] = r.values[i];
    writable::strings names(static_cast<R_xlen_t>(r.names.size()));
    for (std::size_t i = 0; i < r.names.size(); ++i)
        names[static_cast<R_xlen_t>(i)] = r.names[i];
    return writable::list({"values"_nm = values, "names"_nm = names,
                           "spec"_nm = writable::strings({r.spec})});
}

[[cpp11::register]]
list ch_dist_spec_run_(std::string spec_json, std::string method, std::string args_json) {
    return pack(supp::run_dist(spec_json, method, args_json));
}

[[cpp11::register]]
list ch_copula_run_(std::string spec_json, std::string method, std::string args_json) {
    return pack(supp::run_copula(spec_json, method, args_json));
}

[[cpp11::register]]
list ch_mvdist_run_(std::string spec_json, std::string method, std::string args_json) {
    return pack(supp::run_mvdist(spec_json, method, args_json));
}
```

- [ ] **Step 4: Write the R surface**

The serializer already exists and must be extended, not duplicated: `corehydror/R/spec.R:41` defines `to_spec_json()`, which recurses through lists and already has a `corehydro_dist` branch emitting `{"family":..., "parameters":[...]}`. Read that file first. Two changes there:

```r
  # in to_spec_json(), replacing the existing corehydro_dist branch at spec.R:45-47:
  if (inherits(x, "corehydro_dist")) {
    # A composite carries its own nested spec; a flat family serializes from family + params.
    if (!is.null(x$spec)) return(x$spec)
    return(to_spec_json(list(family = x$family, parameters = spec_array(x$params))))
  }
```

Because `to_spec_json()` recurses, a composite constructor builds its spec by handing it a list with the component `corehydro_dist` objects still in place, and nesting takes care of itself. In `corehydror/R/distribution.R`:

```r
# Internal: run a method against a corehydro_dist and return the numeric result. `args` is
# always serialized as a JSON array, even when it has length one.
dist_run <- function(d, method, args = numeric(0)) {
  res <- ch_dist_spec_run_(to_spec_json(d), method, to_spec_json(spec_array(as.double(args))))
  out <- res$values
  if (length(res$names)) names(out) <- res$names
  out
}

#' Truncate a distribution
#'
#' @param d a `corehydro_dist`.
#' @param min,max the truncation bounds.
#' @return a `corehydro_dist` of family `"TruncatedDistribution"`.
#' @examples
#' dist_pdf(dist_truncated(distribution("Normal", c(2, 1)), 1.1, 2.11), 1.5)
#' @export
dist_truncated <- function(d, min, max) {
  if (!inherits(d, "corehydro_dist")) stop("`d` must be a corehydro_dist", call. = FALSE)
  if (!is.numeric(min) || length(min) != 1L || !is.numeric(max) || length(max) != 1L)
    stop("`min` and `max` must each be a single number", call. = FALSE)
  if (min >= max) stop("`min` must be less than `max`", call. = FALSE)
  new_composite_dist("TruncatedDistribution", to_spec_json(list(
    family = "TruncatedDistribution",
    base = d,
    bounds = spec_array(c(as.double(min), as.double(max)))
  )))
}
```

Write `dist_mixture`, `dist_competing_risks`, `dist_empirical`, and `dist_kde` in the same shape, each validating its arguments before serializing:

- `dist_mixture(components, weights, zero_inflated = FALSE, zero_weight = 0)`: every element of `components` must be a `corehydro_dist`; `length(weights)` must equal `length(components)`; emits `components`, `weights`, `zero_inflated`, `zero_weight`.
- `dist_competing_risks(components, minimum_of = TRUE, dependency = "Independent", correlation = NULL)`: `dependency` is matched against `c("Independent", "PerfectlyPositive", "PerfectlyNegative", "CorrelationMatrix")` (confirm the exact set against `competing_risks.hpp` before writing); `correlation` is a square numeric matrix emitted as an array of arrays.
- `dist_empirical(x, p, p_transform = c("NormalZ", "None"), p_descending = FALSE)`: `length(x)` must equal `length(p)`; `p_transform` through `match.arg()`.
- `dist_kde(data, kernel = c("Gaussian", "Epanechnikov", "Triangular", "Uniform"), bandwidth = NULL, bounded_by_data = TRUE)`: `kernel` through `match.arg()`; `bandwidth = NULL` omits the key so the C++ Silverman default applies.

The shared constructor:

```r
# Internal: a composite carries its serialized spec instead of a flat parameter vector, so
# `params` is NULL and every verb routes through dist_run().
new_composite_dist <- function(family, spec_json) {
  structure(list(family = family, params = NULL, spec = spec_json),
            class = "corehydro_dist")
}
```

Re-route the existing verbs so a composite goes through `dist_run()` while a flat family keeps its current fast path. For example:

```r
dist_pdf <- function(d, x) {
  if (is.null(d$spec)) return(ch_dist_pdf_v_(d$family, d$params, as.double(x)))
  dist_run(d, "pdf", as.double(x))
}
```

Apply the same two-line pattern to `dist_log_pdf`, `dist_cdf`, `dist_quantile`, `dist_random`, `dist_moments`, and `dist_log_likelihood`. `dist_lmoments` gets the refusal:

```r
dist_lmoments <- function(d) {
  if (!is.null(d$spec))
    stop(sprintf(
      "linear moments are not available for '%s'; no composite distribution implements %s",
      d$family, "ILinearMomentEstimation upstream"), call. = FALSE)
  ch_dist_linear_moments_(d$family, d$params)
}
```

`dist_params` on a composite returns the spec payload rather than an empty vector; `print.corehydro_dist` prints the family plus a one-line summary of the payload for composites.

In `distribution()`, add the guard before the existing name check:

```r
  structured <- c("TruncatedDistribution", "Mixture", "CompetingRisks", "Empirical",
                  "KernelDensity")
  ctor <- c(TruncatedDistribution = "dist_truncated()", Mixture = "dist_mixture()",
            CompetingRisks = "dist_competing_risks()", Empirical = "dist_empirical()",
            KernelDensity = "dist_kde()")
  if (family %in% structured) {
    stop(sprintf("'%s' has no flat parameter vector; use %s instead", family, ctor[[family]]),
         call. = FALSE)
  }
```

and widen `distribution_names()`:

```r
distribution_names <- function(kind = c("flat", "structured", "all")) {
  kind <- match.arg(kind)
  flat <- ch_dist_names_()
  structured <- c("TruncatedDistribution", "Mixture", "CompetingRisks", "Empirical",
                  "KernelDensity")
  switch(kind, flat = setdiff(flat, structured), structured = structured,
         all = union(setdiff(flat, structured), structured))
}
```

- [ ] **Step 5: Regenerate, install, and run the tests**

```bash
Rscript -e 'cpp11::cpp_register("corehydror")'
Rscript -e 'roxygen2::roxygenise("corehydror")'
R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror", filter = "distribution")'
```

Expected: PASS, 0 failures.

- [ ] **Step 6: Confirm the rest of the R suite still passes**

```bash
Rscript -e 'testthat::test_local("corehydror")'
```

Expected: the Task 0 count plus the new tests, 0 failures.

- [ ] **Step 7: Commit**

```bash
git add corehydror/src/dist_spec.cpp corehydror/src/cpp11.cpp corehydror/R/cpp11.R \
        corehydror/R/distribution.R corehydror/NAMESPACE corehydror/man \
        corehydror/tests/testthat/test-distribution.R
git commit -m "feat: composite distributions in R

dist_truncated/dist_mixture/dist_competing_risks/dist_empirical/dist_kde return
the same corehydro_dist every existing verb accepts, over the shared runner.
distribution() now points the five structured families at their constructor
instead of failing later inside every verb."
```

---

## Task 5: The R copula and multivariate surface

**Files:**
- Create: `corehydror/R/copula.R`
- Create: `corehydror/R/mvdist.R`
- Create: `corehydror/tests/testthat/test-copula.R`
- Create: `corehydror/tests/testthat/test-mvdist.R`

**Interfaces:**
- Consumes: `ch_copula_run_` and `ch_mvdist_run_` from Task 4; `to_spec_json`, `spec_array`, `spec_number`, `spec_string` from `corehydror/R/spec.R`.
- Produces: the exported surface listed in the spec. `corehydro_copula` is
  `list(family =, theta =, df =, margin_x =, margin_y =, spec =)` with class `"corehydro_copula"`;
  `corehydro_mvdist` is `list(family =, spec =)` with class `"corehydro_mvdist"`.

- [ ] **Step 1: Write the failing tests**

Create `corehydror/tests/testthat/test-copula.R`:

```r
# Behavioural tests for the copula surface. Oracle VALUES live in fixtures/ and are asserted by
# test-fixtures.R; this file asserts argument handling, error messages, and object shape.

set.seed(1)
x <- c(135.9, 104.1, 108.7, 99.3, 134.7, 91.0, 77.3, 115.4, 109.0, 79.0)
y <- c(1.9, 1.3, 1.4, 1.2, 1.8, 1.1, 0.9, 1.5, 1.4, 1.0)

test_that("copula() builds and evaluates", {
  cop <- copula("Clayton", theta = 2)
  expect_s3_class(cop, "corehydro_copula")
  expect_equal(cop$family, "Clayton")
  expect_true(is.finite(copula_pdf(cop, 0.3, 0.7)))
  expect_true(copula_cdf(cop, 0.3, 0.7) > 0)
  expect_equal(copula_params(cop), 2)
})

test_that("tail dependence comes back named", {
  td <- copula_tail_dependence(copula("Clayton", theta = 2))
  expect_named(td, c("lower", "upper"))
  expect_equal(unname(td[["lower"]]), 2^(-1 / 2), tolerance = 1e-12)
})

test_that("the two exceedance probabilities differ and both lie in the unit interval", {
  cop <- copula("Gumbel", theta = 2)
  a <- copula_exceedance(cop, 0.9, 0.9, type = "and")
  o <- copula_exceedance(cop, 0.9, 0.9, type = "or")
  expect_true(a >= 0 && a <= 1 && o >= 0 && o <= 1)
  expect_false(isTRUE(all.equal(a, o)))
})

test_that("copula_bounds reports the theta range", {
  b <- copula_bounds(copula("Clayton", theta = 2))
  expect_named(b, c("minimum", "maximum"))
})

test_that("a seeded copula draw is reproducible and has the right shape", {
  cop <- copula("Clayton", theta = 2,
                margin_x = distribution("Normal", c(0, 1)),
                margin_y = distribution("Normal", c(0, 1)))
  d <- copula_random(cop, 5, seed = 12345)
  expect_equal(dim(d), c(5L, 2L))
  expect_identical(d, copula_random(cop, 5, seed = 12345))
})

test_that("copula_fit returns a usable copula", {
  cop <- copula_fit("Clayton", x, y, method = "mpl")
  expect_s3_class(cop, "corehydro_copula")
  expect_true(is.finite(cop$theta))
  expect_true(is.finite(copula_pdf(cop, 0.3, 0.7)))
})

test_that("the tau method is refused where upstream has no SetThetaFromTau", {
  expect_error(copula_fit("Joe", x, y, method = "tau"), "tau")
})

test_that("the three log-likelihoods run and differ", {
  cop <- copula("Clayton", theta = 2,
                margin_x = distribution("Normal", c(110, 20)),
                margin_y = distribution("Normal", c(1.4, 0.3)))
  ll_ifm <- copula_log_likelihood(cop, x, y, method = "ifm")
  ll_ps  <- copula_log_likelihood(cop, x, y, method = "pseudo")
  expect_true(is.finite(ll_ifm) && is.finite(ll_ps))
  expect_false(isTRUE(all.equal(ll_ifm, ll_ps)))
})

test_that("copula_names lists the seven families", {
  expect_length(copula_names(), 7L)
  expect_true("StudentT" %in% copula_names())
})

test_that("a copula round-trips through save and load", {
  cop <- copula("Frank", theta = 3)
  f <- tempfile(); saveRDS(cop, f)
  expect_equal(copula_cdf(readRDS(f), 0.4, 0.6), copula_cdf(cop, 0.4, 0.6))
  unlink(f)
})
```

Create `corehydror/tests/testthat/test-mvdist.R`:

```r
# Behavioural tests for the multivariate surface. Oracle VALUES live in fixtures/ and are
# asserted by test-fixtures.R; this file asserts shape, argument handling, and error messages.

S <- matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), nrow = 3)

test_that("mvdist_normal builds and evaluates", {
  mv <- mvdist_normal(c(1, 2, 3), S)
  expect_s3_class(mv, "corehydro_mvdist")
  expect_equal(mvdist_dimension(mv), 3L)
  expect_equal(mvdist_mean(mv), c(1, 2, 3))
  expect_equal(dim(mvdist_covariance(mv)), c(3L, 3L))
  expect_true(is.finite(mvdist_pdf(mv, c(1, 2, 3))))
})

test_that("marginal takes 1-based indices and returns a new object", {
  mv <- mvdist_normal(c(1, 2, 3), S)
  m <- mvdist_marginal(mv, c(1, 3))
  expect_s3_class(m, "corehydro_mvdist")
  expect_equal(mvdist_dimension(m), 2L)
  expect_equal(mvdist_mean(m), c(1, 3))
})

test_that("conditional takes 1-based indices and returns the complement", {
  mv <- mvdist_normal(c(1, 2, 3), S)
  cd <- mvdist_conditional(mv, given = 2, values = 5)
  expect_equal(mvdist_dimension(cd), 2L)
  expect_equal(mvdist_mean(cd), c(1, 3))
})

test_that("interval integrates to one over the whole space", {
  mv <- mvdist_normal(c(0, 0), diag(2))
  expect_equal(mvdist_interval(mv, c(-100, -100), c(100, 100)), 1, tolerance = 1e-6)
})

test_that("a seeded multivariate draw has the right shape and repeats", {
  mv <- mvdist_normal(c(0, 0), diag(2))
  d <- mvdist_random(mv, 4, seed = 12345)
  expect_equal(dim(d), c(4L, 2L))
  expect_identical(d, mvdist_random(mv, 4, seed = 12345))
  expect_equal(dim(mvdist_random(mv, 4, seed = 12345, method = "latin_hypercube")), c(4L, 2L))
})

test_that("latin hypercube needs an explicit seed", {
  mv <- mvdist_normal(c(0, 0), diag(2))
  expect_error(mvdist_random(mv, 4, method = "latin_hypercube"), "seed")
})

test_that("the other four families build", {
  expect_true(is.finite(mvdist_pdf(mvdist_dirichlet(c(2, 3, 4)), c(0.2, 0.3, 0.5))))
  expect_true(is.finite(mvdist_pdf(mvdist_multinomial(10, c(0.2, 0.3, 0.5)), c(2, 3, 5))))
  expect_equal(mvdist_dimension(mvdist_student_t(5, c(0, 0))), 2L)
})

test_that("the upstream gaps are named, not leaked", {
  expect_error(mvdist_cdf(mvdist_dirichlet(c(2, 3)), c(0.5, 0.5)), "Dirichlet")
  expect_error(mvdist_marginal(mvdist_student_t(5, c(0, 0)), 1), "MultivariateStudentT")
  expect_error(mvdist_interval(mvdist_student_t(5, c(0, 0)), c(-1, -1), c(1, 1)),
               "MultivariateStudentT")
})
```

- [ ] **Step 2: Run to verify they fail**

```bash
Rscript -e 'testthat::test_local("corehydror", filter = "copula|mvdist")'
```

Expected: FAIL, `could not find function "copula"`.

- [ ] **Step 3: Write `corehydror/R/copula.R`**

Follow `R/distribution.R`'s house style: roxygen with a runnable `@examples` on every export, validation before serialization, `stop(..., call. = FALSE)`.

```r
# The bivariate copula surface. Every verb serializes a corehydro_copula to the dist_spec.hpp
# grammar and runs one method through ch_copula_run_; nothing holds C++ state.

kCopulaFamilies <- c("AliMikhailHaq", "Clayton", "Frank", "Gumbel", "Joe", "Normal", "StudentT")

copula_run <- function(cop, method, args = numeric(0)) {
  res <- ch_copula_run_(cop$spec, method, to_spec_json(spec_array(as.double(args))))
  out <- res$values
  if (length(res$names)) names(out) <- res$names
  out
}

#' Construct a bivariate copula
#'
#' @param family one of [copula_names()].
#' @param theta the dependence parameter.
#' @param df degrees of freedom, required for `"StudentT"` and ignored otherwise.
#' @param margin_x,margin_y optional `corehydro_dist` marginals. Both are required by
#'   [copula_random()] on the data scale and by the IFM and full log-likelihoods.
#' @return a `corehydro_copula`.
#' @examples
#' copula_pdf(copula("Clayton", theta = 2), 0.3, 0.7)
#' @export
copula <- function(family, theta, df = NULL, margin_x = NULL, margin_y = NULL) {
  if (!is.character(family) || length(family) != 1L || !family %in% kCopulaFamilies) {
    stop(sprintf("`family` must be one of %s", paste(kCopulaFamilies, collapse = ", ")),
         call. = FALSE)
  }
  if (!is.numeric(theta) || length(theta) != 1L || !is.finite(theta)) {
    stop("`theta` must be a single finite number", call. = FALSE)
  }
  if (identical(family, "StudentT") && is.null(df)) {
    stop("`df` is required for the StudentT copula", call. = FALSE)
  }
  for (nm in c("margin_x", "margin_y")) {
    m <- get(nm)
    if (!is.null(m) && !inherits(m, "corehydro_dist")) {
      stop(sprintf("`%s` must be a corehydro_dist", nm), call. = FALSE)
    }
  }
  spec <- to_spec_json(list(family = family, theta = as.double(theta),
                            df = if (is.null(df)) NULL else as.double(df),
                            margin_x = margin_x, margin_y = margin_y))
  structure(list(family = family, theta = as.double(theta), df = df,
                 margin_x = margin_x, margin_y = margin_y, spec = spec),
            class = "corehydro_copula")
}

#' Copula tail dependence
#'
#' @param cop a `corehydro_copula`.
#' @return a named numeric of length two, `lower` and `upper`.
#' @examples
#' copula_tail_dependence(copula("Clayton", theta = 2))
#' @export
copula_tail_dependence <- function(cop) {
  stopifnot(inherits(cop, "corehydro_copula"))
  copula_run(cop, "tail_dependence")
}
```

`to_spec_json()` drops `NULL` entries from a named list (`spec.R:55`), so the optional `df` and marginals need no branching.

`copula()` validates `family` against `kCopulaFamilies`, requires `theta` to be a single finite number, requires `df` when `family == "StudentT"`, checks that `margin_x` and `margin_y` are `corehydro_dist` objects when supplied, and stores the serialized spec. `copula_fit()` serializes a `{"fit": {...}}` spec, runs `theta` and `parameters` to read the fitted values back, and returns a fully parameterized `corehydro_copula` (so a fitted copula and a constructed one are the same object).

`copula_random()` reshapes the `2n` flat values into an `n x 2` matrix. `copula_log_likelihood()` maps `method` to `log_likelihood_pseudo` / `log_likelihood_ifm` / `log_likelihood_full` and concatenates `x` then `y` into the args array. `copula_exceedance()` maps `type` to `exceedance_or` / `exceedance_and`. `copula_names()` returns `kCopulaFamilies`.

- [ ] **Step 4: Write `corehydror/R/mvdist.R`**

Same shape. The five constructors serialize their spec; the verbs route through `ch_mvdist_run_`. Three details:

```r
# Internal: user-facing indices are 1-based (matching trend()/model_parameter()); the spec and
# the C++ take 0-based.
mv_indices <- function(idx, n, what) {
  idx <- as.integer(idx)
  if (any(idx < 1L) || any(idx > n))
    stop(sprintf("`%s` must be between 1 and %d", what, n), call. = FALSE)
  idx - 1L
}
```

`mvdist_marginal()` and `mvdist_conditional()` read `res$spec` and wrap it in a new `corehydro_mvdist`:

```r
mvdist_run <- function(mv, method, args = numeric(0)) {
  ch_mvdist_run_(mv$spec, method, to_spec_json(spec_array(as.double(args))))
}

#' Marginal distribution of a multivariate normal
#'
#' @param mv a `corehydro_mvdist` of family `"MultivariateNormal"`.
#' @param indices the 1-based dimensions to keep.
#' @return a `corehydro_mvdist` over those dimensions.
#' @examples
#' mvdist_mean(mvdist_marginal(mvdist_normal(c(1, 2, 3), diag(3)), c(1, 3)))
#' @export
mvdist_marginal <- function(mv, indices) {
  stopifnot(inherits(mv, "corehydro_mvdist"))
  idx <- mv_indices(indices, mvdist_dimension(mv), "indices")
  res <- mvdist_run(mv, "marginal", idx)
  structure(list(family = "MultivariateNormal", spec = res$spec), class = "corehydro_mvdist")
}
```

`mvdist_random()` errors when `method = "latin_hypercube"` and `seed` is `NULL`, then reshapes `n * dimension` row-major values into an `n x dimension` matrix (`matrix(res$values, nrow = n, byrow = TRUE)`). `mvdist_covariance()` reshapes `dimension^2` row-major values the same way.

- [ ] **Step 5: Regenerate, install, and run the tests**

```bash
Rscript -e 'roxygen2::roxygenise("corehydror")'
R CMD INSTALL corehydror
Rscript -e 'testthat::test_local("corehydror", filter = "copula|mvdist")'
```

Expected: PASS, 0 failures.

- [ ] **Step 6: Run the whole R suite**

```bash
Rscript -e 'testthat::test_local("corehydror")'
```

Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add corehydror/R/copula.R corehydror/R/mvdist.R corehydror/NAMESPACE corehydror/man \
        corehydror/tests/testthat/test-copula.R corehydror/tests/testthat/test-mvdist.R
git commit -m "feat: the copula and multivariate surface in R

Seven copulas with pdf/cdf/tail dependence/joint exceedance/three
log-likelihoods/MPL-IFM-MLE-tau fitting, and five multivariate distributions
with MultivariateNormal's marginal, conditional and rectangle probability.
Indices are 1-based, matching trend() and model_parameter()."
```

---

## Task 6: The Python surface

**Files:**
- Create: `corehydropy/src/bindings/dist_spec.cpp`
- Create: `corehydropy/src/corehydropy/copula.py`
- Create: `corehydropy/src/corehydropy/mvdist.py`
- Create: `corehydropy/tests/test_copula.py`, `corehydropy/tests/test_mvdist.py`
- Modify: `corehydropy/src/corehydropy/distributions.py`, `__init__.py`
- Modify: `corehydropy/src/bindings/bindings.hpp`, `gev.cpp`, `corehydropy/CMakeLists.txt`
- Modify: `corehydropy/tests/test_distributions.py`

**Interfaces:**
- Consumes: the same three runner entry points as Task 4.
- Produces: `_core.dist_spec_run`, `_core.copula_run`, `_core.mvdist_run`, each returning a dict with keys `values`, `names`, `spec`; and the Python twins of every R export from Tasks 4 and 5, named identically.

- [ ] **Step 1: Write the failing tests**

Create `corehydropy/tests/test_copula.py` and `test_mvdist.py` as direct translations of the R files from Task 5: same cases, same order, same names where Python allows, with the header comment naming the R file they mirror (follow `corehydropy/tests/test_fit.py:1-20`). Add the composite cases from Task 4 to `corehydropy/tests/test_distributions.py` the same way. Use `pytest.raises(ValueError, match=...)` where R uses `expect_error`, and `np.testing.assert_array_equal` where R uses `expect_identical` on a numeric vector.

- [ ] **Step 2: Run to verify they fail**

```bash
pixi run python -m pytest corehydropy/tests/test_copula.py -q
```

Expected: FAIL, `ImportError: cannot import name 'copula'`.

- [ ] **Step 3: Write the pybind11 glue**

Create `corehydropy/src/bindings/dist_spec.cpp` with a `register_dist_spec(py::module_& m)` exposing the three entry points, following `corehydropy/src/bindings/estimation.cpp:492-552`'s shape (no docstring on the `m.def`; the explanation lives in the C++ comment above it):

```cpp
    m.def(
        "dist_spec_run",
        [](const std::string& spec_json, const std::string& method, const std::string& args_json) {
            supp::DistResult r = supp::run_dist(spec_json, method, args_json);
            py::dict out;
            out["values"] = r.values;
            out["names"] = r.names;
            out["spec"] = r.spec;
            return out;
        },
        py::arg("spec_json"), py::arg("method"), py::arg("args_json"));
```

plus `copula_run` and `mvdist_run` in the same shape. Declare `register_dist_spec` in `bindings.hpp`, call it from `PYBIND11_MODULE` in `gev.cpp` with a one-line comment, and add the source to `corehydropy/CMakeLists.txt`.

- [ ] **Step 4: Write the Python surface**

`distributions.py` gains the five module functions returning `Distribution`, the `spec` payload on `Distribution`, the two targeted errors in `Distribution.__init__`, and `distribution_names(kind="flat")`. `copula.py` and `mvdist.py` mirror `corehydror/R/copula.R` and `mvdist.R` verb for verb, with the verbs as methods on `Copula` and `MultivariateDistribution` and `copula_fit` / `copula_names` as module functions:

```python
_COPULA_FAMILIES = ("AliMikhailHaq", "Clayton", "Frank", "Gumbel", "Joe", "Normal", "StudentT")


class Copula:
    """A bivariate copula.

    Stateless: the object holds its spec as a JSON string and every verb runs one
    method through ``_core.copula_run``. Nothing holds C++ state, so instances
    pickle and compare across processes.
    """

    def __init__(self, family, theta, df=None, margin_x=None, margin_y=None):
        if family not in _COPULA_FAMILIES:
            raise ValueError(f"family must be one of {', '.join(_COPULA_FAMILIES)}")
        if family == "StudentT" and df is None:
            raise ValueError("df is required for the StudentT copula")
        spec = {"family": family, "theta": float(theta)}
        if df is not None:
            spec["df"] = float(df)
        for key, m in (("margin_x", margin_x), ("margin_y", margin_y)):
            if m is not None:
                if not isinstance(m, Distribution):
                    raise TypeError(f"{key} must be a Distribution")
                spec[key] = json.loads(m.to_json())
        self._family = family
        self._spec = json.dumps(spec)

    def _run(self, method, args=()):
        res = _core.copula_run(self._spec, method, json.dumps([float(a) for a in args]))
        return res["values"], res["names"]

    def tail_dependence(self) -> dict:
        """Lower and upper tail dependence coefficients."""
        values, names = self._run("tail_dependence")
        return dict(zip(names, values))
```

`Distribution.to_json()` must return the same grammar the R `to_spec_json()` emits, so a composite returns its stored spec and a flat family returns `{"family": ..., "parameters": [...]}`. Every public object gets numpydoc and a `to_json()`; `__getstate__`/`__setstate__` are unnecessary because the state is plain strings and lists, but add a pickle test to prove it.

Indices are 1-based in the Python signatures too, converted to 0-based before serialization, matching `models.py:77`'s existing rule.

Export everything from `corehydropy/src/corehydropy/__init__.py`.

- [ ] **Step 5: Build, install, and run the tests**

```bash
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests/test_copula.py corehydropy/tests/test_mvdist.py \
    corehydropy/tests/test_distributions.py -q
```

Expected: PASS, 0 failures.

- [ ] **Step 6: Check R and Python agree**

```bash
Rscript -e 'library(corehydror); cat(sprintf("%.17g\n", dist_pdf(dist_mixture(list(distribution("Normal", c(0,1)), distribution("Normal", c(5,1))), c(0.5,0.5)), 1.5)))'
pixi run python -c "
from corehydropy import distribution, dist_mixture
m = dist_mixture([distribution('Normal', [0,1]), distribution('Normal', [5,1])], [0.5,0.5])
print('%.17g' % m.pdf(1.5))"
```

Expected: byte-identical output from both.

- [ ] **Step 7: Commit**

```bash
git add corehydropy/src corehydropy/tests corehydropy/CMakeLists.txt
git commit -m "feat: the composite, copula and multivariate surface in Python

The twin of the R surface, name for name, over the same three runner entry
points."
```

---

## Task 7: The R and Python fixture runners delegate, and the bespoke glue is deleted

**Files:**
- Modify: `corehydror/tests/testthat/test-fixtures.R`, `corehydropy/tests/test_fixtures.py`
- Modify: `corehydror/src/dist.cpp`, `mvd.cpp`, `copula.cpp`
- Modify: `corehydropy/src/bindings/dist.cpp`, `mvd.cpp`, `copula.cpp`

**Interfaces:**
- Consumes: `ch_dist_spec_run_` / `ch_copula_run_` / `ch_mvdist_run_` and their Python twins.
- Produces: nothing new. The gate is that no oracle value moves.

The functions to delete, all fixture-only: `ch_trunc_*` (`dist.cpp:174-203`), `ch_emp_*` (:221-252), `ch_kde_*` (:278-308), `ch_mix_*` (:339-386), `ch_cr_*` (:442-507), `ch_cop_val_` / `ch_cop_fit_` (`copula.cpp:45`, `:107`), and every `ch_dirichlet_*` / `ch_multinomial_*` / `ch_bve_*` / `ch_mvn_*` / `ch_mvt_*` in `mvd.cpp`, with the pybind11 twins in the matching files.

- [ ] **Step 1: Capture the pre-change baselines**

```bash
Rscript -e 'testthat::test_local("corehydror", filter = "fixtures")' 2>&1 | tail -3
pixi run python -m pytest corehydropy/tests/test_fixtures.py -q 2>&1 | tail -3
```

Record both PASS counts. They must be identical after this task.

- [ ] **Step 2: Rewrite the R fixture runner's three paths**

Replace `build_composite_data` + `dispatch_composite` (`test-fixtures.R:150-270`) with a spec builder plus one call:

```r
# --- Composite distribution path -------------------------------------------------------
# The fixture construct schema IS the dist_spec.hpp grammar, so a composite case serializes
# straight through ch_dist_spec_run_. Adding a composite needs no change here at all.

composite_spec_json <- function(target, construct, datasets = list()) {
  construct$family <- target
  if (is.character(construct$data)) construct$data <- as.double(unlist(datasets[[construct$data]]))
  jsonlite::toJSON(construct, auto_unbox = TRUE, digits = I(17), null = "null")
}
```

`jsonlite` is already a test-only `Suggests` dependency, so the test runner may use it; the package code may not. Then the case loop calls `ch_dist_spec_run_(spec, fixture_method(method), args_json)` and indexes `values` exactly as the C++ runner does in Task 3, with the same `fixture_method` and `fixture_pick` mapping. Do the same for `run_copula_case` and the multivariate path, deleting `dispatch_copula`, `build_copula_params`, `dispatch_multivariate`, `run_mvn_case`, and `flatten_mv_args` once nothing references them.

- [ ] **Step 3: Rewrite the Python fixture runner's three paths**

The identical rewrite in `corehydropy/tests/test_fixtures.py`, using `json.dumps` where R uses `jsonlite::toJSON`, deleting `_build_composite`, `_dispatch_composite`, `_build_copula_params`, `_dispatch_copula`, `_run_copula_case`'s bespoke half, `_dispatch_multivariate`, `_run_mvn_case`, and `_flatten_mv_args`.

- [ ] **Step 4: Delete the bespoke glue**

Remove the functions listed above from all six glue files. Re-run `cpp11::cpp_register("corehydror")` so `cpp11.cpp` and `R/cpp11.R` drop them too.

- [ ] **Step 5: Rebuild both packages and compare the counts**

```bash
Rscript -e 'cpp11::cpp_register("corehydror")'
R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror", filter = "fixtures")' 2>&1 | tail -3
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests/test_fixtures.py -q 2>&1 | tail -3
```

Expected: the exact counts from Step 1, 0 failures in both. A moved value means the delegation is wrong; do not touch a fixture in this task.

- [ ] **Step 6: Run both suites in full**

```bash
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pytest corehydropy/tests -q
```

Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add corehydror corehydropy
git commit -m "refactor: the R and Python fixture runners drive the shared runner

Deletes about thirty fixture-only glue functions across six files; the
composite, copula and multivariate paths now serialize a spec and call one
entry point, exactly as the C++ runner does. Pass counts are identical before
and after."
```

---

## Task 8: Pin every newly exposed verb

**Files:**
- Modify: `fixtures/distributions/univariate/{truncated_distribution,mixture,competing_risks,empirical_distribution,kernel_density}.json`
- Modify: `fixtures/distributions/copulas/*.json` (all seven)
- Modify: `tools/oracle_emitter/Program.cs`
- Modify: `fixtures/README.md`

**Interfaces:**
- Consumes: the four runners from Tasks 3, 6, and 7.
- Produces: fixture cases for every verb in the gap table, and the emitter arms that reproduce them.

**The gap table** (measured from the current fixture corpus; each cell is a verb with no oracle):

| Target | Verbs to cover |
|---|---|
| TruncatedDistribution | `log_pdf`, `random_value`, `log_likelihood` |
| Mixture | `log_pdf`, `random_value`, `log_likelihood`, `median`, `mode`, `skewness`, `kurtosis` |
| CompetingRisks | `random_value`, `log_likelihood`, `mode` |
| Empirical | `pdf`, `log_pdf`, `random_value`, `log_likelihood`, `mean`, `mode`, `sd`, `skewness`, `kurtosis` |
| KernelDensity | `log_pdf`, `random_value`, `log_likelihood`, `mean`, `median`, `mode`, `sd`, `skewness`, `kurtosis` |
| All seven copulas | `or_exceedance`, `and_exceedance`, `log_likelihood_pseudo`, `log_likelihood_ifm`, `log_likelihood_full`, `theta_minimum`, `theta_maximum` |
| Copulas other than Normal | `log_pdf` |
| Copulas other than StudentT | `inverse_cdf` |

MultivariateNormal, MultivariateStudentT, Dirichlet, and Multinomial are already covered for every verb this phase exposes; add nothing there.

The emitter needs no grammar work. It already parses the fixture construct schema, which is what Task 1 promoted, so `BuildComposite` (`Program.cs:91-201`), `BuildCopula` (`:1224`), and `BuildMultivariate` (`:922`) read the phase-3 grammar as they stand. Only the new assertion methods are missing.

- [ ] **Step 1: Add the assertion methods to the four runners**

`log_likelihood`, `theta_minimum`, and `theta_maximum` do not exist as fixture methods yet. Add them to `core/tests/test_fixtures.cpp`'s method mapping, the R and Python runners' mapping, and `tools/oracle_emitter/Program.cs`'s `Dispatch` / `DispatchCopula`. In the emitter, `log_likelihood` is `dist.LogLikelihood(sample)`, `theta_minimum` / `theta_maximum` are `copula.ThetaMinimum` / `ThetaMaximum`; read the C# property names in the submodule before writing, and do not guess.

- [ ] **Step 2: Add the fixture cases with placeholder expectations**

For each row of the gap table, add cases to the fixture file with `"expected": 0` and the tolerance the neighbouring cases use, following the file's existing case shape. Give every case a `name` that says what it covers (`mixture_log_pdf`, `clayton_and_exceedance`, and so on). Set `"source"` on any new case whose value does not come from the C# test files to the reference you actually used.

- [ ] **Step 3: Emit the real values**

```bash
python3 tools/verify_oracles.py --emit 2>&1 | tail -20
```

Read `tools/verify_oracles.py --help` first; if it has no emit mode, run the emitter directly (`dotnet run --project tools/oracle_emitter -- <fixture>`) and paste the printed values into the fixtures. Either way the value written to a fixture must be the one the real C# library produced, never one the C++ produced.

- [ ] **Step 4: Run the gate**

```bash
python3 tools/verify_oracles.py
```

Expected: the Task 0 reproduced count plus the new cases, `0 failed`, `11 skipped`. No new skips. If a value will not reproduce, record why in `docs/upstream-csharp-issues.md` and assert a structural invariant instead; do not loosen a tolerance and do not add an `oracle_skip`.

- [ ] **Step 5: Run all four runners**

```bash
./core/build/test_fixtures fixtures 2>&1 | tail -3
Rscript -e 'testthat::test_local("corehydror", filter = "fixtures")' 2>&1 | tail -3
pixi run python -m pytest corehydropy/tests/test_fixtures.py -q 2>&1 | tail -3
```

Expected: each count grown by the new assertions, 0 failures.

- [ ] **Step 6: Document the grammar and the new methods**

In `fixtures/README.md`, document the `dist_spec.hpp` grammar, the `target`/`params` aliasing, and the three new assertion methods, following the file's existing section layout.

- [ ] **Step 7: Commit**

```bash
git add fixtures tools/oracle_emitter/Program.cs
git commit -m "test: pin every newly exposed distribution verb against the real C# library

Covers the measured gaps: composite log_pdf/random/log_likelihood and the
missing moments, and the copula joint-exceedance probabilities, three
log-likelihoods, log_pdf, inverse_cdf and theta bounds."
```

---

## Task 9: Cross-language digest fixtures

**Files:**
- Create: `fixtures/distributions/cross_language/{composite_digest,copula_digest,mvdist_digest}.json`
- Modify: the four runners, if the digest assertion method does not already exist

**Interfaces:**
- Consumes: the `random` methods on all three runners.
- Produces: three fixtures proving a seeded draw is bit-identical in R and Python.

- [ ] **Step 1: Find the existing digest precedent**

```bash
grep -rl "short_exact" fixtures/ | head
grep -rn "short_exact" core/tests/test_fixtures.cpp | head
```

Read one of those fixtures and the runner arm that serves it. The new fixtures must use the same mechanism, not a new one.

- [ ] **Step 2: Write the three fixtures**

Each asserts a small number of individual draws at exact tolerance (the `short_exact` mode) from a seeded stream:

- `composite_digest.json`: a two-component Normal mixture, `random` with `n = 20, seed = 12345`, asserting draws 0, 9, and 19.
- `copula_digest.json`: Clayton with `theta = 2` and Normal marginals, `random` with `n = 20, seed = 12345`, asserting the first and last pair.
- `mvdist_digest.json`: a 3-dimensional MultivariateNormal, `random_lhs` with `n = 20, seed = 12345`, asserting three coordinates.

- [ ] **Step 3: Pin them against C#**

```bash
python3 tools/verify_oracles.py
```

Expected: reproduced count grown by the new assertions, `0 failed`. Add the emitter arms if the gate reports the new targets as undriven.

- [ ] **Step 4: Prove R and Python agree**

```bash
./core/build/test_fixtures fixtures 2>&1 | tail -3
Rscript -e 'testthat::test_local("corehydror", filter = "fixtures")' 2>&1 | tail -3
pixi run python -m pytest corehydropy/tests/test_fixtures.py -q 2>&1 | tail -3
```

Expected: all three pass the new assertions. That is the cross-language proof: the same pinned digits are checked by all three runners.

- [ ] **Step 5: Commit**

```bash
git add fixtures tools/oracle_emitter/Program.cs core/tests/test_fixtures.cpp
git commit -m "test: cross-language digests for seeded composite, copula and MVN draws"
```

---

## Task 10: Reference documentation

**Files:**
- Modify: `corehydror/_pkgdown.yml`, `site/_quarto.yml`
- Modify: the roxygen blocks in `corehydror/R/{distribution,copula,mvdist}.R` and the numpydoc in the three Python modules

**Interfaces:**
- Consumes: every export from Tasks 4, 5, and 6.
- Produces: a reference index entry for each, in both halves of the site.

- [ ] **Step 1: List every new export**

```bash
grep -c "^export" corehydror/NAMESPACE
git diff main --stat -- corehydror/NAMESPACE
pixi run python -c "import corehydropy; print(len(corehydropy.__all__) if hasattr(corehydropy,'__all__') else 'no __all__')"
```

Write the list down. Every name on it needs an entry in both files.

- [ ] **Step 2: Add the pkgdown sections**

In `corehydror/_pkgdown.yml`, add "Copulas" and "Multivariate distributions" sections and extend the distributions section with the five composite constructors. Follow the existing section syntax exactly.

- [ ] **Step 3: Add the quartodoc sections**

In `site/_quarto.yml`, add the matching `quartodoc.sections` entries.

- [ ] **Step 4: Build the site**

```bash
pixi run docs
```

Expected: success. pkgdown fails loudly on a missing reference-index entry, so a clean build is the check.

- [ ] **Step 5: Inspect both halves**

```bash
pixi run docs-serve
```

Open `/reference/` (Python) and `/r/reference/` (R) and confirm the new sections render with every function present. `quarto preview` cannot serve the pkgdown half; use `docs-serve`.

- [ ] **Step 6: Commit**

```bash
git add corehydror/_pkgdown.yml site/_quarto.yml corehydror/man corehydror/R corehydropy/src
git commit -m "docs: reference entries for the composite, copula and multivariate surface"
```

---

## Task 11: The two worked example pairs

**Files:**
- Create: `site/examples/26-copulas-and-joint-frequency/{python.ipynb, r.qmd}`
- Create: `site/examples/27-composite-distributions/{python.ipynb, r.qmd}`
- Modify: `site/examples/index.qmd`, `site/examples/coverage.qmd`
- Modify: `site/_freeze/` (regenerated, committed)

**Interfaces:**
- Consumes: the full R and Python surface.
- Produces: two example pairs, each ending in an executable reproduction check like the existing thirteen.

- [ ] **Step 1: Read two existing examples end to end**

```bash
ls site/examples/
```

Read `site/examples/25-estimation-methods/r.qmd` and its notebook twin in full. Match their structure, their front matter, their prose register, and the shape of their closing reproduction check.

- [ ] **Step 2: Write `26-copulas-and-joint-frequency`**

Peak and volume pairs; fit by MPL and by IFM; the and-joint exceedance probability of a coincident event; tail dependence; a cross-check against the existing `bivariate_analysis()` model path. Every number the prose states must be printed by a chunk on the same page. Prose claims about how a method works must be checked against the C++ or C# source, not assumed: a wrong sentence here is the visible end of a real gap.

- [ ] **Step 3: Write `27-composite-distributions`**

A mixed-population flood record as a two-component mixture; a kernel density against the parametric fit; a truncated distribution for a physical lower bound. Same rules.

- [ ] **Step 4: Execute the notebooks and regenerate the freeze**

```bash
pixi run jupyter nbconvert --to notebook --execute --inplace \
    site/examples/26-copulas-and-joint-frequency/python.ipynb
pixi run jupyter nbconvert --to notebook --execute --inplace \
    site/examples/27-composite-distributions/python.ipynb
pixi run docs
```

Expected: both notebooks execute cleanly with outputs committed, and `site/_freeze/` updates for the two R twins.

- [ ] **Step 5: List them on the index and coverage pages**

Add both pairs to `site/examples/index.qmd` and to the coverage page, following the existing entries' format.

- [ ] **Step 6: Verify the reproduction checks pass in both languages**

Run each page's closing check by hand in R and in Python and confirm the two agree to the tolerance the page states.

- [ ] **Step 7: Commit**

```bash
git add site/examples site/_freeze
git commit -m "docs: worked example pairs for copulas and composite distributions"
```

---

## Task 12: Release 0.5.0 and the full verification sweep

**Files:**
- Modify: `CHANGELOG.md`, `corehydror/DESCRIPTION`, `corehydropy/pyproject.toml`, the core version stamp
- Modify: `.claude/CLAUDE.md` (the Status section)

**Interfaces:**
- Consumes: everything.
- Produces: a phase that is done and provably so.

- [ ] **Step 1: Find the core version stamp**

```bash
grep -rn "0\.4\.0" core/ corehydror/DESCRIPTION corehydropy/pyproject.toml | grep -v build
```

Every hit is a place to bump.

- [ ] **Step 2: Bump all three to 0.5.0 and write the changelog entry**

Follow the 0.4.0 entry's structure in `CHANGELOG.md`. State what became reachable, not how it was built.

- [ ] **Step 3: Run the full verification sweep**

```bash
cmake -S core -B core/build && cmake --build core/build -j && ctest --test-dir core/build
python3 tools/verify_oracles.py
Rscript -e 'cpp11::cpp_register("corehydror")'
R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
R CMD build corehydror && R CMD check --as-cran corehydror_*.tar.gz
pixi run docs
```

Expected: ctest and both test suites grown against the Task 0 baselines with 0 failures; the oracle gate grown with `0 failed` and `11 skipped`; `R CMD check --as-cran` at its three known NOTEs with no WARNING; the docs build clean.

- [ ] **Step 4: Run the end-to-end cross-language check**

```r
peaks   <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500)
volumes <- c(41000, 52000, 27000, 78000, 61000, 47000, 31000, 96000)
mix <- dist_mixture(list(distribution("Normal", c(12000, 2000)),
                         distribution("Normal", c(25000, 4000))), c(0.7, 0.3))
sprintf("%.17g", dist_pdf(mix, 15000))
sprintf("%.17g", dist_random(mix, 5, seed = 12345))
cop <- copula_fit("Clayton", peaks, volumes, method = "mpl",
                  margin_x = "GeneralizedExtremeValue", margin_y = "LogNormal")
sprintf("%.17g", copula_exceedance(cop, 0.99, 0.99, type = "and"))
mv <- mvdist_normal(c(0, 0, 0), diag(3))
sprintf("%.17g", mvdist_mean(mvdist_conditional(mv, given = 2, values = 1.5)))
```

Run the identical calls in Python. Every printed figure must match byte for byte. If `copula_fit`'s `margin_x` argument turns out to take a `corehydro_dist` rather than a family name, fix this snippet and the spec's copy of it rather than the API.

- [ ] **Step 5: Update the repo status note**

Add a paragraph to the Status section of `.claude/CLAUDE.md` describing what this phase delivered and the final measured numbers, matching the register of the phase paragraphs already there.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md corehydror/DESCRIPTION corehydropy/pyproject.toml core .claude/CLAUDE.md
git commit -m "chore: release 0.5.0

The distribution layer is complete in both packages: five composite families,
seven copulas, five multivariate distributions."
```

- [ ] **Step 7: Report, do not push**

Report the final measured numbers against the Task 0 baselines. Do not push and do not open a PR without being asked.

---

## Notes for the implementer

- Two names in Task 1 Step 3 and two in Task 2 Step 3 are marked "check the header before compiling". They are the only places this plan asks you to confirm a spelling rather than giving it. Do confirm them; a guessed enum name compiles into a wrong branch more often than it fails to compile.
- Task 3 and Task 7 are refactors whose whole gate is a count that does not move. If a count moves, the delegation is wrong. Never edit a fixture to make a refactor pass.
- The phase adds no numerical code. If a verb seems to be missing from the core, it is missing from C# too. Surface an error naming the limitation; do not implement it.
