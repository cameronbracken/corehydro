# Callback Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user's own R or Python function drive the core: MCMC with custom priors and a custom log-likelihood (un-gating Gibbs), bootstrap with custom resample/fit/statistic delegates, GMM with custom moment conditions, and the math verbs (root finding, quadrature, derivatives).

**Architecture:** Two new header-only pieces in the C++ core. `callback_guard.hpp` holds the exception guard that today lives inside `optimizer_runner.hpp`, generalized over the other callback signatures, so there is one guard implementation rather than two. `callback_runner.hpp` plus four group headers under `numerics/support/callback/` dispatch a group and method against live `std::function`s, mirroring how `toolbox_runner.hpp` and its `numerics/support/toolbox/` group headers already work. Four callers drive it and none owns evaluation logic: the cpp11 glue, the pybind11 glue, the C++ fixture runner, and the dotnet oracle emitter. No C++ object crosses the language boundary except the RNG handle, which borrows a reference for the duration of one call.

**Tech Stack:** C++17 (header-only, no external deps), cpp11 for R, pybind11 for Python, nlohmann/json in the C++ test runner only, jsonlite in the R test runner only, dotnet 10 for the oracle gate.

Design document: `docs/superpowers/specs/2026-08-14-callback-surface-design.md`. Read it before Task 1.

## Global Constraints

- **No external C++ dependencies in `core/`.** Do not add Eigen, fmt, or anything else. The core is self-contained by design.
- **Never use `M_PI`.** Use `corehydro::numerics::kPi` (`core/include/corehydro/numerics/tools.hpp`).
- **Never name a namespace alias `gamma` or `stat`.** Both collide with C runtime symbols.
- **Pass `-Wall/-Wextra` only to non-MSVC compilers.** Windows CI is in the matrix and has bitten this repo twice.
- **Oracle values live only in `fixtures/*.json`.** Never hardcode an expected numeric value in a test file.
- **After editing any `corehydror/src/*.cpp`,** run `Rscript -e 'cpp11::cpp_register("corehydror")'` before `R CMD INSTALL corehydror`.
- **`corehydror/NAMESPACE` is maintained by hand** (it says so on line 1). Every new export needs an `export()` line added manually. roxygen does not generate it.
- **Every new export must appear in BOTH `corehydror/_pkgdown.yml` and the `quartodoc.sections` in `site/_quarto.yml`.** pkgdown errors on a reference-index entry it cannot find.
- **pytest reads fixtures materialized into site-packages by pip**, not the repo symlink. After editing a fixture, re-run `pixi run python -m pip install --force-reinstall --no-deps ./corehydropy` before the count changes.
- **After any core change that alters a class layout,** rebuild R clean: `R CMD INSTALL --preclean corehydror`. Stale `.o` files from a prior ABI return garbage or abort R.
- **Commits are GPG-signed automatically.** Identity `Cam Bracken <cameron.bracken@pm.me>`. Do NOT add a `Co-Authored-By` trailer. Push only when asked.
- **Ported files carry a provenance header:** `// ported from: <upstream path> @ <sha>`. The pinned shas are Numerics `2a0357a` and RMC-BestFit `c2e6192`. Note the upstream filename typo: `AdaptiveGuassKronrod.cs`.
- **Branch:** `surface-callback-layer`, already created from `origin/main` and carrying the design commit.

## Baselines that must not regress

Current counts, measured on the design commit. Every task grows them; no task may shrink them.

| Gate | Current |
|---|---|
| ctest | 84/84 |
| oracle gate (`verify_oracles.py`) | 5209 reproduced, 0 failed, 11 skipped |
| testthat | 5634/0 |
| pytest | 1344 |
| `R CMD check --as-cran` | 3 NOTEs, no WARNING |

The 11 oracle skips are the documented GEV standard-error set. No task may add a skip.

## Build and test commands

```bash
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py
```

## Reference implementations to copy

Read these before writing anything. Every task below follows their shape.

| Concern | Model to copy |
|---|---|
| Runner header, four callers, no logic of its own | `core/include/corehydro/numerics/support/optimizer_runner.hpp` |
| Group-per-header layout under one dispatcher | `core/include/corehydro/numerics/support/toolbox_runner.hpp` + `numerics/support/toolbox/` |
| cpp11 callback glue | `corehydror/src/toolbox.cpp`, `ch_optim_run_` (lines 52-80) |
| pybind11 callback glue | `corehydropy/src/bindings/toolbox.cpp`, `optim_run` |
| C++ fixture runner arm | `core/tests/test_fixtures.cpp`, `run_optimizer_kind` (line 1646) |
| Fixture naming a callback | `fixtures/toolbox/optimizers.json`, the `"objective": "FXYZ"` key |
| ctest harness macros | `core/tests/check.hpp`: `CHECK_EQ`, `CHECK_TRUE`, `CHECK_NEAR`, `CHECK_THROWS`, `CHECK_THROWS_MSG`, ending `return chtest::summary("<suite>");`. Model file: `core/tests/test_dist_runner.cpp` |
| R result packing for MCMC | `corehydror/src/mcmc.cpp`, `ch_mcmc_run_` (lines 61-180) |

## File Structure

**Create in `core/`:**

| File | Responsibility |
|---|---|
| `include/corehydro/numerics/support/callback_guard.hpp` | `GuardedCall<TResult, TArgs...>`: latch the first host exception, short-circuit later calls, rethrow on demand. Nothing else. |
| `include/corehydro/numerics/support/callback_runner.hpp` | Parse `options_json`, dispatch to a group, return a flat `CallbackResult`. No math. |
| `include/corehydro/numerics/support/callback/common.hpp` | Shared `CallbackResult` and option-access helpers for the four groups. |
| `include/corehydro/numerics/support/callback/math.hpp` | `detail::run_math`: root finding, quadrature, derivative, gradient, hessian. |
| `include/corehydro/numerics/support/callback/mcmc.hpp` | `detail::run_mcmc`: build priors from dist specs, construct the named sampler, sample, pack results. |
| `include/corehydro/numerics/support/callback/bootstrap.hpp` | `detail::run_bootstrap`: wire the four delegates, run, return confidence intervals. |
| `include/corehydro/numerics/support/callback/gmm.hpp` | `detail::run_gmm`: the delegate-constructor GMM path. |
| `include/corehydro/numerics/math/integration/integrator.hpp` | Ported `Integrator` base: settings, status, evaluation count. |
| `tests/test_callback_runner.cpp` | ctest suite for the runner and guard. |

**Modify in `core/`:**

| File | Change |
|---|---|
| `include/corehydro/numerics/support/optimizer_runner.hpp` | Delete the inline `GuardedObjective`; alias it to `callback_guard.hpp`'s guard. Behavior identical. |
| `include/corehydro/numerics/math/integration/adaptive_gauss_kronrod.hpp` | Complete the port onto the `Integrator` base. |
| `tests/test_fixtures.cpp` | Add `run_callback_kind` beside `run_optimizer_kind`. |
| `CMakeLists.txt` | Register `test_callback_runner`. |

**Create in the packages:**

| File | Responsibility |
|---|---|
| `corehydror/src/callback.cpp` | cpp11 glue: one `ch_callback_*` entry per group, plus the RNG handle external pointer. |
| `corehydror/R/callback.R` | `mcmc_posterior`, `bootstrap_custom`, `fit_gmm_moments`, `root_find`, `quadrature`, `derivative`, `gradient`, `hessian` with roxygen docs and argument validation. |
| `corehydropy/src/bindings/callback.cpp` | pybind11 glue, mirroring the cpp11 glue call for call, plus the RNG handle class. |
| `corehydropy/src/corehydropy/callback.py` | The Python verbs, mirroring `callback.R` verb for verb. |
| `fixtures/callback/*.json` | The `callback` fixture kind: `math.json`, `mcmc.json`, `bootstrap.json`, `gmm.json`, `callback_cross_language.json`. |

**Modify in the packages:** `corehydror/NAMESPACE`, `corehydror/_pkgdown.yml`, `corehydropy/src/corehydropy/__init__.py`, `corehydropy/src/bindings/module.cpp`, `site/_quarto.yml`, `site/status.qmd`, `CHANGELOG.md`, both version files, `tools/oracle_emitter/`, and both fixture runners (`corehydror/tests/testthat/test-fixtures.R`, `corehydropy/tests/test_fixtures.py`).

## Task sequence and dependencies

Task 1 establishes the guard, the runner, the fixture kind, and the four-caller wiring using the smallest group. Every later task adds one group to machinery that already exists. Task 2 is independent of Tasks 3-7. Task 3 must precede Tasks 5 and 6, which need the RNG handle. Task 8 is last.

---

### Task 1: Callback guard and runner foundation, proven with the math group

Builds the whole pattern end to end on the three math verbs that need no new porting: `root_find`, `derivative`, `gradient`, `hessian`. Quadrature waits for Task 2.

**Files:**
- Create: `core/include/corehydro/numerics/support/callback_guard.hpp`
- Create: `core/include/corehydro/numerics/support/callback_runner.hpp`
- Create: `core/include/corehydro/numerics/support/callback/common.hpp`
- Create: `core/include/corehydro/numerics/support/callback/math.hpp`
- Create: `core/tests/test_callback_runner.cpp`
- Create: `corehydror/src/callback.cpp`, `corehydror/R/callback.R`
- Create: `corehydropy/src/bindings/callback.cpp`, `corehydropy/src/corehydropy/callback.py`
- Create: `fixtures/callback/math.json`
- Modify: `core/include/corehydro/numerics/support/optimizer_runner.hpp` (guard extraction)
- Modify: `core/CMakeLists.txt`, `core/tests/test_fixtures.cpp`
- Modify: `corehydror/NAMESPACE`, `corehydropy/src/corehydropy/__init__.py`, `corehydropy/src/bindings/module.cpp`
- Modify: `corehydror/tests/testthat/test-fixtures.R`, `corehydropy/tests/test_fixtures.py`
- Test: `core/tests/test_callback_runner.cpp`, `corehydror/tests/testthat/test-callback.R`, `corehydropy/tests/test_callback.py`

**Interfaces:**
- Produces, consumed by every later task:
  - `corehydro::numerics::support::GuardedCall<TResult, TArgs...>` with `operator()(TArgs...)`, `aborted()`, `rethrow_if_aborted()`, `call_count()`, and a constructor taking `(std::function<TResult(TArgs...)> fn, TResult sentinel)`.
  - `struct CallbackResult { std::vector<double> values; std::vector<std::string> names; std::vector<int> dims; std::string status; }` in `callback/common.hpp`.
  - `CallbackResult run_callback(const std::string& group, const std::string& method, const std::string& options_json, const CallbackSet& callbacks)` in `callback_runner.hpp`, where `CallbackSet` is a struct of named `std::function` members, empty by default, defined in `common.hpp`.
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Read the two reference headers before writing**

Read `core/include/corehydro/numerics/support/optimizer_runner.hpp` lines 1-160 (the file header, `Objective`, `OptimResult`, and `GuardedObjective`) and `core/include/corehydro/numerics/support/toolbox_runner.hpp` in full. The new runner is a hybrid: the guard and callback plumbing of the first, the group dispatch of the second.

- [ ] **Step 2: Write the failing ctest for the guard**

Create `core/tests/test_callback_runner.cpp`:

```cpp
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/support/callback_guard.hpp"
#include "check.hpp"

namespace sup = corehydro::numerics::support;

struct HostError : std::runtime_error {
    HostError() : std::runtime_error("host language error") {}
};

int main() {
    // A guard that never sees a throw is transparent and counts real calls.
    {
        sup::GuardedCall<double, const std::vector<double>&> g(
            [](const std::vector<double>& p) { return p[0] * 2.0; }, -1.0);
        CHECK_EQ(g({3.0}), 6.0);
        CHECK_EQ(g({4.0}), 8.0);
        CHECK_TRUE(!g.aborted());
        CHECK_EQ(g.call_count(), 2);
    }

    // The first throw latches, later calls short-circuit to the sentinel WITHOUT re-entering
    // the host function, and the stored exception rethrows with its original type.
    {
        int entries = 0;
        sup::GuardedCall<double, const std::vector<double>&> g(
            [&entries](const std::vector<double>&) -> double {
                ++entries;
                throw HostError();
            },
            -1.0);
        CHECK_EQ(g({1.0}), -1.0);
        CHECK_EQ(g({2.0}), -1.0);
        CHECK_EQ(entries, 1);          // second call never reached the host
        CHECK_EQ(g.call_count(), 0);   // no call completed
        CHECK_TRUE(g.aborted());
        CHECK_THROWS_MSG(g.rethrow_if_aborted(), "host language error");
    }

    // A vector-returning guard (the shape the Gibbs proposal and bootstrap resample need).
    {
        sup::GuardedCall<std::vector<double>, const std::vector<double>&> g(
            [](const std::vector<double>& p) { return std::vector<double>{p[0], p[0]}; },
            std::vector<double>{});
        CHECK_EQ(g({5.0}).size(), std::size_t{2});
        CHECK_TRUE(!g.aborted());
    }

    return chtest::summary("callback_runner");
}
```

- [ ] **Step 3: Register the test and run it to verify it fails**

Add to `core/CMakeLists.txt` beside the other `add_executable`/`add_test` pairs (copy the exact form used for `test_dist_runner`):

```cmake
add_executable(test_callback_runner tests/test_callback_runner.cpp)
target_include_directories(test_callback_runner PRIVATE include tests)
add_test(NAME test_callback_runner COMMAND test_callback_runner)
```

Run: `cmake -S core -B core/build && cmake --build core/build 2>&1 | tail -20`
Expected: FAIL, `callback_guard.hpp: No such file or directory`.

- [ ] **Step 4: Write the guard**

Create `core/include/corehydro/numerics/support/callback_guard.hpp`:

```cpp
// corehydro ADDITION -- no upstream C# counterpart.
//
// The one place a host-language (R/Python) exception thrown inside a callback is latched so it
// survives a ported class's catch-all. Extracted from optimizer_runner.hpp (where it was
// GuardedObjective, serving Optimizer::minimize()'s catch-all) and generalized, because the
// callback surface has five more crossing points with different signatures: the MCMC
// log-likelihood, the Gibbs proposal, the HMC/NUTS gradient, the four bootstrap delegates, and
// the GMM moment conditions. Every one of them sits under ported code that may catch broadly,
// and a raw host exception must not be allowed to unwind a C++ frame it did not originate in.
//
// Contract: the FIRST throw is stored and an abort flag latches. Every later call returns the
// sentinel WITHOUT re-entering the host, so one run can never raise a second host exception (or
// a second R longjmp) once the first is captured. The caller rethrows after the ported algorithm
// returns, inside the same protected frame the callback was invoked from.
#pragma once
#include <exception>
#include <functional>
#include <utility>

namespace corehydro::numerics::support {

template <typename TResult, typename... TArgs>
class GuardedCall {
   public:
    using Function = std::function<TResult(TArgs...)>;

    // `sentinel` is returned for every call after an abort. Each call site chooses a value its
    // ported consumer treats as unconditionally rejected: +/-infinity for an objective (see
    // optimizer_runner.hpp), -infinity for a log-likelihood, an empty vector for a proposal.
    GuardedCall(Function fn, TResult sentinel)
        : fn_(std::move(fn)), sentinel_(std::move(sentinel)) {}

    TResult operator()(TArgs... args) {
        if (aborted_) return sentinel_;
        try {
            TResult v = fn_(std::forward<TArgs>(args)...);
            ++call_count_;
            return v;
        } catch (...) {
            error_ = std::current_exception();
            aborted_ = true;
            return sentinel_;
        }
    }

    bool aborted() const { return aborted_; }
    void rethrow_if_aborted() const {
        if (error_) std::rethrow_exception(error_);
    }
    // Calls that actually completed in the host. Excludes short-circuited calls.
    int call_count() const { return call_count_; }
    explicit operator bool() const { return static_cast<bool>(fn_); }

   private:
    Function fn_;
    TResult sentinel_;
    bool aborted_ = false;
    std::exception_ptr error_;
    int call_count_ = 0;
};

}  // namespace corehydro::numerics::support
```

- [ ] **Step 5: Run the ctest to verify it passes**

Run: `cmake --build core/build && ctest --test-dir core/build -R test_callback_runner --output-on-failure`
Expected: PASS, `[PASS] callback_runner`.

- [ ] **Step 6: Switch optimizer_runner.hpp onto the shared guard**

In `optimizer_runner.hpp`, delete the `GuardedObjective` class body and replace it with an alias, keeping the existing explanatory comment block above it (it documents why the guard exists at all and must not be lost):

```cpp
#include "corehydro/numerics/support/callback_guard.hpp"

// ... existing comment block explaining the guard ...
//
// The guard itself now lives in callback_guard.hpp, shared with callback_runner.hpp. The only
// optimizer-specific part is the sentinel: minimizing wants a small value, so a huge one is
// worst (+inf); maximizing wants a large one, so a hugely negative one is worst (-inf). Either
// way the optimizer treats the aborted point as unconditionally rejected.
using GuardedObjective = GuardedCall<double, const std::vector<double>&>;

inline GuardedObjective make_guarded_objective(const Objective& fn, bool maximize) {
    return GuardedObjective(fn, maximize ? -std::numeric_limits<double>::infinity()
                                         : std::numeric_limits<double>::infinity());
}
```

In `run_optimizer`, replace `GuardedObjective guarded(objective, maximize);` with
`GuardedObjective guarded = make_guarded_objective(objective, maximize);` and replace every
`guarded.function_evaluations()` call with `guarded.call_count()`.

- [ ] **Step 7: Run the full ctest suite to prove the extraction changed nothing**

Run: `cmake --build core/build && ctest --test-dir core/build`
Expected: 85/85 (the 84 existing plus `test_callback_runner`). If any optimizer test now fails, the extraction is wrong; fix it before continuing rather than adjusting the test.

- [ ] **Step 8: Commit the guard**

```bash
git add core/include/corehydro/numerics/support/callback_guard.hpp core/tests/test_callback_runner.cpp core/CMakeLists.txt core/include/corehydro/numerics/support/optimizer_runner.hpp
git commit -m "refactor(core): extract the callback guard into its own header

The optimizer objective was the only host-language callback in the core, so
its guard lived inside optimizer_runner.hpp. The callback surface adds five
more crossing points with different signatures, so the guard becomes a
generic GuardedCall<TResult, TArgs...> and the optimizer aliases it."
```

- [ ] **Step 9: Write the failing ctest for the math group**

Add `#include "corehydro/numerics/support/callback_runner.hpp"` to the test file, then append the following before `return chtest::summary(...)`:

```cpp
    // --- math group -----------------------------------------------------------------------
    {
        sup::CallbackSet cbs;
        cbs.scalar = [](double x) { return x * x - 2.0; };
        sup::CallbackResult r =
            sup::run_callback("math", "root_find", R"({"lower": 0.0, "upper": 2.0})", cbs);
        CHECK_NEAR(r.values.at(0), 1.4142135623730951, 1e-12);
    }
    {
        sup::CallbackSet cbs;
        cbs.vector_scalar = [](const std::vector<double>& p) {
            return (1.0 - p[0]) * (1.0 - p[0]) + 100.0 * (p[1] - p[0] * p[0]) * (p[1] - p[0] * p[0]);
        };
        sup::CallbackResult r =
            sup::run_callback("math", "gradient", R"({"point": [1.0, 1.0]})", cbs);
        CHECK_EQ(r.values.size(), std::size_t{2});
        CHECK_NEAR(r.values.at(0), 0.0, 1e-6);
        CHECK_NEAR(r.values.at(1), 0.0, 1e-6);
    }
    // A host exception inside the callback survives the ported algorithm and reaches the caller.
    {
        sup::CallbackSet cbs;
        cbs.scalar = [](double) -> double { throw HostError(); };
        CHECK_THROWS_MSG(
            sup::run_callback("math", "root_find", R"({"lower": 0.0, "upper": 2.0})", cbs),
            "host language error");
    }
```

- [ ] **Step 10: Run it to verify it fails**

Run: `cmake --build core/build 2>&1 | tail -10`
Expected: FAIL, `callback_runner.hpp: No such file or directory`.

- [ ] **Step 11: Write common.hpp**

Create `core/include/corehydro/numerics/support/callback/common.hpp`:

```cpp
// corehydro ADDITION -- no upstream C# counterpart. Shared types for the callback groups,
// sibling of numerics/support/toolbox/common.hpp.
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::support {

using corehydro::models::spec::JsonValue;

// Every callback a group may need. A caller fills only the members its group uses; the rest stay
// empty and each group validates the ones it requires. One struct rather than a variant keeps the
// four bindings' call sites uniform: build the set, name the group and method, call.
struct CallbackSet {
    // f(x) -> y. Root finding, quadrature, single-variable differentiation.
    std::function<double(double)> scalar;
    // f(theta) -> y. Log-likelihood, gradient/hessian target, GMM penalty.
    std::function<double(const std::vector<double>&)> vector_scalar;
    // f(theta) -> vector. Gradient callback, GMM moment conditions, bootstrap statistic.
    std::function<std::vector<double>(const std::vector<double>&)> vector_vector;
    // f(theta, rng) -> vector. Gibbs proposal.
    std::function<std::vector<double>(const std::vector<double>&,
                                      corehydro::numerics::sampling::MersenneTwister&)>
        vector_rng;
    // f(data, theta, rng) -> data. Bootstrap resample.
    std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&,
                                      corehydro::numerics::sampling::MersenneTwister&)>
        data_rng;
    // f(data) -> theta. Bootstrap fit.
    std::function<std::vector<double>(const std::vector<double>&)> data_vector;
    // f(data, index) -> data. Bootstrap jackknife.
    std::function<std::vector<double>(const std::vector<double>&, int)> data_index;
    // f(theta) -> matrix, row-major with dims. GMM jacobian, pointwise moment conditions.
    std::function<std::pair<std::vector<double>, std::vector<int>>(const std::vector<double>&)>
        vector_matrix;
};

// Flat result surface every binding and every fixture assertion reads. `dims` is empty for a
// plain vector, or {rows, cols} row-major for a matrix. `names` labels `values` where a group has
// something to say (parameter names, statistic names); otherwise empty.
struct CallbackResult {
    std::vector<double> values;
    std::vector<std::string> names;
    std::vector<int> dims;
    std::string status;
};

namespace detail {

inline JsonValue parse_options(const std::string& options_json) {
    if (options_json.empty()) return JsonValue{};
    return corehydro::models::spec::parse_json(options_json);
}

inline double require_double(const JsonValue& o, const char* key, const char* group) {
    if (!o.contains(key))
        throw std::invalid_argument(std::string(group) + " requires the option '" + key + "'");
    return o.at(key).as_double();
}

inline std::vector<double> require_vector(const JsonValue& o, const char* key, const char* group) {
    if (!o.contains(key))
        throw std::invalid_argument(std::string(group) + " requires the option '" + key + "'");
    return o.at(key).as_double_vector();
}

}  // namespace detail
}  // namespace corehydro::numerics::support
```

Check the MersenneTwister include path against the tree before compiling; if the header sits elsewhere, use the path `mcmc_sampler.hpp` uses for it.

- [ ] **Step 12: Write the math group**

Create `core/include/corehydro/numerics/support/callback/math.hpp`:

```cpp
// corehydro ADDITION -- no upstream C# counterpart. The math group of callback_runner.hpp:
// the ported Numerics routines whose input is a user function rather than serializable data.
// Root finding is Brent (numerics/math/rootfinding/brent.hpp); differentiation is the ported
// NumericalDerivative (numerics/math/differentiation/numerical_derivative.hpp). Quadrature is
// added in Task 2, once AdaptiveGaussKronrod is a complete port.
#pragma once
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"
#include "corehydro/numerics/support/callback/common.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"

namespace corehydro::numerics::support::detail {

namespace rootfinding = corehydro::numerics::math::rootfinding;
namespace differentiation = corehydro::numerics::math::differentiation;

inline CallbackResult run_math(const std::string& method, const JsonValue& o,
                               const CallbackSet& cbs) {
    CallbackResult r;
    r.status = "Success";

    if (method == "root_find") {
        if (!cbs.scalar) throw std::invalid_argument("math/root_find requires a scalar function");
        GuardedCall<double, double> g(cbs.scalar, std::numeric_limits<double>::quiet_NaN());
        double lower = require_double(o, "lower", "math/root_find");
        double upper = require_double(o, "upper", "math/root_find");
        double root = rootfinding::solve([&g](double x) { return g(x); }, lower, upper);
        g.rethrow_if_aborted();
        r.values = {root};
        r.names = {"root"};
        return r;
    }

    if (method == "derivative") {
        if (!cbs.scalar) throw std::invalid_argument("math/derivative requires a scalar function");
        GuardedCall<double, double> g(cbs.scalar, std::numeric_limits<double>::quiet_NaN());
        double point = require_double(o, "point", "math/derivative");
        double d = differentiation::derivative([&g](double x) { return g(x); }, point);
        g.rethrow_if_aborted();
        r.values = {d};
        r.names = {"derivative"};
        return r;
    }

    if (method == "gradient" || method == "hessian") {
        if (!cbs.vector_scalar)
            throw std::invalid_argument("math/" + method + " requires a vector function");
        GuardedCall<double, const std::vector<double>&> g(
            cbs.vector_scalar, -std::numeric_limits<double>::infinity());
        std::vector<double> point = require_vector(o, "point", "math");
        auto f = [&g](const std::vector<double>& p) { return g(p); };
        if (method == "gradient") {
            r.values = differentiation::gradient(f, point);
            g.rethrow_if_aborted();
            r.dims = {static_cast<int>(r.values.size())};
        } else {
            std::vector<std::vector<double>> h = differentiation::hessian(f, point);
            g.rethrow_if_aborted();
            int n = static_cast<int>(h.size());
            r.values.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
            for (const auto& row : h) r.values.insert(r.values.end(), row.begin(), row.end());
            r.dims = {n, n};
        }
        return r;
    }

    throw std::invalid_argument("unknown math method: " + method);
}

}  // namespace corehydro::numerics::support::detail
```

Confirm the exact `gradient`/`hessian`/`derivative` signatures and any bounds arguments against `numerical_derivative.hpp` lines 87-210 before compiling; pass the bounds overloads only if the free functions require them.

- [ ] **Step 13: Write the runner**

Create `core/include/corehydro/numerics/support/callback_runner.hpp`:

```cpp
// corehydro ADDITION -- no upstream C# counterpart (sibling of toolbox_runner.hpp,
// optimizer_runner.hpp, distributions/support/dist_runner.hpp, and
// estimation/support/fit_runner.hpp).
//
// The single place a ported class runs against a live host-language callback other than an
// optimizer objective. Upstream's own API for these classes IS delegate-driven -- see
// MCMCSampler.cs:43, Gibbs.cs:33, HMC.cs:42, Bootstrap.cs:128-152, and
// GeneralizedMethodOfMoments.cs:24-58 -- so this runner surfaces the real C# contract; the model
// registries under sampling/mcmc/ and sampling/bootstrap/ are corehydro fixture scaffolding, not
// the upstream shape.
//
// Four callers drive it and none owns any evaluation logic: the cpp11 glue
// (corehydror/src/callback.cpp), the pybind11 glue (corehydropy/src/bindings/callback.cpp), the
// C++ fixture runner (core/tests/test_fixtures.cpp), and the dotnet oracle emitter, which
// serializes the identical options grammar and dispatches equivalent C# delegates.
//
// Split of responsibilities, matching toolbox_runner.hpp: serializable configuration (sampler
// name, iteration counts, seed, prior specs, CI method) travels in `options_json`; bulk data
// travels as native double vectors; the callbacks arrive as std::function members of a
// CallbackSet. No callback registry lives here -- every caller supplies its own closures, so
// every fixture case exercises the real host callback path, which is the thing this file exists
// to make safe.
#pragma once
#include <string>

#include "corehydro/numerics/support/callback/bootstrap.hpp"
#include "corehydro/numerics/support/callback/common.hpp"
#include "corehydro/numerics/support/callback/gmm.hpp"
#include "corehydro/numerics/support/callback/math.hpp"
#include "corehydro/numerics/support/callback/mcmc.hpp"

namespace corehydro::numerics::support {

// Runs `method` of `group` (one of "math", "mcmc", "bootstrap", "gmm") against the callbacks in
// `cbs`, configured by `options_json`, and returns a flat CallbackResult. Each group header
// documents its own options grammar and which CallbackSet members it requires.
inline CallbackResult run_callback(const std::string& group, const std::string& method,
                                   const std::string& options_json, const CallbackSet& cbs) {
    JsonValue o = detail::parse_options(options_json);
    if (group == "math") return detail::run_math(method, o, cbs);
    if (group == "mcmc") return detail::run_mcmc(method, o, cbs);
    if (group == "bootstrap") return detail::run_bootstrap(method, o, cbs);
    if (group == "gmm") return detail::run_gmm(method, o, cbs);
    throw std::invalid_argument("unknown callback group: " + group);
}

}  // namespace corehydro::numerics::support
```

For this task, create `callback/mcmc.hpp`, `callback/bootstrap.hpp`, and `callback/gmm.hpp` as stubs whose `run_mcmc`/`run_bootstrap`/`run_gmm` throw `std::invalid_argument("the <group> group lands in Task N")`, so the runner compiles now and each later task replaces one stub. Each stub keeps the `#pragma once` and the include of `common.hpp`.

- [ ] **Step 14: Run the ctest to verify it passes**

Run: `cmake --build core/build && ctest --test-dir core/build -R test_callback_runner --output-on-failure`
Expected: PASS.

- [ ] **Step 15: Commit the core half**

```bash
git add core/include/corehydro/numerics/support/callback_runner.hpp core/include/corehydro/numerics/support/callback/ core/tests/test_callback_runner.cpp
git commit -m "feat(core): add the callback runner with its math group"
```

- [ ] **Step 16: Write the failing R test**

Create `corehydror/tests/testthat/test-callback.R`:

```r
test_that("root_find solves a user-written R function", {
  expect_equal(root_find(function(x) x^2 - 2, lower = 0, upper = 2), sqrt(2), tolerance = 1e-12)
})

test_that("gradient and hessian differentiate a user-written R function", {
  rosenbrock <- function(p) (1 - p[1])^2 + 100 * (p[2] - p[1]^2)^2
  expect_equal(gradient(rosenbrock, c(1, 1)), c(0, 0), tolerance = 1e-6)
  h <- hessian(rosenbrock, c(1, 1))
  expect_equal(dim(h), c(2L, 2L))
  expect_equal(h[1, 2], h[2, 1], tolerance = 1e-6)
})

test_that("an error raised inside the callback reaches the caller unchanged", {
  expect_error(
    root_find(function(x) stop("my own error"), lower = 0, upper = 2),
    "my own error"
  )
})

test_that("a callback returning a non-scalar is rejected", {
  expect_error(
    root_find(function(x) c(1, 2), lower = 0, upper = 2),
    "single number"
  )
})
```

- [ ] **Step 17: Run it to verify it fails**

Run: `Rscript -e 'testthat::test_local("corehydror", filter = "callback")'`
Expected: FAIL, `could not find function "root_find"`.

- [ ] **Step 18: Write the cpp11 glue**

Create `corehydror/src/callback.cpp`. Copy the shape of `ch_optim_run_` in `src/toolbox.cpp` exactly, including its non-scalar-return check:

```cpp
// cpp11 glue for the callback surface (R/callback.R). Unlike the toolbox verbs, which pass
// serializable data through ch_toolbox_run_, every verb here takes a live R function, so it goes
// through callback_runner.hpp and its guard. Mirrors corehydropy's src/bindings/callback.cpp
// call for call.
#include <string>
#include <vector>

#include "corehydro/numerics/support/callback_runner.hpp"
#include "cpp11.hpp"

using namespace cpp11;
namespace sup = corehydro::numerics::support;

namespace {

// Converts an R closure into the scalar signature, raising the same "single number" error the
// optimizer glue raises. The throw travels back through GuardedCall, which latches it and lets
// run_callback rethrow it inside this same cpp11-protected frame.
std::function<double(double)> as_scalar_fn(function f) {
    return [f](double x) mutable -> double {
        sexp out = f(writable::doubles({x}));
        doubles v = as_doubles(out);
        if (v.size() != 1)
            throw std::runtime_error(
                "the function must return a single number; got a value of length " +
                std::to_string(static_cast<long long>(v.size())));
        return v[0];
    };
}

std::function<double(const std::vector<double>&)> as_vector_scalar_fn(function f) {
    return [f](const std::vector<double>& p) mutable -> double {
        writable::doubles par(static_cast<R_xlen_t>(p.size()));
        for (std::size_t i = 0; i < p.size(); ++i) par[static_cast<R_xlen_t>(i)] = p[i];
        sexp out = f(par);
        doubles v = as_doubles(out);
        if (v.size() != 1)
            throw std::runtime_error(
                "the function must return a single number; got a value of length " +
                std::to_string(static_cast<long long>(v.size())));
        return v[0];
    };
}

list pack(const sup::CallbackResult& r) {
    writable::doubles values(static_cast<R_xlen_t>(r.values.size()));
    for (std::size_t i = 0; i < r.values.size(); ++i) values[static_cast<R_xlen_t>(i)] = r.values[i];
    writable::strings names(static_cast<R_xlen_t>(r.names.size()));
    for (std::size_t i = 0; i < r.names.size(); ++i) names[static_cast<R_xlen_t>(i)] = r.names[i];
    writable::integers dims(static_cast<R_xlen_t>(r.dims.size()));
    for (std::size_t i = 0; i < r.dims.size(); ++i) dims[static_cast<R_xlen_t>(i)] = r.dims[i];
    return writable::list({"values"_nm = values, "names"_nm = names, "dims"_nm = dims,
                           "status"_nm = writable::strings({r.status})});
}

}  // namespace

[[cpp11::register]]
list ch_callback_math_(std::string method, std::string options_json, function f) {
    sup::CallbackSet cbs;
    if (method == "root_find" || method == "derivative" || method == "quadrature")
        cbs.scalar = as_scalar_fn(f);
    else
        cbs.vector_scalar = as_vector_scalar_fn(f);
    return pack(sup::run_callback("math", method, options_json, cbs));
}
```

- [ ] **Step 19: Write the R wrappers**

Create `corehydror/R/callback.R` with `root_find`, `derivative`, `gradient`, and `hessian`. Each validates its arguments in R (so the error names the user's argument, not an internal one), builds the options JSON with `jsonlite::toJSON(auto_unbox = TRUE)`, and calls the glue. Full roxygen blocks with `@examples` that run without `\donttest`, since these are all fast. `hessian` reshapes using `dims`:

```r
#' Find a root of a user-written function
#'
#' Solves `f(x) = 0` on `[lower, upper]` with the ported Numerics Brent root finder. `f` must
#' change sign across the interval.
#'
#' @param f a function taking one number and returning one number.
#' @param lower,upper the bracketing interval.
#' @return the root, a single number.
#' @examples
#' root_find(function(x) x^2 - 2, lower = 0, upper = 2)
#' @export
root_find <- function(f, lower, upper) {
  stopifnot(is.function(f), is.numeric(lower), is.numeric(upper),
            length(lower) == 1L, length(upper) == 1L)
  opts <- jsonlite::toJSON(list(lower = lower, upper = upper), auto_unbox = TRUE)
  ch_callback_math_("root_find", as.character(opts), f)$values[[1]]
}
```

Write `derivative(f, x)`, `gradient(f, x)`, and `hessian(f, x)` in the same shape; `hessian` returns
`matrix(res$values, nrow = res$dims[1], ncol = res$dims[2], byrow = TRUE)`.

- [ ] **Step 20: Register, install, and run the R test**

```bash
Rscript -e 'cpp11::cpp_register("corehydror")'
R CMD INSTALL corehydror
Rscript -e 'testthat::test_local("corehydror", filter = "callback")'
```
Expected: PASS, 4 tests. Add `export(root_find)`, `export(derivative)`, `export(gradient)`, and `export(hessian)` to `corehydror/NAMESPACE` by hand before installing.

- [ ] **Step 21: Write the failing Python test**

Create `corehydropy/tests/test_callback.py`:

```python
import math

import pytest

import corehydropy as ch


def test_root_find_solves_a_python_function():
    assert ch.root_find(lambda x: x**2 - 2, lower=0, upper=2) == pytest.approx(math.sqrt(2), abs=1e-12)


def test_gradient_and_hessian_differentiate_a_python_function():
    def rosenbrock(p):
        return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2

    g = ch.gradient(rosenbrock, [1.0, 1.0])
    assert g == pytest.approx([0.0, 0.0], abs=1e-6)
    h = ch.hessian(rosenbrock, [1.0, 1.0])
    assert h.shape == (2, 2)
    assert h[0, 1] == pytest.approx(h[1, 0], abs=1e-6)


def test_an_error_inside_the_callback_reaches_the_caller():
    with pytest.raises(ValueError, match="my own error"):
        ch.root_find(lambda x: (_ for _ in ()).throw(ValueError("my own error")), lower=0, upper=2)
```

- [ ] **Step 22: Run it to verify it fails**

Run: `pixi run python -m pytest corehydropy/tests/test_callback.py -q`
Expected: FAIL, `AttributeError: module 'corehydropy' has no attribute 'root_find'`.

- [ ] **Step 23: Write the pybind11 glue and Python wrappers**

Create `corehydropy/src/bindings/callback.cpp` mirroring the cpp11 glue, registering `callback_math` on the module; add `register_callback(m);` to `corehydropy/src/bindings/module.cpp` beside the existing `register_toolbox(m);`. Create `corehydropy/src/corehydropy/callback.py` with `root_find`, `derivative`, `gradient`, and `hessian`, numpydoc docstrings, and the same validation as the R side; `hessian` returns `np.asarray(res["values"]).reshape(res["dims"])`. Export all four from `__init__.py`'s `__all__`.

- [ ] **Step 24: Install and run the Python test**

```bash
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests/test_callback.py -q
```
Expected: PASS, 3 tests.

- [ ] **Step 25: Add the callback fixture kind**

Create `fixtures/callback/math.json` with the `callback` kind. Each case names its callback from the catalog rather than carrying code, exactly as `fixtures/toolbox/optimizers.json` does with `"objective"`:

```json
{
  "kind": "callback",
  "cases": [
    {
      "name": "root_find_sqrt2",
      "construct": {
        "group": "math",
        "method": "root_find",
        "callback": "x_squared_minus_two",
        "options": {"lower": 0.0, "upper": 2.0}
      },
      "assertions": [
        {"method": "value", "args": [0], "expected": 1.4142135623730951,
         "mode": "rel", "tol": 1e-12, "source": "Test_BrentSearch.Test_Root"}
      ]
    }
  ]
}
```

Pin every expected value from a real `Test_Numerics` case; find the matching tests under `upstream/Numerics/Test_Numerics/Mathematics/` and cite the test name in `source`. Do not invent a value.

Wire the kind into all four runners:
1. `core/tests/test_fixtures.cpp`: add `run_callback_kind` beside `run_optimizer_kind` (line 1646), resolving `"callback"` to a C++ closure by name.
2. `corehydror/tests/testthat/test-fixtures.R`: resolve the same name to a native R closure.
3. `corehydropy/tests/test_fixtures.py`: resolve it to a native Python closure.
4. `tools/oracle_emitter/`: dispatch the equivalent C# delegate.

Each runner writes its own closure for `x_squared_minus_two`. That is the point: every fixture case exercises the real host callback path in that language.

- [ ] **Step 26: Run all four runners**

```bash
cmake --build core/build && ctest --test-dir core/build
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy && pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py
```
Expected: ctest 85/85; testthat above 5634 with 0 failures; pytest above 1344; oracle gate above 5209 reproduced with 0 failed and still 11 skipped.

- [ ] **Step 27: Commit**

```bash
git add -A
git commit -m "feat: surface root finding and numerical differentiation over a user function

First verbs on the callback runner: root_find(), derivative(), gradient()
and hessian() take an R or Python function directly. Adds the callback
fixture kind across all four runners, each writing the catalog function as
a native closure so every case exercises the real host callback path."
```

---

### Task 2: Complete the AdaptiveGaussKronrod port and add quadrature()

**Files:**
- Create: `core/include/corehydro/numerics/math/integration/integrator.hpp`
- Modify: `core/include/corehydro/numerics/math/integration/adaptive_gauss_kronrod.hpp`
- Modify: `core/include/corehydro/numerics/support/callback/math.hpp`
- Modify: `corehydror/src/callback.cpp`, `corehydror/R/callback.R`, `corehydror/NAMESPACE`
- Modify: `corehydropy/src/bindings/callback.cpp`, `corehydropy/src/corehydropy/callback.py`, `__init__.py`
- Modify: `fixtures/callback/math.json`
- Test: `core/tests/test_callback_runner.cpp`, `corehydror/tests/testthat/test-callback.R`, `corehydropy/tests/test_callback.py`

**Interfaces:**
- Consumes: `run_callback`, `CallbackSet::scalar`, `GuardedCall` from Task 1.
- Produces: `math/quadrature` returning `values = {integral}` with `names = {"integral"}`, plus `status` from the ported `IntegrationStatus` and a second value `function_evaluations`.

- [ ] **Step 1: Read both sources side by side**

Read `upstream/Numerics/Numerics/Mathematics/Integration/Support/Integrator.cs` (150 lines), `upstream/Numerics/Numerics/Mathematics/Integration/AdaptiveGuassKronrod.cs` (313 lines), and the current partial port `core/include/corehydro/numerics/math/integration/adaptive_gauss_kronrod.hpp` (147 lines). The port keeps the G10K21 nodes and the recursion; what is missing is the `Integrator` base (settings, status, evaluation counting) and whatever the C# class does beyond the bare recursion.

- [ ] **Step 2: Write the failing ctest**

Append to `core/tests/test_callback_runner.cpp` a case pinning an integral whose value comes from a `Test_Numerics` integration test, plus a status check:

```cpp
    {
        sup::CallbackSet cbs;
        cbs.scalar = [](double x) { return x * x; };
        sup::CallbackResult r = sup::run_callback(
            "math", "quadrature", R"({"lower": 0.0, "upper": 3.0})", cbs);
        CHECK_NEAR(r.values.at(0), 9.0, 1e-10);
        CHECK_EQ(r.status, std::string("Success"));
        CHECK_TRUE(r.values.at(1) > 0.0);  // function evaluations were counted
    }
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cmake --build core/build && ctest --test-dir core/build -R test_callback_runner --output-on-failure`
Expected: FAIL, `unknown math method: quadrature`.

- [ ] **Step 4: Port the Integrator base**

Create `core/include/corehydro/numerics/math/integration/integrator.hpp` with the provenance header `// ported from: Numerics/Mathematics/Integration/Support/Integrator.cs @ 2a0357a`, mirroring the C# class member for member: the abstract integration interface, `Min`/`Max`, absolute and relative tolerance, maximum iterations and function evaluations, the `IntegrationStatus` enum (port it beside, from `IntegrationStatus.cs`), the evaluation counter, and the convergence and validation checks. Structural mirroring is the rule: keep the C# method order and names so a future upstream diff maps line for line.

- [ ] **Step 5: Complete the AGK port onto that base**

Rewrite `adaptive_gauss_kronrod.hpp` so the class derives from `Integrator` and carries the full C# surface, keeping the existing G10K21 node and weight tables (they are already correct and already oracle-backed through `VonMises::CDF`). Preserve the existing free-function `integrate(...)` as a thin wrapper so no existing caller changes; `VonMises::CDF` must keep working untouched.

- [ ] **Step 6: Run the full ctest suite**

Run: `cmake --build core/build && ctest --test-dir core/build`
Expected: 85/85. `VonMises` fixture cases must still pass with unchanged values; if any moves, the port changed behavior it should not have.

- [ ] **Step 7: Add the quadrature arm to the math group**

In `callback/math.hpp`, add the arm before the final throw:

```cpp
    if (method == "quadrature") {
        if (!cbs.scalar) throw std::invalid_argument("math/quadrature requires a scalar function");
        GuardedCall<double, double> g(cbs.scalar, std::numeric_limits<double>::quiet_NaN());
        double lower = require_double(o, "lower", "math/quadrature");
        double upper = require_double(o, "upper", "math/quadrature");
        integration::AdaptiveGaussKronrod agk([&g](double x) { return g(x); }, lower, upper);
        if (o.contains("absolute_tolerance"))
            agk.absolute_tolerance = o.at("absolute_tolerance").as_double();
        if (o.contains("relative_tolerance"))
            agk.relative_tolerance = o.at("relative_tolerance").as_double();
        if (o.contains("max_function_evaluations"))
            agk.max_function_evaluations = o.at("max_function_evaluations").as_int();
        agk.integrate();
        g.rethrow_if_aborted();
        r.values = {agk.result(), static_cast<double>(agk.function_evaluations())};
        r.names = {"integral", "function_evaluations"};
        r.status = integration::status_name(agk.status());
        return r;
    }
```

Adjust member and method names to whatever the completed port actually exposes.

- [ ] **Step 8: Run the ctest to verify it passes**

Run: `cmake --build core/build && ctest --test-dir core/build -R test_callback_runner --output-on-failure`
Expected: PASS.

- [ ] **Step 9: Add the R and Python verbs**

Add `quadrature(f, lower, upper, ...)` to both packages, returning the integral as a single number with the status and evaluation count attached as attributes in R (`attr(x, "status")`) and as fields on a small result object in Python. Add the export lines to `NAMESPACE` and `__all__`. Add a test to each package's callback test file integrating a function whose closed form is known.

Name it `quadrature`, not `integrate`: the latter masks `stats::integrate` in R.

- [ ] **Step 10: Add fixture cases and run all four runners**

Add quadrature cases to `fixtures/callback/math.json` pinned from `Test_Numerics` integration tests, with the callback name resolved natively in each of the four runners. Then run the full command block from Task 1 Step 26.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat(core): complete the AdaptiveGaussKronrod port and surface quadrature()

The integrator was a minimal port carrying only what VonMises::CDF needed.
It now derives from a ported Integrator base with settings, status and
evaluation counting, so quadrature() reports the same shape optim_minimize()
does."
```

---

### Task 3: The RNG handle

The bootstrap resample and Gibbs proposal delegates receive the core's seeded `MersenneTwister&`. This task surfaces that as an opaque handle in each package, with the lifetime rule enforced.

**Files:**
- Modify: `corehydror/src/callback.cpp` (external pointer), `corehydror/R/callback.R`
- Modify: `corehydropy/src/bindings/callback.cpp` (pybind11 class), `corehydropy/src/corehydropy/callback.py`
- Create: `fixtures/callback/rng_handle.json`
- Test: `corehydror/tests/testthat/test-callback.R`, `corehydropy/tests/test_callback.py`

**Interfaces:**
- Produces, consumed by Tasks 5 and 6: an R external pointer of class `corehydro_rng` with `rng_uniform(handle, n)` and `rng_integers(handle, n, min, max)`, and a Python class `Rng` with `.uniform(n)` and `.integers(n, min, max)`. Both borrow a `MersenneTwister&` and invalidate when the call that created them returns.

- [ ] **Step 1: Write the failing R test**

The handle cannot be constructed by a user, so it is tested through a verb that hands one out. Use a temporary test-only entry point rather than waiting for Task 5:

```r
test_that("the rng handle draws from the core seeded stream and invalidates after the call", {
  seen <- NULL
  captured <- NULL
  out <- ch_rng_probe_(seed = 12345, n = 3, f = function(rng) {
    captured <<- rng
    seen <<- rng_uniform(rng, 3)
    seen
  })
  expect_length(seen, 3)
  expect_true(all(seen > 0 & seen < 1))
  expect_equal(out, seen)
  # The handle borrows the generator; using it after the call must error, not read freed memory.
  expect_error(rng_uniform(captured, 1), "no longer valid")
})
```

`ch_rng_probe_` is a small registered entry point that creates a `MersenneTwister` from `seed`, hands a handle to `f`, and returns what `f` returned. Keep it registered but unexported; the fixture runners use it too.

- [ ] **Step 2: Run it to verify it fails**

Run: `Rscript -e 'testthat::test_local("corehydror", filter = "callback")'`
Expected: FAIL, `could not find function "ch_rng_probe_"`.

- [ ] **Step 3: Implement the handle in the cpp11 glue**

The handle wraps a raw pointer plus a validity flag held in a `shared_ptr`, so the wrapper can be invalidated when the owning call returns even though R still holds the SEXP:

```cpp
// Borrows the core generator for the duration of one call. The `valid` flag is cleared by
// RngScope's destructor, so an R user who stores the handle and calls it later gets an error
// instead of a dangling read.
struct RngBorrow {
    corehydro::numerics::sampling::MersenneTwister* prng = nullptr;
    bool valid = false;
};

class RngScope {
   public:
    explicit RngScope(corehydro::numerics::sampling::MersenneTwister& prng)
        : borrow_(std::make_shared<RngBorrow>()) {
        borrow_->prng = &prng;
        borrow_->valid = true;
    }
    ~RngScope() { borrow_->valid = false; }
    RngScope(const RngScope&) = delete;
    RngScope& operator=(const RngScope&) = delete;
    std::shared_ptr<RngBorrow> handle() const { return borrow_; }

   private:
    std::shared_ptr<RngBorrow> borrow_;
};
```

Expose it as a cpp11 `external_pointer<std::shared_ptr<RngBorrow>>`, and have `ch_rng_uniform_` and `ch_rng_integers_` check `valid` first, raising `"this random number generator handle is no longer valid; it can only be used inside the callback it was given to"`.

- [ ] **Step 4: Run the R test to verify it passes**

Run: `Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL corehydror && Rscript -e 'testthat::test_local("corehydror", filter = "callback")'`
Expected: PASS.

- [ ] **Step 5: Mirror it in Python**

Implement the same borrow-and-invalidate scheme as a pybind11 class `Rng` with `uniform(n)` and `integers(n, min, max)`, raising `RuntimeError` with the same message when used after its scope ends. Add the matching pytest.

- [ ] **Step 6: Pin the stream across languages**

Add `fixtures/callback/rng_handle.json` asserting the exact first draws from seed 12345, sourced from the C# `Random`/`MersenneTwister` test in `Test_Numerics`. Wire it into all four runners. This fixture is what proves a handle draw is the core stream and not a language-local generator, so R and Python agree.

- [ ] **Step 7: Run all four runners and commit**

Run the full command block from Task 1 Step 26, then:

```bash
git add -A
git commit -m "feat: give callbacks a handle on the core seeded generator

The bootstrap resample and Gibbs proposal delegates take the core's
MersenneTwister. The handle borrows it for one call and invalidates on
return, so a stored handle errors instead of reading freed memory."
```

---

### Task 4: mcmc_posterior over a user log-likelihood

**Files:**
- Modify: `core/include/corehydro/numerics/support/callback/mcmc.hpp` (replace the Task 1 stub)
- Modify: `corehydror/src/callback.cpp`, `corehydror/R/callback.R`, `corehydror/NAMESPACE`
- Modify: `corehydropy/src/bindings/callback.cpp`, `corehydropy/src/corehydropy/callback.py`, `__init__.py`
- Create: `fixtures/callback/mcmc.json`
- Test: `core/tests/test_callback_runner.cpp`, both package test files

**Interfaces:**
- Consumes: `run_callback`, `CallbackSet::vector_scalar`, `GuardedCall` (Task 1).
- Produces: `mcmc/sample`, returning the same field set `ch_mcmc_run_` returns today (`corehydror/src/mcmc.cpp` lines 56-180): parameter names, per-chain draws, acceptance rates, MAP values and fitness, posterior mean, standard deviation, median, credible bounds, R-hat, and effective sample size.

- [ ] **Step 1: Read the existing MCMC glue and reuse its packing**

Read `corehydror/src/mcmc.cpp` in full. It already builds all eight samplers from `(priors, log_likelihood)` and packs `MCMCResults` into an R list. The new path differs in exactly two ways: the priors come from `distribution()` specs instead of `build_model`, and the log-likelihood is the user's function instead of a registry closure. Extract the sampler-construction switch and the result packing into shared helpers rather than copying them, so the registry path and the callback path cannot drift.

- [ ] **Step 2: Write the failing ctest**

Append to `core/tests/test_callback_runner.cpp` a run of RWMH against a Gaussian log-density built from arithmetic only, with two uniform priors, asserting the posterior mean lands near the data mean and that the run is reproducible under a fixed seed:

```cpp
    {
        const std::vector<double> data = {4.9, 5.1, 5.0, 5.2, 4.8};
        sup::CallbackSet cbs;
        cbs.vector_scalar = [&data](const std::vector<double>& p) {
            double mu = p[0], sigma = p[1];
            if (sigma <= 0.0) return -std::numeric_limits<double>::infinity();
            double acc = 0.0;
            for (double x : data) acc += (x - mu) * (x - mu);
            return -0.5 * acc / (sigma * sigma) -
                   static_cast<double>(data.size()) * std::log(sigma);
        };
        std::string options = R"({
            "sampler": "RWMH", "iterations": 2000, "warmup": 500, "chains": 2, "seed": 12345,
            "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]},
                       {"family": "Uniform", "parameters": [0.01, 5.0]}]})";
        sup::CallbackResult r1 = sup::run_callback("mcmc", "sample", options, cbs);
        sup::CallbackResult r2 = sup::run_callback("mcmc", "sample", options, cbs);
        CHECK_EQ(r1.values, r2.values);  // a seeded run is deterministic
    }
```

Assert the posterior mean against a fixture value, not a literal, once the fixture exists; the ctest here proves determinism and wiring only.

- [ ] **Step 3: Run it to verify it fails**

Expected: FAIL, `the mcmc group lands in Task 4`.

- [ ] **Step 4: Implement the mcmc group**

Replace the stub `callback/mcmc.hpp`. It must:
1. Build the prior list by feeding each entry of `options["priors"]` through the existing dist-spec builder (`numerics/distributions/support/dist_spec.hpp`), producing `vector<shared_ptr<UnivariateDistributionBase>>`. Reuse that builder; do not write a second family parser.
2. Wrap the user log-likelihood in `GuardedCall<double, const std::vector<double>&>` with sentinel `-infinity`, which every sampler already treats as an unconditionally rejected point.
3. Construct the sampler named by `options["sampler"]`, mirroring the switch in `corehydror/src/mcmc.cpp` lines 72-115.
4. Apply the shared settings that are present: `iterations`, `warmup`, `chains`, `thinning`, `seed`, `initialize` (`MAP` or `Randomize`).
5. Call `sampler->sample()`, build `MCMCResults`, `rethrow_if_aborted()`, and flatten into `CallbackResult`.

Document in the header that `initialize = "MAP"` reaches the user's function through DifferentialEvolution, so it costs callback crossings before sampling begins.

- [ ] **Step 5: Run the ctest to verify it passes**

Run: `cmake --build core/build && ctest --test-dir core/build -R test_callback_runner --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Add the R and Python verbs**

`mcmc_posterior(log_likelihood, priors, sampler = "RWMH", iterations = NULL, warmup = NULL, chains = NULL, thinning = NULL, seed = 12345, initialize = c("MAP", "Randomize"))`, where `priors` is a list of `distribution()` objects. Return the same list shape `mcmc_sample()` returns, so a user can move between them. Mirror it in Python, taking a list of `Distribution` objects.

The roxygen and numpydoc blocks must both carry the cross-language warning from the design document: draws come from the core generator, but the log-density is the caller's own arithmetic, so a seeded run reproduces between R and Python only if that function returns bit-identical values, and one flipped accept or reject changes the whole chain.

- [ ] **Step 7: Write the package tests**

In each package: a run against a Gaussian log-density recovers a known mean within tolerance; two seeded runs are identical; an error raised inside the log-likelihood reaches the caller; a prior list whose length disagrees with the parameter vector errors clearly.

- [ ] **Step 8: Add the fixture and run all four runners**

Create `fixtures/callback/mcmc.json` naming its log-density from the catalog, pinned against a real C# run driven by the emitter with the equivalent delegate. Use an arithmetic-only log-density so the value reproduces exactly in all four languages. Wire the catalog entry into all four runners, then run the full command block.

- [ ] **Step 9: Benchmark the callback crossing and document the cost**

The design flags callback volume as a risk: a 100,000-iteration chain means 100,000 crossings into
R. Measure it rather than guessing.

```r
data <- rnorm(50, 5, 1)                                    # drawn once, OUTSIDE the closure
ll <- function(p) -0.5 * sum((data - p[1])^2)
priors <- list(distribution("Uniform", c(0, 10)))
system.time(mcmc_posterior(ll, priors, iterations = 10000, seed = 12345))
system.time(mcmc_sample(data, "Normal", iterations = 10000, seed = 12345))
```

Record both timings in the roxygen block for `mcmc_posterior` as a "Performance" section, stating
the measured ratio against the registry path so a user can decide between them. If the callback
path is more than roughly 20 times slower, profile the crossing before continuing: the likely
cause is allocating a new R vector per call rather than reusing one.

- [ ] **Step 10: Confirm an interrupt during a long chain is handled**

Start a long run and press Ctrl-C in an interactive R session, then the same in Python. Both must
return control with an interrupt condition, not hang and not segfault. If R does not respond,
add an `R_CheckUserInterrupt()` call in the glue's callback wrapper and re-test, remembering that
it must be reached through cpp11's protected frame.

Record the outcome in the header comment of `corehydror/src/callback.cpp`.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat: sample a user-written posterior with mcmc_posterior()

Upstream's MCMCSampler takes (priors, logLikelihood); the packages could
only reach it through the built-in registry. mcmc_posterior() takes a list
of distribution() priors and an R or Python log-likelihood, over seven
samplers. Gibbs follows in the next task."
```

---

### Task 5: The Gibbs proposal and HMC/NUTS gradient callbacks

**Files:**
- Modify: `core/include/corehydro/numerics/support/callback/mcmc.hpp`
- Modify: both glues, both wrappers, `fixtures/callback/mcmc.json`
- Test: `core/tests/test_callback_runner.cpp`, both package test files

**Interfaces:**
- Consumes: the RNG handle (Task 3), the mcmc group (Task 4).
- Produces: `sampler = "Gibbs"` accepting a `proposal` callback of `(parameters, rng) -> parameters`; `sampler = "HMC"` and `"NUTS"` accepting an optional `gradient` callback of `(parameters) -> vector`.

- [ ] **Step 1: Write the failing R test**

```r
test_that("Gibbs runs with a user-written proposal", {
  ll <- function(p) -0.5 * sum((c(4.9, 5.1, 5.0) - p[1])^2)
  proposal <- function(p, rng) p + (rng_uniform(rng, length(p)) - 0.5) * 0.1
  fit <- mcmc_posterior(ll, priors = list(distribution("Uniform", c(0, 10))),
                        sampler = "Gibbs", proposal = proposal,
                        iterations = 500, seed = 12345)
  expect_equal(fit$posterior_mean[[1]], 5.0, tolerance = 0.5)
})

test_that("HMC accepts an analytic gradient", {
  ll <- function(p) -0.5 * sum((c(4.9, 5.1, 5.0) - p[1])^2)
  grad <- function(p) sum(c(4.9, 5.1, 5.0) - p[1])
  fit <- mcmc_posterior(ll, priors = list(distribution("Uniform", c(0, 10))),
                        sampler = "HMC", gradient = grad, iterations = 500, seed = 12345)
  expect_equal(fit$posterior_mean[[1]], 5.0, tolerance = 0.5)
})
```

- [ ] **Step 2: Run it to verify it fails**

Expected: FAIL, unused argument `proposal`.

- [ ] **Step 3: Implement**

In `callback/mcmc.hpp`, guard the proposal with `GuardedCall<std::vector<double>, const std::vector<double>&, MersenneTwister&>` (sentinel: an empty vector, which the Gibbs arm must treat as a rejected proposal) and the gradient with a `GuardedCall` returning `linalg::Vector`. Pass the proposal through to the `Gibbs` constructor and the gradient to `HMC`/`NUTS`, leaving the gradient empty when the caller supplies none so the ported bound-aware finite-difference default applies, exactly as the constructor already does.

In both glues, create an `RngScope` around the generator before invoking the R or Python proposal, and hand the handle in as the second argument.

- [ ] **Step 4: Run the tests**

Both package tests, plus a ctest asserting that Gibbs with a seeded proposal reproduces across two identical runs.

- [ ] **Step 5: Add fixture cases, run all four runners, commit**

Gibbs becomes reachable for the first time in either package, so the status page row claiming seven of eight samplers changes in Task 8. Note that here in the commit message.

```bash
git add -A
git commit -m "feat: expose the Gibbs proposal and the HMC/NUTS gradient callbacks

Gibbs needs a model-specific conditional proposal, which is why it was the
one ported sampler neither package could reach. It now takes an R or Python
proposal receiving the core generator, completing all eight samplers."
```

---

### Task 6: bootstrap_custom over the four delegates

**Files:**
- Modify: `core/include/corehydro/numerics/support/callback/bootstrap.hpp` (replace the Task 1 stub)
- Modify: both glues, both wrappers, `NAMESPACE`, `__all__`
- Create: `fixtures/callback/bootstrap.json`
- Test: `core/tests/test_callback_runner.cpp`, both package test files

**Interfaces:**
- Consumes: the RNG handle (Task 3), `run_callback` (Task 1).
- Produces: `bootstrap/run`, returning per-statistic point estimates and confidence bounds, plus `names` labelling them.

- [ ] **Step 1: Read the bootstrap surface**

Read `core/include/corehydro/numerics/sampling/bootstrap/bootstrap.hpp` lines 81-200 and 361-420, and `bootstrap_results.hpp`. The delegates are `ResampleFn`, `FitFn`, `StatisticFn`, `JackknifeFn`, `SampleSizeFn`, and `TransformFn`; `get_confidence_intervals(BootstrapCIMethod, alpha)` produces the result. `TData` is instantiated as a `std::vector<double>` for this surface.

- [ ] **Step 2: Write the failing ctest**

A bootstrap of the mean over a fixed dataset with an iid resample, asserting the interval brackets the sample mean, that a seeded run is reproducible, and that requesting BCa without a jackknife callback raises a clear error before the first replicate rather than partway through.

- [ ] **Step 3: Implement the bootstrap group**

Wire all four callbacks through guards, validate that BCa has a jackknife callback up front, run, and return `get_confidence_intervals` for the requested method. Options grammar: `{"replicates": 1000, "seed": 12345, "alpha": 0.1, "ci_method": "Percentile|BiasCorrected|Normal|BootstrapT|BCa", "parameters": [...]}`.

- [ ] **Step 4: Add the R and Python verbs**

`bootstrap_custom(data, resample, fit, statistic, jackknife = NULL, replicates = 1000, alpha = 0.1, ci_method = "Percentile", seed = 12345, parameters = NULL)`. The `resample` callback receives `(data, parameters, rng)`; `fit` receives `(data)`; `statistic` receives `(parameters)`; `jackknife` receives `(data, index)`. Document each signature in the roxygen and numpydoc blocks, since a wrong argument order is the likeliest user error and the C++ side cannot detect it.

- [ ] **Step 5: Write the package tests**

In each package, four tests: an iid bootstrap of the mean brackets the sample mean; two seeded
runs are identical; `ci_method = "BCa"` without `jackknife` errors with a message naming
`jackknife`; and an error raised inside `statistic` reaches the caller. The R version of the
first:

```r
test_that("bootstrap_custom brackets the sample mean", {
  x <- c(4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7)
  res <- bootstrap_custom(
    data = x,
    resample = function(data, parameters, rng) {
      data[floor(rng_uniform(rng, length(data)) * length(data)) + 1L]
    },
    fit = function(data) mean(data),
    statistic = function(parameters) parameters,
    replicates = 500, seed = 12345
  )
  expect_true(res$lower[[1]] < mean(x) && mean(x) < res$upper[[1]])
})
```

- [ ] **Step 6: Add the fixture and run all four runners**

Create `fixtures/callback/bootstrap.json` with a seeded iid bootstrap whose bounds are pinned
against a real C# run driven by the emitter with the equivalent delegates. Resolve the catalog
callbacks natively in each of the four runners, then run the full command block from Task 1
Step 26.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: bootstrap a user-written statistic with bootstrap_custom()

Upstream's Bootstrap<TData> is delegate-driven (ResampleFunction,
FitFunction, StatisticFunction, JackknifeFunction); the packages reached it
only through the built-in registry. All four delegates now take an R or
Python function, over the five confidence-interval methods."
```

---

### Task 7: fit_gmm_moments over user moment conditions

**Files:**
- Modify: `core/include/corehydro/numerics/support/callback/gmm.hpp` (replace the Task 1 stub)
- Modify: both glues, both wrappers, `NAMESPACE`, `__all__`
- Create: `fixtures/callback/gmm.json`
- Test: `core/tests/test_callback_runner.cpp`, both package test files

**Interfaces:**
- Consumes: `run_callback` (Task 1).
- Produces: `gmm/fit`, returning fitted parameters, standard errors, the covariance matrix with `dims`, and the J statistic.

- [ ] **Step 1: Read the delegate constructor**

Read `core/include/corehydro/estimation/generalized_method_of_moments.hpp` around the delegate constructor (the C# line 143 overload, the one that does NOT take an `IGMMModel`) and `estimation/gmm_delegates.hpp`. The required delegate is `MomentConditionFunction`, returning a `MomentConditionResult` of a moment vector and a weighting matrix; `JacobianFunction`, `PenaltyFunction`, and `PointwiseMomentConditionFunction` are optional.

- [ ] **Step 2: Write the failing ctest**

A just-identified two-parameter problem whose GMM solution has a closed form, asserting the fitted parameters and that the J statistic is NaN, which is structurally correct for a just-identified fit and is already documented in `docs/upstream-csharp-issues.md`.

- [ ] **Step 3: Implement the gmm group**

Replace the stub `callback/gmm.hpp`. Guard the moment-condition callback with a `GuardedCall` returning `MomentConditionResult`, sentinel a default-constructed result, and pass it plus any optional callbacks to the delegate constructor. Options grammar: `{"initial": [...], "lower": [...], "upper": [...], "sample_size": 100, "number_of_moment_conditions": 2, "optimizer": "BFGS"}`. Fill `CallbackResult` with the fitted parameters, then the standard errors, then the covariance matrix with `dims = {n, n}`, setting `names` so the bindings can split the flat vector apart again.

- [ ] **Step 4: Convert the host return shape in both glues**

The callback returns two things, a moment vector and a weighting matrix, so this conversion is the part a user is most likely to get wrong. In R accept a list with elements `g` (a numeric vector) and `s` (a numeric matrix); in Python accept a tuple `(g, s)` or a dict with those keys. Name both elements in the error:

```cpp
if (!out.contains("g") || !out.contains("s"))
    throw std::runtime_error(
        "the moment condition function must return a list with elements 'g' (the moment "
        "vector) and 's' (the weighting matrix)");
```

- [ ] **Step 5: Add the R and Python verbs**

`fit_gmm_moments(moment_conditions, initial, lower = NULL, upper = NULL, sample_size, jacobian = NULL, penalty = NULL, optimizer = "BFGS")`, returning a `corehydro_fit`-shaped object so `coef()`, `confint()`, and `print()` behave exactly as they do for `fit_gmm()`. Document the return shape of `moment_conditions` in both docstrings with a worked snippet.

- [ ] **Step 6: Write the package tests**

In each package: the closed-form problem from Step 2 recovers its known solution; the J statistic is `NaN` for a just-identified fit; a callback returning the wrong shape errors with a message naming `g` and `s`; an error raised inside the callback reaches the caller.

- [ ] **Step 7: Add the fixture, run all four runners, and commit**

Pin `fixtures/callback/gmm.json` against a real C# run driven by the emitter, then run the full command block from Task 1 Step 26.

```bash
git add -A
git commit -m "feat: fit user-written moment conditions with fit_gmm_moments()"
```

---

### Task 8: Cross-language proof, docs, examples, status sweep, release

**Files:**
- Create: `fixtures/callback/callback_cross_language.json`
- Create: `site/examples/14-custom-posterior/{python.ipynb,r.qmd}`, `site/examples/15-custom-bootstrap/{python.ipynb,r.qmd}`
- Modify: `corehydror/_pkgdown.yml`, `site/_quarto.yml`, `site/status.qmd`, `CHANGELOG.md`, `corehydror/DESCRIPTION`, `corehydropy/pyproject.toml`, `.claude/CLAUDE.md`

- [ ] **Step 1: Write the cross-language fixture**

Create `fixtures/callback/callback_cross_language.json` on the model of `fixtures/toolbox/toolbox_cross_language.json`: one case name nesting a seeded `mcmc` sub-block and a seeded `bootstrap` sub-block. Both must use callbacks built from `+ - * /` only, so they are IEEE-deterministic and reproduce exactly in all four runners. The Gaussian kernel `-0.5 * sum((x - mu)^2) / sigma^2` is the log-density to use; it has no transcendental calls.

Assert exact equality (`mode: "exact"` or a 0 tolerance, matching how the existing cross-language fixtures spell it) on the posterior mean and the bootstrap bounds.

- [ ] **Step 2: Prove the claim rather than assuming it**

```bash
Rscript -e 'testthat::test_local("corehydror", filter = "fixtures")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests/test_fixtures.py -q
```
Expected: both green on the new case. If the values differ between R and Python, do NOT loosen the tolerance. Find out why: either the catalog function is not arithmetic-only in one language, or a draw is being taken outside the core. Fix the cause.

- [ ] **Step 3: Write the two worked examples**

`14-custom-posterior`: fit a model whose likelihood is not in the package, using `mcmc_posterior()` with explicit priors, showing the trace, R-hat, and the posterior summary. `15-custom-bootstrap`: bootstrap a statistic the package does not provide, using `bootstrap_custom()` with an iid resample.

Both pages end with an executable reproduction check comparing against the values in the cross-language fixture, at 1e-15 relative tolerance, following the established pattern. Both must state the honest cross-language limit in prose: parameters reproduce because the generator is in the core, but a value computed by re-evaluating the user's own function may not, and MCMC amplifies that.

Python examples are notebooks committed WITH outputs (`jupyter nbconvert --to notebook --execute --inplace`). R examples are Quarto with `freeze: auto`, and `site/_freeze/` must be re-rendered and committed.

- [ ] **Step 4: Add every new export to both documentation indexes**

New exports: `mcmc_posterior`, `bootstrap_custom`, `fit_gmm_moments`, `root_find`, `quadrature`, `derivative`, `gradient`, `hessian`, and the RNG handle verbs. Add each to `corehydror/_pkgdown.yml` and to `quartodoc.sections` in `site/_quarto.yml`. pkgdown fails the build on a missing entry, which is the check that this step happened.

Run: `make docs`
Expected: the site builds with no missing-reference error.

- [ ] **Step 5: Refresh the status page**

In `site/status.qmd`:
- Correct the two stale rows: `Distributions` (multivariate) and `Distributions.Copulas` are marked "Internal" but have been public since v0.5.0 (`copula()`, `copula_fit()`, `mvdist_*`).
- Update the `Sampling.MCMC` row: all eight samplers are now reachable, not seven.
- Update `Sampling.Bootstrap`, `Mathematics.Integration`, and the `Mathematics` row for root finding and derivatives.
- Update the "Status as of" line to v0.7.0.

- [ ] **Step 6: Changelog, version bump, and context file**

Add the 0.7.0 section to `CHANGELOG.md` in the established voice, bump `corehydror/DESCRIPTION` and `corehydropy/pyproject.toml` to 0.7.0, and add the phase paragraph to the Status section of `.claude/CLAUDE.md`, including the honest cross-language limit and any divergence found along the way.

- [ ] **Step 7: Run every gate and record the numbers**

```bash
cmake --build core/build && ctest --test-dir core/build
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy && pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py
R CMD build corehydror && R CMD check --as-cran corehydror_0.7.0.tar.gz
```

Every count must exceed its baseline; oracle failures must be 0 and skips still 11; `R CMD check` must hold at 3 NOTEs with no WARNING. Record the actual numbers in the changelog entry and the context file. Do not write a number you have not seen in the output.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: the callback surface for R and Python (v0.7.0)"
```

---

## Verification checklist

Run before opening the PR. Each line is a command plus the property it proves.

- [ ] `ctest --test-dir core/build` -- above 84/84, all passing.
- [ ] `Rscript -e 'testthat::test_local("corehydror")'` -- above 5634, 0 failures.
- [ ] `pixi run python -m pytest corehydropy/tests -q` -- above 1344, 0 failures.
- [ ] `python3 tools/verify_oracles.py` -- above 5209 reproduced, 0 failed, exactly 11 skipped.
- [ ] `R CMD check --as-cran` -- 3 NOTEs, no WARNING.
- [ ] `make docs` -- builds with no missing reference entry.
- [ ] Every new export appears in `corehydror/_pkgdown.yml` AND `site/_quarto.yml`.
- [ ] The cross-language fixture passes in all four runners with no loosened tolerance and no `oracle_skip`.
- [ ] `grep -rn "GuardedObjective" core/include` -- the guard has exactly one implementation.
- [ ] Every ported file carries a `// ported from: <path> @ <sha>` header.
- [ ] No `Co-Authored-By` trailer in any commit on the branch.
