# Numerics Toolbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Numerics utility layer (goodness of fit, correlation, autocorrelation, statistics, histogram, interpolation, regression, Sobol, stratification, joint probability, link and trend functions, and the six ported optimizers) callable from R and Python.

**Architecture:** Two new header-only runners in the C++ core. `toolbox_runner.hpp` takes a group name, a method name, bulk `double` vectors, and a JSON options blob, and returns a flat result; it is driven by four callers (cpp11 glue, pybind11 glue, the C++ fixture runner, the dotnet oracle emitter). `optimizer_runner.hpp` takes a spec plus a live `std::function` objective, so it is driven by three callers plus a C# delegate in the emitter. No C++ object crosses the language boundary; every call builds, evaluates, and drops.

**Tech Stack:** C++17 (header-only, no external deps), cpp11 for R, pybind11 for Python, nlohmann/json in the C++ test runner only, jsonlite in the R test runner only, dotnet 10 for the oracle gate.

Design document: `docs/superpowers/specs/2026-08-12-numerics-toolbox-design.md`. Read it before Task 1.

## Global Constraints

- **No external C++ dependencies in `core/`.** Do not add Eigen, fmt, or anything else. The core is self-contained by design.
- **Never use `M_PI`.** Use `corehydro::numerics::kPi` (`core/include/corehydro/numerics/tools.hpp`).
- **Never name a namespace alias `gamma` or `stat`.** Both collide with C runtime symbols.
- **Oracle values live only in `fixtures/*.json`.** Never hardcode an expected numeric value in a test file.
- **After editing any `corehydror/src/*.cpp`,** run `Rscript -e 'cpp11::cpp_register("corehydror")'` before `R CMD INSTALL corehydror`.
- **`corehydror/NAMESPACE` is maintained by hand** (it says so on line 1). Every new export needs an `export()` line added manually. roxygen does not generate it.
- **Every new export must appear in BOTH `corehydror/_pkgdown.yml` and the `quartodoc.sections` in `site/_quarto.yml`.** pkgdown errors on a reference-index entry it cannot find.
- **pytest reads fixtures materialized into site-packages by pip**, not the repo symlink. After editing a fixture, re-run `pixi run python -m pip install --force-reinstall --no-deps ./corehydropy` before the fixture count changes.
- **Commits are GPG-signed automatically.** Identity `Cam Bracken <cameron.bracken@pm.me>`. Do NOT add a `Co-Authored-By` trailer. Push only when asked.
- **Ported files carry a provenance header:** `// ported from: <upstream path> @ <sha>`. The pinned shas are Numerics `2a0357a` and RMC-BestFit `c2e6192`.
- **Branch:** `surface-numerics-toolbox`, already created from `origin/main` and carrying the design commit.

## Build and test commands

```bash
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py          # dev-only dotnet gate
```

## File Structure

**Created**

| File | Responsibility |
|---|---|
| `core/include/corehydro/numerics/support/toolbox_runner.hpp` | The one place a toolbox method is dispatched. `run_toolbox(group, method, data, options_json) -> ToolboxResult` |
| `core/include/corehydro/numerics/support/optimizer_runner.hpp` | `run_optimizer(spec_json, objective) -> OptimResult`, plus the named analytic objective registry the fixtures and emitter share |
| `core/include/corehydro/numerics/data/autocorrelation.hpp` | Port of `Numerics/Data/Statistics/Autocorrelation.cs` (Correlation / Covariance / Partial, plus the confidence interval) |
| `core/tests/test_toolbox_runner.cpp` | ctest for toolbox dispatch, option parsing, error messages |
| `core/tests/test_optimizer_runner.cpp` | ctest for optimizer dispatch and the callback abort path |
| `core/tests/test_autocorrelation.cpp` | ctest for the new port |
| `corehydror/src/toolbox.cpp` | cpp11 glue: `ch_toolbox_run_`, `ch_optim_run_` |
| `corehydror/R/toolbox.R` | R verbs for statistics, histogram, interpolation, regression, sampling utilities, links, trends |
| `corehydror/R/gof.R` | R verbs for goodness of fit |
| `corehydror/R/optim.R` | `optim_minimize()`, `optim_maximize()`, `print.corehydro_optim` |
| `corehydror/tests/testthat/test-toolbox.R`, `test-gof.R`, `test-optim.R` | behavioral tests |
| `corehydropy/src/bindings/toolbox.cpp` | pybind11 glue: `toolbox_run`, `optim_run` |
| `corehydropy/src/corehydropy/toolbox.py`, `gof.py`, `optim.py` | the Python twins |
| `corehydropy/tests/test_toolbox.py`, `test_gof.py`, `test_optim.py` | behavioral tests |
| `fixtures/toolbox/*.json` | the new `toolbox` and `optimizer` fixture kinds |
| `site/examples/12-model-evaluation/{python.ipynb,r.qmd}` | worked example pair 1 |
| `site/examples/13-custom-objective/{python.ipynb,r.qmd}` | worked example pair 2 |

**Modified**

| File | Change |
|---|---|
| `core/CMakeLists.txt` | three new test targets in `BF_TESTS` |
| `core/tests/test_fixtures.cpp` | `toolbox` and `optimizer` kinds; `dispatch_goodness_of_fit` delegates to the runner |
| `corehydror/tests/testthat/test-fixtures.R` | same two kinds; stop skipping `goodness_of_fit` |
| `corehydropy/tests/test_fixtures.py` | same two kinds |
| `tools/oracle_emitter/Program.cs` | `toolbox` and `optimizer` branches |
| `corehydror/NAMESPACE`, `corehydror/_pkgdown.yml`, `site/_quarto.yml` | new exports |
| `corehydropy/CMakeLists.txt`, `corehydropy/src/bindings/bindings.hpp`, `corehydropy/src/bindings/gev.cpp` | register the new binding module |
| `corehydropy/src/corehydropy/__init__.py` | re-export the new surface |
| `corehydror/DESCRIPTION`, `corehydror/NEWS.md`, `CHANGELOG.md`, `corehydropy/pyproject.toml`, `core/include/corehydro/version.hpp` | version 0.6.0 |
| `upstream/CLAUDE.md` | record the `Correlation` matrix-overload severance and the `Autocorrelation.cs` port |

---

### Task 1: Toolbox runner foundation, proven with correlation

**Files:**
- Create: `core/include/corehydro/numerics/support/toolbox_runner.hpp`
- Create: `core/tests/test_toolbox_runner.cpp`
- Create: `corehydror/src/toolbox.cpp`
- Create: `corehydror/R/toolbox.R`
- Create: `corehydropy/src/bindings/toolbox.cpp`
- Create: `corehydropy/src/corehydropy/toolbox.py`
- Create: `fixtures/toolbox/correlation.json`
- Modify: `core/CMakeLists.txt`, `core/tests/test_fixtures.cpp`, `corehydror/tests/testthat/test-fixtures.R`, `corehydropy/tests/test_fixtures.py`, `tools/oracle_emitter/Program.cs`, `corehydror/NAMESPACE`, `corehydropy/CMakeLists.txt`, `corehydropy/src/bindings/bindings.hpp`, `corehydropy/src/bindings/gev.cpp`, `corehydropy/src/corehydropy/__init__.py`
- Test: `core/tests/test_toolbox_runner.cpp`, `fixtures/toolbox/correlation.json`

**Interfaces:**
- Produces: `corehydro::numerics::support::ToolboxResult { std::vector<double> values; std::vector<std::string> names; std::vector<int> dims; std::string spec; }` and `ToolboxResult run_toolbox(const std::string& group, const std::string& method, const std::vector<std::vector<double>>& data, const std::string& options_json)`.
- Produces: R internal `toolbox_run(group, method, data = list(), options = list())` returning `list(values, names, dims, spec)`; Python internal `_toolbox_run(group, method, data, options)` returning a dict with the same four keys.
- Produces: the `toolbox` fixture kind, case shape `{name, data, options, assertions:[{method, index?, label?, select?, expected, mode, tol}]}`.
- Every later task adds a group arm to `run_toolbox` and R/Python verbs on top of these two helpers.

- [ ] **Step 1: Read the design and the two reference runners**

Read, in this order:
- `docs/superpowers/specs/2026-08-12-numerics-toolbox-design.md`
- `core/include/corehydro/numerics/distributions/support/dist_runner.hpp` (the pattern being followed)
- `core/include/corehydro/models/json_lite.hpp` (the only JSON API available inside the core: `at`, `contains`, `items`, `entries`, `as_double`, `as_int`, `as_string`, `as_bool`, `as_double_vector`, `value_or`)
- `core/include/corehydro/numerics/data/correlation.hpp` (the group being wired first)

- [ ] **Step 2: Write the failing ctest**

Create `core/tests/test_toolbox_runner.cpp`:

```cpp
// ctest for the shared toolbox runner: dispatch, option parsing, and error messages.
#include <cmath>
#include <string>
#include <vector>

#include "corehydro/numerics/support/toolbox_runner.hpp"
#include "test_support.hpp"

namespace tb = corehydro::numerics::support;

int main() {
    const std::vector<double> x{14.0, 8.0, 32.0, 7.0, 3.0, 15.0};
    const std::vector<double> y{10.0, 5.0, 7.0, 4.0, 3.0, 8.0};

    auto pearson = tb::run_toolbox("correlation", "pearson", {x, y}, "{}");
    CHTEST_EXPECT_EQ(pearson.values.size(), std::size_t{1});
    CHTEST_EXPECT_NEAR(pearson.values[0], 0.54502739907793, 1e-12);

    auto spearman = tb::run_toolbox("correlation", "spearman", {x, y}, "{}");
    CHTEST_EXPECT_NEAR(spearman.values[0], 0.771428571428571, 1e-12);

    auto tau = tb::run_toolbox("correlation", "kendall", {x, y}, "{}");
    CHTEST_EXPECT_NEAR(tau.values[0], 0.6, 1e-12);

    // An unknown group names the group; an unknown method names the method.
    bool threw_group = false;
    try {
        tb::run_toolbox("nope", "pearson", {x, y}, "{}");
    } catch (const std::exception& e) {
        threw_group = std::string(e.what()).find("nope") != std::string::npos;
    }
    CHTEST_EXPECT_TRUE(threw_group);

    bool threw_method = false;
    try {
        tb::run_toolbox("correlation", "nope", {x, y}, "{}");
    } catch (const std::exception& e) {
        threw_method = std::string(e.what()).find("nope") != std::string::npos;
    }
    CHTEST_EXPECT_TRUE(threw_method);

    // Too few data vectors is an error, not a crash.
    bool threw_arity = false;
    try {
        tb::run_toolbox("correlation", "pearson", {x}, "{}");
    } catch (const std::exception&) {
        threw_arity = true;
    }
    CHTEST_EXPECT_TRUE(threw_arity);

    return chtest::summary();
}
```

Check the macro names actually used in `core/tests/test_support.hpp` first and match them; if the header spells them differently (for example `CHECK_NEAR`), use its spelling throughout this file.

- [ ] **Step 3: Register the test and run it to see it fail**

Add `test_toolbox_runner` to the `BF_TESTS` list in `core/CMakeLists.txt` (alphabetical position, beside `test_dist_runner`).

Run:
```bash
cmake -S core -B core/build && cmake --build core/build 2>&1 | tail -20
```
Expected: FAIL, `fatal error: 'corehydro/numerics/support/toolbox_runner.hpp' file not found`.

- [ ] **Step 4: Write the runner**

Create `core/include/corehydro/numerics/support/toolbox_runner.hpp`:

```cpp
// corehydro ADDITION -- no upstream C# counterpart (sibling of
// distributions/support/dist_runner.hpp and estimation/support/fit_runner.hpp).
//
// The single place a Numerics utility method is dispatched in this repo. Four callers drive it
// and none owns any evaluation logic: the cpp11 glue (corehydror/src/toolbox.cpp), the pybind11
// glue (corehydropy/src/bindings/toolbox.cpp), the C++ fixture runner (core/tests/
// test_fixtures.cpp), and the dotnet oracle emitter (tools/oracle_emitter/Program.cs), which
// reads the same GRAMMAR against the real C# statics. A fixture case, an oracle replay, and a
// user's correlation() call are the same code path.
//
// Bulk data travels as native double vectors, not JSON: a goodness-of-fit call carries two
// series of arbitrary length and paying a JSON parse for them would be pointless. Scalars,
// enum names, and flags travel in `options_json`.
//
// Stateless by construction: one call builds whatever it needs, evaluates once, and drops it.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/data/correlation.hpp"

namespace corehydro::numerics::support {

using corehydro::models::spec::JsonValue;

// Flat result surface every binding and every fixture assertion reads. `values` holds whatever
// the method returns, in method order; `names` labels them when the method returns a named set;
// `dims` carries {rows, columns} when the method returns a matrix flattened row-major into
// `values`, and is empty otherwise; `spec` carries a child object back and is empty otherwise.
struct ToolboxResult {
    std::vector<double> values;
    std::vector<std::string> names;
    std::vector<int> dims;
    std::string spec;
};

namespace detail {

inline const std::vector<double>& data_at(const std::vector<std::vector<double>>& data,
                                          std::size_t i, const std::string& group,
                                          const std::string& method) {
    if (i >= data.size())
        throw std::runtime_error("toolbox method '" + group + "." + method + "' needs " +
                                 std::to_string(i + 1) + " data vector(s), got " +
                                 std::to_string(data.size()));
    return data[i];
}

inline ToolboxResult scalar(double v) {
    ToolboxResult r;
    r.values = {v};
    return r;
}

// --- group arms -------------------------------------------------------------------------
// One function per group. Later tasks add arms beside this one; nothing else in the file
// changes when a group is added except the dispatch table at the bottom.

inline ToolboxResult run_correlation(const std::string& method,
                                     const std::vector<std::vector<double>>& data,
                                     const JsonValue& options) {
    (void)options;
    const std::vector<double>& x = data_at(data, 0, "correlation", method);
    const std::vector<double>& y = data_at(data, 1, "correlation", method);
    if (method == "pearson") return scalar(numerics::data::pearson(x, y));
    if (method == "spearman") return scalar(numerics::data::spearman(x, y));
    if (method == "kendall") return scalar(numerics::data::kendalls_tau(x, y));
    throw std::runtime_error("unknown correlation method: " + method);
}

}  // namespace detail

inline ToolboxResult run_toolbox(const std::string& group, const std::string& method,
                                 const std::vector<std::vector<double>>& data,
                                 const std::string& options_json) {
    JsonValue options = models::spec::parse_json(options_json.empty() ? "{}" : options_json);
    if (group == "correlation") return detail::run_correlation(method, data, options);
    throw std::runtime_error("unknown toolbox group: " + group);
}

}  // namespace corehydro::numerics::support
```

Confirm the correlation function names and namespace against `core/include/corehydro/numerics/data/correlation.hpp` before compiling: the header declares them `inline` in `corehydro::numerics::data`.

- [ ] **Step 5: Run the ctest to verify it passes**

```bash
cmake --build core/build && ctest --test-dir core/build -R test_toolbox_runner --output-on-failure
```
Expected: PASS.

- [ ] **Step 6: Commit the runner**

```bash
git add core/include/corehydro/numerics/support/toolbox_runner.hpp core/tests/test_toolbox_runner.cpp core/CMakeLists.txt
git commit -m "feat(core): add the shared toolbox runner with the correlation group"
```

- [ ] **Step 7: Write the R glue**

Create `corehydror/src/toolbox.cpp`:

```cpp
// cpp11 glue for the shared toolbox runner: the entry point behind every verb in R/toolbox.R,
// R/gof.R and R/optim.R. Takes a group, a method, a list of numeric data vectors, and a JSON
// options object (assembled R-side by to_spec_json()), and returns the flat ToolboxResult as a
// named list. Mirrors corehydropy/src/bindings/toolbox.cpp exactly.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <string>
#include <vector>

#include "corehydro/numerics/support/toolbox_runner.hpp"

using namespace cpp11;
namespace tb = corehydro::numerics::support;

static list pack(const tb::ToolboxResult& r) {
    writable::doubles values(static_cast<R_xlen_t>(r.values.size()));
    for (std::size_t i = 0; i < r.values.size(); ++i)
        values[static_cast<R_xlen_t>(i)] = r.values[i];
    writable::strings names(static_cast<R_xlen_t>(r.names.size()));
    for (std::size_t i = 0; i < r.names.size(); ++i)
        names[static_cast<R_xlen_t>(i)] = r.names[i];
    writable::integers dims(static_cast<R_xlen_t>(r.dims.size()));
    for (std::size_t i = 0; i < r.dims.size(); ++i)
        dims[static_cast<R_xlen_t>(i)] = r.dims[i];
    return writable::list({"values"_nm = values, "names"_nm = names, "dims"_nm = dims,
                           "spec"_nm = writable::strings({r.spec})});
}

[[cpp11::register]]
list ch_toolbox_run_(std::string group, std::string method, list data, std::string options_json) {
    std::vector<std::vector<double>> vecs;
    vecs.reserve(static_cast<std::size_t>(data.size()));
    for (R_xlen_t i = 0; i < data.size(); ++i) {
        doubles col(data[i]);
        vecs.emplace_back(col.begin(), col.end());
    }
    return pack(tb::run_toolbox(group, method, vecs, options_json));
}
```

- [ ] **Step 8: Write the R wrapper and the correlation verb**

Create `corehydror/R/toolbox.R`:

```r
# The Numerics toolbox surface. Every verb serializes its options to the toolbox_runner.hpp
# grammar and runs one method through ch_toolbox_run_; bulk data goes across as numeric vectors,
# not JSON. Nothing holds C++ state.

# Internal: one call into the shared runner. `data` is a list of numeric vectors, `options` a
# named list serialized with the same emitter every other spec uses.
toolbox_run <- function(group, method, data = list(), options = list()) {
  opts <- if (length(options) == 0L) "{}" else to_spec_json(options)
  ch_toolbox_run_(group, method, lapply(data, as.double), opts)
}

# Internal: reject the two mistakes every paired-series verb can make, naming the argument.
check_pair <- function(x, y, x_name = "x", y_name = "y") {
  if (!is.numeric(x) || !is.numeric(y)) {
    stop(sprintf("`%s` and `%s` must be numeric vectors", x_name, y_name), call. = FALSE)
  }
  if (length(x) != length(y)) {
    stop(sprintf("`%s` and `%s` must have the same length; got %d and %d",
                 x_name, y_name, length(x), length(y)), call. = FALSE)
  }
  if (length(x) < 2L) {
    stop(sprintf("`%s` and `%s` must have at least two elements", x_name, y_name), call. = FALSE)
  }
  invisible(NULL)
}

#' Correlation between two samples
#'
#' Mirrors the C# `Correlation` class of the Numerics library. Upstream's matrix overloads
#' (`Pearson(double[,])`, `Spearman(double[,])`) are not ported, so only the paired-vector forms
#' are available here.
#'
#' @param x,y numeric vectors of equal length, at least two elements.
#' @param method one of `"pearson"` (the default), `"spearman"`, or `"kendall"`.
#' @return a single numeric correlation coefficient.
#' @examples
#' x <- c(14, 8, 32, 7, 3, 15)
#' y <- c(10, 5, 7, 4, 3, 8)
#' correlation(x, y)
#' correlation(x, y, method = "kendall")
#' @export
correlation <- function(x, y, method = c("pearson", "spearman", "kendall")) {
  method <- match.arg(method)
  check_pair(x, y)
  toolbox_run("correlation", method, list(x, y))$values[[1]]
}
```

- [ ] **Step 9: Export it and install**

Add to `corehydror/NAMESPACE`, after the `export(latin_hypercube)` line:
```
export(correlation)
```

Run:
```bash
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL corehydror
Rscript -e 'library(corehydror); print(correlation(c(14,8,32,7,3,15), c(10,5,7,4,3,8)))'
```
Expected: `[1] 0.5450274`.

- [ ] **Step 10: Write the Python glue**

Create `corehydropy/src/bindings/toolbox.cpp`:

```cpp
// pybind11 glue for the shared toolbox runner: the entry point behind every verb in
// corehydropy.toolbox, corehydropy.gof and corehydropy.optim. Mirrors corehydror's
// src/toolbox.cpp exactly, one entry point per language.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "corehydro/numerics/support/toolbox_runner.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace tb = corehydro::numerics::support;

static py::dict pack(const tb::ToolboxResult& r) {
    py::dict out;
    out["values"] = r.values;
    out["names"] = r.names;
    out["dims"] = r.dims;
    out["spec"] = r.spec;
    return out;
}

void register_toolbox(py::module_& m) {
    m.def(
        "toolbox_run",
        [](const std::string& group, const std::string& method,
           const std::vector<std::vector<double>>& data, const std::string& options_json) {
            return pack(tb::run_toolbox(group, method, data, options_json));
        },
        py::arg("group"), py::arg("method"), py::arg("data"), py::arg("options_json"));
}
```

Add `void register_toolbox(pybind11::module_& m);` to `corehydropy/src/bindings/bindings.hpp`, add `register_toolbox(m);` beside `register_dist_spec(m);` in `corehydropy/src/bindings/gev.cpp`, and add `src/bindings/toolbox.cpp` to the `pybind11_add_module` source list in `corehydropy/CMakeLists.txt`.

- [ ] **Step 11: Write the Python wrapper**

Create `corehydropy/src/corehydropy/toolbox.py`:

```python
"""The Numerics toolbox surface. Every verb serializes its options to the
``toolbox_runner.hpp`` grammar and runs one method through ``_core.toolbox_run``; bulk data goes
across as numeric vectors, not JSON. Mirrors ``corehydror``'s ``R/toolbox.R`` verb for verb.
"""

from __future__ import annotations

import json

import numpy as np

from . import _core

__all__ = ["correlation"]


def _toolbox_run(group: str, method: str, data=None, options=None) -> dict:
    """Internal: one call into the shared runner."""
    vectors = [np.asarray(d, dtype=float).ravel().tolist() for d in (data or [])]
    return _core.toolbox_run(group, method, vectors, json.dumps(options or {}))


def _check_pair(x, y, x_name: str = "x", y_name: str = "y"):
    """Internal: reject the two mistakes every paired-series verb can make, naming the argument."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size:
        raise ValueError(
            f"`{x_name}` and `{y_name}` must have the same length; got {xa.size} and {ya.size}"
        )
    if xa.size < 2:
        raise ValueError(f"`{x_name}` and `{y_name}` must have at least two elements")
    return xa, ya


def correlation(x, y, method: str = "pearson") -> float:
    """Correlation between two samples.

    Mirrors the C# ``Correlation`` class of the Numerics library. Upstream's matrix overloads
    are not ported, so only the paired-vector forms are available here.

    Parameters
    ----------
    x, y : array_like
        Numeric vectors of equal length, at least two elements.
    method : {"pearson", "spearman", "kendall"}
        Which coefficient to compute.

    Returns
    -------
    float

    Examples
    --------
    >>> from corehydropy import correlation
    >>> round(correlation([14, 8, 32, 7, 3, 15], [10, 5, 7, 4, 3, 8]), 6)
    0.545027
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"`method` must be one of 'pearson', 'spearman', 'kendall'; got {method!r}")
    xa, ya = _check_pair(x, y)
    return float(_toolbox_run("correlation", method, [xa, ya])["values"][0])
```

Add `from .toolbox import correlation` to `corehydropy/src/corehydropy/__init__.py` and add `"correlation"` to its `__all__`.

Run:
```bash
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -c "from corehydropy import correlation; print(correlation([14,8,32,7,3,15],[10,5,7,4,3,8]))"
```
Expected: `0.5450273990779301` (or the same value to 15 digits).

- [ ] **Step 12: Commit the glue**

```bash
git add corehydror/src corehydror/R/toolbox.R corehydror/NAMESPACE corehydropy/src corehydropy/CMakeLists.txt
git commit -m "feat: expose correlation in R and Python over the toolbox runner"
```

- [ ] **Step 13: Write the fixture**

Create `fixtures/toolbox/correlation.json`. Values are the literals in
`upstream/Numerics/Test_Numerics/Data/Statistics/Test_Correlation.cs`:

```json
{
 "kind": "toolbox",
 "group": "correlation",
 "source": "Numerics/Data/Statistics/Correlation.cs",
 "reference": "literals transcribed from Test_Numerics/Data/Statistics/Test_Correlation.cs (Test_Pearson, Test_Pearson_Big, Test_Spearman, Test_Spearman_Big, Test_KendallsTau, Test_KendallsTau_Big)",
 "datasets": {
  "small_x": [14, 8, 32, 7, 3, 15],
  "small_y": [10, 5, 7, 4, 3, 8],
  "big_x": [230408, 288010, 345611, 403213, 460815, 518417, 576019, 633612, 691223, 748825, 806427, 864029, 921631, 1036834, 1152038],
  "big_y": [1519.7, 1520.5, 1520.9, 1521.7, 1523.5, 1525.9, 1528.4, 1530.9, 1533.2, 1534.7, 1535.9, 1538, 1541.3, 1547.7, 1552.7]
 },
 "cases": [
  {
   "name": "small_sample",
   "data": ["small_x", "small_y"],
   "assertions": [
    {"method": "pearson", "expected": 0.54502739907793, "mode": "abs", "tol": 1e-10},
    {"method": "spearman", "expected": 0.771428571428571, "mode": "abs", "tol": 1e-10},
    {"method": "kendall", "expected": 0.6, "mode": "abs", "tol": 1e-10}
   ]
  },
  {
   "name": "monotone_sample",
   "data": ["big_x", "big_y"],
   "assertions": [
    {"method": "pearson", "expected": 0.988054377242161, "mode": "abs", "tol": 1e-10},
    {"method": "spearman", "expected": 1, "mode": "abs", "tol": 1e-10},
    {"method": "kendall", "expected": 1, "mode": "abs", "tol": 1e-10}
   ]
  }
 ]
}
```

Document the new kind in `fixtures/README.md` beside the existing kinds: `toolbox` cases carry
`data` (a list of dataset names or inline arrays), an optional `options` object, and assertions
carrying `method`, an optional `index` (default 0), an optional `label` (select from `names`
instead of by index), and an optional `select` of `"value"` (default), `"length"`, `"rows"`, or
`"columns"`.

- [ ] **Step 14: Wire the kind into the C++ fixture runner**

In `core/tests/test_fixtures.cpp`, add after the `run_data_utility` function:

```cpp
// --- toolbox path -----------------------------------------------------------------------
// Every Numerics utility group (correlation, goodness of fit, statistics, interpolation, ...)
// runs through numerics/support/toolbox_runner.hpp. A case carries its data vectors and an
// options object; each assertion names a method and selects one value out of the result.

static std::vector<std::vector<double>> toolbox_data(const json& c, const json& datasets) {
    std::vector<std::vector<double>> out;
    if (!c.contains("data")) return out;
    for (const auto& d : c["data"]) {
        std::vector<double> v;
        if (d.is_string()) {
            for (const auto& e : datasets[d.get<std::string>()]) v.push_back(parse_num(e));
        } else {
            for (const auto& e : d) v.push_back(parse_num(e));
        }
        out.push_back(std::move(v));
    }
    return out;
}

static double toolbox_select(const corehydro::numerics::support::ToolboxResult& r, const json& as) {
    std::string select = as.value("select", std::string("value"));
    if (select == "length") return static_cast<double>(r.values.size());
    if (select == "rows") return r.dims.empty() ? -1.0 : static_cast<double>(r.dims[0]);
    if (select == "columns") return r.dims.size() < 2 ? -1.0 : static_cast<double>(r.dims[1]);
    std::size_t i = 0;
    if (as.contains("label")) {
        std::string label = as["label"].get<std::string>();
        bool found = false;
        for (std::size_t k = 0; k < r.names.size(); ++k)
            if (r.names[k] == label) { i = k; found = true; break; }
        if (!found) throw std::runtime_error("toolbox result has no label '" + label + "'");
    } else {
        i = static_cast<std::size_t>(as.value("index", 0));
    }
    if (i >= r.values.size())
        throw std::runtime_error("toolbox result index out of range");
    return r.values[i];
}

static void run_toolbox_kind(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    std::string group = spec["group"].get<std::string>();
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        auto data = toolbox_data(c, datasets);
        std::string options = c.contains("options") ? c["options"].dump() : "{}";
        for (const auto& as : c["assertions"]) {
            std::string where = "toolbox/" + group + "/" + name;
            auto r = corehydro::numerics::support::run_toolbox(
                group, as["method"].get<std::string>(), data, options);
            check_value(toolbox_select(r, as), as, where);
        }
    }
}
```

Add `#include "corehydro/numerics/support/toolbox_runner.hpp"` beside the other core includes at
the top, and add the kind to the dispatch near line 3185:

```cpp
        } else if (kind == "toolbox") {
            run_toolbox_kind(spec);
```

Run:
```bash
cmake --build core/build && ctest --test-dir core/build -R test_fixtures --output-on-failure 2>&1 | tail -5
```
Expected: PASS with 6 more checks than before.

- [ ] **Step 15: Wire the kind into the R fixture runner**

In `corehydror/tests/testthat/test-fixtures.R`, add beside `dispatch_data_utility`:

```r
# toolbox [group, data, options; assertions carry method/index/label/select]: every Numerics
# utility group runs through ch_toolbox_run_. Mirrors run_toolbox_kind in
# core/tests/test_fixtures.cpp.
toolbox_case_data <- function(case, datasets) {
  if (is.null(case$data)) {
    return(list())
  }
  lapply(case$data, function(d) {
    if (is.character(d)) as.double(unlist(datasets[[d]])) else as.double(unlist(d))
  })
}

toolbox_select <- function(r, a) {
  select <- if (is.null(a$select)) "value" else a$select
  if (identical(select, "length")) {
    return(as.double(length(r$values)))
  }
  if (identical(select, "rows")) {
    return(as.double(r$dims[[1]]))
  }
  if (identical(select, "columns")) {
    return(as.double(r$dims[[2]]))
  }
  i <- if (!is.null(a$label)) match(a$label, r$names) else (if (is.null(a$index)) 0L else as.integer(a$index)) + 1L
  if (is.na(i) || i > length(r$values)) {
    stop("toolbox result selection out of range")
  }
  as.double(r$values[[i]])
}
```

and the kind, beside the `data_utility` branch:

```r
    if (identical(spec$kind, "toolbox")) {
      ns <- asNamespace("corehydror")
      datasets <- spec$datasets
      for (case in spec$cases) {
        data <- toolbox_case_data(case, datasets)
        opts <- if (is.null(case$options)) "{}" else ns$to_spec_json(case$options)
        for (a in case$assertions) {
          r <- ns$ch_toolbox_run_(spec$group, a$method, data, opts)
          check_assertion(toolbox_select(r, a), a)
        }
      }
      next
    }
```

Run:
```bash
Rscript -e 'testthat::test_local("corehydror")' 2>&1 | tail -5
```
Expected: PASS, 6 more assertions than the previous run.

- [ ] **Step 16: Wire the kind into the Python fixture runner**

In `corehydropy/tests/test_fixtures.py`, add `"toolbox"` to the kind tuple in `_load_cases`, and
add the branch in `test_fixture_case`:

```python
    if kind == "toolbox":
        _run_toolbox_case(target, case, datasets)
        return
```

with, beside `_dispatch_data_utility`:

```python
# toolbox [group, data, options; assertions carry method/index/label/select]: every Numerics
# utility group runs through _core.toolbox_run. Mirrors run_toolbox_kind in
# core/tests/test_fixtures.cpp.
def _toolbox_case_data(case, datasets):
    out = []
    for d in case.get("data", []):
        src = datasets[d] if isinstance(d, str) else d
        out.append([_num(v) for v in src])
    return out


def _toolbox_select(r, a):
    select = a.get("select", "value")
    if select == "length":
        return float(len(r["values"]))
    if select == "rows":
        return float(r["dims"][0])
    if select == "columns":
        return float(r["dims"][1])
    if "label" in a:
        i = list(r["names"]).index(a["label"])
    else:
        i = int(a.get("index", 0))
    return float(r["values"][i])


def _run_toolbox_case(group, case, datasets):
    data = _toolbox_case_data(case, datasets)
    options = json.dumps(case.get("options", {}))
    for a in case["assertions"]:
        r = _core.toolbox_run(group, a["method"], data, options)
        _check(_toolbox_select(r, a), a)
```

`_load_cases` passes `spec.get("target", kind)` as `target`; for a toolbox spec there is no
`target`, so pass the group instead by adding `spec.get("group", "")` handling: in `_load_cases`,
where the tuple is built, use `spec.get("target", spec.get("group", kind))`.

Run:
```bash
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q 2>&1 | tail -3
```
Expected: PASS, 6 more tests than the previous run.

- [ ] **Step 17: Wire the kind into the oracle emitter**

In `tools/oracle_emitter/Program.cs`, add a branch beside the `goodness_of_fit` one:

```csharp
    // --- toolbox branch ------------------------------------------------------------------
    // Every Numerics utility group. Mirrors the GRAMMAR of numerics/support/toolbox_runner.hpp
    // against the real C# statics; the dispatch below is this file's own transcription, so an
    // oracle never runs the code under test.
    if (kindStr == "toolbox")
    {
        var tbSets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var tbDs))
            foreach (var kv in tbDs.EnumerateObject())
                tbSets[kv.Name] = kv.Value.EnumerateArray().Select(ParseNum).ToArray();

        string group = root.GetProperty("group").GetString()!;
        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            var data = new List<double[]>();
            if (c.TryGetProperty("data", out var dataNode))
                foreach (var d in dataNode.EnumerateArray())
                    data.Add(d.ValueKind == JsonValueKind.String
                        ? tbSets[d.GetString()!]
                        : d.EnumerateArray().Select(ParseNum).ToArray());
            JsonElement opts = c.TryGetProperty("options", out var o) ? o : default;

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                string where = $"toolbox/{group}/{caseName}/{method}";
                double actual;
                try { actual = ToolboxDispatch(group, method, data, opts, asrt); }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); continue; }
                if (Compare(actual, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
            }
        }
        continue;
    }
```

and the dispatch helper, near the other static helpers at the bottom of the file:

```csharp
// One arm per toolbox group. Later tasks extend this switch; the shape never changes.
static double ToolboxDispatch(string group, string method, List<double[]> data,
                              JsonElement options, JsonElement assertion)
{
    switch (group)
    {
        case "correlation":
            return method switch
            {
                "pearson"  => Correlation.Pearson(data[0], data[1]),
                "spearman" => Correlation.Spearman(data[0], data[1]),
                "kendall"  => Correlation.KendallsTau(data[0], data[1]),
                _ => throw new Exception($"unknown correlation method: {method}")
            };
        default:
            throw new Exception($"unknown toolbox group: {group}");
    }
}
```

Add `using Numerics.Data.Statistics;` at the top of `Program.cs` if it is not already there, and
make sure `Numerics/Data/Statistics/Correlation.cs` is in the emitter's subset-compile file list
(see how `GoodnessOfFit.cs` is listed in `tools/oracle_emitter/oracle_emitter.csproj`).

Run:
```bash
python3 tools/verify_oracles.py 2>&1 | tail -5
```
Expected: `reproduced` count up by 6, `failed 0`.

- [ ] **Step 18: Commit the fixture and the four runners**

```bash
git add fixtures core/tests corehydror/tests corehydropy/tests tools/oracle_emitter fixtures/README.md
git commit -m "test: add the toolbox fixture kind across all four runners"
```

---

### Task 2: Goodness of fit

**Files:**
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (add the `gof` arm)
- Create: `corehydror/R/gof.R`, `corehydropy/src/corehydropy/gof.py`
- Create: `fixtures/toolbox/goodness_of_fit_metrics.json`
- Modify: `core/tests/test_fixtures.cpp`, `corehydror/tests/testthat/test-fixtures.R`, `corehydropy/tests/test_fixtures.py` (route the existing `goodness_of_fit` kind through the runner and stop skipping it in R and Python)
- Modify: `tools/oracle_emitter/Program.cs` (the `gof` arm of `ToolboxDispatch`)
- Modify: `corehydror/NAMESPACE`, `corehydropy/src/corehydropy/__init__.py`
- Test: `corehydror/tests/testthat/test-gof.R`, `corehydropy/tests/test_gof.py`

**Interfaces:**
- Consumes: `run_toolbox`, `toolbox_run` (R), `_toolbox_run` (Python) from Task 1.
- Produces: group `"gof"` with methods `metrics` (all continuous metrics as a named set), each metric name individually (`rmse`, `mse`, `mae`, `mape`, `smape`, `nse`, `log_nse`, `kge`, `kge_mod`, `pbias`, `rsr`, `pearson`, `r_squared`, `d`, `d_mod`, `d_ref`, `ve`), `classification` (named set), `aic`, `aicc`, `bic`, `aic_weights`, `rmse_weights`, `ks`, `ad`, `chi_squared`, `rmse_dist`.
- Produces: R `goodness_of_fit()`, `classification_metrics()`, `gof_test()`, `gof_rmse()`, `aic()`, `aicc()`, `bic()`, `aic_weights()`, `rmse_weights()`; the same names in Python.

- [ ] **Step 1: Read the source of truth**

Read `core/include/corehydro/numerics/data/goodness_of_fit.hpp` end to end and write down the
exact C++ name of each static. The R metric names in this task map to them as:
`rmse`→`rmse(observed, modeled, k)`, `mse`→`mse`, `mae`→`mae`, `mape`→`mape`, `smape`→`smape`,
`nse`→`nash_sutcliffe_efficiency`, `log_nse`→`log_nash_sutcliffe_efficiency`,
`kge`→`kling_gupta_efficiency`, `kge_mod`→`kling_gupta_efficiency_mod`, `pbias`→`pbias`,
`rsr`→`rsr`, `pearson`→`pearson`, `r_squared`→`r_squared`, `d`→`index_of_agreement`,
`d_mod`→`modified_index_of_agreement`, `d_ref`→`refined_index_of_agreement`,
`ve`→`volumetric_efficiency`.

- [ ] **Step 2: Write the failing ctest additions**

Append to `core/tests/test_toolbox_runner.cpp`, before `return chtest::summary();`:

```cpp
    // gof: the named set and the individual metric agree, and the set is labelled.
    const std::vector<double> obs{2.0, 4.0, 6.0, 8.0, 10.0};
    const std::vector<double> mod{2.2, 3.9, 6.4, 7.5, 10.1};
    auto set = tb::run_toolbox("gof", "metrics", {obs, mod}, "{}");
    CHTEST_EXPECT_EQ(set.values.size(), set.names.size());
    CHTEST_EXPECT_EQ(set.values.size(), std::size_t{17});
    auto one = tb::run_toolbox("gof", "nse", {obs, mod}, "{}");
    std::size_t nse_at = 0;
    for (std::size_t i = 0; i < set.names.size(); ++i)
        if (set.names[i] == "nse") nse_at = i;
    CHTEST_EXPECT_NEAR(set.values[nse_at], one.values[0], 0.0);

    // aic takes its arguments from options, not from a data vector.
    auto aic = tb::run_toolbox("gof", "aic", {}, "{\"k\":2,\"log_likelihood\":-121.01131220612}");
    CHTEST_EXPECT_NEAR(aic.values[0], 246.02262441224, 1e-9);

    // The distribution-backed tests build their model from a dist spec in the options.
    auto ks = tb::run_toolbox("gof", "ks", {obs},
                              "{\"model\":{\"family\":\"Normal\",\"parameters\":[6.0,3.0]}}");
    CHTEST_EXPECT_TRUE(ks.values[0] > 0.0 && ks.values[0] < 1.0);
```

Run: `cmake --build core/build && ctest --test-dir core/build -R test_toolbox_runner --output-on-failure`
Expected: FAIL, `unknown toolbox group: gof`.

- [ ] **Step 3: Add the gof arm to the runner**

In `toolbox_runner.hpp`, add the includes:

```cpp
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/distributions/support/dist_spec.hpp"
```

and the arm inside `namespace detail`:

```cpp
// Builds the model a distribution-backed goodness-of-fit method needs from the options blob:
// {"model": {"family": ..., "parameters": [...]}} in the dist_spec.hpp grammar, so a truncated
// or mixture model works here exactly as it does in dist_pdf().
inline std::unique_ptr<distributions::UnivariateDistributionBase> options_model(
    const JsonValue& options, const std::string& method) {
    if (!options.contains("model"))
        throw std::runtime_error("toolbox method 'gof." + method +
                                 "' needs a 'model' distribution spec in its options");
    return distributions::support::build_univariate(options.at("model"));
}

inline ToolboxResult run_gof(const std::string& method,
                             const std::vector<std::vector<double>>& data,
                             const JsonValue& options) {
    using GOF = numerics::data::GoodnessOfFit;

    // Information criteria: scalars only, all from options.
    if (method == "aic")
        return scalar(GOF::aic(options.at("k").as_int(), options.at("log_likelihood").as_double()));
    if (method == "aicc")
        return scalar(GOF::aicc(options.at("n").as_int(), options.at("k").as_int(),
                                options.at("log_likelihood").as_double()));
    if (method == "bic")
        return scalar(GOF::bic(options.at("n").as_int(), options.at("k").as_int(),
                               options.at("log_likelihood").as_double()));
    if (method == "aic_weights") {
        ToolboxResult r;
        r.values = GOF::aic_weights(data_at(data, 0, "gof", method));
        return r;
    }
    if (method == "rmse_weights") {
        ToolboxResult r;
        r.values = GOF::rmse_weights(data_at(data, 0, "gof", method));
        return r;
    }

    // Distribution-backed methods: one series plus a model spec.
    if (method == "ks" || method == "ad" || method == "chi_squared" || method == "rmse_dist") {
        std::vector<double> obs = data_at(data, 0, "gof", method);
        std::sort(obs.begin(), obs.end());  // every one of these requires ascending input
        std::unique_ptr<distributions::UnivariateDistributionBase> m = options_model(options, method);
        if (method == "ks") return scalar(GOF::kolmogorov_smirnov(obs, *m));
        if (method == "ad") return scalar(GOF::anderson_darling(obs, *m));
        if (method == "chi_squared") return scalar(GOF::chi_squared(obs, *m));
        if (data.size() > 1) return scalar(GOF::rmse(obs, data[1], *m));
        return scalar(GOF::rmse(obs, *m));
    }

    // Classification metrics: two series plus a threshold.
    if (method == "classification") {
        const std::vector<double>& o = data_at(data, 0, "gof", method);
        const std::vector<double>& p = data_at(data, 1, "gof", method);
        double t = options.at("threshold").as_double();
        GOF::ConfusionMatrix cm = GOF::confusion_matrix(o, p, t);
        ToolboxResult r;
        r.values = {GOF::accuracy(o, p, t),    GOF::precision(o, p, t),
                    GOF::recall(o, p, t),      GOF::f1_score(o, p, t),
                    GOF::specificity(o, p, t), GOF::balanced_accuracy(o, p, t),
                    static_cast<double>(cm.tp), static_cast<double>(cm.tn),
                    static_cast<double>(cm.fp), static_cast<double>(cm.fn)};
        r.names = {"accuracy", "precision", "recall", "f1", "specificity",
                   "balanced_accuracy", "tp", "tn", "fp", "fn"};
        return r;
    }

    // Continuous metrics: two series, optional k for the RMSE denominator.
    const std::vector<double>& o = data_at(data, 0, "gof", method);
    const std::vector<double>& m = data_at(data, 1, "gof", method);
    int k = options.value_or("k", 0);
    const std::vector<std::pair<const char*, double>> all = {
        {"rmse", GOF::rmse(o, m, k)},
        {"mse", GOF::mse(o, m)},
        {"mae", GOF::mae(o, m)},
        {"mape", GOF::mape(o, m)},
        {"smape", GOF::smape(o, m)},
        {"nse", GOF::nash_sutcliffe_efficiency(o, m)},
        {"log_nse", GOF::log_nash_sutcliffe_efficiency(o, m)},
        {"kge", GOF::kling_gupta_efficiency(o, m)},
        {"kge_mod", GOF::kling_gupta_efficiency_mod(o, m)},
        {"pbias", GOF::pbias(o, m)},
        {"rsr", GOF::rsr(o, m)},
        {"pearson", GOF::pearson(o, m)},
        {"r_squared", GOF::r_squared(o, m)},
        {"d", GOF::index_of_agreement(o, m)},
        {"d_mod", GOF::modified_index_of_agreement(o, m)},
        {"d_ref", GOF::refined_index_of_agreement(o, m)},
        {"ve", GOF::volumetric_efficiency(o, m)},
    };
    if (method == "metrics") {
        ToolboxResult r;
        for (const auto& kv : all) {
            r.values.push_back(kv.second);
            r.names.push_back(kv.first);
        }
        return r;
    }
    for (const auto& kv : all)
        if (method == kv.first) return scalar(kv.second);
    throw std::runtime_error("unknown gof method: " + method);
}
```

Add `#include <algorithm>`, `#include <memory>`, and `#include <utility>` to the header's include
block, and add the dispatch line in `run_toolbox`:

```cpp
    if (group == "gof") return detail::run_gof(method, data, options);
```

Note the `all` vector evaluates every metric even when one is asked for. That is deliberate: the
series are small, the cost is a few passes, and one code path means the set and the single value
can never disagree. If a metric ever throws on a valid input, split it out then, not now.

- [ ] **Step 4: Run the ctest**

```bash
cmake --build core/build && ctest --test-dir core/build -R test_toolbox_runner --output-on-failure
```
Expected: PASS.

- [ ] **Step 5: Commit the runner arm**

```bash
git add core/include/corehydro/numerics/support/toolbox_runner.hpp core/tests/test_toolbox_runner.cpp
git commit -m "feat(core): add the goodness-of-fit group to the toolbox runner"
```

- [ ] **Step 6: Write the R verbs**

Create `corehydror/R/gof.R`:

```r
# The goodness-of-fit surface. Every verb runs one method of the "gof" group through the shared
# toolbox runner; nothing holds C++ state.

kGofMetrics <- c("rmse", "mse", "mae", "mape", "smape", "nse", "log_nse", "kge", "kge_mod",
                 "pbias", "rsr", "pearson", "r_squared", "d", "d_mod", "d_ref", "ve")

#' Goodness-of-fit metrics for a modeled series
#'
#' Mirrors the continuous metrics of the C# `GoodnessOfFit` class of the Numerics library.
#'
#' @param observed,modeled numeric vectors of equal length.
#' @param metrics `"all"` (the default) for every metric as a named vector, or a character vector
#'   of metric names drawn from `rmse`, `mse`, `mae`, `mape`, `smape`, `nse`, `log_nse`, `kge`,
#'   `kge_mod`, `pbias`, `rsr`, `pearson`, `r_squared`, `d`, `d_mod`, `d_ref`, `ve`.
#' @param k degrees-of-freedom correction subtracted from the sample size in the RMSE
#'   denominator. Default 0.
#' @return a named numeric vector.
#' @examples
#' obs <- c(2, 4, 6, 8, 10)
#' mod <- c(2.2, 3.9, 6.4, 7.5, 10.1)
#' goodness_of_fit(obs, mod)
#' goodness_of_fit(obs, mod, metrics = c("nse", "kge"))
#' @export
goodness_of_fit <- function(observed, modeled, metrics = "all", k = 0) {
  check_pair(observed, modeled, "observed", "modeled")
  if (!identical(metrics, "all")) {
    unknown <- setdiff(metrics, kGofMetrics)
    if (length(unknown) > 0L) {
      stop(sprintf("unknown metric(s): %s. Available: %s",
                   paste(unknown, collapse = ", "), paste(kGofMetrics, collapse = ", ")),
           call. = FALSE)
    }
  }
  r <- toolbox_run("gof", "metrics", list(observed, modeled), list(k = k))
  out <- stats::setNames(r$values, r$names)
  if (identical(metrics, "all")) out else out[metrics]
}

#' Classification metrics for a thresholded series
#'
#' Mirrors the classification region of the C# `GoodnessOfFit` class: a value at or above
#' `threshold` counts as a positive.
#'
#' @param observed,modeled numeric vectors of equal length.
#' @param threshold the value separating a positive from a negative.
#' @return a named numeric vector of accuracy, precision, recall, f1, specificity,
#'   balanced_accuracy, and the four confusion-matrix counts tp, tn, fp, fn.
#' @examples
#' obs <- c(1, 5, 2, 9, 4)
#' mod <- c(2, 6, 1, 8, 3)
#' classification_metrics(obs, mod, threshold = 4)
#' @export
classification_metrics <- function(observed, modeled, threshold) {
  check_pair(observed, modeled, "observed", "modeled")
  if (!is.numeric(threshold) || length(threshold) != 1L) {
    stop("`threshold` must be a single number", call. = FALSE)
  }
  r <- toolbox_run("gof", "classification", list(observed, modeled), list(threshold = threshold))
  stats::setNames(r$values, r$names)
}

#' Goodness-of-fit test statistic for a fitted distribution
#'
#' @param x numeric vector of observations. Sorted internally, as the C# methods require.
#' @param d a `corehydro_dist` object.
#' @param test one of `"ks"` (Kolmogorov-Smirnov D), `"ad"` (Anderson-Darling A squared), or
#'   `"chi_squared"`.
#' @return a single numeric test statistic.
#' @examples
#' x <- c(2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9)
#' gof_test(x, distribution("Normal", c(3.4, 1.1)))
#' @export
gof_test <- function(x, d, test = c("ks", "ad", "chi_squared")) {
  test <- match.arg(test)
  if (!inherits(d, "corehydro_dist")) {
    stop("`d` must be a corehydro_dist object; create one with distribution()", call. = FALSE)
  }
  toolbox_run("gof", test, list(x), list(model = d))$values[[1]]
}

#' Root mean squared error of a fitted distribution
#'
#' @param x numeric vector of observations.
#' @param d a `corehydro_dist` object.
#' @param plotting_positions optional numeric vector of exceedance probabilities. `NULL` (the
#'   default) uses Weibull positions, matching the C# overload.
#' @return a single numeric RMSE.
#' @examples
#' x <- c(2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9)
#' gof_rmse(x, distribution("Normal", c(3.4, 1.1)))
#' @export
gof_rmse <- function(x, d, plotting_positions = NULL) {
  if (!inherits(d, "corehydro_dist")) {
    stop("`d` must be a corehydro_dist object; create one with distribution()", call. = FALSE)
  }
  data <- if (is.null(plotting_positions)) list(x) else list(x, plotting_positions)
  toolbox_run("gof", "rmse_dist", data, list(model = d))$values[[1]]
}

#' Information criteria
#'
#' @param n sample size.
#' @param k number of estimated parameters.
#' @param log_likelihood the maximized log-likelihood.
#' @return a single numeric criterion value.
#' @examples
#' aic(k = 2, log_likelihood = -121.01131220612)
#' bic(n = 30, k = 2, log_likelihood = -121.01131220612)
#' @export
aic <- function(k, log_likelihood) {
  toolbox_run("gof", "aic", list(), list(k = k, log_likelihood = log_likelihood))$values[[1]]
}

#' @rdname aic
#' @export
aicc <- function(n, k, log_likelihood) {
  toolbox_run("gof", "aicc", list(),
              list(n = n, k = k, log_likelihood = log_likelihood))$values[[1]]
}

#' @rdname aic
#' @export
bic <- function(n, k, log_likelihood) {
  toolbox_run("gof", "bic", list(),
              list(n = n, k = k, log_likelihood = log_likelihood))$values[[1]]
}

#' Model weights from a vector of criteria
#'
#' @param aic numeric vector of AIC values, one per candidate model.
#' @param rmse numeric vector of RMSE values, one per candidate model.
#' @return a numeric vector of weights summing to one.
#' @examples
#' aic_weights(c(246.0, 248.8, 251.2))
#' rmse_weights(c(1.2, 1.5, 2.0))
#' @export
aic_weights <- function(aic) {
  toolbox_run("gof", "aic_weights", list(aic))$values
}

#' @rdname aic_weights
#' @export
rmse_weights <- function(rmse) {
  toolbox_run("gof", "rmse_weights", list(rmse))$values
}
```

`to_spec_json()` already serializes a `corehydro_dist` into `{"family":...,"parameters":[...]}`,
which is exactly the `model` block the runner expects, so `list(model = d)` needs no special
handling.

Add to `corehydror/NAMESPACE`:
```
export(goodness_of_fit)
export(classification_metrics)
export(gof_test)
export(gof_rmse)
export(aic)
export(aicc)
export(bic)
export(aic_weights)
export(rmse_weights)
```

- [ ] **Step 7: Write the R behavioral test**

Create `corehydror/tests/testthat/test-gof.R`:

```r
test_that("goodness_of_fit returns every metric, and subsets agree with the whole", {
  obs <- c(2, 4, 6, 8, 10)
  mod <- c(2.2, 3.9, 6.4, 7.5, 10.1)
  all <- goodness_of_fit(obs, mod)
  expect_length(all, 17L)
  expect_true(all(c("nse", "kge", "rsr", "ve") %in% names(all)))
  expect_equal(goodness_of_fit(obs, mod, metrics = "nse")[["nse"]], all[["nse"]])
})

test_that("an unknown metric name is rejected and names the offender", {
  expect_error(goodness_of_fit(1:5, 1:5, metrics = "nsee"), "nsee")
})

test_that("mismatched lengths are rejected before reaching C++", {
  expect_error(goodness_of_fit(1:5, 1:4), "same length")
})

test_that("gof_test accepts a corehydro_dist and rejects anything else", {
  x <- c(2.1, 3.4, 1.8, 4.9, 3.3, 2.7, 5.1, 3.9)
  d <- distribution("Normal", c(3.4, 1.1))
  expect_true(gof_test(x, d, "ks") > 0)
  expect_true(gof_test(x, d, "ad") > 0)
  expect_error(gof_test(x, "Normal"), "corehydro_dist")
})

test_that("the information criteria match the closed forms", {
  expect_equal(aic(k = 2, log_likelihood = -100), 204)
  expect_equal(sum(aic_weights(c(246, 248.8, 251.2))), 1)
})
```

Run:
```bash
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL corehydror
Rscript -e 'testthat::test_local("corehydror")' 2>&1 | tail -5
```
Expected: PASS.

- [ ] **Step 8: Write the Python twin**

Create `corehydropy/src/corehydropy/gof.py` mirroring `gof.R` function for function, with the same
names, arguments, defaults, and error messages. Return dicts where R returns a named vector:

```python
"""The goodness-of-fit surface. Every verb runs one method of the ``"gof"`` group through the
shared toolbox runner. Mirrors ``corehydror``'s ``R/gof.R`` verb for verb; where R returns a named
vector this returns a dict.
"""

from __future__ import annotations

import numpy as np

from .toolbox import _check_pair, _toolbox_run

__all__ = [
    "goodness_of_fit",
    "classification_metrics",
    "gof_test",
    "gof_rmse",
    "aic",
    "aicc",
    "bic",
    "aic_weights",
    "rmse_weights",
]

_GOF_METRICS = (
    "rmse", "mse", "mae", "mape", "smape", "nse", "log_nse", "kge", "kge_mod",
    "pbias", "rsr", "pearson", "r_squared", "d", "d_mod", "d_ref", "ve",
)


def goodness_of_fit(observed, modeled, metrics="all", k: int = 0) -> dict:
    """Goodness-of-fit metrics for a modeled series.

    Mirrors the continuous metrics of the C# ``GoodnessOfFit`` class of the Numerics library.

    Parameters
    ----------
    observed, modeled : array_like
        Numeric vectors of equal length.
    metrics : str or sequence of str
        ``"all"`` (the default) for every metric, or the metric names to keep.
    k : int
        Degrees-of-freedom correction subtracted from the sample size in the RMSE denominator.

    Returns
    -------
    dict

    Examples
    --------
    >>> from corehydropy import goodness_of_fit
    >>> round(goodness_of_fit([2, 4, 6, 8, 10], [2.2, 3.9, 6.4, 7.5, 10.1])["nse"], 4)
    0.9895
    """
    obs, mod = _check_pair(observed, modeled, "observed", "modeled")
    if metrics != "all":
        unknown = [m for m in metrics if m not in _GOF_METRICS]
        if unknown:
            raise ValueError(
                f"unknown metric(s): {', '.join(unknown)}. Available: {', '.join(_GOF_METRICS)}"
            )
    r = _toolbox_run("gof", "metrics", [obs, mod], {"k": int(k)})
    out = dict(zip(r["names"], r["values"]))
    return out if metrics == "all" else {m: out[m] for m in metrics}
```

Write the remaining eight functions in the same file following the same shape:
`classification_metrics(observed, modeled, threshold)` returning a dict,
`gof_test(x, d, test="ks")` and `gof_rmse(x, d, plotting_positions=None)` taking a
`Distribution` and passing `{"model": json.loads(d.to_json())}` in options (check how
`corehydropy/src/corehydropy/distributions.py` exposes the spec; use that accessor rather than
re-serializing by hand), `aic(k, log_likelihood)`, `aicc(n, k, log_likelihood)`,
`bic(n, k, log_likelihood)`, `aic_weights(aic)` and `rmse_weights(rmse)` returning
`np.ndarray`. Each gets a numpydoc docstring with a runnable `Examples` block.

Re-export all nine from `corehydropy/src/corehydropy/__init__.py`.

Create `corehydropy/tests/test_gof.py` as the twin of `test-gof.R`, assertion for assertion.

Run:
```bash
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests/test_gof.py -q
```
Expected: PASS.

- [ ] **Step 9: Route the existing goodness_of_fit fixture kind through the runner**

The `goodness_of_fit` kind already exists with 14 pinned cases and its own dispatcher in the C++
runner and the emitter, and the R and Python runners skip it. Change three things without touching
a single pinned value:

1. In `core/tests/test_fixtures.cpp`, rewrite `dispatch_goodness_of_fit` to delegate:

```cpp
// Delegates to numerics/support/toolbox_runner.hpp so a fixture case and a user's
// goodness_of_fit() call are one code path. The function-name-to-method mapping is this file's
// only remaining knowledge of the group.
static double dispatch_goodness_of_fit(const std::string& fn, const std::vector<double>& args,
                                       const std::vector<double>& obs,
                                       const std::vector<double>& mod) {
    namespace tb = corehydro::numerics::support;
    static const std::map<std::string, std::string> kMethods = {
        {"MSE", "mse"}, {"MAE", "mae"},
        {"NashSutcliffeEfficiency", "nse"},
        {"KlingGuptaEfficiency", "kge"}, {"KlingGuptaEfficiencyMod", "kge_mod"},
        {"PBIAS", "pbias"}, {"RSR", "rsr"},
        {"IndexOfAgreement", "d"}, {"ModifiedIndexOfAgreement", "d_mod"},
        {"RefinedIndexOfAgreement", "d_ref"}, {"VolumetricEfficiency", "ve"},
    };
    if (fn == "AIC")
        return tb::run_toolbox("gof", "aic", {},
                               "{\"k\":" + std::to_string(static_cast<int>(args[0])) +
                                   ",\"log_likelihood\":" + format_g17(args[1]) + "}").values[0];
    if (fn == "AICc" || fn == "BIC")
        return tb::run_toolbox("gof", fn == "AICc" ? "aicc" : "bic", {},
                               "{\"n\":" + std::to_string(static_cast<int>(args[0])) +
                                   ",\"k\":" + std::to_string(static_cast<int>(args[1])) +
                                   ",\"log_likelihood\":" + format_g17(args[2]) + "}").values[0];
    auto it = kMethods.find(fn);
    if (it == kMethods.end())
        throw std::runtime_error("unknown goodness_of_fit function: " + fn);
    return tb::run_toolbox("gof", it->second, {obs, mod}, "{}").values[0];
}
```

`format_g17` may not exist in this file. If it does not, add it right above:

```cpp
static std::string format_g17(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}
```

2. In `corehydror/tests/testthat/test-fixtures.R`, add a `goodness_of_fit` branch that runs the
   same mapping through `ch_toolbox_run_`, and remove `goodness_of_fit` from whatever skip logic
   excludes it.
3. In `corehydropy/tests/test_fixtures.py`, add `"goodness_of_fit"` to the kind tuple in
   `_load_cases` and the matching branch in `test_fixture_case`.

Run all three suites. Expected: the C++ count is unchanged (the same 14 cases, now via the
runner), and R and Python each gain 14 assertions.

- [ ] **Step 10: Extend the fixture to the metrics the existing file misses**

Create `fixtures/toolbox/goodness_of_fit_metrics.json` with `"group": "gof"`, covering the
metrics the old file never pinned: `rmse` (with and without `k`), `mape`, `smape`, `log_nse`,
`pearson`, `r_squared`, the `classification` set, the three distribution-backed tests, and
`rmse_dist`, plus `aic_weights` and `rmse_weights`.

Source the expected values from
`upstream/Numerics/Test_Numerics/Data/Statistics/Test_GoodnessOfFit.cs`. For any metric that file
does not cover, curate the value with the emitter's dump mode rather than computing it in C++:

```bash
cd tools/oracle_emitter && dotnet run -- --dump ../../fixtures/toolbox/goodness_of_fit_metrics.json
```

Check how `--dump` is spelled in `Program.cs` before running it and follow that spelling. Never
fill an expected value from the C++ implementation being tested.

- [ ] **Step 11: Add the gof arm to the emitter**

Extend `ToolboxDispatch` in `tools/oracle_emitter/Program.cs`:

```csharp
        case "gof":
            {
                double[] o = data.Count > 0 ? data[0] : Array.Empty<double>();
                double[] m = data.Count > 1 ? data[1] : Array.Empty<double>();
                int k = options.ValueKind == JsonValueKind.Object && options.TryGetProperty("k", out var kEl)
                    ? kEl.GetInt32() : 0;
                switch (method)
                {
                    case "rmse": return GoodnessOfFit.RMSE(o, m, k);
                    case "mse": return GoodnessOfFit.MSE(o, m);
                    case "mae": return GoodnessOfFit.MAE(o, m);
                    case "mape": return GoodnessOfFit.MAPE(o, m);
                    case "smape": return GoodnessOfFit.SMAPE(o, m);
                    case "nse": return GoodnessOfFit.NashSutcliffeEfficiency(o, m);
                    case "log_nse": return GoodnessOfFit.LogNashSutcliffeEfficiency(o, m);
                    case "kge": return GoodnessOfFit.KlingGuptaEfficiency(o, m);
                    case "kge_mod": return GoodnessOfFit.KlingGuptaEfficiencyMod(o, m);
                    case "pbias": return GoodnessOfFit.PBIAS(o, m);
                    case "rsr": return GoodnessOfFit.RSR(o, m);
                    case "pearson": return GoodnessOfFit.Pearson(o, m);
                    case "r_squared": return GoodnessOfFit.RSquared(o, m);
                    case "d": return GoodnessOfFit.IndexOfAgreement(o, m);
                    case "d_mod": return GoodnessOfFit.ModifiedIndexOfAgreement(o, m);
                    case "d_ref": return GoodnessOfFit.RefinedIndexOfAgreement(o, m);
                    case "ve": return GoodnessOfFit.VolumetricEfficiency(o, m);
                    default: throw new Exception($"unknown gof method: {method}");
                }
            }
```

Verify every C# method name against `upstream/Numerics/Numerics/Data/Statistics/GoodnessOfFit.cs`
before compiling; the names above are the C++ port's names and the C# spellings may differ in
case. Add the `metrics`, `classification`, `aic`, `aicc`, `bic`, `aic_weights`, `rmse_weights`,
`ks`, `ad`, `chi_squared` and `rmse_dist` methods the same way. For `metrics` and
`classification`, honor the assertion's `index` or `label` by building the same ordered list the
runner builds and selecting from it.

Run:
```bash
python3 tools/verify_oracles.py 2>&1 | tail -5
```
Expected: `failed 0`, reproduced count up by the number of new assertions.

- [ ] **Step 12: Commit**

```bash
git add core corehydror corehydropy fixtures tools
git commit -m "feat: expose the goodness-of-fit surface in R and Python"
```

---

### Task 3: Autocorrelation port, spectra, and statistics

**Files:**
- Create: `core/include/corehydro/numerics/data/autocorrelation.hpp`, `core/tests/test_autocorrelation.cpp`
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (the `statistics` and `spectra` arms)
- Modify: `corehydror/R/toolbox.R`, `corehydropy/src/corehydropy/toolbox.py`
- Create: `fixtures/toolbox/statistics.json`, `fixtures/toolbox/autocorrelation.json`
- Modify: `core/CMakeLists.txt`, `tools/oracle_emitter/Program.cs`, `corehydror/NAMESPACE`, `corehydropy/src/corehydropy/__init__.py`, `upstream/CLAUDE.md`
- Test: `core/tests/test_autocorrelation.cpp`, `corehydror/tests/testthat/test-toolbox.R`, `corehydropy/tests/test_toolbox.py`

**Interfaces:**
- Produces: `corehydro::numerics::data::Autocorrelation` with `enum class Type { Correlation, Covariance, Partial }`, `static std::optional<std::vector<std::array<double,2>>> function(const std::vector<double>& data, int lag_max = -1, Type type = Type::Correlation)`, and `static std::vector<double> correlation_confidence_interval(int sample_size, double interval = 0.95)`.
- Produces: toolbox group `"statistics"` with methods `summary`, `product_moments`, `l_moments`, `ranks`, `percentile`, `running`, `running_covariance`; group `"spectra"` with `autocorrelation`, `autocorrelation_ci`, `cross_correlation`, `dft`, `dft_real`.
- Produces: R `summary_statistics()`, `product_moments()`, `l_moments()`, `ranks()`, `percentile()`, `running_statistics()`, `running_covariance()`, `autocorrelation()`, `cross_correlation()`, `dft()`, `dft_real()`; the same in Python.

- [ ] **Step 1: Read the C# source you are porting**

Read `upstream/Numerics/Numerics/Data/Statistics/Autocorrelation.cs` in full (324 lines) and
`upstream/Numerics/Test_Numerics/Data/Statistics/Test_Autocorrelation.cs`. Note the exact default
`lagMax` rule, the tie handling, and what `Function` returns when the series is too short (a
nullable `double[,]`, which the port models as `std::optional`).

Note for the provenance header: this file is ported at Numerics `2a0357a`.

- [ ] **Step 2: Write the failing ctest for the port**

Create `core/tests/test_autocorrelation.cpp` asserting, at minimum: the correlation at lag 0 is 1,
the returned pair count equals `lag_max + 1`, a series shorter than two elements returns
`std::nullopt`, the covariance type at lag 0 equals the population variance, the partial
autocorrelation at lag 1 equals the correlation at lag 1, and
`correlation_confidence_interval(100)` is symmetric about zero. Use only relationships that hold
by construction, never a numeric literal copied from the C++ implementation.

Add `test_autocorrelation` to `BF_TESTS` in `core/CMakeLists.txt`.

Run: `cmake -S core -B core/build && cmake --build core/build`
Expected: FAIL, header not found.

- [ ] **Step 3: Write the port**

Create `core/include/corehydro/numerics/data/autocorrelation.hpp` with the provenance header
`// ported from: Numerics/Data/Statistics/Autocorrelation.cs @ 2a0357a`, mirroring the C# class
and method layout line for line as every other ported file does. Reuse
`corehydro::numerics::data::mean` and `variance` from `statistics.hpp` and
`corehydro::numerics::math::fourier` where the C# calls the same helpers. Do not invent a faster
algorithm: fidelity beats speed here, and the oracle gate will catch any divergence.

Run: `cmake --build core/build && ctest --test-dir core/build -R test_autocorrelation --output-on-failure`
Expected: PASS.

- [ ] **Step 4: Pin the port against the C# oracle**

Create `fixtures/toolbox/autocorrelation.json` with `"group": "spectra"`, transcribing the
literals from `Test_Autocorrelation.cs`. Add the `spectra` arm to `ToolboxDispatch` in the
emitter, calling `Autocorrelation.Function` and `Autocorrelation.CorrelationConfidenceInterval`
against the real library, and make sure `Autocorrelation.cs` is in the emitter's compile list.

Run: `python3 tools/verify_oracles.py`
Expected: `failed 0`.

- [ ] **Step 5: Record the port in the upstream notes**

Add an entry to `upstream/CLAUDE.md` stating that `Numerics/Data/Statistics/Autocorrelation.cs`
was unported through phase 3 and is ported as of this phase, and that
`Correlation.Pearson(double[,])` / `Correlation.Spearman(double[,])` (the matrix overloads) remain
unported.

Commit:
```bash
git add core fixtures tools upstream/CLAUDE.md
git commit -m "feat(core): port Autocorrelation.cs with autocovariance, PACF and confidence bands"
```

- [ ] **Step 6: Add the statistics and spectra arms to the runner**

In `toolbox_runner.hpp`, add includes for `statistics.hpp`, `running_statistics.hpp`,
`running_covariance_matrix.hpp`, `autocorrelation.hpp`, and `math/fourier/fourier.hpp`, then the
two arms in `namespace detail`:

```cpp
inline ToolboxResult run_statistics(const std::string& method,
                                    const std::vector<std::vector<double>>& data,
                                    const JsonValue& options) {
    namespace nd = numerics::data;
    const std::vector<double>& x = data_at(data, 0, "statistics", method);

    if (method == "summary" || method == "running") {
        // "running" seeds the accumulator from a prior state carried in the options; "summary"
        // is the same call with no prior state. One code path, so a chunked run and a one-shot
        // run can never disagree.
        nd::RunningStatistics rs;
        if (method == "running" && options.contains("state")) {
            const JsonValue& s = options.at("state");
            rs = nd::RunningStatistics::from_state(
                s.at("n").as_int(), s.at("m1").as_double(), s.at("m2").as_double(),
                s.at("m3").as_double(), s.at("m4").as_double(), s.at("minimum").as_double(),
                s.at("maximum").as_double());
        }
        rs.push(x);
        ToolboxResult r;
        r.values = {static_cast<double>(rs.count()), rs.minimum(), rs.maximum(), rs.mean(),
                    rs.variance(), rs.standard_deviation(), rs.coefficient_of_variation(),
                    rs.skewness(), rs.kurtosis(), rs.m1_state(), rs.m2_state(), rs.m3_state(),
                    rs.m4_state()};
        r.names = {"n", "minimum", "maximum", "mean", "variance", "sd", "cv", "skewness",
                   "kurtosis", "m1", "m2", "m3", "m4"};
        return r;
    }
    if (method == "product_moments") {
        ToolboxResult r;
        r.values = nd::product_moments(x);
        r.names = {"mean", "sd", "skewness", "kurtosis"};
        return r;
    }
    if (method == "l_moments") {
        ToolboxResult r;
        r.values = nd::linear_moments(x);
        r.names = {"l1", "l2", "t3", "t4"};
        return r;
    }
    if (method == "ranks") {
        ToolboxResult r;
        r.values = nd::ranks_in_place(x);
        return r;
    }
    if (method == "percentile") {
        ToolboxResult r;
        bool sorted = options.value_or("sorted", false);
        for (double p : data_at(data, 1, "statistics", method))
            r.values.push_back(nd::percentile(x, p, sorted));
        return r;
    }
    throw std::runtime_error("unknown statistics method: " + method);
}
```

`RunningStatistics` has no `from_state` constructor or `m1_state()` accessors today. Before
writing this arm, read `core/include/corehydro/numerics/data/running_statistics.hpp` and add the
minimum accessors needed to round-trip the accumulator: a constructor taking
`(n, m1, m2, m3, m4, min, max)` and const accessors for the four raw moments. Both are additive,
mirror no C# API, and get a comment saying so (`// corehydro ADDITION: the stateless R/Python
surface serializes the accumulator, so the five raw fields must round-trip`). Confirm the exact
private member names before writing it. Adding members to this class changes its layout, so
rebuild R with `R CMD INSTALL --preclean corehydror` afterward.

Write `run_spectra` the same way: `autocorrelation` returning lag and value interleaved with
`dims = {n, 2}`, `autocorrelation_ci` returning the two-element band, `cross_correlation` calling
`fourier::correlation`, and `dft` / `dft_real` returning the transformed vector (both C# functions
mutate their input, so copy first).

Extend the ctest in `core/tests/test_toolbox_runner.cpp` for both groups, then run the ctest.

- [ ] **Step 7: Write the R and Python verbs**

Add to `corehydror/R/toolbox.R`, each with full roxygen and a runnable `@examples`:

```r
summary_statistics(x)
product_moments(x)
l_moments(x)
ranks(x)
percentile(x, probs, sorted = FALSE)
running_statistics(x, state = NULL)
running_covariance(x, state = NULL)
autocorrelation(x, max_lag = NULL, type = c("correlation", "covariance", "partial"),
                confidence_level = 0.95)
cross_correlation(x, y)
dft(x, inverse = FALSE)
dft_real(x, inverse = FALSE)
```

`running_statistics()` returns a `corehydro_running` classed list carrying `n`, `minimum`,
`maximum`, `m1`, `m2`, `m3`, `m4` plus the derived statistics, with a `print` method. Passing that
object back as `state` continues the accumulation:

```r
#' Streaming summary statistics
#'
#' Accumulates count, extremes, and the first four moments over one or more chunks of data,
#' mirroring the C# `RunningStatistics` class. The accumulator state travels in the return value,
#' so a chunked run holds no C++ state and the result serializes with `save()`.
#'
#' @param x numeric vector, the next chunk of data.
#' @param state a `corehydro_running` object from a previous call, or `NULL` (the default) to
#'   start a fresh accumulator.
#' @return a `corehydro_running` list with `n`, `minimum`, `maximum`, `mean`, `variance`, `sd`,
#'   `cv`, `skewness`, `kurtosis`, and the four raw moments `m1` to `m4`.
#' @examples
#' s <- running_statistics(c(1, 2, 3))
#' s <- running_statistics(c(4, 5, 6), state = s)
#' s$n
#' s$mean
#' @export
running_statistics <- function(x, state = NULL) {
  if (!is.null(state) && !inherits(state, "corehydro_running")) {
    stop("`state` must be a corehydro_running object from a previous call", call. = FALSE)
  }
  opts <- if (is.null(state)) {
    list()
  } else {
    list(state = list(n = state$n, m1 = state$m1, m2 = state$m2, m3 = state$m3, m4 = state$m4,
                      minimum = state$minimum, maximum = state$maximum))
  }
  r <- toolbox_run("statistics", if (is.null(state)) "summary" else "running", list(x), opts)
  structure(as.list(stats::setNames(r$values, r$names)), class = "corehydro_running")
}

#' @export
print.corehydro_running <- function(x, ...) {
  cat(sprintf("<corehydro_running> n = %d, mean = %.6g, sd = %.6g\n", x$n, x$mean, x$sd))
  invisible(x)
}
```

Write the Python twins in `corehydropy/src/corehydropy/toolbox.py`, with `running_statistics()`
returning a `RunningStatistics` dataclass-like wrapper carrying the same fields and a `__repr__`.

Add every new name to `corehydror/NAMESPACE` (including `S3method(print, corehydro_running)`) and
to the Python `__init__.py`.

- [ ] **Step 8: Write the behavioral tests**

Create `corehydror/tests/testthat/test-toolbox.R` and `corehydropy/tests/test_toolbox.py` covering:
chunked `running_statistics()` reaching the same mean, variance, skewness, and kurtosis as a
single call over the concatenated data; `percentile()` accepting a vector of probabilities;
`autocorrelation()` at lag 0 equal to 1 for the correlation type; `dft(dft(x, inverse = TRUE))`
round-tripping to within 1e-12; `ranks()` handling ties; and each argument-validation error naming
its argument.

Run all three suites, then `python3 tools/verify_oracles.py`.

- [ ] **Step 9: Commit**

```bash
git add core corehydror corehydropy fixtures tools
git commit -m "feat: expose summary, streaming and spectral statistics in R and Python"
```

---

### Task 4: Histogram and interpolation

**Files:**
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (the `histogram` and `interpolation` arms)
- Modify: `corehydror/R/toolbox.R`, `corehydropy/src/corehydropy/toolbox.py`, `corehydror/NAMESPACE`, `corehydropy/src/corehydropy/__init__.py`
- Create: `fixtures/toolbox/histogram.json`, `fixtures/toolbox/interpolation.json`
- Modify: `tools/oracle_emitter/Program.cs`, `core/tests/test_toolbox_runner.cpp`
- Test: `corehydror/tests/testthat/test-toolbox.R`, `corehydropy/tests/test_toolbox.py`

**Interfaces:**
- Consumes: `run_toolbox`, `toolbox_run`, `_toolbox_run`.
- Produces: group `"histogram"` with methods `bins` (returns lower, upper, midpoint, frequency flattened row-major with `dims = {n, 4}`) and `statistics` (mean, median, mode, sd as a named set); group `"interpolation"` with `linear` and `bilinear`.
- Produces: R `histogram()`, `interpolate()`, `interpolate_2d()`; the same in Python.

- [ ] **Step 1: Read the headers**

`core/include/corehydro/numerics/data/histogram.hpp` and everything under
`core/include/corehydro/numerics/data/interpolation/`. Note that `Transform` is
`{None, Logarithmic, NormalZ}` and `SortOrder` is `{Ascending, Descending}`, both in
`corehydro::numerics::data`.

- [ ] **Step 2: Write the failing ctest additions**

Append to `core/tests/test_toolbox_runner.cpp`:

```cpp
    // histogram: Rice-rule bin count, and the frequencies sum to the sample size.
    const std::vector<double> h{1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 8.0, 9.0};
    auto bins = tb::run_toolbox("histogram", "bins", {h}, "{}");
    CHTEST_EXPECT_EQ(bins.dims.size(), std::size_t{2});
    CHTEST_EXPECT_EQ(bins.dims[1], 4);
    double total = 0.0;
    for (int row = 0; row < bins.dims[0]; ++row)
        total += bins.values[static_cast<std::size_t>(row * 4 + 3)];
    CHTEST_EXPECT_NEAR(total, 10.0, 0.0);

    // interpolation: a point on a knot returns that knot's y exactly.
    const std::vector<double> ix{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> iy{10.0, 20.0, 30.0, 40.0};
    auto lin = tb::run_toolbox("interpolation", "linear", {ix, iy, {2.5, 3.0}}, "{}");
    CHTEST_EXPECT_EQ(lin.values.size(), std::size_t{2});
    CHTEST_EXPECT_NEAR(lin.values[0], 25.0, 1e-12);
    CHTEST_EXPECT_NEAR(lin.values[1], 30.0, 1e-12);
```

Run the ctest. Expected: FAIL, `unknown toolbox group: histogram`.

- [ ] **Step 3: Write the two arms**

Add to `namespace detail` in `toolbox_runner.hpp`:

```cpp
inline ToolboxResult run_histogram(const std::string& method,
                                   const std::vector<std::vector<double>>& data,
                                   const JsonValue& options) {
    const std::vector<double>& x = data_at(data, 0, "histogram", method);
    // bins == 0 means the Rice-rule constructor; explicit bounds override the data range.
    int bins = options.value_or("bins", 0);
    numerics::data::Histogram h = bins > 0 ? numerics::data::Histogram(x, bins)
                                           : numerics::data::Histogram(x);
    if (method == "bins") {
        ToolboxResult r;
        for (int i = 0; i < h.number_of_bins(); ++i) {
            numerics::data::Histogram::Bin b = h.bin(i);
            r.values.push_back(b.lower_bound);
            r.values.push_back(b.upper_bound);
            r.values.push_back(b.midpoint());
            r.values.push_back(static_cast<double>(b.frequency));
        }
        r.names = {"lower", "upper", "midpoint", "frequency"};
        r.dims = {h.number_of_bins(), 4};
        return r;
    }
    if (method == "statistics") {
        ToolboxResult r;
        r.values = {h.mean(), h.median(), h.mode(), h.standard_deviation(),
                    h.lower_bound(), h.upper_bound(), h.bin_width(),
                    static_cast<double>(h.number_of_bins())};
        r.names = {"mean", "median", "mode", "sd", "lower", "upper", "bin_width", "bins"};
        return r;
    }
    throw std::runtime_error("unknown histogram method: " + method);
}
```

Check `Histogram`'s constructors and accessors against the header before compiling: the
explicit-bounds constructor may take `(data, bins)` only, in which case `lower` and `upper`
options are not supported and the R argument list must drop them.

And the interpolation arm, with two enum parsers beside it:

```cpp
inline numerics::data::Transform parse_transform(const std::string& s) {
    if (s == "none") return numerics::data::Transform::None;
    if (s == "log") return numerics::data::Transform::Logarithmic;
    if (s == "normal_z") return numerics::data::Transform::NormalZ;
    throw std::runtime_error("unknown transform '" + s + "'; expected none, log, or normal_z");
}

inline numerics::data::SortOrder parse_sort_order(const std::string& s) {
    if (s == "ascending") return numerics::data::SortOrder::Ascending;
    if (s == "descending") return numerics::data::SortOrder::Descending;
    throw std::runtime_error("unknown sort order '" + s + "'; expected ascending or descending");
}

inline ToolboxResult run_interpolation(const std::string& method,
                                       const std::vector<std::vector<double>>& data,
                                       const JsonValue& options) {
    namespace nd = numerics::data;
    nd::SortOrder order = parse_sort_order(options.value_or("sort_order", "ascending"));

    if (method == "linear") {
        const std::vector<double>& x = data_at(data, 0, "interpolation", method);
        const std::vector<double>& y = data_at(data, 1, "interpolation", method);
        const std::vector<double>& xout = data_at(data, 2, "interpolation", method);
        nd::Linear interp(x, y, order);
        interp.x_transform = parse_transform(options.value_or("x_transform", "none"));
        interp.y_transform = parse_transform(options.value_or("y_transform", "none"));
        bool extrapolate = options.value_or("extrapolate", false);
        ToolboxResult r;
        for (double v : xout)
            r.values.push_back(extrapolate ? interp.extrapolate(v) : interp.interpolate(v));
        return r;
    }

    if (method == "bilinear") {
        const std::vector<double>& x1 = data_at(data, 0, "interpolation", method);
        const std::vector<double>& x2 = data_at(data, 1, "interpolation", method);
        const std::vector<double>& flat = data_at(data, 2, "interpolation", method);
        const std::vector<double>& x1out = data_at(data, 3, "interpolation", method);
        const std::vector<double>& x2out = data_at(data, 4, "interpolation", method);
        if (flat.size() != x1.size() * x2.size())
            throw std::runtime_error("bilinear 'y' holds " + std::to_string(flat.size()) +
                                     " values, expected " + std::to_string(x1.size()) + " x " +
                                     std::to_string(x2.size()));
        if (x1out.size() != x2out.size())
            throw std::runtime_error("bilinear needs one x2 value per x1 value");
        std::vector<std::vector<double>> y(x1.size(), std::vector<double>(x2.size()));
        for (std::size_t i = 0; i < x1.size(); ++i)
            for (std::size_t j = 0; j < x2.size(); ++j) y[i][j] = flat[i * x2.size() + j];
        nd::Bilinear interp(x1, x2, y, order);
        interp.x1_transform = parse_transform(options.value_or("x1_transform", "none"));
        interp.x2_transform = parse_transform(options.value_or("x2_transform", "none"));
        interp.y_transform = parse_transform(options.value_or("y_transform", "none"));
        ToolboxResult r;
        for (std::size_t i = 0; i < x1out.size(); ++i)
            r.values.push_back(interp.interpolate(x1out[i], x2out[i]));
        return r;
    }
    throw std::runtime_error("unknown interpolation method: " + method);
}
```

Confirm before compiling: that `Transform` and `SortOrder` live in `corehydro::numerics::data`
(they are declared in `interpolation/transform.hpp` and `interpolation/sort_order.hpp`), that
`Linear::extrapolate` and `Bilinear`'s three transform members are public, and that `Bilinear`'s
constructor takes `y` as `std::vector<std::vector<double>>` indexed `[x1][x2]` in that order. If
the row and column roles are reversed, transpose the fill loop and say so in a comment.

Add both dispatch lines to `run_toolbox`:

```cpp
    if (group == "histogram") return detail::run_histogram(method, data, options);
    if (group == "interpolation") return detail::run_interpolation(method, data, options);
```

Run the ctest. Expected: PASS.

- [ ] **Step 4: Write the R and Python verbs**

```r
#' Bin a sample into a histogram
#'
#' Mirrors the C# `Histogram` class of the Numerics library. With `bins = NULL` the bin count
#' follows the Rice rule, `ceiling(2 * n^(1/3)) + 1`, exactly as the C# data constructor does.
#'
#' @param x numeric vector of observations.
#' @param bins optional number of bins. `NULL` (the default) uses the Rice rule.
#' @return a data frame with columns `lower`, `upper`, `midpoint`, and `frequency`, carrying the
#'   histogram's `mean`, `median`, `mode`, `sd`, and `bin_width` as attributes.
#' @examples
#' h <- histogram(c(1, 2, 2.5, 3, 3.5, 4, 5, 7, 8, 9))
#' h
#' attr(h, "mode")
#' @export
histogram <- function(x, bins = NULL) {
  if (!is.numeric(x) || length(x) < 2L) {
    stop("`x` must be a numeric vector with at least two elements", call. = FALSE)
  }
  opts <- if (is.null(bins)) list() else list(bins = as.integer(bins))
  b <- toolbox_run("histogram", "bins", list(x), opts)
  out <- as.data.frame(matrix(b$values, ncol = 4L, byrow = TRUE))
  names(out) <- b$names
  s <- toolbox_run("histogram", "statistics", list(x), opts)
  for (i in seq_along(s$names)) attr(out, s$names[[i]]) <- s$values[[i]]
  out
}

#' Interpolate a paired series
#'
#' Mirrors the C# `Linear` interpolater of the Numerics library, including its x and y
#' transforms.
#'
#' @param x,y numeric vectors of equal length defining the knots.
#' @param xout numeric vector of positions to interpolate at.
#' @param x_transform,y_transform one of `"none"` (the default), `"log"`, or `"normal_z"`.
#' @param sort_order `"ascending"` (the default) or `"descending"`, describing `x`.
#' @param extrapolate whether to extend the end segments beyond the knots. Default `FALSE`,
#'   which clamps to the end knot, matching the C# default.
#' @return a numeric vector the same length as `xout`.
#' @examples
#' interpolate(c(1, 2, 3, 4), c(10, 20, 30, 40), c(1.5, 2.5))
#' @export
interpolate <- function(x, y, xout, x_transform = c("none", "log", "normal_z"),
                        y_transform = c("none", "log", "normal_z"),
                        sort_order = c("ascending", "descending"), extrapolate = FALSE) {
  check_pair(x, y)
  toolbox_run("interpolation", "linear", list(x, y, xout),
              list(x_transform = match.arg(x_transform), y_transform = match.arg(y_transform),
                   sort_order = match.arg(sort_order), extrapolate = isTRUE(extrapolate)))$values
}
```

`interpolate_2d(x1, x2, y, x1out, x2out, ...)` follows the same shape, flattening `y` with
`as.double(t(y))` and validating `dim(y)` against `length(x1)` by `length(x2)` before the call.
Write the Python twins with the same names, arguments, defaults, and messages; `histogram()`
returns a dict of numpy arrays plus a `statistics` key. Add every name to `NAMESPACE` and
`__init__.py`.

- [ ] **Step 5: Write the fixtures and the emitter arms**

`fixtures/toolbox/histogram.json` and `fixtures/toolbox/interpolation.json`, values transcribed
from `upstream/Numerics/Test_Numerics/Data/Statistics/Test_Histogram.cs` and
`upstream/Numerics/Test_Numerics/Data/Paired Data/Test_PairedDataInterpolation.cs`. Add the two
arms to `ToolboxDispatch` and the two C# files to the emitter's compile list.

Run: `python3 tools/verify_oracles.py`. Expected: `failed 0`.

- [ ] **Step 6: Write the behavioral tests and commit**

Extend `test-toolbox.R` and `test_toolbox.py`: bin frequencies sum to `length(x)`; an out-of-range
`xout` clamps to the end knot without `extrapolate` and does not with it; an unknown transform
name is rejected and lists the accepted values; `interpolate_2d()` rejects a `y` matrix whose
dimensions do not match `length(x1)` by `length(x2)`.

```bash
git add core corehydror corehydropy fixtures tools
git commit -m "feat: expose histogram and interpolation in R and Python"
```

---

### Task 5: Linear regression

**Files:**
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (the `regression` arm)
- Modify: `corehydror/R/toolbox.R`, `corehydropy/src/corehydropy/toolbox.py`, `corehydror/NAMESPACE`, `corehydropy/src/corehydropy/__init__.py`
- Create: `fixtures/toolbox/linear_regression.json`
- Modify: `tools/oracle_emitter/Program.cs`, `core/tests/test_toolbox_runner.cpp`
- Test: `corehydror/tests/testthat/test-toolbox.R`, `corehydropy/tests/test_toolbox.py`

**Interfaces:**
- Produces: group `"regression"` with methods `fit` (named set: parameters, standard errors, r_squared, adj_r_squared, sigma, df, n, plus the covariance matrix in a second call), `covariance` (`dims = {p, p}`), `residuals`, `predict`, `prediction_intervals` (`dims = {n, 3}`).
- Produces: R `linear_regression()` returning a `corehydro_lm` classed list with `coef`, `vcov`, `residuals`, `predict`, `print`, `summary` methods; Python `linear_regression()` returning a `LinearRegressionResult`.

- [ ] **Step 1: Read the header**

`core/include/corehydro/numerics/data/regression/linear_regression.hpp`. The constructor takes a
`Matrix` of predictors with one column per observation or one row per observation (read the
constructor's validation message at line 57 to be sure which), a `Vector` of responses, and a
`has_intercept` flag. `predict` takes a `Matrix`; `prediction_intervals(x, alpha)` returns an
n by 3 `Matrix` of lower, upper, mean.

- [ ] **Step 2: Write the failing ctest**

Append to `core/tests/test_toolbox_runner.cpp` a case fitting a two-predictor model whose exact
solution is known by construction (build `y` as an exact linear combination of the predictors, fit
it, and assert the coefficients recover the combination to 1e-9 and `r_squared` is 1 to 1e-12).
Data crosses as: vector 0 = the flattened predictor matrix (row-major), vector 1 = `y`, with
`options = {"rows": n, "columns": p, "intercept": true}`.

Run the ctest. Expected: FAIL, unknown group.

- [ ] **Step 3: Write the arm**

Add to `namespace detail` in `toolbox_runner.hpp`, with an include for
`corehydro/numerics/data/regression/linear_regression.hpp`:

```cpp
// Predictors cross as one flattened row-major vector plus `rows` and `columns`, because the
// binding layer has no matrix type in common between R (column-major) and Python (numpy). The
// wrappers transpose on the way in; this is the one place that layout is assumed.
inline numerics::data::LinearRegression build_regression(
    const std::vector<std::vector<double>>& data, const JsonValue& options,
    const std::string& method) {
    namespace ml = math::linalg;
    const std::vector<double>& flat = data_at(data, 0, "regression", method);
    const std::vector<double>& yv = data_at(data, 1, "regression", method);
    int rows = options.value_or("rows", static_cast<int>(yv.size()));
    int cols = options.value_or("columns", 1);
    bool intercept = options.value_or("intercept", true);
    if (flat.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols))
        throw std::runtime_error("regression predictors hold " + std::to_string(flat.size()) +
                                 " values, expected rows * columns = " +
                                 std::to_string(rows * cols));
    return numerics::data::LinearRegression(ml::Matrix(rows, cols, flat), ml::Vector(yv),
                                            intercept);
}

inline ToolboxResult run_regression(const std::string& method,
                                    const std::vector<std::vector<double>>& data,
                                    const JsonValue& options) {
    namespace ml = math::linalg;
    numerics::data::LinearRegression fit = build_regression(data, options, method);
    ToolboxResult r;

    if (method == "fit") {
        const std::vector<double>& beta = fit.parameters();
        const std::vector<double>& se = fit.parameter_standard_errors();
        for (std::size_t i = 0; i < beta.size(); ++i) {
            r.values.push_back(beta[i]);
            r.names.push_back("beta_" + std::to_string(i + 1));
        }
        for (std::size_t i = 0; i < se.size(); ++i) {
            r.values.push_back(se[i]);
            r.names.push_back("se_" + std::to_string(i + 1));
        }
        r.values.push_back(fit.r_squared());                               r.names.push_back("r_squared");
        r.values.push_back(fit.adj_r_squared());                           r.names.push_back("adj_r_squared");
        r.values.push_back(fit.standard_error());                          r.names.push_back("sigma");
        r.values.push_back(static_cast<double>(fit.degrees_of_freedom()));  r.names.push_back("df");
        r.values.push_back(static_cast<double>(fit.sample_size()));         r.names.push_back("n");
        return r;
    }
    if (method == "covariance") {
        const ml::Matrix& c = fit.covariance();
        for (int i = 0; i < c.number_of_rows(); ++i)
            for (int j = 0; j < c.number_of_columns(); ++j) r.values.push_back(c(i, j));
        r.dims = {c.number_of_rows(), c.number_of_columns()};
        return r;
    }
    if (method == "residuals") {
        r.values = fit.residuals();
        return r;
    }
    if (method == "predict" || method == "prediction_intervals") {
        const std::vector<double>& nd = data_at(data, 2, "regression", method);
        int nrows = options.at("predict_rows").as_int();
        int ncols = options.value_or("columns", 1);
        ml::Matrix xp(nrows, ncols, nd);
        if (method == "predict") {
            r.values = fit.predict(xp);
            return r;
        }
        double alpha = options.value_or("alpha", 0.1);
        ml::Matrix pi = fit.prediction_intervals(xp, alpha);
        for (int i = 0; i < pi.number_of_rows(); ++i)
            for (int j = 0; j < pi.number_of_columns(); ++j) r.values.push_back(pi(i, j));
        r.names = {"lower", "upper", "mean"};
        r.dims = {pi.number_of_rows(), pi.number_of_columns()};
        return r;
    }
    throw std::runtime_error("unknown regression method: " + method);
}
```

Add the dispatch line `if (group == "regression") return detail::run_regression(method, data, options);`.

Two things to confirm against the headers before compiling: the ordering `Matrix(rows, columns,
flat)` assumes (`matrix.hpp:113`), and that `prediction_intervals` returns its three columns as
lower, upper, mean in that order (`linear_regression.hpp:122-125` says it does).

Run the ctest. Expected: PASS.

- [ ] **Step 4: Write the R and Python verbs**

```r
#' Ordinary least squares by singular value decomposition
#'
#' Mirrors the C# `LinearRegression` class of the Numerics library.
#'
#' @param x a numeric matrix of predictors with one row per observation, or a numeric vector for
#'   a single predictor.
#' @param y numeric vector of responses, one per row of `x`.
#' @param intercept whether to fit an intercept. Default `TRUE`.
#' @return a `corehydro_lm` list with `coefficients`, `standard_errors`, `covariance`,
#'   `residuals`, `r_squared`, `adj_r_squared`, `sigma`, `df`, and `n`.
#' @examples
#' x <- cbind(c(1, 2, 3, 4, 5), c(2, 1, 4, 3, 5))
#' y <- c(3.1, 4.2, 8.1, 9.2, 13.0)
#' fit <- linear_regression(x, y)
#' coef(fit)
#' @export
linear_regression <- function(x, y, intercept = TRUE) {
  x <- if (is.matrix(x)) x else matrix(as.double(x), ncol = 1L)
  if (!is.numeric(y) || length(y) != nrow(x)) {
    stop(sprintf("`y` must be numeric with one value per row of `x`; got %d and %d",
                 length(y), nrow(x)), call. = FALSE)
  }
  # t(x) flattens row-major, which is the layout the runner's Matrix(rows, columns, flat) wants.
  flat <- as.double(t(x))
  opts <- list(rows = nrow(x), columns = ncol(x), intercept = intercept)
  v <- toolbox_run("regression", "fit", list(flat, y), opts)
  named <- stats::setNames(v$values, v$names)
  p <- ncol(x) + as.integer(isTRUE(intercept))
  cov <- toolbox_run("regression", "covariance", list(flat, y), opts)
  res <- toolbox_run("regression", "residuals", list(flat, y), opts)
  structure(list(
    coefficients = named[seq_len(p)],
    standard_errors = named[p + seq_len(p)],
    covariance = matrix(cov$values, nrow = cov$dims[[1]], ncol = cov$dims[[2]], byrow = TRUE),
    residuals = res$values,
    r_squared = named[["r_squared"]],
    adj_r_squared = named[["adj_r_squared"]],
    sigma = named[["sigma"]],
    df = named[["df"]],
    n = named[["n"]],
    x = x,
    y = as.double(y),
    intercept = isTRUE(intercept)
  ), class = "corehydro_lm")
}
```

Implement `coef.corehydro_lm`, `vcov.corehydro_lm`, `residuals.corehydro_lm`,
`predict.corehydro_lm(object, newdata, interval = FALSE, level = 0.90)`, `print.corehydro_lm`,
and `summary.corehydro_lm` following the shapes already used for `corehydro_fit` in
`corehydror/R/fit.R`. Read that file first and match it. Python returns a
`LinearRegressionResult` with the same fields plus a `predict()` method.

Add the export and the six `S3method()` lines to `NAMESPACE`.

- [ ] **Step 5: Fixture, emitter, tests, commit**

`fixtures/toolbox/linear_regression.json` from
`upstream/Numerics/Test_Numerics/Data/Regression/Test_LinearRegression.cs`. Add the `regression`
arm to `ToolboxDispatch`. Behavioral tests: coefficients match `stats::lm()` on the same data to
1e-8 in R and `numpy.linalg.lstsq` in Python (a genuinely independent check, worth having);
`intercept = FALSE` drops the column; a `y` of the wrong length is rejected naming both lengths.

```bash
git add core corehydror corehydropy fixtures tools
git commit -m "feat: expose linear regression in R and Python"
```

---

### Task 6: Sobol, stratification, and joint probability

**Files:**
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (the `sampling` and `probability` arms)
- Modify: `corehydror/R/toolbox.R`, `corehydropy/src/corehydropy/toolbox.py`, `corehydror/NAMESPACE`, `corehydropy/src/corehydropy/__init__.py`
- Modify: `corehydror/src/sobol.cpp`, `corehydropy/src/bindings/sobol.cpp` (fold into the runner)
- Create: `fixtures/toolbox/sampling.json`, `fixtures/toolbox/joint_probability.json`
- Modify: `tools/oracle_emitter/Program.cs`, `core/tests/test_toolbox_runner.cpp`
- Test: `corehydror/tests/testthat/test-toolbox.R`, `corehydropy/tests/test_toolbox.py`

**Interfaces:**
- Produces: group `"sampling"` with `sobol` (`dims = {n, dimension}`) and `stratify` (`dims = {bins, 4}` for lower, upper, midpoint, weight); group `"probability"` with `joint`.
- Produces: R `sobol_sequence()`, `stratify()`, `joint_probability()`; the same in Python.

- [ ] **Step 1: Read the headers and the existing Sobol glue**

`core/include/corehydro/numerics/sampling/{sobol,stratify,stratification_options,stratification_bin}.hpp`,
`core/include/corehydro/numerics/data/probability.hpp`, and the existing `corehydror/src/sobol.cpp`.

The Sobol constructor needs the direction-numbers file path. It is installed at
`corehydror/inst/extdata/new-joe-kuo-6.21201` and located at runtime with
`system.file("extdata", "new-joe-kuo-6.21201", package = "corehydror")`. Find how the Python
package locates the same file (grep `new-joe-kuo` under `corehydropy/`) and keep both mechanisms;
the runner takes the resolved path as an option, so path resolution stays in the wrappers where
it belongs.

- [ ] **Step 2: Write the failing ctest**

Assert that `sobol` returns `n * dimension` values with `dims = {n, dimension}`, that every value
lies in [0, 1), that `skip` moves the stream (point 1 with `skip = 1` equals point 2 with
`skip = 0`), that `stratify` returns bins whose weights sum to 1, and that
`joint_probability` of independent 0.5 and 0.5 is 0.25 while the perfectly-positive form is 0.5.
The ctest can hardcode the direction-file path as `"core/data/new-joe-kuo-6.21201"` relative to the
build directory; check how `core/tests` already reaches that file (grep `new-joe-kuo` under
`core/`) and follow it.

- [ ] **Step 3: Write the two arms, then the wrappers**

```cpp
inline ToolboxResult run_sampling(const std::string& method,
                                  const std::vector<std::vector<double>>& data,
                                  const JsonValue& options) {
    namespace ns = numerics::sampling;
    (void)data;
    if (method == "sobol") {
        // The direction-numbers file ships with each package; the wrapper resolves the path and
        // passes it here, so path lookup stays out of the core.
        int dimension = options.value_or("dimension", 1);
        int n = options.value_or("n", 1);
        int skip = options.value_or("skip", 0);
        ns::SobolSequence sobol(dimension, options.at("path").as_string());
        ToolboxResult r;
        if (skip > 0) sobol.skip_to(skip);
        for (int i = 0; i < n; ++i) {
            std::vector<double> pt = sobol.next_double();
            r.values.insert(r.values.end(), pt.begin(), pt.end());
        }
        r.dims = {n, dimension};
        return r;
    }
    if (method == "stratify") {
        ns::StratificationOptions opts(options.at("lower").as_double(),
                                       options.at("upper").as_double(),
                                       options.at("bins").as_int(),
                                       options.value_or("probability", false));
        std::vector<ns::StratificationBin> bins =
            ns::Stratify::XValues(opts, options.value_or("logarithmic", false));
        ToolboxResult r;
        for (const ns::StratificationBin& b : bins) {
            r.values.push_back(b.lower_bound());
            r.values.push_back(b.upper_bound());
            r.values.push_back(b.midpoint());
            r.values.push_back(b.weight);
        }
        r.names = {"lower", "upper", "midpoint", "weight"};
        r.dims = {static_cast<int>(bins.size()), 4};
        return r;
    }
    throw std::runtime_error("unknown sampling method: " + method);
}

// probability.hpp offers two families: the plain one over a probability vector and a
// dependency type, and the indicator form (a 0/1 flag per component) that additionally accepts a
// correlation matrix and routes to the HPCM path. Both are exposed; `indicators` in the options
// selects the second.
inline ToolboxResult run_probability(const std::string& method,
                                     const std::vector<std::vector<double>>& data,
                                     const JsonValue& options) {
    namespace nd = numerics::data;
    if (method != "joint") throw std::runtime_error("unknown probability method: " + method);
    const std::vector<double>& p = data_at(data, 0, "probability", method);

    std::string dep = options.value_or("dependency", "independent");
    nd::DependencyType type;
    if (dep == "independent") type = nd::DependencyType::Independent;
    else if (dep == "positive") type = nd::DependencyType::PerfectlyPositive;
    else if (dep == "negative") type = nd::DependencyType::PerfectlyNegative;
    else if (dep == "correlation") type = nd::DependencyType::CorrelationMatrix;
    else
        throw std::runtime_error("unknown dependency '" + dep +
                                 "'; expected independent, positive, negative, or correlation");

    if (data.size() < 2) {
        if (type == nd::DependencyType::CorrelationMatrix)
            throw std::runtime_error(
                "dependency 'correlation' needs an indicator vector and a correlation matrix");
        return scalar(nd::joint_probability(p, type));
    }

    // Indicator form: data[1] is the 0/1 indicator vector, data[2] the flattened correlation
    // matrix when there is one.
    const std::vector<double>& ind_d = data[1];
    std::vector<int> indicators(ind_d.size());
    for (std::size_t i = 0; i < ind_d.size(); ++i) indicators[i] = static_cast<int>(ind_d[i]);
    if (data.size() < 3) return scalar(nd::joint_probability(p, indicators, nullptr, type));

    std::size_t n = p.size();
    if (data[2].size() != n * n)
        throw std::runtime_error("the correlation matrix must be " + std::to_string(n) + " by " +
                                 std::to_string(n));
    nd::Matrix2D c(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) c[i][j] = data[2][i * n + j];
    return scalar(nd::joint_probability(p, indicators, &c, type));
}
```

`Matrix2D` is the alias `probability.hpp` already uses; confirm its namespace before compiling.
The R verb is therefore
`joint_probability(p, dependency = c("independent", "positive", "negative", "correlation"), indicators = NULL, correlation = NULL)`,
with `correlation` requiring `indicators`, and the Python twin matches.

Then delete `ch_sobol_generate_` and `ch_sobol_skip_to_` from `corehydror/src/sobol.cpp` and the
matching pybind entries, and point the R and Python fixture runners' Latin-hypercube and Sobol
dispatch at the toolbox runner. Verify nothing else calls them first:

```bash
grep -rn "ch_sobol_generate_\|ch_sobol_skip_to_\|sobol_generate\|sobol_skip_to" corehydror corehydropy
```

If the file ends up empty, delete it and drop it from `corehydropy/CMakeLists.txt`.

- [ ] **Step 4: Fixtures, emitter, tests, commit**

`fixtures/toolbox/sampling.json` from `Test_SobolSequence.cs` and `Test_Stratification.cs`;
`fixtures/toolbox/joint_probability.json` curated with the emitter's dump mode against the real
`Probability` class (upstream has no dedicated test file for it, so say so in the fixture's
`reference` field). Add both arms to `ToolboxDispatch`.

```bash
git add core corehydror corehydropy fixtures tools
git commit -m "feat: expose Sobol, stratification and joint probability in R and Python"
```

---

### Task 7: Link functions and trend evaluation

**Files:**
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (the `link` and `trend` arms)
- Modify: `corehydror/R/toolbox.R`, `corehydropy/src/corehydropy/toolbox.py`, `corehydror/NAMESPACE`, `corehydropy/src/corehydropy/__init__.py`
- Create: `fixtures/toolbox/link_functions.json`, `fixtures/toolbox/trend_functions.json`
- Modify: `tools/oracle_emitter/Program.cs`, `core/tests/test_toolbox_runner.cpp`
- Test: `corehydror/tests/testthat/test-toolbox.R`, `corehydropy/tests/test_toolbox.py`

**Interfaces:**
- Produces: group `"link"` with `link`, `inverse_link`, `d_link`, `names`; group `"trend"` with `predict`, `parameters`, `names`.
- Produces: R `link_function()`, `link()`, `link_inverse()`, `link_derivative()`, `link_names()`, `trend_predict()`, `trend_parameters()`, `trend_names()`; the same in Python.

- [ ] **Step 1: Read the factories and note every constructor argument**

`core/include/corehydro/numerics/functions/link_function_factory.hpp` and
`link_function_type.hpp` (the 7 Numerics links) and
`core/include/corehydro/models/link_functions/best_fit_link_function_factory.hpp` plus the six
BestFit link headers. Write down each link's constructor arguments. Known so far:
`ASinHLink(gamma0, scale, epsilon = 0, delta = 1)`, `SESLink(a, use_adaptive_lambda = true,
parent_indicator = 0)`, `LogSESLink(sigma0, a = 1, lambda = 0.2)`,
`LogASinHLink(sigma0, log_scale, epsilon = 0, delta = 1)`, and `CenteredLink(inner, mu0, ...)`,
which nests another link. Confirm each against its header; the defaults above are read from the
declarations and must not be guessed at.

- [ ] **Step 2: Define the link spec and write the failing ctest**

The spec block, parsed by the runner:

```json
{"type": "ASinH", "parameters": {"gamma0": 1.0, "scale": 2.0},
 "inner": {"type": "Log"}}
```

`type` names either factory's enum (the runner tries the Numerics factory first and falls back to
the BestFit one, exactly as `BestFitLinkFunctionFactory` does for its own default case);
`parameters` is a named object, absent for the parameterless links; `inner` is present only for
`Centered`.

Write the ctest asserting `inverse_link(link(x))` round-trips to 1e-12 for every one of the
thirteen types at a value inside its domain, and that `d_link` at that value is finite.

- [ ] **Step 3: Write the two arms**

```cpp
// Builds a link from the options spec. The Numerics factory is tried first and the BestFit one
// second, mirroring how BestFitLinkFunctionFactory falls through to the Numerics factory for a
// type it does not own. `parameters` is a named object because the six parameterized links take
// different arguments and a positional array would be unreadable at the call site.
inline std::unique_ptr<numerics::functions::ILinkFunction> build_link(const JsonValue& spec) {
    const std::string& type = spec.at("type").as_string();
    const JsonValue* p = spec.contains("parameters") ? &spec.at("parameters") : nullptr;
    auto opt = [&](const char* key, double dflt) {
        return p && p->contains(key) ? p->at(key).as_double() : dflt;
    };
    if (type == "Identity" || type == "Log" || type == "Logit" || type == "Probit" ||
        type == "ComplementaryLogLog" || type == "FisherZ")
        return numerics::functions::LinkFunctionFactory::create(
            numerics::functions::parse_link_function_type(type));
    if (type == "YeoJohnson")
        return std::make_unique<numerics::functions::YeoJohnsonLink>(opt("lambda", 1.0));
    if (type == "ASinH")
        return std::make_unique<models::link_functions::ASinHLink>(
            opt("gamma0", 0.0), opt("scale", 1.0), opt("epsilon", 0.0), opt("delta", 1.0));
    if (type == "SES")
        return std::make_unique<models::link_functions::SESLink>(opt("a", 1.0));
    if (type == "LogSES")
        return std::make_unique<models::link_functions::LogSESLink>(
            opt("sigma0", 1.0), opt("a", 1.0), opt("lambda", 0.2));
    if (type == "LogASinH")
        return std::make_unique<models::link_functions::LogASinHLink>(
            opt("sigma0", 1.0), opt("log_scale", 1.0), opt("epsilon", 0.0), opt("delta", 1.0));
    if (type == "Centered") {
        if (!spec.contains("inner"))
            throw std::runtime_error("link type 'Centered' needs an 'inner' link spec");
        return std::make_unique<models::link_functions::CenteredLink>(
            build_link(spec.at("inner")), opt("mu0", 0.0));
    }
    throw std::runtime_error("unknown link type: " + type);
}

inline ToolboxResult run_link(const std::string& method,
                              const std::vector<std::vector<double>>& data,
                              const JsonValue& options) {
    if (!options.contains("link"))
        throw std::runtime_error("toolbox group 'link' needs a 'link' spec in its options");
    std::unique_ptr<numerics::functions::ILinkFunction> l = build_link(options.at("link"));
    const std::vector<double>& x = data_at(data, 0, "link", method);
    ToolboxResult r;
    for (double v : x) {
        if (method == "link") r.values.push_back(l->link(v));
        else if (method == "inverse_link") r.values.push_back(l->inverse_link(v));
        else if (method == "d_link") r.values.push_back(l->d_link(v));
        else throw std::runtime_error("unknown link method: " + method);
    }
    return r;
}

inline ToolboxResult run_trend(const std::string& method,
                               const std::vector<std::vector<double>>& data,
                               const JsonValue& options) {
    if (!options.contains("trend"))
        throw std::runtime_error("toolbox group 'trend' needs a 'trend' spec in its options");
    std::unique_ptr<models::trend_functions::ITrendModel> t =
        models::spec::build_spec_trend(options.at("trend"));
    ToolboxResult r;
    if (method == "predict") {
        for (double i : data_at(data, 0, "trend", method))
            r.values.push_back(t->predict(static_cast<int>(i)));
        return r;
    }
    if (method == "parameters") {
        for (const models::ModelParameter& p : t->parameters()) {
            r.values.push_back(p.value());
            r.names.push_back(p.name());
        }
        return r;
    }
    throw std::runtime_error("unknown trend method: " + method);
}
```

Three things to settle against the headers before this compiles:

1. `LinkFunctionFactory::create` and any string-to-enum helper. If no
   `parse_link_function_type` exists, write one beside `build_link` listing the seven names.
2. Each parameterized link's real constructor argument list and defaults. The values above are
   read from the declarations but verify every one; a wrong default silently changes results.
3. `build_spec_trend`. `model_spec.hpp:288` parses the `trends` array inline. Extract the
   single-entry body into `inline std::unique_ptr<ITrendModel> build_spec_trend(const JsonValue&)`
   in that same file and call it from both the loop and here, so there is one trend parser. Also
   confirm `ModelParameter`'s value and name accessors are spelled `value()` and `name()`.

- [ ] **Step 4: R and Python verbs**

```r
kLinkTypes <- c("Identity", "Log", "Logit", "Probit", "ComplementaryLogLog", "FisherZ",
                "YeoJohnson", "ASinH", "SES", "LogSES", "LogASinH", "Centered")

#' Construct a link function
#'
#' Mirrors the seven Numerics link functions and the six RMC.BestFit ones. Six of the thirteen
#' take construction parameters, passed through `...` by name.
#'
#' @param type one of `link_names()`.
#' @param ... named construction parameters for the parameterized links, for example
#'   `gamma0` and `scale` for `"ASinH"`, or `lambda` for `"YeoJohnson"`.
#' @param inner a `corehydro_link` wrapped by `"Centered"`, ignored for every other type.
#' @return a `corehydro_link` spec list.
#' @examples
#' l <- link_function("Log")
#' link(l, c(1, 10, 100))
#' link_inverse(l, c(0, 1, 2))
#' @export
link_function <- function(type, ..., inner = NULL) {
  if (!type %in% kLinkTypes) {
    stop(sprintf("unknown link type \"%s\". Available: %s", type,
                 paste(kLinkTypes, collapse = ", ")), call. = FALSE)
  }
  params <- list(...)
  if (identical(type, "Centered") && is.null(inner)) {
    stop("link type \"Centered\" needs an `inner` link", call. = FALSE)
  }
  structure(list(type = type, parameters = if (length(params) == 0L) NULL else params,
                 inner = inner), class = "corehydro_link")
}

#' Evaluate a link function
#'
#' @param l a `corehydro_link` from [link_function()].
#' @param x,eta numeric vectors on the data and linear-predictor scales.
#' @return a numeric vector the same length as the input.
#' @examples
#' l <- link_function("Logit")
#' link(l, c(0.1, 0.5, 0.9))
#' @export
link <- function(l, x) link_eval(l, "link", x)

#' @rdname link
#' @export
link_inverse <- function(l, eta) link_eval(l, "inverse_link", eta)

#' @rdname link
#' @export
link_derivative <- function(l, x) link_eval(l, "d_link", x)

#' @rdname link
#' @export
link_names <- function() kLinkTypes

link_eval <- function(l, method, v) {
  if (!inherits(l, "corehydro_link")) {
    stop("`l` must be a corehydro_link object; create one with link_function()", call. = FALSE)
  }
  toolbox_run("link", method, list(v), list(link = unclass_link(l)))$values
}

# Internal: strip the class so to_spec_json() emits a plain nested object.
unclass_link <- function(l) {
  spec <- list(type = l$type, parameters = l$parameters)
  if (!is.null(l$inner)) spec$inner <- unclass_link(l$inner)
  spec
}

#' @export
print.corehydro_link <- function(x, ...) {
  cat(sprintf("<corehydro_link> %s\n", x$type))
  invisible(x)
}
```

`trend_predict(tr, index)` takes the `corehydro_trend` object `trend()` already returns and a
1-based index vector, converting to the 0-based index the spec wants exactly as `mv_indices()`
does in `R/mvdist.R`. `trend_parameters(tr)` returns a named numeric vector and `trend_names()`
the eleven `TrendModelType` names. Write the Python twins and add
`S3method(print, corehydro_link)` plus every export to `NAMESPACE`.

- [ ] **Step 5: Fixtures, emitter, tests, commit**

Curate `fixtures/toolbox/link_functions.json` and `fixtures/toolbox/trend_functions.json` against
the real C# factories with the emitter's dump mode; check whether upstream has test files for
either (`ls upstream/Numerics/Test_Numerics/Functions upstream/RMC-BestFit/**/Test*Link*`) and
prefer transcribed literals where they exist.

```bash
git add core corehydror corehydropy fixtures tools
git commit -m "feat: expose link functions and trend evaluation in R and Python"
```

---

### Task 8: The optimizers

**Files:**
- Create: `core/include/corehydro/numerics/support/optimizer_runner.hpp`, `core/tests/test_optimizer_runner.cpp`
- Modify: `corehydror/src/toolbox.cpp`, `corehydropy/src/bindings/toolbox.cpp`
- Create: `corehydror/R/optim.R`, `corehydropy/src/corehydropy/optim.py`
- Create: `fixtures/toolbox/optimizers.json`
- Modify: `core/CMakeLists.txt`, `core/tests/test_fixtures.cpp`, `corehydror/tests/testthat/test-fixtures.R`, `corehydropy/tests/test_fixtures.py`, `tools/oracle_emitter/Program.cs`, `corehydror/NAMESPACE`, `corehydropy/src/corehydropy/__init__.py`
- Test: `core/tests/test_optimizer_runner.cpp`, `corehydror/tests/testthat/test-optim.R`, `corehydropy/tests/test_optim.py`

**Interfaces:**
- Produces: `corehydro::numerics::support::Objective = std::function<double(const std::vector<double>&)>`, `OptimResult { std::vector<double> parameters; double value; int iterations; int function_evaluations; std::string status; std::vector<double> hessian; std::vector<int> hessian_dims; }`, `OptimResult run_optimizer(const std::string& spec_json, const Objective& objective)`, and `Objective builtin_objective(const std::string& name)`.
- Produces: R `optim_minimize()`, `optim_maximize()`, `print.corehydro_optim`; Python `optim_minimize()`, `optim_maximize()`, `OptimResult`.

- [ ] **Step 1: Read the optimizer headers and the C# test functions**

`core/include/corehydro/numerics/math/optimization/{differential_evolution,bfgs,powell,mlsl,nelder_mead,brent_search}.hpp`
and `support/optimizer.hpp`. Note that `minimize()` and `maximize()` end in a catch-all that
records `Status = Failure` and swallows the exception when `report_failure` is false, and that
`nelder_mead.hpp` and `brent_search.hpp` deliberately do NOT derive from `Optimizer` (see the
header comment in `support/optimizer.hpp`), so their construction and result extraction differ.
Handle that difference explicitly in the runner rather than forcing a common base.

Read `upstream/Numerics/Test_Numerics/Mathematics/Optimization/TestFunctions.cs` and transcribe
the fourteen functions the six ported optimizers' tests call: `FX`, `FXYZ`, `DeJong`,
`Rosenbrock`, `SumOfPowerFunctions`, `McCormick`, `Matyas`, `Booth`, `Beale`, `ThreeHumpCamel`,
`Rastrigin`, `GoldsteinPrice`, `Eggholder`, `Ackley`.

- [ ] **Step 2: Write the failing ctest**

Create `core/tests/test_optimizer_runner.cpp`:

```cpp
// ctest for the optimizer runner: dispatch over the built-in objectives, and the callback
// abort path that keeps a throwing objective from being swallowed by Optimizer::minimize().
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/support/optimizer_runner.hpp"
#include "test_support.hpp"

namespace tb = corehydro::numerics::support;

int main() {
    // De Jong (sphere) has its minimum at the origin; every method should get close.
    for (const char* method : {"de", "bfgs", "powell", "nelder_mead"}) {
        std::string spec = std::string("{\"method\":\"") + method +
                           "\",\"lower\":[-5,-5],\"upper\":[5,5],\"initial\":[1,1],\"seed\":12345}";
        tb::OptimResult r = tb::run_optimizer(spec, tb::builtin_objective("DeJong"));
        CHTEST_EXPECT_EQ(r.parameters.size(), std::size_t{2});
        CHTEST_EXPECT_TRUE(r.value < 1e-6);
    }

    // A throwing objective surfaces as that exception, not as a silent Failure status.
    bool rethrown = false;
    try {
        tb::run_optimizer("{\"method\":\"de\",\"lower\":[-5,-5],\"upper\":[5,5],\"seed\":1}",
                          [](const std::vector<double>&) -> double {
                              throw std::runtime_error("objective exploded");
                          });
    } catch (const std::exception& e) {
        rethrown = std::string(e.what()).find("objective exploded") != std::string::npos;
    }
    CHTEST_EXPECT_TRUE(rethrown);

    // A seeded DE run is reproducible.
    std::string spec = "{\"method\":\"de\",\"lower\":[-5,-5],\"upper\":[5,5],\"seed\":777}";
    auto a = tb::run_optimizer(spec, tb::builtin_objective("Rosenbrock"));
    auto b = tb::run_optimizer(spec, tb::builtin_objective("Rosenbrock"));
    CHTEST_EXPECT_NEAR(a.value, b.value, 0.0);
    CHTEST_EXPECT_NEAR(a.parameters[0], b.parameters[0], 0.0);

    return chtest::summary();
}
```

Add `test_optimizer_runner` to `BF_TESTS`. Run the build. Expected: FAIL, header not found.

- [ ] **Step 3: Write the runner**

Create `core/include/corehydro/numerics/support/optimizer_runner.hpp`. The spec grammar:

```json
{"method": "de|bfgs|powell|mlsl|nelder_mead|brent",
 "lower": [...], "upper": [...], "initial": [...],
 "maximize": false, "seed": 12345,
 "objective": "DeJong",
 "control": {"max_iterations": 1000, "max_function_evaluations": 100000,
             "absolute_tolerance": 1e-8, "relative_tolerance": 1e-8,
             "report_failure": true, "compute_hessian": false,
             "population_size": 30}}
```

The heart of the file is the guarded objective:

```cpp
// Wraps the caller's objective so a host-language exception (an R error arriving as a
// cpp11::unwind_exception, a Python error as pybind11::error_already_set) cannot be swallowed by
// Optimizer::minimize()'s catch-all. The first throw is stored, an abort flag latches, and every
// later evaluation short-circuits without re-entering the host so the run winds down in a few
// evaluations instead of the full budget. run_optimizer rethrows once the optimizer returns.
class GuardedObjective {
   public:
    GuardedObjective(Objective fn, bool maximize) : fn_(std::move(fn)), maximize_(maximize) {}

    double operator()(const std::vector<double>& p) {
        if (aborted_) return sentinel();
        try {
            double v = fn_(p);
            return v;
        } catch (...) {
            error_ = std::current_exception();
            aborted_ = true;
            return sentinel();
        }
    }

    bool aborted() const { return aborted_; }
    void rethrow_if_aborted() const { if (error_) std::rethrow_exception(error_); }

   private:
    // The optimizer is always driven through minimize(); a maximize request negates the
    // objective instead, so the sentinel is the same worst-possible value either way.
    double sentinel() const { return std::numeric_limits<double>::infinity(); }

    Objective fn_;
    bool maximize_;
    bool aborted_ = false;
    std::exception_ptr error_;
};
```

Decide one thing explicitly and write it in a comment: whether a maximize request calls the
optimizer's `maximize()` or negates the objective and calls `minimize()`. Calling `maximize()`
matches C# and keeps the seeded stream identical to a C# run, so prefer it and make the sentinel
sign follow the direction. Adjust `sentinel()` to return negative infinity when `maximize_` is
true, and say why in the comment.

`builtin_objective(name)` returns the transcribed test function by name and throws listing the
fourteen accepted names. Each transcription carries a
`// transcribed from: Test_Numerics/Mathematics/Optimization/TestFunctions.cs @ 2a0357a` comment.
These exist for the fixtures and the emitter, not for users.

Run the ctest. Expected: PASS.

- [ ] **Step 4: Commit the runner**

```bash
git add core/include/corehydro/numerics/support/optimizer_runner.hpp core/tests/test_optimizer_runner.cpp core/CMakeLists.txt
git commit -m "feat(core): add the optimizer runner with a guarded objective callback"
```

- [ ] **Step 5: Add the glue**

In `corehydror/src/toolbox.cpp`:

```cpp
[[cpp11::register]]
list ch_optim_run_(std::string spec_json, function objective) {
    tb::OptimResult r = tb::run_optimizer(spec_json, [&](const std::vector<double>& p) -> double {
        writable::doubles par(static_cast<R_xlen_t>(p.size()));
        for (std::size_t i = 0; i < p.size(); ++i) par[static_cast<R_xlen_t>(i)] = p[i];
        sexp out = objective(par);
        doubles v(out);
        if (v.size() != 1)
            throw std::runtime_error(
                "the objective must return a single number; got a value of length " +
                std::to_string(static_cast<long long>(v.size())));
        return v[0];
    });
    writable::doubles params(static_cast<R_xlen_t>(r.parameters.size()));
    for (std::size_t i = 0; i < r.parameters.size(); ++i)
        params[static_cast<R_xlen_t>(i)] = r.parameters[i];
    writable::doubles hess(static_cast<R_xlen_t>(r.hessian.size()));
    for (std::size_t i = 0; i < r.hessian.size(); ++i)
        hess[static_cast<R_xlen_t>(i)] = r.hessian[i];
    writable::integers hdims(static_cast<R_xlen_t>(r.hessian_dims.size()));
    for (std::size_t i = 0; i < r.hessian_dims.size(); ++i)
        hdims[static_cast<R_xlen_t>(i)] = r.hessian_dims[i];
    return writable::list({"parameters"_nm = params,
                           "value"_nm = writable::doubles({r.value}),
                           "iterations"_nm = writable::integers({r.iterations}),
                           "function_evaluations"_nm = writable::integers({r.function_evaluations}),
                           "status"_nm = writable::strings({r.status}),
                           "hessian"_nm = hess,
                           "hessian_dims"_nm = hdims});
}
```

A non-numeric return raises inside the `doubles v(out)` conversion, which cpp11 turns into an R
error; that error travels back out through the guard and is rethrown, which is the behavior the
test in step 7 asserts.

In `corehydropy/src/bindings/toolbox.cpp`, the twin using a `py::function` and
`py::cast<double>`, raising `std::runtime_error` with the same message when the return is not a
scalar.

- [ ] **Step 6: Write the R and Python verbs**

Create `corehydror/R/optim.R`:

```r
#' Minimize or maximize a user-written objective
#'
#' Runs one of the six ported Numerics optimizers over an R function. The optimizer's random
#' number generator lives in C++, so a seeded run reproduces exactly, and reproduces identically
#' in corehydropy.
#'
#' @param objective a function taking a numeric parameter vector and returning a single number.
#' @param lower,upper numeric vectors of parameter bounds, the same length as the parameter
#'   vector. Required for `"de"`, `"mlsl"` and `"brent"`.
#' @param initial optional numeric vector of starting values. Required for `"bfgs"`, `"powell"`
#'   and `"nelder_mead"`.
#' @param method one of `"de"` (differential evolution, the default), `"bfgs"`, `"powell"`,
#'   `"mlsl"`, `"nelder_mead"`, or `"brent"`.
#' @param seed optional integer seed for the stochastic methods (`"de"`, `"mlsl"`).
#' @param control a named list of optimizer settings: `max_iterations`,
#'   `max_function_evaluations`, `absolute_tolerance`, `relative_tolerance`, `population_size`,
#'   `compute_hessian`, and `report_failure` (default `TRUE`, which surfaces a configuration
#'   failure as an R error rather than returning a failed status quietly).
#' @return a `corehydro_optim` list with `parameters`, `value`, `iterations`,
#'   `function_evaluations`, `status`, and `hessian` when requested.
#' @examples
#' rosenbrock <- function(p) (1 - p[1])^2 + 100 * (p[2] - p[1]^2)^2
#' fit <- optim_minimize(rosenbrock, lower = c(-5, -5), upper = c(5, 5), seed = 42)
#' round(fit$parameters, 3)
#' @export
optim_minimize <- function(objective, lower = NULL, upper = NULL, initial = NULL,
                           method = c("de", "bfgs", "powell", "mlsl", "nelder_mead", "brent"),
                           seed = NULL, control = list()) {
  optim_run(objective, lower, upper, initial, match.arg(method), seed, control,
            maximize = FALSE)
}

#' @rdname optim_minimize
#' @export
optim_maximize <- function(objective, lower = NULL, upper = NULL, initial = NULL,
                           method = c("de", "bfgs", "powell", "mlsl", "nelder_mead", "brent"),
                           seed = NULL, control = list()) {
  optim_run(objective, lower, upper, initial, match.arg(method), seed, control,
            maximize = TRUE)
}

kOptimControl <- c("max_iterations", "max_function_evaluations", "absolute_tolerance",
                   "relative_tolerance", "population_size", "compute_hessian", "report_failure")
kOptimNeedsBounds <- c("de", "mlsl", "brent")
kOptimNeedsInitial <- c("bfgs", "powell", "nelder_mead")

# Internal: validate everything R-side, then make one call. Both verbs share this so their
# messages and defaults can never drift apart.
optim_run <- function(objective, lower, upper, initial, method, seed, control, maximize) {
  if (!is.function(objective)) {
    stop("`objective` must be a function taking a numeric vector and returning one number",
         call. = FALSE)
  }
  if (method %in% kOptimNeedsBounds && (is.null(lower) || is.null(upper))) {
    stop(sprintf("method \"%s\" needs `lower` and `upper` bounds", method), call. = FALSE)
  }
  if (method %in% kOptimNeedsInitial && is.null(initial)) {
    stop(sprintf("method \"%s\" needs `initial` starting values", method), call. = FALSE)
  }
  if (!is.null(lower) && !is.null(upper)) {
    if (length(lower) != length(upper)) {
      stop(sprintf("`lower` and `upper` must have the same length; got %d and %d",
                   length(lower), length(upper)), call. = FALSE)
    }
    if (any(lower >= upper)) {
      stop("every `lower` bound must be below its `upper` bound", call. = FALSE)
    }
  }
  unknown <- setdiff(names(control), kOptimControl)
  if (length(unknown) > 0L) {
    stop(sprintf("unknown control name(s): %s. Available: %s",
                 paste(unknown, collapse = ", "), paste(kOptimControl, collapse = ", ")),
         call. = FALSE)
  }
  spec <- to_spec_json(list(
    method = method, maximize = maximize,
    lower = if (is.null(lower)) NULL else spec_array(as.double(lower)),
    upper = if (is.null(upper)) NULL else spec_array(as.double(upper)),
    initial = if (is.null(initial)) NULL else spec_array(as.double(initial)),
    seed = if (is.null(seed)) NULL else as.integer(seed),
    control = if (length(control) == 0L) NULL else control
  ))
  r <- ch_optim_run_(spec, objective)
  if (length(r$hessian) > 0L) {
    r$hessian <- matrix(r$hessian, nrow = r$hessian_dims[[1]], ncol = r$hessian_dims[[2]],
                        byrow = TRUE)
  } else {
    r$hessian <- NULL
  }
  r$hessian_dims <- NULL
  structure(r, class = "corehydro_optim")
}

#' @export
print.corehydro_optim <- function(x, ...) {
  cat(sprintf("<corehydro_optim> %s after %d iterations (%d evaluations)\n",
              x$status, x$iterations, x$function_evaluations))
  cat(sprintf("  value: %.8g\n", x$value))
  cat(sprintf("  parameters: %s\n", paste(format(x$parameters, digits = 6), collapse = ", ")))
  invisible(x)
}
```

Add `export(optim_minimize)`, `export(optim_maximize)`, and `S3method(print, corehydro_optim)` to
`NAMESPACE`. Write the Python twins in `corehydropy/src/corehydropy/optim.py` with identical
validation, identical messages, and an `OptimResult` wrapper carrying the same fields plus a
`__repr__`, and re-export both from `__init__.py`.

- [ ] **Step 7: Write the behavioral tests**

`corehydror/tests/testthat/test-optim.R` and `corehydropy/tests/test_optim.py`:

```r
test_that("a seeded DE run reproduces exactly", {
  f <- function(p) sum(p^2)
  a <- optim_minimize(f, lower = c(-5, -5), upper = c(5, 5), seed = 99)
  b <- optim_minimize(f, lower = c(-5, -5), upper = c(5, 5), seed = 99)
  expect_identical(a$parameters, b$parameters)
  expect_identical(a$value, b$value)
})

test_that("an error inside the objective reaches the caller intact", {
  f <- function(p) stop("boom in the objective")
  expect_error(optim_minimize(f, lower = c(-1, -1), upper = c(1, 1), seed = 1),
               "boom in the objective")
})

test_that("an objective returning the wrong shape is rejected by message", {
  expect_error(optim_minimize(function(p) c(1, 2), lower = c(-1, -1), upper = c(1, 1)),
               "single number")
  expect_error(optim_minimize(function(p) "nope", lower = c(-1, -1), upper = c(1, 1)))
})

test_that("optim_maximize finds the peak of a concave objective", {
  f <- function(p) -((p[1] - 2)^2 + (p[2] + 1)^2)
  fit <- optim_maximize(f, lower = c(-10, -10), upper = c(10, 10), seed = 7)
  expect_equal(fit$parameters, c(2, -1), tolerance = 1e-4)
})

test_that("a method that needs bounds says so when they are missing", {
  expect_error(optim_minimize(function(p) sum(p^2), method = "de"), "lower")
})
```

After the error tests pass, confirm R is still healthy in the same session (the whole point of the
guard): the suite continuing to run further tests after the throwing-objective test is that
confirmation, so put the throwing test early in the file rather than last.

- [ ] **Step 8: Fixtures and the emitter**

Create `fixtures/toolbox/optimizers.json` with `"kind": "optimizer"`, cases naming a built-in
objective, a method, bounds, and a seed, asserting the converged value and parameters against the
literals in the six upstream optimizer test files. Wire the `optimizer` kind into all four
runners: C++ calls `run_optimizer(spec, builtin_objective(name))`; R and Python call
`ch_optim_run_` / `optim_run` with a wrapper around the same objective **implemented in the
fixture runner** so the callback path itself is exercised by every fixture case; the emitter
drives the real C# optimizer with the C# `TestFunctions` delegate.

Run:
```bash
ctest --test-dir core/build && Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy && pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py
```
Expected: all green, `failed 0`.

- [ ] **Step 9: Commit**

```bash
git add core corehydror corehydropy fixtures tools
git commit -m "feat: expose the six ported optimizers over an R and Python objective"
```

---

### Task 9: Cross-language digest, docs, examples, release

**Files:**
- Create: `fixtures/toolbox/toolbox_cross_language.json`
- Create: `site/examples/12-model-evaluation/{python.ipynb,r.qmd}`, `site/examples/13-custom-objective/{python.ipynb,r.qmd}`
- Modify: `corehydror/_pkgdown.yml`, `site/_quarto.yml`, `site/examples/index.qmd` (or whatever indexes the examples; check how example 11 is listed)
- Modify: `corehydror/DESCRIPTION`, `corehydror/NEWS.md`, `CHANGELOG.md`, `corehydropy/pyproject.toml`, `core/include/corehydro/version.hpp`
- Modify: `.claude/CLAUDE.md` (status paragraph)

- [ ] **Step 1: Write the cross-language digest fixture**

Model it on the existing `fixtures/estimation/fit_cross_language.json`: a `short_exact` digest
asserting a handful of values from one seeded DE run over a built-in objective, one Sobol block,
and one stratification, every runner asserting the same digest. Its job is to prove R and Python
agree bit for bit.

Run all four runners. Expected: identical values everywhere.

- [ ] **Step 2: Verify the R and Python surfaces agree by inspection**

```bash
Rscript -e 'cat(sort(getNamespaceExports("corehydror")), sep = "\n")' > /tmp/r_exports.txt
pixi run python -c "import corehydropy; print('\n'.join(sorted(corehydropy.__all__)))" > /tmp/py_exports.txt
diff /tmp/r_exports.txt /tmp/py_exports.txt
```

Differences are expected where the languages differ idiomatically (Python folds verbs into
classes), but every function added by this phase must appear on both sides with the same name.
Fix any that do not.

- [ ] **Step 3: Add every export to both documentation indexes**

Add the new sections to `corehydror/_pkgdown.yml`: Goodness of fit, Statistics, Interpolation and
regression, Sampling utilities, Optimizers. Add the matching `quartodoc.sections` entries to
`site/_quarto.yml`.

```bash
pixi run docs 2>&1 | tail -20
```
Expected: build succeeds. pkgdown fails loudly on any export missing from the reference index, so
a clean build is the check.

- [ ] **Step 4: Write example pair 12, model evaluation**

`site/examples/12-model-evaluation/`: fit several candidate distributions to one flood series,
rank them with `goodness_of_fit()`, `aic_weights()`, and `gof_test()`, and plot the fitted curves
against plotting positions. The Python half is a Jupyter notebook committed WITH outputs
(`jupyter nbconvert --to notebook --execute --inplace python.ipynb`); the R half is Quarto with
`freeze: auto`, rendered locally with `site/_freeze/` committed. Both end in an executable
reproduction check comparing a computed value against a literal at 1e-15 relative tolerance,
matching the existing examples.

- [ ] **Step 5: Write example pair 13, a custom objective**

`site/examples/13-custom-objective/`: write a small likelihood in R and in Python, optimize it
with `optim_minimize(method = "de", seed = ...)` and again with `"bfgs"`, show the two agree, and
show the seeded DE result is identical across the two languages. End with the same executable
reproduction check.

- [ ] **Step 6: Bump the version to 0.6.0**

Update `corehydror/DESCRIPTION`, `corehydropy/pyproject.toml`, and `core/include/corehydro/version.hpp`
(grep `0.5.0` across the repo to catch every stamp). Write the `CHANGELOG.md` and
`corehydror/NEWS.md` entries: the new surface by group, the three bugs or gaps this phase closed
(the unported `Autocorrelation.cs`, the unrecorded `Correlation` matrix severance, and whatever
else surfaced during implementation), and the counts from the final verification run.

- [ ] **Step 7: Full verification sweep**

```bash
cmake --build core/build && ctest --test-dir core/build 2>&1 | tail -5
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")' 2>&1 | tail -5
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q 2>&1 | tail -5
python3 tools/verify_oracles.py 2>&1 | tail -5
R CMD build corehydror && R CMD check --as-cran corehydror_0.6.0.tar.gz 2>&1 | tail -20
```

Record the exact numbers. Expected: ctest all pass, testthat 0 failures, pytest 0 failures,
oracle gate `failed 0`, and `R CMD check` at three NOTEs with no WARNING (the CRAN-incoming
non-FOSS-license note, the long-path note listing vendored core headers, and the local HTML-tidy
note). Any fourth NOTE or any WARNING is a regression to fix before finishing.

- [ ] **Step 8: Update the repository status notes**

Update the Status section of `.claude/CLAUDE.md` with a paragraph for this phase in the voice of
the existing ones: what shipped, the final counts from step 7, what was found and fixed, and the
severances. Add the toolbox runner and the optimizer runner to the layout section.

- [ ] **Step 9: Commit and hand off**

```bash
git add -A
git commit -m "feat: the numerics toolbox layer for R and Python (v0.6.0)"
```

Do not push and do not open a PR until asked. When asked, use the
superpowers:finishing-a-development-branch skill.

---

## Verification checklist

Run before declaring the phase done. Every line needs real output, not an assumption.

- [ ] `ctest --test-dir core/build` all pass, including the three new test targets
- [ ] `testthat::test_local("corehydror")` 0 failures
- [ ] `pytest corehydropy/tests -q` 0 failures
- [ ] `python3 tools/verify_oracles.py` reports `failed 0`
- [ ] `R CMD check --as-cran` at three NOTEs, no WARNING
- [ ] `pixi run docs` builds
- [ ] every new export appears in `corehydror/NAMESPACE`, `corehydror/_pkgdown.yml`, and `site/_quarto.yml`
- [ ] the cross-language digest fixture asserts identical values in R and Python
- [ ] no fixture carries an expected value computed by this repo's own C++
