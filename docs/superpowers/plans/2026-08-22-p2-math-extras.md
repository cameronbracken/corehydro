# P2 Math Extras Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the remaining Numerics math extras -- root finding (Bisection, NewtonRaphson,
Secant), the seven integration classes plus the deferred Integration.cs statics, RungeKutta,
CubicSpline and Polynomial interpolation, QRDecomposition and GaussJordanElimination, Debye and
Evaluate, and the Functions layer (IUnivariateFunction, LinearFunction, PowerFunction) -- validated
by fixtures and the dotnet oracle gate, exposed in R and Python, documented, shipped as branch
`port-math-extras` at v0.9.0.

**Architecture:** Callable-driven classes (root finders, integrators, the ODE solver) join the
callback `math` group as new method names beside the existing `root_find`/`quadrature`, with new
`CallbackSet` members for the new callback shapes (a second scalar for Newton's derivative, a
two-argument scalar for f(x,y)/f(t,y), a vector+weight scalar for Vegas, and the vector-to-vector
plus vector-to-matrix pair for the multivariate Newton system). Data-driven classes join the
toolbox: CubicSpline/Polynomial as `interpolation` group methods, and three new one-header groups
`linalg` (QR, Gauss-Jordan), `special` (Debye, Evaluate), and `functions` (Linear/Power function
evaluation). Every class also gets its C# unit tests transcribed 1:1 into ctest suites (the
C#-test-literal oracle standard the analysis suites use), and every runner-reachable value is
pinned through the four-runner fixture path with dotnet-gate reproduction.

**Tech Stack:** C++17 header-only core, cpp11 (R), pybind11 (Python), nlohmann/json fixture
runner, dotnet 10 oracle emitter.

## Global Constraints

- Every ported file carries `// ported from: <path> @ 2a0357a` and mirrors the C# class/method
  layout line-for-line where possible.
- Upstream paths contain spaces (`Numerics/Mathematics/Root Finding/`, `.../Linear Algebra/`,
  `.../Special Functions/`, `.../ODE Solvers/`) -- quote them in every shell command.
- Oracle values live ONLY in `fixtures/*.json`; ctest suites transcribing C# test methods carry
  C#-test-literal assertions (the established second oracle class); never invent expected values.
- Every new fixture case gets a dotnet emitter driver; `python3 tools/verify_oracles.py` must end
  `0 failed`. Every new fixture-catalog callback is defined in ALL FOUR catalogs (C++
  `core/tests/fixture_callback_catalog.cpp`, R `test-fixtures.R` `callback_fixture_function`,
  Python `test_fixtures.py` `_callback_fixture_function`, C# `Program.cs` `Callback*Function`) as
  plain arithmetic with explicit loops -- no `sum()`/`mean()`/LINQ -- because the C++ catalog TU is
  compiled `-ffp-contract=off` and stands in for host-language arithmetic.
- Every seeded stream (Monte Carlo, Miser, Vegas) gets cross-language digest cases in
  `fixtures/callback/callback_cross_language.json` at ZERO tolerance.
- No `M_PI` (use `corehydro::numerics::kPi`); no namespace aliases named `gamma` or `stat`; no
  file-local `const` used implicitly in a capture-less lambda (MSVC C3493 -- use file-scope
  `constexpr`); no new external C++ dependencies.
- After any core class-layout change: `R CMD INSTALL --preclean corehydror`.
- After editing any `corehydror/src/*.cpp`: `Rscript -e 'cpp11::cpp_register("corehydror")'`.
- pytest reads fixtures materialized by pip; re-run
  `pixi run python -m pip install --force-reinstall --no-deps ./corehydropy` after fixture edits.
- P2 touches only Numerics sources (clean UTF-8; grep is fine). Read files from the submodule at
  its pinned SHA (`git -C upstream/Numerics show 'HEAD:<path>'` or direct reads -- the tree is
  clean).
- Every new R export goes in BOTH `corehydror/_pkgdown.yml` and `site/_quarto.yml`
  `quartodoc.sections`; new Python exports also join the module `__all__`.
- Commits are GPG-signed, identity `Cam Bracken <cameron.bracken@pm.me>`, no Co-Authored-By
  trailers. Push only when Cam asks (the ship task pushes; that is the standing exception).
- Branch base: `port-distribution-gaps` (PR #26 is green but unmerged; if it merges first, `main`
  is identical -- rebase then).
- Baseline numbers at branch creation (measured at `16989b9`, after the P1 final-review commit
  added a ctest suite): ctest 91/91; oracle gate 5568 reproduced / 0 failed / 11 skipped;
  testthat 6344/0; pytest 1520. Task 1 re-measures ctest after its first full run; if any number
  differs, the ledger's fresh measurement governs (the P1 process note: never reconcile against a
  stale census).
- Deliberate scope cut, documented not silent: `TabularFunction` depends on the unported Paired
  Data subsystem (UncertainOrderedPairedData / OrderedPairedData / Ordinate / UncertainOrdinate,
  scheduled P4) and is severed to P4 with an in-header note plus an `upstream/CLAUDE.md` entry.

---

### Task 1: Branch + root-finding ports (Bisection, NewtonRaphson, Secant)

**Files:**
- Create: `core/include/corehydro/numerics/math/rootfinding/bisection.hpp`
- Create: `core/include/corehydro/numerics/math/rootfinding/newton_raphson.hpp`
- Create: `core/include/corehydro/numerics/math/rootfinding/secant.hpp`
- Create: `core/tests/test_root_finding_extras.cpp`
- Modify: `core/CMakeLists.txt` (add `test_root_finding_extras` to the `BF_TESTS` list)

**Interfaces:**
- Produces, in namespace `corehydro::numerics::math::rootfinding` (mirroring `brent.hpp`'s
  free-function style -- open it first and copy its layout):
  - `bisection.hpp`: `inline double bisection_solve(const std::function<double(double)>& f,
    double first_guess, double lower_bound, double upper_bound, double tolerance = 1E-8,
    int max_iterations = 1000, bool report_failure = true)`.
  - `newton_raphson.hpp`: `inline double newton_raphson_solve(f, df, double first_guess,
    double tolerance = 1E-8, int max_iterations = 1000, bool report_failure = true)` (f and df
    both `const std::function<double(double)>&`); `inline double newton_raphson_robust_solve(f,
    df, double first_guess, double lower_bound, double upper_bound, double tolerance = 1E-8,
    int max_iterations = 1000, bool report_failure = true)`; `inline linalg::Vector
    newton_raphson_solve_system(const std::function<linalg::Vector(const linalg::Vector&)>& f,
    const std::function<linalg::Matrix(const linalg::Vector&)>& df, const linalg::Vector&
    first_guess, double tolerance = 1E-8, int max_iterations = 1000, bool report_failure =
    true)`.
  - `secant.hpp`: `inline double secant_solve(f, double lower_bound, double upper_bound,
    double tolerance = 1E-8, int max_iterations = 1000, bool report_failure = true)`.

- [ ] **Step 1: Create the branch**

```bash
cd /Users/cam/projects/usace/rmc/corehydro
git checkout port-distribution-gaps && git pull && git checkout -b port-math-extras
```

- [ ] **Step 2: Read the C# sources and tests**

```bash
for f in Bisection NewtonRaphson Secant; do
  git -C upstream/Numerics show "HEAD:Numerics/Mathematics/Root Finding/$f.cs" > /tmp/$f.cs
done
for f in Test_Bisection Test_NewtonRaphson Test_Secant TestFunctions; do
  git -C upstream/Numerics show "HEAD:Test_Numerics/Mathematics/Root Finding/$f.cs" > /tmp/$f.cs
done
```

- [ ] **Step 3: Write the failing ctest suite first**

Create `core/tests/test_root_finding_extras.cpp` transcribing all 25 C# test methods 1:1
(Test_Bisection 8, Test_NewtonRaphson 13 including the three multivariate `Multi_*` methods,
Test_Secant 4), with the shared `TestFunctions.cs` fixtures (`Quadratic`, `Cubic`,
`Trigonometric`, `Exponential`, `Power` and their `*_Deriv` companions) ported as file-local
static functions at the top. Copy the harness style from `core/tests/test_pivotal_bootstrap.cpp`
(`tests/check.hpp` macros, one function per C# test method, `main` calling each). Assertion
tolerances come from the C# asserts (typically `1E-5`). Register `test_root_finding_extras` in
the `BF_TESTS` list in `core/CMakeLists.txt`.

```bash
cmake -S core -B core/build && cmake --build core/build --target test_root_finding_extras -j
```
Expected: FAIL to compile -- the three headers do not exist.

- [ ] **Step 4: Port the three headers**

Transcribe each C# file member-for-member with the signatures from the Interfaces block.
Conventions from the C# source: `Bisection.Solve` takes a `firstGuess` AND a bracket (unusual --
keep it); NewtonRaphson uses `Tools.DoubleMachineEpsilon` -- write
`std::numeric_limits<double>::epsilon()` inline with a `// C#: Tools.DoubleMachineEpsilon`
comment; Secant uses `Tools.Swap` -- use `std::swap`. The multivariate solve uses
`linalg::Vector` / `linalg::Matrix` (already ported at `numerics/math/linalg/`) and solves the
Jacobian system the same way the C# does (read which decomposition C# calls and use the ported
one). Error messages transcribe verbatim (the ctest suite asserts throw behavior).

- [ ] **Step 5: Run the suite until green, then the full ctest**

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_root_finding_extras
ctest --test-dir core/build
```
Expected: new suite passes; full suite green at baseline + 1 (92/92 at the 91 baseline).

- [ ] **Step 6: Commit**

```bash
git add core/include/corehydro/numerics/math/rootfinding/{bisection,newton_raphson,secant}.hpp \
        core/tests/test_root_finding_extras.cpp core/CMakeLists.txt
git commit -m "feat: port Bisection, NewtonRaphson, and Secant root finders"
```

---

### Task 2: Root-finding callback surface

**Files:**
- Modify: `core/include/corehydro/numerics/support/callback/common.hpp` (new `CallbackSet`
  members)
- Modify: `core/include/corehydro/numerics/support/callback/math.hpp` (extend `root_find`, add
  `root_find_newton` and `root_find_system`)
- Modify: `core/tests/fixture_callback_catalog.cpp` + `.hpp` (new catalog entries)
- Modify: `corehydror/src/callback.cpp` (new `ch_callback_math2_` entry point + predicate update),
  `corehydror/R/callback.R` (extend `root_find`, add `root_find_system`)
- Modify: `corehydropy/src/bindings/callback.cpp` (mirror `callback_math2`),
  `corehydropy/src/corehydropy/callback.py` (extend `root_find`, add `root_find_system`,
  `__all__`)
- Modify: `fixtures/callback/math.json` (new cases), `fixtures/README.md` (method vocabulary)
- Modify: `tools/oracle_emitter/Program.cs` (new math arms + catalog functions)
- Test: existing R/Python callback test files (locate:
  `grep -rln 'root_find' corehydror/tests corehydropy/tests`)

**Interfaces:**
- Consumes: the three Task 1 headers.
- Produces, on the callback `math` group:
  - `root_find` gains option `method`: `"brent"` (default, existing behavior) | `"bisection"` |
    `"secant"`. Bisection additionally requires option `first_guess`. All share
    `lower`/`upper`/`tolerance`/`max_iterations`. Result unchanged: `values={root}`,
    `names={"root"}`.
  - `root_find_newton` (new): callbacks f (scalar) + df (second scalar, new `CallbackSet` member
    `scalar_deriv`); options `first_guess` (required), `lower`+`upper` (optional -- both present
    selects `newton_raphson_robust_solve`, else `newton_raphson_solve`), `tolerance`,
    `max_iterations`. Result `values={root}`, `names={"root"}`.
  - `root_find_system` (new): callbacks F (vector-to-vector) + J (vector-to-matrix, row-major)
    -- reuse the existing `CallbackSet` members of those shapes if present (read `common.hpp`;
    the mcmc/gmm groups already marshal both shapes), else add `vector_vector` /
    `vector_matrix`; options `first_guess` (vector), `tolerance`, `max_iterations`. Result
    `values=root vector`, `dims={n}`.
- Produces, user-facing (same names and defaults in R and Python):
  - `root_find(f, lower, upper, method = c("brent", "bisection", "secant", "newton"),
    df = NULL, first_guess = NULL, tolerance = NULL, max_iterations = NULL)` -- `method="newton"`
    requires `df` and `first_guess` and routes to `root_find_newton` (bracket optional, selecting
    the robust variant); `method="bisection"` requires `first_guess`.
  - `root_find_system(f, jacobian, first_guess, tolerance = NULL, max_iterations = NULL)` -> the
    root vector (numeric vector / np.ndarray).

- [ ] **Step 1: Read the current glue end-to-end**

Read `callback/common.hpp`, `callback/math.hpp` fully, `corehydror/src/callback.cpp`
`ch_callback_math_` (the method-name predicate near line 444), and the Python twin near line 512
of `corehydropy/src/bindings/callback.cpp`, so the new arms ride the existing `GuardedCall`
pattern (sentinel, `try/catch` with `rethrow_if_aborted`, second `rethrow_if_aborted` on
success -- the derivative arm is the template).

- [ ] **Step 2: Write failing R and Python tests**

In the existing callback test files, add: a bisection root of x^2-2 on [0,4] with first_guess 1
equal to `sqrt(2)` within 1e-5; a newton root of the same with analytic df; a
`root_find_system` case solving the linear system from Test_NewtonRaphson `Multi_LinearSystem`
(transcribe its F, J, and expected root). Run both suites; expected: FAIL (unknown
argument/method).

- [ ] **Step 3: Implement the C++ arms, then the two glues**

`math.hpp`: the `root_find` arm branches on `o.contains("method")` (absent = brent, preserving
every existing fixture); `root_find_newton` guards both `scalar` and `scalar_deriv` with separate
`GuardedCall`s sharing the run's abort state; `root_find_system` marshals the vector/matrix
callbacks. R: add

```cpp
[[cpp11::register]]
list ch_callback_math2_(std::string method, std::string options_json, function f, function g) {
    sup::CallbackSet cbs;
    if (method == "root_find_newton") {
        cbs.scalar = as_scalar_fn(f);
        cbs.scalar_deriv = as_scalar_fn(g);
    } else if (method == "root_find_system") {
        cbs.vector_vector = as_vector_vector_fn(f);
        cbs.vector_matrix = as_matrix_fn(g);
    } else {
        stop("unknown two-callback math method");
    }
    return pack(sup::run_callback("math", method, options_json, cbs));
}
```

(adjust member/converter names to what `common.hpp` and `callback.cpp` actually define). Python
mirrors it as `callback_math2`. Re-run `Rscript -e 'cpp11::cpp_register("corehydror")'`,
reinstall both packages, run the two new tests until green.

- [ ] **Step 4: Fixture + emitter parity**

Extend `fixtures/callback/math.json`: cases `root_find_bisection_quadratic`,
`root_find_secant_cubic`, `newton_quadratic`, `newton_robust_trig`, `system_linear` --
expected values from the C# test literals, catalog names `Root_Quadratic`/`Root_Cubic` (existing)
plus new `RootD_Quadratic`, `Root_Trigonometric`, `RootD_Trigonometric`, `Sys_Linear_F`,
`Sys_Linear_J` defined in all four catalogs (C++, R, Python, C# emitter -- transcribe the
TestFunctions.cs bodies). Emitter: add the `root_find` method-option branch and the
`root_find_newton`/`root_find_system` arms driving the real C# `Bisection.Solve` /
`Secant.Solve` / `NewtonRaphson.Solve` / `RobustSolve` / vector `Solve`, beside the existing
Brent arm at Program.cs line ~5705. Document the new methods in `fixtures/README.md`'s callback
section.

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: fixture runner green with new checks counted; gate `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: expose bisection, secant, Newton, and system root finding"
```

---

### Task 3: Deterministic integrator ports

**Files:**
- Create: `core/include/corehydro/numerics/math/integration/simpsons_rule.hpp`
- Create: `core/include/corehydro/numerics/math/integration/trapezoidal_rule.hpp`
- Create: `core/include/corehydro/numerics/math/integration/adaptive_simpsons_rule.hpp`
- Create: `core/include/corehydro/numerics/math/integration/adaptive_simpsons_rule_2d.hpp`
- Create: `core/include/corehydro/numerics/math/integration/adaptive_gauss_lobatto.hpp`
- Modify: `core/include/corehydro/numerics/math/integration/integration.hpp` (fill in the four
  deferred statics: `gauss_legendre` 10-point, `trapezoidal_rule`, `simpsons_rule`, `midpoint`;
  delete the deferral note)
- Create: `core/tests/test_integration_extras.cpp`
- Modify: `core/CMakeLists.txt` (`BF_TESTS` += `test_integration_extras`)

**Interfaces:**
- Consumes: `Integrator` base (`integration/support/integrator.hpp`), `StratificationBin`
  (`numerics/sampling/stratification_bin.hpp`).
- Produces, in namespace `corehydro::numerics::math::integration`, each `class X : public
  Integrator` mirroring `adaptive_gauss_kronrod.hpp`'s style:
  - `SimpsonsRule(std::function<double(double)> function, double min, double max)`;
    `TrapezoidalRule(...)` same shape; both `void integrate() override`.
  - `AdaptiveSimpsonsRule(function, min, max)` with `int min_depth = 0`, `int max_depth = 100`,
    `double standard_error()` and `void integrate(const std::vector<sampling::StratificationBin>&
    bins)` overload beside `integrate()`.
  - `AdaptiveSimpsonsRule2D(std::function<double(double, double)> function, double min_x,
    double max_x, double min_y, double max_y)` with `min_depth`/`max_depth`/`standard_error()`.
  - `AdaptiveGaussLobatto(function, min, max)` -- the C# `private static bool terminate` becomes
    an instance member (documented deviation: the static is not thread-safe; note it in the
    header).
  - `integration.hpp` statics: `gauss_legendre(f, a, b)`, `trapezoidal_rule(f, a, b, int steps =
    2)`, `simpsons_rule(f, a, b, int steps = 2)`, `midpoint(f, a, b, int steps = 2)` beside the
    existing `gauss_legendre20`.

- [ ] **Step 1: Read the C# sources and tests**

```bash
for f in SimpsonsRule TrapezoidalRule AdaptiveSimpsonsRule AdaptiveSimpsonsRule2D \
         AdaptiveGaussLobatto Integration; do
  git -C upstream/Numerics show "HEAD:Numerics/Mathematics/Integration/$f.cs" > /tmp/$f.cs
done
for f in Test_SimpsonsRule Test_TrapezoidalRule Test_AdaptiveSimpsonsRule \
         Test_AdaptiveSimpsonsRule2D Test_AdaptiveGaussLobatto Test_Integration Integrands; do
  git -C upstream/Numerics show "HEAD:Test_Numerics/Mathematics/Integration/$f.cs" > /tmp/$f.cs
done
```

- [ ] **Step 2: Write the failing ctest suite first**

Create `core/tests/test_integration_extras.cpp`: port `Integrands.cs` as file-local statics
(`FX3`, `Cosine`, `Sine`, `FXX`, `FXXX`, `PI2D`, plus the Gamma/CVaR integrands the per-class
tests define inline -- read each test file; the `SumOfNormals` family stays out until Task 5).
The four 1D suites (Simpsons/Trapezoidal/AdaptiveSimpsons/GaussLobatto) share the same 7 test
names -- write one templated helper run four times. Add the 4 `AdaptiveSimpsonsRule2D` methods
and the 5 `Test_Integration` static-rule methods. C# tolerances (`1E-3` abs, typically) govern.
Register in `BF_TESTS`.

```bash
cmake -S core -B core/build && cmake --build core/build --target test_integration_extras -j
```
Expected: FAIL to compile.

- [ ] **Step 3: Port the five classes and the four statics**

Transcribe in C# file order onto the ported `Integrator` base (fields
`min_iterations`/`max_iterations`/`min_function_evaluations`/`max_function_evaluations`/
`absolute_tolerance`/`relative_tolerance`/`report_failure`; use `clear_results()`, `validate()`,
`update_status()`, `evaluate_convergence()` exactly as `adaptive_gauss_kronrod.hpp` does).
Provenance headers at `@ 2a0357a`.

- [ ] **Step 4: Green, then full ctest, then commit**

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_integration_extras
ctest --test-dir core/build
git add core/include/corehydro/numerics/math/integration/*.hpp \
        core/tests/test_integration_extras.cpp core/CMakeLists.txt
git commit -m "feat: port the deterministic integrators and Integration statics"
```
Expected: full ctest green at baseline + 2 (93/93 at the 91 baseline).

---

### Task 4: Deterministic quadrature surface (method option + quadrature_2d)

**Files:**
- Modify: `core/include/corehydro/numerics/support/callback/common.hpp` (`CallbackSet` member
  `scalar_xy` = `std::function<double(double, double)>`)
- Modify: `core/include/corehydro/numerics/support/callback/math.hpp` (`quadrature` method
  option + new `quadrature_2d`)
- Modify: `core/tests/fixture_callback_catalog.cpp` + `.hpp`, `corehydror/src/callback.cpp` +
  `corehydror/R/callback.R`, `corehydropy/src/bindings/callback.cpp` +
  `corehydropy/src/corehydropy/callback.py`
- Modify: `fixtures/callback/math.json`, `fixtures/README.md`, `tools/oracle_emitter/Program.cs`

**Interfaces:**
- Consumes: Task 3 classes.
- Produces, on the callback `math` group:
  - `quadrature` gains option `method`: `"gauss_kronrod"` (default, existing) | `"simpsons"` |
    `"trapezoidal"` | `"adaptive_simpsons"` | `"gauss_lobatto"` | `"gauss_legendre"` |
    `"gauss_legendre20"` | `"simpsons_fixed"` | `"trapezoidal_fixed"` | `"midpoint"`. The
    `*_fixed`/`midpoint` statics take option `steps` (default 2); `adaptive_simpsons` takes
    `min_depth`/`max_depth`. Integrator-class methods keep the existing result triple
    `{integral, function_evaluations, standard_error}` + status (standard_error 0.0 where the
    class has none); statics return `values={integral}`, `names={"integral"}`, status
    `"Success"`.
  - `quadrature_2d` (new): callback `scalar_xy`; options `min_x`/`max_x`/`min_y`/`max_y`
    (required), `absolute_tolerance`/`relative_tolerance`/`min_depth`/`max_depth` (optional) ->
    `AdaptiveSimpsonsRule2D`. Result triple + status as `quadrature`.
- Produces, user-facing:
  - `quadrature()` gains `method`, `steps`, `min_depth`, `max_depth` arguments (R `NULL` /
    Python `None` defaults; option written only when supplied, per the file's stated
    convention).
  - `quadrature_2d(f, min_x, max_x, min_y, max_y, absolute_tolerance = NULL, relative_tolerance
    = NULL, min_depth = NULL, max_depth = NULL)` -- R returns the value with the existing
    `status`/`standard_error` attributes; Python returns `QuadratureResult`.

- [ ] **Step 1: Write failing R and Python tests**

Add: `quadrature(f, 0, 1, method = "gauss_lobatto")` of x^3 equal to 0.25 within 1e-3;
`quadrature(..., method = "midpoint", steps = 1000)`; a `quadrature_2d` of x+y on the unit
square equal to 1.0 within 1e-3 (Test_AdaptiveSimpsonsRule2D `XPlusY`). Expected: FAIL.

- [ ] **Step 2: Implement the C++ arm extension and `quadrature_2d`, then the glues**

The `quadrature` arm reads `method` (absent = `"gauss_kronrod"`, preserving every existing
fixture) and switches over the Task 3 classes/statics; each class instantiation copies the
optional-tolerance plumbing the existing AGK block uses. `quadrature_2d` guards `scalar_xy` with
`GuardedCall<double, double, double>`. Glue: `ch_callback_math_`'s predicate keeps
`quadrature` under `scalar`; `quadrature_2d` marshals the two-argument R/Python function via a
new `as_scalar_xy_fn` (R: call `f(x, y)`; Python: `f(x, y)`). Register, reinstall, tests green.

- [ ] **Step 3: Fixture + emitter parity**

New `math.json` cases pinning each method name once from the C# test literals (e.g.
`quadrature_lobatto_fxx` from Test_AdaptiveGaussLobatto, `quadrature_simpsons_cosine`,
`quadrature_gauss_legendre_fx3`, `quadrature_midpoint_fx3` from Test_Integration,
`quadrature_2d_x_plus_y` with new catalog names `Quad2D_XPlusY`, `Quad2D_PI2D` in all four
catalogs). Emitter: extend the quadrature arm's method switch to construct the real C# classes
(`new SimpsonsRule(f, a, b)` etc.) and the statics (`Integration.Midpoint(f, a, b, steps)`),
plus a `quadrature_2d` arm and `CallbackScalarXyFunction` catalog. Update `fixtures/README.md`.

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: green; `0 failed`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: expose the deterministic quadrature methods and 2D integration"
```

---

### Task 5: Stochastic integrator ports (MonteCarlo, Miser, Vegas)

**Files:**
- Modify: `core/include/corehydro/numerics/sampling/mersenne_twister.hpp` (add
  `std::vector<double> next_doubles(int n)` -- a loop over `next_double()`, matching C#
  `NextDoubles` stream order exactly; verify against the C# `MersenneTwister.NextDoubles`
  source first)
- Create: `core/include/corehydro/numerics/math/integration/monte_carlo_integration.hpp`
- Create: `core/include/corehydro/numerics/math/integration/miser.hpp`
- Create: `core/include/corehydro/numerics/math/integration/vegas.hpp`
- Create: `core/tests/test_integration_random.cpp`
- Modify: `core/CMakeLists.txt` (`BF_TESTS` += `test_integration_random`)

**Interfaces:**
- Consumes: `Integrator`, `MersenneTwister`, `SobolSequence` (`numerics/sampling/sobol.hpp`).
- Produces, in `corehydro::numerics::math::integration`:
  - `MonteCarloIntegration(std::function<double(const std::vector<double>&)> function,
    int dimensions, std::vector<double> min, std::vector<double> max)` with a settable RNG
    (mirror how ported classes take a seed/engine -- read how `bootstrap.hpp` or the samplers
    hold their `MersenneTwister` and copy that idiom), `bool use_sobol_sequence = true`,
    `standard_error()`.
  - `Miser(...)` same ctor shape plus `double fraction = 0.1`,
    `int minimum_number_of_subregion_points = 15`, `int minimum_number_of_bisections = 60`,
    `double dither = 0.0`.
  - `Vegas(std::function<double(const std::vector<double>&, double)> function, int dimensions,
    min, max)` -- the integrand takes the weight as its second argument -- plus
    `check_convergence = true`, `initialize = 0`, `independent_evaluations = 1000`,
    `function_calls = 10000`, `alpha = 1.5`, `number_of_bins` (with the C# validation body),
    `tail_focus_parameter = 1.0`, `standard_error()`, `chi_squared()`, and
    `void configure_for_rare_events(double target_probability)`.

- [ ] **Step 1: Read the C# sources and tests**

```bash
for f in MonteCarloIntegration Miser Vegas; do
  git -C upstream/Numerics show "HEAD:Numerics/Mathematics/Integration/$f.cs" > /tmp/$f.cs
done
for f in Test_MonteCarlo Test_Miser Test_Vegas; do
  git -C upstream/Numerics show "HEAD:Test_Numerics/Mathematics/Integration/$f.cs" > /tmp/$f.cs
done
git -C upstream/Numerics grep -n "NextDoubles" -- Numerics/Sampling
```

- [ ] **Step 2: Write the failing ctest suite first**

`test_integration_random.cpp`: the `SumOfNormals` integrand family from `Integrands.cs` (the
`mu20`/`sigma20` literal arrays plus the Normal-pdf sum written as explicit arithmetic -- the
ported `distributions::Normal` may be used here since this is core-only code), `PI`, `GSL`; the
5 Test_MonteCarlo + 5 Test_Miser + 9 Test_Vegas methods 1:1, including the three
`PowerTransform_*` rare-event tests and the range-check assertions. Reproduce each test's RNG
setup exactly as C# does it (default-constructed `MersenneTwister` where the test does, the
test's seed where it seeds); C# relative tolerances (`1E-2`/`1E-3 * trueResult`) govern.
Register in `BF_TESTS`. Expected: compile FAIL.

- [ ] **Step 3: Port `next_doubles`, then the three classes**

`next_doubles(n)` must consume the stream in the same order as C# `NextDoubles` (read the C#
body -- if it is a simple loop over `NextDouble()`, mirror that). Then transcribe the three
integrators; the Sobol path uses the ported `SobolSequence::next_double()` (returns the whole
point). Note in the Vegas header that `Grid` is exposed as a row-major accessor.

- [ ] **Step 4: Green, full ctest, commit**

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_integration_random
ctest --test-dir core/build
git add -A && git commit -m "feat: port MonteCarloIntegration, Miser, and Vegas"
```
Expected: full ctest green at baseline + 3 (94/94 at the 91 baseline). If a stochastic assertion is flaky where C# is not, the streams differ -- fix
the port, do not widen tolerance.

---

### Task 6: Stochastic quadrature surface (quadrature_nd + quadrature_vegas)

**Files:**
- Modify: `core/include/corehydro/numerics/support/callback/common.hpp` (`CallbackSet` member
  `vector_weight` = `std::function<double(const std::vector<double>&, double)>`)
- Modify: `core/include/corehydro/numerics/support/callback/math.hpp` (methods `quadrature_nd`,
  `quadrature_vegas`)
- Modify: the four catalogs, both glues (`callback.cpp`/`callback.R`,
  `bindings/callback.cpp`/`callback.py`), `fixtures/callback/math.json`,
  `fixtures/callback/callback_cross_language.json`, `fixtures/README.md`,
  `tools/oracle_emitter/Program.cs`

**Interfaces:**
- Consumes: Task 5 classes.
- Produces, on the callback `math` group:
  - `quadrature_nd`: callback `vector_scalar`; options `method`: `"monte_carlo"` (default) |
    `"miser"`, `min` (vector), `max` (vector), `seed` (int; seeds the class RNG -- absent means
    the C# default construction), `use_sobol` (bool, default true),
    `max_function_evaluations`, plus Miser's `fraction` / `min_subregion_points` /
    `min_bisections` / `dither`. Result `values={integral, function_evaluations,
    standard_error}` + status.
  - `quadrature_vegas`: callback `vector_weight`; options `min`, `max`, `seed`, `use_sobol`,
    `independent_evaluations`, `function_calls`, `alpha`, `number_of_bins`,
    `tail_focus_parameter`, `initialize`, `check_convergence`, `target_probability` (present ->
    `configure_for_rare_events`). Result `values={integral, function_evaluations,
    standard_error, chi_squared}` + status.
- Produces, user-facing: `quadrature_nd(f, min, max, method = c("monte_carlo", "miser",
  "vegas"), seed = NULL, use_sobol = TRUE, ...)` in both languages, routing `method="vegas"` to
  the `quadrature_vegas` runner method -- for vegas the user's `f` takes `(x, weight)`; R
  returns value with attributes (`standard_error`, `status`, and `chi_squared` for vegas),
  Python returns `QuadratureResult` (chi_squared added as an optional field).

- [ ] **Step 1: Write failing R and Python tests**

A seeded `quadrature_nd` of the 3D `SumOfNormals` catalog integrand asserting R and Python
return IDENTICAL values (the callback-layer determinism contract), and a seeded
`quadrature_nd(..., method = "vegas")` where `f <- function(x, w) ...` runs without error and
returns a finite value with a `chi_squared` attribute. Expected: FAIL.

- [ ] **Step 2: Implement the two arms and the glues**

Both arms follow the derivative-arm `GuardedCall` template; `quadrature_vegas` guards
`vector_weight` with `GuardedCall<double, const std::vector<double>&, double>`. Seed handling:
`seed` present -> construct/seed the class RNG the same way the emitter will (pick the idiom the
ported samplers use so C# `new MersenneTwister(seed)` matches). Glues: `quadrature_nd` joins the
`vector_scalar` predicate branch; `quadrature_vegas` gets an `as_vector_weight_fn` converter.
Register, reinstall, tests green.

- [ ] **Step 3: Fixtures -- pinned cases plus zero-tolerance digests**

`math.json`: seeded `quadrature_nd_pi_3d` (catalog `Nd_PI`), `quadrature_miser_gsl`
(`Nd_GSL`), `quadrature_vegas_sum_normals` (`NdW_SumOfNormals3` -- the weight-ignoring wrapper,
exactly as the C# tests wrap it), values pinned from the driven C# classes via
`verify_oracles.py --dump` (the C# test literals are true-value checks at loose tolerance; the
pinned oracle is the seeded C# result). Catalog bodies: explicit loops, Normal pdf written out
as `exp(-0.5*z*z) / (sigma * sqrt(2*pi))` arithmetic in all four languages, `mu20`/`sigma20`
inlined as literals. `callback_cross_language.json`: one seeded case per class (monte_carlo,
miser, vegas) at ZERO tolerance -- these prove the R/Python bit-identity invariant; if C#-vs-C++
drifts on the pinned cases, check FMA contraction first (the `-ffp-contract=off` precedent)
before touching anything.

- [ ] **Step 4: Emitter + gate**

`Program.cs`: `quadrature_nd`/`quadrature_vegas` arms constructing the real C#
`MonteCarloIntegration`/`Miser`/`Vegas` with the same option plumbing and seed, plus
`CallbackVectorWeightFunction` and the `Nd_*`/`NdW_*` catalog entries.

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: green; `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: expose seeded Monte Carlo, Miser, and Vegas integration"
```

---

### Task 7: RungeKutta port + ode_solve surface

**Files:**
- Create: `core/include/corehydro/numerics/math/ode/runge_kutta.hpp` (new directory)
- Create: `core/tests/test_runge_kutta.cpp`; Modify: `core/CMakeLists.txt`
- Modify: `callback/math.hpp` (method `ode_solve`), the four catalogs, both glues, both public
  APIs, `fixtures/callback/ode.json` (new file), `fixtures/README.md`,
  `tools/oracle_emitter/Program.cs`

**Interfaces:**
- Produces, in `corehydro::numerics::math::ode` (free functions, `brent.hpp` style):
  `second_order(const std::function<double(double, double)>& f, double initial_value,
  double start_time, double end_time, int time_steps) -> std::vector<double>`; `fourth_order`
  same shape; `fourth_order_step(f, initial_value, start_time, double dt) -> double`;
  `fehlberg(f, initial_value, start_time, double dt, double dt_min, double tolerance = 1E-3)
  -> double`; `cash_karp(...)` same shape as fehlberg.
- Produces, on the callback `math` group: `ode_solve` -- callback `scalar_xy` (f(t, y));
  options `method`: `"rk4"` (default) | `"rk2"` | `"rkf"` | `"cash_karp"`, `initial_value`,
  `start_time`, and per method: rk2/rk4 `end_time` + `time_steps` (result = the solution
  vector, `dims={len}`); rk4 with `dt` and no `end_time` = the single-step overload; rkf /
  cash_karp `dt` + `dt_min` + `tolerance` (result `values={value}`, `names={"value"}`).
- Produces, user-facing: `ode_solve(f, initial_value, start_time, end_time = NULL, time_steps =
  NULL, dt = NULL, dt_min = NULL, method = c("rk4", "rk2", "rkf", "cash_karp"), tolerance =
  NULL)` in both languages; vector return for the array methods, scalar for the adaptive ones.

- [ ] **Step 1: Read, then failing ctest**

```bash
git -C upstream/Numerics show "HEAD:Numerics/Mathematics/ODE Solvers/RungeKutta.cs" > /tmp/RungeKutta.cs
git -C upstream/Numerics show "HEAD:Test_Numerics/Mathematics/ODE Solvers/Test_RungeKutta.cs" > /tmp/Test_RungeKutta.cs
```

`test_runge_kutta.cpp` transcribes the 5 test methods 1:1 with their literals; register in
`BF_TESTS`; expect compile FAIL; port the header; suite green; full ctest at baseline + 4 (95/95 at the 91 baseline).

- [ ] **Step 2: Failing R/Python test, then the arm and glues**

Test: `ode_solve` of dy/dt = y from y(0)=1 to t=1 with 100 rk4 steps approximates e within
1e-5 in both languages. The `ode_solve` arm reuses `scalar_xy` (added Task 4); glue predicate
routes `ode_solve` through `as_scalar_xy_fn`. Register, reinstall, green.

- [ ] **Step 3: Fixture + emitter, commit**

New `fixtures/callback/ode.json` (kind `callback`, group `math`): one case per method with the
Test_RungeKutta literals, catalog names `Ode_*` (the test's RHS functions) in all four
catalogs; emitter arm drives the real C# `RungeKutta.*`. Gate `0 failed`; then:

```bash
git add -A && git commit -m "feat: port RungeKutta and expose ode_solve"
```

---

### Task 8: CubicSpline + Polynomial interpolation

**Files:**
- Create: `core/include/corehydro/numerics/data/interpolation/cubic_spline.hpp`
- Create: `core/include/corehydro/numerics/data/interpolation/polynomial.hpp`
- Create: `core/tests/test_interpolation_extras.cpp`; Modify: `core/CMakeLists.txt`
- Modify: `core/include/corehydro/numerics/support/toolbox/interpolation.hpp` (methods
  `cubic_spline`, `polynomial`)
- Modify: `fixtures/toolbox/interpolation.json`, `fixtures/README.md`,
  `tools/oracle_emitter/Program.cs` (`InterpolationDispatch` arms)
- Modify: `corehydror/R/toolbox.R` (`interpolate()` gains `method` + `order`),
  `corehydropy/src/corehydropy/toolbox.py` (same)

**Interfaces:**
- Consumes: `Interpolater` base (`interpolater.hpp`: ctor `(x, y, SortOrder)`, pure virtual
  `double base_interpolate(double x, int index) const`).
- Produces, in `corehydro::numerics::data`:
  - `CubicSpline(std::vector<double> x, std::vector<double> y, SortOrder sort_order =
    SortOrder::Ascending)` calling `set_second_derivatives()` in the ctor (public, re-callable);
    `double base_interpolate(double x, int index) const override`.
  - `Polynomial(int order, x, y, SortOrder)` (throws when `order >= count`); `double error()
    const`; `base_interpolate` override -- the C# writes `Error` inside `BaseInterpolate`, so
    `error_` is `mutable` (documented in-header deviation rationale: the ported base signature
    is const). The C# `Subset` extension becomes an index-offset loop.
- Produces, toolbox `interpolation` methods (data `[x, y, xout]`): `cubic_spline` (option
  `sort_order`), `polynomial` (options `order` required, `sort_order`) -> `values` per xout
  point.
- Produces, user-facing: `interpolate(x, y, xout, method = c("linear", "cubic_spline",
  "polynomial"), order = NULL, x_transform = ..., y_transform = ..., sort_order = ...,
  extrapolate = FALSE)` -- transforms/extrapolate stay linear-only (validate and error
  otherwise, matching C# which has neither on these classes).

- [ ] **Step 1: Read, then failing ctest**

```bash
for f in CubicSpline Polynomial; do
  git -C upstream/Numerics show "HEAD:Numerics/Data/Interpolation/$f.cs" > /tmp/$f.cs
done
for f in Test_CubicSpline Test_Polynomial; do
  git -C upstream/Numerics show "HEAD:Test_Numerics/Data/Interpolation/$f.cs" > /tmp/$f.cs
done
```

`test_interpolation_extras.cpp` transcribes all 12 test methods (including the 34-element
R-validated `true_Y` array); register; compile FAIL; port the two headers; green; full ctest
full ctest at baseline + 5 (96/96 at the 91 baseline); commit `feat: port CubicSpline and Polynomial interpolation`.

- [ ] **Step 2: Toolbox methods + failing R/Python tests + fixture + emitter**

Add the two arms to `run_interpolation` beside `linear` (copy its shape; per-point
`interpolate(v)` loop). R/Python tests: `interpolate(x, y, xout, method = "cubic_spline")`
reproduces a Test_CubicSpline literal; `method = "polynomial", order = 3` reproduces a
Test_Polynomial literal. Fixture: new cases in `interpolation.json` with datasets from the two
C# test files; emitter `InterpolationDispatch` gains `case "cubic_spline"` / `case
"polynomial"` driving the real C# classes. Verify all four runners + gate; commit
`feat: expose cubic spline and polynomial interpolation`.

---

### Task 9: Linear algebra (QRDecomposition + GaussJordanElimination) + toolbox linalg group

**Files:**
- Create: `core/include/corehydro/numerics/math/linalg/qr_decomposition.hpp`
- Create: `core/include/corehydro/numerics/math/linalg/gauss_jordan_elimination.hpp`
- Create: `core/tests/test_linalg_extras.cpp`; Modify: `core/CMakeLists.txt`
- Create: `core/include/corehydro/numerics/support/toolbox/linalg.hpp`
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (include + dispatch line
  + the group-count header comment)
- Create: `fixtures/toolbox/linalg.json`; Modify: `fixtures/README.md`
- Modify: `tools/oracle_emitter/Program.cs` (`ToolboxDispatch` `case "linalg"` +
  `LinalgDispatch`)
- Create/Modify: R `qr_decomposition()` + `qr_solve()` + `gauss_jordan()` in
  `corehydror/R/toolbox.R`, Python twins in `corehydropy/src/corehydropy/toolbox.py`
- Modify: `corehydror/_pkgdown.yml` + `site/_quarto.yml` (new section, Task 12 finalizes)

**Interfaces:**
- Produces, in `corehydro::numerics::math::linalg` (mirroring `lu_decomposition.hpp`'s style):
  `QRDecomposition(const Matrix& a)` with `Matrix q() const`, `Matrix r() const`, `Vector
  solve(const Vector& b) const`, `Matrix solve(const Matrix& b) const`; and
  `inline void gauss_jordan_solve(Matrix& a, Matrix& b)` (in-place: on return `a` is the
  inverse, `b` the solution set -- the C# `Solve(ref, ref)`).
- Produces, toolbox group `linalg` -- matrices travel as ONE flattened row-major data vector
  with `rows`/`cols` in options (the bilinear-y_flat precedent): `qr_q` (data `[a_flat]`) ->
  Q with dims; `qr_r` -> R with dims; `qr_solve` (data `[a_flat, b]`) -> vector; `qr_solve_matrix`
  (data `[a_flat, b_flat]`, option `b_cols`) -> matrix; `gauss_jordan_inverse` (data
  `[a_flat, b_flat]`, option `b_cols`) -> the inverse; `gauss_jordan_solution` (same) -> the
  solution set.
- Produces, user-facing: `qr_decomposition(a)` -> list/dict with `q` and `r` matrices;
  `qr_solve(a, b)` -> vector or matrix by the shape of `b`; `gauss_jordan(a, b)` -> list/dict
  with `inverse` and `solution`. (R: matrices in, `as.double` row-major flattening inside the
  wrapper; Python: numpy in/out.)

- [ ] **Step 1: Read, then failing ctest**

```bash
for f in QRDecomposition GaussJordanElimination; do
  git -C upstream/Numerics show "HEAD:Numerics/Mathematics/Linear Algebra/$f.cs" > /tmp/$f.cs
done
for f in Test_QRDecomposition Test_GaussJordanElimination; do
  git -C upstream/Numerics show "HEAD:Test_Numerics/Mathematics/Linear Algebra/$f.cs" > /tmp/$f.cs
done
```

`test_linalg_extras.cpp` transcribes the 9 QR methods (square, overdetermined,
underdetermined, vector and matrix solves) and the 1 Gauss-Jordan method (exact `AreEqual`,
tol 0 -- keep it exact); register; compile FAIL; port; green; full ctest at baseline + 6 (97/97 at the 91 baseline); commit
`feat: port QRDecomposition and GaussJordanElimination`.

- [ ] **Step 2: The linalg toolbox group + verbs + fixture + emitter**

`toolbox/linalg.hpp`: `inline ToolboxResult run_linalg(const std::string& method, const
std::vector<std::vector<double>>& data, const JsonValue& options)` in
`corehydro::numerics::support::detail`, with a local `to_matrix(data_at(...), rows, cols)`
helper; wire the include + `if (group == "linalg") return detail::run_linalg(...)` into
`toolbox_runner.hpp`. Failing R/Python tests first: `qr_solve` on the Test_QRDecomposition
square system reproduces its expected vector; `gauss_jordan` reproduces `true_IA`. Then the
verbs (roxygen + docstrings), `fixtures/toolbox/linalg.json` (kind `toolbox`, group `linalg`,
datasets = the C# test matrices flattened), the emitter `LinalgDispatch` driving the real C#
classes, `fixtures/README.md`. All four runners + gate green; commit
`feat: expose QR and Gauss-Jordan through a linalg toolbox group`.

---

### Task 10: Special functions (Debye + Evaluate) + toolbox special group

**Files:**
- Create: `core/include/corehydro/numerics/math/special/debye.hpp`
- Create: `core/include/corehydro/numerics/math/special/evaluate.hpp`
- Create: `core/include/corehydro/numerics/support/toolbox/special.hpp`
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp`
- Create: `fixtures/toolbox/special_functions.json`; Modify: `fixtures/README.md`
- Modify: `tools/oracle_emitter/Program.cs` (`case "special"` + `SpecialDispatch`)
- Create: R `debye()` + `polynomial_eval()` (in `corehydror/R/toolbox.R`), Python twins +
  `__all__`

**Interfaces:**
- Produces, in the namespace the existing `special/` headers use (open `gamma.hpp` and mirror
  it): `debye.hpp` -> `inline double debye_function(double x)` (C# `Debye.Function`);
  `evaluate.hpp` -> `inline double evaluate_polynomial(const std::vector<double>& coefficients,
  double x)`, `evaluate_polynomial_rev(coefficients, x, int n = -1)`,
  `evaluate_polynomial_rev_1(coefficients, x)`. `gamma.hpp`'s private inlined
  `polynomial_rev` copy stays as is; add a cross-reference comment in `evaluate.hpp` noting the
  deliberate duplication (changing gamma.hpp risks oracle-visible churn for zero gain).
- Produces, toolbox group `special`: `debye` (data `[x]`) -> values per x; `polynomial` (data
  `[coefficients, x]`) -> values per x; `polynomial_rev` (same, option `n` optional);
  `polynomial_rev_1` (same).
- Produces, user-facing: `debye(x)` (vectorized); `polynomial_eval(coefficients, x, variant =
  c("standard", "reverse", "reverse_unit"), n = NULL)`.

- [ ] **Step 1: Read, port, pin**

```bash
for f in Debye Evaluate; do
  git -C upstream/Numerics show "HEAD:Numerics/Mathematics/Special Functions/$f.cs" > /tmp/$f.cs
done
git -C upstream/Numerics show "HEAD:Test_Numerics/Mathematics/Special Functions/Test_SpecialFunctions.cs" > /tmp/Test_SpecialFunctions.cs
```

Port the two headers. No dedicated ctest suite: the oracle path here is the fixture (next
step), whose cases carry the `Test_Debye` / `Test_Polynomial` / `Test_PolynomialRev` /
`Test_PolynomialRev_1` array literals (input array + expected array, tol `1E-4` as in C#).

- [ ] **Step 2: Group, fixture, emitter, verbs**

Failing R/Python tests first (`debye(0.5)` against a Test_Debye literal; `polynomial_eval`
against a Test_Polynomial literal), then `toolbox/special.hpp` + the dispatch line + the two
verbs, then `fixtures/toolbox/special_functions.json` and the emitter `SpecialDispatch`
driving `Numerics.Mathematics.SpecialFunctions.Debye.Function` / `Evaluate.*`. All runners +
gate green.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: port Debye and Evaluate with a special toolbox group"
```

---

### Task 11: Functions layer (IUnivariateFunction, LinearFunction, PowerFunction)

**Files:**
- Create: `core/include/corehydro/numerics/functions/i_univariate_function.hpp`
- Create: `core/include/corehydro/numerics/functions/linear_function.hpp`
- Create: `core/include/corehydro/numerics/functions/power_function.hpp`
- Create: `core/tests/test_univariate_functions.cpp`; Modify: `core/CMakeLists.txt`
- Create: `core/include/corehydro/numerics/support/toolbox/functions.hpp`
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp`
- Create: `fixtures/toolbox/univariate_functions.json`; Modify: `fixtures/README.md`
- Modify: `tools/oracle_emitter/Program.cs` (`case "functions"` + `FunctionsDispatch`)
- Modify: `upstream/CLAUDE.md` (TabularFunction severance entry)
- Create: R `univariate_function()` in `corehydror/R/toolbox.R`, Python twin + `__all__`

**Interfaces:**
- Produces, in `corehydro::numerics::functions` (style-model: `i_link_function.hpp`):
  `IUnivariateFunction` with the 12 C# members (`number_of_parameters()`, `parameters_valid()`,
  `minimum()`/`set_minimum`, `maximum()`/`set_maximum`, `minimum_of_parameters()`,
  `maximum_of_parameters()`, `is_deterministic()`/setter, `confidence_level()`/setter,
  `set_parameters(const std::vector<double>&)`, `validate_parameters(params, bool throw_on_error)`,
  `function(double x)`, `inverse_function(double y)`); `LinearFunction` (alpha, beta, sigma;
  three ctors) and `PowerFunction` (alpha, beta, xi, sigma; `is_inverse` extra; the `Minimum`
  setter throws as in C#), both holding a ported `distributions::Normal` for the
  confidence-level path. The severed `TabularFunction` gets a note in
  `i_univariate_function.hpp`'s header (depends on Paired Data, P4) and an `upstream/CLAUDE.md`
  severance line.
- Produces, toolbox group `functions`: methods `evaluate` and `inverse` -- data `[x]`; options
  `function`: `"linear"` | `"power"`, `parameters` (array: linear `[alpha, beta, sigma]`,
  power `[alpha, beta, xi, sigma]`), `is_inverse` (power only), `confidence_level` (present ->
  `is_deterministic = false` + the level) -> values per x.
- Produces, user-facing: `univariate_function(type = c("linear", "power"), parameters, x,
  inverse = FALSE, is_inverse = FALSE, confidence_level = NULL)` in both languages.

- [ ] **Step 1: Read, then failing ctest**

```bash
for f in IUnivariateFunction LinearFunction PowerFunction; do
  git -C upstream/Numerics show "HEAD:Numerics/Functions/$f.cs" > /tmp/$f.cs
done
git -C upstream/Numerics show "HEAD:Test_Numerics/Functions/Test_Functions.cs" > /tmp/Test_Functions.cs
```

`test_univariate_functions.cpp` transcribes the 6 non-tabular Test_Functions methods
(`Test_Linear_Function`, `Test_Linear_Function_Inverse`, `Test_Power_Function`,
`Test_Power_Function_Inverse`, `Test_InversePower_Function`,
`Test_InversePower_Function_Inverse`); `Test_Tabular_Function` is NOT transcribed (severed, with
a comment saying why). Register; compile FAIL; port the three headers; green; full ctest at baseline + 7 (98/98 at the 91 baseline); commit
`feat: port the univariate Functions layer (Linear, Power)`.

- [ ] **Step 2: Group, fixture, emitter, verb, severance note**

Failing R/Python tests (linear function evaluate/inverse against Test_Functions literals),
then `toolbox/functions.hpp` + dispatch + verbs, `fixtures/toolbox/univariate_functions.json`
(cases incl. a `confidence_level` case pinned from the driven C#), emitter `FunctionsDispatch`
driving the real C# `LinearFunction`/`PowerFunction`, the `upstream/CLAUDE.md` severance line,
`fixtures/README.md`. All runners + gate green; commit
`feat: expose linear and power univariate functions`.

---

### Task 12: Documentation and worked example 18

**Files:**
- Modify: R roxygen for every new/extended verb; regenerate via
  `Rscript -e 'roxygen2::roxygenise("corehydror")'`; Python docstrings
- Modify: `corehydror/_pkgdown.yml` + `site/_quarto.yml` -- new exports land as: `root_find`
  (updated), `root_find_system`, `quadrature` (updated), `quadrature_2d`, `quadrature_nd`,
  `ode_solve` in "Root finding, integration and differentiation"; `interpolate` (updated) in
  "Interpolation and regression"; NEW parallel section "Linear algebra and special functions"
  holding `qr_decomposition`, `qr_solve`, `gauss_jordan`, `debye`, `polynomial_eval`,
  `univariate_function` (same title, same order, in both files)
- Create: `site/examples/18-numerical-methods/{python.ipynb, r.qmd}`
- Modify: the site examples index listing (find it:
  `grep -rn '17-' site/_quarto.yml site/examples/index.qmd`)

**Interfaces:**
- Consumes: everything shipped in Tasks 1-11.

- [ ] **Step 1: Reference docs + index contract**

Every new export gets title, description, every parameter, return value, one runnable example
(CRAN time limits); verify `Rscript -e 'pkgdown::check_pkgdown("corehydror")'` passes and the
quartodoc build lists every Python export. Elements-of-style conventions throughout.

- [ ] **Step 2: Example 18 -- numerical methods**

Copy the structure of the `site/examples/17-*` pair. Content: root finding (brent vs newton on
the same function), `ode_solve` of a decay equation against the analytic answer, quadrature
methods compared on one integrand plus a seeded rare-event Vegas run, `interpolate` with
cubic spline vs linear on a stage-discharge-like curve, a `qr_solve` linear system, and a
power function evaluated and inverted. Each page ends with the executable reproduction check
(pinned literal at 1e-15 relative tolerance for deterministic values; seeded values pinned
exactly). Python notebook committed WITH outputs (`jupyter nbconvert --to notebook --execute
--inplace ...`); R rendered so `site/_freeze/` updates.

- [ ] **Step 3: Build and verify the site, commit**

```bash
pixi run docs
git add -A && git commit -m "docs: reference updates and worked example 18"
```
Expected: clean build; the new page renders in both languages.

---

### Task 13: Version 0.9.0, full verification, ship

**Files:**
- Modify: `corehydror/DESCRIPTION` (0.9.0), `corehydropy/pyproject.toml` (0.9.0),
  `corehydror/NEWS.md`, `CHANGELOG.md`
- Modify: `.claude/CLAUDE.md` (Status: append the P2 paragraph with final numbers)

- [ ] **Step 1: Bump versions and write release notes**

NEWS/CHANGELOG: root finding (bisection/secant/Newton scalar + robust + system), the seven
integration classes plus the Integration statics (deterministic methods, 2D, seeded
MC/Miser/Vegas), RungeKutta `ode_solve`, cubic spline and polynomial interpolation, the
`linalg`/`special`/`functions` toolbox groups, and the TabularFunction severance to P4.

- [ ] **Step 2: Full verification (evidence before assertions)**

```bash
cmake --build core/build -j && ctest --test-dir core/build
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py
```
Expected: ctest green at baseline + 7 (98/98 at the 91 baseline); testthat 0 failures above 6344; pytest 0 failures above 1520; gate `0
failed` with reproduced count > 5568 and skips still 11. Record the final numbers in
NEWS/CHANGELOG and `.claude/CLAUDE.md`.

- [ ] **Step 3: R CMD check regression**

```bash
R CMD build corehydror && R CMD check --as-cran corehydror_0.9.0.tar.gz
```
Expected: the same three known NOTEs, no new WARNING or NOTE.

- [ ] **Step 4: Commit, push, PR, CI**

```bash
git add -A && git commit -m "chore: release v0.9.0 (math extras)"
git push -u origin port-math-extras
gh pr create --title "P2: math extras -- root finding, integration, ODE, interpolation, linalg, functions (v0.9.0)" \
  --body "$(sed -n '/^## v0.9.0/,/^## /p' CHANGELOG.md)"
gh run watch $(gh run list --branch port-math-extras -L1 --json databaseId -q '.[0].databaseId') --exit-status
```
Expected: CI green on the full matrix. If PR #26 is still open, base this PR on
`port-distribution-gaps` (`gh pr create --base port-distribution-gaps`); retarget to `main`
after #26 merges. Do not merge without Cam's go-ahead.

---

## Self-review notes

- Spec coverage: root finding -> Tasks 1-2; the seven integration classes + the deferred
  Integration statics -> Tasks 3-6; RungeKutta -> Task 7; CubicSpline/Polynomial -> Task 8;
  QR/Gauss-Jordan + the spec's named `linalg` group -> Task 9; Debye/Evaluate -> Task 10; the
  Functions layer -> Task 11 (TabularFunction severed to P4 with the Paired Data subsystem it
  depends on -- a documented dependency-ordering cut, not a silent skip); example 18 + docs ->
  Task 12; version/ship -> Task 13.
- The callback-vs-toolbox split follows the spec sentence exactly: callable-driven ->
  callback `math` methods; data-driven -> toolbox (`interpolation` additions + new `linalg`
  group, plus `special`/`functions` on the same cheap one-header pattern).
- Type consistency: `scalar_xy` introduced in Task 4 is consumed by Task 7 (`ode_solve`);
  `scalar_deriv`/`vector_vector`/`vector_matrix` from Task 2 and `vector_weight` from Task 6
  live in the same `CallbackSet`; the `rows`/`cols` flattening convention is stated once in
  Task 9 and reused by its verbs.
- Seeded-stream invariant: Task 6 adds the three zero-tolerance cross-language digest cases.
- Class-layout note: all P2 additions are new headers plus additive members on `CallbackSet`
  (a POD-ish struct in the glue path) -- the preclean R rebuild is exercised in Task 13
  regardless.
- Deliberate non-placeholders: port tasks cite the exact C# source as the implementation
  reference (the established port convention); code blocks appear where the code is genuinely
  new (glue entry points, dispatch shapes, option surfaces).
