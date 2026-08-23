# P3 Optimizers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the remaining Numerics optimization layer -- the four global optimizers
(ParticleSwarm, ShuffledComplexEvolution, SimulatedAnnealing, MultiStart), the three local
optimizers (ADAM, GoldenSection, GradientDescent), the constrained optimizer (AugmentedLagrange
with its Constraint / ConstraintType / IConstraint types), and the dynamic-programming trio
(BinaryHeap, Dijkstra, Network) -- validated by ctest transcriptions of the C# test files, pinned
by fixtures through all four runners with dotnet-gate reproduction, exposed in R and Python,
documented, shipped as branch `port-optimizers` at v0.10.0.

**Architecture:** The seven new optimizers are real `Optimizer` subclasses dispatched through the
existing `optimizer_runner.hpp` as new `method` names on `optim_minimize()` / `optim_maximize()`.
Two of them (ADAM, GradientDescent) take an optional analytic gradient callback and
AugmentedLagrange takes one constraint callback per constraint, so the runner's single-objective
entry point grows an `OptimCallbacks` struct carrying the objective plus those extras, every one of
them guarded through a shared `CallbackAbortState` exactly as `callback_runner.hpp` guards its
groups. The dynamic trio is not an optimizer at all (its input is a graph, not a callable), so it
joins the toolbox as a new one-header `network` group on the same pattern P2's `linalg` / `special`
/ `functions` groups established, reaching users as `shortest_path()`.

**Tech Stack:** C++17 header-only core, cpp11 (R), pybind11 (Python), nlohmann/json fixture runner,
dotnet 10 oracle emitter.

## Global Constraints

- Every ported file carries `// ported from: <path> @ 2a0357a` and mirrors the C# class/method
  layout line-for-line where possible.
- Upstream paths contain spaces (`Numerics/Mathematics/Optimization/Global/`, `.../Local/`,
  `.../Constrained/Constraint/`, `.../Dynamic/`) -- quote them in every shell command. Read the C#
  from the submodule at its pinned SHA (`git -C upstream/Numerics show 'HEAD:<path>'`); the tree is
  clean and Numerics is valid UTF-8, so grep over it is fine (unlike RMC.BestFit).
- Oracle values live ONLY in `fixtures/*.json`; ctest suites transcribing C# test methods carry
  C#-test-literal assertions (the established second oracle class); never invent expected values.
- Every new fixture case gets a dotnet emitter driver; `python3 tools/verify_oracles.py` must end
  `0 failed`. Every new fixture-catalog objective, gradient, or constraint function is defined in
  ALL FOUR catalogs (C++ `core/tests/optimization_test_functions.hpp`, R `test-fixtures.R`
  `optimizer_fixture_objective`, Python `test_fixtures.py` `_optimizer_fixture_objective`, C#
  `Program.cs` `OptimizerTestFunction`) as plain arithmetic with explicit loops -- no
  `sum()`/`mean()`/LINQ -- because the C++ fixture catalog TU is compiled `-ffp-contract=off` and
  stands in for host-language arithmetic.
- Seeded global optimizers get cross-language digest cases in
  `fixtures/toolbox/toolbox_cross_language.json`. The measured, documented cross-language limit
  from P2 carries forward verbatim and is NOT re-litigated: a seeded run's PARAMETERS come from the
  shared C++ PRNG and reproduce bit-exact, but the reported objective VALUE is the user's own
  R/Python arithmetic and does not. Assert what reproduces; if a new method's parameters do NOT
  reproduce bit-for-bit either, measure why, write the measurement down, and assert the
  deterministic invariants that do -- never loosen a tolerance and never add an `oracle_skip` mask.
- If a stochastic ctest assertion fails where the C# test passes, the PRNG stream diverged -- fix
  the port, do not widen the tolerance.
- No `M_PI` (use `corehydro::numerics::kPi`); no namespace aliases named `gamma` or `stat`; no
  file-local `const` used implicitly in a capture-less lambda (MSVC C3493 -- use file-scope
  `constexpr`); no new external C++ dependencies.
- Task 1 changes the `Optimizer` base class layout (a public objective-function setter), so EVERY
  R rebuild in this phase uses `R CMD INSTALL --preclean corehydror`.
- After editing any `corehydror/src/*.cpp`: `Rscript -e 'cpp11::cpp_register("corehydror")'`.
- pytest reads fixtures materialized by pip; re-run
  `pixi run python -m pip install --force-reinstall --no-deps ./corehydropy` after fixture edits.
- Every new R export goes in BOTH `corehydror/_pkgdown.yml` and `site/_quarto.yml`
  `quartodoc.sections`; new Python exports also join the module `__all__`.
- Commits are GPG-signed, identity `Cam Bracken <cameron.bracken@pm.me>`, no Co-Authored-By
  trailers. Push only when Cam asks (the ship task pushes; that is the standing exception).
- Branch base: `port-math-extras` (PR #27 is open and stacked on PR #26; if both merge first,
  `main` is identical -- rebase then).
- Baseline numbers at branch creation, from the P2 release commit: ctest 98/98 (test_fixtures 5762
  checks); oracle gate 5751 reproduced / 0 failed / 11 skipped; testthat 6622/0; pytest 1619. Task 1
  re-measures ctest on its first full run; if any number differs, the fresh measurement governs
  (the standing process note: never reconcile against a stale census).
- Faithfulness note the porter must NOT "improve": both MLSL and MultiStart accept a `LocalMethod`
  but construct only the BFGS / NelderMead / Powell branches, throwing "Unsupported local method"
  for ADAM and GradientDescent. That is upstream behavior; mirror it (see
  `optimization/support/local_method.hpp`'s header, which already documents it for MLSL).

---

### Task 1: Branch, `Optimizer` objective setter, and the three local optimizers

**Files:**
- Modify: `core/include/corehydro/numerics/math/optimization/support/optimizer.hpp` (public
  objective-function accessor + setter)
- Create: `core/include/corehydro/numerics/math/optimization/adam.hpp`
- Create: `core/include/corehydro/numerics/math/optimization/golden_section.hpp`
- Create: `core/include/corehydro/numerics/math/optimization/gradient_descent.hpp`
- Create: `core/tests/test_local_optimizers.cpp`
- Modify: `core/CMakeLists.txt` (`BF_TESTS` += `test_local_optimizers`)

**Interfaces:**
- Produces, on `Optimizer` (namespace `corehydro::numerics::math::optimization`), mirroring C#'s
  `public Func<double[], double> ObjectiveFunction { get; set; }` whose setter null-checks:

```cpp
    // The objective function to evaluate. C# exposes this as a public settable property whose
    // setter throws on null; AugmentedLagrange REPLACES the inner optimizer's objective with its
    // own augmented Lagrangian through it, which is why it is public rather than protected.
    const Objective& objective_function() const { return objective_function_; }
    void set_objective_function(Objective value) {
        if (!value) throw ArgumentException("The objective function cannot be null.");
        objective_function_ = std::move(value);
    }
```

- Produces, three `class X : public Optimizer` headers (style model: `bfgs.hpp`):
  - `adam.hpp`: `ADAM(Objective objective_function, int number_of_parameters,
    std::vector<double> initial_values, std::vector<double> lower_bounds,
    std::vector<double> upper_bounds, double alpha = 0.001,
    std::function<std::vector<double>(const std::vector<double>&)> gradient = nullptr)`; public
    fields `alpha` (ctor-set), `beta1 = 0.9`, `beta2 = 0.999`, `gradient`; accessors
    `initial_values()`, `lower_bounds()`, `upper_bounds()`; `void optimize() override`.
  - `gradient_descent.hpp`: `GradientDescent(...)` -- identical ctor shape and identical public
    surface minus `beta1`/`beta2`.
  - `golden_section.hpp`: `GoldenSection(std::function<double(double)> objective_function,
    double lower_bound, double upper_bound)` (the base is constructed with
    `[f](std::vector<double>& x) { return f(x[0]); }, 1`, exactly as C# does);
    accessors `lower_bound()`, `upper_bound()`; `void optimize() override`.
- Consumed by: Task 5 (`golden_section` runner arm), Task 6 (`adam` / `gradient_descent` runner
  arms), Task 7 (the objective setter, for AugmentedLagrange).

- [ ] **Step 1: Create the branch and record the baseline**

```bash
cd /Users/cam/projects/usace/rmc/corehydro
git checkout port-math-extras && git pull && git checkout -b port-optimizers
cmake -S core -B core/build && cmake --build core/build -j && ctest --test-dir core/build
```
Expected: 98/98. Record the actual number; every later "baseline + N" in this plan is relative to
what this run reports.

- [ ] **Step 2: Read the C# sources and tests**

```bash
cd /Users/cam/projects/usace/rmc/corehydro
for f in "Local/ADAM" "Local/GoldenSection" "Local/GradientDescent" "Support/Optimizer"; do
  git -C upstream/Numerics show "HEAD:Numerics/Mathematics/Optimization/$f.cs" \
    > /tmp/$(basename "$f").cs
done
for f in "Local/Test_Adam" "Local/Test_GoldenSection" "Local/Test_GradientDescent"; do
  git -C upstream/Numerics show "HEAD:Test_Numerics/Mathematics/Optimization/$f.cs" \
    > /tmp/$(basename "$f").cs
done
git -C upstream/Numerics show "HEAD:Test_Numerics/Mathematics/Optimization/TestFunctions.cs" \
  > /tmp/TestFunctions.cs
```

All sixteen `TestFunctions.cs` objectives are ALREADY ported at
`core/tests/optimization_test_functions.hpp` (`fx`, `fxyz`, `de_jong`, `sum_of_power_functions`,
`rosenbrock`, `booth`, `matyas`, `mccormick`, `rastrigin`, `ackley`, `beale`, `goldstein_price`,
`bukin`, `three_hump_camel`, `eggholder`, `tp2`). Do not re-port them.

- [ ] **Step 3: Write the failing ctest suite first**

Create `core/tests/test_local_optimizers.cpp` transcribing all 17 C# test methods 1:1: Test_Adam's
7 (`Test_FXYZ`, `Test_DeJong`, `Test_SumOfPowerFunctions`, `Test_Rosenbrock`, `Test_Booth`,
`Test_Matyas`, `Test_McCormick`), Test_GradientDescent's identical 7, and Test_GoldenSection's 3
(`Test_Minimize`, `Test_Maximize`, `Test_DeJong`). Copy the harness style from
`core/tests/test_root_finding_extras.cpp` (`tests/check.hpp` macros, one function per C# test
method, `main` calling each), and include `optimization_test_functions.hpp` for the objectives.
Each test's bounds, initial values, per-solver property overrides (several set `MaxIterations` or
`Alpha` inline -- transcribe every one) and assertion tolerances come from the C# file verbatim.
Register `test_local_optimizers` in `BF_TESTS` in `core/CMakeLists.txt`.

```bash
cmake -S core -B core/build && cmake --build core/build --target test_local_optimizers -j
```
Expected: FAIL to compile -- the three headers do not exist.

- [ ] **Step 4: Add the objective setter, then port the three headers**

Apply the `objective_function()` / `set_objective_function()` block from the Interfaces section to
`optimizer.hpp` (public section, beside `number_of_parameters()`), and note in that file's header
comment that the setter exists because AugmentedLagrange replaces the inner optimizer's objective.

Then transcribe the three C# files member-for-member. Conventions from the C# source: ADAM and
GradientDescent call `NumericalDerivative.Gradient((x) => Evaluate(x, ref cancel), p)` when no
analytic gradient was supplied -- use the ported
`corehydro::numerics::math::differentiation::gradient` with the same default step behavior; ADAM's
`Tools.DoubleMachineEpsilon` becomes `std::numeric_limits<double>::epsilon()` with a
`// C#: Tools.DoubleMachineEpsilon` comment; GoldenSection's two golden-ratio constants
(`R = 0.61803399`, `C = 1.0 - R`) go at file scope as `constexpr` (MSVC C3493). Every validation
message transcribes verbatim.

- [ ] **Step 5: Run the suite until green, then the full ctest**

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_local_optimizers
ctest --test-dir core/build
```
Expected: new suite passes; full suite green at baseline + 1 (99/99 at the 98 baseline).

- [ ] **Step 6: Commit**

```bash
git add core/include/corehydro/numerics/math/optimization/{adam,golden_section,gradient_descent}.hpp \
        core/include/corehydro/numerics/math/optimization/support/optimizer.hpp \
        core/tests/test_local_optimizers.cpp core/CMakeLists.txt
git commit -m "feat: port the ADAM, GoldenSection, and GradientDescent optimizers"
```

---

### Task 2: ParticleSwarm and ShuffledComplexEvolution

**Files:**
- Create: `core/include/corehydro/numerics/math/optimization/particle_swarm.hpp`
- Create: `core/include/corehydro/numerics/math/optimization/shuffled_complex_evolution.hpp`
- Create: `core/tests/test_global_optimizers.cpp`
- Modify: `core/CMakeLists.txt` (`BF_TESTS` += `test_global_optimizers`)

**Interfaces:**
- Consumes: `Optimizer` base; `sampling::MersenneTwister` (`next_double()`, `next()`);
  `sampling::LatinHypercube::random(sample_size, dimension, seed)` returning a
  `math::linalg::Matrix2D`; `data::RunningStatistics` (`push(double)`, `mean()`,
  `standard_deviation()`); `ParameterSet`.
- Produces, in `corehydro::numerics::math::optimization`:
  - `ParticleSwarm(Objective objective_function, int number_of_parameters,
    std::vector<double> lower_bounds, std::vector<double> upper_bounds)`; public fields
    `population_size = 30`, `prng_seed = 12345`; accessors `lower_bounds()`, `upper_bounds()`;
    `void optimize() override`; private nested `struct Particle { ParameterSet parameter_set;
    ParameterSet best_parameter_set; std::vector<double> velocity; }`.
  - `ShuffledComplexEvolution(Objective objective_function, int number_of_parameters,
    std::vector<double> lower_bounds, std::vector<double> upper_bounds)`; public fields
    `prng_seed = 12345`, `complexes = 5`, `cce_iterations` (ctor-set to
    `2 * number_of_parameters + 1`), `tolerance_steps = 20`; accessors `lower_bounds()`,
    `upper_bounds()`; `void optimize() override`; private helpers `evolve_complex`, `is_feasible`,
    `reflection`, `contraction`, `smallest_hypercube`, `trapezoidal`, and the nested
    `struct PointFitness { ParameterSet parameter_set; int index; PointFitness clone() const; }`.
- Consumed by: Task 5 (the `particle_swarm` / `sce` runner arms).

- [ ] **Step 1: Read the C# sources and tests**

```bash
for f in "Global/ParticleSwarm" "Global/ShuffledComplexEvolution"; do
  git -C upstream/Numerics show "HEAD:Numerics/Mathematics/Optimization/$f.cs" \
    > /tmp/$(basename "$f").cs
done
for f in "Global/Test_ParticleSwarm" "Global/Test_ShuffledComplexEvolution"; do
  git -C upstream/Numerics show "HEAD:Test_Numerics/Mathematics/Optimization/$f.cs" \
    > /tmp/$(basename "$f").cs
done
```

- [ ] **Step 2: Write the failing ctest suite first**

Create `core/tests/test_global_optimizers.cpp` transcribing 28 test methods 1:1 -- Test_ParticleSwarm's
14 and Test_ShuffledComplexEvolution's 14, both files carrying the same names in the same order
(`Test_FXYZ`, `Test_DeJong`, `Test_SumOfPowerFunctions`, `Test_Rosenbrock`, `Test_Booth`,
`Test_Matyas`, `Test_McCormick`, `Test_Beale`, `Test_GoldsteinPrice`, `Test_Rastrigin`,
`Test_Ackley`, `Test_ThreeHumpCamel`, `Test_Eggholder`, `Test_TP2`). Two transcription details the
C# carries and the port must keep: `Test_Rosenbrock` sets `MaxIterations = 100000` for
ParticleSwarm but not for ShuffledComplexEvolution, and `Test_TP2` asserts a DISJUNCTION
(`match1 || match2` over the two symmetric optima), not two separate equalities. Name the C++
functions `pso_test_fxyz` / `sce_test_fxyz` and so on so both families live in one file without
collision. Register `test_global_optimizers` in `BF_TESTS`.

```bash
cmake --build core/build --target test_global_optimizers -j
```
Expected: FAIL to compile.

- [ ] **Step 3: Port the two headers**

Transcribe each C# file member-for-member. Details that are oracle-visible and easy to get wrong:

- ParticleSwarm draws its initial population from `LatinHypercube.Random(PopulationSize, D,
  PRNGSeed)` AND a separate `new MersenneTwister(PRNGSeed)` for velocities; both are seeded with
  the same value and are independent streams. The inertia weight is recomputed INSIDE the
  parameter loop (`w = wmax - (wmax - wmin) * Iterations / MaxIterations`, with an integer
  `Iterations`/`MaxIterations` division in C# that promotes to double because `(wmax - wmin)` is
  double -- write it so the arithmetic order matches). The four Alam constants (`wmin = 0.4`,
  `wmax = 0.9`, `c1 = c2 = 2.05`) go at file scope as `constexpr`.
- ShuffledComplexEvolution seeds one `MersenneTwister(PRNGSeed)` and then draws `Complexes`
  child seeds from it via `prng.Next()` -- the ported `next()` must be used, not `next_double()`.
  Its two `List<T>.Sort` calls use an explicit comparison that falls back to `Index` on a fitness
  tie; `std::sort` is NOT a faithful stand-in for `List<T>.Sort` in general (see
  `models/data_frame/data_frame_plotting.hpp`'s ported .NET introsort), but here the comparator is
  a total order with no ties, so `std::sort` with the same comparator is exact -- write that
  reasoning into the header as a comment so a future reader does not have to re-derive it.
  `Array.FindIndex(cdf, p => rnd <= p)` returning -1 falls back to `cdf.Length - 1`; keep it.

- [ ] **Step 4: Green, then full ctest, then commit**

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_global_optimizers
ctest --test-dir core/build
git add core/include/corehydro/numerics/math/optimization/{particle_swarm,shuffled_complex_evolution}.hpp \
        core/tests/test_global_optimizers.cpp core/CMakeLists.txt
git commit -m "feat: port the ParticleSwarm and ShuffledComplexEvolution optimizers"
```
Expected: full ctest green at baseline + 2 (100/100 at the 98 baseline).

---

### Task 3: SimulatedAnnealing and MultiStart

**Files:**
- Create: `core/include/corehydro/numerics/math/optimization/simulated_annealing.hpp`
- Create: `core/include/corehydro/numerics/math/optimization/multi_start.hpp`
- Modify: `core/tests/test_global_optimizers.cpp` (28 more transcribed methods)

**Interfaces:**
- Consumes: `Optimizer` base; `MersenneTwister`; `math::linalg::Vector` (`operator+`,
  `operator*(Vector, double)`, `to_array()`); `distributions::Uniform` (`inverse_cdf`);
  `LocalMethod`; `BFGS`, `NelderMead`, `Powell`.
- Produces, in `corehydro::numerics::math::optimization`:
  - `SimulatedAnnealing(Objective objective_function, int number_of_parameters,
    std::vector<double> lower_bounds, std::vector<double> upper_bounds)`; public fields
    `prng_seed = 12345`, `initial_temperature = 10`, `min_temperature = 0.1`,
    `cooling_rate = 0.95`, `update_cycles = 4`, `temperature_cycles = 10`,
    `tolerance_steps = 20`; accessors `lower_bounds()`, `upper_bounds()`;
    `void optimize() override`.
  - `MultiStart(Objective objective_function, int number_of_parameters,
    std::vector<double> initial_values, std::vector<double> lower_bounds,
    std::vector<double> upper_bounds, LocalMethod method = LocalMethod::BFGS)`; public fields
    `prng_seed = 12345`, `method`, `local_absolute_tolerance = 1E-8`,
    `local_relative_tolerance = 1E-8`, `polish = true`; accessors `initial_values()`,
    `lower_bounds()`, `upper_bounds()`; `void optimize() override`; private
    `std::unique_ptr<Optimizer> get_local_optimizer(std::vector<double>& initial_values,
    double relative_tolerance, double absolute_tolerance, bool& cancel)`.
- Consumed by: Task 5 (the `simulated_annealing` / `multi_start` runner arms).

- [ ] **Step 1: Read the C# sources and tests**

```bash
for f in "Global/SimulatedAnnealing" "Global/MultiStart"; do
  git -C upstream/Numerics show "HEAD:Numerics/Mathematics/Optimization/$f.cs" \
    > /tmp/$(basename "$f").cs
done
for f in "Global/Test_SimulatedAnnealing" "Global/Test_MultiStart"; do
  git -C upstream/Numerics show "HEAD:Test_Numerics/Mathematics/Optimization/$f.cs" \
    > /tmp/$(basename "$f").cs
done
```

- [ ] **Step 2: Extend the failing ctest suite**

Add 28 more transcribed methods to `core/tests/test_global_optimizers.cpp` (`sa_test_*` and
`ms_test_*`), the same 14 names each. Transcription details: Test_SimulatedAnnealing's
`Test_FXYZ` overrides `AbsoluteTolerance = 1E-6, RelativeTolerance = 1E-4` inline and the whole
file asserts at `1E-2` (with `Test_Rosenbrock`'s per-parameter assertions at `1E-1`) -- take every
tolerance from the C# file, never a rounder number. Test_MultiStart's cases all pass an `initial`
vector; transcribe each.

```bash
cmake --build core/build --target test_global_optimizers -j
```
Expected: FAIL to compile.

- [ ] **Step 3: Port the two headers**

Details that are oracle-visible:

- `SimulatedAnnealing::optimize` NEVER converges early -- there is no `CheckConvergence` call and
  the only exit is `UpdateStatus(OptimizationStatus.MaximumIterationsReached)`. Do not "fix" that;
  note it in the header (`tolerance_steps` is declared and validated but unused by the algorithm,
  exactly as upstream). The Corana step-size update, the `continue` that skips an out-of-bounds
  proposal WITHOUT evaluating it, and the end-of-outer-iteration reset of `x`/`fx` to
  `BestParameterSet` are all load-bearing. `acceptances.Fill(0)` is a Numerics array extension --
  use `std::fill`.
- `MultiStart::optimize` sets `MaxIterations = 100` in its CONSTRUCTOR (not as a field default),
  runs the local solver once per iteration starting from `InitialValues` at iteration 0, and then
  polishes from `BestParameterSet` when `Polish` is true. `GetLocalOptimizer` mutates the
  `initialValues` list it is handed via `RepairParameter` (C# passes `IList<double>` by reference)
  -- take a `std::vector<double>&` so the repair is observable to the caller exactly as in C#. Each
  local solver is constructed over `(x) => Evaluate(x, ref localCancel)` -- the PARENT's `Evaluate`,
  so the parent's best-tracking and evaluation budget see every inner evaluation -- with
  `MaxFunctionEvaluations = MaxFunctionEvaluations - FunctionEvaluations`. The ADAM /
  GradientDescent arms throw "Unsupported local method"; keep that (see Global Constraints).

- [ ] **Step 4: Green, then full ctest, then commit**

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_global_optimizers
ctest --test-dir core/build
git add core/include/corehydro/numerics/math/optimization/{simulated_annealing,multi_start}.hpp \
        core/tests/test_global_optimizers.cpp
git commit -m "feat: port the SimulatedAnnealing and MultiStart optimizers"
```
Expected: full ctest green at baseline + 2 (100/100 -- no new suite, 56 more checks in the
existing one).

---

### Task 4: The constraint layer and AugmentedLagrange

**Files:**
- Create: `core/include/corehydro/numerics/math/optimization/constraint/constraint_type.hpp`
- Create: `core/include/corehydro/numerics/math/optimization/constraint/i_constraint.hpp`
- Create: `core/include/corehydro/numerics/math/optimization/constraint/constraint.hpp`
- Create: `core/include/corehydro/numerics/math/optimization/augmented_lagrange.hpp`
- Create: `core/tests/test_augmented_lagrange.cpp`
- Modify: `core/CMakeLists.txt` (`BF_TESTS` += `test_augmented_lagrange`)

**Interfaces:**
- Consumes: `Optimizer` (including Task 1's `set_objective_function`), `BFGS`, `ParameterSet`.
- Produces, in `corehydro::numerics::math::optimization`:
  - `enum class ConstraintType { EqualTo, GreaterThanOrEqualTo, LesserThanOrEqualTo };` (C#
    declaration order -- the enum's integer values are not oracle-visible, but keep the order).
  - `class IConstraint` -- pure-virtual `type()`, `number_of_parameters()`, `function()` (returning
    `const std::function<double(const std::vector<double>&)>&`), `value()`, `tolerance()`; virtual
    destructor.
  - `class Constraint : public IConstraint` -- `Constraint(std::function<double(const
    std::vector<double>&)> constraint_function, int number_of_parameters, double value,
    ConstraintType type, double tolerance = 1E-8)` plus the five overrides.
  - `class AugmentedLagrange : public Optimizer` -- `AugmentedLagrange(Objective
    objective_function, Optimizer& optimizer, std::vector<std::shared_ptr<IConstraint>>
    constraints)`; accessors `Optimizer& optimizer()`, `const std::vector<double>& lambda()`,
    `mu()`, `nu()`, `const std::vector<std::shared_ptr<IConstraint>>& constraints()`;
    `void optimize() override`; private `double augmented_lagrangian_function(const
    std::vector<double>& x)`.
- Consumed by: Task 7 (the `augmented_lagrange` runner arm).

- [ ] **Step 1: Read the C# sources and tests**

```bash
for f in "Constrained/AugmentedLagrange" "Constrained/Constraint/Constraint" \
         "Constrained/Constraint/ConstraintType" "Constrained/Constraint/IConstraint"; do
  git -C upstream/Numerics show "HEAD:Numerics/Mathematics/Optimization/$f.cs" \
    > /tmp/$(basename "$f").cs
done
git -C upstream/Numerics show \
  "HEAD:Test_Numerics/Mathematics/Optimization/Constrained/Test_AugmentedLagrange.cs" \
  > /tmp/Test_AugmentedLagrange.cs
```

- [ ] **Step 2: Write the failing ctest suite first**

Create `core/tests/test_augmented_lagrange.cpp` transcribing all 6 C# test methods 1:1: `Test_1`,
`Test_2`, `Test_Haimes_5_2`, `Test_RosenbrockDisk`, `Test_MixedConstraints`,
`Test_MixedConstraints_Binding`. Each defines its objective and constraint functions inline in the
C# -- transcribe them as file-local statics with explicit loops (no accumulating helper), including
`Test_1`'s `NB[i] = (20 * x[i] - x[i] * x[i] - 24) / Math.Pow(1.10, i)` and its `-Tools.Sum(NB)`
return. `double.MinValue` / `double.MaxValue` bounds in `Test_1` and `Test_2` become
`std::numeric_limits<double>::lowest()` / `::max()` (C#'s `double.MinValue` IS the most negative
finite double, i.e. `lowest()`, NOT `min()`). Assert the point, the fitness (note `Test_1`/`Test_2`
assert `-solver.BestParameterSet.Fitness` while `Test_Haimes_5_2`/`Test_RosenbrockDisk` assert it
unnegated -- transcribe each as written), and the multipliers (`Lambda[0]`, `Mu[0]`), at the C#
tolerances (`1E-3`, `1E-2`, `1E-4`, `0.1`, `0.5`, and `Test_RosenbrockDisk`'s `Mu[0]` at EXACT
equality). Register `test_augmented_lagrange` in `BF_TESTS`.

```bash
cmake --build core/build --target test_augmented_lagrange -j
```
Expected: FAIL to compile.

- [ ] **Step 3: Port the four headers**

The three constraint files are near-mechanical. For `augmented_lagrange.hpp`:

- The C# holds `public Optimizer Optimizer { get; }` -- a live reference to a caller-owned object
  whose `ObjectiveFunction` the constructor OVERWRITES with `augmentedLagrangianFunction`. Store an
  `Optimizer*` (bound from the ctor's `Optimizer&`) and document in the header that the inner
  optimizer must outlive the AugmentedLagrange and that its objective is replaced on construction
  -- both are C# semantics, not a port choice.
- The ctor rejects an AugmentedLagrange inner optimizer (`optimizer.GetType() ==
  typeof(AugmentedLagrange)`); use `dynamic_cast<AugmentedLagrange*>(&optimizer) != nullptr`.
- `rho = 1`, `rhoMax = 1e+1`, `rhoMin = 1e-6` are private fields, and `tau = 0.5`, `gam = 10` are
  the Birgin & Martinez constants inside `Optimize()`; put the latter two at file scope as
  `constexpr` (MSVC C3493).
- The three multiplier vectors are sized by COUNTING constraints of each type in the ctor and are
  indexed by three separate running indices inside both loops. Keep the counting, the ordering, and
  the `Math.Min(Math.Max(...))` clamps exactly.
- `augmented_lagrangian_function` calls `_primaryObjectiveFunction` DIRECTLY (not through
  `Evaluate`), so inner-optimizer evaluations do NOT update this optimizer's own best/trace; only
  the explicit `Evaluate(currentValues, ref cancel)` calls in `Optimize()` do. That asymmetry is
  load-bearing for the fitness the tests assert -- do not route it through `evaluate()`.

- [ ] **Step 4: Green, then full ctest, then commit**

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_augmented_lagrange
ctest --test-dir core/build
git add core/include/corehydro/numerics/math/optimization/constraint/*.hpp \
        core/include/corehydro/numerics/math/optimization/augmented_lagrange.hpp \
        core/tests/test_augmented_lagrange.cpp core/CMakeLists.txt
git commit -m "feat: port the Constraint types and the AugmentedLagrange optimizer"
```
Expected: full ctest green at baseline + 3 (101/101 at the 98 baseline).

---

### Task 5: Runner and user surface for the five gradient-free methods

**Files:**
- Modify: `core/include/corehydro/numerics/support/optimizer_runner.hpp` (five new arms, new
  control knobs, `local_method` on `mlsl`)
- Modify: `corehydror/R/optim.R` (method vocabulary + per-method control table)
- Modify: `corehydropy/src/corehydropy/optim.py` (the same, verb for verb)
- Modify: `fixtures/toolbox/optimizers.json`, `fixtures/toolbox/toolbox_cross_language.json`,
  `fixtures/README.md`
- Modify: `tools/oracle_emitter/Program.cs` (new optimizer arms + control plumbing)
- Test: `corehydror/tests/testthat/test-optim.R`,
  `corehydropy/tests/test_optim.py` (locate with `grep -rln optim_minimize corehydror/tests corehydropy/tests`)

**Interfaces:**
- Consumes: Tasks 1-3's `GoldenSection`, `ParticleSwarm`, `ShuffledComplexEvolution`,
  `SimulatedAnnealing`, `MultiStart`.
- Produces, on `run_optimizer`'s `method` vocabulary: `"particle_swarm"`, `"sce"`,
  `"simulated_annealing"`, `"multi_start"`, `"golden_section"` beside the existing six.
  Requirements per method: `"golden_section"` needs `lower`/`upper` only (1-D, like `"brent"`);
  `"particle_swarm"`, `"sce"` and `"simulated_annealing"` need `lower`/`upper`; `"multi_start"`
  additionally needs `initial`. All five are Optimizer-base methods, so all five accept
  `max_function_evaluations` / `report_failure` / `compute_hessian`; all except
  `"golden_section"` accept `seed`.
- Produces, new `control` keys (each applied only when present, leaving the ported class default
  otherwise): `population_size` (now `"de"` AND `"particle_swarm"`), `complexes`,
  `cce_iterations`, `tolerance_steps` (`"sce"`, and `"simulated_annealing"` which declares but
  does not use it -- see Task 3), `initial_temperature`, `min_temperature`, `cooling_rate`,
  `update_cycles`, `temperature_cycles`, `local_method` (`"multi_start"` AND `"mlsl"`, one of
  `"bfgs"`, `"nelder_mead"`, `"powell"`), `local_absolute_tolerance`, `local_relative_tolerance`,
  `polish`.
- Produces, user-facing (identical names and defaults in R and Python): `optim_minimize()` /
  `optim_maximize()` gain the five method names and the new control keys; no new arguments.

- [ ] **Step 1: Replace the ad-hoc control validation with a per-method table**

`optim.R`'s current validation carries three separate special-case checks (`kOptimBaseMethods`
gating three control names, a `population_size`-is-`"de"`-only check, and
`kOptimNeedsInitial`). Eleven methods with overlapping control sets cannot be expressed that way
without the checks drifting apart from Python's. Replace them with one table, in both languages,
and drive every check off it:

```r
# Per-method: which control names it accepts, whether it needs `initial`, and whether it is
# stochastic (accepts `seed`). Every validation message below reads off this one table so the R
# and Python surfaces cannot drift.
kOptimMethods <- list(
  de                  = list(controls = c(kOptimBaseControls, "population_size"),
                             needs_initial = FALSE, stochastic = TRUE),
  particle_swarm      = list(controls = c(kOptimBaseControls, "population_size"),
                             needs_initial = FALSE, stochastic = TRUE),
  sce                 = list(controls = c(kOptimBaseControls, "complexes", "cce_iterations",
                                          "tolerance_steps"),
                             needs_initial = FALSE, stochastic = TRUE),
  simulated_annealing = list(controls = c(kOptimBaseControls, "initial_temperature",
                                          "min_temperature", "cooling_rate", "update_cycles",
                                          "temperature_cycles", "tolerance_steps"),
                             needs_initial = FALSE, stochastic = TRUE),
  multi_start         = list(controls = c(kOptimBaseControls, "local_method",
                                          "local_absolute_tolerance", "local_relative_tolerance",
                                          "polish"),
                             needs_initial = TRUE, stochastic = TRUE),
  mlsl                = list(controls = c(kOptimBaseControls, "local_method"),
                             needs_initial = TRUE, stochastic = TRUE),
  bfgs                = list(controls = kOptimBaseControls, needs_initial = TRUE,
                             stochastic = FALSE),
  powell              = list(controls = kOptimBaseControls, needs_initial = TRUE,
                             stochastic = FALSE),
  nelder_mead         = list(controls = kOptimScalarControls, needs_initial = TRUE,
                             stochastic = FALSE),
  brent               = list(controls = kOptimScalarControls, needs_initial = FALSE,
                             stochastic = FALSE),
  golden_section      = list(controls = kOptimBaseControls, needs_initial = FALSE,
                             stochastic = FALSE)
)
# The three tolerance/iteration knobs every method takes, plus the three only the Optimizer-base
# methods take. "nelder_mead"/"brent" are the two standalone classes (see optimizer_runner.hpp);
# "golden_section" IS an Optimizer subclass, so it takes the full base set.
kOptimScalarControls <- c("max_iterations", "absolute_tolerance", "relative_tolerance")
kOptimBaseControls <- c(kOptimScalarControls, "max_function_evaluations", "report_failure",
                        "compute_hessian")
```

(Define the two constant vectors ABOVE `kOptimMethods` in the file; they are shown after it here
only for readability.) Rewrite `optim_run`'s checks to read `kOptimMethods[[method]]`, and mirror
the whole table in `optim.py` as a module-level dict.

- [ ] **Step 2: Write failing R and Python tests**

In the existing optim test files add, for each of the five new methods, a minimization of the
Booth function (`(x + 2y - 7)^2 + (2x + y - 5)^2`, optimum `(1, 3)`) over `lower = c(-10, -10)`,
`upper = c(10, 10)` -- `initial = c(0, 0)` for `"multi_start"` -- asserting `parameters` within
`1e-3` of `c(1, 3)`; a `golden_section` minimization of `(x - 2)^2` on `[0, 5]` asserting the
parameter within `1e-4` of 2; and one negative test per language asserting that
`control = list(cooling_rate = 0.9)` on `method = "de"` errors with a message naming the method.
Run both suites; expected: FAIL (unknown method).

- [ ] **Step 3: Implement the five runner arms**

Each arm copies the shape of the existing `"de"` arm: construct with `adapted`, apply
`spec["seed"]` to the class's `prng_seed`, `apply_common_controls` + `apply_optimizer_controls`,
then the method's own control keys, then the `try { maximize/minimize } catch { rethrow_if_aborted;
throw; }` + `rethrow_if_aborted()` + `fill_optimizer_result` sequence. `"golden_section"` builds
its 1-D objective like the `"brent"` arm
(`[&guarded](double x) { return guarded(std::vector<double>{x}); }`) but, being a real Optimizer
subclass, DOES use `fill_optimizer_result` (unlike `"brent"`). Add the shared `local_method` parse
helper:

```cpp
// Parses the LocalMethod control shared by "mlsl" and "multi_start". Only the three methods the
// two classes actually construct are accepted (ADAM/GradientDescent throw "Unsupported local
// method" inside both -- see local_method.hpp), so rejecting them here names the option rather
// than surfacing the inner class's message.
inline opt::LocalMethod parse_local_method(const std::string& s) {
    if (s == "bfgs") return opt::LocalMethod::BFGS;
    if (s == "nelder_mead") return opt::LocalMethod::NelderMead;
    if (s == "powell") return opt::LocalMethod::Powell;
    throw std::runtime_error("unknown local_method: " + s +
                             " (expected \"bfgs\", \"nelder_mead\" or \"powell\")");
}
```

and apply it in both the `"mlsl"` and `"multi_start"` arms. Update the file-header grammar comment
(it currently says "one of six") and the `run_optimizer` doc block.

- [ ] **Step 4: Rebuild both packages and get the tests green**

```bash
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
```

- [ ] **Step 5: Fixture and emitter parity**

Extend `fixtures/toolbox/optimizers.json` with cases pinning each of the five methods against
`TestFunctions.cs` objectives, expected values transcribed from the C# test literals: at minimum
`particle_swarm_fxyz`, `particle_swarm_eggholder` (the hardest stream check in the phase --
`-959.6407` at `(512, 404.2319)`), `sce_dejong`, `sce_booth`, `simulated_annealing_booth` (at the
C# `1E-2` tolerance), `multi_start_fxyz`, `multi_start_rosenbrock`, and `golden_section_fx`
(Test_GoldenSection's `Test_Minimize` literals) plus `golden_section_fx_max` (`Test_Maximize`).
Add one `local_method` case (`multi_start` with `control.local_method = "powell"`) so the new
control has an oracle.

Add four seeded digest cases to `fixtures/toolbox/toolbox_cross_language.json` -- one per stochastic
new method (`particle_swarm`, `sce`, `simulated_annealing`, `multi_start`) -- asserting the
PARAMETERS at zero tolerance, not the objective value, exactly as the existing `de` sub-block does
and for the reason stated in Global Constraints. Run them in R and Python before pinning; if a
method's parameters do not agree bit-for-bit between the two, do not pin them: measure where the
streams part (compare the first differing evaluation), write the measurement into the fixture's
`reference` string and into `docs/upstream-csharp-issues.md` if the cause is upstream, and assert
the deterministic invariants that do reproduce.

Emitter: extend the `kindStr == "optimizer"` branch's `method switch` (Program.cs ~line 5746) with
the five real C# classes, add the seed assignment for the four stochastic ones beside the existing
`DifferentialEvolution`/`MLSL` `is` checks, and add a control-application block reading
`construct.control` so the pinned `local_method` case drives the real `MultiStart.Method`. Document
the new methods and control keys in `fixtures/README.md`.

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: fixture runner green with new checks counted; gate `0 failed`, skips still 11.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: expose particle swarm, SCE, simulated annealing, multi-start and golden section"
```

---

### Task 6: The gradient-taking methods (ADAM, GradientDescent)

**Files:**
- Modify: `core/include/corehydro/numerics/support/optimizer_runner.hpp` (`OptimCallbacks`,
  `GuardedGradient`, two new arms)
- Modify: `corehydror/src/toolbox.cpp` (new `ch_optim_run_grad_` entry point),
  `corehydror/R/optim.R` (`gradient` argument)
- Modify: `corehydropy/src/bindings/toolbox.cpp` (`optim_run_grad`),
  `corehydropy/src/corehydropy/optim.py` (`gradient` argument)
- Modify: `core/tests/test_fixtures.cpp`, `corehydror/tests/testthat/test-fixtures.R`,
  `corehydropy/tests/test_fixtures.py` (the optimizer-kind runners gain a `gradient` catalog
  lookup), `core/tests/optimization_test_functions.hpp` (gradient catalog)
- Modify: `fixtures/toolbox/optimizers.json`, `fixtures/README.md`,
  `tools/oracle_emitter/Program.cs`

**Interfaces:**
- Consumes: Task 1's `ADAM` and `GradientDescent`.
- Produces, in `corehydro::numerics::support`:

```cpp
// A gradient callback: the trial point in, one partial derivative per parameter out.
using Gradient = std::function<std::vector<double>(const std::vector<double>&)>;

// Everything a run may need from the host language. `objective` is always required; `gradient` is
// read only by the "adam"/"gradient_descent" methods and `constraints` only by
// "augmented_lagrange" (Task 7), positionally matched to the spec's "constraints" array. Every
// callback present is guarded, and all the guards share ONE abort state, so a throw in any of them
// short-circuits the rest instead of re-entering the host mid-unwind.
struct OptimCallbacks {
    Objective objective;
    Gradient gradient;
    std::vector<Objective> constraints;
};

inline OptimResult run_optimizer(const std::string& spec_json, const OptimCallbacks& callbacks);
// Existing two-argument form, kept so every current caller compiles unchanged.
inline OptimResult run_optimizer(const std::string& spec_json, const Objective& objective) {
    return run_optimizer(spec_json, OptimCallbacks{objective, nullptr, {}});
}

using GuardedGradient = GuardedCall<std::vector<double>, const std::vector<double>&>;
```

  The gradient guard's sentinel is an EMPTY vector; both ported classes would then index out of
  range, so the arms must gate on `guarded.aborted()` -- see Step 3.
- Produces, on `run_optimizer`'s `method` vocabulary: `"adam"`, `"gradient_descent"`; both need
  `initial`/`lower`/`upper`, both accept the base control set plus `alpha`, and `"adam"` also
  `beta1` / `beta2`. Neither is stochastic (no `seed`).
- Produces, user-facing: `optim_minimize()` / `optim_maximize()` gain
  `gradient = NULL` (R) / `gradient=None` (Python) -- a function taking the parameter vector and
  returning a vector of the same length, accepted only by `"adam"` and `"gradient_descent"`, an
  error for every other method. Omitted, both classes fall back to the ported
  `NumericalDerivative.Gradient`, exactly as C# does.

- [ ] **Step 1: Write failing R and Python tests**

In the optim test files: minimize `f(p) = (p1 - 3)^2 + (p2 + 1)^2` from `initial = c(0, 0)` over
`[-10, 10]^2` with `method = "gradient_descent"` and an analytic
`gradient = function(p) c(2 * (p[1] - 3), 2 * (p[2] + 1))`, `control = list(alpha = 0.1)`,
asserting `parameters` within `1e-3` of `c(3, -1)`; the same with `method = "adam"`; the same
WITHOUT a gradient, asserting the numerically differentiated run lands within `1e-3` too; and a
negative test asserting `gradient =` on `method = "de"` errors naming the two methods that accept
it. Expected: FAIL.

- [ ] **Step 2: Implement the runner arms and both glues**

The two arms build the class with the guarded gradient only when one was supplied:

```cpp
    } else if (method == "adam" || method == "gradient_descent") {
        int D = static_cast<int>(initial.size());
        // Shares the objective guard's abort state so a throw in either callback stops both.
        GuardedGradient guarded_grad(callbacks.gradient, std::vector<double>{},
                                     guarded.abort_state());
        // The ported classes call the gradient unconditionally when the field is set, so the
        // field stays EMPTY unless the host supplied one -- an empty std::function is exactly
        // what C#'s null Gradient means, and both classes already branch on it.
        opt::ADAM::GradientFn grad_fn = nullptr;
        if (callbacks.gradient)
            grad_fn = [&guarded_grad](const std::vector<double>& p) { return guarded_grad(p); };
        ...
```

with `alpha` read from `control` (defaulting to the ctor's 0.001) and `beta1`/`beta2` applied for
ADAM only. Both arms use the standard try/catch + `rethrow_if_aborted()` + `fill_optimizer_result`
sequence; because the gradient guard shares the objective guard's state, `guarded.rethrow_if_aborted()`
alone covers both.

R glue -- a second entry point beside `ch_optim_run_`, since cpp11 registration is by signature:

```cpp
[[cpp11::register]]
list ch_optim_run_grad_(std::string spec_json, function objective, function gradient) {
    tb::OptimCallbacks cbs;
    cbs.objective = [&](const std::vector<double>& p) -> double {
        return cpp11::as_cpp<double>(objective(cpp11::as_sexp(p)));
    };
    cbs.gradient = [&](const std::vector<double>& p) -> std::vector<double> {
        return cpp11::as_cpp<std::vector<double>>(gradient(cpp11::as_sexp(p)));
    };
    return pack_optim(tb::run_optimizer(spec_json, cbs));
}
```

(match the surrounding file's actual conversion helpers and result packer -- read `ch_optim_run_`
first and copy its idioms exactly). Python mirrors it as `optim_run_grad`. `optim.R` /
`optim.py` call the gradient entry point when `gradient` is non-NULL and the existing one
otherwise.

```bash
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
```

- [ ] **Step 3: Fixture, catalog, and emitter parity**

Add an `optional gradient` key to the optimizer fixture grammar: `construct.gradient` names a
catalog gradient function, looked up in all four runners beside the objective. Define the gradient
catalog entries in all four catalogs -- `Grad_FXYZ`, `Grad_DeJong`, `Grad_Booth` -- as explicit
analytic derivatives written out term by term (no loops over a helper, no autodiff), per the
`-ffp-contract=off` rule in Global Constraints.

New `optimizers.json` cases from the C# test literals: `adam_fxyz`, `adam_booth`,
`gradient_descent_fxyz`, `gradient_descent_sum_of_power_functions` (Test_GradientDescent's
`Test_SumOfPowerFunctions` asserts its parameters at `1E-2`, looser than its `1E-4` fitness --
transcribe both), plus `adam_fxyz_analytic_gradient` and
`gradient_descent_booth_analytic_gradient` carrying `construct.gradient`. Emitter: two new arms
constructing the real C# `ADAM`/`GradientDescent`, a `CallbackOptimizerGradient(name)` catalog
mirroring the C++ one, and the `construct.gradient` plumbing into the ctor's `gradient` parameter.

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: green; `0 failed`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: expose ADAM and gradient descent with an optional analytic gradient"
```

---

### Task 7: The constrained surface (AugmentedLagrange)

**Files:**
- Modify: `core/include/corehydro/numerics/support/optimizer_runner.hpp` (`augmented_lagrange`
  arm, `OptimResult` multipliers)
- Modify: `corehydror/src/toolbox.cpp` (`ch_optim_run_constrained_`),
  `corehydror/R/optim.R` (`optim_constraint()` + `constraints` / `inner` arguments)
- Modify: `corehydropy/src/bindings/toolbox.cpp` (`optim_run_constrained`),
  `corehydropy/src/corehydropy/optim.py` (`Constraint` + the same arguments, `__all__`)
- Modify: the three fixture runners (constraint catalog lookup),
  `core/tests/optimization_test_functions.hpp`
- Modify: `fixtures/toolbox/optimizers.json`, `fixtures/README.md`,
  `tools/oracle_emitter/Program.cs`

**Interfaces:**
- Consumes: Task 4's `AugmentedLagrange` / `Constraint` / `ConstraintType`; Task 6's
  `OptimCallbacks`.
- Produces, on `OptimResult`, three new members (empty for every method except
  `"augmented_lagrange"`):

```cpp
    std::vector<double> lambda;  // equality multipliers
    std::vector<double> mu;      // "lesser than or equal to" multipliers
    std::vector<double> nu;      // "greater than or equal to" multipliers
```

- Produces, on the spec grammar:

```json
{"method": "augmented_lagrange",
 "constraints": [{"type": "eq", "value": 22.0, "tolerance": 1e-8}],
 "inner": {"method": "bfgs", "initial": [7, 7, 8], "lower": [...], "upper": [...],
           "control": {"max_iterations": 1000}},
 "control": {"max_iterations": 1000}}
```

  `constraints[i]` pairs positionally with `callbacks.constraints[i]`; `type` is `"eq"` |
  `"le"` | `"ge"` mapping onto `EqualTo` / `LesserThanOrEqualTo` / `GreaterThanOrEqualTo`;
  `tolerance` defaults to `1E-8`. `inner.method` is any non-`augmented_lagrange` method the runner
  already knows, built by the same arm the top-level dispatch uses; when `inner` is absent the
  inner optimizer is `"bfgs"` over the top-level `initial`/`lower`/`upper`.
- Produces, user-facing:
  - R `optim_constraint(f, value, type = c("eq", "le", "ge"), tolerance = 1e-8)` returning a
    `corehydro_constraint` classed list; Python `Constraint(function, value, type="eq",
    tolerance=1e-8)` (a dataclass).
  - `optim_minimize()` / `optim_maximize()` gain `constraints = NULL` (a list of
    `optim_constraint()` objects, required by and exclusive to `method = "augmented_lagrange"`) and
    `inner = NULL` (a named list `list(method =, initial =, lower =, upper =, control =)`,
    accepted only by that method).
  - The returned object gains `multipliers` -- a list/dict with `equality`, `less_than`,
    `greater_than` -- present only for `"augmented_lagrange"`.
- Produces, fixture assertion methods: `multiplier` with `args: [set, index]` where `set` is
  `"lambda"` | `"mu"` | `"nu"`.

- [ ] **Step 1: Write failing R and Python tests**

Transcribe `Test_Haimes_5_2` into both languages: objective `(x - 2)^2 + (y - 4)^2 + 5`, a single
`"le"` constraint `(x - 6)^2 + (y - 10)^2 + 6` at value `13.31`, inner BFGS from `c(5, 5)` over
`[0, 10]^2`; assert `parameters` within `1e-2` of `c(4.5, 7.75)`, `value` within `1e-2` of `25.31`,
and `multipliers$less_than[1]` within `1e-2` of `1.67`. Add a negative test: `constraints =` on
`method = "de"` errors naming `"augmented_lagrange"`. Expected: FAIL.

- [ ] **Step 2: Implement the runner arm**

Factor the existing per-method construction into a helper the arm can reuse for the inner
optimizer, rather than duplicating eleven arms:

```cpp
// Builds one optimizer from a spec object, for the top-level dispatch AND for
// augmented_lagrange's "inner" sub-spec. Returns null for the two standalone classes
// ("nelder_mead"/"brent"), which are not Optimizer subclasses and therefore cannot be an
// AugmentedLagrange inner optimizer -- the arm rejects them by name before calling this.
inline std::unique_ptr<opt::Optimizer> make_optimizer(const std::string& method,
                                                      const JsonValue& spec,
                                                      const opt::Optimizer::Objective& adapted);
```

The `"augmented_lagrange"` arm then: builds the inner optimizer through `make_optimizer` over the
`inner` sub-spec (defaulting to `"bfgs"` over the top-level vectors); wraps each host constraint
in its own `GuardedObjective` sharing `guarded.abort_state()`; builds one
`opt::Constraint` per spec entry over that guard; constructs
`opt::AugmentedLagrange(adapted, *inner, constraints)`; runs it through the standard try/catch +
`rethrow_if_aborted()` sequence; fills the result with `fill_optimizer_result` plus the three
multiplier vectors. Reject `inner.method == "augmented_lagrange"`, `"nelder_mead"` and `"brent"`
with named errors before construction.

- [ ] **Step 3: Both glues and both public surfaces**

R: `ch_optim_run_constrained_(std::string spec_json, function objective, list constraints)`
converting each element of `constraints` to an `Objective` (same conversion `ch_optim_run_` uses);
Python: `optim_run_constrained(spec_json, objective, constraints)` taking a list of callables.
`optim.R` / `optim.py` split each `optim_constraint()` object into its serializable half (into the
spec's `constraints` array, in order) and its function half (into the callback list, same order) --
say so in a comment, because the positional pairing is the contract.

```bash
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
```

- [ ] **Step 4: Fixture, catalog, and emitter parity**

Add the constraint catalog entries to all four catalogs, transcribed from the C# test bodies with
explicit loops: `AL1_Objective` (`-sum((20*x[i] - x[i]^2 - 24) / 1.10^i)`), `AL2_Objective`,
`Haimes_Primary`, `Haimes_Secondary`, `RosenbrockDisk_Objective`, `Disk` (`x^2 + y^2`), `SumAll`
(`Tools.Sum`), `X0`, `X1`. New `optimizers.json` cases pinning all four literal-bearing C# tests --
`augmented_lagrange_test1` (point, `-fitness`, `lambda[0]`), `augmented_lagrange_test2`,
`augmented_lagrange_haimes`, `augmented_lagrange_rosenbrock_disk` (its `Mu[0]` at EXACT equality,
`tol: 0`) -- plus `augmented_lagrange_mixed` from `Test_MixedConstraints`, which is the only case
exercising all three constraint types at once. Emitter: an `"augmented_lagrange"` arm building the
real C# `Constraint[]` and inner optimizer, and the `multiplier` assertion method reading
`solver.Lambda` / `.Mu` / `.Nu`.

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: green; `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: expose constrained optimization through augmented Lagrange"
```

---

### Task 8: BinaryHeap and Dijkstra

**Files:**
- Create: `core/include/corehydro/numerics/math/optimization/dynamic/binary_heap.hpp`
- Create: `core/include/corehydro/numerics/math/optimization/dynamic/dijkstra.hpp`
- Create: `core/tests/test_network_optimization.cpp`
- Modify: `core/CMakeLists.txt` (`BF_TESTS` += `test_network_optimization`)

**Interfaces:**
- Produces, in `corehydro::numerics::math::optimization`:
  - `template <typename T> class BinaryHeap` with nested
    `struct Node { float weight; int index; T value; Node(); Node(float, int, T); }`;
    `explicit BinaryHeap(int heap_size)`; `int count() const`; `void add(const Node&)`;
    `Node remove_min()`; `void decrease_key(const Node&)`; `void replace(const Node&)`; private
    `bubble_up(int)`, `bubble_down(int)`, a `std::vector<Node> heap_`, a
    `std::unordered_map<int, int> position_map_`, `int n_`.
  - `struct Edge { int from_index; int to_index; float weight; int index; Edge(); Edge(int, int,
    float, int); }`.
  - `namespace dijkstra` free functions (C#'s `static class Dijkstra`):
    `bool path_exists(const std::vector<std::array<float, 3>>& result_table, int node_index)`;
    `std::vector<std::array<float, 3>> solve(const std::vector<Edge>& edges, int
    destination_index, int node_count = -1, const std::vector<std::vector<Edge>>* edges_to_nodes =
    nullptr)`; and the `std::vector<int> destination_indices` overload with the same trailing
    parameters. The result table is C#'s `float[nNodes, 3]` with column order
    `{NEXT_NODE = 0, EDGE_INDEX = 1, COST = 2}` -- name those three as file-scope `constexpr int`
    exactly as C# names them.
- Consumed by: Task 9 (`Network`), Task 10 (the toolbox `network` group).

**WEIGHTS ARE `float`, NOT `double`.** C# declares `float Weight`, `float[,] resultTable` and
accumulates `float newCost = cost + edge.Weight`. Single-precision accumulation is observable in
the result table, so the port uses `float` throughout and only widens to `double` at the toolbox
boundary (Task 10). Do not "upgrade" it; state the reason in the header.

- [ ] **Step 1: Read the C# sources and tests**

```bash
for f in "Dynamic/BinaryHeap" "Dynamic/Dijkstra"; do
  git -C upstream/Numerics show "HEAD:Numerics/Mathematics/Optimization/$f.cs" \
    > /tmp/$(basename "$f").cs
done
for f in "Dynamic/BinaryHeapTesting" "Dynamic/DijkstraTesting"; do
  git -C upstream/Numerics show "HEAD:Test_Numerics/Mathematics/Optimization/$f.cs" \
    > /tmp/$(basename "$f").cs
done
```

- [ ] **Step 2: Write the failing ctest suite first**

Create `core/tests/test_network_optimization.cpp` transcribing all 17 methods 1:1:
BinaryHeapTesting's 9 (including `HeapCapacityExceededTest`, which asserts a throw -- transcribe
the message verbatim from the C# `InvalidOperationException("Heap is full.")`, and
`DecreaseKeyTest`) and DijkstraTesting's 8 (`SimpleEdgeGraphCost`, `SimpleNetworkRouting`,
`BidirectionalRouting`, `DisconnectedNodesTest`, `MultipleDestSharedPath`, `DisconnectedComponent`,
`DisconnectedComponent2`, `TrianglePath`). The Dijkstra tests assert exact `float` equality
(`Assert.AreEqual(3f, result[2, 2])`) and `float.IsPositiveInfinity` -- assert exact equality and
`std::isinf` in the port, not an epsilon comparison. `SimpleNetworkRouting` and later tests build a
`Network`; those specific methods stay commented out with a `// Task 9` marker until Network lands,
then get uncommented there. Register `test_network_optimization` in `BF_TESTS`.

```bash
cmake --build core/build --target test_network_optimization -j
```
Expected: FAIL to compile.

- [ ] **Step 3: Port the two headers**

`BinaryHeap` is a straight transcription; `Dictionary<int,int>` becomes `std::unordered_map<int,
int>` (only membership and lookup are used, never iteration order, so this is exact).
`Dijkstra::Solve`'s multi-destination overload calls the single-destination one per destination
and keeps the cheaper path; the single-destination overload's `Console.WriteLine` unreachable-node
report is a diagnostic with no return-value effect -- drop it and say so in a comment (a
`// severed: Console.WriteLine diagnostic, no numeric effect` line). Note in the header that the
C# builds `edgesFromNodes` from `edge.ToIndex` in BOTH overloads, including the one whose
parameter is named `edgesFromNodes` -- transcribe as written, do not "fix" the apparent naming
inconsistency.

- [ ] **Step 4: Green, then full ctest, then commit**

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_network_optimization
ctest --test-dir core/build
git add core/include/corehydro/numerics/math/optimization/dynamic/{binary_heap,dijkstra}.hpp \
        core/tests/test_network_optimization.cpp core/CMakeLists.txt
git commit -m "feat: port the BinaryHeap and Dijkstra shortest-path solver"
```
Expected: full ctest green at baseline + 4 (102/102 at the 98 baseline).

---

### Task 9: Network, and the `GetPath` defect investigation

**Files:**
- Create: `core/include/corehydro/numerics/math/optimization/dynamic/network.hpp`
- Modify: `core/tests/test_network_optimization.cpp` (uncomment the Network-dependent methods)
- Modify: `docs/upstream-csharp-issues.md` (a new entry, if the investigation confirms the defect)
- Modify: `upstream/CLAUDE.md` (a severance line, if `GetPath` is severed)

**Interfaces:**
- Consumes: Task 8's `Edge`, `BinaryHeap`, `dijkstra::solve`.
- Produces, in `corehydro::numerics::math::optimization`:
  `class Network` -- `Network(std::vector<Edge> edges, std::vector<int> destination_indices)`;
  accessors `destination_indices()`, `incoming_edges()`, `outgoing_edges()`; `std::vector<std::array<float, 3>>
  solve(int destination_index) const`, `solve(const std::vector<int>& destination_indices) const`,
  `solve(const std::vector<float>& edge_weights) const`. `get_path` is ported or severed per the
  Step 2 finding.
- Consumed by: Task 10.

- [ ] **Step 1: Read the C# source**

```bash
git -C upstream/Numerics show "HEAD:Numerics/Mathematics/Optimization/Dynamic/Network.cs" \
  > /tmp/Network.cs
```

- [ ] **Step 2: Investigate `GetPath` before porting it**

`Network.GetPath` contains `Array.BinarySearch(edgesToRemove, edge)` where `edgesToRemove` is
`int[]` and `edge` is an `Edge` struct. That resolves to `Array.BinarySearch(Array, object)`, whose
comparer calls `((IComparable)int).CompareTo(Edge)`. No upstream test covers `GetPath` (the eight
DijkstraTesting methods exercise `Dijkstra.Solve` and `Network.Solve` only). Determine empirically
what the real C# does rather than reasoning about it: write a throwaway program under
`/tmp/getpath_probe/` referencing the Numerics project, build a small graph, call
`GetPath(new[]{0}, startNode)`, and run it.

```bash
mkdir -p /tmp/getpath_probe && cd /tmp/getpath_probe
dotnet new console -o . --force
# add: <ProjectReference Include="/Users/cam/projects/usace/rmc/corehydro/upstream/Numerics/Numerics/Numerics.csproj" />
dotnet run
```

- If it throws (the expected outcome): port `get_path` structurally anyway, with a header note
  naming the exact line, the observed exception, and the fact that no upstream test reaches it;
  add an entry to `docs/upstream-csharp-issues.md` recording the probe output verbatim; do NOT
  expose it in Task 10's toolbox group, and record the severance in `upstream/CLAUDE.md`.
- If it returns a path: port it faithfully and expose it in Task 10 as a `network_path` method,
  pinned by a fixture case driven from the emitter like every other value.

Either way the finding is written down, with the command that produced it.

- [ ] **Step 3: Port `network.hpp` and uncomment the Network ctest methods**

The three `Solve` overloads are thin forwarders to `dijkstra::solve` over the cached
`_incomingEdges`. The constructor builds `_outgoingEdges` / `_incomingEdges` -- read the ctor body
and transcribe its indexing exactly (it is the source of the caching the `Solve` overloads pass
down). Uncomment the Task-8-deferred test methods and run them.

- [ ] **Step 4: Green, then full ctest, then commit**

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_network_optimization
ctest --test-dir core/build
git add -A && git commit -m "feat: port the Network routing container"
```
Expected: full ctest green at baseline + 4, with the previously commented methods now running.

---

### Task 10: The `network` toolbox group and `shortest_path()`

**Files:**
- Create: `core/include/corehydro/numerics/support/toolbox/network.hpp`
- Modify: `core/include/corehydro/numerics/support/toolbox_runner.hpp` (include + dispatch line +
  the group-count header comment: fourteen groups become fifteen)
- Create: `fixtures/toolbox/network.json`; Modify: `fixtures/README.md`
- Modify: `tools/oracle_emitter/Program.cs` (`ToolboxDispatch` `case "network"` +
  `NetworkDispatch`)
- Modify: `corehydror/R/toolbox.R` (`shortest_path()`),
  `corehydropy/src/corehydropy/toolbox.py` (the twin + `__all__`)
- Modify: `corehydror/_pkgdown.yml`, `site/_quarto.yml` (Task 11 finalizes the section text)

**Interfaces:**
- Consumes: Tasks 8-9.
- Produces, toolbox group `network`, in `corehydro::numerics::support::detail`:
  `inline ToolboxResult run_network(const std::string& method, const
  std::vector<std::vector<double>>& data, const JsonValue& options)`. Edges travel as four parallel
  data vectors `[from, to, weight, index]` (the toolbox convention is flat double vectors; `from`,
  `to` and `index` are whole numbers cast to `int`, `weight` is cast to `float` -- see Task 8's
  precision note, and say so in the header). Methods:
  - `dijkstra` -- options `destinations` (int array, required), `node_count` (optional, default
    -1) -> the result table flattened row-major with `dims = {node_count, 3}`.
  - `network_solve` -- same inputs, routed through `Network` (which caches the incoming-edge lists)
    rather than the free function.
  - `network_solve_weights` -- a fifth data vector `edge_weights`; options `destinations` are
    taken from the constructed `Network`'s own destination list, matching the C# overload.
- Produces, user-facing (identical in both languages): `shortest_path(from, to, weight,
  destinations, edge_index = NULL, node_count = NULL, edge_weights = NULL)` -> a table with one row
  per node and columns `next_node`, `edge_index`, `cost` (R: a data frame; Python: a numpy array
  with a documented column order). `edge_index` defaults to `0:(n_edges - 1)`; `edge_weights`, when
  supplied, selects the custom-weight overload. Unreachable nodes carry `cost = Inf` and
  `next_node = -1`, exactly as the C# result table does -- document that in both help pages.

- [ ] **Step 1: Write failing R and Python tests**

Transcribe DijkstraTesting's `SimpleEdgeGraphCost` graph into both languages and assert the cost
column against its literals (`0`, `3`, `4`, `6`, `7`, and `Inf` for the disconnected node).
Expected: FAIL (function not found).

- [ ] **Step 2: Implement the group, the dispatch line, and both verbs**

`toolbox/network.hpp` follows `toolbox/linalg.hpp`'s shape: a local `to_edges(data)` helper doing
the double-to-int/float narrowing with a length-agreement check, a `result_table_result(table)`
serializer setting `dims = {n, 3}`, and one arm per method. Wire
`if (group == "network") return detail::run_network(method, data, options);` into
`toolbox_runner.hpp` and update its header's group count and list. Write the two verbs with full
roxygen / docstrings.

```bash
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
```

- [ ] **Step 3: Fixture and emitter parity**

`fixtures/toolbox/network.json` (kind `toolbox`, group `network`) carrying one case per
DijkstraTesting method whose graph is small enough to inline -- at minimum `SimpleEdgeGraphCost`,
`MultipleDestSharedPath`, `DisconnectedComponent` and `TrianglePath` -- with the C# assertion
literals as expected values at `tol: 0` (they are exact `float` equalities in C#), plus one
`network_solve_weights` case. Emitter: `ToolboxDispatch` gains `case "network"` and a
`NetworkDispatch` driving the real C# `Dijkstra.Solve` / `Network.Solve`.

```bash
cmake --build core/build -j && ctest --test-dir core/build -R test_fixtures
python3 tools/verify_oracles.py
```
Expected: green; `0 failed`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: expose Dijkstra shortest paths through a network toolbox group"
```

---

### Task 11: Documentation and worked example 19

**Files:**
- Modify: R roxygen for `optim_minimize`, `optim_maximize`, `optim_constraint`, `shortest_path`;
  regenerate via `Rscript -e 'roxygen2::roxygenise("corehydror")'`; the matching Python docstrings
- Modify: `corehydror/_pkgdown.yml` + `site/_quarto.yml` -- the existing "Optimizers" section text
  ("The six ported Numerics optimizers ...") becomes thirteen and gains `optim_constraint`; a new
  parallel entry adds `shortest_path` (put it in the same section as the other graph/utility verbs
  -- read the file and pick the section whose title already fits, rather than inventing a
  one-entry section)
- Create: `site/examples/19-global-optimizers/{python.ipynb, r.qmd}`
- Modify: the site examples index (find it: `grep -rn '18-' site/_quarto.yml site/examples/index.qmd`)

**Interfaces:**
- Consumes: everything shipped in Tasks 1-10.

- [ ] **Step 1: Reference docs and the index contract**

Every new or changed export gets title, description, every parameter, `\value`, and one runnable
example inside CRAN time limits. `optim_minimize`'s help must state, per method: which of
`initial` / `seed` / `gradient` / `constraints` / `inner` it accepts, and which control keys apply
-- the per-method table from Task 5 is the source of truth, so write the help off it. Verify:

```bash
Rscript -e 'pkgdown::check_pkgdown("corehydror")'
```
Expected: no missing-topic error. Confirm the quartodoc build lists every Python export.
Elements-of-style conventions throughout.

- [ ] **Step 2: Example 19 -- global optimizers**

Copy the structure of the `site/examples/18-numerical-methods` pair. Content: the Eggholder
function optimized by particle swarm, SCE and simulated annealing with their seeded results
compared side by side against `de`; a constrained problem (Haimes 5.2) through
`optim_constraint()` + `method = "augmented_lagrange"`, showing the multiplier; ADAM with and
without an analytic gradient on the same problem, comparing evaluation counts; and one
`shortest_path()` graph. State the cross-language contract in prose, in the terms P2's example 13
established: a seeded run's PARAMETERS reproduce bit-for-bit between R and Python because the PRNG
lives in the shared C++ core, while the reported objective VALUE comes from re-evaluating the
user's own R or Python arithmetic and is not bit-guaranteed. If Task 5's digest measurement found
anything narrower than that for a specific method, say THAT instead -- the example states what was
measured, not what would be convenient.

Each page ends with the executable reproduction check (pinned literal at 1e-15 relative tolerance
for deterministic values; seeded values pinned exactly). Python notebook committed WITH outputs
(`jupyter nbconvert --to notebook --execute --inplace site/examples/19-global-optimizers/python.ipynb`);
R rendered so `site/_freeze/` updates.

- [ ] **Step 3: Build and verify the site, commit**

```bash
pixi run docs
git add -A && git commit -m "docs: reference updates and worked example 19"
```
Expected: clean build; the new page renders in both languages.

---

### Task 12: Version 0.10.0, full verification, ship

**Files:**
- Modify: `corehydror/DESCRIPTION` (0.10.0), `corehydropy/pyproject.toml` (0.10.0),
  `corehydror/NEWS.md`, `CHANGELOG.md`
- Modify: `.claude/CLAUDE.md` (Layout section: the new optimizer/dynamic headers and the fifteenth
  toolbox group; Status: append the P3 paragraph with final numbers)

- [ ] **Step 1: Bump versions and write release notes**

NEWS/CHANGELOG: the four global optimizers, the three local optimizers, constrained optimization
through augmented Lagrange with `optim_constraint()`, the optional analytic gradient on ADAM and
gradient descent, `local_method` now reaching `mlsl` as well as `multi_start`, `shortest_path()`
over the new `network` toolbox group, and whatever Task 9 concluded about `Network.GetPath`.

- [ ] **Step 2: Full verification (evidence before assertions)**

```bash
cmake --build core/build -j && ctest --test-dir core/build
Rscript -e 'cpp11::cpp_register("corehydror")' && R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
python3 tools/verify_oracles.py
```
Expected: ctest green at baseline + 4 (102/102 at the 98 baseline); testthat 0 failures above 6622;
pytest 0 failures above 1619; gate `0 failed` with reproduced count > 5751 and skips still 11.
Record the final numbers in NEWS/CHANGELOG and `.claude/CLAUDE.md`.

- [ ] **Step 3: R CMD check regression**

```bash
R CMD build corehydror && R CMD check --as-cran corehydror_0.10.0.tar.gz
```
Expected: the same three known NOTEs (CRAN-incoming non-FOSS license, long paths listing vendored
core headers, local HTML-tidy version), no new WARNING or NOTE.

- [ ] **Step 4: Commit, push, PR, CI**

```bash
git add -A && git commit -m "chore: release v0.10.0 (optimizers)"
git push -u origin port-optimizers
gh pr create --base port-math-extras \
  --title "P3: optimizers -- global, local, constrained, and shortest path (v0.10.0)" \
  --body "$(sed -n '/^## v0.10.0/,/^## /p' CHANGELOG.md)"
gh run watch $(gh run list --branch port-optimizers -L1 --json databaseId -q '.[0].databaseId') --exit-status
```
Expected: CI green on the full matrix. Retarget the PR to `main` once PRs #26 and #27 merge. Do not
merge without Cam's go-ahead.

---

## Self-review notes

- Spec coverage: the spec's P3 paragraph names "the seven new optimizers as `Optimizer` subclasses
  dispatched through the existing `optimizer_runner` (new method names on `optim_minimize`)" ->
  Tasks 1-3 port them and Tasks 5-6 dispatch them; "AugmentedLagrange with a constraint spec added
  to the options grammar" -> Tasks 4 and 7 (the constraint spec is a spec-level array paired
  positionally with a callback list, since a constraint's function cannot be serialized);
  "Dijkstra / Network / BinaryHeap as a small toolbox `network` group" -> Tasks 8-10; "Seeded
  global optimizers get cross-language digest fixtures" -> Task 5 Step 5 (four cases, one per
  stochastic new method); "the documented callback-path limit is stated in the examples, not
  papered over" -> Task 11 Step 2; example 19 and the version bump -> Tasks 11 and 12.
- Type consistency: `OptimCallbacks` and `Gradient` are introduced in Task 6 and consumed by Task 7
  (`constraints`); `OptimResult`'s three multiplier vectors are introduced in Task 7 and read by
  Task 7's fixture assertions only; `make_optimizer` is introduced in Task 7 and used for both the
  top-level and the inner dispatch; the four-parallel-vector edge transport is stated once in Task
  10 and used by its verbs; `set_objective_function` is added in Task 1 and consumed only in Task
  7; the per-method control table is defined once in Task 5 and is the source of truth for Task 6's
  and Task 7's new keys and for Task 11's help text.
- Task ordering: every port task precedes the surface task that dispatches it; Task 6 precedes Task
  7 because `OptimCallbacks` is where constraints live; Task 8 precedes Task 9 because `Network`
  forwards to `dijkstra::solve`; Task 9 precedes Task 10 because two of the group's three methods
  go through `Network`.
- Deliberate non-placeholders: port tasks cite the exact C# source as the implementation reference
  (the established port convention) and call out by name every transcription detail measured to be
  oracle-visible; code blocks appear where the code is genuinely new (the objective setter, the
  callbacks struct, the control table, the guarded-gradient arm, the local-method parser,
  `make_optimizer`'s signature).
- One open question is deliberately left open rather than guessed: whether `Network.GetPath` runs
  at all in the real C#. Task 9 Step 2 specifies the probe, both outcomes, and the write-up either
  way.
