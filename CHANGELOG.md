# Changelog

All notable changes to corehydro (the shared C++ core, the `corehydror` R package, and
the `corehydropy` Python package) are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). The three components are versioned together.

## [Unreleased]

## [0.11.0] - 2026-08-25

The data-and-testing layer that closes out the port: `Numerics.Data.Statistics.HypothesisTests`,
the two `RMC.BestFit.Models.DataFrame` facades that class was blocking, the `Correlation` matrix
overloads, and the whole `Numerics.Data.Paired Data` subsystem -- `Ordinate`, `LineSimplification`,
`OrderedPairedData`, `UncertainOrdinate`, `UncertainOrderedPairedData`, and their one consumer,
`TabularFunction`, completing the `IUnivariateFunction` trio. This closes branch
`port-data-and-tests` and its stack (P1-P4, `port-distribution-gaps` through this branch).

### Added

- **Twelve hypothesis tests**, ported from `Numerics/Data/Statistics/HypothesisTests.cs`, reachable
  as `hypothesis_test(x, y = NULL, method = ...)`: `"one_sample_t"`, `"equal_variance_t"`,
  `"unequal_variance_t"`, `"paired_t"`, `"f"`, `"f_models"` (returns a named
  `f_statistic`/`p_value` pair rather than a bare p-value), `"jarque_bera"`, `"wald_wolfowitz"`,
  `"ljung_box"`, `"mann_whitney"`, `"mann_kendall"`, and `"linear_trend"`. They dispatch through a
  new `hypothesis` toolbox group, the sixteenth. **`UnimodalityTest`, the thirteenth C# method, is
  deferred to the next phase**: it trains a `Numerics.MachineLearning.GaussianMixtureModel` at
  k = 1 and k = 2, and the Machine Learning layer has no port yet.
- **The two severed RMC.BestFit `DataFrame` facades**, un-gated now that `HypothesisTests` exists
  under them, reachable as `analysis_data_hypothesis_test()` and `analysis_data_statistics()`
  through a new `data_frame` fixture kind. Of the Hypothesis Testing region's eleven members, nine
  ship: `jarque_bera_test`, `ljung_box_test`, `equal_variance_t_test`, `unequal_variance_t_test`,
  `f_test`, `linear_trend_test`, `wald_wolfowitz_test`, `mann_whitney_test`, and
  `mann_kendall_test`, every one reading the exact series only and splitting two-sample tests on
  the data index rather than array position. The Summary Statistics region's three members all
  ship: `summary_statistics_exact_data_only()` and `summary_statistics_all_data()` (each a
  twenty-key ordered result -- record length, low-outlier count, the raw and log-space moments,
  and seven exceedance-probability quantiles) and `set_standardized_values()`. **Twelve of
  fourteen members ship**; `unimodality_test` and `summary_hypothesis_test` are deferred with
  `UnimodalityTest`, since the latter calls the former inside a try/catch that would NaN nine
  working results if it shipped alone.
- **The `Correlation` matrix overloads**, ported from `Numerics/Data/Statistics/Correlation.cs`
  (`Pearson(double[,])` and `Spearman(double[,])`), reachable through `correlation()`'s new matrix
  path: called with a matrix or data frame and no `y`, it returns the full p-by-p correlation
  matrix instead of one pairwise value. Upstream has no `KendallsTau(double[,])` overload, so
  `method = "kendall"` combined with a matrix is a rejection naming the reason, not a silent
  pairwise fallback.
- **The Paired Data subsystem**, ported from `Numerics/Data/Paired Data/` in full: `Ordinate` and
  `LineSimplification`, `OrderedPairedData` (with the six previously-severed `Search.cs` overloads
  it needed un-severed alongside it), `UncertainOrdinate`, `UncertainOrderedPairedData`, and
  `TabularFunction` (`Numerics/Functions/TabularFunction.cs`), the last of the three
  `IUnivariateFunction` implementations. Reachable through a new `paired_data` toolbox group (the
  seventeenth) as five verbs: `curve_interpolate()`, `curve_area()`, `curve_simplify()` (Douglas-
  Peucker, Visvalingam-Whyatt, or Lang), `uncertain_curve_sample()` (a curve whose y-values are
  distributions, sampled at a probability or at each distribution's mean), and
  `tabular_function()`.
- **A worked example pair, 28** (examples 20 through 27 shipped in earlier phases): the Harricana
  River annual peaks through the independence, homogeneity, and normality tests, the same record
  through `analysis_data()`, a correlation matrix over three series, and a reservoir stage-storage
  curve interpolated in linear and log space, simplified by all three algorithms, and sampled
  under uncertainty at its median and its mean.

### Fixed

- `Correlation`'s and `HypothesisTests`' four missing `Statistics`/`Tools` helpers
  (`mean_variance`, the tie-returning `ranks_in_place` overload, the vector `percentile` overload,
  and `tools::pow(double, int)`) are now ported, closing the last gap the P0-P3 phases had left in
  `numerics/data/statistics.hpp`.

### Notes

- **`curve_simplify(method = "lang")` reproduces an upstream defect rather than fixing it.**
  `OrderedPairedData.LangSimplify` never force-keeps a curve's last point the way its two sibling
  algorithms do, so it can silently drop it -- on the `sin` curve at `tolerance = 0.01,
  look_ahead = 2` (upstream's own `Test_LangSimplify` case), it returns three points and drops the
  fourth, `(6.28, 0)`. Upstream's own test cannot detect this because its comparison loop is
  bounded by the RESULT length rather than the expected length. Verified against the real,
  compiled C# library, pinned by ctest and by a fixture the dotnet oracle gate replays at exact
  tolerance. This and sixteen other new findings from this phase are recorded in
  `docs/upstream-csharp-issues.md`.

### Validation

ctest 108/108 (the fixture suite alone 6139 checks); oracle gate 6128 reproduced, 0 failed, 11
skipped (the documented GEV standard-error set, unchanged); testthat 7161/0; pytest 1795 passed.
`R CMD check --as-cran` holds at the same three NOTEs
(the CRAN-incoming non-FOSS-license note, the long-path note listing vendored core headers, and a
local HTML-tidy-version note) with no WARNING.

## [0.10.0] - 2026-08-23

The rest of the Numerics optimization layer. `optim_minimize()` grew from six methods to fourteen:
four global searches (particle swarm, shuffled complex evolution, simulated annealing,
multi-start), three local ones (ADAM, gradient descent, golden section), and the first constrained
method, augmented Lagrange, which takes one or more constraint functions and reports the Lagrange
multiplier on each. ADAM and gradient descent accept an optional analytic gradient, which on a
five-dimensional problem cut the objective calls from 103,141 to 8,596. The dynamic-programming
trio (BinaryHeap, Dijkstra, Network) is not an optimizer -- its input is a graph, not a callable --
so it joins the toolbox as a fifteenth group and reaches users as `shortest_path()`. This is the
third step of the release arc laid out in
`docs/superpowers/specs/2026-08-20-remaining-port-and-v1-release-design.md`, following the
math-extras release (0.9.0).

### Added

- **Four global optimizers**, ported from `Numerics/Mathematics/Optimization/Global/`, as new
  `method` names on `optim_minimize()` / `optim_maximize()`: `"particle_swarm"` (`ParticleSwarm`,
  the Kochenderfer-Wheeler swarm at Alam's recommended inertia weights), `"sce"`
  (`ShuffledComplexEvolution`, the SCE-UA of Duan et al.),
  `"simulated_annealing"` (`SimulatedAnnealing`, Corana adaptive step control),
  and `"multi_start"` (`MultiStart`, repeated local searches from uniform restarts). Each is
  seeded, and each carries its own settings through `control`: `population_size` for the swarm;
  `complexes`, `cce_iterations` and `tolerance_steps` for SCE; `initial_temperature`,
  `min_temperature`, `cooling_rate`, `update_cycles`, `temperature_cycles` and `tolerance_steps`
  for annealing; `local_method`, `local_absolute_tolerance`, `local_relative_tolerance` and
  `polish` for multi-start.
- **Three local optimizers**, ported from `Numerics/Mathematics/Optimization/Local/`:
  `"adam"` (`ADAM`), `"gradient_descent"` (`GradientDescent`), and `"golden_section"`
  (`GoldenSection`, one-dimensional like `"brent"`). ADAM and gradient descent take `alpha` (the
  step size) through `control`, and ADAM the two decay factors `beta1` and `beta2`.
- **An optional analytic gradient.** `optim_minimize()` / `optim_maximize()` gain a `gradient`
  argument -- an R function or Python callable returning one partial derivative per parameter --
  accepted by `"adam"` and `"gradient_descent"` and an error for every other method. Omitted, both
  methods differentiate numerically, as the C# classes do with a null gradient. It is the optimizer
  surface's second host-language callback after the objective itself, and it goes through the same
  abort-state guard: an exception raised inside the gradient survives the ported optimizer's own
  catch-all rather than being replaced by an internal C++ one.
- **Constrained optimization**, ported from `Numerics/Mathematics/Optimization/Constrained/`.
  `method = "augmented_lagrange"` runs `AugmentedLagrange` over a borrowed inner optimizer, with
  two new arguments on both verbs: `constraints`, a list of the new `optim_constraint()` (R) /
  `Constraint` (Python) objects, each pairing a function with a `value`, a `type` (`"eq"`, `"le"`,
  `"ge"`) and a `tolerance`; and `inner`, an optional spec naming the inner optimizer and its own
  bounds, seed and control (defaulting to BFGS over the top-level vectors). The result gains
  `multipliers`, the three Lagrange multiplier vectors (`equality`, `less_than`, `greater_than`),
  one entry per constraint of that type. Which inner optimizer is chosen can change which local
  solution the constrained problem converges to -- worked example 19 shows BFGS and Powell landing
  on 25.32 and 28.31 on the same Haimes problem, in the C# library exactly as in the port.
- **`local_method` on `"mlsl"`.** MLSL's choice of local solver (`"bfgs"`, `"nelder_mead"`,
  `"powell"`) was ported at Phase 6 but never exposed; it is now a `control` setting on `"mlsl"` as
  well as on the new `"multi_start"`. Both methods reject `"adam"` and `"gradient_descent"` as a
  local method, which is upstream behavior and not a gap in the port.
- **`shortest_path()`**, over a new `network` toolbox group (the fifteenth), backed by the ported
  `BinaryHeap`, `Dijkstra` and `Network` from `Numerics/Mathematics/Optimization/Dynamic/`. It
  takes a graph as three parallel vectors (`from`, `to`, `weight`, plus an optional `edge_index`)
  and one or more
  `destinations`, and returns the routing table Dijkstra produces: the next node on the cheapest
  path, the edge to take, and the cost, one row per node. Node indices are **0-based in both
  languages**, matching the C# literals; an unreachable node reports `next_node = -1`,
  `edge_index = -1` and `cost = Inf`. Edge weights are single precision in the C# library and the
  port keeps them so, widening to double only at the language boundary.
- **A worked example pair, 19, global and constrained optimization**, walking four seeded global
  searches on the Eggholder function (including the two that miss the optimum at their defaults,
  and what to change), the constrained Haimes problem with its multiplier put to work as a shadow
  price and checked against an actual bound relaxation, ADAM with and without an analytic gradient,
  and `shortest_path()` over the upstream ten-node routing graph. It ends, as every example does,
  in an executable reproduction check.

### Changed

- **`optim_maximize()` rejects `method = "augmented_lagrange"` by name.** The upstream C# class
  always drives its inner optimizer through `Minimize()` over an augmented Lagrangian built from
  the raw objective, so a maximize request flips the reported sign without flipping the search
  direction and returns the constrained MINIMUM labelled `Success` (measured; see
  `docs/upstream-csharp-issues.md`). The ported class still mirrors upstream so a fixture can pin
  it; the guard lives on the public verbs, and its message names the exact workaround -- minimizing
  `-f` under the same constraints is exactly maximizing `f`.
- **Optimizer settings are validated against a single per-method table** in both packages, replacing
  three ad-hoc special cases. A `control` setting that belongs to a different method is now an
  error naming both the setting and the method it belongs to, rather than a silent no-op.
- **The optimizer runner builds every method through one `make_optimizer` helper**, so the inner
  optimizer of a constrained run and a top-level run go through the identical construction path
  rather than two copies of the same switch.

### Fixed

- **`Dijkstra.Solve` had no real bounds check on its node indices**, so a `node_count` smaller than
  the graph indexed past the end of its own arrays. In C# that is an `IndexOutOfRangeException`; in
  the port it was undefined behavior, confirmed with AddressSanitizer, and through the Python
  package it terminated the interpreter rather than raising. The four sites C# indexes a node-sized
  array now check in place -- in place, not up front, because the C# check is lazy and an
  out-of-range index on an edge the search never relaxes must still return a table -- and
  `shortest_path()` additionally rejects `node_count < max(from, to) + 1` up front so the error
  names the argument.

### Notes

- **Three upstream defects in `Network` are documented rather than papered over**, all measured
  against the real C# library and written up in `docs/upstream-csharp-issues.md`. The class cannot
  be constructed at all (its constructor sizes both edge caches one element short and never sets
  its node count), `Solve(float[] edgeWeights)` silently ignores the weights it is given, and
  `GetPath` binary-searches an `int[]` for an `Edge` and so can only throw, return null, or return
  an empty list. The port mirrors the second and third exactly and pins them with tests; it
  diverges on the first, deliberately and visibly, because a constructor that always throws has no
  behavior to be faithful to. None of this reaches the user surface: `shortest_path()` goes through
  the free solver, which works, and `GetPath` is severed (see `upstream/CLAUDE.md`).
- **A seeded `"particle_swarm"` or `"sce"` run reproduces between R and Python but not necessarily
  against the C# library**, and the reason is measured rather than assumed. Both algorithms branch
  on comparisons between accumulated sums; clang and gcc contract `a*b + c` into a fused
  multiply-add by default and .NET never does, so one contracted expression flips an accept-or-
  reject and every later draw is spent differently (particle swarm on Booth: 3,073 iterations and
  92,220 evaluations against the packages' 3,855 and 115,680). Compiled with `-ffp-contract=off`
  the port is bit-identical to C# on every construct tried. `"simulated_annealing"` and
  `"multi_start"` reproduce the C# library exactly, down to the evaluation count, and `"adam"`,
  `"gradient_descent"`, `"augmented_lagrange"` and `shortest_path()` have no PRNG at all. The
  affected fixture cases pin only what survives both paths -- iteration counts, evaluation counts,
  status, and the parameters that land exactly on a bound. No tolerance was loosened and no oracle
  was skipped.
- **`method = "multi_start"` can report a value that its own reported parameters do not produce.**
  Upstream's polish step clamps the best point back onto the bounds in place, after its fitness was
  recorded, so a local search that wandered outside the box leaves an out-of-box value attached to
  an on-the-bound point. Measured on the Eggholder function: value `-959.8293` at
  `(512, 404.3228)`, where the objective is `-959.6312`. The real C# library returns the same
  numbers bit for bit, so this is faithful reproduction; it is written up in
  `docs/upstream-csharp-issues.md`, and worked example 19 does not showcase this method.

### Validation

ctest 102/102 (the fixture suite alone 5953 checks); oracle gate 5942 reproduced, 0 failed, 11
skipped (the documented GEV standard-error set, unchanged); testthat 6916/0; pytest 1711 passed.
`R CMD check --as-cran` holds at the same three NOTEs
(the CRAN-incoming non-FOSS-license note, the long-path note listing vendored core headers, and a
local HTML-tidy-version note) with no WARNING.

## [0.9.0] - 2026-08-22

The last major slice of Numerics that is not a distribution, model, or estimator: root finding,
integration, an ODE solver, spline and polynomial interpolation, linear algebra, and two small
special-function/general-function groups, all reachable against a user-written R or Python
function. This is the second step of the release arc laid out in
`docs/superpowers/specs/2026-08-20-remaining-port-and-v1-release-design.md`, following the
distribution-gaps release (0.8.0).

### Added

- **Root finding**, ported from `Numerics/Mathematics/Root Finding/`. `root_find()` gains a
  `method` option over its existing bracketing interface -- `"brent"` (the default), `"bisection"`,
  `"secant"`, and `"newton"` (which takes an analytic derivative `df` and a `first_guess` instead of
  a bracket) -- and a new export, `root_find_system()`, solves a vector-valued system with
  multivariate Newton-Raphson given the system function, its Jacobian, and a starting vector.
- **Integration**, ported from `Numerics/Mathematics/Integration/` (the seven integration classes)
  plus the `Integration` statics. `quadrature()` gains a `method` option covering ten deterministic
  rules: `"gauss_kronrod"` (the existing adaptive default), `"simpsons"`, `"trapezoidal"`,
  `"adaptive_simpsons"`, `"gauss_lobatto"`, `"gauss_legendre"`, `"gauss_legendre20"`,
  `"simpsons_fixed"`, `"trapezoidal_fixed"`, and `"midpoint"`. `quadrature_2d()` integrates a
  two-argument function over a rectangle. `quadrature_nd()` integrates over an arbitrary-dimension
  box with three seeded methods -- `"monte_carlo"`, `"miser"` (stratified, recursive), and
  `"vegas"` (importance-sampled, with rare-event configuration).
- **`ode_solve()`**, the ported `RungeKutta` solver (`"rk4"`, `"rk2"`, `"rkf"`, `"cash_karp"`) for
  `dy/dt = f(t, y)` from an initial value, returning an array of length `time_steps` (not
  `time_steps + 1`), matching the C# convention.
- **Cubic spline and polynomial interpolation**, ported from `Numerics/Data/Interpolation/`.
  `interpolate()` gains `method = "cubic_spline"` and `method = "polynomial"` (the latter requiring
  an `order`) beside the existing linear method; the `x_transform`/`y_transform`/`extrapolate`
  arguments remain linear-only, an enforced guard rather than a silent no-op, matching the C#
  `CubicSpline`/`Polynomial` classes, which expose neither a transform surface nor an
  `Extrapolate()` method.
- **A `linalg` toolbox group**: `qr_decomposition()` and `qr_solve()` (ported `QRDecomposition`),
  and `gauss_jordan()` (ported `GaussJordanElimination`, an in-place row-reduction solve).
- **A `special` toolbox group**: `debye()` (the Debye function) and `polynomial_eval()` (Horner's
  method, with `variant` covering the standard, reverse, and reverse-unit coefficient
  conventions). `debye()` is ported from `Numerics/Mathematics/Special Functions/Debye.cs`, and
  `polynomial_eval()` from `Numerics/Mathematics/Special Functions/Evaluate.cs`.
- **A `functions` toolbox group**: `univariate_function()`, evaluating the ported
  `Numerics.Functions.LinearFunction` and `PowerFunction` -- forward and inverse, and
  `PowerFunction`'s own `IsInverse` switch -- over an optional normally-distributed noise path
  selected by `confidence_level`. `TabularFunction`, the third `IUnivariateFunction`
  implementation, depends on the still-unported Paired Data subsystem and is severed to a later
  release rather than silently dropped; see `upstream/CLAUDE.md`.
- The `toolbox_runner` dispatch grows from eleven groups to fourteen (`linalg`, `special`,
  `functions` added); the `CallbackSet` used by the `math`-group callback surface grows two new
  members, `scalar_deriv` and `vector_weight`, alongside the existing `scalar_xy`.
- A worked example pair, **18, numerical methods**, walks the whole surface -- root finding,
  quadrature (including the three Monte Carlo families), the ODE solver, spline/polynomial
  interpolation, and the three new toolbox groups -- ending in an executable reproduction check.

### Changed

- **`root_find()`'s `tolerance` argument moved from positional slot 4 to slot 7.** The new
  `method`, `df`, and `first_guess` arguments landed ahead of it to keep the bracketing arguments
  together. A 0.8.0 caller passing `tolerance` positionally now gets a clear argument-validation
  error instead of a value landing in the wrong parameter.
- **`quadrature()` gained `method` between `upper` and `absolute_tolerance`.** Same effect: a
  0.8.0 positional call shifts by one argument and now fails loudly rather than misbehaving
  silently.

### Notes

- **Two of the three seeded Monte Carlo integrators carry a measured, honestly documented rounding
  difference at the current shipped build; no tolerance was loosened and no oracle was skipped.**
  `"monte_carlo"`'s integral reproduces bit-for-bit across all four runners (C++, R, Python, and
  the real C# library) and is pinned at zero tolerance. `"miser"` measured 1 ULP off C#, and
  `"vegas"` measured 2-3 ULP between the actual installed R and Python packages, from
  floating-point contraction differences in the variance/chi-squared accumulation. The affected
  fixture cases assert the seeded evaluation counts and solver status instead of the integral
  value, with the measurements recorded in the fixture and in worked example 18.

### Validation

ctest 98/98 (the fixture suite alone 5762 checks); oracle gate 5751 reproduced, 0 failed, 11
skipped (the documented GEV standard-error set, unchanged); testthat 6622/0; pytest 1619 passed.
`R CMD check --as-cran` holds at the same three NOTEs (the CRAN-incoming non-FOSS-license note,
the long-path note listing vendored core headers, and a local HTML-tidy-version note) with no
WARNING.

## [0.8.0] - 2026-08-21

Two gaps the ten-phase port left open, closed. `GeneralizedNormal` was the one univariate family
that had a name in the Numerics type enum and nothing behind it, so the library now carries **43**
univariate distributions rather than 42, and `fit_distributions()` fits and ranks the **15**
candidates the C# `FittingAnalysis` ranks rather than 14. The covariance-aware pivotal bootstrap,
severed at Phase 3 as that phase's final task, is ported in full and reachable from both packages
as `bootstrap_custom(run_type = "pivotal")`.

The pivotal workflow is what you want when a plain bootstrap interval is too narrow because each
replicate's own estimation uncertainty is thrown away. A regular run keeps one point estimate per
replicate; a pivotal run keeps the estimate **and** its covariance, standardizes each replicate
against the parent fit in link space, and reinflates through the parent covariance, so the
replicate ensemble carries the within-replicate uncertainty the point estimates dropped. The raw
(non-pivotal) ensemble is returned beside the pivotal one, so the two intervals can be compared in
the same run.

### Added

- **`GeneralizedNormal`**, the 43rd univariate distribution, ported from
  `Numerics/Distributions/Univariate/GeneralizedNormal.cs`. It is the three-parameter log-normal,
  parameterized by location, scale and shape, with the shape's sign setting which tail is bounded.
  Reachable by name through every existing verb -- R `distribution("GeneralizedNormal", c(100, 10,
  0))`, Python `Distribution("GeneralizedNormal", [100, 10, 0])` -- with moments, L-moments, MLE
  and seeded sampling. With it, **every family in the univariate type enum now constructs** except
  the three C# itself marks unsupported by the factory (CompetingRisks, Mixture, UserDefined).
- **The covariance-aware pivotal bootstrap**, through nine new arguments on `bootstrap_custom()`
  that carry the same names and defaults in R and Python: `run_type` (`"regular"` or `"pivotal"`),
  `fit_with_covariance` (the fitting function a pivotal run uses instead of `fit`, returning the
  parameters and their covariance), `original_covariance`, `pivotal_links`,
  `pivotal_invalid_draw_policy`, `regularize_pivotal_covariances`, `pivotal_z_limit`,
  `add_pivotal_jitter` and `pivotal_jitter_scale`. `pivotal_links` takes one **link function name**
  per parameter (`"Identity"`, `"Log"`, `"Logit"`, `"Probit"`, `"ComplementaryLogLog"`,
  `"YeoJohnson"`, `"FisherZ"`, or a null for the identity), not a callback: standardization happens
  in link space, so `"Log"` on a scale parameter keeps every reinflated draw positive. A pivotal
  result adds `pivotal_diagnostics` (the six replicate counts, named after the upstream fields) and
  the raw block `raw_estimate` / `raw_lower` / `raw_upper` / `raw_standard_error` / `raw_mean` /
  `raw_valid_count` / `raw_parameter_estimate` / `raw_parameter_lower` / `raw_parameter_upper`;
  every result now carries `run_type`. A pivotal run is percentile-only and refuses the arguments
  it cannot use (`fit`, `jackknife`, `inner_replicates`), as a regular run refuses the pivotal ones.
- **The `IStandardError` capability mixin** in the core
  (`numerics/distributions/base/i_standard_error.hpp`), with generic dispatch for
  `parameter_covariance`, `quantile_variance`, `quantile_gradient` and `quantile_jacobian` through
  the shared `dist_runner`. A family that declares the mixin is now reached by the same one-runner
  path every other distribution verb uses, instead of the bespoke GEV-only path that carried these
  four methods before.
- Two worked example pairs, each ending in an executable reproduction check: **16, the
  15-candidate fitting surface** (ranking an annual-peak record by AIC, BIC and RMSE, and what
  it means that the three criteria pick three different winners) and **17, the pivotal bootstrap**
  (a pivotal interval against its own raw block, a link on a bounded parameter, and the three
  invalid-draw policies side by side).

### Changed

- **`fit_distributions()` returns 15 rows, not 14.** GeneralizedNormal enters the candidate list at
  the position C# gives it, fifth, between GeneralizedLogistic and GeneralizedPareto, so **any code
  indexing the result positionally past the fourth row shifts by one.** Ranking by
  `which.min(aic)` / `min()` is unaffected except that a new candidate can now win.
- **L-moments form their integer products in floating point, so long samples return correct
  values.** `l_moments()`, and every distribution fit that estimates from sample L-moments,
  computed the probability-weighted-moment weights `(i - 2) * (i - 1)` and
  `(i - 3) * (i - 2) * (i - 1)` in 32-bit integers. The triple product exceeds the 32-bit range at
  `i = 1293` and the pair product at `i = 46,343`, which corrupts the L-kurtosis of any sample of
  **1293 or more points**. On an evenly spaced series whose true L-kurtosis is 0, the old code
  returned -0.185 at 1293 points and -1.45 at 1300, with no warning. The products are now formed in
  `double`. Below 1293 points they are exact integers well under 2^53 and the result is
  bit-identical to before, so nothing changes for a shorter sample. Above it corehydro returns the
  mathematically correct weight. This is a deliberate divergence from upstream: the C# Numerics
  library and RMC-BestFit still wrap, so **corehydro and RMC-BestFit disagree on the L-kurtosis of
  any sample of 1293 or more points.** C# defines signed overflow as wrapping, while C++ leaves it
  undefined, so keeping the C# expression was not an option either.

### Fixed

- **`fit_distributions()` reported the new candidate's name as `"Unknown"` in both languages.**
  The distribution-name lookup in the R and Python analysis glue is a local switch with an
  `"Unknown"` default, and it had fourteen arms. Adding the distribution and adding the fifteenth
  candidate each left it alone, and the fixture pins that candidate's AIC, BIC and converged flag
  but not its name, so every count and every metric passed while the row printed `Unknown`. Found
  by writing the worked example, which is the only place the column is read by eye. Both packages'
  tests now assert the name of the fifth candidate, not just how many there are.
- **The oracle gate silently skipped the standard-error methods for every family that implements
  them.** The emitter returned "no value" for `parameter_covariance`, `quantile_variance`,
  `quantile_gradient` and `quantile_se` unconditionally, which was right for GEV's eleven
  documented skips but meant any other implementer would have been skipped rather than reproduced.
  It now skips GEV by name and dispatches the rest for real, through the same capability cast the
  other three runners use.
- **Python and R disagreed on which pivotal option values they would refuse.** Python passed
  `pivotal_jitter_scale` through `float()` unchecked, so a `NaN` there silently emptied the
  retained ensemble under the default drop policy, and it accepted an infinite `pivotal_z_limit`
  that R refuses. Both now raise on a non-finite scale and on a non-positive or infinite limit,
  with the same wording, so the two languages refuse exactly the same inputs.
- **A Python `fit_with_covariance` returning a bare list of numbers was read as a one-parameter
  fit** with a 1x1 covariance instead of being refused, the same mistake the moment-condition
  callback already guards against. It now raises.
- **`fit_distributions()` could rank GeneralizedPareto as failed, or rank it differently between
  platforms and between runs.** `GeneralizedPareto::get_parameter_constraints` returned two values
  for a three-parameter distribution, so every caller that reads all three read one element past
  the end of the vector. Inside `fit_distributions()` that out-of-bounds value became the
  GeneralizedPareto candidate's starting parameter, its bounds, and its prior. On the fitting
  dataset used by the test fixtures the candidate came back failed, with an AIC of `NaN`, where the
  C# library converges it at an AIC of 423.31, the lowest of any candidate. What was read is
  whatever the allocator happened to leave in that memory, so the outcome was not stable across
  platforms, compilers, or runs. **Results from `fit_distributions()` involving GeneralizedPareto
  may change.** Refit if you kept earlier output. The candidate is now pinned against the real C#
  library in `fixtures/analyses/fit_distributions_smoke.json`, which asserted only the Normal
  candidate before and so did not catch this. The same read reached the GeneralizedPareto seed in
  eight other places: a model's default parameters and its trend model, the MCMC model registry,
  the copula marginal pre-fit, the mixture, competing-risks and point-process component seeds, and
  Bulletin 17C. This was a port defect, not upstream behaviour; in C# the index is in range.

### Documentation

- **Only one of the three standard-error methods can be pinned for GeneralizedNormal, and the
  reason is upstream.** C# `GeneralizedNormal.ParameterCovariance` and `QuantileVariance` both
  throw `NotImplementedException`, so there is no C# answer to reproduce. The port mirrors the
  throw rather than inventing a number, and the two methods are declared anyway so that a caller
  capability-casting to `IStandardError` gets the same "not implemented" answer instead of a
  silently wrong one. `quantile_gradient` is the one that is pinnable, and it is pinned at three
  exceedance probabilities against values read from the real C# library. The C# test for it
  compares the analytic gradient against the numerical derivative the method itself calls, so it
  is a tautology and carries no literal to scrape.
- `site/status.qmd`, `site/index.qmd` and `README.md` brought to v0.8.0: 43 univariate families,
  and the `Sampling.Bootstrap` row no longer calls the pivotal workflow an omission.
- Reference documentation for the new family and the pivotal arguments in both packages, with
  runnable examples.

### Validation

ctest 90/90 (the fixture suite alone 5579 checks); oracle gate 5568 reproduced, 0 failed, 11
skipped (the documented GEV standard-error set, unchanged); testthat 6344/0; pytest 1520 passed.
`R CMD check --as-cran` holds at three NOTEs (the CRAN-incoming non-FOSS-license note, the
long-path note listing vendored core headers, and a local HTML-tidy-version note) with no WARNING.
The pivotal cross-language fixture passes at zero tolerance in all four runners; no `oracle_skip`
and no loosened tolerance was added anywhere in this release.

## [0.7.0] - 2026-08-20

The callback layer. Five upstream classes are delegate-driven by design, and until now both
packages could reach them only through internal registries, so the model always had to be one the
port already knew how to build. A user's own R or Python function can now drive them directly:
a log-likelihood, a Gibbs proposal, an HMC/NUTS gradient, the four bootstrap delegates, and a set
of GMM moment conditions. One shared header carries the whole surface,
`numerics/support/callback_runner.hpp`, dispatching to one standalone-compiling header per group
(`callback/math.hpp`, `mcmc.hpp`, `bootstrap.hpp`, `gmm.hpp`, `rng.hpp`), driven identically by
the cpp11 glue, the pybind11 glue, the C++ fixture runner, and the dotnet oracle emitter, so a
fixture case, an oracle replay, and a user's `mcmc_posterior()` call are the same code path.

Every crossing of the host-language boundary goes through `numerics/support/callback_guard.hpp`.
An exception raised inside an R or Python callback cannot travel through the ported C# algorithm
that called it, because that algorithm has its own catch-all, so the guard latches the first host
exception, substitutes a sentinel, and rethrows once the ported call has unwound. Guards that
belong to one run share an abort state, which is what stops a latched log-likelihood from letting
the sampler re-enter the host through the proposal with an unwind already pending.

### Added

- **`mcmc_posterior()`** -- sample a posterior whose likelihood is not in the package, over
  priors you choose. This is upstream's own `MCMCSampler(priorDistributions, logLikelihood)`
  constructor. All EIGHT ported samplers are reachable, including **Gibbs**, which neither
  package could run before: Gibbs needs a model-specific conditional proposal, and there was no
  way to pass one. HMC and NUTS take an optional analytic `gradient` beside it; left unset, the
  ported bound-aware finite-difference default still applies.
- **`bootstrap_custom()`** -- a confidence interval on any statistic you can compute from a fitted
  parameter set, over all four upstream delegates (`resample`, `fit`, `statistic`, `jackknife`)
  and all five interval methods (Percentile, BiasCorrected, Normal, BootstrapT, BCa), plus
  `inner_replicates` for the studentized method and `max_retries` for the failed-replicate loop.
- **`fit_gmm_moments()`** -- fit moment conditions you write, reaching
  `GeneralizedMethodOfMoments`'s second (delegate) constructor. `fit_gmm()` reaches the first,
  whose `IGMMModel` interface has exactly one implementation, so it could fit a Bulletin 17C
  flood-frequency model and nothing else. Returns the same `corehydro_fit` / `Fit` object
  `fit_gmm()` does, with an optional analytic `jacobian` and `penalty`.
- **`root_find()`, `derivative()`, `gradient()`, `hessian()`, `quadrature()`** -- Brent root
  finding, numerical differentiation, and adaptive Gauss-Kronrod integration over a plain R or
  Python function. R's `gradient()` and `hessian()` collide with `numDeriv` and `pracma`, so the
  documentation calls them `corehydror::gradient()`.
- **A handle on the core's seeded generator**, passed to the callbacks that need randomness (the
  bootstrap resample and the Gibbs proposal). R gets `rng_uniform()` and `rng_integers()`; Python
  gets an `Rng` class that users cannot construct. Drawing from `runif()`, `sample()` or
  `numpy.random` instead would leave the seeded run unreproducible, so the handle is the only way
  a callback should get a random number. It borrows the generator for one call and invalidates on
  return, so a stored handle raises instead of reading freed memory.
- Two worked example pairs: a custom posterior (a hand-written likelihood through
  `mcmc_posterior()`, with the trace, R-hat and posterior summary, and Gibbs over an exact
  conditional proposal) and a custom bootstrap (a statistic the package does not provide, through
  `bootstrap_custom()`).
- A cross-language digest fixture (`fixtures/callback/callback_cross_language.json`, a new
  `callback_cross_language` fixture kind) nesting one seeded Gibbs posterior and one seeded
  bootstrap interval under a single case, asserted at **zero tolerance** rather than a relative
  one, so all four runners are held to bit equality on the callback path.

### Fixed

- **`Numerics/Mathematics/Integration/AdaptiveGaussKronrod` was a minimal port**, carrying only
  what `VonMises::CDF` needed. It now derives from a ported `Integrator` base with settings,
  status and function-evaluation counting, and reports its standard error, which the earlier port
  computed and dropped.
- **The R RNG handle could be handed a foreign external pointer and crash the session.**
  `borrow_from()` accepted any `EXTPTRSXP` and cast it, but every package that hands R a pointer
  produces that same SEXP type, so an `Rcpp::XPtr`, or the address slot of a registered native
  routine, was cast and dereferenced, segfaulting and aborting R. The pointer now carries a
  `corehydro_rng` tag that R code cannot forge, and the tag is required before the cast.
- **`rng_integers()` had signed integer overflow on a wide span.** `max - min` was forwarded to
  the generator unchecked, so a range above `int.MaxValue` wrapped negative and returned
  -208904155 in both languages instead of raising. The span is taken in int64 now, in the handle
  and in the ported generator, reproducing C#'s own throw.
- **The optimizer and callback guards' abort flag was private to each guard**, so the promise that
  one run can never raise a second host exception held only for a single callback. Guards that
  belong to one run now share a `CallbackAbortState`, which keeps the FIRST exception rather than
  the last, and every drive site rethrows off that shared state rather than off whichever guard
  happens to be asked.
- **A NaN return from the Gibbs proposal was accepted in both languages** and surfaced two calls
  downstream inside the user's own log-likelihood, because the proposal goes through a different
  converter from the gradient and neither language checked there. Both refuse it by name now, as
  does a NaN gradient in Python, which R already refused.
- **The `bootstrap_custom()` jackknife help text claimed `data[-index]` returns the sample
  untouched at index 0.** In R that spelling is `data[-0]`, which is the empty vector. Corrected
  in the roxygen block, in a user-visible error string, and in the tests.
- **The J statistic is no longer printed where it cannot be trusted.** At zero over-identifying
  degrees of freedom the residual covariance it is scaled by is theoretically zero, so the number
  is whatever inverting a numerically singular matrix returned; `print()` / `summary()` name the
  reason instead. The same line is now gated on the statistic being finite, so a fit whose
  covariance could not be inverted reports that rather than printing `nan`. The field itself is
  untouched on both fits.
- An R log-likelihood returning `NA`, the shape a model indexing past its priors takes where
  Python raises `IndexError`, is refused by name in both packages instead of walking a motionless
  chain. A `NULL`/`None` seed is refused by name rather than failing inside the JSON builder. A
  fractional callback count and a whole number spelled as a float are now accepted and refused
  identically in the two languages.
- `QuadratureResult` in Python could not be pickled or deep-copied: it is a `float` subclass whose
  `__new__` takes three arguments, which the default protocol cannot rebuild. `__getnewargs__`
  restores both round trips, matching `OptimResult`.
- Both wrappers wrote every option key on every call for `root_find`, `derivative` and
  `quadrature`, duplicating the C# defaults and defeating the documented rule that an absent key
  leaves the ported default in force. A key is sent only when the caller supplies it now.
- The `mcmc_sampler` fixture kind carried a third copy of the sampler-construction switch beside
  the two glues. It now calls the shared builder in
  `numerics/sampling/mcmc/support/mcmc_run.hpp`, so the registry path and the callback path build
  and report a run through the same code.
- The C++ fixture runner is compiled with `-ffp-contract=off` on non-MSVC compilers. Its fixture
  callback catalog is written as C++ lambdas standing in for the R closures, Python functions and
  C# delegates the other three runners write for the same names, and clang and gcc contract
  `a*b + c` into a fused multiply-add by default, so those lambdas were computing a different
  function from the one the fixture names. MSVC needs no flag: `/fp:precise` does not contract.

### Documentation

- **An entry in `docs/upstream-csharp-issues.md` for the over-identified GMM J statistic.**
  `fit_gmm_moments()` produced the first over-identified GMM fit either package could run, and the
  statistic does not reproduce against the real C# library: `J = 214.59` there against
  `J = -129.46` here, from parameters agreeing to 2e-11. It structurally cannot. The moment
  residual covariance `V = S - D(D'S^-1 D)^-1 D'` has rank exactly `q - p` for any `q` and `p`, so
  it is singular whether the fit is over-identified or not, and `V.inverse()` amplifies the
  optimizer's 1e-8 convergence tolerance rather than any property of the data. The port's inputs
  are correct: `g' V^+ g` through a pseudo-inverse gives 2.3466, matching the textbook
  `n g' S^-1 g` on the same fit. The remedy belongs upstream, in the inversion.
- `site/status.qmd` corrected and brought to v0.7.0. The `Sampling.MCMC` row said seven of eight
  samplers were exposed; all eight are. The multivariate `Distributions` and
  `Distributions.Copulas` rows still said "Internal" although both have been public since v0.5.0.
  No row is marked "Internal" any more.
- `site/examples/25-estimation-methods` no longer prints the J statistic of its zero-degrees-of-
  freedom Bulletin 17C fit as a result, and no longer pins it in the reproduction check.
- Reference entries for every new R export and Python name in `corehydror/_pkgdown.yml` and the
  `quartodoc.sections` of `site/_quarto.yml`.

### Validation

ctest 87/87; oracle gate 5426 reproduced, 0 failed, 11 skipped (the documented GEV standard-error
set, unchanged); testthat 6144/0; pytest 1487 passed. `R CMD check --as-cran` holds at three NOTEs
(the CRAN-incoming non-FOSS-license note, the long-path note listing vendored core headers, and a
local HTML-tidy-version note) with no WARNING. The cross-language fixture passes at zero tolerance
in all four runners, with no `oracle_skip` and no loosened tolerance anywhere in the branch.

## [0.6.0] - 2026-08-13

The numerics toolbox layer. Every general-purpose Numerics utility that is not itself a
distribution, copula, model, or estimator -- correlation, goodness of fit, descriptive and
running statistics, spectral analysis, histograms, interpolation, linear regression, Sobol
sequences, stratification, joint probability, link functions, trend evaluation, and all six
ported optimizers -- is reachable from R and Python for the first time. Two shared runners carry
the whole surface: `numerics/support/toolbox_runner.hpp` (eleven groups, each a standalone header
under `numerics/support/toolbox/`) for everything that operates on serializable data, and
`numerics/support/optimizer_runner.hpp` for the six optimizers, which take a live host-language
callback instead. The cpp11 glue, the pybind11 glue, the C++ fixture runner, and the dotnet oracle
emitter all drive the same two runners, so a fixture case, an oracle replay, and a user's
`correlation()` or `optim_minimize()` call are the same code path -- the pattern the estimation
and distribution layers already established, now covering the last major slice of the port.

### Added

- **`correlation()`** -- Pearson, Spearman, and Kendall's tau.
- **`goodness_of_fit()`, `classification_metrics()`, `gof_test()`, `gof_rmse()`, `aic()`,
  `aicc()`, `bic()`, `aic_weights()`, `rmse_weights()`** -- the C# `GoodnessOfFit` class's
  seventeen continuous metrics (RMSE, NSE, KGE, and friends) plus six binary classification
  metrics, the Kolmogorov-Smirnov/Anderson-Darling/chi-squared test statistics for a fitted
  distribution, the three information criteria, and Akaike/RMSE model weights.
- **`summary_statistics()`, `product_moments()`, `l_moments()`, `ranks()`, `percentile()`,
  `running_statistics()`, `running_covariance()`, `autocorrelation()`, `cross_correlation()`,
  `dft()`, `dft_real()`** -- descriptive and streaming (online) statistics, autocorrelation with
  PACF and confidence bands, and the discrete Fourier transform.
- **`histogram()`, `interpolate()`, `interpolate_2d()`** -- histogram binning, and one- and
  two-dimensional interpolation with optional axis transforms (log, normal-Z).
- **`linear_regression()`** -- ordinary least squares, returning a `corehydro_lm` (R) /
  `LinearRegressionResult` (Python) with `print`/`summary`/`coef`/`vcov`/`residuals`/`predict`.
- **`sobol_sequence()`, `stratify()`, `joint_probability()`** -- quasi-random Sobol sequences,
  deterministic interval stratification, and joint exceedance probability under independence,
  positive/negative dependency, or an explicit correlation.
- **`link_function()`, `link()`, `link_inverse()`, `link_derivative()`, `link_names()`,
  `trend_predict()`, `trend_parameters()`, `trend_names()`** -- twelve link types (the seven
  standard Numerics links -- identity, log, logit, probit, complementary-log-log, Fisher-z,
  Yeo-Johnson -- plus the five BestFit-specific links -- ASinH, SES, log-SES, log-ASinH, Centered)
  and evaluation of the eleven BestFit trend models. Python's `link_function()` takes `lambda_`
  rather than `lambda` (a reserved word), the one deliberate naming difference from R, matching
  the existing `corehydropy.stats` precedent.
- **`optim_minimize()`, `optim_maximize()`** -- the six ported optimizers (Differential Evolution,
  BFGS, Powell, MLSL, Nelder-Mead, Brent) over a user-written R or Python objective, the same
  optimizers the estimation layer already used internally, now public. A seeded `"de"`/`"mlsl"`
  run's PRNG lives in the shared C++ core, so its candidate-generation stream -- and therefore its
  result, in the well-separated cases this release's own fixtures and examples exercise -- is
  reproducible run to run and identical language to language.
- Two worked example pairs: model evaluation (ranking candidate distributions with the new
  goodness-of-fit surface) and a custom objective (fitting a hand-written likelihood with
  `optim_minimize()`/`optim_maximize()`, global versus local, cross-language DE reproduction).
- A cross-language digest fixture (`fixtures/toolbox/toolbox_cross_language.json`, a new
  `toolbox_cross_language` fixture kind) pinning one seeded DE run, one Sobol block, and one
  stratification together, so all four runners assert the same values from the same construct --
  the toolbox layer's counterpart to the estimation layer's `fit_cross_language.json`.

### Fixed

- **`Numerics/Data/Statistics/Autocorrelation.cs` was never ported**, through Phase 0-10, the
  upstream sync, and the estimation and distribution surface phases that followed -- nothing
  needed it until the toolbox layer's `spectra` group did. It is now
  `numerics/data/autocorrelation.hpp`, reachable via `autocorrelation()`/the `autocorrelation_ci`
  verb (the `TimeSeries` overloads remain a documented severance; see the header's own comment).
  Its sibling `Correlation.cs`'s matrix overloads (`Pearson(double[,])`/`Spearman(double[,])`,
  column-pairwise correlation matrices over an `[n, p]` table) were a real but previously
  unrecorded severance -- no caller has needed them yet -- now written up in `upstream/CLAUDE.md`,
  alongside the Autocorrelation port.
- **`linear_regression()`'s `vcov()` was unscaled.** The C++ runner's `covariance()` correctly
  returns the raw `(X'X)^-1` term (matching the C# `LinearRegression.Covariance`), but the R and
  Python wrappers passed it straight through instead of multiplying by `sigma^2` the way the fit
  itself already does for `standard_errors`, so `vcov(fit)` disagreed with
  `sqrt(diag(vcov(fit))) == standard_errors` by a factor of `sigma^2`.
- **The optimizer callback guard had a hole for `"bfgs"` and `"mlsl"`.** Every `run_optimizer` arm
  called the optimizer and checked the guard only afterward, with no `try` around the call itself.
  When the guard's sentinel value drove BFGS/MLSL's first gradient probe to throw a C++
  `std::domain_error` internally, `Optimizer::minimize()`'s own catch-all rethrew THAT exception
  before the guard's stored one was ever consulted, dropping the user's original R/Python
  exception -- precisely the half-unwound-host-stack hazard the guard exists to prevent. All six
  arms now wrap the call in a `try` that always prefers the guard's stored exception.
- `goodness_of_fit(metrics = "nse")` in Python iterated the string's individual characters instead
  of treating it as one metric name; a bare string is now normalized to a one-element list before
  validation, matching the R twin's existing behavior.
- `classification_metrics()` had an invented threshold argument neither R, Python, nor the C#
  source has; dropped, and both label vectors are documented as already-binary.
- `histogram()` rejected a single-point sample even though the C# `Histogram(data)` constructor
  accepts one (widening `upper_bound_ == lower_bound_` to `[x, x+1]`), an invented minimum that
  was never in the C# source; that check is dropped, keeping only the empty-input rejection, and
  a non-positive explicit `bins` now rejects with a named error instead of silently falling back
  to the Rice rule. A toolbox dispatch error from the C# emitter also leaked its internal message
  instead of naming the method; both languages now validate and report symmetrically.
- The joint-probability runner threw for one half of the `dependency = "correlation"` no-matrix
  case while silently returning `NaN` for the other; both now fall through to `NaN`, matching the
  C# source, and the R/Python wrappers reject the combination symmetrically instead of leaking the
  asymmetry. `stratify()` now also rejects `lower >= upper` instead of silently returning zero
  rows.
- `running_covariance()`'s resume state was not array-wrapped, so a single-variable accumulator
  failed to round-trip through a second call.
- **Two latent emitter-vs-runner divergences around `label`/dims selects, found in this branch's
  final review.** The dotnet oracle emitter's `ToolboxSelectFlat` helper (used for groups whose C#
  result is a positional flattened array -- Sobol, Stratify, Histogram bins, linear regression's
  coefficient/covariance/residual matrices) had no `names` array to look an assertion's `label` key
  up against, unlike the C++/R/Python `toolbox_select` helpers, which honor it; it now throws on
  `label` instead of silently falling through to index 0. Symmetrically, the groups whose C++
  `ToolboxResult` never sets `dims` at all (`interpolation.linear`/`bilinear`,
  `regression.residuals`/`predict`, `statistics.ranks`/`percentile`, every `gof` method) now throw
  on `select: "rows"`/`"columns"` instead of answering a fabricated or unrelated value. No fixture
  paired either combination with one of those groups, so both were latent; see
  `ToolboxSelectFlat`'s and `ToolboxSelectFlatNoDims`'s header comments in
  `tools/oracle_emitter/Program.cs` and `fixtures/README.md`'s `toolbox` section.

### Documentation

- Reference entries for every new R export and Python name, in six new `_pkgdown.yml`/
  `quartodoc.sections` groups: correlation and goodness of fit, statistics, interpolation and
  regression, sampling utilities, links and trends, and optimizers. `correlation()` itself,
  shipped in this phase's first task, had been missing from both indexes until now.

### Validation

ctest 84/84; oracle gate 5209 reproduced, 0 failed, 11 skipped (the documented GEV standard-error
set, unchanged); testthat 5634/0; pytest 1344 passed. `R CMD check --as-cran` holds at three NOTEs
(the CRAN-incoming non-FOSS-license note, the long-path note listing vendored core headers, and a
local HTML-tidy-version note) with no WARNING.

## [0.5.0] - 2026-08-12

The distribution layer is complete. Five composite univariate distributions, all seven bivariate
copulas, and five multivariate distributions are reachable from R and Python for the first time,
each through one shared grammar and one runner (`dist_spec.hpp`/`dist_runner.hpp`) called by the R
glue, the Python glue, and the C++ fixture runner, so a fixture case and a user's `dist_pdf()`
call are the same code path. The dotnet oracle emitter reads the same grammar but keeps its own
dispatch against the real C# objects, so the oracle stays independent of the code it checks.

### Added

- **`dist_truncated()`, `dist_mixture()`, `dist_competing_risks()`, `dist_empirical()`,
  `dist_kde()`** -- the five composite distributions, each returning the same `corehydro_dist` /
  `Distribution` object that every existing distribution verb already accepts (`dist_pdf`,
  `dist_cdf`, `dist_quantile`, moments, `dist_random`, and so on).
- **`copula()` / `copula_fit()`** -- all seven bivariate copulas (Clayton, AliMikhailHaq, Frank,
  Gumbel, Joe, Normal, StudentT), with verbs for density, distribution, inverse CDF, tail
  dependence, joint exceedance probability, the three log-likelihoods (full, pseudo, and
  IFM-conditional), and theta bounds. `copula_fit()`'s `margin_x`/`margin_y` accept either a
  family name, fit as part of the tau/MPL/IFM/MLE estimation, or an already-fitted distribution,
  used as given.
- **`mvdist_normal()`, `mvdist_student_t()`, `mvdist_dirichlet()`, `mvdist_multinomial()`,
  `mvdist_bivariate_empirical()`** -- the five multivariate distributions. MultivariateNormal adds
  `mvdist_marginal()`, `mvdist_conditional()`, and rectangle probability, plus `seed` and the Genz
  quasi-Monte-Carlo integrator settings on the CDF so a call at dimension three or higher is
  reproducible instead of clock-seeded.
- Two worked example pairs: copulas and joint frequency analysis, and composite distributions.

### Development notes

Four defects in the new surface, each introduced and repaired inside this release. No earlier
version exposed any of these functions, so nothing here changes behaviour a user could have
depended on. They are recorded because they are the mistakes this kind of work invites, and the
next person adding a surface should know what to look for.

- **`copula_fit(method = "mpl")`, the default, returned a meaningless theta.** The
  pseudo-likelihood objective expects plotting positions and does not rank internally; the fit
  path handed it raw data instead, every log-density landed off the unit square, the objective
  was a constant, and BrentSearch returned an arbitrary interior point (theta 0.335 where the true
  fit is 65.229). The plotting-position transform now lives in the shared C++ grammar, so both
  languages get it from the same compiled code rather than each needing its own port.
- **A copula spec naming a bare marginal family under IFM fit its theta against a default
  Normal(0, 1)** instead of the fitted marginal. The grammar now MLE-fits a parameterless marginal
  before the copula fit runs.
- **A MultivariateNormal CDF at dimension three or higher had no seed in the spec grammar**, so it
  ran the Genz integrator off a clock seed and repeated calls, including calls across R and
  Python, disagreed. `seed`, `max_evaluations`, `abs_error`, and `rel_error` are now spec keys,
  applied after construction because the constructor resets the integrator's defaults.
- `copula_pdf()`, `copula_log_pdf()`, and `copula_cdf()` are now vectorized over their second
  argument in the core, so both languages inherit it; previously the vectorization was documented
  but not implemented, and extra values were silently dropped.

### Internal

Restored fidelity to the C# source at eight sites, found only once the composite distributions
got their first real oracle:

- Central moments on Mixture, CompetingRisks, and TruncatedDistribution now call the fixed
  1000-step quadrature overload the C# calls, replacing an adaptive Gauss-Kronrod integration.
- `mode()` on Mixture, CompetingRisks, TruncatedDistribution, Empirical, and KernelDensity now
  runs the same bounded BrentSearch the C# runs (over the 0.001/0.999 inverse-CDF bracket),
  replacing a grid scan, a ternary search, or a base-mode clamp depending on the class. The
  Empirical mode was 43% off (8669.72 against the C# value of 15198.51) and is now bit-exact.
- TruncatedDistribution's central moments integrated over the truncation bounds instead of the two
  1e-8 quantiles of the truncated distribution, a third bug found while fixing the first two.
- About thirty fixture-only glue functions per language, one per composite/copula/multivariate
  case that used to need its own entry point, are deleted; every one of those paths now goes
  through the shared runner.

### Documentation

- Reference entries for every new R export and Python name (41 R exports plus two S3 methods, 15
  Python names).

### Validation

ctest 81/81; oracle gate 4767 reproduced, 0 failed, 11 skipped (the documented GEV standard-error
set, unchanged); testthat 4886/0; pytest 1007 passed. `R CMD check --as-cran` holds at three NOTEs
(the CRAN-incoming non-FOSS-license note, the long-path note listing vendored core headers, and a
local HTML-tidy-version note) with no WARNING.

## [0.4.0] - 2026-08-11

The estimation layer. A model built with `model_univariate()`, `model_bulletin17c()`, or any of
the other `model_*()` constructors can now be fit directly: maximum likelihood, maximum a
posteriori, Bayesian MCMC, and generalized method of moments are each one function call, returning
a fit object with `coef()`, `confint()`, `AIC()`, `logLik()`, and `summary()` methods rather than a
bag of raw numbers. Every fit runs through one shared C++ entry point, so a fixture case, the
dotnet oracle gate, and a user's `fit_mle()` call are now provably the same code path rather than
three that happen to agree.

### Added

- **`fit_mle()` / `fit_map()`** -- maximum likelihood and maximum a posteriori, with a
  Hessian-based covariance, optional profile-likelihood confidence intervals, and a choice of six
  optimizers (Nelder-Mead, Brent, BFGS, Powell, Differential Evolution, Multilevel Single Linkage).
- **`fit_bayesian()`** -- Bayesian MCMC fitting over DEMCz, DEMCzs, ARWMH, or NUTS, returning the
  raw chains in the `posterior`-package axis order, the thinned posterior draw matrix, the MAP and
  posterior-mean point estimates, the mean log-likelihood trace, a posterior summary with R-hat and
  effective sample size, and DIC, WAIC and LOOIC, the last two with their effective parameter
  counts and LOOIC with its standard error. `credible_level` sets the width of the reported
  credible interval, and a setting outside
  the range the sampler accepts is reported with the setting named rather than as a bare
  "configuration is not valid".
- **`fit_gmm()`** -- generalized method of moments for Bulletin 17C models, with the sandwich
  covariance and the J-statistic overidentification test. Method of moments computes no likelihood,
  so the fit reports `NA` (`None` in Python) for the log-likelihood, AIC, and BIC.
- **`fit_diagnostics()`** -- Cook's distance, leverage, and observation influence off a MAP or GMM
  fit; leverage, PSIS-LOO Pareto-k, and prior influence off a Bayesian fit.
- **`quantile_variance()`** -- the Cohn-style delta-method variance of a fitted quantile at a given
  annual exceedance probability, off a `fit_gmm()` fit.
- `confint()`, `coef()`, `vcov()`, `logLik()`, `AIC()`, `BIC()`, `print()`, and `summary()` methods
  on the new `corehydro_fit` object in both packages.
- A worked example pair on fitting a Log-Pearson Type III record by all four methods and comparing
  the resulting quantile estimates.

### Internal

None of these three reached a released user-facing function. The first two were in the unexported
fixture glue this release rerouted through the shared runner; the third was caught inside this
release, before any of it shipped. They are recorded because the new fit surface inherits the
corrected behaviour.

- A model with fewer than two parameters reports `NaN` covariance, standard errors, and
  correlation. `GetCovarianceMatrix` throws below two parameters in C#, and R does not zero-fill a
  freshly allocated numeric vector, so the old fixture glue returned whatever was already sitting
  in that memory. The fixture path still writes explicit zeros there, deliberately, because that
  is the contract its pinned oracles were recorded against.
- A failed fit names the estimator and the optimizer in its error, rather than the internal message
  "failed for a fixture case" the fixture glue raised.
- `confint()`'s default level is 0.95 in both languages, matching base R's `confint()` convention.
  The Python half briefly defaulted to 0.9 during development of this release.

### Documentation

- A reference page and worked example for each of the five new functions.

## [0.3.0] - 2026-08-11

The data and model layer. Censored observations, nonstationary trends, and custom parameter
bounds and priors are reachable from R and Python for the first time: the analyses previously
took a plain numeric vector and built their model internally, so the capabilities Bulletin 17C
exists for had no way in.

### Added

- **`analysis_data()` / `AnalysisData`** -- the ported RMC.BestFit `DataFrame`. Carries exact
  observations plus the three censored types: historical and paleoflood observations known only
  within a range, perception thresholds over years with no gauge, and observations whose
  measurement error is itself a distribution. Low outliers come from the Multiple Grubbs-Beck test
  or an explicit threshold.
- **`analysis_data_summary()`** -- Hirsch-Stedinger plotting positions, low-outlier flags, record
  length, and the arrival rate off a frame.
- **`threshold_diagnostics()`** -- the mean residual life and GPD parameter stability plots for
  choosing a peaks-over-threshold cutoff.
- **Twelve `model_*()` constructors** covering all nine RMC.BestFit model families, plus `trend()`
  for nonstationary parameters, `model_parameter()` for bounds, fixed flags, and priors, and the
  verbs `model_validate()`, `model_simulate()`, `model_log_likelihood()`, and `model_parameters()`.
- **Every analysis accepts a model** in place of its data argument. The existing convenience
  signatures are unchanged.
- `training_time_steps` on the four time-series model constructors. It is a model property whose
  data-driven default exceeds the series length for any series shorter than 30, so a model built
  outside an analysis previously had no way to set it.
- Two worked example pairs: censored flood frequency, and nonstationary frequency analysis.

### Fixed

- **Bulletin 17C over a censored record no longer throws.** The regression-on-order-statistics
  imputation reads each observation's plotting position, and those are recomputed in the C#
  application by the `INotifyPropertyChanged` cascade that this port replaced with an
  explicit-call contract. Frames built from a spec now honour that contract, so a record with low
  outliers or a perception threshold reaches the moment machinery fully computed instead of with
  every position at its default. The same gap is present in upstream C# for any headless caller;
  it is written up in `docs/upstream-csharp-issues.md` with the evidence.
- **`composite_analysis()` no longer drops censoring**, handing each child fit a clone of the
  frame rather than collapsing it to its exact series.
- The R package no longer calls `jsonlite` from exported functions while declaring it only in
  `Suggests`; spec serialization is internal, so the package has no runtime dependency.
- The core version stamp had drifted a release behind the two packages; all three now read 0.3.0.

### Documentation

- `exceedance_probabilities` must be in ascending order. An unsorted vector fails validation with
  a message that does not say why.

## [0.2.0] - 2026-07-22

Upstream sync. corehydro is now validated against **Numerics v2.1.4** and **RMC.BestFit v2.0.0**
(previously Numerics `a2c4dbf` and RMC.BestFit v2.0-beta.5). Both upstream releases include fixes
RMC made in response to corehydro's own port audit, so a number of results change: where the port
faithfully mirrored an upstream bug, it now mirrors the fix.

### Changed

Values that change for existing code:

- **Student-t density** now includes the `1/sigma` Jacobian. Any `dist_pdf` on a Student-t with
  `sigma != 1` changes by a factor of `1/sigma`; the old values did not integrate to 1. The
  extreme-tail quantile path also now applies the location and scale transform.
- **Pearson type III and log-Pearson type III L-skewness** carries the sign of the skew. Negative
  skew fits return a negative L-skewness where they previously returned a positive one.
- **Beta and generalized Beta mode** returns a point inside the support for U- and J-shaped cases.
  The old formula could return a value far outside the support.
- **Plotting positions for censored and threshold data** are a faithful port of peakFQ's
  ARRANGE2/PPLOT2/PLPOS. Frames mixing exact, interval, threshold, and uncertain data get
  different positions than before, duplicates are spread deterministically, and invalid frames
  now raise instead of returning a value.
- **Zero-inflated mixtures** renormalize component weights to `1 - zero_weight` when zero
  inflation is set, rather than leaving them unnormalized.
- **Generalized logistic and log-Pearson type III L-moments** near the degenerate parameter values
  use upstream's series expansions, replacing corehydro's own limit branches.
- **Bulletin 17C** uncertainty quantification reworked: retry and acceptance behavior, an adaptive
  Mahalanobis threshold, warm-started replicates for censored and threshold data, and a reworked
  pivotal path with a guarded Yeo-Johnson link. Diagnostics report attempted versus retained
  replicates.
- **Generalized method of moments** accepts an estimate on any non-failure optimizer termination
  with a finite best point, and falls back to Nelder-Mead after a BFGS failure.
- **Rating curve default priors** widened: coefficient bounds from -5..5 to -10..10, exponent
  lower bound from 0.5 to 0.

### Added

- `try_create_distribution`, and on the multivariate normal `try_set_parameters`,
  `try_set_covariance`, `is_density_valid`, `marginal`, and `conditional`.
- Empirical distributions accept duplicate ordinates and validate ascending order.
- Bivariate distribution pseudo-likelihood fitting works: it computes the plotting positions it
  needs instead of failing.
- Time-series models handle a Box-Cox or Yeo-Johnson lambda-fit failure with a validation message
  instead of a crash.
- `docs/upstream-sync.md`, the repeatable process for absorbing the next RMC release.

### Fixed

- Multivariate normal CDF over rank-deficient or perfectly correlated covariance matrices. The old
  code silently returned wrong collapsed-CDF values whenever the redundant dimension did not sort
  to the first pivot.
- Archimedean copulas (Clayton, Gumbel, Joe) report `parameters_valid` correctly, copula clones
  deep-copy their marginals, and non-finite parameters are rejected.
- Descending-order search, guarded log10 in bilinear interpolation, out-of-range histogram
  binning, and an underflow guard in the joint-probability calculation.
- Normal distribution CDF in the far tails, which previously cancelled to exactly 0 or 1 for
  `|z|` beyond about 6. This one was a corehydro bug, not an upstream change.
- Jeffreys scale priors no longer fail for single-parameter families, and model clones keep their
  zero-inflation state.
- Threshold-series processing is idempotent.

### Validation

ctest 78/78; oracle gate 4497 values reproduced against the real C# libraries, 0 failed, 11
skipped (the documented generalized extreme value standard-error set); testthat 4253; pytest 789.

## [0.1.0] - 2026-07-11

First tagged release. Everything below is new.

### Added

- **Shared C++17 core** with full parity to the USACE-RMC Numerics and RMC-BestFit C#
  libraries (the probability, estimation, and analysis layers): all 42 univariate
  distributions, multivariate distributions and seven bivariate copulas, eight MCMC
  samplers (RWMH, ARWMH, DEMCz, DEMCzs, HMC, NUTS, Gibbs, SNIS) with Gelman-Rubin and
  ESS diagnostics, bootstrap uncertainty, the MLE/MAP/Bayesian/GMM estimators, all
  RMC-BestFit model families (flood frequency, Bulletin 17C, mixtures, competing risks,
  point process, time series, spatial GEV, rating curves, bivariate/copula), the full
  Analyses layer, and the Diagnostics layer (leverage, PSIS-LOO influence, prior
  influence, predictive checks). See the porting status page for the exact scope.
- **R package `corehydror`** and **Python package `corehydropy`**, thin bindings over
  the same compiled core. A bit-exact Mersenne Twister port means seeded results are
  identical across R, Python, and the upstream C# libraries.
- **Public distribution API**: construct any of 38 families by name; density, CDF,
  quantile, moments, L-moments, log-likelihood, seeded random generation, and fitting
  by MLE, L-moments, or product moments.
- **Analysis functions** (19 in each package): `univariate_analysis`,
  `fit_distributions`, `bulletin17c_analysis`, mixture/competing-risk/point-process/
  composite/spatial-GEV analyses, AR/MA/ARIMA/ARIMAX time-series analyses, bivariate
  and coincident-frequency analyses, rating curves, bootstrap uncertainty, predictive
  checks, and estimation diagnostics.
- **`mcmc_sample()`**: direct MCMC over any distribution family (7 samplers) with
  chains, acceptance rates, MAP, posterior summaries, R-hat, and ESS.
- **Statistics utilities**: Multiple Grubbs-Beck low-outlier test, Box-Cox and
  Yeo-Johnson transforms, plotting positions, Latin hypercube sampling.
- **Validation**: a language-neutral oracle fixture suite consumed by C++, R, and
  Python runners, plus a dev-only dotnet gate that replays every fixture against the
  real C# libraries (4,100+ values reproduced, 0 failures).
- **Documentation site** with worked examples in both languages ported from the
  official Numerics-Python-Examples repository (11 example pairs, each ending in an
  executable reproduction check against the upstream C# outputs), a Python API
  reference (quartodoc), an R API reference (pkgdown), and a porting status page.
- **Developer tooling**: pixi environment with tasks mirroring the Makefile targets,
  GitHub Actions CI (3 OS matrix) and Pages deployment.

### Changed

- Renamed from `bestfit` (packages `bestfitr`/`bestfitpy`) to **corehydro**
  (`corehydror`/`corehydropy`), reflecting the goal of carrying code from both
  USACE-RMC and HEC libraries in one package family.

[Unreleased]: https://github.com/cameronbracken/corehydro/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/cameronbracken/corehydro/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/cameronbracken/corehydro/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/cameronbracken/corehydro/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/cameronbracken/corehydro/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/cameronbracken/corehydro/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/cameronbracken/corehydro/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cameronbracken/corehydro/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cameronbracken/corehydro/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cameronbracken/corehydro/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cameronbracken/corehydro/releases/tag/v0.1.0
