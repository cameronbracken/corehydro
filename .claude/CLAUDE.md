# CLAUDE.md — corehydro

Context for Claude Code working in the `corehydro` repo. See `PLAN.md` (same dir) for the
full approved architecture and phasing.

## What this is

`corehydro` provides **R (`corehydror`) and Python (`corehydropy`) packages** for stochastic
hydrology / flood-frequency / extreme-value analysis, built on a **single shared C++17 core**
that is a faithful port of the USACE-RMC C# libraries **Numerics** and **RMC.BestFit**.

Write the math once in C++, bind it twice (cpp11 for R, pybind11 for Python). Because both
packages run the same compiled code with a bit-exact Mersenne Twister, seeded results are
identical across R and Python. Hard requirement: **publishable to CRAN and PyPI**.

Upstream C# sources are vendored as **dev-only git submodules** at `upstream/Numerics` and
`upstream/RMC-BestFit` (shallow, tracking the official USACE-RMC `main` branches; pinned via
gitlink). They are the diff baseline for the upstream-sync workflow and are NOT referenced by
either package (CRAN/PyPI sdists and CI are unaffected; `actions/checkout` uses `submodules:
false`). See `upstream/CLAUDE.md` for the per-library architecture notes (consolidated there and
tracked by this repo). `dotnet` **is now installed**, so oracle values are curated from the C#
test files AND verified reproducible against the real Numerics library (see below). (The user's
own forks live elsewhere on disk under different names.)

## Layout & the vendoring invariant

- `core/` — **canonical** C++17 core (`include/`, `src/`, `tests/`, `CMakeLists.txt`). All
  numerical development happens here.
- `fixtures/` — **canonical** language-neutral oracle fixtures (JSON). Single source of truth
  for expected values; see `fixtures/README.md` for the schema.
- `corehydror/`, `corehydropy/` — the packages. Each vendors the core + fixtures as **subtree
  symlinks** into `core/{include,data}` and `fixtures/` (NOT committed copies): e.g.
  `corehydror/src/corehydro_core/include -> ../../../core/include`. Editing a core header is live
  through the symlink; nothing to re-sync.
- Builds dereference the symlinks into self-contained, symlink-free artifacts: `R CMD build`
  does it automatically for the R tarball; `tools/materialize_core.py` does it for Python (run in
  a throwaway checkout, e.g. `make build-py`). CI runs materialize in the R + Python jobs (also
  covers Windows, where a checkout may materialize a symlink as a text stub). See
  `docs/superpowers/specs/2026-07-08-shared-core-symlink-vendoring-design.md`.
- `tools/oracle_emitter/` (C#) + `tools/verify_oracles.py` — the dotnet oracle gate: replays every
  fixture against the real Numerics library and fails on any value that doesn't reproduce to
  tolerance. **Dev-only** (needs `dotnet` + the submodule); not wired into CI.
- An auto-scraper to harvest the C# test literals *en masse* is still planned for the bulk port;
  for now fixtures are curated by hand and confirmed by the dotnet gate.

The univariate distribution layer lives under `core/include/corehydro/numerics/distributions/`:
`base/` holds `UnivariateDistributionBase`, the type enum, the factory, and the `IEstimation` /
`ILinearMomentEstimation` / `IStandardError` capability mixins; all 43 distributions derive from
the base.
`distributions/multivariate/` holds Dirichlet, Multinomial, BivariateEmpirical,
MultivariateNormal (with the ported Genz MVNDST integrator), and MultivariateStudentT, plus the
`MultivariateDistributionBase` and factory. `distributions/copulas/` holds all seven bivariate
copulas (Clayton, AliMikhailHaq, Frank, Gumbel, Joe, Normal, StudentT), the shared
`BivariateCopula`/`ArchimedeanCopula` base classes, `BivariateCopulaEstimation` (tau/MPL/IFM/MLE
fits), the `IMaximumLikelihoodEstimation` mixin, and the copula factory.

`distributions/support/dist_spec.hpp` + `dist_runner.hpp` (both corehydro additions with no
upstream C# counterpart, siblings of `models/model_spec.hpp` and
`estimation/support/fit_runner.hpp`) are the one place a distribution, copula, or multivariate
object is built from a spec and the one place a method on it is dispatched. `dist_spec.hpp` is the
fixture `construct` schema promoted to a first-class contract -- `{"family": ..., "parameters":
[...]}` down through the composite forms (`TruncatedDistribution`/`Mixture`/`CompetingRisks`/
`Empirical`/`KernelDensity`, each nesting child specs) and the copula/multivariate forms, plus the
MultivariateNormal Genz-integrator settings (`seed`/`max_evaluations`/`abs_error`/`rel_error`,
applied after construction since the constructor resets them) and an `ifm`/`mle` marginal
pre-fit so a copula spec naming a bare family fits it before use rather than evaluating against a
default-constructed Normal(0,1). `dist_runner.hpp` dispatches `run_dist`/`run_copula`/`run_mvdist`
over that built object and returns a flat `DistResult` (`values`/`names`/`spec`); it is stateless
by construction; a seeded `random` call returns the whole draw vector in one call so a rebuild can
never split the RNG stream. Four callers drive this pair and none owns any evaluation logic: the
cpp11 glue (`corehydror/src/dist_spec.cpp`), the pybind11 glue
(`corehydropy/src/bindings/dist_spec.cpp`), the C++ fixture runner
(`core/tests/test_fixtures.cpp`), and the dotnet oracle emitter -- each serializes its native
construct to the grammar and calls the runner, so a fixture case, an oracle replay, and a user's
`dist_pdf()` (or `copula_pdf()`, `mvdist_pdf()`) call are the same code path. The five composite
univariate distributions (`dist_truncated`/`dist_mixture`/`dist_competing_risks`/`dist_empirical`/
`dist_kde`), the copula surface (`copula`/`copula_fit` plus the pdf/cdf/inverse-cdf/tail-dependence/
joint-exceedance/log-likelihood/theta-bounds verbs), and the five multivariate distributions
(`mvdist_normal`/`mvdist_student_t`/`mvdist_dirichlet`/`mvdist_multinomial`/
`mvdist_bivariate_empirical`, with MultivariateNormal's marginal/conditional/rectangle-probability)
all reach both languages through this one pair of headers rather than per-family glue.

`core/include/corehydro/numerics/support/toolbox_runner.hpp` (siblings of `dist_runner.hpp` and
`estimation/support/fit_runner.hpp`, likewise corehydro additions with no upstream C# counterpart)
is the one place a general-purpose Numerics utility method is dispatched: fifteen groups --
`correlation`, `gof`, `statistics`, `spectra`, `histogram`, `interpolation`, `regression`,
`sampling`, `probability`, `link`, `trend`, `linalg`, `special`, `functions`, `network` -- each a
standalone-compiling header under
`numerics/support/toolbox/` (plus `common.hpp` for the shared `ToolboxResult`/data-access
helpers), holding that group's `detail::run_<group>` function. Bulk data travels as native double
vectors (not JSON, unlike `dist_spec`'s construct grammar) since a goodness-of-fit call carrying
two arbitrary-length series has no business paying a JSON parse; scalars, enum names, and flags
travel in a small `options_json`. The `network` group is the outlier in that list: its subject,
`numerics/math/optimization/dynamic/` (BinaryHeap, Dijkstra, Network), is an optimization class
whose input is a graph rather than a callable, so it joins the toolbox rather than the optimizer
runner and reaches users as `shortest_path()`. `numerics/support/optimizer_runner.hpp` is the
sibling for the fourteen ported optimizers (Differential Evolution, ParticleSwarm,
ShuffledComplexEvolution, SimulatedAnnealing, MultiStart, MLSL, BFGS, Powell, ADAM,
GradientDescent, Nelder-Mead, Brent, GoldenSection, AugmentedLagrange): unlike
every other toolbox/fixture surface, an optimizer's INPUT is a live callable, not serializable
data, so it is a separate runner rather than a `run_toolbox` group, and `GuardedObjective` there
is the one place a host-language (R/Python) callback crosses into the shared core, latching the
first host-language exception a callback throws so it survives the ported `Optimizer::minimize()`
catch-all instead of being replaced by an internal C++ exception. Four callers drive each runner
and none owns any evaluation logic: the cpp11 glue (`corehydror/src/toolbox.cpp`), the pybind11
glue (`corehydropy/src/bindings/toolbox.cpp`), the C++ fixture runner
(`core/tests/test_fixtures.cpp`), and the dotnet oracle emitter -- so a fixture case, an oracle
replay, and a user's `correlation()` or `optim_minimize()` call are the same code path. The
`toolbox`/`optimizer` fixture kinds carry this surface's oracle values; the one purpose-built
`toolbox_cross_language` kind (`fixtures/toolbox/toolbox_cross_language.json`) nests an
`optimizer`-kind case and two `toolbox`-kind (group `sampling`) cases under one case name, proving
a seeded DE run's parameters and two deterministic generators reproduce identically across all
four runners in one fixture, the toolbox layer's counterpart to the estimation layer's
`fit_cross_language.json`.

`core/include/corehydro/numerics/math/optimization/` carries every ported optimizer beside
`support/optimizer.hpp` (the shared base) and `support/parameter_set.hpp`: the global set
(`differential_evolution.hpp`, `particle_swarm.hpp`, `shuffled_complex_evolution.hpp`,
`simulated_annealing.hpp`, `multi_start.hpp`, `mlsl.hpp`), the local set (`bfgs.hpp`,
`powell.hpp`, `adam.hpp`, `gradient_descent.hpp`, `golden_section.hpp`, plus the two standalone
Phase-0 classes `nelder_mead.hpp` and `brent_search.hpp`), and the constrained
`augmented_lagrange.hpp` over `constraint/` (`constraint_type.hpp`, `i_constraint.hpp`,
`constraint.hpp`). `dynamic/` holds the dynamic-programming trio (`binary_heap.hpp`,
`dijkstra.hpp`, `network.hpp`) that the `network` toolbox group dispatches. Two of these headers
deliberately reproduce upstream array aliasing that is oracle-visible and must not be "cleaned
up" -- `shuffled_complex_evolution.hpp`'s reused scratch point and `multi_start.hpp`'s
re-seated initial-values array; each says so in a numbered transcription note, as does
`network.hpp` for the three defects that make the C# `Network` class unconstructible.

`core/include/corehydro/numerics/support/callback_runner.hpp` is the third runner in that family
and the one place a ported class whose INPUT is a live host-language function is dispatched:
`run_callback(group, method, options_json, callbacks)` over one standalone-compiling header per
group under `numerics/support/callback/` -- `math.hpp` (root finding, derivatives, quadrature),
`mcmc.hpp` (a user log-likelihood plus the Gibbs proposal and the HMC/NUTS gradient),
`bootstrap.hpp` (upstream's four delegates), `gmm.hpp` (moment conditions, Jacobian, penalty) and
`rng.hpp` (the borrowed generator handle). `numerics/support/callback_guard.hpp` is the generic
`GuardedCall<TResult, TArgs...>` every crossing goes through, including the optimizer surface,
which aliases it: a host exception cannot travel through the ported algorithm's own catch-all, so
the guard latches the first one, substitutes a sentinel, and rethrows once the ported call has
unwound. Guards belonging to one run share a `CallbackAbortState`, which is what stops a latched
log-likelihood from letting the sampler re-enter the host through the proposal. The four callers
are the cpp11 glue (`corehydror/src/callback.cpp`), the pybind11 glue
(`corehydropy/src/bindings/callback.cpp`), the C++ fixture runner, and the dotnet oracle emitter.
The `callback` fixture kind carries this surface's oracle values; the purpose-built
`callback_cross_language` kind (`fixtures/callback/callback_cross_language.json`) nests a seeded
Gibbs posterior and a seeded bootstrap interval under one case name at ZERO tolerance, the
callback layer's counterpart to the two files above.

`core/include/corehydro/numerics/sampling/mcmc/` holds the MCMC subsystem: `mcmc_sampler.hpp` (the
shared base -- seeding cascade, chain initialization incl. the MAP/DE/Hessian path, the serial
`sample()` driver), all 8 concrete samplers (`rwmh.hpp`, `arwmh.hpp`, `demcz.hpp`, `demczs.hpp`,
`hmc.hpp`, `nuts.hpp`, `gibbs.hpp`, `snis.hpp`), `model_registry.hpp` (the corehydro-addition model
registry the fixtures build against), and the diagnostics/results support headers (Gelman-Rubin
R-hat, ESS, `MCMCResults`/`MCMCDiagnostics`). `core/include/corehydro/numerics/sampling/bootstrap/`
holds the regular (non-pivotal) `Bootstrap<TData>` port -- `Run`/`RunDoubleBootstrap`/
`RunWithStudentizedBootstrap`, the five `GetConfidenceIntervals` methods, and its own model
registry (the covariance-aware pivotal workflow is a documented omission, tracked as a separate
severable follow-up -- see the file header). `core/include/corehydro/numerics/math/optimization/`
holds `ParameterSet`, `Optimizer` (the shared base every optimizer -- currently
DifferentialEvolution -- builds on), and
`DifferentialEvolution` itself, the global optimizer the MCMC MAP-initialization path depends on.

`core/include/corehydro/models/` holds the RMC.BestFit Models port (Phase 5). `models/data_frame/`
is the input-data container layer: `data_types/` (the `Data` base plus ExactData, IntervalData,
UncertainData, ThresholdData), `data_collections/` (the generic `DataSeries` plus the four typed
series), `data_frame.hpp` (`DataFrame` -- FullTimeSeries threshold expansion,
ProcessThresholdSeries, MGBT and explicit-threshold low outliers) with `data_frame_plotting.hpp`
(the Hirsch-Stedinger plotting positions, including a faithful port of .NET's ArraySortHelper
introsort because `List<T>.Sort` tie order is oracle-visible), and `threshold_diagnostics.hpp`
(mean residual life + GPD parameter stability). `models/trend_functions/` holds the ten trend
models plus `general_linear_function.hpp` on the shared `support/` base
(ITrendModel/TrendModelBase and the type enum). `models/univariate_distribution/` holds the four
models: `univariate_distribution_model.hpp` (with its nonstationary companion
`univariate_distribution_model_trends.hpp` and `base/univariate_distribution_model_base.hpp`),
`mixture_model.hpp`, `competing_risks_model.hpp`, and `point_process_model.hpp`.
`models/support/` carries the Phase 4 model-support types (ModelParameter, DataComponent,
PriorComponent, ModelBase, QuantilePrior and its interfaces) plus `subscript_formatter.hpp` and
`validation_result.hpp`; `models/json_lite.hpp` + `models/model_spec.hpp` are the shared fixture
spec builder all three runners and the oracle emitter drive. The MultipleGrubbsBeckTest
low-outlier test lives at `numerics/data/multiple_grubbs_beck_test.hpp`.

Phase 6 added the Bulletin17C GMM track. `core/include/corehydro/numerics/functions/` holds the
Numerics link-function layer: `i_link_function.hpp` (the `ILinkFunction` interface),
`link_controller.hpp` (the null-means-identity controller), `link_function_factory.hpp` +
`link_function_type.hpp` (the enum factory), and the seven standard links -- `identity_link.hpp`,
`log_link.hpp`, `logit_link.hpp`, `probit_link.hpp`, `complementary_log_log_link.hpp`,
`fisher_z_link.hpp`, and `yeo_johnson_link.hpp` (whose transform lives at
`numerics/data/yeo_johnson.hpp`). The optimizer additions sit under
`core/include/corehydro/numerics/math/optimization/` as real `Optimizer` subclasses beside
`differential_evolution.hpp`/`nelder_mead.hpp`/`brent_search.hpp`: `bfgs.hpp`, `powell.hpp`, and
`mlsl.hpp`, with `support/local_method.hpp` (the LocalMethod enum). These un-gate the three Phase
4 MLE/MAP throws. `core/include/corehydro/models/link_functions/` holds the six BestFit links --
`asinh_link.hpp`, `ses_link.hpp`, `log_ses_link.hpp`, `log_asinh_link.hpp`, `centered_link.hpp`,
`yeo_johnson_link.hpp` -- plus `best_fit_link_function_factory.hpp`. The two penalties live at
`models/support/parameter_penalty.hpp` and `models/support/quantile_penalty.hpp`. The GMM
estimator is `core/include/corehydro/estimation/generalized_method_of_moments.hpp` with its delegate
aliases in `estimation/gmm_delegates.hpp` and the `IGMMModel` interface at
`models/support/i_gmm_model.hpp`. Bulletin17C is
`models/univariate_distribution/bulletin17c_distribution.hpp` plus its moment heart
`models/univariate_distribution/bulletin17c_moment_machinery.hpp`.

Phase 7a added the four remaining ModelBase model families under `core/include/corehydro/models/`.
`models/time_series/` holds `auto_regressive.hpp`, `moving_average.hpp`, `arima.hpp`, and
`arimax.hpp`, plus `transform_type.hpp` (the None/Logarithmic/BoxCox/YeoJohnson `Transform` enum).
All four models are `ModelBase + ISimulatable`; the AR/MA warm-up and the conditional-vs-all-t
likelihood divergence are preserved rather than reconciled, and the ARIMAX covariate forecast-tail
extension (CovariateExtensionMethod BlockBootstrap/KNN) is a documented severance.
`models/spatial_extremes/` holds `spatial_gev.hpp` (the hierarchical Renard GEV) over
`spatial_correlation/` (`correlation_function_type.hpp`, `i_correlation_model.hpp`,
`basic_exponential.hpp`, `powered_exponential.hpp`, `spherical.hpp`) and `copula_models/`
(`cached_multivariate_normal.hpp`, `gaussian_copula.hpp`, `spatial_regression_errors.hpp`);
SpatialGEV preserves a non-canonical spatial-error log-density decomposition (the spatial-error
densities are counted in DataLogLikelihood and also emitted as SpatialError prior components, with
WAIC/LOO excluding them). `models/rating_curve/` holds `rating_curve.hpp` (the BaRatin
matrix-of-controls addition-mode stage-discharge model -- 1-3 segments, log10-space Normal residual
likelihood, optional Jeffreys 1/sigma). `models/bivariate_distribution/` holds
`bivariate_distribution.hpp` (two IUnivariateModel marginals plus a ported Numerics BivariateCopula
via CreateCopula; CopulaEstimationMethod InferenceFromMargins default / PseudoLikelihood).

The Phase 7a prerequisites: `numerics/data/box_cox.hpp` is the ported Numerics BoxCox transform
(Transform/InverseTransform/FitLambda via BrentSearch), mirroring `numerics/data/yeo_johnson.hpp`.
`numerics/data/time_series/time_series.hpp` is the **thin** TimeSeries adapter over
`std::vector<double>` + TimeInterval + StartDate (Count/indexer/Add/Clone/ValuesToArray/
ValuesToList/MeanValue/StandardDeviation/MinValue/Difference), with `support/time_interval.hpp` and
`support/time_block_window.hpp` beside it; the heavy 2,334-line Numerics container
(interpolation / file-I/O / hypothesis tests) is a documented severance. `statistics::maximum` was
added additively to `numerics/data/statistics.hpp`, and `univariate_distribution_model.hpp` carries
the one authorized structural change (the P1 IUnivariateModel accessor resolution -- the covariant
const-pointer `distribution()`, the re-exposed `data_frame()` overrides, `is_nonstationary`/
`validate`).

Phase 8 added the user-facing Analyses layer under `core/include/corehydro/analyses/`, mirroring the
C# `RMC.BestFit.Analyses` namespace across three subdirs. `analyses/support/` holds the shared base
and interfaces: `analysis_base.hpp`, `i_analysis.hpp`, `i_bayesian_analysis.hpp`,
`i_univariate_analysis.hpp`, `i_probability_ordinates.hpp`, plus `bootstrap_diagnostics.hpp` and
`cohn_confidence_interval_result.hpp` (the B17C UQ result DTOs). `analyses/univariate/` holds
`univariate_analysis.hpp` (the Bayesian frequency analysis) and `bulletin17c_analysis.hpp` (the LP3
GMM flood-frequency analysis with its three UQ paths -- MultivariateNormal default, parametric
Bootstrap, and the deterministic Cohn-style delta-method CI). `analyses/distribution_fitting/` holds
`fitting_analysis.hpp` (the 14-candidate MLE ranking). The two Numerics output types are
`numerics/data/probability_ordinates.hpp` (the `: List<double>` ordinate grid + 25 default
exceedance probabilities) and
`numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp` (the
mode/mean/lower/upper curve container). The DataFrame bootstrap surface
(JackKnife/Resample/BootstrapDataFrame/ShiftDistribution) is added additively to
`models/data_frame/data_frame.hpp` -- the Phase-8 un-deferral of the surface Phase 5/6 had severed,
consumed only by Bulletin17CAnalysis. `models/distribution_fitting/fitted_distribution.hpp` is the
FittedDistribution DTO (introduced in A6). The user-facing R/Python analysis API lives at
`corehydror/R/analysis.R` + `corehydror/src/analysis.cpp` (cpp11 `ch_analysis_*`) and
`corehydropy/src/corehydropy/analysis.py` + `corehydropy/src/bindings/analysis.cpp`, with the `analysis`
fixture kind wired into all three runners -- exposing `univariate_analysis` / `fit_distributions` /
`bulletin17c_analysis`.

Phase 9a added the per-family analysis orchestrators and the essential Diagnostics layer.
`core/include/corehydro/analyses/univariate/` gains `mixture_analysis.hpp`,
`point_process_analysis.hpp`, and `competing_risk_analysis.hpp` (Bayesian clones of
`univariate_analysis.hpp`, wrapping MixtureModel/PointProcessModel/CompetingRisksModel).
`core/include/corehydro/analyses/time_series/` holds `ar_analysis.hpp`, `ma_analysis.hpp`,
`arima_analysis.hpp`, and `arimax_analysis.hpp` (deriving `AnalysisBase` only, with the extra
`ForecastingTimeSteps` field, building UncertaintyAnalysisResults from the model Predict ensemble).
`core/include/corehydro/diagnostics/` holds the three Diagnostics classes:
`leverage_diagnostics.hpp` (Cook's-distance + variance-influence decomposition at the MAP point via
a numerical Hessian, plus the public `compute_numerical_hessian_public`/`compute_gen_var_public`
statics), `influence_diagnostics.hpp` (the PSIS-LOO Pareto-k wrapper over the already-computed
`elpd_loo`), and `prior_influence_diagnostics.hpp` (prior-to-data influence off the seeded MCMC
posterior). Wiring these un-stubbed the 6 previously-throwing estimator diagnostic methods (the
`[[noreturn]] throw "deferred"` bodies replaced in place): `maximum_a_posteriori.hpp`
`compute_leverage_diagnostics`; `bayesian_analysis.hpp` `compute_influence_diagnostics` /
`compute_prior_influence_diagnostics` / `compute_leverage_diagnostics`; and the GMM quartet in
`generalized_method_of_moments.hpp` (`get_observation_influence` / `get_cooks_distance` /
`get_influence_diagnostics` x2 / `get_leverage_diagnostics`) -- the GMM un-stubs omit the two
Model-is-Bulletin17CDistribution penalty branches (unreachable for any non-B17C IGMMModel, exactly
as C# skips them). The user-facing R/Python surface widens the Phase-8 binding pattern
(`corehydror/src/analysis.cpp` + `R/analysis.R`, `corehydropy/src/bindings/analysis.cpp` +
`analysis.py`): the seven per-family analyses dispatch through the same one-run-function-per-analysis
glue, and an `estimation_diagnostics` accessor exposes the leverage/influence/prior-influence DTOs
off the fitted estimator. The ctest suites are
`core/tests/test_univariate_family_analyses.cpp`, `test_time_series_analyses.cpp`,
`test_leverage_diagnostics.cpp`, and `test_influence_diagnostics.cpp`.

Phase 10 completed the port -- FULL PARITY. The five remaining analysis orchestrators landed under
`core/include/corehydro/analyses/`: `spatial_extremes/` holds `spatial_gev_analysis.hpp` with its two
result DTOs `spatial_gev_site_results.hpp` and `spatial_gev_cross_validation_results.hpp`;
`bivariate/` holds `bivariate_analysis.hpp` (the joint-marginals + copula frequency analysis) and
`coincident_frequency_analysis.hpp` (the conditional-frequency law over a bivariate copula);
`rating_curve/rating_curve_analysis.hpp` is the BaRatin stage-discharge analysis; and
`univariate/composite_analysis.hpp` (the weighted multi-family aggregate) is fed by
`support/weighted_univariate_analysis.hpp`. Each is a faithful AnalysisBase clone of the Phase-8
`univariate_analysis.hpp` template. The Numerics BootstrapAnalysis frequentist engine is
`numerics/distributions/uncertainty_analysis/bootstrap_analysis.hpp` (five CI methods --
Percentile/BiasCorrected/Normal cube-root/Bootstrap-t/BCa), over the new
`numerics/distributions/base/i_bootstrappable.hpp` mixin whose `bootstrap()` override is added to
Normal (a class-layout change -- preclean R rebuild after). The two formerly-throwing Bulletin17C
uncertainty dispatch arms in `analyses/univariate/bulletin17c_analysis.hpp` (the ~496-500 throwing
cases) are replaced by LinkedMultivariateNormal (its ~13 link-builder helpers +
InfluenceStatistics; constructs MultivariateNormal not MVT, center-shift commented out as in C#) and
the pivot / BiasCorrected bootstrap (reusing the A8 parametric-bootstrap fallback). The four
predictive-check classes landed under `core/include/corehydro/diagnostics/` beside the Phase-9a
leverage/influence/prior-influence diagnostics: `posterior_predictive_check.hpp`,
`prior_predictive_check.hpp`, `predictive_check_results.hpp`, and `predictive_summary.hpp`. The
Phase-1 follow-ups closed out: distribution `ParameterNames` on
`numerics/distributions/base/univariate_distribution_base.hpp` + each concrete distribution (with
`models/support/model_parameter.hpp` carrying the ModelParameter naming), and the mutable
UserDefined MCMC seeding hook (`seed_population`/`seed_chain` on
`numerics/sampling/mcmc/base/mcmc_sampler.hpp` + `estimation/bayesian_analysis.hpp`, wired into the
MixtureAnalysis EM-seed path). All of this surface is bound in R and Python via the shared
`analysis_runner.hpp` driven identically by the three harnesses, plus `corehydror/src/analysis.cpp` +
`R/analysis.R` and `corehydropy/src/bindings/analysis.cpp` + `analysis.py`. The ctest suites are
`core/tests/test_rating_curve_analysis.cpp`, `test_bivariate_analysis.cpp`,
`test_coincident_frequency_analysis.cpp`, `test_spatial_gev_analysis.cpp`,
`test_composite_analysis.cpp`, `test_bootstrap_analysis.cpp`, `test_predictive_checks.cpp`, and
`test_parameter_names.cpp`.

## Build & test commands

```bash
# C++ core
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build
# R  (regenerate registration only after editing corehydror/src/*.cpp)
Rscript -e 'cpp11::cpp_register("corehydror")'; R CMD INSTALL corehydror
Rscript -e 'testthat::test_local("corehydror")'
# Python (use the pixi env: `pixi run test-py`, or directly:)
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
# pytest reads the fixtures MATERIALIZED into site-packages by pip, not the repo symlink, so a
# fixture edit means nothing to a pytest count until the pip install above is re-run.
# vendoring: the core + fixtures are subtree symlinks; builds dereference them (R CMD build for R,
# tools/materialize_core.py for Python). No sync/drift guard needed. To get a symlink-free tree:
python3 tools/materialize_core.py    # (CI/release only; rewrites the working tree)
# oracle reproduction gate (dev-only; needs dotnet + the upstream submodule)
python3 tools/verify_oracles.py
# documentation site (Quarto + quartodoc + pkgdown; see "Documentation site" below)
make docs        # builds site/_site; serve with `make docs-serve` (NOT quarto preview)
```

Toolchain present: clang++, cmake, R 4.6.1 (`/opt/homebrew/bin`), python3.14 via the pixi env (`.pixi/`),
roxygen2/jsonlite/testthat/cpp11 installed, and **dotnet 10** (for the oracle gate; not in CI).
After any core change that alters a class layout, rebuild R clean (`R CMD INSTALL --preclean
corehydror`) — stale `corehydror/src/*.o` from a prior ABI can otherwise return garbage / abort R.

## Validation model (DRY)

Oracle values live ONLY in `fixtures/*.json`. Three thin generic runners load the same JSON and
apply every assertion: C++ `core/tests/test_fixtures.cpp` (nlohmann/json, vendored test-only under
`core/tests/third_party/`), R `corehydror/tests/testthat/test-fixtures.R` (jsonlite), Python
`corehydropy/tests/test_fixtures.py` (stdlib). The runners are **polymorphic**: non-GEV targets are
built through the factory and dispatched on `UnivariateDistributionBase` (+ capability casts) via
the `ch_dist_*` (R) / `_core.dist_*` (Py) glue; GEV keeps a bespoke path for its standard-error
methods. **Adding a distribution = new fixture file + a couple of dispatch entries per runner** —
no new per-distribution glue. Don't hardcode oracle values in test files. The dotnet gate
(`verify_oracles.py`) is the fourth, dev-only check that the fixtures still match the C# source.

## Conventions & gotchas

- **Structural mirroring:** C++ mirrors the C# file/class/method layout so upstream diffs map
  almost line-for-line. Each ported file carries a `// ported from: <path> @ <sha>` header.
- **Portability (learned from CI):** never use `M_PI` (absent under strict `-std=c++17` on Linux
  and on MSVC) — use `corehydro::numerics::kPi`. Don't name a namespace alias `gamma` (clashes with
  glibc's libm `gamma()`) or `stat` (clashes with the MSVC/POSIX CRT `stat` symbol). Pass
  `-Wall/-Wextra` only to non-MSVC compilers in CMake. MSVC raises C3493 on the implicit use of a
  `const` function-local inside a lambda with no default capture (clang/gcc treat it as a constant
  expression); capturing it explicitly then trips clang's -Wunused-lambda-capture. Put the constant
  at file scope as `constexpr`.
- **Self-contained core:** no external C++ deps (port Numerics' own linear algebra / RNG). Keeps
  the CRAN dependency surface empty and preserves oracle fidelity. Don't add Eigen to the core.
- **CRAN:** `corehydror` uses `License: file LICENSE` (R can't standardize the `0BSD` token).
  Makevars: `CXX_STD = CXX17`, no `-O3/-march/-Werror`. cpp11 internal functions (`ch_gev_*`,
  `ch_dist_*`) are unexported — tests reach them via `asNamespace("corehydror")`. After editing any
  `corehydror/src/*.cpp`, re-run `cpp11::cpp_register("corehydror")`.
- **Mutation:** the global "never mutate" rule is relaxed for these binding/model objects (they
  mirror the C# stateful API), matching the upstream design.

## Documentation site

One GitHub Pages site (deployed by `.github/workflows/docs.yml` via actions/deploy-pages;
Pages source must be set to "GitHub Actions" in repo settings). Three build halves:

- `site/` -- the root **Quarto website** (landing page + examples + the quartodoc-generated
  Python reference under `site/reference/`, gitignored). Dual theme flatly/darkly with
  earth-tone accents in `site/styles/*.scss` (the palette hexes there are the single source
  of truth; `corehydror/_pkgdown.yml` `bslib.primary` mirrors them).
- `corehydror/_pkgdown.yml` -- the R reference, built by pkgdown into `corehydror/docs/`
  (gitignored) and copied to `/r/` of the assembled site.
- Examples live at `site/examples/<nn>-<slug>/{python.ipynb, r.qmd}` -- Python examples are
  **Jupyter notebooks committed WITH outputs** (Quarto renders stored outputs, never
  re-executes them; re-run with `jupyter nbconvert --to notebook --execute --inplace` after
  editing), R examples are Quarto with `freeze: auto` and **`site/_freeze/` committed**
  (render locally and commit the updated freeze after editing an executed chunk).

Contracts: every new package export must be added to BOTH `corehydror/_pkgdown.yml` (pkgdown
errors on missing reference-index entries) and the `quartodoc.sections` in
`site/_quarto.yml`. `site/requirements.txt` pins the docs Python deps for CI (griffe<2 until
quartodoc supports griffe 2.x; `pixi.toml` carries the same pin). Local build: `pixi run docs`
(or `make docs` with the tools on PATH); inspect with `pixi run docs-serve` (`quarto preview`
cannot serve the pkgdown `/r/` half). Reproduction-check literals in the R examples compare
with 1e-15 relative tolerance (R's decimal parser can land one ulp off a written literal);
the bit-exactness guarantees themselves are enforced by the fixture suite.

Local dev environment: `pixi.toml` provides the portable toolchain (python + docs deps,
cmake, make, quarto, pandoc) with tasks that call the Makefile targets 1:1 (`pixi run
test-core|test-r|test-py|build-r|build-py|materialize|oracles|docs|docs-serve`). R and
dotnet are deliberately NOT pixi-managed: inside `pixi run` they fall through to the system
installs (homebrew R with its existing dev library; dotnet 10), while pkgdown/knitr pick up
pixi's pandoc from PATH (no RSTUDIO_PANDOC needed). `pixi.lock` is committed; `.pixi/` is
ignored.

## Git & CI

- Commits are GPG-signed automatically (key `F4C82FB462F850C1`, "Cam Bracken (GitHub)"),
  verified on GitHub. Identity: `Cam Bracken <cameron.bracken@pm.me>`. Push only when asked.
- `.github/workflows/ci.yml`: `sync-check` gate → `core` (3 OS) + `r-cmd-check` (3 OS) +
  `python` (3 OS × {3.10, 3.12}). Use `gh run watch <id> --exit-status` to follow a run.
- Never commit: OS junk, IDE settings, secrets, the dotnet build output, other AI-tool files
  (`.gitignore` covers these). **Exception (deliberate):** this repo *does* track three curated
  context files — `.claude/CLAUDE.md`, `.claude/PLAN.md`, and `upstream/CLAUDE.md` — because the
  C++/R/Python-from-C# port is a complex process whose plan and porting guidance should travel with
  the repo. The `.gitignore` un-ignores exactly those three; everything else in `.claude/` (settings,
  scratch plans) stays ignored.

## Status

Phase 0, Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6, Phase 7a, Phase 8, Phase 9a, and
Phase 10 are **complete** -- **FULL PARITY with the USACE-RMC Numerics / RMC.BestFit C# libraries is
reached**; Phases 1-4 are merged (latest: PR #6) with CI green on the full matrix. Phase 1 delivered the full
Numerics math/RNG foundation plus all 42 univariate distributions; CI is green on 3 platforms for
that merge. Phase 2 delivered the multivariate distributions and copula layer -- Dirichlet, Multinomial,
BivariateEmpirical, MultivariateNormal (Genz MVNDST), MultivariateStudentT; all seven bivariate
copulas with shared estimation (tau/MPL/IFM/MLE) and an `IMaximumLikelihoodEstimation` mixin; and
CompetingRisks' correlated dependency modes un-deferred from Phase 1. Phase 3 delivered
Sampling/MCMC -- all 8 samplers (RWMH, ARWMH, DEMCz, DEMCzs, HMC, NUTS, Gibbs, SNIS) on the shared
`MCMCSampler` base plus diagnostics/results (Gelman-Rubin R-hat, ESS) and the DifferentialEvolution
optimizer stack the MAP-initialization path needs -- and the regular (non-pivotal) Bootstrap
workflow (Percentile/BiasCorrected/BCa/Normal/BootstrapT); the covariance-aware pivotal bootstrap
was scoped as the phase's severable final task and is tracked separately rather than landing on
this branch. Phase 4 delivered BestFit's `Estimation` layer -- MaximumLikelihood,
MaximumAPosteriori, and BayesianAnalysis (DEMCz/DEMCzs/ARWMH/NUTS, with DIC/WAIC/LOOIC
diagnostics) on top of the Models slice (ModelParameter/DataComponent/PriorComponent, IModel/
ModelBase, UnivariateDistributionModel including the Jeffreys 1/scale prior) and the Estimation
support layer (MatrixRegularization, Stratify/StratificationOptions, NumericalDiff,
OptimizationMethod); GMM + IGMMModel + BFGS + the Bulletin17C coupling, the alternate optimizers
(Powell/MLSL/LocalMethod), and the Diagnostics/LeverageDiagnostics layer are severed to follow-ups.
Phase 5 delivered BestFit's flood-frequency Models core, the first slice of the Models phase --
MultipleGrubbsBeckTest; the DataFrame layer (censored data types, four series collections,
FullTimeSeries/ProcessThresholdSeries/MGBT low outliers, Hirsch-Stedinger plotting positions via a
faithful .NET ArraySortHelper introsort port, ThresholdDiagnostics); the trend/link functions
including GeneralLinearFunction; and the four models (nonstationary + censored
UnivariateDistributionModel extended in place, MixtureModel with EM + zero inflation,
CompetingRisksModel, non-seasonal PointProcessModel), with the `model_estimation` fixture kind
extended across all three runners. M14 re-pinned every Phase 5 fixture against real C# oracles and
fixed a real port divergence (the C# `Mixture.SetParameters(ref)` weight-normalization write-back
into the optimizer's arrays; the documented residual deviation is that BayesianAnalysis hands the
const-ref MCMC samplers a mutable copy). Severed from Phase 5: the DataFrame
hypothesis-test/summary-statistics facades, the seasonal PointProcess path +
GeneratePOTTimeSeries + CreateBlockSeries (need the unported TimeSeries container), the DataFrame
bootstrap/resampling surface (Phase 6), ExactData.DateTime, FittedDistribution, USGSRawText, and
all XML/INPC. Everything ported through Phase 5 is fixture-validated in C++/R/Python and
reproduced against the real Numerics/RMC.BestFit libraries by the dotnet oracle gate (3837
reproduced, 0 failed, 11 documented GEV std-err skips); seeded MCMC chains, bootstrap replicate
streams, a DEMCzs posterior chain digest, and the Phase 5 model simulation digests are all proven
bit-identical across R and Python via `short_exact`-style digest fixtures. Phase 6 delivered
BestFit's Bulletin17C GMM track, the second slice of the Models phase -- the Numerics
link-function layer (ILinkFunction/LinkController/LinkFunctionFactory + the seven standard links
identity/log/logit/probit/complementary-log-log/Fisher-z/Yeo-Johnson, the last over a ported
YeoJohnson transform); the six BestFit links (asinh, SES, log-SES, log-asinh, centered,
Yeo-Johnson) on BestFitLinkFunctionFactory; ParameterPenalty/QuantilePenalty; the distribution
moment machinery (ConditionalMoments/ParametersFromMoments/QuantileGradientForMoments) added
additively to the Phase 1 distributions; the BFGS/Powell/MLSL optimizers (with LocalMethod)
un-gating the three Phase 4 MLE/MAP throws; and GeneralizedMethodOfMoments, IGMMModel, and
Bulletin17CDistribution with its moment heart. Everything ported through Phase 6 is
fixture-validated in C++/R/Python and reproduced against the real Numerics/RMC.BestFit libraries
by the dotnet oracle gate (3871 reproduced, 0 failed, 11 documented GEV std-err skips); seeded GMM
covariance/standard errors reproduce to ~1e-12 and the MLSL seeded stream is bit-identical across
R and Python. Severed to follow-ups (documented in headers): the GMM Influence/Leverage
Diagnostics region (RMC.BestFit.Diagnostics unported, ships as throwing stubs), the DataFrame
JackKnife/Resample/BootstrapDataFrame/ShiftDistribution surface (Bulletin17CAnalysis-only, Phase
7), and the Numerics Functions/ non-link classes. B17C GMM is always just-identified, so its
J-statistic p-value is structurally NaN and no over-identified oracle is reachable (see
`docs/upstream-csharp-issues.md`). Pending: CI run and PR for the Phase 6 branch. See `PLAN.md`.

Phase 7a delivered the four remaining ModelBase model families, fit by the already-ported
MLE/MAP/Bayesian estimators -- TimeSeries (AutoRegressive/MovingAverage/ARIMA/ARIMAX, preserving
the AR/MA warm-up and the conditional-vs-all-t likelihood divergence), SpatialExtremes (the
SpatialGEV hierarchical Renard model over the three correlation models plus
CachedMultivariateNormal, GaussianCopula, and SpatialRegressionErrors, keeping the non-canonical
spatial-error log-density decomposition), RatingCurve (BaRatin addition-mode stage-discharge,
log10-space Normal residual likelihood, optional Jeffreys 1/sigma), and BivariateDistribution (two
IUnivariateModel marginals plus a ported Numerics copula via CreateCopula, IFM default /
PseudoLikelihood). These sit atop the P1/P2 prerequisites: the one authorized structural change
(the IUnivariateModel accessor resolution on `univariate_distribution_model.hpp`),
`statistics::maximum` added additively to `numerics/data/statistics.hpp`, the ported Numerics
BoxCox transform (`numerics/data/box_cox.hpp`), and the thin
`numerics/data/time_series/time_series.hpp` adapter. Everything is fixture-validated in C++/R/Python
and reproduced against the real Numerics/RMC.BestFit libraries by the dotnet oracle gate (3930
reproduced, 0 failed, 11 documented GEV std-err skips; ctest 49/49, test_fixtures 3941 checks;
testthat 3539/0; pytest 563); seeded GenerateRandomValues/Series draws reproduce the C# Mersenne
Twister stream bit-for-bit and are bit-identical across R and Python via the two `*_sim` digest
fixtures. Documented severances carried in the ported headers: the ARIMAX covariate forecast-tail
extension (CovariateExtensionMethod BlockBootstrap/KNN) and the heavy 2,334-line Numerics
TimeSeries container.

Phase 8 delivered the user-facing BestFit `Analyses` layer, the last slice of the port. The three
exported analyses -- `univariate_analysis` (Bayesian MCMC frequency curve + credible band +
goodness-of-fit), `fit_distributions` (the 14-candidate MLE ranking surface), and
`bulletin17c_analysis` (the LP3 flood-frequency GMM fit) -- are bound in both packages over the
shared C++ UnivariateAnalysis/FittingAnalysis/Bulletin17CAnalysis, with a new `analysis` fixture
kind across all three runners. The Numerics output types landed: ProbabilityOrdinates (ordinate
grid + 25 default exceedance probabilities) and UncertaintyAnalysisResults. The DataFrame bootstrap
surface (JackKnife/Resample/BootstrapDataFrame/ShiftDistribution) was un-deferred from Phase 5/6 and
added additively to `data_frame.hpp`, consumed only by Bulletin17CAnalysis, which ships three UQ
paths (MultivariateNormal default + parametric Bootstrap + the deterministic Cohn-style delta-method
CI). The dotnet emitter now subset-compiles the minimal RMC.BestFit Analyses closure (via a local
CS0104-patched Bulletin17CAnalysis.cs) and drives the real C# analyses to tighten the smoke fixtures
to exact oracles. Everything is fixture-validated in C++/R/Python and reproduced against the real
Numerics/RMC.BestFit libraries by the dotnet oracle gate (3950 reproduced, 0 failed, 11 documented
GEV std-err skips; ctest 56/56; testthat 3585/0; pytest 574). Deferred to a remaining Phase 9: the
per-family analysis orchestrators (Composite/Mixture/PointProcess/CompetingRisk/the four
TimeSeries/SpatialGEV/Bivariate/RatingCurve/CoincidentFrequency, plus Weighted/Batch), the
LinkedMultivariateNormal path (+ its ~13 link-builder helpers + InfluenceStatistics) and the
BiasCorrected/pivot Bootstrap, the Numerics `BootstrapAnalysis` frequentist engine (unused by this
scope), the `Diagnostics/` layer (LeverageDiagnostics/Influence, still throwing stubs), and GMM
report generation. Carried-forward BUG (A11): seeded DEMCz/DEMCzs runs with `thinning_interval > 1`
are NOT oracle-guaranteed C#-vs-C++ until bisected -- single-step / thin=1 is bit-identical, so every
shipped Bayesian fixture (all thin=1) is unaffected. Pending: CI run and PR for the phase8-analyses
branch. See `PLAN.md`.

Phase 9a delivered the per-family analysis orchestrators and the essential Diagnostics layer, the
low-risk parity tail of the Analyses phase. The univariate-family analyses (mixture_analysis,
point_process_analysis, competing_risk_analysis under `analyses/univariate/`) and the four
TimeSeries analyses (ar_analysis, ma_analysis, arima_analysis, arimax_analysis under
`analyses/time_series/`) are mechanical Bayesian clones of the Phase-8 UnivariateAnalysis
(validate -> BayesianAnalysis.estimate() -> UncertaintyAnalysisResults). The Diagnostics layer
(`diagnostics/`) added LeverageDiagnostics (Cook's-distance + variance-influence decomposition at
the MAP point via a numerical Hessian), InfluenceDiagnostics (the PSIS-LOO Pareto-k wrapper), and
PriorInfluenceDiagnostics (prior-to-data influence off the seeded posterior); wiring them un-stubbed
the 6 previously-throwing estimator methods across MAP (compute_leverage_diagnostics),
BayesianAnalysis (compute_influence_/compute_prior_influence_/compute_leverage_diagnostics), and the
GMM quartet (get_observation_influence/get_cooks_distance/get_influence_diagnostics
x2/get_leverage_diagnostics). All seven analyses plus a diagnostics accessor are user-callable in
both packages, and the dotnet emitter now compiles the REAL Diagnostics classes in place (replacing
the deleted DiagnosticsStubs.cs) and drives all seven analyses. Everything is fixture-validated in
C++/R/Python and reproduced against the real Numerics/RMC.BestFit libraries by the dotnet oracle
gate (verify_oracles 4003 reproduced, 0 failed, 14 skipped; ctest 60/60; testthat 3687/0; pytest
590). Fidelity notes (docs/upstream-csharp-issues.md): the AR/MA/Mixture seeded-DEMCzs analysis
curves diverge C#-vs-C++ by inherent chaotic short-chain sensitivity (the deterministic
DataLogLikelihood matches C# to <= 3 ulp, Mixture bit-identical, across 238 param vectors; a 100-iter
chain on a flat AR/MA intercept ridge or the symmetric bimodal Mixture surface flips an accept/reject
or DE basin), so those three fixtures assert only structural invariants (curve_length; AR
mode_curve[0]) with NO oracle_skip and NO tolerance loosening -- matching the Phase-3 HMC/NUTS
precedent; CompetingRisk/PointProcess tightened to exact (~1e-10); ARIMA/ARIMAX structural. Three
PriorInfluenceDiagnostics assertions carry oracle_skip (the ported empty ModelParameter names
collapse the two Normal parameter priors into one component, a deterministic dedup, not a stream
divergence). Deferred to Phase 10: CompositeAnalysis + the Numerics BootstrapAnalysis engine +
WeightedUnivariateAnalysis; SpatialGEVAnalysis (+ its 2 DTOs); BivariateAnalysis +
CoincidentFrequencyAnalysis; RatingCurveAnalysis; the B17C-deferred uncertainty methods (LinkedMVN +
link-builders + pivot/BiasCorrected bootstrap + InfluenceStatistics); and the two predictive checks
(PosteriorPredictiveCheck/PriorPredictiveCheck). Permanent skips: GMM report generation
(presentation-only text) and BatchAnalysisRunner + options (GUI batch scheduling). See `PLAN.md`.

Phase 10 delivered the final parity tail and closes the port -- **FULL PARITY**. The five remaining
user-facing analysis orchestrators landed: RatingCurveAnalysis (`analyses/rating_curve/`),
BivariateAnalysis + CoincidentFrequencyAnalysis (`analyses/bivariate/`), SpatialGEVAnalysis plus its
two DTOs SpatialGEVSiteResults / SpatialGEVCrossValidationResults (`analyses/spatial_extremes/`), and
CompositeAnalysis (`analyses/univariate/`) + WeightedUnivariateAnalysis (`analyses/support/`) -- each
a faithful AnalysisBase clone of the Phase-8 template. The Numerics BootstrapAnalysis frequentist
engine (`numerics/distributions/uncertainty_analysis/bootstrap_analysis.hpp`) shipped with all five
CI methods (Percentile / BiasCorrected / Normal cube-root / Bootstrap-t / BCa) over the new
IBootstrappable mixin (`numerics/distributions/base/i_bootstrappable.hpp`, wired onto Normal). The two
formerly-throwing Bulletin17C uncertainty dispatch arms in `bulletin17c_analysis.hpp` were un-gated:
LinkedMultivariateNormal (its ~13 link-builder helpers + InfluenceStatistics, constructing
MultivariateNormal not MVT with the center-shift commented out exactly as C#) and the pivot /
BiasCorrected bootstrap. The four predictive-check classes (PosteriorPredictiveCheck /
PriorPredictiveCheck / PredictiveCheckResults / PredictiveSummary) landed under `diagnostics/`,
completing that layer beside the Phase-9a leverage/influence/prior-influence diagnostics. The Phase-1
follow-ups closed out: distribution ParameterNames on the base + every concrete distribution (+
ModelParameter naming) and the mutable UserDefined MCMC seeding hook (seed_population/seed_chain on
`mcmc_sampler.hpp` + `bayesian_analysis.hpp`, wired into the MixtureAnalysis EM-seed path). The
user-facing R/Python API binds every one of these. With this, every distribution, the
multivariate/copula layer, MCMC + bootstrap, all estimators (MLE/MAP/Bayesian/GMM/BootstrapAnalysis +
the B17C-deferred LinkedMVN + pivot uncertainty), all model families, the complete Analyses layer
(the five remaining orchestrators), and the complete Diagnostics layer (leverage/influence/
prior-influence + the four predictive checks) are ported and validated. Everything is
fixture-validated in C++/R/Python and reproduced against the real Numerics/RMC.BestFit libraries by
the dotnet oracle gate (verify_oracles 4069 reproduced, 0 failed, 11 skipped; ctest 69/69; testthat
3770/0; pytest 606). The ONLY remaining permanent skips are presentation-only with no
numeric/statistical surface: Bulletin17CAnalysis GMM report generation (~607 lines of StringBuilder
text) and BatchAnalysisRunner + BatchAnalysisResult/Options (the WPF batch scheduler) -- R/Python
users supply their own. Honest fidelity note (the Phase-9a chaotic-sensitivity precedent, documented
in `docs/upstream-csharp-issues.md`): five seeded-DEMCzs analysis curves (Bivariate / Coincident-curve
/ Composite / RatingCurve / SpatialGEV) reproduce their posterior MAP C#-vs-C++ only to ~1e-6 by
short-chain amplification of sub-1e-8 model-density ULP drift (the deterministic copula-MLE /
log-likelihood / Normal-MLE paths reproduce to 1e-8/1e-9), so those fixtures assert only the
deterministic structural invariants that reproduce bit-identically across all four runners -- NO
oracle_skip mask, NO loosened tolerance. Pending: FULL PARITY is reached; only the CI run and PR ship
step remains, driven as a separate workflow run (per the standing "WORKFLOW RESUME is UNSAFE"
instruction). See `PLAN.md`.

The docs-and-examples effort (branch `distribution-api`, July 2026) followed Phase 10. It added the
public distribution API in both packages (R `distribution()`/`dist_*` verbs over a `corehydro_dist`
classed list; Python `Distribution` class), the stats utilities (`mgbt_test`, `box_cox*`,
`yeo_johnson*`, `plotting_positions`, `latin_hypercube`), and public `mcmc_sample()` (7 samplers over
the uniform-constraints registry model; no custom priors). New oracle surface: `random_value`
(seeded-draw) cases in six univariate fixtures and the `data_utility` fixture kind
(`fixtures/data/statistics_utilities.json`), both wired through all four runners and pinned by the
dotnet gate (4109 reproduced / 0 failed). GEV parity fix: the class now declares
IEstimation/ILinearMomentEstimation (matching C#), so the generic `dist_fit`/`dist_lmoments` path
handles GEV. The documentation site (see "Documentation site" above) ships all 16 upstream
Numerics-Python-Examples items as 11 ported/recast example pairs (Python notebooks + R Quarto twins,
every page ending in an executable reproduction check) plus a coverage page for the 5 out-of-scope
notebooks. Notable findings recorded in the examples: C# `LnNormal` is parameterized by REAL-space
mean/sd; the 04 RWMH acceptance streams reproduce the real C# run bit-for-bit while the DE/MAP
optimizer carries a ~2-ulp fitness drift (posterior tables match at displayed precision); Box-Cox/
Yeo-Johnson fitted lambdas agree across R/Python only to ~1e-8 (Brent argmin ulp drift).

The upstream sync (branch `upstream-sync-2026-07`, July 2026) re-established 1:1 parity against
**Numerics v2.1.4 (`2a0357a`)** and **RMC.BestFit v2.0.0 (`c2e6192`)**, up from `a2c4dbf` /
`fc28c0c`. That is the version pair the packages are now validated against, and the version bump
to **0.2.0** records it. The repeatable release-absorption process is `docs/upstream-sync.md`;
read it before starting the next sync. Final numbers: **ctest 78/78; oracle gate 4497 reproduced /
0 failed / 11 skipped; testthat 4253/0; pytest 789** (pre-sync baseline 4109/0/11 at the old pins;
the post-bump census was 4099/10/11, meaning exactly 10 pinned values moved under the new C#). The
11 skips are unchanged and are the documented GEV standard-error set: `parameter_covariance` x6,
`quantile_gradient` x3, `quantile_variance` x1, `quantile_se` x1.

The defining property of this release pair: both incorporate fixes RMC made in response to
corehydro's own port audit, so the ground truth flipped in two directions. Where the C++ mirrored
an old C# bug faithfully, the C++ adopted the fix and the oracle re-pinned; where the C++ carried
a documented intentional divergence, upstream adopted our behavior and the divergence retired.
`docs/upstream-csharp-issues.md` now carries a per-entry **Status:** bullet plus a reconciliation
summary: 29 entries fixed upstream and ported (1 more partly), and six intentional C++ divergences
retired (GeneralizedLogistic κ→0, LogPearsonTypeIII large-α, MixtureModel.Clone zero-inflation,
the MVN COVSRT bounds guard, the Jeffreys single-parameter guard, and the BivariateDistribution
PseudoLikelihood estimate path). Everything upstream changed is either ported or covered by an
explicit severance note; the severances are enumerated in `upstream/CLAUDE.md`, the new ones being
`AnalysisProgress.cs` (GUI progress plumbing, which is what makes the fifteen analysis
orchestrators' diffs look large), `Series.cs` (the observable-collection container behind the
severed heavy TimeSeries), `TimeSeriesDownload.cs` (network gauge retrieval), and the new
`RMC.BestFit.App` / `.UI` / `.Api` projects. `BatchAnalysisRunner` and the B17C GMM report text
remain the only permanent presentation-only skips. Carried port-side notes: seeded DEMCz/DEMCzs
runs with `thinning_interval > 1` are still not oracle-guaranteed (every shipped fixture uses
thin=1), and the B17C interval-censored bootstrap carries a measured, reproducible GMM
stopping-rule divergence documented in the issues log.

The estimation layer (branch `surface-estimation-layer`, August 2026) put a model built with any
`model_*()` constructor within one function call of a fit. The four already-ported C++ estimators
-- `MaximumLikelihood`, `MaximumAPosteriori`, `BayesianAnalysis`, `GeneralizedMethodOfMoments` --
now run behind a single shared entry point, `core/include/corehydro/estimation/support/
fit_runner.hpp`: the cpp11 glue, the pybind11 glue, the C++ fixture runner, and the dotnet oracle
emitter all serialize to the same JSON construct and call it, so a fixture case, the oracle gate,
and a user's `fit_mle()` call are the same code path. The four verbs are `fit_mle()`, `fit_map()`,
`fit_bayesian()`, and `fit_gmm()`, each returning a `corehydro_fit` / `Fit` object with
`coef`/`parameters`, `confint()`, `AIC()`, `logLik()`, `print()`/`summary()`, plus
`fit_diagnostics()` (leverage, PSIS-LOO influence, and prior influence off a MAP, Bayesian, or GMM
fit) and `quantile_variance()` (the delta-method variance of a fitted quantile off a GMM fit).
Three fixes landed alongside the new surface: the R glue no longer returns uninitialized memory in
place of zeros for a sub-two-parameter covariance, a failed fit's error names the estimator and
optimizer instead of the internal fixture-glue message that used to leak through, and `confint()`'s
default level now agrees at 0.95 between R and Python. Everything is fixture-validated in
C++/R/Python and reproduced against the real Numerics/RMC.BestFit libraries by the dotnet oracle
gate; the version bump to **0.4.0** records it. Final numbers: **ctest 80/80; oracle gate 4623
reproduced, 0 failed, 11 skipped; testthat 4624/0; pytest 928**; `R CMD check --as-cran` holds at
three NOTEs (the CRAN-incoming non-FOSS-license note, the long-path note listing vendored core
headers, and a local HTML-tidy-version note) with no WARNING.

The distribution layer (branch `surface-distribution-layer`, August 2026) made three families
reachable from both languages for the first time: the five composite univariate distributions
(`dist_truncated`/`dist_mixture`/`dist_competing_risks`/`dist_empirical`/`dist_kde`), all seven
bivariate copulas over a `copula()`/`copula_fit()` surface, and the five multivariate
distributions, with MultivariateNormal's marginal, conditional, and rectangle probability. All
three go through one shared JSON spec grammar and one runner,
`core/include/corehydro/numerics/distributions/support/dist_spec.hpp` and `dist_runner.hpp` (see
"Layout & the vendoring invariant" above), that the R glue, the Python glue, the C++ fixture
runner, and the dotnet oracle emitter all drive, replacing about thirty fixture-only glue
functions per language with the same one-runner pattern the estimation layer established for
fitting. Three bugs surfaced and were fixed because these paths got their first real oracle:
`copula_fit(method = "mpl")`, the default, returned a meaningless theta (0.335 where the true fit
is 65.229) because the pseudo-likelihood objective needs plotting positions and was handed raw
data instead; the grammar had no key for the MultivariateNormal Genz integrator seed, so a CDF
above dimension two was not reproducible and R and Python could not agree; and the IFM copula fit
path evaluated a bare marginal family name against a default-constructed Normal(0,1) instead of
fitting it first. Fixing the port fidelity at eight further C# sites turned up a third class of
bug: central moments on Mixture/CompetingRisks/TruncatedDistribution now use the fixed-1000-step
quadrature C# calls rather than adaptive Gauss-Kronrod, and `mode()` on five classes now runs the
same bounded BrentSearch C# runs; the Empirical mode had been 43% wrong (8669.72 against the C#
value of 15198.51), invisible until the verb got its first oracle. The version bump to **0.5.0**
records it. Final numbers: **ctest 81/81; oracle gate 4767 reproduced, 0 failed, 11 skipped;
testthat 4886/0; pytest 1007**; `R CMD check --as-cran` holds at the same three NOTEs with no
WARNING. The end-to-end cross-language check (a mixture pdf and seeded draw, a Clayton copula MPL
fit and joint exceedance probability, and a trivariate normal conditional mean) reproduces byte
for byte between R and Python.

The numerics toolbox layer (branch `surface-numerics-toolbox`, August 2026) made the last major
slice of Numerics -- everything that is not a distribution, copula, model, or estimator --
reachable from R and Python: correlation, goodness of fit (17 continuous metrics, 6 classification
metrics, 3 distribution test statistics, 3 information criteria, 2 model-weight verbs), descriptive
and streaming statistics, spectral analysis (autocorrelation with PACF and confidence bands,
cross-correlation, DFT), histograms, interpolation, linear regression, Sobol sequences,
stratification, joint probability, the link-function layer, trend evaluation, and all six ported
optimizers over a user-written R or Python objective. Two shared runners carry it (see "Layout &
the vendoring invariant" above): `toolbox_runner.hpp` (eleven groups, each a standalone header
under `numerics/support/toolbox/`) for serializable-data verbs, and `optimizer_runner.hpp` for the
six optimizers, which take a live host-language callback instead -- the first place an R/Python
callback crosses into the shared core, with `GuardedObjective` protecting a host-language exception
raised inside it from being swallowed by the ported `Optimizer::minimize()`'s own catch-all. Four
bugs surfaced and were fixed because this surface got its first real oracle or its first public
callers: `Numerics/Data/Statistics/Autocorrelation.cs` had been unported since Phase 0 (nothing
needed it until the `spectra` group did) and is now ported, with its sibling `Correlation.cs`'s
matrix overloads recorded as a documented (previously unrecorded) severance alongside it;
`linear_regression()`'s `vcov()` was unscaled, returning the raw `(X'X)^-1` term instead of that
term times `sigma^2`, disagreeing with its own `standard_errors` by a factor of `sigma^2`; and the
optimizer callback guard had a hole for `"bfgs"`/`"mlsl"` that let an internal C++ exception from
the guard's own sentinel value replace the user's original R/Python exception before the guard was
consulted, now closed by wrapping all six optimizer arms in a `try` that always prefers the guard's
stored exception. A purpose-built `toolbox_cross_language` fixture kind
(`fixtures/toolbox/toolbox_cross_language.json`) nests a seeded DE run and two deterministic
generators (Sobol, stratify) under one case, the toolbox layer's counterpart to the estimation
layer's `fit_cross_language.json`; it also surfaced an honest, non-guaranteed corner of the
cross-language promise, written up in the two new worked examples (12, model evaluation; 13, a
custom objective) rather than papered over: a seeded `"de"`/`"mlsl"` run's PARAMETERS reproduce
bit-exact across languages (the PRNG lives entirely in the shared C++ core), but the reported
objective VALUE does not, because it comes from re-evaluating the user's OWN R/Python likelihood
code, and R's and Python's floating-point libraries do not guarantee bit-identical rounding for the
same formula -- BFGS, being local and gradient-based, is sensitive enough to those same sub-ulp
differences that even its parameters drift slightly (~1e-10) language to language. `correlation()`,
shipped in this phase's first task but never added to either documentation index, is swept in along
with every other new export across six new `_pkgdown.yml`/`quartodoc.sections` groups. The version
bump to **0.6.0** records it. Final numbers: **ctest 84/84; oracle gate 5209 reproduced, 0 failed,
11 skipped; testthat 5634/0; pytest 1344**; `R CMD check --as-cran` holds at the same three NOTEs
with no WARNING.

The callback layer (branch `surface-callback-layer`, August 2026) opened the host-language boundary.
Five upstream classes are delegate-driven by design, and both packages could previously reach them
only through internal registries, so the model always had to be one the port already knew how to
build. Now a user's own R or Python function drives them: `mcmc_posterior()` takes a log-likelihood
and priors and reaches ALL EIGHT samplers (Gibbs was unreachable in both packages before this,
because it needs a conditional proposal callback; HMC/NUTS take an optional analytic gradient),
`bootstrap_custom()` takes upstream's four delegates (resample/fit/statistic/jackknife) over the
five interval methods, `fit_gmm_moments()` takes moment conditions and reaches
`GeneralizedMethodOfMoments`'s SECOND constructor (the first, `IGMMModel`, has exactly one
implementation, so `fit_gmm()` could fit Bulletin 17C and nothing else), and `root_find()` /
`derivative()` / `gradient()` / `hessian()` / `quadrature()` run over a plain function. A callback
that needs randomness gets a HANDLE on the generator the run is already using (R `rng_uniform()` /
`rng_integers()`, Python's unconstructable `Rng`), borrowed for one call and invalidated on return.
The shared runner and guard are described under "Layout & the vendoring invariant" above. THE
HONEST CROSS-LANGUAGE LIMIT, which the two new worked examples (14, a custom posterior; 15, a custom
bootstrap) state in prose rather than as a footnote: on the registry path a seeded run is
bit-identical between R and Python because every operation happens in the shared core, but on the
callback path the log-density (or statistic, or moment condition) is the user's OWN R/Python
arithmetic, and the two languages do not guarantee identical rounding for the same formula. MCMC
amplifies that brutally, since one flipped accept-or-reject changes every state after it, so chains
diverge outright rather than drift. `fixtures/callback/callback_cross_language.json` is the proof
and the boundary: arithmetic-only callbacks are IEEE-deterministic, and its seeded Gibbs posterior
and seeded bootstrap interval are asserted at ZERO tolerance in all four runners. Measured while
building it, and worth knowing before writing another such fixture: R and Python agree bit for bit
on EVERY callback case in that directory, including the DEMCz two-parameter regression posterior,
but they agree with the real C# library only where the arithmetic producing the number lives in the
CALLBACK rather than in the compiled core, because clang and gcc contract `a*b + c` into a fused
multiply-add by default and .NET does not (1.5e-14 on that posterior mean). That is also why
`core/CMakeLists.txt` now passes `-ffp-contract=off` to `test_fixtures` on non-MSVC compilers: its
fixture callback catalog stands in for the R/Python/C# ones and must compute the same arithmetic.
The phase's most significant finding is a new entry in `docs/upstream-csharp-issues.md`: the
over-identified GMM J statistic does not reproduce C#-vs-C++ and STRUCTURALLY CANNOT, because
`V = S - D(D'S^-1 D)^-1 D'` has rank exactly `q - p` for any `q` and `p` and upstream inverts it, so
`V.inverse()` amplifies the optimizer's 1e-8 convergence tolerance (C# 214.59 against the core's
-129.46 on parameters agreeing to 2e-11; `g' V^+ g = 2.3466` through a pseudo-inverse matches the
textbook `n g' S^-1 g = 2.3466`, proving the port's S, D and g are all correct). `j_stat` is
therefore left unasserted in the fixture with NO `oracle_skip` and NO loosened tolerance, and both
packages' `print()`/`summary()` refuse to display it at zero degrees of freedom or when it is not
finite. The version bump to **0.7.0** records it. Final numbers: **ctest 87/87; oracle gate 5426
reproduced, 0 failed, 11 skipped; testthat 6144/0; pytest 1487**; `R CMD check --as-cran` holds at
the same three NOTEs with no WARNING.

The distribution-gaps phase (P1, branch `port-distribution-gaps`, August 2026) closed the two gaps
the ten-phase port left open, and is the first step of the release arc laid out in
`docs/superpowers/specs/2026-08-20-remaining-port-and-v1-release-design.md`. **GeneralizedNormal**
(`numerics/distributions/generalized_normal.hpp`, the three-parameter log-normal) was the one
univariate family that had a name in the type enum and nothing behind it, so the layer now carries
**43** distributions rather than 42, every family in the enum constructs except the three C# itself
marks unsupported by the factory (CompetingRisks/Mixture/UserDefined), and `fit_distributions()`
ranks the **15** candidates the C# `FittingAnalysis` ranks rather than 14 (GeneralizedNormal enters
the candidate list fifth, so the fixture, the emitter's `DistributionList` filter, both packages'
count literals, and the re-indexed assertions all moved in one commit). The distribution is the
first to declare the new **`IStandardError`** mixin
(`numerics/distributions/base/i_standard_error.hpp`), whose dispatched methods reach both languages
through the shared `dist_runner` generic dispatch, a generic path added beside -- not replacing --
the bespoke GEV slice, which still carries GEV's own standard errors; of the trio only
`quantile_gradient` is oracle-pinnable for this family (C# throws `NotImplementedException` for the
other two), and `quantile_jacobian` is ctest-covered by an analytic identity rather than
runner-dispatched. The **covariance-aware pivotal bootstrap**, severed at Phase 3 as that phase's
severable final task, is ported in full: three support headers under
`numerics/sampling/bootstrap/support/` (`pivotal_bootstrap_context.hpp`,
`pivotal_bootstrap_diagnostics.hpp`, `pivotal_bootstrap_invalid_draw_policy.hpp`), the omitted
run-type branches in `bootstrap.hpp`, and a `test_pivotal_bootstrap.cpp` transcribing all 18 C#
test methods 1:1 (154 checks, including the 2500-replicate statistical-coverage test at the C#
tolerances). It reaches users as `bootstrap_custom(run_type = "pivotal")` over nine new arguments
carrying the same names and defaults in R and Python, with `pivotal_diagnostics` and the raw
(non-pivotal) block returned beside the pivotal one. Three findings worth carrying forward. First,
**only `quantile_gradient` of the standard-error trio is pinnable for this family**: C#
`GeneralizedNormal.ParameterCovariance` and `QuantileVariance` both throw
`NotImplementedException`, so the port mirrors the throw and there is no oracle to reproduce; the
methods are declared anyway so a capability cast gets the same "not implemented" answer rather than
a silently wrong number, and the gradient's own C# test compares the analytic gradient against the
numerical derivative the method calls internally (a tautology, no literal to scrape), so its three
pinned probabilities were curated from `verify_oracles.py --dump`. Second, **the pivotal `links`
surface takes name strings, not callbacks** -- one of the seven `LinkFunctionType` names per
parameter (or a null for identity), resolved core-side through `LinkFunctionFactory`, because the
five BestFit-specific links are parameterized (a spec object, not a name) and a name keeps the
emitter's resolution a one-line `Enum.Parse`. Third, the **`type_name` "Unknown" bug**: the R and
Python analysis glue each carry a local distribution-name switch with an `"Unknown"` default and
fourteen arms, and neither the port task nor the 15-candidate task touched it, so
`fit_distributions()` printed the new candidate as `Unknown` while every count and metric passed --
found only by writing the worked example, since the fixture pins that candidate's aic/bic/converged
but not its name. Both packages now assert the name. Two worked example pairs landed (16, the
15-candidate fitting surface; 17, the pivotal bootstrap), and `site/status.qmd` / `site/index.qmd` /
`README.md` were brought to 43 families with the bootstrap row's "documented omission" retired. The
version bump to **0.8.0** records it. Final numbers: **ctest 90/90 (test_fixtures 5579 checks);
oracle gate 5568 reproduced, 0 failed, 11 skipped; testthat 6344/0; pytest 1520**; `R CMD check
--as-cran` holds at the same three NOTEs with no WARNING. No `oracle_skip` and no loosened
tolerance was added anywhere in the phase.

The math-extras phase (P2, branch `port-math-extras`, August 2026) closed the last major slice of
Numerics that is not a distribution, model, or estimator, and is the second step of the release arc
laid out in `docs/superpowers/specs/2026-08-20-remaining-port-and-v1-release-design.md`. **Root
finding** (`Numerics/Mathematics/Root Finding/`) folded Bisection/Secant/Newton into `root_find()`'s
existing bracketing interface as a `method` option beside the default Brent, and added
`root_find_system()` for a vector-valued system via multivariate Newton-Raphson. **Integration**
(`Numerics/Mathematics/Integration/`, the seven integration classes plus the `Integration` statics)
widened `quadrature()` to ten deterministic methods, added `quadrature_2d()` for a rectangle, and
added `quadrature_nd()` for an arbitrary-dimension box over three seeded methods -- Monte Carlo,
Miser, and Vegas (the last with rare-event configuration). **RungeKutta** is `ode_solve()`.
**CubicSpline/Polynomial** extend `interpolate(method =)` beside the existing linear method, with
the transform/extrapolation arguments enforced as linear-only (a guard, not a silent no-op),
matching the C# classes, which have neither. Three new toolbox groups -- `linalg`
(`qr_decomposition()`/`qr_solve()`/`gauss_jordan()`), `special` (`debye()`/`polynomial_eval()`),
and `functions` (`univariate_function()`, the ported `LinearFunction`/`PowerFunction`) -- widen
`toolbox_runner`'s dispatch from eleven groups to fourteen, all on the same standalone-header
pattern the eleven established; the `CallbackSet` gains `scalar_deriv` and `vector_weight` beside
the existing `scalar_xy`. `TabularFunction`, the third `IUnivariateFunction` implementation, is
severed to a later release rather than silently dropped: it depends on the still-unported Paired
Data subsystem (see `upstream/CLAUDE.md`). One worked example pair, 18, exercises the whole surface
ending in an executable reproduction check. The honest fidelity note (established precedent: state
it, do not paper over): at the shipped seeds, `quadrature_nd()`'s `"monte_carlo"` integral
reproduces bit-for-bit across C++, R, Python, and the real C# library and is pinned at zero
tolerance; `"miser"` measured 1 ULP off C#, and `"vegas"` measured 2-3 ULP between the actual
installed R and Python packages, from floating-point contraction differences in the
variance/chi-squared accumulation -- the affected fixture cases assert seeded evaluation counts and
solver status instead of the integral value, with the measurements documented, and no tolerance was
loosened and no oracle was skipped anywhere in the phase. The version bump to **0.9.0** records it.
Final numbers: **ctest 98/98 (test_fixtures 5762 checks); oracle gate 5751 reproduced, 0 failed, 11
skipped; testthat 6622/0; pytest 1619**; `R CMD check --as-cran` holds at the same three NOTEs with
no WARNING.

The optimizers phase (P3, branch `port-optimizers`, August 2026) closed the rest of the Numerics
optimization layer and is the third step of the release arc laid out in
`docs/superpowers/specs/2026-08-20-remaining-port-and-v1-release-design.md`. `optim_minimize()` /
`optim_maximize()` went from six methods to **fourteen**: the four global optimizers
(`particle_swarm`, `sce`, `simulated_annealing`, `multi_start`), the three local ones (`adam`,
`gradient_descent`, `golden_section`), and the constrained `augmented_lagrange` over a new
constraint layer (`optim_constraint()` / `Constraint`, the `constraints` and `inner` arguments, and
the three Lagrange multiplier vectors on the result). `adam`/`gradient_descent` take an optional
analytic `gradient` callback -- the second host-language function to cross into the shared core,
guarded through the same `CallbackAbortState` the callback layer uses. `mlsl` gained the
`local_method` control it had been ported with but never exposed. The dynamic-programming trio
(BinaryHeap, Dijkstra, Network) is not an optimizer -- its input is a graph -- so it joined the
toolbox as the fifteenth group, `network`, reaching users as `shortest_path()` (0-based node
indices in BOTH languages, single-precision weights kept single-precision through the core).
The ctest suites are `core/tests/test_local_optimizers.cpp`, `test_global_optimizers.cpp`,
`test_augmented_lagrange.cpp`, and `test_network_optimization.cpp`, all transcribing the upstream
MSTest methods 1:1 and each carrying a clearly-marked supplement where mutation testing showed the
C# assertions guard the ANSWER but not the ALGORITHM.

Four findings worth carrying forward. First, **`test_global_optimizers` is compiled
`-ffp-contract=off`** and the reason is measured: ParticleSwarm and SCE branch on comparisons
between accumulated sums, .NET never fuses `a*b + c` and clang/gcc do by default, so one contracted
expression flips an accept/reject and every later PRNG draw is spent differently. With the flag the
port is bit-identical to the real C# library on all ten configurations tried; without it SCE's
Eggholder oracle misses outright. The shipped R and Python packages pass no such flag, so a seeded
`particle_swarm` or `sce` run reproduces R-to-Python but NOT necessarily package-to-C# -- unlike
P2's limit, the divergence is inside the shared core, so even the parameters drift. Those two
fixture cases pin only what survives both paths (iteration count, evaluation count, status, a
parameter that lands exactly on a bound); `simulated_annealing` and `multi_start` reproduce C#
exactly down to the evaluation count and ARE pinned exactly. Second, **two ported headers reproduce
upstream array aliasing on purpose** and must not be cleaned up: SCE reuses one scratch point across
its beta loop and stores it by reference, so a later write silently moves an already-scored
sub-complex entry (without it the port does not reproduce C#), and MultiStart re-seats its `values`
pointer onto `InitialValues`, so a finished run's `InitialValues` holds the last sampled restart.
MultiStart's polish step is the same shape with a numeric consequence: it clamps the recorded best
point after its fitness was recorded, so the reported value need not be attained at the reported
parameters (measured on Eggholder; C# returns the same numbers bit for bit). Third, the shipped C#
**`Network` class is entirely unreachable code** -- its constructor sizes both edge caches one short
and never sets `_nodeCount`, `Solve(float[])` ignores the weights it is handed, and `GetPath`
binary-searches an `int[]` for an `Edge` so it can only throw, return null, or return an empty
list. The port diverges on the constructor alone (a ctor that always throws has no behavior to be
faithful to, and the fix is independently checkable: `network.solve(d)` then equals
`dijkstra::solve(edges, d)`, which does run in C#), mirrors the other two exactly with tests
pinning them, severs `GetPath`, and routes the user verb through the free solver. Fourth,
**`AugmentedLagrange` cannot maximize**: `Optimize()` always drives the inner optimizer through
`Minimize()`, so a maximize request returns the constrained minimum labelled Success. The ported
class mirrors it (a fixture must be able to pin upstream); the guard lives on the two public verbs,
which reject the method by name and state the exact workaround. All four are written up in
`docs/upstream-csharp-issues.md`, and one worked example pair, **19**, exercises the whole surface
ending in an executable reproduction check. The version bump to **0.10.0** records it. Final
numbers: **ctest 102/102 (test_fixtures 5953 checks); oracle gate 5942 reproduced, 0 failed, 11
skipped; testthat 6916/0; pytest 1711**; `R CMD check --as-cran` holds at the same three NOTEs with
no WARNING. No `oracle_skip` and no loosened tolerance was added anywhere in the phase.
