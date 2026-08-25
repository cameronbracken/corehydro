# CLAUDE.md — `upstream/` (vendored USACE-RMC C# sources)

Context for the two C# libraries that `corehydro` ports from. They live here as **dev-only git
submodules**, pinned to a release tag on the official USACE-RMC `main` branches:

- `upstream/Numerics/` — `github.com/USACE-RMC/Numerics` (the numerical-computing core).
  Pinned at **`2a0357a` = v2.1.4**.
- `upstream/RMC-BestFit/` — `github.com/USACE-RMC/RMC-BestFit` (the Bayesian flood-frequency app
  library). Pinned at **`c2e6192` = v2.0.0**.

The July 2026 upstream sync moved both pins from `a2c4dbf` / `fc28c0c` and restored 1:1 parity.
The repeatable process for absorbing the next release is `docs/upstream-sync.md`.

They are the **diff baseline** for the upstream-sync workflow and the source the C++ core mirrors
file-for-file. They are **never vendored into the shipped packages** (`actions/checkout` uses
`submodules: false`; the CRAN/PyPI sdists don't reference them). `dotnet` is installed locally, so
the C# tests can be run to verify oracle values (`tools/oracle_emitter` / `tools/verify_oracles.py`),
but CI does not build them. When porting, open the matching `.cs` file here, mirror its class/method
layout into `core/`, and carry a `// ported from: <path> @ <sha>` provenance header.

This file consolidates the per-repo guidance that previously lived at
`upstream/Numerics/CLAUDE.md` and `upstream/RMC-BestFit/CLAUDE.md`, so the submodule working trees
stay clean while the context is tracked by the `corehydro` repo.

**Reading BestFit sources:** the working-tree copies under `upstream/RMC-BestFit/src/` are not
valid UTF-8, so a text search over the checkout silently misses matches and returns mojibake. Read
them with `git show c2e6192:<path>` and diff them with `git diff`, never with `grep` over the
working tree.

---

## Numerics (`upstream/Numerics/`)

**Numerics** (`RMC.Numerics` on NuGet) is a free, open-source .NET numerical-computing library from
the U.S. Army Corps of Engineers Risk Management Center (USACE-RMC). It covers probability
distributions, statistics, numerical methods, optimization, machine learning, and Bayesian MCMC,
with a focus on hydrology and quantitative risk assessment. Much of the code has **life-safety
applications** (flood frequency, dam safety), so numerical correctness and validation against
reference implementations are paramount. The canonical upstream is `github.com/USACE-RMC/Numerics`.

### Commands

Multi-targets `net10.0;net9.0;net8.0;net481` (SDK 10.0.100, see `global.json`). .NET Framework
4.8.1 only builds/tests on Windows; on macOS/Linux restrict to a modern TFM with `-f net10.0`.

```bash
# Build just the library (skip net481 off-Windows)
dotnet build Numerics/Numerics.csproj -c Release -f net10.0
# Run the full test suite (1000+ tests, MSTest)
dotnet test -f net10.0
# Run one test class or method by name
dotnet test -f net10.0 --filter "FullyQualifiedName~Test_Normal"
dotnet test -f net10.0 --filter "Name=Test_PDF"
```

A clean build is **zero errors and zero warnings** (`Nullable`, `EnforceCodeStyleInBuild`,
warning-sensitive; `CS1587` suppressed because license headers sit in XML-comment blocks).

### Architecture

Single library project `Numerics/` plus test project `Test_Numerics/`; the folder layout mirrors the
namespace tree and the tests mirror the library folder-for-folder. Root namespace `Numerics`:

- **`Distributions`** — the core. 40+ univariate distributions in `Distributions/Univariate/`, plus
  `Bivariate Copulas` and `Multivariate`. Univariate types derive from `UnivariateDistributionBase`
  and implement `IUnivariateDistribution`/`IDistribution` (in `Univariate/Base/`).
  `UnivariateDistributionFactory` + the `UnivariateDistributionType` enum give string/enum-driven
  construction. Estimation (MLE, MoM, L-moments) is under `Univariate/Parameter Estimation/`;
  `Univariate/Uncertainty Analysis/` handles confidence intervals and Bayesian uncertainty.
- **`Mathematics`** — `Integration`, `Differentiation`, `Optimization`, `Root Finding`,
  `Linear Algebra`, `ODE Solvers`, `Fourier Methods`, `Special Functions`.
- **`Sampling`** — RNG (`MersenneTwister`, `SobolSequence`, Latin hypercube, stratification,
  bootstrap) and MCMC samplers (`RWMH`, `ARWMH`, `DEMCz`, `DEMCzs`, `HMC`, `NUTS`, `Gibbs`, `SNIS`).
- **`Data`** — `Interpolation`, `Regression`, `Statistics` (descriptive, goodness-of-fit, hypothesis
  tests), `Time Series`, `Paired Data`.
- **`Machine Learning`**, **`Functions`** (`IUnivariateFunction` + impls), **`Utilities`** (`Tools`,
  `ExtensionMethods`, `JsonConverters`).

The Sobol generator embeds `Properties/new-joe-kuo-6.21201` as a resource.

### Conventions

- **XML documentation is mandatory** on public types/members; most algorithm files carry `<remarks>`
  with author attribution and literature `<references>`.
- **Validate, don't just assert.** Tests check numerical output against a known reference (R, SciPy,
  Mathematica, published tables) and cite the source — they double as the oracle values `bestfit`
  ports into `fixtures/`.
- **MSTest** (`[TestClass]`/`[TestMethod]`), one test class per type, named `Test_<Type>`.
- `ImplicitUsings` is on. Reproducible builds + SourceLink are on.

### Port history notes

- **`Numerics/Data/Statistics/Autocorrelation.cs`** was unported through Phase 0-10 (the port
  that reached FULL PARITY on every distribution/estimator/model/analysis surface never needed
  it) and through the upstream sync and the estimation/distribution surface phases that followed.
  It was ported in the `surface-numerics-toolbox` branch's Task 3 (August 2026) as
  `core/include/corehydro/numerics/data/autocorrelation.hpp`, reachable from R and Python via the
  `spectra` toolbox group's `autocorrelation`/`autocorrelation_ci` verbs. Only the
  `IList<double>` overloads are ported; the `TimeSeries` overloads remain a documented severance
  (see that header's own file comment). Its sibling `Correlation.cs`'s matrix overloads
  (`Correlation.Pearson(double[,])` / `Correlation.Spearman(double[,])`, column-pairwise
  correlation matrices over an `[n, p]` table) were ported in the P4 "data and tests" phase's
  Task 4 (August 2026) as `pearson_matrix`/`spearman_matrix` in
  `numerics/data/correlation.hpp`, reachable from R and Python via `correlation(x, y = NULL)`
  (a matrix or data frame in `x`). Neither overload had an upstream test or in-library caller,
  so every oracle value is curated via `oracle_emitter --dump` against the real
  `Numerics.Data.Statistics.Correlation` rather than scraped from a C# test file. There is no
  `KendallsTau(double[,])` overload upstream, so the matrix severance is now exactly that one
  method -- rejected by name in both languages, not silently unavailable.
- The **Paired Data subsystem** (`Numerics/Data/Paired Data/`: `Ordinate`, `OrderedPairedData`,
  `UncertainOrdinate`, `UncertainOrderedPairedData`, `LineSimplification`) and its consumer
  **`Numerics/Functions/TabularFunction.cs`** were both unported through every prior phase; nothing
  in scope needed a curve container until the P4 "data and tests" phase's Tasks 7-9 (August 2026)
  ported the whole subsystem (`core/include/corehydro/numerics/data/paired_data/`) and
  TabularFunction itself (`numerics/functions/tabular_function.hpp`), retiring both severances in
  one pass -- TabularFunction could not have been ported earlier against a stand-in container
  without losing fidelity to its `UncertainOrderedPairedData`-backed `IsDeterministic` toggle. The
  same tasks also un-severed the six `OrderedPairedData`/`Ordinate` overloads of
  `Numerics/Data/Interpolation/Support/Search.cs`'s `Sequential`/`Bisection`/`Hunt` (C# lines 167,
  254, 444, 545, 782, 925) -- unported since Phase 2 because there was no `OrderedPairedData` to
  search -- while the plain `IList<double>` `Hunt` overload (C# line 650) remains unported (no
  caller in this port's scope needs a Hunt search over a bare array; see `search.hpp`'s own header
  for the fidelity notes on both un-severed families). All three retirements are reachable from R
  and Python via the P4 Task 10 `curve_*`/`uncertain_curve_*`/`tabular_function` toolbox surface.
- **`RMC.BestFit`'s `DataFrame.UnimodalityTest` and `DataFrame.SummaryHypothesisTest`** (both in
  `#region Hypothesis Testing`, `Models/DataFrame/DataFrame.cs`) are the two members the P4 "data
  and tests" phase's Task 5 left deferred when it un-gated the other eleven `#region Hypothesis
  Testing`/`#region Summary Statistics` facade members. Both need
  `Numerics.MachineLearning.GaussianMixtureModel`, which remains unported (see
  `core/include/corehydro/numerics/data/hypothesis_tests.hpp`'s own header note on
  `UnimodalityTest`). Deferred together to **P5**, when `GaussianMixtureModel` lands, because
  `SummaryHypothesisTest` calls all ten hypothesis-test facades (the nine already-ported ones plus
  Unimodality) inside one `try`/`catch` that NaNs its entire ten-key result dictionary if any
  single call throws -- shipping it with a throwing `UnimodalityTest` arm would silently NaN nine
  otherwise-working results rather than surface the real gap. `docs/upstream-csharp-issues.md`
  separately records a Mann-Whitney argument-selection pattern in `SummaryHypothesisTest` that
  looks suspicious on a first read but was checked against the real ternary semantics and is not a
  bug.

---

## RMC.BestFit (`upstream/RMC-BestFit/`)

RMC-BestFit is a Bayesian estimation and flood-frequency fitting application from USACE-RMC. **This
repository contains only the computational model library (`RMC.BestFit`) and its unit tests** — the
WPF desktop GUI is a separate, not-yet-open-sourced codebase. There is no entry point and nothing to
"run"; the library is consumed by the GUI and exercised here through tests. `examples/` holds
tutorial `.bestfit` projects (SQLite) and `.md` walkthroughs — data/docs, not code.

### Build and test

Toolchain is **.NET 10** (`net10.0`), `Nullable` + `ImplicitUsings` on. There is **no `.sln`** —
build the projects directly:

```bash
dotnet build src/RMC.BestFit/RMC.BestFit.csproj
dotnet test  src/RMC.BestFit.Tests/RMC.BestFit.Tests.csproj   # MSTest, ~2790 tests / 120 classes
dotnet test  src/RMC.BestFit.Tests/RMC.BestFit.Tests.csproj --filter "FullyQualifiedName~BayesianAnalysisTests"
```

Since v2.0.0 the repo has an `RMC.BestFit.sln`, and `RMC.BestFit.csproj` takes Numerics as a
`PackageReference` on the published `RMC.Numerics` package rather than a `HintPath` to a sibling
build. The oracle emitter deliberately does NOT use that package reference: it subset-compiles the
BestFit sources directly against the local `upstream/Numerics` ProjectReference, so the oracles
come from the pinned submodule rather than whatever NuGet resolves. Keep it that way.

For oracle work `corehydro` does not need a full BestFit build; it reads expected values from the
`.cs` test files and reproduces them through `tools/oracle_emitter` against the local Numerics
build.

### Architecture

Separates **models** (the math: a distribution, its parameters, its likelihood) from **analyses**
(orchestration: take data, fit a model, produce results). Two parallel base hierarchies:

- **`Models/Support/ModelBase.cs` + `IModel`** — a fittable model owns a `List<ModelParameter>` (value
  + prior) and exposes the likelihood surface estimators optimize/sample: `LogLikelihood`,
  `DataLogLikelihood`, `PriorLogLikelihood`, plus pointwise variants for WAIC / PSIS-LOO. Models
  `Clone()`, serialize to/from XML, are **mutable**, and implement `INotifyPropertyChanged` (WPF
  binding objects — the "never mutate" rule does not apply to them).
- **`Analyses/Support/AnalysisBase.cs` + `IAnalysis`** — a runnable analysis with a lifecycle
  (`RunAsync`, `Validate`, events, cancellation, an `IsEstimated` flag, a `_reprocessGate` semaphore
  serializing runs against background reprocesses).

Subsystems: **`Models/`** (`UnivariateDistribution/` incl. `Bulletin17CDistribution`, `MixtureModel`,
`CompetingRisksModel`, `PointProcessModel`; `BivariateDistribution/` copulas; `RatingCurve/`;
`TimeSeries/` AR/MA/ARIMA/ARIMAX; `SpatialExtremes/`; and `DataFrame/` — distinguishing
`ExactData`/`IntervalData`/`ThresholdData`/`UncertainData` so the likelihood handles censored /
interval / measurement-error observations uniformly), **`Estimation/`** (`MaximumLikelihood`,
`MaximumAPosteriori`, `GeneralizedMethodOfMoments`, `BayesianAnalysis`/MCMC, plus `NumericalDiff` /
`OptimizationMethod`), **`Analyses/`** (one per workflow), **`Diagnostics/`** (posterior/prior
predictive checks, influence/leverage). Tests mirror the source layout by domain.

Note: `Analyses/TimeSeries/Support/**` and `Analyses/Univariate/Support/**` are `Compile Remove`d
from `RMC.BestFit.csproj` — files placed there are not built.

v2.0.0 also added three sibling projects that this port does not cover: `src/RMC.BestFit.App`
(the WPF desktop GUI, now open-sourced), `src/RMC.BestFit.UI` (its wrapper layer), and
`src/RMC.BestFit.Api` (a REST/MCP server). They are out of scope for a statistics-library port;
R and Python users reach the same functionality through corehydro's own APIs.

---

## What is deliberately not ported

These upstream areas are severed by design. Each is also noted in the C++ header that would
otherwise own it, and each is expected to reappear in the next sync's provenance sweep as a
changed-upstream file whose pin was deliberately not moved.

- **`Numerics/Data/Time Series/Support/TimeSeriesDownload.cs`** — network retrieval of gauge
  records (USGS and similar). v2.1.4 rewrote it substantially (+754/-237 lines). corehydro takes
  data from the caller; R and Python users have far better HTTP tooling natively.
- **`Numerics/Data/Time Series/Support/Series.cs`** — the observable-collection container behind
  the heavy Numerics `TimeSeries`. v2.1.4 changed its `Remove` / `RemoveAt` / `Clear` semantics.
  corehydro's `numerics/data/time_series/time_series.hpp` is a thin adapter over
  `std::vector<double>` with no mutation surface of that shape, so the change has nothing to land
  on. The 2,334-line Numerics container itself (interpolation, file I/O, hypothesis tests) remains
  a documented severance.
- **`src/RMC.BestFit/Analyses/Support/AnalysisProgress.cs`** — new in v2.0.0. GUI progress
  plumbing: phase percentages, a safe progress reporter, and an `AsyncLocal` parallelism budget.
  Its adoption is what makes the fifteen analysis orchestrators' diffs look large; those hunks are
  progress reporting, `ParallelOptions` construction, and XML-restore state normalization, none of
  which has a numeric surface. Order-independent seeded loops mean no effect on any oracle.
- **`src/RMC.BestFit/Analyses/Support/BatchAnalysisRunner.cs`** and its options and result types —
  WPF batch scheduling. Permanent skip.
- **Bulletin17CAnalysis GMM report generation** — roughly 600 lines of `StringBuilder` text.
  Presentation only, no numeric or statistical surface.
- **XML persistence** throughout, including the `PlottingPositionVersion` stamps, the
  `TrainingTimeSteps` restore, and `FromXElement` recomputation. corehydro severs XML entirely.
  One consequence is recorded in `docs/upstream-csharp-issues.md`: a bare `ExactData.PlottingPosition`
  mutation bumps the C# cache version but not this port's, because the port has no INPC layer.
- **`Tools.ParallelAdd` hardening** — corehydro uses serial reductions by design, which is
  stronger than the upstream property.
- **`Numerics/Mathematics/Optimization/Dynamic/Network.cs`'s two `GetPath` overloads** — the
  alternate-route search. Ported structurally into
  `core/include/corehydro/numerics/math/optimization/dynamic/network.hpp` so upstream diffs keep
  mapping, but severed from the R/Python surface: MEASURED against the real library, the method
  cannot return a path. Its edge filter is `Array.BinarySearch(int[] edgesToRemove, Edge edge)`,
  which binds the `(Array, object)` overload and throws `InvalidOperationException: Failed to
  compare two elements in the array.` for any non-empty removal list, and the one input that gets
  past it (an empty list, where a zero-length search never invokes the comparer) then falls out of
  a `do { ... } while (heap.Count == 0)` loop with `foundPath` still false. Throw, null, or an
  empty list -- never a path. No upstream test reaches it. The toolbox `network` group exposes the
  `Solve` overloads only. Full write-up, with the probe transcript, in
  `docs/upstream-csharp-issues.md`; the same investigation recorded two further defects in that
  file, one of which (`Network`'s constructor sizing both edge caches one element short, so every
  construction throws) is the port's one intentional divergence in that header.
