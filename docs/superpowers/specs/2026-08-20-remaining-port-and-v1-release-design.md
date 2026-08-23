# Remaining port closure and the 1.0.0 release

Date: 2026-08-20
Status: approved design, pending implementation plans (one per phase)

## Goal

Port every remaining portable file in upstream Numerics and RMC.BestFit, expose each feature in
both R (`corehydror`) and Python (`corehydropy`) with documentation and worked examples, then take
both packages through review and release to CRAN and PyPI at version 1.0.0.

Nine phases. P1 through P6 close the port gaps, each an independently shippable branch following
the established pattern: core port with `// ported from:` provenance, curated fixtures, dotnet
oracle gate extension, shared-runner bindings, pkgdown/quartodoc entries, worked examples, minor
version bump, PR. P7 through P9 are the release arc: academic code review, documentation review,
and release engineering ending in the CRAN submission and PyPI publish.

## Scope

Everything portable. The only exclusions are the existing documented permanent skips, all of which
have no numeric or statistical surface: TimeSeriesDownload (network retrieval), XML persistence
and INPC throughout, AnalysisProgress / BatchAnalysisRunner and its option and result types (GUI
scheduling), the Bulletin17C GMM report text, JsonConverters, SafeProgressReporter, and the
AssemblyInfo / Resources files.

### Inventory of unported code (verified 2026-08-20 by diffing upstream trees against provenance headers)

BestFit is fully ported except the permanent skips. The gaps are all in Numerics:

| Subsystem | Contents | Approx. size |
|---|---|---|
| Heavy TimeSeries container | `TimeSeries.cs`, `Series.cs` base, BlockFunctionType / MathFunctionType / SmoothingFunctionType enums | 2,800 lines |
| Paired Data | OrderedPairedData, UncertainOrderedPairedData, Ordinate, UncertainOrdinate, LineSimplification | 5,300 lines |
| Machine Learning | DecisionTree, RandomForest, KNearestNeighbors, NaiveBayes, GeneralizedLinearModel; KMeans, GaussianMixtureModel, JenksNaturalBreaks; DecisionNode, JenksCluster | 4,100 lines |
| Integration extras | SimpsonsRule, TrapezoidalRule, AdaptiveSimpsonsRule (1D and 2D), AdaptiveGaussLobatto, MonteCarloIntegration, Miser, Vegas | 2,500 lines |
| Optimization extras | Global: ParticleSwarm, ShuffledComplexEvolution, SimulatedAnnealing, MultiStart. Local: ADAM, GoldenSection, GradientDescent. Constrained: AugmentedLagrange plus Constraint types. Dynamic: Dijkstra, Network, BinaryHeap | 2,800 lines |
| Root finding extras | Bisection, NewtonRaphson, Secant | 575 lines |
| HypothesisTests | `HypothesisTests.cs`; un-gates the severed DataFrame hypothesis-test and summary-statistics facades | 350 lines |
| GeneralizedNormal | The one unported univariate distribution; restores the C# 15-candidate FittingAnalysis set (currently 14) | 560 lines |
| Pivotal bootstrap | PivotalBootstrapContext / Diagnostics / InvalidDrawPolicy plus the omitted run-type branches in `bootstrap.hpp` (the Phase-3 severance) | 150 lines plus branches |
| Small items | QRDecomposition, GaussJordanElimination, RungeKutta, CubicSpline, Polynomial, Debye, Evaluate, `Functions/` (IUnivariateFunction, Linear, Power, Tabular), Correlation matrix overloads | 1,600 lines |

Downstream unlocks once the TimeSeries container lands: seasonal PointProcess plus
GeneratePOTTimeSeries and CreateBlockSeries on DataFrame, the ARIMAX covariate forecast-tail
extension (BlockBootstrap / KNN), and the Autocorrelation TimeSeries overloads.

Dependency verified in source: `TimeSeries.ResampleWithKNN` constructs the Machine Learning
`KNearestNeighbors`, so ML must land before the TimeSeries container.

## Invariants (unchanged from prior phases)

- Oracle values live only in `fixtures/*.json`; the four runners (C++, R, Python, dotnet emitter)
  drive the same code path.
- Every new fixture kind or target gets an emitter driver; the dotnet gate must reproduce every
  pinned value.
- Every seeded stream gets a cross-language digest fixture proving bit-identity R vs Python.
- Structural mirroring with provenance headers; no new external C++ dependencies; CRAN and PyPI
  publishability preserved; `-ffp-contract=off` discipline where fixture callbacks stand in for
  host-language arithmetic.
- BestFit sources are read with `git show`, never grep over the working tree (not valid UTF-8).
- After any core class-layout change, R is rebuilt with `--preclean`.

## Phases

### P1. Distribution gaps (v0.8.0)

GeneralizedNormal: new `generalized_normal.hpp`, factory and enum arms, ParameterNames, fixture
file with dotnet-gate pinning, reachable through the existing `distribution()` / `dist_*` surface.
FittingAnalysis goes from 14 to 15 candidates: `fit_distributions_smoke.json` is re-pinned with
shifted candidate indices, and the emitter stops removing GeneralizedNormal from the C# list in
the same change.

Pivotal bootstrap: port the three support files and replace the omitted pivotal run-type branches
in `bootstrap.hpp`, surfaced through `bootstrap_custom` with the new diagnostics DTO. Class layout
changes here require the preclean R rebuild.

### P2. Math extras (v0.9.0)

Root finding (Bisection, NewtonRaphson, Secant), the seven integration classes, RungeKutta,
CubicSpline and Polynomial interpolation, QRDecomposition and GaussJordanElimination, Debye and
Evaluate, and the `Functions/` layer. Callable-driven classes (integrators, root finders, the ODE
solver) join the callback `math` group as new method names; data-driven classes join the toolbox
(`interpolation` group additions plus a new `linalg` group). Everything is exposed: a runner-group
method is cheap and bit-exact parity with C# workflows is the product.

### P3. Optimizers (v0.10.0)

The seven new optimizers as `Optimizer` subclasses dispatched through the existing
`optimizer_runner` (new method names on `optim_minimize`), AugmentedLagrange with a constraint
spec added to the options grammar, and Dijkstra / Network / BinaryHeap as a small toolbox
`network` group. Seeded global optimizers get cross-language digest fixtures; the documented
callback-path limit (user arithmetic is not bit-guaranteed across languages) is stated in the
examples, not papered over.

### P4. Data and tests (v0.11.0)

HypothesisTests, then the severed DataFrame hypothesis-test and summary-statistics facades it
un-gates; the Paired Data layer; the Correlation matrix overloads. New toolbox groups
`hypothesis` and `paired_data`; the DataFrame facades reach the existing model surface.

### P5. Machine Learning (v0.12.0)

The eight ML classes plus support types as a new `ml` runner group with `ml_*` verbs in both
packages (matrix in, native-vector transport like the toolbox). Seeded fits (RandomForest,
KMeans, GMM) get digest fixtures. KNN lands here because P6 needs it.

### P6. TimeSeries container and unlocks (v0.13.0)

The heavy TimeSeries and its Series base (observable-collection semantics, INPC, XML, and file
I/O each remain documented in-file severances) plus the three support enums. Then un-gate
everything waiting on it: seasonal PointProcess, GeneratePOTTimeSeries and CreateBlockSeries on
DataFrame, the ARIMAX covariate forecast-tail extension, and the Autocorrelation TimeSeries
overloads. Exposed as a toolbox `timeseries` group (moving averages, smoothing, block
aggregation, seeded resampling) plus the existing model and analysis surfaces. Last of the port
phases because it re-opens the most ported files and depends on P5's KNN.

Worked examples for P1 through P6 continue the site numbering, one pair (Python notebook plus R
Quarto twin, each ending in an executable reproduction check) per feature area: 16 fitting with all 15
candidates, 17 pivotal bootstrap, 18 numerical methods, 19 global optimizers, 20 hypothesis tests
and paired data, 21 machine learning, 22 the time-series toolkit with seasonal POT and ARIMAX
forecasting.

### P7. Academic code review (v0.14.0 with P8)

An adversarial scientific review of the full numerical surface, run as independent review passes
with findings triaged into fixes. Any fix that changes a value is re-pinned through the dotnet
gate. Dimensions:

- Formula correctness against the literature references upstream cites in its XML docs.
- Edge cases: parameter bounds, degenerate inputs, empty data, NaN propagation.
- RNG and seeding contracts; every documented cross-language guarantee re-verified.
- Statistical semantics of the public APIs (does `confint` mean what a statistician expects;
  are units, tail conventions, and probability orientations stated and consistent).
- Re-read every entry in `docs/upstream-csharp-issues.md` and confirm each documented divergence
  is still justified after P1 through P6.
- JOSS-style artifact review: statement of need, installation, example coverage, contribution
  guidelines, test transparency. Produces CONTRIBUTING.md and the citation collateral P8 ships.

### P8. Documentation review (v0.14.0 with P7)

- Every exported R function: Rd completeness (title, description, every parameter, `\value`, a
  runnable example inside CRAN time limits); pkgdown index complete and organized.
- Every public Python function and class: docstring completeness; quartodoc sections complete.
- Site pass: quickstart, installation pages ready to switch to CRAN/PyPI instructions, every
  example page renders, every reproduction check passes, cross-links sound, prose per the
  elements-of-style conventions.
- READMEs: create `corehydror/README.md`, refresh `corehydropy/README.md` (it is the PyPI long
  description), align the repo README.
- At least one getting-started vignette for the R package (CRAN convention).
- Citation and attribution: `corehydror/inst/CITATION`, a repo-root `CITATION.cff`, citation
  guidance in the Python README, and upstream attribution (USACE-RMC as copyright holder of the
  ported algorithms in `Authors@R` and the LICENSE files).
- Consolidated 1.0.0 release notes in NEWS.md and CHANGELOG.md.

### P9. Release engineering and publishing (v1.0.0)

CRAN track:

- Size audit: the dereferenced source tarball must stay under CRAN's 5 MB guidance (fixtures are
  2.1 MB and core headers 4.4 MB uncompressed today; measure the built tarball and trim or
  compress what the R package actually ships if needed).
- Check-time budget: examples under 5 seconds each, total check time CRAN-acceptable; long tests
  marked `skip_on_cran` while the full suite stays in CI.
- Sanitizer and valgrind passes clean (ASAN, UBSAN, valgrind) on the final surface, extending the
  existing sanitizer work.
- Portability rehearsal: win-builder (release and devel), R-hub v2 matrix, macOS builder.
- DESCRIPTION polish: Title and Description per CRAN conventions, `Authors@R` including USACE-RMC
  as cph, URL and BugReports fields; resolve the 0BSD `file LICENSE` strategy and explain it in
  `cran-comments.md` along with the vendored-header path note.
- Submit; respond to CRAN reviewer feedback. The manual steps that only the maintainer can do
  (submission confirmation email) are documented for Cam.

PyPI track:

- Wheel matrix via cibuildwheel in a tag-driven GitHub Actions release workflow: manylinux and
  musllinux x86_64 and aarch64, macOS arm64 and x86_64, Windows amd64, for CPython 3.10 through
  3.13 (add 3.14 if the toolchain supports it at release time); sdist built through `tools/materialize_core.py` (symlink-free, verified installable).
- Metadata: classifiers, 0BSD license expression, README as long description, project URLs
  (docs site, repository, issues).
- TestPyPI rehearsal end to end, then publish with trusted publishing (OIDC); no long-lived
  tokens.

Repo and release:

- Confirm name availability on CRAN (`corehydror`) and PyPI (`corehydropy`) before anything else
  in P9.
- Both packages to 1.0.0; tag `v1.0.0`; GitHub release with the consolidated notes; docs site
  installation instructions switch to CRAN/PyPI; optional Zenodo DOI archive (ask Cam at P9).
- Phase-complete condition: wheels and sdist live on PyPI and the CRAN submission made. CRAN
  acceptance timing is outside our control; absorbing reviewer feedback is an explicit follow-up
  task, and 1.0.0 is not re-tagged for CRAN-requested metadata tweaks (patch releases instead).

## Version trajectory

0.8.0 (P1), 0.9.0 (P2), 0.10.0 (P3), 0.11.0 (P4), 0.12.0 (P5), 0.13.0 (P6), 0.14.0 (P7+P8),
1.0.0 (P9). The 1.0.0 statement: full portable parity with upstream Numerics and RMC.BestFit,
reviewed, documented, and published.

## Risks

- The `fit_distributions` 14-to-15 re-pin shifts candidate indices; the fixture, the emitter, and
  both packages' tests must move in one change.
- Pivotal bootstrap and the TimeSeries/DataFrame work change class layouts; preclean R rebuilds,
  and P6 deliberately isolates the most cross-cutting edits.
- CRAN package size and check time are the most likely submission blockers; P9 measures both
  before submitting and the fallback (trimming shipped fixtures, `skip_on_cran`) is defined.
- The 0BSD license token is not in R's license database; the `file LICENSE` route plus
  cran-comments explanation is the plan, with relicensing explicitly out of scope.
- Seeded global optimizers and ML fits may surface new chaotic-sensitivity cases; the established
  precedent applies (prove amplification by conditioning, assert structural invariants, never
  loosen tolerance or add oracle_skip masks to hide it).

## Out of scope

The permanent skips listed under Scope; the RMC.BestFit.App / .UI / .Api projects; relicensing;
absorbing future upstream releases (the `docs/upstream-sync.md` process covers those separately).
