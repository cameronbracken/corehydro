# Changelog

All notable changes to corehydro (the shared C++ core, the `corehydror` R package, and
the `corehydropy` Python package) are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). The three components are versioned together.

## [Unreleased]

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
  `trend_predict()`, `trend_parameters()`, `trend_names()`** -- the Numerics link-function layer
  (identity, log, logit, probit, complementary-log-log, Fisher-z, Yeo-Johnson) and evaluation of
  the ten BestFit trend models. Python's `link_function()` takes `lambda_` rather than `lambda`
  (a reserved word), the one deliberate naming difference from R, matching the existing
  `corehydropy.stats` precedent.
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
  unrecorded severance -- no caller has needed them yet -- now written up in
  `numerics/data/correlation.hpp`'s own file comment and `upstream/CLAUDE.md`, alongside the
  Autocorrelation port.
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
- `histogram()` rejected too few or too many explicit bins inconsistently between languages, and a
  toolbox dispatch error from the C# emitter leaked its internal message instead of naming the
  method; both now validate and report symmetrically.
- The joint-probability runner threw for one half of the `dependency = "correlation"` no-matrix
  case while silently returning `NaN` for the other; both now fall through to `NaN`, matching the
  C# source, and the R/Python wrappers reject the combination symmetrically instead of leaking the
  asymmetry. `stratify()` now also rejects `lower >= upper` instead of silently returning zero
  rows.
- `running_covariance()`'s resume state was not array-wrapped, so a single-variable accumulator
  failed to round-trip through a second call.

### Known limitation

- The dotnet oracle emitter's `ToolboxSelectFlat` helper (used for groups whose C# result is a
  positional flattened array -- Sobol, Stratify, Histogram bins, linear regression's
  coefficient/covariance/residual matrices) has no `names` array to look an assertion's `label`
  key up against, unlike the C++/R/Python `toolbox_select` helpers, which honor it. Recorded
  rather than fixed: no fixture pairs `label` with one of those flattened-matrix groups today, so
  the gap is latent. See `ToolboxSelectFlat`'s header comment in `tools/oracle_emitter/Program.cs`
  and `fixtures/README.md`'s `toolbox` section.

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

[Unreleased]: https://github.com/cameronbracken/corehydro/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/cameronbracken/corehydro/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/cameronbracken/corehydro/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cameronbracken/corehydro/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cameronbracken/corehydro/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cameronbracken/corehydro/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cameronbracken/corehydro/releases/tag/v0.1.0
