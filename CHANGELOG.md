# Changelog

All notable changes to corehydro (the shared C++ core, the `corehydror` R package, and
the `corehydropy` Python package) are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). The three components are versioned together.

## [Unreleased]

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
  raw chains in the `posterior`-package axis order, a posterior summary with R-hat and effective
  sample size, and DIC/WAIC/LOOIC.
- **`fit_gmm()`** -- generalized method of moments for Bulletin 17C models, with the sandwich
  covariance and the J-statistic overidentification test.
- **`fit_diagnostics()`** -- Cook's distance, leverage, and observation influence off a MAP or GMM
  fit; leverage, PSIS-LOO Pareto-k, and prior influence off a Bayesian fit.
- **`quantile_variance()`** -- the Cohn-style delta-method variance of a fitted quantile at a given
  annual exceedance probability, off a `fit_gmm()` fit.
- `confint()`, `coef()`, `vcov()`, `logLik()`, `AIC()`, `BIC()`, `print()`, and `summary()` methods
  on the new `corehydro_fit` object in both packages.
- A worked example pair on fitting a Log-Pearson Type III record by all four methods and comparing
  the resulting quantile estimates.

### Fixed

- **A model with fewer than two parameters no longer returns garbage covariance, standard errors,
  or correlation from R.** `GetCovarianceMatrix` throws below two parameters in C#, and R does not
  zero-fill a freshly allocated numeric vector, so the old fixture glue was handing back whatever
  was already sitting in that memory rather than the zeros it appeared to return.
- **A failed fit now names the estimator and the optimizer in its error**, rather than the internal
  message "failed for a fixture case" that leaked through from the pre-phase glue.
- **`confint()`'s default level agreed between R and Python before this shipped, not after**: both
  now default to 0.95, matching base R's `confint()` convention.

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

[Unreleased]: https://github.com/cameronbracken/corehydro/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/cameronbracken/corehydro/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cameronbracken/corehydro/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cameronbracken/corehydro/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cameronbracken/corehydro/releases/tag/v0.1.0
