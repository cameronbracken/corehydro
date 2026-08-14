# corehydror 0.6.0

The numerics toolbox layer. Every general-purpose Numerics utility that is not itself a
distribution, copula, model, or estimator is now reachable from R: correlation, goodness of fit,
descriptive and running statistics, spectral analysis, histograms, interpolation, linear
regression, Sobol sequences, stratification, joint probability, link functions, trend evaluation,
and all six ported optimizers. See `CHANGELOG.md` at the repository root for the full account,
including the internal runner design and the review-round fixes; this file lists the
package-facing additions and fixes.

## New features

* `correlation()` -- Pearson, Spearman, and Kendall's tau.
* `goodness_of_fit()`, `classification_metrics()`, `gof_test()`, `gof_rmse()`, `aic()`, `aicc()`,
  `bic()`, `aic_weights()`, `rmse_weights()` -- continuous and binary-classification
  goodness-of-fit metrics, distribution test statistics, information criteria, and model weights.
* `summary_statistics()`, `product_moments()`, `l_moments()`, `ranks()`, `percentile()`,
  `running_statistics()`, `running_covariance()`, `autocorrelation()`, `cross_correlation()`,
  `dft()`, `dft_real()` -- descriptive and streaming statistics, autocorrelation with PACF and
  confidence bands, and the discrete Fourier transform.
* `histogram()`, `interpolate()`, `interpolate_2d()` -- histogram binning and one/two-dimensional
  interpolation with optional axis transforms.
* `linear_regression()` -- ordinary least squares, returning a `corehydro_lm` with
  `print`/`summary`/`coef`/`vcov`/`residuals`/`predict` methods.
* `sobol_sequence()`, `stratify()`, `joint_probability()` -- quasi-random sequences, interval
  stratification, and joint exceedance probability.
* `link_function()`, `link()`, `link_inverse()`, `link_derivative()`, `link_names()`,
  `trend_predict()`, `trend_parameters()`, `trend_names()` -- the Numerics link functions and the
  ten BestFit trend models.
* `optim_minimize()`, `optim_maximize()` -- the six ported optimizers (Differential Evolution,
  BFGS, Powell, MLSL, Nelder-Mead, Brent) over a user-written R objective.
* Two new worked examples: model evaluation (ranking candidate distributions with
  `goodness_of_fit()`/`aic_weights()`/`gof_test()`) and a custom objective (fitting a
  hand-written likelihood with `optim_minimize()`).

## Bug fixes

* `linear_regression()`'s `vcov()` was unscaled -- it returned the raw `(X'X)^-1` term rather than
  that term times `sigma^2`, disagreeing with `sqrt(diag(vcov(fit))) == standard_errors` by a
  factor of `sigma^2`. Now scaled correctly.
* The `optim_minimize()`/`optim_maximize()` callback guard had a hole for `method = "bfgs"` and
  `method = "mlsl"`: an R condition raised inside the objective could be replaced by an internal
  C++ exception before it reached R. All six methods now always propagate the original R
  condition.
* `classification_metrics()` had an undocumented, invented threshold argument that neither this
  package nor the underlying C# source has; removed.
* `stratify()` now rejects `lower >= upper` instead of silently returning zero rows, and
  `joint_probability(dependency = "correlation")` rejects an inconsistent matrix/no-matrix
  argument combination instead of one half throwing and the other silently returning `NaN`.
* `running_covariance()`'s resume state failed to round-trip for a single-variable accumulator.

## Internal

* `Numerics/Data/Statistics/Autocorrelation.cs`, unported since the project began, is now ported
  (`autocorrelation()`/its confidence-band verb); its sibling `Correlation.cs`'s matrix overloads
  remain a documented severance (no caller needs a full correlation matrix yet).
* The toolbox surface is served by two new shared C++ runners
  (`numerics/support/toolbox_runner.hpp`, `numerics/support/optimizer_runner.hpp`) driven
  identically by the R glue, the Python glue, the C++ fixture runner, and the dotnet oracle
  emitter.

# corehydror 0.5.0

The distribution layer: five composite univariate distributions (`dist_truncated()`,
`dist_mixture()`, `dist_competing_risks()`, `dist_empirical()`, `dist_kde()`), all seven bivariate
copulas (`copula()`/`copula_fit()`), and the five multivariate distributions
(`mvdist_normal()`, `mvdist_student_t()`, `mvdist_dirichlet()`, `mvdist_multinomial()`,
`mvdist_bivariate_empirical()`) are reachable from R for the first time.

# corehydror 0.4.0

The estimation layer: `fit_mle()`, `fit_map()`, `fit_bayesian()`, and `fit_gmm()` fit any
`model_*()` object directly, returning a `corehydro_fit` with `coef()`, `confint()`, `AIC()`,
`logLik()`, `summary()`, `fit_diagnostics()`, and `quantile_variance()`.

# corehydror 0.3.0

The data and model layer: `analysis_data()` for censored observations (historical/paleoflood
intervals, perception thresholds, measurement uncertainty, low outliers), twelve `model_*()`
constructors covering every RMC.BestFit model family, and nonstationary trends and per-parameter
priors/bounds via `trend()`/`model_parameter()`.

# corehydror 0.2.0

Upstream sync to Numerics v2.1.4 and RMC.BestFit v2.0.0.

# corehydror 0.1.0

Initial release (renamed from `bestfitr`). All 42 univariate distributions, the full Numerics
MCMC/bootstrap/optimization foundation, and the complete RMC.BestFit Models, Estimation,
Analyses, and Diagnostics layers.
