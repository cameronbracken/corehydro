# corehydror 0.8.0

Two gaps the port left open, closed. `GeneralizedNormal` was the one univariate family named in
the library's type enum with nothing behind it, so the package now carries 43 distributions rather
than 42 and `fit_distributions()` ranks 15 candidates rather than 14. The covariance-aware pivotal
bootstrap is ported and reachable as `bootstrap_custom(run_type = "pivotal")`. This release also
carries two memory-safety and undefined-behaviour fixes from an AddressSanitizer and
UndefinedBehaviorSanitizer run, both user-visible. See `CHANGELOG.md` at the repository root for
the full account.

## New features

* `GeneralizedNormal`, the three-parameter log-normal, available by name from
  `distribution()` and every `dist_*` verb: `distribution("GeneralizedNormal", c(100, 10, 0))`.
  Location, scale and shape, with the shape's sign setting which tail is bounded. Every family in
  the library's univariate type enum now constructs.
* `bootstrap_custom(run_type = "pivotal")` -- the covariance-aware pivotal bootstrap. A regular run
  keeps one point estimate per replicate and throws that replicate's own estimation uncertainty
  away; a pivotal run keeps the estimate and its covariance, standardizes each replicate against
  the parent fit in link space, and reinflates through the parent covariance. Fit through
  `fit_with_covariance` instead of `fit`. Nine new arguments carry it: `run_type`,
  `fit_with_covariance`, `original_covariance`, `pivotal_links`, `pivotal_invalid_draw_policy`,
  `regularize_pivotal_covariances`, `pivotal_z_limit`, `add_pivotal_jitter` and
  `pivotal_jitter_scale`. `pivotal_links` takes a link function **name** per parameter (`"Log"` on
  a scale parameter keeps every reinflated draw positive), not a function. A pivotal result adds
  `pivotal_diagnostics` and the raw (non-pivotal) block beside the pivotal one, so the two
  intervals can be compared in the same run; it is percentile-only, and refuses the arguments it
  cannot use.

## Changes to results

* `fit_distributions()` returns 15 rows, not 14. GeneralizedNormal enters the candidate list fifth,
  where the C# library puts it, so **code indexing the result positionally past the fourth row
  shifts by one**. Ranking with `which.min(aic)` is unaffected, except that a new candidate can now
  win.
* `l_moments()`, and every fit that estimates from sample L-moments, now forms its weight products
  in floating point. Those products overflowed a 32-bit integer at 1293 points, so the L-kurtosis
  of any sample of **1293 or more points** was wrong: on an evenly spaced series whose true
  L-kurtosis is 0, the old code returned -0.185 at 1293 points and -1.45 at 1300. Shorter samples
  are bit-identical to before. This is a deliberate divergence from upstream, which still wraps, so
  corehydror and RMC-BestFit disagree on the L-kurtosis of any sample of 1293 or more points.

## Bug fixes

* `fit_distributions()` printed the new candidate's name as `"Unknown"`. The name lookup in the
  binding glue had fourteen arms and a default; the fixture pins that candidate's AIC, BIC and
  converged flag but not its name, so every count and metric passed while the row printed
  `Unknown`. The test suite now asserts the name too.
* `fit_distributions()` could report the GeneralizedPareto candidate as failed, or rank it
  differently between platforms and between runs. The distribution returned two parameter
  constraints where it has three parameters, so the candidate's starting value, bounds and prior
  were read one element past the end of a vector, from whatever the allocator had left there. On
  the dataset used by the test fixtures the candidate came back with an AIC of `NaN` where it
  should converge at 423.31, the lowest of any candidate. **Results from `fit_distributions()`
  involving GeneralizedPareto may change; refit if you kept earlier output.** The same read reached
  the GeneralizedPareto seed used by `model_*()` defaults and trend models, `mcmc_sample()`'s model
  registry, the copula marginal pre-fit, the mixture, competing-risks and point-process component
  seeds, and Bulletin 17C.

## Changes to results

* `l_moments()`, and every fit that estimates from sample L-moments, now forms its weight products
  in floating point. Those products overflowed a 32-bit integer at 1293 points, so the L-kurtosis
  of any sample of **1293 or more points** was wrong: on an evenly spaced series whose true
  L-kurtosis is 0, the old code returned -0.185 at 1293 points and -1.45 at 1300. Shorter samples
  are bit-identical to before. This is a deliberate divergence from upstream, which still wraps, so
  corehydror and RMC-BestFit disagree on the L-kurtosis of any sample of 1293 or more points.

# corehydror 0.7.0

The callback layer. Five upstream classes are delegate-driven by design, and until now the package
could reach them only through internal registries, so the model always had to be one the port
already knew how to build. Your own R function can now drive them: a log-likelihood, a Gibbs
proposal, an HMC/NUTS gradient, the four bootstrap delegates, and a set of GMM moment conditions.
See `CHANGELOG.md` at the repository root for the full account, including the guard design and the
review-round fixes; this file lists the package-facing additions and fixes.

## New features

* `mcmc_posterior()` -- sample a posterior whose likelihood is not in the package, over priors you
  choose. All eight ported samplers are reachable, including `"Gibbs"`, which needs a
  model-specific conditional `proposal` and so could not be run before. `"HMC"` and `"NUTS"` take
  an optional analytic `gradient`.
* `bootstrap_custom()` -- a confidence interval on any statistic you can compute from a fitted
  parameter set, over the four delegates `resample`, `fit`, `statistic` and `jackknife`, and the
  five interval methods Percentile, BiasCorrected, Normal, BootstrapT and BCa.
* `fit_gmm_moments()` -- fit moment conditions you write, with an optional analytic `jacobian` and
  `penalty`. Returns the same `corehydro_fit` object `fit_gmm()` does. `fit_gmm()` itself can only
  fit a `model_bulletin17c()` model.
* `root_find()`, `derivative()`, `gradient()`, `hessian()`, `quadrature()` -- Brent root finding,
  numerical differentiation, and adaptive Gauss-Kronrod integration over a plain R function.
  `gradient()` and `hessian()` collide with `numDeriv` and `pracma`, so the examples call them as
  `corehydror::gradient()`.
* `rng_uniform()`, `rng_integers()` -- draw from the handle a callback is given on the generator
  the run is already using. Drawing with `runif()` or `sample()` instead leaves a seeded run
  unreproducible.

## Bug fixes

* The RNG handle refuses a foreign external pointer. It previously accepted any `EXTPTRSXP` and
  cast it, so an `Rcpp::XPtr`, or the address slot of a registered native routine, segfaulted and
  aborted the R session.
* `rng_integers()` no longer overflows on a span wider than `.Machine$integer.max`; it raises, as
  the ported generator does.
* `print()` and `summary()` on a GMM fit no longer show the J statistic where it cannot be
  trusted: at zero over-identifying degrees of freedom, or when the statistic is not finite. The
  `$j_stat` field is untouched.
* The `bootstrap_custom()` jackknife documentation claimed `data[-index]` returns the sample
  untouched at index 0; in R that is `data[-0]`, the empty vector.
* An R log-likelihood returning `NA` is refused by name instead of walking a motionless chain, and
  a `NULL` seed is refused by name instead of failing inside the JSON builder.

## Documentation

* Two worked examples: a custom posterior and a custom bootstrap.

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
  eleven BestFit trend models.
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
