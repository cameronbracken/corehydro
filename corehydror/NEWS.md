# corehydror 0.10.0

The rest of the Numerics optimization layer. `optim_minimize()` grew from six methods to
fourteen, gained an optional analytic gradient and its first constrained method, and the
dynamic-programming trio arrives as `shortest_path()`. See `CHANGELOG.md` at the repository root
for the full account.

## New features

* Four new global optimizers on `optim_minimize()` / `optim_maximize()`: `"particle_swarm"`,
  `"sce"` (shuffled complex evolution), `"simulated_annealing"` and `"multi_start"`. All four are
  seeded, and each takes its own settings through `control` -- `population_size`; `complexes` /
  `cce_iterations` / `tolerance_steps`; `initial_temperature` / `min_temperature` /
  `cooling_rate` / `update_cycles` / `temperature_cycles` / `tolerance_steps`; and `local_method` /
  `local_absolute_tolerance` / `local_relative_tolerance` / `polish`.
* Three new local optimizers: `"adam"`, `"gradient_descent"` and `"golden_section"` (the last
  one-dimensional, like `"brent"`). ADAM and gradient descent take `alpha` through `control`, and
  ADAM the decay factors `beta1` and `beta2`.
* A `gradient` argument on both verbs: a function returning one partial derivative per parameter,
  accepted by `"adam"` and `"gradient_descent"`. Supplying it cut a five-dimensional gradient
  descent from 103,141 objective calls to 8,596.
* Constrained optimization through `method = "augmented_lagrange"`, with the new
  `optim_constraint()` constructor and the two new arguments `constraints` and `inner`. The result
  carries `multipliers`, the three Lagrange multiplier vectors.
* `local_method` is now a `control` setting on `"mlsl"` as well as on the new `"multi_start"`.
* `shortest_path()` -- Dijkstra shortest paths over a graph given as parallel `from`, `to` and
  `weight` vectors plus a set of destinations, returning the next node, edge and cost for every
  node, backed by the ported `BinaryHeap` / `Dijkstra` / `Network`. Node indices are
  0-based, matching the C# literals and the Python twin.
* A worked example pair, 19, covering the seeded global searches, the constrained problem and its
  shadow price, the analytic gradient, and `shortest_path()`, ending in an executable reproduction
  check.

## Changes to results

* `optim_maximize()` now rejects `method = "augmented_lagrange"` by name. The upstream C# class
  always minimizes internally, so a maximize request returned the constrained minimum labelled
  `Success`. Negate the objective and call `optim_minimize()`; the error message says so.
* A `control` setting that belongs to a different method is now an error naming both, rather than
  a silent no-op. Every method's settings are validated against one table.

## Bug fixes

* `shortest_path()` rejects a `node_count` smaller than the graph rather than reading past the end
  of its arrays. The unguarded core code was undefined behavior, and could terminate the session
  instead of raising; C# raises an `IndexOutOfRangeException` there, and the port now does the
  equivalent.

## Notes

* A seeded `"particle_swarm"` or `"sce"` run reproduces bit-for-bit between corehydror and
  corehydropy, but not necessarily against the C# library: both branch on comparisons between
  accumulated sums, and a compiler that emits fused multiply-add takes a different search path.
  The difference is measured and documented rather than hidden -- `"simulated_annealing"` and
  `"multi_start"` reproduce C# exactly, and the deterministic methods have no PRNG at all.
* `method = "multi_start"` can report a value its own reported parameters do not produce, because
  upstream's polish step clamps the best point onto the bounds after recording its fitness. The
  real C# library returns the same numbers; see `docs/upstream-csharp-issues.md`.

# corehydror 0.9.0

The last unported slice of Numerics that is not a distribution, model, or estimator: root
finding, integration, ODE solving, spline/polynomial interpolation, linear algebra, and two small
special-function/general-function groups, all callable against a user-written R function. See
`CHANGELOG.md` at the repository root for the full account.

## New features

* `root_find()` gains a `method` option (`"brent"`, the default, `"bisection"`, `"secant"`,
  `"newton"`) over the existing bracketing interface, and `root_find_system()` solves a
  vector-valued system with multivariate Newton-Raphson given `f`, its `jacobian`, and a
  `first_guess`.
* `quadrature()` gains a `method` option covering ten deterministic rules (`"gauss_kronrod"`, the
  default adaptive rule, plus `"simpsons"`, `"trapezoidal"`, `"adaptive_simpsons"`,
  `"gauss_lobatto"`, `"gauss_legendre"`, `"gauss_legendre20"`, `"simpsons_fixed"`,
  `"trapezoidal_fixed"`, `"midpoint"`). `quadrature_2d()` integrates over a rectangle, and
  `quadrature_nd()` integrates over an arbitrary-dimensional box with three seeded Monte Carlo
  families (`"monte_carlo"`, `"miser"`, `"vegas"`, the last with rare-event configuration).
* `ode_solve()` -- the ported `RungeKutta` solver (`"rk4"`, `"rk2"`, `"rkf"`, `"cash_karp"`) for
  `dy/dt = f(t, y)` from an initial value over a fixed number of time steps.
* `interpolate()` gains `method = "cubic_spline"` and `method = "polynomial"` alongside the
  existing linear method; the transform and extrapolation arguments remain linear-only, matching
  the C# `CubicSpline`/`Polynomial` classes, which have neither.
* Three small toolbox groups: `qr_decomposition()`, `qr_solve()`, and `gauss_jordan()` (linear
  algebra); `debye()` and `polynomial_eval()` (special functions, the last with three literal
  conventions -- standard, reverse, reverse-unit); and `univariate_function()`, evaluating the
  ported `LinearFunction` and `PowerFunction` (forward, inverse, and the optional
  normally-distributed noise path).
* A worked example pair, 18, walks all of the above ending in an executable reproduction check.

## Notes

* `TabularFunction`, the third `IUnivariateFunction` implementation, depends on the still-unported
  Paired Data subsystem and is not exposed; it is deferred, not silently dropped, and is tracked
  for a later release.
* Two of the three seeded Monte Carlo integrators in `quadrature_nd()` carry a measured, honestly
  documented rounding difference at the current shipped build: `"monte_carlo"` reproduces
  bit-for-bit against the real C# library and across R, Python, and C++; `"miser"` measured 1 ULP
  off C#, and `"vegas"` 2-3 ULP between R and Python, from floating-point contraction in the
  variance/chi-squared accumulation. No tolerance was loosened and no oracle was skipped to hide
  this; the affected fixture cases assert the seeded evaluation counts and status instead of the
  integral value, with the measurements recorded alongside them.

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
