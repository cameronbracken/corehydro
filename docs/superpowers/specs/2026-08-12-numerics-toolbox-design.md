# Numerics Toolbox: Design

Phase 4 of the roadmap "Surface the full core in R and Python"
(`~/.claude/plans/surface-all-bestfit-functionality-stateless-graham.md`). Phase 1 delivered the
data and model layer (PR #18, v0.3.0), phase 2 the estimation layer (PR #19, v0.4.0), and phase 3
the distribution layer (PR #20, v0.5.0).

## Goal

Make the Numerics utility layer reachable from R and Python: goodness of fit, correlation and
autocorrelation, summary statistics, histograms, interpolation, linear regression, Sobol
sequences, stratification, joint probability, link functions, trend evaluation, and the six
ported optimizers. Every one of these is ported and tested in C++ today and invisible to users of
either package.

The phase adds glue, fixtures, docs, and examples. It adds numerical code exactly once:
`Numerics/Data/Statistics/Autocorrelation.cs` turns out to be unported and unrecorded, so this
phase ports it (see the autocorrelation row below).

## What the core supports

Read from the headers, not inferred.

| Group | Header | Available |
|---|---|---|
| Goodness of fit | `numerics/data/goodness_of_fit.hpp` | AIC, AICc, BIC, AIC weights, RMSE in three forms (two series with a degrees-of-freedom correction, a series against a fitted distribution at Weibull plotting positions, and a series against a fitted distribution at supplied plotting positions), RMSE weights, MSE, MAE, MAPE, sMAPE, NSE, log NSE, KGE, modified KGE, PBIAS, RSR, Pearson, R squared, index of agreement (plain, modified, refined), volumetric efficiency, accuracy, confusion matrix, precision, recall, F1, specificity, balanced accuracy, Kolmogorov-Smirnov D, Anderson-Darling A squared, chi squared |
| Correlation | `numerics/data/correlation.hpp` | Pearson, Spearman, Kendall's tau |
| Autocorrelation and spectra | `numerics/math/fourier/fourier.hpp`, plus a new `numerics/data/autocorrelation.hpp` | Ported today: `fft`, `real_fft`, cross correlation, and `Fourier::autocorrelation`, which returns lag and ACF pairs only. Upstream's `Data/Statistics/Autocorrelation.cs` (autocovariance, partial autocorrelation, and `CorrelationConfidenceInterval`) is unported and undocumented, so this phase ports it as a faithful new header pinned against `Test_Autocorrelation.cs` |
| Summary statistics | `numerics/data/statistics.hpp` | mean, variance, standard deviation, maximum, product moments, linear moments, ranks, percentile |
| Streaming statistics | `numerics/data/running_statistics.hpp`, `running_covariance_matrix.hpp` | count, min, max, mean, sample and population variance, standard deviation, coefficient of variation, skewness, kurtosis, a static merge of two accumulators; sample and population covariance and correlation matrices |
| Histogram | `numerics/data/histogram.hpp` | Rice-rule and explicit-bin construction, bin bounds, midpoints and frequencies, mean, median, mode, standard deviation |
| Interpolation | `numerics/data/interpolation/` | `Linear` and `Bilinear` over the `Interpolater` base: x and y transforms (None, Logarithmic, NormalZ), ascending or descending sort order, sequential, bisection and hunt search, and `Linear::extrapolate` |
| Linear regression | `numerics/data/regression/linear_regression.hpp` | SVD fit with or without intercept, parameters and standard errors, covariance, residuals, standard error, degrees of freedom, R squared and adjusted R squared, `predict`, `prediction_intervals` |
| Sampling utilities | `numerics/sampling/` | `SobolSequence` (up to 21201 dimensions, direction numbers already installed at `corehydror/inst/extdata/new-joe-kuo-6.21201`), `Stratify::XValues` over one or many `StratificationOptions`, `LatinHypercube` (already exported) |
| Joint probability | `numerics/data/probability.hpp` | independent, perfectly positive and perfectly negative joint probability over a probability vector, plus the indicator form (a 0/1 flag per component) which additionally takes a correlation matrix and routes to the HPCM path |
| Link functions | `numerics/functions/`, `models/link_functions/` | the 7 Numerics links (identity, log, logit, probit, complementary log-log, Yeo-Johnson, Fisher z) and the 6 BestFit links (asinh, SES, log SES, log asinh, centered, Yeo-Johnson), each with `link`, `inverse_link` and `d_link` |
| Trend functions | `models/trend_functions/` | the 11 trend model types with `predict(index)` and their parameters |
| Optimizers | `numerics/math/optimization/` | DifferentialEvolution, BFGS, Powell, MLSL, NelderMead, BrentSearch, each over an objective `std::function<double(const std::vector<double>&)>` |

## Architecture

### Two runners, split at the callback line

Phases 2 and 3 established one runner driven by four callers: the cpp11 glue, the pybind11 glue,
the C++ fixture runner, and the dotnet oracle emitter. Phase 4 keeps that shape but splits it in
two, because one group of the toolbox cannot be described by a serializable spec.

**`core/include/corehydro/numerics/support/toolbox_runner.hpp`** handles the ten data and utility
groups. Signature:

```cpp
ToolboxResult run_toolbox(const std::string& group, const std::string& method,
                          const std::vector<std::vector<double>>& data,
                          const std::string& options_json);

struct ToolboxResult {
    std::vector<double> values;
    std::vector<std::string> names;   // labels when the method returns a named set
    std::vector<int> dims;            // rows and columns when the method returns a matrix
    std::string spec;                 // a child spec when the method returns an object
};
```

Bulk data travels as native `double` vectors, not JSON. A goodness-of-fit call carries two series
of arbitrary length, and paying a JSON parse for them would be pointless. Everything else (method
options, enum names, transforms, bin counts, seeds) travels in `options_json`.

Where a method needs a distribution (the three goodness-of-fit tests) or a trend object, the
runner builds it from the existing spec grammar: `dist_spec.hpp` for distributions, and the
`trends` block `model_spec.hpp:288` already parses for trends. Link functions have no spec today
(`model_spec.hpp` carries only the `use_log_link_for_*` toggles), so this phase adds a small
`link` block, owned by the toolbox runner and shaped like the others.

**`core/include/corehydro/numerics/support/optimizer_runner.hpp`** handles the six optimizers:

```cpp
OptimResult run_optimizer(const std::string& spec_json, const Objective& objective);
```

The objective is a live function, so this runner has three callers rather than four. The emitter
supplies a C# delegate against the real Numerics optimizers instead of replaying a spec, and the
fixture path names one of the built-in analytic objectives described below.

### Glue

Two new entry points per language: `ch_toolbox_run_` and `ch_optim_run_` in cpp11, `toolbox_run`
and `optim_run` in pybind11. The existing unexported `ch_sobol_generate_` and `ch_sobol_skip_to_`
fold into the toolbox runner so the Sobol path has one implementation rather than two.

## The R surface

New files `corehydror/R/gof.R`, `R/toolbox.R` and `R/optim.R`, plus additions to `R/stats.R`.
Metric groups that are large and homogeneous get a selector verb; calls whose shape genuinely
differs stay separate.

```r
# Fit and error metrics
goodness_of_fit(observed, modeled, metrics = "all", k = 0)   # k corrects RMSE's denominator
classification_metrics(observed, modeled, threshold)
gof_test(x, d, test = c("ks", "ad", "chi_squared"))     # d is a corehydro_dist
gof_rmse(x, d, plotting_positions = NULL)               # RMSE of a fitted distribution;
                                                        # NULL uses Weibull positions
aic(k, log_likelihood); aicc(n, k, log_likelihood); bic(n, k, log_likelihood)
aic_weights(aic); rmse_weights(rmse)

# Correlation, autocorrelation, spectra
correlation(x, y, method = c("pearson", "spearman", "kendall"))
autocorrelation(x, max_lag = NULL, type = c("correlation", "covariance", "partial"),
                confidence_level = 0.95)
cross_correlation(x, y)
dft(x, inverse = FALSE); dft_real(x, inverse = FALSE)

# Summary and streaming statistics
summary_statistics(x)
product_moments(x); l_moments(x); ranks(x); percentile(x, probs)
running_statistics(x, state = NULL); running_covariance(x, state = NULL)

# Histogram and interpolation
histogram(x, bins = NULL, lower = NULL, upper = NULL)
interpolate(x, y, xout, x_transform = "none", y_transform = "none",
            sort_order = "ascending", extrapolate = FALSE)
interpolate_2d(x1, x2, y, x1out, x2out, x1_transform = "none",
               x2_transform = "none", y_transform = "none")

# Linear regression
linear_regression(x, y, intercept = TRUE)
predict(fit, newdata, interval = FALSE, level = 0.90)

# Sampling utilities
sobol_sequence(n, dimension = 1, skip = 0)
stratify(lower, upper, bins, probability = FALSE, logarithmic = FALSE)

# Joint probability
joint_probability(p, dependency = c("independent", "positive", "negative", "correlation"),
                  indicators = NULL, correlation = NULL)

# Link and trend functions
link_function(type, ...); link_names()
link(l, x); link_inverse(l, eta); link_derivative(l, x)
trend_predict(tr, index); trend_parameters(tr); trend_names()

# Optimizers
optim_minimize(objective, lower, upper, initial = NULL,
               method = c("de", "bfgs", "powell", "mlsl", "nelder_mead", "brent"),
               seed = NULL, control = list())
optim_maximize(objective, lower, upper, initial = NULL, method = ..., seed = NULL,
               control = list())
```

Three naming decisions and their reasons:

- `dft()` and `dft_real()` rather than `fft()`, which would mask `stats::fft`.
- `optim_minimize()` and `optim_maximize()` rather than `optimize()`, which would mask
  `base::optimize`. The pair mirrors the C# `Optimizer.Minimize` and `Maximize`.
- `correlation()`, `interpolate()` and `stratify()` mask nothing in base R or the recommended
  packages. `histogram()` shares a name with `lattice::histogram`, a recommended package that is
  not attached by default, so the collision is a documented note rather than a rename. The
  alternative, `hist_bins()`, reads worse for the primary verb of a group.

`link_function()` returns a `corehydro_link` spec list. Six of the thirteen links take
construction parameters (`ASinHLink(gamma0, scale, epsilon, delta)`, `SESLink(a,
use_adaptive_lambda, parent_indicator)`, `LogSESLink(sigma0, a, lambda)`, `LogASinHLink(sigma0,
log_scale, epsilon, delta)`, the Yeo-Johnson lambda, and `CenteredLink`, which wraps an inner
link), so the spec carries named parameters and, for the centered link, a nested child spec.

`trend_predict()` takes the `corehydro_trend` object phase 1 already exports, so trend evaluation
needs no new constructor.

`running_statistics()` and `running_covariance()` keep the accumulator's state in the return
value: a classed list carrying the count, the four raw moments, and the min and max (the running
covariance carries the mean vector and the scatter matrix). Passing that list back in as `state`
continues the accumulation. The streaming property survives without a C++ object crossing the
boundary, and the state serializes with `save()` and `pickle` like every other object in the
package.

## The Python surface

Mirrors R exactly: same function names, same argument names, same defaults, in
`corehydropy/src/corehydropy/gof.py`, `toolbox.py` and `optim.py`, with the statistics additions in
`stats.py`. Results are dicts or numpy arrays rather than named vectors; `linear_regression()`
returns a light `LinearRegressionResult` wrapper matching how `Fit` and `Distribution` already
wrap stateless calls. No pandas dependency.

## The optimizer callback contract

The objective is an R closure or a Python callable. Everything else about the run travels in the
spec.

**Error trapping.** The wrapper catches whatever the callback throws (`cpp11::unwind_exception` in
R, `pybind11::error_already_set` in Python), stores it in an `std::exception_ptr`, latches an
aborted flag, and returns positive infinity for a minimize or negative infinity for a maximize.
Every later evaluation short-circuits on the flag without re-entering the host, so the run winds
down in a few evaluations rather than the full budget. The wrapper rethrows the stored exception
once `minimize()` returns.

This trap is required, not defensive. `Optimizer::minimize()` ends in a catch-all that swallows
any exception and records `Status = Failure`. Without the trap, an R error inside the objective
leaves R unwound halfway, and a Python error stays set but never raised.

**Invalid returns.** A return that is not a finite-or-NaN scalar double (`NULL`, `NA`, a vector, a
string) is an error routed through the same abort path, with a message naming the parameter vector
that produced it.

**Failure reporting.** `report_failure` defaults to `TRUE`, so a configuration error surfaces as an
R condition or a Python exception. `control$report_failure = FALSE` restores the C# behavior of
keeping the best parameter set found so far.

**Threads.** None of the six optimizers spawns a thread, so there is no GIL or R reentrancy hazard
beyond the single call.

**Reproducibility.** The RNG stays in C++. A seeded DE or MLSL run over the same objective is
bit-identical between R and Python, and a digest fixture proves it.

## Fixtures and the oracle gate

A new `toolbox` fixture kind under `fixtures/toolbox/`, wired into all four runners the way the
`analysis`, `model_estimation` and `data_utility` kinds already are. Oracle values come from the
upstream C# tests that exist for every group: `Test_GoodnessOfFit`, `Test_Correlation`,
`Test_Autocorrelation`, `Test_Histogram`, `Test_PairedDataInterpolation`, `Test_LinearRegression`,
`Test_SobolSequence`, `Test_Stratification`, and the optimizer tests under
`Mathematics/Optimization`. Every value is pinned by `python3 tools/verify_oracles.py` against the
real libraries.

`summary_statistics()` returns an assembled named vector rather than a single C# call. Each
component is pinned individually against the C# `RunningStatistics` and `Statistics` methods it
comes from.

Optimizer oracles need an objective that exists on both sides of the gate, so `optimizer_runner.hpp`
carries a registry of named analytic objectives transcribed from the upstream
`Test_Numerics/Mathematics/Optimization/TestFunctions.cs`. The six ported optimizers' tests call
fourteen of them: `FX`, `FXYZ`, `DeJong`, `Rosenbrock`, `SumOfPowerFunctions`, `McCormick`,
`Matyas`, `Booth`, `Beale`, `ThreeHumpCamel`, `Rastrigin`, `GoldsteinPrice`, `Eggholder`, and
`Ackley`. A fixture names an objective, and the emitter drives the same function as a C# delegate. The registry is fixture and oracle infrastructure, not a user-facing surface.

The user-callback path itself is covered by behavioral tests, not by the oracle gate. The gate
cannot run an R closure, and this spec does not pretend otherwise.

## Testing

- ctest: `core/tests/test_toolbox_runner.cpp` and `test_optimizer_runner.cpp` for dispatch,
  option parsing, and the abort path with a throwing objective.
- testthat: `test-gof.R`, `test-toolbox.R`, `test-optim.R` for argument validation, error
  messages, spec round-tripping, and error propagation out of a failing objective.
- pytest: the same three files as twins.
- One cross-language digest fixture proving a seeded DE run, a Sobol block, and a stratification
  agree byte for byte between R and Python.

## Docs and release

Every new export goes into both `corehydror/_pkgdown.yml` and the `quartodoc.sections` in
`site/_quarto.yml`, under new sections: Goodness of fit, Statistics, Interpolation and regression,
Sampling utilities, and Optimizers. Every R export gets roxygen with a runnable `@examples`, and
every Python export gets a docstring with a runnable example.

Two worked example pairs (a Python notebook and an R Quarto twin, each ending in an executable
reproduction check):

1. Evaluating and comparing fitted models: fit candidates, rank them with `goodness_of_fit()`,
   AIC weights, and the KS and AD tests.
2. Optimizing a user-written objective: the callback path with DE and BFGS, showing a seeded run
   reproducing across both languages.

Branch `surface-numerics-toolbox`, version bump to 0.6.0 with CHANGELOG and NEWS entries.
`R CMD check --as-cran` is expected to hold at its current three NOTEs with no WARNING.

## Sequencing

The phase adds roughly 35 exports across fourteen core groups, so the implementation plan splits
into tasks that land independently, each carrying its own fixtures, tests, and reference docs:

1. `toolbox_runner.hpp` plus the two glue entry points and the `toolbox` fixture kind in all four
   runners, proven end to end with the correlation group. Everything else depends on this.
2. Goodness of fit and the information criteria.
3. The `Autocorrelation.cs` port, spectra, summary and streaming statistics.
4. Histogram, interpolation, linear regression.
5. Sobol, stratification, joint probability.
6. Link functions and trend evaluation.
7. `optimizer_runner.hpp`, the objective registry, and the two optimizer verbs.
8. Docs, the two example pairs, and the release.

Tasks 2 through 6 are independent of each other. Task 7 depends only on task 1 for the glue
pattern, not for the runner itself.

## Out of scope

- **Seven unported optimizers.** Upstream ships Adam, GradientDescent, GoldenSection, MultiStart,
  ParticleSwarm, ShuffledComplexEvolution, and SimulatedAnnealing. Porting them is new math, not
  new glue. They stay out and get recorded as a severance.
- **The math layer.** Matrix decompositions, special functions, integration, root finding, and
  numerical derivatives are phase 5. `dft()` and `dft_real()` are the one exception, because
  upstream files Fourier under Data and Statistics and tests it there.
- **Correlation matrices.** Upstream's `Correlation.Pearson(double[,])` and
  `Correlation.Spearman(double[,])` matrix overloads are unported. The vector forms are all the
  port carries, so the surface offers only those.
- **`Interpolater` search tuning.** `set_search_start()` and `use_smart_search()` are
  performance controls on a stateful object. The stateless verbs use the defaults.
