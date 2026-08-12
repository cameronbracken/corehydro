# Estimation layer: surfacing the fitters in R and Python

Phase 2 of the roadmap in `.claude/plans/surface-all-bestfit-functionality-stateless-graham.md`.
Phase 1 (data and model layer, v0.3.0) shipped as PR #18.

## Context

The core carries four fitted-estimator families, all ported and fixture-validated:
`MaximumLikelihood`, `MaximumAPosteriori`, `BayesianAnalysis` and
`GeneralizedMethodOfMoments`. None of them is callable from R or Python. The only way to fit
anything today is to run a whole analysis, which picks its own estimator and returns a frequency
curve rather than a fit.

The glue already exists but is shaped for fixtures, not for users. `ch_estimation_run_`,
`ch_estimation_bayes_run_`, `ch_estimation_gmm_run_` and `ch_bootstrap_run_` are unexported, they
return a fixed bundle of whatever the fixture corpus asserts, and they hide most of the estimator
surface: `profile_likelihood()`, `parameter_confidence_intervals()`, `total_function_evaluations()`,
the optimization status, the Hessian toggle, the choice of optimizer, the MCMC convergence
diagnostics (R-hat, ESS, acceptance rates, mean log-likelihood), `pareto_k`, the point-estimator
setting, and every sampler knob.

Two behaviours in that glue are acceptable for a fixture harness and not acceptable for a user
API: a single-parameter model gets a covariance matrix of silent zeros, and a failed `estimate()`
raises `stop("... failed for a fixture case")`.

Goal: make all four estimators callable over the phase-1 model specs, with a result object that
composes with the rest of the package, complete reference docs, and one worked example pair. No
new numerics.

## Decisions taken

- **Four separate verbs**, `fit_mle` / `fit_map` / `fit_bayesian` / `fit_gmm`, not one `fit()` with
  a method argument. Their argument sets are disjoint (an optimizer and a Hessian toggle for
  MLE and MAP, a sampler plus a dozen knobs for Bayesian, a strategy and an iteration cap for
  GMM), so a single signature would push every method-specific argument into an untyped control
  list that neither language's tooling can check.
- **A classed fit with base-generic integration.** `corehydro_fit` in R with `print`, `summary`,
  `coef`, `vcov`, `confint` and `logLik`; a `Fit` wrapper class in Python. The fit carries the
  fitted model spec, so it composes back into the phase-1 surface.
- **Posterior draws as a 3-D array**, `[iteration, chain, parameter]`. That is the axis order the
  R `posterior` package's `draws_array` wants and what arviz reshapes from, so neither package
  needs a conversion dependency to be useful.
- **Bootstrap is out of scope.** See "Out of scope" below.
- **Shared C++ runner, thin call sites.** One implementation of a fit in the repo, driven
  identically by four callers. Alternatives considered: widening the fixture glue in place (one
  code path, but every edit risks 4542 pinned assertions) and adding parallel user glue beside it
  (no regression risk, but two implementations that drift and no oracle coverage for the new one).
- **All four already-ported extras are in scope**: profile likelihood and profile confidence
  intervals, optimizer selection and status, fit diagnostics re-cut to take a model, and the GMM
  extras.

## Architecture

### The shared runner

New header `core/include/corehydro/estimation/support/fit_runner.hpp`, built to the contract
`analyses/support/analysis_runner.hpp` established in phase 8. It takes a JSON construct (a model
spec, a datasets map, and the estimator settings) and returns one `FitResult` struct. Four callers
drive it and none owns any fit logic:

- `corehydror/src/estimation.cpp` (cpp11)
- `corehydropy/src/bindings/estimation.cpp` (pybind11)
- `core/tests/` (ctest and the fixture runner)
- `tools/oracle_emitter/` (the dotnet gate)

`FitResult` holds the common block (parameter values and names, log-likelihood, AIC, BIC,
covariance, standard errors, correlation, converged flag, status, function evaluations, the fitted
spec) plus three optional blocks: the profile block (profile likelihood grids and profile
intervals), the Bayesian block (raw chains, thinned output, MAP, posterior mean, acceptance rates,
mean log-likelihood, per-parameter statistics, DIC, WAIC, LOOIC, LOOIC standard error, Pareto-k),
and the GMM block (J-statistic and its p-value, iteration count, convergence flag, optimizer
fallback count).

### One glue entry point per language

`ch_fit_run_(target, construct_json, datasets_json)` in R and `fit_run(...)` in Python, dispatching
on `target` in `{MaximumLikelihood, MaximumAPosteriori, BayesianAnalysis, GMM}`. This is the
`.ch_extended_run` shape from phase 8, so the four verbs are pure R and pure Python: they assemble
a construct and shape a result. A fifth estimator later touches one dispatch arm, not four glue
functions in two languages.

Two auxiliary entry points stay separate because their argument is only known at call time, the
`ch_estimation_bic_` precedent: `ch_fit_quantile_variance_(fit_construct, aep)` and
`ch_fit_diagnostics_(fit_construct)`.

### The fixture glue delegates

`ch_estimation_run_`, `ch_estimation_bayes_run_`, `ch_estimation_gmm_run_` and
`ch_estimation_bic_` keep their exact signatures and return shapes, and compute nothing
themselves: each builds a construct, calls the runner, and picks the fields its dispatcher reads.
Their fixture semantics are preserved verbatim, including the seeded post-fit draw and the
`n_params >= 2` covariance guard, which becomes an explicit guard in the runner rather than an
implicit one. The 4542 pinned assertions must pass unchanged.

### Cost model

Always computed: parameters, log-likelihood, AIC, BIC, status, function evaluations.

On by default, disableable with `hessian = FALSE`: covariance, standard errors, correlation. It is
a numerical differentiation pass, and some models do not need it.

Off by default, requested by argument: profile likelihood and profile confidence intervals, at
`bins * n_params` likelihood evaluations each.

### Error handling

Three changes relative to the fixture glue, applied in the runner so all four callers get them:

1. A model with fewer than two parameters returns an `NA` covariance matrix, `NA` standard errors
   and an `NA` correlation matrix, not zeros. The C# `GetCovarianceMatrix` throws below two
   parameters, so `NA` is the honest report.
2. A failed `estimate()` raises an error naming the estimator, the optimizer and the model family.
3. Every fit result carries `converged`, so callers check rather than infer.

### Statelessness

A fit carries the model spec with the fitted values applied. It round-trips through `save()` and
`pickle`, `model_simulate(f)` and `model_log_likelihood(f)` accept it, and a downstream analysis
rebuilds from it. No C++ object lifetime crosses the language boundary, preserving the object model
phase 1 was built on.

## The R surface

New file `corehydror/R/fit.R`. The first argument is a `corehydro_model`, or a numeric vector plus
a `distribution` name for the convenience path, matching how phase 1 left the analyses.

```r
fit_mle(model, distribution = NULL, optimizer = "NelderMead",
        hessian = TRUE, profile = FALSE, profile_bins = 100)

fit_map(model, distribution = NULL, optimizer = "NelderMead",
        hessian = TRUE, profile = FALSE, profile_bins = 100)

fit_bayesian(model, distribution = NULL, sampler = "DEMCzs", chains = 4,
             iterations = 3000, warmup = NULL, thinning_interval = 1,
             initial_iterations = NULL, output_length = 10000, seed = 12345,
             point_estimator = "MAP", credible_level = 0.9, ...)

fit_gmm(model, strategy = "Iterative", optimizer = "BFGS", max_iterations = 50)
```

`optimizer` takes the six ported methods: `Brent`, `BFGS`, `NelderMead`, `Powell`,
`DifferentialEvolution`, `MultilevelSingleLinkage`. `sampler` takes the four that
`BayesianAnalysis::set_up_sampler` actually constructs: `DEMCz`, `DEMCzs`, `ARWMH`, `NUTS`. The
other three exposed by `mcmc_sample()` (`RWMH`, `HMC`, `SNIS`) are reachable only through raw
sampling, which is what `mcmc_sample()` stays for; `fit_bayesian` rejects them with a message
saying so. `strategy` takes `OneStep`, `TwoStep`, `Iterative`. `point_estimator` takes the
`PointEstimateType` values.

The `...` on `fit_bayesian` carries the sampler-specific knobs the core already has (`jump`,
`jump_threshold`, `snooker_threshold`, `noise`, `scale`, `beta`, `max_tree_depth`). Each is
validated against the chosen sampler, so a knob that sampler ignores is an error rather than a
silent no-op.

`fit_gmm` requires a `model_bulletin17c()`. `IGMMModel` has exactly one implementation, and
passing any other model is an error that says so.

### The fit object

A `corehydro_fit` classed list. Common fields:

| field | contents |
|---|---|
| `method` | the estimator name |
| `parameters` | named numeric vector, names from `ParameterNames` |
| `log_likelihood` | maximum log-likelihood at the fit |
| `aic`, `bic` | information criteria, `bic` at `nobs` |
| `covariance`, `standard_errors`, `correlation` | the Hessian stack, or `NA` |
| `converged`, `status`, `function_evaluations` | optimizer bookkeeping |
| `model` | the fitted `corehydro_model` spec |
| `nobs` | observation count behind `bic` and `logLik` |

`fit_bayesian` adds `draws` (raw chains, `[iteration, chain, parameter]`, dimnames on all three
axes), `posterior` (the thinned flattened output the analyses consume, `output_length x n_params`),
`map`, `posterior_mean`, `acceptance_rates`, `mean_log_likelihood`, `summary` (a data frame with
one row per parameter: mean, median, sd, lower, upper, rhat, ess), and `dic`, `waic`, `looic`,
`looic_se`, `pareto_k`.

`fit_gmm` adds `j_stat`, `j_stat_pval`, `gmm_iterations`, `converged_within_tolerance`,
`optimizer_fallback_count`.

`profile = TRUE` adds `profile`, a list of two-column matrices, one per parameter.

### S3 methods

`print`, `summary`, `coef`, `vcov`, `confint`, and `logLik` carrying `df` and `nobs` attributes so
base `AIC()` and `BIC()` work with no method of their own. `confint(f, level = 0.9)` returns
profile likelihood intervals for MLE and MAP fits and credible intervals for a Bayesian fit;
requesting profile intervals from a fit built without `profile = TRUE` computes them on demand
rather than erroring.

### Two more verbs

```r
fit_diagnostics(f)              # Cook's distance, observation influence, leverage,
                                # Pareto-k, prior influence
quantile_variance(f, aep)       # B17C delta-method Var(Q_p), GMM fits only
```

`fit_diagnostics()` is what makes the diagnostics layer reachable for censored and nonstationary
models. The existing `estimation_diagnostics(data, distribution, ...)` export is unchanged and
keeps working.

Because a fit is a stateless spec, `fit_diagnostics()`, `quantile_variance()` and an on-demand
`confint()` rebuild the model from the fit and re-run the estimator. This is the
`ch_estimation_bic_` precedent: a deterministic optimizer reproduces the identical fit, and a
seeded Bayesian run reproduces the identical chain. Where the rebuild would be wasteful the caller
avoids it by asking for the value up front (`profile = TRUE`). A test asserts that on-demand and
up-front profile intervals are identical.

## The Python surface

New module `corehydropy/src/corehydropy/fit.py`, mirroring R argument for argument with
`snake_case` kwargs and type hints:

```python
fit_mle(model, distribution=None, optimizer="NelderMead",
        hessian=True, profile=False, profile_bins=100) -> Fit
fit_map(...) -> Fit
fit_bayesian(model, distribution=None, sampler="DEMCzs", chains=4,
             iterations=3000, warmup=None, thinning_interval=1,
             initial_iterations=None, output_length=10000, seed=12345,
             point_estimator="MAP", credible_level=0.9, **sampler_kwargs) -> Fit
fit_gmm(model, strategy="Iterative", optimizer="BFGS", max_iterations=50) -> Fit
```

`Fit` is a light wrapper class in the shape `Distribution` already established in
`distributions.py`. Properties: `parameters` (dict), `parameter_names`, `covariance`,
`standard_errors`, `correlation`, `draws`, `posterior` (numpy arrays), `log_likelihood`, `aic`,
`bic`, `converged`, `status`, `function_evaluations`, `model`, and the Bayesian and GMM blocks
under the same names as R. Methods: `confint(level=0.9)`, `summary()`, `diagnostics()`,
`quantile_variance(aep)`, `to_model()`, `to_json()`, `__repr__`.

`draws` carries the same `(iteration, chain, parameter)` axis order as R. No new dependency:
numpy is already required and nothing pulls in pandas.

## Fixtures and the oracle gate

Oracle values live only in `fixtures/*.json` and the dotnet gate must reproduce them. New
`model_estimation` cases cover the surface that has never been asserted:

- Profile confidence intervals and a profile likelihood grid, for an MLE and a MAP fit.
- `total_function_evaluations` and the optimization status.
- R-hat, ESS and per-chain acceptance rates off a seeded chain, plus `mean_log_likelihood`.
- The per-parameter posterior summary (mean, median, sd, credible bounds).
- `pareto_k`.
- The GMM extras that lack coverage: iteration count, convergence flag, optimizer fallback count,
  and `quantile_variance` at a stated AEP.
- One fit under each of the six optimizers on the same model, pinning that the optimizer argument
  reaches the estimator.

All of it is ported C#, so `tools/oracle_emitter/` drives the real library for each. The gate must
grow by the new cases with 0 failed, no new skips, and no loosened tolerance. If a value will not
reproduce, record why in `docs/upstream-csharp-issues.md` and assert the structural invariant
instead, following the phase 9a and 10 precedent.

Cross-language identity gets a seeded digest case in the phase-1 shape: one censored, nonstationary
model fit by `fit_bayesian` must agree bit for bit between R and Python.

All three runners (`core/tests/test_fixtures.cpp`, `corehydror/tests/testthat/test-fixtures.R`,
`corehydropy/tests/test_fixtures.py`) get the new assertion arms wired in following the existing
dispatch pattern.

## Behavioural tests

`corehydror/tests/testthat/test-fit.R` and `corehydropy/tests/test_fit.py`:

- Argument validation and error messages: unknown optimizer, unknown sampler, a sampler knob that
  the chosen sampler ignores, `fit_gmm` on a non-B17C model, a numeric first argument with no
  `distribution`.
- The single-parameter `NA` covariance, and that it is `NA` rather than zero.
- The failed-fit error names the estimator and the optimizer.
- Base `AIC(f)` equals `f$aic` and `BIC(f)` equals `f$bic`.
- `save()` and `pickle` round-trip a fit and its model.
- `model_simulate(f, n, seed)` equals `model_simulate(f$model, n, seed)`.
- `confint()` on a fit built without `profile = TRUE` computes intervals on demand.

## Docs and examples

Every new export appears in both `corehydror/_pkgdown.yml` and the `quartodoc.sections` of
`site/_quarto.yml`, under a new "Estimation" section in each. pkgdown errors on a missing
reference-index entry, so this is a hard contract. Every R export gets roxygen with a runnable
`@examples`; every Python export gets numpydoc.

One new example pair, `site/examples/25-estimation-methods/{python.ipynb, r.qmd}`: the same
censored record fit by MLE, MAP and Bayesian, compared on AIC, BIC and DIC, with profile intervals
set against credible intervals, ending in the executable reproduction check the other eleven pairs
end in. The Python notebook is committed with outputs; the R twin needs `site/_freeze/` regenerated
and committed. Update `site/examples/index.qmd` and the coverage page.

## Verification

Run from the repo root, in this order. Every command must pass before the phase is called done.

```bash
# 1. C++ core: 79/79 currently, must not regress
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build

# 2. Oracle gate: 4542 reproduced / 0 failed / 11 skipped currently.
#    Must grow by the new cases with 0 failures and no new skips.
python3 tools/verify_oracles.py

# 3. R: 4446 / 0 currently
Rscript -e 'cpp11::cpp_register("corehydror")'
R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'

# 4. Python: 884 currently
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q

# 5. Package checks
R CMD build corehydror && R CMD check --as-cran corehydror_*.tar.gz
pixi run docs && pixi run docs-serve
```

End-to-end check that the phase delivered its point, run in both languages and compared:

```r
d <- analysis_data(exact = peaks, mgbt_low_outliers = TRUE)
m <- model_univariate("LogPearsonTypeIII", d)

mle   <- fit_mle(m, optimizer = "BFGS", profile = TRUE)
bayes <- fit_bayesian(m, sampler = "DEMCz", iterations = 500, seed = 12345)

coef(mle); confint(mle, level = 0.9); AIC(mle)
bayes$summary                    # rhat and ess per parameter
dim(bayes$draws)                 # iterations x chains x parameters
```

The identical Python call must return the same numbers to every digit.

## Out of scope

**Bootstrap over model specs.** The roadmap listed `bootstrap_model` in this phase with no new
C++. That is wrong. `Bootstrap<TData>` is a delegate-driven engine and its model registry has one
entry, built to replay `Test_Bootstrap.cs`. Bootstrapping an arbitrary model spec means writing new
resample, fit and statistic delegates over `ModelBase`. That is a corehydro invention with no C#
counterpart, so the dotnet gate could not pin it and it would ship validated only against itself.
`bootstrap_analysis()` already covers parametric bootstrap confidence intervals for the thirteen
distributions implementing `IBootstrappable`. Revisit as its own phase, with the validation problem
stated up front.

**Everything in roadmap phases 3 through 6**: composite distribution families, copulas and
multivariate distributions in `distribution()`, the Numerics toolbox, the math layer, and the final
docs sweep.

## Notes and constraints

- Branch off `main` once PR #18 merges. Do not push or open a PR without being asked. Commits are
  GPG-signed as `Cam Bracken <cameron.bracken@pm.me>` with no `Co-Authored-By` trailer.
- `core/` and `fixtures/` are vendored into both packages as subtree symlinks. Edits are live
  through the symlink.
- No new external C++ dependency in the core, and no new R or Python dependency.
- After editing any `corehydror/src/*.cpp`, re-run `cpp11::cpp_register("corehydror")`. The runner
  adds no member to an existing class, so no preclean is required, but run
  `R CMD INSTALL --preclean` anyway before the final verification pass.
- Bump to 0.4.0 and add a `CHANGELOG.md` entry when the phase lands.
