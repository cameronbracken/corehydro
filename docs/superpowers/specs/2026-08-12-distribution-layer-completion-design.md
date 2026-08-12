# Distribution Layer Completion: Design

Phase 3 of the roadmap "Surface the full core in R and Python"
(`~/.claude/plans/surface-all-bestfit-functionality-stateless-graham.md`). Phase 1 delivered the
data and model layer (PR #18, v0.3.0); phase 2 delivered the estimation layer (PR #19, v0.4.0).

## Goal

Make the composite univariate families, the seven bivariate copulas, and the five multivariate
distributions reachable from R and Python. All three groups are ported, fixture-validated, and
invisible to users today: their only glue is roughly thirty bespoke functions that the fixture
runners call and nothing else.

This phase adds no numerical code. Every verb it exposes already exists in the core.

## What the core supports

Read from the headers, not inferred. The table drives what the phase can expose.

| Group | Available | Absent upstream |
|---|---|---|
| Truncated, Mixture, CompetingRisks, Empirical, KernelDensity | `pdf`, `log_pdf`, `cdf`, `inverse_cdf`, seeded `generate_random_values`, the eight moments, `log_likelihood` | linear moments (none declares `ILinearMomentEstimation`); a general `estimate()` (Mixture and CompetingRisks implement MLE only, already reachable through `model_mixture()` and `fit_mle()`) |
| The seven copulas | `pdf`, `log_pdf`, `cdf`, `inverse_cdf`, upper and lower tail dependence, or/and joint exceedance, seeded random pairs, theta bounds, `pseudo_log_likelihood` / `ifm_log_likelihood` / `log_likelihood`, MPL / IFM / MLE estimation | a `tau()` accessor. Kendall's tau is a fit method, not a property, and only Clayton, Gumbel, and AliMikhailHaq implement `set_theta_from_tau` |
| MultivariateNormal | `pdf`, `log_pdf`, `cdf`, `interval` (rectangle probability), `marginal`, `conditional`, `mahalanobis`, `inverse_cdf`, seeded random and Latin hypercube draws | |
| MultivariateStudentT | `pdf`, `log_pdf`, `cdf`, `mahalanobis`, `inverse_cdf`, seeded random and Latin hypercube draws | `marginal`, `conditional`, `interval`. Adding them would be new math with no C# counterpart, so they stay out |
| Dirichlet, Multinomial | `pdf`, `log_pdf`, seeded draws, mean, variance, covariance | `cdf` throws `std::logic_error` by design |
| BivariateEmpirical | `cdf` (bilinear interpolation) | `pdf` returns NaN, an upstream stub |

Source lines for each claim live in the headers under
`core/include/corehydro/numerics/distributions/`.

## Architecture

### One spec grammar

The fixture `construct` blocks are already a nested grammar. `fixtures/distributions/univariate/
mixture.json` carries `{"components": [{"target": "Normal", "params": [10, 2]}, ...], "weights":
[...]}`; `truncated_distribution.json` carries `{"base": {...}, "bounds": [1.1, 2.11]}`. This
phase promotes that convention to the canonical distribution spec, as phase 2 promoted the fit
construct.

The fixtures spell the keys `target` and `params`; `model_spec.hpp:125` spells them `family` and
`parameters`. Users and documentation get `family` and `parameters`; the builder accepts both
spellings, documented in `fixtures/README.md`. A mechanical key rename across every fixture file
would touch 4623 pinned assertions for no gain.

### One runner, four callers

Two new header-only files, both additive:

- `core/include/corehydro/numerics/distributions/support/dist_spec.hpp` builds objects from JSON:
  `build_univariate(spec)` (flat families and the five composites, nested to any depth),
  `build_copula(spec)`, `build_multivariate(spec)`.
- `core/include/corehydro/numerics/distributions/support/dist_runner.hpp` holds the entry points,
  shaped like `run_fit()` in `estimation/support/fit_runner.hpp`: `run_dist(spec_json, method,
  args)`, `run_copula(...)`, `run_mvdist(...)`, each returning
  `DistResult { std::vector<double> values; std::string spec; }`. The `spec` field carries a child
  object back when a method returns a distribution, so `marginal` and `conditional` compose without
  a C++ object crossing the language boundary.

Four callers drive the runner and none owns evaluation logic:

1. the cpp11 glue,
2. the pybind11 glue,
3. `core/tests/test_fixtures.cpp`, whose `build_composite`, `build_multivariate`, and
   `build_copula` are rewritten to delegate,
4. `tools/oracle_emitter/Program.cs`, whose `BuildComposite`, `BuildMultivariate`, and
   `BuildCopula` read the same grammar to build the C# objects.

A fixture case, an oracle replay, and a user's `dist_mixture()` call are then one code path.

The runner must serve a whole seeded draw vector from a single call. The present `*_seq` glue
exists precisely so a seeded run is not restarted between assertions, and that property has to
survive the rewrite.

### Glue: three functions added per language, roughly thirty deleted

`ch_dist_spec_run_`, `ch_copula_run_`, and `ch_mvdist_run_`, with the `dist_spec_run` /
`copula_run` / `mvdist_run` twins in pybind11, replace the whole bespoke set: `ch_trunc_*`,
`ch_emp_*`, `ch_kde_*`, `ch_mix_*`, `ch_cr_*`, `ch_cop_*`, `ch_dirichlet_*`, `ch_multinomial_*`,
`ch_bve_*`, `ch_mvn_*`, and `ch_mvt_*`. The R and Python fixture runners switch to the new entry
points, which is what frees the old functions.

The flat `ch_dist_*` verbs stay. They work, and `distribution("Normal", c(0, 1))` should not pay a
JSON parse.

`build_spec_distribution` in `model_spec.hpp` delegates to the new builder. A model prior can then
be a truncated or mixture distribution, which no spec expresses today.

## The R surface

Composite constructors return the same `corehydro_dist` that every existing verb accepts, in
`corehydror/R/distribution.R`:

```r
dist_truncated(d, min, max)
dist_mixture(components, weights, zero_inflated = FALSE, zero_weight = 0)
dist_competing_risks(components, minimum_of = TRUE, dependency = "Independent",
                     correlation = NULL)
dist_empirical(x, p, p_transform = c("NormalZ", "None"), p_descending = FALSE)
dist_kde(data, kernel = c("Gaussian", "Epanechnikov", "Triangular", "Uniform"),
         bandwidth = NULL, bounded_by_data = TRUE)
```

`bandwidth = NULL` selects the Silverman rule, matching the C++ default; `bounded_by_data = TRUE`
matches `kernel_density.hpp:272`.

Copulas, in a new `corehydror/R/copula.R`:

```r
copula(family, theta, df = NULL, margin_x = NULL, margin_y = NULL)
copula_pdf(cop, u, v);  copula_log_pdf(cop, u, v);  copula_cdf(cop, u, v)
copula_inverse_cdf(cop, u, v)
copula_random(cop, n, seed = NULL)
copula_tail_dependence(cop)                        # c(lower = , upper = )
copula_exceedance(cop, u, v, type = c("or", "and"))
copula_log_likelihood(cop, x, y, method = c("pseudo", "ifm", "full"))
copula_params(cop);  copula_bounds(cop);  copula_names()
copula_fit(family, x, y, method = c("mpl", "ifm", "mle", "tau"),
           margin_x = NULL, margin_y = NULL)
print.corehydro_copula
```

`copula_fit()` returns a `corehydro_copula` carrying the fitted theta and the fitted marginals, so
it feeds straight back into the other verbs. `method = "tau"` is available for Clayton, Gumbel, and
AliMikhailHaq, and errors elsewhere with that reason. `method = "mle"` requires both marginals to
implement `IMaximumLikelihoodEstimation`, as upstream does.

Multivariate distributions, in a new `corehydror/R/mvdist.R`:

```r
mvdist_normal(mean, covariance = NULL);  mvdist_student_t(df, location, scale = NULL)
mvdist_dirichlet(alpha);                 mvdist_multinomial(trials, probabilities)
mvdist_bivariate_empirical(x1, x2, p, x1_transform = "None", x2_transform = "None",
                           p_transform = "None")
mvdist_pdf(mv, x);  mvdist_log_pdf(mv, x);  mvdist_cdf(mv, x);  mvdist_mahalanobis(mv, x)
mvdist_random(mv, n, seed = NULL, method = c("random", "latin_hypercube"))
mvdist_mean(mv);  mvdist_covariance(mv);  mvdist_dimension(mv)
mvdist_interval(mv, lower, upper)                  # MultivariateNormal
mvdist_marginal(mv, indices)                       # MultivariateNormal
mvdist_conditional(mv, given, values)              # MultivariateNormal
print.corehydro_mvdist
```

Indices are 1-based in both languages, matching the rule phase 1 set for `trend()` and
`model_parameter()`, and are converted to the 0-based form the spec and the C++ take.
`method = "latin_hypercube"` requires an explicit `seed`, because
`latin_hypercube_random_values` takes no default seed upstream.

### Object shape and back-compatibility

`corehydro_dist` keeps `family` and `params` for flat families and gains an optional `spec`
payload for composites. `save()` round-trips, and `print()` keeps its shape.

Three behaviours change:

- `dist_lmoments()` on a composite errors and names the reason. No composite declares
  `ILinearMomentEstimation`.
- `dist_params()` on an Empirical or KernelDensity object returns the structured payload, because
  `get_parameters()` is empty for both.
- `distribution("Empirical", c(1, 2))` errors at construction and names `dist_empirical()`.
  Today it passes the name check and then throws inside every verb, because
  `EmpiricalDistribution::set_parameters` throws by design. `distribution("KernelDensity", ...)`
  gets the same treatment; it is factory-constructible but absent from `distribution_names()`.
  `distribution_names("structured")` lists the five families that need constructors. The C++
  factory name list is untouched, so the model and spec paths are unaffected.

The three upstream stubs raise in the runner, naming the limitation, so R and Python get the same
message from one place: `mvdist_cdf` on Dirichlet or Multinomial, `mvdist_pdf` on
BivariateEmpirical, and `mvdist_marginal` / `mvdist_conditional` / `mvdist_interval` on
MultivariateStudentT. Guarding in the runner rather than in each language costs nothing here,
because no fixture pins any of the three: `bivariate_empirical.json` asserts `cdf` and never
`pdf`, and Dirichlet and Multinomial assert `pdf` and `log_pdf` and never `cdf`. If a later phase
needs to pin the raw NaN, the guard moves up into the two language layers.

## The Python surface

`corehydropy/src/corehydropy/copula.py` and `mvdist.py`, with additions to `distributions.py`.
Names mirror R exactly. Module functions build objects, verbs are methods, matching how
`Distribution` already works and how `models.py` pairs functions with light wrapper classes.

- `dist_truncated`, `dist_mixture`, `dist_competing_risks`, `dist_empirical`, `dist_kde` return a
  `Distribution`, so every existing method works on them.
- `copula(...)` returns a `Copula` with `.pdf(u, v)`, `.log_pdf`, `.cdf`, `.inverse_cdf`,
  `.random(n, seed)`, `.tail_dependence()`, `.exceedance(u, v, type)`,
  `.log_likelihood(x, y, method)`, `.params`, `.bounds`. `copula_fit` and `copula_names` are module
  functions.
- `mvdist_normal(...)` and its four siblings return a `MultivariateDistribution` with `.pdf`,
  `.log_pdf`, `.cdf`, `.mahalanobis`, `.random`, `.mean`, `.covariance`, `.dimension`,
  `.interval(lower, upper)`, `.marginal(indices)`, `.conditional(given, values)`.

Every object carries `to_json()` and pickles. numpy arrays are accepted; pandas is not a
dependency.

## Fixtures and the oracle gate

The three fixture kinds (`univariate_distribution`, `multivariate_distribution`,
`bivariate_copula`), all seventeen fixture files, and the emitter drivers already exist. The rule
for this phase: every newly exposed verb ships pinned. Any verb without an assertion method or a
case gets both, and the emitter gets the matching dispatch arm.

The measured gaps, from the current fixture files and the dispatchers in `test_fixtures.cpp`:

| Target | Unpinned verbs to cover |
|---|---|
| TruncatedDistribution | `log_pdf`, `random_value`, `log_likelihood` |
| Mixture | `log_pdf`, `random_value`, `log_likelihood`, `median`, `mode`, `skewness`, `kurtosis` |
| CompetingRisks | `random_value`, `log_likelihood`, `mode` |
| Empirical | `pdf`, `log_pdf`, `random_value`, `log_likelihood`, `mean`, `mode`, `sd`, `skewness`, `kurtosis` |
| KernelDensity | `log_pdf`, `random_value`, `log_likelihood`, `mean`, `median`, `mode`, `sd`, `skewness`, `kurtosis` |
| All seven copulas | `or_exceedance`, `and_exceedance`, the three log-likelihoods, `theta_minimum`, `theta_maximum` |
| Copulas other than Normal | `log_pdf` |
| Copulas other than StudentT | `inverse_cdf` |

`log_likelihood` and the theta bounds are new assertion methods in all four runners.
MultivariateNormal, MultivariateStudentT, Dirichlet, and Multinomial are already covered for every
verb this phase exposes.

Three cross-language digest fixtures, following the `short_exact` precedent, prove that a seeded
composite draw, a seeded copula pair stream, and a seeded MVN Latin hypercube sample are
bit-identical in R and Python.

No fixture tolerance is loosened and no `oracle_skip` is added to make a case pass. A value that
will not reproduce gets an entry in `docs/upstream-csharp-issues.md` and a structural assertion
instead, following the phase 9a and 10 precedent.

Behavioural tests live in `corehydror/tests/testthat/test-copula.R`, `test-mvdist.R`, and
additions to `test-distribution.R`, with the Python twins: spec round-tripping, argument
validation and error text, and the three stub errors.

### Risk

Switching the R and Python fixture runners off the thirty bespoke functions touches many pinned
assertions at once. It gets its own task, gated by the fixture suites, and changes no oracle
value. If a number moves, the delegation is wrong.

## Documentation

Every new export must appear in both `corehydror/_pkgdown.yml` and the `quartodoc.sections` of
`site/_quarto.yml`; pkgdown errors on a missing reference-index entry. Each file gains a "Copulas"
section and a "Multivariate distributions" section, and the distributions section gains the five
composite constructors. Every R export gets roxygen with a runnable `@examples`; every Python
export gets numpydoc.

Two example pairs under `site/examples/`, each ending in an executable reproduction check like the
existing thirteen:

- `26-copulas-and-joint-frequency`: peak and volume pairs, fit by MPL and by IFM, the and-joint
  exceedance probability of a coincident event, tail dependence, cross-checked against the
  existing `bivariate_analysis()` model path.
- `27-composite-distributions`: a mixed-population flood record as a two-component mixture, a
  kernel density against the parametric fit, and a truncated distribution for a physical lower
  bound.

Python notebooks are committed with outputs (`jupyter nbconvert --to notebook --execute
--inplace`); the R twins need `site/_freeze/` regenerated and committed. `site/examples/index.qmd`
and the coverage page list both pairs.

## Verification

Run from the repo root, in this order. Every command must pass before the phase is done.

```bash
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build
python3 tools/verify_oracles.py
Rscript -e 'cpp11::cpp_register("corehydror")'
R CMD INSTALL --preclean corehydror
Rscript -e 'testthat::test_local("corehydror")'
pixi run python -m pip install --force-reinstall --no-deps ./corehydropy
pixi run python -m pytest corehydropy/tests -q
R CMD build corehydror && R CMD check --as-cran corehydror_*.tar.gz
pixi run docs && pixi run docs-serve
```

Baselines are re-measured when the branch is cut rather than copied from the notes, which
disagree (testthat 4624 against 4655, pytest 928 against 935). Two figures are consistent in both
records and must not regress: ctest 80/80, and the oracle gate at 4623 reproduced, 0 failed, 11
skipped. The gate must grow by the new cases with no new failures and no new skips.
`R CMD check --as-cran` must hold at its three known NOTEs with no WARNING.

The end-to-end check that the phase delivered its point, run in both languages and compared:

```r
peaks   <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500)
volumes <- c(41000, 52000, 27000, 78000, 61000, 47000, 31000, 96000)
mix    <- dist_mixture(list(distribution("Normal", c(12000, 2000)),
                            distribution("Normal", c(25000, 4000))),
                       weights = c(0.7, 0.3))
dist_pdf(mix, 15000)
dist_random(mix, 5, seed = 12345)

cop <- copula_fit("Clayton", peaks, volumes, method = "mpl",
                  margin_x = "GeneralizedExtremeValue", margin_y = "LogNormal")
copula_exceedance(cop, 0.99, 0.99, type = "and")

mv  <- mvdist_normal(c(0, 0, 0), diag(3))
mvdist_conditional(mv, given = 2, values = 1.5)
```

The identical Python calls must return bit-identical numbers.

## Delivery

- Branch `surface-distribution-layer`, cut from `main` with this spec committed. Do not push and
  do not open a PR without being asked.
- Commits are GPG-signed as `Cam Bracken <cameron.bracken@pm.me>`, with no `Co-Authored-By`
  trailer.
- No new external C++ dependency in `core/`, no new R dependency, no new Python dependency beyond
  numpy.
- Version 0.5.0 in `corehydror/DESCRIPTION`, `corehydropy/pyproject.toml`, and the core version
  stamp, with a `CHANGELOG.md` entry.
