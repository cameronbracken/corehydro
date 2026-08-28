# corehydro

corehydro is an (unofficial) C++ port of tools developed by the 
[United States Army Corps of Engineers](https://www.usace.army.mil/) (USACE) 
[Risk Management Center](https://github.com/USACE-RMC) (RMC) and 
[Hydrologic Engineering Center](https://github.com/HydrologicEngineeringCenter) (HEC)
for stochastic and computational hydrology including distribution fitting and sampling, 
timeseries modeling, uncertainty quantification, optimization, machine learning, flood and precip 
frequency estimation, and a whole lot more. 

The motivation behind this porting effort is to increase awareness 
and accessibility of these amazing software packages (see the [Why?](#why) section 
for more info). Currently this project has ported pieces of the following libraries/packages:
- [RMC-BestFit](https://github.com/USACE-RMC/RMC-BestFit) (statistical engine fully ported)
- [Numerics](https://github.com/USACE-RMC/Numerics) (probability, sampling, and estimation layers)

See the [porting status page](https://cameronbracken.github.io/corehydro/status.html)
for the details of what is currently implemented in the core library and in the packages. 

## R and Python packages

R (`corehydror`) and Python (`corehydropy`) packages are also available with bindings to call the 
functions in the core library. Both packages call the same code, so a seeded run produces identical 
results in each.

The one condition on that is the callback surface, where the ported routine's input is a function 
you write rather than data. The draws still come from the core generator, but your own R or Python 
arithmetic decides the values, and the two languages do not guarantee identical rounding for the 
same formula. A seeded run there reproduces across the two languages if and only if your function 
returns bit-identical values, which arithmetic (`+ - * /`) does and a call into a platform math 
library is not guaranteed to.

## Documentation

- [Documentation site](https://cameronbracken.github.io/corehydro/) with worked examples
  in both languages (Jupyter notebooks for Python, Quarto for R), ported from the
  official [Numerics-Python-Examples](https://github.com/USACE-RMC/Numerics-Python-Examples)
- [Python API reference](https://cameronbracken.github.io/corehydro/reference/)
- [R API reference](https://cameronbracken.github.io/corehydro/r/)

## Development status
Early development. All [RMC.BestFit](https://github.com/USACE-RMC/RMC-BestFit) features have 
been ported but more testing and documentation are needed. 

Neither package is on CRAN or PyPI yet.

## Install

At the moment, the packages can only be installed by compiling 
from source which requires a C++17 compiler. Confirmed working compilers: 

- **macOS**: clang++ ‘Apple clang version 21.0.0 (clang-2100.1.1.101)’ from Xcode command-line tools
- **Linux**: gcc/clang (whatever version GitHub Actions provides)
- **Windows**: gcc/clang (whatever version GitHub Actions provides)

R:

```r
# install.packages("pak")
pak::pak("cameronbracken/corehydro/corehydror")
```

Python (3.10+):

```bash
pip install "git+https://github.com/cameronbracken/corehydro.git#subdirectory=corehydropy"
```

## Quick start

Fit a distribution and read a frequency curve. The two snippets return the same numbers.

R:

```r
library(corehydror)

peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
           19200, 13800, 25600, 10500, 16900)

# Point-estimate GEV fit
gev_fit(peaks, method = "mle")

# Bayesian frequency analysis: parameters, mean curve, and a 90% credible band
fit <- univariate_analysis(peaks, "GeneralizedExtremeValue", seed = 12345)
fit$parameters
fit$mean_curve   # quantile at each of the 25 default exceedance probabilities
fit$lower_ci     # credible band
```

Python:

```python
import corehydropy as ch

peaks = [
    12500,
    15300,
    8900,
    22100,
    18700,
    14200,
    9800,
    28500,
    17400,
    11600,
    19200,
    13800,
    25600,
    10500,
    16900,
]

ch.gev_fit(peaks, method="mle")

fit = ch.univariate_analysis(peaks, "GeneralizedExtremeValue", seed=12345)
fit["parameters"]
fit["mean_curve"]
fit["lower_ci"]
```

Bulletin 17C (log-Pearson Type III) flood-frequency, the USACE standard, with Cohn delta-method
confidence intervals:

```r
b17c <- bulletin17c_analysis(peaks, confidence_level = 0.90, seed = 12345)
b17c$point_estimates   # log10 space
b17c$lower_ci          # discharge space
b17c$upper_ci
```

## Features

**Probability Distributions.** 43 univariate families with density, CDF, quantile, moments, and fitting
(maximum likelihood, L-moments, product moments). Names accepted by the analysis functions include
`Normal`, `LogNormal`, `Gumbel`, `Weibull`, `GeneralizedExtremeValue`, `GeneralizedPareto`,
`GeneralizedLogistic`, `LogPearsonTypeIII`, `PearsonTypeIII`, `KappaFour`, and more. The core library also
carries the multivariate distributions and seven bivariate copulas.

**Analyses.** Easily conduct common statistical hydrology analyses with your own data. Each analysis returns a named list (R) / dict (Python) with fitted parameters, a frequency or forecast curve, a credible band, and goodness-of-fit metrics.

| Function | Purpose |
|----------|---------|
| `univariate_analysis` | Bayesian MCMC frequency curve for one distribution |
| `fit_distributions` | fit and rank 15 candidate distributions by AIC / BIC / RMSE |
| `bulletin17c_analysis` | LP3 flood-frequency with Cohn delta-method intervals |
| `mixture_analysis` | finite mixture of 1-3 component distributions |
| `competing_risk_analysis` | maximum of several independent parents |
| `point_process_analysis` | peaks-over-threshold point process |
| `ar_analysis`, `ma_analysis`, `arima_analysis`, `arimax_analysis` | Bayesian time-series models with forecasting |
| `spatial_gev_analysis` | hierarchical spatial GEV over gauged sites |
| `bivariate_analysis`, `coincident_frequency_analysis` | copula joint-exceedance and conditional frequency |
| `rating_curve_analysis` | BaRatin stage-discharge rating curve |
| `composite_analysis` | competing-risks / mixture / model-average aggregate |
| `bootstrap_analysis` | parametric-bootstrap confidence bands |
| `posterior_predictive_check`, `prior_predictive_check` | model-adequacy checks |
| `estimation_diagnostics` | leverage, PSIS-LOO influence, prior influence |

**Time series.** `time_series()` (R) / `TimeSeries` (Python) carries dated observations on a
stated interval, and the verbs over it cover the container the USACE library builds its
flood-frequency inputs from: moving windows and smoothing, missing-value interpolation and date
filling, interval conversion, summary and per-month statistics, the duration curve, annual maxima
by calendar or water year, peaks over a threshold with an independence criterion, seasonal
decomposition, and two seeded resamplers (a conditional k-nearest-neighbour bootstrap and a
fixed-block bootstrap). Dates are `POSIXct` in R and `datetime64` in Python.

Every function is documented in the package help (eg. `?univariate_analysis` in R, `help(...)` in
Python), with additional information about MCMC sampler choice, credible level, and seeding.

## Reproducibility

Both packages use the same implementation of the Mersenne Twister random number generator ported from the original USACE library, so the same random seed gives identical output across R and Python and across platforms:

```r
univariate_analysis(peaks, "Normal", seed = 42)$parameters   # R
```
```python
ch.univariate_analysis(peaks, "Normal", seed=42)["parameters"]  # same numbers
```

## Layout

| Path | Purpose |
|------|---------|
| `core/` |  C++17 core library (headers, sources, tests) |
| `fixtures/` | language-neutral oracle fixtures (JSON) validating both packages |
| `corehydror/` | R package (cpp11) |
| `corehydropy/` | Python package (scikit-build-core + pybind11) |
| `tools/` | build and validation scripts |

`corehydror` and `corehydropy` share the same code from `core/` and `fixtures/` using symlinks which are resolved at build time. See
`.claude/CLAUDE.md`, `.claude/PLAN.md` and `docs/` for the port architecture and development workflow.

## Build from source

```bash
# C++ core
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build

# R package
Rscript -e 'cpp11::cpp_register("corehydror")'   # only after editing corehydror/src/*.cpp
R CMD INSTALL corehydror
Rscript -e 'testthat::test_local("corehydror")'

# Python package
pip install ./corehydropy
pytest corehydropy/tests
```

## Why?
The US Army Corps of Engineers Risk Management Center (USACE-RMC) has recently released open source versions of some of their core libraries for stochastic hydrology. These libraries represent the state of the art for dam safety risk assessment, flood and precip frequency analysis, and many more common engineering hydrology calculations. The goal of porting these libraries and developing the packages it to make these incredible tools available to a wider audience to enable greater adoption by both practitioners and researchers. The ported C++ code is designed to exactly reproduce the original C# code whenever possible, up to compiler and platform differences.

## AI Use Statement
Anthropic's Claude was used to facilitate the porting process, Fable and Opus 4.8 for planning, Sonnet 5 and Haiku 4.5 for implementation.

## Credit 
All credit for the implementation of these tools goes to [Haden Smith](https://github.com/HadenSmith) and the contributors to [Numerics](https://github.com/USACE-RMC/Numerics/graphs/contributors) and [RMC.BestFit](https://github.com/USACE-RMC/RMC.BestFit/graphs/contributors).

## Other related porting projects

- [hecfda](https://github.com/cameronbracken/hecfda)

## License
The C++ core and both packages are released under the [Zero-Clause BSD (0BSD) license](https://github.com/cameronbracken/corehydro/blob/main/LICENSE), matching the USACE-RMC libraries.
