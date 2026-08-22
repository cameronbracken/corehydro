// Generic, fixture-driven validation for the C++ core.
//
// Reads the language-neutral oracle fixtures (the single source of truth shared with
// the R and Python packages) and checks every assertion. No oracle values live here --
// only the dispatch from fixture method names to the core API. The fixtures directory is
// passed as argv[1] (CMake points it at the repo's canonical fixtures/).
//
// Two code paths: the GEV slice keeps its bespoke dispatch (location/scale/shape names,
// standard-error methods); every other distribution goes through the polymorphic
// UnivariateDistributionBase + factory path, which is what new distributions plug into.
// Special functions use a flat target->lambda map.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "corehydro/analyses/distribution_fitting/fitting_analysis.hpp"
#include "corehydro/analyses/time_series/ar_analysis.hpp"
#include "corehydro/analyses/time_series/arima_analysis.hpp"
#include "corehydro/analyses/time_series/arimax_analysis.hpp"
#include "corehydro/analyses/time_series/ma_analysis.hpp"
#include "corehydro/analyses/univariate/bulletin17c_analysis.hpp"
#include "corehydro/analyses/univariate/competing_risk_analysis.hpp"
#include "corehydro/analyses/univariate/mixture_analysis.hpp"
#include "corehydro/analyses/univariate/point_process_analysis.hpp"
#include "corehydro/analyses/support/analysis_runner.hpp"
#include "corehydro/analyses/univariate/univariate_analysis.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/estimation/generalized_method_of_moments.hpp"
#include "corehydro/estimation/maximum_a_posteriori.hpp"
#include "corehydro/estimation/maximum_likelihood.hpp"
#include "corehydro/estimation/optimization_method.hpp"
#include "corehydro/estimation/support/fit_runner.hpp"
#include "corehydro/models/data_frame/data_collections/exact_series.hpp"
#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/model_spec.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model.hpp"
#include "corehydro/numerics/data/box_cox.hpp"
#include "corehydro/numerics/data/correlation.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/histogram.hpp"
#include "corehydro/numerics/data/interpolation/bilinear.hpp"
#include "corehydro/numerics/data/interpolation/search.hpp"
#include "corehydro/models/data_frame/threshold_diagnostics.hpp"
#include "corehydro/numerics/data/multiple_grubbs_beck_test.hpp"
#include "corehydro/numerics/data/plotting_positions.hpp"
#include "corehydro/numerics/data/yeo_johnson.hpp"
#include "corehydro/numerics/support/toolbox_runner.hpp"
#include "corehydro/numerics/support/callback_runner.hpp"
#include "corehydro/numerics/support/rng_handle.hpp"
#include "corehydro/numerics/support/optimizer_runner.hpp"
#include "corehydro/numerics/sampling/latin_hypercube.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_standard_error.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/copulas/base/bivariate_copula_estimation.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_factory.hpp"
#include "corehydro/numerics/distributions/copulas/clayton_copula.hpp"
#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "corehydro/numerics/distributions/generalized_extreme_value.hpp"
#include "corehydro/numerics/distributions/kernel_density.hpp"
#include "corehydro/numerics/distributions/competing_risks.hpp"
#include "corehydro/numerics/distributions/mixture.hpp"
#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution.hpp"
#include "corehydro/numerics/distributions/multivariate/bivariate_empirical.hpp"
#include "corehydro/numerics/distributions/multivariate/dirichlet.hpp"
#include "corehydro/numerics/distributions/multivariate/multinomial.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_student_t.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/support/dist_runner.hpp"
#include "corehydro/numerics/distributions/truncated_distribution.hpp"
#include "corehydro/numerics/data/running_covariance_matrix.hpp"
#include "corehydro/numerics/data/running_statistics.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/fourier/fourier.hpp"
#include "corehydro/numerics/math/linalg/cholesky_decomposition.hpp"
#include "corehydro/numerics/math/linalg/lu_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/math/special/beta.hpp"
#include "corehydro/numerics/math/special/bessel.hpp"
#include "corehydro/numerics/math/special/erf.hpp"
#include "corehydro/numerics/math/special/factorial.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/sampling/bootstrap/bootstrap.hpp"
#include "corehydro/numerics/sampling/bootstrap/ci_method_names.hpp"
#include "corehydro/numerics/sampling/bootstrap/model_registry.hpp"
#include "corehydro/numerics/sampling/mcmc/arwmh.hpp"
#include "corehydro/numerics/sampling/mcmc/demcz.hpp"
#include "corehydro/numerics/sampling/mcmc/demczs.hpp"
#include "corehydro/numerics/sampling/mcmc/gibbs.hpp"
#include "corehydro/numerics/sampling/mcmc/hmc.hpp"
#include "corehydro/numerics/sampling/mcmc/model_registry.hpp"
#include "corehydro/numerics/sampling/mcmc/nuts.hpp"
#include "corehydro/numerics/sampling/mcmc/rwmh.hpp"
#include "corehydro/numerics/sampling/mcmc/snis.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_diagnostics.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_results.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_run.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"
#include "check.hpp"
#include "fixture_callback_catalog.hpp"
#include "third_party/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace dist = corehydro::numerics::distributions;
namespace supp = corehydro::numerics::distributions::support;
namespace tbx = corehydro::numerics::support;
namespace prob = corehydro::numerics::data::probability;
using dist::EstimationMethod;
using dist::GeneralizedExtremeValue;

static double parse_num(const json& v) {
    if (v.is_string()) {
        std::string s = v.get<std::string>();
        if (s == "nan") return std::numeric_limits<double>::quiet_NaN();
        if (s == "inf") return std::numeric_limits<double>::infinity();
        if (s == "-inf") return -std::numeric_limits<double>::infinity();
        throw std::runtime_error("unexpected string value: " + s);
    }
    return v.get<double>();
}

// --- Shared assertion checking ---------------------------------------------------------

static void check_value(double actual, const json& as, const std::string& where) {
    std::string mode = as["mode"].get<std::string>();
    bool ok;
    if (mode == "equal") {
        double e = parse_num(as["expected"]);
        ok = std::isnan(e) ? std::isnan(actual) : (actual == e);
    } else if (mode == "abs") {
        ok = std::fabs(actual - as["expected"].get<double>()) <= as["tol"].get<double>();
    } else if (mode == "rel") {
        double e = as["expected"].get<double>();
        ok = std::fabs(actual - e) / std::fabs(e) <= as["tol"].get<double>();
    } else {
        throw std::runtime_error("unknown comparison mode: " + mode);
    }
    if (ok) {
        chtest::report_pass();
    } else {
        // The actual is printed at full precision because a mismatch here is usually a question of
        // HOW FAR off, not whether: a fixture pinned at tol 0 and one pinned at rel 1e-12 fail the
        // same way, and only the digits distinguish a real divergence from a last-ulp one.
        std::ostringstream msg;
        msg << where << ": value mismatch, expected " << as["expected"] << ", got "
            << std::setprecision(17) << actual;
        chtest::report_fail(__FILE__, __LINE__, msg.str());
    }
}

static void check_bool(bool actual, const json& as, const std::string& where) {
    if (actual == as["expected"].get<bool>())
        chtest::report_pass();
    else
        chtest::report_fail(__FILE__, __LINE__, where + ": bool mismatch");
}

// --- GEV slice (bespoke) ---------------------------------------------------------------

static EstimationMethod parse_method(const std::string& m) {
    if (m == "mom") return EstimationMethod::MethodOfMoments;
    if (m == "lmom") return EstimationMethod::MethodOfLinearMoments;
    return EstimationMethod::MaximumLikelihood;
}

static GeneralizedExtremeValue build_gev(const json& construct, const json& datasets) {
    if (construct.contains("params")) {
        auto p = construct["params"];
        return GeneralizedExtremeValue(parse_num(p[0]), parse_num(p[1]), parse_num(p[2]));
    }
    const auto& fit = construct["fit"];
    std::vector<double> data;
    for (const auto& v : datasets[fit["dataset"].get<std::string>()]) data.push_back(v.get<double>());
    GeneralizedExtremeValue g;
    g.estimate(data, parse_method(fit["method"].get<std::string>()));
    return g;
}

static double dispatch_gev(const GeneralizedExtremeValue& g, const std::string& m, const json& a) {
    if (m == "mean") return g.mean();
    if (m == "median") return g.median();
    if (m == "mode") return g.mode();
    if (m == "sd") return g.standard_deviation();
    if (m == "skewness") return g.skewness();
    if (m == "kurtosis") return g.kurtosis();
    if (m == "minimum") return g.minimum();
    if (m == "maximum") return g.maximum();
    if (m == "pdf") return g.pdf(a[0].get<double>());
    if (m == "cdf") return g.cdf(a[0].get<double>());
    if (m == "quantile") return g.inverse_cdf(a[0].get<double>());
    if (m == "param") {
        std::string n = a[0].get<std::string>();
        return n == "location" ? g.xi() : n == "scale" ? g.alpha() : g.kappa();
    }
    if (m == "linear_moment")
        return g.linear_moments_from_parameters({g.xi(), g.alpha(), g.kappa()})[a[0].get<int>()];
    if (m == "quantile_gradient") return g.quantile_gradient(a[0].get<double>())[a[1].get<int>()];
    if (m == "parameter_covariance")
        return g.parameter_covariance(a[0].get<int>())[a[1].get<int>()][a[2].get<int>()];
    if (m == "quantile_variance") return g.quantile_variance(a[0].get<double>(), a[1].get<int>());
    if (m == "quantile_se")
        return std::sqrt(g.quantile_variance(a[0].get<double>(), a[1].get<int>()));
    throw std::runtime_error("unknown GEV fixture method: " + m);
}

static void run_gev(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        GeneralizedExtremeValue g = build_gev(c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = name + "/" + method;
            if (as["mode"].get<std::string>() == "bool")
                check_bool(g.parameters_valid(), as, where);
            else
                check_value(dispatch_gev(g, method, args), as, where);
        }
    }
}

// --- Special-function path ------------------------------------------------------------

namespace sf = corehydro::numerics::math::special;
namespace la = corehydro::numerics::math::linalg;
// Alias must not be named `stat` (collides with the MSVC/POSIX CRT symbol, like the
// glibc `gamma` clash documented in .claude/CLAUDE.md).
namespace bfdata = corehydro::numerics::data;
namespace bfsamp = corehydro::numerics::sampling;
namespace bfutil = corehydro::numerics::utilities;
namespace bffourier = corehydro::numerics::math::fourier;
namespace bfdiff = corehydro::numerics::math::differentiation;

// Correlation fixture args are [x..., y...] concatenated and split at the midpoint
// (equal-length samples) -- see fixtures/special_functions/correlation.json / README.md.
static void correlation_split(const std::vector<double>& a, std::vector<double>& x,
                               std::vector<double>& y) {
    std::size_t mid = a.size() / 2;
    x.assign(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(mid));
    y.assign(a.begin() + static_cast<std::ptrdiff_t>(mid), a.end());
}

// Cholesky fixture args are a flattened row-major n*n matrix, with n inferred from the
// args length per the convention documented in fixtures/special_functions/cholesky.json.
static int cholesky_square_n(std::size_t len) {
    int n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(len))));
    if (static_cast<std::size_t>(n) * static_cast<std::size_t>(n) != len)
        throw std::runtime_error("Cholesky fixture args: length is not a perfect square");
    return n;
}

// solve_element args are [flattened n*n matrix, n-length rhs vector, index i], i.e.
// n*n + n + 1 == len; solve the quadratic for n.
static int cholesky_solve_n(std::size_t len) {
    double n_double = (-1.0 + std::sqrt(1.0 + 4.0 * (static_cast<double>(len) - 1.0))) / 2.0;
    int n = static_cast<int>(std::lround(n_double));
    if (static_cast<std::size_t>(n) * static_cast<std::size_t>(n) + static_cast<std::size_t>(n) + 1 != len)
        throw std::runtime_error("Cholesky fixture args: length does not fit n*n+n+1");
    return n;
}

static la::Matrix cholesky_matrix_from_flat(const std::vector<double>& a, int n) {
    std::vector<double> flat(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n) * n);
    return la::Matrix(n, n, flat);
}

// RunningCovariance fixture args: [size, num_pushes, data_flat(num_pushes*size), trailing
// index/indices] -- see fixtures/special_functions/running_covariance.json for the convention.
// Routed through run_toolbox (numerics/support/toolbox_runner.hpp's "statistics.running_covariance"
// method) rather than building bfdata::RunningCovarianceMatrix directly, so these pinned
// special_function values also exercise the toolbox dispatch path the running_covariance() verb
// runs through in both languages. The toolbox convention transposes the fixture's row-major push
// data into one column vector per variable; running_covariance_element() below then knows the
// flat-packed result layout (n, mean, covariance, sample_covariance, sample_correlation,
// population_covariance, population_correlation -- see run_statistics's own comment) well enough
// to pick a single (block, i, j) element back out.
static tbx::ToolboxResult running_covariance_toolbox(const std::vector<double>& a, int size, int num_pushes) {
    std::vector<std::vector<double>> columns(static_cast<std::size_t>(size),
                                             std::vector<double>(static_cast<std::size_t>(num_pushes)));
    for (int p = 0; p < num_pushes; ++p)
        for (int j = 0; j < size; ++j)
            columns[static_cast<std::size_t>(j)][static_cast<std::size_t>(p)] =
                a[2 + static_cast<std::size_t>(p) * static_cast<std::size_t>(size) + static_cast<std::size_t>(j)];
    return tbx::run_toolbox("statistics", "running_covariance", columns, "{}");
}

// block: 0 = mean (i only), 1 = covariance, 2 = sample_covariance, 3 = sample_correlation,
// 4 = population_covariance, 5 = population_correlation (each size x size, i, j both used).
static double running_covariance_element(const tbx::ToolboxResult& r, int size, int block, int i, int j = 0) {
    std::size_t block_size = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
    std::size_t offset = 1;  // skip n
    if (block == 0) return r.values[offset + static_cast<std::size_t>(i)];
    offset += static_cast<std::size_t>(size) + static_cast<std::size_t>(block - 1) * block_size;
    return r.values[offset + static_cast<std::size_t>(i) * static_cast<std::size_t>(size) + static_cast<std::size_t>(j)];
}

// RunningStatistics combine fixture args: [n1, sample1(n1 values), sample2(remaining
// values)] -- a "split-index" convention, distinct from Correlation's equal-length
// two-halves split (Test_Combine/Test_Add split their 69-value sample into UNEQUAL 48/21
// sub-samples, so a fixed midpoint doesn't apply). See
// fixtures/special_functions/running_statistics.json for the full convention. Uses
// operator+ (rather than calling RunningStatistics::combine() directly), which exercises
// both -- operator+ is a one-line forwarder to combine().
static bfdata::RunningStatistics running_statistics_combined(const std::vector<double>& a) {
    std::size_t n1 = static_cast<std::size_t>(a[0]);
    std::vector<double> sample1(a.begin() + 1, a.begin() + 1 + static_cast<std::ptrdiff_t>(n1));
    std::vector<double> sample2(a.begin() + 1 + static_cast<std::ptrdiff_t>(n1), a.end());
    return bfdata::RunningStatistics(sample1) + bfdata::RunningStatistics(sample2);
}

// RunningStatistics.clone_* fixture args convention (fixtures/special_functions/running_statistics.json):
// args = the flat sample (same convention as the plain per-property targets above); builds
// RunningStatistics(sample).clone() and reads one property off the CLONE. Exercises the new
// v2.1.4 Clone() method (see running_statistics.hpp's file header) -- in particular
// clone_skewness/clone_kurtosis would surface a clone() that forgot to copy m3_/m4_.
static bfdata::RunningStatistics running_statistics_clone(const std::vector<double>& a) {
    return bfdata::RunningStatistics(a).clone();
}

// Fourier fixture args conventions (fixtures/special_functions/fourier.json):
//  - Fourier.fft_at / Fourier.real_fft_at: args = [data..., inverse (0/1), index] -- n =
//    len(args) - 2; runs fft()/real_fft() in place on a copy of `data`, returns data[index].
//  - Fourier.correlation_at: args = [data1..., data2..., index] -- equal-length data1/data2
//    concatenated (n = (len(args)-1)/2), returns correlation(data1, data2)[index].
//  - Fourier.autocorrelation_at: args = [series..., lag_max, lag] -- n = len(args) - 2;
//    autocorrelation(series, (int)lag_max) (lag_max == -1 triggers the default auto-lag),
//    returns the acf value (column 1) at row `lag`.
//
// fft_at/real_fft_at/correlation_at are routed through run_toolbox (numerics/support/
// toolbox_runner.hpp's "spectra.dft"/"dft_real"/"cross_correlation" methods, each a direct
// wrapper with no behavior substitution), so these pinned special_function values also exercise
// the toolbox dispatch path the dft()/dft_real()/cross_correlation() verbs run through.
// autocorrelation_at stays a DIRECT call to bffourier::autocorrelation(): the toolbox
// "spectra.autocorrelation" method wraps the newer, oracle-proven-equivalent
// data::Autocorrelation class instead (see autocorrelation.hpp), not this FFT-based function, so
// routing this particular target through it would silently swap which implementation gets
// exercised rather than merely adding a toolbox-path check.
static double fourier_fft_at(const std::vector<double>& a) {
    std::size_t n = a.size() - 2;
    std::vector<double> data(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
    bool inverse = a[n] != 0.0;
    int index = static_cast<int>(a[n + 1]);
    std::string opts = inverse ? "{\"inverse\":true}" : "{}";
    return tbx::run_toolbox("spectra", "dft", {data}, opts).values.at(static_cast<std::size_t>(index));
}
static double fourier_real_fft_at(const std::vector<double>& a) {
    std::size_t n = a.size() - 2;
    std::vector<double> data(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
    bool inverse = a[n] != 0.0;
    int index = static_cast<int>(a[n + 1]);
    std::string opts = inverse ? "{\"inverse\":true}" : "{}";
    return tbx::run_toolbox("spectra", "dft_real", {data}, opts).values.at(static_cast<std::size_t>(index));
}
static double fourier_correlation_at(const std::vector<double>& a) {
    std::size_t n = (a.size() - 1) / 2;
    std::vector<double> data1(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
    std::vector<double> data2(a.begin() + static_cast<std::ptrdiff_t>(n),
                               a.begin() + static_cast<std::ptrdiff_t>(2 * n));
    int index = static_cast<int>(a[2 * n]);
    return tbx::run_toolbox("spectra", "cross_correlation", {data1, data2}, "{}")
        .values.at(static_cast<std::size_t>(index));
}
static double fourier_autocorrelation_at(const std::vector<double>& a) {
    std::size_t n = a.size() - 2;
    std::vector<double> series(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
    int lag_max = static_cast<int>(a[n]);
    int lag = static_cast<int>(a[n + 1]);
    auto acf = bffourier::autocorrelation(series, lag_max);
    if (!acf) throw std::runtime_error("Fourier.autocorrelation_at: autocorrelation returned no value");
    return (*acf)[static_cast<std::size_t>(lag)][1];
}

// Closed registry of named functions for the numerical_derivative fixture -- MUST match
// tools/oracle_emitter/Program.cs's resolver exactly (the emitter runs the REAL C#
// NumericalDerivative against these same two functions):
//  - "quadratic": f(x) = sum_i (x_i - i)^2, i 0-based; analytic gradient 2*(x_i - i),
//    analytic Hessian 2*I (diagonal only) -- a smooth, unbounded-friendly sanity check.
//  - "normal_loglik": Normal(mu=x0, sigma=x1).LogLikelihood(sample) on a small embedded
//    5-point sample -- the exact shape (a 2-parameter log-likelihood) MCMC's default
//    HMC/NUTS gradient differentiates.
static const std::vector<double>& numerical_derivative_normal_sample() {
    static const std::vector<double> sample = {9.0, 10.0, 11.0, 12.0, 13.0};
    return sample;
}
static double numerical_derivative_quadratic(const std::vector<double>& x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        double d = x[i] - static_cast<double>(i);
        s += d * d;
    }
    return s;
}
static double numerical_derivative_normal_loglik(const std::vector<double>& x) {
    dist::Normal n(x[0], x[1]);
    return n.log_likelihood(numerical_derivative_normal_sample());
}

// numerical_derivative fixture args convention
// (fixtures/special_functions/numerical_derivative.json):
//   gradient_element: args = [p, theta(p values), lower(p values), upper(p values), index]
//   hessian_element:  args = [p, theta(p values), lower(p values), upper(p values), i, j]
// `p` is an explicit leading arg (not inferred from length, matching the
// Extensions.next_doubles_grid convention) for clarity. lower/upper always carry an
// explicit p-length bound array using the JSON "-inf"/"inf" literals for "unbounded"
// dimensions, rather than a presence flag -- AvailableLeft/AvailableRight's null-vs-value
// behavior is bitwise identical to a value of +-infinity (theta[j] - (-inf) == +inf either
// way), so this is a behavior-preserving flattening of the C# nullable-array API onto a
// flat numeric args convention. rel_step/abs_step/max_backtrack always use the library
// defaults (not fixture-configurable), matching every real call site (HMC.cs's Gradient
// call and Optimizer.cs's Hessian call both omit them).
static void numerical_derivative_parse(const std::vector<double>& a, std::vector<double>& theta,
                                        std::vector<double>& lower, std::vector<double>& upper,
                                        std::size_t& next) {
    std::size_t p = static_cast<std::size_t>(a[0]);
    theta.assign(a.begin() + 1, a.begin() + 1 + static_cast<std::ptrdiff_t>(p));
    lower.assign(a.begin() + 1 + static_cast<std::ptrdiff_t>(p), a.begin() + 1 + 2 * static_cast<std::ptrdiff_t>(p));
    upper.assign(a.begin() + 1 + 2 * static_cast<std::ptrdiff_t>(p),
                  a.begin() + 1 + 3 * static_cast<std::ptrdiff_t>(p));
    next = 1 + 3 * p;
}
static double numerical_derivative_gradient_element(const bfdiff::ScalarFunction& f, const std::vector<double>& a) {
    std::vector<double> theta, lower, upper;
    std::size_t next;
    numerical_derivative_parse(a, theta, lower, upper, next);
    int index = static_cast<int>(a[next]);
    auto grad = bfdiff::gradient(f, theta, lower, upper);
    return grad[static_cast<std::size_t>(index)];
}
static double numerical_derivative_hessian_element(const bfdiff::ScalarFunction& f, const std::vector<double>& a) {
    std::vector<double> theta, lower, upper;
    std::size_t next;
    numerical_derivative_parse(a, theta, lower, upper, next);
    int i = static_cast<int>(a[next]);
    int j = static_cast<int>(a[next + 1]);
    auto hess = bfdiff::hessian(f, theta, lower, upper);
    return hess[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
}

// DifferentialEvolution fixture args convention (fixtures/special_functions/differential_evolution.json):
//   args = [fn_id, direction, D, lower(D values), upper(D values), index]
// fn_id: 0 = "quadratic" (numerical_derivative_quadratic), 1 = "normal_loglik"
// (numerical_derivative_normal_loglik) -- REUSES the P3.3 closed named-function registry
// above (see numerical_derivative_{quadratic,normal_loglik}) rather than porting a second
// registry, so the emitter runs the REAL C# DifferentialEvolution against the identical
// objective this runner does. direction: 0 = minimize(), 1 = maximize(). index: 0..D-1
// selects best_parameter_set().values[index]; index == D selects
// best_parameter_set().fitness. Every other DifferentialEvolution/Optimizer knob
// (PRNGSeed, PopulationSize, Mutation, DitherRate, CrossoverProbability, MaxIterations,
// tolerances, ReportFailure) is left at its library default -- matching the only real call
// site (MCMCSampler.cs's MAP init, a later task), which overrides none of them except
// ReportFailure (irrelevant here: whether hitting Max*Reached throws-then-is-swallowed or
// never throws at all, BestParameterSet ends up identical either way -- see optimizer.hpp's
// file header for the full analysis).
//
// Task 8 (the optimizer runner) routes this dispatch through tbx::run_optimizer rather than
// constructing DifferentialEvolution directly, so this fixture -- and its pinned oracle values --
// now also exercises optimizer_runner.hpp's own dispatch/spec-parsing path, the same one the R and
// Python glue use. `index == D` (the fitness case) needs one adjustment: OptimResult::value is
// deliberately un-scaled back to the RAW objective's own sign convention (see
// optimizer_runner.hpp's fill_optimizer_result), whereas this fixture's pinned literals were
// curated against C#'s `BestParameterSet.Fitness`, which is the INTERNAL function_scale-scaled
// value (negative of the raw value under Maximize()). Re-applying that scale here keeps the
// already-pinned oracle values unchanged rather than re-curating them.
static double differential_evolution_best_value(const std::vector<double>& a) {
    int fn_id = static_cast<int>(a[0]);
    int direction = static_cast<int>(a[1]);
    int D = static_cast<int>(a[2]);
    std::vector<double> lower(a.begin() + 3, a.begin() + 3 + D);
    std::vector<double> upper(a.begin() + 3 + D, a.begin() + 3 + 2 * D);
    int index = static_cast<int>(a[static_cast<std::size_t>(3 + 2 * D)]);

    tbx::Objective f = fn_id == 0 ? tbx::Objective(numerical_derivative_quadratic)
                                  : tbx::Objective(numerical_derivative_normal_loglik);
    json spec;
    spec["method"] = "de";
    spec["lower"] = lower;
    spec["upper"] = upper;
    spec["maximize"] = (direction == 1);
    tbx::OptimResult r = tbx::run_optimizer(spec.dump(), f);
    if (index == D) return direction == 1 ? -r.value : r.value;
    return r.parameters.at(static_cast<std::size_t>(index));
}

// Histogram fixture args convention (fixtures/special_functions/histogram.json): args =
// [explicit_bins, data...] for the whole-histogram scalar targets (explicit_bins == 0 uses
// the Rice-Rule ctor; explicit_bins > 0 uses the explicit-bin-count ctor). The
// bin_*_at/get_bin_index_of element-lookup targets append one trailing probe value (a bin
// index, or an x value to look up) after `data`; `trailing` tells histogram_build() how
// many args at the end of `a` are NOT part of `data`.
static bfdata::Histogram histogram_build(const std::vector<double>& a, std::size_t trailing) {
    int explicit_bins = static_cast<int>(a[0]);
    std::vector<double> data(a.begin() + 1, a.end() - static_cast<std::ptrdiff_t>(trailing));
    if (explicit_bins > 0) return bfdata::Histogram(data, explicit_bins);
    return bfdata::Histogram(data);
}

// Same [explicit_bins, ...] convention as histogram_build(), but for the run_toolbox-routed
// Histogram.* table entries below: builds the "histogram" group's options_json (bins > 0 sets
// the "bins" option; bins == 0 omits it, selecting the Rice-Rule constructor).
static std::string histogram_toolbox_options(int explicit_bins) {
    return explicit_bins > 0 ? "{\"bins\":" + std::to_string(explicit_bins) + "}" : "{}";
}

// Histogram.adapt_* fixture args convention (fixtures/special_functions/histogram.json):
// args = [explicit_bins, num_adds, data..., adds(num_adds)...] -- builds the histogram via
// the same explicit_bins/data convention as histogram_build() above, then replays each of
// `adds` through a scalar add_data(double) call, in order. Exercises the v2.1.4
// AddData-endpoint-adapt fix (see histogram.hpp's file header): before the fix, an add
// value strictly outside the current bounds threw instead of widening the endpoint bin.
static bfdata::Histogram histogram_build_adapt(const std::vector<double>& a) {
    int explicit_bins = static_cast<int>(a[0]);
    int num_adds = static_cast<int>(a[1]);
    std::vector<double> data(a.begin() + 2, a.end() - static_cast<std::ptrdiff_t>(num_adds));
    bfdata::Histogram h = explicit_bins > 0 ? bfdata::Histogram(data, explicit_bins) : bfdata::Histogram(data);
    for (auto it = a.end() - static_cast<std::ptrdiff_t>(num_adds); it != a.end(); ++it) h.add_data(*it);
    return h;
}

// Bilinear.log_floor_* fixture args convention (fixtures/special_functions/bilinear.json):
// args = [x1_query, x2_query] against a FIXED 3x3 identity grid ({0, 1E-15, 1} on both
// axes, y[i][j] = x1_values[i] for every j) with X1/X2/Y all Transform::Logarithmic -- the
// exact grid the new v2.1.4 Test_LogarithmicFloorMatchesLinearInterpolation uses to prove
// Bilinear's guarded log10 floor now matches Linear's (see bilinear.hpp's file header).
// Routed through run_toolbox (numerics/support/toolbox_runner.hpp's "interpolation.bilinear"
// method) rather than calling bfdata::Bilinear directly, so this pinned value also exercises
// the toolbox dispatch path interpolate_2d() runs through in both languages.
static double bilinear_log_floor_value(const std::vector<double>& a) {
    std::vector<double> coords = {0.0, 1e-15, 1.0};
    std::vector<double> flat = {0.0, 0.0, 0.0, 1e-15, 1e-15, 1e-15, 1.0, 1.0, 1.0};
    std::string opts = "{\"x1_transform\":\"log\",\"x2_transform\":\"log\",\"y_transform\":\"log\"}";
    auto r = tbx::run_toolbox("interpolation", "bilinear", {coords, coords, flat, {a[0]}, {a[1]}}, opts);
    return r.values.at(0);
}

// Probability.hpcm_* fixture args convention (fixtures/special_functions/probability.json):
// args = [p_0..p_(n-1), ind_0..ind_(n-1), corr(n*n flattened row-major)] for hpcm_joint; n is
// inferred as the unique n solving 2n + n^2 = len(args). hpcm_conditional_at appends one
// trailing 0-based component index (so its own args length is one more). Exercises the
// v2.1.4 minimumCdf-guard fix in joint_probability_hpcm's "First cycle" (see probability.hpp's
// file header) with the new Test_JointProbabilityHPCM_ExtremeProbabilitiesRemainFinite inputs
// (a probability of exactly 0 and a subnormal 1E-320), which previously could divide by a
// near-zero standard-normal CDF in that unguarded first cycle.
static int probability_hpcm_n(std::size_t len) {
    for (int n = 1; n <= 20; ++n)
        if (static_cast<std::size_t>(2 * n + n * n) == len) return n;
    throw std::runtime_error("cannot infer n for Probability.hpcm args");
}

static double probability_hpcm_joint(const std::vector<double>& a, std::vector<double>* conditional = nullptr) {
    int n = probability_hpcm_n(a.size());
    std::vector<double> probabilities(a.begin(), a.begin() + n);
    std::vector<int> indicators(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        indicators[static_cast<std::size_t>(i)] = static_cast<int>(a[static_cast<std::size_t>(n + i)]);
    prob::Matrix2D corr(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n)));
    std::size_t base = static_cast<std::size_t>(2 * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            corr[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                a[base + static_cast<std::size_t>(i * n + j)];
    return prob::joint_probability_hpcm(probabilities, indicators, corr, conditional);
}

static double probability_hpcm_conditional_at(const std::vector<double>& a) {
    int idx = static_cast<int>(a.back());
    std::vector<double> body(a.begin(), a.end() - 1);
    int n = probability_hpcm_n(body.size());
    std::vector<double> cond(static_cast<std::size_t>(n));
    probability_hpcm_joint(body, &cond);
    return cond[static_cast<std::size_t>(idx)];
}

// Dispatch table: maps "Module.method" → a free function of (vector<double>) → double.
static const std::map<std::string, std::function<double(const std::vector<double>&)>>&
special_function_table() {
    static const std::map<std::string, std::function<double(const std::vector<double>&)>> t = {
        // Cholesky family (args: flattened row-major matrix, n inferred from length --
        // see fixtures/special_functions/cholesky.json for the full convention)
        {"Cholesky.determinant", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size());
            return la::CholeskyDecomposition(cholesky_matrix_from_flat(a, n)).determinant();
        }},
        {"Cholesky.log_determinant", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size());
            return la::CholeskyDecomposition(cholesky_matrix_from_flat(a, n)).log_determinant();
        }},
        {"Cholesky.inverse_element", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size() - 2);
            la::CholeskyDecomposition chol(cholesky_matrix_from_flat(a, n));
            int i = static_cast<int>(a[static_cast<std::size_t>(n) * n]);
            int j = static_cast<int>(a[static_cast<std::size_t>(n) * n + 1]);
            return chol.inverse_a()(i, j);
        }},
        {"Cholesky.solve_element", [](const std::vector<double>& a) {
            int n = cholesky_solve_n(a.size());
            la::CholeskyDecomposition chol(cholesky_matrix_from_flat(a, n));
            std::vector<double> rhs(a.begin() + static_cast<std::ptrdiff_t>(n) * n,
                                     a.begin() + static_cast<std::ptrdiff_t>(n) * n + n);
            int i = static_cast<int>(a[static_cast<std::size_t>(n) * n + static_cast<std::size_t>(n)]);
            return chol.solve(la::Vector(std::move(rhs)))[i];
        }},
        // Returns 1.0 if the matrix is positive-definite (construction succeeds), 0.0 if
        // the ctor throws std::runtime_error (non-PD or NaN diagonal) -- pins the
        // exception condition against the real C# behavior (see Program.cs's resolver).
        {"Cholesky.is_positive_definite", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size());
            try {
                return la::CholeskyDecomposition(cholesky_matrix_from_flat(a, n)).is_positive_definite()
                           ? 1.0
                           : 0.0;
            } catch (const std::runtime_error&) {
                return 0.0;
            }
        }},
        // Erf family
        {"Erf.function",      [](const std::vector<double>& a) { return sf::erf::function(a[0]); }},
        {"Erf.erfc",          [](const std::vector<double>& a) { return sf::erf::erfc(a[0]); }},
        {"Erf.inverse_erf",   [](const std::vector<double>& a) { return sf::erf::inverse_erf(a[0]); }},
        {"Erf.inverse_erfc",  [](const std::vector<double>& a) { return sf::erf::inverse_erfc(a[0]); }},
        // Gamma family
        {"Gamma.function",               [](const std::vector<double>& a) { return sf::function(a[0]); }},
        {"Gamma.log_gamma",              [](const std::vector<double>& a) { return sf::log_gamma(a[0]); }},
        {"Gamma.digamma",                [](const std::vector<double>& a) { return sf::digamma(a[0]); }},
        {"Gamma.trigamma",               [](const std::vector<double>& a) { return sf::trigamma(a[0]); }},
        {"Gamma.lower_incomplete",       [](const std::vector<double>& a) { return sf::lower_incomplete(a[0], a[1]); }},
        {"Gamma.upper_incomplete",       [](const std::vector<double>& a) { return sf::upper_incomplete(a[0], a[1]); }},
        {"Gamma.inverse_lower_incomplete", [](const std::vector<double>& a) { return sf::inverse_lower_incomplete(a[0], a[1]); }},
        {"Gamma.inverse_upper_incomplete", [](const std::vector<double>& a) { return sf::inverse_upper_incomplete(a[0], a[1]); }},
        // Beta family
        {"Beta.function",           [](const std::vector<double>& a) { return sf::beta::function(a[0], a[1]); }},
        {"Beta.incomplete",         [](const std::vector<double>& a) { return sf::beta::incomplete(a[0], a[1], a[2]); }},
        {"Beta.incbcf",             [](const std::vector<double>& a) { return sf::beta::detail::incbcf(a[0], a[1], a[2]); }},
        {"Beta.incbd",              [](const std::vector<double>& a) { return sf::beta::detail::incbd(a[0], a[1], a[2]); }},
        {"Beta.power_series",       [](const std::vector<double>& a) { return sf::beta::detail::power_series(a[0], a[1], a[2]); }},
        {"Beta.incomplete_inverse", [](const std::vector<double>& a) { return sf::beta::incomplete_inverse(a[0], a[1], a[2]); }},
        // Factorial family
        {"Factorial.function",             [](const std::vector<double>& a) { return sf::factorial::function(static_cast<int>(a[0])); }},
        {"Factorial.log_factorial",        [](const std::vector<double>& a) { return sf::factorial::log_factorial(static_cast<int>(a[0])); }},
        {"Factorial.binomial_coefficient", [](const std::vector<double>& a) { return sf::factorial::binomial_coefficient(static_cast<int>(a[0]), static_cast<int>(a[1])); }},
        // Bessel family
        {"Bessel.i0", [](const std::vector<double>& a) { return sf::bessel::i0(a[0]); }},
        {"Bessel.i1", [](const std::vector<double>& a) { return sf::bessel::i1(a[0]); }},
        // Correlation family (args: [x..., y...], split at the midpoint -- see
        // fixtures/special_functions/correlation.json for the full convention). Routed through
        // run_toolbox (numerics/support/toolbox_runner.hpp) rather than calling bfdata::pearson
        // etc. directly, so these pinned special_function values also exercise the toolbox
        // dispatch path the "correlation" verb runs through in both languages.
        {"Correlation.pearson", [](const std::vector<double>& a) {
            std::vector<double> x, y;
            correlation_split(a, x, y);
            return tbx::run_toolbox("correlation", "pearson", {x, y}, "{}").values.at(0);
        }},
        {"Correlation.spearman", [](const std::vector<double>& a) {
            std::vector<double> x, y;
            correlation_split(a, x, y);
            return tbx::run_toolbox("correlation", "spearman", {x, y}, "{}").values.at(0);
        }},
        {"Correlation.kendalls_tau", [](const std::vector<double>& a) {
            std::vector<double> x, y;
            correlation_split(a, x, y);
            return tbx::run_toolbox("correlation", "kendall", {x, y}, "{}").values.at(0);
        }},
        // LU family (args: flattened row-major matrix, n inferred from length -- reuses
        // the Cholesky-fixture flatten helpers above, which are generic matrix-args
        // conventions, not Cholesky-specific; see fixtures/special_functions/lu_decomposition.json)
        {"LU.determinant", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size());
            return la::LUDecomposition(cholesky_matrix_from_flat(a, n)).determinant();
        }},
        {"LU.inverse_element", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size() - 2);
            la::LUDecomposition lu(cholesky_matrix_from_flat(a, n));
            int i = static_cast<int>(a[static_cast<std::size_t>(n) * n]);
            int j = static_cast<int>(a[static_cast<std::size_t>(n) * n + 1]);
            return lu.inverse_a()(i, j);
        }},
        {"LU.solve_element", [](const std::vector<double>& a) {
            int n = cholesky_solve_n(a.size());
            la::LUDecomposition lu(cholesky_matrix_from_flat(a, n));
            std::vector<double> rhs(a.begin() + static_cast<std::ptrdiff_t>(n) * n,
                                     a.begin() + static_cast<std::ptrdiff_t>(n) * n + n);
            int i = static_cast<int>(a[static_cast<std::size_t>(n) * n + static_cast<std::size_t>(n)]);
            return lu.solve(la::Vector(std::move(rhs)))[i];
        }},
        // Percentile (args: [data_1..data_n, k, data_is_sorted (0.0/1.0)] -- see
        // fixtures/special_functions/percentile.json for the convention). Routed through
        // run_toolbox (numerics/support/toolbox_runner.hpp's "statistics.percentile" method)
        // rather than calling bfdata::percentile directly, so these pinned special_function
        // values also exercise the toolbox dispatch path the percentile() verb runs through.
        {"Statistics.percentile", [](const std::vector<double>& a) {
            std::size_t n = a.size() - 2;
            std::vector<double> data(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
            double k = a[n];
            bool sorted = a[n + 1] != 0.0;
            std::string opts = sorted ? "{\"sorted\":true}" : "{}";
            return tbx::run_toolbox("statistics", "percentile", {data, {k}}, opts).values.at(0);
        }},
        // Extensions/MersenneTwister ranged-draw family (see
        // fixtures/special_functions/extension_methods.json for the conventions)
        {"Extensions.next_doubles_grid", [](const std::vector<double>& a) {
            // args: [n, dim, seed, row, col]
            int n = static_cast<int>(a[0]);
            int dim = static_cast<int>(a[1]);
            bfsamp::MersenneTwister rng(static_cast<std::uint32_t>(a[2]));
            int row = static_cast<int>(a[3]);
            int col = static_cast<int>(a[4]);
            auto grid = bfutil::next_doubles(rng, n, dim);
            return grid[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
        }},
        {"Extensions.next_integers_at", [](const std::vector<double>& a) {
            // args: [n, seed, i]
            int n = static_cast<int>(a[0]);
            bfsamp::MersenneTwister rng(static_cast<std::uint32_t>(a[1]));
            int i = static_cast<int>(a[2]);
            auto values = bfutil::next_integers(rng, n);
            return static_cast<double>(values[static_cast<std::size_t>(i)]);
        }},
        {"Mt.next_range", [](const std::vector<double>& a) {
            // args: [seed, min, max, i] -- draws next(min, max) (i+1) times, 0-based,
            // returning the i-th draw.
            bfsamp::MersenneTwister rng(static_cast<std::uint32_t>(a[0]));
            int min_v = static_cast<int>(a[1]);
            int max_v = static_cast<int>(a[2]);
            int i = static_cast<int>(a[3]);
            int result = 0;
            for (int k = 0; k <= i; ++k) result = rng.next(min_v, max_v);
            return static_cast<double>(result);
        }},
        // RunningCovarianceMatrix family (args: [size, num_pushes, data_flat, trailing
        // index/indices] -- see fixtures/special_functions/running_covariance.json). Routed
        // through run_toolbox (running_covariance_toolbox()/running_covariance_element() above)
        // rather than building bfdata::RunningCovarianceMatrix directly.
        {"RunningCovariance.mean_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto r = running_covariance_toolbox(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            return running_covariance_element(r, size, 0, i);
        }},
        {"RunningCovariance.covariance_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto r = running_covariance_toolbox(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            int j = static_cast<int>(a[base + 1]);
            return running_covariance_element(r, size, 1, i, j);
        }},
        {"RunningCovariance.sample_covariance_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto r = running_covariance_toolbox(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            int j = static_cast<int>(a[base + 1]);
            return running_covariance_element(r, size, 2, i, j);
        }},
        {"RunningCovariance.sample_correlation_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto r = running_covariance_toolbox(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            int j = static_cast<int>(a[base + 1]);
            return running_covariance_element(r, size, 3, i, j);
        }},
        {"RunningCovariance.population_covariance_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto r = running_covariance_toolbox(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            int j = static_cast<int>(a[base + 1]);
            return running_covariance_element(r, size, 4, i, j);
        }},
        {"RunningCovariance.population_correlation_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto r = running_covariance_toolbox(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            int j = static_cast<int>(a[base + 1]);
            return running_covariance_element(r, size, 5, i, j);
        }},
        // RunningStatistics family (args: the flat sample; see
        // fixtures/special_functions/running_statistics.json). The nine properties the toolbox
        // "statistics.summary" method exposes are routed through run_toolbox, so these pinned
        // special_function values also exercise the toolbox dispatch path the
        // running_statistics() verb runs through in both languages. The four population-
        // normalized variants (population_variance/population_standard_deviation/
        // population_skewness/population_kurtosis) have no run_statistics equivalent -- the
        // "summary" method's fixed 13-name result mirrors only what running_statistics() exposes
        // -- so those four, plus combined_*/clone_* below (which exercise combine()/clone(), not
        // reachable through any toolbox verb), stay direct bfdata::RunningStatistics calls.
        {"RunningStatistics.mean", [](const std::vector<double>& a) {
            return tbx::run_toolbox("statistics", "summary", {a}, "{}").values.at(3);  // "mean"
        }},
        {"RunningStatistics.variance", [](const std::vector<double>& a) {
            return tbx::run_toolbox("statistics", "summary", {a}, "{}").values.at(4);  // "variance"
        }},
        {"RunningStatistics.standard_deviation", [](const std::vector<double>& a) {
            return tbx::run_toolbox("statistics", "summary", {a}, "{}").values.at(5);  // "sd"
        }},
        {"RunningStatistics.population_variance", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).population_variance(); }},
        {"RunningStatistics.population_standard_deviation", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).population_standard_deviation(); }},
        {"RunningStatistics.coefficient_of_variation", [](const std::vector<double>& a) {
            return tbx::run_toolbox("statistics", "summary", {a}, "{}").values.at(6);  // "cv"
        }},
        {"RunningStatistics.skewness", [](const std::vector<double>& a) {
            return tbx::run_toolbox("statistics", "summary", {a}, "{}").values.at(7);  // "skewness"
        }},
        {"RunningStatistics.population_skewness", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).population_skewness(); }},
        {"RunningStatistics.kurtosis", [](const std::vector<double>& a) {
            return tbx::run_toolbox("statistics", "summary", {a}, "{}").values.at(8);  // "kurtosis"
        }},
        {"RunningStatistics.population_kurtosis", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).population_kurtosis(); }},
        {"RunningStatistics.minimum", [](const std::vector<double>& a) {
            return tbx::run_toolbox("statistics", "summary", {a}, "{}").values.at(1);  // "minimum"
        }},
        {"RunningStatistics.maximum", [](const std::vector<double>& a) {
            return tbx::run_toolbox("statistics", "summary", {a}, "{}").values.at(2);  // "maximum"
        }},
        {"RunningStatistics.count", [](const std::vector<double>& a) {
            return tbx::run_toolbox("statistics", "summary", {a}, "{}").values.at(0);  // "n"
        }},
        // RunningStatistics combine family (args: [n1, sample1(n1), sample2(m)] -- see
        // running_statistics_combined() above and fixtures/special_functions/running_statistics.json)
        {"RunningStatistics.combined_minimum", [](const std::vector<double>& a) { return running_statistics_combined(a).minimum(); }},
        {"RunningStatistics.combined_maximum", [](const std::vector<double>& a) { return running_statistics_combined(a).maximum(); }},
        {"RunningStatistics.combined_mean", [](const std::vector<double>& a) { return running_statistics_combined(a).mean(); }},
        {"RunningStatistics.combined_variance", [](const std::vector<double>& a) { return running_statistics_combined(a).variance(); }},
        {"RunningStatistics.combined_standard_deviation", [](const std::vector<double>& a) { return running_statistics_combined(a).standard_deviation(); }},
        {"RunningStatistics.combined_coefficient_of_variation", [](const std::vector<double>& a) { return running_statistics_combined(a).coefficient_of_variation(); }},
        {"RunningStatistics.combined_skewness", [](const std::vector<double>& a) { return running_statistics_combined(a).skewness(); }},
        {"RunningStatistics.combined_kurtosis", [](const std::vector<double>& a) { return running_statistics_combined(a).kurtosis(); }},
        {"RunningStatistics.combined_count", [](const std::vector<double>& a) {
            return static_cast<double>(running_statistics_combined(a).count());
        }},
        // RunningStatistics.clone_* family (args: the flat sample -- see
        // running_statistics_clone() above and fixtures/special_functions/running_statistics.json)
        {"RunningStatistics.clone_mean", [](const std::vector<double>& a) { return running_statistics_clone(a).mean(); }},
        {"RunningStatistics.clone_variance", [](const std::vector<double>& a) { return running_statistics_clone(a).variance(); }},
        {"RunningStatistics.clone_skewness", [](const std::vector<double>& a) { return running_statistics_clone(a).skewness(); }},
        {"RunningStatistics.clone_kurtosis", [](const std::vector<double>& a) { return running_statistics_clone(a).kurtosis(); }},
        {"RunningStatistics.clone_minimum", [](const std::vector<double>& a) { return running_statistics_clone(a).minimum(); }},
        {"RunningStatistics.clone_maximum", [](const std::vector<double>& a) { return running_statistics_clone(a).maximum(); }},
        {"RunningStatistics.clone_count", [](const std::vector<double>& a) {
            return static_cast<double>(running_statistics_clone(a).count());
        }},
        // Fourier family (see fourier_*_at() above for the args conventions)
        {"Fourier.fft_at", fourier_fft_at},
        {"Fourier.real_fft_at", fourier_real_fft_at},
        {"Fourier.correlation_at", fourier_correlation_at},
        {"Fourier.autocorrelation_at", fourier_autocorrelation_at},
        // NumericalDerivative family (closed function registry; see
        // numerical_derivative_{quadratic,normal_loglik} and the args convention above)
        {"NumericalDerivative.gradient_element_quadratic", [](const std::vector<double>& a) {
            return numerical_derivative_gradient_element(numerical_derivative_quadratic, a);
        }},
        {"NumericalDerivative.gradient_element_normal_loglik", [](const std::vector<double>& a) {
            return numerical_derivative_gradient_element(numerical_derivative_normal_loglik, a);
        }},
        {"NumericalDerivative.hessian_element_quadratic", [](const std::vector<double>& a) {
            return numerical_derivative_hessian_element(numerical_derivative_quadratic, a);
        }},
        {"NumericalDerivative.hessian_element_normal_loglik", [](const std::vector<double>& a) {
            return numerical_derivative_hessian_element(numerical_derivative_normal_loglik, a);
        }},
        // DifferentialEvolution family (see differential_evolution_best_value() above and
        // fixtures/special_functions/differential_evolution.json for the args convention)
        {"DifferentialEvolution.best_value", differential_evolution_best_value},
        // Histogram family (args: [explicit_bins, data..., trailing probe?] -- see
        // histogram_build() above and fixtures/special_functions/histogram.json). The
        // whole-histogram scalar targets and the bin_*_at element lookups are routed through
        // run_toolbox (numerics/support/toolbox_runner.hpp's "histogram.statistics"/"histogram.bins"
        // methods) rather than calling bfdata::Histogram directly, so these pinned special_function
        // values also exercise the toolbox dispatch path histogram() runs through in both
        // languages -- and, for bin_*_at, the FIRST toolbox method to return a real matrix through
        // `dims`. data_count and get_bin_index_of have no toolbox-method equivalent (neither
        // "statistics" nor "bins" exposes them) and adapt_* needs AddData(), which the toolbox
        // arm's stateless one-shot construction never calls, so those stay direct calls, mirroring
        // the RunningStatistics.population_variance precedent above.
        {"Histogram.number_of_bins", [](const std::vector<double>& a) {
            std::vector<double> data(a.begin() + 1, a.end());
            auto opts = histogram_toolbox_options(static_cast<int>(a[0]));
            return tbx::run_toolbox("histogram", "statistics", {data}, opts).values.at(7);  // "bins"
        }},
        {"Histogram.bin_width", [](const std::vector<double>& a) {
            std::vector<double> data(a.begin() + 1, a.end());
            auto opts = histogram_toolbox_options(static_cast<int>(a[0]));
            return tbx::run_toolbox("histogram", "statistics", {data}, opts).values.at(6);  // "bin_width"
        }},
        {"Histogram.lower_bound", [](const std::vector<double>& a) {
            std::vector<double> data(a.begin() + 1, a.end());
            auto opts = histogram_toolbox_options(static_cast<int>(a[0]));
            return tbx::run_toolbox("histogram", "statistics", {data}, opts).values.at(4);  // "lower"
        }},
        {"Histogram.upper_bound", [](const std::vector<double>& a) {
            std::vector<double> data(a.begin() + 1, a.end());
            auto opts = histogram_toolbox_options(static_cast<int>(a[0]));
            return tbx::run_toolbox("histogram", "statistics", {data}, opts).values.at(5);  // "upper"
        }},
        {"Histogram.data_count", [](const std::vector<double>& a) {
            return static_cast<double>(histogram_build(a, 0).data_count());
        }},
        {"Histogram.mean", [](const std::vector<double>& a) {
            std::vector<double> data(a.begin() + 1, a.end());
            auto opts = histogram_toolbox_options(static_cast<int>(a[0]));
            return tbx::run_toolbox("histogram", "statistics", {data}, opts).values.at(0);  // "mean"
        }},
        {"Histogram.median", [](const std::vector<double>& a) {
            std::vector<double> data(a.begin() + 1, a.end());
            auto opts = histogram_toolbox_options(static_cast<int>(a[0]));
            return tbx::run_toolbox("histogram", "statistics", {data}, opts).values.at(1);  // "median"
        }},
        {"Histogram.mode", [](const std::vector<double>& a) {
            std::vector<double> data(a.begin() + 1, a.end());
            auto opts = histogram_toolbox_options(static_cast<int>(a[0]));
            return tbx::run_toolbox("histogram", "statistics", {data}, opts).values.at(2);  // "mode"
        }},
        {"Histogram.standard_deviation", [](const std::vector<double>& a) {
            std::vector<double> data(a.begin() + 1, a.end());
            auto opts = histogram_toolbox_options(static_cast<int>(a[0]));
            return tbx::run_toolbox("histogram", "statistics", {data}, opts).values.at(3);  // "sd"
        }},
        {"Histogram.bin_lower_bound_at", [](const std::vector<double>& a) {
            int probe = static_cast<int>(a.back());
            std::vector<double> data(a.begin() + 1, a.end() - 1);
            auto opts = histogram_toolbox_options(static_cast<int>(a[0]));
            auto r = tbx::run_toolbox("histogram", "bins", {data}, opts);
            return r.values.at(static_cast<std::size_t>(probe) * 4 + 0);
        }},
        {"Histogram.bin_upper_bound_at", [](const std::vector<double>& a) {
            int probe = static_cast<int>(a.back());
            std::vector<double> data(a.begin() + 1, a.end() - 1);
            auto opts = histogram_toolbox_options(static_cast<int>(a[0]));
            auto r = tbx::run_toolbox("histogram", "bins", {data}, opts);
            return r.values.at(static_cast<std::size_t>(probe) * 4 + 1);
        }},
        {"Histogram.bin_frequency_at", [](const std::vector<double>& a) {
            int probe = static_cast<int>(a.back());
            std::vector<double> data(a.begin() + 1, a.end() - 1);
            auto opts = histogram_toolbox_options(static_cast<int>(a[0]));
            auto r = tbx::run_toolbox("histogram", "bins", {data}, opts);
            return r.values.at(static_cast<std::size_t>(probe) * 4 + 3);
        }},
        {"Histogram.get_bin_index_of", [](const std::vector<double>& a) {
            return static_cast<double>(histogram_build(a, 1).get_bin_index_of(a.back()));
        }},
        // Histogram.adapt_* family (args: [explicit_bins, num_adds, data..., adds...] -- see
        // histogram_build_adapt() above and fixtures/special_functions/histogram.json)
        {"Histogram.adapt_lower_bound", [](const std::vector<double>& a) {
            return histogram_build_adapt(a).lower_bound();
        }},
        {"Histogram.adapt_upper_bound", [](const std::vector<double>& a) {
            return histogram_build_adapt(a).upper_bound();
        }},
        {"Histogram.adapt_bin_first_lower_bound", [](const std::vector<double>& a) {
            auto h = histogram_build_adapt(a);
            return h.bin(0).lower_bound;
        }},
        {"Histogram.adapt_bin_last_upper_bound", [](const std::vector<double>& a) {
            auto h = histogram_build_adapt(a);
            return h.bin(h.number_of_bins() - 1).upper_bound;
        }},
        {"Histogram.adapt_bin_first_frequency", [](const std::vector<double>& a) {
            auto h = histogram_build_adapt(a);
            return static_cast<double>(h.bin(0).frequency);
        }},
        {"Histogram.adapt_bin_last_frequency", [](const std::vector<double>& a) {
            auto h = histogram_build_adapt(a);
            return static_cast<double>(h.bin(h.number_of_bins() - 1).frequency);
        }},
        {"Histogram.adapt_data_count", [](const std::vector<double>& a) {
            return static_cast<double>(histogram_build_adapt(a).data_count());
        }},
        // PlottingPositions family (args: [N, alpha, i] for function_at; [N, i] for
        // weibull_at -- see fixtures/special_functions/plotting_positions.json)
        {"PlottingPositions.function_at", [](const std::vector<double>& a) {
            auto pp = bfdata::plotting_positions::function(static_cast<int>(a[0]), a[1]);
            return pp[static_cast<std::size_t>(static_cast<int>(a[2]))];
        }},
        {"PlottingPositions.weibull_at", [](const std::vector<double>& a) {
            auto pp = bfdata::plotting_positions::weibull(static_cast<int>(a[0]));
            return pp[static_cast<std::size_t>(static_cast<int>(a[1]))];
        }},
        // Search family (args: [values..., x, start] -- see
        // fixtures/special_functions/search.json). Stays a direct bfdata::search:: call: neither
        // toolbox method the "interpolation" group exposes ("linear"/"bilinear") returns a search
        // index, only an interpolated y -- the same "no run_toolbox-reachable equivalent" reasoning
        // that keeps RunningStatistics.population_variance and Fourier.autocorrelation_at direct
        // above.
        {"Search.sequential", [](const std::vector<double>& a) {
            std::size_t n = a.size() - 2;
            std::vector<double> values(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
            return static_cast<double>(
                bfdata::search::sequential(a[n], values, static_cast<int>(a[n + 1])));
        }},
        {"Search.bisection", [](const std::vector<double>& a) {
            std::size_t n = a.size() - 2;
            std::vector<double> values(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
            return static_cast<double>(
                bfdata::search::bisection(a[n], values, static_cast<int>(a[n + 1])));
        }},
        // Search.*_descending family (args: [values..., x, start], same convention as
        // Search.sequential/bisection above but with SortOrder::Descending -- v2.1.4 fixed
        // bisection()'s descending branch, previously dead code that always returned
        // `start`; see search.hpp's file header)
        {"Search.sequential_descending", [](const std::vector<double>& a) {
            std::size_t n = a.size() - 2;
            std::vector<double> values(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
            return static_cast<double>(bfdata::search::sequential(
                a[n], values, static_cast<int>(a[n + 1]), bfdata::SortOrder::Descending));
        }},
        {"Search.bisection_descending", [](const std::vector<double>& a) {
            std::size_t n = a.size() - 2;
            std::vector<double> values(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
            return static_cast<double>(bfdata::search::bisection(
                a[n], values, static_cast<int>(a[n + 1]), bfdata::SortOrder::Descending));
        }},
        // MCMCDiagnostics.MinimumSampleSize (args: [quantile, tolerance, probability] --
        // see fixtures/special_functions/mcmc_diagnostics.json)
        {"MCMCDiagnostics.minimum_sample_size", [](const std::vector<double>& a) {
            return static_cast<double>(
                corehydro::numerics::sampling::mcmc::minimum_sample_size(a[0], a[1], a[2]));
        }},
        // Bilinear.log_floor_* family (args: [x1_query, x2_query] -- see
        // bilinear_log_floor_value() above and fixtures/special_functions/bilinear.json)
        {"Bilinear.log_floor_value", bilinear_log_floor_value},
        // Probability.hpcm_* family (args: see probability_hpcm_joint()/
        // probability_hpcm_conditional_at() above and
        // fixtures/special_functions/probability.json). Task 6 routes hpcm_joint (but not
        // hpcm_conditional_at, which needs the conditional_probabilities out-param the
        // "probability" toolbox group's "joint" method does not expose) through run_toolbox,
        // the same "pin once, exercise the toolbox dispatch too" pattern Correlation.*/
        // Bilinear.log_floor_value use -- so this pinned value is also a cross-language check.
        {"Probability.hpcm_joint",
         [](const std::vector<double>& a) {
             int n = probability_hpcm_n(a.size());
             std::vector<double> p(a.begin(), a.begin() + n);
             std::vector<double> ind(a.begin() + n, a.begin() + 2 * n);
             std::vector<double> corr(a.begin() + 2 * n, a.end());
             return tbx::run_toolbox("probability", "joint", {p, ind, corr},
                                     "{\"dependency\":\"correlation\"}")
                 .values.at(0);
         }},
        {"Probability.hpcm_conditional_at", probability_hpcm_conditional_at},
        // Tools.log10 (args: [x] -- see fixtures/special_functions/tools.json)
        {"Tools.log10", [](const std::vector<double>& a) { return corehydro::numerics::clamped_log10(a[0]); }},
    };
    return t;
}

// Most special_function fixtures dispatch every case through one file-level `target`
// (e.g. "Erf.function"). The Cholesky fixture instead groups several related dispatch
// keys ("Cholesky.determinant", "Cholesky.inverse_element", ...) in one file, so each
// case may override the target; a case without its own `target` falls back to the
// file-level one, preserving the single-target files' existing behavior unchanged.
static void run_special_function(const json& spec) {
    std::string file_target = spec["target"].get<std::string>();
    const auto& table = special_function_table();
    for (const auto& c : spec["cases"]) {
        std::string target = c.value("target", file_target);
        auto it = table.find(target);
        if (it == table.end())
            throw std::runtime_error("unknown special-function target: " + target);
        const auto& fn = it->second;
        std::string name = c["name"].get<std::string>();
        std::vector<double> args;
        for (const auto& v : c["args"]) args.push_back(parse_num(v));
        double actual = fn(args);
        for (const auto& as : c["assertions"]) {
            std::string where = target + "/" + name + "/" + as["method"].get<std::string>();
            check_value(actual, as, where);
        }
    }
}

// --- Generic polymorphic path ----------------------------------------------------------

static dist::ParameterEstimationMethod parse_pe_method(const std::string& m) {
    if (m == "mom") return dist::ParameterEstimationMethod::MethodOfMoments;
    if (m == "lmom") return dist::ParameterEstimationMethod::MethodOfLinearMoments;
    return dist::ParameterEstimationMethod::MaximumLikelihood;
}

static std::unique_ptr<dist::UnivariateDistributionBase> build_generic(const std::string& target,
                                                                       const json& construct,
                                                                       const json& datasets) {
    auto d = dist::create_distribution(target);
    if (construct.contains("params")) {
        std::vector<double> p;
        for (const auto& v : construct["params"]) p.push_back(parse_num(v));
        d->set_parameters(p);
        return d;
    }
    const auto& fit = construct["fit"];
    std::vector<double> data;
    for (const auto& v : datasets[fit["dataset"].get<std::string>()]) data.push_back(v.get<double>());
    auto* est = dynamic_cast<dist::IEstimation*>(d.get());
    if (est == nullptr) throw std::runtime_error(target + " does not support estimation");
    est->estimate(data, parse_pe_method(fit["method"].get<std::string>()));
    return d;
}

// Parses a "dependency" fixture string into a Probability::DependencyType (CompetingRisks).
static prob::DependencyType parse_dependency(const std::string& d) {
    if (d == "Independent") return prob::DependencyType::Independent;
    if (d == "PerfectlyPositive") return prob::DependencyType::PerfectlyPositive;
    if (d == "PerfectlyNegative") return prob::DependencyType::PerfectlyNegative;
    if (d == "CorrelationMatrix") return prob::DependencyType::CorrelationMatrix;
    throw std::runtime_error("unknown dependency type: " + d);
}

// --- Delegation to the shared distribution runner ----------------------------------------
//
// The fixture `construct` schema IS the dist_spec.hpp grammar, so the bridge only has to
// resolve a dataset NAME into an inline array, spell the handful of keys the two schemas
// disagree on, and hand the object to run_dist / run_copula / run_mvdist. Every value the
// runner produces is then pinned by exactly the corpus the bespoke glue was pinned by.
//
// Two properties of the runner keep a narrow bespoke path alive; both are deliberate, and
// both are recorded in the phase-3 task report:
//
//   1. `json_lite`, the runner's JSON reader, has no NaN or Infinity literal (see its file
//      header), while the corpus deliberately pins non-finite-PARAMETER validity cases:
//      Empirical `p`, KernelDensity `bandwidth`, every copula `theta`/`df`, and
//      BivariateEmpirical `x1`/`x2`/`p`. Every such case asserts nothing but
//      `parameters_valid`, so each is built locally and read directly.
//
//   2. The runner is stateless by construction -- one call builds an object, evaluates once
//      and drops it -- and it exposes the user-facing verb set, not the whole fixture
//      vocabulary. Two groups therefore stay on the bespoke dispatcher: methods with no
//      runner counterpart (`mvndst` and its two status arms, `log_multivariate_beta`,
//      `cdf_xy(_after_set_parameters)`, `dependency_change`), and MultivariateNormal's
//      `cdf`/`interval` in a case that consumes the persistent MVNUNI stream more than once,
//      since those pin a SEQUENCE off one object (r_mvtnorm_4d_sequential asserts eleven
//      advancing cdf values). A case that consumes the stream at most once delegates: `seed`
//      is a grammar key, so a rebuilt object starts the same stream.

static bool json_has_non_finite(const json& v) {
    if (v.is_string()) {
        const std::string s = v.get<std::string>();
        return s == "nan" || s == "inf" || s == "-inf";
    }
    if (v.is_array() || v.is_object())
        for (const auto& e : v)
            if (json_has_non_finite(e)) return true;
    return false;
}

// "data" is the only dataset-by-name key in the univariate grammar (KernelDensity); "base"
// and "components" nest.
static json inline_datasets(const json& construct, const json& datasets) {
    json out = construct;
    if (out.contains("data") && out["data"].is_string())
        out["data"] = datasets[out["data"].get<std::string>()];
    if (out.contains("base")) out["base"] = inline_datasets(out["base"], datasets);
    if (out.contains("components"))
        for (auto& c : out["components"]) c = inline_datasets(c, datasets);
    return out;
}

static json args_array(const json& as) {
    return as.contains("args") && as["args"].is_array() ? as["args"] : json::array();
}

static supp::JsonValue to_spec(const json& j) {
    return corehydro::models::spec::parse_json(j.dump());
}

// The fixture method vocabulary predates the runner's; map the differences in one place.
// `random_value` args are [sample_size, seed, index]: the runner's "random" reads only the
// first two and returns the whole draw, so the args pass through unchanged and fixture_pick
// does the indexing.
static std::string fixture_method(const std::string& m) {
    if (m == "param") return "parameters";
    if (m == "random_value") return "random";
    return m;  // pdf, log_pdf, cdf, quantile, mean, ..., log_likelihood pass straight through
}

// `param` and `random_value` index into the vector the runner returns whole.
static double fixture_pick(const supp::DistResult& r, const std::string& m, const json& a) {
    if (m == "param") return r.values.at(static_cast<std::size_t>(a[0].get<int>()));
    if (m == "random_value") return r.values.at(static_cast<std::size_t>(a[2].get<int>()));
    return r.values.at(0);
}

// Limitation 1 above: the two composite families whose fixture cases carry a non-finite
// literal the runner's JSON reader cannot encode. Built here so those validity cases still
// assert; every other composite is built by the shared spec builder.
static std::unique_ptr<dist::UnivariateDistributionBase> build_non_finite_composite(
    const std::string& target, const json& construct) {
    if (target == "Empirical") {
        std::vector<double> xv, pv;
        for (const auto& v : construct["x"]) xv.push_back(parse_num(v));
        for (const auto& v : construct["p"]) pv.push_back(parse_num(v));
        auto pt = dist::EmpiricalTransform::NormalZ;
        if (construct.contains("p_transform")) {
            std::string t = construct["p_transform"].get<std::string>();
            if (t == "None") pt = dist::EmpiricalTransform::None;
            else if (t == "NormalZ") pt = dist::EmpiricalTransform::NormalZ;
            else throw std::runtime_error("unknown p_transform: " + t);
        }
        // p_descending DECLARES the probability order (mirrors C#'s explicit `probabilityOrder`
        // argument -- NOT auto-detected); false is the ordinary ascending-CDF case.
        bool p_descending = construct.value("p_descending", false);
        return std::make_unique<dist::EmpiricalDistribution>(std::move(xv), std::move(pv), pt,
                                                              p_descending);
    }
    if (target == "KernelDensity") {
        std::vector<double> data;
        for (const auto& v : construct["data"]) data.push_back(parse_num(v));
        std::string kernel_str = construct.value("kernel", std::string("Gaussian"));
        dist::KernelType kt = dist::KernelType::Gaussian;
        if      (kernel_str == "Epanechnikov") kt = dist::KernelType::Epanechnikov;
        else if (kernel_str == "Gaussian")     kt = dist::KernelType::Gaussian;
        else if (kernel_str == "Triangular")   kt = dist::KernelType::Triangular;
        else if (kernel_str == "Uniform")      kt = dist::KernelType::Uniform;
        else throw std::runtime_error("unknown kernel type: " + kernel_str);
        std::unique_ptr<dist::KernelDensity> kde;
        if (construct.contains("bandwidth"))
            // parse_num (not .get<double>()) so a "nan"/"inf" string literal (the v2.1.4
            // Bandwidth NaN/Infinity-rejection case) parses instead of throwing a JSON
            // type_error.
            kde = std::make_unique<dist::KernelDensity>(std::move(data), kt,
                                                        parse_num(construct["bandwidth"]));
        else
            kde = std::make_unique<dist::KernelDensity>(std::move(data), kt);
        if (construct.contains("bounded_by_data"))
            kde->set_bounded_by_data(construct["bounded_by_data"].get<bool>());
        return kde;
    }
    throw std::runtime_error("composite target '" + target +
                             "' has a non-finite construct the shared spec grammar cannot "
                             "encode and no local builder here");
}

static bool is_composite_target(const std::string& target) {
    return target == "TruncatedDistribution" || target == "Empirical"
        || target == "KernelDensity" || target == "Mixture" || target == "CompetingRisks";
}

// Non-const: "set_parameters" (below) mutates `d` in place. Every other branch only calls
// const accessors, so this is a pure widening of what the reference can do, not a behavior
// change for any existing caller (the sole call site in run_generic already holds `d` via a
// non-const std::unique_ptr).
static double dispatch_generic(dist::UnivariateDistributionBase& d, const std::string& m,
                               const json& a) {
    // Mutates the already-built `d` in place with a new flat parameter vector, mirroring the
    // C# SetParameters entry point -- lets a case exercise a "construct valid -> SetParameters
    // invalid -> recheck -> SetParameters valid -> recheck" sequence on ONE persistent object
    // (needed for TruncatedDistribution's parameter-validity fixture; the flat, non-composite
    // targets in this same validation wave don't need it -- their construct+params already IS
    // a fresh-construct-then-SetParameters call, see gumbel.json/etc.). Returns a dummy value;
    // pair with a mode:"equal", expected:0 assertion and check the resulting state with a
    // separate "parameters_valid"/"param" assertion right after.
    if (m == "set_parameters") {
        std::vector<double> p;
        for (const auto& v : a) p.push_back(parse_num(v));
        d.set_parameters(p);
        return 0.0;
    }
    // CompetingRisks-only: verifies the v2.1.4 Dependency setter fix (changing Dependency
    // mid-lifetime invalidates the cached MVN) and that PerfectlyNegative no longer zeroes
    // the public CorrelationMatrix. ONE self-contained call (mirrors BivariateEmpirical's
    // cdf_xy_after_set_parameters -- works identically whether a runner holds the
    // persistent `d` across a case's assertions, like this one, or rebuilds fresh per
    // dispatch, like R/Python): CDF under the CURRENT dependency, read back
    // correlation_matrix()[i, j], switch to `dependency2`, CDF again -- returns the value
    // named by `field` ("cdf1"/"correlation"/"cdf2"). args = [x, dependency2, i, j, field].
    if (m == "dependency_change") {
        auto& cr = dynamic_cast<dist::CompetingRisks&>(d);
        double x = a[0].get<double>();
        double cdf1 = cr.cdf(x);
        double corr_ij = cr.correlation_matrix()[static_cast<std::size_t>(a[2].get<int>())]
                                                  [static_cast<std::size_t>(a[3].get<int>())];
        cr.set_dependency(parse_dependency(a[1].get<std::string>()));
        double cdf2 = cr.cdf(x);
        std::string field = a[4].get<std::string>();
        if (field == "cdf1") return cdf1;
        if (field == "correlation") return corr_ij;
        if (field == "cdf2") return cdf2;
        throw std::runtime_error("unknown dependency_change field: " + field);
    }
    if (m == "mean") return d.mean();
    if (m == "median") return d.median();
    if (m == "mode") return d.mode();
    if (m == "sd") return d.standard_deviation();
    if (m == "skewness") return d.skewness();
    if (m == "kurtosis") return d.kurtosis();
    if (m == "minimum") return d.minimum();
    if (m == "maximum") return d.maximum();
    if (m == "pdf") return d.pdf(a[0].get<double>());
    if (m == "log_pdf") return d.log_pdf(a[0].get<double>());
    if (m == "cdf") return d.cdf(a[0].get<double>());
    if (m == "quantile") return d.inverse_cdf(a[0].get<double>());
    if (m == "param") return d.get_parameters()[a[0].get<int>()];
    if (m == "random_value") {
        // args: [sample_size, seed, index] -- one draw from the seeded MT stream.
        return d.generate_random_values(a[0].get<int>(), a[1].get<int>())[a[2].get<int>()];
    }
    if (m == "linear_moment") {
        const auto* lm = dynamic_cast<const dist::ILinearMomentEstimation*>(&d);
        if (lm == nullptr) throw std::runtime_error("distribution has no L-moments");
        return lm->linear_moments_from_parameters(d.get_parameters())[a[0].get<int>()];
    }
    // The IStandardError surface, reached by the same capability cast `linear_moment` uses.
    // Args mirror the GEV slice's flattened convention above (dispatch_gev), so one fixture
    // vocabulary covers both the bespoke GEV path and every factory-built family:
    // parameter_covariance [sample_size, row, col], quantile_variance [probability,
    // sample_size], quantile_gradient [probability, index]. The estimation method is fixed at
    // MaximumLikelihood, matching dist_runner.hpp's arms.
    if (m == "parameter_covariance" || m == "quantile_variance" || m == "quantile_gradient") {
        const auto* se = dynamic_cast<const dist::IStandardError*>(&d);
        if (se == nullptr)
            throw std::runtime_error("distribution does not implement IStandardError: " + m);
        const auto mle = dist::ParameterEstimationMethod::MaximumLikelihood;
        if (m == "parameter_covariance")
            return se->parameter_covariance(a[0].get<int>(), mle)[a[1].get<int>()]
                                                                 [a[2].get<int>()];
        if (m == "quantile_variance")
            return se->quantile_variance(a[0].get<double>(), a[1].get<int>(), mle);
        return se->quantile_gradient(a[0].get<double>())[a[1].get<int>()];
    }
    // Static GammaDistribution utility, not tied to `d`'s own parameters -- args:
    // [skewness, probability]. Only meaningful for target "GammaDistribution", but the
    // call itself doesn't touch `d` at all (mirrors the emitter's Dispatch "partial_kp").
    if (m == "partial_kp")
        return dist::GammaDistribution::partial_kp(a[0].get<double>(), a[1].get<double>());
    throw std::runtime_error("unknown fixture method: " + m);
}

// Composite targets go through the shared spec grammar + run_dist; the 38 flat families keep
// build_generic/dispatch_generic (their `fit` construct is a fixture-only convention the
// grammar does not carry).
static void run_generic(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    bool composite = is_composite_target(target);
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        if (composite) {
            json cspec = inline_datasets(c["construct"], datasets);
            cspec["family"] = target;
            // See limitation 1 above: a construct carrying a "nan"/"inf" literal cannot be
            // serialized into the grammar, so its (validity-only) assertions run locally.
            const bool encodable = !json_has_non_finite(c["construct"]);
            // Built lazily and held for the whole case, so the bespoke arms see one object in
            // assertion order exactly as they did before.
            std::unique_ptr<dist::UnivariateDistributionBase> local;
            auto local_object = [&]() -> dist::UnivariateDistributionBase& {
                if (!local)
                    local = encodable ? supp::build_univariate(to_spec(cspec))
                                      : build_non_finite_composite(target, cspec);
                return *local;
            };
            for (const auto& as : c["assertions"]) {
                std::string method = as["method"].get<std::string>();
                json args = args_array(as);
                std::string where = target + "/" + name + "/" + method;
                const bool is_bool = as["mode"].get<std::string>() == "bool";
                // `dependency_change` has no runner counterpart (limitation 2); a non-finite
                // evaluation point is limitation 1 applied to the args rather than the
                // construct (no composite case does this today, unlike the multivariate
                // log_pdf-at-infinity cases, but the routing is the same).
                if (!encodable || method == "dependency_change" || json_has_non_finite(args)) {
                    if (is_bool)
                        check_bool(local_object().parameters_valid(), as, where);
                    else
                        check_value(dispatch_generic(local_object(), method, args), as, where);
                    continue;
                }
                if (method == "set_parameters") {
                    // The runner is stateless, so a SetParameters round trip is carried on the
                    // spec: every later assertion in this case rebuilds with it applied, which
                    // is what dist_spec's "set_parameters" key exists for. A second
                    // set_parameters simply replaces the first, exactly as the in-place
                    // mutation did. The 0.0 mirrors dispatch_generic's dummy return.
                    cspec["set_parameters"] = as["args"];
                    check_value(0.0, as, where);
                    continue;
                }
                if (is_bool) {
                    // The old dispatcher ignored the assertion's method for bool mode and read
                    // parameters_valid(); keep that exactly.
                    auto r = supp::run_dist(cspec.dump(), "parameters_valid", "[]");
                    check_bool(r.values.at(0) != 0.0, as, where);
                } else {
                    auto r = supp::run_dist(cspec.dump(), fixture_method(method), args.dump());
                    check_value(fixture_pick(r, method, args), as, where);
                }
            }
            continue;
        }
        std::unique_ptr<dist::UnivariateDistributionBase> d =
            build_generic(target, c["construct"], datasets);
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = target + "/" + name + "/" + method;
            if (as["mode"].get<std::string>() == "bool")
                check_bool(d->parameters_valid(), as, where);
            else
                check_value(dispatch_generic(*d, method, args), as, where);
        }
    }
}

// --- goodness_of_fit path -------------------------------------------------------------

static std::string format_g17(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}

// Delegates to numerics/support/toolbox_runner.hpp so a fixture case and a user's
// goodness_of_fit() call are one code path. The function-name-to-method mapping is this file's
// only remaining knowledge of the group.
static double dispatch_gof(const std::string& fn, const std::vector<double>& args,
                            const std::vector<double>& obs, const std::vector<double>& mod) {
    static const std::map<std::string, std::string> kMethods = {
        {"MSE", "mse"}, {"MAE", "mae"},
        {"NashSutcliffeEfficiency", "nse"},
        {"KlingGuptaEfficiency", "kge"}, {"KlingGuptaEfficiencyMod", "kge_mod"},
        {"PBIAS", "pbias"}, {"RSR", "rsr"},
        {"IndexOfAgreement", "d"}, {"ModifiedIndexOfAgreement", "d_mod"},
        {"RefinedIndexOfAgreement", "d_ref"}, {"VolumetricEfficiency", "ve"},
    };
    if (fn == "AIC")
        return tbx::run_toolbox("gof", "aic", {},
                                "{\"k\":" + std::to_string(static_cast<int>(args[0])) +
                                    ",\"log_likelihood\":" + format_g17(args[1]) + "}").values[0];
    if (fn == "AICc" || fn == "BIC")
        return tbx::run_toolbox("gof", fn == "AICc" ? "aicc" : "bic", {},
                                "{\"n\":" + std::to_string(static_cast<int>(args[0])) +
                                    ",\"k\":" + std::to_string(static_cast<int>(args[1])) +
                                    ",\"log_likelihood\":" + format_g17(args[2]) + "}").values[0];
    auto it = kMethods.find(fn);
    if (it == kMethods.end())
        throw std::runtime_error("unknown goodness_of_fit function: " + fn);
    return tbx::run_toolbox("gof", it->second, {obs, mod}, "{}").values[0];
}

static void run_goodness_of_fit(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        std::string fn = c["function"].get<std::string>();
        std::vector<double> args;
        if (c.contains("args"))
            for (const auto& v : c["args"]) args.push_back(parse_num(v));
        std::vector<double> obs, mod;
        if (c.contains("observed_dataset"))
            for (const auto& v : datasets[c["observed_dataset"].get<std::string>()])
                obs.push_back(v.get<double>());
        if (c.contains("modeled_dataset"))
            for (const auto& v : datasets[c["modeled_dataset"].get<std::string>()])
                mod.push_back(v.get<double>());
        double actual = dispatch_gof(fn, args, obs, mod);
        for (const auto& as : c["assertions"]) {
            std::string where = "gof/" + name;
            check_value(actual, as, where);
        }
    }
}

// --- data_utility path ------------------------------------------------------------------
// Small data-statistics utilities: MGBT count, Box-Cox / Yeo-Johnson lambda + transform,
// plotting positions, Latin hypercube sampling. Same flat shape as goodness_of_fit
// (one function per case, single computed value checked against each assertion).

static double dispatch_data_utility(const std::string& fn, const std::vector<double>& args,
                                    const std::vector<double>& data) {
    namespace nd = corehydro::numerics::data;
    if (fn == "MGBT") return static_cast<double>(nd::MultipleGrubbsBeckTest::function(data));
    if (fn == "BoxCoxLambda") return nd::BoxCox::fit_lambda(data);
    if (fn == "BoxCoxTransform")
        return nd::BoxCox::transform(data, args[0])[static_cast<std::size_t>(args[1])];
    if (fn == "YeoJohnsonLambda") return nd::YeoJohnson::fit_lambda(data);
    if (fn == "YeoJohnsonTransform")
        return nd::YeoJohnson::transform(data, args[0])[static_cast<std::size_t>(args[1])];
    if (fn == "PlottingPosition")
        return nd::plotting_positions::function(static_cast<int>(args[0]),
                                                args[1])[static_cast<std::size_t>(args[2])];
    if (fn == "LHSRandom" || fn == "LHSMedian") {
        // args: [sample_size, dimension, seed, row, col]
        auto m = fn == "LHSRandom"
            ? corehydro::numerics::sampling::LatinHypercube::random(
                  static_cast<int>(args[0]), static_cast<int>(args[1]), static_cast<int>(args[2]))
            : corehydro::numerics::sampling::LatinHypercube::median(
                  static_cast<int>(args[0]), static_cast<int>(args[1]), static_cast<int>(args[2]));
        return m[static_cast<std::size_t>(args[3])][static_cast<std::size_t>(args[4])];
    }
    // Threshold-selection diagnostics (models/data_frame/threshold_diagnostics.hpp). Both take
    // args [u_min, u_max, n_thresholds, confidence_level, point_index]; the function name picks
    // the field. `*PointCount` ignores point_index and returns how many candidate thresholds
    // survived the minimum-exceedance and fit filters.
    if (fn.rfind("MRL", 0) == 0 || fn.rfind("GPDStability", 0) == 0) {
        auto u_min = args[0];
        auto u_max = args[1];
        auto n = static_cast<int>(args[2]);
        auto cl = args[3];
        auto i = args.size() > 4 ? static_cast<std::size_t>(args[4]) : 0;
        if (fn.rfind("MRL", 0) == 0) {
            auto r = corehydro::models::ThresholdDiagnostics::compute_mean_residual_life(
                data, u_min, u_max, n, cl);
            if (fn == "MRLPointCount") return static_cast<double>(r.points.size());
            const auto& pt = r.points.at(i);
            if (fn == "MRLThreshold") return pt.threshold;
            if (fn == "MRLMeanExcess") return pt.mean_excess;
            if (fn == "MRLLowerCI") return pt.lower_ci;
            if (fn == "MRLUpperCI") return pt.upper_ci;
            if (fn == "MRLCount") return pt.exceedance_count;
        } else {
            auto r = corehydro::models::ThresholdDiagnostics::compute_parameter_stability(
                data, u_min, u_max, n, cl);
            if (fn == "GPDStabilityPointCount") return static_cast<double>(r.points.size());
            const auto& pt = r.points.at(i);
            if (fn == "GPDStabilityThreshold") return pt.threshold;
            if (fn == "GPDStabilityModifiedScale") return pt.modified_scale;
            if (fn == "GPDStabilityModifiedScaleLowerCI") return pt.modified_scale_lower_ci;
            if (fn == "GPDStabilityModifiedScaleUpperCI") return pt.modified_scale_upper_ci;
            if (fn == "GPDStabilityShape") return pt.shape;
            if (fn == "GPDStabilityShapeLowerCI") return pt.shape_lower_ci;
            if (fn == "GPDStabilityShapeUpperCI") return pt.shape_upper_ci;
            if (fn == "GPDStabilityCount") return pt.exceedance_count;
        }
    }
    throw std::runtime_error("unknown data_utility function: " + fn);
}

static void run_data_utility(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        std::string fn = c["function"].get<std::string>();
        std::vector<double> args;
        if (c.contains("args"))
            for (const auto& v : c["args"]) args.push_back(parse_num(v));
        std::vector<double> data;
        if (c.contains("dataset"))
            // parse_num (not v.get<double>()) so a "nan"/"inf"/"-inf" string literal inside
            // a dataset (the v2.1.4 FitLambda invalid-sample cases) parses instead of
            // throwing a JSON type_error.
            for (const auto& v : datasets[c["dataset"].get<std::string>()]) data.push_back(parse_num(v));
        double actual = dispatch_data_utility(fn, args, data);
        for (const auto& as : c["assertions"]) {
            std::string where = "data_utility/" + name;
            check_value(actual, as, where);
        }
    }
}

// --- toolbox path -----------------------------------------------------------------------
// Every Numerics utility group (correlation, goodness of fit, statistics, interpolation, ...)
// runs through numerics/support/toolbox_runner.hpp. A case carries its data vectors and an
// options object; each assertion names a method and selects one value out of the result.

static std::vector<std::vector<double>> toolbox_data(const json& c, const json& datasets) {
    std::vector<std::vector<double>> out;
    if (!c.contains("data")) return out;
    for (const auto& d : c["data"]) {
        std::vector<double> v;
        if (d.is_string()) {
            for (const auto& e : datasets[d.get<std::string>()]) v.push_back(parse_num(e));
        } else {
            for (const auto& e : d) v.push_back(parse_num(e));
        }
        out.push_back(std::move(v));
    }
    return out;
}

static double toolbox_select(const tbx::ToolboxResult& r, const json& as, const std::string& group) {
    std::string select = as.value("select", std::string("value"));
    if (select == "length") return static_cast<double>(r.values.size());
    if (select == "rows") {
        if (r.dims.empty())
            throw std::runtime_error("toolbox select 'rows' has no dims (group '" + group + "')");
        return static_cast<double>(r.dims[0]);
    }
    if (select == "columns") {
        if (r.dims.size() < 2)
            throw std::runtime_error("toolbox select 'columns' has no dims (group '" + group + "')");
        return static_cast<double>(r.dims[1]);
    }
    std::size_t i = 0;
    if (as.contains("label")) {
        std::string label = as["label"].get<std::string>();
        bool found = false;
        for (std::size_t k = 0; k < r.names.size(); ++k)
            if (r.names[k] == label) { i = k; found = true; break; }
        if (!found) throw std::runtime_error("toolbox result has no label '" + label + "'");
    } else {
        i = static_cast<std::size_t>(as.value("index", 0));
    }
    if (i >= r.values.size())
        throw std::runtime_error("toolbox result index out of range");
    return r.values[i];
}

// Set once in main() from argv[1] (the fixtures directory); the Sobol direction-numbers file
// lives at core/data/ next to it. Path resolution is a wrapper concern (see
// numerics/support/toolbox/sampling.hpp's file header) -- fixtures/toolbox/sampling.json's own
// `options` never carries a `path` key, so this harness injects its own resolved path before
// calling run_toolbox, the same way corehydror's sobol_sequence() and corehydropy's
// sobol_sequence() resolve theirs.
static std::string g_sobol_path;

static void run_toolbox_kind(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    std::string group = spec["group"].get<std::string>();
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        auto data = toolbox_data(c, datasets);
        json options = c.contains("options") ? c["options"] : json::object();
        // Only the "sobol" method reads a path (SobolSequence's constructor only touches it
        // when dimension > 1); "stratify" never does, so this stays scoped to that one method
        // rather than the whole "sampling" group.
        bool is_sobol = !c["assertions"].empty() &&
                        c["assertions"][0]["method"].get<std::string>() == "sobol";
        if (group == "sampling" && is_sobol) options["path"] = g_sobol_path;
        std::string options_str = options.dump();
        for (const auto& as : c["assertions"]) {
            std::string where = "toolbox/" + group + "/" + name;
            auto r = tbx::run_toolbox(group, as["method"].get<std::string>(), data, options_str);
            check_value(toolbox_select(r, as, group), as, where);
        }
    }
}

// --- optimizer path (Task 8) -------------------------------------------------------------
//
// fixtures/toolbox/optimizers.json cases name a built-in objective by the same name
// TestFunctions.cs uses (DeJong/FXYZ/Booth/McCormick/FX/...); fixture_catalog::optimizer_objective
// maps that name to the real C++ function. `construct` is passed straight through to
// tbx::run_optimizer as the spec JSON (minus the "objective" key, which is not part of the
// runner's own grammar -- see optimizer_runner.hpp's file header on why no objective registry
// lives there).
static void run_optimizer_kind(const json& spec) {
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        json construct = c["construct"];
        std::string objective_name = construct.value("objective", "DeJong");
        construct.erase("objective");
        tbx::OptimResult r =
            tbx::run_optimizer(construct.dump(), fixture_catalog::optimizer_objective(objective_name));
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            std::string where = "optimizer/" + name + "/" + method;
            if (method == "value") {
                check_value(r.value, as, where);
            } else if (method == "parameter") {
                std::size_t i = static_cast<std::size_t>(as["args"][0].get<int>());
                check_value(r.parameters.at(i), as, where);
            } else if (method == "status") {
                if (r.status == as["expected"].get<std::string>())
                    chtest::report_pass();
                else
                    chtest::report_fail(__FILE__, __LINE__,
                                        where + ": expected status " +
                                            as["expected"].get<std::string>() + ", got " + r.status);
            } else {
                throw std::runtime_error("unknown optimizer fixture assertion method: " + method);
            }
        }
    }
}

// --- callback path (callback surface, Task 1) --------------------------------------------
//
// fixtures/callback/*.json cases name a built-in callback by the same name the fixture's own
// `callbacks` block documents; fixture_catalog::callback_set maps that name to a real C++ lambda --
// the C++ analogue of the native closures the R and Python fixture runners (and the dotnet
// emitter's delegates) write for the same names, so every case exercises the real host-language
// callback path callback_runner.hpp exists to protect. The catalog itself lives in
// fixture_callback_catalog.cpp, which is compiled with -ffp-contract=off while this file is not;
// see that file's header and core/CMakeLists.txt.

// One callback case: build the delegate set the group needs out of the catalog, run it, and
// apply the case's assertions. Factored out of run_callback_kind so run_callback_cross_language_kind
// (below) can drive the identical path for each of its nested sub-blocks -- the cross-language
// fixture nests two blocks of this exact shape and must not grow an evaluation path of its own.
static void run_one_callback_case(const std::string& where_prefix, const json& construct,
                                  const json& assertions) {
    tbx::CallbackSet cbs;
    fixture_catalog::callback_set(construct["callback"].get<std::string>(), cbs);
    // The two other delegates the mcmc group's samplers take, each resolved out of the same
    // catalog: a Gibbs proposal and an HMC/NUTS gradient. Absent keys leave the set's members
    // empty, which is what "no proposal" and "the ported default gradient" mean.
    if (construct.contains("proposal"))
        fixture_catalog::callback_set(construct["proposal"].get<std::string>(), cbs);
    if (construct.contains("gradient"))
        fixture_catalog::callback_set(construct["gradient"].get<std::string>(), cbs);
    // The bootstrap group's other three delegates and the gmm group's two optional ones,
    // resolved out of the same catalog (each group's `callback` key names its own required
    // delegate: the resample, or the moment conditions). An absent key means that delegate is
    // not supplied, which is what "no jackknife", "the ported numerical Jacobian" and "no
    // penalty" mean. "df" (P2 "math extras") is math/root_find_newton's second callback, the
    // analytic derivative alongside `callback`'s own f; "jacobian" doubles as math/root_find_system's
    // J, reusing the same key gmm's jacobian already uses since both resolve to `cbs.vector_matrix`.
    for (const char* key :
         {"fit", "fit_with_covariance", "statistic", "jackknife", "jacobian", "penalty", "df"})
        if (construct.contains(key)) fixture_catalog::callback_set(construct[key].get<std::string>(), cbs);
    json options = construct.contains("options") ? construct["options"] : json::object();
    tbx::CallbackResult r =
        tbx::run_callback(construct["group"].get<std::string>(),
                          construct["method"].get<std::string>(), options.dump(), cbs);
    for (const auto& as : assertions) {
        std::string method = as["method"].get<std::string>();
        std::string where = where_prefix + "/" + method;
        std::size_t i = as.contains("args") ? static_cast<std::size_t>(as["args"][0].get<int>())
                                            : std::size_t{0};
        if (method == "value") {
            check_value(r.values.at(i), as, where);
        } else if (method == "named") {
            // Reads a value by the label the group gave it rather than by position: the mcmc
            // group's summary block is long and its indices shift with the chain and parameter
            // counts, so "posterior_mean[0]" says what it pins and "12" does not.
            std::string want = as["name"].get<std::string>();
            auto it = std::find(r.names.begin(), r.names.end(), want);
            if (it == r.names.end())
                throw std::runtime_error(where + ": no result named '" + want + "'");
            check_value(r.values.at(static_cast<std::size_t>(it - r.names.begin())), as, where);
        } else if (method == "dim") {
            check_value(static_cast<double>(r.dims.at(i)), as, where);
        } else if (method == "status") {
            if (r.status == as["expected"].get<std::string>())
                chtest::report_pass();
            else
                chtest::report_fail(__FILE__, __LINE__,
                                    where + ": expected status " +
                                        as["expected"].get<std::string>() + ", got " + r.status);
        } else {
            throw std::runtime_error("unknown callback fixture assertion method: " + method);
        }
    }
}

static void run_callback_kind(const json& spec) {
    for (const auto& c : spec["cases"]) {
        run_one_callback_case("callback/" + c["name"].get<std::string>(), c["construct"],
                              c["assertions"]);
    }
}

// --- callback_cross_language path (Task 8) -----------------------------------------------
//
// a case of fixtures/callback/callback_cross_language.json nests its sub-blocks -- EVERY key but
// "name", each shaped exactly like a "callback"-kind case's construct/assertions -- under one case
// name, because the file's job is proving they reproduce identically across languages in ONE
// guarantee rather than in a file each. (The second case adds a purpose: its two callbacks carry
// the contractible arithmetic that makes core/CMakeLists.txt's -ffp-contract=off on the catalog a
// flag this suite can detect the loss of.) The labels are the case's own -- "mcmc", "bootstrap",
// "pivotal" -- and are read OFF the case rather than listed here, so a case may nest one block or
// five without a runner change; only "name" is reserved. Reuses run_one_callback_case verbatim; no
// new evaluation logic, just the nesting. The assertions there are spelled mode "abs" with tol 0,
// so this runner's agreement with the R, Python and C# ones is bit equality rather than a tolerance.
static void run_callback_cross_language_kind(const json& spec) {
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        for (auto it = c.begin(); it != c.end(); ++it) {
            if (it.key() == "name") continue;
            const json& block = it.value();
            run_one_callback_case("callback_cross_language/" + name + "/" + it.key(),
                                  block["construct"], block["assertions"]);
        }
    }
}

// --- toolbox_cross_language path (Task 9) -----------------------------------------------
//
// fixtures/toolbox/toolbox_cross_language.json's one case nests three sub-blocks -- "optimizer"
// (shaped exactly like an "optimizer"-kind case's construct/assertions), "sobol" and "stratify"
// (each shaped exactly like a "toolbox"-kind group-"sampling" case's options/assertions) -- under
// one case name, because the fixture's whole job is proving all three reproduce identically
// across languages in ONE guarantee rather than three separate files. Reuses
// fixture_catalog::optimizer_objective and tbx::run_toolbox/toolbox_select (below) verbatim; no
// new evaluation logic, just the nesting.
static void run_toolbox_cross_language_kind(const json& spec) {
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();

        // optimizer sub-block
        {
            json construct = c["optimizer"]["construct"];
            std::string objective_name = construct.value("objective", "DeJong");
            construct.erase("objective");
            tbx::OptimResult r =
                tbx::run_optimizer(construct.dump(), fixture_catalog::optimizer_objective(objective_name));
            for (const auto& as : c["optimizer"]["assertions"]) {
                std::string method = as["method"].get<std::string>();
                std::string where = "toolbox_cross_language/" + name + "/optimizer/" + method;
                if (method == "value") {
                    check_value(r.value, as, where);
                } else if (method == "parameter") {
                    std::size_t i = static_cast<std::size_t>(as["args"][0].get<int>());
                    check_value(r.parameters.at(i), as, where);
                } else if (method == "status") {
                    if (r.status == as["expected"].get<std::string>())
                        chtest::report_pass();
                    else
                        chtest::report_fail(__FILE__, __LINE__,
                                            where + ": expected status " +
                                                as["expected"].get<std::string>() + ", got " + r.status);
                } else {
                    throw std::runtime_error(
                        "unknown toolbox_cross_language optimizer assertion method: " + method);
                }
            }
        }
        // sobol / stratify sub-blocks: group "sampling", no positional data, options only.
        for (const char* sub : {"sobol", "stratify"}) {
            json options = c[sub].value("options", json::object());
            if (std::string(sub) == "sobol") options["path"] = g_sobol_path;
            std::string options_str = options.dump();
            for (const auto& as : c[sub]["assertions"]) {
                std::string where = "toolbox_cross_language/" + name + "/" + sub;
                auto r = tbx::run_toolbox("sampling", sub, {}, options_str);
                check_value(toolbox_select(r, as, "sampling"), as, where);
            }
        }
    }
}

// --- multivariate_distribution path -----------------------------------------------------
//
// The only partly delegated path. run_mvdist covers the verbs this phase exposes -- see
// mv_delegated below -- and the rest of the pinned surface stays on dispatch_multivariate,
// which dynamic_casts to the concrete type because multivariate targets share no arithmetic
// surface beyond dimension/pdf/log_pdf/cdf/parameters_valid (no factory, no common
// Mean/Variance/Covariance signature across Dirichlet/Multinomial/BivariateEmpirical).
// Extending the runner's method table shrinks the bespoke half; nothing else has to move.

static std::vector<double> parse_num_vec(const json& arr) {
    std::vector<double> v;
    for (const auto& e : arr) v.push_back(parse_num(e));
    return v;
}

static std::vector<int> parse_int_vec(const json& arr) {
    std::vector<int> v;
    for (const auto& e : arr) v.push_back(e.get<int>());
    return v;
}

// Translates a fixture multivariate construct into the dist_spec grammar. The only schema
// difference is Multinomial's parameter spelling (n / p here, trials / probabilities there);
// the four MultivariateNormal integrator settings (seed / max_evaluations / abs_error /
// rel_error) are grammar keys now and pass straight through.
static json mvdist_spec(const std::string& target, const json& construct) {
    json out = construct;
    out["family"] = target;
    if (target == "Multinomial") {
        out.erase("n");
        out.erase("p");
        out["trials"] = construct["n"];
        out["probabilities"] = construct["p"];
    }
    return out;
}

// The object the bespoke arms use: built through the shared spec builder wherever the grammar
// reaches, which now includes the four MultivariateNormal integrator settings (`seed`,
// `max_evaluations`, `abs_error`, `rel_error`). BivariateEmpirical's non-finite validity cases
// are limitation 1 and are built directly.
static std::unique_ptr<dist::MultivariateDistribution> build_multivariate_local(
    const std::string& target, const json& construct) {
    if (target == "BivariateEmpirical" && json_has_non_finite(construct)) {
        std::vector<double> x1 = parse_num_vec(construct["x1"]);
        std::vector<double> x2 = parse_num_vec(construct["x2"]);
        std::vector<std::vector<double>> p;
        for (const auto& row : construct["p"]) p.push_back(parse_num_vec(row));
        auto parse_transform = [&](const char* key) {
            bfdata::Transform t = bfdata::Transform::None;
            if (!construct.contains(key)) return t;
            std::string str = construct[key].get<std::string>();
            if (str == "None") t = bfdata::Transform::None;
            else if (str == "Logarithmic") t = bfdata::Transform::Logarithmic;
            else if (str == "NormalZ") t = bfdata::Transform::NormalZ;
            else throw std::runtime_error("unknown transform: " + str);
            return t;
        };
        return std::make_unique<dist::BivariateEmpirical>(
            std::move(x1), std::move(x2), std::move(p), parse_transform("x1_transform"),
            parse_transform("x2_transform"), parse_transform("p_transform"));
    }
    return supp::build_multivariate(to_spec(mvdist_spec(target, construct)));
}

// Shared lookup for the "random_value"/"lhs_value" seeded-sampling oracle methods, common
// to every multivariate target that implements generate_random_values (all four) /
// latin_hypercube_random_values (MultivariateNormal, MultivariateStudentT only -- see
// fixtures/README.md). args = [sample_size, seed, row, col]: construct a FRESH draw (the
// method itself seeds its own MersenneTwister from `seed`, so this is stateless -- no
// persistent-instance batching needed, unlike MultivariateNormal's MVNUNI-seeded cdf/
// interval/mvndst path above).
template <typename Dist>
static double random_value_at(const Dist& d, const json& a) {
    auto sample = d.generate_random_values(a[0].get<int>(), a[1].get<int>());
    return sample[static_cast<std::size_t>(a[2].get<int>())][static_cast<std::size_t>(a[3].get<int>())];
}

template <typename Dist>
static double lhs_value_at(const Dist& d, const json& a) {
    auto sample = d.latin_hypercube_random_values(a[0].get<int>(), a[1].get<int>());
    return sample[static_cast<std::size_t>(a[2].get<int>())][static_cast<std::size_t>(a[3].get<int>())];
}

// Non-const: BivariateEmpirical's "set_parameters" (below) mutates `d` in place, mirroring
// dispatch_generic's own "set_parameters" precedent -- lets a case exercise the v2.1.4
// stale-cache fix ("construct -> cdf -> set_parameters (new grid) -> cdf again") on ONE
// persistent object. Every other branch only calls const accessors, so this is a pure
// widening of what the reference can do (the sole call site in run_multivariate already
// holds `d` via a non-const std::unique_ptr).
static double dispatch_multivariate(dist::MultivariateDistribution& d, const std::string& target,
                                    const std::string& m, const json& a) {
    if (m == "dimension") return d.dimension();
    if (m == "pdf") return d.pdf(parse_num_vec(a[0]));
    if (m == "log_pdf") return d.log_pdf(parse_num_vec(a[0]));
    if (m == "cdf") return d.cdf(parse_num_vec(a[0]));

    if (target == "Dirichlet") {
        const auto& dd = dynamic_cast<const dist::Dirichlet&>(d);
        if (m == "alpha") return dd.alpha()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "alpha_sum") return dd.alpha_sum();
        if (m == "mean") return dd.mean()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "variance") return dd.variance()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "mode") return dd.mode()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "covariance") return dd.covariance(a[0].get<int>(), a[1].get<int>());
        if (m == "log_multivariate_beta") return dist::Dirichlet::log_multivariate_beta(parse_num_vec(a));
        if (m == "random_value") return random_value_at(dd, a);
    } else if (target == "Multinomial") {
        const auto& mm = dynamic_cast<const dist::Multinomial&>(d);
        if (m == "number_of_trials") return mm.number_of_trials();
        if (m == "mean") return mm.mean()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "variance") return mm.variance()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "covariance") return mm.covariance(a[0].get<int>(), a[1].get<int>());
        if (m == "random_value") return random_value_at(mm, a);
    } else if (target == "BivariateEmpirical") {
        auto& bb = dynamic_cast<dist::BivariateEmpirical&>(d);
        if (m == "cdf_xy") return bb.cdf(a[0].get<double>(), a[1].get<double>());
        // v2.1.4: verifies the stale-cache fix in ONE self-contained call (works
        // identically whether a runner holds a persistent object across a case's
        // assertions, like this one, or rebuilds fresh per dispatch, like R/Python) --
        // cdf() once (forces the bilinear cache to build against the CURRENT grid),
        // set_parameters() with a REPLACEMENT grid, then cdf() again; a stale cache would
        // still reflect the OLD grid on the second call. args = [[x1_new...], [x2_new...],
        // [[p_row0...], ...], x1_eval, x2_eval].
        if (m == "cdf_xy_after_set_parameters") {
            bb.cdf(a[3].get<double>(), a[4].get<double>());
            std::vector<double> x1 = parse_num_vec(a[0]);
            std::vector<double> x2 = parse_num_vec(a[1]);
            std::vector<std::vector<double>> p;
            for (const auto& row : a[2]) p.push_back(parse_num_vec(row));
            bb.set_parameters(std::move(x1), std::move(x2), std::move(p));
            return bb.cdf(a[3].get<double>(), a[4].get<double>());
        }
    } else if (target == "MultivariateNormal") {
        const auto& nn = dynamic_cast<const dist::MultivariateNormal&>(d);
        if (m == "mean") return nn.mean()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "median") return nn.median()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "mode") return nn.mode()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "sd") return nn.standard_deviation()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "variance") return nn.variance()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "covariance") return nn.covariance(a[0].get<int>(), a[1].get<int>());
        if (m == "mahalanobis") return nn.mahalanobis(parse_num_vec(a[0]));
        if (m == "inverse_cdf") return nn.inverse_cdf(parse_num_vec(a[0]))[static_cast<std::size_t>(a[1].get<int>())];
        if (m == "interval") return nn.interval(parse_num_vec(a[0]), parse_num_vec(a[1]));
        if (m == "random_value") return random_value_at(nn, a);
        if (m == "lhs_value") return lhs_value_at(nn, a);
        if (m == "mvndst") {
            // args = [n, [lower...], [upper...], [infin...], [correl...], maxpts, abseps, releps]
            int n = a[0].get<int>();
            std::vector<double> lower = parse_num_vec(a[1]);
            std::vector<double> upper = parse_num_vec(a[2]);
            std::vector<int> infin;
            for (const auto& v : a[3]) infin.push_back(v.get<int>());
            std::vector<double> correl = parse_num_vec(a[4]);
            int maxpts = a[5].get<int>();
            double abseps = a[6].get<double>();
            double releps = a[7].get<double>();
            double error = 0, value = 0;
            int inform = 0;
            nn.mvndst(n, lower, upper, infin, correl, maxpts, abseps, releps, error, value, inform);
            return value;
        }
        if (m == "mvndst_inform" || m == "mvndst_error") {
            // Same args shape as "mvndst" above -- a separate method since dispatch_multivariate
            // returns one double per call and the v2.1.4 status-code cases assert INFORM/ERROR.
            int n = a[0].get<int>();
            std::vector<double> lower = parse_num_vec(a[1]);
            std::vector<double> upper = parse_num_vec(a[2]);
            std::vector<int> infin;
            for (const auto& v : a[3]) infin.push_back(v.get<int>());
            std::vector<double> correl = parse_num_vec(a[4]);
            int maxpts = a[5].get<int>();
            double abseps = a[6].get<double>();
            double releps = a[7].get<double>();
            double error = 0, value = 0;
            int inform = 0;
            nn.mvndst(n, lower, upper, infin, correl, maxpts, abseps, releps, error, value, inform);
            return m == "mvndst_inform" ? inform : error;
        }
        if (m == "is_density_valid") return nn.is_density_valid() ? 1.0 : 0.0;
        if (m == "marginal_mean") {
            auto marginal = nn.marginal(parse_int_vec(a[0]));
            return marginal.mean()[static_cast<std::size_t>(a[1].get<int>())];
        }
        if (m == "marginal_covariance") {
            auto marginal = nn.marginal(parse_int_vec(a[0]));
            return marginal.covariance(a[1].get<int>(), a[2].get<int>());
        }
        if (m == "marginal_log_pdf") {
            auto marginal = nn.marginal(parse_int_vec(a[0]));
            return marginal.log_pdf(parse_num_vec(a[1]));
        }
        if (m == "marginal_dimension") return nn.marginal(parse_int_vec(a[0])).dimension();
        if (m == "conditional_mean") {
            auto conditional = nn.conditional(parse_int_vec(a[0]), parse_num_vec(a[1]));
            return conditional.mean()[static_cast<std::size_t>(a[2].get<int>())];
        }
        if (m == "conditional_covariance") {
            auto conditional = nn.conditional(parse_int_vec(a[0]), parse_num_vec(a[1]));
            return conditional.covariance(a[2].get<int>(), a[3].get<int>());
        }
        if (m == "conditional_dimension")
            return nn.conditional(parse_int_vec(a[0]), parse_num_vec(a[1])).dimension();
    } else if (target == "MultivariateStudentT") {
        const auto& tt = dynamic_cast<const dist::MultivariateStudentT&>(d);
        if (m == "degrees_of_freedom") return tt.degrees_of_freedom();
        if (m == "mean") return tt.mean()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "median") return tt.median()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "mode") return tt.mode()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "sd") return tt.standard_deviation()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "variance") return tt.variance()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "covariance") return tt.covariance(a[0].get<int>(), a[1].get<int>());
        if (m == "mahalanobis") return tt.mahalanobis(parse_num_vec(a[0]));
        if (m == "inverse_cdf") return tt.inverse_cdf(parse_num_vec(a[0]))[static_cast<std::size_t>(a[1].get<int>())];
        if (m == "random_value") return random_value_at(tt, a);
        if (m == "lhs_value") return lhs_value_at(tt, a);
    }
    throw std::runtime_error("unknown multivariate fixture method: " + target + "/" + m);
}

// MultivariateNormal's CDF above dimension 2, its Interval, and MVNDST itself all draw from the
// instance's persistent MVNUNI stream, so each call ADVANCES it. A case that makes more than one
// such call pins a sequence off one object, which a stateless runner cannot reproduce by
// construction.
static bool mvn_consumes_stream(const std::string& m) {
    return m == "cdf" || m == "interval" || m == "mvndst" || m == "mvndst_inform" ||
           m == "mvndst_error";
}

// Methods run_mvdist covers. What is left on dispatch_multivariate is exactly three groups:
// the MVNDST integrator internals (mvndst and its two status arms), BivariateEmpirical's
// cdf_xy(_after_set_parameters) and Dirichlet's static log_multivariate_beta, which have no
// runner verb; and MultivariateNormal's cdf/interval in a case that makes more than one
// stream-consuming call (`mvn_stream_isolated` false -- r_mvtnorm_4d_sequential's eleven
// advancing cdf values are the reason). With `seed` in the grammar a SINGLE such call
// reproduces exactly, so those cases delegate.
static bool mv_delegated(const std::string& target, const std::string& m,
                         bool mvn_stream_isolated) {
    if (m == "mvndst" || m == "mvndst_inform" || m == "mvndst_error") return false;
    if (target == "MultivariateNormal" && (m == "cdf" || m == "interval") && !mvn_stream_isolated)
        return false;
    return m == "dimension" || m == "pdf" || m == "log_pdf" || m == "cdf" || m == "mahalanobis" ||
           m == "mean" || m == "variance" || m == "sd" || m == "covariance" ||
           m == "median" || m == "mode" || m == "inverse_cdf" || m == "interval" ||
           m == "degrees_of_freedom" || m == "alpha" || m == "alpha_sum" ||
           m == "number_of_trials" ||
           m == "random_value" || m == "lhs_value" || m == "marginal_dimension" ||
           m == "marginal_mean" || m == "marginal_covariance" || m == "marginal_log_pdf" ||
           m == "conditional_dimension" || m == "conditional_mean" ||
           m == "conditional_covariance";
}

static std::size_t square_dim(std::size_t n) {
    auto d = static_cast<std::size_t>(std::llround(std::sqrt(static_cast<double>(n))));
    if (d * d != n) throw std::runtime_error("a covariance result is not a square matrix");
    return d;
}

// run_mvdist returns whole vectors, so every fixture method that names one element indexes in
// here. Conventions preserved verbatim from the deleted dispatch_multivariate arms:
// mean/variance/sd/median/mode/alpha take [i]; covariance takes [i, j] against a row-major
// dimension^2 block; inverse_cdf takes [probabilities, i] and interval [lower, upper];
// random_value/lhs_value take [sample_size, seed, row, col] against a row-major
// sample_size x dimension block; marginal_* take [indices, ...] and conditional_* take
// [indices, values, ...], both evaluated against the child distribution the runner hands back
// as a spec.
static double dispatch_multivariate_delegated(const std::string& spec, const std::string& m,
                                              const json& a) {
    auto run = [&](const std::string& s, const std::string& method, const json& args) {
        return supp::run_mvdist(s, method, args.dump());
    };
    if (m == "dimension" || m == "alpha_sum" || m == "degrees_of_freedom" ||
        m == "number_of_trials")
        return run(spec, m, json::array()).values.at(0);
    if (m == "pdf" || m == "log_pdf" || m == "cdf" || m == "mahalanobis")
        return run(spec, m, a[0]).values.at(0);
    if (m == "inverse_cdf")
        return run(spec, m, a[0]).values.at(static_cast<std::size_t>(a[1].get<int>()));
    if (m == "interval") {
        json bounds = a[0];
        for (const auto& v : a[1]) bounds.push_back(v);
        return run(spec, "interval", bounds).values.at(0);
    }
    if (m == "mean" || m == "variance" || m == "sd" || m == "median" || m == "mode" ||
        m == "alpha")
        return run(spec, m, json::array()).values.at(static_cast<std::size_t>(a[0].get<int>()));
    if (m == "covariance") {
        auto r = run(spec, "covariance", json::array());
        std::size_t dim = square_dim(r.values.size());
        return r.values.at(static_cast<std::size_t>(a[0].get<int>()) * dim +
                           static_cast<std::size_t>(a[1].get<int>()));
    }
    if (m == "random_value" || m == "lhs_value") {
        json args = {a[0], a[1]};
        auto r = run(spec, m == "lhs_value" ? "random_lhs" : "random", args);
        std::size_t n = static_cast<std::size_t>(a[0].get<int>());
        std::size_t dim = r.values.size() / n;
        return r.values.at(static_cast<std::size_t>(a[2].get<int>()) * dim +
                           static_cast<std::size_t>(a[3].get<int>()));
    }
    if (m.rfind("marginal_", 0) == 0 || m.rfind("conditional_", 0) == 0) {
        const bool marginal = m.rfind("marginal_", 0) == 0;
        json child_args = a[0];
        if (!marginal)
            for (const auto& v : a[1]) child_args.push_back(v);
        const std::string child = run(spec, marginal ? "marginal" : "conditional", child_args).spec;
        // The trailing arguments shift by one for conditional_* (it consumes the values array).
        const std::size_t base = marginal ? 1 : 2;
        const std::string leaf = m.substr(m.find('_') + 1);
        if (leaf == "dimension") return run(child, "dimension", json::array()).values.at(0);
        if (leaf == "log_pdf") return run(child, "log_pdf", a[base]).values.at(0);
        if (leaf == "mean")
            return run(child, "mean", json::array())
                .values.at(static_cast<std::size_t>(a[base].get<int>()));
        if (leaf == "covariance") {
            auto r = run(child, "covariance", json::array());
            std::size_t dim = square_dim(r.values.size());
            return r.values.at(static_cast<std::size_t>(a[base].get<int>()) * dim +
                               static_cast<std::size_t>(a[base + 1].get<int>()));
        }
        throw std::runtime_error("unhandled child method: " + m);
    }
    throw std::runtime_error("method '" + m + "' is not delegated to run_mvdist");
}

static void run_multivariate(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        // Limitation 1: a construct carrying a "nan"/"inf" literal cannot be serialized into
        // the grammar, so its (validity-only) assertions run locally.
        const bool encodable = !json_has_non_finite(c["construct"]);
        const std::string cspec =
            encodable ? mvdist_spec(target, c["construct"]).dump() : std::string();
        // A case whose MVNUNI stream is consumed at most once has no sequence to preserve, so
        // its cdf/interval delegates; anything more stays whole on the local object below.
        int stream_calls = 0;
        if (target == "MultivariateNormal")
            for (const auto& as : c["assertions"])
                if (mvn_consumes_stream(as["method"].get<std::string>())) ++stream_calls;
        const bool stream_isolated = stream_calls <= 1;
        // Built lazily and held for the whole case, so the bespoke arms see one object in
        // assertion order exactly as they did before -- MultivariateNormal's MVNUNI stream
        // advances across a case's cdf/mvndst assertions and the oracles pin that sequence.
        std::unique_ptr<dist::MultivariateDistribution> local;
        auto local_object = [&]() -> dist::MultivariateDistribution& {
            if (!local) local = build_multivariate_local(target, c["construct"]);
            return *local;
        };
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = args_array(as);
            std::string where = target + "/" + name + "/" + method;
            const bool is_bool = as["mode"].get<std::string>() == "bool";
            // Limitation 1 again, on the evaluation point rather than the construct: MVN's and
            // MVT's log_pdf-at-infinity cases pass an infinite coordinate, which the grammar's
            // reader cannot carry either.
            if (!encodable || json_has_non_finite(args)) {
                // The old dispatcher ignored the assertion's method for bool mode.
                if (is_bool)
                    check_bool(local_object().parameters_valid(), as, where);
                else
                    check_value(dispatch_multivariate(local_object(), target, method, args), as,
                                where);
                continue;
            }
            if (is_bool) {
                auto r = supp::run_mvdist(cspec, "parameters_valid", "[]");
                check_bool(r.values.at(0) != 0.0, as, where);
                continue;
            }
            if (mv_delegated(target, method, stream_isolated))
                check_value(dispatch_multivariate_delegated(cspec, method, args), as, where);
            else
                check_value(dispatch_multivariate(local_object(), target, method, args), as, where);
        }
    }
}

// --- bivariate_copula path --------------------------------------------------------------
//
// Fully delegated to run_copula: every copula shares BivariateCopula's uniform
// theta/get_copula_parameters/pdf/cdf/... API, so there is no per-target branching left here
// at all (the "tau" method-of-moments fit, whose SetThetaFromTau is a member of each concrete
// Archimedean class rather than of IBivariateCopula, is dispatched by
// copulas::set_theta_from_tau in dist_spec.hpp). The one local path is limitation 1: the
// non-finite theta/df validity cases, which the grammar cannot encode.

namespace cop = corehydro::numerics::distributions::copulas;

// Translates a fixture copula construct into the dist_spec grammar. The schema differences are
// the marginal spelling (positional "marginals" here, margin_x / margin_y there) and the fit
// samples (dataset NAMES here, inline arrays there). The fixture's bare-family marginal
// convention needs no translation: dist_spec's build_copula MLE-fits a parameterless marginal
// to its own sample, which is what the fixture means and what IFM requires.
static json copula_spec(const std::string& target, const json& construct, const json& datasets) {
    auto margin = [](const std::string& family, const json& params) {
        json m;
        m["family"] = family;
        if (!params.is_null()) m["parameters"] = params;
        return m;
    };
    json out = construct;
    out["family"] = target;

    if (construct.contains("marginals")) {
        const auto& marg = construct["marginals"];
        out.erase("marginals");
        out["margin_x"] = margin(marg["targets"][0].get<std::string>(), marg["params"][0]);
        out["margin_y"] = margin(marg["targets"][1].get<std::string>(), marg["params"][1]);
    }

    if (construct.contains("fit")) {
        const auto& fit = construct["fit"];
        json f = fit;
        std::vector<double> x, y;
        for (const auto& v : datasets[fit["x"].get<std::string>()]) x.push_back(parse_num(v));
        for (const auto& v : datasets[fit["y"].get<std::string>()]) y.push_back(parse_num(v));
        f["x"] = x;
        f["y"] = y;
        if (fit.contains("marginals")) {
            const std::string fx = fit["marginals"][0].get<std::string>();
            const std::string fy = fit["marginals"][1].get<std::string>();
            f.erase("marginals");
            f["margin_x"] = margin(fx, json());
            f["margin_y"] = margin(fy, json());
        }
        out["fit"] = f;
    }
    return out;
}

static std::string fixture_copula_method(const std::string& m) {
    if (m == "upper_tail_dependence" || m == "lower_tail_dependence") return "tail_dependence";
    if (m == "theta_minimum" || m == "theta_maximum") return "bounds";
    if (m == "or_exceedance") return "exceedance_or";
    if (m == "and_exceedance") return "exceedance_and";
    if (m == "random_value") return "random";
    // pdf, log_pdf, cdf, inverse_cdf, theta, df and the three log_likelihood_* verbs pass
    // straight through.
    return m;
}

// The runner returns a whole vector for the methods the fixture indexes into. `random` comes
// back as all the x draws followed by all the y draws, so a fixture (row, col) with
// args = [sample_size, seed, row, col] lands at col * sample_size + row.
static double fixture_copula_pick(const supp::DistResult& r, const std::string& m, const json& a) {
    if (m == "lower_tail_dependence" || m == "theta_minimum") return r.values.at(0);
    if (m == "upper_tail_dependence" || m == "theta_maximum") return r.values.at(1);
    if (m == "inverse_cdf") return r.values.at(static_cast<std::size_t>(a[2].get<int>()));
    if (m == "marginal_param") return r.values.at(static_cast<std::size_t>(a[1].get<int>()));
    if (m == "random_value") {
        std::size_t n = static_cast<std::size_t>(a[0].get<int>());
        return r.values.at(static_cast<std::size_t>(a[3].get<int>()) * n +
                           static_cast<std::size_t>(a[2].get<int>()));
    }
    return r.values.at(0);
}

// The three copula log-likelihood verbs take a paired SAMPLE, which the runner reads as one
// flat "all x then all y" args array. Spelling 200 numbers per assertion into the fixture
// would drown the file, so those assertions name their two datasets instead --
// args = ["<x dataset>", "<y dataset>"] -- and every runner splices the named arrays here.
// Documented under `bivariate_copula` in fixtures/README.md.
static bool is_copula_log_likelihood(const std::string& m) {
    return m == "log_likelihood_pseudo" || m == "log_likelihood_ifm" || m == "log_likelihood_full";
}

static json copula_sample_args(const json& a, const json& datasets) {
    json out = json::array();
    for (const auto& name : a) {
        const std::string key = name.get<std::string>();
        if (!datasets.contains(key))
            throw std::runtime_error("copula log-likelihood args name an unknown dataset: " + key);
        for (const auto& v : datasets[key]) out.push_back(parse_num(v));
    }
    return out;
}

static void run_bivariate_copula(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        // Limitation 1: a "nan"/"inf" theta or df cannot be serialized into the grammar. Every
        // such case asserts parameters_valid alone.
        if (json_has_non_finite(c["construct"])) {
            auto local = cop::create_copula(target);
            std::vector<double> params = {parse_num(c["construct"]["theta"])};
            if (c["construct"].contains("df")) params.push_back(parse_num(c["construct"]["df"]));
            local->set_copula_parameters(params);
            for (const auto& as : c["assertions"]) {
                std::string where = target + "/" + name + "/" + as["method"].get<std::string>();
                if (as["mode"].get<std::string>() != "bool")
                    throw std::runtime_error(where +
                                             ": a non-finite copula construct can only assert "
                                             "parameters_valid");
                check_bool(local->parameters_valid(), as, where);
            }
            continue;
        }

        const std::string cspec = copula_spec(target, c["construct"], datasets).dump();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = args_array(as);
            std::string where = target + "/" + name + "/" + method;
            if (as["mode"].get<std::string>() == "bool") {
                // The old dispatcher ignored the assertion's method for bool mode and read
                // parameters_valid(); keep that exactly.
                auto r = supp::run_copula(cspec, "parameters_valid", "[]");
                check_bool(r.values.at(0) != 0.0, as, where);
                continue;
            }
            if (method == "marginal_param") {
                // args = ("x" | "y", index): the side picks the runner method, the index picks
                // the value out of that marginal's parameter vector.
                std::string side = args[0].get<std::string>();
                auto r = supp::run_copula(
                    cspec, side == "x" ? "marginal_x_parameters" : "marginal_y_parameters", "[]");
                check_value(fixture_copula_pick(r, method, args), as, where);
                continue;
            }
            if (is_copula_log_likelihood(method)) args = copula_sample_args(args, datasets);
            auto r = supp::run_copula(cspec, fixture_copula_method(method), args.dump());
            check_value(fixture_copula_pick(r, method, args), as, where);
        }
    }
}

// --- mcmc_sampler path -------------------------------------------------------------------
//
// One sampler run per case: build the model via the registry, construct the sampler (RWMH
// today; extensible via a target-name switch as later samplers land), apply non-default
// settings, sample() ONCE, and cache both the sampler and its post-processed MCMCResults for
// every assertion in the case (mirrors the "single stateful glue call; no seq machinery"
// contract fixtures/README.md documents for this kind).

namespace mcmc = corehydro::numerics::sampling::mcmc;

// `proposal_sigma` sentinel strings -- see fixtures/README.md's mcmc_sampler schema.
// "zeros": the literal `Matrix(D)` the C# Test_RWMH.cs test constructs (safe only when
// MAP initialization is expected to override it before first use -- see "identity" below).
// "identity": D x D identity matrix. NOT present in the upstream C# test; added because a
// Randomize-initialized RWMH with a literal all-zero proposal covariance throws
// (CholeskyDecomposition rejects a non-positive-definite matrix) on its very first
// ChainIteration -- confirmed against the real C# library. Any Randomize-init fixture case
// therefore needs a non-degenerate proposal_sigma; identity is the simplest one.
//
// The sentinel STRINGS are parsed by mcmc_run.hpp's parse_proposal_sigma, along with everything
// else about building a sampler: this file reads the fixture's `settings` object into an
// MCMCRunSettings and hands it over, exactly as the R and Python glues do with their own native
// settings containers. There is one switch over sampler names in the repo and it is not here.
static mcmc::MCMCRunSettings read_settings(const json& settings) {
    mcmc::MCMCRunSettings s;
    auto read_int = [&settings](const char* key, std::optional<int>& slot) {
        if (settings.contains(key)) slot = settings[key].get<int>();
    };
    auto read_double = [&settings](const char* key, std::optional<double>& slot) {
        if (settings.contains(key)) slot = settings[key].get<double>();
    };
    auto read_string = [&settings](const char* key, std::optional<std::string>& slot) {
        if (settings.contains(key)) slot = settings[key].get<std::string>();
    };
    read_string("initialize", s.initialize);
    read_int("prng_seed", s.prng_seed);
    read_int("initial_iterations", s.initial_iterations);
    read_int("warmup_iterations", s.warmup_iterations);
    read_int("iterations", s.iterations);
    read_int("number_of_chains", s.number_of_chains);
    read_int("thinning_interval", s.thinning_interval);
    read_int("output_length", s.output_length);
    read_string("proposal_sigma", s.proposal_sigma);
    read_double("step_size", s.step_size);
    read_int("steps", s.steps);
    read_int("max_tree_depth", s.max_tree_depth);
    if (settings.contains("adapt_mass_matrix")) s.adapt_mass_matrix = settings["adapt_mass_matrix"].get<bool>();
    read_double("scale", s.scale);
    read_double("beta", s.beta);
    read_double("jump", s.jump);
    read_double("jump_threshold", s.jump_threshold);
    read_double("snooker_threshold", s.snooker_threshold);
    read_double("noise", s.noise);
    return s;
}

// Builds + configures + samples() one sampler from a {"model": {...}, "settings": {...}}
// construct. `sampler_target`: the fixture's file-level "target" (the sampler type, e.g. "RWMH").
static std::unique_ptr<mcmc::MCMCSampler> build_and_sample(const std::string& sampler_target,
                                                             const json& construct, const json& datasets) {
    const auto& model_spec = construct["model"];
    std::vector<double> data;
    for (const auto& v : datasets[model_spec["dataset"].get<std::string>()]) data.push_back(parse_num(v));
    auto model = mcmc::build_model(model_spec["name"].get<std::string>(),
                                    model_spec["family"].get<std::string>(), data);

    mcmc::MCMCRunCallbacks callbacks;
    callbacks.proposal = model.proposal;
    auto sampler = mcmc::build_sampler(sampler_target, model.priors, model.log_likelihood,
                                        read_settings(construct.value("settings", json::object())),
                                        callbacks);
    sampler->sample();
    return sampler;
}

static double dispatch_mcmc(const mcmc::MCMCSampler& sampler, const mcmc::MCMCResults& results,
                             const std::string& m, const json& a) {
    auto idx = [&](int i) { return static_cast<std::size_t>(a[static_cast<std::size_t>(i)].get<int>()); };
    if (m == "posterior_mean") return results.parameter_results[idx(0)].summary_statistics.mean;
    if (m == "posterior_sd") return results.parameter_results[idx(0)].summary_statistics.standard_deviation;
    if (m == "posterior_median") return results.parameter_results[idx(0)].summary_statistics.median;
    if (m == "posterior_lower_ci") return results.parameter_results[idx(0)].summary_statistics.lower_ci;
    if (m == "posterior_upper_ci") return results.parameter_results[idx(0)].summary_statistics.upper_ci;
    if (m == "chain_value") return sampler.markov_chains()[idx(0)][idx(1)].values[idx(2)];
    if (m == "chain_fitness") return sampler.markov_chains()[idx(0)][idx(1)].fitness;
    if (m == "map_value") return results.map.values[idx(0)];
    if (m == "map_fitness") return results.map.fitness;
    if (m == "acceptance_rate") return sampler.acceptance_rates()[idx(0)];
    if (m == "mean_log_likelihood") return sampler.mean_log_likelihood()[idx(0)];
    if (m == "rhat") return results.parameter_results[idx(0)].summary_statistics.rhat;
    if (m == "ess") return results.parameter_results[idx(0)].summary_statistics.ess;
    throw std::runtime_error("unknown mcmc_sampler fixture method: " + m);
}

static void run_mcmc_sampler(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        auto sampler = build_and_sample(target, c["construct"], datasets);
        mcmc::MCMCResults results(*sampler);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = target + "/" + name + "/" + method;
            check_value(dispatch_mcmc(*sampler, results, method, args), as, where);
        }
    }
}

// --- bootstrap path ----------------------------------------------------------------------
//
// One bootstrap run per case (mirrors mcmc_sampler's single-stateful-glue-call contract): build
// the model via the registry, configure (replicates/seed/max_retries), run() or
// run_with_studentized_bootstrap() ONCE, then get_confidence_intervals() ONCE with the case's
// ci_method/alpha; every assertion in the case reads that single cached (bootstrap, results)
// pair. See fixtures/README.md's bootstrap schema.


struct BootstrapCase {
    bfsamp::Bootstrap<std::vector<double>> boot;
    bfsamp::BootstrapResults results;
};

static BootstrapCase build_and_run_bootstrap(const json& construct, const json& datasets) {
    std::string model_name = construct["model"].get<std::string>();
    double mu = construct.value("mu", 0.0);
    double sigma = construct.value("sigma", 0.0);
    int sample_size = construct.value("sample_size", 0);
    std::vector<double> probabilities;
    for (const auto& v : construct["probabilities"]) probabilities.push_back(parse_num(v));
    std::vector<double> sample_data;
    if (construct.contains("dataset"))
        for (const auto& v : datasets[construct["dataset"].get<std::string>()]) sample_data.push_back(parse_num(v));

    auto boot = bfsamp::build_bootstrap_model(model_name, mu, sigma, sample_size, probabilities, sample_data);
    if (construct.contains("replicates")) boot.replicates = construct["replicates"].get<int>();
    if (construct.contains("seed")) boot.prng_seed = construct["seed"].get<int>();
    if (construct.contains("max_retries")) boot.max_retries = construct["max_retries"].get<int>();

    std::string run = construct.value("run", "regular");
    if (run == "regular")
        boot.run();
    else if (run == "studentized")
        boot.run_with_studentized_bootstrap();
    else
        throw std::runtime_error("unknown bootstrap run kind: " + run);

    auto method = bfsamp::parse_bootstrap_ci_method(construct["ci_method"].get<std::string>());
    double alpha = construct.value("alpha", 0.1);
    bfsamp::BootstrapResults results = boot.get_confidence_intervals(method, alpha);

    return BootstrapCase{std::move(boot), std::move(results)};
}

static double dispatch_bootstrap(const BootstrapCase& bc, const std::string& m, const json& a) {
    auto idx = [&](int i) { return static_cast<std::size_t>(a[static_cast<std::size_t>(i)].get<int>()); };
    if (m == "statistic_lower_ci") return bc.results.statistic_results[idx(0)].lower_ci;
    if (m == "statistic_upper_ci") return bc.results.statistic_results[idx(0)].upper_ci;
    if (m == "parameter_lower_ci") return bc.results.parameter_results[idx(0)].lower_ci;
    if (m == "parameter_upper_ci") return bc.results.parameter_results[idx(0)].upper_ci;
    if (m == "population_estimate") return bc.results.parameter_results[idx(0)].population_estimate;
    if (m == "valid_count") return bc.results.statistic_results[idx(0)].valid_count;
    if (m == "replicate_value") return bc.boot.bootstrap_parameter_sets()[idx(0)].values[idx(1)];
    throw std::runtime_error("unknown bootstrap fixture method: " + m);
}

static void run_bootstrap(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        auto bc = build_and_run_bootstrap(c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = "Bootstrap/" + name + "/" + method;
            check_value(dispatch_bootstrap(bc, method, args), as, where);
        }
    }
}

// --- model_estimation path -----------------------------------------------------------------
//
// One estimate() run per case (mirrors mcmc_sampler's/bootstrap's single-stateful-glue-call
// contract): build the model named by `construct.model` through the SHARED spec builder
// (models/model_spec.hpp -- `type` selects UnivariateDistributionModel (default, incl.
// censored DataFrames and nonstationary trend specs), MixtureModel, CompetingRisksModel, or
// PointProcessModel; there is no separate closed-name registry like mcmc/model_registry.hpp
// needs, since the spec's factory calls plus each model's default-parameter machinery are
// already enough), construct the estimator named by the file-level `target`, call
// `estimate()` ONCE, then dispatch every assertion in the case against the cached result.
// The fit itself is not built here: the construct is serialized and handed to the SHARED
// runner (corehydro/estimation/support/fit_runner.hpp's `run_fit`), the same one the cpp11 and
// pybind11 glue and the dotnet oracle emitter drive, which returns the whole surface flat in a
// `FitResult`. The `Simulation` target (M13) builds the model, skips the fit, and caches ONE
// seeded ISimulatable::generate_random_values draw instead; the `simulated_value [i]` method
// asserts individual draws (the chain_value digest precedent).
// See fixtures/README.md's model_estimation section for the schema.
//
// WIRED (T11 + T12): MaximumLikelihood / MaximumAPosteriori share `parameter [p]`,
// `max_log_likelihood []`, `aic []`, `bic [n]`, `covariance [i,j]`, `standard_error [p]`,
// `correlation [i,j]` (the same FitResult fields back both). BayesianAnalysis (T12) adds
// `dic []`, `waic []`, `looic []`, `posterior_mean [p]`, and the seeded
// `chain_value [chain,iter,param]` digest -- a disjoint surface off the FitResult's Bayesian
// block (it shares no methods with ML/MAP).
namespace estimation = corehydro::estimation;

// The optimizer-method and GMM-strategy parsers this file used to carry now live on the shared
// fit runner (corehydro/estimation/support/fit_runner.hpp's `parse_optimizer` /
// `parse_gmm_strategy`), reached through `run_fit`'s construct instead of being applied here.

static estimation::SamplerType parse_sampler_type(const std::string& s) {
    if (s == "DEMCz") return estimation::SamplerType::DEMCz;
    if (s == "DEMCzs") return estimation::SamplerType::DEMCzs;
    if (s == "ARWMH") return estimation::SamplerType::ARWMH;
    if (s == "NUTS") return estimation::SamplerType::NUTS;
    throw std::runtime_error("unknown model_estimation sampler: " + s);
}

// Holds one built-and-run case. The estimator variant this used to carry is gone: every fit now
// goes through the shared runner (corehydro/estimation/support/fit_runner.hpp's `run_fit`), which
// returns the whole surface flat in a `FitResult`, so `dispatch_estimation` reads fields instead
// of calling estimator methods. `has_fit` is false for the estimator-less Simulation/Validate
// targets (the old std::monostate arm).
//
// `model` is the ModelBase the estimator-less targets need, and the one the M14 DataFrame / Task-16
// Validate surface reads under ANY target. Those two surfaces are pure functions of the CONSTRUCT
// (low outliers and thresholds are set at construction; plotting positions are of the collections,
// never of the fit), so this is a fresh build rather than the post-fit model the old code happened
// to hold -- exactly what the R (`ch_model_data_frame_`/`ch_model_validate_`) and Python
// (`model_data_frame`/`model_validate`) harnesses have always done for the same assertions. The
// GMM target leaves it null, as before: a Bulletin17CDistribution is not a ModelBase.
//
// `construct_json` + `data` are kept for the ONE arm that must refit live: `quantile_variance`'s
// AEP is only known at assertion-dispatch time (the `bic [n]` precedent).
struct EstimationCase {
    std::unique_ptr<corehydro::models::ModelBase> model;
    std::string target;
    bool has_fit = false;
    corehydro::estimation::support::FitResult fit;
    std::vector<double> simulated;  // Simulation target, and the ML/MAP/GMM seeded-draw digest
    std::string construct_json;
    std::vector<double> data;

    // Task 9. The case's FULL construct -- the fixture's own `construct` object with the
    // `settings` sub-object hoisted to the top level, which is exactly the shape
    // fit_runner.hpp's `run_fit`/`run_fit_diagnostics` read (see apply_bayesian_settings). The
    // narrow `construct_json` above is deliberately left alone: it backs the pinned
    // parameter/max_log_likelihood/covariance oracles, and widening it would move them.
    //
    // The two memoized results below are LAZY, and only the Task-9 arms touch them: a case that
    // asserts nothing new never pays for the extra run. Both runs are deterministic functions of
    // the same construct (seeded MCMC / RNG-free optimizers), so the profile grid and the
    // diagnostics are the same fit's, not a different one's -- the `bic [n]` lazy-rebuild
    // precedent this file already uses.
    std::string full_construct_json;
    bool full_fit_done = false;
    corehydro::estimation::support::FitResult full_fit;
    bool full_diagnostics_done = false;
    corehydro::estimation::support::FitDiagnostics full_diagnostics;
};

// Fixture target name -> the name `run_fit`/`run_fit_diagnostics` dispatch on. The only one that
// differs is GMM, whose fixture-level target spells out the C# class name.
static std::string fit_runner_target(const std::string& target) {
    return target == "GeneralizedMethodOfMoments" ? "GMM" : target;
}

static const corehydro::estimation::support::FitResult& full_fit_of(EstimationCase& ec) {
    if (!ec.full_fit_done) {
        ec.full_fit = corehydro::estimation::support::run_fit(fit_runner_target(ec.target),
                                                              ec.full_construct_json, ec.data);
        ec.full_fit_done = true;
    }
    return ec.full_fit;
}

static const corehydro::estimation::support::FitDiagnostics& full_diagnostics_of(
    EstimationCase& ec) {
    if (!ec.full_diagnostics_done) {
        ec.full_diagnostics = corehydro::estimation::support::run_fit_diagnostics(
            fit_runner_target(ec.target), ec.full_construct_json, ec.data);
        ec.full_diagnostics_done = true;
    }
    return ec.full_diagnostics;
}

// Seeded ISimulatable draw, flattened to a 1-D vector so the `simulated_value [i]` digest works
// uniformly across model types. Most Phase 4-7 models are ISimulatable<std::vector<double>> and
// pass through unchanged; BivariateDistribution is ISimulatable<Matrix2D> (n-row x 2-col), so its
// draw is flattened ROW-MAJOR (i = row*2 + col) -- the same order the R/Python glue and the README
// schema use. Throws if the model is neither.
static std::vector<double> simulate_flat(corehydro::models::ModelBase* model, int sample_size,
                                         int seed) {
    if (auto* s = dynamic_cast<corehydro::models::ISimulatable<std::vector<double>>*>(model))
        return s->generate_random_values(sample_size, seed);
    if (auto* s = dynamic_cast<
            corehydro::models::ISimulatable<std::vector<std::vector<double>>>*>(model)) {
        std::vector<std::vector<double>> mat = s->generate_random_values(sample_size, seed);
        std::vector<double> flat;
        for (const auto& row : mat)
            for (double v : row) flat.push_back(v);
        return flat;
    }
    throw std::runtime_error(
        "model_estimation Simulation target: model is not ISimulatable<vector> or "
        "ISimulatable<Matrix2D>");
}

static EstimationCase build_and_run_estimation(const std::string& target, const json& construct,
                                                 const json& datasets) {
    const auto& model_spec = construct["model"];
    std::vector<double> data;
    if (model_spec.contains("dataset"))
        for (const auto& v : datasets[model_spec["dataset"].get<std::string>()]) data.push_back(parse_num(v));

    EstimationCase ec;
    ec.target = target;
    ec.data = data;

    // Task 9: the FULL construct the fit runner reads (see EstimationCase's note). `settings` is
    // hoisted to the top level -- where apply_bayesian_settings looks for the Bayesian knobs --
    // and every other key (model, optimizer, profile, profile_bins, alpha, sampler, strategy,
    // ...) is passed through untouched; run_fit ignores the ones its target does not use. The
    // three harnesses build this the same way, so they hand the runner byte-identical constructs.
    {
        json full = construct;
        if (full.contains("settings")) {
            for (auto it = full["settings"].begin(); it != full["settings"].end(); ++it)
                full[it.key()] = it.value();
            full.erase("settings");
        }
        // fit_runner.hpp's run_fit/run_fit_diagnostics default `optimizer` to
        // DifferentialEvolution for every target, including GMM -- but the narrow GMM path just
        // above defaults it to BFGS (matching the C# GMM ctor default). Without this, a GMM case
        // that omits `optimizer` and asserts one of the sixteen wider fit-surface methods would
        // read that method off a DifferentialEvolution fit while `parameter`/`j_stat` came from a
        // BFGS fit -- two different fits, silently. Write the same BFGS default here so both
        // paths agree, mirroring the narrow path's own explicit default a few lines below.
        if (target == "GeneralizedMethodOfMoments" && !full.contains("optimizer"))
            full["optimizer"] = "BFGS";
        ec.full_construct_json = full.dump();
    }

    // GMM fits the CONCRETE Bulletin17CDistribution (not a ModelBase -- see model_spec.hpp's
    // build_bulletin17c_model wiring note), optionally caching a seeded draw from the fitted model
    // for the `simulated_value` digest (the DRY choice: `simulated_value` is already dispatched
    // from ec.simulated for every target, so riding the GMM case needs no new arm).
    //
    // The construct handed to `run_fit` reproduces this arm's old knob application exactly:
    // `optimizer` still defaults to BFGS here (the runner's own default is DifferentialEvolution,
    // so it is always written explicitly), an absent `strategy` leaves the class default (which is
    // Iterative -- the value the runner would write anyway), and `max_gmm_iterations` is forwarded
    // when present (`build_and_fit_gmm` applies it only when positive; every fixture value is).
    if (target == "GeneralizedMethodOfMoments") {
        json c = json::object();
        c["model"] = model_spec;
        c["optimizer"] = construct.value("optimizer", std::string("BFGS"));
        if (construct.contains("strategy")) c["strategy"] = construct["strategy"];
        if (construct.contains("max_gmm_iterations"))
            c["max_gmm_iterations"] = construct["max_gmm_iterations"];
        ec.construct_json = c.dump();
        ec.fit = corehydro::estimation::support::run_fit("GMM", ec.construct_json, data);
        ec.has_fit = true;
        if (construct.contains("sample_size")) {
            // Draw from the FITTED model, rebuilt from the runner's `model_spec` (the construct's
            // model object re-emitted with the fitted `parameter_values`, which the spec builder
            // applies through the same set_parameter_values the old in-place pin called).
            auto fitted =
                corehydro::models::spec::build_bulletin17c_from_json(ec.fit.model_spec, data);
            ec.simulated = fitted->generate_random_values(construct["sample_size"].get<int>(),
                                                          construct.value("seed", -1));
        }
        return ec;
    }

    // One shared construction path for all three harnesses: serialize the spec back to JSON
    // and hand it to models/model_spec.hpp (see that header for the schema). This model backs the
    // estimator-less targets and the M14 DataFrame / Task-16 Validate surface (see EstimationCase).
    ec.model = corehydro::models::spec::build_model_from_json(model_spec.dump(), data);

    if (target == "Simulation") {
        ec.simulated = simulate_flat(ec.model.get(), construct["sample_size"].get<int>(),
                                     construct.value("seed", -1));
        return ec;
    }
    // Validate (Task 16): builds the model and stops -- no estimator, no seeded draw. Lets a
    // case assert `is_valid`/`validation_message_contains` (below) against ModelBase::validate()
    // without needing to fit or simulate, e.g. the TimeSeries transform-lambda-failure cases.
    if (target == "Validate") {
        return ec;
    }
    if (target == "MaximumLikelihood" || target == "MaximumAPosteriori") {
        json c = json::object();
        c["model"] = model_spec;
        // Same default as before, and the same one `run_fit` would apply; written explicitly so
        // the construct is self-describing.
        c["optimizer"] = construct.value("optimizer", std::string("DifferentialEvolution"));
        ec.construct_json = c.dump();
        ec.fit = corehydro::estimation::support::run_fit(target, ec.construct_json, data);
        ec.has_fit = true;
        // Optional seeded-draw digest off the FITTED model (P3): when `sample_size` is present,
        // rebuild from the runner's `model_spec` (fitted `parameter_values` applied by the spec
        // builder) and cache one seeded draw -- the same shared `simulated_value` arm the
        // Simulation/GMM targets use.
        if (construct.contains("sample_size")) {
            auto fitted =
                corehydro::models::spec::build_model_from_json(ec.fit.model_spec, data);
            ec.simulated = simulate_flat(fitted.get(), construct["sample_size"].get<int>(),
                                         construct.value("seed", -1));
        }
        return ec;
    }
    if (target == "BayesianAnalysis") {
        // Mirrors the oracle emitter's BuildEstimation: the fixture's sampler type (defaulting to
        // DEMCzs) plus whichever settings it supplies. `run_fit` turns OFF the two "use defaults"
        // flags and applies each knob only when its key is present, which is exactly what the
        // `settings.contains(...)` cascade here used to do; the settings sub-object is simply
        // hoisted to the construct's top level, where apply_bayesian_settings reads it. The seeded
        // chain digest reproduces bit-identically against the real C# (see bayes_normal.json).
        json c = json::object();
        c["model"] = model_spec;
        c["sampler"] = construct.value("sampler", std::string("DEMCzs"));
        if (construct.contains("settings"))
            for (auto it = construct["settings"].begin(); it != construct["settings"].end(); ++it)
                c[it.key()] = it.value();
        ec.construct_json = c.dump();
        ec.fit = corehydro::estimation::support::run_fit("BayesianAnalysis", ec.construct_json, data);
        ec.has_fit = true;
        return ec;
    }
    throw std::runtime_error("unknown model_estimation target: " + target);
}

// The DataFrame assertion surface (M14): methods reachable from the model's DataFrame under
// ANY model_estimation target, corroborating the M1/M5 ctest oracles through the PUBLIC path.
// `plotting_position [kind, i]` reads item i's plotting position from the named series
// ("exact" | "interval" | "uncertain", in spec order) after ONE calculate_plotting_positions()
// pass (idempotent -- a pure function of the collections + the plotting parameter; the
// threshold series is NOT exposed because the C# assigns its positions to a sorted CLONE, so
// the original items never carry one). `number_of_low_outliers`/`low_outlier_threshold` read
// the frame's current state (set by the spec's `mgbt_low_outliers` MGBT trigger, or the
// explicit `low_outlier_threshold`).
static double dispatch_model_data_frame(corehydro::models::ModelBase& model, const std::string& m,
                                        const json& a) {
    auto* udm = dynamic_cast<corehydro::models::UnivariateDistributionModelBase*>(&model);
    if (udm == nullptr || !udm->has_data_frame())
        throw std::runtime_error("model_estimation data-frame method on a model without a DataFrame");
    auto& df = udm->data_frame();
    if (m == "number_of_low_outliers") return df.number_of_low_outliers();
    if (m == "low_outlier_threshold") return df.low_outlier_threshold();
    // plotting_position [kind, i]
    df.calculate_plotting_positions();
    std::string kind = a[0].get<std::string>();
    std::size_t i = static_cast<std::size_t>(a[1].get<int>());
    if (kind == "exact") return df.exact_series()[i].plotting_position();
    if (kind == "interval") return df.interval_series()[i].plotting_position();
    if (kind == "uncertain") return df.uncertain_series()[i].plotting_position();
    throw std::runtime_error("unknown plotting_position series kind: " + kind);
}

// The Validate surface (Task 16): works under ANY target (it reads the model, not the
// estimator), mirroring the M14 DataFrame surface. `is_valid` returns ModelBase::validate()'s
// bool as 1.0/0.0 (the `converged_within_tolerance` boolean-as-double precedent);
// `validation_message_contains [substring]` returns 1.0 if any validation message contains the
// given substring, else 0.0 -- a structural check (not a byte-exact C# message pin), since the
// fixture-checkable contract here is "the failure is captured as a message", not the message's
// literal text (the C++ message text itself is hand-verified against C# in the model headers).
static double dispatch_model_validate(corehydro::models::ModelBase& model, const std::string& m,
                                      const json& a) {
    auto result = model.validate();
    if (m == "is_valid") return result.is_valid ? 1.0 : 0.0;
    if (m == "validation_message_contains") {
        std::string needle = a[0].get<std::string>();
        for (const auto& msg : result.validation_messages)
            if (msg.find(needle) != std::string::npos) return 1.0;
        return 0.0;
    }
    throw std::runtime_error("unknown validate fixture method: " + m);
}

// Every arm below reads a field off the `FitResult` the shared runner already produced, so the
// per-estimator std::visit this used to be collapses into one flat lookup: GMM and ML/MAP share
// the parameter/standard_error/covariance/correlation names because they are the same fields.
// Two arms still evaluate live, because their argument is only known here: `bic [n]` (the C#
// GetBIC takes an actual sample size) and `quantile_variance [aep]`.
static double dispatch_estimation(EstimationCase& ec, const std::string& m, const json& a) {
    auto idx = [&](int i) { return static_cast<std::size_t>(a[static_cast<std::size_t>(i)].get<int>()); };
    // The seeded-simulation digest (M13): reads the vector cached at build time, so it works
    // for the Simulation target (no fit) too.
    if (m == "simulated_value") return ec.simulated.at(idx(0));
    // The M14 DataFrame surface works under any target (it reads the model, not the fit).
    if (m == "plotting_position" || m == "number_of_low_outliers" || m == "low_outlier_threshold")
        return dispatch_model_data_frame(*ec.model, m, a);
    if (m == "is_valid" || m == "validation_message_contains")
        return dispatch_model_validate(*ec.model, m, a);

    // --- Task 9: the wider fit surface -----------------------------------------------------
    //
    // Everything below reads the LAZY full-construct fit (or its diagnostics), not the narrow
    // `ec.fit` the older arms read -- see EstimationCase's note for why the two constructs are
    // kept separate. These arms come first because they are target-agnostic: the profile block
    // is ML/MAP-only and the posterior block BayesianAnalysis-only, but which one a fixture may
    // ask for is decided by the FitResult field being empty, not by a target branch.
    if (m == "profile_lower" || m == "profile_upper" || m == "profile_value" ||
        m == "function_evaluations" || m == "status_is" || m == "nobs" ||
        m == "prior_log_likelihood" || m == "rhat" || m == "ess" || m == "acceptance_rate" ||
        m == "posterior_median" || m == "posterior_sd" || m == "posterior_lower" ||
        m == "posterior_upper") {
        const auto& f = full_fit_of(ec);
        if (m == "profile_lower") return f.profile_lower.at(idx(0));
        if (m == "profile_upper") return f.profile_upper.at(idx(0));
        // profile_value [param, bin, col]: profile_grid is n_params x bins x 2, row-major, with
        // col 0 = the parameter value at that bin's midpoint and col 1 = the profile
        // log-likelihood there (fit_runner.hpp's fill_profile).
        if (m == "profile_value") {
            std::size_t bins = static_cast<std::size_t>(f.profile_bins);
            return f.profile_grid.at((idx(0) * bins + idx(1)) * 2u + idx(2));
        }
        if (m == "function_evaluations") return f.function_evaluations;
        // status_is [name]: 1.0 when the optimizer status matches the given name, else 0.0 (the
        // `validation_message_contains` boolean-as-double precedent -- the fixture schema carries
        // no string comparison).
        if (m == "status_is") return f.status == a[0].get<std::string>() ? 1.0 : 0.0;
        if (m == "nobs") return f.nobs;
        if (m == "prior_log_likelihood") return f.prior_log_likelihood;
        if (m == "rhat") return f.rhat.at(idx(0));
        if (m == "ess") return f.ess.at(idx(0));
        if (m == "acceptance_rate") return f.acceptance_rates.at(idx(0));
        if (m == "posterior_median") return f.summary_median.at(idx(0));
        if (m == "posterior_sd") return f.summary_sd.at(idx(0));
        if (m == "posterior_lower") return f.summary_lower.at(idx(0));
        return f.summary_upper.at(idx(0));  // posterior_upper
    }
    // The PSIS-LOO Pareto-k surface lives on FitDiagnostics (the InfluenceDiagnostics wrapper),
    // not on FitResult, so it takes the second lazy runner call.
    if (m == "pareto_k" || m == "max_pareto_k") {
        const auto& d = full_diagnostics_of(ec);
        if (m == "pareto_k") return d.pareto_k.at(idx(0));
        return d.max_pareto_k;
    }

    if (!ec.has_fit)
        throw std::runtime_error("unknown Simulation/Validate fixture method: " + m);

    const auto& f = ec.fit;
    int n = static_cast<int>(f.parameters.size());
    // Shared by ML/MAP and GMM (B11). Also reachable when ec.target == "BayesianAnalysis" (the
    // target-specific dispatch below falls through to here for any method it doesn't itself
    // recognize), where FitResult leaves standard_errors/covariance/correlation EMPTY -- so these
    // use .at() rather than operator[] to turn an out-of-range fixture typo into a clear
    // std::out_of_range instead of indexing UB. No fixture reaches that combination today.
    //
    // Divergence note for whoever adds one: for a 1-parameter ML/MAP fit the three harnesses
    // disagree on these fields' VALUES too -- this C++ runner returns the fit_runner's NaN, the R
    // glue writes explicit zeros (corehydror/src/estimation.cpp), and the Python glue omits the
    // keys entirely (corehydropy/src/bindings/estimation.cpp) -- so a new assertion here needs a
    // harness-specific expectation, not one shared literal.
    if (m == "parameter") return f.parameters.at(idx(0));
    if (m == "standard_error") return f.standard_errors.at(idx(0));
    if (m == "covariance")
        return f.covariance.at(idx(0) * static_cast<std::size_t>(n) + idx(1));
    if (m == "correlation")
        return f.correlation.at(idx(0) * static_cast<std::size_t>(n) + idx(1));

    if (ec.target == "GeneralizedMethodOfMoments") {
        if (m == "j_stat") return f.j_stat;
        if (m == "j_stat_pval") return f.j_stat_pval;
        // T13: GMMIterations/ConvergedWithinTolerance (off-by-one fix) and
        // OptimizerFallbackCount (sticky BFGS->NelderMead fallback).
        if (m == "gmm_iterations") return f.gmm_iterations;
        if (m == "converged_within_tolerance") return f.converged_within_tolerance ? 1.0 : 0.0;
        if (m == "optimizer_fallback_count") return f.optimizer_fallback_count;
        if (m == "quantile_variance") {
            // args[0] is the annual EXCEEDANCE probability (AEP); the C# QuantileVariance takes a
            // NON-exceedance probability. run_fit_quantile_variance IS the old body of this arm
            // (rebuild the deterministic GMM fit, evaluate the B17C delta-method Var(Q_p)),
            // lifted onto the shared runner; it is the same live-rebuild the R/Python harnesses
            // have always taken for this assertion.
            return corehydro::estimation::support::run_fit_quantile_variance(ec.construct_json,
                                                                             ec.data,
                                                                             a[0].get<double>());
        }
        throw std::runtime_error("unknown GMM fixture method: " + m);
    }
    if (ec.target == "BayesianAnalysis") {
        if (m == "dic") return f.dic;
        if (m == "waic") return f.waic;
        if (m == "looic") return f.looic;
        if (m == "posterior_mean") return f.posterior_mean[idx(0)];
        // `draws` is the chain-major flatten of MCMCResults::markov_chains, itself a copy of
        // sampler()->markov_chains() (mcmc_results.hpp:35-37), so this is the same
        // chains[chain][iter].values[param] lookup as before.
        if (m == "chain_value") {
            std::size_t iters = static_cast<std::size_t>(f.chain_dims[1]);
            std::size_t params = static_cast<std::size_t>(f.chain_dims[2]);
            return f.draws[(idx(0) * iters + idx(1)) * params + idx(2)];
        }
        throw std::runtime_error("unknown BayesianAnalysis fixture method: " + m);
    }
    if (m == "max_log_likelihood") return f.log_likelihood;
    if (m == "aic") return f.aic;
    if (m == "bic") {
        // `n` is a SAMPLE SIZE supplied by the assertion, not an index, so this stays live.
        // MaximumLikelihood::get_bic(n) (maximum_likelihood.hpp:448-453) IS this call --
        // GoodnessOfFit::bic(n, number_of_parameters(), maximum_log_likelihood()) -- evaluated
        // off the runner's FitResult rather than re-derived.
        return corehydro::numerics::data::GoodnessOfFit::bic(a[0].get<int>(), n, f.log_likelihood);
    }
    throw std::runtime_error("unknown model_estimation fixture method: " + m);
}

static void run_model_estimation(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        auto ec = build_and_run_estimation(target, c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = target + "/" + name + "/" + method;
            check_value(dispatch_estimation(ec, method, args), as, where);
        }
    }
}

// --- analysis (Phase 8: user-facing Analyses layer) -------------------------------------
//
// A10: the three finished analyses (UnivariateAnalysis / FittingAnalysis / Bulletin17CAnalysis)
// become fixture-checkable through this stateful kind, mirroring model_estimation: one
// build+run per case caches a flat result surface, then every assertion dispatches against it.
// The construct fields map 1:1 onto the R/Python glue arguments (ch_analysis_*_ /
// analysis_*), so all three harnesses build byte-identical analyses from the same spec.
// A10 authors LOOSE, self-computed smoke oracles here; A11's emitter may tighten them.
namespace an = corehydro::analyses;

static an::UncertaintyMethod parse_uncertainty_method(const std::string& s) {
    if (s == "MultivariateNormal") return an::UncertaintyMethod::MultivariateNormal;
    if (s == "Bootstrap") return an::UncertaintyMethod::Bootstrap;
    // X12: the two B17C uncertainty paths un-gated in the core by X8/X9.
    if (s == "LinkedMultivariateNormal") return an::UncertaintyMethod::LinkedMultivariateNormal;
    if (s == "BiasCorrectedBootstrap") return an::UncertaintyMethod::BiasCorrectedBootstrap;
    throw std::runtime_error("unsupported/ deferred uncertainty method: " + s);
}

// Flat result surface every analysis assertion reads (only the fields the target populates are
// filled). Curve/CI vectors are indexed by the exceedance grid; the FittingAnalysis fields carry
// one entry per candidate.
struct AnalysisResult {
    std::vector<double> parameters, mode_curve, mean_curve, lower_ci, upper_ci;
    std::vector<double> exceedance, point_estimates, beta1, nu, quantile_variance;
    std::vector<double> cand_aic, cand_bic, cand_rmse, cand_converged;
    double aic = std::numeric_limits<double>::quiet_NaN();
    double bic = std::numeric_limits<double>::quiet_NaN();
    double dic = std::numeric_limits<double>::quiet_NaN();
    double rmse = std::numeric_limits<double>::quiet_NaN();
    double confidence_level = std::numeric_limits<double>::quiet_NaN();
    int candidate_count = 0;

    // --- Diagnostics slice (D5; target == "Diagnostics"). Additive: only the Diagnostics
    // target populates these; every other analysis target leaves them at the defaults. ---
    // Leverage (D3): per-observation arrays + totals + the prior-component count.
    std::vector<double> lev_obs_leverage, lev_obs_fit, lev_obs_var, lev_obs_value;
    int lev_count = 0;
    int lev_prior_count = 0;
    double total_leverage = std::numeric_limits<double>::quiet_NaN();
    double total_fit_influence = std::numeric_limits<double>::quiet_NaN();
    double total_variance_influence = std::numeric_limits<double>::quiet_NaN();
    // Influence (D4 PSIS-LOO): per-observation arrays + summary scalars.
    std::vector<double> inf_pareto_k, inf_elpd_loo;
    int inf_count = 0;
    double mean_pareto_k = std::numeric_limits<double>::quiet_NaN();
    double max_pareto_k = std::numeric_limits<double>::quiet_NaN();
    int count_pareto_k_above_05 = 0, count_pareto_k_above_07 = 0, count_pareto_k_above_10 = 0;
    double proportion_problematic = std::numeric_limits<double>::quiet_NaN();
    double is_reliable = std::numeric_limits<double>::quiet_NaN();
    // Prior influence (D4): summary scalars + count.
    int pri_count = 0;
    double total_prior_log_likelihood = std::numeric_limits<double>::quiet_NaN();
    double total_data_log_likelihood = std::numeric_limits<double>::quiet_NaN();
    double prior_to_data_ratio = std::numeric_limits<double>::quiet_NaN();
    double is_prior_influential = std::numeric_limits<double>::quiet_NaN();
    double mean_prior_precision_share = std::numeric_limits<double>::quiet_NaN();

    // --- X11 extended-analysis slice (the five new analyses + bootstrap + predictive checks).
    // Populated only by the run_extended_analysis targets; every other target leaves it empty. ---
    corehydro::analyses::support::ExtendedAnalysisResult ext;

    // --- T19: BootstrapDiagnostics slice (target == "Bulletin17CAnalysis" with
    // uncertainty_method "Bootstrap"/"BiasCorrectedBootstrap"). Populated from
    // Bulletin17CAnalysis::bootstrap_results() when non-null; every other uncertainty_method
    // leaves these at their defaults. ---
    bool boot_has_results = false;
    int boot_total_replicates = 0;
    int boot_attempted_replicates = 0;
    int boot_failed_replicates = 0;
    int boot_valid_replicates = 0;
    int boot_retained_replicates = 0;
    double boot_failure_rate = std::numeric_limits<double>::quiet_NaN();
    int boot_total_retries = 0;
    double boot_average_retries = std::numeric_limits<double>::quiet_NaN();
    int boot_pivot_rejections = 0;
    int boot_mahalanobis_rejections = 0;
    int boot_transform_failures = 0;
    int boot_status_success_count = 0;
    int boot_status_max_iterations_count = 0;
    int boot_status_max_function_evaluations_count = 0;
    int boot_status_failure_count = 0;
    int boot_status_none_count = 0;
    int boot_optimizer_fallbacks = 0;
};

// Applies the shared Bayesian MCMC knobs from a construct object (D5; mirrors the R/Python
// analysis glue). Used by the D5 per-family + diagnostics analysis branches.
static void apply_analysis_bayes_knobs(estimation::BayesianAnalysis& ba, const json& construct) {
    ba.set_type(parse_sampler_type(construct.value("sampler", std::string("DEMCzs"))));
    if (construct.contains("credible_level"))
        ba.set_credible_interval_width(construct["credible_level"].get<double>());
    if (construct.contains("seed")) ba.set_prng_seed(construct["seed"].get<int>());
    if (construct.contains("output_length")) ba.set_output_length(construct["output_length"].get<int>());
    if (construct.contains("iterations")) {
        int it = construct["iterations"].get<int>();
        ba.set_iterations(it);
        ba.set_warmup_iterations(std::max(50, it / 2));
    }
    if (construct.contains("thinning_interval"))
        ba.set_thinning_interval(construct["thinning_interval"].get<int>());
    if (construct.contains("number_of_chains"))
        ba.set_number_of_chains(construct["number_of_chains"].get<int>());
    if (construct.contains("initial_iterations"))
        ba.set_initial_iterations(construct["initial_iterations"].get<int>());
}

// Fills the UncertaintyAnalysisResults-shaped surface into `r` from an IUnivariateAnalysis-style
// analysis (mixture / competing-risk / point-process; D5). Mirrors ch_analysis_univariate_run_.
template <typename AnalysisT>
static void collect_univariate_family_results(AnalysisT& analysis, AnalysisResult& r) {
    const auto* results = analysis.analysis_results();
    if (results == nullptr) return;
    auto* pe = analysis.get_point_estimate_distribution();
    if (pe != nullptr) r.parameters = pe->get_parameters();
    r.mode_curve = results->mode_curve;
    r.mean_curve = results->mean_curve;
    for (const auto& ci : results->confidence_intervals) {
        r.lower_ci.push_back(ci[0]);
        r.upper_ci.push_back(ci[1]);
    }
    r.aic = results->aic;
    r.bic = results->bic;
    r.dic = results->dic;
    r.rmse = results->rmse;
}

// Builds + runs one univariate-family analysis (D5). Casts the ModelBase to the concrete model,
// hands ownership to the analysis, applies ordinates + Bayesian knobs, and collects the surface.
template <typename AnalysisT, typename ModelT>
static AnalysisResult run_univariate_family_analysis(std::unique_ptr<corehydro::models::ModelBase> base,
                                                     const json& construct,
                                                     const std::vector<double>& ep) {
    auto* raw = dynamic_cast<ModelT*>(base.get());
    if (raw == nullptr) throw std::runtime_error("analysis requires a matching model spec");
    base.release();
    std::unique_ptr<ModelT> model(raw);
    AnalysisT analysis(std::move(model));
    if (!ep.empty()) {
        analysis.probability_ordinates().clear();
        for (double p : ep) analysis.probability_ordinates().push_back(p);
    }
    apply_analysis_bayes_knobs(analysis.bayesian_analysis(), construct);
    analysis.run();
    AnalysisResult r;
    collect_univariate_family_results(analysis, r);
    return r;
}

// Builds + runs one time-series analysis (D5). Sets the optional training/forecasting horizons,
// applies Bayesian knobs, and reads the forecast curves; the point-estimate parameters come from
// the BayesianAnalysis posterior (the time-series analyses expose no distribution accessor).
template <typename AnalysisT, typename ModelT>
static AnalysisResult run_time_series_analysis(std::unique_ptr<corehydro::models::ModelBase> base,
                                               const json& construct) {
    auto* raw = dynamic_cast<ModelT*>(base.get());
    if (raw == nullptr) throw std::runtime_error("analysis requires a matching time_series model spec");
    base.release();
    std::unique_ptr<ModelT> model(raw);
    if (construct.contains("training_time_steps")) {
        model->set_use_default_training_steps(false);
        model->set_training_time_steps(construct["training_time_steps"].get<int>());
    }
    AnalysisT analysis(std::move(model));
    if (construct.contains("forecasting_time_steps"))
        analysis.set_forecasting_time_steps(construct["forecasting_time_steps"].get<int>());
    apply_analysis_bayes_knobs(analysis.bayesian_analysis(), construct);
    analysis.run();
    AnalysisResult r;
    const auto* results = analysis.analysis_results();
    if (results == nullptr) return r;
    const auto& ba = analysis.bayesian_analysis();
    if (ba.results()) {
        r.parameters = ba.point_estimator() == estimation::PointEstimateType::PosteriorMean
                           ? ba.results()->posterior_mean.values
                           : ba.results()->map.values;
    }
    r.mode_curve = results->mode_curve;
    r.mean_curve = results->mean_curve;
    for (const auto& ci : results->confidence_intervals) {
        r.lower_ci.push_back(ci[0]);
        r.upper_ci.push_back(ci[1]);
    }
    r.aic = results->aic;
    r.bic = results->bic;
    r.dic = results->dic;
    r.rmse = results->rmse;
    return r;
}

// Builds a Normal (or any univariate) model, runs a BayesianAnalysis, and computes all three
// diagnostics off that fit (D5). Mirrors ch_analysis_diagnostics_run_.
static AnalysisResult run_diagnostics_analysis(std::unique_ptr<corehydro::models::ModelBase> base,
                                               const json& construct) {
    corehydro::models::ModelBase& model = *base;
    estimation::BayesianAnalysis ba(model);
    ba.set_use_simulation_defaults(false);
    ba.set_use_advanced_simulation_defaults(false);
    apply_analysis_bayes_knobs(ba, construct);
    AnalysisResult r;
    if (!ba.estimate()) return r;

    auto lev = ba.compute_leverage_diagnostics();
    r.lev_count = lev.count();
    r.lev_prior_count = static_cast<int>(lev.prior_components().size());
    r.total_leverage = lev.total_leverage();
    r.total_fit_influence = lev.total_fit_influence();
    r.total_variance_influence = lev.total_variance_influence();
    for (const auto& o : lev.observations()) {
        r.lev_obs_leverage.push_back(o.leverage());
        r.lev_obs_fit.push_back(o.fit_influence());
        r.lev_obs_var.push_back(o.variance_influence());
        r.lev_obs_value.push_back(o.value());
    }

    auto inf = ba.compute_influence_diagnostics();
    r.inf_count = inf.count();
    r.mean_pareto_k = inf.mean_pareto_k();
    r.max_pareto_k = inf.max_pareto_k();
    r.count_pareto_k_above_05 = inf.count_pareto_k_above_05();
    r.count_pareto_k_above_07 = inf.count_pareto_k_above_07();
    r.count_pareto_k_above_10 = inf.count_pareto_k_above_10();
    r.proportion_problematic = inf.proportion_problematic();
    r.is_reliable = inf.is_reliable() ? 1.0 : 0.0;
    for (const auto& o : inf.observations()) {
        r.inf_pareto_k.push_back(o.pareto_k());
        r.inf_elpd_loo.push_back(o.elpd_loo());
    }

    auto pri = ba.compute_prior_influence_diagnostics(construct.value("thin_every", 10));
    r.pri_count = pri.count();
    r.total_prior_log_likelihood = pri.total_prior_log_likelihood();
    r.total_data_log_likelihood = pri.total_data_log_likelihood();
    r.prior_to_data_ratio = pri.prior_to_data_ratio();
    r.is_prior_influential = pri.is_prior_influential() ? 1.0 : 0.0;
    r.mean_prior_precision_share = pri.mean_prior_precision_share();
    return r;
}

static AnalysisResult build_and_run_analysis(const std::string& target, const json& construct,
                                             const json& datasets) {
    auto resolve_dataset = [&](const std::string& key) {
        std::vector<double> data;
        for (const auto& v : datasets[key]) data.push_back(parse_num(v));
        return data;
    };
    auto ordinates = [&]() {
        std::vector<double> ep;
        if (construct.contains("exceedance_probabilities"))
            for (const auto& v : construct["exceedance_probabilities"]) ep.push_back(parse_num(v));
        return ep;
    };
    auto apply_ordinates = [](corehydro::numerics::data::ProbabilityOrdinates& po,
                              const std::vector<double>& ep) {
        if (ep.empty()) return;
        po.clear();
        for (double p : ep) po.push_back(p);
    };

    AnalysisResult r;

    if (target == "FittingAnalysis") {
        std::vector<double> data = resolve_dataset(construct["dataset"].get<std::string>());
        auto df = std::make_unique<corehydro::models::DataFrame>();
        df->set_exact_series(corehydro::models::ExactSeries(data));
        df->calculate_plotting_positions();
        an::FittingAnalysis analysis(std::move(df));
        analysis.run();
        const auto& fitted = analysis.fitted_distributions();
        r.candidate_count = static_cast<int>(fitted.size());
        for (const auto& fd : fitted) {
            r.cand_aic.push_back(fd.aic());
            r.cand_bic.push_back(fd.bic());
            r.cand_rmse.push_back(fd.rmse());
            r.cand_converged.push_back(fd.fit_succeeded() ? 1.0 : 0.0);
        }
        return r;
    }

    // --- X11 extended analyses (composite / spatial_gev / bivariate / coincident / rating_curve /
    // bootstrap / prior + posterior predictive) through the shared run_extended_analysis path. The
    // C++ runner re-serializes the nlohmann construct + datasets to strings and calls the same
    // json_lite-parsing entry point the cpp11 / pybind bindings call, so all three agree. These
    // targets carry their own model shapes (inline spatial/rating/bivariate arrays, child family
    // lists), so they are handled BEFORE the shared model_spec/dataset extraction below. ---
    static const std::set<std::string> kExtendedTargets = {
        "CompositeAnalysis",           "SpatialGEVAnalysis",  "BivariateAnalysis",
        "CoincidentFrequencyAnalysis", "RatingCurveAnalysis", "BootstrapAnalysis",
        "PriorPredictiveCheck",        "PosteriorPredictiveCheck"};
    if (kExtendedTargets.count(target) > 0) {
        r.ext = corehydro::analyses::support::run_extended_analysis(target, construct.dump(),
                                                                 datasets.dump());
        // Mirror the shared UncertaintyAnalysisResults surface into the flat fields so the base
        // accessors (parameter / mode_curve / aic / ...) read the extended result too.
        r.parameters = r.ext.parameters;
        r.mode_curve = r.ext.mode_curve;
        r.mean_curve = r.ext.mean_curve;
        r.lower_ci = r.ext.lower_ci;
        r.upper_ci = r.ext.upper_ci;
        r.aic = r.ext.aic;
        r.bic = r.ext.bic;
        r.dic = r.ext.dic;
        r.rmse = r.ext.rmse;
        return r;
    }

    const json& model_spec = construct["model"];
    // T19: an inline `data_frame` (mixed exact/interval/threshold/uncertain series) is valid
    // without a `dataset` reference -- mirrors build_and_run_estimation's guard so a
    // Bulletin17CAnalysis case can force low outliers / censored data onto the parent frame
    // (e.g. to exercise the parametric-bootstrap clone_with_data_frame warm-start condition).
    std::vector<double> data;
    if (model_spec.contains("dataset"))
        data = resolve_dataset(model_spec["dataset"].get<std::string>());

    if (target == "UnivariateAnalysis") {
        auto base = corehydro::models::spec::build_model_from_json(model_spec.dump(), data);
        auto* raw = dynamic_cast<corehydro::models::UnivariateDistributionModel*>(base.get());
        if (raw == nullptr) throw std::runtime_error("UnivariateAnalysis requires a univariate_distribution model");
        base.release();
        std::unique_ptr<corehydro::models::UnivariateDistributionModel> model(raw);
        an::UnivariateAnalysis analysis(std::move(model));
        apply_ordinates(analysis.probability_ordinates(), ordinates());
        auto& ba = analysis.bayesian_analysis();
        ba.set_type(parse_sampler_type(construct.value("sampler", std::string("DEMCzs"))));
        if (construct.contains("credible_level"))
            ba.set_credible_interval_width(construct["credible_level"].get<double>());
        if (construct.contains("seed")) ba.set_prng_seed(construct["seed"].get<int>());
        if (construct.contains("output_length")) ba.set_output_length(construct["output_length"].get<int>());
        if (construct.contains("iterations")) {
            int it = construct["iterations"].get<int>();
            ba.set_iterations(it);
            ba.set_warmup_iterations(std::max(50, it / 2));
        }
        // Optional explicit MCMC knobs (A11): the default thinning_interval=20 exposes a
        // C#-vs-C++ divergence in the thinned population-sampler stream (documented in
        // docs/upstream-csharp-issues.md); the fixture pins thinning_interval=1 (bayes_normal's
        // proven bit-identical path). All four runners honor the same override.
        if (construct.contains("thinning_interval"))
            ba.set_thinning_interval(construct["thinning_interval"].get<int>());
        if (construct.contains("number_of_chains"))
            ba.set_number_of_chains(construct["number_of_chains"].get<int>());
        if (construct.contains("initial_iterations"))
            ba.set_initial_iterations(construct["initial_iterations"].get<int>());
        analysis.run();
        const auto* results = analysis.analysis_results();
        if (results != nullptr) {
            auto* pe = analysis.get_point_estimate_distribution();
            if (pe != nullptr) r.parameters = pe->get_parameters();
            r.mode_curve = results->mode_curve;
            r.mean_curve = results->mean_curve;
            for (const auto& ci : results->confidence_intervals) {
                r.lower_ci.push_back(ci[0]);
                r.upper_ci.push_back(ci[1]);
            }
            r.aic = results->aic;
            r.bic = results->bic;
            r.dic = results->dic;
            r.rmse = results->rmse;
        }
        return r;
    }

    // --- D5: per-family + diagnostics analyses (build the model via the shared spec builder,
    // run the matching analysis, collect into the flat surface). ---
    if (target == "MixtureAnalysis") {
        return run_univariate_family_analysis<an::MixtureAnalysis, corehydro::models::MixtureModel>(
            corehydro::models::spec::build_model_from_json(model_spec.dump(), data), construct,
            ordinates());
    }
    if (target == "CompetingRiskAnalysis") {
        return run_univariate_family_analysis<an::CompetingRiskAnalysis,
                                              corehydro::models::CompetingRisksModel>(
            corehydro::models::spec::build_model_from_json(model_spec.dump(), data), construct,
            ordinates());
    }
    if (target == "PointProcessAnalysis") {
        return run_univariate_family_analysis<an::PointProcessAnalysis,
                                              corehydro::models::PointProcessModel>(
            corehydro::models::spec::build_model_from_json(model_spec.dump(), data), construct,
            ordinates());
    }
    if (target == "ARAnalysis") {
        return run_time_series_analysis<an::ARAnalysis, corehydro::models::AutoRegressive>(
            corehydro::models::spec::build_model_from_json(model_spec.dump(), data), construct);
    }
    if (target == "MAAnalysis") {
        return run_time_series_analysis<an::MAAnalysis, corehydro::models::MovingAverage>(
            corehydro::models::spec::build_model_from_json(model_spec.dump(), data), construct);
    }
    if (target == "ARIMAAnalysis") {
        return run_time_series_analysis<an::ARIMAAnalysis, corehydro::models::ARIMA>(
            corehydro::models::spec::build_model_from_json(model_spec.dump(), data), construct);
    }
    if (target == "ARIMAXAnalysis") {
        return run_time_series_analysis<an::ARIMAXAnalysis, corehydro::models::ARIMAX>(
            corehydro::models::spec::build_model_from_json(model_spec.dump(), data), construct);
    }
    if (target == "Diagnostics") {
        return run_diagnostics_analysis(
            corehydro::models::spec::build_model_from_json(model_spec.dump(), data), construct);
    }

    if (target == "Bulletin17CAnalysis") {
        auto model = corehydro::models::spec::build_bulletin17c_from_json(model_spec.dump(), data);
        an::Bulletin17CAnalysis analysis(std::move(model));
        analysis.set_uncertainty_method(
            parse_uncertainty_method(construct.value("uncertainty_method", std::string("MultivariateNormal"))));
        apply_ordinates(analysis.probability_ordinates(), ordinates());
        auto& ba = analysis.bayesian_analysis();
        if (construct.contains("confidence_level"))
            ba.set_credible_interval_width(construct["confidence_level"].get<double>());
        if (construct.contains("seed")) ba.set_prng_seed(construct["seed"].get<int>());
        if (construct.contains("output_length")) ba.set_output_length(construct["output_length"].get<int>());
        analysis.run();
        auto ci = analysis.compute_cohn_style_confidence_intervals();
        if (ci.has_value()) {
            r.exceedance = ci->exceedance_probabilities;
            r.point_estimates = ci->point_estimates;
            r.lower_ci = ci->lower_ci;
            r.upper_ci = ci->upper_ci;
            r.beta1 = ci->beta1;
            r.nu = ci->nu;
            r.quantile_variance = ci->quantile_variance;
            r.confidence_level = ci->confidence_level;
        }
        if (analysis.gmm() != nullptr && analysis.gmm()->is_estimated())
            r.parameters = analysis.gmm()->best_parameter_set().values;
        // T19: the genuinely ensemble-derived UncertaintyAnalysisResults surface (distinct from
        // the RNG-free Cohn CI above) -- MeanCurve is built from the ACTUAL sampled parameter
        // sets (BayesianAnalysis.Results.Output), so unlike the Cohn CI it DOES depend on which
        // bootstrap replicates were drawn (hence on the parametric-bootstrap warm-start path).
        // Reuses the generic mode_curve/mean_curve dispatch every other analysis target already
        // shares -- no new fixture method needed.
        if (analysis.analysis_results() != nullptr) {
            r.mode_curve = analysis.analysis_results()->mode_curve;
            r.mean_curve = analysis.analysis_results()->mean_curve;
        }
        // T19: BootstrapDiagnostics slice, populated when the uncertainty method actually ran a
        // bootstrap arm (Bootstrap / BiasCorrectedBootstrap).
        if (const auto* boot = analysis.bootstrap_results(); boot != nullptr) {
            r.boot_has_results = true;
            r.boot_total_replicates = boot->total_replicates();
            r.boot_attempted_replicates = boot->attempted_replicates();
            r.boot_failed_replicates = boot->failed_replicates();
            r.boot_valid_replicates = boot->valid_replicates();
            r.boot_retained_replicates = boot->retained_replicates();
            r.boot_failure_rate = boot->failure_rate();
            r.boot_total_retries = boot->total_retries();
            r.boot_average_retries = boot->average_retries();
            r.boot_pivot_rejections = boot->pivot_rejections();
            r.boot_mahalanobis_rejections = boot->mahalanobis_rejections();
            r.boot_transform_failures = boot->transform_failures();
            r.boot_status_success_count = boot->status_success_count();
            r.boot_status_max_iterations_count = boot->status_maximum_iterations_count();
            r.boot_status_max_function_evaluations_count =
                boot->status_maximum_function_evaluations_count();
            r.boot_status_failure_count = boot->status_failure_count();
            r.boot_status_none_count = boot->status_none_count();
            r.boot_optimizer_fallbacks = boot->optimizer_fallbacks();
        }
        return r;
    }

    throw std::runtime_error("unknown analysis target: " + target);
}

static double dispatch_analysis(const AnalysisResult& r, const std::string& m, const json& a) {
    auto idx = [&](int i) { return static_cast<std::size_t>(a[static_cast<std::size_t>(i)].get<int>()); };
    if (m == "candidate_count") return static_cast<double>(r.candidate_count);
    if (m == "candidate_aic") return r.cand_aic.at(idx(0));
    if (m == "candidate_bic") return r.cand_bic.at(idx(0));
    if (m == "candidate_rmse") return r.cand_rmse.at(idx(0));
    if (m == "candidate_converged") return r.cand_converged.at(idx(0));
    if (m == "parameter") return r.parameters.at(idx(0));
    if (m == "mode_curve") return r.mode_curve.at(idx(0));
    if (m == "mean_curve") return r.mean_curve.at(idx(0));
    if (m == "lower_ci") return r.lower_ci.at(idx(0));
    if (m == "upper_ci") return r.upper_ci.at(idx(0));
    if (m == "exceedance_probability") return r.exceedance.at(idx(0));
    if (m == "point_estimate") return r.point_estimates.at(idx(0));
    if (m == "beta1") return r.beta1.at(idx(0));
    if (m == "nu") return r.nu.at(idx(0));
    if (m == "quantile_variance") return r.quantile_variance.at(idx(0));
    if (m == "aic") return r.aic;
    if (m == "bic") return r.bic;
    if (m == "dic") return r.dic;
    if (m == "rmse") return r.rmse;
    if (m == "confidence_level") return r.confidence_level;
    // D5: time-series curve length (structural invariant).
    if (m == "curve_length") return static_cast<double>(r.mode_curve.size());
    // D5: leverage diagnostics.
    if (m == "leverage_count") return static_cast<double>(r.lev_count);
    if (m == "leverage_prior_count") return static_cast<double>(r.lev_prior_count);
    if (m == "total_leverage") return r.total_leverage;
    if (m == "total_fit_influence") return r.total_fit_influence;
    if (m == "total_variance_influence") return r.total_variance_influence;
    if (m == "obs_leverage") return r.lev_obs_leverage.at(idx(0));
    if (m == "obs_fit_influence") return r.lev_obs_fit.at(idx(0));
    if (m == "obs_variance_influence") return r.lev_obs_var.at(idx(0));
    if (m == "obs_value") return r.lev_obs_value.at(idx(0));
    // D5: influence (PSIS-LOO) diagnostics.
    if (m == "influence_count") return static_cast<double>(r.inf_count);
    if (m == "mean_pareto_k") return r.mean_pareto_k;
    if (m == "max_pareto_k") return r.max_pareto_k;
    if (m == "count_pareto_k_above_05") return static_cast<double>(r.count_pareto_k_above_05);
    if (m == "count_pareto_k_above_07") return static_cast<double>(r.count_pareto_k_above_07);
    if (m == "count_pareto_k_above_10") return static_cast<double>(r.count_pareto_k_above_10);
    if (m == "proportion_problematic") return r.proportion_problematic;
    if (m == "is_reliable") return r.is_reliable;
    if (m == "pareto_k") return r.inf_pareto_k.at(idx(0));
    if (m == "elpd_loo") return r.inf_elpd_loo.at(idx(0));
    // D5: prior influence diagnostics.
    if (m == "prior_influence_count") return static_cast<double>(r.pri_count);
    if (m == "total_prior_log_likelihood") return r.total_prior_log_likelihood;
    if (m == "total_data_log_likelihood") return r.total_data_log_likelihood;
    if (m == "prior_to_data_ratio") return r.prior_to_data_ratio;
    if (m == "is_prior_influential") return r.is_prior_influential;
    if (m == "mean_prior_precision_share") return r.mean_prior_precision_share;
    // X11 extended analyses.
    if (m == "z_output") return r.ext.z_output_values.at(idx(0));
    if (m == "z_output_length") return static_cast<double>(r.ext.z_output_values.size());
    if (m == "site_count") return static_cast<double>(r.ext.site_count);
    if (m == "site_location_mean") return r.ext.site_location_mean.at(idx(0));
    if (m == "site_scale_mean") return r.ext.site_scale_mean.at(idx(0));
    if (m == "site_shape_mean") return r.ext.site_shape_mean.at(idx(0));
    if (m == "site_quantile_mean") return r.ext.site0_quantile_mean.at(idx(0));
    if (m == "cv_mae") return r.ext.cv_mae;
    if (m == "cv_rmse") return r.ext.cv_rmse;
    if (m == "cv_mean_bias") return r.ext.cv_mean_bias;
    if (m == "mean_p_value") return r.ext.mean_p_value;
    if (m == "sd_p_value") return r.ext.sd_p_value;
    if (m == "skewness_p_value") return r.ext.skewness_p_value;
    if (m == "min_p_value") return r.ext.min_p_value;
    if (m == "max_p_value") return r.ext.max_p_value;
    if (m == "predictive_replicates") return static_cast<double>(r.ext.number_of_replicates);
    if (m == "has_misfit") return r.ext.has_misfit;
    if (m == "number_of_valid_draws") return static_cast<double>(r.ext.number_of_valid_draws);
    if (m == "summary_mean_quantile") return r.ext.summary_mean_quantiles.at(idx(0));
    if (m == "summary_sd_quantile") return r.ext.summary_sd_quantiles.at(idx(0));
    if (m == "summary_min_quantile") return r.ext.summary_min_quantiles.at(idx(0));
    if (m == "summary_max_quantile") return r.ext.summary_max_quantiles.at(idx(0));
    // T19: BootstrapDiagnostics dispatch (Bulletin17CAnalysis, Bootstrap/BiasCorrectedBootstrap).
    if (m == "boot_has_results") return r.boot_has_results ? 1.0 : 0.0;
    if (m == "boot_total_replicates") return static_cast<double>(r.boot_total_replicates);
    if (m == "boot_attempted_replicates") return static_cast<double>(r.boot_attempted_replicates);
    if (m == "boot_failed_replicates") return static_cast<double>(r.boot_failed_replicates);
    if (m == "boot_valid_replicates") return static_cast<double>(r.boot_valid_replicates);
    if (m == "boot_retained_replicates") return static_cast<double>(r.boot_retained_replicates);
    if (m == "boot_failure_rate") return r.boot_failure_rate;
    if (m == "boot_total_retries") return static_cast<double>(r.boot_total_retries);
    if (m == "boot_average_retries") return r.boot_average_retries;
    if (m == "boot_pivot_rejections") return static_cast<double>(r.boot_pivot_rejections);
    if (m == "boot_mahalanobis_rejections") return static_cast<double>(r.boot_mahalanobis_rejections);
    if (m == "boot_transform_failures") return static_cast<double>(r.boot_transform_failures);
    if (m == "boot_status_success_count") return static_cast<double>(r.boot_status_success_count);
    if (m == "boot_status_max_iterations_count")
        return static_cast<double>(r.boot_status_max_iterations_count);
    if (m == "boot_status_max_function_evaluations_count")
        return static_cast<double>(r.boot_status_max_function_evaluations_count);
    if (m == "boot_status_failure_count") return static_cast<double>(r.boot_status_failure_count);
    if (m == "boot_status_none_count") return static_cast<double>(r.boot_status_none_count);
    if (m == "boot_optimizer_fallbacks") return static_cast<double>(r.boot_optimizer_fallbacks);
    throw std::runtime_error("unknown analysis fixture method: " + m);
}

static void run_analysis(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        AnalysisResult r = build_and_run_analysis(target, c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = target + "/" + name + "/" + method;
            check_value(dispatch_analysis(r, method, args), as, where);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <fixtures-dir>\n", argv[0]);
        return 2;
    }
    g_sobol_path = (fs::path(argv[1]) / ".." / "core" / "data" / "new-joe-kuo-6.21201").string();
    int files = 0;
    for (const auto& entry : fs::recursive_directory_iterator(argv[1])) {
        if (entry.path().extension() != ".json") continue;
        ++files;
        std::ifstream in(entry.path());
        json spec = json::parse(in);
        std::string kind = spec.value("kind", "");
        if (kind == "special_function") {
            run_special_function(spec);
        } else if (kind == "goodness_of_fit") {
            run_goodness_of_fit(spec);
        } else if (kind == "data_utility") {
            run_data_utility(spec);
        } else if (kind == "toolbox") {
            run_toolbox_kind(spec);
        } else if (kind == "optimizer") {
            run_optimizer_kind(spec);
        } else if (kind == "callback") {
            run_callback_kind(spec);
        } else if (kind == "callback_cross_language") {
            run_callback_cross_language_kind(spec);
        } else if (kind == "toolbox_cross_language") {
            run_toolbox_cross_language_kind(spec);
        } else if (kind == "univariate_distribution") {
            if (spec.value("target", "") == "GeneralizedExtremeValue")
                run_gev(spec);
            else
                run_generic(spec);
        } else if (kind == "multivariate_distribution") {
            run_multivariate(spec);
        } else if (kind == "bivariate_copula") {
            run_bivariate_copula(spec);
        } else if (kind == "mcmc_sampler") {
            run_mcmc_sampler(spec);
        } else if (kind == "bootstrap") {
            run_bootstrap(spec);
        } else if (kind == "model_estimation") {
            run_model_estimation(spec);
        } else if (kind == "analysis") {
            run_analysis(spec);
        }
    }
    if (files == 0) {
        std::fprintf(stderr, "no fixtures found under %s\n", argv[1]);
        return 2;
    }
    return chtest::summary("fixtures");
}
