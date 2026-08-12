// cpp11 glue exposing the polymorphic univariate-distribution surface of the shared C++
// core to R (Normal, Uniform, Exponential, ... -- everything built on
// UnivariateDistributionBase + the factory). GEV keeps its own bespoke glue in gev.cpp.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <memory>
#include <string>
#include <vector>

#include "corehydro/numerics/data/probability.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/competing_risks.hpp"
#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/kernel_density.hpp"

namespace dist = corehydro::numerics::distributions;
using namespace cpp11;

static std::unique_ptr<dist::UnivariateDistributionBase> make_dist(const std::string& target,
                                                                   doubles params) {
    auto d = dist::create_distribution(target);
    d->set_parameters(std::vector<double>(params.begin(), params.end()));
    return d;
}

static dist::ParameterEstimationMethod parse_method(const std::string& m) {
    if (m == "mom" || m == "moments") return dist::ParameterEstimationMethod::MethodOfMoments;
    if (m == "lmom" || m == "lmoments")
        return dist::ParameterEstimationMethod::MethodOfLinearMoments;
    if (m == "mle") return dist::ParameterEstimationMethod::MaximumLikelihood;
    stop("unknown estimation method '%s' (use 'mom', 'lmom', or 'mle')", m.c_str());
}

[[cpp11::register]]
doubles ch_dist_moments_(std::string target, doubles params) {
    auto d = make_dist(target, params);
    writable::doubles out({d->mean(), d->median(), d->mode(), d->standard_deviation(),
                           d->skewness(), d->kurtosis(), d->minimum(), d->maximum()});
    out.names() = {"mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"};
    return out;
}

[[cpp11::register]]
double ch_dist_pdf_(std::string target, doubles params, double x) {
    return make_dist(target, params)->pdf(x);
}

[[cpp11::register]]
double ch_dist_cdf_(std::string target, doubles params, double x) {
    return make_dist(target, params)->cdf(x);
}

[[cpp11::register]]
double ch_dist_quantile_(std::string target, doubles params, double p) {
    return make_dist(target, params)->inverse_cdf(p);
}

[[cpp11::register]]
bool ch_dist_valid_(std::string target, doubles params) {
    return make_dist(target, params)->parameters_valid();
}

[[cpp11::register]]
doubles ch_dist_fit_(std::string target, doubles data, std::string method) {
    auto d = dist::create_distribution(target);
    auto* est = dynamic_cast<dist::IEstimation*>(d.get());
    if (est == nullptr) stop("distribution '%s' does not support estimation", target.c_str());
    est->estimate(std::vector<double>(data.begin(), data.end()), parse_method(method));
    return writable::doubles(d->get_parameters());
}

[[cpp11::register]]
doubles ch_dist_linear_moments_(std::string target, doubles params) {
    auto d = make_dist(target, params);
    auto* lm = dynamic_cast<dist::ILinearMomentEstimation*>(d.get());
    if (lm == nullptr) stop("distribution '%s' has no L-moments", target.c_str());
    return writable::doubles(lm->linear_moments_from_parameters(d->get_parameters()));
}

// GammaDistribution::partial_kp is a static utility (not tied to any distribution
// instance's own parameters) used by the fixture runner to pin the v2.1.4
// near-zero-skew derivative-limit fix; not otherwise part of the public API.
[[cpp11::register]]
double ch_dist_gamma_partial_kp_(double skewness, double probability) {
    return dist::GammaDistribution::partial_kp(skewness, probability);
}

// --- Public-API additions (consumed by R/distribution.R) ----------------------------
// The scalar entry points above are kept untouched for the fixture runner; these
// vectorized variants construct the distribution once and loop in C++.

[[cpp11::register]]
doubles ch_dist_random_(std::string target, doubles params, int sample_size, int seed) {
    return writable::doubles(make_dist(target, params)->generate_random_values(sample_size, seed));
}

[[cpp11::register]]
doubles ch_dist_pdf_v_(std::string target, doubles params, doubles xs) {
    auto d = make_dist(target, params);
    writable::doubles out(xs.size());
    for (R_xlen_t i = 0; i < xs.size(); ++i) out[i] = d->pdf(xs[i]);
    return out;
}

[[cpp11::register]]
doubles ch_dist_cdf_v_(std::string target, doubles params, doubles xs) {
    auto d = make_dist(target, params);
    writable::doubles out(xs.size());
    for (R_xlen_t i = 0; i < xs.size(); ++i) out[i] = d->cdf(xs[i]);
    return out;
}

[[cpp11::register]]
doubles ch_dist_quantile_v_(std::string target, doubles params, doubles probs) {
    auto d = make_dist(target, params);
    writable::doubles out(probs.size());
    for (R_xlen_t i = 0; i < probs.size(); ++i) out[i] = d->inverse_cdf(probs[i]);
    return out;
}

[[cpp11::register]]
doubles ch_dist_log_pdf_v_(std::string target, doubles params, doubles xs) {
    auto d = make_dist(target, params);
    writable::doubles out(xs.size());
    for (R_xlen_t i = 0; i < xs.size(); ++i) out[i] = d->log_pdf(xs[i]);
    return out;
}

[[cpp11::register]]
double ch_dist_log_likelihood_(std::string target, doubles params, doubles data) {
    return make_dist(target, params)
        ->log_likelihood(std::vector<double>(data.begin(), data.end()));
}

[[cpp11::register]]
list ch_dist_parameter_names_(std::string target) {
    auto d = dist::create_distribution(target);
    writable::strings full;
    for (const auto& s : d->parameter_names()) full.push_back(s);
    writable::strings short_form;
    for (const auto& s : d->parameter_names_short_form()) short_form.push_back(s);
    writable::list out({full, short_form});
    out.names() = {"full", "short"};
    return out;
}

[[cpp11::register]]
strings ch_dist_names_() {
    writable::strings out;
    for (const auto& s : dist::distribution_names()) out.push_back(s);
    return out;
}

// --- Composite glue: the fixture cases the shared distribution runner cannot serve --------
// Everything else about a composite distribution (TruncatedDistribution / Empirical /
// KernelDensity / Mixture / CompetingRisks) now goes through ch_dist_spec_run_ in dist_spec.cpp,
// which drives the same dist_runner.hpp entry point the C++ fixture runner, the Python glue and
// the dotnet oracle emitter drive. Three fixture cases cannot: the runner's JSON grammar has no
// NaN/Infinity literal, so Empirical's non-finite `p` and KernelDensity's non-finite `bandwidth`
// validity cases keep a narrow constructor call here, and `dependency_change` (which mutates one
// CompetingRisks object mid-call) has no runner verb. See test-fixtures.R's delegation comment.

// --- Composite glue: EmpiricalDistribution ------------------------------------------
// p_transform: "NormalZ" (default) or "None". p_descending (v2.1.4): DECLARES the
// probability order (mirrors C#'s explicit `probabilityOrder` argument -- NOT auto-detected
// from the data; see empirical_distribution.hpp's header note) -- false (the ordinary
// ascending-CDF case) unless the fixture opts into the descending survival-function encoding.

static dist::EmpiricalDistribution make_empirical(doubles x_vals, doubles p_vals,
                                                   std::string p_transform, bool p_descending) {
    std::vector<double> xv(x_vals.begin(), x_vals.end());
    std::vector<double> pv(p_vals.begin(), p_vals.end());
    dist::EmpiricalTransform pt = dist::EmpiricalTransform::NormalZ;
    if (p_transform == "None") pt = dist::EmpiricalTransform::None;
    return dist::EmpiricalDistribution(std::move(xv), std::move(pv), pt, p_descending);
}

[[cpp11::register]]
bool ch_emp_valid_(doubles x_vals, doubles p_vals, std::string p_transform, bool p_descending) {
    return make_empirical(x_vals, p_vals, p_transform, p_descending).parameters_valid();
}

// --- Composite glue: KernelDensity --------------------------------------------------
// kernel: "Gaussian" | "Epanechnikov" | "Triangular" | "Uniform"
// bandwidth: negative value means use Silverman's rule (auto).

static dist::KernelType parse_kernel_type(const std::string& s) {
    if (s == "Epanechnikov") return dist::KernelType::Epanechnikov;
    if (s == "Gaussian")     return dist::KernelType::Gaussian;
    if (s == "Triangular")   return dist::KernelType::Triangular;
    if (s == "Uniform")      return dist::KernelType::Uniform;
    stop("unknown kernel type '%s'", s.c_str());
}

static dist::KernelDensity make_kde(doubles data, std::string kernel,
                                    double bandwidth, bool bounded_by_data) {
    std::vector<double> dv(data.begin(), data.end());
    dist::KernelType kt = parse_kernel_type(kernel);
    dist::KernelDensity kde = bandwidth < 0.0
        ? dist::KernelDensity(dv, kt)
        : dist::KernelDensity(dv, kt, bandwidth);
    kde.set_bounded_by_data(bounded_by_data);
    return kde;
}

[[cpp11::register]]
bool ch_kde_valid_(doubles data, std::string kernel, double bandwidth, bool bounded_by_data) {
    return make_kde(data, kernel, bandwidth, bounded_by_data).parameters_valid();
}

// --- Composite glue: CompetingRisks -------------------------------------------------
// Accepts (component_targets, component_params_list, minimum_of_rv, dependency,
// correlation). component_params_list is a list-of-doubles R list. minimum_of_rv = TRUE for
// min-of-components (series system, default); FALSE for max-of-components (parallel system).
// dependency is one of "Independent"/"PerfectlyPositive"/"PerfectlyNegative"/
// "CorrelationMatrix"; correlation is a list of numeric row vectors (only consulted when
// dependency == "CorrelationMatrix", may be an empty list otherwise).

static corehydro::numerics::data::probability::DependencyType parse_dependency(
    const std::string& d) {
    namespace prob = corehydro::numerics::data::probability;
    if (d == "Independent") return prob::DependencyType::Independent;
    if (d == "PerfectlyPositive") return prob::DependencyType::PerfectlyPositive;
    if (d == "PerfectlyNegative") return prob::DependencyType::PerfectlyNegative;
    if (d == "CorrelationMatrix") return prob::DependencyType::CorrelationMatrix;
    stop("unknown dependency type '%s'", d.c_str());
}

static dist::CompetingRisks make_competing_risks(strings comp_targets,
                                                  list comp_params_list,
                                                  bool minimum_of_rv,
                                                  const std::string& dependency,
                                                  list correlation) {
    int K = static_cast<int>(comp_targets.size());
    std::vector<std::unique_ptr<dist::UnivariateDistributionBase>> comps;
    comps.reserve(K);
    for (int i = 0; i < K; ++i) {
        auto d = dist::create_distribution(std::string(comp_targets[i]));
        doubles p = comp_params_list[i];
        d->set_parameters(std::vector<double>(p.begin(), p.end()));
        comps.push_back(std::move(d));
    }
    dist::CompetingRisks cr(std::move(comps));
    cr.minimum_of_random_variables = minimum_of_rv;
    cr.set_dependency(parse_dependency(dependency));
    // Unconditional on `dependency` (matches test_fixtures.cpp's build_composite and the
    // dotnet emitter's BuildComposite) -- a caller may set CorrelationMatrix up front and
    // only later switch Dependency to CorrelationMatrix (the v2.1.4
    // dependency_change fixture), so gating this on `dependency == "CorrelationMatrix"`
    // left correlation_matrix_ empty for any OTHER initial dependency, an
    // out-of-bounds read waiting to happen the first time a fixture read it back.
    if (correlation.size() > 0) {
        corehydro::numerics::data::probability::Matrix2D corr;
        corr.reserve(correlation.size());
        for (R_xlen_t i = 0; i < correlation.size(); ++i) {
            doubles row = correlation[i];
            corr.emplace_back(row.begin(), row.end());
        }
        cr.set_correlation_matrix(std::move(corr));
    }
    return cr;
}

// v2.1.4: verifies the Dependency setter fix (changing Dependency mid-lifetime invalidates
// the cached MVN) and that PerfectlyNegative no longer zeroes the public CorrelationMatrix.
// ONE self-contained call (R has no persistent object across fixture assertions): CDF
// under the FIRST dependency, read CorrelationMatrix[i, j] back, then switch to
// `dependency2` and CDF again -- returns the value the fixture asked for via `field`
// ("cdf1", "correlation", "cdf2").
[[cpp11::register]]
double ch_cr_dependency_change_(strings comp_targets, list comp_params_list, bool minimum_of_rv,
                                 std::string dependency, std::string dependency2, list correlation,
                                 double x, std::string field, int i, int j) {
    auto cr = make_competing_risks(comp_targets, comp_params_list, minimum_of_rv, dependency, correlation);
    double cdf1 = cr.cdf(x);
    double corr_ij = cr.correlation_matrix()[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    cr.set_dependency(parse_dependency(dependency2));
    double cdf2 = cr.cdf(x);
    if (field == "cdf1") return cdf1;
    if (field == "correlation") return corr_ij;
    if (field == "cdf2") return cdf2;
    stop("unknown ch_cr_dependency_change_ field '%s'", field.c_str());
}
