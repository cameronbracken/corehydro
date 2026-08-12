// pybind11 glue exposing the polymorphic univariate-distribution surface of the shared
// C++ core to Python (Normal, Uniform, Exponential, ... -- everything on
// UnivariateDistributionBase + the factory). GEV keeps its own bespoke class in gev.cpp.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
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
#include "bindings.hpp"

namespace py = pybind11;
namespace dist = corehydro::numerics::distributions;

static std::unique_ptr<dist::UnivariateDistributionBase> make_dist(
    const std::string& target, const std::vector<double>& params) {
    auto d = dist::create_distribution(target);
    d->set_parameters(params);
    return d;
}

static dist::ParameterEstimationMethod parse_method(const std::string& m) {
    if (m == "mom" || m == "moments") return dist::ParameterEstimationMethod::MethodOfMoments;
    if (m == "lmom" || m == "lmoments")
        return dist::ParameterEstimationMethod::MethodOfLinearMoments;
    if (m == "mle") return dist::ParameterEstimationMethod::MaximumLikelihood;
    throw py::value_error("unknown estimation method '" + m + "' (use 'mom', 'lmom', or 'mle')");
}

void register_distributions(py::module_& m) {
    m.def("dist_moments", [](const std::string& target, const std::vector<double>& params) {
        auto d = make_dist(target, params);
        // std::map keeps a stable, language-neutral key set; the Python wrapper orders output.
        return std::map<std::string, double>{
            {"mean", d->mean()},        {"median", d->median()},
            {"mode", d->mode()},        {"sd", d->standard_deviation()},
            {"skewness", d->skewness()}, {"kurtosis", d->kurtosis()},
            {"minimum", d->minimum()},  {"maximum", d->maximum()}};
    });
    m.def("dist_pdf", [](const std::string& t, const std::vector<double>& p, double x) {
        return make_dist(t, p)->pdf(x);
    });
    m.def("dist_cdf", [](const std::string& t, const std::vector<double>& p, double x) {
        return make_dist(t, p)->cdf(x);
    });
    m.def("dist_quantile", [](const std::string& t, const std::vector<double>& p, double prob) {
        return make_dist(t, p)->inverse_cdf(prob);
    });
    m.def("dist_valid", [](const std::string& t, const std::vector<double>& p) {
        return make_dist(t, p)->parameters_valid();
    });
    m.def("dist_fit", [](const std::string& target, const std::vector<double>& data,
                         const std::string& method) {
        auto d = dist::create_distribution(target);
        auto* est = dynamic_cast<dist::IEstimation*>(d.get());
        if (est == nullptr)
            throw py::value_error("distribution '" + target + "' does not support estimation");
        est->estimate(data, parse_method(method));
        return d->get_parameters();
    });
    m.def("dist_linear_moments", [](const std::string& t, const std::vector<double>& p) {
        auto d = make_dist(t, p);
        auto* lm = dynamic_cast<dist::ILinearMomentEstimation*>(d.get());
        if (lm == nullptr) throw py::value_error("distribution '" + t + "' has no L-moments");
        return lm->linear_moments_from_parameters(d->get_parameters());
    });
    // GammaDistribution::partial_kp is a static utility (not tied to any distribution
    // instance's own parameters) used by the fixture runner to pin the v2.1.4
    // near-zero-skew derivative-limit fix; not otherwise part of the public API.
    m.def("dist_gamma_partial_kp", [](double skewness, double probability) {
        return dist::GammaDistribution::partial_kp(skewness, probability);
    });

    // --- Public-API additions (consumed by corehydropy.distributions) --------------------
    // The scalar entry points above are kept untouched for the fixture runners; these
    // vectorized variants construct the distribution once and loop in C++.

    m.def("dist_random", [](const std::string& t, const std::vector<double>& p,
                            int sample_size, int seed) {
        return make_dist(t, p)->generate_random_values(sample_size, seed);
    });
    m.def("dist_pdf_v", [](const std::string& t, const std::vector<double>& p,
                           const std::vector<double>& xs) {
        auto d = make_dist(t, p);
        std::vector<double> out(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = d->pdf(xs[i]);
        return out;
    });
    m.def("dist_cdf_v", [](const std::string& t, const std::vector<double>& p,
                           const std::vector<double>& xs) {
        auto d = make_dist(t, p);
        std::vector<double> out(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = d->cdf(xs[i]);
        return out;
    });
    m.def("dist_quantile_v", [](const std::string& t, const std::vector<double>& p,
                                const std::vector<double>& probs) {
        auto d = make_dist(t, p);
        std::vector<double> out(probs.size());
        for (std::size_t i = 0; i < probs.size(); ++i) out[i] = d->inverse_cdf(probs[i]);
        return out;
    });
    m.def("dist_log_pdf_v", [](const std::string& t, const std::vector<double>& p,
                               const std::vector<double>& xs) {
        auto d = make_dist(t, p);
        std::vector<double> out(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = d->log_pdf(xs[i]);
        return out;
    });
    m.def("dist_log_likelihood", [](const std::string& t, const std::vector<double>& p,
                                    const std::vector<double>& data) {
        return make_dist(t, p)->log_likelihood(data);
    });
    m.def("dist_parameter_names", [](const std::string& t) {
        auto d = dist::create_distribution(t);
        return std::map<std::string, std::vector<std::string>>{
            {"full", d->parameter_names()}, {"short", d->parameter_names_short_form()}};
    });
    m.def("dist_names", []() { return dist::distribution_names(); });

    // --- Composite glue: the fixture cases the shared runner cannot serve ---------------
    // Everything else about a composite distribution (TruncatedDistribution / Empirical /
    // KernelDensity / Mixture / CompetingRisks) now goes through _core.dist_spec_run in
    // dist_spec.cpp, which drives the same dist_runner.hpp entry point the C++ fixture runner,
    // the R glue and the dotnet oracle emitter drive. Three fixture cases cannot: the runner's
    // JSON grammar has no NaN/Infinity literal, so Empirical's non-finite `p` and
    // KernelDensity's non-finite `bandwidth` validity cases keep a narrow constructor call
    // here, and `dependency_change` (which mutates one CompetingRisks object mid-call) has no
    // runner verb. See test_fixtures.py's delegation comment.

    // --- Composite glue: EmpiricalDistribution -----------------------------------------
    // Accepts (x_vals, p_vals, p_transform, p_descending). p_transform: "NormalZ" (default)
    // or "None". p_descending (v2.1.4): DECLARES
    // the probability order (mirrors C#'s explicit `probabilityOrder` argument -- NOT
    // auto-detected from the data; see empirical_distribution.hpp's header note) -- false
    // (the ordinary ascending-CDF case) unless the fixture opts into the descending
    // survival-function encoding.

    auto parse_emp_transform = [](const std::string& s) {
        if (s == "None") return dist::EmpiricalTransform::None;
        if (s == "NormalZ") return dist::EmpiricalTransform::NormalZ;
        throw py::value_error("unknown p_transform: " + s);
        return dist::EmpiricalTransform::NormalZ;  // unreachable
    };

    m.def("emp_valid", [parse_emp_transform](const std::vector<double>& xv,
                                              const std::vector<double>& pv,
                                              const std::string& pt_str, bool p_descending) {
        return dist::EmpiricalDistribution(xv, pv, parse_emp_transform(pt_str), p_descending)
            .parameters_valid();
    });

    // --- Composite glue: KernelDensity -------------------------------------------------
    // Accepts (data, kernel, bandwidth, bounded_by_data).
    // kernel: "Gaussian" | "Epanechnikov" | "Triangular" | "Uniform"
    // bandwidth: negative value means use Silverman's rule (auto).

    auto parse_kernel_type = [](const std::string& s) {
        if (s == "Epanechnikov") return dist::KernelType::Epanechnikov;
        if (s == "Gaussian")     return dist::KernelType::Gaussian;
        if (s == "Triangular")   return dist::KernelType::Triangular;
        if (s == "Uniform")      return dist::KernelType::Uniform;
        throw py::value_error("unknown kernel type: " + s);
        return dist::KernelType::Gaussian;  // unreachable
    };

    auto make_kde = [parse_kernel_type](const std::vector<double>& data,
                                        const std::string& kernel,
                                        double bandwidth,
                                        bool bounded_by_data) {
        dist::KernelType kt = parse_kernel_type(kernel);
        dist::KernelDensity kde = bandwidth < 0.0
            ? dist::KernelDensity(data, kt)
            : dist::KernelDensity(data, kt, bandwidth);
        kde.set_bounded_by_data(bounded_by_data);
        return kde;
    };

    m.def("kde_valid", [make_kde](const std::vector<double>& data, const std::string& kernel,
                                   double bandwidth, bool bounded_by_data) {
        return make_kde(data, kernel, bandwidth, bounded_by_data).parameters_valid();
    });

    // --- Composite glue: CompetingRisks -----------------------------------------------
    // Accepts (comp_targets, comp_params, minimum_of_rv, dependency, correlation) where
    // comp_params is a list of param vectors (one per component). minimum_of_rv=true ->
    // min-of-components (default); false -> max-of-components. dependency is one of
    // "Independent"/"PerfectlyPositive"/"PerfectlyNegative"/"CorrelationMatrix";
    // correlation is a list of row vectors (only consulted when dependency ==
    // "CorrelationMatrix", may be empty otherwise).

    auto parse_dependency = [](const std::string& d) {
        namespace prob = corehydro::numerics::data::probability;
        if (d == "Independent") return prob::DependencyType::Independent;
        if (d == "PerfectlyPositive") return prob::DependencyType::PerfectlyPositive;
        if (d == "PerfectlyNegative") return prob::DependencyType::PerfectlyNegative;
        if (d == "CorrelationMatrix") return prob::DependencyType::CorrelationMatrix;
        throw py::value_error("unknown dependency type '" + d + "'");
    };

    auto make_competing_risks = [parse_dependency](
                                     const std::vector<std::string>& comp_targets,
                                     const std::vector<std::vector<double>>& comp_params,
                                     bool minimum_of_rv, const std::string& dependency,
                                     const std::vector<std::vector<double>>& correlation) {
        int K = static_cast<int>(comp_targets.size());
        std::vector<std::unique_ptr<dist::UnivariateDistributionBase>> comps;
        comps.reserve(K);
        for (int i = 0; i < K; ++i) {
            auto d = dist::create_distribution(comp_targets[i]);
            d->set_parameters(comp_params[i]);
            comps.push_back(std::move(d));
        }
        dist::CompetingRisks cr(std::move(comps));
        cr.minimum_of_random_variables = minimum_of_rv;
        cr.set_dependency(parse_dependency(dependency));
        // Unconditional on `dependency` (matches test_fixtures.cpp's build_composite and
        // the dotnet emitter's BuildComposite) -- a caller may set CorrelationMatrix up
        // front and only later switch Dependency to CorrelationMatrix (the v2.1.4
        // dependency_change fixture), so gating this on `dependency == "CorrelationMatrix"`
        // left correlation_matrix_ empty for any OTHER initial dependency, an
        // out-of-bounds read waiting to happen the first time a fixture read it back.
        if (!correlation.empty()) cr.set_correlation_matrix(correlation);
        return cr;
    };

    m.def("cr_dependency_change", [make_competing_risks, parse_dependency](
                                       const std::vector<std::string>& ct,
                                       const std::vector<std::vector<double>>& cp, bool minimum_of_rv,
                                       const std::string& dependency, const std::string& dependency2,
                                       const std::vector<std::vector<double>>& correlation, double x,
                                       const std::string& field, int i, int j) {
        auto cr = make_competing_risks(ct, cp, minimum_of_rv, dependency, correlation);
        double cdf1 = cr.cdf(x);
        double corr_ij = cr.correlation_matrix()[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        cr.set_dependency(parse_dependency(dependency2));
        double cdf2 = cr.cdf(x);
        if (field == "cdf1") return cdf1;
        if (field == "correlation") return corr_ij;
        if (field == "cdf2") return cdf2;
        throw py::value_error("unknown cr_dependency_change field: " + field);
    });
}
