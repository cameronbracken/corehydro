// corehydro ADDITION -- no upstream C# counterpart (sibling of models/model_spec.hpp and
// estimation/support/fit_runner.hpp).
//
// The one place a distribution object is built from a spec. The grammar is the fixture
// `construct` schema promoted to a first-class contract, so a fixture case, an oracle replay,
// and a user's dist_mixture() call build the identical object:
//
//   {"family": "Normal", "parameters": [0, 1]}
//   {"family": "TruncatedDistribution", "base": <spec>, "bounds": [lo, hi]}
//   {"family": "Mixture", "components": [<spec>...], "weights": [...],
//    "zero_inflated": false, "zero_weight": 0.0}
//   {"family": "CompetingRisks", "components": [<spec>...],
//    "minimum_of_random_variables": true, "dependency": "Independent", "correlation": [[...]]}
//   {"family": "Empirical", "x": [...], "p": [...], "p_transform": "NormalZ",
//    "p_descending": false}
//   {"family": "KernelDensity", "data": [...], "kernel": "Gaussian", "bandwidth": h,
//    "bounded_by_data": true}
//
// A MultivariateNormal spec accepts four optional integrator settings beside its mean and
// covariance -- "seed", "max_evaluations", "abs_error" and "rel_error". They configure the Genz
// quasi-Monte-Carlo integrator behind the CDF for dimension >= 3; without "seed" the instance
// clock-seeds and repeated calls disagree. An absent key leaves the ported default untouched.
//
// `target`/`params` are accepted as aliases of `family`/`parameters` because every fixture file
// spells them that way; renaming keys across the pinned corpus would buy nothing.
//
// An optional "set_parameters" array is applied after construction, serving the fixture arms
// that assert a value after a SetParameters round trip.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/competing_risks.hpp"
#include "corehydro/numerics/distributions/copulas/base/bivariate_copula_estimation.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_factory.hpp"
#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "corehydro/numerics/distributions/kernel_density.hpp"
#include "corehydro/numerics/distributions/mixture.hpp"
#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution.hpp"
#include "corehydro/numerics/distributions/multivariate/bivariate_empirical.hpp"
#include "corehydro/numerics/distributions/multivariate/dirichlet.hpp"
#include "corehydro/numerics/distributions/multivariate/multinomial.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_student_t.hpp"
#include "corehydro/numerics/distributions/truncated_distribution.hpp"

namespace corehydro::numerics::distributions::support {

using models::spec::JsonValue;

// Same alias competing_risks.hpp uses (competing_risks.hpp:92): CompetingRisks::correlation_matrix
// / set_correlation_matrix are spelled in terms of it.
namespace prob = corehydro::numerics::data::probability;

// "family" with "target" accepted as an alias.
inline std::string spec_family(const JsonValue& s) {
    if (s.contains("family")) return s.at("family").as_string();
    if (s.contains("target")) return s.at("target").as_string();
    throw std::runtime_error("distribution spec: missing required key 'family'");
}

// "parameters" with "params" accepted as an alias; empty when neither is present.
inline std::vector<double> spec_parameters(const JsonValue& s) {
    if (s.contains("parameters")) return s.at("parameters").as_double_vector();
    if (s.contains("params")) return s.at("params").as_double_vector();
    return {};
}

inline bool is_composite_family(const std::string& family) {
    return family == "TruncatedDistribution" || family == "Empirical" ||
           family == "KernelDensity" || family == "Mixture" || family == "CompetingRisks";
}

std::unique_ptr<UnivariateDistributionBase> build_univariate(const JsonValue& spec);

namespace detail {

// Lifted verbatim from core/tests/test_fixtures.cpp:967 (`static prob::DependencyType
// parse_dependency(const std::string&)`), placed in this namespace so it is the single
// spelling both the fixture runner and this shared builder use.
inline prob::DependencyType parse_dependency(const std::string& d) {
    if (d == "Independent") return prob::DependencyType::Independent;
    if (d == "PerfectlyPositive") return prob::DependencyType::PerfectlyPositive;
    if (d == "PerfectlyNegative") return prob::DependencyType::PerfectlyNegative;
    if (d == "CorrelationMatrix") return prob::DependencyType::CorrelationMatrix;
    throw std::runtime_error("unknown dependency type: " + d);
}

inline EmpiricalTransform parse_empirical_transform(const std::string& t) {
    if (t == "None") return EmpiricalTransform::None;
    if (t == "NormalZ") return EmpiricalTransform::NormalZ;
    throw std::runtime_error("unknown p_transform '" + t + "'; expected None or NormalZ");
}

inline KernelType parse_kernel_type(const std::string& k) {
    if (k == "Epanechnikov") return KernelType::Epanechnikov;
    if (k == "Gaussian") return KernelType::Gaussian;
    if (k == "Triangular") return KernelType::Triangular;
    if (k == "Uniform") return KernelType::Uniform;
    throw std::runtime_error("unknown kernel '" + k +
                             "'; expected Gaussian, Epanechnikov, Triangular or Uniform");
}

inline std::unique_ptr<UnivariateDistributionBase> build_flat(const std::string& family,
                                                              const JsonValue& spec) {
    std::unique_ptr<UnivariateDistributionBase> d;
    try {
        d = create_distribution(family);
    } catch (const std::exception&) {
        throw std::runtime_error("unknown distribution family '" + family + "'");
    }
    std::vector<double> p = spec_parameters(spec);
    if (!p.empty()) d->set_parameters(p);
    return d;
}

}  // namespace detail

// Builds any univariate distribution, flat or composite, nested to any depth.
inline std::unique_ptr<UnivariateDistributionBase> build_univariate(const JsonValue& spec) {
    std::string family = spec_family(spec);
    std::unique_ptr<UnivariateDistributionBase> out;

    if (family == "TruncatedDistribution") {
        std::unique_ptr<UnivariateDistributionBase> base = build_univariate(spec.at("base"));
        const std::vector<double> b = spec.at("bounds").as_double_vector();
        if (b.size() != 2)
            throw std::runtime_error("TruncatedDistribution: 'bounds' needs exactly two values");
        out = std::make_unique<TruncatedDistribution>(std::move(base), b[0], b[1]);
    } else if (family == "Mixture") {
        std::vector<double> weights = spec.at("weights").as_double_vector();
        std::vector<std::unique_ptr<UnivariateDistributionBase>> comps;
        for (const JsonValue& c : spec.at("components").items())
            comps.push_back(build_univariate(c));
        auto mix = std::make_unique<Mixture>(std::move(weights), std::move(comps));
        // IsZeroInflated before ZeroWeight, matching the C# Clone() initializer order: both
        // setters renormalize the component weights as a side effect.
        mix->set_is_zero_inflated(spec.value_or("zero_inflated", false));
        mix->set_zero_weight(spec.value_or("zero_weight", 0.0));
        out = std::move(mix);
    } else if (family == "CompetingRisks") {
        std::vector<std::unique_ptr<UnivariateDistributionBase>> comps;
        for (const JsonValue& c : spec.at("components").items())
            comps.push_back(build_univariate(c));
        auto cr = std::make_unique<CompetingRisks>(std::move(comps));
        cr->minimum_of_random_variables = spec.value_or("minimum_of_random_variables", true);
        if (spec.contains("dependency"))
            cr->set_dependency(detail::parse_dependency(spec.at("dependency").as_string()));
        if (spec.contains("correlation")) {
            prob::Matrix2D corr;
            for (const JsonValue& row : spec.at("correlation").items())
                corr.push_back(row.as_double_vector());
            cr->set_correlation_matrix(std::move(corr));
        }
        out = std::move(cr);
    } else if (family == "Empirical") {
        std::vector<double> x = spec.at("x").as_double_vector();
        std::vector<double> p = spec.at("p").as_double_vector();
        EmpiricalTransform pt = EmpiricalTransform::NormalZ;
        if (spec.contains("p_transform"))
            pt = detail::parse_empirical_transform(spec.at("p_transform").as_string());
        out = std::make_unique<EmpiricalDistribution>(std::move(x), std::move(p), pt,
                                                      spec.value_or("p_descending", false));
    } else if (family == "KernelDensity") {
        std::vector<double> data = spec.at("data").as_double_vector();
        KernelType kt = KernelType::Gaussian;
        if (spec.contains("kernel")) kt = detail::parse_kernel_type(spec.at("kernel").as_string());
        std::unique_ptr<KernelDensity> kde;
        if (spec.contains("bandwidth"))
            kde = std::make_unique<KernelDensity>(std::move(data), kt,
                                                  spec.at("bandwidth").as_double());
        else
            kde = std::make_unique<KernelDensity>(std::move(data), kt);
        if (spec.contains("bounded_by_data"))
            kde->set_bounded_by_data(spec.at("bounded_by_data").as_bool());
        out = std::move(kde);
    } else {
        out = detail::build_flat(family, spec);
    }

    if (spec.contains("set_parameters")) out->set_parameters(spec.at("set_parameters").as_double_vector());
    return out;
}

}  // namespace corehydro::numerics::distributions::support

// corehydro ADDITION, placed in the vendored `copulas` namespace (not `support`) so
// build_copula below can call it unqualified as `copulas::set_theta_from_tau`. SetThetaFromTau
// is a member of each concrete Archimedean class in the C# source, not part of
// IBivariateCopula/IArchimedeanCopula, so it cannot be called through the BivariateCopula base
// pointer. Lifted from the fixture runner's `set_theta_from_tau_dispatch`
// (test_fixtures.cpp:1588-1605) -- the exact branch set: Clayton, AliMikhailHaq, Gumbel (Joe
// has no SetThetaFromTau in the C# source). Returns false rather than throwing for any other
// family, so the caller composes its own error message naming the family.
namespace corehydro::numerics::distributions::copulas {

inline bool set_theta_from_tau(BivariateCopula& copula, const std::vector<double>& x,
                               const std::vector<double>& y) {
    if (auto* c = dynamic_cast<ClaytonCopula*>(&copula)) {
        c->set_theta_from_tau(x, y);
        return true;
    }
    if (auto* c = dynamic_cast<AMHCopula*>(&copula)) {
        c->set_theta_from_tau(x, y);
        return true;
    }
    if (auto* c = dynamic_cast<GumbelCopula*>(&copula)) {
        c->set_theta_from_tau(x, y);
        return true;
    }
    return false;
}

}  // namespace corehydro::numerics::distributions::copulas

namespace corehydro::numerics::distributions::support {

// Lifted from the fixture runner's `build_multivariate`'s local `parse_transform` lambda
// (test_fixtures.cpp:1366-1375). Same accepted strings as BivariateEmpirical's transform
// grammar.
inline corehydro::numerics::data::Transform parse_bivariate_transform(const std::string& t) {
    using corehydro::numerics::data::Transform;
    if (t == "None") return Transform::None;
    if (t == "Logarithmic") return Transform::Logarithmic;
    if (t == "NormalZ") return Transform::NormalZ;
    throw std::runtime_error("unknown transform '" + t + "'; expected None, Logarithmic or NormalZ");
}

// MLE-fits a marginal to its own sample when, and only when, its spec named a family without
// parameters. IFM (BivariateCopulaEstimation's `ifm`) optimizes theta ALONE and takes the
// marginals as already fitted, so a bare family left default-constructed would have theta
// optimized against a standard Normal(0, 1) instead of against the data -- a wrong answer with
// no error. Full MLE re-estimates the marginals itself, so the pre-fit only supplies the
// starting object there; MPL ignores the marginals and `tau` does not use them.
inline void prefit_marginal(UnivariateDistributionBase* marginal, const JsonValue& margin_spec,
                            const std::vector<double>& sample, const char* which) {
    if (marginal == nullptr) return;
    if (!spec_parameters(margin_spec).empty()) return;  // used exactly as given
    auto* est = dynamic_cast<IEstimation*>(marginal);
    if (est == nullptr)
        throw std::runtime_error(std::string("copula fit: ") + which + " names '" +
                                 spec_family(margin_spec) +
                                 "' without parameters, but that family does not implement "
                                 "IEstimation, so it cannot be fitted to the sample");
    est->estimate(sample, ParameterEstimationMethod::MaximumLikelihood);
}

// A copula spec is either parameterized ({"theta": x, "df"?: y}) or fitted
// ({"fit": {"x", "y", "method", "margin_x"?, "margin_y"?}}). Marginals attach in both forms.
// In the fit form a marginal that names a family with no parameters is MLE-fitted to its own
// sample first (x for margin_x, y for margin_y) -- see prefit_marginal above.
inline std::unique_ptr<copulas::BivariateCopula> build_copula(const JsonValue& spec) {
    std::string family = spec_family(spec);
    std::unique_ptr<copulas::BivariateCopula> c;
    try {
        c = copulas::create_copula(family);
    } catch (const std::exception&) {
        throw std::runtime_error("unknown copula family '" + family +
                                 "'; expected AliMikhailHaq, Clayton, Frank, Gumbel, Joe, "
                                 "Normal or StudentT");
    }

    auto attach = [&](const JsonValue& holder) {
        if (holder.contains("margin_x"))
            c->marginal_distribution_x =
                std::shared_ptr<UnivariateDistributionBase>(build_univariate(holder.at("margin_x")));
        if (holder.contains("margin_y"))
            c->marginal_distribution_y =
                std::shared_ptr<UnivariateDistributionBase>(build_univariate(holder.at("margin_y")));
    };

    if (spec.contains("fit")) {
        const JsonValue& fit = spec.at("fit");
        attach(fit);
        std::vector<double> x = fit.at("x").as_double_vector();
        std::vector<double> y = fit.at("y").as_double_vector();
        std::string method = fit.value_or("method", "mpl");
        if (method == "ifm" || method == "mle") {
            if (fit.contains("margin_x"))
                prefit_marginal(c->marginal_distribution_x.get(), fit.at("margin_x"), x, "margin_x");
            if (fit.contains("margin_y"))
                prefit_marginal(c->marginal_distribution_y.get(), fit.at("margin_y"), y, "margin_y");
        }
        if (method == "tau") {
            // Only three concrete copulas implement SetThetaFromTau upstream.
            if (!copulas::set_theta_from_tau(*c, x, y))
                throw std::runtime_error("method 'tau' is not available for '" + family +
                                         "'; upstream implements SetThetaFromTau for Clayton, "
                                         "Gumbel and AliMikhailHaq only");
        } else if (method == "mpl") {
            copulas::estimate(*c, x, y, copulas::CopulaEstimationMethod::PseudoLikelihood);
        } else if (method == "ifm") {
            copulas::estimate(*c, x, y, copulas::CopulaEstimationMethod::InferenceFromMargins);
        } else if (method == "mle") {
            copulas::estimate(*c, x, y, copulas::CopulaEstimationMethod::FullLikelihood);
        } else {
            throw std::runtime_error("unknown copula fit method '" + method +
                                     "'; expected mpl, ifm, mle or tau");
        }
        return c;
    }

    std::vector<double> params = {spec.at("theta").as_double()};
    if (spec.contains("df")) params.push_back(spec.at("df").as_double());
    c->set_copula_parameters(params);
    attach(spec);
    return c;
}

inline std::unique_ptr<MultivariateDistribution> build_multivariate(const JsonValue& spec) {
    std::string family = spec_family(spec);
    auto rows = [](const JsonValue& m) {
        std::vector<std::vector<double>> out;
        for (const JsonValue& row : m.items()) out.push_back(row.as_double_vector());
        return out;
    };
    if (family == "MultivariateNormal") {
        std::vector<double> mean = spec.at("mean").as_double_vector();
        std::unique_ptr<MultivariateNormal> m =
            spec.contains("covariance")
                ? std::make_unique<MultivariateNormal>(std::move(mean), rows(spec.at("covariance")))
                : std::make_unique<MultivariateNormal>(std::move(mean));
        // The four Genz-integrator settings, applied AFTER construction because
        // set_parameters resets max_evaluations to its 1000 x dimension default. An absent key
        // leaves the ported default untouched. `seed` is what makes a dimension >= 3 CDF
        // reproducible at all: MVNDST draws from the instance's own Mersenne Twister, which
        // clock-seeds otherwise.
        if (spec.contains("seed")) m->set_mvnuni_seed(spec.at("seed").as_int());
        if (spec.contains("max_evaluations"))
            m->set_max_evaluations(spec.at("max_evaluations").as_int());
        if (spec.contains("abs_error")) m->set_absolute_error(spec.at("abs_error").as_double());
        if (spec.contains("rel_error")) m->set_relative_error(spec.at("rel_error").as_double());
        return m;
    }
    if (family == "MultivariateStudentT") {
        double df = spec.at("df").as_double();
        std::vector<double> loc = spec.at("location").as_double_vector();
        if (!spec.contains("scale"))
            return std::make_unique<MultivariateStudentT>(df, std::move(loc));
        return std::make_unique<MultivariateStudentT>(df, std::move(loc), rows(spec.at("scale")));
    }
    if (family == "Dirichlet") return std::make_unique<Dirichlet>(spec.at("alpha").as_double_vector());
    if (family == "Multinomial")
        return std::make_unique<Multinomial>(spec.at("trials").as_int(),
                                             spec.at("probabilities").as_double_vector());
    if (family == "BivariateEmpirical") {
        auto tf = [&](const char* key) {
            return parse_bivariate_transform(spec.value_or(key, "None"));
        };
        return std::make_unique<BivariateEmpirical>(
            spec.at("x1").as_double_vector(), spec.at("x2").as_double_vector(),
            rows(spec.at("p")), tf("x1_transform"), tf("x2_transform"), tf("p_transform"));
    }
    throw std::runtime_error("unknown multivariate family '" + family +
                             "'; expected MultivariateNormal, MultivariateStudentT, Dirichlet, "
                             "Multinomial or BivariateEmpirical");
}

}  // namespace corehydro::numerics::distributions::support
