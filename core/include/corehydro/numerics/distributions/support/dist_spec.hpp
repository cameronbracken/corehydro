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
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/competing_risks.hpp"
#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "corehydro/numerics/distributions/kernel_density.hpp"
#include "corehydro/numerics/distributions/mixture.hpp"
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
