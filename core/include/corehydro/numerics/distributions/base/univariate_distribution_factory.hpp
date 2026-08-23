// ported from: Numerics/Distributions/Univariate/Base/UnivariateDistributionFactory.cs @ 2a0357a
//
// Constructs a default-parameterized distribution from its type. The C# if/else chain
// becomes a switch; the XElement overload is dropped (serialization is a desktop concern).
// Only ported distributions have a case -- requesting an unported type throws rather than
// silently returning a Deterministic placeholder, so gaps surface immediately.
//
// v2.1.4 (2a0357a): the C# factory itself was rewritten from an if/else chain (which fell
// through to `return new Deterministic()` for ANY unhandled type -- including Deterministic,
// Empirical, KernelDensity, VonMises, CompetingRisks, Mixture, UserDefined, and any genuinely
// undefined enum value) to an explicit switch with a case for every defined type, throwing
// `NotSupportedException` for the three composite/user types (CompetingRisks/Mixture/
// UserDefined -- "requires external components") and `ArgumentOutOfRangeException` for
// anything undefined. This port's switch already had explicit cases for every ported type
// (never had the C# bug), EXCEPT KernelDensity -- added below, matching the new C# case list.
// The only types still reaching `default:` are the three C# calls NotSupportedException on
// (CompetingRisks/Mixture/UserDefined) plus any undefined enum value; this port's single
// `default:` throw doesn't distinguish "legitimately unsupported" from "not yet ported,"
// unlike C#'s two distinct exception types, since the header comment above already treats
// every unhandled case identically (gaps surface immediately, by design).
//
// `try_create_distribution` adapts C#'s new `bool TryCreateDistribution(type, out dist)` to
// this factory's existing return-a-unique_ptr idiom: nullptr on any failure (unsupported,
// unported, or undefined), matching the throwing overload's classification exactly (it just
// catches whatever `create_distribution` throws).
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/bernoulli.hpp"
#include "corehydro/numerics/distributions/beta_distribution.hpp"
#include "corehydro/numerics/distributions/binomial.hpp"
#include "corehydro/numerics/distributions/cauchy.hpp"
#include "corehydro/numerics/distributions/deterministic.hpp"
#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "corehydro/numerics/distributions/exponential.hpp"
#include "corehydro/numerics/distributions/triangular.hpp"
#include "corehydro/numerics/distributions/generalized_beta.hpp"
#include "corehydro/numerics/distributions/generalized_extreme_value.hpp"
#include "corehydro/numerics/distributions/generalized_logistic.hpp"
#include "corehydro/numerics/distributions/generalized_normal.hpp"
#include "corehydro/numerics/distributions/generalized_pareto.hpp"
#include "corehydro/numerics/distributions/geometric.hpp"
#include "corehydro/numerics/distributions/chi_squared.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/gumbel.hpp"
#include "corehydro/numerics/distributions/inverse_chi_squared.hpp"
#include "corehydro/numerics/distributions/inverse_gamma.hpp"
#include "corehydro/numerics/distributions/kernel_density.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/log_normal.hpp"
#include "corehydro/numerics/distributions/logistic.hpp"
#include "corehydro/numerics/distributions/noncentral_t.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/pareto.hpp"
#include "corehydro/numerics/distributions/kappa_four.hpp"
#include "corehydro/numerics/distributions/log_pearson_type_iii.hpp"
#include "corehydro/numerics/distributions/pearson_type_iii.hpp"
#include "corehydro/numerics/distributions/pert.hpp"
#include "corehydro/numerics/distributions/pert_percentile.hpp"
#include "corehydro/numerics/distributions/pert_percentile_z.hpp"
#include "corehydro/numerics/distributions/poisson.hpp"
#include "corehydro/numerics/distributions/rayleigh.hpp"
#include "corehydro/numerics/distributions/student_t.hpp"
#include "corehydro/numerics/distributions/truncated_normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/distributions/uniform_discrete.hpp"
#include "corehydro/numerics/distributions/von_mises.hpp"
#include "corehydro/numerics/distributions/weibull.hpp"

namespace corehydro::numerics::distributions {

inline std::unique_ptr<UnivariateDistributionBase> create_distribution(
    UnivariateDistributionType type) {
    switch (type) {
        case UnivariateDistributionType::Bernoulli:
            return std::make_unique<Bernoulli>();
        case UnivariateDistributionType::Beta:
            return std::make_unique<BetaDistribution>();
        case UnivariateDistributionType::Binomial:
            return std::make_unique<Binomial>();
        case UnivariateDistributionType::Cauchy:
            return std::make_unique<Cauchy>();
        case UnivariateDistributionType::Deterministic:
            return std::make_unique<Deterministic>();
        case UnivariateDistributionType::Empirical:
            return std::make_unique<EmpiricalDistribution>();
        case UnivariateDistributionType::Exponential:
            return std::make_unique<Exponential>();
        case UnivariateDistributionType::GeneralizedBeta:
            return std::make_unique<GeneralizedBeta>();
        case UnivariateDistributionType::GeneralizedExtremeValue:
            return std::make_unique<GeneralizedExtremeValue>();
        case UnivariateDistributionType::GeneralizedLogistic:
            return std::make_unique<GeneralizedLogistic>();
        case UnivariateDistributionType::GeneralizedNormal:
            return std::make_unique<GeneralizedNormal>();
        case UnivariateDistributionType::GeneralizedPareto:
            return std::make_unique<GeneralizedPareto>();
        case UnivariateDistributionType::Geometric:
            return std::make_unique<Geometric>();
        case UnivariateDistributionType::ChiSquared:
            return std::make_unique<ChiSquared>();
        case UnivariateDistributionType::GammaDistribution:
            return std::make_unique<GammaDistribution>();
        case UnivariateDistributionType::Gumbel:
            return std::make_unique<Gumbel>();
        case UnivariateDistributionType::InverseChiSquared:
            return std::make_unique<InverseChiSquared>();
        case UnivariateDistributionType::InverseGamma:
            return std::make_unique<InverseGamma>();
        case UnivariateDistributionType::KappaFour:
            return std::make_unique<KappaFour>();
        case UnivariateDistributionType::KernelDensity:
            return std::make_unique<KernelDensity>();
        case UnivariateDistributionType::LogPearsonTypeIII:
            return std::make_unique<LogPearsonTypeIII>();
        case UnivariateDistributionType::LnNormal:
            return std::make_unique<LnNormal>();
        case UnivariateDistributionType::LogNormal:
            return std::make_unique<LogNormal>();
        case UnivariateDistributionType::Logistic:
            return std::make_unique<Logistic>();
        case UnivariateDistributionType::NoncentralT:
            return std::make_unique<NoncentralT>();
        case UnivariateDistributionType::Normal:
            return std::make_unique<Normal>();
        case UnivariateDistributionType::Pareto:
            return std::make_unique<Pareto>();
        case UnivariateDistributionType::PearsonTypeIII:
            return std::make_unique<PearsonTypeIII>();
        case UnivariateDistributionType::Pert:
            return std::make_unique<Pert>();
        case UnivariateDistributionType::PertPercentile:
            return std::make_unique<PertPercentile>();
        case UnivariateDistributionType::PertPercentileZ:
            return std::make_unique<PertPercentileZ>();
        case UnivariateDistributionType::Poisson:
            return std::make_unique<Poisson>();
        case UnivariateDistributionType::Rayleigh:
            return std::make_unique<Rayleigh>();
        case UnivariateDistributionType::StudentT:
            return std::make_unique<StudentT>();
        case UnivariateDistributionType::Triangular:
            return std::make_unique<Triangular>();
        case UnivariateDistributionType::TruncatedNormal:
            return std::make_unique<TruncatedNormal>();
        case UnivariateDistributionType::Uniform:
            return std::make_unique<Uniform>();
        case UnivariateDistributionType::UniformDiscrete:
            return std::make_unique<UniformDiscrete>();
        case UnivariateDistributionType::VonMises:
            return std::make_unique<VonMises>();
        case UnivariateDistributionType::Weibull:
            return std::make_unique<Weibull>();
        default:
            throw std::invalid_argument("univariate distribution type not yet ported");
    }
}

// Adapts C# `bool TryCreateDistribution(UnivariateDistributionType, out UnivariateDistributionBase?)`
// to this factory's existing unique_ptr-returning idiom: nullptr on any failure (unsupported
// composite type, unported type, or an undefined enum value) instead of an out-parameter +
// bool. Never throws.
inline std::unique_ptr<UnivariateDistributionBase> try_create_distribution(
    UnivariateDistributionType type) noexcept {
    try {
        return create_distribution(type);
    } catch (...) {
        return nullptr;
    }
}

// Construct from the C# type name (the value stored in fixtures' "target" field).
inline std::unique_ptr<UnivariateDistributionBase> create_distribution(const std::string& name) {
    if (name == "Bernoulli") return create_distribution(UnivariateDistributionType::Bernoulli);
    if (name == "Beta") return create_distribution(UnivariateDistributionType::Beta);
    if (name == "Binomial") return create_distribution(UnivariateDistributionType::Binomial);
    if (name == "Cauchy") return create_distribution(UnivariateDistributionType::Cauchy);
    if (name == "Deterministic") return create_distribution(UnivariateDistributionType::Deterministic);
    if (name == "Empirical") return create_distribution(UnivariateDistributionType::Empirical);
    if (name == "Exponential") return create_distribution(UnivariateDistributionType::Exponential);
    if (name == "GeneralizedBeta")
        return create_distribution(UnivariateDistributionType::GeneralizedBeta);
    if (name == "GeneralizedExtremeValue")
        return create_distribution(UnivariateDistributionType::GeneralizedExtremeValue);
    if (name == "GeneralizedLogistic")
        return create_distribution(UnivariateDistributionType::GeneralizedLogistic);
    if (name == "GeneralizedNormal")
        return create_distribution(UnivariateDistributionType::GeneralizedNormal);
    if (name == "GeneralizedPareto")
        return create_distribution(UnivariateDistributionType::GeneralizedPareto);
    if (name == "Geometric") return create_distribution(UnivariateDistributionType::Geometric);
    if (name == "ChiSquared") return create_distribution(UnivariateDistributionType::ChiSquared);
    if (name == "GammaDistribution") return create_distribution(UnivariateDistributionType::GammaDistribution);
    if (name == "Gumbel") return create_distribution(UnivariateDistributionType::Gumbel);
    if (name == "InverseChiSquared") return create_distribution(UnivariateDistributionType::InverseChiSquared);
    if (name == "InverseGamma") return create_distribution(UnivariateDistributionType::InverseGamma);
    if (name == "KappaFour")
        return create_distribution(UnivariateDistributionType::KappaFour);
    if (name == "KernelDensity")
        return create_distribution(UnivariateDistributionType::KernelDensity);
    if (name == "LogPearsonTypeIII")
        return create_distribution(UnivariateDistributionType::LogPearsonTypeIII);
    if (name == "LnNormal") return create_distribution(UnivariateDistributionType::LnNormal);
    if (name == "LogNormal") return create_distribution(UnivariateDistributionType::LogNormal);
    if (name == "Logistic") return create_distribution(UnivariateDistributionType::Logistic);
    if (name == "NoncentralT") return create_distribution(UnivariateDistributionType::NoncentralT);
    if (name == "Normal") return create_distribution(UnivariateDistributionType::Normal);
    if (name == "Pareto") return create_distribution(UnivariateDistributionType::Pareto);
    if (name == "PearsonTypeIII") return create_distribution(UnivariateDistributionType::PearsonTypeIII);
    if (name == "Pert") return create_distribution(UnivariateDistributionType::Pert);
    if (name == "PertPercentile")
        return create_distribution(UnivariateDistributionType::PertPercentile);
    if (name == "PertPercentileZ")
        return create_distribution(UnivariateDistributionType::PertPercentileZ);
    if (name == "Poisson") return create_distribution(UnivariateDistributionType::Poisson);
    if (name == "Rayleigh") return create_distribution(UnivariateDistributionType::Rayleigh);
    if (name == "StudentT") return create_distribution(UnivariateDistributionType::StudentT);
    if (name == "Triangular") return create_distribution(UnivariateDistributionType::Triangular);
    if (name == "TruncatedNormal") return create_distribution(UnivariateDistributionType::TruncatedNormal);
    if (name == "Uniform") return create_distribution(UnivariateDistributionType::Uniform);
    if (name == "UniformDiscrete")
        return create_distribution(UnivariateDistributionType::UniformDiscrete);
    if (name == "VonMises") return create_distribution(UnivariateDistributionType::VonMises);
    if (name == "Weibull") return create_distribution(UnivariateDistributionType::Weibull);
    throw std::invalid_argument("unknown distribution name: " + name);
}

// corehydro addition (not in the C# factory): the factory-constructible type names, in the
// order of the string overload above. Shared by the R and Python binding layers so the
// public `distribution_names()` surface has a single source of truth.
inline std::vector<std::string> distribution_names() {
    return {
        "Bernoulli",
        "Beta",
        "Binomial",
        "Cauchy",
        "Deterministic",
        "Empirical",
        "Exponential",
        "GeneralizedBeta",
        "GeneralizedExtremeValue",
        "GeneralizedLogistic",
        "GeneralizedNormal",
        "GeneralizedPareto",
        "Geometric",
        "ChiSquared",
        "GammaDistribution",
        "Gumbel",
        "InverseChiSquared",
        "InverseGamma",
        "KappaFour",
        "LogPearsonTypeIII",
        "LnNormal",
        "LogNormal",
        "Logistic",
        "NoncentralT",
        "Normal",
        "Pareto",
        "PearsonTypeIII",
        "Pert",
        "PertPercentile",
        "PertPercentileZ",
        "Poisson",
        "Rayleigh",
        "StudentT",
        "Triangular",
        "TruncatedNormal",
        "Uniform",
        "UniformDiscrete",
        "VonMises",
        "Weibull",
    };
}

}  // namespace corehydro::numerics::distributions
