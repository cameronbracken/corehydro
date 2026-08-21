// C++-only ctest guarding the IMaximumLikelihoodEstimation contract that
// `GetParameterConstraints` returns arrays of length `NumberOfParameters`.
//
// Why this exists: `GeneralizedPareto::get_parameter_constraints` used to return only the
// two-element (alpha, kappa) slice, because the port folded MLE's C# `.Subset(1)` call
// (GeneralizedPareto.cs line 570) into the constraint method itself. Every OTHER caller of
// the interface -- `UnivariateDistributionModel::set_default_parameters`, `set_trend_model`,
// the MCMC model registry, the copula IFM/MPL marginal pre-fit, the mixture / competing-risks
// / point-process component seeds, and Bulletin17C -- indexes those arrays by
// `distribution->number_of_parameters()`, so a GPD reached them with `initials[2]` past the
// end of a two-element vector. ASan reported it as a heap-buffer-overflow READ at
// `univariate_distribution_model_trends.hpp:92` from three ctest binaries
// (test_univariate_distribution_model, test_fitting_analysis, test_fixtures); nothing FAILED,
// which is why it survived. In C# the same index is in range, so this was a port defect, not
// upstream behaviour.
//
// The test is structural on purpose: the out-of-bounds read changed no assertable output, so
// the honest regression guard is the size contract every caller already assumes. Revert the
// `generalized_pareto.hpp` fix and the GeneralizedPareto rows below fail 3 checks.
//
// No fixture equivalent: `fixtures/README.md` has no schema for "assert this array's length,"
// and array lengths are a structural/interface property rather than a numeric oracle to
// reproduce against the real C#.
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "check.hpp"

using corehydro::numerics::distributions::create_distribution;
using corehydro::numerics::distributions::IMaximumLikelihoodEstimation;
using corehydro::numerics::distributions::UnivariateDistributionType;

namespace {

// A strictly positive, non-degenerate continuous sample every continuous family accepts
// (the discrete families that reject it are skipped via the try/catch below).
std::vector<double> sample() {
    return {1.2, 2.4, 3.1, 4.7, 5.3, 6.8, 7.4, 8.9, 9.1, 10.6,
            11.2, 12.8, 13.5, 14.1, 15.9, 16.3, 17.7, 18.2, 19.6, 20.4};
}

// Every type with a real class and a factory case (the four the factory rejects --
// CompetingRisks / Mixture / UserDefined / GeneralizedNormal -- are omitted; see
// test_univariate_distribution_factory.cpp).
constexpr std::array<UnivariateDistributionType, 39> kConstructibleTypes = {
    UnivariateDistributionType::ChiSquared,
    UnivariateDistributionType::Bernoulli,
    UnivariateDistributionType::Beta,
    UnivariateDistributionType::Binomial,
    UnivariateDistributionType::Cauchy,
    UnivariateDistributionType::Deterministic,
    UnivariateDistributionType::Empirical,
    UnivariateDistributionType::Exponential,
    UnivariateDistributionType::GammaDistribution,
    UnivariateDistributionType::GeneralizedBeta,
    UnivariateDistributionType::GeneralizedExtremeValue,
    UnivariateDistributionType::GeneralizedLogistic,
    UnivariateDistributionType::GeneralizedPareto,
    UnivariateDistributionType::Geometric,
    UnivariateDistributionType::Gumbel,
    UnivariateDistributionType::InverseChiSquared,
    UnivariateDistributionType::InverseGamma,
    UnivariateDistributionType::KappaFour,
    UnivariateDistributionType::KernelDensity,
    UnivariateDistributionType::LnNormal,
    UnivariateDistributionType::Logistic,
    UnivariateDistributionType::LogNormal,
    UnivariateDistributionType::LogPearsonTypeIII,
    UnivariateDistributionType::NoncentralT,
    UnivariateDistributionType::Normal,
    UnivariateDistributionType::Pareto,
    UnivariateDistributionType::PearsonTypeIII,
    UnivariateDistributionType::Pert,
    UnivariateDistributionType::PertPercentile,
    UnivariateDistributionType::PertPercentileZ,
    UnivariateDistributionType::Poisson,
    UnivariateDistributionType::Rayleigh,
    UnivariateDistributionType::StudentT,
    UnivariateDistributionType::Triangular,
    UnivariateDistributionType::TruncatedNormal,
    UnivariateDistributionType::Uniform,
    UnivariateDistributionType::UniformDiscrete,
    UnivariateDistributionType::VonMises,
    UnivariateDistributionType::Weibull,
};

}  // namespace

int main() {
    const std::vector<double> data = sample();

    // Every IMaximumLikelihoodEstimation implementer must fill all three arrays to
    // number_of_parameters(), because that is the length every caller indexes to.
    for (UnivariateDistributionType type : kConstructibleTypes) {
        std::unique_ptr<corehydro::numerics::distributions::UnivariateDistributionBase> dist;
        try {
            dist = create_distribution(type);
        } catch (...) {
            continue;  // Not constructible here; test_univariate_distribution_factory covers it.
        }
        auto* mle = dynamic_cast<IMaximumLikelihoodEstimation*>(dist.get());
        if (mle == nullptr) continue;  // Family has no MLE surface; nothing to check.

        std::vector<double> initials, lowers, uppers;
        try {
            mle->get_parameter_constraints(data, initials, lowers, uppers);
        } catch (...) {
            // A family that rejects this particular sample (the discrete ones) still has no
            // size contract to violate. Skipping keeps the test about lengths, not support.
            continue;
        }

        const std::size_t n = static_cast<std::size_t>(dist->number_of_parameters());
        CHECK_EQ(initials.size(), n);
        CHECK_EQ(lowers.size(), n);
        CHECK_EQ(uppers.size(), n);
    }

    // The headline case, asserted directly so it cannot be skipped by the loop's guards:
    // GeneralizedPareto has three parameters and must report three constraints.
    {
        auto gpa = create_distribution(UnivariateDistributionType::GeneralizedPareto);
        auto* mle = dynamic_cast<IMaximumLikelihoodEstimation*>(gpa.get());
        CHECK_TRUE(mle != nullptr);
        std::vector<double> initials, lowers, uppers;
        mle->get_parameter_constraints(data, initials, lowers, uppers);
        CHECK_EQ(gpa->number_of_parameters(), 3);
        CHECK_EQ(initials.size(), std::size_t{3});
        CHECK_EQ(lowers.size(), std::size_t{3});
        CHECK_EQ(uppers.size(), std::size_t{3});

        // Each initial lies inside its own bounds, and the bounds are ordered -- what the
        // out-of-bounds read could not guarantee for the third (kappa) slot.
        for (std::size_t i = 0; i < 3; ++i) {
            CHECK_TRUE(lowers[i] < uppers[i]);
            CHECK_TRUE(initials[i] >= lowers[i] && initials[i] <= uppers[i]);
        }
        // C# 533-534 fixes the shape bounds at [-10, 10]; they are now the third slot, not
        // the second, which is exactly what the truncation got wrong.
        CHECK_EQ(lowers[2], -10.0);
        CHECK_EQ(uppers[2], 10.0);
    }

    return chtest::summary("parameter_constraint_sizes");
}
