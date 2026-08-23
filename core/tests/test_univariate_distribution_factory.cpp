// C++-only ctest for the v2.1.4 Numerics `UnivariateDistributionFactory` switch rewrite
// (Base/UnivariateDistributionFactory.cs @ 2a0357a): the C# if/else chain (which silently fell
// through to `return new Deterministic()` for any unhandled type) became an explicit switch
// with a case for every defined enum value, throwing `NotSupportedException` for the three
// composite/user types (CompetingRisks/Mixture/UserDefined) and `ArgumentOutOfRangeException`
// for anything undefined, plus a new `bool TryCreateDistribution(type, out dist)` entry point.
//
// Mirrors `Test_UnivariateDistributionFactory.cs` (new @ 2a0357a):
// `EveryDefinedDistributionTypeIsHandledExplicitly` and
// `UndefinedDistributionTypeThrowsArgumentOutOfRange`. This port's switch never had the C# bug
// (every ported type already had an explicit case; T7 additionally added the missing
// `KernelDensity` case -- see univariate_distribution_factory.hpp), so this test is adapted
// rather than transcribed line-for-line: it partitions the FULL C++ enum into "ported" (every
// type with a real class and factory case -- must succeed) and "unsupported"
// (CompetingRisks/Mixture/UserDefined, exactly C#'s NotSupportedException trio). Those three
// are now the ONLY defined types that fail: GeneralizedNormal used to sit in this set as the
// port's one missing class and moved to the ported half when it landed. This port's
// `create_distribution` / `try_create_distribution` do not distinguish an unsupported type
// from an undefined one (both fall through the same `default:` and throw
// std::invalid_argument / return nullptr), unlike C#'s two distinct exception types -- this
// test only checks "succeeds" vs. "fails," not the failure reason.
//
// No fixture equivalent: `fixtures/README.md` has no "assert this throws" schema (same gap
// noted in test_empirical_distribution.cpp), and factory completeness is a structural/
// enumeration check, not a numeric oracle -- there is nothing to reproduce against the real C#
// beyond "every ported type round-trips its own Type," which this test already asserts
// directly.
#include <array>
#include <stdexcept>

#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "check.hpp"

using corehydro::numerics::distributions::create_distribution;
using corehydro::numerics::distributions::try_create_distribution;
using corehydro::numerics::distributions::UnivariateDistributionType;

namespace {

// Every defined UnivariateDistributionType, in the enum's own declaration order (mirrors C#'s
// `Enum.GetValues(typeof(UnivariateDistributionType))`).
constexpr std::array<UnivariateDistributionType, 43> kAllTypes = {
    UnivariateDistributionType::ChiSquared,
    UnivariateDistributionType::Bernoulli,
    UnivariateDistributionType::Beta,
    UnivariateDistributionType::Binomial,
    UnivariateDistributionType::Cauchy,
    UnivariateDistributionType::CompetingRisks,
    UnivariateDistributionType::Deterministic,
    UnivariateDistributionType::Empirical,
    UnivariateDistributionType::Exponential,
    UnivariateDistributionType::GammaDistribution,
    UnivariateDistributionType::GeneralizedBeta,
    UnivariateDistributionType::GeneralizedExtremeValue,
    UnivariateDistributionType::GeneralizedLogistic,
    UnivariateDistributionType::GeneralizedNormal,
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
    UnivariateDistributionType::Mixture,
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
    UnivariateDistributionType::UserDefined,
    UnivariateDistributionType::VonMises,
    UnivariateDistributionType::Weibull,
};

// The three C# NotSupportedException composite/user types ("requires external components").
bool is_unsupported(UnivariateDistributionType t) {
    return t == UnivariateDistributionType::CompetingRisks ||
           t == UnivariateDistributionType::Mixture ||
           t == UnivariateDistributionType::UserDefined;
}

// Mirrors Test_UnivariateDistributionFactory.EveryDefinedDistributionTypeIsHandledExplicitly.
void test_every_defined_type_handled_explicitly() {
    for (UnivariateDistributionType type : kAllTypes) {
        if (is_unsupported(type)) {
            CHECK_THROWS(create_distribution(type));
            CHECK_TRUE(try_create_distribution(type) == nullptr);
        } else {
            auto dist = create_distribution(type);
            CHECK_TRUE(dist != nullptr);
            CHECK_TRUE(dist->type() == type);

            auto tried = try_create_distribution(type);
            CHECK_TRUE(tried != nullptr);
            CHECK_TRUE(tried->type() == type);
        }
    }
}

// Mirrors Test_UnivariateDistributionFactory.UndefinedDistributionTypeThrowsArgumentOutOfRange.
void test_undefined_type_throws() {
    auto undefined = static_cast<UnivariateDistributionType>(999999);
    CHECK_THROWS(create_distribution(undefined));
    CHECK_TRUE(try_create_distribution(undefined) == nullptr);
}

}  // namespace

int main() {
    test_every_defined_type_handled_explicitly();
    test_undefined_type_throws();
    return chtest::summary("test_univariate_distribution_factory");
}
