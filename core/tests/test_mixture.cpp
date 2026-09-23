// C++-only ctest for the v2.1.4 Numerics `Mixture` zero-inflation setter semantics
// (IsZeroInflated/ZeroWeight -> NormalizeComponentWeights/RefreshConfigurationState) and
// Clone()'s IsZeroInflated-then-ZeroWeight re-application.
//
// These mirror `Test_Mixture_ZeroInflation_RescalesComponentWeights` and
// `Test_Mixture_Clone_PreservesZeroInflation`
// (Numerics/Test_Numerics/Distributions/Univariate/Test_Mixture.cs @ 2a0357a) directly.
// Neither fits the declarative fixture shape: the first is an inherently STATEFUL sequence
// (set ZeroWeight, then IsZeroInflated=true, assert, then IsZeroInflated=false, assert again)
// on ONE persistent object, and the fixture composite `construct` schema builds a fresh
// object per case; the second compares two independently-constructed objects (original vs.
// `Clone()`) including `AreNotSame`-style identity, which the fixture assertion shape (one
// scalar value per assertion) cannot express at all. Both are pinned instead as C++-only
// oracles transcribing the deterministic C# arithmetic directly (see fixtures/README.md's
// "C++-only ctest" precedent, e.g. test_mixture_em_seeding.cpp).
//
// The weight-RESCALE outcome itself (not the stateful path to it) IS additionally covered as
// a cross-language oracle fixture case in fixtures/distributions/univariate/mixture.json,
// via the composite construct's "zero_inflated"/"zero_weight" fields -- see that file's
// `zero_inflation_rescales_weights` case.
#include <memory>
#include <vector>

#include "corehydro/numerics/distributions/mixture.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "check.hpp"

using corehydro::numerics::distributions::Mixture;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::ParameterEstimationMethod;
using corehydro::numerics::distributions::UnivariateDistributionBase;
using corehydro::numerics::distributions::Uniform;

namespace {

std::vector<std::unique_ptr<UnivariateDistributionBase>> two_normals() {
    std::vector<std::unique_ptr<UnivariateDistributionBase>> comps;
    comps.push_back(std::make_unique<Normal>(0.0, 1.0));
    comps.push_back(std::make_unique<Normal>(5.0, 2.0));
    return comps;
}

// Mirrors Test_Mixture_ZeroInflation_RescalesComponentWeights: ZeroWeight set FIRST (while
// IsZeroInflated is still false, so no rescale fires yet), then IsZeroInflated=true triggers
// the rescale against the now-current ZeroWeight; then IsZeroInflated=false leaves the
// (still zero-inflation-scaled) weights in place, which makes the mixture invalid since they
// no longer sum to 1.
void test_zero_inflation_rescales_component_weights() {
    Mixture mix(std::vector<double>{0.25, 0.75}, two_normals());

    mix.set_zero_weight(0.2);
    mix.set_is_zero_inflated(true);

    CHECK_TRUE(mix.parameters_valid());
    CHECK_NEAR(mix.weights()[0], 0.2, 1e-12);
    CHECK_NEAR(mix.weights()[1], 0.6, 1e-12);
    CHECK_NEAR(mix.zero_weight() + mix.weights()[0] + mix.weights()[1], 1.0, 1e-12);

    mix.set_is_zero_inflated(false);

    CHECK_TRUE(!mix.parameters_valid());
}

// Mirrors Test_Mixture_Clone_PreservesZeroInflation: a single-component zero-inflated mixture
// clones with IsZeroInflated/ZeroWeight/weights/PDF/CDF preserved, and the cloned component
// distribution is a genuinely separate object (not aliased).
void test_clone_preserves_zero_inflation() {
    std::vector<std::unique_ptr<UnivariateDistributionBase>> comps;
    comps.push_back(std::make_unique<Normal>(100.0, 10.0));
    Mixture mix(std::vector<double>{1.0}, std::move(comps));
    mix.set_is_zero_inflated(true);
    mix.set_zero_weight(0.5);

    auto cloned = mix.clone();
    auto* clone = dynamic_cast<Mixture*>(cloned.get());
    CHECK_TRUE(clone != nullptr);

    CHECK_TRUE(mix.parameters_valid());
    CHECK_TRUE(clone->parameters_valid());
    CHECK_TRUE(clone->is_zero_inflated());
    CHECK_NEAR(mix.weights()[0], 0.5, 1e-12);
    CHECK_NEAR(mix.zero_weight(), clone->zero_weight(), 0.0);
    CHECK_NEAR(mix.weights()[0], clone->weights()[0], 0.0);
    CHECK_NEAR(mix.pdf(100.0), clone->pdf(100.0), 1e-12);
    CHECK_NEAR(mix.cdf(100.0), clone->cdf(100.0), 1e-12);
    CHECK_TRUE(&mix.component(0) != &clone->component(0));
}

void test_v220_simplex_normalization_does_not_modify_caller() {
    Mixture mix(std::vector<double>{0.5, 0.5}, two_normals());
    mix.set_is_zero_inflated(true);
    mix.set_zero_weight(0.2);
    std::vector<double> parameters{2.0, 1.0, 0.0, 1.0, 5.0, 2.0};
    const auto original = parameters;

    mix.set_parameters_normalized(parameters);

    CHECK_TRUE(parameters == original);
    CHECK_NEAR(mix.weights()[0], 0.8 * 2.0 / 3.0, 1e-15);
    CHECK_NEAR(mix.weights()[1], 0.8 / 3.0, 1e-15);
}

void test_v220_support_ignores_zero_weight_components() {
    std::vector<std::unique_ptr<UnivariateDistributionBase>> components;
    components.push_back(std::make_unique<Uniform>(0.0, 1.0));
    components.push_back(std::make_unique<Uniform>(-100.0, 100.0));
    Mixture mix(std::vector<double>{1.0, 0.0}, std::move(components));

    CHECK_NEAR(mix.minimum(), 0.0, 0.0);
    CHECK_NEAR(mix.maximum(), 1.0, 0.0);
}

void test_v220_impossible_zero_inflated_em_row_throws() {
    Mixture mix(std::vector<double>{0.5, 0.5}, two_normals());
    mix.set_is_zero_inflated(true);
    mix.set_zero_weight(0.2);

    CHECK_THROWS(mix.estimate({-1.0, 1.0, 2.0, 3.0},
                              ParameterEstimationMethod::MaximumLikelihood));
}

}  // namespace

int main() {
    test_zero_inflation_rescales_component_weights();
    test_clone_preserves_zero_inflation();
    test_v220_simplex_normalization_does_not_modify_caller();
    test_v220_support_ignores_zero_weight_components();
    test_v220_impossible_zero_inflated_em_row_throws();
    return chtest::summary("test_mixture");
}
