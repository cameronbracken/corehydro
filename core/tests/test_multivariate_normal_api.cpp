// C++-only ctest for the v2.1.4 MultivariateNormal Try/Marginal/Conditional API
// (Numerics/Distributions/Multivariate/MultivariateNormal.cs @ 7e8e8d1) -- the parts of the
// new Test_Numerics/Distributions/Multivariate/Test_MultivariateNormal.cs coverage that do not
// fit the declarative fixture shape: a STATEFUL non-throwing-mutation sequence
// (TrySetCovariance/TrySetParameters/IsDensityValid, where each assertion depends on the
// PRECEDING one having mutated the SAME object) and index-validation THROWS (Marginal/
// Conditional given empty/duplicate/out-of-range indices, or a values-length mismatch). Per
// the test_mixture.cpp/test_copula_clone.cpp precedent (see fixtures/README.md's "C++-only
// ctest" convention), the numeric closed-form values for Marginal/Conditional themselves ARE
// fixture-covered (fixtures/distributions/multivariate/multivariate_normal.json's
// marginal_subset_and_values/conditional_*_closed_form cases) -- only the parts below are not.
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "check.hpp"

namespace dist = corehydro::numerics::distributions;
namespace la = corehydro::numerics::math::linalg;

namespace {

// Mirrors Test_TrySetCovariance_NonThrowingInvalidState: a positive-definite swap evaluates
// normally, a non-positive-definite swap marks the density invalid (LogPDF = -inf, PDF = 0)
// WITHOUT throwing, and a subsequent valid swap restores evaluation exactly. TrySetParameters
// also moves the mean. Each step's outcome depends on the PRECEDING step's mutation of the
// SAME object -- not expressible as independent fixture assertions.
void test_try_set_covariance_non_throwing_invalid_state() {
    dist::MultivariateNormal mvn(std::vector<double>{1.0, 2.0},
                                  std::vector<std::vector<double>>{{1.0, 0.3}, {0.3, 2.0}});
    double valid_log_pdf = mvn.log_pdf({1.2, 1.8});
    CHECK_TRUE(mvn.is_density_valid());
    CHECK_TRUE(!std::isnan(valid_log_pdf) && !std::isinf(valid_log_pdf));

    // A non-positive-definite covariance: correlation beyond one.
    CHECK_TRUE(!mvn.try_set_covariance({{1.0, 1.5}, {1.5, 1.0}}));
    CHECK_TRUE(!mvn.is_density_valid());
    CHECK_TRUE(std::isinf(mvn.log_pdf({1.2, 1.8})) && mvn.log_pdf({1.2, 1.8}) < 0);
    CHECK_EQ(mvn.pdf({1.2, 1.8}), 0.0);

    // Restore with the original covariance: the density comes back exactly.
    CHECK_TRUE(mvn.try_set_covariance({{1.0, 0.3}, {0.3, 2.0}}));
    CHECK_TRUE(mvn.is_density_valid());
    CHECK_NEAR(mvn.log_pdf({1.2, 1.8}), valid_log_pdf, 1e-12);

    // TrySetParameters also moves the mean.
    CHECK_TRUE(mvn.try_set_parameters({0.0, 0.0}, {{1.0, 0.0}, {0.0, 1.0}}));
    CHECK_NEAR(mvn.log_pdf({0.0, 0.0}), -std::log(2.0 * corehydro::numerics::kPi), 1e-12);
}

// Mirrors Test_Marginal_SubsetAndValidation's validation half (the closed-form subset/value
// checks are fixture-covered): Marginal throws on an empty index list, a duplicate index, and
// an out-of-range index.
void test_marginal_validates_indices() {
    dist::MultivariateNormal mvn(
        std::vector<double>{1.0, 2.0, 3.0},
        std::vector<std::vector<double>>{{4.0, 1.2, 0.5}, {1.2, 9.0, 2.1}, {0.5, 2.1, 16.0}});

    CHECK_THROWS(mvn.marginal({}));
    CHECK_THROWS(mvn.marginal({0, 0}));
    CHECK_THROWS(mvn.marginal({3}));
}

// Mirrors Test_Conditional_ClosedFormsAndValidation's validation half (the closed-form Schur-
// complement checks are fixture-covered): Conditional throws when every dimension is observed
// (nothing left to condition on) and when the observed-values length does not match the
// observed-indices length.
void test_conditional_validates_indices_and_values() {
    dist::MultivariateNormal mvn(
        std::vector<double>{1.0, 2.0, 3.0},
        std::vector<std::vector<double>>{{4.0, 1.2, 0.5}, {1.2, 9.0, 2.1}, {0.5, 2.1, 16.0}});

    CHECK_THROWS(mvn.conditional({0, 1, 2}, {1.0, 2.0, 3.0}));
    CHECK_THROWS(mvn.conditional({1}, {1.0, 2.0}));
}

void test_singular_value_decomposition_path() {
    dist::MultivariateNormal mvn(
        std::vector<double>{0.0, 0.0},
        std::vector<std::vector<double>>{{1.0, 1.0}, {1.0, 1.0}},
        la::DecompositionMethod::SingularValue);
    CHECK_TRUE(mvn.decomposition() == la::DecompositionMethod::SingularValue);
    CHECK_TRUE(!mvn.is_positive_definite());
    CHECK_NEAR(mvn.pdf({0.0, 0.0}), 0.28209479177387814, 1e-14);
    CHECK_NEAR(mvn.mahalanobis({1.0, 1.0}), 1.0, 1e-13);
    CHECK_EQ(mvn.pdf({1.0, -1.0}), 0.0);
    CHECK_TRUE(std::isinf(mvn.log_pdf({1.0, -1.0})));

    auto draw = mvn.generate_random_values(1, 12345).at(0);
    CHECK_NEAR(draw[0], draw[1], 1e-13);

    CHECK_THROWS(dist::MultivariateNormal(
        std::vector<double>{0.0, 0.0},
        std::vector<std::vector<double>>{{1.0, 2.0}, {2.0, 1.0}},
        la::DecompositionMethod::SingularValue));
    CHECK_THROWS(dist::MultivariateNormal(
        std::vector<double>{0.0, 0.0},
        std::vector<std::vector<double>>{{1.0, 0.5}, {0.25, 1.0}},
        la::DecompositionMethod::SingularValue));
}

}  // namespace

int main() {
    test_try_set_covariance_non_throwing_invalid_state();
    test_marginal_validates_indices();
    test_conditional_validates_indices_and_values();
    test_singular_value_decomposition_path();
    return chtest::summary("test_multivariate_normal_api");
}
