// Numerics v2.2 dependency-simulation regressions for CompetingRisks.
#include <cmath>
#include <memory>
#include <vector>

#include "corehydro/numerics/data/probability.hpp"
#include "corehydro/numerics/distributions/competing_risks.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "check.hpp"

using corehydro::numerics::data::probability::DependencyType;
using corehydro::numerics::distributions::CompetingRisks;
using corehydro::numerics::distributions::LnNormal;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::UnivariateDistributionBase;

namespace {

std::vector<std::unique_ptr<UnivariateDistributionBase>> transformed_marginals() {
    std::vector<std::unique_ptr<UnivariateDistributionBase>> components;
    components.push_back(std::make_unique<Normal>(10.0, 2.0));
    components.push_back(std::make_unique<LnNormal>(12.0, 3.0));
    return components;
}

void test_perfectly_positive_simulation_is_seeded() {
    CompetingRisks risks(transformed_marginals());
    risks.set_dependency(DependencyType::PerfectlyPositive);

    const auto first = risks.generate_random_values_with_dependency(16, 42);
    const auto second = risks.generate_random_values_with_dependency(16, 42);
    CHECK_TRUE(first == second);
    for (double value : first) CHECK_TRUE(std::isfinite(value));
}

void test_perfectly_negative_simulation_is_seeded() {
    CompetingRisks risks(transformed_marginals());
    risks.set_dependency(DependencyType::PerfectlyNegative);

    const auto first = risks.generate_random_values_with_dependency(16, 42);
    const auto second = risks.generate_random_values_with_dependency(16, 42);
    CHECK_TRUE(first == second);
    for (double value : first) CHECK_TRUE(std::isfinite(value));
}

void test_correlation_matrix_simulation_is_seeded() {
    CompetingRisks risks(transformed_marginals());
    risks.set_dependency(DependencyType::CorrelationMatrix);
    risks.set_correlation_matrix({{1.0, 0.5}, {0.5, 1.0}});

    const auto first = risks.generate_random_values_with_dependency(16, 42);
    const auto second = risks.generate_random_values_with_dependency(16, 42);
    CHECK_TRUE(first == second);
    for (double value : first) CHECK_TRUE(std::isfinite(value));
}

}  // namespace

int main() {
    test_perfectly_positive_simulation_is_seeded();
    test_perfectly_negative_simulation_is_seeded();
    test_correlation_matrix_simulation_is_seeded();
    return chtest::summary("test_competing_risks");
}
