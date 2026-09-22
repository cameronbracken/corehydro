#include <array>
#include <cmath>
#include <vector>

#include "corehydro/numerics/math/special/debye.hpp"
#include "corehydro/numerics/math/special/erf.hpp"
#include "corehydro/numerics/math/special/factorial.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/data/global_sensitivity.hpp"
#include "corehydro/numerics/data/probability.hpp"
#include "corehydro/numerics/support/toolbox_runner.hpp"
#include "check.hpp"

namespace special = corehydro::numerics::math::special;

namespace {

void test_debye_order_one() {
    const std::array<double, 7> x{-100.0, -1.0, -0.001, 0.0, 0.001, 1.0, 100.0};
    const std::array<double, 7> expected{50.01644934066848, 1.2775046341122482,
                                         1.0002500277777775, 1.0, 0.9997500277777776,
                                         0.7775046341122482, 0.016449340668482266};
    for (std::size_t i = 0; i < x.size(); ++i) {
        CHECK_NEAR(special::debye_function_order_one(x[i]), expected[i],
                   std::fabs(expected[i]) * 1e-14);
    }
}

void test_factorial_combination_iteration() {
    std::vector<int> combination{0, 1};
    CHECK_TRUE(special::factorial::next_combination(combination, 4));
    CHECK_EQ(combination[0], 0);
    CHECK_EQ(combination[1], 2);
    std::vector<int> empty;
    std::vector<int> duplicate{0, 0};
    std::vector<int> out_of_range{0, 3};
    std::vector<int> valid{0};
    CHECK_THROWS(special::factorial::next_combination(empty, 3));
    CHECK_THROWS(special::factorial::next_combination(duplicate, 3));
    CHECK_THROWS(special::factorial::next_combination(out_of_range, 3));
    CHECK_THROWS(special::factorial::next_combination(valid, -1));

    const auto all = special::factorial::all_combinations_lazy(3);
    const std::vector<std::vector<int>> expected{{0}, {1}, {2}, {0, 1}, {0, 2},
                                                  {1, 2}, {0, 1, 2}};
    CHECK_EQ(all, expected);
}

void test_error_function_far_tail() {
    CHECK_NEAR(special::erf::erfc(6.0), 2.1519736712498913e-17,
               2.1519736712498913e-29);
    CHECK_NEAR(special::erf::erfc(15.0), 7.212994172451207e-100,
               7.212994172451207e-112);
    CHECK_NEAR(special::erf::inverse_erfc(1e-100), 15.065574702592645,
               15.065574702592645e-9);
}

void test_incomplete_gamma_reference_values() {
    CHECK_NEAR(special::incomplete(1.5, 0.1), 0.9886559833621947, 1e-11);
    CHECK_NEAR(special::incomplete(500.0, 500.0), 0.5059471461707603, 1e-11);
    CHECK_NEAR(special::incomplete(0.075, 0.5), 0.30146464169666126, 1e-11);
    CHECK_NEAR(special::incomplete(10500.0, 10400.0), 0.8366349011570255, 1e-5);
}

void test_global_sensitivity_tied_ranks_and_dispatch() {
    std::vector<double> x(32, 1.0);
    std::vector<double> y(32);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = i < 16 ? 0.0 : 1.0;
    namespace gs = corehydro::numerics::data::global_sensitivity;
    CHECK_EQ(gs::first_order_sobol(x, y, 2), 1.0);
    CHECK_EQ(gs::pawn(x, y, 2), std::vector<double>({0.5, 0.5}));
    CHECK_EQ(gs::borgonovo_delta(x, y, 2, 2), 0.5);

    const auto dispatched = corehydro::numerics::support::run_toolbox(
        "statistics", "first_order_sobol", {x, y}, "{\"bins\":2}");
    CHECK_EQ(dispatched.values[0], 1.0);
}

void test_single_factor_probability() {
    namespace probability = corehydro::numerics::data::probability;
    const std::vector<double> probabilities{0.01, 0.05, 0.2, 0.001};
    CHECK_NEAR(probability::union_single_factor(probabilities, 0.0),
               probability::independent_union(probabilities),
               1e-12 * probability::independent_union(probabilities));
    CHECK_EQ(probability::union_single_factor({0.05, 0.05, 0.05}, 1.0), 0.05);
    CHECK_EQ(probability::union_single_factor({0.0, 0.0}, 0.5), 0.0);
    CHECK_EQ(probability::union_single_factor({0.1, 1.0}, 0.5), 1.0);

    std::vector<double> thresholds;
    for (double probability_value : std::vector<double>{0.01, 0.2, 0.7}) {
        thresholds.push_back(corehydro::numerics::distributions::Normal::standard_z(probability_value));
    }
    std::vector<double> conditional(5, -99.0);
    probability::single_factor_conditional_probabilities(thresholds, 0.0, 1.7, conditional);
    CHECK_NEAR(conditional[0], 0.01, 1e-14);
    CHECK_NEAR(conditional[1], 0.2, 1e-14);
    CHECK_NEAR(conditional[2], 0.7, 1e-14);
    CHECK_EQ(conditional[3], -99.0);
    CHECK_THROWS(special::factorial::all_combinations(31));

    const auto complete = probability::independent_exclusive_lazy({0.2, 0.3}, true);
    CHECK_EQ(complete.status, probability::ExclusiveEnumerationStatus::Complete);
    CHECK_EQ(complete.indicators.size(), std::size_t{4});
    CHECK_NEAR(complete.probabilities[0], 0.56, 1e-15);
    CHECK_NEAR(complete.probabilities[1], 0.14, 1e-15);
    CHECK_NEAR(complete.probabilities[2], 0.24, 1e-15);
    CHECK_NEAR(complete.probabilities[3], 0.06, 1e-15);

    const auto capped = probability::independent_exclusive_lazy({0.2, 0.3, 0.4}, false, 2);
    CHECK_EQ(capped.status, probability::ExclusiveEnumerationStatus::Capped);
    CHECK_EQ(capped.indicators.back(), std::vector<int>({1, 1, 1}));
    double capped_mass = 0.0;
    for (double value : capped.probabilities) capped_mass += value;
    CHECK_NEAR(capped_mass, 1.0 - 0.8 * 0.7 * 0.6, 1e-15);

    const auto dense_rows = special::factorial::all_combinations(3);
    const std::vector<int> counts{3, 3, 1};
    std::vector<double> pooled_probabilities;
    std::vector<std::vector<int>> pooled_indicators;
    CHECK_TRUE(!probability::independent_exclusive(
        {0.3, 0.2, 0.1}, counts, dense_rows, pooled_probabilities,
        pooled_indicators, 0.0, 0.0));
    CHECK_EQ(pooled_probabilities.size(), std::size_t{7});
    CHECK_EQ(pooled_indicators, dense_rows);

    const auto positive = probability::positively_dependent_exclusive_lazy(
        {0.4, 0.3, 0.2, 0.1}, 0.0, 0.0);
    CHECK_EQ(positive.status, probability::ExclusiveEnumerationStatus::Complete);
    CHECK_EQ(positive.indicators.size(), std::size_t{15});
    CHECK_NEAR(positive.probabilities[0], 0.1, 1e-15);
    CHECK_NEAR(positive.probabilities[3], 0.0, 1e-15);

    probability::Matrix2D correlation(4, std::vector<double>(4, 0.25));
    for (std::size_t i = 0; i < correlation.size(); ++i) correlation[i][i] = 1.0;
    probability::ExclusiveEnumerationStatus pcm_status{};
    const double lazy_pcm = probability::union_pcm_lazy(
        {0.32, 0.27, 0.19, 0.11}, correlation, pcm_status, 0.0, 0.0);
    CHECK_EQ(pcm_status, probability::ExclusiveEnumerationStatus::Complete);
    CHECK_NEAR(lazy_pcm,
               probability::union_pcm({0.32, 0.27, 0.19, 0.11}, correlation, 0.0, 0.0),
               0.0);
    const auto exclusive_pcm = probability::exclusive_pcm_lazy(
        {0.32, 0.27, 0.19, 0.11}, correlation, 0.0, 0.0);
    CHECK_EQ(exclusive_pcm.status, probability::ExclusiveEnumerationStatus::Complete);
    CHECK_EQ(exclusive_pcm.indicators, special::factorial::all_combinations(4));
    double exclusive_mass = 0.0;
    for (double value : exclusive_pcm.probabilities) exclusive_mass += value;
    CHECK_NEAR(exclusive_mass, lazy_pcm, 1e-14);
}

}  // namespace

int main() {
    test_debye_order_one();
    test_factorial_combination_iteration();
    test_error_function_far_tail();
    test_incomplete_gamma_reference_values();
    test_global_sensitivity_tied_ranks_and_dispatch();
    test_single_factor_probability();
    return chtest::summary("test_numerics_v220_foundations");
}
