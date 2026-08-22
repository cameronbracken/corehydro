// Transcribed C# oracle tests for the three stochastic integrators (Task 5):
//   upstream/Numerics/Test_Numerics/Mathematics/Integration/Test_MonteCarlo.cs @ 2a0357a (5)
//   upstream/Numerics/Test_Numerics/Mathematics/Integration/Test_Miser.cs @ 2a0357a (5)
//   upstream/Numerics/Test_Numerics/Mathematics/Integration/Test_Vegas.cs @ 2a0357a (9)
// Shared integrands transcribed from Integrands.cs: PI, GSL, SumOfNormals (with its mu20/sigma20
// literal arrays). All 19 upstream [TestMethod]s are transcribed with their exact tolerances,
// unaltered. These are internal-support ports validated against the C# test oracles themselves
// (no fixtures/ entry -- fixtures/ is the public estimation/dist/toolbox API surface only).
//
// PowerTransform_ProbabilityRange note: the C# test's per-sample `Assert.IsTrue(p >= 0 && p <=
// 1)` throws INSIDE the integrand, which Vegas.Integrate()'s own try/catch would catch and
// rethrow (ReportFailure defaults true), failing the test via an unhandled exception. This port
// tracks the same condition with a captured bool instead of throwing from inside the integrand,
// checked once after integrate() returns -- an equivalent, simpler transcription of the same
// assertion.
//
// SOBOL_DATA_PATH is injected by CMakeLists.txt via target_compile_definitions (this test is
// registered through the plain BF_TESTS loop, which does not pass argv, unlike test_sobol/
// test_fixtures) -- see the file's CMakeLists.txt comment for why.
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/integration/miser.hpp"
#include "corehydro/numerics/math/integration/monte_carlo_integration.hpp"
#include "corehydro/numerics/math/integration/vegas.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"

using corehydro::numerics::distributions::Normal;
using corehydro::numerics::math::integration::Miser;
using corehydro::numerics::math::integration::MonteCarloIntegration;
using corehydro::numerics::math::integration::Vegas;
using corehydro::numerics::sampling::MersenneTwister;

namespace {

#ifndef SOBOL_DATA_PATH
#error "SOBOL_DATA_PATH must be defined by CMakeLists.txt"
#endif
const std::string kSobolPath = SOBOL_DATA_PATH;

// ============================== Integrands.cs ==============================

double integrand_pi(const std::vector<double>& v) {
    double x = v[0], y = v[1];
    return (x * x + y * y < 1.0) ? 1.0 : 0.0;
}

double integrand_gsl(const std::vector<double>& v) {
    double A = 1.0 / (corehydro::numerics::kPi * corehydro::numerics::kPi * corehydro::numerics::kPi);
    return A / (1.0 - std::cos(v[0]) * std::cos(v[1]) * std::cos(v[2]));
}

const double kMu20[20] = {10, 30, 17, 99, 68, 26, 35, 55, 13, 59,
                          12, 28, 49, 54, 20, 47, 12, 76, 70, 57};
const double kSigma20[20] = {2,  15, 5,  14, 7,  24, 29, 22, 22, 1,
                             3,  28, 19, 18, 4,  24, 23, 26, 26, 19};

double integrand_sum_of_normals(const std::vector<double>& p) {
    double result = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        result += kMu20[i] + kSigma20[i] * Normal::standard_z(p[i]);
    }
    return result;
}

std::vector<double> normals_min(int n) { return std::vector<double>(static_cast<std::size_t>(n), 1E-16); }
std::vector<double> normals_max(int n) {
    return std::vector<double>(static_cast<std::size_t>(n), 1.0 - 1E-16);
}

// ============================== Test_MonteCarlo.cs ==============================

void mc_pi() {
    MonteCarloIntegration mc(integrand_pi, 2, {-1, -1}, {1, 1});
    mc.random = MersenneTwister(12345);
    mc.min_iterations = 1000;
    mc.max_iterations = 100000000;
    mc.absolute_tolerance = 1E-8;
    mc.relative_tolerance = 1E-4;
    mc.integrate();
    double result = mc.result();
    double true_result = 3.14;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

void mc_gsl() {
    MonteCarloIntegration mc(integrand_gsl, 3, {0, 0, 0},
                             {corehydro::numerics::kPi, corehydro::numerics::kPi,
                              corehydro::numerics::kPi});
    mc.random = MersenneTwister(12345);
    mc.min_iterations = 1000;
    mc.max_iterations = 100000000;
    mc.absolute_tolerance = 1E-8;
    mc.relative_tolerance = 1E-4;
    mc.integrate();
    double result = mc.result();
    double true_result = 1.3932039296856768591842462603255;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

void mc_sum_of_three_normals() {
    MonteCarloIntegration mc(integrand_sum_of_normals, 3, normals_min(3), normals_max(3));
    mc.random = MersenneTwister(12345);
    mc.min_iterations = 1000;
    mc.max_iterations = 100000000;
    mc.absolute_tolerance = 1E-8;
    mc.relative_tolerance = 1E-4;
    mc.integrate();
    double result = mc.result();
    double true_result = 57;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

void mc_sum_of_five_normals() {
    MonteCarloIntegration mc(integrand_sum_of_normals, 5, normals_min(5), normals_max(5));
    mc.random = MersenneTwister(12345);
    mc.min_iterations = 1000;
    mc.max_iterations = 100000000;
    mc.absolute_tolerance = 1E-8;
    mc.relative_tolerance = 1E-4;
    mc.integrate();
    double result = mc.result();
    double true_result = 224;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

void mc_sum_of_twenty_normals() {
    MonteCarloIntegration mc(integrand_sum_of_normals, 20, normals_min(20), normals_max(20));
    mc.random = MersenneTwister(12345);
    mc.min_iterations = 1000;
    mc.max_iterations = 100000000;
    mc.absolute_tolerance = 1E-8;
    mc.relative_tolerance = 1E-4;
    mc.integrate();
    double result = mc.result();
    double true_result = 837;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

// ============================== Test_Miser.cs ==============================

void miser_pi() {
    Miser miser(integrand_pi, 2, {-1, -1}, {1, 1}, kSobolPath);
    miser.random = MersenneTwister(12345);
    miser.integrate();
    double result = miser.result();
    double true_result = 3.14;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

void miser_gsl() {
    Miser miser(integrand_gsl, 3, {0, 0, 0},
               {corehydro::numerics::kPi, corehydro::numerics::kPi, corehydro::numerics::kPi},
               kSobolPath);
    miser.random = MersenneTwister(12345);
    miser.integrate();
    double result = miser.result();
    double true_result = 1.3932039296856768591842462603255;
    CHECK_NEAR(result, true_result, 1E-2 * true_result);
}

void miser_sum_of_three_normals() {
    Miser miser(integrand_sum_of_normals, 3, normals_min(3), normals_max(3), kSobolPath);
    miser.random = MersenneTwister(12345);
    miser.integrate();
    double result = miser.result();
    double true_result = 57;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

void miser_sum_of_five_normals() {
    Miser miser(integrand_sum_of_normals, 5, normals_min(5), normals_max(5), kSobolPath);
    miser.random = MersenneTwister(12345);
    miser.integrate();
    double result = miser.result();
    double true_result = 224;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

void miser_sum_of_twenty_normals() {
    Miser miser(integrand_sum_of_normals, 20, normals_min(20), normals_max(20), kSobolPath);
    miser.random = MersenneTwister(12345);
    miser.integrate();
    double result = miser.result();
    double true_result = 837;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

// ============================== Test_Vegas.cs ==============================

void vegas_pi() {
    Vegas vegas([](const std::vector<double>& x, double) { return integrand_pi(x); }, 2,
               {-1, -1}, {1, 1}, kSobolPath);
    vegas.integrate();
    double result = vegas.result();
    double true_result = 3.1416;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

void vegas_gsl() {
    Vegas vegas([](const std::vector<double>& x, double) { return integrand_gsl(x); }, 3,
               {0, 0, 0},
               {corehydro::numerics::kPi, corehydro::numerics::kPi, corehydro::numerics::kPi},
               kSobolPath);
    vegas.integrate();
    double result = vegas.result();
    double true_result = 1.3932039296856768591842462603255;
    CHECK_NEAR(result, true_result, 1E-2 * true_result);
}

void vegas_sum_of_three_normals() {
    Vegas vegas([](const std::vector<double>& x, double) { return integrand_sum_of_normals(x); },
               3, normals_min(3), normals_max(3), kSobolPath);
    vegas.integrate();
    double result = vegas.result();
    double true_result = 57;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

void vegas_sum_of_five_normals() {
    Vegas vegas([](const std::vector<double>& x, double) { return integrand_sum_of_normals(x); },
               5, normals_min(5), normals_max(5), kSobolPath);
    vegas.integrate();
    double result = vegas.result();
    double true_result = 224;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

void vegas_sum_of_twenty_normals() {
    Vegas vegas([](const std::vector<double>& x, double) { return integrand_sum_of_normals(x); },
               20, normals_min(20), normals_max(20), kSobolPath);
    vegas.integrate();
    double result = vegas.result();
    double true_result = 837;
    CHECK_NEAR(result, true_result, 1E-3 * true_result);
}

// Test 1: Verify backward compatibility - Power Transform with gamma=1 should match standard Vegas.
void vegas_power_transform_backward_compatibility() {
    std::vector<double> min = normals_min(5), max = normals_max(5);

    Vegas vegas_standard(
        [](const std::vector<double>& x, double) { return integrand_sum_of_normals(x); }, 5, min,
        max, kSobolPath);
    vegas_standard.integrate();
    double result_standard = vegas_standard.result();

    Vegas vegas_power(
        [](const std::vector<double>& x, double) { return integrand_sum_of_normals(x); }, 5, min,
        max, kSobolPath);
    vegas_power.tail_focus_parameter = 1.0;
    vegas_power.integrate();
    double result_power = vegas_power.result();

    double true_result = 224;
    CHECK_NEAR(result_standard, true_result, 1E-2 * true_result);
    CHECK_NEAR(result_power, true_result, 1E-2 * true_result);
    CHECK_NEAR(result_standard, result_power, 1E-2 * true_result);
}

// Test 2: Find probability of rare upper tail event P(Sum > threshold).
void vegas_power_transform_rare_upper_tail_event() {
    std::vector<double> min = normals_min(5), max = normals_max(5);
    double threshold = 291.0;

    Vegas vegas(
        [threshold](const std::vector<double>& x, double) {
            double sum = integrand_sum_of_normals(x);
            return (sum > threshold) ? 1.0 : 0.0;
        },
        5, min, max, kSobolPath);

    vegas.tail_focus_parameter = 2.0;
    vegas.set_number_of_bins(100);
    vegas.function_calls = 50000;
    vegas.independent_evaluations = 5;
    vegas.alpha = 1.8;

    vegas.integrate();
    double failure_probability = vegas.result();

    CHECK_TRUE(failure_probability > 5E-4);
    CHECK_TRUE(failure_probability < 5E-3);

    double relative_error = vegas.standard_error() / std::fabs(failure_probability);
    CHECK_TRUE(relative_error < 0.5);
}

// Test 3: Adaptive configuration for very rare events (p ~ 1e-6) via ConfigureForRareEvents().
void vegas_power_transform_very_rare_event() {
    std::vector<double> min = normals_min(3), max = normals_max(3);
    double threshold = 128.5;

    Vegas vegas(
        [threshold](const std::vector<double>& x, double) {
            double sum = integrand_sum_of_normals(x);
            return (sum > threshold) ? 1.0 : 0.0;
        },
        3, min, max, kSobolPath);

    vegas.configure_for_rare_events(1e-5);
    vegas.function_calls = 100000;
    vegas.independent_evaluations = 5;

    vegas.integrate();
    double failure_probability = vegas.result();

    CHECK_TRUE(failure_probability > 1E-7);
    CHECK_TRUE(failure_probability < 1E-4);

    double relative_error = vegas.standard_error() / std::fabs(failure_probability);
    CHECK_TRUE(relative_error < 1.0);
    CHECK_TRUE(!std::isnan(failure_probability));
    CHECK_TRUE(!std::isinf(failure_probability));
}

// Test 4: BONUS - Verify that the integrand receives correct probabilities (power transform
// doesn't break the probability input range).
void vegas_power_transform_probability_range() {
    std::vector<double> min(3, 0.0), max(3, 1.0);

    double min_observed = 1.0, max_observed = 0.0;
    int sample_count = 0;
    bool range_ok = true;

    Vegas vegas(
        [&](const std::vector<double>& x, double) {
            sample_count++;
            for (double p : x) {
                min_observed = std::min(min_observed, p);
                max_observed = std::max(max_observed, p);
                if (!(p >= 0.0 && p <= 1.0)) range_ok = false;
            }
            return integrand_sum_of_normals(x);
        },
        3, min, max, kSobolPath);

    vegas.tail_focus_parameter = 4.0;
    vegas.function_calls = 10000;
    vegas.independent_evaluations = 2;

    vegas.integrate();

    CHECK_TRUE(range_ok);
    CHECK_TRUE(sample_count > 0);
    CHECK_TRUE(max_observed > 0.99);
    CHECK_TRUE(min_observed < 0.5);
}

}  // namespace

int main() {
    // Test_MonteCarlo.cs
    mc_pi();
    mc_gsl();
    mc_sum_of_three_normals();
    mc_sum_of_five_normals();
    mc_sum_of_twenty_normals();

    // Test_Miser.cs
    miser_pi();
    miser_gsl();
    miser_sum_of_three_normals();
    miser_sum_of_five_normals();
    miser_sum_of_twenty_normals();

    // Test_Vegas.cs
    vegas_pi();
    vegas_gsl();
    vegas_sum_of_three_normals();
    vegas_sum_of_five_normals();
    vegas_sum_of_twenty_normals();
    vegas_power_transform_backward_compatibility();
    vegas_power_transform_rare_upper_tail_event();
    vegas_power_transform_very_rare_event();
    vegas_power_transform_probability_range();

    return chtest::summary("test_integration_random");
}
