// Transcribed C# oracle tests for the local optimizers (Task B5 added BFGS; Task B6 added
// the Powell section; MLSL lives in test_optimizers_global.cpp):
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Local/Test_BFGS.cs @ 2a0357a
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Local/Test_Powell.cs @ a2c4dbf
//
// All 8 upstream [TestMethod]s of each class are transcribed with their exact fitness +
// coordinate oracles and tolerances (1E-4 throughout, except the BFGS SumOfPowerFunctions
// coordinates at 1E-3), unaltered. These are internal-support ports validated against the
// C# test oracles themselves, so there is no fixtures/ entry for this file (fixtures/ is
// the public estimation API surface only). Skipped upstream methods: none (neither
// Test_BFGS.cs nor Test_Powell.cs has XML or INotifyPropertyChanged tests).
//
// SUPPLEMENT (clearly marked, not from Test_BFGS.cs): direct unit checks for the two
// Tools.cs functions this task ports -- sum_product (Tools.SumProduct, used by the BFGS
// strong-Wolfe line search) and normalized_distance (Tools.NormalizedDistance, whose only
// caller, MLSL, arrives in B6) -- against hand-computed values, since no upstream C# test
// exercises them directly at this layer.
#include <cmath>
#include <vector>

#include "corehydro/numerics/math/optimization/bfgs.hpp"
#include "corehydro/numerics/math/optimization/powell.hpp"
#include "corehydro/numerics/tools.hpp"
#include "check.hpp"
#include "optimization_test_functions.hpp"

using corehydro::numerics::math::optimization::BFGS;
using corehydro::numerics::math::optimization::OptimizationStatus;
using corehydro::numerics::math::optimization::Powell;

namespace {

// ============================== BFGS (Test_BFGS.cs) ==============================

// Test the BFGS algorithm with a simple 3-dimensional test function.
void bfgs_fxyz() {
    std::vector<double> initial = {0.2, 0.5, 0.5};
    std::vector<double> lower = {0.0, 0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = BFGS(test_functions::fxyz, 3, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double z = solution[2];
    double validX = 0.125;
    double validY = 0.2;
    double validZ = 0.35;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
    CHECK_NEAR(z, validZ, 1E-4);
}

// Test the BFGS algorithm with the De Jong Function in 5-D.
void bfgs_de_jong() {
    std::vector<double> initial = {1.0, -1.0, 2.0, -2.0, 1.0};
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    auto solver = BFGS(test_functions::de_jong, 5, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the BFGS algorithm with the Sum of Power functions in 3-D.
void bfgs_sum_of_power_functions() {
    std::vector<double> initial = {0.5, -0.5, 0.5};
    std::vector<double> lower = {-1.0, -1.0, -1.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = BFGS(test_functions::sum_of_power_functions, 3, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-3);
}

// Test the BFGS algorithm with the Rosenbrock Function in 2-D.
void bfgs_rosenbrock() {
    std::vector<double> initial = {0, 0};
    std::vector<double> lower = {-2.048, -2.048};
    std::vector<double> upper = {2.048, 2.048};
    auto solver = BFGS(test_functions::rosenbrock, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {1.0, 1.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the BFGS algorithm with the Booth Function.
void bfgs_booth() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = BFGS(test_functions::booth, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 1.0;
    double validY = 3.0;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the BFGS algorithm with the Matyas Function.
void bfgs_matyas() {
    std::vector<double> initial = {1.0, -1.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = BFGS(test_functions::matyas, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 0.0;
    double validY = 0.0;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the BFGS algorithm with the McCormick Function.
void bfgs_mccormick() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-1.5, -3.0};
    std::vector<double> upper = {4.0, 4.0};
    auto solver = BFGS(test_functions::mccormick, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = -1.9133;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = -0.54719;
    double validY = -1.54719;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the BFGS algorithm with the Beale Function.
void bfgs_beale() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-4.5, -4.5};
    std::vector<double> upper = {4.5, 4.5};
    auto solver = BFGS(test_functions::beale, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 3.0;
    double validY = 0.5;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// ============================== Powell (Test_Powell.cs) ==============================

// Test the Powell algorithm with a simple 3-dimensional test function.
void powell_fxyz() {
    std::vector<double> initial = {0.2, 0.5, 0.5};
    std::vector<double> lower = {0.0, 0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = Powell(test_functions::fxyz, 3, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double z = solution[2];
    double validX = 0.125;
    double validY = 0.2;
    double validZ = 0.35;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
    CHECK_NEAR(z, validZ, 1E-4);
}

// Test the Powell algorithm with the De Jong Function in 5-D.
void powell_de_jong() {
    std::vector<double> initial = {1.0, -1.0, 2.0, -2.0, 1.0};
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    auto solver = Powell(test_functions::de_jong, 5, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Powell algorithm with the Sum of Power functions in 3-D.
void powell_sum_of_power_functions() {
    std::vector<double> initial = {0.5, -0.5, 0.5};
    std::vector<double> lower = {-1.0, -1.0, -1.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = Powell(test_functions::sum_of_power_functions, 3, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Powell algorithm with the Rosenbrock Function in 2-D.
void powell_rosenbrock() {
    std::vector<double> initial = {0, 0};
    std::vector<double> lower = {-2.048, -2.048};
    std::vector<double> upper = {2.048, 2.048};
    auto solver = Powell(test_functions::rosenbrock, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {1.0, 1.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Powell algorithm with the Booth Function.
void powell_booth() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = Powell(test_functions::booth, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 1.0;
    double validY = 3.0;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the Powell algorithm with the Matyas Function.
void powell_matyas() {
    std::vector<double> initial = {1.0, -1.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = Powell(test_functions::matyas, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 0.0;
    double validY = 0.0;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the Powell algorithm with the McCormick Function.
void powell_mccormick() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-1.5, -3.0};
    std::vector<double> upper = {4.0, 4.0};
    auto solver = Powell(test_functions::mccormick, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = -1.9133;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = -0.54719;
    double validY = -1.54719;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the Powell algorithm with the Beale Function.
void powell_beale() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-4.5, -4.5};
    std::vector<double> upper = {4.5, 4.5};
    auto solver = Powell(test_functions::beale, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 3.0;
    double validY = 0.5;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// ==================== Numerics v2.2.0 optimizer regressions ====================

void bfgs_stationary_start_does_not_search() {
    int gradients = 0;
    auto solver = BFGS(
        [](std::vector<double>& p) { return p[0] * p[0]; }, 1, {0.0}, {-10.0}, {10.0},
        [&gradients](const std::vector<double>& p) {
            ++gradients;
            return std::vector<double>{2.0 * p[0]};
        });
    solver.compute_hessian = false;
    solver.minimize();
    CHECK_TRUE(solver.status() == OptimizationStatus::Success);
    CHECK_EQ(solver.iterations(), 0);
    CHECK_EQ(solver.function_evaluations(), 1);
    CHECK_EQ(gradients, 1);
}

void bfgs_maximize_scales_supplied_gradient() {
    std::vector<double> buffer(1);
    auto solver = BFGS(
        [](std::vector<double>& p) { return -(p[0] - 2.0) * (p[0] - 2.0); },
        1, {0.0}, {-10.0}, {10.0},
        [&buffer](const std::vector<double>& p) {
            buffer[0] = -2.0 * (p[0] - 2.0);
            return buffer;
        });
    solver.compute_hessian = false;
    solver.maximize();
    CHECK_TRUE(solver.status() == OptimizationStatus::Success);
    CHECK_NEAR(solver.best_parameter_set().values[0], 2.0, 1e-8);
}

void bfgs_boundary_uses_projected_gradient_and_bounded_hessian() {
    int outside = 0;
    auto objective = [&outside](std::vector<double>& p) {
        if (p[0] < 0.0 || p[0] > 1.0 || p[1] < 0.0 || p[1] > 1.0) ++outside;
        return (p[0] - 2.0) * (p[0] - 2.0) + (p[1] - 0.3) * (p[1] - 0.3);
    };
    auto solver = BFGS(objective, 2, {0.0, 0.0}, {0.0, 0.0}, {1.0, 1.0},
                       [](const std::vector<double>& p) {
                           return std::vector<double>{2.0 * (p[0] - 2.0), 2.0 * (p[1] - 0.3)};
                       });
    solver.minimize();
    CHECK_TRUE(solver.status() == OptimizationStatus::Success);
    CHECK_EQ(solver.best_parameter_set().values[0], 1.0);
    CHECK_NEAR(solver.best_parameter_set().values[1], 0.3, 1e-8);
    CHECK_TRUE(solver.hessian().has_value());
    CHECK_EQ(outside, 0);
}

void bfgs_reports_line_search_and_invalid_gradient_statuses() {
    auto inconsistent = BFGS([](std::vector<double>&) { return 1.0; }, 1, {0.0}, {-10.0}, {10.0},
                             [](const std::vector<double>&) { return std::vector<double>{1.0}; });
    inconsistent.report_failure = false;
    inconsistent.minimize();
    CHECK_TRUE(inconsistent.status() == OptimizationStatus::LineSearchFailed);
    CHECK_EQ(inconsistent.iterations(), 0);
    CHECK_TRUE(inconsistent.hessian().has_value());

    auto invalid = BFGS([](std::vector<double>& p) { return p[0] * p[0]; }, 1,
                        {1.0}, {-10.0}, {10.0}, [](const std::vector<double>&) {
                            return std::vector<double>{std::numeric_limits<double>::quiet_NaN()};
                        });
    invalid.report_failure = false;
    invalid.compute_hessian = false;
    invalid.minimize();
    CHECK_TRUE(invalid.status() == OptimizationStatus::Failure);
}

void bfgs_preserves_evaluation_budget() {
    auto solver = BFGS([](std::vector<double>& p) { return -p[0]; }, 1, {0.0},
                       {-std::numeric_limits<double>::infinity()},
                       {std::numeric_limits<double>::infinity()},
                       [](const std::vector<double>&) { return std::vector<double>{-1.0}; });
    solver.max_function_evaluations = 10;
    solver.report_failure = false;
    solver.compute_hessian = false;
    solver.minimize();
    CHECK_TRUE(solver.status() == OptimizationStatus::MaximumFunctionEvaluationsReached);
    CHECK_EQ(solver.function_evaluations(), 10);
}

double corner_quadratic(std::vector<double>& p) {
    return (p[0] - 20.0) * (p[0] - 20.0) + (p[1] - 20.0) * (p[1] - 20.0);
}

void powell_never_evaluates_outside_bounds() {
    int outside = 0;
    auto solver = Powell(
        [&outside](std::vector<double>& p) {
            if (p[0] < 0.0 || p[0] > 1.0 || p[1] < 0.0 || p[1] > 1.0) ++outside;
            return corner_quadratic(p);
        },
        2, {0.5, 0.5}, {0.0, 0.0}, {1.0, 1.0});
    solver.report_failure = false;
    solver.record_traces = false;
    solver.minimize();
    CHECK_EQ(outside, 0);
    CHECK_EQ(solver.best_parameter_set().values[0], 1.0);
    CHECK_EQ(solver.best_parameter_set().values[1], 1.0);
    CHECK_EQ(solver.best_parameter_set().fitness, 722.0);
}

void powell_handles_degenerate_and_infinite_boxes() {
    auto pinned = Powell([](std::vector<double>& p) {
        return (p[0] - 3.0) * (p[0] - 3.0) + (p[1] + 2.0) * (p[1] + 2.0);
    }, 2, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0});
    pinned.report_failure = false;
    pinned.compute_hessian = false;
    pinned.minimize();
    CHECK_TRUE(pinned.status() == OptimizationStatus::Success);
    CHECK_EQ(pinned.best_parameter_set().fitness, 13.0);

    double inf = std::numeric_limits<double>::infinity();
    auto unbounded = Powell([](std::vector<double>& p) {
        return (p[0] - 30.0) * (p[0] - 30.0) + (p[1] + 40.0) * (p[1] + 40.0);
    }, 2, {0.0, 0.0}, {-inf, -inf}, {inf, inf});
    unbounded.report_failure = false;
    unbounded.compute_hessian = false;
    unbounded.minimize();
    CHECK_NEAR(unbounded.best_parameter_set().values[0], 30.0, 1e-4);
    CHECK_NEAR(unbounded.best_parameter_set().values[1], -40.0, 1e-4);
}

// ==================== SUPPLEMENT: Tools additions (B5, hand-computed) ====================
// Not from Test_BFGS.cs -- direct unit checks for the two Tools.cs functions B5 ports.

// Tools.SumProduct: dot product of two equal-length lists.
void tools_sum_product() {
    // (1*4) + (2*5) + (3*6) = 32
    CHECK_NEAR(corehydro::numerics::sum_product({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}), 32.0, 1e-12);
    // Single element: (-2.5)*4 = -10
    CHECK_NEAR(corehydro::numerics::sum_product({-2.5}, {4.0}), -10.0, 1e-12);
    // Empty first list -> NaN (C# returns double.NaN).
    CHECK_TRUE(std::isnan(corehydro::numerics::sum_product({}, {})));
    // Mismatched lengths -> NaN (C# returns double.NaN).
    CHECK_TRUE(std::isnan(corehydro::numerics::sum_product({1.0, 2.0}, {1.0})));
}

// Tools.NormalizedDistance: Euclidean distance after min-max normalizing each dimension.
void tools_normalized_distance() {
    // Dimension 1: range 10, normalized delta (3-1)/10 = 0.2.
    // Dimension 2: range 4,  normalized delta (2-0)/4  = 0.5.
    // Distance = sqrt(0.2^2 + 0.5^2) = sqrt(0.29).
    CHECK_NEAR(corehydro::numerics::normalized_distance({1.0, 0.0}, {3.0, 2.0}, {0.0, -2.0},
                                                      {10.0, 2.0}),
               std::sqrt(0.29), 1e-12);
    // Identical points -> 0.
    CHECK_NEAR(corehydro::numerics::normalized_distance({0.5, 0.5}, {0.5, 0.5}, {0.0, 0.0},
                                                      {1.0, 1.0}),
               0.0, 1e-15);
    // Degenerate dimension (range <= 0) contributes nothing: only dim 2 counts,
    // delta (4-1)/10 = 0.3.
    CHECK_NEAR(corehydro::numerics::normalized_distance({1.0, 1.0}, {2.0, 4.0}, {5.0, 0.0},
                                                      {5.0, 10.0}),
               0.3, 1e-12);
    // NaN range likewise contributes nothing -> all dims degenerate -> 0.
    CHECK_NEAR(corehydro::numerics::normalized_distance({1.0}, {2.0}, {0.0}, {std::nan("")}),
               0.0, 1e-15);
}

}  // namespace

int main() {
    // Test_BFGS.cs
    bfgs_fxyz();
    bfgs_de_jong();
    bfgs_sum_of_power_functions();
    bfgs_rosenbrock();
    bfgs_booth();
    bfgs_matyas();
    bfgs_mccormick();
    bfgs_beale();
    // Test_Powell.cs
    powell_fxyz();
    powell_de_jong();
    powell_sum_of_power_functions();
    powell_rosenbrock();
    powell_booth();
    powell_matyas();
    powell_mccormick();
    powell_beale();
    bfgs_stationary_start_does_not_search();
    bfgs_maximize_scales_supplied_gradient();
    bfgs_boundary_uses_projected_gradient_and_bounded_hessian();
    bfgs_reports_line_search_and_invalid_gradient_statuses();
    bfgs_preserves_evaluation_budget();
    powell_never_evaluates_outside_bounds();
    powell_handles_degenerate_and_infinite_boxes();
    // Supplement: Tools additions (B5)
    tools_sum_product();
    tools_normalized_distance();
    return chtest::summary("test_optimizers_local");
}
