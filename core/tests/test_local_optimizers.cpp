// Transcribed C# oracle tests for the three remaining local optimizers (P3 Task 1):
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Local/Test_Adam.cs @ 2a0357a
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Local/Test_GradientDescent.cs @ 2a0357a
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Local/Test_GoldenSection.cs @ 2a0357a
//
// All 17 upstream [TestMethod]s (7 + 7 + 3) are transcribed in C# file order with their exact
// fitness + coordinate oracles, their inline per-solver property overrides (several set
// MaxIterations = 100000), and their tolerances (1E-4 throughout, except the McCormick
// coordinates at 1E-2 for both gradient methods and the GradientDescent SumOfPowerFunctions
// coordinates at 1E-2), unaltered. These are internal-support ports validated against the C#
// test oracles themselves, so there is no fixtures/ entry for this file (Task 5/6 adds the
// public-surface fixtures). Skipped upstream methods: none.
//
// The shared objectives come from optimization_test_functions.hpp (the already-ported
// TestFunctions.cs), matching the style of test_optimizers_local.cpp.
//
// SUPPLEMENT (clearly marked, not from any of the three C# test files): direct unit checks for
// the Optimizer base's new public `set_objective_function()` -- the port of C#'s settable
// `ObjectiveFunction` property, added in this task because AugmentedLagrange replaces the
// inner optimizer's objective through it. No upstream C# test exercises the property directly,
// so the two checks below (the replacement takes effect; a null objective throws the C#
// message) are hand-written against the C# property body.
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/math/optimization/adam.hpp"
#include "corehydro/numerics/math/optimization/golden_section.hpp"
#include "corehydro/numerics/math/optimization/gradient_descent.hpp"
#include "check.hpp"
#include "optimization_test_functions.hpp"

using corehydro::numerics::math::optimization::ADAM;
using corehydro::numerics::math::optimization::GoldenSection;
using corehydro::numerics::math::optimization::GradientDescent;

namespace {

// ============================== ADAM (Test_Adam.cs) ==============================

// Test the ADAM algorithm with a simple 3-dimensional test function.
void adam_fxyz() {
    std::vector<double> initial = {0.2, 0.5, 0.5};
    std::vector<double> lower = {0.0, 0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = ADAM(test_functions::fxyz, 3, initial, lower, upper);
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

// Test the ADAM algorithm with the De Jong Function in 5-D.
void adam_de_jong() {
    std::vector<double> initial = {1.0, -1.0, 2.0, -2.0, 1.0};
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    auto solver = ADAM(test_functions::de_jong, 5, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the ADAM algorithm with the Sum of Power functions in 2-D.
void adam_sum_of_power_functions() {
    std::vector<double> initial = {0.5, -0.5};
    std::vector<double> lower = {-1.0, -1.0};
    std::vector<double> upper = {1.0, 1.0};
    auto solver = ADAM(test_functions::sum_of_power_functions, 2, initial, lower, upper);
    solver.max_iterations = 100000;  // Requires a lot of iterations
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the ADAM algorithm with the Rosenbrock Function in 2-D.
void adam_rosenbrock() {
    std::vector<double> initial = {0, 0};
    std::vector<double> lower = {-2.048, -2.048};
    std::vector<double> upper = {2.048, 2.048};
    auto solver = ADAM(test_functions::rosenbrock, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {1.0, 1.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the ADAM algorithm with the Booth Function.
void adam_booth() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = ADAM(test_functions::booth, 2, initial, lower, upper);
    solver.max_iterations = 100000;  // Requires a lot of iterations
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

// Test the ADAM algorithm with the Matyas Function.
void adam_matyas() {
    std::vector<double> initial = {1.0, -1.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = ADAM(test_functions::matyas, 2, initial, lower, upper);
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

// Test the ADAM algorithm with the McCormick Function.
void adam_mccormick() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-1.5, -3.0};
    std::vector<double> upper = {4.0, 4.0};
    auto solver = ADAM(test_functions::mccormick, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = -1.9133;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = -0.54719;
    double validY = -1.54719;
    // Level of precision in parameters is not great
    CHECK_NEAR(x, validX, 1E-2);
    CHECK_NEAR(y, validY, 1E-2);
}

// ==================== GradientDescent (Test_GradientDescent.cs) ====================

// Test the Gradient Descent algorithm with a simple 3-dimensional test function.
void gradient_descent_fxyz() {
    std::vector<double> initial = {0.2, 0.5, 0.5};
    std::vector<double> lower = {0.0, 0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = GradientDescent(test_functions::fxyz, 3, initial, lower, upper);
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

// Test the Gradient Descent algorithm with the De Jong Function in 5-D.
void gradient_descent_de_jong() {
    std::vector<double> initial = {1.0, -1.0, 2.0, -2.0, 1.0};
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    auto solver = GradientDescent(test_functions::de_jong, 5, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Gradient Descent algorithm with the Sum of Power functions in 2-D.
void gradient_descent_sum_of_power_functions() {
    std::vector<double> initial = {0.5, -0.5};
    std::vector<double> lower = {-1.0, -1.0};
    std::vector<double> upper = {1.0, 1.0};
    auto solver = GradientDescent(test_functions::sum_of_power_functions, 2, initial, lower, upper);
    solver.max_iterations = 100000;  // Requires a lot of iterations
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0};
    // Level of precision in parameters is not great
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-2);
}

// Test the Gradient Descent algorithm with the Rosenbrock Function in 2-D.
void gradient_descent_rosenbrock() {
    std::vector<double> initial = {0, 0};
    std::vector<double> lower = {-2.048, -2.048};
    std::vector<double> upper = {2.048, 2.048};
    auto solver = GradientDescent(test_functions::rosenbrock, 2, initial, lower, upper);
    solver.max_iterations = 100000;  // Requires a lot of iterations
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {1.0, 1.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Gradient Descent algorithm with the Booth Function.
void gradient_descent_booth() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = GradientDescent(test_functions::booth, 2, initial, lower, upper);
    solver.max_iterations = 100000;  // Requires a lot of iterations
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

// Test the Gradient Descent algorithm with the Matyas Function.
void gradient_descent_matyas() {
    std::vector<double> initial = {1.0, -1.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = GradientDescent(test_functions::matyas, 2, initial, lower, upper);
    solver.max_iterations = 100000;  // Requires a lot of iterations
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

// Test the Gradient Descent algorithm with the McCormick Function.
void gradient_descent_mccormick() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-1.5, -3.0};
    std::vector<double> upper = {4.0, 4.0};
    auto solver = GradientDescent(test_functions::mccormick, 2, initial, lower, upper);
    solver.max_iterations = 100000;  // Requires a lot of iterations
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = -1.9133;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = -0.54719;
    double validY = -1.54719;
    // Level of precision in parameters is not great
    CHECK_NEAR(x, validX, 1E-2);
    CHECK_NEAR(y, validY, 1E-2);
}

// ==================== GoldenSection (Test_GoldenSection.cs) ====================

// Test to find the minimum of a one dimensional function using the Golden-Section method.
void golden_section_minimize() {
    double lower = -3.0;
    double upper = 3.0;
    auto solver = GoldenSection(test_functions::fx, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    double X = solver.best_parameter_set().values[0];
    double trueX = 1.0;
    CHECK_NEAR(X, trueX, 1E-4);
}

// Test to find the maximum of a one dimensional function using the Golden-Section method.
void golden_section_maximize() {
    double lower = -3.0;
    double upper = 3.0;
    auto solver = GoldenSection(test_functions::fx, lower, upper);
    solver.maximize();
    double F = -1 * solver.best_parameter_set().fitness;
    double trueF = 9.4815;
    CHECK_NEAR(F, trueF, 1E-4);
    double X = solver.best_parameter_set().values[0];
    double trueX = -1.6667;
    CHECK_NEAR(X, trueX, 1E-4);
}

// Test the Golden-Section algorithm with De Jong's function in 1-D.
void golden_section_de_jong() {
    double lower = -5.12;
    double upper = 5.12;
    auto solver = GoldenSection([](double x) { return test_functions::de_jong({x}); }, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    double X = solver.best_parameter_set().values[0];
    double trueX = 0.0;
    CHECK_NEAR(X, trueX, 1E-4);
}

// ============ SUPPLEMENT: Optimizer::set_objective_function (see file header) ============

// Replacing the objective after construction changes what the solver minimizes -- the
// AugmentedLagrange use case this setter exists for.
void objective_setter_replaces_objective() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = GradientDescent(test_functions::matyas, 2, initial, lower, upper);
    // Swap in a shifted quadratic whose minimum is at (2, -3) rather than Matyas' (0, 0).
    solver.set_objective_function([](std::vector<double>& p) {
        return (p[0] - 2.0) * (p[0] - 2.0) + (p[1] + 3.0) * (p[1] + 3.0);
    });
    solver.max_iterations = 100000;
    solver.minimize();
    CHECK_NEAR(solver.best_parameter_set().fitness, 0.0, 1E-4);
    CHECK_NEAR(solver.best_parameter_set().values[0], 2.0, 1E-2);
    CHECK_NEAR(solver.best_parameter_set().values[1], -3.0, 1E-2);
}

// A null objective throws, with the C# property setter's message.
void objective_setter_rejects_null() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = GradientDescent(test_functions::matyas, 2, initial, lower, upper);
    bool threw = false;
    std::string message;
    try {
        solver.set_objective_function(nullptr);
    } catch (const corehydro::numerics::math::optimization::ArgumentException& ex) {
        threw = true;
        message = ex.what();
    }
    CHECK_TRUE(threw);
    CHECK_TRUE(message == "The objective function cannot be null.");
    // The original objective is untouched.
    CHECK_TRUE(static_cast<bool>(solver.objective_function()));
}

}  // namespace

int main() {
    // Test_Adam.cs
    adam_fxyz();
    adam_de_jong();
    adam_sum_of_power_functions();
    adam_rosenbrock();
    adam_booth();
    adam_matyas();
    adam_mccormick();
    // Test_GradientDescent.cs
    gradient_descent_fxyz();
    gradient_descent_de_jong();
    gradient_descent_sum_of_power_functions();
    gradient_descent_rosenbrock();
    gradient_descent_booth();
    gradient_descent_matyas();
    gradient_descent_mccormick();
    // Test_GoldenSection.cs
    golden_section_minimize();
    golden_section_maximize();
    golden_section_de_jong();
    // Supplement: Optimizer::set_objective_function (P3 Task 1)
    objective_setter_replaces_objective();
    objective_setter_rejects_null();
    return chtest::summary("test_local_optimizers");
}
