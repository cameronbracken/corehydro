// Transcribed C# oracle tests for the constrained optimizer (P3 Task 4):
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Constrained/Test_AugmentedLagrange.cs
//   @ 2a0357a
//
// All 6 upstream [TestMethod]s are transcribed in C# file order with their exact point,
// fitness and multiplier oracles and their tolerances (1E-3 for Test_1/Test_2, 1E-2 for
// Test_Haimes_5_2, 1E-4 for Test_RosenbrockDisk plus its EXACT-equality Mu[0], and 0.1/0.5 for
// the two mixed-constraint tests), unaltered. Two transcription details worth naming:
//
//   * Test_1 and Test_2 assert `-solver.BestParameterSet.Fitness` (their objectives return a
//     negated net benefit), while Test_Haimes_5_2 and Test_RosenbrockDisk assert the fitness
//     unnegated. Each is transcribed exactly as written.
//   * Test_1's and Test_2's BFGS bounds are `double.MinValue` / `double.MaxValue`. C#'s
//     `double.MinValue` IS the most negative finite double, so it ports as
//     `std::numeric_limits<double>::lowest()`, NOT `::min()`.
//
// Each test defines its objective and constraint functions inline in the C#; they are
// transcribed here as file-local statics with explicit loops (no accumulating helper), so the
// arithmetic matches the C# expression order term for term. These are internal-support ports
// validated against the C# test oracles themselves, so there is no fixtures/ entry for this
// file (Task 7 adds the public-surface fixtures). Skipped upstream methods: none. A clearly
// marked SUPPLEMENT at the bottom covers the constructor contract none of the six reach.
//
// MEASURED against the REAL C# Numerics library at 2a0357a (a scratch dotnet console app
// running the six tests verbatim, printing at G17): all six reproduce C#'s iteration count,
// function-evaluation count and status EXACTLY, and every point, fitness and multiplier agrees
// to at worst ~2e-9 as this suite is compiled (Apple clang 21, arm64). Compiled with
// `-ffp-contract=off` the port is BIT-IDENTICAL to C# on all six at every digit, so the whole
// residual is fused-multiply-add contraction inside the inner BFGS, not a port divergence.
// Unlike test_global_optimizers, this suite does NOT carry that flag: the C# tolerances
// transcribed below (1E-4 at the tightest) sit five orders of magnitude above the contracted
// drift, so nothing here needs de-contracted arithmetic to mean what it says. The most
// amplified quantity measured is Test_MixedConstraints_Binding's mu[0] (C# 3.8484910806747981
// against 3.8516998788154417 contracted, 3.2e-3), which no C# test asserts.
#include <cmath>
#include <limits>
#include <vector>

#include "corehydro/numerics/math/optimization/augmented_lagrange.hpp"
#include "corehydro/numerics/math/optimization/bfgs.hpp"
#include "check.hpp"

using corehydro::numerics::math::optimization::AugmentedLagrange;
using corehydro::numerics::math::optimization::BFGS;
using corehydro::numerics::math::optimization::Constraint;
using corehydro::numerics::math::optimization::ConstraintType;
using corehydro::numerics::math::optimization::IConstraint;

namespace {

// ===================== Test_AugmentedLagrange.cs, in file order =====================

// Test_1: Solution is from a Water Economics graduate course.
double test_1_constraint(const std::vector<double>& x) {
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); i++) sum += x[i];
    return sum;
}

double test_1_objective(const std::vector<double>& x) {
    std::vector<double> NB(3);
    for (int i = 0; i < 3; i++) {
        NB[i] = (20 * x[i] - x[i] * x[i] - 24) / std::pow(1.10, i);
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < NB.size(); i++) sum += NB[i];
    return -sum;
}

void augmented_lagrange_test_1() {
    auto constraint = std::make_shared<Constraint>(test_1_constraint, 3, 22, ConstraintType::EqualTo);
    std::vector<double> initial = {7, 7, 8};
    std::vector<double> lower = {std::numeric_limits<double>::lowest(),
                                std::numeric_limits<double>::lowest(),
                                std::numeric_limits<double>::lowest()};
    std::vector<double> upper = {std::numeric_limits<double>::max(),
                                std::numeric_limits<double>::max(),
                                std::numeric_limits<double>::max()};
    auto inner_solver = BFGS(test_1_objective, 3, initial, lower, upper);
    auto solver = AugmentedLagrange(test_1_objective, inner_solver,
                                    std::vector<std::shared_ptr<IConstraint>>{constraint});
    solver.minimize();

    // Point
    CHECK_NEAR(solver.best_parameter_set().values[0], 7.583082, 1E-3);
    CHECK_NEAR(solver.best_parameter_set().values[1], 7.341390, 1E-3);
    CHECK_NEAR(solver.best_parameter_set().values[2], 7.075529, 1E-3);
    // Function
    CHECK_NEAR(-solver.best_parameter_set().fitness, 188.5655, 1E-3);
    // Multiplier
    CHECK_NEAR(solver.lambda()[0], 4.833835, 1E-3);
}

// Test_2: Solution is from a Water Economics graduate course.
double test_2_constraint(const std::vector<double>& x) {
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); i++) sum += x[i];
    return sum;
}

double test_2_objective(const std::vector<double>& x) {
    std::vector<double> NB(2);
    NB[0] = 60 * x[0] - 0.5 * x[0] * x[0];
    NB[1] = (64 * x[1] - 0.5 * x[1] * x[1]) / 1.5;
    double sum = 0.0;
    for (std::size_t i = 0; i < NB.size(); i++) sum += NB[i];
    return -sum;
}

void augmented_lagrange_test_2() {
    auto constraint = std::make_shared<Constraint>(test_2_constraint, 2, 100, ConstraintType::EqualTo);
    std::vector<double> initial = {50, 50};
    std::vector<double> lower = {std::numeric_limits<double>::lowest(),
                                std::numeric_limits<double>::lowest()};
    std::vector<double> upper = {std::numeric_limits<double>::max(),
                                std::numeric_limits<double>::max()};
    auto inner_solver = BFGS(test_2_objective, 2, initial, lower, upper);
    auto solver = AugmentedLagrange(test_2_objective, inner_solver,
                                    std::vector<std::shared_ptr<IConstraint>>{constraint});
    solver.minimize();

    // Point
    CHECK_NEAR(solver.best_parameter_set().values[0], 50.4, 1E-3);
    CHECK_NEAR(solver.best_parameter_set().values[1], 49.6, 1E-3);
    // Function
    CHECK_NEAR(-solver.best_parameter_set().fitness, 3050.133, 1E-3);
    // Multiplier
    CHECK_NEAR(solver.lambda()[0], 9.6, 1E-3);
}

// Test the Lagrange Augmented Algorithm with example problem 5.2 from "Risk Modeling,
// Assessment, and Management".
double haimes_primary(const std::vector<double>& x) {
    return std::pow(x[0] - 2, 2) + std::pow(x[1] - 4, 2) + 5;
}

double haimes_secondary(const std::vector<double>& x) {
    return std::pow(x[0] - 6, 2) + std::pow(x[1] - 10, 2) + 6;
}

void augmented_lagrange_haimes_5_2() {
    // Set up inner solver
    std::vector<double> initial = {5, 5};
    std::vector<double> lower = {0, 0};
    std::vector<double> upper = {10, 10};
    auto inner_solver = BFGS(haimes_primary, 2, initial, lower, upper);
    // Set up constraint
    auto constraint =
        std::make_shared<Constraint>(haimes_secondary, 2, 13.31, ConstraintType::LesserThanOrEqualTo);
    // Solve
    auto solver = AugmentedLagrange(haimes_primary, inner_solver,
                                    std::vector<std::shared_ptr<IConstraint>>{constraint});
    solver.minimize();

    // Point
    CHECK_NEAR(solver.best_parameter_set().values[0], 4.5, 1E-2);
    CHECK_NEAR(solver.best_parameter_set().values[1], 7.75, 1E-2);
    // Function
    CHECK_NEAR(solver.best_parameter_set().fitness, 25.31, 1E-2);
    // Multiplier
    CHECK_NEAR(solver.mu()[0], 1.67, 1E-2);
}

// Test the Augmented Lagrange algorithm on the Rosenbrock Function constrained to a disk.
double rosenbrock_disk_constraint(const std::vector<double>& x) {
    return (x[0] * x[0]) + (x[1] * x[1]);
}

double rosenbrock_disk_objective(const std::vector<double>& x) {
    return std::pow(1 - x[0], 2) + 100 * std::pow(x[1] - x[0] * x[0], 2);
}

void augmented_lagrange_rosenbrock_disk() {
    // Set up inner solver
    std::vector<double> initial = {0, 0};
    std::vector<double> lower = {-1.5, -1.5};
    std::vector<double> upper = {1.5, 1.5};
    auto inner_solver = BFGS(rosenbrock_disk_objective, 2, initial, lower, upper);
    // Set up constraint
    auto constraint = std::make_shared<Constraint>(rosenbrock_disk_constraint, 2, 2,
                                                  ConstraintType::LesserThanOrEqualTo);
    // Solve
    auto solver = AugmentedLagrange(rosenbrock_disk_objective, inner_solver,
                                    std::vector<std::shared_ptr<IConstraint>>{constraint});
    solver.minimize();

    // Point
    CHECK_NEAR(solver.best_parameter_set().values[0], 1.0, 1E-4);
    CHECK_NEAR(solver.best_parameter_set().values[1], 1.0, 1E-4);
    // Function
    CHECK_NEAR(solver.best_parameter_set().fitness, 0.0, 1E-4);
    // Multiplier (asserted at EXACT equality in the C#)
    CHECK_EQ(solver.mu()[0], 0.0);
}

// Tests AugmentedLagrange with mixed constraint types (equality + lesser-than + greater-than).
// This previously caused IndexOutOfRangeException due to incorrect multiplier array indexing.
double mixed_objective(const std::vector<double>& x) { return x[0] * x[0] + x[1] * x[1]; }

double mixed_equality_constraint(const std::vector<double>& x) { return x[0] + x[1]; }

double mixed_x0_constraint(const std::vector<double>& x) { return x[0]; }

double mixed_x1_constraint(const std::vector<double>& x) { return x[1]; }

void augmented_lagrange_mixed_constraints() {
    // Constraints
    auto equality_constraint =
        std::make_shared<Constraint>(mixed_equality_constraint, 2, 4.0, ConstraintType::EqualTo);
    auto less_than_constraint =
        std::make_shared<Constraint>(mixed_x0_constraint, 2, 3.0, ConstraintType::LesserThanOrEqualTo);
    auto greater_than_constraint = std::make_shared<Constraint>(
        mixed_x1_constraint, 2, 0.5, ConstraintType::GreaterThanOrEqualTo);

    // Inner solver
    std::vector<double> initial = {1, 3};
    std::vector<double> lower = {-10, -10};
    std::vector<double> upper = {10, 10};
    auto inner_solver = BFGS(mixed_objective, 2, initial, lower, upper);

    // Solve with all three constraint types
    std::vector<std::shared_ptr<IConstraint>> constraints = {equality_constraint, less_than_constraint,
                                                             greater_than_constraint};
    auto solver = AugmentedLagrange(mixed_objective, inner_solver, constraints);
    solver.minimize();

    // Solution should be (2, 2)
    CHECK_NEAR(solver.best_parameter_set().values[0], 2.0, 0.1);
    CHECK_NEAR(solver.best_parameter_set().values[1], 2.0, 0.1);
    // Objective = 4+4 = 8
    CHECK_NEAR(solver.best_parameter_set().fitness, 8.0, 0.5);
}

// Tests AugmentedLagrange with mixed constraints where the inequality constraints are binding.
double mixed_binding_objective(const std::vector<double>& x) {
    return std::pow(x[0] - 5, 2) + std::pow(x[1] - 5, 2);
}

void augmented_lagrange_mixed_constraints_binding() {
    auto equality_constraint =
        std::make_shared<Constraint>(mixed_equality_constraint, 2, 4.0, ConstraintType::EqualTo);
    auto less_than_constraint =
        std::make_shared<Constraint>(mixed_x0_constraint, 2, 1.0, ConstraintType::LesserThanOrEqualTo);
    auto greater_than_constraint = std::make_shared<Constraint>(
        mixed_x1_constraint, 2, 2.0, ConstraintType::GreaterThanOrEqualTo);

    std::vector<double> initial = {0.5, 3.5};
    std::vector<double> lower = {-10, -10};
    std::vector<double> upper = {10, 10};
    auto inner_solver = BFGS(mixed_binding_objective, 2, initial, lower, upper);

    std::vector<std::shared_ptr<IConstraint>> constraints = {equality_constraint, less_than_constraint,
                                                             greater_than_constraint};
    auto solver = AugmentedLagrange(mixed_binding_objective, inner_solver, constraints);
    solver.minimize();

    // Solution should be (1, 3)
    CHECK_NEAR(solver.best_parameter_set().values[0], 1.0, 0.1);
    CHECK_NEAR(solver.best_parameter_set().values[1], 3.0, 0.1);
}

// ===================== SUPPLEMENT (not from Test_AugmentedLagrange.cs) =====================
//
// Every one of the six C# tests uses AT MOST ONE constraint of each type and asserts only the
// converged answer, so three parts of the constructor's contract are unguarded by them: the
// COUNTING that sizes the three multiplier vectors (a shared index would go undetected while
// each vector has length one), the REPLACEMENT of the inner optimizer's objective with the
// augmented Lagrangian, and the two constructor rejections. The values below are not derived
// here -- they were read off the REAL C# Numerics library at 2a0357a driven by a scratch
// console app, which reported: `lambda=2 mu=2 nu=1` for the five-constraint mix;
// `inner.ObjectiveFunction` = 25 at (5, 0) BEFORE construction and 27.625 after, 8 at (2, 2)
// and 8.5 at (2.5, 1.5); `ArgumentOutOfRangeException` "There must be at least one constraint."
// for an empty list; and an `ArgumentException` for a nested AugmentedLagrange.
//
// The three augmented-Lagrangian probes are exact in binary floating point (sums and products
// of dyadic rationals with the zero-initialized multipliers and rho = 1), so they are safe to
// assert across toolchains, unlike an iteration count off a BFGS run.
//
// Honest coverage gap, measured rather than papered over: disabling the `if (ICM > tau *
// prevICM) rho *= gam` penalty escalation leaves all 22 transcribed checks green. The branch IS
// reached -- with it disabled, Test_MixedConstraints walks 28 outer iterations instead of 13 and
// Test_MixedConstraints_Binding 185 instead of 15 -- but the C# tolerances there (0.1 on the
// point, 0.5 on the fitness) are loose enough that the answer still lands. Pinning those counts
// would pin a BFGS path over pow()/sqrt() across three CI platforms, which this repo does not
// do, so the measurement is recorded here instead of being turned into a brittle assertion.
double supplement_objective(const std::vector<double>& x) { return x[0] * x[0] + x[1] * x[1]; }

double supplement_x0_minus_x1(const std::vector<double>& x) { return x[0] - x[1]; }

void augmented_lagrange_sizes_multipliers_by_type() {
    auto c1 = std::make_shared<Constraint>(mixed_equality_constraint, 2, 4.0, ConstraintType::EqualTo);
    auto c2 = std::make_shared<Constraint>(mixed_x0_constraint, 2, 3.0, ConstraintType::LesserThanOrEqualTo);
    auto c3 = std::make_shared<Constraint>(mixed_x1_constraint, 2, 0.5, ConstraintType::GreaterThanOrEqualTo);
    auto c4 = std::make_shared<Constraint>(supplement_x0_minus_x1, 2, 0.0, ConstraintType::EqualTo);
    auto c5 = std::make_shared<Constraint>(mixed_x1_constraint, 2, 9.0, ConstraintType::LesserThanOrEqualTo);
    std::vector<double> initial = {1, 3};
    std::vector<double> lower = {-10, -10};
    std::vector<double> upper = {10, 10};
    auto inner_solver = BFGS(supplement_objective, 2, initial, lower, upper);
    std::vector<std::shared_ptr<IConstraint>> constraints = {c1, c2, c3, c4, c5};
    auto solver = AugmentedLagrange(supplement_objective, inner_solver, constraints);

    CHECK_EQ(solver.lambda().size(), std::size_t{2});
    CHECK_EQ(solver.mu().size(), std::size_t{2});
    CHECK_EQ(solver.nu().size(), std::size_t{1});
    CHECK_EQ(solver.constraints().size(), std::size_t{5});
    CHECK_EQ(solver.number_of_parameters(), 2);
    CHECK_TRUE(&solver.optimizer() == &inner_solver);
    // The constraint bundle is stored, not copied apart.
    CHECK_EQ(solver.constraints()[3]->value(), 0.0);
    CHECK_TRUE(solver.constraints()[3]->type() == ConstraintType::EqualTo);
    CHECK_EQ(solver.constraints()[0]->number_of_parameters(), 2);
    CHECK_EQ(solver.constraints()[0]->tolerance(), 1E-8);
}

void augmented_lagrange_replaces_inner_objective() {
    auto c1 = std::make_shared<Constraint>(mixed_equality_constraint, 2, 4.0, ConstraintType::EqualTo);
    auto c2 = std::make_shared<Constraint>(mixed_x0_constraint, 2, 3.0, ConstraintType::LesserThanOrEqualTo);
    auto c3 = std::make_shared<Constraint>(mixed_x1_constraint, 2, 0.5, ConstraintType::GreaterThanOrEqualTo);
    std::vector<double> initial = {1, 3};
    std::vector<double> lower = {-10, -10};
    std::vector<double> upper = {10, 10};
    auto inner_solver = BFGS(supplement_objective, 2, initial, lower, upper);

    std::vector<double> probe = {5, 0};
    // Before construction the inner optimizer still evaluates its own objective.
    CHECK_EQ(inner_solver.objective_function()(probe), 25.0);

    std::vector<std::shared_ptr<IConstraint>> constraints = {c1, c2, c3};
    auto solver = AugmentedLagrange(supplement_objective, inner_solver, constraints);

    // After construction it evaluates the augmented Lagrangian: 25 + 0.5*(5+0-4)^2 (equality,
    // always penalized) + 0.5*(5-3)^2 (lesser-than, violated) + 0.5*(0.5-0)^2 (greater-than,
    // violated) = 27.625.
    CHECK_EQ(inner_solver.objective_function()(probe), 27.625);
    // A feasible point pays no penalty at all: only the equality term can contribute, and it is
    // exactly satisfied here.
    std::vector<double> feasible = {2, 2};
    CHECK_EQ(inner_solver.objective_function()(feasible), 8.0);
    // Both inequalities slack, equality satisfied.
    std::vector<double> half = {2.5, 1.5};
    CHECK_EQ(inner_solver.objective_function()(half), 8.5);
    // None of those evaluations went through this optimizer's own evaluate() (see the header's
    // transcription note 2).
    CHECK_EQ(solver.function_evaluations(), 0);
}

void augmented_lagrange_constructor_rejections() {
    std::vector<double> initial = {1, 3};
    std::vector<double> lower = {-10, -10};
    std::vector<double> upper = {10, 10};
    auto inner_solver = BFGS(supplement_objective, 2, initial, lower, upper);

    // An empty constraint list (C#: ArgumentOutOfRangeException, same message).
    CHECK_THROWS_MSG(AugmentedLagrange(supplement_objective, inner_solver,
                                       std::vector<std::shared_ptr<IConstraint>>{}),
                     "There must be at least one constraint.");

    // A nested AugmentedLagrange inner optimizer. C# reports this through an ArgumentException
    // whose Message is the string "optimizer" and whose ParamName is the sentence -- upstream
    // has the two-argument overload's parameters swapped -- so the port uses the sentence, which
    // is what the swap was clearly meant to say.
    auto c1 = std::make_shared<Constraint>(mixed_equality_constraint, 2, 4.0, ConstraintType::EqualTo);
    auto outer = AugmentedLagrange(supplement_objective, inner_solver,
                                   std::vector<std::shared_ptr<IConstraint>>{c1});
    CHECK_THROWS_MSG(AugmentedLagrange(supplement_objective, outer,
                                       std::vector<std::shared_ptr<IConstraint>>{c1}),
                     "The inner optimizer cannot also be an Augmented Lagrange optimizer.");
}

}  // namespace

int main() {
    augmented_lagrange_test_1();
    augmented_lagrange_test_2();
    augmented_lagrange_haimes_5_2();
    augmented_lagrange_rosenbrock_disk();
    augmented_lagrange_mixed_constraints();
    augmented_lagrange_mixed_constraints_binding();
    // Supplement (see the block comment above): the constructor contract the six C# tests do
    // not reach.
    augmented_lagrange_sizes_multipliers_by_type();
    augmented_lagrange_replaces_inner_objective();
    augmented_lagrange_constructor_rejections();
    return chtest::summary("test_augmented_lagrange");
}
