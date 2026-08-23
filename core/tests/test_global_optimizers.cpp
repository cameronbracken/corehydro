// Transcribed C# oracle tests for the global optimizers (P3 Tasks 2-3):
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Global/Test_ParticleSwarm.cs @ 2a0357a
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Global/Test_ShuffledComplexEvolution.cs @ 2a0357a
//
// Both upstream files carry the SAME 14 [TestMethod] names in the same order (Test_FXYZ,
// Test_DeJong, Test_SumOfPowerFunctions, Test_Rosenbrock, Test_Booth, Test_Matyas,
// Test_McCormick, Test_Beale, Test_GoldsteinPrice, Test_Rastrigin, Test_Ackley,
// Test_ThreeHumpCamel, Test_Eggholder, Test_TP2), so the C++ functions are prefixed pso_ / sce_
// to keep both families in one file without collision. Every bound, inline property override
// (ParticleSwarm's Test_Rosenbrock sets MaxIterations = 100000 AND PopulationSize = 100; SCE's
// Test_Eggholder sets Complexes = 30 and CCEIterations = 10) and tolerance is transcribed
// verbatim from the C# file -- never rounded, never loosened. These are internal-support ports
// validated against the C# test oracles themselves, so there is no fixtures/ entry for this file
// (Task 5 adds the public-surface fixtures).
//
// Skipped upstream methods: none. Two upstream transcription quirks preserved rather than
// "improved":
//   * Test_ShuffledComplexEvolution.Test_Rastrigin has EVERY assertion commented out upstream,
//     above the comment "SCE fail to converge on this test" -- it constructs and runs the solver
//     and asserts nothing. Transcribed as written: the run must complete without throwing, and no
//     value is checked. Inventing an oracle there would be inventing an expected value.
//   * Test_TP2 (both files) asserts a DISJUNCTION over the two symmetric optima
//     (match1 || match2), not two separate equalities.
//
// The shared objectives come from optimization_test_functions.hpp (the already-ported
// TestFunctions.cs), matching the style of test_local_optimizers.cpp.
//
// THIS TARGET IS COMPILED WITH -ffp-contract=off (see core/CMakeLists.txt, which carries the
// measurement). Both algorithms are chaotically sensitive -- their search paths branch on
// comparisons between accumulated sums -- and clang/gcc fuse multiply-adds that .NET never fuses.
// With contraction off, both classes reproduce the real C# library bit-for-bit on every
// configuration measured (iteration count, evaluation count, fitness and coordinates at G17); with
// it on, Test_Eggholder's SCE oracle is missed outright. The oracles below are C# test literals, so
// the C++ has to compute the same arithmetic .NET does for the comparison to mean anything.
#include <cmath>
#include <vector>

#include "corehydro/numerics/math/optimization/particle_swarm.hpp"
#include "corehydro/numerics/math/optimization/shuffled_complex_evolution.hpp"
#include "check.hpp"
#include "optimization_test_functions.hpp"

using corehydro::numerics::math::optimization::ParticleSwarm;
using corehydro::numerics::math::optimization::ShuffledComplexEvolution;

namespace {

// ======================= ParticleSwarm (Test_ParticleSwarm.cs) =======================

// Test the Particle Swarm algorithm with a simple 3-dimensional test function.
void pso_test_fxyz() {
    std::vector<double> lower = {0.0, 0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = ParticleSwarm(test_functions::fxyz, 3, lower, upper);
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

// Test the Particle Swarm algorithm with the De Jong Function in 5-D.
void pso_test_de_jong() {
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    auto solver = ParticleSwarm(test_functions::de_jong, 5, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); ++i) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Particle Swarm algorithm with the Sum of Power functions in 3-D.
void pso_test_sum_of_power_functions() {
    std::vector<double> lower = {-1.0, -1.0, -1.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = ParticleSwarm(test_functions::sum_of_power_functions, 3, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); ++i) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Particle Swarm algorithm with the Rosenbrock Function in 5-D.
void pso_test_rosenbrock() {
    std::vector<double> lower = {-2.048, -2.048, -2.048, -2.048, -2.048};
    std::vector<double> upper = {2.048, 2.048, 2.048, 2.048, 2.048};
    auto solver = ParticleSwarm(test_functions::rosenbrock, 5, lower, upper);
    solver.max_iterations = 100000;
    solver.population_size = 100;
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {1.0, 1.0, 1.0, 1.0, 1.0};
    for (std::size_t i = 0; i < valid.size(); ++i) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Particle Swarm algorithm with the Booth Function.
void pso_test_booth() {
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = ParticleSwarm(test_functions::booth, 2, lower, upper);
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

// Test the Particle Swarm algorithm with the Matyas Function.
void pso_test_matyas() {
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = ParticleSwarm(test_functions::matyas, 2, lower, upper);
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

// Test the Particle Swarm algorithm with the McCormick Function.
void pso_test_mccormick() {
    std::vector<double> lower = {-1.5, -3.0};
    std::vector<double> upper = {4.0, 4.0};
    auto solver = ParticleSwarm(test_functions::mccormick, 2, lower, upper);
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

// Test the Particle Swarm algorithm with the Beale Function.
void pso_test_beale() {
    std::vector<double> lower = {-4.5, -4.5};
    std::vector<double> upper = {4.5, 4.5};
    auto solver = ParticleSwarm(test_functions::beale, 2, lower, upper);
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

// Test the Particle Swarm algorithm with the Goldstein-Price Function.
void pso_test_goldstein_price() {
    std::vector<double> lower = {-2.0, -2.0};
    std::vector<double> upper = {2.0, 2.0};
    auto solver = ParticleSwarm(test_functions::goldstein_price, 2, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 3.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 0.0;
    double validY = -1.0;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the Particle Swarm algorithm with the Rastrigin Function.
void pso_test_rastrigin() {
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    auto solver = ParticleSwarm(test_functions::rastrigin, 5, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); ++i) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Particle Swarm algorithm with the Ackley Function.
void pso_test_ackley() {
    std::vector<double> lower = {-5.0, -5.0};
    std::vector<double> upper = {5.0, 5.0};
    auto solver = ParticleSwarm(test_functions::ackley, 2, lower, upper);
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

// Test the Particle Swarm algorithm with the three hump camel Function.
void pso_test_three_hump_camel() {
    std::vector<double> lower = {-5.0, -5.0};
    std::vector<double> upper = {5.0, 5.0};
    auto solver = ParticleSwarm(test_functions::three_hump_camel, 2, lower, upper);
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

// Test the Particle Swarm algorithm with the Eggholder Function.
void pso_test_eggholder() {
    std::vector<double> lower = {-512.0, -512.0};
    std::vector<double> upper = {512.0, 512.0};
    auto solver = ParticleSwarm(test_functions::eggholder, 2, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = -959.6407;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 512.0;
    double validY = 404.2319;
    CHECK_NEAR(x, validX, 1E-3);
    CHECK_NEAR(y, validY, 1E-3);
}

// Test the Particle Swarm algorithm with the tp2 Function.
void pso_test_tp2() {
    std::vector<double> lower = {0.0, 0.0};
    std::vector<double> upper = {2.0, 2.0};
    auto solver = ParticleSwarm(test_functions::tp2, 2, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 1.0;
    double validY = 0.666667;

    bool match1 = std::fabs(x - validX) < 1E-4 && std::fabs(y - validY) < 1E-4;
    bool match2 = std::fabs(x - validY) < 1E-4 && std::fabs(y - validX) < 1E-4;
    CHECK_TRUE(match1 || match2);
}

// ============ ShuffledComplexEvolution (Test_ShuffledComplexEvolution.cs) ============

// Test the SCE-UA algorithm with a simple 3-dimensional test function.
void sce_test_fxyz() {
    std::vector<double> lower = {0.0, 0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = ShuffledComplexEvolution(test_functions::fxyz, 3, lower, upper);
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

// Test the SCE-UA algorithm with the De Jong Function in 5-D.
void sce_test_de_jong() {
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    auto solver = ShuffledComplexEvolution(test_functions::de_jong, 5, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); ++i) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the SCE-UA algorithm with the Sum of Power functions in 3-D.
void sce_test_sum_of_power_functions() {
    std::vector<double> lower = {-1.0, -1.0, -1.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = ShuffledComplexEvolution(test_functions::sum_of_power_functions, 3, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); ++i) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the SCE-UA algorithm with the Rosenbrock Function in 5-D.
void sce_test_rosenbrock() {
    std::vector<double> lower = {-2.048, -2.048, -2.048, -2.048, -2.048};
    std::vector<double> upper = {2.048, 2.048, 2.048, 2.048, 2.048};
    auto solver = ShuffledComplexEvolution(test_functions::rosenbrock, 5, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {1.0, 1.0, 1.0, 1.0, 1.0};
    for (std::size_t i = 0; i < valid.size(); ++i) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the SCE-UA algorithm with the Booth Function.
void sce_test_booth() {
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = ShuffledComplexEvolution(test_functions::booth, 2, lower, upper);
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

// Test the SCE-UA algorithm with the Matyas Function.
void sce_test_matyas() {
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = ShuffledComplexEvolution(test_functions::matyas, 2, lower, upper);
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

// Test the SCE-UA algorithm with the McCormick Function.
void sce_test_mccormick() {
    std::vector<double> lower = {-1.5, -3.0};
    std::vector<double> upper = {4.0, 4.0};
    auto solver = ShuffledComplexEvolution(test_functions::mccormick, 2, lower, upper);
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

// Test the SCE-UA algorithm with the Beale Function.
void sce_test_beale() {
    std::vector<double> lower = {-4.5, -4.5};
    std::vector<double> upper = {4.5, 4.5};
    auto solver = ShuffledComplexEvolution(test_functions::beale, 2, lower, upper);
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

// Test the SCE-UA algorithm with the Goldstein-Price Function.
void sce_test_goldstein_price() {
    std::vector<double> lower = {-2.0, -2.0};
    std::vector<double> upper = {2.0, 2.0};
    auto solver = ShuffledComplexEvolution(test_functions::goldstein_price, 2, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 3.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 0.0;
    double validY = -1.0;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the SCE-UA algorithm with the Rastrigin Function.
//
// UPSTREAM QUIRK, PRESERVED: every assertion in this C# test method is commented out beneath
// the comment "SCE fail to converge on this test". The method constructs the solver, minimizes,
// reads the fitness, and asserts nothing. Transcribed as written -- the only thing this test
// establishes is that the run completes without throwing, which the harness's own
// exception-free completion covers. No oracle is invented here.
void sce_test_rastrigin() {
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    auto solver = ShuffledComplexEvolution(test_functions::rastrigin, 5, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    (void)F;
    // double trueF = 0.0;
    //
    // SCE fail to converge on this test
    //
    // CHECK_NEAR(F, trueF, 1E-4);
    // auto solution = solver.best_parameter_set().values;
    // std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    // for (std::size_t i = 0; i < valid.size(); ++i) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the SCE-UA algorithm with the Ackley Function.
void sce_test_ackley() {
    std::vector<double> lower = {-5.0, -5.0};
    std::vector<double> upper = {5.0, 5.0};
    auto solver = ShuffledComplexEvolution(test_functions::ackley, 2, lower, upper);
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

// Test the SCE-UA algorithm with the three hump camel Function.
void sce_test_three_hump_camel() {
    std::vector<double> lower = {-5.0, -5.0};
    std::vector<double> upper = {5.0, 5.0};
    auto solver = ShuffledComplexEvolution(test_functions::three_hump_camel, 2, lower, upper);
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

// Test the SCE-UA algorithm with the Eggholder Function.
void sce_test_eggholder() {
    std::vector<double> lower = {-512.0, -512.0};
    std::vector<double> upper = {512.0, 512.0};
    auto solver = ShuffledComplexEvolution(test_functions::eggholder, 2, lower, upper);
    solver.complexes = 30;
    solver.cce_iterations = 10;
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = -959.6407;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 512.0;
    double validY = 404.2319;
    CHECK_NEAR(x, validX, 1E-3);
    CHECK_NEAR(y, validY, 1E-3);
}

// Test the SCE-UA algorithm with the tp2 Function.
void sce_test_tp2() {
    std::vector<double> lower = {0.0, 0.0};
    std::vector<double> upper = {2.0, 2.0};
    auto solver = ShuffledComplexEvolution(test_functions::tp2, 2, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 1.0;
    double validY = 0.666667;

    bool match1 = std::fabs(x - validX) < 1E-4 && std::fabs(y - validY) < 1E-4;
    bool match2 = std::fabs(x - validY) < 1E-4 && std::fabs(y - validX) < 1E-4;
    CHECK_TRUE(match1 || match2);
}

// ===================== SUPPLEMENT (not from any C# test file) =====================
//
// WHY THIS EXISTS. shuffled_complex_evolution.hpp's transcription hazard 4 reproduces an upstream
// array-aliasing effect (EvolveComplex's scratch point `p` becomes the sub-complex entry's Values
// array, so a later write silently moves that entry). It changes the search PATH, not the optimum,
// and every one of the 14 transcribed SCE methods above passes with the aliasing removed --
// measured, by deleting the three `values_alias_p = true` assignments and re-running. So the
// transcriptions guard the answer but not the quirk, and a future reader who "cleans up" the flag
// would get a green suite and a different algorithm.
//
// WHAT IT ASSERTS. The iteration and function-evaluation counts of two seeded SCE runs, taken from
// the REAL Numerics library at 2a0357a (driven by a scratch dotnet console app -- the same oracle
// class tools/verify_oracles.py uses, not an invented value):
//     5-D over [-5.12, 5.12]: Iterations 57, FunctionEvaluations 9483
//     2-D over [-10, 10]:     Iterations 52, FunctionEvaluations 5003
// Removing the aliasing moves the 5-D run to 8744 evaluations and the 2-D run off its count too,
// which is what makes these two integers a real guard.
//
// WHY THESE COUNTS ARE PORTABLE. The objective below is written as explicit multiply-and-add with
// no std::pow and no transcendental, and everything else on the path (MersenneTwister,
// LatinHypercube, the trapezoidal CDF, the reflection/contraction arithmetic) is add / multiply /
// divide / compare only. With contraction off -- which this target enforces -- every one of those
// is exactly specified by IEEE 754, so the whole run is bit-identical on any conforming toolchain.
// A libm-dependent objective (Eggholder's sin, De Jong's std::pow) would NOT be safe to pin this
// way, which is why the catalog's functions are not reused here.
double sphere(const std::vector<double>& x) {
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) sum += x[i] * x[i];
    return sum;
}

void sce_aliasing_reproduces_csharp_evaluation_counts() {
    {
        std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
        std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
        auto solver = ShuffledComplexEvolution(sphere, 5, lower, upper);
        solver.minimize();
        CHECK_EQ(solver.iterations(), 57);
        CHECK_EQ(solver.function_evaluations(), 9483);
    }
    {
        std::vector<double> lower = {-10.0, -10.0};
        std::vector<double> upper = {10.0, 10.0};
        auto solver = ShuffledComplexEvolution(sphere, 2, lower, upper);
        solver.minimize();
        CHECK_EQ(solver.iterations(), 52);
        CHECK_EQ(solver.function_evaluations(), 5003);
    }
}

}  // namespace

int main() {
    // Test_ParticleSwarm.cs
    pso_test_fxyz();
    pso_test_de_jong();
    pso_test_sum_of_power_functions();
    pso_test_rosenbrock();
    pso_test_booth();
    pso_test_matyas();
    pso_test_mccormick();
    pso_test_beale();
    pso_test_goldstein_price();
    pso_test_rastrigin();
    pso_test_ackley();
    pso_test_three_hump_camel();
    pso_test_eggholder();
    pso_test_tp2();
    // Test_ShuffledComplexEvolution.cs
    sce_test_fxyz();
    sce_test_de_jong();
    sce_test_sum_of_power_functions();
    sce_test_rosenbrock();
    sce_test_booth();
    sce_test_matyas();
    sce_test_mccormick();
    sce_test_beale();
    sce_test_goldstein_price();
    sce_test_rastrigin();
    sce_test_ackley();
    sce_test_three_hump_camel();
    sce_test_eggholder();
    sce_test_tp2();
    // Supplement: the upstream aliasing quirk (transcription hazard 4)
    sce_aliasing_reproduces_csharp_evaluation_counts();
    return chtest::summary("test_global_optimizers");
}
