// ported from: Numerics/Test_Numerics/Mathematics/Optimization/TestFunctions.cs @ a2c4dbf
//
// Shared analytic objective functions for the optimizer ctests (B5 BFGS; B6 reuses this
// header for Powell and MLSL). ALL 16 functions from the C# file are ported verbatim --
// including FX (1-D, used by the upstream golden-section/Brent tests) and Bukin (unused by
// the upstream BFGS/Powell/MLSL tests) -- so later optimizer tasks never have to touch this
// file. Formulas are transcribed exactly; Math.PI -> corehydro::numerics::kPi and
// Math.E -> corehydro::numerics::kE per the repo's no-M_PI portability rule.
//
// Signatures take `const std::vector<double>&` (the C# `double[]`), which converts
// implicitly to the Optimizer base's mutable-reference Objective std::function.
#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

#include "corehydro/numerics/tools.hpp"

namespace test_functions {

// A simple one-dimensional test function. Test area restricted to [-3, 3].
// Local minimum: f(1) = 0. Local maximum: f(-1.6667) = 9.4815.
inline double fx(double x) {
    double F = (x + 3.0) * std::pow(x - 1.0, 2.0);
    return F;
}

// A simple 3-dimensional test function. Global minimum: f(0.125, 0.2, 0.35) = 0.
inline double fxyz(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    double z = parms[2];
    double F = std::pow(4.0 * x - 0.5, 2.0) + std::pow(3.0 * y - 0.6, 2.0) +
               std::pow(2.0 * z - 0.7, 2.0);
    return F;
}

// The De Jong (or sphere) function. Test area usually restricted to hypercube
// [-5.12, 5.12]. Global minimum: f(0, 0) = 0.
inline double de_jong(const std::vector<double>& x) {
    std::size_t n = x.size();
    double F = 0;
    for (std::size_t i = 0; i < n; i++) F += std::pow(x[i], 2);
    return F;
}

// The Sum of Different Power Functions. Test area usually restricted to hypercube
// [-1, 1]. Global minimum: f(0, 0) = 0.
inline double sum_of_power_functions(const std::vector<double>& x) {
    std::size_t n = x.size();
    double F = 0;
    for (std::size_t i = 0; i < n; i++)
        F += std::pow(std::fabs(x[i]), static_cast<double>(i) + 2.0);
    return F;
}

// The Rosenbrock Function. Test area usually restricted to hypercube [-2.048, 2.048].
// Global minimum: f(1, 1) = 0.
inline double rosenbrock(const std::vector<double>& parms) {
    std::size_t n = parms.size();
    double F = 0;
    for (std::size_t i = 0; i + 1 < n; i++)
        F += 100 * std::pow(parms[i + 1] - parms[i] * parms[i], 2) + std::pow(1 - parms[i], 2);
    return F;
}

// The Booth Function. Global minimum: f(1, 3) = 0.
inline double booth(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    double F = std::pow(x + 2 * y - 7, 2) + std::pow(2 * x + y - 5, 2);
    return F;
}

// The Matyas Function. Global minimum: f(0, 0) = 0.
inline double matyas(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    double F = 0.26 * (x * x + y * y) - 0.48 * x * y;
    return F;
}

// The McCormick Function. Global minimum: f(-0.54719, -1.54719) = -1.9133.
inline double mccormick(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    double F = std::sin(x + y) + std::pow(x - y, 2) - 1.5 * x + 2.5 * y + 1.0;
    return F;
}

// The Rastrigin Function. Global minimum: f(0, 0) = 0.
inline double rastrigin(const std::vector<double>& parms) {
    double A = 10;
    std::size_t n = parms.size();
    double F = A * static_cast<double>(n);
    for (std::size_t i = 0; i < n; i++)
        F += parms[i] * parms[i] - A * std::cos(2 * corehydro::numerics::kPi * parms[i]);
    return F;
}

// The Ackley Function. Global minimum: f(0, 0) = 0.
inline double ackley(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    double F = -20 * std::exp(-0.2 * std::sqrt(0.5 * (x * x + y * y))) -
               std::exp(0.5 * (std::cos(2 * corehydro::numerics::kPi * x) +
                               std::cos(2 * corehydro::numerics::kPi * y))) +
               corehydro::numerics::kE + 20;
    return F;
}

// The Beale Function. Global minimum: f(3, 0.5) = 0.
inline double beale(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    double F = std::pow(1.5 - x + x * y, 2) + std::pow(2.25 - x + x * y * y, 2) +
               std::pow(2.625 - x + x * y * y * y, 2);
    return F;
}

// The Goldstein-Price Function. Global minimum: f(0, -1) = 3.
inline double goldstein_price(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    double F = (1 + std::pow(x + y + 1, 2) *
                        (19 - 14 * x + 3 * x * x - 14 * y + 6 * x * y + 3 * y * y)) *
               (30 + std::pow(2 * x - 3 * y, 2) *
                         (18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y));
    return F;
}

// The Bukin Function. Global minimum: f(-10, 1) = 0.
inline double bukin(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    double F = 100 * std::sqrt(std::fabs(y - 0.01 * x * x)) + 0.01 * std::fabs(x + 10);
    return F;
}

// The three hump camel Function. Global minimum: f(0, 0) = 0.
inline double three_hump_camel(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    double F = 2 * std::pow(x, 2) - 1.05 * std::pow(x, 4) + std::pow(x, 6) / 6 + x * y +
               std::pow(y, 2);
    return F;
}

// The Eggholder Function. Global minimum: f(512, 404.2319) = -959.6407.
inline double eggholder(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    double F = -(y + 47) * std::sin(std::sqrt(std::fabs((x / 2) + (y + 47)))) -
               x * std::sin(std::sqrt(std::fabs(x - (y + 47))));
    return F;
}

// tp2 function - Multiple local optima and 2 global minimizers, by virtue of symmetry.
// Global minimum: f(2/3, 1) = f(1, 2/3) = 0.
inline double tp2(const std::vector<double>& parms) {
    double p1 = parms[0];
    double p2 = parms[1];
    int i;
    double x, F = 0;

    for (i = 0; i <= 86; i++) {
        x = 3.1 + 0.15 * i;
        F = F + std::pow(((std::sin(p1 * x) + std::sin(p2 * x)) -
                          (std::sin((2.0 / 3.0) * x) + std::sin(1 * x))),
                         2);
    }
    return F;
}

// --- corehydro ADDITION -- no upstream C# counterpart --------------------------------------
//
// TestFunctions.cs carries no gradients: every upstream ADAM / GradientDescent test runs those
// classes' finite-difference fallback. These three are the hand-differentiated gradients of
// `fxyz`, `de_jong` and `booth` above, and exist so the optimizer fixtures can exercise the
// runner's optional analytic-gradient callback (a second host-language function crossing into the
// core). Each is written out term by term -- no loop over a shared helper, no autodiff -- so the
// R, Python and C# fixture catalogs reproduce the identical arithmetic in the identical order.
inline std::vector<double> grad_fxyz(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    double z = parms[2];
    return {8.0 * (4.0 * x - 0.5), 6.0 * (3.0 * y - 0.6), 4.0 * (2.0 * z - 0.7)};
}

inline std::vector<double> grad_de_jong(const std::vector<double>& x) {
    std::size_t n = x.size();
    std::vector<double> g(n);
    for (std::size_t i = 0; i < n; i++) g[i] = 2.0 * x[i];
    return g;
}

inline std::vector<double> grad_booth(const std::vector<double>& parms) {
    double x = parms[0];
    double y = parms[1];
    return {2.0 * (x + 2.0 * y - 7.0) + 4.0 * (2.0 * x + y - 5.0),
            4.0 * (x + 2.0 * y - 7.0) + 2.0 * (2.0 * x + y - 5.0)};
}

// --- Test_AugmentedLagrange.cs's own inline objectives and constraint functions ---------------
//
// Unlike every function above, these do NOT come from TestFunctions.cs: the constrained tests
// define their objective and their constraint inline in each [TestMethod] body, so each is
// transcribed here term for term in the C# expression order, and the two accumulating ones spell
// their sum out as an explicit loop (the C# `Tools.Sum` is itself a plain `for` loop over
// `sum += values[i]`, verified at 2a0357a). The optimizer fixture catalog names them, so the R,
// Python and C# fixture catalogs carry the same four transcriptions -- an objective and a
// constraint are the same `double(const std::vector<double>&)` shape, so both live in one catalog
// rather than two.
//
// Test_1: maximize the net benefit of allocating a fixed supply across three periods.
inline double al1_objective(const std::vector<double>& x) {
    std::vector<double> NB(3);
    for (int i = 0; i < 3; i++) {
        NB[i] = (20 * x[i] - x[i] * x[i] - 24) / std::pow(1.10, i);
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < NB.size(); i++) sum += NB[i];
    return -sum;
}

// Test_2: the same, over two users with different benefit functions.
inline double al2_objective(const std::vector<double>& x) {
    std::vector<double> NB(2);
    NB[0] = 60 * x[0] - 0.5 * x[0] * x[0];
    NB[1] = (64 * x[1] - 0.5 * x[1] * x[1]) / 1.5;
    double sum = 0.0;
    for (std::size_t i = 0; i < NB.size(); i++) sum += NB[i];
    return -sum;
}

// Test_1's and Test_2's equality constraint: Tools.Sum(x).
inline double sum_all(const std::vector<double>& x) {
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); i++) sum += x[i];
    return sum;
}

// Test_Haimes_5_2, example problem 5.2 from "Risk Modeling, Assessment, and Management".
inline double haimes_primary(const std::vector<double>& x) {
    return std::pow(x[0] - 2, 2) + std::pow(x[1] - 4, 2) + 5;
}

inline double haimes_secondary(const std::vector<double>& x) {
    return std::pow(x[0] - 6, 2) + std::pow(x[1] - 10, 2) + 6;
}

// Test_RosenbrockDisk's objective. NOT the same expression as `rosenbrock` above: the C# test
// writes the two-dimensional case out by hand, `(1 - x)^2` FIRST and the 100-weighted term second,
// where TestFunctions.Rosenbrock loops with the two terms the other way round.
inline double rosenbrock_disk_objective(const std::vector<double>& x) {
    return std::pow(1 - x[0], 2) + 100 * std::pow(x[1] - x[0] * x[0], 2);
}

// Test_RosenbrockDisk's constraint (x^2 + y^2 <= 2), and Test_MixedConstraints's objective, which
// upstream writes as the identical expression.
inline double disk(const std::vector<double>& x) { return (x[0] * x[0]) + (x[1] * x[1]); }

// Test_MixedConstraints's three constraint functions.
inline double sum_xy(const std::vector<double>& x) { return x[0] + x[1]; }
inline double x0(const std::vector<double>& x) { return x[0]; }
inline double x1(const std::vector<double>& x) { return x[1]; }

}  // namespace test_functions
