// Standalone tests for the Bisection, NewtonRaphson, and Secant root-finding routines.
//
// Transcribed 1:1 (structure, inputs and oracle values unaltered) from
//   upstream/Numerics/Test_Numerics/Mathematics/Root Finding/Test_Bisection.cs @ 2a0357a
//   upstream/Numerics/Test_Numerics/Mathematics/Root Finding/Test_NewtonRaphson.cs @ 2a0357a
//   upstream/Numerics/Test_Numerics/Mathematics/Root Finding/Test_Secant.cs @ 2a0357a
// -- all 25 [TestMethod]s (8 + 13 + 4) in C# file order. The shared test fixtures live in
//   upstream/Numerics/Test_Numerics/Mathematics/Root Finding/TestFunctions.cs @ 2a0357a
// and are transcribed below as file-local static functions, matching the style of the other
// standalone ctest suites in this directory (see test_pivotal_bootstrap.cpp).
#include <cmath>
#include <functional>
#include <stdexcept>

#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/math/rootfinding/bisection.hpp"
#include "corehydro/numerics/math/rootfinding/newton_raphson.hpp"
#include "corehydro/numerics/math/rootfinding/secant.hpp"
#include "corehydro/numerics/tools.hpp"
#include "check.hpp"

using corehydro::numerics::kPi;
using corehydro::numerics::math::linalg::Matrix;
using corehydro::numerics::math::linalg::Vector;
using corehydro::numerics::math::rootfinding::bisection_solve;
using corehydro::numerics::math::rootfinding::newton_raphson_robust_solve;
using corehydro::numerics::math::rootfinding::newton_raphson_solve;
using corehydro::numerics::math::rootfinding::newton_raphson_solve_system;
using corehydro::numerics::math::rootfinding::secant_solve;

namespace {

// ---- TestFunctions.cs -------------------------------------------------------------------

// A quadratic test function. [0, 4] x = sqrt(2)
double Quadratic(double x) { return std::pow(x, 2) - 2; }
// First derivative of quadratic function.
double Quadratic_Deriv(double x) { return 2 * x; }

// A cubic test function. [-1, 5] x = 1.32472
double Cubic(double x) { return x * x * x - x - 1.0; }
// First derivative of cubic function.
double Cubic_Deriv(double x) { return 3.0 * (x * x) - 1.0; }

// A trigonometric test function. [0, 3.14] x = 1.12191713
double Trigonometric(double x) { return 2 * std::sin(x) - 3 * std::cos(x) - 0.5; }
// First derivative of the trigonometric function.
double Trigonometric_Deriv(double x) { return 2 * std::cos(x) + 3 * std::sin(x); }

// An exponential test function. [-2, 2] x = 0.567143290
double Exponential(double x) { return std::exp(-x) - x; }
// First derivative of the exponential function.
double Exponential_Deriv(double x) { return -std::exp(-x) - 1; }

// A power test function. [0, 2] x = 1.0
double Power(double x) { return std::pow(x, 10) - 1; }
// First derivative of the power function.
double Power_Deriv(double x) { return 10 * std::pow(x, 9); }

// Test_Secant.cs's local Undefined(x) = 1/x fixture (discontinuous within the test interval).
double Undefined(double x) { return 1.0 / x; }

// ---- Test_Bisection.cs -------------------------------------------------------------------

void test_bisection_quadratic() {
    double initial = 1.0;
    double lower = 0;
    double upper = 4;
    double X = bisection_solve(Quadratic, initial, lower, upper);
    double trueX = std::sqrt(2.0);
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_bisection_cubic() {
    double initial = 1.0;
    double lower = -1;
    double upper = 5;
    double X = bisection_solve(Cubic, initial, lower, upper);
    double trueX = 1.32472;
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_bisection_trigonometric() {
    double initial = 0.5;
    double lower = 0;
    double upper = kPi;
    double X = bisection_solve(Trigonometric, initial, lower, upper);
    double trueX = 1.12191713;
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_bisection_exponential() {
    double initial = 1.0;
    double lower = -2;
    double upper = 2;
    double X = bisection_solve(Exponential, initial, lower, upper);
    double trueX = 0.567143290;
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_bisection_power() {
    double initial = 0.2;
    double lower = 0;
    double upper = 2;
    double X = bisection_solve(Power, initial, lower, upper);
    double trueX = 1.0;
    CHECK_NEAR(X, trueX, 1E-5);
}

// Test_BisectionEdge1: first guess outside the bracket.
void test_bisection_edge1() {
    double initial = -1.0;
    double lower = 1.0;
    double upper = 5.0;
    CHECK_THROWS(bisection_solve(Cubic, initial, lower, upper));
}

// Test_BisectionEdge2: upper bound less than lower bound.
void test_bisection_edge2() {
    double initial = 1.0;
    double lower = 5;
    double upper = 0.0;
    CHECK_THROWS(bisection_solve(Cubic, initial, lower, upper));
}

// Test_BisectionEdge3: root not bracketed.
void test_bisection_edge3() {
    double initial = 3.0;
    double lower = 2;
    double upper = 5.0;
    CHECK_THROWS(bisection_solve(Cubic, initial, lower, upper));
}

// ---- Test_NewtonRaphson.cs ---------------------------------------------------------------

void test_newton_raphson_quadratic() {
    double initial = 1.0;
    double X = newton_raphson_solve(Quadratic, Quadratic_Deriv, initial);
    double trueX = std::sqrt(2.0);
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_newton_raphson_robust_quadratic() {
    double initial = 1.0;
    double lower = 0;
    double upper = 4;
    double X = newton_raphson_robust_solve(Quadratic, Quadratic_Deriv, initial, lower, upper);
    double trueX = std::sqrt(2.0);
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_newton_raphson_cubic() {
    double initial = 1.0;
    double X = newton_raphson_solve(Cubic, Cubic_Deriv, initial);
    double trueX = 1.32472;
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_newton_raphson_robust_cubic() {
    double initial = 1.0;
    double lower = -1;
    double upper = 5;
    double X = newton_raphson_robust_solve(Cubic, Cubic_Deriv, initial, lower, upper);
    double trueX = 1.32472;
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_newton_raphson_trigonometric() {
    double initial = 0.5;
    double X = newton_raphson_solve(Trigonometric, Trigonometric_Deriv, initial);
    double trueX = 1.12191713;
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_newton_raphson_robust_trigonometric() {
    double initial = 0.5;
    double lower = 0;
    double upper = kPi;
    double X = newton_raphson_robust_solve(Trigonometric, Trigonometric_Deriv, initial, lower, upper);
    double trueX = 1.12191713;
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_newton_raphson_exponential() {
    double initial = 1.0;
    double X = newton_raphson_solve(Exponential, Exponential_Deriv, initial);
    double trueX = 0.567143290;
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_newton_raphson_robust_exponential() {
    double initial = 1.0;
    double lower = -2;
    double upper = 2;
    double X = newton_raphson_robust_solve(Exponential, Exponential_Deriv, initial, lower, upper);
    double trueX = 0.567143290;
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_newton_raphson_power() {
    double initial = 0.2;
    double X = newton_raphson_solve(Power, Power_Deriv, initial);
    double trueX = 1.0;
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_newton_raphson_robust_power() {
    double initial = 0.2;
    double lower = 0;
    double upper = 2;
    double X = newton_raphson_robust_solve(Power, Power_Deriv, initial, lower, upper);
    double trueX = 1.0;
    CHECK_NEAR(X, trueX, 1E-5);
}

// Test_Multi_LinearSystem: F([x;y]) = [3x + y - 9, x + 2y - 8]  => root x* = [2;3]
void test_newton_raphson_multi_linear_system() {
    std::function<Vector(const Vector&)> F = [](const Vector& v) {
        return Vector(std::vector<double>{3.0 * v[0] + v[1] - 9.0, v[0] + 2.0 * v[1] - 8.0});
    };
    std::function<Matrix(const Vector&)> J = [](const Vector&) {
        return Matrix(2, 2, std::vector<double>{3.0, 1.0, 1.0, 2.0});
    };
    Vector x0(std::vector<double>{0.0, 0.0});
    Vector expected(std::vector<double>{2.0, 3.0});

    Vector root = newton_raphson_solve_system(F, J, x0);
    CHECK_NEAR(root[0], expected[0], 1E-5);
    CHECK_NEAR(root[1], expected[1], 1E-5);
}

// Test_Multi_Nonlinear: F2([x;y]) = [x^2 + y^2 - 5, x*y - 2]  => root = [2;1]
void test_newton_raphson_multi_nonlinear() {
    std::function<Vector(const Vector&)> F = [](const Vector& v) {
        return Vector(std::vector<double>{v[0] * v[0] + v[1] * v[1] - 5, v[0] * v[1] - 2});
    };
    std::function<Matrix(const Vector&)> J = [](const Vector& v) {
        return Matrix(2, 2, std::vector<double>{2.0 * v[0], 2.0 * v[1], v[1], v[0]});
    };
    Vector x0(std::vector<double>{3.0, 2.0});
    Vector expected(std::vector<double>{2.0, 1.0});

    Vector root = newton_raphson_solve_system(F, J, x0);
    CHECK_NEAR(root[0], expected[0], 1E-5);
    CHECK_NEAR(root[1], expected[1], 1E-5);
}

// Test_Multi_DecoupledNonlinear: F([x;y;z]) = [x^2 + y - 6, x + y^2 - 6, z - 2] => root (2,2,2)
void test_newton_raphson_multi_decoupled_nonlinear() {
    std::function<Vector(const Vector&)> F = [](const Vector& v) {
        return Vector(
            std::vector<double>{v[0] * v[0] + v[1] - 6, v[0] + v[1] * v[1] - 6, v[2] - 2});
    };
    std::function<Matrix(const Vector&)> J = [](const Vector& v) {
        return Matrix(3, 3, std::vector<double>{2 * v[0], 1, 0, 1, 2 * v[1], 0, 0, 0, 1});
    };
    Vector x0(std::vector<double>{1.0, 3.0, 0.0});
    Vector expected(std::vector<double>{2.0, 2.0, 2.0});

    Vector root = newton_raphson_solve_system(F, J, x0);
    CHECK_NEAR(root[0], expected[0], 1E-5);
    CHECK_NEAR(root[1], expected[1], 1E-5);
}

// ---- Test_Secant.cs ----------------------------------------------------------------------

void test_secant_quadratic() {
    double lower = 0;
    double upper = 4;
    double X = secant_solve(Quadratic, lower, upper);
    double trueX = std::sqrt(2.0);
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_secant_cubic() {
    double lower = -1;
    double upper = 5;
    double X = secant_solve(Cubic, lower, upper);
    double trueX = 1.32472;
    CHECK_NEAR(X, trueX, 1E-5);
}

void test_secant_exponential() {
    double lower = -2;
    double upper = 2;
    double X = secant_solve(Exponential, lower, upper);
    double trueX = 0.567143290;
    CHECK_NEAR(X, trueX, 1E-5);
}

// Test_Edge: discontinuous function within the interval.
void test_secant_edge() {
    double lower = -6.0;
    double upper = 5.0;
    CHECK_THROWS(secant_solve(Undefined, lower, upper));
}

}  // namespace

int main() {
    test_bisection_quadratic();
    test_bisection_cubic();
    test_bisection_trigonometric();
    test_bisection_exponential();
    test_bisection_power();
    test_bisection_edge1();
    test_bisection_edge2();
    test_bisection_edge3();

    test_newton_raphson_quadratic();
    test_newton_raphson_robust_quadratic();
    test_newton_raphson_cubic();
    test_newton_raphson_robust_cubic();
    test_newton_raphson_trigonometric();
    test_newton_raphson_robust_trigonometric();
    test_newton_raphson_exponential();
    test_newton_raphson_robust_exponential();
    test_newton_raphson_power();
    test_newton_raphson_robust_power();
    test_newton_raphson_multi_linear_system();
    test_newton_raphson_multi_nonlinear();
    test_newton_raphson_multi_decoupled_nonlinear();

    test_secant_quadratic();
    test_secant_cubic();
    test_secant_exponential();
    test_secant_edge();

    return chtest::summary("root_finding_extras");
}
