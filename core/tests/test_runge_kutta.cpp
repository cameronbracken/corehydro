// Standalone tests for the RungeKutta ODE solver.
//
// Transcribed 1:1 (structure, inputs and oracle values unaltered) from
//   upstream/Numerics/Test_Numerics/Mathematics/ODE Solvers/Test_RungeKutta.cs @ 2a0357a
// -- all 5 [TestMethod]s in C# file order. The shared RHS fixture (`y - t^2 + 1`, whose exact
// solution is `(t + 1)^2 - 0.5 exp(t)`) is transcribed below as a file-local static function,
// matching the style of the other standalone ctest suites in this directory (see
// test_root_finding_extras.cpp).
#include <vector>

#include "corehydro/numerics/math/ode/runge_kutta.hpp"
#include "check.hpp"

using corehydro::numerics::math::ode::cash_karp;
using corehydro::numerics::math::ode::fehlberg;
using corehydro::numerics::math::ode::fourth_order;
using corehydro::numerics::math::ode::fourth_order_step;
using corehydro::numerics::math::ode::second_order;

namespace {

// Test_RungeKutta.cs's shared ODE: dy/dt = y - t^2 + 1.
double ode(double t, double y) { return y - (t * t) + 1.0; }

const std::vector<double> kTestValid = {0.5, 1.425639364649936, 2.640859085770477,
                                        4.009155464830968, 5.305471950534675};

// Test_2ndRK
void test_2nd_rk() {
    double initial = 0.5, start = 0.0, end = 2.0;
    int steps = 5;
    std::vector<double> results = second_order(ode, initial, start, end, steps);
    for (std::size_t i = 0; i < kTestValid.size(); ++i) CHECK_NEAR(results[i], kTestValid[i], 1);
}

// Test_4thRK
void test_4th_rk() {
    double initial = 0.5, start = 0.0, end = 2.0;
    int steps = 5;
    std::vector<double> results = fourth_order(ode, initial, start, end, steps);
    for (std::size_t i = 0; i < kTestValid.size(); ++i) CHECK_NEAR(results[i], kTestValid[i], 1E-2);
}

// Test_4thRK_2: the single-step overload, called in a loop.
void test_4th_rk_2() {
    double initial = 0.5, start = 0.0, dt = 0.5;
    std::vector<double> results(kTestValid.size());
    results[0] = initial;
    for (std::size_t i = 1; i < results.size(); ++i) {
        results[i] = fourth_order_step(ode, initial, start, dt);
        initial = results[i];
        start += dt;
    }
    for (std::size_t i = 0; i < kTestValid.size(); ++i) CHECK_NEAR(results[i], kTestValid[i], 1E-2);
}

// Test_RKF
void test_rkf() {
    double initial = 0.5, start = 0.0, dt = 0.5, dt_min = 0.001;
    std::vector<double> results(kTestValid.size());
    results[0] = initial;
    for (std::size_t i = 1; i < results.size(); ++i) {
        results[i] = fehlberg(ode, initial, start, dt, dt_min);
        initial = results[i];
        start += dt;
    }
    for (std::size_t i = 0; i < kTestValid.size(); ++i) CHECK_NEAR(results[i], kTestValid[i], 1E-4);
}

// Test_RKCK
void test_rkck() {
    double initial = 0.5, start = 0.0, dt = 0.5, dt_min = 0.001;
    std::vector<double> results(kTestValid.size());
    results[0] = initial;
    for (std::size_t i = 1; i < results.size(); ++i) {
        results[i] = cash_karp(ode, initial, start, dt, dt_min);
        initial = results[i];
        start += dt;
    }
    for (std::size_t i = 0; i < kTestValid.size(); ++i) CHECK_NEAR(results[i], kTestValid[i], 1E-3);
}

}  // namespace

int main() {
    test_2nd_rk();
    test_4th_rk();
    test_4th_rk_2();
    test_rkf();
    test_rkck();

    return chtest::summary("runge_kutta");
}
