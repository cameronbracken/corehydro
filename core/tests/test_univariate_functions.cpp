// Transcribed C# oracle tests for the Numerics univariate Functions layer (Task 11):
//   upstream/Numerics/Test_Numerics/Functions/Test_Functions.cs   @ 2a0357a
//
// Six of the seven Test_Functions.cs methods are transcribed here: Test_Linear_Function,
// Test_Linear_Function_Inverse, Test_Power_Function, Test_Power_Function_Inverse,
// Test_InversePower_Function, Test_InversePower_Function_Inverse.
//
// Test_Tabular_Function is NOT transcribed: TabularFunction (Numerics/Functions/
// TabularFunction.cs) is a documented severance, not ported. It is built entirely on
// UncertainOrderedPairedData/OrderedPairedData/Ordinate/UncertainOrdinate (the Numerics.Data
// "Paired Data" subsystem), which this repo has not ported -- scheduled for Phase P4. See
// i_univariate_function.hpp's file header and upstream/CLAUDE.md's "What is deliberately not
// ported" section.
//
// The C# test file mixes computed-expected literals (e.g. `double valid1 = (5*6) + -2;`) and
// hand-transcribed decimal literals found via R's norminv() (e.g. `30.0234692505882`); both
// forms are transcribed verbatim below, matching each source line.
#include <cmath>

#include "corehydro/numerics/functions/linear_function.hpp"
#include "corehydro/numerics/functions/power_function.hpp"
#include "check.hpp"

using corehydro::numerics::functions::LinearFunction;
using corehydro::numerics::functions::PowerFunction;

namespace {

void test_linear_function() {
    // Default constructor with alpha = 0 and beta = 1.
    LinearFunction func0;
    double y0 = func0.function(6);
    double valid0 = 6;
    CHECK_NEAR(y0, valid0, 1e-6);

    double alpha = -2;
    double beta = 5;
    double sigma = 3;

    LinearFunction func1(alpha, beta);
    double y1 = func1.function(6);
    double valid1 = (5 * 6) + -2;
    CHECK_NEAR(y1, valid1, 1e-6);

    LinearFunction func2(alpha, beta, sigma);
    double y2 = func2.function(6);
    CHECK_NEAR(y2, valid1, 1e-6);

    func2.set_confidence_level(0.75);
    double y3 = func2.function(6);
    // Found using R's norminv() function for epsilon from the "drcarlate" package.
    double valid3 = 30.0234692505882;
    CHECK_NEAR(y3, valid3, 1e-6);
}

void test_linear_function_inverse() {
    LinearFunction func(10, 0.5, 20);
    double y = func.function(400);
    double x = func.inverse_function(y);
    CHECK_NEAR(x, 400, 1e-6);

    func.set_confidence_level(0.75);
    double yy = func.function(400);
    double xx = func.inverse_function(yy);
    CHECK_NEAR(xx, 400, 1e-6);
}

void test_power_function() {
    // Default constructor with alpha = 1, beta = 1.5, and xi = 0.
    PowerFunction func0;
    double y0 = func0.function(6);
    double valid0 = 1 * std::pow(6 - 0, 1.5);
    CHECK_NEAR(y0, valid0, 1e-6);

    double alpha = 5;
    double beta = 2;
    double sigma = 3;
    double xi = 0;

    PowerFunction func1(alpha, beta, xi);
    double y1 = func1.function(6);
    double valid1 = alpha * std::pow(6 - xi, beta);
    CHECK_NEAR(y1, valid1, 1e-6);

    PowerFunction func2(alpha, beta, xi, sigma);
    double y2 = func2.function(6);
    CHECK_NEAR(y2, valid1, 1e-6);

    func2.set_confidence_level(0.75);
    double y3 = func2.function(6);
    // Found using R's norminv() function for epsilon from the "drcarlate" package.
    double valid3 = 1361.61408399941;
    CHECK_NEAR(y3, valid3, 1e-6);
}

void test_power_function_inverse() {
    PowerFunction func(10, 2, 0, 0.1);
    double y = func.function(400);
    double x = func.inverse_function(y);
    CHECK_NEAR(x, 400, 1e-6);

    func.set_confidence_level(0.75);
    double yy = func.function(400);
    double xx = func.inverse_function(yy);
    CHECK_NEAR(xx, 400, 1e-6);
}

void test_inverse_power_function() {
    double alpha = 5;
    double beta = 2;
    double sigma = 3;
    double xi = 0;

    PowerFunction func(alpha, beta, xi);
    func.set_is_inverse(true);
    double y = func.function(6);
    double valid = std::sqrt(6.0 / alpha) + xi;
    CHECK_NEAR(y, valid, 1e-6);

    PowerFunction func2(alpha, beta, xi, sigma);
    func2.set_is_inverse(true);
    double y2 = func2.function(6);
    CHECK_NEAR(y2, valid, 1e-6);

    func2.set_confidence_level(0.75);
    double y3 = func2.function(6);
    // Found using R's norminv() function for epsilon from the "drcarlate" package.
    double valid3 = 0.398290417772997;
    CHECK_NEAR(y3, valid3, 1e-6);
}

void test_inverse_power_function_inverse() {
    PowerFunction func(10, 2, 0, 0.1);
    func.set_is_inverse(true);
    double y = func.function(6);
    double x = func.inverse_function(y);
    CHECK_NEAR(x, 6, 1e-6);

    func.set_confidence_level(0.75);
    double yy = func.function(6);
    double xx = func.inverse_function(yy);
    CHECK_NEAR(xx, 6, 1e-6);
}

}  // namespace

int main() {
    test_linear_function();
    test_linear_function_inverse();
    test_power_function();
    test_power_function_inverse();
    test_inverse_power_function();
    test_inverse_power_function_inverse();
    return chtest::summary("test_univariate_functions");
}
