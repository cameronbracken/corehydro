// ported from: Numerics/Mathematics/Special Functions/Debye.cs @ 7e8e8d1
//
// The Debye function:
//
//              x
//   D(x) = x/x^n ∫  t^n / (e^t - 1) dt
//              0
//
// mirroring the C# `Debye` class's single public member, `Function`, method-for-method: a
// truncated Taylor series for x <= 0.1, a rational (Chebyshev-style) approximation for
// 0.1 < x <= 7.25, and an asymptotic exponential series for x > 7.25.
#pragma once
#include <cmath>
#include <stdexcept>

#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::special {

// Computes the Debye function. Ported from Debye.Function.
inline double debye_function(double x) {
    if (x < 0.0) throw std::out_of_range("debye_function: x must be positive");

    if (x == 0.0) {
        return 1.0;
    } else if (x > 0.0 && x <= 0.1) {
        constexpr double t = 5.952380953E-4;
        return 1.0 - 0.375 * x + x * x * (0.05 - t * x * x);
    } else if (x > 0.1 && x <= 7.25) {
        return ((((0.0946173 * x - 4.432582) * x + 85.07724) * x - 800.6087) * x + 3953.632) /
               ((((x + 15.121491) * x + 143.155337) * x + 682.0012) * x + 3953.632);
    } else {
        // x > 7.25. Mirrors the C# loop condition `i <= N` exactly, where N = 25/x is a
        // double: comparing the loop counter directly against N (rather than pre-truncating
        // N to an int) matches C#'s implicit int-to-double promotion in that comparison.
        double N = 25.0 / x;
        double D = 0.0;
        if (x <= 25.0) {
            double D2 = 1.0;
            for (int i = 1; i <= N; ++i) {
                D2 *= std::exp(-x);
                double x3 = i * x;
                D += D2 * (6.0 + x3 * (6.0 + x3 * (3.0 + x3))) / std::pow(static_cast<double>(i), 4);
            }
        }
        // x > 25 falls straight through with D left at 0.0, exactly as the C# source does
        // (its `else if (x > 25)` branch never touches D).
        return 3.0 * (6.493939402 - D) / (x * x * x);
    }
}

// Computes the order-1 Debye function D1(x) for any real argument.
inline double debye_function_order_one(double x) {
    static constexpr double coefficients[] = {
        2.7777777777777776e-02,  -2.7777777777777778e-04, 4.7241118669690098e-06,
        -9.1857730746619641e-08, 1.8978869988971000e-09,  -4.0647616451442256e-11,
        8.9216910204564523e-13,  -1.9939295860721074e-14, 4.5189800296199183e-16,
        -1.0356517612181247e-17, 2.3952186210261870e-19,  -5.5817858743250090e-21};

    if (x == 0.0) return 1.0;
    if (x < 0.0) return debye_function_order_one(-x) - 0.5 * x;
    if (x <= 1.0) {
        const double squared = x * x;
        double power = 1.0;
        double sum = 0.0;
        for (double coefficient : coefficients) {
            power *= squared;
            sum += coefficient * power;
        }
        return 1.0 - 0.25 * x + sum;
    }

    double remainder = 0.0;
    for (int k = 1; k <= 1000; ++k) {
        const double kd = static_cast<double>(k);
        const double term = std::exp(-kd * x) * (1.0 / kd + 1.0 / (kd * kd * x));
        remainder += term;
        if (term < 1e-20) break;
    }
    return corehydro::numerics::kPi * corehydro::numerics::kPi / (6.0 * x) - remainder;
}

}  // namespace corehydro::numerics::math::special
