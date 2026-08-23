// ported from: Numerics/Mathematics/Special Functions/Debye.cs @ 2a0357a
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

}  // namespace corehydro::numerics::math::special
