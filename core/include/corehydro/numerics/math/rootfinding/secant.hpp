// ported from: Numerics/Mathematics/Root Finding/Secant.cs @ 2a0357a
//
// The secant method for root finding (Sprott, "Numerical Recipes ... in Basic", 1991).
#pragma once
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>

namespace corehydro::numerics::math::rootfinding {

inline double secant_solve(const std::function<double(double)>& f, double lower_bound,
                           double upper_bound, double tolerance = 1E-8, int max_iterations = 1000,
                           bool report_failure = true) {
    // validate inputs
    if (upper_bound < lower_bound)
        throw std::out_of_range("The upper bound (b) cannot be less than the lower bound (a).");

    // Define variables
    bool solution_found = false;
    double xl, root;
    double x1 = lower_bound;
    double x2 = upper_bound;
    double fl = f(x1);
    double fh = f(x2);

    // Pick the bound with the smaller function value as the most recent guess
    if (std::fabs(fl) < std::fabs(fh)) {
        root = x1;
        xl = x2;
        std::swap(fl, fh);  // C#: Tools.Swap
    } else {
        xl = x1;
        root = x2;
    }

    // Secant loop
    for (int i = 1; i <= max_iterations; ++i) {
        double dx = (xl - root) * fh / (fh - fl);
        xl = root;
        fl = fh;
        root += dx;
        fh = f(root);
        if (std::fabs(dx) < tolerance || fh == 0.0) {
            solution_found = true;
            break;
        }
    }

    // return results of solver
    if (!solution_found && report_failure)
        throw std::invalid_argument("Secant method failed to find root.");
    return root;
}

}  // namespace corehydro::numerics::math::rootfinding
