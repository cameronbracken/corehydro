// ported from: Numerics/Mathematics/Root Finding/Bisection.cs @ 2a0357a
//
// The bisection method for root finding (Press et al., "Numerical Recipes", 3rd ed., 2017).
// Unlike brent.hpp's `solve`, C# `Bisection.Solve` takes a `firstGuess` AND a bracket -- an
// unusual signature (the guess seeds the running root, the bracket only seeds the initial
// step direction and sign check) that is kept exactly as the C# source has it.
#pragma once
#include <cmath>
#include <functional>
#include <stdexcept>

namespace corehydro::numerics::math::rootfinding {

inline double bisection_solve(const std::function<double(double)>& f, double first_guess,
                              double lower_bound, double upper_bound, double tolerance = 1E-8,
                              int max_iterations = 1000, bool report_failure = true) {
    // Define variables
    double root = first_guess;
    bool solution_found = false;
    double xl = lower_bound;
    double xh = upper_bound;
    double fl = f(xl);
    double fmid = f(xh);
    double dx, xmid;

    // validate inputs
    if (upper_bound < lower_bound)
        throw std::out_of_range("The upper bound (b) cannot be less than the lower bound (a).");
    if (root < lower_bound || root > upper_bound)
        throw std::out_of_range("The first guess must be between the upper and lower bound.");
    if (fl * fmid >= 0.0)
        throw std::invalid_argument("Bisection method failed because the root is not bracketed.");

    if (fl < 0.0)
        // Orient the search so that f>0 lies at X+DX
        dx = xh - xl;
    else
        dx = xl - xh;

    // Bisection loop
    for (int i = 1; i <= max_iterations; ++i) {
        dx *= 0.5;
        xmid = root + dx;
        fmid = f(xmid);
        if (fmid <= 0) root = xmid;
        // check if the solution meets required tolerance
        if (std::fabs(dx) <= tolerance || fmid == 0.0) {
            // a solution has been achieved, so exit loop
            solution_found = true;
            break;
        }
    }

    // return results of solver
    if (!solution_found && report_failure)
        throw std::invalid_argument("Bisection method failed to find root.");
    return root;
}

}  // namespace corehydro::numerics::math::rootfinding
