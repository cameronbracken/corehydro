// ported from: Numerics/Mathematics/Root Finding/NewtonRaphson.cs @ 2a0357a
//
// Newton-Raphson root finding: the basic (unbracketed) method, the robust variant that falls
// back to bisection when Newton over/undershoots the bracket, and the multivariate solve over
// `linalg::Vector`/`linalg::Matrix` that solves J*deltaX = F(x) via `Matrix::inverse()`
// (LU-decomposition backed), matching the C# `J.Inverse().Multiply(fx)` call exactly.
#pragma once
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

#include "corehydro/numerics/math/linalg/lu_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::rootfinding {

// Use the basic Newton-Raphson method to find a solution of the equation f(x)=0.
// The basic algorithm aborts immediately if the root leaves the bound interval.
inline double newton_raphson_solve(const std::function<double(double)>& f,
                                   const std::function<double(double)>& df, double first_guess,
                                   double tolerance = 1E-8, int max_iterations = 1000,
                                   bool report_failure = true) {
    // Define variables
    double root = first_guess;
    bool solution_found = false;
    double y;
    double y_prime;
    double x0;
    double x1;
    // C#: Tools.DoubleMachineEpsilon -- don't want to divide by a number smaller than this
    const double eps = kDoubleMachineEpsilon;

    // Newton-Raphson loop
    for (int i = 1; i <= max_iterations; ++i) {
        // do Newton's computation
        x0 = root;
        y = f(x0);
        y_prime = df(x0);
        if (std::fabs(y_prime) < eps) {
            // the denominator is too small
            solution_found = false;
            break;
        }
        x1 = x0 - y / y_prime;
        root = x1;
        // check if the result is within the desired tolerance
        if (std::fabs(x1 - x0) < tolerance) {
            solution_found = true;
            break;
        }
    }

    // return results of solver
    if (!solution_found && report_failure)
        throw std::invalid_argument("Newton-Raphson method failed to find root.");
    return root;
}

// Use the robust Newton-Raphson method to find a solution of the equation f(x)=0.
// Robust Newton-Raphson method falls back to bisection when over or undershooting the bounds.
inline double newton_raphson_robust_solve(const std::function<double(double)>& f,
                                          const std::function<double(double)>& df,
                                          double first_guess, double lower_bound,
                                          double upper_bound, double tolerance = 1E-8,
                                          int max_iterations = 1000, bool report_failure = true) {
    // Define variables
    double root = first_guess;
    bool solution_found = false;
    double xl = lower_bound;
    double xh = upper_bound;
    double fl = f(xl);
    double fh = f(xh);

    // validate inputs
    if (upper_bound < lower_bound)
        throw std::out_of_range("The upper bound (b) cannot be less than the lower bound (a).");
    if (root < lower_bound || root > upper_bound)
        throw std::out_of_range("The first guess must be between the upper and lower bound.");
    if (fl * fh >= 0.0)
        throw std::invalid_argument(
            "Robust Newton-Raphson method failed because the root is not bracketed.");

    if (fl < 0.0) {
        // Orient the search so that f(lowerBound)<0
        xl = lower_bound;
        xh = upper_bound;
    } else {
        xh = lower_bound;
        xl = upper_bound;
    }
    // The "step-size before last"
    double dxold = std::fabs(upper_bound - lower_bound);
    // The last step
    double dx = dxold;
    double F = f(root);
    double dF = df(root);
    // Robust Newton-Raphson loop
    for (int i = 1; i <= max_iterations; ++i) {
        // Bisect if Newton is out of range, or not decreasing fast enough.
        if ((((root - xh) * dF - F) * ((root - xl) * dF - F) > 0.0) ||
            (std::fabs(2.0 * F) > std::fabs(dxold * dF))) {
            dxold = dx;
            dx = 0.5 * (xh - xl);
            root = xl + dx;
            if (xl == root) {
                // Change in root is negligible.
                // Bisection step acceptable.
                solution_found = true;
                break;
            }
        } else {
            dxold = dx;
            dx = F / dF;
            double temp = root;
            root -= dx;
            if (temp == root) {
                // Change in root is negligible.
                // Newton step acceptable.
            }
        }
        // Check convergence
        if (std::fabs(dx) < tolerance) {
            solution_found = true;
            break;
        }
        // The one new function evaluation per iteration.
        F = f(root);
        dF = df(root);
        // Maintain the bracket on the root.
        if (F < 0.0)
            xl = root;
        else
            xh = root;
    }

    // return results of solver
    if (!solution_found && report_failure)
        throw std::invalid_argument("Robust Newton-Raphson method failed to find root.");
    return root;
}

// Solves a system of nonlinear equations using the Newton-Raphson method.
// f: the system of equations, taking a vector and returning a vector of the same dimension.
// df: the Jacobian matrix function, taking x and returning the matrix of partial derivatives.
// Iterates x_{n+1} = x_n - J(x_n)^{-1} * f(x_n).
inline linalg::Vector newton_raphson_solve_system(
    const std::function<linalg::Vector(const linalg::Vector&)>& f,
    const std::function<linalg::Matrix(const linalg::Vector&)>& df,
    const linalg::Vector& first_guess, double tolerance = 1E-8, int max_iterations = 1000,
    bool report_failure = true) {
    linalg::Vector x = first_guess;
    bool converged = false;
    // C#: Tools.DoubleMachineEpsilon
    const double eps = kDoubleMachineEpsilon;

    for (int iter = 1; iter <= max_iterations; ++iter) {
        linalg::Vector fx = f(x);
        linalg::Matrix J = df(x);

        // Check for near-singular Jacobian
        if (std::fabs(J.determinant()) < eps) {
            if (report_failure)
                throw std::invalid_argument("Jacobian is singular or nearly singular.");
            break;
        }

        // Solve J*deltaX = F(x)  ->  deltaX = J^-1 * F(x)
        linalg::Vector delta_x = J.inverse().multiply(fx);

        // Update
        linalg::Vector x_new = x - delta_x;

        // Check convergence: step size and residual
        if (delta_x.norm() < tolerance && f(x_new).norm() < tolerance) {
            x = x_new;
            converged = true;
            break;
        }

        x = x_new;
    }

    if (!converged && report_failure)
        throw std::invalid_argument("Newton-Raphson did not converge after " +
                                    std::to_string(max_iterations) + " iterations.");

    return x;
}

}  // namespace corehydro::numerics::math::rootfinding
