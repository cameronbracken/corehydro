// ported from: Numerics/Mathematics/Root Finding/Brent.cs @ 2a0357a
//
// Brent's method for root finding (Sprott, "Numerical Recipes ... in Basic", 1991).
#pragma once
#include <cmath>
#include <functional>
#include <stdexcept>

#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::rootfinding {

// Fortran-style sign transfer: |a| if b >= 0 else -|a|  (mirrors Tools.Sign).
inline double sign(double a, double b) {
    return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

// The ported defaults, named rather than left as bare literals in the signature. A caller that
// must pass `tolerance` positionally in order to reach `max_iterations` would otherwise have to
// write `1E-8` itself -- a second copy of a value that belongs here, in the ported routine, so
// that a change to the C# default lands in exactly one place. support/callback/math.hpp's
// root_find arm is that caller, and the promise not to restate these is a documented one on both
// packages' `root_find()` help pages.
inline constexpr double kDefaultTolerance = 1E-8;
inline constexpr int kDefaultMaxIterations = 1000;

inline double solve(const std::function<double(double)>& f, double lower_bound,
                    double upper_bound, double tolerance = kDefaultTolerance,
                    int max_iterations = kDefaultMaxIterations, bool report_failure = true) {
    if (upper_bound < lower_bound)
        throw std::out_of_range("upper bound cannot be less than the lower bound");

    bool solution_found = false;
    const double EPS = kDoubleMachineEpsilon;
    double a = lower_bound, b = upper_bound, c = upper_bound, d = 0, e = 0;
    double fa = f(a), fb = f(b), fc, p, q, r, s, tol1, xm, root = 0;

    if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0))
        throw std::invalid_argument("Brent's method failed because the root is not bracketed.");
    fc = fb;

    for (int i = 1; i <= max_iterations; ++i) {
        if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            c = a;
            fc = fa;
            e = d = b - a;
        }
        if (std::fabs(fc) < std::fabs(fb)) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        tol1 = 2.0 * EPS * std::fabs(b) + 0.5 * tolerance;
        xm = 0.5 * (c - b);

        if (std::fabs(xm) <= tol1 || fb == 0.0) {
            root = b;
            solution_found = true;
            break;
        }
        if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb)) {
            s = fb / fa;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = std::fabs(p);
            double min1 = 3.0 * xm * q - std::fabs(tol1 * q);
            double min2 = std::fabs(e * q);
            if (2.0 * p < (min1 < min2 ? min1 : min2)) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if (std::fabs(d) > tol1)
            b += d;
        else
            b += sign(tol1, xm);
        fb = f(b);
    }

    if (!solution_found && report_failure)
        throw std::runtime_error("Brent's method failed to find root.");
    return solution_found ? root : b;
}

}  // namespace corehydro::numerics::math::rootfinding
