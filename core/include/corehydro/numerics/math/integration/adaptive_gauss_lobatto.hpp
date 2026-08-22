// ported from: Numerics/Mathematics/Integration/AdaptiveGaussLobatto.cs @ 2a0357a
//
// Adaptive integration by the Gauss-Lobatto method with a Kronrod extension (Numerical Recipes
// "adaptlob"): a 4-point Gauss-Lobatto estimate `i2` and its 7-point Kronrod extension `i1` are
// compared against a fixed 13-point reference `iS` computed once on the whole interval, and each
// sub-interval recurses into six children (bounded by `mll`/`ml`/`m`/`mr`/`mrr`) until `i1` and
// `i2` agree to `toler * iS` or the interval has shrunk to machine precision.
//
// PORTING DEVIATION: the C# class carries `private static bool terminate` -- a class-level,
// not instance-level, flag that a recursive call latches to `false` the first time a
// sub-interval is bounded by machine-epsilon on either side ("the interval contains no more
// machine numbers") and never resets. Being `static`, it is shared and mutated across every
// `AdaptiveGaussLobatto` instance in the process and is not thread-safe. This port makes it an
// ordinary instance member (`terminate_`, default `true`), so each object gets its own flag
// rather than one shared mutable global; nothing in this port (or in the C# source) reads the
// flag back out, so the change only affects incidental cross-instance state leakage, not any
// oracle-visible value.
//
// SEQUENCING NOTE: `adaptlob`'s six-way recursive sum sets `Status` at each leaf via
// `UpdateStatus`, so the value `Status` holds after the OUTER call returns is whichever leaf
// executed LAST -- C# guarantees strict left-to-right evaluation of `a + b + c + d + e + f`, so
// that is always the rightmost (`mrr` to `b`) branch. C++ gives no such guarantee for the
// built-in `+` operator's operand evaluation order, so the six recursive calls below are bound to
// named locals in explicit left-to-right statement order before being summed, reproducing the
// C# evaluation order (and therefore which branch's status "wins") deterministically rather than
// leaving it compiler-dependent.
#pragma once
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>

#include "corehydro/numerics/math/integration/support/integration_status.hpp"
#include "corehydro/numerics/math/integration/support/integrator.hpp"

namespace corehydro::numerics::math::integration {

namespace detail_gauss_lobatto {
// Abscissas for Gauss-Lobatto-Kronrod quadrature. alpha/beta are computed via sqrt, exactly as
// the C# static field initializers compute them at class-load time (IEEE754 sqrt is correctly
// rounded, so the two are bit-identical); x1/x2/x3 are C#'s own literal values.
inline const double kAlpha = std::sqrt(2.0 / 3.0);
inline const double kBeta = 1.0 / std::sqrt(5.0);
inline constexpr double kX1 = 0.942882415695480;
inline constexpr double kX2 = 0.641853342345781;
inline constexpr double kX3 = 0.236383199662150;
inline const double kNodes[12] = {0.0,   -kX1,  -kAlpha, -kX2, -kBeta, -kX3,
                                   0.0,   kX3,   kBeta,   kX2,  kAlpha, kX1};
}  // namespace detail_gauss_lobatto

/// A class that performs adaptive integration by the Gauss-Lobatto method with a Kronrod
/// extension.
class AdaptiveGaussLobatto : public Integrator {
   public:
    /// Constructs a new adaptive Gauss-Lobatto method. `function` is the function to integrate,
    /// `min` and `max` the bounds the integral is computed under.
    AdaptiveGaussLobatto(std::function<double(double)> function, double min, double max) {
        if (max <= min)
            throw std::out_of_range(
                "The maximum value cannot be less than or equal to the minimum value.");
        if (!function) throw std::invalid_argument("The function cannot be null.");
        function_ = std::move(function);
        a_ = min;
        b_ = max;
    }

    /// The unidimensional function to integrate.
    const std::function<double(double)>& function() const { return function_; }

    /// The minimum value under which the integral must be computed.
    double min() const { return a_; }

    /// The maximum value under which the integral must be computed.
    double max() const { return b_; }

    /// Evaluates the integral.
    void integrate() override {
        clear_results();
        validate();

        try {
            double y[13];
            double m = 0.5 * (a_ + b_);
            double h = 0.5 * (b_ - a_);
            double fa = y[0] = function_(a_);
            double fb = y[12] = function_(b_);
            function_evaluations_ += 2;
            for (int i = 1; i < 12; i++) {
                y[i] = function_(m + detail_gauss_lobatto::kNodes[i] * h);
                function_evaluations_++;
            }

            // 4-point Gauss-Lobatto formula
            double i2 = (h / 6.0) * (y[0] + y[12] + 5.0 * (y[4] + y[8]));
            // 7-point Kronrod extension
            double i1 = (h / 1470.0) * (77.0 * (y[0] + y[12]) + 432.0 * (y[2] + y[10]) +
                                        625.0 * (y[4] + y[8]) + 672.0 * y[6]);
            // 13-point Kronrod extension
            double is_ref = h * (0.0158271919734802 * (y[0] + y[12]) +
                                 0.0942738402188500 * (y[1] + y[11]) +
                                 0.155071987336585 * (y[2] + y[10]) +
                                 0.188821573960182 * (y[3] + y[9]) +
                                 0.199773405226859 * (y[4] + y[8]) +
                                 0.224926465333340 * (y[5] + y[7]) + 0.242611071901408 * y[6]);
            double erri1 = std::fabs(i1 - is_ref);
            double erri2 = std::fabs(i2 - is_ref);
            double r = (erri2 != 0.0) ? erri1 / erri2 : 1.0;
            toler_ = (r > 0.0 && r < 1.0) ? relative_tolerance / r : relative_tolerance;
            if (is_ref == 0.0) is_ref = b_ - a_;
            is_ref = std::fabs(is_ref);

            result_ = adaptlob(function_, a_, b_, fa, fb, is_ref);

        } catch (...) {
            status_ = IntegrationStatus::Failure;
            if (report_failure) throw;
        }
    }

   private:
    double a_ = 0.0, b_ = 0.0;
    std::function<double(double)> function_;
    double toler_ = 0.0;

    // See the "PORTING DEVIATION" header note: an instance member replacing the C# `static bool
    // terminate` field.
    bool terminate_ = true;

    /// Helper function for recursion.
    double adaptlob(const std::function<double(double)>& function, double a, double b, double fa,
                    double fb, double is_ref) {
        double m = 0.5 * (a + b);
        double h = 0.5 * (b - a);
        double mll = m - detail_gauss_lobatto::kAlpha * h;
        double ml = m - detail_gauss_lobatto::kBeta * h;
        double mr = m + detail_gauss_lobatto::kBeta * h;
        double mrr = m + detail_gauss_lobatto::kAlpha * h;
        double fmll = function(mll);
        double fml = function(ml);
        double fm = function(m);
        double fmr = function(mr);
        double fmrr = function(mrr);

        iterations_++;
        function_evaluations_ += 5;

        // 4-point Gauss-Lobatto formula
        double i2 = h / 6.0 * (fa + fb + 5.0 * (fml + fmr));
        // 7-point Kronrod extension
        double i1 = h / 1470.0 *
                    (77.0 * (fa + fb) + 432.0 * (fmll + fmrr) + 625.0 * (fml + fmr) + 672.0 * fm);

        // Check iterations
        if (iterations_ >= max_iterations) {
            // Terminate recursion
            update_status(IntegrationStatus::MaximumIterationsReached);
            return i1;
        }
        // Check function evaluations
        if (function_evaluations_ >= max_function_evaluations) {
            // Terminate recursion
            update_status(IntegrationStatus::MaximumFunctionEvaluationsReached);
            return i1;
        }
        // Check tolerance
        if (std::fabs(i1 - i2) <= toler_ * is_ref || mll <= a || b <= mrr) {
            if ((mll <= a || b <= mrr) && terminate_) {
                // Interval contains no more machine numbers
                terminate_ = false;
            }
            // Terminate recursion
            update_status(IntegrationStatus::Success);
            return i1;
        } else {
            // Subdivide interval. See the "SEQUENCING NOTE" header comment: each call is bound to
            // a named local, in order, before the sum -- reproducing C#'s guaranteed left-to-right
            // evaluation of `a + b + c + d + e + f`, which built-in `operator+` does not give for
            // free in C++.
            double r1 = adaptlob(function, a, mll, fa, fmll, is_ref);
            double r2 = adaptlob(function, mll, ml, fmll, fml, is_ref);
            double r3 = adaptlob(function, ml, m, fml, fm, is_ref);
            double r4 = adaptlob(function, m, mr, fm, fmr, is_ref);
            double r5 = adaptlob(function, mr, mrr, fmr, fmrr, is_ref);
            double r6 = adaptlob(function, mrr, b, fmrr, fb, is_ref);
            return r1 + r2 + r3 + r4 + r5 + r6;
        }
    }
};

}  // namespace corehydro::numerics::math::integration
