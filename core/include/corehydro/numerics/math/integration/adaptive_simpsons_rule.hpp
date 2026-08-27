// ported from: Numerics/Mathematics/Integration/AdaptiveSimpsonsRule.cs @ 2a0357a
//
// Adaptive Simpson's rule: the three-point Simpson estimate over [a, b] is compared against the
// sum of the two-panel estimate over [a, m] and [m, b]; if the two disagree by more than
// tolerance, the interval is subdivided and the recursion repeats on each half. Mirrors
// AdaptiveGaussKronrod's style: derives from the ported Integrator, assigns Status directly and
// rethrows in its own catch (rather than routing through Integrator::update_status), and carries
// MinDepth/MaxDepth/StandardError and the StratificationBin overload.
#pragma once
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/integration/support/integration_status.hpp"
#include "corehydro/numerics/math/integration/support/integrator.hpp"
#include "corehydro/numerics/sampling/stratification_bin.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::integration {

/// A class that performs adaptive Simpson's integration.
class AdaptiveSimpsonsRule : public Integrator {
   public:
    /// Constructs a new adaptive Simpson's rule. `function` is the function to integrate, `min`
    /// and `max` the bounds the integral is computed under.
    AdaptiveSimpsonsRule(std::function<double(double)> function, double min, double max) {
        if (max <= min)
            throw std::out_of_range(
                "The maximum value cannot be less than or equal to the minimum value.");
        if (!function) throw std::invalid_argument("The function cannot be null.");
        function_ = std::move(function);
        a_ = min;
        b_ = max;
    }

   private:
    double a_ = 0.0, b_ = 0.0, squared_error_ = 0.0;
    std::function<double(double)> function_;

   public:
    /// The unidimensional function to integrate.
    const std::function<double(double)>& function() const { return function_; }

    /// The minimum value under which the integral must be computed.
    double min() const { return a_; }

    /// The maximum value under which the integral must be computed.
    double max() const { return b_; }

    /// The minimum recursion depth. Default = 0.
    int min_depth = 0;

    /// The maximum recursion depth. Default = 100.
    int max_depth = 100;

    /// Returns an approximate measure of the standard error of the integration.
    double standard_error() const { return standard_error_; }

    /// Evaluates the integral.
    void integrate() override {
        squared_error_ = 0;
        standard_error_ = 0;
        clear_results();
        validate();

        try {
            // First evaluation of Simpson's rule on the whole interval
            double m = (a_ + b_) / 2.0;
            double fa = function_(a_);
            double fb = function_(b_);
            double fm = function_(m);
            function_evaluations_ += 3;  // Count the three evaluations: fa, fb, fm
            double whole = std::fabs(b_ - a_) / 6.0 * (fa + 4.0 * fm + fb);

            // Recursively sub-divide
            result_ =
                    adaptive_simpsons(function_, a_, fa, b_, fb, max_depth, whole, m, fm, a_, b_);

            // Standard error calculated after recursion completes
            standard_error_ = std::sqrt(squared_error_);

            if (function_evaluations_ >= max_function_evaluations) {
                status_ = IntegrationStatus::MaximumFunctionEvaluationsReached;
            } else {
                status_ = IntegrationStatus::Success;
            }
        } catch (...) {
            status_ = IntegrationStatus::Failure;
            if (report_failure) throw;
        }
    }

    /// Evaluates the integral over user-provided stratification bins.
    void integrate(const std::vector<sampling::StratificationBin>& bins) {
        squared_error_ = 0;
        standard_error_ = 0;
        clear_results();
        validate();

        try {
            double mu = 0;
            for (std::size_t i = 0; i < bins.size(); i++) {
                // First evaluation of Simpson's rule on the bin interval
                double bin_a = bins[i].lower_bound();
                double bin_b = bins[i].upper_bound();
                double m = (bin_a + bin_b) / 2.0;
                double fa = function_(bin_a);
                double fb = function_(bin_b);
                double fm = function_(m);
                function_evaluations_ += 3;  // Count the three evaluations: fa, fb, fm
                double whole = std::fabs(bin_b - bin_a) / 6.0 * (fa + 4.0 * fm + fb);

                // Recursively sub-divide
                mu += adaptive_simpsons(function_, bin_a, fa, bin_b, fb, max_depth, whole, m, fm,
                                        bin_a, bin_b);
            }

            // Final result and standard error
            result_ = mu;
            standard_error_ = std::sqrt(squared_error_);

            if (function_evaluations_ >= max_function_evaluations) {
                status_ = IntegrationStatus::MaximumFunctionEvaluationsReached;
            } else {
                status_ = IntegrationStatus::Success;
            }
        } catch (...) {
            status_ = IntegrationStatus::Failure;
            if (report_failure) throw;
        }
    }

   private:
    double standard_error_ = 0.0;

    /// A helper function to Integrate(). `a`/`fa`/`b`/`fb` are the current sub-interval and its
    /// function values, `depth` the recursions remaining, `whole` the three-point Simpson
    /// evaluation already computed on [a, b], `m`/`fm` its midpoint, and `a0`/`b0` the ORIGINAL
    /// bounds, used to scale the relative tolerance by interval length.
    double adaptive_simpsons(const std::function<double(double)>& f, double a, double fa,
                             double b, double fb, int depth, double whole, double m, double fm,
                             double a0, double b0) {
        double h = (b - a) * 0.5;
        double lm = a + h * 0.5;      // left mid
        double rm = a + h + h * 0.5;  // right mid
        double flm = f(lm), frm = f(rm);
        function_evaluations_ += 2;

        double left = h / 6.0 * (fa + 4.0 * flm + fm);
        double right = h / 6.0 * (fm + 4.0 * frm + fb);

        // Calculate error for current interval
        double error = (left + right - whole);
        double delta = error / 15.0;  // Richardson correction

        // Richardson-based convergence tolerance
        double tolerance_scaled = 15.0 * (relative_tolerance * std::fabs(b - a) / std::fabs(b0 - a0));

        // Absolute and Relative tolerance checks
        bool absolute_tolerance_reached = std::fabs(error) <= absolute_tolerance;
        bool relative_tolerance_reached = std::fabs(error) <= tolerance_scaled;

        // Check if convergence criteria are met
        if (depth <= 0 || std::fabs(a - b) <= kDoubleMachineEpsilon ||
            function_evaluations_ >= max_function_evaluations ||
            (function_evaluations_ >= min_function_evaluations && depth <= max_depth - min_depth &&
             (absolute_tolerance_reached || relative_tolerance_reached))) {
            // Convergence is reached
            squared_error_ += delta * delta;  // Accumulate squared errors
            return left + right + delta;
        } else {
            // Recursively subdivide the intervals and accumulate results
            double left_result = adaptive_simpsons(f, a, fa, m, fm, depth - 1, left, lm, flm, a0, b0);
            double right_result = adaptive_simpsons(f, m, fm, b, fb, depth - 1, right, rm, frm, a0, b0);
            return left_result + right_result;
        }
    }
};

}  // namespace corehydro::numerics::math::integration
