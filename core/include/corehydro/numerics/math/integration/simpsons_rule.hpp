// ported from: Numerics/Mathematics/Integration/SimpsonsRule.cs @ 2a0357a
//
// Simpson's rule integration by repeated trapezoidal refinement combined with Richardson
// extrapolation: `Next()` doubles the number of trapezoidal panels each call (the classic
// Numerical Recipes `qtrap` refinement), and each new trapezoidal estimate `st` is combined
// with the PREVIOUS one `ost` via `s = (4*st - ost) / 3`, the Simpson correction that cancels
// the trapezoidal rule's leading error term. Iteration continues until `EvaluateConvergence`
// (both absolute and relative tolerance) is satisfied, `MaxFunctionEvaluations` is hit, or
// `MaxIterations` is exhausted.
//
// This is the first integrator in this port whose `Integrate()` routes exceptions through the
// base `Integrator::update_status` (AdaptiveGaussKronrod assigns Status directly and rethrows in
// its own catch instead, exactly as its C# source does; see integrator.hpp's header comment).
#pragma once
#include <functional>
#include <stdexcept>
#include <utility>

#include "corehydro/numerics/math/integration/support/integration_status.hpp"
#include "corehydro/numerics/math/integration/support/integrator.hpp"

namespace corehydro::numerics::math::integration {

/// A class for Simpson's rule integration. Integration steps are refined until convergence.
class SimpsonsRule : public Integrator {
   public:
    /// Constructs a new Simpson's rule class. `function` is the function to integrate, `min` and
    /// `max` the bounds the integral is computed under.
    SimpsonsRule(std::function<double(double)> function, double min, double max) {
        if (!function) throw std::invalid_argument("The function cannot be null.");
        if (max <= min)
            throw std::out_of_range(
                "The maximum value cannot be less than or equal to the minimum value.");
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
            double s = 0.0, os = 0.0;
            double st = 0.0, ost = 0.0;
            for (int i = 0; i < max_iterations; i++) {
                st = next();
                s = (4.0 * st - ost) / 3.0;

                // Check function evaluations
                if (function_evaluations_ >= max_function_evaluations) {
                    update_status(IntegrationStatus::MaximumFunctionEvaluationsReached);
                    return;
                }

                // Check convergence
                if (i > 2) {
                    if (evaluate_convergence(os, s) || (s == 0.0 && os == 0.0)) {
                        result_ = s;
                        update_status(IntegrationStatus::Success);
                        return;
                    }
                }
                os = s;
                ost = st;
            }
            // If we get to here, then the maximum number of steps were reached before converging.
            update_status(IntegrationStatus::MaximumIterationsReached);
        } catch (...) {
            update_status(IntegrationStatus::Failure, std::current_exception());
        }
    }

   private:
    double a_ = 0.0, b_ = 0.0, s_ = 0.0;
    std::function<double(double)> function_;

    /// Returns the value of the integral at the nth step of refinement.
    double next() {
        iterations_++;
        if (iterations_ == 1) {
            s_ = 0.5 * (b_ - a_) * (function_(a_) + function_(b_));
            function_evaluations_ += 2;
        } else {
            int it = 1;
            for (int j = 1; j < iterations_ - 1; j++) it <<= 1;
            double tnm = it;
            double del = (b_ - a_) / tnm;
            double x = a_ + 0.5 * del;
            double sum = 0.0;
            for (int j = 0; j < it; j++, x += del) {
                sum += function_(x);
                function_evaluations_ += 1;
            }
            s_ = 0.5 * (s_ + (b_ - a_) * sum / tnm);
        }
        return s_;
    }
};

}  // namespace corehydro::numerics::math::integration
