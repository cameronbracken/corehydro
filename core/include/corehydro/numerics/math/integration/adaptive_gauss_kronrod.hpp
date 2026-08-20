// ported from: Numerics/Mathematics/Integration/AdaptiveGuassKronrod.cs @ 2a0357a
// (upstream's filename carries the "Guass" typo; the class itself is AdaptiveGaussKronrod.)
//
// Adaptive Gauss-Kronrod quadrature over a pair of nested rules: the 10-point Gauss rule and its
// 21-point Kronrod extension (G10K21). The Kronrod rule reuses every Gauss point and adds eleven
// more, giving both a higher-order estimate and an error indicator; intervals are subdivided until
// the difference between the two estimates meets the tolerance.
//
// COMPLETENESS: this file was a MINIMAL port through Phase 10 and the surface phases that followed
// -- it carried the node tables and the recursion as free functions, hard-coding the settings that
// VonMises::CDF (its only caller at the time) needed, because the Integrator base had no other
// consumer. It is now the full class: it derives from the ported Integrator (settings, status,
// evaluation counting, validation), carries MinDepth/MaxDepth/StandardError and the
// StratificationBin overload, and mirrors the C# member and method order so an upstream diff maps
// line for line. The G10K21 node and weight tables are UNCHANGED from the minimal port -- they were
// already verbatim from the C# source and already oracle-backed through VonMises.
//
// The free `integrate(...)` below is a corehydro addition with no C# counterpart, kept from the
// minimal port so its two callers (VonMises::CDF and MultipleGrubbsBeckTest, both of which predate
// the class) do not change. It is a thin wrapper: construct, set, integrate, read Result.
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

/// A class that performs adaptive Gauss-Kronrod integration.
class AdaptiveGaussKronrod : public Integrator {
   public:
    /// Constructs a new adaptive Gauss-Kronrod rule. `function` is the function to integrate,
    /// `min` and `max` the bounds the integral is computed under.
    AdaptiveGaussKronrod(std::function<double(double)> function, double min, double max) {
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

    // Gauss-Kronrod G10K21 nodes and weights (for interval [-1, 1]).
    // 10-point Gauss nodes (symmetric, so only the positive values are stored).
    static constexpr double kXGauss[5] = {
        0.973906528517171720077964012084452,
        0.865063366688984510732096688423493,
        0.679409568299024406234327365114874,
        0.433395394129247190799265943165784,
        0.148874338981631210884826001129720
    };

    // 10-point Gauss weights.
    static constexpr double kWGauss[5] = {
        0.066671344308688137593568809893332,
        0.149451349150580593145776339657697,
        0.219086362515982043995534934228163,
        0.269266719309996355091226921569469,
        0.295524224714752870173892994651338
    };

    // 21-point Kronrod nodes (symmetric, so only the positive values are stored).
    // These include all 10 Gauss nodes plus 11 additional nodes.
    static constexpr double kXKronrod[11] = {
        0.995657163025808080735527280689003,
        0.973906528517171720077964012084452,
        0.930157491355708226001207180059508,
        0.865063366688984510732096688423493,
        0.780817726586416897063717578345042,
        0.679409568299024406234327365114874,
        0.562757134668604683339000099272694,
        0.433395394129247190799265943165784,
        0.294392862701460198131126603103866,
        0.148874338981631210884826001129720,
        0.000000000000000000000000000000000
    };

    // 21-point Kronrod weights.
    static constexpr double kWKronrod[11] = {
        0.011694638867371874278064396062192,
        0.032558162307964727478818972459390,
        0.054755896574351996031381300244580,
        0.075039674810919952767043140916190,
        0.093125454583697605535065465083366,
        0.109387158802297641899210590325805,
        0.123491976262065851077958109831074,
        0.134709217311473325928054001771707,
        0.142775938577060080797094273138717,
        0.147739104901338491374841515972068,
        0.149445554002916905664936468389821
    };

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
            // Initial evaluation using Gauss-Kronrod rule on the whole interval
            auto [kronrod_result, gauss_result] = evaluate_gauss_kronrod(a_, b_);

            // Recursively sub-divide
            result_ = adaptive_gk(function_, a_, b_, max_depth, kronrod_result, gauss_result, a_, b_);

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
                // Initial evaluation using Gauss-Kronrod rule on the bin interval
                double bin_a = bins[i].lower_bound();
                double bin_b = bins[i].upper_bound();
                auto [kronrod_result, gauss_result] = evaluate_gauss_kronrod(bin_a, bin_b);

                // Recursively sub-divide
                mu += adaptive_gk(function_, bin_a, bin_b, max_depth, kronrod_result, gauss_result,
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

    /// Evaluates the Gauss-Kronrod G10K21 rule over the interval [a, b], returning
    /// (Kronrod estimate, Gauss estimate).
    std::pair<double, double> evaluate_gauss_kronrod(double a, double b) {
        double center = 0.5 * (a + b);
        double half_length = 0.5 * (b - a);

        double result_gauss = 0.0;
        double result_kronrod = 0.0;

        // Evaluate at center point (x = 0)
        double f0 = function_(center);
        result_kronrod += kWKronrod[10] * f0;
        function_evaluations_++;

        // Evaluate at symmetric pairs of points
        for (int i = 0; i < 10; i++) {
            double abscissa = half_length * kXKronrod[i];
            double fval1 = function_(center - abscissa);
            double fval2 = function_(center + abscissa);
            double fsum = fval1 + fval2;

            result_kronrod += kWKronrod[i] * fsum;

            // Add to Gauss result if this is a Gauss node.
            // Gauss nodes are at indices 1, 3, 5, 7, 9 of the Kronrod array
            if (i % 2 == 1) {
                int gauss_index = i / 2;
                result_gauss += kWGauss[gauss_index] * fsum;
            }

            function_evaluations_ += 2;
        }

        result_gauss *= half_length;
        result_kronrod *= half_length;

        return {result_kronrod, result_gauss};
    }

    /// A helper function for adaptive Gauss-Kronrod integration. Subdivides [a, b] until the error
    /// between the Gauss and Kronrod estimates is small enough, and returns the Kronrod estimate.
    /// `a0`/`b0` are the original bounds, used to scale the relative tolerance by interval length.
    double adaptive_gk(const std::function<double(double)>& f, double a, double b, int depth,
                       double kronrod_whole, double gauss_whole, double a0, double b0) {
        // Error estimate: difference between Kronrod and Gauss results
        double error = std::fabs(kronrod_whole - gauss_whole);

        // Scaled tolerance based on interval length relative to original domain
        double tolerance_scaled = relative_tolerance * std::fabs(b - a) / std::fabs(b0 - a0);

        // Absolute and Relative tolerance checks
        bool absolute_tolerance_reached = error <= absolute_tolerance;
        bool relative_tolerance_reached = error <= tolerance_scaled * std::fabs(kronrod_whole);

        // Check if convergence criteria are met
        if (depth <= 0 || std::fabs(a - b) <= kDoubleMachineEpsilon ||
            function_evaluations_ >= max_function_evaluations ||
            (function_evaluations_ >= min_function_evaluations && depth <= max_depth - min_depth &&
             (absolute_tolerance_reached || relative_tolerance_reached))) {
            // Convergence is reached
            squared_error_ += error * error;  // Accumulate squared errors
            return kronrod_whole;             // Return the more accurate Kronrod estimate
        } else {
            // Subdivide the interval at the midpoint
            double m = (a + b) / 2.0;

            // Evaluate Gauss-Kronrod on left half
            auto [kronrod_left, gauss_left] = evaluate_gauss_kronrod(a, m);

            // Evaluate Gauss-Kronrod on right half
            auto [kronrod_right, gauss_right] = evaluate_gauss_kronrod(m, b);

            // Recursively subdivide the intervals and accumulate results
            auto left_result = adaptive_gk(f, a, m, depth - 1, kronrod_left, gauss_left, a0, b0);
            auto right_result = adaptive_gk(f, m, b, depth - 1, kronrod_right, gauss_right, a0, b0);

            return left_result + right_result;
        }
    }
};

/// corehydro addition, no C# counterpart. Integrates `f` over [a, b] with the adaptive G10K21 rule
/// and returns the result, for the two ported callers that predate the class (VonMises::CDF and
/// MultipleGrubbsBeckTest). Every default matches the C# Integrator base-class default.
inline double integrate(
        const std::function<double(double)>& f,
        double a, double b,
        double abs_tol     = 1e-8,
        double rel_tol     = 1e-8,
        int    max_depth   = 100,
        int    max_evals   = 10000000) {
    AdaptiveGaussKronrod agk(f, a, b);
    agk.absolute_tolerance = abs_tol;
    agk.relative_tolerance = rel_tol;
    agk.max_depth = max_depth;
    agk.max_function_evaluations = max_evals;
    agk.integrate();
    return agk.result();
}

}  // namespace corehydro::numerics::math::integration
