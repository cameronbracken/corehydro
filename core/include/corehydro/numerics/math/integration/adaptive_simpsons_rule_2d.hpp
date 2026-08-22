// ported from: Numerics/Mathematics/Integration/AdaptiveSimpsonsRule2D.cs @ 2a0357a
//
// Adaptive Simpson's rule extended to two dimensions: the rectangular domain is evaluated with
// the tensor-product 3x3-point Simpson's rule (nine function evaluations: four corners, four
// edge midpoints, one center), compared against the sum of the four quadrant sub-evaluations,
// and subdivided into quadrants when the two disagree by more than tolerance. Same style as
// AdaptiveSimpsonsRule: derives from Integrator, assigns Status directly and rethrows in its own
// catch.
#pragma once
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>

#include "corehydro/numerics/math/integration/support/integration_status.hpp"
#include "corehydro/numerics/math/integration/support/integrator.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::integration {

/// A class that performs adaptive Simpson's integration in two dimensions.
class AdaptiveSimpsonsRule2D : public Integrator {
   public:
    /// Constructs a new adaptive Simpson's rule for 2D integration. `function` is the function
    /// to integrate, `min_x`/`max_x`/`min_y`/`max_y` the bounds the integral is computed under.
    AdaptiveSimpsonsRule2D(std::function<double(double, double)> function, double min_x,
                           double max_x, double min_y, double max_y) {
        if (max_x <= min_x)
            throw std::out_of_range(
                "The maximum x-value cannot be less than or equal to the minimum x-value.");
        if (max_y <= min_y)
            throw std::out_of_range(
                "The maximum y-value cannot be less than or equal to the minimum y-value.");
        if (!function) throw std::invalid_argument("The function cannot be null.");
        function_ = std::move(function);
        ax_ = min_x;
        bx_ = max_x;
        ay_ = min_y;
        by_ = max_y;
    }

   private:
    // 2D bounds
    double ax_ = 0.0, bx_ = 0.0, ay_ = 0.0, by_ = 0.0, squared_error_ = 0.0;
    std::function<double(double, double)> function_;

   public:
    /// The 2D function to integrate.
    const std::function<double(double, double)>& function() const { return function_; }

    /// The minimum x-value under which the integral must be computed.
    double min_x() const { return ax_; }

    /// The maximum x-value under which the integral must be computed.
    double max_x() const { return bx_; }

    /// The minimum y-value under which the integral must be computed.
    double min_y() const { return ay_; }

    /// The maximum y-value under which the integral must be computed.
    double max_y() const { return by_; }

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
            // Initial evaluation of Simpson's rule on the entire 2D domain
            double mx = (ax_ + bx_) / 2.0;
            double my = (ay_ + by_) / 2.0;

            // Evaluate at the four corners
            double f_ax_ay = function_(ax_, ay_);
            double f_bx_ay = function_(bx_, ay_);
            double f_ax_by = function_(ax_, by_);
            double f_bx_by = function_(bx_, by_);

            // Evaluate at edge midpoints
            double f_mx_ay = function_(mx, ay_);
            double f_mx_by = function_(mx, by_);
            double f_ax_my = function_(ax_, my);
            double f_bx_my = function_(bx_, my);

            // Evaluate at the center
            double f_mx_my = function_(mx, my);

            function_evaluations_ += 9;  // Count all 9 evaluations

            // 2D Simpson's rule: (hx/3) * (hy/3) * weighted sum
            // Apply Simpson's in x for each y, then Simpson's in y
            double hx = (bx_ - ax_) / 2.0;
            double hy = (by_ - ay_) / 2.0;
            double whole = (hx / 3.0) * (hy / 3.0) *
                           ((f_ax_ay + 4 * f_mx_ay + f_bx_ay) +
                            4 * (f_ax_my + 4 * f_mx_my + f_bx_my) +
                            (f_ax_by + 4 * f_mx_by + f_bx_by));

            // Recursively sub-divide
            result_ = adaptive_simpsons_2d(function_, ax_, bx_, ay_, by_, f_ax_ay, f_bx_ay,
                                           f_ax_by, f_bx_by, f_mx_ay, f_mx_by, f_ax_my, f_bx_my,
                                           f_mx_my, max_depth, whole, mx, my, ax_, bx_, ay_, by_);

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

   private:
    double standard_error_ = 0.0;

    /// A helper function to perform adaptive Simpson's rule in two dimensions. Recursively
    /// subdivides the 2D integration domain into quadrants and applies Simpson's rule to each.
    /// `ax`/`bx`/`ay`/`by` are the bounds of the current rectangular interval, the nine `f_*`
    /// parameters its corner/edge-midpoint/center function values, `depth` the recursions
    /// remaining, `whole` the Simpson evaluation already computed on the current rectangle,
    /// `mx`/`my` its midpoint, and `ax0`/`bx0`/`ay0`/`by0` the ORIGINAL domain bounds, used to
    /// scale the relative tolerance by the area ratio.
    double adaptive_simpsons_2d(const std::function<double(double, double)>& f, double ax,
                                double bx, double ay, double by, double f_ax_ay, double f_bx_ay,
                                double f_ax_by, double f_bx_by, double f_mx_ay, double f_mx_by,
                                double f_ax_my, double f_bx_my, double f_mx_my, int depth,
                                double whole, double mx, double my, double ax0, double bx0,
                                double ay0, double by0) {
        // Midpoints for subdividing into four quadrants
        double lmx = (ax + mx) / 2.0;  // left midpoint in X
        double rmx = (mx + bx) / 2.0;  // right midpoint in X
        double lmy = (ay + my) / 2.0;  // lower midpoint in Y
        double rmy = (my + by) / 2.0;  // upper midpoint in Y

        // Evaluate at new edge midpoints and quadrant centers
        double f_lmx_ay = f(lmx, ay);
        double f_rmx_ay = f(rmx, ay);
        double f_lmx_by = f(lmx, by);
        double f_rmx_by = f(rmx, by);

        double f_ax_lmy = f(ax, lmy);
        double f_bx_lmy = f(bx, lmy);
        double f_ax_rmy = f(ax, rmy);
        double f_bx_rmy = f(bx, rmy);

        double f_mx_lmy = f(mx, lmy);
        double f_mx_rmy = f(mx, rmy);
        double f_lmx_my = f(lmx, my);
        double f_rmx_my = f(rmx, my);

        double f_lmx_lmy = f(lmx, lmy);
        double f_rmx_lmy = f(rmx, lmy);
        double f_lmx_rmy = f(lmx, rmy);
        double f_rmx_rmy = f(rmx, rmy);

        function_evaluations_ += 16;  // Count the 16 new function evaluations

        // Calculate Simpson's rule for each of the four quadrants
        double hx_left = (mx - ax) / 2.0;
        double hx_right = (bx - mx) / 2.0;
        double hy_lower = (my - ay) / 2.0;
        double hy_upper = (by - my) / 2.0;

        // Bottom-left quadrant: (ax, mx) x (ay, my)
        double q1 = (hx_left / 3.0) * (hy_lower / 3.0) *
                    ((f_ax_ay + 4 * f_lmx_ay + f_mx_ay) +
                     4 * (f_ax_lmy + 4 * f_lmx_lmy + f_mx_lmy) +
                     (f_ax_my + 4 * f_lmx_my + f_mx_my));

        // Bottom-right quadrant: (mx, bx) x (ay, my)
        double q2 = (hx_right / 3.0) * (hy_lower / 3.0) *
                    ((f_mx_ay + 4 * f_rmx_ay + f_bx_ay) +
                     4 * (f_mx_lmy + 4 * f_rmx_lmy + f_bx_lmy) +
                     (f_mx_my + 4 * f_rmx_my + f_bx_my));

        // Top-left quadrant: (ax, mx) x (my, by)
        double q3 = (hx_left / 3.0) * (hy_upper / 3.0) *
                    ((f_ax_my + 4 * f_lmx_my + f_mx_my) +
                     4 * (f_ax_rmy + 4 * f_lmx_rmy + f_mx_rmy) +
                     (f_ax_by + 4 * f_lmx_by + f_mx_by));

        // Top-right quadrant: (mx, bx) x (my, by)
        double q4 = (hx_right / 3.0) * (hy_upper / 3.0) *
                    ((f_mx_my + 4 * f_rmx_my + f_bx_my) +
                     4 * (f_mx_rmy + 4 * f_rmx_rmy + f_bx_rmy) +
                     (f_mx_by + 4 * f_rmx_by + f_bx_by));

        double sum = q1 + q2 + q3 + q4;

        // Calculate the error
        double error = sum - whole;
        double delta = error / 15.0;  // Richardson correction

        // Richardson-based convergence tolerance (scaled by domain-area ratio)
        double area_ratio = ((bx - ax) * (by - ay)) / ((bx0 - ax0) * (by0 - ay0));
        double tolerance_scaled = 15.0 * relative_tolerance * area_ratio;

        // Absolute and Relative tolerance checks
        bool absolute_tolerance_reached = std::fabs(error) <= absolute_tolerance;
        bool relative_tolerance_reached = std::fabs(error) <= tolerance_scaled;

        // Check if convergence criteria are met
        if (depth <= 0 || std::fabs(ax - bx) <= kDoubleMachineEpsilon ||
            std::fabs(ay - by) <= kDoubleMachineEpsilon ||
            function_evaluations_ >= max_function_evaluations ||
            (function_evaluations_ >= min_function_evaluations && depth <= max_depth - min_depth &&
             (absolute_tolerance_reached || relative_tolerance_reached))) {
            // Convergence is reached, accumulate the error
            squared_error_ += delta * delta;
            return sum + delta;
        } else {
            // Recursively subdivide the four quadrants
            double q1_result = adaptive_simpsons_2d(
                    f, ax, mx, ay, my, f_ax_ay, f_mx_ay, f_ax_my, f_mx_my, f_lmx_ay, f_lmx_my,
                    f_ax_lmy, f_mx_lmy, f_lmx_lmy, depth - 1, q1, lmx, lmy, ax0, bx0, ay0, by0);

            double q2_result = adaptive_simpsons_2d(
                    f, mx, bx, ay, my, f_mx_ay, f_bx_ay, f_mx_my, f_bx_my, f_rmx_ay, f_rmx_my,
                    f_mx_lmy, f_bx_lmy, f_rmx_lmy, depth - 1, q2, rmx, lmy, ax0, bx0, ay0, by0);

            double q3_result = adaptive_simpsons_2d(
                    f, ax, mx, my, by, f_ax_my, f_mx_my, f_ax_by, f_mx_by, f_lmx_my, f_lmx_by,
                    f_ax_rmy, f_mx_rmy, f_lmx_rmy, depth - 1, q3, lmx, rmy, ax0, bx0, ay0, by0);

            double q4_result = adaptive_simpsons_2d(
                    f, mx, bx, my, by, f_mx_my, f_bx_my, f_mx_by, f_bx_by, f_rmx_my, f_rmx_by,
                    f_mx_rmy, f_bx_rmy, f_rmx_rmy, depth - 1, q4, rmx, rmy, ax0, bx0, ay0, by0);

            return q1_result + q2_result + q3_result + q4_result;
        }
    }
};

}  // namespace corehydro::numerics::math::integration
