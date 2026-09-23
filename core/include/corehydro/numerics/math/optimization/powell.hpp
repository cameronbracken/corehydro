// ported from: Numerics/Mathematics/Optimization/Local/Powell.cs @ 7e8e8d1
//
// Powell's direction-set optimization method (Numerical Recipes, Press et al.; see the
// C# file's references): minimizes without derivatives by bi-directionally line-searching
// along a maintained set of direction vectors via Brent's method, replacing directions
// with the average displacement as the search proceeds. A REAL `Optimizer` subclass
// deriving from the ported base exactly as bfgs.hpp / differential_evolution.hpp do (NOT
// the Phase 4 estimation/support/optimizer_adapters.hpp `detail::` stopgap).
//
// Transcription notes:
//
// 1. Ctor validation: the C# throws ArgumentOutOfRangeException for the three checks
//    (length; upper < lower; initial within bounds). Per the repo-wide stand-in
//    documented in optimizer.hpp's file header (and used by BFGS), every C#
//    ArgumentException-family throw ports as `ArgumentException` (kind Other) so the
//    base's minimize()/maximize() catch filter behaves exactly like the C#'s.
//
// 2. LineMinimization constructs a standalone `BrentSearch` over the 1-D slice function
//    -- `new BrentSearch(func, 0d, 1d)` with the Powell instance's tolerances, then
//    `Bracket(0.1)`, then `Minimize()` -- exactly as the C#. The ported BrentSearch is
//    the deliberately-standalone Phase 0 class (see brent_search.hpp's header), extended
//    additively in this task with `bracket()` and `best_fitness()`; its one documented
//    shape drift from the C# is that, not deriving from the Optimizer base, it performs
//    NO post-success Hessian computation. In the C#, BrentSearch.Minimize() numerically
//    differentiates the slice function at the line minimum (ComputeHessian defaults
//    true), and those probe calls route through Powell's Evaluate -- so the C# Powell
//    logs a few extra function evaluations per line search and its best-parameter
//    tracking sees the probe points. Neither effect alters the accepted line minimum
//    (the Hessian runs AFTER Brent's loop) and any probe-point "improvement" is O(h^2)
//    around an already-converged 1-D minimum -- orders of magnitude below the 1E-4
//    upstream oracle tolerances, which reproduce (see test_optimizers_local.cpp). A
//    second, Task-11-added shape drift: a `bracket()` failure (see brent_search.hpp's
//    header for the v2.1.4 hardening and its Status-surface divergence) now always
//    propagates here as a thrown C++ exception, rather than the narrower C# case where a
//    Bracket()-originated MaximumIterationsReached is silently swallowed by this
//    Optimizer's own catch filter. Not reachable by any current fixture -- every
//    LineMinimization slice objective brackets in a handful of iterations.
//
// 3. `ximat` (C# `double[,]`) ports as a plain vector-of-vectors, mirroring the C# 2-D
//    array shape rather than reaching for linalg::Matrix.
#pragma once
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::optimization {

class Powell : public Optimizer {
   public:
    // Construct a new Powell optimization method.
    //   objective_function:   the objective function to evaluate.
    //   number_of_parameters: the number of parameters in the objective function.
    //   initial_values:       an array of initial values to evaluate.
    //   lower_bounds:         an array of lower bounds (inclusive) of the interval
    //                         containing the optimal point.
    //   upper_bounds:         an array of upper bounds (inclusive) of the interval
    //                         containing the optimal point.
    Powell(Objective objective_function, int number_of_parameters,
           std::vector<double> initial_values, std::vector<double> lower_bounds,
           std::vector<double> upper_bounds)
        : Optimizer(std::move(objective_function), number_of_parameters) {
        // Check if the length of the initial, lower and upper bounds equal the number of
        // parameters.
        if (static_cast<int>(initial_values.size()) != number_of_parameters ||
            static_cast<int>(lower_bounds.size()) != number_of_parameters ||
            static_cast<int>(upper_bounds.size()) != number_of_parameters) {
            throw ArgumentException(
                "The initial values and lower and upper bounds must be the same length as the "
                "number of parameters.");
        }
        // Check if the initial values are between the lower and upper values.
        for (std::size_t j = 0; j < initial_values.size(); j++) {
            if (upper_bounds[j] < lower_bounds[j]) {
                throw ArgumentException("The upper bound cannot be less than the lower bound.");
            }
            if (initial_values[j] < lower_bounds[j] || initial_values[j] > upper_bounds[j]) {
                throw ArgumentException(
                    "The initial values must be between the upper and lower bounds.");
            }
        }
        initial_values_ = std::move(initial_values);
        lower_bounds_ = std::move(lower_bounds);
        upper_bounds_ = std::move(upper_bounds);
    }

    // An array of initial values to evaluate.
    const std::vector<double>& initial_values() const { return initial_values_; }

    // An array of lower bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }

    // An array of upper bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

   protected:
    const std::vector<double>& parameter_lower_bounds() const override { return lower_bounds_; }
    const std::vector<double>& parameter_upper_bounds() const override { return upper_bounds_; }

    void optimize() override {
        // Set variables
        int i, j, D = number_of_parameters_, ibig;
        bool cancel = false;
        double t, fret, fp, fptt, delta;
        auto p = initial_values_;
        std::vector<double> pt(static_cast<std::size_t>(D));
        std::vector<double> ptt(static_cast<std::size_t>(D));
        std::vector<double> xi(static_cast<std::size_t>(D));  // Direction vector
        // Set the initial matrix for directions
        // and save the initial point
        std::vector<std::vector<double>> ximat(static_cast<std::size_t>(D),
                                               std::vector<double>(static_cast<std::size_t>(D)));
        for (i = 0; i < D; i++) {
            ximat[i][i] = 1.0;
            pt[i] = p[i];
        }
        // initial function evaluation
        fret = evaluate(p, cancel);
        while (iterations_ < max_iterations) {
            fp = fret;
            ibig = 0;
            delta = 0.0;  // Will be the biggest function decrease.
            // In each iteration, loop over all directions in the set.
            for (i = 0; i < D; i++) {
                // Copy the direction
                for (j = 0; j < D; j++) xi[j] = ximat[j][i];
                fptt = fret;
                fret = line_minimization(p, xi, cancel);
                if (cancel == true) return;
                // And record it if it is the larges decrease so far.
                if (fptt - fret > delta) {
                    delta = fptt - fret;
                    ibig = i + 1;
                }
            }
            // Check convergence
            if (check_convergence(fp, fret)) {
                update_status(OptimizationStatus::Success);
                return;
            }
            // Construct the extrapolated point and save the average direction moved.
            // Save the old starting point.
            bool extrapolation_is_feasible = true;
            for (j = 0; j < D; j++) {
                ptt[j] = 2.0 * p[j] - pt[j];
                if (ptt[j] < lower_bounds_[static_cast<std::size_t>(j)] ||
                    ptt[j] > upper_bounds_[static_cast<std::size_t>(j)])
                    extrapolation_is_feasible = false;
                xi[j] = p[j] - pt[j];
                pt[j] = p[j];
            }
            fptt = extrapolation_is_feasible ? evaluate(ptt, cancel)
                                             : std::numeric_limits<double>::infinity();
            if (cancel == true) return;
            if (fptt < fp) {
                t = 2.0 * (fp - 2.0 * fret + fptt) * sqr(fp - fret - delta) -
                    delta * sqr(fp - fptt);
                if (t < 0.0) {
                    // Move to the minimum of the new direction and save the new direction
                    fret = line_minimization(p, xi, cancel);
                    if (cancel == true) return;
                    for (j = 0; j < D; j++) {
                        ximat[j][ibig - 1] = ximat[j][D - 1];
                        ximat[j][D - 1] = xi[j];
                    }
                }
            }

            iterations_ += 1;
        }

        // If we made it to here, the maximum iterations were reached.
        update_status(OptimizationStatus::MaximumIterationsReached);
    }

   private:
    std::vector<double> initial_values_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;

    // Auxiliary line minimization routine.
    //   start_point: the initial point (moved to the line minimum in place).
    //   direction:   the initial direction (replaced by the actual displacement).
    //   cancel:      determines if the solver should be canceled.
    double line_minimization(std::vector<double>& start_point, std::vector<double>& direction,
                             bool& cancel) {
        int D = number_of_parameters_;
        bool c = cancel;
        auto interval = feasible_step_interval(start_point, direction);
        double alpha_min = interval.first;
        double alpha_max = interval.second;

        double zero_step = evaluate(start_point, c);
        cancel = c;
        if (cancel) return std::numeric_limits<double>::quiet_NaN();
        if (alpha_min == 0.0 && alpha_max == 0.0) return zero_step;

        auto func = [this, &start_point, &direction, D, alpha_min, alpha_max, &c](double alpha) {
            double step = std::max(alpha_min, std::min(alpha_max, alpha));
            std::vector<double> x(static_cast<std::size_t>(D));
            for (int i = 0; i < D; i++)
                x[static_cast<std::size_t>(i)] = repair_parameter(
                    start_point[static_cast<std::size_t>(i)] +
                        step * direction[static_cast<std::size_t>(i)],
                    lower_bounds_[static_cast<std::size_t>(i)],
                    upper_bounds_[static_cast<std::size_t>(i)]);
            return evaluate(x, c);
        };

        BrentSearch bracketing(func, 0.0, 1.0);
        bracketing.relative_tolerance = relative_tolerance;
        bracketing.absolute_tolerance = absolute_tolerance;
        bracketing.bracket(bracketing_step(alpha_min, alpha_max));
        cancel = c;
        if (cancel) return std::numeric_limits<double>::quiet_NaN();
        double lower = std::max(alpha_min, std::min(alpha_max, bracketing.lower_bound()));
        double upper = std::max(alpha_min, std::min(alpha_max, bracketing.upper_bound()));

        BrentSearch brent(func, lower, upper);
        brent.relative_tolerance = relative_tolerance;
        brent.absolute_tolerance = absolute_tolerance;
        brent.minimize();
        cancel = c;
        if (cancel) return std::numeric_limits<double>::quiet_NaN();
        double xmin = brent.best_parameter();
        double fmin = brent.best_fitness();

        if (upper == alpha_max && upper != 0.0) {
            double at_end = func(alpha_max);
            if (at_end < fmin) { xmin = alpha_max; fmin = at_end; }
        }
        if (lower == alpha_min && lower != 0.0) {
            double at_end = func(alpha_min);
            if (at_end < fmin) { xmin = alpha_min; fmin = at_end; }
        }
        cancel = c;
        if (cancel) return std::numeric_limits<double>::quiet_NaN();
        if (!(fmin < zero_step)) return zero_step;

        for (int j = 0; j < number_of_parameters_; j++) {
            direction[j] *= xmin;
            start_point[j] += direction[j];
            // Make sure the parameter is within bounds
            start_point[j] = repair_parameter(start_point[j], lower_bounds_[j], upper_bounds_[j]);
        }
        return fmin;
    }

    static double bracketing_step(double alpha_min, double alpha_max) {
        constexpr double default_step = 0.1;
        if (alpha_max >= default_step) return default_step;
        if (alpha_min <= -default_step) return -default_step;
        double wider = alpha_max >= -alpha_min ? alpha_max : alpha_min;
        double half = 0.5 * wider;
        return half == 0.0 ? wider : half;
    }

    std::pair<double, double> feasible_step_interval(
        const std::vector<double>& start_point, const std::vector<double>& direction) const {
        double alpha_min = -std::numeric_limits<double>::infinity();
        double alpha_max = std::numeric_limits<double>::infinity();
        bool moves = false;
        for (int i = 0; i < number_of_parameters_; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            double d = direction[ui];
            if (d == 0.0) continue;
            moves = true;
            double to_upper = (upper_bounds_[ui] - start_point[ui]) / d;
            double to_lower = (lower_bounds_[ui] - start_point[ui]) / d;
            double high = d > 0.0 ? to_upper : to_lower;
            double low = d > 0.0 ? to_lower : to_upper;
            if (high < alpha_max) alpha_max = high;
            if (low > alpha_min) alpha_min = low;
        }
        if (!moves) return {0.0, 0.0};
        if (!(alpha_max > 0.0)) alpha_max = 0.0;
        if (!(alpha_min < 0.0)) alpha_min = 0.0;
        return {alpha_min, alpha_max};
    }
};

}  // namespace corehydro::numerics::math::optimization
