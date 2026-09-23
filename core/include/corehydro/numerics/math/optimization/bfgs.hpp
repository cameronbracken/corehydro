// ported from: Numerics/Mathematics/Optimization/Local/BFGS.cs @ 7e8e8d1
//
// The Broyden-Fletcher-Goldfarb-Shanno (BFGS) local optimizer: an iterative method for
// unconstrained nonlinear optimization that gradually improves an approximation to the
// inverse Hessian of the loss function from gradient evaluations via a generalized secant
// method (Numerical Recipes, Press et al.; see the C# file's references). This is the
// first REAL local Optimizer subclass in the port -- it derives from the ported Optimizer
// base exactly as differential_evolution.hpp does, NOT via the Phase 4
// estimation/support/optimizer_adapters.hpp `detail::` stopgap (which wrapped the
// standalone Phase 0 nelder_mead/brent_search; B5+ optimizers make that pattern
// unnecessary going forward).
//
// Transcription notes:
//
// 1. Gradient member. C# exposes a settable public field `Func<double[], double[]>?
//    Gradient` (also settable via the ctor's optional trailing parameter); null means
//    "use finite differences". Ported as a public std::function member named `gradient`
//    with the same two entry points; an empty std::function is the null state. Every
//    numeric fallback routes through the ported bound-aware
//    differentiation::gradient(f, point) overload -- the same call shape as the C#'s
//    `NumericalDerivative.Gradient(x => Evaluate(x, ref cancel), p)` (function + point,
//    default steps, no bounds) -- with the probe function calling the BASE's evaluate()
//    so function-evaluation counting, best-parameter tracking, and the cancellation
//    cascade all mirror the C# exactly. The lambda copies its (const-ref) probe point
//    into a local because Objective/evaluate take a mutable reference (see
//    optimizer.hpp's MUTABLE-POINT SEMANTICS note). optimize()'s two identical inline
//    ternaries share the small private numerical_gradient() helper; LineSearch/Zoom keep
//    theirs inline because of the cancelFlag round-trip (note 2).
//
// 2. C# LineSearch/Zoom capture quirk. Inside LineSearch/Zoom, `cancel` is a `ref`
//    parameter, which C# lambdas cannot capture -- the C# copies it into a local
//    `cancelFlag`, lets the gradient lambda capture that, then writes it back. C++
//    reference parameters have no such restriction, but the local-copy round-trip is
//    transcribed anyway so the code maps line-for-line onto the C#.
//
// 3. LineSearchArmijo is ported for structural fidelity but is NOT called by optimize()
//    -- exactly as in the C#, where the strong-Wolfe LineSearch superseded it. Its
//    `throw new Exception("Roundoff problem in line search.")` is a genuine C# throw
//    (not a Debug.WriteLine/swallowed guard), so it ports as a real std::runtime_error;
//    if it ever fired it would surface through the base's minimize()/maximize()
//    Failure-status path like any other objective-side exception.
//
// 4. `TOLX` in optimize() is declared-and-unused in the C# too (LineSearchArmijo has its
//    own local TOLX); kept, with a (void) cast to satisfy -Wall/-Wextra.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::optimization {

class BFGS : public Optimizer {
   public:
    // The function for evaluating the gradient of the objective function (C#'s settable
    // public `Gradient` field). Empty (the default) means fall back to finite differences.
    using GradientFunction = std::function<std::vector<double>(const std::vector<double>&)>;

    // Construct a new BFGS optimization method. `gradient` is optional; the default uses
    // finite differences (mirrors the C# ctor's optional trailing parameter).
    BFGS(Objective objective_function, int number_of_parameters,
         std::vector<double> initial_values, std::vector<double> lower_bounds,
         std::vector<double> upper_bounds, GradientFunction gradient_function = nullptr)
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
        gradient = std::move(gradient_function);
    }

    // An array of initial values to evaluate.
    const std::vector<double>& initial_values() const { return initial_values_; }

    // An array of lower bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }

    // An array of upper bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

    // The function for evaluating the gradient of the objective function (see the class
    // header note 1; C# public field `Gradient`).
    GradientFunction gradient;

   protected:
    const std::vector<double>& parameter_lower_bounds() const override { return lower_bounds_; }
    const std::vector<double>& parameter_upper_bounds() const override { return upper_bounds_; }

    void optimize() override {
        const int n = number_of_parameters_;
        bool cancel = false;
        auto x = initial_values_;
        double f = evaluate_objective(x, cancel);
        if (cancel) return;
        if (!is_finite(f)) throw ArgumentException("The initial objective value must be finite.");
        auto g = evaluate_gradient(x, cancel);
        if (cancel) return;

        auto inverse_hessian = linalg::Matrix::identity(n);
        std::vector<double> projected(static_cast<std::size_t>(n));
        std::vector<double> direction(static_cast<std::size_t>(n));
        double stpmax = 100.0 * std::max(std::sqrt(sum_product(x, x)), static_cast<double>(n));

        while (true) {
            if (projected_gradient(x, g, projected) <= absolute_tolerance) {
                best_parameter_set_ = ParameterSet(x, f);
                update_status(OptimizationStatus::Success);
                return;
            }
            if (iterations_ >= max_iterations) {
                update_status(OptimizationStatus::MaximumIterationsReached);
                return;
            }

            for (int i = 0; i < n; ++i) {
                direction[static_cast<std::size_t>(i)] = 0.0;
                for (int j = 0; j < n; ++j)
                    direction[static_cast<std::size_t>(i)] -=
                        inverse_hessian(i, j) * projected[static_cast<std::size_t>(j)];
            }
            make_feasible(x, direction);
            double slope = sum_product(g, direction);
            if (!is_finite(slope) || slope >= 0.0) {
                inverse_hessian = linalg::Matrix::identity(n);
                for (int i = 0; i < n; ++i)
                    direction[static_cast<std::size_t>(i)] = -projected[static_cast<std::size_t>(i)];
            }

            bool can_restart = false;
            for (int i = 0; i < n; ++i)
                can_restart = can_restart ||
                              direction[static_cast<std::size_t>(i)] != -projected[static_cast<std::size_t>(i)];
            std::vector<double> next_x, next_g;
            double next_f = f;
            bool accepted = line_search(x, f, g, direction, stpmax, next_x, next_f, next_g, cancel);
            if (!accepted && !cancel && can_restart) {
                inverse_hessian = linalg::Matrix::identity(n);
                for (int i = 0; i < n; ++i)
                    direction[static_cast<std::size_t>(i)] = -projected[static_cast<std::size_t>(i)];
                accepted = line_search(x, f, g, direction, stpmax, next_x, next_f, next_g, cancel);
            }
            if (cancel) return;
            if (!accepted) {
                update_status(OptimizationStatus::LineSearchFailed);
                return;
            }
            ++iterations_;

            std::vector<double> step(static_cast<std::size_t>(n));
            std::vector<double> change(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                step[static_cast<std::size_t>(i)] = next_x[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)];
                change[static_cast<std::size_t>(i)] = next_g[static_cast<std::size_t>(i)] - g[static_cast<std::size_t>(i)];
            }
            update_inverse_hessian(inverse_hessian, step, change);
            x = std::move(next_x);
            f = next_f;
            g = std::move(next_g);
        }
    }

   private:
    std::vector<double> initial_values_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;

    double evaluate_objective(std::vector<double>& x, bool& cancel) {
        auto incumbent = best_parameter_set_;
        double f = evaluate(x, cancel);
        if (!is_finite(f)) best_parameter_set_ = std::move(incumbent);
        return f;
    }

    std::vector<double> evaluate_gradient(const std::vector<double>& x, bool& cancel) {
        std::vector<double> g;
        if (gradient) {
            g = gradient(x);
            if (static_cast<int>(g.size()) != number_of_parameters_)
                throw ArgumentException("The gradient must contain one value per parameter.");
            for (double& value : g) value *= static_cast<double>(function_scale_);
        } else {
            bool stopped = cancel;
            g = differentiation::gradient(
                [this, &stopped](const std::vector<double>& point) {
                    if (stopped) return std::numeric_limits<double>::quiet_NaN();
                    auto trial = point;
                    return evaluate_objective(trial, stopped);
                },
                x, lower_bounds_, upper_bounds_);
            cancel = stopped;
            if (cancel) return g;
        }
        for (double value : g)
            if (!is_finite(value))
                throw ArgumentException("The gradient must contain only finite values.");
        return g;
    }

    double projected_gradient(const std::vector<double>& x, const std::vector<double>& g,
                              std::vector<double>& projected) const {
        double norm = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            projected[i] = ((x[i] <= lower_bounds_[i] && g[i] > 0.0) ||
                            (x[i] >= upper_bounds_[i] && g[i] < 0.0) ||
                            lower_bounds_[i] == upper_bounds_[i])
                               ? 0.0
                               : g[i];
            norm = std::max(norm, std::fabs(projected[i]));
        }
        return norm;
    }

    void make_feasible(const std::vector<double>& x, std::vector<double>& direction) const {
        for (std::size_t i = 0; i < x.size(); ++i)
            if ((x[i] <= lower_bounds_[i] && direction[i] < 0.0) ||
                (x[i] >= upper_bounds_[i] && direction[i] > 0.0))
                direction[i] = 0.0;
    }

    static void update_inverse_hessian(linalg::Matrix& h, const std::vector<double>& s,
                                       const std::vector<double>& y) {
        const int n = static_cast<int>(s.size());
        std::vector<double> hy(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                hy[static_cast<std::size_t>(i)] += h(i, j) * y[static_cast<std::size_t>(j)];
        double ys = sum_product(y, s);
        double yhy = sum_product(y, hy);
        double floor = std::sqrt(kDoubleMachineEpsilon) * std::sqrt(sum_product(y, y)) *
                       std::sqrt(sum_product(s, s));
        if (!is_finite(ys) || !is_finite(yhy) || yhy <= 0.0 || ys <= floor) return;
        std::vector<double> v(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            v[static_cast<std::size_t>(i)] = s[static_cast<std::size_t>(i)] / ys -
                                             hy[static_cast<std::size_t>(i)] / yhy;
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                double value = h(i, j) + s[static_cast<std::size_t>(i)] * s[static_cast<std::size_t>(j)] / ys -
                               hy[static_cast<std::size_t>(i)] * hy[static_cast<std::size_t>(j)] / yhy +
                               yhy * v[static_cast<std::size_t>(i)] * v[static_cast<std::size_t>(j)];
                if (!is_finite(value)) {
                    h = linalg::Matrix::identity(n);
                    return;
                }
                h(i, j) = h(j, i) = value;
            }
        }
    }

    std::vector<double> trial_point(const std::vector<double>& x0, const std::vector<double>& p,
                                    double alpha) const {
        std::vector<double> x(x0.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = repair_parameter(x0[i] + alpha * p[i], lower_bounds_[i], upper_bounds_[i]);
        return x;
    }

    bool try_roundoff_convergence(const std::vector<double>& trial, double value,
                                  double initial_value, std::vector<double>& result_gradient,
                                  bool& cancel) {
        double roundoff = 8.0 * kDoubleMachineEpsilon * std::fabs(initial_value);
        if (!gradient || !is_finite(value) || std::fabs(value - initial_value) > roundoff) return false;
        auto candidate = evaluate_gradient(trial, cancel);
        std::vector<double> projected(static_cast<std::size_t>(number_of_parameters_));
        if (cancel || projected_gradient(trial, candidate, projected) > absolute_tolerance) return false;
        result_gradient = std::move(candidate);
        return true;
    }

    bool line_search(const std::vector<double>& x0, double f0, const std::vector<double>& g0,
                     std::vector<double>& p, double stpmax, std::vector<double>& x, double& f,
                     std::vector<double>& g, bool& cancel) {
        x = x0;
        f = f0;
        g = g0;
        double norm = std::sqrt(sum_product(p, p));
        if (norm > stpmax)
            for (double& value : p) value *= stpmax / norm;
        double slope0 = sum_product(g0, p);
        if (!is_finite(slope0) || slope0 >= 0.0) return false;

        double limit = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < p.size(); ++i) {
            if (p[i] > 0.0) limit = std::min(limit, (upper_bounds_[i] - x0[i]) / p[i]);
            else if (p[i] < 0.0) limit = std::min(limit, (lower_bounds_[i] - x0[i]) / p[i]);
        }
        double alpha = std::min(1.0, limit);
        double previous = 0.0, f_previous = f0, slope_previous = slope0;
        for (int iteration = 0; iteration < 20; ++iteration) {
            auto trial = trial_point(x0, p, alpha);
            if (alpha <= 0.0 || trial == x0) return false;
            double value = evaluate_objective(trial, cancel);
            if (cancel) return false;
            if (!is_finite(value) || value > f0 + 1e-4 * alpha * slope0) {
                std::vector<double> stationary;
                if (try_roundoff_convergence(trial, value, f0, stationary, cancel)) {
                    x = std::move(trial); f = value; g = std::move(stationary); return true;
                }
                if (cancel) return false;
                return zoom(x0, f0, g0, p, slope0, previous, f_previous, slope_previous,
                            alpha, value, std::numeric_limits<double>::quiet_NaN(), x, f, g, cancel);
            }

            auto trial_gradient = evaluate_gradient(trial, cancel);
            if (cancel) return false;
            double slope = sum_product(trial_gradient, p);
            if (std::fabs(slope) <= -0.9 * slope0 || (alpha == limit && slope < 0.0)) {
                x = std::move(trial); f = value; g = std::move(trial_gradient); return true;
            }
            if (!is_finite(slope)) return false;
            if (iteration > 0 && value >= f_previous)
                return zoom(x0, f0, g0, p, slope0, previous, f_previous, slope_previous,
                            alpha, value, slope, x, f, g, cancel);
            if (slope >= 0.0)
                return zoom(x0, f0, g0, p, slope0, alpha, value, slope,
                            previous, f_previous, slope_previous, x, f, g, cancel);
            previous = alpha;
            f_previous = value;
            slope_previous = slope;
            alpha = std::min(2.0 * alpha, limit);
            if (alpha == previous) return false;
        }
        return false;
    }

    bool zoom(const std::vector<double>& x0, double f0, const std::vector<double>& g0,
              const std::vector<double>& p, double slope0, double low, double f_low,
              double slope_low, double high, double f_high, double slope_high,
              std::vector<double>& x, double& f, std::vector<double>& g, bool& cancel) {
        x = x0; f = f0; g = g0;
        std::vector<double> previous_trial;
        for (int iteration = 0; iteration < 20; ++iteration) {
            double alpha = interpolate(low, f_low, slope_low, high, f_high, slope_high);
            if (alpha == low || alpha == high) return false;
            auto trial = trial_point(x0, p, alpha);
            if (trial == x0 || (!previous_trial.empty() && trial == previous_trial)) return false;
            previous_trial = trial;
            double value = evaluate_objective(trial, cancel);
            if (cancel) return false;
            if (!is_finite(value) || value > f0 + 1e-4 * alpha * slope0) {
                std::vector<double> stationary;
                if (try_roundoff_convergence(trial, value, f0, stationary, cancel)) {
                    x = std::move(trial); f = value; g = std::move(stationary); return true;
                }
                if (cancel) return false;
                high = alpha; f_high = value; slope_high = std::numeric_limits<double>::quiet_NaN();
            } else {
                auto trial_gradient = evaluate_gradient(trial, cancel);
                if (cancel) return false;
                double slope = sum_product(trial_gradient, p);
                if (std::fabs(slope) <= -0.9 * slope0) {
                    x = std::move(trial); f = value; g = std::move(trial_gradient); return true;
                }
                if (!is_finite(slope)) return false;
                if (value >= f_low) {
                    high = alpha; f_high = value; slope_high = slope; continue;
                }
                if (slope * (high - low) >= 0.0) {
                    high = low; f_high = f_low; slope_high = slope_low;
                }
                low = alpha; f_low = value; slope_low = slope;
            }
        }
        return false;
    }

    static double interpolate(double a, double fa, double ga, double b, double fb, double gb) {
        double width = b - a;
        double left = std::min(a, b) + 0.1 * std::fabs(width);
        double right = std::max(a, b) - 0.1 * std::fabs(width);
        double candidate = std::numeric_limits<double>::quiet_NaN();
        if (is_finite(gb) && is_finite(fb)) {
            double d1 = ga + gb - 3.0 * (fb - fa) / width;
            double radicand = d1 * d1 - ga * gb;
            if (radicand >= 0.0) {
                double d2 = std::copysign(std::sqrt(radicand), width);
                candidate = b - width * (gb + d2 - d1) / (gb - ga + 2.0 * d2);
            }
        }
        if (!is_finite(candidate) || candidate <= left || candidate >= right)
            candidate = a - ga * width * width / (2.0 * (fb - fa - ga * width));
        if (!is_finite(candidate) || candidate <= std::min(a, b) || candidate >= std::max(a, b))
            return a + 0.5 * width;
        return std::max(left, std::min(right, candidate));
    }
};

}  // namespace corehydro::numerics::math::optimization
