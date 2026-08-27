// ported from: Numerics/Mathematics/Optimization/Local/GradientDescent.cs @ 2a0357a
//
// The Gradient Descent optimization algorithm -- a first-order iterative method for
// unconstrained nonlinear optimization that steps against the gradient by a fixed learning
// rate Alpha. The objective function must be differentiable and convex.
//
// Transcription notes:
//
// 1. Gradient member. C# exposes a settable public field `Func<double[], double[]>? Gradient`
//    (also settable via the ctor's optional trailing parameter); null means "use finite
//    differences". Ported as a public std::function member named `gradient` with the same two
//    entry points; an empty std::function is the null state. The numeric fallback routes
//    through the ported differentiation::gradient(f, point) overload -- the same call shape as
//    the C#'s `NumericalDerivative.Gradient(x => Evaluate(x, ref cancel), p0)` (function +
//    point, default steps, no bounds) -- with the probe function calling the BASE's evaluate()
//    so function-evaluation counting, best-parameter tracking, and the cancellation cascade all
//    mirror the C# exactly. This is the same pattern bfgs.hpp established (see its note 1).
//
// 2. The C# class is ADAM.cs minus the two moment-decay factors: the ctor, the validation
//    order and messages, the accessors and optimize()'s loop scaffolding are identical, and
//    only the parameter update differs (a plain `p0[i] -= Alpha * g[i]`). The duplication is
//    preserved rather than factored into a shared base, per this port's structural-mirroring
//    convention.
#pragma once
#include <cmath>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"

namespace corehydro::numerics::math::optimization {

class GradientDescent : public Optimizer {
   public:
    // The function for evaluating the gradient of the objective function (C#'s settable
    // public `Gradient` field). Empty (the default) means fall back to finite differences.
    using GradientFunction = std::function<std::vector<double>(const std::vector<double>&)>;

    // Construct a new Gradient Descent optimization method. `alpha` is the step size, or
    // learning rate; `gradient` is optional, and the default uses finite differences (mirrors
    // the C# ctor's two optional trailing parameters).
    GradientDescent(Objective objective_function, int number_of_parameters,
                    std::vector<double> initial_values, std::vector<double> lower_bounds,
                    std::vector<double> upper_bounds, double alpha = 0.001,
                    GradientFunction gradient = nullptr)
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
        // `this->` because the ctor parameters deliberately carry the C# parameter names,
        // which shadow the same-named public members.
        this->alpha = alpha;
        this->gradient = std::move(gradient);
    }

    // An array of initial values to evaluate.
    const std::vector<double>& initial_values() const { return initial_values_; }

    // An array of lower bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }

    // An array of upper bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

    // Gets and sets the step size, or learning rate. Default = 0.001.
    double alpha = 0.001;

    // The function for evaluating the gradient of the objective function (see the class
    // header note 1; C# public field `Gradient`).
    GradientFunction gradient;

   protected:
    void optimize() override {
        int D = number_of_parameters_;
        bool cancel = false;
        auto p0 = initial_values_;
        std::vector<double> g(static_cast<std::size_t>(D), 0.0);
        double f0 = evaluate(p0, cancel);
        double f1;

        while (iterations_ < max_iterations) {
            // Get gradient with respect to objective function
            g = gradient ? gradient(p0) : numerical_gradient(p0, cancel);
            if (cancel) return;

            // Update parameters
            for (int i = 0; i < D; i++) {
                std::size_t ui = static_cast<std::size_t>(i);
                // Update parameter
                p0[ui] = p0[ui] - alpha * g[ui];
                // Make sure the parameter is within the bounds.
                p0[ui] = repair_parameter(p0[ui], lower_bounds_[ui], upper_bounds_[ui]);
            }

            // Evaluate the objective function at the new point
            f1 = evaluate(p0, cancel);
            if (cancel) return;

            // Check convergence.
            if (check_convergence(f0, f1)) {
                update_status(OptimizationStatus::Success);
                return;
            }

            f0 = f1;
            iterations_ += 1;
        }

        // If we made it to here, the maximum iterations were reached.
        update_status(OptimizationStatus::MaximumIterationsReached);
    }

   private:
    std::vector<double> initial_values_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;

    // Finite-difference stand-in for the C#'s
    // `NumericalDerivative.Gradient((x) => Evaluate(x, ref cancel), p0)` (see class header
    // note 1).
    std::vector<double> numerical_gradient(const std::vector<double>& p, bool& cancel) {
        return differentiation::gradient(
            [this, &cancel](const std::vector<double>& x) {
                auto values = x;  // Objective/evaluate take a mutable reference
                return evaluate(values, cancel);
            },
            p);
    }
};

}  // namespace corehydro::numerics::math::optimization
