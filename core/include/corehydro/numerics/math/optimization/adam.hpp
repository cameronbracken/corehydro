// ported from: Numerics/Mathematics/Optimization/Local/ADAM.cs @ 2a0357a
//
// The Adaptive Movement (Adam) optimization algorithm -- an extension of gradient descent that
// maintains exponentially decaying first- and second-moment estimates of the gradient and takes
// a bias-corrected step of size Alpha (Kingma and Ba, 2014; see the C# file's references). The
// objective function must be differentiable and convex.
//
// Transcription notes:
//
// 1. Gradient member. C# exposes a settable public field `Func<double[], double[]>? Gradient`
//    (also settable via the ctor's optional trailing parameter); null means "use finite
//    differences". Ported as a public std::function member named `gradient` with the same two
//    entry points; an empty std::function is the null state. The numeric fallback routes
//    through the ported differentiation::gradient(f, point) overload -- the same call shape as
//    the C#'s `NumericalDerivative.Gradient(x => Evaluate(x, ref cancel), p)` (function +
//    point, default steps, no bounds) -- with the probe function calling the BASE's evaluate()
//    so function-evaluation counting, best-parameter tracking, and the cancellation cascade all
//    mirror the C# exactly. This is the same pattern bfgs.hpp established (see its note 1).
//
// 2. `Tools.DoubleMachineEpsilon` in the parameter update is the C# stabilizing epsilon added
//    to sqrt(vhat); ported as the same file-scope `kDoubleMachineEpsilon` constant the rest of
//    the port uses.
#pragma once
#include <cmath>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::optimization {

class ADAM : public Optimizer {
   public:
    // The function for evaluating the gradient of the objective function (C#'s settable
    // public `Gradient` field). Empty (the default) means fall back to finite differences.
    using GradientFunction = std::function<std::vector<double>(const std::vector<double>&)>;

    // Construct a new Adam optimization method. `alpha` is the step size, or learning rate;
    // `gradient` is optional, and the default uses finite differences (mirrors the C# ctor's
    // two optional trailing parameters).
    ADAM(Objective objective_function, int number_of_parameters, std::vector<double> initial_values,
         std::vector<double> lower_bounds, std::vector<double> upper_bounds, double alpha = 0.001,
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

    // Gets and sets the decay factor for the first momentum. Default = 0.9.
    double beta1 = 0.9;

    // Gets and sets the decay factor for infinity norm. Default = 0.999.
    double beta2 = 0.999;

    // The function for evaluating the gradient of the objective function (see the class
    // header note 1; C# public field `Gradient`).
    GradientFunction gradient;

   protected:
    void optimize() override {
        int D = number_of_parameters_;
        bool cancel = false;
        auto p = initial_values_;
        std::vector<double> g(static_cast<std::size_t>(D), 0.0);
        std::vector<double> m(static_cast<std::size_t>(D), 0.0);
        std::vector<double> v(static_cast<std::size_t>(D), 0.0);
        double mhat, vhat;
        double f0 = evaluate(p, cancel);
        double f1;

        while (iterations_ < max_iterations) {
            // Get gradient with respect to objective function
            g = gradient ? gradient(p) : numerical_gradient(p, cancel);
            if (cancel) return;

            // Update parameters
            for (int i = 0; i < D; i++) {
                std::size_t ui = static_cast<std::size_t>(i);
                // Update biased first and second moment estimates
                m[ui] = beta1 * m[ui] + (1.0 - beta1) * g[ui];
                v[ui] = beta2 * v[ui] + (1.0 - beta2) * g[ui] * g[ui];
                // Compute biased corrected moment estimates
                mhat = m[ui] / (1.0 - std::pow(beta1, iterations_ + 1));
                vhat = v[ui] / (1.0 - std::pow(beta2, iterations_ + 1));
                // Update parameter
                // C#: Tools.DoubleMachineEpsilon
                p[ui] = p[ui] - alpha * mhat / (std::sqrt(vhat) + kDoubleMachineEpsilon);
                // Make sure the parameter is within the bounds.
                p[ui] = repair_parameter(p[ui], lower_bounds_[ui], upper_bounds_[ui]);
            }

            // Evaluate the objective function at the new point
            f1 = evaluate(p, cancel);
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
    // `NumericalDerivative.Gradient((x) => Evaluate(x, ref cancel), p)` (see class header
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
