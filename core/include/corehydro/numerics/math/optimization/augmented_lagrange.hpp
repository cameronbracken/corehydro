// ported from: Numerics/Mathematics/Optimization/Constrained/AugmentedLagrange.cs @ 2a0357a
//
// The Augmented Lagrange constrained optimization method: the constrained problem is replaced
// by a series of unconstrained problems solved by an INNER optimizer, whose objective is the
// Lagrangian of the constrained problem plus a quadratic penalty term. The multiplier and
// penalty updates follow Birgin & Martinez.
//
// Transcription notes:
//
// 1. THE INNER OPTIMIZER IS BORROWED, AND ITS OBJECTIVE IS REPLACED. C# holds `public
//    Optimizer Optimizer { get; }` -- a reference to a caller-owned object -- and the
//    constructor OVERWRITES its `ObjectiveFunction` property with this class's augmented
//    Lagrangian. Both are C# semantics, not a port choice, so the port stores a raw
//    `Optimizer*` bound from the constructor's `Optimizer&` and calls
//    `set_objective_function()` on it (the public setter optimizer.hpp exists for). Two
//    consequences a caller must respect: the inner optimizer MUST outlive the
//    AugmentedLagrange, and after construction it can no longer be used on its own, because
//    its objective is no longer the one it was built with.
//
//    The replacement lambda captures `this`, so an AugmentedLagrange must not be moved or
//    copied after construction (the inner optimizer would keep calling the moved-from object).
//    Copy/move are therefore deleted rather than left to silently misbehave; C# reference
//    types have the same property for free.
//
// 2. `augmented_lagrangian_function` calls the PRIMARY objective directly, NOT through the
//    base's `evaluate()`. Inner-optimizer evaluations therefore do not update this optimizer's
//    own best parameter set, function-evaluation counter, or trace -- only the two explicit
//    `evaluate(current_values, cancel)` calls in `optimize()` do. That asymmetry is
//    load-bearing for the fitness the C# tests assert; do not route it through `evaluate()`.
//
// 3. The three multiplier vectors are sized by COUNTING the constraints of each type in the
//    constructor, and are then walked by three separate running indices inside both loops --
//    that is what makes a mixed-constraint problem index correctly (see the C# test
//    `Test_MixedConstraints`, whose comment records the IndexOutOfRangeException that the
//    per-type indexing fixed). The counting, the ordering and the `Math.Min(Math.Max(...))`
//    clamps are all preserved exactly.
//
// 4. `rho`/`rho_max`/`rho_min` are private fields in the C# too (so a caller cannot tune them);
//    `tau` and `gam`, the Birgin & Martinez magic parameters, are `const` locals inside the C#
//    `Optimize()` and are hoisted here to file-scope `constexpr` (MSVC C3493 -- see CLAUDE.md).
//
// 5. C# throws `new ArgumentException(nameof(optimizer), "The inner optimizer cannot also be an
//    Augmented Lagrange optimizer.")`, which passes the argument NAME as the message and the
//    sentence as the paramName -- the two-argument overload's parameters are (message,
//    paramName), so upstream has them swapped. A C++ ArgumentException carries one string, so
//    the port uses the sentence, which is what the swap was clearly meant to say.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/optimization/constraint/constraint.hpp"
#include "corehydro/numerics/math/optimization/constraint/constraint_type.hpp"
#include "corehydro/numerics/math/optimization/constraint/i_constraint.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"

namespace corehydro::numerics::math::optimization {

// Magic parameters from Birgin & Martinez (C# `const double tau = 0.5, gam = 10;`, declared
// inside Optimize(); hoisted to file scope for MSVC -- see note 4).
constexpr double kAugmentedLagrangeTau = 0.5;
constexpr double kAugmentedLagrangeGam = 10;

class AugmentedLagrange : public Optimizer {
   public:
    // Constructs a new Augmented Lagrange optimizer. `optimizer` is the internal optimizer to
    // use; it is borrowed, must outlive this object, and has its objective function replaced
    // (see note 1).
    AugmentedLagrange(Objective objective_function, Optimizer& optimizer,
                      std::vector<std::shared_ptr<IConstraint>> constraints)
        : Optimizer(objective_function, optimizer.number_of_parameters()) {
        // Check if there are constraints
        if (constraints.empty()) {
            throw ArgumentException("There must be at least one constraint.");
        }
        // Constraints
        constraints_ = std::move(constraints);
        std::size_t lambda_count = 0, mu_count = 0, nu_count = 0;
        for (std::size_t i = 0; i < constraints_.size(); i++) {
            if (constraints_[i]->type() == ConstraintType::EqualTo) lambda_count++;
            if (constraints_[i]->type() == ConstraintType::LesserThanOrEqualTo) mu_count++;
            if (constraints_[i]->type() == ConstraintType::GreaterThanOrEqualTo) nu_count++;
        }
        lambda_ = std::vector<double>(lambda_count, 0.0);
        mu_ = std::vector<double>(mu_count, 0.0);
        nu_ = std::vector<double>(nu_count, 0.0);

        // Set up objective functions and optimizer
        if (dynamic_cast<AugmentedLagrange*>(&optimizer) != nullptr)
            throw ArgumentException("The inner optimizer cannot also be an Augmented Lagrange optimizer.");
        primary_objective_function_ = std::move(objective_function);
        optimizer_ = &optimizer;
        optimizer_->set_objective_function(
            [this](std::vector<double>& x) { return augmented_lagrangian_function(x); });
    }

    // The inner optimizer's objective holds a pointer to this object (see note 1).
    AugmentedLagrange(const AugmentedLagrange&) = delete;
    AugmentedLagrange& operator=(const AugmentedLagrange&) = delete;
    AugmentedLagrange(AugmentedLagrange&&) = delete;
    AugmentedLagrange& operator=(AugmentedLagrange&&) = delete;

    // The internal optimizer to use in the Augmented Lagrange method.
    Optimizer& optimizer() { return *optimizer_; }
    const Optimizer& optimizer() const { return *optimizer_; }

    // Returns the list of constraints.
    const std::vector<std::shared_ptr<IConstraint>>& constraints() const { return constraints_; }

    // Returns the Lagrangian equality multipliers.
    const std::vector<double>& lambda() const { return lambda_; }

    // Returns the "Lesser than" inequality multipliers.
    const std::vector<double>& mu() const { return mu_; }

    // Returns the "Greater than" inequality multipliers.
    const std::vector<double>& nu() const { return nu_; }

   protected:
    void optimize() override {
        double ICM = std::numeric_limits<double>::infinity();
        double min_penalty = std::numeric_limits<double>::infinity();
        double penalty;
        double min_fitness = std::numeric_limits<double>::infinity();
        double current_fitness;
        bool min_feasible = false;
        bool cancel = false;

        optimizer_->minimize();
        auto current_values = optimizer_->best_parameter_set().values;

        std::fill(lambda_.begin(), lambda_.end(), 0.0);
        std::fill(mu_.begin(), mu_.end(), 0.0);
        std::fill(nu_.begin(), nu_.end(), 0.0);
        rho_ = 1;

        // Starting rho suggested by B & M
        if (!lambda_.empty() || !mu_.empty() || !nu_.empty()) {
            bool feasible = true;
            double con2 = 0;
            penalty = 0;

            // Evaluate function
            current_fitness = evaluate(current_values, cancel);

            // For each constraint
            for (std::size_t i = 0; i < constraints_.size(); i++) {
                double actual = constraints_[i]->function()(current_values);
                double c = 0;

                switch (constraints_[i]->type()) {
                    case ConstraintType::EqualTo:
                        c = actual - constraints_[i]->value();
                        penalty += std::fabs(c);
                        con2 += c * c;
                        feasible = feasible && std::fabs(c) <= constraints_[i]->tolerance();
                        break;

                    case ConstraintType::GreaterThanOrEqualTo:
                        c = constraints_[i]->value() - actual;
                        if (c > 0) {
                            penalty += c;
                            con2 += c * c;
                        }
                        feasible = feasible && c <= constraints_[i]->tolerance();
                        break;

                    case ConstraintType::LesserThanOrEqualTo:
                        c = actual - constraints_[i]->value();
                        if (c > 0) {
                            penalty += c;
                            con2 += c * c;
                        }
                        feasible = feasible && c <= constraints_[i]->tolerance();
                        break;
                }
            }

            min_fitness = current_fitness;
            min_penalty = penalty;
            min_feasible = feasible;
            double num = 2.0 * std::fabs(min_fitness);
            if (num < 1e-300) {
                rho_ = rho_min_;
            } else if (con2 < 1e-300) {
                rho_ = rho_max_;
            } else {
                rho_ = num / con2;
                if (rho_ < rho_min_)
                    rho_ = rho_min_;
                else if (rho_ > rho_max_)
                    rho_ = rho_max_;
            }
        }

        while (iterations_ < max_iterations) {
            double prev_ICM = ICM;

            optimizer_->minimize();
            current_values = optimizer_->best_parameter_set().values;
            current_fitness = evaluate(current_values, cancel);
            if (cancel) return;

            ICM = 0;
            penalty = 0;
            bool feasible = true;

            // Update lambdas
            std::size_t lambda_idx = 0, mu_idx = 0, nu_idx = 0;
            for (std::size_t i = 0; i < constraints_.size(); i++) {
                double actual = constraints_[i]->function()(current_values);
                double c = 0;
                double new_lambda = 0;

                switch (constraints_[i]->type()) {
                    case ConstraintType::EqualTo:
                        c = actual - constraints_[i]->value();
                        new_lambda = lambda_[lambda_idx] + rho_ * c;
                        penalty += std::fabs(c);
                        feasible = feasible && std::fabs(c) <= constraints_[i]->tolerance();
                        ICM = std::max(ICM, std::fabs(c));
                        lambda_[lambda_idx] = std::min(std::max(-1e20, new_lambda), 1e20);
                        lambda_idx++;
                        break;

                    case ConstraintType::LesserThanOrEqualTo:
                        c = actual - constraints_[i]->value();
                        new_lambda = mu_[mu_idx] + rho_ * c;
                        penalty += c > 0 ? c : 0;
                        feasible = feasible && c <= constraints_[i]->tolerance();
                        ICM = std::max(ICM, std::fabs(std::max(c, -mu_[mu_idx] / rho_)));
                        mu_[mu_idx] = std::min(std::max(0.0, new_lambda), 1e20);
                        mu_idx++;
                        break;

                    case ConstraintType::GreaterThanOrEqualTo:
                        c = constraints_[i]->value() - actual;
                        new_lambda = nu_[nu_idx] + rho_ * c;
                        penalty += c > 0 ? c : 0;
                        feasible = feasible && c <= constraints_[i]->tolerance();
                        ICM = std::max(ICM, std::fabs(std::max(c, -nu_[nu_idx] / rho_)));
                        nu_[nu_idx] = std::min(std::max(0.0, new_lambda), 1e20);
                        nu_idx++;
                        break;
                }
            }
            // Update rho
            if (ICM > kAugmentedLagrangeTau * prev_ICM) {
                rho_ *= kAugmentedLagrangeGam;
            }

            // Determine if the solution as converged.
            bool a = !min_feasible || penalty < min_penalty || current_fitness < min_fitness;
            bool b = !min_feasible && penalty < min_penalty;
            if ((feasible && a) || b) {
                if (feasible && check_convergence(min_fitness, current_fitness)) {
                    update_status(OptimizationStatus::Success);
                    return;
                }

                best_parameter_set_ = ParameterSet(current_values, current_fitness);
                min_fitness = current_fitness;
                min_penalty = penalty;
                min_feasible = feasible;

            } else if (ICM == 0) {
                update_status(OptimizationStatus::Success);
                return;
            }

            iterations_ += 1;
        }

        // If we made it to here, the maximum iterations were reached.
        update_status(OptimizationStatus::MaximumIterationsReached);
    }

   private:
    // The Augmented Lagrangian objective function (see note 2: it calls the primary objective
    // directly, never through evaluate()).
    double augmented_lagrangian_function(std::vector<double>& x) {
        double phi = primary_objective_function_(x);
        double rho2 = 0.5 * rho_;

        std::size_t lambda_idx = 0, mu_idx = 0, nu_idx = 0;
        for (std::size_t i = 0; i < constraints_.size(); i++) {
            double actual = constraints_[i]->function()(x);
            double c = 0;

            switch (constraints_[i]->type()) {
                case ConstraintType::EqualTo:
                    c = actual - constraints_[i]->value();
                    phi += rho2 * std::pow(c + lambda_[lambda_idx] / rho_, 2.0);
                    lambda_idx++;
                    break;

                case ConstraintType::LesserThanOrEqualTo:
                    c = actual - constraints_[i]->value();
                    if (c > 0) phi += rho2 * std::pow(c + mu_[mu_idx] / rho_, 2.0);
                    mu_idx++;
                    break;

                case ConstraintType::GreaterThanOrEqualTo:
                    c = constraints_[i]->value() - actual;
                    if (c > 0) phi += rho2 * std::pow(c + nu_[nu_idx] / rho_, 2.0);
                    nu_idx++;
                    break;
            }
        }

        return phi;
    }

    double rho_ = 1;
    double rho_max_ = 1e+1;
    double rho_min_ = 1e-6;
    std::vector<double> lambda_;  // equality multipliers
    std::vector<double> mu_;      // "lesser than"  inequality multipliers
    std::vector<double> nu_;      // "greater than" inequality multipliers
    std::vector<std::shared_ptr<IConstraint>> constraints_;
    Objective primary_objective_function_;
    Optimizer* optimizer_ = nullptr;
};

}  // namespace corehydro::numerics::math::optimization
