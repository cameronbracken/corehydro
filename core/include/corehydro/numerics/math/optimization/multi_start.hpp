// ported from: Numerics/Mathematics/Optimization/Global/MultiStart.cs @ 2a0357a
//
// The Multi-Start (MS) global optimization method (Kan, Boender & Timmer 1985): run a local search
// from the user's initial point, then from MaxIterations - 1 further points drawn uniformly over
// the box, and keep the best local minimum found. The only stopping criterion implemented upstream
// is the maximum number of local searches, which is why the constructor pins MaxIterations to 100.
//
// This is MLSL's simpler sibling and mlsl.hpp is the style model: the two C# classes share the
// GetLocalOptimizer body almost verbatim (same three LocalMethod branches, same in-place bound
// repair, same localCancel closure, same MaxFunctionEvaluations budget arithmetic), and the same
// transcription notes apply. See mlsl.hpp notes 3, 4 and 5 -- in particular note 5 for why the
// NelderMead branch wraps the deliberately-standalone Phase 0 NelderMead in a real Optimizer
// subclass rather than constructing it directly.
//
// Transcription hazards specific to this file:
//
// 1. MaxIterations = 100 is set in the CONSTRUCTOR, not as a field initializer, so it overrides
//    the base's 10,000 for every MultiStart -- and a caller who sets MaxIterations afterwards
//    still wins, exactly as in C#.
//
// 2. UPSTREAM ARRAY ALIASING, reproduced deliberately. C# allocates `var values = new double[D];`
//    and then, on the first iteration, does `values = InitialValues;` -- an array REFERENCE
//    assignment, not a copy. From that point on `values` IS the InitialValues array, so every
//    later iteration's uniform draw writes through it and the caller-visible InitialValues
//    property ends the run holding the LAST sampled point rather than the user's starting point.
//    The search path is unaffected (a single scratch buffer either way), but the observable
//    InitialValues is, so the port mirrors it with a pointer that is re-seated to the
//    initial_values_ member instead of copying into a local. `RepairParameter` in
//    get_local_optimizer writes through the same pointer, which is the second half of the same
//    C# behavior (see mlsl.hpp note 3).
//
// 3. The polish step passes `BestParameterSet.Values` to GetLocalOptimizer, which repairs THAT
//    array in place -- so the reported best point can be clamped to the bounds by the polish call
//    even before the polish search runs. Mirrored by passing best_parameter_set_.values as the
//    non-const reference, exactly as mlsl.hpp does.
//
// 4. Each local solver is constructed over `(x) => Evaluate(x, ref localCancel)` -- the PARENT's
//    evaluate() -- so the parent's best-tracking, trace, evaluation counter and
//    MaxFunctionEvaluations budget see every inner evaluation. The budget handed to the child is
//    `MaxFunctionEvaluations - FunctionEvaluations`, recomputed at each construction.
//
// 5. The ADAM and GradientDescent LocalMethod arms throw "Unsupported local method" here just as
//    they do in MLSL, even though both classes are now ported (see support/local_method.hpp's
//    header). That is upstream behavior, not a gap in this port.
#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/math/optimization/bfgs.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/optimization/powell.hpp"
#include "corehydro/numerics/math/optimization/support/local_method.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::math::optimization {

class MultiStart : public Optimizer {
   public:
    // Construct a new multi-start optimization method.
    //   objective_function:   the objective function to evaluate.
    //   number_of_parameters: the number of parameters in the objective function.
    //   initial_values:       an array of initial values to evaluate.
    //   lower_bounds:         an array of lower bounds (inclusive) of the interval containing the
    //                         optimal point.
    //   upper_bounds:         an array of upper bounds (inclusive) of the interval containing the
    //                         optimal point.
    //   method:               the local optimization method to use. Default = BFGS.
    MultiStart(Objective objective_function, int number_of_parameters,
               std::vector<double> initial_values, std::vector<double> lower_bounds,
               std::vector<double> upper_bounds, LocalMethod method = LocalMethod::BFGS)
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
        this->method = method;  // the parameter shadows the member, as `Method = method` in C#
        max_iterations = 100;   // set in the ctor, not as a field default -- hazard 1
    }

    // An array of initial values to evaluate.
    const std::vector<double>& initial_values() const { return initial_values_; }

    // An array of lower bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }

    // An array of upper bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

    // The pseudo random number generator (PRNG) seed.
    int prng_seed = 12345;

    // The local search method to use. Default = BFGS.
    LocalMethod method = LocalMethod::BFGS;

    // The desired absolute tolerance for the local solution. Default = ~Sqrt(1E-16), or 1E-8.
    double local_absolute_tolerance = 1E-8;

    // The desired relative tolerance for the local solution. Default = ~Sqrt(1E-16), or 1E-8.
    double local_relative_tolerance = 1E-8;

    // If true (default), then a final local search is used to polish the best population member
    // at the end, which can improve the optimization slightly.
    bool polish = true;

   protected:
    void optimize() override {
        int i, j, D = number_of_parameters_;
        bool cancel = false;
        std::unique_ptr<Optimizer> solver;

        // Set lower and upper bounds and
        // create uniform distributions for each parameter
        std::vector<distributions::Uniform> uniform_dists;
        for (i = 0; i < D; i++)
            uniform_dists.emplace_back(lower_bounds_[static_cast<std::size_t>(i)],
                                       upper_bounds_[static_cast<std::size_t>(i)]);

        // Set solver parameters
        sampling::MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        // `values` stands in for C#'s `double[] values`, an array REFERENCE that gets re-seated to
        // the InitialValues array on the first iteration and then written through -- see hazard 2.
        std::vector<double> values_storage(static_cast<std::size_t>(D));
        std::vector<double>* values = &values_storage;

        while (iterations_ < max_iterations) {
            if (iterations_ == 0) {
                values = &initial_values_;  // C#: values = InitialValues (reference assignment)
            } else {
                // Step 1. Draw a point from the uniform distribution over S.
                for (j = 0; j < D; j++)
                    (*values)[static_cast<std::size_t>(j)] =
                        uniform_dists[static_cast<std::size_t>(j)].inverse_cdf(prng.next_double());
            }

            // Step 2. Apply P to the new sample point.
            solver = get_local_optimizer(*values, local_relative_tolerance,
                                         local_absolute_tolerance, cancel);
            solver->minimize();
            if (cancel) return;

            iterations_ += 1;
        }

        // Polish the final result
        if (polish) {
            solver = get_local_optimizer(best_parameter_set_.values, relative_tolerance,
                                         absolute_tolerance, cancel);
            solver->minimize();
            if (cancel) return;
        }

        // If we made it to here, the maximum iterations were reached.
        update_status(OptimizationStatus::MaximumIterationsReached);
    }

   private:
    std::vector<double> initial_values_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;

    // Wraps the standalone Phase 0 NelderMead in a real Optimizer subclass for the
    // GetLocalOptimizer NelderMead branch (mlsl.hpp note 5, which carries the full reasoning and
    // the two documented residual deviations). mlsl.hpp holds a byte-identical private nested copy.
    // It is duplicated rather than shared because the C# constructs `new NelderMead(...)` inline in
    // BOTH files -- the wrapper is a port artifact standing in for that expression, so each file
    // carrying its own keeps the structural mirroring intact and keeps this header from depending
    // on an unrelated optimizer. If a third caller ever needs it, promote it to a support header
    // and update both call sites together.
    class NelderMeadLocalSolver final : public Optimizer {
       public:
        NelderMeadLocalSolver(Objective objective_function, int number_of_parameters,
                              std::vector<double> initial_values,
                              std::vector<double> lower_bounds, std::vector<double> upper_bounds)
            : Optimizer(std::move(objective_function), number_of_parameters),
              initial_values_(std::move(initial_values)),
              lower_bounds_(std::move(lower_bounds)),
              upper_bounds_(std::move(upper_bounds)) {}

       protected:
        void optimize() override {
            bool cancel = false;
            NelderMead solver(
                // Route every simplex evaluation through the BASE's evaluate() so best-tracking,
                // evaluation counting, the MaxFunctionEvaluations budget, and the throw-through-
                // cancel cascade all behave exactly like the real C# NelderMead subclass.
                [this, &cancel](std::vector<double>& x) { return evaluate(x, cancel); },
                number_of_parameters_, initial_values_, lower_bounds_, upper_bounds_);
            solver.max_iterations = max_iterations;
            solver.relative_tolerance = relative_tolerance;
            solver.absolute_tolerance = absolute_tolerance;
            solver.minimize();
            // The standalone class does not expose converged-vs-max-iterations; Success is
            // reported unconditionally (documented limitation, mlsl.hpp note 5(a)).
            update_status(OptimizationStatus::Success);
        }

       private:
        std::vector<double> initial_values_;
        std::vector<double> lower_bounds_;
        std::vector<double> upper_bounds_;
    };

    // Returns an optimizer for the local search.
    //   initial_values:     an array of initial values to evaluate (repaired IN PLACE, mutating
    //                       the caller's vector exactly as the C# repairs the caller's IList --
    //                       see mlsl.hpp note 3 and hazards 2/3 above).
    //   relative_tolerance: the desired relative tolerance for the solution.
    //   absolute_tolerance: the desired absolute tolerance for the solution.
    //   cancel:             by ref. Determines if the solver should be canceled.
    std::unique_ptr<Optimizer> get_local_optimizer(std::vector<double>& initial_values,
                                                   double relative_tolerance,
                                                   double absolute_tolerance, bool& cancel) {
        // Heap closure standing in for the C# captured local `localCancel` (mlsl.hpp note 4).
        auto local_cancel = std::make_shared<bool>(false);
        std::unique_ptr<Optimizer> solver;

        // Make sure the parameters are within the bounds.
        for (int i = 0; i < number_of_parameters_; i++)
            initial_values[static_cast<std::size_t>(i)] =
                repair_parameter(initial_values[static_cast<std::size_t>(i)],
                                 lower_bounds_[static_cast<std::size_t>(i)],
                                 upper_bounds_[static_cast<std::size_t>(i)]);

        if (method == LocalMethod::BFGS) {
            solver = std::make_unique<BFGS>(
                [this, local_cancel](std::vector<double>& x) { return evaluate(x, *local_cancel); },
                number_of_parameters_, initial_values, lower_bounds_, upper_bounds_);
        } else if (method == LocalMethod::NelderMead) {
            // The standalone Phase 0 NelderMead is not an Optimizer subclass; MLSL already carries
            // the wrapper that makes it one, and it is reused here rather than duplicated (see
            // mlsl.hpp note 5 for the two documented residual deviations).
            solver = std::make_unique<NelderMeadLocalSolver>(
                [this, local_cancel](std::vector<double>& x) { return evaluate(x, *local_cancel); },
                number_of_parameters_, initial_values, lower_bounds_, upper_bounds_);
        } else if (method == LocalMethod::Powell) {
            solver = std::make_unique<Powell>(
                [this, local_cancel](std::vector<double>& x) { return evaluate(x, *local_cancel); },
                number_of_parameters_, initial_values, lower_bounds_, upper_bounds_);
        } else {
            // C# NotSupportedException; a plain std::runtime_error routes through the base
            // minimize()/maximize() catch-all to Failure, exactly like the C# catch-all handles
            // NotSupportedException.
            throw std::runtime_error("Unsupported local method");
        }
        solver->relative_tolerance = relative_tolerance;
        solver->absolute_tolerance = absolute_tolerance;
        solver->max_function_evaluations = max_function_evaluations - function_evaluations_;
        cancel = cancel || *local_cancel;
        return solver;
    }
};

}  // namespace corehydro::numerics::math::optimization
