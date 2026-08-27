// corehydro ADDITION -- an adapter, no upstream C# counterpart.
//
// In C# `NelderMead` IS an `Optimizer` subclass, so any call site holding an `Optimizer` can be
// handed one. In this port `nelder_mead.hpp` is a deliberately STANDALONE Phase-0 class (see its
// own header), so the two ported call sites that need a Nelder-Mead behind an `Optimizer`
// reference need an adapter: `MLSL::get_local_optimizer`'s `LocalMethod::NelderMead` branch, and
// `GeneralizedLinearModel::set_optimizer`'s default. This header is that adapter, shared by both
// rather than duplicated -- it was written for MLSL first (as a private nested class) and
// promoted here unchanged when the GLM port needed the same thing.
//
// Every simplex evaluation is routed through the base `Optimizer::evaluate()` so best-tracking,
// evaluation counting and the budget cascade behave exactly as they would for a real C#
// `NelderMead` subclass.
//
// DOCUMENTED LIMITATION, inherited from the standalone class: `nelder_mead.hpp` does not expose
// converged-versus-hit-max-iterations, so this adapter reports `OptimizationStatus::Success`
// unconditionally. A caller that needs to distinguish the two must read the standalone class
// directly.
#pragma once
#include <utility>
#include <vector>

#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"

namespace corehydro::numerics::math::optimization {

class NelderMeadSolver final : public Optimizer {
   public:
    NelderMeadSolver(Objective objective_function, int number_of_parameters,
                     std::vector<double> initial_values, std::vector<double> lower_bounds,
                     std::vector<double> upper_bounds)
        : Optimizer(std::move(objective_function), number_of_parameters),
          initial_values_(std::move(initial_values)),
          lower_bounds_(std::move(lower_bounds)),
          upper_bounds_(std::move(upper_bounds)) {}

   protected:
    void optimize() override {
        bool cancel = false;
        NelderMead solver(
            [this, &cancel](std::vector<double>& x) { return evaluate(x, cancel); },
            number_of_parameters_, initial_values_, lower_bounds_, upper_bounds_);
        solver.max_iterations = max_iterations;
        solver.relative_tolerance = relative_tolerance;
        solver.absolute_tolerance = absolute_tolerance;
        solver.minimize();
        update_status(OptimizationStatus::Success);
    }

   private:
    std::vector<double> initial_values_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
};

}  // namespace corehydro::numerics::math::optimization
