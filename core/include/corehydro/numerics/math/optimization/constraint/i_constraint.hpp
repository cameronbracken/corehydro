// ported from: Numerics/Mathematics/Optimization/Constrained/Constraint/IConstraint.cs @ 2a0357a
//
// Interface for constraints. C#'s read-only properties become const member functions; the C#
// `Func<double[], double> Function` becomes a std::function taking a CONST reference, unlike
// the Optimizer base's `Objective` (which is non-const because a real RMC.BestFit objective
// writes back into the point -- see optimizer.hpp's MUTABLE-POINT SEMANTICS note). No ported
// constraint function mutates its argument, and AugmentedLagrange only ever passes it a point
// it owns.
#pragma once
#include <functional>
#include <vector>

#include "corehydro/numerics/math/optimization/constraint/constraint_type.hpp"

namespace corehydro::numerics::math::optimization {

class IConstraint {
   public:
    using ConstraintFunction = std::function<double(const std::vector<double>&)>;

    virtual ~IConstraint() = default;

    // The constraint type.
    virtual ConstraintType type() const = 0;

    // The number of parameters to evaluate in the function.
    virtual int number_of_parameters() const = 0;

    // Calculates the left hand side of the constraint.
    virtual const ConstraintFunction& function() const = 0;

    // The value on the right hand side of the constraint equation.
    virtual double value() const = 0;

    // The violation tolerance for the constraint. Default = 1E-8.
    virtual double tolerance() const = 0;
};

}  // namespace corehydro::numerics::math::optimization
