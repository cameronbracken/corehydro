// ported from: Numerics/Mathematics/Optimization/Constrained/Constraint/Constraint.cs @ 2a0357a
//
// The base implementation of IConstraint: an immutable bundle of a constraint function, its
// parameter count, the right-hand-side value, the constraint type, and a violation tolerance.
// The C# class validates nothing in its constructor (not even a null function or a negative
// parameter count) -- that is upstream behavior and is mirrored here rather than "improved".
#pragma once
#include <utility>

#include "corehydro/numerics/math/optimization/constraint/constraint_type.hpp"
#include "corehydro/numerics/math/optimization/constraint/i_constraint.hpp"

namespace corehydro::numerics::math::optimization {

class Constraint : public IConstraint {
   public:
    // Construct a new constraint.
    Constraint(ConstraintFunction constraint_function, int number_of_parameters, double value,
               ConstraintType type, double tolerance = 1E-8)
        : function_(std::move(constraint_function)),
          number_of_parameters_(number_of_parameters),
          value_(value),
          type_(type),
          tolerance_(tolerance) {}

    // The constraint type.
    ConstraintType type() const override { return type_; }

    // The number of parameters to evaluate in the function.
    int number_of_parameters() const override { return number_of_parameters_; }

    // Calculates the left hand side of the constraint.
    const ConstraintFunction& function() const override { return function_; }

    // The value on the right hand side of the constraint equation.
    double value() const override { return value_; }

    // The violation tolerance for the constraint. Default = 1E-8.
    double tolerance() const override { return tolerance_; }

   private:
    ConstraintFunction function_;
    int number_of_parameters_;
    double value_;
    ConstraintType type_;
    double tolerance_;
};

}  // namespace corehydro::numerics::math::optimization
