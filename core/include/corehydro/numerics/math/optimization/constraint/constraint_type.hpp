// ported from: Numerics/Mathematics/Optimization/Constrained/Constraint/ConstraintType.cs @ 2a0357a
//
// Enumeration of constraint types consumed by Constraint / IConstraint / AugmentedLagrange.
// The declaration order matches the C# exactly (EqualTo, GreaterThanOrEqualTo,
// LesserThanOrEqualTo), which is not the order AugmentedLagrange's switch statements list them
// in -- keep it as written.
#pragma once

namespace corehydro::numerics::math::optimization {

enum class ConstraintType {
    // Equality constraint, h(x) = 0
    EqualTo,

    // Inequality constraint for greater than or equal to, h(x) >= 0
    GreaterThanOrEqualTo,

    // Inequality constraint for lesser than or equal to, h(x) <= 0
    LesserThanOrEqualTo
};

}  // namespace corehydro::numerics::math::optimization
