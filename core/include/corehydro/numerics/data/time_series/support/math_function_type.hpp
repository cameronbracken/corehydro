// ported from: Numerics/Data/Time Series/Support/MathFunctionType.cs @ 2a0357a
//
// The in-place value transformations the TimeSeries math region applies, in C# declaration
// order. Upstream declares the enum but never branches on it -- the container exposes one
// method per member instead -- and the R/Python `timeseries` toolbox group is its first real
// consumer, dispatching one `math` method over the member name. The directory mirrors the C#
// folder layout (Numerics/Data/Time Series/Support/); the namespace stays flat like C#'s
// Numerics.Data.
#pragma once

namespace corehydro::numerics::data {

enum class MathFunctionType {
    Add,
    Subtract,
    Multiply,
    Divide,
    Logarithm,
    Exponentiate,
    Inverse,
    Replace,
    Interpolate,
};

}  // namespace corehydro::numerics::data
