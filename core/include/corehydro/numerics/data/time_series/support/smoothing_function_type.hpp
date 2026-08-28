// ported from: Numerics/Data/Time Series/Support/SmoothingFunctionType.cs @ 2a0357a
//
// The smoothing applied before a block-series or peaks-over-threshold extraction, in C#
// declaration order (note that `None` is LAST, not first, so a default-initialized value is
// `Difference`; every call site passes it explicitly). The directory mirrors the C# folder
// layout (Numerics/Data/Time Series/Support/); the namespace stays flat like C#'s Numerics.Data.
#pragma once

namespace corehydro::numerics::data {

enum class SmoothingFunctionType {
    Difference,
    MovingAverage,
    MovingSum,
    None,
};

}  // namespace corehydro::numerics::data
