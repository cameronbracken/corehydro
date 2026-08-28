// ported from: Numerics/Data/Time Series/Support/BlockFunctionType.cs @ 2a0357a
//
// The function applied over each time block by the five block-series methods
// (CalendarYearSeries / CustomYearSeries x2 / MonthlySeries / QuarterlySeries), in C#
// declaration order. The directory mirrors the C# folder layout (Numerics/Data/Time
// Series/Support/); the namespace stays flat like C#'s Numerics.Data.
#pragma once

namespace corehydro::numerics::data {

enum class BlockFunctionType {
    Minimum,
    Maximum,
    Average,
    Sum,
};

}  // namespace corehydro::numerics::data
