// ported from: Numerics/Data/Paired Data/ExtrapolationSides.cs @ 7e8e8d1
#pragma once

namespace corehydro::numerics::data::paired_data {

enum class ExtrapolationSides {
    None = 0,
    Below = 1,
    Above = 2,
    Both = 3,
};

constexpr bool includes(ExtrapolationSides policy, ExtrapolationSides side) {
    return (static_cast<int>(policy) & static_cast<int>(side)) != 0;
}

}  // namespace corehydro::numerics::data::paired_data
