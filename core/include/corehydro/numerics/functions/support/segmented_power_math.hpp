// corehydro addition: shared BaRatin addition-mode evaluation used by the Numerics
// SegmentedPowerFunction and the RMC.BestFit RatingCurve model.
#pragma once

#include <cmath>
#include <vector>

namespace corehydro::numerics::functions::support {

inline double segmented_power_addition(const std::vector<double>& parameters,
                                       int number_of_segments, double stage) {
    if (stage <= parameters.front()) return 0.0;
    double discharge = 0.0;
    for (int k = 0; k < number_of_segments; ++k) {
        const double depth = stage - parameters[static_cast<std::size_t>(3 * k)];
        if (depth > 0.0)
            discharge += std::pow(10.0, parameters[static_cast<std::size_t>(3 * k + 1)]) *
                         std::pow(depth, parameters[static_cast<std::size_t>(3 * k + 2)]);
    }
    return discharge;
}

}  // namespace corehydro::numerics::functions::support
