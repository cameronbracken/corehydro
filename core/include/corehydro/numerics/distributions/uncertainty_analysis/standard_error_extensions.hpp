// ported from: Numerics/Distributions/Univariate/Uncertainty Analysis/StandardErrorExtensions.cs @ 7e8e8d1
#pragma once

#include <vector>

#include "corehydro/numerics/distributions/base/distribution_numerics.hpp"
#include "corehydro/numerics/distributions/base/i_standard_error.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"

namespace corehydro::numerics::distributions {

inline double log_abs_quantile_jacobian(const IStandardError& distribution,
                                        const std::vector<double>& probabilities) {
    const auto* univariate = dynamic_cast<const UnivariateDistributionBase*>(&distribution);
    const int parameter_count =
        univariate != nullptr ? univariate->number_of_parameters()
                              : static_cast<int>(probabilities.size());
    const auto matrix = distribution_numerics::quantile_gradient_matrix(
        distribution, parameter_count, probabilities);
    int sign = 0;
    return distribution_numerics::log_abs_determinant(matrix, sign);
}

}  // namespace corehydro::numerics::distributions
