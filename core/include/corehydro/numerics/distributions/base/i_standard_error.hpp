// ported from: Numerics/Distributions/Univariate/Uncertainty Analysis/IStandardError.cs @ 2a0357a
//
// Capability interface: a distribution that can compute standard errors. Mirrors the C#
// IStandardError mixin -- the generic fixture runner / bindings dynamic_cast to this to
// compute standard errors and quantile variances, rather than every distribution exposing
// these methods on the common base (Uniform has none).
#pragma once
#include <vector>

#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"

namespace corehydro::numerics::distributions {

class IStandardError {
   public:
    virtual ~IStandardError() = default;
    virtual math::linalg::Matrix2D parameter_covariance(
        int sample_size, ParameterEstimationMethod method) const = 0;
    virtual double quantile_variance(
        double probability, int sample_size, ParameterEstimationMethod method) const = 0;
    virtual std::vector<double> quantile_gradient(double probability) const = 0;
    virtual math::linalg::Matrix2D quantile_jacobian(
        const std::vector<double>& probabilities, double& determinant) const = 0;
};

}  // namespace corehydro::numerics::distributions
