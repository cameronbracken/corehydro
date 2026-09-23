// ported from: Numerics/Distributions/Univariate/Base/DistributionMomentIntegration.cs @ 7e8e8d1
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/math/integration/adaptive_gauss_kronrod.hpp"
#include "corehydro/numerics/math/integration/support/integration_status.hpp"

namespace corehydro::numerics::distributions::distribution_moment_integration {

inline double limit(double width, double scale) {
    if (width <= 0.0) return 0.0;
    if (width == std::numeric_limits<double>::infinity()) return 1.0;
    double ratio = width / scale;
    return std::isinf(ratio) ? 1.0 : ratio / (1.0 + ratio);
}

inline std::vector<double> compute(const std::function<double(double)>& log_density,
                                   double minimum, double maximum, double center, double scale) {
    if (!std::isfinite(center) || !(scale > 0.0) || !std::isfinite(scale))
        throw std::runtime_error(
            "a finite reference and positive local scale are required for moment integration");
    const double lower = limit(center - minimum, scale);
    const double upper = limit(maximum - center, scale);
    const double log_scale = std::log(scale);
    auto moment = [&](int order, double offset) {
        auto side = [&](bool positive, double side_limit) {
            if (side_limit == 0.0) return 0.0;
            auto function = [&](double t) {
                const double magnitude = t / (1.0 - t);
                const double coordinate = positive ? magnitude : -magnitude;
                const double log_density_value = log_density(center + scale * coordinate);
                if (log_density_value == -std::numeric_limits<double>::infinity()) return 0.0;
                const double centered = coordinate - offset;
                if (order != 0 && centered == 0.0) return 0.0;
                const double log_value = log_density_value + log_scale - 2.0 * std::log1p(-t) +
                                         (order == 0 ? 0.0
                                                     : order * std::log(std::fabs(centered)));
                const double value = std::exp(log_value);
                if (!std::isfinite(value))
                    throw std::runtime_error("a moment is divergent or unresolved");
                return order % 2 != 0 && centered < 0.0 ? -value : value;
            };
            math::integration::AdaptiveGaussKronrod integrator(function, 0.0, side_limit);
            integrator.relative_tolerance = 1e-8;
            integrator.absolute_tolerance = 1e-10;
            integrator.max_function_evaluations = 200000;
            integrator.report_failure = true;
            integrator.integrate();
            if (integrator.status() != math::integration::IntegrationStatus::Success ||
                !std::isfinite(integrator.result()) ||
                !std::isfinite(integrator.standard_error()) ||
                integrator.standard_error() >
                    std::max(1e-10, std::fabs(integrator.result()) * 1e-8))
                throw std::runtime_error("moment integration did not meet its error tolerance");
            return integrator.result();
        };
        return side(false, lower) + side(true, upper);
    };
    const double mass = moment(0, 0.0);
    if (std::fabs(mass - 1.0) > 5e-8)
        throw std::runtime_error("moment integration did not recover unit mass");
    const double offset = moment(1, 0.0);
    const double variance = moment(2, offset);
    if (!(variance > 0.0)) throw std::runtime_error("moment integration variance is not positive");
    const double third = moment(3, offset);
    const double fourth = moment(4, offset);
    return {center + scale * offset, scale * std::sqrt(variance),
            third / variance / std::sqrt(variance), fourth / variance / variance};
}

}  // namespace corehydro::numerics::distributions::distribution_moment_integration
