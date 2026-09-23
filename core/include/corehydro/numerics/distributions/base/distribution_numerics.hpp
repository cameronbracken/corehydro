// ported from: Numerics/Distributions/Univariate/Base/DistributionNumerics.cs @ 7e8e8d1
//           +  Numerics/Distributions/Univariate/Base/DistributionParameterBounds.cs @ 7e8e8d1
//           +  Numerics/Distributions/Univariate/Base/DistributionTailTransform.cs @ 7e8e8d1
//           +  Numerics/Distributions/Univariate/Base/DistributionUncertaintyNumerics.cs @ 7e8e8d1
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "corehydro/numerics/distributions/base/i_standard_error.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions::distribution_numerics {

using Constraints = std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>;

inline bool finite(double value) { return std::isfinite(value); }

template <class Legacy, class Fallback>
Constraints prefer_legacy_constraints(Legacy&& legacy, Fallback&& fallback) {
    try {
        Constraints result = legacy();
        const auto& initial = std::get<0>(result);
        const auto& lower = std::get<1>(result);
        const auto& upper = std::get<2>(result);
        bool usable = initial.size() == lower.size() && lower.size() == upper.size();
        for (std::size_t i = 0; usable && i < initial.size(); ++i)
            usable = finite(initial[i]) && finite(lower[i]) && finite(upper[i]) &&
                     lower[i] < upper[i] && lower[i] <= initial[i] && initial[i] <= upper[i];
        if (usable) return result;
    } catch (const std::exception&) {
    }
    return fallback();
}

inline double standardize(double x, double location, double scale) {
    double difference = x - location;
    return std::isinf(difference) && finite(x) && finite(location)
               ? x / scale - location / scale
               : difference / scale;
}

inline double log1m_exp(double a) {
    if (a > 0.0 || std::isnan(a)) return std::numeric_limits<double>::quiet_NaN();
    return a < -0.69314718055994530942 ? std::log1p(-std::exp(a)) : std::log(-std::expm1(a));
}

inline double log_difference(double a, double b) {
    if (std::isnan(a) || std::isnan(b) || b > a) return std::numeric_limits<double>::quiet_NaN();
    if (a == b) return -std::numeric_limits<double>::infinity();
    return a + log1m_exp(b - a);
}

inline double log_sum(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    if (a == std::numeric_limits<double>::infinity() ||
        b == std::numeric_limits<double>::infinity())
        return std::numeric_limits<double>::infinity();
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;
    double larger = std::max(a, b);
    return larger + std::log1p(std::exp(std::min(a, b) - larger));
}

inline double normal_log_cdf(double z) {
    if (std::isnan(z)) return z;
    if (z > 0.0) return log1m_exp(normal_log_cdf(-z));
    if (z >= -10.0) return std::log(0.5 * std::erfc(-z / kSqrt2));
    if (z == -std::numeric_limits<double>::infinity()) return z;
    double x = -z;
    double fraction = 0.0;
    for (int i = 64; i >= 1; --i) fraction = i / (x + fraction);
    return -(0.5 * x) * x - kLogSqrt2PI - std::log(x + fraction);
}

inline double normal_log_survival(double z) { return normal_log_cdf(-z); }

inline double exprel(double x) { return x == 0.0 ? 1.0 : std::expm1(x) / x; }

inline double exprel_derivative(double x) {
    if (std::fabs(x) >= 0.1) return ((x - 1.0) * std::exp(x) + 1.0) / (x * x);
    double sum = 0.5;
    double term = 0.5;
    for (int n = 1; n < 20; ++n) {
        term *= x * (n + 1.0) / n / (n + 2.0);
        sum += term;
        if (std::fabs(term) <= std::fabs(sum) * 1e-17) break;
    }
    return sum;
}

inline void validate_probability(double probability) {
    if (!(probability > 0.0 && probability < 1.0))
        throw std::out_of_range("probability must be finite and strictly between zero and one");
}

inline void validate_sample_size(int sample_size) {
    if (sample_size <= 0) throw std::out_of_range("sample size must be positive");
}

inline void validate_confidence_inputs(int sample_size, const std::vector<double>& quantiles,
                                       const std::vector<double>& percentiles,
                                       int minimum_sample_size = 1) {
    if (sample_size < minimum_sample_size || quantiles.empty() || percentiles.empty())
        throw std::out_of_range("insufficient observations or empty probability list");
    for (double p : quantiles) validate_probability(p);
    for (double p : percentiles) validate_probability(p);
}

inline void validate_sample(const std::vector<double>& sample, int minimum_count = 2,
                            bool positive = false) {
    if (sample.size() < static_cast<std::size_t>(minimum_count))
        throw std::out_of_range("insufficient observations to initialize the distribution");
    bool distinct = false;
    for (double value : sample) {
        if (!finite(value) || (positive && value <= 0.0))
            throw std::out_of_range("observations do not satisfy the initialization domain");
        distinct = distinct || value != sample.front();
    }
    if (!distinct) throw std::out_of_range("a constant sample cannot initialize a positive scale");
}

inline math::linalg::Matrix2D quantile_gradient_matrix(
    const IStandardError& distribution, int parameter_count,
    const std::vector<double>& probabilities) {
    if (probabilities.empty() || probabilities.size() != static_cast<std::size_t>(parameter_count))
        throw std::out_of_range("provide one probability per distribution parameter");
    math::linalg::Matrix2D matrix(probabilities.size(),
                                  std::vector<double>(probabilities.size()));
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        validate_probability(probabilities[i]);
        auto gradient = distribution.quantile_gradient(probabilities[i]);
        if (gradient.size() != probabilities.size())
            throw std::out_of_range("quantile Jacobian must be square");
        for (std::size_t j = 0; j < gradient.size(); ++j) {
            if (!finite(gradient[j])) throw std::runtime_error("nonfinite quantile derivative");
            matrix[i][j] = gradient[j];
        }
    }
    return matrix;
}

inline double log_abs_determinant(const math::linalg::Matrix2D& matrix, int& sign) {
    const std::size_t n = matrix.size();
    for (const auto& row : matrix)
        if (row.size() != n) throw std::invalid_argument("matrix must be square");
    if (n == 0) {
        sign = 1;
        return 0.0;
    }
    if (n == 2) {
        const double a = matrix[0][0], b = matrix[0][1];
        const double c = matrix[1][0], d = matrix[1][1];
        if (!finite(a) || !finite(b) || !finite(c) || !finite(d))
            throw std::runtime_error("determinant requires finite entries");
        const int ad_sign = (a == 0.0 || d == 0.0) ? 0 : (std::signbit(a) == std::signbit(d) ? 1 : -1);
        const int bc_sign = (b == 0.0 || c == 0.0) ? 0 : (std::signbit(b) == std::signbit(c) ? 1 : -1);
        const double ad = ad_sign == 0 ? -std::numeric_limits<double>::infinity()
                                       : std::log(std::fabs(a)) + std::log(std::fabs(d));
        const double bc = bc_sign == 0 ? -std::numeric_limits<double>::infinity()
                                       : std::log(std::fabs(b)) + std::log(std::fabs(c));
        if (ad_sign == 0 && bc_sign == 0) {
            sign = 0;
            return -std::numeric_limits<double>::infinity();
        }
        if (ad_sign != bc_sign) {
            sign = ad_sign == 0 ? -bc_sign : ad_sign;
            return log_sum(ad, bc);
        }
        if (ad == bc) {
            sign = 0;
            return -std::numeric_limits<double>::infinity();
        }
        sign = (ad > bc ? ad_sign : -bc_sign);
        return log_difference(std::max(ad, bc), std::min(ad, bc));
    }
    math::linalg::Matrix2D a = matrix;
    double log_value = 0.0;
    sign = 1;
    for (std::size_t i = 0; i < n; ++i) {
        double scale = 0.0;
        for (double value : a[i]) scale = std::max(scale, std::fabs(value));
        if (!finite(scale)) throw std::runtime_error("determinant requires finite entries");
        if (scale == 0.0) {
            sign = 0;
            return -std::numeric_limits<double>::infinity();
        }
        log_value += std::log(scale);
        for (double& value : a[i]) value /= scale;
    }
    for (std::size_t j = 0; j < n; ++j) {
        std::size_t pivot = j;
        for (std::size_t i = j + 1; i < n; ++i)
            if (std::fabs(a[i][j]) > std::fabs(a[pivot][j])) pivot = i;
        if (std::fabs(a[pivot][j]) < 1e-14) {
            sign = 0;
            return -std::numeric_limits<double>::infinity();
        }
        if (pivot != j) {
            std::swap(a[pivot], a[j]);
            sign = -sign;
        }
        double diagonal = a[j][j];
        sign *= diagonal < 0.0 ? -1 : 1;
        log_value += std::log(std::fabs(diagonal));
        for (std::size_t i = j + 1; i < n; ++i) {
            double ratio = a[i][j] / diagonal;
            for (std::size_t k = j + 1; k < n; ++k) a[i][k] -= ratio * a[j][k];
        }
    }
    return log_value;
}

inline math::linalg::Matrix2D quantile_jacobian(const IStandardError& distribution,
                                                 int parameter_count,
                                                 const std::vector<double>& probabilities,
                                                 double& determinant) {
    auto matrix = quantile_gradient_matrix(distribution, parameter_count, probabilities);
    int sign = 0;
    double logarithm = log_abs_determinant(matrix, sign);
    determinant = sign == 0 ? 0.0 : sign * std::exp(logarithm);
    return matrix;
}

inline double initialization_scale(const std::vector<double>& sample) {
    double scale = 0.0;
    for (double value : sample) scale = std::max(scale, std::fabs(value));
    if (!(scale > 0.0) || !finite(scale))
        throw std::out_of_range("finite nonzero sample magnitude required");
    return scale;
}

inline void positive_parameter_bounds(double initial, double& lower, double& upper) {
    if (!(initial > 0.0) || !finite(initial))
        throw std::out_of_range("initial parameter must be finite and positive");
    lower = std::max(std::numeric_limits<double>::denorm_min(),
                     std::min(kDoubleMachineEpsilon, initial / 10.0));
    double decade = std::pow(10.0, std::ceil(std::log10(initial) + 1.0));
    upper = finite(decade) ? std::max(initial, decade) : std::numeric_limits<double>::max();
    if (!(lower < upper)) throw std::out_of_range("ordered parameter bounds cannot be represented");
}

inline void location_parameter_bounds(double& initial, double scale, double data_minimum,
                                      double data_maximum, bool upper_at_minimum, double& lower,
                                      double& upper) {
    if (!finite(initial) || !(scale > 0.0) || !finite(scale))
        throw std::out_of_range("finite location and positive scale required");
    double magnitude = std::max(
        std::fabs(initial),
        std::max(scale, std::max(std::fabs(data_minimum), std::fabs(data_maximum))));
    double radius = std::pow(10.0, std::ceil(std::log10(magnitude) + 1.0));
    if (!finite(radius)) radius = std::numeric_limits<double>::max();
    lower = -radius;
    upper = upper_at_minimum ? data_minimum : radius;
    if (!(lower < upper)) throw std::out_of_range("ordered location bounds cannot be represented");
    if (initial < lower || initial > upper) initial = lower / 2.0 + upper / 2.0;
}

inline double hosking_shape_transform(double x, double location, double scale, double shape) {
    double standardized = standardize(x, location, scale);
    if (shape == 0.0 || std::isnan(standardized)) return standardized;
    double product = shape * standardized;
    if (finite(product)) return product == 0.0 ? standardized : -std::log1p(-product) / shape;
    if (!finite(x) || !finite(location)) return -std::log1p(-product) / shape;
    double difference = x - location;
    double log_difference_value;
    if (finite(difference)) {
        log_difference_value = std::log(std::fabs(difference));
    } else {
        double magnitude = std::max(std::fabs(x), std::fabs(location));
        log_difference_value =
            std::log(magnitude) + std::log(std::fabs(x / magnitude - location / magnitude));
    }
    double log_product = std::log(std::fabs(shape)) + log_difference_value - std::log(scale);
    bool negative_product = shape > 0.0 ? x < location : x > location;
    double log_support = negative_product ? log_sum(0.0, log_product) : log1m_exp(log_product);
    return -log_support / shape;
}

inline double scaled_quantile_variance(const math::linalg::Matrix2D& covariance,
                                       const std::vector<double>& gradient, double scale = 1.0) {
    if (!(scale > 0.0) || !finite(scale) || covariance.size() != gradient.size())
        throw std::out_of_range("invalid covariance, gradient, or scale");
    for (const auto& row : covariance)
        if (row.size() != gradient.size()) throw std::out_of_range("dimensions must agree");
    std::vector<double> logs(gradient.size());
    for (std::size_t i = 0; i < gradient.size(); ++i) {
        if (!finite(gradient[i])) throw std::runtime_error("quantile gradient is nonfinite");
        logs[i] = std::log(std::fabs(gradient[i]));
        for (double value : covariance[i])
            if (!finite(value)) throw std::runtime_error("covariance is nonfinite");
    }
    double largest = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < gradient.size(); ++i)
        for (std::size_t j = 0; j < gradient.size(); ++j)
            if (covariance[i][j] != 0.0 && gradient[i] != 0.0 && gradient[j] != 0.0)
                largest = std::max(largest, std::log(std::fabs(covariance[i][j])) + logs[i] + logs[j]);
    if (largest == -std::numeric_limits<double>::infinity()) return 0.0;
    double quadratic = 0.0;
    double correction = 0.0;
    for (std::size_t i = 0; i < gradient.size(); ++i) {
        for (std::size_t j = 0; j < gradient.size(); ++j) {
            if (covariance[i][j] == 0.0 || gradient[i] == 0.0 || gradient[j] == 0.0) continue;
            double term = std::copysign(1.0, covariance[i][j] * gradient[i] * gradient[j]) *
                          std::exp(std::log(std::fabs(covariance[i][j])) + logs[i] + logs[j] - largest);
            double next = quadratic + term;
            correction += std::fabs(quadratic) >= std::fabs(term) ? (quadratic - next) + term
                                                                  : (term - next) + quadratic;
            quadratic = next;
        }
    }
    quadratic += correction;
    if (!finite(quadratic) || quadratic < 0.0)
        throw std::runtime_error("quantile variance is not a nonnegative quadratic form");
    if (quadratic == 0.0) return 0.0;
    return std::exp(2.0 * std::log(scale) + largest + std::log(quadratic));
}

inline double gamma_scaled_fisher_residual(double shape) {
    if (!(shape > 0.0) || !finite(shape)) throw std::out_of_range("shape must be positive");
    double shifted = shape;
    double recurrence = 0.0;
    while (shifted < 32.0) {
        double ratio = shape / shifted;
        recurrence += ratio * ratio;
        shifted += 1.0;
    }
    double r = 1.0 / shifted;
    double s = r * r;
    double residual = 0.5 + r * (1.0 / 6.0 + s * (-1.0 / 30.0 + s * (1.0 / 42.0 +
        s * (-1.0 / 30.0 + s * (5.0 / 66.0 - s * 691.0 / 2730.0)))));
    if (shifted == shape) return residual;
    double scale_ratio = shape / shifted;
    return recurrence - shape + shape * scale_ratio + scale_ratio * scale_ratio * residual;
}

inline double scaled_exprel_product_core(double scale, double value, double argument,
                                         bool derivative) {
    if (value == 0.0) return 0.0;
    if (argument == std::numeric_limits<double>::infinity())
        return derivative ? argument : std::copysign(argument, value);
    if (argument == -std::numeric_limits<double>::infinity()) return 0.0;
    double divided = derivative ? exprel_derivative(argument) : exprel(argument);
    double factor = derivative ? value * value : value;
    double result = scale * (factor * divided);
    if (finite(result) && result != 0.0) return result;
    double log_divided;
    if (argument > 50.0)
        log_divided = derivative
                          ? argument + std::log(argument - 1.0) +
                                std::log1p(std::exp(-argument) / (argument - 1.0)) -
                                2.0 * std::log(argument)
                          : argument + std::log1p(-std::exp(-argument)) - std::log(argument);
    else if (argument < -50.0)
        log_divided = derivative
                          ? std::log1p((argument - 1.0) * std::exp(argument)) -
                                2.0 * std::log(-argument)
                          : std::log(-std::expm1(argument)) - std::log(-argument);
    else
        log_divided = std::log(divided);
    double logarithm = std::log(scale) + (derivative ? 2.0 : 1.0) * std::log(std::fabs(value)) +
                       log_divided;
    return (derivative ? 1.0 : std::copysign(1.0, value)) * std::exp(logarithm);
}

inline double scaled_exprel_product(double scale, double value, double argument) {
    return scaled_exprel_product_core(scale, value, argument, false);
}

inline double scaled_exprel_derivative_product(double scale, double value, double argument) {
    return scaled_exprel_product_core(scale, value, argument, true);
}

inline bool is_continuous(UnivariateDistributionType type) {
    switch (type) {
        case UnivariateDistributionType::Normal:
        case UnivariateDistributionType::Logistic:
        case UnivariateDistributionType::LnNormal:
        case UnivariateDistributionType::LogNormal:
        case UnivariateDistributionType::PearsonTypeIII:
        case UnivariateDistributionType::LogPearsonTypeIII:
        case UnivariateDistributionType::Exponential:
        case UnivariateDistributionType::GammaDistribution:
        case UnivariateDistributionType::Weibull:
        case UnivariateDistributionType::Gumbel:
        case UnivariateDistributionType::GeneralizedExtremeValue:
        case UnivariateDistributionType::GeneralizedPareto:
        case UnivariateDistributionType::GeneralizedNormal:
        case UnivariateDistributionType::GeneralizedLogistic:
        case UnivariateDistributionType::KappaFour:
        case UnivariateDistributionType::Uniform:
            return true;
        default:
            return false;
    }
}

template <class Distribution>
double collapsed_continuous_log_interval(const Distribution& distribution, double lower,
                                         double upper) {
    if (!is_continuous(distribution.type())) return -std::numeric_limits<double>::infinity();
    lower = std::max(lower, distribution.minimum());
    upper = std::min(upper, distribution.maximum());
    const double width = upper - lower;
    if (!(width > 0.0) || !finite(width)) return -std::numeric_limits<double>::infinity();
    constexpr double nodes[] = {0.019855071751231884, 0.10166676129318663,
                                0.23723379504183551,  0.4082826787521751,
                                0.5917173212478249,   0.7627662049581645,
                                0.8983332387068134,   0.9801449282487681};
    constexpr double weights[] = {0.05061426814518813, 0.11119051722668724,
                                  0.15685332293894365, 0.181341891689181,
                                  0.181341891689181,   0.15685332293894365,
                                  0.11119051722668724, 0.05061426814518813};
    double sum = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < 8; ++i)
        sum = log_sum(sum, std::log(weights[i]) +
                               distribution.log_pdf(lower + width * nodes[i]));
    return std::log(width) + sum;
}

}  // namespace corehydro::numerics::distributions::distribution_numerics
