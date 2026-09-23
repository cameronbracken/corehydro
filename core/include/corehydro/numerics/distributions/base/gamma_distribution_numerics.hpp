// ported from: Numerics/Distributions/Univariate/Base/GammaDistributionNumerics.cs @ 7e8e8d1
#pragma once
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/distribution_numerics.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"

namespace corehydro::numerics::distributions::distribution_numerics {

namespace detail {

inline void validate_gamma_shape(double shape) {
    if (!(shape > 0.0) || !finite(shape)) throw std::out_of_range("gamma shape must be positive and finite");
}

inline const std::array<double, 58>& integer_zeta_values() {
    static const std::array<double, 58> values = [] {
        std::array<double, 58> result{};
        const double literals[] = {
            1.6449340668482264365, 1.2020569031595942854, 1.0823232337111381915,
            1.0369277551433699263, 1.0173430619844491397, 1.0083492773819228268,
            1.0040773561979443394, 1.0020083928260822144, 1.0009945751278180853,
            1.0004941886041194646, 1.0002460865533080483, 1.0001227133475784891,
            1.0000612481350587048, 1.0000305882363070205, 1.0000152822594086519};
        for (std::size_t i = 0; i < 15; ++i) result[i] = literals[i];
        for (int n = 17; n < 60; ++n) {
            double sum = 1.0;
            for (int k = 2; k <= 32; ++k) sum += std::pow(k, -n);
            result[static_cast<std::size_t>(n - 2)] = sum;
        }
        return result;
    }();
    return values;
}

inline double zeta_integer(int n) {
    if (n < 2) throw std::out_of_range("zeta integer argument must be at least two");
    if (n < 60) return integer_zeta_values()[static_cast<std::size_t>(n - 2)];
    double sum = 1.0;
    for (int k = 2; k <= 32; ++k) sum += std::pow(k, -n);
    return sum;
}

inline double log_gamma_one_plus(double a) {
    if (a > 0.5) return math::special::log_gamma(a + 1.0);
    double result = -0.57721566490153286061 * a;
    double power = a;
    for (int n = 2; n < 60; ++n) {
        power *= -a;
        double term = -power * zeta_integer(n) / n;
        result += term;
        if (std::fabs(term) <= std::fabs(result) * 1e-17) break;
    }
    return result;
}

inline double log1p_minus_x(double x) {
    if (std::fabs(x) >= 0.25) return std::log1p(x) - x;
    double power = -x * x;
    double sum = power / 2.0;
    for (int n = 3; n < 100; ++n) {
        power *= -x;
        double term = power / n;
        sum += term;
        if (std::fabs(term) <= std::fabs(sum) * 1e-17) break;
    }
    return sum;
}

inline double stirling_remainder(double a) {
    double r = 1.0 / a;
    double r2 = r * r;
    return r * (1.0 / 12.0 + r2 * (-1.0 / 360.0 + r2 * (1.0 / 1260.0 +
        r2 * (-1.0 / 1680.0 + r2 * (1.0 / 1188.0 - r2 * 691.0 / 360360.0)))));
}

inline double gamma_log_kernel(double a, double x) {
    if (a < 16.0) return a * std::log(x) - x - math::special::log_gamma(a);
    double delta = (x - a) / a;
    double deviance = std::fabs(delta) < 0.25 ? a * log1p_minus_x(delta)
                                              : a * (std::log(x) - std::log(a)) - (x - a);
    return deviance + 0.5 * std::log(a) - kLogSqrt2PI - stirling_remainder(a);
}

inline double gamma_kernel_shape_derivative(double a, double x) {
    if (a < 16.0) return std::log(x) - math::special::digamma(a);
    double inverse = 1.0 / a;
    double square = inverse * inverse;
    double residual = 0.5 * inverse + square * (1.0 / 12.0 - square * (1.0 / 120.0 -
        square * (1.0 / 252.0 - square / 240.0)));
    double delta = (x - a) / a;
    return (std::fabs(delta) < 0.25 ? std::log1p(delta) : std::log(x) - std::log(a)) + residual;
}

inline const std::vector<std::vector<double>>& temme_coefficients() {
    static const std::vector<std::vector<double>> values = {
        {-.3333333333333333333,.0833333333333333333,-.01481481481481481481,.001157407407407407407,.0003527336860670194,-.0001787551440329218,3.919263178522438E-5,-2.185448510679992E-6,-1.85406221071516E-6,8.296711340953087E-7,-1.766595273682608E-7,6.707853543401498E-9,1.026180978424031E-8,-4.382036018453353E-9,9.14769958223679E-10,-2.551419399494625E-11,-5.830772132550426E-11,2.436194802066742E-11},
        {-.001851851851851851852,-.003472222222222222222,.002645502645502645503,-.0009902263374485596,.000205761316872428,-4.018775720164609E-7,-1.809855033448998E-5,7.64916091608111E-6,-1.612090089456345E-6,4.647127802807434E-9,1.378633446915721E-7,-5.752545603517705E-8,1.195162859977815E-8,-1.754324171974765E-11,-1.009154371060041E-9,4.162792991842583E-10},
        {.004133597883597883598,-.00268132716049382716,.0007716049382716049,2.009387860082305E-6,-.0001073665322636516,5.292344882912013E-5,-1.276063518861873E-5,3.423578734096138E-8,1.372195730906293E-6,-6.298992138380055E-7,1.428061420606424E-7,-2.047709842199087E-10,-1.409252991086752E-8,6.228974084922022E-9},
        {.0006494341563786008,.0002294720936213992,-.0004691894943952557,.0002677206320628389,-7.561801671883977E-5,-2.396505113867297E-7,1.10826541153473E-5,-5.674952826991597E-6,1.423090073243588E-6,-2.786108029152814E-11,-1.695840409193028E-7,8.099464905388083E-8}};
    return values;
}

inline double gamma_temme(double a, double delta, bool upper, double& derivative) {
    double eta = delta == 0.0 ? 0.0 : std::copysign(std::sqrt(-2.0 * log1p_minus_x(delta)), delta);
    double root = std::sqrt(a);
    double z = eta * root;
    double eta_derivative = delta == 0.0 ? -1.0 / a : -delta / eta / a;
    double z_derivative = eta / (2.0 * root) + root * eta_derivative;
    double series = 0.0, series_eta = 0.0, series_a = 0.0, inverse_power = 1.0;
    const auto& tables = temme_coefficients();
    for (std::size_t k = 0; k < tables.size(); ++k) {
        const auto& coefficients = tables[k];
        double value = coefficients.back();
        double dvalue = 0.0;
        for (int n = static_cast<int>(coefficients.size()) - 2; n >= 0; --n) {
            dvalue = dvalue * eta + value;
            value = value * eta + coefficients[static_cast<std::size_t>(n)];
        }
        series += inverse_power * value;
        series_eta += inverse_power * dvalue;
        series_a -= static_cast<double>(k) * inverse_power / a * value;
        inverse_power /= a;
    }
    double normal = upper ? normal_log_survival(z) : normal_log_cdf(z);
    double log_phi = -(0.5 * z) * z - kLogSqrt2PI;
    double signed_correction = (upper ? 1.0 : -1.0) * series *
                               std::exp(log_phi - 0.5 * std::log(a) - normal);
    double log_value = normal + std::log1p(signed_correction);
    double correction_derivative =
        (series_eta * eta_derivative + series_a - (z * z_derivative + 0.5 / a) * series) /
        root;
    derivative = std::exp(log_phi - log_value) *
                 (upper ? -z_derivative + correction_derivative
                        : z_derivative - correction_derivative);
    return log_value;
}

inline double gamma_small_upper(double a, double x, double& derivative) {
    double sum = 0.0, derivative_sum = 0.0, power = 1.0;
    for (int n = 1; n <= 1000; ++n) {
        power *= -x / n;
        double term = power / (a + n);
        sum += term;
        derivative_sum -= term / (a + n);
        if (std::fabs(term) < std::fabs(sum) * 2e-16) break;
    }
    double log_x = std::log(x);
    double u = a * log_x - log_gamma_one_plus(a);
    double leading = std::exp(u);
    double factor = a * leading;
    double q = -std::expm1(u) - factor * sum;
    double du = log_x - math::special::digamma(a + 1.0);
    derivative = (-leading * du - factor * (du * sum + derivative_sum) - leading * sum) / q;
    return std::log(q);
}

inline double gamma_log_tail(double a, double x, bool upper, double& derivative) {
    validate_gamma_shape(a);
    derivative = 0.0;
    if (std::isnan(x)) {
        derivative = x;
        return x;
    }
    if (x <= 0.0) return upper ? 0.0 : -std::numeric_limits<double>::infinity();
    if (x == std::numeric_limits<double>::infinity())
        return upper ? -std::numeric_limits<double>::infinity() : 0.0;
    double delta = (x - a) / a;
    if (a >= 10000.0 && std::fabs(delta) < 0.1) return gamma_temme(a, delta, upper, derivative);
    if (a < 1.0 && x <= 1.0 &&
        (upper || a * std::log(x) - log_gamma_one_plus(a) > -0.6931471805599453)) {
        double log_q = gamma_small_upper(a, x, derivative);
        if (upper) return log_q;
        double log_p = log1m_exp(log_q);
        derivative = -derivative * std::exp(log_q - log_p);
        return log_p;
    }
    if (x < a + 1.0) {
        double sum = 1.0, term = 1.0, dsum = 0.0, dterm = 0.0;
        for (int n = 1; n <= 100000; ++n) {
            double ratio = x / (a + n);
            dterm = dterm * ratio - term * ratio / (a + n);
            term *= ratio;
            sum += term;
            dsum += dterm;
            if (term <= sum * 2e-16 &&
                std::fabs(dterm) <= std::max(1.0, std::fabs(dsum)) * 2e-16) {
                double log_value = a < 16.0
                                       ? a * std::log(x) - x - log_gamma_one_plus(a) + std::log(sum)
                                       : gamma_log_kernel(a, x) - std::log(a) + std::log(sum);
                double dlog = a < 16.0
                                  ? std::log(x) - math::special::digamma(a + 1.0) + dsum / sum
                                  : gamma_kernel_shape_derivative(a, x) - 1.0 / a + dsum / sum;
                if (!upper) {
                    derivative = dlog;
                    return log_value;
                }
                double complement = log1m_exp(log_value);
                derivative = -dlog * std::exp(log_value - complement);
                return complement;
            }
        }
        throw std::runtime_error("lower gamma series did not converge");
    }
    constexpr double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, dc = 0.0;
    double d = 1.0 / b, dd = d * d, h = d, dh = dd;
    for (int n = 1; n <= 100000; ++n) {
        double an = n * (a - n);
        b += 2.0;
        double denominator = an * d + b;
        double derivative_denominator = n * d + an * dd - 1.0;
        double next_c = b + an / c;
        double next_dc = -1.0 + n / c - an * dc / (c * c);
        if (std::fabs(denominator) < tiny) denominator = std::copysign(tiny, denominator);
        if (std::fabs(next_c) < tiny) next_c = std::copysign(tiny, next_c);
        d = 1.0 / denominator;
        dd = -derivative_denominator * d * d;
        c = next_c;
        dc = next_dc;
        double factor = d * c;
        double derivative_factor = dd * c + d * dc;
        dh = dh * factor + h * derivative_factor;
        h *= factor;
        if (std::fabs(factor - 1.0) <= 4e-16 && std::fabs(derivative_factor) <= 4e-16) {
            double log_value = gamma_log_kernel(a, x) + std::log(h);
            double dlog = gamma_kernel_shape_derivative(a, x) + dh / h;
            if (upper) {
                derivative = dlog;
                return log_value;
            }
            double complement = log1m_exp(log_value);
            derivative = -dlog * std::exp(log_value - complement);
            return complement;
        }
    }
    throw std::runtime_error("upper gamma continued fraction did not converge");
}

}  // namespace detail

inline double accurate_trigamma(double shape) {
    detail::validate_gamma_shape(shape);
    double sum = 0.0;
    while (shape < 32.0) {
        double inverse = 1.0 / shape;
        sum += inverse * inverse;
        shape += 1.0;
    }
    double r = 1.0 / shape, s = r * r;
    return sum + r + 0.5 * s + r * s * (1.0 / 6.0 + s * (-1.0 / 30.0 +
        s * (1.0 / 42.0 + s * (-1.0 / 30.0 + s * (5.0 / 66.0 - s * 691.0 / 2730.0)))));
}

inline double gamma_log_cdf(double shape, double x) {
    double derivative;
    return detail::gamma_log_tail(shape, x, false, derivative);
}

inline double gamma_log_survival(double shape, double x) {
    double derivative;
    return detail::gamma_log_tail(shape, x, true, derivative);
}

inline double gamma_log_density(double shape, double x) {
    detail::validate_gamma_shape(shape);
    if (std::isnan(x)) return x;
    if (x < 0.0 || x == std::numeric_limits<double>::infinity())
        return -std::numeric_limits<double>::infinity();
    if (x == 0.0)
        return shape == 1.0 ? 0.0 : shape < 1.0 ? std::numeric_limits<double>::infinity()
                                                    : -std::numeric_limits<double>::infinity();
    return shape == 1.0 ? -x : detail::gamma_log_kernel(shape, x) - std::log(x);
}

inline double gamma_quantile_shape_derivative(double shape, double unit_quantile) {
    detail::validate_gamma_shape(shape);
    if (unit_quantile == 0.0) return 0.0;
    if (!(unit_quantile > 0.0) || !finite(unit_quantile))
        throw std::out_of_range("unit quantile must be positive and finite");
    bool upper = unit_quantile >= shape;
    double derivative;
    double log_value = detail::gamma_log_tail(shape, unit_quantile, upper, derivative);
    if (derivative == 0.0) return 0.0;
    return (upper ? std::copysign(1.0, derivative) : -std::copysign(1.0, derivative)) *
           std::exp(std::log(std::fabs(derivative)) + log_value -
                    gamma_log_density(shape, unit_quantile));
}

inline double gamma_inverse_cdf(double shape, double probability, bool upper_tail = false) {
    detail::validate_gamma_shape(shape);
    if (!(probability >= 0.0 && probability <= 1.0))
        throw std::out_of_range("probability outside closed unit interval");
    if (probability == 0.0) return upper_tail ? std::numeric_limits<double>::infinity() : 0.0;
    if (probability == 1.0) return upper_tail ? 0.0 : std::numeric_limits<double>::infinity();
    if (probability > 0.5) {
        probability = 1.0 - probability;
        upper_tail = !upper_tail;
    }
    double target = std::log(probability);
    if (shape == 1.0) return upper_tail ? -target : -std::log1p(-probability);
    double small_log = (target + detail::log_gamma_one_plus(shape)) / shape;
    if (!upper_tail && small_log < -36.0) return std::exp(small_log);
    double root = std::sqrt(shape);
    double z = math::special::detail::normal_standard_z(probability) * (upper_tail ? -1.0 : 1.0);
    double w = 1.0 - 1.0 / (9.0 * shape) + z / (3.0 * root);
    double guess = w > 0.0 ? shape * w * w * w : std::exp(small_log);
    if (!(guess > 0.0) || !finite(guess)) guess = std::max(shape, 1.0);
    double lower = 0.0, upper = std::max(std::max(shape, 1.0), guess);
    auto below = [&](double value) {
        double derivative;
        double log_value = detail::gamma_log_tail(shape, value, upper_tail, derivative);
        return upper_tail ? log_value > target : log_value < target;
    };
    while (below(upper)) {
        lower = upper;
        upper = upper < std::numeric_limits<double>::max() / 2.0
                    ? upper * 2.0
                    : std::numeric_limits<double>::max();
        if (lower == upper) return std::numeric_limits<double>::infinity();
    }
    double x = guess > lower && guess < upper ? guess : lower + (upper - lower) / 2.0;
    for (int iteration = 0; iteration < 100; ++iteration) {
        double derivative;
        double log_value = detail::gamma_log_tail(shape, x, upper_tail, derivative);
        double residual = log_value - target;
        if (std::fabs(residual) <= 8e-15 * std::max(1.0, std::fabs(target))) return x;
        if (upper_tail ? residual > 0.0 : residual < 0.0) lower = x;
        else upper = x;
        double slope = std::exp(gamma_log_density(shape, x) - log_value) *
                       (upper_tail ? -1.0 : 1.0);
        double next = x - residual / slope;
        if (!(next > lower && next < upper) || !finite(next)) next = lower + (upper - lower) / 2.0;
        if (next == x || next == lower || next == upper) return next;
        x = next;
    }
    throw std::runtime_error("gamma quantile solve did not converge");
}

inline double log1p_minus_x(double x) { return detail::log1p_minus_x(x); }
inline double log_gamma_one_plus(double a) { return detail::log_gamma_one_plus(a); }
inline double zeta_integer(int n) { return detail::zeta_integer(n); }

}  // namespace corehydro::numerics::distributions::distribution_numerics
