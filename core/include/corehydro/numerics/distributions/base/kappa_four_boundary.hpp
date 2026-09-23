// ported from: Numerics/Distributions/Univariate/Base/KappaFourBoundary.cs @ 7e8e8d1
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace corehydro::numerics::distributions::distribution_numerics {

class KappaFourBoundary {
   private:
    struct Pair {
        double high;
        double low;
        explicit Pair(double high_value, double low_value = 0.0)
            : high(high_value), low(low_value) {}
    };

    static Pair log_two() { return Pair(0.6931471805599453, 2.3190468138462996e-17); }

    static Pair normalize(double high, double low) {
        const double sum = high + low;
        if (!std::isfinite(sum)) return Pair(sum);
        const double part = sum - high;
        return Pair(sum, (high - (sum - part)) + (low - part));
    }

    static Pair add(Pair first, Pair second) {
        const double sum = first.high + second.high;
        if (!std::isfinite(sum)) return Pair(sum);
        const double part = sum - first.high;
        const double error = (first.high - (sum - part)) + (second.high - part);
        return normalize(sum, error + first.low + second.low);
    }

    static Pair subtract(Pair first, Pair second) {
        return add(first, Pair(-second.high, -second.low));
    }

    static Pair abs_pair(Pair value) {
        return value.high < 0.0 ? Pair(-value.high, -value.low) : value;
    }

    static std::uint64_t bits(double value) {
        std::uint64_t result;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }

    static double from_bits(std::uint64_t value) {
        double result;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }

    static Pair multiply(Pair first, Pair second) {
        const double product = first.high * second.high;
        if (!std::isfinite(product)) return Pair(product);
        constexpr std::uint64_t mask = ~((std::uint64_t{1} << 27) - 1);
        const double first_high = from_bits(bits(first.high) & mask);
        const double second_high = from_bits(bits(second.high) & mask);
        const double first_low = first.high - first_high;
        const double second_low = second.high - second_high;
        double error = ((first_high * second_high - product) + first_high * second_low +
                        first_low * second_high) +
                       first_low * second_low;
        error += first.high * second.low + first.low * second.high + first.low * second.low;
        return normalize(product, error);
    }

    static Pair divide(Pair numerator, Pair denominator) {
        const double quotient = numerator.high / denominator.high;
        if (!std::isfinite(quotient)) return Pair(quotient);
        Pair result(quotient);
        Pair residual = subtract(numerator, multiply(denominator, result));
        const double correction = residual.high / denominator.high;
        result = add(result, Pair(correction));
        residual = subtract(residual, multiply(denominator, Pair(correction)));
        return add(result, Pair(residual.high / denominator.high));
    }

    static Pair scale(Pair value, int exponent) {
        while (exponent > 512) {
            value = Pair(value.high * 1.3407807929942597e154,
                         value.low * 1.3407807929942597e154);
            exponent -= 512;
        }
        while (exponent < -512) {
            value = Pair(value.high * 7.458340731200207e-155,
                         value.low * 7.458340731200207e-155);
            exponent += 512;
        }
        const double factor = std::pow(2.0, exponent);
        return Pair(value.high * factor, value.low * factor);
    }

    static Pair log_pair(Pair value) {
        if (!(value.high > 0.0) || std::isinf(value.high)) return Pair(std::log(value.high));
        double leading = value.high;
        int correction = 0;
        if (leading < 2.2250738585072014e-308) {
            leading *= 18014398509481984.0;
            correction = -54;
        }
        int exponent = static_cast<int>((bits(leading) >> 52) & 0x7ffU) - 1023 + correction;
        Pair reduced = scale(value, -exponent);
        if (reduced.high > 1.4142135623730951) {
            reduced = scale(reduced, -1);
            ++exponent;
        }
        const Pair ratio = divide(subtract(reduced, Pair(1.0)), add(reduced, Pair(1.0)));
        const Pair square = multiply(ratio, ratio);
        Pair term = ratio;
        Pair sum = ratio;
        for (int index = 1; index <= 24; ++index) {
            term = multiply(term, square);
            sum = add(sum, divide(term, Pair(2.0 * index + 1.0)));
        }
        return add(multiply(sum, Pair(2.0)), multiply(log_two(), Pair(exponent)));
    }

    static Pair log_one_plus_relative(Pair value) {
        Pair sum(1.0);
        Pair term(1.0);
        const Pair negative(-value.high, -value.low);
        for (int index = 1; index <= 40; ++index) {
            term = multiply(term, negative);
            sum = add(sum, divide(term, Pair(index + 1.0)));
        }
        return sum;
    }

    static Pair standardized_difference(double x, double xi, double alpha) {
        Pair difference = subtract(Pair(x), Pair(xi));
        if (std::isinf(difference.high)) {
            difference = subtract(Pair(x * 0.5), Pair(xi * 0.5));
            return multiply(divide(difference, Pair(alpha)), Pair(2.0));
        }
        return divide(difference, Pair(alpha));
    }

   public:
    static bool try_lower_endpoint(double xi, double alpha, double kappa, double hondo,
                                   double& endpoint) {
        endpoint = std::numeric_limits<double>::quiet_NaN();
        if (!(hondo > 0.0) || !std::isfinite(hondo)) return false;
        const Pair log_h = log_pair(Pair(hondo));
        const Pair exponent = multiply(Pair(-kappa), log_h);
        if (!std::isfinite(exponent.high) || std::fabs(exponent.high) > 700.0) return false;
        Pair standard(0.0);
        if (std::fabs(exponent.high) < 0.5) {
            Pair relative(1.0);
            Pair term(1.0);
            for (int n = 1; n <= 32; ++n) {
                term = divide(multiply(term, exponent), Pair(n + 1.0));
                relative = add(relative, term);
            }
            standard = multiply(log_h, relative);
        } else {
            const int power = static_cast<int>(std::round(exponent.high / log_two().high));
            const Pair reduced = subtract(exponent, multiply(Pair(power), log_two()));
            Pair sum(1.0);
            Pair term(1.0);
            for (int n = 1; n <= 32; ++n) {
                term = divide(multiply(term, reduced), Pair(n));
                sum = add(sum, term);
            }
            standard = divide(subtract(Pair(1.0), scale(sum, power)), Pair(kappa));
        }
        endpoint = add(Pair(xi), multiply(Pair(alpha), standard)).high;
        return std::isfinite(endpoint);
    }

    static double log_probability(double x, double xi, double alpha, double kappa,
                                  double hondo) {
        const Pair y = standardized_difference(x, xi, alpha);
        if (y.high == 0.0 && hondo == 1.0 && x > xi) {
            const Pair difference = subtract(Pair(x), Pair(xi));
            return subtract(log_pair(difference), log_pair(Pair(alpha))).high;
        }
        const Pair log_h = log_pair(Pair(hondo));
        Pair z(0.0);
        if (kappa == 0.0) {
            z = subtract(log_h, y);
        } else {
            const Pair product = multiply(Pair(-kappa), y);
            if (std::fabs(product.high) < 0.125) {
                z = subtract(log_h, multiply(y, log_one_plus_relative(product)));
            } else {
                Pair log_base(0.0);
                if (std::isinf(product.high) && product.high > 0.0) {
                    log_base = add(log_pair(Pair(std::fabs(kappa))), log_pair(abs_pair(y)));
                } else {
                    const Pair basis = add(Pair(1.0), product);
                    if (basis.high <= 0.0) return 0.0;
                    log_base = log_pair(basis);
                }
                z = divide(add(log_base, multiply(Pair(kappa), log_h)), Pair(kappa));
            }
        }
        if (z.high >= 0.0) return -std::numeric_limits<double>::infinity();
        return std::log(-std::expm1(z.high)) / hondo;
    }

    static double log_t(double x, double xi, double alpha, double kappa) {
        const Pair y = standardized_difference(x, xi, alpha);
        if (kappa == 0.0) return -y.high;
        const Pair basis = subtract(Pair(1.0), multiply(Pair(kappa), y));
        if (basis.high <= 0.0) return std::log(basis.high) / kappa;
        return divide(log_pair(basis), Pair(kappa)).high;
    }
};

}  // namespace corehydro::numerics::distributions::distribution_numerics
