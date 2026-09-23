// ported from: Numerics/Distributions/Univariate/Base/KappaExpectedInformation.cs @ 7e8e8d1
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/distribution_numerics.hpp"

namespace corehydro::numerics::distributions::distribution_numerics {

class KappaExpectedInformation {
   public:
    using Matrix = std::vector<std::vector<double>>;

    static Matrix parameter_covariance(double alpha, double kappa, double hondo,
                                       int sample_size, int parameter_count) {
        validate_sample_size(sample_size);
        if (!finite(alpha) || alpha <= 0.0) throw std::out_of_range("alpha must be positive");
        std::vector<double> means;
        std::vector<double> mean_errors;
        Matrix information_errors;
        Matrix covariance = invert_information(expected_information(
            kappa, hondo, parameter_count, means, mean_errors, information_errors));
        for (int i = 0; i < parameter_count; ++i) {
            for (int j = i; j < parameter_count; ++j) {
                double value = covariance[static_cast<std::size_t>(i)]
                                         [static_cast<std::size_t>(j)];
                if (value != 0.0) {
                    const double logarithm =
                        std::log(std::fabs(value)) - std::log(sample_size) +
                        (i < 2 ? std::log(alpha) : 0.0) +
                        (j < 2 ? std::log(alpha) : 0.0);
                    value = std::copysign(std::exp(logarithm), value);
                    if (!finite(value))
                        throw std::runtime_error(
                            "local MLE covariance is outside the finite range");
                }
                covariance[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = value;
                covariance[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = value;
            }
        }
        return covariance;
    }

    static Matrix expected_information(double kappa, double hondo, int parameter_count,
                                       std::vector<double>& score_means,
                                       std::vector<double>& score_mean_errors,
                                       Matrix& information_errors) {
        validate_domain(kappa, hondo, parameter_count);
        const double boundary = std::max(kappa, std::max(hondo, kappa * hondo));
        const int power = boundary > 0.4 ? 128 : 8;
        const Interval integrals = integrate(kappa, hondo, parameter_count, power);
        score_means.assign(static_cast<std::size_t>(parameter_count), 0.0);
        score_mean_errors.assign(static_cast<std::size_t>(parameter_count), 0.0);
        Matrix information(static_cast<std::size_t>(parameter_count),
                           std::vector<double>(static_cast<std::size_t>(parameter_count)));
        information_errors = information;
        for (int i = 0; i < parameter_count; ++i) {
            score_means[static_cast<std::size_t>(i)] = integrals.values[static_cast<std::size_t>(i)];
            score_mean_errors[static_cast<std::size_t>(i)] =
                integrals.errors[static_cast<std::size_t>(i)];
            if (std::fabs(score_means[static_cast<std::size_t>(i)]) >
                4.0 * score_mean_errors[static_cast<std::size_t>(i)] + 32.0 * kRoundoff)
                throw std::runtime_error(
                    "integrated fixed-observation score does not have zero mean");
        }
        std::size_t index = static_cast<std::size_t>(parameter_count);
        for (int i = 0; i < parameter_count; ++i)
            for (int j = i; j < parameter_count; ++j, ++index) {
                information[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    information[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] =
                        integrals.values[index];
                information_errors[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    information_errors[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] =
                        integrals.errors[index];
            }
        return information;
    }

   private:
    static constexpr double kRelativeTolerance = 1e-10;
    static constexpr double kAbsoluteTolerance = 1e-12;
    static constexpr double kRoundoff = 2.2204460492503131e-16;
    static constexpr int kMaximumIntervals = 2048;
    static constexpr std::array<double, 11> kNodes = {
        .995657163025808080735527280689003, .973906528517171720077964012084452,
        .930157491355708226001207180059508, .865063366688984510732096688423493,
        .780817726586416897063717578345042, .679409568299024406234327365114874,
        .562757134668604683339000099272694, .433395394129247190799265943165784,
        .294392862701460198131126603103866, .148874338981631210884826001129720, 0.0};
    static constexpr std::array<double, 11> kKronrodWeights = {
        .011694638867371874278064396062192, .032558162307964727478818972459390,
        .054755896574351996031381300244580, .075039674810919952767043140916190,
        .093125454583697605535065465083366, .109387158802297641899210590325805,
        .123491976262065851077958109831074, .134709217311473325928054001771707,
        .142775938577060080797094273138717, .147739104901338491374841515972068,
        .149445554002916905664936468389821};
    static constexpr std::array<double, 5> kGaussWeights = {
        .066671344308688137593568809893332, .149451349150580593145776339657697,
        .219086362515982043995534934228163, .269266719309996355091226921569469,
        .295524224714752870173892994651338};

    struct Interval {
        double lower = 0.0;
        double upper = 0.0;
        std::vector<double> values;
        std::vector<double> errors;
    };

    static void validate_domain(double kappa, double hondo, int count) {
        if (count != 3 && count != 4) throw std::out_of_range("parameter count must be 3 or 4");
        if (!finite(kappa) || !finite(hondo)) throw std::out_of_range("shapes must be finite");
        if (kappa >= 0.5 || hondo >= 0.5 || kappa * hondo >= 0.5)
            throw std::out_of_range(
                "regular information requires kappa, hondo, and their product below one half");
    }

    static double second_exponential_relative(double x) {
        if (std::fabs(x) >= 0.1) return (std::expm1(x) - x) / x / x;
        double sum = 0.5;
        double term = 0.5;
        for (int n = 1; n < 24; ++n) {
            term *= x / (n + 2.0);
            sum += term;
            if (std::fabs(term) < std::fabs(sum) * 1e-17) break;
        }
        return sum;
    }

    static double log_absolute_expm1(double x) {
        return x > 36.0 ? x + std::log1p(-std::exp(-x)) : std::log(std::fabs(std::expm1(x)));
    }

    static std::array<double, 4> weighted_scores(double log_p, double log_q, double kappa,
                                                 double hondo, double log_root) {
        const double u = -log_p;
        double log_t;
        double log_a;
        const double log_absolute_h = std::log(std::fabs(hondo));
        if (log_q < -36.0 && log_absolute_h + log_q < -36.0) {
            log_t = log_a = log_q;
        } else if (hondo == 0.0) {
            log_t = log_a = std::log(u);
        } else {
            const double hu = log_q < -36.0
                                  ? std::copysign(std::exp(log_absolute_h + log_q), hondo)
                                  : hondo * u;
            log_t = log_absolute_expm1(-hu) - log_absolute_h;
            log_a = log_absolute_expm1(hu) - log_absolute_h;
        }
        const double y = -log_t;
        const double root = std::exp(log_root);
        const double log_one_minus_k = std::log1p(-kappa);
        const double log_one_minus_h = std::log1p(-hondo);
        const double c_root = std::exp(log_one_minus_k + log_root) -
                              std::exp(log_one_minus_h + log_a + log_root);
        const double c_exponential_root =
            std::exp(log_one_minus_k + kappa * y + log_root) -
            std::exp(log_one_minus_h + log_a + kappa * y + log_root);
        double scale_score;
        double kappa_score;
        if (std::fabs(kappa * y) < 0.1 || kappa == 0.0) {
            scale_score = -root + c_root * y * exprel(kappa * y);
            kappa_score = y * root -
                          c_root * y * y * second_exponential_relative(kappa * y);
        } else {
            scale_score = -root + (c_exponential_root - c_root) / kappa;
            kappa_score = y * root +
                          ((1.0 + kappa * y) * c_root - c_exponential_root) /
                              (kappa * kappa);
        }
        const double hondo_score =
            std::fabs(hondo * u) < 0.1 || hondo == 0.0
                ? u * root - (1.0 - hondo) * u * u * root *
                               second_exponential_relative(hondo * u)
                : (u * root - std::exp(log_one_minus_h + log_a + log_root)) / hondo;
        return {c_exponential_root, scale_score, kappa_score, hondo_score};
    }

    static std::vector<double> integrand(double coordinate, double kappa, double hondo,
                                         int count, int power) {
        const double log_coordinate = std::log(coordinate);
        const double log_tail = power * log_coordinate - std::log(2.0);
        const double log_other = log1m_exp(log_tail);
        const double log_root =
            0.5 * (std::log(power / 2.0) + (power - 1.0) * log_coordinate);
        const auto lower = weighted_scores(log_tail, log_other, kappa, hondo, log_root);
        const auto upper = weighted_scores(log_other, log_tail, kappa, hondo, log_root);
        std::vector<double> values(static_cast<std::size_t>(count + count * (count + 1) / 2));
        const double root = std::exp(log_root);
        for (int i = 0; i < count; ++i)
            values[static_cast<std::size_t>(i)] = (lower[static_cast<std::size_t>(i)] +
                                                    upper[static_cast<std::size_t>(i)]) *
                                                   root;
        std::size_t index = static_cast<std::size_t>(count);
        for (int i = 0; i < count; ++i)
            for (int j = i; j < count; ++j, ++index)
                values[index] = lower[static_cast<std::size_t>(i)] *
                                    lower[static_cast<std::size_t>(j)] +
                                upper[static_cast<std::size_t>(i)] *
                                    upper[static_cast<std::size_t>(j)];
        for (double value : values)
            if (!finite(value)) throw std::runtime_error("nonfinite information integrand");
        return values;
    }

    static Interval evaluate(double a, double b, double kappa, double hondo, int count,
                             int power) {
        const double center = 0.5 * (a + b);
        const double half = 0.5 * (b - a);
        std::array<std::vector<double>, 21> node_values;
        node_values[20] = integrand(center, kappa, hondo, count, power);
        const std::size_t dimension = node_values[20].size();
        std::vector<double> kronrod(dimension);
        std::vector<double> gauss(dimension);
        std::vector<double> absolute(dimension);
        for (std::size_t j = 0; j < dimension; ++j) {
            kronrod[j] = kKronrodWeights[10] * node_values[20][j];
            absolute[j] = kKronrodWeights[10] * std::fabs(node_values[20][j]);
        }
        for (int i = 0; i < 10; ++i) {
            node_values[static_cast<std::size_t>(2 * i)] =
                integrand(center - half * kNodes[static_cast<std::size_t>(i)], kappa, hondo,
                          count, power);
            node_values[static_cast<std::size_t>(2 * i + 1)] =
                integrand(center + half * kNodes[static_cast<std::size_t>(i)], kappa, hondo,
                          count, power);
            for (std::size_t j = 0; j < dimension; ++j) {
                const double sum = node_values[static_cast<std::size_t>(2 * i)][j] +
                                   node_values[static_cast<std::size_t>(2 * i + 1)][j];
                kronrod[j] += kKronrodWeights[static_cast<std::size_t>(i)] * sum;
                absolute[j] += kKronrodWeights[static_cast<std::size_t>(i)] *
                               (std::fabs(node_values[static_cast<std::size_t>(2 * i)][j]) +
                                std::fabs(node_values[static_cast<std::size_t>(2 * i + 1)][j]));
                if (i % 2 == 1)
                    gauss[j] += kGaussWeights[static_cast<std::size_t>(i / 2)] * sum;
            }
        }
        std::vector<double> errors(dimension);
        for (std::size_t j = 0; j < dimension; ++j) {
            const double mean = kronrod[j] / 2.0;
            double deviation = kKronrodWeights[10] * std::fabs(node_values[20][j] - mean);
            for (int i = 0; i < 10; ++i)
                deviation += kKronrodWeights[static_cast<std::size_t>(i)] *
                             (std::fabs(node_values[static_cast<std::size_t>(2 * i)][j] - mean) +
                              std::fabs(node_values[static_cast<std::size_t>(2 * i + 1)][j] - mean));
            deviation *= half;
            double error = std::fabs(kronrod[j] - gauss[j]) * half;
            if (deviation != 0.0 && error != 0.0)
                error = deviation * std::min(1.0, std::pow(200.0 * error / deviation, 1.5));
            errors[j] = std::max(error, 50.0 * kRoundoff * half * absolute[j]);
            kronrod[j] *= half;
        }
        return {a, b, std::move(kronrod), std::move(errors)};
    }

    static Interval integrate(double kappa, double hondo, int count, int power) {
        Interval initial = evaluate(0.0, 1.0, kappa, hondo, count, power);
        std::vector<Interval> intervals{initial};
        Interval total{0.0, 0.0, initial.values, initial.errors};
        while (true) {
            bool success = true;
            for (std::size_t j = 0; j < total.values.size(); ++j)
                success = success && total.errors[j] <=
                                           kAbsoluteTolerance +
                                               kRelativeTolerance * std::fabs(total.values[j]);
            if (success) return total;
            if (intervals.size() >= kMaximumIntervals)
                throw std::runtime_error("expected-information quadrature did not converge");
            double worst = -1.0;
            std::size_t selected = 0;
            for (std::size_t i = 0; i < intervals.size(); ++i)
                for (std::size_t j = 0; j < total.values.size(); ++j) {
                    const double ratio = intervals[i].errors[j] /
                                         (kAbsoluteTolerance +
                                          kRelativeTolerance * std::fabs(total.values[j]));
                    if (ratio > worst) {
                        worst = ratio;
                        selected = i;
                    }
                }
            const Interval old = intervals[selected];
            const double center = 0.5 * (old.lower + old.upper);
            if (center == old.lower || center == old.upper)
                throw std::runtime_error("expected-information subdivision exhausted");
            intervals[selected] = evaluate(old.lower, center, kappa, hondo, count, power);
            intervals.push_back(evaluate(center, old.upper, kappa, hondo, count, power));
            std::fill(total.values.begin(), total.values.end(), 0.0);
            std::fill(total.errors.begin(), total.errors.end(), 0.0);
            for (const auto& interval : intervals)
                for (std::size_t j = 0; j < total.values.size(); ++j) {
                    total.values[j] += interval.values[j];
                    total.errors[j] += interval.errors[j];
                }
        }
    }

    static Matrix invert_information(const Matrix& information) {
        const std::size_t count = information.size();
        std::vector<double> scale(count);
        Matrix normalized(count, std::vector<double>(count));
        Matrix lower(count, std::vector<double>(count));
        for (std::size_t i = 0; i < count; ++i) {
            if (!(information[i][i] > 0.0) || !finite(information[i][i]))
                throw std::runtime_error("expected information has invalid diagonal");
            scale[i] = std::sqrt(information[i][i]);
        }
        for (std::size_t i = 0; i < count; ++i)
            for (std::size_t j = 0; j < count; ++j)
                normalized[i][j] = information[i][j] / scale[i] / scale[j];
        for (std::size_t i = 0; i < count; ++i)
            for (std::size_t j = 0; j <= i; ++j) {
                double value = normalized[i][j];
                for (std::size_t m = 0; m < j; ++m) value -= lower[i][m] * lower[j][m];
                if (i == j) {
                    if (!(value > 0.0) || !finite(value))
                        throw std::runtime_error("expected information is not positive definite");
                    lower[i][j] = std::sqrt(value);
                } else {
                    lower[i][j] = value / lower[j][j];
                }
            }
        Matrix inverse(count, std::vector<double>(count));
        for (std::size_t column = 0; column < count; ++column) {
            std::vector<double> solution(count);
            for (std::size_t i = 0; i < count; ++i) {
                double value = i == column ? 1.0 : 0.0;
                for (std::size_t j = 0; j < i; ++j) value -= lower[i][j] * solution[j];
                solution[i] = value / lower[i][i];
            }
            for (std::size_t ii = count; ii-- > 0;) {
                double value = solution[ii];
                for (std::size_t j = ii + 1; j < count; ++j)
                    value -= lower[j][ii] * solution[j];
                solution[ii] = value / lower[ii][ii];
                inverse[ii][column] = solution[ii];
            }
            for (std::size_t i = 0; i < count; ++i) {
                double value = 0.0;
                for (std::size_t j = 0; j < count; ++j)
                    value += normalized[i][j] * solution[j];
                if (!finite(value) ||
                    std::fabs(value - (i == column ? 1.0 : 0.0)) > 1e-9)
                    throw std::runtime_error("expected-information inverse residual failed");
            }
        }
        for (std::size_t i = 0; i < count; ++i)
            for (std::size_t j = i; j < count; ++j) {
                const double value =
                    0.5 * (inverse[i][j] + inverse[j][i]) / scale[i] / scale[j];
                if (!finite(value)) throw std::runtime_error("nonfinite information inverse");
                inverse[i][j] = inverse[j][i] = value;
            }
        return inverse;
    }
};

}  // namespace corehydro::numerics::distributions::distribution_numerics
