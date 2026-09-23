// ported from: Numerics/Distributions/Univariate/GeneralizedExtremeValue.cs @ 7e8e8d1
//
// Generalized Extreme Value distribution: parameters ξ (location), α (scale),
// κ (shape). Distribution-core surface (moments, PDF/CDF/InverseCDF, log-likelihood).
// L-moment / MLE estimation lands in a later increment (needs Brent, Statistics,
// NelderMead). Logic mirrors the C# source method-for-method.
#pragma once
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_standard_error.hpp"
#include "corehydro/numerics/distributions/base/kappa_expected_information.hpp"
#include "corehydro/numerics/distributions/base/gamma_distribution_numerics.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

// GEV-specific estimation-method enum (predates the shared ParameterEstimationMethod;
// kept so the GEV bindings/fixtures continue to resolve "mom"/"lmom"/"mle" unchanged).
enum class EstimationMethod { MethodOfMoments, MethodOfLinearMoments, MaximumLikelihood };

// Mirrors the C# class declaration: `GeneralizedExtremeValue : UnivariateDistributionBase,
// IEstimation, IMaximumLikelihoodEstimation, ILinearMomentEstimation, ...` -- the IEstimation
// and ILinearMomentEstimation mixins postdate the Phase-0 GEV port and were retrofitted so the
// factory-dispatched public API (dist_fit / dist_lmoments) treats GEV like every other family.
class GeneralizedExtremeValue : public UnivariateDistributionBase,
                                public IEstimation,
                                public ILinearMomentEstimation,
                                public IMaximumLikelihoodEstimation,
                                public IStandardError {
   public:
    GeneralizedExtremeValue() { set_parameters(100.0, 10.0, 0.0); }
    GeneralizedExtremeValue(double location, double scale, double shape) {
        set_parameters(location, scale, shape);
    }

    double xi() const { return xi_; }
    double alpha() const { return alpha_; }
    double kappa() const { return kappa_; }

    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::GeneralizedExtremeValue;
    }
    int number_of_parameters() const override { return 3; }
    std::vector<double> get_parameters() const override { return {xi_, alpha_, kappa_}; }

    void set_parameters(double location, double scale, double shape) {
        xi_ = location;
        alpha_ = scale;
        kappa_ = shape;
        parameters_valid_ = validate(location, scale, shape);
    }

    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1], p[2]); }

    // --- Parameter display names (X1; C# GeneralizedExtremeValue.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Location (\xCE\xBE)", "Scale (\xCE\xB1)", "Shape (\xCE\xBA)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBE", "\xCE\xB1", "\xCE\xBA"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<GeneralizedExtremeValue>(xi_, alpha_, kappa_);
    }

    // --- Estimation ---
    // IEstimation override (the shared enum); forwards to the legacy GEV-specific enum path.
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        switch (method) {
            case ParameterEstimationMethod::MethodOfMoments:
                estimate(sample, EstimationMethod::MethodOfMoments);
                break;
            case ParameterEstimationMethod::MethodOfLinearMoments:
                estimate(sample, EstimationMethod::MethodOfLinearMoments);
                break;
            default:
                estimate(sample, EstimationMethod::MaximumLikelihood);
                break;
        }
    }

    void estimate(const std::vector<double>& sample, EstimationMethod method) {
        distribution_numerics::validate_sample(sample, 4);
        if (method == EstimationMethod::MethodOfMoments) {
            set_parameters(direct_method_of_moments(data::product_moments(sample)));
        } else if (method == EstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else {
            set_parameters(mle(sample));
        }
    }

    // Solve for the shape parameter κ given the skewness coefficient.
    double solve_for_kappa(double skew) const {
        namespace g = math::special;
        if (skew > 1.14 && skew < 10.0) {
            return 0.2858221 - 0.357983 * skew + 0.116659 * std::pow(skew, 2) -
                   0.022725 * std::pow(skew, 3) + 0.002604 * std::pow(skew, 4) -
                   0.000161 * std::pow(skew, 5) + 0.000004 * std::pow(skew, 6);
        } else if (skew == 1.14) {
            return 0.0;
        } else if (skew >= 0.0 && skew < 1.14) {
            return 0.277648 - 0.322016 * skew + 0.060278 * std::pow(skew, 2) +
                   0.016759 * std::pow(skew, 3) - 0.005873 * std::pow(skew, 4) -
                   0.00244 * std::pow(skew, 5) - 0.00005 * std::pow(skew, 6);
        } else if (skew < 0.0 && skew >= -2.0) {
            return math::rootfinding::solve(
                [skew](double x) {
                    double U1 = g::lanczos(1.0 + x);
                    double U2 = g::lanczos(1.0 + 2.0 * x);
                    double U3 = g::lanczos(1.0 + 3.0 * x);
                    double k = sign(x) * (-U3 + 3.0 * U1 * U2 - 2.0 * std::pow(U1, 3)) /
                               std::pow(U2 - U1 * U1, 1.5);
                    return k - skew;
                },
                -(1.0 / 3.0), 1.0);
        } else if (skew < -2.0) {
            return -0.50405 - 0.00861 * skew + 0.015497 * std::pow(skew, 2) +
                   0.005613 * std::pow(skew, 3) + 0.00087 * std::pow(skew, 4) +
                   0.000065 * std::pow(skew, 5);
        }
        return kNaN;
    }

    std::vector<double> direct_method_of_moments(const std::vector<double>& moments) const {
        namespace g = math::special;
        double k = solve_for_kappa(moments[2]);
        double a, x;
        if (std::fabs(k) <= kNearZero) {
            a = std::sqrt(6.0) / kPi * moments[1];
            x = moments[0] - a * kEuler;
        } else {
            double U1 = g::function(1.0 + k);
            double U2 = g::function(1.0 + 2.0 * k);
            a = std::sqrt(moments[1] * moments[1] * k * k / (U2 - U1 * U1));
            x = moments[0] - a / k * (1.0 - U1);
        }
        return {x, a, k};
    }

    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        namespace g = math::special;
        double L1 = moments[0], L2 = moments[1], T3 = moments[2];
        double kappa;
        if (std::fabs(T3) <= 0.5) {
            double c = 2.0 / (3.0 + T3) - std::log(2.0) / std::log(3.0);
            kappa = 7.859 * c + 2.9554 * c * c;
        } else {
            kappa = math::rootfinding::solve(
                [T3](double x) {
                    return T3 - (2.0 * (1.0 - std::pow(3.0, -x)) / (1.0 - std::pow(2.0, -x)) - 3.0);
                },
                -1.0, 10.0);
        }
        double alpha = L2 * kappa / ((1.0 - std::pow(2.0, -kappa)) * g::function(1.0 + kappa));
        double xi = L1 - alpha * (1.0 - g::function(1.0 + kappa)) / kappa;
        return {xi, alpha, kappa};
    }

    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        namespace g = math::special;
        double xi = parameters[0], alpha = parameters[1], kappa = parameters[2];
        if (kappa <= -1.0) throw std::out_of_range("L-moments require kappa > -1");
        double L1 = xi + alpha * (1.0 - g::function(1.0 + kappa)) / kappa;
        double L2 = alpha * (1.0 - std::pow(2.0, -kappa)) * g::function(1.0 + kappa) / kappa;
        double T3 = 2.0 * (1.0 - std::pow(3.0, -kappa)) / (1.0 - std::pow(2.0, -kappa)) - 3.0;
        double T4 = (5.0 * (1.0 - std::pow(4.0, -kappa)) - 10.0 * (1.0 - std::pow(3.0, -kappa)) +
                     6.0 * (1.0 - std::pow(2.0, -kappa))) /
                    (1.0 - std::pow(2.0, -kappa));
        return {L1, L2, T3, T4};
    }

    // Initial values + bounds for MLE (location/scale/shape).
    void get_parameter_constraints(const std::vector<double>& sample, std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        initials = parameters_from_linear_moments(data::linear_moments(sample));
        lowers.assign(3, 0.0);
        uppers.assign(3, 0.0);
        if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
        double locExp = std::ceil(std::log10(std::fabs(initials[0])) + 1.0);
        lowers[0] = -std::pow(10.0, locExp);
        uppers[0] = std::pow(10.0, locExp);
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(std::fabs(initials[1])) + 1.0));
        lowers[2] = -10.0;
        uppers[2] = 10.0;
        if (initials[2] <= lowers[2] || initials[2] >= uppers[2]) initials[2] = 0.0;
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            GeneralizedExtremeValue g;
            g.set_parameters(x[0], x[1], x[2]);
            return g.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 3, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

    // --- Standard error / quantile uncertainty (MLE) ---

    // Expected Fisher information matrix (3x3) for the given sample size.
    math::linalg::Matrix2D expected_information_matrix(int sample_size) const {
        distribution_numerics::validate_sample_size(sample_size);
        if (!parameters_valid_) throw std::out_of_range("GEV: invalid parameters");
        std::vector<double> means, mean_errors;
        math::linalg::Matrix2D errors;
        auto information =
            distribution_numerics::KappaExpectedInformation::expected_information(
                kappa_, 0.0, 3, means, mean_errors, errors);
        for (int i = 0; i < 3; ++i)
            for (int j = i; j < 3; ++j) {
                double value = information[static_cast<std::size_t>(i)]
                                          [static_cast<std::size_t>(j)];
                if (value != 0.0) {
                    const double logarithm =
                        std::log(std::fabs(value)) + std::log(sample_size) -
                        (i < 2 ? std::log(alpha_) : 0.0) -
                        (j < 2 ? std::log(alpha_) : 0.0);
                    value = std::copysign(std::exp(logarithm), value);
                    if (!std::isfinite(value))
                        throw std::runtime_error(
                            "GEV expected information is outside the finite range");
                }
                information[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = value;
                information[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = value;
            }
        return information;
    }

    // Parameter covariance = inverse of the expected information matrix (MLE only).
    math::linalg::Matrix2D parameter_covariance(int sample_size) const {
        return distribution_numerics::KappaExpectedInformation::parameter_covariance(
            alpha_, kappa_, 0.0, sample_size, 3);
    }

    math::linalg::Matrix2D parameter_covariance(
        int sample_size, ParameterEstimationMethod method) const override {
        if (method != ParameterEstimationMethod::MaximumLikelihood)
            throw std::logic_error("GEV covariance is implemented only for maximum likelihood");
        if (!parameters_valid_) throw std::out_of_range("GEV: invalid parameters");
        return parameter_covariance(sample_size);
    }

    // Gradient of the quantile (InverseCDF) wrt {location, scale, shape}.
    std::vector<double> quantile_gradient(double probability) const override {
        distribution_numerics::validate_probability(probability);
        if (!parameters_valid_) throw std::out_of_range("GEV: invalid parameters");
        const double logarithm = std::log(-std::log(probability));
        const double product = kappa_ * logarithm;
        return {1.0,
                product == -kInf
                    ? 1.0 / kappa_
                    : distribution_numerics::scaled_exprel_product(
                          1.0, -logarithm, product),
                product == -kInf
                    ? -(alpha_ / kappa_) / kappa_
                    : -distribution_numerics::scaled_exprel_derivative_product(
                          alpha_, logarithm, product)};
    }

    // Jacobian of the quantile transformation for a set of probabilities (one per
    // parameter), plus its determinant (C# IStandardError.QuantileJacobian, line 747;
    // ported additively in M12 -- PointProcessModel's multi-quantile prior branch needs the
    // determinant). C# ArgumentOutOfRangeException -> std::out_of_range.
    math::linalg::Matrix2D quantile_jacobian(const std::vector<double>& probabilities,
                                             double& determinant) const override {
        return distribution_numerics::quantile_jacobian(
            *this, number_of_parameters(), probabilities, determinant);
    }

    // Delta-method variance of the quantile (MLE).
    double quantile_variance(double probability, int sample_size) const {
        return quantile_variance(probability, sample_size,
                                 ParameterEstimationMethod::MaximumLikelihood);
    }

    double quantile_variance(double probability, int sample_size,
                             ParameterEstimationMethod method) const override {
        distribution_numerics::validate_probability(probability);
        if (!parameters_valid_) throw std::out_of_range("GEV: invalid parameters");
        GeneralizedExtremeValue unit(0.0, 1.0, kappa_);
        const double logarithm = std::log(-std::log(probability));
        const double product = kappa_ * logarithm;
        const double scale_gradient = distribution_numerics::scaled_exprel_product(
            alpha_, -logarithm, product);
        const double shape_gradient =
            -distribution_numerics::scaled_exprel_derivative_product(
                alpha_, logarithm, product);
        return distribution_numerics::scaled_quantile_variance(
            unit.parameter_covariance(sample_size, method),
            {alpha_, scale_gradient, shape_gradient});
    }

    // --- Moments ---
    double mean() const override {
        if (kappa_ <= -1.0) return kNaN;
        if (std::fabs(kappa_) <= 0.05) return xi_ + alpha_ * small_shape_standardized_moments()[0];
        const double logarithm = math::special::log_gamma(1.0 + kappa_);
        if (logarithm == 0.0) return xi_;
        const double magnitude =
            logarithm > 0.0
                ? logarithm + distribution_numerics::log1m_exp(-logarithm)
                : distribution_numerics::log1m_exp(logarithm);
        return xi_ - std::copysign(1.0, kappa_) * std::copysign(1.0, logarithm) *
                         std::exp(std::log(alpha_) + magnitude - std::log(std::fabs(kappa_)));
    }

    double median() const override {
        return inverse_cdf(0.5);
    }

    double mode() const override {
        if (kappa_ >= 1.0) return maximum();
        const double logarithm = std::log1p(-kappa_);
        return xi_ - alpha_ * logarithm * distribution_numerics::exprel(kappa_ * logarithm);
    }

    double standard_deviation() const override {
        if (kappa_ <= -0.5) return kNaN;
        if (std::fabs(kappa_) <= 0.05)
            return alpha_ * small_shape_standardized_moments()[1];
        if (kappa_ == 1.0) return alpha_;
        return std::exp(std::log(alpha_) + 0.5 * log_power_variance(kappa_) -
                        std::log(std::fabs(kappa_)));
    }

    double skewness() const override {
        if (kappa_ <= -1.0 / 3.0) return kNaN;
        if (std::fabs(kappa_) <= 0.05) return small_shape_standardized_moments()[2];
        if (kappa_ == 1.0) return -2.0;
        const double l1 = math::special::log_gamma(1.0 + kappa_);
        const double l2 = math::special::log_gamma(1.0 + 2.0 * kappa_);
        const double l3 = math::special::log_gamma(1.0 + 3.0 * kappa_);
        if (std::isinf(l3) && l3 > 0.0) return -kInf;
        const double largest = std::max(l3, std::max(l1 + l2, 3.0 * l1));
        const double centered = std::exp(l3 - largest) -
                                3.0 * std::exp(l1 + l2 - largest) +
                                2.0 * std::exp(3.0 * l1 - largest);
        return centered == 0.0
                   ? 0.0
                   : -std::copysign(1.0, kappa_) * std::copysign(1.0, centered) *
                         std::exp(largest + std::log(std::fabs(centered)) -
                                  1.5 * log_power_variance(kappa_));
    }

    double kurtosis() const override {
        if (kappa_ <= -0.25) return kNaN;
        if (std::fabs(kappa_) <= 0.05) return small_shape_standardized_moments()[3];
        if (kappa_ == 1.0) return 9.0;
        const double l1 = math::special::log_gamma(1.0 + kappa_);
        const double l2 = math::special::log_gamma(1.0 + 2.0 * kappa_);
        const double l3 = math::special::log_gamma(1.0 + 3.0 * kappa_);
        const double l4 = math::special::log_gamma(1.0 + 4.0 * kappa_);
        if (std::isinf(l4) && l4 > 0.0) return kInf;
        const double largest = std::max(std::max(l4, l1 + l3),
                                        std::max(2.0 * l1 + l2, 4.0 * l1));
        const double centered =
            std::exp(l4 - largest) - 4.0 * std::exp(l1 + l3 - largest) +
            6.0 * std::exp(2.0 * l1 + l2 - largest) -
            3.0 * std::exp(4.0 * l1 - largest);
        return std::exp(largest + std::log(centered) -
                        2.0 * log_power_variance(kappa_));
    }

    double minimum() const override {
        if (kappa_ >= 0.0) return -kInf;
        return xi_ + alpha_ / kappa_;
    }
    double maximum() const override {
        if (kappa_ <= 0.0) return kInf;
        return xi_ + alpha_ / kappa_;
    }

    // --- Distribution functions ---
    double pdf(double x) const override { return std::exp(log_pdf(x)); }

    double log_pdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("GEV: invalid parameters");
        if (x < minimum() || x > maximum() || std::isinf(x)) return -kInf;
        if (kappa_ > 0.0 && x == maximum())
            return kappa_ < 1.0 ? -kInf : kappa_ == 1.0 ? -std::log(alpha_) : kInf;
        if (kappa_ < 0.0 && x == minimum()) return -kInf;
        const double y = distribution_numerics::hosking_shape_transform(x, xi_, alpha_, kappa_);
        const double value = -(1.0 - kappa_) * y - std::exp(-y) - std::log(alpha_);
        return std::isnan(value) ? -kInf : value;
    }

    double cdf(double x) const override { return std::exp(log_cdf(x)); }

    double log_cdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("GEV: invalid parameters");
        if (x <= minimum()) return -kInf;
        if (x >= maximum()) return 0.0;
        return -std::exp(-distribution_numerics::hosking_shape_transform(
            x, xi_, alpha_, kappa_));
    }

    double ccdf(double x) const override { return -std::expm1(log_cdf(x)); }

    double log_ccdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("GEV: invalid parameters");
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return -kInf;
        const double y =
            distribution_numerics::hosking_shape_transform(x, xi_, alpha_, kappa_);
        const double exponential = std::exp(-y);
        return exponential == 0.0 ? -y : distribution_numerics::log1m_exp(-exponential);
    }

    double inverse_cdf(double probability) const override {
        if (!(probability >= 0.0 && probability <= 1.0))
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        const double logarithm = std::log(-std::log(probability));
        const double product = kappa_ * logarithm;
        const double unit_quantile =
            product == -kInf ? 1.0 / kappa_
                             : distribution_numerics::scaled_exprel_product(
                                   1.0, -logarithm, product);
        const double displacement =
            product == -kInf ? alpha_ / kappa_
                             : distribution_numerics::scaled_exprel_product(
                                   alpha_, -logarithm, product);
        return std::isinf(displacement) && std::isfinite(unit_quantile)
                   ? alpha_ * (xi_ / alpha_ + unit_quantile)
                   : xi_ + displacement;
    }

    double log_likelihood(const std::vector<double>& sample) const {
        double ll = 0.0;
        for (double v : sample) ll += log_pdf(v);
        if (std::isnan(ll) || std::isinf(ll)) return -kInf;
        return ll;
    }

   private:
    static double log_power_variance(double power) {
        const double first = math::special::log_gamma(1.0 + power);
        const double second = math::special::log_gamma(1.0 + 2.0 * power);
        return std::isinf(second) && second > 0.0
                   ? kInf
                   : second + distribution_numerics::log1m_exp(2.0 * first - second);
    }

    static double normalized_log_gamma_difference(double kappa, int order) {
        double sum = 0.0;
        double power = 1.0;
        for (int n = order; n <= 32; ++n) {
            const double factor =
                order == 2 ? std::pow(2.0, n) - 2.0
                           : order == 3
                                 ? std::pow(3.0, n) - 3.0 * std::pow(2.0, n) + 3.0
                                 : std::pow(4.0, n) - 4.0 * std::pow(3.0, n) +
                                       6.0 * std::pow(2.0, n) - 4.0;
            sum += (n % 2 == 0 ? 1.0 : -1.0) *
                   distribution_numerics::zeta_integer(n) * factor * power / n;
            power *= kappa;
        }
        return sum;
    }

    std::vector<double> small_shape_standardized_moments() const {
        const double k = kappa_;
        if (k == 0.0)
            return {kEuler, kPi / std::sqrt(6.0), 1.1395470994046487, 5.4};
        const double logarithm = distribution_numerics::log_gamma_one_plus(k);
        const double mean_value =
            -(logarithm / k) * distribution_numerics::exprel(logarithm);
        const double a = normalized_log_gamma_difference(k, 2);
        const double b = normalized_log_gamma_difference(k, 3);
        const double c = normalized_log_gamma_difference(k, 4);
        const double a2 = k * k * a;
        const double b3 = k * k * k * b;
        const double c4 = k * k * k * k * c;
        const double u = std::expm1(a2);
        const double v = std::expm1(b3);
        const double variance = a * distribution_numerics::exprel(a2);
        const double third = b * distribution_numerics::exprel(b3);
        const double fourth = c * distribution_numerics::exprel(c4);
        const double skew_value =
            -(k * variance * variance * (3.0 + u) +
              std::exp(3.0 * a2) * third) /
            (variance * std::sqrt(variance));
        const double kurtosis_value =
            (variance * variance *
                 (3.0 + u * (16.0 + u * (15.0 + u * (6.0 + u)))) +
             12.0 * k * third * std::exp(3.0 * a2) * a *
                 distribution_numerics::exprel(3.0 * a2) +
             std::exp(6.0 * a2) *
                 (k * k * third * third * (6.0 + v * (4.0 + v)) +
                  std::exp(4.0 * b3) * fourth)) /
            (variance * variance);
        return {mean_value, std::exp(logarithm) * std::sqrt(variance), skew_value,
                kurtosis_value};
    }

    // kNearZero / kNaN / kInf are inherited (protected) from UnivariateDistributionBase.
    static double sign(double x) { return (x > 0) - (x < 0); }

    static bool validate(double location, double scale, double shape) {
        if (std::isnan(location) || std::isinf(location)) return false;
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        if (std::isnan(shape) || std::isinf(shape)) return false;
        return true;
    }

    double xi_ = 0.0;
    double alpha_ = 0.0;
    double kappa_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
