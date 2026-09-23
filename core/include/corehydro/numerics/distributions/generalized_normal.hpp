// ported from: Numerics/Distributions/Univariate/GeneralizedNormal.cs @ 7e8e8d1
//
// The Generalized Normal distribution (LogNormal-3), parameterized by location ξ (Xi),
// scale α (Alpha), and shape κ (Kappa). κ→0 is the plain Normal; κ < 0 is bounded below at
// ξ + α/κ; κ > 0 is bounded above at the same point.
//
// Structurally the closest kin to generalized_extreme_value.hpp and generalized_logistic.hpp
// (same three parameters, same interface set, same L-moment/MLE shape), and this file mirrors
// the C# member order.
//
// C# members with no counterpart here, severed layer-wide across every ported distribution
// (WPF/serialization surface, never referenced by the numeric core): DisplayName /
// ShortDisplayName, ParametersToString (its first column IS `parameter_names()` below),
// GetParameterPropertyNames, MinimumOfParameters / MaximumOfParameters.
//
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_bootstrappable.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_standard_error.hpp"
#include "corehydro/numerics/distributions/base/kappa_four_boundary.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class GeneralizedNormal : public UnivariateDistributionBase,
                          public IEstimation,
                          public IMaximumLikelihoodEstimation,
                          public ILinearMomentEstimation,
                          public IStandardError,
                          public IBootstrappable {
   public:
    // Location 100, scale 10, shape 0 (C# GeneralizedNormal(), line 26).
    GeneralizedNormal() { set_parameters(100.0, 10.0, 0.0); }

    // C# GeneralizedNormal(double location, double scale, double shape), line 37.
    GeneralizedNormal(double location, double scale, double shape) {
        set_parameters(location, scale, shape);
    }

    // --- Parameter accessors (C# Xi / Alpha / Kappa properties, lines 51/65/79) ---
    double xi() const { return xi_; }
    double alpha() const { return alpha_; }
    double kappa() const { return kappa_; }

    // --- Identity / parameters ---
    int number_of_parameters() const override { return 3; }

    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::GeneralizedNormal;
    }

    // C# ParametersToString column 0 (line 115) and ParameterNamesShortForm (line 131).
    std::vector<std::string> parameter_names() const override {
        return {"Location (\xCE\xBE)", "Scale (\xCE\xB1)", "Shape (\xCE\xBA)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBE", "\xCE\xB1", "\xCE\xBA"};
    }

    // C# GetParameters (line 143).
    std::vector<double> get_parameters() const override { return {xi_, alpha_, kappa_}; }

    // --- Moments / support ---
    // C# Mean / StandardDeviation / Skewness / Kurtosis read the analytical shifted-lognormal
    // moment cache.
    double mean() const override {
        ensure_moments();
        return u_[0];
    }

    // C# Median (line 163).
    double median() const override { return inverse_cdf(0.5); }

    double mode() const override {
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        return quantile_at_latent(kappa_);
    }

    double standard_deviation() const override {
        ensure_moments();
        return u_[1];
    }

    double skewness() const override {
        ensure_moments();
        return u_[2];
    }

    double kurtosis() const override {
        ensure_moments();
        return u_[3];
    }

    // C# Minimum (line 222): unbounded below unless κ < -NearZero.
    double minimum() const override {
        if (kappa_ >= 0.0) return -kInf;
        return finite_shape_endpoint();
    }

    // C# Maximum (line 238): unbounded above unless κ > NearZero.
    double maximum() const override {
        if (kappa_ <= 0.0) return kInf;
        return finite_shape_endpoint();
    }

    // --- Estimation ---
    // C# Estimate (line 266): L-moments or MLE only; anything else throws
    // NotImplementedException.
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        distribution_numerics::validate_sample(sample, 4);
        if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::logic_error(
                "GeneralizedNormal: only the method of linear moments and maximum likelihood "
                "estimation methods are implemented");
        }
    }

    // C# Bootstrap (line 283, IBootstrappable): draw a fresh sample from the current
    // parameters, re-fit by `method`, and return the fitted distribution.
    std::unique_ptr<UnivariateDistributionBase> bootstrap(ParameterEstimationMethod method,
                                                          int sample_size,
                                                          int seed = -1) const override {
        auto new_distribution = std::make_unique<GeneralizedNormal>(xi_, alpha_, kappa_);
        auto sample = new_distribution->generate_random_values(sample_size, seed);
        new_distribution->estimate(sample, method);
        if (!new_distribution->parameters_valid())
            throw std::runtime_error("Bootstrapped distribution parameters are invalid.");
        return new_distribution;
    }

    // C# SetParameters(double, double, double) (line 299). The C# body validates, then assigns
    // through the three property setters, each of which re-validates; the last assignment
    // decides _parametersValid, so validating once after all three assignments is equivalent
    // (and is what every sibling three-parameter port does).
    void set_parameters(double location, double scale, double shape) {
        xi_ = location;
        alpha_ = scale;
        kappa_ = shape;
        parameters_valid_ = validate(location, scale, shape);
        moments_computed_ = false;  // C#: each property setter clears _momentsComputed
    }

    // C# SetParameters(IList<double>) (line 310).
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2]);
    }

    // C# ValidateParameters(double, double, double, bool) (line 322): ξ and κ must be numbers,
    // α must be positive. The C# `throwException` flag and the returned exception object
    // collapse to a bool here, matching every sibling port; the throwing call sites below
    // raise std::invalid_argument instead.
    static bool validate(double location, double scale, double shape) {
        if (std::isnan(location) || std::isinf(location)) return false;
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        if (std::isnan(shape) || std::isinf(shape)) return false;
        return true;
    }

    // C# ParametersFromLinearMoments (line 351): Hosking's rational approximation for κ, then
    // α and ξ in closed form. Coefficients transcribed exactly from the C# source.
    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        if (moments.size() < 3 || !std::isfinite(moments[0]) ||
            !std::isfinite(moments[1]) || moments[1] <= 0.0 ||
            !std::isfinite(moments[2]) || std::fabs(moments[2]) >= 1.0)
            throw std::out_of_range(
                "finite L-moments require positive L-scale and absolute L-skewness below one");
        double L1 = moments[0];
        double L2 = moments[1];
        double T3 = moments[2];

        double E0 = 2.0466534;
        double E1 = -3.6544371;
        double E2 = 1.8396733;
        double E3 = -0.20360244;
        double F1 = -2.0182173;
        double F2 = 1.2420401;
        double F3 = -0.21741801;

        double kappa = -T3 *
                       (E0 + E1 * std::pow(T3, 2.0) + E2 * std::pow(T3, 4.0) +
                        E3 * std::pow(T3, 6.0)) /
                       (1.0 + F1 * std::pow(T3, 2.0) + F2 * std::pow(T3, 4.0) +
                        F3 * std::pow(T3, 6.0));
        double alpha = L2 / normal_l_scale(kappa);
        double xi = L1 - GeneralizedNormal(0.0, alpha, kappa).mean();
        return {xi, alpha, kappa};
    }

    // C# LinearMomentsFromParameters (line 372). Coefficients transcribed exactly, including
    // the `x * Math.Pow(10, -n)` spelling of each constant.
    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double xi = parameters[0];
        double alpha = parameters[1];
        double kappa = parameters[2];
        if (!validate(xi, alpha, kappa))
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");

        double A0 = 4.8860251 * std::pow(10.0, -1);
        double A1 = 4.4493076 * std::pow(10.0, -3);
        double A2 = 8.8027039 * std::pow(10.0, -4);
        double A3 = 1.1507084 * std::pow(10.0, -6);
        double B1 = 6.4662924 * std::pow(10.0, -2);
        double B2 = 3.3090406 * std::pow(10.0, -3);
        double B3 = 7.4290680 * std::pow(10.0, -5);
        double C0 = 1.8756590 * std::pow(10.0, -1);
        double C1 = -2.5352147 * std::pow(10.0, -3);
        double C2 = 2.6995102 * std::pow(10.0, -4);
        double C3 = -1.8446680 * std::pow(10.0, -6);
        double D1 = 8.2325617 * std::pow(10.0, -2);
        double D2 = 4.2681448 * std::pow(10.0, -3);
        double D3 = 1.1653690 * std::pow(10.0, -4);
        double tau40 = 0.12260171954089095;

        double L1 = GeneralizedNormal(xi, alpha, kappa).mean();
        double L2 = alpha * normal_l_scale(kappa);
        double T3 = -kappa *
                    (A0 + A1 * std::pow(kappa, 2.0) + A2 * std::pow(kappa, 4.0) +
                     A3 * std::pow(kappa, 6.0)) /
                    (1.0 + B1 * std::pow(kappa, 2.0) + B2 * std::pow(kappa, 4.0) +
                     B3 * std::pow(kappa, 6.0));
        double T4 = tau40 + std::pow(kappa, 2.0) *
                                (C0 + C1 * std::pow(kappa, 2.0) + C2 * std::pow(kappa, 4.0) +
                                 C3 * std::pow(kappa, 6.0)) /
                                (1.0 + D1 * std::pow(kappa, 2.0) + D2 * std::pow(kappa, 4.0) +
                                 D3 * std::pow(kappa, 6.0));
        return {L1, L2, T3, T4};
    }

    // C# GetParameterConstraints (line 402): the C# Tuple<initial, lower, upper> becomes the
    // three out-parameters this port's IMaximumLikelihoodEstimation mixin uses.
    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials, std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        distribution_numerics::validate_sample(sample, 4);
        auto constraints = distribution_numerics::prefer_legacy_constraints(
            [&]() { return legacy_parameter_constraints(sample); },
            [&]() { return robust_parameter_constraints(sample); });
        initials = std::move(std::get<0>(constraints));
        lowers = std::move(std::get<1>(constraints));
        uppers = std::move(std::get<2>(constraints));
    }

    // C# MLE (line 429): Nelder-Mead (downhill simplex) maximizing the log-likelihood. The C#
    // `solver.ReportFailure = true` line has no counterpart -- this port's NelderMead has no
    // ReportFailure switch and always reports (see brent_search.hpp's file header for the same
    // decision on the optimizer family).
    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            GeneralizedNormal gno;
            gno.set_parameters(x);
            return gno.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, number_of_parameters(), initials, lowers,
                                              uppers);
        solver.maximize();
        const auto result = solver.best_parameters();
        if (solver.status() != math::optimization::OptimizationStatus::Success ||
            !validate(result[0], result[1], result[2]) ||
            !std::isfinite(GeneralizedNormal(result[0], result[1], result[2])
                               .log_likelihood(sample)))
            throw std::runtime_error(
                "Generalized normal maximum likelihood estimation failed or returned a "
                "nonfinite fit");
        return result;
    }

    // --- Distribution functions ---
    // C# PDF (line 451).
    double pdf(double x) const override { return std::exp(log_pdf(x)); }

    double log_pdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        if (std::isnan(x)) return x;
        if (std::isinf(x) || x <= minimum() || x >= maximum()) return -kInf;
        const double z = latent_normal(x);
        return -std::log(alpha_) - kLogSqrt2PI + z * (kappa_ - z / 2.0);
    }

    // C# CDF (line 463).
    double cdf(double x) const override { return std::exp(log_cdf(x)); }

    double log_cdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        if (x <= minimum()) return -kInf;
        if (x >= maximum()) return 0.0;
        return distribution_numerics::normal_log_cdf(latent_normal(x));
    }

    double ccdf(double x) const override { return std::exp(log_ccdf(x)); }

    double log_ccdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return -kInf;
        return distribution_numerics::normal_log_survival(latent_normal(x));
    }

    // C# InverseCDF (line 478). C# ArgumentOutOfRangeException -> std::out_of_range, matching
    // every sibling port.
    double inverse_cdf(double probability) const override {
        if (!(probability >= 0.0 && probability <= 1.0))
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        return quantile_at_latent(Normal::standard_z(probability));
    }

    // C# Clone (line 501).
    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<GeneralizedNormal>(xi_, alpha_, kappa_);
    }

    math::linalg::Matrix2D parameter_covariance(int sample_size,
                                                ParameterEstimationMethod method) const override {
        distribution_numerics::validate_sample_size(sample_size);
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        if (method != ParameterEstimationMethod::MaximumLikelihood)
            throw std::logic_error(
                "Generalized-normal covariance is implemented only for maximum likelihood");
        double c, inverse_r, log_inverse_r;
        normal_information_factors(kappa_, c, inverse_r, log_inverse_r);
        const double root_n = std::sqrt(static_cast<double>(sample_size));
        const double scale = alpha_ / root_n;
        const double shaped_scale = alpha_ * kappa_ / root_n;
        const double log_alpha = std::log(alpha_);
        const double log_n = std::log(static_cast<double>(sample_size));
        const double log_k = std::log(std::fabs(kappa_));
        math::linalg::Matrix2D covariance(3, std::vector<double>(3, 0.0));
        covariance[0][0] = scale * scale * (1.0 + c * c * inverse_r);
        covariance[1][1] = shaped_scale * shaped_scale + scale * scale / 2.0;
        covariance[2][2] = (kappa_ / 2.0 / sample_size) * kappa_ +
                           inverse_r / sample_size;
        covariance[0][1] = -scale * shaped_scale;
        if (!std::isfinite(covariance[0][1]) || covariance[0][1] == 0.0)
            covariance[0][1] =
                -std::copysign(std::exp(2.0 * log_alpha + log_k - log_n), kappa_);
        covariance[0][2] = c * inverse_r * alpha_ / sample_size;
        if (c > 0.0 && (covariance[0][2] == 0.0 || !std::isfinite(covariance[0][2])))
            covariance[0][2] =
                std::exp(log_alpha + std::log(c) + log_inverse_r - log_n);
        covariance[1][2] = alpha_ * kappa_ / 2.0 / sample_size;
        if (!std::isfinite(covariance[1][2]) || covariance[1][2] == 0.0)
            covariance[1][2] = std::copysign(
                std::exp(log_alpha + log_k - std::log(2.0) - log_n), kappa_);
        covariance[1][0] = covariance[0][1];
        covariance[2][0] = covariance[0][2];
        covariance[2][1] = covariance[1][2];
        return covariance;
    }

    double quantile_variance(double probability, int sample_size,
                             ParameterEstimationMethod method) const override {
        distribution_numerics::validate_probability(probability);
        distribution_numerics::validate_sample_size(sample_size);
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        if (method != ParameterEstimationMethod::MaximumLikelihood)
            throw std::logic_error(
                "Generalized-normal covariance is implemented only for maximum likelihood");
        double c, inverse_r, log_inverse_r;
        normal_information_factors(kappa_, c, inverse_r, log_inverse_r);
        (void)inverse_r;
        const double z = Normal::standard_z(probability);
        const double argument = -kappa_ * z;
        const double log_scale =
            std::log(alpha_) - 0.5 * std::log(static_cast<double>(sample_size));
        const double first = 2.0 * (log_scale + argument) + std::log1p(z * z / 2.0);
        if (std::isinf(first) && first > 0.0) return kInf;
        double log_residual;
        if (std::fabs(kappa_) < 0.1) {
            const double residual =
                c - z * z * distribution_numerics::exprel_derivative(argument);
            log_residual = std::log(std::fabs(residual));
        } else {
            const double negative = -(kappa_ / 2.0) * kappa_;
            double numerator;
            if (argument > 1.0) {
                numerator = distribution_numerics::log_sum(
                    argument + std::log(argument - 1.0), negative);
            } else {
                const double positive =
                    argument == 1.0 || (std::isinf(argument) && argument < 0.0)
                        ? -kInf
                        : argument + std::log1p(-argument);
                numerator = distribution_numerics::log_difference(
                    std::max(positive, negative), std::min(positive, negative));
            }
            log_residual = numerator - 2.0 * std::log(std::fabs(kappa_));
        }
        const double second = 2.0 * (log_scale + log_residual) + log_inverse_r;
        return std::exp(distribution_numerics::log_sum(first, second));
    }

    std::vector<double> quantile_gradient(double probability) const override {
        distribution_numerics::validate_probability(probability);
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        const double z = Normal::standard_z(probability);
        const double v = -kappa_ * z;
        const double shape =
            v < -50.0
                ? -std::exp(std::log(alpha_) - 2.0 * std::log(std::fabs(kappa_)))
                : v > 50.0
                      ? -std::exp(std::log(alpha_) + v + std::log(v - 1.0) -
                                  2.0 * std::log(std::fabs(kappa_)))
                      : -alpha_ * z * z * distribution_numerics::exprel_derivative(v);
        const double scale =
            v > 50.0
                ? -std::copysign(std::exp(v - std::log(std::fabs(kappa_))), kappa_)
                : v < -50.0 ? 1.0 / kappa_ : z * distribution_numerics::exprel(v);
        return {1.0, scale, shape};
    }

    // C# QuantileJacobian (line 534): one gradient per probability (there must be exactly
    // NumberOfParameters of them), plus the 3x3 determinant. C# `out determinant` -> the
    // non-const reference. C# ArgumentOutOfRangeException -> std::out_of_range.
    math::linalg::Matrix2D quantile_jacobian(const std::vector<double>& probabilities,
                                             double& determinant) const override {
        return distribution_numerics::quantile_jacobian(
            *this, number_of_parameters(), probabilities, determinant);
    }

   private:
    distribution_numerics::Constraints legacy_parameter_constraints(
        const std::vector<double>& sample) const {
        auto initials = legacy_constraint_parameters_from_linear_moments(
            data::linear_moments(sample));
        std::vector<double> lowers(3), uppers(3);
        if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
        lowers[0] = -std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
        uppers[0] = std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(std::fabs(initials[1]))) + 1.0);
        lowers[2] = -10.0;
        uppers[2] = 10.0;
        if (initials[2] <= lowers[2] || initials[2] >= uppers[2]) initials[2] = 0.0;
        return {initials, lowers, uppers};
    }

    std::vector<double> legacy_constraint_parameters_from_linear_moments(
        const std::vector<double>& moments) const {
        const double l1 = moments[0];
        const double l2 = moments[1];
        const double t3 = moments[2];
        constexpr double e0 = 2.0466534;
        constexpr double e1 = -3.6544371;
        constexpr double e2 = 1.8396733;
        constexpr double e3 = -0.20360244;
        constexpr double f1 = -2.0182173;
        constexpr double f2 = 1.2420401;
        constexpr double f3 = -0.21741801;
        const double kappa =
            -t3 * (e0 + e1 * std::pow(t3, 2.0) + e2 * std::pow(t3, 4.0) +
                   e3 * std::pow(t3, 6.0)) /
            (1.0 + f1 * std::pow(t3, 2.0) + f2 * std::pow(t3, 4.0) +
             f3 * std::pow(t3, 6.0));
        const double alpha =
            (l2 * kappa * std::exp(-(kappa * kappa) / 2.0)) /
            (1.0 - 2.0 * Normal::standard_cdf(-kappa / kSqrt2));
        const double xi = l1 - alpha * (1.0 - std::exp(kappa * kappa / 2.0)) / kappa;
        return {xi, alpha, kappa};
    }

    distribution_numerics::Constraints robust_parameter_constraints(
        const std::vector<double>& sample) const {
        double magnitude = 0.0;
        for (double value : sample) magnitude = std::max(magnitude, std::fabs(value));
        std::vector<double> scaled(sample.size());
        for (std::size_t i = 0; i < sample.size(); ++i) scaled[i] = sample[i] / magnitude;
        const auto moments = data::linear_moments(scaled);
        auto initials = parameters_from_linear_moments(moments);
        initials[0] *= magnitude;
        initials[1] *= magnitude;
        GeneralizedNormal candidate(initials[0], initials[1], initials[2]);
        if (!candidate.parameters_valid() || !std::isfinite(candidate.log_likelihood(sample)))
            initials = {moments[0] * magnitude,
                        moments[1] * std::sqrt(kPi) * magnitude, 0.0};
        std::vector<double> lowers(3), uppers(3);
        const double location_magnitude = std::max(std::fabs(initials[0]), initials[1]);
        lowers[0] = -finite_decimal_bound(location_magnitude);
        uppers[0] = -lowers[0];
        lowers[1] = std::min(kDoubleMachineEpsilon, initials[1] / 10.0);
        uppers[1] = finite_decimal_bound(initials[1]);
        lowers[2] = -10.0;
        uppers[2] = 10.0;
        if (initials[2] <= lowers[2] || initials[2] >= uppers[2]) initials[2] = 0.0;
        candidate.set_parameters(initials);
        if (!candidate.parameters_valid() || !std::isfinite(candidate.log_likelihood(sample)) ||
            initials[0] <= lowers[0] || initials[0] >= uppers[0] ||
            initials[1] <= lowers[1] || initials[1] >= uppers[1])
            throw std::runtime_error(
                "the sample does not admit a finite supported generalized-normal initializer "
                "within finite bounds");
        return {initials, lowers, uppers};
    }

    static double finite_decimal_bound(double value) {
        const double bound = std::pow(10.0, std::ceil(std::log10(value)) + 1.0);
        return std::isinf(bound) ? std::numeric_limits<double>::max() : bound;
    }

    std::vector<double> analytical_moments() const {
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        if (kappa_ == 0.0) return {xi_, alpha_, 0.0, 3.0};
        const double v = kappa_ * kappa_;
        const double t = std::expm1(v);
        double mean_value, sd_value, skew_value;
        if (v <= 0.5) {
            const double relative = distribution_numerics::exprel(v);
            mean_value = xi_ - alpha_ * kappa_ * 0.5 *
                                   distribution_numerics::exprel(0.5 * v);
            sd_value = alpha_ * std::exp(0.5 * v) * std::sqrt(relative);
            skew_value = -kappa_ * (t + 3.0) * std::sqrt(relative);
        } else {
            const double log_t =
                v > 36.0 ? v + std::log1p(-std::exp(-v)) : std::log(t);
            const double log_half =
                v > 72.0 ? v / 2.0 + std::log1p(-std::exp(-v / 2.0))
                         : std::log(std::expm1(v / 2.0));
            mean_value = xi_ - std::copysign(
                                   std::exp(std::log(alpha_) + log_half -
                                            std::log(std::fabs(kappa_))),
                                   kappa_);
            sd_value = std::exp(std::log(alpha_) + v / 2.0 + log_t / 2.0 -
                                std::log(std::fabs(kappa_)));
            const double log_sum =
                v > 36.0 ? v + std::log1p(2.0 * std::exp(-v)) : std::log(t + 3.0);
            skew_value = -std::copysign(std::exp(log_sum + log_t / 2.0), kappa_);
        }
        return {mean_value, sd_value, skew_value,
                3.0 + t * (16.0 + t * (15.0 + t * (6.0 + t)))};
    }

    static double normal_l_scale(double kappa) {
        const double v = kappa * kappa;
        if (std::fabs(kappa) < 0.5) {
            double sum = 1.0;
            double power = 1.0;
            for (int n = 1; n < 24; ++n) {
                power *= -v / (4.0 * n);
                const double term = power / (2.0 * n + 1.0);
                sum += term;
                if (std::fabs(term) < std::fabs(sum) * 1e-17) break;
            }
            return std::exp(v / 2.0) * sum / std::sqrt(kPi);
        }
        return std::exp(v / 2.0) *
               (1.0 - 2.0 * Normal::standard_cdf(-std::fabs(kappa) / kSqrt2)) /
               std::fabs(kappa);
    }

    static void normal_information_factors(double kappa, double& c, double& inverse_r,
                                           double& log_inverse_r) {
        const double v = kappa * kappa;
        c = 0.5 * distribution_numerics::exprel(-v / 2.0);
        if (v < 0.1) {
            double sum = 1.5;
            double term = 1.5;
            for (int m = 2; m < 24; ++m) {
                term *= v * (m + 2.0) / (m + 1.0) / (m + 1.0);
                sum += term;
                if (std::fabs(term) < std::fabs(sum) * 1e-17) break;
            }
            inverse_r = 1.0 / sum;
            log_inverse_r = -std::log(sum);
        } else {
            log_inverse_r =
                std::isinf(v) ? -kInf
                              : 2.0 * std::log(v) - v -
                                    (v > 350.0
                                         ? std::log1p(v)
                                         : std::log(1.0 + v -
                                                    (1.0 + 2.0 * v) * std::exp(-v)));
            inverse_r = std::exp(log_inverse_r);
        }
        if (std::isinf(v)) c = 0.0;
    }

    double finite_shape_endpoint() const {
        const double shift = alpha_ / kappa_;
        return std::isinf(shift) && std::signbit(xi_) != std::signbit(shift)
                   ? (xi_ * kappa_ + alpha_) / kappa_
                   : xi_ + shift;
    }

    double latent_normal(double x) const {
        const double y = distribution_numerics::standardize(x, xi_, alpha_);
        if (kappa_ == 0.0) return y;
        const double product = -kappa_ * y;
        if (product < -0.9)
            return -distribution_numerics::KappaFourBoundary::log_t(
                x, xi_, alpha_, kappa_);
        return distribution_numerics::hosking_shape_transform(
            x, xi_, alpha_, kappa_);
    }

    double quantile_at_latent(double z) const {
        const double v = -kappa_ * z;
        const double standard =
            v > 50.0 ? -std::copysign(1.0, kappa_) *
                           std::exp(v - std::log(std::fabs(kappa_)))
                     : v < -50.0 ? 1.0 / kappa_ : z * distribution_numerics::exprel(v);
        const double offset =
            v > 50.0 ? -std::copysign(1.0, kappa_) *
                           std::exp(std::log(alpha_) + v - std::log(std::fabs(kappa_)))
                     : alpha_ * standard;
        const double value = xi_ + offset;
        if (std::isinf(value) && std::isfinite(standard)) {
            const double combined = xi_ / alpha_ + standard;
            if (std::isfinite(combined)) return alpha_ * combined;
        }
        return value;
    }

    void ensure_moments() const {
        if (!moments_computed_) {
            u_ = analytical_moments();
            moments_computed_ = true;
        }
    }

    double xi_ = 0.0;     // location
    double alpha_ = 0.0;  // scale
    double kappa_ = 0.0;  // shape

    mutable bool moments_computed_ = false;
    mutable std::vector<double> u_ = {kNaN, kNaN, kNaN, kNaN};
};

}  // namespace corehydro::numerics::distributions
