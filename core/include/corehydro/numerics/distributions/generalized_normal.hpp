// ported from: Numerics/Distributions/Univariate/GeneralizedNormal.cs @ 2a0357a
//
// The Generalized Normal distribution (LogNormal-3), parameterized by location ξ (Xi),
// scale α (Alpha), and shape κ (Kappa). κ→0 is the plain Normal; κ < 0 is bounded below at
// ξ + α/κ; κ > 0 is bounded above at the same point.
//
// Structurally the closest kin to generalized_extreme_value.hpp and generalized_logistic.hpp
// (same three parameters, same interface set, same L-moment/MLE shape), and this file mirrors
// the C# member order. Two members differ from those two siblings because the C# source does:
//   * the product moments are NOT closed form -- C# reads CentralMoments(1000) lazily into a
//     cached u[4], so this port calls the base class's faithful `central_moments(1000)`;
//   * Mode is a bounded BrentSearch maximization of the PDF over
//     [InverseCDF(0.001), InverseCDF(0.999)], not an analytic expression.
//
// C# members with no counterpart here, severed layer-wide across every ported distribution
// (WPF/serialization surface, never referenced by the numeric core): DisplayName /
// ShortDisplayName, ParametersToString (its first column IS `parameter_names()` below),
// GetParameterPropertyNames, MinimumOfParameters / MaximumOfParameters.
//
// ParameterCovariance and QuantileVariance throw here because they throw upstream
// (NotImplementedException on GeneralizedNormal.cs:509 and :515). They are declared anyway:
// the C# class declares IStandardError, so the port does too, and a caller that dynamic_casts
// to the mixin must get the same "not implemented" answer C# gives rather than a silent
// wrong number. C# NotImplementedException -> std::logic_error, following von_mises.hpp /
// generalized_pareto.hpp.
#pragma once
#include <cmath>
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
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/brent_search.hpp"
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
    // C# Mean / StandardDeviation / Skewness / Kurtosis (lines 149/180/194/208) all read the
    // lazily computed CentralMoments(1000) cache.
    double mean() const override {
        ensure_moments();
        return u_[0];
    }

    // C# Median (line 163).
    double median() const override { return inverse_cdf(0.5); }

    // C# Mode (line 169): BrentSearch maximizing the PDF over
    // [InverseCDF(0.001), InverseCDF(0.999)], returning BestParameterSet.Values[0].
    double mode() const override {
        math::optimization::BrentSearch brent([this](double x) { return pdf(x); },
                                              inverse_cdf(0.001), inverse_cdf(0.999));
        brent.maximize();
        return brent.best_parameter();
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
        if (kappa_ >= -kNearZero) return -kInf;
        return xi_ + alpha_ / kappa_;
    }

    // C# Maximum (line 238): unbounded above unless κ > NearZero.
    double maximum() const override {
        if (kappa_ <= kNearZero) return kInf;
        return xi_ + alpha_ / kappa_;
    }

    // --- Estimation ---
    // C# Estimate (line 266): L-moments or MLE only; anything else throws
    // NotImplementedException.
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
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
        double alpha = (L2 * kappa * std::exp(-(kappa * kappa) / 2.0)) /
                       (1.0 - 2.0 * Normal::standard_cdf(-kappa / kSqrt2));
        double xi = L1 - alpha * (1.0 - std::exp(kappa * kappa / 2.0)) / kappa;
        return {xi, alpha, kappa};
    }

    // C# LinearMomentsFromParameters (line 372). Coefficients transcribed exactly, including
    // the `x * Math.Pow(10, -n)` spelling of each constant.
    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double xi = parameters[0];
        double alpha = parameters[1];
        double kappa = parameters[2];

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
        double tau40 = 1.2260172 * std::pow(10.0, -1);

        double L1 = xi + alpha * (1.0 - std::exp(kappa * kappa / 2.0)) / kappa;
        double L2 = (alpha / kappa) * std::exp(kappa * kappa / 2.0) *
                    (1.0 - 2.0 * Normal::standard_cdf(-kappa / kSqrt2));
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
        lowers.assign(static_cast<std::size_t>(number_of_parameters()), 0.0);
        uppers.assign(static_cast<std::size_t>(number_of_parameters()), 0.0);
        // Get initial values
        initials = parameters_from_linear_moments(data::linear_moments(sample));
        // Get bounds of location
        if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
        lowers[0] = -std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
        uppers[0] = std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
        // Get bounds of scale
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(std::fabs(initials[1]))) + 1.0);
        // Get bounds of shape
        lowers[2] = -10.0;
        uppers[2] = 10.0;
        // Correct initial value of kappa if necessary
        if (initials[2] <= lowers[2] || initials[2] >= uppers[2]) initials[2] = 0.0;
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
        return solver.best_parameters();
    }

    // --- Distribution functions ---
    // C# PDF (line 451).
    double pdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        if (x < minimum() || x > maximum()) return 0.0;
        double y = (x - xi_) / alpha_;
        if (std::fabs(kappa_) > kNearZero) y = -std::log(1.0 - kappa_ * y) / kappa_;
        return 1.0 / alpha_ * std::exp(kappa_ * y - y * y / 2.0) / kSqrt2PI;
    }

    // C# CDF (line 463).
    double cdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        double y = (x - xi_) / alpha_;
        if (std::fabs(kappa_) > kNearZero) y = -std::log(1.0 - kappa_ * y) / kappa_;
        return Normal::standard_cdf(y);
    }

    // C# InverseCDF (line 478). C# ArgumentOutOfRangeException -> std::out_of_range, matching
    // every sibling port.
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        if (std::fabs(kappa_) <= kNearZero) return xi_ + alpha_ * Normal::standard_z(probability);
        return xi_ - alpha_ / kappa_ * (std::exp(-kappa_ * Normal::standard_z(probability)) - 1.0);
    }

    // C# Clone (line 501).
    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<GeneralizedNormal>(xi_, alpha_, kappa_);
    }

    // --- IStandardError ---
    // C# ParameterCovariance (line 507): throw new NotImplementedException().
    math::linalg::Matrix2D parameter_covariance(int sample_size,
                                                ParameterEstimationMethod method) const override {
        (void)sample_size;
        (void)method;
        throw std::logic_error(
            "GeneralizedNormal::parameter_covariance is not implemented upstream");
    }

    // C# QuantileVariance (line 513): throw new NotImplementedException().
    double quantile_variance(double probability, int sample_size,
                             ParameterEstimationMethod method) const override {
        (void)probability;
        (void)sample_size;
        (void)method;
        throw std::logic_error("GeneralizedNormal::quantile_variance is not implemented upstream");
    }

    // C# QuantileGradient (line 519): the numerical gradient of InverseCDF(probability) with
    // respect to {ξ, α, κ}, evaluated at the current parameters. C# validates against
    // (Xi, _alpha, Kappa) -- the private backing field, identical in value to the property.
    std::vector<double> quantile_gradient(double probability) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedNormal: invalid parameters");
        return math::differentiation::gradient(
            [probability](const std::vector<double>& x) {
                GeneralizedNormal gno;
                gno.set_parameters(x);
                return gno.inverse_cdf(probability);
            },
            get_parameters());
    }

    // C# QuantileJacobian (line 534): one gradient per probability (there must be exactly
    // NumberOfParameters of them), plus the 3x3 determinant. C# `out determinant` -> the
    // non-const reference. C# ArgumentOutOfRangeException -> std::out_of_range.
    math::linalg::Matrix2D quantile_jacobian(const std::vector<double>& probabilities,
                                             double& determinant) const override {
        if (static_cast<int>(probabilities.size()) != number_of_parameters()) {
            throw std::out_of_range(
                "The number of probabilities must be the same length as the number of "
                "distribution parameters.");
        }
        // Get gradients
        auto dQp1 = quantile_gradient(probabilities[0]);
        auto dQp2 = quantile_gradient(probabilities[1]);
        auto dQp3 = quantile_gradient(probabilities[2]);
        // Compute determinant
        // |a b c|
        // |d e f|
        // |g h i|
        // |A| = a(ei - fh) - b(di - fg) + c(dh - eg)
        double a = dQp1[0];
        double b = dQp1[1];
        double c = dQp1[2];
        double d = dQp2[0];
        double e = dQp2[1];
        double f = dQp2[2];
        double g = dQp3[0];
        double h = dQp3[1];
        double i = dQp3[2];
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        // Return Jacobian
        return {{a, b, c}, {d, e, f}, {g, h, i}};
    }

   private:
    // C# reads the CentralMoments(1000) quadruple into `u` on first access and caches it
    // behind `_momentsComputed` (GeneralizedNormal.cs:45-46, 153-158). `mutable` here because
    // the four C# properties are const accessors in this port.
    void ensure_moments() const {
        if (!moments_computed_) {
            u_ = central_moments(1000);
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
