// ported from: Numerics/Distributions/Univariate/GeneralizedPareto.cs @ 2a0357a
//
// Generalized Pareto distribution: parameters ξ (location), α (scale), κ (shape).
// Mirrors the C# source method-for-method. The IBootstrappable and WPF helpers are not
// ported (desktop / uncertainty-analysis concerns); of the IStandardError surface only
// ParameterCovariance is ported (M5: ThresholdDiagnostics' parameter-stability plot
// needs it) -- QuantileVariance/QuantileGradient/QuantileJacobian remain unported.
// κ→0 limit branch: exponential distribution on [ξ, ∞).
// κ > 0: bounded above at ξ + α/κ.
// κ < 0: heavy tail, unbounded above.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class GeneralizedPareto : public UnivariateDistributionBase,
                          public IEstimation,
                          public ILinearMomentEstimation,
                          public IMaximumLikelihoodEstimation {
   public:
    GeneralizedPareto() { set_parameters(100.0, 10.0, 0.0); }
    GeneralizedPareto(double location, double scale, double shape) {
        set_parameters(location, scale, shape);
    }

    double xi() const { return xi_; }
    double alpha() const { return alpha_; }
    double kappa() const { return kappa_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::GeneralizedPareto;
    }
    int number_of_parameters() const override { return 3; }
    std::vector<double> get_parameters() const override { return {xi_, alpha_, kappa_}; }

    void set_parameters(double location, double scale, double shape) {
        xi_ = location;
        alpha_ = scale;
        kappa_ = shape;
        parameters_valid_ = validate(location, scale, shape);
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2]);
    }

    // --- Moments / support ---
    double mean() const override {
        if (std::fabs(kappa_) <= kNearZero) return xi_ + alpha_;
        if (std::fabs(kappa_) < 1.0) return xi_ + alpha_ / (1.0 + kappa_);
        return kNaN;
    }

    double median() const override {
        if (std::fabs(kappa_) <= kNearZero)
            return xi_ - std::log(0.5) * alpha_;
        return xi_ + alpha_ * (std::pow(2.0, -kappa_) - 1.0) / kappa_;
    }

    double mode() const override {
        if (std::fabs(kappa_) <= kNearZero) return xi_;
        return xi_ + alpha_ * (std::pow(1.0 + kappa_, -kappa_) - 1.0) / kappa_;
    }

    double standard_deviation() const override {
        if (std::fabs(kappa_) <= kNearZero) return alpha_;
        if (std::fabs(kappa_) < 0.5) {
            double num = alpha_ * alpha_;
            double den = (1.0 + 2.0 * kappa_) * std::pow(1.0 + kappa_, 2.0);
            return std::sqrt(num / den);
        }
        return kNaN;
    }

    double skewness() const override {
        if (std::fabs(kappa_) <= kNearZero) return 2.0;
        if (std::fabs(kappa_) < 1.0 / 3.0) {
            double num = 2.0 * (1.0 - kappa_) * std::sqrt(1.0 + 2.0 * kappa_);
            double den = 1.0 + 3.0 * kappa_;
            return num / den;
        }
        return kNaN;
    }

    double kurtosis() const override {
        if (std::fabs(kappa_) <= kNearZero) return 9.0;
        if (std::fabs(kappa_) < 0.25) {
            double num = 3.0 * (1.0 + 2.0 * kappa_) *
                         (3.0 - kappa_ + 2.0 * std::pow(kappa_, 2.0));
            double den = (1.0 + 3.0 * kappa_) * (1.0 + 4.0 * kappa_);
            return num / den;
        }
        return kNaN;
    }

    double minimum() const override { return xi_; }

    double maximum() const override {
        if (kappa_ <= kNearZero) return kInf;
        return xi_ + alpha_ / kappa_;
    }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (x < minimum() || x > maximum()) return 0.0;
        double y = (x - xi_) / alpha_;
        if (std::fabs(kappa_) > kNearZero)
            y = -std::log(1.0 - kappa_ * y) / kappa_;
        return std::exp(-(1.0 - kappa_) * y) / alpha_;
    }

    double cdf(double x) const override {
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        double y = (x - xi_) / alpha_;
        if (std::fabs(kappa_) > kNearZero)
            y = -std::log(1.0 - kappa_ * y) / kappa_;
        return 1.0 - std::exp(-y);
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (std::fabs(kappa_) <= kNearZero)
            return xi_ - alpha_ * std::log(1.0 - probability);
        return xi_ + alpha_ / kappa_ * (1.0 - std::pow(1.0 - probability, kappa_));
    }

    // --- Parameter display names (X1; C# GeneralizedPareto.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Location (\xCE\xBE)", "Scale (\xCE\xB1)", "Shape (\xCE\xBA)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBE", "\xCE\xB1", "\xCE\xBA"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<GeneralizedPareto>(xi_, alpha_, kappa_);
    }

    // Parameter covariance matrix, 3x3 over {location, scale, shape} (C#
    // ParameterCovariance, line 635). Closed-form asymptotic formulas; only the
    // MethodOfMoments and MaximumLikelihood branches exist upstream (anything else
    // throws, C# NotImplementedException -> std::logic_error).
    math::linalg::Matrix2D parameter_covariance(
        int sample_size, ParameterEstimationMethod estimation_method) const {
        if (estimation_method != ParameterEstimationMethod::MethodOfMoments &&
            estimation_method != ParameterEstimationMethod::MaximumLikelihood) {
            throw std::logic_error(
                "GeneralizedPareto::parameter_covariance is only implemented for the "
                "method of moments and maximum likelihood estimation methods.");
        }
        double a = alpha_;
        double k = kappa_;
        double n = static_cast<double>(sample_size);
        math::linalg::Matrix2D covar(3, std::vector<double>(3, 0.0));
        covar[0][0] = n * a * a / ((n + 2.0 * k) * std::pow(n + k, 2.0));  // location
        if (estimation_method == ParameterEstimationMethod::MethodOfMoments) {
            double num = std::pow(1.0 + k, 2.0) * (1.0 + 6.0 * k + 12.0 * std::pow(k, 2.0));
            double den = (1.0 + 2.0 * k) * (1.0 + 3.0 * k) * (1.0 + 4.0 * k);
            covar[1][1] = 2.0 * a * a / n * num / den;  // scale
            //
            num = std::pow(1.0 + k, 2.0) * std::pow(1.0 + 2.0 * k, 2.0) *
                  (1.0 + k + 6.0 * std::pow(k, 2.0));
            covar[2][2] = 1.0 / n * num / den;  // shape
            //
            covar[0][1] = 0.0;
            covar[0][2] = 0.0;
            covar[1][0] = 0.0;
            covar[2][0] = 0.0;
            //
            num = std::pow(1.0 + k, 2.0) * (1.0 + 2.0 * k) *
                  (1.0 + 4.0 * k + 12.0 * std::pow(k, 2.0));
            den = (1.0 + 2.0 * k) * (1.0 + 3.0 * k) * (1.0 + 4.0 * k);
            covar[2][1] = a / n * num / den;  // scale & shape
            covar[1][2] = covar[2][1];
        } else {
            covar[1][1] = (1.0 - k) * (2.0 * a * a) / n;      // scale
            covar[2][2] = 1.0 / n * std::pow(1.0 - k, 2.0);   // shape
            //
            covar[0][1] = 0.0;
            covar[0][2] = 0.0;
            covar[1][0] = 0.0;
            covar[2][0] = 0.0;
            //
            covar[2][1] = a / n * (1.0 - k);  // scale & shape
            covar[1][2] = covar[2][1];
        }
        return covar;
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample,
                  ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            set_parameters(direct_method_of_moments(data::product_moments(sample)));
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else {
            set_parameters(mle(sample));
        }
    }

    // Solve for the shape κ given a skewness coefficient using Brent's method.
    double solve_for_kappa(double skew) const {
        if (std::fabs(skew) < 10.0) {
            return math::rootfinding::solve(
                [skew](double x) {
                    double k = 2.0 * (1.0 - x) * std::sqrt(1.0 + 2.0 * x) / (1.0 + 3.0 * x);
                    return k - skew;
                },
                -(1.0 / 3.0), 1.0 / 3.0);
        }
        return kNaN;
    }

    // Direct method of moments using product moments.
    std::vector<double> direct_method_of_moments(const std::vector<double>& moments) const {
        double k = solve_for_kappa(moments[2]);
        double a, x;
        if (std::fabs(k) <= kNearZero) {
            x = moments[0] - moments[1];
            a = moments[1];
        } else {
            a = std::sqrt(moments[1] * moments[1] * std::pow(1.0 + k, 2.0) *
                          (1.0 + 2.0 * k));
            x = moments[0] - a / (1.0 + k);
        }
        return {x, a, k};
    }

    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        double L1 = moments[0];
        double L2 = moments[1];
        double T3 = moments[2];
        double kappa = (1.0 - 3.0 * T3) / (1.0 + T3);
        double alpha = (1.0 + kappa) * (2.0 + kappa) * L2;
        double xi = L1 - (2.0 + kappa) * L2;
        return {xi, alpha, kappa};
    }

    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double xi = parameters[0];
        double alpha = parameters[1];
        double kappa = parameters[2];
        if (kappa <= -1.0)
            throw std::out_of_range("L-moments can only be defined for kappa > -1");
        double L1 = xi + alpha / (1.0 + kappa);
        double L2 = alpha / ((1.0 + kappa) * (2.0 + kappa));
        double T3 = (1.0 - kappa) / (3.0 + kappa);
        double T4 = (1.0 - kappa) * (2.0 - kappa) / ((3.0 + kappa) * (4.0 + kappa));
        return {L1, L2, T3, T4};
    }

    // Initial values and bounds for MLE optimization.
    // C# returns the FULL NumberOfParameters-length (xi, alpha, kappa) arrays here
    // (GeneralizedPareto.cs 515-549); the xi-is-fixed 2-parameter slice belongs to MLE,
    // which applies `.Subset(1)` at the NelderMead call (line 570). Folding that slice
    // into this method truncated the arrays to 2 for every OTHER caller of the
    // IMaximumLikelihoodEstimation interface -- see mle() below.
    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        auto all_initials = parameters_from_linear_moments(data::linear_moments(sample));
        double min_data = *std::min_element(sample.begin(), sample.end());
        // Location bound (upper = min_data so xi <= min_data)
        if (all_initials[0] == 0.0) all_initials[0] = kDoubleMachineEpsilon;
        double loc_lower = all_initials[0] -
                           std::pow(10.0, std::ceil(std::log10(std::fabs(all_initials[0]))));
        double loc_upper = min_data + kDoubleMachineEpsilon;
        // Scale bounds
        double scl_lower = kDoubleMachineEpsilon;
        double scl_upper = std::pow(10.0,
                               std::ceil(std::log10(std::fabs(all_initials[1])) + 1.0));
        // Correct xi initial if out of range
        if (all_initials[0] <= loc_lower || all_initials[0] >= loc_upper)
            all_initials[0] = 0.5 * (loc_lower + loc_upper);
        if (all_initials[1] <= scl_lower || all_initials[1] >= scl_upper)
            all_initials[1] = 0.5 * (scl_lower + scl_upper);
        if (all_initials[2] <= -10.0 || all_initials[2] >= 10.0) all_initials[2] = 0.0;
        // C# 548: the full 3-parameter (xi, alpha, kappa) constraint triple.
        initials = {all_initials[0], all_initials[1], all_initials[2]};
        lowers = {loc_lower, scl_lower, -10.0};
        uppers = {loc_upper, scl_upper, 10.0};
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> all_initials, all_lowers, all_uppers;
        get_parameter_constraints(sample, all_initials, all_lowers, all_uppers);
        // C# 570: `Initials.Subset(1)` / `Lowers.Subset(1)` / `Uppers.Subset(1)` -- xi is
        // fixed at min(sample), so only (alpha, kappa) reach the optimizer.
        std::vector<double> initials(all_initials.begin() + 1, all_initials.end());
        std::vector<double> lowers(all_lowers.begin() + 1, all_lowers.end());
        std::vector<double> uppers(all_uppers.begin() + 1, all_uppers.end());
        double xi_fixed = *std::min_element(sample.begin(), sample.end());
        auto log_lh = [&sample, xi_fixed](const std::vector<double>& x) {
            GeneralizedPareto g;
            g.set_parameters(xi_fixed, x[0], x[1]);
            return g.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 2, initials, lowers, uppers);
        solver.maximize();
        auto best = solver.best_parameters();
        return {xi_fixed, best[0], best[1]};
    }

   private:
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
