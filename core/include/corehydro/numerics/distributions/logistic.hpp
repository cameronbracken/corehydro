// ported from: Numerics/Distributions/Univariate/Logistic.cs @ 7e8e8d1
//
// The Logistic distribution with location ξ and scale α. Logic mirrors the C# source
// method-for-method. IStandardError, IBootstrappable, and the WPF helpers are not ported
// (desktop / uncertainty-analysis concerns). ILinearMomentEstimation is not implemented
// in the C# source and is therefore absent here.
// Re-audited against v2.1.4's "Harden distribution parameter validation" wave: C#'s
// SetParameters now assigns Xi/Alpha directly before computing _parametersValid once from
// the final (location, scale) pair (the old code let the Xi/Alpha property setters
// revalidate against a stale sibling field). This port already assigned xi_/alpha_
// directly and validated once at the end -- pre-aligned, no code change.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class Logistic : public UnivariateDistributionBase,
                 public IEstimation,
                 public IMaximumLikelihoodEstimation {
   public:
    Logistic() { set_parameters(0.0, 0.1); }
    Logistic(double location, double scale) { set_parameters(location, scale); }

    double xi() const { return xi_; }
    double alpha() const { return alpha_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Logistic;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {xi_, alpha_}; }

    void set_parameters(double location, double scale) {
        xi_ = location;
        alpha_ = scale;
        parameters_valid_ = validate(location, scale);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override { return xi_; }
    double median() const override { return inverse_cdf(0.5); }
    double mode() const override { return xi_; }
    double standard_deviation() const override { return alpha_ * kPi / std::sqrt(3.0); }
    double skewness() const override { return 0.0; }
    double kurtosis() const override { return 3.0 + 6.0 / 5.0; }
    double minimum() const override { return -kInf; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Logistic: invalid parameters");
        const double magnitude =
            std::fabs(distribution_numerics::standardize(x, xi_, alpha_));
        if (magnitude > 36.0) return std::exp(log_pdf(x));
        const double tail = std::exp(-magnitude);
        return (tail / (1.0 + tail)) / (1.0 + tail) / alpha_;
    }

    double log_pdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Logistic: invalid parameters");
        const double magnitude =
            std::fabs(distribution_numerics::standardize(x, xi_, alpha_));
        return -std::log(alpha_) - magnitude - 2.0 * std::log1p(std::exp(-magnitude));
    }

    double cdf(double x) const override { return std::exp(log_cdf(x)); }

    double log_cdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Logistic: invalid parameters");
        const double z = distribution_numerics::standardize(x, xi_, alpha_);
        return z >= 0.0 ? -std::log1p(std::exp(-z)) : z - std::log1p(std::exp(z));
    }

    double ccdf(double x) const override { return std::exp(log_ccdf(x)); }

    double log_ccdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Logistic: invalid parameters");
        const double z = distribution_numerics::standardize(x, xi_, alpha_);
        return z >= 0.0 ? -z - std::log1p(std::exp(-z)) : -std::log1p(std::exp(z));
    }

    double inverse_cdf(double probability) const override {
        if (std::isnan(probability) || probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return xi_ + alpha_ * (std::log(probability) - std::log1p(-probability));
    }

    // --- Parameter display names (X1; C# Logistic.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Location (\xCE\xBE)", "Scale (\xCE\xB1)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBE", "\xCE\xB1"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Logistic>(xi_, alpha_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        distribution_numerics::validate_sample(sample, 4);
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            set_parameters(parameters_from_moments(data::product_moments(sample)));
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::invalid_argument("estimation method not implemented for Logistic");
        }
    }

    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        double xi = moments[0];
        double alpha = moments[1] * std::sqrt(3.0) / kPi;
        return {xi, alpha};
    }

    void get_parameter_constraints(const std::vector<double>& sample, std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        distribution_numerics::validate_sample(sample, 4);
        auto constraints = distribution_numerics::prefer_legacy_constraints(
            [&]() { return legacy_parameter_constraints(sample); },
            [&]() { return robust_parameter_constraints(sample); });
        initials = std::move(std::get<0>(constraints));
        lowers = std::move(std::get<1>(constraints));
        uppers = std::move(std::get<2>(constraints));
    }

   private:
    distribution_numerics::Constraints legacy_parameter_constraints(
        const std::vector<double>& sample) const {
        auto initials = parameters_from_moments(data::product_moments(sample));
        std::vector<double> lowers(2), uppers(2);
        // bounds for location
        double xi0 = initials[0] != 0.0 ? initials[0] : kDoubleMachineEpsilon;
        lowers[0] = -std::pow(10.0, std::ceil(std::log10(std::fabs(xi0)) + 1.0));
        uppers[0] =  std::pow(10.0, std::ceil(std::log10(std::fabs(xi0)) + 1.0));
        // bounds for scale
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(initials[1]) + 1.0));
        return {initials, lowers, uppers};
    }

    distribution_numerics::Constraints robust_parameter_constraints(
        const std::vector<double>& sample) const {
        Normal normal;
        auto constraints = normal.robust_parameter_constraints(sample);
        const double correction = std::sqrt(3.0) / kPi;
        std::get<0>(constraints)[1] *= correction;
        std::get<1>(constraints)[1] *= correction;
        std::get<2>(constraints)[1] *= correction;
        return constraints;
    }

   public:
    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            Logistic lo;
            lo.set_parameters(x[0], x[1]);
            return lo.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 2, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

   private:
    static bool validate(double location, double scale) {
        if (std::isnan(location) || std::isinf(location)) return false;
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        return true;
    }

    double xi_ = 0.0;
    double alpha_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
