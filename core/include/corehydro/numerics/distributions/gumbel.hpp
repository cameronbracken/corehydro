// ported from: Numerics/Distributions/Univariate/Gumbel.cs @ 7e8e8d1
//
// The Gumbel (Extreme Value Type I) distribution with location ξ and scale α. Logic
// mirrors the C# source method-for-method. The WPF helpers, IBootstrappable, and
// IStandardError interfaces are not ported (desktop / uncertainty-analysis concerns).
// Re-audited against v2.1.4's "Harden distribution parameter validation" wave: C#'s
// SetParameters now assigns Xi/Alpha directly (bypassing the Xi/Alpha property setters,
// which independently revalidate against a possibly-stale sibling field) before computing
// _parametersValid once from the final (location, scale) pair. This port already assigned
// xi_/alpha_ directly and validated once at the end -- pre-aligned, no code change.
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
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class Gumbel : public UnivariateDistributionBase,
               public IEstimation,
               public ILinearMomentEstimation,
               public IMaximumLikelihoodEstimation {
   public:
    Gumbel() { set_parameters(100.0, 10.0); }
    Gumbel(double location, double scale) { set_parameters(location, scale); }

    double xi() const { return xi_; }
    double alpha() const { return alpha_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Gumbel;
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
    double mean() const override { return xi_ + alpha_ * kEuler; }
    double median() const override { return xi_ - alpha_ * std::log(std::log(2.0)); }
    double mode() const override { return xi_; }
    double standard_deviation() const override {
        return alpha_ * (kPi / std::sqrt(6.0));
    }
    double skewness() const override { return 1.1396; }
    double kurtosis() const override { return 3.0 + 12.0 / 5.0; }
    double minimum() const override { return -kInf; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override { return std::exp(log_pdf(x)); }

    double log_pdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Gumbel: invalid parameters");
        const double z = distribution_numerics::standardize(x, xi_, alpha_);
        const double value = -(z + std::exp(-z)) - std::log(alpha_);
        return std::isnan(value) ? -kInf : value;
    }

    double cdf(double x) const override { return std::exp(log_cdf(x)); }

    double log_cdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Gumbel: invalid parameters");
        return -std::exp(-distribution_numerics::standardize(x, xi_, alpha_));
    }

    double ccdf(double x) const override { return -std::expm1(log_cdf(x)); }

    double log_ccdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Gumbel: invalid parameters");
        const double z = distribution_numerics::standardize(x, xi_, alpha_);
        const double exponential = std::exp(-z);
        return exponential == 0.0 ? -z : distribution_numerics::log1m_exp(-exponential);
    }

    double inverse_cdf(double probability) const override {
        if (!(probability >= 0.0 && probability <= 1.0))
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        const double unit_quantile = -std::log(-std::log(probability));
        const double displacement = alpha_ * unit_quantile;
        return std::isinf(displacement) && std::isfinite(unit_quantile)
                   ? alpha_ * (xi_ / alpha_ + unit_quantile)
                   : xi_ + displacement;
    }

    // --- Parameter display names (X1; C# Gumbel.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Location (\xCE\xBE)", "Scale (\xCE\xB1)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBE", "\xCE\xB1"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Gumbel>(xi_, alpha_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            set_parameters(parameters_from_moments(data::product_moments(sample)));
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else {
            set_parameters(mle(sample));
        }
    }

    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        double a = std::sqrt(6.0) / kPi * moments[1];
        double x = moments[0] - a * kEuler;
        return {x, a};
    }

    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        double L1 = moments[0];
        double L2 = moments[1];
        double alpha = L2 / kLog2;
        double xi = L1 - alpha * kEuler;
        return {xi, alpha};
    }

    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double xi = parameters[0];
        double alpha = parameters[1];
        double L1 = xi + alpha * kEuler;
        double L2 = alpha * std::log(2.0);
        double T3 = std::log(9.0 / 8.0) / std::log(2.0);
        double T4 = (16.0 * std::log(2.0) - 10.0 * std::log(3.0)) / std::log(2.0);
        return {L1, L2, T3, T4};
    }

    void get_parameter_constraints(const std::vector<double>& sample, std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        initials = parameters_from_linear_moments(data::linear_moments(sample));
        lowers.resize(2);
        uppers.resize(2);
        if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
        lowers[0] = -std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
        uppers[0] =  std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(initials[1]) + 1.0));
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            Gumbel g;
            g.set_parameters(x[0], x[1]);
            return g.log_likelihood(sample);
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
