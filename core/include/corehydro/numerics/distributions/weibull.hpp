// ported from: Numerics/Distributions/Univariate/Weibull.cs @ 7e8e8d1
//
// Weibull distribution with scale λ (lambda) and shape κ (kappa). Logic mirrors the C#
// source method-for-method. The WPF helpers, IBootstrappable, and IStandardError
// interfaces are not ported (desktop / uncertainty-analysis concerns).
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
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace sf_wb = corehydro::numerics::math::special;

class Weibull : public UnivariateDistributionBase,
                public IEstimation,
                public IMaximumLikelihoodEstimation {
   public:
    // Constructs a Weibull distribution with scale = 10 and shape = 2.
    Weibull() { set_parameters(10.0, 2.0); }

    // Constructs a Weibull distribution with given scale λ and shape κ.
    Weibull(double scale, double shape) { set_parameters(scale, shape); }

    double lambda() const { return lambda_; }
    double kappa()  const { return kappa_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Weibull;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {lambda_, kappa_}; }

    void set_parameters(double scale, double shape) {
        lambda_ = scale;
        kappa_  = shape;
        parameters_valid_ = validate(scale, shape);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override {
        return lambda_ * sf_wb::function(1.0 + 1.0 / kappa_);
    }

    double median() const override {
        return lambda_ * std::pow(std::log(2.0), 1.0 / kappa_);
    }

    double mode() const override {
        if (kappa_ <= 1.0)
            return 0.0;
        return lambda_ * std::pow((kappa_ - 1.0) / kappa_, 1.0 / kappa_);
    }

    double standard_deviation() const override {
        double mu = mean();
        return std::sqrt(lambda_ * lambda_ * sf_wb::function(1.0 + 2.0 / kappa_) - mu * mu);
    }

    double skewness() const override {
        double mu    = mean();
        double sigma = standard_deviation();
        return (sf_wb::function(1.0 + 3.0 / kappa_) * lambda_ * lambda_ * lambda_
                - 3.0 * mu * sigma * sigma - mu * mu * mu)
               / (sigma * sigma * sigma);
    }

    double kurtosis() const override {
        double g1 = sf_wb::function(1.0 + 1.0 / kappa_);
        double g2 = sf_wb::function(1.0 + 2.0 / kappa_);
        double g3 = sf_wb::function(1.0 + 3.0 / kappa_);
        double g4 = sf_wb::function(1.0 + 4.0 / kappa_);
        double num = -6.0 * g1 * g1 * g1 * g1
                     + 12.0 * g2 * g1 * g1
                     - 3.0 * g2 * g2
                     - 4.0 * g1 * g3
                     + g4;
        double den = (g2 - g1 * g1) * (g2 - g1 * g1);
        return 3.0 + num / den;
    }

    double minimum() const override { return 0.0; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override { return std::exp(log_pdf(x)); }

    double log_pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("Weibull: invalid parameters");
        if (x < minimum() || x == kInf) return -kInf;
        if (x == 0.0)
            return kappa_ == 1.0 ? -std::log(lambda_)
                                 : kappa_ < 1.0 ? kInf : -kInf;
        const double logarithm = log_standardized_value(x);
        const double power = std::exp(kappa_ * logarithm);
        if (power == kInf) return -kInf;
        const double value = std::log(kappa_) - std::log(lambda_) +
                             (kappa_ - 1.0) * logarithm - power;
        return std::isnan(value) ? -kInf : value;
    }

    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("Weibull: invalid parameters");
        if (x <= minimum()) return 0.0;
        return -std::expm1(-std::exp(kappa_ * log_standardized_value(x)));
    }

    double log_cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("Weibull: invalid parameters");
        if (x <= 0.0) return -kInf;
        const double logarithm = kappa_ * log_standardized_value(x);
        const double power = std::exp(logarithm);
        return power == 0.0 ? logarithm : distribution_numerics::log1m_exp(-power);
    }

    double ccdf(double x) const override { return std::exp(log_ccdf(x)); }

    double log_ccdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("Weibull: invalid parameters");
        return x <= 0.0 ? 0.0 : -std::exp(kappa_ * log_standardized_value(x));
    }

    double inverse_cdf(double probability) const override {
        if (!(probability >= 0.0 && probability <= 1.0))
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("Weibull: invalid parameters");
        if (kappa_ == 1.0) return -lambda_ * std::log1p(-probability);
        const double logarithm = std::log(-std::log1p(-probability)) / kappa_;
        const double unit_quantile = std::exp(logarithm);
        return unit_quantile < 1e-200 || std::isinf(unit_quantile)
                   ? std::exp(std::log(lambda_) + logarithm)
                   : lambda_ * unit_quantile;
    }

    // --- Parameter display names (X1; C# Weibull.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Scale (\xCE\xBB)", "Shape (\xCE\xBA)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBB", "\xCE\xBA"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Weibull>(lambda_, kappa_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::invalid_argument("Weibull: only MaximumLikelihood estimation is supported");
        }
    }

    // GetParameterConstraints: uses SolveMLE for initial values
    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        distribution_numerics::validate_sample(sample, 2);
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
        auto initials = legacy_constraint_solve_mle(sample);
        std::vector<double> lowers(2), uppers(2);
        lowers[0] = kDoubleMachineEpsilon;
        uppers[0] = std::pow(10.0, std::ceil(std::log10(initials[0]) + 1.0));
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(initials[1]) + 1.0));
        return {initials, lowers, uppers};
    }

    distribution_numerics::Constraints robust_parameter_constraints(
        const std::vector<double>& sample) const {
        distribution_numerics::validate_sample(sample, 2, true);
        auto initials = solve_mle(sample);
        std::vector<double> lowers(2), uppers(2);
        distribution_numerics::positive_parameter_bounds(
            initials[0], lowers[0], uppers[0]);
        distribution_numerics::positive_parameter_bounds(
            initials[1], lowers[1], uppers[1]);
        return {initials, lowers, uppers};
    }

   public:
    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            Weibull w;
            w.set_parameters(x[0], x[1]);
            return w.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 2, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

    // SolveMLE: iterative closed-form Weibull MLE
    // (Qiao & Tsokos 1994 / Math.NET)
    std::vector<double> legacy_constraint_solve_mle(
        const std::vector<double>& samples) const {
        double n = static_cast<double>(samples.size());
        if (n <= 1.0)
            throw std::invalid_argument("Weibull::solve_mle: need more than 1 data point");

        double s1 = 0.0, s2 = 0.0, s3 = 0.0;
        double previous_c = static_cast<double>(std::numeric_limits<int>::min());
        double c = 10.0;  // initial shape
        double b = 0.0;   // scale

        while (std::fabs(c - previous_c) >= 0.0001) {
            s1 = 0.0; s2 = 0.0; s3 = 0.0;
            for (double x : samples) {
                if (x > 0.0) {
                    s1 += std::log(x);
                    s2 += std::pow(x, c);
                    s3 += std::pow(x, c) * std::log(x);
                }
            }
            double q_of_c = n * s2 / (n * s3 - s1 * s2);
            previous_c = c;
            c = (c + q_of_c) / 2.0;
        }

        b = 0.0;
        for (double x : samples) {
            if (x > 0.0) b += std::pow(x, c);
        }
        b = std::pow(b / n, 1.0 / c);

        return {b, c};
    }

    std::vector<double> solve_mle(const std::vector<double>& samples) const {
        distribution_numerics::validate_sample(samples, 2, true);
        const double n = static_cast<double>(samples.size());
        const double scale = distribution_numerics::initialization_scale(samples);
        std::vector<double> log_ratios(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double ratio = samples[i] / scale;
            log_ratios[i] = ratio > 0.0 ? std::log(ratio)
                                        : std::log(samples[i]) - std::log(scale);
        }
        double previous_c = static_cast<double>(std::numeric_limits<int>::min());
        double c = 10.0;
        while (std::fabs(c - previous_c) >= 0.0001) {
            double s1 = 0.0;
            double s2 = 0.0;
            double s3 = 0.0;
            for (double logarithm : log_ratios) {
                const double weight = std::exp(c * logarithm);
                s1 += logarithm;
                s2 += weight;
                s3 += weight * logarithm;
            }
            const double q_of_c = n * s2 / (n * s3 - s1 * s2);
            previous_c = c;
            c = (c + q_of_c) / 2.0;
            if (!(c > 0.0) || !std::isfinite(c))
                throw std::runtime_error(
                    "Weibull initialization did not produce a finite positive shape");
        }
        double b = 0.0;
        for (double logarithm : log_ratios) b += std::exp(c * logarithm);
        b = scale * std::pow(b / n, 1.0 / c);
        return {b, c};
    }

   private:
    double log_standardized_value(double x) const {
        const double ratio = x / lambda_;
        return ratio > 0.0 && !std::isinf(ratio) ? std::log(ratio)
                                                 : std::log(x) - std::log(lambda_);
    }

    static bool validate(double scale, double shape) {
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        if (std::isnan(shape) || std::isinf(shape) || shape <= 0.0) return false;
        return true;
    }

    double lambda_ = 10.0;
    double kappa_  = 2.0;
};

}  // namespace corehydro::numerics::distributions
