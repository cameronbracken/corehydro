// ported from: Numerics/Distributions/Univariate/LogNormal.cs @ 7e8e8d1
//
// The Log-Normal distribution with configurable logarithm base, location, and scale.
// B4 adds ParametersFromMoments/MomentsFromParameters for the Bulletin 17C GMM track.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_bootstrappable.hpp"
#include "corehydro/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_standard_error.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class LogNormal : public UnivariateDistributionBase,
                  public IEstimation,
                  public ILinearMomentEstimation,
                  public IMaximumLikelihoodEstimation,
                  public IStandardError,
                  public IBootstrappable {
   public:
    // Default constructor: mu=3, sigma=0.5, base=10 (mirrors C# default)
    LogNormal() { set_parameters(3.0, 0.5); }
    LogNormal(double mean_of_log, double std_dev_of_log) {
        set_parameters(mean_of_log, std_dev_of_log);
    }

    double mu() const { return mu_; }
    double sigma() const { return sigma_; }
    double base() const { return base_; }
    void set_base(double value) {
        if (!(value > 1.0) || !std::isfinite(value))
            throw std::out_of_range("The logarithm base must be finite and greater than one.");
        base_ = value;
    }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::LogNormal;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {mu_, sigma_}; }

    void set_parameters(double mean_of_log, double std_dev_of_log) {
        if (std_dev_of_log < 1E-16 && std::signbit(std_dev_of_log) == false)
            std_dev_of_log = 1E-16;
        mu_ = mean_of_log;
        sigma_ = std_dev_of_log;
        parameters_valid_ = validate(mean_of_log, std_dev_of_log);
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1]);
    }

    // --- Moments / support ---
    // Mean = exp((mu + 0.5*sigma^2*ln(base)) * ln(base))
    double mean() const override {
        double lnB = std::log(base_);
        return std::exp((mu_ + 0.5 * sigma_ * sigma_ * lnB) * lnB);
    }

    double median() const override { return inverse_cdf(0.5); }

    double mode() const override {
        const double log_base = std::log(base_);
        return std::exp(mu_ * log_base - std::pow(sigma_ * log_base, 2.0));
    }

    // StandardDeviation = sqrt(exp((2*mu + a)*ln(base)) * (exp(a*ln(base)) - 1))
    // where a = sigma^2 * ln(base)
    double standard_deviation() const override {
        double lnB = std::log(base_);
        const double variance = std::pow(sigma_ * lnB, 2.0);
        const double log_excess = variance > 0.5
                                      ? variance + std::log1p(-std::exp(-variance))
                                      : std::log(std::expm1(variance));
        return std::exp(mu_ * lnB + 0.5 * variance + 0.5 * log_excess);
    }

    double skewness() const override {
        double lnB = std::log(base_);
        const double variance = std::pow(sigma_ * lnB, 2.0);
        return (std::exp(variance) + 2.0) * std::sqrt(std::expm1(variance));
    }

    double kurtosis() const override {
        double lnB = std::log(base_);
        double variance = sigma_ * sigma_ * lnB * lnB;
        return 3.0 + std::expm1(4.0 * variance) + 2.0 * std::expm1(3.0 * variance) +
               3.0 * std::expm1(2.0 * variance);
    }

    double minimum() const override { return 0.0; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    // PDF = exp(-0.5*d^2) / (sqrt(2pi)*sigma) * (K/x),  d = (log_base(x) - mu)/sigma
    double pdf(double x) const override { return std::exp(log_pdf(x)); }

    double log_pdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("LogNormal: invalid parameters");
        if (x <= 0.0 || x == kInf) return -kInf;
        const double log_x = std::log(x);
        const double log_base = std::log(base_);
        const double z = distribution_numerics::standardize(log_x / log_base, mu_, sigma_);
        return -0.5 * z * z - std::log(sigma_) - kLogSqrt2PI - std::log(log_base) -
               log_x;
    }

    // CDF = 0.5*(1 + erf((log_base(x) - mu) / (sigma*sqrt(2))))
    double cdf(double x) const override { return std::exp(log_cdf(x)); }

    double log_cdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("LogNormal: invalid parameters");
        return x <= 0.0
                   ? -kInf
                   : distribution_numerics::normal_log_cdf(
                         distribution_numerics::standardize(
                             std::log(x) / std::log(base_), mu_, sigma_));
    }

    double ccdf(double x) const override { return std::exp(log_ccdf(x)); }

    double log_ccdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("LogNormal: invalid parameters");
        return x <= 0.0
                   ? 0.0
                   : distribution_numerics::normal_log_survival(
                         distribution_numerics::standardize(
                             std::log(x) / std::log(base_), mu_, sigma_));
    }

    // InverseCDF = exp((mu - sigma*sqrt(2)*inverse_erfc(2p)) / K)
    // where K = 1/ln(base), so divide by K = multiply by ln(base)
    double inverse_cdf(double probability) const override {
        if (std::isnan(probability) || probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        // inverse_erfc(y) via Wichura: Normal.StandardZ(-0.5*y+1)*sqrt(2)/2
        // For erfc(x)=2p: erfc_arg = 2p
        return std::exp((mu_ + sigma_ * Normal::standard_z(probability)) / k());
    }

    // --- Parameter display names (X1; C# LogNormal.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Mean (of log) (\xC2\xB5)", "Std Dev (of log) (\xCF\x83)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xC2\xB5", "\xCF\x83"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        auto clone = std::make_unique<LogNormal>(mu_, sigma_);
        clone->set_base(base_);
        return clone;
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        distribution_numerics::validate_sample(sample, 4, true);
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            set_parameters(indirect_mom(sample));
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(indirect_lmom(sample)));
        } else {
            set_parameters(mle(sample));
        }
    }

    std::unique_ptr<UnivariateDistributionBase> bootstrap(ParameterEstimationMethod method,
                                                          int sample_size,
                                                          int seed = -1) const override {
        auto distribution = std::make_unique<LogNormal>(mu_, sigma_);
        distribution->set_base(base_);
        const auto sample = distribution->generate_random_values(sample_size, seed);
        distribution->estimate(sample, method);
        if (!distribution->parameters_valid())
            throw std::runtime_error("Bootstrapped distribution parameters are invalid.");
        return distribution;
    }

    // IndirectMethodOfMoments: compute product moments of log-transformed data
    std::vector<double> indirect_mom(const std::vector<double>& sample) const {
        std::vector<double> log_sample;
        log_sample.reserve(sample.size());
        double lnB = std::log(base_);
        distribution_numerics::validate_sample(sample, 4, true);
        for (double v : sample) log_sample.push_back(std::log(v) / lnB);
        return data::product_moments(log_sample);  // returns {mean, sd, skew, kurtosis}
    }

    // IndirectMethodOfLinearMoments: compute linear moments of log-transformed data
    std::vector<double> indirect_lmom(const std::vector<double>& sample) const {
        std::vector<double> log_sample;
        log_sample.reserve(sample.size());
        double lnB = std::log(base_);
        distribution_numerics::validate_sample(sample, 4, true);
        for (double v : sample) log_sample.push_back(std::log(v) / lnB);
        return data::linear_moments(log_sample);  // returns {L1, L2, T3, T4}
    }

    // ParametersFromMoments: real-space {mean, sd} -> configured-base
    // log-space {mu, sigma}. C# Math.Log(x, Base) = ln(x)/ln(Base) with Base = 10.
    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        const auto natural = LnNormal::direct_mom(moments[0], moments[1]);
        const double lnB = std::log(base_);
        return {natural[0] / lnB, natural[1] / lnB};
    }

    // MomentsFromParameters (C# LogNormal.cs:419): {Mean, StandardDeviation, Skewness,
    // Kurtosis} of a LogNormal built from the configured-base log-space parameters.
    std::vector<double> moments_from_parameters(const std::vector<double>& parameters) const {
        LogNormal dist;
        dist.set_base(base_);
        dist.set_parameters(parameters);
        double m1 = dist.mean();
        double m2 = dist.standard_deviation();
        double m3 = dist.skewness();
        double m4 = dist.kurtosis();
        return {m1, m2, m3, m4};
    }

    // ParametersFromLinearMoments: mu = L1, sigma = L2 * sqrt(pi)
    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        double mu = moments[0];
        double sigma = moments[1] * std::sqrt(kPi);
        return {mu, sigma};
    }

    // LinearMomentsFromParameters: L1=mu, L2=sigma*pi^{-0.5}, T3=0, T4=30*pi^{-1}*atan(sqrt(2))-9
    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double L1 = parameters[0];
        double L2 = parameters[1] * std::pow(kPi, -0.5);
        double T3 = 0.0;
        double T4 = 30.0 * std::pow(kPi, -1.0) * std::atan(kSqrt2) - 9.0;
        return {L1, L2, T3, T4};
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

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample, this](const std::vector<double>& x) {
            LogNormal ln;
            ln.set_base(base_);
            ln.set_parameters(x[0], x[1]);
            return ln.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 2, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

    math::linalg::Matrix2D parameter_covariance(
        int sample_size, ParameterEstimationMethod method) const override {
        distribution_numerics::validate_sample_size(sample_size);
        if (method != ParameterEstimationMethod::MethodOfMoments &&
            method != ParameterEstimationMethod::MaximumLikelihood)
            throw std::logic_error(
                "LogNormal covariance is implemented only for moments and maximum likelihood");
        if (!parameters_valid_) throw std::out_of_range("LogNormal: invalid parameters");
        const double scaled = sigma_ / std::sqrt(static_cast<double>(sample_size));
        const double variance = scaled * scaled;
        return {{variance, 0.0}, {0.0, variance / 2.0}};
    }

    double quantile_variance(double probability, int sample_size,
                             ParameterEstimationMethod method) const override {
        distribution_numerics::validate_probability(probability);
        distribution_numerics::validate_sample_size(sample_size);
        if (method != ParameterEstimationMethod::MethodOfMoments &&
            method != ParameterEstimationMethod::MaximumLikelihood)
            throw std::logic_error(
                "LogNormal quantile variance is implemented only for moments and maximum likelihood");
        if (!parameters_valid_) throw std::out_of_range("LogNormal: invalid parameters");
        const double z = Normal::standard_z(probability);
        const double log_base = std::log(base_);
        const double log_quantile = (mu_ + sigma_ * z) * log_base;
        const double log_standard_error =
            log_quantile + std::log(log_base) + std::log(sigma_) -
            0.5 * std::log(static_cast<double>(sample_size));
        return std::exp(2.0 * log_standard_error + std::log1p(0.5 * z * z));
    }

    std::vector<double> quantile_gradient(double probability) const override {
        distribution_numerics::validate_probability(probability);
        if (!parameters_valid_) throw std::out_of_range("LogNormal: invalid parameters");
        const double z = Normal::standard_z(probability);
        const double factor = inverse_cdf(probability) * std::log(base_);
        return {factor, factor * z};
    }

    math::linalg::Matrix2D quantile_jacobian(
        const std::vector<double>& probabilities, double& determinant) const override {
        return distribution_numerics::quantile_jacobian(
            *this, number_of_parameters(), probabilities, determinant);
    }

   private:
    distribution_numerics::Constraints legacy_parameter_constraints(
        const std::vector<double>& sample) const {
        std::vector<double> transformed(sample.size());
        for (std::size_t i = 0; i < sample.size(); ++i)
            transformed[i] = std::log(sample[i] > 0.0 ? sample[i] : 0.1) / std::log(base_);
        const auto moments = data::product_moments(transformed);
        std::vector<double> initials = {moments[0], moments[1]};
        std::vector<double> lowers(2), uppers(2);
        const double real_location = std::exp(initials[0] / k());
        if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
        lowers[0] = std::floor(
            std::log(std::pow(10.0, std::floor(std::log10(real_location)) - 1.0)) /
            std::log(base_));
        uppers[0] = std::ceil(
            std::log(std::pow(10.0, std::ceil(std::log10(real_location)) + 1.0)) /
            std::log(base_));
        const double real_scale = std::exp(initials[1] / k());
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::ceil(
            std::log(std::pow(10.0, std::ceil(std::log10(real_scale) + 1.0))) /
            std::log(base_));
        if (std::isnan(uppers[1])) uppers[1] = 4.0;
        return {initials, lowers, uppers};
    }

    distribution_numerics::Constraints robust_parameter_constraints(
        const std::vector<double>& sample) const {
        distribution_numerics::validate_sample(sample, 4, true);
        std::vector<double> transformed(sample.size());
        for (std::size_t i = 0; i < sample.size(); ++i)
            transformed[i] = std::log(sample[i]) / std::log(base_);
        Normal normal;
        return normal.robust_parameter_constraints(transformed);
    }

    static bool validate(double mu, double sigma) {
        if (std::isnan(mu) || std::isinf(mu)) return false;
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) return false;
        return true;
    }

    // K = 1/ln(base) — the log correction factor
    double k() const { return 1.0 / std::log(base_); }

    // Wichura AS241 standard-normal quantile (used by inverse_cdf / inverse_erfc)
    static double wichura_z(double p) {
        static const double a[8] = {3.3871328727963666080,    1.3314166789178437745e+2,
                                     1.9715909503065514427e+3,  1.3731693765509461125e+4,
                                     4.5921953931549871457e+4,  6.7265770927008700853e+4,
                                     3.3430575583588128105e+4,  2.5090809287301226727e+3};
        static const double b[8] = {1.0,                       4.2313330701600911252e+1,
                                     6.8718700749205790830e+2,  5.3941960214247511077e+3,
                                     2.1213794301586595867e+4,  3.9307895800092710610e+4,
                                     2.8729085735721942674e+4,  5.2264952788528545610e+3};
        static const double c[8] = {1.42343711074968357734,    4.63033784615654529590,
                                     5.76949722146069140550,    3.64784832476320460504,
                                     1.27045825245236838258,    2.41780725177450611770e-1,
                                     2.27238449892691845833e-2, 7.74545014278341407640e-4};
        static const double d[8] = {1.0,                       2.05319162663775882187,
                                     1.67638483018380384940,    6.89767334985100004550e-1,
                                     1.48103976427480074590e-1, 1.51986665636164571966e-2,
                                     5.47593808499534494600e-4, 1.05075007164441684324e-9};
        static const double e[8] = {6.65790464350110377720,    5.46378491116411436990,
                                     1.78482653991729133580,    2.96560571828504891230e-1,
                                     2.65321895265761230930e-2, 1.24266094738807843860e-3,
                                     2.71155556874348757815e-5, 2.01033439929228813265e-7};
        static const double f[8] = {1.0,                       5.99832206555887937690e-1,
                                     1.36929880922735805310e-1, 1.48753612908506148525e-2,
                                     7.86869131145613259100e-4, 1.84631831751005468180e-5,
                                     1.42151175831644588870e-7, 2.04426310338993978564e-15};
        if (p <= 0.0) return -kInf;
        if (p >= 1.0) return kInf;
        auto poly = [](const double* coeffs, int n, double x) {
            double v = 0.0;
            for (int i = n - 1; 0 <= i; --i) v = v * x + coeffs[i];
            return v;
        };
        double q = p - 0.5;
        double r, value;
        if (std::fabs(q) <= 0.425) {
            r = 0.180625 - q * q;
            value = q * poly(a, 8, r) / poly(b, 8, r);
        } else {
            r = q < 0.0 ? p : 1.0 - p;
            r = std::sqrt(-std::log(r));
            if (r <= 5.0) {
                r -= 1.6;
                value = poly(c, 8, r) / poly(d, 8, r);
            } else {
                r -= 5.0;
                value = poly(e, 8, r) / poly(f, 8, r);
            }
            if (q < 0.0) value = -value;
        }
        return value;
    }

    double mu_ = 3.0;
    double sigma_ = 0.5;
    double base_ = 10.0;
};

}  // namespace corehydro::numerics::distributions
