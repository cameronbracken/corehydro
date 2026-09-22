// ported from: Numerics/Distributions/Univariate/LogNormal.cs @ 7e8e8d1
//
// The Log-Normal distribution (base 10) with location µ (mean of log) and scale σ (std dev of log).
// Logic mirrors the C# source method-for-method. The base is fixed at 10 (the C# default);
// IBootstrappable, IStandardError, and the Monte Carlo confidence-interval helper are not ported.
// B4 adds ParametersFromMoments/MomentsFromParameters for the Bulletin 17C GMM track.
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
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class LogNormal : public UnivariateDistributionBase,
                  public IEstimation,
                  public ILinearMomentEstimation,
                  public IMaximumLikelihoodEstimation {
   public:
    // Default constructor: mu=3, sigma=0.5, base=10 (mirrors C# default)
    LogNormal() { set_parameters(3.0, 0.5); }
    LogNormal(double mean_of_log, double std_dev_of_log) {
        set_parameters(mean_of_log, std_dev_of_log);
    }

    double mu() const { return mu_; }
    double sigma() const { return sigma_; }

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
        double lnB = std::log(kBase);
        return std::exp((mu_ + 0.5 * sigma_ * sigma_ * lnB) * lnB);
    }

    double median() const override { return inverse_cdf(0.5); }

    // Mode = exp(mu / K) = exp(mu * ln(base)) = base^mu
    double mode() const override {
        return std::exp(mu_ / k());
    }

    // StandardDeviation = sqrt(exp((2*mu + a)*ln(base)) * (exp(a*ln(base)) - 1))
    // where a = sigma^2 * ln(base)
    double standard_deviation() const override {
        double lnB = std::log(kBase);
        double a = sigma_ * sigma_ * lnB;
        double log_pre = (2.0 * mu_ + a) * lnB;
        double exp_a = std::exp(a * lnB);
        double variance = std::exp(log_pre) * (exp_a - 1.0);
        return std::sqrt(variance);
    }

    double skewness() const override {
        double lnB = std::log(kBase);
        double a = sigma_ * sigma_ * lnB;
        double mu1 = (mu_ + 0.5 * a) * lnB;
        double mu2 = (2.0 * mu_ + 2.0 * a) * lnB;
        double mu3 = (3.0 * mu_ + 4.5 * a) * lnB;
        double m1 = std::exp(mu1);
        double m2 = std::exp(mu2);
        double m3 = std::exp(mu3);
        double third_cm = m3 - 3.0 * m2 * m1 + 2.0 * m1 * m1 * m1;
        double sd = standard_deviation();
        return third_cm / (sd * sd * sd);
    }

    double kurtosis() const override {
        double lnB = std::log(kBase);
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
        const double log_base = std::log(kBase);
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
                             std::log(x) / std::log(kBase), mu_, sigma_));
    }

    double ccdf(double x) const override { return std::exp(log_ccdf(x)); }

    double log_ccdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("LogNormal: invalid parameters");
        return x <= 0.0
                   ? 0.0
                   : distribution_numerics::normal_log_survival(
                         distribution_numerics::standardize(
                             std::log(x) / std::log(kBase), mu_, sigma_));
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
        return std::make_unique<LogNormal>(mu_, sigma_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            set_parameters(indirect_mom(sample));
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(indirect_lmom(sample)));
        } else {
            set_parameters(mle(sample));
        }
    }

    // IndirectMethodOfMoments: compute product moments of log-transformed data
    std::vector<double> indirect_mom(const std::vector<double>& sample) const {
        std::vector<double> log_sample;
        log_sample.reserve(sample.size());
        double lnB = std::log(kBase);
        for (double v : sample) {
            double x = (v > 0.0) ? v : 0.1;
            log_sample.push_back(std::log(x) / lnB);
        }
        return data::product_moments(log_sample);  // returns {mean, sd, skew, kurtosis}
    }

    // IndirectMethodOfLinearMoments: compute linear moments of log-transformed data
    std::vector<double> indirect_lmom(const std::vector<double>& sample) const {
        std::vector<double> log_sample;
        log_sample.reserve(sample.size());
        double lnB = std::log(kBase);
        for (double v : sample) {
            double x = (v > 0.0) ? v : 0.1;
            log_sample.push_back(std::log(x) / lnB);
        }
        return data::linear_moments(log_sample);  // returns {L1, L2, T3, T4}
    }

    // ParametersFromMoments (C# LogNormal.cs:408): real-space {mean, sd} -> base-10
    // log-space {mu, sigma}. C# Math.Log(x, Base) = ln(x)/ln(Base) with Base = 10.
    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        double mean = moments[0];
        double standard_deviation = moments[1];
        double lnB = std::log(kBase);
        double variance = standard_deviation * standard_deviation;
        double mu = std::log(mean * mean / std::sqrt(variance + mean * mean)) / lnB;
        double sigma = std::sqrt(std::log(1.0 + variance / (mean * mean)) / lnB);
        return {mu, sigma};
    }

    // MomentsFromParameters (C# LogNormal.cs:419): {Mean, StandardDeviation, Skewness,
    // Kurtosis} of a LogNormal built from the (base-10 log-space) parameters.
    std::vector<double> moments_from_parameters(const std::vector<double>& parameters) const {
        LogNormal dist;
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
        auto mom = indirect_mom(sample);
        initials = {mom[0], mom[1]};
        lowers.resize(2);
        uppers.resize(2);
        // Bounds of mu
        double real = std::exp(initials[0] / k());
        lowers[0] = kDoubleMachineEpsilon;
        double up0 = std::ceil(std::log(std::pow(10.0, std::ceil(std::log10(real) + 1.0))) / std::log(kBase));
        uppers[0] = std::isnan(up0) ? 5.0 : up0;
        // Bounds of sigma
        double real2 = std::exp(initials[1] / k());
        lowers[1] = kDoubleMachineEpsilon;
        double up1 = std::ceil(std::log(std::pow(10.0, std::ceil(std::log10(real2) + 1.0))) / std::log(kBase));
        uppers[1] = std::isnan(up1) ? 4.0 : up1;
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            LogNormal ln;
            ln.set_parameters(x[0], x[1]);
            return ln.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 2, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

   private:
    static bool validate(double mu, double sigma) {
        if (std::isnan(mu) || std::isinf(mu)) return false;
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) return false;
        return true;
    }

    // K = 1/ln(base) — the log correction factor
    double k() const { return 1.0 / std::log(kBase); }

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
    static constexpr double kBase = 10.0;  // base of logarithm (mirrors C# default)
};

}  // namespace corehydro::numerics::distributions
