// ported from: Numerics/Distributions/Univariate/LnNormal.cs @ 7e8e8d1
//
// The Ln-Normal (Galton) distribution. Parameters are the real-space mean and standard deviation;
// internally the distribution stores the natural-log-space parameters mu_ and sigma_.
// Constructor SetParameters(mean, sd) calls DirectMethodOfMoments to derive mu and sigma.
// GetParameters returns [Mean, StandardDeviation] (real-space, not log-space).
// Logic mirrors the C# source method-for-method. IBootstrappable and IStandardError are
// not ported (desktop / advanced analysis concerns). B4 adds ParametersFromMoments/
// MomentsFromParameters and the ConditionalMoments override for the Bulletin 17C GMM track.
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

class LnNormal : public UnivariateDistributionBase,
                 public IEstimation,
                 public ILinearMomentEstimation,
                 public IMaximumLikelihoodEstimation {
   public:
    // Default constructor: real-space mean=10, sd=10 (mirrors C# default)
    LnNormal() { set_parameters(10.0, 10.0); }
    LnNormal(double mean, double standard_deviation) { set_parameters(mean, standard_deviation); }

    double mu() const { return mu_; }
    double sigma() const { return sigma_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::LnNormal;
    }
    int number_of_parameters() const override { return 2; }
    // Returns real-space [Mean, StandardDeviation], mirroring C# GetParameters
    std::vector<double> get_parameters() const override {
        return {mean(), standard_deviation()};
    }

    // SetParameters(mean, sd) — real-space inputs, converted to log-space via DirectMethodOfMoments
    void set_parameters(double mean_val, double sd_val) {
        auto parms = direct_mom(mean_val, sd_val);
        // Validate against original real-space values
        parameters_valid_ = validate(mean_val, sd_val);
        if (!std::isnan(parms[0]) && !std::isnan(parms[1])) {
            mu_ = parms[0];
            sigma_ = parms[1] < 1e-16 ? 1e-16 : parms[1];
            physical_mean_ = mean_val;
            physical_standard_deviation_ = sd_val;
            has_physical_moments_ = true;
        } else {
            // Keep old values but mark invalid (matches C# behavior)
            parameters_valid_ = false;
        }
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override {
        return has_physical_moments_ ? physical_mean_
                                     : std::exp(mu_ + sigma_ * sigma_ / 2.0);
    }
    double median() const override { return std::exp(mu_); }
    double mode() const override { return std::exp(mu_ - sigma_ * sigma_); }
    double standard_deviation() const override {
        if (has_physical_moments_) return physical_standard_deviation_;
        const double variance = sigma_ * sigma_;
        const double log_excess = variance > 0.5
                                      ? variance + std::log1p(-std::exp(-variance))
                                      : std::log(std::expm1(variance));
        return std::exp(mu_ + 0.5 * variance + 0.5 * log_excess);
    }
    double skewness() const override {
        double s2 = sigma_ * sigma_;
        return (std::exp(s2) + 2.0) * std::sqrt(std::expm1(s2));
    }
    double kurtosis() const override {
        double s2 = sigma_ * sigma_;
        return 3.0 + std::expm1(4.0 * s2) + 2.0 * std::expm1(3.0 * s2) +
               3.0 * std::expm1(2.0 * s2);
    }
    double minimum() const override { return 0.0; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    // PDF = exp(-0.5*((ln(x)-mu)/sigma)^2) / (sqrt(2pi)*sigma*x)
    double pdf(double x) const override { return std::exp(log_pdf(x)); }

    double log_pdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("LnNormal: invalid parameters");
        if (x <= 0.0 || x == kInf) return -kInf;
        const double log_x = std::log(x);
        const double z = distribution_numerics::standardize(log_x, mu_, sigma_);
        return -0.5 * z * z - std::log(sigma_) - kLogSqrt2PI - log_x;
    }

    // CDF = 0.5*(1 + erf((ln(x)-mu)/(sigma*sqrt(2))))
    double cdf(double x) const override { return std::exp(log_cdf(x)); }

    double log_cdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("LnNormal: invalid parameters");
        return x <= 0.0 ? -kInf
                        : distribution_numerics::normal_log_cdf(
                              distribution_numerics::standardize(std::log(x), mu_, sigma_));
    }

    double ccdf(double x) const override { return std::exp(log_ccdf(x)); }

    double log_ccdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("LnNormal: invalid parameters");
        return x <= 0.0 ? 0.0
                        : distribution_numerics::normal_log_survival(
                              distribution_numerics::standardize(std::log(x), mu_, sigma_));
    }

    // InverseCDF = exp(mu - sigma*sqrt(2)*inverse_erfc(2p))
    double inverse_cdf(double probability) const override {
        if (std::isnan(probability) || probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return std::exp(mu_ + sigma_ * Normal::standard_z(probability));
    }

    // --- Parameter display names (X1; C# LnNormal.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Mean (\xC2\xB5)", "Std Dev (\xCF\x83)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xC2\xB5", "\xCF\x83"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        // Clone via internal log-space params (skip DirectMethodOfMoments roundtrip)
        auto* c = new LnNormal();
        c->mu_ = mu_;
        c->sigma_ = sigma_;
        c->parameters_valid_ = parameters_valid_;
        c->has_physical_moments_ = has_physical_moments_;
        c->physical_mean_ = physical_mean_;
        c->physical_standard_deviation_ = physical_standard_deviation_;
        return std::unique_ptr<LnNormal>(c);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        distribution_numerics::validate_sample(sample, 4, true);
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            auto parms = indirect_mom(sample);
            mu_ = parms[0];
            sigma_ = parms[1];
            parameters_valid_ = validate_log_params(mu_, sigma_);
            has_physical_moments_ = false;
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            auto parms = parameters_from_linear_moments(indirect_lmom(sample));
            mu_ = parms[0];
            sigma_ = parms[1];
            parameters_valid_ = validate_log_params(mu_, sigma_);
            has_physical_moments_ = false;
        } else {
            auto parms = mle(sample);
            set_parameters(parms);
        }
    }

    // IndirectMethodOfMoments: log-transform the sample, compute product moments
    static std::vector<double> indirect_mom(const std::vector<double>& sample) {
        std::vector<double> log_sample;
        log_sample.reserve(sample.size());
        for (double v : sample) {
            log_sample.push_back(std::log(v > 0.0 ? v : 0.1));
        }
        return data::product_moments(log_sample);
    }

    // IndirectMethodOfLinearMoments: log-transform the sample, compute linear moments
    std::vector<double> indirect_lmom(const std::vector<double>& sample) const {
        std::vector<double> log_sample;
        log_sample.reserve(sample.size());
        for (double v : sample) {
            log_sample.push_back(std::log(v > 0.0 ? v : 0.1));
        }
        return data::linear_moments(log_sample);
    }

    // DirectMethodOfMoments: real-space mean, sd → log-space mu, sigma
    static std::vector<double> direct_mom(double mean_val, double sd_val) {
        if (!(mean_val > 0.0) || !(sd_val > 0.0) || !std::isfinite(mean_val) ||
            !std::isfinite(sd_val))
            return {kNaN, kNaN};
        const double log_ratio = std::log(sd_val) - std::log(mean_val);
        const double variance = log_ratio > 0.0
                                    ? 2.0 * log_ratio + std::log1p(std::exp(-2.0 * log_ratio))
                                    : std::log1p(std::exp(2.0 * log_ratio));
        return {std::log(mean_val) - 0.5 * variance, std::sqrt(variance)};
    }

    // ParametersFromMoments (C# LnNormal.cs:356): real-space {mean, sd} -> log-space
    // {mu, sigma}. The C# body duplicates DirectMethodOfMoments verbatim (guards
    // included), so this port delegates to direct_mom.
    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        return direct_mom(moments[0], moments[1]);
    }

    // MomentsFromParameters (C# LnNormal.cs:370): {Mean, StandardDeviation, Skewness,
    // Kurtosis} of an LnNormal built from the (real-space) parameters.
    std::vector<double> moments_from_parameters(const std::vector<double>& parameters) const {
        LnNormal dist;
        dist.set_parameters(parameters);
        double m1 = dist.mean();
        double m2 = dist.standard_deviation();
        double m3 = dist.skewness();
        double m4 = dist.kurtosis();
        return {m1, m2, m3, m4};
    }

    // ParametersFromLinearMoments: mu = L1, sigma = L2*sqrt(pi)  (log-space)
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

   private:
    distribution_numerics::Constraints legacy_parameter_constraints(
        const std::vector<double>& sample) const {
        auto moments = data::product_moments(sample);
        std::vector<double> initials = {moments[0], moments[1]};
        std::vector<double> lowers(2), uppers(2);
        lowers[0] = kDoubleMachineEpsilon;
        uppers[0] = std::pow(10.0, std::ceil(std::log10(initials[0]) + 1.0));
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(initials[1]) + 1.0));
        return {initials, lowers, uppers};
    }

    distribution_numerics::Constraints robust_parameter_constraints(
        const std::vector<double>& sample) const {
        distribution_numerics::validate_sample(sample, 4, true);
        Normal normal;
        auto constraints = normal.robust_parameter_constraints(sample);
        auto& initials = std::get<0>(constraints);
        auto& lowers = std::get<1>(constraints);
        lowers[0] = std::min(kDoubleMachineEpsilon, initials[0] / 10.0);
        return constraints;
    }

   public:
    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            LnNormal ln;
            ln.set_parameters(x[0], x[1]);
            return ln.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 2, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

    // ConditionalMoments override (C# LnNormal.cs:560): closed-form truncated-lognormal
    // raw moments via shifted standard-normal CDF differences, converted to central
    // moments about the UNCONDITIONAL mean muX = exp(mu + sigma^2/2).
    std::vector<double> conditional_moments(double a, double b) const override {
        if (a >= b) return {kNaN, kNaN, kNaN, kNaN};

        // Log-space Normal parameters: Y = ln X ~ N(mu, sigma^2)
        double mu = mu_;
        double sigma = sigma_;
        if (!(sigma > 0.0)) return {kNaN, kNaN, kNaN, kNaN};

        // Map bounds to log-space [A, B], allowing a <= 0 (A = -inf) and b = +inf
        double A = (a > 0.0) ? std::log(a) : -kInf;
        double B = std::isinf(b) && b > 0.0 ? kInf : (b > 0.0 ? std::log(b) : -kInf);

        // Standardized limits for Y
        double alpha = (A - mu) / sigma;  // can be -inf
        double beta = (B - mu) / sigma;   // can be +inf

        // Standard normal CDF helper (consistent with the Normal code)
        auto Phi = [](double x) { return 0.5 * (1.0 + std::erf(x / kSqrt2)); };

        double PhiA = (std::isinf(alpha) && alpha < 0.0) ? 0.0 : Phi(alpha);
        double PhiB = (std::isinf(beta) && beta > 0.0) ? 1.0 : Phi(beta);

        double Z = PhiB - PhiA;  // normalization (P(a < X < b))
        if (Z <= 1e-15) return {kNaN, kNaN, kNaN, kNaN};

        // Raw truncated moments: E[X^k | a < X < b] for k = 1..4
        double Eraw[5] = {0.0, 0.0, 0.0, 0.0, 0.0};  // filled for 1..4

        for (int k = 1; k <= 4; k++) {
            double ks = k * sigma;
            // shifted limits: Phi(beta - k*sigma) - Phi(alpha - k*sigma)
            double PhiBk = (std::isinf(beta) && beta > 0.0) ? 1.0 : Phi(beta - ks);
            double PhiAk = (std::isinf(alpha) && alpha < 0.0) ? 0.0 : Phi(alpha - ks);

            double scale = std::exp(k * mu + 0.5 * k * k * sigma * sigma);
            Eraw[k] = scale * (PhiBk - PhiAk) / Z;
        }

        // Unconditional mean of X (about which we take central moments)
        double muX = std::exp(mu + 0.5 * sigma * sigma);

        // Central moments about mu_X (unconditional)
        double m1 = Eraw[1];
        double m2 = Eraw[2] - 2.0 * muX * Eraw[1] + muX * muX;
        double m3 = Eraw[3] - 3.0 * muX * Eraw[2] + 3.0 * muX * muX * Eraw[1] - muX * muX * muX;
        double m4 = Eraw[4]
                  - 4.0 * muX * Eraw[3]
                  + 6.0 * muX * muX * Eraw[2]
                  - 4.0 * muX * muX * muX * Eraw[1]
                  + muX * muX * muX * muX;

        return {m1, m2, m3, m4};
    }

   private:
    static bool validate(double mean_val, double sd_val) {
        if (std::isnan(mean_val) || std::isinf(mean_val)) return false;
        if (std::isnan(sd_val) || std::isinf(sd_val) || sd_val <= 0.0) return false;
        return true;
    }

    static bool validate_log_params(double mu, double sigma) {
        if (std::isnan(mu) || std::isinf(mu)) return false;
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) return false;
        return true;
    }

    // Wichura AS241 standard-normal quantile (used by inverse_cdf)
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

    double mu_ = 0.0;
    double sigma_ = 1.0;
    bool has_physical_moments_ = false;
    double physical_mean_ = 0.0;
    double physical_standard_deviation_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
