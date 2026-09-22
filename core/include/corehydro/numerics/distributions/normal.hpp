// ported from: Numerics/Distributions/Univariate/Normal.cs @ 7e8e8d1
//
// The Normal (Gaussian) distribution, parameters µ (location) and σ (scale).
// CDF uses std::erf (matches the C# instance Normal.CDF(x), itself erf-based -- see the
// precision-fix note below for why the STATIC standard_cdf helper is different); InverseCDF
// ports Wichura's AS241 (r8_normal_01_cdf_inverse) verbatim, the same routine the C# uses.
// Logic mirrors the C# source method-for-method; the WPF confidence-interval helpers
// (Normal/NoncentralT/MonteCarlo) are not ported. B4 adds ParametersFromMoments/
// MomentsFromParameters, QuantileGradient, and the ConditionalMoments override for the
// Bulletin 17C GMM track.
//
// Precision fix (discovered during the v2.1.4 sync's T1 task while adding oracle coverage
// for Probability::joint_probability_hpcm, unrelated to any C# diff): `standard_cdf` used
// to compute Phi(z) = 0.5*(1 + erf(z/sqrt2)) directly. For z <~ -6, `erf(z/sqrt2)` is so
// close to -1 that `1 + erf(...)` suffers catastrophic cancellation and rounds to EXACTLY
// 0.0 in double precision (e.g. standard_cdf(-9) returned 0.0 instead of the true
// ~1.13E-19) -- silently wrong, not merely imprecise. The real C# `Normal.StandardCDF`
// does NOT hit this: it delegates to `MultivariateNormal.MVNPHI`, a Chebyshev-series
// algorithm (Schonfelder 1978) accurate across the whole range, which this port's own
// `MultivariateNormal::mvnphi` already mirrors faithfully for the bivariate-CDF machinery
// -- but header-ordering makes calling it from here impractical (multivariate_normal.hpp
// already depends on this file). Fixed with the standard numerically-stable identity
// Phi(z) = 0.5*erfc(-z/sqrt2) instead: mathematically identical to the erf form (erfc(x) =
// 1 - erf(x)), but `erfc` is specifically designed not to lose precision as its argument
// grows, so it never cancels down to exactly 0/1. Verified to reproduce the real C#
// MVNPHI-based value at z=-9 to ~1E-15 RELATIVE precision (1.1285884059538425E-19 here vs.
// 1.128588405953841E-19 from `oracle_emitter --dump`; the two differ by ~6 ULP, NOT
// bit-for-bit -- C#'s MVNPHI is itself only a ~1E-15-accurate Chebyshev-series
// approximation, per its own doc comment, while this erfc form is the more accurate of the
// two), comfortably within the fixture's 1E-8 relative tolerance. Agrees with the old erf
// formula to ~1E-16 for every ordinary z any existing fixture exercises, so no other oracle
// value moves. See fixtures/special_functions/probability.json's `extreme_probabilities_*`
// cases (Probability.hpcm_conditional_at, index 0) for the regression pin.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_bootstrappable.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class Normal : public UnivariateDistributionBase,
               public IEstimation,
               public ILinearMomentEstimation,
               public IMaximumLikelihoodEstimation,
               public IBootstrappable {
   public:
    Normal() { set_parameters(0.0, 1.0); }
    explicit Normal(double mean) { set_parameters(mean, 1.0); }
    Normal(double mean, double standard_deviation) { set_parameters(mean, standard_deviation); }

    double mu() const { return mu_; }
    double sigma() const { return sigma_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override { return UnivariateDistributionType::Normal; }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {mu_, sigma_}; }

    void set_parameters(double location, double scale) {
        if (scale < 1E-16 && std::signbit(scale) == false) scale = 1E-16;
        mu_ = location;
        sigma_ = scale;
        parameters_valid_ = validate(location, scale);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override { return mu_; }
    double median() const override { return mu_; }
    double mode() const override { return mu_; }
    double standard_deviation() const override { return sigma_; }
    double skewness() const override { return 0.0; }
    double kurtosis() const override { return 3.0; }
    double minimum() const override { return -kInf; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override { return std::exp(log_pdf(x)); }

    double log_pdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Normal: invalid parameters");
        const double z = distribution_numerics::standardize(x, mu_, sigma_);
        return -0.5 * z * z - std::log(sigma_) - kLogSqrt2PI;
    }

    double cdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Normal: invalid parameters");
        return standard_cdf(distribution_numerics::standardize(x, mu_, sigma_));
    }

    double log_cdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Normal: invalid parameters");
        return distribution_numerics::normal_log_cdf(
            distribution_numerics::standardize(x, mu_, sigma_));
    }

    double ccdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Normal: invalid parameters");
        return standard_cdf(-distribution_numerics::standardize(x, mu_, sigma_));
    }

    double log_ccdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Normal: invalid parameters");
        return distribution_numerics::normal_log_survival(
            distribution_numerics::standardize(x, mu_, sigma_));
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return mu_ + sigma_ * standard_z(probability);
    }

    // --- Parameter display names (X1; C# Normal.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Mean (\xC2\xB5)", "Std Dev (\xCF\x83)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xC2\xB5", "\xCF\x83"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Normal>(mu_, sigma_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            set_parameters(data::product_moments(sample));  // {mean, sd, ...}; first two used
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else {
            set_parameters(mle(sample));
        }
    }

    // Bootstrap (C# Normal.cs:239, IBootstrappable): draw a fresh sample from the current
    // parameters, re-fit by `method`, and return the fitted Normal; throws if the fit is
    // invalid. ADDITIVE-ONLY: a new method on this oracle-locked Phase 1 file (X7) -- no
    // existing member's layout or signature changes.
    std::unique_ptr<UnivariateDistributionBase> bootstrap(ParameterEstimationMethod method,
                                                          int sample_size,
                                                          int seed = -1) const override {
        auto new_distribution = std::make_unique<Normal>(mu_, sigma_);
        auto sample = new_distribution->generate_random_values(sample_size, seed);
        new_distribution->estimate(sample, method);
        if (!new_distribution->parameters_valid())
            throw std::runtime_error("Bootstrapped distribution parameters are invalid.");
        return new_distribution;
    }

    // ParametersFromMoments (C# Normal.cs:267): the Normal is parameterized by its
    // first two moments (C# moments.ToArray().Subset(0, 1)).
    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        return {moments[0], moments[1]};
    }

    // MomentsFromParameters (C# Normal.cs:273): {Mean, StandardDeviation, Skewness,
    // Kurtosis} of a Normal built from the parameters.
    std::vector<double> moments_from_parameters(const std::vector<double>& parameters) const {
        Normal dist;
        dist.set_parameters(parameters);
        double m1 = dist.mean();
        double m2 = dist.standard_deviation();
        double m3 = dist.skewness();
        double m4 = dist.kurtosis();
        return {m1, m2, m3, m4};
    }

    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        double mu = moments[0];
        double sigma = moments[1] * std::sqrt(kPi);
        return {mu, sigma};
    }

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
        auto moments = data::product_moments(sample);
        initials = {moments[0], moments[1]};
        lowers.assign(2, 0.0);
        uppers.assign(2, 0.0);
        if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
        double locExp = std::ceil(std::log10(std::fabs(initials[0])) + 1.0);
        lowers[0] = -std::pow(10.0, locExp);
        uppers[0] = std::pow(10.0, locExp);
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(initials[1]) + 1.0));
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            Normal n;
            n.set_parameters(x[0], x[1]);
            return n.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 2, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

    // Gradient of the quantile function wrt {mu, sigma} (C# IStandardError.QuantileGradient,
    // Normal.cs:833). Q(p) = mu + sigma*z(p), so dQ/dmu = 1, dQ/dsigma = z(p). C#
    // ValidateParameters(..., true) throw -> std::invalid_argument.
    std::vector<double> quantile_gradient(double probability) const {
        // Validate parameters
        if (!parameters_valid_) throw std::invalid_argument("Normal: invalid parameters");
        double z = standard_z(probability);
        return {
            1.0,  // dQ/dmu
            z     // dQ/dsigma
        };
    }

    // ConditionalMoments override (C# Normal.cs:873): closed-form truncated-normal
    // moments; m2-m4 are central about the UNCONDITIONAL mean, mirroring the base virtual.
    std::vector<double> conditional_moments(double a, double b) const override {
        if (a >= b) return {kNaN, kNaN, kNaN, kNaN};

        // Unconditional parameters
        double mu = mu_;
        double sigma = sigma_;

        // Standardized limits
        double alpha_s = (a - mu) / sigma;
        double beta_s = (b - mu) / sigma;

        // CDF of standard normal
        double Phi_a = 0.5 * (1.0 + std::erf(alpha_s / kSqrt2));
        double Phi_b = 0.5 * (1.0 + std::erf(beta_s / kSqrt2));
        double Z = Phi_b - Phi_a;  // normalization

        // PDF of standard normal
        double phi_a = std::exp(-0.5 * alpha_s * alpha_s) / kSqrt2PI;
        double phi_b = std::exp(-0.5 * beta_s * beta_s) / kSqrt2PI;

        // auxiliary ratios
        double lambda = (phi_a - phi_b) / Z;                    // E[Z]
        double delta = (alpha_s * phi_a - beta_s * phi_b) / Z;  // E[Z^2]-1
        double tau = ((alpha_s * alpha_s + 2.0) * phi_a -
                      (beta_s * beta_s + 2.0) * phi_b) / Z;     // E[Z^3]
        double kap = (alpha_s * alpha_s * alpha_s * phi_a -
                      beta_s * beta_s * beta_s * phi_b) / Z;    // for E[Z^4]

        // now build the moments
        double m1 = mu + sigma * lambda;
        double m2 = sigma * sigma * (1.0 + delta);
        double m3 = sigma * sigma * sigma * tau;
        double m4 = sigma * sigma * sigma * sigma * (3.0 + kap + 3.0 * delta);

        return {m1, m2, m3, m4};
    }

    // --- Standard normal helpers (static, mirror the C# public API) ---

    // Standard normal PDF φ(z). Mirrors Normal.StandardPDF(Z) in C#.
    static double standard_pdf(double z) {
        return std::exp(-0.5 * z * z) / kSqrt2PI;
    }

    // Standard normal CDF Φ(z). Mirrors Normal.StandardCDF(Z) in C# (which delegates to the
    // accurate MVNPHI algorithm, unlike the instance cdf(x) above -- see this file's header
    // for why standard_cdf specifically needs the numerically-stable erfc form).
    static double standard_cdf(double z) { return 0.5 * std::erfc(-z / kSqrt2); }

    // Z variate of the standard normal for a probability (Wichura AS241).
    static double standard_z(double probability) {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        return r8_normal_01_cdf_inverse(probability);
    }

    // Standard normal log-PDF log φ(z). Mirrors Normal.StandardLogPDF(Z) (Numerics
    // Normal.cs:591) -> `-0.5 * Z * Z - Tools.LogSqrt2PI` term-for-term. ADDITIVE-ONLY: a
    // new static helper the GaussianCopula (S2) LogPDF needs; no existing line altered.
    // (The C# uses the Tools.LogSqrt2PI constant, already ported as
    // corehydro::numerics::kLogSqrt2PI in tools.hpp -- deviating from the S2 brief, which
    // suggested computing std::log(kSqrt2PI); the C# source governs and kLogSqrt2PI mirrors
    // Tools.LogSqrt2PI exactly.)
    static double standard_log_pdf(double z) {
        return -0.5 * z * z - kLogSqrt2PI;
    }

   private:
    static bool validate(double location, double scale) {
        if (std::isnan(location) || std::isinf(location)) return false;
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        return true;
    }

    // R8POLY_VALUE evaluates a polynomial p(x) = a[0] + a[1] x + ... + a[n-1] x^(n-1).
    static double r8poly_value(int n, const double a[], double x) {
        double value = 0.0;
        for (int i = n - 1; 0 <= i; --i) value = value * x + a[i];
        return value;
    }

    // R8_NORMAL_01_CDF_INVERSE inverts the standard normal CDF (Wichura, AS241, 1988).
    // Accurate to ~1 part in 1e16. Original FORTRAN77 by Michael Wichura; C++ by John Burkardt.
    static double r8_normal_01_cdf_inverse(double p) {
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
        if (1.0 <= p) return kInf;

        double q = p - 0.5;
        double r, value;
        if (std::fabs(q) <= 0.425) {
            r = 0.180625 - q * q;
            value = q * r8poly_value(8, a, r) / r8poly_value(8, b, r);
        } else {
            r = q < 0.0 ? p : 1.0 - p;
            r = std::sqrt(-std::log(r));
            if (r <= 5.0) {
                r = r - 1.6;
                value = r8poly_value(8, c, r) / r8poly_value(8, d, r);
            } else {
                r = r - 5.0;
                value = r8poly_value(8, e, r) / r8poly_value(8, f, r);
            }
            if (q < 0.0) value = -value;
        }
        return value;
    }

    double mu_ = 0.0;
    double sigma_ = 1.0;
};

}  // namespace corehydro::numerics::distributions
