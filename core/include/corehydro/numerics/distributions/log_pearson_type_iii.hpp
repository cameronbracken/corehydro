// ported from: Numerics/Distributions/Univariate/LogPearsonTypeIII.cs @ 7e8e8d1
//
// Log-Pearson Type III distribution parameterized by mean µ, standard deviation σ,
// and skew γ of the log-transformed (base-10) data. Wraps PearsonTypeIII in log10
// space. Mirrors the C# source method-for-method. Standard USACE flood-frequency
// distribution (Bulletin 17C). v2.1.4 (2a0357a) signs L-skewness by the skew
// (`T3 *= Math.Sign(gamma)`), adds the T3==0/gamma==0 exact limits, and refines the
// alpha>=100 Stirling correction in both L-moment directions (retiring the earlier
// intentional divergence that had only the forward method's branch, see
// docs/upstream-csharp-issues.md).
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
#include "corehydro/numerics/distributions/pearson_type_iii.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/special/erf.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace sf = corehydro::numerics::math::special;

class LogPearsonTypeIII : public UnivariateDistributionBase,
                          public IEstimation,
                          public ILinearMomentEstimation,
                          public IMaximumLikelihoodEstimation {
   public:
    // Constructs a LogPearsonTypeIII with mean(log)=3, sd(log)=0.5, skew(log)=0.
    LogPearsonTypeIII() { set_parameters(3.0, 0.5, 0.0); }

    // Constructs a LogPearsonTypeIII with given mean, sd, and skew of log10(X).
    LogPearsonTypeIII(double mean_of_log, double sd_of_log, double skew_of_log) {
        set_parameters(mean_of_log, sd_of_log, skew_of_log);
    }

    double mu()    const { return mu_; }
    double sigma() const { return sigma_; }
    double gamma_param() const { return gamma_; }

    // Derived gamma-parameterization (log-space shifted Gamma) – mirrors C# Xi/Beta/Alpha.
    double xi()    const { return mu_ - 2.0 * sigma_ / gamma_; }
    double beta()  const { return 0.5 * sigma_ * gamma_; }
    double alpha() const { return 4.0 / (gamma_ * gamma_); }

    // Log-correction factor: K = 1 / ln(base), base = 10.
    // log_base(x) = ln(x) * K  (log10(x) = ln(x)/ln(10))
    // K = 1/ln(10)
    double k_factor() const { return 1.0 / std::log(kBase); }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::LogPearsonTypeIII;
    }
    int number_of_parameters() const override { return 3; }
    std::vector<double> get_parameters() const override { return {mu_, sigma_, gamma_}; }

    void set_parameters(double mean_of_log, double sd_of_log, double skew_of_log) {
        parameters_valid_ = validate(mean_of_log, sd_of_log, skew_of_log);
        mu_    = mean_of_log;
        sigma_ = sd_of_log;
        gamma_ = skew_of_log;
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2]);
    }

    // --- Moments / support ---
    // Mean of X (not of log X). Mirrors C# Mean property.
    double mean() const override {
        return std::exp(mu_ * std::log(kBase) + log_moment_shape(1));
    }

    double median() const override { return inverse_cdf(0.5); }

    // Mode of X. Mirrors C# Mode property.
    double mode() const override {
        const double log_base = std::log(kBase);
        const double scale = sigma_ * log_base;
        if (gamma_ == 0.0) return std::exp(mu_ * log_base - scale * scale);
        const double beta_log = beta() * log_base;
        const double shape = alpha();
        if (gamma_ > 0.0 && shape <= 1.0) return minimum();
        if (gamma_ < 0.0) {
            if (shape < 1.0) return maximum();
            if (shape == 1.0) return beta_log < -1.0 ? 0.0 : maximum();
            if (beta_log <= -1.0) return 0.0;
        }
        return std::exp(mu_ * log_base - (scale * scale + beta_log) / (1.0 + beta_log));
    }

    // Standard deviation of X. Mirrors C# StandardDeviation property.
    double standard_deviation() const override {
        const double first = log_moment_shape(1);
        const double second = log_moment_shape(2);
        if (std::isinf(first) || std::isinf(second)) return kInf;
        const double delta = second - 2.0 * first;
        const double log_excess =
            delta > 0.5 ? delta + std::log1p(-std::exp(-delta)) : std::log(std::expm1(delta));
        return std::exp(mu_ * std::log(kBase) + first + 0.5 * log_excess);
    }

    // Skewness of X. Mirrors C# Skewness property.
    double skewness() const override {
        if (std::isinf(log_moment_shape(3)))
            return std::isinf(log_moment_shape(2)) ? kNaN : kInf;
        const double scale = sigma_ * std::log(kBase);
        if (std::fabs(scale) * (1.0 + std::fabs(gamma_)) < 0.01)
            return small_scale_standardized_moments()[0];
        const double first = log_moment_shape(1);
        const double d2 = log_moment_shape(2) - 2.0 * first;
        const double d3 = log_moment_shape(3) - 3.0 * first;
        const double log_variance =
            d2 > 0.5 ? d2 + std::log1p(-std::exp(-d2)) : std::log(std::expm1(d2));
        const double normalized_third =
            -std::expm1(-d3) - 3.0 * std::exp(d2 - d3) * -std::expm1(-d2);
        if (normalized_third == 0.0) return 0.0;
        return std::copysign(
            std::exp(d3 - 1.5 * log_variance + std::log(std::fabs(normalized_third))),
            normalized_third);
    }

    // Kurtosis of X. Mirrors C# Kurtosis property.
    double kurtosis() const override {
        if (std::isinf(log_moment_shape(4)))
            return std::isinf(log_moment_shape(2)) ? kNaN : kInf;
        const double scale = sigma_ * std::log(kBase);
        if (std::fabs(scale) * (1.0 + std::fabs(gamma_)) < 0.01)
            return small_scale_standardized_moments()[1];
        const double first = log_moment_shape(1);
        const double d2 = log_moment_shape(2) - 2.0 * first;
        const double d3 = log_moment_shape(3) - 3.0 * first;
        const double d4 = log_moment_shape(4) - 4.0 * first;
        const double log_variance =
            d2 > 0.5 ? d2 + std::log1p(-std::exp(-d2)) : std::log(std::expm1(d2));
        const double normalized_fourth =
            -std::expm1(-d4) - 4.0 * std::exp(d3 - d4) * -std::expm1(-d3) +
            6.0 * std::exp(d2 - d4) * -std::expm1(-d2);
        return std::exp(d4 - 2.0 * log_variance + std::log(normalized_fourth));
    }

    // Minimum of X. Mirrors C# Minimum property.
    double minimum() const override {
        return gamma_ > 0.0 ? std::exp(xi() * std::log(kBase)) : 0.0;
    }

    // Maximum of X. Mirrors C# Maximum property.
    double maximum() const override {
        return gamma_ < 0.0 ? std::exp(xi() * std::log(kBase)) : kInf;
    }

    // --- Distribution functions ---
    double pdf(double x) const override {
        return std::exp(log_pdf(x));
    }

    double log_pdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("LogPearsonTypeIII: invalid parameters");
        const double log_base = std::log(kBase);
        const double boundary = gamma_ == 0.0 ? kNaN : std::exp(xi() * log_base);
        const double lower = gamma_ > 0.0 ? boundary : 0.0;
        const double upper = gamma_ < 0.0 ? boundary : kInf;
        if (x < lower || x > upper || x == kInf) return -kInf;
        if (x == 0.0) {
            if (gamma_ >= 0.0) return -kInf;
            const double rate = -1.0 / (beta() * log_base);
            if (rate > 1.0) return -kInf;
            if (rate < 1.0 || alpha() > 1.0) return kInf;
            return alpha() == 1.0 ? -xi() * log_base : -kInf;
        }
        const double log_x = std::log(x);
        const double transformed = ((gamma_ > 0.0 || gamma_ < 0.0) && x == boundary)
                                       ? xi()
                                       : log_x / log_base;
        return PearsonTypeIII(mu_, sigma_, gamma_).log_pdf(transformed) - std::log(log_base) -
               log_x;
    }

    double cdf(double x) const override {
        return std::exp(log_cdf(x));
    }

    double log_cdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("LogPearsonTypeIII: invalid parameters");
        if (x <= minimum()) return -kInf;
        if (x >= maximum()) return 0.0;
        return PearsonTypeIII(mu_, sigma_, gamma_).log_cdf(std::log(x) / std::log(kBase));
    }

    double ccdf(double x) const override { return std::exp(log_ccdf(x)); }

    double log_ccdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("LogPearsonTypeIII: invalid parameters");
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return -kInf;
        return PearsonTypeIII(mu_, sigma_, gamma_).log_ccdf(std::log(x) / std::log(kBase));
    }

    double inverse_cdf(double probability) const override {
        if (std::isnan(probability) || probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (!parameters_valid_)
            throw std::invalid_argument("LogPearsonTypeIII: invalid parameters");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return std::exp(PearsonTypeIII(mu_, sigma_, gamma_).inverse_cdf(probability) *
                        std::log(kBase));
    }

    // --- Parameter display names (X1; C# LogPearsonTypeIII.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Mean (of log) (\xC2\xB5)", "Std Dev (of log) (\xCF\x83)", "Skew (of log) (\xCE\xB3)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xC2\xB5", "\xCF\x83", "\xCE\xB3"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<LogPearsonTypeIII>(mu_, sigma_, gamma_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        distribution_numerics::validate_sample(sample, 4, true);
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            // Indirect MoM: transform to log10, compute product moments.
            auto log_sample = transform_log(sample);
            auto moments = data::product_moments(log_sample);
            set_parameters(moments[0], moments[1], moments[2]);
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            // Indirect L-moments: transform to log10, compute L-moments, then fit.
            auto log_sample = transform_log(sample);
            auto lmom = data::linear_moments(log_sample);
            set_parameters(parameters_from_linear_moments(lmom));
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::invalid_argument("LogPearsonTypeIII: unsupported estimation method");
        }
    }

    // ParametersFromMoments (C# LogPearsonTypeIII.cs:559): the LP3 is parameterized by
    // the first three (log10-space) moments (C# moments.ToArray().Subset(0, 2)).
    // Added in B4 for the Bulletin 17C GMM track.
    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        return {moments[0], moments[1], moments[2]};
    }

    // MomentsFromParameters (C# LogPearsonTypeIII.cs:565): {Mean, StandardDeviation,
    // Skewness, Kurtosis} of an LP3 built from the parameters. Note these are the
    // REAL-SPACE moments of X, not the log-space parameters (upstream asymmetry; C#
    // governs). Added in B4.
    std::vector<double> moments_from_parameters(const std::vector<double>& parameters) const {
        LogPearsonTypeIII dist;
        dist.set_parameters(parameters);
        double m1 = dist.mean();
        double m2 = dist.standard_deviation();
        double m3 = dist.skewness();
        double m4 = dist.kurtosis();
        return {m1, m2, m3, m4};
    }

    // ParametersFromLinearMoments: rational-function approximation (Hosking).
    // Mirrors C# ParametersFromLinearMoments exactly (same code as PearsonTypeIII).
    // v2.1.4 adds the T3==0 exact Normal limit and refines the alpha>=100 Stirling
    // correction (now an `inverseAlpha`-based expansion, matching C# bit-for-bit).
    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        double L1 = moments[0];
        double L2 = moments[1];
        double T3 = moments[2];
        if (T3 == 0.0) {
            return {L1, L2 * std::sqrt(kPi), 0.0};
        }
        double alpha_val = kNaN;
        double z;
        if (std::fabs(T3) > 0.0 && std::fabs(T3) < 1.0 / 3.0) {
            z = 3.0 * kPi * T3 * T3;
            alpha_val = (1.0 + 0.2906 * z)
                        / (z + 0.1882 * z * z + 0.0442 * z * z * z);
        } else if (std::fabs(T3) >= 1.0 / 3.0 && std::fabs(T3) < 1.0) {
            z = 1.0 - std::fabs(T3);
            alpha_val = (0.36067 * z - 0.59567 * z * z + 0.25361 * z * z * z)
                        / (1.0 - 2.78861 * z + 2.56096 * z * z - 0.77045 * z * z * z);
        }
        double mu_val    = L1;
        double gamma_val = 2.0 * std::pow(alpha_val, -0.5)
                           * (T3 >= 0.0 ? 1.0 : -1.0);
        double sigma_val;
        if (alpha_val < 100.0) {
            sigma_val = L2 * std::sqrt(kPi) * std::sqrt(alpha_val)
                        * sf::function(alpha_val) / sf::function(alpha_val + 0.5);
        } else {
            double inverse_alpha = 1.0 / alpha_val;
            double correction = 1.0 - inverse_alpha / 8.0
                                + inverse_alpha * inverse_alpha / 128.0;
            sigma_val = std::sqrt(kPi) * L2 / correction;
        }
        return {mu_val, sigma_val, gamma_val};
    }

    // LinearMomentsFromParameters: rational-function approximation (Hosking).
    // Returns L-moments of log(X) (the underlying PT3). Mirrors C# exactly. v2.1.4
    // adds the gamma==0 exact Normal limit, simplifies L1 to exactly mu (was
    // xi + alpha*beta, algebraically identical but ulp-different), refines the
    // alpha>=100 Stirling correction, and signs T3 by Sign(gamma) (SIGNED
    // L-skewness) -- this retires the earlier intentional divergence documented in
    // docs/upstream-csharp-issues.md ("LogPearsonTypeIII.LinearMomentsFromParameters
    // overflows for small skew"): upstream now carries the matching alpha>=100
    // branch itself.
    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double mu_val    = parameters[0];
        double sigma_val = parameters[1];
        double gamma_val = parameters[2];
        if (gamma_val == 0.0) {
            return {mu_val, sigma_val / std::sqrt(kPi), 0.0, 0.12260172};
        }
        double alpha_val = 4.0 / (gamma_val * gamma_val);
        double beta_val  = 0.5 * sigma_val * gamma_val;
        double L1 = mu_val;
        double L2;
        if (alpha_val < 100.0) {
            L2 = std::fabs(beta_val * sf::function(alpha_val + 0.5)
                           / (std::sqrt(kPi) * sf::function(alpha_val)));
        } else {
            double inverse_alpha = 1.0 / alpha_val;
            double correction = 1.0 - inverse_alpha / 8.0
                                + inverse_alpha * inverse_alpha / 128.0;
            L2 = sigma_val / std::sqrt(kPi) * correction;
        }
        // Approximations accurate to 1e-6 (mirrors C# exactly)
        constexpr double A0 = 0.32573501,  A1 = 0.1686915,   A2 = 0.078327243, A3 = -0.0029120539;
        constexpr double B1 = 0.46697102,  B2 = 0.24255406;
        constexpr double C0 = 0.12260172,  C1 = 0.05373013,  C2 = 0.043384378, C3 = 0.011101277;
        constexpr double D1 = 0.18324466,  D2 = 0.20166036;
        constexpr double E1 = 2.3807576,   E2 = 1.5931792,   E3 = 0.11618371;
        constexpr double F1 = 5.1533299,   F2 = 7.142526,    F3 = 1.9745056;
        constexpr double G1 = 2.1235833,   G2 = 4.1670213,   G3 = 3.1925299;
        constexpr double H1 = 9.0551443,   H2 = 26.649995,   H3 = 26.193668;
        double T3, T4;
        if (alpha_val >= 1.0) {
            T3 = std::pow(alpha_val, -0.5)
                 * (A0 + A1 * std::pow(alpha_val, -1) + A2 * std::pow(alpha_val, -2)
                    + A3 * std::pow(alpha_val, -3))
                 / (1.0 + B1 * std::pow(alpha_val, -1) + B2 * std::pow(alpha_val, -2));
            T4 = (C0 + C1 * std::pow(alpha_val, -1) + C2 * std::pow(alpha_val, -2)
                  + C3 * std::pow(alpha_val, -3))
                 / (1.0 + D1 * std::pow(alpha_val, -1) + D2 * std::pow(alpha_val, -2));
        } else {
            T3 = (1.0 + E1 * alpha_val + E2 * alpha_val * alpha_val
                  + E3 * alpha_val * alpha_val * alpha_val)
                 / (1.0 + F1 * alpha_val + F2 * alpha_val * alpha_val
                    + F3 * alpha_val * alpha_val * alpha_val);
            T4 = (1.0 + G1 * alpha_val + G2 * alpha_val * alpha_val
                  + G3 * alpha_val * alpha_val * alpha_val)
                 / (1.0 + H1 * alpha_val + H2 * alpha_val * alpha_val
                    + H3 * alpha_val * alpha_val * alpha_val);
        }
        T3 *= (gamma_val > 0.0) - (gamma_val < 0.0);  // Math.Sign(gamma)

        return {L1, L2, T3, T4};
    }

    // GetParameterConstraints: mirrors C# GetParameterConstraints for MLE.
    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        distribution_numerics::validate_sample(sample, 4);
        auto constraints = distribution_numerics::prefer_legacy_constraints(
            [&] { return legacy_parameter_constraints(sample); },
            [&] { return robust_parameter_constraints(sample); });
        initials = std::get<0>(constraints);
        lowers = std::get<1>(constraints);
        uppers = std::get<2>(constraints);
    }

    distribution_numerics::Constraints robust_parameter_constraints(
        const std::vector<double>& sample) const {
        distribution_numerics::validate_sample(sample, 4, true);
        return PearsonTypeIII().robust_parameter_constraints(transform_log(sample));
    }

   private:
    distribution_numerics::Constraints legacy_parameter_constraints(
        const std::vector<double>& sample) const {
        std::vector<double> transformed(sample.size());
        for (std::size_t i = 0; i < sample.size(); ++i)
            transformed[i] = std::log(sample[i] > 0.0 ? sample[i] : 0.01) / std::log(kBase);
        auto mom = data::product_moments(transformed);
        std::vector<double> initials = {mom[0], mom[1], mom[2]};
        std::vector<double> lowers(3);
        std::vector<double> uppers(3);
        double K = k_factor();
        // Bounds of mu
        double real_mu = std::exp(initials[0] / K);
        if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
        lowers[0] = std::floor(std::log(std::pow(10.0, std::floor(std::log10(real_mu)) - 1.0))
                               / std::log(kBase));
        uppers[0] = std::ceil(std::log(std::pow(10.0, std::ceil(std::log10(real_mu)) + 1.0))
                              / std::log(kBase));
        // Bounds of sigma
        double real_sigma = std::exp(initials[1] / K);
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::ceil(std::log(std::pow(10.0, std::ceil(std::log10(real_sigma) + 1.0)))
                              / std::log(kBase));
        if (std::isnan(uppers[1])) uppers[1] = 4.0;
        // Bounds of gamma
        lowers[2] = -6.0;
        uppers[2] = 6.0;
        // Correct initial skew if out of range
        if (initials[2] <= lowers[2] || initials[2] >= uppers[2]) {
            initials[2] = 0.01;
        }
        return {initials, lowers, uppers};
    }

   public:

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            LogPearsonTypeIII lp3;
            lp3.set_parameters(x[0], x[1], x[2]);
            return lp3.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 3, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

   private:
    double log_moment_shape(int order) const {
        const double scale = order * sigma_ * std::log(kBase);
        const double argument = gamma_ * scale / 2.0;
        if (!(argument < 1.0)) return kInf;
        if (std::fabs(argument) < 0.01) {
            double sum = 0.5;
            double term = 1.0;
            for (int k = 1; k <= 12; ++k) {
                term *= argument;
                sum += term / (k + 2.0);
            }
            return scale * scale * sum;
        }
        return alpha() * (-std::log1p(-argument) - argument);
    }

    static std::vector<double> multiply_series(const std::vector<double>& left,
                                               const std::vector<double>& right) {
        std::vector<double> result(left.size(), 0.0);
        for (std::size_t n = 0; n < result.size(); ++n)
            for (std::size_t k = 0; k <= n; ++k) result[n] += left[k] * right[n - k];
        return result;
    }

    std::vector<double> small_scale_standardized_moments() const {
        constexpr int order = 12;
        std::vector<std::vector<double>> raw(4, std::vector<double>(order + 1, 0.0));
        for (int r = 1; r <= 4; ++r) {
            raw[static_cast<std::size_t>(r - 1)][0] = 1.0;
            for (int n = 2; n <= order; ++n)
                for (int k = 2; k <= n; ++k)
                    raw[static_cast<std::size_t>(r - 1)][static_cast<std::size_t>(n)] +=
                        std::pow(static_cast<double>(r), k) * std::pow(gamma_ / 2.0, k - 2) *
                        raw[static_cast<std::size_t>(r - 1)]
                           [static_cast<std::size_t>(n - k)] /
                        n;
        }
        const auto square = multiply_series(raw[0], raw[0]);
        const auto cube = multiply_series(square, raw[0]);
        const auto fourth = multiply_series(square, square);
        const auto second_first = multiply_series(raw[1], raw[0]);
        const auto third_first = multiply_series(raw[2], raw[0]);
        const auto second_square = multiply_series(raw[1], square);
        const double scale = sigma_ * std::log(kBase);
        double variance = 0.0;
        double third = 0.0;
        double fourth_central = 0.0;
        for (int n = order; n >= 2; --n)
            variance = variance * scale + raw[1][static_cast<std::size_t>(n)] -
                       square[static_cast<std::size_t>(n)];
        for (int n = order; n >= 3; --n)
            third = third * scale + raw[2][static_cast<std::size_t>(n)] -
                    3.0 * second_first[static_cast<std::size_t>(n)] +
                    2.0 * cube[static_cast<std::size_t>(n)];
        for (int n = order; n >= 4; --n)
            fourth_central = fourth_central * scale + raw[3][static_cast<std::size_t>(n)] -
                             4.0 * third_first[static_cast<std::size_t>(n)] +
                             6.0 * second_square[static_cast<std::size_t>(n)] -
                             3.0 * fourth[static_cast<std::size_t>(n)];
        return {third / std::pow(variance, 1.5), fourth_central / (variance * variance)};
    }

    // Transform a validated positive sample to log-base coordinates.
    std::vector<double> transform_log(const std::vector<double>& sample) const {
        distribution_numerics::validate_sample(sample, 4, true);
        std::vector<double> result;
        result.reserve(sample.size());
        for (double v : sample) result.push_back(std::log(v) / std::log(kBase));
        return result;
    }

    static bool validate(double mu, double sigma, double gamma) {
        if (std::isnan(mu)    || std::isinf(mu))    return false;
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) return false;
        if (std::isnan(gamma) || std::isinf(gamma)) return false;
        if (gamma > 6.0 || gamma < -6.0) return false;
        return true;
    }

    static constexpr double kBase    = 10.0;
    static constexpr double kSqrt2   = 1.4142135623730951;
    static constexpr double kSqrt2PI = 2.5066282746310002;

    double mu_    = 3.0;
    double sigma_ = 0.5;
    double gamma_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
