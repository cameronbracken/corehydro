// ported from: Numerics/Distributions/Univariate/Exponential.cs @ 7e8e8d1
//
// The (two-parameter) Exponential distribution, location ξ and scale α. Logic mirrors
// the C# source method-for-method. B4 adds QuantileGradient and the ConditionalMoments
// override for the Bulletin 17C GMM track.
#pragma once
#include <string>
#include <algorithm>
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
#include "corehydro/numerics/math/special/factorial.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class Exponential : public UnivariateDistributionBase,
                    public IEstimation,
                    public ILinearMomentEstimation,
                    public IMaximumLikelihoodEstimation {
   public:
    Exponential() { set_parameters(100.0, 10.0); }
    Exponential(double location, double scale) { set_parameters(location, scale); }

    double xi() const { return xi_; }
    double alpha() const { return alpha_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Exponential;
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
    double mean() const override { return xi_ + alpha_; }
    double median() const override { return xi_ - std::log(0.5) * alpha_; }
    double mode() const override { return xi_; }
    double standard_deviation() const override { return alpha_; }
    double skewness() const override { return 2.0; }
    double kurtosis() const override { return 9.0; }
    double minimum() const override { return xi_; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override { return std::exp(log_pdf(x)); }

    double log_pdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Exponential: invalid parameters");
        if (x < minimum() || x > maximum()) return -kInf;
        const double value = -std::log(alpha_) -
                             distribution_numerics::standardize(x, xi_, alpha_);
        return std::isnan(value) ? -kInf : value;
    }

    double cdf(double x) const override {
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        return -std::expm1(-distribution_numerics::standardize(x, xi_, alpha_));
    }

    double log_cdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Exponential: invalid parameters");
        if (x <= xi_) return -kInf;
        return distribution_numerics::log1m_exp(
            -distribution_numerics::standardize(x, xi_, alpha_));
    }

    double ccdf(double x) const override { return std::exp(log_ccdf(x)); }

    double log_ccdf(double x) const override {
        if (!parameters_valid_) throw std::out_of_range("Exponential: invalid parameters");
        return x <= xi_ ? 0.0 : -distribution_numerics::standardize(x, xi_, alpha_);
    }

    double inverse_cdf(double probability) const override {
        if (!(probability >= 0.0 && probability <= 1.0))
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        const double unit_quantile = -std::log1p(-probability);
        const double displacement = alpha_ * unit_quantile;
        return std::isinf(displacement) && std::isfinite(unit_quantile)
                   ? alpha_ * (xi_ / alpha_ + unit_quantile)
                   : xi_ + displacement;
    }

    // --- Parameter display names (X1; C# Exponential.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Location (\xCE\xBE)", "Scale (\xCE\xB1)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBE", "\xCE\xB1"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Exponential>(xi_, alpha_);
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
        return {moments[0] - moments[1], moments[1]};
    }

    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        double L1 = moments[0], L2 = moments[1];
        double alpha = 2.0 * L2;
        double xi = L1 - alpha;
        return {xi, alpha};
    }

    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double xi = parameters[0], alpha = parameters[1];
        return {xi + alpha, 0.5 * alpha, 1.0 / 3.0, 1.0 / 6.0};
    }

    void get_parameter_constraints(const std::vector<double>& sample, std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        auto moments = data::product_moments(sample);
        double N = static_cast<double>(sample.size());
        double min_data = *std::min_element(sample.begin(), sample.end());
        initials = {(N * min_data - moments[0]) / (N - 1.0), N * (moments[0] - min_data) / (N - 1.0)};
        lowers.assign(2, 0.0);
        uppers.assign(2, 0.0);
        if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
        lowers[0] = initials[0] - std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0]))));
        uppers[0] = min_data + kDoubleMachineEpsilon;
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(initials[1]) + 1.0));
        if (initials[0] <= lowers[0] || initials[0] >= uppers[0])
            initials[0] = 0.5 * (lowers[0] + uppers[0]);
        if (initials[1] <= lowers[1] || initials[1] >= uppers[1])
            initials[1] = 0.5 * (lowers[1] + uppers[1]);
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            Exponential e;
            e.set_parameters(x[0], x[1]);
            return e.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 2, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

    // Gradient of the quantile function wrt {location, scale} (C#
    // IStandardError.QuantileGradient, Exponential.cs:457). Q(p) = xi - alpha*log(1-p).
    // C# ValidateParameters(..., true) throw -> std::invalid_argument.
    std::vector<double> quantile_gradient(double probability) const {
        distribution_numerics::validate_probability(probability);
        // Validate parameters
        if (!parameters_valid_) throw std::invalid_argument("Exponential: invalid parameters");
        return {
            1.0,                           // location
            -std::log1p(-probability)  // scale
        };
    }

    // ConditionalMoments override (C# Exponential.cs:495): closed-form truncated-
    // exponential raw moments via lower incomplete gamma partial sums, converted to
    // central moments about the UNCONDITIONAL mean mu = xi + alpha.
    std::vector<double> conditional_moments(double a, double b) const override {
        if (a >= b) return {kNaN, kNaN, kNaN, kNaN};

        double xi = xi_;
        double alpha = alpha_;
        if (!(alpha > 0.0)) return {kNaN, kNaN, kNaN, kNaN};

        // Map to Y = X - xi, truncated on (A, B)
        // Note: support is y >= 0
        double A = std::max(0.0, a - xi);
        double B = b - xi;

        // interval entirely left of support or invalid
        if (std::isnan(A) || std::isnan(B) || B <= 0.0) return {kNaN, kNaN, kNaN, kNaN};

        // Standardized limits t = y/alpha
        double tA = A / alpha;
        double tB = std::isinf(B) ? kInf : (B / alpha);

        // Normalizing probability Z = P(A < Y < B) = e^{-tA} - e^{-tB}
        double eA = std::exp(-tA);
        double eB = std::isinf(tB) ? 0.0 : std::exp(-tB);
        double Z = eA - eB;
        if (Z <= 1e-15) return {kNaN, kNaN, kNaN, kNaN};

        // Helper: S_n(t) = e^{-t} * sum_{k=0}^n t^k/k!
        // (appears in lower incomplete gamma(n+1, t) = n! * (1 - S_n(t)))
        auto Sn = [](double t, int n) {
            if (std::isinf(t)) return 0.0;
            double term = 1.0;  // t^0/0!
            double sum = term;
            for (int k = 1; k <= n; k++) {
                term *= t / k;  // t^k/k!
                sum += term;
            }
            return std::exp(-t) * sum;
        };

        // Precompute factorials for n = 0..4
        const double fact[5] = {1.0, 1.0, 2.0, 6.0, 24.0};

        // Raw truncated moments of Y: E[Y^n | A<Y<B] for n = 0..4
        // Using: int_A^B y^n (1/alpha) e^{-y/alpha} dy
        //          = alpha^n * [g(n+1, B/alpha) - g(n+1, A/alpha)]
        // and g(n+1, t) = n! * (1 - S_n(t)).
        double EY[5];
        EY[0] = 1.0;  // by definition under conditioning

        for (int n = 1; n <= 4; n++) {
            double SnA = Sn(tA, n);
            double SnB = Sn(tB, n);
            // alpha^n n! [S_n(tA) - S_n(tB)]
            double numer = std::pow(alpha, n) * fact[n] * (SnA - SnB);
            EY[n] = numer / Z;
        }

        // Convert to raw moments of X via binomial expansion:
        // E[X^k] = sum_{r=0}^k C(k,r) xi^(k-r) E[Y^r]
        double EX[5];
        for (int k = 0; k <= 4; k++) {
            double sum = 0.0;
            for (int r = 0; r <= k; r++) {
                double bc = math::special::factorial::binomial_coefficient(k, r);
                sum += bc * std::pow(xi, k - r) * EY[r];
            }
            EX[k] = sum;
        }

        // Central moments about the *unconditional* mean mu = xi + alpha
        double mu = xi + alpha;
        double m1 = EX[1];
        double m2 = EX[2] - 2.0 * mu * EX[1] + mu * mu;
        double m3 = EX[3] - 3.0 * mu * EX[2] + 3.0 * mu * mu * EX[1] - mu * mu * mu;
        double m4 = EX[4]
                  - 4.0 * mu * EX[3]
                  + 6.0 * mu * mu * EX[2]
                  - 4.0 * mu * mu * mu * EX[1]
                  + mu * mu * mu * mu;

        return {m1, m2, m3, m4};
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
