// ported from: Numerics/Distributions/Bivariate Copulas/JoeCopula.cs @ 7e8e8d1
//
// The Joe copula. theta in [1, +inf). No PDF/CDF override -- both resolve through
// ArchimedeanCopula's generic Genest-1986 forms built from the generator functions below.
// generator_prime_inverse has no closed form (Brent.Solve on [0, 1], mirroring
// AMH/Gumbel). InverseCDF also has no closed form: it solves the conditional distribution
// C(v|u) = p for v via a second, distinct Brent root-find over [0, 1] (mirroring
// GumbelCopula's InverseCDF override, not the generic ArchimedeanCopula::inverse_cdf).
// Like ClaytonCopula/GumbelCopula (and unlike AMH/Frank), JoeCopula does NOT override
// ValidateParameter, so it inherits ArchimedeanCopula's validate_parameter directly --
// including the v2.1.4 fix that makes ParametersValid report true for an in-range theta
// (previously always false; see clayton_copula.hpp / archimedean_copula.hpp /
// docs/upstream-csharp-issues.md). Clone() deep-copies attached marginals via
// BivariateCopula::clone_marginal (v2.1.4, Task 8).
//
// Numerics v2.2.0 adds SetThetaFromTau using the Joe tau series and Brent inversion.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/correlation.hpp"
#include "corehydro/numerics/distributions/copulas/base/archimedean_copula.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"

namespace corehydro::numerics::distributions::copulas {

class JoeCopula : public ArchimedeanCopula {
   public:
    // Constructs a Joe copula with a dependency theta = 2.
    JoeCopula() { set_theta(2.0); }

    // Constructs a Joe copula with a specified theta.
    explicit JoeCopula(double theta) { set_theta(theta); }

    // Constructs a Joe copula with a specified theta and marginal distributions.
    JoeCopula(double theta, std::shared_ptr<UnivariateDistributionBase> marginal_distribution_x_,
              std::shared_ptr<UnivariateDistributionBase> marginal_distribution_y_) {
        set_theta(theta);
        marginal_distribution_x = std::move(marginal_distribution_x_);
        marginal_distribution_y = std::move(marginal_distribution_y_);
    }

    CopulaType type() const override { return CopulaType::Joe; }

    double theta_minimum() const override { return 1.0; }
    double theta_maximum() const override { return std::numeric_limits<double>::infinity(); }

    double generator(double t) const override {
        double a = 1.0 - t;
        return -std::log(1.0 - sign(a) * std::pow(std::fabs(a), theta()));
    }

    double generator_inverse(double t) const override {
        double a = 1.0 - std::exp(-t);
        return 1.0 - sign(a) * std::pow(std::fabs(a), 1.0 / theta());
    }

    double generator_prime(double t) const override {
        double a = 1.0 - t;
        return -(theta() * sign(a) * std::pow(std::fabs(a), theta() - 1.0)) /
               (1.0 - sign(a) * std::pow(std::fabs(a), theta()));
    }

    double generator_prime2(double t) const override {
        double a = 1.0 - t;
        double num = theta() * (theta() + sign(a) * std::pow(std::fabs(a), theta()) - 1.0) *
                     sign(a) * std::pow(std::fabs(a), theta() - 2.0);
        double aa = 1.0 - sign(a) * std::pow(std::fabs(a), theta());
        double den = sign(aa) * std::pow(std::fabs(aa), 2.0);
        return num / den;
    }

    double generator_prime_inverse(double t) const override {
        return corehydro::numerics::math::rootfinding::solve(
            [this, t](double x) { return generator_prime(x) - t; }, 0.0, 1.0);
    }

    // Solves the conditional distribution C(v|u) = p for v via Brent root-find (no closed
    // form for Joe).
    double inverse_conditional_cdf(double u, double t) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double p = t;
        double th = theta();
        auto f = [u, p, th](double x) {
                double a = std::pow(1.0 - u, th);
                double b = std::pow(1.0 - x, th);
                double vu = -(b - 1.0) * std::pow(a - a * b + b, (-th + 1.0) / th) *
                            std::pow(1.0 - u, th - 1.0);
                return vu - p;
            };
        if (f(1.0) <= 0.0) return 1.0;
        return corehydro::numerics::math::rootfinding::solve(f, 0.0, 1.0);
    }

    std::array<double, 2> inverse_cdf(double u, double v) const override {
        return {u, inverse_conditional_cdf(u, v)};
    }

    // Gets the upper tail dependence coefficient lambda_U = 2 - 2^(1/theta).
    double upper_tail_dependence() const override { return 2.0 - std::pow(2.0, 1.0 / theta()); }

    // Gets the lower tail dependence coefficient lambda_L = 0. The Joe copula has no lower
    // tail dependence.
    double lower_tail_dependence() const override { return 0.0; }

    std::unique_ptr<BivariateCopula> clone() const override {
        return std::make_unique<JoeCopula>(theta(), clone_marginal(marginal_distribution_x),
                                            clone_marginal(marginal_distribution_y));
    }

    static double kendalls_tau_from_theta(double theta_value) {
        if (theta_value < 1.0 || !std::isfinite(theta_value))
            throw std::out_of_range(
                "The dependency parameter theta must be finite and greater than or equal to 1.");
        double sum = 0.0;
        for (int k = kTauSeriesTerms; k >= 1; --k) {
            sum += 1.0 / (static_cast<double>(k) * (theta_value * k + 2.0) *
                          (theta_value * (k - 1.0) + 2.0));
        }
        return 1.0 - 4.0 * (sum + tau_series_tail(theta_value));
    }

    void set_theta_from_tau(const std::vector<double>& sample_data_x,
                            const std::vector<double>& sample_data_y) {
        double tau = corehydro::numerics::data::kendalls_tau(sample_data_x, sample_data_y);
        constexpr double lower = 1.0;
        constexpr double upper = 100.0;
        if (tau < 0.0 || tau > kendalls_tau_from_theta(upper))
            throw std::invalid_argument(
                "For the Joe copula, tau is outside the fitting range [0, 0.98025].");
        if (tau <= kendalls_tau_from_theta(lower)) {
            set_theta(lower);
            return;
        }
        set_theta(corehydro::numerics::math::rootfinding::solve(
            [tau](double value) { return kendalls_tau_from_theta(value) - tau; }, lower,
            upper));
    }

    math::linalg::Matrix2D parameter_constraints(const std::vector<double>&,
                                                  const std::vector<double>&) const override {
        return {{1.0, 100.0}};
    }

   private:
    static constexpr int kTauSeriesTerms = 1000;
    static constexpr int kTauSeriesTailOrder = 6;

    static double tau_series_tail(double theta_value) {
        double a = kTauSeriesTerms + 1.0;
        double inverse = 1.0 / a;
        double p = 2.0 / theta_value;
        double q = (2.0 - theta_value) / theta_value;
        double h = 1.0;
        double q_power = 1.0;
        double a_power = inverse * inverse * inverse;
        double sign_value = 1.0;
        double tail = 0.0;
        for (int j = 0; j <= kTauSeriesTailOrder; ++j) {
            double m = 3.0 + j;
            double zeta = a_power * a / (m - 1.0) + 0.5 * a_power +
                          m / 12.0 * a_power * inverse -
                          m * (m + 1.0) * (m + 2.0) / 720.0 * a_power * inverse * inverse *
                              inverse;
            tail += sign_value * h * zeta;
            sign_value = -sign_value;
            q_power *= q;
            h = p * h + q_power;
            a_power *= inverse;
        }
        return tail / (theta_value * theta_value);
    }

    // Math.Sign(a): -1, 0, or 1 (distinct from Tools.Sign's Fortran-style 2-arg transfer
    // used by brent_search.hpp).
    static double sign(double a) { return a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : 0.0); }
};

}  // namespace corehydro::numerics::distributions::copulas
