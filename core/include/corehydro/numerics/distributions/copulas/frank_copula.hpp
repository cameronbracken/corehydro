// ported from: Numerics/Distributions/Bivariate Copulas/FrankCopula.cs @ 7e8e8d1
//
// The Frank copula. theta unbounded (-inf, +inf); custom closed-form PDF/CDF/InverseCDF
// (all five generator functions also have closed forms, including generator_prime_inverse
// -- no Brent root-solve is needed anywhere in this file, unlike AMH/Gumbel/Joe). Like
// AMHCopula, FrankCopula OVERRIDES ValidateParameter with a CORRECT implementation (`return
// null;` in range) -- it does NOT inherit ArchimedeanCopula's "always-false ParametersValid"
// bug, RESOLVED upstream in v2.1.4 (see amh_copula.hpp / clayton_copula.hpp /
// docs/upstream-csharp-issues.md). v2.1.4 also added a NaN/Inf check ahead of the range check
// here (mirrored below). Clone() deep-copies attached marginals via
// BivariateCopula::clone_marginal (v2.1.4, Task 8).
//
// Numerics v2.2.0 adds SetThetaFromTau using the Debye order-one relation and Brent inversion.
#pragma once
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/correlation.hpp"
#include "corehydro/numerics/distributions/copulas/base/archimedean_copula.hpp"
#include "corehydro/numerics/math/special/debye.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"

namespace corehydro::numerics::distributions::copulas {

class FrankCopula : public ArchimedeanCopula {
   public:
    // Constructs a Frank copula with a dependency theta = 2.
    FrankCopula() { set_theta(2.0); }

    // Constructs a Frank copula with a specified theta.
    explicit FrankCopula(double theta) { set_theta(theta); }

    // Constructs a Frank copula with a specified theta and marginal distributions.
    FrankCopula(double theta, std::shared_ptr<UnivariateDistributionBase> marginal_distribution_x_,
                std::shared_ptr<UnivariateDistributionBase> marginal_distribution_y_) {
        set_theta(theta);
        marginal_distribution_x = std::move(marginal_distribution_x_);
        marginal_distribution_y = std::move(marginal_distribution_y_);
    }

    CopulaType type() const override { return CopulaType::Frank; }

    double theta_minimum() const override { return -std::numeric_limits<double>::infinity(); }
    double theta_maximum() const override { return std::numeric_limits<double>::infinity(); }

    // Correct override (does NOT reproduce ArchimedeanCopula's now-resolved ParametersValid
    // bug -- see file header).
    std::optional<std::string> validate_parameter(double parameter,
                                                    bool throw_exception) const override {
        if (std::isnan(parameter) || std::isinf(parameter)) {
            std::string msg = "The dependency parameter must be finite.";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        if (parameter < theta_minimum()) {
            std::string msg = "The dependency parameter theta (theta) must be greater than or "
                               "equal to " +
                               std::to_string(theta_minimum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        if (parameter > theta_maximum()) {
            std::string msg =
                "The dependency parameter theta (theta) must be less than or equal to " +
                std::to_string(theta_maximum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        return std::nullopt;
    }

    double generator(double t) const override {
        if (std::fabs(theta()) < 1e-10) return t;
        return -std::log((std::exp(-theta() * t) - 1.0) / (std::exp(-theta()) - 1.0));
    }

    double generator_inverse(double t) const override {
        return -std::log(std::exp(-theta() - t) - std::exp(-t) + 1.0) / theta();
    }

    double generator_prime(double t) const override {
        return theta() / (1.0 - std::exp(theta() * t));
    }

    double generator_prime2(double t) const override {
        return theta() * theta() * std::exp(theta() * t) /
               std::pow(1.0 - std::exp(theta() * t), 2.0);
    }

    double generator_prime_inverse(double t) const override {
        return std::log((theta() - t) / -t) / theta();
    }

    double pdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double num = theta() * (std::exp(theta()) - 1.0) * std::exp(theta() * (1.0 + u + v));
        double den = std::pow(std::exp(theta() * (u + v)) - std::exp(theta() * (1.0 + u)) -
                                   std::exp(theta() * (1.0 + v)) + std::exp(theta()),
                               2.0);
        return num / den;
    }

    double cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        return -(1.0 / theta()) *
               std::log(1.0 + (std::exp(-theta() * u) - 1.0) * (std::exp(-theta() * v) - 1.0) /
                                  (std::exp(-theta()) - 1.0));
    }

    double inverse_conditional_cdf(double u, double t) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double a = -std::fabs(theta());
        double s = theta() > 0.0 ? 1.0 - t : t;
        double v = -1.0 / a *
                   std::log((-s * (std::exp(-a) - 1.0) /
                                 (std::exp(-a * u) * (s - 1.0) - s)) +
                            1.0);
        return theta() > 0.0 ? 1.0 - v : v;
    }

    std::array<double, 2> inverse_cdf(double u, double v) const override {
        return {u, inverse_conditional_cdf(u, v)};
    }

    // Gets the upper tail dependence coefficient lambda_U = 0. The Frank copula has no tail
    // dependence.
    double upper_tail_dependence() const override { return 0.0; }

    // Gets the lower tail dependence coefficient lambda_L = 0. The Frank copula has no tail
    // dependence.
    double lower_tail_dependence() const override { return 0.0; }

    std::unique_ptr<BivariateCopula> clone() const override {
        return std::make_unique<FrankCopula>(theta(), clone_marginal(marginal_distribution_x),
                                              clone_marginal(marginal_distribution_y));
    }

    static double kendalls_tau_from_theta(double theta_value) {
        if (theta_value == 0.0 || !std::isfinite(theta_value))
            throw std::out_of_range(
                "The dependency parameter theta must be finite and non-zero.");
        return 1.0 - 4.0 / theta_value *
                         (1.0 - corehydro::numerics::math::special::debye_function_order_one(
                                    theta_value));
    }

    void set_theta_from_tau(const std::vector<double>& sample_data_x,
                            const std::vector<double>& sample_data_y) {
        double tau = corehydro::numerics::data::kendalls_tau(sample_data_x, sample_data_y);
        if (std::fabs(tau) > kendalls_tau_from_theta(100.0))
            throw std::invalid_argument(
                "For the Frank copula, tau is outside the fitting range [-0.96065, 0.96065].");
        double near_independence = tau > 0.0 ? 0.001 : -0.001;
        if (std::fabs(tau) <= std::fabs(kendalls_tau_from_theta(near_independence))) {
            set_theta(near_independence);
            return;
        }
        double lower = tau > 0.0 ? 0.001 : -100.0;
        double upper = tau > 0.0 ? 100.0 : -0.001;
        set_theta(corehydro::numerics::math::rootfinding::solve(
            [tau](double value) { return kendalls_tau_from_theta(value) - tau; }, lower,
            upper));
    }

    math::linalg::Matrix2D parameter_constraints(const std::vector<double>& sample_data_x,
                                                  const std::vector<double>& sample_data_y) const override {
        double tau = corehydro::numerics::data::kendalls_tau(sample_data_x, sample_data_y);
        double L = tau > 0.0 ? 0.001 : -100.0;
        double U = tau > 0.0 ? 100.0 : -0.001;
        return {{L, U}};
    }
};

}  // namespace corehydro::numerics::distributions::copulas
