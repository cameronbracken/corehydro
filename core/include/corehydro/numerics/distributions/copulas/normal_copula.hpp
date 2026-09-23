// ported from: Numerics/Distributions/Bivariate Copulas/NormalCopula.cs @ 7e8e8d1
//
// The Gaussian (Normal) elliptical copula. theta = rho in [-1, +1]; extends BivariateCopula
// DIRECTLY (not ArchimedeanCopula -- the Normal copula has no Archimedean generator).
// PDF is the closed-form Gaussian copula density built from Normal::standard_z (the C#
// Normal.StandardZ quantile function). CDF delegates to the already-ported deterministic
// Drezner/Genz bivariate normal CDF, MultivariateNormal::bivariate_cdf -- passing the
// negated quantiles per the C# comment (BivariateCDF computes Phi2(-h,-k;r), so
// C(u,v) = Phi2(Phi^-1(u), Phi^-1(v); r) requires arguments -Phi^-1(u), -Phi^-1(v)).
// InverseCDF is the standard conditional-normal simulation formula. Zero tail dependence
// (a defining property of the Gaussian copula, unlike the Student's t copula).
//
// Unlike ArchimedeanCopula's ValidateParameter (see archimedean_copula.hpp's header for the
// upstream ParametersValid bug it reproduced, RESOLVED in v2.1.4), NormalCopula.ValidateParameter
// in the C# source returns `null` (not a non-null "Parameter is valid" sentinel) in its
// in-range branch -- ported here as std::nullopt, so ParametersValid always behaved correctly
// for this copula. v2.1.4 added a NaN/Inf check ahead of the range check here (mirrored below).
// Clone() deep-copies attached marginals via BivariateCopula::clone_marginal (v2.1.4, Task 8).
#pragma once
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/copulas/base/bivariate_copula.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions::copulas {

class NormalCopula : public BivariateCopula {
   public:
    // Constructs a bivariate Gaussian copula with a correlation rho = 0.0.
    NormalCopula() { set_theta(0.0); }

    // Constructs a bivariate Gaussian copula with a specified correlation rho.
    explicit NormalCopula(double rho) { set_theta(rho); }

    // Constructs a bivariate Gaussian copula with a specified theta and marginal
    // distributions.
    NormalCopula(double rho, std::shared_ptr<UnivariateDistributionBase> marginal_distribution_x_,
                 std::shared_ptr<UnivariateDistributionBase> marginal_distribution_y_) {
        set_theta(rho);
        marginal_distribution_x = std::move(marginal_distribution_x_);
        marginal_distribution_y = std::move(marginal_distribution_y_);
    }

    CopulaType type() const override { return CopulaType::Normal; }

    double theta_minimum() const override { return -1.0; }
    double theta_maximum() const override { return 1.0; }

    // Correct override (returns nullopt when in range) -- does NOT reproduce
    // ArchimedeanCopula's (now-resolved) ParametersValid bug (see that file's header and
    // docs/upstream-csharp-issues.md).
    std::optional<std::string> validate_parameter(double parameter,
                                                    bool throw_exception) const override {
        if (std::isnan(parameter) || std::isinf(parameter)) {
            std::string msg = "The correlation parameter must be finite.";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        if (parameter < theta_minimum()) {
            std::string msg =
                "The correlation parameter rho (rho) must be greater than " +
                std::to_string(theta_minimum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        if (parameter > theta_maximum()) {
            std::string msg = "The correlation parameter rho (rho) must be less than " +
                               std::to_string(theta_maximum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        return std::nullopt;
    }

    int number_of_copula_parameters() const override { return 1; }
    std::vector<double> get_copula_parameters() const override { return {theta()}; }
    void set_copula_parameters(const std::vector<double>& parameters) override {
        set_theta(parameters[0]);
    }

    math::linalg::Matrix2D parameter_constraints(const std::vector<double>&,
                                                  const std::vector<double>&) const override {
        return {{-1.0 + corehydro::numerics::kDoubleMachineEpsilon,
                 1.0 - corehydro::numerics::kDoubleMachineEpsilon}};
    }

    double pdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double r = theta();
        double s = Normal::standard_z(u);
        double t = Normal::standard_z(v);
        return 1.0 / std::sqrt(1.0 - r * r) *
               std::exp(-(r * r * s * s + r * r * t * t - 2.0 * r * s * t) / (2.0 * (1.0 - r * r)));
    }

    double cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        // BivariateCDF implements Genz's BVND which computes Phi2(-h,-k;r). To get the
        // copula C(u,v) = Phi2(Phi^-1(u), Phi^-1(v); r), pass -Phi^-1(u) = Phi^-1(1-u) as
        // arguments.
        return corehydro::numerics::distributions::MultivariateNormal::bivariate_cdf(
            -Normal::standard_z(u), -Normal::standard_z(v), theta());
    }

    double conditional_cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double r = theta();
        return Normal::standard_cdf((Normal::standard_z(v) - r * Normal::standard_z(u)) /
                                    std::sqrt(1.0 - r * r));
    }

    double inverse_conditional_cdf(double u, double t) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double z1 = Normal::standard_z(u);
        double z2 = Normal::standard_z(t);
        double r = theta();
        double w2 = r * z1 + std::sqrt(1.0 - r * r) * z2;
        return Normal::standard_cdf(w2);
    }

    std::array<double, 2> inverse_cdf(double u, double v) const override {
        return {u, inverse_conditional_cdf(u, v)};
    }

    // Gets the upper tail dependence coefficient lambda_U = 0. The Normal copula has no
    // upper tail dependence.
    double upper_tail_dependence() const override { return 0.0; }

    // Gets the lower tail dependence coefficient lambda_L = 0. The Normal copula has no
    // lower tail dependence.
    double lower_tail_dependence() const override { return 0.0; }

    std::unique_ptr<BivariateCopula> clone() const override {
        return std::make_unique<NormalCopula>(theta(), clone_marginal(marginal_distribution_x),
                                               clone_marginal(marginal_distribution_y));
    }
};

}  // namespace corehydro::numerics::distributions::copulas
