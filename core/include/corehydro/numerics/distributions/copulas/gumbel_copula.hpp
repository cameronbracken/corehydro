// ported from: Numerics/Distributions/Bivariate Copulas/GumbelCopula.cs @ 7e8e8d1
//
// The Gumbel (Gumbel-Hougaard) copula. theta in [1, +inf). No PDF/CDF override -- both
// resolve through ArchimedeanCopula's generic Genest-1986 forms built from the generator
// functions below. InverseCDF has no closed form: it solves the conditional distribution
// C(v|u) = p for v via Brent.Solve on [0, 1] (a nested/different root-solve from the
// generic ArchimedeanCopula::inverse_cdf, which instead composes
// generator/generator_prime/generator_prime_inverse -- Gumbel overrides InverseCDF outright
// with this closed conditional-probability solve, matching the C# source). Like
// ClaytonCopula (and unlike AMH/Frank), GumbelCopula does NOT override ValidateParameter,
// so it inherits ArchimedeanCopula's validate_parameter directly -- including the v2.1.4 fix
// that makes ParametersValid report true for an in-range theta (previously always false; see
// clayton_copula.hpp / archimedean_copula.hpp / docs/upstream-csharp-issues.md). Clone()
// deep-copies attached marginals via BivariateCopula::clone_marginal (v2.1.4, Task 8).
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "corehydro/numerics/data/correlation.hpp"
#include "corehydro/numerics/distributions/copulas/base/archimedean_copula.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"

namespace corehydro::numerics::distributions::copulas {

class GumbelCopula : public ArchimedeanCopula {
   public:
    // Constructs a Gumbel copula with a dependency theta = 2.
    GumbelCopula() { set_theta(2.0); }

    // Constructs a Gumbel copula with a specified theta.
    explicit GumbelCopula(double theta) { set_theta(theta); }

    // Constructs a Gumbel copula with a specified theta and marginal distributions.
    GumbelCopula(double theta, std::shared_ptr<UnivariateDistributionBase> marginal_distribution_x_,
                 std::shared_ptr<UnivariateDistributionBase> marginal_distribution_y_) {
        set_theta(theta);
        marginal_distribution_x = std::move(marginal_distribution_x_);
        marginal_distribution_y = std::move(marginal_distribution_y_);
    }

    CopulaType type() const override { return CopulaType::Gumbel; }

    double theta_minimum() const override { return 1.0; }
    double theta_maximum() const override { return std::numeric_limits<double>::infinity(); }

    double generator(double t) const override {
        double a = -std::log(t);
        return sign(a) * std::pow(std::fabs(a), theta());
    }

    double generator_inverse(double t) const override {
        double a = -t;
        return std::exp(sign(a) * std::pow(std::fabs(a), 1.0 / theta()));
    }

    double generator_prime(double t) const override {
        double a = std::log(t);
        return -theta() * sign(a) * std::pow(std::fabs(a), theta() - 1.0) / t;
    }

    double generator_prime2(double t) const override {
        double a = -std::log(t);
        return theta() * sign(a) * std::pow(std::fabs(a), theta() - 2.0) *
               (-theta() + std::log(t) + 1.0) / (t * t);
    }

    double generator_prime_inverse(double t) const override {
        return corehydro::numerics::math::rootfinding::solve(
            [this, t](double x) { return generator_prime(x) - t; }, 0.0, 1.0);
    }

    // Solves the conditional distribution C(v|u) = p for v via Brent root-find (no closed
    // form for Gumbel).
    double inverse_conditional_cdf(double u, double t) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double p = t;
        double th = theta();
        auto f = [u, p, th](double x) {
                double lu = -std::log(u);
                double lx = -std::log(x);
                double s = std::pow(lu, th) + std::pow(lx, th);
                double vu = std::pow(lu, th - 1.0) * std::exp(-std::pow(s, 1.0 / th)) *
                            std::pow(s, 1.0 / th - 1.0) / u;
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

    // Gets the lower tail dependence coefficient lambda_L = 0. The Gumbel copula has no
    // lower tail dependence.
    double lower_tail_dependence() const override { return 0.0; }

    std::unique_ptr<BivariateCopula> clone() const override {
        return std::make_unique<GumbelCopula>(theta(), clone_marginal(marginal_distribution_x),
                                               clone_marginal(marginal_distribution_y));
    }

    // Estimates the dependency parameter using the method of moments.
    void set_theta_from_tau(const std::vector<double>& sample_data_x,
                             const std::vector<double>& sample_data_y) {
        double tau = corehydro::numerics::data::kendalls_tau(sample_data_x, sample_data_y);
        set_theta(1.0 / (1.0 - tau));
    }

    math::linalg::Matrix2D parameter_constraints(const std::vector<double>&,
                                                  const std::vector<double>&) const override {
        return {{1.0, 100.0}};
    }

   private:
    // Math.Sign(a): -1, 0, or 1 (distinct from Tools.Sign's Fortran-style 2-arg transfer
    // used by brent_search.hpp).
    static double sign(double a) { return a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : 0.0); }
};

}  // namespace corehydro::numerics::distributions::copulas
