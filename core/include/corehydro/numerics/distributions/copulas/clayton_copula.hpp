// ported from: Numerics/Distributions/Bivariate Copulas/ClaytonCopula.cs @ 7e8e8d1
//
// The Clayton copula. theta in [-1, +inf); closed-form CDF/InverseCDF (overriding
// ArchimedeanCopula's generic Genest-1986 forms, which PDF still uses via virtual CDF
// dispatch). set_theta_from_tau is the method-of-moments fit: theta = 2*tau/(1-tau),
// Kendall's tau closed form for Clayton (not part of the shared BivariateCopula API --
// C# does not declare it on IBivariateCopula/IArchimedeanCopula either, it is a member of
// the concrete ClaytonCopula class only; the fixture/glue "tau" fit mode therefore
// dynamic_casts to ClaytonCopula, mirroring how the C# BivariateCopulaEstimation test flow
// calls `copula.SetThetaFromTau(...)` directly on a concrete ClaytonCopula, not through
// BivariateCopulaEstimation.Estimate).
//
// ClaytonCopula does NOT override ValidateParameter, so it inherits ArchimedeanCopula's
// (NaN/Inf-first, then range-checked) validate_parameter directly -- including the v2.1.4 fix
// that makes ParametersValid report true for an in-range theta (previously always false; see
// archimedean_copula.hpp / docs/upstream-csharp-issues.md). Clone() deep-copies attached
// marginals via BivariateCopula::clone_marginal (v2.1.4, Task 8).
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

namespace corehydro::numerics::distributions::copulas {

class ClaytonCopula : public ArchimedeanCopula {
   public:
    // Constructs a Clayton copula with a dependency theta = 2.
    ClaytonCopula() { set_theta(2.0); }

    // Constructs a Clayton copula with a specified theta.
    explicit ClaytonCopula(double theta) { set_theta(theta); }

    // Constructs a Clayton copula with a specified theta and marginal distributions.
    ClaytonCopula(double theta, std::shared_ptr<UnivariateDistributionBase> marginal_distribution_x_,
                  std::shared_ptr<UnivariateDistributionBase> marginal_distribution_y_) {
        set_theta(theta);
        marginal_distribution_x = std::move(marginal_distribution_x_);
        marginal_distribution_y = std::move(marginal_distribution_y_);
    }

    CopulaType type() const override { return CopulaType::Clayton; }

    double theta_minimum() const override { return -1.0; }
    double theta_maximum() const override { return std::numeric_limits<double>::infinity(); }

    double generator(double t) const override {
        double a = t;
        return sign(a) * std::pow(std::fabs(a), -theta()) - 1.0;
    }

    double generator_inverse(double t) const override {
        double a = 1.0 + t;
        return sign(a) * std::pow(std::fabs(a), -1.0 / theta());
    }

    double generator_prime(double t) const override {
        double a = t;
        return -theta() * sign(a) * std::pow(std::fabs(a), -theta() - 1.0);
    }

    double generator_prime2(double t) const override {
        double a = t;
        return -theta() * sign(a) * std::pow(std::fabs(a), -theta() - 2.0) * (-theta() - 1.0);
    }

    double generator_prime_inverse(double t) const override {
        double a = t / -theta();
        return sign(a) * std::pow(std::fabs(a), -1.0 / (theta() + 1.0));
    }

    double cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        return std::pow(std::max(std::pow(u, -theta()) + std::pow(v, -theta()) - 1.0, 0.0),
                         -1.0 / theta());
    }

    double inverse_conditional_cdf(double u, double t) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        return std::pow(
            std::pow(u, -theta()) * (std::pow(t, -theta() / (theta() + 1.0)) - 1.0) + 1.0,
            -1.0 / theta());
    }

    std::array<double, 2> inverse_cdf(double u, double v) const override {
        return {u, inverse_conditional_cdf(u, v)};
    }

    // Gets the upper tail dependence coefficient lambda_U = 0. The Clayton copula has no
    // upper tail dependence.
    double upper_tail_dependence() const override { return 0.0; }

    // Gets the lower tail dependence coefficient lambda_L = 2^(-1/theta).
    double lower_tail_dependence() const override {
        if (theta() <= 0.0) return 0.0;
        return std::pow(2.0, -1.0 / theta());
    }

    std::unique_ptr<BivariateCopula> clone() const override {
        return std::make_unique<ClaytonCopula>(theta(), clone_marginal(marginal_distribution_x),
                                                clone_marginal(marginal_distribution_y));
    }

    // Estimates the dependency parameter using the method of moments.
    void set_theta_from_tau(const std::vector<double>& sample_data_x,
                             const std::vector<double>& sample_data_y) {
        double tau = corehydro::numerics::data::kendalls_tau(sample_data_x, sample_data_y);
        set_theta((2.0 * tau) / (1.0 - tau));
    }

    math::linalg::Matrix2D parameter_constraints(const std::vector<double>&,
                                                  const std::vector<double>&) const override {
        return {{-1.0, 100.0}};
    }

   private:
    // Math.Sign(a): -1, 0, or 1 (distinct from Tools.Sign's Fortran-style 2-arg transfer
    // used by brent_search.hpp).
    static double sign(double a) { return a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : 0.0); }
};

}  // namespace corehydro::numerics::distributions::copulas
