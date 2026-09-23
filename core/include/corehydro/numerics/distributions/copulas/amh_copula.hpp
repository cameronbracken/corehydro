// ported from: Numerics/Distributions/Bivariate Copulas/AMHCopula.cs @ 7e8e8d1
//
// The Ali-Mikhail-Haq (AMH) copula. theta in [-1, +1]; custom closed-form PDF (CDF falls
// through to ArchimedeanCopula's generic Genest-1986 form via generator/generator_inverse);
// InverseCDF is the closed-form Johnson (1987, p.362) quadratic solution. Unlike
// ClaytonCopula, AMHCopula OVERRIDES ValidateParameter with a CORRECT implementation
// (`return null;` in the C# in-range branch) -- i.e. AMH does NOT inherit
// ArchimedeanCopula's "always-false ParametersValid" bug, RESOLVED upstream in v2.1.4 (see
// clayton_copula.hpp / docs/upstream-csharp-issues.md); confirmed by reading AMHCopula.cs's
// own ValidateParameter override, which is textually identical to ArchimedeanCopula's except
// the final branch returns `null` instead of a non-null exception. v2.1.4 also added a
// NaN/Inf check ahead of the range check here (mirrored below), matching every other copula's
// finite-theta guard. generator_prime_inverse has no closed form and is solved via
// Brent.Solve(x => GeneratorPrime(x) - t, 0, 1) (corehydro::numerics::math::rootfinding::solve).
// set_theta_from_tau is also a Brent root-solve (no closed form): it throws
// (std::runtime_error, mirroring the C# bare `throw new Exception(...)`) if the sample's
// Kendall's tau falls outside AMH's achievable range [(5 - 8 ln 2)/3, 1/3] ~= [-0.1817, 0.3333].
// Clone() deep-copies attached marginals via BivariateCopula::clone_marginal (v2.1.4, Task 8).
//
// NOTE: the C# default constructor's XML doc says "a dependency theta = 2" but the body
// actually sets `Theta = 0.0d` (theta=2 would be out of AMH's [-1,1] range) -- transcribed
// as the actual code behavior (0.0), not the (apparently stale/copy-pasted) doc comment.
#pragma once
#include <algorithm>
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
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions::copulas {

class AMHCopula : public ArchimedeanCopula {
   public:
    // Constructs an AMH copula with a dependency theta = 0 (see file header note).
    AMHCopula() { set_theta(0.0); }

    // Constructs an AMH copula with a specified theta.
    explicit AMHCopula(double theta) { set_theta(theta); }

    // Constructs an AMH copula with a specified theta and marginal distributions.
    AMHCopula(double theta, std::shared_ptr<UnivariateDistributionBase> marginal_distribution_x_,
              std::shared_ptr<UnivariateDistributionBase> marginal_distribution_y_) {
        set_theta(theta);
        marginal_distribution_x = std::move(marginal_distribution_x_);
        marginal_distribution_y = std::move(marginal_distribution_y_);
    }

    CopulaType type() const override { return CopulaType::AliMikhailHaq; }

    double theta_minimum() const override { return -1.0; }
    double theta_maximum() const override { return 1.0; }

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
        return std::log((1.0 - theta() * (1.0 - t)) / t);
    }

    double generator_inverse(double t) const override {
        return (1.0 - theta()) / (std::exp(t) - theta());
    }

    double generator_prime(double t) const override {
        return (theta() - 1.0) / (t * (theta() * (t - 1.0) + 1.0));
    }

    double generator_prime2(double t) const override {
        double num = (theta() - 1.0) * (theta() * (2.0 * t - 1.0) + 1.0);
        double a = theta() * (t - 1.0) * t + t;
        double den = sign(a) * std::pow(std::fabs(a), 2.0);
        return -num / den;
    }

    double generator_prime_inverse(double t) const override {
        return corehydro::numerics::math::rootfinding::solve(
            [this, t](double x) { return generator_prime(x) - t; }, 0.0, 1.0);
    }

    double pdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        return (-1.0 + theta() * theta() * (-1.0 + u + v - u * v) -
                theta() * (-2.0 + u + v + u * v)) /
               std::pow(-1.0 + theta() * (-1.0 + u) * (-1.0 + v), 3.0);
    }

    // Johnson (1987, p.362) closed-form conditional quantile.
    double inverse_conditional_cdf(double u, double t) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double w = t;
        double b = 1.0 - u;
        double A = w * std::pow(theta() * b, 2.0) - theta();
        double B = theta() + 1.0 - 2.0 * theta() * b * w;
        double C = w - 1.0;
        double v = (-B + std::sqrt(B * B - 4.0 * A * C)) / 2.0 / A;
        return 1.0 - v;
    }

    std::array<double, 2> inverse_cdf(double u, double v) const override {
        return {u, inverse_conditional_cdf(u, v)};
    }

    // Gets the upper tail dependence coefficient lambda_U = 0. The AMH copula has no tail
    // dependence.
    double upper_tail_dependence() const override { return 0.0; }

    // Gets the lower tail dependence coefficient lambda_L = 0. The AMH copula has no tail
    // dependence.
    double lower_tail_dependence() const override { return 0.0; }

    std::unique_ptr<BivariateCopula> clone() const override {
        return std::make_unique<AMHCopula>(theta(), clone_marginal(marginal_distribution_x),
                                            clone_marginal(marginal_distribution_y));
    }

    // Estimates the dependency parameter using the method of moments.
    void set_theta_from_tau(const std::vector<double>& sample_data_x,
                             const std::vector<double>& sample_data_y) {
        double tau = corehydro::numerics::data::kendalls_tau(sample_data_x, sample_data_y);

        if ((tau < (5.0 - 8.0 * std::log(2.0)) / 3.0) || (tau > 1.0 / 3.0))
            throw std::runtime_error(
                "For the AMH copula, tau must be in [(5 - 8 log 2) / 3, 1 / 3] ~= [-0.1817, "
                "0.3333]. The dependency in the data is too strong to use the AMH copula.");

        double L = tau > 0.0 ? 0.001 : -1.0 + corehydro::numerics::kDoubleMachineEpsilon;
        double U = tau > 0.0 ? 1.0 - corehydro::numerics::kDoubleMachineEpsilon : -0.001;

        set_theta(corehydro::numerics::math::rootfinding::solve(
            [tau](double t) {
                double x = 1.0 - 2.0 * (std::pow(1.0 - t, 2.0) * std::log(-t + 1.0) + t) / (3.0 * t * t);
                return x - tau;
            },
            L, U));
    }

    math::linalg::Matrix2D parameter_constraints(const std::vector<double>&,
                                                  const std::vector<double>&) const override {
        return {{-1.0 + corehydro::numerics::kDoubleMachineEpsilon,
                 1.0 - corehydro::numerics::kDoubleMachineEpsilon}};
    }

   private:
    // Math.Sign(a): -1, 0, or 1 (distinct from Tools.Sign's Fortran-style 2-arg transfer
    // used by brent_search.hpp).
    static double sign(double a) { return a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : 0.0); }
};

}  // namespace corehydro::numerics::distributions::copulas
