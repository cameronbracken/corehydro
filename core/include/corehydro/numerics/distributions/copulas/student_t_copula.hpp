// ported from: Numerics/Distributions/Bivariate Copulas/StudentTCopula.cs @ 7e8e8d1
//
// The bivariate Student's t elliptical copula. Extends BivariateCopula DIRECTLY (not
// ArchimedeanCopula -- the t-copula has no Archimedean generator, same as NormalCopula).
// TWO parameters: theta = rho in [-1, 1], plus degrees_of_freedom nu (continuous double,
// not rounded to an integer; validity floor nu > 2). number_of_copula_parameters() == 2,
// so bivariate_copula_estimation.hpp's generic MPL/IFM/MLE machinery automatically selects
// NelderMead (rather than 1D BrentSearch) for this type with zero estimation-layer changes.
//
// PDF is computed in log-space (log_gamma terms) for numerical stability, matching the C#
// source exactly. CDF builds a 2x2 MultivariateStudentT(nu, {0,0}, {{1,r},{r,1}}) and calls
// its CDF. InverseCDF uses the conditional-t_{nu+1} simulation formula. Tail dependence is
// symmetric (lambda_U == lambda_L) via the univariate StudentT CDF of the standard
// closed-form expression.
//
// Two ValidateParameter-family methods, mirroring the C# source exactly:
//   - `validate_parameter(double, bool)` overrides the BivariateCopula pure virtual and
//     simply delegates to `validate_parameters(parameter, nu_, throw_exception)` -- the
//     base's `set_theta` calls this (virtually) with only rho, so it must fold in the
//     CURRENT nu_ to validate the pair.
//   - `validate_parameters(double rho, double degrees_of_freedom, bool throw_exception)` is
//     an ADDITIONAL public member (not part of the BivariateCopula interface -- C#'s
//     ValidateParameters is likewise a StudentTCopula-only method, not on
//     IBivariateCopula/IArchimedeanCopula) that validates both parameters together and is
//     what set_degrees_of_freedom and set_copula_parameters actually call.
//
// Like NormalCopula (and unlike the Archimedean copulas), both ValidateParameter-family
// methods return std::nullopt (C# `null`) in their in-range branches -- StudentTCopula does
// NOT reproduce ArchimedeanCopula's (now-resolved) ParametersValid bug (see
// archimedean_copula.hpp). v2.1.4 added a NaN/Inf check on rho ahead of the range check in
// validate_parameters (mirrored below); the degrees-of-freedom NaN/Inf check already existed.
// Clone() deep-copies attached marginals via BivariateCopula::clone_marginal (v2.1.4, Task 8).
//
// Fidelity note (not exercised by the fixture): `set_copula_parameters`'s clamp uses
// `std::max(2.0 + 1e-10, parameters[1])`, a direct transcription of the C#
// `Math.Max(2.0 + 1E-10, parameters[1])`. The two are NOT equivalent for a NaN `df`: .NET's
// `Math.Max` explicitly special-cases NaN and returns NaN if either operand is NaN, while
// `std::max`'s default `operator<` comparator treats every comparison against NaN as false,
// so `std::max(2.0 + 1e-10, NaN)` returns `2.0 + 1e-10` (silently valid) rather than NaN
// (silently invalid, as in C#). This is a general `std::max`/`Math.Max` NaN-handling
// mismatch already latent wherever the port uses `std::max` for a C# `Math.Max` call (e.g.
// ClaytonCopula::cdf's `std::max(std::pow(u,-theta)+std::pow(v,-theta)-1.0, 0.0)`), not a
// StudentTCopula-specific bug; no fixture case constructs with `"df": "nan"` to avoid
// baking in a value that depends on this divergence.
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
#include "corehydro/numerics/distributions/multivariate/multivariate_student_t.hpp"
#include "corehydro/numerics/distributions/student_t.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions::copulas {

class StudentTCopula : public BivariateCopula {
   public:
    // Constructs a bivariate Student's t-copula with default parameters: rho = 0, nu = 5.
    // Mirrors the C# constructor's field/property assignment order exactly: `_nu` is set as
    // a plain field BEFORE `Theta` (whose setter validates against the now-current nu_).
    StudentTCopula() {
        nu_ = 5.0;
        set_theta(0.0);
    }

    // Constructs a bivariate Student's t-copula with specified correlation and degrees of
    // freedom.
    StudentTCopula(double rho, double degrees_of_freedom) {
        nu_ = degrees_of_freedom;
        set_theta(rho);
    }

    // Constructs a bivariate Student's t-copula with specified parameters and marginal
    // distributions.
    StudentTCopula(double rho, double degrees_of_freedom,
                   std::shared_ptr<UnivariateDistributionBase> marginal_distribution_x_,
                   std::shared_ptr<UnivariateDistributionBase> marginal_distribution_y_) {
        nu_ = degrees_of_freedom;
        set_theta(rho);
        marginal_distribution_x = std::move(marginal_distribution_x_);
        marginal_distribution_y = std::move(marginal_distribution_y_);
    }

    // Gets the degrees of freedom nu. Continuous; must be greater than 2.
    double degrees_of_freedom() const { return nu_; }

    // Sets the degrees of freedom nu, re-validating the (rho, nu) pair. Mirrors the C#
    // `DegreesOfFreedom` setter: validates against the CANDIDATE value (not yet assigned)
    // and the CURRENT theta, then assigns unconditionally (an out-of-range nu is stored,
    // only parameters_valid() flips false; it is not clamped or rejected here -- clamping
    // to > 2 only happens in set_copula_parameters, mirroring SetCopulaParameters).
    void set_degrees_of_freedom(double value) {
        parameters_valid_ = !validate_parameters(theta(), value, false).has_value();
        nu_ = value;
    }

    CopulaType type() const override { return CopulaType::StudentT; }

    double theta_minimum() const override { return -1.0; }
    double theta_maximum() const override { return 1.0; }

    // The base BivariateCopula::set_theta calls this (virtually) with only the candidate
    // rho; delegate to validate_parameters so parameters_valid() reflects both rho and the
    // CURRENT nu_.
    std::optional<std::string> validate_parameter(double parameter,
                                                    bool throw_exception) const override {
        return validate_parameters(parameter, nu_, throw_exception);
    }

    // Validates the correlation rho and degrees of freedom nu together. Returns nullopt
    // when both parameters are in their valid domains. NOT part of the BivariateCopula
    // interface (see file header).
    std::optional<std::string> validate_parameters(double rho, double degrees_of_freedom,
                                                     bool throw_exception) const {
        if (std::isnan(rho) || std::isinf(rho)) {
            std::string msg = "The correlation parameter must be finite.";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        if (rho < theta_minimum()) {
            std::string msg = "The correlation parameter rho (rho) must be greater than " +
                               std::to_string(theta_minimum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        if (rho > theta_maximum()) {
            std::string msg = "The correlation parameter rho (rho) must be less than " +
                               std::to_string(theta_maximum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        if (std::isnan(degrees_of_freedom) || std::isinf(degrees_of_freedom) ||
            degrees_of_freedom <= 2.0) {
            std::string msg = "The degrees of freedom must be greater than 2.";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        return std::nullopt;
    }

    int number_of_copula_parameters() const override { return 2; }
    std::vector<double> get_copula_parameters() const override { return {theta(), degrees_of_freedom()}; }

    // nu is clamped to the valid domain (nu > 2) but otherwise kept continuous -- no integer
    // rounding. The clamp uses 2 + 1e-10 because 2.0 + kDoubleMachineEpsilon rounds back to
    // 2.0 in IEEE 754 (the ULP at 2.0 is 2^-51, larger than epsilon = 2^-53).
    void set_copula_parameters(const std::vector<double>& parameters) override {
        set_theta(parameters[0]);
        set_degrees_of_freedom(std::max(2.0 + 1e-10, parameters[1]));
    }

    // The upper bound on nu is 30 rather than +inf: for nu >= ~30 the t-copula is
    // empirically indistinguishable from the Gaussian copula at typical hydrologic sample
    // sizes, so the likelihood is nearly flat past that point (see the C# source's remarks).
    math::linalg::Matrix2D parameter_constraints(const std::vector<double>&,
                                                  const std::vector<double>&) const override {
        return {{-1.0 + corehydro::numerics::kDoubleMachineEpsilon,
                 1.0 - corehydro::numerics::kDoubleMachineEpsilon},
                {2.0 + 1e-10, 30.0}};
    }

    // The t-copula density, computed in log-space for numerical stability:
    //   log c = LogGamma((nu+2)/2) + LogGamma(nu/2) - 2*LogGamma((nu+1)/2)
    //         - 0.5*log(1-rho^2)
    //         - ((nu+2)/2)*log(1 + Q/(nu*(1-rho^2)))
    //         + ((nu+1)/2)*log(1 + x1^2/nu) + ((nu+1)/2)*log(1 + x2^2/nu)
    // where x1 = t_nu^-1(u), x2 = t_nu^-1(v), Q = x1^2 - 2*rho*x1*x2 + x2^2.
    double pdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);

        double r = theta();
        double nu = nu_;

        StudentT t_dist(0.0, 1.0, nu_);
        double x1 = t_dist.inverse_cdf(u);
        double x2 = t_dist.inverse_cdf(v);

        double r2 = r * r;
        double log_c = math::special::log_gamma((nu + 2.0) / 2.0) +
                        math::special::log_gamma(nu / 2.0) -
                        2.0 * math::special::log_gamma((nu + 1.0) / 2.0) -
                        0.5 * std::log(1.0 - r2);

        double Q = x1 * x1 - 2.0 * r * x1 * x2 + x2 * x2;
        log_c -= ((nu + 2.0) / 2.0) * std::log(1.0 + Q / (nu * (1.0 - r2)));
        log_c += ((nu + 1.0) / 2.0) * std::log(1.0 + x1 * x1 / nu);
        log_c += ((nu + 1.0) / 2.0) * std::log(1.0 + x2 * x2 / nu);

        return std::exp(log_c);
    }

    // C(u, v) = F2(t_nu^-1(u), t_nu^-1(v); rho, nu) where F2 is the bivariate Student's t
    // CDF, evaluated via MultivariateStudentT.
    double cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);

        double r = theta();
        StudentT t_dist(0.0, 1.0, nu_);
        double x1 = t_dist.inverse_cdf(u);
        double x2 = t_dist.inverse_cdf(v);

        std::vector<std::vector<double>> scale_matrix = {{1.0, r}, {r, 1.0}};
        MultivariateStudentT mvt(nu_, std::vector<double>{0.0, 0.0}, scale_matrix);
        return mvt.cdf(std::vector<double>{x1, x2});
    }

    double conditional_cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double r = theta();
        double nu = nu_;
        StudentT t_nu(0.0, 1.0, nu);
        StudentT t_nu1(0.0, 1.0, nu + 1.0);
        double x1 = t_nu.inverse_cdf(u);
        double x2 = t_nu.inverse_cdf(v);
        double scale = std::sqrt((1.0 - r * r) * (nu + x1 * x1) / (nu + 1.0));
        return t_nu1.cdf((x2 - r * x1) / scale);
    }

    // Conditional-sampling InverseCDF using the conditional distribution of the bivariate
    // Student's t: X2 | X1 = x1 ~ t_{nu+1}(rho*x1, sqrt((1-rho^2)*(nu+x1^2)/(nu+1))).
    // 1) x1 = t_nu^-1(u).
    // 2) Sample from the conditional t_{nu+1} distribution using v.
    // 3) Transform the conditional sample back to uniform: v' = t_nu(x2).
    double inverse_conditional_cdf(double u, double t) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);

        double r = theta();
        double nu = nu_;

        StudentT t_nu(0.0, 1.0, nu_);
        double x1 = t_nu.inverse_cdf(u);

        StudentT t_nu1(0.0, 1.0, nu_ + 1.0);
        double z2 = t_nu1.inverse_cdf(t);

        double conditional_scale = std::sqrt((1.0 - r * r) * (nu + x1 * x1) / (nu + 1.0));
        double x2 = r * x1 + conditional_scale * z2;

        return t_nu.cdf(x2);
    }

    std::array<double, 2> inverse_cdf(double u, double v) const override {
        return {u, inverse_conditional_cdf(u, v)};
    }

    // The t-copula has symmetric upper and lower tail dependence:
    //   lambda_U = lambda_L = 2 * t_{nu+1}(-sqrt((nu+1)*(1-rho)/(1+rho)))
    // where t_{nu+1} is the CDF of the univariate Student's t with nu+1 degrees of freedom.
    // For rho = -1, lambda = 0. For rho = 1, lambda = 1. As nu -> infinity, lambda -> 0
    // (Normal copula limit).
    double upper_tail_dependence() const override {
        double r = theta();
        double nu = nu_;

        if (r >= 1.0) return 1.0;
        if (r <= -1.0) return 0.0;

        double arg = -std::sqrt((nu + 1.0) * (1.0 - r) / (1.0 + r));
        StudentT t_dist(0.0, 1.0, nu_ + 1.0);
        return 2.0 * t_dist.cdf(arg);
    }

    // The t-copula has symmetric tail dependence, so lambda_L = lambda_U.
    double lower_tail_dependence() const override { return upper_tail_dependence(); }

    std::unique_ptr<BivariateCopula> clone() const override {
        return std::make_unique<StudentTCopula>(theta(), nu_, clone_marginal(marginal_distribution_x),
                                                 clone_marginal(marginal_distribution_y));
    }

   private:
    double nu_ = 5.0;
};

}  // namespace corehydro::numerics::distributions::copulas
