// ported from: Numerics/Functions/PowerFunction.cs @ 2a0357a
//
// A power function with (optionally) normally distributed noise:
// Y = alpha * (X - xi)^beta * epsilon, epsilon ~ Normal(0, sigma) (log-space). `IsInverse` swaps
// which of Function()/InverseFunction() applies the forward power law vs. its algebraic inverse
// (see the four Function()/InverseFunction() branches below -- transcribed verbatim from the C#,
// not simplified, since IsInverse x IsDeterministic is a real 2x2, not two independent switches).
// `Minimum` is DERIVED from Xi (get returns Xi) and its setter throws -- Xi is the single source
// of truth for the function's lower domain bound, matching the C# `NotSupportedException`
// (-> std::logic_error, the port's convention for NotSupportedException/NotImplementedException).
// Holds a ported `distributions::Normal` (mean 0, scale sigma) for the confidence-level path, same
// as LinearFunction.
//
// Each parameter setter re-validates and caches `parameters_valid_` eagerly, matching the C#
// property setters verbatim (see linear_function.hpp's header for why a transiently-stale cache
// during construction is harmless: function()/inverse_function() re-validate against current
// state whenever the cached flag reads false). C# ArgumentOutOfRangeException -> std::out_of_range.
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/functions/i_univariate_function.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::functions {

class PowerFunction : public IUnivariateFunction {
   public:
    // Deterministic default: alpha=1, beta=1.5, xi=0.
    PowerFunction() {
        set_alpha(1.0);
        set_beta(1.5);
        set_xi(0.0);
        is_deterministic_ = true;
    }

    // Deterministic: given coefficient, exponent, and (optional, default 0) location.
    PowerFunction(double alpha, double beta, double xi = 0.0) {
        set_alpha(alpha);
        set_beta(beta);
        set_xi(xi);
        is_deterministic_ = true;
    }

    // Non-deterministic: given coefficient, exponent, location, and log-space standard error.
    PowerFunction(double alpha, double beta, double xi, double sigma) {
        set_alpha(alpha);
        set_beta(beta);
        set_sigma(sigma);
        set_xi(xi);
        is_deterministic_ = false;
    }

    double alpha() const { return alpha_; }
    void set_alpha(double value) {
        parameters_valid_ = validate_parameters({value, beta_, xi_, sigma_}, false);
        alpha_ = value;
    }

    double beta() const { return beta_; }
    void set_beta(double value) {
        parameters_valid_ = validate_parameters({alpha_, value, xi_, sigma_}, false);
        beta_ = value;
    }

    double xi() const { return xi_; }
    void set_xi(double value) {
        parameters_valid_ = validate_parameters({alpha_, beta_, value, sigma_}, false);
        xi_ = value;
    }

    double sigma() const { return sigma_; }
    void set_sigma(double value) {
        parameters_valid_ = validate_parameters({alpha_, beta_, xi_, value}, false);
        sigma_ = value;
        normal_.set_parameters(0.0, sigma_);
    }

    int number_of_parameters() const override { return 4; }
    bool parameters_valid() const override { return parameters_valid_; }

    // Derived from xi; the setter throws (NotSupportedException in C#).
    double minimum() const override { return xi_; }
    void set_minimum(double /*value*/) override {
        throw std::logic_error("Minimum is derived from Xi and cannot be set directly.");
    }
    double maximum() const override { return maximum_; }
    void set_maximum(double value) override { maximum_ = value; }

    std::vector<double> minimum_of_parameters() const override { return {0.0, -10.0, 0.0, 0.0}; }
    std::vector<double> maximum_of_parameters() const override {
        return {std::numeric_limits<double>::max(), 10.0, std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max()};
    }

    bool is_deterministic() const override { return is_deterministic_; }
    void set_is_deterministic(bool value) override { is_deterministic_ = value; }

    // Whether the power function should be inverted.
    bool is_inverse() const { return is_inverse_; }
    void set_is_inverse(bool value) { is_inverse_ = value; }

    double confidence_level() const override { return confidence_level_; }
    void set_confidence_level(double value) override { confidence_level_ = value; }

    void set_parameters(const std::vector<double>& parameters) override {
        parameters_valid_ = validate_parameters(parameters, false);
        alpha_ = parameters[0];
        beta_ = parameters[1];
        xi_ = parameters[2];
        sigma_ = parameters[3];
        normal_.set_parameters(0.0, sigma_);
    }

    bool validate_parameters(const std::vector<double>& parameters,
                             bool throw_on_error) const override {
        if (is_deterministic_ == false && parameters[3] <= 0.0) {
            if (throw_on_error)
                throw std::out_of_range("Standard error must be greater than zero.");
            return false;
        }
        return true;
    }

    double function(double x) const override {
        if (parameters_valid_ == false) validate_parameters({alpha_, beta_, xi_, sigma_}, true);

        if (x <= xi_)
            x = xi_ + corehydro::numerics::kDoubleMachineEpsilon;
        else if (x >= maximum_)
            x = maximum_;

        double y;
        if (is_inverse_) {
            if (is_deterministic_ || confidence_level_ < 0.0 || confidence_level_ > 1.0) {
                y = std::exp((std::log(x) - std::log(alpha_)) / beta_) + xi_;
            } else {
                y = std::exp((std::log(x) - std::log(alpha_) - normal_.inverse_cdf(confidence_level_)) /
                             beta_) +
                    xi_;
            }
        } else {
            if (is_deterministic_ || confidence_level_ < 0.0 || confidence_level_ > 1.0) {
                y = std::exp(std::log(alpha_) + beta_ * std::log(x - xi_));
            } else {
                y = std::exp(std::log(alpha_) + beta_ * std::log(x - xi_) +
                             normal_.inverse_cdf(confidence_level_));
            }
        }
        return y;
    }

    double inverse_function(double y) const override {
        if (parameters_valid_ == false) validate_parameters({alpha_, beta_, xi_, sigma_}, true);

        double x;
        if (is_inverse_) {
            if (is_deterministic_ || confidence_level_ < 0.0 || confidence_level_ > 1.0) {
                x = std::exp(std::log(alpha_) + beta_ * std::log(y - xi_));
            } else {
                x = std::exp(std::log(alpha_) + beta_ * std::log(y - xi_) +
                             normal_.inverse_cdf(confidence_level_));
            }
        } else {
            if (is_deterministic_ || confidence_level_ < 0.0 || confidence_level_ > 1.0) {
                x = std::exp((std::log(y) - std::log(alpha_)) / beta_) + xi_;
            } else {
                x = std::exp((std::log(y) - std::log(alpha_) - normal_.inverse_cdf(confidence_level_)) /
                             beta_) +
                    xi_;
            }
        }
        if (x < minimum()) return minimum();
        if (x > maximum_) return maximum_;
        return x;
    }

   private:
    bool parameters_valid_ = true;
    double alpha_ = 0.0, beta_ = 0.0, xi_ = 0.0, sigma_ = 0.0;
    double maximum_ = std::numeric_limits<double>::max();
    bool is_deterministic_ = false;
    bool is_inverse_ = false;
    double confidence_level_ = -1.0;
    distributions::Normal normal_;
};

}  // namespace corehydro::numerics::functions
