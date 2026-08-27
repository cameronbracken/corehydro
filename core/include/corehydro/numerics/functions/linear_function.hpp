// ported from: Numerics/Functions/LinearFunction.cs @ 2a0357a
//
// A simple linear function with a single predictor, slope and intercept, and (optionally)
// normally distributed noise: Y = alpha + beta*X + epsilon, epsilon ~ Normal(0, sigma). Three
// constructors mirror the three C# ctors: deterministic default (alpha=0, beta=1), deterministic
// (alpha, beta), and non-deterministic (alpha, beta, sigma). Holds a ported `distributions::Normal`
// (mean 0, scale sigma) for the confidence-level path -- function()/inverse_function() add/remove
// `normal_.inverse_cdf(confidence_level_)` exactly as the C# does.
//
// Each parameter setter re-validates and caches `parameters_valid_` eagerly, matching the C#
// property setters verbatim -- including the same transient staleness during construction (a
// setter can cache `false` against a not-yet-fully-initialized IsDeterministic before the ctor's
// own trailing `IsDeterministic = ...` line runs). This is harmless: function()/inverse_function()
// re-run validate_parameters() against the CURRENT (fully-constructed) state whenever the cached
// flag reads false, so a stale cache only costs one extra validation call, never a wrong answer.
// C# ArgumentOutOfRangeException -> std::out_of_range (matching the sibling link-function headers'
// convention); C# InvalidOperationException (InverseFunction's Beta==0 guard) -> std::runtime_error.
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/functions/i_univariate_function.hpp"

namespace corehydro::numerics::functions {

class LinearFunction : public IUnivariateFunction {
   public:
    // Deterministic default: intercept 0, slope 1.
    LinearFunction() {
        set_alpha(0.0);
        set_beta(1.0);
        is_deterministic_ = true;
    }

    // Deterministic: given intercept and slope.
    LinearFunction(double alpha, double beta) {
        set_alpha(alpha);
        set_beta(beta);
        is_deterministic_ = true;
    }

    // Non-deterministic: given intercept, slope, and standard error.
    LinearFunction(double alpha, double beta, double sigma) {
        set_alpha(alpha);
        set_beta(beta);
        set_sigma(sigma);
        is_deterministic_ = false;
    }

    double alpha() const { return alpha_; }
    void set_alpha(double value) {
        parameters_valid_ = validate_parameters({value, beta_, sigma_}, false);
        alpha_ = value;
    }

    double beta() const { return beta_; }
    void set_beta(double value) {
        parameters_valid_ = validate_parameters({alpha_, value, sigma_}, false);
        beta_ = value;
    }

    double sigma() const { return sigma_; }
    void set_sigma(double value) {
        parameters_valid_ = validate_parameters({alpha_, beta_, value}, false);
        sigma_ = value;
        normal_.set_parameters(0.0, sigma_);
    }

    int number_of_parameters() const override { return 3; }
    bool parameters_valid() const override { return parameters_valid_; }

    double minimum() const override { return minimum_; }
    void set_minimum(double value) override { minimum_ = value; }
    double maximum() const override { return maximum_; }
    void set_maximum(double value) override { maximum_ = value; }

    std::vector<double> minimum_of_parameters() const override {
        return {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), 0.0};
    }
    std::vector<double> maximum_of_parameters() const override {
        return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max()};
    }

    bool is_deterministic() const override { return is_deterministic_; }
    void set_is_deterministic(bool value) override { is_deterministic_ = value; }

    double confidence_level() const override { return confidence_level_; }
    void set_confidence_level(double value) override { confidence_level_ = value; }

    void set_parameters(const std::vector<double>& parameters) override {
        parameters_valid_ = validate_parameters(parameters, false);
        alpha_ = parameters[0];
        beta_ = parameters[1];
        sigma_ = parameters[2];
        normal_.set_parameters(0.0, sigma_);
    }

    bool validate_parameters(const std::vector<double>& parameters,
                             bool throw_on_error) const override {
        if (is_deterministic_ == false && parameters[2] <= 0.0) {
            if (throw_on_error)
                throw std::out_of_range("Standard error must be greater than zero.");
            return false;
        }
        return true;
    }

    double function(double x) const override {
        if (parameters_valid_ == false) validate_parameters({alpha_, beta_, sigma_}, true);

        if (x <= minimum_) {
            if (is_deterministic_ || confidence_level_ < 0.0 || confidence_level_ > 1.0)
                return alpha_ + beta_ * minimum_;
            return alpha_ + beta_ * minimum_ + normal_.inverse_cdf(confidence_level_);
        }
        if (x >= maximum_) {
            if (is_deterministic_ || confidence_level_ < 0.0 || confidence_level_ > 1.0)
                return alpha_ + beta_ * maximum_;
            return alpha_ + beta_ * maximum_ + normal_.inverse_cdf(confidence_level_);
        }
        if (is_deterministic_ || confidence_level_ < 0.0 || confidence_level_ > 1.0)
            return alpha_ + beta_ * x;
        return alpha_ + beta_ * x + normal_.inverse_cdf(confidence_level_);
    }

    double inverse_function(double y) const override {
        if (parameters_valid_ == false) validate_parameters({alpha_, beta_, sigma_}, true);
        // C# `double.Epsilon` is the smallest positive double (denormalized min), not machine
        // epsilon.
        if (std::fabs(beta_) < std::numeric_limits<double>::denorm_min())
            throw std::runtime_error("Cannot compute inverse function when Beta is zero.");

        double x;
        if (is_deterministic_ || confidence_level_ < 0.0 || confidence_level_ > 1.0) {
            x = (y - alpha_) / beta_;
        } else {
            x = (y - alpha_ - normal_.inverse_cdf(confidence_level_)) / beta_;
        }
        if (x < minimum_) return minimum_;
        if (x > maximum_) return maximum_;
        return x;
    }

   private:
    bool parameters_valid_ = true;
    double alpha_ = 0.0, beta_ = 0.0, sigma_ = 0.0;
    double minimum_ = std::numeric_limits<double>::lowest();
    double maximum_ = std::numeric_limits<double>::max();
    bool is_deterministic_ = false;
    double confidence_level_ = -1.0;
    distributions::Normal normal_;
};

}  // namespace corehydro::numerics::functions
