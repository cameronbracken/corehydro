// ported from: Numerics/Functions/SegmentedPowerFunction.cs @ 7e8e8d1
#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/functions/i_univariate_function.hpp"
#include "corehydro/numerics/functions/support/segmented_power_math.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"

namespace corehydro::numerics::functions {

class SegmentedPowerFunction : public IUnivariateFunction {
   public:
    SegmentedPowerFunction() : SegmentedPowerFunction(1) {}

    explicit SegmentedPowerFunction(int number_of_segments)
        : number_of_segments_(number_of_segments) {
        if (number_of_segments < 1)
            throw std::out_of_range("The number of segments must be at least one.");
        parameters_.assign(static_cast<std::size_t>(3 * number_of_segments + 1), 0.0);
        for (int k = 0; k < number_of_segments_; ++k) {
            parameters_[static_cast<std::size_t>(3 * k)] = k;
            parameters_[static_cast<std::size_t>(3 * k + 2)] = 1.5;
        }
        parameters_.back() = 0.1;
        normal_.set_parameters(0.0, 0.1);
        is_deterministic_ = true;
    }

    explicit SegmentedPowerFunction(const std::vector<double>& parameters)
        : number_of_segments_(static_cast<int>((parameters.size() - 1) / 3)),
          parameters_(parameters.size(), 0.0) {
        if (parameters.size() < 4 || (parameters.size() - 1) % 3 != 0)
            throw std::invalid_argument("The parameter vector length must be 3*segments + 1.");
        is_deterministic_ = false;
        validate_parameters(parameters, true);
        set_parameters(parameters);
    }

    std::unique_ptr<IUnivariateFunction> clone() const override {
        return std::make_unique<SegmentedPowerFunction>(*this);
    }

    int number_of_segments() const { return number_of_segments_; }
    double sigma() const { return parameters_.back(); }
    double breakpoint(int segment_one_based) const { return segment_value(segment_one_based, 0); }
    double log10_alpha(int segment_one_based) const { return segment_value(segment_one_based, 1); }
    double beta(int segment_one_based) const { return segment_value(segment_one_based, 2); }

    int number_of_parameters() const override { return 3 * number_of_segments_ + 1; }
    bool parameters_valid() const override { return parameters_valid_; }
    double minimum() const override { return parameters_.front(); }
    void set_minimum(double) override {
        throw std::runtime_error("Minimum is derived from the first breakpoint.");
    }
    double maximum() const override { return maximum_; }
    void set_maximum(double value) override {
        if (!std::isfinite(value) || value <= minimum())
            throw std::out_of_range("Maximum must be finite and greater than the first breakpoint.");
        maximum_ = value;
    }
    std::vector<double> minimum_of_parameters() const override {
        std::vector<double> result(static_cast<std::size_t>(number_of_parameters()),
                                   std::numeric_limits<double>::lowest());
        for (int k = 0; k < number_of_segments_; ++k)
            result[static_cast<std::size_t>(3 * k + 2)] =
                std::numeric_limits<double>::epsilon();
        result.back() = 0.0;
        return result;
    }
    std::vector<double> maximum_of_parameters() const override {
        return std::vector<double>(static_cast<std::size_t>(number_of_parameters()),
                                   std::numeric_limits<double>::max());
    }
    bool is_deterministic() const override { return is_deterministic_; }
    void set_is_deterministic(bool value) override { is_deterministic_ = value; }
    double confidence_level() const override { return confidence_level_; }
    void set_confidence_level(double value) override { confidence_level_ = value; }

    void set_parameters(const std::vector<double>& parameters) override {
        if (!validate_parameters(parameters, false)) return;
        parameters_ = parameters;
        parameters_valid_ = true;
        normal_.set_parameters(0.0, parameters_.back());
    }

    bool validate_parameters(const std::vector<double>& parameters,
                             bool throw_on_error) const override {
        auto reject = [&](const char* message) {
            if (throw_on_error) throw std::out_of_range(message);
            return false;
        };
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters()))
            return reject("The parameter vector length must be 3*segments + 1.");
        for (double value : parameters)
            if (!std::isfinite(value)) return reject("All parameters must be finite.");
        for (int k = 0; k < number_of_segments_; ++k) {
            if (parameters[static_cast<std::size_t>(3 * k + 2)] <= 0.0)
                return reject("Exponents must be greater than zero.");
            if (k > 0 && parameters[static_cast<std::size_t>(3 * k)] <=
                             parameters[static_cast<std::size_t>(3 * (k - 1))])
                return reject("Breakpoints must be strictly increasing.");
        }
        if (parameters.front() >= maximum_)
            return reject("The first breakpoint must be less than Maximum.");
        if (!is_deterministic_ && parameters.back() <= 0.0)
            return reject("Standard error must be greater than zero.");
        return true;
    }

    double function(double x) const override {
        if (!parameters_valid_) validate_parameters(parameters_, true);
        if (x >= maximum_) x = maximum_;
        double q = deterministic_function(x);
        if (q <= 0.0 || is_deterministic_ || confidence_level_ < 0.0 || confidence_level_ > 1.0)
            return q;
        return q * std::pow(10.0, normal_.inverse_cdf(confidence_level_));
    }

    double inverse_function(double y) const override {
        if (!parameters_valid_) validate_parameters(parameters_, true);
        if (std::isnan(y) || y == -std::numeric_limits<double>::infinity())
            throw std::out_of_range("The inverse value must not be NaN or negative infinity.");
        if (y == std::numeric_limits<double>::infinity()) return maximum_;
        if (!is_deterministic_ && confidence_level_ >= 0.0 && confidence_level_ <= 1.0)
            y /= std::pow(10.0, normal_.inverse_cdf(confidence_level_));
        const double lower = minimum();
        if (y <= 0.0) return lower;
        double upper = parameters_[static_cast<std::size_t>(3 * (number_of_segments_ - 1))] + 1.0;
        if (upper <= lower) upper = lower + 1.0;
        while (deterministic_function(upper) < y) {
            if (upper >= maximum_) return maximum_;
            upper = lower + 2.0 * (upper - lower);
            if (upper > maximum_) upper = maximum_;
        }
        return math::rootfinding::solve(
            [&](double x) { return deterministic_function(x) - y; }, lower, upper);
    }

   private:
    double segment_value(int segment_one_based, int offset) const {
        if (segment_one_based < 1 || segment_one_based > number_of_segments_)
            throw std::out_of_range("Segment index is out of range.");
        return parameters_[static_cast<std::size_t>(3 * (segment_one_based - 1) + offset)];
    }

    double deterministic_function(double x) const {
        return support::segmented_power_addition(parameters_, number_of_segments_, x);
    }

    bool parameters_valid_ = true;
    int number_of_segments_;
    std::vector<double> parameters_;
    distributions::Normal normal_;
    bool is_deterministic_ = false;
    double confidence_level_ = -1.0;
    double maximum_ = std::numeric_limits<double>::max();
};

}  // namespace corehydro::numerics::functions
