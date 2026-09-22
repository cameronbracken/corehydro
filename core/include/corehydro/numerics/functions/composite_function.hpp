// ported from: Numerics/Functions/CompositeFunction.cs @ 7e8e8d1
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/functions/composite_function_mode.hpp"
#include "corehydro/numerics/functions/i_univariate_function.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"

namespace corehydro::numerics::functions {

class CompositeFunction : public IUnivariateFunction {
   public:
    explicit CompositeFunction(std::vector<std::unique_ptr<IUnivariateFunction>> functions)
        : functions_(std::move(functions)) {
        validate_children();
        weights_.assign(functions_.size(), 1.0 / static_cast<double>(functions_.size()));
    }

    CompositeFunction(std::vector<std::unique_ptr<IUnivariateFunction>> functions,
                      std::vector<double> weights)
        : functions_(std::move(functions)), weights_(std::move(weights)) {
        validate_children();
        if (!validate_parameters(weights_, false))
            throw std::out_of_range("Composite weights are invalid.");
    }

    CompositeFunction(const CompositeFunction& other)
        : weights_(other.weights_), mode_(other.mode_),
          confidence_level_(other.confidence_level_), parameters_valid_(other.parameters_valid_) {
        functions_.reserve(other.functions_.size());
        for (const auto& child : other.functions_) {
            auto copy = child->clone();
            if (!copy) throw std::runtime_error("A child function does not support cloning.");
            functions_.push_back(std::move(copy));
        }
    }
    CompositeFunction& operator=(const CompositeFunction&) = delete;

    std::unique_ptr<IUnivariateFunction> clone() const override {
        return std::make_unique<CompositeFunction>(*this);
    }

    CompositeFunctionMode mode() const { return mode_; }
    void set_mode(CompositeFunctionMode value) { mode_ = value; }
    const std::vector<double>& weights() const { return weights_; }
    const std::vector<std::unique_ptr<IUnivariateFunction>>& functions() const { return functions_; }

    int number_of_parameters() const override { return static_cast<int>(functions_.size()); }
    bool parameters_valid() const override { return parameters_valid_; }
    double minimum() const override {
        double result = std::numeric_limits<double>::max();
        for (const auto& child : functions_) result = std::min(result, child->minimum());
        return result;
    }
    void set_minimum(double) override {
        throw std::runtime_error("Minimum is derived from child functions.");
    }
    double maximum() const override {
        double result = std::numeric_limits<double>::lowest();
        for (const auto& child : functions_) result = std::max(result, child->maximum());
        return result;
    }
    void set_maximum(double) override {
        throw std::runtime_error("Maximum is derived from child functions.");
    }
    std::vector<double> minimum_of_parameters() const override {
        return std::vector<double>(functions_.size(), 0.0);
    }
    std::vector<double> maximum_of_parameters() const override {
        return std::vector<double>(functions_.size(), 1.0);
    }
    bool is_deterministic() const override {
        for (const auto& child : functions_)
            if (!child->is_deterministic()) return false;
        if (mode_ == CompositeFunctionMode::Mixture) {
            int reachable = 0;
            for (double weight : weights_)
                if (weight > 0.0) ++reachable;
            return reachable <= 1;
        }
        return true;
    }
    void set_is_deterministic(bool value) override {
        for (auto& child : functions_) child->set_is_deterministic(value);
    }
    double confidence_level() const override { return confidence_level_; }
    void set_confidence_level(double value) override { confidence_level_ = value; }

    void set_parameters(const std::vector<double>& parameters) override {
        if (!validate_parameters(parameters, false)) return;
        weights_ = parameters;
        parameters_valid_ = true;
    }
    bool validate_parameters(const std::vector<double>& parameters,
                             bool throw_on_error) const override {
        auto reject = [&](const char* message) {
            if (throw_on_error) throw std::out_of_range(message);
            return false;
        };
        if (parameters.size() != functions_.size())
            return reject("The weight count must match the child function count.");
        double sum = 0.0;
        for (double weight : parameters) {
            if (!std::isfinite(weight) || weight < 0.0)
                return reject("Weights must be non-negative and finite.");
            sum += weight;
        }
        if (std::abs(sum - 1.0) > 1E-8) return reject("Weights must sum to one.");
        return true;
    }

    double function(double x) const override {
        if (!parameters_valid_) validate_parameters(weights_, true);
        if (is_deterministic() || confidence_level_ < 0.0 || confidence_level_ > 1.0)
            return weighted_configured(x, false);
        if (mode_ == CompositeFunctionMode::Mixture) {
            double remainder = 0.0;
            std::size_t index = select_child(confidence_level_, remainder);
            return evaluate_child(index, remainder, x, false);
        }
        double result = 0.0;
        for (std::size_t i = 0; i < functions_.size(); ++i)
            result += weights_[i] * evaluate_child(i, confidence_level_, x, false);
        return result;
    }

    double inverse_function(double y) const override {
        if (!std::isfinite(y)) throw std::out_of_range("The inverse value must be finite.");
        if (mode_ == CompositeFunctionMode::Mixture && !is_deterministic() &&
            confidence_level_ >= 0.0 && confidence_level_ <= 1.0) {
            double remainder = 0.0;
            std::size_t index = select_child(confidence_level_, remainder);
            return evaluate_child(index, remainder, y, true);
        }
        double lower = minimum(), upper = maximum();
        if (!std::isfinite(lower) || lower == std::numeric_limits<double>::lowest()) lower = -1.0;
        if (!std::isfinite(upper) || upper == std::numeric_limits<double>::max()) upper = 1.0;
        for (int i = 0; function(lower) > y && i < 128; ++i) lower = lower < 0 ? 2 * lower : 2 * (lower - 1);
        for (int i = 0; function(upper) < y && i < 128; ++i) upper = upper > 0 ? 2 * upper : 2 * (upper + 1);
        if (function(lower) > y || function(upper) < y)
            throw std::out_of_range("The value is outside the invertible range.");
        return math::rootfinding::solve([&](double x) { return function(x) - y; }, lower, upper);
    }

   private:
    void validate_children() const {
        if (functions_.empty()) throw std::invalid_argument("At least one child function is required.");
        for (const auto& child : functions_)
            if (!child) throw std::invalid_argument("Child functions cannot contain null entries.");
    }

    double weighted_configured(double value, bool inverse) const {
        double result = 0.0;
        for (std::size_t i = 0; i < functions_.size(); ++i) {
            auto child = functions_[i]->clone();
            if (child) {
                result += weights_[i] *
                          (inverse ? child->inverse_function(value) : child->function(value));
            } else {
                std::lock_guard<std::mutex> lock(external_child_mutex_);
                result += weights_[i] * (inverse ? functions_[i]->inverse_function(value)
                                                : functions_[i]->function(value));
            }
        }
        return result;
    }

    double evaluate_child(std::size_t index, double level, double value, bool inverse) const {
        auto child = functions_[index]->clone();
        if (child) {
            child->set_confidence_level(level);
            return inverse ? child->inverse_function(value) : child->function(value);
        }
        std::lock_guard<std::mutex> lock(external_child_mutex_);
        double previous = functions_[index]->confidence_level();
        functions_[index]->set_confidence_level(level);
        try {
            double result = inverse ? functions_[index]->inverse_function(value)
                                    : functions_[index]->function(value);
            functions_[index]->set_confidence_level(previous);
            return result;
        } catch (...) {
            functions_[index]->set_confidence_level(previous);
            throw;
        }
    }

    std::size_t select_child(double level, double& remainder) const {
        double cumulative = 0.0;
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            double next = cumulative + weights_[i];
            if (weights_[i] > 0.0 && (level <= next || i + 1 == weights_.size())) {
                remainder = (level - cumulative) / weights_[i];
                remainder = std::clamp(remainder, 0.0, 1.0);
                return i;
            }
            cumulative = next;
        }
        throw std::runtime_error("Composite mixture has no reachable child.");
    }

    std::vector<std::unique_ptr<IUnivariateFunction>> functions_;
    std::vector<double> weights_;
    CompositeFunctionMode mode_ = CompositeFunctionMode::WeightedAverage;
    double confidence_level_ = -1.0;
    bool parameters_valid_ = true;
    inline static std::mutex external_child_mutex_;
};

}  // namespace corehydro::numerics::functions
