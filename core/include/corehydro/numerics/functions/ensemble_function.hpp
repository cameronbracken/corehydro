// ported from: Numerics/Functions/EnsembleFunction.cs @ 7e8e8d1
#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/functions/i_univariate_function.hpp"
#include "corehydro/numerics/functions/tabular_function.hpp"

namespace corehydro::numerics::functions {

class EnsembleFunction {
   public:
    EnsembleFunction(std::unique_ptr<IUnivariateFunction> function_template,
                     std::vector<std::vector<double>> parameter_sets)
        : template_(std::move(function_template)), parameter_sets_(std::move(parameter_sets)) {
        if (!template_) throw std::invalid_argument("The template function is required.");
        if (dynamic_cast<TabularFunction*>(template_.get()) != nullptr)
            throw std::runtime_error(
                "TabularFunction templates cannot be configured from ensemble parameter sets.");
        if (!template_->clone())
            throw std::runtime_error("The template function type does not support cloning.");
        if (parameter_sets_.empty())
            throw std::invalid_argument("At least one parameter set is required.");
        for (std::size_t i = 0; i < parameter_sets_.size(); ++i) {
            const auto& values = parameter_sets_[i];
            if (values.size() != static_cast<std::size_t>(template_->number_of_parameters()))
                throw std::invalid_argument("An ensemble parameter set has the wrong length.");
            for (double value : values)
                if (!std::isfinite(value))
                    throw std::invalid_argument("An ensemble parameter set is non-finite.");
            auto candidate = template_->clone();
            if (!candidate->validate_parameters(values, false))
                throw std::invalid_argument("An ensemble parameter set is invalid.");
        }
    }

    int count() const { return static_cast<int>(parameter_sets_.size()); }

    std::unique_ptr<IUnivariateFunction> sample_at(int index) const {
        if (index < 0 || index >= count()) throw std::out_of_range("Posterior index is out of range.");
        auto result = template_->clone();
        result->set_parameters(parameter_sets_[static_cast<std::size_t>(index)]);
        if (!result->parameters_valid())
            result->validate_parameters(parameter_sets_[static_cast<std::size_t>(index)], true);
        return result;
    }

    std::unique_ptr<IUnivariateFunction> sample(double percentile) const {
        if (!std::isfinite(percentile) || percentile < 0.0 || percentile > 1.0)
            throw std::out_of_range("The percentile must be between 0 and 1.");
        int index = static_cast<int>(std::floor(percentile * count()));
        if (index >= count()) index = count() - 1;
        return sample_at(index);
    }

   private:
    std::unique_ptr<IUnivariateFunction> template_;
    std::vector<std::vector<double>> parameter_sets_;
};

}  // namespace corehydro::numerics::functions
