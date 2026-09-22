// ported from: Numerics/Functions/UnivariateFunctionFactory.cs @ 7e8e8d1
//
// The C# XML dispatch is represented by the shared JSON spec grammar used by the R, Python,
// fixture, and oracle runners. XML persistence remains outside the corehydro package boundary.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/functions/composite_function.hpp"
#include "corehydro/numerics/functions/linear_function.hpp"
#include "corehydro/numerics/functions/power_function.hpp"
#include "corehydro/numerics/functions/segmented_power_function.hpp"
#include "corehydro/numerics/functions/tabular_function.hpp"
#include "corehydro/numerics/functions/univariate_function_type.hpp"

namespace corehydro::numerics::functions {

inline std::unique_ptr<IUnivariateFunction> create_function(UnivariateFunctionType type) {
    switch (type) {
        case UnivariateFunctionType::Linear:
            return std::make_unique<LinearFunction>();
        case UnivariateFunctionType::Power:
            return std::make_unique<PowerFunction>();
        case UnivariateFunctionType::SegmentedPower:
            return std::make_unique<SegmentedPowerFunction>();
        case UnivariateFunctionType::Tabular:
            throw std::runtime_error("Tabular functions require paired data.");
        case UnivariateFunctionType::Composite:
            throw std::runtime_error("Composite functions require child functions.");
    }
    throw std::out_of_range("The function type is not defined.");
}

inline UnivariateFunctionType function_type(const IUnivariateFunction& function) {
    if (dynamic_cast<const LinearFunction*>(&function)) return UnivariateFunctionType::Linear;
    if (dynamic_cast<const PowerFunction*>(&function)) return UnivariateFunctionType::Power;
    if (dynamic_cast<const TabularFunction*>(&function)) return UnivariateFunctionType::Tabular;
    if (dynamic_cast<const SegmentedPowerFunction*>(&function))
        return UnivariateFunctionType::SegmentedPower;
    if (dynamic_cast<const CompositeFunction*>(&function)) return UnivariateFunctionType::Composite;
    throw std::runtime_error("Unknown univariate function type.");
}

inline std::unique_ptr<IUnivariateFunction> build_function_spec(
    const models::spec::JsonValue& spec) {
    const std::string type = spec.at("type").as_string();
    const bool deterministic = spec.value_or("is_deterministic", true);
    const double confidence = spec.value_or("confidence_level", -1.0);

    if (type == "linear") {
        auto result = std::make_unique<LinearFunction>();
        result->set_is_deterministic(deterministic);
        result->set_parameters(spec.at("parameters").as_double_vector());
        result->set_confidence_level(confidence);
        if (spec.contains("minimum")) result->set_minimum(spec.at("minimum").as_double());
        if (spec.contains("maximum")) result->set_maximum(spec.at("maximum").as_double());
        return result;
    }
    if (type == "power") {
        auto result = std::make_unique<PowerFunction>();
        result->set_is_deterministic(deterministic);
        result->set_is_inverse(spec.value_or("is_inverse", false));
        result->set_parameters(spec.at("parameters").as_double_vector());
        result->set_confidence_level(confidence);
        if (spec.contains("maximum")) result->set_maximum(spec.at("maximum").as_double());
        return result;
    }
    if (type == "segmented_power") {
        std::vector<double> parameters = spec.at("parameters").as_double_vector();
        if (parameters.size() < 4 || (parameters.size() - 1) % 3 != 0)
            throw std::invalid_argument("The parameter vector length must be 3*segments + 1.");
        auto result = std::make_unique<SegmentedPowerFunction>(
            static_cast<int>((parameters.size() - 1) / 3));
        result->set_is_deterministic(deterministic);
        result->set_parameters(parameters);
        result->set_confidence_level(confidence);
        if (spec.contains("maximum")) result->set_maximum(spec.at("maximum").as_double());
        return result;
    }
    if (type == "composite") {
        std::vector<std::unique_ptr<IUnivariateFunction>> children;
        for (const auto& child : spec.at("functions").items())
            children.push_back(build_function_spec(child));
        std::unique_ptr<CompositeFunction> result;
        if (spec.contains("weights"))
            result = std::make_unique<CompositeFunction>(
                std::move(children), spec.at("weights").as_double_vector());
        else
            result = std::make_unique<CompositeFunction>(std::move(children));
        const std::string mode = spec.value_or("mode", "weighted_average");
        if (mode == "weighted_average")
            result->set_mode(CompositeFunctionMode::WeightedAverage);
        else if (mode == "mixture")
            result->set_mode(CompositeFunctionMode::Mixture);
        else
            throw std::runtime_error("Unknown composite function mode: " + mode);
        result->set_confidence_level(confidence);
        return result;
    }
    if (type == "tabular")
        throw std::runtime_error("Tabular function specs are built by the paired-data runner.");
    throw std::runtime_error("Unknown function spec type: " + type);
}

}  // namespace corehydro::numerics::functions
