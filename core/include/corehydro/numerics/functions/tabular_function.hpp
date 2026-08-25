// ported from: Numerics/Functions/TabularFunction.cs @ 2a0357a
//
// P4 Task 9: the third and last IUnivariateFunction implementation (see
// i_univariate_function.hpp), built entirely on the Paired Data subsystem -- an
// UncertainOrderedPairedData (uncertain_ordered_paired_data.hpp) holding the caller's uncertain
// curve, plus a sampled OrderedPairedData (ordered_paired_data.hpp) used for the actual
// Function()/InverseFunction() interpolation. Constructing a TabularFunction runs
// `paired_data.curve_sample()` (the MEAN curve) once up front; ConfidenceLevel's setter re-runs
// curve_sample at a specific probability instead.
//
// CONSTRUCTION divergence, unavoidable and inherent to this port's value semantics (see
// uncertain_ordinate.hpp's own header note): C#'s constructor stores the CALLER's
// UncertainOrderedPairedData BY REFERENCE (`_pairedData = pairedData;`, no clone) -- later
// external mutation of the caller's object is visible through the TabularFunction. This port's
// UncertainOrderedPairedData is a value type (its implicit copy constructor deep-clones every
// ordinate, see that class's own header note), and the constructor below takes its argument BY
// VALUE, so a TabularFunction owns an independent copy from the moment it is built; no such
// external-mutation visibility is possible in C++ here. Nothing in the ported test surface
// depends on it.
//
// `IsDeterministic`'s setter (C# lines ~68-85) rebuilds `_pairedData` via
// UncertainOrderedPairedData's third constructor -- see that constructor's own header note: it
// only re-labels the collection's `Distribution` metadata, it does NOT convert the individual Y
// distributions to Deterministic. That upstream quirk is preserved here unmodified.
#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/data/paired_data/ordered_paired_data.hpp"
#include "corehydro/numerics/data/paired_data/uncertain_ordered_paired_data.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/functions/i_univariate_function.hpp"

namespace corehydro::numerics::functions {

class TabularFunction : public IUnivariateFunction {
   public:
    // C# constructor (lines ~24-28). See header note above on the by-value-vs-by-reference
    // divergence.
    explicit TabularFunction(data::paired_data::UncertainOrderedPairedData paired_data)
        : paired_data_(std::move(paired_data)), opd_(paired_data_.curve_sample()) {}

    // C# PairedData (line ~35).
    const data::paired_data::UncertainOrderedPairedData& paired_data() const { return paired_data_; }

    // C# XTransform / YTransform (lines ~40-46). Default = None.
    data::Transform x_transform() const { return x_transform_; }
    void set_x_transform(data::Transform value) { x_transform_ = value; }
    data::Transform y_transform() const { return y_transform_; }
    void set_y_transform(data::Transform value) { y_transform_ = value; }

    int number_of_parameters() const override { return 1; }
    bool parameters_valid() const override { return paired_data_.is_valid(); }

    double minimum() const override { return minimum_; }
    void set_minimum(double value) override { minimum_ = value; }
    double maximum() const override { return maximum_; }
    void set_maximum(double value) override { maximum_ = value; }

    std::vector<double> minimum_of_parameters() const override {
        return {std::numeric_limits<double>::lowest()};
    }
    std::vector<double> maximum_of_parameters() const override {
        return {std::numeric_limits<double>::max()};
    }

    // C# IsDeterministic (lines ~67-85). See header note above for what the setter's rebuild
    // does (and does not do) to the Y distributions.
    bool is_deterministic() const override {
        return paired_data_.distribution() == distributions::UnivariateDistributionType::Deterministic;
    }
    void set_is_deterministic(bool value) override {
        if (value) {
            paired_data_ = data::paired_data::UncertainOrderedPairedData(
                paired_data_, paired_data_.strict_x(), paired_data_.order_x(), paired_data_.strict_y(),
                paired_data_.order_y(), distributions::UnivariateDistributionType::Deterministic);
        }
    }

    // C# ConfidenceLevel (lines ~90-104): re-runs curve_sample every time the level changes.
    double confidence_level() const override { return confidence_level_; }
    void set_confidence_level(double value) override {
        confidence_level_ = value;
        if (confidence_level_ < 0.0) {
            opd_ = paired_data_.curve_sample();
        } else {
            opd_ = paired_data_.curve_sample(confidence_level_);
        }
    }

    // C# AllowNegativeYValues (line ~110). Default = true.
    bool allow_negative_y_values() const { return allow_negative_y_values_; }
    void set_allow_negative_y_values(bool value) { allow_negative_y_values_ = value; }

    // C# SetParameters (lines ~115-119): not implemented -- the tabular function uses uncertain
    // paired data as its input, not a flat parameter vector. C# NotImplementedException -> this
    // port's std::logic_error convention (see power_function.hpp's header note).
    void set_parameters(const std::vector<double>& /*parameters*/) override {
        throw std::logic_error(
            "Not implemented: the tabular function uses uncertain paired data as its input.");
    }

    // C# ValidateParameters (lines ~123-132): ignores `parameters` entirely and validates the
    // paired data instead. C# ArgumentOutOfRangeException -> std::out_of_range (this port's
    // convention, see i_univariate_function.hpp's own header note).
    bool validate_parameters(const std::vector<double>& /*parameters*/,
                             bool throw_on_error) const override {
        auto errors = paired_data_.get_errors();
        if (!errors.empty()) {
            if (throw_on_error)
                throw std::out_of_range("The uncertain ordered paired data has errors.");
            return false;
        }
        return true;
    }

    // C# Function(double) (lines ~136-143).
    double function(double x) const override {
        if (!parameters_valid()) validate_parameters({0.0}, true);
        double y = opd_.get_y_from_x(x, x_transform_, y_transform_);
        if (!allow_negative_y_values_ && (std::isnan(y) || y < 0.0)) y = 0.0;
        return y;
    }

    // C# InverseFunction(double) (lines ~147-153).
    double inverse_function(double y) const override {
        if (!parameters_valid()) validate_parameters({0.0}, true);
        if (!allow_negative_y_values_ && (std::isnan(y) || y < 0.0)) y = 0.0;
        return opd_.get_x_from_y(y, x_transform_, y_transform_);
    }

   private:
    data::paired_data::UncertainOrderedPairedData paired_data_;
    data::paired_data::OrderedPairedData opd_;
    data::Transform x_transform_ = data::Transform::None;
    data::Transform y_transform_ = data::Transform::None;
    double minimum_ = std::numeric_limits<double>::lowest();
    double maximum_ = std::numeric_limits<double>::max();
    double confidence_level_ = -1.0;
    bool allow_negative_y_values_ = true;
};

}  // namespace corehydro::numerics::functions
