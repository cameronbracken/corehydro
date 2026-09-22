// ported from: Numerics/Distributions/Multivariate/BivariateEmpirical.cs @ 7e8e8d1
//
// A bivariate empirical CDF defined on a rectangular X1 x X2 grid of cumulative
// probabilities, evaluated via 2D (bilinear) interpolation with independent optional
// transforms on X1, X2, and the probability surface.
//
// PDF is not implemented upstream. C# returns double.NaN so the missing joint density
// stays explicit rather than fabricating a value; ported verbatim.
//
// v2.1.4 (RESOLVED -- see docs/upstream-csharp-issues.md for the pre-fix history):
// set_parameters() now nulls the cached Bilinear interpolator (mirrors the new C#
// `bilinear = null;` at the end of `SetParameters`), so a second set_parameters() call
// after a cdf() call correctly rebuilds against the NEW grid instead of continuing to
// interpolate against the stale one. validate_parameters() also gained finite-value
// checks on x1/x2/p (NaN/Inf), checked immediately before each ascending-order/range
// check, in the same relative position as the new C# guards -- a NaN previously slipped
// through the ascending-order comparisons undetected (NaN compares false in both
// directions) and reported `parameters_valid() == true`.
#pragma once
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/interpolation/bilinear.hpp"
#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution.hpp"
#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution_type.hpp"

namespace corehydro::numerics::distributions {

class BivariateEmpirical : public MultivariateDistribution {
   public:
    // Constructs an empty Bivariate Empirical CDF (parameters set later via set_parameters).
    explicit BivariateEmpirical(data::Transform x1_transform_arg = data::Transform::None,
                                 data::Transform x2_transform_arg = data::Transform::None,
                                 data::Transform probability_transform_arg = data::Transform::None)
        : x1_transform(x1_transform_arg),
          x2_transform(x2_transform_arg),
          probability_transform(probability_transform_arg) {}

    // x1_values: primary values (rows of p_values). x2_values: secondary values (columns
    // of p_values). p_values: probability grid, 0 <= p <= 1.
    BivariateEmpirical(std::vector<double> x1_values, std::vector<double> x2_values,
                        std::vector<std::vector<double>> p_values,
                        data::Transform x1_transform_arg = data::Transform::None,
                        data::Transform x2_transform_arg = data::Transform::None,
                        data::Transform probability_transform_arg = data::Transform::None)
        : x1_transform(x1_transform_arg),
          x2_transform(x2_transform_arg),
          probability_transform(probability_transform_arg) {
        set_parameters(std::move(x1_values), std::move(x2_values), std::move(p_values));
    }

    // The interpolation transform for the X1-values, X2-values, and probability surface.
    data::Transform x1_transform = data::Transform::None;
    data::Transform x2_transform = data::Transform::None;
    data::Transform probability_transform = data::Transform::None;

    const std::vector<double>& x1_values() const { return x1_values_; }
    const std::vector<double>& x2_values() const { return x2_values_; }
    const std::vector<std::vector<double>>& probability_values() const { return p_values_; }

    // --- Identity / parameters ---
    int dimension() const override { return 2; }
    MultivariateDistributionType type() const override {
        return MultivariateDistributionType::BivariateEmpiricalDistribution;
    }
    std::string display_name() const override { return "Bivariate Empirical"; }
    std::string short_display_name() const override { return "Bi. Emp"; }
    bool parameters_valid() const override { return parameters_valid_; }

    // Returns an empty string if valid, else the (first) validation error message
    // (mirrors the C# ArgumentOutOfRangeException? return, minus the exception object).
    static std::string validate_parameters(const std::vector<double>& x1_values,
                                            const std::vector<double>& x2_values,
                                            const std::vector<std::vector<double>>& p_values) {
        if (x1_values.size() < 2) return "There must be at least 2 primary values.";
        if (x2_values.size() < 2) return "There must be at least 2 secondary values.";

        // Check if X1 and X2 are in ascending order
        for (double v : x1_values)
            if (!std::isfinite(v)) return "Primary values must be finite.";
        for (std::size_t i = 1; i < x1_values.size(); ++i)
            if (x1_values[i] <= x1_values[i - 1])
                return "Primary values must be in ascending order.";
        for (double v : x2_values)
            if (!std::isfinite(v)) return "Secondary values must be finite.";
        for (std::size_t i = 1; i < x2_values.size(); ++i)
            if (x2_values[i] <= x2_values[i - 1])
                return "Secondary values must be in ascending order.";

        // Check if probabilities are between 0 and 1 and in ascending order
        if (p_values.size() != x1_values.size())
            return "The number of rows in the probability array must be the same length "
                   "as the X1 array.";
        for (const auto& row : p_values) {
            if (row.size() != x2_values.size())
                return "The number of columns in the probability array must be the same "
                       "length as the X2 array.";
        }
        for (const auto& row : p_values)
            for (double v : row)
                if (!std::isfinite(v) || v < 0.0 || v > 1.0)
                    return "Probability values must be equal to or between 0 and 1.";

        return "";
    }

    // Set the distribution parameters. v2.1.4: nulls the cached bilinear interpolator
    // (mirrors the new C# `bilinear = null;`) so a subsequent cdf() rebuilds against the
    // NEW grid rather than continuing to interpolate against a stale cache.
    void set_parameters(std::vector<double> x1_values, std::vector<double> x2_values,
                         std::vector<std::vector<double>> p_values) {
        parameters_valid_ = validate_parameters(x1_values, x2_values, p_values).empty();
        x1_values_ = std::move(x1_values);
        x2_values_ = std::move(x2_values);
        p_values_ = std::move(p_values);
        bilinear_.reset();
    }

    // --- Distribution functions ---

    double pdf(const std::vector<double>& x) const override { return pdf(x[0], x[1]); }

    // Upstream stub: the PDF is not implemented (the C# comment notes it would need
    // numerical differentiation of the CDF); mirrors `return double.NaN;` verbatim.
    double pdf(double /*x1*/, double /*x2*/) const { return kNaN; }

    double cdf(const std::vector<double>& x) const override { return cdf(x[0], x[1]); }

    double cdf(double x1, double x2) const {
        // Validate parameters
        if (!parameters_valid_) {
            std::string err = validate_parameters(x1_values_, x2_values_, p_values_);
            if (!err.empty()) throw std::out_of_range(err);
        }
        // Make sure the transforms are up-to-date.
        if (!bilinear_)
            bilinear_ = std::make_unique<data::Bilinear>(x1_values_, x2_values_, p_values_);
        bilinear_->x1_transform = x1_transform;
        bilinear_->x2_transform = x2_transform;
        bilinear_->y_transform = probability_transform;
        return bilinear_->interpolate(x1, x2);
    }

    std::unique_ptr<MultivariateDistribution> clone() const override {
        return std::make_unique<BivariateEmpirical>(x1_values_, x2_values_, p_values_,
                                                     x1_transform, x2_transform,
                                                     probability_transform);
    }

   private:
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    std::vector<double> x1_values_;
    std::vector<double> x2_values_;
    std::vector<std::vector<double>> p_values_;
    bool parameters_valid_ = true;
    mutable std::unique_ptr<data::Bilinear> bilinear_;
};

}  // namespace corehydro::numerics::distributions
