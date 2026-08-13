// ported from: Numerics/Distributions/Univariate/TruncatedDistribution.cs @ 2a0357a
//
// A general truncated probability distribution. Wraps any UnivariateDistributionBase
// and restricts it to [min, max]: PDF is renormalized by F(max)-F(min); CDF and InverseCDF
// are shifted accordingly. Moments use central_moments(1000), the fixed-step trapezoidal
// overload the C# integer literal in `CentralMoments(1000)` binds to (the four moment getters
// at TruncatedDistribution.cs lines 140, 171, 185, 199); it integrates over
// [InverseCDF(1e-8), InverseCDF(1-1e-8)] of the truncated distribution. This file used to
// integrate with adaptive Gauss-Kronrod over [min, max] instead.
// Mode is BrentSearch maximizing the truncated PDF over [InverseCDF(0.001), InverseCDF(0.999)]
// (TruncatedDistribution.cs line 154); it used to clamp the base distribution's mode into
// [min, max], which is wrong for a multimodal base and only approximate otherwise.
// type() delegates to the base distribution (mirrors C# `_baseDist.Type`).
// No IEstimation / ILinearMomentEstimation (TruncatedDistribution does not implement them in C#).
//
// Re-audited against v2.1.4's "Harden distribution parameter validation" wave: C# centralized
// validation into a private `ValidateParametersCore(parameters, throwException, out
// lowerProbability, out upperProbability)` consulted by BOTH the constructor and
// SetParameters (previously the constructor called `_baseDist.CDF(min)`/`CDF(max)` directly,
// unconditionally, with no validation at all). validate_parameters_core() below is the C++
// mirror: it validates the base params on a scratch CLONE (this port has no read-only,
// non-mutating `ValidateParameters(vector)` on the polymorphic base -- only the mutating
// set_parameters()+parameters_valid() pair -- so a clone stands in for C#'s non-mutating
// `_baseDist.ValidateParameters(baseParameters, false)` check), then the finite/ordered bounds
// check, then the same clone's CDF(min)/CDF(max) for the zero-mass check -- matching C#'s
// "validate on a clone, mutate the real base only after" sequencing. SetParameters also gained
// an explicit parameter-count check that THROWS before touching any state (previously any
// mismatch fell through to undefined std::vector iterator behavior); pdf/cdf/inverse_cdf now
// throw when `!parameters_valid_`, matching the C# guard `if (_parametersValid == false)
// ValidateParameters(GetParameters, true);` at the top of each. Regarding the base-parameter
// slice: this port already sliced `params.begin() .. params.begin() + n_base` correctly (a
// plain iterator range); the design-doc concern that C#'s own `Subset(0, count-3)` looked
// off-by-one turned out not to be a real divergence once `Subset(int, int)`'s INCLUSIVE-end-
// index convention is accounted for (see `Numerics/Utilities/ExtensionMethods.cs`), so there
// was no actual slicing bug to retire on either side.
// None of this is fixture-observable for the throwing behaviors (count mismatch, evaluating
// an invalid instance) -- the schema has no "expect throw" assertion -- so only the silent
// `parameters_valid` semantics (construct-then-SetParameters recovery) are fixture-tested; see
// truncated_distribution.json's sequential case.
#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/optimization/brent_search.hpp"

namespace corehydro::numerics::distributions {

class TruncatedDistribution : public UnivariateDistributionBase {
   public:
    /// Construct a truncated distribution from a cloned base + bounds.
    TruncatedDistribution(const UnivariateDistributionBase& base, double min_val, double max_val)
        : base_dist_(base.clone()), min_(min_val), max_(max_val) {
        init_from_current_state();
    }

    /// Construct from an rvalue unique_ptr (takes ownership).
    TruncatedDistribution(std::unique_ptr<UnivariateDistributionBase> base,
                          double min_val, double max_val)
        : base_dist_(std::move(base)), min_(min_val), max_(max_val) {
        init_from_current_state();
    }

    // Accessors matching C# properties
    const UnivariateDistributionBase& base_distribution() const { return *base_dist_; }
    double min_bound() const { return min_; }
    double max_bound() const { return max_; }

    // --- Identity / parameters ---
    // Mirrors C#: `public override UnivariateDistributionType Type => _baseDist.Type;`
    UnivariateDistributionType type() const override { return base_dist_->type(); }
    int number_of_parameters() const override { return base_dist_->number_of_parameters() + 2; }

    // {base_params..., min, max}  -- mirrors C# GetParameters
    std::vector<double> get_parameters() const override {
        auto p = base_dist_->get_parameters();
        p.push_back(min_);
        p.push_back(max_);
        return p;
    }

    // Mirrors C# SetParameters: first N-2 go to base; last 2 are min/max. Throws when the
    // flattened parameter count doesn't match (mirrors C#'s ArgumentException) -- checked
    // FIRST, before any state is touched.
    void set_parameters(const std::vector<double>& params) override {
        if (static_cast<int>(params.size()) != number_of_parameters()) {
            throw std::invalid_argument(
                "TruncatedDistribution: the length of the parameter array is invalid");
        }
        double lower_cdf, upper_cdf;
        bool valid = validate_parameters_core(params, lower_cdf, upper_cdf);
        int n_base = base_dist_->number_of_parameters();
        std::vector<double> base_params(params.begin(), params.begin() + n_base);
        // Mutate the real base only AFTER validating on the scratch clone above (mirrors C#:
        // `_baseDist.SetParameters(...)` runs unconditionally, even when validation failed).
        base_dist_->set_parameters(base_params);
        min_ = params[params.size() - 2];
        max_ = params[params.size() - 1];
        f_min_ = lower_cdf;
        f_max_ = upper_cdf;
        parameters_valid_ = valid;
        moments_computed_ = false;
    }

    // --- Moments / support ---
    double mean() const override {
        if (!moments_computed_) compute_moments();
        return u_[0];
    }
    double median() const override { return inverse_cdf(0.5); }
    double mode() const override {
        // Mirrors C# Mode (TruncatedDistribution.cs line 154): BrentSearch maximizing the
        // TRUNCATED PDF over [InverseCDF(0.001), InverseCDF(0.999)], returning
        // BestParameterSet.Values[0]. This used to clamp the base distribution's mode into
        // [min_, max_], which is wrong for a multimodal base and only approximate otherwise.
        double lo = inverse_cdf(0.001);
        double hi = inverse_cdf(0.999);
        math::optimization::BrentSearch brent([this](double x) { return pdf(x); }, lo, hi);
        brent.maximize();
        return brent.best_parameter();
    }
    double standard_deviation() const override {
        if (!moments_computed_) compute_moments();
        return u_[1];
    }
    double skewness() const override {
        if (!moments_computed_) compute_moments();
        return u_[2];
    }
    double kurtosis() const override {
        if (!moments_computed_) compute_moments();
        return u_[3];
    }
    // Mirrors C#: Minimum = Math.Max(_baseDist.Minimum, Min)
    double minimum() const override {
        return std::max(base_dist_->minimum(), min_);
    }
    // Mirrors C#: Maximum = Math.Min(_baseDist.Maximum, Max)
    double maximum() const override {
        return std::min(base_dist_->maximum(), max_);
    }

    // --- Distribution functions ---
    // Mirrors C#: PDF = basePDF(x) / (Fmax - Fmin) on [min, max], 0 outside. Mirrors C#'s
    // `if (_parametersValid == false) ValidateParameters(GetParameters, true);` guard.
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("TruncatedDistribution: invalid parameters");
        if (x < min_ || x > max_) return 0.0;
        return base_dist_->pdf(x) / (f_max_ - f_min_);
    }

    // Mirrors C#: CDF = (baseCDF(x) - Fmin) / (Fmax - Fmin)
    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("TruncatedDistribution: invalid parameters");
        if (x <= min_) return 0.0;
        if (x >= max_) return 1.0;
        return (base_dist_->cdf(x) - f_min_) / (f_max_ - f_min_);
    }

    // Mirrors C#: InverseCDF = base.InverseCDF(p * (Fmax - Fmin) + Fmin). C# checks
    // `_parametersValid` FIRST (before the probability-range check and the 0/1 early
    // returns) -- see Numerics/Distributions/Univariate/TruncatedDistribution.cs's
    // InverseCDF; matches pdf()/cdf() above, which already guard first.
    double inverse_cdf(double probability) const override {
        if (!parameters_valid_) throw std::invalid_argument("TruncatedDistribution: invalid parameters");
        if (probability < 0.0 || probability > 1.0)
            throw std::invalid_argument("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return base_dist_->inverse_cdf(probability * (f_max_ - f_min_) + f_min_);
    }

    // --- Parameter display names (X1; C# TruncatedDistribution.cs ParametersToString col0 +
    // ParameterNamesShortForm): the base distribution's names followed by "Min"/"Max" (C#
    // 78-107). ---
    std::vector<std::string> parameter_names() const override {
        std::vector<std::string> names = base_dist_->parameter_names();
        names.push_back("Min");
        names.push_back("Max");
        return names;
    }
    std::vector<std::string> parameter_names_short_form() const override {
        std::vector<std::string> names = base_dist_->parameter_names_short_form();
        names.push_back("Min");
        names.push_back("Max");
        return names;
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<TruncatedDistribution>(*base_dist_, min_, max_);
    }

   private:
    std::unique_ptr<UnivariateDistributionBase> base_dist_;
    double min_;
    double max_;
    double f_min_;  // baseCDF(min_)
    double f_max_;  // baseCDF(max_)

    // Lazy moment cache (mutable so const moment accessors can populate it).
    mutable bool moments_computed_ = false;
    mutable double u_[4] = {kNaN, kNaN, kNaN, kNaN};  // [mean, sd, skewness, kurtosis]

    // Builds the constructor's initial flat parameter vector (current base params + min_ +
    // max_) and validates it via validate_parameters_core, mirroring the C# constructor's
    // `ValidateParametersCore(parameters, false, out _Fmin, out _Fmax)` call.
    void init_from_current_state() {
        std::vector<double> params = base_dist_->get_parameters();
        params.push_back(min_);
        params.push_back(max_);
        parameters_valid_ = validate_parameters_core(params, f_min_, f_max_);
    }

    // Validates a flattened base-distribution parameter vector and its truncation bounds
    // WITHOUT mutating this object (mirrors C#'s private ValidateParametersCore). On
    // success, `lower_cdf`/`upper_cdf` receive the base distribution's CDF at min/max
    // (evaluated on a scratch clone at the CANDIDATE base params); on failure both are NaN.
    bool validate_parameters_core(const std::vector<double>& params,
                                   double& lower_cdf, double& upper_cdf) const {
        lower_cdf = kNaN;
        upper_cdf = kNaN;
        if (static_cast<int>(params.size()) != number_of_parameters()) return false;
        int n_base = base_dist_->number_of_parameters();
        std::vector<double> base_params(params.begin(), params.begin() + n_base);
        // Validate (and, if valid, evaluate CDF at the bounds) on a scratch clone -- this
        // port has no read-only ValidateParameters(vector) on the polymorphic base, only the
        // mutating set_parameters()+parameters_valid() pair, so a clone stands in for C#'s
        // non-mutating `_baseDist.ValidateParameters(baseParameters, false)` check.
        auto candidate = base_dist_->clone();
        candidate->set_parameters(base_params);
        if (!candidate->parameters_valid()) return false;
        double lo = params[params.size() - 2];
        double hi = params[params.size() - 1];
        if (std::isnan(lo) || std::isnan(hi) || std::isinf(lo) || std::isinf(hi) || lo >= hi) {
            return false;
        }
        lower_cdf = candidate->cdf(lo);
        upper_cdf = candidate->cdf(hi);
        if (std::fabs(lower_cdf - upper_cdf) < 1e-15) return false;
        return true;
    }

    // Mirrors the four C# moment getters (TruncatedDistribution.cs lines 140, 171, 185, 199),
    // each of which fills `u` from CentralMoments(1000). The integer literal binds to the base
    // class's fixed-step TRAPEZOIDAL overload (`CentralMoments(int steps)`), which integrates
    // over [InverseCDF(1e-8), InverseCDF(1-1e-8)] of the TRUNCATED distribution, not the
    // adaptive-tolerance one -- see docs/upstream-csharp-issues.md. This file used to call
    // adaptive Gauss-Kronrod over [min_, max_] instead.
    void compute_moments() const {
        auto mom = central_moments(1000);
        u_[0] = mom[0];
        u_[1] = mom[1];
        u_[2] = mom[2];
        u_[3] = mom[3];
        moments_computed_ = true;
    }
};

}  // namespace corehydro::numerics::distributions
