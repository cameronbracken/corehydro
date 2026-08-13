// ported from: Numerics/Distributions/Univariate/EmpiricalDistribution.cs @ 2a0357a
//
// Univariate Empirical distribution: a piecewise-linear CDF defined by (x, p) pairs.
// CDF and InverseCDF use linear interpolation in transformed space (default: NormalZ
// transform on probability values, mirroring C# ProbabilityTransform = Transform.NormalZ).
// PDF is the numerical first derivative of CDF using a two-point finite-difference with
// adaptive step size (mirrors C# NumericalDerivative.CalculateStepSize).
// Moments use the C# CentralMoments(300) trapezoidal scheme over
// [InverseCDF(1e-8), InverseCDF(1-1e-8)] with 300 equal-width bins.
// No IEstimation / ILinearMomentEstimation: EmpiricalDistribution is non-parametric and
// does not fit from data via the estimation interface (IBootstrappable not ported).
// The Convolve FFT static is skipped (external dependency); all other methods are faithful
// ports of the C# source including the boundary and clamping logic.
// NOTE: set_parameters(vector) throws (mirrors C# NotImplementedException), matching
// the fact that Empirical is constructed with structured x/p arrays, not a flat vector.
//
// X5 ADDITIVE PORT (Numerics EmpiricalDistribution.XTransform @ 2a0357a): the x-value transform
// (None / Logarithmic / NormalZ) was previously hardcoded to None. It is now a settable field
// (default None -> identical prior behavior, all existing fixtures byte-green), applied in the
// get_y_from_x / get_x_from_y interpolation exactly as OrderedPairedData.BaseInterpolate does.
// CompositeAnalysis's CreateEmpiricalCDF path builds Mixture/CompetingRisks empirical CDFs with
// XTransform = Logarithmic, so the two composite distributions need a log-space-capable backing.
//
// v2.1.4 (2a0357a) ValidateData wave: C# now runs a `ValidateData` check on every construction
// path and every `SetParameters` call, and gates MOST of it lazily -- `_parametersValid` is
// recomputed eagerly, but `CDF`/`InverseCDF` re-check it and throw (C# `ArgumentOutOfRangeException`)
// the first time the distribution is actually *used* while invalid, rather than throwing at
// construction time. The ONE exception is a length mismatch between `x` and `p`: C#'s
// `SetParameters` overloads throw `ArgumentException` EAGERLY, right at the call site, before
// `ValidateData` (and every other rule below) ever runs -- this port's constructor mirrors that
// exactly (see the (x, p, transform, p_descending) constructor below). Every OTHER rule stays
// lazy: at least two ordinates; `x` nondecreasing (C# `strictX` flipped `true` -> `false` at
// every call site in this diff, so duplicate/tied X values -- e.g. repeated flood values, or
// bootstrap resamples -- are now VALID, matching `OrderedPairedData`'s relaxed `StrictX =
// false`); every probability finite and in [0, 1]; and the probabilities strictly monotonic in
// the DECLARED direction (C#'s `OrderedPairedData` keeps `StrictY = true` always, and its
// `OrderY` is whatever the caller passed as `probabilityOrder` -- Ascending, the hardcoded
// default for the plain `SetParameters(x, p)` overload and the ordinary CDF case, or Descending,
// an explicit opt-in treated as a survival-function encoding and flipped via `1 - p` in CDF /
// `1 - probability` in InverseCDF; see `cdf()`/`inverse_cdf()` below). This is validated against
// the DECLARED order, not auto-detected from the data: a descending p array constructed without
// opting into Descending is INVALID (confirmed against the real C# via the dotnet oracle gate --
// an earlier auto-detect design in this port reproduced a false positive the gate caught).
// The C# `OrderedPairedData(orderedPairedData)` constructor overload additionally changed its
// ascending-X-violation throw from a plain `ArgumentException` to `ArgumentOutOfRangeException`
// -- this port has no separate `OrderedPairedData` type (the x/p tables are plain vectors owned
// directly by this class), so there is no equivalent second throw site to migrate; the single
// C++ convention below (`std::invalid_argument`, mirroring the C# `ArgumentOutOfRangeException`
// thrown lazily from `CDF`/`InverseCDF`) covers every invalid-data case uniformly.
//
// NOT EXPOSED (deliberately): C#'s 4-arg `SetParameters(x, p, XOrder, probabilityOrder)` lets
// the caller declare `XOrder` as `None` or `Descending` too, but ValidateData rejects both
// unconditionally (`data.OrderX != SortOrder.Ascending` is always an error) -- i.e. `XOrder` is
// only ever meaningfully `Ascending` in practice. This port's surface reflects that: there is no
// XOrder parameter at all (x is always validated as the Ascending case), only the `p_descending`
// axis above, which is the one C# axis that actually has an observable valid alternative.
#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

/// Transform applied to probability values during CDF/InverseCDF interpolation.
/// Mirrors the C# Transform enum (only the two variants used by EmpiricalDistribution).
enum class EmpiricalTransform {
    None,     ///< No transform — linear interpolation in raw probability space.
    NormalZ,  ///< Normal probability paper — interpolate in Normal.StandardZ space.
};

/// Univariate Empirical distribution.
/// Stores an ordered (x, p) CDF table and interpolates with optional transforms.
class EmpiricalDistribution : public UnivariateDistributionBase {
    // KernelDensity's internal CDF table is a genuine C# architectural exception: real C#
    // KernelDensity.CreateCDF builds a raw OrderedPairedData directly (never an
    // EmpiricalDistribution), so its CDF/InverseCDF never run ValidateData and can never throw
    // on tied cumulative probabilities -- which DOES happen for any compact-support kernel
    // (Epanechnikov/Triangular/Uniform) with a small bandwidth, since the density is exactly
    // zero over long stretches between sample clusters. This port has no separate
    // OrderedPairedData type, so KernelDensity is friended to reach the private
    // create_raw_table() factory below (bypassing validate_data() entirely, mirroring the raw
    // OPD's total lack of a ValidateData concept) instead of going through the normal, checked
    // public constructor every OTHER internal consumer (Mixture/CompetingRisks's
    // CreateEmpiricalCDF, which DOES construct a real EmpiricalDistribution in C# too, so they
    // correctly keep using the checked constructor -- see kernel_density.hpp's header note).
    friend class KernelDensity;

   public:
    // --- Construction ------------------------------------------------------------------

    /// Default constructor: x = {-0.5, 0, 0.5}, p = {0.1, 0.5, 0.9}, NormalZ transform.
    /// Mirrors C# public EmpiricalDistribution() with SetParameters([-0.5,0,0.5],[0.1,0.5,0.9]).
    EmpiricalDistribution() {
        set_xy({-0.5, 0.0, 0.5}, {0.1, 0.5, 0.9});
    }

    /// Construct from x and p arrays. `p_descending` opts into the survival-function encoding
    /// (mirrors C#'s `SetParameters(x, p, XOrder, probabilityOrder)` with `probabilityOrder =
    /// SortOrder.Descending`; default false matches the plain `SetParameters(x, p)` overload,
    /// which hardcodes `SortOrder.Ascending` for both XOrder and probabilityOrder). This is a
    /// DECLARED direction, not auto-detected from the data: a descending p array constructed
    /// with `p_descending = false` (the default) is INVALID -- C# validates against the
    /// caller-declared order, not the data's actual direction (confirmed against the real C#
    /// via the dotnet oracle gate: an ascending-declared-but-descending-data case fails
    /// ValidateData with "Y values must increase").
    ///
    /// A length mismatch between x and p throws EAGERLY, right here -- this mirrors C#'s
    /// `SetParameters(x, p[, XOrder, probabilityOrder])`, which throws `ArgumentException`
    /// immediately, before `ValidateData` (and thus before any of the OTHER rules below) ever
    /// runs. Every other ValidateData rule (too few ordinates, non-nondecreasing x, out-of-range/
    /// non-finite/non-monotonic p) stays LAZY exactly as C# has it -- construction succeeds with
    /// `parameters_valid() == false`, and cdf()/inverse_cdf() throw std::invalid_argument on
    /// first use (mirrors C#'s lazily-thrown `ArgumentOutOfRangeException`; see cdf()/
    /// inverse_cdf() below). This port's convention: std::invalid_argument for both throw sites,
    /// even though C# uses two different exception types (ArgumentException vs.
    /// ArgumentOutOfRangeException) for the eager-vs-lazy cases.
    EmpiricalDistribution(std::vector<double> x_values, std::vector<double> p_values,
                          EmpiricalTransform p_transform = EmpiricalTransform::NormalZ,
                          bool p_descending = false) {
        if (x_values.size() != p_values.size())
            throw std::invalid_argument(
                "EmpiricalDistribution: x and p arrays must have the same length");
        p_transform_ = p_transform;
        p_descending_ = p_descending;
        set_xy(std::move(x_values), std::move(p_values));
    }

    // --- Property accessors ---

    /// The probability transform used for CDF/InverseCDF interpolation.
    EmpiricalTransform probability_transform() const { return p_transform_; }
    void set_probability_transform(EmpiricalTransform t) {
        p_transform_ = t;
        moments_computed_ = false;
    }

    /// The x-value transform used for CDF/InverseCDF interpolation (X1-era addition; mirrors
    /// C# EmpiricalDistribution.XTransform, default None). None reproduces the prior behavior
    /// exactly, so all existing fixtures stay byte-green.
    data::Transform x_transform() const { return x_transform_; }
    void set_x_transform(data::Transform t) {
        x_transform_ = t;
        moments_computed_ = false;
    }

    const std::vector<double>& x_values() const { return x_; }
    const std::vector<double>& p_values() const { return p_; }

    // --- Identity / parameters ---------------------------------------------------------

    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Empirical;
    }

    /// Returns 2 (the parameter count attribute; the actual storage is the x/p arrays).
    int number_of_parameters() const override { return 2; }

    /// Returns empty: the distribution cannot be represented as a flat numeric vector.
    /// Mirrors C# GetParameters → return [].
    std::vector<double> get_parameters() const override { return {}; }

    /// Throws — EmpiricalDistribution cannot be set from a flat parameter vector.
    /// Mirrors C# SetParameters(IList<double>) → throw NotImplementedException().
    void set_parameters(const std::vector<double>& /*params*/) override {
        throw std::logic_error("EmpiricalDistribution::set_parameters(vector) is not supported; "
                               "use the (x, p) constructor.");
    }

    // --- Support -----------------------------------------------------------------------

    /// Mirrors C# Minimum = XValues.First().
    double minimum() const override { return x_.front(); }

    /// Mirrors C# Maximum = XValues.Last().
    double maximum() const override { return x_.back(); }

    // --- Moments -----------------------------------------------------------------------

    double mean() const override {
        if (!moments_computed_) compute_moments();
        return u_[0];
    }
    double median() const override { return inverse_cdf(0.5); }
    double mode() const override {
        // Mirrors C# Mode (EmpiricalDistribution.cs line 280): BrentSearch maximizing PDF over
        // [InverseCDF(0.001), InverseCDF(0.999)], returning BestParameterSet.Values[0]. This
        // used to scan a 1000-point uniform grid, which found a different local optimum of the
        // jagged finite-difference PDF than Brent does; Brent reproduces C# bit-for-bit.
        // The degenerate-interval guard is kept: this class validates lazily, so an invalid
        // parameter set can leave the two quantiles crossed, and the C# BrentSearch
        // constructor throws on upper < lower.
        double lo = inverse_cdf(0.001);
        double hi = inverse_cdf(0.999);
        if (lo >= hi) return 0.5 * (lo + hi);
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

    // --- Distribution functions --------------------------------------------------------

    /// Mirrors C# CDF(X): lazily throws if invalid (ValidateData, v2.1.4), then GetYFromX with
    /// XTransform=None, ProbabilityTransform, clamped [0,1]. A descending probability order
    /// (survival-function encoding) flips via 1-p, exactly as C# CDF does for
    /// `OrderY == SortOrder.Descending`.
    double cdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("EmpiricalDistribution: invalid parameters (nondecreasing "
                                        "x, matching-length finite p in [0,1], strictly monotonic p)");
        double raw = get_y_from_x(x);
        double p = p_descending_ ? 1.0 - raw : raw;
        return p < 0.0 ? 0.0 : p > 1.0 ? 1.0 : p;
    }

    /// Mirrors C# InverseCDF: lazily throws if invalid (ValidateData, v2.1.4), probability
    /// range guard, boundary shortcuts, then GetXFromY (1-probability when the probability
    /// order is descending, mirroring CDF's flip).
    double inverse_cdf(double probability) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("EmpiricalDistribution: invalid parameters (nondecreasing "
                                        "x, matching-length finite p in [0,1], strictly monotonic p)");
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability <= 1e-16) return minimum();
        if (probability >= 1.0 - 1e-16) return maximum();
        double x = p_descending_ ? get_x_from_y(1.0 - probability) : get_x_from_y(probability);
        double lo = minimum(), hi = maximum();
        return x < lo ? lo : x > hi ? hi : x;
    }

    /// Mirrors C# PDF(X): numerical derivative of CDF with adaptive step size.
    double pdf(double x) const override {
        if (x < minimum() || x > maximum()) return 0.0;
        double h = calculate_step_size(x);
        double dFdx = 0.0;
        if (x <= x_.front()) {
            // One-sided forward difference at left boundary
            double xb = x_.front();
            double hb = calculate_step_size(xb);
            dFdx = (cdf(xb + hb) - cdf(xb)) / hb;
        } else if (x >= x_.back()) {
            // One-sided backward difference at right boundary
            double xe = x_.back();
            double he = calculate_step_size(xe);
            dFdx = (cdf(xe) - cdf(xe - he)) / he;
        } else {
            // Central difference
            dFdx = (cdf(x + h) - cdf(x - h)) / (2.0 * h);
        }
        return dFdx < 0.0 ? 0.0 : dFdx;
    }

    // --- Parameter display names (X1; C# EmpiricalDistribution.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"X Values", "P Values"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"X()", "P()"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        auto c = std::make_unique<EmpiricalDistribution>(x_, p_, p_transform_, p_descending_);
        c->x_transform_ = x_transform_;
        return c;
    }

   private:
    std::vector<double> x_;  // nondecreasing x values (duplicates allowed, v2.1.4)
    std::vector<double> p_;  // strictly monotonic probability values (ascending or descending)
    EmpiricalTransform p_transform_ = EmpiricalTransform::NormalZ;
    data::Transform x_transform_ = data::Transform::None;  // mirrors C# XTransform (default None)

    // The DECLARED direction of p_ (mirrors the C# caller-configured `probabilityOrder`; an
    // explicit constructor parameter -- see the (x, p, transform, p_descending) constructor
    // above -- NOT auto-detected from the data). false (Ascending) for the common CDF case,
    // matching the 2-arg `SetParameters(x, p)` overload's hardcoded `SortOrder.Ascending`.
    bool p_descending_ = false;

    mutable bool moments_computed_ = false;
    mutable double u_[4] = {kNaN, kNaN, kNaN, kNaN};  // [mean, sd, skew, kurt]

    void set_xy(std::vector<double> x, std::vector<double> p) {
        x_ = std::move(x);
        p_ = std::move(p);
        moments_computed_ = false;
        parameters_valid_ = validate_data();
    }

    // --- Raw/trusted construction (KernelDensity only; see the friend declaration above) -----

    // Tag distinguishing the raw-table constructor below from the normal, checked public one
    // (same argument types otherwise). Private: only reachable via create_raw_table(), which is
    // itself private + friended to KernelDensity.
    struct RawTableTag {};

    // Skips validate_data() entirely and always reports parameters_valid() == true -- mirrors
    // the real C# KernelDensity.CreateCDF building a raw OrderedPairedData directly (which has
    // no ValidateData concept at all, so it can never throw on ties). x/p are trusted as-is;
    // the interpolation code (get_y_from_x/get_x_from_y/bisect_p) already tolerates ties
    // gracefully (the `if (y2 == y1) return ...` guards), matching what raw OPD interpolation
    // would do.
    EmpiricalDistribution(RawTableTag, std::vector<double> x, std::vector<double> p,
                          EmpiricalTransform p_transform, bool p_descending)
        : p_transform_(p_transform), p_descending_(p_descending) {
        x_ = std::move(x);
        p_ = std::move(p);
        parameters_valid_ = true;
    }

    // Private factory KernelDensity actually calls (the constructor itself is private, and
    // std::make_unique can't reach a private constructor even through a friend, so this
    // wraps `new` directly).
    static std::unique_ptr<EmpiricalDistribution> create_raw_table(
        std::vector<double> x, std::vector<double> p, EmpiricalTransform p_transform,
        bool p_descending) {
        return std::unique_ptr<EmpiricalDistribution>(new EmpiricalDistribution(
            RawTableTag{}, std::move(x), std::move(p), p_transform, p_descending));
    }

    // Mirrors C# EmpiricalDistribution.ValidateData (v2.1.4): at least two ordinates, matching
    // x/p lengths, nondecreasing x (strictX = false), finite probabilities in [0, 1], and
    // probabilities strictly monotonic in the DECLARED direction (p_descending_; strictY = true
    // always -- C# rejects anything else via its `OrderedPairedData.IsValid`/`GetErrors`
    // machinery, validated against the caller-declared `probabilityOrder`, not the data's actual
    // direction). cdf()/inverse_cdf() throw std::invalid_argument (mirrors C#'s lazily-thrown
    // ArgumentOutOfRangeException) rather than validating eagerly here -- EXCEPT the length
    // check just below, which can never actually be false when this is reached (the public
    // constructor already throws EAGERLY on a length mismatch, before set_xy()/validate_data()
    // ever run); it's kept only because C#'s own ValidateData has this exact same redundant
    // check (structural mirroring), not because it's reachable in this port.
    bool validate_data() const {
        const std::size_t n = x_.size();
        if (n < 2 || p_.size() != n) return false;

        for (std::size_t i = 1; i < n; ++i) {
            if (x_[i] < x_[i - 1]) return false;
        }
        for (double v : p_) {
            if (!std::isfinite(v) || v < 0.0 || v > 1.0) return false;
        }
        for (std::size_t i = 1; i < n; ++i) {
            bool ok = p_descending_ ? (p_[i] < p_[i - 1]) : (p_[i] > p_[i - 1]);
            if (!ok) return false;
        }
        return true;
    }

    // --- Interpolation helpers ---------------------------------------------------------

    // Apply probability transform (going into the interpolation space).
    double transform_p(double p) const {
        if (p_transform_ == EmpiricalTransform::NormalZ) {
            return Normal::standard_z(p);
        }
        return p;
    }

    // Invert probability transform (coming out of the interpolation space).
    double untransform_p(double z) const {
        if (p_transform_ == EmpiricalTransform::NormalZ) {
            return Normal::standard_cdf(z);
        }
        return z;
    }

    // Apply the x-value transform (going into the interpolation space). Mirrors the
    // OrderedPairedData.BaseInterpolate x-transform branch (None / Logarithmic via Tools.Log10 /
    // NormalZ via Normal.StandardZ).
    double transform_x(double x) const {
        switch (x_transform_) {
            case data::Transform::Logarithmic:
                return corehydro::numerics::clamped_log10(x);
            case data::Transform::NormalZ:
                return Normal::standard_z(x);
            default:
                return x;
        }
    }

    // Invert the x-value transform (coming out of the interpolation space).
    double untransform_x(double v) const {
        switch (x_transform_) {
            case data::Transform::Logarithmic:
                return std::pow(10.0, v);
            case data::Transform::NormalZ:
                return Normal::standard_cdf(v);
            default:
                return v;
        }
    }

    // Binary search: returns index i such that x_[i] <= query < x_[i+1].
    // Precondition: x_.front() < query < x_.back().
    int bisect_x(double query) const {
        int lo = 0, hi = static_cast<int>(x_.size()) - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (query >= x_[mid]) lo = mid; else hi = mid;
        }
        return lo;
    }

    // Binary search on p_ array. Direction-aware (p_descending_): ascending p_ compares
    // query >= p_[mid], descending p_ compares query <= p_[mid].
    int bisect_p(double query) const {
        int lo = 0, hi = static_cast<int>(p_.size()) - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            bool advance = p_descending_ ? (query <= p_[mid]) : (query >= p_[mid]);
            if (advance) lo = mid; else hi = mid;
        }
        return lo;
    }

    /// Mirrors OrderedPairedData.GetYFromX(x, XTransform=None, ProbabilityTransform).
    /// Boundary: x <= x[0] → p[0]; x >= x[n-1] → p[n-1]; otherwise linear interpolate.
    double get_y_from_x(double x) const {
        int n = static_cast<int>(x_.size());
        if (n == 0) return kNaN;
        if (n == 1) return p_[0];
        if (x <= x_[0]) return p_[0];
        if (x >= x_[n - 1]) return p_[n - 1];
        int i = bisect_x(x);
        // Apply the x-transform (transforms are monotonic increasing, so bisect_x on raw x
        // still selects the correct bracketing interval).
        double tx = transform_x(x);
        double x1 = transform_x(x_[i]), x2 = transform_x(x_[i + 1]);
        double y1 = transform_p(p_[i]), y2 = transform_p(p_[i + 1]);
        if (x2 == x1) return untransform_p(y1);
        double y = y1 + (tx - x1) / (x2 - x1) * (y2 - y1);
        return untransform_p(y);
    }

    /// Mirrors OrderedPairedData.GetXFromY(p, XTransform=None, ProbabilityTransform).
    /// Boundary depends on p_'s direction (p_descending_): ascending -> p<=p[0] gives x[0],
    /// p>=p[n-1] gives x[n-1]; descending -> the comparisons flip. Otherwise linear interpolate.
    double get_x_from_y(double prob) const {
        int n = static_cast<int>(p_.size());
        if (n == 0) return kNaN;
        if (n == 1) return x_[0];
        if (!p_descending_) {
            if (prob <= p_[0]) return x_[0];
            if (prob >= p_[n - 1]) return x_[n - 1];
        } else {
            if (prob >= p_[0]) return x_[0];
            if (prob <= p_[n - 1]) return x_[n - 1];
        }
        // Transform the query into the interpolation space.
        double y = transform_p(prob);
        int i = bisect_p(prob);
        double y1 = transform_p(p_[i]), y2 = transform_p(p_[i + 1]);
        double x1 = transform_x(x_[i]), x2 = transform_x(x_[i + 1]);
        if (y2 == y1) return untransform_x(x1);
        double x = x1 + (y - y1) / (y2 - y1) * (x2 - x1);
        // Back-transform out of the x interpolation space.
        return untransform_x(x);
    }

    // --- Numerical derivative step size -----------------------------------------------

    /// Mirrors NumericalDerivative.CalculateStepSize(x, order=1).
    /// h = eps^(1/(1+order)) * (1 + |x|) = sqrt(eps) * (1 + |x|) for first derivative.
    static double calculate_step_size(double x, int order = 1) {
        return std::pow(kDoubleMachineEpsilon, 1.0 / (1.0 + order)) * (1.0 + std::fabs(x));
    }

    // --- Moments (CentralMoments(300) from C# base class) ------------------------------

    /// Mirrors C# CentralMoments(int steps = 300): trapezoidal integration over
    /// 300 equal-width bins from InverseCDF(1e-8) to InverseCDF(1-1e-8).
    void compute_moments() const {
        const double a = inverse_cdf(1e-8);
        const double b = inverse_cdf(1.0 - 1e-8);
        if (a >= b) {
            u_[0] = a;
            u_[1] = u_[2] = u_[3] = kNaN;
            moments_computed_ = true;
            return;
        }

        const int steps = 300;
        // C# uses a "chain" bin structure: xl = LB, xu = xl + delta; next: xl = xu, xu = xl+delta.
        // This avoids floating-point drift. We reproduce it exactly.
        double delta = (b - a) / steps;
        double sum_u1 = 0.0, sum_u2 = 0.0;

        // Build bin boundaries using the chain approach to match C# exactly.
        // Store upper bounds for reuse.
        std::vector<double> ub(steps);
        {
            double xl = a;
            for (int i = 0; i < steps; ++i) {
                double xu = xl + delta;
                ub[i] = xu;
                xl = xu;
            }
        }

        // First pass: mean and standard deviation.
        // bin[0]: representative = UpperBound = ub[0]
        double df0 = cdf(ub[0]);
        sum_u1 += ub[0] * df0;
        sum_u2 += ub[0] * ub[0] * df0;

        // interior bins i=1..steps-2: representative = Midpoint
        for (int i = 1; i < steps - 1; ++i) {
            double lo = ub[i - 1];   // previous upper bound = current lower bound
            double hi = ub[i];
            double mid = 0.5 * (lo + hi);
            double df = cdf(hi) - cdf(lo);
            sum_u1 += mid * df;
            sum_u2 += mid * mid * df;
        }

        // last bin: representative = LowerBound = ub[steps-2]
        double lb_last = ub[steps - 2];
        double df_last = 1.0 - cdf(lb_last);
        sum_u1 += lb_last * df_last;
        sum_u2 += lb_last * lb_last * df_last;

        double u1 = sum_u1;
        double u2 = std::sqrt(sum_u2 - u1 * u1);

        // Second pass: skewness and kurtosis.
        double sum_u3 = 0.0, sum_u4 = 0.0;

        // bin[0]: UpperBound
        {
            double z = (ub[0] - u1) / u2;
            sum_u3 += z * z * z * df0;
            sum_u4 += z * z * z * z * df0;
        }
        // interior bins
        for (int i = 1; i < steps - 1; ++i) {
            double lo = ub[i - 1];
            double hi = ub[i];
            double mid = 0.5 * (lo + hi);
            double df = cdf(hi) - cdf(lo);
            double z = (mid - u1) / u2;
            sum_u3 += z * z * z * df;
            sum_u4 += z * z * z * z * df;
        }
        // last bin: LowerBound
        {
            double z = (lb_last - u1) / u2;
            sum_u3 += z * z * z * df_last;
            sum_u4 += z * z * z * z * df_last;
        }

        u_[0] = u1;
        u_[1] = u2;
        u_[2] = sum_u3;
        u_[3] = sum_u4;
        moments_computed_ = true;
    }
};

}  // namespace corehydro::numerics::distributions
