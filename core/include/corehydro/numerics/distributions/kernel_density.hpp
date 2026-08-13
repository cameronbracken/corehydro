// ported from: Numerics/Distributions/Univariate/KernelDensity.cs @ 2a0357a
//
// Re-audited against v2.1.4's "Harden distribution parameter validation" wave: the private
// Bandwidth ValidateParameters(double, bool) now rejects NaN/Infinity explicitly (previously
// only `value <= 0d`, which is false for +Infinity, so an infinite bandwidth could silently
// read as valid). The three constructors below previously computed
// `parameters_valid_ = bandwidth_ > 0.0 && !sample_data_.empty();` inline, which has the same
// gap (`Infinity > 0.0` is true); all three now route through the shared validate_bandwidth()
// helper, which adds the missing isnan/isinf rejection. (KernelDensity has no public
// Bandwidth setter in this port -- see the constructor-only note below -- so this is
// exercised via the explicit-bandwidth constructor's `construct.bandwidth` fixture field,
// not a post-construction mutation.)
//
// Kernel density estimation distribution. Mirrors the C# source method-for-method.
// IBootstrappable is not ported (desktop/uncertainty concern). The weighted (per-point
// weights) constructor and weighted PDF/moments are NOT ported (out of scope for the
// current port); only the three unweighted C# constructors are mirrored.
// C# uses Parallel.For for the PDF sum; the C++ port uses a sequential loop (no
// external threading dependency; CRAN/PyPI packages are single-threaded).
// Moments are the sample product-moments (mean/sd/skewness/kurtosis of the sample
// data), exactly as in the C# source (Statistics.ProductMoments).
// CDF/InverseCDF are built via a 1000-bin midpoint-rule integration over
// [Minimum, Maximum], normalized, and interpolated in NormalZ probability space using
// EmpiricalDistribution's RAW/trusted construction path (create_raw_table(), not the normal
// checked public constructor -- see create_cdf()'s own comment and empirical_distribution.hpp's
// friend declaration), mirroring C# CreateCDF, which builds a raw OrderedPairedData directly
// and never goes through EmpiricalDistribution/ValidateData at all.
// XTransform=None and ProbabilityTransform=NormalZ are the C# defaults, which are
// the defaults of EmpiricalDistribution (EmpiricalTransform::NormalZ).
// The Triangular kernel uses Triangular(-1,0,1).pdf(u) and the Uniform kernel uses
// Uniform(-1,1).pdf(u), mirroring the C# private nested classes exactly.
// No IEstimation / ILinearMomentEstimation: KernelDensity is non-parametric.
//
// T7 (factory completeness, v2.1.4): the C# parameterless `KernelDensity()` constructor
// (30 random Normal(0,1) draws via the shared/clock-seeded RNG, Gaussian kernel, auto
// bandwidth) is now ported too, so `UnivariateDistributionFactory::create_distribution`
// can add a `KernelDensity` case -- previously this class had no default constructor, and
// the factory had no case for it, matching the pre-2a0357a C# factory's own gap.
//
// T7 review fix (v2.1.4 ValidateData parity): the same task's EmpiricalDistribution
// ValidateData wave initially made `cdf_table_`'s construction go through the normal, checked
// EmpiricalDistribution constructor -- a corehydro-introduced regression, since real C#
// KernelDensity never touches EmpiricalDistribution/ValidateData at all (see create_cdf()'s
// comment). Fixed by routing through create_raw_table() instead; see
// test_kernel_density.cpp for the regression test (compact kernel + tiny bandwidth, cdf() must
// not throw).
#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/triangular.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

/// Kernel type enumeration (mirrors C# KernelDensity.KernelType).
enum class KernelType {
    Epanechnikov,  ///< 0.75(1-u^2) for |u|<=1, else 0.
    Gaussian,      ///< Standard normal PDF. Default.
    Triangular,    ///< Triangular(-1,0,1) PDF = (1-|u|) for |u|<=1, else 0.
    Uniform        ///< Uniform(-1,1) PDF = 0.5 for |u|<=1, else 0.
};

/// Kernel density estimation distribution.
/// Construct from sample data, an optional kernel type, and an optional bandwidth.
/// When bandwidth is omitted, Silverman's rule of thumb is applied.
class KernelDensity : public UnivariateDistributionBase {
   public:
    // --- Construction ------------------------------------------------------------------

    /// Constructs a Gaussian kernel density distribution from 30 random samples of a
    /// standard Normal distribution, using the default (Silverman) bandwidth. Mirrors C#
    /// `KernelDensity()` -- non-deterministic by design (an unseeded/clock-seeded draw), same
    /// as the C# default constructor; needed so `UnivariateDistributionFactory::create_distribution`
    /// has a default-constructible case for `KernelDensity` (v2.1.4 factory completeness --
    /// see univariate_distribution_factory.hpp). Not used by any oracle fixture.
    KernelDensity() : kernel_type_(KernelType::Gaussian) {
        init_sample(Normal().generate_random_values(30));
        bandwidth_ = bandwidth_rule(sample_data_);
        parameters_valid_ = validate_bandwidth(bandwidth_) && !sample_data_.empty();
    }

    /// Construct with Gaussian kernel and auto bandwidth.
    explicit KernelDensity(std::vector<double> sample_data)
        : kernel_type_(KernelType::Gaussian) {
        init_sample(std::move(sample_data));
        bandwidth_ = bandwidth_rule(sample_data_);
        parameters_valid_ = validate_bandwidth(bandwidth_) && !sample_data_.empty();
    }

    /// Construct with specified kernel and auto bandwidth.
    KernelDensity(std::vector<double> sample_data, KernelType kernel)
        : kernel_type_(kernel) {
        init_sample(std::move(sample_data));
        bandwidth_ = bandwidth_rule(sample_data_);
        parameters_valid_ = validate_bandwidth(bandwidth_) && !sample_data_.empty();
    }

    /// Construct with specified kernel and explicit bandwidth.
    KernelDensity(std::vector<double> sample_data, KernelType kernel, double bandwidth)
        : kernel_type_(kernel), bandwidth_(bandwidth) {
        init_sample(std::move(sample_data));
        parameters_valid_ = validate_bandwidth(bandwidth_) && !sample_data_.empty();
    }

    // --- Property accessors ------------------------------------------------------------

    KernelType kernel_type() const { return kernel_type_; }
    double bandwidth() const { return bandwidth_; }
    bool bounded_by_data() const { return bounded_by_data_; }

    /// Set whether Minimum/Maximum are clamped to the sample range.
    /// When true (default), Minimum=min(data), Maximum=max(data).
    /// When false, Minimum=min(data)-3*h, Maximum=max(data)+3*h.
    /// Changing this invalidates the internal CDF table.
    void set_bounded_by_data(bool v) {
        bounded_by_data_ = v;
        cdf_created_ = false;
    }

    const std::vector<double>& sample_data() const { return sample_data_; }
    int sample_size() const { return static_cast<int>(sample_data_.size()); }

    // --- Identity / parameters ---------------------------------------------------------

    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::KernelDensity;
    }

    /// Mirrors C# NumberOfParameters = 3 (data, kernel, bandwidth).
    int number_of_parameters() const override { return 3; }

    /// Returns empty: mirrors C# GetParameters → [].
    std::vector<double> get_parameters() const override { return {}; }

    /// Throws: KernelDensity cannot be set from a flat parameter vector.
    void set_parameters(const std::vector<double>& /*params*/) override {
        throw std::logic_error(
            "KernelDensity::set_parameters(vector) is not supported; "
            "use the sample-data constructor.");
    }

    // --- Support -----------------------------------------------------------------------

    /// Mirrors C# Minimum: data min when BoundedByData, else data min - 3*h.
    double minimum() const override {
        if (sample_data_.empty()) return kNaN;
        double mn = *std::min_element(sample_data_.begin(), sample_data_.end());
        return bounded_by_data_ ? mn : mn - 3.0 * bandwidth_;
    }

    /// Mirrors C# Maximum: data max when BoundedByData, else data max + 3*h.
    double maximum() const override {
        if (sample_data_.empty()) return kNaN;
        double mx = *std::max_element(sample_data_.begin(), sample_data_.end());
        return bounded_by_data_ ? mx : mx + 3.0 * bandwidth_;
    }

    // --- Moments -----------------------------------------------------------------------

    /// Sample mean (matches C# u1 from ProductMoments).
    double mean() const override { return u_[0]; }

    /// Sample standard deviation (matches C# u2 from ProductMoments).
    double standard_deviation() const override { return u_[1]; }

    /// Bias-corrected sample skewness (matches C# u3 from ProductMoments).
    double skewness() const override { return u_[2]; }

    /// Bias-corrected excess kurtosis (matches C# u4 from ProductMoments).
    double kurtosis() const override { return u_[3]; }

    /// Median via InverseCDF(0.5).
    double median() const override { return inverse_cdf(0.5); }

    /// Mode: BrentSearch maximizing the PDF over [InverseCDF(0.001), InverseCDF(0.999)],
    /// mirroring C# Mode (KernelDensity.cs line 320). This used to scan a 1000-point uniform
    /// grid, which landed one grid step away from the C# value.
    double mode() const override {
        double lo = inverse_cdf(0.001);
        double hi = inverse_cdf(0.999);
        if (lo >= hi) return 0.5 * (lo + hi);
        math::optimization::BrentSearch brent([this](double x) { return pdf(x); }, lo, hi);
        brent.maximize();
        return brent.best_parameter();
    }

    // --- Silverman bandwidth rule ------------------------------------------------------

    /// Silverman's rule of thumb: h = sigma * (4/(3n))^(1/5).
    /// Mirrors C# BandwidthRule(IList<double> sampleData) using Statistics.StandardDeviation.
    static double bandwidth_rule(const std::vector<double>& data) {
        if (data.empty()) return kNaN;
        auto m = data::product_moments(data);  // [mean, sample_sd, skew, kurt]
        double sigma = m[1];
        return sigma * std::pow(4.0 / (3.0 * static_cast<double>(data.size())), 0.2);
    }

    // --- Distribution functions --------------------------------------------------------

    /// PDF: (1/(n*h)) * sum_i K((x-xi)/h).
    /// Mirrors C# PDF(double x) — no bounds check; the sum is always computed.
    double pdf(double x) const override {
        if (sample_data_.empty()) return kNaN;
        double total = 0.0;
        const double h = bandwidth_;
        for (const double xi : sample_data_) {
            total += kernel_function((x - xi) / h);
        }
        return total / (static_cast<double>(sample_size()) * h);
    }

    /// CDF: boundary clamp then interpolation in the internal NormalZ table.
    /// Mirrors C# CDF(double x): x<Minimum→0, x>Maximum→1, else table lookup.
    double cdf(double x) const override {
        double lo = minimum(), hi = maximum();
        if (lo == hi) return kNaN;
        if (x < lo) return 0.0;
        if (x > hi) return 1.0;
        if (!cdf_created_) create_cdf();
        return cdf_table_->cdf(x);
    }

    /// InverseCDF: boundary guards then table lookup.
    /// Mirrors C# InverseCDF(double probability).
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        double lo = minimum(), hi = maximum();
        if (lo == hi) return kNaN;
        if (!cdf_created_) create_cdf();
        return cdf_table_->inverse_cdf(probability);
    }

    // --- Parameter display names (X1; C# KernelDensity.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Sample Data", "Kernel Type", "Bandwidth"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"Data()", "Kernel", "BW"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        auto c = std::make_unique<KernelDensity>(sample_data_, kernel_type_, bandwidth_);
        c->set_bounded_by_data(bounded_by_data_);
        return c;
    }

   private:
    std::vector<double> sample_data_;
    KernelType kernel_type_;
    double bandwidth_ = 1.0;
    bool bounded_by_data_ = true;
    double u_[4] = {kNaN, kNaN, kNaN, kNaN};

    mutable bool cdf_created_ = false;
    mutable std::unique_ptr<EmpiricalDistribution> cdf_table_;

    // Mirrors C# KernelDensity's private Bandwidth ValidateParameters(double, bool): the
    // bandwidth must be finite and strictly positive. Shared by all three constructors so
    // the NaN/Infinity rejection lives in exactly one place.
    static bool validate_bandwidth(double value) {
        return !std::isnan(value) && !std::isinf(value) && value > 0.0;
    }

    // Initialise from sample data: store and compute product moments.
    void init_sample(std::vector<double> data) {
        sample_data_ = std::move(data);
        if (!sample_data_.empty()) {
            auto m = data::product_moments(sample_data_);
            u_[0] = m[0]; u_[1] = m[1]; u_[2] = m[2]; u_[3] = m[3];
        }
        cdf_created_ = false;
    }

    // Kernel function K(u) — mirrors the C# private nested kernel classes.
    double kernel_function(double u) const {
        switch (kernel_type_) {
            case KernelType::Epanechnikov:
                // Mirrors C# EpanechnikovKernel: 0.75(1-u^2) for |u|<=1, else 0.
                return (std::fabs(u) <= 1.0) ? 0.75 * (1.0 - u * u) : 0.0;
            case KernelType::Gaussian:
                // Mirrors C# GaussianKernel: Normal.StandardPDF(u) = exp(-u^2/2)/sqrt(2pi).
                return std::exp(-0.5 * u * u) / corehydro::numerics::kSqrt2PI;
            case KernelType::Triangular:
                // Mirrors C# TriangularKernel: Triangular(-1,0,1).PDF(u).
                return Triangular(-1.0, 0.0, 1.0).pdf(u);
            case KernelType::Uniform:
                // Mirrors C# UniformKernel: Uniform(-1,1).PDF(u) = 0.5 for |u|<=1, else 0.
                return Uniform(-1.0, 1.0).pdf(u);
            default:
                return 0.0;
        }
    }

    // Build the internal CDF table. Mirrors C# CreateCDF():
    //   1. 1000 equal-width bins [Minimum, Maximum], midpoint-rule integration.
    //   2. Cumulative sum, normalized to 1.
    //   3. Stored as a raw (x, p) table -- real C# CreateCDF builds an OrderedPairedData
    //      directly here, NEVER an EmpiricalDistribution, so its CDF/InverseCDF never run
    //      EmpiricalDistribution's v2.1.4 ValidateData and can never throw on ties. This port
    //      has no separate OrderedPairedData type, so cdf_table_ uses EmpiricalDistribution's
    //      private create_raw_table() factory (KernelDensity is friended for exactly this) to
    //      reproduce that: for a compact-support kernel (Epanechnikov/Triangular/Uniform) with
    //      a small bandwidth, the density is exactly zero over long stretches between sample
    //      clusters, so this cumulative-sum p array WILL have tied runs -- the checked public
    //      EmpiricalDistribution constructor would (correctly, for every OTHER caller) mark
    //      that invalid and throw on use; that would be a corehydro-introduced divergence here,
    //      since real C# never throws for this case. See test_kernel_density.cpp for the
    //      regression coverage.
    void create_cdf() const {
        constexpr int kBins = 1000;
        const double lo = minimum(), hi = maximum();
        const double width = (hi - lo) / kBins;

        std::vector<double> xv(kBins), pv(kBins);

        // bin[i].Midpoint = lo + (i+0.5)*width, bin[i].Weight = width
        xv[0] = lo + 0.5 * width;
        pv[0] = pdf(xv[0]) * width;
        for (int i = 1; i < kBins; ++i) {
            xv[i] = lo + (i + 0.5) * width;
            pv[i] = pv[i - 1] + pdf(xv[i]) * width;
        }

        // Normalize so pv.back() = 1.
        const double total = pv[kBins - 1];
        if (total > 0.0) {
            for (double& p : pv) p /= total;
        }

        cdf_table_ = EmpiricalDistribution::create_raw_table(
            std::move(xv), std::move(pv), EmpiricalTransform::NormalZ, /*p_descending=*/false);
        cdf_created_ = true;
    }
};

}  // namespace corehydro::numerics::distributions
