// ported from: Numerics/Data/Statistics/RunningStatistics.cs @ 2a0357a
//
// Welford's online algorithm for the first four central moments (mean, variance,
// skewness, kurtosis), ported verbatim -- including the exact operator-precedence
// grouping of the sample/population skewness and kurtosis formulas and the m2/m3/m4
// update recurrences in push(), all of which mix `long`-typed sample-count arithmetic
// with `double` moment arithmetic in ways that matter bit-for-bit (C# converts a `long`
// operand to `double` only at the point it first combines with a `double`, so
// parenthesized all-`long` subexpressions are computed as exact integers first). ARWMH's
// step-size adaptation and Bootstrap's summary statistics consume this class.
//
// Member order mirrors the C# source: default ctor, values-list ctor, `count()`,
// `minimum()`, `maximum()`, `mean()`, `variance()`, `population_variance()`,
// `standard_deviation()`, `population_standard_deviation()`, `coefficient_of_variation()`,
// `skewness()`, `population_skewness()`, `kurtosis()`, `population_kurtosis()`,
// `push(double)`, `push(vector<double>)`, `clone()`, static `combine()`, `operator+`.
//
// v2.1.4 sync (Numerics 41f78b4): ADDITIVE, not a fix -- C# added a public `Clone()` method
// and changed `Combine(a, b)` to return `b.Clone()`/`a.Clone()` (rather than the bare `b`/
// `a` reference) when the other operand is empty, so a caller mutating the returned
// instance can no longer accidentally mutate one of the two inputs. This was never
// observable in this C++ port: `combine()`'s `return b;`/`return a;` already returned an
// independent copy (C++ value semantics -- there is no reference to alias, unlike a C#
// class instance), so no behavior changed here. `clone()` is added below purely for API
// parity with the new C# method, and `combine()` now calls it explicitly (mirroring the C#
// line-for-line) even though `return *this;` was already equivalent.
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace corehydro::numerics::data {

namespace detail {
// Math.Min(double, double) narrow port. The BCL's Math.Min propagates NaN (returns NaN
// if either argument is NaN); plain `std::min` does NOT (a comparison against NaN is
// always false, so `std::min(a, b)` degrades to "return a" when b is NaN). Used by
// combine() below to transcribe C#'s `Math.Min(a._min, b._min)`.
inline double nan_min(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return b < a ? b : a;
}

// Math.Max(double, double) narrow port; see nan_min above.
inline double nan_max(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return b > a ? b : a;
}
}  // namespace detail

class RunningStatistics {
   public:
    // Constructs an empty running statistics class.
    RunningStatistics() = default;

    // Constructs running statistics based on a list of values.
    explicit RunningStatistics(const std::vector<double>& values) { push(values); }

    // Gets the total number of samples.
    std::int64_t count() const { return n_; }

    // Returns the minimum value in the sample data. Returns NaN if data is empty or if
    // any entry is NaN.
    double minimum() const { return n_ > 0 ? min_ : kNaN; }

    // Returns the maximum value in the sample data. Returns NaN if data is empty or if
    // any entry is NaN.
    double maximum() const { return n_ > 0 ? max_ : kNaN; }

    // Returns the sample mean, an estimate of the population mean. Returns NaN if data is
    // empty or if any entry is NaN.
    double mean() const { return n_ > 0 ? m1_ : kNaN; }

    // Returns the sample variance (N-1 normalizer / Bessel's correction). Returns NaN if
    // data has less than two entries or if any entry is NaN.
    double variance() const { return n_ < 2 ? kNaN : m2_ / static_cast<double>(n_ - 1); }

    // Returns the population variance (N normalizer). Returns NaN if data is empty or if
    // any entry is NaN.
    double population_variance() const { return n_ < 2 ? kNaN : m2_ / static_cast<double>(n_); }

    // Returns the sample standard deviation (N-1 normalizer). Returns NaN if data has
    // less than two entries or if any entry is NaN.
    double standard_deviation() const {
        return n_ < 2 ? kNaN : std::sqrt(m2_ / static_cast<double>(n_ - 1));
    }

    // Returns the population standard deviation (N normalizer). Returns NaN if data is
    // empty or if any entry is NaN.
    double population_standard_deviation() const {
        return n_ < 2 ? kNaN : std::sqrt(m2_ / static_cast<double>(n_));
    }

    // Returns the coefficient of variation of the sample.
    double coefficient_of_variation() const { return standard_deviation() / mean(); }

    // Returns the sample skewness (Bessel's correction; type 2). Returns NaN if data has
    // less than three entries or if any entry is NaN.
    double skewness() const {
        if (n_ < 3) return kNaN;
        double nd = static_cast<double>(n_);
        return nd * m3_ * std::sqrt(m2_ / static_cast<double>(n_ - 1)) /
               (m2_ * m2_ * static_cast<double>(n_ - 2)) * static_cast<double>(n_ - 1);
    }

    // Returns the population skewness (no normalizer; type 1). Returns NaN if data has
    // less than two entries or if any entry is NaN.
    double population_skewness() const {
        if (n_ < 2) return kNaN;
        return std::sqrt(static_cast<double>(n_)) * m3_ / std::pow(m2_, 1.5);
    }

    // Returns the sample kurtosis (Bessel's correction; type 2). Returns NaN if data has
    // less than four entries or if any entry is NaN.
    double kurtosis() const {
        if (n_ < 4) return kNaN;
        double nd = static_cast<double>(n_);
        double denom = static_cast<double>((n_ - 2) * (n_ - 3));
        return (nd * nd - 1.0) / denom * (nd * m4_ / (m2_ * m2_) - 3.0 + 6.0 / (nd + 1.0));
    }

    // Returns the population kurtosis (no normalizer; type 1). Returns NaN if data has
    // less than three entries or if any entry is NaN.
    double population_kurtosis() const {
        if (n_ < 3) return kNaN;
        return static_cast<double>(n_) * m4_ / (m2_ * m2_) - 3.0;
    }

    // Updates the running statistics by adding another data value (in-place).
    void push(double value) {
        n_ += 1;
        double d = value - m1_;
        double s = d / static_cast<double>(n_);
        double s2 = s * s;
        double t = d * s * static_cast<double>(n_ - 1);
        m1_ += s;
        double poly = static_cast<double>(n_ * n_ - 3 * n_ + 3);
        m4_ += t * s2 * poly + 6.0 * s2 * m2_ - 4.0 * s * m3_;
        m3_ += t * s * static_cast<double>(n_ - 2) - 3.0 * s * m2_;
        m2_ += t;
        // Update min
        if (value < min_ || std::isnan(value)) min_ = value;
        // Update max
        if (value > max_ || std::isnan(value)) max_ = value;
    }

    // Updates the running statistics by adding a sequence of data values (in-place).
    void push(const std::vector<double>& values) {
        for (double value : values) push(value);
    }

    // corehydro ADDITION: the stateless R/Python surface serializes the accumulator between
    // calls (there is no persistent C++ object on that side), so a round-trip is needed to
    // resume push()ing across calls. No C# counterpart.
    static RunningStatistics from_state(std::int64_t n, double m1, double m2, double m3, double m4,
                                        double min, double max) {
        RunningStatistics rs;
        rs.n_ = n;
        rs.min_ = min;
        rs.max_ = max;
        rs.m1_ = m1;
        rs.m2_ = m2;
        rs.m3_ = m3;
        rs.m4_ = m4;
        return rs;
    }

    // corehydro ADDITION: raw-moment accessors for the same round-trip. No C# counterpart.
    double m1_state() const { return m1_; }
    double m2_state() const { return m2_; }
    double m3_state() const { return m3_; }
    double m4_state() const { return m4_; }

    // Creates a copy of the running statistics, independent of the original (mirrors
    // RunningStatistics.Clone(), added in v2.1.4; see the file header -- the C++ value-type
    // port already gives an independent copy via the implicit copy constructor, so this
    // exists purely for API parity with the C# class).
    RunningStatistics clone() const { return *this; }

    // Create a new running statistics over the combined samples of two existing running
    // statistics.
    static RunningStatistics combine(const RunningStatistics& a, const RunningStatistics& b) {
        if (a.n_ == 0) return b.clone();
        if (b.n_ == 0) return a.clone();

        std::int64_t n = a.n_ + b.n_;
        double an = static_cast<double>(a.n_);
        double bn = static_cast<double>(b.n_);
        double nd = static_cast<double>(n);

        double d = b.m1_ - a.m1_;
        double d2 = d * d;
        double d3 = d2 * d;
        double d4 = d2 * d2;

        double m1 = (an * a.m1_ + bn * b.m1_) / nd;
        double m2 = a.m2_ + b.m2_ + d2 * an * bn / nd;
        double m3 = a.m3_ + b.m3_ +
                    d3 * an * bn * static_cast<double>(a.n_ - b.n_) / static_cast<double>(n * n) +
                    3.0 * d * (an * b.m2_ - bn * a.m2_) / nd;
        double m4 = a.m4_ + b.m4_ +
                    d4 * an * bn *
                        static_cast<double>(a.n_ * a.n_ - a.n_ * b.n_ + b.n_ * b.n_) /
                        static_cast<double>(n * n * n) +
                    6.0 * d2 *
                        (static_cast<double>(a.n_ * a.n_) * b.m2_ +
                         static_cast<double>(b.n_ * b.n_) * a.m2_) /
                        static_cast<double>(n * n) +
                    4.0 * d * (an * b.m3_ - bn * a.m3_) / nd;
        double min_v = detail::nan_min(a.min_, b.min_);
        double max_v = detail::nan_max(a.max_, b.max_);

        RunningStatistics result;
        result.n_ = n;
        result.m1_ = m1;
        result.m2_ = m2;
        result.m3_ = m3;
        result.m4_ = m4;
        result.min_ = min_v;
        result.max_ = max_v;
        return result;
    }

   private:
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    std::int64_t n_ = 0;
    double min_ = std::numeric_limits<double>::infinity();
    double max_ = -std::numeric_limits<double>::infinity();
    double m1_ = 0.0;
    double m2_ = 0.0;
    double m3_ = 0.0;
    double m4_ = 0.0;
};

// Create a new running statistics over the combined samples of two existing running
// statistics.
inline RunningStatistics operator+(const RunningStatistics& a, const RunningStatistics& b) {
    return RunningStatistics::combine(a, b);
}

}  // namespace corehydro::numerics::data
