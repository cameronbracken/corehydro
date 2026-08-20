// ported from: Numerics/Data/Statistics/Statistics.cs @ 2a0357a
//
// Sample statistics needed by distribution estimation: product moments
// (mean, stdev, bias-corrected skew & excess kurtosis), linear (L-)moments, and
// rank statistics (ranks_in_place, which Correlation::spearman consumes). Phase 3 adds
// `percentile` (~line 544), the single-`k` overload only -- its zero-based linear-
// interpolation convention (R `quantile()` Type 7) is the oracle for MCMC posterior
// median/credible-interval reporting.
//
// ranks_in_place ports only the single-return-value RanksInPlace(double[]) overload
// (exact-equality tie runs); the RanksInPlace(double[], out ties) overload (tolerance-
// based ties via AlmostEquals, returning a tie-count array) is omitted -- no caller
// ported so far needs the tie-count output. The `Percentile(IList<double>, IList<double>
// k, bool)` array overload and `FiveNumberSummary`/`SevenNumberSummary` are omitted too --
// each caller ported so far needs only single-`k` calls; add them if a later target does.
//
// P3.3 adds `mean()`, the plain `Statistics.Mean(IList<double>)` overload -- distinct from
// product_moments()'s internal mean, which requires N>=4 and returns NaN below that floor.
// Fourier::autocorrelation (math/fourier/fourier.hpp) needs a mean with no minimum-sample-
// size requirement, matching the C# call site (`Statistics.Mean(series)`, not
// `Statistics.ProductMoments`). ParallelMean and the other overloads are not ported.
//
// P3.10 adds `variance()`/`standard_deviation()`, the plain `Statistics.Variance(IList
// <double>)`/`StandardDeviation(IList<double>)` overloads (N-1 Bessel-corrected sample
// variance via the same running-difference recurrence as the C# source, distinct from
// `product_moments()`'s internal stdev which requires N>=4) -- Bootstrap's SE/CI computation
// needs a 2-sample-minimum variance with no such floor. `PopulationVariance`/
// `PopulationStandardDeviation` (N normalizer) are not ported -- no caller needs them yet.
//
// P1 adds `maximum()`, the plain `Statistics.Maximum(IList<double>)` overload
// (Statistics.cs:90-105), for SpatialGEV::SetDefaultParameters (SpatialGEV.cs:480,487,494).
// The C# `data == null` throw has no C++ analogue (a `const std::vector<double>&` is never
// null); the `Minimum` overload is not ported -- no caller needs it yet.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace corehydro::numerics::data {

// Computes the arithmetic sample mean. Returns NaN for an empty sequence (mirrors
// Statistics.Mean's `IList<double>` overload).
inline double mean(const std::vector<double>& data) {
    if (data.empty()) return std::numeric_limits<double>::quiet_NaN();
    double sum = 0.0;
    for (double x : data) sum += x;
    return sum / static_cast<double>(data.size());
}

// Estimates the unbiased sample variance (N-1 normalizer / Bessel's correction) via the same
// running-difference recurrence as C# Statistics.Variance. Returns NaN for fewer than two
// entries.
inline double variance(const std::vector<double>& data) {
    if (data.size() <= 1) return std::numeric_limits<double>::quiet_NaN();
    double variance_ = 0.0;
    double t = data[0];
    for (std::size_t i = 1; i < data.size(); ++i) {
        double di = static_cast<double>(i);
        t += data[i];
        double diff = (di + 1.0) * data[i] - t;
        variance_ += diff * diff / ((di + 1.0) * di);
    }
    return variance_ / (static_cast<double>(data.size()) - 1.0);
}

// Sample standard deviation (sqrt of `variance`). Mirrors Statistics.StandardDeviation.
inline double standard_deviation(const std::vector<double>& data) { return std::sqrt(variance(data)); }

// Returns the largest value of the unsorted data array (mirrors Statistics.Maximum's
// `IList<double>` overload). Returns NaN for an empty sequence or if any entry is NaN; the
// running max is seeded at -inf and an all-empty result collapses back to NaN.
inline double maximum(const std::vector<double>& data) {
    if (data.empty()) return std::numeric_limits<double>::quiet_NaN();

    double max = -std::numeric_limits<double>::infinity();
    for (double x : data) {
        if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
        if (x > max) max = x;
    }
    return std::isinf(max) && max < 0.0 ? std::numeric_limits<double>::quiet_NaN() : max;
}

// Returns {mean, stdev (sample), bias-corrected skewness, bias-corrected excess kurtosis}.
inline std::vector<double> product_moments(const std::vector<double>& data) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    double N = static_cast<double>(data.size());
    if (N < 4) return {kNaN, kNaN, kNaN, kNaN};

    double X1 = 0, X2 = 0, X3 = 0, X4 = 0;
    for (double x : data) {
        double x2 = x * x;
        X1 += x;
        X2 += x2;
        X3 += x2 * x;
        X4 += x2 * x2;
    }
    double U1 = X1 / N, U2 = X2 / N, U3 = X3 / N, U4 = X4 / N;
    double m2 = (U2 - U1 * U1) * (N / (N - 1));  // sample variance
    double S = std::sqrt(m2);
    double U1_2 = U1 * U1, U1_3 = U1_2 * U1, U1_4 = U1_3 * U1;
    double S3 = S * S * S, S4 = S3 * S;
    double c3 = U3 - 3 * U1 * U2 + 2 * U1_3;
    double c4 = U4 - 4 * U1 * U3 + 6 * U2 * U1_2 - 3 * U1_4;
    double G = (N * N) / ((N - 1) * (N - 2)) * (c3 / S3);
    double K = ((N * N) * (N + 1)) / ((N - 1) * (N - 2) * (N - 3)) * (c4 / S4) -
               3.0 * (N - 1) * (N - 1) / ((N - 2) * (N - 3));
    return {U1, S, G, K};
}

// Returns {L1 (L-mean), L2 (L-scale), T3 (L-skewness), T4 (L-kurtosis)}.
inline std::vector<double> linear_moments(const std::vector<double>& data) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    double N = static_cast<double>(data.size());
    if (N < 4) return {kNaN, kNaN, kNaN, kNaN};

    std::vector<double> sorted(data);
    std::sort(sorted.begin(), sorted.end());

    double B0 = 0, B1 = 0, B2 = 0, B3 = 0;
    for (int i = 1; i <= static_cast<int>(N); ++i) {
        B0 += sorted[i - 1];
        if (i > 1) B1 += (i - 1) / (N - 1) * sorted[i - 1];
        if (i > 2) B2 += (i - 2) * (i - 1) / ((N - 2) * (N - 1)) * sorted[i - 1];
        if (i > 3)
            B3 += (i - 3) * (i - 2) * (i - 1) / ((N - 3) * (N - 2) * (N - 1)) * sorted[i - 1];
    }
    B0 /= N;
    B1 /= N;
    B2 /= N;
    B3 /= N;
    double L1 = B0;
    double L2 = 2 * B1 - B0;
    double T3 = 2 * (3 * B2 - B0) / (2 * B1 - B0) - 3;
    double T4 = 5 * (2 * (2 * B3 - 3 * B2) + B0) / (2 * B1 - B0) + 6;
    return {L1, L2, T3, T4};
}

// Returns the rank of each entry of the unsorted data array. Tied values (exact
// equality) receive the average rank of their run (mirrors C# Statistics.RanksInPlace).
inline std::vector<double> ranks_in_place(const std::vector<double>& data) {
    const int n = static_cast<int>(data.size());
    std::vector<double> ranks(static_cast<std::size_t>(n), 0.0);

    // Co-sorted index array: index[k] is the original position of the k-th smallest
    // value (mirrors Array.Sort(work, index) sorting a cloned `work` and permuting a
    // parallel `index` array to match).
    std::vector<int> index(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) index[static_cast<std::size_t>(i)] = i;
    std::sort(index.begin(), index.end(),
              [&data](int a, int b) { return data[static_cast<std::size_t>(a)] < data[static_cast<std::size_t>(b)]; });

    std::vector<double> work(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        work[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(index[static_cast<std::size_t>(i)])];

    // Assign the average rank (b+a-1)/2 + 1 to the tie run [a, b) (mirrors C# RanksTies).
    auto ranks_ties = [&ranks, &index](int a, int b) {
        double rank = (b + a - 1) / 2.0 + 1;
        for (int k = a; k < b; ++k) ranks[static_cast<std::size_t>(index[static_cast<std::size_t>(k)])] = rank;
    };

    int previous_index = 0;
    for (int i = 1; i < n; ++i) {
        if (std::fabs(work[static_cast<std::size_t>(i)] - work[static_cast<std::size_t>(previous_index)]) <= 0)
            continue;

        if (i == previous_index + 1) {
            ranks[static_cast<std::size_t>(index[static_cast<std::size_t>(previous_index)])] =
                static_cast<double>(i);
        } else {
            ranks_ties(previous_index, i);
        }
        previous_index = i;
    }
    ranks_ties(previous_index, n);

    return ranks;
}

// Returns the k-th percentile of `data` (k in [0, 1]) using zero-based linear
// interpolation (R `quantile()` Type 7). If `data_is_sorted` is false (the default), a
// sorted copy is taken first; pass true when `data` is already sorted to skip the copy.
inline double percentile(const std::vector<double>& data, double k, bool data_is_sorted = false) {
    int n = static_cast<int>(data.size());
    if (n == 0) throw std::invalid_argument("Sequence contains no elements.");
    if (k < 0.0 || k > 1.0) throw std::out_of_range("k must be in [0,1].");

    // A NaN `k` slips through the range check above -- every comparison against NaN is false --
    // and would reach `static_cast<int>(std::floor(h))` below. Converting a NaN double to `int`
    // is UNDEFINED BEHAVIOUR in C++, and the platforms disagree about it in the worst possible
    // way: AArch64's `fcvtzs` saturates to 0, so the expression falls out as NaN and nothing
    // appears wrong, while x86-64's `cvttsd2si` yields INT_MIN, which then indexes `sorted`
    // roughly 17 GB below its base and segfaults. Not hypothetical -- Bootstrap's BCa
    // acceleration constant is `0 / 0` whenever every jackknife sample fails, and that NaN
    // arrives here through `Normal::standard_cdf` as `k`.
    //
    // The C# this is ported from has exactly the same hole in its range check, but .NET's
    // float-to-int conversion is DEFINED to saturate NaN to 0 (.NET Core 3.0 onward), so C#
    // evaluates `sortedData[0] + NaN * (sortedData[0] - sortedData[0])` and hands the caller a
    // NaN. Returning NaN here reproduces the C# result exactly while removing the UB; it is a
    // fidelity fix, not a behaviour change (macOS already produced NaN by accident of ISA).
    if (std::isnan(k)) return std::numeric_limits<double>::quiet_NaN();

    std::vector<double> sorted_copy;
    const std::vector<double>* sorted = &data;
    if (!data_is_sorted) {
        sorted_copy = data;
        std::sort(sorted_copy.begin(), sorted_copy.end());
        sorted = &sorted_copy;
    }

    // Trivial cases
    if (n == 1 || k == 0.0) return (*sorted)[0];
    if (k == 1.0) return (*sorted)[static_cast<std::size_t>(n - 1)];

    // Zero-based linear interpolation (Type 7)
    double h = (n - 1) * k;
    int lower = static_cast<int>(std::floor(h));
    int upper = static_cast<int>(std::ceil(h));
    double w = h - lower;
    return (*sorted)[static_cast<std::size_t>(lower)] +
           w * ((*sorted)[static_cast<std::size_t>(upper)] - (*sorted)[static_cast<std::size_t>(lower)]);
}

}  // namespace corehydro::numerics::data
