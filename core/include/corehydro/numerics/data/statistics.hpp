// ported from: Numerics/Data/Statistics/Statistics.cs @ 2a0357a
//
// Sample statistics needed by distribution estimation: product moments
// (mean, stdev, bias-corrected skew & excess kurtosis), linear (L-)moments, and
// rank statistics (ranks_in_place, which Correlation::spearman consumes). Phase 3 adds
// `percentile` (~line 544), the single-`k` overload only -- its zero-based linear-
// interpolation convention (R `quantile()` Type 7) is the oracle for MCMC posterior
// median/credible-interval reporting.
//
// P4 Task 1 ports the tolerance-based tie overload `RanksInPlace(double[], out double[]
// ties)` and the array overload `Percentile(IList<double>, IList<double> k, bool)` --
// both were previously omitted (see the numbered fidelity note on the tie overload below,
// which is the important divergence from the exact-equality overload directly above it).
// `FiveNumberSummary`/`SevenNumberSummary` remain omitted -- no caller ported so far needs
// them; add them if a later target does.
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
//
// P4 Task 1 adds `mean_variance()` (Statistics.cs:262 -- literally `(Mean(data),
// Variance(data))`, so nothing new to verify beyond the pair construction), the
// `ranks_in_place(data, ties)` tie overload (Statistics.cs:674), and the vector `percentile`
// overload (Statistics.cs:573-583). All three exist for HypothesisTests (P4 Task 2), which
// this port had not reached yet when the no-ties `ranks_in_place` and single-`k` `percentile`
// were ported.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/tools.hpp"

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

// Estimates the arithmetic sample mean and the unbiased (N-1) sample variance in one call.
// Mirrors Statistics.MeanVariance, which is literally `(Mean(data), Variance(data))` -- this
// is an identity over `mean()`/`variance()` above, not an independent computation.
inline std::pair<double, double> mean_variance(const std::vector<double>& data) {
    return {mean(data), variance(data)};
}

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
        // DELIBERATE DIVERGENCE (docs/upstream-csharp-issues.md): C# forms the weight
        // numerators `(i-2)*(i-1)` and `(i-3)*(i-2)*(i-1)` in `int`. The triple product
        // exceeds int32 at i = 1293, and C#'s default unchecked context WRAPS silently --
        // the real Numerics library returns T4 = -0.185 for a 1293-point arithmetic
        // sequence whose L-kurtosis is 0. In C++ that same overflow is undefined
        // behaviour, which CRAN's UBSan run reports. Forming the products in `double`
        // removes the UB and is bit-identical to the C# for every sample below the
        // overflow (the products are exact integers far under 2^53); above it the port
        // returns the mathematically correct weight where C# returns a wrapped one.
        const double di = static_cast<double>(i);
        B0 += sorted[i - 1];
        if (i > 1) B1 += (di - 1) / (N - 1) * sorted[i - 1];
        if (i > 2) B2 += (di - 2) * (di - 1) / ((N - 2) * (N - 1)) * sorted[i - 1];
        if (i > 3)
            B3 += (di - 3) * (di - 2) * (di - 1) / ((N - 3) * (N - 2) * (N - 1)) *
                  sorted[i - 1];
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

// Returns the rank of each entry of the unsorted data array, using a TOLERANCE-based tie test
// instead of exact equality, and also reports the length of each tie run through the output
// parameter `ties` (mirrors C# `Statistics.RanksInPlace(double[], out double[] ties)`).
//
// Three oracle-visible fidelity points differ from the exact-equality `ranks_in_place` overload
// directly above, each transcribed on purpose because HypothesisTests (P4 Task 2) sums over
// `ties` (MannWhitneyTest's tie correction `T`, MannKendallTest's `varS`):
//
// 1. The tie test is `AlmostEquals(work[i], work[previous], Tools.DoubleMachineEpsilon)`, i.e.
//    `|work[i] - work[previous]| <= kDoubleMachineEpsilon`, NOT the no-ties overload's
//    `fabs(...) <= 0` (exact equality). A run within one ULP of `kDoubleMachineEpsilon` ties
//    here where it would not tie above.
// 2. `ties` is allocated at `data.size()` and written ONLY in the multi-element-run branch, as
//    `ties[i - 1] = t`. It is therefore a SPARSE array of run lengths indexed by the position
//    (0-based) at which each run closes, not a per-element tie count and not a compact list.
// 3. The final `ranks_ties(...)` call after the loop -- which closes whatever run was still
//    open when the loop ended -- never writes `ties[...]`, because that write only happens
//    inside the loop's own else-branch. This is a genuine upstream defect: a tie run ending at
//    the very last element of `data` is never recorded in `ties`, even though its rank
//    averaging is still applied correctly. It MUST be reproduced (not "fixed") because it is
//    the input HypothesisTests' oracle-pinned callers were fit against.
inline std::vector<double> ranks_in_place(const std::vector<double>& data, std::vector<double>& ties) {
    const int n = static_cast<int>(data.size());
    std::vector<double> ranks(static_cast<std::size_t>(n), 0.0);
    ties.assign(static_cast<std::size_t>(n), 0.0);

    std::vector<int> index(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) index[static_cast<std::size_t>(i)] = i;
    std::sort(index.begin(), index.end(),
              [&data](int a, int b) { return data[static_cast<std::size_t>(a)] < data[static_cast<std::size_t>(b)]; });

    std::vector<double> work(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        work[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(index[static_cast<std::size_t>(i)])];

    auto ranks_ties = [&ranks, &index](int a, int b) {
        double rank = (b + a - 1) / 2.0 + 1;
        for (int k = a; k < b; ++k) ranks[static_cast<std::size_t>(index[static_cast<std::size_t>(k)])] = rank;
    };

    int previous_index = 0;
    int t = 0;
    for (int i = 1; i < n; ++i) {
        if (std::fabs(work[static_cast<std::size_t>(i)] - work[static_cast<std::size_t>(previous_index)]) <=
            kDoubleMachineEpsilon) {
            t += 1;
            continue;
        }

        if (i == previous_index + 1) {
            ranks[static_cast<std::size_t>(index[static_cast<std::size_t>(previous_index)])] =
                static_cast<double>(i);
            t = 0;
        } else {
            ranks_ties(previous_index, i);
            ties[static_cast<std::size_t>(i - 1)] = static_cast<double>(t);
            t = 0;
        }
        previous_index = i;
    }
    // Fidelity point 3: this closing call never writes `ties` -- a trailing tie run's length is
    // never recorded, reproducing the upstream defect.
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

// Returns the k-th percentile of `data` for every k in `k`, sorting `data` ONCE and calling the
// scalar `percentile` with `data_is_sorted = true` for each entry (mirrors C#
// `Percentile(IList<double>, IList<double> k, bool)`, which does the same -- NOT a naive loop
// over the unsorted-input scalar overload, which would re-sort per call).
inline std::vector<double> percentile(const std::vector<double>& data, const std::vector<double>& k,
                                       bool data_is_sorted = false) {
    std::vector<double> sorted_copy;
    const std::vector<double>* sorted = &data;
    if (!data_is_sorted) {
        sorted_copy = data;
        std::sort(sorted_copy.begin(), sorted_copy.end());
        sorted = &sorted_copy;
    }

    std::vector<double> result(k.size());
    for (std::size_t i = 0; i < k.size(); ++i) result[i] = percentile(*sorted, k[i], true);
    return result;
}

}  // namespace corehydro::numerics::data
