// ported from: Numerics/Data/Statistics/Statistics.cs @ 7e8e8d1
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
// `SevenNumberSummary` remains omitted -- no caller ported so far needs it; add it if a
// later target does. (`FiveNumberSummary` landed in P5; see the note at the bottom.)
//
// P3.3 adds `mean()`, the plain `Statistics.Mean(IList<double>)` overload -- distinct from
// product_moments()'s internal mean, which requires N>=4 and returns NaN below that floor.
// Fourier::autocorrelation (math/fourier/fourier.hpp) needs a mean with no minimum-sample-
// size requirement, matching the C# call site (`Statistics.Mean(series)`, not
// `Statistics.ProductMoments`). (`ParallelMean` landed in P5; see the note at the bottom.)
//
// P3.10 adds `variance()`/`standard_deviation()`, the plain `Statistics.Variance(IList
// <double>)`/`StandardDeviation(IList<double>)` overloads (N-1 Bessel-corrected sample
// variance via the same running-difference recurrence as the C# source, distinct from
// `product_moments()`'s internal stdev which requires N>=4) -- Bootstrap's SE/CI computation
// needs a 2-sample-minimum variance with no such floor. (`PopulationVariance`/
// `PopulationStandardDeviation` landed in P5; see the note at the bottom.)
//
// P1 adds `maximum()`, the plain `Statistics.Maximum(IList<double>)` overload
// (Statistics.cs:90-105), for SpatialGEV::SetDefaultParameters (SpatialGEV.cs:480,487,494).
// The C# `data == null` throw has no C++ analogue (a `const std::vector<double>&` is never
// null).
//
// P4 Task 5 adds `minimum()`, the plain `Statistics.Minimum(IList<double>)` overload
// (Statistics.cs:63-82), for DataFrame::summary_statistics_exact_data_only. Same shape as
// `maximum()` (seed at the opposite infinity, same NaN-propagation and empty-sequence
// handling), and the same C# `data == null` throw has no C++ analogue.
//
// P4 Task 1 adds `mean_variance()` (Statistics.cs:262 -- literally `(Mean(data),
// Variance(data))`, so nothing new to verify beyond the pair construction), the
// `ranks_in_place(data, ties)` tie overload (Statistics.cs:674), and the vector `percentile`
// overload (Statistics.cs:573-583). All three exist for HypothesisTests (P4 Task 2), which
// this port had not reached yet when the no-ties `ranks_in_place` and single-`k` `percentile`
// were ported.
//
// P5 Task 1 adds the five members the Machine Learning layer needs: `population_variance`
// (Statistics.cs:218) and `population_standard_deviation` (:253) for DecisionTree's variance
// reduction, `parallel_mean` (:143) for RandomForest's and kNN's prediction-interval mean
// column, `five_number_summary` (:590) for GeneralizedLinearModel's residual report, and
// `entropy` (:736) for DecisionTree's classification information gain. v2.2.0 makes
// `parallel_mean` deterministic with a fixed chunk reduction, mirrored below without threads.
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/data/weight_type.hpp"

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
        volatile double product = (di + 1.0) * data[i];
        double diff = product - t;
        volatile double square = diff * diff;
        volatile double increment = square / ((di + 1.0) * di);
        variance_ += increment;
    }
    return variance_ / (static_cast<double>(data.size()) - 1.0);
}

// Sample standard deviation (sqrt of `variance`). Mirrors Statistics.StandardDeviation.
inline double standard_deviation(const std::vector<double>& data) { return std::sqrt(variance(data)); }

// Evaluates the population variance (N normalizer, Statistics.cs:218). Same running-difference
// recurrence as `variance()` above, divided by N rather than N-1 -- NOT `variance() * (n-1)/n`,
// which is algebraically identical but rounds differently. Returns NaN for an empty sequence.
inline double population_variance(const std::vector<double>& data) {
    if (data.empty()) return std::numeric_limits<double>::quiet_NaN();
    double variance_ = 0.0;
    double t = data[0];
    for (std::size_t i = 1; i < data.size(); ++i) {
        double di = static_cast<double>(i);
        t += data[i];
        volatile double product = (di + 1.0) * data[i];
        double diff = product - t;
        volatile double square = diff * diff;
        volatile double increment = square / ((di + 1.0) * di);
        variance_ += increment;
    }
    return variance_ / static_cast<double>(data.size());
}

// Population standard deviation (sqrt of `population_variance`, Statistics.cs:253).
inline double population_standard_deviation(const std::vector<double>& data) {
    return std::sqrt(population_variance(data));
}

// Deterministic fixed-chunk reduction used by v2.2.0 ParallelMean. C++ executes the chunks
// serially, but preserves the shipped association order: 64 balanced chunk sums followed by a
// chunk-order reduction. Small samples use the ordinary sequential mean.
inline double parallel_mean(const std::vector<double>& data) {
    if (data.empty()) return std::numeric_limits<double>::quiet_NaN();
    constexpr std::size_t kChunks = 64;
    constexpr std::size_t kSequentialThreshold = 8192;
    if (data.size() < kSequentialThreshold) return mean(data);
    const std::size_t chunks = std::min(kChunks, data.size());
    std::vector<double> chunk_sums(chunks, 0.0);
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        const std::size_t start = chunk * data.size() / chunks;
        const std::size_t end = (chunk + 1) * data.size() / chunks;
        for (std::size_t i = start; i < end; ++i) chunk_sums[chunk] += data[i];
    }
    double total = 0.0;
    for (double chunk_sum : chunk_sums) total += chunk_sum;
    return total / static_cast<double>(data.size());
}

// Estimates the arithmetic sample mean and the unbiased (N-1) sample variance in one call.
// Mirrors Statistics.MeanVariance, which is literally `(Mean(data), Variance(data))` -- this
// is an identity over `mean()`/`variance()` above, not an independent computation.
inline std::pair<double, double> mean_variance(const std::vector<double>& data) {
    return {mean(data), variance(data)};
}

inline double jackknife_standard_error(
    const std::vector<double>& data,
    const std::function<double(const std::vector<double>&)>& statistic) {
    if (!statistic) throw std::invalid_argument("statistic callback must be provided");
    if (data.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (data.size() == 1) return 0.0;
    const double theta = statistic(data);
    constexpr std::size_t kChunks = 64;
    const std::size_t chunks = std::min(kChunks, data.size());
    std::vector<double> chunk_sums(chunks, 0.0);
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        const std::size_t start = chunk * data.size() / chunks;
        const std::size_t end = (chunk + 1) * data.size() / chunks;
        for (std::size_t i = start; i < end; ++i) {
            std::vector<double> sample;
            sample.reserve(data.size() - 1);
            sample.insert(sample.end(), data.begin(), data.begin() + static_cast<std::ptrdiff_t>(i));
            sample.insert(sample.end(), data.begin() + static_cast<std::ptrdiff_t>(i + 1), data.end());
            const double difference = statistic(sample) - theta;
            chunk_sums[chunk] += difference * difference;
        }
    }
    double sum_squares = 0.0;
    for (double chunk_sum : chunk_sums) sum_squares += chunk_sum;
    return std::sqrt((static_cast<double>(data.size()) - 1.0) /
                     static_cast<double>(data.size()) * sum_squares);
}

inline std::vector<double> jackknife_sample(
    const std::vector<double>& data,
    const std::function<double(const std::vector<double>&)>& statistic) {
    if (!statistic) throw std::invalid_argument("statistic callback must be provided");
    std::vector<double> result(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        std::vector<double> sample;
        sample.reserve(data.size() - 1);
        sample.insert(sample.end(), data.begin(), data.begin() + static_cast<std::ptrdiff_t>(i));
        sample.insert(sample.end(), data.begin() + static_cast<std::ptrdiff_t>(i + 1), data.end());
        result[i] = statistic(sample);
    }
    return result;
}

// Returns the smallest value of the unsorted data array (mirrors Statistics.Minimum's
// `IList<double>` overload; P4 Task 5, for DataFrame::summary_statistics_exact_data_only).
// Returns NaN for an empty sequence or if any entry is NaN; the running min is seeded at
// +inf and an all-empty result collapses back to NaN.
inline double minimum(const std::vector<double>& data) {
    if (data.empty()) return std::numeric_limits<double>::quiet_NaN();

    double min = std::numeric_limits<double>::infinity();
    for (double x : data) {
        if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
        if (x < min) min = x;
    }
    return std::isinf(min) && min > 0.0 ? std::numeric_limits<double>::quiet_NaN() : min;
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

    const double shift = data[0];
    double X1 = 0, X2 = 0, X3 = 0, X4 = 0;
    for (double x : data) {
        const double y = x - shift;
        const double y2 = y * y;
        X1 += y;
        X2 += y2;
        X3 += y2 * y;
        X4 += y2 * y2;
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
    return {shift + U1, S, G, K};
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
        // v2.2.0 forms the b2 and b3 numerators in double. This avoids the int32 overflow
        // at i = 46,343 and i = 1,293 while preserving exact integer products below those
        // thresholds.
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
    ranks_ties(previous_index, n);
    if (t > 0) ties[static_cast<std::size_t>(n - 1)] = static_cast<double>(t);

    return ranks;
}

// Returns the k-th percentile of `data` (k in [0, 1]) using zero-based linear
// interpolation (R `quantile()` Type 7). If `data_is_sorted` is false (the default), a
// sorted copy is taken first; pass true when `data` is already sorted to skip the copy.
inline double percentile(const std::vector<double>& data, double k, bool data_is_sorted = false) {
    int n = static_cast<int>(data.size());
    if (n == 0) throw std::invalid_argument("Sequence contains no elements.");
    if (std::isnan(k) || k < 0.0 || k > 1.0) throw std::out_of_range("k must be in [0,1].");

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
    double lower_value = (*sorted)[static_cast<std::size_t>(lower)];
    // Preserve the separate multiply and add used by the C# expression.
    volatile double interpolation =
        w * ((*sorted)[static_cast<std::size_t>(upper)] - lower_value);
    return lower_value + interpolation;
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

// Estimates the 5-number summary {min, 25th, 50th, 75th, max} (Statistics.cs:590). C# copies and
// sorts once, then calls the sorted-input `Percentile` three times; `min`/`max` come straight off
// the sorted copy rather than through `Minimum`/`Maximum`, so a NaN entry does NOT collapse the
// whole result to NaN the way `minimum()` would. Mirrored.
inline std::vector<double> five_number_summary(const std::vector<double>& data) {
    if (data.empty()) throw std::invalid_argument("Sequence contains no elements.");
    std::vector<double> sorted(data);
    std::sort(sorted.begin(), sorted.end());
    return {sorted.front(), percentile(sorted, 0.25, true), percentile(sorted, 0.50, true),
            percentile(sorted, 0.75, true), sorted.back()};
}

namespace detail {

inline double validate_weights(const std::vector<double>& data,
                               const std::vector<double>& weights) {
    if (data.size() != weights.size()) {
        throw std::invalid_argument("data and weights must have the same length");
    }
    double total = 0.0;
    for (double weight : weights) {
        if (!corehydro::numerics::is_finite(weight) || weight < 0.0) {
            throw std::out_of_range("weights must be finite and non-negative");
        }
        total += weight;
    }
    return total;
}

struct WeightedCentralSums {
    double mean;
    double s2;
    double s3;
    double s4;
};

inline WeightedCentralSums weighted_central_sums(const std::vector<double>& data,
                                                 const std::vector<double>& weights,
                                                 double total) {
    double sum = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) sum += weights[i] * data[i];
    const double weighted_mean = sum / total;
    double s2 = 0.0;
    double s3 = 0.0;
    double s4 = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        const double centered = data[i] - weighted_mean;
        const double centered2 = centered * centered;
        s2 += weights[i] * centered2;
        s3 += weights[i] * centered2 * centered;
        s4 += weights[i] * centered2 * centered2;
    }
    return {weighted_mean, s2, s3, s4};
}

inline double effective_sample_size(const std::vector<double>& weights, double total,
                                    WeightType weight_type) {
    if (weight_type == WeightType::Frequency) return total;
    double sum_squares = 0.0;
    for (double weight : weights) sum_squares += weight * weight;
    return total * total / sum_squares;
}

struct WeightedSample {
    std::vector<double> values;
    std::vector<double> weights;
    std::vector<double> prefix;
    double total;
};

inline WeightedSample prepare_weighted_sample(const std::vector<double>& data,
                                              const std::vector<double>& weights,
                                              bool data_is_sorted) {
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (weights[i] > 0.0) pairs.emplace_back(data[i], weights[i]);
    }
    if (!data_is_sorted) {
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& left, const auto& right) { return left.first < right.first; });
    }
    WeightedSample sample;
    sample.values.reserve(pairs.size());
    sample.weights.reserve(pairs.size());
    sample.prefix.reserve(pairs.size());
    double running = 0.0;
    for (const auto& [value, weight] : pairs) {
        sample.values.push_back(value);
        sample.weights.push_back(weight);
        sample.prefix.push_back(running);
        running += weight;
    }
    sample.total = running;
    return sample;
}

inline double weighted_percentile(const WeightedSample& sample, double k) {
    const std::size_t n = sample.values.size();
    if (n == 1 || k == 0.0) return sample.values.front();
    if (k == 1.0) return sample.values.back();
    auto position = [&sample](std::size_t i) {
        const double denominator = sample.total - sample.weights[i];
        return denominator > 0.0 ? sample.prefix[i] / denominator : (i == 0 ? 0.0 : 1.0);
    };
    std::size_t low = 0;
    std::size_t high = n - 1;
    while (high - low > 1) {
        const std::size_t middle = (low + high) >> 1;
        if (position(middle) <= k)
            low = middle;
        else
            high = middle;
    }
    const double p_low = position(low);
    const double p_high = position(high);
    if (k <= p_low) return sample.values[low];
    if (k >= p_high) return sample.values[high];
    const double fraction = (k - p_low) / (p_high - p_low);
    return sample.values[low] + fraction * (sample.values[high] - sample.values[low]);
}

}  // namespace detail

inline double mean(const std::vector<double>& data, const std::vector<double>& weights) {
    const double total = detail::validate_weights(data, weights);
    if (data.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (total <= 0.0) throw std::invalid_argument("weights must not all be zero");
    double sum = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) sum += weights[i] * data[i];
    return sum / total;
}

inline double variance(const std::vector<double>& data, const std::vector<double>& weights,
                       WeightType weight_type = WeightType::Frequency) {
    const double total = detail::validate_weights(data, weights);
    if (data.size() <= 1) return std::numeric_limits<double>::quiet_NaN();
    if (total <= 0.0) throw std::invalid_argument("weights must not all be zero");
    const auto sums = detail::weighted_central_sums(data, weights, total);
    double denominator;
    if (weight_type == WeightType::Frequency) {
        denominator = total - 1.0;
    } else {
        double sum_squares = 0.0;
        for (double weight : weights) sum_squares += weight * weight;
        denominator = total - sum_squares / total;
    }
    if (denominator <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return sums.s2 / denominator;
}

inline double standard_deviation(const std::vector<double>& data,
                                 const std::vector<double>& weights,
                                 WeightType weight_type = WeightType::Frequency) {
    return std::sqrt(variance(data, weights, weight_type));
}

inline double skewness(const std::vector<double>& data, const std::vector<double>& weights,
                       WeightType weight_type = WeightType::Frequency) {
    const double total = detail::validate_weights(data, weights);
    if (data.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (total <= 0.0) throw std::invalid_argument("weights must not all be zero");
    const auto sums = detail::weighted_central_sums(data, weights, total);
    const double n = detail::effective_sample_size(weights, total, weight_type);
    if (n <= 2.0) return std::numeric_limits<double>::quiet_NaN();
    const double m2 = sums.s2 / total;
    const double m3 = sums.s3 / total;
    const double g = m3 / std::pow(m2, 1.5);
    return std::sqrt(n * (n - 1.0)) / (n - 2.0) * g;
}

inline double kurtosis(const std::vector<double>& data, const std::vector<double>& weights,
                       WeightType weight_type = WeightType::Frequency) {
    const double total = detail::validate_weights(data, weights);
    if (data.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (total <= 0.0) throw std::invalid_argument("weights must not all be zero");
    const auto sums = detail::weighted_central_sums(data, weights, total);
    const double n = detail::effective_sample_size(weights, total, weight_type);
    if (n <= 3.0) return std::numeric_limits<double>::quiet_NaN();
    const double m2 = sums.s2 / total;
    const double m4 = sums.s4 / total;
    const double a = n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
    const double b = m4 / (m2 * m2) * ((n - 1.0) * (n - 1.0) / n);
    const double c = (n - 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0));
    return a * b - 3.0 * c;
}

inline double percentile(const std::vector<double>& data, double k,
                         const std::vector<double>& weights, bool data_is_sorted = false) {
    const double total = detail::validate_weights(data, weights);
    if (data.empty()) throw std::invalid_argument("Sequence contains no elements.");
    if (std::isnan(k) || k < 0.0 || k > 1.0) throw std::out_of_range("k must be in [0,1].");
    if (total <= 0.0) throw std::invalid_argument("weights must not all be zero");
    return detail::weighted_percentile(
        detail::prepare_weighted_sample(data, weights, data_is_sorted), k);
}

inline std::vector<double> percentile(const std::vector<double>& data,
                                      const std::vector<double>& k,
                                      const std::vector<double>& weights,
                                      bool data_is_sorted = false) {
    const double total = detail::validate_weights(data, weights);
    if (k.empty()) return {};
    if (data.empty()) throw std::invalid_argument("Sequence contains no elements.");
    if (total <= 0.0) throw std::invalid_argument("weights must not all be zero");
    const auto sample = detail::prepare_weighted_sample(data, weights, data_is_sorted);
    std::vector<double> result;
    result.reserve(k.size());
    for (double probability : k) {
        if (std::isnan(probability) || probability < 0.0 || probability > 1.0) {
            throw std::out_of_range("k must be in [0,1].");
        }
        result.push_back(detail::weighted_percentile(sample, probability));
    }
    return result;
}

// Returns the standardized values (x - mean) / sd.
//
// This is a `Tools.cs` member (Tools.cs:351), not a `Statistics.cs` one, and it is ported HERE
// rather than in tools.hpp only because it calls `Statistics.MeanStandardDeviation` and
// `tools.hpp` is included BY this header -- putting it there would be a circular include. The
// namespace therefore differs from the C# class; tools.hpp carries a pointer note. Degenerate
// spread (sd <= 0 or NaN) returns the ZERO-INITIALIZED result array, not NaNs -- upstream's
// early return leaves `result` untouched.
inline std::vector<double> standardize(const std::vector<double>& values) {
    std::vector<double> result(values.size(), 0.0);
    double mu = mean(values);
    double sd = standard_deviation(values);
    if (sd <= 0.0 || std::isnan(sd)) return result;
    for (std::size_t i = 0; i < values.size(); ++i) result[i] = (values[i] - mu) / sd;
    return result;
}

// Computes the entropy of `data` under `pdf` (Statistics.cs:736): -sum(p * log(p)) over
// p = pdf(x), skipping any point whose density is not strictly positive. DecisionTree's
// classification information gain is the only ported caller.
inline double entropy(const std::vector<double>& data,
                      const std::function<double(double)>& pdf) {
    double sum = 0.0;
    for (double x : data) {
        double p = pdf(x);
        if (p > 0.0) sum += p * std::log(p);
    }
    return -sum;
}

}  // namespace corehydro::numerics::data
