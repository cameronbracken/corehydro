// corehydro ADDITION -- LINQ ordering semantics, no single upstream C# counterpart.
//
// Three `Numerics.MachineLearning` call sites depend on the ORDER LINQ produces, not just on the
// set of values it produces, and both orders are oracle-visible. Reproducing them in a named
// helper keeps the dependence explicit and stops a call site reaching for an unordered container.
//
// `Enumerable.Distinct()` yields values in FIRST-APPEARANCE order (it is a streaming operator
// backed by a hash set, emitting each value the first time it is seen). Two call sites:
//   - NaiveBayes ctor: `Classes = y.Distinct().ToArray()`. `Means[i]`/`StandardDeviations[i]`/
//     `Priors[i]` are all indexed by position in THIS array, so a training response whose class
//     labels do not happen to arrive in sorted order produces a differently-ordered model. The
//     iris test data is sorted by species, which hides the dependence; a shuffled response would
//     not.
//   - DecisionTree::BestSplit, classification branch: `x.Distinct().ToArray()` supplies the
//     candidate thresholds, and `BestSplit` keeps the FIRST candidate that attains the best
//     performance (strict `>`), so the evaluation order decides every tie.
//
// `.GroupBy(i => i).OrderByDescending(grp => grp.Count()).Select(grp => grp.Key).First()` is the
// "most common value" idiom, and it resolves a tie in favor of the value that appeared FIRST:
// `GroupBy` yields groups in first-appearance order, and LINQ's `OrderByDescending` is a STABLE
// sort, so equal counts keep that order. Two call sites: DecisionTree's classification leaf value
// and KNearestNeighbors' classification prediction. `std::max_element` over a `std::map` would
// break the first ties toward the smallest KEY instead, which is a different answer.
#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace corehydro::numerics::machine_learning::support {

// C# `Enumerable.Distinct()`: the distinct values in first-appearance order.
inline std::vector<double> distinct_in_first_appearance_order(const std::vector<double>& values) {
    std::vector<double> distinct;
    for (double v : values) {
        bool seen = false;
        for (double d : distinct) {
            if (d == v) {
                seen = true;
                break;
            }
        }
        if (!seen) distinct.push_back(v);
    }
    return distinct;
}

// C# `.GroupBy(i => i).OrderByDescending(grp => grp.Count()).Select(grp => grp.Key).First()`:
// the most common value, ties going to whichever value appeared first. Throws on empty input,
// mirroring `First()`'s `InvalidOperationException`.
inline double most_common_first_appearance_wins(const std::vector<double>& values) {
    std::vector<double> keys = distinct_in_first_appearance_order(values);
    if (keys.empty()) throw std::runtime_error("Sequence contains no elements.");
    std::size_t best = 0;
    std::size_t best_count = 0;
    for (std::size_t k = 0; k < keys.size(); ++k) {
        std::size_t count = 0;
        for (double v : values)
            if (v == keys[k]) ++count;
        // Strict `>` keeps the earlier group on a tie, which is what the stable
        // OrderByDescending does.
        if (count > best_count) {
            best_count = count;
            best = k;
        }
    }
    return keys[best];
}

}  // namespace corehydro::numerics::machine_learning::support
