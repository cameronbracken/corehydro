// ported from: Numerics/Machine Learning/Unsupervised/JenksNaturalBreaks.cs @ 2a0357a
//
// The Jenks optimization method (natural-breaks classification): partition a one-dimensional
// sample into k classes minimizing the within-class sum of squared deviations. Upstream follows
// the simple-statistics `jenks.js` formulation.
//
// References (upstream's):
//   http://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization
//   https://github.com/simple-statistics/simple-statistics/blob/main/src/jenks.js
//
// Three transcription notes, each on something a "cleanup" would silently change:
//
// 1. THE CONSTRUCTOR ESTIMATES. C# calls `Estimate()` from the constructor, so a constructed
//    object is always fitted and `Clusters`/`Breaks` are never null after construction. Mirrored:
//    there is no separate `train()` on this class, unlike KMeans and GaussianMixtureModel.
// 2. THE DYNAMIC PROGRAM IS 1-BASED with `[n+1, k+1]` tables and a `double.MaxValue` sentinel in
//    the variance table. The inner update compares with `>=`, not `>`, so when a later starting
//    index `i3` ties the incumbent cost, the LATER index wins and the class boundary moves. The
//    loop bounds and that comparison are transcribed exactly; an off-by-one or a strict `>`
//    changes every break.
// 3. `v` IS DECLARED OUTSIDE THE OUTER LOOP and its last value is what `varianceCombinations[l,
//    1]` records after the inner loop finishes -- i.e. the single-class cost of the whole prefix
//    `[0, l)`, which is only correct because the inner loop's final iteration (`m == l`) covers
//    exactly that prefix. Keeping the declaration where upstream has it makes that dependence
//    visible rather than accidental.
//
// `GoodnessOfVarianceFit` is a computed property in C# (recomputed on every read); it is a plain
// const method here for the same reason.
#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/machine_learning/support/jenks_cluster.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::machine_learning {

class JenksNaturalBreaks {
   public:
    // Creates a new Jenks natural-breaks optimization and estimates it immediately.
    JenksNaturalBreaks(const std::vector<double>& data, int number_of_clusters,
                       bool is_data_sorted = false) {
        // (The C# `data == null` throw has no C++ analogue -- a const reference is never null.)
        if (data.empty()) throw std::invalid_argument("The data array is empty.");
        if (number_of_clusters <= 0)
            throw std::invalid_argument("The number of clusters must be greater than zero.");
        if (number_of_clusters > static_cast<int>(data.size()))
            throw std::invalid_argument(
                "The number of clusters cannot be greater than the length of the data array.");

        // Sort the data in numerical order.
        sorted_data_ = data;
        if (!is_data_sorted) std::sort(sorted_data_.begin(), sorted_data_.end());

        number_of_clusters_ = number_of_clusters;
        estimate();
    }

    // The array of sorted input data.
    const std::vector<double>& sorted_data() const { return sorted_data_; }

    // The number of clusters.
    int number_of_clusters() const { return number_of_clusters_; }

    // The estimated clusters.
    const std::vector<JenksCluster>& clusters() const { return clusters_; }

    // The break points (each cluster's maximum value).
    const std::vector<double>& breaks() const { return breaks_; }

    // The goodness-of-variance-fit measure. The closer to 1, the better the fit.
    double goodness_of_variance_fit() const {
        double mean = 0.0;
        for (double x : sorted_data_) mean += x;
        mean /= static_cast<double>(sorted_data_.size());
        // Sum of squared deviations from data.
        double sdam = 0;
        for (std::size_t i = 0; i < sorted_data_.size(); i++) sdam += sqr(sorted_data_[i] - mean);
        // Sum of squared deviations from clusters.
        double sdcm = 0;
        for (std::size_t i = 0; i < clusters_.size(); i++)
            sdcm += clusters_[i].sum_of_squared_deviations();
        return (sdam - sdcm) / sdam;
    }

   private:
    // Estimate the Jenks natural breaks.
    void estimate() {
        int n = static_cast<int>(sorted_data_.size());
        int k = number_of_clusters_;

        // Initialize the matrices (1-based, [n+1] x [k+1]).
        std::vector<std::vector<int>> lower_class_limits(
            static_cast<std::size_t>(n + 1), std::vector<int>(static_cast<std::size_t>(k + 1), 0));
        std::vector<std::vector<double>> variance_combinations(
            static_cast<std::size_t>(n + 1),
            std::vector<double>(static_cast<std::size_t>(k + 1), 0.0));
        for (int i = 1; i <= k; i++) {
            lower_class_limits[1][static_cast<std::size_t>(i)] = 1;
            variance_combinations[1][static_cast<std::size_t>(i)] = 0;
            for (int j = 2; j <= n; j++)
                variance_combinations[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] =
                    std::numeric_limits<double>::max();
        }

        double v = 0;
        for (int l = 2; l <= n; l++) {
            double s1 = 0;
            double s2 = 0;
            double w = 0;
            for (int m = 1; m <= l; m++) {
                int i3 = l - m + 1;
                double val = sorted_data_[static_cast<std::size_t>(i3 - 1)];

                s2 += val * val;
                s1 += val;

                w++;
                v = s2 - (s1 * s1) / w;
                int i4 = i3 - 1;
                if (i4 != 0) {
                    for (int j = 2; j <= k; j++) {
                        // `>=`, not `>`: a tie moves the class limit to the later i3.
                        if (variance_combinations[static_cast<std::size_t>(l)]
                                                 [static_cast<std::size_t>(j)] >=
                            (v + variance_combinations[static_cast<std::size_t>(i4)]
                                                      [static_cast<std::size_t>(j - 1)])) {
                            lower_class_limits[static_cast<std::size_t>(l)]
                                              [static_cast<std::size_t>(j)] = i3;
                            variance_combinations[static_cast<std::size_t>(l)]
                                                 [static_cast<std::size_t>(j)] =
                                v + variance_combinations[static_cast<std::size_t>(i4)]
                                                         [static_cast<std::size_t>(j - 1)];
                        }
                    }
                }
            }
            lower_class_limits[static_cast<std::size_t>(l)][1] = 1;
            variance_combinations[static_cast<std::size_t>(l)][1] = v;
        }

        // Create clusters by walking the class limits back from the top.
        int kk = n;
        std::vector<int> kclass(static_cast<std::size_t>(k), 0);
        kclass[static_cast<std::size_t>(k - 1)] = n - 1;
        for (int j = k; j >= 2; j--) {
            int id = lower_class_limits[static_cast<std::size_t>(kk)][static_cast<std::size_t>(j)] - 2;
            kclass[static_cast<std::size_t>(j - 2)] = id;
            kk = lower_class_limits[static_cast<std::size_t>(kk)][static_cast<std::size_t>(j)] - 1;
        }

        clusters_.clear();
        breaks_.assign(static_cast<std::size_t>(k), 0.0);
        clusters_.emplace_back(sorted_data_, 0, kclass[0]);
        breaks_[0] = clusters_[0].max_value();
        for (int i = 1; i < k; i++) {
            clusters_.emplace_back(sorted_data_, kclass[static_cast<std::size_t>(i - 1)] + 1,
                                   kclass[static_cast<std::size_t>(i)]);
            breaks_[static_cast<std::size_t>(i)] =
                clusters_[static_cast<std::size_t>(i)].max_value();
        }
    }

    std::vector<double> sorted_data_;
    int number_of_clusters_ = 0;
    std::vector<JenksCluster> clusters_;
    std::vector<double> breaks_;
};

}  // namespace corehydro::numerics::machine_learning
