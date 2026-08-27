// ported from: Numerics/Machine Learning/Unsupervised/KMeans.cs @ 2a0357a
//
// k-Means clustering: partition n observations into k clusters, each observation belonging to the
// cluster with the nearest centroid.
//
// Reference (upstream's): https://en.wikipedia.org/wiki/K-means_clustering
//
// Five transcription notes, each on something a "cleanup" would silently change:
//
// 1. `Parallel.For` -> SERIAL LOOP, and that is exact. `GetLabels` writes only `labels[idx]` per
//    iteration and reads nothing another iteration writes, so serial execution gives bit-identical
//    results. Nothing else in this class is parallel.
// 2. `Train` runs the E-step, compares the new labels against the previous ones, and BREAKS
//    BEFORE the M-step when nothing moved. So `means()` on a converged fit is the centroid set the
//    LAST E-step assigned against -- one M-step behind the final labels -- and `iterations()`
//    counts E-steps. This is why upstream's iris test can assert the label counts and the means
//    together.
// 3. `Iterations` is the loop variable itself, so a run that never converges leaves it at
//    `MaxIterations + 1`. Mirrored.
// 4. `GetCentroids` divides by `count[k] > 0 ? count[k] : 1`, so an EMPTY cluster keeps an
//    all-zero centroid rather than becoming NaN. That zero centroid then competes for points in
//    the next E-step. Mirrored; do not "fix" it to a re-seed.
// 5. The k-means++ selection loop divides `D[i] /= sum` IN PLACE while accumulating `cdf[i]` and
//    breaks on the first `u <= cdf[i]`. If rounding leaves `u` above the final CDF entry, no
//    break fires and `idx` KEEPS ITS PREVIOUS VALUE -- the index chosen for the previous center,
//    or the uniform draw from step 1. That is reachable, and it is what upstream does; there is
//    no fallback to add.
//
// The `float[]` constructor overload is not ported: C++ has no overload-resolution reason for it
// (a caller with floats converts at the call site) and it is byte-for-byte the `double[]` body.
#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::numerics::machine_learning {

class KMeans {
   public:
    // Creates a new k-Means clustering analysis over a single predictor column.
    KMeans(const std::vector<double>& x, int k) : KMeans(math::linalg::Matrix(x), k) {}

    // Creates a new k-Means clustering analysis.
    KMeans(const math::linalg::Matrix& x, int k)
        : k_(k),
          x_(x),
          dimension_(x.number_of_columns()),
          means_(static_cast<std::size_t>(k),
                 std::vector<double>(static_cast<std::size_t>(x.number_of_columns()), 0.0)),
          labels_(static_cast<std::size_t>(x.number_of_rows()), 0) {}

    // The number of clusters.
    int k() const { return k_; }

    // The matrix of predictor values.
    const math::linalg::Matrix& x() const { return x_; }

    // The dimensionality (number of features) of the data space.
    int dimension() const { return dimension_; }

    // The cluster means, k rows by `dimension` columns.
    const std::vector<std::vector<double>>& means() const { return means_; }

    // The cluster label assigned to each data point.
    const std::vector<int>& labels() const { return labels_; }

    // The maximum iterations in the clustering algorithm. Default = 1,000.
    int max_iterations() const { return max_iterations_; }
    void set_max_iterations(int value) { max_iterations_ = value; }

    // The number of iterations required to find the clusters.
    int iterations() const { return iterations_; }

    // Estimates the k-Means clusters. A `seed` of zero or less clock-seeds the generator.
    void train(int seed = -1, bool k_means_plus_plus = true) {
        // 1. Initialize cluster centers.
        means_ = initialize(x_, k_, seed, k_means_plus_plus);

        // 2. Optimize clusters.
        for (iterations_ = 1; iterations_ <= max_iterations_; iterations_++) {
            // E-step: assign samples to the closest centroids.
            std::vector<int> old_labels = labels_;
            labels_ = get_labels(means_);

            // Stop when the E-step doesn't change the assignment of any data point.
            bool labels_changed = false;
            for (std::size_t i = 0; i < labels_.size(); i++) {
                if (old_labels[i] != labels_[i]) {
                    labels_changed = true;
                    break;
                }
            }
            if (labels_changed == false) break;

            // M-step: calculate new centroids from the clusters.
            means_ = get_centroids(labels_);
        }
    }

    // Initializes the centroids, either uniformly at random or by k-Means++ (the default).
    static std::vector<std::vector<double>> initialize(const math::linalg::Matrix& x, int k,
                                                       int seed = -1,
                                                       bool k_means_plus_plus = true) {
        std::vector<std::vector<double>> centroids(
            static_cast<std::size_t>(k),
            std::vector<double>(static_cast<std::size_t>(x.number_of_columns()), 0.0));
        sampling::MersenneTwister rnd = seed > 0 ? sampling::MersenneTwister(
                                                       static_cast<std::uint32_t>(seed))
                                                 : sampling::MersenneTwister();

        if (k_means_plus_plus == false) {
            std::vector<int> rnd_idxs =
                utilities::next_integers(rnd, 0, x.number_of_rows(), k, false);
            std::sort(rnd_idxs.begin(), rnd_idxs.end());
            for (int i = 0; i < k; i++)
                for (int j = 0; j < x.number_of_columns(); j++)
                    centroids[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                        x(rnd_idxs[static_cast<std::size_t>(i)], j);
            return centroids;
        }

        // Initialize using k-Means++ (http://en.wikipedia.org/wiki/K-means%2B%2B).

        // 1. Choose one center uniformly at random from among the data points.
        int idx = rnd.next(0, x.number_of_rows());
        for (int j = 0; j < x.number_of_columns(); j++)
            centroids[0][static_cast<std::size_t>(j)] = x(idx, j);

        for (int c = 1; c < k; c++) {
            // 2. For each data point x not chosen yet, compute D(x), the distance between x and
            // the nearest center that has already been chosen.
            double sum = 0;
            std::vector<double> D(static_cast<std::size_t>(x.number_of_rows()), 0.0);
            for (int i = 0; i < x.number_of_rows(); i++) {
                std::vector<double> xi = x.row(i);

                double min = distance(xi, centroids[0]);
                for (int j = 1; j < c; j++) {
                    double d = distance(xi, centroids[static_cast<std::size_t>(j)]);
                    if (d < min) min = d;
                }

                D[static_cast<std::size_t>(i)] = min;
                sum += min;
            }

            // Following Accord.Net's checks. Note (upstream's): the following checks could have
            // been avoided by adding a small value to each distance, but are kept as-is to avoid
            // breaking the random pattern in existing code.
            if (sum == 0) {
                // Degenerate case: all points are the same, choose any of them.
                idx = rnd.next(0, x.number_of_rows());
            } else {
                // 3. Choose one new data point at random as a new center, using a weighted
                //    probability distribution where a point x is chosen with probability
                //    proportional to D(x)^2.
                double u = rnd.next_double();
                std::vector<double> cdf(static_cast<std::size_t>(x.number_of_rows()), 0.0);
                for (std::size_t i = 0; i < D.size(); i++) {
                    D[i] /= sum;
                    cdf[i] = i == 0 ? D[i] : cdf[i - 1] + D[i];
                    if (u <= cdf[i]) {
                        idx = static_cast<int>(i);
                        break;
                    }
                }
                // (No else: if `u` falls past the last cdf entry, `idx` keeps its previous
                // value. See transcription note 5.)
            }
            for (int j = 0; j < x.number_of_columns(); j++)
                centroids[static_cast<std::size_t>(c)][static_cast<std::size_t>(j)] = x(idx, j);
        }

        return centroids;
    }

   private:
    // Gets the cluster label of each data point given the centroids.
    std::vector<int> get_labels(const std::vector<std::vector<double>>& centroids) const {
        // Assign samples to the closest centroids. (C# Parallel.For -- see note 1.)
        std::vector<int> labels(static_cast<std::size_t>(x_.number_of_rows()), 0);
        for (int idx = 0; idx < x_.number_of_rows(); idx++)
            labels[static_cast<std::size_t>(idx)] = get_closest_centroid(x_.row(idx), centroids);
        return labels;
    }

    // Gets the centroids given the cluster labels.
    std::vector<std::vector<double>> get_centroids(const std::vector<int>& labels) const {
        std::vector<double> count(static_cast<std::size_t>(k_), 0.0);
        std::vector<std::vector<double>> centroids(
            static_cast<std::size_t>(k_),
            std::vector<double>(static_cast<std::size_t>(dimension_), 0.0));

        // Get sums and counts.
        for (int i = 0; i < x_.number_of_rows(); i++) {
            std::size_t l = static_cast<std::size_t>(labels[static_cast<std::size_t>(i)]);
            count[l]++;
            for (int j = 0; j < dimension_; j++)
                centroids[l][static_cast<std::size_t>(j)] += x_(i, j);
        }

        // Get the mean of each cluster. An empty cluster divides by 1, keeping a zero centroid
        // rather than producing NaN (see note 4).
        for (int kk = 0; kk < k_; kk++)
            for (int j = 0; j < dimension_; j++)
                centroids[static_cast<std::size_t>(kk)][static_cast<std::size_t>(j)] /=
                    count[static_cast<std::size_t>(kk)] > 0
                        ? count[static_cast<std::size_t>(kk)]
                        : 1;

        return centroids;
    }

    // Returns the index of the centroid closest to the sample vector.
    int get_closest_centroid(const std::vector<double>& sample,
                             const std::vector<std::vector<double>>& centroids) const {
        double min = std::numeric_limits<double>::max();
        int min_idx = 0;
        for (int kk = 0; kk < k_; kk++) {
            double dist = distance(sample, centroids[static_cast<std::size_t>(kk)]);
            if (dist < min) {
                min = dist;
                min_idx = kk;
            }
        }
        return min_idx;
    }

    int k_;
    math::linalg::Matrix x_;
    int dimension_;
    std::vector<std::vector<double>> means_;
    std::vector<int> labels_;
    int max_iterations_ = 1000;
    int iterations_ = 0;
};

}  // namespace corehydro::numerics::machine_learning
