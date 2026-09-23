// ported from: Numerics/Machine Learning/Supervised/KNearestNeighbors.cs @ 7e8e8d1
//
// The k-nearest-neighbors algorithm, for regression (inverse-squared-distance weighted average of
// the k neighbors' responses) or classification (their most common response).
//
// Reference (upstream's): https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
//
// Six transcription notes, each on something a "cleanup" would silently change:
//
// 1. Distance ties are broken by the original training index. This makes the result independent
//    of the host sort implementation while preserving the first-row-wins behavior.
// 2. Both neighbor lookup and prediction reject a query whose column count differs from the
//    training matrix.
// 3. The regression prediction is an inverse-SQUARED-distance weighted average with `w = 1` when
//    the distance is exactly 0, and it accumulates `knn[j] = y * w` and then divides each term by
//    `sum` SEPARATELY (`avg += knn[j] / sum`) rather than dividing once at the end. The division
//    order is oracle-visible; keep it.
// 4. The classification prediction is the first-appearance-wins mode over the k neighbours'
//    responses (see support/linq_order.hpp).
// 5. `PredictionIntervals` has NO classification branch -- it always takes percentiles and the
//    mean, even when `IsRegression` is false. Mirrored.
// 6. `Parallel.For` -> SERIAL LOOP, exact for the same reason as in random_forest.hpp: each
//    iteration writes only its own index, and `PredictionIntervals` draws all `realizations`
//    seeds up front. The mean column comes from `Statistics.ParallelMean`, whose PLINQ
//    partitioned sum is not reproducible across machines -- see `parallel_mean`'s own note in
//    numerics/data/statistics.hpp. v2.2.0 uses the sequential mean instead.
//
// The `double[,]` constructor overload is not ported, for the same reason as in decision_tree.hpp.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/machine_learning/support/linq_order.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::numerics::machine_learning {

class KNearestNeighbors {
   public:
    // Creates a new k-NN model over a single predictor column.
    KNearestNeighbors(const std::vector<double>& x, const std::vector<double>& y, int k)
        : KNearestNeighbors(math::linalg::Matrix(x), math::linalg::Vector(y), k) {}

    // Creates a new k-NN model.
    KNearestNeighbors(const math::linalg::Matrix& x, const math::linalg::Vector& y, int k)
        : k_(k), y_(y), x_(x) {
        if (y_.length() != x_.number_of_rows())
            throw std::invalid_argument("The y vector must be the same length as the x matrix.");
        if (y_.length() < 10)
            throw std::invalid_argument("There must be at least ten training data points.");
    }

    // The number of nearest neighbors.
    int k() const { return k_; }

    // The training vector of response values.
    const math::linalg::Vector& y() const { return y_; }

    // The training matrix of predictor values.
    const math::linalg::Matrix& x() const { return x_; }

    // The number of features in the training matrix of predictor values.
    int number_of_features() const { return x_.number_of_columns(); }

    // Determines whether this is for regression or classification. Default = regression.
    bool is_regression() const { return is_regression_; }
    void set_is_regression(bool value) { is_regression_ = value; }

    // Returns the indexes of the k nearest neighbors, `k` per query row, flattened row-major:
    // entry `i * k + j` is the j-th nearest neighbor of query row `i`.
    std::optional<std::vector<int>> get_neighbors(const math::linalg::Matrix& x) const {
        return k_nn(x_, y_, x);
    }
    std::optional<std::vector<int>> get_neighbors(const std::vector<double>& x) const {
        return k_nn(x_, y_, math::linalg::Matrix(x));
    }

    // Returns the prediction for each query row.
    std::optional<std::vector<double>> predict(const math::linalg::Matrix& x) const {
        return k_nn_predict(x_, y_, x);
    }
    std::optional<std::vector<double>> predict(const std::vector<double>& x) const {
        return k_nn_predict(x_, y_, math::linalg::Matrix(x));
    }

    // Returns the prediction from a single bootstrap resample of the training set. A `seed` of
    // zero or less clock-seeds the generator.
    std::optional<std::vector<double>> bootstrap_predict(const math::linalg::Matrix& x,
                                                          int seed = -1) const {
        return k_nn_bootstrap_predict(x_, y_, x, seed);
    }
    std::optional<std::vector<double>> bootstrap_predict(const std::vector<double>& x,
                                                          int seed = -1) const {
        return k_nn_bootstrap_predict(x_, y_, math::linalg::Matrix(x), seed);
    }

    // Returns bootstrapped prediction intervals as an n-by-4 matrix with columns lower, median,
    // upper, mean. `alpha` is the confidence level; the default 0.1 gives 90% intervals.
    math::linalg::Matrix prediction_intervals(const math::linalg::Matrix& x, int seed = -1,
                                               int realizations = 1000,
                                               double alpha = 0.1) const {
        return k_nn_prediction_intervals(x_, y_, x, seed, realizations, alpha);
    }
    math::linalg::Matrix prediction_intervals(const std::vector<double>& x, int seed = -1,
                                               int realizations = 1000,
                                               double alpha = 0.1) const {
        return k_nn_prediction_intervals(x_, y_, math::linalg::Matrix(x), seed, realizations,
                                          alpha);
    }

   private:
    // A structure for storing a k-NN item.
    struct KnnItem {
        int index = 0;
        double distance = 0.0;
    };

    // Sorts training rows by distance, then by original training index (note 1).
    static std::vector<KnnItem> sorted_items(const math::linalg::Matrix& x_train,
                                             const std::vector<double>& point) {
        std::vector<KnnItem> items(static_cast<std::size_t>(x_train.number_of_rows()));
        for (int idx = 0; idx < x_train.number_of_rows(); idx++) {
            items[static_cast<std::size_t>(idx)].index = idx;
            items[static_cast<std::size_t>(idx)].distance = distance(point, x_train.row(idx));
        }
        std::sort(items.begin(), items.end(), [](const KnnItem& a, const KnnItem& b) {
            if (a.distance != b.distance) return a.distance < b.distance;
            return a.index < b.index;
        });
        return items;
    }

    // Returns the indexes of the k nearest neighbors.
    std::optional<std::vector<int>> k_nn(const math::linalg::Matrix& x_train,
                                         const math::linalg::Vector& /*y_train*/,
                                         const math::linalg::Matrix& x_test) const {
        if (x_test.number_of_columns() != x_train.number_of_columns()) return std::nullopt;
        int r = x_test.number_of_rows();
        std::vector<int> result(static_cast<std::size_t>(r) * static_cast<std::size_t>(k_), 0);
        for (int i = 0; i < r; i++) {
            std::vector<KnnItem> items = sorted_items(x_train, x_test.row(i));
            for (int j = 0; j < k_; j++)
                result[static_cast<std::size_t>(i * k_ + j)] =
                    items[static_cast<std::size_t>(j)].index;
        }
        return result;
    }

    // Returns the prediction from the k nearest neighbors.
    std::optional<std::vector<double>> k_nn_predict(const math::linalg::Matrix& x_train,
                                                     const math::linalg::Vector& y_train,
                                                     const math::linalg::Matrix& x_test) const {
        if (x_test.number_of_columns() != x_train.number_of_columns()) return std::nullopt;
        int r = x_test.number_of_rows();
        std::vector<double> result(static_cast<std::size_t>(r), 0.0);
        for (int i = 0; i < r; i++) {
            std::vector<KnnItem> items = sorted_items(x_train, x_test.row(i));
            std::vector<double> knn(static_cast<std::size_t>(k_), 0.0);

            if (is_regression_ == true) {
                // The inverse-distance-weighted average of the k nearest neighbors' values
                // (note 3: each term is divided by `sum` separately).
                double sum = 0;
                double avg = 0;
                for (int j = 0; j < k_; j++) {
                    double d = items[static_cast<std::size_t>(j)].distance;
                    double w = d > 0 ? 1.0 / sqr(d) : 1;
                    knn[static_cast<std::size_t>(j)] =
                        y_train[items[static_cast<std::size_t>(j)].index] * w;
                    sum += w;
                }
                for (int j = 0; j < k_; j++) avg += knn[static_cast<std::size_t>(j)] / sum;

                result[static_cast<std::size_t>(i)] = avg;
            } else {
                // The most common value (note 4).
                for (int j = 0; j < k_; j++)
                    knn[static_cast<std::size_t>(j)] =
                        y_train[items[static_cast<std::size_t>(j)].index];

                result[static_cast<std::size_t>(i)] =
                    support::most_common_first_appearance_wins(knn);
            }
        }
        return result;
    }

    // Returns the prediction from one bootstrap resample of the training set.
    std::optional<std::vector<double>> k_nn_bootstrap_predict(
        const math::linalg::Matrix& x_train, const math::linalg::Vector& y_train,
        const math::linalg::Matrix& x_test, int seed = -1) const {
        sampling::MersenneTwister rnd =
            seed > 0 ? sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                     : sampling::MersenneTwister();
        std::vector<int> idxs =
            utilities::next_integers(rnd, 0, x_train.number_of_rows(), x_train.number_of_rows());
        math::linalg::Matrix boot_x(x_train.number_of_rows(), x_train.number_of_columns());
        math::linalg::Vector boot_y(y_train.length());
        for (int i = 0; i < x_train.number_of_rows(); i++) {
            for (int j = 0; j < x_train.number_of_columns(); j++)
                boot_x(i, j) = x_train(idxs[static_cast<std::size_t>(i)], j);
            boot_y[i] = y_train[idxs[static_cast<std::size_t>(i)]];
        }
        return k_nn_predict(boot_x, boot_y, x_test);
    }

    // Returns the bootstrapped prediction intervals.
    math::linalg::Matrix k_nn_prediction_intervals(const math::linalg::Matrix& x_train,
                                                    const math::linalg::Vector& y_train,
                                                    const math::linalg::Matrix& x_test,
                                                    int seed = -1, int realizations = 1000,
                                                    double alpha = 0.1) const {
        double percentiles[3] = {alpha / 2.0, 0.5, 1.0 - alpha / 2.0};
        math::linalg::Matrix output(x_test.number_of_rows(), 4);  // lower, median, upper, mean

        std::vector<std::vector<double>> boot_results(
            static_cast<std::size_t>(x_test.number_of_rows()),
            std::vector<double>(static_cast<std::size_t>(realizations), 0.0));
        sampling::MersenneTwister rnd =
            seed > 0 ? sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                     : sampling::MersenneTwister();
        std::vector<int> seeds = utilities::next_integers(rnd, realizations);

        // Bootstrap the predictions (note 6: serial, and exact -- the seeds are drawn up front).
        for (int idx = 0; idx < realizations; idx++) {
            std::optional<std::vector<double>> column = k_nn_bootstrap_predict(
                x_train, y_train, x_test, seeds[static_cast<std::size_t>(idx)]);
            // `PredictionIntervals` has no shape guard of its own (upstream has none either), so a
            // query with the wrong column count reaches here as a null. C# dereferences it with
            // `!` and throws NullReferenceException; throwing beats undefined behavior.
            if (!column.has_value())
                throw std::runtime_error(
                    "KNearestNeighbors::prediction_intervals: the query matrix has " +
                    std::to_string(x_test.number_of_columns()) + " columns, expected " +
                    std::to_string(x_train.number_of_columns()));
            for (int i = 0; i < x_test.number_of_rows(); i++)
                boot_results[static_cast<std::size_t>(i)][static_cast<std::size_t>(idx)] =
                    (*column)[static_cast<std::size_t>(i)];
        }

        // Process the results. Note 5: there is no classification branch here.
        for (int idx = 0; idx < x_test.number_of_rows(); idx++) {
            std::vector<double> values = boot_results[static_cast<std::size_t>(idx)];
            std::sort(values.begin(), values.end());

            for (int j = 0; j < 3; j++)
                output(idx, j) = data::percentile(values, percentiles[j], true);

            output(idx, 3) = data::mean(values);
        }

        return output;
    }

    int k_;
    math::linalg::Vector y_;
    math::linalg::Matrix x_;
    bool is_regression_ = true;
};

}  // namespace corehydro::numerics::machine_learning
