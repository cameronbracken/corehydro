// ported from: Numerics/Machine Learning/Supervised/RandomForest.cs @ 2a0357a
//
// Random forest for regression and classification: an ensemble of bootstrapped decision trees
// whose per-row spread becomes a prediction interval.
//
// Reference (upstream's): https://en.wikipedia.org/wiki/Random_forest
//
// Five transcription notes, each on something a "cleanup" would silently change:
//
// 1. `Parallel.For` -> SERIAL LOOP, and that is exact. `Train` draws ALL `NumberOfTrees` seeds up
//    front with `Random.NextIntegers(NumberOfTrees)` and only then builds each tree from its own
//    generator, so the parallel loop is order-independent and a serial one reproduces it bit for
//    bit. The same holds for both `Parallel.For`s in `Predict`, which write only their own index.
// 2. `BootstrapDecisionTree` passes the SAME `seed` twice: once to its local resampling generator
//    and once to the `DecisionTree` constructor. Those are TWO SEPARATE generator instances, so
//    the tree's feature draws start from a fresh stream while the resampling stream has already
//    consumed `X.NumberOfRows` draws. Sharing one generator between them -- the obvious
//    "cleanup" -- changes every seeded oracle.
// 3. `Predict`'s classification branch applies `Math.Floor` to each percentile AND to the mean,
//    so all four columns come back integral.
// 4. `Predict` computes the three percentile columns with `Statistics.Percentile(values, p,
//    true)` over an already-sorted row, but the mean column with `Statistics.ParallelMean`. That
//    method's PLINQ partitioned sum makes its last bits depend on the machine's core count -- see
//    `numerics/data/statistics.hpp`'s `parallel_mean` note and
//    docs/upstream-csharp-issues.md. The port sums serially, so the mean column is the one value
//    here that is not bit-reproducible against C#.
// 4b. `Predict` does NOT guard `X.NumberOfColumns != Dimensions` the way `DecisionTree::Predict`
//    does; it only checks `IsTrained`. A too-narrow query matrix therefore reaches the trees,
//    each of which returns null from its own guard, and C# dereferences that null. The port keeps
//    the missing guard (so the shapes it DOES accept behave identically) but returns the empty
//    optional instead of dereferencing null -- see the note at the call site.
// 5. `MinimumSplitSize`, `MaxDepth`, `Features` and `IsRegression` are copied onto each tree at
//    construction, so changing them after `Train()` affects nothing until the next `Train()`.
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
#include "corehydro/numerics/machine_learning/supervised/decision_tree.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::numerics::machine_learning {

class RandomForest {
   public:
    // Creates a new random forest over a single predictor column. A `seed` of zero or less
    // clock-seeds the generator.
    RandomForest(const std::vector<double>& x, const std::vector<double>& y, int seed = -1)
        : RandomForest(math::linalg::Matrix(x), math::linalg::Vector(y), seed) {}

    // Creates a new random forest.
    RandomForest(const math::linalg::Matrix& x, const math::linalg::Vector& y, int seed = -1)
        : y_(y),
          x_(x),
          dimensions_(x.number_of_columns()),
          features_(std::max(1, x.number_of_columns() - 1)),
          random_(seed > 0 ? sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                           : sampling::MersenneTwister()) {
        if (y_.length() != x_.number_of_rows())
            throw std::invalid_argument("The y vector must be the same length as the x matrix.");
        if (y_.length() < 10)
            throw std::invalid_argument("There must be at least ten training data points.");
    }

    // The number of trees to use in the random forest. Default = 1000.
    int number_of_trees() const { return number_of_trees_; }
    void set_number_of_trees(int value) { number_of_trees_ = value; }

    // The minimum split size of the samples. Default = 2.
    int minimum_split_size() const { return minimum_split_size_; }
    void set_minimum_split_size(int value) { minimum_split_size_ = value; }

    // The maximum recursion depth. Default = 100.
    int max_depth() const { return max_depth_; }
    void set_max_depth(int value) { max_depth_ = value; }

    // The dimensionality (total number of features) of the data space.
    int dimensions() const { return dimensions_; }

    // The number of random sub-features to evaluate in the tree recursion.
    int features() const { return features_; }
    void set_features(int value) { features_ = value; }

    // The random number generator used to seed the trees.
    sampling::MersenneTwister& random() { return random_; }

    // The training vector of response values.
    const math::linalg::Vector& y() const { return y_; }

    // The training matrix of predictor values.
    const math::linalg::Matrix& x() const { return x_; }

    // The fitted decision trees.
    const std::vector<DecisionTree>& decision_trees() const { return decision_trees_; }

    // Determines whether this is for regression or classification. Default = regression.
    bool is_regression() const { return is_regression_; }
    void set_is_regression(bool value) { is_regression_ = value; }

    // Determines if the random forest has been trained.
    bool is_trained() const { return is_trained_; }

    // Trains the random forest.
    void train() {
        is_trained_ = false;
        features_ = std::min(features_, dimensions_);
        decision_trees_.clear();
        decision_trees_.reserve(static_cast<std::size_t>(number_of_trees_));
        // All seeds are drawn up front, which is what makes the C# Parallel.For reproducible and
        // this serial loop identical to it (transcription note 1).
        std::vector<int> seeds = utilities::next_integers(random_, number_of_trees_);

        for (int idx = 0; idx < number_of_trees_; idx++) {
            decision_trees_.push_back(bootstrap_decision_tree(seeds[static_cast<std::size_t>(idx)]));
            decision_trees_[static_cast<std::size_t>(idx)].train();
        }

        is_trained_ = true;
    }

    // Returns the prediction intervals as an n-by-4 matrix with columns lower, median, upper,
    // mean. `alpha` is the confidence level; the default 0.1 gives 90% intervals. Returns an
    // empty optional (the C# null) if the forest is untrained.
    std::optional<math::linalg::Matrix> predict(const math::linalg::Matrix& x,
                                                 double alpha = 0.1) const {
        if (!is_trained_) return std::nullopt;
        // Transcription note 4b: upstream has NO column-count guard here (unlike DecisionTree),
        // and would dereference the null each tree returns. Returning the empty optional keeps
        // every accepted shape identical while giving the rejected one defined behavior.
        if (x.number_of_columns() != dimensions_) return std::nullopt;

        double percentiles[3] = {alpha / 2.0, 0.5, 1.0 - alpha / 2.0};
        math::linalg::Matrix output(x.number_of_rows(), 4);  // lower, median, upper, mean

        // Bootstrap the predictions: boot_results[i][t] is tree t's prediction for row i.
        std::vector<std::vector<double>> boot_results(
            static_cast<std::size_t>(x.number_of_rows()),
            std::vector<double>(static_cast<std::size_t>(number_of_trees_), 0.0));
        for (int idx = 0; idx < number_of_trees_; idx++) {
            std::optional<std::vector<double>> column =
                decision_trees_[static_cast<std::size_t>(idx)].predict(x);
            // Unreachable given the guard above (every tree carries the same `dimensions_` and is
            // trained), but C# dereferences this null with `!` and would throw
            // NullReferenceException; throwing beats undefined behavior if the invariant ever
            // changes.
            if (!column.has_value())
                throw std::runtime_error("RandomForest::predict: a tree returned no prediction");
            for (int i = 0; i < x.number_of_rows(); i++)
                boot_results[static_cast<std::size_t>(i)][static_cast<std::size_t>(idx)] =
                    (*column)[static_cast<std::size_t>(i)];
        }

        // Process the results.
        for (int idx = 0; idx < x.number_of_rows(); idx++) {
            std::vector<double> values = boot_results[static_cast<std::size_t>(idx)];
            std::sort(values.begin(), values.end());

            if (is_regression_) {
                for (int j = 0; j < 3; j++)
                    output(idx, j) = data::percentile(values, percentiles[j], true);
                output(idx, 3) = data::parallel_mean(values);
            } else {
                // Transcription note 3: the classification branch floors all four columns.
                for (int j = 0; j < 3; j++)
                    output(idx, j) = std::floor(data::percentile(values, percentiles[j], true));
                output(idx, 3) = std::floor(data::parallel_mean(values));
            }
        }

        return output;
    }

    // Convenience overload for a single predictor column (C# `Predict(double[], double)`).
    std::optional<math::linalg::Matrix> predict(const std::vector<double>& x,
                                                 double alpha = 0.1) const {
        return predict(math::linalg::Matrix(x), alpha);
    }

   private:
    // Returns a bootstrapped decision tree. See transcription note 2 on the two generators.
    DecisionTree bootstrap_decision_tree(int seed = -1) const {
        sampling::MersenneTwister rnd =
            seed > 0 ? sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                     : sampling::MersenneTwister();
        std::vector<int> idxs =
            utilities::next_integers(rnd, 0, x_.number_of_rows(), x_.number_of_rows());
        math::linalg::Matrix boot_x(x_.number_of_rows(), x_.number_of_columns());
        math::linalg::Vector boot_y(y_.length());
        for (int i = 0; i < x_.number_of_rows(); i++) {
            for (int j = 0; j < x_.number_of_columns(); j++)
                boot_x(i, j) = x_(idxs[static_cast<std::size_t>(i)], j);
            boot_y[i] = y_[idxs[static_cast<std::size_t>(i)]];
        }
        DecisionTree tree(boot_x, boot_y, seed);
        tree.set_minimum_split_size(minimum_split_size_);
        tree.set_max_depth(max_depth_);
        tree.set_features(features_);
        tree.set_is_regression(is_regression_);
        return tree;
    }

    math::linalg::Vector y_;
    math::linalg::Matrix x_;
    int dimensions_;
    int features_;
    sampling::MersenneTwister random_;
    std::vector<DecisionTree> decision_trees_;
    int number_of_trees_ = 1000;
    int minimum_split_size_ = 2;
    int max_depth_ = 100;
    bool is_regression_ = true;
    bool is_trained_ = false;
};

}  // namespace corehydro::numerics::machine_learning
