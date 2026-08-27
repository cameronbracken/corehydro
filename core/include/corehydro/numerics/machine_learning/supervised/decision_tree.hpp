// ported from: Numerics/Machine Learning/Supervised/DecisionTree.cs @ 2a0357a
//
// The decision-tree learning algorithm, for regression (splits on variance reduction) or
// classification (splits on information gain).
//
// Reference (upstream's): https://en.wikipedia.org/wiki/Decision_tree_learning
//
// Six transcription notes, each on something a "cleanup" would silently change:
//
// 1. `GrowTree` CALLS `BestSplit` BEFORE TESTING THE STOPPING CRITERIA, so the PRNG advances by
//    one `next_integers(0, Dimensions, Features, false)` draw at EVERY node -- including the
//    leaves whose stopping test fires immediately afterwards. Hoisting the stopping test for
//    efficiency changes the random stream and therefore every seeded oracle downstream, including
//    every RandomForest fit.
// 2. `numberOfLabels` is `yTrain.Length` for regression (so the `<= 1` stopping test only fires
//    on a single-row node) and the DISTINCT count for classification.
// 3. `BestSplit` seeds `best` with `double.MinValue` and uses a strict `>`; `VarianceReduction`
//    and `InformationGain` also RETURN `double.MinValue` for a degenerate split (one side empty).
//    So when every candidate is degenerate, `bestFeatureIndex` stays -1 and the node becomes a
//    leaf; and when two candidates tie, the EARLIER one wins. Use
//    `std::numeric_limits<double>::lowest()`, not `-INFINITY`, or a `-inf` performance would
//    compare equal to the seed and the first candidate would never be taken.
// 4. Regression thresholds are the RAW COLUMN (duplicates included, evaluated repeatedly);
//    classification thresholds are `x.Distinct()` in first-appearance order (see
//    support/linq_order.hpp). The regression redundancy is stream-neutral but the evaluation
//    ORDER decides ties under note 3, so both are transcribed as written.
// 5. The classification leaf value is the first-appearance-wins mode (again linq_order.hpp), and
//    the classification `Entropy` builds an O(n) relative-frequency lambda that
//    `Statistics.Entropy` then calls once per point -- O(n^2) per candidate threshold. The
//    regression `Entropy` branch constructs a `KernelDensity` over `y`; nothing reaches it in a
//    regression fit (regression splits on variance reduction, never on entropy), but it is
//    reachable if a caller flips `is_regression` between `train()` calls, so it is ported.
// 6. `TraverseTree` falls back to `node.Value` -- NaN for an internal node -- when a child is
//    null, rather than throwing.
//
// The `double[,]` constructor overload is not ported: C++ has no overload-resolution reason for
// it, and `Matrix::from_columns` / the flattened `Matrix(rows, cols, flat)` ctor cover both
// caller shapes explicitly (see matrix.hpp on why a `vector<vector<double>>` ctor pair would be
// ambiguous).
#pragma once
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/kernel_density.hpp"
#include "corehydro/numerics/machine_learning/support/decision_node.hpp"
#include "corehydro/numerics/machine_learning/support/linq_order.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::numerics::machine_learning {

class DecisionTree {
   public:
    // Creates a new decision tree over a single predictor column. A `seed` of zero or less
    // clock-seeds the generator.
    DecisionTree(const std::vector<double>& x, const std::vector<double>& y, int seed = -1)
        : DecisionTree(math::linalg::Matrix(x), math::linalg::Vector(y), seed) {}

    // Creates a new decision tree.
    DecisionTree(const math::linalg::Matrix& x, const math::linalg::Vector& y, int seed = -1)
        : y_(y),
          x_(x),
          dimensions_(x.number_of_columns()),
          features_(std::max(1, x.number_of_columns() - 1)),
          root_(std::make_shared<DecisionNode>()),
          random_(seed > 0 ? sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                           : sampling::MersenneTwister()) {
        if (y_.length() != x_.number_of_rows())
            throw std::invalid_argument("The y vector must be the same length as the x matrix.");
        if (y_.length() < 10)
            throw std::invalid_argument("There must be at least ten training data points.");
    }

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

    // The random number generator used within the tree estimation.
    sampling::MersenneTwister& random() { return random_; }

    // The root node of the decision tree.
    const std::shared_ptr<DecisionNode>& root() const { return root_; }

    // The training vector of response values.
    const math::linalg::Vector& y() const { return y_; }

    // The training matrix of predictor values.
    const math::linalg::Matrix& x() const { return x_; }

    // Determines whether this is for regression or classification. Default = regression.
    bool is_regression() const { return is_regression_; }
    void set_is_regression(bool value) { is_regression_ = value; }

    // Determines if the decision tree has been trained.
    bool is_trained() const { return is_trained_; }

    // Trains the decision tree.
    void train() {
        is_trained_ = false;
        features_ = std::min(features_, dimensions_);
        root_ = grow_tree(x_, y_);
        is_trained_ = true;
    }

    // Returns the prediction for each row of `x`. Returns an empty optional (the C# null) if the
    // tree is untrained or `x` has the wrong number of columns.
    std::optional<std::vector<double>> predict(const math::linalg::Matrix& x) const {
        if (!is_trained_ || x.number_of_columns() != dimensions_) return std::nullopt;
        std::vector<double> result(static_cast<std::size_t>(x.number_of_rows()), 0.0);
        for (int i = 0; i < x.number_of_rows(); i++)
            result[static_cast<std::size_t>(i)] = traverse_tree(x.row(i), *root_);
        return result;
    }

    // Convenience overload for a single predictor column (C# `Predict(double[])`).
    std::optional<std::vector<double>> predict(const std::vector<double>& x) const {
        return predict(math::linalg::Matrix(x));
    }

   private:
    // Grows the decision tree recursively.
    std::shared_ptr<DecisionNode> grow_tree(const math::linalg::Matrix& x_train,
                                            const math::linalg::Vector& y_train, int depth = 0) {
        int number_of_samples = x_train.number_of_rows();
        int number_of_labels =
            is_regression_
                ? y_train.length()
                : static_cast<int>(
                      support::distinct_in_first_appearance_order(y_train.to_array()).size());

        // Find the best split. NOTE (note 1): this runs BEFORE the stopping criteria, so the
        // PRNG advances at every node, leaves included.
        std::vector<int> feature_idxs =
            utilities::next_integers(random_, 0, dimensions_, features_, false);
        int best_index = 0;
        double best_threshold = 0;
        best_split(x_train, y_train.to_array(), feature_idxs, best_index, best_threshold);

        // Check the stopping criteria.
        if (best_index == -1 || depth >= max_depth_ || number_of_labels <= 1 ||
            number_of_samples < minimum_split_size_) {
            std::shared_ptr<DecisionNode> leaf = std::make_shared<DecisionNode>();
            leaf->is_leaf_node = true;
            if (is_regression_) {
                // If regression, return the average of Y.
                leaf->value = data::mean(y_train.to_array());
            } else {
                // If classification, return the most common value.
                leaf->value = support::most_common_first_appearance_wins(y_train.to_array());
            }
            return leaf;
        }

        // Create child nodes.
        std::vector<int> left_idxs;
        std::vector<int> right_idxs;
        split(x_train.column(best_index), best_threshold, left_idxs, right_idxs);

        // Split to the left.
        math::linalg::Matrix x_left(static_cast<int>(left_idxs.size()),
                                     x_train.number_of_columns());
        math::linalg::Vector y_left(static_cast<int>(left_idxs.size()));
        for (std::size_t i = 0; i < left_idxs.size(); i++) {
            y_left[static_cast<int>(i)] = y_train[left_idxs[i]];
            for (int j = 0; j < x_train.number_of_columns(); j++)
                x_left(static_cast<int>(i), j) = x_train(left_idxs[i], j);
        }
        std::shared_ptr<DecisionNode> left = grow_tree(x_left, y_left, depth + 1);

        // Split to the right.
        math::linalg::Matrix x_right(static_cast<int>(right_idxs.size()),
                                      x_train.number_of_columns());
        math::linalg::Vector y_right(static_cast<int>(right_idxs.size()));
        for (std::size_t i = 0; i < right_idxs.size(); i++) {
            y_right[static_cast<int>(i)] = y_train[right_idxs[i]];
            for (int j = 0; j < x_train.number_of_columns(); j++)
                x_right(static_cast<int>(i), j) = x_train(right_idxs[i], j);
        }
        std::shared_ptr<DecisionNode> right = grow_tree(x_right, y_right, depth + 1);

        // Return the decision node.
        std::shared_ptr<DecisionNode> node = std::make_shared<DecisionNode>();
        node->feature_index = best_index;
        node->threshold = best_threshold;
        node->left = left;
        node->right = right;
        return node;
    }

    // Returns the best split feature index and threshold.
    void best_split(const math::linalg::Matrix& x_train, const std::vector<double>& y_train,
                    const std::vector<int>& indices, int& best_feature_index,
                    double& best_threshold) const {
        double best = std::numeric_limits<double>::lowest();
        best_feature_index = -1;
        best_threshold = 0;

        for (std::size_t i = 0; i < indices.size(); i++) {
            std::vector<double> x = x_train.column(indices[i]);
            // Note 4: regression evaluates the raw column, classification the distinct values.
            std::vector<double> thresholds =
                is_regression_ ? x : support::distinct_in_first_appearance_order(x);
            for (std::size_t j = 0; j < thresholds.size(); j++) {
                // Test the split's variance reduction or information gain.
                double performance = is_regression_
                                          ? variance_reduction(x, y_train, thresholds[j])
                                          : information_gain(x, y_train, thresholds[j]);
                // Keep track of the best value (strict `>` -- earlier candidates win ties).
                if (performance > best) {
                    best = performance;
                    best_feature_index = indices[i];
                    best_threshold = thresholds[j];
                }
            }
        }
    }

    // Computes the variance reduction for the threshold.
    double variance_reduction(const std::vector<double>& x, const std::vector<double>& y,
                              double threshold) const {
        // Parent variance.
        double parent_variance = data::population_variance(y);

        // Create children.
        std::vector<int> left_idxs;
        std::vector<int> right_idxs;
        split(x, threshold, left_idxs, right_idxs);

        if (left_idxs.empty() || right_idxs.empty())
            return std::numeric_limits<double>::lowest();

        // Calculate the weighted average variance of the children.
        double n = static_cast<double>(y.size());
        double n_left = static_cast<double>(left_idxs.size());
        double n_right = static_cast<double>(right_idxs.size());
        std::vector<double> y_left(left_idxs.size());
        for (std::size_t i = 0; i < left_idxs.size(); i++)
            y_left[i] = y[static_cast<std::size_t>(left_idxs[i])];
        std::vector<double> y_right(right_idxs.size());
        for (std::size_t i = 0; i < right_idxs.size(); i++)
            y_right[i] = y[static_cast<std::size_t>(right_idxs[i])];
        double var_left = data::population_variance(y_left);
        double var_right = data::population_variance(y_right);
        double child_variance = n_left / n * var_left + n_right / n * var_right;

        // Return the variance reduction.
        return parent_variance - child_variance;
    }

    // Returns the information gain of the split threshold.
    double information_gain(const std::vector<double>& x, const std::vector<double>& y,
                            double threshold) const {
        // Parent entropy.
        double parent_e = entropy(y);

        // Create children.
        std::vector<int> left_idxs;
        std::vector<int> right_idxs;
        split(x, threshold, left_idxs, right_idxs);

        if (left_idxs.empty() || right_idxs.empty())
            return std::numeric_limits<double>::lowest();

        // Calculate the weighted average entropy of the children.
        double n = static_cast<double>(y.size());
        double nl = static_cast<double>(left_idxs.size());
        double nr = static_cast<double>(right_idxs.size());
        std::vector<double> yl(left_idxs.size());
        for (std::size_t i = 0; i < left_idxs.size(); i++)
            yl[i] = y[static_cast<std::size_t>(left_idxs[i])];
        std::vector<double> yr(right_idxs.size());
        for (std::size_t i = 0; i < right_idxs.size(); i++)
            yr[i] = y[static_cast<std::size_t>(right_idxs[i])];
        double el = entropy(yl);
        double er = entropy(yr);
        double children_e = nl / n * el + nr / n * er;

        return parent_e - children_e;
    }

    // Computes the entropy for a vector of y-values (see transcription note 5).
    double entropy(const std::vector<double>& y) const {
        if (is_regression_ == true) {
            // Use kernel density.
            distributions::KernelDensity kde(y);
            return data::entropy(y, [&kde](double x) { return kde.pdf(x); });
        }
        // Use a histogram.
        return data::entropy(y, [&y](double x) {
            double n = 0;
            for (std::size_t i = 0; i < y.size(); i++) {
                if (x == y[i]) n++;
            }
            return n / static_cast<double>(y.size());
        });
    }

    // Splits the x-column on the threshold.
    static void split(const std::vector<double>& x, double threshold,
                      std::vector<int>& left_indices, std::vector<int>& right_indices) {
        left_indices.clear();
        right_indices.clear();
        for (std::size_t i = 0; i < x.size(); i++) {
            if (x[i] <= threshold) {
                left_indices.push_back(static_cast<int>(i));
            } else {
                right_indices.push_back(static_cast<int>(i));
            }
        }
    }

    // Traverses the tree and returns the leaf node value.
    static double traverse_tree(const std::vector<double>& x, const DecisionNode& node) {
        if (node.is_leaf_node == true) return node.value;
        if (x[static_cast<std::size_t>(node.feature_index)] <= node.threshold)
            return node.left != nullptr ? traverse_tree(x, *node.left) : node.value;
        return node.right != nullptr ? traverse_tree(x, *node.right) : node.value;
    }

    math::linalg::Vector y_;
    math::linalg::Matrix x_;
    int dimensions_;
    int features_;
    std::shared_ptr<DecisionNode> root_;
    sampling::MersenneTwister random_;
    int minimum_split_size_ = 2;
    int max_depth_ = 100;
    bool is_regression_ = true;
    bool is_trained_ = false;
};

}  // namespace corehydro::numerics::machine_learning
