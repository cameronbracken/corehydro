// ported from: Numerics/Machine Learning/Supervised/DecisionTree.cs @ 7e8e8d1
//
// Decision tree for regression or classification. Training carries a multiset of source-row
// indices through the recursion, sorts each candidate feature once, and evaluates every distinct
// threshold in one sweep. Bootstrap trees therefore share the parent data and store only indices.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/machine_learning/support/decision_node.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::numerics::machine_learning {

class DecisionTree {
   public:
    DecisionTree(const std::vector<double>& x, const std::vector<double>& y, int seed = -1)
        : DecisionTree(math::linalg::Matrix(x), math::linalg::Vector(y), seed) {}

    DecisionTree(const math::linalg::Matrix& x, const math::linalg::Vector& y, int seed = -1)
        : DecisionTree(std::make_shared<const math::linalg::Matrix>(x),
                       std::make_shared<const math::linalg::Vector>(y), std::vector<int>{}, seed,
                       false) {}

    DecisionTree(std::shared_ptr<const math::linalg::Matrix> x,
                 std::shared_ptr<const math::linalg::Vector> y,
                 std::vector<int> sample_indices, int seed)
        : DecisionTree(std::move(x), std::move(y), std::move(sample_indices), seed, true) {}

    int minimum_split_size() const { return minimum_split_size_; }
    void set_minimum_split_size(int value) { minimum_split_size_ = value; }
    int max_depth() const { return max_depth_; }
    void set_max_depth(int value) { max_depth_ = value; }
    int dimensions() const { return dimensions_; }
    int features() const { return features_; }
    void set_features(int value) { features_ = value; }
    sampling::MersenneTwister& random() { return random_; }
    const std::shared_ptr<DecisionNode>& root() const { return root_; }
    const math::linalg::Vector& y() const { return *y_; }
    const math::linalg::Matrix& x() const { return *x_; }
    bool is_regression() const { return is_regression_; }
    void set_is_regression(bool value) { is_regression_ = value; }
    bool is_trained() const { return is_trained_; }

    void train() {
        is_trained_ = false;
        features_ = std::min(features_, dimensions_);
        std::vector<int> indices;
        if (has_sample_indices_) {
            indices = sample_indices_;
        } else {
            indices.resize(static_cast<std::size_t>(x_->number_of_rows()));
            for (int i = 0; i < x_->number_of_rows(); ++i) indices[static_cast<std::size_t>(i)] = i;
        }
        partition_scratch_.resize(indices.size());
        root_ = grow_tree(indices, 0, static_cast<int>(indices.size()), 0);
        partition_scratch_.clear();
        is_trained_ = true;
    }

    std::optional<std::vector<double>> predict(const math::linalg::Matrix& x) const {
        if (!is_trained_ || x.number_of_columns() != dimensions_) return std::nullopt;
        std::vector<double> result(static_cast<std::size_t>(x.number_of_rows()));
        for (int i = 0; i < x.number_of_rows(); ++i)
            result[static_cast<std::size_t>(i)] = predict_row(x.row(i));
        return result;
    }

    std::optional<std::vector<double>> predict(const std::vector<double>& x) const {
        return predict(math::linalg::Matrix(x));
    }

    double predict_row(const std::vector<double>& x) const { return traverse_tree(x, *root_); }

   private:
    struct SortedValue {
        double key;
        double value;
        int order;
    };

    DecisionTree(std::shared_ptr<const math::linalg::Matrix> x,
                 std::shared_ptr<const math::linalg::Vector> y,
                 std::vector<int> sample_indices, int seed, bool has_sample_indices)
        : y_(std::move(y)),
          x_(std::move(x)),
          dimensions_(x_->number_of_columns()),
          features_(std::max(1, x_->number_of_columns() - 1)),
          root_(std::make_shared<DecisionNode>()),
          random_(seed > 0 ? sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                           : sampling::MersenneTwister()),
          sample_indices_(std::move(sample_indices)),
          has_sample_indices_(has_sample_indices) {
        if (y_->length() != x_->number_of_rows())
            throw std::invalid_argument("The y vector must be the same length as the x matrix.");
        if (y_->length() < 10)
            throw std::invalid_argument("There must be at least ten training data points.");
    }

    static bool same_label(double a, double b) {
        return a == b || (std::isnan(a) && std::isnan(b));
    }

    static double multiply_subtract(double a, double b, double c) {
        // The C# recurrence rounds the product before subtracting.
        volatile double product = a * b;
        return product - c;
    }

    static double update_variance_accumulator(double accumulator, double difference,
                                              double denominator) {
        // Avoid contraction across the division so split ties match the C# sweep.
        volatile double square = difference * difference;
        volatile double increment = square / denominator;
        return accumulator + increment;
    }

    int count_distinct_labels(const std::vector<int>& indices, int lo, int hi) const {
        std::vector<double> distinct;
        for (int i = lo; i < hi; ++i) {
            double value = (*y_)[indices[static_cast<std::size_t>(i)]];
            bool found = false;
            for (double prior : distinct) found = found || same_label(value, prior);
            if (!found) distinct.push_back(value);
        }
        return static_cast<int>(distinct.size());
    }

    std::shared_ptr<DecisionNode> create_leaf(const std::vector<int>& indices, int lo,
                                              int hi) const {
        auto leaf = std::make_shared<DecisionNode>();
        leaf->is_leaf_node = true;
        if (is_regression_) {
            double sum = 0.0;
            for (int i = lo; i < hi; ++i) sum += (*y_)[indices[static_cast<std::size_t>(i)]];
            leaf->value = sum / (hi - lo);
            return leaf;
        }
        std::vector<double> labels;
        std::vector<int> counts;
        for (int i = lo; i < hi; ++i) {
            double value = (*y_)[indices[static_cast<std::size_t>(i)]];
            std::size_t at = 0;
            while (at < labels.size() && !same_label(value, labels[at])) ++at;
            if (at == labels.size()) {
                labels.push_back(value);
                counts.push_back(1);
            } else {
                ++counts[at];
            }
        }
        int best = -1;
        for (std::size_t i = 0; i < labels.size(); ++i) {
            if (counts[i] > best) {
                best = counts[i];
                leaf->value = labels[i];
            }
        }
        return leaf;
    }

    std::shared_ptr<DecisionNode> grow_tree(std::vector<int>& indices, int lo, int hi,
                                            int depth) {
        int samples = hi - lo;
        int labels = count_distinct_labels(indices, lo, hi);
        std::vector<int> feature_indices =
            utilities::next_integers(random_, 0, dimensions_, features_, false);
        if (depth >= max_depth_ || labels <= 1 || samples < minimum_split_size_)
            return create_leaf(indices, lo, hi);

        int best_feature = -1;
        double best_threshold = 0.0;
        best_split(indices, lo, hi, feature_indices, best_feature, best_threshold);
        if (best_feature == -1) return create_leaf(indices, lo, hi);

        int split = stable_partition(indices, lo, hi, best_feature, best_threshold);
        auto left = grow_tree(indices, lo, split, depth + 1);
        auto right = grow_tree(indices, split, hi, depth + 1);
        auto node = std::make_shared<DecisionNode>();
        node->feature_index = best_feature;
        node->threshold = best_threshold;
        node->left = std::move(left);
        node->right = std::move(right);
        return node;
    }

    double range_variance(const std::vector<int>& indices, int lo, int hi) const {
        int count = 0;
        double sum = 0.0;
        double acc = 0.0;
        for (int i = lo; i < hi; ++i) {
            ++count;
            double value = (*y_)[indices[static_cast<std::size_t>(i)]];
            if (count == 1) {
                sum = value;
            } else {
                sum += value;
                double diff = multiply_subtract(count, value, sum);
                acc = update_variance_accumulator(acc, diff, count * (count - 1.0));
            }
        }
        return count == 0 ? std::numeric_limits<double>::quiet_NaN() : acc / count;
    }

    static double counts_entropy(const std::vector<std::pair<double, int>>& counts, int size) {
        double sum = 0.0;
        for (const auto& entry : counts) {
            if (entry.second <= 0) continue;
            double p = static_cast<double>(entry.second) / size;
            sum += entry.second * p * std::log(p);
        }
        return -sum;
    }

    static void increment_count(std::vector<std::pair<double, int>>& counts, double value,
                                int amount) {
        for (auto& entry : counts) {
            if (same_label(entry.first, value)) {
                entry.second += amount;
                return;
            }
        }
        counts.emplace_back(value, amount);
    }

    bool sweep_variance(const std::vector<int>& indices, int lo, int hi, int feature,
                        const std::vector<SortedValue>& sorted, int nan_count,
                        double& best_gain, double& best_threshold) const {
        int m = hi - lo;
        int valid = static_cast<int>(sorted.size());
        std::vector<double> right_acc(static_cast<std::size_t>(valid + 1));
        int right_seed = 0;
        double acc_right = 0.0;
        double sum_right = 0.0;
        if (nan_count > 0) {
            for (int i = lo; i < hi; ++i) {
                int row = indices[static_cast<std::size_t>(i)];
                if (!std::isnan((*x_)(row, feature))) continue;
                ++right_seed;
                double value = (*y_)[row];
                if (right_seed == 1) {
                    sum_right = value;
                } else {
                    sum_right += value;
                    double diff = multiply_subtract(right_seed, value, sum_right);
                    acc_right = update_variance_accumulator(
                        acc_right, diff, right_seed * (right_seed - 1.0));
                }
            }
        }
        right_acc[static_cast<std::size_t>(valid)] = acc_right;
        for (int i = valid - 1; i >= 0; --i) {
            int k = right_seed + valid - i;
            double value = sorted[static_cast<std::size_t>(i)].value;
            if (k == 1) {
                sum_right = value;
            } else {
                sum_right += value;
                double diff = multiply_subtract(k, value, sum_right);
                acc_right = update_variance_accumulator(acc_right, diff, k * (k - 1.0));
            }
            right_acc[static_cast<std::size_t>(i)] = acc_right;
        }

        best_gain = std::numeric_limits<double>::lowest();
        best_threshold = 0.0;
        bool found = false;
        int count_left = 0;
        double sum_left = 0.0;
        double acc_left = 0.0;
        double parent = range_variance(indices, lo, hi);
        for (int i = 0; i < valid; ++i) {
            ++count_left;
            double value = sorted[static_cast<std::size_t>(i)].value;
            if (count_left == 1) {
                sum_left = value;
            } else {
                sum_left += value;
                double diff = multiply_subtract(count_left, value, sum_left);
                acc_left = update_variance_accumulator(
                    acc_left, diff, count_left * (count_left - 1.0));
            }
            if (i + 1 < valid && sorted[static_cast<std::size_t>(i + 1)].key ==
                                         sorted[static_cast<std::size_t>(i)].key)
                continue;
            int count_right = m - count_left;
            if (count_right == 0) break;
            // C# rounds both products before adding the weighted child variances.
            volatile double left_weighted =
                count_left / static_cast<double>(m) * (acc_left / count_left);
            volatile double right_weighted =
                count_right / static_cast<double>(m) *
                (right_acc[static_cast<std::size_t>(i + 1)] / count_right);
            double child = left_weighted + right_weighted;
            double gain = parent - child;
            if (gain > best_gain) {
                best_gain = gain;
                best_threshold = sorted[static_cast<std::size_t>(i)].key;
                found = true;
            }
        }
        return found;
    }

    bool sweep_entropy(const std::vector<int>& indices, int lo, int hi,
                       const std::vector<SortedValue>& sorted, double& best_gain,
                       double& best_threshold) const {
        int m = hi - lo;
        std::vector<std::pair<double, int>> right_counts;
        for (int i = lo; i < hi; ++i) {
            double value = (*y_)[indices[static_cast<std::size_t>(i)]];
            if (!std::isnan(value)) increment_count(right_counts, value, 1);
        }
        double parent = counts_entropy(right_counts, m);
        std::vector<std::pair<double, int>> left_counts;
        best_gain = std::numeric_limits<double>::lowest();
        best_threshold = 0.0;
        bool found = false;
        int count_left = 0;
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            ++count_left;
            double value = sorted[i].value;
            if (!std::isnan(value)) {
                increment_count(left_counts, value, 1);
                increment_count(right_counts, value, -1);
            }
            if (i + 1 < sorted.size() && sorted[i + 1].key == sorted[i].key) continue;
            int count_right = m - count_left;
            if (count_right == 0) break;
            double child = count_left / static_cast<double>(m) *
                               counts_entropy(left_counts, count_left) +
                           count_right / static_cast<double>(m) *
                               counts_entropy(right_counts, count_right);
            double gain = parent - child;
            if (gain > best_gain) {
                best_gain = gain;
                best_threshold = sorted[i].key;
                found = true;
            }
        }
        return found;
    }

    void best_split(const std::vector<int>& indices, int lo, int hi,
                    const std::vector<int>& feature_indices, int& best_feature,
                    double& best_threshold) const {
        double best = std::numeric_limits<double>::lowest();
        best_feature = -1;
        best_threshold = 0.0;
        for (int feature : feature_indices) {
            std::vector<SortedValue> sorted;
            sorted.reserve(static_cast<std::size_t>(hi - lo));
            int nan_count = 0;
            for (int i = lo; i < hi; ++i) {
                int row = indices[static_cast<std::size_t>(i)];
                double key = (*x_)(row, feature);
                if (std::isnan(key)) {
                    ++nan_count;
                } else {
                    sorted.push_back({key, (*y_)[row], i - lo});
                }
            }
            if (sorted.empty()) continue;
            std::sort(sorted.begin(), sorted.end(), [](const SortedValue& a, const SortedValue& b) {
                if (a.key < b.key) return true;
                if (b.key < a.key) return false;
                return a.order < b.order;
            });
            double gain = 0.0;
            double threshold = 0.0;
            bool found = is_regression_
                             ? sweep_variance(indices, lo, hi, feature, sorted, nan_count, gain,
                                              threshold)
                             : sweep_entropy(indices, lo, hi, sorted, gain, threshold);
            if (found && gain > best) {
                best = gain;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }

    int stable_partition(std::vector<int>& indices, int lo, int hi, int feature,
                         double threshold) {
        int left = lo;
        int right_count = 0;
        for (int i = lo; i < hi; ++i) {
            int row = indices[static_cast<std::size_t>(i)];
            if ((*x_)(row, feature) <= threshold) {
                indices[static_cast<std::size_t>(left++)] = row;
            } else {
                partition_scratch_[static_cast<std::size_t>(right_count++)] = row;
            }
        }
        for (int i = 0; i < right_count; ++i)
            indices[static_cast<std::size_t>(left + i)] =
                partition_scratch_[static_cast<std::size_t>(i)];
        return left;
    }

    static double traverse_tree(const std::vector<double>& x, const DecisionNode& node) {
        if (node.is_leaf_node) return node.value;
        if (x[static_cast<std::size_t>(node.feature_index)] <= node.threshold)
            return node.left ? traverse_tree(x, *node.left) : node.value;
        return node.right ? traverse_tree(x, *node.right) : node.value;
    }

    std::shared_ptr<const math::linalg::Vector> y_;
    std::shared_ptr<const math::linalg::Matrix> x_;
    int dimensions_;
    int features_;
    std::shared_ptr<DecisionNode> root_;
    sampling::MersenneTwister random_;
    std::vector<int> sample_indices_;
    bool has_sample_indices_ = false;
    std::vector<int> partition_scratch_;
    int minimum_split_size_ = 2;
    int max_depth_ = 100;
    bool is_regression_ = true;
    bool is_trained_ = false;
};

}  // namespace corehydro::numerics::machine_learning
