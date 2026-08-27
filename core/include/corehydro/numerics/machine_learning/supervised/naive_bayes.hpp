// ported from: Numerics/Machine Learning/Supervised/NaiveBayes.cs @ 2a0357a
//
// Gaussian naive Bayes classification: each feature is assumed conditionally normal given the
// class, and independent of the other features.
//
// Reference (upstream's): https://en.wikipedia.org/wiki/Naive_Bayes_classifier
//
// Four transcription notes, each on something a "cleanup" would silently change:
//
// 1. `Classes` is `y.Distinct()` in FIRST-APPEARANCE order (see support/linq_order.hpp), so
//    `Means[i]`, `StandardDeviations[i]` and `Priors[i]` index classes in the order they appear
//    in the training response, NOT in sorted order. The iris test data happens to arrive sorted
//    by species, which hides the dependence; a shuffled response would not.
// 2. The per-class standard deviation is the naive two-moment form
//    `sqrt(max(0, (u2 - u1^2) * n/(n-1)))`, NOT `Statistics.StandardDeviation`. It is
//    catastrophically cancelling for large-magnitude features -- and it is what the oracle sees,
//    so the stable running-difference recurrence must NOT be substituted. The `Math.Max(0, ...)`
//    is what keeps a cancelled negative from becoming NaN.
// 3. A class with one or zero members gets a hard-coded standard deviation of 1e-6 rather than
//    the formula (which would be 0/0). A class with zero members also gets `Means[i, j] = 0/0 =
//    NaN`, since the mean has no such guard.
// 4. `MAP` seeds `max` with `double.MinValue` and uses a strict `>`, so ties go to the FIRST
//    class in `Classes` order; and it constructs a fresh `Normal(mean, sd)` per feature per class
//    per prediction row rather than caching. The construction is kept -- it validates the
//    parameters, so a degenerate class surfaces as a throw rather than as a silent NaN.
//
// The `double[,]` constructor overload is not ported, for the same reason as in decision_tree.hpp.
#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/machine_learning/support/linq_order.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"

namespace corehydro::numerics::machine_learning {

class NaiveBayes {
   public:
    // Creates a new naive Bayes classifier over a single predictor column.
    NaiveBayes(const std::vector<double>& x, const std::vector<double>& y)
        : NaiveBayes(math::linalg::Matrix(x), math::linalg::Vector(y)) {}

    // Creates a new naive Bayes classifier.
    NaiveBayes(const math::linalg::Matrix& x, const math::linalg::Vector& y)
        : y_(y),
          x_(x),
          classes_(support::distinct_in_first_appearance_order(y.to_array())) {
        if (y_.length() != x_.number_of_rows())
            throw std::invalid_argument("The y vector must be the same length as the x matrix.");
        if (y_.length() < 10)
            throw std::invalid_argument("There must be at least ten training data points.");
        if (classes_.size() < 1)
            throw std::invalid_argument("There must be at least 1 class to predict.");
    }

    // The training vector of response values.
    const math::linalg::Vector& y() const { return y_; }

    // The training matrix of predictor values.
    const math::linalg::Matrix& x() const { return x_; }

    // The distinct classes from the training set, in first-appearance order (note 1).
    const std::vector<double>& classes() const { return classes_; }

    // The mean of each feature given each class: `means()[class][feature]`.
    const std::vector<std::vector<double>>& means() const { return means_; }

    // The standard deviation of each feature given each class.
    const std::vector<std::vector<double>>& standard_deviations() const {
        return standard_deviations_;
    }

    // The prior probability of each class.
    const std::vector<double>& priors() const { return priors_; }

    // Determines if the classifier has been trained.
    bool is_trained() const { return is_trained_; }

    // Trains the naive Bayes classifier.
    void train() {
        // Set up the training outputs.
        is_trained_ = false;
        int n_samples = x_.number_of_rows();
        int n_features = x_.number_of_columns();
        int n_classes = static_cast<int>(classes_.size());
        means_.assign(static_cast<std::size_t>(n_classes),
                      std::vector<double>(static_cast<std::size_t>(n_features), 0.0));
        standard_deviations_.assign(static_cast<std::size_t>(n_classes),
                                    std::vector<double>(static_cast<std::size_t>(n_features), 0.0));
        priors_.assign(static_cast<std::size_t>(n_classes), 0.0);

        for (int i = 0; i < n_classes; i++) {
            std::size_t is = static_cast<std::size_t>(i);
            // Compute the priors as the relative frequency of each class.
            for (int k = 0; k < n_samples; k++) {
                if (y_[k] == classes_[is]) priors_[is]++;
            }
            priors_[is] /= n_samples;

            // Compute the mean and standard deviation of each feature j given the class i.
            for (int j = 0; j < n_features; j++) {
                std::size_t js = static_cast<std::size_t>(j);
                double x = 0;   // sum
                double x2 = 0;  // sum of X^2
                double u1, u2;
                double n = 0;
                // Compute the sums.
                for (int k = 0; k < n_samples; k++) {
                    if (y_[k] == classes_[is]) {
                        x += x_(k, j);
                        x2 += std::pow(x_(k, j), 2);
                        n++;
                    }
                }
                // Compute the averages.
                u1 = x / n;
                u2 = x2 / n;
                // Set the means.
                means_[is][js] = u1;
                // Set the standard deviations (notes 2 and 3).
                if (n <= 1) {
                    standard_deviations_[is][js] = 1e-6;
                } else {
                    standard_deviations_[is][js] =
                        std::sqrt(std::max(0.0, (u2 - std::pow(u1, 2.0)) * (n / (n - 1))));
                }
            }
        }

        is_trained_ = true;
    }

    // Returns the predicted class for each row of `x`. Returns an empty optional (the C# null) if
    // the classifier is untrained or `x` has the wrong number of columns.
    std::optional<std::vector<double>> predict(const math::linalg::Matrix& x) const {
        if (!is_trained_ || x.number_of_columns() != x_.number_of_columns()) return std::nullopt;
        std::vector<double> result(static_cast<std::size_t>(x.number_of_rows()), 0.0);
        for (int i = 0; i < x.number_of_rows(); i++)
            result[static_cast<std::size_t>(i)] = map(x.row(i));
        return result;
    }

    // Convenience overload for a single predictor column (C# `Predict(double[])`).
    std::optional<std::vector<double>> predict(const std::vector<double>& x) const {
        return predict(math::linalg::Matrix(x));
    }

   private:
    // Returns the class with the maximum posterior probability (note 4).
    //
    // P(y)   = the prior probability, the relative frequency of each class.
    // P(x|y) = the conditional probability, from the Normal PDF.
    double map(const std::vector<double>& x) const {
        int n_features = x_.number_of_columns();
        int n_classes = static_cast<int>(classes_.size());
        double max = std::numeric_limits<double>::lowest();
        int max_idx = 0;

        for (int i = 0; i < n_classes; i++) {
            std::size_t is = static_cast<std::size_t>(i);
            // Compute the log-likelihood of each class.
            double log_lh = std::log(priors_[is]);
            for (int j = 0; j < n_features; j++) {
                std::size_t js = static_cast<std::size_t>(j);
                distributions::Normal norm(means_[is][js], standard_deviations_[is][js]);
                log_lh += norm.log_pdf(x[js]);
            }
            // Keep track of the maximum (strict `>` -- ties go to the earlier class).
            if (log_lh > max) {
                max = log_lh;
                max_idx = i;
            }
        }
        return classes_[static_cast<std::size_t>(max_idx)];
    }

    math::linalg::Vector y_;
    math::linalg::Matrix x_;
    std::vector<double> classes_;
    std::vector<std::vector<double>> means_;
    std::vector<std::vector<double>> standard_deviations_;
    std::vector<double> priors_;
    bool is_trained_ = false;
};

}  // namespace corehydro::numerics::machine_learning
