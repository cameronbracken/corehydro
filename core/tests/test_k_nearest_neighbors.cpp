// P5 Task 9 -- KNearestNeighbors.
//
// Transcribes all four [TestMethod]s of
// upstream/Numerics/Test_Numerics/Machine Learning/Supervised/Test_kNN.cs @ 2a0357a.
//
// Unlike the DecisionTree and RandomForest suites, this one has a REAL oracle:
// Test_kNN_Iris asserts the full 60-value classification prediction vector against R's `class`
// package, and Test_kNN_Classification asserts an exact label.
//
// Test_GetNeighbors_MultiRow queries the exact center of a symmetric cluster where four training
// points tie on distance, so which neighbors come back depends on the sort's tie permutation --
// but its C# assertions are index-RANGE inequalities that any ordering satisfies. The COREHYDRO
// SUPPLEMENT pins the exact indices, measured against the real library. Note the honest limit
// recorded at that call site: at 12 training points the .NET introsort takes its stable
// insertion-sort path, so that case alone does not pin the introsort -- a separate 24-point case
// below the threshold does, and `std::sort` was verified to give a different answer there.
#include <cmath>
#include <optional>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/regression/linear_regression.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/machine_learning/supervised/k_nearest_neighbors.hpp"
#include "data/fpp3_dataset.hpp"
#include "data/iris_dataset.hpp"

namespace ml = corehydro::numerics::machine_learning;
namespace la = corehydro::numerics::math::linalg;
namespace nd = corehydro::numerics::data;
namespace iris = corehydro::testdata::iris;
namespace fpp3 = corehydro::testdata::fpp3;

namespace {

std::vector<double> subset(const std::vector<double>& v, int start, int end) {
    return std::vector<double>(v.begin() + start, v.begin() + end + 1);
}
std::vector<double> subset(const std::vector<double>& v, int start) {
    return std::vector<double>(v.begin() + start, v.end());
}

la::Matrix iris_train() {
    return la::Matrix::from_columns({iris::kSepalLengthTrain, iris::kSepalWidthTrain,
                                      iris::kPetalLengthTrain, iris::kPetalWidthTrain});
}
la::Matrix iris_test() {
    return la::Matrix::from_columns({iris::kSepalLengthTest, iris::kSepalWidthTest,
                                      iris::kPetalLengthTest, iris::kPetalWidthTest});
}

// --- Transcribed from Test_kNN.cs ---------------------------------------------------------

void test_knn_iris() {
    ml::KNearestNeighbors knn(iris_train(), la::Vector(iris::kSpeciesTrain), 5);
    knn.set_is_regression(false);

    std::optional<std::vector<double>> prediction = knn.predict(iris_test());
    CHECK_TRUE(prediction.has_value());

    // The full 60-value expected vector from the C# test (R 'class' package output).
    const double true_pred[60] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 3, 2, 3, 3,
                                  3, 3, 3, 3, 2, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3};
    CHECK_EQ(static_cast<int>(prediction->size()), 60);
    for (std::size_t i = 0; i < prediction->size(); i++)
        CHECK_EQ((*prediction)[i], true_pred[i]);
}

void test_knn_classification() {
    std::vector<double> y = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<double> x1 = {1, 2, 3, 3, 3.5, 2, 2, 1, 5, 3, 1.5, 7, 6, 3.8, 5.6, 4, 2};
    std::vector<double> x2 = {12, 5, 6, 10, 8, 11, 9, 7, 3, 2, 9, 2, 1, 1, 4, 2, 5};

    ml::KNearestNeighbors knn(la::Matrix::from_columns({x1, x2}), la::Vector(y), 3);
    knn.set_is_regression(false);
    std::optional<std::vector<double>> pred = knn.predict(la::Matrix(1, 2, {2.5, 7.0}));
    CHECK_TRUE(pred.has_value());
    CHECK_EQ((*pred)[0], 0.0);
}

void test_knn_regression() {
    // The C# test standardizes each predictor before fitting.
    std::vector<double> income = nd::standardize(fpp3::kIncome);
    std::vector<double> production = nd::standardize(fpp3::kProduction);
    std::vector<double> savings = nd::standardize(fpp3::kSavings);
    std::vector<double> unemployment = nd::standardize(fpp3::kUnemployment);

    const int t_idx = 118;
    la::Matrix x_training = la::Matrix::from_columns(
        {subset(income, 0, t_idx), subset(production, 0, t_idx), subset(savings, 0, t_idx),
         subset(unemployment, 0, t_idx)});
    la::Vector y_training(subset(fpp3::kConsumption, 0, t_idx));

    la::Matrix x_test = la::Matrix::from_columns(
        {subset(income, t_idx + 1), subset(production, t_idx + 1), subset(savings, t_idx + 1),
         subset(unemployment, t_idx + 1)});
    std::vector<double> y_test = subset(fpp3::kConsumption, t_idx + 1);

    ml::KNearestNeighbors knn(x_training, y_training, 5);
    std::optional<std::vector<double>> knn_predict = knn.predict(x_test);
    CHECK_TRUE(knn_predict.has_value());

    nd::regression::LinearRegression lm(x_training, y_training);
    std::vector<double> lm_predict = lm.predict(x_test);

    double knn_r2 = nd::GoodnessOfFit::r_squared(y_test, *knn_predict);
    double lm_r2 = nd::GoodnessOfFit::r_squared(y_test, lm_predict);

    // kNN is better.
    CHECK_TRUE(knn_r2 > lm_r2);
}

void test_get_neighbors_multi_row() {
    // 2D dataset: 12 points in two well-separated clusters.
    // Cluster A near (0,0): indices 0-5. Cluster B near (100,100): indices 6-11.
    std::vector<double> x1 = {0, 1, 0, 1, 2, 0, 100, 101, 100, 101, 102, 100};
    std::vector<double> x2 = {0, 0, 1, 1, 0, 2, 100, 100, 101, 101, 100, 102};
    std::vector<double> y = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

    ml::KNearestNeighbors knn(la::Matrix::from_columns({x1, x2}), la::Vector(y), 2);

    // Multi-row query: two points near opposite clusters.
    std::optional<std::vector<int>> neighbors =
        knn.get_neighbors(la::Matrix(2, 2, {0.5, 0.5, 100.5, 100.5}));
    CHECK_TRUE(neighbors.has_value());
    CHECK_EQ(static_cast<int>(neighbors->size()), 4);

    // First query (0.5, 0.5): the nearest neighbors are from cluster A (indices 0-5).
    CHECK_TRUE((*neighbors)[0] < 6);
    CHECK_TRUE((*neighbors)[1] < 6);
    // Second query (100.5, 100.5): from cluster B (indices 6-11).
    CHECK_TRUE((*neighbors)[2] >= 6);
    CHECK_TRUE((*neighbors)[3] >= 6);

    // COREHYDRO SUPPLEMENT -- the exact indices, MEASURED against the real library. Both queries
    // sit at the exact center of a symmetric four-point square, so points 0, 1, 2, 3 are all at
    // distance sqrt(0.5) from (0.5, 0.5) and 6, 7, 8, 9 all at sqrt(0.5) from (100.5, 100.5);
    // WHICH two come back is decided entirely by the sort's tie permutation. The C# assertions
    // above are index-RANGE inequalities and would pass on any of the six orderings.
    //
    // Honest limit of THIS case: with only 12 training points the .NET introsort takes its
    // insertion-sort path (partition size <= 16), which is stable, so it does not distinguish the
    // ported introsort from `std::stable_sort` -- only from `std::sort`. The case below it does.
    CHECK_EQ((*neighbors)[0], 0);
    CHECK_EQ((*neighbors)[1], 1);
    CHECK_EQ((*neighbors)[2], 6);
    CHECK_EQ((*neighbors)[3], 7);

    // The whole tied group, same measured order.
    ml::KNearestNeighbors knn4(la::Matrix::from_columns({x1, x2}), la::Vector(y), 4);
    std::optional<std::vector<int>> four = knn4.get_neighbors(la::Matrix(1, 2, {0.5, 0.5}));
    CHECK_TRUE(four.has_value());
    const int expected_four[4] = {0, 1, 2, 3};
    for (int i = 0; i < 4; i++) CHECK_EQ((*four)[static_cast<std::size_t>(i)], expected_four[i]);
}

// COREHYDRO SUPPLEMENT -- the tie permutation ABOVE the introsort's 16-element insertion-sort
// threshold, which is where the .NET sort stops being stable and where a `std::sort` port would
// silently return different neighbors.
//
// 24 training points at x = -3, -2, -1, 1, 2, 3 repeated four times, queried at 0: distances 1,
// 2 and 3 each occur eight times. The full 24-index order below was MEASURED against the real
// library; `std::sort` over the same distances gives a different permutation (verified), so this
// case is the one that actually pins numerics/utilities/dotnet_sort.hpp for kNN.
void test_tie_permutation_above_the_insertion_sort_threshold() {
    std::vector<double> xs;
    for (int rep = 0; rep < 4; rep++)
        for (double v : {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0}) xs.push_back(v);
    std::vector<double> ys;
    for (int i = 0; i < 24; i++) ys.push_back(static_cast<double>(i));

    ml::KNearestNeighbors knn(xs, ys, 24);
    std::optional<std::vector<int>> order = knn.get_neighbors(std::vector<double>{0.0});
    CHECK_TRUE(order.has_value());
    const int expected[24] = {8, 2,  3,  21, 20, 9,  14, 15, 1,  4,  7,  16,
                              10, 22, 19, 13, 0,  18, 11, 12, 6,  5,  17, 23};
    for (int i = 0; i < 24; i++) CHECK_EQ((*order)[static_cast<std::size_t>(i)], expected[i]);

    // The first eight are the k = 8 answer, as they must be.
    ml::KNearestNeighbors knn8(xs, ys, 8);
    std::optional<std::vector<int>> eight = knn8.get_neighbors(std::vector<double>{0.0});
    CHECK_TRUE(eight.has_value());
    for (int i = 0; i < 8; i++) CHECK_EQ((*eight)[static_cast<std::size_t>(i)], expected[i]);
}

// --- COREHYDRO SUPPLEMENT (no C# counterpart) ---------------------------------------------

void test_zero_distance_weight_branch() {
    // Transcription note 3: a query landing exactly on a training row gets `w = 1` for that
    // neighbor instead of 1/d^2. With k = 1 the prediction is therefore that row's response
    // exactly, rather than an infinity-weighted average.
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<double> y = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    ml::KNearestNeighbors knn(x, y, 1);
    std::optional<std::vector<double>> p = knn.predict(std::vector<double>{4.0});
    CHECK_TRUE(p.has_value());
    CHECK_EQ((*p)[0], 40.0);

    // Two neighbors, one at distance 0 and one at distance 1: weights 1 and 1, so the average is
    // the midpoint. (A "cleaner" 1/d^2 with a small epsilon would give something else.)
    ml::KNearestNeighbors knn2(x, y, 2);
    std::optional<std::vector<double>> p2 = knn2.predict(std::vector<double>{4.0});
    CHECK_TRUE(p2.has_value());
    CHECK_NEAR((*p2)[0], 35.0, 1e-12);
}

void test_guards_and_null_returns() {
    CHECK_THROWS_MSG(
        ml::KNearestNeighbors(std::vector<double>(10, 1.0), std::vector<double>(9, 1.0), 3),
        "same length");
    CHECK_THROWS_MSG(
        ml::KNearestNeighbors(std::vector<double>(9, 1.0), std::vector<double>(9, 1.0), 3),
        "at least ten");

    ml::KNearestNeighbors knn(iris_train(), la::Vector(iris::kSpeciesTrain), 5);
    CHECK_EQ(knn.number_of_features(), 4);
    CHECK_EQ(knn.k(), 5);
    CHECK_TRUE(knn.is_regression());

    // predict() rejects a query with the wrong column count by returning the C# null.
    CHECK_TRUE(!knn.predict(std::vector<double>{1.0}).has_value());
    CHECK_TRUE(knn.predict(iris_test()).has_value());

    // get_neighbors() does NOT reject it -- upstream's guard there is a tautology (transcription
    // note 2), and `Tools.Distance` loops over the QUERY row, so a NARROWER query computes a
    // partial-dimension distance and returns neighbors rather than erroring. Mirrored.
    std::optional<std::vector<int>> narrow = knn.get_neighbors(std::vector<double>{5.0});
    CHECK_TRUE(narrow.has_value());
    CHECK_EQ(static_cast<int>(narrow->size()), 5);
    // A WIDER query would read past the end of each training row in C++ (C# throws
    // IndexOutOfRangeException); the port throws instead of reading out of bounds.
    CHECK_THROWS(knn.get_neighbors(la::Matrix(1, 5, {1.0, 2.0, 3.0, 4.0, 5.0})));
}

void test_bootstrap_and_prediction_intervals() {
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<double> y = {10, 21, 29, 41, 50, 61, 69, 81, 90, 101, 109, 121};
    ml::KNearestNeighbors knn(x, y, 3);

    // A seeded bootstrap prediction is reproducible.
    std::optional<std::vector<double>> a = knn.bootstrap_predict(std::vector<double>{5.5}, 123);
    std::optional<std::vector<double>> b = knn.bootstrap_predict(std::vector<double>{5.5}, 123);
    CHECK_TRUE(a.has_value() && b.has_value());
    CHECK_EQ((*a)[0], (*b)[0]);
    CHECK_TRUE(std::isfinite((*a)[0]));

    // The interval table is n-by-4 and ordered.
    la::Matrix pi = knn.prediction_intervals(std::vector<double>{2.5, 8.5}, 456, 100, 0.1);
    CHECK_EQ(pi.number_of_rows(), 2);
    CHECK_EQ(pi.number_of_columns(), 4);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) CHECK_TRUE(std::isfinite(pi(i, j)));
        CHECK_TRUE(pi(i, 0) <= pi(i, 1));
        CHECK_TRUE(pi(i, 1) <= pi(i, 2));
    }
    CHECK_TRUE(pi(0, 1) < pi(1, 1));  // the response is increasing in x

    // Seeded reproducibility of the whole table.
    la::Matrix again = knn.prediction_intervals(std::vector<double>{2.5, 8.5}, 456, 100, 0.1);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) CHECK_EQ(pi(i, j), again(i, j));

    // prediction_intervals has no shape guard of its own (upstream has none either), so a query
    // with the wrong column count reaches the inner predict as a null. C# dereferences it; the
    // port throws instead of invoking undefined behavior.
    ml::KNearestNeighbors wide(la::Matrix::from_columns({x, x}), la::Vector(y), 3);
    CHECK_THROWS_MSG(wide.prediction_intervals(std::vector<double>{2.5}, 1, 5, 0.1),
                     "expected 2");

    // Transcription note 5: prediction_intervals has NO classification branch, so a
    // classification model still gets fractional percentile/mean columns rather than floored
    // labels -- unlike RandomForest::predict.
    ml::KNearestNeighbors clf(la::Matrix::from_columns({x}), la::Vector(std::vector<double>{
                                  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}),
                              3);
    clf.set_is_regression(false);
    la::Matrix cpi = clf.prediction_intervals(std::vector<double>{6.5}, 456, 100, 0.1);
    CHECK_EQ(cpi.number_of_rows(), 1);
    // Every column is still a percentile/mean of the resampled label predictions, so the mean
    // column is free to be fractional.
    CHECK_TRUE(std::isfinite(cpi(0, 3)));
}

}  // namespace

int main() {
    test_knn_iris();
    test_knn_classification();
    test_knn_regression();
    test_get_neighbors_multi_row();
    test_tie_permutation_above_the_insertion_sort_threshold();
    test_zero_distance_weight_branch();
    test_guards_and_null_returns();
    test_bootstrap_and_prediction_intervals();
    return chtest::summary("test_k_nearest_neighbors");
}
