// P5 Task 8 -- RandomForest.
//
// Transcribes both [TestMethod]s of
// upstream/Numerics/Test_Numerics/Machine Learning/Supervised/Test_RandomForest.cs @ 2a0357a, at
// the C# default of 1,000 trees.
//
// WALL TIME (measured on this machine before transcribing, to settle whether the C# default was
// affordable serially -- the C# test hides the cost behind Parallel.For): one classification tree
// on the 90-row iris split is 9.8 ms in this suite's unoptimized debug build and 2.3 ms at -O2;
// one regression tree on the 119-row fpp3 split is 15.6 ms and 1.8 ms. So 1,000 trees is about
// 10 s and 16 s respectively in the debug build. That is affordable, so the C# defaults are kept
// and nothing here deviates from upstream's parameters.
//
// Like Test_DecisionTree.cs, both C# assertions are INEQUALITIES, so the COREHYDRO SUPPLEMENT
// below is what holds this port to the algorithm: the interval-column ordering, the
// classification flooring, the untrained/wrong-shape null returns, and seeded determinism.
#include <cmath>
#include <optional>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/regression/linear_regression.hpp"
#include "corehydro/numerics/machine_learning/supervised/random_forest.hpp"
#include "data/fpp3_dataset.hpp"
#include "data/iris_dataset.hpp"

namespace ml = corehydro::numerics::machine_learning;
namespace la = corehydro::numerics::math::linalg;
namespace nd = corehydro::numerics::data;
namespace iris = corehydro::testdata::iris;
namespace fpp3 = corehydro::testdata::fpp3;

namespace {

// C# `Subset(start, end)` is INCLUSIVE of `end`; `Subset(start)` runs to the end of the array.
std::vector<double> subset(const std::vector<double>& v, int start, int end) {
    return std::vector<double>(v.begin() + start, v.begin() + end + 1);
}
std::vector<double> subset(const std::vector<double>& v, int start) {
    return std::vector<double>(v.begin() + start, v.end());
}

std::vector<double> matrix_column(const la::Matrix& m, int j) { return m.column(j); }

la::Matrix iris_train() {
    return la::Matrix::from_columns({iris::kSepalLengthTrain, iris::kSepalWidthTrain,
                                      iris::kPetalLengthTrain, iris::kPetalWidthTrain});
}
la::Matrix iris_test() {
    return la::Matrix::from_columns({iris::kSepalLengthTest, iris::kSepalWidthTest,
                                      iris::kPetalLengthTest, iris::kPetalWidthTest});
}

// --- Transcribed from Test_RandomForest.cs ------------------------------------------------

void test_random_forest_iris() {
    la::Vector y_training(iris::kSpeciesTrain);

    ml::RandomForest rf(iris_train(), y_training, 12345);
    rf.set_is_regression(false);
    rf.set_features(4);
    rf.train();

    std::optional<la::Matrix> prediction = rf.predict(iris_test());
    CHECK_TRUE(prediction.has_value());
    // Column 1 is the MEDIAN, which is what the C# test scores (not the mean).
    double accuracy = nd::GoodnessOfFit::accuracy(iris::kSpeciesTest, matrix_column(*prediction, 1));

    // Accuracy should be greater than or equal to 90%.
    CHECK_TRUE(accuracy >= 90.0);
}

void test_random_forest_regression() {
    const int t_idx = 118;
    la::Matrix x_training = la::Matrix::from_columns(
        {subset(fpp3::kIncome, 0, t_idx), subset(fpp3::kProduction, 0, t_idx),
         subset(fpp3::kSavings, 0, t_idx), subset(fpp3::kUnemployment, 0, t_idx)});
    la::Vector y_training(subset(fpp3::kConsumption, 0, t_idx));

    la::Matrix x_test = la::Matrix::from_columns(
        {subset(fpp3::kIncome, t_idx + 1), subset(fpp3::kProduction, t_idx + 1),
         subset(fpp3::kSavings, t_idx + 1), subset(fpp3::kUnemployment, t_idx + 1)});
    std::vector<double> y_test = subset(fpp3::kConsumption, t_idx + 1);

    ml::RandomForest rf(x_training, y_training, 12345);
    rf.set_features(4);
    rf.train();
    std::optional<la::Matrix> rf_predict = rf.predict(x_test);
    CHECK_TRUE(rf_predict.has_value());

    nd::regression::LinearRegression lm(x_training, y_training);
    std::vector<double> lm_predict = lm.predict(x_test);

    // Column 3 is the MEAN, which is what the C# test scores.
    double rf_r2 = nd::GoodnessOfFit::r_squared(y_test, matrix_column(*rf_predict, 3));
    double lm_r2 = nd::GoodnessOfFit::r_squared(y_test, lm_predict);

    // The random forest is better (the point upstream's DecisionTree regression test sets up).
    CHECK_TRUE(rf_r2 > lm_r2);
}

// --- COREHYDRO SUPPLEMENT (no C# counterpart) ---------------------------------------------

void test_prediction_interval_shape_and_ordering() {
    // A small forest on a clean two-group problem: every column is finite and ordered, and the
    // interval brackets the truth.
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105};
    std::vector<double> y = {10, 11, 10, 11, 10, 11, 100, 101, 100, 101, 100, 101};
    ml::RandomForest rf(x, y, 42);
    rf.set_number_of_trees(50);
    rf.train();

    CHECK_EQ(static_cast<int>(rf.decision_trees().size()), 50);
    std::optional<la::Matrix> p = rf.predict(std::vector<double>{2.0, 103.0});
    CHECK_TRUE(p.has_value());
    CHECK_EQ(p->number_of_rows(), 2);
    CHECK_EQ(p->number_of_columns(), 4);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) CHECK_TRUE(std::isfinite((*p)(i, j)));
        CHECK_TRUE((*p)(i, 0) <= (*p)(i, 1));  // lower <= median
        CHECK_TRUE((*p)(i, 1) <= (*p)(i, 2));  // median <= upper
        CHECK_TRUE((*p)(i, 3) >= (*p)(i, 0) && (*p)(i, 3) <= (*p)(i, 2));  // mean inside
    }
    // Row 0 sits in the low group, row 1 in the high group.
    CHECK_TRUE((*p)(0, 2) < 50.0);
    CHECK_TRUE((*p)(1, 0) > 50.0);

    // A wider alpha gives a narrower interval.
    std::optional<la::Matrix> wide = rf.predict(std::vector<double>{2.0, 103.0}, 0.5);
    CHECK_TRUE(wide.has_value());
    for (int i = 0; i < 2; i++)
        CHECK_TRUE((*wide)(i, 2) - (*wide)(i, 0) <= (*p)(i, 2) - (*p)(i, 0));
    // The median and mean columns do not depend on alpha.
    for (int i = 0; i < 2; i++) {
        CHECK_EQ((*wide)(i, 1), (*p)(i, 1));
        CHECK_EQ((*wide)(i, 3), (*p)(i, 3));
    }
}

void test_classification_floors_every_column() {
    // Transcription note 3: the classification branch floors all four columns, including the
    // mean -- so a forest that is split between class 1 and class 2 reports 1, not 1.5.
    ml::RandomForest rf(iris_train(), la::Vector(iris::kSpeciesTrain), 12345);
    rf.set_is_regression(false);
    rf.set_features(4);
    rf.set_number_of_trees(25);
    rf.train();
    std::optional<la::Matrix> p = rf.predict(iris_test());
    CHECK_TRUE(p.has_value());
    for (int i = 0; i < p->number_of_rows(); i++)
        for (int j = 0; j < 4; j++) {
            CHECK_EQ((*p)(i, j), std::floor((*p)(i, j)));
            CHECK_TRUE((*p)(i, j) >= 1.0 && (*p)(i, j) <= 3.0);
        }

    // The regression branch does NOT floor, on the same data.
    ml::RandomForest reg(iris_train(), la::Vector(iris::kSpeciesTrain), 12345);
    reg.set_features(4);
    reg.set_number_of_trees(25);
    reg.train();
    std::optional<la::Matrix> q = reg.predict(iris_test());
    CHECK_TRUE(q.has_value());
    bool any_fractional = false;
    for (int i = 0; i < q->number_of_rows(); i++)
        if ((*q)(i, 3) != std::floor((*q)(i, 3))) any_fractional = true;
    CHECK_TRUE(any_fractional);
}

void test_guards_and_null_returns() {
    std::vector<double> x10(10, 1.0);
    std::vector<double> y10(10, 2.0);

    CHECK_THROWS_MSG(ml::RandomForest(std::vector<double>(10, 1.0), std::vector<double>(9, 1.0)),
                     "same length");
    CHECK_THROWS_MSG(ml::RandomForest(std::vector<double>(9, 1.0), std::vector<double>(9, 1.0)),
                     "at least ten");

    ml::RandomForest untrained(x10, y10, 3);
    CHECK_TRUE(!untrained.predict(std::vector<double>{1.0}).has_value());
    CHECK_TRUE(!untrained.is_trained());

    // Defaults, straight from the C# property initializers.
    CHECK_EQ(untrained.number_of_trees(), 1000);
    CHECK_EQ(untrained.minimum_split_size(), 2);
    CHECK_EQ(untrained.max_depth(), 100);
    CHECK_TRUE(untrained.is_regression());
    CHECK_EQ(untrained.dimensions(), 1);
    CHECK_EQ(untrained.features(), 1);  // max(1, dimensions - 1)

    // A trained forest returns the empty optional for a matrix with the wrong column count.
    // Upstream has NO such guard here (unlike DecisionTree::Predict) and would dereference the
    // null each tree returns -- see transcription note 4b in random_forest.hpp.
    ml::RandomForest rf(iris_train(), la::Vector(iris::kSpeciesTrain), 3);
    rf.set_number_of_trees(5);
    rf.train();
    CHECK_TRUE(!rf.predict(std::vector<double>{1.0, 2.0}).has_value());
    CHECK_TRUE(rf.predict(iris_test()).has_value());
}

void test_seeded_determinism_and_tree_independence() {
    // Two forests at the same seed agree bit for bit, including the mean column.
    ml::RandomForest a(iris_train(), la::Vector(iris::kSpeciesTrain), 777);
    ml::RandomForest b(iris_train(), la::Vector(iris::kSpeciesTrain), 777);
    a.set_number_of_trees(20);
    b.set_number_of_trees(20);
    a.train();
    b.train();
    std::optional<la::Matrix> pa = a.predict(iris_test());
    std::optional<la::Matrix> pb = b.predict(iris_test());
    CHECK_TRUE(pa.has_value() && pb.has_value());
    for (int i = 0; i < pa->number_of_rows(); i++)
        for (int j = 0; j < 4; j++) CHECK_EQ((*pa)(i, j), (*pb)(i, j));

    // Transcription note 1: the seeds are drawn up front, so growing the forest one tree at a
    // time in a different ORDER would give the same trees. Check the consequence that is
    // observable from outside: each tree is a genuinely different bootstrap, so at least two
    // trees disagree somewhere on the test set.
    bool any_disagreement = false;
    std::optional<std::vector<double>> first = a.decision_trees()[0].predict(iris_test());
    for (std::size_t t = 1; t < a.decision_trees().size() && !any_disagreement; t++) {
        std::optional<std::vector<double>> other = a.decision_trees()[t].predict(iris_test());
        for (std::size_t i = 0; i < first->size(); i++)
            if ((*first)[i] != (*other)[i]) any_disagreement = true;
    }
    CHECK_TRUE(any_disagreement);
}

}  // namespace

int main() {
    test_random_forest_iris();
    test_random_forest_regression();
    test_prediction_interval_shape_and_ordering();
    test_classification_floors_every_column();
    test_guards_and_null_returns();
    test_seeded_determinism_and_tree_independence();
    return chtest::summary("test_random_forest");
}
