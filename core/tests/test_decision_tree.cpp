// P5 Task 7 -- DecisionTree and DecisionNode.
//
// Transcribes both [TestMethod]s of
// upstream/Numerics/Test_Numerics/Machine Learning/Supervised/Test_DecisionTree.cs @ 2a0357a.
//
// Both C# assertions are INEQUALITIES (classification accuracy at least 90%, and the tree's
// regression R-squared BELOW a linear model's -- upstream's own comment says the second test is
// "mainly meant for demonstration"), so neither pins a value. The COREHYDRO SUPPLEMENT below is
// what actually holds this port to the algorithm: a hand-built problem whose split and leaf values
// are computable by hand, the null-returning guards, and the seeded-determinism contract every
// RandomForest oracle in this phase rests on.
#include <cmath>
#include <optional>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/regression/linear_regression.hpp"
#include "corehydro/numerics/machine_learning/supervised/decision_tree.hpp"
#include "data/fpp3_dataset.hpp"
#include "data/iris_dataset.hpp"

namespace ml = corehydro::numerics::machine_learning;
namespace la = corehydro::numerics::math::linalg;
namespace nd = corehydro::numerics::data;
namespace iris = corehydro::testdata::iris;
namespace fpp3 = corehydro::testdata::fpp3;

namespace {

// C# `Subset(start, end)` is INCLUSIVE of `end`; `Subset(start)` runs to the end of the array.
// (Numerics/Utilities/ExtensionMethods.cs lines 266 and 285 -- checked, not assumed.)
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

// --- Transcribed from Test_DecisionTree.cs ------------------------------------------------

void test_decision_tree_iris() {
    la::Vector y_training(iris::kSpeciesTrain);
    la::Matrix x_training = iris_train();

    ml::DecisionTree tree(x_training, y_training, 12345);
    tree.set_is_regression(false);
    tree.set_features(4);
    tree.train();

    std::optional<std::vector<double>> prediction = tree.predict(iris_test());
    CHECK_TRUE(prediction.has_value());
    double accuracy = nd::GoodnessOfFit::accuracy(iris::kSpeciesTest, *prediction);

    // Accuracy should be greater than or equal to 90%.
    CHECK_TRUE(accuracy >= 90.0);
}

void test_decision_tree_regression() {
    // Create the training data (rows [0, 118], inclusive -- 119 rows).
    const int t_idx = 118;
    la::Matrix x_training = la::Matrix::from_columns(
        {subset(fpp3::kIncome, 0, t_idx), subset(fpp3::kProduction, 0, t_idx),
         subset(fpp3::kSavings, 0, t_idx), subset(fpp3::kUnemployment, 0, t_idx)});
    la::Vector y_training(subset(fpp3::kConsumption, 0, t_idx));

    // Create the test data (rows [119, end] -- 68 rows).
    la::Matrix x_test = la::Matrix::from_columns(
        {subset(fpp3::kIncome, t_idx + 1), subset(fpp3::kProduction, t_idx + 1),
         subset(fpp3::kSavings, t_idx + 1), subset(fpp3::kUnemployment, t_idx + 1)});
    std::vector<double> y_test = subset(fpp3::kConsumption, t_idx + 1);

    ml::DecisionTree tree(x_training, y_training, 12345);
    tree.set_features(4);
    tree.train();
    std::optional<std::vector<double>> tree_predict = tree.predict(x_test);
    CHECK_TRUE(tree_predict.has_value());

    // Create the linear regression.
    nd::regression::LinearRegression lm(x_training, y_training);
    std::vector<double> lm_predict = lm.predict(x_test);

    // Get the R-squared of the predictions.
    double tree_r2 = nd::GoodnessOfFit::r_squared(y_test, *tree_predict);
    double lm_r2 = nd::GoodnessOfFit::r_squared(y_test, lm_predict);

    // Linear regression is better (upstream's point: use a Random Forest for regression).
    CHECK_TRUE(tree_r2 < lm_r2);
}

// --- COREHYDRO SUPPLEMENT (no C# counterpart) ---------------------------------------------

void test_hand_computable_regression_split() {
    // A 12-point 1-D problem with a clean gap at x = 5: y is 10 below the gap and 100 above.
    //
    // The whole tree shape is pinned below because it is a MEASURED C# oracle (probed against the
    // real library at this seed) and because it demonstrates transcription note 2's real
    // consequence: for regression, `numberOfLabels` is the SAMPLE COUNT, not the distinct-value
    // count, so a pure node does not stop -- and `VarianceReduction` on a pure node is 0, which
    // still beats the `double.MinValue` seed, so a split is always found. A default regression
    // tree therefore recurses until every leaf holds ONE observation, memorizing the training
    // data. That is what makes upstream's own regression test expect the tree to lose to a linear
    // model. See docs/upstream-csharp-issues.md.
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105};
    std::vector<double> y = {10, 10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 100};
    ml::DecisionTree tree(x, y, 7);
    tree.train();

    CHECK_TRUE(tree.root() != nullptr);
    CHECK_TRUE(!tree.root()->is_leaf_node);
    CHECK_EQ(tree.root()->feature_index, 0);
    CHECK_EQ(tree.root()->threshold, 6.0);
    // An internal node's `value` stays NaN.
    CHECK_TRUE(std::isnan(tree.root()->value));

    // The measured C# tree, left branch: a right-leaning chain splitting off one point at a
    // time at thresholds 1, 2, 3, 4, 5, ending in two singleton leaves.
    const ml::DecisionNode* n = tree.root()->left.get();
    for (double expected_threshold : {1.0, 2.0, 3.0, 4.0, 5.0}) {
        CHECK_TRUE(n != nullptr && !n->is_leaf_node);
        CHECK_EQ(n->feature_index, 0);
        CHECK_EQ(n->threshold, expected_threshold);
        CHECK_TRUE(n->left != nullptr && n->left->is_leaf_node);
        CHECK_EQ(n->left->value, 10.0);
        n = n->right.get();
    }
    CHECK_TRUE(n != nullptr && n->is_leaf_node);
    CHECK_EQ(n->value, 10.0);

    // The same shape on the right branch, at thresholds 100, 101, 102, 103, 104.
    n = tree.root()->right.get();
    for (double expected_threshold : {100.0, 101.0, 102.0, 103.0, 104.0}) {
        CHECK_TRUE(n != nullptr && !n->is_leaf_node);
        CHECK_EQ(n->threshold, expected_threshold);
        CHECK_TRUE(n->left != nullptr && n->left->is_leaf_node);
        CHECK_EQ(n->left->value, 100.0);
        n = n->right.get();
    }
    CHECK_TRUE(n != nullptr && n->is_leaf_node);
    CHECK_EQ(n->value, 100.0);

    // A leaf keeps the -1 / NaN defaults for the fields it does not use.
    CHECK_EQ(tree.root()->left->left->feature_index, -1);
    CHECK_TRUE(std::isnan(tree.root()->left->left->threshold));

    // Predictions follow the splits: `<= threshold` goes left. C# returns {10, 10, 100, 100}.
    std::optional<std::vector<double>> p =
        tree.predict(std::vector<double>{0.0, 6.0, 6.5, 200.0});
    CHECK_TRUE(p.has_value());
    CHECK_EQ((*p)[0], 10.0);
    CHECK_EQ((*p)[1], 10.0);
    CHECK_EQ((*p)[2], 100.0);
    CHECK_EQ((*p)[3], 100.0);
}

void test_hand_computable_classification_leaf() {
    // Classification over the same clean split: the leaves are pure classes, so the
    // most-common-value leaf rule is unambiguous.
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105};
    std::vector<double> y = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    ml::DecisionTree tree(x, y, 7);
    tree.set_is_regression(false);
    tree.train();
    std::optional<std::vector<double>> p = tree.predict(std::vector<double>{2.0, 104.0});
    CHECK_TRUE(p.has_value());
    CHECK_EQ((*p)[0], 0.0);
    CHECK_EQ((*p)[1], 1.0);
}

void test_guards_and_null_returns() {
    std::vector<double> x10(10, 1.0);
    std::vector<double> y10(10, 2.0);

    // The two constructor guards.
    CHECK_THROWS_MSG(ml::DecisionTree(std::vector<double>(10, 1.0), std::vector<double>(9, 1.0)),
                     "same length");
    CHECK_THROWS_MSG(ml::DecisionTree(std::vector<double>(9, 1.0), std::vector<double>(9, 1.0)),
                     "at least ten");

    // predict() before train() returns the C# null.
    ml::DecisionTree untrained(x10, y10, 3);
    CHECK_TRUE(!untrained.predict(std::vector<double>{1.0}).has_value());

    // A trained tree rejects a matrix with the wrong column count, also by returning null.
    ml::DecisionTree tree(iris_train(), la::Vector(iris::kSpeciesTrain), 3);
    tree.train();
    CHECK_TRUE(!tree.predict(std::vector<double>{1.0, 2.0}).has_value());  // 1 column, needs 4
    CHECK_TRUE(tree.predict(iris_test()).has_value());

    // Defaults, straight from the C# property initializers.
    ml::DecisionTree fresh(x10, y10, 3);
    CHECK_EQ(fresh.minimum_split_size(), 2);
    CHECK_EQ(fresh.max_depth(), 100);
    CHECK_TRUE(fresh.is_regression());
    CHECK_TRUE(!fresh.is_trained());
    CHECK_EQ(fresh.dimensions(), 1);
    // `Features` defaults to max(1, Dimensions - 1), so a single-column problem gets 1.
    CHECK_EQ(fresh.features(), 1);
    ml::DecisionTree wide(iris_train(), la::Vector(iris::kSpeciesTrain), 3);
    CHECK_EQ(wide.features(), 3);
}

void test_seeded_determinism() {
    // Two trees at the same seed give bit-identical predictions -- the contract RandomForest and
    // every seeded ML fixture in this phase depend on.
    ml::DecisionTree a(iris_train(), la::Vector(iris::kSpeciesTrain), 12345);
    ml::DecisionTree b(iris_train(), la::Vector(iris::kSpeciesTrain), 12345);
    a.set_is_regression(false);
    b.set_is_regression(false);
    a.set_features(4);
    b.set_features(4);
    a.train();
    b.train();
    std::optional<std::vector<double>> pa = a.predict(iris_test());
    std::optional<std::vector<double>> pb = b.predict(iris_test());
    CHECK_TRUE(pa.has_value() && pb.has_value());
    for (std::size_t i = 0; i < pa->size(); i++) CHECK_EQ((*pa)[i], (*pb)[i]);

    // A different seed draws different feature subsets. With features = dimensions the subsets
    // are the same SET every time, so the fit is seed-independent here -- assert that rather
    // than a difference that need not exist.
    ml::DecisionTree c(iris_train(), la::Vector(iris::kSpeciesTrain), 999);
    c.set_is_regression(false);
    c.set_features(4);
    c.train();
    std::optional<std::vector<double>> pc = c.predict(iris_test());
    CHECK_TRUE(pc.has_value());
    for (std::size_t i = 0; i < pa->size(); i++) CHECK_EQ((*pa)[i], (*pc)[i]);
}

void test_max_depth_and_minimum_split_size() {
    // A depth cap of 0 makes the root itself a leaf, so every prediction is the training mean.
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105};
    std::vector<double> y = {10, 10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 100};
    ml::DecisionTree capped(x, y, 7);
    capped.set_max_depth(0);
    capped.train();
    CHECK_TRUE(capped.root()->is_leaf_node);
    CHECK_EQ(capped.root()->value, 55.0);  // the mean of six 10s and six 100s

    // A minimum split size above the sample count does the same.
    ml::DecisionTree unsplittable(x, y, 7);
    unsplittable.set_minimum_split_size(13);
    unsplittable.train();
    CHECK_TRUE(unsplittable.root()->is_leaf_node);
    CHECK_EQ(unsplittable.root()->value, 55.0);
}

}  // namespace

int main() {
    test_decision_tree_iris();
    test_decision_tree_regression();
    test_hand_computable_regression_split();
    test_hand_computable_classification_leaf();
    test_guards_and_null_returns();
    test_seeded_determinism();
    test_max_depth_and_minimum_split_size();
    return chtest::summary("test_decision_tree");
}
