// P5 Task 7 -- DecisionTree and DecisionNode.
//
// Transcribes both [TestMethod]s of
// upstream/Numerics/Test_Numerics/Machine Learning/Supervised/Test_DecisionTree.cs @ 7e8e8d1.
//
// Both C# assertions are INEQUALITIES (classification accuracy at least 90%, and the tree's
// regression R-squared BELOW a linear model's -- upstream's own comment says the second test is
// "mainly meant for demonstration"), so neither pins a value. The COREHYDRO SUPPLEMENT below is
// what actually holds this port to the algorithm: a hand-built problem whose split and leaf values
// are computable by hand, the null-returning guards, and the seeded-determinism contract every
// RandomForest oracle in this phase rests on.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
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

std::uint64_t fnv1a(const std::string& value) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t csharp_bits(double value) {
    if (std::isnan(value)) return 0xFFF8000000000000ULL;
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void serialize_tree(const ml::DecisionNode* node, std::ostringstream& out) {
    if (node == nullptr) return;
    out << node->feature_index << ';' << std::uppercase << std::hex << std::setw(16)
        << std::setfill('0') << csharp_bits(node->threshold) << ';' << std::setw(16)
        << csharp_bits(node->value) << std::dec << ';' << (node->is_leaf_node ? '1' : '0') << '|';
    serialize_tree(node->left.get(), out);
    serialize_tree(node->right.get(), out);
}

std::uint64_t tree_golden_hash(const ml::DecisionTree& tree) {
    std::ostringstream out;
    serialize_tree(tree.root().get(), out);
    return fnv1a(out.str());
}

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
    CHECK_EQ(tree_golden_hash(tree), 0x944188AA31E203A1ULL);
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
    CHECK_EQ(tree_golden_hash(tree), 0x3D2E4098296867CBULL);
}

// --- COREHYDRO SUPPLEMENT (no C# counterpart) ---------------------------------------------

void test_hand_computable_regression_split() {
    // A 12-point 1-D problem with a clean gap at x = 5: y is 10 below the gap and 100 above.
    //
    // v2.2 counts distinct responses when testing node purity. The clean root split therefore
    // produces two pure leaves instead of chains of zero-gain splits.
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

    CHECK_TRUE(tree.root()->left != nullptr && tree.root()->left->is_leaf_node);
    CHECK_TRUE(tree.root()->right != nullptr && tree.root()->right->is_leaf_node);
    CHECK_EQ(tree.root()->left->value, 10.0);
    CHECK_EQ(tree.root()->right->value, 100.0);

    // A leaf keeps the -1 / NaN defaults for the fields it does not use.
    CHECK_EQ(tree.root()->left->feature_index, -1);
    CHECK_TRUE(std::isnan(tree.root()->left->threshold));

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

void test_exact_split_oracles() {
    const std::vector<double> x0 = {
        4.744781, 6.255026, 1.160139, 8.741791, 0.319224, 1.201623, 7.703759, 8.249999,
        7.763458, 2.561846, 7.844684, 9.530505, 5.237435, 7.938164, 7.634402, 3.538497,
        3.908757, 0.798954, 4.233405, 4.819062, 4.704432, 8.131584, 7.138561, 5.718968,
        0.230697, 8.428248, 2.642866, 0.617403, 9.190608, 5.605545, 9.120073, 5.506271,
        1.210414, 7.754063, 6.823279, 8.004035, 7.097406, 5.353946, 3.109105, 1.01253};
    const std::vector<double> x1 = {
        6.041412, 1.540177, 0.705583, 2.134676, 3.987951, 8.706857, 4.997143, 7.688693,
        7.269118, 9.894784, 2.850013, 2.908263, 4.640485, 9.239211, 2.437277, 9.911633,
        5.759533, 1.255405, 2.324209, 0.226532, 2.977218, 1.713005, 4.749645, 2.982202,
        8.679028, 2.531092, 8.802673, 1.569402, 4.644899, 2.859959, 8.467362, 0.291196,
        5.153139, 1.003399, 3.90648, 6.822431, 9.421524, 6.81047, 2.76754, 4.48085};
    const std::vector<double> x2 = {
        8.287756, 2.053361, 1.083714, 0.926085, 2.324783, 6.528751, 1.224496, 0.580007,
        5.802834, 7.568654, 5.376091, 3.487552, 8.249803, 4.264968, 8.381896, 1.098973,
        6.334179, 6.910272, 7.131958, 8.413748, 7.770878, 5.584611, 1.671792, 9.032317,
        4.131488, 1.241772, 5.149096, 3.04752, 6.142418, 9.40855, 4.380326, 6.494782,
        3.475771, 7.189178, 5.217128, 8.68896, 2.452654, 1.702436, 4.397793, 0.220322};
    const std::vector<double> y = {
        9.203648, 6.973726, 5.396438, 7.922964, 5.253539, 8.372256, 6.887424, 10.483275,
        10.398861, 8.988918, 7.161026, 7.715756, 6.251278, 10.311859, 7.417448, 9.087316,
        9.140687, 5.769176, 6.332959, 6.354408, 6.481743, 7.217564, 7.019442, 6.991094,
        7.940477, 7.538989, 8.986207, 5.176143, 7.738771, 6.852212, 11.024926, 6.459788,
        8.495679, 7.49478, 7.280696, 10.587507, 10.360391, 9.330927, 5.997098, 5.517587};
    ml::DecisionTree exact(la::Matrix::from_columns({x0, x1, x2}), la::Vector(y), 12345);
    exact.set_features(3);
    exact.train();
    CHECK_EQ(exact.root()->feature_index, 1);
    CHECK_EQ(exact.root()->threshold, 4.997143);

    std::vector<double> tied_x;
    for (int i = 0; i < 6; ++i) {
        tied_x.push_back(0.0);
        tied_x.push_back(1.0);
        tied_x.push_back(2.0);
    }
    const std::vector<double> tied_y = {
        999902848968.3931, 1000000117649.4011, 1000097151031.6069,
        999896129867.9054, 999995338077.3125, 1000103870132.0946,
        999901810715.8024, 999996663357.6908, 1000098189284.1976,
        999898807264.0746, 1000003336642.3092, 1000101192735.9254,
        999896810273.0493, 1000004661922.6875, 1000103189726.9507,
        999901331037.1741, 999999882350.5989, 1000098668962.8259};
    ml::DecisionTree tied(tied_x, tied_y, 12345);
    tied.set_features(1);
    tied.train();
    CHECK_EQ(tied.root()->threshold, 0.0);
}

void test_iris_golden_structure() {
    ml::DecisionTree tree(iris_train(), la::Vector(iris::kSpeciesTrain), 12345);
    tree.set_is_regression(false);
    tree.set_features(4);
    tree.train();

    auto split = [](const std::shared_ptr<ml::DecisionNode>& node, int feature,
                    double threshold) {
        CHECK_TRUE(node != nullptr && !node->is_leaf_node);
        CHECK_EQ(node->feature_index, feature);
        CHECK_EQ(node->threshold, threshold);
    };
    auto leaf = [](const std::shared_ptr<ml::DecisionNode>& node, double value) {
        CHECK_TRUE(node != nullptr && node->is_leaf_node);
        CHECK_EQ(node->value, value);
    };

    const auto& root = tree.root();
    split(root, 2, 4.4);
    split(root->left, 2, 1.9);
    leaf(root->left->left, 1.0);
    leaf(root->left->right, 2.0);
    split(root->right, 2, 5.0);
    split(root->right->left, 2, 4.8);
    leaf(root->right->left->left, 2.0);
    split(root->right->left->right, 0, 6.1);
    leaf(root->right->left->right->left, 3.0);
    leaf(root->right->left->right->right, 2.0);
    leaf(root->right->right, 3.0);
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
    test_exact_split_oracles();
    test_iris_golden_structure();
    return chtest::summary("test_decision_tree");
}
