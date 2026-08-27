// P5 Task 10 -- NaiveBayes.
//
// Transcribes Test_NaiveBayes_Iris from
// upstream/Numerics/Test_Numerics/Machine Learning/Supervised/Test_NaiveBayes.cs @ 2a0357a --
// the richest oracle in this subsystem: twelve conditional means, twelve conditional standard
// deviations and three priors at 1e-6, plus the exact 60-value prediction vector (which contains
// two deliberate misclassifications, at positions 42 and 53). Expected values are R's `naiveBayes`
// output.
#include <cmath>
#include <optional>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/machine_learning/supervised/naive_bayes.hpp"
#include "data/iris_dataset.hpp"

namespace ml = corehydro::numerics::machine_learning;
namespace la = corehydro::numerics::math::linalg;
namespace iris = corehydro::testdata::iris;

namespace {

la::Matrix iris_train() {
    return la::Matrix::from_columns({iris::kSepalLengthTrain, iris::kSepalWidthTrain,
                                      iris::kPetalLengthTrain, iris::kPetalWidthTrain});
}
la::Matrix iris_test() {
    return la::Matrix::from_columns({iris::kSepalLengthTest, iris::kSepalWidthTest,
                                      iris::kPetalLengthTest, iris::kPetalWidthTest});
}

// --- Transcribed from Test_NaiveBayes.cs --------------------------------------------------

void test_naive_bayes_iris() {
    la::Matrix x_training = iris_train();
    ml::NaiveBayes naive_bayes(x_training, la::Vector(iris::kSpeciesTrain));
    naive_bayes.train();

    // Test the means.
    // Setosa -- class 1.
    CHECK_NEAR(naive_bayes.means()[0][0], 5.016667, 1e-6);
    CHECK_NEAR(naive_bayes.means()[0][1], 3.456667, 1e-6);
    CHECK_NEAR(naive_bayes.means()[0][2], 1.466667, 1e-6);
    CHECK_NEAR(naive_bayes.means()[0][3], 0.220000, 1e-6);
    // Versicolor -- class 2.
    CHECK_NEAR(naive_bayes.means()[1][0], 5.916667, 1e-6);
    CHECK_NEAR(naive_bayes.means()[1][1], 2.750000, 1e-6);
    CHECK_NEAR(naive_bayes.means()[1][2], 4.220000, 1e-6);
    CHECK_NEAR(naive_bayes.means()[1][3], 1.303333, 1e-6);
    // Virginica -- class 3.
    CHECK_NEAR(naive_bayes.means()[2][0], 6.740000, 1e-6);
    CHECK_NEAR(naive_bayes.means()[2][1], 3.033333, 1e-6);
    CHECK_NEAR(naive_bayes.means()[2][2], 5.680000, 1e-6);
    CHECK_NEAR(naive_bayes.means()[2][3], 2.063333, 1e-6);

    // Test the standard deviations.
    // Setosa -- class 1.
    CHECK_NEAR(naive_bayes.standard_deviations()[0][0], 0.3097088, 1e-6);
    CHECK_NEAR(naive_bayes.standard_deviations()[0][1], 0.3490710, 1e-6);
    CHECK_NEAR(naive_bayes.standard_deviations()[0][2], 0.1881550, 1e-6);
    CHECK_NEAR(naive_bayes.standard_deviations()[0][3], 0.08051558, 1e-6);
    // Versicolor -- class 2.
    CHECK_NEAR(naive_bayes.standard_deviations()[1][0], 0.5414434, 1e-6);
    CHECK_NEAR(naive_bayes.standard_deviations()[1][1], 0.3202908, 1e-6);
    CHECK_NEAR(naive_bayes.standard_deviations()[1][2], 0.4373904, 1e-6);
    CHECK_NEAR(naive_bayes.standard_deviations()[1][3], 0.20924055, 1e-6);
    // Virginica -- class 3.
    CHECK_NEAR(naive_bayes.standard_deviations()[2][0], 0.5763141, 1e-6);
    CHECK_NEAR(naive_bayes.standard_deviations()[2][1], 0.2928261, 1e-6);
    CHECK_NEAR(naive_bayes.standard_deviations()[2][2], 0.5019960, 1e-6);
    CHECK_NEAR(naive_bayes.standard_deviations()[2][3], 0.29182344, 1e-6);

    // Test the priors.
    double n = static_cast<double>(x_training.number_of_rows());
    CHECK_NEAR(naive_bayes.priors()[0], 30.0 / n, 1e-6);
    CHECK_NEAR(naive_bayes.priors()[1], 30.0 / n, 1e-6);
    CHECK_NEAR(naive_bayes.priors()[2], 30.0 / n, 1e-6);

    // Make predictions.
    std::optional<std::vector<double>> prediction = naive_bayes.predict(iris_test());
    CHECK_TRUE(prediction.has_value());
    const double true_pred[60] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 3, 3,
                                  3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3};
    CHECK_EQ(static_cast<int>(prediction->size()), 60);
    for (std::size_t i = 0; i < prediction->size(); i++)
        CHECK_EQ((*prediction)[i], true_pred[i]);
}

// --- COREHYDRO SUPPLEMENT (no C# counterpart) ---------------------------------------------

void test_classes_are_in_first_appearance_order() {
    // Transcription note 1: `classes()` follows the training response's order, not sorted order,
    // and every per-class array is indexed by that position. The iris data arrives sorted by
    // species, which hides it; this shuffled response does not.
    std::vector<double> y = {7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5};
    std::vector<double> x = {1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5};
    ml::NaiveBayes nb(x, y);
    CHECK_EQ(static_cast<int>(nb.classes().size()), 3);
    CHECK_EQ(nb.classes()[0], 7.0);
    CHECK_EQ(nb.classes()[1], 3.0);
    CHECK_EQ(nb.classes()[2], 5.0);

    nb.train();
    // The means follow the same order, so means()[0] belongs to class 7, not to class 3.
    CHECK_NEAR(nb.means()[0][0], 1.0, 1e-12);
    CHECK_NEAR(nb.means()[1][0], 9.0, 1e-12);
    CHECK_NEAR(nb.means()[2][0], 5.0, 1e-12);
    for (std::size_t i = 0; i < 3; i++) CHECK_NEAR(nb.priors()[i], 5.0 / 15.0, 1e-12);

    // Prediction returns the class LABEL, not its index.
    std::optional<std::vector<double>> p = nb.predict(std::vector<double>{1.0, 9.0, 5.0});
    CHECK_TRUE(p.has_value());
    CHECK_EQ((*p)[0], 7.0);
    CHECK_EQ((*p)[1], 3.0);
    CHECK_EQ((*p)[2], 5.0);
}

void test_singleton_class_gets_the_floor_standard_deviation() {
    // Transcription note 3: a class with one member gets sd = 1e-6 rather than 0/0, so the Normal
    // constructor accepts it and the class stays predictable.
    std::vector<double> y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 100};
    ml::NaiveBayes nb(x, y);
    nb.train();
    CHECK_EQ(nb.standard_deviations()[1][0], 1e-6);
    CHECK_NEAR(nb.means()[1][0], 100.0, 1e-12);
    CHECK_NEAR(nb.priors()[1], 0.1, 1e-12);

    // The singleton class is still reachable, at its own point.
    std::optional<std::vector<double>> p = nb.predict(std::vector<double>{100.0, 5.0});
    CHECK_TRUE(p.has_value());
    CHECK_EQ((*p)[0], 1.0);
    CHECK_EQ((*p)[1], 0.0);
}

void test_standard_deviation_uses_the_two_moment_form() {
    // Transcription note 2: the per-class sd is the naive `sqrt((u2 - u1^2) * n/(n-1))`, which
    // agrees with the stable sample standard deviation to rounding on well-scaled data. Pinned
    // as an identity here (there is no separate C# literal for a hand-made class).
    std::vector<double> y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    std::vector<double> x = {2, 4, 4, 4, 5, 20, 22, 22, 22, 23};
    ml::NaiveBayes nb(x, y);
    nb.train();
    // mean 3.8; sum sq dev = 3.24+0.04*3+1.44 = 4.8; sample variance 1.2; sd sqrt(1.2).
    CHECK_NEAR(nb.means()[0][0], 3.8, 1e-12);
    CHECK_NEAR(nb.standard_deviations()[0][0], std::sqrt(1.2), 1e-10);
    CHECK_NEAR(nb.means()[1][0], 21.8, 1e-12);
    CHECK_NEAR(nb.standard_deviations()[1][0], std::sqrt(1.2), 1e-9);

    // The `Math.Max(0, ...)` guard: a zero-variance class cancels to (possibly negative) noise,
    // and the guard keeps the result real rather than NaN. A constant class gives sd exactly 0.
    std::vector<double> yc = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    std::vector<double> xc = {7, 7, 7, 7, 7, 9, 9, 9, 9, 9};
    ml::NaiveBayes flat(xc, yc);
    flat.train();
    CHECK_EQ(flat.standard_deviations()[0][0], 0.0);
    CHECK_EQ(flat.standard_deviations()[1][0], 0.0);
    CHECK_TRUE(!std::isnan(flat.standard_deviations()[0][0]));

    // A degenerate (sd = 0) class still PREDICTS rather than erroring: the log-density is +inf at
    // the class mean and -inf everywhere else, so the argmax picks the matching class and, at a
    // point matching neither, falls to the FIRST class (transcription note 4's tie rule).
    // MEASURED against the real library, which returns exactly these three values from exactly
    // these sd = 0 parameters.
    std::optional<std::vector<double>> p = flat.predict(std::vector<double>{7.0, 9.0, 8.0});
    CHECK_TRUE(p.has_value());
    CHECK_EQ((*p)[0], 0.0);
    CHECK_EQ((*p)[1], 1.0);
    CHECK_EQ((*p)[2], 0.0);
}

void test_guards_and_null_returns() {
    CHECK_THROWS_MSG(ml::NaiveBayes(std::vector<double>(10, 1.0), std::vector<double>(9, 1.0)),
                     "same length");
    CHECK_THROWS_MSG(ml::NaiveBayes(std::vector<double>(9, 1.0), std::vector<double>(9, 1.0)),
                     "at least ten");

    ml::NaiveBayes untrained(iris_train(), la::Vector(iris::kSpeciesTrain));
    CHECK_TRUE(!untrained.is_trained());
    CHECK_TRUE(!untrained.predict(iris_test()).has_value());

    untrained.train();
    CHECK_TRUE(untrained.is_trained());
    CHECK_TRUE(untrained.predict(iris_test()).has_value());
    // The wrong column count returns the C# null.
    CHECK_TRUE(!untrained.predict(std::vector<double>{1.0}).has_value());
}

}  // namespace

int main() {
    test_naive_bayes_iris();
    test_classes_are_in_first_appearance_order();
    test_singleton_class_gets_the_floor_standard_deviation();
    test_standard_deviation_uses_the_two_moment_form();
    test_guards_and_null_returns();
    return chtest::summary("test_naive_bayes");
}
