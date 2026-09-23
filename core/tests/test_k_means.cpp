// P5 Task 3 -- KMeans.
//
// Transcribes Test_KMeans_Iris from
// upstream/Numerics/Test_Numerics/Machine Learning/Unsupervised/Test_KMeans.cs @ 7e8e8d1.
// Upstream's expected cluster means are R's `kmeans` output on the iris measurements, asserted at
// 1e-6 -- so reproducing them at that tolerance from a seeded run pins the whole k-means++
// initialization stream, not just the arithmetic.
//
// Everything after the transcribed method is a COREHYDRO SUPPLEMENT, clearly marked.
#include <cmath>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/machine_learning/unsupervised/k_means.hpp"
#include "data/iris_dataset.hpp"

namespace ml = corehydro::numerics::machine_learning;
namespace la = corehydro::numerics::math::linalg;
namespace iris = corehydro::testdata::iris;

namespace {

la::Matrix iris_features() {
    return la::Matrix::from_columns(
        {iris::kSepalLength, iris::kSepalWidth, iris::kPetalLength, iris::kPetalWidth});
}

int count_label(const std::vector<int>& labels, int value) {
    int n = 0;
    for (int l : labels)
        if (l == value) n++;
    return n;
}

// --- Transcribed from Test_KMeans.cs ------------------------------------------------------

void test_kmeans_iris() {
    ml::KMeans k_means(iris_features(), 3);
    k_means.train(12345);

    // Test cluster counts.
    CHECK_EQ(count_label(k_means.labels(), 0), 62);
    CHECK_EQ(count_label(k_means.labels(), 1), 38);
    CHECK_EQ(count_label(k_means.labels(), 2), 50);

    // Test cluster means.
    const double true_mean1[] = {5.901613, 2.748387, 4.393548, 1.433871};
    const double true_mean2[] = {6.850000, 3.073684, 5.742105, 2.071053};
    const double true_mean3[] = {5.006000, 3.428000, 1.462000, 0.246000};
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(k_means.means()[0][static_cast<std::size_t>(i)], true_mean1[i], 1e-6);
        CHECK_NEAR(k_means.means()[1][static_cast<std::size_t>(i)], true_mean2[i], 1e-6);
        CHECK_NEAR(k_means.means()[2][static_cast<std::size_t>(i)], true_mean3[i], 1e-6);
    }
}

// --- COREHYDRO SUPPLEMENT (no C# counterpart) ---------------------------------------------

void test_seeded_determinism() {
    // A seeded run is reproducible -- the contract every ML fixture in this phase relies on.
    ml::KMeans a(iris_features(), 3);
    ml::KMeans b(iris_features(), 3);
    a.train(12345);
    b.train(12345);
    CHECK_EQ(a.iterations(), b.iterations());
    for (std::size_t i = 0; i < a.labels().size(); i++) CHECK_EQ(a.labels()[i], b.labels()[i]);
    for (std::size_t k = 0; k < a.means().size(); k++)
        for (std::size_t j = 0; j < a.means()[k].size(); j++)
            CHECK_EQ(a.means()[k][j], b.means()[k][j]);

    // The fit converged well inside the iteration cap.
    CHECK_TRUE(a.iterations() >= 2);
    CHECK_TRUE(a.iterations() <= a.max_iterations());
}

void test_random_initialization_branch() {
    // `kMeansPlusPlus = false` -- the uniform-initialization branch, which no C# test reaches.
    ml::KMeans k_means(iris_features(), 3);
    k_means.train(12345, false);
    CHECK_EQ(static_cast<int>(k_means.labels().size()), 150);
    for (int l : k_means.labels()) CHECK_TRUE(l >= 0 && l < 3);
    // Every cluster is non-empty and the means are finite.
    for (int c = 0; c < 3; c++) {
        CHECK_TRUE(count_label(k_means.labels(), c) > 0);
        for (double m : k_means.means()[static_cast<std::size_t>(c)]) CHECK_TRUE(std::isfinite(m));
    }
}

void test_single_column_and_shape() {
    // The single-column constructor gives an n-by-1 problem.
    ml::KMeans k_means(iris::kPetalLength, 2);
    CHECK_EQ(k_means.dimension(), 1);
    CHECK_EQ(k_means.k(), 2);
    CHECK_EQ(k_means.x().number_of_rows(), 150);
    k_means.train(999);
    CHECK_EQ(static_cast<int>(k_means.means().size()), 2);
    CHECK_EQ(static_cast<int>(k_means.means()[0].size()), 1);
    // Petal length is strongly bimodal (setosa against the rest), so two clusters separate it.
    double lo = std::min(k_means.means()[0][0], k_means.means()[1][0]);
    double hi = std::max(k_means.means()[0][0], k_means.means()[1][0]);
    CHECK_TRUE(lo < 2.5);
    CHECK_TRUE(hi > 4.0);
}

void test_means_are_the_cluster_averages_of_the_previous_step() {
    // Transcription note 2: `train` breaks BEFORE the M-step, so the reported means are one
    // M-step behind the reported labels. On a converged fit that means recomputing the centroids
    // from the final labels reproduces the reported means exactly -- the E-step that ended the
    // loop assigned against them and changed nothing.
    ml::KMeans k_means(iris_features(), 3);
    k_means.train(12345);
    la::Matrix x = iris_features();

    std::vector<std::vector<double>> recomputed(3, std::vector<double>(4, 0.0));
    std::vector<double> counts(3, 0.0);
    for (int i = 0; i < x.number_of_rows(); i++) {
        std::size_t l = static_cast<std::size_t>(k_means.labels()[static_cast<std::size_t>(i)]);
        counts[l]++;
        for (int j = 0; j < 4; j++) recomputed[l][static_cast<std::size_t>(j)] += x(i, j);
    }
    for (std::size_t c = 0; c < 3; c++)
        for (std::size_t j = 0; j < 4; j++) recomputed[c][j] /= counts[c];

    for (std::size_t c = 0; c < 3; c++)
        for (std::size_t j = 0; j < 4; j++)
            CHECK_NEAR(k_means.means()[c][j], recomputed[c][j], 1e-12);
}

void test_k_equals_one_reports_the_sample_mean() {
    std::vector<double> x = {1.0, 2.0, 3.0, 10.0, 11.0, 12.0};
    ml::KMeans one(x, 1);
    one.train(7);
    CHECK_EQ(one.means()[0][0], 6.5);
    CHECK_EQ(one.iterations(), 1);
    for (int l : one.labels()) CHECK_EQ(l, 0);

    // Two well-separated groups converge properly. C# at the same seed reports the clusters in
    // this order (means[0] = 11, means[1] = 2) and Iterations = 2; pinned exactly.
    ml::KMeans two(x, 2);
    two.train(7);
    CHECK_NEAR(two.means()[0][0], 11.0, 1e-12);
    CHECK_NEAR(two.means()[1][0], 2.0, 1e-12);
    CHECK_EQ(two.iterations(), 2);
}

void test_empty_cluster_relocation_and_k_validation() {
    std::vector<double> flat(24, 0.0);
    for (int i = 6; i < 12; ++i) {
        flat[static_cast<std::size_t>(2 * i)] = 10.0;
        flat[static_cast<std::size_t>(2 * i + 1)] = 10.0;
    }
    ml::KMeans relocated(la::Matrix(12, 2, flat), 3);
    relocated.train(12345);
    double inertia = 0.0;
    for (int i = 0; i < 12; ++i) {
        int label = relocated.labels()[static_cast<std::size_t>(i)];
        for (int j = 0; j < 2; ++j) {
            double d = flat[static_cast<std::size_t>(2 * i + j)] -
                       relocated.means()[static_cast<std::size_t>(label)][static_cast<std::size_t>(j)];
            CHECK_TRUE(std::isfinite(relocated.means()[static_cast<std::size_t>(label)]
                                                       [static_cast<std::size_t>(j)]));
            inertia += d * d;
        }
    }
    CHECK_EQ(inertia, 0.0);
    CHECK_THROWS_MSG(ml::KMeans(la::Matrix(12, 2), 0), "between 1");
    CHECK_THROWS_MSG(ml::KMeans(la::Matrix(12, 2), 13), "between 1");
}

}  // namespace

int main() {
    test_kmeans_iris();
    test_seeded_determinism();
    test_random_initialization_branch();
    test_single_column_and_shape();
    test_means_are_the_cluster_averages_of_the_previous_step();
    test_k_equals_one_reports_the_sample_mean();
    test_empty_cluster_relocation_and_k_validation();
    return chtest::summary("test_k_means");
}
