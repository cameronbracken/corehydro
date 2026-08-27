// P5 Task 4 -- GaussianMixtureModel.
//
// Transcribes Test_GMM_Iris from
// upstream/Numerics/Test_Numerics/Machine Learning/Unsupervised/Test_GMM.cs @ 2a0357a. Upstream's
// expected weights and means are R `mclust` output, asserted at 1e-2 -- a loose tolerance because
// it compares two DIFFERENT EM implementations, not because the fit is imprecise.
//
// That looseness is why the COREHYDRO SUPPLEMENT below matters more here than in the other ML
// suites: at 1e-2 against a third-party package, the C# assertions alone would pass on a port with
// a broken log-sum-exp, a mislabeled component, or a covariance that never updated. The supplement
// adds the exact identities that are available without a C# literal -- the mixture-weight simplex,
// the responsibility rows, the closed-form single-component fit, and the seeded-determinism
// contract every ML fixture in this phase relies on.
#include <cmath>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/machine_learning/unsupervised/gaussian_mixture_model.hpp"
#include "data/iris_dataset.hpp"

namespace ml = corehydro::numerics::machine_learning;
namespace la = corehydro::numerics::math::linalg;
namespace nd = corehydro::numerics::data;
namespace iris = corehydro::testdata::iris;

namespace {

la::Matrix iris_features() {
    return la::Matrix::from_columns(
        {iris::kSepalLength, iris::kSepalWidth, iris::kPetalLength, iris::kPetalWidth});
}

// --- Transcribed from Test_GMM.cs ---------------------------------------------------------

void test_gmm_iris() {
    ml::GaussianMixtureModel gmm(iris_features(), 3);
    gmm.train(12345);

    // Test cluster weights.
    CHECK_NEAR(gmm.weights()[0], 0.3005423, 1e-2);
    CHECK_NEAR(gmm.weights()[1], 0.3661243, 1e-2);
    CHECK_NEAR(gmm.weights()[2], 0.3333333, 1e-2);

    // Test cluster means.
    const double true_mean1[] = {5.915044, 2.777451, 4.204002, 1.298935};
    const double true_mean2[] = {6.546807, 2.949613, 5.482252, 1.985523};
    const double true_mean3[] = {5.006000, 3.428000, 1.462000, 0.246000};
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(gmm.means()[0][static_cast<std::size_t>(i)], true_mean1[i], 1e-2);
        CHECK_NEAR(gmm.means()[1][static_cast<std::size_t>(i)], true_mean2[i], 1e-2);
        CHECK_NEAR(gmm.means()[2][static_cast<std::size_t>(i)], true_mean3[i], 1e-2);
    }
}

// --- COREHYDRO SUPPLEMENT (no C# counterpart) ---------------------------------------------

void test_fit_invariants() {
    ml::GaussianMixtureModel gmm(iris_features(), 3);
    gmm.train(12345);

    // The mixing weights are a probability vector.
    double w = 0;
    for (double wk : gmm.weights()) {
        CHECK_TRUE(wk > 0.0 && wk < 1.0);
        w += wk;
    }
    CHECK_NEAR(w, 1.0, 1e-12);

    // Every covariance is symmetric with a positive diagonal.
    for (int k = 0; k < 3; k++) {
        const la::Matrix& s = gmm.sigmas()[static_cast<std::size_t>(k)];
        CHECK_EQ(s.number_of_rows(), 4);
        CHECK_EQ(s.number_of_columns(), 4);
        for (int i = 0; i < 4; i++) {
            CHECK_TRUE(s(i, i) > 0.0);
            for (int j = i + 1; j < 4; j++) CHECK_NEAR(s(i, j), s(j, i), 1e-12);
        }
    }

    // Each row of the responsibility matrix is a probability vector, and the label is that
    // row's argmax.
    CHECK_EQ(static_cast<int>(gmm.likelihood_matrix().size()), 150);
    for (std::size_t i = 0; i < gmm.likelihood_matrix().size(); i++) {
        double row_sum = 0;
        std::size_t argmax = 0;
        for (std::size_t k = 0; k < 3; k++) {
            double r = gmm.likelihood_matrix()[i][k];
            CHECK_TRUE(r >= 0.0 && r <= 1.0);
            row_sum += r;
            if (r > gmm.likelihood_matrix()[i][argmax]) argmax = k;
        }
        CHECK_NEAR(row_sum, 1.0, 1e-12);
        CHECK_EQ(gmm.labels()[i], static_cast<int>(argmax));
    }

    // The fit converged, so the log-likelihood was recorded (transcription note 2).
    CHECK_TRUE(std::isfinite(gmm.log_likelihood()));
    CHECK_TRUE(gmm.iterations() >= 2);
    CHECK_TRUE(gmm.iterations() < gmm.max_iterations());
}

void test_single_component_closed_form() {
    // A one-component GMM has a closed form: the mean is the sample mean and the covariance is
    // the POPULATION covariance (the M-step divides by the responsibility total, which is n).
    // This is the one exact check available for this class without a C# literal, and it is what
    // HypothesisTests::unimodality_test's k = 1 arm depends on.
    ml::GaussianMixtureModel gmm(iris::kSepalLength, 1);
    gmm.train(4321);

    CHECK_NEAR(gmm.means()[0][0], nd::mean(iris::kSepalLength), 1e-8);
    CHECK_NEAR(gmm.sigmas()[0](0, 0), nd::population_variance(iris::kSepalLength), 1e-8);
    CHECK_NEAR(gmm.weights()[0], 1.0, 1e-12);
    for (int l : gmm.labels()) CHECK_EQ(l, 0);
    // Every responsibility is exactly 1 with a single component.
    for (std::size_t i = 0; i < gmm.likelihood_matrix().size(); i++)
        CHECK_NEAR(gmm.likelihood_matrix()[i][0], 1.0, 1e-15);

    // UPSTREAM DEFECT, mirrored and pinned. The E-step forms
    //   likelihood[i][k] = -0.5 * (quadform + logDet(Sigma_k)) + log(weight_k)
    // which OMITS the multivariate-normal normalizing constant -0.5 * D * log(2*pi). So
    // `LogLikelihood` is short of the true Gaussian mixture log-likelihood by exactly
    // n * D/2 * log(2*pi) -- here 150 * 0.5 * log(2*pi) = 137.84. Verified below by asserting
    // both the reported value and the properly normalized one.
    //
    // It is a constant for fixed n and D, so it changes nothing about the EM fit, and it CANCELS
    // in the only place upstream consumes it: HypothesisTests.UnimodalityTest's likelihood-ratio
    // statistic 2 * (logLH2 - logLH1) compares two fits over the same sample. It would matter to
    // anyone using this value as a model-selection criterion. See docs/upstream-csharp-issues.md.
    double mu = gmm.means()[0][0];
    double var = gmm.sigmas()[0](0, 0);
    double normalized = 0;
    for (double x : iris::kSepalLength)
        normalized += -0.5 * (std::log(2.0 * corehydro::numerics::kPi) + std::log(var) +
                              (x - mu) * (x - mu) / var);
    double omitted_constant =
        0.5 * static_cast<double>(iris::kSepalLength.size()) * std::log(2.0 * corehydro::numerics::kPi);
    CHECK_NEAR(gmm.log_likelihood(), normalized + omitted_constant, 1e-6);
}

void test_seeded_determinism() {
    ml::GaussianMixtureModel a(iris_features(), 3);
    ml::GaussianMixtureModel b(iris_features(), 3);
    a.train(12345);
    b.train(12345);
    CHECK_EQ(a.iterations(), b.iterations());
    CHECK_EQ(a.log_likelihood(), b.log_likelihood());
    for (std::size_t k = 0; k < 3; k++) {
        CHECK_EQ(a.weights()[k], b.weights()[k]);
        for (std::size_t j = 0; j < 4; j++) CHECK_EQ(a.means()[k][j], b.means()[k][j]);
    }
    for (std::size_t i = 0; i < a.labels().size(); i++) CHECK_EQ(a.labels()[i], b.labels()[i]);
}

void test_tolerance_and_iteration_cap() {
    // A tight iteration cap stops the run before convergence, which leaves `log_likelihood()` at
    // its default 0 -- transcription note 2, mirrored from upstream rather than "fixed".
    ml::GaussianMixtureModel capped(iris_features(), 3);
    capped.set_max_iterations(3);
    capped.train(12345);
    CHECK_EQ(capped.iterations(), 4);  // the loop variable overshoots the cap by one
    CHECK_EQ(capped.log_likelihood(), 0.0);
    // The fit itself is still usable -- weights and labels were produced.
    double w = 0;
    for (double wk : capped.weights()) w += wk;
    CHECK_NEAR(w, 1.0, 1e-12);

    // A loose tolerance converges sooner than the default.
    ml::GaussianMixtureModel loose(iris_features(), 3);
    loose.set_tolerance(1e-2);
    loose.train(12345);
    ml::GaussianMixtureModel tight(iris_features(), 3);
    tight.train(12345);
    CHECK_TRUE(loose.iterations() <= tight.iterations());
    CHECK_TRUE(std::isfinite(loose.log_likelihood()));
}

}  // namespace

int main() {
    test_gmm_iris();
    test_fit_invariants();
    test_single_component_closed_form();
    test_seeded_determinism();
    test_tolerance_and_iteration_cap();
    return chtest::summary("test_gaussian_mixture_model");
}
