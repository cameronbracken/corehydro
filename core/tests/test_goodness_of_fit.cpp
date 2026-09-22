// Standalone tests for corehydro::numerics::data::GoodnessOfFit.
//
// Oracle values are taken directly from the C# unit tests in
// upstream/Numerics/Test_Numerics/Data/Statistics/Test_GoodnessOfFit.cs
// and verified against the real Numerics library via tools/verify_oracles.py.
//
// The test data used for the statistical tests (KS, AD, Chi-squared) is the
// same construction as the C# tests:
//   Normal(100, 15),  data[i-1] = model.InverseCDF(i / 31.0),  i = 1..30
//
// Tolerances mirror the C# Assert.AreEqual tolerances.
#include <algorithm>
#include <cmath>
#include <vector>

#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "check.hpp"

using corehydro::numerics::data::GoodnessOfFit;
using corehydro::numerics::distributions::Normal;

namespace {

// Build the 30-point test dataset used by the C# GoodnessOfFit tests:
//   data[i-1] = Normal(100,15).InverseCDF(i/31)  for i = 1..30  (already sorted)
std::vector<double> make_test_data() {
    Normal norm(100.0, 15.0);
    std::vector<double> data(30);
    for (int i = 1; i <= 30; ++i)
        data[static_cast<std::size_t>(i - 1)] = norm.inverse_cdf(static_cast<double>(i) / 31.0);
    return data;
}

// logL = norm.LogLikelihood(data) used for AIC/AICc/BIC tests.
// Derived from: AIC(2, logL) == 246.02262441224  => logL = (4 - 246.02262441224) / 2
// (verified by running the C# oracle)
constexpr double kLogL = -121.01131220612;

// --- Information Criteria -----------------------------------------------

void test_aic() {
    double aic = GoodnessOfFit::aic(2, kLogL);
    CHECK_NEAR(aic, 246.02262441224, 1e-6);
}

void test_aicc() {
    double aicc = GoodnessOfFit::aicc(30, 2, kLogL);
    CHECK_NEAR(aicc, 246.467068856684, 1e-6);
}

void test_bic() {
    double bic_val = GoodnessOfFit::bic(30, 2, kLogL);
    CHECK_NEAR(bic_val, 248.825019175564, 1e-6);
}

void test_aic_weights() {
    std::vector<double> vals = {8.66, 5.6, 38.0};
    auto w = GoodnessOfFit::aic_weights(vals);
    // Expected from R qpcR::akaike.weights()
    CHECK_NEAR(w[0], 1.779937e-01, 1e-6);
    CHECK_NEAR(w[1], 8.220063e-01, 1e-6);
    CHECK_NEAR(w[2], 7.573637e-08, 1e-6);
}

// --- Error Metrics -------------------------------------------------------

void test_rmse_arrays() {
    // C# Test_RMSE1: observed = 1..30, modeled = Normal(100,15) quantiles
    auto data = make_test_data();
    std::vector<double> observed(30);
    for (int i = 0; i < 30; ++i) observed[static_cast<std::size_t>(i)] = i + 1;
    double rmse = GoodnessOfFit::rmse(observed, data, 2);
    CHECK_NEAR(rmse, 87.6252835187426, 1e-6);
}

void test_rmse_model() {
    // C# Test_RMSE2: RMSE(observed=1..30, norm) with Weibull PP
    Normal norm(100.0, 15.0);
    std::vector<double> observed(30);
    for (int i = 0; i < 30; ++i) observed[static_cast<std::size_t>(i)] = i + 1;
    double rmse = GoodnessOfFit::rmse(observed, norm);
    CHECK_NEAR(rmse, 87.6252835187426, 1e-6);
}

void test_rmse_pp() {
    // C# Test_RMSE3: RMSE with explicit Weibull PP
    Normal norm(100.0, 15.0);
    std::vector<double> observed(30);
    for (int i = 0; i < 30; ++i) observed[static_cast<std::size_t>(i)] = i + 1;
    std::vector<double> pp(30);
    for (int i = 1; i <= 30; ++i)
        pp[static_cast<std::size_t>(i - 1)] = static_cast<double>(i) / 31.0;
    double rmse = GoodnessOfFit::rmse(observed, pp, norm);
    CHECK_NEAR(rmse, 87.6252835187426, 1e-6);
}

void test_rmse_parameter_adjustment() {
    const std::vector<double> observed{0, 0, 0, 0};
    CHECK_NEAR(GoodnessOfFit::rmse(observed, {1, 2, 3, 4}, 1), std::sqrt(10.0), 1e-12);
    CHECK_NEAR(GoodnessOfFit::rmse(observed, {4, 1, 2, 3}, 1), std::sqrt(10.0), 1e-12);
    CHECK_THROWS(GoodnessOfFit::rmse(observed, observed, -1));
    CHECK_THROWS(GoodnessOfFit::rmse(observed, observed, 4));
}

void test_rmse_weights() {
    std::vector<double> vals = {8.66, 5.6, 38.0};
    auto w = GoodnessOfFit::rmse_weights(vals);
    CHECK_NEAR(w[0], 0.29041255, 1e-6);
    CHECK_NEAR(w[1], 0.69450458, 1e-6);
    CHECK_NEAR(w[2], 0.01508287, 1e-6);
}

void test_mse() {
    std::vector<double> obs = {2.5, 0.0, 2.1, 1.4};
    std::vector<double> mod = {3.0, -0.5, 2.0, 1.5};
    CHECK_NEAR(GoodnessOfFit::mse(obs, mod), 0.13, 1e-6);
}

void test_mae() {
    std::vector<double> obs = {2.5, 0.0, 2.1, 1.4};
    std::vector<double> mod = {3.0, -0.5, 2.0, 1.5};
    CHECK_NEAR(GoodnessOfFit::mae(obs, mod), 0.3, 1e-6);
}

void test_mape() {
    std::vector<double> obs = {2.5, 1.0, 2.1, 1.4};
    std::vector<double> mod = {3.0, 0.5, 2.0, 1.5};
    CHECK_NEAR(GoodnessOfFit::mape(obs, mod), 20.475, 1e-2);
}

void test_smape() {
    std::vector<double> obs = {2.5, 1.0, 2.1, 1.4};
    std::vector<double> mod = {3.0, 0.5, 2.0, 1.5};
    CHECK_NEAR(GoodnessOfFit::smape(obs, mod), 24.155, 1e-2);
}

// --- Efficiency Coefficients --------------------------------------------

void test_nse() {
    std::vector<double> obs = {2.5, 0.0, 2.1, 1.4, 3.2, 2.8, 1.9, 0.5};
    std::vector<double> mod = {3.0, -0.5, 2.0, 1.5, 3.0, 2.9, 2.1, 0.8};
    CHECK_NEAR(GoodnessOfFit::nash_sutcliffe_efficiency(obs, mod), 0.918981, 1e-5);
}

void test_kge() {
    std::vector<double> obs = {2.5, 0.0, 2.1, 1.4, 3.2, 2.8, 1.9, 0.5};
    std::vector<double> mod = {3.0, -0.5, 2.0, 1.5, 3.0, 2.9, 2.1, 0.8};
    CHECK_NEAR(GoodnessOfFit::kling_gupta_efficiency(obs, mod), 0.88573, 1e-4);
}

void test_kge_mod() {
    std::vector<double> obs = {2.5, 0.0, 2.1, 1.4, 3.2, 2.8, 1.9, 0.5};
    std::vector<double> mod = {3.0, -0.5, 2.0, 1.5, 3.0, 2.9, 2.1, 0.8};
    CHECK_NEAR(GoodnessOfFit::kling_gupta_efficiency_mod(obs, mod), 0.91295, 1e-4);
}

// --- Bias Metrics -------------------------------------------------------

void test_pbias() {
    std::vector<double> obs = {2.5, 0.0, 2.1, 1.4, 3.2, 2.8, 1.9, 0.5};
    std::vector<double> mod = {3.0, -0.5, 2.0, 1.5, 3.0, 2.9, 2.1, 0.8};
    CHECK_NEAR(GoodnessOfFit::pbias(obs, mod), 2.777778, 1e-5);
}

void test_rsr() {
    std::vector<double> obs = {2.5, 0.0, 2.1, 1.4, 3.2, 2.8, 1.9, 0.5};
    std::vector<double> mod = {3.0, -0.5, 2.0, 1.5, 3.0, 2.9, 2.1, 0.8};
    CHECK_NEAR(GoodnessOfFit::rsr(obs, mod), 0.284637521, 1e-5);
}

// --- Correlation and Determination --------------------------------------

void test_r_squared() {
    auto data = make_test_data();
    std::vector<double> observed(30);
    for (int i = 0; i < 30; ++i) observed[static_cast<std::size_t>(i)] = i + 1;
    CHECK_NEAR(GoodnessOfFit::r_squared(observed, data), 0.9803475, 1e-6);
}

// --- Index of Agreement -------------------------------------------------

void test_index_of_agreement() {
    std::vector<double> obs = {2.5, 0.0, 2.1, 1.4, 3.2, 2.8, 1.9, 0.5};
    std::vector<double> mod = {3.0, -0.5, 2.0, 1.5, 3.0, 2.9, 2.1, 0.8};
    CHECK_NEAR(GoodnessOfFit::index_of_agreement(obs, mod), 0.98147, 1e-4);
}

void test_modified_index_of_agreement() {
    std::vector<double> obs = {2.5, 0.0, 2.1, 1.4, 3.2, 2.8, 1.9, 0.5};
    std::vector<double> mod = {3.0, -0.5, 2.0, 1.5, 3.0, 2.9, 2.1, 0.8};
    CHECK_NEAR(GoodnessOfFit::modified_index_of_agreement(obs, mod), 0.86301, 1e-4);
}

void test_refined_index_of_agreement() {
    std::vector<double> obs = {2.5, 0.0, 2.1, 1.4, 3.2, 2.8, 1.9, 0.5};
    std::vector<double> mod = {3.0, -0.5, 2.0, 1.5, 3.0, 2.9, 2.1, 0.8};
    CHECK_NEAR(GoodnessOfFit::refined_index_of_agreement(obs, mod), 0.85714, 1e-4);
}

void test_volumetric_efficiency() {
    std::vector<double> obs = {2.5, 1.0, 2.1, 1.4, 3.2, 2.8, 1.9, 0.5};
    std::vector<double> mod = {3.0, 0.5, 2.0, 1.5, 3.0, 2.9, 2.1, 0.8};
    CHECK_NEAR(GoodnessOfFit::volumetric_efficiency(obs, mod), 0.87013, 1e-4);
}

// --- Classification Metrics ---------------------------------------------

void test_precision() {
    std::vector<double> obs = {1, 0, 1, 1, 0, 1, 0, 0};
    std::vector<double> mod = {1, 0, 1, 0, 0, 1, 1, 0};
    CHECK_NEAR(GoodnessOfFit::precision(obs, mod), 0.75, 1e-10);
}

void test_recall() {
    std::vector<double> obs = {1, 0, 1, 1, 0, 1, 0, 0};
    std::vector<double> mod = {1, 0, 1, 0, 0, 1, 1, 0};
    CHECK_NEAR(GoodnessOfFit::recall(obs, mod), 0.75, 1e-10);
}

void test_f1_score() {
    std::vector<double> obs = {1, 0, 1, 1, 0, 1, 0, 0};
    std::vector<double> mod = {1, 0, 1, 0, 0, 1, 1, 0};
    CHECK_NEAR(GoodnessOfFit::f1_score(obs, mod), 0.75, 1e-10);
}

void test_balanced_accuracy() {
    // All-zero predictions on 90/10 imbalanced data
    std::vector<double> obs = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    std::vector<double> mod = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    CHECK_NEAR(GoodnessOfFit::balanced_accuracy(obs, mod), 0.5, 1e-10);
}

// --- Statistical Tests --------------------------------------------------

void test_kolmogorov_smirnov() {
    // C# Test_KSTest: data = Normal(100,15).InverseCDF(i/31) for i=1..30,
    // model = Normal(100, 15). Expected: 0.032258 (tol 1e-6)
    auto data = make_test_data();
    Normal norm(100.0, 15.0);
    double ks = GoodnessOfFit::kolmogorov_smirnov(data, norm);
    CHECK_NEAR(ks, 0.032258, 1e-6);
}

void test_anderson_darling() {
    // C# Test_ADTest: same data, model fitted to data (mean/sd of data).
    // Expected: 0.044781 (tol 1e-6)
    auto data = make_test_data();
    // Compute mean and sample stdev of data (mirrors Statistics.Mean / StandardDeviation)
    int n = static_cast<int>(data.size());
    double mean = 0.0;
    for (double v : data) mean += v;
    mean /= n;
    double var = 0.0;
    for (double v : data) {
        double d = v - mean;
        var += d * d;
    }
    double sd = std::sqrt(var / (n - 1));
    Normal norm(mean, sd);
    double ad = GoodnessOfFit::anderson_darling(data, norm);
    CHECK_NEAR(ad, 0.044781, 1e-6);
}

void test_chi_squared() {
    // C# Test_ChiSquaredTest: model fitted to data (mean/sd of data).
    // Expected: 0.9279124 (tol 1e-6)
    auto data = make_test_data();
    int n = static_cast<int>(data.size());
    double mean = 0.0;
    for (double v : data) mean += v;
    mean /= n;
    double var = 0.0;
    for (double v : data) {
        double d = v - mean;
        var += d * d;
    }
    double sd = std::sqrt(var / (n - 1));
    Normal norm(mean, sd);
    double chi2 = GoodnessOfFit::chi_squared(data, norm);
    CHECK_NEAR(chi2, 0.9279124, 1e-6);
}

}  // namespace

int main() {
    // Information criteria
    test_aic();
    test_aicc();
    test_bic();
    test_aic_weights();

    // Error metrics
    test_rmse_arrays();
    test_rmse_model();
    test_rmse_pp();
    test_rmse_parameter_adjustment();
    test_rmse_weights();
    test_mse();
    test_mae();
    test_mape();
    test_smape();

    // Efficiency coefficients
    test_nse();
    test_kge();
    test_kge_mod();

    // Bias metrics
    test_pbias();
    test_rsr();

    // Correlation
    test_r_squared();

    // Index of agreement
    test_index_of_agreement();
    test_modified_index_of_agreement();
    test_refined_index_of_agreement();
    test_volumetric_efficiency();

    // Classification
    test_precision();
    test_recall();
    test_f1_score();
    test_balanced_accuracy();

    // Statistical tests
    test_kolmogorov_smirnov();
    test_anderson_darling();
    test_chi_squared();

    return chtest::summary("goodness_of_fit");
}
