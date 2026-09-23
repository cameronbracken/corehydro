// Standalone tests for corehydro::numerics::BootstrapAnalysis + the IBootstrappable capability.
//
// Transcribed VERBATIM (oracle values unaltered) from
//   upstream/Numerics/Test_Numerics/Distributions/Univariate/Test_BootstrapAnalysis.cs @ 2a0357a
//
// BootstrapAnalysis is an internal Numerics support engine (not exposed through the R/Python
// public API), so per the tolerance/oracle policy its oracles are transcribed here in a C++-only
// ctest rather than living in fixtures/*.json. The R-boot CI arrays (true05/true95) are compared at
// 1% relative tolerance (0.01 * true[i]) exactly as the C# test does -- these are RNG-parity checks
// that ride on the codebase's proven MersenneTwister bit-parity. The Bootstrap-t and BCa tests
// compare against Normal.MonteCarloConfidenceIntervals, which the C# Normal exposes; that WPF helper
// is not ported onto Normal (its header documents the omission), so it is transcribed here as a
// local test helper (ChiSquared + NextDoubles), matching the C# formula-for-formula. The
// equivalence test is RNG-free (fixed shared bootDistributions) and asserts at 1e-8.
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/chi_squared.hpp"
#include "corehydro/numerics/distributions/log_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/bootstrap_analysis.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/standard_error_extensions.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"
#include "check.hpp"

using corehydro::numerics::BootstrapAnalysis;
using corehydro::numerics::distributions::ChiSquared;
using corehydro::numerics::distributions::LogNormal;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::ParameterEstimationMethod;
using corehydro::numerics::distributions::UncertaintyAnalysisResults;
using corehydro::numerics::distributions::UnivariateDistributionBase;

namespace {

// The 25 non-exceedance probabilities shared by the CI tests (Test_BootstrapAnalysis.cs:35).
const std::vector<double> kProbs25 = {0.999999, 0.999998, 0.999995, 0.99999, 0.99998, 0.99995,
                                      0.9999,   0.9998,   0.9995,   0.999,   0.998,   0.995,
                                      0.99,     0.98,     0.95,     0.9,     0.8,     0.7,
                                      0.5,      0.3,      0.2,      0.1,     0.05,    0.02,
                                      0.01};

// Local transcription of Normal.MonteCarloConfidenceIntervals (Numerics/Distributions/Univariate/
// Normal.cs:733) -- the "same sampling approach as HEC-FDA" the C# test uses to build its truth
// intervals. Not ported onto Normal itself (WPF helper); kept here as a test oracle helper.
std::vector<std::array<double, 2>> monte_carlo_ci(const Normal& dist, int sampleSize,
                                                  int realizations,
                                                  const std::vector<double>& quantiles,
                                                  const std::array<double, 2>& percentiles) {
    double originalMean = dist.mean();
    double originalStdDev = dist.standard_deviation();

    corehydro::numerics::sampling::MersenneTwister r(12345);
    auto rndMean = corehydro::numerics::utilities::next_doubles(r, realizations);
    auto rndStdDev = corehydro::numerics::utilities::next_doubles(r, realizations);

    std::vector<Normal> mc(static_cast<std::size_t>(realizations));
    for (int idx = 0; idx < realizations; ++idx) {
        Normal meanDist(originalMean, originalStdDev / std::sqrt(static_cast<double>(sampleSize)));
        double newMu = meanDist.inverse_cdf(rndMean[static_cast<std::size_t>(idx)]);
        ChiSquared chi(sampleSize - 1);
        double newSigma = std::sqrt((sampleSize - 1) * std::pow(originalStdDev, 2.0) /
                                    chi.inverse_cdf(rndStdDev[static_cast<std::size_t>(idx)]));
        mc[static_cast<std::size_t>(idx)] = Normal(newMu, newSigma);
    }

    std::vector<std::array<double, 2>> out(quantiles.size());
    for (std::size_t i = 0; i < quantiles.size(); ++i) {
        std::vector<double> xValues(static_cast<std::size_t>(realizations));
        for (int idx = 0; idx < realizations; ++idx)
            xValues[static_cast<std::size_t>(idx)] =
                mc[static_cast<std::size_t>(idx)].inverse_cdf(quantiles[i]);
        for (int j = 0; j < 2; ++j)
            out[i][static_cast<std::size_t>(j)] =
                corehydro::numerics::data::percentile(xValues, percentiles[static_cast<std::size_t>(j)]);
    }
    return out;
}

// Test_NormalCI (line 33): Normal method CI vs 'boot' package, 1% relative tol.
void test_normal_ci() {
    Normal dist(3.122599, 0.5573654);
    BootstrapAnalysis boot(dist, ParameterEstimationMethod::MethodOfMoments, 100);
    auto CIs = boot.normal_quantile_ci(kProbs25);

    const std::vector<double> true05 = {5.451061, 5.3807, 5.284459, 5.208962, 5.130887, 5.023248,
                                        4.938038, 4.849109, 4.724954, 4.625182, 4.519401, 4.368283,
                                        4.243235, 4.106139, 3.899265, 3.713675, 3.485485, 3.317521,
                                        3.031074, 2.732176, 2.546064, 2.283231, 2.063337, 1.813848,
                                        1.646703};
    const std::vector<double> true95 = {6.098854, 6.010689, 5.890184, 5.795726, 5.698123, 5.563704,
                                        5.457429, 5.34666, 5.192298, 5.068532, 4.937633, 4.751333,
                                        4.597948, 4.430811, 4.181338, 3.961466, 3.698674, 3.512604,
                                        3.213796, 2.927438, 2.759516, 2.531367, 2.345799, 2.138941,
                                        2.001853};
    for (std::size_t i = 0; i < kProbs25.size(); ++i) {
        CHECK_NEAR(CIs[i][0], true05[i], 0.01 * true05[i]);
        CHECK_NEAR(CIs[i][1], true95[i], 0.01 * true95[i]);
    }
}

// Test_PercentileCI (line 60): Percentile method CI vs 'boot' package, 1% relative tol.
void test_percentile_ci() {
    Normal dist(3.122599, 0.5573654);
    BootstrapAnalysis boot(dist, ParameterEstimationMethod::MethodOfMoments, 100);
    auto CIs = boot.percentile_quantile_ci(kProbs25);

    const std::vector<double> true05 = {5.448065, 5.378149, 5.280887, 5.205002, 5.127415, 5.020843,
                                        4.935679, 4.847712, 4.723488, 4.623268, 4.517452, 4.367232,
                                        4.24338, 4.106147, 3.900035, 3.714043, 3.486811, 3.318758,
                                        3.032698, 2.734326, 2.548027, 2.284871, 2.06533, 1.816327,
                                        1.648123};
    const std::vector<double> true95 = {6.095642, 6.00775, 5.887441, 5.792823, 5.696038, 5.562841,
                                        5.457508, 5.346516, 5.191728, 5.067822, 4.9371, 4.750642,
                                        4.596812, 4.428688, 4.178886, 3.95975, 3.69779, 3.51265,
                                        3.215176, 2.930193, 2.761301, 2.533379, 2.3472, 2.139633,
                                        2.003285};
    for (std::size_t i = 0; i < kProbs25.size(); ++i) {
        CHECK_NEAR(CIs[i][0], true05[i], 0.01 * true05[i]);
        CHECK_NEAR(CIs[i][1], true95[i], 0.01 * true95[i]);
    }
}

// Test_BootstrapTCI (line 88): Bootstrap-t CI vs the Monte-Carlo (Noncentral-t) truth, 1% rel tol.
void test_bootstrap_t_ci() {
    Normal dist(3.122599, 0.5573654);
    BootstrapAnalysis boot(dist, ParameterEstimationMethod::MethodOfMoments, 100);
    auto CIs = boot.bootstrap_t_quantile_ci(kProbs25);

    auto trueCIs = monte_carlo_ci(dist, 100, 10000, kProbs25, {0.05, 0.95});
    for (std::size_t i = 0; i < kProbs25.size(); ++i) {
        CHECK_NEAR(CIs[i][0], trueCIs[i][0], 0.01 * trueCIs[i][0]);
        CHECK_NEAR(CIs[i][1], trueCIs[i][1], 0.01 * trueCIs[i][1]);
    }
}

// Test_BCaCI (line 114): BCa CI vs the Monte-Carlo (Noncentral-t) truth, 1% rel tol.
void test_bca_ci() {
    const std::vector<double> sampleData = {
        3.292764, 3.354733, 2.945348, 2.773251, 3.302944, 2.091022, 3.315049, 2.861908, 2.85792,
        2.540339, 2.941876, 3.908656, 3.185314, 3.260108, 2.624734, 3.40845, 2.556821, 2.834211,
        3.560356, 3.149362, 3.389811, 3.727893, 2.677836, 2.223431, 2.201145, 3.902549, 2.759176,
        3.31019, 3.306062, 2.918845, 3.405937, 4.098417, 4.024595, 3.816223, 3.127136, 3.245594,
        2.837957, 2.168975, 3.883867, 3.012901, 3.564255, 1.809821, 2.469867, 3.46857, 3.427226,
        3.730365, 2.293451, 3.283702, 3.291594, 2.346601, 2.729807, 3.973846, 3.026795, 3.175831,
        2.664512, 3.138977, 3.345586, 3.411898, 4.072533, 1.826528, 3.074796, 2.328734, 3.276652,
        3.794981, 2.70656, 2.083811, 3.44407, 3.796744, 3.258427, 2.352164, 3.027308, 2.607675,
        2.475324, 4.165256, 3.701353, 3.4713, 3.413129, 2.59423, 3.238124, 3.510629, 3.322692,
        3.521572, 2.847815, 4.238555, 3.48561, 3.93355, 3.336021, 2.846023, 3.268262, 3.412435,
        2.518049, 2.572459, 3.943473, 2.80409, 2.509684, 3.343666, 2.747478, 4.07886, 2.700101,
        2.652727};
    Normal dist(3.122599, 0.5573654);
    BootstrapAnalysis boot(dist, ParameterEstimationMethod::MethodOfMoments, 100);
    auto CIs = boot.bca_quantile_ci(sampleData, kProbs25);

    auto trueCIs = monte_carlo_ci(dist, 100, 10000, kProbs25, {0.05, 0.95});
    for (std::size_t i = 0; i < kProbs25.size(); ++i) {
        CHECK_NEAR(CIs[i][0], trueCIs[i][0], 0.01 * trueCIs[i][0]);
        CHECK_NEAR(CIs[i][1], trueCIs[i][1], 0.01 * trueCIs[i][1]);
    }
}

// Test_BootstrapAnalysis_UncertaintyAnalysisResults_Equivalence (line 140): RNG-free equivalence at
// 1e-8 of the Estimate() path vs the UncertaintyAnalysisResults compute-ctor, sharing one fixed
// set of bootstrapped distributions.
void test_uar_equivalence() {
    const std::vector<double> probabilities = {0.999, 0.998, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8,
                                               0.7,   0.5,   0.3,   0.2,  0.1,  0.05, 0.02, 0.01};
    double alpha = 0.1;
    Normal dist(3.122599, 0.5573654);
    BootstrapAnalysis boot(dist, ParameterEstimationMethod::MethodOfMoments, 100);

    // Generate bootstrap distributions once and share between both methods.
    auto bootDistributions = boot.distributions();
    std::vector<const UnivariateDistributionBase*> view;
    view.reserve(bootDistributions.size());
    for (const auto& d : bootDistributions) view.push_back(d.get());

    // Reference result from BootstrapAnalysis.Estimate.
    auto reference = boot.estimate(probabilities, alpha, &view, /*recordParameterSets=*/false);

    // Result from UncertaintyAnalysisResults constructor.
    UncertaintyAnalysisResults result(dist, view, probabilities, alpha);

    // ModeCurve
    CHECK_EQ(result.mode_curve.size(), reference.mode_curve.size());
    for (std::size_t i = 0; i < reference.mode_curve.size(); ++i)
        CHECK_NEAR(result.mode_curve[i], reference.mode_curve[i], 1e-8);

    // ConfidenceIntervals
    CHECK_EQ(result.confidence_intervals.size(), reference.confidence_intervals.size());
    for (std::size_t i = 0; i < reference.confidence_intervals.size(); ++i) {
        CHECK_NEAR(result.confidence_intervals[i][0], reference.confidence_intervals[i][0], 1e-8);
        CHECK_NEAR(result.confidence_intervals[i][1], reference.confidence_intervals[i][1], 1e-8);
    }

    // MeanCurve
    CHECK_EQ(result.mean_curve.size(), reference.mean_curve.size());
    for (std::size_t i = 0; i < reference.mean_curve.size(); ++i)
        CHECK_NEAR(result.mean_curve[i], reference.mean_curve[i], 1e-8);
}

// Guard: a distribution that is not IBootstrappable is rejected; sampleSize/replications bounds.
void test_guards() {
    Normal dist(3.122599, 0.5573654);
    // sampleSize < 10 throws.
    CHECK_THROWS(BootstrapAnalysis(dist, ParameterEstimationMethod::MethodOfMoments, 5));
    // replications < 100 throws.
    CHECK_THROWS(BootstrapAnalysis(dist, ParameterEstimationMethod::MethodOfMoments, 100, 50));
}

void test_log_normal_bootstrap_preserves_configured_base() {
    LogNormal distribution(3.0, 0.25);
    distribution.set_base(2.0);
    BootstrapAnalysis bootstrap(distribution, ParameterEstimationMethod::MethodOfMoments,
                                20, 100, 42);

    auto samples = bootstrap.distributions();
    CHECK_EQ(bootstrap.failed_replications(), 0);
    for (const auto& sample : samples) {
        const auto* log_normal = dynamic_cast<const LogNormal*>(sample.get());
        CHECK_TRUE(log_normal != nullptr);
        CHECK_NEAR(log_normal->base(), 2.0, 0.0);
    }
}

void test_log_abs_quantile_jacobian_extension() {
    LogNormal distribution(3.0, 0.25);
    const std::vector<double> probabilities{0.25, 0.75};
    double determinant = 0.0;
    distribution.quantile_jacobian(probabilities, determinant);
    const double expected = std::log(std::fabs(determinant));
    CHECK_NEAR(corehydro::numerics::distributions::log_abs_quantile_jacobian(
                   distribution, probabilities),
               expected, 1e-14);
}

}  // namespace

int main() {
    test_normal_ci();
    test_percentile_ci();
    test_bootstrap_t_ci();
    test_bca_ci();
    test_uar_equivalence();
    test_guards();
    test_log_normal_bootstrap_preserves_configured_base();
    test_log_abs_quantile_jacobian_extension();
    return chtest::summary("bootstrap_analysis");
}
