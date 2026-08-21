// Structural / behavioral tests for corehydro::analyses::FittingAnalysis + the
// corehydro::models::FittedDistribution DTO (A6).
//
// These transcribe the STRUCTURAL C# tests from
//   RMC.BestFit.Tests/DistributionFitting/FittingAnalysisTests.cs
// that exercise the ported surface. Per the Phase-8 policy there are NO per-candidate GoF
// oracle NUMBERS here (AIC/BIC/RMSE depend on the seeded DifferentialEvolution optimizer and
// are the A11 emitter's job). A real run() over a small exact-data frame is included but only
// its STRUCTURAL / finite properties are asserted, never exact GoF values.
//
// CANDIDATE COUNT: the C# DistributionList has 15 candidates and so does the port, so the two C#
// count tests (Constructor_InitializesDistributionList, Constructor_InitializesFittedDistributions)
// and the membership test (DistributionList_ContainsAll15Distributions) transcribe directly.
// (Through Phase 10 the port held only 14 -- GeneralizedNormal had no distribution-factory case --
// which is why the git history of this file counts to 14.)
//
// SKIPPED C# tests (all target dropped XML/INPC/cancellation surface):
//   ToXElement_CreatesValidXml, XmlRoundTrip_PreservesConfiguration,
//   ToXElement_ContainsProbabilityOrdinates, ToXElement_ContainsFittedDistributions,
//   Constructor_WithXElement_NullDataFrame_ThrowsArgumentNullException,
//   Constructor_WithXElement_NullXElement_ThrowsArgumentNullException,
//   DataFrame_Change_RaisesPropertyChanged, CancelAnalysis_WhenNotRunning_IsSafe,
//   ProbabilityOrdinatesChange_RaisesProbabilityOrdinatesPropertyChanged,
//   ProbabilityOrdinatesChange_DoesNotReplaceFittedDistributionsList,
//   ProbabilityOrdinatesChange_DoesNotFireSpuriousFittedDistributionsChange.
// The non-INPC intent of the ProbabilityOrdinates behavior tests (ordinate change is a no-op on
// fit state, does not throw) is kept below as test_probability_ordinates_change_is_noop.
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "corehydro/analyses/distribution_fitting/fitting_analysis.hpp"
#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/data_frame/data_collections/exact_series.hpp"
#include "corehydro/models/distribution_fitting/fitted_distribution.hpp"
#include "corehydro/numerics/data/probability_ordinates.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "check.hpp"

using corehydro::analyses::FittingAnalysis;
using corehydro::models::DataFrame;
using corehydro::models::ExactSeries;
using corehydro::models::FittedDistribution;
using corehydro::numerics::data::ProbabilityOrdinates;
using UDT = corehydro::numerics::distributions::UnivariateDistributionType;

namespace {

// The 15 candidate types, in the C# DistributionList order.
const std::vector<UDT>& expected_types() {
    static const std::vector<UDT> t = {
        UDT::Exponential,         UDT::GammaDistribution, UDT::GeneralizedExtremeValue,
        UDT::GeneralizedLogistic, UDT::GeneralizedNormal, UDT::GeneralizedPareto,
        UDT::Gumbel,              UDT::KappaFour,         UDT::LnNormal,
        UDT::Logistic,            UDT::LogNormal,         UDT::LogPearsonTypeIII,
        UDT::Normal,              UDT::PearsonTypeIII,    UDT::Weibull};
    return t;
}

// Sample annual peak flow data (cfs) from FittingAnalysisTests.cs (first 30 values).
const std::vector<double>& sample_peaks() {
    static const std::vector<double> v = {
        45000, 38000, 52000, 61000, 33000, 49000, 55000, 42000, 67000, 39000,
        48000, 51000, 36000, 58000, 44000, 53000, 47000, 62000, 41000, 50000,
        37000, 54000, 46000, 59000, 43000, 56000, 40000, 63000, 35000, 57000};
    return v;
}

// The small 10-value inline frame from CreateSmallTestDataFrame (cheap for a real run()).
const std::vector<double>& small_values() {
    static const std::vector<double> v = {12500.0, 15300, 8900,  22100, 18700,
                                          14200,   9800,  28500, 17400, 11600};
    return v;
}

std::unique_ptr<DataFrame> make_frame(const std::vector<double>& values) {
    auto df = std::make_unique<DataFrame>();
    df->set_exact_series(ExactSeries(values));
    df->calculate_plotting_positions();
    return df;
}

std::unique_ptr<FittingAnalysis> make_analysis(const std::vector<double>& values) {
    return std::make_unique<FittingAnalysis>(make_frame(values));
}

// ---- Constructor initializes the candidate list to the 15 C# types ----
// (C# Constructor_InitializesDistributionList / DistributionList_ContainsAll15Distributions.)
void test_candidate_count_and_membership() {
    auto analysis = make_analysis(sample_peaks());
    CHECK_EQ(analysis->distribution_list().size(), static_cast<std::size_t>(15));
    CHECK_EQ(analysis->distribution_list().size(), expected_types().size());
    for (std::size_t i = 0; i < expected_types().size(); ++i) {
        CHECK_TRUE(analysis->distribution_list()[i]->type() == expected_types()[i]);
    }
}

// ---- Constructor initializes FittedDistributions to one default entry per candidate ----
// (C# Constructor_InitializesFittedDistributions.)
void test_fitted_distributions_initialized() {
    auto analysis = make_analysis(sample_peaks());
    CHECK_EQ(analysis->fitted_distributions().size(), static_cast<std::size_t>(15));
    for (const auto& fd : analysis->fitted_distributions()) {
        CHECK_TRUE(!fd.fit_succeeded());
        CHECK_TRUE(std::isnan(fd.aic()));
        CHECK_TRUE(std::isnan(fd.bic()));
        CHECK_TRUE(std::isnan(fd.rmse()));
        CHECK_TRUE(fd.distribution() != nullptr);  // default holds a candidate clone
        CHECK_TRUE(fd.error_message().empty());
    }
    // Each fitted entry mirrors its candidate type, in order.
    for (std::size_t i = 0; i < expected_types().size(); ++i) {
        CHECK_TRUE(analysis->fitted_distributions()[i].distribution()->type() ==
                   expected_types()[i]);
    }
}

// ---- Constructor initializes ProbabilityOrdinates to the defaults, value-stable to 1e-10 ----
// (C# Constructor_InitializesProbabilityOrdinates + ordinate-preservation intent.)
void test_probability_ordinates_defaults() {
    auto analysis = make_analysis(sample_peaks());
    ProbabilityOrdinates defaults;  // fresh instance = the 25 standard defaults
    CHECK_TRUE(analysis->probability_ordinates().count() > 0);
    CHECK_EQ(analysis->probability_ordinates().count(), defaults.count());
    for (std::size_t i = 0; i < defaults.count(); ++i) {
        CHECK_NEAR(analysis->probability_ordinates()[i], defaults[i], 1e-10);
    }
}

// ---- Null / unset DataFrame throws (C# Constructor_WithNullDataFrame_ThrowsArgumentNullException) ----
void test_null_dataframe_throws() {
    CHECK_THROWS(FittingAnalysis(std::unique_ptr<DataFrame>{}));
}

// ---- is_estimated false before run(); clear_results keeps it false (C# IsEstimated transitions) ----
void test_is_estimated_before_run_and_clear() {
    auto analysis = make_analysis(sample_peaks());
    CHECK_TRUE(!analysis->is_estimated());
    analysis->clear_results();
    CHECK_TRUE(!analysis->is_estimated());
    CHECK_EQ(analysis->fitted_distributions().size(), static_cast<std::size_t>(15));
}

// ---- Ordinate change on a fresh (not-estimated) analysis is a no-op on fit state ----
// (C# ProbabilityOrdinatesChange_WhenNotEstimated_DoesNotThrow, INPC-free.)
void test_probability_ordinates_change_is_noop() {
    auto analysis = make_analysis(small_values());
    analysis->probability_ordinates().add(0.005);
    CHECK_TRUE(!analysis->is_estimated());
    CHECK_EQ(analysis->fitted_distributions().size(), static_cast<std::size_t>(15));
}

// ---- A real run() over a small exact frame: structural / finite properties only ----
// (No exact GoF oracle -- those land via the A11 emitter.)
void test_run_structural() {
    auto analysis = make_analysis(small_values());
    analysis->run();
    CHECK_TRUE(analysis->is_estimated());
    CHECK_EQ(analysis->fitted_distributions().size(), static_cast<std::size_t>(15));
    bool any_succeeded = false;
    for (const auto& fd : analysis->fitted_distributions()) {
        CHECK_TRUE(fd.distribution() != nullptr);
        if (fd.fit_succeeded()) {
            any_succeeded = true;
            CHECK_TRUE(std::isfinite(fd.aic()));
            CHECK_TRUE(std::isfinite(fd.bic()));
            CHECK_TRUE(std::isfinite(fd.rmse()));
        }
    }
    CHECK_TRUE(any_succeeded);  // at least one candidate fits this well-behaved data
}

// ---- Validate returns valid for a properly configured analysis ----
void test_validate_valid() {
    auto analysis = make_analysis(sample_peaks());
    CHECK_TRUE(analysis->validate().is_valid);
}

}  // namespace

int main() {
    test_candidate_count_and_membership();
    test_fitted_distributions_initialized();
    test_probability_ordinates_defaults();
    test_null_dataframe_throws();
    test_is_estimated_before_run_and_clear();
    test_probability_ordinates_change_is_noop();
    test_run_structural();
    test_validate_valid();

    return chtest::summary("fitting_analysis");
}
