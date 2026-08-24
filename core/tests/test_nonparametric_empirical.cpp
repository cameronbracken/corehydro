// Standalone tests for DataFrame::get_nonparametric_moments() / get_nonparametric_moments_ros()
// duplicate-value handling (BestFit v2.0.0's CreateEmpiricalDistributionWithUniqueValues,
// T12).
//
// Oracle is the upstream C# test class @ c2e6192:
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/DataFrame/NonparametricEmpiricalTests.cs
// Only the two GetNonparametricMoments*/duplicate-value cases are transcribed here: the
// other two C# tests (SummaryAndStandardizedValues_DuplicateValues_ReturnFiniteResults,
// EmpiricalConsumers_AllValuesIdentical_ReportUnavailableResults) additionally exercise
// SummaryStatisticsAllData/SetStandardizedValues, which are now ported (P4 Task 5) and
// transcribed instead in test_data_frame_facades.cpp, alongside the exact-data
// hypothesis-test facade; the degenerate-input half of the SECOND test (all-identical
// values -> GetNonparametricMoments returns null/nullopt) is still covered below since
// get_nonparametric_moments() itself IS ported (and was already, since B9).
//
// These checks are invariant-based (finite / null), not pinned to a specific C# oracle
// literal -- GetNonparametricMoments has no fixture-dispatch surface (it is consumed only
// internally by Bulletin17CDistribution's ROS default-initialization path), matching the
// established "throw-behavior / structural-invariant tests go in a C++-only ctest"
// precedent (see the file header on data_frame_plotting.hpp's tie-separation regressions).
#include <cmath>
#include <optional>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "check.hpp"

using corehydro::models::DataFrame;
using corehydro::models::ExactSeries;

static bool all_finite(const std::vector<double>& v) {
    for (double x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

// C# GetNonparametricMoments_DuplicateValues_ReturnsFiniteMoments: five distinct values
// across eight observations -- collapsing ties still leaves a valid empirical
// distribution (>= 2 distinct values), so the moments must be finite, not NaN/thrown.
static void test_get_nonparametric_moments_duplicate_values_returns_finite_moments() {
    DataFrame df;
    df.set_exact_series(ExactSeries(
        std::vector<double>{100.0, 100.0, 125.0, 125.0, 150.0, 175.0, 200.0, 200.0}));
    df.calculate_plotting_positions();

    std::optional<std::vector<double>> moments = df.get_nonparametric_moments();

    CHECK_TRUE(moments.has_value());
    if (moments.has_value()) CHECK_TRUE(all_finite(*moments));
}

// C# GetNonparametricMomentsROS_DuplicateValues_ReturnsFiniteMoments: the B17C
// initialization path exercised by negatively skewed and bootstrap samples -- repeated
// values among the uncensored/imputed set must not produce a degenerate (duplicate-X)
// EmpiricalDistribution.
static void test_get_nonparametric_moments_ros_duplicate_values_returns_finite_moments() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>{
        10.0, 100.0, 100.0, 125.0, 125.0, 150.0, 150.0, 200.0, 200.0, 250.0}));
    df.set_low_outlier_threshold(50.0);
    df.set_low_outliers_from_threshold();
    df.calculate_plotting_positions();

    std::optional<std::vector<double>> moments =
        df.get_nonparametric_moments_ros(/*use_log10_values=*/true);

    CHECK_TRUE(moments.has_value());
    if (moments.has_value()) CHECK_TRUE(all_finite(*moments));
}

// C# EmpiricalConsumers_AllValuesIdentical_ReportUnavailableResults (the ported half):
// an all-identical sample collapses to a single distinct X-value after deduping, so
// get_nonparametric_moments() must report unavailable (nullopt) rather than construct a
// one-point EmpiricalDistribution.
static void test_get_nonparametric_moments_all_values_identical_returns_nullopt() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>(10, 100.0)));
    df.calculate_plotting_positions();

    CHECK_TRUE(!df.get_nonparametric_moments().has_value());
}

// Degenerate-input companion for the ROS variant: an all-identical UNCENSORED set (no low
// outliers) falls through to get_nonparametric_moments(), which must likewise report
// unavailable rather than throw building a duplicate-X EmpiricalDistribution.
static void test_get_nonparametric_moments_ros_all_values_identical_returns_nullopt() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>(10, 100.0)));
    df.calculate_plotting_positions();

    CHECK_TRUE(!df.get_nonparametric_moments_ros().has_value());
}

int main() {
    test_get_nonparametric_moments_duplicate_values_returns_finite_moments();
    test_get_nonparametric_moments_ros_duplicate_values_returns_finite_moments();
    test_get_nonparametric_moments_all_values_identical_returns_nullopt();
    test_get_nonparametric_moments_ros_all_values_identical_returns_nullopt();
    return chtest::summary("nonparametric_empirical");
}
