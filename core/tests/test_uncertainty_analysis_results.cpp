// Standalone tests for corehydro::numerics::distributions::UncertaintyAnalysisResults.
//
// Oracle for behavior is the C# source itself (upstream/Numerics/Numerics/Distributions/
// Univariate/Uncertainty Analysis/UncertaintyAnalysisResults.cs @ 2a0357a) and the equivalence
// STRUCTURE of Test_BootstrapAnalysis.cs::Test_BootstrapAnalysis_UncertaintyAnalysisResults_
// Equivalence. BootstrapAnalysis is not ported (Phase 9), so instead of replaying it we hand-build
// a deterministic fixed fan of sampled Normal distributions around the same parent the C# test uses
// -- Normal(3.122599, 0.5573654) -- and side-compute the expected ModeCurve/ConfidenceIntervals in
// the test, then assert the class reproduces them. This is internal core math not exposed to
// R/Python, so oracles are transcribed/side-computed here; no fixtures/ entry.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "check.hpp"

using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::UncertaintyAnalysisResults;
using corehydro::numerics::distributions::UnivariateDistributionBase;

namespace {

// The same 16 non-exceedance probabilities the C# equivalence test uses (decreasing order).
const std::vector<double> kProbabilities = {0.999, 0.998, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8,
                                            0.7,   0.5,   0.3,   0.2,  0.1,  0.05, 0.02, 0.01};

// A deterministic, fixed fan of sampled Normal distributions around the parent used by the C#
// test. Twenty dists with hand-listed (mean, sd) pairs -- no sampling, so the result is exactly
// reproducible in a side computation.
struct FixedSample {
    std::vector<Normal> dists;
    std::vector<const UnivariateDistributionBase*> ptrs;

    FixedSample() {
        const std::vector<std::array<double, 2>> params = {
            {3.05, 0.52}, {3.10, 0.54}, {3.12, 0.55}, {3.15, 0.56}, {3.20, 0.57},
            {2.98, 0.50}, {3.08, 0.53}, {3.18, 0.58}, {3.22, 0.59}, {3.00, 0.51},
            {3.13, 0.555}, {3.09, 0.545}, {3.17, 0.565}, {3.06, 0.525}, {3.24, 0.60},
            {2.95, 0.49}, {3.11, 0.548}, {3.19, 0.575}, {3.07, 0.535}, {3.14, 0.558}};
        dists.reserve(params.size());
        for (const auto& p : params) dists.emplace_back(p[0], p[1]);
        ptrs.reserve(dists.size());
        for (const auto& d : dists) ptrs.push_back(&d);
    }
};

// ---- ModeCurve: pure inverse-cdf identity of the parent (rel 1e-9) ----
void test_mode_curve_is_parent_inverse_cdf_identity() {
    Normal parent(3.122599, 0.5573654);
    FixedSample s;
    UncertaintyAnalysisResults r(parent, s.ptrs, kProbabilities, 0.1);

    CHECK_EQ(r.mode_curve.size(), kProbabilities.size());
    for (std::size_t i = 0; i < kProbabilities.size(); ++i) {
        double expected = parent.inverse_cdf(kProbabilities[i]);
        CHECK_NEAR(r.mode_curve[i], expected, std::fabs(expected) * 1e-9 + 1e-12);
    }
    // The parent pointer is recorded.
    CHECK_TRUE(r.parent_distribution == &parent);
}

// ---- ConfidenceIntervals: side-computed percentile oracle (1e-8) ----
void test_confidence_intervals_match_side_computed_percentiles() {
    Normal parent(3.122599, 0.5573654);
    FixedSample s;
    double alpha = 0.1;
    UncertaintyAnalysisResults r(parent, s.ptrs, kProbabilities, alpha);

    double lowerCI = alpha / 2.0;
    double upperCI = 1.0 - alpha / 2.0;

    CHECK_EQ(r.confidence_intervals.size(), kProbabilities.size());
    for (std::size_t i = 0; i < kProbabilities.size(); ++i) {
        std::vector<double> vals;
        for (const auto* d : s.ptrs) vals.push_back(d->inverse_cdf(kProbabilities[i]));
        std::sort(vals.begin(), vals.end());
        double expLo = corehydro::numerics::data::percentile(vals, lowerCI, true);
        double expHi = corehydro::numerics::data::percentile(vals, upperCI, true);
        CHECK_NEAR(r.confidence_intervals[i][0], expLo, 1e-8);
        CHECK_NEAR(r.confidence_intervals[i][1], expHi, 1e-8);
        // Lower bound never above upper bound.
        CHECK_TRUE(r.confidence_intervals[i][0] <= r.confidence_intervals[i][1]);
    }
}

// ---- MeanCurve: finite, correct length, monotone consistent with decreasing probabilities ----
void test_mean_curve_finite_length_and_monotone() {
    Normal parent(3.122599, 0.5573654);
    FixedSample s;
    UncertaintyAnalysisResults r(parent, s.ptrs, kProbabilities, 0.1);

    CHECK_EQ(r.mean_curve.size(), kProbabilities.size());
    for (std::size_t i = 0; i < r.mean_curve.size(); ++i) {
        CHECK_TRUE(std::isfinite(r.mean_curve[i]));
    }
    // Probabilities are strictly decreasing, so the interpolated quantile mean curve is
    // strictly decreasing too (the C# append rule guarantees the point set is strictly
    // increasing in x = expected probability, and the interpolator is monotone).
    for (std::size_t i = 1; i < r.mean_curve.size(); ++i) {
        CHECK_TRUE(r.mean_curve[i] < r.mean_curve[i - 1]);
    }
}

// ---- recordParameterSets = true ----
void test_record_parameter_sets_captures_each_dists_parameters() {
    Normal parent(3.122599, 0.5573654);
    FixedSample s;
    UncertaintyAnalysisResults r(parent, s.ptrs, kProbabilities, 0.1, 0.001, 1.0 - 1e-9, true);

    CHECK_EQ(r.parameter_sets.size(), s.ptrs.size());
    for (std::size_t i = 0; i < s.ptrs.size(); ++i) {
        auto expected = s.ptrs[i]->get_parameters();
        CHECK_EQ(r.parameter_sets[i].values.size(), expected.size());
        for (std::size_t j = 0; j < expected.size(); ++j) {
            CHECK_TRUE(r.parameter_sets[i].values[j] == expected[j]);
        }
        CHECK_TRUE(std::isnan(r.parameter_sets[i].fitness));
    }
}

// ---- Fit scalars default to NaN ----
void test_fit_scalars_default_nan() {
    Normal parent(3.122599, 0.5573654);
    FixedSample s;
    UncertaintyAnalysisResults r(parent, s.ptrs, kProbabilities, 0.1);
    CHECK_TRUE(std::isnan(r.aic));
    CHECK_TRUE(std::isnan(r.bic));
    CHECK_TRUE(std::isnan(r.dic));
    CHECK_TRUE(std::isnan(r.rmse));
    CHECK_TRUE(std::isnan(r.erl));

    // Empty DTO ctor also leaves them NaN.
    UncertaintyAnalysisResults empty;
    CHECK_TRUE(std::isnan(empty.aic));
    CHECK_TRUE(std::isnan(empty.erl));
    CHECK_TRUE(empty.parent_distribution == nullptr);
}

// ---- Without recordParameterSets, parameter_sets stays empty ----
void test_no_record_leaves_parameter_sets_empty() {
    Normal parent(3.122599, 0.5573654);
    FixedSample s;
    UncertaintyAnalysisResults r(parent, s.ptrs, kProbabilities, 0.1);
    CHECK_EQ(r.parameter_sets.size(), static_cast<std::size_t>(0));
}

// ---- Guard cases ----
void test_guards_throw() {
    Normal parent(3.122599, 0.5573654);
    FixedSample s;
    std::vector<const UnivariateDistributionBase*> emptyDists;
    std::vector<double> emptyProbs;

    // Empty sampled dists.
    CHECK_THROWS(UncertaintyAnalysisResults(parent, emptyDists, kProbabilities, 0.1));
    // Empty probabilities.
    CHECK_THROWS(UncertaintyAnalysisResults(parent, s.ptrs, emptyProbs, 0.1));
    // Alpha out of (0,1).
    CHECK_THROWS(UncertaintyAnalysisResults(parent, s.ptrs, kProbabilities, 0.0));
    CHECK_THROWS(UncertaintyAnalysisResults(parent, s.ptrs, kProbabilities, 1.0));
    CHECK_THROWS(UncertaintyAnalysisResults(parent, s.ptrs, kProbabilities, -0.5));
    CHECK_THROWS(UncertaintyAnalysisResults(parent, s.ptrs, kProbabilities, 1.5));

    std::vector<const UnivariateDistributionBase*> allFailed{nullptr, nullptr};
    CHECK_THROWS(UncertaintyAnalysisResults(parent, allFailed, kProbabilities, 0.1));
}

void test_failed_ensemble_members_are_excluded_and_recorded_as_nan() {
    Normal parent(3.0, 0.5);
    Normal successful(4.0, 0.75);
    std::vector<const UnivariateDistributionBase*> ensemble{&successful, nullptr};
    const std::vector<double> probabilities{0.25, 0.5, 0.75};

    UncertaintyAnalysisResults result(parent, ensemble, probabilities, 0.1, 0.001,
                                      1.0 - 1e-9, true);
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        const double expected = successful.inverse_cdf(probabilities[i]);
        CHECK_NEAR(result.confidence_intervals[i][0], expected, 1e-12);
        CHECK_NEAR(result.confidence_intervals[i][1], expected, 1e-12);
        CHECK_NEAR(result.mean_curve[i], expected, 1e-8);
    }
    CHECK_EQ(result.parameter_sets[1].values.size(), static_cast<std::size_t>(2));
    CHECK_TRUE(std::isnan(result.parameter_sets[1].values[0]));
    CHECK_TRUE(std::isnan(result.parameter_sets[1].values[1]));
}

}  // namespace

int main() {
    test_mode_curve_is_parent_inverse_cdf_identity();
    test_confidence_intervals_match_side_computed_percentiles();
    test_mean_curve_finite_length_and_monotone();
    test_record_parameter_sets_captures_each_dists_parameters();
    test_fit_scalars_default_nan();
    test_no_record_leaves_parameter_sets_empty();
    test_guards_throw();
    test_failed_ensemble_members_are_excluded_and_recorded_as_nan();

    return chtest::summary("uncertainty_analysis_results");
}
