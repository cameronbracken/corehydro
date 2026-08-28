// R1 support ctest (C++-only): the RatingCurve (BaRatin matrix-of-controls, ADDITION mode)
// stage-discharge ModelBase family. RatingCurve's full-fit likelihood / MLE / MAP / posterior
// numeric oracles come from the P4 dotnet emitter, NOT R1, so this file transcribes only the
// STRUCTURAL / CLOSED-FORM Predict / ALIGNMENT / DETERMINISM assertions (internal support test
// territory -- hardcoded oracles are correct here; public-API oracle values stay in fixtures/).
//
// Structural / closed-form oracles transcribed (values unaltered) from
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/RatingCurve/RatingCurveTests.cs @ fc28c0c
//
// The C# Predict oracles are exact closed-form expressions (no RNG); reproduced here to rel tol
// 1e-9 with the SAME literal parameter vectors and stages the C# uses. The alignment /
// likelihood-equality test (DataLogLikelihood_UsesOnlyCommonDates) checks two computations are
// equal, not a hardcoded value -- also in scope. Determinism tests use the ported bit-exact
// MersenneTwister (GenerateSyntheticData / GenerateRandomValues both draw from MersenneTwister in
// the C# source, so same-seed reproducibility holds).
//
// Adapted / skipped C# test methods (see task-R1-report.md for the full list + reasons):
//   - NumberOfSegments_Change_1To2_DoesNotThrowDuringPropertyChangeCascade: the PropertyChanged
//     cascade is INotifyPropertyChanged plumbing (project-wide non-port). Adapted to assert the
//     segment bump rebuilds to 7 params and validate() does not throw.
//   - SetParameterValues_Null_ThrowsArgumentNullException: VACUOUS (const-ref std::vector; no null).
//   - The seeded Predict(parameters, stage, seed) overload substitutes the ported MersenneTwister
//     for C#'s System.Random; its exact VALUES are a P4 concern. Only same/different-seed behavior
//     is exercised here.
//
// P4 landed (see test_rating_curve_p4_fixed_param_oracles below + fixtures/estimation/*):
//   - exact DataLogLikelihood + Residuals at fixed parameters, dumped from the real RMC.BestFit
//     via the dotnet emitter and asserted here (route b, C++-only, 1e-9 abs);
//   - the exact MLE fit (parameters, log-likelihood, AIC) and the seeded cross-language
//     GenerateRandomValues draw are oracle-verified against the real C# in rating_curve_smoke.json.
// Still deferred (severable follow-up): the seeded posterior (Bayesian) cross-language digest.
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "corehydro/models/rating_curve/rating_curve.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "check.hpp"

namespace {
// P6 index swap: the TimeSeries container indexes by DateTime rather than by an integer offset.
// Nothing in this suite reads the index for meaning (the ARMA families index by POSITION, the
// rating curve by date EQUALITY), so every series here is anchored at one arbitrary date; the
// mapping is monotone, so every oracle below is unchanged by the swap.
const corehydro::numerics::data::DateTime kT0(1900, 1, 1);


using corehydro::models::RatingCurve;
using corehydro::numerics::data::TimeInterval;
using corehydro::numerics::data::TimeSeries;

// Compile-time mirror of the C# `RatingCurve : ModelBase, ISimulatable<double[]>` declaration.
static_assert(std::is_base_of<corehydro::models::ModelBase, RatingCurve>::value,
              "RatingCurve must derive from ModelBase");
static_assert(
    std::is_base_of<corehydro::models::ISimulatable<std::vector<double>>, RatingCurve>::value,
    "RatingCurve must implement ISimulatable<std::vector<double>>");

// ---- Test-data helpers (mirror the C# private fixtures). ----

// Synthetic stage values (5-15 ft) -- C# s_stage.
const std::vector<double> kStage = {5.0, 6.5, 7.5, 9.0, 10.5, 12.0, 14.0};
// Synthetic discharge values (100-5000 cfs) -- C# s_discharge.
const std::vector<double> kDischarge = {110.0, 300.0, 600.0, 1200.0, 2000.0, 3500.0, 5100.0};

TimeSeries make_stage() { return TimeSeries(TimeInterval::OneDay, kT0, kStage); }
TimeSeries make_discharge() { return TimeSeries(TimeInterval::OneDay, kT0, kDischarge); }

// Builds a TimeSeries of consecutive daily ordinates starting `start` days after kT0 (mirrors the
// C# MakeSeries helper, which advances DateTime by +1 day per ordinate).
TimeSeries make_series(long start, const std::vector<double>& values) {
    return TimeSeries(TimeInterval::OneDay, kT0.add_days(static_cast<double>(start)), values);
}

bool messages_contain(const corehydro::models::ValidationResult& r, const std::string& needle) {
    for (const auto& m : r.validation_messages)
        if (m.find(needle) != std::string::npos) return true;
    return false;
}

// rel-tol wrapper: absolute tolerance scaled to the expected magnitude (brief's rel 1e-9 policy).
double reltol(double expected, double rel) { return std::fabs(expected) * rel + 1e-12; }

// ============================ Construction / parameter layout ============================

void test_construction_and_counts() {
    // Constructor_Default_Creates1SegmentModel + Has4Parameters.
    RatingCurve def;
    CHECK_EQ(def.number_of_segments(), 1);
    CHECK_TRUE(def.number_of_parameters() > 0);
    CHECK_EQ(static_cast<int>(def.parameters().size()), 4);

    // Constructor_2Segments_Has7Parameters.
    RatingCurve m2(make_stage(), make_discharge(), 2);
    CHECK_EQ(static_cast<int>(m2.parameters().size()), 7);

    // Constructor_3Segments_Has10Parameters.
    RatingCurve m3(make_stage(), make_discharge(), 3);
    CHECK_EQ(static_cast<int>(m3.parameters().size()), 10);

    // NumberOfSegments_Change_From1To2_RebuildParameters.
    RatingCurve mc(make_stage(), make_discharge(), 1);
    CHECK_EQ(static_cast<int>(mc.parameters().size()), 4);
    mc.set_number_of_segments(2);
    CHECK_EQ(static_cast<int>(mc.parameters().size()), 7);
}

void test_parameter_names() {
    // Parameters_FirstParam_IsZeroFlowStageH1.
    RatingCurve m1(make_stage(), make_discharge(), 1);
    CHECK_TRUE(m1.parameters()[0].name().find("Zero-Flow") != std::string::npos);
    // "h" + subscript-1 (U+2081, UTF-8 \xE2\x82\x81).
    CHECK_TRUE(m1.parameters()[0].name().find("h\xE2\x82\x81") != std::string::npos);

    // Parameters_LastParam_IsScaleSigma_AndPositive ("Scale (σ)"; σ = U+03C3 \xCF\x83).
    const auto& sigma = m1.parameters().back();
    CHECK_TRUE(sigma.name().find("\xCF\x83") != std::string::npos);
    CHECK_TRUE(sigma.is_positive());

    // Parameters_2Segment_ContainsActivationStageH2.
    RatingCurve m2(make_stage(), make_discharge(), 2);
    bool has_h2 = false;
    for (const auto& p : m2.parameters())
        if (p.name().find("Activation") != std::string::npos) has_h2 = true;
    CHECK_TRUE(has_h2);

    // Parameters_2Segment_Index4_IsCoefficientAlpha2 (index 4 is control-2 coefficient).
    CHECK_TRUE(m2.parameters()[4].name().find("Coefficient") != std::string::npos);
    CHECK_TRUE(m2.parameters()[4].name().find("Location") == std::string::npos);

    // Parameters_3Segment_Indices4And7_AreCoefficientsAlpha2Alpha3.
    RatingCurve m3(make_stage(), make_discharge(), 3);
    CHECK_TRUE(m3.parameters()[4].name().find("Coefficient") != std::string::npos);
    CHECK_TRUE(m3.parameters()[7].name().find("Coefficient") != std::string::npos);
}

// ============================ Addition-mode discharge composition ============================

void test_predict_composition() {
    // Predict_2Seg_AtBreakpoint_EqualsMainChannelAlone.
    {
        RatingCurve m(make_stage(), make_discharge(), 2);
        std::vector<double> parms = {0.2, 1.0, 2.5, 8.0, 2.0, 1.6, 0.05};
        m.set_parameter_values(parms);
        double h2 = parms[3];
        double q_main = std::pow(10, parms[1]) * std::pow(h2 - parms[0], parms[2]);
        double q_at_h2 = m.predict(parms, h2);
        CHECK_NEAR(q_at_h2, q_main, reltol(q_main, 1e-9));
    }
    // Predict_2Seg_AboveBreakpoint_SumsMainAndOverbank.
    {
        RatingCurve m(make_stage(), make_discharge(), 2);
        std::vector<double> parms = {0.2, 1.0, 2.5, 8.0, 2.0, 1.6, 0.05};
        m.set_parameter_values(parms);
        double h = 11.0;
        double q_main = std::pow(10, parms[1]) * std::pow(h - parms[0], parms[2]);
        double q_over = std::pow(10, parms[4]) * std::pow(h - parms[3], parms[5]);
        double expected = q_main + q_over;
        CHECK_NEAR(m.predict(parms, h), expected, reltol(expected, 1e-9));
    }
    // Predict_3Seg_BetweenBreakpoints_SumsFirstTwoControls.
    {
        RatingCurve m(make_stage(), make_discharge(), 3);
        std::vector<double> parms = {0.2, 1.0, 2.5, 6.0, 1.5, 2.0, 10.0, 2.0, 1.5, 0.05};
        m.set_parameter_values(parms);
        double h = 8.0;  // h2 < h < h3
        double q1 = std::pow(10, parms[1]) * std::pow(h - parms[0], parms[2]);
        double q2 = std::pow(10, parms[4]) * std::pow(h - parms[3], parms[5]);
        double expected = q1 + q2;
        CHECK_NEAR(m.predict(parms, h), expected, reltol(expected, 1e-9));
    }
    // Predict_3Seg_AboveLastBreakpoint_SumsAllThreeControls.
    {
        RatingCurve m(make_stage(), make_discharge(), 3);
        std::vector<double> parms = {0.2, 1.0, 2.5, 6.0, 1.5, 2.0, 10.0, 2.0, 1.5, 0.05};
        m.set_parameter_values(parms);
        double h = 13.0;  // above h3
        double q1 = std::pow(10, parms[1]) * std::pow(h - parms[0], parms[2]);
        double q2 = std::pow(10, parms[4]) * std::pow(h - parms[3], parms[5]);
        double q3 = std::pow(10, parms[7]) * std::pow(h - parms[6], parms[8]);
        double expected = q1 + q2 + q3;
        CHECK_NEAR(m.predict(parms, h), expected, reltol(expected, 1e-9));
    }
    // Predict_2Seg_BelowBreakpoint_IndependentOfSecondControl.
    {
        RatingCurve m(make_stage(), make_discharge(), 2);
        std::vector<double> parms1 = {0.2, 1.0, 2.5, 8.0, 2.0, 1.6, 0.05};
        std::vector<double> parms2 = {0.2, 1.0, 2.5, 8.0, 4.0, 1.2, 0.05};
        double stage = 5.0;  // below h2 = 8
        m.set_parameter_values(parms1);
        double q1 = m.predict(parms1, stage);
        m.set_parameter_values(parms2);
        double q2 = m.predict(parms2, stage);
        CHECK_NEAR(q1, q2, 1e-12);
    }
    // Predict_WithValidParameters_ReturnsPositiveDischarge + HigherStage_YieldsHigherDischarge.
    {
        RatingCurve m(make_stage(), make_discharge(), 1);
        std::vector<double> parms = {4.0, 0.5, 1.8, 0.1};
        m.set_parameter_values(parms);
        CHECK_TRUE(m.predict(parms, 8.0) > 0);
        CHECK_TRUE(m.predict(parms, 10.0) > m.predict(parms, 7.0));
    }
}

// ============================ GetLog10Alpha / GetLocation ============================

void test_indexers() {
    // GetLog10Alpha single/two/three-segment.
    {
        RatingCurve m;
        std::vector<double> parms = {0.5, 1.23, 2.0, 0.05};
        m.set_parameter_values(parms);
        CHECK_NEAR(m.get_log10_alpha(1, parms), 1.23, 1e-12);
    }
    {
        RatingCurve m(make_stage(), make_discharge(), 2);
        std::vector<double> parms = {0.2, 1.0, 2.5, 8.0, 2.15, 1.6, 0.05};
        m.set_parameter_values(parms);
        CHECK_NEAR(m.get_log10_alpha(2, parms), 2.15, 1e-12);
    }
    {
        RatingCurve m(make_stage(), make_discharge(), 3);
        std::vector<double> parms = {0.2, 1.0, 2.5, 6.0, 1.5, 2.0, 10.0, 2.4, 1.5, 0.05};
        m.set_parameter_values(parms);
        CHECK_NEAR(m.get_log10_alpha(3, parms), 2.4, 1e-12);
    }
    // GetLog10Alpha out-of-range throws (segment 0 and segment above count).
    {
        RatingCurve m;
        std::vector<double> parms = {0.5, 1.23, 2.0, 0.05};
        CHECK_THROWS(m.get_log10_alpha(0, parms));
        CHECK_THROWS(m.get_log10_alpha(2, parms));
    }
    // GetLocation single/two/three-segment (b_k = h_k under addition mode).
    {
        RatingCurve m;
        std::vector<double> parms = {0.5, 1.23, 2.0, 0.05};
        m.set_parameter_values(parms);
        CHECK_NEAR(m.get_location(1, parms), 0.5, 1e-12);
    }
    {
        RatingCurve m(make_stage(), make_discharge(), 2);
        std::vector<double> parms = {0.2, 1.0, 2.5, 8.0, 2.0, 1.6, 0.05};
        m.set_parameter_values(parms);
        CHECK_NEAR(m.get_location(2, parms), parms[3], 1e-12);
    }
    {
        RatingCurve m(make_stage(), make_discharge(), 3);
        std::vector<double> parms = {0.2, 1.0, 2.5, 6.0, 1.5, 2.0, 10.0, 2.0, 1.5, 0.05};
        m.set_parameter_values(parms);
        CHECK_NEAR(m.get_location(2, parms), parms[3], 1e-12);
        CHECK_NEAR(m.get_location(3, parms), parms[6], 1e-12);
    }
}

// ============================ Ordering constraints (Validate) ============================

void test_validate_ordering() {
    // Validate_2Segment_Xi1GreaterThanOrEqualH2_IsInvalid.
    {
        RatingCurve m(make_stage(), make_discharge(), 2);
        std::vector<double> parms = {9.0, 1.0, 2.5, 8.0, 2.0, 1.6, 0.05};  // h1 > h2
        m.set_parameter_values(parms);
        auto r = m.validate();
        CHECK_TRUE(!r.is_valid);
        CHECK_TRUE(messages_contain(r, "Segment ordering"));
    }
    // Validate_3Segment_H2GreaterThanOrEqualH3_IsInvalid.
    {
        RatingCurve m(make_stage(), make_discharge(), 3);
        std::vector<double> parms = {0.2, 1.0, 2.5, 10.0, 1.5, 2.0, 10.0, 2.0, 1.5, 0.05};
        m.set_parameter_values(parms);
        auto r = m.validate();
        CHECK_TRUE(!r.is_valid);
    }
    // NumberOfSegments_Change_1To2 (INPC cascade adapted): bump rebuilds to 7 params, no throw.
    {
        RatingCurve m(make_stage(), make_discharge(), 1);
        m.set_number_of_segments(2);
        bool threw = false;
        try {
            (void)m.validate();
        } catch (...) {
            threw = true;
        }
        CHECK_TRUE(!threw);
        CHECK_EQ(static_cast<int>(m.parameters().size()), 7);
    }
}

// ============================ SetParameterValues ============================

void test_set_parameter_values() {
    // SetParameterValues_StoresAllValues.
    {
        RatingCurve m(make_stage(), make_discharge(), 1);
        std::vector<double> vals = {4.0, 0.5, 1.8, 0.1};
        m.set_parameter_values(vals);
        for (std::size_t i = 0; i < vals.size(); ++i)
            CHECK_NEAR(m.parameters()[i].value(), vals[i], 1e-10);
    }
    // SetParameterValues_WrongCount_ThrowsArgumentException.
    {
        RatingCurve m(make_stage(), make_discharge(), 1);
        std::vector<double> bad = {1.0, 2.0};  // need 4
        CHECK_THROWS(m.set_parameter_values(bad));
    }
}

// ============================ UseJeffreysRuleForScale ============================

void test_jeffreys_flag() {
    RatingCurve def;
    CHECK_TRUE(def.use_jeffreys_rule_for_scale());  // default true

    RatingCurve m;
    m.set_use_jeffreys_rule_for_scale(false);
    CHECK_TRUE(!m.use_jeffreys_rule_for_scale());

    // Prior includes / excludes the Jeffreys 1/sigma term depending on the flag (finite, no NaN).
    RatingCurve mj(make_stage(), make_discharge(), 1);
    std::vector<double> parms = {0.2, 0.5, 1.8, 0.05};
    mj.set_parameter_values(parms);
    double with_j = mj.prior_log_likelihood(parms);
    mj.set_use_jeffreys_rule_for_scale(false);
    double no_j = mj.prior_log_likelihood(parms);
    CHECK_TRUE(!std::isnan(with_j) && !std::isnan(no_j));
    // Jeffreys subtracts log(sigma); with sigma < 1 that ADDS a positive term.
    CHECK_NEAR(with_j - no_j, -std::log(0.05), 1e-10);
}

// ============================ Date-alignment (inner join) ============================

void test_alignment() {
    // Validate_MismatchedCounts_WithEnoughCommonDates_IsValid (120 stage / 100 discharge, 100 shared).
    {
        std::vector<double> stage_vals, discharge_vals;
        for (int i = 0; i < 120; ++i) stage_vals.push_back(1.0 + 0.05 * i);
        for (int i = 0; i < 100; ++i) discharge_vals.push_back(10.0 + 2.0 * i);
        RatingCurve m(make_series(0, stage_vals), make_series(0, discharge_vals), 1);
        auto r = m.validate();
        CHECK_TRUE(r.is_valid);
        CHECK_TRUE(!messages_contain(r, "common dates"));
    }
    // Validate_FewerThanTenCommonDates_IsInvalid (zero overlap).
    {
        std::vector<double> stage_vals, discharge_vals;
        for (int i = 0; i < 50; ++i) stage_vals.push_back(1.0 + 0.1 * i);
        for (int i = 0; i < 50; ++i) discharge_vals.push_back(10.0 + 2.0 * i);
        // stage indices 0..49; discharge indices 5000.. (no overlap).
        RatingCurve m(make_series(0, stage_vals), make_series(5000, discharge_vals), 1);
        auto r = m.validate();
        CHECK_TRUE(!r.is_valid);
        CHECK_TRUE(messages_contain(r, "common dates"));
    }
    // Validate_IdenticalSeries_StillValid.
    {
        std::vector<double> stage_vals, discharge_vals;
        for (int i = 0; i < 20; ++i) {
            double h = 1.0 + 0.1 * i;
            stage_vals.push_back(h);
            discharge_vals.push_back(2.0 * std::pow(h - 0.2, 1.8));
        }
        RatingCurve m(make_series(0, stage_vals), make_series(0, discharge_vals), 1);
        CHECK_TRUE(m.validate().is_valid);
    }
    // DataLogLikelihood_UsesOnlyCommonDates: extra unmatched stage obs must be dropped.
    {
        std::vector<double> stage_vals, discharge_vals;
        for (int i = 0; i < 30; ++i) {
            double h = 1.0 + 0.1 * i;
            stage_vals.push_back(h);
            discharge_vals.push_back(2.0 * std::pow(h - 0.2, 1.8));
        }
        TimeSeries stage_aligned = make_series(0, stage_vals);
        TimeSeries discharge_aligned = make_series(0, discharge_vals);
        RatingCurve model_aligned(stage_aligned, discharge_aligned, 1);
        std::vector<double> parms = {0.2, std::log10(2.0), 1.8, 0.05};
        model_aligned.set_parameter_values(parms);
        double aligned_ll = model_aligned.data_log_likelihood(parms);

        // Extended stage: original 30 (indices 0..29) + 15 unmatched (large indices).
        TimeSeries stage_extended(TimeInterval::OneDay);
        for (int i = 0; i < 30; ++i)
            stage_extended.add(TimeSeries::Ordinate(kT0.add_days(i),
                                                    stage_vals[static_cast<std::size_t>(i)]));
        for (int i = 30; i < 45; ++i)
            stage_extended.add(
                TimeSeries::Ordinate(kT0.add_days(5.0 * 365 + (i - 30)), 5.0 + 0.1 * (i - 30)));

        RatingCurve model_extended(stage_extended, discharge_aligned, 1);
        model_extended.set_use_default_flat_priors(false);
        model_extended.set_parameter_values(parms);
        double extended_ll = model_extended.data_log_likelihood(parms);

        CHECK_NEAR(aligned_ll, extended_ll, 1e-10);
        CHECK_TRUE(std::isfinite(aligned_ll));
    }
    // Residuals_ReturnsOneValuePerCommonDate (40 stage / 25 discharge -> 25 residuals).
    {
        std::vector<double> stage_vals, discharge_vals;
        for (int i = 0; i < 40; ++i) stage_vals.push_back(1.0 + 0.1 * i);
        for (int i = 0; i < 25; ++i)
            discharge_vals.push_back(2.0 * std::pow(stage_vals[static_cast<std::size_t>(i)] - 0.2, 1.8));
        RatingCurve m(make_series(0, stage_vals), make_series(0, discharge_vals), 1);
        m.set_use_default_flat_priors(false);
        std::vector<double> parms = {0.2, std::log10(2.0), 1.8, 0.05};
        m.set_parameter_values(parms);
        CHECK_EQ(static_cast<int>(m.residuals(parms).size()), 25);
    }
    // Pointwise data-likelihood length == aligned count; components too.
    {
        std::vector<double> stage_vals, discharge_vals;
        for (int i = 0; i < 15; ++i) {
            double h = 1.0 + 0.1 * i;
            stage_vals.push_back(h);
            discharge_vals.push_back(2.0 * std::pow(h - 0.2, 1.8));
        }
        RatingCurve m(make_series(0, stage_vals), make_series(0, discharge_vals), 1);
        std::vector<double> parms = {0.2, std::log10(2.0), 1.8, 0.05};
        m.set_parameter_values(parms);
        CHECK_EQ(static_cast<int>(m.pointwise_data_log_likelihood(parms).size()), 15);
        CHECK_EQ(static_cast<int>(m.pointwise_data_log_likelihood_components(parms).size()), 15);
        // Scalar sum equals sum of pointwise entries.
        double scalar = m.data_log_likelihood(parms);
        double sum = 0.0;
        for (double v : m.pointwise_data_log_likelihood(parms)) sum += v;
        CHECK_NEAR(scalar, sum, 1e-9);
    }
}

// ============================ GenerateSyntheticData / GenerateRandomValues ============================

void test_generation() {
    // GenerateSyntheticData_ReturnsTwoSeriesOfRequestedSize.
    {
        RatingCurve m;
        std::vector<double> parms = {0.5, 0.3, 1.8, 0.05};
        m.set_parameter_values(parms);
        auto syn = m.generate_synthetic_data(20, 1.0, 5.0, 42);
        CHECK_EQ(syn.stage_data.count(), 20);
        CHECK_EQ(syn.discharge_data.count(), 20);
    }
    // GenerateSyntheticData_AllDischargesPositive.
    {
        RatingCurve m;
        std::vector<double> parms = {0.5, 0.3, 1.8, 0.05};
        m.set_parameter_values(parms);
        auto syn = m.generate_synthetic_data(50, 1.0, 5.0, 123);
        bool all_pos = true;
        for (int i = 0; i < syn.discharge_data.count(); ++i)
            if (syn.discharge_data[i].value() <= 0) all_pos = false;
        CHECK_TRUE(all_pos);
    }
    // GenerateSyntheticData_ZeroSampleSize_Throws.
    {
        RatingCurve m;
        std::vector<double> parms = {0.5, 0.3, 1.8, 0.05};
        m.set_parameter_values(parms);
        CHECK_THROWS(m.generate_synthetic_data(0, 1.0, 5.0));
    }
    // GenerateSyntheticData_SameSeed_ProducesSameResult.
    {
        RatingCurve m;
        std::vector<double> parms = {0.5, 0.3, 1.8, 0.05};
        m.set_parameter_values(parms);
        auto s1 = m.generate_synthetic_data(10, 1.0, 5.0, 999);
        auto s2 = m.generate_synthetic_data(10, 1.0, 5.0, 999);
        bool same = true;
        for (int i = 0; i < s1.discharge_data.count(); ++i)
            if (std::fabs(s1.discharge_data[i].value() - s2.discharge_data[i].value()) > 1e-10)
                same = false;
        CHECK_TRUE(same);
    }
    // GenerateRandomValues_ReturnsSizedArrayWithPositiveValues.
    {
        RatingCurve m(make_stage(), make_discharge(), 1);
        std::vector<double> parms = {4.0, 0.5, 1.8, 0.1};
        m.set_parameter_values(parms);
        auto samples = m.generate_random_values(30, 42);
        CHECK_EQ(static_cast<int>(samples.size()), 30);
        bool all_pos = true;
        for (double v : samples)
            if (v <= 0) all_pos = false;
        CHECK_TRUE(all_pos);
    }
    // GenerateRandomValues_SameSeed_Deterministic.
    {
        RatingCurve m(make_stage(), make_discharge(), 1);
        std::vector<double> parms = {4.0, 0.5, 1.8, 0.1};
        m.set_parameter_values(parms);
        auto run1 = m.generate_random_values(20, 7);
        auto run2 = m.generate_random_values(20, 7);
        bool same = run1.size() == run2.size();
        for (std::size_t i = 0; i < run1.size(); ++i)
            if (run1[i] != run2[i]) same = false;
        CHECK_TRUE(same);
    }
    // GenerateRandomValues_ZeroSampleSize_Throws.
    {
        RatingCurve m(make_stage(), make_discharge(), 1);
        std::vector<double> parms = {4.0, 0.5, 1.8, 0.1};
        m.set_parameter_values(parms);
        CHECK_THROWS(m.generate_random_values(0));
    }
    // Seeded Predict overload: same seed reproduces, different seed diverges (MersenneTwister
    // substituted for C#'s System.Random; exact values are a P4 concern).
    {
        RatingCurve m(make_stage(), make_discharge(), 1);
        std::vector<double> parms = {4.0, 0.5, 1.8, 0.1};
        m.set_parameter_values(parms);
        double a = m.predict(parms, 8.0, 11);
        double b = m.predict(parms, 8.0, 11);
        CHECK_NEAR(a, b, 1e-12);
        double c = m.predict(parms, 8.0, 99);
        CHECK_TRUE(a != c);
        CHECK_TRUE(a > 0 && c > 0);
    }
}

// P4 oracles (route b): exact fixed-parameter DataLogLikelihood + Residuals dumped from the REAL
// RMC.BestFit RatingCurve via tools/oracle_emitter (Numerics @ a2c4dbf, RMC-BestFit @ fc28c0c) on
// the shared single-segment stage/discharge fixture. Deterministic (no fit) -> hardcoded C++-only
// oracles at 1e-9 abs. The exact MLE fit + seeded draw are oracle-verified cross-language in
// fixtures/estimation/rating_curve_smoke.json.
void test_rating_curve_p4_fixed_param_oracles() {
    std::vector<double> stage{1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.6, 2.9, 3.1, 3.4,
                              3.7, 4.0, 4.3, 4.6, 5.0};
    std::vector<double> discharge{5.0,  7.1,  11.0, 16.2, 20.5,  28.0,  37.1, 47.9,
                                  55.6, 68.0, 82.3, 98.1, 115.4, 134.8, 160.2};
    RatingCurve m(make_series(0, stage), make_series(0, discharge), 1);
    // params [h0, log10(a), b, sigma] = [0.0, 0.45, 2.44, 0.02].
    std::vector<double> p{0.0, 0.45, 2.44, 0.02};
    m.set_parameter_values(p);
    CHECK_NEAR(m.data_log_likelihood(p), -255.24744448940857, 1e-9);
    std::vector<double> res = m.residuals(p);
    CHECK_NEAR(res[0], 0.24897000433601885, 1e-9);
    CHECK_NEAR(res[7], 0.1020843985411104, 1e-9);
    CHECK_NEAR(res[14], 0.04917570116833314, 1e-9);
}

}  // namespace

int main() {
    test_construction_and_counts();
    test_parameter_names();
    test_predict_composition();
    test_indexers();
    test_validate_ordering();
    test_set_parameter_values();
    test_jeffreys_flag();
    test_alignment();
    test_generation();
    test_rating_curve_p4_fixed_param_oracles();

    return chtest::summary("rating_curve");
}
