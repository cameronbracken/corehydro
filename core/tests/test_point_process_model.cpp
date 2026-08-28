// Standalone test for corehydro::models::PointProcessModel (Phase 5, M12).
//
// Oracle for behavior is the C# source itself:
//   - upstream/RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/
//     PointProcessModel.cs @ c2e6192, and its base UnivariateDistributionModelBase.cs
//     @ fc28c0c;
//   - a transcription of the upstream test class
//     RMC.BestFit.Tests/Univariate/PointProcessModelTests.cs @ fc28c0c. Skipped methods
//     (full list with reasons in the M12 report):
//       XML surface (7): Test_ToXElement_ContainsPointProcessModelElement /
//         _ContainsThresholdAttribute / _ContainsTotalYearsAttribute /
//         _ContainsIsSeasonalAttribute / _ContainsDistributionElement,
//         Test_RoundTrip_PreservesAllProperties, Test_RoundTrip_PreservesSeasonalSettings
//         (ToXElement / the XElement ctor are a project-wide non-port);
//       Test_SetParameterValues_NullParameters_ThrowsException (a std::vector<double>
//         argument cannot be null -- type-system guarantee);
// The five seasonal DATA-path tests skipped through P5 -- Test_SetAMSData_Seasonal_CreatesPOTDays,
// Test_Seasonal_DataLogLikelihood_ReturnsFinite, Test_JeffreysPrior_Seasonal_AppliedToBothGEVs,
// Test_PointwisePriorLogLikelihood_Sum_Equals_PriorLogLikelihood_Seasonal and
// Test_PointProcess_SeasonalFloodAnalysis -- are TRANSCRIBED as of P6, which ported the TimeSeries
// container that path needs. They are at the end of this file, under "P6: the seasonal data path".
// Hardcoded oracles are allowed here (internal support layer); public-API oracle values stay
// in fixtures/ only (the fixture wiring for the Models layer arrives in M13/M14).
//
// Test-surface adaptations forced by the port (each noted at the test):
//   - The non-seasonal C# helpers construct ExactData from a bare index and this file does the
//     same. The SEASONAL helper builds from a DateTime, as C# does, now that ExactData carries
//     one (P6); the C# `ExactData(DateTime, value)` ctor is `base(dateTime.Year, value, 0)`, so
//     the index is still the year either way.
//   - C# `Assert.AreSame(df, model.DataFrame)` maps to has_data_frame() + content checks
//     (the move-only C++ frame cannot alias).
//   - C# `Assert.AreNotSame(a, b)` maps to a pointer-identity check.
//   - C# `Assert.IsInstanceOfType(x, typeof(T))` maps to a type()-enum check.
//   - Test_SetDefaultThresholdAndTotalYears_TotalYearsEventSeesUpdatedLambda subscribes to
//     the (unported) INPC PropertyChanged event to observe Lambda at TotalYears-notification
//     time; the C++ transcription checks Lambda immediately after `UseDefaults = true`
//     returns -- the same lambda-refresh-before-notification ordering rule, observed at the
//     only point the eventless port exposes.
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/models/univariate_distribution/point_process_model.hpp"
#include "corehydro/numerics/data/time_series/support/time_block_window.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/competing_risks.hpp"
#include "corehydro/numerics/distributions/generalized_extreme_value.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "check.hpp"

using corehydro::models::DataFrame;
using corehydro::models::ExactData;
using corehydro::models::ExactSeries;
using corehydro::models::IntervalData;
using corehydro::models::PointProcessModel;
using corehydro::models::PriorComponentType;
using corehydro::models::ThresholdData;
using corehydro::models::UncertainData;
using corehydro::models::ValidationResult;
using corehydro::numerics::data::TimeBlockWindow;
using corehydro::numerics::distributions::CompetingRisks;
using corehydro::numerics::distributions::GeneralizedExtremeValue;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::UnivariateDistributionBase;
using corehydro::numerics::distributions::UnivariateDistributionType;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ===========================================================================================
// Test data helpers (PointProcessModelTests.cs, "Test Data Helper" region). The C# builds
// ExactData from DateTime; ExactData(DateTime, value) is base(dateTime.Year, value, 0), so
// the year is the index (adaptation note in the file header).
// ===========================================================================================

// Creates a sample POT data frame with exact observations.
DataFrame create_pot_data_frame() {
    DataFrame df;
    std::vector<ExactData> data{
        ExactData(1990, 1500),  // 1990-03-15
        ExactData(1990, 1800),  // 1990-05-20
        ExactData(1991, 2000),  // 1991-04-10
        ExactData(1992, 1700),  // 1992-03-25
        ExactData(1992, 2200),  // 1992-06-05
        ExactData(1993, 2500),  // 1993-04-01
        ExactData(1994, 1600),  // 1994-05-15
        ExactData(1995, 3000),  // 1995-03-30
        ExactData(1996, 2100),  // 1996-04-20
        ExactData(1997, 2800)   // 1997-05-10
    };
    df.set_exact_series(ExactSeries(data));
    return df;
}

// Creates a seasonal POT data frame with events from different seasons. Since P6 this builds its
// ordinates from DATES, exactly as the C# helper does -- which is what the seasonal data path
// reads to compute each event's day of the year.
DataFrame create_seasonal_pot_data_frame() {
    DataFrame df;
    std::vector<ExactData> data;

    // Winter/Spring events (season 1).
    for (int year = 1990; year < 2000; ++year)
        data.emplace_back(corehydro::numerics::data::DateTime(year, 2, 15),
                          1500 + (year - 1990) * 100);

    // Summer events (season 2).
    for (int year = 1990; year < 2000; ++year)
        data.emplace_back(corehydro::numerics::data::DateTime(year, 7, 15),
                          2000 + (year - 1990) * 150);

    df.set_exact_series(ExactSeries(data));
    return df;
}

// Creates a simple annual maximum data frame.
DataFrame create_ams_data_frame() {
    DataFrame df;
    std::vector<ExactData> data;
    for (int i = 0; i < 20; ++i) data.emplace_back(1980 + i, 2000 + i * 100);
    df.set_exact_series(ExactSeries(data));
    return df;
}

std::vector<double> parameter_values(const PointProcessModel& model) {
    std::vector<double> p;
    p.reserve(model.parameters().size());
    for (const auto& mp : model.parameters()) p.push_back(mp.value());
    return p;
}

bool any_message_contains(const ValidationResult& result, const std::string& needle) {
    for (const std::string& m : result.validation_messages)
        if (m.find(needle) != std::string::npos) return true;
    return false;
}

std::unique_ptr<CompetingRisks> make_gev_competing_risks(double xi, double alpha,
                                                         double kappa) {
    std::vector<std::unique_ptr<UnivariateDistributionBase>> comps;
    comps.push_back(std::make_unique<GeneralizedExtremeValue>(xi, alpha, kappa));
    return std::make_unique<CompetingRisks>(std::move(comps));
}

// ===========================================================================================
// Constructor tests.
// ===========================================================================================

// Test_Constructor_EmptyConstructor_CreatesDefaultModel
void test_constructor_empty_constructor_creates_default_model() {
    PointProcessModel model;

    CHECK_TRUE(model.has_distribution());
    CHECK_TRUE(!model.is_seasonal());
}

// Test_Constructor_EmptyConstructor_HasSingleGEV (IsInstanceOfType -> type enum).
void test_constructor_empty_constructor_has_single_gev() {
    PointProcessModel model;

    CHECK_EQ(model.distribution()->component_count(), 1);
    CHECK_TRUE(model.distribution()->component(0).type() ==
               UnivariateDistributionType::GeneralizedExtremeValue);
}

// Test_Constructor_WithDataAndDistribution_SetsProperties (AreSame -> content check).
void test_constructor_with_data_and_distribution_sets_properties() {
    DataFrame df = create_pot_data_frame();
    auto dist = make_gev_competing_risks(2000, 500, 0.1);

    PointProcessModel model(std::move(df), *dist);

    CHECK_TRUE(model.has_data_frame());
    CHECK_EQ(model.data_frame().exact_series().count(), static_cast<std::size_t>(10));
    CHECK_TRUE(model.has_distribution());
}

// Test_Constructor_WithDataAndDistribution_ClonesDistribution (AreNotSame -> pointer check).
void test_constructor_with_data_and_distribution_clones_distribution() {
    DataFrame df = create_pot_data_frame();
    auto dist = make_gev_competing_risks(2000, 500, 0.1);

    PointProcessModel model(std::move(df), *dist);

    CHECK_TRUE(model.distribution() != dist.get());
}

// ===========================================================================================
// Property tests.
// ===========================================================================================

// Test_Distribution_SetAndGet
void test_distribution_set_and_get() {
    PointProcessModel model;
    auto new_dist = make_gev_competing_risks(1000, 200, 0.0);

    model.set_distribution(std::move(new_dist));

    CHECK_TRUE(model.has_distribution());
}

// Test_Threshold_SetAndGet
void test_threshold_set_and_get() {
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    model.set_threshold(1000);

    CHECK_NEAR(model.threshold(), 1000.0, 0.0);
}

// Test_TotalYears_SetAndGet
void test_total_years_set_and_get() {
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    model.set_total_years(25);

    CHECK_NEAR(model.total_years(), 25.0, 0.0);
}

// Test_TotalYears_UpdatesLambda
void test_total_years_updates_lambda() {
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    model.set_total_years(10);

    // Lambda = events / years = 10 events / 10 years = 1.0
    CHECK_NEAR(model.lambda(), 1.0, 1e-10);
}

// Test_UseDefaults_SetsThresholdAndYears
void test_use_defaults_sets_threshold_and_years() {
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_use_defaults(false);
    model.set_data_frame(std::move(df));

    model.set_use_defaults(true);

    CHECK_TRUE(!std::isnan(model.threshold()));
    CHECK_TRUE(!std::isnan(model.total_years()));
}

// Test_IsSeasonal_SetAndGet
void test_is_seasonal_set_and_get() {
    PointProcessModel model;

    model.set_is_seasonal(true);

    CHECK_TRUE(model.is_seasonal());
}

// Test_IsSeasonal_True_CreatesTwoGEVs
void test_is_seasonal_true_creates_two_gevs() {
    PointProcessModel model;

    model.set_is_seasonal(true);

    CHECK_EQ(model.distribution()->component_count(), 2);
}

// Test_IsSeasonal_False_CreatesSingleGEV
void test_is_seasonal_false_creates_single_gev() {
    PointProcessModel model;
    model.set_is_seasonal(true);   // First set to true
    model.set_is_seasonal(false);  // Then back to false

    CHECK_EQ(model.distribution()->component_count(), 1);
}

// Test_TimeBlock_SetAndGet
void test_time_block_set_and_get() {
    PointProcessModel model;

    model.set_time_block(TimeBlockWindow::CalendarYear);

    CHECK_TRUE(model.time_block() == TimeBlockWindow::CalendarYear);
}

// Test_StartMonth_SetAndGet
void test_start_month_set_and_get() {
    PointProcessModel model;

    model.set_start_month(1);  // January

    CHECK_EQ(model.start_month(), 1);
}

// Test_Lambda_CalculatedFromEventsAndYears
void test_lambda_calculated_from_events_and_years() {
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_total_years(8);  // 8 years for 10 events

    double expected = 10.0 / 8.0;
    CHECK_NEAR(model.lambda(), expected, 1e-10);
}

// ===========================================================================================
// SetDistribution tests.
// ===========================================================================================

// Test_SetDistribution_NonSeasonal_SingleGEV
void test_set_distribution_non_seasonal_single_gev() {
    PointProcessModel model;
    model.set_is_seasonal(false);

    CHECK_EQ(model.distribution()->component_count(), 1);
    CHECK_TRUE(model.distribution()->component(0).type() ==
               UnivariateDistributionType::GeneralizedExtremeValue);
}

// Test_SetDistribution_Seasonal_TwoGEVs
void test_set_distribution_seasonal_two_gevs() {
    PointProcessModel model;
    model.set_is_seasonal(true);

    CHECK_EQ(model.distribution()->component_count(), 2);
    CHECK_TRUE(model.distribution()->component(0).type() ==
               UnivariateDistributionType::GeneralizedExtremeValue);
    CHECK_TRUE(model.distribution()->component(1).type() ==
               UnivariateDistributionType::GeneralizedExtremeValue);
}

// ===========================================================================================
// SetDefaultParameters tests.
// ===========================================================================================

// Test_SetDefaultParameters_NonSeasonal_ThreeGEVParameters
void test_set_default_parameters_non_seasonal_three_gev_parameters() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    // Non-seasonal: 3 GEV parameters (xi, alpha, kappa)
    CHECK_EQ(model.number_of_parameters(), 3);
}

// Test_SetDefaultParameters_Seasonal_EightParameters. NOTE: with the deferred seasonal
// block-maxima path the GEV parameter VALUES are NaN (the C# derives them from the seasonal
// AMS data), but the parameter COUNT asserted here is data-independent.
void test_set_default_parameters_seasonal_eight_parameters() {
    DataFrame df = create_seasonal_pot_data_frame();
    PointProcessModel model;
    model.set_is_seasonal(true);
    model.set_data_frame(std::move(df));

    // Seasonal: 2 change points + 3 GEV params x 2 seasons = 8
    CHECK_EQ(model.number_of_parameters(), 8);
}

// Test_SetDefaultParameters_Seasonal_HasChangePoints
void test_set_default_parameters_seasonal_has_change_points() {
    DataFrame df = create_seasonal_pot_data_frame();
    PointProcessModel model;
    model.set_is_seasonal(true);
    model.set_data_frame(std::move(df));

    CHECK_TRUE(model.parameters()[0].name().find("Change Point") != std::string::npos);
    CHECK_TRUE(model.parameters()[1].name().find("Change Point") != std::string::npos);
}

// Test_SetDefaultParameters_HasUniformPriors (IsInstanceOfType -> type enum).
void test_set_default_parameters_has_uniform_priors() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    for (const auto& param : model.parameters()) {
        CHECK_TRUE(param.prior_distribution().type() == UnivariateDistributionType::Uniform);
    }
}

// ===========================================================================================
// SetDefaultThresholdAndTotalYears tests.
// ===========================================================================================

// Test_SetDefaultThresholdAndTotalYears_SetsThreshold (AreNotEqual(NaN baseline, x) ->
// !isnan; see the C# test's own remark on the BitDecrement rounding at this scale).
void test_set_default_threshold_and_total_years_sets_threshold() {
    DataFrame df = create_pot_data_frame();
    double min_value = df.exact_series().minimum_value();
    PointProcessModel model;
    model.set_use_defaults(false);
    model.set_data_frame(std::move(df));
    // Baseline threshold before the method call (NaN on this path).
    double baseline_threshold = model.threshold();
    CHECK_TRUE(std::isnan(baseline_threshold));

    model.set_default_threshold_and_total_years();

    CHECK_TRUE(!std::isnan(model.threshold()));
    CHECK_NEAR(model.threshold(), min_value, 1e-6);
}

// Test_SetDefaultThresholdAndTotalYears_SetsTotalYears
void test_set_default_threshold_and_total_years_sets_total_years() {
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_use_defaults(false);
    model.set_data_frame(std::move(df));

    model.set_default_threshold_and_total_years();

    CHECK_TRUE(model.total_years() > 0);
}

// Test_UseDefaults_RecomputesTotalYearsAfterManualOverride
void test_use_defaults_recomputes_total_years_after_manual_override() {
    DataFrame df = create_pot_data_frame();
    double index_span = static_cast<double>(df.exact_series().index_span());
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_use_defaults(false);
    model.set_total_years(123.0);

    model.set_use_defaults(true);

    CHECK_NEAR(model.total_years(), index_span, 1e-10);
}

// Test_SetDefaultThresholdAndTotalYears_TotalYearsEventSeesUpdatedLambda (INPC adaptation;
// see the file header: Lambda is checked right after UseDefaults = true returns, pinning
// the same CalculateLambda-before-notification ordering).
void test_set_default_threshold_and_total_years_total_years_event_sees_updated_lambda() {
    DataFrame df = create_pot_data_frame();
    double count = static_cast<double>(df.exact_series().count());
    double index_span = static_cast<double>(df.exact_series().index_span());
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_use_defaults(false);
    model.set_total_years(123.0);

    model.set_use_defaults(true);

    CHECK_NEAR(model.lambda(), count / index_span, 1e-10);
}

// Test_SetDefaultThresholdAndTotalYears_UsesPeaksOverThresholdInput
void test_set_default_threshold_and_total_years_uses_peaks_over_threshold_input() {
    DataFrame df = create_pot_data_frame();
    double pot_threshold = df.exact_series().minimum_value() - 100.0;
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    model.set_default_threshold_and_total_years(pot_threshold, true);

    CHECK_NEAR(model.threshold(), pot_threshold, 1e-10);
}

// Test_SetDefaultThresholdAndTotalYears_ClampsPeaksOverThresholdInput
void test_set_default_threshold_and_total_years_clamps_peaks_over_threshold_input() {
    DataFrame df = create_pot_data_frame();
    double exact_minimum = df.exact_series().minimum_value();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    model.set_default_threshold_and_total_years(exact_minimum + 100.0, true);

    CHECK_TRUE(model.threshold() < exact_minimum);
}

// Test_SetDefaultThresholdAndTotalYears_TotalYearsUsesExactSpanOnly
void test_set_default_threshold_and_total_years_total_years_uses_exact_span_only() {
    DataFrame df = create_pot_data_frame();
    double expected_span = static_cast<double>(df.exact_series().index_span());
    ThresholdData threshold_data(1000, 2000, 500.0);
    threshold_data.set_number_above(3);
    df.threshold_series().add(std::move(threshold_data));
    df.uncertain_series().add(UncertainData(500, std::make_unique<Normal>(1000.0, 100.0)));
    df.interval_series().add(IntervalData(2500, 900.0, 1000.0, 1100.0));
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    model.set_default_threshold_and_total_years(std::nullopt, true);

    CHECK_NEAR(model.total_years(), expected_span, 1e-10);
}

// ===========================================================================================
// CalculateLambda tests.
// ===========================================================================================

// Test_CalculateLambda_ReturnsEventsPerYear
void test_calculate_lambda_returns_events_per_year() {
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_total_years(10);

    model.calculate_lambda();

    // 10 events / 10 years = 1.0
    CHECK_NEAR(model.lambda(), 1.0, 1e-10);
}

// Test_CalculateLambda_NullDataFrame_ReturnsNaN
void test_calculate_lambda_null_data_frame_returns_nan() {
    PointProcessModel model;
    model.calculate_lambda();

    CHECK_TRUE(std::isnan(model.lambda()));
}

// Test_CalculateLambda_ZeroYears_ReturnsNaN
void test_calculate_lambda_zero_years_returns_nan() {
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_use_defaults(false);
    model.set_data_frame(std::move(df));
    model.set_total_years(0);

    model.calculate_lambda();

    CHECK_TRUE(std::isnan(model.lambda()));
}

// ===========================================================================================
// SetAMSData tests.
// ===========================================================================================

// Test_SetAMSData_NonSeasonal_CreatesBlockMaxima
void test_set_ams_data_non_seasonal_creates_block_maxima() {
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_is_seasonal(false);
    model.set_data_frame(std::move(df));

    CHECK_TRUE(model.ams_data_frame().exact_series().count() > 0);
}

// Test_SetAMSData_Seasonal_CreatesPOTDays: SKIPPED -- the seasonal POT day-of-year
// population is part of the deferred TimeSeries/DateTime path (file header).

// EXTRA (not in the C# suite; M10/M11 precedent): pins the non-seasonal block-maxima
// content and its observable ordering. The C# GroupBy(Index).ToDictionary enumerates in
// first-appearance order of the index (GroupBy is documented ordered; Dictionary without
// removals enumerates in insertion order), so the port reproduces exactly that order.
void test_set_ams_data_block_maxima_values_and_order() {
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    const auto& ams = model.ams_data_frame().exact_series();
    CHECK_EQ(ams.count(), static_cast<std::size_t>(8));

    const int expected_indices[8] = {1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997};
    const double expected_maxima[8] = {1800, 2000, 2200, 2500, 1600, 3000, 2100, 2800};
    for (std::size_t i = 0; i < 8; ++i) {
        CHECK_EQ(ams[i].index(), expected_indices[i]);
        CHECK_NEAR(ams[i].value(), expected_maxima[i], 0.0);
    }
}

// ===========================================================================================
// LogLikelihood tests.
// ===========================================================================================

// Test_DataLogLikelihood_ReturnsFiniteValue
void test_data_log_likelihood_returns_finite_value() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    std::vector<double> parameters = parameter_values(model);
    double data_log_lh = model.data_log_likelihood(parameters);

    CHECK_TRUE(!std::isnan(data_log_lh));
    CHECK_TRUE(data_log_lh != kInf);
}

// Test_PriorLogLikelihood_ReturnsFiniteValue
void test_prior_log_likelihood_returns_finite_value() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    std::vector<double> parameters = parameter_values(model);
    double prior_log_lh = model.prior_log_likelihood(parameters);

    CHECK_TRUE(!std::isnan(prior_log_lh));
    CHECK_TRUE(prior_log_lh != kInf);
}

// Test_DataLogLikelihood_NullDataFrame_ReturnsNegativeInfinity
void test_data_log_likelihood_null_data_frame_returns_negative_infinity() {
    PointProcessModel model;

    std::vector<double> params{1, 2, 3};  // mutable lvalue (M14 signature)
    double result = model.data_log_likelihood(params);

    CHECK_TRUE(result == -kInf);
}

// Test_DataLogLikelihood_NaNThreshold_ReturnsNegativeInfinity
void test_data_log_likelihood_nan_threshold_returns_negative_infinity() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_use_defaults(false);
    model.set_data_frame(std::move(df));
    model.set_threshold(kNaN);

    std::vector<double> parameters = parameter_values(model);
    double result = model.data_log_likelihood(parameters);

    CHECK_TRUE(result == -kInf);
}

// Test_PointwiseDataLogLikelihood_ReturnsCorrectCount
void test_pointwise_data_log_likelihood_returns_correct_count() {
    DataFrame df = create_ams_data_frame();
    std::size_t exact_count = df.exact_series().count();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    std::vector<double> parameters = parameter_values(model);
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(parameters);

    CHECK_EQ(pointwise.size(), exact_count);
}

// EXTRA (not in the C# suite): the pointwise sum invariant the C# remarks document
// (sum(pointwise) == DataLogLikelihood, rate term distributed across exact observations).
void test_pointwise_data_log_likelihood_sum_equals_data_log_likelihood() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    std::vector<double> parameters = parameter_values(model);
    double scalar = model.data_log_likelihood(parameters);
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(parameters);
    double sum = 0.0;
    for (double v : pointwise) sum += v;

    CHECK_NEAR(sum, scalar, 1e-9);

    // The components wrapper mirrors the pointwise values one-to-one.
    auto components = model.pointwise_data_log_likelihood_components(parameters);
    CHECK_EQ(components.size(), pointwise.size());
    double component_sum = 0.0;
    for (const auto& c : components) component_sum += c.log_likelihood();
    CHECK_NEAR(component_sum, scalar, 1e-9);
}

// ===========================================================================================
// Seasonal model tests (parameter surface only; the seasonal DATA path is deferred).
// ===========================================================================================

// Test_Seasonal_ChangePointsHaveCorrectBounds
void test_seasonal_change_points_have_correct_bounds() {
    DataFrame df = create_seasonal_pot_data_frame();
    PointProcessModel model;
    model.set_is_seasonal(true);
    model.set_data_frame(std::move(df));

    // K1 bounds
    CHECK_NEAR(model.parameters()[0].lower_bound(), 10.0, 0.0);
    CHECK_NEAR(model.parameters()[0].upper_bound(), 170.0, 0.0);

    // K2 bounds
    CHECK_NEAR(model.parameters()[1].lower_bound(), 171.0, 0.0);
    CHECK_NEAR(model.parameters()[1].upper_bound(), 330.0, 0.0);
}

// Test_Seasonal_ChangePointDefaultValues
void test_seasonal_change_point_default_values() {
    DataFrame df = create_seasonal_pot_data_frame();
    PointProcessModel model;
    model.set_is_seasonal(true);
    model.set_data_frame(std::move(df));

    // Default K1 = 90 (around April 1)
    CHECK_NEAR(model.parameters()[0].value(), 90.0, 0.0);

    // Default K2 = 250 (around September 7)
    CHECK_NEAR(model.parameters()[1].value(), 250.0, 0.0);
}

// Test_Seasonal_DataLogLikelihood_ReturnsFinite: SKIPPED -- the default seasonal GEV
// parameters derive from the deferred seasonal block-maxima path (file header).

// ===========================================================================================
// Clone tests.
// ===========================================================================================

// Test_Clone_CreatesIndependentCopy (AreNotSame -> pointer checks).
void test_clone_creates_independent_copy() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    PointProcessModel clone = model.clone();

    CHECK_TRUE(&clone != &model);
    CHECK_TRUE(clone.distribution() != model.distribution());
}

// Test_Clone_PreservesThreshold
void test_clone_preserves_threshold() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_threshold(1234.5);

    PointProcessModel clone = model.clone();

    CHECK_NEAR(clone.threshold(), 1234.5, 0.0);
}

// Test_Clone_PreservesTotalYears
void test_clone_preserves_total_years() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_total_years(50);

    PointProcessModel clone = model.clone();

    CHECK_NEAR(clone.total_years(), 50.0, 0.0);
}

// Test_Clone_PreservesIsSeasonal
void test_clone_preserves_is_seasonal() {
    DataFrame df = create_seasonal_pot_data_frame();
    PointProcessModel model;
    model.set_is_seasonal(true);
    model.set_data_frame(std::move(df));

    PointProcessModel clone = model.clone();

    CHECK_TRUE(clone.is_seasonal());
}

// Test_Clone_PreservesTimeBlockSettings
void test_clone_preserves_time_block_settings() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_time_block(TimeBlockWindow::CalendarYear);
    model.set_start_month(1);

    PointProcessModel clone = model.clone();

    CHECK_TRUE(clone.time_block() == TimeBlockWindow::CalendarYear);
    CHECK_EQ(clone.start_month(), 1);
}

// Test_Clone_ParametersAreIndependent
void test_clone_parameters_are_independent() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    double original_value = model.parameters()[0].value();

    PointProcessModel clone = model.clone();
    model.parameters()[0].set_value(99999);

    CHECK_NEAR(clone.parameters()[0].value(), original_value, 0.0);
}

// ===========================================================================================
// Validation tests.
// ===========================================================================================

// Test_Validate_ValidModel_ReturnsTrue
void test_validate_valid_model_returns_true() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    ValidationResult result = model.validate();

    CHECK_TRUE(result.is_valid);
}

// Test_Validate_NullDataFrame_ReturnsFalse
void test_validate_null_data_frame_returns_false() {
    PointProcessModel model;

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_message_contains(result, "Data frame"));
}

// Test_Validate_NaNThreshold_ReturnsFalse
void test_validate_nan_threshold_returns_false() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_use_defaults(false);
    model.set_data_frame(std::move(df));
    model.set_threshold(kNaN);

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_message_contains(result, "Threshold"));
}

// Test_Validate_InvalidTotalYears_ReturnsFalse
void test_validate_invalid_total_years_returns_false() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_use_defaults(false);
    model.set_data_frame(std::move(df));
    model.set_threshold(1000);
    model.set_total_years(0);

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_message_contains(result, "TotalYears"));
}

// Test_Validate_InvalidStartMonth_ReturnsFalse. NOTE: with the deferred seasonal AMS path
// the C++ result also carries the "AMSDataFrame has no exact data" message the C# does not
// hit; the transcribed assertions are unaffected.
void test_validate_invalid_start_month_returns_false() {
    DataFrame df = create_seasonal_pot_data_frame();
    PointProcessModel model;
    model.set_is_seasonal(true);
    model.set_data_frame(std::move(df));
    model.set_start_month(13);  // Invalid month

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_message_contains(result, "StartMonth"));
}

// ===========================================================================================
// SetParameterValues tests.
// ===========================================================================================

// Test_SetParameterValues_UpdatesParameters
void test_set_parameter_values_updates_parameters() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    std::vector<double> new_values{2500.0, 750.0, 0.1};
    model.set_parameter_values(new_values);

    CHECK_NEAR(model.parameters()[0].value(), 2500.0, 0.0);
    CHECK_NEAR(model.parameters()[1].value(), 750.0, 0.0);
    CHECK_NEAR(model.parameters()[2].value(), 0.1, 0.0);
}

// Test_SetParameterValues_NullParameters_ThrowsException: SKIPPED -- a std::vector<double>
// argument cannot be null (type-system guarantee).

// Test_SetParameterValues_WrongCount_ThrowsException (C# ArgumentException).
void test_set_parameter_values_wrong_count_throws() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    std::vector<double> wrong_count{1.0};
    CHECK_THROWS(model.set_parameter_values(wrong_count));
}

// ===========================================================================================
// GetDistribution tests.
// ===========================================================================================

// Test_GetDistribution_ReturnsClonedDistribution
void test_get_distribution_returns_cloned_distribution() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    std::vector<double> parameters = parameter_values(model);
    auto dist = model.get_distribution(parameters);

    CHECK_TRUE(dist != nullptr);
    CHECK_TRUE(dist.get() != model.distribution());
}

// Test_GetDistribution_AppliesParameters
void test_get_distribution_applies_parameters() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    std::vector<double> parameters{2000.0, 500.0, 0.1};
    auto dist = model.get_distribution(parameters);

    const auto& gev =
        static_cast<const GeneralizedExtremeValue&>(dist->component(0));
    CHECK_NEAR(gev.xi(), 2000.0, 1e-10);
    CHECK_NEAR(gev.alpha(), 500.0, 1e-10);
    CHECK_NEAR(gev.kappa(), 0.1, 1e-10);
}

// ===========================================================================================
// Engineering application tests.
// ===========================================================================================

// Test_PointProcess_FloodFrequencyAnalysis
void test_point_process_flood_frequency_analysis() {
    // Typical POT flood frequency analysis scenario
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    ValidationResult result = model.validate();
    CHECK_TRUE(result.is_valid);

    std::vector<double> parameters = parameter_values(model);
    double data_log_lh = model.data_log_likelihood(parameters);
    CHECK_TRUE(data_log_lh != -kInf);
}

// Test_PointProcess_SeasonalFloodAnalysis: SKIPPED -- Validate() requires seasonal AMS
// exact data, produced only by the deferred seasonal block-maxima path (file header).

// Test_PointProcess_WaterYearConvention
void test_point_process_water_year_convention() {
    // Water year starting in October
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_time_block(TimeBlockWindow::WaterYear);
    model.set_start_month(10);
    model.set_data_frame(std::move(df));

    CHECK_TRUE(model.time_block() == TimeBlockWindow::WaterYear);
    CHECK_EQ(model.start_month(), 10);
}

// ===========================================================================================
// Jeffreys prior tests.
// ===========================================================================================

// Test_JeffreysPrior_AffectsPriorLogLikelihood
void test_jeffreys_prior_affects_prior_log_likelihood() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    std::vector<double> parameters = parameter_values(model);

    model.set_use_jeffreys_rule_for_scale(false);
    double prior_no_jeffreys = model.prior_log_likelihood(parameters);

    model.set_use_jeffreys_rule_for_scale(true);
    double prior_with_jeffreys = model.prior_log_likelihood(parameters);

    CHECK_TRUE(prior_no_jeffreys != prior_with_jeffreys);
}

// Test_JeffreysPrior_Seasonal_AppliedToBothGEVs: SKIPPED -- the default seasonal parameter
// priors are NaN-bounded without the deferred seasonal AMS path; evaluating them throws in
// the port where the C# (with real data) returns finite values (file header).

// EXTRA (not in the C# suite): the non-seasonal Jeffreys pointwise component -- one
// "Jeffreys Scale: Scale" entry (the seasonal counterpart is exercised upstream only
// through the deferred data path).
void test_jeffreys_prior_non_seasonal_pointwise_component() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_use_jeffreys_rule_for_scale(true);

    std::vector<double> parameters = parameter_values(model);
    auto priors = model.pointwise_prior_log_likelihood(parameters);

    int jeffreys_count = 0;
    bool label_found = false;
    for (const auto& p : priors) {
        if (p.type() == PriorComponentType::JeffreysScalePrior) {
            ++jeffreys_count;
            if (p.name() == "Jeffreys Scale: Scale") label_found = true;
        }
    }
    CHECK_EQ(jeffreys_count, 1);
    CHECK_TRUE(label_found);
}

// ===========================================================================================
// Edge cases.
// ===========================================================================================

// Test_PointProcess_SingleDataPoint
void test_point_process_single_data_point() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<ExactData>{ExactData(2000, 1000)}));

    PointProcessModel model;
    model.set_data_frame(std::move(df));

    CHECK_TRUE(model.has_distribution());
}

// Test_PointProcess_LargeValues
void test_point_process_large_values() {
    DataFrame df;
    std::vector<ExactData> data;
    for (int i = 0; i < 10; ++i) data.emplace_back(2000 + i, 1e8 + i * 1e6);
    df.set_exact_series(ExactSeries(data));

    PointProcessModel model;
    model.set_data_frame(std::move(df));

    ValidationResult result = model.validate();
    CHECK_TRUE(result.is_valid);
}

// Test_PointProcess_SmallValues
void test_point_process_small_values() {
    DataFrame df;
    std::vector<ExactData> data;
    for (int i = 0; i < 10; ++i) data.emplace_back(2000 + i, 0.01 + i * 0.001);
    df.set_exact_series(ExactSeries(data));

    PointProcessModel model;
    model.set_data_frame(std::move(df));

    ValidationResult result = model.validate();
    CHECK_TRUE(result.is_valid);
}

// Test_PointProcess_ManyEventsPerYear
void test_point_process_many_events_per_year() {
    // High-frequency POT data
    DataFrame df;
    std::vector<ExactData> data;
    for (int i = 0; i < 100; ++i) data.emplace_back(2000 + i / 10, 1000 + i * 50);
    df.set_exact_series(ExactSeries(data));

    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_total_years(10);

    // Lambda should be ~10 events per year
    CHECK_NEAR(model.lambda(), 10.0, 1e-10);
}

// ===========================================================================================
// Quantile prior tests.
// ===========================================================================================

// Test_SetDefaultQuantilePriors_SingleQuantile
void test_set_default_quantile_priors_single_quantile() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_enable_quantile_priors(true);
    model.set_use_single_quantile(true);

    model.set_default_quantile_priors();

    CHECK_EQ(model.quantile_priors().size(), static_cast<std::size_t>(1));
}

// Test_SetDefaultQuantilePriors_ThreeQuantiles_SingleGEV
void test_set_default_quantile_priors_three_quantiles_single_gev() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_enable_quantile_priors(true);
    model.set_use_single_quantile(false);

    model.set_default_quantile_priors();

    // Single GEV allows 3 quantile priors (one per parameter)
    CHECK_EQ(model.quantile_priors().size(), static_cast<std::size_t>(3));
}

// Test_SetDefaultQuantilePriors_Disabled_EmptyList
void test_set_default_quantile_priors_disabled_empty_list() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_enable_quantile_priors(false);

    model.set_default_quantile_priors();

    CHECK_EQ(model.quantile_priors().size(), static_cast<std::size_t>(0));
}

// ===========================================================================================
// Pointwise vs scalar prior sum-equality (CON-4).
// ===========================================================================================

// Test_PointwisePriorLogLikelihood_Sum_Equals_PriorLogLikelihood_NonSeasonal
void test_pointwise_prior_log_likelihood_sum_equals_prior_log_likelihood_non_seasonal() {
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_use_jeffreys_rule_for_scale(false);
    model.set_default_parameters();

    std::vector<double> p = parameter_values(model);
    double scalar = model.prior_log_likelihood(p);
    auto pointwise = model.pointwise_prior_log_likelihood(p);
    double pointwise_sum = 0.0;
    for (const auto& c : pointwise) pointwise_sum += c.log_likelihood();

    CHECK_NEAR(pointwise_sum, scalar, 1e-9);
}

// Test_PointwisePriorLogLikelihood_Sum_Equals_PriorLogLikelihood_Seasonal: SKIPPED -- the
// default seasonal parameter priors are NaN-bounded without the deferred seasonal AMS path
// (file header).

// ===========================================================================================
// ISimulatable guards + seeded determinism (EXTRA; M10/M11 precedent -- no upstream test
// exercises the GenerateRandomValues stream; M14 pins the exact digests).
// ===========================================================================================

void test_generate_random_values_guards_and_deterministic_seed() {
    DataFrame df = create_ams_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_parameter_values({2000.0, 500.0, 0.1});

    // sampleSize <= 0 -> C# ArgumentOutOfRangeException.
    CHECK_THROWS(model.generate_random_values(0));

    // Null distribution -> C# InvalidOperationException.
    PointProcessModel no_dist;
    no_dist.set_distribution(nullptr);
    CHECK_THROWS(no_dist.generate_random_values(5));

    // Seeded generation is deterministic.
    std::vector<double> a = model.generate_random_values(10, 12345);
    std::vector<double> b = model.generate_random_values(10, 12345);
    CHECK_EQ(a.size(), static_cast<std::size_t>(10));
    CHECK_TRUE(a == b);

    // A different seed gives a different stream.
    std::vector<double> c = model.generate_random_values(10, 54321);
    CHECK_TRUE(a != c);
}

// ===================== P6: the seasonal data path =====================
// The five tests P5 and earlier skipped because the seasonal branch of set_ams_data() was a
// no-op. P6 ported the TimeSeries container it needs, so each is transcribed from
// src/RMC.BestFit.Tests/Univariate/PointProcessModelTests.cs @ c2e6192.

// C# Test_SetAMSData_Seasonal_CreatesPOTDays.
void test_set_ams_data_seasonal_creates_pot_days() {
    DataFrame df = create_seasonal_pot_data_frame();
    PointProcessModel model;
    model.set_is_seasonal(true);
    model.set_data_frame(std::move(df));

    CHECK_TRUE(model.pot_days().size() > 0);

    // corehydro supplement: the day of year is read AFTER the water-year shift, so a February
    // event reports its water-year day rather than day 46. February 15 shifted forward by three
    // months is May 15 -- day 135 or 136 depending on the leap year -- and July 15 becomes
    // October 15, day 288 or 289.
    //
    // The ORDER is the upstream defect documented in docs/upstream-csharp-issues.md and in the
    // model header: ShiftDatesByMonth sorts by time, so POTDays comes out in DATE order (Feb
    // 1990, Jul 1990, Feb 1991, ...) while the likelihood pairs it positionally with the
    // UNSORTED exact series (all Februaries, then all Julys). This pins the ported order, which
    // is C#'s.
    CHECK_EQ(static_cast<int>(model.pot_days().size()), 20);
    CHECK_TRUE(model.pot_days()[0] == 135 || model.pot_days()[0] == 136);
    CHECK_TRUE(model.pot_days()[1] == 288 || model.pot_days()[1] == 289);
    CHECK_TRUE(model.pot_days()[2] == 135 || model.pot_days()[2] == 136);
    // The block series is populated too, one maximum per water year.
    CHECK_TRUE(model.ams_data_frame().exact_series().count() > 0);
}

// C# Test_Seasonal_DataLogLikelihood_ReturnsFinite.
void test_seasonal_data_log_likelihood_returns_finite() {
    DataFrame df = create_seasonal_pot_data_frame();
    PointProcessModel model;
    model.set_is_seasonal(true);
    model.set_data_frame(std::move(df));

    std::vector<double> parameters;
    for (const auto& p : model.parameters()) parameters.push_back(p.value());
    double data_log_lh = model.data_log_likelihood(parameters);
    CHECK_TRUE(!std::isnan(data_log_lh));
}

// C# Test_JeffreysPrior_Seasonal_AppliedToBothGEVs.
void test_jeffreys_prior_seasonal_applied_to_both_gevs() {
    DataFrame df = create_seasonal_pot_data_frame();
    PointProcessModel model;
    model.set_is_seasonal(true);
    model.set_use_jeffreys_rule_for_scale(true);
    model.set_data_frame(std::move(df));

    std::vector<double> parameters;
    for (const auto& p : model.parameters()) parameters.push_back(p.value());
    auto priors = model.pointwise_prior_log_likelihood(parameters);

    int jeffreys_count = 0;
    for (const auto& p : priors)
        if (p.type() == PriorComponentType::JeffreysScalePrior) ++jeffreys_count;
    CHECK_EQ(jeffreys_count, 2);
}

// C# Test_PointwisePriorLogLikelihood_Sum_Equals_PriorLogLikelihood_Seasonal.
void test_pointwise_prior_sum_equals_prior_seasonal() {
    DataFrame df = create_seasonal_pot_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));
    model.set_use_jeffreys_rule_for_scale(false);
    model.set_is_seasonal(true);
    model.set_default_parameters();

    std::vector<double> p;
    for (const auto& param : model.parameters()) p.push_back(param.value());
    double scalar = model.prior_log_likelihood(p);
    double pointwise_sum = 0;
    for (const auto& c : model.pointwise_prior_log_likelihood(p)) pointwise_sum += c.log_likelihood();
    CHECK_NEAR(pointwise_sum, scalar, 1e-9);
}

// C# Test_PointProcess_SeasonalFloodAnalysis.
void test_point_process_seasonal_flood_analysis() {
    DataFrame df = create_seasonal_pot_data_frame();
    PointProcessModel model;
    model.set_is_seasonal(true);
    model.set_data_frame(std::move(df));

    CHECK_TRUE(model.validate().is_valid);
    CHECK_EQ(model.distribution()->component_count(), 2);
}

// corehydro supplement: generate_pot_time_series, whose only C# test coverage is indirect. It is
// the seeded synthetic-record path, so what is asserted is the contract -- determinism under a
// seed, dates inside the requested span, magnitudes above the threshold, sorted output -- and the
// guards.
void test_generate_pot_time_series() {
    using corehydro::numerics::data::DateTime;
    DataFrame df = create_pot_data_frame();
    PointProcessModel model;
    model.set_data_frame(std::move(df));

    DateTime start(2000, 1, 1);
    auto a = model.generate_pot_time_series(start, 10.0, 12345);
    auto b = model.generate_pot_time_series(start, 10.0, 12345);
    CHECK_EQ(a.count(), b.count());
    for (int i = 0; i < a.count(); ++i) {
        CHECK_TRUE(a[i].index() == b[i].index());
        CHECK_NEAR(a[i].value(), b[i].value(), 0.0);
    }
    CHECK_TRUE(a.count() > 0);
    CHECK_TRUE(a.time_interval() == corehydro::numerics::data::TimeInterval::OneDay);
    DateTime end = start.add_days(10.0 * 365.25);
    for (int i = 0; i < a.count(); ++i) {
        CHECK_TRUE(a[i].index() >= start && a[i].index() <= end);
        CHECK_TRUE(a[i].value() > model.threshold());
        if (i > 0) CHECK_TRUE(a[i - 1].index() <= a[i].index());
    }

    CHECK_THROWS(model.generate_pot_time_series(start, 0.0, 12345));
    CHECK_THROWS(model.generate_pot_time_series(start, -1.0, 12345));
}

}  // namespace

int main() {
    // Constructor tests.
    test_constructor_empty_constructor_creates_default_model();
    test_constructor_empty_constructor_has_single_gev();
    test_constructor_with_data_and_distribution_sets_properties();
    test_constructor_with_data_and_distribution_clones_distribution();

    // Property tests.
    test_distribution_set_and_get();
    test_threshold_set_and_get();
    test_total_years_set_and_get();
    test_total_years_updates_lambda();
    test_use_defaults_sets_threshold_and_years();
    test_is_seasonal_set_and_get();
    test_is_seasonal_true_creates_two_gevs();
    test_is_seasonal_false_creates_single_gev();
    test_time_block_set_and_get();
    test_start_month_set_and_get();
    test_lambda_calculated_from_events_and_years();

    // SetDistribution tests.
    test_set_distribution_non_seasonal_single_gev();
    test_set_distribution_seasonal_two_gevs();

    // SetDefaultParameters tests.
    test_set_default_parameters_non_seasonal_three_gev_parameters();
    test_set_default_parameters_seasonal_eight_parameters();
    test_set_default_parameters_seasonal_has_change_points();
    test_set_default_parameters_has_uniform_priors();

    // SetDefaultThresholdAndTotalYears tests.
    test_set_default_threshold_and_total_years_sets_threshold();
    test_set_default_threshold_and_total_years_sets_total_years();
    test_use_defaults_recomputes_total_years_after_manual_override();
    test_set_default_threshold_and_total_years_total_years_event_sees_updated_lambda();
    test_set_default_threshold_and_total_years_uses_peaks_over_threshold_input();
    test_set_default_threshold_and_total_years_clamps_peaks_over_threshold_input();
    test_set_default_threshold_and_total_years_total_years_uses_exact_span_only();

    // CalculateLambda tests.
    test_calculate_lambda_returns_events_per_year();
    test_calculate_lambda_null_data_frame_returns_nan();
    test_calculate_lambda_zero_years_returns_nan();

    // SetAMSData tests.
    test_set_ams_data_non_seasonal_creates_block_maxima();
    test_set_ams_data_block_maxima_values_and_order();

    // LogLikelihood tests.
    test_data_log_likelihood_returns_finite_value();
    test_prior_log_likelihood_returns_finite_value();
    test_data_log_likelihood_null_data_frame_returns_negative_infinity();
    test_data_log_likelihood_nan_threshold_returns_negative_infinity();
    test_pointwise_data_log_likelihood_returns_correct_count();
    test_pointwise_data_log_likelihood_sum_equals_data_log_likelihood();

    // Seasonal model tests.
    test_seasonal_change_points_have_correct_bounds();
    test_seasonal_change_point_default_values();

    // Clone tests.
    test_clone_creates_independent_copy();
    test_clone_preserves_threshold();
    test_clone_preserves_total_years();
    test_clone_preserves_is_seasonal();
    test_clone_preserves_time_block_settings();
    test_clone_parameters_are_independent();

    // Validation tests.
    test_validate_valid_model_returns_true();
    test_validate_null_data_frame_returns_false();
    test_validate_nan_threshold_returns_false();
    test_validate_invalid_total_years_returns_false();
    test_validate_invalid_start_month_returns_false();

    // SetParameterValues tests.
    test_set_parameter_values_updates_parameters();
    test_set_parameter_values_wrong_count_throws();

    // GetDistribution tests.
    test_get_distribution_returns_cloned_distribution();
    test_get_distribution_applies_parameters();

    // Engineering application tests.
    test_point_process_flood_frequency_analysis();
    test_point_process_water_year_convention();

    // Jeffreys prior tests.
    test_jeffreys_prior_affects_prior_log_likelihood();
    test_jeffreys_prior_non_seasonal_pointwise_component();

    // Edge cases.
    test_point_process_single_data_point();
    test_point_process_large_values();
    test_point_process_small_values();
    test_point_process_many_events_per_year();

    // Quantile prior tests.
    test_set_default_quantile_priors_single_quantile();
    test_set_default_quantile_priors_three_quantiles_single_gev();
    test_set_default_quantile_priors_disabled_empty_list();

    // Pointwise vs scalar prior sum-equality (CON-4).
    test_pointwise_prior_log_likelihood_sum_equals_prior_log_likelihood_non_seasonal();

    // ISimulatable guards + seeded determinism.
    test_generate_random_values_guards_and_deterministic_seed();

    // P6: the seasonal data path.
    test_set_ams_data_seasonal_creates_pot_days();
    test_seasonal_data_log_likelihood_returns_finite();
    test_jeffreys_prior_seasonal_applied_to_both_gevs();
    test_pointwise_prior_sum_equals_prior_seasonal();
    test_point_process_seasonal_flood_analysis();
    test_generate_pot_time_series();

    return chtest::summary("point_process_model");
}
