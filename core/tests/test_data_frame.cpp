// Standalone tests for the DataFrame series collections and the DataFrame core (M4):
// DataSeries / ExactSeries / IntervalSeries / ThresholdSeries / UncertainSeries and
// corehydro::models::DataFrame.
//
// Oracles are the upstream C# test classes @ fc28c0c under
// upstream/RMC-BestFit/src/RMC.BestFit.Tests/:
//   DataFrame/ExactSeriesTests.cs, DataFrame/IntervalSeriesTests.cs,
//   DataFrame/ThresholdSeriesTests.cs, DataFrame/UncertainSeriesTests.cs,
//   DataFrame/DataFrameTests.cs, DataFrame/ExactDataHypothesisTests.cs (the two
//   low-outlier methods only), Univariate/ThresholdLikelihoodGuardTests.cs (the
//   DataFrame/FullTimeSeries state assertions only),
// transcribed method-for-method below (same section order). Deliberately NOT transcribed
// (project-wide deferrals / later milestones -- the full list is in the M4 report):
//   - every ToXElement / XElement / XML round-trip test (XML is not ported)
//   - every CollectionChanged / PropertyChanged test (INPC is not ported); the
//     SuppressCollectionChanged surface does not exist in the C++ port
//   - DataFrameConcurrencyTests.cs entirely (the C++ port is single-threaded by design)
//   - the plotting-position tests (CalculatePlottingPositions is M5)
//   - the hypothesis-test methods of ExactDataHypothesisTests.cs (Numerics
//     HypothesisTests is unported; the DataFrame facade is deferred)
//   - ExactDataProcessTests.cs entirely (CreateBlockSeries / peaks-over-threshold /
//     USGS need the unported TimeSeries container; those members are deferred)
//   - the log-likelihood assertions of ThresholdLikelihoodGuardTests.cs (M8)
//
// Where the C# relied on the INPC event chain (ThresholdSeries.Add auto-deriving
// NumberBelow, exact-series changes recalculating Lambda), the C++ tests call the
// documented explicit triggers process_threshold_series() / calculate_lambda() -- see the
// invalidation-strategy note in data_frame.hpp.
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "corehydro/models/data_frame/data_collections/data_series.hpp"
#include "corehydro/models/data_frame/data_collections/exact_series.hpp"
#include "corehydro/models/data_frame/data_collections/interval_series.hpp"
#include "corehydro/models/data_frame/data_collections/threshold_series.hpp"
#include "corehydro/models/data_frame/data_collections/uncertain_series.hpp"
#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/data_frame/data_types/exact_data.hpp"
#include "corehydro/models/data_frame/data_types/interval_data.hpp"
#include "corehydro/models/data_frame/data_types/threshold_data.hpp"
#include "corehydro/models/data_frame/data_types/uncertain_data.hpp"
#include "corehydro/numerics/data/interpolation/sort_order.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/log_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/triangular.hpp"
#include "check.hpp"

using corehydro::models::DataFrame;
using corehydro::models::ExactData;
using corehydro::models::ExactSeries;
using corehydro::models::IntervalData;
using corehydro::models::IntervalSeries;
using corehydro::models::ThresholdData;
using corehydro::models::ThresholdSeries;
using corehydro::models::UncertainData;
using corehydro::models::UncertainSeries;
using corehydro::models::ValidationResult;
using corehydro::numerics::data::SortOrder;
using corehydro::numerics::distributions::LogNormal;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::Triangular;
using corehydro::numerics::distributions::UnivariateDistributionType;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kDoubleMax = std::numeric_limits<double>::max();
// C# double.MinValue is the most-negative finite double.
constexpr double kDoubleMin = std::numeric_limits<double>::lowest();

// Mirrors the C# LINQ `messages.Any(m => m.Contains(substr))`.
bool any_contains(const std::vector<std::string>& messages, const std::string& substr) {
    for (const auto& m : messages) {
        if (m.find(substr) != std::string::npos) return true;
    }
    return false;
}

// C# DataFrameTests SampleAnnualPeaks (annual peak flows, cfs).
const std::vector<double>& sample_annual_peaks() {
    static const std::vector<double> peaks = {
        45000, 38000, 52000, 61000, 33000, 49000, 55000, 42000, 67000, 39000,
        48000, 51000, 36000, 58000, 44000, 53000, 47000, 62000, 41000, 50000,
        37000, 54000, 46000, 59000, 43000, 56000, 40000, 63000, 35000, 57000};
    return peaks;
}

// C# DataFrameTests.CreateTestDataFrame(count).
DataFrame create_test_data_frame(std::size_t count = 30) {
    DataFrame df;
    std::vector<double> data(sample_annual_peaks().begin(),
                             sample_annual_peaks().begin() +
                                 static_cast<std::ptrdiff_t>(count));
    df.set_exact_series(ExactSeries(data));
    return df;
}

// C# DataFrameTests.CreateMixedDataFrame. In C# ThresholdSeries.Add auto-derives
// NumberBelow through the event chain; here the caller runs process_threshold_series().
DataFrame create_mixed_data_frame() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>{45000, 52000, 61000, 49000, 55000,
                                                        67000, 48000, 58000, 53000, 62000}));
    df.uncertain_series().add(UncertainData(1889, std::make_unique<Normal>(85000, 10000)));
    df.uncertain_series().add(UncertainData(1913, std::make_unique<Normal>(75000, 8000)));
    df.interval_series().add(IntervalData(1500, 60000, 80000, 100000));
    df.interval_series().add(IntervalData(1700, 50000, 70000, 90000));
    ThresholdData threshold(1850, 1920, 40000);
    threshold.set_number_above(5);
    df.threshold_series().add(threshold);
    df.process_threshold_series();
    return df;
}

ThresholdData make_threshold(int start, int end, double value, int number_above = 0) {
    ThresholdData t(start, end, value);
    t.set_number_above(number_above);
    return t;
}

// ---------------------------------------------------------------------------
// DataSeries base surface (exercised through ExactSeries; no dedicated C# test
// class -- the IList surface is validated against the C# source)
// ---------------------------------------------------------------------------

void test_series_add_insert_remove_clear() {
    ExactSeries series;
    series.add(ExactData(1980, 1.0));
    series.add(ExactData(1982, 3.0));
    series.insert(1, ExactData(1981, 2.0));
    CHECK_EQ(series.count(), std::size_t{3});
    CHECK_EQ(series[0].index(), 1980);
    CHECK_EQ(series[1].index(), 1981);
    CHECK_EQ(series[2].index(), 1982);

    // Remove by identity (C# Remove(item)) and by position (C# RemoveAt).
    CHECK_TRUE(series.contains(series[1]));
    CHECK_EQ(series.index_of(series[1]), 1);
    CHECK_TRUE(series.remove(series[1]));
    CHECK_EQ(series.count(), std::size_t{2});
    series.remove_at(0);
    CHECK_EQ(series.count(), std::size_t{1});
    CHECK_EQ(series[0].index(), 1982);

    series.clear();
    CHECK_EQ(series.count(), std::size_t{0});
}

void test_series_helper_lists() {
    ExactSeries series;
    series.add(ExactData(1980, 10.0, 0.25));
    series.add(ExactData(1981, 20.0, 0.5));
    // ValuesToList / ValuesToArray both map to std::vector<double>.
    std::vector<double> values = series.values_to_list();
    CHECK_EQ(values.size(), std::size_t{2});
    CHECK_NEAR(values[0], 10.0, 0.0);
    CHECK_NEAR(values[1], 20.0, 0.0);
    std::vector<double> pps = series.plotting_positions_to_list();
    CHECK_NEAR(pps[0], 0.25, 0.0);
    CHECK_NEAR(pps[1], 0.5, 0.0);
    std::vector<int> indices = series.indices_to_list();
    CHECK_EQ(indices[0], 1980);
    CHECK_EQ(indices[1], 1981);
}

// ---------------------------------------------------------------------------
// ExactSeries (transcribed from ExactSeriesTests.cs)
// ---------------------------------------------------------------------------

// C# Test_Constructor_Empty_HasZeroCount
void test_exact_series_constructor_empty_has_zero_count() {
    ExactSeries series;
    CHECK_EQ(series.count(), std::size_t{0});
    CHECK_TRUE(!series.enforce_unique_index());
}

// C# Test_Constructor_FromValues_PopulatesWithSequentialIndexes
void test_exact_series_constructor_from_values_populates_sequential_indexes() {
    ExactSeries series(std::vector<double>{10.0, 20.0, 30.0});
    CHECK_EQ(series.count(), std::size_t{3});
    CHECK_EQ(series[0].index(), 0);
    CHECK_NEAR(series[0].value(), 10.0, 0.0);
    CHECK_EQ(series[2].index(), 2);
    CHECK_NEAR(series[2].value(), 30.0, 0.0);
}

// C# Test_Constructor_FromExactDataList_ClonesEntries
void test_exact_series_constructor_from_list_clones_entries() {
    std::vector<ExactData> source = {ExactData(1985, 75000.0), ExactData(1986, 82000.0)};
    ExactSeries series(source);
    CHECK_EQ(series.count(), std::size_t{2});
    // Mutate the source after construction; the series must hold its own clones.
    source[0] = ExactData(2000, 1.0);
    CHECK_EQ(series[0].index(), 1985);
    CHECK_NEAR(series[0].value(), 75000.0, 0.0);
}

// C# Test_MedianValue_OddCount_ReturnsMiddleValue
void test_exact_series_median_odd_count_returns_middle_value() {
    ExactSeries series(std::vector<double>{1.0, 5.0, 3.0});  // sorted: 1, 3, 5
    CHECK_NEAR(series.median_value(), 3.0, 1e-12);
}

// C# Test_MedianValue_EvenCount_ReturnsAverageOfTwoMiddle
void test_exact_series_median_even_count_returns_average_of_two_middle() {
    ExactSeries series(std::vector<double>{4.0, 1.0, 2.0, 3.0});  // sorted: 1, 2, 3, 4
    CHECK_NEAR(series.median_value(), 2.5, 1e-12);
}

// C# Test_MedianValue_Empty_ReturnsZero
void test_exact_series_median_empty_returns_zero() {
    ExactSeries series;
    CHECK_NEAR(series.median_value(), 0.0, 0.0);
}

// C# Test_MedianValue_SingleValue_ReturnsThatValue
void test_exact_series_median_single_value_returns_that_value() {
    ExactSeries series(std::vector<double>{42.0});
    CHECK_NEAR(series.median_value(), 42.0, 0.0);
}

// C# Test_UpperMiddleValue_EvenCount_ReturnsValueAtMidIndex
void test_exact_series_upper_middle_even_count_returns_value_at_mid_index() {
    // For even N=4 the upper-middle is the value at index N/2 = 2 of the sorted array.
    ExactSeries series(std::vector<double>{4.0, 1.0, 2.0, 3.0});  // sorted: 1, 2, 3, 4
    CHECK_NEAR(series.upper_middle_value(), 3.0, 0.0);
}

// C# Test_UpperMiddleValue_OddCount_EqualsMedian
void test_exact_series_upper_middle_odd_count_equals_median() {
    ExactSeries series(std::vector<double>{1.0, 5.0, 3.0});
    CHECK_NEAR(series.upper_middle_value(), series.median_value(), 0.0);
}

// C# Test_MinimumValue_Empty_ReturnsMaxSentinel / Test_MaximumValue_Empty_ReturnsMinSentinel
void test_exact_series_min_max_value_empty_returns_sentinels() {
    ExactSeries series;
    CHECK_NEAR(series.minimum_value(), kDoubleMax, 0.0);
    CHECK_NEAR(series.maximum_value(), kDoubleMin, 0.0);
}

// C# Test_MinMaxValue_Populated_ReturnsCorrectExtremes
void test_exact_series_min_max_value_populated_returns_correct_extremes() {
    ExactSeries series(std::vector<double>{7.0, 2.0, 9.0, 4.0});
    CHECK_NEAR(series.minimum_value(), 2.0, 0.0);
    CHECK_NEAR(series.maximum_value(), 9.0, 0.0);
}

// C# Test_MinMaxIndex_Empty_ReturnsSentinels
void test_exact_series_min_max_index_empty_returns_sentinels() {
    ExactSeries series;
    CHECK_EQ(series.minimum_index(), -100000);
    CHECK_EQ(series.maximum_index(), 100000);
}

// C# Test_IndexSpan_Empty_ReturnsZero
void test_exact_series_index_span_empty_returns_zero() {
    ExactSeries series;
    CHECK_EQ(series.index_span(), 0);
}

// C# Test_IndexSpan_Populated_ReturnsMaxMinusMinPlusOne
void test_exact_series_index_span_populated() {
    ExactSeries series(std::vector<ExactData>{ExactData(1985, 1.0), ExactData(1990, 2.0),
                                              ExactData(1995, 3.0)});
    CHECK_EQ(series.index_span(), 11);  // 1995 - 1985 + 1
}

// C# Test_UniqueIndices_CountsDistinctIndexes
void test_exact_series_unique_indices_counts_distinct_indexes() {
    ExactSeries series(std::vector<ExactData>{ExactData(1985, 1.0), ExactData(1985, 2.0),
                                              ExactData(1990, 3.0)});
    CHECK_EQ(series.unique_indices(), 2);
}

// C# Test_SortByIndex_Ascending_OrdersByIndex
void test_exact_series_sort_by_index_ascending() {
    ExactSeries series(std::vector<ExactData>{ExactData(1990, 1.0), ExactData(1980, 2.0),
                                              ExactData(1985, 3.0)});
    series.sort_by_index(SortOrder::Ascending);
    CHECK_EQ(series[0].index(), 1980);
    CHECK_EQ(series[1].index(), 1985);
    CHECK_EQ(series[2].index(), 1990);
}

// C# Test_SortByIndex_Descending_OrdersByIndexDescending
void test_exact_series_sort_by_index_descending() {
    ExactSeries series(std::vector<ExactData>{ExactData(1985, 1.0), ExactData(1990, 2.0),
                                              ExactData(1980, 3.0)});
    series.sort_by_index(SortOrder::Descending);
    CHECK_EQ(series[0].index(), 1990);
    CHECK_EQ(series[2].index(), 1980);
}

// C# Test_Sort_Ascending_OrdersByValue
void test_exact_series_sort_ascending_orders_by_value() {
    ExactSeries series(std::vector<double>{5.0, 1.0, 3.0});
    series.sort(SortOrder::Ascending);
    CHECK_NEAR(series[0].value(), 1.0, 0.0);
    CHECK_NEAR(series[1].value(), 3.0, 0.0);
    CHECK_NEAR(series[2].value(), 5.0, 0.0);
}

// C# Test_Sort_Descending_OrdersByValueDescending
void test_exact_series_sort_descending_orders_by_value() {
    ExactSeries series(std::vector<double>{1.0, 5.0, 3.0});
    series.sort(SortOrder::Descending);
    CHECK_NEAR(series[0].value(), 5.0, 0.0);
    CHECK_NEAR(series[2].value(), 1.0, 0.0);
}

// C# Test_Validate_EmptySeries_ReturnsValid
void test_exact_series_validate_empty_returns_valid() {
    ExactSeries series;
    ValidationResult result = series.validate();
    CHECK_TRUE(result.is_valid);
    CHECK_EQ(result.validation_messages.size(), std::size_t{0});
}

// C# Test_Validate_DuplicateIndexes_RejectedWhenEnforced
void test_exact_series_validate_duplicate_indexes_rejected_when_enforced() {
    ExactSeries series(std::vector<ExactData>{ExactData(1985, 1.0), ExactData(1985, 2.0)});
    series.set_enforce_unique_index(true);
    ValidationResult result = series.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "Duplicate"));
}

// C# Test_Validate_DuplicateIndexes_AcceptedWhenNotEnforced
void test_exact_series_validate_duplicate_indexes_accepted_when_not_enforced() {
    ExactSeries series(std::vector<ExactData>{ExactData(1985, 1.0), ExactData(1985, 2.0)});
    CHECK_TRUE(series.validate().is_valid);
}

// C# Test_Clone_ProducesIndependentCopy
void test_exact_series_clone_produces_independent_copy() {
    ExactSeries original(std::vector<double>{1.0, 2.0, 3.0});
    ExactSeries clone = original.clone();
    clone.add(ExactData(99, 99.0));
    CHECK_EQ(original.count(), std::size_t{3});
    CHECK_EQ(clone.count(), std::size_t{4});
}

// C# Test_ToList_ReturnsClonedItems
void test_exact_series_to_list_returns_cloned_items() {
    ExactSeries series(std::vector<ExactData>{ExactData(1985, 1.0)});
    std::vector<ExactData> list = series.to_list();
    list[0] = ExactData(2000, 999.0);
    CHECK_EQ(series[0].index(), 1985);
}

// C# Test_ValuesToArray_ReturnsValuesInOrder
void test_exact_series_values_to_array_returns_values_in_order() {
    ExactSeries series(std::vector<double>{10.0, 20.0, 30.0});
    std::vector<double> values = series.values_to_list();
    CHECK_EQ(values.size(), std::size_t{3});
    CHECK_NEAR(values[0], 10.0, 0.0);
    CHECK_NEAR(values[1], 20.0, 0.0);
    CHECK_NEAR(values[2], 30.0, 0.0);
}

// ---------------------------------------------------------------------------
// IntervalSeries (transcribed from IntervalSeriesTests.cs)
// ---------------------------------------------------------------------------

// C# Test_Constructor_Empty_HasZeroCount
void test_interval_series_constructor_empty_has_zero_count() {
    IntervalSeries series;
    CHECK_EQ(series.count(), std::size_t{0});
}

// C# Test_Constructor_FromList_ClonesEntries
void test_interval_series_constructor_from_list_clones_entries() {
    std::vector<IntervalData> source = {IntervalData(1900, 5000, 7000, 9000)};
    IntervalSeries series(source);
    CHECK_EQ(series.count(), std::size_t{1});
    source[0] = IntervalData(2000, 1, 2, 3);
    CHECK_EQ(series[0].index(), 1900);
}

// C# Test_MinimumValue_Empty_ReturnsMaxSentinel / Test_MaximumValue_Empty_ReturnsMinSentinel
void test_interval_series_min_max_value_empty_returns_sentinels() {
    IntervalSeries series;
    CHECK_NEAR(series.minimum_value(), kDoubleMax, 0.0);
    CHECK_NEAR(series.maximum_value(), kDoubleMin, 0.0);
}

// C# Test_MinimumValue_UsesLowerValueAcrossEntries
void test_interval_series_minimum_value_uses_lower_value() {
    IntervalSeries series(std::vector<IntervalData>{
        IntervalData(1500, 60000, 80000, 100000), IntervalData(1700, 50000, 70000, 90000)});
    CHECK_NEAR(series.minimum_value(), 50000, 1e-12);
}

// C# Test_MaximumValue_UsesUpperValueAcrossEntries
void test_interval_series_maximum_value_uses_upper_value() {
    IntervalSeries series(std::vector<IntervalData>{
        IntervalData(1500, 60000, 80000, 100000), IntervalData(1700, 50000, 70000, 90000)});
    CHECK_NEAR(series.maximum_value(), 100000, 1e-12);
}

// C# Test_MinMaxIndex_Empty_ReturnsSentinels
void test_interval_series_min_max_index_empty_returns_sentinels() {
    IntervalSeries series;
    CHECK_EQ(series.minimum_index(), -100000);
    CHECK_EQ(series.maximum_index(), 100000);
}

// C# Test_SortByIndex_Ascending_OrdersByIndex
void test_interval_series_sort_by_index_ascending() {
    IntervalSeries series(std::vector<IntervalData>{IntervalData(1700, 1, 2, 3),
                                                    IntervalData(1500, 1, 2, 3),
                                                    IntervalData(1800, 1, 2, 3)});
    series.sort_by_index(SortOrder::Ascending);
    CHECK_EQ(series[0].index(), 1500);
    CHECK_EQ(series[1].index(), 1700);
    CHECK_EQ(series[2].index(), 1800);
}

// C# Test_Sort_Ascending_OrdersByValue
void test_interval_series_sort_ascending_orders_by_value() {
    IntervalSeries series(std::vector<IntervalData>{IntervalData(1500, 1, 5, 10),
                                                    IntervalData(1700, 1, 1, 10),
                                                    IntervalData(1800, 1, 3, 10)});
    series.sort(SortOrder::Ascending);
    CHECK_NEAR(series[0].value(), 1.0, 0.0);
    CHECK_NEAR(series[1].value(), 3.0, 0.0);
    CHECK_NEAR(series[2].value(), 5.0, 0.0);
}

// C# Test_Validate_EmptySeries_NoDataFrame_ReturnsValid
void test_interval_series_validate_empty_no_data_frame_returns_valid() {
    IntervalSeries series;
    ValidationResult result = series.validate(nullptr);
    CHECK_TRUE(result.is_valid);
    CHECK_EQ(result.validation_messages.size(), std::size_t{0});
}

// C# Test_Validate_OverlapsWithExactSeries_FailsValidation
void test_interval_series_validate_overlaps_with_exact_fails() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<ExactData>{ExactData(1500, 70000)}));
    df.interval_series().add(IntervalData(1500, 60000, 80000, 100000));
    ValidationResult result = df.interval_series().validate(&df);
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "overlaps with exact"));
}

// C# Test_Clone_ProducesIndependentCopy
void test_interval_series_clone_produces_independent_copy() {
    IntervalSeries original(std::vector<IntervalData>{IntervalData(1500, 1, 2, 3)});
    IntervalSeries clone = original.clone();
    clone.add(IntervalData(2000, 4, 5, 6));
    CHECK_EQ(original.count(), std::size_t{1});
    CHECK_EQ(clone.count(), std::size_t{2});
}

// C# Test_ToList_ReturnsClonedItems
void test_interval_series_to_list_returns_cloned_items() {
    IntervalSeries series(std::vector<IntervalData>{IntervalData(1500, 1, 2, 3)});
    std::vector<IntervalData> list = series.to_list();
    list[0] = IntervalData(2000, 9, 9, 9);
    CHECK_EQ(series[0].index(), 1500);
}

// ---------------------------------------------------------------------------
// ThresholdSeries (transcribed from ThresholdSeriesTests.cs)
// ---------------------------------------------------------------------------

// C# Test_Constructor_Empty_HasZeroCount
void test_threshold_series_constructor_empty_has_zero_count() {
    ThresholdSeries series;
    CHECK_EQ(series.count(), std::size_t{0});
}

// C# Test_Constructor_FromList_ClonesEntries
void test_threshold_series_constructor_from_list_clones_entries() {
    std::vector<ThresholdData> source = {make_threshold(1850, 1920, 40000, 5)};
    ThresholdSeries series(source);
    CHECK_EQ(series.count(), std::size_t{1});
    source[0] = make_threshold(0, 1, 0);
    CHECK_EQ(series[0].start_index(), 1850);
    CHECK_EQ(series[0].end_index(), 1920);
    CHECK_NEAR(series[0].value(), 40000, 1e-12);
    CHECK_EQ(series[0].number_above(), 5);
}

// C# Test_MinimumValue_Empty_ReturnsMaxSentinel / Test_MaximumValue_Empty_ReturnsMinSentinel
void test_threshold_series_min_max_value_empty_returns_sentinels() {
    ThresholdSeries series;
    CHECK_NEAR(series.minimum_value(), kDoubleMax, 0.0);
    CHECK_NEAR(series.maximum_value(), kDoubleMin, 0.0);
}

// C# Test_MinMaxValue_Populated_ReturnsCorrectExtremes
void test_threshold_series_min_max_value_populated() {
    ThresholdSeries series(std::vector<ThresholdData>{make_threshold(1800, 1850, 30000),
                                                      make_threshold(1851, 1900, 50000),
                                                      make_threshold(1901, 1950, 40000)});
    CHECK_NEAR(series.minimum_value(), 30000, 1e-12);
    CHECK_NEAR(series.maximum_value(), 50000, 1e-12);
}

// C# Test_MinIndex_UsesStartIndex
void test_threshold_series_min_index_uses_start_index() {
    ThresholdSeries series(std::vector<ThresholdData>{make_threshold(1851, 1900, 50000),
                                                      make_threshold(1800, 1850, 30000)});
    CHECK_EQ(series.minimum_index(), 1800);
}

// C# Test_MaxIndex_UsesEndIndex
void test_threshold_series_max_index_uses_end_index() {
    ThresholdSeries series(std::vector<ThresholdData>{make_threshold(1800, 1850, 30000),
                                                      make_threshold(1851, 1900, 50000)});
    CHECK_EQ(series.maximum_index(), 1900);
}

// C# Test_MinMaxIndex_Empty_ReturnsSentinels
void test_threshold_series_min_max_index_empty_returns_sentinels() {
    ThresholdSeries series;
    CHECK_EQ(series.minimum_index(), -100000);
    CHECK_EQ(series.maximum_index(), 100000);
}

// C# Test_SortByIndex_Ascending_OrdersByStartIndex
void test_threshold_series_sort_by_index_ascending_orders_by_start_index() {
    ThresholdSeries series(std::vector<ThresholdData>{make_threshold(1900, 1950, 1.0),
                                                      make_threshold(1800, 1850, 1.0),
                                                      make_threshold(1851, 1899, 1.0)});
    series.sort_by_index(SortOrder::Ascending);
    CHECK_EQ(series[0].start_index(), 1800);
    CHECK_EQ(series[1].start_index(), 1851);
    CHECK_EQ(series[2].start_index(), 1900);
}

// C# Test_Sort_Descending_OrdersByValueDescending
void test_threshold_series_sort_descending_orders_by_value() {
    ThresholdSeries series(std::vector<ThresholdData>{make_threshold(1800, 1850, 30000),
                                                      make_threshold(1851, 1900, 50000),
                                                      make_threshold(1901, 1950, 40000)});
    series.sort(SortOrder::Descending);
    CHECK_NEAR(series[0].value(), 50000, 1e-12);
    CHECK_NEAR(series[1].value(), 40000, 1e-12);
    CHECK_NEAR(series[2].value(), 30000, 1e-12);
}

// C# Test_Validate_EmptySeries_ReturnsValid
void test_threshold_series_validate_empty_returns_valid() {
    ThresholdSeries series;
    ValidationResult result = series.validate();
    CHECK_TRUE(result.is_valid);
    CHECK_EQ(result.validation_messages.size(), std::size_t{0});
}

// C# Test_Validate_NonOverlappingPeriods_ReturnsValid
void test_threshold_series_validate_non_overlapping_periods_returns_valid() {
    ThresholdSeries series(std::vector<ThresholdData>{make_threshold(1800, 1850, 30000, 1),
                                                      make_threshold(1851, 1900, 40000, 2)});
    CHECK_TRUE(series.validate().is_valid);
}

// C# Test_Validate_OverlappingPeriods_FailsValidation
void test_threshold_series_validate_overlapping_periods_fails() {
    ThresholdSeries series(std::vector<ThresholdData>{
        make_threshold(1800, 1900, 30000, 1),
        make_threshold(1850, 1950, 40000, 2)});  // overlaps 1850-1900
    ValidationResult result = series.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "overlap"));
}

// C# Test_Clone_ProducesIndependentCopy
void test_threshold_series_clone_produces_independent_copy() {
    ThresholdSeries original(std::vector<ThresholdData>{make_threshold(1800, 1850, 30000)});
    ThresholdSeries clone = original.clone();
    clone.add(make_threshold(1851, 1900, 40000));
    CHECK_EQ(original.count(), std::size_t{1});
    CHECK_EQ(clone.count(), std::size_t{2});
}

// C# Test_ToList_ReturnsClonedItems
void test_threshold_series_to_list_returns_cloned_items() {
    ThresholdSeries series(std::vector<ThresholdData>{make_threshold(1800, 1850, 30000)});
    std::vector<ThresholdData> list = series.to_list();
    list[0] = make_threshold(0, 1, 999);
    CHECK_EQ(series[0].start_index(), 1800);
}

// C# Test_ToXElement_RoundTrip_PreservesNumberAbove -- XML is not ported; the equivalent
// deep-copy assertion here is that Clone preserves NumberAbove.
void test_threshold_series_clone_preserves_number_above() {
    ThresholdSeries original(std::vector<ThresholdData>{make_threshold(1850, 1920, 40000, 5)});
    ThresholdSeries clone = original.clone();
    CHECK_EQ(clone[0].number_above(), 5);
}

// ---------------------------------------------------------------------------
// UncertainSeries (transcribed from UncertainSeriesTests.cs)
// ---------------------------------------------------------------------------

UncertainData make_uncertain(int index, double mean, double stdev) {
    return UncertainData(index, std::make_unique<Normal>(mean, stdev));
}

// C# Test_Constructor_Empty_HasZeroCount
void test_uncertain_series_constructor_empty_has_zero_count() {
    UncertainSeries series;
    CHECK_EQ(series.count(), std::size_t{0});
}

// C# Test_Constructor_FromList_ClonesEntries
void test_uncertain_series_constructor_from_list_clones_entries() {
    std::vector<UncertainData> source;
    source.push_back(make_uncertain(1889, 85000, 10000));
    UncertainSeries series(source);
    CHECK_EQ(series.count(), std::size_t{1});
    source[0] = make_uncertain(0, 0, 1);
    CHECK_EQ(series[0].index(), 1889);
}

// C# Test_MinimumValue_Empty_ReturnsMaxSentinel / Test_MaximumValue_Empty_ReturnsMinSentinel
void test_uncertain_series_min_max_value_empty_returns_sentinels() {
    UncertainSeries series;
    CHECK_NEAR(series.minimum_value(), kDoubleMax, 0.0);
    CHECK_NEAR(series.maximum_value(), kDoubleMin, 0.0);
}

// Not in C# as a populated-value test: MinimumValue/MaximumValue use the distribution's
// 5th/95th percentiles (LowerValue/UpperValue) per the C# source.
void test_uncertain_series_min_max_value_uses_percentiles() {
    UncertainSeries series;
    series.add(make_uncertain(1889, 85000, 10000));
    series.add(make_uncertain(1913, 75000, 8000));
    Normal low(75000, 8000);
    Normal high(85000, 10000);
    CHECK_NEAR(series.minimum_value(), low.inverse_cdf(0.05), 1e-6);
    CHECK_NEAR(series.maximum_value(), high.inverse_cdf(0.95), 1e-6);
}

// C# Test_MinMaxIndex_Empty_ReturnsSentinels
void test_uncertain_series_min_max_index_empty_returns_sentinels() {
    UncertainSeries series;
    CHECK_EQ(series.minimum_index(), -100000);
    CHECK_EQ(series.maximum_index(), 100000);
}

// C# Test_MinMaxIndex_Populated_ReturnsCorrectExtremes
void test_uncertain_series_min_max_index_populated() {
    UncertainSeries series;
    series.add(make_uncertain(1900, 1, 1));
    series.add(make_uncertain(1880, 1, 1));
    series.add(make_uncertain(1920, 1, 1));
    CHECK_EQ(series.minimum_index(), 1880);
    CHECK_EQ(series.maximum_index(), 1920);
}

// C# Test_SortByIndex_Ascending_OrdersByIndex
void test_uncertain_series_sort_by_index_ascending() {
    UncertainSeries series;
    series.add(make_uncertain(1900, 1, 1));
    series.add(make_uncertain(1880, 1, 1));
    series.add(make_uncertain(1920, 1, 1));
    series.sort_by_index(SortOrder::Ascending);
    CHECK_EQ(series[0].index(), 1880);
    CHECK_EQ(series[1].index(), 1900);
    CHECK_EQ(series[2].index(), 1920);
}

// C# Test_Sort_Ascending_OrdersByValue
void test_uncertain_series_sort_ascending_orders_by_value() {
    UncertainSeries series;
    series.add(make_uncertain(1900, 50000, 1));
    series.add(make_uncertain(1880, 30000, 1));
    series.add(make_uncertain(1920, 40000, 1));
    series.sort(SortOrder::Ascending);
    CHECK_NEAR(series[0].value(), 30000, 1e-9);
    CHECK_NEAR(series[1].value(), 40000, 1e-9);
    CHECK_NEAR(series[2].value(), 50000, 1e-9);
}

// C# Test_Validate_EmptySeries_NoDataFrame_ReturnsValid
void test_uncertain_series_validate_empty_no_data_frame_returns_valid() {
    UncertainSeries series;
    ValidationResult result = series.validate(nullptr);
    CHECK_TRUE(result.is_valid);
    CHECK_EQ(result.validation_messages.size(), std::size_t{0});
}

// C# Test_Validate_OverlapsWithExactSeries_FailsValidation
void test_uncertain_series_validate_overlaps_with_exact_fails() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<ExactData>{ExactData(1889, 50000)}));
    df.uncertain_series().add(make_uncertain(1889, 85000, 10000));
    ValidationResult result = df.uncertain_series().validate(&df);
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "overlaps with exact"));
}

// C# Test_Clone_ProducesIndependentCopy
void test_uncertain_series_clone_produces_independent_copy() {
    UncertainSeries original;
    original.add(make_uncertain(1889, 85000, 10000));
    UncertainSeries clone = original.clone();
    clone.add(make_uncertain(1900, 60000, 5000));
    CHECK_EQ(original.count(), std::size_t{1});
    CHECK_EQ(clone.count(), std::size_t{2});
}

// C# Test_ToList_ReturnsClonedItems
void test_uncertain_series_to_list_returns_cloned_items() {
    UncertainSeries series;
    series.add(make_uncertain(1889, 85000, 10000));
    std::vector<UncertainData> list = series.to_list();
    list[0] = make_uncertain(2000, 1, 1);
    CHECK_EQ(series[0].index(), 1889);
}

// ---------------------------------------------------------------------------
// DataFrame: Construction (transcribed from DataFrameTests.cs)
// ---------------------------------------------------------------------------

// C# Constructor_Empty_CreatesEmptyDataFrame
void test_data_frame_constructor_empty_creates_empty_data_frame() {
    DataFrame df;
    CHECK_EQ(df.exact_series().count(), std::size_t{0});
    CHECK_EQ(df.uncertain_series().count(), std::size_t{0});
    CHECK_EQ(df.interval_series().count(), std::size_t{0});
    CHECK_EQ(df.threshold_series().count(), std::size_t{0});
}

// C# Constructor_Empty_InitializesDefaultProperties
void test_data_frame_constructor_empty_initializes_default_properties() {
    DataFrame df;
    CHECK_NEAR(df.plotting_parameter(), 0.0, 0.0);  // Weibull default
    CHECK_EQ(df.number_of_low_outliers(), 0);
    CHECK_NEAR(df.low_outlier_threshold(), 0.0, 0.0);
    CHECK_NEAR(df.lambda(), 1.0, 0.0);
}

// ---------------------------------------------------------------------------
// DataFrame: series storage (transcribed from DataFrameTests.cs)
// ---------------------------------------------------------------------------

// C# ExactSeries_SetAndGet_StoresDataCorrectly
void test_data_frame_exact_series_set_and_get_stores_data_correctly() {
    DataFrame df;
    df.set_exact_series(ExactSeries(sample_annual_peaks()));
    CHECK_EQ(df.exact_series().count(), sample_annual_peaks().size());
    for (std::size_t i = 0; i < sample_annual_peaks().size(); i++) {
        CHECK_NEAR(df.exact_series()[i].value(), sample_annual_peaks()[i], 1e-10);
    }
}

// C# ExactSeries_SmallSample_StoresCorrectly
void test_data_frame_exact_series_small_sample() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>{45000, 52000, 61000, 49000, 55000}));
    CHECK_EQ(df.exact_series().count(), std::size_t{5});
}

// C# ExactSeries_SingleValue_StoresCorrectly
void test_data_frame_exact_series_single_value() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>{42000.0}));
    CHECK_EQ(df.exact_series().count(), std::size_t{1});
    CHECK_NEAR(df.exact_series()[0].value(), 42000.0, 0.0);
}

// C# ExactSeries_EmptyArray_StoresCorrectly
void test_data_frame_exact_series_empty_array() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>{}));
    CHECK_EQ(df.exact_series().count(), std::size_t{0});
}

// C# ExactSeries_ValuesToArray_ReturnsCorrectValues
void test_data_frame_exact_series_values_to_array() {
    DataFrame df = create_test_data_frame(10);
    std::vector<double> values = df.exact_series().values_to_list();
    CHECK_EQ(values.size(), std::size_t{10});
    for (std::size_t i = 0; i < 10; i++) {
        CHECK_NEAR(values[i], sample_annual_peaks()[i], 1e-10);
    }
}

// C# UncertainSeries_AddUncertainData_StoresCorrectly
void test_data_frame_uncertain_series_add_stores_correctly() {
    DataFrame df;
    df.uncertain_series().add(UncertainData(1889, std::make_unique<Normal>(85000, 10000)));
    CHECK_EQ(df.uncertain_series().count(), std::size_t{1});
    CHECK_NEAR(df.uncertain_series()[0].value(), 85000, 1e-10);
}

// C# UncertainSeries_MultipleItems_StoresCorrectly
void test_data_frame_uncertain_series_multiple_items() {
    DataFrame df;
    df.uncertain_series().add(make_uncertain(1889, 85000, 10000));
    df.uncertain_series().add(make_uncertain(1913, 75000, 8000));
    df.uncertain_series().add(make_uncertain(1927, 90000, 12000));
    CHECK_EQ(df.uncertain_series().count(), std::size_t{3});
}

// C# UncertainSeries_DifferentDistributions_StoresCorrectly
void test_data_frame_uncertain_series_different_distributions() {
    DataFrame df;
    df.uncertain_series().add(UncertainData(1889, std::make_unique<Normal>(85000, 10000)));
    df.uncertain_series().add(UncertainData(1913, std::make_unique<LogNormal>(11.2, 0.3)));
    df.uncertain_series().add(
        UncertainData(1927, std::make_unique<Triangular>(70000, 90000, 110000)));
    CHECK_EQ(df.uncertain_series().count(), std::size_t{3});
    CHECK_TRUE(df.uncertain_series()[0].distribution().type() ==
               UnivariateDistributionType::Normal);
    CHECK_TRUE(df.uncertain_series()[1].distribution().type() ==
               UnivariateDistributionType::LogNormal);
    CHECK_TRUE(df.uncertain_series()[2].distribution().type() ==
               UnivariateDistributionType::Triangular);
}

// C# IntervalSeries_AddIntervalData_StoresCorrectly
void test_data_frame_interval_series_add_stores_correctly() {
    DataFrame df;
    df.interval_series().add(IntervalData(1500, 60000, 80000, 100000));
    CHECK_EQ(df.interval_series().count(), std::size_t{1});
    CHECK_NEAR(df.interval_series()[0].lower_value(), 60000, 0.0);
    CHECK_NEAR(df.interval_series()[0].value(), 80000, 0.0);
    CHECK_NEAR(df.interval_series()[0].upper_value(), 100000, 0.0);
}

// C# IntervalSeries_MultipleItems_StoresCorrectly
void test_data_frame_interval_series_multiple_items() {
    DataFrame df;
    df.interval_series().add(IntervalData(1500, 60000, 80000, 100000));
    df.interval_series().add(IntervalData(1700, 50000, 70000, 90000));
    df.interval_series().add(IntervalData(1800, 40000, 60000, 80000));
    CHECK_EQ(df.interval_series().count(), std::size_t{3});
}

// C# ThresholdSeries_AddThresholdData_StoresCorrectly. In C# NumberBelow is derived by the
// event chain on Add; here the documented explicit trigger process_threshold_series() runs.
void test_data_frame_threshold_series_add_stores_correctly() {
    DataFrame df;
    df.threshold_series().add(make_threshold(1850, 1920, 40000, 5));
    df.process_threshold_series();
    CHECK_EQ(df.threshold_series().count(), std::size_t{1});
    const ThresholdData& stored = df.threshold_series()[0];
    CHECK_EQ(stored.start_index(), 1850);
    CHECK_EQ(stored.end_index(), 1920);
    CHECK_NEAR(stored.value(), 40000, 0.0);
    CHECK_EQ(stored.number_above(), 5);
    // Duration = 1920 - 1850 + 1 = 71. With no overlapping explicit data,
    // NumberBelow = Duration - NumberAbove = 71 - 5 = 66.
    CHECK_EQ(stored.number_below(), 66);
}

// C# ThresholdSeries_MultipleThresholds_StoresCorrectly
void test_data_frame_threshold_series_multiple_thresholds() {
    DataFrame df;
    df.threshold_series().add(make_threshold(1800, 1850, 30000, 3));
    df.threshold_series().add(make_threshold(1851, 1900, 40000, 2));
    df.threshold_series().add(make_threshold(1901, 1950, 50000, 1));
    CHECK_EQ(df.threshold_series().count(), std::size_t{3});
}

// C# ThresholdSeries_NumberAbove_StoresCorrectly
void test_data_frame_threshold_series_number_above() {
    DataFrame df;
    df.threshold_series().add(make_threshold(1800, 1900, 100000, 5));
    CHECK_EQ(df.threshold_series()[0].number_above(), 5);
}

// C# DataFrame_CombinedExactAndThreshold_StoresCorrectly
void test_data_frame_combined_exact_and_threshold() {
    DataFrame df = create_test_data_frame(20);
    df.threshold_series().add(make_threshold(1850, 1920, 35000, 3));
    CHECK_EQ(df.exact_series().count(), std::size_t{20});
    CHECK_EQ(df.threshold_series().count(), std::size_t{1});
}

// C# DataFrame_AllDataTypes_StoresCorrectly
void test_data_frame_all_data_types() {
    DataFrame df = create_mixed_data_frame();
    CHECK_TRUE(df.exact_series().count() > 0);
    CHECK_TRUE(df.uncertain_series().count() > 0);
    CHECK_TRUE(df.interval_series().count() > 0);
    CHECK_TRUE(df.threshold_series().count() > 0);
}

// ---------------------------------------------------------------------------
// DataFrame: Lambda (transcribed from DataFrameTests.cs + the C# source)
// ---------------------------------------------------------------------------

// C# Lambda_ExactDataOnly_CalculatedCorrectly (passes on the default 1.0 in C# too:
// the ExactSeries property setter does not trigger CalculateLambda).
void test_data_frame_lambda_exact_data_only() {
    DataFrame df = create_test_data_frame(30);
    CHECK_NEAR(df.lambda(), 1.0, 0.01);
}

// Not in C# as a test; transcribes DataFrame.CalculateLambda (events / index span).
void test_data_frame_calculate_lambda() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<ExactData>{
        ExactData(1985, 1.0), ExactData(1990, 2.0), ExactData(1995, 3.0)}));
    df.calculate_lambda();
    CHECK_NEAR(df.lambda(), 3.0 / 11.0, 1e-12);  // 3 events over span 11

    DataFrame empty;
    empty.calculate_lambda();
    CHECK_NEAR(empty.lambda(), 0.0, 0.0);  // no events -> 0

    DataFrame df2;
    df2.set_exact_series(ExactSeries(sample_annual_peaks()));  // indexes 0..29
    df2.calculate_lambda();
    CHECK_NEAR(df2.lambda(), 1.0, 1e-12);

    // SetLambda overrides directly.
    df2.set_lambda(2.5);
    CHECK_NEAR(df2.lambda(), 2.5, 0.0);
}

// ---------------------------------------------------------------------------
// DataFrame: TotalRecordLength (transcribed from DataFrameTests.cs)
// ---------------------------------------------------------------------------

// C# TotalRecordLength_ExactDataOnly_ReturnsCorrectValue
void test_data_frame_total_record_length_exact_only() {
    DataFrame df = create_test_data_frame(30);
    CHECK_EQ(df.total_record_length(), 30);
}

// C# TotalRecordLength_WithThresholdData_ReturnsCorrectValue
void test_data_frame_total_record_length_with_threshold() {
    DataFrame df = create_test_data_frame(30);
    df.threshold_series().add(make_threshold(1850, 1920, 40000, 3));
    df.process_threshold_series();
    int total = df.total_record_length();
    CHECK_TRUE(total > 30);
    // Derived from the C# source: exact indexes 0..29 do not overlap 1850-1920, so
    // NumberBelow = 71 - 3 = 68 and total = 30 + 68 + 3 = 101.
    CHECK_EQ(total, 101);
}

// ---------------------------------------------------------------------------
// DataFrame: ZeroValueRelativeFrequency (transcribes the C# source; no upstream test)
// ---------------------------------------------------------------------------

void test_data_frame_zero_value_relative_frequency() {
    DataFrame empty;
    CHECK_NEAR(empty.zero_value_relative_frequency(), 0.0, 0.0);

    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>{-10, -5, 0, 5, 10}));
    CHECK_NEAR(df.zero_value_relative_frequency(), 3.0 / 5.0, 1e-15);

    // Uncertain and interval values enter both numerator and denominator.
    df.uncertain_series().add(make_uncertain(1900, -3.0, 1.0));  // mean <= 0 counts
    df.interval_series().add(IntervalData(1901, 1.0, 2.0, 3.0));  // value > 0
    CHECK_NEAR(df.zero_value_relative_frequency(), 4.0 / 7.0, 1e-15);
}

// ---------------------------------------------------------------------------
// DataFrame: ProcessThresholdSeries (transcribes the C# source; NumberBelow
// derivations per the DataFrameTests remarks)
// ---------------------------------------------------------------------------

void test_data_frame_process_threshold_no_overlap() {
    DataFrame df;
    df.threshold_series().add(make_threshold(1850, 1920, 40000, 5));
    df.process_threshold_series();
    CHECK_EQ(df.threshold_series()[0].number_above(), 5);
    CHECK_EQ(df.threshold_series()[0].number_below(), 66);  // 71 - 5
}

void test_data_frame_process_threshold_overlapping_points_reduce_number_below() {
    DataFrame df;
    // Exact points at 1850, 1851, 1852 fall inside the threshold window.
    df.set_exact_series(ExactSeries(std::vector<ExactData>{
        ExactData(1850, 1.0), ExactData(1851, 2.0), ExactData(1852, 3.0)}));
    df.uncertain_series().add(make_uncertain(1900, 50000, 1000));   // inside
    df.interval_series().add(IntervalData(1700, 1.0, 2.0, 3.0));    // outside
    df.threshold_series().add(make_threshold(1850, 1920, 40000, 5));
    df.process_threshold_series();
    // NumberBelow = 71 - 5 - (3 exact + 1 uncertain overlapping) = 62.
    CHECK_EQ(df.threshold_series()[0].number_above(), 5);
    CHECK_EQ(df.threshold_series()[0].number_below(), 62);
}

void test_data_frame_process_threshold_zeroes_number_above_when_fully_covered() {
    DataFrame df;
    // Duration 5, NumberAbove 2, three overlapping exact points: nBelow = 5 - 2 - 3 = 0
    // -> the C# zeroes NumberAbove ("all years are accounted for by explicit data").
    df.set_exact_series(ExactSeries(std::vector<ExactData>{
        ExactData(2000, 1.0), ExactData(2001, 2.0), ExactData(2002, 3.0)}));
    df.threshold_series().add(make_threshold(2000, 2004, 100.0, 2));
    df.process_threshold_series();
    CHECK_EQ(df.threshold_series()[0].number_above(), 0);
    CHECK_EQ(df.threshold_series()[0].number_below(), 0);
}

void test_data_frame_process_threshold_clamps_negative_number_below() {
    DataFrame df;
    // Duration 5, NumberAbove 2, five overlapping exact points: nBelow = 5 - 2 - 5 = -2
    // -> NumberAbove is kept (nBelow != 0), NumberBelow clamps to 0.
    df.set_exact_series(ExactSeries(std::vector<ExactData>{
        ExactData(2000, 1.0), ExactData(2001, 2.0), ExactData(2002, 3.0),
        ExactData(2003, 4.0), ExactData(2004, 5.0)}));
    df.threshold_series().add(make_threshold(2000, 2004, 100.0, 2));
    df.process_threshold_series();
    CHECK_EQ(df.threshold_series()[0].number_above(), 2);
    CHECK_EQ(df.threshold_series()[0].number_below(), 0);
}

// BestFit v2.0.0 (ThresholdData source/effective count split, T12): repeated calls to
// process_threshold_series() must be idempotent. Before the split, a second call would
// re-derive NumberAbove from the ALREADY-REDUCED effective NumberAbove left by the first
// call rather than the user's original input, permanently losing the exceedance count once
// explicit data fully covered the window. Same scenario as
// test_data_frame_process_threshold_zeroes_number_above_when_fully_covered above (Duration
// 5, NumberAbove 2, three overlapping exact points -> NumberAbove zeroed on the first
// pass); a SECOND call must reproduce exactly the same effective state, not zero out
// anything further or diverge.
void test_data_frame_process_threshold_series_is_idempotent() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<ExactData>{
        ExactData(2000, 1.0), ExactData(2001, 2.0), ExactData(2002, 3.0)}));
    df.threshold_series().add(make_threshold(2000, 2004, 100.0, 2));

    df.process_threshold_series();
    CHECK_EQ(df.threshold_series()[0].source_number_above(), 2);  // stable user input
    CHECK_EQ(df.threshold_series()[0].number_above(), 0);         // fully covered -> zeroed
    CHECK_EQ(df.threshold_series()[0].number_below(), 0);

    df.process_threshold_series();  // second pass: must reproduce the SAME state
    CHECK_EQ(df.threshold_series()[0].source_number_above(), 2);
    CHECK_EQ(df.threshold_series()[0].number_above(), 0);
    CHECK_EQ(df.threshold_series()[0].number_below(), 0);

    // Removing the overlapping data and reprocessing must RESTORE the original exceedance
    // count from the stable source input -- proof the reduction isn't a one-way ratchet.
    df.set_exact_series(ExactSeries());
    df.process_threshold_series();
    CHECK_EQ(df.threshold_series()[0].source_number_above(), 2);
    CHECK_EQ(df.threshold_series()[0].number_above(), 2);
    CHECK_EQ(df.threshold_series()[0].number_below(), 3);  // Duration 5 - NumberAbove 2
}

// ---------------------------------------------------------------------------
// DataFrame: FullTimeSeries expansion (the DataFrame-state part of
// ThresholdLikelihoodGuardTests.NonstationaryUnivariateThresholdLikelihood_
// SplitThresholdsRemainFinite; log-likelihood assertions are M8)
// ---------------------------------------------------------------------------

void test_data_frame_full_time_series_splits_threshold_into_single_counts() {
    DataFrame df;
    df.threshold_series().add(make_threshold(2000, 2002, 100.0, 1));
    df.process_threshold_series();  // NumberBelow = 3 - 1 = 2
    CHECK_EQ(df.threshold_series()[0].number_below(), 2);

    const auto& full = df.full_time_series();
    CHECK_EQ(full.size(), std::size_t{3});
    // Left (below) thresholds at 2000 and 2001; right (above) threshold at 2002.
    for (std::size_t i = 0; i < full.size(); i++) {
        const auto* t = dynamic_cast<const ThresholdData*>(full[i].get());
        CHECK_TRUE(t != nullptr);
        if (t == nullptr) continue;
        CHECK_EQ(t->start_index(), 2000 + static_cast<int>(i));
        CHECK_EQ(t->end_index(), 2000 + static_cast<int>(i));
        CHECK_EQ(t->number_above() + t->number_below(), 1);
    }
    const auto* first = dynamic_cast<const ThresholdData*>(full[0].get());
    const auto* last = dynamic_cast<const ThresholdData*>(full[2].get());
    CHECK_EQ(first->number_below(), 1);
    CHECK_EQ(first->number_above(), 0);
    CHECK_EQ(last->number_below(), 0);
    CHECK_EQ(last->number_above(), 1);
}

// Transcribes CreateFullTimeSeries element-by-element for the mixed frame (derived
// from the C# source; no single upstream test asserts the full contents).
void test_data_frame_full_time_series_mixed_contents_and_ordering() {
    DataFrame df = create_mixed_data_frame();
    // NumberBelow = 71 - 5 - (2 uncertain at 1889/1913 inside 1850-1920) = 64.
    CHECK_EQ(df.threshold_series()[0].number_below(), 64);
    CHECK_EQ(df.total_record_length(), 83);  // 10 + 2 + 2 + 64 + 5

    df.create_full_time_series();
    const auto& full = df.full_time_series();
    CHECK_EQ(full.size(), std::size_t{83});

    // Chronological ordering by index.
    for (std::size_t i = 1; i < full.size(); i++) {
        CHECK_TRUE(full[i - 1]->index() <= full[i]->index());
    }

    // Threshold expansion skips indexes occupied by explicit data (1889 and 1913).
    int n_threshold = 0, n_exact = 0, n_uncertain = 0, n_interval = 0;
    bool occupied_indexes_are_not_thresholds = true;
    for (const auto& item : full) {
        if (const auto* t = dynamic_cast<const ThresholdData*>(item.get())) {
            n_threshold++;
            CHECK_EQ(t->number_above() + t->number_below(), 1);
            if (t->index() == 1889 || t->index() == 1913)
                occupied_indexes_are_not_thresholds = false;
        } else if (dynamic_cast<const ExactData*>(item.get())) {
            n_exact++;
        } else if (dynamic_cast<const UncertainData*>(item.get())) {
            n_uncertain++;
        } else if (dynamic_cast<const IntervalData*>(item.get())) {
            n_interval++;
        }
    }
    CHECK_TRUE(occupied_indexes_are_not_thresholds);
    CHECK_EQ(n_threshold, 69);  // 64 below + 5 above
    CHECK_EQ(n_exact, 10);
    CHECK_EQ(n_uncertain, 2);
    CHECK_EQ(n_interval, 2);
}

// The lazy full_time_series() getter rebuilds when the cached size no longer matches
// TotalRecordLength (the C# getter's own rebuild condition, minus the locking).
void test_data_frame_full_time_series_lazy_rebuild() {
    DataFrame empty;
    CHECK_EQ(empty.full_time_series().size(), std::size_t{0});

    DataFrame df = create_mixed_data_frame();
    CHECK_EQ(df.full_time_series().size(), std::size_t{83});
    df.exact_series().add(ExactData(2024, 75000.0));
    CHECK_EQ(df.total_record_length(), 84);
    CHECK_EQ(df.full_time_series().size(), std::size_t{84});
}

// ---------------------------------------------------------------------------
// DataFrame: low outliers (transcribed from ExactDataHypothesisTests.cs)
// ---------------------------------------------------------------------------

// The USGS station records; oracle counts come from HEC-SSP (C# Test_MultipleGrubbsBeck).
const std::vector<double>& mgbt_sample() {
    static const std::vector<double> v = {
        423.823157583651, 769.119600544858, 1420.9840065658,  81.1048635557593,
        279.928123967548, 75.85549285788,   545.403510679849, 2765.04138262183,
        99.2328081106953, 1151.90527161336, 35.0188163971524, 93.2163892297505,
        174.065209255604, 284.811439281534, 69.4129231117978, 1393.70251526941,
        366.909211754559, 57.3577922448949, 507.512883027978, 3408.66910217811,
        994.625641160531, 99.0457917640901, 253.32702656322,  155.691675526921,
        644.834237627036, 81.0277133561875, 655.861119414072, 49.0010266733043,
        216.450613452697, 625.639165872462};
    return v;
}

const std::vector<double>& mgbt_blu() {
    static const std::vector<double> v = {
        12903,  10108,  7401.3, 7233.3, 7167.3, 7116.3, 6930.3, 6929,   6870,   6768,
        6742.7, 6213.3, 6166.3, 6071.7, 6044.3, 5857,   5779,   5289.7, 5247.3, 5243,
        5208.3, 5050.3, 4887.7, 4821.3, 4801,   4789.7, 4713,   4426.7, 4308.7, 4265,
        4263.3, 4199,   4138.7, 4093.3, 4035,   4023,   3964,   3929.7, 3826.3, 3698.3,
        3667.7, 3654,   3647.3, 3644,   3578,   3532,   3511.3, 3510.7, 3463.3, 3374.3,
        3349,   3312.3, 3263,   3226,   3126.7, 3119.3, 3084,   3081,   3054.3, 3045.7,
        2968.7, 2964,   2907.3, 2889.3, 2850.3, 2759.7, 2663,   2643,   2631.3, 2596.3,
        2580.3, 2433.7, 2295.3, 2095.3, 2024,   1963,   1938.7, 1785.3, 1266.7, 1142.3,
        74.7};
    return v;
}

const std::vector<double>& mgbt_hck() {
    static const std::vector<double> v = {
        23520, 16220, 15290, 14750, 13980, 13740, 13030, 12930, 12550, 12140, 11470,
        10610, 10220, 10060, 10010, 9880,  9090,  8780,  8700,  8420,  8410,  8280,
        8140,  7990,  7770,  7540,  7460,  7400,  7400,  6770,  6590,  6500,  6420,
        6200,  6190,  6160,  6100,  5900,  5800,  5730,  5690,  5670,  5630,  5620,
        5590,  5580,  5560,  5440,  5410,  5340,  5330,  5310,  5150,  5080,  4690,
        4680,  4620,  4600,  4580,  4490,  4380,  4110,  4080,  4050,  4020,  4010,
        3980,  3960,  3900,  3840,  3770,  3700,  3670,  3620,  3300,  3220,  2820,
        2810,  2670,  2440,  2370,  2030,  1750,  1570,  920};
    return v;
}

const std::vector<double>& mgbt_lop() {
    static const std::vector<double> v = {
        60350, 44630, 38800, 34640, 34510, 34050, 33270, 32570, 29750, 28080, 27860,
        27210, 26930, 26920, 26580, 24320, 24260, 24060, 23630, 23530, 23100, 21970,
        21930, 21660, 21270, 21260, 20520, 19370, 19370, 18800, 17970, 17630, 17420,
        16240, 16060, 16010, 15980, 15970, 15850, 15810, 15700, 15640, 15420, 15150,
        15100, 15090, 14890, 14820, 14580, 14130, 14120, 14090, 13460, 13350, 13350,
        13330, 13180, 12930, 12740, 12610, 12580, 12350, 11900, 11650, 11420, 11390,
        11250, 11120, 11050, 10580, 10180, 9890,  9710,  9130,  8780,  8390,  8180,
        8110,  6980,  6230,  6150,  6000,  4630,  4560,  2840};
    return v;
}

// C# Test_MultipleGrubbsBeck (SetLowOutliersFromMGBT). The full 8-station sweep lives in
// test_multiple_grubbs_beck.cpp against MultipleGrubbsBeckTest::function; this test
// exercises the DataFrame wrapper on a representative subset (BLU=3, HCK=1, LOP=1,
// sample=0) plus the threshold/flag side effects the C# method adds on top.
void test_data_frame_set_low_outliers_from_mgbt() {
    DataFrame df;
    df.set_exact_series(ExactSeries(mgbt_blu()));
    df.set_low_outliers_from_mgbt();
    CHECK_EQ(df.number_of_low_outliers(), 3);
    // Threshold = first sorted value larger than the N flagged outliers.
    CHECK_NEAR(df.low_outlier_threshold(), 1785.3, 1e-9);
    int flagged = 0;
    for (std::size_t i = 0; i < df.exact_series().count(); i++) {
        if (df.exact_series()[i].is_low_outlier()) flagged++;
    }
    CHECK_EQ(flagged, 3);

    df.set_exact_series(ExactSeries(mgbt_hck()));
    df.set_low_outliers_from_mgbt();
    CHECK_EQ(df.number_of_low_outliers(), 1);

    df.set_exact_series(ExactSeries(mgbt_lop()));
    df.set_low_outliers_from_mgbt();
    CHECK_EQ(df.number_of_low_outliers(), 1);

    df.set_exact_series(ExactSeries(mgbt_sample()));
    df.set_low_outliers_from_mgbt();
    CHECK_EQ(df.number_of_low_outliers(), 0);
    CHECK_NEAR(df.low_outlier_threshold(), 0.0, 0.0);  // no outliers -> threshold 0
}

// C# Test_Threshold (SetLowOutliersFromThreshold), same representative subset.
void test_data_frame_set_low_outliers_from_threshold() {
    DataFrame df;
    df.set_exact_series(ExactSeries(mgbt_blu()));
    df.set_low_outlier_threshold(1785.3);
    df.set_low_outliers_from_threshold();
    CHECK_EQ(df.number_of_low_outliers(), 3);

    df.set_exact_series(ExactSeries(mgbt_hck()));
    df.set_low_outlier_threshold(1570);
    df.set_low_outliers_from_threshold();
    CHECK_EQ(df.number_of_low_outliers(), 1);

    df.set_exact_series(ExactSeries(mgbt_lop()));
    df.set_low_outlier_threshold(4560);
    df.set_low_outliers_from_threshold();
    CHECK_EQ(df.number_of_low_outliers(), 1);

    df.set_exact_series(ExactSeries(mgbt_sample()));
    df.set_low_outlier_threshold(0);
    df.set_low_outliers_from_threshold();
    CHECK_EQ(df.number_of_low_outliers(), 0);
}

// Transcribes the C# guard clauses (ArgumentException -> std::invalid_argument).
void test_data_frame_low_outlier_guards() {
    DataFrame small;
    small.set_exact_series(ExactSeries(std::vector<double>{1, 2, 3}));  // < 10 items
    CHECK_THROWS(small.set_low_outliers_from_mgbt());
    CHECK_THROWS(small.set_low_outliers_from_threshold());

    // Threshold above the upper-middle value would censor more than 50 percent.
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    df.set_low_outlier_threshold(9.0);
    CHECK_THROWS(df.set_low_outliers_from_threshold());
}

// Transcribes ClearLowOutliers.
void test_data_frame_clear_low_outliers() {
    DataFrame df;
    df.set_exact_series(ExactSeries(mgbt_blu()));
    df.set_low_outliers_from_mgbt();
    CHECK_EQ(df.number_of_low_outliers(), 3);
    df.clear_low_outliers();
    CHECK_EQ(df.number_of_low_outliers(), 0);
    for (std::size_t i = 0; i < df.exact_series().count(); i++) {
        CHECK_TRUE(!df.exact_series()[i].is_low_outlier());
    }
}

// ---------------------------------------------------------------------------
// DataFrame: PlottingParameter property (the plain property only; the
// CalculatePlottingPositions side effect is the M5 stub)
// ---------------------------------------------------------------------------

// C# DataFrame_VariousPlottingParameters_WorksCorrectly (DataRow sweep)
void test_data_frame_various_plotting_parameters() {
    const double params[] = {0.0, 0.40, 0.44, 0.50};  // Weibull/Cunnane/Gringorten/Hazen
    for (double p : params) {
        DataFrame df = create_test_data_frame();
        df.set_plotting_parameter(p);
        CHECK_NEAR(df.plotting_parameter(), p, 0.0);
    }
}

// ---------------------------------------------------------------------------
// DataFrame: Validate / Clone
// ---------------------------------------------------------------------------

// C# Test_PlottingPositions_InvalidInputs_Throw (BestFit v2.0.0): set_plotting_parameter()
// now validates eagerly and throws (mirrors ArgumentOutOfRangeException) rather than
// leaving an out-of-range value for DataFrame::validate() to catch later -- so an
// out-of-range/non-finite value can no longer reach the object at all through the public
// setter. DataFrame::validate()'s own plotting-parameter check (still ported, for
// structural parity with the C#) is consequently unreachable via this port's public API;
// see its header comment.
void test_data_frame_set_plotting_parameter_rejects_out_of_range() {
    DataFrame df = create_test_data_frame(10);
    CHECK_TRUE(df.validate().is_valid);
    CHECK_THROWS(df.set_plotting_parameter(1.5));
    CHECK_THROWS(df.set_plotting_parameter(-0.01));
    CHECK_THROWS(df.set_plotting_parameter(1.0));  // half-open: [0, 1)
    CHECK_THROWS(df.set_plotting_parameter(std::numeric_limits<double>::quiet_NaN()));
    CHECK_THROWS(df.set_plotting_parameter(std::numeric_limits<double>::infinity()));
    // A rejected assignment must leave the prior (valid) state untouched.
    CHECK_NEAR(df.plotting_parameter(), 0.0, 0.0);
    CHECK_TRUE(df.validate().is_valid);
}

// Validate aggregates the four series validations (exercised through the overlap rule).
void test_data_frame_validate_aggregates_series_validations() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<ExactData>{ExactData(1500, 70000)}));
    df.interval_series().add(IntervalData(1500, 60000, 80000, 100000));
    ValidationResult result = df.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "overlaps with exact"));
}

// The C# Clone() round-trips through XElement; the port is a direct deep clone with the
// same observable result (series contents + the scalar properties, lazy full series).
void test_data_frame_clone_is_deep_and_preserves_state() {
    DataFrame original = create_mixed_data_frame();
    original.set_plotting_parameter(0.44);
    original.set_low_outlier_threshold(15000.0);
    original.set_lambda(0.75);

    DataFrame clone = original.clone();
    CHECK_EQ(clone.exact_series().count(), original.exact_series().count());
    CHECK_EQ(clone.uncertain_series().count(), original.uncertain_series().count());
    CHECK_EQ(clone.interval_series().count(), original.interval_series().count());
    CHECK_EQ(clone.threshold_series().count(), original.threshold_series().count());
    CHECK_NEAR(clone.plotting_parameter(), 0.44, 0.0);
    CHECK_NEAR(clone.low_outlier_threshold(), 15000.0, 0.0);
    CHECK_NEAR(clone.lambda(), 0.75, 0.0);
    CHECK_EQ(clone.threshold_series()[0].number_below(), 64);  // derived state carried

    // Deep: mutating the clone leaves the original untouched.
    clone.exact_series().add(ExactData(2024, 1.0));
    clone.exact_series()[0].set_value(-1.0);
    CHECK_EQ(original.exact_series().count(), std::size_t{10});
    CHECK_NEAR(original.exact_series()[0].value(), 45000.0, 0.0);

    // The clone starts with an empty full series and lazily rebuilds (C# remark).
    CHECK_EQ(clone.full_time_series().size(), std::size_t{84});
}

// ---------------------------------------------------------------------------
// DataFrame: Edge cases (transcribed from DataFrameTests.cs)
// ---------------------------------------------------------------------------

// C# DataFrame_WithNegativeValues_HandlesCorrectly
void test_data_frame_with_negative_values() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>{-10, -5, 0, 5, 10, 15, 20, 25, 30, 35}));
    std::vector<double> values = df.exact_series().values_to_list();
    bool any_negative = false;
    for (double v : values) any_negative = any_negative || v < 0;
    CHECK_TRUE(any_negative);
}

// C# DataFrame_LargeDataset_HandlesCorrectly
void test_data_frame_large_dataset() {
    DataFrame df;
    std::vector<double> large(1000);
    for (std::size_t i = 0; i < large.size(); i++)
        large[i] = 30000.0 + static_cast<double>(i + 1) * 100.0;
    df.set_exact_series(ExactSeries(large));
    CHECK_EQ(df.exact_series().count(), std::size_t{1000});
}

// C# DataFrame_SpecialDoubleValues_HandlesCorrectly
void test_data_frame_special_double_values() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>{1.0, 2.0, kNaN, 4.0, kInf}));
    CHECK_EQ(df.exact_series().count(), std::size_t{5});
    CHECK_TRUE(std::isnan(df.exact_series().values_to_list()[2]));
}

// C# DataFrame_VerySmallValues_HandlesCorrectly
void test_data_frame_very_small_values() {
    DataFrame df;
    df.set_exact_series(ExactSeries(
        std::vector<double>{1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}));
    CHECK_EQ(df.exact_series().count(), std::size_t{10});
    CHECK_NEAR(df.exact_series()[0].value(), 1e-10, 1e-15);
}

// C# DataFrame_VeryLargeValues_HandlesCorrectly
void test_data_frame_very_large_values() {
    DataFrame df;
    df.set_exact_series(ExactSeries(
        std::vector<double>{1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19}));
    CHECK_EQ(df.exact_series().count(), std::size_t{10});
    CHECK_NEAR(df.exact_series()[9].value(), 1e19, 1e14);
}

// C# DataFrame_IdenticalValues_HandlesCorrectly
void test_data_frame_identical_values() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<double>(20, 50000.0)));
    CHECK_EQ(df.exact_series().count(), std::size_t{20});
    bool all_equal = true;
    for (std::size_t i = 0; i < df.exact_series().count(); i++)
        all_equal = all_equal && df.exact_series()[i].value() == 50000.0;
    CHECK_TRUE(all_equal);
}

// C# DataFrame_ClearAndRebuild_WorksCorrectly
void test_data_frame_clear_and_rebuild() {
    DataFrame df = create_mixed_data_frame();
    df.exact_series().clear();
    CHECK_EQ(df.exact_series().count(), std::size_t{0});
    df.set_exact_series(
        ExactSeries(std::vector<double>{100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}));
    CHECK_EQ(df.exact_series().count(), std::size_t{10});
}

// ---------------------------------------------------------------------------
// DataFrame: flood frequency scenarios (transcribed from DataFrameTests.cs)
// ---------------------------------------------------------------------------

// C# Scenario_SystematicRecord_WorksCorrectly (the C# fills values from a seeded
// System.Random; any deterministic positive values satisfy the same assertions).
void test_data_frame_scenario_systematic_record() {
    DataFrame df;
    for (int year = 1970; year < 2020; year++) {
        double value = 30000.0 + static_cast<double>((year * 997) % 40000);
        df.exact_series().add(ExactData(year, value));
    }
    CHECK_EQ(df.exact_series().count(), std::size_t{50});
    CHECK_TRUE(df.validate().is_valid);
}

// C# Scenario_SystematicWithHistorical_WorksCorrectly
void test_data_frame_scenario_systematic_with_historical() {
    DataFrame df = create_test_data_frame(30);
    df.uncertain_series().add(UncertainData(1889, std::make_unique<Normal>(95000, 15000)));
    df.threshold_series().add(ThresholdData(1850, 1920, 80000));
    CHECK_EQ(df.exact_series().count(), std::size_t{30});
    CHECK_EQ(df.uncertain_series().count(), std::size_t{1});
    CHECK_EQ(df.threshold_series().count(), std::size_t{1});
}

// C# Scenario_Paleoflood_WorksCorrectly
void test_data_frame_scenario_paleoflood() {
    DataFrame df = create_test_data_frame(30);
    df.interval_series().add(IntervalData(1200, 80000, 100000, 120000));
    df.interval_series().add(IntervalData(1450, 90000, 110000, 130000));
    df.interval_series().add(IntervalData(1650, 70000, 90000, 110000));
    df.threshold_series().add(ThresholdData(1000, 2000, 150000));
    CHECK_EQ(df.exact_series().count(), std::size_t{30});
    CHECK_EQ(df.interval_series().count(), std::size_t{3});
    CHECK_EQ(df.threshold_series().count(), std::size_t{1});
}

// ===================== P6: time-series import =====================
// The four ExactDataProcessTests methods that could not run through P5 because
// CreateBlockSeries / CreatePeaksOverThresholdSeries needed the unported TimeSeries container.
// Transcribed from src/RMC.BestFit.Tests/DataFrame/ExactDataProcessTests.cs @ c2e6192
// (Test_CalendarYearSeries, Test_WaterYearSeries, Test_CustomYearSeries, Test_PeaksOverThreshold);
// its fifth method, Test_USGS*, is the network-import severance.
namespace p6_import {

using corehydro::numerics::data::BlockFunctionType;
using corehydro::numerics::data::DateTime;
using corehydro::numerics::data::SmoothingFunctionType;
using corehydro::numerics::data::TimeBlockWindow;
using corehydro::numerics::data::TimeInterval;
using corehydro::numerics::data::TimeSeries;

const std::vector<double> kValues = {
    122,  244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331, 225, 174, 98.8, 149,
    238,  262, 132, 235, 216, 240, 230, 192, 195, 172, 173, 172, 153, 142, 317, 161, 201,  204,
    194,  164, 183, 161, 167, 179, 185, 117, 192, 337, 125, 166, 99.1, 202, 230, 158, 262, 154,
    164,  182, 164, 183, 171, 250, 184, 205, 237, 177, 239, 187, 180, 173, 174};

TimeSeries monthly() { return TimeSeries(TimeInterval::OneMonth, DateTime(2023, 1, 1), kValues); }

void check_exact_values(const DataFrame& df, const std::vector<double>& expected) {
    CHECK_EQ(static_cast<int>(df.exact_series().count()), static_cast<int>(expected.size()));
    for (std::size_t i = 0; i < expected.size(); ++i)
        CHECK_NEAR(df.exact_series()[i].value(), expected[i], 0.0);
}

void test_create_block_series() {
    TimeSeries ts = monthly();

    DataFrame calendar;
    calendar.create_block_series(ts, TimeBlockWindow::CalendarYear, BlockFunctionType::Maximum,
                                 SmoothingFunctionType::None, 1, 12, 1);
    check_exact_values(calendar, {263, 331, 317, 337, 262, 239});

    DataFrame water;
    water.create_block_series(ts, TimeBlockWindow::WaterYear, BlockFunctionType::Maximum,
                              SmoothingFunctionType::None, 10, 9, 1);
    check_exact_values(water, {263, 331, 317, 204, 337, 250});

    DataFrame custom;
    custom.create_block_series(ts, TimeBlockWindow::CustomYear, BlockFunctionType::Maximum,
                               SmoothingFunctionType::None, 1, 12, 1);
    check_exact_values(custom, {263, 331, 317, 337, 262, 239});
    custom.create_block_series(ts, TimeBlockWindow::CustomYear, BlockFunctionType::Maximum,
                               SmoothingFunctionType::None, 10, 3, 1);
    check_exact_values(custom, {244, 331, 240, 204, 337, 250});

    // corehydro supplement: each ordinate's index is its observation's YEAR, its date rides
    // along, and lambda is pinned to 1 (one event per block, by construction).
    CHECK_EQ(calendar.exact_series()[0].index(), 2023);
    CHECK_EQ(calendar.exact_series()[0].date_time().year(), 2023);
    CHECK_TRUE(calendar.exact_series()[0].date_time() != DateTime());
    CHECK_NEAR(calendar.lambda(), 1.0, 0.0);
}

void test_create_peaks_over_threshold_series() {
    TimeSeries ts = monthly();
    DataFrame df;

    df.create_peaks_over_threshold_series(ts, 100, 2);
    check_exact_values(df, {331, 337, 262});
    df.create_peaks_over_threshold_series(ts, 90, 1);
    check_exact_values(df, {337});
    df.create_peaks_over_threshold_series(ts, 150, 5);
    check_exact_values(df, {331, 240, 317, 337});
    df.create_peaks_over_threshold_series(ts, 200, 2);
    check_exact_values(df, {263, 331, 262, 317, 337, 250});

    // corehydro supplement: unlike the block series, lambda here is the observed RATE -- six
    // events over the record's six-year span.
    CHECK_NEAR(df.lambda(), 6.0 / 6.0, 1e-12);
}

}  // namespace p6_import

}  // namespace

int main() {
    // DataSeries base surface
    test_series_add_insert_remove_clear();
    test_series_helper_lists();

    // ExactSeries (ExactSeriesTests.cs)
    test_exact_series_constructor_empty_has_zero_count();
    test_exact_series_constructor_from_values_populates_sequential_indexes();
    test_exact_series_constructor_from_list_clones_entries();
    test_exact_series_median_odd_count_returns_middle_value();
    test_exact_series_median_even_count_returns_average_of_two_middle();
    test_exact_series_median_empty_returns_zero();
    test_exact_series_median_single_value_returns_that_value();
    test_exact_series_upper_middle_even_count_returns_value_at_mid_index();
    test_exact_series_upper_middle_odd_count_equals_median();
    test_exact_series_min_max_value_empty_returns_sentinels();
    test_exact_series_min_max_value_populated_returns_correct_extremes();
    test_exact_series_min_max_index_empty_returns_sentinels();
    test_exact_series_index_span_empty_returns_zero();
    test_exact_series_index_span_populated();
    test_exact_series_unique_indices_counts_distinct_indexes();
    test_exact_series_sort_by_index_ascending();
    test_exact_series_sort_by_index_descending();
    test_exact_series_sort_ascending_orders_by_value();
    test_exact_series_sort_descending_orders_by_value();
    test_exact_series_validate_empty_returns_valid();
    test_exact_series_validate_duplicate_indexes_rejected_when_enforced();
    test_exact_series_validate_duplicate_indexes_accepted_when_not_enforced();
    test_exact_series_clone_produces_independent_copy();
    test_exact_series_to_list_returns_cloned_items();
    test_exact_series_values_to_array_returns_values_in_order();

    // IntervalSeries (IntervalSeriesTests.cs)
    test_interval_series_constructor_empty_has_zero_count();
    test_interval_series_constructor_from_list_clones_entries();
    test_interval_series_min_max_value_empty_returns_sentinels();
    test_interval_series_minimum_value_uses_lower_value();
    test_interval_series_maximum_value_uses_upper_value();
    test_interval_series_min_max_index_empty_returns_sentinels();
    test_interval_series_sort_by_index_ascending();
    test_interval_series_sort_ascending_orders_by_value();
    test_interval_series_validate_empty_no_data_frame_returns_valid();
    test_interval_series_validate_overlaps_with_exact_fails();
    test_interval_series_clone_produces_independent_copy();
    test_interval_series_to_list_returns_cloned_items();

    // ThresholdSeries (ThresholdSeriesTests.cs)
    test_threshold_series_constructor_empty_has_zero_count();
    test_threshold_series_constructor_from_list_clones_entries();
    test_threshold_series_min_max_value_empty_returns_sentinels();
    test_threshold_series_min_max_value_populated();
    test_threshold_series_min_index_uses_start_index();
    test_threshold_series_max_index_uses_end_index();
    test_threshold_series_min_max_index_empty_returns_sentinels();
    test_threshold_series_sort_by_index_ascending_orders_by_start_index();
    test_threshold_series_sort_descending_orders_by_value();
    test_threshold_series_validate_empty_returns_valid();
    test_threshold_series_validate_non_overlapping_periods_returns_valid();
    test_threshold_series_validate_overlapping_periods_fails();
    test_threshold_series_clone_produces_independent_copy();
    test_threshold_series_to_list_returns_cloned_items();
    test_threshold_series_clone_preserves_number_above();

    // UncertainSeries (UncertainSeriesTests.cs)
    test_uncertain_series_constructor_empty_has_zero_count();
    test_uncertain_series_constructor_from_list_clones_entries();
    test_uncertain_series_min_max_value_empty_returns_sentinels();
    test_uncertain_series_min_max_value_uses_percentiles();
    test_uncertain_series_min_max_index_empty_returns_sentinels();
    test_uncertain_series_min_max_index_populated();
    test_uncertain_series_sort_by_index_ascending();
    test_uncertain_series_sort_ascending_orders_by_value();
    test_uncertain_series_validate_empty_no_data_frame_returns_valid();
    test_uncertain_series_validate_overlaps_with_exact_fails();
    test_uncertain_series_clone_produces_independent_copy();
    test_uncertain_series_to_list_returns_cloned_items();

    // DataFrame (DataFrameTests.cs)
    test_data_frame_constructor_empty_creates_empty_data_frame();
    test_data_frame_constructor_empty_initializes_default_properties();
    test_data_frame_exact_series_set_and_get_stores_data_correctly();
    test_data_frame_exact_series_small_sample();
    test_data_frame_exact_series_single_value();
    test_data_frame_exact_series_empty_array();
    test_data_frame_exact_series_values_to_array();
    test_data_frame_uncertain_series_add_stores_correctly();
    test_data_frame_uncertain_series_multiple_items();
    test_data_frame_uncertain_series_different_distributions();
    test_data_frame_interval_series_add_stores_correctly();
    test_data_frame_interval_series_multiple_items();
    test_data_frame_threshold_series_add_stores_correctly();
    test_data_frame_threshold_series_multiple_thresholds();
    test_data_frame_threshold_series_number_above();
    test_data_frame_combined_exact_and_threshold();
    test_data_frame_all_data_types();
    test_data_frame_lambda_exact_data_only();
    test_data_frame_calculate_lambda();
    test_data_frame_total_record_length_exact_only();
    test_data_frame_total_record_length_with_threshold();
    test_data_frame_zero_value_relative_frequency();

    // ProcessThresholdSeries derivations
    test_data_frame_process_threshold_no_overlap();
    test_data_frame_process_threshold_overlapping_points_reduce_number_below();
    test_data_frame_process_threshold_zeroes_number_above_when_fully_covered();
    test_data_frame_process_threshold_clamps_negative_number_below();
    test_data_frame_process_threshold_series_is_idempotent();

    // FullTimeSeries expansion (incl. ThresholdLikelihoodGuardTests DataFrame state)
    test_data_frame_full_time_series_splits_threshold_into_single_counts();
    test_data_frame_full_time_series_mixed_contents_and_ordering();
    test_data_frame_full_time_series_lazy_rebuild();

    // Low outliers (ExactDataHypothesisTests.cs)
    test_data_frame_set_low_outliers_from_mgbt();
    test_data_frame_set_low_outliers_from_threshold();
    test_data_frame_low_outlier_guards();
    test_data_frame_clear_low_outliers();

    // PlottingParameter / Validate / Clone
    test_data_frame_various_plotting_parameters();
    test_data_frame_set_plotting_parameter_rejects_out_of_range();
    test_data_frame_validate_aggregates_series_validations();
    test_data_frame_clone_is_deep_and_preserves_state();

    // Edge cases and scenarios
    test_data_frame_with_negative_values();
    test_data_frame_large_dataset();
    test_data_frame_special_double_values();
    test_data_frame_very_small_values();
    test_data_frame_very_large_values();
    test_data_frame_identical_values();
    test_data_frame_clear_and_rebuild();
    test_data_frame_scenario_systematic_record();
    test_data_frame_scenario_systematic_with_historical();
    test_data_frame_scenario_paleoflood();

    // P6: time-series import.
    p6_import::test_create_block_series();
    p6_import::test_create_peaks_over_threshold_series();

    return chtest::summary("data_frame");
}
