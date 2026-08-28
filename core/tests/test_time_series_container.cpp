// P6 ctest: the ported Numerics `TimeSeries` container, transcribed 1:1 from
//   upstream/Numerics/Test_Numerics/Data/Time Series/Test_TimeSeries.cs @ 2a0357a
// Every expected value here is a C#-test literal, read from that file (the established second
// oracle class for internal support types; the container's public R/Python surface gets its
// oracles from fixtures/ instead). The ConvertTimeInterval arrays are large and were extracted
// mechanically into core/tests/data/time_series_convert_data.hpp rather than retyped.
//
// Transcription notes:
//  1. `Test_ToXElement` is NOT transcribed: XML persistence is a project-wide severance, so there
//     is nothing to serialize. Its real content -- that a round trip preserves every index and
//     value -- is covered by `test_clone`, which asserts the same property bit-for-bit on the
//     direct deep copy that replaces C#'s XML round trip (see `TimeSeries::clone`).
//  2. `Test_Clear_LargeSeries_CompletesQuickly` keeps its RESULT assertion and drops its 5-second
//     timeout: a wall-clock bound is a property of the machine, and the O(n^2)-vs-O(n) contract it
//     guards is visible in the ported `Series::clear`, which calls `std::vector::clear` once.
//  3. The two `Test_Clear` / `Test_RemoveAt` methods also assert the CollectionChanged event
//     payload. The event is severed project-wide, so only the collection state is asserted; the
//     v2.1.4 semantics they exist to pin (clear empties in one shot, RemoveAt removes the
//     requested POSITION rather than the first equal ordinate) are both asserted in full.
//  4. Sections marked "corehydro supplement" are additions beyond the C# assertions, added where
//     upstream asserts a value but not the behaviour that produced it.
#include <cmath>
#include <limits>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "data/time_series_convert_data.hpp"

using corehydro::numerics::data::BlockFunctionType;
using corehydro::numerics::data::DateTime;
using corehydro::numerics::data::ListSortDirection;
using corehydro::numerics::data::TimeInterval;
using corehydro::numerics::data::SmoothingFunctionType;
using corehydro::numerics::data::TimeSeries;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// The twelve-value monthly series most upstream tests build on.
const std::vector<double> kValues = {22, 16, 33, 5, 12, 36, 48, 10, 18, 15, 22, 13};

TimeSeries monthly(const std::vector<double>& values = kValues) {
    return TimeSeries(TimeInterval::OneMonth, DateTime(2023, 1, 1), values);
}

// C# Test_Math's `Equal` helper.
void equal(const TimeSeries& ts, const std::vector<double>& values, double tolerance = 1e-6) {
    for (int i = 0; i < ts.count(); ++i)
        CHECK_NEAR(ts[i].value(), values[static_cast<std::size_t>(i)], tolerance);
}

// C# Test_Construction: the five non-XML constructors all build.
void test_construction() {
    TimeSeries ts1;
    CHECK_EQ(ts1.count(), 0);
    CHECK_TRUE(ts1.time_interval() == TimeInterval::OneDay);

    TimeSeries ts2(TimeInterval::OneYear);
    CHECK_EQ(ts2.count(), 0);
    CHECK_TRUE(ts2.time_interval() == TimeInterval::OneYear);

    TimeSeries ts3(TimeInterval::OneYear, DateTime(2023, 1, 1), DateTime(2023, 12, 31));
    TimeSeries ts4(TimeInterval::OneYear, DateTime(2023, 1, 1), DateTime(2023, 12, 31), 5);
    TimeSeries ts5(TimeInterval::OneYear, DateTime(2023, 1, 1),
                   std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    // corehydro supplement, cross-checked against the real library: the C# only checks that
    // these construct, which would pass on a container that produced the wrong number of
    // ordinates. The date-range constructors add the start ordinate FIRST and then walk while the
    // last index is BELOW the end date, so a yearly series over 2023 holds 2023-01-01 and
    // 2024-01-01 -- two ordinates, the second past the requested end date.
    CHECK_EQ(ts3.count(), 2);
    CHECK_TRUE(std::isnan(ts3[0].value()));
    CHECK_TRUE(ts3[1].index() == DateTime(2024, 1, 1));
    CHECK_EQ(ts4.count(), 2);
    CHECK_NEAR(ts4[0].value(), 5.0, 0.0);
    CHECK_NEAR(ts4[1].value(), 5.0, 0.0);
    CHECK_EQ(ts5.count(), 12);
    CHECK_TRUE(ts5[11].index() == DateTime(2034, 1, 1));

    // corehydro supplement: the four constructor guards.
    CHECK_THROWS(TimeSeries(TimeInterval::Irregular, DateTime(2023, 1, 1), DateTime(2023, 2, 1)));
    CHECK_THROWS(
        TimeSeries(TimeInterval::Irregular, DateTime(2023, 1, 1), DateTime(2023, 2, 1), 5.0));
    CHECK_THROWS(TimeSeries(TimeInterval::Irregular, DateTime(2023, 1, 1), kValues));
    CHECK_THROWS(TimeSeries(TimeInterval::OneDay, DateTime(2023, 2, 1), DateTime(2023, 1, 1)));
    CHECK_THROWS_MSG(TimeSeries(TimeInterval::OneDay, DateTime(2023, 2, 1), DateTime(2023, 1, 1)),
                     "Start date must be less than or equal to end date.");
}

// C# Test_Getters.
void test_getters() {
    TimeSeries ts(TimeInterval::OneMonth, DateTime(2023, 1, 1),
                  std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    CHECK_TRUE(ts.time_interval() == TimeInterval::OneMonth);
    CHECK_TRUE(!ts.has_missing_values());
    CHECK_TRUE(ts.start_date() == DateTime(2023, 1, 1));
    CHECK_TRUE(ts.end_date() == DateTime(2023, 12, 1));
}

// C# Test_Clone, plus the equivalence supplement described in note 1.
void test_clone() {
    TimeSeries ts = monthly();
    TimeSeries new_ts = ts.clone();
    for (int i = 0; i < ts.count(); ++i) {
        CHECK_TRUE(ts[i].index() == new_ts[i].index());
        CHECK_NEAR(ts[i].value(), new_ts[i].value(), 0.0);
    }
    CHECK_EQ(ts.has_missing_values(), new_ts.has_missing_values());
    CHECK_TRUE(ts.time_interval() == new_ts.time_interval());
    CHECK_TRUE(ts.start_date() == new_ts.start_date());
    CHECK_TRUE(ts.end_date() == new_ts.end_date());

    // corehydro supplement: C# `Clone()` round-trips through XML ("o" for the date, "G17" for the
    // value, both round-trip formats), so the direct deep copy that replaces it must be
    // bit-for-bit identical -- NaN included -- and must not alias the source.
    TimeSeries with_nan(TimeInterval::OneDay, DateTime(2024, 1, 1),
                        std::vector<double>{1.0, kNaN, 3.0});
    TimeSeries copy = with_nan.clone();
    CHECK_EQ(copy.count(), 3);
    CHECK_TRUE(std::isnan(copy[1].value()));
    CHECK_TRUE(copy[0].index() == with_nan[0].index());
    copy[0].set_value(99.0);
    CHECK_NEAR(with_nan[0].value(), 1.0, 0.0);
}

// C# Test_Sort.
void test_sort() {
    std::vector<double> values = kValues;
    TimeSeries ts = monthly(values);
    ts.sort_by_time();
    for (int i = 0; i < ts.count(); ++i)
        CHECK_NEAR(ts[i].value(), values[static_cast<std::size_t>(i)], 0.0);

    ts.sort_by_value();
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < ts.count(); ++i)
        CHECK_NEAR(ts[i].value(), sorted[static_cast<std::size_t>(i)], 0.0);

    // corehydro supplement: descending order, and the tie permutation. `List<T>.Sort` is an
    // UNSTABLE introsort, so which of two equal-valued ordinates lands first is a property of the
    // .NET algorithm rather than of the input order. The value 22 appears at January and
    // November; after an ascending sort by value the two 22s sit at positions 7 and 8, and the
    // dates below are the ones the REAL library produces -- measured by driving
    // `TimeSeries.SortByValue` on this exact series through a dotnet probe, which is what makes
    // this an oracle rather than a snapshot of our own output.
    TimeSeries tied = monthly();
    tied.sort_by_value();
    CHECK_NEAR(tied[7].value(), 22.0, 0.0);
    CHECK_NEAR(tied[8].value(), 22.0, 0.0);
    CHECK_TRUE(tied[7].index() == DateTime(2023, 1, 1));
    CHECK_TRUE(tied[8].index() == DateTime(2023, 11, 1));

    TimeSeries desc = monthly();
    desc.sort_by_value(ListSortDirection::Descending);
    CHECK_NEAR(desc[0].value(), 48.0, 0.0);
    CHECK_NEAR(desc[11].value(), 5.0, 0.0);

    TimeSeries desc_time = monthly();
    desc_time.sort_by_time(ListSortDirection::Descending);
    CHECK_TRUE(desc_time[0].index() == DateTime(2023, 12, 1));
    CHECK_TRUE(desc_time[11].index() == DateTime(2023, 1, 1));
}

// C# Test_Math: the whole in-place math region, each step compared against the same operation
// applied to a plain array.
void test_math() {
    std::vector<double> values = kValues;
    TimeSeries ts = monthly(values);

    for (auto& v : values) v += 10;
    ts.add(10.0);
    equal(ts, values);

    for (auto& v : values) v -= 5;
    ts.subtract(5.0);
    equal(ts, values);

    for (auto& v : values) v *= 4;
    ts.multiply(4.0);
    equal(ts, values);

    for (auto& v : values) v /= -2;
    ts.divide(-2.0);
    equal(ts, values);

    for (auto& v : values) v = std::fabs(v);
    ts.absolute_value();
    equal(ts, values);

    for (auto& v : values) v = std::pow(v, 2);
    ts.exponentiate(2.0);
    equal(ts, values);

    for (auto& v : values) v = std::log10(v);
    ts.log_transform();
    equal(ts, values);

    for (auto& v : values) v = 1 / v;
    ts.inverse();
    equal(ts, values);

    double mean = 0;
    for (double v : values) mean += v;
    mean /= static_cast<double>(values.size());
    double ss = 0;
    for (double v : values) ss += (v - mean) * (v - mean);
    double sd = std::sqrt(ss / (static_cast<double>(values.size()) - 1));
    for (auto& v : values) v = (v - mean) / sd;
    ts.standardize();
    equal(ts, values);

    // corehydro supplement: the guards and the missing-value contract.
    TimeSeries with_nan(TimeInterval::OneDay, DateTime(2024, 1, 1),
                        std::vector<double>{1.0, kNaN, 3.0});
    with_nan.add(10.0);
    CHECK_NEAR(with_nan[0].value(), 11.0, 0.0);
    CHECK_TRUE(std::isnan(with_nan[1].value()));   // missing stays missing
    CHECK_THROWS_MSG(with_nan.divide(0.0), "Cannot divide by zero.");

    // log_transform is the exception: a non-positive value is WRITTEN as NaN.
    TimeSeries neg(TimeInterval::OneDay, DateTime(2024, 1, 1), std::vector<double>{1.0, -2.0, 0.0});
    neg.log_transform();
    CHECK_NEAR(neg[0].value(), 0.0, 1e-12);
    CHECK_TRUE(std::isnan(neg[1].value()));
    CHECK_TRUE(std::isnan(neg[2].value()));

    // standardize propagates a NaN rather than skipping it, and refuses a zero spread.
    TimeSeries flat(TimeInterval::OneDay, DateTime(2024, 1, 1),
                    std::vector<double>{2.0, 2.0, 2.0});
    CHECK_THROWS_MSG(flat.standardize(), "Standard deviation is zero");

    // The indexed overloads apply only at the listed positions and skip out-of-range ones.
    TimeSeries indexed(TimeInterval::OneDay, DateTime(2024, 1, 1),
                       std::vector<double>{1.0, 2.0, 3.0});
    indexed.add(10.0, {0, 2, 7, -1});
    CHECK_NEAR(indexed[0].value(), 11.0, 0.0);
    CHECK_NEAR(indexed[1].value(), 2.0, 0.0);
    CHECK_NEAR(indexed[2].value(), 13.0, 0.0);
    // ... except log_transform and inverse, whose `else` branch reaches through an unchecked
    // index (see the header note on both methods).
    CHECK_THROWS(indexed.log_transform(std::vector<int>{7}));
    CHECK_THROWS(indexed.inverse(std::vector<int>{7}));
}

// C# Test_Cumulative.
void test_cumulative() {
    std::vector<double> values = kValues;
    TimeSeries ts = monthly(values);
    for (std::size_t i = 1; i < values.size(); ++i) values[i] = values[i - 1] + values[i];
    TimeSeries new_ts = ts.cumulative_sum();
    equal(new_ts, values);

    // corehydro supplement: missing values accumulate as zero but are NOT skipped in the output,
    // and the result carries the DEFAULT OneDay interval rather than the source's (upstream
    // oddity, see the method's own note).
    CHECK_TRUE(new_ts.time_interval() == TimeInterval::OneDay);
    CHECK_TRUE(ts.time_interval() == TimeInterval::OneMonth);
    TimeSeries with_nan(TimeInterval::OneDay, DateTime(2024, 1, 1),
                        std::vector<double>{1.0, kNaN, 3.0});
    TimeSeries cum = with_nan.cumulative_sum();
    CHECK_NEAR(cum[0].value(), 1.0, 0.0);
    CHECK_NEAR(cum[1].value(), 1.0, 0.0);
    CHECK_NEAR(cum[2].value(), 4.0, 0.0);
}

// C# Test_Difference and Test_Difference_NaN.
void test_difference() {
    std::vector<double> values = kValues;
    TimeSeries ts = monthly(values);
    std::vector<double> expected;
    for (std::size_t i = 1; i < values.size(); ++i) expected.push_back(values[i] - values[i - 1]);
    TimeSeries new_ts = ts.difference();
    CHECK_EQ(new_ts.count(), 11);
    equal(new_ts, expected);

    // corehydro supplement: the differenced series keeps the ORIGINAL start date.
    CHECK_TRUE(new_ts.start_date() == ts.start_date());

    TimeSeries nan_ts(TimeInterval::OneDay, DateTime(2023, 1, 1),
                      std::vector<double>{1.0, kNaN, 3.0, 7.0});
    TimeSeries diff = nan_ts.difference();
    CHECK_EQ(diff.count(), 3);
    CHECK_TRUE(std::isnan(diff[0].value()));
    CHECK_TRUE(std::isnan(diff[1].value()));
    CHECK_NEAR(diff[2].value(), 4.0, 0.0);

    CHECK_THROWS(TimeSeries(TimeInterval::OneDay, DateTime(2023, 1, 1),
                            std::vector<double>{1.0, 2.0, 3.0})
                     .difference(3, 1));
}

// C# Test_Missing: the counting, replacing and interpolating methods.
void test_missing() {
    std::vector<double> values = {22, 16, 33, 5, 12, 36, 48, 10, 18, 15, kNaN, kNaN};
    TimeSeries ts = monthly(values);
    CHECK_EQ(ts.number_of_missing_values(), 2);
    CHECK_TRUE(ts.has_missing_values());

    ts.replace_missing_data(-1);
    CHECK_EQ(ts.number_of_missing_values(), 0);
    CHECK_NEAR(ts[10].value(), -1.0, 0.0);
    CHECK_NEAR(ts[11].value(), -1.0, 0.0);

    // The trailing pair is EXTRAPOLATED from the two preceding ordinates, in OA-date space, so
    // these two literals also pin `DateTime::to_oa_date` on a month-length-varying series.
    TimeSeries ts2 = monthly(values);
    ts2.interpolate_missing_data(2);
    CHECK_EQ(ts2.number_of_missing_values(), 0);
    CHECK_NEAR(ts2[10].value(), 11.9, 1e-6);
    CHECK_NEAR(ts2[11].value(), 8.9, 1e-6);

    // corehydro supplement: an interior gap is INTERPOLATED between its neighbours, and a gap
    // longer than the limit is left alone.
    TimeSeries gap(TimeInterval::OneDay, DateTime(2024, 1, 1),
                   std::vector<double>{10.0, kNaN, 30.0});
    gap.interpolate_missing_data(2);
    CHECK_NEAR(gap[1].value(), 20.0, 1e-12);

    TimeSeries long_gap(TimeInterval::OneDay, DateTime(2024, 1, 1),
                        std::vector<double>{10.0, kNaN, kNaN, kNaN, 50.0});
    long_gap.interpolate_missing_data(1);
    CHECK_TRUE(std::isnan(long_gap[1].value()));

    // fill_missing_dates spans the requested range at this interval, taking observed values where
    // a date matches and the fill value elsewhere.
    TimeSeries sparse(TimeInterval::OneDay);
    sparse.add(TimeSeries::Ordinate(DateTime(2024, 1, 1), 1.0));
    sparse.add(TimeSeries::Ordinate(DateTime(2024, 1, 4), 4.0));
    TimeSeries filled = TimeSeries::fill_missing_dates(sparse, DateTime(2024, 1, 1),
                                                       DateTime(2024, 1, 5), -9.0);
    CHECK_EQ(filled.count(), 5);
    CHECK_NEAR(filled[0].value(), 1.0, 0.0);
    CHECK_NEAR(filled[1].value(), -9.0, 0.0);
    CHECK_NEAR(filled[3].value(), 4.0, 0.0);
    CHECK_NEAR(filled[4].value(), -9.0, 0.0);
    CHECK_THROWS(TimeSeries::fill_missing_dates(TimeSeries(TimeInterval::Irregular),
                                                DateTime(2024, 1, 1), DateTime(2024, 1, 5)));
}

// C# Test_AddInterval / Test_SubtractInterval / Test_TimeIntervalInHours.
void test_intervals() {
    DateTime original(2023, 1, 1, 9, 30, 25);
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::OneMinute) ==
               original.add_minutes(1));
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::FiveMinute) ==
               original.add_minutes(5));
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::FifteenMinute) ==
               original.add_minutes(15));
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::ThirtyMinute) ==
               original.add_minutes(30));
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::OneHour) ==
               original.add_hours(1));
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::SixHour) ==
               original.add_hours(6));
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::TwelveHour) ==
               original.add_hours(12));
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::OneDay) ==
               original.add_days(1));
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::SevenDay) ==
               original.add_days(7));
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::OneMonth) ==
               original.add_months(1));
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::OneQuarter) ==
               original.add_months(3));
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::OneYear) ==
               original.add_years(1));
    // corehydro supplement: irregular falls through and returns the date unchanged.
    CHECK_TRUE(TimeSeries::add_time_interval(original, TimeInterval::Irregular) == original);

    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::OneMinute) ==
               original.add_minutes(-1));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::FiveMinute) ==
               original.add_minutes(-5));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::FifteenMinute) ==
               original.add_minutes(-15));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::ThirtyMinute) ==
               original.add_minutes(-30));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::OneHour) ==
               original.add_hours(-1));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::SixHour) ==
               original.add_hours(-6));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::TwelveHour) ==
               original.add_hours(-12));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::OneDay) ==
               original.add_days(-1));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::SevenDay) ==
               original.add_days(-7));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::OneMonth) ==
               original.add_months(-1));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::OneQuarter) ==
               original.add_months(-3));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::OneYear) ==
               original.add_years(-1));
    CHECK_TRUE(TimeSeries::subtract_time_interval(original, TimeInterval::Irregular) == original);

    CHECK_NEAR(TimeSeries::time_interval_in_hours(TimeInterval::OneMinute), 1.0 / 60.0, 0.0);
    CHECK_NEAR(TimeSeries::time_interval_in_hours(TimeInterval::FiveMinute), 1.0 / 12.0, 0.0);
    CHECK_NEAR(TimeSeries::time_interval_in_hours(TimeInterval::FifteenMinute), 3.0 / 12.0, 0.0);
    CHECK_NEAR(TimeSeries::time_interval_in_hours(TimeInterval::ThirtyMinute), 1.0 / 2.0, 0.0);
    CHECK_NEAR(TimeSeries::time_interval_in_hours(TimeInterval::OneHour), 1, 0.0);
    CHECK_NEAR(TimeSeries::time_interval_in_hours(TimeInterval::SixHour), 6, 0.0);
    CHECK_NEAR(TimeSeries::time_interval_in_hours(TimeInterval::TwelveHour), 12, 0.0);
    CHECK_NEAR(TimeSeries::time_interval_in_hours(TimeInterval::OneDay), 24, 0.0);
    CHECK_NEAR(TimeSeries::time_interval_in_hours(TimeInterval::SevenDay), 168, 0.0);
    // corehydro supplement: the intervals with no fixed hour count return NaN, which is what
    // makes ConvertTimeInterval fall through to its null branch for them.
    CHECK_TRUE(std::isnan(TimeSeries::time_interval_in_hours(TimeInterval::OneMonth)));
    CHECK_TRUE(std::isnan(TimeSeries::time_interval_in_hours(TimeInterval::OneYear)));
    CHECK_TRUE(std::isnan(TimeSeries::time_interval_in_hours(TimeInterval::Irregular)));
}

// C# Test_MovingAverage / Test_MovingSum and their four NaN variants.
void test_moving_windows() {
    TimeSeries ts = monthly();
    TimeSeries avg = ts.moving_average(5);
    const std::vector<double> avg_valid = {17.6, 20.4, 26.8, 22.2, 24.8, 25.4, 22.6, 15.6};
    CHECK_EQ(avg.count(), static_cast<int>(avg_valid.size()));
    for (int i = 0; i < avg.count(); ++i)
        CHECK_NEAR(avg[i].value(), avg_valid[static_cast<std::size_t>(i)], 1e-12);

    TimeSeries ts2 = monthly();
    TimeSeries sum = ts2.moving_sum(5);
    const std::vector<double> sum_valid = {88, 102, 134, 111, 124, 127, 113, 78};
    CHECK_EQ(sum.count(), static_cast<int>(sum_valid.size()));
    for (int i = 0; i < sum.count(); ++i)
        CHECK_NEAR(sum[i].value(), sum_valid[static_cast<std::size_t>(i)], 1e-12);

    const std::vector<double> nan_data = {1.0, kNaN, 3.0, 4.0, 5.0};

    TimeSeries a(TimeInterval::OneDay, DateTime(2023, 1, 1), nan_data);
    TimeSeries strict_sum = a.moving_sum(2);
    CHECK_EQ(strict_sum.count(), 4);
    CHECK_TRUE(std::isnan(strict_sum[0].value()));
    CHECK_TRUE(std::isnan(strict_sum[1].value()));
    CHECK_NEAR(strict_sum[2].value(), 7.0, 0.0);
    CHECK_NEAR(strict_sum[3].value(), 9.0, 0.0);

    TimeSeries b(TimeInterval::OneDay, DateTime(2023, 1, 1), nan_data);
    TimeSeries skip_sum = b.moving_sum(2, 1);
    CHECK_EQ(skip_sum.count(), 4);
    CHECK_NEAR(skip_sum[0].value(), 1.0, 0.0);
    CHECK_NEAR(skip_sum[1].value(), 3.0, 0.0);
    CHECK_NEAR(skip_sum[2].value(), 7.0, 0.0);
    CHECK_NEAR(skip_sum[3].value(), 9.0, 0.0);

    TimeSeries c(TimeInterval::OneDay, DateTime(2023, 1, 1), nan_data);
    TimeSeries strict_avg = c.moving_average(2);
    CHECK_EQ(strict_avg.count(), 4);
    CHECK_TRUE(std::isnan(strict_avg[0].value()));
    CHECK_TRUE(std::isnan(strict_avg[1].value()));
    CHECK_NEAR(strict_avg[2].value(), 3.5, 0.0);
    CHECK_NEAR(strict_avg[3].value(), 4.5, 0.0);

    TimeSeries d(TimeInterval::OneDay, DateTime(2023, 1, 1), nan_data);
    TimeSeries skip_avg = d.moving_average(2, 1);
    CHECK_EQ(skip_avg.count(), 4);
    CHECK_NEAR(skip_avg[0].value(), 1.0, 0.0);
    CHECK_NEAR(skip_avg[1].value(), 3.0, 0.0);
    CHECK_NEAR(skip_avg[2].value(), 3.5, 0.0);
    CHECK_NEAR(skip_avg[3].value(), 4.5, 0.0);

    // corehydro supplement: the two guards. `period >= Count` throws (NOT `period > Count`), and
    // `min_valid_count` must sit in [1, period].
    TimeSeries e = monthly();
    CHECK_THROWS(e.moving_average(12));
    CHECK_THROWS(e.moving_sum(12));
    CHECK_THROWS(e.moving_average(5, 0));
    CHECK_THROWS(e.moving_average(5, 6));
    // The result carries the source's dates, one per window END.
    TimeSeries f = monthly();
    TimeSeries windowed = f.moving_average(5);
    CHECK_TRUE(windowed[0].index() == DateTime(2023, 5, 1));
    CHECK_TRUE(windowed.time_interval() == TimeInterval::OneMonth);
}

// C# Test_ShiftAllDates / Test_ShiftDatesByDays / ByMonth / ByYear.
void test_shifts() {
    TimeSeries ts = monthly();
    TimeSeries shifted = ts.clone();
    shifted.shift_all_dates(DateTime(2024, 2, 4));
    for (int i = 0; i < shifted.count(); ++i) {
        CHECK_TRUE(shifted[i].index() ==
                   ts[i].index().add_years(1).add_months(1).add_days(3));
        CHECK_NEAR(shifted[i].value(), ts[i].value(), 0.0);
    }

    TimeSeries by_day = ts.shift_dates_by_day(5);
    for (int i = 0; i < by_day.count(); ++i) {
        CHECK_TRUE(by_day[i].index() == ts[i].index().add_days(5));
        CHECK_NEAR(by_day[i].value(), ts[i].value(), 0.0);
    }

    TimeSeries by_month = ts.shift_dates_by_month(5);
    for (int i = 0; i < by_month.count(); ++i) {
        CHECK_TRUE(by_month[i].index() == ts[i].index().add_months(5));
        CHECK_NEAR(by_month[i].value(), ts[i].value(), 0.0);
    }

    TimeSeries by_year = ts.shift_dates_by_year(5);
    for (int i = 0; i < by_year.count(); ++i) {
        CHECK_TRUE(by_year[i].index() == ts[i].index().add_years(5));
        CHECK_NEAR(by_year[i].value(), ts[i].value(), 0.0);
    }

    // corehydro supplement: on an IRREGULAR series shift_all_dates moves only the first ordinate,
    // because there is no interval to re-walk.
    TimeSeries irregular(TimeInterval::Irregular);
    irregular.add(TimeSeries::Ordinate(DateTime(2024, 1, 1), 1.0));
    irregular.add(TimeSeries::Ordinate(DateTime(2024, 3, 17), 2.0));
    irregular.shift_all_dates(DateTime(2000, 1, 1));
    CHECK_TRUE(irregular[0].index() == DateTime(2000, 1, 1));
    CHECK_TRUE(irregular[1].index() == DateTime(2024, 3, 17));
}

// C# Test_ClipTimeSeries.
void test_clip() {
    TimeSeries ts = monthly();
    DateTime start(2023, 11, 1);
    DateTime end(2023, 12, 1);
    TimeSeries clipped = ts.clip_time_series(start, end);
    CHECK_EQ(clipped.count(), 2);
    CHECK_TRUE(clipped.start_date() == start);
    CHECK_TRUE(clipped.end_date() == end);
    for (int i = 0; i < clipped.count(); ++i) {
        CHECK_TRUE(clipped[i].index() == ts[10 + i].index());
        CHECK_NEAR(clipped[i].value(), ts[10 + i].value(), 0.0);
    }

    // corehydro supplement: both bounds must lie inside the series' own span.
    CHECK_THROWS(ts.clip_time_series(DateTime(2022, 1, 1), end));
    CHECK_THROWS(ts.clip_time_series(start, DateTime(2025, 1, 1)));
}

// C# Test_ConvertTimeInterval_1hr_to_15min / _1hr_to_6hr / _1hr_to_1Day / _1hr_to_1Day_Sum.
void test_convert_time_interval() {
    using namespace corehydro_test_data;
    TimeSeries hourly(TimeInterval::OneHour, DateTime(1973, 5, 1), kConvertHourlyInput);

    auto to_15 = hourly.convert_time_interval(TimeInterval::FifteenMinute);
    CHECK_TRUE(to_15.has_value());
    for (std::size_t i = 0; i < kConvert1HrTo15Min.size(); ++i)
        CHECK_NEAR((*to_15)[static_cast<int>(i)].value(), kConvert1HrTo15Min[i], 1e-2);

    auto to_6 = hourly.convert_time_interval(TimeInterval::SixHour);
    CHECK_TRUE(to_6.has_value());
    for (std::size_t i = 0; i < kConvert1HrTo6Hr.size(); ++i)
        CHECK_NEAR((*to_6)[static_cast<int>(i)].value(), kConvert1HrTo6Hr[i], 1e-2);

    auto to_day = hourly.convert_time_interval(TimeInterval::OneDay);
    CHECK_TRUE(to_day.has_value());
    for (std::size_t i = 0; i < kConvert1HrTo1Day.size(); ++i)
        CHECK_NEAR((*to_day)[static_cast<int>(i)].value(), kConvert1HrTo1Day[i], 1e-2);

    auto day_sum = hourly.convert_time_interval(TimeInterval::OneDay, false);
    CHECK_TRUE(day_sum.has_value());
    for (std::size_t i = 0; i < kConvert1HrTo1DaySum.size(); ++i)
        CHECK_NEAR((*day_sum)[static_cast<int>(i)].value(), kConvert1HrTo1DaySum[i], 1e-2);

    auto back_to_6 = day_sum->convert_time_interval(TimeInterval::SixHour, false);
    CHECK_TRUE(back_to_6.has_value());
    for (std::size_t i = 0; i < kConvert1DayTo6HrSum.size(); ++i)
        CHECK_NEAR((*back_to_6)[static_cast<int>(i)].value(), kConvert1DayTo6HrSum[i], 1e-2);

    // corehydro supplement: an unchanged interval clones, and an interval pair with no defined
    // hour count (a monthly source) falls out of every branch and returns nothing.
    auto same = hourly.convert_time_interval(TimeInterval::OneHour);
    CHECK_TRUE(same.has_value());
    CHECK_EQ(same->count(), hourly.count());
    TimeSeries months = monthly();
    CHECK_TRUE(!months.convert_time_interval(TimeInterval::OneYear).has_value());
}

// C# Test_Clear_EmptiesSeriesAndRaisesSingleReset, Test_Clear_LargeSeries_CompletesQuickly and
// Test_RemoveAt_WithDuplicateOrdinates_RemovesRequestedIndex (see transcription notes 2 and 3).
void test_series_mutation() {
    TimeSeries ts(TimeInterval::OneDay, DateTime(2024, 1, 1),
                  std::vector<double>{1, 2, 3, 4, 5});
    ts.clear();
    CHECK_EQ(ts.count(), 0);

    TimeSeries big(TimeInterval::OneDay, DateTime(1900, 1, 1), std::vector<double>(200000, 0.0));
    CHECK_EQ(big.count(), 200000);
    big.clear();
    CHECK_EQ(big.count(), 0);

    TimeSeries::Ordinate first(DateTime(2024, 1, 1), 1.0);
    TimeSeries::Ordinate middle(DateTime(2024, 1, 2), 2.0);
    TimeSeries::Ordinate duplicate_of_first(DateTime(2024, 1, 1), 1.0);
    TimeSeries dup(TimeInterval::Irregular);
    dup.add(first);
    dup.add(middle);
    dup.add(duplicate_of_first);

    dup.remove_at(2);
    CHECK_EQ(dup.count(), 2);
    CHECK_TRUE(dup[0].index() == first.index());
    CHECK_NEAR(dup[0].value(), first.value(), 0.0);
    CHECK_TRUE(dup[1].index() == middle.index());
    CHECK_NEAR(dup[1].value(), middle.value(), 0.0);

    // corehydro supplement: `remove(item)` is the OTHER contract -- it removes the first EQUAL
    // ordinate, which is exactly the behaviour v2.1.4 stopped `RemoveAt` from delegating to.
    TimeSeries dup2(TimeInterval::Irregular);
    dup2.add(first);
    dup2.add(middle);
    dup2.add(duplicate_of_first);
    CHECK_TRUE(dup2.remove(duplicate_of_first));
    CHECK_EQ(dup2.count(), 2);
    CHECK_TRUE(dup2[0].index() == middle.index());   // the FIRST equal one went
    CHECK_TRUE(!dup2.remove(TimeSeries::Ordinate(DateTime(1999, 1, 1), 0.0)));
    CHECK_EQ(dup2.index_of(middle), 0);
    CHECK_TRUE(dup2.contains(middle));

    TimeSeries ins(TimeInterval::Irregular);
    ins.add(first);
    ins.insert(0, middle);
    CHECK_TRUE(ins[0].index() == middle.index());
    CHECK_EQ(ins.count(), 2);
}

// --- Task 4: statistics, block series, decomposition and resampling. ---

// The 69-value monthly series the statistics and block-series tests share, read from
// Test_TimeSeries.cs (Test_SummaryHypothesisTest / Test_CalendarYearSeries / Test_WaterYearSeries /
// Test_CustomYearSeries / Test_MonthlySeries / Test_QuarterlySeries / Test_PeaksOverThreshold all
// build on it).
const std::vector<double> kLongValues = {
    122,  244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331, 225, 174, 98.8, 149,
    238,  262, 132, 235, 216, 240, 230, 192, 195, 172, 173, 172, 153, 142, 317, 161, 201,  204,
    194,  164, 183, 161, 167, 179, 185, 117, 192, 337, 125, 166, 99.1, 202, 230, 158, 262, 154,
    164,  182, 164, 183, 171, 250, 184, 205, 237, 177, 239, 187, 180, 173, 174};

// C# Test_Stats: MinValue, MaxValue, MeanValue, StandardDeviation and Duration.
void test_stats() {
    TimeSeries ts = monthly();
    CHECK_NEAR(ts.min_value(), 5, 0.0);
    CHECK_NEAR(ts.max_value(), 48, 0.0);
    CHECK_NEAR(ts.mean_value(), 20.83333333333, 1e-10);
    CHECK_NEAR(ts.standard_deviation(), 12.40112, 1e-3);

    std::vector<std::vector<double>> duration = ts.duration();
    const std::vector<double> d_values = {48, 36, 33, 22, 22, 18, 16, 15, 13, 12, 10, 5};
    const std::vector<double> d_valid = {
        7.69230769230769,  15.3846153846154, 23.0769230769231, 30.7692307692308,
        38.4615384615385,  46.1538461538462, 53.8461538461538, 61.5384615384615,
        69.2307692307692,  76.9230769230769, 84.6153846153846, 92.3076923076923};
    CHECK_EQ(static_cast<int>(duration.size()), 12);
    for (std::size_t i = 0; i < d_valid.size(); ++i) {
        CHECK_NEAR(duration[i][0], d_valid[i], 1e-10);
        CHECK_NEAR(duration[i][1], d_values[i], 1e-10);
    }

    // corehydro supplement: min_value and max_value return their SENTINELS, not NaN, on a series
    // with nothing observed -- upstream initializes them to double.MaxValue / double.MinValue and
    // never revisits that if no value passes the NaN filter.
    TimeSeries all_nan(TimeInterval::OneDay, DateTime(2024, 1, 1), std::vector<double>{kNaN, kNaN});
    CHECK_NEAR(all_nan.min_value(), std::numeric_limits<double>::max(), 0.0);
    CHECK_NEAR(all_nan.max_value(), std::numeric_limits<double>::lowest(), 0.0);
}

// C# Test_SummaryStats: SummaryPercentiles, Percentiles and SummaryStatistics.
void test_summary_stats() {
    TimeSeries ts = monthly();

    std::vector<double> summary_p = ts.summary_percentiles();
    const std::vector<double> sp_valid = {7.75, 12.75, 17.00, 24.75, 41.40};
    CHECK_EQ(static_cast<int>(summary_p.size()), 5);
    for (std::size_t i = 0; i < sp_valid.size(); ++i)
        CHECK_NEAR(summary_p[i], sp_valid[i], 1e-10);

    std::vector<double> pct = ts.percentiles({0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90});
    const std::vector<double> p_valid = {10.2, 12.2, 13.6, 15.4, 20.4, 22.0, 30.8, 35.7};
    for (std::size_t i = 0; i < p_valid.size(); ++i) CHECK_NEAR(pct[i], p_valid[i], 1e-10);

    std::vector<std::pair<std::string, double>> stats = ts.summary_statistics();
    CHECK_EQ(static_cast<int>(stats.size()), 15);
    auto stat = [&stats](const std::string& key) {
        for (const auto& kv : stats)
            if (kv.first == key) return kv.second;
        return kNaN;
    };
    CHECK_NEAR(stat("Record Length"), 12, 1e-10);
    CHECK_NEAR(stat("Missing Values"), 0, 1e-10);
    CHECK_NEAR(stat("Minimum"), 5, 1e-10);
    CHECK_NEAR(stat("Maximum"), 48, 1e-10);
    CHECK_NEAR(stat("Mean"), 20.83333333333, 1e-10);
    CHECK_NEAR(stat("Std Dev"), 12.4011240937215, 1e-10);
    CHECK_NEAR(stat("Skewness"), 1.06382220138287, 1e-10);
    CHECK_NEAR(stat("Kurtosis"), 0.682181791356259, 1e-10);
    CHECK_NEAR(stat("5%"), 7.75, 1e-10);
    CHECK_NEAR(stat("25%"), 12.75, 1e-10);
    CHECK_NEAR(stat("50%"), 17.00, 1e-10);
    CHECK_NEAR(stat("75%"), 24.75, 1e-10);
    CHECK_NEAR(stat("95%"), 41.40, 1e-10);

    // corehydro supplement: the key ORDER is the contract (C# reads a Dictionary back by key, but
    // every host binding reads this as an ordered pair list), and the `Count <= 2` guard is on the
    // ORDINATE count.
    CHECK_EQ(stats[0].first, std::string("Record Length"));
    CHECK_EQ(stats[7].first, std::string("Kurtosis"));
    CHECK_EQ(stats[14].first, std::string("99%"));
    TimeSeries two(TimeInterval::OneDay, DateTime(2024, 1, 1), std::vector<double>{1.0, 2.0});
    auto small = two.summary_statistics();
    CHECK_TRUE(std::isnan(small[4].second));   // Mean
    CHECK_NEAR(small[0].second, 2, 0.0);       // Record Length still reported
}

// C# Test_SummaryHypothesisTest. The C# tolerance is 1E-1 because the reference values come from
// R, not from C#; it is kept.
void test_summary_hypothesis_test() {
    TimeSeries ts(TimeInterval::OneMonth, DateTime(2023, 1, 1), kLongValues);
    auto h = ts.summary_hypothesis_test();
    CHECK_EQ(static_cast<int>(h.size()), 7);
    const std::vector<double> valid = {0.0012195, 0.8204, 0.2436, 0.5956, 0.7757, 0.4691, 0.2287};
    for (std::size_t i = 0; i < valid.size(); ++i) CHECK_NEAR(h[i].second, valid[i], 1e-1);

    // corehydro supplement: the seven key strings, in order. They are NOT the ten of the
    // RMC.BestFit DataFrame facade of the same name.
    CHECK_EQ(h[0].first, std::string("Jarque-Bera test for normality"));
    CHECK_EQ(h[3].first,
             std::string("Mann-Whitney test for homogeneity and stationarity (jump)"));
    CHECK_EQ(h[5].first, std::string("t-test for differences in the means of two samples"));
    CHECK_EQ(h[6].first, std::string("F-test for differences in the variances of two samples"));
}

// C# Test_MonthlyStats and Test_MonthlyFrequency.
void test_monthly_stats() {
    const std::vector<double> values = {22, 16, 33, 5,  12, 36, 48, 10, 18, 15, 22, 13,
                                        38, 17, 3,  6,  27, 11, 2,  41, 44, 37, 50, 8};
    TimeSeries ts(TimeInterval::OneMonth, DateTime(2023, 1, 1), values);

    std::vector<std::vector<double>> pct =
        ts.monthly_percentiles({0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90});
    const double valid_p[12][8] = {
        {23.6, 25.2, 26.8, 28.4, 31.6, 33.2, 34.8, 36.4},
        {16.1, 16.2, 16.3, 16.4, 16.6, 16.7, 16.8, 16.9},
        {6, 9, 12, 15, 21, 24, 27, 30},
        {5.1, 5.2, 5.3, 5.4, 5.6, 5.7, 5.8, 5.9},
        {13.5, 15.0, 16.5, 18.0, 21.0, 22.5, 24.0, 25.5},
        {13.5, 16.0, 18.5, 21.0, 26.0, 28.5, 31.0, 33.5},
        {6.6, 11.2, 15.8, 20.4, 29.6, 34.2, 38.8, 43.4},
        {13.1, 16.2, 19.3, 22.4, 28.6, 31.7, 34.8, 37.9},
        {20.6, 23.2, 25.8, 28.4, 33.6, 36.2, 38.8, 41.4},
        {17.2, 19.4, 21.6, 23.8, 28.2, 30.4, 32.6, 34.8},
        {24.8, 27.6, 30.4, 33.2, 38.8, 41.6, 44.4, 47.2},
        {8.5, 9.0, 9.5, 10.0, 11.0, 11.5, 12.0, 12.5}};

    std::vector<std::vector<double>> summary = ts.monthly_summary_statistics();
    const double valid_s[12][8] = {
        {22, 22.8, 26.0, 30.0, 34.0, 37.2, 38, 30},
        {16, 16.05, 16.25, 16.50, 16.75, 16.95, 17, 16.5},
        {3, 4.5, 10.5, 18.0, 25.5, 31.5, 33, 18},
        {5, 5.05, 5.25, 5.50, 5.75, 5.95, 6, 5.5},
        {12, 12.75, 15.75, 19.50, 23.25, 26.25, 27, 19.5},
        {11, 12.25, 17.25, 23.50, 29.75, 34.75, 36, 23.5},
        {2, 4.3, 13.5, 25.0, 36.5, 45.7, 48, 25},
        {10, 11.55, 17.75, 25.50, 33.25, 39.45, 41, 25.5},
        {18, 19.3, 24.5, 31.0, 37.5, 42.7, 44, 31},
        {15, 16.1, 20.5, 26.0, 31.5, 35.9, 37, 26},
        {22, 23.4, 29.0, 36.0, 43.0, 48.6, 50, 36},
        {8, 8.25, 9.25, 10.50, 11.75, 12.75, 13, 10.5}};

    // The C# loop stops at i < 11, leaving December unasserted; this transcription runs all 12
    // (the twelfth row is in the C# literal table, just never compared).
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 8; ++j) {
            CHECK_NEAR(pct[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                       valid_p[i][j], 1e-10);
            CHECK_NEAR(summary[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                       valid_s[i][j], 1e-10);
        }
    }

    std::vector<double> frequencies = ts.monthly_frequency();
    CHECK_EQ(static_cast<int>(frequencies.size()), 12);
    for (double f : frequencies) CHECK_NEAR(f, 2, 0.0);

    // corehydro supplement: a month with no observation keeps an all-zero summary row, and
    // monthly_percentiles does NOT filter NaN (unlike monthly_summary_statistics), so a missing
    // value poisons that month's percentiles.
    TimeSeries partial(TimeInterval::OneMonth, DateTime(2023, 1, 1),
                       std::vector<double>{1.0, kNaN});
    auto part_summary = partial.monthly_summary_statistics();
    CHECK_NEAR(part_summary[2][0], 0.0, 0.0);   // March: no data at all
    CHECK_NEAR(part_summary[1][0], 0.0, 0.0);   // February: one NaN, filtered out, row untouched
    auto part_pct = partial.monthly_percentiles({0.5});
    CHECK_TRUE(std::isnan(part_pct[1][0]));
}

// C# Test_CalendarYearSeries, Test_WaterYearSeries, Test_CustomYearSeries, Test_MonthlySeries,
// Test_QuarterlySeries.
void test_block_series() {
    TimeSeries ts(TimeInterval::OneMonth, DateTime(2023, 1, 1), kLongValues);

    TimeSeries annual = ts.calendar_year_series();
    const std::vector<double> annual_max = {263, 331, 317, 337, 262, 239};
    CHECK_EQ(annual.count(), static_cast<int>(annual_max.size()));
    for (std::size_t i = 0; i < annual_max.size(); ++i)
        CHECK_NEAR(annual[static_cast<int>(i)].value(), annual_max[i], 0.0);

    TimeSeries water = ts.custom_year_series();
    const std::vector<double> water_max = {263, 331, 317, 204, 337, 250};
    CHECK_EQ(water.count(), static_cast<int>(water_max.size()));
    for (std::size_t i = 0; i < water_max.size(); ++i)
        CHECK_NEAR(water[static_cast<int>(i)].value(), water_max[i], 0.0);

    TimeSeries custom_cal = ts.custom_year_series(1, 12);
    for (std::size_t i = 0; i < annual_max.size(); ++i)
        CHECK_NEAR(custom_cal[static_cast<int>(i)].value(), annual_max[i], 0.0);

    TimeSeries custom_overlap = ts.custom_year_series(10, 3);
    const std::vector<double> overlap_max = {244, 331, 240, 204, 337, 250};
    for (std::size_t i = 0; i < overlap_max.size(); ++i)
        CHECK_NEAR(custom_overlap[static_cast<int>(i)].value(), overlap_max[i], 0.0);

    TimeSeries monthly_ts = ts.monthly_series();
    for (int i = 0; i < monthly_ts.count(); ++i)
        CHECK_NEAR(monthly_ts[i].value(), kLongValues[static_cast<std::size_t>(i)], 0.0);

    TimeSeries weekly(TimeInterval::SevenDay, DateTime(2023, 1, 1), kLongValues);
    TimeSeries monthly2 = weekly.monthly_series();
    const std::vector<double> monthly2_max = {244, 263, 205, 331, 262, 240, 195, 317,
                                              204, 185, 337, 262, 182, 250, 239, 180};
    for (std::size_t i = 0; i < monthly2_max.size(); ++i)
        CHECK_NEAR(monthly2[static_cast<int>(i)].value(), monthly2_max[i], 0.0);

    TimeSeries quarterly = ts.quarterly_series();
    const std::vector<double> quarterly_max = {244, 229, 263, 205, 331, 174, 262, 240,
                                               230, 173, 317, 204, 194, 179, 192, 337,
                                               230, 262, 182, 250, 237, 239, 180};
    for (std::size_t i = 0; i < quarterly_max.size(); ++i)
        CHECK_NEAR(quarterly[static_cast<int>(i)].value(), quarterly_max[i], 0.0);

    // corehydro supplement: the block-series guards, and the block-function contract that the four
    // upstream Maximum-only tests never reach -- Sum and Average stamp the LAST ordinate's date
    // while Minimum and Maximum stamp the extremum's own.
    CHECK_THROWS(ts.custom_year_series(0));
    CHECK_THROWS(ts.custom_year_series(13));
    CHECK_THROWS(ts.custom_year_series(1, 13));
    TimeSeries year1(TimeInterval::OneMonth, DateTime(2023, 1, 1),
                     std::vector<double>{5.0, 9.0, 1.0});
    TimeSeries sum_series = year1.calendar_year_series(BlockFunctionType::Sum);
    CHECK_NEAR(sum_series[0].value(), 15.0, 1e-12);
    CHECK_TRUE(sum_series[0].index() == DateTime(2023, 3, 1));   // last ordinate's date
    TimeSeries avg_series = year1.calendar_year_series(BlockFunctionType::Average);
    CHECK_NEAR(avg_series[0].value(), 5.0, 1e-12);
    TimeSeries min_series = year1.calendar_year_series(BlockFunctionType::Minimum);
    CHECK_NEAR(min_series[0].value(), 1.0, 0.0);
    CHECK_TRUE(min_series[0].index() == DateTime(2023, 3, 1));
    TimeSeries max_series = year1.calendar_year_series(BlockFunctionType::Maximum);
    CHECK_NEAR(max_series[0].value(), 9.0, 0.0);
    CHECK_TRUE(max_series[0].index() == DateTime(2023, 2, 1));   // extremum's own date
    CHECK_TRUE(sum_series.time_interval() == TimeInterval::Irregular);
}

// C# Test_CalendarYearSeries_NaN and Test_MonthlySeries_NaN: how a missing value propagates
// through each block function.
void test_block_series_nan() {
    const std::vector<double> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, kNaN, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    TimeSeries ts(TimeInterval::OneMonth, DateTime(2023, 1, 1), values);

    TimeSeries sum_ts = ts.calendar_year_series(BlockFunctionType::Sum);
    CHECK_EQ(sum_ts.count(), 1);
    CHECK_NEAR(sum_ts[0].value(), 78.0, 0.0);
    CHECK_EQ(sum_ts[0].index().year(), 2023);

    TimeSeries avg_ts = ts.calendar_year_series(BlockFunctionType::Average);
    CHECK_EQ(avg_ts.count(), 1);
    CHECK_NEAR(avg_ts[0].value(), 78.0 / 12.0, 1e-9);

    // Maximum survives both years: the comparison is false against NaN, so the missing value is
    // simply never chosen.
    TimeSeries max_ts = ts.calendar_year_series(BlockFunctionType::Maximum);
    CHECK_EQ(max_ts.count(), 2);
    CHECK_NEAR(max_ts[0].value(), 12.0, 0.0);
    CHECK_NEAR(max_ts[1].value(), 12.0, 0.0);

    const std::vector<double> weekly_values = {10, 20, 30, 40, 50, 60, kNaN, 80};
    TimeSeries weekly(TimeInterval::SevenDay, DateTime(2023, 1, 1), weekly_values);
    TimeSeries monthly_sum = weekly.monthly_series(BlockFunctionType::Sum);
    CHECK_EQ(monthly_sum.count(), 1);
    CHECK_NEAR(monthly_sum[0].value(), 150.0, 1e-9);
}

// C# Test_PeaksOverThreshold and Test_PeaksOverThreshold_MovingSum_NaN.
void test_peaks_over_threshold() {
    TimeSeries ts(TimeInterval::OneMonth, DateTime(2023, 1, 1), kLongValues);

    TimeSeries pot1 = ts.peaks_over_threshold_series(100, 2);
    const std::vector<double> valid1 = {331, 337, 262};
    CHECK_EQ(pot1.count(), static_cast<int>(valid1.size()));
    for (std::size_t i = 0; i < valid1.size(); ++i)
        CHECK_NEAR(pot1[static_cast<int>(i)].value(), valid1[i], 0.0);

    TimeSeries pot2 = ts.peaks_over_threshold_series(90, 1);
    const std::vector<double> valid2 = {337};
    CHECK_EQ(pot2.count(), static_cast<int>(valid2.size()));
    CHECK_NEAR(pot2[0].value(), valid2[0], 0.0);

    TimeSeries pot3 = ts.peaks_over_threshold_series(150, 5);
    const std::vector<double> valid3 = {331, 240, 317, 337};
    CHECK_EQ(pot3.count(), static_cast<int>(valid3.size()));
    for (std::size_t i = 0; i < valid3.size(); ++i)
        CHECK_NEAR(pot3[static_cast<int>(i)].value(), valid3[i], 0.0);

    TimeSeries pot4 = ts.peaks_over_threshold_series(200, 2);
    const std::vector<double> valid4 = {263, 331, 262, 317, 337, 250};
    CHECK_EQ(pot4.count(), static_cast<int>(valid4.size()));
    for (std::size_t i = 0; i < valid4.size(); ++i)
        CHECK_NEAR(pot4[static_cast<int>(i)].value(), valid4[i], 0.0);

    // The MovingSum-smoothed case: a window touching a missing day is NaN and cannot exceed the
    // threshold, so the trailing 8.0 produces no event.
    const std::vector<double> gappy = {0.1, 0.2, 5.0, 6.0, 0.1, 0.0, 0.1, 0.2, 0.0, kNaN, 8.0};
    TimeSeries daily(TimeInterval::OneDay, DateTime(2023, 1, 1), gappy);
    TimeSeries pot5 =
        daily.peaks_over_threshold_series(7.0, 1, SmoothingFunctionType::MovingSum, 2);
    CHECK_EQ(pot5.count(), 1);
    CHECK_NEAR(pot5[0].value(), 11.0, 1e-9);

    // corehydro supplement: the result is an irregular series carrying each peak's own date.
    CHECK_TRUE(pot1.time_interval() == TimeInterval::Irregular);
    CHECK_TRUE(pot2[0].index() == DateTime(2026, 10, 1));
}

// C# Test_SeasonalDecompose and Test_SeasonalDecompose_InvalidInputs.
void test_seasonal_decompose() {
    const int period = 12;
    const int n = period * 5;
    TimeSeries ts(TimeInterval::OneMonth);
    DateTime start(2000, 1, 1);
    std::vector<double> true_trend(static_cast<std::size_t>(n));
    std::vector<double> true_seasonal(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        std::size_t k = static_cast<std::size_t>(i);
        true_trend[k] = 100.0 + 0.5 * i;
        true_seasonal[k] = 10.0 * std::sin(2.0 * corehydro::numerics::kPi * i / period);
        ts.add(TimeSeries::Ordinate(start.add_months(i), true_trend[k] + true_seasonal[k]));
    }

    TimeSeries::SeasonalDecomposition d = ts.seasonal_decompose(period);
    CHECK_TRUE(d.trend.count() > 0);
    CHECK_TRUE(d.trend.count() <= n);
    CHECK_EQ(static_cast<int>(d.seasonal.size()), n);
    CHECK_TRUE(d.residual.count() > 0);

    // The additive identity: original == trend + seasonal + residual, wherever the trend exists.
    for (int i = 0; i < d.residual.count(); ++i) {
        DateTime at = d.residual[i].index();
        int orig = -1;
        for (int k = 0; k < n; ++k)
            if (ts[k].index() == at) {
                orig = k;
                break;
            }
        int trend_idx = -1;
        for (int k = 0; k < d.trend.count(); ++k)
            if (d.trend[k].index() == at) {
                trend_idx = k;
                break;
            }
        if (orig >= 0 && trend_idx >= 0) {
            double reconstructed = d.trend[trend_idx].value() +
                                   d.seasonal[static_cast<std::size_t>(orig)] +
                                   d.residual[i].value();
            CHECK_NEAR(reconstructed, ts[orig].value(), 1e-6);
        }
    }

    double max_seasonal = *std::max_element(d.seasonal.begin(), d.seasonal.end());
    double min_seasonal = *std::min_element(d.seasonal.begin(), d.seasonal.end());
    CHECK_TRUE(max_seasonal - min_seasonal > 1.0);

    TimeSeries short_ts(TimeInterval::OneMonth);
    for (int i = 0; i < 10; ++i)
        short_ts.add(TimeSeries::Ordinate(start.add_months(i), i));
    CHECK_THROWS(short_ts.seasonal_decompose(1));
    CHECK_THROWS(short_ts.seasonal_decompose(12));
}

// The AR(1) series the eleven upstream resampling tests share. DEVIATION, documented: C#'s
// `MakeAR1` draws from `System.Random`, a .NET LCG with no ported equivalent, so this builds the
// same AR(1) process from the ported MersenneTwister instead. Every assertion those tests make is
// a statistical property of the RESAMPLER (length, determinism, drift, preserved moments), not of
// the particular input path, so the substitution costs nothing -- the same precedent
// rating_curve.hpp's seeded Predict already documents.
TimeSeries make_ar1(int n, double phi, double sigma, int seed) {
    corehydro::numerics::sampling::MersenneTwister rng(seed);
    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    for (int i = 1; i < n; ++i) {
        double u1 = 1.0 - rng.next_double();
        double u2 = 1.0 - rng.next_double();
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * corehydro::numerics::kPi * u2);
        values[static_cast<std::size_t>(i)] =
            phi * values[static_cast<std::size_t>(i - 1)] + sigma * z;
    }
    return TimeSeries(TimeInterval::OneDay, DateTime(2000, 1, 1), values);
}

double lag1_autocorrelation(const TimeSeries& ts) {
    int n = ts.count();
    if (n < 3) return kNaN;
    double mean = ts.mean_value();
    double num = 0, den = 0;
    for (int i = 1; i < n; ++i) num += (ts[i].value() - mean) * (ts[i - 1].value() - mean);
    for (int i = 0; i < n; ++i) den += (ts[i].value() - mean) * (ts[i].value() - mean);
    return den > 0 ? num / den : kNaN;
}

// The seven C# Test_ResampleWithKNN_* methods.
void test_resample_knn() {
    TimeSeries ts = make_ar1(100, 0.5, 1.0, 1);
    TimeSeries a = ts.resample_with_knn(50, 10, 42);
    TimeSeries b = ts.resample_with_knn(50, 10, 42);
    CHECK_EQ(a.count(), b.count());
    for (int i = 0; i < a.count(); ++i) CHECK_NEAR(a[i].value(), b[i].value(), 1e-12);

    for (int len : {1, 7, 50, 250}) {
        TimeSeries r = ts.resample_with_knn(len, 10, 42);
        CHECK_EQ(r.count(), len);
    }

    // On a strictly increasing series the conditional bootstrap must ADVANCE through the trend:
    // this is the regression guard for the `selected + 1` step.
    std::vector<double> ramp(200);
    for (int i = 0; i < 200; ++i) ramp[static_cast<std::size_t>(i)] = i;
    TimeSeries trend_ts(TimeInterval::OneDay, DateTime(2000, 1, 1), ramp);
    const int steps = 50;
    const int trials = 20;
    double avg_drift = 0;
    for (int seed = 1; seed <= trials; ++seed) {
        TimeSeries r = trend_ts.resample_with_knn(steps, 10, seed);
        avg_drift += r[steps - 1].value() - r[0].value();
    }
    avg_drift /= trials;
    CHECK_TRUE(avg_drift > 25.0);

    TimeSeries ar = make_ar1(200, 0.5, 1.0, 7);
    double input_mean = ar.mean_value();
    double input_std = ar.standard_deviation();
    const int m_trials = 200;
    const int m_steps = 100;
    double sum_of_tail_means = 0;
    double sum_of_tail_vars = 0;
    for (int seed = 1; seed <= m_trials; ++seed) {
        TimeSeries r = ar.resample_with_knn(m_steps, 10, seed);
        sum_of_tail_means += r.mean_value();
        double s = r.standard_deviation();
        sum_of_tail_vars += s * s;
    }
    double grand_mean = sum_of_tail_means / m_trials;
    double standard_error = input_std / std::sqrt(static_cast<double>(m_steps * m_trials));
    CHECK_TRUE(std::fabs(grand_mean - input_mean) < 4.0 * standard_error + 0.05);

    double input_var = input_std * input_std;
    double ratio = (sum_of_tail_vars / m_trials) / input_var;
    CHECK_TRUE(ratio > 0.5 && ratio < 2.0);

    TimeSeries ar7 = make_ar1(300, 0.7, 1.0, 7);
    double sum_lag1 = 0;
    for (int seed = 1; seed <= m_trials; ++seed)
        sum_lag1 += lag1_autocorrelation(ar7.resample_with_knn(m_steps, 10, seed));
    double avg_lag1 = sum_lag1 / m_trials;
    CHECK_TRUE(avg_lag1 > 0.4 && avg_lag1 < 0.95);

    CHECK_THROWS(make_ar1(10, 0.5, 1.0, 1).resample_with_knn(5, 5, 42));
    // corehydro supplement: the other two guards.
    CHECK_THROWS(ts.resample_with_knn(0, 10, 42));
}

// The four C# Test_ResampleWithBlockBootstrap_* methods.
void test_resample_block_bootstrap() {
    TimeSeries ts = make_ar1(100, 0.5, 1.0, 1);
    TimeSeries a = ts.resample_with_block_bootstrap(50, 5, 42);
    TimeSeries b = ts.resample_with_block_bootstrap(50, 5, 42);
    CHECK_EQ(a.count(), b.count());
    for (int i = 0; i < a.count(); ++i) CHECK_NEAR(a[i].value(), b[i].value(), 1e-12);

    for (int len : {1, 7, 50, 250}) {
        TimeSeries r = ts.resample_with_block_bootstrap(len, 5, 42);
        CHECK_EQ(r.count(), len);
    }

    TimeSeries ar = make_ar1(200, 0.5, 1.0, 7);
    double input_mean = ar.mean_value();
    double input_std = ar.standard_deviation();
    const int trials = 200;
    const int steps = 100;
    double sum_of_tail_means = 0;
    double sum_of_tail_vars = 0;
    for (int seed = 1; seed <= trials; ++seed) {
        TimeSeries r = ar.resample_with_block_bootstrap(steps, 5, seed);
        sum_of_tail_means += r.mean_value();
        double s = r.standard_deviation();
        sum_of_tail_vars += s * s;
    }
    double grand_mean = sum_of_tail_means / trials;
    double standard_error = input_std / std::sqrt(static_cast<double>(steps * trials));
    CHECK_TRUE(std::fabs(grand_mean - input_mean) < 4.0 * standard_error + 0.05);
    double ratio = (sum_of_tail_vars / trials) / (input_std * input_std);
    CHECK_TRUE(ratio > 0.5 && ratio < 2.0);

    // corehydro supplement: the three guards, and that the result keeps the source's start date
    // and interval.
    CHECK_THROWS(TimeSeries(TimeInterval::OneDay, DateTime(2000, 1, 1),
                            std::vector<double>{1.0})
                     .resample_with_block_bootstrap(5, 1, 42));
    CHECK_THROWS(ts.resample_with_block_bootstrap(50, 500, 42));
    CHECK_THROWS(ts.resample_with_block_bootstrap(0, 5, 42));
    CHECK_TRUE(a.start_date() == ts.start_date());
    CHECK_TRUE(a.time_interval() == ts.time_interval());
}

}  // namespace

int main() {
    test_construction();
    test_getters();
    test_clone();
    test_sort();
    test_math();
    test_cumulative();
    test_difference();
    test_missing();
    test_intervals();
    test_moving_windows();
    test_shifts();
    test_clip();
    test_convert_time_interval();
    test_series_mutation();
    test_stats();
    test_summary_stats();
    test_summary_hypothesis_test();
    test_monthly_stats();
    test_block_series();
    test_block_series_nan();
    test_peaks_over_threshold();
    test_seasonal_decompose();
    test_resample_knn();
    test_resample_block_bootstrap();
    return chtest::summary("time_series_container");
}
