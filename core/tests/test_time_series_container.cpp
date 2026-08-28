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
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "data/time_series_convert_data.hpp"

using corehydro::numerics::data::BlockFunctionType;
using corehydro::numerics::data::DateTime;
using corehydro::numerics::data::ListSortDirection;
using corehydro::numerics::data::TimeInterval;
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
    return chtest::summary("time_series_container");
}
