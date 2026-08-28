// P6 support ctest (C++-only): the `DateTime` value type the ported TimeSeries container indexes
// by. `DateTime` is a corehydro ADDITION -- it stands in for `System.DateTime`, which has no
// upstream source file to port -- so its oracle is the real .NET BCL rather than a C# test literal.
//
// EVERY expected value below was produced by a throwaway `dotnet run` probe against
// `System.DateTime` on .NET 10 (2026-08-27) and copied here unaltered. The probe is not committed
// (it has no dependency on the pinned submodules and nothing to re-run against), but it is
// reproducible in ten lines: construct the same dates, print `ToString("o")` and `.Ticks`. Where a
// value here disagrees with intuition -- `AddDays(12.3456789)` landing one tick BELOW the obvious
// product, `ToOADate()` of a pre-1899 date carrying a sign-extended day and a positive time -- the
// probe governs and the comment says what the probe measured.
//
// The `add_days` algorithm is the one measurement worth carrying forward. Modern .NET does NOT
// compute `(long)(value * TicksPerDay)`: over 300,000 random values that model disagreed 192 times,
// always by exactly one tick. What reproduces the BCL EXACTLY, 0 mismatches over 300,000 draws for
// all five fractional Add* methods, is a whole/fraction split --
//     (long)trunc(v) * ticks_per_unit + (long)((v - trunc(v)) * ticks_per_unit)
// -- which is what `date_time.hpp` implements. `PointProcessModel::generate_pot_time_series` calls
// `AddDays` with a fractional offset and then reads `DayOfYear` off the result, so this is not
// academic: a one-tick difference across midnight moves an event into the other season.
#include <cmath>
#include <limits>
#include <string>

#include "check.hpp"
#include "corehydro/numerics/data/time_series/support/date_time.hpp"

using corehydro::numerics::data::DateTime;

namespace {

// Compares a DateTime against its calendar components, as the probe printed them.
void check_parts(const DateTime& d, int year, int month, int day, int hour = 0, int minute = 0,
                 int second = 0) {
    CHECK_EQ(d.year(), year);
    CHECK_EQ(d.month(), month);
    CHECK_EQ(d.day(), day);
    CHECK_EQ(d.hour(), hour);
    CHECK_EQ(d.minute(), minute);
    CHECK_EQ(d.second(), second);
}

// 1. Construction, ticks, and the calendar components.
void test_construction_and_components() {
    DateTime a(2023, 1, 1);
    check_parts(a, 2023, 1, 1);
    CHECK_EQ(a.ticks(), static_cast<int64_t>(638081280000000000LL));
    CHECK_EQ(a.day_of_year(), 1);

    CHECK_EQ(DateTime(2024, 12, 31).day_of_year(), 366);   // leap
    CHECK_EQ(DateTime(2023, 12, 31).day_of_year(), 365);
    CHECK_EQ(DateTime(2024, 12, 31).ticks(), static_cast<int64_t>(638712000000000000LL));

    DateTime feb29(2024, 2, 29);
    check_parts(feb29, 2024, 2, 29);
    CHECK_EQ(feb29.ticks(), static_cast<int64_t>(638447616000000000LL));
    CHECK_EQ(feb29.day_of_year(), 60);
    CHECK_EQ(feb29.day_of_week(), 4);   // Thursday, with Sunday == 0

    // default(DateTime) is tick 0 == 0001-01-01, which the seasonal PointProcess path tests for.
    DateTime zero;
    CHECK_EQ(zero.ticks(), static_cast<int64_t>(0));
    check_parts(zero, 1, 1, 1);
    CHECK_EQ(zero.day_of_week(), 1);   // Monday
    CHECK_TRUE(zero == DateTime::min_value());

    CHECK_EQ(DateTime::max_value().ticks(), static_cast<int64_t>(3155378975999999999LL));
    check_parts(DateTime::max_value(), 9999, 12, 31, 23, 59, 59);
    CHECK_EQ(DateTime::max_value().day_of_week(), 5);   // Friday

    // Time of day survives construction and is readable back.
    DateTime t(2023, 5, 17, 4, 3, 2, 123);
    check_parts(t, 2023, 5, 17, 4, 3, 2);
    CHECK_EQ(t.millisecond(), 123);

    CHECK_EQ(DateTime::days_in_month(2024, 2), 29);
    CHECK_EQ(DateTime::days_in_month(2023, 2), 28);
    CHECK_EQ(DateTime::days_in_month(2023, 12), 31);
    CHECK_TRUE(DateTime::is_leap_year(2000));
    CHECK_TRUE(!DateTime::is_leap_year(1900));
    CHECK_TRUE(DateTime::is_leap_year(2024));
}

// 2. The constructor's range guard. .NET raises ArgumentOutOfRangeException for every one of
// these with the same message; the port throws std::out_of_range carrying it.
void test_construction_guards() {
    CHECK_THROWS(DateTime(2023, 0, 1));
    CHECK_THROWS(DateTime(2023, 13, 1));
    CHECK_THROWS(DateTime(2023, 1, 0));
    CHECK_THROWS(DateTime(2023, 1, 32));
    CHECK_THROWS(DateTime(2023, 2, 29));    // not a leap year
    CHECK_THROWS(DateTime(0, 1, 1));
    CHECK_THROWS(DateTime(10000, 1, 1));
    CHECK_THROWS_MSG(DateTime(2023, 2, 29), "un-representable");
    CHECK_THROWS(DateTime(2023, 1, 1, 24, 0, 0));
    CHECK_THROWS(DateTime(2023, 1, 1, 0, 60, 0));
    CHECK_THROWS(DateTime(static_cast<int64_t>(-1)));
    CHECK_THROWS(DateTime(static_cast<int64_t>(3155378976000000000LL)));
}

// 3. AddMonths clamps the day to the target month and keeps the time of day.
void test_add_months() {
    DateTime t(2023, 1, 31, 13, 45, 30);
    DateTime a = t.add_months(1);
    check_parts(a, 2023, 2, 28, 13, 45, 30);
    CHECK_EQ(a.ticks(), static_cast<int64_t>(638131887300000000LL));

    check_parts(DateTime(2024, 1, 31).add_months(1), 2024, 2, 29);
    check_parts(DateTime(2023, 3, 31).add_months(-1), 2023, 2, 28);
    check_parts(DateTime(2023, 1, 31).add_months(12), 2024, 1, 31);
    check_parts(DateTime(2023, 1, 31).add_months(0), 2023, 1, 31);
    check_parts(DateTime(2023, 1, 15).add_months(-14), 2021, 11, 15);

    // The clamp is not sticky: walking twelve months from January 31 does NOT recover 31 after
    // February. These twelve dates are the probe's, in order.
    const int expected_month[12] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1};
    const int expected_day[12] = {28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28};
    DateTime w(2023, 1, 31);
    for (int i = 0; i < 12; ++i) {
        w = w.add_months(1);
        CHECK_EQ(w.month(), expected_month[i]);
        CHECK_EQ(w.day(), expected_day[i]);
    }
    CHECK_EQ(w.year(), 2024);

    CHECK_THROWS(DateTime(2023, 1, 1).add_months(120001));
    CHECK_THROWS(DateTime(2023, 1, 1).add_months(-120001));
    CHECK_THROWS(DateTime::max_value().add_months(1));
    CHECK_THROWS(DateTime::min_value().add_months(-1));
}

// 4. AddYears is AddMonths(value * 12) under its own range check.
void test_add_years() {
    check_parts(DateTime(2024, 2, 29).add_years(1), 2025, 2, 28);
    check_parts(DateTime(2024, 2, 29).add_years(-1), 2023, 2, 28);
    check_parts(DateTime(2023, 6, 15, 8, 30, 0).add_years(2), 2025, 6, 15, 8, 30, 0);
    CHECK_THROWS(DateTime(2023, 1, 1).add_years(10001));
}

// 5. The fractional adds, including the whole/fraction split described in the file header.
void test_fractional_adds() {
    DateTime b(2000, 1, 1);
    check_parts(b.add_days(1.5), 2000, 1, 2, 12, 0, 0);
    check_parts(b.add_days(-1.5), 1999, 12, 30, 12, 0, 0);
    check_parts(b.add_days(1.0 / 3.0), 2000, 1, 1, 8, 0, 0);

    CHECK_EQ(b.add_days(0.0000001).ticks() - b.ticks(), static_cast<int64_t>(86400));
    CHECK_EQ(b.add_days(0.4999999).ticks() - b.ticks(), static_cast<int64_t>(431999913600LL));
    CHECK_EQ(b.add_days(1e-9).ticks() - b.ticks(), static_cast<int64_t>(864));

    // The discriminating case: the naive single-multiply model gives ...600 here, the BCL gives
    // ...599, and the split model reproduces the BCL.
    CHECK_EQ(b.add_days(12.3456789).ticks() - b.ticks(),
             static_cast<int64_t>(10666666569599LL));
    CHECK_EQ(b.add_days(-12.3456789).ticks() - b.ticks(),
             static_cast<int64_t>(-10666666569599LL));
    CHECK_EQ(b.add_days(0.123456789012345).ticks() - b.ticks(),
             static_cast<int64_t>(106666665706LL));
    CHECK_EQ(b.add_days(365.2425).ticks() - b.ticks(), static_cast<int64_t>(315569520000000LL));

    CHECK_EQ(b.add_hours(-2.7182818).ticks() - b.ticks(), static_cast<int64_t>(-97858144800LL));
    CHECK_EQ(b.add_minutes(-2.7182818).ticks() - b.ticks(), static_cast<int64_t>(-1630969080LL));
    CHECK_EQ(b.add_milliseconds(1.4999).ticks() - b.ticks(), static_cast<int64_t>(14999));
    CHECK_EQ(b.add_seconds(1.5).ticks() - b.ticks(), static_cast<int64_t>(15000000));

    // NaN adds nothing and does NOT throw (the BCL's saturating double-to-long conversion);
    // an infinity trips the value-range guard.
    CHECK_TRUE(b.add_days(std::numeric_limits<double>::quiet_NaN()) == b);
    CHECK_TRUE(b.add_seconds(std::numeric_limits<double>::quiet_NaN()) == b);
    CHECK_THROWS(b.add_days(std::numeric_limits<double>::infinity()));

    // Two distinct guards, exactly as the BCL has them: 3652058 days is a representable OFFSET
    // that lands out of range from this date, 3652059 days is not a representable offset at all.
    CHECK_THROWS_MSG(b.add_days(3652058.0), "un-representable");
    CHECK_THROWS_MSG(b.add_days(3652059.0), "out of range");
    check_parts(b.add_days(-730119.0), 1, 1, 1);
    CHECK_THROWS(b.add_days(-730120.0));
    CHECK_THROWS(DateTime::max_value().add_ticks(1));
    CHECK_THROWS(DateTime::min_value().add_days(-1));
}

// 6. ToOADate, which both InterpolateMissingData overloads interpolate in.
void test_to_oa_date() {
    CHECK_NEAR(DateTime(1899, 12, 30).to_oa_date(), 0.0, 0.0);
    CHECK_NEAR(DateTime(1900, 1, 1).to_oa_date(), 2.0, 0.0);
    CHECK_NEAR(DateTime(2023, 1, 1).to_oa_date(), 44927.0, 0.0);
    CHECK_NEAR(DateTime(2023, 1, 1, 12, 0, 0).to_oa_date(), 44927.5, 0.0);
    CHECK_NEAR(DateTime(1800, 6, 15).to_oa_date(), -36357.0, 0.0);
    // Tick 0 short-circuits to 0.0 rather than taking the negative branch.
    CHECK_NEAR(DateTime().to_oa_date(), 0.0, 0.0);
    // A pre-1899 time of day is stored as a POSITIVE fraction on a negative day count:
    // 1899-12-29T06:00 is 0.75 days before the epoch but reports -1.25.
    CHECK_NEAR(DateTime(1899, 12, 29, 6, 0, 0).to_oa_date(), -1.25, 0.0);
}

// 7. Comparison, ordering, and the difference the ConvertTimeInterval loop bound needs.
void test_comparison_and_difference() {
    DateTime a(2023, 1, 1);
    DateTime b(2023, 12, 31, 18, 30, 0);
    CHECK_TRUE(a < b);
    CHECK_TRUE(b > a);
    CHECK_TRUE(a <= DateTime(2023, 1, 1));
    CHECK_TRUE(a >= DateTime(2023, 1, 1));
    CHECK_TRUE(a == DateTime(2023, 1, 1));
    CHECK_TRUE(a != b);
    CHECK_TRUE(a.compare_to(b) < 0);
    CHECK_TRUE(b.compare_to(a) > 0);
    CHECK_EQ(a.compare_to(DateTime(2023, 1, 1)), 0);
    CHECK_TRUE(DateTime::min_value() < DateTime::max_value());

    CHECK_NEAR(b.subtract_total_hours(a), 8754.5, 0.0);
    CHECK_NEAR(a.subtract_total_hours(b), -8754.5, 0.0);
}

// 8. The corehydro additions that carry a date across the R/Python boundary. Epoch seconds are
// the wire format (see the P6 plan's scope decision 4); ticks are not representable in a double.
void test_corehydro_additions() {
    CHECK_EQ(DateTime(1970, 1, 1).ticks(), static_cast<int64_t>(621355968000000000LL));
    CHECK_NEAR(DateTime(1970, 1, 1).to_unix_seconds(), 0.0, 0.0);
    CHECK_NEAR(DateTime(2023, 1, 1).to_unix_seconds(), 1672531200.0, 0.0);
    CHECK_NEAR(DateTime(1969, 12, 31, 23, 59, 59).to_unix_seconds(), -1.0, 0.0);

    DateTime round_trip = DateTime::from_unix_seconds(DateTime(2023, 5, 17, 4, 3, 2).to_unix_seconds());
    CHECK_TRUE(round_trip == DateTime(2023, 5, 17, 4, 3, 2));

    CHECK_EQ(DateTime(2023, 5, 17, 4, 3, 2, 123).to_iso_string(),
             std::string("2023-05-17T04:03:02.1230000"));
    CHECK_EQ(DateTime(2023, 1, 1).to_iso_string(), std::string("2023-01-01T00:00:00.0000000"));
    CHECK_TRUE(DateTime::parse_iso("2023-05-17T04:03:02.1230000") ==
               DateTime(2023, 5, 17, 4, 3, 2, 123));
    CHECK_TRUE(DateTime::parse_iso("2023-05-17") == DateTime(2023, 5, 17));
    CHECK_TRUE(DateTime::parse_iso("2023-05-17T04:03:02") == DateTime(2023, 5, 17, 4, 3, 2));
    CHECK_THROWS(DateTime::parse_iso("not a date"));
    CHECK_THROWS(DateTime::parse_iso("2023-13-01"));
}

}  // namespace

int main() {
    test_construction_and_components();
    test_construction_guards();
    test_add_months();
    test_add_years();
    test_fractional_adds();
    test_to_oa_date();
    test_comparison_and_difference();
    test_corehydro_additions();
    return chtest::summary("date_time");
}
