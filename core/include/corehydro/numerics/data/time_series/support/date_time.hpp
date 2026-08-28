// corehydro ADDITION -- no upstream C# file. This stands in for `System.DateTime`, which the
// Numerics `TimeSeries : Series<DateTime, double>` container indexes by and which the BCL, not
// Numerics, provides. The container is calendar-driven (block series group by `Index.Year` and
// `Index.Month`, `AddTimeInterval` walks by `AddMonths(1)` / `AddYears(1)`, the seasonal
// PointProcess path reads `DayOfYear`), and the core takes no external dependency, so the subset
// of `System.DateTime` those call sites use is reproduced here.
//
// It is a MIRROR, not a redesign. Every behaviour below was measured against the real BCL
// (.NET 10, 2026-08-27) with a throwaway `dotnet run` probe, and `core/tests/test_date_time.cpp`
// carries the measured values as its oracle. Where the BCL is surprising, this port is surprising
// in the same way:
//
//  1. TICKS. 100-nanosecond units since 0001-01-01T00:00:00, in an int64. `default(DateTime)` is
//     tick 0, which `PointProcessModel.SetAMSData` tests for with `dt == default` to decide
//     whether an ordinate carries a real date -- so a default-constructed DateTime MUST be tick 0
//     and MUST equal `min_value()`.
//  2. THE FRACTIONAL ADDS SPLIT WHOLE FROM FRACTION. Modern .NET does not evaluate
//     `(long)(value * TicksPerDay)`: across 300,000 random values that model disagreed with the
//     BCL 192 times, always by exactly one tick. What reproduces the BCL exactly -- 0 mismatches
//     over 300,000 draws for AddDays, AddHours, AddMinutes, AddSeconds and AddMilliseconds alike
//     -- is `(long)trunc(v) * ticks_per_unit + (long)((v - trunc(v)) * ticks_per_unit)`, which is
//     what `add_units` below does. This is load-bearing:
//     `PointProcessModel::generate_pot_time_series` adds a fractional day offset and then reads
//     `day_of_year()` off the result to pick a season, so a one-tick difference across midnight
//     moves an event into the other season.
//  3. TWO DISTINCT RANGE GUARDS, with two distinct messages, exactly as the BCL has them. A
//     fractional add whose OFFSET cannot be represented at all ("Value to add was out of range.")
//     is rejected before a result that merely falls off the calendar ("The added or subtracted
//     value results in an un-representable DateTime."). From 2000-01-01, `add_days(3652058)`
//     raises the second and `add_days(3652059)` raises the first.
//  4. NaN ADDS NOTHING AND DOES NOT THROW. The BCL's double-to-long conversion saturates NaN to
//     zero, so `AddDays(NaN)` returns the same instant. C++'s conversion is undefined for NaN, so
//     `add_units` special-cases it rather than inheriting undefined behaviour. An infinity still
//     trips the value guard.
//  5. `add_months` CLAMPS THE DAY and keeps the time of day; `add_years(v)` is `add_months(v * 12)`
//     under its own +/-10000 guard. The clamp is not sticky: twelve successive `add_months(1)`
//     from January 31 land on the 28th every month, not back on the 31st.
//  6. `to_oa_date` reproduces the OLE Automation convention including its sign quirk: a pre-1899
//     time of day is a POSITIVE fraction on a NEGATIVE day count, so 1899-12-29T06:00 -- which is
//     0.75 days before the OA epoch -- reports -1.25. Both `TimeSeries::interpolate_missing_data`
//     overloads interpolate in this space, so the quirk is oracle-visible.
//
// Deliberately NOT ported (no call site in this port's scope): DateTimeKind and time zones (every
// date in the container is `Unspecified` in C#, and inventing a zone would invent behaviour),
// culture-aware parsing and formatting, `Ticks`-level arithmetic operators returning `TimeSpan`
// (the one consumer, `ConvertTimeInterval`, needs `Subtract(...).TotalHours` and nothing else),
// and the calendar classes. `from_unix_seconds` / `to_unix_seconds` / `to_iso_string` /
// `parse_iso` are corehydro additions with no BCL counterpart in the container: they are how a
// date crosses into R (POSIXct) and Python (datetime), and epoch SECONDS are the wire format
// because a tick count near 6.3e17 is not exactly representable in a double.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace corehydro::numerics::data {

class DateTime {
   public:
    // --- Tick constants (BCL `DateTime` internals, same names). ---
    static constexpr int64_t kTicksPerMillisecond = 10000;
    static constexpr int64_t kTicksPerSecond = kTicksPerMillisecond * 1000;
    static constexpr int64_t kTicksPerMinute = kTicksPerSecond * 60;
    static constexpr int64_t kTicksPerHour = kTicksPerMinute * 60;
    static constexpr int64_t kTicksPerDay = kTicksPerHour * 24;
    static constexpr int64_t kMaxTicks = 3155378975999999999LL;   // 9999-12-31T23:59:59.9999999
    static constexpr int64_t kMillisPerDay = 86400000LL;
    // Ticks of 1899-12-30, the OLE Automation epoch (`DateTime.DoubleDateOffset`).
    static constexpr int64_t kDoubleDateOffset = 599264352000000000LL;
    // Ticks of 0100-01-01 (`DateTime.OADateMinAsTicks`): the BCL rejects an earlier OA date.
    static constexpr int64_t kOADateMinAsTicks = 31241376000000000LL;
    // Ticks of 1970-01-01, for the corehydro epoch-seconds conversions.
    static constexpr int64_t kUnixEpochTicks = 621355968000000000LL;

    // Constructs 0001-01-01T00:00:00 (C# `default(DateTime)`; see note 1).
    DateTime() = default;

    // Constructs from a raw tick count (C# `DateTime(long ticks)`).
    explicit DateTime(int64_t ticks) : ticks_(ticks) {
        if (ticks < 0 || ticks > kMaxTicks)
            throw std::out_of_range(
                "Ticks must be between DateTime.MinValue.Ticks and DateTime.MaxValue.Ticks.");
    }

    // Constructs from calendar components (C# `DateTime(int, int, int, ...)`). Every invalid
    // combination -- month 0 or 13, day 0 or past the month's length, a non-leap February 29,
    // year 0 or 10000 -- raises the one BCL message.
    DateTime(int year, int month, int day, int hour = 0, int minute = 0, int second = 0,
             int millisecond = 0) {
        if (year < 1 || year > 9999 || month < 1 || month > 12 || day < 1 ||
            day > days_in_month(year, month))
            throw std::out_of_range(
                "Year, Month, and Day parameters describe an un-representable DateTime.");
        if (hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 59 ||
            millisecond < 0 || millisecond > 999)
            throw std::out_of_range(
                "Hour, Minute, Second, and Millisecond parameters describe an un-representable "
                "DateTime.");
        ticks_ = day_number(year, month, day) * kTicksPerDay + hour * kTicksPerHour +
                 minute * kTicksPerMinute + second * kTicksPerSecond +
                 millisecond * kTicksPerMillisecond;
    }

    // --- Statics (C# `DateTime.MinValue` / `MaxValue` / `IsLeapYear` / `DaysInMonth`). ---
    static DateTime min_value() { return DateTime(); }
    static DateTime max_value() { return DateTime(kMaxTicks); }

    static bool is_leap_year(int year) {
        return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    }

    static int days_in_month(int year, int month) {
        static constexpr int kDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        if (month < 1 || month > 12) return 0;
        if (month == 2 && is_leap_year(year)) return 29;
        return kDays[month - 1];
    }

    // --- Components (C# properties of the same names). ---
    int64_t ticks() const { return ticks_; }
    int year() const { return civil().y; }
    int month() const { return civil().m; }
    int day() const { return civil().d; }
    int hour() const { return static_cast<int>(ticks_ / kTicksPerHour % 24); }
    int minute() const { return static_cast<int>(ticks_ / kTicksPerMinute % 60); }
    int second() const { return static_cast<int>(ticks_ / kTicksPerSecond % 60); }
    int millisecond() const { return static_cast<int>(ticks_ / kTicksPerMillisecond % 1000); }

    // Day of the year, 1-366 (C# `DayOfYear`).
    int day_of_year() const {
        Civil c = civil();
        return static_cast<int>(day_number(c.y, c.m, c.d) - day_number(c.y, 1, 1)) + 1;
    }

    // Day of the week with Sunday == 0 (C# `(int)DayOfWeek`, whose formula this is).
    int day_of_week() const { return static_cast<int>((ticks_ / kTicksPerDay + 1) % 7); }

    // Midnight of the same day, and the ticks since it (C# `Date` / `TimeOfDay.Ticks`).
    DateTime date() const { return DateTime(ticks_ - ticks_ % kTicksPerDay); }
    int64_t time_of_day_ticks() const { return ticks_ % kTicksPerDay; }

    // --- Arithmetic (C# `AddTicks` / `AddX` / `AddMonths` / `AddYears`). ---

    DateTime add_ticks(int64_t value) const {
        if (value > kMaxTicks - ticks_ || value < -ticks_)
            throw std::out_of_range(
                "The added or subtracted value results in an un-representable DateTime.");
        return DateTime(ticks_ + value);
    }

    DateTime add_milliseconds(double value) const { return add_units(value, kTicksPerMillisecond); }
    DateTime add_seconds(double value) const { return add_units(value, kTicksPerSecond); }
    DateTime add_minutes(double value) const { return add_units(value, kTicksPerMinute); }
    DateTime add_hours(double value) const { return add_units(value, kTicksPerHour); }
    DateTime add_days(double value) const { return add_units(value, kTicksPerDay); }

    // Adds whole months, clamping the day to the target month's length and keeping the time of
    // day (C# `AddMonths`; see note 5).
    DateTime add_months(int months) const {
        if (months < -120000 || months > 120000)
            throw std::out_of_range("Months value must be between +/-120000.");
        Civil c = civil();
        int y = c.y;
        int m = c.m;
        int d = c.d;
        int i = m - 1 + months;
        if (i >= 0) {
            m = i % 12 + 1;
            y = y + i / 12;
        } else {
            m = 12 + (i + 1) % 12;
            y = y + (i - 11) / 12;
        }
        if (y < 1 || y > 9999)
            throw std::out_of_range(
                "The added or subtracted value results in an un-representable DateTime.");
        int days = days_in_month(y, m);
        if (d > days) d = days;
        return DateTime(day_number(y, m, d) * kTicksPerDay + time_of_day_ticks());
    }

    // C# `AddYears`, which is `AddMonths(value * 12)` under its own guard.
    DateTime add_years(int value) const {
        if (value < -10000 || value > 10000)
            throw std::out_of_range("Years value must be between +/-10000.");
        return add_months(value * 12);
    }

    // Elapsed hours from `other` to this instant (C# `Subtract(other).TotalHours`, the only
    // member of the returned TimeSpan any ported call site reads).
    double subtract_total_hours(const DateTime& other) const {
        return static_cast<double>(ticks_ - other.ticks_) / static_cast<double>(kTicksPerHour);
    }

    // The OLE Automation date (C# `ToOADate`; see note 6).
    double to_oa_date() const {
        if (ticks_ == 0) return 0.0;
        int64_t value = ticks_;
        if (value < kTicksPerDay) value += kDoubleDateOffset;
        if (value < kOADateMinAsTicks)
            throw std::out_of_range("Not a valid OLE Automation Date.");
        int64_t millis = (value - kDoubleDateOffset) / kTicksPerMillisecond;
        if (millis < 0) {
            int64_t frac = millis % kMillisPerDay;
            if (frac != 0) millis -= (kMillisPerDay + frac) * 2;
        }
        return static_cast<double>(millis) / static_cast<double>(kMillisPerDay);
    }

    // --- Comparison (C# operators and `CompareTo`, the latter driving the sort comparators). ---
    bool operator==(const DateTime& o) const { return ticks_ == o.ticks_; }
    bool operator!=(const DateTime& o) const { return ticks_ != o.ticks_; }
    bool operator<(const DateTime& o) const { return ticks_ < o.ticks_; }
    bool operator<=(const DateTime& o) const { return ticks_ <= o.ticks_; }
    bool operator>(const DateTime& o) const { return ticks_ > o.ticks_; }
    bool operator>=(const DateTime& o) const { return ticks_ >= o.ticks_; }
    int compare_to(const DateTime& o) const {
        return ticks_ < o.ticks_ ? -1 : (ticks_ > o.ticks_ ? 1 : 0);
    }

    // --- corehydro additions: the R/Python wire format and an ISO round trip. ---

    // Seconds since 1970-01-01 (R POSIXct in UTC; Python datetime64). Exact for every whole
    // second, which covers every TimeInterval the container supports.
    double to_unix_seconds() const {
        return static_cast<double>(ticks_ - kUnixEpochTicks) / static_cast<double>(kTicksPerSecond);
    }

    static DateTime from_unix_seconds(double seconds) {
        double whole = std::floor(seconds);
        double frac = seconds - whole;
        int64_t ticks = kUnixEpochTicks + static_cast<int64_t>(whole) * kTicksPerSecond +
                        static_cast<int64_t>(std::llround(frac * static_cast<double>(kTicksPerSecond)));
        if (ticks < 0 || ticks > kMaxTicks)
            throw std::out_of_range("The date is outside the representable range.");
        return DateTime(ticks);
    }

    // The BCL "o" round-trip format for an Unspecified date: yyyy-MM-ddTHH:mm:ss.fffffff.
    std::string to_iso_string() const {
        Civil c = civil();
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%07lld", c.y, c.m, c.d,
                      hour(), minute(), second(),
                      static_cast<long long>(ticks_ % kTicksPerSecond));
        return std::string(buf);
    }

    // Parses "yyyy-MM-dd", "yyyy-MM-ddTHH:mm:ss" and the "o" form with 1-7 fractional digits. A
    // trailing "Z" is accepted and ignored (these dates carry no zone; see the severance note).
    static DateTime parse_iso(const std::string& text) {
        std::string s = text;
        if (!s.empty() && (s.back() == 'Z' || s.back() == 'z')) s.pop_back();
        int y = 0, mo = 0, d = 0, h = 0, mi = 0, sec = 0;
        int consumed = 0;
        if (std::sscanf(s.c_str(), "%4d-%2d-%2d%n", &y, &mo, &d, &consumed) != 3)
            throw std::invalid_argument("The date '" + text + "' is not a valid ISO 8601 date.");
        std::size_t pos = static_cast<std::size_t>(consumed);
        int64_t sub_second_ticks = 0;
        if (pos < s.size()) {
            if (s[pos] != 'T' && s[pos] != ' ')
                throw std::invalid_argument("The date '" + text +
                                            "' is not a valid ISO 8601 date.");
            std::string rest = s.substr(pos + 1);
            int rconsumed = 0;
            if (std::sscanf(rest.c_str(), "%2d:%2d:%2d%n", &h, &mi, &sec, &rconsumed) != 3)
                throw std::invalid_argument("The date '" + text +
                                            "' is not a valid ISO 8601 date.");
            std::size_t rpos = static_cast<std::size_t>(rconsumed);
            if (rpos < rest.size()) {
                if (rest[rpos] != '.')
                    throw std::invalid_argument("The date '" + text +
                                                "' is not a valid ISO 8601 date.");
                std::string digits = rest.substr(rpos + 1);
                if (digits.empty() || digits.size() > 7)
                    throw std::invalid_argument("The date '" + text +
                                                "' is not a valid ISO 8601 date.");
                for (char ch : digits)
                    if (ch < '0' || ch > '9')
                        throw std::invalid_argument("The date '" + text +
                                                    "' is not a valid ISO 8601 date.");
                digits.append(7 - digits.size(), '0');
                sub_second_ticks = std::atoll(digits.c_str());
            }
        }
        DateTime out(y, mo, d, h, mi, sec);
        return DateTime(out.ticks() + sub_second_ticks);
    }

   private:
    struct Civil {
        int y;
        int m;
        int d;
    };

    // Days since 0001-01-01 for a calendar date, via the proleptic-Gregorian era algorithm (the
    // BCL's calendar is proleptic Gregorian from year 1, so this agrees with it everywhere).
    static int64_t day_number(int year, int month, int day) {
        int y = year;
        y -= month <= 2 ? 1 : 0;
        const int64_t era = (y >= 0 ? y : y - 399) / 400;
        const int64_t yoe = y - era * 400;
        const int64_t doy = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
        const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        // 719468 shifts to the 1970-01-01 epoch; 719162 shifts back to 0001-01-01.
        return era * 146097 + doe - 719468 + 719162;
    }

    Civil civil() const {
        int64_t z = ticks_ / kTicksPerDay + 719468 - 719162;
        const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
        const int64_t doe = z - era * 146097;
        const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        const int64_t y = yoe + era * 400;
        const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        const int64_t mp = (5 * doy + 2) / 153;
        const int64_t d = doy - (153 * mp + 2) / 5 + 1;
        const int64_t m = mp + (mp < 10 ? 3 : -9);
        return Civil{static_cast<int>(y + (m <= 2 ? 1 : 0)), static_cast<int>(m),
                     static_cast<int>(d)};
    }

    // The shared body of the five fractional adds (see notes 2, 3 and 4).
    DateTime add_units(double value, int64_t ticks_per_unit) const {
        if (std::isnan(value)) return *this;
        // The limit is computed by INTEGER division. Taking `(double)kMaxTicks / unit` instead
        // rounds up (kMaxTicks exceeds 2^53, so its double is 3155378976000000000), which lets
        // one more day through than the BCL does: measured, `AddDays(3652059)` raises "Value to
        // add was out of range" and `AddDays(3652058)` gets past this guard to fail the result
        // check below.
        double limit = static_cast<double>(kMaxTicks / ticks_per_unit);
        if (!(value >= -limit && value <= limit))
            throw std::out_of_range("Value to add was out of range.");
        double whole = std::trunc(value);
        double frac = value - whole;
        int64_t delta = static_cast<int64_t>(whole) * ticks_per_unit +
                        static_cast<int64_t>(frac * static_cast<double>(ticks_per_unit));
        return add_ticks(delta);
    }

    int64_t ticks_ = 0;
};

}  // namespace corehydro::numerics::data
