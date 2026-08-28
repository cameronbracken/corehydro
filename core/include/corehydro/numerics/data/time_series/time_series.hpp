// ported from: Numerics/Data/Time Series/TimeSeries.cs @ 2a0357a
//
// The Numerics `TimeSeries : Series<DateTime, double>` container.
//
// P2 shipped a THIN adapter over this file -- only the surface the ported ModelBase families
// consume, indexed by a plain integer offset because no consumer did calendar arithmetic. P6
// replaces that adapter with the real container and the real index type. The index swap was
// verified value-neutral on its own before any container method landed: the whole ctest suite,
// the fixture runner, testthat, pytest and the dotnet oracle gate all reproduced unchanged.
//
// Index representation. The C# ordinate index is `DateTime` and so is this one
// (`numerics/data/time_series/support/date_time.hpp`, a corehydro addition standing in for the
// BCL type). The adapter's `+1` integer walk is gone: the date-filling constructors advance by
// `add_time_interval`, which is real calendar arithmetic, exactly as C# does.
//
// Deliberately NOT ported (project-wide severances, each recorded here rather than silently
// dropped): the `XElement` constructor and `ToXElement` (XML persistence is severed repo-wide,
// so `clone()` deep-copies instead of round-tripping through XML -- see the note on `clone()`),
// `INotifyPropertyChanged` / `CollectionChanged` (inert in the `Series` base), and
// `Support/TimeSeriesDownload.cs` (network gauge retrieval; R and Python users have far better
// HTTP tooling natively).
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/interpolation/linear.hpp"
#include "corehydro/numerics/data/series_ordinate.hpp"
#include "corehydro/numerics/data/time_series/support/block_function_type.hpp"
#include "corehydro/numerics/data/time_series/support/date_time.hpp"
#include "corehydro/numerics/data/time_series/support/list_sort_direction.hpp"
#include "corehydro/numerics/data/time_series/support/series.hpp"
#include "corehydro/numerics/data/time_series/support/smoothing_function_type.hpp"
#include "corehydro/numerics/data/time_series/support/time_interval.hpp"
#include "corehydro/numerics/utilities/dotnet_sort.hpp"

namespace corehydro::numerics::data {

// A time-series: a collection of time-series ordinates. Mutable, mirroring the C# stateful
// binding API (the never-mutate rule is relaxed for these model/binding objects).
class TimeSeries : public Series<DateTime, double> {
   public:
    // The index type (C# `DateTime`).
    using IndexType = DateTime;
    using Ordinate = SeriesOrdinate<IndexType, double>;

    // Constructs an empty time-series (C# 32; the default interval is OneDay per the C#
    // `_timeInterval` field initializer).
    TimeSeries() = default;

    // Constructs an empty time-series with a specified time interval (C# 38-41).
    explicit TimeSeries(TimeInterval time_interval) : time_interval_(time_interval) {}

    // Constructs a time-series spanning [start_date, end_date] filled with NaN (C# 49-63).
    TimeSeries(TimeInterval time_interval, const DateTime& start_date, const DateTime& end_date)
        : TimeSeries(time_interval, start_date, end_date,
                     std::numeric_limits<double>::quiet_NaN()) {}

    // Constructs a time-series spanning [start_date, end_date] filled with a constant
    // (C# 72-86). The NaN-filling ctor above is the same body with a NaN fixed value, which is
    // exactly how C# writes the two.
    TimeSeries(TimeInterval time_interval, const DateTime& start_date, const DateTime& end_date,
               double fixed_value)
        : time_interval_(time_interval) {
        if (time_interval == TimeInterval::Irregular)
            throw std::invalid_argument(
                "The time interval cannot be irregular with this constructor.");
        if (start_date > end_date)
            throw std::invalid_argument("Start date must be less than or equal to end date.");
        add(Ordinate(start_date, fixed_value));
        while (series_ordinates_.back().index() < end_date)
            add(Ordinate(add_time_interval(series_ordinates_.back().index(), time_interval_),
                         fixed_value));
    }

    // Constructs a time-series from a start date and a list of data values (C# 94-106).
    TimeSeries(TimeInterval time_interval, const DateTime& start_date,
               const std::vector<double>& data)
        : time_interval_(time_interval) {
        if (time_interval == TimeInterval::Irregular)
            throw std::invalid_argument(
                "The time interval cannot be irregular with this constructor.");
        if (data.empty()) return;
        add(Ordinate(start_date, data[0]));
        for (std::size_t i = 1; i < data.size(); ++i)
            add(Ordinate(add_time_interval((*this)[static_cast<int>(i) - 1].index(),
                                           time_interval_),
                         data[i]));
    }

    // The base's ordinate-taking `add` would otherwise be hidden by the value-taking `add`
    // below (C# overloads `Add(SeriesOrdinate)` and `Add(double)` across the same boundary).
    using Series<DateTime, double>::add;

    // --- Properties. ---

    // The time interval (C# `TimeInterval` getter).
    TimeInterval time_interval() const { return time_interval_; }

    // Whether any value is missing (C# `HasMissingValues`).
    bool has_missing_values() const { return number_of_missing_values() > 0; }

    // The earliest / latest index (C# `StartDate` / `EndDate`: `Min`/`Max` over the ordinates,
    // or `default` when empty -- NOT the first and last ordinate, which differ on an unsorted
    // series).
    DateTime start_date() const {
        if (count() == 0) return DateTime();
        DateTime m = series_ordinates_.front().index();
        for (const auto& o : series_ordinates_)
            if (o.index() < m) m = o.index();
        return m;
    }

    DateTime end_date() const {
        if (count() == 0) return DateTime();
        DateTime m = series_ordinates_.front().index();
        for (const auto& o : series_ordinates_)
            if (o.index() > m) m = o.index();
        return m;
    }

    // --- Sorting (C# 176-202). ---

    // Sorts the ordinates by index (C# `SortByTime`). `List<T>.Sort(Comparison<T>)` is an
    // UNSTABLE introsort, and duplicate indexes are legal on an irregular series, so the tie
    // permutation is observable -- hence the ported .NET sort rather than `std::sort`.
    void sort_by_time(ListSortDirection order = ListSortDirection::Ascending) {
        if (order == ListSortDirection::Ascending) {
            utilities::dotnet_list_sort(series_ordinates_, [](const Ordinate& x, const Ordinate& y) {
                return x.index().compare_to(y.index());
            });
        } else {
            utilities::dotnet_list_sort(series_ordinates_, [](const Ordinate& x, const Ordinate& y) {
                return -1 * x.index().compare_to(y.index());
            });
        }
    }

    // Sorts the ordinates by value (C# `SortByValue`), through the same unstable introsort with
    // `double.CompareTo` semantics (NaN sorts below every number and equals itself).
    void sort_by_value(ListSortDirection order = ListSortDirection::Ascending) {
        if (order == ListSortDirection::Ascending) {
            utilities::dotnet_list_sort(series_ordinates_, [](const Ordinate& x, const Ordinate& y) {
                return utilities::compare_double(x.value(), y.value());
            });
        } else {
            utilities::dotnet_list_sort(series_ordinates_, [](const Ordinate& x, const Ordinate& y) {
                return -1 * utilities::compare_double(x.value(), y.value());
            });
        }
    }

    // --- The math region (C# 208-471). Every method mutates IN PLACE, and every one except
    // `log_transform` and `standardize` leaves a missing (NaN) value missing. The C# wraps each
    // body in SuppressCollectionChanged / RaiseCollectionChangedReset, which are inert here. ---

    // C# `Add(double)` / `Add(double, IList<int>)`.
    void add(double constant) {
        for (int i = 0; i <= count() - 1; ++i)
            if (!std::isnan((*this)[i].value()))
                (*this)[i].set_value((*this)[i].value() + constant);
    }
    void add(double constant, const std::vector<int>& indexes) {
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            int k = indexes[i];
            if (in_range(k) && !std::isnan((*this)[k].value()))
                (*this)[k].set_value((*this)[k].value() + constant);
        }
    }

    // C# `Subtract(double)` / `Subtract(double, IList<int>)`.
    void subtract(double constant) {
        for (int i = 0; i <= count() - 1; ++i)
            if (!std::isnan((*this)[i].value()))
                (*this)[i].set_value((*this)[i].value() - constant);
    }
    void subtract(double constant, const std::vector<int>& indexes) {
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            int k = indexes[i];
            if (in_range(k) && !std::isnan((*this)[k].value()))
                (*this)[k].set_value((*this)[k].value() - constant);
        }
    }

    // C# `Multiply(double)` / `Multiply(double, IList<int>)`.
    void multiply(double constant) {
        for (int i = 0; i <= count() - 1; ++i)
            if (!std::isnan((*this)[i].value()))
                (*this)[i].set_value((*this)[i].value() * constant);
    }
    void multiply(double constant, const std::vector<int>& indexes) {
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            int k = indexes[i];
            if (in_range(k) && !std::isnan((*this)[k].value()))
                (*this)[k].set_value((*this)[k].value() * constant);
        }
    }

    // C# `Divide(double)` / `Divide(double, IList<int>)`. Only the whole-series overload guards
    // against a zero divisor; the indexed one divides by zero and stores the infinity.
    void divide(double constant) {
        if (constant == 0) throw std::invalid_argument("Cannot divide by zero.");
        for (int i = 0; i <= count() - 1; ++i)
            if (!std::isnan((*this)[i].value()))
                (*this)[i].set_value((*this)[i].value() / constant);
    }
    void divide(double constant, const std::vector<int>& indexes) {
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            int k = indexes[i];
            if (in_range(k) && !std::isnan((*this)[k].value()))
                (*this)[k].set_value((*this)[k].value() / constant);
        }
    }

    // C# `AbsoluteValue()` / `AbsoluteValue(IList<int>)`.
    void absolute_value() {
        for (int i = 0; i <= count() - 1; ++i)
            if (!std::isnan((*this)[i].value()))
                (*this)[i].set_value(std::fabs((*this)[i].value()));
    }
    void absolute_value(const std::vector<int>& indexes) {
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            int k = indexes[i];
            if (in_range(k) && !std::isnan((*this)[k].value()))
                (*this)[k].set_value(std::fabs((*this)[k].value()));
        }
    }

    // C# `Exponentiate(double)` / `Exponentiate(double, IList<int>)`.
    void exponentiate(double power) {
        for (int i = 0; i <= count() - 1; ++i)
            if (!std::isnan((*this)[i].value()))
                (*this)[i].set_value(std::pow((*this)[i].value(), power));
    }
    void exponentiate(double power, const std::vector<int>& indexes) {
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            int k = indexes[i];
            if (in_range(k) && !std::isnan((*this)[k].value()))
                (*this)[k].set_value(std::pow((*this)[k].value(), power));
        }
    }

    // C# `LogTransform(double baseValue = 10)`. Unlike every other method in this region, a
    // non-positive or missing value is WRITTEN as NaN rather than left alone. C# calls
    // `Math.Log(x, baseValue)`, which is `Log(x) / Log(baseValue)` -- not `Log10` -- so the
    // division is transcribed rather than replaced with `std::log10`.
    void log_transform(double base_value = 10) {
        for (int i = 0; i < count(); ++i) {
            double v = (*this)[i].value();
            if (v > 0 && !std::isnan(v))
                (*this)[i].set_value(std::log(v) / std::log(base_value));
            else
                (*this)[i].set_value(std::numeric_limits<double>::quiet_NaN());
        }
    }

    // C# `LogTransform(IList<int>, double baseValue = 10)`. UPSTREAM ASYMMETRY, mirrored: the
    // `else` branch writes NaN through `this[indexes[i]]` WITHOUT re-checking the bounds its
    // `if` checked, so an out-of-range index raises from the indexer instead of being skipped
    // the way every sibling indexed overload skips it. C++ would read out of bounds, so the
    // bounds check is explicit here and throws at the same point C# does.
    void log_transform(const std::vector<int>& indexes, double base_value = 10) {
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            int k = indexes[i];
            if (in_range(k) && (*this)[k].value() > 0 && !std::isnan((*this)[k].value())) {
                (*this)[k].set_value(std::log((*this)[k].value()) / std::log(base_value));
            } else {
                throw_if_out_of_range(k);
                (*this)[k].set_value(std::numeric_limits<double>::quiet_NaN());
            }
        }
    }

    // C# `Standardize()`. Uses the container's own NaN-skipping mean and standard deviation, and
    // does NOT skip missing values when writing -- a NaN propagates through the subtraction.
    void standardize() {
        double mean = mean_value();
        double std_dev = standard_deviation();
        if (std_dev == 0)
            throw std::runtime_error(
                "Standard deviation is zero, standardization cannot be performed.");
        for (int i = 0; i < count(); ++i)
            (*this)[i].set_value(((*this)[i].value() - mean) / std_dev);
    }

    // C# `Inverse()` / `Inverse(IList<int>)`: 1/x, with zero and NaN both becoming NaN. The
    // indexed overload carries the same out-of-range asymmetry as `log_transform` above.
    void inverse() {
        for (int i = 0; i < count(); ++i) {
            double v = (*this)[i].value();
            if (v != 0 && !std::isnan(v))
                (*this)[i].set_value(1.0 / v);
            else if (v == 0 || std::isnan(v))
                (*this)[i].set_value(std::numeric_limits<double>::quiet_NaN());
        }
    }
    void inverse(const std::vector<int>& indexes) {
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            int k = indexes[i];
            if (in_range(k) && (*this)[k].value() != 0 && !std::isnan((*this)[k].value())) {
                (*this)[k].set_value(1.0 / (*this)[k].value());
            } else {
                throw_if_out_of_range(k);
                if ((*this)[k].value() == 0 || std::isnan((*this)[k].value()))
                    (*this)[k].set_value(std::numeric_limits<double>::quiet_NaN());
            }
        }
    }

    // Returns the cumulative sum, treating missing values as zero while accumulating (C# 476).
    // UPSTREAM ODDITY, mirrored: the result is `new TimeSeries()`, so it carries the DEFAULT
    // OneDay interval rather than this series' interval -- the only method in the class that
    // drops it.
    TimeSeries cumulative_sum() const {
        TimeSeries time_series;
        double sum = 0.0;
        for (int i = 0; i < count(); ++i) {
            if (!std::isnan((*this)[i].value())) sum += (*this)[i].value();
            time_series.add((*this)[i].clone());
            time_series.last().set_value(sum);
        }
        return time_series;
    }

    // --- Missing data (C# 521-704). ---

    // C# `NumberOfMissingValues`.
    int number_of_missing_values() const {
        int n = 0;
        for (const auto& o : series_ordinates_)
            if (std::isnan(o.value())) ++n;
        return n;
    }

    // C# `ReplaceMissingData(double)` / `ReplaceMissingData(IList<int>, double)`.
    void replace_missing_data(double value) {
        for (int i = 0; i < count(); ++i)
            if (std::isnan((*this)[i].value())) (*this)[i].set_value(value);
    }
    void replace_missing_data(const std::vector<int>& indexes, double value) {
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            int k = indexes[i];
            if (in_range(k) && std::isnan((*this)[k].value())) (*this)[k].set_value(value);
        }
    }

    // Interpolates runs of missing values shorter than the limit (C# 561). Sorts by time first,
    // then interpolates LINEARLY IN OA-DATE SPACE (`DateTime.ToOADate()`), so an irregular
    // spacing is honoured. The trailing `j == Count - 1` branch EXTRAPOLATES from the two
    // preceding ordinates instead, and is guarded by `i >= 2` here but not in the indexed
    // overload below.
    void interpolate_missing_data(int max_number_of_missing) {
        sort_by_time();
        for (int i = 1; i < count(); ++i) {
            if (std::isnan((*this)[i].value())) {
                double x = (*this)[i].index().to_oa_date();
                double x1 = (*this)[i - 1].index().to_oa_date();
                double y1 = (*this)[i - 1].value();
                int upper = i + max_number_of_missing;
                for (int j = i; j <= std::min(count() - 1, upper); ++j) {
                    if (!std::isnan((*this)[j].value())) {
                        double x2 = (*this)[j].index().to_oa_date();
                        double y2 = (*this)[j].value();
                        (*this)[i].set_value(y1 + (x - x1) / (x2 - x1) * (y2 - y1));
                        break;
                    }
                    if (j == count() - 1 && i >= 2) {
                        x1 = (*this)[i - 2].index().to_oa_date();
                        double x2 = (*this)[i - 1].index().to_oa_date();
                        y1 = (*this)[i - 2].value();
                        double y2 = (*this)[i - 1].value();
                        (*this)[i].set_value(y1 - (x1 - x) * (y2 - y1) / (x2 - x1));
                    }
                }
            }
        }
    }

    // C# `InterpolateMissingData(int, IList<int>)`. UPSTREAM ASYMMETRY, mirrored: the
    // extrapolation branch reads `this[idx - 2]` with NO `idx >= 2` guard (the non-indexed
    // overload has one), so an index of 1 reads position -1. C++ cannot read position -1, so the
    // guard here throws at the point C#'s indexer would.
    void interpolate_missing_data(int max_number_of_missing, const std::vector<int>& indexes) {
        sort_by_time();
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            int idx = indexes[i];
            if (idx >= 1 && idx < count() && std::isnan((*this)[idx].value())) {
                double x = (*this)[idx].index().to_oa_date();
                double x1 = (*this)[idx - 1].index().to_oa_date();
                double y1 = (*this)[idx - 1].value();
                int upper = idx + max_number_of_missing;
                for (int j = idx; j <= std::min(count() - 1, upper); ++j) {
                    if (!std::isnan((*this)[j].value())) {
                        double x2 = (*this)[j].index().to_oa_date();
                        double y2 = (*this)[j].value();
                        (*this)[idx].set_value(y1 + (x - x1) / (x2 - x1) * (y2 - y1));
                        break;
                    }
                    if (j == count() - 1) {
                        throw_if_out_of_range(idx - 2);
                        x1 = (*this)[idx - 2].index().to_oa_date();
                        double x2 = (*this)[idx - 1].index().to_oa_date();
                        y1 = (*this)[idx - 2].value();
                        double y2 = (*this)[idx - 1].value();
                        (*this)[idx].set_value(y1 - (x1 - x) * (y2 - y1) / (x2 - x1));
                    }
                }
            }
        }
    }

    // Returns a new series covering every date in [start_date, end_date] at this interval,
    // taking each value from the source when a date matches and `value` otherwise (C# 676).
    static TimeSeries fill_missing_dates(const TimeSeries& time_series, const DateTime& start_date,
                                         const DateTime& end_date, double value = 0.0) {
        if (time_series.time_interval() == TimeInterval::Irregular)
            throw std::runtime_error("This method does not work with irregular data.");

        // C# builds a Dictionary<DateTime,int> and only ever looks up in it, so an ordered map
        // gives the same answer. A duplicated date keeps the LAST ordinate, as the C# does.
        std::map<DateTime, int> lookup;
        for (int i = 0; i < time_series.count(); ++i) lookup[time_series[i].index()] = i;

        TimeSeries result(time_series.time_interval());
        for (DateTime date = start_date; date <= end_date;
             date = add_time_interval(date, time_series.time_interval())) {
            auto it = lookup.find(date);
            if (it != lookup.end())
                result.add(Ordinate(date, time_series[it->second].value()));
            else
                result.add(Ordinate(date, value));
        }
        return result;
    }

    // --- Static interval arithmetic (C# 711-868). ---

    // Returns the date advanced by one time interval (C# `AddTimeInterval`). An unrecognized
    // interval -- only `Irregular` reaches the fall-through -- returns the date unchanged.
    static DateTime add_time_interval(const DateTime& time, TimeInterval time_interval) {
        switch (time_interval) {
            case TimeInterval::OneMinute: return time.add_minutes(1.0);
            case TimeInterval::FiveMinute: return time.add_minutes(5.0);
            case TimeInterval::FifteenMinute: return time.add_minutes(15.0);
            case TimeInterval::ThirtyMinute: return time.add_minutes(30.0);
            case TimeInterval::OneHour: return time.add_hours(1.0);
            case TimeInterval::SixHour: return time.add_hours(6.0);
            case TimeInterval::TwelveHour: return time.add_hours(12.0);
            case TimeInterval::OneDay: return time.add_days(1.0);
            case TimeInterval::SevenDay: return time.add_days(7.0);
            case TimeInterval::OneMonth: return time.add_months(1);
            case TimeInterval::OneQuarter: return time.add_months(3);
            case TimeInterval::OneYear: return time.add_years(1);
            default: return time;
        }
    }

    // Returns the date moved back by one time interval (C# `SubtractTimeInterval`).
    static DateTime subtract_time_interval(const DateTime& time, TimeInterval time_interval) {
        switch (time_interval) {
            case TimeInterval::OneMinute: return time.add_minutes(-1.0);
            case TimeInterval::FiveMinute: return time.add_minutes(-5.0);
            case TimeInterval::FifteenMinute: return time.add_minutes(-15.0);
            case TimeInterval::ThirtyMinute: return time.add_minutes(-30.0);
            case TimeInterval::OneHour: return time.add_hours(-1.0);
            case TimeInterval::SixHour: return time.add_hours(-6.0);
            case TimeInterval::TwelveHour: return time.add_hours(-12.0);
            case TimeInterval::OneDay: return time.add_days(-1.0);
            case TimeInterval::SevenDay: return time.add_days(-7.0);
            case TimeInterval::OneMonth: return time.add_months(-1);
            case TimeInterval::OneQuarter: return time.add_months(-3);
            case TimeInterval::OneYear: return time.add_years(-1);
            default: return time;
        }
    }

    // The interval in hours (C# `TimeIntervalInHours`). Month, quarter, year and irregular
    // intervals have no fixed hour count and return NaN, as C# does.
    static double time_interval_in_hours(TimeInterval time_interval) {
        switch (time_interval) {
            case TimeInterval::OneMinute: return 1.0 / 60.0;
            case TimeInterval::FiveMinute: return 5.0 / 60.0;
            case TimeInterval::FifteenMinute: return 15.0 / 60.0;
            case TimeInterval::ThirtyMinute: return 30.0 / 60.0;
            case TimeInterval::OneHour: return 1.0;
            case TimeInterval::SixHour: return 6.0;
            case TimeInterval::TwelveHour: return 12.0;
            case TimeInterval::OneDay: return 24.0;
            case TimeInterval::SevenDay: return 24.0 * 7.0;
            default: return std::numeric_limits<double>::quiet_NaN();
        }
    }

    // --- Moving windows (C# 953-1028). ---

    // Returns the trailing moving average over `period` ordinates (C# 953). The running sum and
    // NaN count are maintained incrementally -- a value that leaves the window is subtracted back
    // out rather than the window being re-summed -- so the accumulation ORDER is oracle-visible.
    // `min_valid_count` defaults to `period` (strict: any missing value in a window gives NaN);
    // a smaller value averages over the observed entries only, matching pandas `min_periods`.
    // NOT const: C# calls `SortByTime()` on the RECEIVER, so a call leaves this series sorted.
    TimeSeries moving_average(int period, int min_valid_count = -1) {
        // C# writes `new ArgumentException(nameof(period), "The period must be less ...")`, which
        // passes the parameter NAME as the message and the message as the parameter name. The
        // intent is transcribed here; the argument swap is cosmetic and C#-specific.
        if (period >= count())
            throw std::invalid_argument(
                "The period must be less than the length of the time-series.");
        int min_count = min_valid_count < 0 ? period : min_valid_count;
        if (min_count < 1 || min_count > period)
            throw std::out_of_range("minValidCount must be between 1 and period.");
        sort_by_time();
        TimeSeries time_series(time_interval_);
        double sum = 0.0;
        int nan_count = 0;
        for (int i = 1; i <= count(); ++i) {
            double new_val = (*this)[i - 1].value();
            if (std::isnan(new_val))
                nan_count++;
            else
                sum += new_val;

            if (i > period) {
                double old_val = (*this)[i - period - 1].value();
                if (std::isnan(old_val))
                    nan_count--;
                else
                    sum -= old_val;
            }

            if (i >= period) {
                int valid_count = period - nan_count;
                double avg = valid_count >= min_count ? sum / valid_count
                                                      : std::numeric_limits<double>::quiet_NaN();
                time_series.add(Ordinate((*this)[i - 1].index(), avg));
            }
        }
        return time_series;
    }

    // Returns the trailing moving sum over `period` ordinates (C# 996), the same incremental
    // window as `moving_average` without the division. Under a relaxed `min_valid_count` the sum
    // is over the OBSERVED entries only -- no rescaling is applied. NOT const, for the same
    // receiver-sorting reason as `moving_average`.
    TimeSeries moving_sum(int period, int min_valid_count = -1) {
        if (period >= count())
            throw std::invalid_argument(
                "The period must be less than the length of the time-series.");
        int min_count = min_valid_count < 0 ? period : min_valid_count;
        if (min_count < 1 || min_count > period)
            throw std::out_of_range("minValidCount must be between 1 and period.");
        sort_by_time();
        TimeSeries time_series(time_interval_);
        double sum = 0.0;
        int nan_count = 0;
        for (int i = 1; i <= count(); ++i) {
            double new_val = (*this)[i - 1].value();
            if (std::isnan(new_val))
                nan_count++;
            else
                sum += new_val;

            if (i > period) {
                double old_val = (*this)[i - period - 1].value();
                if (std::isnan(old_val))
                    nan_count--;
                else
                    sum -= old_val;
            }

            if (i >= period) {
                int valid_count = period - nan_count;
                double output =
                    valid_count >= min_count ? sum : std::numeric_limits<double>::quiet_NaN();
                time_series.add(Ordinate((*this)[i - 1].index(), output));
            }
        }
        return time_series;
    }

    // --- Date shifting and clipping (C# 1129-1225). ---

    // Re-anchors the series at a new start date (C# 1129). On a REGULAR interval every later
    // ordinate is re-walked from the new start, so an irregularly spaced series stored under a
    // regular interval comes out evenly spaced; on `Irregular` only the first ordinate moves.
    void shift_all_dates(const DateTime& new_start_date) {
        if (count() == 0) return;
        (*this)[0].set_index(new_start_date);
        if (time_interval_ != TimeInterval::Irregular) {
            for (int i = 1; i < count(); ++i)
                (*this)[i].set_index(add_time_interval((*this)[i - 1].index(), time_interval_));
        }
    }

    // C# `ShiftDatesByDay` / `ShiftDatesByMonth` / `ShiftDatesByYear`: a NEW series with every
    // date moved, values unchanged. Each sorts the SOURCE by time first (a mutation of the
    // receiver in C#, mirrored here).
    TimeSeries shift_dates_by_day(int number_of_days) {
        sort_by_time();
        TimeSeries time_series(time_interval_);
        for (int i = 0; i < count(); ++i)
            time_series.add(Ordinate((*this)[i].index().add_days(number_of_days),
                                     (*this)[i].value()));
        return time_series;
    }

    TimeSeries shift_dates_by_month(int number_of_months) {
        sort_by_time();
        TimeSeries time_series(time_interval_);
        for (int i = 0; i < count(); ++i)
            time_series.add(Ordinate((*this)[i].index().add_months(number_of_months),
                                     (*this)[i].value()));
        return time_series;
    }

    TimeSeries shift_dates_by_year(int number_of_years) {
        sort_by_time();
        TimeSeries time_series(time_interval_);
        for (int i = 0; i < count(); ++i)
            time_series.add(Ordinate((*this)[i].index().add_years(number_of_years),
                                     (*this)[i].value()));
        return time_series;
    }

    // Returns the ordinates inside [start_date, end_date] (C# 1210). Both bounds must lie inside
    // the series' own span; C# raises ArgumentOutOfRangeException otherwise.
    TimeSeries clip_time_series(const DateTime& start_date_in, const DateTime& end_date_in) const {
        if (start_date_in < start_date())
            throw std::out_of_range(
                "The start date is earlier than the start date of the time-series.");
        if (end_date_in > end_date())
            throw std::out_of_range("The end date is later than the end date of the time-series.");
        TimeSeries time_series(time_interval_);
        for (int i = 0; i < count(); ++i)
            if ((*this)[i].index() >= start_date_in && (*this)[i].index() <= end_date_in)
                time_series.add((*this)[i].clone());
        return time_series;
    }

    // Converts the series to another time interval (C# 1233), returning nothing when neither
    // interval has a defined hour count (both `TimeIntervalInHours` values are NaN, so every
    // comparison below is false and the C# falls out returning null).
    //
    // Four branches, transcribed as written:
    //   - equal step sizes: a clone.
    //   - a SMALLER step, averaging: linear interpolation of the value at each new time.
    //   - a LARGER step, averaging: the mean over each block of `blockDuration` source ordinates.
    //   - a SMALLER step, cumulating: the containing block's value divided by the step ratio,
    //     located with `Linear::search` (the interpolater is built but only searched).
    //   - a LARGER step, cumulating: the sum over each block.
    // The loop bound `N` is a DOUBLE computed from the total hours spanned, not an integer count,
    // and the block branches divide by `min(t + blockDuration, Count) - t`, which is zero or
    // negative once `t` reaches `Count` -- both preserved rather than tidied.
    std::optional<TimeSeries> convert_time_interval(TimeInterval time_interval,
                                                    bool average = true) const {
        double ts = time_interval_in_hours(time_interval_);
        double new_ts = time_interval_in_hours(time_interval);
        int block_duration = static_cast<int>(std::floor(new_ts / ts));
        double n = end_date().subtract_total_hours(start_date()) / new_ts + 1;

        if (new_ts == ts) {
            return clone();
        } else if (new_ts < ts && average == true) {
            double t = 0, value = 0;
            std::vector<double> x(static_cast<std::size_t>(count()));
            std::vector<double> y(static_cast<std::size_t>(count()));
            for (int i = 0; i < count(); ++i) {
                x[static_cast<std::size_t>(i)] = t;
                y[static_cast<std::size_t>(i)] = (*this)[i].value();
                t += ts;
            }
            Linear lin_int(x, y);

            t = 0;
            TimeSeries time_series(time_interval);
            time_series.add(Ordinate(start_date(), (*this)[0].value()));
            for (int i = 1; i < n; ++i) {
                t += new_ts;
                value = lin_int.interpolate(t);
                time_series.add(
                    Ordinate(add_time_interval(time_series[i - 1].index(), time_interval), value));
            }
            return time_series;
        } else if (new_ts > ts && average == true) {
            int t = 0;
            TimeSeries time_series(time_interval);
            for (int i = 0; i < n; ++i) {
                double avg = 0;
                for (int j = t; j < std::min(t + block_duration, count()); ++j)
                    avg += (*this)[j].value();
                avg /= std::min(t + block_duration, count()) - t;
                t += block_duration;
                time_series.add(Ordinate(
                    i == 0 ? start_date()
                           : add_time_interval(time_series[i - 1].index(), time_interval),
                    avg));
            }
            return time_series;
        } else if (new_ts < ts && average == false) {
            double t = 0, value = 0, rate = ts / new_ts;
            std::vector<double> x(static_cast<std::size_t>(count()));
            std::vector<double> y(static_cast<std::size_t>(count()));
            for (int i = 0; i < count(); ++i) {
                x[static_cast<std::size_t>(i)] = t;
                y[static_cast<std::size_t>(i)] = (*this)[i].value();
                t += ts;
            }
            Linear lin_int(x, y);

            t = 0;
            TimeSeries time_series(time_interval);
            time_series.add(Ordinate(start_date(), y[0] / rate));
            for (int i = 1; i < n; ++i) {
                t += new_ts;
                int idx = lin_int.search(t);
                value = y[static_cast<std::size_t>(idx)] / rate;
                time_series.add(
                    Ordinate(add_time_interval(time_series[i - 1].index(), time_interval), value));
            }
            return time_series;
        } else if (new_ts > ts && average == false) {
            int t = 0;
            TimeSeries time_series(time_interval);
            for (int i = 0; i < n; ++i) {
                double sum = 0;
                for (int j = t; j < std::min(t + block_duration, count()); ++j)
                    sum += (*this)[j].value();
                t += block_duration;
                time_series.add(Ordinate(
                    i == 0 ? start_date()
                           : add_time_interval(time_series[i - 1].index(), time_interval),
                    sum));
            }
            return time_series;
        }

        return std::nullopt;
    }

    // --- Summary statistics (C# 1338-1417). ---

    // Min value, skipping NaN (C# 1338).
    double min_value() const {
        double min = std::numeric_limits<double>::max();
        for (int i = 0; i < count(); ++i) {
            double v = (*this)[i].value();
            if (!std::isnan(v) && v < min) min = v;
        }
        return min;
    }

    // Mean of the values, skipping NaN (C# 1370).
    double mean_value() const {
        if (count() == 0) return std::numeric_limits<double>::quiet_NaN();
        double mean = 0.0;
        int n = 0;
        for (int i = 0; i < count(); ++i) {
            double v = (*this)[i].value();
            if (!std::isnan(v)) {
                mean += v;
                n += 1;
            }
        }
        return mean / n;
    }

    // Sample standard deviation via the C# incremental algorithm, skipping NaN (C# 1389).
    double standard_deviation() const {
        if (count() < 2) return std::numeric_limits<double>::quiet_NaN();
        double variance = 0.0;
        double t = 0.0;
        int start_idx = 0;
        for (int i = 0; i < count(); ++i) {
            double v = (*this)[i].value();
            if (!std::isnan(v)) {
                t = v;
                start_idx = i + 1;
                break;
            }
        }
        double n = 1;
        for (int i = start_idx; i < count(); ++i) {
            double v = (*this)[i].value();
            if (!std::isnan(v)) {
                t += v;
                double diff = (i + 1) * v - t;
                variance += diff * diff / ((i + 1.0) * i);
                n += 1;
            }
        }
        return std::sqrt(variance / (n - 1));
    }

    // Returns a time-series of the successive differences (C# 496-515). Copies the value array,
    // repeats `differences` passes each computing temp[i] = result[i+lag] - result[i] over
    // result.size()-lag, throwing when result.size() <= lag; the result keeps the ORIGINAL start
    // date rather than shifting forward by the lag, because C# rebuilds from `StartDate`.
    TimeSeries difference(int lag = 1, int differences = 1) const {
        std::vector<double> result = values_to_array();
        for (int d = 0; d < differences; ++d) {
            if (static_cast<int>(result.size()) <= lag)
                throw std::invalid_argument("The length of the array must be greater than the lag.");
            std::vector<double> temp(result.size() - static_cast<std::size_t>(lag));
            for (std::size_t i = 0; i < result.size() - static_cast<std::size_t>(lag); ++i)
                temp[i] = result[i + static_cast<std::size_t>(lag)] - result[i];
            result = temp;
        }
        return TimeSeries(time_interval_, start_date(), result);
    }

    // Creates a deep copy (C# 2329 `Clone()`, which round-trips through `ToXElement()`). XML is
    // severed, and the round trip is value-preserving anyway: C# writes the index with the "o"
    // format and the value with "G17", both round-trip formats, and `NaN` survives
    // `double.TryParse(..., NumberStyles.Any, ...)`. So a direct deep copy is exactly equivalent,
    // and `core/tests/test_time_series_container.cpp` pins that equivalence bit-for-bit.
    TimeSeries clone() const {
        TimeSeries out(time_interval_);
        out.series_ordinates_ = series_ordinates_;
        return out;
    }

   protected:
    // C#'s indexed math overloads all guard with `indexes[i] >= 0 && indexes[i] < Count`.
    bool in_range(int index) const { return index >= 0 && index < count(); }

    // Two indexed overloads (`log_transform` and `inverse`, and the indexed
    // `interpolate_missing_data`'s extrapolation branch) reach through an index their `if` did
    // NOT bounds-check, which raises from the C# indexer. C++ would read out of bounds, so those
    // sites call this to throw at the same point.
    void throw_if_out_of_range(int index) const {
        if (!in_range(index))
            throw std::out_of_range(
                "Index was out of range. Must be non-negative and less than the size of the "
                "collection.");
    }

    // Whether `end_time` is more than `min_steps_between_events` intervals after `start_time`
    // (C# private `CheckIfMinStepsExceeded`, 876). Reads THIS series' interval, and returns false
    // for an unrecognized (irregular) one, which is what keeps an irregular POT cluster growing
    // until a value drops below the threshold.
    bool check_if_min_steps_exceeded(const DateTime& start_time, const DateTime& end_time,
                                     int min_steps_between_events) const {
        switch (time_interval_) {
            case TimeInterval::OneMinute:
                return end_time > start_time.add_minutes(1.0 * min_steps_between_events);
            case TimeInterval::FiveMinute:
                return end_time > start_time.add_minutes(5.0 * min_steps_between_events);
            case TimeInterval::FifteenMinute:
                return end_time > start_time.add_minutes(15.0 * min_steps_between_events);
            case TimeInterval::ThirtyMinute:
                return end_time > start_time.add_minutes(30.0 * min_steps_between_events);
            case TimeInterval::OneHour:
                return end_time > start_time.add_hours(1.0 * min_steps_between_events);
            case TimeInterval::SixHour:
                return end_time > start_time.add_hours(6.0 * min_steps_between_events);
            case TimeInterval::TwelveHour:
                return end_time > start_time.add_hours(12.0 * min_steps_between_events);
            case TimeInterval::OneDay:
                return end_time > start_time.add_days(min_steps_between_events);
            case TimeInterval::SevenDay:
                return end_time > start_time.add_days(7.0 * min_steps_between_events);
            case TimeInterval::OneMonth:
                return end_time > start_time.add_months(1 * min_steps_between_events);
            case TimeInterval::OneQuarter:
                return end_time > start_time.add_months(3 * min_steps_between_events);
            case TimeInterval::OneYear:
                return end_time > start_time.add_years(1 * min_steps_between_events);
            default: return false;
        }
    }

    TimeInterval time_interval_ = TimeInterval::OneDay;
};

}  // namespace corehydro::numerics::data
