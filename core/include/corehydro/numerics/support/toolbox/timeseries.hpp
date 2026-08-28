// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `timeseries` group's dispatch arms over the ported Numerics `TimeSeries` container
// (numerics/data/time_series/time_series.hpp, P6). This is the nineteenth group and the last
// major Numerics surface to reach R and Python.
//
// DATA LAYOUT. Unlike the `regression`/`ml` groups' matrix layout, a time series is two aligned
// series, so this group takes them that way:
//   data[0] = the ordinate DATES, as seconds since 1970-01-01 (see below)
//   data[1] = the ordinate VALUES
//   data[2] = optional, method-dependent: a second date pair (`clip`, `fill_missing_dates`), an
//             ordinate-index list (the indexed `math` verbs), or a probability list
//             (`percentiles`, `monthly_percentiles`)
// Every scalar, enum name and flag travels in `options_json`, including `time_interval`.
//
// DATES ARE EPOCH SECONDS ON THE WIRE, never .NET ticks. A tick count near 6.3e17 is not exactly
// representable in a double (53 bits of mantissa); epoch seconds for any date in the supported
// range is ~1.7e9, and every `TimeInterval` this container supports lands on a whole second. R's
// POSIXct IS epoch seconds, and numpy's datetime64[s] converts exactly, so the host layers convert
// at the boundary and nothing in between rounds.
//
// SERIES RESULTS ARE TWO-COLUMN MATRICES: a method that returns a series returns
// `dims = {n, 2}` with column 0 the dates (epoch seconds) and column 1 the values, the same
// row-major convention `paired_data`'s curve results use. A method that returns a named set uses
// `names`; one that returns a table uses `dims`.
//
// PURE VERBS OVER MUTATING METHODS. The container's math region mutates in place, because it
// mirrors a stateful C# binding object. Neither host language wants that across the boundary, so
// every arm here COPIES the built series, applies the C# mutator, and returns the result. The
// computation is the C# computation; only the ownership differs.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

namespace ts_data = corehydro::numerics::data;

// Accepts both the C# enum spelling (`"OneDay"`, which the fixtures and the oracle emitter use)
// and the snake-case spelling the R/Python verbs take (`"one_day"`), by normalizing case and
// dropping underscores. One token table, two spellings, no drift.
inline ts_data::TimeInterval parse_time_interval_token(const std::string& s) {
    std::string k;
    for (char c : s)
        if (c != '_' && c != '-')
            k.push_back(static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
    if (k == "oneminute") return ts_data::TimeInterval::OneMinute;
    if (k == "fiveminute") return ts_data::TimeInterval::FiveMinute;
    if (k == "fifteenminute") return ts_data::TimeInterval::FifteenMinute;
    if (k == "thirtyminute") return ts_data::TimeInterval::ThirtyMinute;
    if (k == "onehour") return ts_data::TimeInterval::OneHour;
    if (k == "sixhour") return ts_data::TimeInterval::SixHour;
    if (k == "twelvehour") return ts_data::TimeInterval::TwelveHour;
    if (k == "oneday") return ts_data::TimeInterval::OneDay;
    if (k == "sevenday") return ts_data::TimeInterval::SevenDay;
    if (k == "onemonth") return ts_data::TimeInterval::OneMonth;
    if (k == "onequarter") return ts_data::TimeInterval::OneQuarter;
    if (k == "oneyear") return ts_data::TimeInterval::OneYear;
    if (k == "irregular") return ts_data::TimeInterval::Irregular;
    throw std::runtime_error("unknown time interval '" + s +
                             "'; expected one_minute, five_minute, fifteen_minute, "
                             "thirty_minute, one_hour, six_hour, twelve_hour, one_day, "
                             "seven_day, one_month, one_quarter, one_year, or irregular");
}

inline ts_data::BlockFunctionType parse_block_function_token(const std::string& s) {
    if (s == "minimum" || s == "Minimum") return ts_data::BlockFunctionType::Minimum;
    if (s == "maximum" || s == "Maximum") return ts_data::BlockFunctionType::Maximum;
    if (s == "average" || s == "Average") return ts_data::BlockFunctionType::Average;
    if (s == "sum" || s == "Sum") return ts_data::BlockFunctionType::Sum;
    throw std::runtime_error("unknown block function '" + s +
                             "'; expected minimum, maximum, average, or sum");
}

inline ts_data::SmoothingFunctionType parse_smoothing_function_token(const std::string& s) {
    if (s == "none" || s == "None") return ts_data::SmoothingFunctionType::None;
    if (s == "difference" || s == "Difference") return ts_data::SmoothingFunctionType::Difference;
    if (s == "moving_average" || s == "MovingAverage")
        return ts_data::SmoothingFunctionType::MovingAverage;
    if (s == "moving_sum" || s == "MovingSum") return ts_data::SmoothingFunctionType::MovingSum;
    throw std::runtime_error("unknown smoothing function '" + s +
                             "'; expected none, difference, moving_average, or moving_sum");
}

// Builds the series a call operates on from data[0] (dates as epoch seconds) and data[1]
// (values), plus the `time_interval` option (default `one_day`, the C# field default).
inline ts_data::TimeSeries timeseries_build(const std::vector<std::vector<double>>& data,
                                            const JsonValue& options,
                                            const std::string& method) {
    const std::vector<double>& dates = data_at(data, 0, "timeseries", method);
    const std::vector<double>& values = data_at(data, 1, "timeseries", method);
    if (dates.size() != values.size())
        throw std::runtime_error("toolbox method 'timeseries." + method +
                                 "' needs date and value vectors of the same length; got " +
                                 std::to_string(dates.size()) + " and " +
                                 std::to_string(values.size()));
    ts_data::TimeInterval interval =
        parse_time_interval_token(options.value_or("time_interval", "one_day"));
    ts_data::TimeSeries ts(interval);
    for (std::size_t i = 0; i < dates.size(); ++i)
        ts.add(ts_data::TimeSeries::Ordinate(ts_data::DateTime::from_unix_seconds(dates[i]),
                                             values[i]));
    // TRANSPORT CONVENIENCE, not a container feature: JSON has no NaN literal and the data
    // vectors are plain doubles, so a fixture (or a host caller that would rather not build a
    // NaN) marks missing ordinates by INDEX here. The R and Python verbs pass their own NaNs
    // through `data[1]` directly and never set this.
    if (options.contains("missing_indexes")) {
        for (double idx : options.at("missing_indexes").as_double_vector()) {
            int k = static_cast<int>(idx);
            if (k >= 0 && k < ts.count())
                ts[k].set_value(std::numeric_limits<double>::quiet_NaN());
        }
    }
    return ts;
}

// Flattens a series into a ToolboxResult with dims = {n, 2}: dates (epoch seconds), then values.
inline ToolboxResult timeseries_result(const ts_data::TimeSeries& ts) {
    ToolboxResult r;
    r.dims = {ts.count(), 2};
    r.values.reserve(static_cast<std::size_t>(ts.count()) * 2);
    for (int i = 0; i < ts.count(); ++i) {
        r.values.push_back(ts[i].index().to_unix_seconds());
        r.values.push_back(ts[i].value());
    }
    return r;
}

// Flattens a row-major table into a ToolboxResult with dims = {rows, cols}.
inline ToolboxResult timeseries_table(const std::vector<std::vector<double>>& rows) {
    ToolboxResult r;
    int ncol = rows.empty() ? 0 : static_cast<int>(rows.front().size());
    r.dims = {static_cast<int>(rows.size()), ncol};
    for (const auto& row : rows)
        for (double v : row) r.values.push_back(v);
    return r;
}

// Flattens an ordered key/value list into values + names.
inline ToolboxResult timeseries_named(
    const std::vector<std::pair<std::string, double>>& pairs) {
    ToolboxResult r;
    for (const auto& kv : pairs) {
        r.names.push_back(kv.first);
        r.values.push_back(kv.second);
    }
    return r;
}

// The ordinate-index list the indexed math verbs take, from data[2].
inline std::vector<int> timeseries_indexes(const std::vector<std::vector<double>>& data,
                                           const std::string& method) {
    const std::vector<double>& raw = data_at(data, 2, "timeseries", method);
    std::vector<int> out;
    out.reserve(raw.size());
    for (double v : raw) out.push_back(static_cast<int>(v));
    return out;
}

inline ToolboxResult run_timeseries(const std::string& method,
                                    const std::vector<std::vector<double>>& data,
                                    const JsonValue& options) {
    // --- Date arithmetic (the DateTime value type itself). These take no series: `data[0]` is a
    // list of dates and every other input is an option. They exist so the oracle gate replays the
    // ported .NET calendar semantics against the real System.DateTime, not just the container. ---
    if (method == "date_add" || method == "date_subtract") {
        const std::vector<double>& dates = data_at(data, 0, "timeseries", method);
        ts_data::TimeInterval interval =
            parse_time_interval_token(options.value_or("time_interval", "one_day"));
        int steps = static_cast<int>(options.value_or("steps", 1.0));
        ToolboxResult r;
        for (double d : dates) {
            ts_data::DateTime t = ts_data::DateTime::from_unix_seconds(d);
            for (int s = 0; s < steps; ++s)
                t = method == "date_add"
                        ? ts_data::TimeSeries::add_time_interval(t, interval)
                        : ts_data::TimeSeries::subtract_time_interval(t, interval);
            r.values.push_back(t.to_unix_seconds());
        }
        return r;
    }

    if (method == "date_components") {
        const std::vector<double>& dates = data_at(data, 0, "timeseries", method);
        ToolboxResult r;
        r.dims = {static_cast<int>(dates.size()), 9};
        r.names = {"year",  "month",       "day",         "hour",   "minute",
                   "second", "day_of_year", "day_of_week", "oa_date"};
        for (double d : dates) {
            ts_data::DateTime t = ts_data::DateTime::from_unix_seconds(d);
            r.values.push_back(t.year());
            r.values.push_back(t.month());
            r.values.push_back(t.day());
            r.values.push_back(t.hour());
            r.values.push_back(t.minute());
            r.values.push_back(t.second());
            r.values.push_back(t.day_of_year());
            r.values.push_back(t.day_of_week());
            r.values.push_back(t.to_oa_date());
        }
        return r;
    }

    if (method == "date_add_months" || method == "date_add_years" || method == "date_add_days") {
        const std::vector<double>& dates = data_at(data, 0, "timeseries", method);
        double amount = options.value_or("amount", 1.0);
        ToolboxResult r;
        for (double d : dates) {
            ts_data::DateTime t = ts_data::DateTime::from_unix_seconds(d);
            if (method == "date_add_months")
                t = t.add_months(static_cast<int>(amount));
            else if (method == "date_add_years")
                t = t.add_years(static_cast<int>(amount));
            else
                t = t.add_days(amount);
            r.values.push_back(t.to_unix_seconds());
        }
        return r;
    }

    if (method == "interval_in_hours") {
        return scalar(ts_data::TimeSeries::time_interval_in_hours(
            parse_time_interval_token(options.value_or("time_interval", "one_day"))));
    }

    // --- Everything below operates on a built series. ---
    ts_data::TimeSeries ts = timeseries_build(data, options, method);

    if (method == "moving_average" || method == "moving_sum") {
        int period = static_cast<int>(options.value_or("period", 2.0));
        int min_valid_count = static_cast<int>(options.value_or("min_valid_count", -1.0));
        return timeseries_result(method == "moving_average"
                                     ? ts.moving_average(period, min_valid_count)
                                     : ts.moving_sum(period, min_valid_count));
    }
    if (method == "cumulative_sum") return timeseries_result(ts.cumulative_sum());
    if (method == "difference") {
        int lag = static_cast<int>(options.value_or("lag", 1.0));
        int differences = static_cast<int>(options.value_or("differences", 1.0));
        return timeseries_result(ts.difference(lag, differences));
    }
    if (method == "standardize") {
        ts.standardize();
        return timeseries_result(ts);
    }
    if (method == "sort") {
        bool by_value = options.value_or("by", "time") == std::string("value");
        ts_data::ListSortDirection order =
            options.value_or("order", "ascending") == std::string("descending")
                ? ts_data::ListSortDirection::Descending
                : ts_data::ListSortDirection::Ascending;
        if (by_value)
            ts.sort_by_value(order);
        else
            ts.sort_by_time(order);
        return timeseries_result(ts);
    }

    // The whole in-place math region behind one method name, selected by `function` (the
    // `MathFunctionType` member names). `indexes` in data[2] selects the indexed overload.
    if (method == "math") {
        std::string function = options.value_or("function", "add");
        bool indexed = data.size() > 2;
        std::vector<int> idx = indexed ? timeseries_indexes(data, method) : std::vector<int>{};
        double constant = options.value_or("constant", 0.0);
        if (function == "add" || function == "Add") {
            indexed ? ts.add(constant, idx) : ts.add(constant);
        } else if (function == "subtract" || function == "Subtract") {
            indexed ? ts.subtract(constant, idx) : ts.subtract(constant);
        } else if (function == "multiply" || function == "Multiply") {
            indexed ? ts.multiply(constant, idx) : ts.multiply(constant);
        } else if (function == "divide" || function == "Divide") {
            indexed ? ts.divide(constant, idx) : ts.divide(constant);
        } else if (function == "absolute_value" || function == "AbsoluteValue") {
            indexed ? ts.absolute_value(idx) : ts.absolute_value();
        } else if (function == "exponentiate" || function == "Exponentiate") {
            double power = options.value_or("power", 1.0);
            indexed ? ts.exponentiate(power, idx) : ts.exponentiate(power);
        } else if (function == "logarithm" || function == "Logarithm") {
            double base = options.value_or("base", 10.0);
            indexed ? ts.log_transform(idx, base) : ts.log_transform(base);
        } else if (function == "inverse" || function == "Inverse") {
            indexed ? ts.inverse(idx) : ts.inverse();
        } else if (function == "replace" || function == "Replace") {
            double value = options.value_or("value", 0.0);
            indexed ? ts.replace_missing_data(idx, value) : ts.replace_missing_data(value);
        } else if (function == "interpolate" || function == "Interpolate") {
            int max_missing = static_cast<int>(options.value_or("max_missing", 1.0));
            indexed ? ts.interpolate_missing_data(max_missing, idx)
                    : ts.interpolate_missing_data(max_missing);
        } else {
            throw std::runtime_error("unknown time-series math function '" + function +
                                     "'; expected add, subtract, multiply, divide, "
                                     "absolute_value, exponentiate, logarithm, inverse, replace, "
                                     "or interpolate");
        }
        return timeseries_result(ts);
    }

    if (method == "fill_missing_dates") {
        const std::vector<double>& range = data_at(data, 2, "timeseries", method);
        if (range.size() != 2)
            throw std::runtime_error(
                "toolbox method 'timeseries.fill_missing_dates' needs a {start, end} date pair");
        double value = options.value_or("value", 0.0);
        return timeseries_result(ts_data::TimeSeries::fill_missing_dates(
            ts, ts_data::DateTime::from_unix_seconds(range[0]),
            ts_data::DateTime::from_unix_seconds(range[1]), value));
    }
    if (method == "clip") {
        const std::vector<double>& range = data_at(data, 2, "timeseries", method);
        if (range.size() != 2)
            throw std::runtime_error(
                "toolbox method 'timeseries.clip' needs a {start, end} date pair");
        return timeseries_result(
            ts.clip_time_series(ts_data::DateTime::from_unix_seconds(range[0]),
                                ts_data::DateTime::from_unix_seconds(range[1])));
    }
    if (method == "shift") {
        std::string by = options.value_or("by", "day");
        int amount = static_cast<int>(options.value_or("amount", 0.0));
        if (by == "day") return timeseries_result(ts.shift_dates_by_day(amount));
        if (by == "month") return timeseries_result(ts.shift_dates_by_month(amount));
        if (by == "year") return timeseries_result(ts.shift_dates_by_year(amount));
        if (by == "start") {
            const std::vector<double>& start = data_at(data, 2, "timeseries", method);
            if (start.empty())
                throw std::runtime_error(
                    "toolbox method 'timeseries.shift' with by = start needs a new start date");
            ts.shift_all_dates(ts_data::DateTime::from_unix_seconds(start[0]));
            return timeseries_result(ts);
        }
        throw std::runtime_error("unknown time-series shift '" + by +
                                 "'; expected day, month, year, or start");
    }
    if (method == "convert_interval") {
        ts_data::TimeInterval to =
            parse_time_interval_token(options.value_or("to_interval", "one_day"));
        bool average = options.value_or("average", true);
        std::optional<ts_data::TimeSeries> converted = ts.convert_time_interval(to, average);
        if (!converted.has_value())
            throw std::runtime_error(
                "the time series cannot be converted between these intervals: month, quarter, "
                "year and irregular intervals have no fixed number of hours");
        return timeseries_result(*converted);
    }

    // --- Statistics. ---
    if (method == "summary_statistics") return timeseries_named(ts.summary_statistics());
    if (method == "summary_hypothesis_test") {
        int split = static_cast<int>(options.value_or("split_location", -1.0));
        return timeseries_named(ts.summary_hypothesis_test(split));
    }
    if (method == "percentiles") {
        std::vector<double> probabilities =
            data.size() > 2 ? data[2] : std::vector<double>{0.05, 0.25, 0.5, 0.75, 0.95};
        ToolboxResult r;
        r.values = ts.percentiles(probabilities);
        return r;
    }
    if (method == "duration") return timeseries_table(ts.duration());
    if (method == "monthly_percentiles") {
        std::vector<double> probabilities =
            data.size() > 2 ? data[2] : std::vector<double>{0.05, 0.25, 0.5, 0.75, 0.95};
        return timeseries_table(ts.monthly_percentiles(probabilities));
    }
    if (method == "monthly_summary_statistics") {
        ToolboxResult r = timeseries_table(ts.monthly_summary_statistics());
        r.names = {"minimum", "p05", "p25", "p50", "p75", "p95", "maximum", "mean"};
        return r;
    }
    if (method == "monthly_frequency") {
        ToolboxResult r;
        r.values = ts.monthly_frequency();
        return r;
    }

    // --- Frequency analysis. ---
    if (method == "block_series") {
        std::string block = options.value_or("block", "water_year");
        ts_data::BlockFunctionType block_function =
            parse_block_function_token(options.value_or("block_function", "maximum"));
        ts_data::SmoothingFunctionType smoothing =
            parse_smoothing_function_token(options.value_or("smoothing", "none"));
        int period = static_cast<int>(options.value_or("period", 1.0));
        int start_month = static_cast<int>(options.value_or("start_month", 10.0));
        int end_month = static_cast<int>(options.value_or("end_month", 9.0));
        if (block == "calendar_year")
            return timeseries_result(ts.calendar_year_series(block_function, smoothing, period));
        if (block == "water_year")
            return timeseries_result(
                ts.custom_year_series(start_month, block_function, smoothing, period));
        if (block == "custom_year")
            return timeseries_result(
                ts.custom_year_series(start_month, end_month, block_function, smoothing, period));
        if (block == "quarter")
            return timeseries_result(ts.quarterly_series(block_function, smoothing, period));
        if (block == "month")
            return timeseries_result(ts.monthly_series(block_function, smoothing, period));
        throw std::runtime_error("unknown time block '" + block +
                                 "'; expected calendar_year, water_year, custom_year, quarter, "
                                 "or month");
    }
    if (method == "peaks_over_threshold") {
        double threshold = options.value_or("threshold", 0.0);
        int min_steps = static_cast<int>(options.value_or("min_steps_between_events", 1.0));
        ts_data::SmoothingFunctionType smoothing =
            parse_smoothing_function_token(options.value_or("smoothing", "none"));
        int period = static_cast<int>(options.value_or("period", 1.0));
        return timeseries_result(
            ts.peaks_over_threshold_series(threshold, min_steps, smoothing, period));
    }

    // --- Decomposition and resampling. ---
    if (method == "seasonal_decompose") {
        int period = static_cast<int>(options.value_or("period", 12.0));
        int n = ts.count();
        // The three components have different lengths (trend and residual are only defined where
        // the moving-average window is full), so they are returned aligned to the ORIGINAL series
        // as an n-by-4 table -- date, trend, seasonal, residual -- with NaN where a component is
        // undefined. That is exactly the mapping the C# caller has to reconstruct by hand.
        ts_data::TimeSeries sorted = ts;
        sorted.sort_by_time();
        ts_data::TimeSeries::SeasonalDecomposition d = ts.seasonal_decompose(period);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        ToolboxResult r;
        r.dims = {n, 4};
        r.names = {"date", "trend", "seasonal", "residual"};
        int trend_start = period - 1;
        for (int i = 0; i < n; ++i) {
            r.values.push_back(sorted[i].index().to_unix_seconds());
            int t = i - trend_start;
            r.values.push_back(t >= 0 && t < d.trend.count() ? d.trend[t].value() : nan);
            r.values.push_back(d.seasonal[static_cast<std::size_t>(i)]);
            r.values.push_back(t >= 0 && t < d.residual.count() ? d.residual[t].value() : nan);
        }
        return r;
    }
    if (method == "resample_knn") {
        int time_steps = static_cast<int>(options.value_or("time_steps", 1.0));
        int k = static_cast<int>(options.value_or("k", 5.0));
        int seed = static_cast<int>(options.value_or("seed", 12345.0));
        return timeseries_result(ts.resample_with_knn(time_steps, k, seed));
    }
    if (method == "resample_block_bootstrap") {
        int time_steps = static_cast<int>(options.value_or("time_steps", 1.0));
        int block_size = static_cast<int>(options.value_or("block_size", 5.0));
        int seed = static_cast<int>(options.value_or("seed", 12345.0));
        return timeseries_result(ts.resample_with_block_bootstrap(time_steps, block_size, seed));
    }

    throw std::runtime_error("unknown toolbox method 'timeseries." + method + "'");
}

}  // namespace corehydro::numerics::support::detail
