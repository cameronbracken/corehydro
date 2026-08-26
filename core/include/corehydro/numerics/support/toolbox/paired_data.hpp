// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `paired_data` group's nine dispatch arms over the ported Paired Data subsystem
// (numerics/data/paired_data/{ordered_paired_data,uncertain_ordered_paired_data,
// line_simplification}.hpp, P4 Tasks 7-9). Every method that builds an `OrderedPairedData` reads
// the curve's shape contract from four shared options -- `strict_x`/`strict_y` (bool, default
// `true`) and `order_x`/`order_y` (the `SortOrder` names `"ascending"`/`"descending"`/`"none"`,
// default `"ascending"`) -- and, where the method actually interpolates
// (`interpolate_y`/`interpolate_x`), two more: `x_transform`/`y_transform` (the `Transform` names
// `"none"`/`"logarithmic"` (or its alias `"log"`)/`"normal_z"`, default `"none"`). `line_simplify` builds no
// `OrderedPairedData` at all (`line_simplification::ramer_douglas_peucker` takes a bare ordinate
// list plus an epsilon), so it reads none of these six.
//
// STATELESS DISPATCH (`search`): the C# search methods this un-gates (SequentialSearchX/Y,
// BisectionSearchX/Y, HuntSearchX/Y, the smart SearchX/Y dispatcher, and BinarySearchX/Y) mutate
// `XSearchStart`/`YSearchStart`/`Xcorrelated`/`Ycorrelated` on the instance, so a second search on
// the SAME `OrderedPairedData` can return a different index than the first for the identical
// query, depending on call history (see ordered_paired_data.hpp's own transcription notes 2-3).
// This group builds the object, searches ONCE, and drops it -- there is no way to carry that
// call-history-dependent state across the R/Python boundary meaningfully, so a caller must not
// expect the C# stateful behavior across separate `search()` calls the way a live C# instance
// would provide it.
//
// VALIDATION MESSAGES DO NOT CROSS THE BOUNDARY (Scope decision 5): `is_valid`'s C# counterpart,
// `OrderedPairedData.GetErrors()`, returns a `List<string>` of human-readable messages ("X values
// must increase.", ...). `ToolboxResult` carries doubles and names, not strings, so `is_valid`
// reports only the COUNT of errors (`error_count`) alongside the boolean `is_valid` flag; the
// message text itself is reachable only from C++ (`OrderedPairedData::get_errors()`/
// `UncertainOrderedPairedData::get_errors()`) and is exercised by the ctest suites, not by this
// toolbox surface.
//
// `curve_sample` builds each Y distribution through `distributions::support::build_univariate`
// (numerics/distributions/support/dist_spec.hpp) -- the SAME grammar `dist_spec`/`dist_runner`
// use -- rather than a second distribution parser, keeping "one place a distribution is built
// from a spec" true for this surface too. Its `distributions` array must already be the same
// length as `x`: unlike the user-facing verb's "one distribution recycled across all x"
// convenience, this group does no recycling of its own -- recycling is an R/Python-layer
// decision, and this header stays a thin, stateless builder. `distribution_type` (a family name,
// resolved to the `UnivariateDistributionType` the `UncertainOrderedPairedData` constructor
// needs) is OPTIONAL here even though the C# constructor has no default: when omitted, the first
// built distribution's own `type()` is used. That default is exact, not a guess -- `curve_sample`
// (mean or probability-quantile) never consults `is_valid()`/`distribution()`, so the type is
// inert to the computed curve, and the R/Python `uncertain_curve_sample()` glue relies on exactly
// this default rather than asking the caller to name a type redundantly. The option still exists
// so a fixture case or the oracle emitter can pin the SAME literal type name the C# test
// constructs with (`UnivariateDistributionType.Triangular`, ...).
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/paired_data/line_simplification.hpp"
#include "corehydro/numerics/data/paired_data/ordered_paired_data.hpp"
#include "corehydro/numerics/data/paired_data/uncertain_ordered_paired_data.hpp"
#include "corehydro/numerics/distributions/support/dist_spec.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

namespace pd = corehydro::numerics::data::paired_data;

// This group's SortOrder/Transform token parsing reuses toolbox/common.hpp's shared
// parse_sort_order_token()/parse_transform_token() (see file header for the accepted spellings).
// "none" is a valid SortOrder here, unlike interpolation.hpp's group (which never passes
// allow_none = true), because OrderedPairedData's SortOrder::None is reachable and exercised (the
// shared five-point sin curve's y-axis in the simplification tests uses it); this group's local
// wrapper below just pins allow_none = true at every call site.
inline corehydro::numerics::data::SortOrder paired_data_sort_order(const std::string& s) {
    return parse_sort_order_token(s, /*allow_none=*/true);
}

// Builds an OrderedPairedData from data [x, y] and the four shape-flag options.
inline pd::OrderedPairedData paired_data_build_opd(const std::vector<std::vector<double>>& data,
                                                    const JsonValue& options,
                                                    const std::string& method) {
    const std::vector<double>& x = data_at(data, 0, "paired_data", method);
    const std::vector<double>& y = data_at(data, 1, "paired_data", method);
    if (x.size() != y.size())
        throw std::runtime_error("toolbox method 'paired_data." + method +
                                 "' needs x and y vectors of the same length; got " +
                                 std::to_string(x.size()) + " and " + std::to_string(y.size()));
    bool strict_x = options.value_or("strict_x", true);
    bool strict_y = options.value_or("strict_y", true);
    corehydro::numerics::data::SortOrder order_x =
        paired_data_sort_order(options.value_or("order_x", "ascending"));
    corehydro::numerics::data::SortOrder order_y =
        paired_data_sort_order(options.value_or("order_y", "ascending"));
    return pd::OrderedPairedData(x, y, strict_x, order_x, strict_y, order_y);
}

// Flattens a curve, row-major, into a ToolboxResult with dims = {n, 2}.
inline ToolboxResult curve_result(const pd::OrderedPairedData& opd) {
    ToolboxResult r;
    r.dims = {opd.count(), 2};
    r.values.reserve(static_cast<std::size_t>(opd.count()) * 2);
    for (int i = 0; i < opd.count(); ++i) {
        r.values.push_back(opd[i].x);
        r.values.push_back(opd[i].y);
    }
    return r;
}
inline ToolboxResult curve_result(const std::vector<pd::Ordinate>& ords) {
    ToolboxResult r;
    r.dims = {static_cast<int>(ords.size()), 2};
    r.values.reserve(ords.size() * 2);
    for (const pd::Ordinate& o : ords) {
        r.values.push_back(o.x);
        r.values.push_back(o.y);
    }
    return r;
}

inline ToolboxResult run_paired_data(const std::string& method,
                                     const std::vector<std::vector<double>>& data,
                                     const JsonValue& options) {
    if (method == "interpolate_y") {
        pd::OrderedPairedData opd = paired_data_build_opd(data, options, method);
        const std::vector<double>& xout = data_at(data, 2, "paired_data", method);
        corehydro::numerics::data::Transform xt =
            parse_transform_token(options.value_or("x_transform", "none"));
        corehydro::numerics::data::Transform yt =
            parse_transform_token(options.value_or("y_transform", "none"));
        ToolboxResult r;
        for (double v : xout) r.values.push_back(opd.get_y_from_x(v, xt, yt));
        return r;
    }

    if (method == "interpolate_x") {
        pd::OrderedPairedData opd = paired_data_build_opd(data, options, method);
        const std::vector<double>& yout = data_at(data, 2, "paired_data", method);
        corehydro::numerics::data::Transform xt =
            parse_transform_token(options.value_or("x_transform", "none"));
        corehydro::numerics::data::Transform yt =
            parse_transform_token(options.value_or("y_transform", "none"));
        ToolboxResult r;
        for (double v : yout) r.values.push_back(opd.get_x_from_y(v, xt, yt));
        return r;
    }

    if (method == "area_under_y") {
        pd::OrderedPairedData opd = paired_data_build_opd(data, options, method);
        return scalar(opd.trapezoidal_area_under_y());
    }

    if (method == "area_under_x") {
        pd::OrderedPairedData opd = paired_data_build_opd(data, options, method);
        return scalar(opd.trapezoidal_area_under_x());
    }

    if (method == "simplify") {
        pd::OrderedPairedData opd = paired_data_build_opd(data, options, method);
        if (!options.contains("algorithm"))
            throw std::runtime_error(
                "toolbox method 'paired_data.simplify' needs an 'algorithm' option");
        const std::string& algorithm = options.at("algorithm").as_string();
        if (algorithm == "rdp") {
            if (!options.contains("tolerance"))
                throw std::runtime_error(
                    "toolbox method 'paired_data.simplify' with algorithm 'rdp' needs a "
                    "'tolerance' option");
            return curve_result(opd.douglas_peucker_simplify(options.at("tolerance").as_double()));
        }
        if (algorithm == "visvalingam") {
            if (!options.contains("num_to_keep"))
                throw std::runtime_error(
                    "toolbox method 'paired_data.simplify' with algorithm 'visvalingam' needs a "
                    "'num_to_keep' option");
            return curve_result(
                opd.visvaligam_whyatt_simplify(options.at("num_to_keep").as_int()));
        }
        if (algorithm == "lang") {
            if (!options.contains("tolerance") || !options.contains("look_ahead"))
                throw std::runtime_error(
                    "toolbox method 'paired_data.simplify' with algorithm 'lang' needs "
                    "'tolerance' and 'look_ahead' options");
            return curve_result(opd.lang_simplify(options.at("tolerance").as_double(),
                                                  options.at("look_ahead").as_int()));
        }
        throw std::runtime_error("unknown paired_data.simplify algorithm '" + algorithm +
                                 "'; expected rdp, visvalingam, or lang");
    }

    if (method == "line_simplify") {
        const std::vector<double>& x = data_at(data, 0, "paired_data", method);
        const std::vector<double>& y = data_at(data, 1, "paired_data", method);
        if (x.size() != y.size())
            throw std::runtime_error(
                "toolbox method 'paired_data.line_simplify' needs x and y vectors of the same "
                "length; got " +
                std::to_string(x.size()) + " and " + std::to_string(y.size()));
        if (!options.contains("epsilon"))
            throw std::runtime_error(
                "toolbox method 'paired_data.line_simplify' needs an 'epsilon' option");
        std::vector<pd::Ordinate> ords;
        ords.reserve(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) ords.emplace_back(x[i], y[i]);
        std::vector<pd::Ordinate> output;
        pd::line_simplification::ramer_douglas_peucker(ords, options.at("epsilon").as_double(),
                                                        output);
        return curve_result(output);
    }

    if (method == "search") {
        pd::OrderedPairedData opd = paired_data_build_opd(data, options, method);
        if (!options.contains("value"))
            throw std::runtime_error("toolbox method 'paired_data.search' needs a 'value' option");
        double value = options.at("value").as_double();
        std::string axis = options.value_or("axis", "x");
        std::string algorithm = options.value_or("algorithm", "smart");
        int idx = 0;
        if (axis == "x") {
            if (algorithm == "smart") idx = opd.search_x(value);
            else if (algorithm == "sequential") idx = opd.sequential_search_x(value);
            else if (algorithm == "bisection") idx = opd.bisection_search_x(value);
            else if (algorithm == "hunt") idx = opd.hunt_search_x(value);
            else if (algorithm == "binary") idx = opd.binary_search_x(value);
            else
                throw std::runtime_error("unknown paired_data.search algorithm '" + algorithm +
                                         "'; expected smart, sequential, bisection, hunt, or binary");
        } else if (axis == "y") {
            if (algorithm == "smart") idx = opd.search_y(value);
            else if (algorithm == "sequential") idx = opd.sequential_search_y(value);
            else if (algorithm == "bisection") idx = opd.bisection_search_y(value);
            else if (algorithm == "hunt") idx = opd.hunt_search_y(value);
            else if (algorithm == "binary") idx = opd.binary_search_y(value);
            else
                throw std::runtime_error("unknown paired_data.search algorithm '" + algorithm +
                                         "'; expected smart, sequential, bisection, hunt, or binary");
        } else {
            throw std::runtime_error("toolbox method 'paired_data.search' needs axis 'x' or 'y'; "
                                     "got '" +
                                     axis + "'");
        }
        return scalar(static_cast<double>(idx));
    }

    if (method == "is_valid") {
        pd::OrderedPairedData opd = paired_data_build_opd(data, options, method);
        ToolboxResult r;
        r.values = {opd.is_valid() ? 1.0 : 0.0, static_cast<double>(opd.get_errors().size())};
        r.names = {"is_valid", "error_count"};
        return r;
    }

    if (method == "curve_sample") {
        const std::vector<double>& x = data_at(data, 0, "paired_data", method);
        DistributionList dl = build_distributions_for_x(options, x.size(), "paired_data", method);
        bool strict_x = options.value_or("strict_x", true);
        bool strict_y = options.value_or("strict_y", true);
        corehydro::numerics::data::SortOrder order_x =
            paired_data_sort_order(options.value_or("order_x", "ascending"));
        corehydro::numerics::data::SortOrder order_y =
            paired_data_sort_order(options.value_or("order_y", "ascending"));
        numerics::distributions::UnivariateDistributionType dtype;
        if (options.contains("distribution_type")) {
            const std::string& type_name = options.at("distribution_type").as_string();
            try {
                dtype = numerics::distributions::create_distribution(type_name)->type();
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "toolbox method 'paired_data.curve_sample' has an unknown distribution "
                    "family '" +
                    type_name + "' for 'distribution_type'");
            }
        } else {
            dtype = dl.dists.front()->type();
        }
        pd::UncertainOrderedPairedData uopd(x, dl.ptrs, strict_x, order_x, strict_y, order_y,
                                             dtype);
        pd::OrderedPairedData sampled = options.contains("probability")
                                            ? uopd.curve_sample(options.at("probability").as_double())
                                            : uopd.curve_sample();
        return curve_result(sampled);
    }

    throw std::runtime_error("unknown paired_data method: " + method);
}

}  // namespace corehydro::numerics::support::detail
