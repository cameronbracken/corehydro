// corehydro ADDITION -- shared surface for all toolbox group headers, no upstream C# counterpart.
//
// Holds the ToolboxResult struct and shared detail:: helpers (data_at, scalar) that every
// group header under toolbox/ includes. This header is self-contained and may be included
// before any group header.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/data/interpolation/sort_order.hpp"
#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/distributions/support/dist_spec.hpp"

namespace corehydro::numerics::support {

using corehydro::models::spec::JsonValue;

// Flat result surface every binding and every fixture assertion reads. `values` holds whatever
// the method returns, in method order; `names` labels them when the method returns a named set;
// `dims` carries {rows, columns} when the method returns a matrix flattened row-major into
// `values`, and is empty otherwise; `spec` carries a child object back and is empty otherwise.
struct ToolboxResult {
    std::vector<double> values;
    std::vector<std::string> names;
    std::vector<int> dims;
    std::string spec;
};

namespace detail {

inline const std::vector<double>& data_at(const std::vector<std::vector<double>>& data,
                                          std::size_t i, const std::string& group,
                                          const std::string& method) {
    if (i >= data.size())
        throw std::runtime_error("toolbox method '" + group + "." + method + "' needs " +
                                 std::to_string(i + 1) + " data vector(s), got " +
                                 std::to_string(data.size()));
    return data[i];
}

inline ToolboxResult scalar(double v) {
    ToolboxResult r;
    r.values = {v};
    return r;
}

// Shared `Transform` token parser for every toolbox group that reads a "*_transform" option
// (interpolation, paired_data, functions). Accepts BOTH "log" (interpolation.hpp's original, sole
// pre-existing spelling) and "logarithmic" (paired_data.hpp's and functions.hpp's spelling,
// matching the C# enum member name `Transform.Logarithmic`) as synonyms for
// `Transform::Logarithmic`, so the same value works through whichever host verb it arrives from --
// see the P4 whole-branch-review finding M2, where `curve_interpolate(x_transform = "log")` and
// `interpolate(x_transform = "log")` disagreed on whether "log" was valid.
inline corehydro::numerics::data::Transform parse_transform_token(const std::string& s) {
    if (s == "none") return corehydro::numerics::data::Transform::None;
    if (s == "log" || s == "logarithmic") return corehydro::numerics::data::Transform::Logarithmic;
    if (s == "normal_z") return corehydro::numerics::data::Transform::NormalZ;
    throw std::runtime_error("unknown transform '" + s +
                             "'; expected none, log, logarithmic, or normal_z");
}

// Shared `SortOrder` token parser. `allow_none` widens it to accept "none" (`SortOrder::None`):
// paired_data's `OrderedPairedData` legitimately constructs with it (the shared five-point sin
// curve's y-axis in the simplification tests uses `SortOrder::None`), but the interpolation
// group's Linear/Bilinear/CubicSpline/Polynomial classes only ever branch on
// `SortOrder::Ascending` vs. `::Descending`, so `interpolate()`/`interpolate_2d()` keep rejecting
// "none" at the host layer and never pass `allow_none = true` here.
inline corehydro::numerics::data::SortOrder parse_sort_order_token(const std::string& s,
                                                                    bool allow_none = false) {
    if (s == "ascending") return corehydro::numerics::data::SortOrder::Ascending;
    if (s == "descending") return corehydro::numerics::data::SortOrder::Descending;
    if (allow_none && s == "none") return corehydro::numerics::data::SortOrder::None;
    throw std::runtime_error("unknown sort order '" + s + "'; expected " +
                             (allow_none ? "ascending, descending, or none"
                                         : "ascending or descending"));
}

// Result of build_distributions_for_x(): one built distribution (owning) per element of
// `options["distributions"]`, plus the raw pointers the OrderedPairedData/TabularFunction
// constructors take.
struct DistributionList {
    std::vector<std::unique_ptr<numerics::distributions::UnivariateDistributionBase>> dists;
    std::vector<const numerics::distributions::UnivariateDistributionBase*> ptrs;
};

// Builds one `UnivariateDistributionBase` per element of `options["distributions"]` (the
// `dist_spec` grammar `dist_spec.hpp`/`dist_runner.hpp` use), and validates the list is non-empty
// and has exactly one distribution per element of `x`. Shared by paired_data's `curve_sample` and
// functions' `tabular`/`tabular_inverse`, which built and validated this list with verbatim-
// duplicated logic before (P4 whole-branch-review finding C8).
inline DistributionList build_distributions_for_x(const JsonValue& options, std::size_t x_size,
                                                   const std::string& group,
                                                   const std::string& method) {
    if (!options.contains("distributions"))
        throw std::runtime_error("toolbox method '" + group + "." + method +
                                 "' needs a 'distributions' option");
    DistributionList out;
    for (const JsonValue& spec : options.at("distributions").items()) {
        out.dists.push_back(numerics::distributions::support::build_univariate(spec));
        out.ptrs.push_back(out.dists.back().get());
    }
    if (out.dists.empty())
        throw std::runtime_error("toolbox method '" + group + "." + method +
                                 "' needs at least one distribution");
    if (out.ptrs.size() != x_size)
        throw std::runtime_error("toolbox method '" + group + "." + method +
                                 "' needs one distribution per x value; got " +
                                 std::to_string(out.ptrs.size()) + " distributions for " +
                                 std::to_string(x_size) + " x values");
    return out;
}

}  // namespace detail

}  // namespace corehydro::numerics::support
