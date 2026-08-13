// corehydro ADDITION -- no upstream C# counterpart (sibling of
// distributions/support/dist_runner.hpp and estimation/support/fit_runner.hpp).
//
// The single place a Numerics utility method is dispatched in this repo. Four callers drive it
// and none owns any evaluation logic: the cpp11 glue (corehydror/src/toolbox.cpp), the pybind11
// glue (corehydropy/src/bindings/toolbox.cpp), the C++ fixture runner (core/tests/
// test_fixtures.cpp), and the dotnet oracle emitter (tools/oracle_emitter/Program.cs), which
// reads the same GRAMMAR against the real C# statics. A fixture case, an oracle replay, and a
// user's correlation() call are the same code path.
//
// Bulk data travels as native double vectors, not JSON: a goodness-of-fit call carries two
// series of arbitrary length and paying a JSON parse for them would be pointless. Scalars,
// enum names, and flags travel in `options_json`.
//
// Stateless by construction: one call builds whatever it needs, evaluates once, and drops it.
//
// The seven groups (correlation, gof, statistics, spectra, histogram, interpolation, regression)
// each live in their own header under numerics/support/toolbox/, holding that group's
// detail::run_<group> function and any helper used only by it. This file keeps the pieces those
// group headers share -- ToolboxResult, the detail::data_at/scalar helpers -- defined before
// including the group headers below, plus the run_toolbox dispatch table at the bottom.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/json_lite.hpp"

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

}  // namespace detail

}  // namespace corehydro::numerics::support

// --- group headers -----------------------------------------------------------------------
// One header per group. Later tasks add headers beside these; nothing else in this file changes
// when a group is added except this include block and the dispatch table below. Each group
// header reopens corehydro::numerics::support::detail and uses the ToolboxResult/data_at/scalar
// definitions above, so this include block must stay after them.
#include "corehydro/numerics/support/toolbox/correlation.hpp"
#include "corehydro/numerics/support/toolbox/gof.hpp"
#include "corehydro/numerics/support/toolbox/statistics.hpp"
#include "corehydro/numerics/support/toolbox/spectra.hpp"
#include "corehydro/numerics/support/toolbox/histogram.hpp"
#include "corehydro/numerics/support/toolbox/interpolation.hpp"
#include "corehydro/numerics/support/toolbox/regression.hpp"

namespace corehydro::numerics::support {

inline ToolboxResult run_toolbox(const std::string& group, const std::string& method,
                                 const std::vector<std::vector<double>>& data,
                                 const std::string& options_json) {
    JsonValue options = models::spec::parse_json(options_json.empty() ? "{}" : options_json);
    if (group == "correlation") return detail::run_correlation(method, data, options);
    if (group == "gof") return detail::run_gof(method, data, options);
    if (group == "statistics") return detail::run_statistics(method, data, options);
    if (group == "spectra") return detail::run_spectra(method, data, options);
    if (group == "histogram") return detail::run_histogram(method, data, options);
    if (group == "interpolation") return detail::run_interpolation(method, data, options);
    if (group == "regression") return detail::run_regression(method, data, options);
    throw std::runtime_error("unknown toolbox group: " + group);
}

}  // namespace corehydro::numerics::support
