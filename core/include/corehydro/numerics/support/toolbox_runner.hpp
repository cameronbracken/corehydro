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
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/data/correlation.hpp"

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

// --- group arms -------------------------------------------------------------------------
// One function per group. Later tasks add arms beside this one; nothing else in the file
// changes when a group is added except the dispatch table at the bottom.

inline ToolboxResult run_correlation(const std::string& method,
                                     const std::vector<std::vector<double>>& data,
                                     const JsonValue& options) {
    (void)options;
    const std::vector<double>& x = data_at(data, 0, "correlation", method);
    const std::vector<double>& y = data_at(data, 1, "correlation", method);
    if (method == "pearson") return scalar(numerics::data::pearson(x, y));
    if (method == "spearman") return scalar(numerics::data::spearman(x, y));
    if (method == "kendall") return scalar(numerics::data::kendalls_tau(x, y));
    throw std::runtime_error("unknown correlation method: " + method);
}

}  // namespace detail

inline ToolboxResult run_toolbox(const std::string& group, const std::string& method,
                                 const std::vector<std::vector<double>>& data,
                                 const std::string& options_json) {
    JsonValue options = models::spec::parse_json(options_json.empty() ? "{}" : options_json);
    if (group == "correlation") return detail::run_correlation(method, data, options);
    throw std::runtime_error("unknown toolbox group: " + group);
}

}  // namespace corehydro::numerics::support
