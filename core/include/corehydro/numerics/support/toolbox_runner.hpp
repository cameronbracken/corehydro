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
// detail::run_<group> function and any helper used only by it. Shared types and helpers are in
// toolbox/common.hpp. This file includes all group headers and the run_toolbox dispatch table.
#pragma once

#include "corehydro/numerics/support/toolbox/common.hpp"

// --- group headers -----------------------------------------------------------------------
// One header per group. Later tasks add headers beside these; nothing else in this file changes
// when a group is added except this include block and the dispatch table below.
#include "corehydro/numerics/support/toolbox/correlation.hpp"
#include "corehydro/numerics/support/toolbox/gof.hpp"
#include "corehydro/numerics/support/toolbox/statistics.hpp"
#include "corehydro/numerics/support/toolbox/spectra.hpp"
#include "corehydro/numerics/support/toolbox/histogram.hpp"
#include "corehydro/numerics/support/toolbox/interpolation.hpp"
#include "corehydro/numerics/support/toolbox/regression.hpp"
#include "corehydro/numerics/support/toolbox/sampling.hpp"
#include "corehydro/numerics/support/toolbox/probability.hpp"

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
    if (group == "sampling") return detail::run_sampling(method, data, options);
    if (group == "probability") return detail::run_probability(method, data, options);
    throw std::runtime_error("unknown toolbox group: " + group);
}

}  // namespace corehydro::numerics::support
