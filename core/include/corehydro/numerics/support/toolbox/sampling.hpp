// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `sampling` group's dispatch arm: `sobol` (the quasi-random low-discrepancy
// SobolSequence) and `stratify` (Stratify::XValues over one StratificationOptions). Includes
// toolbox/common.hpp, which defines the shared ToolboxResult/data_at/scalar helpers used here.
//
// Sobol's direction-numbers file (new-joe-kuo-6.21201) ships with each package under a
// language-specific location (R: inst/extdata, resolved with system.file(); Python:
// corehydropy's own data/ package directory, resolved with importlib.resources); path
// resolution is therefore a wrapper concern, not a core one -- the resolved path crosses in
// `options.path`, required even when `dimension == 1` (where SobolSequence never reads it).
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/sampling/sobol.hpp"
#include "corehydro/numerics/sampling/stratification_bin.hpp"
#include "corehydro/numerics/sampling/stratification_options.hpp"
#include "corehydro/numerics/sampling/stratify.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

inline ToolboxResult run_sampling(const std::string& method,
                                  const std::vector<std::vector<double>>& data,
                                  const JsonValue& options) {
    namespace ns = numerics::sampling;
    (void)data;
    if (method == "sobol") {
        int dimension = options.value_or("dimension", 1);
        int n = options.value_or("n", 1);
        int skip = options.value_or("skip", 0);
        ns::SobolSequence sobol(dimension, options.at("path").as_string());
        ToolboxResult r;
        if (skip > 0) sobol.skip_to(skip);
        for (int i = 0; i < n; ++i) {
            std::vector<double> pt = sobol.next_double();
            r.values.insert(r.values.end(), pt.begin(), pt.end());
        }
        r.dims = {n, dimension};
        return r;
    }
    if (method == "stratify") {
        ns::StratificationOptions opts(options.at("lower").as_double(),
                                       options.at("upper").as_double(),
                                       options.at("bins").as_int(),
                                       options.value_or("probability", false));
        std::vector<ns::StratificationBin> bins =
            ns::Stratify::XValues(opts, options.value_or("logarithmic", false));
        ToolboxResult r;
        for (const ns::StratificationBin& b : bins) {
            r.values.push_back(b.lower_bound());
            r.values.push_back(b.upper_bound());
            r.values.push_back(b.midpoint());
            r.values.push_back(b.weight);
        }
        r.names = {"lower", "upper", "midpoint", "weight"};
        r.dims = {static_cast<int>(bins.size()), 4};
        return r;
    }
    throw std::runtime_error("unknown sampling method: " + method);
}

}  // namespace corehydro::numerics::support::detail
