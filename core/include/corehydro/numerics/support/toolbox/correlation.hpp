// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `correlation` group's dispatch arm: Pearson, Spearman, and Kendall's tau, each a
// scalar over two series. Includes toolbox/common.hpp, which defines the shared
// ToolboxResult/data_at/scalar helpers used here.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/correlation.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

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

}  // namespace corehydro::numerics::support::detail
