// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `correlation` group's dispatch arm: Pearson, Spearman, and Kendall's tau, each a
// scalar over two series, plus the Pearson/Spearman matrix overloads (`pearson_matrix` /
// `spearman_matrix`), each over p data vectors (one per column) returning the p-by-p matrix
// flattened row-major with `dims = {p, p}` -- there is no `kendall_matrix` arm, matching
// upstream's Correlation class, which has no KendallsTau(double[,]) overload. Includes
// toolbox/common.hpp, which defines the shared ToolboxResult/data_at/scalar helpers used here.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/correlation.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

// Serializes a p-by-p correlation matrix to a ToolboxResult, row-major, with dims = {p, p} --
// the same convention linalg.hpp's matrix_result uses.
inline ToolboxResult correlation_matrix_result(const std::vector<std::vector<double>>& corr) {
    ToolboxResult r;
    const int p = static_cast<int>(corr.size());
    r.dims = {p, p};
    r.values.reserve(static_cast<std::size_t>(p) * static_cast<std::size_t>(p));
    for (const auto& row : corr)
        for (double v : row) r.values.push_back(v);
    return r;
}

inline ToolboxResult run_correlation(const std::string& method,
                                     const std::vector<std::vector<double>>& data,
                                     const JsonValue& options) {
    (void)options;
    if (method == "pearson_matrix") return correlation_matrix_result(numerics::data::pearson_matrix(data));
    if (method == "spearman_matrix") return correlation_matrix_result(numerics::data::spearman_matrix(data));

    const std::vector<double>& x = data_at(data, 0, "correlation", method);
    const std::vector<double>& y = data_at(data, 1, "correlation", method);
    if (method == "pearson") return scalar(numerics::data::pearson(x, y));
    if (method == "spearman") return scalar(numerics::data::spearman(x, y));
    if (method == "kendall") return scalar(numerics::data::kendalls_tau(x, y));
    throw std::runtime_error("unknown correlation method: " + method);
}

}  // namespace corehydro::numerics::support::detail
