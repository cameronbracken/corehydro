// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `probability` group's `joint` dispatch arm. `numerics/data/probability.hpp` offers
// two families: `joint_probability(probabilities, DependencyType)` over a plain probability
// vector, and the indicator form `joint_probability(probabilities, indicators,
// correlation_matrix*, DependencyType)` -- which, given a non-null correlation matrix and the
// default CorrelationMatrix dependency, routes to `joint_probability_hpcm` (Haden Smith's
// modification of Pandey's Product of Conditional Marginals). `indicators` present in `data`
// selects the second form; `correlation` present too (in addition to `indicators`) supplies the
// matrix that exercises the HPCM path. There is no `*_exceedance_*` family in the ported header.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/probability.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

inline ToolboxResult run_probability(const std::string& method,
                                     const std::vector<std::vector<double>>& data,
                                     const JsonValue& options) {
    namespace nd = numerics::data::probability;
    if (method != "joint") throw std::runtime_error("unknown probability method: " + method);
    const std::vector<double>& p = data_at(data, 0, "probability", method);

    std::string dep = options.value_or("dependency", "independent");
    nd::DependencyType type;
    if (dep == "independent") type = nd::DependencyType::Independent;
    else if (dep == "positive") type = nd::DependencyType::PerfectlyPositive;
    else if (dep == "negative") type = nd::DependencyType::PerfectlyNegative;
    else if (dep == "correlation") type = nd::DependencyType::CorrelationMatrix;
    else
        throw std::runtime_error("unknown dependency '" + dep +
                                 "'; expected independent, positive, negative, or correlation");

    if (data.size() < 2) {
        if (type == nd::DependencyType::CorrelationMatrix)
            throw std::runtime_error(
                "dependency 'correlation' needs an indicator vector and a correlation matrix");
        return scalar(nd::joint_probability(p, type));
    }

    // Indicator form: data[1] is the 0/1 indicator vector, data[2] the flattened correlation
    // matrix when there is one.
    const std::vector<double>& ind_d = data[1];
    std::vector<int> indicators(ind_d.size());
    for (std::size_t i = 0; i < ind_d.size(); ++i) indicators[i] = static_cast<int>(ind_d[i]);
    if (data.size() < 3) return scalar(nd::joint_probability(p, indicators, nullptr, type));

    std::size_t n = p.size();
    if (data[2].size() != n * n)
        throw std::runtime_error("the correlation matrix must be " + std::to_string(n) + " by " +
                                 std::to_string(n));
    nd::Matrix2D c(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) c[i][j] = data[2][i * n + j];
    return scalar(nd::joint_probability(p, indicators, &c, type));
}

}  // namespace corehydro::numerics::support::detail
