// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `special` group's dispatch arms over debye_function and the three evaluate_*
// polynomial helpers (numerics/math/special/{debye,evaluate}.hpp). All four methods evaluate a
// scalar function at every element of a data vector `x` and return the per-element results in
// `values`, the same "vectorize over the last data vector" shape `trend.hpp`'s `predict` arm
// uses -- there is no matrix or named-set result here, so `dims`/`names` are never set.
#pragma once

#include <string>
#include <vector>

#include "corehydro/numerics/math/special/debye.hpp"
#include "corehydro/numerics/math/special/evaluate.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

// debye (data [x]) -> Debye function values per x.
// polynomial / polynomial_rev / polynomial_rev_1 (data [coefficients, x]) -> polynomial values
// per x, evaluated with the shared `coefficients` vector; `polynomial_rev` additionally reads an
// optional integer `n` option (mirrors Evaluate.PolynomialRev's optional `n` parameter).
inline ToolboxResult run_special(const std::string& method,
                                 const std::vector<std::vector<double>>& data,
                                 const JsonValue& options) {
    if (method == "debye") {
        ToolboxResult r;
        for (double x : data_at(data, 0, "special", method))
            r.values.push_back(corehydro::numerics::math::special::debye_function(x));
        return r;
    }

    if (method == "polynomial" || method == "polynomial_rev" || method == "polynomial_rev_1") {
        const std::vector<double>& coefficients = data_at(data, 0, "special", method);
        const std::vector<double>& xs = data_at(data, 1, "special", method);
        ToolboxResult r;
        if (method == "polynomial") {
            for (double x : xs)
                r.values.push_back(
                    corehydro::numerics::math::special::evaluate_polynomial(coefficients, x));
        } else if (method == "polynomial_rev") {
            int n = options.value_or("n", -1);
            for (double x : xs)
                r.values.push_back(corehydro::numerics::math::special::evaluate_polynomial_rev(
                    coefficients, x, n));
        } else {
            for (double x : xs)
                r.values.push_back(
                    corehydro::numerics::math::special::evaluate_polynomial_rev_1(coefficients, x));
        }
        return r;
    }

    throw std::runtime_error("unknown special method: " + method);
}

}  // namespace corehydro::numerics::support::detail
