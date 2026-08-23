// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `functions` group's `evaluate`/`inverse` dispatch arm over the two-name spec grammar
// `{"function": "linear" | "power", "parameters": [...], "is_inverse"?: bool,
// "confidence_level"?: number}`. `function` selects which `IUnivariateFunction`
// (numerics/functions/) to build; `parameters` is a positional array in the same order
// `IUnivariateFunction::set_parameters` takes (linear: `[alpha, beta, sigma]`; power: `[alpha,
// beta, xi, sigma]`) -- unlike `link`'s named `parameters` object, both functions here take one
// fixed, ordered parameter list, so a positional array is unambiguous and matches how every
// distribution/copula spec in this repo already passes parameters. `is_inverse` is
// PowerFunction-only (LinearFunction has no such switch) and scoped-checked here exactly as
// `link`'s `inner` key is scoped to `"Centered"`. `confidence_level`'s mere PRESENCE selects the
// non-deterministic path: it sets `is_deterministic(false)` before `set_parameters` runs (so the
// parameter-3/parameter-4 sigma validation applies) and then applies the level -- ABSENCE means
// deterministic, matching every constructor overload in linear_function.hpp/power_function.hpp
// that omits sigma.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/functions/i_univariate_function.hpp"
#include "corehydro/numerics/functions/linear_function.hpp"
#include "corehydro/numerics/functions/power_function.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

// Builds a LinearFunction/PowerFunction from the options spec. `confidence_level`'s presence (not
// its value) is what switches the function to non-deterministic -- mirroring how every
// non-deterministic C# constructor is the one that takes `sigma`.
inline std::unique_ptr<numerics::functions::IUnivariateFunction> build_univariate_function(
    const JsonValue& options) {
    const std::string& fn = options.at("function").as_string();
    std::vector<double> parameters = options.at("parameters").as_double_vector();
    bool has_confidence = options.contains("confidence_level");

    if (options.contains("is_inverse") && fn != "power")
        throw std::runtime_error("'is_inverse' is only valid for function 'power'; got function '" +
                                 fn + "'");

    std::unique_ptr<numerics::functions::IUnivariateFunction> f;
    if (fn == "linear") {
        auto lf = std::make_unique<numerics::functions::LinearFunction>();
        lf->set_is_deterministic(!has_confidence);
        lf->set_parameters(parameters);
        if (has_confidence) lf->set_confidence_level(options.at("confidence_level").as_double());
        f = std::move(lf);
    } else if (fn == "power") {
        auto pf = std::make_unique<numerics::functions::PowerFunction>();
        pf->set_is_deterministic(!has_confidence);
        pf->set_is_inverse(options.value_or("is_inverse", false));
        pf->set_parameters(parameters);
        if (has_confidence) pf->set_confidence_level(options.at("confidence_level").as_double());
        f = std::move(pf);
    } else {
        throw std::runtime_error("unknown function type: " + fn);
    }
    return f;
}

inline ToolboxResult run_functions(const std::string& method,
                                   const std::vector<std::vector<double>>& data,
                                   const JsonValue& options) {
    if (!options.contains("function"))
        throw std::runtime_error(
            "toolbox group 'functions' needs a 'function' key in its options");
    std::unique_ptr<numerics::functions::IUnivariateFunction> f = build_univariate_function(options);
    const std::vector<double>& x = data_at(data, 0, "functions", method);
    ToolboxResult r;
    for (double v : x) {
        if (method == "evaluate") r.values.push_back(f->function(v));
        else if (method == "inverse") r.values.push_back(f->inverse_function(v));
        else throw std::runtime_error("unknown functions method: " + method);
    }
    return r;
}

}  // namespace corehydro::numerics::support::detail
