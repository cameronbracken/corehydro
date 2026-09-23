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
//
// The `tabular`/`tabular_inverse` methods are the third `IUnivariateFunction`, TabularFunction
// (P4 Task 10), and use an entirely different grammar (no `"function"` key): `options.x` is the
// plain numeric array of the underlying UncertainOrderedPairedData's x-coordinates,
// `options.distributions` an array of `dist_spec` grammar objects (see
// `numerics/distributions/support/dist_spec.hpp`), one per `x` -- built and validated through
// toolbox/common.hpp's shared `build_distributions_for_x()`, which also keeps `paired_data.hpp`'s
// `curve_sample` from duplicating the same loop (P4 whole-branch-review finding C8: this group
// used to be the only one including another group header, `paired_data.hpp`, just to reuse that
// loop and its Transform parser -- both now live in common.hpp instead, so this header needs
// neither). `x_transform`/`y_transform` (`"none"`/`"logarithmic"` (or its alias `"log"`)/
// `"normal_z"`, default `"none"`), `confidence_level` (optional; its presence re-samples the curve
// at that quantile instead of the mean, mirroring `TabularFunction::set_confidence_level`),
// `is_deterministic` (optional bool, default `false`; `true` re-labels the underlying paired data
// Deterministic WITHOUT converting the individual Y distributions -- see `tabular_function.hpp`'s
// own header note, preserved here), and `allow_negative_y_values` (optional bool, default `true`,
// matching the C# class default) round out the options. The curve's shape contract is NOT
// configurable here (unlike `paired_data`'s own methods): `TabularFunction` is always built
// strict/ascending on both axes, matching every use in the ported ctest. `data[0]` holds the
// evaluation points -- `tabular` calls `Function()` at each, `tabular_inverse` calls
// `InverseFunction()`. The distribution type `UncertainOrderedPairedData` needs is derived from
// the FIRST built distribution's own `type()` (never a separate option): unlike
// `paired_data.curve_sample`, no `distribution_type` override exists here, since the brief's
// grammar for this method never lists one.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/support/dist_spec.hpp"
#include "corehydro/numerics/functions/i_univariate_function.hpp"
#include "corehydro/numerics/functions/ensemble_function.hpp"
#include "corehydro/numerics/functions/linear_function.hpp"
#include "corehydro/numerics/functions/power_function.hpp"
#include "corehydro/numerics/functions/tabular_function.hpp"
#include "corehydro/numerics/functions/univariate_function_factory.hpp"
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
    } else if (fn == "segmented_power") {
        if (parameters.size() < 4 || (parameters.size() - 1) % 3 != 0)
            throw std::runtime_error("segmented power parameters must have length 3*segments + 1");
        auto sf = std::make_unique<numerics::functions::SegmentedPowerFunction>(
            static_cast<int>((parameters.size() - 1) / 3));
        sf->set_is_deterministic(!has_confidence);
        sf->set_parameters(parameters);
        if (has_confidence) sf->set_confidence_level(options.at("confidence_level").as_double());
        f = std::move(sf);
    } else {
        throw std::runtime_error("unknown function type: " + fn);
    }
    return f;
}

inline ToolboxResult run_functions(const std::string& method,
                                   const std::vector<std::vector<double>>& data,
                                   const JsonValue& options) {
    if (method == "tabular" || method == "tabular_inverse") {
        const std::vector<double>& eval_points = data_at(data, 0, "functions", method);
        if (!options.contains("x"))
            throw std::runtime_error("toolbox method 'functions." + method + "' needs an 'x' option");
        std::vector<double> x = options.at("x").as_double_vector();
        DistributionList dl = build_distributions_for_x(options, x.size(), "functions", method);
        namespace nd = numerics::data;
        numerics::distributions::UnivariateDistributionType dtype = dl.dists.front()->type();
        numerics::data::paired_data::UncertainOrderedPairedData upd(
            x, dl.ptrs, true, nd::SortOrder::Ascending, true, nd::SortOrder::Ascending, dtype);
        numerics::functions::TabularFunction f(std::move(upd));
        f.set_x_transform(parse_transform_token(options.value_or("x_transform", "none")));
        f.set_y_transform(parse_transform_token(options.value_or("y_transform", "none")));
        f.set_extrapolation(parse_extrapolation_token(options.value_or("extrapolation", "none")));
        if (options.value_or("is_deterministic", false)) f.set_is_deterministic(true);
        if (options.contains("confidence_level"))
            f.set_confidence_level(options.at("confidence_level").as_double());
        f.set_allow_negative_y_values(options.value_or("allow_negative_y_values", true));
        ToolboxResult r;
        for (double v : eval_points)
            r.values.push_back(method == "tabular" ? f.function(v) : f.inverse_function(v));
        return r;
    }

    std::unique_ptr<numerics::functions::IUnivariateFunction> f;
    if (options.contains("spec")) {
        const JsonValue& spec = options.at("spec");
        if (spec.at("type").as_string() == "ensemble") {
            auto function_template = numerics::functions::build_function_spec(spec.at("template"));
            std::vector<std::vector<double>> parameter_sets;
            for (const JsonValue& values : spec.at("parameter_sets").items())
                parameter_sets.push_back(values.as_double_vector());
            numerics::functions::EnsembleFunction ensemble(std::move(function_template),
                                                            std::move(parameter_sets));
            if (spec.contains("sample_index"))
                f = ensemble.sample_at(spec.at("sample_index").as_int());
            else if (spec.contains("sample_percentile"))
                f = ensemble.sample(spec.at("sample_percentile").as_double());
            else
                throw std::runtime_error(
                    "ensemble function specs need sample_index or sample_percentile");
        } else {
            f = numerics::functions::build_function_spec(spec);
        }
        if (options.contains("confidence_level")) {
            f->set_is_deterministic(false);
            f->set_confidence_level(options.at("confidence_level").as_double());
        }
    } else if (options.contains("function")) {
        f = build_univariate_function(options);
    } else {
        throw std::runtime_error(
            "toolbox group 'functions' needs a 'function' or 'spec' key in its options");
    }
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
