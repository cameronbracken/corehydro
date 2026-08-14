// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `link` group's `link`/`inverse_link`/`d_link` dispatch arm over the thirteen-name
// spec grammar `{"type": ..., "parameters"?: {...}, "inner"?: {...}}`. `type` names either the
// seven-member Numerics `LinkFunctionType` enum (numerics/functions/link_function_type.hpp) or
// one of the five BestFit-specific link types (models/link_functions/); the Numerics factory is
// tried first and the BestFit-specific constructors second, mirroring how
// `BestFitLinkFunctionFactory` itself falls through to the Numerics `LinkFunctionFactory` for a
// type it does not own (best_fit_link_function_factory.hpp). `parameters` is a named object
// (not a positional array) because the parameterized types take different argument lists and a
// bare array would be unreadable at the call site; `inner` is present only for `"Centered"`,
// itself a full nested link spec.
//
// Twelve names, not thirteen: the seven Numerics links (Identity, Log, Logit, Probit,
// ComplementaryLogLog, YeoJohnson, FisherZ) plus the five link-specific BestFit types (ASinH,
// SES, LogSES, LogASinH, Centered) -- YeoJohnson counted once even though both factories can
// name it (best_fit_link_function_factory.hpp routes its own YeoJohnson case straight to the
// Numerics class since the upstream C# deleted its BestFit-local duplicate; there was never a
// second YeoJohnson implementation to expose here).
#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/link_functions/asinh_link.hpp"
#include "corehydro/models/link_functions/centered_link.hpp"
#include "corehydro/models/link_functions/log_asinh_link.hpp"
#include "corehydro/models/link_functions/log_ses_link.hpp"
#include "corehydro/models/link_functions/ses_link.hpp"
#include "corehydro/numerics/functions/i_link_function.hpp"
#include "corehydro/numerics/functions/link_function_factory.hpp"
#include "corehydro/numerics/functions/link_function_type.hpp"
#include "corehydro/numerics/functions/yeo_johnson_link.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

// Forward declaration: the Centered entry in link_builder_table() below recurses into
// build_link() for its "inner" spec.
inline std::unique_ptr<numerics::functions::ILinkFunction> build_link(const JsonValue& spec);

// `parameters` is a named object (not a positional array) because the six parameterized links
// take different argument lists and a positional array would be unreadable at the call site.
inline double link_param(const JsonValue& spec, const char* key, double dflt) {
    const JsonValue* p = spec.contains("parameters") ? &spec.at("parameters") : nullptr;
    return p && p->contains(key) ? p->at(key).as_double() : dflt;
}

using LinkBuilderFn = std::function<std::unique_ptr<numerics::functions::ILinkFunction>(const JsonValue&)>;

// The ONE place the twelve accepted link type names are listed -- build_link() and the "names"
// toolbox method (run_link, below) both read this table, so a name can't drift between what a
// spec is allowed to say and what `link_names()` (R/Python) advertises. Every constructor
// argument and default below is read off the real header (numerics/functions/
// yeo_johnson_link.hpp, models/link_functions/*.hpp) rather than guessed: notably CenteredLink
// takes a THIRD constructor argument, `scale` (default 1.0, the C# `CenteredLink.Scale`
// property), that a first draft of this spec omitted -- it is included here as
// `link_param(spec, "scale", 1.0)` so a spec can actually reach a non-default Centered scale.
//
// Twelve names, not thirteen: the seven Numerics links (Identity, Log, Logit, Probit,
// ComplementaryLogLog, YeoJohnson, FisherZ) plus the five link-specific BestFit types (ASinH,
// SES, LogSES, LogASinH, Centered) -- YeoJohnson counted once even though both factories can
// name it (see the file header above).
inline const std::vector<std::pair<std::string, LinkBuilderFn>>& link_builder_table() {
    using LT = numerics::functions::LinkFunctionType;
    static const std::vector<std::pair<std::string, LinkBuilderFn>> table = {
        {"Identity",
         [](const JsonValue&) { return numerics::functions::LinkFunctionFactory::create(LT::Identity); }},
        {"Log",
         [](const JsonValue&) { return numerics::functions::LinkFunctionFactory::create(LT::Log); }},
        {"Logit",
         [](const JsonValue&) { return numerics::functions::LinkFunctionFactory::create(LT::Logit); }},
        {"Probit",
         [](const JsonValue&) { return numerics::functions::LinkFunctionFactory::create(LT::Probit); }},
        {"ComplementaryLogLog",
         [](const JsonValue&) {
             return numerics::functions::LinkFunctionFactory::create(LT::ComplementaryLogLog);
         }},
        {"FisherZ",
         [](const JsonValue&) { return numerics::functions::LinkFunctionFactory::create(LT::FisherZ); }},
        {"YeoJohnson",
         [](const JsonValue& spec) {
             return std::make_unique<numerics::functions::YeoJohnsonLink>(
                 link_param(spec, "lambda", 1.0));
         }},
        {"ASinH",
         [](const JsonValue& spec) {
             return std::make_unique<models::link_functions::ASinHLink>(
                 link_param(spec, "gamma0", 0.0), link_param(spec, "scale", 1.0),
                 link_param(spec, "epsilon", 0.0), link_param(spec, "delta", 1.0));
         }},
        {"SES",
         [](const JsonValue& spec) {
             return std::make_unique<models::link_functions::SESLink>(link_param(spec, "a", 1.0));
         }},
        {"LogSES",
         [](const JsonValue& spec) {
             return std::make_unique<models::link_functions::LogSESLink>(
                 link_param(spec, "sigma0", 1.0), link_param(spec, "a", 1.0),
                 link_param(spec, "lambda", 0.2));
         }},
        {"LogASinH",
         [](const JsonValue& spec) {
             return std::make_unique<models::link_functions::LogASinHLink>(
                 link_param(spec, "sigma0", 1.0), link_param(spec, "log_scale", 1.0),
                 link_param(spec, "epsilon", 0.0), link_param(spec, "delta", 1.0));
         }},
        {"Centered",
         [](const JsonValue& spec) {
             if (!spec.contains("inner"))
                 throw std::runtime_error("link type 'Centered' needs an 'inner' link spec");
             return std::make_unique<models::link_functions::CenteredLink>(
                 build_link(spec.at("inner")), link_param(spec, "mu0", 0.0),
                 link_param(spec, "scale", 1.0));
         }},
    };
    return table;
}

// Builds a link from the options spec, looking its `type` up in link_builder_table() -- the
// single source of truth this dispatch shares with the "names" toolbox method below.
inline std::unique_ptr<numerics::functions::ILinkFunction> build_link(const JsonValue& spec) {
    const std::string& type = spec.at("type").as_string();
    for (const auto& [name, builder] : link_builder_table())
        if (name == type) return builder(spec);
    throw std::runtime_error("unknown link type: " + type);
}

// The twelve link type names build_link() accepts, in table order. R's link_names() and
// Python's link_names() both call through to this via the "names" toolbox method rather than
// holding their own literal, so a type added here is automatically visible to both.
inline const std::vector<std::string>& link_type_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> out;
        for (const auto& [name, builder] : link_builder_table()) {
            (void)builder;
            out.push_back(name);
        }
        return out;
    }();
    return names;
}

inline ToolboxResult run_link(const std::string& method,
                              const std::vector<std::vector<double>>& data,
                              const JsonValue& options) {
    if (method == "names") {
        ToolboxResult r;
        r.names = link_type_names();
        return r;
    }
    if (!options.contains("link"))
        throw std::runtime_error("toolbox group 'link' needs a 'link' spec in its options");
    std::unique_ptr<numerics::functions::ILinkFunction> l = build_link(options.at("link"));
    const std::vector<double>& x = data_at(data, 0, "link", method);
    ToolboxResult r;
    for (double v : x) {
        if (method == "link") r.values.push_back(l->link(v));
        else if (method == "inverse_link") r.values.push_back(l->inverse_link(v));
        else if (method == "d_link") r.values.push_back(l->d_link(v));
        else throw std::runtime_error("unknown link method: " + method);
    }
    return r;
}

}  // namespace corehydro::numerics::support::detail
