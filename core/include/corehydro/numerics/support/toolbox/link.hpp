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

#include <memory>
#include <stdexcept>
#include <string>
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

// String -> LinkFunctionType, the link-group analog of models::spec::parse_trend_model_type.
// No such helper exists on LinkFunctionFactory itself (it dispatches on the enum, not a name),
// so this is new.
inline numerics::functions::LinkFunctionType parse_link_function_type(const std::string& name) {
    using LT = numerics::functions::LinkFunctionType;
    if (name == "Identity") return LT::Identity;
    if (name == "Log") return LT::Log;
    if (name == "Logit") return LT::Logit;
    if (name == "Probit") return LT::Probit;
    if (name == "ComplementaryLogLog") return LT::ComplementaryLogLog;
    if (name == "YeoJohnson") return LT::YeoJohnson;
    if (name == "FisherZ") return LT::FisherZ;
    throw std::runtime_error("unknown link type: " + name);
}

// Builds a link from the options spec. `parameters` is a named object because the six
// parameterized links take different arguments and a positional array would be unreadable at
// the call site. Every constructor argument and default below is read off the real header
// (numerics/functions/yeo_johnson_link.hpp, models/link_functions/*.hpp) rather than guessed:
// notably CenteredLink takes a THIRD constructor argument, `scale` (default 1.0, the C#
// `CenteredLink.Scale` property), that a first draft of this spec omitted -- it is included
// here as `opt("scale", 1.0)` so a spec can actually reach a non-default Centered scale.
inline std::unique_ptr<numerics::functions::ILinkFunction> build_link(const JsonValue& spec) {
    const std::string& type = spec.at("type").as_string();
    const JsonValue* p = spec.contains("parameters") ? &spec.at("parameters") : nullptr;
    auto opt = [&](const char* key, double dflt) {
        return p && p->contains(key) ? p->at(key).as_double() : dflt;
    };
    if (type == "Identity" || type == "Log" || type == "Logit" || type == "Probit" ||
        type == "ComplementaryLogLog" || type == "FisherZ")
        return numerics::functions::LinkFunctionFactory::create(parse_link_function_type(type));
    if (type == "YeoJohnson")
        return std::make_unique<numerics::functions::YeoJohnsonLink>(opt("lambda", 1.0));
    if (type == "ASinH")
        return std::make_unique<models::link_functions::ASinHLink>(
            opt("gamma0", 0.0), opt("scale", 1.0), opt("epsilon", 0.0), opt("delta", 1.0));
    if (type == "SES")
        return std::make_unique<models::link_functions::SESLink>(opt("a", 1.0));
    if (type == "LogSES")
        return std::make_unique<models::link_functions::LogSESLink>(
            opt("sigma0", 1.0), opt("a", 1.0), opt("lambda", 0.2));
    if (type == "LogASinH")
        return std::make_unique<models::link_functions::LogASinHLink>(
            opt("sigma0", 1.0), opt("log_scale", 1.0), opt("epsilon", 0.0), opt("delta", 1.0));
    if (type == "Centered") {
        if (!spec.contains("inner"))
            throw std::runtime_error("link type 'Centered' needs an 'inner' link spec");
        return std::make_unique<models::link_functions::CenteredLink>(
            build_link(spec.at("inner")), opt("mu0", 0.0), opt("scale", 1.0));
    }
    throw std::runtime_error("unknown link type: " + type);
}

inline ToolboxResult run_link(const std::string& method,
                              const std::vector<std::vector<double>>& data,
                              const JsonValue& options) {
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
