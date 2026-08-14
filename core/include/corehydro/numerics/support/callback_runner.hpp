// corehydro ADDITION -- no upstream C# counterpart (sibling of toolbox_runner.hpp,
// optimizer_runner.hpp, distributions/support/dist_runner.hpp, and
// estimation/support/fit_runner.hpp).
//
// The single place a ported class runs against a live host-language callback other than an
// optimizer objective. Upstream's own API for these classes IS delegate-driven -- see
// MCMCSampler.cs, Gibbs.cs, HMC.cs, Bootstrap.cs, and GeneralizedMethodOfMoments.cs -- so this
// runner surfaces the real C# contract; the model registries under sampling/mcmc/ and
// sampling/bootstrap/ are corehydro fixture scaffolding, not the upstream shape.
//
// Four callers drive it and none owns any evaluation logic: the cpp11 glue
// (corehydror/src/callback.cpp), the pybind11 glue (corehydropy/src/bindings/callback.cpp), the
// C++ fixture runner (core/tests/test_fixtures.cpp), and the dotnet oracle emitter, which
// serializes the identical options grammar and dispatches equivalent C# delegates.
//
// Split of responsibilities, matching toolbox_runner.hpp: serializable configuration (sampler
// name, iteration counts, seed, prior specs, CI method) travels in `options_json`; bulk data
// travels as native double vectors; the callbacks arrive as std::function members of a
// CallbackSet. No callback registry lives here -- every caller supplies its own closures, so
// every fixture case exercises the real host callback path, which is the thing this file exists
// to make safe.
#pragma once
#include <stdexcept>
#include <string>

#include "corehydro/numerics/support/callback/bootstrap.hpp"
#include "corehydro/numerics/support/callback/common.hpp"
#include "corehydro/numerics/support/callback/gmm.hpp"
#include "corehydro/numerics/support/callback/math.hpp"
#include "corehydro/numerics/support/callback/mcmc.hpp"

namespace corehydro::numerics::support {

// Runs `method` of `group` (one of "math", "mcmc", "bootstrap", "gmm") against the callbacks in
// `cbs`, configured by `options_json`, and returns a flat CallbackResult. Each group header
// documents its own options grammar and which CallbackSet members it requires.
inline CallbackResult run_callback(const std::string& group, const std::string& method,
                                   const std::string& options_json, const CallbackSet& cbs) {
    JsonValue o = detail::parse_options(options_json);
    if (group == "math") return detail::run_math(method, o, cbs);
    if (group == "mcmc") return detail::run_mcmc(method, o, cbs);
    if (group == "bootstrap") return detail::run_bootstrap(method, o, cbs);
    if (group == "gmm") return detail::run_gmm(method, o, cbs);
    throw std::invalid_argument("unknown callback group: " + group);
}

}  // namespace corehydro::numerics::support
