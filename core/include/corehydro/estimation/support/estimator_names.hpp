// corehydro ADDITION -- no upstream C# counterpart (sibling of
// numerics/sampling/bootstrap/ci_method_names.hpp).
//
// The one place an estimator knob spelled as a STRING becomes the ported enum it names. Extracted
// from estimation/support/fit_runner.hpp when a second surface needed the same three functions:
// numerics/support/callback/gmm.hpp (fit_gmm_moments) accepts the same `optimizer` and `strategy`
// names fit_gmm() does, and including the whole of fit_runner.hpp -- which pulls in every model,
// every analysis and Bulletin17C -- to reach two small parsers would have been the wrong trade, so
// they moved down here where the callback group can include them alone.
//
// fit_runner.hpp includes this header and its callers are unchanged: the names stay in
// `corehydro::estimation::support`, spelled exactly as before. The third function that moved,
// `status_name`, went further down still -- it now lives beside the enum it names, in
// optimization/support/optimization_status.hpp, which is where integration_status.hpp already kept
// its own. fit_runner.hpp's one unqualified `status_name(...)` call reaches it by argument-dependent
// lookup; a forwarder here would only make that call ambiguous.
#pragma once
#include <stdexcept>
#include <string>

#include "corehydro/estimation/generalized_method_of_moments.hpp"
#include "corehydro/estimation/optimization_method.hpp"
#include "corehydro/numerics/math/optimization/support/optimization_status.hpp"

namespace corehydro::estimation::support {

// Optimizer name -> OptimizationMethod. Accepts the "MLSL" alias, matching the pre-phase-2
// glue (corehydror/src/estimation.cpp's parse_optimization_method). Throws naming the value
// it could not parse.
inline OptimizationMethod parse_optimizer(const std::string& s) {
    if (s == "Brent") return OptimizationMethod::Brent;
    if (s == "BFGS") return OptimizationMethod::BFGS;
    if (s == "NelderMead") return OptimizationMethod::NelderMead;
    if (s == "Powell") return OptimizationMethod::Powell;
    if (s == "DifferentialEvolution") return OptimizationMethod::DifferentialEvolution;
    if (s == "MultilevelSingleLinkage" || s == "MLSL")
        return OptimizationMethod::MultilevelSingleLinkage;
    throw std::runtime_error("unknown optimizer '" + s + "'");
}

// The GMM estimation-strategy knob (default Iterative, matching the C# GMM default). Mirrors
// corehydror/src/estimation.cpp's parse_gmm_strategy.
inline GeneralizedMethodOfMoments::GMMEstimationStrategy parse_gmm_strategy(const std::string& s) {
    using Strat = GeneralizedMethodOfMoments::GMMEstimationStrategy;
    if (s == "OneStep") return Strat::OneStep;
    if (s == "TwoStep") return Strat::TwoStep;
    if (s == "Iterative") return Strat::Iterative;
    throw std::runtime_error("unknown GMM estimation strategy '" + s + "'");
}

}  // namespace corehydro::estimation::support
