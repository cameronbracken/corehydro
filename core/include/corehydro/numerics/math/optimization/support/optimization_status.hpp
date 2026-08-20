// ported from: Numerics/Mathematics/Optimization/Support/OptimizationStatus.cs @ 2a0357a
//
// Enumeration of optimization statuses. Verbatim port; member order mirrors the C# source, and
// `status_name` (a corehydro addition, no C# counterpart) writes the C# member names so a fixture,
// a binding, and the dotnet oracle emitter all compare the same strings -- exactly as
// integration/support/integration_status.hpp carries its own. It lives here rather than in one
// runner because THREE surfaces report an optimizer status to a caller: optimizer_runner.hpp
// (optim_minimize), estimation/support/fit_runner.hpp (fit_mle/fit_map/fit_gmm) and
// numerics/support/callback/gmm.hpp (fit_gmm_moments); each of the first two carried its own
// identical copy of the switch until the third would have made a fourth.
#pragma once
#include <string>

namespace corehydro::numerics::math::optimization {

enum class OptimizationStatus {
    // Optimization has not been performed yet.
    None,

    // The optimization method ended successfully.
    Success,

    // The optimization method was stopped because the maximum number of iterations was
    // reached.
    MaximumIterationsReached,

    // The optimization method was stopped because the maximum number of objective
    // function evaluations was reached.
    MaximumFunctionEvaluationsReached,

    // The optimization method was stopped due to internal failure.
    Failure,
};

// The C# member name for a status, for the fixture/binding surface. Every status other than the
// four named ones reports "Failure", matching what both of the copies this replaced did.
inline std::string status_name(OptimizationStatus s) {
    switch (s) {
        case OptimizationStatus::None: return "None";
        case OptimizationStatus::Success: return "Success";
        case OptimizationStatus::MaximumIterationsReached: return "MaximumIterationsReached";
        case OptimizationStatus::MaximumFunctionEvaluationsReached:
            return "MaximumFunctionEvaluationsReached";
        default: return "Failure";
    }
}

}  // namespace corehydro::numerics::math::optimization
