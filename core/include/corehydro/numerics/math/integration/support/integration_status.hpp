// ported from: Numerics/Mathematics/Integration/Support/IntegrationStatus.cs @ 2a0357a
//
// Enumeration of integration statuses. Verbatim port; member order mirrors the C# source, and
// `status_name` writes the C# member names so a fixture, a binding, and the dotnet oracle emitter
// all compare the same strings (the counterpart of optim_status_name in optimizer_runner.hpp).
//
// NAMESPACE NOTE: the C# enum sits in `Numerics.Mathematics`, one level above the
// `Numerics.Mathematics.Integration` namespace its own file lives under, while every consumer of
// it is an integrator. This port puts it in `corehydro::numerics::math::integration` beside
// `Integrator`, matching how OptimizationStatus sits beside Optimizer. Nothing else in the port
// reaches for it, so the shortened namespace costs no fidelity.
#pragma once
#include <string>

namespace corehydro::numerics::math::integration {

enum class IntegrationStatus {
    // Integration has not been performed yet.
    None,

    // The integration ended successfully.
    Success,

    // The integration was stopped because the maximum number of iterations was reached.
    MaximumIterationsReached,

    // The integration was stopped because the maximum number of objective function evaluations
    // was reached.
    MaximumFunctionEvaluationsReached,

    // The integration was stopped due to internal failure.
    Failure,
};

// The C# member name for a status, for the fixture/binding surface.
inline std::string status_name(IntegrationStatus s) {
    switch (s) {
        case IntegrationStatus::None: return "None";
        case IntegrationStatus::Success: return "Success";
        case IntegrationStatus::MaximumIterationsReached: return "MaximumIterationsReached";
        case IntegrationStatus::MaximumFunctionEvaluationsReached:
            return "MaximumFunctionEvaluationsReached";
        default: return "Failure";
    }
}

}  // namespace corehydro::numerics::math::integration
