// ported from: Numerics/Sampling/Bootstrap/Support/PivotalBootstrapInvalidDrawPolicy.cs @ 2a0357a
//
// Specifies how invalid pivotal bootstrap draws are handled after the two-covariance
// transform. Used by the covariance-aware pivotal bootstrap workflow -- see
// bootstrap_fit.hpp's file header for the P3.10/P3.11 task split; this support type is ported
// ahead of the pivotal run methods themselves.
#pragma once

namespace corehydro::numerics::sampling {

// Specifies how invalid pivotal bootstrap draws are handled after the two-covariance transform.
enum class PivotalBootstrapInvalidDrawPolicy {
    // Drop invalid pivotal draws from the retained pivotal ensemble.
    Drop,

    // Replace invalid pivotal draws with the corresponding accepted raw bootstrap fit.
    UseRaw,

    // Replace invalid pivotal draws with the original parent fit.
    UseParent,
};

}  // namespace corehydro::numerics::sampling
