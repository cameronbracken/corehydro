// ported from: Numerics/Sampling/Bootstrap/Support/PivotalBootstrapDiagnostics.cs @ 2a0357a
//
// Stores diagnostic counts from the most recent pivotal bootstrap run. Used by the
// covariance-aware pivotal bootstrap workflow, which is now ported in full -- see
// `bootstrap.hpp`'s file header for the documentation of record.
//
// Documented deviation: the two C# `TimeSpan` timing fields (`ResamplingTime`,
// `TransformationTime`) are dropped -- wall-clock duration is not oracle-comparable across
// C#/C++/R/Python runs, so this port carries only the six deterministic replicate counts.
#pragma once

namespace corehydro::numerics::sampling {

// Stores diagnostic counts from the most recent pivotal bootstrap run.
struct PivotalBootstrapDiagnostics {
    // The number of raw bootstrap replicates requested or supplied.
    int requested_replicates = 0;

    // The number of raw bootstrap fits rejected before transformation.
    int rejected_raw_replicates = 0;

    // The number of raw bootstrap fits that failed after all retries.
    int failed_raw_replicates = 0;

    // The number of raw bootstrap fits accepted for transformation.
    int accepted_raw_replicates = 0;

    // The number of invalid pivotal draws encountered after raw-fit acceptance.
    int invalid_pivotal_replicates = 0;

    // The number of pivotal draws retained after invalid-draw handling.
    int retained_pivotal_replicates = 0;
};

}  // namespace corehydro::numerics::sampling
