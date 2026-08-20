// corehydro ADDITION -- no upstream C# counterpart (a C# caller writes the enum member itself;
// only a JSON-driven caller has to parse a name).
//
// The one place a `BootstrapCIMethod` is spelled out as a string. Four C++ callers need exactly
// this mapping and each had grown its own copy -- the cpp11 glue (corehydror/src/bootstrap.cpp),
// the pybind11 glue (corehydropy/src/bindings/bootstrap.cpp), the C++ fixture runner
// (core/tests/test_fixtures.cpp) and, from Task 6, the bootstrap callback group
// (numerics/support/callback/bootstrap.hpp). Three copies with three different messages is how a
// fifth member gets added to the enum and reaches only some of them, so the mapping lives here and
// the callers convert the ONE exception it throws into whatever their host language wants.
//
// (The dotnet oracle emitter keeps its own `ParseBootstrapCIMethod`, being C#; it is the fifth
// caller of the same table and cannot share a C++ header.)
#pragma once
#include <stdexcept>
#include <string>

#include "corehydro/numerics/sampling/bootstrap/bootstrap_results.hpp"

namespace corehydro::numerics::sampling {

// Maps a confidence-interval method name to its enum member, throwing `std::invalid_argument`
// naming the offending string for anything else.
inline BootstrapCIMethod parse_bootstrap_ci_method(const std::string& name) {
    if (name == "Percentile") return BootstrapCIMethod::Percentile;
    if (name == "BiasCorrected") return BootstrapCIMethod::BiasCorrected;
    if (name == "BCa") return BootstrapCIMethod::BCa;
    if (name == "Normal") return BootstrapCIMethod::Normal;
    if (name == "BootstrapT") return BootstrapCIMethod::BootstrapT;
    throw std::invalid_argument("unknown bootstrap ci_method: " + name);
}

// The inverse, so a result can report the method it was computed with in the same vocabulary.
inline const char* bootstrap_ci_method_name(BootstrapCIMethod method) {
    switch (method) {
        case BootstrapCIMethod::Percentile: return "Percentile";
        case BootstrapCIMethod::BiasCorrected: return "BiasCorrected";
        case BootstrapCIMethod::BCa: return "BCa";
        case BootstrapCIMethod::Normal: return "Normal";
        case BootstrapCIMethod::BootstrapT: return "BootstrapT";
    }
    return "Percentile";  // unreachable: every member is handled above
}

}  // namespace corehydro::numerics::sampling
