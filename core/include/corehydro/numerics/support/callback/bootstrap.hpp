// corehydro ADDITION -- no upstream C# counterpart. The bootstrap group of callback_runner.hpp:
// running the ported Bootstrap workflow against the user's own four delegates (resample, fit,
// statistic, jackknife). STUB -- the real implementation lands in Task 6.
#pragma once
#include <stdexcept>
#include <string>

#include "corehydro/numerics/support/callback/common.hpp"

namespace corehydro::numerics::support::detail {

inline CallbackResult run_bootstrap(const std::string&, const JsonValue&, const CallbackSet&) {
    throw std::invalid_argument("the bootstrap callback group lands in Task 6");
}

}  // namespace corehydro::numerics::support::detail
