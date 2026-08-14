// corehydro ADDITION -- no upstream C# counterpart. The gmm group of callback_runner.hpp:
// running GeneralizedMethodOfMoments against the user's own moment-condition delegates.
// STUB -- the real implementation lands in Task 7.
#pragma once
#include <stdexcept>
#include <string>

#include "corehydro/numerics/support/callback/common.hpp"

namespace corehydro::numerics::support::detail {

inline CallbackResult run_gmm(const std::string&, const JsonValue&, const CallbackSet&) {
    throw std::invalid_argument("the gmm callback group lands in Task 7");
}

}  // namespace corehydro::numerics::support::detail
