// corehydro ADDITION -- no upstream C# counterpart. The mcmc group of callback_runner.hpp:
// running the eight ported samplers against a user-written log-likelihood (plus the Gibbs
// proposal and the HMC/NUTS gradient). STUB -- the real implementation lands in Task 4.
#pragma once
#include <stdexcept>
#include <string>

#include "corehydro/numerics/support/callback/common.hpp"

namespace corehydro::numerics::support::detail {

inline CallbackResult run_mcmc(const std::string&, const JsonValue&, const CallbackSet&) {
    throw std::invalid_argument("the mcmc callback group lands in Task 4");
}

}  // namespace corehydro::numerics::support::detail
