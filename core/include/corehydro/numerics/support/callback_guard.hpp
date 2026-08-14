// corehydro ADDITION -- no upstream C# counterpart.
//
// The one place a host-language (R/Python) exception thrown inside a callback is latched so it
// survives a ported class's catch-all. Extracted from optimizer_runner.hpp (where it was
// GuardedObjective, serving Optimizer::minimize()'s catch-all) and generalized, because the
// callback surface has five more crossing points with different signatures: the MCMC
// log-likelihood, the Gibbs proposal, the HMC/NUTS gradient, the four bootstrap delegates, and
// the GMM moment conditions. Every one of them sits under ported code that may catch broadly,
// and a raw host exception must not be allowed to unwind a C++ frame it did not originate in.
//
// Contract: the FIRST throw is stored and an abort flag latches. Every later call returns the
// sentinel WITHOUT re-entering the host, so one run can never raise a second host exception (or
// a second R longjmp) once the first is captured. The caller rethrows after the ported algorithm
// returns, inside the same protected frame the callback was invoked from.
//
// USE IT IN PAIRS. `rethrow_if_aborted()` after the ported call is necessary but NOT sufficient:
// a sentinel value fed back into ported code can itself raise an INTERNAL C++ exception (a
// domain error, a "root is not bracketed" throw) that propagates before the latch is ever
// consulted, replacing the user's real message. Every drive site must therefore be wrapped:
//
//     try { /* run the ported algorithm */ } catch (...) { g.rethrow_if_aborted(); throw; }
//     g.rethrow_if_aborted();
//
// so the guard's stored host exception always wins over any internal exception it provoked.
#pragma once
#include <exception>
#include <functional>
#include <utility>

namespace corehydro::numerics::support {

template <typename TResult, typename... TArgs>
class GuardedCall {
   public:
    using Function = std::function<TResult(TArgs...)>;

    // `sentinel` is returned for every call after an abort. Each call site chooses a value its
    // ported consumer treats as unconditionally rejected: +/-infinity for an objective (see
    // optimizer_runner.hpp), -infinity for a log-likelihood, an empty vector for a proposal.
    GuardedCall(Function fn, TResult sentinel)
        : fn_(std::move(fn)), sentinel_(std::move(sentinel)) {}

    TResult operator()(TArgs... args) {
        if (aborted_) return sentinel_;
        try {
            TResult v = fn_(std::forward<TArgs>(args)...);
            ++call_count_;
            return v;
        } catch (...) {
            error_ = std::current_exception();
            aborted_ = true;
            return sentinel_;
        }
    }

    bool aborted() const { return aborted_; }
    void rethrow_if_aborted() const {
        if (error_) std::rethrow_exception(error_);
    }
    // Calls that actually completed in the host. Excludes short-circuited calls.
    int call_count() const { return call_count_; }
    explicit operator bool() const { return static_cast<bool>(fn_); }

   private:
    Function fn_;
    TResult sentinel_;
    bool aborted_ = false;
    std::exception_ptr error_;
    int call_count_ = 0;
};

}  // namespace corehydro::numerics::support
