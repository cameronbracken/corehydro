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
// Contract: the FIRST throw is stored and an abort flag latches. Every later call THROUGH A GUARD
// OBSERVING THAT FLAG returns the sentinel WITHOUT re-entering the host. The caller rethrows after
// the ported algorithm returns, inside the same protected frame the callback was invoked from.
//
// WHICH GUARDS OBSERVE THE FLAG is the whole of the contract, and it is a choice the caller makes:
//
//   - A default-constructed guard owns a private abort state. It protects ITS OWN callback only.
//     Two independently constructed guards know nothing of each other.
//   - Guards constructed with the same `CallbackAbortStatePtr` share one abort state. A latch in
//     any one of them short-circuits ALL of them, and the FIRST exception latched is the one
//     stored and rethrown -- a later latch never overwrites it. `rethrow_if_aborted()` on any
//     guard in the group rethrows that first exception.
//
// USE A SHARED STATE WHENEVER A GROUP HAS MORE THAN ONE CALLBACK LIVE AT ONCE (the bootstrap
// group's four delegates; the Gibbs proposal beside the log-likelihood). With private states, an
// R error raised inside the log-likelihood latches only that guard, and the ported algorithm --
// which does not know it is unwinding -- goes on to call the proposal, re-entering R while an
// unwind is already pending. That is precisely the failure this class exists to prevent, so the
// per-guard flag is not enough on its own. Each guard still carries its OWN sentinel, because the
// right "worst value" differs by callback: +/-infinity for an objective, -infinity for a
// log-likelihood, an empty vector for a proposal.
//
// Single-threaded by construction, and deliberately unsynchronized: an R or Python callback can
// only be entered from one thread at a time (R is single-threaded; CPython holds the GIL), and
// every ported algorithm driving these guards runs its callback serially.
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
#include <memory>
#include <utility>

namespace corehydro::numerics::support {

// The latch itself, held privately by one guard or shared by several. Kept a separate type (rather
// than a flag templated into GuardedCall) precisely because the guards sharing it have different
// TResult: the bootstrap group's resample returns a vector and its statistic returns a scalar, and
// they still have to abort as one.
struct CallbackAbortState {
    // Stores `e` only if nothing has latched yet, so the FIRST host exception is the one that
    // reaches the caller. A later latch is dropped, not chained: the first one is the cause, and
    // any later throw is at best a consequence of running on after it.
    void latch(std::exception_ptr e) {
        if (aborted) return;
        aborted = true;
        error = std::move(e);
    }

    bool aborted = false;
    std::exception_ptr error;
};

using CallbackAbortStatePtr = std::shared_ptr<CallbackAbortState>;

// Builds the state a group hands to every guard it constructs.
inline CallbackAbortStatePtr make_abort_state() { return std::make_shared<CallbackAbortState>(); }

// Rethrows the first exception latched into a SHARED abort state directly, without going through
// any one guard. Asking a guard's own `rethrow_if_aborted()` works only because every guard in a
// shared-state group reads the same `error` -- correct today, but it makes correctness depend on
// WHICH guard happens to be asked, a trap for the next guard added to a group. A drive site with
// more than one live guard over one shared state (mcmc's log-likelihood/proposal/gradient trio, for
// instance) should call this on the state itself instead of picking one guard to stand in for the
// group. A null `state` has nothing latched, so it rethrows nothing.
inline void rethrow_if_aborted(const CallbackAbortStatePtr& state) {
    if (state && state->error) std::rethrow_exception(state->error);
}

template <typename TResult, typename... TArgs>
class GuardedCall {
   public:
    using Function = std::function<TResult(TArgs...)>;

    // `sentinel` is returned for every call after an abort. Each call site chooses a value its
    // ported consumer treats as unconditionally rejected: +/-infinity for an objective (see
    // optimizer_runner.hpp), -infinity for a log-likelihood, an empty vector for a proposal.
    //
    // The two-argument form gives this guard a private abort state (the standalone case: the math
    // group, the optimizer surface). The three-argument form joins the guards sharing `state`, so
    // a latch in any of them short-circuits this one too. A null `state` is treated as the
    // standalone case rather than dereferenced.
    GuardedCall(Function fn, TResult sentinel)
        : fn_(std::move(fn)), sentinel_(std::move(sentinel)), state_(make_abort_state()) {}

    GuardedCall(Function fn, TResult sentinel, CallbackAbortStatePtr state)
        : fn_(std::move(fn)),
          sentinel_(std::move(sentinel)),
          state_(state ? std::move(state) : make_abort_state()) {}

    TResult operator()(TArgs... args) {
        if (state_->aborted) return sentinel_;
        try {
            TResult v = fn_(std::forward<TArgs>(args)...);
            ++call_count_;
            return v;
        } catch (...) {
            state_->latch(std::current_exception());
            return sentinel_;
        }
    }

    // True once ANY guard sharing this guard's state has latched.
    bool aborted() const { return state_->aborted; }
    // Rethrows the first exception latched by any guard sharing this guard's state.
    void rethrow_if_aborted() const {
        if (state_->error) std::rethrow_exception(state_->error);
    }
    // Calls that actually completed in THIS guard's host function. Excludes short-circuited calls.
    int call_count() const { return call_count_; }
    // The state to hand the group's other guards.
    const CallbackAbortStatePtr& abort_state() const { return state_; }
    explicit operator bool() const { return static_cast<bool>(fn_); }

   private:
    Function fn_;
    TResult sentinel_;
    CallbackAbortStatePtr state_;
    int call_count_ = 0;
};

}  // namespace corehydro::numerics::support
