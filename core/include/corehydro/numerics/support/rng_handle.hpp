// corehydro ADDITION -- no upstream C# counterpart (sibling of callback_guard.hpp).
//
// The one place the core's seeded generator is handed OUT to a host-language callback.
//
// Upstream's delegate signatures pass the generator by reference -- `Gibbs.Proposal(double[]
// parameters, Random prng)` and `Bootstrap<TData>`'s `Func<TData, ParameterSet, Random, TData>` --
// because a proposal or a resample must draw from the SAME stream the sampler seeded, not from a
// generator of its own. That contract has to survive the crossing into R and Python: if a user's
// proposal called `runif()` or `random.random()` instead, the seeded run would stop being
// reproducible, and it would stop being reproducible SILENTLY, and R and Python would quietly
// disagree. So the reference is surfaced to the host as a handle over this borrow.
//
// A BORROW, NOT AN OWNER. The generator is owned by the C++ frame driving the callback (the MCMC
// sampler, the bootstrap loop, the probe below). The handle only points at it, which is exactly
// what makes it dangerous in a garbage-collected host: an R external pointer or a Python object
// outlives the C++ frame that created it, so a user who writes
//
//     saved <- NULL
//     proposal <- function(theta, rng) { saved <<- rng; ... }
//
// still holds a handle after the sampler returns, and dereferencing the raw pointer then would be
// a read of a destroyed object -- an undefined-behaviour crash inside a package headed for CRAN.
// The `valid` flag closes that: RngScope's destructor clears it when the C++ frame unwinds, and
// every draw checks it first, so the stored handle raises a plain, explanatory error instead.
//
// The flag lives in a shared_ptr so the host's copy and the scope's copy are the same flag. The
// host may hold its copy forever; only the flag's value decides whether a draw is allowed, so
// keeping the handle alive can never keep a dead pointer usable.
//
// WHY THE DRAWS LIVE HERE rather than in each binding: `uniform()` and `integers()` are the whole
// cross-language guarantee. Written twice they could drift (one calling next_double, the other
// next()); written once, an R draw and a Python draw are literally the same instructions on the
// same state, and the bindings are left with nothing but type conversion.
#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::support {

// The message every binding raises for an expired handle. Defined once so the R error and the
// Python RuntimeError are the same sentence, and so a test may match on it in either language.
inline const char* rng_handle_expired_message() {
    return "this random number generator handle is no longer valid; it can only be used inside "
           "the callback it was given to";
}

// The borrowed generator plus the flag saying whether the borrow is still live. Held by
// shared_ptr: one copy inside RngScope, one inside the host-language handle object.
struct RngBorrow {
    corehydro::numerics::sampling::MersenneTwister* prng = nullptr;
    bool valid = false;

    // The single gate every draw goes through. Checks the flag BEFORE the pointer is dereferenced,
    // which is the whole point -- once the scope is gone the pointer may name freed storage, so it
    // must never be read again, not even to test it.
    corehydro::numerics::sampling::MersenneTwister& require() const {
        if (!valid || prng == nullptr) throw std::runtime_error(rng_handle_expired_message());
        return *prng;
    }

    // `n` draws on [0, 1), the ported C# `MersenneTwister.NextDouble()` (GenRandReal2).
    std::vector<double> uniform(int n) const {
        auto& prng_ref = require();
        check_count(n);
        std::vector<double> out;
        out.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) out.push_back(prng_ref.next_double());
        return out;
    }

    // `n` draws on [min_inclusive, max_exclusive), the ported C# `MersenneTwister.Next(int
    // minInclusive, int maxExclusive)`. The upper bound is EXCLUSIVE because that is the C#
    // contract a ported proposal or resample is written against; both bindings say so by name.
    std::vector<int> integers(int n, int min_inclusive, int max_exclusive) const {
        auto& prng_ref = require();
        check_count(n);
        if (min_inclusive >= max_exclusive)
            throw std::invalid_argument(
                "`min` must be below `max` (the upper bound is exclusive)");
        check_span(min_inclusive, max_exclusive);
        std::vector<int> out;
        out.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) out.push_back(prng_ref.next(min_inclusive, max_exclusive));
        return out;
    }

   private:
    static void check_count(int n) {
        if (n < 1) throw std::invalid_argument("`n` must be a single positive whole number");
    }

    // WHAT C# DOES WITH TOO WIDE A SPAN. `MersenneTwister.Next(minInclusive, maxExclusive)`
    // forwards to `Next(maxExclusive - minInclusive)`, and C# computes that subtraction
    // UNCHECKED: any span above int.MaxValue wraps to a negative int, and `Next(int)` then throws
    // ArgumentOutOfRangeException("Must be positive."). So C# THROWS for such a call -- it never
    // returns a wrapped draw. We mirror the throw, and say why in a message a user of the two
    // packages can act on.
    //
    // The subtraction itself is done in int64 deliberately. Written as C# writes it, `max - min`
    // is signed integer overflow, which C++ leaves UNDEFINED rather than wrapping: before this
    // check, `integers(1, -2000000000, 2000000000)` returned an arbitrary out-of-range value
    // (-208904155) in both R and Python instead of raising.
    static void check_span(int min_inclusive, int max_exclusive) {
        constexpr std::int64_t kMaxSpan = 2147483647;  // int32 max, the widest Next(int) accepts
        const std::int64_t span =
            static_cast<std::int64_t>(max_exclusive) - static_cast<std::int64_t>(min_inclusive);
        if (span > kMaxSpan)
            throw std::invalid_argument(
                "the range `min` to `max` is too wide; `max` - `min` must be at most 2147483647");
    }
};

using RngBorrowPtr = std::shared_ptr<RngBorrow>;

// Makes a borrow live for exactly one C++ scope. Construct it around the frame that owns the
// generator, hand `handle()` to the host, and the destructor invalidates every copy the host kept.
// Non-copyable and non-movable: two scopes over one borrow would let the first destructor
// invalidate a borrow the second still considers live.
class RngScope {
   public:
    explicit RngScope(corehydro::numerics::sampling::MersenneTwister& prng)
        : borrow_(std::make_shared<RngBorrow>()) {
        borrow_->prng = &prng;
        borrow_->valid = true;
    }

    ~RngScope() {
        borrow_->valid = false;
        borrow_->prng = nullptr;
    }

    RngScope(const RngScope&) = delete;
    RngScope& operator=(const RngScope&) = delete;
    RngScope(RngScope&&) = delete;
    RngScope& operator=(RngScope&&) = delete;

    const RngBorrowPtr& handle() const { return borrow_; }

   private:
    RngBorrowPtr borrow_;
};

}  // namespace corehydro::numerics::support
