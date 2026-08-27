// ported from: Numerics/Sampling/MersenneTwister.cs @ 2a0357a
//
// MT19937 PRNG (Matsumoto & Nishimura). Verbatim port of the C# MersenneTwister,
// itself a faithful port of the reference mt19937ar.c. The uint32 state machine is
// kept bit-exact so R and Python (calling this same compiled code) produce identical
// random streams from a given seed. All arithmetic is unsigned 32-bit (defined
// wraparound), mirroring C# `uint`.
//
// next()/next(max_exclusive) below extend the Phase 0 port with the integer-draw
// rejection loops (C# System.Random.Next()/Next(int)) that LatinHypercube's shuffle
// and per-column reseeding consume; they draw from the same gen_rand_int32() stream in
// the same order as C#, so seeded call sequences stay bit-exact across languages.
//
// next(min_inclusive, max_exclusive) (Phase 3) adds the two-argument C# `Next(int
// minInclusive, int maxExclusive)` -- a one-line forwarder to the single-argument
// overload, `Next(maxExclusive - minInclusive) + minInclusive` -- that DEMCzs's
// SnookerUpdate (chain-index sampling) and other MCMC samplers consume.
//
// The parameterless ctor (P3.10) ports C#'s clock-seeded `MersenneTwister()`
// (`Initialize((uint)DateTime.UtcNow.Ticks)`), needed by
// `UnivariateDistributionBase::generate_random_values`'s `seed <= 0` branch (Bootstrap's
// "normal_quantiles" model registry entry always passes a positive seed drawn from another
// MT stream, so this branch is unreachable from any fixture -- it exists purely for API
// parity/completeness with the ported virtual method). Non-deterministic by design (matches
// C#); never exercised by an oracle-fixture assertion.
//
// next_doubles(n) (P2/Task 5) ports C#'s `NextDoubles(this Random random, int length)` extension
// method (Numerics/Utilities/ExtensionMethods.cs -- not MersenneTwister.cs itself; C++ has no
// extension methods, so it lands as a member here instead). The C# body is a simple loop over
// `random.NextDouble()`, and this mirrors that exactly, so the stream order matches C# call for
// call. Consumed by Vegas's per-sample draw when `use_sobol_sequence` is false (the 2-D overload,
// `NextDoubles(this Random random, int length, int dimension)`, spins up one side PRNG per
// dimension and is not needed by anything ported so far -- not added here).
#pragma once
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace corehydro::numerics::sampling {

class MersenneTwister {
   public:
    // Clock-seeded default ctor (mirrors C# `MersenneTwister()`). See file header.
    MersenneTwister() {
        auto ticks = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
        initialize(static_cast<std::uint32_t>(ticks));
    }

    explicit MersenneTwister(std::uint32_t seed) { initialize(seed); }

    explicit MersenneTwister(const std::vector<std::uint32_t>& seeds) {
        initialize_by_array(seeds);
    }

    // Initialize with a single seed (reference init_genrand).
    void initialize(std::uint32_t seed) {
        mt_[0] = seed & 0xffffffffU;
        for (mti_ = 1; mti_ < N; ++mti_) {
            mt_[mti_] =
                (1812433253U * (mt_[mti_ - 1] ^ (mt_[mti_ - 1] >> 30)) +
                 static_cast<std::uint32_t>(mti_));
            mt_[mti_] &= 0xffffffffU;
        }
    }

    // Generates a random number on [0, 0xffffffff].
    std::uint32_t gen_rand_int32() {
        std::uint32_t y;
        static const std::uint32_t mag01[2] = {0x0U, kMatrixA};

        if (mti_ >= N) {
            int kk;
            if (mti_ == N + 1) initialize(5489U);  // default seed if uninitialized

            for (kk = 0; kk < N - M; ++kk) {
                y = (mt_[kk] & kUpperMask) | (mt_[kk + 1] & kLowerMask);
                mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 0x1];
            }
            for (; kk < N - 1; ++kk) {
                y = (mt_[kk] & kUpperMask) | (mt_[kk + 1] & kLowerMask);
                mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1];
            }
            y = (mt_[N - 1] & kUpperMask) | (mt_[0] & kLowerMask);
            mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 0x1];

            mti_ = 0;
        }

        y = mt_[mti_++];

        // Tempering
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= y >> 18;
        return y;
    }

    // Generates a random number on [0, 1) (matches C# GenRandReal2 / NextDouble).
    double next_double() { return gen_rand_int32() * (1.0 / 4294967296.0); }

    // Returns a vector of `length` random doubles on [0, 1) (matches C# `Random.NextDoubles(int
    // length)`; see the file header). A simple loop over next_double(), same stream order as C#.
    std::vector<double> next_doubles(int length) {
        std::vector<double> values(static_cast<std::size_t>(length));
        for (int i = 0; i < length; ++i) values[static_cast<std::size_t>(i)] = next_double();
        return values;
    }

    // Generates a non-negative random int on [0, int32::max], rejecting int32::max so the
    // result is uniform over [0, int32::max) (matches C# Next(); GenRandInt31() is folded
    // in here since nothing else in this port needs it standalone).
    int next() {
        int result;
        do {
            result = static_cast<int>(gen_rand_int32() >> 1);
        } while (result == kIntMax);
        return result;
    }

    // Generates a random int on [0, max_exclusive) via uint threshold rejection so every
    // remainder class is equally likely (matches C# Next(int maxExclusive)).
    int next(int max_exclusive) {
        if (max_exclusive <= 0) throw std::invalid_argument("max_exclusive must be positive.");

        std::uint32_t max_u = static_cast<std::uint32_t>(max_exclusive);
        std::uint32_t threshold = kUint32Max - (kUint32Max % max_u);

        std::uint32_t r;
        do {
            r = gen_rand_int32();
        } while (r >= threshold);

        return static_cast<int>(r % max_u);
    }

    // Generates a random int on [min_inclusive, max_exclusive) (matches C# Next(int
    // minInclusive, int maxExclusive), a one-line forwarder to next(max_exclusive -
    // min_inclusive) + min_inclusive).
    int next(int min_inclusive, int max_exclusive) {
        if (min_inclusive >= max_exclusive)
            throw std::invalid_argument("min_inclusive must be less than max_exclusive.");
        // C# computes `maxExclusive - minInclusive` unchecked, so a span wider than int.MaxValue
        // wraps negative and Next(int) throws "Must be positive." The same expression in C++ is
        // signed integer overflow -- undefined, not wrapping -- so the span is taken in int64 and
        // the C# throw is reproduced directly. Behaviour below the boundary is unchanged.
        const std::int64_t span =
            static_cast<std::int64_t>(max_exclusive) - static_cast<std::int64_t>(min_inclusive);
        if (span > static_cast<std::int64_t>(kIntMax))
            throw std::invalid_argument("max_exclusive must be positive.");
        return next(static_cast<int>(span)) + min_inclusive;
    }

   private:
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr std::uint32_t kMatrixA = 0x9908b0dfU;
    static constexpr std::uint32_t kUpperMask = 0x80000000U;
    static constexpr std::uint32_t kLowerMask = 0x7fffffffU;
    static constexpr int kIntMax = std::numeric_limits<int>::max();  // C# int.MaxValue
    static constexpr std::uint32_t kUint32Max = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t mt_[N];
    int mti_ = N + 1;  // mti_ == N+1 means mt_ is not initialized

    // Initialize by an array of seeds (reference init_by_array).
    void initialize_by_array(const std::vector<std::uint32_t>& init_key) {
        const int key_length = static_cast<int>(init_key.size());
        initialize(19650218U);
        int i = 1, j = 0;
        int k = (N > key_length) ? N : key_length;
        for (; k > 0; --k) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525U)) +
                     init_key[j] + static_cast<std::uint32_t>(j);
            mt_[i] &= 0xffffffffU;
            ++i;
            ++j;
            if (i >= N) {
                mt_[0] = mt_[N - 1];
                i = 1;
            }
            if (j >= key_length) j = 0;
        }
        for (k = N - 1; k > 0; --k) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941U)) -
                     static_cast<std::uint32_t>(i);
            mt_[i] &= 0xffffffffU;
            ++i;
            if (i >= N) {
                mt_[0] = mt_[N - 1];
                i = 1;
            }
        }
        mt_[0] = kUpperMask;  // MSB is 1, assuring a non-zero initial array
    }
};

}  // namespace corehydro::numerics::sampling
