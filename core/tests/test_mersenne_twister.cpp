// Pins the C++ MT19937 port to the canonical mt19937ar reference stream.
// Oracle values are the published reference outputs of mt19937ar.c (Matsumoto &
// Nishimura), which the upstream C# MersenneTwister reproduces by construction.
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

#include <cstdint>
#include <vector>

#include "check.hpp"

using corehydro::numerics::sampling::MersenneTwister;

int main() {
    // --- init_by_array({0x123,0x234,0x345,0x456}): canonical first 10 outputs ---
    {
        std::vector<std::uint32_t> key = {0x123, 0x234, 0x345, 0x456};
        MersenneTwister mt(key);
        const std::uint32_t expected[10] = {
            1067595299u,  955945823u,  477289528u,  4107218783u, 4228976476u,
            3344332714u,  3355579695u, 227628506u,  810200273u,  2591290167u};
        for (int i = 0; i < 10; ++i) CHECK_EQ(mt.gen_rand_int32(), expected[i]);
    }

    // --- init_genrand(5489): canonical default-seed first 5 outputs ---
    {
        MersenneTwister mt(5489u);
        const std::uint32_t expected[5] = {3499211612u, 581869302u, 3890346734u,
                                           3586334585u, 545404204u};
        for (int i = 0; i < 5; ++i) CHECK_EQ(mt.gen_rand_int32(), expected[i]);
    }

    // --- next_double() == gen_rand_int32() / 2^32 (matches C# GenRandReal2) ---
    {
        std::vector<std::uint32_t> key = {0x123, 0x234, 0x345, 0x456};
        MersenneTwister a(key);
        MersenneTwister b(key);
        double expected = a.gen_rand_int32() * (1.0 / 4294967296.0);
        CHECK_NEAR(b.next_double(), expected, 0.0);
    }

    // Constructing clock-seeded instances advances only the process-wide clock-seed counter.
    // It cannot alter an explicitly seeded stream.
    {
        MersenneTwister before(12345u);
        MersenneTwister unseeded_one;
        MersenneTwister unseeded_two;
        (void)unseeded_one;
        (void)unseeded_two;
        MersenneTwister after(12345u);
        for (int i = 0; i < 20; ++i)
            CHECK_EQ(before.gen_rand_int32(), after.gen_rand_int32());
    }

    return chtest::summary("mersenne_twister");
}
