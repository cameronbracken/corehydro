// Pins the C++ Sobol port to the canonical reference stream from Test_SobolSequence.cs.
// Oracle values are the published expected output verified against R's randtoolbox::sobol()
// in the upstream C# test (dimension 2, first 10 points).
// The path to the new-joe-kuo-6.21201 file is taken from argv[1] (set by CMake).
#include "corehydro/numerics/sampling/sobol.hpp"

#include <cstdlib>
#include <string>
#include <vector>

#include "check.hpp"

using corehydro::numerics::sampling::SobolSequence;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: test_sobol <path/to/new-joe-kuo-6.21201>\n");
        return 1;
    }
    const std::string path(argv[1]);

    // --- Reference stream: dimension 2, first 10 points (from C# Test_Sobol) --------
    // Verified against randtoolbox::sobol(10, dim=2) in R.
    // These are exact binary fractions so CHECK_EQ (exact equality) is appropriate.
    {
        const double expected[10][2] = {
            {0.5000, 0.5000},
            {0.7500, 0.2500},
            {0.2500, 0.7500},
            {0.3750, 0.3750},
            {0.8750, 0.8750},
            {0.6250, 0.1250},
            {0.1250, 0.6250},
            {0.1875, 0.3125},
            {0.6875, 0.8125},
            {0.9375, 0.0625},
        };

        SobolSequence sobol(2, path);
        for (int i = 0; i < 10; ++i) {
            auto pt = sobol.next_double();
            CHECK_NEAR(pt[0], expected[i][0], 0.0);
            CHECK_NEAR(pt[1], expected[i][1], 0.0);
        }
    }

    // --- Dimension 1: no file needed -------------------------------------------------
    {
        SobolSequence s1(1);
        auto pt = s1.next_double();
        CHECK_NEAR(pt[0], 0.5, 0.0);
    }

    // --- SkipTo: index 0 resets; index 1 == first NextDouble -------------------------
    {
        SobolSequence s_fresh(2, path);
        auto first = s_fresh.next_double();

        SobolSequence s_skip(2, path);
        auto skipped = s_skip.skip_to(1);

        CHECK_NEAR(skipped[0], first[0], 0.0);
        CHECK_NEAR(skipped[1], first[1], 0.0);
    }

    // --- SkipTo(5) should land on the 5th point (0-indexed: points[4]) --------------
    {
        // Advance a fresh sequence 5 steps.
        SobolSequence s_seq(2, path);
        std::vector<double> fifth;
        for (int i = 0; i < 5; ++i) fifth = s_seq.next_double();

        SobolSequence s_jump(2, path);
        auto jumped = s_jump.skip_to(5);

        CHECK_NEAR(jumped[0], fifth[0], 0.0);
        CHECK_NEAR(jumped[1], fifth[1], 0.0);
    }

    // --- Seeded linear scramble and digital shift -----------------------------------
    {
        SobolSequence first(2, path, 12345);
        SobolSequence second(2, path, 12345);
        const double expected[3][2] = {
            {0.659847889995655, 0.5825658737429362},
            {0.8092207866147765, 0.15954710391098081},
            {0.48233205686128877, 0.7857143836043143},
        };
        for (int i = 0; i < 3; ++i) {
            auto a = first.next_double();
            auto b = second.next_double();
            for (int j = 0; j < 2; ++j) {
                CHECK_EQ(a[j], b[j]);
                CHECK_EQ(a[j], expected[i][j]);
            }
        }
        CHECK_TRUE(first.seed().has_value());
        CHECK_EQ(*first.seed(), 12345);
        CHECK_TRUE(!SobolSequence(2, path).seed().has_value());
    }

    return chtest::summary("sobol");
}
