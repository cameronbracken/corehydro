// C++-only ctest guarding `data::linear_moments` against the int32 overflow in its weight
// numerators.
//
// C# `Statistics.LinearMoments` (Numerics/Data/Statistics/Statistics.cs ~509-520) forms
// `(i-2)*(i-1)` and `(i-3)*(i-2)*(i-1)` in `int`. The triple product first exceeds int32 at
// i = 1293 (1290*1291*1292 = 2,151,683,880 > 2^31-1), and C#'s default unchecked context wraps
// silently. UBSan reports the ported C++ expression as `signed integer overflow: 1665390 * 1292
// cannot be represented in type 'int'` -- undefined behaviour, and the kind of finding CRAN's
// sanitizer run rejects a package for.
//
// The real C# library was driven at these sizes to confirm it wraps rather than throws: for the
// 1293-point arithmetic sequence below it returns T4 = -0.18525251648817065, and at 1300 points
// T4 = -1.446418581370934, where the L-kurtosis of an evenly spaced sample is 0. The port now
// forms the products in `double`, which is bit-identical to the C# for every sample under the
// overflow and mathematically correct above it -- a DELIBERATE DIVERGENCE, recorded in
// docs/upstream-csharp-issues.md.
//
// Revert the `statistics.hpp` fix and the T3/T4 checks below fail for n >= 1293.
//
// No fixture equivalent: no shipped fixture carries a sample anywhere near 1293 points, and
// pinning one would pin the wrapped C# answer this test exists to reject.
#include <cmath>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "check.hpp"

using corehydro::numerics::data::linear_moments;

namespace {

// An evenly spaced sample: L-skewness and L-kurtosis are both 0 for any length.
std::vector<double> arithmetic_sequence(int n) {
    std::vector<double> d;
    d.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) d.push_back(1.0 + i * 0.5);
    return d;
}

}  // namespace

int main() {
    // 1292 is the last length whose weights fit in int32; it passed before the fix too, and is
    // here to show the fix did not move the values below the overflow.
    // 1293 is the first overflowing length; 1300 and 5000 are past it.
    for (int n : {1292, 1293, 1300, 5000}) {
        auto lm = linear_moments(arithmetic_sequence(n));
        const double L1 = lm[0], L2 = lm[1], T3 = lm[2], T4 = lm[3];

        // L-mean of 1 + 0.5*(0..n-1) is its arithmetic mean.
        CHECK_NEAR(L1, 1.0 + 0.25 * (n - 1), 1e-9);
        // L-scale of an evenly spaced sample: (n+1)/(n-1) * range/6, range = 0.5*(n-1).
        CHECK_NEAR(L2, (n + 1.0) / 6.0 / 2.0, 1e-9);
        // The two that the wrapped weights destroy: C# returns T4 = -0.185 at n = 1293 and
        // -1.446 at n = 1300.
        CHECK_NEAR(T3, 0.0, 1e-9);
        CHECK_NEAR(T4, 0.0, 1e-9);
    }

    // The moments must be continuous across the overflow boundary; a wrapped weight puts a
    // step there.
    {
        auto below = linear_moments(arithmetic_sequence(1292));
        auto above = linear_moments(arithmetic_sequence(1293));
        CHECK_TRUE(std::fabs(above[3] - below[3]) < 1e-6);
    }

    return chtest::summary("linear_moments_overflow");
}
