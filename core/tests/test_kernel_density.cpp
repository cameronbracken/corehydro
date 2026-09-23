// C++-only regression ctest for a corehydro-introduced divergence caught in code review of
// Task 7 (the EmpiricalDistribution v2.1.4 ValidateData wave): KernelDensity's internal CDF
// table (create_cdf(), see kernel_density.hpp) previously constructed its `cdf_table_` through
// EmpiricalDistribution's normal, CHECKED public constructor. Once that constructor gained
// ValidateData (strict monotonicity checks on the cumulative-probability array), any
// compact-support kernel (Epanechnikov/Triangular/Uniform) with a bandwidth small relative to
// the sample spread would produce a `cdf_table_` with tied consecutive cumulative
// probabilities (the density is exactly zero over long stretches between sample clusters), and
// cdf()/inverse_cdf() would start throwing std::invalid_argument -- even though
// KernelDensity::parameters_valid() reports true.
//
// This is NOT what real C# does: `KernelDensity.CreateCDF` builds a raw `OrderedPairedData`
// directly (never an `EmpiricalDistribution`), so its `CDF`/`InverseCDF` never run
// `ValidateData` and can never throw on this. The fix routes `cdf_table_`'s construction through
// `EmpiricalDistribution::create_raw_table()` (a private, KernelDensity-friended factory that
// skips validate_data() entirely), restoring parity: KernelDensity's own CDF/InverseCDF must
// never throw due to internal-table ties, regardless of kernel/bandwidth choice.
#include <cmath>
#include <limits>
#include <vector>

#include "corehydro/numerics/distributions/kernel_density.hpp"
#include "check.hpp"

using corehydro::numerics::distributions::KernelDensity;
using corehydro::numerics::distributions::KernelType;

namespace {

// Sample points and bandwidth are chosen so the 1000-bin grid over [minimum, maximum] captures
// SOME nonzero-density bins near each point (avoiding an all-zero, then NaN-after-normalization,
// degenerate table -- a separate, unrelated numerical edge case that a too-tiny bandwidth
// relative to the bin width can hit) while still leaving a long exact-zero-density stretch
// between the two clusters, which is what actually produces the tied cumulative probabilities
// this test is pinning: two points 10 apart, bandwidth 0.1 (bin width ~0.0106 over the
// ~10.6-wide range), so each point's compact-support neighborhood (radius 0.1, ~19 bins)
// contributes real mass, but the ~9.8-wide gap between them (~925 bins) is exactly zero density
// throughout.
constexpr double kX0 = 0.0;
constexpr double kX1 = 10.0;
constexpr double kBandwidth = 0.1;
constexpr double kMidGap = 5.0;  // deep in the exact-zero-density gap between the two points

void check_cdf_and_inverse_cdf_do_not_throw(KernelType kernel) {
    KernelDensity kd({kX0, kX1}, kernel, kBandwidth);
    CHECK_TRUE(kd.parameters_valid());

    bool cdf_threw = false;
    double cdf_mid = std::numeric_limits<double>::quiet_NaN();
    try {
        cdf_mid = kd.cdf(kMidGap);
    } catch (...) {
        cdf_threw = true;
    }
    CHECK_TRUE(!cdf_threw);
    CHECK_TRUE(std::isfinite(cdf_mid));
    CHECK_TRUE(cdf_mid >= 0.0 && cdf_mid <= 1.0);

    bool inverse_threw = false;
    double q = std::numeric_limits<double>::quiet_NaN();
    try {
        q = kd.inverse_cdf(0.5);
    } catch (...) {
        inverse_threw = true;
    }
    CHECK_TRUE(!inverse_threw);
    CHECK_TRUE(std::isfinite(q));
}

// Before the fix, all three compact-support kernels would throw building this table (tied
// cumulative probabilities across the long zero-density gap); Gaussian (unbounded support,
// never exactly zero in this range) never exercised the bug, so it's not repeated here -- the
// existing gaussian_unbounded_cdf oracle fixture case already covers that path.
void test_epanechnikov_kernel_tiny_bandwidth_cdf_does_not_throw() {
    check_cdf_and_inverse_cdf_do_not_throw(KernelType::Epanechnikov);
}

void test_uniform_kernel_tiny_bandwidth_cdf_does_not_throw() {
    check_cdf_and_inverse_cdf_do_not_throw(KernelType::Uniform);
}

void test_triangular_kernel_tiny_bandwidth_cdf_does_not_throw() {
    check_cdf_and_inverse_cdf_do_not_throw(KernelType::Triangular);
}

void test_v220_bandwidth_validation_and_degenerate_sample() {
    CHECK_THROWS(KernelDensity({1.0, 2.0}, KernelType::Gaussian,
                               std::numeric_limits<double>::quiet_NaN()));
    CHECK_THROWS(KernelDensity({1.0, 2.0}, KernelType::Gaussian,
                               std::numeric_limits<double>::infinity()));
    CHECK_THROWS(KernelDensity({}, KernelType::Gaussian));
    CHECK_THROWS(KernelDensity({1.0, std::numeric_limits<double>::infinity()},
                               KernelType::Gaussian));
    KernelDensity constant({5.0, 5.0, 5.0, 5.0}, KernelType::Gaussian);
    CHECK_EQ(constant.bandwidth(), 5e-9);
    CHECK_TRUE(std::isfinite(constant.pdf(5.0)));
    KernelDensity zero({0.0, 0.0, 0.0, 0.0}, KernelType::Gaussian);
    CHECK_EQ(zero.bandwidth(), 1e-9);
}

}  // namespace

int main() {
    test_epanechnikov_kernel_tiny_bandwidth_cdf_does_not_throw();
    test_uniform_kernel_tiny_bandwidth_cdf_does_not_throw();
    test_triangular_kernel_tiny_bandwidth_cdf_does_not_throw();
    test_v220_bandwidth_validation_and_degenerate_sample();
    return chtest::summary("test_kernel_density");
}
