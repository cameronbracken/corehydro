// GEV parameter estimation, pinned to the upstream C# oracles
// (Test_GeneralizedExtremeValue.cs; Rao & Hamed, "Flood Frequency Analysis", 2000).
#include "corehydro/numerics/distributions/generalized_extreme_value.hpp"

#include <cmath>
#include <vector>

#include "check.hpp"

using corehydro::numerics::distributions::EstimationMethod;
using corehydro::numerics::distributions::GeneralizedExtremeValue;

// Relative-error check matching the C# `Assert.IsLessThan(0.01, (x-true)/true)` pattern.
#define CHECK_REL(actual, expected, tol) \
    CHECK_TRUE(((actual) - (expected)) / (expected) < (tol))

int main() {
    // White River near Nora, IN (Rao & Hamed Table 7.1.2)
    std::vector<double> sample = {
        23200, 2950, 10300, 23200, 4540, 9960, 10800, 26900, 23300, 20400, 8480, 3150,
        9380, 32400, 20800, 11100, 7270, 9600, 14600, 14300, 22500, 14700, 12700, 9740,
        3050, 8830, 12000, 30400, 27000, 15200, 8040, 11700, 20300, 22700, 30400, 9180,
        4870, 14700, 12800, 13700, 7960, 9830, 12500, 10700, 13200, 14700, 14300, 4050,
        14600, 14400, 19200, 7160, 12100, 8650, 10600, 24500, 14400, 6300, 9560, 15800,
        14300, 28700};

    // --- Method of moments (Example 7.1.1) ---
    {
        GeneralizedExtremeValue g;
        g.estimate(sample, EstimationMethod::MethodOfMoments);
        CHECK_REL(g.xi(), 11012.0, 0.01);
        CHECK_REL(g.alpha(), 6209.4, 0.01);
        CHECK_REL(g.kappa(), 0.0736, 0.01);
    }

    // --- Method of linear moments (Example 7.1.1) ---
    {
        std::vector<double> s = {1953, 1939, 1677, 1692, 2051, 2371, 2022, 1521, 1448, 1825, 1363,
                                 1760, 1672, 1603, 1244, 1521, 1783, 1560, 1357, 1673, 1625, 1425,
                                 1688, 1577, 1736, 1640, 1584, 1293, 1277, 1742, 1491};
        GeneralizedExtremeValue g;
        g.estimate(s, EstimationMethod::MethodOfLinearMoments);
        CHECK_NEAR(g.xi(), 1543.933, 0.001);
        CHECK_NEAR(g.alpha(), 218.1148, 0.001);
        CHECK_NEAR(g.kappa(), 0.1068473, 0.001);
        auto lmom = g.linear_moments_from_parameters({g.xi(), g.alpha(), g.kappa()});
        CHECK_NEAR(lmom[0], 1648.806, 0.001);
        CHECK_NEAR(lmom[1], 138.2366, 0.001);
        CHECK_NEAR(lmom[2], 0.1030703, 0.001);
        CHECK_NEAR(lmom[3], 0.1277244, 0.001);
    }

    // --- Maximum likelihood (Example 7.1.1) ---
    {
        GeneralizedExtremeValue g;
        g.estimate(sample, EstimationMethod::MaximumLikelihood);
        CHECK_REL(g.xi(), 10849.0, 0.01);
        CHECK_REL(g.alpha(), 5745.6, 0.01);
        CHECK_REL(g.kappa(), 0.005, 0.01);
    }

    // --- Quantile round-trip (Example 7.1.2) ---
    {
        GeneralizedExtremeValue g(10849, 5745.6, 0.005);
        double q100 = g.inverse_cdf(0.99);
        CHECK_REL(q100, 36977.0, 0.01);
        CHECK_REL(g.cdf(q100), 0.99, 0.01);
    }

    // --- Standard error (Example 7.1.3), sample size 62 ---
    {
        const int n = static_cast<int>(sample.size());  // 62
        GeneralizedExtremeValue g(10849, 5745.6, 0.005);
        auto partials = g.quantile_gradient(0.99);
        auto covar = g.parameter_covariance(n);
        double qvar = g.quantile_variance(0.99, n);
        CHECK_REL(partials[0], 1.0, 0.01);
        CHECK_REL(partials[1], 4.5472, 0.01);
        CHECK_REL(partials[2], -59861.0, 0.01);
        CHECK_REL(covar[0][0], 664669.0, 0.01);
        CHECK_REL(covar[1][1], 346400.0, 0.01);
        CHECK_REL(covar[2][2], 0.007655, 0.01);
        CHECK_REL(covar[0][1], 176180.0, 0.01);
        CHECK_REL(covar[0][2], 23.977, 0.01);
        CHECK_REL(covar[1][2], 13.8574, 0.01);
        CHECK_REL(qvar, 26445364.0, 0.01);
        CHECK_REL(std::sqrt(qvar), 5142.0, 0.01);
    }

    return chtest::summary("gev_estimation");
}
