// GEV distribution core, pinned to the oracle values in the upstream C# test
// Test_GeneralizedExtremeValue.cs (themselves from Rao & Hamed, "Flood Frequency
// Analysis", CRC Press 2000). Comparison modes/tolerances mirror the C# asserts.
#include "corehydro/numerics/distributions/generalized_extreme_value.hpp"

#include <cmath>
#include <limits>

#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"
#include "check.hpp"

using corehydro::numerics::kEuler;
using corehydro::numerics::distributions::GeneralizedExtremeValue;
namespace sf = corehydro::numerics::math::special;  // not "gamma": clashes with glibc gamma()

constexpr double INF = std::numeric_limits<double>::infinity();

int main() {
    // --- Gamma sanity (leaf dependency) ---
    CHECK_NEAR(sf::function(1.0), 1.0, 1e-12);
    CHECK_NEAR(sf::function(5.0), 24.0, 1e-10);
    CHECK_NEAR(sf::function(0.5), std::sqrt(corehydro::numerics::kPi), 1e-12);

    // --- Construction ---
    {
        GeneralizedExtremeValue g;
        CHECK_EQ(g.xi(), 100.0);
        CHECK_EQ(g.alpha(), 10.0);
        CHECK_EQ(g.kappa(), 0.0);
        GeneralizedExtremeValue g2(-100, 1, 1);
        CHECK_EQ(g2.xi(), -100.0);
        CHECK_EQ(g2.alpha(), 1.0);
        CHECK_EQ(g2.kappa(), 1.0);
    }

    // --- Invalid parameters ---
    {
        CHECK_EQ(GeneralizedExtremeValue(NAN, NAN, NAN).parameters_valid(), false);
        CHECK_EQ(GeneralizedExtremeValue(INF, INF, INF).parameters_valid(), false);
        CHECK_EQ(GeneralizedExtremeValue(100, 0, 1).parameters_valid(), false);  // scale<=0
        CHECK_EQ(GeneralizedExtremeValue(100, 10, 0).parameters_valid(), true);
    }

    // --- Mean ---
    CHECK_EQ(GeneralizedExtremeValue().mean(), 100.0 + 10.0 * kEuler);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 0.9).mean(), 100.42482, 1e-4);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 10).mean(), -3628699.0000000028, 1e-6);

    // --- Median ---
    CHECK_NEAR(GeneralizedExtremeValue().median(), 103.66512, 1e-4);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 0.9).median(), 103.12196423496414, 1e-12);

    // --- Mode ---
    CHECK_EQ(GeneralizedExtremeValue().mode(), 100.0);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 1).mode(), 110.0, 1e-9);

    // --- Standard deviation ---
    CHECK_NEAR(GeneralizedExtremeValue().standard_deviation(), 12.825498, 1e-5);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 0.49).standard_deviation(), 9.280898, 1e-4);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 1).standard_deviation(), 10.0, 1e-12);

    // --- Skewness ---
    CHECK_NEAR(GeneralizedExtremeValue().skewness(), 1.1395470994046487, 1e-15);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 0.3).skewness(), -0.0690175, 1e-3);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 1).skewness(), -2.0, 0.0);

    // --- Kurtosis ---
    CHECK_NEAR(GeneralizedExtremeValue().kurtosis(), 3.0 + 12.0 / 5.0, 1e-9);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 0.24).kurtosis(), 2.7659607, 1e-4);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 1).kurtosis(), 9.0, 0.0);

    // --- Minimum / Maximum ---
    CHECK_EQ(GeneralizedExtremeValue().minimum(), -INF);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, -5).minimum(), 98.0, 1e-9);
    CHECK_EQ(GeneralizedExtremeValue().maximum(), INF);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 1).maximum(), 110.0, 1e-9);

    // --- PDF ---
    CHECK_EQ(GeneralizedExtremeValue().pdf(0), 0.0);
    CHECK_EQ(GeneralizedExtremeValue().pdf(1), 0.0);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 1).pdf(0), 1.67017007902456E-06, 1e-10);

    // --- CDF ---
    CHECK_NEAR(GeneralizedExtremeValue().cdf(100), 0.367879, 1e-4);
    CHECK_NEAR(GeneralizedExtremeValue().cdf(200), 0.9999546, 1e-7);
    CHECK_NEAR(GeneralizedExtremeValue(100, 10, 1).cdf(100), 0.367879, 1e-5);
    CHECK_EQ(GeneralizedExtremeValue(100, 10, 1).cdf(200), 1.0);

    // --- InverseCDF ---
    CHECK_EQ(GeneralizedExtremeValue().inverse_cdf(0), -INF);
    CHECK_NEAR(GeneralizedExtremeValue().inverse_cdf(0.5), 103.66512, 1e-5);
    CHECK_EQ(GeneralizedExtremeValue().inverse_cdf(1), INF);

    // --- LogLikelihood sums LogPDF ---
    {
        GeneralizedExtremeValue g(100, 10, 0.1);
        std::vector<double> s = {95.0, 110.0, 102.0};
        double expected = std::log(g.pdf(95.0)) + std::log(g.pdf(110.0)) + std::log(g.pdf(102.0));
        CHECK_NEAR(g.log_likelihood(s), expected, 1e-12);
    }

    return chtest::summary("generalized_extreme_value");
}
