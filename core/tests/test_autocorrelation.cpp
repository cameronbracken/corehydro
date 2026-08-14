// ctest for the ported Autocorrelation class (numerics/data/autocorrelation.hpp). Every
// assertion is a relationship that holds by construction -- never a numeric literal copied from
// the C++ implementation; the oracle gate (fixtures/toolbox/autocorrelation.json, transcribed
// from Test_Autocorrelation.cs) is what pins actual numbers against the real C# library.
#include <cstddef>
#include <vector>

#include "corehydro/numerics/data/autocorrelation.hpp"
#include "check.hpp"

namespace bfdata = corehydro::numerics::data;
using Type = bfdata::Autocorrelation::Type;

int main() {
    // A short daily series (same shape as Test_Autocorrelation.cs's sample, truncated).
    const std::vector<double> x{142.25, 141.23, 141.33, 140.82, 141.31, 140.58, 141.58, 142.15,
                                 143.07, 142.85, 143.17, 142.54, 143.07, 142.26, 142.97, 143.86,
                                 142.57, 142.19, 142.35, 142.63, 144.15, 144.73, 144.70, 144.97};

    // Correlation at lag 0 is 1 by construction: it is acf[0] divided by itself.
    auto acf = bfdata::Autocorrelation::function(x, 5, Type::Correlation);
    CHECK_TRUE(acf.has_value());
    CHECK_EQ(acf->size(), std::size_t{6});  // lag_max + 1
    CHECK_NEAR((*acf)[0][1], 1.0, 1e-15);
    // The lag column mirrors the row index.
    for (std::size_t i = 0; i < acf->size(); ++i) CHECK_NEAR((*acf)[i][0], static_cast<double>(i), 0.0);

    // Covariance at lag 0 equals the population variance, computed independently of
    // autocorrelation.hpp's own mean()/variance() helpers.
    auto acvf = bfdata::Autocorrelation::function(x, 5, Type::Covariance);
    CHECK_TRUE(acvf.has_value());
    CHECK_EQ(acvf->size(), std::size_t{6});
    double m = 0.0;
    for (double v : x) m += v;
    m /= static_cast<double>(x.size());
    double pop_var = 0.0;
    for (double v : x) pop_var += (v - m) * (v - m);
    pop_var /= static_cast<double>(x.size());
    CHECK_NEAR((*acvf)[0][1], pop_var, 1e-9);

    // Partial autocorrelation at lag 1 equals the correlation at lag 1 (both reduce to
    // acvf[1]/acvf[0] on the Durbin-Levinson recursion's first step).
    auto pacf = bfdata::Autocorrelation::function(x, 5, Type::Partial);
    CHECK_TRUE(pacf.has_value());
    CHECK_EQ(pacf->size(), std::size_t{5});  // lag_max, not lag_max + 1: PACF has no lag-0 row
    CHECK_NEAR((*pacf)[0][1], (*acf)[1][1], 1e-12);

    // A series shorter than two elements returns std::nullopt, for every type.
    CHECK_TRUE(!bfdata::Autocorrelation::function(std::vector<double>{1.0}).has_value());
    CHECK_TRUE(!bfdata::Autocorrelation::function(std::vector<double>{}).has_value());
    CHECK_TRUE(!bfdata::Autocorrelation::function(std::vector<double>{1.0}, -1, Type::Covariance).has_value());
    CHECK_TRUE(!bfdata::Autocorrelation::function(std::vector<double>{1.0}, -1, Type::Partial).has_value());

    // correlation_confidence_interval is symmetric about zero: the standard normal quantile at
    // alpha is the negative of the quantile at 1 - alpha.
    auto ci = bfdata::Autocorrelation::correlation_confidence_interval(100);
    CHECK_EQ(ci.size(), std::size_t{2});
    CHECK_NEAR(ci[0] + ci[1], 0.0, 1e-12);
    CHECK_TRUE(ci[0] < 0.0 && ci[1] > 0.0);

    // The default lag_max only ever widens with more data (floor(min(10*log10(n), n-1))), so a
    // longer series called with the library default must return at least as many pairs.
    auto acf_default_short = bfdata::Autocorrelation::function(std::vector<double>(x.begin(), x.begin() + 5));
    auto acf_default_long = bfdata::Autocorrelation::function(x);
    CHECK_TRUE(acf_default_short.has_value() && acf_default_long.has_value());
    CHECK_TRUE(acf_default_long->size() >= acf_default_short->size());

    return chtest::summary("autocorrelation");
}
