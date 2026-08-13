// ctest for the shared toolbox runner: dispatch, option parsing, and error messages.
#include <cmath>
#include <string>
#include <vector>

#include "corehydro/numerics/support/toolbox_runner.hpp"
#include "check.hpp"

namespace tb = corehydro::numerics::support;

int main() {
    const std::vector<double> x{14.0, 8.0, 32.0, 7.0, 3.0, 15.0};
    const std::vector<double> y{10.0, 5.0, 7.0, 4.0, 3.0, 8.0};

    auto pearson = tb::run_toolbox("correlation", "pearson", {x, y}, "{}");
    CHECK_EQ(pearson.values.size(), std::size_t{1});
    CHECK_NEAR(pearson.values[0], 0.54502739907793, 1e-12);

    auto spearman = tb::run_toolbox("correlation", "spearman", {x, y}, "{}");
    CHECK_NEAR(spearman.values[0], 0.771428571428571, 1e-12);

    auto tau = tb::run_toolbox("correlation", "kendall", {x, y}, "{}");
    CHECK_NEAR(tau.values[0], 0.6, 1e-12);

    // An unknown group names the group; an unknown method names the method.
    bool threw_group = false;
    try {
        tb::run_toolbox("nope", "pearson", {x, y}, "{}");
    } catch (const std::exception& e) {
        threw_group = std::string(e.what()).find("nope") != std::string::npos;
    }
    CHECK_TRUE(threw_group);

    bool threw_method = false;
    try {
        tb::run_toolbox("correlation", "nope", {x, y}, "{}");
    } catch (const std::exception& e) {
        threw_method = std::string(e.what()).find("nope") != std::string::npos;
    }
    CHECK_TRUE(threw_method);

    // Too few data vectors is an error, not a crash.
    bool threw_arity = false;
    try {
        tb::run_toolbox("correlation", "pearson", {x}, "{}");
    } catch (const std::exception&) {
        threw_arity = true;
    }
    CHECK_TRUE(threw_arity);

    // gof: the named set and the individual metric agree, and the set is labelled.
    const std::vector<double> obs{2.0, 4.0, 6.0, 8.0, 10.0};
    const std::vector<double> mod{2.2, 3.9, 6.4, 7.5, 10.1};
    auto set = tb::run_toolbox("gof", "metrics", {obs, mod}, "{}");
    CHECK_EQ(set.values.size(), set.names.size());
    CHECK_EQ(set.values.size(), std::size_t{17});
    auto one = tb::run_toolbox("gof", "nse", {obs, mod}, "{}");
    std::size_t nse_at = 0;
    for (std::size_t i = 0; i < set.names.size(); ++i)
        if (set.names[i] == "nse") nse_at = i;
    CHECK_NEAR(set.values[nse_at], one.values[0], 0.0);

    // aic takes its arguments from options, not from a data vector.
    auto aic = tb::run_toolbox("gof", "aic", {}, "{\"k\":2,\"log_likelihood\":-121.01131220612}");
    CHECK_NEAR(aic.values[0], 246.02262441224, 1e-9);

    // The distribution-backed tests build their model from a dist spec in the options.
    auto ks = tb::run_toolbox("gof", "ks", {obs},
                              "{\"model\":{\"family\":\"Normal\",\"parameters\":[6.0,3.0]}}");
    CHECK_TRUE(ks.values[0] > 0.0 && ks.values[0] < 1.0);

    // --- statistics group ----------------------------------------------------------------
    const std::vector<double> series{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};

    auto summary = tb::run_toolbox("statistics", "summary", {series}, "{}");
    CHECK_EQ(summary.values.size(), summary.names.size());
    CHECK_EQ(summary.values.size(), std::size_t{13});
    std::size_t mean_at = 0, n_at = 0;
    for (std::size_t i = 0; i < summary.names.size(); ++i) {
        if (summary.names[i] == "mean") mean_at = i;
        if (summary.names[i] == "n") n_at = i;
    }
    CHECK_NEAR(summary.values[mean_at], 5.0, 1e-12);
    CHECK_EQ(static_cast<int>(summary.values[n_at]), 8);

    // A chunked "running" call seeded from a prior "summary" call's state matches one shot.
    auto part1 = tb::run_toolbox("statistics", "summary", {{2.0, 4.0, 4.0, 4.0}}, "{}");
    const std::vector<std::string> state_keys{"m1", "m2", "m3", "m4", "minimum", "maximum"};
    std::string state = "{\"state\":{";
    for (const auto& key : state_keys) {
        std::size_t at = 0;
        for (std::size_t i = 0; i < part1.names.size(); ++i)
            if (part1.names[i] == key) at = i;
        state += "\"" + key + "\":" + std::to_string(part1.values[at]) + ",";
    }
    state += "\"n\":" + std::to_string(static_cast<int>(part1.values[n_at])) + "}}";
    auto part2 = tb::run_toolbox("statistics", "running", {{5.0, 5.0, 7.0, 9.0}}, state);
    std::size_t p2_mean_at = 0;
    for (std::size_t i = 0; i < part2.names.size(); ++i)
        if (part2.names[i] == "mean") p2_mean_at = i;
    CHECK_NEAR(part2.values[p2_mean_at], summary.values[mean_at], 1e-9);

    auto pm = tb::run_toolbox("statistics", "product_moments", {obs}, "{}");
    CHECK_EQ(pm.values.size(), std::size_t{4});
    CHECK_EQ(pm.names.size(), std::size_t{4});

    auto lm = tb::run_toolbox("statistics", "l_moments", {obs}, "{}");
    CHECK_EQ(lm.values.size(), std::size_t{4});

    // Tie handling: the two 1.0 entries (indices 1 and 3) split rank 1.5; the lone 2.0 (index 2)
    // gets rank 3; the lone 3.0 (index 0) gets rank 4.
    auto ranks = tb::run_toolbox("statistics", "ranks", {{3.0, 1.0, 2.0, 1.0}}, "{}");
    CHECK_NEAR(ranks.values[0], 4.0, 0.0);
    CHECK_NEAR(ranks.values[1], 1.5, 0.0);
    CHECK_NEAR(ranks.values[2], 3.0, 0.0);
    CHECK_NEAR(ranks.values[3], 1.5, 0.0);

    auto pct = tb::run_toolbox("statistics", "percentile", {obs, {0.0, 0.5, 1.0}}, "{}");
    CHECK_EQ(pct.values.size(), std::size_t{3});
    CHECK_NEAR(pct.values[0], 2.0, 1e-12);   // min
    CHECK_NEAR(pct.values[2], 10.0, 1e-12);  // max

    // running_covariance: five identical pushes of a constant vector (every variable equals the
    // push index) matches the RunningCovariance.mean_element/covariance_element oracle in
    // fixtures/special_functions/running_covariance.json (Mean = [3,3,3,3,3], Covariance = 11 on
    // the diagonal, 10 off-diagonal).
    std::vector<double> col{1.0, 2.0, 3.0, 4.0, 5.0};
    auto rc = tb::run_toolbox("statistics", "running_covariance", {col, col, col, col, col}, "{}");
    CHECK_EQ(static_cast<int>(rc.values[0]), 5);   // n
    CHECK_EQ(rc.dims.size(), std::size_t{2});
    CHECK_EQ(rc.dims[0], 5);
    CHECK_EQ(rc.dims[1], 5);
    // mean block starts at index 1
    for (int i = 0; i < 5; ++i) CHECK_NEAR(rc.values[static_cast<std::size_t>(1 + i)], 3.0, 1e-12);
    // covariance block starts at index 1+5=6; diagonal entries are at 6 + i*5 + i
    CHECK_NEAR(rc.values[6], 11.0, 1e-9);
    CHECK_NEAR(rc.values[7], 10.0, 1e-9);  // (0,1) off-diagonal

    // --- spectra group ---------------------------------------------------------------------
    const std::vector<double> ac_series{5.0, 6.0, 4.0, 7.0, 3.0, 8.0, 2.0, 9.0, 1.0, 10.0};

    auto acf_corr = tb::run_toolbox("spectra", "autocorrelation", {ac_series}, "{\"lag_max\":3}");
    CHECK_EQ(acf_corr.dims.size(), std::size_t{2});
    CHECK_EQ(acf_corr.dims[0], 4);
    CHECK_EQ(acf_corr.dims[1], 2);
    CHECK_NEAR(acf_corr.values[1], 1.0, 1e-12);  // lag 0 correlation is 1 by construction

    auto acf_cov = tb::run_toolbox("spectra", "autocorrelation", {ac_series},
                                   "{\"lag_max\":3,\"type\":\"covariance\"}");
    auto acf_partial = tb::run_toolbox("spectra", "autocorrelation", {ac_series},
                                       "{\"lag_max\":3,\"type\":\"partial\"}");
    CHECK_EQ(acf_partial.dims[0], 3);  // lag_max, not lag_max + 1
    CHECK_NEAR(acf_partial.values[1], acf_corr.values[3], 1e-9);  // PACF at lag 1 == ACF at lag 1

    auto ci = tb::run_toolbox("spectra", "autocorrelation_ci", {}, "{\"sample_size\":100}");
    CHECK_EQ(ci.values.size(), std::size_t{2});
    CHECK_NEAR(ci.values[0] + ci.values[1], 0.0, 1e-12);

    const std::vector<double> a{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> b{4.0, 3.0, 2.0, 1.0};
    auto xcorr = tb::run_toolbox("spectra", "cross_correlation", {a, b}, "{}");
    CHECK_EQ(xcorr.values.size(), std::size_t{4});

    auto fwd = tb::run_toolbox("spectra", "dft", {{1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0}}, "{}");
    CHECK_EQ(fwd.values.size(), std::size_t{8});
    auto real_fwd = tb::run_toolbox("spectra", "dft_real", {a}, "{}");
    CHECK_EQ(real_fwd.values.size(), std::size_t{4});

    return chtest::summary("toolbox_runner");
}
