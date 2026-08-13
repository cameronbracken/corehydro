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

    return chtest::summary("toolbox_runner");
}
