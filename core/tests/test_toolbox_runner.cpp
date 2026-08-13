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

    return chtest::summary("toolbox_runner");
}
