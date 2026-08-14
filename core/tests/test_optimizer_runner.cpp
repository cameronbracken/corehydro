// ctest for the optimizer runner: dispatch over the built-in objectives, and the callback
// abort path that keeps a throwing objective from being swallowed by Optimizer::minimize().
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/support/optimizer_runner.hpp"

#include "check.hpp"
#include "optimization_test_functions.hpp"

namespace tb = corehydro::numerics::support;

int main() {
    // De Jong (sphere) has its minimum at the origin; every method should get close.
    for (const char* method : {"de", "bfgs", "powell", "nelder_mead"}) {
        std::string spec = std::string("{\"method\":\"") + method +
                           "\",\"lower\":[-5,-5],\"upper\":[5,5],\"initial\":[1,1],\"seed\":12345}";
        tb::OptimResult r = tb::run_optimizer(spec, test_functions::de_jong);
        CHECK_EQ(r.parameters.size(), std::size_t{2});
        CHECK_TRUE(r.value < 1e-6);
    }

    // A throwing objective surfaces as that exception, not as a silent Failure status.
    bool rethrown = false;
    try {
        tb::run_optimizer("{\"method\":\"de\",\"lower\":[-5,-5],\"upper\":[5,5],\"seed\":1}",
                          [](const std::vector<double>&) -> double {
                              throw std::runtime_error("objective exploded");
                          });
    } catch (const std::exception& e) {
        rethrown = std::string(e.what()).find("objective exploded") != std::string::npos;
    }
    CHECK_TRUE(rethrown);

    // A seeded DE run is reproducible.
    std::string spec = "{\"method\":\"de\",\"lower\":[-5,-5],\"upper\":[5,5],\"seed\":777}";
    auto a = tb::run_optimizer(spec, test_functions::rosenbrock);
    auto b = tb::run_optimizer(spec, test_functions::rosenbrock);
    CHECK_NEAR(a.value, b.value, 0.0);
    CHECK_NEAR(a.parameters[0], b.parameters[0], 0.0);

    return chtest::summary("toolbox_runner");
}
