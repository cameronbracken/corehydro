// ctest for the optimizer runner: the callback abort path that keeps a throwing objective from
// being swallowed by Optimizer::minimize()/maximize() (kept first and parametrized over every
// method, since that guard is the whole point of this runner), then dispatch over the built-in
// objectives, a proof that a control value actually reaches the optimizer, and a seeded-DE
// reproducibility check.
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/support/optimizer_runner.hpp"

#include "check.hpp"
#include "optimization_test_functions.hpp"

namespace tb = corehydro::numerics::support;

namespace {

// A throwing-objective spec for `method`, with a low max_iterations (this test only cares that
// the ORIGINAL exception surfaces, not that the optimizer converges) and an explicit
// `report_failure`. "bfgs"/"powell"/"mlsl"/"nelder_mead" additionally need `initial`;
// `report_failure` is harmlessly ignored by the "nelder_mead"/"brent" arms (see
// optimizer_runner.hpp: only the four Optimizer-base methods read it).
std::string throwing_spec(const std::string& method, bool report_failure) {
    std::string body = "{\"method\":\"" + method + "\",\"lower\":[-5,-5],\"upper\":[5,5]";
    if (method == "bfgs" || method == "powell" || method == "mlsl" || method == "nelder_mead")
        body += ",\"initial\":[1,1]";
    body += ",\"seed\":1,\"control\":{\"max_iterations\":20,\"report_failure\":";
    body += report_failure ? "true" : "false";
    body += "}}";
    return body;
}

double throwing_objective(const std::vector<double>&) {
    throw std::runtime_error("objective exploded");
}

}  // namespace

int main() {
    // CRITICAL: a throwing objective surfaces as THAT exception -- not a silent Failure status,
    // and not some other exception an optimizer's own internal machinery (e.g. BFGS/MLSL's
    // gradient-probe finiteness check) raises when it is handed the guard's sentinel value. Every
    // one of the six methods, under both report_failure settings, must preserve the original
    // message.
    for (const char* method : {"de", "bfgs", "powell", "mlsl", "nelder_mead", "brent"}) {
        for (bool report_failure : {true, false}) {
            std::string spec = throwing_spec(method, report_failure);
            CHECK_THROWS_MSG(tb::run_optimizer(spec, throwing_objective), "objective exploded");
        }
    }

    // A control key actually reaches the optimizer: a very low max_function_evaluations caps the
    // reported count well below an unrestricted run's (proof against a transposed assignment in
    // apply_common_controls/apply_optimizer_controls going unnoticed).
    {
        std::string capped =
            "{\"method\":\"de\",\"lower\":[-5,-5],\"upper\":[5,5],\"seed\":42,"
            "\"control\":{\"max_function_evaluations\":40}}";
        std::string uncapped = "{\"method\":\"de\",\"lower\":[-5,-5],\"upper\":[5,5],\"seed\":42}";
        tb::OptimResult capped_r = tb::run_optimizer(capped, test_functions::de_jong);
        tb::OptimResult uncapped_r = tb::run_optimizer(uncapped, test_functions::de_jong);
        CHECK_TRUE(capped_r.function_evaluations <= 45);
        CHECK_TRUE(capped_r.function_evaluations < uncapped_r.function_evaluations);
    }

    // De Jong (sphere) has its minimum at the origin; every method should get close.
    for (const char* method : {"de", "bfgs", "powell", "nelder_mead"}) {
        std::string spec = std::string("{\"method\":\"") + method +
                           "\",\"lower\":[-5,-5],\"upper\":[5,5],\"initial\":[1,1],\"seed\":12345}";
        tb::OptimResult r = tb::run_optimizer(spec, test_functions::de_jong);
        CHECK_EQ(r.parameters.size(), std::size_t{2});
        CHECK_TRUE(r.value < 1e-6);
    }

    // A seeded DE run is reproducible.
    std::string spec = "{\"method\":\"de\",\"lower\":[-5,-5],\"upper\":[5,5],\"seed\":777}";
    auto a = tb::run_optimizer(spec, test_functions::rosenbrock);
    auto b = tb::run_optimizer(spec, test_functions::rosenbrock);
    CHECK_NEAR(a.value, b.value, 0.0);
    CHECK_NEAR(a.parameters[0], b.parameters[0], 0.0);

    return chtest::summary("test_optimizer_runner");
}
