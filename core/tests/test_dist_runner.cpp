// ctest coverage of the shared distribution runner: the spec grammar builds what it claims,
// method dispatch returns the right shape, and every guard throws with a message that names
// the thing it could not do. Oracle VALUES live in fixtures/ and are asserted by
// test_fixtures.cpp; this file asserts structure and error paths only.
#include "corehydro/numerics/distributions/support/dist_runner.hpp"

#include <cmath>
#include <string>

#include "check.hpp"

namespace supp = corehydro::numerics::distributions::support;

int main() {
    // A flat family evaluates through the same entry point as a composite.
    {
        supp::DistResult r =
            supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "pdf", "[0]");
        CHECK_NEAR(r.values.at(0), 0.3989422804014327, 1e-12);
        CHECK_EQ(static_cast<int>(r.values.size()), 1);
    }

    // The pointwise verbs vectorize: args length in, values length out.
    {
        supp::DistResult r =
            supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "cdf", "[-1,0,1]");
        CHECK_EQ(static_cast<int>(r.values.size()), 3);
        CHECK_NEAR(r.values.at(1), 0.5, 1e-12);
    }

    // The fixture key spelling is an accepted alias.
    {
        supp::DistResult a =
            supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "pdf", "[0]");
        supp::DistResult b = supp::run_dist(R"({"target":"Normal","params":[0,1]})", "pdf", "[0]");
        CHECK_NEAR(a.values.at(0), b.values.at(0), 0.0);
    }

    // Composites nest to any depth: a truncated mixture is one spec.
    {
        const char* spec = R"({"family":"TruncatedDistribution",
            "base":{"family":"Mixture",
                    "components":[{"family":"Normal","parameters":[0,1]},
                                  {"family":"Normal","parameters":[5,1]}],
                    "weights":[0.5,0.5]},
            "bounds":[-2,7]})";
        supp::DistResult r = supp::run_dist(spec, "cdf", "[7]");
        CHECK_NEAR(r.values.at(0), 1.0, 1e-9);
    }

    // moments returns eight values with their names attached.
    {
        supp::DistResult r =
            supp::run_dist(R"({"family":"Normal","parameters":[3,2]})", "moments", "[]");
        CHECK_EQ(static_cast<int>(r.values.size()), 8);
        CHECK_EQ(static_cast<int>(r.names.size()), 8);
        CHECK_EQ(r.names.at(0) == "mean", true);
        CHECK_NEAR(r.values.at(0), 3.0, 1e-12);
    }

    // A seeded draw comes back whole, so a rebuild never splits a stream.
    {
        supp::DistResult r =
            supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "random", "[5,12345]");
        CHECK_EQ(static_cast<int>(r.values.size()), 5);
        supp::DistResult again =
            supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "random", "[5,12345]");
        CHECK_NEAR(r.values.at(4), again.values.at(4), 0.0);
    }

    // log_likelihood takes the whole sample in args.
    {
        supp::DistResult r = supp::run_dist(R"({"family":"Normal","parameters":[0,1]})",
                                            "log_likelihood", "[-1,0,1]");
        CHECK_EQ(static_cast<int>(r.values.size()), 1);
        CHECK_EQ(std::isfinite(r.values.at(0)), true);
    }

    // KernelDensity takes its data inline and defaults the bandwidth to the Silverman rule.
    {
        supp::DistResult r = supp::run_dist(
            R"({"family":"KernelDensity","data":[1,2,3,4,5,6,7,8,9,10]})", "cdf", "[5.5]");
        CHECK_NEAR(r.values.at(0), 0.5, 0.05);
    }

    // Guards.
    CHECK_THROWS_MSG(supp::run_dist(R"({"family":"NotAFamily"})", "pdf", "[0]"), "unknown distribution family");
    CHECK_THROWS_MSG(supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "not_a_method", "[0]"), "unknown distribution method");
    CHECK_THROWS_MSG(supp::run_dist(R"({"family":"Normal","parameters":[0,1]})", "linear_moments", "[]"), "linear moments");

    return chtest::summary("dist_runner");
}
