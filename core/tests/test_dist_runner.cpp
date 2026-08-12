// ctest coverage of the shared distribution runner: the spec grammar builds what it claims,
// method dispatch returns the right shape, and every guard throws with a message that names
// the thing it could not do. Oracle VALUES live in fixtures/ and are asserted by
// test_fixtures.cpp; this file asserts structure and error paths only.
#include "corehydro/numerics/distributions/support/dist_runner.hpp"

#include <cmath>
#include <string>

#include "check.hpp"
#include "corehydro/models/model_spec.hpp"

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

    // --- copulas ------------------------------------------------------------------------
    {
        supp::DistResult r =
            supp::run_copula(R"({"family":"Clayton","theta":2})", "pdf", "[0.3,0.7]");
        CHECK_EQ(static_cast<int>(r.values.size()), 1);
        CHECK_EQ(std::isfinite(r.values.at(0)), true);
    }
    {
        supp::DistResult r =
            supp::run_copula(R"({"family":"Clayton","theta":2})", "tail_dependence", "[]");
        CHECK_EQ(static_cast<int>(r.values.size()), 2);
        CHECK_EQ(r.names.at(0) == "lower", true);
        // Closed form for Clayton: lambda_L = 2^(-1/theta).
        CHECK_NEAR(r.values.at(0), std::pow(2.0, -0.5), 1e-12);
    }
    {
        supp::DistResult r =
            supp::run_copula(R"({"family":"Clayton","theta":2})", "bounds", "[]");
        CHECK_EQ(static_cast<int>(r.values.size()), 2);
    }
    {
        // A copula with marginals attached samples pairs; 2n values, x-major then y.
        const char* spec = R"({"family":"Clayton","theta":2,
            "margin_x":{"family":"Normal","parameters":[0,1]},
            "margin_y":{"family":"Normal","parameters":[0,1]}})";
        supp::DistResult r = supp::run_copula(spec, "random", "[4,12345]");
        CHECK_EQ(static_cast<int>(r.values.size()), 8);
    }
    {
        // The three log-likelihoods take x then y, split at the halfway point.
        const char* spec = R"({"family":"Clayton","theta":2,
            "margin_x":{"family":"Normal","parameters":[0,1]},
            "margin_y":{"family":"Normal","parameters":[0,1]}})";
        supp::DistResult r =
            supp::run_copula(spec, "log_likelihood_ifm", "[-1,0,1,-0.5,0.2,0.9]");
        CHECK_EQ(std::isfinite(r.values.at(0)), true);
    }
    CHECK_THROWS_MSG(supp::run_copula(R"({"family":"Joe","fit":{"x":[1,2],"y":[1,2],"method":"tau"}})", "theta", "[]"), "tau");

    // --- multivariate -------------------------------------------------------------------
    {
        const char* spec = R"({"family":"MultivariateNormal","mean":[0,0],
                               "covariance":[[1,0],[0,1]]})";
        supp::DistResult r = supp::run_mvdist(spec, "pdf", "[0,0]");
        // Standard bivariate normal at the origin: 1 / (2 pi). std::acos(-1.0) rather than
        // M_PI, which is absent under strict -std=c++17 on Linux and on MSVC.
        CHECK_NEAR(r.values.at(0), 1.0 / (2.0 * std::acos(-1.0)), 1e-12);
    }
    {
        const char* spec = R"({"family":"MultivariateNormal","mean":[1,2,3],
                               "covariance":[[1,0,0],[0,1,0],[0,0,1]]})";
        // marginal returns a child spec, 0-based indices at this layer.
        supp::DistResult r = supp::run_mvdist(spec, "marginal", "[0,2]");
        CHECK_EQ(r.spec.empty(), false);
        supp::DistResult child = supp::run_mvdist(r.spec, "dimension", "[]");
        CHECK_NEAR(child.values.at(0), 2.0, 0.0);
        supp::DistResult mean = supp::run_mvdist(r.spec, "mean", "[]");
        CHECK_NEAR(mean.values.at(1), 3.0, 1e-12);
    }
    {
        const char* spec = R"({"family":"MultivariateNormal","mean":[0,0],
                               "covariance":[[1,0],[0,1]]})";
        // interval takes lower then upper, split at the halfway point.
        supp::DistResult r = supp::run_mvdist(spec, "interval", "[-100,-100,100,100]");
        CHECK_NEAR(r.values.at(0), 1.0, 1e-6);
    }
    CHECK_THROWS_MSG(supp::run_mvdist(R"({"family":"MultivariateStudentT","df":5,"location":[0,0]})", "marginal", "[0]"), "MultivariateStudentT");
    CHECK_THROWS_MSG(supp::run_mvdist(R"({"family":"Dirichlet","alpha":[2,3]})", "cdf", "[0.5,0.5]"), "Dirichlet");

    // A model prior may now be a composite, which no spec could express before.
    {
        const char* prior = R"({"family":"TruncatedDistribution",
            "base":{"family":"Normal","parameters":[0,1]},"bounds":[-1,1]})";
        auto d = corehydro::models::spec::build_spec_distribution(
            corehydro::models::spec::parse_json(prior));
        CHECK_NEAR(d->cdf(1.0), 1.0, 1e-9);
    }

    return chtest::summary("dist_runner");
}
