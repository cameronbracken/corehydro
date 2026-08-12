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

    // A bare-family marginal in a fit spec is MLE-fitted to the sample first, as IFM requires.
    // Without the pre-fit theta is optimized against a default Normal(0,1) and comes out wrong.
    {
        const char* fitted = R"({"family":"Clayton","fit":{
            "x":[135.9,104.1,108.7,99.3,134.7,91.0,77.3,115.4,109.0,79.0],
            "y":[1.9,1.3,1.4,1.2,1.8,1.1,0.9,1.5,1.4,1.0],
            "method":"ifm","margin_x":{"family":"Normal"},"margin_y":{"family":"Normal"}}})";
        supp::DistResult mx = supp::run_copula(fitted, "marginal_x_parameters", "[]");
        // The x sample has mean ~105.4, sd ~19: a fitted marginal, not the default (0, 1).
        CHECK_TRUE(mx.values.at(0) > 50.0);
        CHECK_TRUE(mx.values.at(1) > 5.0);
    }

    // An explicitly parameterized marginal is used as given and NOT refitted.
    {
        const char* given = R"({"family":"Clayton","fit":{
            "x":[135.9,104.1,108.7,99.3,134.7,91.0,77.3,115.4,109.0,79.0],
            "y":[1.9,1.3,1.4,1.2,1.8,1.1,0.9,1.5,1.4,1.0],
            "method":"ifm","margin_x":{"family":"Normal","parameters":[100,20]},
            "margin_y":{"family":"Normal","parameters":[1.4,0.3]}}})";
        supp::DistResult mx = supp::run_copula(given, "marginal_x_parameters", "[]");
        CHECK_NEAR(mx.values.at(0), 100.0, 1e-12);
        CHECK_NEAR(mx.values.at(1), 20.0, 1e-12);
    }

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
    // A seeded MVN CDF is reproducible. Above dimension 2 the CDF is Genz quasi-Monte-Carlo off a
    // per-instance Mersenne Twister; without a seed key in the grammar it clock-seeds and R and
    // Python cannot agree.
    {
        const char* spec = R"({"family":"MultivariateNormal","mean":[0,0,0],
            "covariance":[[1,0.3,0.2],[0.3,1,0.4],[0.2,0.4,1]],"seed":12345})";
        supp::DistResult a = supp::run_mvdist(spec, "cdf", "[1,1,1]");
        supp::DistResult b = supp::run_mvdist(spec, "cdf", "[1,1,1]");
        CHECK_NEAR(a.values.at(0), b.values.at(0), 0.0);
        CHECK_TRUE(a.values.at(0) > 0.0 && a.values.at(0) < 1.0);
    }

    // The three accuracy knobs are honoured too: a one-point budget cannot converge, so it
    // returns a different (unconverged) value than the default budget does.
    {
        const char* tight = R"({"family":"MultivariateNormal","mean":[0,0,0],
            "covariance":[[1,0.3,0.2],[0.3,1,0.4],[0.2,0.4,1]],"seed":12345,
            "max_evaluations":1,"abs_error":1e-12,"rel_error":1e-12})";
        const char* loose = R"({"family":"MultivariateNormal","mean":[0,0,0],
            "covariance":[[1,0.3,0.2],[0.3,1,0.4],[0.2,0.4,1]],"seed":12345})";
        supp::DistResult t = supp::run_mvdist(tight, "cdf", "[1,1,1]");
        supp::DistResult l = supp::run_mvdist(loose, "cdf", "[1,1,1]");
        CHECK_TRUE(std::fabs(t.values.at(0) - l.values.at(0)) > 0.0);
    }

    // A derived child inherits the parent's integrator settings. Both children below are
    // dimension 3, where the CDF is the Genz quasi-Monte-Carlo integral off the instance's own
    // Mersenne Twister; if the seed did not travel into the child spec each call would clock-seed
    // and the two figures would differ (and R and Python could not agree on either).
    {
        const char* parent = R"({"family":"MultivariateNormal","mean":[0,0,0,0],
            "covariance":[[1,0.3,0.2,0.1],[0.3,1,0.4,0.2],[0.2,0.4,1,0.3],[0.1,0.2,0.3,1]],
            "seed":12345})";
        supp::DistResult marg = supp::run_mvdist(parent, "marginal", "[0,1,2]");
        CHECK_EQ(static_cast<int>(supp::run_mvdist(marg.spec, "dimension", "[]").values.at(0)), 3);
        supp::DistResult m1 = supp::run_mvdist(marg.spec, "cdf", "[1,1,1]");
        supp::DistResult m2 = supp::run_mvdist(marg.spec, "cdf", "[1,1,1]");
        CHECK_NEAR(m1.values.at(0), m2.values.at(0), 0.0);
        CHECK_TRUE(m1.values.at(0) > 0.0 && m1.values.at(0) < 1.0);

        supp::DistResult cond = supp::run_mvdist(parent, "conditional", "[3,0.5]");
        CHECK_EQ(static_cast<int>(supp::run_mvdist(cond.spec, "dimension", "[]").values.at(0)), 3);
        supp::DistResult c1 = supp::run_mvdist(cond.spec, "cdf", "[1,1,1]");
        supp::DistResult c2 = supp::run_mvdist(cond.spec, "cdf", "[1,1,1]");
        CHECK_NEAR(c1.values.at(0), c2.values.at(0), 0.0);
        CHECK_TRUE(c1.values.at(0) > 0.0 && c1.values.at(0) < 1.0);

        // The other three integrator settings travel too: a one-evaluation budget cannot
        // converge, so the child it produces disagrees with the default-budget child.
        const char* tight = R"({"family":"MultivariateNormal","mean":[0,0,0,0],
            "covariance":[[1,0.3,0.2,0.1],[0.3,1,0.4,0.2],[0.2,0.4,1,0.3],[0.1,0.2,0.3,1]],
            "seed":12345,"max_evaluations":1,"abs_error":1e-12,"rel_error":1e-12})";
        supp::DistResult tmarg = supp::run_mvdist(tight, "marginal", "[0,1,2]");
        supp::DistResult t1 = supp::run_mvdist(tmarg.spec, "cdf", "[1,1,1]");
        CHECK_TRUE(std::fabs(t1.values.at(0) - m1.values.at(0)) > 0.0);
    }
    // A parent with no integrator settings produces a child with none either: the grammar copies
    // what is there and invents nothing.
    {
        const char* bare = R"({"family":"MultivariateNormal","mean":[1,2,3],
                               "covariance":[[1,0,0],[0,1,0],[0,0,1]]})";
        supp::DistResult r = supp::run_mvdist(bare, "marginal", "[0,1]");
        CHECK_EQ(r.spec.find("seed") == std::string::npos, true);
    }

    CHECK_THROWS_MSG(supp::run_mvdist(R"({"family":"MultivariateStudentT","df":5,"location":[0,0]})", "marginal", "[0]"), "MultivariateStudentT");
    // --- the seven accessor verbs --------------------------------------------------------
    {
        const char* mvn = R"({"family":"MultivariateNormal","mean":[1,2,3],
                              "covariance":[[4,0,0],[0,9,0],[0,0,16]]})";
        // A Gaussian's median and mode are its mean, elementwise.
        supp::DistResult mean = supp::run_mvdist(mvn, "mean", "[]");
        supp::DistResult med = supp::run_mvdist(mvn, "median", "[]");
        supp::DistResult mod = supp::run_mvdist(mvn, "mode", "[]");
        CHECK_EQ(static_cast<int>(med.values.size()), 3);
        CHECK_EQ(static_cast<int>(mod.values.size()), 3);
        CHECK_NEAR(med.values.at(1), mean.values.at(1), 0.0);
        CHECK_NEAR(mod.values.at(2), mean.values.at(2), 0.0);
        // Diagonal covariance: inverse_cdf is the elementwise univariate Normal quantile, so
        // probability 0.5 in every dimension maps back to the mean.
        supp::DistResult inv = supp::run_mvdist(mvn, "inverse_cdf", "[0.5,0.5,0.5]");
        CHECK_EQ(static_cast<int>(inv.values.size()), 3);
        CHECK_NEAR(inv.values.at(0), 1.0, 1e-9);
        CHECK_NEAR(inv.values.at(2), 3.0, 1e-9);
    }
    {
        const char* mvt = R"({"family":"MultivariateStudentT","df":7,"location":[1,2],
                              "scale":[[1,0],[0,1]]})";
        supp::DistResult df = supp::run_mvdist(mvt, "degrees_of_freedom", "[]");
        CHECK_NEAR(df.values.at(0), 7.0, 0.0);
        supp::DistResult med = supp::run_mvdist(mvt, "median", "[]");
        supp::DistResult mod = supp::run_mvdist(mvt, "mode", "[]");
        CHECK_NEAR(med.values.at(0), 1.0, 0.0);
        CHECK_NEAR(mod.values.at(1), 2.0, 0.0);
        // MVT's inverse_cdf takes Dimension + 1 probabilities (the last drives the chi-squared
        // mixing variable) and returns a point of length Dimension.
        supp::DistResult inv = supp::run_mvdist(mvt, "inverse_cdf", "[0.5,0.5,0.5]");
        CHECK_EQ(static_cast<int>(inv.values.size()), 2);
        CHECK_NEAR(inv.values.at(0), 1.0, 1e-9);
        CHECK_THROWS_MSG(supp::run_mvdist(mvt, "inverse_cdf", "[0.5,0.5]"), "Dimension");
    }
    {
        const char* dir = R"({"family":"Dirichlet","alpha":[2,3,5]})";
        supp::DistResult a = supp::run_mvdist(dir, "alpha", "[]");
        supp::DistResult s = supp::run_mvdist(dir, "alpha_sum", "[]");
        CHECK_EQ(static_cast<int>(a.values.size()), 3);
        // alpha_sum is the sum of alpha.
        CHECK_NEAR(s.values.at(0), a.values.at(0) + a.values.at(1) + a.values.at(2), 1e-12);
        supp::DistResult mod = supp::run_mvdist(dir, "mode", "[]");
        CHECK_EQ(static_cast<int>(mod.values.size()), 3);
        // Dirichlet mode_i = (alpha_i - 1) / (alpha_sum - K).
        CHECK_NEAR(mod.values.at(0), 1.0 / 7.0, 1e-12);
    }
    {
        supp::DistResult n =
            supp::run_mvdist(R"({"family":"Multinomial","trials":12,"probabilities":[0.2,0.3,0.5]})",
                             "number_of_trials", "[]");
        CHECK_NEAR(n.values.at(0), 12.0, 0.0);
    }
    // Each verb names the family it is missing from rather than returning a wrong shape.
    CHECK_THROWS_MSG(supp::run_mvdist(R"({"family":"Dirichlet","alpha":[2,3]})", "median", "[]"), "Dirichlet");
    CHECK_THROWS_MSG(supp::run_mvdist(R"({"family":"Multinomial","trials":4,"probabilities":[0.5,0.5]})", "mode", "[]"), "Multinomial");
    CHECK_THROWS_MSG(supp::run_mvdist(R"({"family":"MultivariateNormal","mean":[0,0]})", "alpha", "[]"), "MultivariateNormal");
    CHECK_THROWS_MSG(supp::run_mvdist(R"({"family":"MultivariateNormal","mean":[0,0]})", "degrees_of_freedom", "[]"), "MultivariateNormal");
    CHECK_THROWS_MSG(supp::run_mvdist(R"({"family":"Dirichlet","alpha":[2,3]})", "number_of_trials", "[]"), "Dirichlet");

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
