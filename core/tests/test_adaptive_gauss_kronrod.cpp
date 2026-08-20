// Covers the members the completed AdaptiveGaussKronrod / Integrator port added over the minimal
// port that preceded it: the settings and validation, the status and evaluation counting, the
// standard error, MinDepth, the StratificationBin overload, and the Integrator base's own
// ClearResults / UpdateStatus / EvaluateConvergence.
//
// Analytic identities only, the repo convention for a ctest (see test_dist_runner.cpp). The
// C#-pinned oracles for this class live in fixtures/callback/math.json, and the values its two
// in-tree callers produce are pinned by the VonMises and Bulletin 17C fixtures.
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/math/integration/adaptive_gauss_kronrod.hpp"

namespace agk = corehydro::numerics::math::integration;
namespace sampling = corehydro::numerics::sampling;
using agk::AdaptiveGaussKronrod;
using agk::IntegrationStatus;

// Exposes the Integrator base's protected members, which no concrete integrator routes through
// today (AdaptiveGaussKronrod assigns Status directly and rethrows in its own catch, exactly as
// the C# class does) but which the next ported integrator will.
class ProbeIntegrator : public agk::Integrator {
   public:
    void integrate() override { status_ = IntegrationStatus::Success; }
    using agk::Integrator::clear_results;
    using agk::Integrator::evaluate_convergence;
    using agk::Integrator::update_status;
    using agk::Integrator::validate;
};

int main() {
    // --- the base result surface -------------------------------------------------------------
    {
        AdaptiveGaussKronrod a([](double x) { return x * x; }, 0.0, 3.0);
        // Before any run: None, and no evaluations counted.
        CHECK_TRUE(a.status() == IntegrationStatus::None);
        CHECK_EQ(a.function_evaluations(), 0);
        CHECK_EQ(a.iterations(), 0);
        CHECK_EQ(a.min(), 0.0);
        CHECK_EQ(a.max(), 3.0);
        CHECK_EQ(a.max_depth, 100);
        CHECK_EQ(a.min_depth, 0);

        a.integrate();
        CHECK_NEAR(a.result(), 9.0, 1e-12);
        CHECK_TRUE(a.status() == IntegrationStatus::Success);
        CHECK_EQ(agk::status_name(a.status()), std::string("Success"));
        // The G10K21 rule is exact for a quadratic, so the whole-interval evaluation converges
        // with no subdivision: 1 centre point plus 10 symmetric pairs.
        CHECK_EQ(a.function_evaluations(), 21);
        CHECK_TRUE(a.standard_error() >= 0.0);
        CHECK_TRUE(a.standard_error() < 1e-10);

        // Re-running clears and reproduces, rather than accumulating.
        a.integrate();
        CHECK_EQ(a.function_evaluations(), 21);
        CHECK_NEAR(a.result(), 9.0, 1e-12);
    }

    // A function the rule is not exact for subdivides, and the tolerances steer how far.
    {
        AdaptiveGaussKronrod loose([](double x) { return std::exp(-x * x); }, -6.0, 6.0);
        loose.absolute_tolerance = 1e-3;
        loose.relative_tolerance = 1e-3;
        loose.integrate();
        AdaptiveGaussKronrod tight([](double x) { return std::exp(-x * x); }, -6.0, 6.0);
        tight.absolute_tolerance = 1e-14;
        tight.relative_tolerance = 1e-14;
        tight.integrate();
        CHECK_TRUE(tight.function_evaluations() > loose.function_evaluations());
        // Both still land on sqrt(pi), the tight one more precisely.
        CHECK_NEAR(tight.result(), std::sqrt(corehydro::numerics::kPi), 1e-12);
        CHECK_NEAR(loose.result(), std::sqrt(corehydro::numerics::kPi), 1e-3);
    }

    // MinDepth forces subdivision even where the tolerance is already met: the C# convergence
    // clause is `depth <= MaxDepth - MinDepth`, so a non-zero MinDepth withholds the tolerance
    // exit for that many levels.
    {
        AdaptiveGaussKronrod plain([](double x) { return x * x; }, 0.0, 3.0);
        plain.integrate();
        AdaptiveGaussKronrod deep([](double x) { return x * x; }, 0.0, 3.0);
        deep.min_depth = 3;
        deep.integrate();
        CHECK_EQ(plain.function_evaluations(), 21);
        CHECK_TRUE(deep.function_evaluations() > plain.function_evaluations());
        CHECK_NEAR(deep.result(), 9.0, 1e-10);
    }

    // --- the stratification-bin overload ------------------------------------------------------
    {
        std::vector<sampling::StratificationBin> bins = {
            sampling::StratificationBin(0.0, 1.0),
            sampling::StratificationBin(1.0, 2.0),
            sampling::StratificationBin(2.0, 3.0)};
        AdaptiveGaussKronrod a([](double x) { return x * x; }, 0.0, 3.0);
        a.integrate(bins);
        CHECK_NEAR(a.result(), 9.0, 1e-10);
        CHECK_TRUE(a.status() == IntegrationStatus::Success);
        // Three bins, each converging on its own whole-bin evaluation.
        CHECK_EQ(a.function_evaluations(), 63);
    }

    // --- the evaluation cap -------------------------------------------------------------------
    {
        // 1 / sqrt(x) is unbounded at 0, so the rule subdivides toward the singularity for a
        // long time (1,911 evaluations uncapped) and the cap stops it first.
        AdaptiveGaussKronrod a([](double x) { return x <= 0.0 ? 0.0 : 1.0 / std::sqrt(x); }, 0.0,
                               1.0);
        a.max_function_evaluations = 1000;
        a.integrate();
        CHECK_TRUE(a.status() == IntegrationStatus::MaximumFunctionEvaluationsReached);
        CHECK_EQ(agk::status_name(a.status()),
                 std::string("MaximumFunctionEvaluationsReached"));
        CHECK_TRUE(a.function_evaluations() >= 1000);
    }

    // --- construction and validation ----------------------------------------------------------
    {
        CHECK_THROWS_MSG(AdaptiveGaussKronrod([](double x) { return x; }, 1.0, 1.0),
                         "cannot be less than or equal to");
        CHECK_THROWS_MSG(AdaptiveGaussKronrod([](double x) { return x; }, 2.0, 1.0),
                         "cannot be less than or equal to");
        CHECK_THROWS_MSG(AdaptiveGaussKronrod(std::function<double(double)>{}, 0.0, 1.0),
                         "cannot be null");
    }
    {
        AdaptiveGaussKronrod a([](double x) { return x; }, 0.0, 1.0);
        a.absolute_tolerance = 1e-20;
        CHECK_THROWS_MSG(a.integrate(), "absolute tolerance must be between");
    }
    {
        AdaptiveGaussKronrod a([](double x) { return x; }, 0.0, 1.0);
        a.relative_tolerance = 2.0;
        CHECK_THROWS_MSG(a.integrate(), "relative tolerance must be between");
    }
    {
        AdaptiveGaussKronrod a([](double x) { return x; }, 0.0, 1.0);
        a.max_function_evaluations = 0;
        CHECK_THROWS_MSG(a.integrate(), "maximum number of function evaluations");
    }

    // --- ReportFailure ------------------------------------------------------------------------
    {
        // The default rethrows whatever the integrand raised, after latching Failure.
        AdaptiveGaussKronrod a([](double) -> double { throw std::runtime_error("integrand blew up"); },
                               0.0, 1.0);
        CHECK_THROWS_MSG(a.integrate(), "integrand blew up");
        CHECK_TRUE(a.status() == IntegrationStatus::Failure);
        CHECK_TRUE(std::isnan(a.result()));
    }
    {
        // ReportFailure = false swallows it and leaves the NaN ClearResults set -- the shape
        // MultipleGrubbsBeckTest's GGBCRITP relies on.
        AdaptiveGaussKronrod a([](double) -> double { throw std::runtime_error("integrand blew up"); },
                               0.0, 1.0);
        a.report_failure = false;
        a.integrate();
        CHECK_TRUE(a.status() == IntegrationStatus::Failure);
        CHECK_TRUE(std::isnan(a.result()));
    }

    // --- the free wrapper the two in-tree callers use ------------------------------------------
    {
        CHECK_NEAR(agk::integrate([](double x) { return x * x; }, 0.0, 3.0), 9.0, 1e-12);
        CHECK_NEAR(agk::integrate([](double x) { return std::sin(x); }, 0.0, corehydro::numerics::kPi),
                   2.0, 1e-10);
    }

    // --- the Integrator base's own protected machinery ----------------------------------------
    {
        ProbeIntegrator p;
        p.integrate();
        CHECK_TRUE(p.status() == IntegrationStatus::Success);
        p.clear_results();
        CHECK_TRUE(p.status() == IntegrationStatus::None);
        CHECK_TRUE(std::isnan(p.result()));
        CHECK_EQ(p.iterations(), 0);
        CHECK_EQ(p.function_evaluations(), 0);

        // Convergence needs BOTH the absolute and the relative test, and refuses non-finite input.
        CHECK_TRUE(p.evaluate_convergence(1.0, 1.0 + 1e-12));
        CHECK_TRUE(!p.evaluate_convergence(1.0, 1.5));
        CHECK_TRUE(!p.evaluate_convergence(std::numeric_limits<double>::quiet_NaN(), 1.0));
        CHECK_TRUE(!p.evaluate_convergence(1.0, std::numeric_limits<double>::infinity()));

        // UpdateStatus sets the status, and only rethrows for Failure with ReportFailure set.
        p.update_status(IntegrationStatus::MaximumIterationsReached);
        CHECK_TRUE(p.status() == IntegrationStatus::MaximumIterationsReached);
        CHECK_EQ(agk::status_name(p.status()), std::string("MaximumIterationsReached"));
        p.report_failure = false;
        p.update_status(IntegrationStatus::Failure,
                        std::make_exception_ptr(std::runtime_error("inner")));
        CHECK_TRUE(p.status() == IntegrationStatus::Failure);
        p.report_failure = true;
        CHECK_THROWS_MSG(p.update_status(IntegrationStatus::Failure,
                                         std::make_exception_ptr(std::runtime_error("inner"))),
                         "inner");

        // Validate rejects each bound in the C# order.
        p.min_iterations = 0;
        CHECK_THROWS_MSG(p.validate(), "minimum number of iterations");
        p.min_iterations = 1;
        p.min_function_evaluations = 0;
        CHECK_THROWS_MSG(p.validate(), "minimum number of function evaluations");
        p.min_function_evaluations = 1;
        p.max_iterations = 0;
        CHECK_THROWS_MSG(p.validate(), "maximum number of iterations");
        p.max_iterations = 10;
        p.validate();  // nothing left to reject

        // The public Result writer C# exposes over the protected field.
        p.set_result(4.5);
        CHECK_EQ(p.result(), 4.5);
    }

    return chtest::summary("adaptive_gauss_kronrod");
}
