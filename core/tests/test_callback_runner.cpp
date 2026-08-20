#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"
#include "corehydro/numerics/support/callback_runner.hpp"
#include "corehydro/numerics/support/rng_handle.hpp"

namespace sup = corehydro::numerics::support;

struct HostError : std::runtime_error {
    HostError() : std::runtime_error("host language error") {}
};

int main() {
    // A guard that never sees a throw is transparent and counts real calls.
    {
        sup::GuardedCall<double, const std::vector<double>&> g(
            [](const std::vector<double>& p) { return p[0] * 2.0; }, -1.0);
        CHECK_EQ(g({3.0}), 6.0);
        CHECK_EQ(g({4.0}), 8.0);
        CHECK_TRUE(!g.aborted());
        CHECK_EQ(g.call_count(), 2);
    }

    // The first throw latches, later calls short-circuit to the sentinel WITHOUT re-entering
    // the host function, and the stored exception rethrows with its original type.
    {
        int entries = 0;
        sup::GuardedCall<double, const std::vector<double>&> g(
            [&entries](const std::vector<double>&) -> double {
                ++entries;
                throw HostError();
            },
            -1.0);
        CHECK_EQ(g({1.0}), -1.0);
        CHECK_EQ(g({2.0}), -1.0);
        CHECK_EQ(entries, 1);         // second call never reached the host
        CHECK_EQ(g.call_count(), 0);  // no call completed
        CHECK_TRUE(g.aborted());
        CHECK_THROWS_MSG(g.rethrow_if_aborted(), "host language error");
    }

    // Two guards SHARING an abort state abort as one. This is the case a group with more than one
    // live callback needs (bootstrap's four delegates; the Gibbs proposal beside the
    // log-likelihood): a throw inside one must stop the ported algorithm from re-entering the host
    // through the other while an unwind is already pending.
    {
        auto state = sup::make_abort_state();
        int likelihood_entries = 0, proposal_entries = 0;
        sup::GuardedCall<double, const std::vector<double>&> likelihood(
            [&likelihood_entries](const std::vector<double>&) -> double {
                ++likelihood_entries;
                throw HostError();
            },
            -std::numeric_limits<double>::infinity(), state);
        sup::GuardedCall<std::vector<double>, const std::vector<double>&> proposal(
            [&proposal_entries](const std::vector<double>& p) {
                ++proposal_entries;
                return p;
            },
            std::vector<double>{}, state);

        CHECK_TRUE(likelihood({1.0}) == -std::numeric_limits<double>::infinity());
        CHECK_EQ(likelihood_entries, 1);
        // The OTHER guard is now short-circuited: its sentinel comes back and its host function
        // is never entered.
        CHECK_TRUE(proposal({1.0, 2.0}).empty());
        CHECK_EQ(proposal_entries, 0);
        CHECK_EQ(proposal.call_count(), 0);
        CHECK_TRUE(proposal.aborted());
        // And either guard rethrows the one stored exception.
        CHECK_THROWS_MSG(proposal.rethrow_if_aborted(), "host language error");
        CHECK_THROWS_MSG(likelihood.rethrow_if_aborted(), "host language error");

        // A later latch never overwrites the first: the first exception is the cause, and this is
        // what makes the message the user sees deterministic when several callbacks are live.
        state->latch(std::make_exception_ptr(std::runtime_error("a later error")));
        CHECK_THROWS_MSG(likelihood.rethrow_if_aborted(), "host language error");
    }

    // A standalone guard is unaffected by another standalone guard's abort -- the private-state
    // default that the math group and the optimizer surface rely on.
    {
        int entries = 0;
        sup::GuardedCall<double, const std::vector<double>&> a(
            [](const std::vector<double>&) -> double { throw HostError(); }, -1.0);
        sup::GuardedCall<double, const std::vector<double>&> b(
            [&entries](const std::vector<double>& p) {
                ++entries;
                return p[0];
            },
            -1.0);
        CHECK_EQ(a({1.0}), -1.0);
        CHECK_TRUE(a.aborted());
        CHECK_EQ(b({7.0}), 7.0);
        CHECK_EQ(entries, 1);
        CHECK_TRUE(!b.aborted());
        b.rethrow_if_aborted();  // nothing to rethrow
    }

    // A vector-returning guard (the shape the Gibbs proposal and bootstrap resample need).
    {
        sup::GuardedCall<std::vector<double>, const std::vector<double>&> g(
            [](const std::vector<double>& p) { return std::vector<double>{p[0], p[0]}; },
            std::vector<double>{});
        CHECK_EQ(g({5.0}).size(), std::size_t{2});
        CHECK_TRUE(!g.aborted());
    }

    // --- math group -----------------------------------------------------------------------
    //
    // Analytic identities only (the repo convention for a ctest -- see test_dist_runner.cpp);
    // the C#-derived oracle values for this group live in fixtures/callback/math.json.
    {
        sup::CallbackSet cbs;
        cbs.scalar = [](double x) { return x * x - 2.0; };
        sup::CallbackResult r =
            sup::run_callback("math", "root_find", R"({"lower": 0.0, "upper": 2.0})", cbs);
        CHECK_NEAR(r.values.at(0), 1.4142135623730951, 1e-8);
        CHECK_EQ(r.names.at(0), std::string("root"));
    }
    {
        // f(x) = x^3, f'(2) = 12.
        sup::CallbackSet cbs;
        cbs.scalar = [](double x) { return x * x * x; };
        sup::CallbackResult r = sup::run_callback("math", "derivative", R"({"point": 2.0})", cbs);
        CHECK_NEAR(r.values.at(0), 12.0, 1e-6);
    }
    {
        sup::CallbackSet cbs;
        cbs.vector_scalar = [](const std::vector<double>& p) {
            return (1.0 - p[0]) * (1.0 - p[0]) + 100.0 * (p[1] - p[0] * p[0]) * (p[1] - p[0] * p[0]);
        };
        sup::CallbackResult r =
            sup::run_callback("math", "gradient", R"({"point": [1.0, 1.0]})", cbs);
        CHECK_EQ(r.values.size(), std::size_t{2});
        CHECK_NEAR(r.values.at(0), 0.0, 1e-6);
        CHECK_NEAR(r.values.at(1), 0.0, 1e-6);
        CHECK_EQ(r.dims.at(0), 2);
    }
    {
        // f(x,y) = x^2 + 2y^2 + xy: H = [[2, 1], [1, 4]], constant everywhere.
        sup::CallbackSet cbs;
        cbs.vector_scalar = [](const std::vector<double>& p) {
            return p[0] * p[0] + 2.0 * p[1] * p[1] + p[0] * p[1];
        };
        sup::CallbackResult r =
            sup::run_callback("math", "hessian", R"({"point": [1.0, 2.0]})", cbs);
        CHECK_EQ(r.dims.at(0), 2);
        CHECK_EQ(r.dims.at(1), 2);
        CHECK_NEAR(r.values.at(0), 2.0, 1e-3);
        CHECK_NEAR(r.values.at(1), 1.0, 1e-3);
        CHECK_NEAR(r.values.at(2), 1.0, 1e-3);
        CHECK_NEAR(r.values.at(3), 4.0, 1e-3);
    }

    {
        // The integral of x^2 over [0, 3] is 9, exactly (the G10K21 rule is exact for a
        // quadratic on the first evaluation).
        sup::CallbackSet cbs;
        cbs.scalar = [](double x) { return x * x; };
        sup::CallbackResult r =
            sup::run_callback("math", "quadrature", R"({"lower": 0.0, "upper": 3.0})", cbs);
        CHECK_NEAR(r.values.at(0), 9.0, 1e-10);
        CHECK_EQ(r.names.at(0), std::string("integral"));
        CHECK_EQ(r.names.at(1), std::string("function_evaluations"));
        CHECK_EQ(r.names.at(2), std::string("standard_error"));
        CHECK_EQ(r.status, std::string("Success"));
        CHECK_TRUE(r.values.at(1) > 0.0);  // function evaluations were counted
        // G10K21 is exact for a quadratic, so the Gauss and Kronrod estimates agree, nothing is
        // subdivided and the ported StandardError is zero. The subdividing case below is where it
        // is not.
        CHECK_EQ(r.values.at(2), 0.0);
    }
    {
        // The one case here that reaches the SUBDIVIDING branch: a Lorentzian peak of half-width
        // 0.01 over [-1, 1] (the fixture's Quad_Peak). The C#-pinned oracle for the same run is
        // fixtures/callback/math.json's quadrature_peak_subdivides; this asserts the two
        // properties that make it worth having -- many more than one panel's 21 evaluations, and
        // a standard error the run actually accumulated.
        sup::CallbackSet cbs;
        cbs.scalar = [](double x) { return 1.0 / (1.0 + 1.0e4 * x * x); };
        sup::CallbackResult r = sup::run_callback(
            "math", "quadrature", R"({"lower": -1.0, "upper": 1.0})", cbs);
        CHECK_EQ(r.status, std::string("Success"));
        CHECK_TRUE(r.values.at(1) > 21.0);
        CHECK_TRUE(r.values.at(2) > 0.0);
    }
    {
        // sin(x) over [0, 1] = 1 - cos(1), and the tolerances are settable.
        sup::CallbackSet cbs;
        cbs.scalar = [](double x) { return std::sin(x); };
        sup::CallbackResult r = sup::run_callback(
            "math", "quadrature",
            R"({"lower": 0.0, "upper": 1.0, "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12})",
            cbs);
        CHECK_NEAR(r.values.at(0), 1.0 - std::cos(1.0), 1e-12);
    }
    {
        // A cap low enough to be hit reports the status the ported Integrator base defines
        // rather than "Success". 1/sqrt(x) is unbounded at 0, so the rule never converges.
        sup::CallbackSet cbs;
        cbs.scalar = [](double x) { return x <= 0.0 ? 0.0 : 1.0 / std::sqrt(x); };
        sup::CallbackResult r = sup::run_callback(
            "math", "quadrature",
            R"({"lower": 0.0, "upper": 1.0, "max_function_evaluations": 1000})", cbs);
        CHECK_EQ(r.status, std::string("MaximumFunctionEvaluationsReached"));
        CHECK_TRUE(r.values.at(1) >= 1000.0);
    }

    // A host exception inside the callback survives the ported algorithm and reaches the caller,
    // on EVERY method -- not just one. The guard's sentinel is a value each ported routine can
    // itself reject (NaN drives Brent to its "failed to find root" throw; -inf trips
    // NumericalDerivative's "f(theta) is not finite" domain_error), so a method whose drive site
    // is not wrapped in try/catch reports the INTERNAL error instead of the user's own.
    {
        sup::CallbackSet cbs;
        cbs.scalar = [](double) -> double { throw HostError(); };
        CHECK_THROWS_MSG(
            sup::run_callback("math", "root_find", R"({"lower": 0.0, "upper": 2.0})", cbs),
            "host language error");
        CHECK_THROWS_MSG(sup::run_callback("math", "derivative", R"({"point": 2.0})", cbs),
                         "host language error");
        CHECK_THROWS_MSG(
            sup::run_callback("math", "quadrature",
                              R"({"lower": 0.0, "upper": 3.0, "max_function_evaluations": 5000})",
                              cbs),
            "host language error");
    }
    {
        sup::CallbackSet cbs;
        cbs.vector_scalar = [](const std::vector<double>&) -> double { throw HostError(); };
        CHECK_THROWS_MSG(sup::run_callback("math", "gradient", R"({"point": [1.0, 1.0]})", cbs),
                         "host language error");
        CHECK_THROWS_MSG(sup::run_callback("math", "hessian", R"({"point": [1.0, 1.0]})", cbs),
                         "host language error");
    }

    // Dispatch errors: an unknown group, an unknown method, a missing callback, a missing option.
    {
        sup::CallbackSet cbs;
        cbs.scalar = [](double x) { return x; };
        CHECK_THROWS_MSG(sup::run_callback("nope", "root_find", "{}", cbs),
                         "unknown callback group");
        CHECK_THROWS_MSG(sup::run_callback("math", "nope", "{}", cbs), "unknown math method");
        CHECK_THROWS_MSG(sup::run_callback("math", "gradient", R"({"point": [1.0]})", cbs),
                         "requires a vector function");
        CHECK_THROWS_MSG(sup::run_callback("math", "root_find", R"({"lower": 0.0})", cbs),
                         "requires the option 'upper'");
    }

    // --- mcmc group -----------------------------------------------------------------------
    //
    // Analytic/structural properties only (the repo convention for a ctest); the C#-pinned
    // oracle values for this group live in fixtures/callback/mcmc.json.
    {
        // A Gaussian log-density over five observations, and two chains recovering their mean.
        // Written with `+ - * /` only so the run is IEEE-deterministic in every language -- see
        // the cross-language note in R/callback.R's mcmc_posterior().
        const std::vector<double> data = {4.9, 5.1, 5.0, 5.2, 4.8};
        sup::CallbackSet cbs;
        cbs.vector_scalar = [&data](const std::vector<double>& p) {
            double acc = 0.0;
            for (double x : data) acc += (x - p[0]) * (x - p[0]);
            return -0.5 * acc;
        };
        const std::string options = R"({
            "sampler": "RWMH", "iterations": 500, "warmup": 100, "chains": 2, "thinning": 1,
            "seed": 12345, "output_length": 100, "initialize": "Randomize",
            "proposal_sigma": "identity",
            "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})";
        sup::CallbackResult r1 = sup::run_callback("mcmc", "sample", options, cbs);
        sup::CallbackResult r2 = sup::run_callback("mcmc", "sample", options, cbs);
        CHECK_EQ(r1.values, r2.values);  // a seeded run is deterministic
        CHECK_EQ(r1.status, std::string("Success"));

        // dims = {n_summary, n_chains, n_draws, n_parameters}; n_summary = 1 + chains + 8 * p.
        CHECK_EQ(r1.dims.size(), std::size_t{4});
        CHECK_EQ(r1.dims.at(0), 11);
        CHECK_EQ(r1.dims.at(1), 2);
        CHECK_EQ(r1.dims.at(2), 500);
        CHECK_EQ(r1.dims.at(3), 1);
        // The summary block is named and the draw block follows it unnamed.
        CHECK_EQ(r1.names.size(), std::size_t{11});
        CHECK_EQ(r1.names.at(0), std::string("map_fitness"));
        CHECK_EQ(r1.values.size(), std::size_t{11} + 2 * 500 * 1);

        // The posterior mean of mu lands on the data mean, 5.0.
        std::size_t mean_at = 0;
        for (std::size_t i = 0; i < r1.names.size(); ++i)
            if (r1.names[i] == "posterior_mean[0]") mean_at = i;
        CHECK_TRUE(mean_at > 0);
        CHECK_NEAR(r1.values.at(mean_at), 5.0, 0.2);
    }
    {
        // Every sampler the group builds runs against the same user log-likelihood. Gibbs is the
        // one exception and says so: its proposal callback lands in the next task.
        const std::vector<double> data = {4.9, 5.1, 5.0, 5.2, 4.8};
        sup::CallbackSet cbs;
        cbs.vector_scalar = [&data](const std::vector<double>& p) {
            double acc = 0.0;
            for (double x : data) acc += (x - p[0]) * (x - p[0]);
            return -0.5 * acc;
        };
        // The chain count is not free to choose: the ported DEMCz/DEMCzs demand at least three,
        // and SNIS draws independently rather than running chains at all (its constructor forces
        // one chain and no warm-up, and its own ValidateSettings replaces the base class's).
        const std::vector<std::pair<const char*, const char*>> samplers = {
            {"RWMH", R"("chains": 2, "warmup": 50,)"},
            {"ARWMH", R"("chains": 2, "warmup": 50,)"},
            {"DEMCz", R"("chains": 3, "warmup": 50,)"},
            {"DEMCzs", R"("chains": 3, "warmup": 50,)"},
            {"HMC", R"("chains": 2, "warmup": 50,)"},
            {"NUTS", R"("chains": 2, "warmup": 50,)"},
            {"SNIS", R"("chains": 1,)"},
        };
        for (const auto& entry : samplers) {
            const char* sampler = entry.first;
            std::string options = std::string(R"({"sampler": ")") + sampler + R"(", )" +
                                  entry.second +
                                  R"( "iterations": 100,
                                     "thinning": 1, "seed": 12345, "output_length": 100,
                                     "initialize": "Randomize", "proposal_sigma": "identity",
                                     "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})";
            sup::CallbackResult r = sup::run_callback("mcmc", "sample", options, cbs);
            CHECK_EQ(r.dims.at(3), 1);
            CHECK_TRUE(std::isfinite(r.values.at(0)));
        }
        CHECK_THROWS_MSG(
            sup::run_callback("mcmc", "sample",
                              R"({"sampler": "Gibbs", "iterations": 100, "warmup": 50,
                                  "output_length": 100, "thinning": 1, "seed": 12345,
                                  "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})",
                              cbs),
            "requires a proposal function");
    }

    // --- the Gibbs proposal ----------------------------------------------------------------
    //
    // A model whose full conditional really is uniform, so the proposal is an EXACT Gibbs step
    // (draw from the conditional, accept unconditionally) written with arithmetic alone: x_i ~
    // Uniform(mu - 1, mu + 1), whose conditional for mu under a flat prior is Uniform(max(x) - 1,
    // min(x) + 1). For {4.9, 5.1, 5.0, 5.2, 4.8} that is Uniform(4.2, 5.8), mean exactly 5. The
    // C#-pinned oracle for this same model lives in fixtures/callback/mcmc.json.
    {
        const std::vector<double> data = {4.9, 5.1, 5.0, 5.2, 4.8};
        sup::CallbackSet cbs;
        cbs.vector_scalar = [&data](const std::vector<double>& p) {
            for (double x : data)
                if (x - p[0] > 1.0 || p[0] - x > 1.0)
                    return -std::numeric_limits<double>::infinity();
            return 0.0;
        };
        cbs.vector_rng = [&data](const std::vector<double>&,
                                 corehydro::numerics::sampling::MersenneTwister& prng) {
            sup::RngScope scope(prng);
            double lo = data[0] - 1.0, hi = data[0] + 1.0;
            for (double x : data) {
                if (x - 1.0 > lo) lo = x - 1.0;
                if (x + 1.0 < hi) hi = x + 1.0;
            }
            return std::vector<double>{lo + scope.handle()->uniform(1).at(0) * (hi - lo)};
        };
        const std::string options = R"({
            "sampler": "Gibbs", "iterations": 500, "warmup": 100, "chains": 1, "thinning": 1,
            "seed": 12345, "output_length": 100, "initialize": "Randomize",
            "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})";
        sup::CallbackResult r1 = sup::run_callback("mcmc", "sample", options, cbs);
        sup::CallbackResult r2 = sup::run_callback("mcmc", "sample", options, cbs);
        // A seeded Gibbs run is deterministic. Compared value by value rather than with CHECK_EQ
        // on the whole vector, because Gibbs runs ONE chain by construction and Gelman-Rubin needs
        // at least two, so rhat is NaN here -- a legitimate NaN, but one that compares unequal to
        // itself and would make a whole-vector equality check fail on an identical run.
        CHECK_EQ(r1.values.size(), r2.values.size());
        bool identical = true;
        for (std::size_t i = 0; i < r1.values.size(); ++i) {
            const double a = r1.values[i], b = r2.values[i];
            if (!(a == b || (std::isnan(a) && std::isnan(b)))) identical = false;
        }
        CHECK_TRUE(identical);
        CHECK_EQ(r1.dims.at(1), 1);  // the ctor forces one chain
        CHECK_EQ(r1.dims.at(2), 500);
        std::size_t mean_at = 0, sd_at = 0;
        for (std::size_t i = 0; i < r1.names.size(); ++i) {
            if (r1.names[i] == "posterior_mean[0]") mean_at = i;
            if (r1.names[i] == "posterior_sd[0]") sd_at = i;
        }
        // Every draw is an independent Uniform(4.2, 5.8): mean 5, sd 1.6 / sqrt(12) = 0.4619.
        CHECK_NEAR(r1.values.at(mean_at), 5.0, 0.15);
        CHECK_NEAR(r1.values.at(sd_at), 0.4619, 0.05);
        // Every state the chain recorded lies inside the conditional's support, which is the
        // property that says the proposal -- not some default -- produced them.
        for (std::size_t i = static_cast<std::size_t>(r1.dims.at(0)); i < r1.values.size(); ++i)
            CHECK_TRUE(r1.values[i] >= 4.2 && r1.values[i] <= 5.8);

        // A proposal returning the wrong number of values is refused by name rather than left to
        // whatever the sampler does with a short vector.
        sup::CallbackSet wrong = cbs;
        wrong.vector_rng = [](const std::vector<double>&,
                              corehydro::numerics::sampling::MersenneTwister&) {
            return std::vector<double>{1.0, 2.0};
        };
        CHECK_THROWS_MSG(sup::run_callback("mcmc", "sample", options, wrong),
                         "must return one value per parameter");

        // And it belongs to Gibbs alone.
        CHECK_THROWS_MSG(
            sup::run_callback("mcmc", "sample",
                              R"({"sampler": "RWMH", "iterations": 100, "warmup": 50,
                                  "output_length": 100, "thinning": 1, "seed": 12345,
                                  "proposal_sigma": "identity",
                                  "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})",
                              cbs),
            "only used by the Gibbs sampler");
    }

    // --- the HMC/NUTS gradient -------------------------------------------------------------
    {
        // The Gaussian kernel again, now with its analytic gradient d/dmu = sum(x - mu).
        const std::vector<double> data = {4.9, 5.1, 5.0, 5.2, 4.8};
        sup::CallbackSet cbs;
        cbs.vector_scalar = [&data](const std::vector<double>& p) {
            double acc = 0.0;
            for (double x : data) acc += (x - p[0]) * (x - p[0]);
            return -0.5 * acc;
        };
        int gradient_calls = 0;
        cbs.vector_vector = [&data, &gradient_calls](const std::vector<double>& p) {
            ++gradient_calls;
            double acc = 0.0;
            for (double x : data) acc += x - p[0];
            return std::vector<double>{acc};
        };
        for (const char* sampler : {"HMC", "NUTS"}) {
            gradient_calls = 0;
            std::string options = std::string(R"({"sampler": ")") + sampler +
                                  R"(", "iterations": 200, "warmup": 50, "chains": 2,
                                     "thinning": 1, "seed": 12345, "output_length": 100,
                                     "initialize": "Randomize",
                                     "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})";
            sup::CallbackResult r = sup::run_callback("mcmc", "sample", options, cbs);
            CHECK_TRUE(gradient_calls > 0);  // the user's gradient really was the one used
            std::size_t mean_at = 0;
            for (std::size_t i = 0; i < r.names.size(); ++i)
                if (r.names[i] == "posterior_mean[0]") mean_at = i;
            CHECK_NEAR(r.values.at(mean_at), 5.0, 0.3);

            // Supplying NO gradient still runs, on the ported bound-aware finite-difference
            // default the constructors install.
            sup::CallbackSet no_gradient;
            no_gradient.vector_scalar = cbs.vector_scalar;
            sup::CallbackResult d = sup::run_callback("mcmc", "sample", options, no_gradient);
            CHECK_NEAR(d.values.at(mean_at), 5.0, 0.3);
            // A DELIBERATELY WRONG gradient is what proves the supplied function drives the
            // leapfrog rather than being quietly ignored. Comparing the analytic run with the
            // default one would not: the central difference of a quadratic log-density is exact
            // up to rounding, so those two agree to about 4e-16 (the fixture's two HMC cases say
            // the same thing with real numbers). Half the true gradient rather than zero: a zero
            // gradient leaves the momentum unturned, so NUTS never detects a U-turn and builds its
            // full 2^10 tree every iteration -- correct behaviour, but a million callbacks for one
            // assertion.
            sup::CallbackSet wrong_gradient;
            wrong_gradient.vector_scalar = cbs.vector_scalar;
            wrong_gradient.vector_vector = [&data](const std::vector<double>& p) {
                double acc = 0.0;
                for (double x : data) acc += x - p[0];
                return std::vector<double>{0.5 * acc};
            };
            sup::CallbackResult w = sup::run_callback("mcmc", "sample", options, wrong_gradient);
            CHECK_TRUE(w.values != d.values);
        }
        // A gradient of the wrong length, and a gradient handed to a sampler that has no use for
        // one, are both refused by name.
        sup::CallbackSet wrong = cbs;
        wrong.vector_vector = [](const std::vector<double>&) {
            return std::vector<double>{1.0, 2.0};
        };
        CHECK_THROWS_MSG(
            sup::run_callback("mcmc", "sample",
                              R"({"sampler": "HMC", "iterations": 100, "warmup": 50, "chains": 2,
                                  "thinning": 1, "seed": 12345, "output_length": 100,
                                  "initialize": "Randomize",
                                  "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})",
                              wrong),
            "must return one value per parameter");
        CHECK_THROWS_MSG(
            sup::run_callback("mcmc", "sample",
                              R"({"sampler": "DEMCz", "iterations": 100, "warmup": 50,
                                  "chains": 3, "thinning": 1, "seed": 12345,
                                  "output_length": 100, "initialize": "Randomize",
                                  "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})",
                              cbs),
            "only used by the HMC and NUTS samplers");
    }

    // --- one abort state across the whole group ---------------------------------------------
    //
    // THE property the shared state buys, and nothing else in this file proves it: when the
    // PROPOSAL throws, the log-likelihood guard must short-circuit too, so the sampler cannot
    // re-enter the host through it while an unwind is already pending. With private abort states
    // this test fails loudly rather than subtly -- the log-likelihood below reports being called
    // after the proposal has already thrown, and its message, not the proposal's, is what
    // surfaces. The same check is made from the gradient's side.
    {
        bool proposal_aborted = false;
        sup::CallbackSet cbs;
        cbs.vector_scalar = [&proposal_aborted](const std::vector<double>&) -> double {
            if (proposal_aborted)
                throw std::runtime_error("the log-likelihood was re-entered after the abort");
            return -1.0;
        };
        cbs.vector_rng = [&proposal_aborted](const std::vector<double>&,
                                             corehydro::numerics::sampling::MersenneTwister&)
            -> std::vector<double> {
            proposal_aborted = true;
            throw HostError();
        };
        CHECK_THROWS_MSG(
            sup::run_callback("mcmc", "sample",
                              R"({"sampler": "Gibbs", "iterations": 100, "warmup": 50,
                                  "output_length": 100, "thinning": 1, "seed": 12345,
                                  "initialize": "Randomize",
                                  "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})",
                              cbs),
            "host language error");

        bool gradient_aborted = false;
        sup::CallbackSet grad;
        grad.vector_scalar = [&gradient_aborted](const std::vector<double>&) -> double {
            if (gradient_aborted)
                throw std::runtime_error("the log-likelihood was re-entered after the abort");
            return -1.0;
        };
        grad.vector_vector =
            [&gradient_aborted](const std::vector<double>&) -> std::vector<double> {
            gradient_aborted = true;
            throw HostError();
        };
        for (const char* sampler : {"HMC", "NUTS"}) {
            gradient_aborted = false;
            std::string options = std::string(R"({"sampler": ")") + sampler +
                                  R"(", "iterations": 100, "warmup": 50, "chains": 2,
                                     "thinning": 1, "seed": 12345, "output_length": 100,
                                     "initialize": "Randomize",
                                     "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})";
            CHECK_THROWS_MSG(sup::run_callback("mcmc", "sample", options, grad),
                             "host language error");
        }
    }
    {
        // A host exception inside the log-likelihood reaches the caller on BOTH initialization
        // paths, and they fail differently, which is why the guard is used in pairs. Under
        // "Randomize" nothing throws at all: -infinity is a legal (always rejected) fitness, so
        // the chain runs to completion and only the TRAILING rethrow catches the abort. Under
        // "MAP" the DifferentialEvolution pass is handed nothing but -infinity and can throw from
        // its own internals first, which the wrap catches.
        //
        // This loops the SAME seven-sampler vector the passing-callback test above uses (RWMH,
        // ARWMH, DEMCz, DEMCzs, HMC, NUTS, SNIS), not just RWMH: a guard wired for one sampler's
        // arm and silently missing on another is exactly the shape of bug a previous phase
        // shipped, and a single-sampler throwing test cannot catch it.
        sup::CallbackSet cbs;
        cbs.vector_scalar = [](const std::vector<double>&) -> double { throw HostError(); };
        const std::vector<std::pair<const char*, const char*>> samplers = {
            {"RWMH", R"("chains": 2, "warmup": 50,)"},
            {"ARWMH", R"("chains": 2, "warmup": 50,)"},
            {"DEMCz", R"("chains": 3, "warmup": 50,)"},
            {"DEMCzs", R"("chains": 3, "warmup": 50,)"},
            {"HMC", R"("chains": 2, "warmup": 50,)"},
            {"NUTS", R"("chains": 2, "warmup": 50,)"},
            {"SNIS", R"("chains": 1,)"},
        };
        for (const auto& entry : samplers) {
            const char* sampler = entry.first;
            for (const char* init : {"Randomize", "MAP"}) {
                std::string options = std::string(R"({"sampler": ")") + sampler + R"(", )" +
                                      entry.second +
                                      R"( "iterations": 100, "thinning": 1, "seed": 12345,
                                         "output_length": 100, "proposal_sigma": "identity",
                                         "initialize": ")" + init + R"(",
                                         "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})";
                CHECK_THROWS_MSG(sup::run_callback("mcmc", "sample", options, cbs),
                                 "host language error");
            }
        }
        // The eighth sampler, which needs a proposal before it will run at all. Its proposal here
        // is a working one, so what throws is the log-likelihood -- the same check the seven above
        // make, extended to the arm that has two live callbacks.
        sup::CallbackSet gibbs = cbs;
        gibbs.vector_rng = [](const std::vector<double>& p,
                              corehydro::numerics::sampling::MersenneTwister& prng) {
            sup::RngScope scope(prng);
            return std::vector<double>{p[0] + scope.handle()->uniform(1).at(0) - 0.5};
        };
        for (const char* init : {"Randomize", "MAP"}) {
            std::string options = std::string(R"({"sampler": "Gibbs", "iterations": 100,
                                     "warmup": 50, "thinning": 1, "seed": 12345,
                                     "output_length": 100, "initialize": ")") + init + R"(",
                                     "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})";
            CHECK_THROWS_MSG(sup::run_callback("mcmc", "sample", options, gibbs),
                             "host language error");
        }
    }
    {
        // Dispatch and validation errors.
        sup::CallbackSet empty;
        sup::CallbackSet cbs;
        cbs.vector_scalar = [](const std::vector<double>& p) { return -p[0] * p[0]; };
        CHECK_THROWS_MSG(sup::run_callback("mcmc", "sample", "{}", empty),
                         "requires a log-likelihood function");
        CHECK_THROWS_MSG(sup::run_callback("mcmc", "nope", "{}", cbs), "unknown mcmc method");
        CHECK_THROWS_MSG(sup::run_callback("mcmc", "sample", "{}", cbs),
                         "requires the option 'priors'");
        CHECK_THROWS_MSG(sup::run_callback("mcmc", "sample", R"({"priors": []})", cbs),
                         "at least one prior");
        CHECK_THROWS_MSG(
            sup::run_callback("mcmc", "sample",
                              R"({"sampler": "Nope",
                                  "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}]})",
                              cbs),
            "unknown MCMC sampler");
    }

    // --- bootstrap group --------------------------------------------------------------------
    //
    // Analytic/structural properties only (the repo convention for a ctest); the C#-pinned oracle
    // values for this group live in fixtures/callback/bootstrap.json. The model throughout is the
    // plainest one there is -- an iid resample of a fixed dataset, fitted by its mean -- written
    // with `+ - * /` and an explicit loop, so every runner agrees bit for bit.
    {
        const std::vector<double> data = {4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7};
        double sample_mean = 0.0;
        for (double x : data) sample_mean += x;
        sample_mean /= static_cast<double>(data.size());

        sup::CallbackSet cbs;
        cbs.data_rng = [](const std::vector<double>& d, const std::vector<double>&,
                          corehydro::numerics::sampling::MersenneTwister& prng) {
            sup::RngScope scope(prng);
            const int n = static_cast<int>(d.size());
            std::vector<double> out;
            out.reserve(d.size());
            for (int k : scope.handle()->integers(n, 0, n))
                out.push_back(d[static_cast<std::size_t>(k)]);
            return out;
        };
        cbs.data_vector = [](const std::vector<double>& d) {
            double acc = 0.0;
            for (double x : d) acc += x;
            return std::vector<double>{acc / static_cast<double>(d.size())};
        };
        cbs.vector_vector = [](const std::vector<double>& p) { return p; };
        cbs.data_index = [](const std::vector<double>& d, int index) {
            std::vector<double> out;
            out.reserve(d.size() - 1);
            for (std::size_t i = 0; i < d.size(); ++i)
                if (static_cast<int>(i) != index) out.push_back(d[i]);
            return out;
        };

        const std::string options =
            R"({"data": [4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7], "replicates": 200,
                "seed": 12345, "alpha": 0.1, "ci_method": "Percentile"})";
        sup::CallbackResult r1 = sup::run_callback("bootstrap", "run", options, cbs);
        sup::CallbackResult r2 = sup::run_callback("bootstrap", "run", options, cbs);
        CHECK_EQ(r1.values, r2.values);  // a seeded run is deterministic
        CHECK_EQ(r1.status, std::string("Success"));

        auto named = [](const sup::CallbackResult& r, const std::string& want) {
            for (std::size_t i = 0; i < r.names.size(); ++i)
                if (r.names[i] == want) return r.values.at(i);
            throw std::runtime_error("no result named " + want);
        };
        // The interval brackets the sample mean, which is what a bootstrap of the mean is for.
        CHECK_TRUE(named(r1, "statistic_lower[0]") < sample_mean);
        CHECK_TRUE(sample_mean < named(r1, "statistic_upper[0]"));
        CHECK_NEAR(named(r1, "statistic[0]"), sample_mean, 1e-12);
        CHECK_EQ(named(r1, "failed_replicates"), 0.0);
        CHECK_EQ(named(r1, "replicates"), 200.0);
        // dims = {n_statistics, n_parameters}.
        CHECK_EQ(r1.dims.size(), std::size_t{2});
        CHECK_EQ(r1.dims.at(0), 1);
        CHECK_EQ(r1.dims.at(1), 1);
        // Every CI method runs, and the four that need no extra machinery agree on the point
        // estimate while differing on the bounds.
        for (const char* ci : {"Percentile", "BiasCorrected", "Normal", "BCa"}) {
            std::string opts =
                R"({"data": [4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7], "replicates": 200,
                    "seed": 12345, "alpha": 0.1, "ci_method": ")" +
                std::string(ci) + R"("})";
            sup::CallbackResult r = sup::run_callback("bootstrap", "run", opts, cbs);
            CHECK_NEAR(named(r, "statistic[0]"), sample_mean, 1e-12);
            CHECK_TRUE(named(r, "statistic_lower[0]") < named(r, "statistic_upper[0]"));
        }
        // BootstrapT runs the studentized workflow instead of the regular one, so it is driven
        // with a small inner count -- every inner replicate is another crossing into the host.
        {
            sup::CallbackResult r = sup::run_callback(
                "bootstrap", "run",
                R"({"data": [4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7], "replicates": 40,
                    "inner_replicates": 20, "seed": 12345, "alpha": 0.1,
                    "ci_method": "BootstrapT"})",
                cbs);
            CHECK_TRUE(named(r, "statistic_lower[0]") < named(r, "statistic_upper[0]"));
        }

        // A statistic of more than one value is labelled and bounded per statistic.
        {
            sup::CallbackSet two = cbs;
            two.vector_vector = [](const std::vector<double>& p) {
                return std::vector<double>{p[0], p[0] * p[0]};
            };
            sup::CallbackResult r = sup::run_callback("bootstrap", "run", options, two);
            CHECK_EQ(r.dims.at(0), 2);
            CHECK_NEAR(named(r, "statistic[1]"), sample_mean * sample_mean, 1e-12);
            CHECK_TRUE(named(r, "statistic_lower[1]") < named(r, "statistic_upper[1]"));
        }

        // `parameters` supplied explicitly is used as the original parameter set rather than
        // fitting the original data -- and the population estimate says so.
        {
            sup::CallbackResult r = sup::run_callback(
                "bootstrap", "run",
                R"({"data": [4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7], "replicates": 50,
                    "seed": 12345, "parameters": [7.0]})",
                cbs);
            CHECK_EQ(named(r, "statistic[0]"), 7.0);
        }

        // BCa needs the jackknife delegate, and the refusal must come BEFORE the first replicate
        // rather than after minutes of computation: the ported class only checks it inside
        // GetConfidenceIntervals, i.e. after the whole run. The resample counter is the proof.
        {
            int resample_calls = 0;
            sup::CallbackSet no_jackknife;
            no_jackknife.data_rng = [&resample_calls](
                                        const std::vector<double>& d, const std::vector<double>&,
                                        corehydro::numerics::sampling::MersenneTwister&) {
                ++resample_calls;
                return d;
            };
            no_jackknife.data_vector = cbs.data_vector;
            no_jackknife.vector_vector = cbs.vector_vector;
            CHECK_THROWS_MSG(
                sup::run_callback("bootstrap", "run",
                                  R"({"data": [4.1, 5.2, 4.8, 5.5], "replicates": 200,
                                      "seed": 12345, "ci_method": "BCa"})",
                                  no_jackknife),
                "jackknife");
            CHECK_EQ(resample_calls, 0);
        }

        // A host exception inside EACH of the four delegates reaches the caller. One test per
        // delegate: a guard wired for one and missing on another is exactly the shape of bug a
        // wrapper this wide invites, and a test that only makes `fit` throw cannot catch it.
        {
            sup::CallbackSet t = cbs;
            t.data_rng = [](const std::vector<double>&, const std::vector<double>&,
                            corehydro::numerics::sampling::MersenneTwister&)
                -> std::vector<double> { throw HostError(); };
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", options, t),
                             "host language error");
        }
        {
            sup::CallbackSet t = cbs;
            t.data_vector = [](const std::vector<double>&) -> std::vector<double> {
                throw HostError();
            };
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", options, t),
                             "host language error");
        }
        {
            sup::CallbackSet t = cbs;
            t.vector_vector = [](const std::vector<double>&) -> std::vector<double> {
                throw HostError();
            };
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", options, t),
                             "host language error");
        }
        {
            // The jackknife is only entered on the BCa path, so this is the one arm that needs
            // its own CI method to be reached at all.
            sup::CallbackSet t = cbs;
            t.data_index = [](const std::vector<double>&, int) -> std::vector<double> {
                throw HostError();
            };
            CHECK_THROWS_MSG(
                sup::run_callback("bootstrap", "run",
                                  R"({"data": [4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7],
                                      "replicates": 50, "seed": 12345, "ci_method": "BCa"})",
                                  t),
                "host language error");
        }

        // --- one abort state across all four delegates -------------------------------------
        //
        // THE property the shared state buys, and nothing else in this file proves it for this
        // group: once ANY delegate throws, none of the other three may be entered again, because
        // the ported bootstrap does not know it is unwinding and would otherwise re-enter the host
        // with an unwind already pending. Each block below makes ONE delegate throw and gives
        // ANOTHER a body that reports being entered afterwards -- so a guard given a private abort
        // state fails here loudly, with the re-entry message rather than the host one.
        {
            // resample throws; fit reports re-entry.
            bool aborted = false;
            sup::CallbackSet t = cbs;
            t.data_rng = [&aborted](const std::vector<double>&, const std::vector<double>&,
                                    corehydro::numerics::sampling::MersenneTwister&)
                -> std::vector<double> {
                aborted = true;
                throw HostError();
            };
            t.data_vector = [&aborted, &cbs](const std::vector<double>& d) {
                if (aborted) throw std::runtime_error("the fit function was re-entered after the abort");
                return cbs.data_vector(d);
            };
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", options, t),
                             "host language error");
        }
        {
            // fit throws; the statistic reports re-entry.
            bool aborted = false;
            sup::CallbackSet t = cbs;
            t.data_vector = [&aborted](const std::vector<double>&) -> std::vector<double> {
                aborted = true;
                throw HostError();
            };
            t.vector_vector = [&aborted](const std::vector<double>& p) {
                if (aborted)
                    throw std::runtime_error("the statistic function was re-entered after the abort");
                return p;
            };
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", options, t),
                             "host language error");
        }
        {
            // The statistic throws on its SECOND call (the first is the ported class's own
            // original-parameter evaluation, before any replicate); resample reports re-entry.
            int statistic_calls = 0;
            bool aborted = false;
            sup::CallbackSet t = cbs;
            t.vector_vector = [&statistic_calls, &aborted](const std::vector<double>& p)
                -> std::vector<double> {
                if (++statistic_calls > 1) {
                    aborted = true;
                    throw HostError();
                }
                return p;
            };
            t.data_rng = [&aborted, &cbs](const std::vector<double>& d, const std::vector<double>& p,
                                          corehydro::numerics::sampling::MersenneTwister& prng) {
                if (aborted)
                    throw std::runtime_error("the resample function was re-entered after the abort");
                return cbs.data_rng(d, p, prng);
            };
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", options, t),
                             "host language error");
        }
        {
            // jackknife throws; fit reports re-entry (the BCa acceleration loop calls it next).
            bool aborted = false;
            sup::CallbackSet t = cbs;
            t.data_index = [&aborted](const std::vector<double>&, int) -> std::vector<double> {
                aborted = true;
                throw HostError();
            };
            t.data_vector = [&aborted, &cbs](const std::vector<double>& d) {
                if (aborted) throw std::runtime_error("the fit function was re-entered after the abort");
                return cbs.data_vector(d);
            };
            CHECK_THROWS_MSG(
                sup::run_callback("bootstrap", "run",
                                  R"({"data": [4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7],
                                      "replicates": 50, "seed": 12345, "ci_method": "BCa"})",
                                  t),
                "host language error");
        }

        // Wrong-shaped returns are refused by name rather than left to corrupt a result two calls
        // later. Each check lives inside the guarded function, so it aborts the run exactly as a
        // host-language error does.
        {
            // A fit wider than the parameter set it is fitted against. The parameter count comes
            // from `parameters` when the caller supplies one, so this is the realistic spelling of
            // the mismatch: theta-hat of one width, a fit of another.
            sup::CallbackSet t = cbs;
            t.data_vector = [](const std::vector<double>&) {
                return std::vector<double>{1.0, 2.0};
            };
            CHECK_THROWS_MSG(
                sup::run_callback("bootstrap", "run",
                                  R"({"data": [4.1, 5.2, 4.8, 5.5], "replicates": 20,
                                      "seed": 12345, "parameters": [5.0]})",
                                  t),
                "must return one value per parameter");
        }
        {
            // And a fit whose width CHANGES between calls, which is the only way the derived
            // theta-hat path (no `parameters` key) can disagree with itself.
            int fit_calls = 0;
            sup::CallbackSet t = cbs;
            t.data_vector = [&fit_calls, &cbs](const std::vector<double>& d) {
                if (++fit_calls > 1) return std::vector<double>{1.0, 2.0};
                return cbs.data_vector(d);
            };
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", options, t),
                             "must return one value per parameter");
        }
        {
            sup::CallbackSet t = cbs;
            t.data_rng = [](const std::vector<double>&, const std::vector<double>&,
                            corehydro::numerics::sampling::MersenneTwister&) {
                return std::vector<double>{};
            };
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", options, t),
                             "resample function must return at least one value");
        }
        {
            sup::CallbackSet t = cbs;
            t.vector_vector = [](const std::vector<double>&) { return std::vector<double>{}; };
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", options, t),
                             "statistic function must return at least one value");
        }
        {
            // A jackknife that drops nothing. The ported BCa would divide 0 by 0 and report a
            // NaN interval; this names it instead.
            sup::CallbackSet t = cbs;
            t.data_index = [](const std::vector<double>& d, int) { return d; };
            CHECK_THROWS_MSG(
                sup::run_callback("bootstrap", "run",
                                  R"({"data": [4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7],
                                      "replicates": 50, "seed": 12345, "ci_method": "BCa"})",
                                  t),
                "must return fewer values than it was given");
        }

        // Dispatch and validation errors.
        {
            sup::CallbackSet empty;
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "nope", "{}", cbs),
                             "unknown bootstrap method");
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", "{}", empty),
                             "requires a resample function");
            sup::CallbackSet resample_only;
            resample_only.data_rng = cbs.data_rng;
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", "{}", resample_only),
                             "requires a fit function");
            sup::CallbackSet no_statistic;
            no_statistic.data_rng = cbs.data_rng;
            no_statistic.data_vector = cbs.data_vector;
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", "{}", no_statistic),
                             "requires a statistic function");
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", "{}", cbs),
                             "requires the option 'data'");
            CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", R"({"data": []})", cbs),
                             "at least one value");
            CHECK_THROWS_MSG(
                sup::run_callback("bootstrap", "run",
                                  R"({"data": [1.0, 2.0], "ci_method": "Nope"})", cbs),
                "unknown bootstrap ci_method");
        }
    }

    // --- gmm group ---------------------------------------------------------------------------
    //
    // Analytic/structural properties only (the repo convention for a ctest); the C#-pinned oracle
    // values for this group live in fixtures/callback/gmm.json. The model throughout is the
    // just-identified two-parameter method-of-moments fit of a Normal: theta = (mu, sigma2) and
    //
    //   g(theta) = [ mean(x - mu), mean((x - mu)^2 - sigma2) ]
    //
    // whose unique root -- and therefore the GMM optimum, since q = p makes g(theta-hat) = 0
    // attainable -- is the sample mean and the population variance. That closed form is what makes
    // this a real test rather than a regression against whatever the optimizer happened to return.
    // Arithmetic and an explicit loop only, so all four runners agree bit for bit.
    {
        const std::vector<double> data = {4.1, 5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7};
        const double n = static_cast<double>(data.size());
        double mean_hat = 0.0;
        for (double x : data) mean_hat += x;
        mean_hat /= n;
        double variance_hat = 0.0;
        for (double x : data) variance_hat += (x - mean_hat) * (x - mean_hat);
        variance_hat /= n;

        auto moments = [data, n](const std::vector<double>& p) {
            sup::MomentConditionReturn out;
            double g0 = 0.0, g1 = 0.0, s00 = 0.0, s01 = 0.0, s11 = 0.0;
            for (double x : data) {
                double a = x - p[0];
                double b = a * a - p[1];
                g0 += a;
                g1 += b;
                s00 += a * a;
                s01 += a * b;
                s11 += b * b;
            }
            out.g = {g0 / n, g1 / n};
            out.s = {s00 / n, s01 / n, s01 / n, s11 / n};
            out.s_rows = 2;
            out.s_cols = 2;
            return out;
        };
        // The analytic Jacobian D = dg/dtheta of the same moment conditions, row-major 2 x 2.
        auto jacobian = [data, n](const std::vector<double>& p) {
            double acc = 0.0;
            for (double x : data) acc += x - p[0];
            return std::make_pair(std::vector<double>{-1.0, 0.0, -2.0 * acc / n, -1.0},
                                  std::vector<int>{2, 2});
        };

        sup::CallbackSet cbs;
        cbs.moment_conditions = moments;

        const std::string options =
            R"({"initial": [5.0, 0.5], "lower": [0.0, 0.001], "upper": [10.0, 10.0],
                "sample_size": 8})";

        sup::CallbackResult r1 = sup::run_callback("gmm", "fit", options, cbs);
        sup::CallbackResult r2 = sup::run_callback("gmm", "fit", options, cbs);
        // Bit-for-bit reproducible: there is no RNG anywhere in this fit. Compared value by value
        // rather than with ==, because `j_stat_pval` is a structural NaN and NaN != NaN.
        CHECK_EQ(r1.values.size(), r2.values.size());
        bool identical = r1.names == r2.names;
        for (std::size_t i = 0; i < r1.values.size() && identical; ++i)
            identical = std::isnan(r1.values[i]) ? std::isnan(r2.values[i])
                                                 : r1.values[i] == r2.values[i];
        CHECK_TRUE(identical);
        CHECK_EQ(r1.status, std::string("Success"));

        auto named = [](const sup::CallbackResult& r, const std::string& want) {
            for (std::size_t i = 0; i < r.names.size(); ++i)
                if (r.names[i] == want) return r.values.at(i);
            throw std::runtime_error("no result named " + want);
        };

        // The closed form, which is the whole point of choosing this problem.
        CHECK_NEAR(named(r1, "parameter[0]"), mean_hat, 1e-6);
        CHECK_NEAR(named(r1, "parameter[1]"), variance_hat, 1e-6);
        // A just-identified fit (q = p) has zero degrees of freedom, so Hansen's J cannot test the
        // specification: the p-value is structurally NaN. That is CORRECT, not a defect -- see
        // docs/upstream-csharp-issues.md and fixtures/estimation/gmm_bulletin17c_smoke.json, which
        // pins the same NaN for the B17C fit.
        //
        // J ITSELF IS NOT ASSERTED, and deliberately not: on a just-identified fit the moment
        // residual covariance V = S - D(D'S^-1 D)^-1 D' is THEORETICALLY ZERO, so g' V^-1 g is
        // whatever inverting a numerically singular matrix happens to give. Measured on this one
        // problem, over the four optimizers, all converging to the same parameters to 1e-9:
        // -1.3e-09, 8.6e+19, -7.7e-15, 0.126 -- and it THROWS outright from R and Python under
        // NelderMead, where run_gmm reports NaN instead of failing the fit. Any bound on it would
        // be a fiction. What is asserted is what is meaningful: the fit succeeds, the p-value is
        // NaN, and the degrees of freedom are zero.
        CHECK_EQ(named(r1, "degree_of_freedom"), 0.0);
        CHECK_TRUE(std::isnan(named(r1, "j_stat_pval")));
        // Standard errors and a covariance matrix come back, shaped {p, p}.
        CHECK_EQ(r1.dims.size(), std::size_t{2});
        CHECK_EQ(r1.dims.at(0), 2);
        CHECK_EQ(r1.dims.at(1), 2);
        CHECK_TRUE(named(r1, "standard_error[0]") > 0.0);
        CHECK_TRUE(named(r1, "standard_error[1]") > 0.0);
        CHECK_NEAR(named(r1, "covariance[0,0]"),
                   named(r1, "standard_error[0]") * named(r1, "standard_error[0]"), 1e-12);
        CHECK_EQ(named(r1, "correlation[0,0]"), 1.0);
        CHECK_EQ(named(r1, "sample_size"), 8.0);
        CHECK_EQ(named(r1, "number_of_moment_conditions"), 2.0);

        // The analytic Jacobian replaces the ported numerical one, and lands on the same optimum
        // -- the closed form is the same whichever way the gradient was computed.
        {
            sup::CallbackSet t = cbs;
            t.vector_matrix = jacobian;
            sup::CallbackResult r = sup::run_callback("gmm", "fit", options, t);
            CHECK_NEAR(named(r, "parameter[0]"), mean_hat, 1e-6);
            CHECK_NEAR(named(r, "parameter[1]"), variance_hat, 1e-6);
        }

        // A penalty pulling sigma2 towards 1 moves the estimate away from the closed form and
        // towards the target -- which is what proves the delegate is reached at all.
        {
            sup::CallbackSet t = cbs;
            t.vector_scalar = [](const std::vector<double>& p) {
                double d = p[1] - 1.0;
                return 0.5 * d * d;
            };
            sup::CallbackResult r = sup::run_callback("gmm", "fit", options, t);
            CHECK_TRUE(named(r, "parameter[1]") > variance_hat);
            CHECK_TRUE(named(r, "parameter[1]") < 1.0);
        }

        // Every optimizer and every strategy the verb accepts runs and finds the same optimum.
        for (const char* optimizer : {"BFGS", "NelderMead", "Powell", "MultilevelSingleLinkage"}) {
            std::string opts =
                R"({"initial": [5.0, 0.5], "lower": [0.0, 0.001], "upper": [10.0, 10.0],
                    "sample_size": 8, "optimizer": ")" +
                std::string(optimizer) + R"("})";
            sup::CallbackResult r = sup::run_callback("gmm", "fit", opts, cbs);
            CHECK_NEAR(named(r, "parameter[0]"), mean_hat, 1e-4);
            // Whatever the optimizer, and whatever inverting a singular V gives (see above), an
            // uncomputable J-statistic must not fail the fit.
            CHECK_TRUE(std::isnan(named(r, "j_stat_pval")));
        }
        for (const char* strategy : {"OneStep", "TwoStep", "Iterative"}) {
            std::string opts =
                R"({"initial": [5.0, 0.5], "lower": [0.0, 0.001], "upper": [10.0, 10.0],
                    "sample_size": 8, "strategy": ")" +
                std::string(strategy) + R"("})";
            sup::CallbackResult r = sup::run_callback("gmm", "fit", opts, cbs);
            CHECK_NEAR(named(r, "parameter[0]"), mean_hat, 1e-6);
        }

        // The OVER-IDENTIFIED companion, q = 3 > p = 2: the same Normal model and the same eight
        // observations with mean((x - mu)^3) added as a third moment condition, which is zero for
        // a Normal and so leaves the model itself unchanged. Everything above is just-identified,
        // and three things are reachable ONLY from here -- a non-zero degrees of freedom, the
        // chi-squared p-value branch of post_process(), and the estimator's refusal of OneStep.
        // The C#-pinned oracle for this same fit is fixtures/callback/gmm.json's
        // over_identified_three_moments.
        {
            auto moments3 = [data, n](const std::vector<double>& p) {
                sup::MomentConditionReturn out;
                double g0 = 0.0, g1 = 0.0, g2 = 0.0;
                double s00 = 0.0, s01 = 0.0, s02 = 0.0, s11 = 0.0, s12 = 0.0, s22 = 0.0;
                for (double x : data) {
                    double a = x - p[0];
                    double b = a * a - p[1];
                    double c = a * a * a;
                    g0 += a;
                    g1 += b;
                    g2 += c;
                    s00 += a * a;
                    s01 += a * b;
                    s02 += a * c;
                    s11 += b * b;
                    s12 += b * c;
                    s22 += c * c;
                }
                out.g = {g0 / n, g1 / n, g2 / n};
                out.s = {s00 / n, s01 / n, s02 / n, s01 / n, s11 / n,
                         s12 / n, s02 / n, s12 / n, s22 / n};
                out.s_rows = 3;
                out.s_cols = 3;
                return out;
            };
            sup::CallbackSet t;
            t.moment_conditions = moments3;
            sup::CallbackResult r = sup::run_callback("gmm", "fit", options, t);

            CHECK_EQ(r.status, std::string("Success"));
            CHECK_EQ(named(r, "number_of_moment_conditions"), 3.0);
            CHECK_EQ(named(r, "number_of_parameters"), 2.0);
            CHECK_EQ(named(r, "degree_of_freedom"), 1.0);
            // The covariance stays p x p: it does not grow with q.
            CHECK_EQ(r.dims.at(0), 2);
            CHECK_EQ(r.dims.at(1), 2);
            // With three conditions and two parameters, g cannot be driven to zero, so the
            // weighting matrix decides the trade-off and the estimate leaves the closed form.
            CHECK_TRUE(std::abs(named(r, "parameter[0]") - mean_hat) > 1e-3);
            CHECK_TRUE(named(r, "standard_error[0]") > 0.0);
            CHECK_TRUE(named(r, "standard_error[1]") > 0.0);
            // THE property this case exists for: a non-zero degrees of freedom takes
            // post_process() down its chi-squared branch instead of writing the structural NaN, so
            // the p-value is a real probability. J ITSELF IS STILL NOT ASSERTED, and this is worth
            // being explicit about, because over-identifying is often assumed to fix it and does
            // not: V = S - D(D'S^-1 D)^-1 D' has rank exactly q - p for ANY q and p, so it is
            // singular here too -- the rank only moves from 0 to 1 -- and inverting it amplifies
            // the optimizer's convergence tolerance rather than anything about the data. Measured
            // on THIS fit, the real C# library returns J = 214.59 with a p-value of 0 while this
            // core returns J = -129.46 with a p-value of 1, from parameters that agree to 2e-11.
            // So what is pinned is that the branch was TAKEN, not what it produced.
            CHECK_TRUE(!std::isnan(named(r, "j_stat_pval")));
            CHECK_TRUE(named(r, "j_stat_pval") >= 0.0 && named(r, "j_stat_pval") <= 1.0);

            // And OneStep is refused for an over-identified problem, by the ported estimator's own
            // validation, with its own message -- the one strategy/identification combination the
            // just-identified cases above can never reach.
            CHECK_THROWS_MSG(
                sup::run_callback("gmm", "fit",
                                  R"({"initial": [5.0, 0.5], "lower": [0.0, 0.001],
                                      "upper": [10.0, 10.0], "sample_size": 8,
                                      "strategy": "OneStep"})",
                                  t),
                "over-identified, so you cannot use the one-step estimation method");
        }

        // A host exception inside EACH of the three delegates reaches the caller. One test per
        // delegate: a guard wired for one and missing on another is exactly the shape of bug this
        // invites, and a test that only makes the moment conditions throw cannot catch it.
        {
            // Throwing on the FIRST call is caught by the up-front probe, before the estimator
            // exists, so it reaches the caller without the guard being involved at all. Both are
            // worth pinning, and they are different paths.
            sup::CallbackSet t = cbs;
            t.moment_conditions =
                [](const std::vector<double>&) -> sup::MomentConditionReturn { throw HostError(); };
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", options, t), "host language error");

            // And throwing on a LATER call, which is the one the guard has to carry: by then the
            // estimator is running, its optimizer swallows everything, and only the guard can get
            // the message back out.
            int calls = 0;
            t.moment_conditions = [&calls, moments](const std::vector<double>& p)
                -> sup::MomentConditionReturn {
                if (++calls > 1) throw HostError();
                return moments(p);
            };
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", options, t), "host language error");
        }
        {
            sup::CallbackSet t = cbs;
            t.vector_matrix = [](const std::vector<double>&)
                -> std::pair<std::vector<double>, std::vector<int>> { throw HostError(); };
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", options, t), "host language error");
        }
        {
            sup::CallbackSet t = cbs;
            t.vector_scalar = [](const std::vector<double>&) -> double { throw HostError(); };
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", options, t), "host language error");
        }

        // --- one abort state across all three delegates -------------------------------------
        //
        // THE property the shared state buys, and nothing else in this file proves it for this
        // group: once ANY delegate throws, none of the others may be entered again, because the
        // ported estimator does not know it is unwinding -- its optimizer catches everything and
        // walks on -- and would otherwise re-enter the host with an unwind already pending. Each
        // block makes ONE delegate throw and gives ANOTHER a body that reports being entered
        // afterwards, so a guard given a private abort state fails here loudly, with the re-entry
        // message rather than the host one.
        {
            // The moment conditions throw; the Jacobian reports re-entry (get_gradient calls
            // get_g and then get_jacobian, in that order).
            bool aborted = false;
            sup::CallbackSet t = cbs;
            t.moment_conditions = [&aborted](const std::vector<double>&)
                -> sup::MomentConditionReturn {
                aborted = true;
                throw HostError();
            };
            t.vector_matrix = [&aborted, jacobian](const std::vector<double>& p) {
                if (aborted)
                    throw std::runtime_error("the jacobian function was re-entered after the abort");
                return jacobian(p);
            };
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", options, t), "host language error");
        }
        {
            // The Jacobian throws; the moment conditions report re-entry.
            bool aborted = false;
            sup::CallbackSet t = cbs;
            t.vector_matrix = [&aborted](const std::vector<double>&)
                -> std::pair<std::vector<double>, std::vector<int>> {
                aborted = true;
                throw HostError();
            };
            t.moment_conditions = [&aborted, moments](const std::vector<double>& p) {
                if (aborted)
                    throw std::runtime_error(
                        "the moment condition function was re-entered after the abort");
                return moments(p);
            };
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", options, t), "host language error");
        }
        {
            // The penalty throws on its SECOND call (the first happens inside the same Q() the
            // moment conditions were just evaluated for); the moment conditions report re-entry.
            int penalty_calls = 0;
            bool aborted = false;
            sup::CallbackSet t = cbs;
            t.vector_scalar = [&penalty_calls, &aborted](const std::vector<double>&) -> double {
                if (++penalty_calls > 1) {
                    aborted = true;
                    throw HostError();
                }
                return 0.0;
            };
            t.moment_conditions = [&aborted, moments](const std::vector<double>& p) {
                if (aborted)
                    throw std::runtime_error(
                        "the moment condition function was re-entered after the abort");
                return moments(p);
            };
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", options, t), "host language error");
        }

        // Wrong-shaped returns are refused by name rather than left to corrupt a fit. Each check
        // lives inside the guarded function, so it aborts the run exactly as a host-language
        // error does, and each names the element it is talking about.
        {
            // The moment vector changes length between calls -- the only way it can disagree with
            // the q the up-front probe measured.
            int calls = 0;
            sup::CallbackSet t = cbs;
            t.moment_conditions = [&calls, moments](const std::vector<double>& p) {
                sup::MomentConditionReturn out = moments(p);
                if (++calls > 1) out.g.push_back(0.0);
                return out;
            };
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", options, t),
                             "'g' (the moment vector) must hold one value per moment condition");
        }
        {
            sup::CallbackSet t = cbs;
            t.moment_conditions = [moments](const std::vector<double>& p) {
                sup::MomentConditionReturn out = moments(p);
                out.s_rows = 1;
                out.s_cols = 4;
                return out;
            };
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", options, t),
                             "'s' (the weighting matrix) must be a square matrix");
        }
        {
            // An empty moment vector: caught by the up-front probe, before the estimator exists.
            sup::CallbackSet t;
            t.moment_conditions = [](const std::vector<double>&) {
                return sup::MomentConditionReturn{};
            };
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", options, t),
                             "must return at least one moment condition");
        }
        {
            // A Jacobian of the wrong shape. q x p, not p x q -- and the message says which.
            sup::CallbackSet t = cbs;
            t.vector_matrix = [](const std::vector<double>&) {
                return std::make_pair(std::vector<double>{1.0, 2.0, 3.0}, std::vector<int>{1, 3});
            };
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", options, t),
                             "jacobian function must return a 2 x 2 matrix");
        }
        {
            // A declared moment-condition count that disagrees with what the callback returns.
            CHECK_THROWS_MSG(
                sup::run_callback("gmm", "fit",
                                  R"({"initial": [5.0, 0.5], "lower": [0.0, 0.001],
                                      "upper": [10.0, 10.0], "sample_size": 8,
                                      "number_of_moment_conditions": 3})",
                                  cbs),
                "returned 2");
        }

        // Dispatch and validation errors.
        {
            sup::CallbackSet empty;
            CHECK_THROWS_MSG(sup::run_callback("gmm", "nope", "{}", cbs), "unknown gmm method");
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", "{}", empty),
                             "requires a moment condition function");
            CHECK_THROWS_MSG(sup::run_callback("gmm", "fit", "{}", cbs),
                             "requires the option 'initial'");
            CHECK_THROWS_MSG(
                sup::run_callback("gmm", "fit",
                                  R"({"initial": [5.0, 0.5], "lower": [0.0], "upper": [10.0, 10.0],
                                      "sample_size": 8})",
                                  cbs),
                "same length");
            CHECK_THROWS_MSG(
                sup::run_callback("gmm", "fit",
                                  R"({"initial": [5.0, 0.5], "lower": [0.0, 0.001],
                                      "upper": [10.0, 10.0]})",
                                  cbs),
                "requires the option 'sample_size'");
            CHECK_THROWS_MSG(
                sup::run_callback("gmm", "fit",
                                  R"({"initial": [5.0, 0.5], "lower": [0.0, 0.001],
                                      "upper": [10.0, 10.0], "sample_size": 0})",
                                  cbs),
                "sample_size");
            CHECK_THROWS_MSG(
                sup::run_callback("gmm", "fit",
                                  R"({"initial": [5.0, 0.5], "lower": [0.0, 0.001],
                                      "upper": [10.0, 10.0], "sample_size": 8,
                                      "optimizer": "Nope"})",
                                  cbs),
                "unknown optimizer");
            CHECK_THROWS_MSG(
                sup::run_callback("gmm", "fit",
                                  R"({"initial": [5.0, 0.5], "lower": [0.0, 0.001],
                                      "upper": [10.0, 10.0], "sample_size": 8,
                                      "strategy": "Nope"})",
                                  cbs),
                "unknown GMM estimation strategy");
        }
    }

    return chtest::summary("callback_runner");
}
