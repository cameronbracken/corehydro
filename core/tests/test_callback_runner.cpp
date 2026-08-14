#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"
#include "corehydro/numerics/support/callback_runner.hpp"

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

    // The three groups Tasks 4, 6 and 7 fill in are reachable but explicitly unimplemented.
    {
        sup::CallbackSet cbs;
        CHECK_THROWS_MSG(sup::run_callback("mcmc", "sample", "{}", cbs), "Task 4");
        CHECK_THROWS_MSG(sup::run_callback("bootstrap", "run", "{}", cbs), "Task 6");
        CHECK_THROWS_MSG(sup::run_callback("gmm", "estimate", "{}", cbs), "Task 7");
    }

    return chtest::summary("callback_runner");
}
