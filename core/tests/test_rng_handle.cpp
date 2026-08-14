// The lifetime contract of the RNG handle (numerics/support/rng_handle.hpp) and the rng group of
// the callback runner that hands one out.
//
// Analytic identities only (the repo convention for a ctest): "the handle's draws ARE the core
// generator's draws" is checked by running a bare MersenneTwister beside it, never by pinning a
// literal. The C#-pinned oracle values for the same runs live in fixtures/callback/rng_handle.json.
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/support/callback_runner.hpp"
#include "corehydro/numerics/support/rng_handle.hpp"

namespace sup = corehydro::numerics::support;
namespace samp = corehydro::numerics::sampling;

struct HostError : std::runtime_error {
    HostError() : std::runtime_error("host language error") {}
};

int main() {
    // --- the borrow itself -------------------------------------------------------------------
    //
    // Inside the scope the handle draws the SAME stream a bare generator draws, in the same order.
    {
        samp::MersenneTwister prng(12345U);
        samp::MersenneTwister reference(12345U);
        sup::RngBorrowPtr handle;
        {
            sup::RngScope scope(prng);
            handle = scope.handle();
            std::vector<double> u = handle->uniform(3);
            CHECK_EQ(u.size(), std::size_t{3});
            for (double v : u) CHECK_EQ(v, reference.next_double());
            std::vector<int> k = handle->integers(4, 0, 10);
            CHECK_EQ(k.size(), std::size_t{4});
            for (int v : k) CHECK_EQ(v, reference.next(0, 10));
            // Interleaving matters: the two verbs share one state, so a uniform taken after an
            // integer must continue where the integer left off.
            CHECK_EQ(handle->uniform(1).at(0), reference.next_double());
        }
        // ...and the moment the scope is gone, every draw through the SAME handle object raises
        // rather than reading the generator (which, in the real bindings, is destroyed by now).
        CHECK_THROWS_MSG(handle->uniform(1), "no longer valid");
        CHECK_THROWS_MSG(handle->integers(1, 0, 2), "no longer valid");
        CHECK_TRUE(!handle->valid);
        CHECK_TRUE(handle->prng == nullptr);
    }

    // The scope invalidates on an EXCEPTIONAL unwind too, not only on a normal return. This is the
    // path that matters most: a host callback that throws is exactly when a half-built handle is
    // most likely to be left lying around.
    {
        samp::MersenneTwister prng(7U);
        sup::RngBorrowPtr handle;
        try {
            sup::RngScope scope(prng);
            handle = scope.handle();
            CHECK_EQ(handle->uniform(1).size(), std::size_t{1});
            throw HostError();
        } catch (const HostError&) {
        }
        CHECK_THROWS_MSG(handle->uniform(1), "no longer valid");
    }

    // Two scopes in sequence over the same generator: the second handle is live and CONTINUES the
    // stream, while the first stays dead. A user who stored the handle from call one and used it in
    // call two gets the error, not call two's generator.
    {
        samp::MersenneTwister prng(99U);
        samp::MersenneTwister reference(99U);
        sup::RngBorrowPtr first;
        {
            sup::RngScope scope(prng);
            first = scope.handle();
            CHECK_EQ(first->uniform(1).at(0), reference.next_double());
        }
        {
            sup::RngScope scope(prng);
            sup::RngBorrowPtr second = scope.handle();
            CHECK_EQ(second->uniform(1).at(0), reference.next_double());
            CHECK_THROWS_MSG(first->uniform(1), "no longer valid");
        }
    }

    // Argument validation, before any state is consumed.
    {
        samp::MersenneTwister prng(1U);
        sup::RngScope scope(prng);
        sup::RngBorrowPtr h = scope.handle();
        CHECK_THROWS_MSG(h->uniform(0), "positive whole number");
        CHECK_THROWS_MSG(h->uniform(-1), "positive whole number");
        CHECK_THROWS_MSG(h->integers(0, 0, 5), "positive whole number");
        CHECK_THROWS_MSG(h->integers(2, 5, 5), "must be below");
        CHECK_THROWS_MSG(h->integers(2, 6, 5), "must be below");
        // A rejected call consumed nothing: the next draw is still the seed's first value.
        samp::MersenneTwister reference(1U);
        CHECK_EQ(h->uniform(1).at(0), reference.next_double());
    }

    // --- the rng group ------------------------------------------------------------------------
    //
    // The probe seeds a generator, hands the callback a handle, and returns what it drew. Same
    // seed, same draws -- and they are the bare generator's draws.
    {
        samp::MersenneTwister reference(12345U);
        std::vector<double> expected;
        for (int i = 0; i < 3; ++i) expected.push_back(reference.next_double());

        sup::CallbackSet cbs;
        cbs.vector_rng = [](const std::vector<double>& parameters, samp::MersenneTwister& prng) {
            sup::RngScope scope(prng);
            return scope.handle()->uniform(static_cast<int>(parameters.at(0)));
        };
        sup::CallbackResult r =
            sup::run_callback("rng", "probe", R"({"seed": 12345, "parameters": [3.0]})", cbs);
        CHECK_EQ(r.values.size(), std::size_t{3});
        CHECK_EQ(r.dims.at(0), 3);
        CHECK_EQ(r.status, std::string("Success"));
        for (std::size_t i = 0; i < expected.size(); ++i) CHECK_EQ(r.values.at(i), expected.at(i));

        // A different seed is a different stream; the same seed run twice is the same stream.
        sup::CallbackResult again =
            sup::run_callback("rng", "probe", R"({"seed": 12345, "parameters": [3.0]})", cbs);
        CHECK_EQ(again.values.at(0), r.values.at(0));
        sup::CallbackResult other =
            sup::run_callback("rng", "probe", R"({"seed": 54321, "parameters": [3.0]})", cbs);
        CHECK_TRUE(other.values.at(0) != r.values.at(0));
    }

    // The array-seed form reaches the ported array constructor, C# `new MersenneTwister(int[])` --
    // the seeding upstream's own Test_MersenneTwister uses, and a different stream from any single
    // seed. (fixtures/callback/rng_handle.json pins its values against that upstream test.)
    {
        samp::MersenneTwister reference(std::vector<std::uint32_t>{291U, 564U, 837U, 1110U});
        sup::CallbackSet cbs;
        cbs.vector_rng = [](const std::vector<double>&, samp::MersenneTwister& prng) {
            sup::RngScope scope(prng);
            return scope.handle()->uniform(2);
        };
        sup::CallbackResult r =
            sup::run_callback("rng", "probe", R"({"seeds": [291, 564, 837, 1110]})", cbs);
        CHECK_EQ(r.values.at(0), reference.next_double());
        CHECK_EQ(r.values.at(1), reference.next_double());
        // `seeds` wins over `seed` when both are written, as the group header says.
        sup::CallbackResult both = sup::run_callback(
            "rng", "probe", R"({"seed": 12345, "seeds": [291, 564, 837, 1110]})", cbs);
        CHECK_EQ(both.values.at(0), r.values.at(0));
    }

    // `parameters` is optional and the callback sees it verbatim -- the Gibbs proposal shape.
    {
        sup::CallbackSet cbs;
        cbs.vector_rng = [](const std::vector<double>& parameters, samp::MersenneTwister&) {
            return std::vector<double>{static_cast<double>(parameters.size())};
        };
        sup::CallbackResult r = sup::run_callback("rng", "probe", R"({"seed": 1})", cbs);
        CHECK_EQ(r.values.at(0), 0.0);
        sup::CallbackResult two =
            sup::run_callback("rng", "probe", R"({"seed": 1, "parameters": [4.0, 5.0]})", cbs);
        CHECK_EQ(two.values.at(0), 2.0);
    }

    // A host exception inside the rng callback reaches the caller unchanged, and the sentinel
    // (an empty vector) never reaches it as if it were an answer.
    {
        sup::CallbackSet cbs;
        cbs.vector_rng = [](const std::vector<double>&,
                            samp::MersenneTwister&) -> std::vector<double> { throw HostError(); };
        CHECK_THROWS_MSG(sup::run_callback("rng", "probe", R"({"seed": 1})", cbs),
                         "host language error");
    }

    // A handle a callback leaks OUT of itself is dead by the time run_callback returns, even
    // though the generator it borrowed is still alive at that moment -- the scope, not the
    // generator, is what the flag tracks. This is the C++ statement of the R/Python misuse the
    // bindings' tests exercise.
    {
        sup::RngBorrowPtr leaked;
        sup::CallbackSet cbs;
        cbs.vector_rng = [&leaked](const std::vector<double>&, samp::MersenneTwister& prng) {
            sup::RngScope scope(prng);
            leaked = scope.handle();
            return leaked->uniform(2);
        };
        sup::CallbackResult r = sup::run_callback("rng", "probe", R"({"seed": 3})", cbs);
        CHECK_EQ(r.values.size(), std::size_t{2});
        CHECK_THROWS_MSG(leaked->uniform(1), "no longer valid");
    }

    // Dispatch errors.
    {
        sup::CallbackSet cbs;
        CHECK_THROWS_MSG(sup::run_callback("rng", "probe", R"({"seed": 1})", cbs),
                         "requires a function");
        cbs.vector_rng = [](const std::vector<double>&, samp::MersenneTwister&) {
            return std::vector<double>{1.0};
        };
        CHECK_THROWS_MSG(sup::run_callback("rng", "nope", "{}", cbs), "unknown rng method");
        CHECK_THROWS_MSG(sup::run_callback("rng", "probe", "{}", cbs), "requires the option 'seed'");
    }

    return chtest::summary("rng_handle");
}
