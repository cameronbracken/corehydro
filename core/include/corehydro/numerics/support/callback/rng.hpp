// corehydro ADDITION -- no upstream C# counterpart. The rng group of callback_runner.hpp: the one
// verb whose only job is to hand a callback the core's seeded generator and give back what it drew.
//
// It exists because the two delegates that receive the generator upstream -- the Gibbs proposal
// (`Proposal(double[] parameters, Random prng)`) and the bootstrap resample
// (`Func<TData, ParameterSet, Random, TData>`) -- are expensive to run and drown the property that
// actually needs pinning: that a draw taken inside a host-language callback comes off the core's
// seeded stream and NOT off R's or Python's own generator. A sampler's fixture cannot show that,
// because a divergence there is indistinguishable from a divergence in the sampler. `probe` is the
// same crossing with nothing else in it: seed a MersenneTwister, call the callback once with
// (parameters, generator), return the vector. If R and Python disagree here, the handle is broken;
// if they agree here and a sampler disagrees, the sampler is.
//
// It is deliberately shaped as the Gibbs proposal (`vector_rng`: parameters in, vector out) rather
// than as something simpler, so the fixture exercises the signature the samplers will use.
//
// Options grammar (JSON object):
//
//   probe: {"seed": 12345, "parameters": [3.0]}
//   probe: {"seeds": [291, 564, 837, 1110], "parameters": []}
//
// Exactly one of `seed` and `seeds` is required. `seed` reaches the ported MersenneTwister's
// single-seed constructor, the one C# `new MersenneTwister(int seed)` calls; `seeds` reaches the
// array constructor, C# `new MersenneTwister(int[] seeds)`. Both forms are here because they are
// the two seedings with oracle value: `seeds` is what upstream's own Test_MersenneTwister uses, so
// a fixture on it carries a real upstream test literal, while `seed` is the form every sampler
// and bootstrap actually seeds with. `seeds` wins if both are given.
//
// `parameters` is optional and defaults to empty; it is the `double[] parameters` half of the
// proposal signature, and the fixture catalog uses it to tell its callbacks how many values to
// draw, so one catalog entry serves any count.
//
// The drive site follows callback_guard.hpp's "USE IT IN PAIRS" rule like every other group, even
// though the ported code between the guard and the caller is only this file: the sentinel for a
// vector callback is an EMPTY vector, and a caller reading `values` would otherwise see an empty
// successful result where the user's own error belongs.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/support/callback/common.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"
#include "corehydro/numerics/support/rng_handle.hpp"

namespace corehydro::numerics::support::detail {

inline CallbackResult run_rng(const std::string& method, const JsonValue& o,
                              const CallbackSet& cbs) {
    CallbackResult r;
    r.status = "Success";

    if (method == "probe") {
        if (!cbs.vector_rng)
            throw std::invalid_argument("rng/probe requires a function of (parameters, rng)");
        std::vector<double> parameters =
            o.contains("parameters") ? o.at("parameters").as_double_vector()
                                     : std::vector<double>{};

        // The seed reaches the generator through the C# cast chain: the fixture writes it as a
        // JSON number, C# takes it as `int`, and the ported constructor takes the `(uint)` of that
        // -- so a negative seed is the same 32-bit pattern in both, which int64 -> uint32 here
        // reproduces and a direct double -> uint32 conversion would not.
        auto to_word = [](double v) {
            return static_cast<std::uint32_t>(static_cast<std::int64_t>(v));
        };
        std::vector<std::uint32_t> seeds;
        if (o.contains("seeds"))
            for (double v : o.at("seeds").as_double_vector()) seeds.push_back(to_word(v));
        corehydro::numerics::sampling::MersenneTwister prng =
            seeds.empty()
                ? corehydro::numerics::sampling::MersenneTwister(
                      to_word(require_double(o, "seed", "rng/probe")))
                : corehydro::numerics::sampling::MersenneTwister(seeds);

        GuardedCall<std::vector<double>, const std::vector<double>&,
                    corehydro::numerics::sampling::MersenneTwister&>
            g(cbs.vector_rng, std::vector<double>{});
        std::vector<double> drawn;
        try {
            drawn = g(parameters, prng);
        } catch (...) {
            g.rethrow_if_aborted();
            throw;
        }
        g.rethrow_if_aborted();

        r.values = drawn;
        r.dims = {static_cast<int>(drawn.size())};
        return r;
    }

    throw std::invalid_argument("unknown rng method: " + method);
}

}  // namespace corehydro::numerics::support::detail
