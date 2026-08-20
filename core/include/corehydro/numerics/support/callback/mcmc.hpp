// corehydro ADDITION -- no upstream C# counterpart. The mcmc group of callback_runner.hpp:
// running the ported samplers against a user-written log-likelihood, over user-supplied priors.
//
// This is upstream's OWN constructor, not an addition:
// `MCMCSampler(List<IUnivariateDistribution> priorDistributions, LogLikelihood
// logLikelihoodFunction)` (Sampling/MCMC/Base/MCMCSampler.cs). The model registries under
// sampling/mcmc/ are corehydro fixture scaffolding; a caller of the real C# API passes its own
// priors and its own delegate, which is what this group finally makes reachable from R and Python.
//
// One method, `sample`. Options grammar (JSON object):
//
//   {"sampler": "RWMH", "priors": [{"family": "Uniform", "parameters": [0.0, 10.0]}, ...],
//    "iterations": 2000, "warmup": 500, "chains": 2, "thinning": 1, "seed": 12345,
//    "initialize": "MAP"}
//
// `priors` is REQUIRED and is the same distribution-spec grammar dist_spec.hpp builds from, so a
// prior may be any family the packages can construct (including a composite) rather than a second
// hand-rolled family parser. Its length IS the parameter count: the log-likelihood is called with
// a vector of exactly that many values.
//
// The five short keys above are the user-facing spellings; the sampler's own setting names are
// also accepted for the fields no short name covers, and every one is optional (an absent key
// leaves the ported sampler's default in force -- see mcmc_run.hpp's MCMCRunSettings):
// `initial_iterations`, `output_length`, `proposal_sigma` (RWMH), `step_size`/`steps` (HMC),
// `step_size`/`max_tree_depth`/`adapt_mass_matrix` (NUTS), `scale`/`beta` (ARWMH),
// `jump`/`jump_threshold`/`noise` (DEMCz), plus `snooker_threshold` (DEMCzs).
//
// COST NOTE, and it is a real one: `initialize` defaults to the C# default `MAP`, which runs
// DifferentialEvolution over the user's log-likelihood BEFORE the first chain iteration, then
// draws `initial_iterations` more evaluations from the fitted multivariate normal. Every one of
// those is a crossing back into R or Python. `Randomize` skips the optimizer entirely and costs
// `initial_iterations` evaluations. The chain itself costs
// `(iterations + ceil(output_length / chains)) * thinning * chains` more.
//
// Result layout. A run reports far more numbers than a `values` vector can usefully name, so the
// flat CallbackResult is split, and `dims` says where:
//
//   dims   = {n_summary, n_chains, n_draws, n_parameters}
//   values = [ the n_summary summary scalars, then n_chains * n_draws * n_parameters draws ]
//   names  = the n_summary summary names (values beyond that are the draw block, unnamed)
//
// The summary block is, in order: `map_fitness`; `acceptance_rate[c]` for each chain; then
// `map[j]`, `posterior_mean[j]`, `posterior_sd[j]`, `posterior_median[j]`,
// `posterior_lower_ci[j]`, `posterior_upper_ci[j]`, `rhat[j]` and `ess[j]`, each running over all
// parameters before the next statistic starts. So n_summary = 1 + n_chains + 8 * n_parameters,
// and a caller slices by name rather than by arithmetic. The draw block is row-major
// [chain][draw][parameter].
//
// THE OTHER TWO DELEGATES. Upstream's samplers take two more caller-written functions beside the
// log-likelihood, and both are reachable here:
//
//   - `proposal`, C# `Gibbs.Proposal(double[] parameters, Random prng)`, REQUIRED by Gibbs and
//     accepted by nothing else. It is handed the current parameter vector and THIS CHAIN'S
//     generator (as a borrowed handle -- see rng_handle.hpp) and returns the next parameter
//     vector, which Gibbs accepts unconditionally. That is why Gibbs was the one ported sampler
//     neither package could reach until this callback existed: a conditional proposal is
//     model-specific, so there is nothing sensible to default it to.
//   - `gradient`, C# `HMC.Gradient(IList<double> parameters)`, OPTIONAL for HMC and NUTS and
//     accepted by nothing else. Left unsupplied, the ported constructors install their own
//     bound-aware finite-difference gradient of SafeLogLikelihood, and that default stays in
//     force -- the empty `std::function` reaches the constructor unchanged rather than being
//     replaced here by a hand-rolled equivalent.
//
// A proposal handed to a non-Gibbs sampler, or a gradient handed to anything but HMC/NUTS, is an
// error rather than a silent no-op: a user who writes a gradient and picks RWMH by mistake would
// otherwise get a plausible-looking run that never called their function.
//
// GUARD DISCIPLINE. All three guards share ONE abort state, and that is load-bearing rather than
// tidy. With private states, an R error raised inside the log-likelihood latches only that guard,
// and Gibbs -- which does not know it is unwinding -- calls the proposal on the next iteration,
// re-entering R with an unwind already pending. Sharing makes the first throw short-circuit every
// callback in the group (see callback_guard.hpp's "USE A SHARED STATE" note). Each guard keeps its
// OWN sentinel, because the right "worst value" differs: -infinity for the log-likelihood (an
// unconditionally rejected point), an empty vector for the proposal (which the Gibbs arm turns
// back into the current state, i.e. a rejected proposal), and a zero vector for the gradient (a
// flat surface, which the leapfrog integrator handles without complaint).
//
// The drive site is wrapped AND the trailing rethrow is kept, and both halves earn their place
// here: the sentinels are all values the samplers treat as legal, so an aborted run does NOT throw
// of its own accord -- it walks the whole chain rejecting everything and returns a
// finished-looking result, which only the TRAILING rethrow catches. The wrap catches the other
// half: DifferentialEvolution on the MAP path, handed nothing but -infinity, can throw from its
// own internals (a singular Hessian inverse) before the latch is ever consulted.
#pragma once
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/support/dist_spec.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_run.hpp"
#include "corehydro/numerics/support/callback/common.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"

namespace corehydro::numerics::support::detail {

namespace chmcmc = corehydro::numerics::sampling::mcmc;

// Builds the prior list through the shared distribution-spec builder -- the same grammar and the
// same code path a user's distribution() object serializes to, so there is no second family
// parser to keep in step.
inline chmcmc::PriorList mcmc_priors(const JsonValue& o) {
    if (!o.contains("priors"))
        throw std::invalid_argument("mcmc/sample requires the option 'priors'");
    const std::vector<JsonValue>& specs = o.at("priors").items();
    if (specs.empty())
        throw std::invalid_argument("mcmc/sample requires at least one prior distribution");
    chmcmc::PriorList priors;
    priors.reserve(specs.size());
    for (const JsonValue& spec : specs)
        priors.push_back(std::shared_ptr<distributions::UnivariateDistributionBase>(
            distributions::support::build_univariate(spec)));
    return priors;
}

// Reads the options object into MCMCRunSettings. The five short user-facing keys are accepted
// alongside the sampler's own setting names; a key absent from the options leaves the ported
// default in force.
inline chmcmc::MCMCRunSettings mcmc_settings(const JsonValue& o) {
    chmcmc::MCMCRunSettings s;
    auto read_int = [&o](const char* key, std::optional<int>& slot) {
        if (o.contains(key)) slot = o.at(key).as_int();
    };
    auto read_double = [&o](const char* key, std::optional<double>& slot) {
        if (o.contains(key)) slot = o.at(key).as_double();
    };
    auto read_string = [&o](const char* key, std::optional<std::string>& slot) {
        if (o.contains(key)) slot = o.at(key).as_string();
    };

    read_string("initialize", s.initialize);
    read_int("seed", s.prng_seed);
    read_int("prng_seed", s.prng_seed);
    read_int("initial_iterations", s.initial_iterations);
    read_int("warmup", s.warmup_iterations);
    read_int("warmup_iterations", s.warmup_iterations);
    read_int("iterations", s.iterations);
    read_int("chains", s.number_of_chains);
    read_int("number_of_chains", s.number_of_chains);
    read_int("thinning", s.thinning_interval);
    read_int("thinning_interval", s.thinning_interval);
    read_int("output_length", s.output_length);
    read_string("proposal_sigma", s.proposal_sigma);
    read_double("step_size", s.step_size);
    read_int("steps", s.steps);
    read_int("max_tree_depth", s.max_tree_depth);
    if (o.contains("adapt_mass_matrix")) s.adapt_mass_matrix = o.at("adapt_mass_matrix").as_bool();
    read_double("scale", s.scale);
    read_double("beta", s.beta);
    read_double("jump", s.jump);
    read_double("jump_threshold", s.jump_threshold);
    read_double("snooker_threshold", s.snooker_threshold);
    read_double("noise", s.noise);
    return s;
}

// Flattens a finished run into the layout documented in this file's header.
inline CallbackResult mcmc_flatten(const chmcmc::MCMCRunOutput& o) {
    CallbackResult r;
    r.status = "Success";
    const int c_n = o.number_of_chains;
    const int p_n = o.number_of_parameters;
    const int draws = o.chains.empty() ? 0 : static_cast<int>(o.chains.front().size());

    auto push = [&r](const std::string& name, double value) {
        r.names.push_back(name);
        r.values.push_back(value);
    };
    auto push_each = [&](const char* label, const std::vector<double>& v) {
        for (std::size_t j = 0; j < v.size(); ++j)
            push(std::string(label) + "[" + std::to_string(j) + "]", v[j]);
    };

    push("map_fitness", o.map_fitness);
    push_each("acceptance_rate", o.acceptance_rates);
    push_each("map", o.map_values);
    push_each("posterior_mean", o.posterior_mean);
    push_each("posterior_sd", o.posterior_sd);
    push_each("posterior_median", o.posterior_median);
    push_each("posterior_lower_ci", o.posterior_lower_ci);
    push_each("posterior_upper_ci", o.posterior_upper_ci);
    push_each("rhat", o.rhat);
    push_each("ess", o.ess);
    const int n_summary = static_cast<int>(r.values.size());

    r.values.reserve(r.values.size() +
                     static_cast<std::size_t>(c_n) * static_cast<std::size_t>(draws) *
                         static_cast<std::size_t>(p_n));
    for (const auto& chain : o.chains)
        for (const auto& draw : chain) r.values.insert(r.values.end(), draw.begin(), draw.end());

    r.dims = {n_summary, c_n, draws, p_n};
    return r;
}

// The one sentence naming the samplers each optional delegate belongs to, so the two checks below
// and the R/Python wrappers cannot drift apart in wording.
inline bool sampler_takes_gradient(const std::string& sampler_type) {
    return sampler_type == "HMC" || sampler_type == "NUTS";
}

inline CallbackResult run_mcmc(const std::string& method, const JsonValue& o,
                               const CallbackSet& cbs) {
    if (method != "sample") throw std::invalid_argument("unknown mcmc method: " + method);
    if (!cbs.vector_scalar)
        throw std::invalid_argument("mcmc/sample requires a log-likelihood function");

    chmcmc::PriorList priors = mcmc_priors(o);
    const std::size_t n_parameters = priors.size();
    std::string sampler_type = o.value_or("sampler", "RWMH");

    if (cbs.vector_rng && sampler_type != "Gibbs")
        throw std::invalid_argument("a proposal function is only used by the Gibbs sampler; '" +
                                    sampler_type + "' does not take one");
    if (cbs.vector_vector && !sampler_takes_gradient(sampler_type))
        throw std::invalid_argument(
            "a gradient function is only used by the HMC and NUTS samplers; '" + sampler_type +
            "' does not take one");

    // ONE abort state for all three guards -- see the file header on why sharing it is what stops
    // a latched log-likelihood from letting the sampler re-enter the host through the proposal.
    CallbackAbortStatePtr abort = make_abort_state();
    GuardedCall<double, const std::vector<double>&> log_likelihood(
        cbs.vector_scalar, -std::numeric_limits<double>::infinity(), abort);

    // The length check lives INSIDE the guarded function rather than around it, so a proposal or
    // gradient of the wrong length aborts the run exactly as a host-language error does: latched
    // on the first occurrence, every later callback short-circuited, and the message rethrown from
    // the protected frame instead of surfacing as whatever the sampler does with a short vector.
    std::function<std::vector<double>(const std::vector<double>&, sampling::MersenneTwister&)>
        proposal_fn;
    if (cbs.vector_rng) {
        proposal_fn = [fn = cbs.vector_rng, n_parameters](
                          const std::vector<double>& p,
                          sampling::MersenneTwister& prng) -> std::vector<double> {
            std::vector<double> xp = fn(p, prng);
            if (xp.size() != n_parameters)
                throw std::invalid_argument(
                    "the proposal function must return one value per parameter; it was given " +
                    std::to_string(n_parameters) + " and returned " + std::to_string(xp.size()));
            return xp;
        };
    }
    GuardedCall<std::vector<double>, const std::vector<double>&, sampling::MersenneTwister&>
        proposal(proposal_fn, std::vector<double>{}, abort);

    std::function<std::vector<double>(const std::vector<double>&)> gradient_fn;
    if (cbs.vector_vector) {
        gradient_fn = [fn = cbs.vector_vector,
                       n_parameters](const std::vector<double>& p) -> std::vector<double> {
            std::vector<double> g = fn(p);
            if (g.size() != n_parameters)
                throw std::invalid_argument(
                    "the gradient function must return one value per parameter; it was given " +
                    std::to_string(n_parameters) + " and returned " + std::to_string(g.size()));
            return g;
        };
    }
    GuardedCall<std::vector<double>, const std::vector<double>&> gradient(
        gradient_fn, std::vector<double>(n_parameters, 0.0), abort);

    chmcmc::MCMCRunCallbacks callbacks;
    if (cbs.vector_rng) {
        callbacks.proposal = [&proposal](const std::vector<double>& p,
                                          sampling::MersenneTwister& prng) {
            std::vector<double> xp = proposal(p, prng);
            // The guard's sentinel. Gibbs accepts every proposal unconditionally, so there is no
            // reject branch to hand an empty vector to; returning the CURRENT state is the same
            // thing said in the only vocabulary this sampler has, and it keeps the recorded chain
            // the right width for the run that is about to be thrown away anyway.
            return xp.empty() ? p : xp;
        };
    }
    if (cbs.vector_vector) {
        callbacks.gradient = [&gradient](const std::vector<double>& p) {
            return math::linalg::Vector(gradient(p));
        };
    }
    // No gradient supplied leaves `callbacks.gradient` empty, which is exactly what the ported
    // HMC/NUTS constructors read as "use the bound-aware finite-difference default".

    std::unique_ptr<chmcmc::MCMCSampler> sampler = chmcmc::build_sampler(
        sampler_type, std::move(priors),
        [&log_likelihood](const std::vector<double>& p) { return log_likelihood(p); },
        mcmc_settings(o), callbacks);

    // See the file header on why BOTH halves are load-bearing here. Rethrown off the shared
    // `abort` state directly (callback_guard.hpp's free `rethrow_if_aborted`) rather than off any
    // one guard's own method, so this drive site does not depend on WHICH of the three guards it
    // happens to ask -- all three share `abort`, but that should not be a fact a caller here needs
    // to know.
    try {
        sampler->sample();
    } catch (...) {
        rethrow_if_aborted(abort);
        throw;
    }
    rethrow_if_aborted(abort);

    return mcmc_flatten(chmcmc::collect_run(*sampler));
}

}  // namespace corehydro::numerics::support::detail
