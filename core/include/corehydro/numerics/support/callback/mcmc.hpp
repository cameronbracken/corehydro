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
// GUARD DISCIPLINE. The log-likelihood guard is built on a SHARED abort state even though this
// group has one callback today, because the Gibbs proposal and the HMC/NUTS gradient join it in
// the next task and a group with two live callbacks must abort as one (see callback_guard.hpp's
// "USE A SHARED STATE" note). The drive site is wrapped and the trailing rethrow is kept, and
// both halves earn their place here: the -infinity sentinel is a value the samplers treat as an
// unconditionally rejected point, so an aborted run does NOT throw of its own accord -- it walks
// the whole chain rejecting everything and returns a finished-looking result, which only the
// TRAILING rethrow catches. The wrap catches the other half: DifferentialEvolution on the MAP
// path, handed nothing but -infinity, can throw from its own internals (a singular Hessian
// inverse) before the latch is ever consulted.
#pragma once
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/support/dist_spec.hpp"
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

inline CallbackResult run_mcmc(const std::string& method, const JsonValue& o,
                               const CallbackSet& cbs) {
    if (method != "sample") throw std::invalid_argument("unknown mcmc method: " + method);
    if (!cbs.vector_scalar)
        throw std::invalid_argument("mcmc/sample requires a log-likelihood function");

    chmcmc::PriorList priors = mcmc_priors(o);
    std::string sampler_type = o.value_or("sampler", "RWMH");

    // Shared abort state from the start: Task 5's proposal and gradient guards join THIS state,
    // and a group with more than one live callback has to abort as one. See the file header.
    CallbackAbortStatePtr abort = make_abort_state();
    GuardedCall<double, const std::vector<double>&> log_likelihood(
        cbs.vector_scalar, -std::numeric_limits<double>::infinity(), abort);

    std::unique_ptr<chmcmc::MCMCSampler> sampler = chmcmc::build_sampler(
        sampler_type, std::move(priors),
        [&log_likelihood](const std::vector<double>& p) { return log_likelihood(p); },
        mcmc_settings(o));

    // See the file header on why BOTH halves are load-bearing here.
    try {
        sampler->sample();
    } catch (...) {
        log_likelihood.rethrow_if_aborted();
        throw;
    }
    log_likelihood.rethrow_if_aborted();

    return mcmc_flatten(chmcmc::collect_run(*sampler));
}

}  // namespace corehydro::numerics::support::detail
