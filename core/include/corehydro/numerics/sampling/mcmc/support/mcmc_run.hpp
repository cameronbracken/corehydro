// corehydro ADDITION -- no upstream C# counterpart (sibling of mcmc_results.hpp, and of the
// runner-support headers distributions/support/dist_runner.hpp and
// estimation/support/fit_runner.hpp).
//
// THE one place an MCMC sampler is constructed from a name plus settings, and THE one place a
// finished sampler's posterior surface is read back out. Every caller that runs a chain drives
// these two functions:
//
//   - the registry path: corehydror/src/mcmc.cpp and corehydropy/src/bindings/mcmc.cpp, behind
//     mcmc_sample(), whose priors come from sampling/mcmc/model_registry.hpp;
//   - the callback path: numerics/support/callback/mcmc.hpp, behind mcmc_posterior(), whose
//     priors are user-supplied distributions and whose log-likelihood is a user-written R or
//     Python function.
//
// The two paths differ ONLY in where `priors` and `log_likelihood` come from. Everything after
// that -- which concrete sampler a name selects, which settings that sampler accepts, and what a
// run reports -- is identical, and lives here once so the two cannot drift: a sampler added to
// build_sampler() below is reachable from both paths on the same commit, and a field added to
// MCMCRunOutput is reported by both.
//
// The `callbacks` argument carries the two delegates upstream's own API takes but the registry
// path only half-uses: the Gibbs proposal (C# `Gibbs.Proposal`) and the HMC/NUTS gradient
// (C# `HMC.Gradient`). An empty gradient means the ported bound-aware finite-difference default,
// exactly as the HMC/NUTS constructors already document.
#pragma once
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/sampling/mcmc/arwmh.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "corehydro/numerics/sampling/mcmc/demcz.hpp"
#include "corehydro/numerics/sampling/mcmc/demczs.hpp"
#include "corehydro/numerics/sampling/mcmc/gibbs.hpp"
#include "corehydro/numerics/sampling/mcmc/hmc.hpp"
#include "corehydro/numerics/sampling/mcmc/nuts.hpp"
#include "corehydro/numerics/sampling/mcmc/rwmh.hpp"
#include "corehydro/numerics/sampling/mcmc/snis.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_results.hpp"

namespace corehydro::numerics::sampling::mcmc {

using PriorList = std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>>;

// Every setting either path may pass, each absent unless the caller set it. An absent setting
// leaves the ported sampler's OWN default in force rather than a copy of that default made here,
// so a change to a C# default lands in one place (the same rule callback/math.hpp follows for the
// integrator tolerances).
struct MCMCRunSettings {
    // Shared, base-class settings.
    std::optional<std::string> initialize;  // "MAP" | "Randomize" | "UserDefined"
    std::optional<int> prng_seed;
    std::optional<int> initial_iterations;
    std::optional<int> warmup_iterations;
    std::optional<int> iterations;
    std::optional<int> number_of_chains;
    std::optional<int> thinning_interval;
    std::optional<int> output_length;

    // RWMH: the proposal covariance sentinel, "zeros" or "identity" (see parse_proposal_sigma).
    std::optional<std::string> proposal_sigma;
    // HMC and NUTS.
    std::optional<double> step_size;
    std::optional<int> steps;            // HMC
    std::optional<int> max_tree_depth;   // NUTS
    std::optional<bool> adapt_mass_matrix;  // NUTS
    // ARWMH.
    std::optional<double> scale;
    std::optional<double> beta;
    // DEMCz and DEMCzs.
    std::optional<double> jump;
    std::optional<double> jump_threshold;
    std::optional<double> snooker_threshold;  // DEMCzs only
    std::optional<double> noise;
};

// The delegates upstream's own sampler API takes beside the log-likelihood. Both are empty on the
// registry path unless the registry model supplies one (only "normal_conjugate_gibbs" does).
struct MCMCRunCallbacks {
    // (current parameters, this chain's PRNG) -> proposed parameters. Required by Gibbs.
    Gibbs::Proposal proposal;
    // (parameters) -> gradient vector. Optional for HMC/NUTS; empty selects the ported
    // bound-aware finite-difference gradient the constructors install.
    HMC::Gradient gradient;
};

// `proposal_sigma` sentinel strings -- see fixtures/README.md's mcmc_sampler schema for why
// "identity" exists alongside the C# test's literal "zeros" (an all-zero proposal covariance is
// only safe when MAP initialization is expected to overwrite it before first use).
inline math::linalg::Matrix parse_proposal_sigma(const std::string& s, int dimension) {
    if (s == "zeros") return math::linalg::Matrix(dimension);
    if (s == "identity") return math::linalg::Matrix::identity(dimension);
    throw std::invalid_argument("unknown proposal_sigma sentinel: " + s);
}

inline MCMCSampler::InitializationType parse_initialize(const std::string& s) {
    if (s == "MAP") return MCMCSampler::InitializationType::MAP;
    if (s == "Randomize") return MCMCSampler::InitializationType::Randomize;
    if (s == "UserDefined") return MCMCSampler::InitializationType::UserDefined;
    throw std::invalid_argument("unknown initialize value: " + s);
}

// The eight sampler names build_sampler() accepts, in the order the packages document them.
inline const std::vector<std::string>& sampler_names() {
    static const std::vector<std::string> names = {"RWMH",  "ARWMH", "DEMCz", "DEMCzs",
                                                   "HMC",   "NUTS",  "SNIS",  "Gibbs"};
    return names;
}

// Builds `sampler_type` over `priors` and `log_likelihood`, applies every setting the caller set,
// and returns it ready for sample(). The ONLY switch over sampler names in the codebase.
inline std::unique_ptr<MCMCSampler> build_sampler(const std::string& sampler_type, PriorList priors,
                                                  LogLikelihood log_likelihood,
                                                  const MCMCRunSettings& s,
                                                  const MCMCRunCallbacks& callbacks = {}) {
    if (priors.empty()) throw std::invalid_argument("at least one prior distribution is required");
    int d = static_cast<int>(priors.size());

    std::unique_ptr<MCMCSampler> sampler;
    if (sampler_type == "RWMH") {
        math::linalg::Matrix proposal_sigma =
            s.proposal_sigma ? parse_proposal_sigma(*s.proposal_sigma, d) : math::linalg::Matrix(d);
        sampler = std::make_unique<RWMH>(std::move(priors), std::move(log_likelihood), proposal_sigma);
    } else if (sampler_type == "HMC") {
        sampler = std::make_unique<HMC>(std::move(priors), std::move(log_likelihood), std::nullopt,
                                        s.step_size.value_or(0.1), s.steps.value_or(50),
                                        callbacks.gradient);
    } else if (sampler_type == "NUTS") {
        auto nuts = std::make_unique<NUTS>(std::move(priors), std::move(log_likelihood), std::nullopt,
                                           s.step_size.value_or(0.1), s.max_tree_depth.value_or(10),
                                           callbacks.gradient);
        if (s.adapt_mass_matrix) nuts->adapt_mass_matrix = *s.adapt_mass_matrix;
        sampler = std::move(nuts);
    } else if (sampler_type == "ARWMH") {
        auto arwmh = std::make_unique<ARWMH>(std::move(priors), std::move(log_likelihood));
        if (s.scale) arwmh->scale = *s.scale;
        if (s.beta) arwmh->beta = *s.beta;
        sampler = std::move(arwmh);
    } else if (sampler_type == "Gibbs") {
        if (!callbacks.proposal)
            throw std::invalid_argument("the Gibbs sampler requires a proposal function");
        sampler = std::make_unique<Gibbs>(std::move(priors), std::move(log_likelihood),
                                          callbacks.proposal);
    } else if (sampler_type == "SNIS") {
        sampler = std::make_unique<SNIS>(std::move(priors), std::move(log_likelihood));
    } else if (sampler_type == "DEMCz") {
        auto demcz = std::make_unique<DEMCz>(std::move(priors), std::move(log_likelihood));
        if (s.jump) demcz->jump = *s.jump;
        if (s.jump_threshold) demcz->jump_threshold = *s.jump_threshold;
        if (s.noise) demcz->set_noise(*s.noise);
        sampler = std::move(demcz);
    } else if (sampler_type == "DEMCzs") {
        auto demczs = std::make_unique<DEMCzs>(std::move(priors), std::move(log_likelihood));
        if (s.jump) demczs->jump = *s.jump;
        if (s.jump_threshold) demczs->jump_threshold = *s.jump_threshold;
        if (s.snooker_threshold) demczs->snooker_threshold = *s.snooker_threshold;
        if (s.noise) demczs->set_noise(*s.noise);
        sampler = std::move(demczs);
    } else {
        throw std::invalid_argument("unknown MCMC sampler '" + sampler_type + "'");
    }

    if (s.initialize) sampler->initialize = parse_initialize(*s.initialize);
    if (s.prng_seed) sampler->set_prng_seed(*s.prng_seed);
    if (s.initial_iterations) sampler->set_initial_iterations(*s.initial_iterations);
    if (s.warmup_iterations) sampler->set_warmup_iterations(*s.warmup_iterations);
    if (s.iterations) sampler->set_iterations(*s.iterations);
    if (s.number_of_chains) sampler->set_number_of_chains(*s.number_of_chains);
    if (s.thinning_interval) sampler->set_thinning_interval(*s.thinning_interval);
    if (s.output_length) sampler->output_length = *s.output_length;
    return sampler;
}

// Everything a finished run reports, in language-neutral containers. The R glue turns this into a
// named list, the Python glue into a dict, and callback/mcmc.hpp flattens it into a
// CallbackResult -- none of them reads the sampler or the MCMCResults directly, so all three
// report the same fields.
struct MCMCRunOutput {
    int number_of_chains = 0;
    int number_of_parameters = 0;
    // [chain][draw][parameter] and [chain][draw].
    std::vector<std::vector<std::vector<double>>> chains;
    std::vector<std::vector<double>> chain_fitness;
    std::vector<double> acceptance_rates;
    std::vector<double> map_values;
    double map_fitness = 0.0;
    std::vector<double> mean_log_likelihood;
    std::vector<double> posterior_mean;
    std::vector<double> posterior_sd;
    std::vector<double> posterior_median;
    std::vector<double> posterior_lower_ci;
    std::vector<double> posterior_upper_ci;
    std::vector<double> rhat;
    std::vector<double> ess;
};

// Post-processes `sampler` (which must already have been sample()d) into MCMCRunOutput.
inline MCMCRunOutput collect_run(const MCMCSampler& sampler) {
    MCMCResults results(sampler);
    MCMCRunOutput o;
    o.number_of_chains = sampler.number_of_chains();
    o.number_of_parameters = sampler.number_of_parameters();
    const std::size_t c_n = static_cast<std::size_t>(o.number_of_chains);
    const std::size_t p_n = static_cast<std::size_t>(o.number_of_parameters);

    o.chains.resize(c_n);
    o.chain_fitness.resize(c_n);
    for (std::size_t c = 0; c < c_n; ++c) {
        const auto& chain = sampler.markov_chains()[c];
        o.chains[c].resize(chain.size());
        o.chain_fitness[c].resize(chain.size());
        for (std::size_t i = 0; i < chain.size(); ++i) {
            o.chains[c][i] = chain[i].values;
            o.chain_fitness[c][i] = chain[i].fitness;
        }
    }

    o.acceptance_rates = sampler.acceptance_rates();
    o.mean_log_likelihood = sampler.mean_log_likelihood();
    o.map_values = results.map.values;
    o.map_fitness = results.map.fitness;

    o.posterior_mean.resize(p_n);
    o.posterior_sd.resize(p_n);
    o.posterior_median.resize(p_n);
    o.posterior_lower_ci.resize(p_n);
    o.posterior_upper_ci.resize(p_n);
    o.rhat.resize(p_n);
    o.ess.resize(p_n);
    for (std::size_t j = 0; j < p_n; ++j) {
        const auto& stats = results.parameter_results[j].summary_statistics;
        o.posterior_mean[j] = stats.mean;
        o.posterior_sd[j] = stats.standard_deviation;
        o.posterior_median[j] = stats.median;
        o.posterior_lower_ci[j] = stats.lower_ci;
        o.posterior_upper_ci[j] = stats.upper_ci;
        o.rhat[j] = stats.rhat;
        o.ess[j] = stats.ess;
    }
    return o;
}

}  // namespace corehydro::numerics::sampling::mcmc
