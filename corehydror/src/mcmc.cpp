// cpp11 glue exposing the MCMC sampler surface of the shared C++ core to R, over the built-in
// model registry (R/mcmc.R's mcmc_sample()). Unlike the per-method dispatch style used elsewhere
// in this package (ch_dist_val_, ch_cop_val_, ...), `mcmc_sampler` fixtures are inherently
// STATEFUL -- one sampler construct + settings + a single sample() run backs every assertion
// in a case (see fixtures/README.md's mcmc_sampler schema) -- so this file exposes ONE
// function, `ch_mcmc_run_`, that builds the model, configures and runs the sampler once, and
// returns every value the test-fixtures.R dispatcher needs in one named list. This avoids a
// "seq machinery" batching mechanism entirely: there is only ever one run per case.
//
// The sampler-construction switch and the result post-processing are NOT here: both live in
// numerics/sampling/mcmc/support/mcmc_run.hpp, shared with the Python glue and with the callback
// path behind mcmc_posterior() (numerics/support/callback/mcmc.hpp). This file's whole job is
// building the registry model, translating an R settings list into MCMCRunSettings, and packing
// an MCMCRunOutput into an R list.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "corehydro/numerics/sampling/mcmc/model_registry.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_run.hpp"

namespace mcmc = corehydro::numerics::sampling::mcmc;
using namespace cpp11;

namespace {

// Reads `key` off the R settings list into `slot` when the caller set it, leaving it empty
// otherwise -- the MCMCRunSettings contract (an absent setting keeps the ported default).
void read_int(list settings, const char* key, std::optional<int>& slot) {
    SEXP v = settings[key];
    if (v != R_NilValue) slot = static_cast<int>(as_cpp<double>(v));
}

void read_double(list settings, const char* key, std::optional<double>& slot) {
    SEXP v = settings[key];
    if (v != R_NilValue) slot = as_cpp<double>(v);
}

void read_string(list settings, const char* key, std::optional<std::string>& slot) {
    SEXP v = settings[key];
    if (v != R_NilValue) slot = as_cpp<std::string>(v);
}

void read_bool(list settings, const char* key, std::optional<bool>& slot) {
    SEXP v = settings[key];
    if (v != R_NilValue) slot = as_cpp<bool>(v);
}

mcmc::MCMCRunSettings read_settings(list settings) {
    mcmc::MCMCRunSettings s;
    read_string(settings, "initialize", s.initialize);
    read_int(settings, "prng_seed", s.prng_seed);
    read_int(settings, "initial_iterations", s.initial_iterations);
    read_int(settings, "warmup_iterations", s.warmup_iterations);
    read_int(settings, "iterations", s.iterations);
    read_int(settings, "number_of_chains", s.number_of_chains);
    read_int(settings, "thinning_interval", s.thinning_interval);
    read_int(settings, "output_length", s.output_length);
    read_string(settings, "proposal_sigma", s.proposal_sigma);
    read_double(settings, "step_size", s.step_size);
    read_int(settings, "steps", s.steps);
    read_int(settings, "max_tree_depth", s.max_tree_depth);
    read_bool(settings, "adapt_mass_matrix", s.adapt_mass_matrix);
    read_double(settings, "scale", s.scale);
    read_double(settings, "beta", s.beta);
    read_double(settings, "jump", s.jump);
    read_double(settings, "jump_threshold", s.jump_threshold);
    read_double(settings, "snooker_threshold", s.snooker_threshold);
    read_double(settings, "noise", s.noise);
    return s;
}

writable::doubles as_doubles_vec(const std::vector<double>& v) {
    writable::doubles out(static_cast<R_xlen_t>(v.size()));
    for (std::size_t i = 0; i < v.size(); ++i) out[static_cast<R_xlen_t>(i)] = v[i];
    return out;
}

}  // namespace

// Packs an MCMCRunOutput into the named list test-fixtures.R's mcmc_sampler dispatcher and
// R/mcmc.R's mcmc_sample() read from. Shared with R/callback.R's mcmc_posterior(), which rebuilds
// the identical shape from the flat callback result -- see numerics/support/callback/mcmc.hpp.
//   chains             -- list of NumberOfChains [n_draws x n_params] matrices (MarkovChains)
//   chain_fitness      -- list of NumberOfChains [n_draws] vectors
//   acceptance_rates   -- [n_chains] vector
//   map_values         -- [n_params] vector (MCMCResults.MAP.Values)
//   map_fitness        -- scalar (MCMCResults.MAP.Fitness)
//   mean_log_likelihood -- [iterations] vector
//   posterior_mean/sd/median/lower_ci/upper_ci -- [n_params] vectors
//   rhat/ess           -- [n_params] vectors
static list pack_mcmc_run(const mcmc::MCMCRunOutput& o) {
    int n_chains = o.number_of_chains;
    int p = o.number_of_parameters;

    writable::list chains(n_chains);
    writable::list chain_fitness(n_chains);
    for (int c = 0; c < n_chains; ++c) {
        const auto& chain = o.chains[static_cast<std::size_t>(c)];
        int n = static_cast<int>(chain.size());
        writable::doubles_matrix<by_column> m(n, p);
        writable::doubles fit(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < p; ++j)
                m(i, j) = chain[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            fit[i] = o.chain_fitness[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)];
        }
        chains[c] = m;
        chain_fitness[c] = fit;
    }

    return writable::list({
        "chains"_nm = chains,
        "chain_fitness"_nm = chain_fitness,
        "acceptance_rates"_nm = as_doubles_vec(o.acceptance_rates),
        "map_values"_nm = as_doubles_vec(o.map_values),
        "map_fitness"_nm = writable::doubles({o.map_fitness}),
        "mean_log_likelihood"_nm = as_doubles_vec(o.mean_log_likelihood),
        "posterior_mean"_nm = as_doubles_vec(o.posterior_mean),
        "posterior_sd"_nm = as_doubles_vec(o.posterior_sd),
        "posterior_median"_nm = as_doubles_vec(o.posterior_median),
        "posterior_lower_ci"_nm = as_doubles_vec(o.posterior_lower_ci),
        "posterior_upper_ci"_nm = as_doubles_vec(o.posterior_upper_ci),
        "rhat"_nm = as_doubles_vec(o.rhat),
        "ess"_nm = as_doubles_vec(o.ess),
    });
}

// Builds the named registry model, configures `sampler_type` with `settings` (a named R list;
// every key optional, matching fixtures/README.md), samples once, and returns the packed run.
[[cpp11::register]]
list ch_mcmc_run_(std::string sampler_type, std::string model_name, std::string family,
                   doubles dataset, list settings) {
    std::vector<double> data(dataset.begin(), dataset.end());
    auto model = mcmc::build_model(model_name, family, data);
    mcmc::MCMCRunCallbacks callbacks;
    callbacks.proposal = model.proposal;
    std::unique_ptr<mcmc::MCMCSampler> sampler = mcmc::build_sampler(
        sampler_type, model.priors, model.log_likelihood, read_settings(settings), callbacks);
    sampler->sample();
    return pack_mcmc_run(mcmc::collect_run(*sampler));
}
