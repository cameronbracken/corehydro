// pybind11 glue exposing the MCMC sampler surface of the shared C++ core to Python, over the
// built-in model registry (corehydropy.mcmc's mcmc_sample()). Unlike the per-method dispatch
// style used elsewhere in this package (dist_val, cop_val, ...), `mcmc_sampler` fixtures are
// inherently STATEFUL -- one sampler construct + settings + a single sample() run backs
// every assertion in a case (see fixtures/README.md's mcmc_sampler schema) -- so this file
// exposes ONE function, `mcmc_run`, that builds the model, configures and runs the sampler
// once, and returns every value test_fixtures.py's dispatcher needs in one dict.
//
// The sampler-construction switch and the result post-processing are NOT here: both live in
// numerics/sampling/mcmc/support/mcmc_run.hpp, shared with the R glue and with the callback path
// behind mcmc_posterior() (numerics/support/callback/mcmc.hpp). This file's whole job is building
// the registry model, translating a settings dict into MCMCRunSettings, and packing an
// MCMCRunOutput into a dict.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "bindings.hpp"
#include "corehydro/numerics/sampling/mcmc/model_registry.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_run.hpp"

namespace py = pybind11;
namespace mcmc = corehydro::numerics::sampling::mcmc;

namespace {

// Reads `key` off the settings dict into `slot` when the caller set it, leaving it empty
// otherwise -- the MCMCRunSettings contract (an absent setting keeps the ported default).
template <typename T>
void read_setting(const py::dict& settings, const char* key, std::optional<T>& slot) {
    if (settings.contains(key)) slot = settings[key].cast<T>();
}

mcmc::MCMCRunSettings read_settings(const py::dict& settings) {
    mcmc::MCMCRunSettings s;
    read_setting(settings, "initialize", s.initialize);
    read_setting(settings, "prng_seed", s.prng_seed);
    read_setting(settings, "initial_iterations", s.initial_iterations);
    read_setting(settings, "warmup_iterations", s.warmup_iterations);
    read_setting(settings, "iterations", s.iterations);
    read_setting(settings, "number_of_chains", s.number_of_chains);
    read_setting(settings, "thinning_interval", s.thinning_interval);
    read_setting(settings, "output_length", s.output_length);
    read_setting(settings, "proposal_sigma", s.proposal_sigma);
    read_setting(settings, "step_size", s.step_size);
    read_setting(settings, "steps", s.steps);
    read_setting(settings, "max_tree_depth", s.max_tree_depth);
    read_setting(settings, "adapt_mass_matrix", s.adapt_mass_matrix);
    read_setting(settings, "scale", s.scale);
    read_setting(settings, "beta", s.beta);
    read_setting(settings, "jump", s.jump);
    read_setting(settings, "jump_threshold", s.jump_threshold);
    read_setting(settings, "snooker_threshold", s.snooker_threshold);
    read_setting(settings, "noise", s.noise);
    return s;
}

}  // namespace

// Packs an MCMCRunOutput into the dict test_fixtures.py's mcmc_sampler dispatcher and
// corehydropy.mcmc's mcmc_sample() read from. Shared with corehydropy.callback's
// mcmc_posterior(), which rebuilds the identical shape from the flat callback result -- see
// numerics/support/callback/mcmc.hpp.
//   chains              -- list of NumberOfChains [n_draws][n_params] nested lists
//                          (MarkovChains)
//   chain_fitness       -- list of NumberOfChains [n_draws] lists
//   acceptance_rates    -- [n_chains] list
//   map_values          -- [n_params] list (MCMCResults.MAP.Values)
//   map_fitness         -- scalar (MCMCResults.MAP.Fitness)
//   mean_log_likelihood -- [iterations] list
//   posterior_mean/sd/median/lower_ci/upper_ci -- [n_params] lists
//   rhat/ess            -- [n_params] lists
static py::dict pack_mcmc_run(const mcmc::MCMCRunOutput& o) {
    py::dict out;
    out["chains"] = o.chains;
    out["chain_fitness"] = o.chain_fitness;
    out["acceptance_rates"] = o.acceptance_rates;
    out["map_values"] = o.map_values;
    out["map_fitness"] = o.map_fitness;
    out["mean_log_likelihood"] = o.mean_log_likelihood;
    out["posterior_mean"] = o.posterior_mean;
    out["posterior_sd"] = o.posterior_sd;
    out["posterior_median"] = o.posterior_median;
    out["posterior_lower_ci"] = o.posterior_lower_ci;
    out["posterior_upper_ci"] = o.posterior_upper_ci;
    out["rhat"] = o.rhat;
    out["ess"] = o.ess;
    return out;
}

void register_mcmc(py::module_& m) {
    // Builds the named registry model, configures `sampler_type` with `settings` (a dict; every
    // key optional, matching fixtures/README.md), samples once, and returns the packed run.
    m.def(
        "mcmc_run",
        [](const std::string& sampler_type, const std::string& model_name, const std::string& family,
           const std::vector<double>& dataset, const py::dict& settings) {
            auto model = mcmc::build_model(model_name, family, dataset);
            mcmc::MCMCRunCallbacks callbacks;
            callbacks.proposal = model.proposal;
            std::unique_ptr<mcmc::MCMCSampler> sampler = mcmc::build_sampler(
                sampler_type, model.priors, model.log_likelihood, read_settings(settings), callbacks);
            sampler->sample();
            return pack_mcmc_run(mcmc::collect_run(*sampler));
        },
        py::arg("sampler_type"), py::arg("model_name"), py::arg("family"), py::arg("dataset"),
        py::arg("settings"));
}
