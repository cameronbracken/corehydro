// cpp11 glue exposing the Bootstrap surface (model registry + regular/studentized runs) of
// the shared C++ core to R. Like `mcmc_sampler` fixtures, `bootstrap` fixtures are inherently
// STATEFUL -- one model construct + run + get_confidence_intervals() call backs every
// assertion in a case (see fixtures/README.md's bootstrap schema) -- so this file exposes ONE
// function, `ch_bootstrap_run_`, that builds the model, runs it once, computes confidence
// intervals once, and returns every value test-fixtures.R's dispatcher needs in one named
// list. Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <string>
#include <vector>

#include "corehydro/numerics/sampling/bootstrap/bootstrap.hpp"
#include "corehydro/numerics/sampling/bootstrap/ci_method_names.hpp"
#include "corehydro/numerics/sampling/bootstrap/model_registry.hpp"

namespace bs = corehydro::numerics::sampling;
using namespace cpp11;

// The name-to-enum mapping lives in the core (ci_method_names.hpp) rather than here: this glue,
// the pybind11 glue, the C++ fixture runner and the bootstrap callback group all need it, and
// four copies is how a sixth enum member reaches only some of them. cpp11 turns the
// std::invalid_argument it throws into an R error with the same message.

// Builds the named model, configures it with `construct` (a named R list; see
// fixtures/README.md's bootstrap schema for every key), runs it once, computes confidence
// intervals once, and returns a named list with the full result surface
// `test-fixtures.R`'s bootstrap dispatcher reads assertions from:
//   statistic_lower_ci/statistic_upper_ci -- [n_stats] vectors
//   parameter_lower_ci/parameter_upper_ci/population_estimate -- [n_params] vectors
//   valid_count                           -- [n_stats] vector (StatisticResults[i].ValidCount)
//   replicate_values                      -- [replicates x n_params] matrix
//     (BootstrapParameterSets[idx].Values[p])
[[cpp11::register]]
list ch_bootstrap_run_(std::string model, double mu, double sigma, int sample_size, doubles probabilities,
                        doubles dataset, int replicates, int seed, int max_retries, std::string run,
                        std::string ci_method, double alpha) {
    std::vector<double> probs(probabilities.begin(), probabilities.end());
    std::vector<double> sample_data(dataset.begin(), dataset.end());

    auto boot = bs::build_bootstrap_model(model, mu, sigma, sample_size, probs, sample_data);
    boot.replicates = replicates;
    boot.prng_seed = seed;
    boot.max_retries = max_retries;

    if (run == "regular") {
        boot.run();
    } else if (run == "studentized") {
        boot.run_with_studentized_bootstrap();
    } else {
        stop("unknown bootstrap run kind '%s'", run.c_str());
    }

    bs::BootstrapResults results = boot.get_confidence_intervals(bs::parse_bootstrap_ci_method(ci_method), alpha);

    int n_stats = static_cast<int>(results.statistic_results.size());
    int n_params = static_cast<int>(results.parameter_results.size());

    writable::doubles stat_lo(n_stats), stat_hi(n_stats), valid_count(n_stats);
    for (int i = 0; i < n_stats; ++i) {
        stat_lo[i] = results.statistic_results[static_cast<std::size_t>(i)].lower_ci;
        stat_hi[i] = results.statistic_results[static_cast<std::size_t>(i)].upper_ci;
        valid_count[i] = results.statistic_results[static_cast<std::size_t>(i)].valid_count;
    }

    writable::doubles parm_lo(n_params), parm_hi(n_params), pop_est(n_params);
    for (int p = 0; p < n_params; ++p) {
        parm_lo[p] = results.parameter_results[static_cast<std::size_t>(p)].lower_ci;
        parm_hi[p] = results.parameter_results[static_cast<std::size_t>(p)].upper_ci;
        pop_est[p] = results.parameter_results[static_cast<std::size_t>(p)].population_estimate;
    }

    const auto& parameter_sets = boot.bootstrap_parameter_sets();
    int n_reps = static_cast<int>(parameter_sets.size());
    writable::doubles_matrix<by_column> replicate_values(n_reps, n_params);
    for (int idx = 0; idx < n_reps; ++idx)
        for (int p = 0; p < n_params; ++p)
            replicate_values(idx, p) = parameter_sets[static_cast<std::size_t>(idx)].values[static_cast<std::size_t>(p)];

    return writable::list({
        "statistic_lower_ci"_nm = stat_lo,
        "statistic_upper_ci"_nm = stat_hi,
        "parameter_lower_ci"_nm = parm_lo,
        "parameter_upper_ci"_nm = parm_hi,
        "population_estimate"_nm = pop_est,
        "valid_count"_nm = valid_count,
        "replicate_values"_nm = replicate_values,
    });
}
