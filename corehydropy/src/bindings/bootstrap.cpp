// pybind11 glue exposing the Bootstrap surface (model registry + regular/studentized runs) of
// the shared C++ core to Python. Like `mcmc_sampler` fixtures, `bootstrap` fixtures are
// inherently STATEFUL -- one model construct + run + get_confidence_intervals() call backs
// every assertion in a case (see fixtures/README.md's bootstrap schema) -- so this file
// exposes ONE function, `bootstrap_run`, that builds the model, runs it once, computes
// confidence intervals once, and returns every value test_fixtures.py's dispatcher needs in
// one dict. Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "corehydro/numerics/sampling/bootstrap/bootstrap.hpp"
#include "corehydro/numerics/sampling/bootstrap/ci_method_names.hpp"
#include "corehydro/numerics/sampling/bootstrap/model_registry.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace bs = corehydro::numerics::sampling;

// The name-to-enum mapping lives in the core (ci_method_names.hpp) rather than here: this glue,
// the cpp11 glue, the C++ fixture runner and the bootstrap callback group all need it, and four
// copies is how a sixth enum member reaches only some of them. pybind11 turns the
// std::invalid_argument it throws into the same ValueError this file used to raise itself.

void register_bootstrap(py::module_& m) {
    // Builds the named model, runs it once, computes confidence intervals once, and returns a
    // dict with the full result surface test_fixtures.py's bootstrap dispatcher reads
    // assertions from:
    //   statistic_lower_ci/statistic_upper_ci -- [n_stats] lists
    //   parameter_lower_ci/parameter_upper_ci/population_estimate -- [n_params] lists
    //   valid_count       -- [n_stats] list (StatisticResults[i].ValidCount)
    //   replicate_values  -- [replicates][n_params] nested lists
    //     (BootstrapParameterSets[idx].Values[p])
    m.def(
        "bootstrap_run",
        [](const std::string& model, double mu, double sigma, int sample_size,
           const std::vector<double>& probabilities, const std::vector<double>& dataset, int replicates, int seed,
           int max_retries, const std::string& run, const std::string& ci_method, double alpha) {
            auto boot = bs::build_bootstrap_model(model, mu, sigma, sample_size, probabilities, dataset);
            boot.replicates = replicates;
            boot.prng_seed = seed;
            boot.max_retries = max_retries;

            if (run == "regular") {
                boot.run();
            } else if (run == "studentized") {
                boot.run_with_studentized_bootstrap();
            } else {
                throw py::value_error("unknown bootstrap run kind: " + run);
            }

            bs::BootstrapResults results = boot.get_confidence_intervals(bs::parse_bootstrap_ci_method(ci_method), alpha);

            int n_stats = static_cast<int>(results.statistic_results.size());
            int n_params = static_cast<int>(results.parameter_results.size());

            std::vector<double> stat_lo(static_cast<std::size_t>(n_stats)), stat_hi(static_cast<std::size_t>(n_stats)),
                valid_count(static_cast<std::size_t>(n_stats));
            for (int i = 0; i < n_stats; ++i) {
                stat_lo[static_cast<std::size_t>(i)] = results.statistic_results[static_cast<std::size_t>(i)].lower_ci;
                stat_hi[static_cast<std::size_t>(i)] = results.statistic_results[static_cast<std::size_t>(i)].upper_ci;
                valid_count[static_cast<std::size_t>(i)] =
                    results.statistic_results[static_cast<std::size_t>(i)].valid_count;
            }

            std::vector<double> parm_lo(static_cast<std::size_t>(n_params)), parm_hi(static_cast<std::size_t>(n_params)),
                pop_est(static_cast<std::size_t>(n_params));
            for (int p = 0; p < n_params; ++p) {
                parm_lo[static_cast<std::size_t>(p)] = results.parameter_results[static_cast<std::size_t>(p)].lower_ci;
                parm_hi[static_cast<std::size_t>(p)] = results.parameter_results[static_cast<std::size_t>(p)].upper_ci;
                pop_est[static_cast<std::size_t>(p)] =
                    results.parameter_results[static_cast<std::size_t>(p)].population_estimate;
            }

            const auto& parameter_sets = boot.bootstrap_parameter_sets();
            std::vector<std::vector<double>> replicate_values(parameter_sets.size());
            for (std::size_t idx = 0; idx < parameter_sets.size(); ++idx)
                replicate_values[idx] = parameter_sets[idx].values;

            py::dict out;
            out["statistic_lower_ci"] = stat_lo;
            out["statistic_upper_ci"] = stat_hi;
            out["parameter_lower_ci"] = parm_lo;
            out["parameter_upper_ci"] = parm_hi;
            out["population_estimate"] = pop_est;
            out["valid_count"] = valid_count;
            out["replicate_values"] = replicate_values;
            return out;
        },
        py::arg("model"), py::arg("mu"), py::arg("sigma"), py::arg("sample_size"), py::arg("probabilities"),
        py::arg("dataset"), py::arg("replicates"), py::arg("seed"), py::arg("max_retries"), py::arg("run"),
        py::arg("ci_method"), py::arg("alpha"));
}
