// cpp11 glue for the shared distribution-spec runner: the three entry points behind every
// composite, copula and multivariate verb in R/distribution.R, R/copula.R and R/mvdist.R. Each
// takes a spec in the dist_spec.hpp grammar (assembled R-side by to_spec_json()) plus a method
// name and a JSON args array, and returns the flat DistResult as a named list.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <string>
#include <vector>

#include "corehydro/numerics/distributions/support/dist_runner.hpp"

using namespace cpp11;
namespace supp = corehydro::numerics::distributions::support;

static list pack(const supp::DistResult& r) {
    writable::doubles values(static_cast<R_xlen_t>(r.values.size()));
    for (std::size_t i = 0; i < r.values.size(); ++i)
        values[static_cast<R_xlen_t>(i)] = r.values[i];
    writable::strings names(static_cast<R_xlen_t>(r.names.size()));
    for (std::size_t i = 0; i < r.names.size(); ++i)
        names[static_cast<R_xlen_t>(i)] = r.names[i];
    return writable::list({"values"_nm = values, "names"_nm = names,
                           "spec"_nm = writable::strings({r.spec})});
}

[[cpp11::register]]
list ch_dist_spec_run_(std::string spec_json, std::string method, std::string args_json) {
    return pack(supp::run_dist(spec_json, method, args_json));
}

[[cpp11::register]]
list ch_copula_run_(std::string spec_json, std::string method, std::string args_json) {
    return pack(supp::run_copula(spec_json, method, args_json));
}

[[cpp11::register]]
list ch_mvdist_run_(std::string spec_json, std::string method, std::string args_json) {
    return pack(supp::run_mvdist(spec_json, method, args_json));
}
