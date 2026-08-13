// cpp11 glue for the shared toolbox runner: the entry point behind every verb in R/toolbox.R,
// R/gof.R and R/optim.R. Takes a group, a method, a list of numeric data vectors, and a JSON
// options object (assembled R-side by to_spec_json()), and returns the flat ToolboxResult as a
// named list. Mirrors corehydropy/src/bindings/toolbox.cpp exactly.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <string>
#include <vector>

#include "corehydro/numerics/support/toolbox_runner.hpp"

using namespace cpp11;
namespace tb = corehydro::numerics::support;

static list pack(const tb::ToolboxResult& r) {
    writable::doubles values(static_cast<R_xlen_t>(r.values.size()));
    for (std::size_t i = 0; i < r.values.size(); ++i)
        values[static_cast<R_xlen_t>(i)] = r.values[i];
    writable::strings names(static_cast<R_xlen_t>(r.names.size()));
    for (std::size_t i = 0; i < r.names.size(); ++i)
        names[static_cast<R_xlen_t>(i)] = r.names[i];
    writable::integers dims(static_cast<R_xlen_t>(r.dims.size()));
    for (std::size_t i = 0; i < r.dims.size(); ++i)
        dims[static_cast<R_xlen_t>(i)] = r.dims[i];
    return writable::list({"values"_nm = values, "names"_nm = names, "dims"_nm = dims,
                           "spec"_nm = writable::strings({r.spec})});
}

[[cpp11::register]]
list ch_toolbox_run_(std::string group, std::string method, list data, std::string options_json) {
    std::vector<std::vector<double>> vecs;
    vecs.reserve(static_cast<std::size_t>(data.size()));
    for (R_xlen_t i = 0; i < data.size(); ++i) {
        doubles col(data[i]);
        vecs.emplace_back(col.begin(), col.end());
    }
    return pack(tb::run_toolbox(group, method, vecs, options_json));
}
