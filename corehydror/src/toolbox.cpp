// cpp11 glue for the shared toolbox runner: the entry point behind every verb in R/toolbox.R,
// R/gof.R and R/optim.R. Takes a group, a method, a list of numeric data vectors, and a JSON
// options object (assembled R-side by to_spec_json()), and returns the flat ToolboxResult as a
// named list. Mirrors corehydropy/src/bindings/toolbox.cpp exactly.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/support/toolbox_runner.hpp"
#include "corehydro/numerics/support/optimizer_runner.hpp"

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

// Runs one of the six ported optimizers (R/optim.R) against an R objective function. `objective`
// is called with a single numeric vector argument and must return a single number -- `as_doubles`
// below accepts either a double or an integer return (an all-integer objective like
// `function(p) sum(p > 0)` is common); a non-numeric or wrong-length return raises inside that
// conversion, which cpp11 turns into an R error that travels back out through
// optimizer_runner.hpp's guard and is rethrown (see that header's GuardedObjective for why an R
// error crossing this boundary cannot be allowed to just propagate through
// Optimizer::minimize()'s own catch-all unguarded).
[[cpp11::register]]
list ch_optim_run_(std::string spec_json, function objective) {
    tb::OptimResult r = tb::run_optimizer(spec_json, [&](const std::vector<double>& p) -> double {
        writable::doubles par(static_cast<R_xlen_t>(p.size()));
        for (std::size_t i = 0; i < p.size(); ++i) par[static_cast<R_xlen_t>(i)] = p[i];
        sexp out = objective(par);
        doubles v = as_doubles(out);
        if (v.size() != 1)
            throw std::runtime_error(
                "the objective must return a single number; got a value of length " +
                std::to_string(static_cast<long long>(v.size())));
        return v[0];
    });
    writable::doubles params(static_cast<R_xlen_t>(r.parameters.size()));
    for (std::size_t i = 0; i < r.parameters.size(); ++i)
        params[static_cast<R_xlen_t>(i)] = r.parameters[i];
    writable::doubles hess(static_cast<R_xlen_t>(r.hessian.size()));
    for (std::size_t i = 0; i < r.hessian.size(); ++i)
        hess[static_cast<R_xlen_t>(i)] = r.hessian[i];
    writable::integers hdims(static_cast<R_xlen_t>(r.hessian_dims.size()));
    for (std::size_t i = 0; i < r.hessian_dims.size(); ++i)
        hdims[static_cast<R_xlen_t>(i)] = r.hessian_dims[i];
    return writable::list({"parameters"_nm = params,
                           "value"_nm = writable::doubles({r.value}),
                           "iterations"_nm = writable::integers({r.iterations}),
                           "function_evaluations"_nm = writable::integers({r.function_evaluations}),
                           "status"_nm = writable::strings({r.status}),
                           "hessian"_nm = hess,
                           "hessian_dims"_nm = hdims});
}
