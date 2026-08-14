// cpp11 glue for the callback surface (R/callback.R). Unlike the toolbox verbs, which pass
// serializable data through ch_toolbox_run_, every verb here takes a live R function, so it goes
// through callback_runner.hpp and its guard. Mirrors corehydropy's src/bindings/callback.cpp
// call for call.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/support/callback_runner.hpp"

using namespace cpp11;
namespace sup = corehydro::numerics::support;

namespace {

// Converts an R closure into the scalar signature, raising the same "single number" error the
// optimizer glue raises. The throw travels back through GuardedCall, which latches it and lets
// run_callback rethrow it inside this same cpp11-protected frame.
std::function<double(double)> as_scalar_fn(function f) {
    return [f](double x) mutable -> double {
        sexp out = f(writable::doubles({x}));
        doubles v = as_doubles(out);
        if (v.size() != 1)
            throw std::runtime_error(
                "the function must return a single number; got a value of length " +
                std::to_string(static_cast<long long>(v.size())));
        return v[0];
    };
}

std::function<double(const std::vector<double>&)> as_vector_scalar_fn(function f) {
    return [f](const std::vector<double>& p) mutable -> double {
        writable::doubles par(static_cast<R_xlen_t>(p.size()));
        for (std::size_t i = 0; i < p.size(); ++i) par[static_cast<R_xlen_t>(i)] = p[i];
        sexp out = f(par);
        doubles v = as_doubles(out);
        if (v.size() != 1)
            throw std::runtime_error(
                "the function must return a single number; got a value of length " +
                std::to_string(static_cast<long long>(v.size())));
        return v[0];
    };
}

list pack(const sup::CallbackResult& r) {
    writable::doubles values(static_cast<R_xlen_t>(r.values.size()));
    for (std::size_t i = 0; i < r.values.size(); ++i)
        values[static_cast<R_xlen_t>(i)] = r.values[i];
    writable::strings names(static_cast<R_xlen_t>(r.names.size()));
    for (std::size_t i = 0; i < r.names.size(); ++i) names[static_cast<R_xlen_t>(i)] = r.names[i];
    writable::integers dims(static_cast<R_xlen_t>(r.dims.size()));
    for (std::size_t i = 0; i < r.dims.size(); ++i) dims[static_cast<R_xlen_t>(i)] = r.dims[i];
    return writable::list({"values"_nm = values, "names"_nm = names, "dims"_nm = dims,
                           "status"_nm = writable::strings({r.status})});
}

}  // namespace

// Runs one method of the callback runner's "math" group against an R function. The scalar methods
// (root_find, derivative, quadrature) call `f` with one number; the vector methods (gradient,
// hessian) call it with the whole point. Either way `f` must return a single number.
[[cpp11::register]]
list ch_callback_math_(std::string method, std::string options_json, function f) {
    sup::CallbackSet cbs;
    if (method == "root_find" || method == "derivative" || method == "quadrature")
        cbs.scalar = as_scalar_fn(f);
    else
        cbs.vector_scalar = as_vector_scalar_fn(f);
    return pack(sup::run_callback("math", method, options_json, cbs));
}
