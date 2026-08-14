// pybind11 glue for the callback surface (corehydropy.callback). Unlike the toolbox verbs, which
// pass serializable data through toolbox_run, every verb here takes a live Python function, so it
// goes through callback_runner.hpp and its guard. Mirrors corehydror's src/callback.cpp call for
// call.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "bindings.hpp"
#include "corehydro/numerics/support/callback_runner.hpp"

namespace py = pybind11;
namespace sup = corehydro::numerics::support;

namespace {

// Converts a Python callable into the scalar signature, raising the same "single number" error the
// R glue raises. That raise (a plain C++ exception, not a Python one) is funneled through
// GuardedCall exactly like a genuine Python-side exception (py::error_already_set), which latches
// it and lets run_callback rethrow it once the ported routine returns.
std::function<double(double)> as_scalar_fn(py::function f) {
    return [f](double x) -> double {
        py::object out = f(x);
        try {
            return out.cast<double>();
        } catch (const py::cast_error&) {
            throw std::runtime_error("the function must return a single number; got " +
                                     std::string(py::str(out)));
        }
    };
}

std::function<double(const std::vector<double>&)> as_vector_scalar_fn(py::function f) {
    return [f](const std::vector<double>& p) -> double {
        py::object out = f(p);
        try {
            return out.cast<double>();
        } catch (const py::cast_error&) {
            throw std::runtime_error("the function must return a single number; got " +
                                     std::string(py::str(out)));
        }
    };
}

py::dict pack(const sup::CallbackResult& r) {
    py::dict out;
    out["values"] = r.values;
    out["names"] = r.names;
    out["dims"] = r.dims;
    out["status"] = r.status;
    return out;
}

}  // namespace

void register_callback(py::module_& m) {
    // Runs one method of the callback runner's "math" group against a Python function. The scalar
    // methods (root_find, derivative) call `f` with one number; the vector methods (gradient,
    // hessian) call it with the whole point as a list. Either way `f` must return a single number.
    m.def(
        "callback_math",
        [](const std::string& method, const std::string& options_json, py::function f) {
            sup::CallbackSet cbs;
            if (method == "root_find" || method == "derivative")
                cbs.scalar = as_scalar_fn(f);
            else
                cbs.vector_scalar = as_vector_scalar_fn(f);
            return pack(sup::run_callback("math", method, options_json, cbs));
        },
        py::arg("method"), py::arg("options_json"), py::arg("f"));
}
