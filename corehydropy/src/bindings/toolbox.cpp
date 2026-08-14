// pybind11 glue for the shared toolbox runner: the entry point behind every verb in
// corehydropy.toolbox, corehydropy.gof and corehydropy.optim. Mirrors corehydror's
// src/toolbox.cpp exactly, one entry point per language.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/support/toolbox_runner.hpp"
#include "corehydro/numerics/support/optimizer_runner.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace tb = corehydro::numerics::support;

static py::dict pack(const tb::ToolboxResult& r) {
    py::dict out;
    out["values"] = r.values;
    out["names"] = r.names;
    out["dims"] = r.dims;
    out["spec"] = r.spec;
    return out;
}

static py::dict pack_optim(const tb::OptimResult& r) {
    py::dict out;
    out["parameters"] = r.parameters;
    out["value"] = r.value;
    out["iterations"] = r.iterations;
    out["function_evaluations"] = r.function_evaluations;
    out["status"] = r.status;
    out["hessian"] = r.hessian;
    out["hessian_dims"] = r.hessian_dims;
    return out;
}

void register_toolbox(py::module_& m) {
    m.def(
        "toolbox_run",
        [](const std::string& group, const std::string& method,
           const std::vector<std::vector<double>>& data, const std::string& options_json) {
            return pack(tb::run_toolbox(group, method, data, options_json));
        },
        py::arg("group"), py::arg("method"), py::arg("data"), py::arg("options_json"));

    // Runs one of the six ported optimizers (corehydropy.optim) against a Python objective
    // function. `objective` is called with a single 1D array argument and must return a single
    // number -- a non-scalar return raises the same "single number" std::runtime_error the R glue
    // raises, inside the py::cast<double> conversion below. That raise (a plain C++ exception, not
    // a Python one) is still funneled through optimizer_runner.hpp's GuardedObjective exactly like
    // a genuine Python-side exception (e.g. from a bug in the caller's own objective, which
    // surfaces here as py::error_already_set) -- see that header's GuardedObjective for why an
    // exception crossing this boundary cannot be allowed to just propagate through
    // Optimizer::minimize()'s own catch-all unguarded.
    m.def(
        "optim_run",
        [](const std::string& spec_json, py::function objective) {
            tb::OptimResult r =
                tb::run_optimizer(spec_json, [&](const std::vector<double>& p) -> double {
                    py::object out = objective(p);
                    try {
                        return out.cast<double>();
                    } catch (const py::cast_error&) {
                        throw std::runtime_error(
                            "the objective must return a single number; got " +
                            std::string(py::str(out)));
                    }
                });
            return pack_optim(r);
        },
        py::arg("spec_json"), py::arg("objective"));
}
