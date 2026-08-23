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

// Converts one Python function into the runner's objective callback. `objective` is called with a
// single 1D array argument and must return a single number -- a non-scalar return raises the same
// "single number" std::runtime_error the R glue raises, inside the py::cast<double> below.
static tb::Objective as_objective(py::function objective) {
    return [objective](const std::vector<double>& p) -> double {
        py::object out = objective(p);
        try {
            return out.cast<double>();
        } catch (const py::cast_error&) {
            throw std::runtime_error("the objective must return a single number; got " +
                                     std::string(py::str(out)));
        }
    };
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
    // The three Lagrange multiplier vectors, empty for every method except "augmented_lagrange";
    // corehydropy.optim folds them into the single `multipliers` dict the user sees.
    out["lambda"] = r.lambda;
    out["mu"] = r.mu;
    out["nu"] = r.nu;
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

    // Runs one of the fourteen ported optimizers (corehydropy.optim) against a Python objective
    // function, converted by as_objective above. A non-scalar return raises the same "single
    // number" std::runtime_error the R glue raises. That raise (a plain C++ exception, not a Python
    // one) is still funneled through optimizer_runner.hpp's GuardedObjective exactly like a genuine
    // Python-side exception (e.g. from a bug in the caller's own objective, which surfaces here as
    // py::error_already_set) -- see that header's GuardedObjective for why an exception crossing
    // this boundary cannot be allowed to just propagate through Optimizer::minimize()'s own
    // catch-all unguarded.
    m.def(
        "optim_run",
        [](const std::string& spec_json, py::function objective) {
            tb::OptimCallbacks cbs;
            cbs.objective = as_objective(objective);
            return pack_optim(tb::run_optimizer(spec_json, cbs));
        },
        py::arg("spec_json"), py::arg("objective"));

    // The same, plus a second Python callback: the analytic gradient the "adam" and
    // "gradient_descent" methods take. A separate entry point mirroring the R glue's
    // ch_optim_run_grad_ -- corehydropy.optim calls this one only when the caller supplied a
    // gradient, and optim_run otherwise (an absent gradient is the ported classes' null Gradient,
    // which falls back to NumericalDerivative.Gradient exactly as C# does). Both callbacks are
    // guarded through ONE shared abort state inside the runner, so a Python exception in either
    // stops the other from being re-entered. The runner rejects a wrong-LENGTH gradient by name, so
    // this side only has to reject a non-numeric return, which the cast below does.
    m.def(
        "optim_run_grad",
        [](const std::string& spec_json, py::function objective, py::function gradient) {
            tb::OptimCallbacks cbs;
            cbs.objective = as_objective(objective);
            cbs.gradient = [gradient](const std::vector<double>& p) -> std::vector<double> {
                py::object out = gradient(p);
                try {
                    return out.cast<std::vector<double>>();
                } catch (const py::cast_error&) {
                    throw std::runtime_error(
                        "the gradient must return one number per parameter; got " +
                        std::string(py::str(out)));
                }
            };
            return pack_optim(tb::run_optimizer(spec_json, cbs));
        },
        py::arg("spec_json"), py::arg("objective"), py::arg("gradient"));

    // The same, plus one Python callback per constraint: the "augmented_lagrange" method's
    // constraint functions. A constraint has the same scalar shape as an objective, so it goes
    // through the same as_objective conversion. The i-th element pairs POSITIONALLY with the i-th
    // entry of the spec's `constraints` array (its type/value/tolerance half) -- corehydropy.optim
    // splits each Constraint into those two halves in one pass, and the runner rejects a length
    // mismatch by name. All the callbacks are guarded through ONE shared abort state inside the
    // runner, so a Python exception in any constraint stops the objective and the other
    // constraints from being re-entered.
    m.def(
        "optim_run_constrained",
        [](const std::string& spec_json, py::function objective,
           const std::vector<py::function>& constraints) {
            tb::OptimCallbacks cbs;
            cbs.objective = as_objective(objective);
            cbs.constraints.reserve(constraints.size());
            for (const py::function& c : constraints) cbs.constraints.push_back(as_objective(c));
            return pack_optim(tb::run_optimizer(spec_json, cbs));
        },
        py::arg("spec_json"), py::arg("objective"), py::arg("constraints"));
}
