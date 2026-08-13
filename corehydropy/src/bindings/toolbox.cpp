// pybind11 glue for the shared toolbox runner: the entry point behind every verb in
// corehydropy.toolbox, corehydropy.gof and corehydropy.optim. Mirrors corehydror's
// src/toolbox.cpp exactly, one entry point per language.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "corehydro/numerics/support/toolbox_runner.hpp"
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

void register_toolbox(py::module_& m) {
    m.def(
        "toolbox_run",
        [](const std::string& group, const std::string& method,
           const std::vector<std::vector<double>>& data, const std::string& options_json) {
            return pack(tb::run_toolbox(group, method, data, options_json));
        },
        py::arg("group"), py::arg("method"), py::arg("data"), py::arg("options_json"));
}
