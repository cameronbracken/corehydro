// pybind11 glue for the shared distribution-spec runner: the three entry points behind every
// composite, copula and multivariate verb in corehydropy.distributions, corehydropy.copula and
// corehydropy.mvdist. Each takes a spec in the dist_spec.hpp grammar (assembled Python-side by
// json.dumps()) plus a method name and a JSON args array, and returns the flat DistResult as a
// dict. Mirrors corehydror's src/dist_spec.cpp exactly, one entry point per language.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "corehydro/numerics/distributions/support/dist_runner.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace supp = corehydro::numerics::distributions::support;

static py::dict pack(const supp::DistResult& r) {
    py::dict out;
    out["values"] = r.values;
    out["names"] = r.names;
    out["spec"] = r.spec;
    return out;
}

void register_dist_spec(py::module_& m) {
    m.def(
        "dist_spec_run",
        [](const std::string& spec_json, const std::string& method, const std::string& args_json) {
            return pack(supp::run_dist(spec_json, method, args_json));
        },
        py::arg("spec_json"), py::arg("method"), py::arg("args_json"));

    m.def(
        "copula_run",
        [](const std::string& spec_json, const std::string& method, const std::string& args_json) {
            return pack(supp::run_copula(spec_json, method, args_json));
        },
        py::arg("spec_json"), py::arg("method"), py::arg("args_json"));

    m.def(
        "mvdist_run",
        [](const std::string& spec_json, const std::string& method, const std::string& args_json) {
            return pack(supp::run_mvdist(spec_json, method, args_json));
        },
        py::arg("spec_json"), py::arg("method"), py::arg("args_json"));
}
