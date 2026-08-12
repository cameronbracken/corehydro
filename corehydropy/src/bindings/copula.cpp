// pybind11 glue for the one bivariate-copula fixture case the shared distribution runner
// cannot serve. Every user-facing copula verb -- and every fitted copula -- now goes through
// _core.copula_run in dist_spec.cpp, which drives the same dist_runner.hpp entry point the C++
// fixture runner, the R glue and the dotnet oracle emitter drive. What is left here is the
// non-finite theta/df validity cases: the runner's JSON grammar has no NaN/Infinity literal, so
// those constructs are built directly through the factory instead. Every copula shares
// BivariateCopula's uniform theta/get_copula_parameters/pdf/cdf/... surface (see
// copula_factory.hpp's header comment), so there is no per-type branching here beyond the
// factory itself.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_factory.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace dist = corehydro::numerics::distributions;
namespace cop = corehydro::numerics::distributions::copulas;

static std::unique_ptr<cop::BivariateCopula> make_copula(const std::string& type,
                                                           const std::vector<double>& params) {
    auto c = cop::create_copula(type);
    c->set_copula_parameters(params);
    return c;
}

void register_copulas(py::module_& m) {
    // method + flat numeric args in, double out. Methods: pdf/log_pdf/cdf (args=[u,v]),
    // inverse_cdf (args=[u,v,index]), upper_tail_dependence, lower_tail_dependence, theta,
    // df (2-parameter copulas only; get_copula_parameters()[1]), or_exceedance/
    // and_exceedance (args=[u,v]), parameters_valid, random_value (args=[sample_size, seed,
    // row, col]; stateless -- GenerateRandomValues seeds its own LatinHypercube draw from
    // `seed`). marg_x_target/marg_y_target optionally attach marginals directly (the C#
    // `Copula(theta, marginX, marginY)` ctor path): "" means no marginal for that side.
    //
    // Only `parameters_valid` is reached today (the non-finite theta/df cases described in the
    // file header); the other arms are kept so this stays a faithful fallback for any construct
    // the shared grammar cannot encode, exactly as the fixture runner's own bespoke dispatchers
    // do.
    m.def(
        "cop_val",
        [](const std::string& type, const std::vector<double>& params, const std::string& method,
           const std::vector<double>& args, const std::string& marg_x_target,
           const std::vector<double>& marg_x_params, const std::string& marg_y_target,
           const std::vector<double>& marg_y_params) {
            auto c = make_copula(type, params);
            if (!marg_x_target.empty()) {
                auto mx = dist::create_distribution(marg_x_target);
                mx->set_parameters(marg_x_params);
                c->marginal_distribution_x = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(mx));
            }
            if (!marg_y_target.empty()) {
                auto my = dist::create_distribution(marg_y_target);
                my->set_parameters(marg_y_params);
                c->marginal_distribution_y = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(my));
            }
            if (method == "pdf") return c->pdf(args[0], args[1]);
            if (method == "log_pdf") return c->log_pdf(args[0], args[1]);
            if (method == "cdf") return c->cdf(args[0], args[1]);
            if (method == "inverse_cdf")
                return c->inverse_cdf(args[0], args[1])[static_cast<std::size_t>(args[2])];
            if (method == "upper_tail_dependence") return c->upper_tail_dependence();
            if (method == "lower_tail_dependence") return c->lower_tail_dependence();
            if (method == "theta") return c->theta();
            if (method == "df") return c->get_copula_parameters()[1];
            if (method == "or_exceedance") return c->or_joint_exceedance_probability(args[0], args[1]);
            if (method == "and_exceedance") return c->and_joint_exceedance_probability(args[0], args[1]);
            if (method == "parameters_valid") return c->parameters_valid() ? 1.0 : 0.0;
            if (method == "random_value") {
                auto sample = c->generate_random_values(static_cast<int>(args[0]), static_cast<int>(args[1]));
                return sample[static_cast<std::size_t>(args[2])][static_cast<std::size_t>(args[3])];
            }
            throw py::value_error("unknown copula fixture method: " + method);
        },
        py::arg("type"), py::arg("params"), py::arg("method"), py::arg("args"),
        py::arg("marg_x_target") = "", py::arg("marg_x_params") = std::vector<double>{},
        py::arg("marg_y_target") = "", py::arg("marg_y_params") = std::vector<double>{});
}
