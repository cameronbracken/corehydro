// pybind11 binding exposing the GEV slice of the shared C++ core to Python.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "corehydro/numerics/distributions/generalized_extreme_value.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace dist = corehydro::numerics::distributions;

static dist::EstimationMethod parse_method(const std::string& m) {
    if (m == "mom" || m == "moments") return dist::EstimationMethod::MethodOfMoments;
    if (m == "lmom" || m == "lmoments") return dist::EstimationMethod::MethodOfLinearMoments;
    if (m == "mle") return dist::EstimationMethod::MaximumLikelihood;
    throw py::value_error("unknown estimation method '" + m + "' (use 'mom', 'lmom', or 'mle')");
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core bindings for corehydropy (GEV slice).";

    py::class_<dist::GeneralizedExtremeValue>(m, "GeneralizedExtremeValue")
        .def(py::init<double, double, double>(), py::arg("location"), py::arg("scale"),
             py::arg("shape"))
        .def_property_readonly("location", &dist::GeneralizedExtremeValue::xi)
        .def_property_readonly("scale", &dist::GeneralizedExtremeValue::alpha)
        .def_property_readonly("shape", &dist::GeneralizedExtremeValue::kappa)
        .def_property_readonly("parameters_valid",
                               &dist::GeneralizedExtremeValue::parameters_valid)
        .def("pdf", &dist::GeneralizedExtremeValue::pdf, py::arg("x"))
        .def("cdf", &dist::GeneralizedExtremeValue::cdf, py::arg("x"))
        .def("quantile", &dist::GeneralizedExtremeValue::inverse_cdf, py::arg("p"))
        .def("mean", &dist::GeneralizedExtremeValue::mean)
        .def("median", &dist::GeneralizedExtremeValue::median)
        .def("mode", &dist::GeneralizedExtremeValue::mode)
        .def("standard_deviation", &dist::GeneralizedExtremeValue::standard_deviation)
        .def("skewness", &dist::GeneralizedExtremeValue::skewness)
        .def("kurtosis", &dist::GeneralizedExtremeValue::kurtosis)
        .def("minimum", &dist::GeneralizedExtremeValue::minimum)
        .def("maximum", &dist::GeneralizedExtremeValue::maximum)
        .def("log_likelihood", &dist::GeneralizedExtremeValue::log_likelihood, py::arg("sample"))
        .def("linear_moments_from_parameters",
             &dist::GeneralizedExtremeValue::linear_moments_from_parameters, py::arg("parameters"))
        .def("quantile_gradient", &dist::GeneralizedExtremeValue::quantile_gradient, py::arg("p"))
        .def("parameter_covariance", &dist::GeneralizedExtremeValue::parameter_covariance,
             py::arg("sample_size"))
        .def("quantile_variance", &dist::GeneralizedExtremeValue::quantile_variance, py::arg("p"),
             py::arg("sample_size"));

    // Fit returns {location, scale, shape}.
    m.def(
        "gev_fit",
        [](const std::vector<double>& data, const std::string& method) {
            dist::GeneralizedExtremeValue g;
            g.estimate(data, parse_method(method));
            return std::vector<double>{g.xi(), g.alpha(), g.kappa()};
        },
        py::arg("data"), py::arg("method"));

    // Polymorphic distributions (Normal, Uniform, Exponential, ...).
    register_distributions(m);

    // Multivariate distributions (Dirichlet, Multinomial, BivariateEmpirical).
    register_multivariate(m);

    // Bivariate copulas (Clayton, ...).
    register_copulas(m);

    // Quasi-random sampling (Sobol sequence).
    register_sobol(m);

    // MCMC sampling (model registry + RWMH).
    register_mcmc(m);

    // Bootstrap sampling (model registry + regular/studentized runs).
    register_bootstrap(m);

    // Phase-4 estimation (MaximumLikelihood/MaximumAPosteriori; BayesianAnalysis deferred).
    register_estimation(m);

    // Phase-8 user-facing analyses (UnivariateAnalysis/FittingAnalysis/Bulletin17CAnalysis).
    register_analysis(m);

    // Data-statistics utilities (MGBT, Box-Cox, Yeo-Johnson, plotting positions, LHS).
    register_stats(m);

    // The data layer (DataFrame summaries, model log-likelihood, threshold diagnostics).
    register_data(m);

    // The shared distribution-spec runner (composite distributions, copulas, multivariate).
    register_dist_spec(m);

    // The shared toolbox runner (correlation and the rest of the Numerics utility layer).
    register_toolbox(m);
}
