// Shared declarations so the pybind11 module (defined in gev.cpp) can pull in
// bindings defined in dist.cpp, mvd.cpp, copula.cpp, sobol.cpp, mcmc.cpp, bootstrap.cpp, and
// estimation.cpp.
#pragma once
#include <pybind11/pybind11.h>

void register_distributions(pybind11::module_& m);
void register_multivariate(pybind11::module_& m);
void register_copulas(pybind11::module_& m);
void register_sobol(pybind11::module_& m);
void register_mcmc(pybind11::module_& m);
void register_bootstrap(pybind11::module_& m);
void register_estimation(pybind11::module_& m);
void register_analysis(pybind11::module_& m);
void register_stats(pybind11::module_& m);
void register_data(pybind11::module_& m);
