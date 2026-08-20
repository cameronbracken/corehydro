cmake_minimum_required(VERSION 3.15)
project(corehydropy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core src/bindings/gev.cpp src/bindings/dist.cpp src/bindings/mvd.cpp src/bindings/copula.cpp src/bindings/mcmc.cpp src/bindings/bootstrap.cpp src/bindings/estimation.cpp src/bindings/analysis.cpp src/bindings/stats.cpp src/bindings/data.cpp src/bindings/dist_spec.cpp src/bindings/toolbox.cpp src/bindings/callback.cpp)
target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/corehydro_core/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings)

# Install the compiled module into the corehydropy package.
install(TARGETS _core DESTINATION corehydropy)
