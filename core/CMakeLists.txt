# Dev-only build for the corehydro C++ core test suite.
# The shipped R/Python packages compile the vendored core via Makevars / scikit-build;
# this file exists so contributors (and CI) can build and run the C++ tests directly.
cmake_minimum_required(VERSION 3.15)
project(corehydro_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_library(corehydro_core INTERFACE)
target_include_directories(corehydro_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(corehydro_core INTERFACE cxx_std_17)
# Namespaced alias so external consumers / a future repo split can add_subdirectory(core)
# (or FetchContent) and link `corehydro::core` -- the header-only, dependency-free core.
add_library(corehydro::core ALIAS corehydro_core)

set(BF_TESTS
  test_mersenne_twister
  test_generalized_extreme_value
  test_gev_estimation
  test_goodness_of_fit
  test_mcmc_extra
  test_dist_runner
  test_toolbox_runner
  test_autocorrelation
  test_mixture
  test_mixture_em_seeding
  test_empirical_distribution
  test_univariate_distribution_factory
  test_kernel_density
  test_multiple_grubbs_beck
  test_matrix_regularization
  test_stratification
  test_numerical_diff
  test_model_parameter
  test_model_spec_parameters
  test_model_base
  test_quantile_prior
  test_univariate_distribution_model
  test_mixture_model
  test_competing_risks_model
  test_point_process_model
  test_fit_runner
  test_maximum_likelihood
  test_maximum_a_posteriori
  test_bayesian_analysis
  test_data_types
  test_data_frame
  test_data_frame_bootstrap
  test_plotting_positions_df
  test_nonparametric_empirical
  test_threshold_diagnostics
  test_trend_functions
  test_general_linear_function
  test_link_functions
  test_corehydro_link_functions
  test_penalties
  test_moment_machinery
  test_optimizers_local
  test_optimizers_global
  test_gmm
  test_singular_value_decomposition
  test_linear_regression
  test_bulletin17c
  test_bulletin17c_clone_with_data_frame
  test_p1_iunivariate_model
  test_box_cox
  test_time_series
  test_spatial_correlation
  test_cached_mvn_gaussian_copula
  test_spatial_regression_errors
  test_spatial_gev
  test_time_series_arma
  test_arima
  test_arimax
  test_rating_curve
  test_bivariate_distribution
  test_probability_ordinates
  test_uncertainty_analysis_results
  test_analysis_base
  test_univariate_analysis
  test_rating_curve_analysis
  test_bivariate_analysis
  test_spatial_gev_analysis
  test_fitting_analysis
  test_bulletin17c_analysis
  test_univariate_family_analyses
  test_time_series_analyses
  test_leverage_diagnostics
  test_influence_diagnostics
  test_predictive_checks
  test_parameter_names
  test_composite_analysis
  test_coincident_frequency_analysis
  test_bootstrap_analysis
  test_copula_clone
  test_multivariate_normal_api
  test_brent_search
  test_optimizer_runner
  test_callback_runner
)

# Sobol test takes the direction-number file as its sole argument.
add_executable(test_sobol tests/test_sobol.cpp)
target_link_libraries(test_sobol PRIVATE corehydro_core)
target_include_directories(test_sobol PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(test_sobol PRIVATE
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wextra>)
add_test(NAME test_sobol
         COMMAND test_sobol "${CMAKE_CURRENT_SOURCE_DIR}/data/new-joe-kuo-6.21201")

foreach(t ${BF_TESTS})
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE corehydro_core)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_compile_options(${t} PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wextra>
    # MSVC caps an object file at 65,536 sections; the fixture-driven and dispatch-heavy
    # translation units cross it as the fixture surface grows. /bigobj lifts the cap and is
    # MSVC-only (clang/gcc have no such limit and reject the flag).
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# Generic fixture-driven runner: reads the canonical fixtures/ shared by all languages.
add_executable(test_fixtures tests/test_fixtures.cpp)
target_link_libraries(test_fixtures PRIVATE corehydro_core)
target_include_directories(test_fixtures PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
# This translation unit dispatches every fixture kind for every family, and crossed MSVC's
# 65,536-section object-file cap (error C1128) as the toolbox surface landed. MSVC-only.
target_compile_options(test_fixtures PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
add_test(NAME test_fixtures
         COMMAND test_fixtures ${CMAKE_CURRENT_SOURCE_DIR}/../fixtures)
