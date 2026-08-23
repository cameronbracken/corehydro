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
  test_pivotal_bootstrap
  test_copula_clone
  test_multivariate_normal_api
  test_brent_search
  test_adaptive_gauss_kronrod
  test_optimizer_runner
  test_callback_runner
  test_rng_handle
  test_parameter_constraint_sizes
  test_linear_moments_overflow
  test_generalized_normal_jacobian
  test_root_finding_extras
  test_integration_extras
  test_integration_random
  test_runge_kutta
  test_interpolation_extras
  test_linalg_extras
  test_univariate_functions
  test_local_optimizers
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

# test_integration_random (Miser/Vegas) needs the Sobol direction-numbers file for dimensions > 1,
# but is registered through the plain BF_TESTS loop above (no argv passed, unlike test_sobol/
# test_fixtures below); a compile definition is simpler than special-casing this one target's argv.
target_compile_definitions(test_integration_random PRIVATE
  SOBOL_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/new-joe-kuo-6.21201")

# Generic fixture-driven runner: reads the canonical fixtures/ shared by all languages. It is two
# translation units, and the split is deliberate -- see the -ffp-contract=off comment below before
# merging them back together.
add_executable(test_fixtures tests/test_fixtures.cpp tests/fixture_callback_catalog.cpp)
target_link_libraries(test_fixtures PRIVATE corehydro_core)
target_include_directories(test_fixtures PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
# test_fixtures.cpp dispatches every fixture kind for every family, and crossed MSVC's
# 65,536-section object-file cap (error C1128) as the toolbox surface landed. MSVC-only.
target_compile_options(test_fixtures PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

# -ffp-contract=off is the callback surface's requirement, and it is about correctness of the
# COMPARISON rather than of the core. fixture_callback_catalog.cpp writes the fixture callback
# catalog as C++ lambdas -- the counterparts of the R closures, Python functions and C# delegates
# the other three runners write for the same names -- and every one of them is specified as
# arithmetic in double. Clang and GCC contract `y - a - b*t` and `sum + r*r` into fused
# multiply-adds by default, which computes a DIFFERENT (more accurate) function than the one the
# fixture names; R, Python and .NET never fuse one on their own. Turning contraction off makes the
# C++ catalog evaluate the same arithmetic the other three runners do, which is what the
# bit-equality assertions in fixtures/callback/callback_cross_language.json (mode "abs", tol 0) are
# entitled to assume.
#
# THE FLAG IS EXERCISED, and by exactly one case: that file's
# `linear_model_contractible_arithmetic_short_exact`, which was written for this purpose after the
# flag was found to guard nothing detectable. Remove the flag and test_fixtures fails there twice,
# measured on this machine (Apple clang 21, arm64):
#   .../bootstrap/named: expected 0.03571428571428292, got 0.035714285714283256  [statistic[0],
#     Fit_LinearTrend's intercept on the original sample -- about 50 ulp, because the fused
#     accumulations move the slope in its last bit and `ybar - slope * tbar` then amplifies that]
#   .../mcmc/named:      expected -0.11662553103451964, got -0.11662553103451963  [map_fitness,
#     Mcmc_LinearKernel's own value at the MAP state -- 1 ulp, the size of the effect on one kernel
#     evaluation; 72% of that kernel's values over the case's proposal box move at all]
# The file's other case cannot detect the flag and never could: it names Mcmc_UniformWidthKernel
# (comparisons only) and Resample_Iid (index draws), neither of which has a product to fuse.
#
# TWO LIMITS, both real. The flag is inert on x86-64 unless FMA is enabled there (-mfma, or
# -march=x86-64-v3 and up), because a baseline x86-64 compiler has no fma instruction to contract
# into -- so removing it changes nothing on a default Linux or Windows x86-64 build, and the
# failure above reproduces on arm64 and on any x86-64 build with FMA on. And MSVC needs no flag at
# all: /fp:precise, its default, does not contract, so the asymmetry below is correct.
#
# The catalog is a translation unit of its own precisely so this flag reaches ONLY it. corehydro_core
# is an INTERFACE library, so the header-only core is compiled INTO whichever TU includes it: a flag
# on the whole target would de-contract the ported core as well, and the C++ runner would then be
# exercising a differently-compiled core than the one the R and Python packages build and ship (they
# pass no such flag). Keep the catalog out of test_fixtures.cpp.
set_source_files_properties(tests/fixture_callback_catalog.cpp
  PROPERTIES COMPILE_OPTIONS
    "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-ffp-contract=off>")
add_test(NAME test_fixtures
         COMMAND test_fixtures ${CMAKE_CURRENT_SOURCE_DIR}/../fixtures)
