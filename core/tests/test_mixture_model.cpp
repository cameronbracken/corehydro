// Standalone test for corehydro::models::MixtureModel (Phase 5, M10).
//
// Oracle for behavior is the C# source itself:
//   - upstream/RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/MixtureModel.cs
//     @ fc28c0c, and its base UnivariateDistributionModelBase.cs @ fc28c0c;
//   - a full transcription of the upstream test class
//     RMC.BestFit.Tests/Univariate/MixtureModelTests.cs (every method except the XML /
//     serialization surface: Test_Constructor_XElement_RestoresModel, Test_ToXElement_* (3),
//     Test_RoundTrip_* (2) -- ToXElement / the XElement ctor are project-wide non-ports).
// Hardcoded oracles are allowed here (internal support layer); public-API oracle values stay
// in fixtures/ only (the fixture wiring for the Models layer arrives in M13/M14).
//
// Test-surface adaptations forced by the port (each noted at the test):
//   - C# `Assert.AreSame(df, model.DataFrame)` (reference identity) has no C++ analogue for
//     the move-only value-typed DataFrame; the transcription checks the frame's content.
//   - C# `model.DataFrame = null!` maps to the never-set state: a default-constructed model
//     has no frame (see the base header's nullability note).
//   - The degenerate-data test expects the C# ArgumentOutOfRangeException from Uniform.PDF;
//     the C++ Uniform throws std::out_of_range from the same validity guard.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/models/univariate_distribution/mixture_model.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/gumbel.hpp"
#include "corehydro/numerics/distributions/mixture.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/poisson.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "check.hpp"

using corehydro::models::DataFrame;
using corehydro::models::ExactData;
using corehydro::models::ExactSeries;
using corehydro::models::MixtureModel;
using corehydro::models::ModelParameter;
using corehydro::models::PriorComponentType;
using corehydro::models::ValidationResult;
using corehydro::numerics::distributions::Gumbel;
using corehydro::numerics::distributions::Mixture;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::Poisson;
using corehydro::numerics::distributions::UnivariateDistributionBase;
using corehydro::numerics::distributions::UnivariateDistributionType;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ===========================================================================================
// Test data helpers (MixtureModelTests.cs, "Test Data Helper" region).
// ===========================================================================================

// Creates a sample data frame with positive exact observations.
DataFrame create_sample_data_frame() {
    DataFrame df;
    std::vector<ExactData> data{
        ExactData(1990, 1200), ExactData(1991, 1500), ExactData(1992, 1800),
        ExactData(1993, 2200), ExactData(1994, 2500), ExactData(1995, 2800),
        ExactData(1996, 3200), ExactData(1997, 3600), ExactData(1998, 4000),
        ExactData(1999, 4500)};
    df.set_exact_series(ExactSeries(data));
    return df;
}

// Creates a bimodal data frame typical of mixed flood populations.
DataFrame create_bimodal_data_frame() {
    DataFrame df;
    std::vector<ExactData> data;
    // Low flow population (e.g., baseflow floods).
    for (int i = 0; i < 15; ++i) data.emplace_back(1980 + i, 500 + i * 50);
    // High flow population (e.g., extreme events).
    data.emplace_back(1995, 5000);
    data.emplace_back(1996, 6000);
    data.emplace_back(1997, 7500);
    data.emplace_back(1998, 9000);
    data.emplace_back(1999, 12000);
    df.set_exact_series(ExactSeries(data));
    return df;
}

// Creates a data frame with zero values for zero-inflation testing.
DataFrame create_zero_inflated_data_frame() {
    DataFrame df;
    std::vector<ExactData> data{
        ExactData(1990, 0),   ExactData(1991, 0),   ExactData(1992, 100),
        ExactData(1993, 0),   ExactData(1994, 250), ExactData(1995, 0),
        ExactData(1996, 500), ExactData(1997, 750), ExactData(1998, 0),
        ExactData(1999, 1000)};
    df.set_exact_series(ExactSeries(data));
    return df;
}

std::vector<double> parameter_values(const MixtureModel& model) {
    std::vector<double> p;
    p.reserve(model.parameters().size());
    for (const auto& mp : model.parameters()) p.push_back(mp.value());
    return p;
}

bool any_message_contains(const ValidationResult& result, const std::string& needle) {
    for (const std::string& m : result.validation_messages)
        if (m.find(needle) != std::string::npos) return true;
    return false;
}

// ===========================================================================================
// Constructor tests.
// ===========================================================================================

// Test_Constructor_EmptyConstructor_CreatesDefaultNormalMixture
void test_constructor_empty_creates_default_normal_mixture() {
    MixtureModel model;

    CHECK_TRUE(model.has_mixture());
    CHECK_EQ(model.mixture()->component_count(), 2);
    CHECK_TRUE(model.mixture()->component(0).type() == UnivariateDistributionType::Normal);
    CHECK_TRUE(model.mixture()->component(1).type() == UnivariateDistributionType::Normal);
}

// Test_Constructor_EmptyConstructor_HasEqualWeights
void test_constructor_empty_has_equal_weights() {
    MixtureModel model;

    CHECK_NEAR(model.mixture()->weights()[0], 0.5, 1e-10);
    CHECK_NEAR(model.mixture()->weights()[1], 0.5, 1e-10);
}

// Test_Constructor_WithDataAndMixture_SetsProperties (AreSame -> content check; see header).
void test_constructor_with_data_and_mixture_sets_properties() {
    DataFrame df = create_sample_data_frame();
    std::vector<std::unique_ptr<UnivariateDistributionBase>> comps;
    comps.push_back(std::make_unique<Normal>(1000, 200));
    comps.push_back(std::make_unique<Normal>(3000, 500));
    Mixture owned({0.3, 0.7}, std::move(comps));

    MixtureModel model(std::move(df), owned);

    CHECK_TRUE(model.has_data_frame());
    CHECK_EQ(model.data_frame().exact_series().count(), static_cast<std::size_t>(10));
    CHECK_TRUE(model.has_mixture());
    CHECK_EQ(model.mixture()->component_count(), 2);
}

// Test_Constructor_WithDistributionTypes_CreatesComponents
void test_constructor_with_distribution_types_creates_components() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::GammaDistribution};

    MixtureModel model(std::move(df), types);

    CHECK_EQ(model.mixture()->component_count(), 2);
    CHECK_TRUE(model.mixture()->component(0).type() == UnivariateDistributionType::Normal);
    CHECK_TRUE(model.mixture()->component(1).type() ==
               UnivariateDistributionType::GammaDistribution);
}

// Test_Constructor_WithDistributionTypes_ZeroInflated
void test_constructor_with_distribution_types_zero_inflated() {
    DataFrame df = create_zero_inflated_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};

    MixtureModel model(std::move(df), types, /*is_zero_inflated=*/true);

    CHECK_TRUE(model.is_zero_inflated());
    CHECK_TRUE(model.mixture()->zero_weight() > 0);
}

// Test_Constructor_WithDistributions_UsesProvidedInstances
void test_constructor_with_distributions_uses_provided_instances() {
    DataFrame df = create_sample_data_frame();
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distributions;
    distributions.push_back(std::make_unique<Normal>(1500, 300));
    distributions.push_back(std::make_unique<Gumbel>(2500, 400));

    MixtureModel model(std::move(df), std::move(distributions));

    CHECK_EQ(model.mixture()->component_count(), 2);
    // C# ((Normal)model.Mixture.Distributions[0]).Mu.
    CHECK_NEAR(model.mixture()->component(0).get_parameters()[0], 1500, 1e-10);
}

// Test_Constructor_WithEmptyDistributionTypes_ThrowsException (C# ArgumentException).
void test_constructor_with_empty_distribution_types_throws() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types;

    CHECK_THROWS(MixtureModel(std::move(df), types));
}

// Test_Constructor_WithEmptyDistributions_ThrowsException (C# ArgumentException).
void test_constructor_with_empty_distributions_throws() {
    DataFrame df = create_sample_data_frame();
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distributions;

    CHECK_THROWS(MixtureModel(std::move(df), std::move(distributions)));
}

// ===========================================================================================
// Property tests.
// ===========================================================================================

// Test_Mixture_SetAndGet
void test_mixture_set_and_get() {
    MixtureModel model;
    std::vector<std::unique_ptr<UnivariateDistributionBase>> comps;
    comps.push_back(std::make_unique<Gumbel>(100, 20));
    comps.push_back(std::make_unique<Normal>(200, 50));
    auto new_mixture = std::make_unique<Mixture>(std::vector<double>{0.4, 0.6}, std::move(comps));

    model.set_mixture(std::move(new_mixture));

    CHECK_TRUE(model.has_mixture());
    CHECK_NEAR(model.mixture()->weights()[0], 0.4, 1e-10);
}

// Test_IsZeroInflated_SetAndGet
void test_is_zero_inflated_set_and_get() {
    DataFrame df = create_zero_inflated_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    model.set_is_zero_inflated(true);

    CHECK_TRUE(model.is_zero_inflated());
    CHECK_TRUE(model.mixture()->is_zero_inflated());
}

// Test_IsZeroInflated_SetsZeroWeight
void test_is_zero_inflated_sets_zero_weight() {
    DataFrame df = create_zero_inflated_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    model.set_is_zero_inflated(true);

    // 5 out of 10 values are zero, so ZeroWeight should be ~0.5.
    CHECK_TRUE(model.mixture()->zero_weight() > 0.4 && model.mixture()->zero_weight() < 0.6);
}

// Test_IsZeroInflated_False_ZeroWeightIsZero
void test_is_zero_inflated_false_zero_weight_is_zero() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    model.set_is_zero_inflated(false);

    CHECK_EQ(model.mixture()->zero_weight(), 0.0);
}

// Test_UseSingleQuantile_AlwaysTrue
void test_use_single_quantile_always_true() {
    MixtureModel model;

    CHECK_TRUE(model.use_single_quantile());

    // Setting to false should be ignored.
    model.set_use_single_quantile(false);
    CHECK_TRUE(model.use_single_quantile());
}

// Test_DataFrame_SetTriggersParameterUpdate
void test_data_frame_set_triggers_parameter_update() {
    DataFrame df1 = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df1), types);
    int initial_param_count = model.number_of_parameters();

    DataFrame df2 = create_bimodal_data_frame();
    model.set_data_frame(std::move(df2));

    // Parameters should be recalculated.
    CHECK_EQ(model.number_of_parameters(), initial_param_count);
}

// ===========================================================================================
// SetDefaultMixture tests.
// ===========================================================================================

// Test_SetDefaultMixture_SingleComponent
void test_set_default_mixture_single_component() {
    MixtureModel model;

    model.set_default_mixture({UnivariateDistributionType::Gumbel});

    CHECK_EQ(model.mixture()->component_count(), 1);
    CHECK_TRUE(model.mixture()->component(0).type() == UnivariateDistributionType::Gumbel);
}

// Test_SetDefaultMixture_TwoComponents_EqualWeights
void test_set_default_mixture_two_components_equal_weights() {
    MixtureModel model;

    model.set_default_mixture(
        {UnivariateDistributionType::Normal, UnivariateDistributionType::Gumbel});

    CHECK_NEAR(model.mixture()->weights()[0], 0.5, 1e-10);
    CHECK_NEAR(model.mixture()->weights()[1], 0.5, 1e-10);
}

// Test_SetDefaultMixture_ThreeComponents_EqualWeights
void test_set_default_mixture_three_components_equal_weights() {
    MixtureModel model;

    model.set_default_mixture({UnivariateDistributionType::Normal,
                               UnivariateDistributionType::Gumbel,
                               UnivariateDistributionType::LogNormal});

    double expected_weight = 1.0 / 3.0;
    CHECK_NEAR(model.mixture()->weights()[0], expected_weight, 1e-10);
    CHECK_NEAR(model.mixture()->weights()[1], expected_weight, 1e-10);
    CHECK_NEAR(model.mixture()->weights()[2], expected_weight, 1e-10);
}

// Test_SetDefaultMixture_WeightsSumToOne
void test_set_default_mixture_weights_sum_to_one() {
    MixtureModel model;

    model.set_default_mixture({UnivariateDistributionType::Normal,
                               UnivariateDistributionType::Gumbel,
                               UnivariateDistributionType::GammaDistribution});

    double sum = 0.0;
    for (double w : model.mixture()->weights()) sum += w;
    CHECK_NEAR(sum, 1.0, 1e-10);
}

// Test_SetDefaultMixture_EmptyList_ThrowsException (C# ArgumentException).
void test_set_default_mixture_empty_list_throws() {
    MixtureModel model;
    CHECK_THROWS(model.set_default_mixture({}));
}

// ===========================================================================================
// SetDefaultParameters tests.
// ===========================================================================================

// Test_SetDefaultParameters_TwoComponents_HasWeightParameters
void test_set_default_parameters_two_components_has_weight_parameters() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    // Should have 2 weight parameters + 2*2 distribution parameters = 6.
    CHECK_TRUE(model.number_of_parameters() >= 6);

    // First two parameters should be weights.
    CHECK_TRUE(model.parameters()[0].name().find("Weight") != std::string::npos);
    CHECK_TRUE(model.parameters()[1].name().find("Weight") != std::string::npos);
}

// Test_SetDefaultParameters_SingleComponent_NoWeightParameters
void test_set_default_parameters_single_component_no_weight_parameters() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    // Single component has no weight parameters.
    bool any_weight = false;
    for (const auto& p : model.parameters())
        if (p.name().find("Weight") != std::string::npos) any_weight = true;
    CHECK_TRUE(!any_weight);
}

// Test_SetDefaultParameters_WeightBounds
void test_set_default_parameters_weight_bounds() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    int checked = 0;
    for (const auto& p : model.parameters()) {
        if (p.name().find("Weight") == std::string::npos) continue;
        CHECK_EQ(p.lower_bound(), 0.0);
        CHECK_EQ(p.upper_bound(), 1.0);
        ++checked;
    }
    CHECK_EQ(checked, 2);
}

// Test_SetDefaultParameters_WeightsHaveUniformPrior
void test_set_default_parameters_weights_have_uniform_prior() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Gumbel};
    MixtureModel model(std::move(df), types);

    int checked = 0;
    for (const auto& p : model.parameters()) {
        if (p.name().find("Weight") == std::string::npos) continue;
        CHECK_TRUE(p.prior_distribution().type() == UnivariateDistributionType::Uniform);
        ++checked;
    }
    CHECK_EQ(checked, 2);
}

// ===========================================================================================
// SetDefaultQuantilePriors tests.
// ===========================================================================================

// Test_SetDefaultQuantilePriors_SingleQuantile
void test_set_default_quantile_priors_single_quantile() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);
    model.set_enable_quantile_priors(true);

    model.set_default_quantile_priors();

    // Mixture model uses single quantile prior.
    CHECK_EQ(model.quantile_priors().size(), static_cast<std::size_t>(1));
}

// Test_SetDefaultQuantilePriors_Disabled_EmptyList
void test_set_default_quantile_priors_disabled_empty_list() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);
    model.set_enable_quantile_priors(false);

    model.set_default_quantile_priors();

    CHECK_EQ(model.quantile_priors().size(), static_cast<std::size_t>(0));
}

// Test_SetDefaultQuantilePriors_UsesLnNormalDistribution
void test_set_default_quantile_priors_uses_ln_normal_distribution() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);
    model.set_enable_quantile_priors(true);

    model.set_default_quantile_priors();

    CHECK_TRUE(model.quantile_priors()[0].distribution().type() ==
               UnivariateDistributionType::LnNormal);
}

// ===========================================================================================
// LogLikelihood tests.
// ===========================================================================================

// Test_LogLikelihood_ReturnsFiniteValue
void test_log_likelihood_returns_finite_value() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    double log_lh = model.log_likelihood(parameters);

    CHECK_TRUE(!std::isnan(log_lh));
    CHECK_TRUE(log_lh != kInf);
}

// Test_DataLogLikelihood_ReturnsFiniteValue
void test_data_log_likelihood_returns_finite_value() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Gumbel};
    MixtureModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    double data_log_lh = model.data_log_likelihood(parameters);

    CHECK_TRUE(!std::isnan(data_log_lh));
    CHECK_TRUE(data_log_lh != kInf);
}

// Test_PriorLogLikelihood_ReturnsFiniteValue
void test_prior_log_likelihood_returns_finite_value() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    double prior_log_lh = model.prior_log_likelihood(parameters);

    CHECK_TRUE(!std::isnan(prior_log_lh));
    CHECK_TRUE(prior_log_lh != kInf);
}

// Test_LogLikelihood_EqualsDataPlusPrior
void test_log_likelihood_equals_data_plus_prior() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    double full_log_lh = model.log_likelihood(parameters);
    double data_log_lh = model.data_log_likelihood(parameters);
    double prior_log_lh = model.prior_log_likelihood(parameters);

    // Full = Data + Prior (allowing for numerical precision).
    CHECK_NEAR(full_log_lh, data_log_lh + prior_log_lh, 1e-6);
}

// Test_PointwiseDataLogLikelihood_ReturnsCorrectCount
void test_pointwise_data_log_likelihood_returns_correct_count() {
    DataFrame df = create_sample_data_frame();
    std::size_t exact_count = df.exact_series().count();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(parameters);

    CHECK_EQ(pointwise.size(), exact_count);
}

// Test_PointwiseDataLogLikelihood_SumEqualsDataLogLikelihood
void test_pointwise_data_log_likelihood_sum_equals_data_log_likelihood() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(parameters);
    double data_log_lh = model.data_log_likelihood(parameters);

    double sum = 0.0;
    for (double ll : pointwise) sum += ll;
    CHECK_NEAR(sum, data_log_lh, 1e-6);
}

// ===========================================================================================
// Zero-inflation tests.
// ===========================================================================================

// Test_ZeroInflated_LogLikelihood_HandlesZeroValues
void test_zero_inflated_log_likelihood_handles_zero_values() {
    DataFrame df = create_zero_inflated_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types, /*is_zero_inflated=*/true);

    std::vector<double> parameters = parameter_values(model);
    double log_lh = model.log_likelihood(parameters);

    CHECK_TRUE(!std::isnan(log_lh));
    CHECK_TRUE(log_lh != -kInf);
}

// Test_ZeroInflated_ZeroWeightReflectsData
void test_zero_inflated_zero_weight_reflects_data() {
    DataFrame df = create_zero_inflated_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types, /*is_zero_inflated=*/true);

    // Should reflect proportion of zeros in data.
    CHECK_TRUE(model.mixture()->zero_weight() > 0);
}

// ===========================================================================================
// ExpectationMaximization tests.
// ===========================================================================================

// Test_ExpectationMaximization_ReturnsParameters
void test_expectation_maximization_returns_parameters() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    std::vector<double> parameters;
    corehydro::numerics::math::linalg::Matrix covariance(0, 0);
    int iterations = 0;
    model.expectation_maximization(parameters, covariance, iterations);

    CHECK_TRUE(parameters.size() > 0);
    CHECK_TRUE(iterations > 0);
}

// Test_ExpectationMaximization_ConvergesWithinMaxIterations
void test_expectation_maximization_converges_within_max_iterations() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    int max_iterations = 100;
    std::vector<double> parameters;
    corehydro::numerics::math::linalg::Matrix covariance(0, 0);
    int iterations = 0;
    model.expectation_maximization(parameters, covariance, iterations, max_iterations);

    CHECK_TRUE(iterations <= max_iterations);
}

// Test_ExpectationMaximization_ReturnsCovariance
void test_expectation_maximization_returns_covariance() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    std::vector<double> parameters;
    corehydro::numerics::math::linalg::Matrix covariance(0, 0);
    int iterations = 0;
    model.expectation_maximization(parameters, covariance, iterations);

    CHECK_EQ(covariance.number_of_rows(), static_cast<int>(parameters.size()));
    CHECK_EQ(covariance.number_of_columns(), static_cast<int>(parameters.size()));
}

// Test_ExpectationMaximization_WeightsSumToOne
void test_expectation_maximization_weights_sum_to_one() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    std::vector<double> parameters;
    corehydro::numerics::math::linalg::Matrix covariance(0, 0);
    int iterations = 0;
    model.expectation_maximization(parameters, covariance, iterations);

    // First two parameters are weights.
    double weight_sum = parameters[0] + parameters[1];
    CHECK_NEAR(weight_sum, 1.0, 0.01);  // EM may not be perfectly converged.
}

// ===========================================================================================
// Clone tests.
// ===========================================================================================

// Test_Clone_CreatesIndependentCopy
void test_clone_creates_independent_copy() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Gumbel};
    MixtureModel original(std::move(df), types);
    original.set_is_zero_inflated(false);

    MixtureModel clone = original.clone();

    CHECK_TRUE(&original != &clone);
    CHECK_TRUE(original.mixture() != clone.mixture());
}

// Test_Clone_PreservesIsZeroInflated (extended by BestFit v2.0.0 commit b43943c "Fix
// BestFit port review findings" -- the C# test now additionally checks that the clone's
// likelihood/PDF/CDF surface matches the original's and that its Parameters are
// independent copies, beyond the original flag-equality check).
void test_clone_preserves_is_zero_inflated() {
    DataFrame df = create_zero_inflated_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel original(std::move(df), types, /*is_zero_inflated=*/true);
    std::vector<double> parameters = parameter_values(original);
    double likelihood = original.data_log_likelihood(parameters);
    double pdf = original.mixture()->pdf(100.0);
    double cdf = original.mixture()->cdf(100.0);

    MixtureModel clone = original.clone();

    CHECK_EQ(clone.is_zero_inflated(), original.is_zero_inflated());
    // M9-lesson end-state check (beyond the C# test): the cloned model's wrapped mixture
    // carries the same zero-inflation state as the original's.
    CHECK_TRUE(clone.mixture()->is_zero_inflated());
    CHECK_NEAR(clone.mixture()->zero_weight(), original.mixture()->zero_weight(), 0.0);

    // b43943c additions: the clone's likelihood/PDF/CDF surface matches the original's
    // (proving the re-synced mixture is functionally, not just flag-wise, identical), and
    // its Parameters are independent (mutating the clone's copy leaves the original's
    // untouched).
    std::vector<double> clone_parameters = parameter_values(clone);
    CHECK_NEAR(clone.data_log_likelihood(clone_parameters), likelihood, 1e-12);
    CHECK_NEAR(clone.mixture()->pdf(100.0), pdf, 1e-12);
    CHECK_NEAR(clone.mixture()->cdf(100.0), cdf, 1e-12);

    double original_value = original.parameters()[0].value();
    clone.parameters()[0].set_value(original_value + 1.0);
    CHECK_EQ(original.parameters()[0].value(), original_value);
}

// T14 (BestFit v2.0.0, commit b43943c "Fix BestFit port review findings"): Clone's
// zero-inflation re-sync now reads the MODEL-LEVEL `is_zero_inflated_` flag (matching the
// C# `result.Mixture!.IsZeroInflated = result._isZeroInflated;`), not the source mixture's
// OWN IsZeroInflated -- a distinction the C# test above never exercises because the two
// stay in sync on every path the C# test constructs through. This model nudges the wrapped
// Mixture's zero-inflation state out of sync with the model flag via the mutable mixture()
// accessor (reachable in practice: `mixture()` is publicly mutable, unlike an ordinary
// setter-guarded property) and confirms the clone follows the MODEL flag, with ZeroWeight
// forced to 0.0 when the model flag says "not zero-inflated" (the C# `result._isZeroInflated
// ? Mixture!.ZeroWeight : 0.0` ternary).
void test_clone_zero_inflation_resync_uses_model_flag_not_mixture_flag() {
    DataFrame df = create_zero_inflated_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel original(std::move(df), types, /*is_zero_inflated=*/false);

    original.mixture()->set_is_zero_inflated(true);
    original.mixture()->set_zero_weight(0.25);
    CHECK_TRUE(!original.is_zero_inflated());          // model flag still false
    CHECK_TRUE(original.mixture()->is_zero_inflated());  // mixture's own flag now true

    MixtureModel clone = original.clone();

    CHECK_EQ(clone.is_zero_inflated(), original.is_zero_inflated());  // both false
    CHECK_TRUE(!clone.mixture()->is_zero_inflated());
    CHECK_NEAR(clone.mixture()->zero_weight(), 0.0, 0.0);
}

// Test_Clone_PreservesQuantilePriors
void test_clone_preserves_quantile_priors() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel original(std::move(df), types);
    original.set_enable_quantile_priors(true);
    original.set_default_quantile_priors();

    MixtureModel clone = original.clone();

    CHECK_EQ(clone.quantile_priors().size(), original.quantile_priors().size());
}

// Test_Clone_ParametersAreIndependent
void test_clone_parameters_are_independent() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel original(std::move(df), types);
    double original_value = original.parameters()[0].value();

    MixtureModel clone = original.clone();
    original.parameters()[0].set_value(99999);

    CHECK_EQ(clone.parameters()[0].value(), original_value);
}

// ===========================================================================================
// Validation tests.
// ===========================================================================================

// Test_Validate_ValidModel_ReturnsTrue
void test_validate_valid_model_returns_true() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Gumbel};
    MixtureModel model(std::move(df), types);

    ValidationResult result = model.validate();

    CHECK_TRUE(result.is_valid);
}

// Test_Validate_NullDataFrame_ReturnsFalse (C# `DataFrame = null!` -> never-set state).
void test_validate_null_data_frame_returns_false() {
    MixtureModel model;

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_message_contains(result, "Data frame"));
}

// Test_Validate_NullMixture_ReturnsFalse
void test_validate_null_mixture_returns_false() {
    DataFrame df = create_sample_data_frame();
    MixtureModel model;
    model.set_data_frame(std::move(df));
    model.set_mixture(nullptr);

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_message_contains(result, "Mixture"));
}

// Test_Validate_TooManyComponents_ReturnsFalse
void test_validate_too_many_components_returns_false() {
    DataFrame df = create_sample_data_frame();
    MixtureModel model;
    model.set_data_frame(std::move(df));

    // Create mixture with 4 components (exceeds limit of 3).
    std::vector<std::unique_ptr<UnivariateDistributionBase>> dists;
    dists.push_back(std::make_unique<Normal>(100, 10));
    dists.push_back(std::make_unique<Normal>(200, 20));
    dists.push_back(std::make_unique<Normal>(300, 30));
    dists.push_back(std::make_unique<Normal>(400, 40));
    model.set_mixture(std::make_unique<Mixture>(std::vector<double>{0.25, 0.25, 0.25, 0.25},
                                                std::move(dists)));

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_message_contains(result, "1 to 3"));
}

// Test_Validate_LogDistributionWithNonPositiveData_ReturnsFalse
void test_validate_log_distribution_with_non_positive_data_returns_false() {
    DataFrame df = create_zero_inflated_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::LogNormal};
    MixtureModel model(std::move(df), types, /*is_zero_inflated=*/false);

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_message_contains(result, "log-based") ||
               any_message_contains(result, "non-positive"));
}

// Test_Validate_LogDistributionWithZeroInflation_ReturnsTrue
void test_validate_log_distribution_with_zero_inflation_returns_true() {
    DataFrame df = create_zero_inflated_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::LogNormal};
    MixtureModel model(std::move(df), types, /*is_zero_inflated=*/true);

    ValidationResult result = model.validate();

    // Zero inflation handles the non-positive values.
    CHECK_TRUE(result.is_valid);
}

// ===========================================================================================
// Engineering application tests.
// ===========================================================================================

// Test_MixtureModel_BimodalFloodDistribution
void test_mixture_model_bimodal_flood_distribution() {
    // Typical scenario: snowmelt floods (lower, more frequent) and rainfall floods
    // (higher, less frequent).
    DataFrame df = create_bimodal_data_frame();
    std::vector<UnivariateDistributionType> types{
        UnivariateDistributionType::Normal,  // Snowmelt population
        UnivariateDistributionType::Gumbel   // Rainfall population
    };
    MixtureModel model(std::move(df), types);

    ValidationResult valid = model.validate();
    CHECK_TRUE(valid.is_valid);

    std::vector<double> parameters = parameter_values(model);
    double log_lh = model.log_likelihood(parameters);
    CHECK_TRUE(log_lh != -kInf);
}

// Test_MixtureModel_IntermittentStream
void test_mixture_model_intermittent_stream() {
    // Intermittent stream with many zero-flow years.
    DataFrame df = create_zero_inflated_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::GammaDistribution};
    MixtureModel model(std::move(df), types, /*is_zero_inflated=*/true);

    ValidationResult valid = model.validate();
    CHECK_TRUE(valid.is_valid);

    // Zero weight should be significant for intermittent stream.
    CHECK_TRUE(model.mixture()->zero_weight() > 0.3);
}

// Test_MixtureModel_SeasonalFloodPopulations
void test_mixture_model_seasonal_flood_populations() {
    // Spring snowmelt, summer thunderstorm, fall tropical system.
    DataFrame df = create_bimodal_data_frame();
    std::vector<UnivariateDistributionType> types{
        UnivariateDistributionType::Normal, UnivariateDistributionType::Gumbel,
        UnivariateDistributionType::GeneralizedExtremeValue};
    MixtureModel model(std::move(df), types);

    CHECK_EQ(model.mixture()->component_count(), 3);

    // Weights should sum to 1.
    double weight_sum = 0.0;
    for (double w : model.mixture()->weights()) weight_sum += w;
    CHECK_NEAR(weight_sum, 1.0, 1e-10);
}

// ===========================================================================================
// Edge cases.
// ===========================================================================================

// Numerics v2.2 rejects a one-observation sample before a positive scale can be initialized.
void test_mixture_model_single_data_point() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<ExactData>{ExactData(2000, 1000)}));
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};

    CHECK_THROWS(MixtureModel(std::move(df), types));
}

// Numerics v2.2 rejects a constant initialization sample before constructing the model.
void test_mixture_model_all_same_value_throws_on_degenerate_data() {
    DataFrame df;
    std::vector<ExactData> data;
    for (int i = 0; i < 10; ++i) data.emplace_back(2000 + i, 500);
    df.set_exact_series(ExactSeries(data));

    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CHECK_THROWS(MixtureModel(std::move(df), types));
}

// Test_MixtureModel_LargeValues
void test_mixture_model_large_values() {
    DataFrame df;
    std::vector<ExactData> data;
    for (int i = 0; i < 10; ++i) data.emplace_back(2000 + i, 1e8 + i * 1e6);
    df.set_exact_series(ExactSeries(data));

    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    ValidationResult valid = model.validate();
    CHECK_TRUE(valid.is_valid);
}

// Test_MixtureModel_SmallValues
void test_mixture_model_small_values() {
    DataFrame df;
    std::vector<ExactData> data;
    for (int i = 0; i < 10; ++i) data.emplace_back(2000 + i, 0.001 + i * 0.0001);
    df.set_exact_series(ExactSeries(data));

    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    ValidationResult valid = model.validate();
    CHECK_TRUE(valid.is_valid);
}

// ===========================================================================================
// Jeffreys prior tests.
// ===========================================================================================

// Test_JeffreysPrior_AffectsPriorLogLikelihood
void test_jeffreys_prior_affects_prior_log_likelihood() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);

    model.set_use_jeffreys_rule_for_scale(false);
    double prior_no_jeffreys = model.prior_log_likelihood(parameters);

    model.set_use_jeffreys_rule_for_scale(true);
    double prior_with_jeffreys = model.prior_log_likelihood(parameters);

    // Jeffreys prior adds -log(scale) term, so they should differ.
    CHECK_TRUE(prior_no_jeffreys != prior_with_jeffreys);
}

// T14 (BestFit v2.0.0, commit 1abe795 "Guard composite Jeffreys scale priors"): transcribes
// MixtureModelTests.cs's `CreateModelWithManualPriors` regression-only helper. A
// single-parameter Numerics family (e.g. Poisson) does not implement
// IMaximumLikelihoodEstimation, so the normal BestFit setup cascade (any ctor/setter that
// reaches SetDefaultParameters with both a Mixture and a valid DataFrame present ->
// dynamic_cast<IMaximumLikelihoodEstimation&>) throws std::bad_cast on it -- exactly the bug
// this task guards against, so the crash-inducing cascade cannot be the vehicle for reaching
// it. This helper takes the same escape hatch the C# test uses (constructing before any
// DataFrame is attached, so SetDefaultParameters' early-return guard --
// `!has_data_frame()`, C++'s analogue of the C# `DataFrame == null` check -- short-circuits
// before the per-component cast; the parameterless MixtureModel() ctor's own default
// Normal-Normal set_mixture() call takes the same safe path), then adds one ModelParameter
// per raw distribution parameter directly, mirroring the C# test's manual `Parameters.Add`
// loop.
MixtureModel create_model_with_manual_priors(
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distributions) {
    MixtureModel model;
    model.set_use_default_flat_priors(false);
    std::vector<double> weights(distributions.size(), 1.0 / static_cast<double>(distributions.size()));
    model.set_mixture(std::make_unique<Mixture>(std::move(weights), std::move(distributions)));

    std::vector<double> values = model.mixture()->get_parameters();
    for (std::size_t i = 0; i < values.size(); ++i) {
        double value = values[i];
        double sigma = std::max(1.0, std::abs(value) * 0.1);
        model.parameters().push_back(ModelParameter(
            "", "Parameter " + std::to_string(i + 1), value, -kInf, kInf,
            std::make_unique<Normal>(value, sigma)));
    }
    return model;
}

// Test_JeffreysPrior_OneParameterComponent_OmitsScaleContribution
void test_jeffreys_prior_one_parameter_component_omits_scale_contribution() {
    std::vector<std::unique_ptr<UnivariateDistributionBase>> dists;
    dists.push_back(std::make_unique<Poisson>(2000.0));
    MixtureModel model = create_model_with_manual_priors(std::move(dists));
    model.set_use_jeffreys_rule_for_scale(true);
    std::vector<double> parameters = parameter_values(model);

    double scalar = model.prior_log_likelihood(parameters);
    auto pointwise = model.pointwise_prior_log_likelihood(parameters);

    CHECK_TRUE(std::isfinite(scalar));
    bool any_jeffreys = false;
    for (const auto& p : pointwise)
        if (p.type() == PriorComponentType::JeffreysScalePrior) any_jeffreys = true;
    CHECK_TRUE(!any_jeffreys);

    double sum = 0.0;
    for (const auto& p : pointwise) sum += p.log_likelihood();
    CHECK_NEAR(sum, scalar, 1e-12);
}

// Test_JeffreysPrior_MixedComponents_AppliesOnlyAvailableScaleContribution
void test_jeffreys_prior_mixed_components_applies_only_available_scale_contribution() {
    std::vector<std::unique_ptr<UnivariateDistributionBase>> dists;
    dists.push_back(std::make_unique<Poisson>(2000.0));
    dists.push_back(std::make_unique<Normal>(2000.0, 500.0));
    MixtureModel model = create_model_with_manual_priors(std::move(dists));
    model.set_use_jeffreys_rule_for_scale(true);
    std::vector<double> parameters = parameter_values(model);

    double scalar = model.prior_log_likelihood(parameters);
    auto pointwise = model.pointwise_prior_log_likelihood(parameters);

    int jeffreys_count = 0;
    for (const auto& p : pointwise)
        if (p.type() == PriorComponentType::JeffreysScalePrior) ++jeffreys_count;
    CHECK_EQ(jeffreys_count, 1);

    double sum = 0.0;
    for (const auto& p : pointwise) sum += p.log_likelihood();
    CHECK_NEAR(sum, scalar, 1e-12);
}

// T14 addition (beyond the transcribed C# tests): a non-positive scale on an
// available-scale component returns -Inf IMMEDIATELY (an early function return out of
// prior_log_likelihood), not a +Inf subtraction that could cancel against another +Inf term
// into NaN (the "avoids Inf-Inf NaN" fix the task brief calls out).
void test_jeffreys_prior_nonpositive_scale_returns_negative_infinity() {
    std::vector<std::unique_ptr<UnivariateDistributionBase>> dists;
    dists.push_back(std::make_unique<Poisson>(2000.0));
    dists.push_back(std::make_unique<Normal>(2000.0, 500.0));
    MixtureModel model = create_model_with_manual_priors(std::move(dists));
    model.set_use_jeffreys_rule_for_scale(true);
    std::vector<double> parameters = parameter_values(model);
    // Weights (2 of them) come first in the flat vector, then Poisson's lambda, then Normal's
    // (mu, sigma); the last entry is the Normal scale.
    parameters.back() = -1.0;  // Normal's sigma -> negative (not the [0, 1e-16) clamp zone).

    double scalar = model.prior_log_likelihood(parameters);

    CHECK_TRUE(scalar == -kInf);
}

// ===========================================================================================
// ISimulatable / GenerateRandomValues (no upstream MixtureModelTests method exercises the
// stream itself; these pin the ported guards + determinism of the C# Mixture component-
// selection sampler the model delegates to).
// ===========================================================================================

void test_generate_random_values_guards_and_deterministic_seed() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Normal};
    MixtureModel model(std::move(df), types);

    CHECK_THROWS(model.generate_random_values(0, 42));   // sampleSize must be positive
    CHECK_THROWS(model.generate_random_values(-5, 42));  // sampleSize must be positive

    std::vector<double> a = model.generate_random_values(8, 42);
    std::vector<double> b = model.generate_random_values(8, 42);
    CHECK_EQ(a.size(), static_cast<std::size_t>(8));
    bool identical = true;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) identical = false;
    CHECK_TRUE(identical);
}

}  // namespace

int main() {
    // Constructor tests.
    test_constructor_empty_creates_default_normal_mixture();
    test_constructor_empty_has_equal_weights();
    test_constructor_with_data_and_mixture_sets_properties();
    test_constructor_with_distribution_types_creates_components();
    test_constructor_with_distribution_types_zero_inflated();
    test_constructor_with_distributions_uses_provided_instances();
    test_constructor_with_empty_distribution_types_throws();
    test_constructor_with_empty_distributions_throws();

    // Property tests.
    test_mixture_set_and_get();
    test_is_zero_inflated_set_and_get();
    test_is_zero_inflated_sets_zero_weight();
    test_is_zero_inflated_false_zero_weight_is_zero();
    test_use_single_quantile_always_true();
    test_data_frame_set_triggers_parameter_update();

    // SetDefaultMixture tests.
    test_set_default_mixture_single_component();
    test_set_default_mixture_two_components_equal_weights();
    test_set_default_mixture_three_components_equal_weights();
    test_set_default_mixture_weights_sum_to_one();
    test_set_default_mixture_empty_list_throws();

    // SetDefaultParameters tests.
    test_set_default_parameters_two_components_has_weight_parameters();
    test_set_default_parameters_single_component_no_weight_parameters();
    test_set_default_parameters_weight_bounds();
    test_set_default_parameters_weights_have_uniform_prior();

    // SetDefaultQuantilePriors tests.
    test_set_default_quantile_priors_single_quantile();
    test_set_default_quantile_priors_disabled_empty_list();
    test_set_default_quantile_priors_uses_ln_normal_distribution();

    // LogLikelihood tests.
    test_log_likelihood_returns_finite_value();
    test_data_log_likelihood_returns_finite_value();
    test_prior_log_likelihood_returns_finite_value();
    test_log_likelihood_equals_data_plus_prior();
    test_pointwise_data_log_likelihood_returns_correct_count();
    test_pointwise_data_log_likelihood_sum_equals_data_log_likelihood();

    // Zero-inflation tests.
    test_zero_inflated_log_likelihood_handles_zero_values();
    test_zero_inflated_zero_weight_reflects_data();

    // ExpectationMaximization tests.
    test_expectation_maximization_returns_parameters();
    test_expectation_maximization_converges_within_max_iterations();
    test_expectation_maximization_returns_covariance();
    test_expectation_maximization_weights_sum_to_one();

    // Clone tests.
    test_clone_creates_independent_copy();
    test_clone_preserves_is_zero_inflated();
    test_clone_zero_inflation_resync_uses_model_flag_not_mixture_flag();
    test_clone_preserves_quantile_priors();
    test_clone_parameters_are_independent();

    // Validation tests.
    test_validate_valid_model_returns_true();
    test_validate_null_data_frame_returns_false();
    test_validate_null_mixture_returns_false();
    test_validate_too_many_components_returns_false();
    test_validate_log_distribution_with_non_positive_data_returns_false();
    test_validate_log_distribution_with_zero_inflation_returns_true();

    // Engineering application tests.
    test_mixture_model_bimodal_flood_distribution();
    test_mixture_model_intermittent_stream();
    test_mixture_model_seasonal_flood_populations();

    // Edge cases.
    test_mixture_model_single_data_point();
    test_mixture_model_all_same_value_throws_on_degenerate_data();
    test_mixture_model_large_values();
    test_mixture_model_small_values();

    // Jeffreys prior tests.
    test_jeffreys_prior_affects_prior_log_likelihood();
    test_jeffreys_prior_one_parameter_component_omits_scale_contribution();
    test_jeffreys_prior_mixed_components_applies_only_available_scale_contribution();
    test_jeffreys_prior_nonpositive_scale_returns_negative_infinity();

    // ISimulatable guards + seeded determinism.
    test_generate_random_values_guards_and_deterministic_seed();

    return chtest::summary("mixture_model");
}
