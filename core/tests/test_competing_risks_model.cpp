// Standalone test for corehydro::models::CompetingRisksModel (Phase 5, M11).
//
// Oracle for behavior is the C# source itself:
//   - upstream/RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/
//     CompetingRisksModel.cs @ fc28c0c, and its base UnivariateDistributionModelBase.cs
//     @ fc28c0c;
//   - a full transcription of the upstream test class
//     RMC.BestFit.Tests/Univariate/CompetingRisksModelTests.cs @ fc28c0c (every method
//     except the XML / serialization surface and the null-argument tests that C++'s type
//     system makes untranscribable -- the skip list, with reasons, lives in the M11 report):
//       Test_ToXElement_ContainsCompetingRisksModelElement, Test_ToXElement_
//       ContainsDistributionElement, Test_ToXElement_ContainsParametersElement,
//       Test_ToXElement_ContainsUseSingleQuantileAttribute (ToXElement is a project-wide
//       non-port); Test_Constructor_NullDistribution_ThrowsException (the C++ ctor takes
//       `const CompetingRisks&` -- a reference cannot be null); Test_SetParameterValues_
//       NullParameters_ThrowsException (a std::vector<double> argument cannot be null).
// Hardcoded oracles are allowed here (internal support layer); public-API oracle values stay
// in fixtures/ only (the fixture wiring for the Models layer arrives in M13/M14).
//
// Test-surface adaptations forced by the port (each noted at the test):
//   - C# `Assert.AreNotSame(dist, model.CompetingRisks)` maps to a pointer-identity check.
//   - C# `model.DataFrame = null!` maps to the never-set state: a default-constructed model
//     has no frame (see the base header's nullability note).
//   - C# `Assert.IsInstanceOfType(x, typeof(Normal))` maps to a type()-enum check.
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
#include "corehydro/models/support/data_component.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/models/univariate_distribution/competing_risks_model.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/competing_risks.hpp"
#include "corehydro/numerics/distributions/gumbel.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/poisson.hpp"
#include "check.hpp"

using corehydro::models::CompetingRisksModel;
using corehydro::models::DataComponentType;
using corehydro::models::DataFrame;
using corehydro::models::ExactData;
using corehydro::models::ExactSeries;
using corehydro::models::IntervalData;
using corehydro::models::IntervalSeries;
using corehydro::models::ModelParameter;
using corehydro::models::PriorComponentType;
using corehydro::models::ValidationResult;
using corehydro::numerics::distributions::CompetingRisks;
using corehydro::numerics::distributions::Gumbel;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::Poisson;
using corehydro::numerics::distributions::UnivariateDistributionBase;
using corehydro::numerics::distributions::UnivariateDistributionType;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ===========================================================================================
// Test data helpers (CompetingRisksModelTests.cs, "Test Data Helper" region).
// ===========================================================================================

// Creates a sample data frame with positive exact observations.
DataFrame create_sample_data_frame() {
    DataFrame df;
    std::vector<ExactData> data{
        ExactData(1990, 1500), ExactData(1991, 2000), ExactData(1992, 1800),
        ExactData(1993, 2500), ExactData(1994, 3000), ExactData(1995, 2200),
        ExactData(1996, 3500), ExactData(1997, 2800), ExactData(1998, 4000),
        ExactData(1999, 3200)};
    df.set_exact_series(ExactSeries(data));
    return df;
}

// Creates a data frame with extreme event characteristics.
DataFrame create_extreme_event_data_frame() {
    DataFrame df;
    std::vector<ExactData> data;

    // Regular events.
    for (int i = 0; i < 20; ++i) data.emplace_back(1980 + i, 1000 + i * 100);

    // Extreme events (hurricane-like).
    data.emplace_back(2000, 8000);
    data.emplace_back(2005, 10000);
    data.emplace_back(2010, 12000);

    df.set_exact_series(ExactSeries(data));
    return df;
}

// Creates a data frame with interval data (the upstream helper's 4-arg IntervalData ctor:
// index, lowerValue, value (most-likely, strictly between lower and upper), upperValue).
DataFrame create_interval_data_frame() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<ExactData>{
        ExactData(1990, 2000), ExactData(1991, 2500), ExactData(1992, 3000),
        ExactData(1993, 2200), ExactData(1994, 2700)}));
    df.set_interval_series(IntervalSeries(std::vector<IntervalData>{
        IntervalData(1800, 4000, 5000, 6000), IntervalData(1850, 5000, 6500, 8000)}));
    return df;
}

std::vector<double> parameter_values(const CompetingRisksModel& model) {
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

std::unique_ptr<CompetingRisks> make_competing_risks(
    std::vector<std::unique_ptr<UnivariateDistributionBase>> dists) {
    return std::make_unique<CompetingRisks>(std::move(dists));
}

// ===========================================================================================
// Constructor tests.
// ===========================================================================================

// Test_Constructor_EmptyConstructor_CreatesModel
void test_constructor_empty_constructor_creates_model() {
    CompetingRisksModel model;

    CHECK_TRUE(model.use_single_quantile());
}

// Test_Constructor_WithDistributionTypes_SetsComponents
void test_constructor_with_distribution_types_sets_components() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Gumbel};

    CompetingRisksModel model(std::move(df), types);

    CHECK_TRUE(model.has_competing_risks());
    CHECK_EQ(model.competing_risks()->component_count(), 2);
    CHECK_TRUE(model.competing_risks()->component(0).type() ==
               UnivariateDistributionType::Normal);
    CHECK_TRUE(model.competing_risks()->component(1).type() ==
               UnivariateDistributionType::Gumbel);
}

// Test_Constructor_WithCompetingRisksDistribution_ClonesDistribution (AreNotSame ->
// pointer-identity check; see header).
void test_constructor_with_competing_risks_distribution_clones_distribution() {
    DataFrame df = create_sample_data_frame();
    std::vector<std::unique_ptr<UnivariateDistributionBase>> comps;
    comps.push_back(std::make_unique<Normal>(2000, 500));
    comps.push_back(std::make_unique<Gumbel>(2500, 600));
    CompetingRisks dist(std::move(comps));

    CompetingRisksModel model(std::move(df), dist);

    CHECK_TRUE(model.has_competing_risks());
    CHECK_TRUE(model.competing_risks() != &dist);
}

// Test_Constructor_EmptyDistributionTypes_ThrowsException (C# ArgumentException).
void test_constructor_empty_distribution_types_throws() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types;

    CHECK_THROWS(CompetingRisksModel(std::move(df), types));
}

// Test_Constructor_TooManyDistributions_ThrowsException (C# ArgumentException).
void test_constructor_too_many_distributions_throws() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{
        UnivariateDistributionType::Normal, UnivariateDistributionType::Gumbel,
        UnivariateDistributionType::GammaDistribution,
        UnivariateDistributionType::Weibull};  // Fourth distribution exceeds limit.

    CHECK_THROWS(CompetingRisksModel(std::move(df), types));
}

// Test_Constructor_NullDistribution_ThrowsException: SKIPPED -- the C++ ctor takes
// `const CompetingRisks&`; a reference cannot be null (type-system guarantee).

// ===========================================================================================
// Property tests.
// ===========================================================================================

// Test_CompetingRisks_SetAndGet
void test_competing_risks_set_and_get() {
    CompetingRisksModel model;
    std::vector<std::unique_ptr<UnivariateDistributionBase>> comps;
    comps.push_back(std::make_unique<Normal>(1000, 200));
    comps.push_back(std::make_unique<Gumbel>(1500, 300));

    model.set_competing_risks(make_competing_risks(std::move(comps)));

    CHECK_TRUE(model.has_competing_risks());
    CHECK_EQ(model.competing_risks()->component_count(), 2);
}

// Test_UseSingleQuantile_AlwaysReturnsTrue
void test_use_single_quantile_always_returns_true() {
    CompetingRisksModel model;

    CHECK_TRUE(model.use_single_quantile());

    // Setting to false should be ignored.
    model.set_use_single_quantile(false);
    CHECK_TRUE(model.use_single_quantile());
}

// Test_DataFrame_SetTriggersParameterUpdate
void test_data_frame_set_triggers_parameter_update() {
    DataFrame df1 = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Gumbel};
    CompetingRisksModel model(std::move(df1), types);
    int initial_count = model.number_of_parameters();

    DataFrame df2 = create_extreme_event_data_frame();
    model.set_data_frame(std::move(df2));

    // Parameters should still exist after data change.
    CHECK_EQ(model.number_of_parameters(), initial_count);
}

// ===========================================================================================
// SetDefaultParameters tests.
// ===========================================================================================

// Test_SetDefaultParameters_CreatesParametersForAllComponents
void test_set_default_parameters_creates_parameters_for_all_components() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{
        UnivariateDistributionType::Normal,  // 2 parameters
        UnivariateDistributionType::Gumbel   // 2 parameters
    };
    CompetingRisksModel model(std::move(df), types);

    // Should have 4 total parameters (2 + 2).
    CHECK_EQ(model.number_of_parameters(), 4);
}

// Test_SetDefaultParameters_OwnerNamesAreSet
void test_set_default_parameters_owner_names_are_set() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Gumbel};
    CompetingRisksModel model(std::move(df), types);

    // First component parameters should have D1 owner.
    CHECK_EQ(model.parameters()[0].owner_name(), std::string("D1"));
    CHECK_EQ(model.parameters()[1].owner_name(), std::string("D1"));

    // Second component parameters should have D2 owner.
    CHECK_EQ(model.parameters()[2].owner_name(), std::string("D2"));
    CHECK_EQ(model.parameters()[3].owner_name(), std::string("D2"));
}

// Test_SetDefaultParameters_UniformPriors
void test_set_default_parameters_uniform_priors() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    CHECK_TRUE(model.parameters().size() > 0);
    for (const auto& param : model.parameters()) {
        CHECK_TRUE(param.prior_distribution().type() == UnivariateDistributionType::Uniform);
    }
}

// Test_SetDefaultParameters_ThreeComponents
void test_set_default_parameters_three_components() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{
        UnivariateDistributionType::Normal,             // 2 params
        UnivariateDistributionType::Gumbel,             // 2 params
        UnivariateDistributionType::GammaDistribution   // 2 params
    };
    CompetingRisksModel model(std::move(df), types);

    CHECK_EQ(model.number_of_parameters(), 6);
    bool any_d3 = false;
    for (const auto& p : model.parameters())
        if (p.owner_name() == "D3") any_d3 = true;
    CHECK_TRUE(any_d3);
}

// ===========================================================================================
// SetDefaultQuantilePriors tests.
// ===========================================================================================

// Test_SetDefaultQuantilePriors_SingleQuantile
void test_set_default_quantile_priors_single_quantile() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);
    model.set_enable_quantile_priors(true);

    model.set_default_quantile_priors();

    CHECK_EQ(model.quantile_priors().size(), static_cast<std::size_t>(1));
}

// Test_SetDefaultQuantilePriors_Disabled_EmptyList
void test_set_default_quantile_priors_disabled_empty_list() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);
    model.set_enable_quantile_priors(false);

    model.set_default_quantile_priors();

    CHECK_EQ(model.quantile_priors().size(), static_cast<std::size_t>(0));
}

// Test_SetDefaultQuantilePriors_UsesLnNormalDistribution
void test_set_default_quantile_priors_uses_ln_normal_distribution() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);
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
                                                  UnivariateDistributionType::Gumbel};
    CompetingRisksModel model(std::move(df), types);

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
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    double data_log_lh = model.data_log_likelihood(parameters);

    CHECK_TRUE(!std::isnan(data_log_lh));
    CHECK_TRUE(data_log_lh != kInf);
}

// Test_PriorLogLikelihood_ReturnsFiniteValue
void test_prior_log_likelihood_returns_finite_value() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    double prior_log_lh = model.prior_log_likelihood(parameters);

    CHECK_TRUE(!std::isnan(prior_log_lh));
    CHECK_TRUE(prior_log_lh != kInf);
}

// Test_LogLikelihood_EqualsDataPlusPrior
void test_log_likelihood_equals_data_plus_prior() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    double full_log_lh = model.log_likelihood(parameters);
    double data_log_lh = model.data_log_likelihood(parameters);
    double prior_log_lh = model.prior_log_likelihood(parameters);

    CHECK_NEAR(data_log_lh + prior_log_lh, full_log_lh, 1e-6);
}

// Test_PointwiseDataLogLikelihood_ReturnsCorrectCount
void test_pointwise_data_log_likelihood_returns_correct_count() {
    DataFrame df = create_sample_data_frame();
    std::size_t exact_count = df.exact_series().count();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(parameters);

    CHECK_EQ(pointwise.size(), exact_count);
}

// Test_PointwiseDataLogLikelihood_SumEqualsDataLogLikelihood
void test_pointwise_data_log_likelihood_sum_equals_data_log_likelihood() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(parameters);
    double data_log_lh = model.data_log_likelihood(parameters);

    double sum = 0.0;
    for (double ll : pointwise) sum += ll;
    CHECK_NEAR(data_log_lh, sum, 1e-6);
}

// ===========================================================================================
// PointwiseDataLogLikelihoodComponents tests.
// ===========================================================================================

// Test_PointwiseDataLogLikelihoodComponents_ReturnsAllDataPoints
void test_pointwise_data_log_likelihood_components_returns_all_data_points() {
    DataFrame df = create_interval_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    auto components = model.pointwise_data_log_likelihood_components(parameters);

    // 5 exact + 2 interval = 7 total.
    CHECK_EQ(components.size(), static_cast<std::size_t>(7));
}

// Test_PointwiseDataLogLikelihoodComponents_HasCorrectTypes
void test_pointwise_data_log_likelihood_components_has_correct_types() {
    DataFrame df = create_interval_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    auto components = model.pointwise_data_log_likelihood_components(parameters);

    int exact_count = 0;
    int interval_count = 0;
    for (const auto& c : components) {
        if (c.type() == DataComponentType::Exact) ++exact_count;
        if (c.type() == DataComponentType::Interval) ++interval_count;
    }

    CHECK_EQ(exact_count, 5);
    CHECK_EQ(interval_count, 2);
}

// ===========================================================================================
// PointwisePriorLogLikelihood tests.
// ===========================================================================================

// Test_PointwisePriorLogLikelihood_ReturnsParameterPriors
void test_pointwise_prior_log_likelihood_returns_parameter_priors() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    auto priors = model.pointwise_prior_log_likelihood(parameters);

    bool any_parameter_prior = false;
    for (const auto& p : priors)
        if (p.type() == PriorComponentType::ParameterPrior) any_parameter_prior = true;
    CHECK_TRUE(any_parameter_prior);
}

// Test_PointwisePriorLogLikelihood_WithJeffreysPrior
void test_pointwise_prior_log_likelihood_with_jeffreys_prior() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);
    model.set_use_jeffreys_rule_for_scale(true);

    std::vector<double> parameters = parameter_values(model);
    auto priors = model.pointwise_prior_log_likelihood(parameters);

    bool any_jeffreys = false;
    for (const auto& p : priors)
        if (p.type() == PriorComponentType::JeffreysScalePrior) any_jeffreys = true;
    CHECK_TRUE(any_jeffreys);
}

// Test_PointwisePriorLogLikelihood_WithQuantilePrior
void test_pointwise_prior_log_likelihood_with_quantile_prior() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);
    model.set_enable_quantile_priors(true);
    model.set_default_quantile_priors();
    model.process_quantile_priors();

    std::vector<double> parameters = parameter_values(model);
    auto priors = model.pointwise_prior_log_likelihood(parameters);

    bool any_quantile = false;
    for (const auto& p : priors)
        if (p.type() == PriorComponentType::QuantilePrior) any_quantile = true;
    CHECK_TRUE(any_quantile);
}

// ===========================================================================================
// Clone tests.
// ===========================================================================================

// Test_Clone_CreatesIndependentCopy
void test_clone_creates_independent_copy() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Gumbel};
    CompetingRisksModel original(std::move(df), types);

    CompetingRisksModel clone = original.clone();

    CHECK_TRUE(&original != &clone);
    CHECK_TRUE(original.competing_risks() != clone.competing_risks());
}

// Test_Clone_PreservesQuantilePriors
void test_clone_preserves_quantile_priors() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel original(std::move(df), types);
    original.set_enable_quantile_priors(true);
    original.set_default_quantile_priors();

    CompetingRisksModel clone = original.clone();

    CHECK_EQ(clone.quantile_priors().size(), original.quantile_priors().size());
}

// Test_Clone_ParametersAreIndependent
void test_clone_parameters_are_independent() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel original(std::move(df), types);
    double original_value = original.parameters()[0].value();

    CompetingRisksModel clone = original.clone();
    original.parameters()[0].set_value(99999);

    CHECK_EQ(clone.parameters()[0].value(), original_value);
}

// Test_Clone_PreservesJeffreysRuleSetting
void test_clone_preserves_jeffreys_rule_setting() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel original(std::move(df), types);
    original.set_use_jeffreys_rule_for_scale(true);

    CompetingRisksModel clone = original.clone();

    CHECK_EQ(clone.use_jeffreys_rule_for_scale(), original.use_jeffreys_rule_for_scale());
}

// ===========================================================================================
// Serialization tests: SKIPPED (Test_ToXElement_* x4 -- ToXElement / the XElement ctor are
// project-wide non-ports).
// ===========================================================================================

// ===========================================================================================
// Validation tests.
// ===========================================================================================

// Test_Validate_ValidModel_ReturnsTrue
void test_validate_valid_model_returns_true() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Gumbel};
    CompetingRisksModel model(std::move(df), types);

    ValidationResult result = model.validate();

    CHECK_TRUE(result.is_valid);
}

// Test_Validate_NullDataFrame_ReturnsFalse (C# `DataFrame = null!` -> never-set state).
void test_validate_null_data_frame_returns_false() {
    CompetingRisksModel model;

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_message_contains(result, "Data frame"));
}

// Test_Validate_NullCompetingRisks_ReturnsFalse
void test_validate_null_competing_risks_returns_false() {
    DataFrame df = create_sample_data_frame();
    CompetingRisksModel model;
    model.set_data_frame(std::move(df));
    model.set_competing_risks(nullptr);

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_message_contains(result, "Competing risks"));
}

// Test_Validate_TooManyComponents_ReturnsFalse
void test_validate_too_many_components_returns_false() {
    DataFrame df = create_sample_data_frame();
    CompetingRisksModel model;
    model.set_data_frame(std::move(df));

    // Create distribution with 4 components (exceeds limit of 3).
    std::vector<std::unique_ptr<UnivariateDistributionBase>> dists;
    dists.push_back(std::make_unique<Normal>(100, 10));
    dists.push_back(std::make_unique<Normal>(200, 20));
    dists.push_back(std::make_unique<Normal>(300, 30));
    dists.push_back(std::make_unique<Normal>(400, 40));
    model.set_competing_risks(make_competing_risks(std::move(dists)));

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_message_contains(result, "3"));
}

// Test_Validate_SingleComponent_ReturnsTrue
void test_validate_single_component_returns_true() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Gumbel};
    CompetingRisksModel model(std::move(df), types);

    ValidationResult result = model.validate();

    CHECK_TRUE(result.is_valid);
}

// Test_Validate_ThreeComponents_ReturnsTrue
void test_validate_three_components_returns_true() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{
        UnivariateDistributionType::Normal, UnivariateDistributionType::Gumbel,
        UnivariateDistributionType::GammaDistribution};
    CompetingRisksModel model(std::move(df), types);

    ValidationResult result = model.validate();

    CHECK_TRUE(result.is_valid);
}

// ===========================================================================================
// SetParameterValues tests.
// ===========================================================================================

// Test_SetParameterValues_UpdatesParameters
void test_set_parameter_values_updates_parameters() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> new_values{2500.0, 750.0};
    model.set_parameter_values(new_values);

    CHECK_EQ(model.parameters()[0].value(), 2500.0);
    CHECK_EQ(model.parameters()[1].value(), 750.0);
}

// Test_SetParameterValues_UpdatesDistribution
void test_set_parameter_values_updates_distribution() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> new_values{3000.0, 500.0};
    model.set_parameter_values(new_values);

    // Distribution parameters should be updated (C# (Normal)...Mu / .Sigma).
    std::vector<double> dist_parms = model.competing_risks()->component(0).get_parameters();
    CHECK_NEAR(dist_parms[0], 3000.0, 1e-10);
    CHECK_NEAR(dist_parms[1], 500.0, 1e-10);
}

// Test_SetParameterValues_NullParameters_ThrowsException: SKIPPED -- a std::vector<double>
// argument cannot be null (type-system guarantee).

// Test_SetParameterValues_WrongCount_ThrowsException (C# ArgumentException).
void test_set_parameter_values_wrong_count_throws() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> wrong_count{1.0};  // Should be 2 for Normal.
    CHECK_THROWS(model.set_parameter_values(wrong_count));
}

// ===========================================================================================
// Engineering application tests.
// ===========================================================================================

// Test_CompetingRisks_SnowmeltVsRainfall
void test_competing_risks_snowmelt_vs_rainfall() {
    // Snowmelt floods (spring) vs rainfall floods (summer/fall).
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{
        UnivariateDistributionType::Normal,  // Snowmelt
        UnivariateDistributionType::Gumbel   // Rainfall
    };
    CompetingRisksModel model(std::move(df), types);

    ValidationResult valid = model.validate();
    CHECK_TRUE(valid.is_valid);

    std::vector<double> parameters = parameter_values(model);
    double log_lh = model.log_likelihood(parameters);
    CHECK_TRUE(log_lh != -kInf);
}

// Test_CompetingRisks_WithHurricanes
void test_competing_risks_with_hurricanes() {
    // Regular floods vs hurricane-induced floods.
    DataFrame df = create_extreme_event_data_frame();
    std::vector<UnivariateDistributionType> types{
        UnivariateDistributionType::Normal,                  // Regular
        UnivariateDistributionType::GeneralizedExtremeValue  // Hurricanes
    };
    CompetingRisksModel model(std::move(df), types);

    ValidationResult valid = model.validate();
    CHECK_TRUE(valid.is_valid);
}

// Test_CompetingRisks_ThreeSeasons
void test_competing_risks_three_seasons() {
    // Spring snowmelt, summer thunderstorms, fall tropical systems.
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{
        UnivariateDistributionType::Normal, UnivariateDistributionType::Gumbel,
        UnivariateDistributionType::GeneralizedExtremeValue};
    CompetingRisksModel model(std::move(df), types);

    ValidationResult valid = model.validate();
    CHECK_TRUE(valid.is_valid);

    CHECK_EQ(model.competing_risks()->component_count(), 3);
}

// ===========================================================================================
// Jeffreys prior tests.
// ===========================================================================================

// Test_JeffreysPrior_AffectsPriorLogLikelihood
void test_jeffreys_prior_affects_prior_log_likelihood() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);

    model.set_use_jeffreys_rule_for_scale(false);
    double prior_no_jeffreys = model.prior_log_likelihood(parameters);

    model.set_use_jeffreys_rule_for_scale(true);
    double prior_with_jeffreys = model.prior_log_likelihood(parameters);

    // Jeffreys prior adds -log(scale) term.
    CHECK_TRUE(prior_no_jeffreys != prior_with_jeffreys);
}

// Test_JeffreysPrior_GammaDistribution_UsesFirstParameter
void test_jeffreys_prior_gamma_distribution_uses_first_parameter() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{
        UnivariateDistributionType::GammaDistribution};
    CompetingRisksModel model(std::move(df), types);
    model.set_use_jeffreys_rule_for_scale(true);

    std::vector<double> parameters = parameter_values(model);
    auto priors = model.pointwise_prior_log_likelihood(parameters);

    // Should have Jeffreys scale prior for Gamma's first parameter.
    bool any_jeffreys = false;
    for (const auto& p : priors)
        if (p.type() == PriorComponentType::JeffreysScalePrior) any_jeffreys = true;
    CHECK_TRUE(any_jeffreys);
}

// Test_JeffreysPrior_Weibull_UsesFirstParameter
void test_jeffreys_prior_weibull_uses_first_parameter() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Weibull};
    CompetingRisksModel model(std::move(df), types);
    model.set_use_jeffreys_rule_for_scale(true);

    std::vector<double> parameters = parameter_values(model);
    auto priors = model.pointwise_prior_log_likelihood(parameters);

    // Should have Jeffreys scale prior for Weibull's first parameter (scale).
    bool any_jeffreys = false;
    for (const auto& p : priors)
        if (p.type() == PriorComponentType::JeffreysScalePrior) any_jeffreys = true;
    CHECK_TRUE(any_jeffreys);
}

// T14 (BestFit v2.0.0, commit 1abe795 "Guard composite Jeffreys scale priors"): transcribes
// CompetingRisksModelTests.cs's `CreateModelWithManualPriors` regression-only helper. A
// single-parameter Numerics family (e.g. Poisson) does not implement
// IMaximumLikelihoodEstimation, so the normal BestFit setup cascade
// (CompetingRisksModel(DataFrame, families) -> SetDefaultParameters ->
// dynamic_cast<IMaximumLikelihoodEstimation&>) throws std::bad_cast on it -- exactly the bug
// this task guards against, so the crash-inducing cascade cannot be the vehicle for
// reaching it. This helper takes the same escape hatch the C# test uses (constructing the
// model before any DataFrame is attached, so SetDefaultParameters' early-return guard --
// `!has_data_frame()`, C++'s analogue of the C# `DataFrame == null` check -- short-circuits
// before the per-component cast), then adds one ModelParameter per raw distribution
// parameter directly, mirroring the C# test's manual `Parameters.Add` loop.
CompetingRisksModel create_model_with_manual_priors(
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distributions) {
    CompetingRisksModel model;
    model.set_use_default_flat_priors(false);
    model.set_competing_risks(make_competing_risks(std::move(distributions)));

    std::vector<double> values = model.competing_risks()->get_parameters();
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
    CompetingRisksModel model = create_model_with_manual_priors(std::move(dists));
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
    CompetingRisksModel model = create_model_with_manual_priors(std::move(dists));
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
    CompetingRisksModel model = create_model_with_manual_priors(std::move(dists));
    model.set_use_jeffreys_rule_for_scale(true);
    std::vector<double> parameters = parameter_values(model);
    parameters.back() = -1.0;  // Normal's sigma -> negative (not the [0, 1e-16) clamp zone).

    double scalar = model.prior_log_likelihood(parameters);

    CHECK_TRUE(scalar == -kInf);
}

// ===========================================================================================
// Edge cases.
// ===========================================================================================

// Numerics v2.2 rejects a one-observation sample before a positive scale can be initialized.
void test_competing_risks_single_data_point() {
    DataFrame df;
    df.set_exact_series(ExactSeries(std::vector<ExactData>{ExactData(2000, 1000)}));
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};

    CHECK_THROWS(CompetingRisksModel(std::move(df), types));
}

// Test_CompetingRisks_LargeValues
void test_competing_risks_large_values() {
    DataFrame df;
    std::vector<ExactData> data;
    for (int i = 0; i < 10; ++i) data.emplace_back(2000 + i, 1e8 + i * 1e6);
    df.set_exact_series(ExactSeries(data));

    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    ValidationResult valid = model.validate();
    CHECK_TRUE(valid.is_valid);
}

// Test_CompetingRisks_SmallValues
void test_competing_risks_small_values() {
    DataFrame df;
    std::vector<ExactData> data;
    for (int i = 0; i < 10; ++i) data.emplace_back(2000 + i, 0.001 + i * 0.0001);
    df.set_exact_series(ExactSeries(data));

    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal};
    CompetingRisksModel model(std::move(df), types);

    ValidationResult valid = model.validate();
    CHECK_TRUE(valid.is_valid);
}

// Test_CompetingRisks_WithIntervalData
void test_competing_risks_with_interval_data() {
    DataFrame df = create_interval_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Gumbel};
    CompetingRisksModel model(std::move(df), types);

    std::vector<double> parameters = parameter_values(model);
    double log_lh = model.log_likelihood(parameters);

    CHECK_TRUE(!std::isnan(log_lh));
}

// ===========================================================================================
// ISimulatable / GenerateRandomValues (no upstream CompetingRisksModelTests method exercises
// the stream itself; this pins the ported guards + determinism of the C# CompetingRisks
// per-component min/max sampler the model delegates to -- the M10 precedent).
// ===========================================================================================

void test_generate_random_values_guards_and_deterministic_seed() {
    DataFrame df = create_sample_data_frame();
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Gumbel};
    CompetingRisksModel model(std::move(df), types);

    CHECK_THROWS(model.generate_random_values(0, 42));   // sampleSize must be positive
    CHECK_THROWS(model.generate_random_values(-5, 42));  // sampleSize must be positive

    CompetingRisksModel no_dist;
    CHECK_THROWS(no_dist.generate_random_values(8, 42));  // null CompetingRisks

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
    test_constructor_empty_constructor_creates_model();
    test_constructor_with_distribution_types_sets_components();
    test_constructor_with_competing_risks_distribution_clones_distribution();
    test_constructor_empty_distribution_types_throws();
    test_constructor_too_many_distributions_throws();

    // Property tests.
    test_competing_risks_set_and_get();
    test_use_single_quantile_always_returns_true();
    test_data_frame_set_triggers_parameter_update();

    // SetDefaultParameters tests.
    test_set_default_parameters_creates_parameters_for_all_components();
    test_set_default_parameters_owner_names_are_set();
    test_set_default_parameters_uniform_priors();
    test_set_default_parameters_three_components();

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

    // PointwiseDataLogLikelihoodComponents tests.
    test_pointwise_data_log_likelihood_components_returns_all_data_points();
    test_pointwise_data_log_likelihood_components_has_correct_types();

    // PointwisePriorLogLikelihood tests.
    test_pointwise_prior_log_likelihood_returns_parameter_priors();
    test_pointwise_prior_log_likelihood_with_jeffreys_prior();
    test_pointwise_prior_log_likelihood_with_quantile_prior();

    // Clone tests.
    test_clone_creates_independent_copy();
    test_clone_preserves_quantile_priors();
    test_clone_parameters_are_independent();
    test_clone_preserves_jeffreys_rule_setting();

    // Validation tests.
    test_validate_valid_model_returns_true();
    test_validate_null_data_frame_returns_false();
    test_validate_null_competing_risks_returns_false();
    test_validate_too_many_components_returns_false();
    test_validate_single_component_returns_true();
    test_validate_three_components_returns_true();

    // SetParameterValues tests.
    test_set_parameter_values_updates_parameters();
    test_set_parameter_values_updates_distribution();
    test_set_parameter_values_wrong_count_throws();

    // Engineering application tests.
    test_competing_risks_snowmelt_vs_rainfall();
    test_competing_risks_with_hurricanes();
    test_competing_risks_three_seasons();

    // Jeffreys prior tests.
    test_jeffreys_prior_affects_prior_log_likelihood();
    test_jeffreys_prior_gamma_distribution_uses_first_parameter();
    test_jeffreys_prior_weibull_uses_first_parameter();
    test_jeffreys_prior_one_parameter_component_omits_scale_contribution();
    test_jeffreys_prior_mixed_components_applies_only_available_scale_contribution();
    test_jeffreys_prior_nonpositive_scale_returns_negative_infinity();

    // Edge cases.
    test_competing_risks_single_data_point();
    test_competing_risks_large_values();
    test_competing_risks_small_values();
    test_competing_risks_with_interval_data();

    // ISimulatable guards + seeded determinism.
    test_generate_random_values_guards_and_deterministic_seed();

    return chtest::summary("competing_risks_model");
}
