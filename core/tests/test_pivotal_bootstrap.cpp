// Standalone tests for the covariance-aware PIVOTAL bootstrap workflow on
// corehydro::numerics::sampling::Bootstrap<TData>.
//
// Transcribed 1:1 (structure, inputs and oracle values unaltered) from
//   upstream/Numerics/Test_Numerics/Sampling/Test_PivotalBootstrap.cs @ 2a0357a
// -- all 18 [TestMethod]s in C# file order. The pivotal workflow is an internal Numerics
// engine reached from the packages only through the ported analyses, so per the
// tolerance/oracle policy its oracles are transcribed here in a C++-only ctest rather than
// living in fixtures/*.json (the same standard test_bootstrap_analysis.cpp uses).
//
// Two C# helpers the test file calls are NOT ported onto their C++ classes, so -- exactly as
// test_bootstrap_analysis.cpp already does for the first of them -- they are transcribed here
// as local test helpers, formula-for-formula from the C# source:
//   * Normal.MonteCarloConfidenceIntervals (Normal.cs:733), a WPF/HEC-FDA sampling helper.
//   * Normal.ParameterCovariance (Normal.cs:800), the IStandardError member. `Normal` in this
//     port does not declare IStandardError, and adding it is a class-layout change outside
//     this task's scope; the Normal branch is two closed forms (Var(mu-hat) = sigma^2/n,
//     Var(sigma-hat) = sigma^2/(2n), zero covariance) with no other caller here.
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/chi_squared.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/functions/fisher_z_link.hpp"
#include "corehydro/numerics/functions/i_link_function.hpp"
#include "corehydro/numerics/functions/identity_link.hpp"
#include "corehydro/numerics/functions/log_link.hpp"
#include "corehydro/numerics/functions/yeo_johnson_link.hpp"
#include "corehydro/numerics/math/linalg/cholesky_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/bootstrap/bootstrap.hpp"
#include "corehydro/numerics/sampling/bootstrap/support/bootstrap_fit.hpp"
#include "corehydro/numerics/sampling/bootstrap/support/pivotal_bootstrap_invalid_draw_policy.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"
#include "check.hpp"

using corehydro::numerics::distributions::ChiSquared;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::ParameterEstimationMethod;
using corehydro::numerics::functions::FisherZLink;
using corehydro::numerics::functions::IdentityLink;
using corehydro::numerics::functions::ILinkFunction;
using corehydro::numerics::functions::LogLink;
using corehydro::numerics::functions::YeoJohnsonLink;
using corehydro::numerics::math::linalg::CholeskyDecomposition;
using corehydro::numerics::math::linalg::Matrix;
using corehydro::numerics::math::linalg::Vector;
using corehydro::numerics::math::optimization::ParameterSet;
using corehydro::numerics::sampling::Bootstrap;
using corehydro::numerics::sampling::BootstrapCIMethod;
using corehydro::numerics::sampling::BootstrapFit;
using corehydro::numerics::sampling::BootstrapResults;
using corehydro::numerics::sampling::MersenneTwister;
using corehydro::numerics::sampling::PivotalBootstrapInvalidDrawPolicy;

namespace {

using Data = std::vector<double>;
using Boot = Bootstrap<Data>;
using LinkArray = std::vector<std::shared_ptr<ILinkFunction>>;

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Builds a Matrix from a row-major flat list (the C# `double[,]` literals).
Matrix mat(int n, const std::vector<double>& flat) { return Matrix(n, n, flat); }

// Test_PivotalBootstrap.cs:464 -- `Fit(double[] values, double[,] covariance)`.
BootstrapFit fit_of(const std::vector<double>& values, const Matrix& covariance) {
    return BootstrapFit(ParameterSet(values, kNaN), covariance);
}

// Test_PivotalBootstrap.cs:453 -- `CreatePivotalBootstrap(BootstrapFit parent)`.
Boot create_pivotal_bootstrap(const BootstrapFit& parent) { return Boot(Data{}, parent); }

// Local transcription of Normal.ParameterCovariance (Normal.cs:800), MoM/MLE branch. See the
// file header for why it lives here rather than on `Normal`.
Matrix normal_parameter_covariance(const Normal& dist, int sample_size) {
    double s2 = dist.sigma() * dist.sigma();
    Matrix covar(2, 2);
    covar(0, 0) = s2 / sample_size;
    covar(1, 1) = s2 / (2.0 * sample_size);
    covar(0, 1) = 0.0;
    covar(1, 0) = covar(0, 1);
    return covar;
}

// Local transcription of Normal.MonteCarloConfidenceIntervals (Normal.cs:733) -- the "same
// sampling approach as HEC-FDA" the C# test uses to build its truth intervals.
std::vector<std::array<double, 2>> monte_carlo_ci(const Normal& dist, int sample_size,
                                                  int realizations,
                                                  const std::vector<double>& quantiles,
                                                  const std::array<double, 2>& percentiles) {
    double original_mean = dist.mean();
    double original_std_dev = dist.standard_deviation();

    MersenneTwister r(12345);
    auto rnd_mean = corehydro::numerics::utilities::next_doubles(r, realizations);
    auto rnd_std_dev = corehydro::numerics::utilities::next_doubles(r, realizations);

    std::vector<Normal> mc(static_cast<std::size_t>(realizations));
    for (int idx = 0; idx < realizations; ++idx) {
        Normal mean_dist(original_mean, original_std_dev / std::sqrt(static_cast<double>(sample_size)));
        double new_mu = mean_dist.inverse_cdf(rnd_mean[static_cast<std::size_t>(idx)]);
        ChiSquared chi(sample_size - 1);
        double new_sigma = std::sqrt((sample_size - 1) * std::pow(original_std_dev, 2.0) /
                                     chi.inverse_cdf(rnd_std_dev[static_cast<std::size_t>(idx)]));
        mc[static_cast<std::size_t>(idx)] = Normal(new_mu, new_sigma);
    }

    std::vector<std::array<double, 2>> out(quantiles.size());
    for (std::size_t i = 0; i < quantiles.size(); ++i) {
        std::vector<double> x_values(static_cast<std::size_t>(realizations));
        for (int idx = 0; idx < realizations; ++idx)
            x_values[static_cast<std::size_t>(idx)] =
                mc[static_cast<std::size_t>(idx)].inverse_cdf(quantiles[i]);
        for (int j = 0; j < 2; ++j)
            out[i][static_cast<std::size_t>(j)] = corehydro::numerics::data::percentile(
                x_values, percentiles[static_cast<std::size_t>(j)]);
    }
    return out;
}

// Test_PivotalBootstrap.cs:474 -- `FitNormal(double[] sample)`.
BootstrapFit fit_normal(const Data& sample) {
    Normal distribution;
    distribution.estimate(sample, ParameterEstimationMethod::MaximumLikelihood);
    if (!distribution.parameters_valid())
        throw std::runtime_error("The normal fit produced invalid parameters.");

    return BootstrapFit(ParameterSet(distribution.get_parameters(), kNaN),
                        normal_parameter_covariance(distribution, static_cast<int>(sample.size())));
}

// Test_PivotalBootstrap.cs:513 -- `ExpectedIdentityPivotalValues(parent, raw)`.
std::vector<double> expected_identity_pivotal_values(const BootstrapFit& parent, const BootstrapFit& raw) {
    CholeskyDecomposition parent_cholesky(parent.covariance());
    CholeskyDecomposition raw_cholesky(raw.covariance());
    std::vector<double> difference(static_cast<std::size_t>(parent.parameter_count()));
    for (std::size_t i = 0; i < difference.size(); ++i)
        difference[i] = parent.parameters().values[i] - raw.parameters().values[i];

    std::vector<double> z = raw_cholesky.forward(Vector(difference)).to_array();
    std::vector<double> reinflated = (parent_cholesky.l() * Vector(z)).to_array();
    std::vector<double> expected(difference.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = parent.parameters().values[i] + reinflated[i];
    return expected;
}

// Test_PivotalBootstrap.cs:533 -- the custom `LinearScaleLink` used to prove the link factory
// accepts caller-defined ILinkFunction instances.
class LinearScaleLink final : public ILinkFunction {
   public:
    explicit LinearScaleLink(double scale) : scale_(scale) {}
    double link(double x) const override { return scale_ * x; }
    double inverse_link(double eta) const override { return eta / scale_; }
    double d_link(double) const override { return scale_; }

   private:
    double scale_;
};

// Test_PivotalBootstrap.cs:490 -- `CreateSeededNormalPivotalBootstrap()`.
Boot create_seeded_normal_pivotal_bootstrap() {
    const int sample_size = 25;
    Normal distribution(3.0, 0.7);
    BootstrapFit parent(ParameterSet(distribution.get_parameters(), kNaN),
                        normal_parameter_covariance(distribution, sample_size));
    Boot boot(Data(static_cast<std::size_t>(sample_size), 0.0), parent);
    boot.replicates = 50;
    boot.prng_seed = 8675309;
    boot.resample_function = [](const Data&, const ParameterSet& ps, MersenneTwister& rng) {
        return Normal(ps.values[0], ps.values[1]).generate_random_values(sample_size, rng.next());
    };
    boot.fit_with_covariance_function = fit_normal;
    boot.pivotal_link_factory = [](const corehydro::numerics::sampling::PivotalBootstrapContext&) {
        return LinkArray{nullptr, std::make_shared<LogLink>()};
    };
    boot.pivotal_parameter_validator = [](const std::vector<double>& values) { return values[1] > 0.0; };
    return boot;
}

// --- Tests, in C# file order ----------------------------------------------------------------

// Transform_IdentityLink_UsesBothCovariances (line 25).
void test_transform_identity_link_uses_both_covariances() {
    auto parent = fit_of({10.0, 20.0}, mat(2, {4.0, 0.0, 0.0, 9.0}));
    auto raw = fit_of({8.0, 17.0}, mat(2, {1.0, 0.0, 0.0, 1.0}));
    auto boot = create_pivotal_bootstrap(parent);
    boot.regularize_pivotal_covariances = false;

    boot.transform_pivotal_bootstrap({raw});

    CHECK_EQ(boot.bootstrap_parameter_sets().size(), std::size_t{1});
    CHECK_NEAR(boot.bootstrap_parameter_sets()[0].values[0], 14.0, 1e-12);
    CHECK_NEAR(boot.bootstrap_parameter_sets()[0].values[1], 29.0, 1e-12);
}

// Transform_FullCovariance_UsesCholeskyAlgebra (line 43).
void test_transform_full_covariance_uses_cholesky_algebra() {
    auto parent = fit_of({10.0, 20.0}, mat(2, {4.0, 1.2, 1.2, 9.0}));
    auto raw = fit_of({8.0, 17.0}, mat(2, {1.0, 0.25, 0.25, 2.0}));
    auto boot = create_pivotal_bootstrap(parent);
    boot.regularize_pivotal_covariances = false;

    boot.transform_pivotal_bootstrap({raw});

    std::vector<double> expected = expected_identity_pivotal_values(parent, raw);
    CHECK_NEAR(boot.bootstrap_parameter_sets()[0].values[0], expected[0], 1e-12);
    CHECK_NEAR(boot.bootstrap_parameter_sets()[0].values[1], expected[1], 1e-12);
}

// Transform_CustomLinkFactory_ControlsLinkDefinition (line 61).
void test_transform_custom_link_factory_controls_link_definition() {
    auto parent = fit_of({10.0}, mat(1, {4.0}));
    auto raw = fit_of({8.0}, mat(1, {1.0}));
    auto boot = create_pivotal_bootstrap(parent);
    bool factory_was_called = false;
    int context_parameter_count = -1;
    std::vector<double> context_raw_values;

    boot.regularize_pivotal_covariances = false;
    boot.pivotal_link_factory = [&](const corehydro::numerics::sampling::PivotalBootstrapContext& context) {
        factory_was_called = true;
        context_parameter_count = context.parameter_count();
        context_raw_values = context.get_raw_parameter_values(0);
        return LinkArray{std::make_shared<LinearScaleLink>(2.0)};
    };

    boot.transform_pivotal_bootstrap({raw});

    CHECK_TRUE(factory_was_called);
    CHECK_EQ(context_parameter_count, 1);
    CHECK_EQ(context_raw_values.size(), std::size_t{1});
    CHECK_NEAR(context_raw_values[0], 8.0, 0.0);
    CHECK_TRUE(dynamic_cast<const LinearScaleLink*>(boot.pivotal_links()[0].get()) != nullptr);
    CHECK_NEAR(boot.bootstrap_parameter_sets()[0].values[0], 14.0, 1e-12);
}

// Transform_LogFisherZAndYeoJohnsonLinks_RetainsFiniteDraws (line 88).
void test_transform_log_fisher_z_and_yeo_johnson_links_retains_finite_draws() {
    auto parent = fit_of({1.0, 2.0, 0.25, -0.5}, mat(4, {0.04, 0.0, 0.0, 0.0,
                                                          0.0, 0.09, 0.0, 0.0,
                                                          0.0, 0.0, 0.01, 0.0,
                                                          0.0, 0.0, 0.0, 0.16}));
    const Matrix raw_cov = mat(4, {0.01, 0.0, 0.0, 0.0,
                                   0.0, 0.04, 0.0, 0.0,
                                   0.0, 0.0, 0.0025, 0.0,
                                   0.0, 0.0, 0.0, 0.09});
    std::vector<BootstrapFit> raw_fits = {fit_of({0.9, 1.8, 0.10, -0.7}, raw_cov),
                                          fit_of({1.1, 2.1, 0.35, -0.3}, raw_cov)};
    auto boot = create_pivotal_bootstrap(parent);
    boot.regularize_pivotal_covariances = false;
    boot.pivotal_link_factory = [](const corehydro::numerics::sampling::PivotalBootstrapContext& context) {
        return LinkArray{nullptr, std::make_shared<LogLink>(), std::make_shared<FisherZLink>(),
                         std::make_shared<YeoJohnsonLink>(context.get_raw_parameter_values(3))};
    };
    boot.pivotal_parameter_validator = [](const std::vector<double>& values) {
        return values[1] > 0.0 && values[2] > -1.0 && values[2] < 1.0;
    };

    boot.transform_pivotal_bootstrap(raw_fits);

    CHECK_EQ(boot.bootstrap_parameter_sets().size(), std::size_t{2});
    CHECK_TRUE(boot.pivotal_links()[0] == nullptr);
    CHECK_TRUE(dynamic_cast<const LogLink*>(boot.pivotal_links()[1].get()) != nullptr);
    CHECK_TRUE(dynamic_cast<const FisherZLink*>(boot.pivotal_links()[2].get()) != nullptr);
    CHECK_TRUE(dynamic_cast<const YeoJohnsonLink*>(boot.pivotal_links()[3].get()) != nullptr);
    for (double value : boot.bootstrap_parameter_sets()[0].values)
        CHECK_TRUE(corehydro::numerics::is_finite(value));
}

// Transform_WithStatisticFunction_StoresRawAndPivotalStatistics (line 146).
void test_transform_with_statistic_function_stores_raw_and_pivotal_statistics() {
    auto parent = fit_of({10.0}, mat(1, {4.0}));
    std::vector<BootstrapFit> raw_fits = {
        fit_of({8.0}, mat(1, {1.0})),  fit_of({9.0}, mat(1, {1.0})), fit_of({10.0}, mat(1, {1.0})),
        fit_of({11.0}, mat(1, {1.0})), fit_of({12.0}, mat(1, {1.0}))};
    auto boot = create_pivotal_bootstrap(parent);
    boot.regularize_pivotal_covariances = false;
    boot.statistic_function = [](const ParameterSet& ps) { return std::vector<double>{ps.values[0] * 2.0}; };

    boot.transform_pivotal_bootstrap(raw_fits);
    BootstrapResults pivotal = boot.get_confidence_intervals(BootstrapCIMethod::Percentile, 0.2);
    BootstrapResults raw = boot.get_raw_pivotal_confidence_intervals(0.2);

    CHECK_EQ(pivotal.statistic_results.size(), std::size_t{1});
    CHECK_EQ(raw.statistic_results.size(), std::size_t{1});
    CHECK_NEAR(pivotal.statistic_results[0].population_estimate, 20.0, 1e-12);
    CHECK_NEAR(raw.statistic_results[0].population_estimate, 20.0, 1e-12);
    CHECK_TRUE(raw.parameter_results[0].lower_ci > pivotal.parameter_results[0].lower_ci);
}

// Transform_WithoutStatisticFunction_ProducesParameterIntervalsOnly (line 176).
void test_transform_without_statistic_function_produces_parameter_intervals_only() {
    auto parent = fit_of({10.0}, mat(1, {4.0}));
    auto raw = fit_of({8.0}, mat(1, {1.0}));
    auto boot = create_pivotal_bootstrap(parent);
    boot.regularize_pivotal_covariances = false;

    boot.transform_pivotal_bootstrap({raw});
    BootstrapResults results = boot.get_confidence_intervals(BootstrapCIMethod::Percentile);

    CHECK_EQ(results.parameter_results.size(), std::size_t{1});
    CHECK_EQ(results.statistic_results.size(), std::size_t{0});
    // C#'s `BootstrapStatistics.GetLength(1)`: this port's [replicate][statistic] nesting makes
    // that the inner row width.
    CHECK_EQ(boot.bootstrap_statistics()[0].size(), std::size_t{0});
}

// Transform_InvalidDrawPolicyDrop_DropsInvalidPivotalDraws (line 195).
void test_transform_invalid_draw_policy_drop() {
    auto parent = fit_of({10.0}, mat(1, {4.0}));
    auto raw = fit_of({8.0}, mat(1, {1.0}));
    auto boot = create_pivotal_bootstrap(parent);
    boot.regularize_pivotal_covariances = false;
    boot.pivotal_parameter_validator = [](const std::vector<double>&) { return false; };
    boot.pivotal_invalid_draw_policy = PivotalBootstrapInvalidDrawPolicy::Drop;

    boot.transform_pivotal_bootstrap({raw});

    CHECK_EQ(boot.bootstrap_parameter_sets().size(), std::size_t{0});
    CHECK_TRUE(boot.pivotal_diagnostics().has_value());
    CHECK_EQ(boot.pivotal_diagnostics()->invalid_pivotal_replicates, 1);
    CHECK_EQ(boot.pivotal_diagnostics()->retained_pivotal_replicates, 0);
}

// Transform_InvalidDrawPolicyUseRaw_UsesRawParameters (line 215).
void test_transform_invalid_draw_policy_use_raw() {
    auto parent = fit_of({10.0}, mat(1, {4.0}));
    auto raw = fit_of({8.0}, mat(1, {1.0}));
    auto boot = create_pivotal_bootstrap(parent);
    boot.regularize_pivotal_covariances = false;
    boot.pivotal_parameter_validator = [](const std::vector<double>&) { return false; };
    boot.pivotal_invalid_draw_policy = PivotalBootstrapInvalidDrawPolicy::UseRaw;

    boot.transform_pivotal_bootstrap({raw});

    CHECK_EQ(boot.bootstrap_parameter_sets().size(), std::size_t{1});
    CHECK_NEAR(boot.bootstrap_parameter_sets()[0].values[0], 8.0, 1e-12);
}

// Transform_InvalidDrawPolicyUseParent_UsesParentParameters (line 234).
void test_transform_invalid_draw_policy_use_parent() {
    auto parent = fit_of({10.0}, mat(1, {4.0}));
    auto raw = fit_of({8.0}, mat(1, {1.0}));
    auto boot = create_pivotal_bootstrap(parent);
    boot.regularize_pivotal_covariances = false;
    boot.pivotal_parameter_validator = [](const std::vector<double>&) { return false; };
    boot.pivotal_invalid_draw_policy = PivotalBootstrapInvalidDrawPolicy::UseParent;

    boot.transform_pivotal_bootstrap({raw});

    CHECK_EQ(boot.bootstrap_parameter_sets().size(), std::size_t{1});
    CHECK_NEAR(boot.bootstrap_parameter_sets()[0].values[0], 10.0, 1e-12);
}

// Transform_ReplicateFilter_RejectsRawFitsAndUpdatesDiagnostics (line 253).
void test_transform_replicate_filter_rejects_raw_fits() {
    auto parent = fit_of({10.0}, mat(1, {4.0}));
    std::vector<BootstrapFit> raw_fits = {fit_of({8.0}, mat(1, {1.0})), fit_of({9.0}, mat(1, {1.0})),
                                          fit_of({12.0}, mat(1, {1.0}))};
    auto boot = create_pivotal_bootstrap(parent);
    boot.regularize_pivotal_covariances = false;
    boot.pivotal_replicate_filter = [](const BootstrapFit& f) { return f.parameters().values[0] >= 9.0; };

    boot.transform_pivotal_bootstrap(raw_fits);

    CHECK_EQ(boot.raw_bootstrap_parameter_sets().size(), std::size_t{2});
    CHECK_EQ(boot.bootstrap_parameter_sets().size(), std::size_t{2});
    CHECK_TRUE(boot.pivotal_diagnostics().has_value());
    CHECK_EQ(boot.pivotal_diagnostics()->rejected_raw_replicates, 1);
}

// RunPivotalBootstrap_IgnoresRegularFitFunction (line 277).
void test_run_pivotal_bootstrap_ignores_regular_fit_function() {
    auto parent = fit_of({10.0}, mat(1, {4.0}));
    const std::vector<double> samples = {8.0, 9.0, 10.0, 11.0};
    auto boot = create_pivotal_bootstrap(parent);
    boot.regularize_pivotal_covariances = false;
    boot.replicates = static_cast<int>(samples.size());
    boot.max_retries = 1;
    boot.resample_function = [](const Data& data, const ParameterSet&, MersenneTwister&) { return data; };
    boot.fit_function = [](const Data&) -> ParameterSet {
        throw std::runtime_error("Regular fit should not be used by pivotal bootstrap.");
    };
    int index = -1;
    boot.fit_with_covariance_function = [&samples, &index](const Data&) {
        int next = ++index;
        return fit_of({samples[static_cast<std::size_t>(next)]}, mat(1, {1.0}));
    };

    boot.run_pivotal_bootstrap();

    CHECK_EQ(boot.raw_bootstrap_fits().size(), samples.size());
    CHECK_EQ(boot.bootstrap_parameter_sets().size(), samples.size());
}

// Run_RegularBootstrap_IgnoresPivotalOnlyProperties (line 304).
void test_run_regular_bootstrap_ignores_pivotal_only_properties() {
    Boot boot(Data{1.0}, ParameterSet(std::vector<double>{10.0}, kNaN));
    boot.replicates = 3;
    boot.max_retries = 1;
    boot.resample_function = [](const Data& data, const ParameterSet&, MersenneTwister&) { return data; };
    boot.fit_function = [](const Data&) { return ParameterSet(std::vector<double>{11.0}, kNaN); };
    boot.statistic_function = [](const ParameterSet& ps) { return std::vector<double>{ps.values[0]}; };
    boot.fit_with_covariance_function = [](const Data&) -> BootstrapFit {
        throw std::runtime_error("Pivotal fit should not be used by regular bootstrap.");
    };
    boot.pivotal_link_factory = [](const corehydro::numerics::sampling::PivotalBootstrapContext&) -> LinkArray {
        throw std::runtime_error("Pivotal links should not be used by regular bootstrap.");
    };
    boot.pivotal_replicate_filter = [](const BootstrapFit&) -> bool {
        throw std::runtime_error("Pivotal filters should not be used by regular bootstrap.");
    };
    boot.original_covariance = mat(1, {1.0});

    boot.run();

    CHECK_EQ(boot.bootstrap_parameter_sets().size(), std::size_t{3});
    CHECK_EQ(boot.raw_bootstrap_parameter_sets().size(), std::size_t{0});
}

// RunPivotalBootstrap_WithoutFitWithCovarianceFunction_Throws (line 327).
void test_run_pivotal_bootstrap_without_fit_with_covariance_function_throws() {
    auto parent = fit_of({10.0}, mat(1, {4.0}));
    auto boot = create_pivotal_bootstrap(parent);
    boot.resample_function = [](const Data& data, const ParameterSet&, MersenneTwister&) { return data; };
    boot.fit_with_covariance_function = nullptr;

    CHECK_THROWS_MSG(boot.run_pivotal_bootstrap(), "FitWithCovarianceFunction");
}

// RunPivotalBootstrap_WithoutOriginalCovariance_Throws (line 342).
void test_run_pivotal_bootstrap_without_original_covariance_throws() {
    Boot boot(Data{1.0}, ParameterSet(std::vector<double>{10.0}, kNaN));
    boot.resample_function = [](const Data& data, const ParameterSet&, MersenneTwister&) { return data; };
    boot.fit_with_covariance_function = [](const Data&) { return fit_of({10.0}, mat(1, {1.0})); };

    CHECK_THROWS_MSG(boot.run_pivotal_bootstrap(), "OriginalCovariance");
}

// GetConfidenceIntervals_BCaAfterPivotalRun_Throws (line 356).
void test_get_confidence_intervals_bca_after_pivotal_run_throws() {
    auto parent = fit_of({10.0}, mat(1, {4.0}));
    auto raw = fit_of({8.0}, mat(1, {1.0}));
    auto boot = create_pivotal_bootstrap(parent);
    boot.regularize_pivotal_covariances = false;
    boot.transform_pivotal_bootstrap({raw});

    CHECK_THROWS_MSG(boot.get_confidence_intervals(BootstrapCIMethod::BCa), "Only percentile");
}

// GetRawPivotalConfidenceIntervals_BeforePivotalRun_Throws (line 372).
void test_get_raw_pivotal_confidence_intervals_before_pivotal_run_throws() {
    auto parent = fit_of({10.0}, mat(1, {4.0}));
    auto boot = create_pivotal_bootstrap(parent);

    CHECK_THROWS(boot.get_raw_pivotal_confidence_intervals());
}

// RunPivotalBootstrap_WithSameSeed_IsReproducible (line 384).
void test_run_pivotal_bootstrap_with_same_seed_is_reproducible() {
    Boot first = create_seeded_normal_pivotal_bootstrap();
    Boot second = create_seeded_normal_pivotal_bootstrap();

    first.run_pivotal_bootstrap();
    second.run_pivotal_bootstrap();

    CHECK_EQ(second.bootstrap_parameter_sets().size(), first.bootstrap_parameter_sets().size());
    for (std::size_t i = 0; i < first.bootstrap_parameter_sets().size(); ++i) {
        CHECK_NEAR(first.bootstrap_parameter_sets()[i].values[0],
                   second.bootstrap_parameter_sets()[i].values[0], 1e-12);
        CHECK_NEAR(first.bootstrap_parameter_sets()[i].values[1],
                   second.bootstrap_parameter_sets()[i].values[1], 1e-12);
    }
}

// Run_NormalLocationScale_MatchesObjectiveBayesQuantileIntervals (line 404).
void test_run_normal_location_scale_matches_objective_bayes_quantile_intervals() {
    const int sample_size = 100;
    const int replicates = 2500;
    Normal parent_distribution(3.0, 0.7);
    BootstrapFit parent(ParameterSet(parent_distribution.get_parameters(), kNaN),
                        normal_parameter_covariance(parent_distribution, sample_size));
    Data original_data(static_cast<std::size_t>(sample_size), 0.0);
    const std::vector<double> probabilities = {0.5, 0.95};
    Boot boot(original_data, parent);

    boot.replicates = replicates;
    boot.prng_seed = 12345;
    boot.resample_function = [](const Data&, const ParameterSet& ps, MersenneTwister& rng) {
        return Normal(ps.values[0], ps.values[1]).generate_random_values(sample_size, rng.next());
    };
    boot.fit_with_covariance_function = fit_normal;
    boot.pivotal_link_factory = [](const corehydro::numerics::sampling::PivotalBootstrapContext&) {
        return LinkArray{std::make_shared<IdentityLink>(), std::make_shared<LogLink>()};
    };
    boot.pivotal_parameter_validator = [](const std::vector<double>& values) { return values[1] > 0.0; };
    boot.statistic_function = [&probabilities](const ParameterSet& ps) {
        Normal distribution(ps.values[0], ps.values[1]);
        std::vector<double> out(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i) out[i] = distribution.inverse_cdf(probabilities[i]);
        return out;
    };

    boot.run_pivotal_bootstrap();
    BootstrapResults quantile_intervals = boot.get_confidence_intervals(BootstrapCIMethod::Percentile, 0.1);
    auto objective_bayes_intervals =
        monte_carlo_ci(parent_distribution, sample_size, 12000, probabilities, {0.05, 0.95});

    CHECK_TRUE(static_cast<double>(boot.bootstrap_parameter_sets().size()) > 0.98 * replicates);
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        CHECK_NEAR(quantile_intervals.statistic_results[i].lower_ci, objective_bayes_intervals[i][0],
                   0.12 * parent_distribution.sigma());
        CHECK_NEAR(quantile_intervals.statistic_results[i].upper_ci, objective_bayes_intervals[i][1],
                   0.12 * parent_distribution.sigma());
    }
}

}  // namespace

int main() {
    test_transform_identity_link_uses_both_covariances();
    test_transform_full_covariance_uses_cholesky_algebra();
    test_transform_custom_link_factory_controls_link_definition();
    test_transform_log_fisher_z_and_yeo_johnson_links_retains_finite_draws();
    test_transform_with_statistic_function_stores_raw_and_pivotal_statistics();
    test_transform_without_statistic_function_produces_parameter_intervals_only();
    test_transform_invalid_draw_policy_drop();
    test_transform_invalid_draw_policy_use_raw();
    test_transform_invalid_draw_policy_use_parent();
    test_transform_replicate_filter_rejects_raw_fits();
    test_run_pivotal_bootstrap_ignores_regular_fit_function();
    test_run_regular_bootstrap_ignores_pivotal_only_properties();
    test_run_pivotal_bootstrap_without_fit_with_covariance_function_throws();
    test_run_pivotal_bootstrap_without_original_covariance_throws();
    test_get_confidence_intervals_bca_after_pivotal_run_throws();
    test_get_raw_pivotal_confidence_intervals_before_pivotal_run_throws();
    test_run_pivotal_bootstrap_with_same_seed_is_reproducible();
    test_run_normal_location_scale_matches_objective_bayes_quantile_intervals();
    return chtest::summary("pivotal_bootstrap");
}
