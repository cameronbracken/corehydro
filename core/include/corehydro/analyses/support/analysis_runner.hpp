// corehydro ADDITION -- no upstream C# counterpart (sibling of models/model_spec.hpp).
//
// The shared user-facing runner for the Phase-10 analyses that the Phase-8/9a binding surface
// (analysis.cpp / analysis.py) does NOT yet reach: CompositeAnalysis (X5), SpatialGEVAnalysis
// (X4), BivariateAnalysis (X3), CoincidentFrequencyAnalysis (X6), RatingCurveAnalysis (X3), the
// Numerics BootstrapAnalysis engine (X7), and the two predictive checks (X10). Like
// models/model_spec.hpp this header exists so all THREE fixture/binding harnesses (the C++
// test_fixtures runner, the cpp11 glue, the pybind11 glue) build + run the EXACT same analysis
// from ONE construction path: each caller serializes its native `construct`/`datasets` structure
// to a JSON string and calls `run_extended_analysis(target, construct_json, datasets_json)`, which
// parses with models/json_lite.hpp, builds the model via models::spec::build_model_from_json
// (which already understands spatial_gev / rating_curve / bivariate specs), drives the ported
// analysis once, and packs the full result surface into a flat ExtendedAnalysisResult. The R and
// Python packages call it through one binding each; the C++ runner calls it directly. Because all
// three run the identical compiled core with a bit-exact Mersenne Twister, a seeded call returns
// identical numbers in every language.
//
// Deterministic aggregation (CompositeAnalysis / CoincidentFrequency point-estimate path) needs no
// RNG; the MCMC-driven analyses (Univariate children, Bivariate, SpatialGEV, posterior predictive)
// carry the same seeded reproducibility the rest of the core proves.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/analyses/bivariate/bivariate_analysis.hpp"
#include "corehydro/analyses/bivariate/coincident_frequency_analysis.hpp"
#include "corehydro/analyses/rating_curve/rating_curve_analysis.hpp"
#include "corehydro/analyses/spatial_extremes/spatial_gev_analysis.hpp"
#include "corehydro/analyses/support/weighted_univariate_analysis.hpp"
#include "corehydro/analyses/univariate/composite_analysis.hpp"
#include "corehydro/analyses/univariate/univariate_analysis.hpp"
#include "corehydro/diagnostics/posterior_predictive_check.hpp"
#include "corehydro/diagnostics/prior_predictive_check.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/json_lite.hpp"
#include "corehydro/models/model_spec.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/bootstrap_analysis.hpp"

namespace corehydro::analyses::support {

// Flat result surface every extended-analysis assertion / binding reads. Only the fields the
// requested target populates are filled; every other field keeps its default (empty vector / NaN).
struct ExtendedAnalysisResult {
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    // Shared UncertaintyAnalysisResults surface (composite / spatial / bivariate / coincident /
    // rating-curve / bootstrap).
    std::vector<double> parameters, mode_curve, mean_curve, lower_ci, upper_ci;
    double aic = kNaN, bic = kNaN, dic = kNaN, rmse = kNaN;

    // Coincident frequency: the Z output bins.
    std::vector<double> z_output_values;

    // SpatialGEV site results (per site) + point-estimate quantile curve of site 0.
    int site_count = 0;
    std::vector<double> site_location_mean, site_location_lower, site_location_upper;
    std::vector<double> site_scale_mean, site_scale_lower, site_scale_upper;
    std::vector<double> site_shape_mean, site_shape_lower, site_shape_upper;
    std::vector<double> site0_probabilities;
    std::vector<double> site0_quantile_mean, site0_quantile_lower, site0_quantile_upper,
        site0_quantile_mode;
    // SpatialGEV cross-validation (populated only when construct.cross_validation is true).
    std::vector<double> cv_site_prediction_errors, cv_site_rmse, cv_site_bias, cv_site_crps;
    double cv_mae = kNaN, cv_rmse = kNaN, cv_mean_bias = kNaN;

    // Predictive checks (posterior p-values / misfit; prior summary quantiles).
    int number_of_replicates = 0;
    double mean_p_value = kNaN, sd_p_value = kNaN, skewness_p_value = kNaN, min_p_value = kNaN,
           max_p_value = kNaN;
    double has_misfit = kNaN;  // 0 / 1 (posterior); NaN when not applicable
    int number_of_valid_draws = 0;
    std::vector<double> summary_mean_quantiles, summary_sd_quantiles, summary_min_quantiles,
        summary_max_quantiles;
};

// --- Shared knob helpers (mirror the analysis.cpp glue) -------------------------------------

inline corehydro::estimation::SamplerType parse_runner_sampler(const std::string& s) {
    using ST = corehydro::estimation::SamplerType;
    if (s == "DEMCz") return ST::DEMCz;
    if (s == "DEMCzs") return ST::DEMCzs;
    if (s == "ARWMH") return ST::ARWMH;
    if (s == "NUTS") return ST::NUTS;
    throw std::runtime_error("unknown analysis sampler '" + s + "'");
}

inline corehydro::numerics::distributions::ParameterEstimationMethod parse_estimation_method(
    const std::string& s) {
    using PEM = corehydro::numerics::distributions::ParameterEstimationMethod;
    if (s == "MethodOfMoments") return PEM::MethodOfMoments;
    if (s == "MethodOfLinearMoments") return PEM::MethodOfLinearMoments;
    if (s == "MaximumLikelihood") return PEM::MaximumLikelihood;
    throw std::runtime_error("unknown parameter estimation method '" + s + "'");
}

// Applies the shared Bayesian MCMC knobs from a construct object (mirrors apply_family_bayes_knobs).
inline void apply_runner_bayes_knobs(corehydro::estimation::BayesianAnalysis& ba,
                                     const models::spec::JsonValue& construct) {
    ba.set_type(parse_runner_sampler(construct.value_or("sampler", "DEMCz")));
    if (construct.contains("credible_level")) {
        double cl = construct.at("credible_level").as_double();
        if (cl > 0.0 && cl < 1.0) ba.set_credible_interval_width(cl);
    }
    if (construct.contains("seed")) {
        int seed = construct.at("seed").as_int();
        if (seed >= 0) ba.set_prng_seed(seed);
    }
    if (construct.contains("output_length") && construct.at("output_length").as_int() > 0)
        ba.set_output_length(construct.at("output_length").as_int());
    if (construct.contains("iterations") && construct.at("iterations").as_int() > 0) {
        int it = construct.at("iterations").as_int();
        ba.set_iterations(it);
        ba.set_warmup_iterations(std::max(50, it / 2));
    }
    if (construct.contains("thinning_interval") && construct.at("thinning_interval").as_int() > 0)
        ba.set_thinning_interval(construct.at("thinning_interval").as_int());
    if (construct.contains("number_of_chains"))
        ba.set_number_of_chains(construct.at("number_of_chains").as_int());
}

inline std::vector<double> runner_ordinates(const models::spec::JsonValue& construct) {
    if (construct.contains("exceedance_probabilities"))
        return construct.at("exceedance_probabilities").as_double_vector();
    return {};
}

inline void apply_ordinates(corehydro::numerics::data::ProbabilityOrdinates& po,
                            const std::vector<double>& ep) {
    if (ep.empty()) return;
    po.clear();
    for (double p : ep) po.push_back(p);
}

// Resolves a dataset key against the harness-supplied datasets map.
inline std::vector<double> resolve_dataset(const models::spec::JsonValue& datasets,
                                           const std::string& key) {
    return datasets.at(key).as_double_vector();
}

// The plain value vector a runner needs alongside the model itself: the default sample size for a
// predictive draw, and the observed sample a posterior check compares against.
//
// A model spec that names a `dataset` resolves it from the harness map, which is every caller
// before the data layer landed. A model carrying its own inline `data_frame` (a censored record)
// supplies its exact-series values instead: the interval, threshold, and uncertain series have no
// single value to compare an observed statistic against, and the model itself reads the full frame
// through build_model regardless, so nothing censored is lost from the fit.
inline std::vector<double> resolve_model_data(const models::spec::JsonValue& model,
                                              const models::spec::JsonValue& datasets) {
    if (model.contains("dataset")) return resolve_dataset(datasets, model.at("dataset").as_string());
    if (model.contains("data_frame")) {
        std::vector<double> out;
        const models::spec::JsonValue& df = model.at("data_frame");
        if (df.contains("exact"))
            for (const models::spec::JsonValue& e : df.at("exact").items())
                out.push_back(e.at("value").as_double());
        return out;
    }
    throw std::runtime_error("analysis model spec requires either 'dataset' or 'data_frame'");
}

// Fills the shared UncertaintyAnalysisResults surface into `r`.
inline void collect_uar(const corehydro::numerics::distributions::UncertaintyAnalysisResults& res,
                        ExtendedAnalysisResult& r) {
    r.mode_curve = res.mode_curve;
    r.mean_curve = res.mean_curve;
    for (const auto& ci : res.confidence_intervals) {
        r.lower_ci.push_back(ci[0]);
        r.upper_ci.push_back(ci[1]);
    }
    r.aic = res.aic;
    r.bic = res.bic;
    r.dic = res.dic;
    r.rmse = res.rmse;
}

// --- Per-target builders --------------------------------------------------------------------

// CompositeAnalysis (X5): build + fit one UnivariateAnalysis per child family over the shared
// dataset, then aggregate deterministically. `composite_type` and `average_method` govern.
inline ExtendedAnalysisResult run_composite(const models::spec::JsonValue& construct,
                                            const models::spec::JsonValue& datasets) {
    const models::spec::JsonValue& model = construct.at("model");
    std::vector<double> data = resolve_model_data(model, datasets);

    // The children share the parent's observations. A model naming a `dataset` hands each child
    // the plain vector, which is the vector-constructor path this runner has always taken; a
    // model carrying an inline `data_frame` hands each child a clone of that frame instead, so a
    // censored record stays censored in every child fit rather than silently collapsing to its
    // exact series.
    bool censored = model.contains("data_frame");
    models::DataFrame frame;
    if (censored) frame = models::spec::build_data_frame(model.at("data_frame"));

    // One UnivariateAnalysis per child family (non-movable; held by unique_ptr).
    std::vector<std::unique_ptr<UnivariateAnalysis>> children;
    for (const models::spec::JsonValue& f : model.at("families").items()) {
        auto dist = corehydro::numerics::distributions::create_distribution(f.as_string());
        auto child_model =
            censored ? std::make_unique<models::UnivariateDistributionModel>(frame.clone(),
                                                                             std::move(dist))
                     : std::make_unique<models::UnivariateDistributionModel>(std::move(dist), data);
        auto child = std::make_unique<UnivariateAnalysis>(std::move(child_model));
        apply_ordinates(child->probability_ordinates(), runner_ordinates(construct));
        apply_runner_bayes_knobs(child->bayesian_analysis(), construct);
        child->run();
        children.push_back(std::move(child));
    }

    CompositeAnalysis composite;
    for (auto& c : children)
        composite.analyses().push_back(WeightedUnivariateAnalysis(c.get(), 0.0));

    std::string ct = construct.value_or("composite_type", "CompetingRisks");
    CompositeType composite_type = ct == "Mixture"        ? CompositeType::Mixture
                                   : ct == "ModelAverage" ? CompositeType::ModelAverage
                                                          : CompositeType::CompetingRisks;
    std::string am = construct.value_or("average_method", "AIC");
    AverageMethod average_method = am == "BIC"     ? AverageMethod::BIC
                                   : am == "DIC"   ? AverageMethod::DIC
                                   : am == "WAIC"  ? AverageMethod::WAIC
                                   : am == "LOOIC" ? AverageMethod::LOOIC
                                   : am == "Equal" ? AverageMethod::Equal
                                   : am == "RMSE"  ? AverageMethod::RMSE
                                                   : AverageMethod::AIC;
    composite.set_model_average_method(average_method);
    composite.set_composite_distribution_type(composite_type);
    apply_ordinates(composite.probability_ordinates(), runner_ordinates(construct));
    if (construct.contains("credible_level")) {
        double cl = construct.at("credible_level").as_double();
        if (cl > 0.0 && cl < 1.0) composite.bayesian_analysis().set_credible_interval_width(cl);
    }
    composite.run();

    ExtendedAnalysisResult r;
    const auto* res = composite.analysis_results();
    if (res != nullptr) {
        auto* pe = composite.get_point_estimate_distribution();
        if (pe != nullptr) r.parameters = pe->get_parameters();
        collect_uar(*res, r);
    }
    return r;
}

// SpatialGEVAnalysis (X4): fit + regional/site bands (+ optional leave-one-site-out CV).
inline ExtendedAnalysisResult run_spatial_gev(const models::spec::JsonValue& construct,
                                              const models::spec::JsonValue& datasets) {
    const models::spec::JsonValue& model = construct.at("model");
    std::vector<double> unused;  // spatial_gev carries inline at_site_data / coordinates
    auto base = models::spec::build_model(model, unused);
    auto* raw = dynamic_cast<models::spatial_extremes::SpatialGEV*>(base.get());
    if (raw == nullptr) throw std::runtime_error("spatial_gev_analysis requires a spatial_gev model");
    base.release();
    std::unique_ptr<models::spatial_extremes::SpatialGEV> sg(raw);

    SpatialGEVAnalysis analysis(std::move(sg));
    apply_ordinates(analysis.probability_ordinates(), runner_ordinates(construct));
    apply_runner_bayes_knobs(analysis.bayesian_analysis(), construct);
    analysis.run();
    if (construct.value_or("cross_validation", false)) analysis.run_cross_validation();

    ExtendedAnalysisResult r;
    const auto* res = analysis.analysis_results();
    if (res != nullptr) {
        std::vector<double> map_values = analysis.bayesian_analysis().results()->map.values;
        r.parameters = map_values;
        collect_uar(*res, r);
    }
    const auto* sites = analysis.site_results();
    if (sites != nullptr) {
        r.site_count = static_cast<int>(sites->size());
        for (const auto& sr : *sites) {
            r.site_location_mean.push_back(sr.location_mean);
            r.site_location_lower.push_back(sr.location_lower);
            r.site_location_upper.push_back(sr.location_upper);
            r.site_scale_mean.push_back(sr.scale_mean);
            r.site_scale_lower.push_back(sr.scale_lower);
            r.site_scale_upper.push_back(sr.scale_upper);
            r.site_shape_mean.push_back(sr.shape_mean);
            r.site_shape_lower.push_back(sr.shape_lower);
            r.site_shape_upper.push_back(sr.shape_upper);
        }
        if (!sites->empty()) {
            const auto& s0 = (*sites)[0];
            r.site0_probabilities = s0.probabilities;
            r.site0_quantile_mean = s0.quantile_mean;
            r.site0_quantile_lower = s0.quantile_lower;
            r.site0_quantile_upper = s0.quantile_upper;
            r.site0_quantile_mode = s0.quantile_mode;
        }
    }
    const auto* cv = analysis.cross_validation_results();
    if (cv != nullptr) {
        r.cv_site_prediction_errors = cv->site_prediction_errors;
        r.cv_site_rmse = cv->site_rmse;
        r.cv_site_bias = cv->site_bias;
        r.cv_site_crps = cv->site_crps;
        r.cv_mae = cv->mean_absolute_error;
        r.cv_rmse = cv->root_mean_square_error;
        r.cv_mean_bias = cv->mean_bias;
    }
    return r;
}

// Builds + fits a BivariateAnalysis from a `bivariate` model spec (owns the model).
inline std::unique_ptr<BivariateAnalysis> build_and_fit_bivariate(const models::spec::JsonValue& construct,
                                                                  bool set_ordinates_from_xy) {
    const models::spec::JsonValue& model = construct.at("model");
    std::vector<double> unused;
    auto base = models::spec::build_model(model, unused);
    auto* raw = dynamic_cast<models::BivariateDistribution*>(base.get());
    if (raw == nullptr) throw std::runtime_error("bivariate_analysis requires a bivariate model");
    base.release();
    std::unique_ptr<models::BivariateDistribution> bd(raw);

    auto analysis = std::make_unique<BivariateAnalysis>(std::move(bd));
    if (set_ordinates_from_xy && construct.contains("xy_x") && construct.contains("xy_y")) {
        std::vector<double> xs = construct.at("xy_x").as_double_vector();
        std::vector<double> ys = construct.at("xy_y").as_double_vector();
        std::vector<BivariateAnalysis::XYOrdinate> ord;
        for (std::size_t i = 0; i < xs.size() && i < ys.size(); ++i)
            ord.push_back({xs[i], ys[i]});
        if (!ord.empty()) analysis->set_xy_ordinates(std::move(ord));
    }
    apply_runner_bayes_knobs(analysis->bayesian_analysis(), construct);
    analysis->run();
    return analysis;
}

// BivariateAnalysis (X3): AND-joint-exceedance mode/mean curve + credible band over the XY grid.
inline ExtendedAnalysisResult run_bivariate(const models::spec::JsonValue& construct,
                                            const models::spec::JsonValue& /*datasets*/) {
    auto analysis = build_and_fit_bivariate(construct, /*set_ordinates_from_xy=*/true);
    ExtendedAnalysisResult r;
    const auto* res = analysis->analysis_results();
    if (res != nullptr) {
        if (analysis->bayesian_analysis().results())
            r.parameters = analysis->bayesian_analysis().results()->map.values;
        collect_uar(*res, r);
    }
    return r;
}

// CoincidentFrequencyAnalysis (X6): consumes a fitted bivariate + XY ordinates + response grid.
inline ExtendedAnalysisResult run_coincident(const models::spec::JsonValue& construct,
                                             const models::spec::JsonValue& /*datasets*/) {
    auto biv = build_and_fit_bivariate(construct, /*set_ordinates_from_xy=*/false);

    std::vector<double> x_values = construct.at("x_values").as_double_vector();
    std::vector<double> y_values = construct.at("y_values").as_double_vector();
    int rows = construct.at("response_rows").as_int();
    int cols = construct.at("response_cols").as_int();
    std::vector<double> flat = construct.at("response").as_double_vector();
    std::vector<std::vector<double>> response(static_cast<std::size_t>(rows),
                                              std::vector<double>(static_cast<std::size_t>(cols)));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            response[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                flat[static_cast<std::size_t>(i * cols + j)];

    bivariate::CoincidentFrequencyAnalysis cfa(biv.get(), x_values, y_values, response);
    if (construct.contains("number_of_bins"))
        cfa.set_number_of_bins(construct.at("number_of_bins").as_int());
    if (construct.contains("credible_level")) {
        double cl = construct.at("credible_level").as_double();
        if (cl > 0.0 && cl < 1.0) cfa.bayesian_analysis().credible_interval_width = cl;
    }
    cfa.run();

    ExtendedAnalysisResult r;
    const auto* res = cfa.analysis_results();
    if (res != nullptr) collect_uar(*res, r);
    const auto* zo = cfa.z_output_values();
    if (zo != nullptr) r.z_output_values = *zo;
    return r;
}

// RatingCurveAnalysis (X3): predicted-discharge mode/mean curve + credible band over stage bins.
inline ExtendedAnalysisResult run_rating_curve(const models::spec::JsonValue& construct,
                                               const models::spec::JsonValue& /*datasets*/) {
    const models::spec::JsonValue& model = construct.at("model");
    std::vector<double> unused;
    auto base = models::spec::build_model(model, unused);
    auto* raw = dynamic_cast<models::RatingCurve*>(base.get());
    if (raw == nullptr) throw std::runtime_error("rating_curve_analysis requires a rating_curve model");
    base.release();
    std::unique_ptr<models::RatingCurve> rc(raw);

    RatingCurveAnalysis analysis(std::move(rc));
    if (construct.contains("stage_bins")) {
        analysis.set_use_default_stage_bins(false);
        analysis.set_stage_bins(construct.at("stage_bins").as_int());
    }
    if (construct.contains("min_stage")) analysis.set_min_stage(construct.at("min_stage").as_double());
    if (construct.contains("max_stage")) analysis.set_max_stage(construct.at("max_stage").as_double());
    apply_runner_bayes_knobs(analysis.bayesian_analysis(), construct);
    analysis.run();

    ExtendedAnalysisResult r;
    const auto* res = analysis.analysis_results();
    if (res != nullptr) {
        if (analysis.bayesian_analysis().results())
            r.parameters = analysis.bayesian_analysis().results()->map.values;
        collect_uar(*res, r);
    }
    return r;
}

// BootstrapAnalysis (X7): parametric bootstrap CI on a fitted univariate distribution.
inline ExtendedAnalysisResult run_bootstrap(const models::spec::JsonValue& construct,
                                            const models::spec::JsonValue& datasets) {
    const models::spec::JsonValue& model = construct.at("model");
    std::vector<double> data = resolve_model_data(model, datasets);

    auto dist = corehydro::numerics::distributions::create_distribution(model.at("family").as_string());
    auto method = parse_estimation_method(construct.value_or("estimation_method", "MaximumLikelihood"));
    auto* est = dynamic_cast<corehydro::numerics::distributions::IEstimation*>(dist.get());
    if (est == nullptr) throw std::runtime_error("bootstrap_analysis distribution is not estimable");
    est->estimate(data, method);

    int sample_size = construct.value_or("sample_size", static_cast<int>(data.size()));
    int replications = construct.value_or("replications", 1000);
    int seed = construct.value_or("seed", 12345);
    double alpha = construct.value_or("alpha", 0.1);
    std::vector<double> probs = construct.at("probabilities").as_double_vector();

    corehydro::numerics::BootstrapAnalysis boot(*dist, method, sample_size, replications, seed);
    auto res = boot.estimate(probs, alpha, nullptr, /*recordParameterSets=*/false);

    ExtendedAnalysisResult r;
    r.parameters = dist->get_parameters();
    collect_uar(res, r);
    return r;
}

// PriorPredictiveCheck (X10): prior predictive summary quantiles.
inline ExtendedAnalysisResult run_prior_predictive(const models::spec::JsonValue& construct,
                                                   const models::spec::JsonValue& datasets) {
    const models::spec::JsonValue& model = construct.at("model");
    std::vector<double> data = resolve_model_data(model, datasets);
    auto base = models::spec::build_model(model, data);
    auto* udm = dynamic_cast<models::UnivariateDistributionModel*>(base.get());
    if (udm == nullptr)
        throw std::runtime_error("prior_predictive_check requires a univariate_distribution model");

    diagnostics::PriorPredictiveCheck<models::UnivariateDistributionModel> check(*udm);
    if (construct.contains("seed")) check.set_seed(construct.at("seed").as_int());
    if (construct.contains("number_of_draws"))
        check.set_number_of_draws(construct.at("number_of_draws").as_int());
    int sample_size = construct.value_or("sample_size", static_cast<int>(data.size()));
    diagnostics::PredictiveSummary summary = check.compute_summary(sample_size);

    ExtendedAnalysisResult r;
    r.number_of_valid_draws = summary.number_of_valid_draws;
    r.summary_mean_quantiles = summary.mean_quantiles;
    r.summary_sd_quantiles = summary.sd_quantiles;
    r.summary_min_quantiles = summary.min_quantiles;
    r.summary_max_quantiles = summary.max_quantiles;
    return r;
}

// PosteriorPredictiveCheck (X10): fit a quick MCMC, then the common posterior predictive p-values.
inline ExtendedAnalysisResult run_posterior_predictive(const models::spec::JsonValue& construct,
                                                       const models::spec::JsonValue& datasets) {
    const models::spec::JsonValue& model = construct.at("model");
    std::vector<double> data = resolve_model_data(model, datasets);
    auto base = models::spec::build_model(model, data);
    auto* udm = dynamic_cast<models::UnivariateDistributionModel*>(base.get());
    if (udm == nullptr)
        throw std::runtime_error("posterior_predictive_check requires a univariate_distribution model");

    // Observed data defaults to the fitting dataset.
    std::vector<double> observed =
        construct.contains("observed_dataset")
            ? resolve_dataset(datasets, construct.at("observed_dataset").as_string())
            : data;

    // Fit a quick Bayesian MCMC to obtain posterior draws.
    corehydro::estimation::BayesianAnalysis ba(*udm);
    ba.set_use_simulation_defaults(false);
    ba.set_use_advanced_simulation_defaults(false);
    apply_runner_bayes_knobs(ba, construct);
    if (!ba.estimate() || !ba.results())
        throw std::runtime_error("posterior_predictive_check MCMC fit failed");

    diagnostics::PosteriorPredictiveCheck<models::UnivariateDistributionModel> check(*udm,
                                                                                     *ba.results(),
                                                                                     observed);
    if (construct.contains("check_seed")) check.set_seed(construct.at("check_seed").as_int());
    else if (construct.contains("seed")) check.set_seed(construct.at("seed").as_int());
    int n_rep = construct.value_or("number_of_replicates", 1000);
    diagnostics::PredictiveCheckResults results = check.compute_common_p_values(n_rep);

    ExtendedAnalysisResult r;
    r.number_of_replicates = results.number_of_replicates;
    r.mean_p_value = results.mean_p_value;
    r.sd_p_value = results.sd_p_value;
    r.skewness_p_value = results.skewness_p_value;
    r.min_p_value = results.min_p_value;
    r.max_p_value = results.max_p_value;
    r.has_misfit = results.has_potential_misfit() ? 1.0 : 0.0;
    return r;
}

// The single dispatch entry every harness calls.
inline ExtendedAnalysisResult run_extended_analysis(const std::string& target,
                                                    const std::string& construct_json,
                                                    const std::string& datasets_json) {
    models::spec::JsonValue construct = models::spec::parse_json(construct_json);
    models::spec::JsonValue datasets = models::spec::parse_json(datasets_json);

    if (target == "CompositeAnalysis") return run_composite(construct, datasets);
    if (target == "SpatialGEVAnalysis") return run_spatial_gev(construct, datasets);
    if (target == "BivariateAnalysis") return run_bivariate(construct, datasets);
    if (target == "CoincidentFrequencyAnalysis") return run_coincident(construct, datasets);
    if (target == "RatingCurveAnalysis") return run_rating_curve(construct, datasets);
    if (target == "BootstrapAnalysis") return run_bootstrap(construct, datasets);
    if (target == "PriorPredictiveCheck") return run_prior_predictive(construct, datasets);
    if (target == "PosteriorPredictiveCheck") return run_posterior_predictive(construct, datasets);
    throw std::runtime_error("unknown extended analysis target: " + target);
}

}  // namespace corehydro::analyses::support
