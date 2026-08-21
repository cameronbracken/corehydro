// cpp11 glue exposing the user-facing Analyses layer (A10) of the shared C++ core to R:
// UnivariateAnalysis (Bayesian frequency curve), FittingAnalysis (multi-distribution GoF
// ranking), and Bulletin17CAnalysis (B17C flood-frequency + Cohn-style CIs). These are a
// corehydro-additions binding surface (no direct C# counterpart -- they wrap the ported analysis
// classes); each analysis is inherently STATEFUL, so this file exposes ONE run function per
// analysis that builds the model via the shared spec builder (corehydro/models/model_spec.hpp),
// runs the analysis once, and packs the full result surface into a named list. The exported R
// wrappers live in R/analysis.R; the model-neutral fixture harness (test-fixtures.R) also drives
// these same functions for the `analysis` fixture kind. Core headers are vendored under
// src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "corehydro/analyses/distribution_fitting/fitting_analysis.hpp"
#include "corehydro/analyses/support/analysis_runner.hpp"
#include "corehydro/analyses/time_series/ar_analysis.hpp"
#include "corehydro/analyses/time_series/arima_analysis.hpp"
#include "corehydro/analyses/time_series/arimax_analysis.hpp"
#include "corehydro/analyses/time_series/ma_analysis.hpp"
#include "corehydro/analyses/univariate/bulletin17c_analysis.hpp"
#include "corehydro/analyses/univariate/competing_risk_analysis.hpp"
#include "corehydro/analyses/univariate/mixture_analysis.hpp"
#include "corehydro/analyses/univariate/point_process_analysis.hpp"
#include "corehydro/analyses/univariate/univariate_analysis.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/data_frame/data_collections/exact_series.hpp"
#include "corehydro/models/model_spec.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"

namespace analyses = corehydro::analyses;
namespace models = corehydro::models;
namespace est = corehydro::estimation;
using UDT = corehydro::numerics::distributions::UnivariateDistributionType;
using namespace cpp11;

// The BayesianAnalysis sampler knob (shared with estimation.cpp's parse_sampler_type).
static est::SamplerType parse_analysis_sampler(const std::string& s) {
    if (s == "DEMCz") return est::SamplerType::DEMCz;
    if (s == "DEMCzs") return est::SamplerType::DEMCzs;
    if (s == "ARWMH") return est::SamplerType::ARWMH;
    if (s == "NUTS") return est::SamplerType::NUTS;
    stop("unknown analysis sampler '%s' (use DEMCz, DEMCzs, ARWMH, or NUTS)", s.c_str());
}

// The B17C uncertainty method knob. The two deferred (Phase 9) methods are rejected with a clear
// message rather than dispatched (the C++ analysis would throw from run() otherwise).
static analyses::UncertaintyMethod parse_uncertainty_method(const std::string& s) {
    if (s == "MultivariateNormal") return analyses::UncertaintyMethod::MultivariateNormal;
    if (s == "Bootstrap") return analyses::UncertaintyMethod::Bootstrap;
    // X8/X9: the two previously-deferred methods now ship.
    if (s == "LinkedMultivariateNormal")
        return analyses::UncertaintyMethod::LinkedMultivariateNormal;
    if (s == "BiasCorrectedBootstrap")
        return analyses::UncertaintyMethod::BiasCorrectedBootstrap;
    stop("unknown uncertainty method '%s'", s.c_str());
}

// The C# type name (matching the distribution factory names) for a fitted candidate. Covers the
// 15 FittingAnalysis candidates; a fallback keeps the binding total (never throws on an unexpected
// type). No reverse map exists in the core, so this small local helper mirrors the factory's
// forward name->type table for exactly the candidates FittingAnalysis produces.
static std::string type_name(UDT type) {
    switch (type) {
        case UDT::Exponential: return "Exponential";
        case UDT::GammaDistribution: return "GammaDistribution";
        case UDT::GeneralizedExtremeValue: return "GeneralizedExtremeValue";
        case UDT::GeneralizedLogistic: return "GeneralizedLogistic";
        case UDT::GeneralizedNormal: return "GeneralizedNormal";
        case UDT::GeneralizedPareto: return "GeneralizedPareto";
        case UDT::Gumbel: return "Gumbel";
        case UDT::KappaFour: return "KappaFour";
        case UDT::LnNormal: return "LnNormal";
        case UDT::Logistic: return "Logistic";
        case UDT::LogNormal: return "LogNormal";
        case UDT::LogPearsonTypeIII: return "LogPearsonTypeIII";
        case UDT::Normal: return "Normal";
        case UDT::PearsonTypeIII: return "PearsonTypeIII";
        case UDT::Weibull: return "Weibull";
        default: return "Unknown";
    }
}

// Applies a caller-supplied exceedance grid to an analysis's ProbabilityOrdinates when non-empty,
// leaving the 25 standard defaults in place otherwise (mirrors the C# reset-then-add pattern).
static void set_ordinates(corehydro::numerics::data::ProbabilityOrdinates& po, const doubles& ep) {
    if (ep.size() == 0) return;
    po.clear();
    for (R_xlen_t i = 0; i < ep.size(); ++i) po.push_back(ep[i]);
}

// UnivariateAnalysis (A5): build the UnivariateDistributionModel from the spec, set the Bayesian
// knobs, run(), and return {parameters, mode_curve, mean_curve, lower_ci, upper_ci, aic, bic, dic,
// rmse}. `parameters` are the point-estimate distribution parameters (PosteriorMean by default).
// When the fit yields no results (unestimated), the curve/CI vectors come back empty and the
// scalars NaN.
[[cpp11::register]]
list ch_analysis_univariate_run_(std::string model_json, doubles dataset, std::string sampler,
                                 int iterations, int output_length, double credible_level, int seed,
                                 doubles exceedance_probabilities, int thinning_interval) {
    std::vector<double> data(dataset.begin(), dataset.end());
    std::unique_ptr<models::ModelBase> base = models::spec::build_model_from_json(model_json, data);
    auto* raw = dynamic_cast<models::UnivariateDistributionModel*>(base.get());
    if (raw == nullptr)
        stop("univariate_analysis requires a univariate_distribution model spec");
    base.release();
    std::unique_ptr<models::UnivariateDistributionModel> model(raw);

    analyses::UnivariateAnalysis analysis(std::move(model));
    set_ordinates(analysis.probability_ordinates(), exceedance_probabilities);

    est::BayesianAnalysis& ba = analysis.bayesian_analysis();
    ba.set_type(parse_analysis_sampler(sampler));
    if (credible_level > 0.0 && credible_level < 1.0) ba.set_credible_interval_width(credible_level);
    if (seed >= 0) ba.set_prng_seed(seed);
    if (output_length > 0) ba.set_output_length(output_length);
    if (iterations > 0) {
        ba.set_iterations(iterations);
        ba.set_warmup_iterations(std::max(50, iterations / 2));
    }
    // Optional explicit thinning (A11): the SetDefaultSimulationOptions default (thinning=20 for a
    // 2-parameter DEMCzs run) exposes a C#-vs-C++ divergence in the thinned population-sampler
    // stream (see docs/upstream-csharp-issues.md); passing 1 lands on the proven bit-identical path.
    if (thinning_interval > 0) ba.set_thinning_interval(thinning_interval);

    analysis.run();

    writable::doubles parameters;
    writable::doubles mode_curve, mean_curve, lower_ci, upper_ci;
    double aic = NA_REAL, bic = NA_REAL, dic = NA_REAL, rmse = NA_REAL;

    const auto* results = analysis.analysis_results();
    if (results != nullptr) {
        auto* pe = analysis.get_point_estimate_distribution();
        if (pe != nullptr) {
            std::vector<double> p = pe->get_parameters();
            parameters = writable::doubles(p.begin(), p.end());
        }
        mode_curve = writable::doubles(results->mode_curve.begin(), results->mode_curve.end());
        mean_curve = writable::doubles(results->mean_curve.begin(), results->mean_curve.end());
        std::size_t n = results->confidence_intervals.size();
        lower_ci = writable::doubles(static_cast<R_xlen_t>(n));
        upper_ci = writable::doubles(static_cast<R_xlen_t>(n));
        for (std::size_t i = 0; i < n; ++i) {
            lower_ci[static_cast<R_xlen_t>(i)] = results->confidence_intervals[i][0];
            upper_ci[static_cast<R_xlen_t>(i)] = results->confidence_intervals[i][1];
        }
        aic = results->aic;
        bic = results->bic;
        dic = results->dic;
        rmse = results->rmse;
    }

    return writable::list({
        "parameters"_nm = parameters,
        "mode_curve"_nm = mode_curve,
        "mean_curve"_nm = mean_curve,
        "lower_ci"_nm = lower_ci,
        "upper_ci"_nm = upper_ci,
        "aic"_nm = writable::doubles({aic}),
        "bic"_nm = writable::doubles({bic}),
        "dic"_nm = writable::doubles({dic}),
        "rmse"_nm = writable::doubles({rmse}),
    });
}

// FittingAnalysis (A6): fit the 15 candidate distributions by MLE over an exact-only frame
// built from `dataset`, and return the per-candidate GoF table {distribution, aic, bic, rmse,
// converged}. Ranking is left to the R caller.
[[cpp11::register]]
list ch_analysis_fit_distributions_(doubles dataset) {
    std::vector<double> data(dataset.begin(), dataset.end());
    auto df = std::make_unique<models::DataFrame>();
    df->set_exact_series(models::ExactSeries(data));
    df->calculate_plotting_positions();

    analyses::FittingAnalysis analysis(std::move(df));
    analysis.run();

    const auto& fitted = analysis.fitted_distributions();
    R_xlen_t n = static_cast<R_xlen_t>(fitted.size());
    writable::strings distribution(n);
    writable::doubles aic(n), bic(n), rmse(n);
    writable::logicals converged(n);
    for (R_xlen_t i = 0; i < n; ++i) {
        const auto& fd = fitted[static_cast<std::size_t>(i)];
        distribution[i] = fd.distribution() != nullptr ? type_name(fd.distribution()->type())
                                                       : std::string("Unknown");
        aic[i] = fd.aic();
        bic[i] = fd.bic();
        rmse[i] = fd.rmse();
        converged[i] = cpp11::r_bool(fd.fit_succeeded());
    }

    return writable::list({
        "distribution"_nm = distribution,
        "aic"_nm = aic,
        "bic"_nm = bic,
        "rmse"_nm = rmse,
        "converged"_nm = converged,
    });
}

// Bulletin17CAnalysis (A7-A9): build the B17C distribution from the spec, fit by GMM under the
// requested uncertainty method, and return the Cohn-style delta-method CI surface plus the fitted
// parameters + sandwich covariance. When the Cohn optional is empty (unestimated) the CI vectors
// come back empty.
[[cpp11::register]]
list ch_analysis_b17c_run_(std::string model_json, doubles dataset, std::string uncertainty_method,
                           int output_length, int seed, double confidence_level,
                           doubles exceedance_probabilities) {
    std::vector<double> data(dataset.begin(), dataset.end());
    std::unique_ptr<models::Bulletin17CDistribution> model =
        models::spec::build_bulletin17c_from_json(model_json, data);

    analyses::Bulletin17CAnalysis analysis(std::move(model));
    analysis.set_uncertainty_method(parse_uncertainty_method(uncertainty_method));
    set_ordinates(analysis.probability_ordinates(), exceedance_probabilities);

    est::BayesianAnalysis& ba = analysis.bayesian_analysis();
    if (confidence_level > 0.0 && confidence_level < 1.0)
        ba.set_credible_interval_width(confidence_level);
    if (seed >= 0) ba.set_prng_seed(seed);
    if (output_length > 0) ba.set_output_length(output_length);

    analysis.run();

    writable::doubles exceedance, point_estimates, lower_ci, upper_ci, beta1, nu, quantile_variance;
    double conf = NA_REAL;
    auto ci = analysis.compute_cohn_style_confidence_intervals();
    if (ci.has_value()) {
        exceedance = writable::doubles(ci->exceedance_probabilities.begin(),
                                       ci->exceedance_probabilities.end());
        point_estimates = writable::doubles(ci->point_estimates.begin(), ci->point_estimates.end());
        lower_ci = writable::doubles(ci->lower_ci.begin(), ci->lower_ci.end());
        upper_ci = writable::doubles(ci->upper_ci.begin(), ci->upper_ci.end());
        beta1 = writable::doubles(ci->beta1.begin(), ci->beta1.end());
        nu = writable::doubles(ci->nu.begin(), ci->nu.end());
        quantile_variance =
            writable::doubles(ci->quantile_variance.begin(), ci->quantile_variance.end());
        conf = ci->confidence_level;
    }

    // Fitted parameters + sandwich covariance from the GMM (populated after a successful run()).
    // The covariance is returned as a flat row-major vector plus its dimension so the R wrapper
    // can reshape it to a p x p matrix (avoids a 0x0 matrix in the unestimated case).
    writable::doubles parameters;
    writable::doubles covariance;
    int cov_dim = 0;
    if (analysis.gmm() != nullptr && analysis.gmm()->is_estimated()) {
        const std::vector<double>& best = analysis.gmm()->best_parameter_set().values;
        parameters = writable::doubles(best.begin(), best.end());
        int p = static_cast<int>(best.size());
        cov_dim = p;
        auto cov = analysis.gmm()->get_covariance_matrix();
        covariance = writable::doubles(static_cast<R_xlen_t>(p) * p);
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j) covariance[i * p + j] = cov(i, j);
    }

    // T20: the ensemble frequency curves off AnalysisResults -- the ONLY B17C output that depends
    // on the uncertainty method (the Cohn surface above is computed off the RNG-free GMM point
    // estimate alone), so the pivotal-bootstrap fixture needs them here. Empty when the ensemble
    // was not published (an aborted / degraded uncertainty run).
    writable::doubles mode_curve, mean_curve;
    if (analysis.analysis_results() != nullptr) {
        const auto& ar = *analysis.analysis_results();
        mode_curve = writable::doubles(ar.mode_curve.begin(), ar.mode_curve.end());
        mean_curve = writable::doubles(ar.mean_curve.begin(), ar.mean_curve.end());
    }

    // T19: BootstrapDiagnostics, populated only when the uncertainty method actually ran a
    // bootstrap arm (Bootstrap / BiasCorrectedBootstrap).
    const auto* boot = analysis.bootstrap_results();
    writable::list bootstrap = writable::list({
        "has_results"_nm = writable::logicals({cpp11::r_bool(boot != nullptr)}),
        "total_replicates"_nm = writable::integers({boot != nullptr ? boot->total_replicates() : 0}),
        "attempted_replicates"_nm =
            writable::integers({boot != nullptr ? boot->attempted_replicates() : 0}),
        "failed_replicates"_nm = writable::integers({boot != nullptr ? boot->failed_replicates() : 0}),
        "valid_replicates"_nm = writable::integers({boot != nullptr ? boot->valid_replicates() : 0}),
        "retained_replicates"_nm =
            writable::integers({boot != nullptr ? boot->retained_replicates() : 0}),
        "failure_rate"_nm = writable::doubles({boot != nullptr ? boot->failure_rate() : NA_REAL}),
        "total_retries"_nm = writable::integers({boot != nullptr ? boot->total_retries() : 0}),
        "average_retries"_nm =
            writable::doubles({boot != nullptr ? boot->average_retries() : NA_REAL}),
        "pivot_rejections"_nm =
            writable::integers({boot != nullptr ? boot->pivot_rejections() : 0}),
        "mahalanobis_rejections"_nm =
            writable::integers({boot != nullptr ? boot->mahalanobis_rejections() : 0}),
        "transform_failures"_nm =
            writable::integers({boot != nullptr ? boot->transform_failures() : 0}),
        "status_success_count"_nm =
            writable::integers({boot != nullptr ? boot->status_success_count() : 0}),
        "status_max_iterations_count"_nm =
            writable::integers({boot != nullptr ? boot->status_maximum_iterations_count() : 0}),
        "status_max_function_evaluations_count"_nm = writable::integers(
            {boot != nullptr ? boot->status_maximum_function_evaluations_count() : 0}),
        "status_failure_count"_nm =
            writable::integers({boot != nullptr ? boot->status_failure_count() : 0}),
        "status_none_count"_nm =
            writable::integers({boot != nullptr ? boot->status_none_count() : 0}),
        "optimizer_fallbacks"_nm =
            writable::integers({boot != nullptr ? boot->optimizer_fallbacks() : 0}),
    });

    return writable::list({
        "exceedance_probabilities"_nm = exceedance,
        "point_estimates"_nm = point_estimates,
        "lower_ci"_nm = lower_ci,
        "upper_ci"_nm = upper_ci,
        "confidence_level"_nm = writable::doubles({conf}),
        "beta1"_nm = beta1,
        "nu"_nm = nu,
        "quantile_variance"_nm = quantile_variance,
        "parameters"_nm = parameters,
        "covariance"_nm = covariance,
        "covariance_dim"_nm = writable::integers({cov_dim}),
        "mode_curve"_nm = mode_curve,
        "mean_curve"_nm = mean_curve,
        "bootstrap"_nm = bootstrap,
    });
}

// --- D5: per-family analyses + diagnostics ------------------------------------------------
//
// The seven remaining per-family analyses (D1 Mixture/PointProcess/CompetingRisk, D2 AR/MA/
// ARIMA/ARIMAX) share the UnivariateAnalysis result surface, so ONE dispatch function serves
// all seven (selected by `analysis_type`); the exported R wrappers stay one-per-analysis so the
// user API reads the same as univariate_analysis. Spec assembly + seed plumbing mirror
// ch_analysis_univariate_run_ byte-for-byte; the pybind11 twin (analysis_family_run) is identical.

// The flat UncertaintyAnalysisResults surface every family analysis fills.
struct FamilyRunResult {
    std::vector<double> parameters, mode_curve, mean_curve, lower_ci, upper_ci;
    double aic = NA_REAL, bic = NA_REAL, dic = NA_REAL, rmse = NA_REAL;
};

static void apply_family_bayes_knobs(est::BayesianAnalysis& ba, const std::string& sampler,
                                     int iterations, int output_length, double credible_level,
                                     int seed, int thinning_interval) {
    ba.set_type(parse_analysis_sampler(sampler));
    if (credible_level > 0.0 && credible_level < 1.0) ba.set_credible_interval_width(credible_level);
    if (seed >= 0) ba.set_prng_seed(seed);
    if (output_length > 0) ba.set_output_length(output_length);
    if (iterations > 0) {
        ba.set_iterations(iterations);
        ba.set_warmup_iterations(std::max(50, iterations / 2));
    }
    if (thinning_interval > 0) ba.set_thinning_interval(thinning_interval);
}

// Mixture / CompetingRisk / PointProcess: same shape as UnivariateAnalysis (point-estimate
// distribution + frequency curve on the exceedance grid).
template <typename AnalysisT, typename ModelT>
static FamilyRunResult run_univariate_family(std::unique_ptr<models::ModelBase> base,
                                             const std::string& sampler, int iterations,
                                             int output_length, double credible_level, int seed,
                                             const doubles& ep, int thinning_interval) {
    auto* raw = dynamic_cast<ModelT*>(base.get());
    if (raw == nullptr) stop("analysis model spec does not match the requested analysis type");
    base.release();
    std::unique_ptr<ModelT> model(raw);
    AnalysisT analysis(std::move(model));
    set_ordinates(analysis.probability_ordinates(), ep);
    apply_family_bayes_knobs(analysis.bayesian_analysis(), sampler, iterations, output_length,
                             credible_level, seed, thinning_interval);
    analysis.run();

    FamilyRunResult r;
    const auto* results = analysis.analysis_results();
    if (results != nullptr) {
        auto* pe = analysis.get_point_estimate_distribution();
        if (pe != nullptr) r.parameters = pe->get_parameters();
        r.mode_curve = results->mode_curve;
        r.mean_curve = results->mean_curve;
        for (const auto& ci : results->confidence_intervals) {
            r.lower_ci.push_back(ci[0]);
            r.upper_ci.push_back(ci[1]);
        }
        r.aic = results->aic;
        r.bic = results->bic;
        r.dic = results->dic;
        r.rmse = results->rmse;
    }
    return r;
}

// AR / MA / ARIMA / ARIMAX: forecast curves + posterior point estimate (the time-series analyses
// expose no distribution accessor, so parameters come from the BayesianAnalysis posterior).
template <typename AnalysisT, typename ModelT>
static FamilyRunResult run_time_series_family(std::unique_ptr<models::ModelBase> base,
                                              const std::string& sampler, int iterations,
                                              int output_length, double credible_level, int seed,
                                              int thinning_interval, int training_time_steps,
                                              int forecasting_time_steps) {
    auto* raw = dynamic_cast<ModelT*>(base.get());
    if (raw == nullptr) stop("time_series model spec does not match the requested analysis type");
    base.release();
    std::unique_ptr<ModelT> model(raw);
    if (training_time_steps > 0) {
        model->set_use_default_training_steps(false);
        model->set_training_time_steps(training_time_steps);
    }
    AnalysisT analysis(std::move(model));
    if (forecasting_time_steps >= 0) analysis.set_forecasting_time_steps(forecasting_time_steps);
    apply_family_bayes_knobs(analysis.bayesian_analysis(), sampler, iterations, output_length,
                             credible_level, seed, thinning_interval);
    analysis.run();

    FamilyRunResult r;
    const auto* results = analysis.analysis_results();
    if (results != nullptr) {
        const auto& ba = analysis.bayesian_analysis();
        if (ba.results()) {
            r.parameters = ba.point_estimator() == est::PointEstimateType::PosteriorMean
                               ? ba.results()->posterior_mean.values
                               : ba.results()->map.values;
        }
        r.mode_curve = results->mode_curve;
        r.mean_curve = results->mean_curve;
        for (const auto& ci : results->confidence_intervals) {
            r.lower_ci.push_back(ci[0]);
            r.upper_ci.push_back(ci[1]);
        }
        r.aic = results->aic;
        r.bic = results->bic;
        r.dic = results->dic;
        r.rmse = results->rmse;
    }
    return r;
}

static list pack_family_result(const FamilyRunResult& r) {
    return writable::list({
        "parameters"_nm = writable::doubles(r.parameters.begin(), r.parameters.end()),
        "mode_curve"_nm = writable::doubles(r.mode_curve.begin(), r.mode_curve.end()),
        "mean_curve"_nm = writable::doubles(r.mean_curve.begin(), r.mean_curve.end()),
        "lower_ci"_nm = writable::doubles(r.lower_ci.begin(), r.lower_ci.end()),
        "upper_ci"_nm = writable::doubles(r.upper_ci.begin(), r.upper_ci.end()),
        "aic"_nm = writable::doubles({r.aic}),
        "bic"_nm = writable::doubles({r.bic}),
        "dic"_nm = writable::doubles({r.dic}),
        "rmse"_nm = writable::doubles({r.rmse}),
    });
}

// The single dispatch entry for all seven per-family analyses. `analysis_type` selects the
// analysis (mixture / competing_risk / point_process / ar / ma / arima / arimax); the model_json
// carries the matching model spec. Returns the same named list as ch_analysis_univariate_run_.
[[cpp11::register]]
list ch_analysis_family_run_(std::string analysis_type, std::string model_json, doubles dataset,
                             std::string sampler, int iterations, int output_length,
                             double credible_level, int seed, doubles exceedance_probabilities,
                             int thinning_interval, int training_time_steps,
                             int forecasting_time_steps) {
    std::vector<double> data(dataset.begin(), dataset.end());
    auto base = models::spec::build_model_from_json(model_json, data);

    if (analysis_type == "mixture")
        return pack_family_result(
            run_univariate_family<analyses::MixtureAnalysis, models::MixtureModel>(
                std::move(base), sampler, iterations, output_length, credible_level, seed,
                exceedance_probabilities, thinning_interval));
    if (analysis_type == "competing_risk")
        return pack_family_result(
            run_univariate_family<analyses::CompetingRiskAnalysis, models::CompetingRisksModel>(
                std::move(base), sampler, iterations, output_length, credible_level, seed,
                exceedance_probabilities, thinning_interval));
    if (analysis_type == "point_process")
        return pack_family_result(
            run_univariate_family<analyses::PointProcessAnalysis, models::PointProcessModel>(
                std::move(base), sampler, iterations, output_length, credible_level, seed,
                exceedance_probabilities, thinning_interval));
    if (analysis_type == "ar")
        return pack_family_result(
            run_time_series_family<analyses::ARAnalysis, models::AutoRegressive>(
                std::move(base), sampler, iterations, output_length, credible_level, seed,
                thinning_interval, training_time_steps, forecasting_time_steps));
    if (analysis_type == "ma")
        return pack_family_result(
            run_time_series_family<analyses::MAAnalysis, models::MovingAverage>(
                std::move(base), sampler, iterations, output_length, credible_level, seed,
                thinning_interval, training_time_steps, forecasting_time_steps));
    if (analysis_type == "arima")
        return pack_family_result(
            run_time_series_family<analyses::ARIMAAnalysis, models::ARIMA>(
                std::move(base), sampler, iterations, output_length, credible_level, seed,
                thinning_interval, training_time_steps, forecasting_time_steps));
    if (analysis_type == "arimax")
        return pack_family_result(
            run_time_series_family<analyses::ARIMAXAnalysis, models::ARIMAX>(
                std::move(base), sampler, iterations, output_length, credible_level, seed,
                thinning_interval, training_time_steps, forecasting_time_steps));
    stop("unknown analysis_type '%s'", analysis_type.c_str());
}

// Diagnostics accessor (D3/D4): builds a UnivariateDistributionModel, runs a BayesianAnalysis, and
// computes the three diagnostics off that fit (the estimator that owns all three
// compute_*_diagnostics methods -- see bayesian_analysis.hpp). Returns one named list with a
// leverage / influence / prior_influence sub-list each.
[[cpp11::register]]
list ch_analysis_diagnostics_run_(std::string model_json, doubles dataset, std::string sampler,
                                  int iterations, int output_length, int seed, int thinning_interval,
                                  int thin_every) {
    std::vector<double> data(dataset.begin(), dataset.end());
    auto base = models::spec::build_model_from_json(model_json, data);
    models::ModelBase& model = *base;

    est::BayesianAnalysis ba(model);
    ba.set_use_simulation_defaults(false);
    ba.set_use_advanced_simulation_defaults(false);
    apply_family_bayes_knobs(ba, sampler, iterations, output_length, 0.0, seed, thinning_interval);
    if (!ba.estimate()) stop("BayesianAnalysis::estimate() failed for the diagnostics fit");

    auto lev = ba.compute_leverage_diagnostics();
    R_xlen_t nobs = static_cast<R_xlen_t>(lev.observations().size());
    writable::integers lev_index(nobs);
    writable::doubles lev_leverage(nobs), lev_fit(nobs), lev_var(nobs), lev_value(nobs);
    for (R_xlen_t i = 0; i < nobs; ++i) {
        const auto& o = lev.observations()[static_cast<std::size_t>(i)];
        lev_index[i] = o.index();
        lev_leverage[i] = o.leverage();
        lev_fit[i] = o.fit_influence();
        lev_var[i] = o.variance_influence();
        lev_value[i] = o.value();
    }
    writable::doubles lev_prior(static_cast<R_xlen_t>(lev.prior_components().size()));
    for (std::size_t i = 0; i < lev.prior_components().size(); ++i)
        lev_prior[static_cast<R_xlen_t>(i)] = lev.prior_components()[i].leverage();

    list leverage = writable::list({
        "index"_nm = lev_index,
        "leverage"_nm = lev_leverage,
        "fit_influence"_nm = lev_fit,
        "variance_influence"_nm = lev_var,
        "value"_nm = lev_value,
        "prior_leverage"_nm = lev_prior,
        "total_leverage"_nm = writable::doubles({lev.total_leverage()}),
        "total_fit_influence"_nm = writable::doubles({lev.total_fit_influence()}),
        "total_variance_influence"_nm = writable::doubles({lev.total_variance_influence()}),
    });

    auto inf = ba.compute_influence_diagnostics();
    R_xlen_t ninf = static_cast<R_xlen_t>(inf.observations().size());
    writable::doubles inf_k(ninf), inf_elpd(ninf);
    for (R_xlen_t i = 0; i < ninf; ++i) {
        const auto& o = inf.observations()[static_cast<std::size_t>(i)];
        inf_k[i] = o.pareto_k();
        inf_elpd[i] = o.elpd_loo();
    }
    list influence = writable::list({
        "pareto_k"_nm = inf_k,
        "elpd_loo"_nm = inf_elpd,
        "count"_nm = writable::integers({inf.count()}),
        "mean_pareto_k"_nm = writable::doubles({inf.mean_pareto_k()}),
        "max_pareto_k"_nm = writable::doubles({inf.max_pareto_k()}),
        "count_pareto_k_above_05"_nm = writable::integers({inf.count_pareto_k_above_05()}),
        "count_pareto_k_above_07"_nm = writable::integers({inf.count_pareto_k_above_07()}),
        "count_pareto_k_above_10"_nm = writable::integers({inf.count_pareto_k_above_10()}),
        "proportion_problematic"_nm = writable::doubles({inf.proportion_problematic()}),
        "is_reliable"_nm = writable::logicals({cpp11::r_bool(inf.is_reliable())}),
    });

    auto pri = ba.compute_prior_influence_diagnostics(thin_every > 0 ? thin_every : 10);
    writable::doubles pri_share(static_cast<R_xlen_t>(pri.prior_precision_share().size()));
    for (std::size_t i = 0; i < pri.prior_precision_share().size(); ++i)
        pri_share[static_cast<R_xlen_t>(i)] = pri.prior_precision_share()[i];
    list prior_influence = writable::list({
        "count"_nm = writable::integers({pri.count()}),
        "prior_precision_share"_nm = pri_share,
        "total_prior_log_likelihood"_nm = writable::doubles({pri.total_prior_log_likelihood()}),
        "total_data_log_likelihood"_nm = writable::doubles({pri.total_data_log_likelihood()}),
        "prior_to_data_ratio"_nm = writable::doubles({pri.prior_to_data_ratio()}),
        "is_prior_influential"_nm = writable::logicals({cpp11::r_bool(pri.is_prior_influential())}),
        "mean_prior_precision_share"_nm = writable::doubles({pri.mean_prior_precision_share()}),
    });

    return writable::list({
        "leverage"_nm = leverage,
        "influence"_nm = influence,
        "prior_influence"_nm = prior_influence,
    });
}

// --- X11: the five remaining analyses + BootstrapAnalysis + predictive checks ----------------
//
// These wrap the shared corehydro::analyses::support::run_extended_analysis dispatch (a
// corehydro-addition runner header, sibling of model_spec.hpp) so ALL of CompositeAnalysis (X5),
// SpatialGEVAnalysis (X4), BivariateAnalysis (X3), CoincidentFrequencyAnalysis (X6),
// RatingCurveAnalysis (X3), BootstrapAnalysis (X7), and the PriorPredictiveCheck /
// PosteriorPredictiveCheck (X10) go through ONE construction path shared with the C++ fixture
// runner and the pybind twin (analysis_extended_run). The R wrappers (composite_analysis,
// spatial_gev_analysis, ...) assemble the construct JSON and call this. Returns the union result
// surface as a named list; only the fields the requested target populates are non-empty.
[[cpp11::register]]
list ch_analysis_extended_run_(std::string target, std::string construct_json,
                               std::string datasets_json) {
    corehydro::analyses::support::ExtendedAnalysisResult r =
        corehydro::analyses::support::run_extended_analysis(target, construct_json, datasets_json);

    auto dv = [](const std::vector<double>& v) { return writable::doubles(v.begin(), v.end()); };

    return writable::list({
        "parameters"_nm = dv(r.parameters),
        "mode_curve"_nm = dv(r.mode_curve),
        "mean_curve"_nm = dv(r.mean_curve),
        "lower_ci"_nm = dv(r.lower_ci),
        "upper_ci"_nm = dv(r.upper_ci),
        "aic"_nm = writable::doubles({r.aic}),
        "bic"_nm = writable::doubles({r.bic}),
        "dic"_nm = writable::doubles({r.dic}),
        "rmse"_nm = writable::doubles({r.rmse}),
        "z_output_values"_nm = dv(r.z_output_values),
        "site_count"_nm = writable::integers({r.site_count}),
        "site_location_mean"_nm = dv(r.site_location_mean),
        "site_location_lower"_nm = dv(r.site_location_lower),
        "site_location_upper"_nm = dv(r.site_location_upper),
        "site_scale_mean"_nm = dv(r.site_scale_mean),
        "site_scale_lower"_nm = dv(r.site_scale_lower),
        "site_scale_upper"_nm = dv(r.site_scale_upper),
        "site_shape_mean"_nm = dv(r.site_shape_mean),
        "site_shape_lower"_nm = dv(r.site_shape_lower),
        "site_shape_upper"_nm = dv(r.site_shape_upper),
        "site0_probabilities"_nm = dv(r.site0_probabilities),
        "site0_quantile_mean"_nm = dv(r.site0_quantile_mean),
        "site0_quantile_lower"_nm = dv(r.site0_quantile_lower),
        "site0_quantile_upper"_nm = dv(r.site0_quantile_upper),
        "site0_quantile_mode"_nm = dv(r.site0_quantile_mode),
        "cv_site_prediction_errors"_nm = dv(r.cv_site_prediction_errors),
        "cv_site_rmse"_nm = dv(r.cv_site_rmse),
        "cv_site_bias"_nm = dv(r.cv_site_bias),
        "cv_mae"_nm = writable::doubles({r.cv_mae}),
        "cv_rmse"_nm = writable::doubles({r.cv_rmse}),
        "cv_mean_bias"_nm = writable::doubles({r.cv_mean_bias}),
        "number_of_replicates"_nm = writable::integers({r.number_of_replicates}),
        "mean_p_value"_nm = writable::doubles({r.mean_p_value}),
        "sd_p_value"_nm = writable::doubles({r.sd_p_value}),
        "skewness_p_value"_nm = writable::doubles({r.skewness_p_value}),
        "min_p_value"_nm = writable::doubles({r.min_p_value}),
        "max_p_value"_nm = writable::doubles({r.max_p_value}),
        "has_misfit"_nm = writable::doubles({r.has_misfit}),
        "number_of_valid_draws"_nm = writable::integers({r.number_of_valid_draws}),
        "summary_mean_quantiles"_nm = dv(r.summary_mean_quantiles),
        "summary_sd_quantiles"_nm = dv(r.summary_sd_quantiles),
        "summary_min_quantiles"_nm = dv(r.summary_min_quantiles),
        "summary_max_quantiles"_nm = dv(r.summary_max_quantiles),
    });
}
