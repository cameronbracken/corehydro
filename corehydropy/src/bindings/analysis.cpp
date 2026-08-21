// pybind11 glue exposing the user-facing Analyses layer (A10) of the shared C++ core to Python:
// UnivariateAnalysis (Bayesian frequency curve), FittingAnalysis (multi-distribution GoF ranking),
// and Bulletin17CAnalysis (B17C flood-frequency + Cohn-style CIs). These are a corehydro-additions
// binding surface (no direct C# counterpart -- they wrap the ported analysis classes); each
// analysis is inherently STATEFUL, so this file exposes ONE run function per analysis that builds
// the model via the shared spec builder (corehydro/models/model_spec.hpp), runs the analysis once,
// and returns the full result surface as a dict. The signatures / spec assembly / seed plumbing
// mirror corehydror's analysis.cpp exactly, so a seeded call returns identical numbers in either
// language. Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
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
#include "bindings.hpp"

namespace py = pybind11;
namespace analyses = corehydro::analyses;
namespace models = corehydro::models;
namespace est = corehydro::estimation;
using UDT = corehydro::numerics::distributions::UnivariateDistributionType;

static est::SamplerType parse_analysis_sampler(const std::string& s) {
    if (s == "DEMCz") return est::SamplerType::DEMCz;
    if (s == "DEMCzs") return est::SamplerType::DEMCzs;
    if (s == "ARWMH") return est::SamplerType::ARWMH;
    if (s == "NUTS") return est::SamplerType::NUTS;
    throw py::value_error("unknown analysis sampler '" + s +
                          "' (use DEMCz, DEMCzs, ARWMH, or NUTS)");
}

static analyses::UncertaintyMethod parse_uncertainty_method(const std::string& s) {
    if (s == "MultivariateNormal") return analyses::UncertaintyMethod::MultivariateNormal;
    if (s == "Bootstrap") return analyses::UncertaintyMethod::Bootstrap;
    // X8/X9: the two previously-deferred methods now ship.
    if (s == "LinkedMultivariateNormal")
        return analyses::UncertaintyMethod::LinkedMultivariateNormal;
    if (s == "BiasCorrectedBootstrap")
        return analyses::UncertaintyMethod::BiasCorrectedBootstrap;
    throw py::value_error("unknown uncertainty method '" + s + "'");
}

// The C# type name for a fitted candidate (mirrors the factory's forward name->type table for the
// 15 FittingAnalysis candidates). No reverse map exists in the core; this small local helper keeps
// the binding self-contained and matches corehydror's analysis.cpp type_name.
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

static void set_ordinates(corehydro::numerics::data::ProbabilityOrdinates& po,
                          const std::vector<double>& ep) {
    if (ep.empty()) return;
    po.clear();
    for (double p : ep) po.push_back(p);
}

// --- D5: per-family analyses + diagnostics (twin of corehydror/src/analysis.cpp) -------------
//
// The seven remaining per-family analyses share the UnivariateAnalysis result surface, so ONE
// dispatch function (analysis_family_run) serves all seven; the Python wrappers stay
// one-per-analysis. Spec assembly + seed plumbing mirror analysis_univariate_run byte-for-byte.

struct FamilyRunResult {
    std::vector<double> parameters, mode_curve, mean_curve, lower_ci, upper_ci;
    double aic = std::numeric_limits<double>::quiet_NaN();
    double bic = std::numeric_limits<double>::quiet_NaN();
    double dic = std::numeric_limits<double>::quiet_NaN();
    double rmse = std::numeric_limits<double>::quiet_NaN();
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

template <typename AnalysisT, typename ModelT>
static FamilyRunResult run_univariate_family(std::unique_ptr<models::ModelBase> base,
                                             const std::string& sampler, int iterations,
                                             int output_length, double credible_level, int seed,
                                             const std::vector<double>& ep, int thinning_interval) {
    auto* raw = dynamic_cast<ModelT*>(base.get());
    if (raw == nullptr)
        throw py::value_error("analysis model spec does not match the requested analysis type");
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

template <typename AnalysisT, typename ModelT>
static FamilyRunResult run_time_series_family(std::unique_ptr<models::ModelBase> base,
                                              const std::string& sampler, int iterations,
                                              int output_length, double credible_level, int seed,
                                              int thinning_interval, int training_time_steps,
                                              int forecasting_time_steps) {
    auto* raw = dynamic_cast<ModelT*>(base.get());
    if (raw == nullptr)
        throw py::value_error("time_series model spec does not match the requested analysis type");
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

static py::dict pack_family_result(const FamilyRunResult& r) {
    py::dict out;
    out["parameters"] = r.parameters;
    out["mode_curve"] = r.mode_curve;
    out["mean_curve"] = r.mean_curve;
    out["lower_ci"] = r.lower_ci;
    out["upper_ci"] = r.upper_ci;
    out["aic"] = r.aic;
    out["bic"] = r.bic;
    out["dic"] = r.dic;
    out["rmse"] = r.rmse;
    return out;
}

void register_analysis(py::module_& m) {
    // UnivariateAnalysis (A5): Bayesian frequency curve. Returns {parameters, mode_curve,
    // mean_curve, lower_ci, upper_ci, aic, bic, dic, rmse}. The curve/CI lists are empty and the
    // scalars NaN when the fit yields no results.
    m.def(
        "analysis_univariate_run",
        [](const std::string& model_json, const std::vector<double>& dataset,
           const std::string& sampler, int iterations, int output_length, double credible_level,
           int seed, const std::vector<double>& exceedance_probabilities, int thinning_interval) {
            std::unique_ptr<models::ModelBase> base =
                models::spec::build_model_from_json(model_json, dataset);
            auto* raw = dynamic_cast<models::UnivariateDistributionModel*>(base.get());
            if (raw == nullptr)
                throw py::value_error("univariate_analysis requires a univariate_distribution model spec");
            base.release();
            std::unique_ptr<models::UnivariateDistributionModel> model(raw);

            analyses::UnivariateAnalysis analysis(std::move(model));
            set_ordinates(analysis.probability_ordinates(), exceedance_probabilities);

            est::BayesianAnalysis& ba = analysis.bayesian_analysis();
            ba.set_type(parse_analysis_sampler(sampler));
            if (credible_level > 0.0 && credible_level < 1.0)
                ba.set_credible_interval_width(credible_level);
            if (seed >= 0) ba.set_prng_seed(seed);
            if (output_length > 0) ba.set_output_length(output_length);
            if (iterations > 0) {
                ba.set_iterations(iterations);
                ba.set_warmup_iterations(std::max(50, iterations / 2));
            }
            // Optional explicit thinning (A11): the default thinning=20 for a 2-parameter DEMCzs
            // run exposes a C#-vs-C++ divergence in the thinned population-sampler stream (see
            // docs/upstream-csharp-issues.md); passing 1 lands on the proven bit-identical path.
            if (thinning_interval > 0) ba.set_thinning_interval(thinning_interval);

            analysis.run();

            const double kNaN = std::numeric_limits<double>::quiet_NaN();
            py::dict out;
            std::vector<double> parameters, mode_curve, mean_curve, lower_ci, upper_ci;
            double aic = kNaN, bic = kNaN, dic = kNaN, rmse = kNaN;

            const auto* results = analysis.analysis_results();
            if (results != nullptr) {
                auto* pe = analysis.get_point_estimate_distribution();
                if (pe != nullptr) parameters = pe->get_parameters();
                mode_curve = results->mode_curve;
                mean_curve = results->mean_curve;
                for (const auto& ci : results->confidence_intervals) {
                    lower_ci.push_back(ci[0]);
                    upper_ci.push_back(ci[1]);
                }
                aic = results->aic;
                bic = results->bic;
                dic = results->dic;
                rmse = results->rmse;
            }

            out["parameters"] = parameters;
            out["mode_curve"] = mode_curve;
            out["mean_curve"] = mean_curve;
            out["lower_ci"] = lower_ci;
            out["upper_ci"] = upper_ci;
            out["aic"] = aic;
            out["bic"] = bic;
            out["dic"] = dic;
            out["rmse"] = rmse;
            return out;
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("sampler"), py::arg("iterations"),
        py::arg("output_length"), py::arg("credible_level"), py::arg("seed"),
        py::arg("exceedance_probabilities"), py::arg("thinning_interval") = -1);

    // FittingAnalysis (A6): fit the 15 candidates by MLE, return the per-candidate GoF table.
    m.def(
        "analysis_fit_distributions",
        [](const std::vector<double>& dataset) {
            auto df = std::make_unique<models::DataFrame>();
            df->set_exact_series(models::ExactSeries(dataset));
            df->calculate_plotting_positions();

            analyses::FittingAnalysis analysis(std::move(df));
            analysis.run();

            const auto& fitted = analysis.fitted_distributions();
            std::vector<std::string> distribution;
            std::vector<double> aic, bic, rmse;
            std::vector<bool> converged;
            for (const auto& fd : fitted) {
                distribution.push_back(fd.distribution() != nullptr
                                           ? type_name(fd.distribution()->type())
                                           : std::string("Unknown"));
                aic.push_back(fd.aic());
                bic.push_back(fd.bic());
                rmse.push_back(fd.rmse());
                converged.push_back(fd.fit_succeeded());
            }

            py::dict out;
            out["distribution"] = distribution;
            out["aic"] = aic;
            out["bic"] = bic;
            out["rmse"] = rmse;
            out["converged"] = converged;
            return out;
        },
        py::arg("dataset"));

    // Bulletin17CAnalysis (A7-A9): GMM fit + Cohn-style delta-method CIs. Returns the CI surface
    // plus the fitted parameters and sandwich covariance (a nested p x p list; empty when
    // unestimated).
    m.def(
        "analysis_b17c_run",
        [](const std::string& model_json, const std::vector<double>& dataset,
           const std::string& uncertainty_method, int output_length, int seed,
           double confidence_level, const std::vector<double>& exceedance_probabilities) {
            std::unique_ptr<models::Bulletin17CDistribution> model =
                models::spec::build_bulletin17c_from_json(model_json, dataset);

            analyses::Bulletin17CAnalysis analysis(std::move(model));
            analysis.set_uncertainty_method(parse_uncertainty_method(uncertainty_method));
            set_ordinates(analysis.probability_ordinates(), exceedance_probabilities);

            est::BayesianAnalysis& ba = analysis.bayesian_analysis();
            if (confidence_level > 0.0 && confidence_level < 1.0)
                ba.set_credible_interval_width(confidence_level);
            if (seed >= 0) ba.set_prng_seed(seed);
            if (output_length > 0) ba.set_output_length(output_length);

            analysis.run();

            const double kNaN = std::numeric_limits<double>::quiet_NaN();
            py::dict out;
            std::vector<double> exceedance, point_estimates, lower_ci, upper_ci, beta1, nu,
                quantile_variance;
            double conf = kNaN;
            auto ci = analysis.compute_cohn_style_confidence_intervals();
            if (ci.has_value()) {
                exceedance = ci->exceedance_probabilities;
                point_estimates = ci->point_estimates;
                lower_ci = ci->lower_ci;
                upper_ci = ci->upper_ci;
                beta1 = ci->beta1;
                nu = ci->nu;
                quantile_variance = ci->quantile_variance;
                conf = ci->confidence_level;
            }

            std::vector<double> parameters;
            std::vector<std::vector<double>> covariance;
            if (analysis.gmm() != nullptr && analysis.gmm()->is_estimated()) {
                parameters = analysis.gmm()->best_parameter_set().values;
                int p = static_cast<int>(parameters.size());
                auto cov = analysis.gmm()->get_covariance_matrix();
                covariance.resize(static_cast<std::size_t>(p));
                for (int i = 0; i < p; ++i) {
                    covariance[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(p));
                    for (int j = 0; j < p; ++j)
                        covariance[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = cov(i, j);
                }
            }

            out["exceedance_probabilities"] = exceedance;
            out["point_estimates"] = point_estimates;
            out["lower_ci"] = lower_ci;
            out["upper_ci"] = upper_ci;
            out["confidence_level"] = conf;
            out["beta1"] = beta1;
            out["nu"] = nu;
            out["quantile_variance"] = quantile_variance;
            out["parameters"] = parameters;
            out["covariance"] = covariance;

            // T20: the ensemble frequency curves off AnalysisResults -- the ONLY B17C output that
            // depends on the uncertainty method (the Cohn surface above is computed off the
            // RNG-free GMM point estimate alone), so the pivotal-bootstrap fixture needs them
            // here. Empty when the ensemble was not published (an aborted / degraded run).
            std::vector<double> mode_curve, mean_curve;
            if (analysis.analysis_results() != nullptr) {
                mode_curve = analysis.analysis_results()->mode_curve;
                mean_curve = analysis.analysis_results()->mean_curve;
            }
            out["mode_curve"] = mode_curve;
            out["mean_curve"] = mean_curve;

            // T19: BootstrapDiagnostics, populated only when the uncertainty method actually ran
            // a bootstrap arm (Bootstrap / BiasCorrectedBootstrap).
            const auto* boot = analysis.bootstrap_results();
            py::dict bootstrap;
            bootstrap["has_results"] = boot != nullptr;
            bootstrap["total_replicates"] = boot != nullptr ? boot->total_replicates() : 0;
            bootstrap["attempted_replicates"] = boot != nullptr ? boot->attempted_replicates() : 0;
            bootstrap["failed_replicates"] = boot != nullptr ? boot->failed_replicates() : 0;
            bootstrap["valid_replicates"] = boot != nullptr ? boot->valid_replicates() : 0;
            bootstrap["retained_replicates"] = boot != nullptr ? boot->retained_replicates() : 0;
            bootstrap["failure_rate"] = boot != nullptr ? boot->failure_rate() : kNaN;
            bootstrap["total_retries"] = boot != nullptr ? boot->total_retries() : 0;
            bootstrap["average_retries"] = boot != nullptr ? boot->average_retries() : kNaN;
            bootstrap["pivot_rejections"] = boot != nullptr ? boot->pivot_rejections() : 0;
            bootstrap["mahalanobis_rejections"] =
                boot != nullptr ? boot->mahalanobis_rejections() : 0;
            bootstrap["transform_failures"] = boot != nullptr ? boot->transform_failures() : 0;
            bootstrap["status_success_count"] = boot != nullptr ? boot->status_success_count() : 0;
            bootstrap["status_max_iterations_count"] =
                boot != nullptr ? boot->status_maximum_iterations_count() : 0;
            bootstrap["status_max_function_evaluations_count"] =
                boot != nullptr ? boot->status_maximum_function_evaluations_count() : 0;
            bootstrap["status_failure_count"] = boot != nullptr ? boot->status_failure_count() : 0;
            bootstrap["status_none_count"] = boot != nullptr ? boot->status_none_count() : 0;
            bootstrap["optimizer_fallbacks"] = boot != nullptr ? boot->optimizer_fallbacks() : 0;
            out["bootstrap"] = bootstrap;
            return out;
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("uncertainty_method"),
        py::arg("output_length"), py::arg("seed"), py::arg("confidence_level"),
        py::arg("exceedance_probabilities"));

    // D5: single dispatch for the seven per-family analyses (mixture / competing_risk /
    // point_process / ar / ma / arima / arimax). Returns the same dict as analysis_univariate_run.
    m.def(
        "analysis_family_run",
        [](const std::string& analysis_type, const std::string& model_json,
           const std::vector<double>& dataset, const std::string& sampler, int iterations,
           int output_length, double credible_level, int seed,
           const std::vector<double>& exceedance_probabilities, int thinning_interval,
           int training_time_steps, int forecasting_time_steps) {
            auto base = models::spec::build_model_from_json(model_json, dataset);
            if (analysis_type == "mixture")
                return pack_family_result(
                    run_univariate_family<analyses::MixtureAnalysis, models::MixtureModel>(
                        std::move(base), sampler, iterations, output_length, credible_level, seed,
                        exceedance_probabilities, thinning_interval));
            if (analysis_type == "competing_risk")
                return pack_family_result(
                    run_univariate_family<analyses::CompetingRiskAnalysis,
                                          models::CompetingRisksModel>(
                        std::move(base), sampler, iterations, output_length, credible_level, seed,
                        exceedance_probabilities, thinning_interval));
            if (analysis_type == "point_process")
                return pack_family_result(
                    run_univariate_family<analyses::PointProcessAnalysis,
                                          models::PointProcessModel>(
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
            throw py::value_error("unknown analysis_type '" + analysis_type + "'");
        },
        py::arg("analysis_type"), py::arg("model_json"), py::arg("dataset"), py::arg("sampler"),
        py::arg("iterations"), py::arg("output_length"), py::arg("credible_level"), py::arg("seed"),
        py::arg("exceedance_probabilities"), py::arg("thinning_interval") = -1,
        py::arg("training_time_steps") = -1, py::arg("forecasting_time_steps") = 0);

    // D5: estimation diagnostics accessor (leverage / influence / prior influence) off a fitted
    // BayesianAnalysis (the estimator that owns all three compute_*_diagnostics methods).
    m.def(
        "analysis_diagnostics_run",
        [](const std::string& model_json, const std::vector<double>& dataset,
           const std::string& sampler, int iterations, int output_length, int seed,
           int thinning_interval, int thin_every) {
            auto base = models::spec::build_model_from_json(model_json, dataset);
            models::ModelBase& model = *base;
            est::BayesianAnalysis ba(model);
            ba.set_use_simulation_defaults(false);
            ba.set_use_advanced_simulation_defaults(false);
            apply_family_bayes_knobs(ba, sampler, iterations, output_length, 0.0, seed,
                                     thinning_interval);
            if (!ba.estimate())
                throw std::runtime_error("BayesianAnalysis::estimate() failed for the diagnostics fit");

            auto lev = ba.compute_leverage_diagnostics();
            std::vector<int> lev_index;
            std::vector<double> lev_leverage, lev_fit, lev_var, lev_value;
            for (const auto& o : lev.observations()) {
                lev_index.push_back(o.index());
                lev_leverage.push_back(o.leverage());
                lev_fit.push_back(o.fit_influence());
                lev_var.push_back(o.variance_influence());
                lev_value.push_back(o.value());
            }
            std::vector<double> lev_prior;
            for (const auto& p : lev.prior_components()) lev_prior.push_back(p.leverage());
            py::dict leverage;
            leverage["index"] = lev_index;
            leverage["leverage"] = lev_leverage;
            leverage["fit_influence"] = lev_fit;
            leverage["variance_influence"] = lev_var;
            leverage["value"] = lev_value;
            leverage["prior_leverage"] = lev_prior;
            leverage["total_leverage"] = lev.total_leverage();
            leverage["total_fit_influence"] = lev.total_fit_influence();
            leverage["total_variance_influence"] = lev.total_variance_influence();

            auto inf = ba.compute_influence_diagnostics();
            std::vector<double> inf_k, inf_elpd;
            for (const auto& o : inf.observations()) {
                inf_k.push_back(o.pareto_k());
                inf_elpd.push_back(o.elpd_loo());
            }
            py::dict influence;
            influence["pareto_k"] = inf_k;
            influence["elpd_loo"] = inf_elpd;
            influence["count"] = inf.count();
            influence["mean_pareto_k"] = inf.mean_pareto_k();
            influence["max_pareto_k"] = inf.max_pareto_k();
            influence["count_pareto_k_above_05"] = inf.count_pareto_k_above_05();
            influence["count_pareto_k_above_07"] = inf.count_pareto_k_above_07();
            influence["count_pareto_k_above_10"] = inf.count_pareto_k_above_10();
            influence["proportion_problematic"] = inf.proportion_problematic();
            influence["is_reliable"] = inf.is_reliable();

            auto pri = ba.compute_prior_influence_diagnostics(thin_every > 0 ? thin_every : 10);
            py::dict prior_influence;
            prior_influence["count"] = pri.count();
            prior_influence["prior_precision_share"] = pri.prior_precision_share();
            prior_influence["total_prior_log_likelihood"] = pri.total_prior_log_likelihood();
            prior_influence["total_data_log_likelihood"] = pri.total_data_log_likelihood();
            prior_influence["prior_to_data_ratio"] = pri.prior_to_data_ratio();
            prior_influence["is_prior_influential"] = pri.is_prior_influential();
            prior_influence["mean_prior_precision_share"] = pri.mean_prior_precision_share();

            py::dict out;
            out["leverage"] = leverage;
            out["influence"] = influence;
            out["prior_influence"] = prior_influence;
            return out;
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("sampler"), py::arg("iterations"),
        py::arg("output_length"), py::arg("seed"), py::arg("thinning_interval") = -1,
        py::arg("thin_every") = 10);

    // X11: the five remaining analyses + BootstrapAnalysis + predictive checks. Byte-for-byte twin
    // of corehydror's ch_analysis_extended_run_: wraps the shared run_extended_analysis dispatch (a
    // corehydro-addition runner header, sibling of model_spec.hpp) so CompositeAnalysis (X5),
    // SpatialGEVAnalysis (X4), BivariateAnalysis (X3), CoincidentFrequencyAnalysis (X6),
    // RatingCurveAnalysis (X3), BootstrapAnalysis (X7), and PriorPredictiveCheck /
    // PosteriorPredictiveCheck (X10) all go through ONE construction path shared across the three
    // harnesses. Returns the union result surface as a dict; only the requested target's fields are
    // non-empty.
    m.def(
        "analysis_extended_run",
        [](const std::string& target, const std::string& construct_json,
           const std::string& datasets_json) {
            corehydro::analyses::support::ExtendedAnalysisResult r =
                corehydro::analyses::support::run_extended_analysis(target, construct_json,
                                                                  datasets_json);
            py::dict out;
            out["parameters"] = r.parameters;
            out["mode_curve"] = r.mode_curve;
            out["mean_curve"] = r.mean_curve;
            out["lower_ci"] = r.lower_ci;
            out["upper_ci"] = r.upper_ci;
            out["aic"] = r.aic;
            out["bic"] = r.bic;
            out["dic"] = r.dic;
            out["rmse"] = r.rmse;
            out["z_output_values"] = r.z_output_values;
            out["site_count"] = r.site_count;
            out["site_location_mean"] = r.site_location_mean;
            out["site_location_lower"] = r.site_location_lower;
            out["site_location_upper"] = r.site_location_upper;
            out["site_scale_mean"] = r.site_scale_mean;
            out["site_scale_lower"] = r.site_scale_lower;
            out["site_scale_upper"] = r.site_scale_upper;
            out["site_shape_mean"] = r.site_shape_mean;
            out["site_shape_lower"] = r.site_shape_lower;
            out["site_shape_upper"] = r.site_shape_upper;
            out["site0_probabilities"] = r.site0_probabilities;
            out["site0_quantile_mean"] = r.site0_quantile_mean;
            out["site0_quantile_lower"] = r.site0_quantile_lower;
            out["site0_quantile_upper"] = r.site0_quantile_upper;
            out["site0_quantile_mode"] = r.site0_quantile_mode;
            out["cv_site_prediction_errors"] = r.cv_site_prediction_errors;
            out["cv_site_rmse"] = r.cv_site_rmse;
            out["cv_site_bias"] = r.cv_site_bias;
            out["cv_mae"] = r.cv_mae;
            out["cv_rmse"] = r.cv_rmse;
            out["cv_mean_bias"] = r.cv_mean_bias;
            out["number_of_replicates"] = r.number_of_replicates;
            out["mean_p_value"] = r.mean_p_value;
            out["sd_p_value"] = r.sd_p_value;
            out["skewness_p_value"] = r.skewness_p_value;
            out["min_p_value"] = r.min_p_value;
            out["max_p_value"] = r.max_p_value;
            out["has_misfit"] = r.has_misfit;
            out["number_of_valid_draws"] = r.number_of_valid_draws;
            out["summary_mean_quantiles"] = r.summary_mean_quantiles;
            out["summary_sd_quantiles"] = r.summary_sd_quantiles;
            out["summary_min_quantiles"] = r.summary_min_quantiles;
            out["summary_max_quantiles"] = r.summary_max_quantiles;
            return out;
        },
        py::arg("target"), py::arg("construct_json"), py::arg("datasets_json"));
}
