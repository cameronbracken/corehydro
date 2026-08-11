// corehydro ADDITION -- no upstream C# counterpart (sibling of models/model_spec.hpp and
// analyses/support/analysis_runner.hpp).
//
// The single place a fit happens in this repo. Four callers drive it and none owns any fit
// logic: the cpp11 glue (corehydror/src/estimation.cpp), the pybind11 glue
// (corehydropy/src/bindings/estimation.cpp), the C++ fixture runner (core/tests/
// test_fixtures.cpp) and the dotnet oracle emitter. Each serializes its native construct to
// JSON and calls run_fit(target, construct_json, dataset), which parses with json_lite.hpp,
// builds the model with models::spec::build_model_from_json, constructs the named estimator,
// runs estimate() once, and packs the full surface into a flat FitResult. Because all four
// run the identical compiled core with a bit-exact Mersenne Twister, a seeded call returns
// identical numbers everywhere.
//
// Three deliberate differences from the pre-phase-2 fixture glue this replaced, all because
// this surface now faces users rather than only fixtures:
//   1. A model with fewer than two parameters gets a NaN covariance/SE/correlation, not the
//      silent zeros the old glue returned. C# GetCovarianceMatrix throws below two parameters.
//   2. A failed estimate() throws naming the estimator and the optimizer, instead of "failed
//      for a fixture case".
//   3. Every result carries `converged` and `status` so a caller can check rather than infer.
#pragma once
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/diagnostics/influence_diagnostics.hpp"
#include "corehydro/diagnostics/leverage_diagnostics.hpp"
#include "corehydro/diagnostics/prior_influence_diagnostics.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/estimation/generalized_method_of_moments.hpp"
#include "corehydro/estimation/maximum_a_posteriori.hpp"
#include "corehydro/estimation/maximum_likelihood.hpp"
#include "corehydro/estimation/optimization_method.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/models/json_lite.hpp"
#include "corehydro/models/model_spec.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/support/optimization_status.hpp"

namespace corehydro::estimation::support {

// Same alias MaximumLikelihood/MaximumAPosteriori themselves use for the return of
// profile_likelihood()/parameter_confidence_intervals().
using Matrix = corehydro::numerics::math::linalg::Matrix;

// Flat result surface every fit assertion and binding reads. Only the fields the requested
// target populates are filled; the rest keep their defaults (empty vector / NaN).
struct FitResult {
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    // --- common block, populated by every target ---------------------------------------
    std::string method;
    std::vector<std::string> parameter_names;
    std::vector<double> parameters;
    double log_likelihood = kNaN;
    double prior_log_likelihood = kNaN;
    double aic = kNaN;
    double bic = kNaN;
    int nobs = 0;
    // Row-major n x n; empty when hessian was not requested, all-NaN when n < 2.
    std::vector<double> covariance;
    std::vector<double> standard_errors;
    std::vector<double> correlation;
    bool converged = false;
    std::string status = "None";
    int function_evaluations = 0;
    // The model spec with the fitted values applied, so the caller can rebuild from the fit.
    std::string model_spec;

    // --- profile block, populated only when construct["profile"] is true ----------------
    // n_params * bins * 2, row-major: [parameter][bin][value, profile log-likelihood].
    std::vector<double> profile_grid;
    std::vector<double> profile_lower, profile_upper;  // n_params, profile-likelihood CIs
    int profile_bins = 0;

    // --- Bayesian block ----------------------------------------------------------------
    // Raw chains, flattened CHAIN-major: index = (chain * n_iterations + iter) * n_params + p.
    // This is the exact order ch_estimation_bayes_run_ has always returned, so the fixture
    // `chain_value [chain, iter, param]` arm is unchanged. The bindings permute it to the
    // user-facing [iteration, chain, parameter] array.
    std::vector<double> draws;
    std::vector<int> chain_dims;              // {n_chains, n_iterations, n_params}
    // The thinned posterior the analyses consume, flattened row-major, posterior_rows x n_params.
    std::vector<double> posterior;
    std::size_t posterior_rows = 0;
    std::vector<double> acceptance_rates;     // one per chain
    std::vector<double> mean_log_likelihood;  // one per iteration
    std::vector<double> map, posterior_mean;  // n_params
    std::vector<double> rhat, ess;            // n_params
    std::vector<double> summary_mean, summary_median, summary_sd,
                        summary_lower, summary_upper;  // n_params
    double dic = kNaN, waic = kNaN, waic_pd = kNaN, looic = kNaN, loo_pd = kNaN,
           looic_se = kNaN;
    std::vector<double> pareto_k;

    // --- GMM block ---------------------------------------------------------------------
    double j_stat = kNaN, j_stat_pval = kNaN;
    int gmm_iterations = 0;
    bool converged_within_tolerance = false;
    int optimizer_fallback_count = 0;
};

// Estimation diagnostics off a fit. Every field is populated where the target supports it and
// left empty where it does not: leverage/Cook's distance come from MAP and GMM, Pareto-k and
// prior influence from BayesianAnalysis.
struct FitDiagnostics {
    std::vector<double> cooks_distance;
    std::vector<double> leverage;
    std::vector<double> observation_influence;   // row-major n_obs x n_params
    std::vector<double> pareto_k;
    double max_pareto_k = FitResult::kNaN;
    std::vector<double> prior_influence;
    std::vector<std::string> prior_influence_names;
};

// Maps an OptimizationStatus to the string the bindings surface.
inline std::string status_name(numerics::math::optimization::OptimizationStatus s) {
    using S = numerics::math::optimization::OptimizationStatus;
    switch (s) {
        case S::None: return "None";
        case S::Success: return "Success";
        case S::MaximumIterationsReached: return "MaximumIterationsReached";
        case S::MaximumFunctionEvaluationsReached: return "MaximumFunctionEvaluationsReached";
        default: return "Failure";
    }
}

// Optimizer name -> OptimizationMethod. Accepts the "MLSL" alias, matching the pre-phase-2
// glue (corehydror/src/estimation.cpp's parse_optimization_method). Throws naming the value
// it could not parse.
inline OptimizationMethod parse_optimizer(const std::string& s) {
    if (s == "Brent") return OptimizationMethod::Brent;
    if (s == "BFGS") return OptimizationMethod::BFGS;
    if (s == "NelderMead") return OptimizationMethod::NelderMead;
    if (s == "Powell") return OptimizationMethod::Powell;
    if (s == "DifferentialEvolution") return OptimizationMethod::DifferentialEvolution;
    if (s == "MultilevelSingleLinkage" || s == "MLSL")
        return OptimizationMethod::MultilevelSingleLinkage;
    throw std::runtime_error("unknown optimizer '" + s + "'");
}

// Sampler name -> SamplerType. BayesianAnalysis::set_up_sampler (bayesian_analysis.hpp:429-492)
// constructs exactly these four (mirrors parse_sampler_type in corehydror/src/estimation.cpp);
// RWMH/HMC/SNIS are real MCMC samplers but BayesianAnalysis cannot build them, so the rejection
// names the public function that can: mcmc_sample().
inline SamplerType parse_sampler(const std::string& s) {
    if (s == "DEMCz") return SamplerType::DEMCz;
    if (s == "DEMCzs") return SamplerType::DEMCzs;
    if (s == "ARWMH") return SamplerType::ARWMH;
    if (s == "NUTS") return SamplerType::NUTS;
    throw std::runtime_error(
        "BayesianAnalysis cannot construct sampler '" + s +
        "'; it supports DEMCz, DEMCzs, ARWMH and NUTS. For RWMH, HMC or SNIS use mcmc_sample().");
}

// Point-estimator name -> PointEstimateType, by its C# name (BayesianAnalysis.cs's
// PointEstimateType enum: PosteriorMean, PosteriorMode -- the MAP).
inline PointEstimateType parse_point_estimator(const std::string& s) {
    if (s == "PosteriorMean") return PointEstimateType::PosteriorMean;
    if (s == "PosteriorMode") return PointEstimateType::PosteriorMode;
    throw std::runtime_error("unknown point estimator '" + s + "'");
}

// Parameter display names, falling back to p1..pn when the model leaves them empty (some
// ported models do -- see the PriorInfluenceDiagnostics note in docs/upstream-csharp-issues.md).
// Templated (not `const ModelBase&`) so it also serves Bulletin17CDistribution, which exposes
// the identical `parameters()`/`number_of_parameters()` shape but -- per model_spec.hpp's
// build_bulletin17c_model note -- does NOT derive from ModelBase.
template <typename TModel>
inline std::vector<std::string> parameter_names_of(const TModel& model) {
    std::vector<std::string> names;
    for (int i = 0; i < model.number_of_parameters(); ++i) {
        const std::string& n = model.parameters()[static_cast<std::size_t>(i)].display_name();
        names.push_back(n.empty() ? ("p" + std::to_string(i + 1)) : n);
    }
    return names;
}

// The GMM estimation-strategy knob (default Iterative, matching the C# GMM default). Mirrors
// corehydror/src/estimation.cpp's parse_gmm_strategy.
inline GeneralizedMethodOfMoments::GMMEstimationStrategy parse_gmm_strategy(const std::string& s) {
    using Strat = GeneralizedMethodOfMoments::GMMEstimationStrategy;
    if (s == "OneStep") return Strat::OneStep;
    if (s == "TwoStep") return Strat::TwoStep;
    if (s == "Iterative") return Strat::Iterative;
    throw std::runtime_error("unknown GMM estimation strategy '" + s + "'");
}

// Builds a concrete Bulletin17CDistribution (NOT a ModelBase -- the GMM ctor takes it as
// IGMMModel&; see model_spec.hpp's build_bulletin17c_model note) and fits it by GMM. Copied
// verbatim from `build_and_fit_gmm` at corehydror/src/estimation.cpp:107-120, including
// `post_process(true, true)`: BFGS + the numerical Jacobian have no RNG, so rebuilding this from
// the same construct reproduces the identical fit -- the lazy-rebuild contract
// run_fit_quantile_variance below (and ch_estimation_bic_) rely on.
inline std::unique_ptr<GeneralizedMethodOfMoments> build_and_fit_gmm(
    std::unique_ptr<corehydro::models::Bulletin17CDistribution>& model,
    const std::string& model_json, const std::vector<double>& dataset, const std::string& strategy,
    const std::string& optimizer, int max_gmm_iterations) {
    model = corehydro::models::spec::build_bulletin17c_from_json(model_json, dataset);
    auto gmm = std::make_unique<GeneralizedMethodOfMoments>(*model, parse_optimizer(optimizer));
    gmm->set_estimation_strategy(parse_gmm_strategy(strategy));
    if (max_gmm_iterations > 0) gmm->set_max_gmm_iterations(max_gmm_iterations);
    if (!gmm->estimate())
        throw std::runtime_error("GeneralizedMethodOfMoments::estimate() failed with optimizer " +
                                 optimizer);
    gmm->post_process(/*use_sandwich=*/true, /*compute_jstat=*/true);
    return gmm;
}

// Fills the common block from any estimator exposing the ML/MAP accessor set.
template <typename TEstimator>
void fill_common(FitResult& r, TEstimator& e, corehydro::models::ModelBase& model,
                 bool want_hessian) {
    int n = model.number_of_parameters();
    const std::vector<double>& best = e.best_parameter_set().values;
    r.parameters.assign(best.begin(), best.end());
    r.parameter_names = parameter_names_of(model);
    r.log_likelihood = e.maximum_log_likelihood();
    std::vector<double> values = best;
    r.prior_log_likelihood = model.prior_log_likelihood(values);
    r.nobs = static_cast<int>(model.pointwise_data_log_likelihood(best).size());
    r.aic = e.get_aic();
    r.bic = e.get_bic(r.nobs);
    r.converged = e.status() == numerics::math::optimization::OptimizationStatus::Success;
    r.status = status_name(e.status());
    r.function_evaluations = e.total_function_evaluations();

    if (!want_hessian) return;
    if (n < 2) {
        // C# GetCovarianceMatrix throws below two parameters; report NaN, never zeros.
        r.covariance.assign(static_cast<std::size_t>(n * n), FitResult::kNaN);
        r.standard_errors.assign(static_cast<std::size_t>(n), FitResult::kNaN);
        r.correlation.assign(static_cast<std::size_t>(n * n), FitResult::kNaN);
        return;
    }
    auto cov = e.get_covariance_matrix();
    auto corr = e.get_correlation_matrix();
    r.covariance.resize(static_cast<std::size_t>(n * n));
    r.correlation.resize(static_cast<std::size_t>(n * n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            r.covariance[static_cast<std::size_t>(i * n + j)] = cov(i, j);
            r.correlation[static_cast<std::size_t>(i * n + j)] = corr(i, j);
        }
    std::vector<double> se = e.get_standard_errors();
    r.standard_errors.assign(se.begin(), se.end());
}

// Profile likelihood grid + profile-likelihood confidence intervals. Costs bins * n_params
// likelihood evaluations, so it is off unless the caller asks. `profile_likelihood(bins)`
// returns one Matrix(bins, 2) per parameter with columns [parameter value, log-likelihood];
// `parameter_confidence_intervals(alpha)` returns a single Matrix(n_params, 2) with columns
// [lower bound, upper bound] (confirmed against maximum_likelihood.hpp:221-296).
template <typename TEstimator>
void fill_profile(FitResult& r, const TEstimator& e, int bins, double alpha) {
    int n = static_cast<int>(r.parameters.size());
    r.profile_bins = bins;
    std::vector<Matrix> profiles = e.profile_likelihood(bins);
    r.profile_grid.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(bins) * 2u);
    for (int p = 0; p < n; ++p)
        for (int b = 0; b < bins; ++b) {
            std::size_t base = (static_cast<std::size_t>(p) * static_cast<std::size_t>(bins) +
                                static_cast<std::size_t>(b)) * 2u;
            r.profile_grid[base] = profiles[static_cast<std::size_t>(p)](b, 0);
            r.profile_grid[base + 1] = profiles[static_cast<std::size_t>(p)](b, 1);
        }
    Matrix cis = e.parameter_confidence_intervals(alpha);
    r.profile_lower.resize(static_cast<std::size_t>(n));
    r.profile_upper.resize(static_cast<std::size_t>(n));
    for (int p = 0; p < n; ++p) {
        r.profile_lower[static_cast<std::size_t>(p)] = cis(p, 0);
        r.profile_upper[static_cast<std::size_t>(p)] = cis(p, 1);
    }
}

// Re-serializes the model spec with the fitted values appended, so a caller can rebuild the
// fitted model. Re-emits the parsed construct's model object's own entries (json_lite.hpp has
// no in-place mutation -- see its header) and adds one more key, `parameter_values`. Any
// `parameter_values` entry already present on the input construct (a starting or fixed-value
// guess) is skipped here rather than re-emitted: the fitted values below must be the sole
// `parameter_values` a caller re-parsing model_spec can read, since JsonValue::at() returns the
// first match by insertion order and would otherwise resurrect the stale pre-fit input.
inline std::string fitted_spec(const corehydro::models::spec::JsonValue& model_spec,
                               const std::vector<double>& values) {
    std::string out = "{";
    for (const auto& kv : model_spec.entries()) {
        if (kv.first == "parameter_values") continue;
        out += "\"" + corehydro::models::spec::escape_json_string(kv.first) +
               "\":" + corehydro::models::spec::to_json_string(kv.second) + ",";
    }
    out += "\"parameter_values\":[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out += ",";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", values[i]);
        out += buf;
    }
    out += "]}";
    return out;
}

// Settings cascade copied verbatim from ch_estimation_bayes_run_
// (corehydror/src/estimation.cpp:264-312): the two "use defaults" flags are turned off so the
// explicit settings below aren't clobbered, and every numeric knob is applied only when the
// construct supplies a positive value, exactly matching that `> 0` guard. Factored out of
// run_fit's BayesianAnalysis arm so run_fit_diagnostics's BayesianAnalysis arm below builds the
// identical fit from the identical construct shape -- two dispatchers computing diagnostics off
// the same fit is only meaningful if they agree on what "the same fit" means.
//
// SETTINGS-TRANSPARENCY CONTRACT (decided; do not "fix" by deriving a default here): every knob
// below is applied ONLY when the construct carries its key. An absent key means the ported
// class's own default, full stop -- this function deliberately derives nothing on the caller's
// behalf. That is deliberately narrower than ch_analysis_diagnostics_run_'s
// apply_family_bayes_knobs (corehydror/src/analysis.cpp:343-355), which derives
// `warmup_iterations = max(50, iterations / 2)` whenever `iterations > 0`. The two cascades are
// allowed to disagree because a later task delegates ch_estimation_bayes_run_ onto this exact
// path, and that delegation is only byte-identical to the 4542 pinned oracle assertions it
// backs if this function's behavior does not move out from under it; any derived default belongs
// in the user-facing R/Python verbs (which write an explicit `warmup_iterations` into the
// construct), not here.
//
// Consequence a caller must know: BayesianAnalysis's class default warmup_iterations_ is 1500.
// If the construct omits `warmup_iterations` and the caller also chose a small `iterations`
// (below ~3000), that default warmup exceeds iterations/2 and trips the sampler's own
// "warmup > iterations/2" guard (mcmc_sampler.hpp) rather than silently shrinking. Any caller
// that lets a user pick `iterations` must derive and pass a `warmup_iterations` itself -- this
// function will not do it for them.
inline void apply_bayesian_settings(BayesianAnalysis& ba,
                                    const corehydro::models::spec::JsonValue& construct) {
    ba.set_use_simulation_defaults(false);
    ba.set_use_advanced_simulation_defaults(false);
    int seed = construct.value_or("seed", -1);
    if (seed >= 0) ba.set_prng_seed(seed);
    if (construct.value_or("iterations", 0) > 0)
        ba.set_iterations(construct.value_or("iterations", 0));
    if (construct.value_or("warmup_iterations", 0) > 0)
        ba.set_warmup_iterations(construct.value_or("warmup_iterations", 0));
    if (construct.value_or("number_of_chains", 0) > 0)
        ba.set_number_of_chains(construct.value_or("number_of_chains", 0));
    if (construct.value_or("thinning_interval", 0) > 0)
        ba.set_thinning_interval(construct.value_or("thinning_interval", 0));
    if (construct.value_or("initial_iterations", 0) > 0)
        ba.set_initial_iterations(construct.value_or("initial_iterations", 0));
    if (construct.value_or("output_length", 0) > 0)
        ba.set_output_length(construct.value_or("output_length", 0));

    // Sampler knobs (doubles/int, applied only when the construct actually carries the key, so
    // an unset knob leaves the sampler's own default -- e.g. the
    // set_default_advanced_simulation_options() values -- in place).
    if (construct.contains("jump")) ba.set_jump(construct.at("jump").as_double());
    if (construct.contains("jump_threshold"))
        ba.set_jump_threshold(construct.at("jump_threshold").as_double());
    if (construct.contains("snooker_threshold"))
        ba.set_snooker_threshold(construct.at("snooker_threshold").as_double());
    if (construct.contains("noise")) ba.set_noise(construct.at("noise").as_double());
    if (construct.contains("scale")) ba.set_scale(construct.at("scale").as_double());
    if (construct.contains("beta")) ba.set_beta(construct.at("beta").as_double());
    if (construct.contains("max_tree_depth"))
        ba.set_max_tree_depth(construct.at("max_tree_depth").as_int());
    if (construct.contains("credible_interval_width"))
        ba.set_credible_interval_width(construct.at("credible_interval_width").as_double());
    if (construct.contains("point_estimator"))
        ba.set_point_estimator(parse_point_estimator(construct.at("point_estimator").as_string()));
}

inline FitResult run_fit(const std::string& target, const std::string& construct_json,
                         const std::vector<double>& dataset) {
    corehydro::models::spec::JsonValue construct =
        corehydro::models::spec::parse_json(construct_json);
    const auto& model_json = construct.at("model");
    std::string model_text = corehydro::models::spec::to_json_string(model_json);
    bool want_hessian = construct.value_or("hessian", true);
    std::string optimizer = construct.value_or("optimizer", "DifferentialEvolution");
    bool want_profile = construct.value_or("profile", false);
    int profile_bins = construct.value_or("profile_bins", 100);
    double alpha = construct.value_or("alpha", 0.1);

    if (target == "MaximumLikelihood" || target == "MaximumAPosteriori") {
        std::unique_ptr<corehydro::models::ModelBase> model =
            corehydro::models::spec::build_model_from_json(model_text, dataset);
        OptimizationMethod method = parse_optimizer(optimizer);
        FitResult r;
        r.method = target;
        if (target == "MaximumLikelihood") {
            MaximumLikelihood e(*model, method);
            e.set_compute_hessian(want_hessian);
            if (!e.estimate())
                throw std::runtime_error("MaximumLikelihood::estimate() failed with optimizer " +
                                         optimizer);
            fill_common(r, e, *model, want_hessian);
            if (want_profile) fill_profile(r, e, profile_bins, alpha);
        } else {
            MaximumAPosteriori e(*model, method);
            e.set_compute_hessian(want_hessian);
            if (!e.estimate())
                throw std::runtime_error("MaximumAPosteriori::estimate() failed with optimizer " +
                                         optimizer);
            fill_common(r, e, *model, want_hessian);
            if (want_profile) fill_profile(r, e, profile_bins, alpha);
        }
        r.model_spec = fitted_spec(model_json, r.parameters);
        return r;
    }
    if (target == "BayesianAnalysis") {
        std::unique_ptr<corehydro::models::ModelBase> model =
            corehydro::models::spec::build_model_from_json(model_text, dataset);
        BayesianAnalysis ba(*model, parse_sampler(construct.value_or("sampler", "DEMCzs")));
        apply_bayesian_settings(ba, construct);

        if (!ba.estimate())
            throw std::runtime_error("BayesianAnalysis::estimate() failed for sampler " +
                                     construct.value_or("sampler", "DEMCzs"));

        FitResult r;
        r.method = "BayesianAnalysis";
        r.parameter_names = parameter_names_of(*model);

        const auto& point = ba.point_estimate();
        r.parameters.assign(point.values.begin(), point.values.end());
        std::vector<double> values = point.values;
        r.nobs = static_cast<int>(model->pointwise_data_log_likelihood(values).size());
        // BayesianAnalysis exposes no get_aic()/get_bic() of its own (unlike MLE/MAP), so these
        // are recomputed here from the point estimate with the same -2*logL + 2k / -2*logL +
        // k*log(n) definitions GoodnessOfFit::aic/bic implement for every other estimator.
        // `point.fitness` is the POSTERIOR (data + prior) log-likelihood the sampler tracked for
        // MCMC ParameterSets (unlike the ML/MAP optimizers' negated-fitness convention -- see
        // mcmc_sampler.hpp's mean_log_likelihood_ accumulation), matching MAP's own
        // maximum_log_likelihood(), which is likewise the log-posterior at its optimum -- so aic/
        // bic stay comparable across the MaximumLikelihood/MaximumAPosteriori/BayesianAnalysis
        // targets.
        r.log_likelihood = point.fitness;
        r.prior_log_likelihood = model->prior_log_likelihood(values);
        int n_params = model->number_of_parameters();
        r.aic = corehydro::numerics::data::GoodnessOfFit::aic(n_params, r.log_likelihood);
        r.bic = corehydro::numerics::data::GoodnessOfFit::bic(r.nobs, n_params, r.log_likelihood);

        const auto& results = *ba.results();
        int n_chains = static_cast<int>(results.markov_chains.size());
        int n_iterations = n_chains > 0 ? static_cast<int>(results.markov_chains[0].size()) : 0;
        r.chain_dims = {n_chains, n_iterations, n_params};
        r.draws.resize(static_cast<std::size_t>(n_chains) * static_cast<std::size_t>(n_iterations) *
                       static_cast<std::size_t>(n_params));
        std::size_t k = 0;
        for (int c = 0; c < n_chains; ++c)
            for (int it = 0; it < n_iterations; ++it)
                for (int p = 0; p < n_params; ++p)
                    r.draws[k++] = results.markov_chains[static_cast<std::size_t>(c)]
                                       [static_cast<std::size_t>(it)]
                                       .values[static_cast<std::size_t>(p)];

        r.posterior_rows = results.output.size();
        r.posterior.resize(r.posterior_rows * static_cast<std::size_t>(n_params));
        for (std::size_t row = 0; row < r.posterior_rows; ++row)
            for (int p = 0; p < n_params; ++p)
                r.posterior[row * static_cast<std::size_t>(n_params) + static_cast<std::size_t>(p)] =
                    results.output[row].values[static_cast<std::size_t>(p)];

        r.acceptance_rates = results.acceptance_rates;
        r.mean_log_likelihood = results.mean_log_likelihood;
        r.map.assign(results.map.values.begin(), results.map.values.end());
        r.posterior_mean.assign(results.posterior_mean.values.begin(),
                                results.posterior_mean.values.end());

        r.rhat.resize(static_cast<std::size_t>(n_params));
        r.ess.resize(static_cast<std::size_t>(n_params));
        r.summary_mean.resize(static_cast<std::size_t>(n_params));
        r.summary_median.resize(static_cast<std::size_t>(n_params));
        r.summary_sd.resize(static_cast<std::size_t>(n_params));
        r.summary_lower.resize(static_cast<std::size_t>(n_params));
        r.summary_upper.resize(static_cast<std::size_t>(n_params));
        for (int p = 0; p < n_params; ++p) {
            const auto& stats =
                results.parameter_results[static_cast<std::size_t>(p)].summary_statistics;
            r.rhat[static_cast<std::size_t>(p)] = stats.rhat;
            r.ess[static_cast<std::size_t>(p)] = stats.ess;
            r.summary_mean[static_cast<std::size_t>(p)] = stats.mean;
            r.summary_median[static_cast<std::size_t>(p)] = stats.median;
            r.summary_sd[static_cast<std::size_t>(p)] = stats.standard_deviation;
            r.summary_lower[static_cast<std::size_t>(p)] = stats.lower_ci;
            r.summary_upper[static_cast<std::size_t>(p)] = stats.upper_ci;
        }

        r.dic = ba.dic();
        r.waic = ba.waic();
        r.waic_pd = ba.waic_pd();
        r.looic = ba.looic();
        r.loo_pd = ba.loo_pd();
        r.looic_se = ba.looic_se();
        r.pareto_k = ba.pareto_k();

        r.converged = true;
        r.status = "Success";
        r.model_spec = fitted_spec(model_json, r.parameters);
        return r;
    }
    if (target == "GMM") {
        // GMM fits a Bulletin17CDistribution -- not a ModelBase (see build_and_fit_gmm's header
        // note) -- so this cannot reuse build_model_from_json; guard the model type first, before
        // ever touching the dedicated bulletin17c construction path.
        std::string model_type = model_json.value_or("type", "univariate_distribution");
        if (model_type != "bulletin17c")
            throw std::runtime_error("GMM fits a bulletin17c model only; got type '" + model_type +
                                     "'");

        std::string strategy = construct.value_or("strategy", "Iterative");
        int max_gmm_iterations = construct.value_or("max_gmm_iterations", 0);
        std::unique_ptr<corehydro::models::Bulletin17CDistribution> model;
        std::unique_ptr<GeneralizedMethodOfMoments> gmm =
            build_and_fit_gmm(model, model_text, dataset, strategy, optimizer, max_gmm_iterations);
        int p = model->number_of_parameters();

        FitResult r;
        r.method = "GMM";
        r.parameter_names = parameter_names_of(*model);
        const std::vector<double>& best = gmm->best_parameter_set().values;
        r.parameters.assign(best.begin(), best.end());
        r.standard_errors = gmm->get_standard_errors();

        auto cov = gmm->get_covariance_matrix();
        auto corr = gmm->get_correlation_matrix();
        r.covariance.resize(static_cast<std::size_t>(p * p));
        r.correlation.resize(static_cast<std::size_t>(p * p));
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j) {
                r.covariance[static_cast<std::size_t>(i * p + j)] = cov(i, j);
                r.correlation[static_cast<std::size_t>(i * p + j)] = corr(i, j);
            }

        r.j_stat = gmm->jstat();
        r.j_stat_pval = gmm->jstat_pval();
        r.gmm_iterations = gmm->gmm_iterations();
        r.converged_within_tolerance = gmm->converged_within_tolerance();
        r.optimizer_fallback_count = gmm->optimizer_fallback_count();
        r.converged = true;
        r.status = "Success";
        r.model_spec = fitted_spec(model_json, r.parameters);
        return r;
    }
    throw std::runtime_error("unknown fit target: " + target);
}

// `quantile_variance [aep]`: like `bic [n]`, the AEP is only known at assertion-dispatch time, so
// this rebuilds the same deterministic GMM fit and evaluates the B17C delta-method Var(Q_p) live,
// exactly as ch_estimation_gmm_qvar_ does (corehydror/src/estimation.cpp:457-465). `aep` is the
// annual EXCEEDANCE probability; the C# QuantileVariance takes a NON-exceedance probability, so
// this passes 1 - aep.
inline double run_fit_quantile_variance(const std::string& construct_json,
                                        const std::vector<double>& dataset, double aep) {
    corehydro::models::spec::JsonValue construct =
        corehydro::models::spec::parse_json(construct_json);
    const auto& model_json = construct.at("model");
    std::string model_text = corehydro::models::spec::to_json_string(model_json);
    std::string optimizer = construct.value_or("optimizer", "DifferentialEvolution");
    std::string strategy = construct.value_or("strategy", "Iterative");
    int max_gmm_iterations = construct.value_or("max_gmm_iterations", 0);

    std::unique_ptr<corehydro::models::Bulletin17CDistribution> model;
    std::unique_ptr<GeneralizedMethodOfMoments> gmm =
        build_and_fit_gmm(model, model_text, dataset, strategy, optimizer, max_gmm_iterations);
    return model->quantile_variance(1.0 - aep, gmm->best_parameter_set().values,
                                    gmm->get_covariance_matrix().to_array());
}

// Estimation diagnostics off a fresh fit of `target` (MaximumAPosteriori, BayesianAnalysis, or
// GMM -- the three estimators the Diagnostics layer is wired onto). Populated from the
// already-un-stubbed estimator methods, matching ch_analysis_diagnostics_run_
// (corehydror/src/analysis.cpp:507-588) field-for-field for the BayesianAnalysis case (built
// through the shared apply_bayesian_settings, so the two dispatchers cannot silently diverge) and
// MaximumAPosteriori::{get_cooks_distance, get_observation_influence,
// compute_leverage_diagnostics} / the GeneralizedMethodOfMoments diagnostics quartet for the
// other two. Its BayesianAnalysis arm builds through apply_bayesian_settings and therefore
// inherits that function's settings-transparency contract verbatim: an absent
// `warmup_iterations` in `construct_json` leaves the 1500 class default, so a caller that lets a
// user choose a small `iterations` must pass an explicit `warmup_iterations` or trip the
// sampler's own guard (see apply_bayesian_settings's header comment).
inline FitDiagnostics run_fit_diagnostics(const std::string& target,
                                          const std::string& construct_json,
                                          const std::vector<double>& dataset) {
    corehydro::models::spec::JsonValue construct =
        corehydro::models::spec::parse_json(construct_json);
    const auto& model_json = construct.at("model");
    std::string model_text = corehydro::models::spec::to_json_string(model_json);
    std::string optimizer = construct.value_or("optimizer", "DifferentialEvolution");

    // Row-major flatten of an [n x p] Matrix, shared by the MAP and GMM observation-influence
    // arms below.
    auto flatten_influence = [](const Matrix& m) {
        int n = m.number_of_rows(), p = m.number_of_columns();
        std::vector<double> out(static_cast<std::size_t>(n) * static_cast<std::size_t>(p));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < p; ++j)
                out[static_cast<std::size_t>(i) * static_cast<std::size_t>(p) +
                    static_cast<std::size_t>(j)] = m(i, j);
        return out;
    };

    if (target == "MaximumAPosteriori") {
        std::unique_ptr<corehydro::models::ModelBase> model =
            corehydro::models::spec::build_model_from_json(model_text, dataset);
        MaximumAPosteriori e(*model, parse_optimizer(optimizer));
        // get_observation_influence/get_cooks_distance both need the posterior Hessian.
        e.set_compute_hessian(true);
        if (!e.estimate())
            throw std::runtime_error("MaximumAPosteriori::estimate() failed with optimizer " +
                                     optimizer);

        FitDiagnostics d;
        d.cooks_distance = e.get_cooks_distance();
        d.observation_influence = flatten_influence(e.get_observation_influence());
        // Bind the LeverageDiagnostics temporary to a named variable before iterating:
        // `for (const auto& o : e.compute_leverage_diagnostics().observations())` would bind the
        // loop range to a reference returned from a temporary that is destroyed at the end of the
        // full expression (before C++23's P2718R0 range-for lifetime-extension fix), an immediate
        // dangling reference.
        corehydro::diagnostics::LeverageDiagnostics lev = e.compute_leverage_diagnostics();
        for (const auto& o : lev.observations()) d.leverage.push_back(o.leverage());
        return d;
    }
    if (target == "BayesianAnalysis") {
        std::unique_ptr<corehydro::models::ModelBase> model =
            corehydro::models::spec::build_model_from_json(model_text, dataset);
        BayesianAnalysis ba(*model, parse_sampler(construct.value_or("sampler", "DEMCzs")));
        apply_bayesian_settings(ba, construct);
        if (!ba.estimate())
            throw std::runtime_error("BayesianAnalysis::estimate() failed for sampler " +
                                     construct.value_or("sampler", "DEMCzs"));

        FitDiagnostics d;
        corehydro::diagnostics::LeverageDiagnostics lev = ba.compute_leverage_diagnostics();
        for (const auto& o : lev.observations()) d.leverage.push_back(o.leverage());

        auto inf = ba.compute_influence_diagnostics();
        for (const auto& o : inf.observations()) d.pareto_k.push_back(o.pareto_k());
        d.max_pareto_k = inf.max_pareto_k();

        // thin_every mirrors ch_analysis_diagnostics_run_'s own default of 10.
        int thin_every = construct.value_or("thin_every", 10);
        auto pri = ba.compute_prior_influence_diagnostics(thin_every);
        d.prior_influence = pri.prior_precision_share();
        d.prior_influence_names = parameter_names_of(*model);
        return d;
    }
    if (target == "GMM") {
        std::string model_type = model_json.value_or("type", "univariate_distribution");
        if (model_type != "bulletin17c")
            throw std::runtime_error("GMM fits a bulletin17c model only; got type '" + model_type +
                                     "'");
        std::string strategy = construct.value_or("strategy", "Iterative");
        int max_gmm_iterations = construct.value_or("max_gmm_iterations", 0);
        std::unique_ptr<corehydro::models::Bulletin17CDistribution> model;
        std::unique_ptr<GeneralizedMethodOfMoments> gmm =
            build_and_fit_gmm(model, model_text, dataset, strategy, optimizer, max_gmm_iterations);

        FitDiagnostics d;
        d.cooks_distance = gmm->get_cooks_distance();
        d.observation_influence = flatten_influence(gmm->get_observation_influence());
        corehydro::diagnostics::LeverageDiagnostics lev = gmm->get_leverage_diagnostics();
        for (const auto& o : lev.observations()) d.leverage.push_back(o.leverage());
        return d;
    }
    throw std::runtime_error("unknown fit_diagnostics target: " + target);
}

}  // namespace corehydro::estimation::support
