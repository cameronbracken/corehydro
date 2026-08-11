// cpp11 glue exposing the estimation surface (MaximumLikelihood, MaximumAPosteriori,
// BayesianAnalysis, and -- as of M13 -- the seeded ISimulatable draw) of the shared C++ core
// to R. Like `mcmc_sampler`/`bootstrap` fixtures, `model_estimation` fixtures are inherently
// STATEFUL -- one model construct + a single estimate() run backs every assertion in a case
// (see fixtures/README.md's model_estimation schema) -- so this file exposes ONE function per
// construct shape: `ch_estimation_run_` for MaximumLikelihood/MaximumAPosteriori (shared
// {target, model_json, dataset, optimizer} signature), `ch_estimation_bayes_run_` (T12) for
// BayesianAnalysis (a disjoint {model_json, dataset, sampler, settings...} signature -- a
// sampler type + numeric knobs, not an optimizer string), and `ch_model_simulate_` (M13) for
// the estimator-less Simulation target. Each builds the model, runs its one stateful call,
// and returns every value test-fixtures.R's dispatcher needs. Core headers are vendored under
// src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
//
// M13 MODEL CONSTRUCTION: the flat Phase 4 `family` string became the serialized
// `construct.model` JSON object (`model_json`), parsed and built by the SHARED spec builder
// (corehydro/models/model_spec.hpp) -- the same code path the C++ test runner and the pybind11
// glue call, so all three harnesses construct byte-identical models (UnivariateDistribution
// incl. censored DataFrames + nonstationary trends, Mixture, CompetingRisks, PointProcess).
// The runner re-serializes the parsed fixture spec with jsonlite::toJSON(digits = I(17)),
// which round-trips doubles exactly. `dataset` stays a separate flat argument: the file-level
// `datasets` map is resolved R-side, exactly like every other fixture kind.
//
// `bic` DESIGN NOTE: unlike every other wired ML/MAP method, C# `GetBIC(sampleSize)` takes an
// actual sample size, not a 0-based index. Every other method's value is precomputed once here
// (in `ch_estimation_run_`, matching `ch_mcmc_run_`/`ch_bootstrap_run_`'s "precompute the full
// surface up front" contract), since none of them take a fixture-supplied argument. `bic` is
// the one exception: it is NOT precomputed. `ch_estimation_bic_` below rebuilds the same
// model/estimator (deterministic -- NelderMead/Brent have no randomness and
// DifferentialEvolution's default `prng_seed` is fixed, so re-running `estimate()` reproduces
// the exact same fit) and calls `e.get_bic(n)` live with whatever `n` the fixture's
// `args[0]` supplies, matching C++'s `dispatch_estimation` and the C# `GetBIC(sampleSize)`
// signature exactly. See `test-fixtures.R`'s `bic` dispatch arm.
#include <cpp11.hpp>

#include <memory>
#include <string>
#include <vector>

#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/estimation/generalized_method_of_moments.hpp"
#include "corehydro/estimation/maximum_a_posteriori.hpp"
#include "corehydro/estimation/maximum_likelihood.hpp"
#include "corehydro/estimation/optimization_method.hpp"
#include "corehydro/estimation/support/fit_runner.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/models/model_spec.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/univariate_distribution/base/univariate_distribution_model_base.hpp"
#include "corehydro/models/univariate_distribution/bulletin17c_distribution.hpp"

namespace est = corehydro::estimation;
namespace models = corehydro::models;
using namespace cpp11;

// The shared construction path (see the file header): serialized `construct.model` JSON +
// the R-resolved flat dataset -> a ModelBase through corehydro/models/model_spec.hpp.
static std::unique_ptr<models::ModelBase> build_spec_model(const std::string& model_json,
                                                           const doubles& dataset) {
    std::vector<double> data(dataset.begin(), dataset.end());
    return models::spec::build_model_from_json(model_json, data);
}

// Seeded ISimulatable draw, flattened to a 1-D vector so the `simulated_value [i]` digest works
// uniformly across model types (P3). Most Phase 4-7 models are ISimulatable<std::vector<double>>;
// BivariateDistribution is ISimulatable<Matrix2D> (n-row x 2-col), flattened ROW-MAJOR
// (i = row*2 + col) -- the same order the C++/Python glue and the README schema use.
static std::vector<double> simulate_flat(models::ModelBase* model, int sample_size, int seed) {
    if (auto* s = dynamic_cast<models::ISimulatable<std::vector<double>>*>(model))
        return s->generate_random_values(sample_size, seed);
    if (auto* s = dynamic_cast<models::ISimulatable<std::vector<std::vector<double>>>*>(model)) {
        std::vector<std::vector<double>> mat = s->generate_random_values(sample_size, seed);
        std::vector<double> flat;
        for (const auto& row : mat)
            for (double v : row) flat.push_back(v);
        return flat;
    }
    stop("model_estimation Simulation target: model is not ISimulatable<vector> or ISimulatable<Matrix2D>");
}

// --- construct assembly for the shared fit runner ------------------------------------------
//
// Every fit in this file now goes through corehydro/estimation/support/fit_runner.hpp's
// `run_fit`, which takes ONE serialized construct object rather than a flat argument list. The
// parsers, the GMM build-and-fit cascade and the BayesianAnalysis settings cascade that used to
// live here are that header's `parse_optimizer`/`parse_gmm_strategy`/`parse_sampler`/
// `build_and_fit_gmm`/`apply_bayesian_settings`; they were lifted out of this file verbatim, so
// the fits are unchanged.
//
// SETTINGS TRANSPARENCY. `run_fit` applies a knob only when the construct carries its key, and
// an absent key leaves the ported class's own default. That is exactly the semantics the flat
// `> 0` guards below used to have, so the guards move from the setter call to the construct
// assembly: a non-positive argument OMITS the key instead of writing a zero. Writing a zero
// would not be equivalent -- it would push a 0 into the setter where the old code skipped it --
// and for the Bayesian knobs that would move seeded chains and therefore pinned oracles.
namespace {

// The construct values here (optimizer / strategy / sampler names) are bare identifiers from a
// fixture's construct, so they need no JSON escaping.
std::string json_string(const std::string& s) { return "\"" + s + "\""; }

// Appends `,"key":value` only when `value > 0` -- the flat-argument spelling of the runner's
// "an absent key means the class default" contract (see the note above).
void append_if_positive(std::string& out, const char* key, int value) {
    if (value > 0) out += std::string(",\"") + key + "\":" + std::to_string(value);
}

}  // namespace

// Builds the model named by the serialized `construct.model` spec (`model_json`, via the
// shared spec builder -- see the file header), constructs `target`'s estimator (`optimizer`,
// default "DifferentialEvolution"), runs estimate() once, and returns a named list with the
// full surface test-fixtures.R's model_estimation dispatcher reads assertions from:
//   parameters       -- [n_params] vector (BestParameterSet.Values)
//   max_log_likelihood -- scalar
//   aic              -- scalar
//   covariance       -- [n_params x n_params] matrix
//   standard_errors  -- [n_params] vector
//   correlation      -- [n_params x n_params] matrix (T12)
//
// `bic` is deliberately NOT part of this surface -- see `ch_estimation_bic_` below and the
// file header DESIGN NOTE.
//
// WIRED (Task T11 + T12): parameter/max_log_likelihood/aic/bic/covariance/standard_error/
// correlation. `target == "BayesianAnalysis"` is NOT handled here (see
// `ch_estimation_bayes_run_` below -- disjoint construct shape).
[[cpp11::register]]
list ch_estimation_run_(std::string target, std::string model_json, doubles dataset,
                        std::string optimizer, int sample_size, int seed) {
    // Guard the target BEFORE delegating: `run_fit` also serves BayesianAnalysis, but this
    // entry point never has (that construct shape rides ch_estimation_bayes_run_), so the
    // rejection has to happen here rather than being silently accepted by the runner.
    if (target != "MaximumLikelihood" && target != "MaximumAPosteriori")
        stop("unknown model_estimation target '%s' (BayesianAnalysis uses ch_estimation_bayes_run_)",
             target.c_str());

    std::vector<double> data(dataset.begin(), dataset.end());
    std::string construct =
        "{\"model\":" + model_json + ",\"optimizer\":" + json_string(optimizer) + "}";
    est::support::FitResult r = est::support::run_fit(target, construct, data);

    int n_params = static_cast<int>(r.parameters.size());
    writable::doubles parameters(n_params);
    for (int i = 0; i < n_params; ++i) parameters[i] = r.parameters[static_cast<std::size_t>(i)];

    writable::doubles_matrix<by_column> covariance(n_params, n_params);
    writable::doubles standard_errors(n_params);
    writable::doubles_matrix<by_column> correlation(n_params, n_params);
    // DELIBERATE fixture-path divergence from the runner (the one this delegation carries; see
    // fit_runner.hpp's header note 1). The covariance stack needs >= 2 parameters -- the C#
    // GetCovarianceMatrix throws below that -- so `run_fit` reports NaN for a 1-parameter model,
    // the honest answer for the user-facing surface. This glue backs the PINNED FIXTURE oracles,
    // whose pre-existing contract there is a silent zero (the single-parameter bivariate copula
    // fit; no fixture asserts covariance/SE/correlation for a 1-param model). cpp11's
    // writable::doubles / doubles_matrix allocate through Rf_allocVector(REALSXP, ...), which R
    // does NOT zero-fill for numeric vectors -- so below, when n_params < 2, we WRITE 0.0 into
    // every cell explicitly; the glue produces zeros here, it does not inherit them for free.
    if (n_params >= 2) {
        for (int i = 0; i < n_params; ++i) {
            standard_errors[i] = r.standard_errors[static_cast<std::size_t>(i)];
            for (int j = 0; j < n_params; ++j) {
                covariance(i, j) = r.covariance[static_cast<std::size_t>(i * n_params + j)];
                correlation(i, j) = r.correlation[static_cast<std::size_t>(i * n_params + j)];
            }
        }
    } else {
        for (int i = 0; i < n_params; ++i) {
            standard_errors[i] = 0.0;
            for (int j = 0; j < n_params; ++j) {
                covariance(i, j) = 0.0;
                correlation(i, j) = 0.0;
            }
        }
    }

    // Optional seeded-draw digest off the FITTED model (P3): rebuild from the runner's
    // `model_spec` (the construct's model object re-emitted with the fitted `parameter_values`,
    // which the spec builder applies through the same set_parameter_values the old in-place
    // `model.set_parameter_values(best_values)` called) and cache one seeded draw, so one MLE
    // smoke file covers parameter + max_log_likelihood + a seeded draw. `simulate_flat` handles
    // the bivariate Matrix2D flatten.
    writable::doubles simulated(static_cast<R_xlen_t>(0));
    if (sample_size > 0) {
        std::unique_ptr<models::ModelBase> fitted =
            models::spec::build_model_from_json(r.model_spec, data);
        std::vector<double> draws = simulate_flat(fitted.get(), sample_size, seed);
        simulated = writable::doubles(static_cast<R_xlen_t>(draws.size()));
        for (std::size_t i = 0; i < draws.size(); ++i) simulated[static_cast<R_xlen_t>(i)] = draws[i];
    }

    return writable::list({
        "parameters"_nm = parameters,
        "max_log_likelihood"_nm = writable::doubles({r.log_likelihood}),
        "aic"_nm = writable::doubles({r.aic}),
        "covariance"_nm = covariance,
        "standard_errors"_nm = standard_errors,
        "correlation"_nm = correlation,
        "simulated"_nm = simulated,
    });
}

// `bic [n]` accessor: rebuilds the same model + named estimator, runs `estimate()` once (see
// the file header DESIGN NOTE for why this reproduces the exact same fit as
// `ch_estimation_run_`'s call), and returns `GetBIC(n)` evaluated live at the caller-supplied
// sample size `n` -- matching C++'s `dispatch_estimation` (`est->get_bic(a[0].get<int>())`)
// and the C# `GetBIC(sampleSize)` signature. Deliberately separate from `ch_estimation_run_`
// rather than folded into its returned list, since `n` is only known at assertion-dispatch
// time (a fixture case's `bic` assertion supplies it via `args[0]`), not at construction time.
[[cpp11::register]]
double ch_estimation_bic_(std::string target, std::string model_json, doubles dataset, std::string optimizer, int n) {
    if (target != "MaximumLikelihood" && target != "MaximumAPosteriori")
        stop("unknown model_estimation target '%s' (BayesianAnalysis has no bic method)",
             target.c_str());
    if (n < 1) stop("Sample size must be at least 1.");

    std::vector<double> data(dataset.begin(), dataset.end());
    std::string construct =
        "{\"model\":" + model_json + ",\"optimizer\":" + json_string(optimizer) + "}";
    est::support::FitResult r = est::support::run_fit(target, construct, data);
    // The live rebuild stays: `n` is only known at assertion-dispatch time. `MaximumLikelihood::
    // get_bic(n)` (maximum_likelihood.hpp:448-453) IS this call -- GoodnessOfFit::bic(n,
    // number_of_parameters(), maximum_log_likelihood()) -- so evaluating it off the runner's
    // FitResult is the same function on the same arguments, not a re-derivation.
    return corehydro::numerics::data::GoodnessOfFit::bic(n, static_cast<int>(r.parameters.size()),
                                                         r.log_likelihood);
}

// --- BayesianAnalysis (Task T12) -----------------------------------------------------------
//
// Disjoint construct shape from ML/MAP: a sampler type + numeric knobs, not an optimizer
// string, so this is a separate registered function rather than a `ch_estimation_run_` branch.
// Builds the model, constructs BayesianAnalysis(model, sampler), turns off the two "use
// defaults" flags so the explicit settings below aren't clobbered, applies whichever settings
// the fixture supplies (mirrors `ch_mcmc_run_`'s settings-application convention and the
// emitter's/C++ test_fixtures.cpp's `BuildEstimation` BayesianAnalysis arm), runs `estimate()`
// once, and returns every value test-fixtures.R's model_estimation dispatcher needs:
//   dic / waic / looic     -- scalars
//   posterior_mean         -- [n_params] vector
//   chain_values           -- [n_chains x n_iterations x n_params] flattened; see below
//
// `chain_value [chain, iter, param]` DESIGN NOTE: unlike the scalar/vector surface above, the
// chain digest is a 3-D lookup. R has no native 3-D-ragged-array-from-cpp11 shortcut as clean
// as a nested list, so this returns `chain_values` as a flat numeric vector plus `chain_dims`
// (n_chains, n_iterations, n_params); `dispatch_estimation`'s R-side helper below does the
// row-major index arithmetic (matching the C++/Python/C# access order:
// chains[chain][iter].values[param]).
[[cpp11::register]]
list ch_estimation_bayes_run_(std::string model_json, doubles dataset, std::string sampler,
                               int seed, int iterations, int warmup_iterations,
                               int number_of_chains, int thinning_interval,
                               int initial_iterations, int output_length) {
    std::vector<double> data(dataset.begin(), dataset.end());
    // The `> 0` guards that used to sit on each setter now sit on the construct assembly: a
    // non-positive argument OMITS the key, and `run_fit`'s apply_bayesian_settings then leaves
    // the ported BayesianAnalysis class default untouched -- exactly what skipping the setter
    // did. Passing a literal 0 instead would NOT be equivalent and would move seeded chains.
    // `seed` keeps its own `>= 0` spelling (0 is a legitimate seed); the runner applies it under
    // the identical `seed >= 0` test.
    std::string construct = "{\"model\":" + model_json + ",\"sampler\":" + json_string(sampler);
    if (seed >= 0) construct += ",\"seed\":" + std::to_string(seed);
    append_if_positive(construct, "iterations", iterations);
    append_if_positive(construct, "warmup_iterations", warmup_iterations);
    append_if_positive(construct, "number_of_chains", number_of_chains);
    append_if_positive(construct, "thinning_interval", thinning_interval);
    append_if_positive(construct, "initial_iterations", initial_iterations);
    append_if_positive(construct, "output_length", output_length);
    construct += "}";

    est::support::FitResult r = est::support::run_fit("BayesianAnalysis", construct, data);

    // chain_dims is {n_chains, n_iterations, n_params}; `draws` is already the chain-major flatten
    // this function has always returned (see the FitResult::draws note in fit_runner.hpp), and
    // MCMCResults::markov_chains is a copy of sampler()->markov_chains() (mcmc_results.hpp:35-37).
    int n_params = r.chain_dims[2];
    writable::doubles posterior_mean(n_params);
    for (int i = 0; i < n_params; ++i)
        posterior_mean[i] = r.posterior_mean[static_cast<std::size_t>(i)];

    writable::doubles chain_values(static_cast<R_xlen_t>(r.draws.size()));
    for (std::size_t i = 0; i < r.draws.size(); ++i)
        chain_values[static_cast<R_xlen_t>(i)] = r.draws[i];

    return writable::list({
        "dic"_nm = writable::doubles({r.dic}),
        "waic"_nm = writable::doubles({r.waic}),
        "looic"_nm = writable::doubles({r.looic}),
        "posterior_mean"_nm = posterior_mean,
        "chain_values"_nm = chain_values,
        "chain_dims"_nm = writable::integers({r.chain_dims[0], r.chain_dims[1], r.chain_dims[2]}),
    });
}

// --- DataFrame assertion surface (M14) -------------------------------------------------
//
// Methods reachable from the model's DataFrame under ANY model_estimation target,
// corroborating the M1/M5 ctest oracles through the PUBLIC path. Builds a FRESH model via
// the shared spec builder (the frame surface is a pure function of the construct -- low
// outliers / thresholds are set at construction and plotting positions of the collections,
// never of the fit -- so a rebuild returns byte-identical values; the `bic` lazy-rebuild
// precedent). Returns everything test-fixtures.R's data-frame dispatch arms read:
//   number_of_low_outliers, low_outlier_threshold -- scalars (frame state)
//   pp_exact / pp_interval / pp_uncertain -- plotting-position vectors, in spec order, after
//     ONE calculate_plotting_positions() pass (idempotent). The threshold series is NOT
//     exposed: the C# assigns its positions to a sorted CLONE, so the original items never
//     carry one -- mirroring the C++/emitter dispatchers.
[[cpp11::register]]
list ch_model_data_frame_(std::string model_json, doubles dataset) {
    std::unique_ptr<models::ModelBase> model = build_spec_model(model_json, dataset);
    auto* udm = dynamic_cast<models::UnivariateDistributionModelBase*>(model.get());
    if (udm == nullptr || !udm->has_data_frame())
        stop("model_estimation data-frame method on a model without a DataFrame");
    auto& df = udm->data_frame();
    df.calculate_plotting_positions();

    auto positions = [](const auto& series) {
        writable::doubles out(static_cast<R_xlen_t>(series.count()));
        for (std::size_t i = 0; i < series.count(); ++i)
            out[static_cast<R_xlen_t>(i)] = series[i].plotting_position();
        return out;
    };
    return writable::list({
        "number_of_low_outliers"_nm = writable::doubles({static_cast<double>(df.number_of_low_outliers())}),
        "low_outlier_threshold"_nm = writable::doubles({df.low_outlier_threshold()}),
        "pp_exact"_nm = positions(df.exact_series()),
        "pp_interval"_nm = positions(df.interval_series()),
        "pp_uncertain"_nm = positions(df.uncertain_series()),
    });
}

// --- Validate (Task 16) -----------------------------------------------------------------
//
// The estimator-less `Validate` target: builds the model through the shared spec builder and
// calls `ModelBase::validate()` ONCE (a pure function of the constructed model, so a rebuild
// is fine -- the `bic`/DataFrame lazy-rebuild precedent). Returns `is_valid` + the flat message
// list; test-fixtures.R's `is_valid`/`validation_message_contains` dispatch arms read it. Added
// for the TimeSeries transform-lambda-failure fixtures (BestFit v2.0.0): the failure is only
// oracle-visible through Validate(), not through any numeric estimator surface.
[[cpp11::register]]
list ch_model_validate_(std::string model_json, doubles dataset) {
    std::unique_ptr<models::ModelBase> model = build_spec_model(model_json, dataset);
    models::ValidationResult result = model->validate();
    writable::strings messages(static_cast<R_xlen_t>(result.validation_messages.size()));
    for (std::size_t i = 0; i < result.validation_messages.size(); ++i)
        messages[static_cast<R_xlen_t>(i)] = result.validation_messages[i];
    return writable::list({
        "is_valid"_nm = writable::logicals({cpp11::r_bool(result.is_valid)}),
        "messages"_nm = messages,
    });
}

// --- Simulation (M13) -----------------------------------------------------------------
//
// The estimator-less `Simulation` target: builds the model through the shared spec builder,
// calls the ISimulatable surface (`generate_random_values(sample_size, seed)`) ONCE, and
// returns the seeded draw vector; test-fixtures.R's `simulated_value [i]` dispatch indexes
// it. All four Phase 5 model types implement ISimulatable<std::vector<double>>; the
// dynamic_cast guard mirrors the C++ test runner's.
[[cpp11::register]]
doubles ch_model_simulate_(std::string model_json, doubles dataset, int sample_size, int seed) {
    std::unique_ptr<models::ModelBase> model = build_spec_model(model_json, dataset);
    // simulate_flat handles both ISimulatable<vector<double>> and the bivariate
    // ISimulatable<Matrix2D> (flattened row-major) -- see its header note.
    std::vector<double> draws = simulate_flat(model.get(), sample_size, seed);
    writable::doubles out(static_cast<R_xlen_t>(draws.size()));
    for (std::size_t i = 0; i < draws.size(); ++i) out[static_cast<R_xlen_t>(i)] = draws[i];
    return out;
}

// --- GeneralizedMethodOfMoments (B11) --------------------------------------------------
//
// Disjoint construct shape from ML/MAP (a strategy + max_gmm_iterations instead of just an
// optimizer string) AND a different model type -- the concrete Bulletin17CDistribution the GMM
// ctor takes as IGMMModel& (NOT a ModelBase; see model_spec.hpp), built through the shared spec
// builder's dedicated bulletin17c entry point. So this is a separate registered function,
// mirroring the ch_estimation_bayes_run_ split. Fits once, post_processes, and returns every
// value test-fixtures.R's GMM dispatcher reads:
//   parameters / standard_errors     -- [p] vectors
//   covariance / correlation         -- [p x p] matrices
//   j_stat / j_stat_pval             -- scalars (pval is NaN when q - p == 0)
//   simulated                        -- [sample_size] seeded ISimulatable draw off the FITTED
//                                       model (present only when construct supplies sample_size),
//                                       read by the shared `simulated_value` dispatch arm.
// quantile_variance rides ch_estimation_gmm_qvar_ (per-assertion AEP, exactly like `bic`'s
// per-assertion sample size).
[[cpp11::register]]
list ch_estimation_gmm_run_(std::string model_json, doubles dataset, std::string strategy,
                            std::string optimizer, int max_gmm_iterations, int sample_size,
                            int seed) {
    std::vector<double> data(dataset.begin(), dataset.end());
    // Same construct-assembly guard as ch_estimation_bayes_run_: `max_gmm_iterations` is applied
    // only when positive, which used to be the `if (max_gmm_iterations > 0)` around the setter.
    std::string construct = "{\"model\":" + model_json + ",\"optimizer\":" + json_string(optimizer) +
                            ",\"strategy\":" + json_string(strategy);
    append_if_positive(construct, "max_gmm_iterations", max_gmm_iterations);
    construct += "}";

    est::support::FitResult r = est::support::run_fit("GMM", construct, data);
    int p = static_cast<int>(r.parameters.size());

    writable::doubles parameters(p);
    writable::doubles standard_errors(p);
    writable::doubles_matrix<by_column> covariance(p, p);
    writable::doubles_matrix<by_column> correlation(p, p);
    for (int i = 0; i < p; ++i) {
        parameters[i] = r.parameters[static_cast<std::size_t>(i)];
        standard_errors[i] = r.standard_errors[static_cast<std::size_t>(i)];
        for (int j = 0; j < p; ++j) {
            covariance(i, j) = r.covariance[static_cast<std::size_t>(i * p + j)];
            correlation(i, j) = r.correlation[static_cast<std::size_t>(i * p + j)];
        }
    }

    // Seeded draw off the FITTED B17C model, rebuilt from the runner's `model_spec` (the fitted
    // `parameter_values` are applied by the spec builder through the same set_parameter_values
    // the old in-place pin called).
    writable::doubles simulated(static_cast<R_xlen_t>(0));
    if (sample_size > 0) {
        std::unique_ptr<models::Bulletin17CDistribution> fitted =
            models::spec::build_bulletin17c_from_json(r.model_spec, data);
        std::vector<double> draws = fitted->generate_random_values(sample_size, seed);
        simulated = writable::doubles(static_cast<R_xlen_t>(draws.size()));
        for (std::size_t i = 0; i < draws.size(); ++i) simulated[static_cast<R_xlen_t>(i)] = draws[i];
    }

    return writable::list({
        "parameters"_nm = parameters,
        "standard_errors"_nm = standard_errors,
        "covariance"_nm = covariance,
        "correlation"_nm = correlation,
        "j_stat"_nm = writable::doubles({r.j_stat}),
        "j_stat_pval"_nm = writable::doubles({r.j_stat_pval}),
        // T13: GMMIterations/ConvergedWithinTolerance (off-by-one fix) and
        // OptimizerFallbackCount (sticky BFGS->NelderMead fallback).
        "gmm_iterations"_nm = writable::integers({r.gmm_iterations}),
        "converged_within_tolerance"_nm =
            writable::logicals({cpp11::r_bool(r.converged_within_tolerance)}),
        "optimizer_fallback_count"_nm = writable::integers({r.optimizer_fallback_count}),
        "simulated"_nm = simulated,
    });
}

// `quantile_variance [aep]`: like `bic [n]`, the AEP is only known at assertion-dispatch time,
// so this rebuilds the same deterministic fit and evaluates the B17C delta-method Var(Q_p) live.
// `aep` is the annual EXCEEDANCE probability; the C# QuantileVariance takes a NON-exceedance
// probability, so pass 1 - AEP.
[[cpp11::register]]
double ch_estimation_gmm_qvar_(std::string model_json, doubles dataset, std::string strategy,
                               std::string optimizer, int max_gmm_iterations, double aep) {
    std::vector<double> data(dataset.begin(), dataset.end());
    std::string construct = "{\"model\":" + model_json + ",\"optimizer\":" + json_string(optimizer) +
                            ",\"strategy\":" + json_string(strategy);
    append_if_positive(construct, "max_gmm_iterations", max_gmm_iterations);
    construct += "}";
    // The live rebuild stays (the AEP is only known at assertion-dispatch time);
    // run_fit_quantile_variance IS this function's old body, lifted into fit_runner.hpp.
    return est::support::run_fit_quantile_variance(construct, data, aep);
}

// --- The user-facing fit surface (Task 6) ---------------------------------------------------
//
// Reshapes an n*n row-major flat vector (FitResult's covariance/correlation convention) into an
// n x n matrix, OR -- when the vector is empty (hessian = FALSE, or a target that never
// populates it) -- a zero-length numeric. R's new_fit()/fit_optimized() (corehydror/R/fit.R)
// turn that empty numeric into NULL; see fit_runner.hpp's FitResult doc for the row-major
// convention this mirrors (and ch_estimation_run_ above, which reshapes the identical way for
// the fixture path).
static sexp square_or_empty(const std::vector<double>& v, int n) {
    if (v.empty()) return as_sexp(writable::doubles(static_cast<R_xlen_t>(0)));
    writable::doubles_matrix<by_column> m(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) m(i, j) = v[static_cast<std::size_t>(i * n + j)];
    return as_sexp(m);
}

// Same idea as square_or_empty but rectangular, for FitDiagnostics::observation_influence
// (n_obs x n_params, row-major -- see fit_runner.hpp's flatten_influence).
static sexp matrix_or_empty(const std::vector<double>& v, int nrow, int ncol) {
    if (v.empty()) return as_sexp(writable::doubles(static_cast<R_xlen_t>(0)));
    writable::doubles_matrix<by_column> m(nrow, ncol);
    for (int i = 0; i < nrow; ++i)
        for (int j = 0; j < ncol; ++j) m(i, j) = v[static_cast<std::size_t>(i * ncol + j)];
    return as_sexp(m);
}

// Flattens a double vector into an R numeric of the same length. Shared by ch_fit_run_'s new
// Bayesian/GMM fields below and ch_fit_diagnostics_.
static writable::doubles doubles_of(const std::vector<double>& v) {
    writable::doubles out(static_cast<R_xlen_t>(v.size()));
    for (std::size_t i = 0; i < v.size(); ++i) out[static_cast<R_xlen_t>(i)] = v[i];
    return out;
}

// The user-facing fit entry point. ONE function for all four estimators (MaximumLikelihood,
// MaximumAPosteriori here; BayesianAnalysis and GMM added in a later task): the R verbs in
// R/fit.R assemble `construct_json` (via fit_input()) and read this named list back, picking
// the fields their target populates. Every FitResult field `run_fit` can fill is packed here so
// a later task extends the R side only, not this entry point's shape.
[[cpp11::register]]
list ch_fit_run_(std::string target, std::string construct_json, doubles dataset) {
    std::vector<double> data(dataset.begin(), dataset.end());
    est::support::FitResult r = est::support::run_fit(target, construct_json, data);
    int n = static_cast<int>(r.parameters.size());

    writable::doubles parameters(n);
    writable::strings parameter_names(n);
    for (int i = 0; i < n; ++i) {
        parameters[i] = r.parameters[static_cast<std::size_t>(i)];
        parameter_names[i] = r.parameter_names[static_cast<std::size_t>(i)];
    }

    writable::doubles standard_errors(static_cast<R_xlen_t>(r.standard_errors.size()));
    for (std::size_t i = 0; i < r.standard_errors.size(); ++i)
        standard_errors[static_cast<R_xlen_t>(i)] = r.standard_errors[i];

    writable::doubles profile_grid(static_cast<R_xlen_t>(r.profile_grid.size()));
    for (std::size_t i = 0; i < r.profile_grid.size(); ++i)
        profile_grid[static_cast<R_xlen_t>(i)] = r.profile_grid[i];
    writable::doubles profile_lower(static_cast<R_xlen_t>(r.profile_lower.size()));
    for (std::size_t i = 0; i < r.profile_lower.size(); ++i)
        profile_lower[static_cast<R_xlen_t>(i)] = r.profile_lower[i];
    writable::doubles profile_upper(static_cast<R_xlen_t>(r.profile_upper.size()));
    for (std::size_t i = 0; i < r.profile_upper.size(); ++i)
        profile_upper[static_cast<R_xlen_t>(i)] = r.profile_upper[i];

    // Draws stay flat CHAIN-major (see FitResult::draws's doc) alongside chain_dims; empty for
    // MaximumLikelihood/MaximumAPosteriori targets. A later task's fit_bayesian() reshapes them.
    writable::doubles draws(static_cast<R_xlen_t>(r.draws.size()));
    for (std::size_t i = 0; i < r.draws.size(); ++i) draws[static_cast<R_xlen_t>(i)] = r.draws[i];
    writable::integers chain_dims(static_cast<R_xlen_t>(r.chain_dims.size()));
    for (std::size_t i = 0; i < r.chain_dims.size(); ++i)
        chain_dims[static_cast<R_xlen_t>(i)] = r.chain_dims[i];

    return writable::list({
        "method"_nm = writable::strings({r.method}),
        "parameter_names"_nm = parameter_names,
        "parameters"_nm = parameters,
        "log_likelihood"_nm = writable::doubles({r.log_likelihood}),
        "prior_log_likelihood"_nm = writable::doubles({r.prior_log_likelihood}),
        "aic"_nm = writable::doubles({r.aic}),
        "bic"_nm = writable::doubles({r.bic}),
        "nobs"_nm = writable::integers({r.nobs}),
        "covariance"_nm = square_or_empty(r.covariance, n),
        "standard_errors"_nm = standard_errors,
        "correlation"_nm = square_or_empty(r.correlation, n),
        "converged"_nm = writable::logicals({cpp11::r_bool(r.converged)}),
        "status"_nm = writable::strings({r.status}),
        "function_evaluations"_nm = writable::integers({r.function_evaluations}),
        "model_spec"_nm = writable::strings({r.model_spec}),
        "profile_grid"_nm = profile_grid,
        "profile_lower"_nm = profile_lower,
        "profile_upper"_nm = profile_upper,
        "profile_bins"_nm = writable::integers({r.profile_bins}),
        "draws"_nm = draws,
        "chain_dims"_nm = chain_dims,
        // --- Bayesian-only fields (Task 7); empty for MaximumLikelihood/MaximumAPosteriori ---
        "acceptance_rates"_nm = doubles_of(r.acceptance_rates),
        "dic"_nm = writable::doubles({r.dic}),
        "waic"_nm = writable::doubles({r.waic}),
        "looic"_nm = writable::doubles({r.looic}),
        "rhat"_nm = doubles_of(r.rhat),
        "ess"_nm = doubles_of(r.ess),
        "summary_mean"_nm = doubles_of(r.summary_mean),
        "summary_median"_nm = doubles_of(r.summary_median),
        "summary_sd"_nm = doubles_of(r.summary_sd),
        "summary_lower"_nm = doubles_of(r.summary_lower),
        "summary_upper"_nm = doubles_of(r.summary_upper),
        // --- GMM-only fields (Task 7); NaN/0/false for the other three targets -----------------
        "j_stat"_nm = writable::doubles({r.j_stat}),
        "j_stat_pval"_nm = writable::doubles({r.j_stat_pval}),
        "gmm_iterations"_nm = writable::integers({r.gmm_iterations}),
        "converged_within_tolerance"_nm =
            writable::logicals({cpp11::r_bool(r.converged_within_tolerance)}),
        "optimizer_fallback_count"_nm = writable::integers({r.optimizer_fallback_count}),
    });
}

// --- Estimation diagnostics off a fit (Task 7) ----------------------------------------------
//
// R's fit_diagnostics() reruns the fit's own construct (the JSON fit_input() built, carried on
// the corehydro_fit as $construct_json) through run_fit_diagnostics -- see that function's
// header for why an explicit warmup_iterations on the ORIGINAL construct is what makes this
// agree with the fit it is diagnosing. `target` is one of MaximumAPosteriori/BayesianAnalysis/
// GMM (matching FitDiagnostics's three populated arms); MaximumLikelihood is rejected R-side
// (see fit_diagnostics() in R/fit.R) before this is ever called.
[[cpp11::register]]
list ch_fit_diagnostics_(std::string target, std::string construct_json, doubles dataset) {
    std::vector<double> data(dataset.begin(), dataset.end());
    est::support::FitDiagnostics d =
        est::support::run_fit_diagnostics(target, construct_json, data);

    // observation_influence is n_obs x n_params, row-major (fit_runner.hpp's flatten_influence);
    // n_obs is cooks_distance's length (populated together, MAP/GMM only -- see FitDiagnostics's
    // header), so n_params divides out of the flat length rather than needing a separate field.
    int n_obs = static_cast<int>(d.cooks_distance.size());
    int n_params = (n_obs > 0 && !d.observation_influence.empty())
                       ? static_cast<int>(d.observation_influence.size() /
                                          static_cast<std::size_t>(n_obs))
                       : 0;

    writable::strings prior_influence_names(static_cast<R_xlen_t>(d.prior_influence_names.size()));
    for (std::size_t i = 0; i < d.prior_influence_names.size(); ++i)
        prior_influence_names[static_cast<R_xlen_t>(i)] = d.prior_influence_names[i];

    return writable::list({
        "cooks_distance"_nm = doubles_of(d.cooks_distance),
        "leverage"_nm = doubles_of(d.leverage),
        "observation_influence"_nm = matrix_or_empty(d.observation_influence, n_obs, n_params),
        "pareto_k"_nm = doubles_of(d.pareto_k),
        "max_pareto_k"_nm = writable::doubles({d.max_pareto_k}),
        "prior_influence"_nm = doubles_of(d.prior_influence),
        "prior_influence_names"_nm = prior_influence_names,
    });
}

// `quantile_variance(fit, aep)`: like ch_estimation_gmm_qvar_ (the fixture-path twin), rebuilds
// the same deterministic GMM fit from the fit's own construct and evaluates the B17C
// delta-method Var(Q_p) live -- `aep` is only known at call time, so this cannot be precomputed
// into ch_fit_run_'s result the way every other GMM field is.
[[cpp11::register]]
double ch_fit_quantile_variance_(std::string construct_json, doubles dataset, double aep) {
    std::vector<double> data(dataset.begin(), dataset.end());
    return est::support::run_fit_quantile_variance(construct_json, data, aep);
}
