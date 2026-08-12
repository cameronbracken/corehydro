// pybind11 glue exposing the estimation surface (MaximumLikelihood, MaximumAPosteriori,
// BayesianAnalysis, and -- as of M13 -- the seeded ISimulatable draw) of the shared C++ core
// to Python. Like `mcmc_sampler`/`bootstrap` fixtures, `model_estimation` fixtures are
// inherently STATEFUL -- one model construct + a single estimate() run backs every assertion
// in a case (see fixtures/README.md's model_estimation schema) -- so this file exposes ONE
// function per construct shape: `estimation_run` for MaximumLikelihood/MaximumAPosteriori
// (shared {target, model_json, dataset, optimizer} signature), `estimation_bayes_run` (T12)
// for BayesianAnalysis (a disjoint {model_json, dataset, sampler, settings...} signature -- a
// sampler type + numeric knobs, not an optimizer string), and `model_simulate` (M13) for the
// estimator-less Simulation target, mirroring corehydror's `ch_estimation_run_`/
// `ch_estimation_bayes_run_`/`ch_model_simulate_` split. Each builds the model, runs its one
// stateful call, and returns every value test_fixtures.py's dispatcher needs in one dict.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
//
// M13 MODEL CONSTRUCTION: the flat Phase 4 `family` string became the serialized
// `construct.model` JSON object (`model_json`), parsed and built by the SHARED spec builder
// (corehydro/models/model_spec.hpp) -- the same code path the C++ test runner and the cpp11
// glue call, so all three harnesses construct byte-identical models (UnivariateDistribution
// incl. censored DataFrames + nonstationary trends, Mixture, CompetingRisks, PointProcess).
// The runner re-serializes the parsed fixture spec with `json.dumps()`, which round-trips
// doubles exactly. `dataset` stays a separate flat argument: the file-level `datasets` map is
// resolved Python-side, exactly like every other fixture kind.
//
// `bic` DESIGN NOTE: unlike every other wired ML/MAP method, C# `GetBIC(sampleSize)` takes an
// actual sample size, not a 0-based index. Every other method's value is precomputed once here
// (in `estimation_run`, matching `mcmc_run`/`bootstrap_run`'s "precompute the full surface up
// front" contract), since none of them take a fixture-supplied argument. `bic` is the one
// exception: it is NOT precomputed. `estimation_bic` below rebuilds the same model/estimator
// (deterministic -- NelderMead/Brent have no randomness and DifferentialEvolution's default
// `prng_seed` is fixed, so re-running `estimate()` reproduces the exact same fit) and calls
// `e.get_bic(n)` live with whatever `n` the fixture's `args[0]` supplies, matching C++'s
// `dispatch_estimation` and the C# `GetBIC(sampleSize)` signature exactly. See
// `test_fixtures.py`'s `bic` dispatch arm.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
#include "bindings.hpp"

namespace py = pybind11;
namespace est = corehydro::estimation;
namespace models = corehydro::models;

// --- construct assembly for the shared fit runner ------------------------------------------
//
// Every fit in this file now goes through corehydro/estimation/support/fit_runner.hpp's
// `run_fit`, which takes ONE serialized construct object rather than a flat argument list. The
// parsers, the GMM build-and-fit cascade and the BayesianAnalysis settings cascade that used to
// live here are that header's `parse_optimizer`/`parse_gmm_strategy`/`parse_sampler`/
// `build_and_fit_gmm`/`apply_bayesian_settings`; they were lifted out of the sibling cpp11 glue
// verbatim, so the fits are unchanged.
//
// SETTINGS TRANSPARENCY. `run_fit` applies a knob only when the construct carries its key, and
// an absent key leaves the ported class's own default. That matches this file's pre-existing
// `settings.contains(...)` semantics one-for-one: a key the caller did not pass is simply not
// forwarded into the construct, never forwarded as a zero.
namespace {

// The construct values here (optimizer / strategy / sampler names) are bare identifiers from a
// fixture's construct, so they need no JSON escaping.
std::string json_string(const std::string& s) { return "\"" + s + "\""; }

// Appends `,"key":value` only when `value > 0` -- the runner's "an absent key means the class
// default" contract, matching the `if (max_gmm_iterations > 0)` guard this file used to carry.
void append_if_positive(std::string& out, const char* key, int value) {
    if (value > 0) out += std::string(",\"") + key + "\":" + std::to_string(value);
}

// Forwards `key` into the construct only when the settings dict actually carries it, exactly
// reproducing the `if (settings.contains(key)) ba.set_<key>(...)` cascade this file used to run.
void append_setting(std::string& out, const py::dict& settings, const char* key) {
    if (settings.contains(key))
        out += std::string(",\"") + key + "\":" + std::to_string(settings[key].cast<int>());
}

// Reshapes a row-major flat n x n vector into a nested list, matching estimation_run's/
// estimation_gmm_run's existing covariance/correlation convention -- an empty `v` (hessian not
// requested, or a target that never populates it) comes back as an empty list, which fit.py
// (the Task 8 caller) treats as "not computed" exactly like corehydror's `name_square`. A
// single-parameter model still returns a genuine (NaN-filled) 1x1 nested list, matching
// FitResult's own n < 2 guard -- see fit_runner.hpp's fill_common.
std::vector<std::vector<double>> nested_square(const std::vector<double>& v, int n) {
    std::vector<std::vector<double>> out;
    if (v.empty()) return out;
    out.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j)
            out[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                v[static_cast<std::size_t>(i * n + j)];
    }
    return out;
}

// Same idea, rectangular, for FitDiagnostics::observation_influence (n_obs x n_params,
// row-major -- see fit_runner.hpp's flatten_influence and corehydror's `matrix_or_empty`).
std::vector<std::vector<double>> nested_rect(const std::vector<double>& v, int nrow, int ncol) {
    std::vector<std::vector<double>> out;
    if (v.empty() || nrow == 0 || ncol == 0) return out;
    out.resize(static_cast<std::size_t>(nrow));
    for (int i = 0; i < nrow; ++i) {
        out[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(ncol));
        for (int j = 0; j < ncol; ++j)
            out[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                v[static_cast<std::size_t>(i * ncol + j)];
    }
    return out;
}

}  // namespace

// Seeded ISimulatable draw, flattened to a 1-D vector so the `simulated_value [i]` digest works
// uniformly across model types (P3). Most Phase 4-7 models are ISimulatable<std::vector<double>>;
// BivariateDistribution is ISimulatable<Matrix2D> (n-row x 2-col), flattened ROW-MAJOR
// (i = row*2 + col) -- the same order the C++/R glue and the README schema use.
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
    throw py::value_error(
        "model_estimation Simulation target: model is not ISimulatable<vector> or "
        "ISimulatable<Matrix2D>");
}

void register_estimation(py::module_& m) {
    // Builds the named model (`family` via the distribution factory + `dataset`), constructs
    // `target`'s estimator (`optimizer`, default "DifferentialEvolution"), runs estimate()
    // once, and returns a dict with the full surface test_fixtures.py's model_estimation
    // dispatcher reads assertions from:
    //   parameters          -- [n_params] list (BestParameterSet.Values)
    //   max_log_likelihood  -- scalar
    //   aic                 -- scalar
    //   covariance          -- [n_params][n_params] nested list
    //   standard_errors     -- [n_params] list
    //   correlation         -- [n_params][n_params] nested list (T12)
    //
    // `bic` is deliberately NOT part of this surface -- see `estimation_bic` below and the file
    // header DESIGN NOTE.
    //
    // WIRED (Task T11 + T12): parameter/max_log_likelihood/aic/bic/covariance/standard_error/
    // correlation. `target == "BayesianAnalysis"` is NOT handled here (see
    // `estimation_bayes_run` below -- disjoint construct shape); dic/waic/looic/posterior_mean/
    // chain_value are BayesianAnalysis-only surface exposed there instead.
    m.def(
        "estimation_run",
        [](const std::string& target, const std::string& model_json, const std::vector<double>& dataset,
           const std::string& optimizer, int sample_size, int seed) {
            // Guard the target BEFORE delegating: `run_fit` also serves BayesianAnalysis, but
            // this entry point never has (that construct shape rides estimation_bayes_run), so
            // the rejection has to happen here rather than being silently accepted.
            if (target == "BayesianAnalysis")
                throw py::value_error(
                    "model_estimation target 'BayesianAnalysis' uses estimation_bayes_run, not "
                    "estimation_run (disjoint construct shape)");
            if (target != "MaximumLikelihood" && target != "MaximumAPosteriori")
                throw py::value_error("unknown model_estimation target: " + target);

            std::string construct =
                "{\"model\":" + model_json + ",\"optimizer\":" + json_string(optimizer) + "}";
            est::support::FitResult r = est::support::run_fit(target, construct, dataset);
            int n_params = static_cast<int>(r.parameters.size());

            py::dict out;
            out["parameters"] = r.parameters;
            out["max_log_likelihood"] = r.log_likelihood;
            out["aic"] = r.aic;

            // DELIBERATE fixture-path divergence from the runner (the one this delegation
            // carries; see fit_runner.hpp's header note 1). The covariance stack needs >= 2
            // parameters -- the C# GetCovarianceMatrix throws below that -- so `run_fit` reports
            // NaN for a 1-parameter model, the honest answer for the user-facing surface. This
            // glue backs the PINNED FIXTURE oracles, whose pre-existing contract there is to omit
            // the three keys entirely (the single-parameter bivariate copula fit; no fixture
            // asserts covariance/SE/correlation for a 1-param model). Emitting them only when
            // n_params >= 2 keeps the runner's NaN block out of the fixture dict exactly as
            // before. The user-facing path stays honest.
            if (n_params >= 2) {
                std::vector<std::vector<double>> covariance(static_cast<std::size_t>(n_params));
                std::vector<std::vector<double>> correlation(static_cast<std::size_t>(n_params));
                for (int i = 0; i < n_params; ++i) {
                    covariance[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(n_params));
                    correlation[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(n_params));
                    for (int j = 0; j < n_params; ++j) {
                        covariance[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                            r.covariance[static_cast<std::size_t>(i * n_params + j)];
                        correlation[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                            r.correlation[static_cast<std::size_t>(i * n_params + j)];
                    }
                }
                out["covariance"] = covariance;
                out["standard_errors"] = r.standard_errors;
                out["correlation"] = correlation;
            }

            // Optional seeded-draw digest off the FITTED model (P3): rebuild from the runner's
            // `model_spec` (the construct's model object re-emitted with the fitted
            // `parameter_values`, which the spec builder applies through the same
            // set_parameter_values the old in-place pin called) and cache one seeded draw,
            // mirroring the C++/R/GMM arms. `simulate_flat` handles the bivariate Matrix2D
            // flatten.
            if (sample_size > 0) {
                std::unique_ptr<models::ModelBase> fitted =
                    models::spec::build_model_from_json(r.model_spec, dataset);
                out["simulated"] = simulate_flat(fitted.get(), sample_size, seed);
            }
            return out;
        },
        py::arg("target"), py::arg("model_json"), py::arg("dataset"), py::arg("optimizer"),
        py::arg("sample_size"), py::arg("seed"));

    // `bic [n]` accessor: rebuilds the same model + named estimator, runs `estimate()` once
    // (see the file header DESIGN NOTE for why this reproduces the exact same fit as
    // `estimation_run`'s call), and returns `GetBIC(n)` evaluated live at the caller-supplied
    // sample size `n` -- matching C++'s `dispatch_estimation`
    // (`est->get_bic(a[0].get<int>())`) and the C# `GetBIC(sampleSize)` signature.
    // Deliberately separate from `estimation_run` rather than folded into its returned dict,
    // since `n` is only known at assertion-dispatch time (a fixture case's `bic` assertion
    // supplies it via `args[0]`), not at construction time.
    m.def(
        "estimation_bic",
        [](const std::string& target, const std::string& model_json, const std::vector<double>& dataset,
           const std::string& optimizer, int n) {
            if (target == "BayesianAnalysis")
                throw py::value_error(
                    "model_estimation target 'BayesianAnalysis' has no bic method");
            if (target != "MaximumLikelihood" && target != "MaximumAPosteriori")
                throw py::value_error("unknown model_estimation target: " + target);
            if (n < 1) throw py::value_error("Sample size must be at least 1.");

            std::string construct =
                "{\"model\":" + model_json + ",\"optimizer\":" + json_string(optimizer) + "}";
            est::support::FitResult r = est::support::run_fit(target, construct, dataset);
            // The live rebuild stays: `n` is only known at assertion-dispatch time.
            // `MaximumLikelihood::get_bic(n)` (maximum_likelihood.hpp:448-453) IS this call --
            // GoodnessOfFit::bic(n, number_of_parameters(), maximum_log_likelihood()) -- so
            // evaluating it off the runner's FitResult is the same function on the same
            // arguments, not a re-derivation.
            return corehydro::numerics::data::GoodnessOfFit::bic(
                n, static_cast<int>(r.parameters.size()), r.log_likelihood);
        },
        py::arg("target"), py::arg("model_json"), py::arg("dataset"), py::arg("optimizer"), py::arg("n"));

    // --- BayesianAnalysis (Task T12) -------------------------------------------------------
    //
    // Disjoint construct shape from ML/MAP: a sampler type + numeric knobs, not an optimizer
    // string, so this is a separate registered function rather than an `estimation_run`
    // branch. Builds the model, constructs BayesianAnalysis(model, sampler), turns off the two
    // "use defaults" flags so the explicit settings below aren't clobbered, applies whichever
    // settings the fixture supplies (mirrors `mcmc_run`'s settings-application convention and
    // the emitter's/C++ test_fixtures.cpp's/corehydror's `ch_estimation_bayes_run_` BayesianAnalysis
    // construction), runs `estimate()` once, and returns every value test_fixtures.py's
    // model_estimation dispatcher needs:
    //   dic / waic / looic  -- scalars
    //   posterior_mean      -- [n_params] list
    //   chains              -- [n_chains][n_iterations][n_params] nested list (MarkovChains),
    //                          matching `mcmc_run`'s "chains" convention (unlike R, which has
    //                          no clean 3-D-from-cpp11 shortcut, pybind11/stl handles nested
    //                          std::vector natively, so no flat-plus-dims encoding is needed).
    // `chain_value [chain, iter, param]` on the Python side simply triple-indexes this nested
    // list -- matching the C++/R/C# access order: chains[chain][iter].values[param].
    m.def(
        "estimation_bayes_run",
        [](const std::string& model_json, const std::vector<double>& dataset, const std::string& sampler,
           const py::dict& settings) {
            // Each `settings.contains(key)` guard that used to sit on a setter now sits on the
            // construct assembly: a key the caller omitted is not forwarded, and `run_fit`'s
            // apply_bayesian_settings then leaves the ported BayesianAnalysis class default
            // untouched -- exactly what skipping the setter did. (One narrowing the runner
            // applies on top: it honours `seed` only when it is >= 0. No fixture carries a
            // negative seed, and the sibling cpp11 glue has always had that same `seed >= 0`
            // guard, so the two harnesses agree.)
            std::string construct =
                "{\"model\":" + model_json + ",\"sampler\":" + json_string(sampler);
            append_setting(construct, settings, "seed");
            append_setting(construct, settings, "iterations");
            append_setting(construct, settings, "warmup_iterations");
            append_setting(construct, settings, "number_of_chains");
            append_setting(construct, settings, "thinning_interval");
            append_setting(construct, settings, "initial_iterations");
            append_setting(construct, settings, "output_length");
            construct += "}";

            est::support::FitResult r =
                est::support::run_fit("BayesianAnalysis", construct, dataset);

            // `draws` is the chain-major flatten of MCMCResults::markov_chains, itself a copy of
            // sampler()->markov_chains() (mcmc_results.hpp:35-37); re-nest it into the
            // [chain][iter][param] list this function has always returned.
            int n_chains = r.chain_dims[0], n_iterations = r.chain_dims[1], n_params = r.chain_dims[2];
            std::vector<std::vector<std::vector<double>>> chains(static_cast<std::size_t>(n_chains));
            std::size_t k = 0;
            for (int c = 0; c < n_chains; ++c) {
                chains[static_cast<std::size_t>(c)].resize(static_cast<std::size_t>(n_iterations));
                for (int it = 0; it < n_iterations; ++it) {
                    auto& row = chains[static_cast<std::size_t>(c)][static_cast<std::size_t>(it)];
                    row.assign(r.draws.begin() + static_cast<std::ptrdiff_t>(k),
                               r.draws.begin() + static_cast<std::ptrdiff_t>(k + static_cast<std::size_t>(n_params)));
                    k += static_cast<std::size_t>(n_params);
                }
            }

            py::dict out;
            out["dic"] = r.dic;
            out["waic"] = r.waic;
            out["looic"] = r.looic;
            out["posterior_mean"] = r.posterior_mean;
            out["chains"] = chains;
            return out;
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("sampler"), py::arg("settings"));

    // --- DataFrame assertion surface (M14) ---------------------------------------------
    //
    // Methods reachable from the model's DataFrame under ANY model_estimation target,
    // corroborating the M1/M5 ctest oracles through the PUBLIC path. Builds a FRESH model
    // via the shared spec builder (the frame surface is a pure function of the construct --
    // low outliers / thresholds are set at construction and plotting positions of the
    // collections, never of the fit -- so a rebuild returns byte-identical values; the
    // `bic` lazy-rebuild precedent). Returns everything test_fixtures.py's data-frame
    // dispatch arms read:
    //   number_of_low_outliers, low_outlier_threshold -- scalars (frame state)
    //   pp_exact / pp_interval / pp_uncertain -- plotting-position lists, in spec order,
    //     after ONE calculate_plotting_positions() pass (idempotent). The threshold series
    //     is NOT exposed: the C# assigns its positions to a sorted CLONE, so the original
    //     items never carry one -- mirroring the C++/emitter/R dispatchers.
    m.def(
        "model_data_frame",
        [](const std::string& model_json, const std::vector<double>& dataset) {
            std::unique_ptr<models::ModelBase> model =
                models::spec::build_model_from_json(model_json, dataset);
            auto* udm = dynamic_cast<models::UnivariateDistributionModelBase*>(model.get());
            if (udm == nullptr || !udm->has_data_frame())
                throw py::value_error(
                    "model_estimation data-frame method on a model without a DataFrame");
            auto& df = udm->data_frame();
            df.calculate_plotting_positions();

            auto positions = [](const auto& series) {
                std::vector<double> out(series.count());
                for (std::size_t i = 0; i < series.count(); ++i) out[i] = series[i].plotting_position();
                return out;
            };
            py::dict out;
            out["number_of_low_outliers"] = df.number_of_low_outliers();
            out["low_outlier_threshold"] = df.low_outlier_threshold();
            out["pp_exact"] = positions(df.exact_series());
            out["pp_interval"] = positions(df.interval_series());
            out["pp_uncertain"] = positions(df.uncertain_series());
            return out;
        },
        py::arg("model_json"), py::arg("dataset"));

    // --- Validate (Task 16) -------------------------------------------------------------
    //
    // The estimator-less `Validate` target: builds the model through the shared spec builder
    // and calls `ModelBase::validate()` ONCE (a pure function of the constructed model, so a
    // rebuild is fine -- the `bic`/DataFrame lazy-rebuild precedent). Returns `is_valid` + the
    // flat message list; test_fixtures.py's `is_valid`/`validation_message_contains` dispatch
    // arms read it. Added for the TimeSeries transform-lambda-failure fixtures (BestFit
    // v2.0.0): the failure is only oracle-visible through Validate(), not through any numeric
    // estimator surface.
    m.def(
        "model_validate",
        [](const std::string& model_json, const std::vector<double>& dataset) {
            std::unique_ptr<models::ModelBase> model =
                models::spec::build_model_from_json(model_json, dataset);
            models::ValidationResult result = model->validate();
            py::dict out;
            out["is_valid"] = result.is_valid;
            out["messages"] = result.validation_messages;
            return out;
        },
        py::arg("model_json"), py::arg("dataset"));

    // --- Simulation (M13) -------------------------------------------------------------
    //
    // The estimator-less `Simulation` target: builds the model through the shared spec
    // builder, calls the ISimulatable surface (`generate_random_values(sample_size, seed)`)
    // ONCE, and returns the seeded draw vector; test_fixtures.py's `simulated_value [i]`
    // dispatch indexes it. All four Phase 5 model types implement
    // ISimulatable<std::vector<double>>; the dynamic_cast guard mirrors the C++ test
    // runner's and corehydror's.
    m.def(
        "model_simulate",
        [](const std::string& model_json, const std::vector<double>& dataset, int sample_size,
           int seed) {
            std::unique_ptr<models::ModelBase> model =
                models::spec::build_model_from_json(model_json, dataset);
            // simulate_flat handles both ISimulatable<vector<double>> and the bivariate
            // ISimulatable<Matrix2D> (flattened row-major) -- see its header note.
            return simulate_flat(model.get(), sample_size, seed);
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("sample_size"), py::arg("seed"));

    // --- GeneralizedMethodOfMoments (B11) --------------------------------------------------
    //
    // Disjoint construct shape from ML/MAP (a strategy + max_gmm_iterations instead of just an
    // optimizer string) AND a different model type -- the concrete Bulletin17CDistribution the
    // GMM ctor takes as IGMMModel& (NOT a ModelBase; see model_spec.hpp's build_bulletin17c_model
    // note), built here through the shared spec builder's dedicated bulletin17c entry point. So
    // this is a separate registered function, mirroring the estimation_bayes_run split. Builds
    // the model, fits once, post_processes, and returns every value the GMM dispatcher reads:
    //   parameters/standard_errors     -- [p] lists
    //   covariance/correlation         -- [p][p] nested lists
    //   j_stat/j_stat_pval             -- scalars (pval is NaN when q - p == 0)
    //   simulated                      -- [sample_size] seeded ISimulatable draw off the FITTED
    //                                     model (present only when construct supplies sample_size),
    //                                     read by the shared `simulated_value` dispatch arm.
    // quantile_variance rides estimation_gmm_qvar (it needs a per-assertion AEP, exactly like
    // `bic`'s per-assertion sample size -- see below).
    m.def(
        "estimation_gmm_run",
        [](const std::string& model_json, const std::vector<double>& dataset,
           const std::string& strategy, const std::string& optimizer, int max_gmm_iterations,
           int sample_size, int seed) {
            // Same construct-assembly guard as estimation_bayes_run: `max_gmm_iterations` is
            // applied only when positive, which used to be the guard around the setter.
            std::string construct = "{\"model\":" + model_json +
                                    ",\"optimizer\":" + json_string(optimizer) +
                                    ",\"strategy\":" + json_string(strategy);
            append_if_positive(construct, "max_gmm_iterations", max_gmm_iterations);
            construct += "}";

            est::support::FitResult r = est::support::run_fit("GMM", construct, dataset);
            int p = static_cast<int>(r.parameters.size());

            py::dict out;
            out["parameters"] = r.parameters;
            out["standard_errors"] = r.standard_errors;

            std::vector<std::vector<double>> covariance(static_cast<std::size_t>(p));
            std::vector<std::vector<double>> correlation(static_cast<std::size_t>(p));
            for (int i = 0; i < p; ++i) {
                covariance[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(p));
                correlation[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(p));
                for (int j = 0; j < p; ++j) {
                    covariance[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                        r.covariance[static_cast<std::size_t>(i * p + j)];
                    correlation[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                        r.correlation[static_cast<std::size_t>(i * p + j)];
                }
            }
            out["covariance"] = covariance;
            out["correlation"] = correlation;
            out["j_stat"] = r.j_stat;
            out["j_stat_pval"] = r.j_stat_pval;
            // T13: GMMIterations/ConvergedWithinTolerance (off-by-one fix) and
            // OptimizerFallbackCount (sticky BFGS->NelderMead fallback).
            out["gmm_iterations"] = r.gmm_iterations;
            out["converged_within_tolerance"] = r.converged_within_tolerance;
            out["optimizer_fallback_count"] = r.optimizer_fallback_count;

            // Seeded draw off the FITTED B17C model, rebuilt from the runner's `model_spec` (the
            // fitted `parameter_values` are applied by the spec builder through the same
            // set_parameter_values the old in-place pin called).
            if (sample_size > 0) {
                std::unique_ptr<models::Bulletin17CDistribution> fitted =
                    models::spec::build_bulletin17c_from_json(r.model_spec, dataset);
                out["simulated"] = fitted->generate_random_values(sample_size, seed);
            }
            return out;
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("strategy"), py::arg("optimizer"),
        py::arg("max_gmm_iterations"), py::arg("sample_size"), py::arg("seed"));

    // `quantile_variance [aep]`: like `bic [n]`, the AEP is only known at assertion-dispatch
    // time, so this rebuilds the same deterministic fit (see build_and_fit_gmm) and evaluates
    // the B17C delta-method Var(Q_p) live. args[0] is the annual EXCEEDANCE probability; the C#
    // QuantileVariance takes a NON-exceedance probability, so pass 1 - AEP.
    m.def(
        "estimation_gmm_qvar",
        [](const std::string& model_json, const std::vector<double>& dataset,
           const std::string& strategy, const std::string& optimizer, int max_gmm_iterations,
           double aep) {
            std::string construct = "{\"model\":" + model_json +
                                    ",\"optimizer\":" + json_string(optimizer) +
                                    ",\"strategy\":" + json_string(strategy);
            append_if_positive(construct, "max_gmm_iterations", max_gmm_iterations);
            construct += "}";
            // The live rebuild stays (the AEP is only known at assertion-dispatch time);
            // run_fit_quantile_variance IS this function's old body, lifted into fit_runner.hpp.
            return est::support::run_fit_quantile_variance(construct, dataset, aep);
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("strategy"), py::arg("optimizer"),
        py::arg("max_gmm_iterations"), py::arg("aep"));

    // --- Task 8: the user-facing fit surface (fit_mle/fit_map/fit_bayesian/fit_gmm) ---------
    //
    // ONE entry point for all four fit targets, the pybind11 twin of corehydror's `ch_fit_run_`
    // (corehydror/src/estimation.cpp). Unlike `estimation_run`/`estimation_bayes_run`/
    // `estimation_gmm_run` above (which each assemble their own narrow construct from a fixture's
    // flat arguments), `fit.py` assembles the FULL construct JSON itself -- {"model": ...,
    // settings...} -- exactly as corehydror's `fit_input()` does, so this just runs it and packs
    // FitResult's entire surface into one dict; fit.py picks the fields its target populates.
    // Covariance/correlation are reshaped here into nested lists (matching the existing
    // estimation_run/estimation_gmm_run convention above); draws stay flat CHAIN-major alongside
    // chain_dims -- fit.py transposes them into [iteration, chain, parameter] with numpy, the
    // same axis order corehydror returns (see fit.py's `_new_fit_bayesian`).
    m.def(
        "fit_run",
        [](const std::string& target, const std::string& construct_json,
           const std::vector<double>& dataset) {
            est::support::FitResult r = est::support::run_fit(target, construct_json, dataset);
            int n = static_cast<int>(r.parameters.size());

            py::dict out;
            out["method"] = r.method;
            out["parameter_names"] = r.parameter_names;
            out["parameters"] = r.parameters;
            out["log_likelihood"] = r.log_likelihood;
            out["prior_log_likelihood"] = r.prior_log_likelihood;
            out["aic"] = r.aic;
            out["bic"] = r.bic;
            out["nobs"] = r.nobs;
            out["covariance"] = nested_square(r.covariance, n);
            out["standard_errors"] = r.standard_errors;
            out["correlation"] = nested_square(r.correlation, n);
            out["converged"] = r.converged;
            out["status"] = r.status;
            out["function_evaluations"] = r.function_evaluations;
            out["model_spec"] = r.model_spec;
            out["profile_grid"] = r.profile_grid;
            out["profile_lower"] = r.profile_lower;
            out["profile_upper"] = r.profile_upper;
            out["profile_bins"] = r.profile_bins;
            out["draws"] = r.draws;
            out["chain_dims"] = r.chain_dims;
            // --- Bayesian-only fields; empty/NaN for MaximumLikelihood/MaximumAPosteriori/GMM --
            // `posterior` is the THINNED draw block the analyses consume, kept flat row-major
            // alongside `posterior_rows` (fit.py reshapes it to posterior_rows x n_params with
            // numpy, exactly as it reshapes `draws`); the raw chains stay in `draws` above.
            out["posterior"] = r.posterior;
            out["posterior_rows"] = r.posterior_rows;
            out["map"] = r.map;
            out["posterior_mean"] = r.posterior_mean;
            out["mean_log_likelihood"] = r.mean_log_likelihood;
            out["acceptance_rates"] = r.acceptance_rates;
            out["dic"] = r.dic;
            out["waic"] = r.waic;
            out["waic_pd"] = r.waic_pd;
            out["looic"] = r.looic;
            out["looic_se"] = r.looic_se;
            out["loo_pd"] = r.loo_pd;
            out["rhat"] = r.rhat;
            out["ess"] = r.ess;
            out["summary_mean"] = r.summary_mean;
            out["summary_median"] = r.summary_median;
            out["summary_sd"] = r.summary_sd;
            out["summary_lower"] = r.summary_lower;
            out["summary_upper"] = r.summary_upper;
            // --- GMM-only fields; NaN/0/false for the other three targets ----------------------
            out["j_stat"] = r.j_stat;
            out["j_stat_pval"] = r.j_stat_pval;
            out["gmm_iterations"] = r.gmm_iterations;
            out["converged_within_tolerance"] = r.converged_within_tolerance;
            out["optimizer_fallback_count"] = r.optimizer_fallback_count;
            return out;
        },
        py::arg("target"), py::arg("construct_json"), py::arg("dataset"));

    // `fit.diagnostics()`/`fit_diagnostics(fit)`: reruns the fit's own construct (the JSON
    // fit.py's `_fit_input` built, carried on the Fit as `_construct_json`) through
    // `run_fit_diagnostics` -- see that function's header for why an explicit `warmup_iterations`
    // on the ORIGINAL construct is what makes this agree with the fit it is diagnosing. `target`
    // is one of MaximumAPosteriori/BayesianAnalysis/GMM (matching FitDiagnostics's three
    // populated arms); MaximumLikelihood is rejected fit.py-side before this is ever called,
    // mirroring corehydror's `ch_fit_diagnostics_`/`fit_diagnostics()`.
    m.def(
        "fit_diagnostics",
        [](const std::string& target, const std::string& construct_json,
           const std::vector<double>& dataset) {
            est::support::FitDiagnostics d =
                est::support::run_fit_diagnostics(target, construct_json, dataset);

            // observation_influence is n_obs x n_params, row-major (fit_runner.hpp's
            // flatten_influence); n_obs is cooks_distance's length (populated together, MAP/GMM
            // only -- see FitDiagnostics's header), so n_params divides out of the flat length
            // rather than needing a separate field.
            int n_obs = static_cast<int>(d.cooks_distance.size());
            int n_params = (n_obs > 0 && !d.observation_influence.empty())
                               ? static_cast<int>(d.observation_influence.size() /
                                                   static_cast<std::size_t>(n_obs))
                               : 0;

            py::dict out;
            out["cooks_distance"] = d.cooks_distance;
            out["leverage"] = d.leverage;
            out["observation_influence"] = nested_rect(d.observation_influence, n_obs, n_params);
            out["pareto_k"] = d.pareto_k;
            out["max_pareto_k"] = d.max_pareto_k;
            out["prior_influence"] = d.prior_influence;
            out["prior_influence_names"] = d.prior_influence_names;
            return out;
        },
        py::arg("target"), py::arg("construct_json"), py::arg("dataset"));

    // `quantile_variance(fit, aep)`: like `estimation_gmm_qvar` above, rebuilds the same
    // deterministic GMM fit from the fit's own construct and evaluates the B17C delta-method
    // Var(Q_p) live -- `aep` is only known at call time, so this cannot be precomputed into
    // `fit_run`'s result the way every other GMM field is.
    m.def(
        "fit_quantile_variance",
        [](const std::string& construct_json, const std::vector<double>& dataset, double aep) {
            return est::support::run_fit_quantile_variance(construct_json, dataset, aep);
        },
        py::arg("construct_json"), py::arg("dataset"), py::arg("aep"));
}
