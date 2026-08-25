// corehydro ADDITION -- no upstream C# counterpart.
//
// The shared data-layer runner, the sibling of models/model_spec.hpp and
// analyses/support/analysis_runner.hpp. It exists for the same reason those do: the R glue
// (corehydror/src/data.cpp), the Python glue (corehydropy/src/bindings/data.cpp), and the C++
// fixture runner must drive the DataFrame surface through ONE code path so a given spec yields
// byte-identical numbers in every language.
//
// Four entry points, matching the four data-layer verbs the packages expose:
//
//   summarize_data_frame(json)      -- build a standalone DataFrame from a `data_frame` spec
//                                      object (the same grammar model_spec.hpp parses inside a
//                                      model), run the threshold + plotting-position cascade,
//                                      and flatten what a caller needs to plot or inspect.
//   model_log_likelihood(...)       -- ModelBase::log_likelihood at a caller-supplied parameter
//                                      vector, or at the model's own current values.
//   run_threshold_diagnostics(...)  -- the two ThresholdDiagnostics statics behind one
//                                      name-dispatched call, flattened to parallel vectors.
//   run_data_frame(...)             -- the nine hypothesis-test facades, the two summary-
//                                      statistics facades, and the standardized-value facade
//                                      (P4 Task 6), behind one name-dispatched call. Returns the
//                                      SAME numerics::support::ToolboxResult the toolbox groups
//                                      return, so the three fixture runners' existing
//                                      `toolbox_select` helper (index / label / select:
//                                      "length"/"rows"/"columns") is reused verbatim -- no new
//                                      selection code exists anywhere for this kind.
//
// Everything here returns plain structs of vectors and scalars: the packages hold no C++ object,
// so a result crosses the boundary as data and nothing needs a finalizer.
#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/data_frame/threshold_diagnostics.hpp"
#include "corehydro/models/json_lite.hpp"
#include "corehydro/models/model_spec.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::models::runner {

// --- Data-frame summary ---------------------------------------------------------------------

// The flattened DataFrame view the packages expose. `index` / `value` / `plotting_position` /
// `is_low_outlier` are parallel over the EXACT series only (the systematic record, which is what
// a probability plot draws); the censored series contribute to the plotting positions through
// the Hirsch-Stedinger computation but have no single plotting ordinate of their own.
struct DataFrameSummary {
    std::vector<int> index;
    std::vector<double> value;
    std::vector<double> plotting_position;
    std::vector<int> is_low_outlier;

    int number_of_low_outliers = 0;
    double low_outlier_threshold = 0.0;
    double plotting_parameter = 0.0;
    double lambda = 0.0;
    int total_record_length = 0;
    int exact_count = 0;
    int interval_count = 0;
    int threshold_count = 0;
    int uncertain_count = 0;
    double zero_value_relative_frequency = 0.0;
};

// Builds a standalone DataFrame from a `data_frame` spec object. Shares
// models::spec::build_data_frame, so the exact / interval / threshold / uncertain grammar, the
// `low_outlier_threshold` key, and the `mgbt_low_outliers` trigger behave identically whether
// the frame arrives on its own or nested inside a model spec.
inline DataFrame build_data_frame_from_json(const std::string& data_frame_json) {
    return spec::build_data_frame(spec::parse_json(data_frame_json));
}

// Runs the frame's own cascade and flattens it. `plotting_parameter` follows the C# property
// setter contract (finite, in [0, 1)); it is applied BEFORE the plotting positions are computed
// so the requested convention is the one that lands. calculate_plotting_positions() calls
// process_threshold_series() itself, exactly like the C#.
inline DataFrameSummary summarize_data_frame(const std::string& data_frame_json,
                                             double plotting_parameter) {
    DataFrame df = build_data_frame_from_json(data_frame_json);
    df.set_plotting_parameter(plotting_parameter);
    df.calculate_plotting_positions();
    df.calculate_lambda();

    DataFrameSummary out;
    const ExactSeries& exact = df.exact_series();
    for (std::size_t i = 0; i < exact.count(); ++i) {
        out.index.push_back(exact[i].index());
        out.value.push_back(exact[i].value());
        out.plotting_position.push_back(exact[i].plotting_position());
        out.is_low_outlier.push_back(exact[i].is_low_outlier() ? 1 : 0);
    }

    out.number_of_low_outliers = df.number_of_low_outliers();
    out.low_outlier_threshold = df.low_outlier_threshold();
    out.plotting_parameter = df.plotting_parameter();
    out.lambda = df.lambda();
    out.total_record_length = df.total_record_length();
    out.exact_count = static_cast<int>(exact.count());
    out.interval_count = static_cast<int>(df.interval_series().count());
    out.threshold_count = static_cast<int>(df.threshold_series().count());
    out.uncertain_count = static_cast<int>(df.uncertain_series().count());
    out.zero_value_relative_frequency = df.zero_value_relative_frequency();
    return out;
}

// --- Model log-likelihood -------------------------------------------------------------------

// The decomposed likelihood at one parameter vector. Data-then-prior evaluation order is the
// model's own (see the MUTABLE-PARAMETER note in models/support/model_base.hpp): a mutating
// model such as MixtureModel writes normalized weights back into `p`, and the prior must see
// that write-back, so `total` is taken from log_likelihood() rather than summed here.
struct ModelLogLikelihood {
    double total = 0.0;
    double data = 0.0;
    double prior = 0.0;
    std::vector<double> parameters;
};

// Evaluates the model at `params`, or at the model's own current parameter values when `params`
// is empty. Returns the parameter vector actually used, which is what the model left behind
// after any write-back.
inline ModelLogLikelihood model_log_likelihood(const std::string& model_json,
                                               const std::vector<double>& dataset,
                                               const std::vector<double>& params) {
    std::unique_ptr<ModelBase> model = spec::build_model_from_json(model_json, dataset);

    std::vector<double> p = params;
    if (p.empty()) {
        p.reserve(model->parameters().size());
        for (const ModelParameter& mp : model->parameters()) p.push_back(mp.value());
    }
    if (p.size() != model->parameters().size())
        throw std::invalid_argument("The list of parameter values are the wrong length");

    ModelLogLikelihood out;
    // One log_likelihood() call, then the two parts off the SAME post-write-back vector, so a
    // mutating model is evaluated exactly once and the components stay consistent with `total`.
    out.total = model->log_likelihood(p);
    out.data = model->data_log_likelihood(p);
    out.prior = model->prior_log_likelihood(p);
    out.parameters = p;
    return out;
}

// --- Threshold diagnostics ------------------------------------------------------------------

// Both diagnostics flattened to parallel vectors. Which vectors are populated depends on the
// method: mean_residual_life fills `mean_excess` / `lower_ci` / `upper_ci`, parameter_stability
// fills the modified-scale and shape columns. `threshold` and `exceedance_count` are always
// filled, and both methods drop candidate thresholds with too few exceedances (5 for MRL, 10 for
// stability) or a failed MLE, so the vectors are generally shorter than `n_thresholds`.
struct ThresholdDiagnosticsResult {
    std::vector<double> threshold;
    std::vector<int> exceedance_count;

    std::vector<double> mean_excess;
    std::vector<double> lower_ci;
    std::vector<double> upper_ci;

    std::vector<double> modified_scale;
    std::vector<double> modified_scale_lower_ci;
    std::vector<double> modified_scale_upper_ci;
    std::vector<double> shape;
    std::vector<double> shape_lower_ci;
    std::vector<double> shape_upper_ci;
};

inline ThresholdDiagnosticsResult run_threshold_diagnostics(const std::vector<double>& data,
                                                            const std::string& method,
                                                            double u_min, double u_max,
                                                            int n_thresholds,
                                                            double confidence_level) {
    ThresholdDiagnosticsResult out;
    if (method == "mean_residual_life") {
        MeanResidualLifeResult r = ThresholdDiagnostics::compute_mean_residual_life(
            data, u_min, u_max, n_thresholds, confidence_level);
        for (const MRLPoint& pt : r.points) {
            out.threshold.push_back(pt.threshold);
            out.exceedance_count.push_back(pt.exceedance_count);
            out.mean_excess.push_back(pt.mean_excess);
            out.lower_ci.push_back(pt.lower_ci);
            out.upper_ci.push_back(pt.upper_ci);
        }
        return out;
    }
    if (method == "parameter_stability") {
        ParameterStabilityResult r = ThresholdDiagnostics::compute_parameter_stability(
            data, u_min, u_max, n_thresholds, confidence_level);
        for (const StabilityPoint& pt : r.points) {
            out.threshold.push_back(pt.threshold);
            out.exceedance_count.push_back(pt.exceedance_count);
            out.modified_scale.push_back(pt.modified_scale);
            out.modified_scale_lower_ci.push_back(pt.modified_scale_lower_ci);
            out.modified_scale_upper_ci.push_back(pt.modified_scale_upper_ci);
            out.shape.push_back(pt.shape);
            out.shape_lower_ci.push_back(pt.shape_lower_ci);
            out.shape_upper_ci.push_back(pt.shape_upper_ci);
        }
        return out;
    }
    throw std::invalid_argument(
        "unknown threshold diagnostics method '" + method +
        "'; expected 'mean_residual_life' or 'parameter_stability'");
}

// --- Hypothesis-test and summary-statistics facades (P4 Task 6) ----------------------------
//
// Dispatches the twelve DataFrame facades Task 5 un-gated: the nine hypothesis tests
// (jarque_bera / ljung_box / equal_variance_t / unequal_variance_t / f / linear_trend /
// wald_wolfowitz / mann_whitney / mann_kendall), the two summary-statistics methods
// (summary_exact / summary_all), and standardized (set_standardized_values(), read back off the
// exact series). Every one of these reads the EXACT series only -- no censoring filter, no
// threshold machinery -- so build_data_frame_from_json is not used here; instead this takes
// EITHER a flat `data` vector (data[0] becomes the exact series with sequential 0-based indexes,
// exactly the ExactSeries(std::vector<double>) ctor) OR a `data_frame_json` spec object (passed
// to models::spec::build_data_frame unchanged, so a censored frame -- irrelevant series and all
// -- is still reachable). Exactly one of the two must be supplied; requiring that here, in the
// one shared runner, is what keeps the three fixture runners and the emitter from each growing
// their own copy of the "which path did the caller mean" logic. The plain-vector path is what
// every P4 fixture case and both user-facing verbs (analysis_data_hypothesis_test /
// analysis_data_statistics) use; the spec path exists so a censored frame is still reachable
// through this entry point too, for symmetry with build_data_frame_from_json above.
//
// calculate_plotting_positions() is called before summary_all and standardized (the C# tests do
// it explicitly and both methods read plotting-position complements, so the results are
// meaningless without it) and is NOT called for the nine hypothesis facades or summary_exact,
// none of which read plotting positions.
inline numerics::support::ToolboxResult run_data_frame(const std::string& method,
                                                        const std::vector<std::vector<double>>& data,
                                                        const std::string& data_frame_json,
                                                        const std::string& options_json) {
    using numerics::support::ToolboxResult;

    bool has_json = !data_frame_json.empty();
    bool has_data = !data.empty();
    if (has_json == has_data)
        throw std::invalid_argument(
            "run_data_frame requires exactly one of `data` or `data_frame_json`");

    DataFrame df;
    if (has_json) {
        df = build_data_frame_from_json(data_frame_json);
    } else {
        if (data[0].empty())
            throw std::invalid_argument("run_data_frame requires a non-empty data series");
        df.set_exact_series(ExactSeries(data[0]));
    }

    spec::JsonValue options = spec::parse_json(options_json.empty() ? "{}" : options_json);
    bool use_log10 = options.value_or("use_log10", false);

    static const std::vector<std::string> one_sample_methods = {
        "jarque_bera", "ljung_box", "linear_trend", "wald_wolfowitz", "mann_kendall"};
    static const std::vector<std::string> two_sample_methods = {
        "equal_variance_t", "unequal_variance_t", "f", "mann_whitney"};

    bool is_one_sample =
        std::find(one_sample_methods.begin(), one_sample_methods.end(), method) !=
        one_sample_methods.end();
    bool is_two_sample =
        std::find(two_sample_methods.begin(), two_sample_methods.end(), method) !=
        two_sample_methods.end();

    if (is_one_sample || is_two_sample) {
        double p = 0.0;
        if (method == "jarque_bera") {
            p = df.jarque_bera_test(use_log10);
        } else if (method == "ljung_box") {
            p = df.ljung_box_test(options.value_or("lag_max", -1), use_log10);
        } else if (method == "linear_trend") {
            p = df.linear_trend_test(use_log10);
        } else if (method == "wald_wolfowitz") {
            p = df.wald_wolfowitz_test(use_log10);
        } else if (method == "mann_kendall") {
            p = df.mann_kendall_test(use_log10);
        } else {
            if (!options.contains("index"))
                throw std::invalid_argument("data_frame method '" + method +
                                            "' requires an 'index' option");
            int index = options.at("index").as_int();
            if (method == "equal_variance_t") {
                p = df.equal_variance_t_test(index, use_log10);
            } else if (method == "unequal_variance_t") {
                p = df.unequal_variance_t_test(index, use_log10);
            } else if (method == "f") {
                p = df.f_test(index, use_log10);
            } else {  // mann_whitney
                p = df.mann_whitney_test(index, use_log10);
            }
        }
        ToolboxResult r;
        r.values = {p};
        return r;
    }

    if (method == "summary_exact") {
        ToolboxResult r;
        for (const auto& [key, value] : df.summary_statistics_exact_data_only()) {
            r.names.push_back(key);
            r.values.push_back(value);
        }
        return r;
    }

    if (method == "summary_all") {
        df.calculate_plotting_positions();
        ToolboxResult r;
        for (const auto& [key, value] : df.summary_statistics_all_data()) {
            r.names.push_back(key);
            r.values.push_back(value);
        }
        return r;
    }

    if (method == "standardized") {
        df.calculate_plotting_positions();
        df.set_standardized_values();
        ToolboxResult r;
        const ExactSeries& exact = df.exact_series();
        r.values.reserve(exact.count() * 2);
        for (std::size_t i = 0; i < exact.count(); ++i) {
            r.values.push_back(exact[i].standardized_value());
            r.values.push_back(exact[i].standardized_log10_value());
        }
        r.dims = {static_cast<int>(exact.count()), 2};
        return r;
    }

    throw std::invalid_argument("unknown data_frame method: " + method);
}

}  // namespace corehydro::models::runner
