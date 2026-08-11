// corehydro ADDITION -- no upstream C# counterpart.
//
// The shared data-layer runner, the sibling of models/model_spec.hpp and
// analyses/support/analysis_runner.hpp. It exists for the same reason those do: the R glue
// (corehydror/src/data.cpp), the Python glue (corehydropy/src/bindings/data.cpp), and the C++
// fixture runner must drive the DataFrame surface through ONE code path so a given spec yields
// byte-identical numbers in every language.
//
// Three entry points, matching the three data-layer verbs the packages expose:
//
//   summarize_data_frame(json)      -- build a standalone DataFrame from a `data_frame` spec
//                                      object (the same grammar model_spec.hpp parses inside a
//                                      model), run the threshold + plotting-position cascade,
//                                      and flatten what a caller needs to plot or inspect.
//   model_log_likelihood(...)       -- ModelBase::log_likelihood at a caller-supplied parameter
//                                      vector, or at the model's own current values.
//   run_threshold_diagnostics(...)  -- the two ThresholdDiagnostics statics behind one
//                                      name-dispatched call, flattened to parallel vectors.
//
// Everything here returns plain structs of vectors and scalars: the packages hold no C++ object,
// so a result crosses the boundary as data and nothing needs a finalizer.
#pragma once
#include <string>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/data_frame/threshold_diagnostics.hpp"
#include "corehydro/models/json_lite.hpp"
#include "corehydro/models/model_spec.hpp"
#include "corehydro/models/support/model_base.hpp"

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

}  // namespace corehydro::models::runner
