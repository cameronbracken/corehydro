// cpp11 glue for the data layer: standalone DataFrame summaries, model log-likelihood, and
// the two threshold-selection diagnostics. Consumed by R/data.R and R/model.R.
//
// Every entry point delegates to corehydro/models/data_frame_runner.hpp, the shared runner the
// Python bindings and the C++ fixture runner drive identically, so a given spec returns the same
// numbers in every language. Nothing here holds a C++ object across the boundary: specs arrive
// as JSON strings, results leave as plain vectors.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <string>
#include <vector>

#include "corehydro/models/data_frame_runner.hpp"

namespace runner = corehydro::models::runner;
using namespace cpp11;

static std::vector<double> to_vec(doubles x) {
    return std::vector<double>(x.begin(), x.end());
}

static writable::doubles from_vec(const std::vector<double>& v) {
    writable::doubles out(static_cast<R_xlen_t>(v.size()));
    for (std::size_t i = 0; i < v.size(); ++i) out[static_cast<R_xlen_t>(i)] = v[i];
    return out;
}

static writable::integers from_ivec(const std::vector<int>& v) {
    writable::integers out(static_cast<R_xlen_t>(v.size()));
    for (std::size_t i = 0; i < v.size(); ++i) out[static_cast<R_xlen_t>(i)] = v[i];
    return out;
}

static writable::logicals from_flags(const std::vector<int>& v) {
    writable::logicals out(static_cast<R_xlen_t>(v.size()));
    for (std::size_t i = 0; i < v.size(); ++i) out[static_cast<R_xlen_t>(i)] = r_bool(v[i] != 0);
    return out;
}

// Builds a standalone DataFrame from a `data_frame` spec object, runs the threshold and
// plotting-position cascade, and returns the flattened exact-series view plus the frame-level
// scalars. The censored series contribute to the Hirsch-Stedinger positions but carry no
// plotting ordinate of their own, so only their counts come back.
[[cpp11::register]]
list ch_data_frame_summary_(std::string data_frame_json, double plotting_parameter) {
    runner::DataFrameSummary s = runner::summarize_data_frame(data_frame_json, plotting_parameter);
    return writable::list({
        "index"_nm = from_ivec(s.index),
        "value"_nm = from_vec(s.value),
        "plotting_position"_nm = from_vec(s.plotting_position),
        "is_low_outlier"_nm = from_flags(s.is_low_outlier),
        "number_of_low_outliers"_nm = writable::integers({s.number_of_low_outliers}),
        "low_outlier_threshold"_nm = writable::doubles({s.low_outlier_threshold}),
        "plotting_parameter"_nm = writable::doubles({s.plotting_parameter}),
        "lambda"_nm = writable::doubles({s.lambda}),
        "total_record_length"_nm = writable::integers({s.total_record_length}),
        "exact_count"_nm = writable::integers({s.exact_count}),
        "interval_count"_nm = writable::integers({s.interval_count}),
        "threshold_count"_nm = writable::integers({s.threshold_count}),
        "uncertain_count"_nm = writable::integers({s.uncertain_count}),
        "zero_value_relative_frequency"_nm =
            writable::doubles({s.zero_value_relative_frequency}),
    });
}

// ModelBase::log_likelihood decomposed into its data and prior halves. An empty `params`
// evaluates at the model's own current parameter values; the returned `parameters` vector is the
// one the model actually saw, which for a mutating model (MixtureModel normalizes weights in
// place) differs from what was passed in.
[[cpp11::register]]
list ch_model_log_likelihood_(std::string model_json, doubles dataset, doubles params) {
    runner::ModelLogLikelihood ll =
        runner::model_log_likelihood(model_json, to_vec(dataset), to_vec(params));
    return writable::list({
        "log_likelihood"_nm = writable::doubles({ll.total}),
        "data_log_likelihood"_nm = writable::doubles({ll.data}),
        "prior_log_likelihood"_nm = writable::doubles({ll.prior}),
        "parameters"_nm = from_vec(ll.parameters),
    });
}

// The two ThresholdDiagnostics statics behind one name-dispatched call. Candidate thresholds
// with too few exceedances (or a failed GPD fit) are dropped by the core, so the returned
// vectors are generally shorter than `n_thresholds`; the columns the chosen method does not
// populate come back empty.
[[cpp11::register]]
list ch_threshold_diagnostics_(doubles data, std::string method, double u_min, double u_max,
                               int n_thresholds, double confidence_level) {
    runner::ThresholdDiagnosticsResult r = runner::run_threshold_diagnostics(
        to_vec(data), method, u_min, u_max, n_thresholds, confidence_level);
    return writable::list({
        "threshold"_nm = from_vec(r.threshold),
        "exceedance_count"_nm = from_ivec(r.exceedance_count),
        "mean_excess"_nm = from_vec(r.mean_excess),
        "lower_ci"_nm = from_vec(r.lower_ci),
        "upper_ci"_nm = from_vec(r.upper_ci),
        "modified_scale"_nm = from_vec(r.modified_scale),
        "modified_scale_lower_ci"_nm = from_vec(r.modified_scale_lower_ci),
        "modified_scale_upper_ci"_nm = from_vec(r.modified_scale_upper_ci),
        "shape"_nm = from_vec(r.shape),
        "shape_lower_ci"_nm = from_vec(r.shape_lower_ci),
        "shape_upper_ci"_nm = from_vec(r.shape_upper_ci),
    });
}
