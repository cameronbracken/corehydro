// pybind11 glue for the data layer: standalone DataFrame summaries, model log-likelihood, and
// the two threshold-selection diagnostics. Consumed by corehydropy.data and corehydropy.models.
//
// Every entry point delegates to corehydro/models/data_frame_runner.hpp, the shared runner the R
// glue (corehydror/src/data.cpp) and the C++ fixture runner drive identically, so a given spec
// returns the same numbers in every language. Nothing here holds a C++ object across the
// boundary: specs arrive as JSON strings, results leave as plain dicts of lists.
// Core headers are vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "corehydro/models/data_frame_runner.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace runner = corehydro::models::runner;

void register_data(py::module_& m) {
    // Builds a standalone DataFrame from a `data_frame` spec object, runs the threshold and
    // plotting-position cascade, and returns the flattened exact-series view plus the
    // frame-level scalars. Keys match the R list returned by ch_data_frame_summary_.
    m.def("data_frame_summary", [](const std::string& data_frame_json, double plotting_parameter) {
        runner::DataFrameSummary s =
            runner::summarize_data_frame(data_frame_json, plotting_parameter);
        py::dict out;
        out["index"] = s.index;
        out["value"] = s.value;
        out["plotting_position"] = s.plotting_position;
        // Round-trip the 0/1 flags to real Python bools, matching R's logical vector.
        std::vector<bool> flags;
        flags.reserve(s.is_low_outlier.size());
        for (int f : s.is_low_outlier) flags.push_back(f != 0);
        out["is_low_outlier"] = flags;
        out["number_of_low_outliers"] = s.number_of_low_outliers;
        out["low_outlier_threshold"] = s.low_outlier_threshold;
        out["plotting_parameter"] = s.plotting_parameter;
        out["lambda"] = s.lambda;
        out["total_record_length"] = s.total_record_length;
        out["exact_count"] = s.exact_count;
        out["interval_count"] = s.interval_count;
        out["threshold_count"] = s.threshold_count;
        out["uncertain_count"] = s.uncertain_count;
        out["zero_value_relative_frequency"] = s.zero_value_relative_frequency;
        return out;
    });

    // ModelBase::log_likelihood decomposed into its data and prior halves. An empty `params`
    // evaluates at the model's own current parameter values; the returned `parameters` list is
    // the one the model actually saw, which for a mutating model (MixtureModel normalizes
    // weights in place) differs from what was passed in.
    m.def("model_log_likelihood", [](const std::string& model_json,
                                     const std::vector<double>& dataset,
                                     const std::vector<double>& params) {
        runner::ModelLogLikelihood ll = runner::model_log_likelihood(model_json, dataset, params);
        py::dict out;
        out["log_likelihood"] = ll.total;
        out["data_log_likelihood"] = ll.data;
        out["prior_log_likelihood"] = ll.prior;
        out["parameters"] = ll.parameters;
        return out;
    });

    // The two ThresholdDiagnostics statics behind one name-dispatched call. Candidate thresholds
    // with too few exceedances (or a failed GPD fit) are dropped by the core, so the returned
    // lists are generally shorter than `n_thresholds`; the columns the chosen method does not
    // populate come back empty.
    m.def("threshold_diagnostics", [](const std::vector<double>& data, const std::string& method,
                                      double u_min, double u_max, int n_thresholds,
                                      double confidence_level) {
        runner::ThresholdDiagnosticsResult r = runner::run_threshold_diagnostics(
            data, method, u_min, u_max, n_thresholds, confidence_level);
        py::dict out;
        out["threshold"] = r.threshold;
        out["exceedance_count"] = r.exceedance_count;
        out["mean_excess"] = r.mean_excess;
        out["lower_ci"] = r.lower_ci;
        out["upper_ci"] = r.upper_ci;
        out["modified_scale"] = r.modified_scale;
        out["modified_scale_lower_ci"] = r.modified_scale_lower_ci;
        out["modified_scale_upper_ci"] = r.modified_scale_upper_ci;
        out["shape"] = r.shape;
        out["shape_lower_ci"] = r.shape_lower_ci;
        out["shape_upper_ci"] = r.shape_upper_ci;
        return out;
    });

    // The twelve DataFrame hypothesis-test / summary-statistics facades (P4 Task 6), behind one
    // name-dispatched call: the nine hypothesis tests, summary_exact, summary_all, and
    // standardized. `data` holds the exact-series values (data[0] becomes the exact series with
    // sequential 0-based indexes); `data_frame_json`, when non-empty, instead builds the frame
    // from a `data_frame` spec object and `data` must be empty -- exactly one of the two is
    // required (see data_frame_runner.hpp's run_data_frame for why there are two paths). Packed
    // exactly as toolbox_run's `pack()` does: values/names/dims/spec.
    m.def(
        "data_frame_run",
        [](const std::string& method, const std::vector<std::vector<double>>& data,
           const std::string& data_frame_json, const std::string& options_json) {
            corehydro::numerics::support::ToolboxResult r =
                runner::run_data_frame(method, data, data_frame_json, options_json);
            py::dict out;
            out["values"] = r.values;
            out["names"] = r.names;
            out["dims"] = r.dims;
            out["spec"] = r.spec;
            return out;
        },
        py::arg("method"), py::arg("data"), py::arg("data_frame_json"), py::arg("options_json"));
}
