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

#include "corehydro/estimation/maximum_a_posteriori.hpp"
#include "corehydro/estimation/maximum_likelihood.hpp"
#include "corehydro/estimation/optimization_method.hpp"
#include "corehydro/models/json_lite.hpp"
#include "corehydro/models/model_spec.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/numerics/math/optimization/support/optimization_status.hpp"

namespace corehydro::estimation::support {

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

// Parameter display names, falling back to p1..pn when the model leaves them empty (some
// ported models do -- see the PriorInfluenceDiagnostics note in docs/upstream-csharp-issues.md).
inline std::vector<std::string> parameter_names_of(const corehydro::models::ModelBase& model) {
    std::vector<std::string> names;
    for (int i = 0; i < model.number_of_parameters(); ++i) {
        const std::string& n = model.parameters()[static_cast<std::size_t>(i)].display_name();
        names.push_back(n.empty() ? ("p" + std::to_string(i + 1)) : n);
    }
    return names;
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

inline FitResult run_fit(const std::string& target, const std::string& construct_json,
                         const std::vector<double>& dataset) {
    corehydro::models::spec::JsonValue construct =
        corehydro::models::spec::parse_json(construct_json);
    const auto& model_json = construct.at("model");
    std::string model_text = corehydro::models::spec::to_json_string(model_json);
    bool want_hessian = construct.value_or("hessian", true);
    std::string optimizer = construct.value_or("optimizer", "DifferentialEvolution");

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
        } else {
            MaximumAPosteriori e(*model, method);
            e.set_compute_hessian(want_hessian);
            if (!e.estimate())
                throw std::runtime_error("MaximumAPosteriori::estimate() failed with optimizer " +
                                         optimizer);
            fill_common(r, e, *model, want_hessian);
        }
        r.model_spec = fitted_spec(model_json, r.parameters);
        return r;
    }
    throw std::runtime_error("unknown fit target: " + target);
}

}  // namespace corehydro::estimation::support
