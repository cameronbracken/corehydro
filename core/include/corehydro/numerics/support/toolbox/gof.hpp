// corehydro ADDITION -- toolbox_runner.hpp group header, no upstream C# counterpart.
//
// Holds the `gof` group's dispatch arm: information criteria (aic/aicc/bic), distribution-backed
// tests (ks/ad/chi_squared/rmse_dist) built off a model spec via options_model, classification
// metrics, and the continuous simulated-vs-observed metrics (rmse/mse/mae/... and the combined
// "metrics" call). Included by toolbox_runner.hpp, which defines the shared
// ToolboxResult/data_at/scalar helpers used here; not meant to be included directly.
#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/distributions/support/dist_spec.hpp"

namespace corehydro::numerics::support::detail {

// Builds the model a distribution-backed goodness-of-fit method needs from the options blob:
// {"model": {"family": ..., "parameters": [...]}} in the dist_spec.hpp grammar, so a truncated
// or mixture model works here exactly as it does in dist_pdf().
inline std::unique_ptr<distributions::UnivariateDistributionBase> options_model(
    const JsonValue& options, const std::string& method) {
    if (!options.contains("model"))
        throw std::runtime_error("toolbox method 'gof." + method +
                                 "' needs a 'model' distribution spec in its options");
    return distributions::support::build_univariate(options.at("model"));
}

inline ToolboxResult run_gof(const std::string& method,
                             const std::vector<std::vector<double>>& data,
                             const JsonValue& options) {
    using GOF = numerics::data::GoodnessOfFit;

    // Information criteria: scalars only, all from options.
    if (method == "aic")
        return scalar(GOF::aic(options.at("k").as_int(), options.at("log_likelihood").as_double()));
    if (method == "aicc")
        return scalar(GOF::aicc(options.at("n").as_int(), options.at("k").as_int(),
                                options.at("log_likelihood").as_double()));
    if (method == "bic")
        return scalar(GOF::bic(options.at("n").as_int(), options.at("k").as_int(),
                               options.at("log_likelihood").as_double()));
    if (method == "aic_weights") {
        ToolboxResult r;
        r.values = GOF::aic_weights(data_at(data, 0, "gof", method));
        return r;
    }
    if (method == "rmse_weights") {
        ToolboxResult r;
        r.values = GOF::rmse_weights(data_at(data, 0, "gof", method));
        return r;
    }

    // Distribution-backed methods: one series plus a model spec.
    if (method == "ks" || method == "ad" || method == "chi_squared" || method == "rmse_dist") {
        std::vector<double> obs = data_at(data, 0, "gof", method);
        std::sort(obs.begin(), obs.end());  // every one of these requires ascending input
        std::unique_ptr<distributions::UnivariateDistributionBase> m = options_model(options, method);
        if (method == "ks") return scalar(GOF::kolmogorov_smirnov(obs, *m));
        if (method == "ad") return scalar(GOF::anderson_darling(obs, *m));
        if (method == "chi_squared") return scalar(GOF::chi_squared(obs, *m));
        if (data.size() > 1) return scalar(GOF::rmse(obs, data[1], *m));
        return scalar(GOF::rmse(obs, *m));
    }

    // Classification metrics: two already-binary label vectors, compared elementwise. C#'s
    // classification statics (and this port) take no threshold -- a value equal to its
    // counterpart counts as a match -- and ConfusionMatrix stays private in the port, so it is
    // never surfaced here. Callers threshold their own series before calling this.
    if (method == "classification") {
        const std::vector<double>& o = data_at(data, 0, "gof", method);
        const std::vector<double>& p = data_at(data, 1, "gof", method);
        ToolboxResult r;
        r.values = {GOF::accuracy(o, p),    GOF::precision(o, p),  GOF::recall(o, p),
                    GOF::f1_score(o, p),    GOF::specificity(o, p), GOF::balanced_accuracy(o, p)};
        r.names = {"accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy"};
        return r;
    }

    // Continuous metrics: two series, optional k for the RMSE denominator. Each metric is
    // dispatched individually rather than through one shared eagerly-evaluated list -- MAPE is
    // undefined when `observed` contains a zero, and a caller asking for `mse` on that same
    // series must not fail because of it. Only "metrics" (every metric at once) accepts that
    // an unsuitable series makes the whole set unavailable, exactly as a direct MAPE call would.
    const std::vector<double>& o = data_at(data, 0, "gof", method);
    const std::vector<double>& m = data_at(data, 1, "gof", method);
    int k = options.value_or("k", 0);
    if (method == "metrics") {
        ToolboxResult r;
        r.names = {"rmse", "mse", "mae", "mape", "smape", "nse", "log_nse", "kge", "kge_mod",
                   "pbias", "rsr", "pearson", "r_squared", "d", "d_mod", "d_ref", "ve"};
        r.values = {GOF::rmse(o, m, k),
                    GOF::mse(o, m),
                    GOF::mae(o, m),
                    GOF::mape(o, m),
                    GOF::smape(o, m),
                    GOF::nash_sutcliffe_efficiency(o, m),
                    GOF::log_nash_sutcliffe_efficiency(o, m),
                    GOF::kling_gupta_efficiency(o, m),
                    GOF::kling_gupta_efficiency_mod(o, m),
                    GOF::pbias(o, m),
                    GOF::rsr(o, m),
                    GOF::pearson(o, m),
                    GOF::r_squared(o, m),
                    GOF::index_of_agreement(o, m),
                    GOF::modified_index_of_agreement(o, m),
                    GOF::refined_index_of_agreement(o, m),
                    GOF::volumetric_efficiency(o, m)};
        return r;
    }
    if (method == "rmse") return scalar(GOF::rmse(o, m, k));
    if (method == "mse") return scalar(GOF::mse(o, m));
    if (method == "mae") return scalar(GOF::mae(o, m));
    if (method == "mape") return scalar(GOF::mape(o, m));
    if (method == "smape") return scalar(GOF::smape(o, m));
    if (method == "nse") return scalar(GOF::nash_sutcliffe_efficiency(o, m));
    if (method == "log_nse") return scalar(GOF::log_nash_sutcliffe_efficiency(o, m));
    if (method == "kge") return scalar(GOF::kling_gupta_efficiency(o, m));
    if (method == "kge_mod") return scalar(GOF::kling_gupta_efficiency_mod(o, m));
    if (method == "pbias") return scalar(GOF::pbias(o, m));
    if (method == "rsr") return scalar(GOF::rsr(o, m));
    if (method == "pearson") return scalar(GOF::pearson(o, m));
    if (method == "r_squared") return scalar(GOF::r_squared(o, m));
    if (method == "d") return scalar(GOF::index_of_agreement(o, m));
    if (method == "d_mod") return scalar(GOF::modified_index_of_agreement(o, m));
    if (method == "d_ref") return scalar(GOF::refined_index_of_agreement(o, m));
    if (method == "ve") return scalar(GOF::volumetric_efficiency(o, m));
    throw std::runtime_error("unknown gof method: " + method);
}

}  // namespace corehydro::numerics::support::detail
