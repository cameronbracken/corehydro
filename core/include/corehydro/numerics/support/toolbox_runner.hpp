// corehydro ADDITION -- no upstream C# counterpart (sibling of
// distributions/support/dist_runner.hpp and estimation/support/fit_runner.hpp).
//
// The single place a Numerics utility method is dispatched in this repo. Four callers drive it
// and none owns any evaluation logic: the cpp11 glue (corehydror/src/toolbox.cpp), the pybind11
// glue (corehydropy/src/bindings/toolbox.cpp), the C++ fixture runner (core/tests/
// test_fixtures.cpp), and the dotnet oracle emitter (tools/oracle_emitter/Program.cs), which
// reads the same GRAMMAR against the real C# statics. A fixture case, an oracle replay, and a
// user's correlation() call are the same code path.
//
// Bulk data travels as native double vectors, not JSON: a goodness-of-fit call carries two
// series of arbitrary length and paying a JSON parse for them would be pointless. Scalars,
// enum names, and flags travel in `options_json`.
//
// Stateless by construction: one call builds whatever it needs, evaluates once, and drops it.
#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/data/autocorrelation.hpp"
#include "corehydro/numerics/data/correlation.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/running_covariance_matrix.hpp"
#include "corehydro/numerics/data/running_statistics.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/support/dist_spec.hpp"
#include "corehydro/numerics/math/fourier/fourier.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"

namespace corehydro::numerics::support {

using corehydro::models::spec::JsonValue;

// Flat result surface every binding and every fixture assertion reads. `values` holds whatever
// the method returns, in method order; `names` labels them when the method returns a named set;
// `dims` carries {rows, columns} when the method returns a matrix flattened row-major into
// `values`, and is empty otherwise; `spec` carries a child object back and is empty otherwise.
struct ToolboxResult {
    std::vector<double> values;
    std::vector<std::string> names;
    std::vector<int> dims;
    std::string spec;
};

namespace detail {

inline const std::vector<double>& data_at(const std::vector<std::vector<double>>& data,
                                          std::size_t i, const std::string& group,
                                          const std::string& method) {
    if (i >= data.size())
        throw std::runtime_error("toolbox method '" + group + "." + method + "' needs " +
                                 std::to_string(i + 1) + " data vector(s), got " +
                                 std::to_string(data.size()));
    return data[i];
}

inline ToolboxResult scalar(double v) {
    ToolboxResult r;
    r.values = {v};
    return r;
}

// --- group arms -------------------------------------------------------------------------
// One function per group. Later tasks add arms beside this one; nothing else in the file
// changes when a group is added except the dispatch table at the bottom.

inline ToolboxResult run_correlation(const std::string& method,
                                     const std::vector<std::vector<double>>& data,
                                     const JsonValue& options) {
    (void)options;
    const std::vector<double>& x = data_at(data, 0, "correlation", method);
    const std::vector<double>& y = data_at(data, 1, "correlation", method);
    if (method == "pearson") return scalar(numerics::data::pearson(x, y));
    if (method == "spearman") return scalar(numerics::data::spearman(x, y));
    if (method == "kendall") return scalar(numerics::data::kendalls_tau(x, y));
    throw std::runtime_error("unknown correlation method: " + method);
}

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

// Flattens a Matrix row-major into a plain vector<double> -- the shape run_running_covariance
// below needs to pack a mean vector and several size x size matrices into one ToolboxResult, and
// the shape the "state" option unpacks them back out of on the next call.
inline std::vector<double> flatten_matrix(const math::linalg::Matrix& m) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(m.number_of_rows() * m.number_of_columns()));
    for (int i = 0; i < m.number_of_rows(); ++i)
        for (int j = 0; j < m.number_of_columns(); ++j) out.push_back(m(i, j));
    return out;
}

inline ToolboxResult run_statistics(const std::string& method,
                                    const std::vector<std::vector<double>>& data,
                                    const JsonValue& options) {
    namespace nd = numerics::data;
    const std::vector<double>& x = data_at(data, 0, "statistics", method);

    if (method == "summary" || method == "running") {
        // "running" seeds the accumulator from a prior state carried in the options; "summary"
        // is the same call with no prior state. One code path, so a chunked run and a one-shot
        // run can never disagree.
        nd::RunningStatistics rs;
        if (method == "running" && options.contains("state")) {
            const JsonValue& s = options.at("state");
            rs = nd::RunningStatistics::from_state(
                s.at("n").as_int(), s.at("m1").as_double(), s.at("m2").as_double(),
                s.at("m3").as_double(), s.at("m4").as_double(), s.at("minimum").as_double(),
                s.at("maximum").as_double());
        }
        rs.push(x);
        ToolboxResult r;
        r.values = {static_cast<double>(rs.count()), rs.minimum(), rs.maximum(), rs.mean(),
                    rs.variance(), rs.standard_deviation(), rs.coefficient_of_variation(),
                    rs.skewness(), rs.kurtosis(), rs.m1_state(), rs.m2_state(), rs.m3_state(),
                    rs.m4_state()};
        r.names = {"n", "minimum", "maximum", "mean", "variance", "sd", "cv", "skewness",
                   "kurtosis", "m1", "m2", "m3", "m4"};
        return r;
    }
    if (method == "product_moments") {
        ToolboxResult r;
        r.values = nd::product_moments(x);
        r.names = {"mean", "sd", "skewness", "kurtosis"};
        return r;
    }
    if (method == "l_moments") {
        ToolboxResult r;
        r.values = nd::linear_moments(x);
        r.names = {"l1", "l2", "t3", "t4"};
        return r;
    }
    if (method == "ranks") {
        ToolboxResult r;
        r.values = nd::ranks_in_place(x);
        return r;
    }
    if (method == "percentile") {
        ToolboxResult r;
        bool sorted = options.value_or("sorted", false);
        for (double p : data_at(data, 1, "statistics", method))
            r.values.push_back(nd::percentile(x, p, sorted));
        return r;
    }
    if (method == "running_covariance") {
        // data holds one vector per variable (column), all the same length (the number of new
        // observations to push); this is the transpose of RunningCovarianceMatrix::push()'s
        // one-row-at-a-time C# shape, chosen so a caller's existing "list of series" convention
        // (every other multi-series toolbox method) carries over unchanged.
        if (data.empty())
            throw std::runtime_error("toolbox method 'statistics.running_covariance' needs at "
                                     "least one data vector (one per variable)");
        int size = static_cast<int>(data.size());
        int num_rows = static_cast<int>(data[0].size());
        for (const auto& col : data)
            if (static_cast<int>(col.size()) != num_rows)
                throw std::runtime_error("toolbox method 'statistics.running_covariance' needs "
                                         "every data vector to have the same length");

        nd::RunningCovarianceMatrix rcm(size);
        if (options.contains("state")) {
            const JsonValue& s = options.at("state");
            math::linalg::Matrix mean(size, 1, s.at("mean").as_double_vector());
            math::linalg::Matrix cov(size, size, s.at("covariance").as_double_vector());
            rcm = nd::RunningCovarianceMatrix::from_state(s.at("n").as_int(), mean, cov);
        }
        for (int i = 0; i < num_rows; ++i) {
            std::vector<double> row(static_cast<std::size_t>(size));
            for (int j = 0; j < size; ++j)
                row[static_cast<std::size_t>(j)] = data[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
            rcm.push(row);
        }

        // Packed as [n, mean(size), covariance(size*size), sample_covariance(size*size),
        // sample_correlation(size*size), population_covariance(size*size),
        // population_correlation(size*size)], every matrix block flattened row-major (dims =
        // {size, size} describes each block's shape). Only n/mean/covariance are needed to
        // resume via "state"; the other four blocks are derived and returned for convenience.
        ToolboxResult r;
        r.values.push_back(static_cast<double>(rcm.n()));
        for (auto v : {rcm.mean(), rcm.covariance(), rcm.sample_covariance(), rcm.sample_correlation(),
                       rcm.population_covariance(), rcm.population_correlation()}) {
            std::vector<double> flat = flatten_matrix(v);
            r.values.insert(r.values.end(), flat.begin(), flat.end());
        }
        r.names = {"n"};
        r.dims = {size, size};
        return r;
    }
    throw std::runtime_error("unknown statistics method: " + method);
}

inline ToolboxResult run_spectra(const std::string& method,
                                 const std::vector<std::vector<double>>& data,
                                 const JsonValue& options) {
    namespace nd = numerics::data;
    namespace nf = numerics::math::fourier;

    if (method == "autocorrelation") {
        const std::vector<double>& x = data_at(data, 0, "spectra", method);
        int lag_max = options.value_or("lag_max", -1);
        std::string type_name = options.value_or("type", "correlation");
        nd::Autocorrelation::Type type;
        if (type_name == "correlation") type = nd::Autocorrelation::Type::Correlation;
        else if (type_name == "covariance") type = nd::Autocorrelation::Type::Covariance;
        else if (type_name == "partial") type = nd::Autocorrelation::Type::Partial;
        else throw std::runtime_error("unknown spectra.autocorrelation type: " + type_name);

        auto acf = nd::Autocorrelation::function(x, lag_max, type);
        if (!acf)
            throw std::runtime_error("spectra.autocorrelation: series too short for the requested lag");
        ToolboxResult r;
        r.dims = {static_cast<int>(acf->size()), 2};
        r.values.reserve(acf->size() * 2);
        for (const auto& pair : *acf) {
            r.values.push_back(pair[0]);
            r.values.push_back(pair[1]);
        }
        return r;
    }
    if (method == "autocorrelation_ci") {
        int sample_size = options.at("sample_size").as_int();
        double interval = options.value_or("confidence_level", 0.95);
        ToolboxResult r;
        r.values = nd::Autocorrelation::correlation_confidence_interval(sample_size, interval);
        r.names = {"lower", "upper"};
        return r;
    }
    if (method == "cross_correlation") {
        const std::vector<double>& x = data_at(data, 0, "spectra", method);
        const std::vector<double>& y = data_at(data, 1, "spectra", method);
        ToolboxResult r;
        r.values = nf::correlation(x, y);
        return r;
    }
    if (method == "dft" || method == "dft_real") {
        const std::vector<double>& x = data_at(data, 0, "spectra", method);
        bool inverse = options.value_or("inverse", false);
        std::vector<double> copy = x;  // fft()/real_fft() mutate their argument in place
        if (method == "dft")
            nf::fft(copy, inverse);
        else
            nf::real_fft(copy, inverse);
        ToolboxResult r;
        r.values = std::move(copy);
        return r;
    }
    throw std::runtime_error("unknown spectra method: " + method);
}

}  // namespace detail

inline ToolboxResult run_toolbox(const std::string& group, const std::string& method,
                                 const std::vector<std::vector<double>>& data,
                                 const std::string& options_json) {
    JsonValue options = models::spec::parse_json(options_json.empty() ? "{}" : options_json);
    if (group == "correlation") return detail::run_correlation(method, data, options);
    if (group == "gof") return detail::run_gof(method, data, options);
    if (group == "statistics") return detail::run_statistics(method, data, options);
    if (group == "spectra") return detail::run_spectra(method, data, options);
    throw std::runtime_error("unknown toolbox group: " + group);
}

}  // namespace corehydro::numerics::support
