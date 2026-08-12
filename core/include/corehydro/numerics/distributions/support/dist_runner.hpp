// corehydro ADDITION -- no upstream C# counterpart (sibling of estimation/support/fit_runner.hpp).
//
// The single place a distribution method is dispatched in this repo. Four callers drive it and
// none owns any evaluation logic: the cpp11 glue (corehydror/src/dist_spec.cpp), the pybind11
// glue (corehydropy/src/bindings/dist_spec.cpp), the C++ fixture runner (core/tests/
// test_fixtures.cpp) and the dotnet oracle emitter. Each serializes its native construct to the
// dist_spec.hpp grammar and calls run_dist/run_copula/run_mvdist, so a fixture case and a user's
// dist_pdf() call are the same code path.
//
// Stateless by construction: one call builds the object, evaluates once, and drops it. A seeded
// draw therefore returns the WHOLE vector in one call (method "random", args [n, seed]) so a
// rebuild can never split an RNG stream, which is exactly why the pre-phase-3 glue carried the
// bespoke *_seq entry points this replaces.
#pragma once

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/distributions/support/dist_spec.hpp"

namespace corehydro::numerics::distributions::support {

// Flat result surface every binding and every fixture assertion reads. `values` holds whatever
// the method returns, in method order; `names` labels them when the method returns a named set
// (moments, tail dependence); `spec` carries a child object back when the method returns a
// distribution (MultivariateNormal marginal/conditional) and is empty otherwise.
struct DistResult {
    std::vector<double> values;
    std::vector<std::string> names;
    std::string spec;
};

namespace detail {

inline std::vector<double> arg_numbers(const JsonValue& args) {
    return args.as_double_vector();
}

inline double arg_at(const JsonValue& args, std::size_t i, const char* method) {
    const auto& v = args.items();
    if (i >= v.size())
        throw std::runtime_error(std::string("distribution method '") + method + "' needs " +
                                 std::to_string(i + 1) + " argument(s)");
    return v[i].as_double();
}

}  // namespace detail

// Evaluates `method` against the univariate distribution described by `spec_json`.
//
// Methods, with their args and the length of `values`:
//   pdf / log_pdf / cdf / quantile   args = the evaluation vector, values = same length
//   moments                          args = [],        values = 8, names set
//   mean median mode sd skewness kurtosis minimum maximum
//                                    args = [],        values = 1
//   random                           args = [n, seed], values = n  (seed <= 0 means clock)
//   log_likelihood                   args = the sample, values = 1
//   parameters                       args = [],        values = the flat parameter vector
//   parameters_valid                 args = [],        values = 1 (1.0 or 0.0)
//   linear_moments                   always throws: no composite implements it upstream
inline DistResult run_dist(const std::string& spec_json, const std::string& method,
                           const std::string& args_json) {
    JsonValue spec = models::spec::parse_json(spec_json);
    JsonValue args = models::spec::parse_json(args_json);
    std::unique_ptr<UnivariateDistributionBase> d = build_univariate(spec);
    DistResult r;

    if (method == "pdf" || method == "log_pdf" || method == "cdf" || method == "quantile") {
        for (double x : detail::arg_numbers(args)) {
            if (method == "pdf") r.values.push_back(d->pdf(x));
            else if (method == "log_pdf") r.values.push_back(d->log_pdf(x));
            else if (method == "cdf") r.values.push_back(d->cdf(x));
            else r.values.push_back(d->inverse_cdf(x));
        }
        return r;
    }
    if (method == "moments") {
        r.values = {d->mean(),     d->median(),   d->mode(),    d->standard_deviation(),
                    d->skewness(), d->kurtosis(), d->minimum(), d->maximum()};
        r.names = {"mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"};
        return r;
    }
    if (method == "mean") { r.values = {d->mean()}; return r; }
    if (method == "median") { r.values = {d->median()}; return r; }
    if (method == "mode") { r.values = {d->mode()}; return r; }
    if (method == "sd") { r.values = {d->standard_deviation()}; return r; }
    if (method == "skewness") { r.values = {d->skewness()}; return r; }
    if (method == "kurtosis") { r.values = {d->kurtosis()}; return r; }
    if (method == "minimum") { r.values = {d->minimum()}; return r; }
    if (method == "maximum") { r.values = {d->maximum()}; return r; }
    if (method == "random") {
        int n = static_cast<int>(detail::arg_at(args, 0, "random"));
        int seed = static_cast<int>(detail::arg_at(args, 1, "random"));
        r.values = d->generate_random_values(n, seed);
        return r;
    }
    if (method == "log_likelihood") {
        r.values = {d->log_likelihood(detail::arg_numbers(args))};
        return r;
    }
    if (method == "parameters") {
        r.values = d->get_parameters();
        return r;
    }
    if (method == "parameters_valid") {
        r.values = {d->parameters_valid() ? 1.0 : 0.0};
        return r;
    }
    if (method == "linear_moments")
        throw std::runtime_error("linear moments are not available for '" + spec_family(spec) +
                                 "'; no composite distribution implements ILinearMomentEstimation "
                                 "upstream");
    throw std::runtime_error("unknown distribution method: " + method);
}

// Evaluates `method` against the copula described by `spec_json`.
//
//   pdf / log_pdf / cdf              args = all u then all v, values = n (one per pair)
//   inverse_cdf                      args = [u, v],       values = 2
//   tail_dependence                  args = [],           values = 2, names {lower, upper}
//   exceedance_or / exceedance_and   args = [u, v],       values = 1
//   theta / df                       args = [],           values = 1
//   bounds                           args = [],           values = 2, names {minimum, maximum}
//   parameters                       args = [],           values = the copula parameter vector
//   parameters_valid                 args = [],           values = 1
//   random                           args = [n, seed],    values = 2n (all x, then all y)
//   log_likelihood_pseudo / _ifm / _full
//                                    args = x then y,     values = 1
//   marginal_x_parameters / marginal_y_parameters
//                                    args = [],           values = that marginal's parameters
//
// pdf / log_pdf / cdf take the same split-at-the-halfway-point layout the three log-likelihood
// methods use, so one call evaluates a whole vector of pairs; args = [u, v] is that layout with
// n = 1 and is unchanged.
//
// `log_likelihood_pseudo` transforms x and y to their plotting positions before evaluating:
// BivariateCopula::PseudoLogLikelihood is defined on values already on (0, 1) and does not rank
// internally, so its caller passes the raw paired observations here exactly as it does for the
// IFM and full log-likelihoods (see support::plotting_positions).
inline DistResult run_copula(const std::string& spec_json, const std::string& method,
                             const std::string& args_json) {
    JsonValue spec = models::spec::parse_json(spec_json);
    JsonValue args = models::spec::parse_json(args_json);
    std::unique_ptr<copulas::BivariateCopula> c = build_copula(spec);
    DistResult r;

    auto split_xy = [&]() {
        std::vector<double> all = detail::arg_numbers(args);
        if (all.size() % 2 != 0)
            throw std::runtime_error("copula method '" + method +
                                     "' needs an even number of arguments (x then y)");
        std::size_t h = all.size() / 2;
        return std::make_pair(std::vector<double>(all.begin(), all.begin() + h),
                              std::vector<double>(all.begin() + h, all.end()));
    };

    if (method == "pdf" || method == "log_pdf" || method == "cdf") {
        auto uv = split_xy();
        if (uv.first.empty())
            throw std::runtime_error("copula method '" + method + "' needs at least one (u, v) pair");
        for (std::size_t i = 0; i < uv.first.size(); ++i) {
            if (method == "pdf") r.values.push_back(c->pdf(uv.first[i], uv.second[i]));
            else if (method == "log_pdf") r.values.push_back(c->log_pdf(uv.first[i], uv.second[i]));
            else r.values.push_back(c->cdf(uv.first[i], uv.second[i]));
        }
        return r;
    }
    if (method == "inverse_cdf") {
        std::array<double, 2> uv = c->inverse_cdf(detail::arg_at(args, 0, "inverse_cdf"),
                                                  detail::arg_at(args, 1, "inverse_cdf"));
        r.values = {uv[0], uv[1]};
        return r;
    }
    if (method == "tail_dependence") {
        r.values = {c->lower_tail_dependence(), c->upper_tail_dependence()};
        r.names = {"lower", "upper"};
        return r;
    }
    if (method == "exceedance_or") {
        r.values = {c->or_joint_exceedance_probability(detail::arg_at(args, 0, "exceedance_or"),
                                                       detail::arg_at(args, 1, "exceedance_or"))};
        return r;
    }
    if (method == "exceedance_and") {
        r.values = {c->and_joint_exceedance_probability(detail::arg_at(args, 0, "exceedance_and"),
                                                        detail::arg_at(args, 1, "exceedance_and"))};
        return r;
    }
    if (method == "theta") { r.values = {c->theta()}; return r; }
    if (method == "df") {
        std::vector<double> p = c->get_copula_parameters();
        if (p.size() < 2)
            throw std::runtime_error("copula '" + spec_family(spec) +
                                     "' has no degrees-of-freedom parameter");
        r.values = {p[1]};
        return r;
    }
    if (method == "bounds") {
        r.values = {c->theta_minimum(), c->theta_maximum()};
        r.names = {"minimum", "maximum"};
        return r;
    }
    if (method == "parameters") { r.values = c->get_copula_parameters(); return r; }
    if (method == "parameters_valid") { r.values = {c->parameters_valid() ? 1.0 : 0.0}; return r; }
    if (method == "random") {
        int n = static_cast<int>(detail::arg_at(args, 0, "random"));
        int seed = static_cast<int>(detail::arg_at(args, 1, "random"));
        auto m = c->generate_random_values(n, seed);
        for (const auto& row : m) r.values.push_back(row[0]);
        for (const auto& row : m) r.values.push_back(row[1]);
        return r;
    }
    if (method == "log_likelihood_pseudo" || method == "log_likelihood_ifm" ||
        method == "log_likelihood_full") {
        auto xy = split_xy();
        if (method == "log_likelihood_pseudo")
            r.values = {c->pseudo_log_likelihood(plotting_positions(xy.first),
                                                 plotting_positions(xy.second))};
        else if (method == "log_likelihood_ifm")
            r.values = {c->ifm_log_likelihood(xy.first, xy.second)};
        else
            r.values = {c->log_likelihood(xy.first, xy.second)};
        return r;
    }
    if (method == "marginal_x_parameters" || method == "marginal_y_parameters") {
        const auto& m = method == "marginal_x_parameters" ? c->marginal_distribution_x
                                                          : c->marginal_distribution_y;
        if (!m) throw std::runtime_error("copula method '" + method + "': no marginal is attached");
        r.values = m->get_parameters();
        return r;
    }
    throw std::runtime_error("unknown copula method: " + method);
}

// Serializes a MultivariateNormal back into the grammar so `DistResult::spec` round-trips
// (the marginal/conditional methods below hand a child spec back rather than a C++ object, so
// they compose without anything crossing the language boundary).
inline std::string mvn_spec_string(const MultivariateNormal& m) {
    std::string out = R"({"family":"MultivariateNormal","mean":[)";
    const std::vector<double>& mu = m.mean();
    for (std::size_t i = 0; i < mu.size(); ++i) {
        if (i) out += ",";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", mu[i]);
        out += buf;
    }
    out += R"(],"covariance":[)";
    for (int i = 0; i < m.dimension(); ++i) {
        if (i) out += ",";
        out += "[";
        for (int j = 0; j < m.dimension(); ++j) {
            if (j) out += ",";
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", m.covariance(i, j));
            out += buf;
        }
        out += "]";
    }
    out += "]}";
    return out;
}

// Evaluates `method` against the multivariate distribution described by `spec_json`.
//
//   pdf / log_pdf / cdf     args = the point,            values = 1
//   mahalanobis             args = the point,            values = 1
//   dimension               args = [],                   values = 1
//   mean / variance / sd    args = [],                   values = dimension
//   median / mode           args = [],                   values = dimension
//   covariance              args = [],                   values = dimension^2, row-major
//   inverse_cdf             args = probabilities,        values = dimension
//   random                  args = [n, seed],            values = n * dimension, row-major
//   random_lhs              args = [n, seed],            values = n * dimension, row-major
//   interval                args = lower then upper,     values = 1        (MultivariateNormal)
//   marginal                args = 0-based indices,      spec = the child (MultivariateNormal)
//   conditional             args = indices then values,  spec = the child (MultivariateNormal)
//   degrees_of_freedom      args = [],                   values = 1        (MultivariateStudentT)
//   alpha                   args = [],                   values = dimension        (Dirichlet)
//   alpha_sum               args = [],                   values = 1                (Dirichlet)
//   number_of_trials        args = [],                   values = 1               (Multinomial)
//   parameters_valid        args = [],                   values = 1
//
// `median` and `mode` are MultivariateNormal and MultivariateStudentT (both return the centre)
// plus Dirichlet's `mode`; no other family defines them upstream. `inverse_cdf` is the
// Cholesky map upstream implements, NOT a true multivariate quantile: it turns a vector of
// independent uniforms into one point. MultivariateNormal takes `dimension` probabilities;
// MultivariateStudentT takes `dimension + 1` (the last drives the chi-squared mixing variable)
// and still returns `dimension` values.
//
// `cdf` on Dirichlet or Multinomial and `pdf` on BivariateEmpirical are upstream stubs; both
// surface here as a throw naming the family, not a NaN.
inline DistResult run_mvdist(const std::string& spec_json, const std::string& method,
                             const std::string& args_json) {
    JsonValue spec = models::spec::parse_json(spec_json);
    JsonValue args = models::spec::parse_json(args_json);
    std::string family = spec_family(spec);
    std::unique_ptr<MultivariateDistribution> d = build_multivariate(spec);
    DistResult r;

    auto* mvn = dynamic_cast<MultivariateNormal*>(d.get());

    if (method == "cdf" && (family == "Dirichlet" || family == "Multinomial"))
        throw std::runtime_error("cdf is not implemented for '" + family + "' upstream");
    if (method == "pdf" && family == "BivariateEmpirical")
        throw std::runtime_error("pdf is not implemented for 'BivariateEmpirical' upstream "
                                 "(it returns NaN)");
    if ((method == "marginal" || method == "conditional" || method == "interval") && !mvn)
        throw std::runtime_error("'" + method + "' is available for MultivariateNormal only; '" +
                                 family + "' has no such member upstream");

    if (method == "pdf") { r.values = {d->pdf(detail::arg_numbers(args))}; return r; }
    if (method == "log_pdf") { r.values = {d->log_pdf(detail::arg_numbers(args))}; return r; }
    if (method == "cdf") { r.values = {d->cdf(detail::arg_numbers(args))}; return r; }
    if (method == "dimension") { r.values = {static_cast<double>(d->dimension())}; return r; }
    if (method == "parameters_valid") { r.values = {d->parameters_valid() ? 1.0 : 0.0}; return r; }
    if (method == "marginal") {
        std::vector<int> idx;
        for (double v : detail::arg_numbers(args)) idx.push_back(static_cast<int>(v));
        MultivariateNormal child = mvn->marginal(idx);
        r.spec = mvn_spec_string(child);
        return r;
    }
    if (method == "conditional") {
        std::vector<double> all = detail::arg_numbers(args);
        if (all.size() % 2 != 0)
            throw std::runtime_error("'conditional' needs indices then the same number of values");
        std::size_t h = all.size() / 2;
        std::vector<int> idx;
        for (std::size_t i = 0; i < h; ++i) idx.push_back(static_cast<int>(all[i]));
        std::vector<double> vals(all.begin() + h, all.end());
        MultivariateNormal child = mvn->conditional(idx, vals);
        r.spec = mvn_spec_string(child);
        return r;
    }
    if (method == "interval") {
        std::vector<double> all = detail::arg_numbers(args);
        if (all.size() % 2 != 0)
            throw std::runtime_error("'interval' needs a lower vector then an upper vector");
        std::size_t h = all.size() / 2;
        r.values = {mvn->interval(std::vector<double>(all.begin(), all.begin() + h),
                                  std::vector<double>(all.begin() + h, all.end()))};
        return r;
    }
    // The remaining methods have no shared base signature: MultivariateNormal and
    // MultivariateStudentT carry the accessors, Dirichlet and Multinomial carry their own, and
    // BivariateEmpirical carries none. One branch per concrete type, dynamic_cast up front.
    auto* mvt = dynamic_cast<MultivariateStudentT*>(d.get());
    auto* dir = dynamic_cast<Dirichlet*>(d.get());
    auto* mn = dynamic_cast<Multinomial*>(d.get());

    auto flatten = [&r](const std::vector<std::vector<double>>& m) {
        for (const auto& row : m)
            for (double v : row) r.values.push_back(v);
    };
    auto no_such = [&](const char* what) -> DistResult {
        throw std::runtime_error(std::string("'") + what + "' is not available for '" + family +
                                 "' upstream");
    };

    if (method == "mahalanobis") {
        if (mvn) r.values = {mvn->mahalanobis(detail::arg_numbers(args))};
        else if (mvt) r.values = {mvt->mahalanobis(detail::arg_numbers(args))};
        else return no_such("mahalanobis");
        return r;
    }
    if (method == "mean") {
        if (mvn) r.values = mvn->mean();
        else if (mvt) r.values = mvt->mean();
        else if (dir) r.values = dir->mean();
        else if (mn) r.values = mn->mean();
        else return no_such("mean");
        return r;
    }
    if (method == "variance") {
        if (mvn) r.values = mvn->variance();
        else if (mvt) r.values = mvt->variance();
        else if (dir) r.values = dir->variance();
        else if (mn) r.values = mn->variance();
        else return no_such("variance");
        return r;
    }
    if (method == "sd") {
        if (mvn) r.values = mvn->standard_deviation();
        else if (mvt) r.values = mvt->standard_deviation();
        else return no_such("sd");
        return r;
    }
    if (method == "median") {
        if (mvn) r.values = mvn->median();
        else if (mvt) r.values = mvt->median();
        else return no_such("median");
        return r;
    }
    if (method == "mode") {
        if (mvn) r.values = mvn->mode();
        else if (mvt) r.values = mvt->mode();
        else if (dir) r.values = dir->mode();
        else return no_such("mode");
        return r;
    }
    if (method == "inverse_cdf") {
        if (mvn) r.values = mvn->inverse_cdf(detail::arg_numbers(args));
        else if (mvt) r.values = mvt->inverse_cdf(detail::arg_numbers(args));
        else return no_such("inverse_cdf");
        return r;
    }
    if (method == "degrees_of_freedom") {
        if (!mvt) return no_such("degrees_of_freedom");
        r.values = {mvt->degrees_of_freedom()};
        return r;
    }
    if (method == "alpha") {
        if (!dir) return no_such("alpha");
        r.values = dir->alpha();
        return r;
    }
    if (method == "alpha_sum") {
        if (!dir) return no_such("alpha_sum");
        r.values = {dir->alpha_sum()};
        return r;
    }
    if (method == "number_of_trials") {
        if (!mn) return no_such("number_of_trials");
        r.values = {static_cast<double>(mn->number_of_trials())};
        return r;
    }
    if (method == "covariance") {
        int n = d->dimension();
        if (mvn || mvt || dir || mn) {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    r.values.push_back(mvn   ? mvn->covariance(i, j)
                                       : mvt ? mvt->covariance(i, j)
                                       : dir ? dir->covariance(i, j)
                                             : mn->covariance(i, j));
            return r;
        }
        return no_such("covariance");
    }
    if (method == "random" || method == "random_lhs") {
        int n = static_cast<int>(detail::arg_at(args, 0, method.c_str()));
        int seed = static_cast<int>(detail::arg_at(args, 1, method.c_str()));
        if (method == "random_lhs") {
            if (mvn) flatten(mvn->latin_hypercube_random_values(n, seed));
            else if (mvt) flatten(mvt->latin_hypercube_random_values(n, seed));
            else return no_such("random_lhs");
        } else {
            if (mvn) flatten(mvn->generate_random_values(n, seed));
            else if (mvt) flatten(mvt->generate_random_values(n, seed));
            else if (dir) flatten(dir->generate_random_values(n, seed));
            else if (mn) flatten(mn->generate_random_values(n, seed));
            else return no_such("random");
        }
        return r;
    }
    throw std::runtime_error("unknown multivariate method: " + method);
}

}  // namespace corehydro::numerics::distributions::support
