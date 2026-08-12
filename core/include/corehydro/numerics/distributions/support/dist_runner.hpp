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

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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

}  // namespace corehydro::numerics::distributions::support
