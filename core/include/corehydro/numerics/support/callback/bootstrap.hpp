// corehydro ADDITION -- no upstream C# counterpart. The bootstrap group of callback_runner.hpp:
// running the ported Bootstrap workflow against the user's own delegates.
//
// This is upstream's OWN API, not an addition: `Bootstrap<TData>` (Sampling/Bootstrap/Bootstrap.cs)
// exposes `ResampleFunction` (`Func<TData, ParameterSet, Random, TData>`), `FitFunction`
// (`Func<TData, ParameterSet>`), `StatisticFunction` (`Func<ParameterSet, double[]>`) and
// `JackknifeFunction` (`Func<TData, int, TData>`) as public properties, and a C# caller supplies
// all of them. The model registry under sampling/bootstrap/ is corehydro fixture scaffolding; until
// this group existed the packages could reach the bootstrap only through it, so a user could not
// bootstrap their own statistic at all.
//
// One method, `run`. `TData` is instantiated as `std::vector<double>`. Options grammar (JSON
// object):
//
//   {"data": [...], "replicates": 1000, "seed": 12345, "alpha": 0.1,
//    "ci_method": "Percentile|BiasCorrected|Normal|BootstrapT|BCa",
//    "parameters": [...], "inner_replicates": 300, "max_retries": 20}
//
// `data` is REQUIRED and is the original sample: what the resample delegate is handed and what the
// jackknife delegate leaves one observation out of. It travels in the options JSON rather than as a
// native vector because `run_callback`'s signature carries no data channel -- unlike
// toolbox_runner.hpp, whose goodness-of-fit verbs would pay a JSON parse for two arbitrary-length
// series, a bootstrap parses its data ONCE for a run that then crosses into the host thousands of
// times, so the parse is not measurable.
//
// Every other key is optional and an absent one leaves the ported class's own default in force
// (10,000 replicates, seed 12345, 20 retries, 300 inner replicates, alpha 0.1).
//
// `parameters` is the original parameter set -- C#'s `new Bootstrap<TData>(originalData,
// originalParameters)` second argument, the theta-hat every replicate is compared against. Left
// out, it is the fit of the original data: `fit(data)`, which is what the non-parametric bootstrap
// means by theta-hat and what keeps its length in step with what `fit` returns.
//
// THE CI METHOD PICKS THE RUN. `GetConfidenceIntervals(BootstrapT, ...)` throws unless
// `RunWithStudentizedBootstrap()` was the run that produced the results, and that workflow nests
// `inner_replicates` more resample+fit pairs inside every replicate. So `ci_method` selects the
// workflow here: BootstrapT runs the studentized one, everything else the regular `Run()`. A user
// who asks for BootstrapT with the ported 300 inner replicates and 10,000 outer ones is asking for
// three million crossings back into R or Python, which is why both wrappers document the cost and
// expose `inner_replicates`. `max_retries` (upstream's `MaxRetries`, also a public property, 20 by
// default) multiplies host callback crossings the same way a failed replicate does, so both
// wrappers expose it too, named identically to this key.
//
// BCa IS CHECKED UP FRONT, and that is the one place this group deliberately runs a check EARLIER
// than the ported class does. C#'s `ValidateConfidenceIntervalRequest` refuses BCa without a
// jackknife delegate inside `GetConfidenceIntervals` -- i.e. after the entire run has finished. In
// C# that is a wasted second; here every one of those replicates is a crossing into R or Python, so
// the same refusal after ten minutes of a user's own code is a different thing entirely. The check
// is repeated here before the first replicate; the ported one stays exactly where it is.
//
// Result layout, all of it named (`dims` is `{n_statistics, n_parameters}`):
//
//   replicates, failed_replicates, alpha
//   statistic[i], statistic_lower[i], statistic_upper[i], statistic_se[i], statistic_mean[i],
//     statistic_valid[i]        -- BootstrapResults.StatisticResults[i]
//   parameter[j], parameter_lower[j], ... parameter_valid[j]
//                               -- BootstrapResults.ParameterResults[j]
//
// `statistic[i]` is the PopulationEstimate: the statistic of the original parameter set, not a
// bootstrap average. `statistic_mean[i]` is the mean of the valid replicates, so the two together
// are the bias estimate.
//
// GUARD DISCIPLINE. This group has FOUR live callbacks -- more than any other -- and all four
// guards share ONE abort state. That is load-bearing rather than tidy: with private states, an R
// error raised inside `fit` latches only the fit guard, and the ported loop -- which does not know
// it is unwinding -- calls `resample` on the next replicate, re-entering R with an unwind already
// pending. Sharing makes the first throw short-circuit every callback in the group (see
// callback_guard.hpp's "USE A SHARED STATE" note); core/tests/test_callback_runner.cpp asserts it
// per delegate, by making one throw and giving another a body that reports being entered
// afterwards.
//
// EACH GUARD'S SENTINEL IS THE VALUE ITS PORTED CONSUMER TREATS AS A FAILED REPLICATE, and one of
// them has to be the right LENGTH rather than merely a legal value:
//
//   - resample: an empty sample. Only ever produced after a latch, at which point `fit` -- sharing
//     the state -- short-circuits too, so no ported code ever does arithmetic on it.
//   - fit: NaN parameters. `HasExpectedFiniteParameterValues` reads that as a failed replicate and
//     `continue`s, which is the ported class's own vocabulary for "this draw did not work".
//   - statistic: NaN, ONE PER STATISTIC. The length is the load-bearing part:
//     `ComputeAccelerationConstants` (the BCa path) indexes `jackStats[i]` for every statistic
//     WITHOUT checking its length, so an empty sentinel there is an out-of-bounds read on a
//     `std::vector` -- undefined behaviour, not an exception. That is why the statistic is probed
//     once up front (see below) rather than guessed at.
//   - jackknife: an empty sample, for the same reason as resample.
//
// THE UP-FRONT STATISTIC PROBE. `statistic(parameters)` is evaluated once, directly and unguarded, before
// the ported class is built, purely to learn how many statistics it returns. It is unguarded on
// purpose: nothing ported is running yet, so an exception from it is already the user's own and
// travels straight out. It costs one extra evaluation of a pure function; it buys a sentinel that
// cannot be indexed out of bounds.
//
// WHAT IS VALIDATED, AND WHAT IS DELIBERATELY NOT. Return LENGTHS are checked inside each guarded
// function, so a wrong one aborts the run exactly as a host-language error does -- latched on the
// first occurrence, every later callback short-circuited, and the message rethrown from the
// protected frame. This is stricter than the ported class on purpose: C# marks a wrong-length fit
// as a failed replicate and retries it, so 10,000 replicates fail silently and every interval comes
// back NaN, which tells a user nothing about the mistake they made. Non-finite VALUES are the
// opposite case and are passed straight through: `HasExpectedFiniteParameterValues` and
// `ValidateStatistics` exist precisely to treat those as a failed replicate, and stealing that
// would break a documented upstream behaviour. The bindings' converters follow the same split (see
// their own notes): NA/NaN is refused in the two delegates that return DATA and allowed in the two
// that return numbers the ported class tests for finiteness itself.
//
// THE JACKKNIFE INDEX IS 0-BASED, because upstream's delegate is (`Func<TData, int, TData>` called
// as `JackknifeFunction(originalData, idx)` for `idx` in `[0, SampleSizeFunction(data))`), and both
// packages document it by name. The R spelling of "leave observation `index` out" is therefore
// `data[-(index + 1)]`; `data[-index]`, the mistake that shape invites, is `data[0]` -- the EMPTY
// vector -- at index 0 and silently drops the wrong observation at every later one. So the guarded
// jackknife requires a non-empty sample SHORTER than the one it was given: that catches the empty
// half of the trap and the drops-nothing case, which the ported BCa would otherwise answer with
// `0 / 0` and a NaN interval. The off-by-one half cannot be caught by any check -- the sample is
// the right length and simply names a different observation -- which is why the base is documented
// in both packages rather than merely guarded here.
//
// SampleSizeFunction is not a user callback here. C# makes it a delegate because `TData` is
// generic; `TData` is a `std::vector<double>` on this surface, so its size IS the sample size and
// asking the user for it again would only be a chance to disagree with the data.
//
// THE PIVOTAL RUN TYPE (`"run_type": "pivotal"`) is upstream's OTHER bootstrap mode and this file's
// second workflow. It fits through a FIFTH delegate the regular one does not have -- upstream's
// `FitWithCovarianceFunction` (`Func<TData, BootstrapFit>`), a fit returning the parameters AND
// their covariance -- standardizes every raw fit against the parent through its own Cholesky
// factor, and reinflates it through the parent's. Its options are the C# properties, spelled here
// as the run's own keys:
//
//   {"run_type": "pivotal", "data": [...], "replicates": 200, "seed": 12345, "alpha": 0.1,
//    "parameters": [...], "original_covariance": [[...], [...]],
//    "pivotal_links": ["Log", null], "pivotal_invalid_draw_policy": "drop|use_raw|use_parent",
//    "regularize_pivotal_covariances": true, "pivotal_z_limit": 3.0,
//    "add_pivotal_jitter": false, "pivotal_jitter_scale": 0.01, "max_retries": 20}
//
// THE PARENT FIT IS THE C# SECOND CONSTRUCTOR, not a required matrix. `Bootstrap(TData,
// BootstrapFit)` takes a covariance-aware fit and sets `OriginalCovariance` from it, so when
// `original_covariance` is absent the covariance-aware fit of the ORIGINAL data supplies it --
// evaluated once, up front, directly and unguarded, exactly as the regular run type's theta-hat
// fit is and for the same reason (nothing ported is running yet, so an exception from it is
// already the user's own). `parameters` overrides that fit's parameter vector without touching its
// covariance, which is what C# separating the two properties means.
//
// LINKS TRAVEL AS NAMES, NOT AS A HOST CALLBACK. C#'s `PivotalLinkFactory` is
// `Func<PivotalBootstrapContext, ILinkFunction?[]>`, but every link it can return is a ported class
// this library already owns, so marshalling a host-language factory would buy nothing but a way to
// return something that is not a link at all. `pivotal_links` is therefore one entry per parameter,
// each either a link TYPE NAME or null (null == identity, C#'s own null-element convention), and
// the array is resolved here into the factory the ported class calls. The seven names are the
// members of the Numerics `LinkFunctionType` enum -- Identity, Log, Logit, Probit,
// ComplementaryLogLog, YeoJohnson, FisherZ -- and deliberately NOT the twelve
// numerics/support/toolbox/link.hpp accepts: the five BestFit-specific ones there are
// parameterized (a spec object, not a name), and this surface takes names. The length is checked
// HERE, before the first replicate, rather than inside the ported factory call after every one.
//
// TWO DELEGATES OF THE FIVE ARE DELIBERATELY OFF THIS SURFACE: `PivotalReplicateFilter`
// (`Func<BootstrapFit, bool>`) and `PivotalParameterValidator` (`Func<double[], bool>`). Both are
// live host callbacks the ported class supports and this group could marshal; no upstream example
// drives either from user code, and a filter that silently drops replicates is the last thing a
// user should reach for before they have looked at the diagnostics. The C++ surface has them and
// core/tests/test_pivotal_bootstrap.cpp proves them; a package user does not, and that is a
// documented omission rather than an oversight.
//
// ONLY PERCENTILE INTERVALS EXIST AFTER A PIVOTAL RUN. `ValidateConfidenceIntervalRequest` throws
// for any other method, and that refusal is repeated here BEFORE the first replicate for the same
// reason the BCa one is: in C# the wasted work is a second, and here it is a user's own R or Python
// code running once per replicate first.
//
// The result carries the regular layout PLUS the raw block and the six diagnostic counts:
//
//   raw_statistic[i], raw_statistic_lower[i], ... raw_statistic_valid[i]
//   raw_parameter[j], ... raw_parameter_valid[j]   -- GetRawPivotalConfidenceIntervals(alpha)
//   requested_replicates, rejected_raw_replicates, failed_raw_replicates,
//     accepted_raw_replicates, invalid_pivotal_replicates, retained_pivotal_replicates
//                                                  -- PivotalBootstrapDiagnostics, named verbatim
//
// `dims` is unchanged ({n_statistics, n_parameters}): the raw block has the same shape as the
// pivotal one by construction, so nothing downstream needs a third number to read it.
//
// The drive site is wrapped AND the trailing rethrow is kept, and both halves earn their place: the
// sentinels are all values the ported class treats as legal, so an aborted run does NOT throw of its
// own accord -- it walks every replicate marking it failed and returns a finished-looking result,
// which only the TRAILING rethrow catches. The wrap catches the other half: `ValidateStatistics`
// rejects the NaN statistic sentinel with an INTERNAL exception of its own, which would otherwise
// replace the user's message.
#pragma once
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/functions/i_link_function.hpp"
#include "corehydro/numerics/functions/link_function_factory.hpp"
#include "corehydro/numerics/functions/link_function_type.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/bootstrap/bootstrap.hpp"
#include "corehydro/numerics/sampling/bootstrap/bootstrap_results.hpp"
#include "corehydro/numerics/sampling/bootstrap/ci_method_names.hpp"
#include "corehydro/numerics/sampling/bootstrap/support/bootstrap_fit.hpp"
#include "corehydro/numerics/sampling/bootstrap/support/pivotal_bootstrap_context.hpp"
#include "corehydro/numerics/sampling/bootstrap/support/pivotal_bootstrap_invalid_draw_policy.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/support/callback/common.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"

namespace corehydro::numerics::support::detail {

namespace chboot = corehydro::numerics::sampling;
namespace chopt = corehydro::numerics::math::optimization;
namespace chlin = corehydro::numerics::math::linalg;
namespace chfun = corehydro::numerics::functions;

// One labelled value into the flat result. Free functions rather than lambdas local to
// bootstrap_flatten because the pivotal path below appends its own blocks and counts to the same
// result and must label them identically.
inline void bootstrap_push(CallbackResult& r, const std::string& name, double value) {
    r.names.push_back(name);
    r.values.push_back(value);
}

inline void bootstrap_push_block(CallbackResult& r, const char* label,
                                 const std::vector<chboot::BootstrapStatisticResult>& block) {
    for (std::size_t i = 0; i < block.size(); ++i) {
        const std::string ix = "[" + std::to_string(i) + "]";
        bootstrap_push(r, std::string(label) + ix, block[i].population_estimate);
        bootstrap_push(r, std::string(label) + "_lower" + ix, block[i].lower_ci);
        bootstrap_push(r, std::string(label) + "_upper" + ix, block[i].upper_ci);
        bootstrap_push(r, std::string(label) + "_se" + ix, block[i].standard_error);
        bootstrap_push(r, std::string(label) + "_mean" + ix, block[i].mean);
        bootstrap_push(r, std::string(label) + "_valid" + ix,
                       static_cast<double>(block[i].valid_count));
    }
}

// Flattens one finished BootstrapResults into the layout this file's header documents.
inline CallbackResult bootstrap_flatten(const chboot::BootstrapResults& results, int replicates) {
    CallbackResult r;
    r.status = "Success";

    bootstrap_push(r, "replicates", static_cast<double>(replicates));
    bootstrap_push(r, "failed_replicates", static_cast<double>(results.failed_replicates));
    bootstrap_push(r, "alpha", results.alpha);
    bootstrap_push_block(r, "statistic", results.statistic_results);
    bootstrap_push_block(r, "parameter", results.parameter_results);

    r.dims = {static_cast<int>(results.statistic_results.size()),
              static_cast<int>(results.parameter_results.size())};
    return r;
}

// --- the pivotal run type ---------------------------------------------------------------------

// A JSON array of rows -> Matrix, the spelling every matrix on this project's spec surfaces takes
// (dist_spec.hpp's MultivariateNormal covariance is the same shape). Rectangularity is checked
// here so the message names the row that disagrees rather than leaving Matrix to be built ragged.
inline chlin::Matrix bootstrap_matrix_from_rows(const JsonValue& v, const std::string& what) {
    std::vector<std::vector<double>> rows;
    for (const JsonValue& row : v.items()) rows.push_back(row.as_double_vector());
    if (rows.empty()) throw std::invalid_argument(what + " must have at least one row; it was empty");
    for (const std::vector<double>& row : rows)
        if (row.size() != rows[0].size())
            throw std::invalid_argument(what + " must be rectangular; its first row holds " +
                                        std::to_string(rows[0].size()) + " values and another holds " +
                                        std::to_string(row.size()));
    return chlin::Matrix(std::move(rows));
}

// One link per parameter from `pivotal_links`: a type NAME or null (null == identity, C#'s own
// null-element convention). The seven names are the members of the Numerics LinkFunctionType enum
// -- see the file header on why these seven and not toolbox/link.hpp's twelve.
inline std::shared_ptr<chfun::ILinkFunction> bootstrap_pivotal_link(const JsonValue& v) {
    if (v.is_null()) return nullptr;
    using LT = chfun::LinkFunctionType;
    static const std::vector<std::pair<std::string, LT>> table = {
        {"Identity", LT::Identity},         {"Log", LT::Log},
        {"Logit", LT::Logit},               {"Probit", LT::Probit},
        {"ComplementaryLogLog", LT::ComplementaryLogLog},
        {"YeoJohnson", LT::YeoJohnson},     {"FisherZ", LT::FisherZ}};
    const std::string& name = v.as_string();
    for (const auto& entry : table)
        if (entry.first == name)
            return std::shared_ptr<chfun::ILinkFunction>(chfun::LinkFunctionFactory::create(entry.second));
    throw std::invalid_argument(
        "unknown link type '" + name +
        "'; the pivotal links are named after the Numerics link functions: Identity, Log, Logit, "
        "Probit, ComplementaryLogLog, YeoJohnson, FisherZ -- or null for the identity");
}

// Reads the covariance-aware fit callback's return into a BootstrapFit, checking the two shapes
// the ported class cannot check for itself in a way a user could act on. The length checks live
// INSIDE the guarded call (as every other delegate's do) so a wrong shape aborts the run instead of
// failing all `replicates` silently -- see the file header's "WHAT IS VALIDATED" note. Non-finite
// VALUES pass straight through: `IsValidFit` reads those as a failed replicate, which is upstream's
// own vocabulary and not this group's to steal.
inline chboot::BootstrapFit bootstrap_fit_from_return(const FitWithCovarianceReturn& out,
                                                      std::size_t n_parameters) {
    if (out.parameters.size() != n_parameters)
        throw std::invalid_argument(
            "the fit_with_covariance function must return one value per parameter; the model has " +
            std::to_string(n_parameters) + " and it returned " +
            std::to_string(out.parameters.size()));
    if (out.rows != static_cast<int>(n_parameters) || out.cols != static_cast<int>(n_parameters))
        throw std::invalid_argument(
            "the fit_with_covariance function must return a covariance matrix with one row and one "
            "column per parameter; the model has " +
            std::to_string(n_parameters) + " and it returned " + std::to_string(out.rows) + " x " +
            std::to_string(out.cols));
    return chboot::BootstrapFit(out.parameters,
                                chlin::Matrix(out.rows, out.cols, out.covariance));
}

inline CallbackResult run_bootstrap_pivotal(const JsonValue& o, const CallbackSet& cbs) {
    if (!cbs.data_rng)
        throw std::invalid_argument(
            "bootstrap/run requires a resample function, called with (data, parameters, rng)");
    if (!cbs.data_covariance)
        throw std::invalid_argument(
            "the pivotal bootstrap requires a fit_with_covariance function, called with (data) and "
            "returning the fitted parameters together with their covariance matrix");
    if (!cbs.vector_vector)
        throw std::invalid_argument(
            "bootstrap/run requires a statistic function, called with (parameters)");

    std::vector<double> data = require_vector(o, "data", "bootstrap/run");
    if (data.empty())
        throw std::invalid_argument("bootstrap/run requires 'data' to hold at least one value");

    // Repeated ahead of ValidateConfidenceIntervalRequest for the reason the BCa check is -- see
    // the file header. The wording is the ported class's own.
    const std::string ci_name = o.value_or("ci_method", "Percentile");
    if (chboot::parse_bootstrap_ci_method(ci_name) != chboot::BootstrapCIMethod::Percentile)
        throw std::invalid_argument(
            "Only percentile confidence intervals are supported after a pivotal bootstrap run.");
    const double alpha = o.value_or("alpha", 0.1);

    // The parent fit: C#'s `Bootstrap(TData, BootstrapFit)` second constructor. The covariance-aware
    // fit of the original data supplies whichever half the options do not, and is called directly
    // rather than through a guard because no ported code is running yet, so an exception from it is
    // already the user's own and needs no protection to reach them.
    std::optional<chlin::Matrix> parent_covariance;
    if (o.contains("original_covariance"))
        parent_covariance = bootstrap_matrix_from_rows(o.at("original_covariance"),
                                                       "'original_covariance'");
    std::vector<double> parent_parameters;
    if (o.contains("parameters")) parent_parameters = o.at("parameters").as_double_vector();
    if (!parent_covariance.has_value() || parent_parameters.empty()) {
        FitWithCovarianceReturn probe = cbs.data_covariance(data);
        if (probe.parameters.empty())
            throw std::invalid_argument(
                "the fit_with_covariance function must return at least one parameter; it returned "
                "nothing");
        if (probe.rows != static_cast<int>(probe.parameters.size()) || probe.rows != probe.cols)
            throw std::invalid_argument(
                "the fit_with_covariance function must return a square covariance matrix with one "
                "row per parameter; it returned " +
                std::to_string(probe.parameters.size()) + " parameters and a " +
                std::to_string(probe.rows) + " x " + std::to_string(probe.cols) + " covariance");
        if (parent_parameters.empty()) parent_parameters = probe.parameters;
        if (!parent_covariance.has_value())
            parent_covariance = chlin::Matrix(probe.rows, probe.cols, probe.covariance);
    }
    const std::size_t n_parameters = parent_parameters.size();
    if (parent_covariance->number_of_rows() != static_cast<int>(n_parameters) ||
        parent_covariance->number_of_columns() != static_cast<int>(n_parameters))
        throw std::invalid_argument(
            "'original_covariance' must have one row and one column per parameter; the fit has " +
            std::to_string(n_parameters) + " and it is " +
            std::to_string(parent_covariance->number_of_rows()) + " x " +
            std::to_string(parent_covariance->number_of_columns()));

    // The up-front probe, purely to size the statistic guard's sentinel -- see the file header.
    const std::size_t n_statistics = cbs.vector_vector(parent_parameters).size();
    if (n_statistics == 0)
        throw std::invalid_argument(
            "the statistic function must return at least one value; it returned nothing");

    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    CallbackAbortStatePtr abort = make_abort_state();

    GuardedCall<std::vector<double>, const std::vector<double>&, const std::vector<double>&,
                chboot::MersenneTwister&>
        resample(
            [fn = cbs.data_rng](const std::vector<double>& d, const std::vector<double>& p,
                                chboot::MersenneTwister& prng) {
                std::vector<double> sample = fn(d, p, prng);
                if (sample.empty())
                    throw std::invalid_argument(
                        "the resample function must return at least one value; it returned an "
                        "empty sample");
                return sample;
            },
            std::vector<double>{}, abort);

    // The sentinel is a fit of NaN parameters with a zero covariance of the right size: `IsValidFit`
    // reads the non-finite parameters as a failed replicate and the ported loop retries it, which is
    // the same contract the regular run type's NaN-parameter fit sentinel has. BootstrapFit's own
    // constructor requires the covariance to be square and to match the parameter count, so the
    // sentinel cannot be built any smaller.
    GuardedCall<chboot::BootstrapFit, const std::vector<double>&> fit_with_covariance(
        [fn = cbs.data_covariance, n_parameters](const std::vector<double>& d) {
            return bootstrap_fit_from_return(fn(d), n_parameters);
        },
        chboot::BootstrapFit(std::vector<double>(n_parameters, kNaN),
                             chlin::Matrix(static_cast<int>(n_parameters))),
        abort);

    GuardedCall<std::vector<double>, const std::vector<double>&> statistic(
        [fn = cbs.vector_vector, n_statistics](const std::vector<double>& p) {
            std::vector<double> stats = fn(p);
            if (stats.size() != n_statistics)
                throw std::invalid_argument(
                    "the statistic function must return the same number of values for every draw; "
                    "it returned " +
                    std::to_string(n_statistics) + " the first time and " +
                    std::to_string(stats.size()) + " this time");
            return stats;
        },
        std::vector<double>(n_statistics, kNaN), abort);

    chboot::Bootstrap<std::vector<double>> boot(
        data, chboot::BootstrapFit(chopt::ParameterSet(parent_parameters, kNaN), *parent_covariance));
    boot.resample_function = [&resample](const std::vector<double>& d, const chopt::ParameterSet& p,
                                         chboot::MersenneTwister& prng) {
        return resample(d, p.values, prng);
    };
    boot.fit_with_covariance_function = [&fit_with_covariance](const std::vector<double>& d) {
        return fit_with_covariance(d);
    };
    boot.statistic_function = [&statistic](const chopt::ParameterSet& p) {
        return statistic(p.values);
    };

    if (o.contains("replicates")) boot.replicates = o.at("replicates").as_int();
    if (o.contains("seed")) boot.prng_seed = o.at("seed").as_int();
    if (o.contains("prng_seed")) boot.prng_seed = o.at("prng_seed").as_int();
    if (o.contains("max_retries")) boot.max_retries = o.at("max_retries").as_int();

    if (o.contains("pivotal_links")) {
        chboot::Bootstrap<std::vector<double>>::LinkArray links;
        for (const JsonValue& entry : o.at("pivotal_links").items())
            links.push_back(bootstrap_pivotal_link(entry));
        // Checked here rather than left to CreatePivotalLinks, which raises the same refusal only
        // after every replicate has already called back into the host.
        if (links.size() != n_parameters)
            throw std::invalid_argument(
                "'pivotal_links' must name one link per parameter -- null for the identity; the fit "
                "has " +
                std::to_string(n_parameters) + " parameters and it holds " +
                std::to_string(links.size()));
        boot.pivotal_link_factory = [links](const chboot::PivotalBootstrapContext&) { return links; };
    }
    if (o.contains("pivotal_invalid_draw_policy")) {
        const std::string& policy = o.at("pivotal_invalid_draw_policy").as_string();
        if (policy == "drop")
            boot.pivotal_invalid_draw_policy = chboot::PivotalBootstrapInvalidDrawPolicy::Drop;
        else if (policy == "use_raw")
            boot.pivotal_invalid_draw_policy = chboot::PivotalBootstrapInvalidDrawPolicy::UseRaw;
        else if (policy == "use_parent")
            boot.pivotal_invalid_draw_policy = chboot::PivotalBootstrapInvalidDrawPolicy::UseParent;
        else
            throw std::invalid_argument("unknown pivotal_invalid_draw_policy '" + policy +
                                        "'; expected one of drop, use_raw, use_parent");
    }
    if (o.contains("regularize_pivotal_covariances"))
        boot.regularize_pivotal_covariances = o.at("regularize_pivotal_covariances").as_bool();
    if (o.contains("pivotal_z_limit")) boot.pivotal_z_limit = o.at("pivotal_z_limit").as_double();
    if (o.contains("add_pivotal_jitter")) boot.add_pivotal_jitter = o.at("add_pivotal_jitter").as_bool();
    if (o.contains("pivotal_jitter_scale"))
        boot.pivotal_jitter_scale = o.at("pivotal_jitter_scale").as_double();

    // Both halves are load-bearing here for the reason the regular run type's drive site gives.
    chboot::BootstrapResults results;
    chboot::BootstrapResults raw_results;
    try {
        boot.run_pivotal_bootstrap();
        results = boot.get_confidence_intervals(chboot::BootstrapCIMethod::Percentile, alpha);
        raw_results = boot.get_raw_pivotal_confidence_intervals(alpha);
    } catch (...) {
        rethrow_if_aborted(abort);
        throw;
    }
    rethrow_if_aborted(abort);

    CallbackResult r = bootstrap_flatten(results, boot.replicates);
    bootstrap_push_block(r, "raw_statistic", raw_results.statistic_results);
    bootstrap_push_block(r, "raw_parameter", raw_results.parameter_results);
    // Named verbatim after the PivotalBootstrapDiagnostics fields, so the two packages' result
    // dictionaries carry the DTO's own vocabulary rather than a translation of it.
    const auto& diagnostics = boot.pivotal_diagnostics();
    if (diagnostics.has_value()) {
        bootstrap_push(r, "requested_replicates", diagnostics->requested_replicates);
        bootstrap_push(r, "rejected_raw_replicates", diagnostics->rejected_raw_replicates);
        bootstrap_push(r, "failed_raw_replicates", diagnostics->failed_raw_replicates);
        bootstrap_push(r, "accepted_raw_replicates", diagnostics->accepted_raw_replicates);
        bootstrap_push(r, "invalid_pivotal_replicates", diagnostics->invalid_pivotal_replicates);
        bootstrap_push(r, "retained_pivotal_replicates", diagnostics->retained_pivotal_replicates);
    }
    return r;
}

inline CallbackResult run_bootstrap(const std::string& method, const JsonValue& o,
                                    const CallbackSet& cbs) {
    if (method != "run") throw std::invalid_argument("unknown bootstrap method: " + method);
    // The two workflows upstream's Bootstrap<TData> has. `regular` is the default and the one the
    // ci_method then picks a run for; `pivotal` is the covariance-aware mode, which fits through a
    // delegate of its own -- see the file header.
    const std::string run_type = o.value_or("run_type", "regular");
    if (run_type == "pivotal") return run_bootstrap_pivotal(o, cbs);
    if (run_type != "regular")
        throw std::invalid_argument("unknown run_type '" + run_type +
                                    "'; expected one of regular, pivotal");
    if (!cbs.data_rng)
        throw std::invalid_argument(
            "bootstrap/run requires a resample function, called with (data, parameters, rng)");
    if (!cbs.data_vector)
        throw std::invalid_argument("bootstrap/run requires a fit function, called with (data)");
    if (!cbs.vector_vector)
        throw std::invalid_argument(
            "bootstrap/run requires a statistic function, called with (parameters)");

    std::vector<double> data = require_vector(o, "data", "bootstrap/run");
    if (data.empty())
        throw std::invalid_argument("bootstrap/run requires 'data' to hold at least one value");

    const std::string ci_name = o.value_or("ci_method", "Percentile");
    const chboot::BootstrapCIMethod ci_method = chboot::parse_bootstrap_ci_method(ci_name);
    const double alpha = o.value_or("alpha", 0.1);

    // The one check moved AHEAD of the ported class's own -- see the file header on why refusing
    // this after the whole run is a different thing on a callback surface than it is in C#.
    if (ci_method == chboot::BootstrapCIMethod::BCa && !cbs.data_index)
        throw std::invalid_argument(
            "the BCa confidence interval method requires a jackknife function, called with "
            "(data, index); supply one or choose another ci_method");

    // Theta-hat: the caller's, or the fit of the original data. Called directly rather than through
    // a guard because no ported code is running yet, so an exception from it is already the user's
    // own and needs no protection to reach them.
    std::vector<double> parameters;
    if (o.contains("parameters"))
        parameters = o.at("parameters").as_double_vector();
    else
        parameters = cbs.data_vector(data);
    if (parameters.empty())
        throw std::invalid_argument(
            "the fit function must return at least one parameter; it returned nothing");
    const std::size_t n_parameters = parameters.size();

    // The up-front probe, purely to size the statistic guard's sentinel -- see the file header.
    const std::size_t n_statistics = cbs.vector_vector(parameters).size();
    if (n_statistics == 0)
        throw std::invalid_argument(
            "the statistic function must return at least one value; it returned nothing");

    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    // ONE abort state for all four guards. See the file header.
    CallbackAbortStatePtr abort = make_abort_state();

    // Each length check lives INSIDE the guarded function rather than around it, so a wrong-shaped
    // return aborts the run exactly as a host-language error does instead of surfacing as whatever
    // the ported class does with it (silently: ten thousand failed replicates and a NaN interval).
    GuardedCall<std::vector<double>, const std::vector<double>&, const std::vector<double>&,
                chboot::MersenneTwister&>
        resample(
            [fn = cbs.data_rng](const std::vector<double>& d, const std::vector<double>& p,
                                chboot::MersenneTwister& prng) {
                std::vector<double> sample = fn(d, p, prng);
                if (sample.empty())
                    throw std::invalid_argument(
                        "the resample function must return at least one value; it returned an "
                        "empty sample");
                return sample;
            },
            std::vector<double>{}, abort);

    GuardedCall<std::vector<double>, const std::vector<double>&> fit(
        [fn = cbs.data_vector, n_parameters](const std::vector<double>& d) {
            std::vector<double> fitted = fn(d);
            if (fitted.size() != n_parameters)
                throw std::invalid_argument(
                    "the fit function must return one value per parameter; the model has " +
                    std::to_string(n_parameters) + " and it returned " +
                    std::to_string(fitted.size()));
            return fitted;
        },
        std::vector<double>(n_parameters, kNaN), abort);

    GuardedCall<std::vector<double>, const std::vector<double>&> statistic(
        [fn = cbs.vector_vector, n_statistics](const std::vector<double>& p) {
            std::vector<double> stats = fn(p);
            if (stats.size() != n_statistics)
                throw std::invalid_argument(
                    "the statistic function must return the same number of values for every draw; "
                    "it returned " +
                    std::to_string(n_statistics) + " the first time and " +
                    std::to_string(stats.size()) + " this time");
            return stats;
        },
        std::vector<double>(n_statistics, kNaN), abort);

    GuardedCall<std::vector<double>, const std::vector<double>&, int> jackknife(
        cbs.data_index ? std::function<std::vector<double>(const std::vector<double>&, int)>(
                             [fn = cbs.data_index](const std::vector<double>& d, int index) {
                                 std::vector<double> sample = fn(d, index);
                                 if (sample.empty())
                                     throw std::invalid_argument(
                                         "the jackknife function must return at least one value; "
                                         "it returned an empty sample");
                                 if (sample.size() >= d.size())
                                     throw std::invalid_argument(
                                         "the jackknife function must return fewer values than it "
                                         "was given -- it leaves observation `index` out, and "
                                         "`index` counts from 0; it was given " +
                                         std::to_string(d.size()) + " and returned " +
                                         std::to_string(sample.size()));
                                 return sample;
                             })
                       : std::function<std::vector<double>(const std::vector<double>&, int)>(),
        std::vector<double>{}, abort);

    chboot::Bootstrap<std::vector<double>> boot(data, chopt::ParameterSet(parameters, kNaN));
    boot.resample_function = [&resample](const std::vector<double>& d, const chopt::ParameterSet& p,
                                         chboot::MersenneTwister& prng) {
        return resample(d, p.values, prng);
    };
    // `kNaN` is captured EXPLICITLY, and it has to be spelled out even though no capture is
    // required here by the standard: `ParameterSet(std::vector<double>, double)` takes its
    // fitness BY VALUE, so the lvalue-to-rvalue conversion on a constexpr double yields a
    // constant expression and does not odr-use it. MSVC does not implement that exemption for
    // non-integral constexpr variables and rejects the lambda outright (C3493), which broke
    // every Windows target that includes this header. Capturing it by copy is well-formed
    // everywhere and costs nothing.
    boot.fit_function = [&fit, kNaN](const std::vector<double>& d) {
        return chopt::ParameterSet(fit(d), kNaN);
    };
    boot.statistic_function = [&statistic](const chopt::ParameterSet& p) {
        return statistic(p.values);
    };
    if (cbs.data_index) {
        boot.jackknife_function = [&jackknife](const std::vector<double>& d, int index) {
            return jackknife(d, index);
        };
        // Not a user callback: `TData` is a vector of doubles here, so its size IS the sample size.
        boot.sample_size_function = [](const std::vector<double>& d) {
            return static_cast<int>(d.size());
        };
    }

    if (o.contains("replicates")) boot.replicates = o.at("replicates").as_int();
    if (o.contains("seed")) boot.prng_seed = o.at("seed").as_int();
    if (o.contains("prng_seed")) boot.prng_seed = o.at("prng_seed").as_int();
    if (o.contains("max_retries")) boot.max_retries = o.at("max_retries").as_int();
    if (o.contains("inner_replicates")) boot.inner_replicates = o.at("inner_replicates").as_int();

    // See the file header on why BOTH halves are load-bearing here, and why the rethrow is taken
    // off the shared `abort` state rather than off whichever of the four guards sits closest.
    chboot::BootstrapResults results;
    try {
        if (ci_method == chboot::BootstrapCIMethod::BootstrapT)
            boot.run_with_studentized_bootstrap();
        else
            boot.run();
        results = boot.get_confidence_intervals(ci_method, alpha);
    } catch (...) {
        rethrow_if_aborted(abort);
        throw;
    }
    rethrow_if_aborted(abort);

    return bootstrap_flatten(results, boot.replicates);
}

}  // namespace corehydro::numerics::support::detail
