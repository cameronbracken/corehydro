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
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/bootstrap/bootstrap.hpp"
#include "corehydro/numerics/sampling/bootstrap/bootstrap_results.hpp"
#include "corehydro/numerics/sampling/bootstrap/ci_method_names.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/support/callback/common.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"

namespace corehydro::numerics::support::detail {

namespace chboot = corehydro::numerics::sampling;
namespace chopt = corehydro::numerics::math::optimization;

// Flattens one finished BootstrapResults into the layout this file's header documents.
inline CallbackResult bootstrap_flatten(const chboot::BootstrapResults& results, int replicates) {
    CallbackResult r;
    r.status = "Success";

    auto push = [&r](const std::string& name, double value) {
        r.names.push_back(name);
        r.values.push_back(value);
    };
    auto push_block = [&push](const char* label,
                              const std::vector<chboot::BootstrapStatisticResult>& block) {
        for (std::size_t i = 0; i < block.size(); ++i) {
            const std::string ix = "[" + std::to_string(i) + "]";
            push(std::string(label) + ix, block[i].population_estimate);
            push(std::string(label) + "_lower" + ix, block[i].lower_ci);
            push(std::string(label) + "_upper" + ix, block[i].upper_ci);
            push(std::string(label) + "_se" + ix, block[i].standard_error);
            push(std::string(label) + "_mean" + ix, block[i].mean);
            push(std::string(label) + "_valid" + ix, static_cast<double>(block[i].valid_count));
        }
    };

    push("replicates", static_cast<double>(replicates));
    push("failed_replicates", static_cast<double>(results.failed_replicates));
    push("alpha", results.alpha);
    push_block("statistic", results.statistic_results);
    push_block("parameter", results.parameter_results);

    r.dims = {static_cast<int>(results.statistic_results.size()),
              static_cast<int>(results.parameter_results.size())};
    return r;
}

inline CallbackResult run_bootstrap(const std::string& method, const JsonValue& o,
                                    const CallbackSet& cbs) {
    if (method != "run") throw std::invalid_argument("unknown bootstrap method: " + method);
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
    boot.fit_function = [&fit](const std::vector<double>& d) {
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
