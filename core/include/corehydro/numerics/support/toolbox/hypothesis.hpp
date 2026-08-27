// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `hypothesis` group's dispatch arms over the thirteen ported hypothesis tests
// (numerics/data/hypothesis_tests.hpp, itself a port of the C# `Numerics.Data.Statistics.
// HypothesisTests` static class). Every method but `f_models` reads one data vector (a
// one-sample test) or two (a two-sample test) and returns `detail::scalar(p)`, the 2-sided
// p-value; `f_models` takes NO data (its four inputs -- `sse_restricted`, `sse_full`,
// `df_restricted`, `df_full` -- are all scalar options) and returns a NAMED two-value result,
// `names = {"f_statistic", "p_value"}`.
//
// Each arm inherits the exact C# `ArgumentException` guard `hypothesis_tests.hpp` preserves
// (see that header's own file comment for the two integer-arithmetic quirks and the `Math.Sign`
// mirror it transcribes verbatim rather than "cleans up"):
//   one_sample_t          sample.size() >= 2
//   equal_variance_t       combined sample size >= 3
//   unequal_variance_t     no guard (mirrors upstream: there is none there either)
//   paired_t               the two samples the same length
//   f                      each sample >= 2 observations
//   f_models               df_restricted != df_full; df_full > 0
//   jarque_bera            no guard
//   wald_wolfowitz          no guard
//   mann_kendall             sample.size() >= 10
//   ljung_box                no guard (returns NaN if the ACF cannot be computed)
//   mann_whitney              sample1.size() <= sample2.size(); each > 3; combined > 20
//   linear_trend              indices and sample the same length
//   unimodality               sample.size() >= 10 (and returns NaN, not a throw, if either
//                             internal GaussianMixtureModel fit fails -- upstream catches)
#pragma once

#include <string>
#include <vector>

#include "corehydro/numerics/data/hypothesis_tests.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

namespace ht = corehydro::numerics::data::hypothesis_tests;

inline ToolboxResult run_hypothesis(const std::string& method,
                                    const std::vector<std::vector<double>>& data,
                                    const JsonValue& options) {
    if (method == "one_sample_t") {
        double population_mean = options.value_or("population_mean", 0.0);
        return scalar(
            ht::one_sample_t_test(data_at(data, 0, "hypothesis", method), population_mean));
    }
    if (method == "equal_variance_t")
        return scalar(ht::equal_variance_t_test(data_at(data, 0, "hypothesis", method),
                                                 data_at(data, 1, "hypothesis", method)));
    if (method == "unequal_variance_t")
        return scalar(ht::unequal_variance_t_test(data_at(data, 0, "hypothesis", method),
                                                    data_at(data, 1, "hypothesis", method)));
    if (method == "paired_t")
        return scalar(ht::paired_t_test(data_at(data, 0, "hypothesis", method),
                                         data_at(data, 1, "hypothesis", method)));
    if (method == "f")
        return scalar(ht::f_test(data_at(data, 0, "hypothesis", method),
                                  data_at(data, 1, "hypothesis", method)));
    if (method == "f_models") {
        double sse_restricted = options.at("sse_restricted").as_double();
        double sse_full = options.at("sse_full").as_double();
        int df_restricted = options.at("df_restricted").as_int();
        int df_full = options.at("df_full").as_int();
        double f_stat = 0.0, p_value = 0.0;
        ht::f_test_models(sse_restricted, sse_full, df_restricted, df_full, f_stat, p_value);
        ToolboxResult r;
        r.values = {f_stat, p_value};
        r.names = {"f_statistic", "p_value"};
        return r;
    }
    if (method == "jarque_bera")
        return scalar(ht::jarque_bera_test(data_at(data, 0, "hypothesis", method)));
    if (method == "wald_wolfowitz")
        return scalar(ht::wald_wolfowitz_test(data_at(data, 0, "hypothesis", method)));
    if (method == "mann_kendall")
        return scalar(ht::mann_kendall_test(data_at(data, 0, "hypothesis", method)));
    if (method == "ljung_box") {
        int lag_max = options.value_or("lag_max", -1);
        return scalar(ht::ljung_box_test(data_at(data, 0, "hypothesis", method), lag_max));
    }
    if (method == "mann_whitney")
        return scalar(ht::mann_whitney_test(data_at(data, 0, "hypothesis", method),
                                             data_at(data, 1, "hypothesis", method)));
    if (method == "linear_trend")
        return scalar(ht::linear_trend_test(data_at(data, 0, "hypothesis", method),
                                             data_at(data, 1, "hypothesis", method)));
    if (method == "unimodality")
        return scalar(ht::unimodality_test(data_at(data, 0, "hypothesis", method)));

    throw std::runtime_error("unknown hypothesis method: " + method);
}

}  // namespace corehydro::numerics::support::detail
