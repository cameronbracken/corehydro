// corehydro ADDITION -- toolbox_runner.hpp group header, no upstream C# counterpart.
//
// Holds the `statistics` group's dispatch arm: the running/summary accumulator (with its
// "resume from a prior state" path), product moments, L-moments, ranks, percentiles, and running
// covariance, plus flatten_matrix, the one helper it needs to pack matrix blocks into a flat
// ToolboxResult. Included by toolbox_runner.hpp, which defines the shared ToolboxResult/data_at/
// scalar helpers used here; not meant to be included directly.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/running_covariance_matrix.hpp"
#include "corehydro/numerics/data/running_statistics.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"

namespace corehydro::numerics::support::detail {

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
        // No names: a single "n" label for one value out of 1 + size + 5*size^2 would invite a
        // future label-based lookup to silently read only position 0. The blocked layout is
        // documented above and addressed positionally (mirroring "spectra.autocorrelation",
        // the other dims-shaped result), not by name.
        r.dims = {size, size};
        return r;
    }
    throw std::runtime_error("unknown statistics method: " + method);
}

}  // namespace corehydro::numerics::support::detail
