// ported from: Numerics/Sampling/MCMC/Support/MCMCDiagnostics.cs @ 7e8e8d1
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/fourier/fourier.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"

namespace corehydro::numerics::sampling::mcmc {
namespace fourier = corehydro::numerics::math::fourier;
namespace statistics = corehydro::numerics::data;
using corehydro::numerics::math::optimization::ParameterSet;

struct EffectiveSampleSizeResult {
    std::vector<double> ess;
    std::vector<std::vector<std::array<double, 2>>> average_acf;
};

namespace diagnostics_detail {
inline constexpr double kRankOffset = 3.0 / 8.0;
inline constexpr double kMachineEpsilon = 2.2204460492503131e-16;

inline std::vector<double> flatten(const std::vector<std::vector<double>>& chains) {
    std::size_t total = 0;
    for (const auto& chain : chains) total += chain.size();
    std::vector<double> result;
    result.reserve(total);
    for (const auto& chain : chains) result.insert(result.end(), chain.begin(), chain.end());
    return result;
}

inline bool should_return_nan(const std::vector<std::vector<double>>& chains) {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto& chain : chains) {
        for (double value : chain) {
            if (!std::isfinite(value)) return true;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    }
    return std::abs(maximum - minimum) < kMachineEpsilon;
}

inline double sample_variance(const std::vector<double>& values) {
    if (values.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                        static_cast<double>(values.size());
    double sum = 0.0;
    for (double value : values) sum += (value - mean) * (value - mean);
    return sum / static_cast<double>(values.size() - 1);
}

inline std::vector<std::vector<double>> split_chains(
    const std::vector<std::vector<double>>& chains) {
    const std::size_t half = chains.front().size() / 2;
    std::vector<std::vector<double>> split(chains.size() * 2, std::vector<double>(half));
    for (std::size_t chain = 0; chain < chains.size(); ++chain) {
        std::copy_n(chains[chain].begin(), half, split[chain].begin());
        std::copy(chains[chain].end() - static_cast<std::ptrdiff_t>(half), chains[chain].end(),
                  split[chain + chains.size()].begin());
    }
    return split;
}

inline std::vector<std::vector<double>> rank_normalize(
    const std::vector<std::vector<double>>& chains) {
    const auto values = flatten(chains);
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return values[a] < values[b]; });
    std::vector<double> ranks(values.size());
    std::size_t start = 0;
    while (start < order.size()) {
        std::size_t end = start + 1;
        while (end < order.size() && values[order[end]] == values[order[start]]) ++end;
        const double rank = (static_cast<double>(start) + 1.0 + static_cast<double>(end)) / 2.0;
        for (std::size_t i = start; i < end; ++i) ranks[order[i]] = rank;
        start = end;
    }
    std::vector<std::vector<double>> result(chains.size());
    std::size_t flat = 0;
    const double denominator = static_cast<double>(values.size()) - 2.0 * kRankOffset + 1.0;
    for (std::size_t chain = 0; chain < chains.size(); ++chain) {
        result[chain].resize(chains[chain].size());
        for (std::size_t iteration = 0; iteration < chains[chain].size(); ++iteration) {
            const double probability = (ranks[flat] - kRankOffset) / denominator;
            result[chain][iteration] = distributions::Normal::standard_z(probability);
            ++flat;
        }
    }
    return result;
}

inline std::vector<std::vector<double>> fold_around_median(
    const std::vector<std::vector<double>>& chains) {
    const double median = statistics::percentile(flatten(chains), 0.5);
    auto result = chains;
    for (auto& chain : result)
        for (double& value : chain) value = std::abs(value - median);
    return result;
}

inline double basic_rhat(const std::vector<std::vector<double>>& chains) {
    if (chains.size() < 2 || chains.front().size() < 2 || should_return_nan(chains))
        return std::numeric_limits<double>::quiet_NaN();
    const double n = static_cast<double>(chains.front().size());
    std::vector<double> means(chains.size());
    double within = 0.0;
    for (std::size_t i = 0; i < chains.size(); ++i) {
        means[i] = std::accumulate(chains[i].begin(), chains[i].end(), 0.0) / n;
        within += sample_variance(chains[i]);
    }
    within /= static_cast<double>(chains.size());
    if (!std::isfinite(within) || within <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    const double between = n * sample_variance(means);
    return std::sqrt((between / within + n - 1.0) / n);
}

inline std::vector<double> autocovariance(const std::vector<double>& values, double mean) {
    std::size_t length = 1;
    while (length < 2 * values.size()) length <<= 1;
    std::vector<double> centered(length, 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) centered[i] = values[i] - mean;
    const auto correlation = fourier::correlation(centered, centered);
    std::vector<double> result(values.size());
    for (std::size_t lag = 0; lag < values.size(); ++lag)
        result[lag] = correlation[lag] / static_cast<double>(values.size());
    return result;
}

inline double geyer_effective_sample_size(const std::vector<std::vector<double>>& chains) {
    const std::size_t m = chains.size();
    const std::size_t n = chains.front().size();
    if (n < 3 || should_return_nan(chains)) return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> average_covariance(n, 0.0), means(m);
    for (std::size_t chain = 0; chain < m; ++chain) {
        means[chain] = std::accumulate(chains[chain].begin(), chains[chain].end(), 0.0) /
                       static_cast<double>(n);
        const auto covariance = autocovariance(chains[chain], means[chain]);
        for (std::size_t lag = 0; lag < n; ++lag)
            average_covariance[lag] += covariance[lag] / static_cast<double>(m);
    }
    const double mean_variance = average_covariance[0] * static_cast<double>(n) /
                                 static_cast<double>(n - 1);
    double variance_plus = mean_variance * static_cast<double>(n - 1) / static_cast<double>(n);
    if (m > 1) variance_plus += sample_variance(means);
    if (!std::isfinite(variance_plus) || variance_plus <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();

    std::vector<double> rho(n, 0.0);
    std::size_t index = 0;
    double even = 1.0;
    rho[0] = even;
    double odd = 1.0 - (mean_variance - average_covariance[1]) / variance_plus;
    rho[1] = odd;
    while (index < n - 5 && std::isfinite(even + odd) && even + odd > 0.0) {
        index += 2;
        even = 1.0 - (mean_variance - average_covariance[index]) / variance_plus;
        odd = 1.0 - (mean_variance - average_covariance[index + 1]) / variance_plus;
        if (even + odd >= 0.0) {
            rho[index] = even;
            rho[index + 1] = odd;
        }
    }
    const std::size_t maximum = index;
    if (even > 0.0) rho[maximum] = even;
    index = 0;
    while (index <= maximum && maximum - index >= 4) {
        index += 2;
        const double previous = rho[index - 2] + rho[index - 1];
        const double current = rho[index] + rho[index + 1];
        if (current > previous) rho[index] = rho[index + 1] = previous / 2.0;
    }
    const std::size_t sum_length = std::max(maximum, std::size_t{1});
    double sum = 0.0;
    for (std::size_t lag = 0; lag < sum_length; ++lag) sum += rho[lag];
    double tau = -1.0 + 2.0 * sum + rho[maximum];
    const double total = static_cast<double>(m * n);
    tau = std::max(tau, 1.0 / std::log10(total));
    return total / tau;
}

inline double quantile_effective_sample_size(const std::vector<std::vector<double>>& chains,
                                              double probability) {
    const double threshold = statistics::percentile(flatten(chains), probability);
    auto indicators = chains;
    for (auto& chain : indicators)
        for (double& value : chain) value = value <= threshold ? 1.0 : 0.0;
    return geyer_effective_sample_size(split_chains(indicators));
}

inline double conservative_effective_sample_size(const std::vector<std::vector<double>>& chains) {
    if (chains.empty() || chains.front().size() < 6 || should_return_nan(chains))
        return std::numeric_limits<double>::quiet_NaN();
    const double bulk = geyer_effective_sample_size(rank_normalize(split_chains(chains)));
    const double lower = quantile_effective_sample_size(chains, 0.05);
    const double upper = quantile_effective_sample_size(chains, 0.95);
    if (!std::isfinite(bulk) || !std::isfinite(lower) || !std::isfinite(upper))
        return std::numeric_limits<double>::quiet_NaN();
    return std::min(bulk, std::min(lower, upper));
}

inline void accumulate_average_autocorrelation(
    const std::vector<double>& values, std::vector<std::array<double, 2>>& result,
    std::size_t chain_count) {
    if (values.size() < 2) return;
    const auto acf = fourier::autocorrelation(
        values, static_cast<int>(std::min<std::size_t>(50, values.size() - 1)));
    if (!acf.has_value()) return;
    for (std::size_t lag = 0; lag < acf->size(); ++lag)
        result[lag][1] += (*acf)[lag][1] / static_cast<double>(chain_count);
}
}  // namespace diagnostics_detail

inline double effective_sample_size(const std::vector<double>& series) {
    return diagnostics_detail::conservative_effective_sample_size({series});
}

inline EffectiveSampleSizeResult effective_sample_size(
    const std::vector<std::vector<ParameterSet>>& chains) {
    if (chains.empty()) throw std::invalid_argument("No chains provided.");
    for (const auto& chain : chains)
        if (chain.empty()) throw std::invalid_argument("Every chain must contain at least one iteration.");
    const std::size_t parameters = chains.front().front().values.size();
    if (parameters < 1) throw std::out_of_range("There must be at least one parameter to evaluate.");
    std::size_t common_length = chains.front().size();
    for (const auto& chain : chains) common_length = std::min(common_length, chain.size());
    EffectiveSampleSizeResult result;
    result.ess.resize(parameters);
    result.average_acf.assign(parameters, std::vector<std::array<double, 2>>(51, {0.0, 0.0}));
    for (std::size_t parameter = 0; parameter < parameters; ++parameter) {
        std::vector<std::vector<double>> values(chains.size(), std::vector<double>(common_length));
        for (std::size_t chain = 0; chain < chains.size(); ++chain) {
            for (std::size_t iteration = 0; iteration < common_length; ++iteration)
                values[chain][iteration] = chains[chain][iteration].values[parameter];
            diagnostics_detail::accumulate_average_autocorrelation(
                values[chain], result.average_acf[parameter], chains.size());
        }
        result.ess[parameter] = diagnostics_detail::conservative_effective_sample_size(values);
    }
    return result;
}

inline std::vector<double> gelman_rubin(const std::vector<std::vector<ParameterSet>>& chains,
                                        int warmup_iterations = 0) {
    if (chains.empty()) throw std::invalid_argument("No chains provided.");
    for (const auto& chain : chains)
        if (chain.empty()) throw std::invalid_argument("Every chain must contain at least one iteration.");
    const std::size_t chain_count = chains.size();
    const std::size_t parameters = chains.front().front().values.size();
    const std::size_t iterations = chains.front().size();
    for (const auto& chain : chains)
        if (chain.size() != iterations) throw std::invalid_argument("All chains must have the same length.");
    if (parameters < 1) throw std::out_of_range("There must be at least one parameter to evaluate.");
    if (warmup_iterations < 0 || static_cast<std::size_t>(warmup_iterations) >= iterations)
        throw std::out_of_range("Warmup iterations must leave at least one retained iteration.");
    std::vector<double> result(parameters, std::numeric_limits<double>::quiet_NaN());
    const std::size_t retained_count = iterations - static_cast<std::size_t>(warmup_iterations);
    if (chain_count < 2 || retained_count < 4) return result;
    for (std::size_t parameter = 0; parameter < parameters; ++parameter) {
        std::vector<std::vector<double>> retained(chain_count, std::vector<double>(retained_count));
        for (std::size_t chain = 0; chain < chain_count; ++chain)
            for (std::size_t iteration = 0; iteration < retained_count; ++iteration)
                retained[chain][iteration] =
                    chains[chain][static_cast<std::size_t>(warmup_iterations) + iteration].values[parameter];
        if (diagnostics_detail::should_return_nan(retained)) continue;
        const double rank = diagnostics_detail::basic_rhat(
            diagnostics_detail::rank_normalize(diagnostics_detail::split_chains(retained)));
        const double folded = diagnostics_detail::basic_rhat(diagnostics_detail::rank_normalize(
            diagnostics_detail::split_chains(diagnostics_detail::fold_around_median(retained))));
        if (std::isfinite(rank) && std::isfinite(folded)) result[parameter] = std::max(rank, folded);
    }
    return result;
}

inline int minimum_sample_size(double quantile, double tolerance, double probability) {
    const double n = quantile * (1.0 - quantile) *
                     std::pow(distributions::Normal::standard_z(0.5 * (probability + 1.0)), 2.0) /
                     std::pow(tolerance, 2.0);
    return static_cast<int>(std::round(n / 100.0)) * 100;
}
}  // namespace corehydro::numerics::sampling::mcmc
