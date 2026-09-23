// ported from: Numerics/Data/Statistics/GlobalSensitivity.cs @ 7e8e8d1
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::data::global_sensitivity {

namespace detail {

inline void validate_samples(const std::vector<double>& x, const std::vector<double>& y,
                             int bins) {
    if (x.size() != y.size()) throw std::invalid_argument("input and output must have equal length");
    if (bins < 2) throw std::out_of_range("bin count must be at least two");
    if (x.size() < static_cast<std::size_t>(bins)) {
        throw std::invalid_argument("sample must be at least as long as the bin count");
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!corehydro::numerics::is_finite(x[i]) || !corehydro::numerics::is_finite(y[i])) {
            throw std::out_of_range("sample values must be finite");
        }
    }
}

inline std::vector<std::size_t> sort_indices_by(const std::vector<double>& values) {
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&values](std::size_t first, std::size_t second) {
        if (values[first] < values[second]) return true;
        if (values[second] < values[first]) return false;
        return first < second;
    });
    return order;
}

inline double two_sample_kolmogorov_smirnov(const std::vector<double>& first,
                                            const std::vector<double>& second) {
    std::size_t i = 0;
    std::size_t j = 0;
    double supremum = 0.0;
    while (i < first.size() && j < second.size()) {
        const double value = std::min(first[i], second[j]);
        while (i < first.size() && first[i] == value) ++i;
        while (j < second.size() && second[j] == value) ++j;
        supremum = std::max(supremum,
                            std::fabs(static_cast<double>(i) / first.size() -
                                      static_cast<double>(j) / second.size()));
    }
    return supremum;
}

}  // namespace detail

inline double first_order_sobol(const std::vector<double>& x, const std::vector<double>& y,
                                int bins = 20) {
    detail::validate_samples(x, y, bins);
    const std::size_t n = x.size();
    const auto order = detail::sort_indices_by(x);
    double overall_mean = 0.0;
    for (double value : y) overall_mean += value;
    overall_mean /= static_cast<double>(n);
    double total_variance = 0.0;
    for (double value : y) {
        const double difference = value - overall_mean;
        total_variance += difference * difference;
    }
    total_variance /= static_cast<double>(n);
    if (total_variance <= 0.0) return 0.0;

    double between_variance = 0.0;
    for (int bin = 0; bin < bins; ++bin) {
        const std::size_t start = static_cast<std::size_t>(bin) * n / bins;
        const std::size_t end = static_cast<std::size_t>(bin + 1) * n / bins;
        double bin_mean = 0.0;
        for (std::size_t i = start; i < end; ++i) bin_mean += y[order[i]];
        bin_mean /= static_cast<double>(end - start);
        const double difference = bin_mean - overall_mean;
        between_variance += static_cast<double>(end - start) * difference * difference;
    }
    between_variance /= static_cast<double>(n);
    return std::clamp(between_variance / total_variance, 0.0, 1.0);
}

inline std::vector<double> pawn(const std::vector<double>& x, const std::vector<double>& y,
                                int bins = 20) {
    detail::validate_samples(x, y, bins);
    const std::size_t n = x.size();
    const auto order = detail::sort_indices_by(x);
    std::vector<double> sorted_y(y);
    std::sort(sorted_y.begin(), sorted_y.end());
    std::vector<double> result(static_cast<std::size_t>(bins));
    for (int bin = 0; bin < bins; ++bin) {
        const std::size_t start = static_cast<std::size_t>(bin) * n / bins;
        const std::size_t end = static_cast<std::size_t>(bin + 1) * n / bins;
        std::vector<double> bin_y;
        bin_y.reserve(end - start);
        for (std::size_t i = start; i < end; ++i) bin_y.push_back(y[order[i]]);
        std::sort(bin_y.begin(), bin_y.end());
        result[static_cast<std::size_t>(bin)] =
            detail::two_sample_kolmogorov_smirnov(bin_y, sorted_y);
    }
    return result;
}

inline double pawn_median(const std::vector<double>& x, const std::vector<double>& y,
                          int bins = 20) {
    return corehydro::numerics::data::percentile(pawn(x, y, bins), 0.5);
}

inline double borgonovo_delta(const std::vector<double>& x, const std::vector<double>& y,
                              int x_bins = 20, int y_bins = 20) {
    detail::validate_samples(x, y, x_bins);
    if (y_bins < 2) throw std::out_of_range("class count must be at least two");
    const std::size_t n = x.size();
    if (n < static_cast<std::size_t>(y_bins)) {
        throw std::invalid_argument("sample must be at least as long as the class count");
    }
    const auto x_order = detail::sort_indices_by(x);
    const auto y_order = detail::sort_indices_by(y);
    std::vector<int> y_class(n);
    std::vector<int> class_counts(static_cast<std::size_t>(y_bins));
    for (int class_index = 0; class_index < y_bins; ++class_index) {
        const std::size_t start = static_cast<std::size_t>(class_index) * n / y_bins;
        const std::size_t end = static_cast<std::size_t>(class_index + 1) * n / y_bins;
        class_counts[static_cast<std::size_t>(class_index)] = static_cast<int>(end - start);
        for (std::size_t i = start; i < end; ++i) y_class[y_order[i]] = class_index;
    }
    double delta = 0.0;
    std::vector<int> conditional_counts(static_cast<std::size_t>(y_bins));
    for (int bin = 0; bin < x_bins; ++bin) {
        const std::size_t start = static_cast<std::size_t>(bin) * n / x_bins;
        const std::size_t end = static_cast<std::size_t>(bin + 1) * n / x_bins;
        const std::size_t size = end - start;
        std::fill(conditional_counts.begin(), conditional_counts.end(), 0);
        for (std::size_t i = start; i < end; ++i) {
            ++conditional_counts[static_cast<std::size_t>(y_class[x_order[i]])];
        }
        double total_variation = 0.0;
        for (int class_index = 0; class_index < y_bins; ++class_index) {
            total_variation += std::fabs(
                static_cast<double>(conditional_counts[static_cast<std::size_t>(class_index)]) / size -
                static_cast<double>(class_counts[static_cast<std::size_t>(class_index)]) / n);
        }
        delta += static_cast<double>(size) / n * total_variation;
    }
    return 0.5 * delta;
}

}  // namespace corehydro::numerics::data::global_sensitivity
