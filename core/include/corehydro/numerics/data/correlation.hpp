// ported from: Numerics/Data/Statistics/Correlation.cs @ 2a0357a
//
// Pearson, Spearman, and Kendall's Tau correlation coefficients for two equal-length
// samples, plus the Pearson and Spearman column-pairwise correlation MATRIX overloads
// (Pearson(double[,]) / Spearman(double[,]), lines 87/169) over a set of p columns of n
// observations each. Feeds copula parameter constraint bounds and dependence estimation.
//
// Upstream has NO KendallsTau(double[,]) matrix overload -- the matrix severance is exactly
// those two methods; there is nothing to port for Kendall here.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"

namespace corehydro::numerics::data {

// Computes the Pearson correlation coefficient.
inline double pearson(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    if (sample2.size() != sample1.size())
        throw std::invalid_argument("The sample arrays must be the same length.");

    int n = static_cast<int>(sample1.size());
    // Find means.
    double ax = 0.0, ay = 0.0;
    for (int i = 0; i < n; ++i) {
        ax += sample1[static_cast<std::size_t>(i)];
        ay += sample2[static_cast<std::size_t>(i)];
    }
    ax /= n;
    ay /= n;
    // Compute the correlation coefficient.
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (int i = 0; i < n; ++i) {
        double xt = sample1[static_cast<std::size_t>(i)] - ax;
        double yt = sample2[static_cast<std::size_t>(i)] - ay;
        sxx += xt * xt;
        syy += yt * yt;
        sxy += xt * yt;
    }
    return sxy / std::sqrt(sxx * syy);
}

// Computes the Spearman ranked correlation coefficient.
inline double spearman(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    if (sample2.size() != sample1.size())
        throw std::invalid_argument("The sample arrays must be the same length.");

    std::vector<double> rank1 = ranks_in_place(sample1);
    std::vector<double> rank2 = ranks_in_place(sample2);
    return pearson(rank1, rank2);
}

// Computes the Pearson correlation coefficient matrix for the variables in a set of columns.
// `columns` holds p columns of n observations each; returns the p-by-p matrix C where C[j][k]
// is the Pearson correlation between column j and column k.
//
// Transcribed LITERALLY from Correlation.Pearson(double[,]) rather than as p*(p-1)/2 calls to
// pearson() above: one mean pass over all p columns, then ONE fused pass accumulating each
// column's sum of squares (ss[j]) AND the upper-triangular cross-products (cov[j][k]) together
// in the same i-loop, mirrors the triangle to fill the lower half, then divides. The
// accumulation order differs from pearson()'s own two-pass loop (which computes sxx/syy/sxy for
// exactly one pair at a time, from that pair's own two-column mean), so a "simpler" loop that
// just called pearson() p*(p-1)/2 times would drift in the last bits from the real C# matrix
// overload.
//
// Upstream has NO guard on n == 0 (the column means become NaN, silently); mirrored here rather
// than added. Upstream's `samples` is a genuine [n, p] 2D array so every column is structurally
// the same length; this port's columns are independent std::vectors, so unlike upstream a
// mismatched column length is possible and is guarded against explicitly (a corehydro addition,
// not a C# behavior to reproduce).
inline std::vector<std::vector<double>> pearson_matrix(
    const std::vector<std::vector<double>>& columns) {
    if (columns.empty())
        throw std::invalid_argument("Input must have at least one column.");

    const int p = static_cast<int>(columns.size());
    const int n = static_cast<int>(columns[0].size());
    for (const auto& col : columns) {
        if (static_cast<int>(col.size()) != n)
            throw std::invalid_argument("All columns must have the same length.");
    }

    // 1) Compute means for each column.
    std::vector<double> means(static_cast<std::size_t>(p), 0.0);
    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < n; ++i)
            means[static_cast<std::size_t>(j)] +=
                columns[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
    }
    for (int j = 0; j < p; ++j) means[static_cast<std::size_t>(j)] /= n;

    // 2) Compute sum of squares (ss) and cross-products (cov).
    std::vector<double> ss(static_cast<std::size_t>(p), 0.0);
    std::vector<std::vector<double>> cov(static_cast<std::size_t>(p),
                                         std::vector<double>(static_cast<std::size_t>(p), 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            double dx = columns[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] -
                       means[static_cast<std::size_t>(j)];
            ss[static_cast<std::size_t>(j)] += dx * dx;
            for (int k = j; k < p; ++k) {
                double dy = columns[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] -
                           means[static_cast<std::size_t>(k)];
                cov[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] += dx * dy;
            }
        }
    }
    // Mirror the upper triangle to the lower triangle.
    for (int j = 0; j < p; ++j)
        for (int k = j + 1; k < p; ++k)
            cov[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] =
                cov[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];

    // 3) Build the correlation matrix.
    std::vector<std::vector<double>> corr(static_cast<std::size_t>(p),
                                          std::vector<double>(static_cast<std::size_t>(p), 0.0));
    for (int j = 0; j < p; ++j) {
        for (int k = 0; k < p; ++k) {
            corr[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] =
                cov[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] /
                std::sqrt(ss[static_cast<std::size_t>(j)] * ss[static_cast<std::size_t>(k)]);
        }
    }

    return corr;
}

// Computes the Spearman correlation coefficient matrix: rank-transforms each column with the
// NO-TIES ranks_in_place() overload (the one already ported before P4, not P4 Task 1's new
// tie-returning overload) and delegates to pearson_matrix(), exactly as upstream's
// Spearman(double[,]) rank-transforms with Statistics.RanksInPlace() then delegates to
// Pearson(double[,]).
inline std::vector<std::vector<double>> spearman_matrix(
    const std::vector<std::vector<double>>& columns) {
    if (columns.empty())
        throw std::invalid_argument("Input must have at least one column.");

    std::vector<std::vector<double>> ranks;
    ranks.reserve(columns.size());
    for (const auto& col : columns) ranks.push_back(ranks_in_place(col));
    return pearson_matrix(ranks);
}

// Computes Kendall's Tau ranked correlation coefficient (the O(n^2) direct pair count).
inline double kendalls_tau(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    if (sample2.size() != sample1.size())
        throw std::invalid_argument("The sample arrays must be the same length.");

    int n = static_cast<int>(sample1.size());
    int i = 0, n1 = 0, n2 = 0;
    // Loop over first member of pair and second member.
    for (int j = 0; j <= n - 2; ++j) {
        for (int k = j + 1; k <= n - 1; ++k) {
            double a1 = sample1[static_cast<std::size_t>(j)] - sample1[static_cast<std::size_t>(k)];
            double a2 = sample2[static_cast<std::size_t>(j)] - sample2[static_cast<std::size_t>(k)];
            double aa = a1 * a2;
            if (aa != 0.0) {
                // Neither has a tie.
                n1 += 1;
                n2 += 1;
                i = aa > 0.0 ? i + 1 : i - 1;
            } else {
                // One or both arrays have ties.
                if (a1 != 0.0) n1 += 1;
                if (a2 != 0.0) n2 += 1;
            }
        }
    }
    return static_cast<double>(i) / (std::sqrt(static_cast<double>(n1)) * std::sqrt(static_cast<double>(n2)));
}

}  // namespace corehydro::numerics::data
