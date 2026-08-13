// ported from: Numerics/Data/Statistics/Autocorrelation.cs @ 2a0357a
//
// Computes the autocovariance, autocorrelation, or partial autocorrelation function, plus the
// asymptotic confidence band for ACF/PACF rho values. Unported through corehydro's phases 0-10
// (the port that reached FULL PARITY on every distribution/estimator/model/analysis surface never
// needed it); this task ports it as the first entry in the Numerics utility-toolbox surface.
//
// Only the IList<double> overloads of Function/Covariance/Correlation/Partial are ported; the
// TimeSeries overloads are a documented severance. This port's `numerics/data/time_series/
// time_series.hpp` (Phase 7a) is a thin adapter with no mutation surface tailored to this class,
// and no caller identified so far -- including Fourier's own `autocorrelation()`
// (math/fourier/fourier.hpp), which already works from a plain `vector<double>` -- needs the
// TimeSeries entry point. Add it if a later target does.
//
// C# defaults `lagMax` to `floor(min(10*log10(n), n-1))` when the caller passes -1. n == 0 makes
// log10(n) == -inf; casting that to int is unspecified in C# but undefined behavior in C++
// (math/fourier/fourier.hpp's `autocorrelation()` carries the identical guard already -- see that
// file's header for the full argument), so `default_lag_max` below special-cases n < 1 to a
// sentinel that the immediate `lag_max < 1 || n < 2` check in every caller discards regardless,
// matching observable C# behavior bit for bit.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/normal.hpp"

namespace corehydro::numerics::data {

class Autocorrelation {
   public:
    // Enumeration of the type of autocorrelation.
    enum class Type {
        Correlation,  // Autocorrelation
        Covariance,   // Autocovariance
        Partial       // Partial autocorrelation
    };

    // Computes the autocovariance, autocorrelation, or partial autocorrelation function.
    // `lag_max` defaults to `floor(min(10*log10(N), N-1))` where N is the number of
    // observations, and is automatically limited to one less than N. Returns a vector of
    // {lag, value} pairs (mirroring the C# n x 2 matrix), or std::nullopt where C# returns
    // null (lag_max < 1, or n < 2 after defaulting).
    static std::optional<std::vector<std::array<double, 2>>> function(const std::vector<double>& data,
                                                                       int lag_max = -1,
                                                                       Type type = Type::Correlation) {
        if (type == Type::Correlation) {
            return correlation(data, lag_max);
        } else if (type == Type::Covariance) {
            return covariance(data, lag_max);
        } else if (type == Type::Partial) {
            return partial(data, lag_max);
        }
        return std::nullopt;
    }

    // Get confidence interval for ACF and PACF rho values.
    static std::vector<double> correlation_confidence_interval(int sample_size, double interval = 0.95) {
        double alpha = 0.5 * (1.0 - interval);
        double lo = distributions::Normal::standard_z(alpha) / std::sqrt(static_cast<double>(sample_size));
        double hi = distributions::Normal::standard_z(1.0 - alpha) / std::sqrt(static_cast<double>(sample_size));
        return {lo, hi};
    }

   private:
    static int default_lag_max(int n) {
        if (n < 1) return -1;  // see the file header: avoids UB from log10(0) == -inf
        return static_cast<int>(
            std::floor(std::min(10.0 * std::log10(static_cast<double>(n)), static_cast<double>(n - 1))));
    }

    // Compute the autocovariance function.
    static std::optional<std::vector<std::array<double, 2>>> covariance(const std::vector<double>& data,
                                                                          int lag_max = -1) {
        int n = static_cast<int>(data.size());
        if (lag_max < 0) lag_max = default_lag_max(n);
        if (lag_max < 1 || n < 2) return std::nullopt;
        std::vector<std::array<double, 2>> acvf(static_cast<std::size_t>(lag_max + 1));
        double m = numerics::data::mean(data);
        for (int lag = 0; lag <= lag_max; ++lag) {
            double cov = 0.0;
            for (int t = lag; t < n; ++t)
                cov += (data[static_cast<std::size_t>(t)] - m) * (data[static_cast<std::size_t>(t - lag)] - m);
            acvf[static_cast<std::size_t>(lag)] = {static_cast<double>(lag), cov / static_cast<double>(n)};
        }
        return acvf;
    }

    // Compute the autocorrelation function.
    static std::optional<std::vector<std::array<double, 2>>> correlation(const std::vector<double>& data,
                                                                           int lag_max = -1) {
        int n = static_cast<int>(data.size());
        if (lag_max < 0) lag_max = default_lag_max(n);
        if (lag_max < 1 || n < 2) return std::nullopt;
        auto acf = covariance(data, lag_max);
        if (!acf) return std::nullopt;
        double den = (*acf)[0][1];
        if (den == 0.0) return std::nullopt;
        for (auto& row : *acf) row[1] /= den;
        return acf;
    }

    // Compute the partial autocorrelation function via the Durbin-Levinson algorithm. The
    // result has `lag_max` rows, not `lag_max + 1` (mirrors the C# `new double[lagMax, 2]`):
    // there is no lag-0 row, since the PACF at lag 0 is trivially 1 by definition.
    static std::optional<std::vector<std::array<double, 2>>> partial(const std::vector<double>& data,
                                                                       int lag_max = -1) {
        int n = static_cast<int>(data.size());
        if (lag_max < 0) lag_max = default_lag_max(n);
        if (lag_max < 1 || n < 2) return std::nullopt;
        // First compute the ACVF.
        auto acvf = covariance(data, lag_max);
        if (!acvf) return std::nullopt;
        // Then compute PACF using the Durbin-Levinson algorithm.
        std::vector<double> phis(static_cast<std::size_t>(lag_max + 1), 0.0);
        std::vector<double> phis2(static_cast<std::size_t>(lag_max + 1), 0.0);
        std::vector<std::array<double, 2>> pacf(static_cast<std::size_t>(lag_max));
        phis[0] = (*acvf)[1][1] / (*acvf)[0][1];
        pacf[0] = {1.0, phis[0]};
        double vi = (*acvf)[0][1];
        vi *= 1.0 - phis[0] * phis[0];
        for (int i = 2; i <= lag_max; ++i) {
            for (int j = 0; j < i - 1; ++j)
                phis2[static_cast<std::size_t>(j)] = phis[static_cast<std::size_t>(i - j - 2)];
            double phinn = (*acvf)[static_cast<std::size_t>(i)][1];
            for (int j = 1; j < i; ++j)
                phinn -= phis[static_cast<std::size_t>(j - 1)] * (*acvf)[static_cast<std::size_t>(i - j)][1];
            phinn /= vi;
            for (int j = 0; j < i - 1; ++j)
                phis[static_cast<std::size_t>(j)] -= phinn * phis2[static_cast<std::size_t>(j)];
            vi *= 1.0 - phinn * phinn;
            phis[static_cast<std::size_t>(i - 1)] = phinn;
            pacf[static_cast<std::size_t>(i - 1)] = {static_cast<double>(i), phis[static_cast<std::size_t>(i - 1)]};
        }
        return pacf;
    }
};

}  // namespace corehydro::numerics::data
