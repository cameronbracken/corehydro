// ported from: Numerics/Data/Statistics/HypothesisTests.cs @ 2a0357a
//
// Twelve of the class's thirteen public statics -- every one-sample/two-sample parametric and
// nonparametric hypothesis test upstream ships, over the P4 Task 1 helpers (mean_variance, the
// tie-returning ranks_in_place, the vector percentile -- percentile itself is not used here, but
// the other two are load-bearing) plus numerics::pow (NOT std::pow -- see tools.hpp's own note) and
// numerics::sqr.
//
// Deliberately NOT ported (P4):
//   - UnimodalityTest: it fits Numerics.MachineLearning.GaussianMixtureModel at k = 1 and k = 2
//     (each Train(12345, true)) and forms the likelihood-ratio statistic against ChiSquared(3).
//     The Machine Learning layer is unported; P5 ports it and un-gates this method together with
//     RMC.BestFit DataFrame::summary_hypothesis_test, which cannot ship without it (see
//     models/data_frame/data_frame.hpp).
//
// Every C# ArgumentException guard is preserved with its exact message text, and every C#
// integer-arithmetic quirk that is oracle-visible is transcribed rather than "cleaned up":
//   - MannWhitneyTest: C# `double eU = n1 * n2 / 2;` performs the division in `int` (both n1
//     and n2 are `int`, and the literal `2` is an `int`) and only THEN converts to `double` --
//     so an odd `n1 * n2` truncates toward zero before assignment. Reproduced verbatim below via
//     `static_cast<double>((n1 * n2) / 2)`. `double varU = n1 * n2 / 12d * (...)` differs: `12d`
//     is a C# `double` literal, so `n1 * n2` (still `int * int = int`) is promoted to `double`
//     BEFORE the division -- no truncation there. `double V = R - n1 * (n1 + 1d) / 2d;` similarly
//     has `2d` and `1d` double literals -- no truncation.
//   - MannKendallTest: `Math.Sign(sample[j] - sample[i])` and the later `Math.Sign(S)` are C#'s
//     int-returning sign (-1/0/1); mirrored by a local `sign_of` helper, not `std::signbit` or
//     `tools::sign` (a different, two-argument function already in this codebase).
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/autocorrelation.hpp"
#include "corehydro/numerics/data/regression/linear_regression.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/chi_squared.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/student_t.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/math/special/beta.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::data::hypothesis_tests {

namespace detail {
// Mirrors C# Math.Sign(double) for the finite inputs MannKendallTest evaluates it on
// (Math.Sign throws for NaN; that edge case is not reachable from a finite sample difference).
inline int sign_of(double x) { return (x > 0.0) - (x < 0.0); }
}  // namespace detail

// The one sample t-Test compares the mean of the sample to a hypothesized population mean.
// Returns the 2-sided p-value of the test statistic.
inline double one_sample_t_test(const std::vector<double>& sample, double population_mean = 0.0) {
    if (sample.size() < 2)
        throw std::invalid_argument("Sample must have at least 2 observations.");
    auto [mean, var] = data::mean_variance(sample);
    int N = static_cast<int>(sample.size());
    double se = std::sqrt(var) / std::sqrt(static_cast<double>(N));
    double t = std::fabs((mean - population_mean) / se);
    distributions::StudentT tdist(static_cast<double>(N - 1));
    return (1.0 - tdist.cdf(t)) * 2.0;
}

// The t-test determines if there is a significant difference between the means of two samples
// drawn from populations with equal variances. Returns the 2-sided p-value.
inline double equal_variance_t_test(const std::vector<double>& s1, const std::vector<double>& s2) {
    if (s1.size() + s2.size() < 3)
        throw std::invalid_argument("Combined sample size must be at least 3.");
    auto [m1, v1] = data::mean_variance(s1);
    auto [m2, v2] = data::mean_variance(s2);
    double N1 = static_cast<double>(s1.size());
    double N2 = static_cast<double>(s2.size());
    double df = N1 + N2 - 2.0;
    double svar = ((N1 - 1.0) * v1 + (N2 - 1.0) * v2) / df;
    double t = std::fabs((m1 - m2) / std::sqrt(svar * (1.0 / N1 + 1.0 / N2)));
    distributions::StudentT tdist(df);
    return (1.0 - tdist.cdf(t)) * 2.0;
}

// The t-test determines if there is a significant difference between the means of two samples
// drawn from populations with unequal variances. Returns the 2-sided p-value. (No count guard
// upstream -- none ported here either.)
inline double unequal_variance_t_test(const std::vector<double>& s1, const std::vector<double>& s2) {
    auto [ave1, var1] = data::mean_variance(s1);
    auto [ave2, var2] = data::mean_variance(s2);
    double n1 = static_cast<double>(s1.size());
    double n2 = static_cast<double>(s2.size());
    double t = std::fabs((ave1 - ave2) / std::sqrt(var1 / n1 + var2 / n2));
    double df = numerics::sqr(var1 / n1 + var2 / n2) /
                (numerics::sqr(var1 / n1) / (n1 - 1.0) + numerics::sqr(var2 / n2) / (n2 - 1.0));
    distributions::StudentT tdist(df);
    return (1.0 - tdist.cdf(t)) * 2.0;
}

// The paired t-test determines whether the mean difference between two sets of observations is
// zero. Returns the 2-sided p-value.
inline double paired_t_test(const std::vector<double>& s1, const std::vector<double>& s2) {
    if (s1.size() != s2.size())
        throw std::invalid_argument("The two data samples must be the sample length.");
    auto [ave1, var1] = data::mean_variance(s1);
    auto [ave2, var2] = data::mean_variance(s2);
    int n = static_cast<int>(s1.size());
    double cov = 0.0;
    for (int j = 0; j < n; ++j) cov += (s1[static_cast<std::size_t>(j)] - ave1) * (s2[static_cast<std::size_t>(j)] - ave2);
    double df = static_cast<double>(n - 1);
    cov /= df;
    double sd = std::sqrt((var1 + var2 - 2.0 * cov) / static_cast<double>(n));
    double t = std::fabs((ave1 - ave2) / sd);
    distributions::StudentT tdist(df);
    return (1.0 - tdist.cdf(t)) * 2.0;
}

// The F-test for significantly different variances. Returns the p-value of the test statistic.
inline double f_test(const std::vector<double>& s1, const std::vector<double>& s2) {
    if (s1.size() < 2 || s2.size() < 2)
        throw std::invalid_argument("Each sample must have at least 2 observations.");
    auto [ave1, var1] = data::mean_variance(s1);
    auto [ave2, var2] = data::mean_variance(s2);
    double n1 = static_cast<double>(s1.size());
    double n2 = static_cast<double>(s2.size());
    double df1, df2, f, pVal;
    if (var1 == 0.0 && var2 == 0.0) return 1.0;
    if (var1 > var2) {
        f = var1 / var2;
        df1 = n1 - 1.0;
        df2 = n2 - 1.0;
    } else {
        f = var2 / var1;
        df1 = n2 - 1.0;
        df2 = n1 - 1.0;
    }
    pVal = 2.0 * math::special::beta::incomplete(0.5 * df2, 0.5 * df1, df2 / (df2 + df1 * f));
    if (pVal > 1.0) pVal = 2.0 - pVal;
    return pVal;
}

// The F-test comparing two models. The null states that the restricted and full models are
// equal; sets `f_stat` and `p_value` through the output references.
inline void f_test_models(double sse_restricted, double sse_full, int df_restricted, int df_full,
                           double& f_stat, double& p_value) {
    if (df_restricted == df_full)
        throw std::invalid_argument("Restricted and full model degrees of freedom cannot be equal.");
    if (df_full <= 0)
        throw std::invalid_argument("Full model degrees of freedom must be positive.");
    f_stat = ((sse_restricted - sse_full) / static_cast<double>(df_restricted - df_full)) /
             (sse_full / static_cast<double>(df_full));
    p_value = 2.0 * math::special::beta::incomplete(0.5 * static_cast<double>(df_full),
                                                      0.5 * static_cast<double>(df_restricted),
                                                      static_cast<double>(df_full) /
                                                          (static_cast<double>(df_full) +
                                                           static_cast<double>(df_restricted) * f_stat));
    if (p_value > 1.0) p_value = 2.0 - p_value;
}

// The Jarque-Bera test for normality. Returns the p-value of the JB statistic, a chi-squared
// random variable with 2 degrees of freedom.
inline double jarque_bera_test(const std::vector<double>& sample) {
    auto moments = data::product_moments(sample);
    double S2 = moments[2] * moments[2];
    double K2 = moments[3] * moments[3];
    double N = static_cast<double>(sample.size());
    double JB = N / 6.0 * (S2 + K2 / 4.0);
    distributions::ChiSquared chi(2);
    return 1.0 - chi.cdf(JB);
}

// The Wald and Wolfowitz test for independence and stationarity (trend). Returns the 2-sided
// p-value.
inline double wald_wolfowitz_test(const std::vector<double>& sample) {
    double xn = sample.back(), x1 = sample.front();
    double R = 0, eR, varR, U, S1 = 0, S2 = 0, S3 = 0, S4 = 0;
    int n = static_cast<int>(sample.size());
    for (int i = 0; i < n; ++i) {
        double xi = sample[static_cast<std::size_t>(i)];
        S1 += xi;
        S2 += numerics::sqr(xi);
        S3 += numerics::pow(xi, 3);
        S4 += numerics::pow(xi, 4);
        if (i < n - 1) R += xi * sample[static_cast<std::size_t>(i + 1)];
    }
    R += xn * x1;
    double s12 = numerics::sqr(S1);
    double s22 = numerics::sqr(S2);
    double s14 = numerics::pow(S1, 4);

    eR = (s12 - S2) / static_cast<double>(n - 1);
    varR = (s22 - S4) / static_cast<double>(n - 1) - numerics::sqr(eR);
    varR += (s14 - 4 * s12 * S2 + 4 * S1 * S3 + s22 - 2 * S4) /
            (static_cast<double>(n - 1) * static_cast<double>(n - 2));
    U = std::fabs((R - eR) / std::sqrt(varR));
    return (1.0 - distributions::Normal::standard_cdf(U)) * 2.0;
}

// The Ljung-Box test whether the autocorrelations of the data are different from zero. Returns
// the p-value of the test statistic.
inline double ljung_box_test(const std::vector<double>& sample, int lag_max = -1) {
    int n = static_cast<int>(sample.size());
    if (lag_max < 0)
        lag_max = static_cast<int>(
            std::floor(std::min(10.0 * std::log10(static_cast<double>(n)), static_cast<double>(n - 1))));
    auto acf = Autocorrelation::function(sample, lag_max, Autocorrelation::Type::Correlation);
    if (!acf) return std::numeric_limits<double>::quiet_NaN();
    double Q = 0;
    for (int k = 1; k <= lag_max; ++k)
        Q += numerics::sqr((*acf)[static_cast<std::size_t>(k)][1]) / static_cast<double>(n - k);
    Q *= static_cast<double>(n) * static_cast<double>(n + 2);
    distributions::ChiSquared chi2(lag_max);
    return 1.0 - chi2.cdf(Q);
}

// The Mann-Whitney test for homogeneity and stationarity (jump). `sample1` must be less than or
// equal in length to `sample2`. Returns the p-value of the test statistic.
inline double mann_whitney_test(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    int n1 = static_cast<int>(sample1.size());
    int n2 = static_cast<int>(sample2.size());
    int n = n1 + n2;
    if (n1 > n2)
        throw std::invalid_argument("The first sample must have a length less than or equal to sample 2.");
    if (n1 <= 3 || n2 <= 3) throw std::invalid_argument("Each sample must have a length greater than 3.");
    if (n <= 20) throw std::invalid_argument("The combined sample 1 & 2 must have a length greater than 20.");

    std::vector<double> sample;
    sample.reserve(sample1.size() + sample2.size());
    sample.insert(sample.end(), sample1.begin(), sample1.end());
    sample.insert(sample.end(), sample2.begin(), sample2.end());

    std::vector<double> ties;
    double R = 0, T = 0;
    auto ranks = data::ranks_in_place(sample, ties);

    for (std::size_t i = 0; i < sample1.size(); ++i) R += ranks[i];
    for (std::size_t i = 0; i < ties.size(); ++i)
        T += (numerics::pow(ties[i], 3) - ties[i]) / (static_cast<double>(n) * static_cast<double>(n - 1));

    // C# `double V = R - n1 * (n1 + 1d) / 2d;`: the `1d`/`2d` literals promote to double before
    // any division -- no integer truncation.
    double V = R - static_cast<double>(n1) * (static_cast<double>(n1) + 1.0) / 2.0;
    double W = static_cast<double>(n1) * static_cast<double>(n2) - V;
    double U = std::min(V, W);
    // C# `double eU = n1 * n2 / 2;`: `n1 * n2 / 2` is entirely `int` arithmetic (truncating
    // toward zero) before the assignment converts it to `double`. Reproduced verbatim.
    double eU = static_cast<double>((n1 * n2) / 2);
    // C# `double varU = n1 * n2 / 12d * (...)`: `12d` is a double literal, so `n1 * n2` promotes
    // to double BEFORE the division -- no truncation here.
    double varU = static_cast<double>(n1) * static_cast<double>(n2) / 12.0 * (static_cast<double>(n + 1) - T);
    double z = std::fabs((U - eU) / std::sqrt(varU));
    return (1.0 - distributions::Normal::standard_cdf(z)) * 2.0;
}

// The Mann-Kendall test for homogeneity and stationarity (trend). Returns the 2-sided p-value.
inline double mann_kendall_test(const std::vector<double>& sample) {
    int n = static_cast<int>(sample.size());
    if (n < 10) throw std::invalid_argument("The sample size must be greater than or equal to 10.");

    double S = 0, T = 0;
    for (int i = 0; i < n - 1; ++i)
        for (int j = i + 1; j < n; ++j)
            S += static_cast<double>(
                detail::sign_of(sample[static_cast<std::size_t>(j)] - sample[static_cast<std::size_t>(i)]));

    std::vector<double> ties;
    data::ranks_in_place(sample, ties);  // ranks themselves are unused, matching C#
    for (std::size_t i = 0; i < ties.size(); ++i) T += ties[i] * (ties[i] - 1.0) * (2.0 * ties[i] + 5.0);
    double varS = (static_cast<double>(n) * static_cast<double>(n - 1) * static_cast<double>(2 * n + 5) - T) / 18.0;
    double z = std::fabs((S - static_cast<double>(detail::sign_of(S))) / std::sqrt(varS));

    return (1.0 - distributions::Normal::standard_cdf(z)) * 2.0;
}

// The linear trend test for stationarity (trend). Returns the 2-sided p-value.
inline double linear_trend_test(const std::vector<double>& indices, const std::vector<double>& sample) {
    if (indices.size() != sample.size())
        throw std::invalid_argument("The indices must be the same length as the sample data");
    math::linalg::Matrix x_vals(static_cast<int>(indices.size()), 1, indices);
    math::linalg::Vector y_vals(sample);
    data::regression::LinearRegression lm(x_vals, y_vals, true);
    distributions::StudentT tdist(static_cast<double>(lm.degrees_of_freedom()));
    return (1.0 -
            tdist.cdf(std::fabs(lm.parameters()[1] / lm.parameter_standard_errors()[1]))) *
           2.0;
}

}  // namespace corehydro::numerics::data::hypothesis_tests
