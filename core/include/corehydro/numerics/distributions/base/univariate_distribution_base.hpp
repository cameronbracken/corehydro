// ported from: Numerics/Distributions/Univariate/Base/UnivariateDistributionBase.cs @ 7e8e8d1
//
// Abstract base for every univariate distribution. Declares the distribution-core
// surface as pure virtuals (moments, support, PDF/CDF/InverseCDF, parameters) and
// promotes the shared concrete helpers (Variance, LogPDF, LogLikelihood).
//
// The desktop-app boilerplate of the C# base (XElement serialization, PDF/CDF graph
// builders, equality operators, the AdaptiveGaussKronrod CentralMoments(tolerance)
// overload, ConditionalMean/ConditionalExpectedValue) is intentionally not ported --
// those are WPF/analysis concerns, not the math core. B4 adds the ConditionalMoments
// virtual (line 374): the Bulletin 17C GMM track consumes it. B9 adds the trapezoidal
// CentralMoments(int steps = 300) virtual (line 321): DataFrame's nonparametric-moments
// methods call it with 1000 steps through an EmpiricalDistribution.
//
// P3.10 adds `generate_random_values` (C# `GenerateRandomValues(int sampleSize, int seed =
// -1)`): inverse-CDF sampling off a fresh MersenneTwister, `seed`-constructed when positive
// or clock-seeded otherwise. Bootstrap's "normal_quantiles" model registry entry (and any
// future parametric-bootstrap model) calls this through a distribution instance built from a
// trial `ParameterSet`, exactly mirroring `Test_Bootstrap.cs`'s
// `dist.GenerateRandomValues(sampleSize, rng.Next())` call sites.
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/base/distribution_numerics.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/sampling/stratification_options.hpp"
#include "corehydro/numerics/sampling/stratify.hpp"

namespace corehydro::numerics::distributions {

class UnivariateDistributionBase {
   public:
    virtual ~UnivariateDistributionBase() = default;

    // --- Identity / parameters ---
    virtual UnivariateDistributionType type() const = 0;
    virtual int number_of_parameters() const = 0;
    virtual std::vector<double> get_parameters() const = 0;
    virtual void set_parameters(const std::vector<double>& parameters) = 0;
    bool parameters_valid() const { return parameters_valid_; }

    // --- Parameter display names (X1) ---
    // C# `ParameterNames` (UnivariateDistributionBase.cs:74) derives from the column-0 display
    // names of `ParametersToString`; `ParameterNamesShortForm` (line 80) is the abstract short
    // form. Only these two name lists are ported here (the full 2-D ParametersToString value
    // table is a WPF display concern, not the math core). Each concrete distribution overrides
    // both with its per-parameter name strings (transcribed byte-for-byte from the C#, so the
    // UTF-8 glyphs µ/σ/α/... match the oracle and the name-keyed prior dedup). The base returns
    // an empty list; the composite/dynamic distributions (Mixture, CompetingRisks,
    // TruncatedDistribution) build their lists dynamically like the C#.
    virtual std::vector<std::string> parameter_names() const { return {}; }
    virtual std::vector<std::string> parameter_names_short_form() const { return {}; }

    // --- Moments / support ---
    virtual double mean() const = 0;
    virtual double median() const = 0;
    virtual double mode() const = 0;
    virtual double standard_deviation() const = 0;
    virtual double skewness() const = 0;
    virtual double kurtosis() const = 0;
    virtual double minimum() const = 0;
    virtual double maximum() const = 0;
    double variance() const {
        double s = standard_deviation();
        return s * s;
    }

    // --- Distribution functions ---
    virtual double pdf(double x) const = 0;
    virtual double cdf(double x) const = 0;
    virtual double inverse_cdf(double probability) const = 0;

    virtual double log_pdf(double x) const {
        double f = pdf(x);
        if (std::isnan(f) || std::isinf(f) || f <= 0.0) return -kInf;
        return std::log(f);
    }

    virtual double log_cdf(double x) const {
        double F = cdf(x);
        if (std::isnan(F) || std::isinf(F) || F <= 0.0) return -kInf;
        return std::log(F);
    }

    virtual double ccdf(double x) const { return 1.0 - cdf(x); }

    virtual double log_ccdf(double x) const {
        double cF = ccdf(x);
        if (std::isnan(cF) || std::isinf(cF) || cF <= 0.0) return -kInf;
        return std::log(cF);
    }

    double log_likelihood(const std::vector<double>& sample) const {
        double ll = 0.0;
        for (double v : sample) ll += log_pdf(v);
        if (std::isnan(ll) || std::isinf(ll)) return -kInf;
        return ll;
    }

    // --- Censored likelihood methods (M8, additive port of UnivariateDistributionBase.cs
    // lines 165-201 @ 2a0357a; term-for-term, C# `long` counts -> `long long`). ---

    // C# `LogLikelihood(double value)` (line 165): single data point.
    double log_likelihood(double value) const { return log_pdf(value); }

    // C# `LogLikelihood_LeftCensored(double threshold, long numberBelow)` (line 173).
    double log_likelihood_left_censored(double threshold, long long number_below) const {
        if (number_below < 0) throw std::out_of_range("censored count cannot be negative");
        if (number_below == 0) return 0.0;
        return static_cast<double>(number_below) * log_cdf(threshold);
    }

    // C# `LogLikelihood_RightCensored(double threshold, long numberAbove)` (line 183).
    double log_likelihood_right_censored(double threshold, long long number_above) const {
        if (number_above < 0) throw std::out_of_range("censored count cannot be negative");
        if (number_above == 0) return 0.0;
        return static_cast<double>(number_above) * log_ccdf(threshold);
    }

    // C# `LogLikelihood_Intervals(double lowerLimit, double upperLimit)` (line 199): stable
    // logarithm of the probability on the open-lower, closed-upper interval.
    double log_likelihood_intervals(double lower_limit, double upper_limit) const {
        if (std::isnan(lower_limit) || std::isnan(upper_limit) || lower_limit > upper_limit)
            throw std::out_of_range("interval limits must be ordered and not NaN");
        if (lower_limit == upper_limit) return -kInf;
        const double lower = log_cdf(lower_limit);
        const double upper = log_cdf(upper_limit);
        double result = lower < -0.6931471805599453
                            ? distribution_numerics::log_difference(upper, lower)
                            : distribution_numerics::log_difference(log_ccdf(lower_limit),
                                                                    log_ccdf(upper_limit));
        return result == -kInf ? distribution_numerics::collapsed_continuous_log_interval(
                                     *this, lower_limit, upper_limit)
                               : result;
    }

    // Returns the central moments {Mean, Standard Deviation, Skew, Kurtosis} of the
    // distribution using numerical integration with the trapezoidal rule (C#
    // `CentralMoments(int steps = 300)`, line 321; B9 -- DataFrame's nonparametric-moments
    // methods call it with 1000 steps). First bin's representative is its upper bound,
    // interior bins use midpoints, and the last bin uses its lower bound with
    // dF = 1 - CDF, transcribing the C# scheme term for term. Returns
    // {a, NaN, NaN, NaN} when the 1e-8 support window is degenerate (a >= b).
    virtual std::vector<double> central_moments(int steps = 300) const {
        double a = inverse_cdf(1E-8);
        double b = inverse_cdf(1 - 1E-8);
        if (a >= b) return {a, kNaN, kNaN, kNaN};

        auto bins = sampling::Stratify::XValues(sampling::StratificationOptions(a, b, steps));
        std::vector<double> d_fx(static_cast<std::size_t>(steps));
        std::vector<double> coordinates(static_cast<std::size_t>(steps));
        const double reference = a / 2.0 + b / 2.0;
        const double scale = std::max(std::fabs(a - reference), std::fabs(b - reference));

        // First compute the mean and standard deviation
        d_fx[0] = cdf(bins[0].upper_bound());
        coordinates[0] =
            distribution_numerics::standardize(bins[0].upper_bound(), reference, scale);
        for (int i = 1; i < steps - 1; i++) {
            const auto& bin = bins[static_cast<std::size_t>(i)];
            d_fx[static_cast<std::size_t>(i)] = cdf(bin.upper_bound()) - cdf(bin.lower_bound());
            coordinates[static_cast<std::size_t>(i)] =
                distribution_numerics::standardize(bin.midpoint(), reference, scale);
        }
        const auto& last = bins.back();
        d_fx[static_cast<std::size_t>(steps - 1)] = 1 - cdf(last.lower_bound());
        coordinates[static_cast<std::size_t>(steps - 1)] =
            distribution_numerics::standardize(last.lower_bound(), reference, scale);
        double offset = 0.0;
        for (int i = 0; i < steps; ++i)
            offset += coordinates[static_cast<std::size_t>(i)] * d_fx[static_cast<std::size_t>(i)];
        double second = 0.0, third = 0.0, fourth = 0.0;
        for (int i = 0; i < steps; ++i) {
            const double centered = coordinates[static_cast<std::size_t>(i)] - offset;
            const double square = centered * centered;
            second += square * d_fx[static_cast<std::size_t>(i)];
            third += square * centered * d_fx[static_cast<std::size_t>(i)];
            fourth += square * square * d_fx[static_cast<std::size_t>(i)];
        }
        return {reference + scale * offset, scale * std::sqrt(second),
                third / second / std::sqrt(second), fourth / second / second};
    }

    // Returns conditional central moments (up to 4th order) between [a, b] (C#
    // `ConditionalMoments(double a, double b)`, line 374; B4). Midpoint integration over
    // 300 stratified bins: m1 is the raw conditional mean; m2-m4 are central about the
    // UNCONDITIONAL Mean, each divided by the window probability. Returns the NaN
    // quadruple when a >= b or the window probability is <= 0 / NaN.
    virtual std::vector<double> conditional_moments(double a, double b) const {
        if (a >= b) return {kNaN, kNaN, kNaN, kNaN};

        int steps = 300;
        double l = cdf(a), u = cdf(b), total_prob = u - l;
        (void)l;  // mirrors the C# locals; only the difference is used
        if (total_prob <= 0.0 || std::isnan(total_prob))
            return {kNaN, kNaN, kNaN, kNaN};

        double mean_val = mean();
        auto bins = sampling::Stratify::XValues(sampling::StratificationOptions(a, b, steps));

        // Midpoint integration
        double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;

        for (int i = 0; i < steps; i++) {
            const auto& bin = bins[static_cast<std::size_t>(i)];
            double dF = cdf(bin.upper_bound()) - cdf(bin.lower_bound());
            double x_mid = bin.midpoint();

            double delta = x_mid - mean_val;
            double delta2 = delta * delta;
            double delta3 = delta2 * delta;
            double delta4 = delta2 * delta2;

            sum1 += x_mid * dF;
            sum2 += delta2 * dF;
            sum3 += delta3 * dF;
            sum4 += delta4 * dF;
        }

        double m1 = sum1 / total_prob;
        double m2 = sum2 / total_prob;
        double m3 = sum3 / total_prob;
        double m4 = sum4 / total_prob;

        return {m1, m2, m3, m4};
    }

    // Generates `sample_size` random values via inverse-CDF sampling. `seed > 0` seeds a
    // fresh MersenneTwister deterministically; otherwise a clock-seeded one is used (mirrors
    // C# `GenerateRandomValues(int sampleSize, int seed = -1)` -- see file header).
    virtual std::vector<double> generate_random_values(int sample_size, int seed = -1) const {
        sampling::MersenneTwister rnd = seed > 0 ? sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                                                  : sampling::MersenneTwister();
        std::vector<double> sample(static_cast<std::size_t>(sample_size));
        for (int i = 0; i < sample_size; ++i)
            sample[static_cast<std::size_t>(i)] = inverse_cdf(rnd.next_double());
        return sample;
    }

    virtual std::unique_ptr<UnivariateDistributionBase> clone() const = 0;

   protected:
    static constexpr double kNearZero = 1E-4;  // assessing if a parameter is near zero
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    static constexpr double kInf = std::numeric_limits<double>::infinity();

    bool parameters_valid_ = true;
};

}  // namespace corehydro::numerics::distributions
