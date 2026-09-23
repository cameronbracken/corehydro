// ported from: Numerics/Distributions/Univariate/Uncertainty Analysis/UncertaintyAnalysisResults.cs @ 7e8e8d1
//
// Result container every univariate uncertainty analysis assembles: the mode/point-estimate
// curve, the mean (predictive) curve, the confidence-interval band, the recorded parameter sets,
// and the fit scalars (AIC/BIC/DIC/RMSE/ERL). A HYBRID -- both a plain result DTO (empty ctor) and
// a "compute ctor" that derives its own curves from a parent distribution plus a fixed array of
// sampled distributions. The four Process* methods are transcribed formula-for-formula and
// op-order-for-op-order from the C# so downstream analyses (A5 UnivariateAnalysis,
// A7 Bulletin17CAnalysis) reproduce identical numbers.
//
// DEVIATIONS from the C# source, all deliberate:
//  * Persistence DROPPED (~270 lines): ToByteArray / FromByteArray / FromByteArrayLegacy
//    (System.Text.Json + BinaryFormatter) and ToXElement / FromXElement (XElement) with their JSON
//    converter wiring. Out of scope for this numerical port -- GUI/on-disk serialization with no
//    numerical content; nothing downstream in the ported layers consumes them.
//  * Threading REMOVED: every C# Parallel.For / Tools.ParallelAdd reduction (the CI quantile scan,
//    the min/max scan and CDF sum in the mean curve, the parameter-set capture) becomes a plain
//    serial loop. The reductions are order-independent sums / independent writes, so the serial
//    form is numerically identical to the parallel one.
//  * The C# `using Numerics.Sampling.MCMC` is unused; no dependency on MCMCResults is introduced.
//  * Parent is taken by const reference (never null) instead of a nullable pointer arg, so the C#
//    ArgumentNullException(parentDistribution) guard is structurally unreachable and omitted; the
//    empty-sampled / empty-probabilities / alpha-range guards are preserved as std::invalid_argument
//    / std::out_of_range throws. The C#-null-dist NaN filtering in ProcessConfidenceIntervals is
//    preserved (a caller may still pass a null pointer in the array).
//  * Sampled distributions are passed as a non-owning std::vector<const UnivariateDistributionBase*>,
//    mirroring the C# UnivariateDistributionBase[] of polymorphic dists the array does not own.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/linear.hpp"
#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"

namespace corehydro::numerics::distributions {

class UncertaintyAnalysisResults {
   public:
    // Construct an empty instance (plain result DTO).
    UncertaintyAnalysisResults() = default;

    // Constructs a new instance with computed uncertainty metrics.
    //   parentDistribution  -- the parent (mode/point estimate) distribution.
    //   sampledDistributions -- array of sampled distributions from posterior or bootstrap
    //                           (non-owning; entries may be null, mirroring the C# array).
    //   probabilities       -- array of NON-exceedance probabilities for quantile estimation.
    //   alpha               -- confidence level (default 0.1 for the 90% CI).
    //   minProbability      -- minimum probability for mean-curve range (default 0.001).
    //   maxProbability      -- maximum probability for mean-curve range (default 1 - 1e-9).
    //   recordParameterSets -- if true, stores all parameter sets from the sampled distributions.
    UncertaintyAnalysisResults(
        const UnivariateDistributionBase& parentDistribution,
        const std::vector<const UnivariateDistributionBase*>& sampledDistributions,
        const std::vector<double>& probabilities, double alpha = 0.1,
        double minProbability = 0.001, double maxProbability = 1.0 - 1e-9,
        bool recordParameterSets = false) {
        validate_sampled_distributions(sampledDistributions);
        if (probabilities.empty())
            throw std::invalid_argument("Probabilities cannot be null or empty.");

        process_mode_curve(parentDistribution, probabilities);
        process_confidence_intervals(sampledDistributions, probabilities, alpha);
        process_mean_curve(sampledDistributions, probabilities, minProbability, maxProbability);

        if (recordParameterSets) process_parameter_sets(sampledDistributions);

        // Set default values
        aic = std::numeric_limits<double>::quiet_NaN();
        bic = std::numeric_limits<double>::quiet_NaN();
        dic = std::numeric_limits<double>::quiet_NaN();
        rmse = std::numeric_limits<double>::quiet_NaN();
        erl = std::numeric_limits<double>::quiet_NaN();
    }

    // --- Fields (mirror the C# get/set properties; relaxed-mutation OK for this result object) ---

    // The parent probability distribution (non-owning; nullptr when unset).
    const UnivariateDistributionBase* parent_distribution = nullptr;

    // The array of parameter sets.
    std::vector<math::optimization::ParameterSet> parameter_sets;

    // The confidence intervals, [p][2] (lower, upper) per probability.
    std::vector<std::array<double, 2>> confidence_intervals;

    // The mode (or computed) curve from the parent distribution.
    std::vector<double> mode_curve;

    // The mean (or predictive) curve.
    std::vector<double> mean_curve;

    // Akaike information criteria (AIC) of the fit.
    double aic = std::numeric_limits<double>::quiet_NaN();
    // Bayesian information criteria (BIC) of the fit.
    double bic = std::numeric_limits<double>::quiet_NaN();
    // Deviance Information Criterion (DIC) of the fit.
    double dic = std::numeric_limits<double>::quiet_NaN();
    // Root Mean Square Error (RMSE) of the fit.
    double rmse = std::numeric_limits<double>::quiet_NaN();
    // Effective Record Length (ERL).
    double erl = std::numeric_limits<double>::quiet_NaN();

    // --- Process* methods (public, mirroring the C# layout) ---

    // Process and set the parent distribution and computed curve (mode / plug-in / point estimate).
    void process_mode_curve(const UnivariateDistributionBase& parentDistribution,
                            const std::vector<double>& probabilities) {
        if (probabilities.empty())
            throw std::invalid_argument("Probabilities cannot be null or empty.");

        parent_distribution = &parentDistribution;
        // C# ParentDistribution.InverseCDF(probabilities) returns an array; there is no vector
        // overload on the base, so loop the scalar inverse_cdf.
        mode_curve.assign(probabilities.size(), 0.0);
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            mode_curve[i] = parent_distribution->inverse_cdf(probabilities[i]);
    }

    // Process and set the confidence intervals from a list of sampled distributions. alpha default
    // 0.1 -> 90% confidence intervals.
    void process_confidence_intervals(
        const std::vector<const UnivariateDistributionBase*>& sampledDistributions,
        const std::vector<double>& probabilities, double alpha = 0.1) {
        validate_sampled_distributions(sampledDistributions);
        if (probabilities.empty())
            throw std::invalid_argument("Probabilities cannot be null or empty.");
        if (alpha <= 0.0 || alpha >= 1.0)
            throw std::out_of_range("Alpha must be between 0 and 1.");

        std::size_t B = sampledDistributions.size();
        std::size_t p = probabilities.size();
        double lowerCI = alpha / 2.0;
        double upperCI = 1.0 - alpha / 2.0;
        confidence_intervals.assign(p, {0.0, 0.0});

        // Loop over probabilities and record percentiles (C# Parallel.For -> serial loop).
        for (std::size_t i = 0; i < p; ++i) {
            std::vector<double> x_values(B);
            for (std::size_t idx = 0; idx < B; ++idx) {
                x_values[idx] = sampledDistributions[idx] != nullptr
                                    ? sampledDistributions[idx]->inverse_cdf(probabilities[i])
                                    : std::numeric_limits<double>::quiet_NaN();
            }

            // Filter non-finite values from failed or invalid fits and sort.
            std::vector<double> valid_values;
            valid_values.reserve(B);
            for (std::size_t j = 0; j < B; ++j)
                if (std::isfinite(x_values[j])) valid_values.push_back(x_values[j]);
            if (valid_values.empty())
                throw std::runtime_error(
                    "No finite sampled quantiles are available for the requested probability.");
            std::sort(valid_values.begin(), valid_values.end());

            confidence_intervals[i][0] = data::percentile(valid_values, lowerCI, true);
            confidence_intervals[i][1] = data::percentile(valid_values, upperCI, true);
        }
    }

    // Computes the mean (predictive) curve by averaging CDFs across all sampled distributions,
    // using a log-spaced quantile grid for efficient coverage across wide ranges.
    void process_mean_curve(
        const std::vector<const UnivariateDistributionBase*>& sampledDistributions,
        const std::vector<double>& probabilities, double minProbability = 0.001,
        double maxProbability = 1.0 - 1e-9) {
        validate_sampled_distributions(sampledDistributions);
        if (probabilities.empty())
            throw std::invalid_argument("Probabilities cannot be null or empty.");

        std::size_t B = sampledDistributions.size();
        std::size_t valid_distributions = 0;

        // Compute min and max X values across all distributions (C# Parallel.For -> serial).
        double minX = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        for (std::size_t j = 0; j < B; ++j) {
            if (sampledDistributions[j] != nullptr) {
                ++valid_distributions;
                double innerMin = sampledDistributions[j]->inverse_cdf(minProbability);
                double innerMax = sampledDistributions[j]->inverse_cdf(maxProbability);
                if (innerMin < minX) minX = innerMin;
                if (innerMax > maxX) maxX = innerMax;
            }
        }

        // Create log-spaced quantiles for efficient coverage.
        double shift = minX <= 0.0 ? std::abs(minX) + 1.0 : 0.0;
        double min = minX + shift;
        double max = maxX + shift;
        int order = static_cast<int>(std::floor(std::log10(max) - std::log10(min)));
        int bins = std::max(200, std::min(1000, 100 * order));

        std::vector<double> quantiles(static_cast<std::size_t>(bins));
        double delta = (std::log10(max) - std::log10(min)) / (bins - 1);
        for (int i = 0; i < bins; ++i) {
            double logX = std::log10(min) + i * delta;
            quantiles[static_cast<std::size_t>(i)] = std::pow(10.0, logX) - shift;
        }

        // Compute expected probability for each quantile (C# Parallel.For reduction -> serial sum).
        std::vector<double> expected(static_cast<std::size_t>(bins));
        for (int i = 0; i < bins; ++i) {
            double total = 0.0;
            for (std::size_t j = 0; j < B; ++j) {
                if (sampledDistributions[j] != nullptr)
                    total += sampledDistributions[j]->cdf(quantiles[static_cast<std::size_t>(i)]);
            }
            expected[static_cast<std::size_t>(i)] =
                total / static_cast<double>(valid_distributions);
        }

        // Build monotonic interpolation points.
        std::vector<double> y_vals{quantiles[0]};
        std::vector<double> x_vals{expected[0]};
        double minY = std::numeric_limits<double>::max();
        double maxY = std::numeric_limits<double>::lowest();

        for (int i = 1; i < bins; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            if (expected[ui] > x_vals.back()) {
                minY = std::min(minY, quantiles[ui]);
                maxY = std::max(maxY, quantiles[ui]);
                y_vals.push_back(quantiles[ui]);
                x_vals.push_back(expected[ui]);
            }
        }

        // Determine if a log transform on y is appropriate.
        bool useLogTransform = minY > 0.0 && (std::log10(maxY) - std::log10(minY)) > 1.0;

        // Interpolate mean curve at the requested probabilities.
        data::Linear linint(x_vals, y_vals);
        linint.x_transform = data::Transform::NormalZ;
        linint.y_transform = useLogTransform ? data::Transform::Logarithmic : data::Transform::None;
        mean_curve = linint.interpolate(probabilities);
    }

    // Processes and stores the parameter sets from all sampled distributions.
    void process_parameter_sets(
        const std::vector<const UnivariateDistributionBase*>& sampledDistributions) {
        if (sampledDistributions.empty())
            throw std::invalid_argument("Sampled distributions cannot be null or empty.");

        std::size_t B = sampledDistributions.size();
        std::size_t number_of_parameters =
            parent_distribution != nullptr
                ? static_cast<std::size_t>(parent_distribution->number_of_parameters())
                : 0;
        if (number_of_parameters == 0) {
            for (const auto* distribution : sampledDistributions) {
                if (distribution != nullptr) {
                    number_of_parameters =
                        static_cast<std::size_t>(distribution->number_of_parameters());
                    break;
                }
            }
        }
        if (number_of_parameters == 0)
            throw std::invalid_argument(
                "Every sampled distribution is null; the parameter count is unknown.");
        parameter_sets.assign(B, math::optimization::ParameterSet());
        for (std::size_t idx = 0; idx < B; ++idx) {
            if (sampledDistributions[idx] != nullptr) {
                parameter_sets[idx] = math::optimization::ParameterSet(
                    sampledDistributions[idx]->get_parameters(),
                    std::numeric_limits<double>::quiet_NaN());
            } else {
                parameter_sets[idx] = math::optimization::ParameterSet(
                    std::vector<double>(number_of_parameters,
                                        std::numeric_limits<double>::quiet_NaN()),
                    std::numeric_limits<double>::quiet_NaN());
            }
        }
    }

   private:
    static void validate_sampled_distributions(
        const std::vector<const UnivariateDistributionBase*>& sampledDistributions) {
        if (sampledDistributions.empty())
            throw std::invalid_argument("Sampled distributions cannot be null or empty.");
        for (const auto* distribution : sampledDistributions)
            if (distribution != nullptr) return;
        throw std::runtime_error("At least one sampled distribution must be non-null.");
    }
};

}  // namespace corehydro::numerics::distributions
