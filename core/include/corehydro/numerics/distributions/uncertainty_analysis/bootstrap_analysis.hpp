// ported from: Numerics/Distributions/Univariate/Uncertainty Analysis/BootstrapAnalysis.cs @ 7e8e8d1
//
// The standalone frequentist parametric-bootstrap uncertainty engine. Given an IBootstrappable
// univariate distribution, a parameter-estimation method, a bootstrap sample size and a replication
// count, it resamples the distribution `Replications` times and derives quantile confidence bands by
// five methods (Percentile, Bias-Corrected, Normal cube-root, Bootstrap-t, BCa) plus a full
// UncertaintyAnalysisResults (mode/CI/mean curves) via Estimate().
//
// DEVIATIONS from the C# source, all deliberate:
//  * Threading REMOVED: independent element writes become serial loops. Floating-point reductions
//    retain the upstream fixed 64-chunk partition and serial merge order, so results do not depend
//    on thread count and match the C# accumulation order.
//  * The owned `Distribution` is stored as an owning std::unique_ptr<UnivariateDistributionBase>
//    (a clone of the ctor argument), with cached IBootstrappable*/IEstimation* views. The C#
//    holds the caller's object; cloning keeps ownership simple and the parameters BCa mutates
//    (SampleSize + an in-place Estimate on the sample data) local to this analysis.
//  * The `IUnivariateDistribution[]? distributions = null` optional arguments become a
//    `const std::vector<const UnivariateDistributionBase*>*` (nullptr => generate internally),
//    mirroring the C# "pass a shared bootstrapped set or let me build one" contract. Owning
//    generated sets are held in a local and viewed as non-owning pointers, matching how the C#
//    array of polymorphic distributions is passed by reference.
//  * The C# `XValues[idx] != double.NaN` guards are a C#-language no-op (NaN != anything is always
//    true); the effective test is `value <= population` (NaN compares false), transcribed as that.
//  * `Debug`/swallowed fit failures -> silent no-throw, exactly as the C# try/catch arms already do
//    (a failed replicate becomes a null distribution / NaN column).
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/interpolation/linear.hpp"
#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_bootstrappable.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::numerics {

class BootstrapAnalysis {
    using UnivariateDistributionBase = distributions::UnivariateDistributionBase;
    using IBootstrappable = distributions::IBootstrappable;
    using IEstimation = distributions::IEstimation;
    using ParameterEstimationMethod = distributions::ParameterEstimationMethod;
    using ParameterSet = math::optimization::ParameterSet;
    using DistPtr = std::unique_ptr<UnivariateDistributionBase>;
    using DistView = std::vector<const UnivariateDistributionBase*>;

    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    static constexpr std::size_t kReductionChunks = 64;

   public:
    // Construct a new Bootstrap Analysis. The distribution must be IBootstrappable (the C#
    // `distribution as IBootstrappable == null` guard). sampleSize >= 10, replications >= 100.
    BootstrapAnalysis(const UnivariateDistributionBase& distribution,
                      ParameterEstimationMethod estimationMethod, int sampleSize,
                      int replications = 10000, int seed = 12345) {
        if (dynamic_cast<const IBootstrappable*>(&distribution) == nullptr)
            throw std::invalid_argument("The distribution must implement IBootstrappable.");
        if (sampleSize < 10)
            throw std::out_of_range("The sample size must at least 10.");
        if (replications < 100)
            throw std::out_of_range("The number of bootstrap replications must be at least 100.");

        distribution_ = distribution.clone();
        bootstrappable_ = dynamic_cast<IBootstrappable*>(distribution_.get());
        estimation_ = dynamic_cast<IEstimation*>(distribution_.get());
        estimation_method_ = estimationMethod;
        sample_size_ = sampleSize;
        replications_ = replications;
        prng_seed_ = seed;
    }

    // --- Accessors (mirror the C# get properties) ---
    ParameterEstimationMethod estimation_method() const { return estimation_method_; }
    int sample_size() const { return sample_size_; }
    int replications() const { return replications_; }
    int prng_seed() const { return prng_seed_; }
    int failed_replications() const { return failed_replications_; }

    // Bootstrap a list of fitted distributions.
    std::vector<DistPtr> distributions() {
        failed_replications_ = 0;
        std::vector<DistPtr> boot(static_cast<std::size_t>(replications_));
        sampling::MersenneTwister r(static_cast<std::uint32_t>(prng_seed_));
        auto seeds = utilities::next_integers(r, replications_);
        for (int idx = 0; idx < replications_; ++idx) {
            DistPtr result;
            bool failed = false;
            for (int m = 0; m < retries_; ++m) {
                try {
                    result = bootstrappable_->bootstrap(estimation_method_, sample_size_,
                                                        seeds[static_cast<std::size_t>(idx)] + 10 * m);
                    failed = result == nullptr;
                } catch (...) {
                    failed = true;
                }
                if (!failed) break;
            }
            // MLE and certain L-moments methods can fail; on fail, leave null.
            if (failed) {
                result.reset();
                ++failed_replications_;
            }
            boot[static_cast<std::size_t>(idx)] = std::move(result);
        }
        if (failed_replications_ == replications_)
            throw std::runtime_error("Every bootstrap distribution fit failed.");
        return boot;
    }

    // Return a list of distributions given an array of parameter sets.
    std::vector<DistPtr> distributions(const std::vector<ParameterSet>& parameterSets) {
        failed_replications_ = 0;
        std::vector<DistPtr> boot(parameterSets.size());
        for (std::size_t idx = 0; idx < parameterSets.size(); ++idx) {
            try {
                auto dist = distribution_->clone();
                dist->set_parameters(parameterSets[idx].values);
                if (!dist->parameters_valid())
                    throw std::invalid_argument(
                        "The parameter set does not define a valid distribution");
                boot[idx] = std::move(dist);
            } catch (...) {
                boot[idx].reset();
                ++failed_replications_;
            }
        }
        if (!parameterSets.empty() &&
            failed_replications_ == static_cast<int>(parameterSets.size()))
            throw std::runtime_error("Every bootstrap distribution fit failed.");
        return boot;
    }

    // Bootstrap an array of distribution parameters [B][NumberOfParameters].
    std::vector<std::vector<double>> parameters(const DistView* distributions_in = nullptr) {
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        int np = distribution_->number_of_parameters();
        std::vector<std::vector<double>> out(view.size(),
                                             std::vector<double>(static_cast<std::size_t>(np)));
        for (std::size_t idx = 0; idx < view.size(); ++idx) {
            if (view[idx] != nullptr) {
                auto p = view[idx]->get_parameters();
                for (int i = 0; i < np; ++i) out[idx][static_cast<std::size_t>(i)] = p[static_cast<std::size_t>(i)];
            } else {
                for (int i = 0; i < np; ++i) out[idx][static_cast<std::size_t>(i)] = kNaN;
            }
        }
        return out;
    }

    // Bootstrap an array of distribution parameter sets.
    std::vector<ParameterSet> parameter_sets(const DistView* distributions_in = nullptr) {
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        int np = distribution_->number_of_parameters();
        std::vector<ParameterSet> out(view.size());
        for (std::size_t idx = 0; idx < view.size(); ++idx) {
            if (view[idx] != nullptr) {
                out[idx] = ParameterSet(view[idx]->get_parameters(), kNaN);
            } else {
                out[idx] = ParameterSet(std::vector<double>(static_cast<std::size_t>(np), kNaN), kNaN);
            }
        }
        return out;
    }

    // Bootstrap a list of product moments for each bootstrapped sample [B][4].
    std::vector<std::array<double, 4>> product_moments() {
        std::vector<std::array<double, 4>> out(static_cast<std::size_t>(replications_));
        sampling::MersenneTwister r(static_cast<std::uint32_t>(prng_seed_));
        auto seeds = utilities::next_integers(r, replications_);
        for (int idx = 0; idx < replications_; ++idx) {
            auto moments = data::product_moments(
                distribution_->generate_random_values(sample_size_, seeds[static_cast<std::size_t>(idx)]));
            for (int i = 0; i < 4; ++i)
                out[static_cast<std::size_t>(idx)][static_cast<std::size_t>(i)] = moments[static_cast<std::size_t>(i)];
        }
        return out;
    }

    // Bootstrap a list of linear moments for each bootstrapped sample [B][4].
    std::vector<std::array<double, 4>> linear_moments() {
        std::vector<std::array<double, 4>> out(static_cast<std::size_t>(replications_));
        sampling::MersenneTwister r(static_cast<std::uint32_t>(prng_seed_));
        auto seeds = utilities::next_integers(r, replications_);
        for (int idx = 0; idx < replications_; ++idx) {
            auto moments = data::linear_moments(
                distribution_->generate_random_values(sample_size_, seeds[static_cast<std::size_t>(idx)]));
            for (int i = 0; i < 4; ++i)
                out[static_cast<std::size_t>(idx)][static_cast<std::size_t>(i)] = moments[static_cast<std::size_t>(i)];
        }
        return out;
    }

    // Bootstrap a list of quantiles given input non-exceedance probabilities [B][p].
    std::vector<std::vector<double>> quantiles(const std::vector<double>& probabilities) {
        auto owned = distributions();
        DistView view = to_view(owned);
        return quantiles(probabilities, &view);
    }

    std::vector<std::vector<double>> quantiles(const std::vector<double>& probabilities,
                                               const DistView* distributions_in) {
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        std::vector<std::vector<double>> out(view.size(),
                                             std::vector<double>(probabilities.size()));
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            for (std::size_t idx = 0; idx < view.size(); ++idx)
                out[idx][i] = view[idx] != nullptr ? view[idx]->inverse_cdf(probabilities[i]) : kNaN;
        return out;
    }

    // Bootstrap a list of non-exceedance probabilities given input quantile values [B][q].
    std::vector<std::vector<double>> probabilities(const std::vector<double>& quantiles_in) {
        auto owned = distributions();
        DistView view = to_view(owned);
        return probabilities(quantiles_in, &view);
    }

    std::vector<std::vector<double>> probabilities(const std::vector<double>& quantiles_in,
                                                   const DistView* distributions_in) {
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        std::vector<std::vector<double>> out(view.size(),
                                             std::vector<double>(quantiles_in.size()));
        for (std::size_t i = 0; i < quantiles_in.size(); ++i)
            for (std::size_t idx = 0; idx < view.size(); ++idx)
                out[idx][i] = view[idx] != nullptr ? view[idx]->cdf(quantiles_in[i]) : kNaN;
        return out;
    }

    // Bootstrap full uncertainty analysis results using the percentile method.
    distributions::UncertaintyAnalysisResults estimate(const std::vector<double>& probabilities,
                                                       double alpha = 0.1,
                                                       const DistView* distributions_in = nullptr,
                                                       bool recordParameterSets = true) {
        distributions::UncertaintyAnalysisResults results;
        results.parent_distribution = distribution_.get();

        // Mode curve.
        results.mode_curve.assign(probabilities.size(), 0.0);
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            results.mode_curve[i] = distribution_->inverse_cdf(probabilities[i]);

        // Bootstrapped list of distributions (shared or generated).
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);

        // Parameter sets.
        if (recordParameterSets) results.parameter_sets = parameter_sets(&view);

        // Confidence intervals.
        results.confidence_intervals = percentile_quantile_ci(probabilities, alpha, &view);

        // Log-spaced quantile grid (C# governs the exact bin-count rule).
        auto minMax = compute_min_max_quantiles(0.001, 1.0 - 1e-9, view);
        std::vector<double> quantiles;
        double shift = minMax[0] <= 0.0 ? std::abs(minMax[0]) + 1.0 : 0.0;
        double min = minMax[0] + shift;
        double max = minMax[1] + shift;
        int order = static_cast<int>(std::floor(std::log10(max) - std::log10(min)));
        int bins = std::max(200, std::min(1000, 100 * order));
        const double log_min = std::log10(min);
        double delta = (std::log10(max) - log_min) / (bins - 1);
        quantiles.reserve(static_cast<std::size_t>(bins));
        for (int i = 0; i < bins; ++i)
            quantiles.push_back(std::pow(10.0, log_min + i * delta) - shift);

        // Mean curve.
        results.mean_curve = expected_probabilities(quantiles, probabilities, &view);

        return results;
    }

    // Bootstrap the expected non-exceedance probabilities, interpolated to the desired
    // probabilities (the mean/predictive curve builder).
    std::vector<double> expected_probabilities(const std::vector<double>& quantiles,
                                               const std::vector<double>& probabilities,
                                               const DistView* distributions_in = nullptr) {
        if (quantiles.size() < 2)
            throw std::invalid_argument("At least two quantiles are required");
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);

        std::vector<double> quants = quantiles;
        std::sort(quants.begin(), quants.end());
        std::vector<double> expected = mean_cdfs(quants, view);

        double minY = quants[0];
        double maxY = quants[0];
        std::vector<double> yVals{quants[0]};
        std::vector<double> xVals{expected[0]};
        for (std::size_t i = 1; i < quantiles.size(); ++i) {
            if (expected[i] > xVals.back()) {
                minY = std::min(minY, quantiles[i]);
                maxY = std::max(maxY, quantiles[i]);
                yVals.push_back(quantiles[i]);
                xVals.push_back(expected[i]);
            }
        }
        if (xVals.size() < 2)
            throw std::runtime_error(
                "The mean bootstrap CDF does not contain two distinct probabilities");
        bool useLogTransform = minY > 0.0 && (std::log10(maxY) - std::log10(minY)) > 1.0;

        data::Linear linint(xVals, yVals);
        linint.x_transform = data::Transform::NormalZ;
        linint.y_transform = useLogTransform ? data::Transform::Logarithmic : data::Transform::None;
        return linint.interpolate(probabilities);
    }

    // Bootstrap the expected non-exceedance probabilities given the input quantile values (overload
    // without interpolation to probabilities).
    std::vector<double> expected_probabilities(const std::vector<double>& quantiles,
                                               const DistView* distributions_in) {
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        std::vector<double> quants = quantiles;
        std::sort(quants.begin(), quants.end());
        return mean_cdfs(quants, view);
    }

    // Returns the min and max quantiles from a bootstrap analysis {min, max}.
    std::array<double, 2> compute_min_max_quantiles(double minProbability, double maxProbability,
                                                    const DistView& distributions_in) {
        std::array<double, 2> output = {std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::lowest()};
        for (std::size_t j = 0; j < distributions_in.size(); ++j) {
            if (distributions_in[j] != nullptr) {
                double minX = distributions_in[j]->inverse_cdf(minProbability);
                double maxX = distributions_in[j]->inverse_cdf(maxProbability);
                if (minX < output[0]) output[0] = minX;
                if (maxX > output[1]) output[1] = maxX;
            }
        }
        if (output[0] == std::numeric_limits<double>::max())
            throw std::runtime_error("Every bootstrap distribution fit failed.");
        return output;
    }

    // --- Confidence-interval methods ([p][2] = {lower, upper}) ---

    // Percentile method.
    std::vector<std::array<double, 2>> percentile_quantile_ci(
        const std::vector<double>& probabilities, double alpha = 0.1,
        const DistView* distributions_in = nullptr) {
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        std::array<double, 2> CIs = {alpha / 2.0, 1.0 - alpha / 2.0};
        std::vector<std::array<double, 2>> output(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            std::vector<double> validValues =
                valid_quantiles(view, probabilities[i], "percentile confidence intervals");
            std::sort(validValues.begin(), validValues.end());
            for (int j = 0; j < 2; ++j)
                output[i][static_cast<std::size_t>(j)] =
                    data::percentile(validValues, CIs[static_cast<std::size_t>(j)], true);
        }
        return output;
    }

    // Bias-corrected percentile method.
    std::vector<std::array<double, 2>> bias_corrected_quantile_ci(
        const std::vector<double>& probabilities, double alpha = 0.1,
        const DistView* distributions_in = nullptr) {
        std::vector<double> populationXValues(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            populationXValues[i] = distribution_->inverse_cdf(probabilities[i]);

        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        std::array<double, 2> CIs = {alpha / 2.0, 1.0 - alpha / 2.0};
        std::vector<std::array<double, 2>> output(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            int count_leq = 0;
            std::vector<double> XValues(view.size());
            for (std::size_t idx = 0; idx < view.size(); ++idx) {
                XValues[idx] = view[idx] != nullptr ? view[idx]->inverse_cdf(probabilities[i]) : kNaN;
                // C# `XValues[idx] != double.NaN` is a no-op; effective test is value <= population.
                if (std::isfinite(XValues[idx]) && XValues[idx] <= populationXValues[i])
                    ++count_leq;
            }
            std::vector<double> validValues = finite_values_or_throw(
                XValues, "bias-corrected confidence intervals");
            const double P0 = static_cast<double>(count_leq) /
                              (static_cast<double>(validValues.size()) + 1.0);
            std::sort(validValues.begin(), validValues.end());
            for (int j = 0; j < 2; ++j) {
                double Z0 = distributions::Normal::standard_z(P0);
                double Z = distributions::Normal::standard_z(CIs[static_cast<std::size_t>(j)]);
                double BC = distributions::Normal::standard_cdf(2.0 * Z0 + Z);
                output[i][static_cast<std::size_t>(j)] = data::percentile(validValues, BC, true);
            }
        }
        return output;
    }

    // Normal (standard) method with a cube-root transform.
    std::vector<std::array<double, 2>> normal_quantile_ci(
        const std::vector<double>& probabilities, double alpha = 0.1,
        const DistView* distributions_in = nullptr) {
        std::vector<double> populationXValues(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            populationXValues[i] = cube_root(distribution_->inverse_cdf(probabilities[i]));

        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        std::array<double, 2> CIs = {alpha / 2.0, 1.0 - alpha / 2.0};
        std::vector<std::array<double, 2>> output(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            std::vector<double> XValues(view.size());
            for (std::size_t idx = 0; idx < view.size(); ++idx)
                XValues[idx] = view[idx] != nullptr
                                   ? cube_root(view[idx]->inverse_cdf(probabilities[i]))
                                   : kNaN;
            std::vector<double> validValues = finite_values_or_throw(
                XValues, "normal confidence intervals", 2);
            double SE = data::standard_deviation(validValues);
            for (int j = 0; j < 2; ++j) {
                double Z = distributions::Normal::standard_z(CIs[static_cast<std::size_t>(j)]);
                output[i][static_cast<std::size_t>(j)] = std::pow(populationXValues[i] + SE * Z, 3.0);
            }
        }
        return output;
    }

    // Bias-corrected and accelerated (BCa) percentile method (jackknife acceleration constants).
    std::vector<std::array<double, 2>> bca_quantile_ci(const std::vector<double>& sampleData,
                                                       const std::vector<double>& probabilities,
                                                       double alpha = 0.1) {
        std::array<double, 2> CIs = {alpha / 2.0, 1.0 - alpha / 2.0};
        std::vector<std::array<double, 2>> output(probabilities.size());

        // Estimate distribution (mutates SampleSize and the stored distribution's parameters).
        sample_size_ = static_cast<int>(sampleData.size());
        estimation_->estimate(sampleData, estimation_method_);

        std::vector<double> populationXValues(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            populationXValues[i] = distribution_->inverse_cdf(probabilities[i]);

        auto a = acceleration_constants(sampleData, probabilities, populationXValues);

        auto owned = distributions();
        DistView view = to_view(owned);
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            int count_leq = 0;
            std::vector<double> XValues(static_cast<std::size_t>(replications_));
            for (int idx = 0; idx < replications_; ++idx) {
                XValues[static_cast<std::size_t>(idx)] =
                    view[static_cast<std::size_t>(idx)] != nullptr
                        ? view[static_cast<std::size_t>(idx)]->inverse_cdf(probabilities[i])
                        : kNaN;
                if (std::isfinite(XValues[static_cast<std::size_t>(idx)]) &&
                    XValues[static_cast<std::size_t>(idx)] <= populationXValues[i])
                    ++count_leq;
            }
            std::vector<double> validValues =
                finite_values_or_throw(XValues, "BCa confidence intervals");
            const double P0 = static_cast<double>(count_leq) /
                              (static_cast<double>(validValues.size()) + 1.0);
            std::sort(validValues.begin(), validValues.end());
            for (int j = 0; j < 2; ++j) {
                double Z0 = distributions::Normal::standard_z(P0);
                double Z = distributions::Normal::standard_z(CIs[static_cast<std::size_t>(j)]);
                double num = Z0 + Z;
                double den = 1.0 - a[i] * (Z0 + Z);
                double BC = distributions::Normal::standard_cdf(Z0 + num / den);
                output[i][static_cast<std::size_t>(j)] = data::percentile(validValues, BC, true);
            }
        }
        return output;
    }

    // Bootstrap-t (Student-t) method with cube-root transform + inner bootstrap standard error.
    std::vector<std::array<double, 2>> bootstrap_t_quantile_ci(
        const std::vector<double>& probabilities, double alpha = 0.1) {
        std::size_t p = probabilities.size();
        std::vector<double> populationXValues(p);
        for (std::size_t i = 0; i < p; ++i)
            populationXValues[i] = cube_root(distribution_->inverse_cdf(probabilities[i]));

        std::vector<std::vector<double>> xValues(static_cast<std::size_t>(replications_),
                                                 std::vector<double>(p));
        std::vector<std::vector<double>> studentT(static_cast<std::size_t>(replications_),
                                                  std::vector<double>(p));
        std::array<double, 2> CIs = {alpha / 2.0, 1.0 - alpha / 2.0};
        std::vector<std::array<double, 2>> output(p);

        sampling::MersenneTwister r(static_cast<std::uint32_t>(prng_seed_));
        auto seeds = utilities::next_integers(r, replications_);
        int failed_fits = 0;
        for (int i = 0; i < replications_; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            try {
                auto newDistribution = distribution_->clone();
                auto sample =
                    newDistribution->generate_random_values(sample_size_, seeds[ui]);
                auto* est = dynamic_cast<IEstimation*>(newDistribution.get());
                est->estimate(sample, estimation_method_);

                std::vector<double> bootXValues(p);
                for (std::size_t j = 0; j < p; ++j)
                    bootXValues[j] = cube_root(newDistribution->inverse_cdf(probabilities[j]));

                auto bootSE = bootstrap_standard_error(*newDistribution, probabilities, 300, seeds[ui]);
                for (std::size_t j = 0; j < p; ++j) {
                    xValues[ui][j] = bootXValues[j];
                    studentT[ui][j] = (populationXValues[j] - bootXValues[j]) / bootSE[j];
                }
            } catch (...) {
                ++failed_fits;
                for (std::size_t j = 0; j < p; ++j) {
                    xValues[ui][j] = kNaN;
                    studentT[ui][j] = kNaN;
                }
            }
        }
        if (failed_fits == replications_)
            throw std::runtime_error("Every studentized bootstrap fit failed.");

        for (std::size_t i = 0; i < p; ++i) {
            std::vector<double> XValues;
            std::vector<double> TValues;
            for (int k = 0; k < replications_; ++k) {
                std::size_t uk = static_cast<std::size_t>(k);
                if (std::isfinite(xValues[uk][i]) && std::isfinite(studentT[uk][i])) {
                    XValues.push_back(xValues[uk][i]);
                    TValues.push_back(studentT[uk][i]);
                }
            }
            if (XValues.size() < 2)
                throw std::runtime_error(
                    "Insufficient finite fits for studentized confidence intervals");
            double SE = data::standard_deviation(XValues);
            std::sort(TValues.begin(), TValues.end());
            for (int j = 0; j < 2; ++j) {
                double T = data::percentile(TValues, CIs[static_cast<std::size_t>(j)], true);
                output[i][static_cast<std::size_t>(j)] = std::pow(populationXValues[i] + SE * T, 3.0);
            }
        }
        return output;
    }

   private:
    // Estimates the acceleration constants for each probability (jackknife).
    std::vector<double> acceleration_constants(const std::vector<double>& sampleData,
                                               const std::vector<double>& probabilities,
                                               const std::vector<double>& thetaHats) {
        std::size_t N = sampleData.size();
        std::size_t p = probabilities.size();
        std::vector<double> a(p, 0.0);
        if (N == 0) return a;
        const std::size_t chunks = std::min(kReductionChunks, N);
        std::vector<std::vector<double>> chunk_i2(chunks, std::vector<double>(p, 0.0));
        std::vector<std::vector<double>> chunk_i3(chunks, std::vector<double>(p, 0.0));
        std::vector<std::size_t> chunk_valid(chunks, 0);

        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            const std::size_t start = chunk * N / chunks;
            const std::size_t end = (chunk + 1) * N / chunks;
            for (std::size_t idx = start; idx < end; ++idx) {
                std::vector<double> jackSample;
                jackSample.reserve(N - 1);
                for (std::size_t k = 0; k < N; ++k)
                    if (k != idx) jackSample.push_back(sampleData[k]);

                auto newDistribution = distribution_->clone();
                try {
                    auto* est = dynamic_cast<IEstimation*>(newDistribution.get());
                    est->estimate(jackSample, estimation_method_);
                    for (std::size_t i = 0; i < p; ++i) {
                        double thetaJack = newDistribution->inverse_cdf(probabilities[i]);
                        const double difference = thetaHats[i] - thetaJack;
                        chunk_i2[chunk][i] += difference * difference;
                        chunk_i3[chunk][i] += difference * difference * difference;
                    }
                    ++chunk_valid[chunk];
                } catch (...) {
                    // MLE and certain L-moments methods can fail to find a solution.
                }
            }
        }
        std::size_t valid_count = 0;
        for (std::size_t count : chunk_valid) valid_count += count;
        if (valid_count == 0)
            throw std::runtime_error("Every jackknife acceleration fit failed.");
        for (std::size_t i = 0; i < p; ++i) {
            double second = 0.0;
            double third = 0.0;
            for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
                second += chunk_i2[chunk][i];
                third += chunk_i3[chunk][i];
            }
            a[i] = second > 0.0 && std::isfinite(second)
                       ? third / (std::pow(second, 1.5) * 6.0)
                       : 0.0;
        }
        return a;
    }

    // Estimates the standard error for each probability using the parametric bootstrap (300 reps).
    std::vector<double> bootstrap_standard_error(const UnivariateDistributionBase& parentDist,
                                                 const std::vector<double>& probabilities,
                                                 int replications = 300, int seed = 12345) {
        std::size_t p = probabilities.size();
        sampling::MersenneTwister r(static_cast<std::uint32_t>(seed));
        auto seeds = utilities::next_integers(r, replications);
        std::vector<std::vector<double>> xValues(static_cast<std::size_t>(replications),
                                                 std::vector<double>(p, kNaN));
        std::vector<double> se(p);
        int failed_fits = 0;
        for (int i = 0; i < replications; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            try {
                auto bootDist = parentDist.clone();
                auto sample = bootDist->generate_random_values(sample_size_, seeds[ui]);
                auto* est = dynamic_cast<IEstimation*>(bootDist.get());
                est->estimate(sample, estimation_method_);
                for (std::size_t j = 0; j < p; ++j)
                    xValues[ui][j] = cube_root(bootDist->inverse_cdf(probabilities[j]));
            } catch (...) {
                // On fail, leave the NaN-initialized column.
                ++failed_fits;
            }
        }
        if (failed_fits == replications)
            throw std::runtime_error("Every inner bootstrap fit failed.");
        for (std::size_t i = 0; i < p; ++i) {
            std::vector<double> col(static_cast<std::size_t>(replications));
            for (int k = 0; k < replications; ++k) col[static_cast<std::size_t>(k)] = xValues[static_cast<std::size_t>(k)][i];
            const auto successful =
                finite_values_or_throw(col, "inner bootstrap standard errors", 2);
            se[i] = data::standard_deviation(successful);
        }
        return se;
    }

    // Build a non-owning view of an owning distribution vector.
    static DistView to_view(const std::vector<DistPtr>& owned) {
        DistView view;
        view.reserve(owned.size());
        for (const auto& d : owned) view.push_back(d.get());
        return view;
    }

    // Resolve the optional distributions argument: use the caller's view when provided, else
    // generate an owning set (stored in `owned` to keep it alive) and view that.
    DistView resolve(const DistView* distributions_in, std::vector<DistPtr>& owned) {
        if (distributions_in != nullptr) return *distributions_in;
        owned = distributions();
        return to_view(owned);
    }

    // Collect the inverse-CDF quantiles at `probability` across the view, dropping null dists.
    static std::vector<double> valid_quantiles(const DistView& view, double probability,
                                               const char* operation) {
        std::vector<double> x(view.size());
        for (std::size_t idx = 0; idx < view.size(); ++idx)
            x[idx] = view[idx] != nullptr ? view[idx]->inverse_cdf(probability) : kNaN;
        return finite_values_or_throw(x, operation);
    }

    static std::vector<double> finite_values_or_throw(const std::vector<double>& values,
                                                      const char* operation,
                                                      std::size_t minimum_count = 1) {
        std::vector<double> out;
        out.reserve(values.size());
        for (double v : values)
            if (std::isfinite(v)) out.push_back(v);
        if (out.size() < minimum_count)
            throw std::runtime_error(std::string("Insufficient finite fits for ") + operation);
        return out;
    }

    static std::size_t successful_distribution_count(const DistView& view) {
        std::size_t count = 0;
        for (const auto* distribution : view)
            if (distribution != nullptr) ++count;
        if (count == 0) throw std::runtime_error("Every bootstrap distribution fit failed.");
        return count;
    }

    static std::vector<double> mean_cdfs(const std::vector<double>& quantiles,
                                         const DistView& view) {
        if (quantiles.empty()) return {};
        if (view.empty()) throw std::runtime_error("No bootstrap distributions were supplied");
        const std::size_t chunks = std::min(kReductionChunks, view.size());
        std::vector<std::vector<double>> chunk_sums(
            chunks, std::vector<double>(quantiles.size(), 0.0));
        std::vector<std::size_t> chunk_valid(chunks, 0);
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            const std::size_t start = chunk * view.size() / chunks;
            const std::size_t end = (chunk + 1) * view.size() / chunks;
            for (std::size_t index = start; index < end; ++index) {
                if (view[index] == nullptr) continue;
                ++chunk_valid[chunk];
                for (std::size_t i = 0; i < quantiles.size(); ++i)
                    chunk_sums[chunk][i] += view[index]->cdf(quantiles[i]);
            }
        }
        std::size_t valid_count = 0;
        for (std::size_t count : chunk_valid) valid_count += count;
        if (valid_count == 0)
            throw std::runtime_error("Every bootstrap distribution fit failed");
        std::vector<double> expected(quantiles.size(), 0.0);
        for (std::size_t i = 0; i < quantiles.size(); ++i) {
            for (std::size_t chunk = 0; chunk < chunks; ++chunk)
                expected[i] += chunk_sums[chunk][i];
            expected[i] /= static_cast<double>(valid_count);
        }
        return expected;
    }

    static double cube_root(double value) {
        if (value == 0.0) return value;
        return value < 0.0 ? -std::pow(-value, 1.0 / 3.0) : std::pow(value, 1.0 / 3.0);
    }

    DistPtr distribution_;
    IBootstrappable* bootstrappable_ = nullptr;
    IEstimation* estimation_ = nullptr;
    ParameterEstimationMethod estimation_method_ = ParameterEstimationMethod::MethodOfMoments;
    int sample_size_ = 0;
    int replications_ = 0;
    int prng_seed_ = 0;
    int retries_ = 20;
    int failed_replications_ = 0;
};

}  // namespace corehydro::numerics
