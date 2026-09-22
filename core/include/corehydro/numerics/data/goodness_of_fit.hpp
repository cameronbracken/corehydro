// ported from: Numerics/Data/Statistics/GoodnessOfFit.cs @ 7e8e8d1
//
// Goodness-of-fit measures for evaluating model performance.
// Includes information criteria (AIC, BIC), error metrics (RMSE, MAE, MSE),
// efficiency coefficients (Nash-Sutcliffe, Kling-Gupta), bias metrics (PBIAS),
// and statistical tests (Kolmogorov-Smirnov, Anderson-Darling, Chi-Squared).
//
// The three statistical tests (KolmogorovSmirnov, AndersonDarling, ChiSquared)
// depend on UnivariateDistributionBase and a minimal Histogram helper that
// mirrors Histogram.cs (Rice Rule binning). All other functions are pure-numeric
// and have no distribution dependency.
//
// NOT exposed to R/Python yet -- C++ and oracle validation only (YAGNI).
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"

namespace corehydro::numerics::data {

// ---------------------------------------------------------------------------
// Internal: minimal Histogram used only by ChiSquared.
// Mirrors Numerics/Data/Statistics/Histogram.cs (Rice Rule constructor).
// ---------------------------------------------------------------------------
namespace detail {

struct HistBin {
    double lower_bound;
    double upper_bound;
    int frequency = 0;
};

struct Histogram {
    double lower_bound;
    double upper_bound;
    double bin_width;
    int number_of_bins;
    std::vector<HistBin> bins;

    // Rice Rule constructor — mirrors Histogram(IList<double> data)
    explicit Histogram(const std::vector<double>& data) {
        int n = static_cast<int>(data.size());
        // Rice Rule: ceil(2 * n^(1/3)) + 1
        number_of_bins = static_cast<int>(std::ceil(2.0 * std::pow(n, 1.0 / 3.0)) + 1.0);
        lower_bound = *std::min_element(data.begin(), data.end());
        upper_bound = *std::max_element(data.begin(), data.end());
        if (upper_bound == lower_bound) upper_bound = lower_bound + 1.0;
        bin_width = (upper_bound - lower_bound) / number_of_bins;

        // Create bins
        bins.resize(static_cast<std::size_t>(number_of_bins));
        double xl = lower_bound;
        for (int i = 0; i < number_of_bins; ++i) {
            double xu = xl + bin_width;
            bins[static_cast<std::size_t>(i)] = {xl, xu, 0};
            xl = xu;
        }
        bins.back().upper_bound = upper_bound;  // guarantee exact upper bound

        // Assign data to bins
        for (double v : data) {
            if (v <= lower_bound) {
                bins.front().lower_bound = v;
                bins.front().frequency++;
            } else if (v >= upper_bound) {
                bins.back().upper_bound = v;
                bins.back().frequency++;
            } else {
                // Binary search for the bin
                int lo = 0, hi = number_of_bins - 1;
                while (lo < hi) {
                    int mid = (lo + hi) / 2;
                    if (bins[static_cast<std::size_t>(mid)].upper_bound < v)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                bins[static_cast<std::size_t>(lo)].frequency++;
            }
        }
    }
};

}  // namespace detail

// ---------------------------------------------------------------------------
// GoodnessOfFit: mirrors GoodnessOfFit.cs, method for method.
// ---------------------------------------------------------------------------

class GoodnessOfFit {
   public:
    // -----------------------------------------------------------------------
    // Information Criteria
    // -----------------------------------------------------------------------

    /// AIC = -2*logLikelihood + 2*numberOfParameters
    static double aic(int number_of_parameters, double log_likelihood) {
        return (-2.0 * log_likelihood) + (2.0 * number_of_parameters);
    }

    /// AICc: small-sample correction
    static double aicc(int sample_size, int number_of_parameters, double log_likelihood) {
        double a = aic(number_of_parameters, log_likelihood);
        double k = number_of_parameters;
        return a + (2.0 * k * k + 2.0 * k) / (sample_size - k - 1.0);
    }

    /// BIC = -2*logLikelihood + numberOfParameters*log(sampleSize)
    static double bic(int sample_size, int number_of_parameters, double log_likelihood) {
        return (-2.0 * log_likelihood) + (number_of_parameters * std::log(sample_size));
    }

    /// Akaike weights: normalised exp(-0.5*(AIC - min(AIC)))
    static std::vector<double> aic_weights(const std::vector<double>& aic_values) {
        double min_aic = *std::min_element(aic_values.begin(), aic_values.end());
        std::vector<double> num(aic_values.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < aic_values.size(); ++i) {
            num[i] = std::exp(-0.5 * (aic_values[i] - min_aic));
            sum += num[i];
        }
        std::vector<double> weights(aic_values.size());
        for (std::size_t i = 0; i < aic_values.size(); ++i)
            weights[i] = num[i] / sum;
        return weights;
    }

    // -----------------------------------------------------------------------
    // Error Metrics
    // -----------------------------------------------------------------------

    /// RMSE with degrees-of-freedom correction k
    static double rmse(const std::vector<double>& observed,
                       const std::vector<double>& modeled, int k = 0) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        const int observation_count = static_cast<int>(observed.size());
        if (k < 0 || k >= observation_count) {
            throw std::out_of_range(
                "The number of model parameters must be nonnegative and less than the number of observations.");
        }
        const int degrees_of_freedom = observation_count - k;
        double sse = 0.0;
        for (int i = 0; i < observation_count; ++i) {
            double d = modeled[static_cast<std::size_t>(i)] - observed[static_cast<std::size_t>(i)];
            sse += d * d;
        }
        return std::sqrt(sse / degrees_of_freedom);
    }

    /// RMSE(observed, model): Weibull plotting positions, sort observed, compare to InverseCDF
    static double rmse(const std::vector<double>& observed_values,
                       const distributions::UnivariateDistributionBase& model) {
        std::vector<double> obs(observed_values);
        std::sort(obs.begin(), obs.end());
        int n = static_cast<int>(obs.size());
        // Weibull PP: pp[i-1] = i / (n+1)
        std::vector<double> modeled(static_cast<std::size_t>(n));
        for (int i = 1; i <= n; ++i)
            modeled[static_cast<std::size_t>(i - 1)] =
                model.inverse_cdf(static_cast<double>(i) / (n + 1.0));
        return rmse(obs, modeled, model.number_of_parameters());
    }

    /// RMSE(observed, plotting_positions, model)
    static double rmse(const std::vector<double>& observed_values,
                       const std::vector<double>& plotting_positions,
                       const distributions::UnivariateDistributionBase& model) {
        std::vector<double> modeled(plotting_positions.size());
        for (std::size_t i = 0; i < plotting_positions.size(); ++i)
            modeled[i] = model.inverse_cdf(plotting_positions[i]);
        return rmse(observed_values, modeled, model.number_of_parameters());
    }

    /// Inverse-MSE weights
    static std::vector<double> rmse_weights(const std::vector<double>& rmse_values) {
        std::vector<double> inv_mse(rmse_values.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < rmse_values.size(); ++i) {
            inv_mse[i] = 1.0 / (rmse_values[i] * rmse_values[i]);
            sum += inv_mse[i];
        }
        std::vector<double> weights(rmse_values.size());
        for (std::size_t i = 0; i < rmse_values.size(); ++i)
            weights[i] = inv_mse[i] / sum;
        return weights;
    }

    /// MSE
    static double mse(const std::vector<double>& observed, const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        int n = static_cast<int>(observed.size());
        double sse = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = modeled[static_cast<std::size_t>(i)] - observed[static_cast<std::size_t>(i)];
            sse += d * d;
        }
        return sse / n;
    }

    /// MAE
    static double mae(const std::vector<double>& observed, const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        int n = static_cast<int>(observed.size());
        double sum_abs = 0.0;
        for (int i = 0; i < n; ++i)
            sum_abs +=
                std::fabs(modeled[static_cast<std::size_t>(i)] - observed[static_cast<std::size_t>(i)]);
        return sum_abs / n;
    }

    /// MAPE (throws if any observed value is zero)
    static double mape(const std::vector<double>& observed, const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        int n = static_cast<int>(observed.size());
        double sum_pct = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(observed[static_cast<std::size_t>(i)]) <
                std::numeric_limits<double>::epsilon())
                throw std::invalid_argument(
                    "MAPE cannot be calculated when observed values contain zero or near-zero values.");
            sum_pct += std::fabs((observed[static_cast<std::size_t>(i)] -
                                  modeled[static_cast<std::size_t>(i)]) /
                                 observed[static_cast<std::size_t>(i)]);
        }
        return 100.0 * sum_pct / n;
    }

    /// sMAPE (throws if both observed and modeled are zero at any point)
    static double smape(const std::vector<double>& observed, const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        int n = static_cast<int>(observed.size());
        double sum_sym = 0.0;
        for (int i = 0; i < n; ++i) {
            std::size_t idx = static_cast<std::size_t>(i);
            double denom = std::fabs(observed[idx]) + std::fabs(modeled[idx]);
            if (denom < std::numeric_limits<double>::epsilon())
                throw std::invalid_argument(
                    "sMAPE cannot be calculated when both observed and modeled values are zero.");
            sum_sym += std::fabs(observed[idx] - modeled[idx]) / denom;
        }
        return 200.0 * sum_sym / n;
    }

    // -----------------------------------------------------------------------
    // Efficiency Coefficients
    // -----------------------------------------------------------------------

    /// Nash-Sutcliffe Efficiency
    static double nash_sutcliffe_efficiency(const std::vector<double>& observed,
                                            const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        int n = static_cast<int>(observed.size());
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += observed[static_cast<std::size_t>(i)];
        double obs_mean = sum / n;
        double num = 0.0, den = 0.0;
        for (int i = 0; i < n; ++i) {
            std::size_t idx = static_cast<std::size_t>(i);
            double d = observed[idx] - modeled[idx];
            num += d * d;
            double dev = observed[idx] - obs_mean;
            den += dev * dev;
        }
        return 1.0 - (num / den);
    }

    /// Log Nash-Sutcliffe Efficiency
    static double log_nash_sutcliffe_efficiency(const std::vector<double>& observed,
                                                const std::vector<double>& modeled,
                                                double epsilon = -1.0) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        if (epsilon < 0.0) {
            double s = 0.0;
            for (double v : observed) s += v;
            epsilon = (s / observed.size()) / 100.0;
        }
        std::vector<double> log_obs(observed.size()), log_mod(modeled.size());
        for (std::size_t i = 0; i < observed.size(); ++i) {
            log_obs[i] = std::log(observed[i] + epsilon);
            log_mod[i] = std::log(modeled[i] + epsilon);
        }
        return nash_sutcliffe_efficiency(log_obs, log_mod);
    }

    /// Kling-Gupta Efficiency (2009)
    static double kling_gupta_efficiency(const std::vector<double>& observed,
                                         const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        int n = static_cast<int>(observed.size());
        double sum_obs = 0.0, sum_mod = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_obs += observed[static_cast<std::size_t>(i)];
            sum_mod += modeled[static_cast<std::size_t>(i)];
        }
        double obs_mean = sum_obs / n, mod_mean = sum_mod / n;
        double var_obs = 0.0, var_mod = 0.0, cov = 0.0;
        for (int i = 0; i < n; ++i) {
            std::size_t idx = static_cast<std::size_t>(i);
            double od = observed[idx] - obs_mean, md = modeled[idx] - mod_mean;
            var_obs += od * od;
            var_mod += md * md;
            cov += od * md;
        }
        double obs_sd = std::sqrt(var_obs / (n - 1));
        double mod_sd = std::sqrt(var_mod / (n - 1));
        if (obs_sd < std::numeric_limits<double>::epsilon() ||
            mod_sd < std::numeric_limits<double>::epsilon())
            return -10.0;
        double r = cov / (n - 1) / (obs_sd * mod_sd);
        double alpha = mod_sd / obs_sd;
        double beta = mod_mean / obs_mean;
        double ed = std::sqrt((r - 1.0) * (r - 1.0) + (alpha - 1.0) * (alpha - 1.0) +
                              (beta - 1.0) * (beta - 1.0));
        return 1.0 - ed;
    }

    /// Modified Kling-Gupta Efficiency (2012): replaces alpha with CV ratio
    static double kling_gupta_efficiency_mod(const std::vector<double>& observed,
                                              const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        int n = static_cast<int>(observed.size());
        double sum_obs = 0.0, sum_mod = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_obs += observed[static_cast<std::size_t>(i)];
            sum_mod += modeled[static_cast<std::size_t>(i)];
        }
        double obs_mean = sum_obs / n, mod_mean = sum_mod / n;
        double var_obs = 0.0, var_mod = 0.0, cov = 0.0;
        for (int i = 0; i < n; ++i) {
            std::size_t idx = static_cast<std::size_t>(i);
            double od = observed[idx] - obs_mean, md = modeled[idx] - mod_mean;
            var_obs += od * od;
            var_mod += md * md;
            cov += od * md;
        }
        double obs_sd = std::sqrt(var_obs / (n - 1));
        double mod_sd = std::sqrt(var_mod / (n - 1));
        if (obs_sd < std::numeric_limits<double>::epsilon() ||
            mod_sd < std::numeric_limits<double>::epsilon())
            return -10.0;
        double r = cov / (n - 1) / (obs_sd * mod_sd);
        double gamma = (mod_sd / mod_mean) / (obs_sd / obs_mean);
        double beta = mod_mean / obs_mean;
        double ed = std::sqrt((r - 1.0) * (r - 1.0) + (gamma - 1.0) * (gamma - 1.0) +
                              (beta - 1.0) * (beta - 1.0));
        return 1.0 - ed;
    }

    // -----------------------------------------------------------------------
    // Bias Metrics
    // -----------------------------------------------------------------------

    /// Percent Bias: 100 * sum(M-O) / sum(O)
    static double pbias(const std::vector<double>& observed, const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        double sum_diff = 0.0, sum_obs = 0.0;
        for (std::size_t i = 0; i < observed.size(); ++i) {
            sum_diff += modeled[i] - observed[i];
            sum_obs += observed[i];
        }
        return 100.0 * sum_diff / sum_obs;
    }

    /// RMSE / StdDev(observations) ratio  [population std dev, N denominator]
    static double rsr(const std::vector<double>& observed, const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        int n = static_cast<int>(observed.size());
        double sse = 0.0, sum_obs = 0.0;
        for (std::size_t i = 0; i < observed.size(); ++i) {
            double d = modeled[i] - observed[i];
            sse += d * d;
            sum_obs += observed[i];
        }
        double rmse_val = std::sqrt(sse / n);
        double obs_mean = sum_obs / n;
        double var = 0.0;
        for (double v : observed) {
            double dev = v - obs_mean;
            var += dev * dev;
        }
        double std_dev = std::sqrt(var / n);
        return rmse_val / std_dev;
    }

    // -----------------------------------------------------------------------
    // Correlation and Determination
    // -----------------------------------------------------------------------

    /// Pearson correlation coefficient (mirrors Correlation.Pearson in C#)
    static double pearson(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size())
            throw std::invalid_argument("The samples arrays must be the same length.");
        int n = static_cast<int>(x.size());
        double ax = 0.0, ay = 0.0;
        for (int i = 0; i < n; ++i) {
            ax += x[static_cast<std::size_t>(i)];
            ay += y[static_cast<std::size_t>(i)];
        }
        ax /= n;
        ay /= n;
        double sxx = 0.0, syy = 0.0, sxy = 0.0;
        for (int i = 0; i < n; ++i) {
            std::size_t idx = static_cast<std::size_t>(i);
            double xt = x[idx] - ax, yt = y[idx] - ay;
            sxx += xt * xt;
            syy += yt * yt;
            sxy += xt * yt;
        }
        return sxy / std::sqrt(sxx * syy);
    }

    /// R² = Pearson²
    static double r_squared(const std::vector<double>& observed,
                            const std::vector<double>& modeled) {
        double r = pearson(observed, modeled);
        return r * r;
    }

    // -----------------------------------------------------------------------
    // Index of Agreement Metrics
    // -----------------------------------------------------------------------

    /// Willmott Index of Agreement (d)
    static double index_of_agreement(const std::vector<double>& observed,
                                     const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        int n = static_cast<int>(observed.size());
        double sum = 0.0;
        for (double v : observed) sum += v;
        double obs_mean = sum / n;
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < observed.size(); ++i) {
            double d = modeled[i] - observed[i];
            num += d * d;
            double dd = std::fabs(modeled[i] - obs_mean) + std::fabs(observed[i] - obs_mean);
            den += dd * dd;
        }
        return 1.0 - (num / den);
    }

    /// Modified Index of Agreement (d1) — absolute values
    static double modified_index_of_agreement(const std::vector<double>& observed,
                                              const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        int n = static_cast<int>(observed.size());
        double sum = 0.0;
        for (double v : observed) sum += v;
        double obs_mean = sum / n;
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < observed.size(); ++i) {
            num += std::fabs(modeled[i] - observed[i]);
            den += std::fabs(modeled[i] - obs_mean) + std::fabs(observed[i] - obs_mean);
        }
        return 1.0 - (num / den);
    }

    /// Refined Index of Agreement (dr) — range -1 to 1
    static double refined_index_of_agreement(const std::vector<double>& observed,
                                             const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        int n = static_cast<int>(observed.size());
        double sum = 0.0;
        for (double v : observed) sum += v;
        double obs_mean = sum / n;
        double sum_abs_err = 0.0, sum_abs_dev = 0.0;
        for (std::size_t i = 0; i < observed.size(); ++i) {
            sum_abs_err += std::fabs(modeled[i] - observed[i]);
            sum_abs_dev += std::fabs(observed[i] - obs_mean);
        }
        double c = 2.0 * sum_abs_dev;
        if (sum_abs_err <= c)
            return 1.0 - (sum_abs_err / c);
        else
            return (c / sum_abs_err) - 1.0;
    }

    /// Volumetric Efficiency (VE) = 1 - sum|M-O| / sum|O|
    static double volumetric_efficiency(const std::vector<double>& observed,
                                        const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        double sum_abs_err = 0.0, sum_abs_obs = 0.0;
        for (std::size_t i = 0; i < observed.size(); ++i) {
            sum_abs_err += std::fabs(modeled[i] - observed[i]);
            sum_abs_obs += std::fabs(observed[i]);
        }
        if (std::fabs(sum_abs_obs) < std::numeric_limits<double>::epsilon())
            throw std::invalid_argument(
                "VE cannot be calculated when sum of absolute observed values is zero.");
        return 1.0 - (sum_abs_err / sum_abs_obs);
    }

    // -----------------------------------------------------------------------
    // Classification Metrics
    // -----------------------------------------------------------------------

    static double accuracy(const std::vector<double>& observed,
                           const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        double cnt = 0;
        for (std::size_t i = 0; i < observed.size(); ++i)
            if (observed[i] == modeled[i]) cnt++;
        return 100.0 * cnt / observed.size();
    }

   private:
    struct ConfusionMatrix { int tp = 0, tn = 0, fp = 0, fn = 0; };

    static ConfusionMatrix confusion_matrix(const std::vector<double>& observed,
                                            const std::vector<double>& modeled) {
        if (observed.size() != modeled.size())
            throw std::invalid_argument(
                "The number of observed values must equal the number of modeled values.");
        ConfusionMatrix cm;
        for (std::size_t i = 0; i < observed.size(); ++i) {
            bool obs = std::fabs(observed[i] - 1.0) < 0.01;
            bool mod = std::fabs(modeled[i] - 1.0) < 0.01;
            if (obs && mod) cm.tp++;
            else if (!obs && !mod) cm.tn++;
            else if (!obs && mod) cm.fp++;
            else if (obs && !mod) cm.fn++;
        }
        return cm;
    }

   public:
    static double precision(const std::vector<double>& observed,
                            const std::vector<double>& modeled) {
        auto cm = confusion_matrix(observed, modeled);
        if (cm.tp + cm.fp == 0)
            throw std::invalid_argument(
                "Cannot calculate Precision when there are no positive predictions.");
        return static_cast<double>(cm.tp) / (cm.tp + cm.fp);
    }

    static double recall(const std::vector<double>& observed,
                         const std::vector<double>& modeled) {
        auto cm = confusion_matrix(observed, modeled);
        if (cm.tp + cm.fn == 0)
            throw std::invalid_argument(
                "Cannot calculate Recall when there are no actual positive cases.");
        return static_cast<double>(cm.tp) / (cm.tp + cm.fn);
    }

    static double f1_score(const std::vector<double>& observed,
                           const std::vector<double>& modeled) {
        auto cm = confusion_matrix(observed, modeled);
        if (2 * cm.tp + cm.fp + cm.fn == 0)
            throw std::invalid_argument(
                "Cannot calculate F1 Score when there are no positive predictions or actual "
                "positives.");
        return (2.0 * cm.tp) / (2 * cm.tp + cm.fp + cm.fn);
    }

    static double specificity(const std::vector<double>& observed,
                              const std::vector<double>& modeled) {
        auto cm = confusion_matrix(observed, modeled);
        if (cm.tn + cm.fp == 0)
            throw std::invalid_argument(
                "Cannot calculate Specificity when there are no actual negative cases.");
        return static_cast<double>(cm.tn) / (cm.tn + cm.fp);
    }

    static double balanced_accuracy(const std::vector<double>& observed,
                                    const std::vector<double>& modeled) {
        return (recall(observed, modeled) + specificity(observed, modeled)) / 2.0;
    }

    // -----------------------------------------------------------------------
    // Statistical Tests
    // -----------------------------------------------------------------------

    /// Kolmogorov-Smirnov test statistic D.
    /// observedValues must be sorted in ascending order.
    static double kolmogorov_smirnov(const std::vector<double>& observed_values,
                                     const distributions::UnivariateDistributionBase& model) {
        if (observed_values.empty())
            throw std::invalid_argument("There must be more than one observed value.");
        int n = static_cast<int>(observed_values.size());
        double D = -std::numeric_limits<double>::infinity();
        for (int i = 1; i <= n; ++i) {
            double x = observed_values[static_cast<std::size_t>(i - 1)];
            double F = model.cdf(x);
            double left = F - static_cast<double>(i - 1) / n;
            double right = static_cast<double>(i) / n - F;
            D = std::max(D, std::max(left, right));
        }
        return D;
    }

    /// Anderson-Darling test statistic A².
    /// observedValues must be sorted in ascending order.
    static double anderson_darling(const std::vector<double>& observed_values,
                                   const distributions::UnivariateDistributionBase& model) {
        if (observed_values.empty())
            throw std::invalid_argument("There must be more than one observed value.");
        int n = static_cast<int>(observed_values.size());
        double S = 0.0;
        for (int i = 1; i <= n; ++i) {
            double x_i = observed_values[static_cast<std::size_t>(i - 1)];
            double x_ni = observed_values[static_cast<std::size_t>(n - i)];
            S += (2.0 * i - 1.0) * (model.log_cdf(x_i) + model.log_ccdf(x_ni));
        }
        return -n - S / n;
    }

    /// Chi-Squared test statistic (Rice Rule binning — mirrors Histogram.cs).
    /// observedValues must be sorted in ascending order.
    static double chi_squared(const std::vector<double>& observed_values,
                              const distributions::UnivariateDistributionBase& model) {
        if (observed_values.empty())
            throw std::invalid_argument("There must be more than one observed value.");
        int n = static_cast<int>(observed_values.size());
        detail::Histogram hist(observed_values);
        double x2 = 0.0;
        for (const auto& bin : hist.bins) {
            double e = n * (model.cdf(bin.upper_bound) - model.cdf(bin.lower_bound));
            double diff = bin.frequency - e;
            x2 += diff * diff / e;
        }
        return x2;
    }
};

}  // namespace corehydro::numerics::data
