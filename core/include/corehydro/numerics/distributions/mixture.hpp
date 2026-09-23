// ported from: Numerics/Distributions/Univariate/Mixture.cs @ 7e8e8d1
//
// Mixture distribution, including positive-hurdle semantics when zero inflation is enabled.
// log_pdf: log-sum-exp stability (mirrors C# LogPDF).
// InverseCDF: scale-aware Brent root finding on log CDF or log survival, with the same
//   empirical fallback used by C# when a finite bracket cannot be solved.
// Moments: exact component-moment aggregation, with numerical positive-conditional moments
//   for hurdle components.
// Mode: BrentSearch maximizing the PDF over [InverseCDF(0.001), InverseCDF(0.999)]
//   (Mixture.cs line 337), replacing an earlier ternary search.
// IEstimation: MLE via EM algorithm (mirrors C# MLE/Estimate exactly):
//   E-step: log-sum-exp normalized component responsibilities.
//   M-step: normalize component responsibility mass onto the configured simplex, then
//   optimize component parameters with NelderMead and each component's shipped constraints.
//   Only ParameterEstimationMethod::MaximumLikelihood is supported; others throw.
// Zero-inflation: is_zero_inflated()/set_is_zero_inflated() and zero_weight()/set_zero_weight()
//   (mirrors C# IsZeroInflated/ZeroWeight -- see the v2.1.4 note below for the setter semantics).
// Mixture is composite-only and requires weights plus components.
// type() returns UnivariateDistributionType::Mixture (mirrors C#).
//
// v2.1.4 (313d7ba "Harden distribution parameter validation" + 7f8c652 "Preserve valid
// zero-inflated mixture weights"): IsZeroInflated/ZeroWeight became properties with side
// effects instead of plain auto-properties. Ported as set_is_zero_inflated()/set_zero_weight()
// (is_zero_inflated_/zero_weight_ are now private backing fields, no longer public data
// members -- every write goes through the setter, matching the C# where there is no way to
// bypass the property). Each setter: (1) stores the new value, (2) if IsZeroInflated is
// (now) true, calls normalize_component_weights() to rescale the finite, nonnegative
// component weights so they sum to 1 - ZeroWeight (invalid/negative/non-finite ZeroWeight or
// component weights leave the weights untouched, so ValidateParameters can still report the
// original error), (3) unconditionally calls refresh_configuration_state() to recompute
// parameters_valid_ and reset the moment/empirical-CDF caches. Setting IsZeroInflated then
// ZeroWeight in that order (as C#'s Clone() object initializer and every call site in this
// codebase do) performs the rescale TWICE -- once against the new object's default
// ZeroWeight=0 (a no-op renormalize-to-1), then again against the real ZeroWeight -- which is
// intentionally transcribed as-is rather than special-cased, since it reproduces the real C#
// bit-for-bit (including the tiny floating-point churn from the double rescale). The three
// other SetParameters overloads (weights+distributions x2, weights+flat-params) also gained a
// `_parametersValid = ValidateParameters(...) is null` recompute at their end (previously only
// the IList<double> override and SetParameters(ref) did); set_parameters(weights, parameters)
// below now does the same (comment updated -- no longer "does NOT update the flag").
//
// M10 additions (completing the C# Mixture surface the MixtureModel port consumes):
//   - IMaximumLikelihoodEstimation base + get_parameter_constraints (C# line 595):
//     weight rows first (equal initials, [0,1] bounds), then each component's own
//     IMaximumLikelihoodEstimation constraints.
//   - set_parameters(weights, parameters) (C# SetParameters(double[], double[]), line 411):
//     weights + component-parameter slices (v2.1.4 added the validity recompute -- see the
//     v2.1.4 note above).
//   - set_parameters_normalized(parameters&) (C# SetParameters(ref double[]), line 476):
//     weights normalized to sum to 1 (or 1 - ZeroWeight) through a private copy, leaving
//     the caller's coordinates unchanged, followed by the validity update.
//   - generate_random_values override (C# line 984): component-selection sampling from a
//     seeded MersenneTwister (u picks the component through the cumulative weights, a second
//     draw feeds the component's InverseCDF); zero inflation prepends a Deterministic(0)
//     "component" with weight ZeroWeight. Replaces the base-class inverse-CDF stream so
//     seeded mixture streams are bit-identical to the C#.
//
// X5 ADDITIVE PORT (Mixture.cs XTransform/ProbabilityTransform/CreateEmpiricalCDF @ 2a0357a): the
// CompositeAnalysis aggregation builds a Mixture per posterior realisation, sets XTransform =
// Logarithmic / ProbabilityTransform = NormalZ, and calls CreateEmpiricalCDF() so the InverseCDF
// the UncertaintyAnalysisResults reads is the fast piecewise empirical curve. NEW methods/fields
// only -- existing behavior and all existing fixtures stay byte-green (empirical_cdf_created_
// defaults false; the pre-existing root-find path is unchanged until CreateEmpiricalCDF()
// is explicitly called).
#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/deterministic.hpp"
#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "corehydro/numerics/distributions/base/distribution_moment_integration.hpp"
#include "corehydro/numerics/distributions/base/distribution_snapshot.hpp"
#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/sampling/stratify.hpp"
#include "corehydro/numerics/sampling/stratification_options.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class Mixture : public UnivariateDistributionBase,
                public IEstimation,
                public IMaximumLikelihoodEstimation {
   public:
    // Construct from weights + already-created component unique_ptrs (takes ownership).
    Mixture(std::vector<double> weights,
            std::vector<std::unique_ptr<UnivariateDistributionBase>> components)
        : weights_(std::move(weights)), components_(std::move(components)) {
        validate_and_set();
    }

    // Convenience: construct by cloning components from raw pointers.
    Mixture(const std::vector<double>& weights,
            const std::vector<UnivariateDistributionBase*>& components) {
        weights_ = weights;
        components_.reserve(components.size());
        for (auto* c : components) components_.push_back(c->clone());
        validate_and_set();
    }

    // Zero-inflation option (mirrors C# IsZeroInflated / ZeroWeight properties; v2.1.4 gave
    // both setters the renormalization side effect documented in the header comment above --
    // no longer plain public data members, since C# offers no way to bypass a property
    // setter). Reads: is_zero_inflated() / zero_weight(). Writes: set_is_zero_inflated(bool) /
    // set_zero_weight(double).
    bool is_zero_inflated() const { return is_zero_inflated_; }
    double zero_weight() const { return zero_weight_; }

    // Mirrors C# `IsZeroInflated` setter (Mixture.cs): store, then (if now true) rescale
    // component weights onto the 1 - ZeroWeight simplex, then refresh validity/caches.
    void set_is_zero_inflated(bool value) {
        is_zero_inflated_ = value;
        if (is_zero_inflated_) normalize_component_weights();
        refresh_configuration_state();
    }

    // Mirrors C# `ZeroWeight` setter (Mixture.cs): store, then (if IsZeroInflated) rescale
    // component weights onto the 1 - ZeroWeight simplex, then refresh validity/caches.
    void set_zero_weight(double value) {
        zero_weight_ = value;
        if (is_zero_inflated_) normalize_component_weights();
        refresh_configuration_state();
    }

    // X5: the x-value / probability transforms the empirical CDF interpolates in (mirrors C#
    // XTransform / ProbabilityTransform; defaults None / NormalZ).
    data::Transform x_transform = data::Transform::None;
    data::Transform probability_transform = data::Transform::NormalZ;

    // EM convergence settings (mirrors C# MaxIterations / Tolerance).
    int max_iterations = 1000;
    double tolerance = 1E-8;

    // Accessors.
    const std::vector<double>& weights() const { return weights_; }
    int component_count() const { return static_cast<int>(components_.size()); }
    const UnivariateDistributionBase& component(int i) const { return *components_[i]; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Mixture;
    }

    // K + sum of each component's NumberOfParameters (mirrors C#).
    int number_of_parameters() const override {
        int sum = static_cast<int>(components_.size());
        for (const auto& c : components_) sum += c->number_of_parameters();
        return sum;
    }

    // [w0, w1, ..., wK-1, p0_0, p0_1, ..., p1_0, p1_1, ...] (mirrors C# GetParameters).
    std::vector<double> get_parameters() const override {
        std::vector<double> result = weights_;
        for (const auto& c : components_) {
            auto p = c->get_parameters();
            result.insert(result.end(), p.begin(), p.end());
        }
        return result;
    }

    // Mirrors C# SetParameters(IList<double>): first K = weights, rest = component params.
    void set_parameters(const std::vector<double>& parameters) override {
        int K = static_cast<int>(components_.size());
        if (static_cast<int>(parameters.size()) != number_of_parameters())
            throw std::invalid_argument("parameter array length mismatch for Mixture");
        int t = 0;
        for (int i = 0; i < K; i++) weights_[i] = parameters[t++];
        for (int i = 0; i < K; i++) {
            int n = components_[i]->number_of_parameters();
            std::vector<double> p(parameters.begin() + t, parameters.begin() + t + n);
            components_[i]->set_parameters(p);
            t += n;
        }
        parameters_valid_ = validate_weights() && validate_components();
        moments_computed_ = false;
    }

    // Mirrors C# SetParameters(double[] weights, double[] parameters) (Mixture.cs line 411):
    // sets the weights and distributes the component-parameter slices. v2.1.4 added a
    // `_parametersValid` recompute at the end (previously this overload alone skipped it,
    // unlike SetParameters(IList<double>) and SetParameters(ref)) -- so this now recomputes
    // too. The EM E-step (MixtureModel) still drives intermediate, not-yet-converged weights
    // through this overload; parameters_valid_ is transiently false during those iterations,
    // which is harmless since pdf()/cdf()/log_pdf() never gate on it (matching the C#).
    void set_parameters(const std::vector<double>& weights,
                        const std::vector<double>& parameters) {
        if (weights.size() != components_.size())
            throw std::invalid_argument(
                "The weight and distribution arrays must have the same length.");
        int np = 0;
        for (const auto& c : components_) np += c->number_of_parameters();
        if (static_cast<int>(parameters.size()) != np)
            throw std::invalid_argument("The length of the parameter array is invalid.");

        weights_ = weights;
        int t = 0;
        for (std::size_t i = 0; i < components_.size(); ++i) {
            int n = components_[i]->number_of_parameters();
            std::vector<double> p(parameters.begin() + t, parameters.begin() + t + n);
            components_[i]->set_parameters(p);
            t += n;
        }
        parameters_valid_ = validate_weights() && validate_components();
        moments_computed_ = false;
    }

    // Mirrors C# SetParameters(ref double[] parameters): normalize a private copy of the
    // weights onto the configured simplex. Numerics v2.2 deliberately stopped modifying the
    // caller's array, despite retaining the restored ref signature for compatibility.
    void set_parameters_normalized(std::vector<double>& parameters) {
        if (weights_.empty()) return;
        if (components_.empty()) return;
        if (components_.size() == 1 &&
            static_cast<int>(parameters.size()) == components_[0]->number_of_parameters()) {
            weights_[0] = is_zero_inflated_ ? 1.0 - zero_weight_ : 1.0;
            components_[0]->set_parameters(parameters);
        } else {
            if (static_cast<int>(parameters.size()) != number_of_parameters())
                throw std::invalid_argument("The length of the parameter array is invalid.");

            // Get the weights.
            int K = static_cast<int>(components_.size());
            int t = 0;  // keep track of parameter index

            double sum = 0.0;
            for (int i = 0; i < K; ++i) {
                weights_[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(i)];
                sum += weights_[static_cast<std::size_t>(i)];
                t++;
            }

            if (sum <= 0.0) {
                // If weights sum to 0, reset to be uniformly distributed.
                double w = is_zero_inflated_ ? (1.0 - zero_weight_) / K : 1.0 / K;
                for (int i = 0; i < K; ++i) {
                    weights_[static_cast<std::size_t>(i)] = w;
                }
            } else {
                // Normalize weights to sum to 1.
                double c = is_zero_inflated_ ? (1.0 - zero_weight_) / sum : 1.0 / sum;
                for (int i = 0; i < K; ++i) {
                    weights_[static_cast<std::size_t>(i)] *= c;
                }
            }

            // Set distribution parameters.
            for (std::size_t i = 0; i < components_.size(); ++i) {
                int n = components_[i]->number_of_parameters();
                std::vector<double> parms(parameters.begin() + t, parameters.begin() + t + n);
                components_[i]->set_parameters(parms);
                t += n;
            }
        }

        parameters_valid_ = validate_weights() && validate_components();
        moments_computed_ = false;
    }

    // Mirrors C# GetParameterConstraints (Mixture.cs line 595): weight rows first (equal
    // initials in [0,1]), then each component's own IMaximumLikelihoodEstimation
    // constraints. A component without the capability throws std::bad_cast (the C# hard
    // cast's InvalidCastException).
    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials, std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        int n = number_of_parameters();
        int K = static_cast<int>(components_.size());
        initials.assign(static_cast<std::size_t>(n), 0.0);
        lowers.assign(static_cast<std::size_t>(n), 0.0);
        uppers.assign(static_cast<std::size_t>(n), 0.0);

        // Weights are first.
        int t = 0;
        for (int i = 0; i < K; ++i) {
            initials[static_cast<std::size_t>(i)] =
                is_zero_inflated_ ? (1.0 - zero_weight_) / K : 1.0 / K;
            lowers[static_cast<std::size_t>(i)] = 0.0;
            uppers[static_cast<std::size_t>(i)] = 1.0;
            t += 1;
        }

        for (int i = 0; i < K; ++i) {
            const auto& est = dynamic_cast<const IMaximumLikelihoodEstimation&>(
                *components_[static_cast<std::size_t>(i)]);
            std::vector<double> ci, cl, cu;
            est.get_parameter_constraints(sample, ci, cl, cu);

            int np_i = components_[static_cast<std::size_t>(i)]->number_of_parameters();
            for (int j = t; j < t + np_i; ++j) {
                initials[static_cast<std::size_t>(j)] = ci[static_cast<std::size_t>(j - t)];
                lowers[static_cast<std::size_t>(j)] = cl[static_cast<std::size_t>(j - t)];
                uppers[static_cast<std::size_t>(j)] = cu[static_cast<std::size_t>(j - t)];
            }
            t += np_i;
        }
    }

    // Mirrors C# GenerateRandomValues(int sampleSize, int seed = -1) (Mixture.cs line 984):
    // component-selection sampling (NOT the base-class inverse-CDF stream). Each draw
    // consumes one uniform for the component pick and one for the component's InverseCDF;
    // zero inflation prepends a Deterministic(0) pseudo-component with weight ZeroWeight.
    // C# `new double[sampleSize]` zero-fill is mirrored: if u falls past the cumulative
    // weights (weights not summing to 1) the sample stays 0.
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        sampling::MersenneTwister rnd =
            seed > 0 ? sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                     : sampling::MersenneTwister();
        std::vector<double> weights;
        std::vector<const UnivariateDistributionBase*> distributions;
        Deterministic zero_component(0.0);
        if (is_zero_inflated_) {
            weights.push_back(zero_weight_);
            distributions.push_back(&zero_component);
        }
        for (std::size_t i = 0; i < components_.size(); ++i) {
            weights.push_back(weights_[i]);
            distributions.push_back(components_[i].get());
        }

        std::vector<double> sample(static_cast<std::size_t>(sample_size), 0.0);
        for (int i = 0; i < sample_size; ++i) {
            double u = rnd.next_double();
            double cdf_w = 0.0;
            for (std::size_t j = 0; j < distributions.size(); ++j) {
                cdf_w = j == 0 ? weights[j] : cdf_w + weights[j];
                if (u <= cdf_w) {
                    sample[static_cast<std::size_t>(i)] =
                        distributions[j]->inverse_cdf(rnd.next_double());
                    break;
                }
            }
        }
        return sample;
    }

    // --- Moments / support ---
    double mean() const override {
        refresh_cached_configuration();
        if (!moments_computed_) compute_moments();
        return u_[0];
    }
    double median() const override { return inverse_cdf(0.5); }
    double mode() const override {
        // Mirrors C# Mode (Mixture.cs line 337): BrentSearch maximizing the PDF over
        // [InverseCDF(0.001), InverseCDF(0.999)], returning BestParameterSet.Values[0].
        // This used to ternary-search the same interval, which agrees only where the mixture
        // density is unimodal; Brent reproduces C# bit-for-bit.
        double lo = inverse_cdf(0.001);
        double hi = inverse_cdf(0.999);
        math::optimization::BrentSearch brent([this](double x) { return pdf(x); }, lo, hi);
        brent.maximize();
        return brent.best_parameter();
    }
    double standard_deviation() const override {
        refresh_cached_configuration();
        if (!moments_computed_) compute_moments();
        return u_[1];
    }
    double skewness() const override {
        refresh_cached_configuration();
        if (!moments_computed_) compute_moments();
        return u_[2];
    }
    double kurtosis() const override {
        refresh_cached_configuration();
        if (!moments_computed_) compute_moments();
        return u_[3];
    }
    // Minimum = min over components (mirrors C# Distributions.Min(p => p.Minimum)).
    double minimum() const override {
        validate_evaluation();
        double m = kInf;
        if (is_zero_inflated_ && zero_weight_ > 0.0) m = 0.0;
        for (std::size_t i = 0; i < components_.size(); ++i)
            if (weights_[i] > 0.0) m = std::min(m, components_[i]->minimum());
        if (is_zero_inflated_) m = std::max(0.0, m);
        return m;
    }
    // Maximum = max over components (mirrors C# Distributions.Max(p => p.Maximum)).
    double maximum() const override {
        validate_evaluation();
        double m = -kInf;
        for (std::size_t i = 0; i < components_.size(); ++i)
            if (weights_[i] > 0.0) m = std::max(m, components_[i]->maximum());
        return m;
    }

    // --- Distribution functions ---
    // Mirrors C# PDF: f = Σ wᵢ fᵢ(x); clamp to [0, ∞).
    double pdf(double x) const override { return std::exp(log_pdf(x)); }

    // Mirrors C# LogPDF: log-sum-exp over log(wᵢ) + log fᵢ(x).
    double log_pdf(double x) const override {
        validate_evaluation();
        if (is_zero_inflated_ && x <= 0.0)
            return x == 0.0 ? std::log(zero_weight_) : -kInf;
        std::vector<double> lnf;
        for (std::size_t i = 0; i < components_.size(); ++i)
            if (weights_[i] > 0.0)
                lnf.push_back(std::log(weights_[i]) +
                              (is_zero_inflated_ ? positive_conditional_log_pdf(i, x)
                                                 : components_[i]->log_pdf(x)));
        return log_sum_exp(lnf);
    }

    // Mirrors C# CDF: F = Σ wᵢ Fᵢ(x); clamped to [0,1].
    double cdf(double x) const override { return std::exp(log_cdf(x)); }

    double log_cdf(double x) const override {
        validate_evaluation();
        if (is_zero_inflated_ && x < 0.0) return -kInf;
        double total = is_zero_inflated_ ? std::log(zero_weight_) : -kInf;
        for (std::size_t i = 0; i < components_.size(); ++i)
            if (weights_[i] > 0.0) {
                const double value = is_zero_inflated_
                                         ? positive_conditional_log_cdf(i, x)
                                         : components_[i]->log_cdf(x);
                total = distribution_numerics::log_sum(
                    total, std::log(weights_[i]) + value);
            }
        return std::min(0.0, total);
    }

    double ccdf(double x) const override { return std::exp(log_ccdf(x)); }

    double log_ccdf(double x) const override {
        validate_evaluation();
        if (is_zero_inflated_ && x < 0.0) return 0.0;
        double total = -kInf;
        for (std::size_t i = 0; i < components_.size(); ++i)
            if (weights_[i] > 0.0) {
                const double value = is_zero_inflated_
                                         ? positive_conditional_log_ccdf(i, x)
                                         : components_[i]->log_ccdf(x);
                total = distribution_numerics::log_sum(
                    total, std::log(weights_[i]) + value);
            }
        return std::min(0.0, total);
    }

    // Mirrors C# InverseCDF: Brent solve on CDF(y) - probability = 0.
    // Bracket is derived from per-component InverseCDF values.
    // On bracket or solve failure, falls back to the empirical approximation.
    double inverse_cdf(double probability) const override {
        refresh_cached_configuration();
        validate_evaluation();
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (is_zero_inflated_ && probability <= zero_weight_) return 0.0;

        // Single component, not zero-inflated: delegate.
        if (components_.size() == 1 && !is_zero_inflated_)
            return components_[0]->inverse_cdf(probability);

        // X5: once CreateEmpiricalCDF() has been called (composite aggregation path), read the
        // fast piecewise empirical curve, mirroring C# `if (_empiricalCDFCreated) x =
        // _empiricalCDF.InverseCDF(probability)`.
        if (empirical_cdf_created_) {
            double xe = empirical_cdf_->inverse_cdf(probability);
            double mn0 = minimum(), mx0 = maximum();
            return xe < mn0 ? mn0 : xe > mx0 ? mx0 : xe;
        }

        // Derive the bracket from each active component at the same probability.
        const double component_probability = is_zero_inflated_
            ? (probability - zero_weight_) / (1.0 - zero_weight_)
            : probability;
        double minX = kInf, maxX = -kInf;
        for (std::size_t i = 0; i < components_.size(); ++i) {
            if (weights_[i] == 0.0) continue;
            const double quantile = is_zero_inflated_
                                        ? positive_conditional_quantile(i, component_probability)
                                        : components_[i]->inverse_cdf(component_probability);
            minX = std::min(minX, quantile);
            maxX = std::max(maxX, quantile);
        }
        if (minX == maxX) return std::clamp(minX, minimum(), maximum());

        double x = 0.0;
        try {
            const double width = maxX - minX;
            const auto argument = [minX, maxX, width](double t) {
                return std::isfinite(width) ? minX + width * t
                                            : (1.0 - t) * minX + t * maxX;
            };
            const auto residual = [this, probability, &argument](double t) {
                const double value = argument(t);
                return probability <= 0.5
                           ? log_cdf(value) - std::log(probability)
                           : log_ccdf(value) - std::log1p(-probability);
            };
            const double scale = std::isfinite(width)
                                     ? width
                                     : std::max(std::fabs(minX), std::fabs(maxX));
            x = argument(math::rootfinding::solve(
                residual, 0.0, 1.0, 1E-6 / std::max(1.0, scale), 100, true));
        } catch (...) {
            if (!empirical_cdf_created_) const_cast<Mixture*>(this)->create_empirical_cdf();
            x = empirical_cdf_->inverse_cdf(probability);
        }
        double mn = minimum(), mx = maximum();
        return x < mn ? mn : x > mx ? mx : x;
    }

    // --- Estimation (IEstimation) ---
    // Mirrors C# Estimate: only MLE supported (others throw NotImplementedException).
    void estimate(const std::vector<double>& sample,
                  ParameterEstimationMethod method) override {
        if (method != ParameterEstimationMethod::MaximumLikelihood)
            throw std::runtime_error("Mixture only supports MaximumLikelihood estimation");
        set_parameters(mle(sample));
    }

    // --- Parameter display names (X1; C# Mixture.cs ParameterNames / ParameterNamesShortForm).
    // BOTH are OVERRIDDEN dynamically in C# (154-195): they are NOT the ParametersToString col-0
    // "Weights"/"Distributions" entries, but "Weight 1..n" ("W1..Wn" short) then "D{i+1} {sub}"
    // over every component's own names (long form) / short form (C# 154-195). ---
    std::vector<std::string> parameter_names() const override {
        std::vector<std::string> result;
        for (int i = 1; i <= component_count(); ++i)
            result.push_back("Weight " + std::to_string(i));
        for (int i = 0; i < component_count(); ++i) {
            std::vector<std::string> sub = component(i).parameter_names();
            for (const std::string& s : sub)
                result.push_back("D" + std::to_string(i + 1) + " " + s);
        }
        return result;
    }
    std::vector<std::string> parameter_names_short_form() const override {
        std::vector<std::string> result;
        for (int i = 1; i <= component_count(); ++i) result.push_back("W" + std::to_string(i));
        for (int i = 0; i < component_count(); ++i) {
            std::vector<std::string> sub = component(i).parameter_names_short_form();
            for (const std::string& s : sub)
                result.push_back("D" + std::to_string(i + 1) + " " + s);
        }
        return result;
    }

    // X5: builds the piecewise EmpiricalDistribution backing the composite InverseCDF (mirrors C#
    // Mixture.CreateEmpiricalCDF, Mixture.cs:939). Verbatim port; the backing EmpiricalDistribution
    // carries this XTransform / ProbabilityTransform. cdf() here is the mixture CDF (zero-inflation
    // included), so a zero-inflated mixture bakes its inflation into the empirical curve.
    void create_empirical_cdf() {
        double min_p = 1E-16;
        double max_p = 1.0 - 1E-16;
        double minX = is_zero_inflated_ ? 0.0 : kInf;
        double maxX = -kInf;
        for (std::size_t i = 0; i < components_.size(); ++i) {
            if (weights_[i] <= 0.0) continue;
            if (!is_zero_inflated_)
                minX = std::min(minX, components_[i]->inverse_cdf(min_p));
            maxX = std::max(
                maxX, is_zero_inflated_ ? positive_conditional_quantile(i, max_p)
                                        : components_[i]->inverse_cdf(max_p));
        }
        double shift = 0.0;
        if (minX <= 0.0) shift = std::fabs(minX) + 1.0;
        double mn = minX + shift;
        double mx = maxX + shift;
        int order = static_cast<int>(std::floor(std::log10(mx) - std::log10(mn)));
        int binN = std::max(200, 100 * order) - 1;

        auto bins = sampling::Stratify::XValues(
            sampling::StratificationOptions(minX, maxX, binN, false),
            true);
        std::vector<double> x_values, p_values;
        double x = bins.front().lower_bound();
        double p = cdf(bins.front().lower_bound());
        x_values.push_back(x);
        p_values.push_back(p);
        for (std::size_t i = 1; i < bins.size(); ++i) {
            x = bins[i].lower_bound();
            p = cdf(x);
            if (x > x_values.back() && p > p_values.back()) {
                x_values.push_back(x);
                p_values.push_back(p);
            }
        }
        x = maxX;
        p = cdf(x);
        if (x > x_values.back() && p > p_values.back()) {
            x_values.push_back(x);
            p_values.push_back(p);
        }

        auto ecdf = std::make_unique<EmpiricalDistribution>(
            x_values, p_values,
            probability_transform == data::Transform::NormalZ ? EmpiricalTransform::NormalZ
                                                              : EmpiricalTransform::None);
        ecdf->set_x_transform(x_transform);
        empirical_cdf_ = std::move(ecdf);
        empirical_cdf_created_ = true;
        moments_computed_ = false;
    }

    // Mirrors C# Clone(): `new Mixture(Weights.ToArray(), dists) { IsZeroInflated =
    // IsZeroInflated, ZeroWeight = ZeroWeight, ... }`. The object-initializer order
    // (IsZeroInflated before ZeroWeight) matters for bit-exactness -- see the v2.1.4 header
    // note -- so this calls the two setters in that same order rather than copying the fields
    // directly.
    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        std::vector<std::unique_ptr<UnivariateDistributionBase>> cloned;
        cloned.reserve(components_.size());
        for (const auto& c : components_) cloned.push_back(c->clone());
        auto m = std::make_unique<Mixture>(weights_, std::move(cloned));
        m->set_is_zero_inflated(is_zero_inflated_);
        m->set_zero_weight(zero_weight_);
        m->max_iterations = max_iterations;
        m->tolerance = tolerance;
        m->x_transform = x_transform;
        m->probability_transform = probability_transform;
        return m;
    }

   private:
    std::vector<double> weights_;
    std::vector<std::unique_ptr<UnivariateDistributionBase>> components_;
    bool is_zero_inflated_ = false;
    double zero_weight_ = 0.0;

    // X5: lazily-built empirical CDF backing (mirrors C# _empiricalCDF / _empiricalCDFCreated).
    std::unique_ptr<EmpiricalDistribution> empirical_cdf_;
    bool empirical_cdf_created_ = false;

    // Lazy moment cache.
    mutable bool moments_computed_ = false;
    mutable double u_[4] = {kNaN, kNaN, kNaN, kNaN};  // [mean, sd, skewness, kurtosis]
    mutable std::optional<DistributionSnapshot> configuration_snapshot_;

    void refresh_cached_configuration() const {
        if (configuration_snapshot_ && configuration_snapshot_->matches(this)) return;
        moments_computed_ = false;
        const_cast<Mixture*>(this)->empirical_cdf_created_ = false;
        configuration_snapshot_ = DistributionSnapshot::try_capture(this);
    }

    double positive_log_mass(std::size_t index) const {
        const double value = components_[index]->log_ccdf(0.0);
        if (!std::isfinite(value) || value > 0.0)
            throw std::runtime_error("active mixture component has no positive mass");
        return value;
    }

    double positive_conditional_log_pdf(std::size_t index, double x) const {
        return x > 0.0 ? components_[index]->log_pdf(x) - positive_log_mass(index) : -kInf;
    }

    double positive_conditional_log_cdf(std::size_t index, double x) const {
        return x <= 0.0
                   ? -kInf
                   : components_[index]->log_likelihood_intervals(0.0, x) -
                         positive_log_mass(index);
    }

    double positive_conditional_log_ccdf(std::size_t index, double x) const {
        return x <= 0.0
                   ? 0.0
                   : std::min(0.0, components_[index]->log_ccdf(x) -
                                       positive_log_mass(index));
    }

    double positive_conditional_quantile(std::size_t index, double probability) const {
        const auto& distribution = *components_[index];
        const double lower = std::max(0.0, distribution.minimum());
        if (probability == 0.0) return lower;
        if (probability == 1.0) return distribution.maximum();
        const double target = positive_log_mass(index) + std::log1p(-probability);
        double scale = distribution.inverse_cdf(0.75) - distribution.inverse_cdf(0.25);
        if (!(scale > 0.0) || !std::isfinite(scale)) scale = std::max(1.0, std::fabs(lower));
        double upper = std::min(distribution.maximum(), lower + scale);
        for (int i = 0; distribution.log_ccdf(upper) > target && i < 1024; ++i) {
            scale *= 2.0;
            const double next = lower + scale;
            upper = std::min(distribution.maximum(),
                             std::isfinite(next) ? next
                                                 : std::numeric_limits<double>::max());
        }
        if (!(upper > lower) || distribution.log_ccdf(upper) > target)
            throw std::runtime_error("positive mixture quantile could not be bracketed");
        const double width = upper - lower;
        return lower + width * math::rootfinding::solve(
                                   [&](double t) {
                                       return distribution.log_ccdf(lower + width * t) - target;
                                   },
                                   0.0, 1.0, 1e-6 / std::max(1.0, width), 100, true);
    }

    void validate_evaluation() const {
        if (!validate_weights() || !validate_components())
            throw std::out_of_range("Mixture: invalid weights or component parameters");
        if (is_zero_inflated_) {
            if (!std::isfinite(zero_weight_) || zero_weight_ < 0.0 || zero_weight_ >= 1.0)
                throw std::out_of_range("Mixture: zero weight must be in [0, 1)");
            for (std::size_t i = 0; i < components_.size(); ++i)
                if (weights_[i] > 0.0) (void)positive_log_mass(i);
        }
    }

    // Mirrors C# NormalizeComponentWeights (Mixture.cs, v2.1.4): rescales finite, nonnegative
    // component weights so they sum to 1 - zero_weight_. Bails out (leaving weights_
    // untouched) if there are no weights, zero_weight_ is not a finite value in [0, 1], any
    // component weight is not finite/nonnegative, or the weights sum to <= 0 or +inf --
    // exactly the C# guard order, so ValidateParameters can still surface the original error.
    void normalize_component_weights() {
        if (weights_.empty() || std::isnan(zero_weight_) || std::isinf(zero_weight_) ||
            zero_weight_ < 0.0 || zero_weight_ > 1.0) {
            return;
        }

        double sum = 0.0;
        for (double w : weights_) {
            if (std::isnan(w) || std::isinf(w) || w < 0.0) return;
            sum += w;
        }

        if (sum <= 0.0 || std::isinf(sum)) return;

        double scale = (1.0 - zero_weight_) / sum;
        for (double& w : weights_) w *= scale;
    }

    // Mirrors C# RefreshConfigurationState (Mixture.cs, v2.1.4): recomputes parameters_valid_
    // and resets the moment / empirical-CDF caches after a zero-inflation configuration change.
    void refresh_configuration_state() {
        if (weights_.empty() || components_.empty()) {
            parameters_valid_ = false;
        } else {
            parameters_valid_ = validate_weights() && validate_components();
        }
        moments_computed_ = false;
        empirical_cdf_created_ = false;
    }

    void validate_and_set() {
        if (weights_.size() != components_.size())
            throw std::invalid_argument("Mixture: weights and components must have the same size");
        parameters_valid_ = validate_weights() && validate_components();
        moments_computed_ = false;
    }

    bool validate_weights() const {
        if (is_zero_inflated_ && (!std::isfinite(zero_weight_) || zero_weight_ < 0.0 ||
                                  zero_weight_ >= 1.0))
            return false;
        double sum = is_zero_inflated_ ? zero_weight_ : 0.0;
        for (double w : weights_) {
            if (w < 0.0 || w > 1.0) return false;
            sum += w;
        }
        // Mirrors C# AlmostEquals(1, 1e-8).
        return std::fabs(sum - 1.0) <= 1e-8;
    }

    bool validate_components() const {
        for (const auto& c : components_)
            if (!c->parameters_valid()) return false;
        return true;
    }

    // log-sum-exp trick: log(Σ exp(lnf[i])).
    static double log_sum_exp(const std::vector<double>& lnf) {
        if (lnf.empty()) return -std::numeric_limits<double>::infinity();
        double max_val = *std::max_element(lnf.begin(), lnf.end());
        if (std::isinf(max_val)) return max_val;
        double sum = 0.0;
        for (double v : lnf) sum += std::exp(v - max_val);
        return max_val + std::log(sum);
    }

    // Mirrors C# ComputeMoments() (Mixture.cs line 312): CentralMoments(1000). The integer
    // literal binds to the base class's fixed-step TRAPEZOIDAL overload (`CentralMoments(int
    // steps)`), not the adaptive-tolerance one -- see docs/upstream-csharp-issues.md. This
    // file used to call adaptive Gauss-Kronrod here, which agreed with C# only to about six
    // digits; the 1000-step trapezoid reproduces it to ~1e-14 relative.
    void compute_moments() const {
        validate_evaluation();
        struct Moments {
            double weight;
            double mean;
            double sd;
            double skew;
            double kurt;
        };
        std::vector<Moments> values;
        if (is_zero_inflated_ && zero_weight_ > 0.0)
            values.push_back({zero_weight_, 0.0, 0.0, 0.0, 0.0});
        for (std::size_t i = 0; i < components_.size(); ++i) {
            if (weights_[i] == 0.0) continue;
            const auto& distribution = *components_[i];
            if (is_zero_inflated_ && distribution.log_cdf(0.0) != -kInf) {
                const double center = positive_conditional_quantile(i, 0.5);
                const double scale = positive_conditional_quantile(i, 0.75) -
                                     positive_conditional_quantile(i, 0.25);
                const auto moments = distribution_moment_integration::compute(
                    [&](double x) { return positive_conditional_log_pdf(i, x); },
                    std::max(0.0, distribution.minimum()), distribution.maximum(),
                    center, scale);
                values.push_back(
                    {weights_[i], moments[0], moments[1], moments[2], moments[3]});
            } else {
                values.push_back({weights_[i], distribution.mean(),
                                  distribution.standard_deviation(), distribution.skewness(),
                                  distribution.kurtosis()});
            }
        }
        const double reference = values.front().mean;
        double offset = 0.0;
        double total_weight = 0.0;
        for (const auto& value : values) {
            offset += value.weight * (value.mean - reference);
            total_weight += value.weight;
        }
        u_[0] = reference * total_weight + offset;
        double scale = 0.0;
        for (const auto& value : values)
            scale = std::max(scale, std::max(value.sd, std::fabs(value.mean - u_[0])));
        if (!std::isfinite(scale)) {
            u_[1] = scale;
            u_[2] = u_[3] = kNaN;
            moments_computed_ = true;
            return;
        }
        if (scale == 0.0) {
            u_[1] = 0.0;
            u_[2] = u_[3] = kNaN;
            moments_computed_ = true;
            return;
        }
        double m2 = 0.0, m3 = 0.0, m4 = 0.0;
        for (const auto& value : values) {
            const double d = distribution_numerics::standardize(value.mean, u_[0], scale);
            const double sd = value.sd / scale;
            const double variance = sd * sd;
            const double d2 = d * d;
            const double third = sd == 0.0 ? 0.0 : value.skew * variance * sd;
            const double fourth = sd == 0.0 ? 0.0 : value.kurt * variance * variance;
            m2 += value.weight * (variance + d2);
            m3 += value.weight * (third + 3.0 * d * variance + d * d2);
            m4 += value.weight *
                  (fourth + 4.0 * d * third + 6.0 * d2 * variance + d2 * d2);
        }
        u_[1] = scale * std::sqrt(m2);
        u_[2] = m3 / m2 / std::sqrt(m2);
        u_[3] = m4 / m2 / m2;
        moments_computed_ = true;
    }

    // EM-based MLE. Mirrors C# MLE() method.
    std::vector<double> mle(const std::vector<double>& sample) const {
        validate_evaluation();
        int N = static_cast<int>(sample.size());
        int K = static_cast<int>(components_.size());
        if (is_zero_inflated_)
            for (std::size_t i = 0; i < sample.size(); ++i) {
                if (sample[i] < 0.0)
                    throw std::runtime_error(
                        "Mixture EM cannot fit negative values in a zero-inflated model");
                if (sample[i] == 0.0 && zero_weight_ == 0.0)
                    throw std::runtime_error("Mixture EM row has zero total probability");
            }
        // Total component parameters (excluding weights).
        int Np = 0;
        for (const auto& c : components_) Np += c->number_of_parameters();

        // Get each component's shipped maximum-likelihood initials and bounds.
        std::vector<double> initials, lowers, uppers;
        for (int i = 0; i < K; ++i) {
            const auto* mle_component =
                dynamic_cast<const IMaximumLikelihoodEstimation*>(components_[i].get());
            if (mle_component == nullptr)
                throw std::runtime_error(
                    "Mixture component does not provide maximum-likelihood constraints");
            std::vector<double> component_initials, component_lowers, component_uppers;
            mle_component->get_parameter_constraints(
                sample, component_initials, component_lowers, component_uppers);
            initials.insert(initials.end(), component_initials.begin(), component_initials.end());
            lowers.insert(lowers.end(), component_lowers.begin(), component_lowers.end());
            uppers.insert(uppers.end(), component_uppers.begin(), component_uppers.end());
        }

        // EM weights (start uniform, not zero-inflated adjusted for now).
        std::vector<double> mle_weights(K, is_zero_inflated_
            ? (1.0 - zero_weight_) / K : 1.0 / K);
        std::vector<double> mle_params = initials;

        // Responsibility matrix [N][K] (row-major).
        std::vector<std::vector<double>> likelihood(N, std::vector<double>(K, 0.0));

        // E-step: compute normalized log-responsibilities.
        // Returns total log-likelihood; mirrors C# EStep.
        auto e_step = [&](const std::vector<double>& x) -> double {
            // Build a clone with current weights + params.
            auto dist_clone = static_cast<Mixture*>(this->clone().release());
            std::unique_ptr<Mixture> dist_ptr(dist_clone);
            // Set weights (don't validate -- may be unnormalized during EM).
            for (int k = 0; k < K; ++k) dist_ptr->weights_[k] = mle_weights[k];
            // Set component params.
            int t = 0;
            for (int i = 0; i < K; ++i) {
                int n = dist_ptr->components_[i]->number_of_parameters();
                std::vector<double> p(x.begin() + t, x.begin() + t + n);
                dist_ptr->components_[i]->set_parameters(p);
                t += n;
            }
            // Compute log-likelihoods.
            for (int i = 0; i < N; ++i) {
                if (is_zero_inflated_ && sample[i] == 0.0) {
                    if (!(zero_weight_ > 0.0) || !std::isfinite(zero_weight_))
                        throw std::runtime_error("Mixture EM row has zero total probability");
                    for (int k = 0; k < K; ++k) likelihood[i][k] = 0.0;
                    continue;
                }
                for (int k = 0; k < K; ++k)
                    likelihood[i][k] =
                        mle_weights[k] == 0.0
                            ? -kInf
                            : std::log(mle_weights[k]) +
                                  (is_zero_inflated_
                                       ? dist_ptr->positive_conditional_log_pdf(
                                             static_cast<std::size_t>(k), sample[i])
                                       : dist_ptr->components_[k]->log_pdf(sample[i]));
            }
            // Log-sum-exp normalization per sample point.
            double logLH = 0.0;
            for (int i = 0; i < N; ++i) {
                if (is_zero_inflated_ && sample[i] == 0.0) {
                    logLH += std::log(zero_weight_);
                    continue;
                }
                double max_val = -std::numeric_limits<double>::infinity();
                for (int k = 0; k < K; ++k)
                    if (likelihood[i][k] > max_val) max_val = likelihood[i][k];
                if (std::isinf(max_val)) {
                    throw std::runtime_error("Mixture EM row has zero total probability");
                }
                double sum = 0.0;
                for (int k = 0; k < K; ++k) sum += std::exp(likelihood[i][k] - max_val);
                double tmp = max_val + std::log(sum);
                for (int k = 0; k < K; ++k)
                    likelihood[i][k] = std::exp(likelihood[i][k] - tmp);
                logLH += tmp;
            }
            return logLH;
        };

        // M-step: update weights, optimize component params via NelderMead.
        auto m_step = [&](const std::vector<double>& x) -> std::vector<double> {
            // Update weights.
            for (int k = 0; k < K; ++k) {
                double wgt = 0.0;
                for (int i = 0; i < N; ++i) {
                    if (!is_zero_inflated_ || sample[i] > 0.0)
                        wgt += likelihood[i][k];
                }
                mle_weights[k] = wgt;
            }
            const double weight_sum =
                std::accumulate(mle_weights.begin(), mle_weights.end(), 0.0);
            if (!(weight_sum > 0.0) || !std::isfinite(weight_sum))
                throw std::runtime_error(
                    "Mixture EM has no finite positive responsibility mass");
            const double target = is_zero_inflated_ ? 1.0 - zero_weight_ : 1.0;
            for (double& weight : mle_weights) weight *= target / weight_sum;
            // NelderMead on component parameters only (weights held fixed).
            auto log_lh_fn = [&](const std::vector<double>& p) -> double {
                auto dist_clone = static_cast<Mixture*>(this->clone().release());
                std::unique_ptr<Mixture> dc(dist_clone);
                for (int k = 0; k < K; ++k) dc->weights_[k] = mle_weights[k];
                int t2 = 0;
                for (int i = 0; i < K; ++i) {
                    int n = dc->components_[i]->number_of_parameters();
                    std::vector<double> cp(p.begin() + t2, p.begin() + t2 + n);
                    dc->components_[i]->set_parameters(cp);
                    t2 += n;
                }
                double lh = dc->log_likelihood(sample);
                if (std::isnan(lh) || std::isinf(lh)) return -std::numeric_limits<double>::infinity();
                return lh;
            };
            math::optimization::NelderMead solver(log_lh_fn, Np, x, lowers, uppers);
            solver.maximize();
            return solver.best_parameters();
        };

        // EM iterations (mirrors C# loop).
        double old_log_lh = std::numeric_limits<double>::lowest();
        double new_log_lh = std::numeric_limits<double>::lowest();
        for (int iter = 1; iter <= max_iterations; ++iter) {
            new_log_lh = e_step(mle_params);
            // Convergence check before M-step (mirrors C# order).
            if (std::fabs((old_log_lh - new_log_lh) / old_log_lh) < tolerance) break;
            mle_params = m_step(mle_params);
            old_log_lh = new_log_lh;
        }

        // Return [weights, component_params].
        std::vector<double> result = mle_weights;
        result.insert(result.end(), mle_params.begin(), mle_params.end());
        return result;
    }
};

}  // namespace corehydro::numerics::distributions
