// ported from: Numerics/Distributions/Univariate/Mixture.cs @ 2a0357a
//
// Mixture distribution: PDF = Σ wᵢ fᵢ(x); CDF = Σ wᵢ Fᵢ(x).
// log_pdf: log-sum-exp stability (mirrors C# LogPDF).
// InverseCDF: Brent root-finding on CDF(y) - p. Falls back to coarse bisection on
//   bracket failure (the C# falls back to EmpiricalDistribution; we skip that
//   dependency and bisect instead — divergence noted here).
// Moments: central_moments(1000), the fixed-step trapezoidal overload the C# integer
//   literal in `CentralMoments(1000)` binds to (Mixture.cs line 312). This file used to
//   call adaptive Gauss-Kronrod instead, which agreed with the C# only to about six digits.
// Mode: BrentSearch maximizing the PDF over [InverseCDF(0.001), InverseCDF(0.999)]
//   (Mixture.cs line 337), replacing an earlier ternary search.
// IEstimation: MLE via EM algorithm (mirrors C# MLE/Estimate exactly):
//   E-step: log-sum-exp normalized component responsibilities.
//   M-step: update weights (Σ resp / N), optimize component params via NelderMead.
//   Parameter constraints for NelderMead: call component estimate() to get initial
//   values; bounds set to ±10x of sample statistics (C# calls GetParameterConstraints
//   via IMaximumLikelihoodEstimation which is not ported; functional divergence for MLE
//   only — PDF/CDF/moments are exact).
//   Only ParameterEstimationMethod::MaximumLikelihood is supported; others throw.
// Zero-inflation: is_zero_inflated()/set_is_zero_inflated() and zero_weight()/set_zero_weight()
//   (mirrors C# IsZeroInflated/ZeroWeight -- see the v2.1.4 note below for the setter semantics).
// Not in the factory — Mixture is composite-only (requires weights + components).
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
//     IMaximumLikelihoodEstimation constraints. The internal mle() below still uses its
//     Phase 2 heuristic bounds (documented divergence, unchanged in M10).
//   - set_parameters(weights, parameters) (C# SetParameters(double[], double[]), line 411):
//     weights + component-parameter slices (v2.1.4 added the validity recompute -- see the
//     v2.1.4 note above).
//   - set_parameters_normalized(parameters&) (C# SetParameters(ref double[]), line 476):
//     weights normalized to sum to 1 (or 1 - ZeroWeight) and written BACK into the passed
//     vector, single-component special case, then validity update.
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
// only -- existing behaviour and all existing fixtures stay byte-green (empirical_cdf_created_
// defaults false; the pre-existing root-find/bisection path is unchanged until CreateEmpiricalCDF()
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

    // Mirrors C# SetParameters(ref double[] parameters) (Mixture.cs line 476): weights are
    // normalized to sum to 1 (or to 1 - ZeroWeight when zero-inflated) and written BACK into
    // the passed vector -- the C# `ref` side effect the MixtureModel likelihood surface
    // depends on. Single-component special case: `parameters` carries only the component's
    // parameters and the weight is derived. Guard deviation (documented): the C# indexes an
    // undersized array and throws IndexOutOfRangeException; C++ makes that an explicit
    // std::invalid_argument instead of UB.
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
                    parameters[static_cast<std::size_t>(i)] = w;
                }
            } else {
                // Normalize weights to sum to 1.
                double c = is_zero_inflated_ ? (1.0 - zero_weight_) / sum : 1.0 / sum;
                for (int i = 0; i < K; ++i) {
                    weights_[static_cast<std::size_t>(i)] *= c;
                    parameters[static_cast<std::size_t>(i)] = weights_[static_cast<std::size_t>(i)];
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
        if (!moments_computed_) compute_moments();
        return u_[1];
    }
    double skewness() const override {
        if (!moments_computed_) compute_moments();
        return u_[2];
    }
    double kurtosis() const override {
        if (!moments_computed_) compute_moments();
        return u_[3];
    }
    // Minimum = min over components (mirrors C# Distributions.Min(p => p.Minimum)).
    double minimum() const override {
        double m = kInf;
        for (const auto& c : components_) m = std::min(m, c->minimum());
        return m;
    }
    // Maximum = max over components (mirrors C# Distributions.Max(p => p.Maximum)).
    double maximum() const override {
        double m = -kInf;
        for (const auto& c : components_) m = std::max(m, c->maximum());
        return m;
    }

    // --- Distribution functions ---
    // Mirrors C# PDF: f = Σ wᵢ fᵢ(x); clamp to [0, ∞).
    double pdf(double x) const override {
        double f = 0.0;
        if (is_zero_inflated_ && x <= 0.0) {
            f = zero_weight_;
        } else {
            for (int i = 0; i < static_cast<int>(components_.size()); ++i)
                f += weights_[i] * components_[i]->pdf(x);
        }
        return f < 0.0 ? 0.0 : f;
    }

    // Mirrors C# LogPDF: log-sum-exp over log(wᵢ) + log fᵢ(x).
    double log_pdf(double x) const override {
        std::vector<double> lnf;
        if (is_zero_inflated_ && x <= 0.0) {
            lnf.push_back(std::log(zero_weight_));
        } else {
            for (int i = 0; i < static_cast<int>(components_.size()); ++i)
                lnf.push_back(std::log(weights_[i]) + components_[i]->log_pdf(x));
        }
        return log_sum_exp(lnf);
    }

    // Mirrors C# CDF: F = Σ wᵢ Fᵢ(x); clamped to [0,1].
    double cdf(double x) const override {
        double F = 0.0;
        if (is_zero_inflated_) {
            F = zero_weight_;
            if (x > 0.0) {
                for (int i = 0; i < static_cast<int>(components_.size()); ++i)
                    F += weights_[i] * components_[i]->cdf(x);
            }
        } else {
            for (int i = 0; i < static_cast<int>(components_.size()); ++i)
                F += weights_[i] * components_[i]->cdf(x);
        }
        return F < 0.0 ? 0.0 : F > 1.0 ? 1.0 : F;
    }

    // Mirrors C# InverseCDF: Brent solve on CDF(y) - probability = 0.
    // Bracket is derived from per-component InverseCDF values.
    // On bracket / solve failure, falls back to bisection on [min, max] (C# falls back
    // to EmpiricalDistribution; we skip that dependency — noted in header comment).
    double inverse_cdf(double probability) const override {
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

        // Derive bracket from component InverseCDF values (mirrors C# logic).
        double adj_prob = is_zero_inflated_
            ? (probability - zero_weight_) / (1.0 - zero_weight_)
            : probability;
        double minX = kInf, maxX = -kInf;
        for (const auto& c : components_) {
            minX = std::min(minX, c->inverse_cdf(std::max(adj_prob, 1e-15)));
            maxX = std::max(maxX, c->inverse_cdf(std::min(probability, 1.0 - 1e-15)));
        }
        // Expand bracket if needed (mirrors Brent.Bracket behavior for zero-inflated).
        if (is_zero_inflated_) {
            // Expand bracket until CDF(lo) < probability < CDF(hi)
            int attempts = 0;
            while (cdf(minX) >= probability && attempts++ < 100) minX -= 1.0;
            attempts = 0;
            while (cdf(maxX) <= probability && attempts++ < 100) maxX += 1.0;
        }
        double x = 0.0;
        try {
            auto fn = [this, probability](double y) { return probability - cdf(y); };
            x = math::rootfinding::solve(fn, minX, maxX, 1E-6, 100, true);
        } catch (...) {
            // Bisection fallback over [minimum, maximum].
            double lo = minimum(), hi = maximum();
            if (std::isinf(lo)) lo = minX - 50.0;
            if (std::isinf(hi)) hi = maxX + 50.0;
            for (int i = 0; i < 200; ++i) {
                double mid = 0.5 * (lo + hi);
                if (cdf(mid) < probability) lo = mid; else hi = mid;
            }
            x = 0.5 * (lo + hi);
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
        double minX = kInf, maxX = -kInf;
        for (const auto& c : components_) {
            minX = std::min(minX, c->inverse_cdf(min_p));
            maxX = std::max(maxX, c->inverse_cdf(max_p));
        }
        double shift = 0.0;
        if (minX <= 0.0) shift = std::fabs(minX) + 1.0;
        double mn = minX + shift;
        double mx = maxX + shift;
        int order = static_cast<int>(std::floor(std::log10(mx) - std::log10(mn)));
        int binN = std::max(200, 100 * order) - 1;

        auto bins = sampling::Stratify::XValues(
            sampling::StratificationOptions(minX, maxX, binN, false),
            x_transform == data::Transform::Logarithmic);
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
        if (is_zero_inflated_ && (zero_weight_ < 0.0 || zero_weight_ > 1.0)) return false;
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
        auto mom = central_moments(1000);
        u_[0] = mom[0];
        u_[1] = mom[1];
        u_[2] = mom[2];
        u_[3] = mom[3];
        moments_computed_ = true;
    }

    // EM-based MLE. Mirrors C# MLE() method.
    std::vector<double> mle(const std::vector<double>& sample) const {
        int N = static_cast<int>(sample.size());
        int K = static_cast<int>(components_.size());
        // Total component parameters (excluding weights).
        int Np = 0;
        for (const auto& c : components_) Np += c->number_of_parameters();

        // Get parameter constraints for each component: call estimate to get initials,
        // use wide bounds (C# calls IMaximumLikelihoodEstimation.GetParameterConstraints;
        // that interface is not ported -- functional divergence for bounds only).
        std::vector<double> initials, lowers, uppers;
        {
            auto tmp_stats = data::product_moments(sample);
            double sample_mean = tmp_stats[0];
            double sample_sd = std::max(tmp_stats[1], 1e-10);
            for (int i = 0; i < K; ++i) {
                auto comp_clone = components_[i]->clone();
                // Try to get initial values via estimate.
                auto* est = dynamic_cast<IEstimation*>(comp_clone.get());
                if (est) {
                    try { est->estimate(sample, ParameterEstimationMethod::MaximumLikelihood); }
                    catch (...) { comp_clone = components_[i]->clone(); }
                }
                auto p = comp_clone->get_parameters();
                int n = static_cast<int>(p.size());
                for (int j = 0; j < n; ++j) {
                    double v = std::isfinite(p[j]) && std::fabs(p[j]) > 1e-10 ? p[j] : sample_mean;
                    initials.push_back(v);
                    // Wide bounds: ±max(100*|v|, 1000*sample_sd)
                    double range = std::max(100.0 * std::fabs(v), 100.0 * sample_sd);
                    // LIMITATION: coarse lower-bound heuristic. Only the first parameter of
                    // each component gets a symmetric bound; params j>0 are floored at 1e-15,
                    // which wrongly forbids legitimately-negative non-first parameters (e.g. a
                    // GEV/GeneralizedLogistic shape kappa). The correct per-parameter bounds
                    // need IMaximumLikelihoodEstimation.GetParameterConstraints, which is not
                    // ported. The composite MLE path is not oracle-covered (C# fits seeded-RNG
                    // samples that cannot be transcribed), so this is a documented, untested
                    // deferral; all current mixture fixtures use Normal-only components.
                    lowers.push_back(j == 0 ? v - range : 1e-15);
                    uppers.push_back(v + range);
                }
            }
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
            for (int k = 0; k < K; ++k) {
                for (int i = 0; i < N; ++i) {
                    if (is_zero_inflated_ && sample[i] <= 0.0) {
                        likelihood[i][k] = std::log(zero_weight_);
                    } else {
                        likelihood[i][k] = std::log(mle_weights[k])
                            + dist_ptr->components_[k]->log_pdf(sample[i]);
                    }
                }
            }
            // Log-sum-exp normalization per sample point.
            double logLH = 0.0;
            for (int i = 0; i < N; ++i) {
                double max_val = -std::numeric_limits<double>::infinity();
                for (int k = 0; k < K; ++k)
                    if (likelihood[i][k] > max_val) max_val = likelihood[i][k];
                if (std::isinf(max_val)) {
                    for (int k = 0; k < K; ++k) likelihood[i][k] = 0.0;
                    return -std::numeric_limits<double>::infinity();
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
                mle_weights[k] = wgt / N;
            }
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
