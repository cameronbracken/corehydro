// ported from: Numerics/Distributions/Univariate/CompetingRisks.cs @ 2a0357a
//
// Competing risks (series / parallel system) distribution.
// Constructor: CompetingRisks(vector of component unique_ptrs).
// Two modes via minimum_of_random_variables (default true, mirrors C#):
//   true  → min-of-components: CDF = Union   = 1 − ∏ Si(x) = 1 − ∏(1−Fi(x))
//   false → max-of-components: CDF = Product = ∏ Fi(x)
// PDF (min-rule, Independent): f(x) = [Σ hᵢ(x)] · S(x) where hᵢ=fᵢ/Sᵢ, S=∏Sᵢ
// PDF (max-rule, Independent): f(x) = [Σ fᵢ/Fᵢ] · ∏Fᵢ
// LogPDF (Independent): numerically stable log-space versions using log-sum-exp.
// PDF/LogPDF for any non-Independent dependency: central-difference numerical derivative
//   of cdf() (mirrors C# NumericalDerivative.Derivative -- narrow port of just
//   CentralDifference + CalculateStepSize(order=1), see numerical_derivative() below).
// InverseCDF: Brent root-finding on CDF(y) − p = 0; bracket from per-component quantiles.
//   On solve failure, falls back to bisection on [minimum, maximum].
// Moments: central_moments(1000), the fixed-step trapezoidal overload the C# integer literal
//   in `CentralMoments(1000)` binds to (CompetingRisks.cs line 217). This file used to call
//   adaptive Gauss-Kronrod instead, which agreed with the C# only to about six digits.
// Mode: BrentSearch maximizing the PDF over [InverseCDF(0.001), InverseCDF(0.999)]
//   (CompetingRisks.cs line 241), replacing an earlier ternary search.
// IEstimation: MLE via Nelder-Mead on total log-likelihood (mirrors C# MLE()); initial
//   values from per-component estimate() if available, else sample statistics.
//   Only MaximumLikelihood is supported; others throw (mirrors C# Estimate()).
//
// DEPENDENCY MODES (this task): all four Probability.DependencyType modes are now
// supported -- `dependency` (mirrors the C# `Dependency` property) plus
// `correlation_matrix()`/`set_correlation_matrix()` (mirrors `CorrelationMatrix`, whose
// setter side effect -- invalidating the lazily-cached MVN -- is preserved). CDF/PDF for
// Independent and PerfectlyPositive route through probability::union_probability /
// probability::joint_probability (the 2-arg dependency-dispatch overloads, no
// correlation matrix needed). PerfectlyNegative and CorrelationMatrix lazily build a
// MultivariateNormal via create_multivariate_normal() (mirrors C#
// CreateMultivariateNormal()) purely to hold a mu/sigma-validated covariance matrix, then
// route through probability::union_pcm / probability::joint_probability(..., &cov) (the
// HPCM engine). See probability.hpp's header comment for the full trace showing this path
// is deterministic (no RNG) regardless of component count. Only 2- and 3-component
// configurations are fixture-covered (see fixtures/distributions/univariate/
// competing_risks.json); the algorithm itself is a verbatim, dimension-general port.
// GenerateRandomValues: ported override (C# line 1095) -- per-component inverse-CDF draws,
//   min/max per sample (see the method comment; GenerateRandomValuesWithDependency is not
//   ported, no ported caller).
// type() returns UnivariateDistributionType::CompetingRisks (mirrors C#).
// Not wired into the flat factory (composite-only, like Mixture).
//
// X5 ADDITIVE PORT (CompetingRisks.cs XTransform/ProbabilityTransform/CreateEmpiricalCDF @ 2a0357a):
// the CompositeAnalysis aggregation builds a CompetingRisks per posterior realisation, sets
// XTransform = Logarithmic / ProbabilityTransform = NormalZ, and calls CreateEmpiricalCDF() so the
// InverseCDF the UncertaintyAnalysisResults reads is the fast piecewise empirical curve. These are
// NEW methods/fields only -- existing behaviour and all existing fixtures stay byte-green because
// empirical_cdf_created_ defaults false (the pre-existing root-find/bisection path is unchanged
// until CreateEmpiricalCDF() is explicitly called).
//
// v2.1.4 (this task): `Dependency` gained a side-effecting setter (invalidates the cached
// MVN when the dependency mode actually changes) -- `dependency` is now a private field
// with `dependency()`/`set_dependency()` accessors instead of the plain public field the
// prior (no-side-effect) C# auto-property justified. `PerfectlyNegative`'s
// create_multivariate_normal() branch no longer zeroes the public correlation_matrix() as
// a side effect (RESOLVED, see docs/upstream-csharp-issues.md). The flat set_parameters()
// now throws on a flattened-length mismatch (mirrors the new C# `ArgumentException`) and a
// new public validate_parameters(parameters, throw_exception) mirrors C#
// ValidateParameters's throwException contract for the empty-Distributions case (not
// reachable via any current call site, since this class always holds >= 1 component once
// constructed -- ported for structural fidelity per the task brief).
#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/data/probability.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/sampling/stratify.hpp"
#include "corehydro/numerics/sampling/stratification_options.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace prob = corehydro::numerics::data::probability;

class CompetingRisks : public UnivariateDistributionBase, public IEstimation {
   public:
    // Construct from already-created component unique_ptrs (takes ownership).
    explicit CompetingRisks(
        std::vector<std::unique_ptr<UnivariateDistributionBase>> components)
        : components_(std::move(components)) {
        validate_and_set();
    }

    // Convenience: construct by cloning components from raw pointers.
    explicit CompetingRisks(const std::vector<UnivariateDistributionBase*>& components) {
        components_.reserve(components.size());
        for (auto* c : components) components_.push_back(c->clone());
        validate_and_set();
    }

    // Whether the composite is min-of-components (true, default) or max-of-components (false).
    // Mirrors C# MinimumOfRandomVariables property.
    bool minimum_of_random_variables = true;

    // The dependency between the marginal distributions. v2.1.4: Dependency gained a
    // side-effecting setter (changing the mode invalidates the cached MVN), so this is now
    // a private field with accessors rather than the plain public field the prior
    // no-side-effect C# auto-property justified. Mirrors C# `Dependency { get; set; }`.
    prob::DependencyType dependency() const { return dependency_; }
    void set_dependency(prob::DependencyType value) {
        if (dependency_ != value) {
            dependency_ = value;
            mvn_created_ = false;
        }
    }

    // X5: the x-value / probability transforms the empirical CDF interpolates in (mirrors C#
    // XTransform / ProbabilityTransform; defaults None / NormalZ). Plain public fields, matching
    // the C# auto-properties and the convention above.
    data::Transform x_transform = data::Transform::None;
    data::Transform probability_transform = data::Transform::NormalZ;

    // The correlation matrix used for modeling dependency between the marginal
    // distributions. Only used when dependency == CorrelationMatrix. Mirrors C#
    // CorrelationMatrix property.
    const prob::Matrix2D& correlation_matrix() const { return correlation_matrix_; }
    void set_correlation_matrix(prob::Matrix2D m) {
        correlation_matrix_ = std::move(m);
        mvn_created_ = false;
    }

    // Accessors.
    int component_count() const { return static_cast<int>(components_.size()); }
    const UnivariateDistributionBase& component(int i) const { return *components_[i]; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::CompetingRisks;
    }

    // Sum of each component's NumberOfParameters (mirrors C#).
    int number_of_parameters() const override {
        int sum = 0;
        for (const auto& c : components_) sum += c->number_of_parameters();
        return sum;
    }

    // [p0_0, p0_1, ..., p1_0, p1_1, ...] (mirrors C# GetParameters).
    std::vector<double> get_parameters() const override {
        std::vector<double> result;
        for (const auto& c : components_) {
            auto p = c->get_parameters();
            result.insert(result.end(), p.begin(), p.end());
        }
        return result;
    }

    // Mirrors C# SetParameters(IList<double>): distribute flat array to components in
    // order. v2.1.4: throws on a flattened-length mismatch (mirrors the new C#
    // `ArgumentException`); an empty component list marks parameters invalid without
    // throwing (mirrors C#'s early `_parametersValid = false; return;`).
    void set_parameters(const std::vector<double>& parameters) override {
        if (components_.empty()) {
            parameters_valid_ = false;
            return;
        }
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters())) {
            throw std::invalid_argument("The length of the parameter array is invalid.");
        }
        int t = 0;
        for (const auto& c : components_) {
            int n = c->number_of_parameters();
            std::vector<double> p(parameters.begin() + t, parameters.begin() + t + n);
            c->set_parameters(p);
            t += n;
        }
        parameters_valid_ = validate_components();
        moments_computed_ = false;
        mvn_created_ = false;
    }

    // Validates the flat parameter vector: returns nullopt if valid, else an error
    // message. Mirrors C# `ValidateParameters(IList<double>, bool throwException)` --
    // note the `parameters` argument itself is not consulted beyond its role in the
    // caller's length check (matches C#: only Distributions.Count and each component's
    // ParametersValid matter here). Not reachable via any current call site (this class
    // always holds >= 1 component once constructed), ported for structural fidelity.
    std::optional<std::string> validate_parameters(const std::vector<double>& /*parameters*/,
                                                     bool throw_exception) const {
        if (components_.empty()) {
            if (throw_exception) throw std::out_of_range("There must be at least 1 distribution.");
            return "There must be at least 1 distribution.";
        }
        for (const auto& c : components_) {
            if (!c->parameters_valid()) {
                if (throw_exception)
                    throw std::out_of_range("One of the distributions have invalid parameters.");
                return "One of the distributions have invalid parameters.";
            }
        }
        return std::nullopt;
    }

    // --- Moments / support ---
    double mean() const override {
        if (!moments_computed_) compute_moments();
        return u_[0];
    }
    double median() const override { return inverse_cdf(0.5); }
    double mode() const override {
        // Mirrors C# Mode (CompetingRisks.cs line 241): BrentSearch maximizing the PDF over
        // [InverseCDF(0.001), InverseCDF(0.999)], returning BestParameterSet.Values[0].
        // This used to ternary-search the same interval, which lands ~2e-8 away.
        double a = inverse_cdf(0.001);
        double b = inverse_cdf(0.999);
        math::optimization::BrentSearch brent([this](double x) { return pdf(x); }, a, b);
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
    // Mirrors C# PDF: exact for Independent; numerical derivative of cdf() for every
    // other dependency mode (PerfectlyPositive/PerfectlyNegative/CorrelationMatrix).
    // Returns at least kMinDensity to prevent log-likelihood issues (mirrors C#).
    double pdf(double x) const override {
        if (components_.size() == 1) return components_[0]->pdf(x);
        double f;
        if (dependency_ == prob::DependencyType::Independent) {
            f = minimum_of_random_variables ? pdf_min_independent(x) : pdf_max_independent(x);
        } else {
            f = numerical_derivative(x);
        }
        return f < kMinDensity ? kMinDensity : f;
    }

    // Mirrors C# LogPDF: numerically stable log-space computation for Independent; falls
    // back to log(numerical derivative of cdf()) for every other dependency mode.
    double log_pdf(double x) const override {
        if (components_.size() == 1) return components_[0]->log_pdf(x);
        if (dependency_ == prob::DependencyType::Independent) {
            return minimum_of_random_variables ? log_pdf_min_independent(x)
                                                : log_pdf_max_independent(x);
        }
        double pdf_val = numerical_derivative(x);
        return pdf_val > kMinDensity ? std::log(pdf_val) : -std::numeric_limits<double>::infinity();
    }

    // Mirrors C# CDF verbatim, including its dependency-mode dispatch:
    //   min-rule, Independent/PerfectlyPositive: Probability.Union
    //   min-rule, PerfectlyNegative/CorrelationMatrix: Probability.UnionPCM(cov)
    //   max-rule, Independent/PerfectlyPositive: Probability.JointProbability(dependency)
    //   max-rule, PerfectlyNegative/CorrelationMatrix: Probability.JointProbability(ind, cov)
    // See probability.hpp's header comment: the PerfectlyNegative/CorrelationMatrix paths
    // route through JointProbabilityHPCM (deterministic, no RNG), never MultivariateNormal
    // .CDF()'s seeded Genz-Bretz integrator.
    double cdf(double x) const override {
        if (components_.size() == 1) return components_[0]->cdf(x);
        std::size_t n = components_.size();
        std::vector<int> ind(n, 1);
        std::vector<double> cdf_vals(n);
        for (std::size_t i = 0; i < n; ++i) cdf_vals[i] = components_[i]->cdf(x);

        bool correlated = dependency_ == prob::DependencyType::PerfectlyNegative ||
                           dependency_ == prob::DependencyType::CorrelationMatrix;
        double p;
        if (minimum_of_random_variables) {
            if (correlated) {
                if (!mvn_created_) create_multivariate_normal();
                prob::Matrix2D cov = mvn_->covariance();
                p = prob::union_pcm(cdf_vals, cov);
            } else {
                p = prob::union_probability(cdf_vals, dependency_);
            }
        } else {
            if (correlated) {
                if (!mvn_created_) create_multivariate_normal();
                prob::Matrix2D cov = mvn_->covariance();
                p = prob::joint_probability(cdf_vals, ind, &cov);
            } else {
                p = prob::joint_probability(cdf_vals, dependency_);
            }
        }
        return p < 0.0 ? 0.0 : p > 1.0 ? 1.0 : p;
    }

    // Mirrors C# InverseCDF: Brent root-finding on CDF(y) − probability = 0.
    // Bracket from per-component quantile values. Falls back to bisection on failure.
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (components_.size() == 1) return components_[0]->inverse_cdf(probability);

        // X5: once CreateEmpiricalCDF() has been called (composite aggregation path), read the
        // fast piecewise empirical curve, mirroring C# `if (_empiricalCDFCreated) x =
        // _empiricalCDF.InverseCDF(probability)`.
        if (empirical_cdf_created_) {
            double xe = empirical_cdf_->inverse_cdf(probability);
            double mn0 = minimum(), mx0 = maximum();
            return xe < mn0 ? mn0 : xe > mx0 ? mx0 : xe;
        }

        // Build bracket from per-component InverseCDF values (mirrors C#).
        double minX = kInf, maxX = -kInf;
        for (const auto& c : components_) {
            double q = c->inverse_cdf(probability);
            minX = std::min(minX, q);
            maxX = std::max(maxX, q);
        }
        if (minX >= maxX) maxX = minX + 1.0;

        double x = 0.0;
        try {
            auto fn = [this, probability](double y) { return probability - cdf(y); };
            // Expand bracket if not straddling zero (mirrors C# Brent.Bracket).
            int expand_attempts = 0;
            while (fn(minX) * fn(maxX) > 0.0 && expand_attempts++ < 200) {
                double mid = 0.5 * (minX + maxX);
                double half_range = 1.5 * (maxX - minX);
                minX = mid - half_range;
                maxX = mid + half_range;
            }
            x = math::rootfinding::solve(fn, minX, maxX, 1E-6, 100, true);
        } catch (...) {
            // Bisection fallback (mirrors C# fall-back to EmpiricalDistribution;
            // we bisect instead — noted in the header comment).
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
    // Mirrors C# Estimate: only MaximumLikelihood supported.
    void estimate(const std::vector<double>& sample,
                  ParameterEstimationMethod method) override {
        if (method != ParameterEstimationMethod::MaximumLikelihood)
            throw std::runtime_error("CompetingRisks only supports MaximumLikelihood estimation");
        set_parameters(mle(sample));
    }

    // Mirrors C# GenerateRandomValues(int sampleSize, int seed = -1) override (CompetingRisks
    // .cs line 1095, added for the M11 CompetingRisksModel which delegates here): for each
    // sample, draw one value from EVERY component (each draw consumes one uniform from the
    // shared Mersenne Twister stream, j-major within each i), then take the minimum (min-rule)
    // or maximum (max-rule). NOT the base class's composite inverse-CDF sampler. The
    // dependency-aware variant `GenerateRandomValuesWithDependency` (C# line 1124) is not
    // ported -- no ported caller uses it.
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        // Create PRNG for generating random numbers.
        sampling::MersenneTwister rnd =
            seed > 0 ? sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                     : sampling::MersenneTwister();
        std::vector<double> sample(static_cast<std::size_t>(sample_size));
        // Generate values.
        for (int i = 0; i < sample_size; ++i) {
            double x_min = std::numeric_limits<double>::max();
            double x_max = std::numeric_limits<double>::lowest();  // C# double.MinValue
            for (std::size_t j = 0; j < components_.size(); ++j) {
                double x = components_[j]->inverse_cdf(rnd.next_double());
                if (x < x_min) x_min = x;
                if (x > x_max) x_max = x;
            }
            sample[static_cast<std::size_t>(i)] =
                minimum_of_random_variables ? x_min : x_max;
        }
        // Return array of random values.
        return sample;
    }

    // --- Parameter display names (X1; C# CompetingRisks.cs ParameterNames /
    // ParameterNamesShortForm). BOTH are OVERRIDDEN dynamically in C# (148-180): they are NOT the
    // ParametersToString col-0 single "Distributions" entry, but a per-component list built as
    // "D{i+1} {sub}" over every component's own names (long form) / short form (C# 148-180). ---
    std::vector<std::string> parameter_names() const override {
        std::vector<std::string> result;
        for (int i = 0; i < component_count(); ++i) {
            std::vector<std::string> sub = component(i).parameter_names();
            for (const std::string& s : sub)
                result.push_back("D" + std::to_string(i + 1) + " " + s);
        }
        return result;
    }
    std::vector<std::string> parameter_names_short_form() const override {
        std::vector<std::string> result;
        for (int i = 0; i < component_count(); ++i) {
            std::vector<std::string> sub = component(i).parameter_names_short_form();
            for (const std::string& s : sub)
                result.push_back("D" + std::to_string(i + 1) + " " + s);
        }
        return result;
    }

    // X5: builds the piecewise EmpiricalDistribution backing the composite InverseCDF (mirrors C#
    // CompetingRisks.CreateEmpiricalCDF, CompetingRisks.cs:1050). Verbatim port: per-component
    // 1e-16/1-1e-16 quantile range, positive shift, a log10-spread bin count, then a Stratify grid
    // whose LowerBounds seed strictly-increasing (x, CDF(x)) pairs, capped by maxX. The backing
    // EmpiricalDistribution carries this XTransform / ProbabilityTransform.
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

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        std::vector<std::unique_ptr<UnivariateDistributionBase>> cloned;
        cloned.reserve(components_.size());
        for (const auto& c : components_) cloned.push_back(c->clone());
        auto cr = std::make_unique<CompetingRisks>(std::move(cloned));
        cr->minimum_of_random_variables = minimum_of_random_variables;
        cr->set_dependency(dependency_);
        cr->x_transform = x_transform;
        cr->probability_transform = probability_transform;
        // Mirrors C# Clone(): `if (CorrelationMatrix != null) cr.CorrelationMatrix = ...`.
        if (!correlation_matrix_.empty()) cr->set_correlation_matrix(correlation_matrix_);
        return cr;
    }

   private:
    std::vector<std::unique_ptr<UnivariateDistributionBase>> components_;

    // Backing field for dependency()/set_dependency() (v2.1.4 -- see the accessor comment).
    prob::DependencyType dependency_ = prob::DependencyType::Independent;

    // X5: lazily-built empirical CDF backing (mirrors C# _empiricalCDF / _empiricalCDFCreated).
    std::unique_ptr<EmpiricalDistribution> empirical_cdf_;
    bool empirical_cdf_created_ = false;

    // Lazy moment cache.
    mutable bool moments_computed_ = false;
    mutable double u_[4] = {kNaN, kNaN, kNaN, kNaN};  // [mean, sd, skewness, kurtosis]

    // User-supplied correlation matrix (CorrelationMatrix dependency mode) and the lazily-
    // built MultivariateNormal cache (mirrors C# _correlationMatrix/_mvnCreated/_mvn).
    // `mutable` because create_multivariate_normal() is called lazily from const cdf().
    mutable prob::Matrix2D correlation_matrix_;
    mutable bool mvn_created_ = false;
    mutable std::unique_ptr<MultivariateNormal> mvn_;

    // Soft floor used in tail arithmetic (mirrors C# _minDensity / _logZero).
    static constexpr double kMinDensity = 1E-300;
    static constexpr double kLogZero = -745.0;

    void validate_and_set() {
        parameters_valid_ = !components_.empty() && validate_components();
        moments_computed_ = false;
    }

    // Central-difference numerical derivative of cdf() at x. Narrow port of C#
    // NumericalDerivative.Derivative(f, point) = CentralDifference(f, point,
    // CalculateStepSize(point, order=1)), the only NumericalDerivative member
    // CompetingRisks.cs calls (via `NumericalDerivative.Derivative(CDF, x)`).
    double numerical_derivative(double x) const {
        double h = std::sqrt(kDoubleMachineEpsilon) * (1.0 + std::fabs(x));
        return (cdf(x + h) - cdf(x - h)) / (2.0 * h);
    }

    // Lazily builds the MultivariateNormal instance used to hold the correlation matrix
    // for PerfectlyNegative/CorrelationMatrix dependency. Mirrors C#
    // CreateMultivariateNormal(). v2.1.4 (RESOLVED, see docs/upstream-csharp-issues.md):
    // the PerfectlyNegative branch no longer overwrites the public correlation_matrix()
    // with a fresh all-zero D x D matrix as a side effect -- the synthetic rho matrix
    // passed to the MVN is built entirely into the local `sigma` array below, and
    // correlation_matrix() now reads back whatever the caller last set (or empty),
    // unmutated by a CDF/PDF evaluation.
    void create_multivariate_normal() const {
        int D = component_count();
        std::vector<double> mu(static_cast<std::size_t>(D), 0.0);
        prob::Matrix2D sigma(static_cast<std::size_t>(D), std::vector<double>(static_cast<std::size_t>(D), 0.0));
        if (dependency_ == prob::DependencyType::PerfectlyNegative) {
            double rho = -1.0 / (D - 1.0) + std::sqrt(kDoubleMachineEpsilon);
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j)
                    sigma[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = (i == j) ? 1.0 : rho;
        } else {
            // Divergence note: C#'s equivalent loop indexes `CorrelationMatrix[i, j]`
            // directly. If a caller sets dependency = CorrelationMatrix without ever
            // calling the `CorrelationMatrix` setter (the field defaults to null), C#
            // throws a catchable `NullReferenceException` on first access; if a caller
            // supplies a matrix sized for a different component count, C# throws
            // `IndexOutOfRangeException`. `std::vector::operator[]` performs no bounds
            // check, so a verbatim transcription would be undefined behavior (an
            // out-of-bounds read, observed to crash the process) in the same misuse
            // scenario instead of a catchable exception. The guard below throws
            // `std::out_of_range` before the first out-of-range access, reproducing the
            // C# observable behavior (a thrown exception on misuse) without restructuring
            // the loop.
            if (correlation_matrix_.size() != static_cast<std::size_t>(D)) {
                throw std::out_of_range(
                    "CorrelationMatrix must be set and sized to match the number of "
                    "components before using dependency = CorrelationMatrix.");
            }
            for (int i = 0; i < D; ++i) {
                if (correlation_matrix_[static_cast<std::size_t>(i)].size() != static_cast<std::size_t>(D)) {
                    throw std::out_of_range(
                        "CorrelationMatrix must be set and sized to match the number of "
                        "components before using dependency = CorrelationMatrix.");
                }
            }
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j)
                    sigma[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                        correlation_matrix_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
        mvn_ = std::make_unique<MultivariateNormal>(std::move(mu), std::move(sigma));
        mvn_created_ = true;
    }

    bool validate_components() const {
        for (const auto& c : components_)
            if (!c->parameters_valid()) return false;
        return true;
    }

    // log-sum-exp trick: log(Σ exp(v[i])).
    static double log_sum_exp(const std::vector<double>& v) {
        if (v.empty()) return -std::numeric_limits<double>::infinity();
        double max_val = *std::max_element(v.begin(), v.end());
        if (std::isinf(max_val)) return max_val;
        double sum = 0.0;
        for (double x : v) sum += std::exp(x - max_val);
        return max_val + std::log(sum);
    }

    // PDF for min-rule (Independent): f(x) = [Σ hᵢ(x)] · S(x).
    // Mirrors C# PDFMinimumIndependent.
    double pdf_min_independent(double x) const {
        double sum_hazard = 0.0;
        double product_survival = 1.0;
        for (const auto& c : components_) {
            double si = 1.0 - c->cdf(x);  // CCDF / survival
            double fi = c->pdf(x);
            product_survival *= si;
            if (si > kMinDensity) {
                sum_hazard += fi / si;
            } else if (fi > kMinDensity) {
                sum_hazard += fi / kMinDensity;
            }
        }
        return sum_hazard * product_survival;
    }

    // PDF for max-rule (Independent): f(x) = [Σ fᵢ/Fᵢ] · ∏Fᵢ.
    // Mirrors C# PDFMaximumIndependent.
    double pdf_max_independent(double x) const {
        double sum_ratio = 0.0;
        double product_cdf = 1.0;
        for (const auto& c : components_) {
            double fi = c->cdf(x);
            double pi = c->pdf(x);
            product_cdf *= fi;
            if (fi > kMinDensity) {
                sum_ratio += pi / fi;
            } else if (pi > kMinDensity) {
                sum_ratio += pi / kMinDensity;
            }
        }
        return sum_ratio * product_cdf;
    }

    // Log-PDF for min-rule. Mirrors C# LogPDFMinimumIndependent.
    double log_pdf_min_independent(double x) const {
        int n = component_count();
        std::vector<double> log_survival(n), log_hazard(n);
        double sum_log_survival = 0.0;
        bool all_survival_zero = true;

        for (int i = 0; i < n; ++i) {
            double si = 1.0 - components_[i]->cdf(x);
            double fi = components_[i]->pdf(x);

            if (si > kMinDensity) {
                log_survival[i] = std::log(si);
                all_survival_zero = false;
            } else {
                log_survival[i] = kLogZero;
            }
            sum_log_survival += log_survival[i];

            if (fi > kMinDensity && si > kMinDensity) {
                log_hazard[i] = std::log(fi) - std::log(si);
            } else if (fi <= kMinDensity) {
                log_hazard[i] = kLogZero;
            } else {
                log_hazard[i] = std::log(fi) - kLogZero;
            }
        }

        if (all_survival_zero) return kLogZero;
        double log_sum_h = log_sum_exp(log_hazard);
        double result = log_sum_h + sum_log_survival;
        return (std::isnan(result) || std::isinf(result)) ? kLogZero : result;
    }

    // Log-PDF for max-rule. Mirrors C# LogPDFMaximumIndependent.
    double log_pdf_max_independent(double x) const {
        int n = component_count();
        std::vector<double> log_cdf_v(n), log_ratio(n);
        double sum_log_cdf = 0.0;
        bool all_cdf_zero = true;

        for (int i = 0; i < n; ++i) {
            double fi = components_[i]->cdf(x);
            double pi = components_[i]->pdf(x);

            if (fi > kMinDensity) {
                log_cdf_v[i] = std::log(fi);
                all_cdf_zero = false;
            } else {
                log_cdf_v[i] = kLogZero;
            }
            sum_log_cdf += log_cdf_v[i];

            if (pi > kMinDensity && fi > kMinDensity) {
                log_ratio[i] = std::log(pi) - std::log(fi);
            } else if (pi <= kMinDensity) {
                log_ratio[i] = kLogZero;
            } else {
                log_ratio[i] = std::log(pi) - kLogZero;
            }
        }

        if (all_cdf_zero) return kLogZero;
        double log_sum_r = log_sum_exp(log_ratio);
        double result = log_sum_r + sum_log_cdf;
        return (std::isnan(result) || std::isinf(result)) ? kLogZero : result;
    }

    // Mirrors C# ComputeMoments() (CompetingRisks.cs line 217): CentralMoments(1000). The
    // integer literal binds to the base class's fixed-step TRAPEZOIDAL overload
    // (`CentralMoments(int steps)`), not the adaptive-tolerance one -- see
    // docs/upstream-csharp-issues.md. This file used to call adaptive Gauss-Kronrod here,
    // which agreed with C# only to about six digits; the 1000-step trapezoid reproduces it
    // to ~1e-12 relative.
    void compute_moments() const {
        auto mom = central_moments(1000);
        u_[0] = mom[0];
        u_[1] = mom[1];
        u_[2] = mom[2];
        u_[3] = mom[3];
        moments_computed_ = true;
    }

    // Nelder-Mead MLE. Mirrors C# MLE() method.
    std::vector<double> mle(const std::vector<double>& sample) const {
        int K = static_cast<int>(components_.size());
        int Np = number_of_parameters();

        // Get initial values for each component via estimate() if available.
        std::vector<double> initials, lowers, uppers;
        {
            auto tmp_stats = data::product_moments(sample);
            double sample_mean = tmp_stats[0];
            double sample_sd = std::max(tmp_stats[1], 1e-10);
            for (int i = 0; i < K; ++i) {
                auto comp_clone = components_[i]->clone();
                auto* est = dynamic_cast<IEstimation*>(comp_clone.get());
                if (est) {
                    try { est->estimate(sample, ParameterEstimationMethod::MaximumLikelihood); }
                    catch (...) { comp_clone = components_[i]->clone(); }
                }
                auto p = comp_clone->get_parameters();
                int n_comp = static_cast<int>(p.size());
                for (int j = 0; j < n_comp; ++j) {
                    double v = std::isfinite(p[j]) && std::fabs(p[j]) > 1e-10 ? p[j] : sample_mean;
                    initials.push_back(v);
                    double range = std::max(100.0 * std::fabs(v), 100.0 * sample_sd);
                    // LIMITATION: every parameter is floored at 1e-15, which wrongly forbids
                    // legitimately-negative parameters (e.g. a Normal/Gumbel location). Correct
                    // per-parameter bounds need IMaximumLikelihoodEstimation.GetParameterConstraints
                    // (not ported). The composite MLE path is not oracle-covered (C# fits
                    // seeded-RNG samples that cannot be transcribed), so this is a documented,
                    // untested deferral; PDF/CDF/moments are exact mirrors of C#.
                    lowers.push_back(1e-15);
                    uppers.push_back(v + range);
                }
            }
        }

        // Nelder-Mead on total log-likelihood over all component parameters.
        auto log_lh_fn = [this, &sample](const std::vector<double>& x) -> double {
            auto dist_clone = static_cast<CompetingRisks*>(this->clone().release());
            std::unique_ptr<CompetingRisks> dc(dist_clone);
            dc->set_parameters(x);
            double lh = dc->log_likelihood(sample);
            if (std::isnan(lh) || std::isinf(lh)) return -std::numeric_limits<double>::infinity();
            return lh;
        };
        math::optimization::NelderMead solver(log_lh_fn, Np, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }
};

}  // namespace corehydro::numerics::distributions
