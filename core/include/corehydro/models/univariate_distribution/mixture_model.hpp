// ported from: src/RMC.BestFit/Models/UnivariateDistribution/MixtureModel.cs @ c2e6192
//
// Mixture model for univariate distributions, with optional zero inflation: a finite
// mixture of 1-3 component distributions wrapped around the ported Numerics `Mixture`
// (numerics/distributions/mixture.hpp), with Bayesian priors on parameters and a single
// quantile prior, a zero-inflated mass at zero, and an internal Expectation-Maximization
// fitter (NelderMead M-step) for approximate MLE. Derives from the M8
// `UnivariateDistributionModelBase` (DataFrame owner, Jeffreys toggle, IQuantilePriors
// state) and implements `ISimulatable<std::vector<double>>` (C# ISimulatable<double[]>) and
// `IUnivariateModel` (models/support/...). Sibling of the M9 UnivariateDistributionModel;
// mirrors the C# method-for-method.
//
// Nullability mapping:
//   - C# `Mixture?` is genuinely nullable ("Mixture = null" is a tested state), so the
//     wrapped distribution is a `std::unique_ptr<Mixture>` with nullptr == C# null;
//     `mixture()` returns the raw pointer (IUnivariateModel's `distribution()` convention).
//   - C# `DataFrame = null` maps to the base's never-set optional (see the base header).
//
// Numerics v2.2 retains `SetParameters(ref double[] parameters)` for binary compatibility but
// normalizes a private copy. The caller's optimizer coordinates therefore remain unchanged.
// The C++ port preserves that behavior in `Mixture::set_parameters_normalized`.
//
// EM covariance: the C# `out double[,] covariance` maps to an output-reference
// `numerics::math::linalg::Matrix&` (the type NumericalDiff::compute_hessian already
// returns); `out double[] parameters` / `out int iterations` map to output references.
//
// Clone(): C# aliases the DataFrame into `new MixtureModel(DataFrame, Mixture!)`; the
// value-typed move-only C++ frame is DEEP-COPIED instead (M9 precedent), and a model with
// no frame cannot be cloned (std::invalid_argument). Zero-inflation re-sync: the object
// initializer writes `_isZeroInflated` DIRECTLY to the field, and the ctor path's Mixture
// setter runs with the DEFAULT zero-inflation state, leaving the cloned Mixture's
// IsZeroInflated/ZeroWeight stomped to the ctor defaults (false/0) even when the original
// was zero-inflated. Originally an untested upstream inconsistency the C++ port corrected on
// its own (a documented deviation); BestFit v2.0.0 (commit b43943c "Fix BestFit port review
// findings") added the SAME two-line re-sync to C# itself
// (`result.Mixture!.IsZeroInflated = result._isZeroInflated; result.Mixture.ZeroWeight =
// result._isZeroInflated ? Mixture!.ZeroWeight : 0.0;`), so this is now a PLAIN PORT, not a
// divergence -- ported literally below, reading the model-level `is_zero_inflated_` flag
// (not the source mixture's own IsZeroInflated, which a caller could in principle have
// nudged out of sync via the mutable `mixture()` accessor) and conditionally zeroing
// ZeroWeight to match.
//
// SKIPPED (project-wide deferrals): the XElement ctor (C# line 108) and ToXElement (line
// 1302) -- XML serialization; INotifyPropertyChanged / RaisePropertyChange / event
// (un)subscription; the [Category]/[DisplayName]/[Description]/[Browsable] attributes.
// The C# DataFrame_PropertyChanged handler (line 346) ports as the explicit-invalidation
// data_frame_property_changed() override (the M4->M8 cadence; see the base header).
// ModelParameter.Name for component parameters: the C# reads
// `component.ParametersToString[j, 0]`; ParametersToString is not on the ported
// distribution base (Phase 4 decision, display-only), so component-parameter names stay ""
// while OwnerName ("D1", "D2", ...) and the weight names ("Weight (w<sub>)") port fully.
//
// EXCEPTION-TYPE MAPPING for THIS file: C# ArgumentNullException/ArgumentException ->
// std::invalid_argument; ArgumentOutOfRangeException -> std::out_of_range;
// InvalidOperationException -> std::runtime_error.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/estimation/numerical_diff.hpp"
#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/support/data_component.hpp"
#include "corehydro/models/support/i_univariate_model.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "corehydro/models/support/quantile_prior.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/support/subscript_formatter.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/models/univariate_distribution/base/univariate_distribution_model_base.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/mixture.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/math/integration/integration.hpp"
#include "corehydro/numerics/math/linalg/lu_decomposition.hpp"  // defines Matrix::inverse()
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class MixtureModel : public UnivariateDistributionModelBase,
                     public ISimulatable<std::vector<double>>,
                     public IUnivariateModel {
   public:
    using DistributionBase = numerics::distributions::UnivariateDistributionBase;
    using DistributionType = numerics::distributions::UnivariateDistributionType;
    using Mixture = numerics::distributions::Mixture;
    using Matrix = numerics::math::linalg::Matrix;

    // --- Construction (C# region, lines 36-159) --------------------------------------------

    // C# parameterless ctor (line 42): default Normal-Normal mixture and a single quantile
    // prior.
    MixtureModel() {
        use_single_quantile_ = true;
        set_default_mixture({DistributionType::Normal, DistributionType::Normal});
    }

    // C# `MixtureModel(DataFrame, Mixture)` (line 53). The C# clones the passed
    // distribution; so does this (the argument stays untouched).
    MixtureModel(DataFrame data_frame, const Mixture& distribution) {
        use_single_quantile_ = true;
        set_mixture(clone_mixture(distribution));
        set_data_frame(std::move(data_frame));
    }

    // C# `MixtureModel(DataFrame, List<UnivariateDistributionBase>, bool)` (line 69): uses
    // the user-provided component distributions directly (ownership transfers) with equal
    // weights. Throws std::invalid_argument (C# ArgumentException) on an empty list --
    // AFTER the DataFrame/IsZeroInflated assignments, matching the C# statement order.
    MixtureModel(DataFrame data_frame,
                 std::vector<std::unique_ptr<DistributionBase>> distributions,
                 bool is_zero_inflated = false) {
        use_single_quantile_ = true;
        set_data_frame(std::move(data_frame));
        set_is_zero_inflated(is_zero_inflated);

        if (distributions.empty())
            throw std::invalid_argument("At least one component distribution is required.");

        std::size_t k = distributions.size();
        std::vector<double> weights(k, 1.0 / static_cast<double>(k));

        // Use the user-provided distributions directly.
        set_mixture(std::make_unique<Mixture>(std::move(weights), std::move(distributions)));
    }

    // C# `MixtureModel(DataFrame, List<UnivariateDistributionType>, bool)` (line 94):
    // component distribution types with equal initial weights.
    MixtureModel(DataFrame data_frame, const std::vector<DistributionType>& distribution_types,
                 bool is_zero_inflated = false) {
        use_single_quantile_ = true;
        set_data_frame(std::move(data_frame));
        set_is_zero_inflated(is_zero_inflated);
        set_default_mixture(distribution_types);
    }

    // SKIPPED: the XElement ctor (C# line 108) -- XML serialization is a project-wide
    // deferral.

    // Move-only, like the base (deep copies go through clone()).
    MixtureModel(MixtureModel&&) = default;
    MixtureModel& operator=(MixtureModel&&) = default;

    // --- Members (C# region, lines 161-341) ------------------------------------------------

    // C# `IsSupportedDistributionType` (line 190): the 15-family whitelist (the C#
    // _supportedDistributionTypes HashSet, line 166).
    static bool is_supported_distribution_type(DistributionType distribution_type) {
        switch (distribution_type) {
            case DistributionType::Exponential:
            case DistributionType::GammaDistribution:
            case DistributionType::GeneralizedExtremeValue:
            case DistributionType::GeneralizedLogistic:
            case DistributionType::GeneralizedNormal:
            case DistributionType::GeneralizedPareto:
            case DistributionType::Gumbel:
            case DistributionType::KappaFour:
            case DistributionType::LnNormal:
            case DistributionType::Logistic:
            case DistributionType::LogNormal:
            case DistributionType::LogPearsonTypeIII:
            case DistributionType::Normal:
            case DistributionType::PearsonTypeIII:
            case DistributionType::Weibull:
                return true;
            default:
                return false;
        }
    }

    // C# `DataFrame` override (line 199): store, reprocess thresholds at the model boundary
    // (the M4->M8 cadence), sync the wrapped mixture's ZeroWeight to the frame, then
    // SetDefaultParameters when UseDefaultFlatPriors. (The C# null-assignment branch has no
    // C++ trigger -- see the base header's nullability note.)
    void set_data_frame(DataFrame data_frame) override {
        data_frame_ = std::move(data_frame);
        data_frame_->process_threshold_series();

        if (mixture_ != nullptr) {
            if (is_zero_inflated_) {
                mixture_->set_zero_weight(data_frame_->zero_value_relative_frequency());
            } else {
                mixture_->set_zero_weight(0.0);
            }
        }

        if (use_default_flat_priors()) set_default_parameters();
    }

    // IUnivariateModel's data_frame() accessors (the base's non-virtual accessors carry the
    // same unguarded-deref contract; check has_data_frame() first where the frame may be
    // absent).
    DataFrame& data_frame() override { return *data_frame_; }
    const DataFrame& data_frame() const override { return *data_frame_; }

    // C# `Mixture` property (line 242; nullable -- nullptr == C# null).
    bool has_mixture() const { return mixture_ != nullptr; }
    Mixture* mixture() { return mixture_.get(); }
    const Mixture* mixture() const { return mixture_.get(); }

    // C# setter (line 245): assign, sync IsZeroInflated/ZeroWeight onto the wrapped
    // distribution, then SetDefaultParameters + SetDefaultQuantilePriors (unconditionally).
    void set_mixture(std::unique_ptr<Mixture> mixture) {
        mixture_ = std::move(mixture);

        if (mixture_ != nullptr) {
            mixture_->set_is_zero_inflated(is_zero_inflated_);
            if (is_zero_inflated_) {
                mixture_->set_zero_weight(
                    has_data_frame() ? data_frame().zero_value_relative_frequency() : 0.0);
            } else {
                mixture_->set_zero_weight(0.0);
            }
        }

        set_default_parameters();
        set_default_quantile_priors();
    }

    // C# `IUnivariateModel.Distribution` explicit implementation (line 269): the wrapped
    // mixture, or nullptr when not set. Non-owning.
    const DistributionBase* distribution() const override { return mixture_.get(); }

    // C# `IsNonstationary => false` (line 276): mixture component distributions are treated
    // as stationary within this model.
    bool is_nonstationary() const override { return false; }

    // C# `IsZeroInflated` (line 285): whether a zero-inflated weight is assigned to values
    // <= 0. The setter propagates to the Numerics Mixture and resets the default parameters.
    bool is_zero_inflated() const { return is_zero_inflated_; }
    void set_is_zero_inflated(bool value) {
        if (is_zero_inflated_ == value) return;
        is_zero_inflated_ = value;

        if (mixture_ != nullptr) {
            mixture_->set_is_zero_inflated(is_zero_inflated_);
            if (is_zero_inflated_) {
                mixture_->set_zero_weight(
                    has_data_frame() ? data_frame().zero_value_relative_frequency() : 0.0);
            } else {
                mixture_->set_zero_weight(0.0);
            }
        }

        if (use_default_flat_priors_) set_default_parameters();
    }

    // C# `UseSingleQuantile` override (line 322): the mixture model supports only a single
    // quantile prior; attempts to disable are ignored.
    bool use_single_quantile() const override { return true; }
    void set_use_single_quantile(bool value) override {
        // Mixture model is restricted to a single quantile prior.
        if (!value) return;

        if (!use_single_quantile_) {
            use_single_quantile_ = true;
            set_default_quantile_priors();
        }
    }

    // --- Methods (C# region, lines 343-1537) -----------------------------------------------

    // C# `SetDefaultMixture(IList<UnivariateDistributionType>)` (line 381): equal weights
    // (last weight closes the sum to exactly one), factory-constructed components.
    void set_default_mixture(const std::vector<DistributionType>& distribution_types) {
        if (distribution_types.empty())
            throw std::invalid_argument("At least one distribution type is required.");

        std::vector<double> weights;
        std::vector<std::unique_ptr<DistributionBase>> distributions;
        double w = 1.0 / static_cast<double>(distribution_types.size());
        double sum = 0.0;

        for (std::size_t i = 0; i < distribution_types.size(); ++i) {
            if (i != distribution_types.size() - 1) {
                sum += w;
                weights.push_back(w);
            } else {
                // Ensure exact sum of weights is one.
                weights.push_back(1.0 - sum);
            }

            // C# UnivariateDistributionFactory.CreateDistribution (no whitelist check here;
            // Validate() reports unsupported component types).
            distributions.push_back(
                numerics::distributions::create_distribution(distribution_types[i]));
        }

        set_mixture(std::make_unique<Mixture>(std::move(weights), std::move(distributions)));
    }

    // C# `SetDefaultParameters` override (line 411): weight parameters (bounds [0,1],
    // Uniform(0,1) priors; skipped entirely for a single-component mixture), then per
    // component the constraint-driven ModelParameters (OwnerName "D<i>", Uniform
    // (lower, upper) prior, IsPositive when the lower bound is the machine epsilon).
    void set_default_parameters() override {
        // (Handler removal/re-add is INPC plumbing, skipped.)
        parameters_.clear();

        if (mixture_ == nullptr || !has_data_frame() || !data_frame().validate().is_valid ||
            mixture_->component_count() == 0) {
            return;
        }

        // Priors for mixture weights.
        int k = mixture_->component_count();
        if (k > 1) {
            double w0 = 1.0 / static_cast<double>(k);
            for (int i = 0; i < k; ++i) {
                parameters_.emplace_back(
                    "", "Weight (w" + to_subscript(i + 1) + ")", w0, 0.0, 1.0,
                    std::make_unique<numerics::distributions::Uniform>(0.0, 1.0));
            }
        }

        // Priors for distribution parameters.
        std::vector<double> sample;
        sample.reserve(data_frame().exact_series().count());
        for (std::size_t i = 0; i < data_frame().exact_series().count(); ++i)
            sample.push_back(data_frame().exact_series()[i].value());

        for (int i = 0; i < k; ++i) {
            const DistributionBase& component = mixture_->component(i);
            const auto& mle_component =
                dynamic_cast<const numerics::distributions::IMaximumLikelihoodEstimation&>(
                    component);

            std::vector<double> initials, lowers, uppers;
            mle_component.get_parameter_constraints(sample, initials, lowers, uppers);

            // Name stays "" -- component.ParametersToString is not on the ported
            // distribution base (see the file header).
            for (int j = 0; j < component.number_of_parameters(); ++j) {
                std::size_t sj = static_cast<std::size_t>(j);
                parameters_.emplace_back(
                    "D" + std::to_string(i + 1), "", initials[sj], lowers[sj], uppers[sj],
                    std::make_unique<numerics::distributions::Uniform>(lowers[sj], uppers[sj]),
                    /*is_positive=*/lowers[sj] == numerics::kDoubleMachineEpsilon);
            }
        }
    }

    // C# `SetDefaultQuantilePriors` override (line 485): disabled -> clear both lists;
    // enabled -> exactly ONE LnNormal prior at alpha = 0.1 with mu = Round(InverseCDF(1 -
    // alpha), 2), sigma = Round(0.15 * mu, 2); an existing list is shrunk from the back
    // (the extend branch is ported but unreachable with qCount == 1 and a non-empty list).
    // Unlike the UnivariateDistributionModel sibling, the C# ends with
    // ProcessQuantilePriors().
    void set_default_quantile_priors() override {
        if (mixture_ == nullptr) return;

        // (Handler removal for the old priors is INPC plumbing, skipped.)

        // If quantile priors are not enabled, clear the list and return.
        if (!enable_quantile_priors()) {
            quantile_priors_.clear();
            quantile_priors_true_.clear();
            return;
        }

        std::vector<QuantilePrior> priors;
        const std::size_t q_count = 1;  // mixture model uses single quantile prior

        if (quantile_priors_.size() >= 1) {
            priors = quantile_priors_;

            while (priors.size() > q_count) priors.pop_back();

            while (priors.size() < q_count) {
                priors.emplace_back(priors.back().alpha() / 10.0,
                                    std::make_unique<numerics::distributions::LnNormal>());
                double mu =
                    round_half_even_2(mixture_->inverse_cdf(1.0 - priors.back().alpha()));
                double sigma = round_half_even_2(mu * 0.15);
                priors.back().distribution().set_parameters({mu, sigma});
            }
        } else {
            for (std::size_t i = 1; i <= q_count; ++i) {
                priors.emplace_back(std::pow(10.0, -static_cast<double>(i)),
                                    std::make_unique<numerics::distributions::LnNormal>());
                double mu =
                    round_half_even_2(mixture_->inverse_cdf(1.0 - priors[i - 1].alpha()));
                double sigma = round_half_even_2(mu * 0.15);
                priors[i - 1].distribution().set_parameters({mu, sigma});
            }
        }

        // Reset the quantile priors with the new list (handler re-adds skipped).
        quantile_priors_ = std::move(priors);

        process_quantile_priors();
    }

    // C# `ProcessQuantilePriors` override (line 546): the single prior is cloned through.
    // The multi-quantile branch is ported for structure but unreachable -- UseSingleQuantile
    // is hard-true for this model ("left here for possible future extension", per the C#).
    void process_quantile_priors() override {
        quantile_priors_true_.clear();

        if (quantile_priors_.size() == 1) {
            quantile_priors_true_.push_back(quantile_priors_[0].clone());
            return;
        }

        if (!use_single_quantile() && mixture_ != nullptr &&
            static_cast<int>(quantile_priors_.size()) == mixture_->number_of_parameters()) {
            quantile_priors_true_.push_back(quantile_priors_[0].clone());

            for (std::size_t i = 1; i < quantile_priors_.size(); ++i) {
                double mu_diff = quantile_priors_[i].distribution().mean() -
                                 quantile_priors_[i - 1].distribution().mean();
                double sigma_diff = std::sqrt(quantile_priors_[i].distribution().variance() +
                                              quantile_priors_[i - 1].distribution().variance());

                auto gamma_dist = std::make_unique<numerics::distributions::GammaDistribution>();
                gamma_dist->set_parameters(
                    gamma_dist->parameters_from_moments({mu_diff, sigma_diff}));

                quantile_priors_true_.emplace_back(quantile_priors_[i].alpha(),
                                                   std::move(gamma_dist));
            }
        }
    }

    // C# `ExpectationMaximization(out parameters, out covariance, out iterations,
    // maxIterations = 1000, tolerance = 1E-8)` (line 586): EM for approximate MLE
    // (weights via responsibilities, component parameters via a NelderMead M-step over the
    // full censored likelihood), then the covariance -- multinomial approximation for the
    // weight block, inverse negative Hessian of the E-step log-likelihood for the
    // component-parameter block. The C# `out` parameters map to output references.
    void expectation_maximization(std::vector<double>& parameters, Matrix& covariance,
                                  int& iterations, int max_iterations = 1000,
                                  double tolerance = 1E-8) const {
        parameters.clear();
        covariance = Matrix(0, 0);
        iterations = 0;

        if (mixture_ == nullptr || !has_data_frame()) return;

        std::unique_ptr<DistributionBase> model_owner = mixture_->clone();
        Mixture& model = static_cast<Mixture&>(*model_owner);

        data_frame().create_full_time_series();
        const int N = data_frame().total_record_length();
        int Np = 0;
        for (int i = 0; i < model.component_count(); ++i)
            Np += model.component(i).number_of_parameters();
        const int K = model.component_count();

        if (N == 0 || K == 0) return;

        // Get constraints.
        std::vector<double> sample;
        sample.reserve(data_frame().exact_series().count());
        for (std::size_t i = 0; i < data_frame().exact_series().count(); ++i)
            sample.push_back(data_frame().exact_series()[i].value());

        std::vector<double> all_initials, all_lowers, all_uppers;
        model.get_parameter_constraints(sample, all_initials, all_lowers, all_uppers);
        // C# tuple.ItemN.Subset(K): elements K..end; Subset(0, K - 1): elements 0..K-1.
        std::vector<double> initial_values(all_initials.begin() + K, all_initials.end());
        std::vector<double> lower_bounds(all_lowers.begin() + K, all_lowers.end());
        std::vector<double> upper_bounds(all_uppers.begin() + K, all_uppers.end());

        // Initialize EM state.
        std::vector<double> mle_weights(all_initials.begin(), all_initials.begin() + K);
        std::vector<double> mle_parameters = initial_values;
        std::vector<std::vector<double>> likelihood(
            static_cast<std::size_t>(N), std::vector<double>(static_cast<std::size_t>(K), 0.0));

        double old_log_lh = -std::numeric_limits<double>::infinity();
        double new_log_lh = -std::numeric_limits<double>::infinity();

        const DataFrame& df = data_frame();
        const double neg_inf = -std::numeric_limits<double>::infinity();

        // E-step: compute responsibilities and log-likelihood (C# local function EStep).
        auto e_step = [&](const std::vector<double>& x) -> double {
            // Set mixture parameters (weights held at the current EM state).
            model.set_parameters(mle_weights, x);

            // Outer loop for computing the likelihoods.
            const std::vector<std::unique_ptr<Data>>& fts = df.full_time_series();
            for (int k = 0; k < K; ++k) {
                std::size_t sk = static_cast<std::size_t>(k);
                for (std::size_t i = 0; i < fts.size(); ++i) {
                    const Data& data = *fts[i];

                    // Exact Data.
                    if (const auto* exact = dynamic_cast<const ExactData*>(&data)) {
                        if (!exact->is_low_outlier()) {
                            if (model.is_zero_inflated() && data.value() <= 0.0) {
                                likelihood[i][sk] = std::log(model.zero_weight());
                            } else {
                                likelihood[i][sk] = std::log(mle_weights[sk]) +
                                                    model.component(k).log_pdf(data.value());
                            }
                        } else {
                            // Low outliers are treated as left censored.
                            likelihood[i][sk] =
                                std::log(mle_weights[sk]) +
                                model.component(k).log_cdf(df.low_outlier_threshold());
                        }
                    }
                    // Uncertain Data.
                    else if (const auto* uncertain = dynamic_cast<const UncertainData*>(&data)) {
                        const DistributionBase& dist = uncertain->distribution();
                        double lower_probability = 1E-8;
                        double upper_probability = 1.0 - 1E-8;
                        double a = dist.inverse_cdf(lower_probability);
                        double b = dist.inverse_cdf(upper_probability);
                        double mass = upper_probability - lower_probability;
                        if (numerics::is_finite(a) && numerics::is_finite(b) &&
                            numerics::is_finite(mass) && mass > 0.0 && a < b &&
                            mle_weights[sk] > 0) {
                            double ep =
                                numerics::math::integration::Integration::gauss_legendre20(
                                    [&](double q) {
                                        return dist.pdf(q) * model.component(k).pdf(q);
                                    },
                                    a, b) /
                                mass;
                            likelihood[i][sk] = std::log(mle_weights[sk]) +
                                                (ep > 0 ? std::log(ep) : neg_inf);
                        } else {
                            likelihood[i][sk] = neg_inf;
                        }
                    }
                    // Interval data: guard the log argument against CDF differences <= 0.
                    else if (const auto* interval = dynamic_cast<const IntervalData*>(&data)) {
                        double cdf_diff = model.component(k).cdf(interval->upper_value()) -
                                          model.component(k).cdf(interval->lower_value());
                        likelihood[i][sk] =
                            cdf_diff > 0 ? std::log(mle_weights[sk]) + std::log(cdf_diff)
                                         : neg_inf;
                    }
                    // Threshold data. Invariant from DataFrame.CreateFullTimeSeries: each
                    // threshold ordinate carries either NumberBelow == 1 OR NumberAbove ==
                    // 1, never both; the -inf fall-through guards malformed data.
                    else if (const auto* threshold = dynamic_cast<const ThresholdData*>(&data)) {
                        if (threshold->number_below() == 1 && threshold->number_above() == 0) {
                            likelihood[i][sk] = std::log(mle_weights[sk]) +
                                                model.component(k).log_cdf(threshold->value());
                        } else if (threshold->number_below() == 0 &&
                                   threshold->number_above() == 1) {
                            likelihood[i][sk] = std::log(mle_weights[sk]) +
                                                model.component(k).log_ccdf(threshold->value());
                        } else {
                            likelihood[i][sk] = neg_inf;
                        }
                    }
                }
            }

            // Normalize the unnormalized log likelihoods with log-sum-exp and accumulate
            // the true log-likelihood.
            double lh = 0.0;
            for (int i = 0; i < N; ++i) {
                std::size_t si = static_cast<std::size_t>(i);
                // Get max likelihood.
                double max = neg_inf;
                for (int k = 0; k < K; ++k) {
                    if (likelihood[si][static_cast<std::size_t>(k)] > max)
                        max = likelihood[si][static_cast<std::size_t>(k)];
                }

                if (!numerics::is_finite(max)) continue;  // sum <= 0 is excluded above

                // log-sum-exp trick begins here.
                double sum = 0.0;
                for (int k = 0; k < K; ++k)
                    sum += std::exp(likelihood[si][static_cast<std::size_t>(k)] - max);

                if (sum <= 0) continue;

                double tmp = max + std::log(sum);
                lh += tmp;

                for (int k = 0; k < K; ++k) {
                    std::size_t sk = static_cast<std::size_t>(k);
                    likelihood[si][sk] = std::exp(likelihood[si][sk] - tmp);
                    // After normalization, likelihood is a probability in [0,1]; if not
                    // finite, set to 0.
                    if (!numerics::is_finite(likelihood[si][sk])) likelihood[si][sk] = 0.0;
                }
            }

            if (!numerics::is_finite(lh)) return neg_inf;
            return lh;
        };

        // The log-likelihood function to maximize in the M-step: weights held fixed, only
        // the distribution parameters are solved (C# local function logLH).
        auto log_lh_fn = [&](const std::vector<double>& x) -> double {
            // Set mixture parameters.
            model.set_parameters(mle_weights, x);

            double lh = 0.0;
            // Exact Data.
            for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
                const ExactData& data = df.exact_series()[i];
                if (!data.is_low_outlier()) {
                    lh += model.log_likelihood(data.value());
                } else {
                    // Low outliers are treated as left censored.
                    lh += model.log_likelihood_left_censored(df.low_outlier_threshold(), 1);
                }
            }

            // Uncertain Data.
            for (std::size_t i = 0; i < df.uncertain_series().count(); ++i) {
                const DistributionBase& dist = df.uncertain_series()[i].distribution();
                double lower_probability = 1E-8;
                double upper_probability = 1.0 - 1E-8;
                double a = dist.inverse_cdf(lower_probability);
                double b = dist.inverse_cdf(upper_probability);
                double mass = upper_probability - lower_probability;
                if (numerics::is_finite(a) && numerics::is_finite(b) &&
                    numerics::is_finite(mass) && mass > 0.0 && a < b) {
                    // Normalize by retained ME mass so truncated 1E-8 tails do not damp the
                    // uncertain likelihood contribution.
                    double ep = numerics::math::integration::Integration::gauss_legendre20(
                                    [&](double q) { return dist.pdf(q) * model.pdf(q); }, a,
                                    b) /
                                mass;
                    lh += ep > 0 ? std::log(ep) : neg_inf;
                } else {
                    lh += neg_inf;
                }
            }

            // Interval Data.
            for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
                const IntervalData& data = df.interval_series()[i];
                lh += model.log_likelihood_intervals(data.lower_value(), data.upper_value());
            }

            // Threshold Data.
            for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
                const ThresholdData& data = df.threshold_series()[i];
                if (data.number_below() > 0)
                    lh += model.log_likelihood_left_censored(data.value(), data.number_below());
                if (data.number_above() > 0)
                    lh +=
                        model.log_likelihood_right_censored(data.value(), data.number_above());
            }

            // Return the full log-likelihood.
            if (!numerics::is_finite(lh)) return neg_inf;
            return lh;
        };

        // M-step: update weights and continuous parameters (C# local function MStep).
        auto m_step = [&](const std::vector<double>& x) -> std::vector<double> {
            // Get updated weights.
            const std::vector<std::unique_ptr<Data>>& fts = df.full_time_series();
            for (int k = 0; k < K; ++k) {
                double wgt = 0.0;
                for (std::size_t i = 0; i < fts.size(); ++i) {
                    const Data& data = *fts[i];
                    if (!is_zero_inflated_ || data.value() > 0.0) {
                        wgt += likelihood[i][static_cast<std::size_t>(k)];
                    }
                }
                mle_weights[static_cast<std::size_t>(k)] = wgt / N;
            }
            // MLE (C# line 774: new NelderMead(logLH, Np, x, lowerBounds, upperBounds);
            // Maximize()).
            numerics::math::optimization::NelderMead solver(log_lh_fn, Np, x, lower_bounds,
                                                            upper_bounds);
            solver.maximize();
            return solver.best_parameters();
        };

        // Estimate using the EM Algorithm.
        for (iterations = 1; iterations <= max_iterations; ++iterations) {
            // Perform the expectation step.
            new_log_lh = e_step(mle_parameters);

            // Check convergence.
            if (numerics::is_finite(old_log_lh)) {
                if (std::fabs((old_log_lh - new_log_lh) / old_log_lh) < tolerance) break;
            }

            // Perform the maximization step.
            mle_parameters = m_step(mle_parameters);

            // Update log-likelihood state.
            old_log_lh = new_log_lh;
        }

        // Return the full list of distribution parameters.
        parameters = mle_weights;
        parameters.insert(parameters.end(), mle_parameters.begin(), mle_parameters.end());

        // Estimate the covariance matrix of parameters: get the Hessian of the E-step
        // log-likelihood and invert it (Fisher information).
        Matrix hessian = estimation::NumericalDiff::compute_hessian(
            [&](const std::vector<double>& x) { return e_step(x); }, mle_parameters,
            static_cast<int>(mle_parameters.size()));
        Matrix fisher = hessian * -1.0;
        fisher = fisher.inverse();
        covariance = Matrix(static_cast<int>(parameters.size()),
                            static_cast<int>(parameters.size()));
        // Mixing weights covariance (multinomial approximation).
        for (int i = 0; i < K; ++i) {
            double wi = mle_weights[static_cast<std::size_t>(i)];
            covariance(i, i) = wi * (1.0 - wi) / N;

            for (int j = 0; j < K; ++j) {
                if (i == j) continue;

                double wj = mle_weights[static_cast<std::size_t>(j)];
                // Equicorrelation rho between MLE Dirichlet weights, nudged by a tiny
                // epsilon so the covariance stays strictly positive-definite for downstream
                // Cholesky. Guard K == 1 (no off-diagonal for a single component).
                if (K <= 1) {
                    covariance(i, j) = 0.0;
                    continue;
                }
                double rho = -1.0 / (K - 1.0) + numerics::kDoubleMachineEpsilon;
                double vari = wi * (1.0 - wi) / N;
                double varj = wj * (1.0 - wj) / N;
                covariance(i, j) = rho * std::sqrt(vari * varj);
            }
        }
        // Component parameter covariance from Fisher information.
        for (int i = 0; i < fisher.number_of_rows(); ++i) {
            for (int j = 0; j < fisher.number_of_columns(); ++j) {
                covariance(i + K, j + K) = fisher(i, j);
            }
        }
    }

    // C# `LogLikelihood` override (line 913): data likelihood (early -inf), plus prior,
    // then the finite collapse.
    double log_likelihood(std::vector<double>& parameters) const override {
        // Get the data likelihood.
        double data_log_lh = data_log_likelihood(parameters);
        if (!numerics::is_finite(data_log_lh)) return -std::numeric_limits<double>::infinity();

        // Get the prior likelihood.
        double prior_log_lh = prior_log_likelihood(parameters);

        // The full likelihood.
        double log_lh = data_log_lh + prior_log_lh;

        return numerics::is_finite(log_lh) ? log_lh
                                           : -std::numeric_limits<double>::infinity();
    }

    // C# `DataLogLikelihood` override: the full censored likelihood over a working copy of
    // the mixture. Numerics v2.2 normalizes weights inside the mixture without modifying the
    // caller's optimizer coordinates.
    double data_log_likelihood(std::vector<double>& parameters) const override {
        if (mixture_ == nullptr || !has_data_frame())
            return -std::numeric_limits<double>::infinity();

        std::unique_ptr<DistributionBase> model_owner = mixture_->clone();
        Mixture& model = static_cast<Mixture&>(*model_owner);
        double log_lh = 0.0;
        const double neg_inf = -std::numeric_limits<double>::infinity();

        // Set model parameters (weights normalized in place, like the C# ref call).
        model.set_parameters_normalized(parameters);

        const DataFrame& df = data_frame();

        // Exact Data.
        for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
            const ExactData& data = df.exact_series()[i];
            if (!data.is_low_outlier()) {
                log_lh += model.log_likelihood(data.value());
            } else {
                log_lh += model.log_likelihood_left_censored(df.low_outlier_threshold(), 1);
            }
        }

        // Uncertain Data.
        for (std::size_t i = 0; i < df.uncertain_series().count(); ++i) {
            const DistributionBase& dist = df.uncertain_series()[i].distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            if (numerics::is_finite(a) && numerics::is_finite(b) && numerics::is_finite(mass) &&
                mass > 0.0 && a < b) {
                double ep = numerics::math::integration::Integration::gauss_legendre20(
                                [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                            mass;
                log_lh += ep > 0 ? std::log(ep) : neg_inf;
            } else {
                log_lh += neg_inf;
            }
        }

        // Interval Data.
        for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
            const IntervalData& data = df.interval_series()[i];
            log_lh += model.log_likelihood_intervals(data.lower_value(), data.upper_value());
        }

        // Threshold Data.
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            if (data.number_below() > 0)
                log_lh += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                log_lh +=
                    model.log_likelihood_right_censored(data.value(), data.number_above());
        }
        return log_lh;
    }

    // C# `PointwiseDataLogLikelihood` override (line 996): one entry per observation;
    // threshold entries combine the left and right censored contributions.
    std::vector<double> pointwise_data_log_likelihood(
        const std::vector<double>& parameters) const override {
        if (mixture_ == nullptr || !has_data_frame()) return {};

        std::unique_ptr<DistributionBase> model_owner = mixture_->clone();
        Mixture& model = static_cast<Mixture&>(*model_owner);
        const double neg_inf = -std::numeric_limits<double>::infinity();

        // Set model parameters (the C# copies the array first here too).
        std::vector<double> parms_copy = parameters;
        model.set_parameters_normalized(parms_copy);

        std::vector<double> result;
        const DataFrame& df = data_frame();

        // Exact Data.
        for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
            const ExactData& data = df.exact_series()[i];
            if (!data.is_low_outlier()) {
                result.push_back(model.log_likelihood(data.value()));
            } else {
                result.push_back(
                    model.log_likelihood_left_censored(df.low_outlier_threshold(), 1));
            }
        }

        // Uncertain Data.
        for (std::size_t i = 0; i < df.uncertain_series().count(); ++i) {
            const DistributionBase& dist = df.uncertain_series()[i].distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            if (numerics::is_finite(a) && numerics::is_finite(b) && numerics::is_finite(mass) &&
                mass > 0.0 && a < b) {
                double ep = numerics::math::integration::Integration::gauss_legendre20(
                                [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                            mass;
                result.push_back(ep > 0 ? std::log(ep) : neg_inf);
            } else {
                result.push_back(neg_inf);
            }
        }

        // Interval Data.
        for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
            const IntervalData& data = df.interval_series()[i];
            result.push_back(
                model.log_likelihood_intervals(data.lower_value(), data.upper_value()));
        }

        // Threshold Data: combine left and right censored contributions for this threshold.
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            double ll = 0.0;
            if (data.number_below() > 0)
                ll += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                ll += model.log_likelihood_right_censored(data.value(), data.number_above());
            result.push_back(ll);
        }

        return result;
    }

    // C# `PointwiseDataLogLikelihoodComponents` override (line 1067).
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& parameters) const override {
        if (mixture_ == nullptr || !has_data_frame()) return {};

        std::unique_ptr<DistributionBase> model_owner = mixture_->clone();
        Mixture& model = static_cast<Mixture&>(*model_owner);
        const double neg_inf = -std::numeric_limits<double>::infinity();

        std::vector<double> parms_copy = parameters;
        model.set_parameters_normalized(parms_copy);

        std::vector<DataComponent> result;
        int idx = 0;
        const DataFrame& df = data_frame();

        // Exact Data.
        for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
            const ExactData& data = df.exact_series()[i];
            double log_lh;
            double value = data.value();

            if (!data.is_low_outlier()) {
                log_lh = model.log_likelihood(data.value());
            } else {
                log_lh = model.log_likelihood_left_censored(df.low_outlier_threshold(), 1);
                value = df.low_outlier_threshold();
            }

            result.emplace_back(idx++, log_lh, value, DataComponentType::Exact, 1,
                                std::to_string(data.index()));
        }

        // Uncertain Data.
        for (std::size_t i = 0; i < df.uncertain_series().count(); ++i) {
            const UncertainData& data = df.uncertain_series()[i];
            const DistributionBase& dist = data.distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            double log_lh = neg_inf;
            if (numerics::is_finite(a) && numerics::is_finite(b) && numerics::is_finite(mass) &&
                mass > 0.0 && a < b) {
                double ep = numerics::math::integration::Integration::gauss_legendre20(
                                [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                            mass;
                log_lh = ep > 0 ? std::log(ep) : neg_inf;
            }
            result.emplace_back(idx++, log_lh, dist.mean(), DataComponentType::Uncertain, 1,
                                std::to_string(data.index()));
        }

        // Interval Data.
        for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
            const IntervalData& data = df.interval_series()[i];
            double log_lh =
                model.log_likelihood_intervals(data.lower_value(), data.upper_value());
            double midpoint = (data.lower_value() + data.upper_value()) / 2.0;

            result.emplace_back(idx++, log_lh, midpoint, DataComponentType::Interval, 1,
                                std::to_string(data.index()));
        }

        // Threshold Data.
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            double ll = 0.0;
            if (data.number_below() > 0)
                ll += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                ll += model.log_likelihood_right_censored(data.value(), data.number_above());

            int total_count = data.number_below() + data.number_above();
            result.emplace_back(idx++, ll, data.value(), DataComponentType::LeftCensored,
                                total_count, "Threshold " + std::to_string(i + 1));
        }

        return result;
    }

    // C# `PriorLogLikelihood` override (line 1146): parameter priors (evaluated at the
    // NORMALIZED weights, per the C# ref side effect -- file header), the Jeffreys 1/scale
    // term per component (Gamma/Weibull carry the scale in parameter 0, all other supported
    // families in parameter 1), and the single quantile prior. Like the C#, the raw sum is
    // returned (no finite collapse; LogLikelihood collapses). T14 (BestFit v2.0.0, commit
    // 1abe795): a component with no applicable scale term (a single-parameter family, e.g.
    // Poisson -- reachable here since components are not constrained to the
    // UnivariateDistributionModel 15-family whitelist) is skipped entirely rather than
    // indexed out of range; a component with a non-positive scale returns -Inf IMMEDIATELY
    // (an early function return, not merely a loop break) rather than subtracting +Inf, which
    // could previously cancel against another +Inf contribution into a NaN.
    double prior_log_likelihood(std::vector<double>& parameters) const override {
        if (mixture_ == nullptr) return -std::numeric_limits<double>::infinity();

        std::unique_ptr<DistributionBase> model_owner = mixture_->clone();
        Mixture& model = static_cast<Mixture&>(*model_owner);
        int k = model.component_count();
        double log_lh = 0.0;

        // Set model parameters (weights normalized in the CALLER'S vector, like the C#
        // `Mixture!.SetParameters(ref parameters)` at line 1156; the priors below read the
        // normalized values).
        model.set_parameters_normalized(parameters);

        // Parameter Priors.
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            log_lh += parameters_[i].prior_distribution().log_pdf(parameters[i]);
        }

        // Jeffreys rule on scale parameters for each component (T14; see the method header).
        if (use_jeffreys_rule_for_scale()) {
            for (int j = 0; j < k; ++j) {
                const DistributionBase& dist = model.component(j);
                double scale;
                if (!try_get_jeffreys_scale_parameter(dist, scale)) continue;

                if (scale <= 0) return -std::numeric_limits<double>::infinity();
                log_lh -= std::log(scale);
            }
        }

        // Single quantile prior.
        if (enable_quantile_priors() && quantile_priors_true_.size() == 1) {
            log_lh += quantile_priors_true_[0].distribution().log_pdf(
                model.inverse_cdf(1.0 - quantile_priors_true_[0].alpha()));
        }

        return log_lh;
    }

    // C# `PointwisePriorLogLikelihood` override (line 1195): per-parameter prior components
    // (evaluated at the RAW proposal -- the C# normalizes a separate copy; file header),
    // per-component Jeffreys components, and the single quantile-prior component.
    std::vector<PriorComponent> pointwise_prior_log_likelihood(
        const std::vector<double>& parameters) const override {
        std::vector<PriorComponent> result;

        if (mixture_ == nullptr) return result;

        std::unique_ptr<DistributionBase> model_owner = mixture_->clone();
        Mixture& model = static_cast<Mixture&>(*model_owner);
        int k = model.component_count();

        // Set model parameters (separate copy, like the C# parmsCopy).
        std::vector<double> parms_copy = parameters;
        model.set_parameters_normalized(parms_copy);

        // Parameter Priors.
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            double ll = parameters_[i].prior_distribution().log_pdf(parameters[i]);
            const std::string& param_name = parameters_[i].owner_name().empty()
                                                ? parameters_[i].name()
                                                : parameters_[i].owner_name();
            result.emplace_back("Parameter Prior: " + param_name, ll,
                                PriorComponentType::ParameterPrior);
        }

        // Jeffreys rule on scale parameters for each component. A component with no
        // applicable scale term is skipped (no component appended); the scale NAME is looked
        // up but discarded (C# `out _`) since this label numbers components positionally
        // (T14; see the scalar prior_log_likelihood's header note).
        if (use_jeffreys_rule_for_scale()) {
            for (int j = 0; j < k; ++j) {
                const DistributionBase& dist = model.component(j);
                double scale;
                std::string scale_name;
                if (!try_get_jeffreys_scale_parameter(dist, scale, scale_name)) continue;

                result.emplace_back("Jeffreys Scale: Component " + std::to_string(j + 1),
                                    scale > 0 ? -std::log(scale)
                                              : -std::numeric_limits<double>::infinity(),
                                    PriorComponentType::JeffreysScalePrior);
            }
        }

        // Single quantile prior. Label: the C# formats alpha with :P2 (percent, two
        // decimals); approximated with printf (display-only).
        if (enable_quantile_priors() && quantile_priors_true_.size() == 1) {
            double quantile = model.inverse_cdf(1.0 - quantile_priors_true_[0].alpha());
            double ll = quantile_priors_true_[0].distribution().log_pdf(quantile);
            char alpha_buf[64];
            std::snprintf(alpha_buf, sizeof alpha_buf, "%.2f%%",
                          quantile_priors_true_[0].alpha() * 100.0);
            result.emplace_back(std::string("Quantile Prior: p=") + alpha_buf, ll,
                                PriorComponentType::QuantilePrior);
        }

        return result;
    }

    // C# `SetParameterValues` override (line 1260): length guard, write the ModelParameter
    // values, then push the (normalized) proposal into the wrapped mixture. The C#
    // dereferences `Mixture!` unguarded (NullReferenceException on a null mixture); the C++
    // throws std::runtime_error instead of crashing.
    void set_parameter_values(const std::vector<double>& parameters) override {
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters()))
            throw std::invalid_argument("The length of the parameter list is incorrect.");

        for (std::size_t i = 0; i < parameters.size(); ++i)
            parameters_[i].set_value(parameters[i]);

        if (mixture_ == nullptr)
            throw std::runtime_error("Mixture distribution is null.");  // C# NRE guard
        std::vector<double> parms_copy = parameters;
        mixture_->set_parameters_normalized(parms_copy);
    }

    // C# `Clone()` override (line 1276): deep, independent copy -- cloned Parameters and
    // QuantilePriors, the flag fields, then ProcessQuantilePriors(). See the file header
    // for the DataFrame deep-copy and zero-inflation re-sync divergences.
    MixtureModel clone() const {
        if (!has_data_frame())
            throw std::invalid_argument("dataFrame");  // C++ divergence: frame required
        if (mixture_ == nullptr)
            throw std::invalid_argument("mixture");  // C# `Mixture!` would throw

        std::vector<ModelParameter> parms;
        parms.reserve(parameters_.size());
        for (std::size_t i = 0; i < parameters_.size(); ++i)
            parms.push_back(parameters_[i].clone());

        std::vector<QuantilePrior> quants;
        quants.reserve(quantile_priors_.size());
        for (std::size_t i = 0; i < quantile_priors_.size(); ++i)
            quants.push_back(quantile_priors_[i].clone());

        MixtureModel result(data_frame().clone(), *mixture_);
        // C# object-initializer field writes (no setter side effects).
        result.is_zero_inflated_ = is_zero_inflated_;
        result.use_default_flat_priors_ = use_default_flat_priors_;
        result.use_jeffreys_rule_for_scale_ = use_jeffreys_rule_for_scale_;
        result.enable_quantile_priors_ = enable_quantile_priors_;
        result.use_single_quantile_ = true;
        result.parameters_ = std::move(parms);
        result.quantile_priors_ = std::move(quants);

        // T14 end-state re-sync (BestFit v2.0.0, commit b43943c; file header): the ctor path
        // above ran the Mixture setter with the default zero-inflation state and stomped the
        // cloned mixture's IsZeroInflated/ZeroWeight; restore them from the ORIGINAL MODEL's
        // state (`is_zero_inflated_`, matching `result._isZeroInflated` in the C# -- already
        // copied onto `result` above) so the clone's likelihood surface matches the
        // original's. Ported literally: `result.Mixture!.IsZeroInflated =
        // result._isZeroInflated; result.Mixture.ZeroWeight = result._isZeroInflated ?
        // Mixture!.ZeroWeight : 0.0;`. v2.1.4 turned Mixture's IsZeroInflated/ZeroWeight into
        // setters that renormalize component weights as a side effect (see mixture.hpp's
        // header note), so this restore is no longer a bit-exact field copy -- it goes
        // through set_is_zero_inflated()/set_zero_weight() in that order (matching every
        // other IsZeroInflated-then-ZeroWeight call site) and ends up numerically equal to
        // the original's weights to floating-point precision, not bit-identical.
        result.mixture_->set_is_zero_inflated(is_zero_inflated_);
        result.mixture_->set_zero_weight(is_zero_inflated_ ? mixture_->zero_weight() : 0.0);

        result.process_quantile_priors();
        return result;
    }

    // SKIPPED: ToXElement (C# line 1302) -- XML serialization is a project-wide deferral.

    // C# `Validate` override (line 1329).
    ValidationResult validate() const override {
        ValidationResult result;

        // Data frame.
        if (!has_data_frame()) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: Data frame is null.");
            return result;
        }

        // Mixture distribution.
        if (mixture_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: Mixture distribution is null.");
            return result;
        }

        // Validate data frame.
        ValidationResult data_valid = data_frame().validate();
        if (!data_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              data_valid.validation_messages.begin(),
                                              data_valid.validation_messages.end());
        }

        if (mixture_->component_count() == 0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Mixture distribution has no component distributions.");
        }

        if (mixture_->component_count() < 1 || mixture_->component_count() > 3) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Mixture model currently supports 1 to 3 component distributions.");
        }

        // Validate uncertain-data ME bounds before likelihood evaluation. Each uncertain
        // integral is normalized by the retained mass in the 1E-8 probability window.
        for (std::size_t i = 0; i < data_frame().uncertain_series().count(); ++i) {
            const UncertainData& data = data_frame().uncertain_series()[i];
            const DistributionBase& dist = data.distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double lower = dist.inverse_cdf(lower_probability);
            double upper = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;

            if (!numerics::is_finite(lower) || !numerics::is_finite(upper) ||
                !numerics::is_finite(mass) || mass <= 0.0 || lower >= upper) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: Uncertain data at index " + std::to_string(data.index()) +
                    " has invalid measurement-error integration bounds at the 1E-8 "
                    "probability window.");
            }
        }

        // Component distribution type check. (The C# message quotes the enum name; the
        // ported enum has no name surface, so the message carries the index only.)
        for (int i = 0; i < mixture_->component_count(); ++i) {
            if (!is_supported_distribution_type(mixture_->component(i).type())) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: Component distribution " + std::to_string(i + 1) +
                    " has an unsupported type.");
            }
        }

        // Zero-inflation checks.
        if (is_zero_inflated_) {
            if (mixture_->zero_weight() < 0.0 || mixture_->zero_weight() >= 1.0 ||
                std::isnan(mixture_->zero_weight()) || std::isinf(mixture_->zero_weight())) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: ZeroWeight must be in [0, 1) for a zero-inflated model.");
            }
        } else {
            if (mixture_->zero_weight() != 0.0) {
                // Warning only -- does not invalidate, exactly like the C#.
                result.validation_messages.push_back(
                    "Warning: ZeroWeight is non-zero while IsZeroInflated is false. This "
                    "will be reset to zero at runtime.");
            }
        }

        // Check log distributions against data and retained ME support.
        for (int i = 0; i < mixture_->component_count(); ++i) {
            const DistributionBase& dist = mixture_->component(i);

            if (dist.type() == DistributionType::LnNormal ||
                dist.type() == DistributionType::LogNormal ||
                dist.type() == DistributionType::LogPearsonTypeIII) {
                bool has_non_positive_exact = false;
                if (!is_zero_inflated_ && data_frame().exact_series().count() > 0) {
                    for (std::size_t j = 0; j < data_frame().exact_series().count(); ++j) {
                        const ExactData& data = data_frame().exact_series()[j];
                        if (!data.is_low_outlier() && data.value() <= 0.0) {
                            has_non_positive_exact = true;
                            break;
                        }
                    }
                }

                bool has_non_positive_uncertain = false;
                for (std::size_t j = 0; j < data_frame().uncertain_series().count(); ++j) {
                    const DistributionBase& me_dist =
                        data_frame().uncertain_series()[j].distribution();
                    double lower = me_dist.inverse_cdf(1E-8);

                    // Zero inflation can account for exact zeros, but it cannot make a log
                    // component valid over negative retained ME support.
                    if (!numerics::is_finite(lower) || lower <= 0.0) {
                        has_non_positive_uncertain = true;
                        break;
                    }
                }

                if (has_non_positive_exact || has_non_positive_uncertain) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Component distribution " + std::to_string(i + 1) +
                        " is log-based but data include non-positive values or retained "
                        "uncertain support.");
                }
            }
        }

        // Parameter priors.
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            ValidationResult valid = parameters_[i].validate();
            if (!valid.is_valid) {
                result.is_valid = false;
                result.validation_messages.insert(result.validation_messages.end(),
                                                  valid.validation_messages.begin(),
                                                  valid.validation_messages.end());
            }
        }

        // Quantile priors.
        for (std::size_t i = 0; i < quantile_priors_.size(); ++i) {
            ValidationResult valid = quantile_priors_[i].validate();
            if (!valid.is_valid) {
                result.is_valid = false;
                result.validation_messages.insert(result.validation_messages.end(),
                                                  valid.validation_messages.begin(),
                                                  valid.validation_messages.end());
            }

            if (i >= 1) {
                if (quantile_priors_[i].alpha() >= quantile_priors_[i - 1].alpha()) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Quantile priors must have strictly decreasing exceedance "
                        "probabilities (Alpha).");
                }

                if (quantile_priors_[i].distribution().mean() <=
                    quantile_priors_[i - 1].distribution().mean()) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Quantile prior means must increase with decreasing "
                        "exceedance probability.");
                }

                if (quantile_priors_[i].distribution().inverse_cdf(0.05) <=
                        quantile_priors_[i - 1].distribution().inverse_cdf(0.05) ||
                    quantile_priors_[i].distribution().inverse_cdf(0.95) <=
                        quantile_priors_[i - 1].distribution().inverse_cdf(0.95)) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Quantile prior 5 percent and 95 percent bounds must "
                        "increase with decreasing exceedance probability.");
                }
            }
        }

        return result;
    }

    // C# `GenerateRandomValues(int sampleSize, int seed = -1)` (line 1528, ISimulatable):
    // guards, then delegates to Mixture.GenerateRandomValues -- the ported component-
    // selection Mersenne Twister sampler (seed > 0 deterministic, else clock-seeded).
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size <= 0)
            throw std::out_of_range("Sample size must be positive.");
        if (mixture_ == nullptr)
            throw std::runtime_error(
                "Mixture distribution cannot be null when generating random values.");

        return mixture_->generate_random_values(sample_size, seed);
    }

   protected:
    // C# `DataFrame_PropertyChanged` override (line 346): the explicit-invalidation
    // equivalent for in-place frame mutations (see the base header's cadence note).
    // Reprocess thresholds, sync the mixture's ZeroWeight, SetDefaultParameters when flat
    // priors. NEVER called from the likelihood paths.
    void data_frame_property_changed() override {
        if (!has_data_frame()) return;

        data_frame().process_threshold_series();

        if (mixture_ != nullptr) {
            if (is_zero_inflated_) {
                mixture_->set_zero_weight(data_frame().zero_value_relative_frequency());
            } else {
                mixture_->set_zero_weight(0.0);
            }
        }

        if (use_default_flat_priors()) set_default_parameters();
    }

   private:
    // C# `Mixture = (Mixture)distribution.Clone()`.
    static std::unique_ptr<Mixture> clone_mixture(const Mixture& distribution) {
        std::unique_ptr<DistributionBase> base = distribution.clone();
        return std::unique_ptr<Mixture>(static_cast<Mixture*>(base.release()));
    }

    // C# `Math.Round(value, 2)` (MidpointRounding.ToEven); the
    // univariate_distribution_model.hpp precedent.
    static double round_half_even_2(double value) {
        return std::nearbyint(value * 100.0) / 100.0;
    }

    std::unique_ptr<Mixture> mixture_;  // C# _mixture (line 195; nullptr == C# null)
    bool is_zero_inflated_ = false;     // C# _isZeroInflated (line 196)
};

}  // namespace corehydro::models
