// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/DistributionFitting/FittingAnalysis.cs @ fc28c0c
//
// The second concrete Analyses-layer port (A6): multi-distribution goodness-of-fit ranking. It
// fits each candidate distribution to the input DataFrame by MAXIMUM LIKELIHOOD (not Bayesian --
// that is A5's UnivariateAnalysis) and stores a FittedDistribution DTO (AIC / BIC / RMSE + a
// fit-succeeded flag) per candidate. Ranking is left to the consumer.
//
// CANDIDATE-LIST COUNT (15, matching C#). The C# DistributionList (C# 215-231) has 15 candidates
// and so does this port, in the C# order:
//   Exponential, GammaDistribution, GeneralizedExtremeValue, GeneralizedLogistic,
//   GeneralizedNormal, GeneralizedPareto, Gumbel, KappaFour, LnNormal, Logistic, LogNormal,
//   LogPearsonTypeIII, Normal, PearsonTypeIII, Weibull.
// Every downstream count (DistributionList / FittedDistributions size) is 15. (Through Phase 10
// this list held only 14: GeneralizedNormal existed in the type enum but had no factory case, so
// it was skipped. Porting the distribution and giving the factory its case restored the 15th
// candidate at its C# position, between GeneralizedLogistic and GeneralizedPareto.)
//
// OWNERSHIP (deviation from the C# GC references).
//   * The C# ctor takes a DataFrame GC reference (`?? throw ArgumentNullException`). Here the
//     analysis OWNS the frame; the ctor takes a `std::unique_ptr<DataFrame>` and null-guards it
//     (empty pointer -> `std::invalid_argument`, the ArgumentNullException analogue), mirroring
//     A5's UnivariateAnalysis ownership. The default ctor leaves the frame UNSET (the C# default
//     ctor leaves _dataFrame null and _fittedDistributions empty).
//   * Setting the frame runs `process_threshold_series()` then `clear_results()` (the C# DataFrame
//     setter, C# 152-166). Because a candidate list of owning unique_ptrs is not copyable, the
//     analysis is NON-copyable / NON-movable.
//
// RUN (deviations, documented):
//   * C# `async Task RunAsync(SafeProgressReporter?)` -> synchronous `void run()` (A4/A5 rule).
//   * C# `Parallel.For` over the candidates -> a SERIAL loop. The loop bodies are independent
//     (each fits its own model and stores by index), so ordering / results are unaffected; the
//     interlocked progress counter and the CancellationToken cooperative checks are dropped.
//   * The per-candidate try/catch (C# 383-401) is kept as a SILENT no-throw guard: on any
//     exception the candidate's ErrorMessage is set and the loop continues (a single candidate's
//     failure must NOT abort the whole run). The C# Debug.WriteLine trace becomes a comment.
//   * No cancellation path: the CancellationTokenSource / wasCanceled machinery is dropped, so
//     `set_is_estimated(true)` runs unconditionally after the loop (C# sets IsEstimated=true only
//     when not canceled; with cancellation removed that is always).
//
// SKIPPED (WPF/threading/serialization, no numerical content; dropped per A4/A5):
//   * `ToXElement()` and the `XElement` ctor (C# 72-129, 438) -- XML (de)serialization.
//   * `DataFrame_PropertyChanged` (C# 233) / `ProbabilityOrdinates_CollectionChanged` (C# 275) --
//     INotifyPropertyChanged / INotifyCollectionChanged handlers.
//   * `OnAnalysisStarting` / `OnAnalysisCompleted` / `AnalysisRunCompletedEventArgs`,
//     `SafeProgressReporter` progress reporting, `CancellationTokenSource` / `CancelAnalysis` --
//     WPF run-lifecycle / threading. (All inherited from / paired with AnalysisBase's dropped
//     surface.)
//   * every `RaisePropertyChange` call -- no notification system in this port.
#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/analyses/support/analysis_base.hpp"
#include "corehydro/analyses/support/i_probability_ordinates.hpp"
#include "corehydro/estimation/maximum_likelihood.hpp"
#include "corehydro/estimation/optimization_method.hpp"
#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/distribution_fitting/fitted_distribution.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/probability_ordinates.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"

namespace corehydro::analyses {

class FittingAnalysis : public AnalysisBase, public IProbabilityOrdinates {
   public:
    using DataFrame = corehydro::models::DataFrame;
    using FittedDistribution = corehydro::models::FittedDistribution;
    using ProbabilityOrdinates = corehydro::numerics::data::ProbabilityOrdinates;
    using UnivariateDistributionBase = corehydro::numerics::distributions::UnivariateDistributionBase;
    using UnivariateDistributionType = corehydro::numerics::distributions::UnivariateDistributionType;
    using UnivariateDistributionModel = corehydro::models::UnivariateDistributionModel;
    using MaximumLikelihood = corehydro::estimation::MaximumLikelihood;
    using OptimizationMethod = corehydro::estimation::OptimizationMethod;
    using GoodnessOfFit = corehydro::numerics::data::GoodnessOfFit;

    // C# default ctor `FittingAnalysis()` (C# 47): no data frame, empty fitted list. The candidate
    // list is still built (the C# field initializer runs regardless).
    FittingAnalysis() { build_distribution_list(); }

    // C# `FittingAnalysis(DataFrame)` (C# 60): `DataFrame = dataFrame ?? throw
    // ArgumentNullException`. Here the null-guard maps to the empty-unique_ptr case (see the file
    // header OWNERSHIP note). Setting the frame processes thresholds + clears results.
    explicit FittingAnalysis(std::unique_ptr<DataFrame> data_frame) {
        build_distribution_list();
        if (data_frame == nullptr) {
            throw std::invalid_argument("dataFrame");  // C# ArgumentNullException
        }
        set_data_frame(std::move(*data_frame));
    }

    ~FittingAnalysis() override = default;

    // Non-copyable / non-movable: the candidate list owns unique_ptrs (deviation).
    FittingAnalysis(const FittingAnalysis&) = delete;
    FittingAnalysis& operator=(const FittingAnalysis&) = delete;
    FittingAnalysis(FittingAnalysis&&) = delete;
    FittingAnalysis& operator=(FittingAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `DataFrame` (C# 143). Unguarded deref: check has_data_frame() where absence is possible.
    DataFrame& data_frame() { return *data_frame_; }
    const DataFrame& data_frame() const { return *data_frame_; }
    bool has_data_frame() const { return data_frame_ != nullptr; }

    // C# `ProbabilityOrdinates` (C# 170). IProbabilityOrdinates override (mutable reference).
    ProbabilityOrdinates& probability_ordinates() override { return probability_ordinates_; }
    const ProbabilityOrdinates& probability_ordinates() const { return probability_ordinates_; }

    // C# `FittedDistributions` (C# 195): one FittedDistribution per candidate (15).
    const std::vector<FittedDistribution>& fitted_distributions() const {
        return fitted_distributions_;
    }

    // C# `DistributionList` (C# 215): the 15 candidate distributions (owning). Exposed so
    // consumers / tests can inspect the candidate set and its types.
    const std::vector<std::unique_ptr<UnivariateDistributionBase>>& distribution_list() const {
        return distribution_list_;
    }

    // --- Methods ---------------------------------------------------------------------------

    // C# `ClearResults` (C# 250): rebuild the fitted list with one default (fit-not-succeeded)
    // FittedDistribution per candidate, then reset IsEstimated to false.
    void clear_results() {
        fitted_distributions_.clear();
        fitted_distributions_.reserve(distribution_list_.size());
        for (const auto& candidate : distribution_list_) {
            fitted_distributions_.emplace_back(candidate->clone());
        }
        set_is_estimated(false);
    }

    // C# `RunAsync` (C# 261), synchronous. Guards (unset frame / invalid config) -> clear_results
    // -> serial fit over the 15 candidates -> IsEstimated = true.
    void run() override {
        // C# `if (DataFrame == null) throw InvalidOperationException` (C# 264).
        if (!has_data_frame()) {
            throw std::runtime_error("DataFrame must be set before running the fitting analysis.");
        }
        // C# `if (Validate().IsValid == false) throw InvalidOperationException` (C# 267).
        if (!validate().is_valid) {
            throw std::runtime_error(
                "Analysis is not valid. Please check the configuration before running the "
                "analysis.");
        }

        clear_results();

        const int n = static_cast<int>(distribution_list_.size());
        for (int idx = 0; idx < n; ++idx) {
            const std::size_t i = static_cast<std::size_t>(idx);
            try {
                // C# `new UnivariateDistribution(DataFrame, DistributionList[idx])` -- the C++
                // model owns its frame, so hand it a deep clone of the analysis's frame.
                UnivariateDistributionModel model(data_frame_->clone(),
                                                  distribution_list_[i]->clone());
                model.set_use_jeffreys_rule_for_scale(false);  // To be consistent with MLE (C# 302).
                model.set_default_parameters();

                MaximumLikelihood mle(model, OptimizationMethod::DifferentialEvolution);
                mle.set_report_failure(true);
                mle.set_compute_hessian(false);
                mle.optimizer().max_iterations = 10000;
                mle.optimizer().max_function_evaluations = 100000;

                mle.estimate();

                if (mle.is_estimated()) {
                    // C# `dist.Distribution.SetParameters(mle.BestParameterSet.Values)` (C# 335).
                    model.distribution().set_parameters(mle.best_parameter_set().values);

                    double aic = mle.get_aic();
                    double bic = mle.get_bic(data_frame_->total_record_length());

                    // values = Exact ++ Uncertain ++ Interval (C# 340-342).
                    std::vector<double> values = data_frame_->exact_series().values_to_list();
                    {
                        std::vector<double> u = data_frame_->uncertain_series().values_to_list();
                        values.insert(values.end(), u.begin(), u.end());
                        std::vector<double> iv = data_frame_->interval_series().values_to_list();
                        values.insert(values.end(), iv.begin(), iv.end());
                    }

                    // probs = PlottingPositionComplement of Exact ++ Uncertain ++ Interval, same
                    // order (C# 344-346). NOTE the COMPLEMENT, not the raw plotting position.
                    std::vector<double> probs;
                    probs.reserve(values.size());
                    for (std::size_t k = 0; k < data_frame_->exact_series().count(); ++k)
                        probs.push_back(data_frame_->exact_series()[k].plotting_position_complement());
                    for (std::size_t k = 0; k < data_frame_->uncertain_series().count(); ++k)
                        probs.push_back(
                            data_frame_->uncertain_series()[k].plotting_position_complement());
                    for (std::size_t k = 0; k < data_frame_->interval_series().count(); ++k)
                        probs.push_back(
                            data_frame_->interval_series()[k].plotting_position_complement());

                    double rmse = GoodnessOfFit::rmse(values, probs, model.distribution());

                    // C# `Tools.IsFinite(aic) && Tools.IsFinite(bic) && Tools.IsFinite(rmse)`.
                    bool success = std::isfinite(aic) && std::isfinite(bic) && std::isfinite(rmse);

                    // ShowResults == success, per C# (C# 355-361).
                    fitted_distributions_[i] = FittedDistribution(model.distribution().clone(), aic,
                                                                  bic, rmse, success, success);
                } else {
                    // C# `new FittedDistribution(dist.Distribution.Clone())` (C# 366).
                    fitted_distributions_[i] = FittedDistribution(model.distribution().clone());
                }
            } catch (const std::exception& ex) {
                // C# Debug.WriteLine + record ErrorMessage (C# 383-401): silent no-throw guard so a
                // single candidate's failure does not abort the whole run. The default entry built
                // by clear_results() survives; only its diagnostic is set.
                fitted_distributions_[i].set_error_message(ex.what());
            }
        }

        // No cancellation path (deviation): IsEstimated becomes true unconditionally (C# 411).
        set_is_estimated(true);
    }

    // C# `Validate` (C# 442), const per the A4 IAnalysis contract. Combines the DataFrame and
    // ProbabilityOrdinates validations (invalid if either is; concatenate messages). Reads the
    // ProbabilityOrdinates member DIRECTLY -- the non-const interface accessor cannot be called
    // from this const method, and its validate() is const.
    corehydro::models::ValidationResult validate() const override {
        corehydro::models::ValidationResult result;

        // Validate the data frame. An unset frame is invalid (the C# would NPE on a null frame;
        // this port reports it instead of dereferencing).
        if (!has_data_frame()) {
            result.is_valid = false;
            result.validation_messages.push_back("DataFrame is not set.");
        } else {
            corehydro::models::ValidationResult data_valid = data_frame_->validate();
            if (!data_valid.is_valid) {
                result.is_valid = false;
                result.validation_messages.insert(result.validation_messages.end(),
                                                  data_valid.validation_messages.begin(),
                                                  data_valid.validation_messages.end());
            }
        }

        // Validate the probability ordinates.
        auto prob_valid = probability_ordinates_.validate();
        if (!prob_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              prob_valid.messages.begin(), prob_valid.messages.end());
        }

        return result;
    }

   private:
    // Builds the 15 candidate distributions, in the C# order (see the file header COUNT note).
    // Called from both constructors (the C# field initializer runs on every construction path).
    void build_distribution_list() {
        static const UnivariateDistributionType kCandidates[] = {
            UnivariateDistributionType::Exponential,
            UnivariateDistributionType::GammaDistribution,
            UnivariateDistributionType::GeneralizedExtremeValue,
            UnivariateDistributionType::GeneralizedLogistic,
            UnivariateDistributionType::GeneralizedNormal,
            UnivariateDistributionType::GeneralizedPareto,
            UnivariateDistributionType::Gumbel,
            UnivariateDistributionType::KappaFour,
            UnivariateDistributionType::LnNormal,
            UnivariateDistributionType::Logistic,
            UnivariateDistributionType::LogNormal,
            UnivariateDistributionType::LogPearsonTypeIII,
            UnivariateDistributionType::Normal,
            UnivariateDistributionType::PearsonTypeIII,
            UnivariateDistributionType::Weibull};

        distribution_list_.clear();
        for (UnivariateDistributionType type : kCandidates) {
            distribution_list_.push_back(
                corehydro::numerics::distributions::create_distribution(type));
        }
    }

    // C# DataFrame setter (C# 152-166): assign, ProcessThresholdSeries, ClearResults. The
    // PropertyChanged raise and the DataFrame_PropertyChanged (un)subscription are dropped.
    void set_data_frame(DataFrame data_frame) {
        data_frame_ = std::make_unique<DataFrame>(std::move(data_frame));
        data_frame_->process_threshold_series();
        clear_results();
    }

    // Owned data frame (deviation); nullptr when unset. unique_ptr so DataFrame (move-only) can be
    // held optionally without an extra flag.
    std::unique_ptr<DataFrame> data_frame_;
    ProbabilityOrdinates probability_ordinates_;
    std::vector<FittedDistribution> fitted_distributions_;
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distribution_list_;
};

}  // namespace corehydro::analyses
