// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataFrame.cs @ c2e6192
//
// The input data frame containing exact, uncertain, interval, and threshold data series --
// the data container every univariate model consumes. This class satisfies the forward
// declaration in models/support/i_univariate_model.hpp.
//
// INVALIDATION STRATEGY (replaces the C# INPC event plumbing). Upstream, every series
// CollectionChanged / item PropertyChanged event re-ran CalculatePlottingPositions (which
// first runs ProcessThresholdSeries) and, for the exact series, CalculateLambda. The C++
// port has no events; the replacement is:
//   1. full_time_series() recomputes LAZILY on access: it rebuilds via
//      create_full_time_series() whenever the cached list size no longer matches
//      TotalRecordLength() -- exactly the C# getter's own rebuild condition. The upstream
//      double-checked-lock / Volatile.Read machinery simplifies to this plain lazy rebuild
//      because the C++ port is single-threaded by design.
//   2. Threshold-derived state (ThresholdData::NumberBelow, and the zeroing of
//      NumberAbove for fully covered windows) is recomputed EXPLICITLY: call
//      process_threshold_series() after mutating any series or threshold counts. Upstream
//      this ran automatically on every collection change; calculate_plotting_positions()
//      calls it first (as the C# method itself does, lines 1142-1144), restoring the
//      upstream trigger point for plotting-position consumers.
//   3. Lambda is recomputed EXPLICITLY via calculate_lambda() (upstream: triggered by
//      exact-series collection-changed events) or pinned via set_lambda().
//
// Ported surface: the four owned series, FullTimeSeries/CreateFullTimeSeries,
// ProcessThresholdSeries, TotalRecordLength, ZeroValueRelativeFrequency, Lambda
// (SetLambda/CalculateLambda), the low-outlier surface (NumberOfLowOutliers,
// LowOutlierThreshold, ClearLowOutliers, SetLowOutliersFromMGBT,
// SetLowOutliersFromThreshold), the validating PlottingParameter property (BestFit v2.0.0;
// see below), Validate(), a
// direct deep Clone() (the C# clones via an XElement round trip; the direct clone has the
// same observable result, including the empty lazily-rebuilt full series noted in the C#
// remarks), (M5) CalculatePlottingPositions / ApplyLangbeinConversion -- the
// Hirsch-Stedinger censored plotting positions, defined out-of-line in
// data_frame_plotting.hpp (included at the bottom of this file) -- and (A3) the
// bootstrap/resampling surface: JackKnife, Resample, BootstrapDataFrame, and
// ShiftDistribution (the `#region Bootstrap Methods`, DataFrame.cs 2059-2543).
// Invalidation contract: BootstrapDataFrame and Resample call process_threshold_series()
// as the last (or only-when-create_full_time_series) step matching the C#; the C#
// SuppressCollectionChanged event-suppression lines have no C++ equivalent (no events) and
// are dropped. ShiftDistribution is C# `private static`; the C++ port exposes it
// `public static` (access modifier only, per the ThresholdData::set_number_below
// internal->public precedent) so the arm-by-arm shift is directly testable.
//
// P4 TASK 5 + P5 TASK 6 (hypothesis-test and summary-statistics facades): all eleven C#
// `#region Hypothesis Testing` members are ported over the numerics::data::
// hypothesis_tests free functions -- jarque_bera_test, ljung_box_test,
// equal_variance_t_test, unequal_variance_t_test, f_test, linear_trend_test,
// wald_wolfowitz_test, mann_whitney_test, mann_kendall_test -- and all three
// `#region Summary Statistics` members that were previously deferred --
// summary_statistics_exact_data_only, summary_statistics_all_data, and
// set_standardized_values (GetNonparametricMoments/GetNonparametricMomentsROS were
// already ported additively in B9; the two summary methods and set_standardized_values
// share their `create_empirical_distribution_with_unique_values` private static tail).
// P5 landed the two P4 had deferred, together in one commit as planned: unimodality_test
// and summary_hypothesis_test, both un-gated by
// numerics/machine_learning/unsupervised/gaussian_mixture_model.hpp. summary_hypothesis_test
// was the consequential one to defer correctly, because it calls all ten hypothesis-test
// facades inside ONE try/catch that NaNs the entire ten-key result dictionary if ANY single
// call throws -- shipping it earlier with a throwing UnimodalityTest arm would have silently
// NaN'd nine otherwise-working results rather than surfacing the real gap. Both C# regions
// are now ported in full.
//
// P6 landed the last two this file had deferred: create_block_series and
// create_peaks_over_threshold_series, both un-gated by the ported
// numerics/data/time_series/time_series.hpp container.
//
// Deliberately NOT ported (unrelated to the above):
//   - CreateFromUSGS + USGSRawText (network import)
//   - XML (ToXElement / XElement constructor), INotifyPropertyChanged, and the
//     concurrency machinery (_syncRoot, Volatile, SnapshotNonNull)
//
// BESTFIT v2.0.0 ADDITIONS (T12): CalculatePlottingPositions was rewritten to the
// peakFQ-faithful ARRANGE2/PPLOT2/PLPOS scheme -- see data_frame_plotting.hpp's file header
// for the algorithm summary and the strict-validation / tie-ordering notes.
// ProcessThresholdSeries became idempotent via the ThresholdData source/effective count
// split (see that header's own note) -- process_threshold_series() below now reads
// source_number_above() rather than number_above() and writes through
// set_processed_counts(), so calling it more than once, or after further series mutation,
// is safe. plotting_position_version() exposes a monotonically increasing counter bumped
// once per calculate_plotting_positions() call (PORTED NON-GUI CORE ONLY -- see its own
// accessor comment for what is deliberately not ported: the XML round-trip stamp and the
// wider per-mutation event bump). set_plotting_parameter() now validates eagerly (throws
// std::out_of_range, C# ArgumentOutOfRangeException) instead of accepting any value.
// get_nonparametric_moments()/get_nonparametric_moments_ros() gained the
// CreateEmpiricalDistributionWithUniqueValues repeated-X-value dedupe (a private static
// helper below) before constructing their EmpiricalDistribution, returning an empty
// optional for a degenerate (fewer than two distinct values) sample instead of
// constructing an invalid one-point distribution.
//
// FOLLOW-UPS (M5 ledger finding): create_full_time_series() below sorts the combined series
// with std::stable_sort, while M5 proved the .NET List<T>.Sort introsort tie order IS
// oracle-visible in plotting positions (data_frame_plotting.hpp carries the faithful
// detail::dotnet_list_sort port for its three internal sorts for exactly that reason). No
// tie-sensitive oracle exercises THIS sort today, but a future oracle pinned to a frame with
// equal-index ties could trip on the tie order; if one does, switch this sort to the
// dotnet_list_sort port rather than loosening the fixture. Do not change the sort without a
// tie-sensitive oracle proving which order the C# produces here.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "corehydro/models/data_frame/data_collections/data_series.hpp"
#include "corehydro/models/data_frame/data_collections/exact_series.hpp"
#include "corehydro/models/data_frame/data_collections/interval_series.hpp"
#include "corehydro/models/data_frame/data_collections/threshold_series.hpp"
#include "corehydro/models/data_frame/data_collections/uncertain_series.hpp"
#include "corehydro/models/data_frame/data_types/data.hpp"
#include "corehydro/models/data_frame/data_types/exact_data.hpp"
#include "corehydro/models/data_frame/data_types/interval_data.hpp"
#include "corehydro/models/data_frame/data_types/threshold_data.hpp"
#include "corehydro/models/data_frame/data_types/uncertain_data.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/numerics/data/hypothesis_tests.hpp"
#include "corehydro/numerics/data/time_series/support/time_block_window.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "corehydro/numerics/data/multiple_grubbs_beck_test.hpp"
#include "corehydro/numerics/data/regression/linear_regression.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/binomial.hpp"
#include "corehydro/numerics/distributions/empirical_distribution.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/generalized_beta.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/log_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/pert.hpp"
#include "corehydro/numerics/distributions/student_t.hpp"
#include "corehydro/numerics/distributions/triangular.hpp"
#include "corehydro/numerics/distributions/truncated_normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::models {

class DataFrame {
   public:
    // Constructs an empty data frame (C# line 32; the event hookups are not ported).
    DataFrame() = default;

    // The series are owning containers, so the frame is move-only; use clone() for a
    // deep copy (mirrors the C# Clone()).
    DataFrame(const DataFrame&) = delete;
    DataFrame& operator=(const DataFrame&) = delete;
    DataFrame(DataFrame&&) noexcept = default;
    DataFrame& operator=(DataFrame&&) noexcept = default;

    // --- The exact data series collection (C# property, line 128). ---
    ExactSeries& exact_series() { return exact_series_; }
    const ExactSeries& exact_series() const { return exact_series_; }
    void set_exact_series(ExactSeries series) { exact_series_ = std::move(series); }

    // --- The uncertain data series collection (C# property, line 143). ---
    UncertainSeries& uncertain_series() { return uncertain_series_; }
    const UncertainSeries& uncertain_series() const { return uncertain_series_; }
    void set_uncertain_series(UncertainSeries series) {
        uncertain_series_ = std::move(series);
    }

    // --- The interval data series collection (C# property, line 158). ---
    IntervalSeries& interval_series() { return interval_series_; }
    const IntervalSeries& interval_series() const { return interval_series_; }
    void set_interval_series(IntervalSeries series) { interval_series_ = std::move(series); }

    // --- The threshold data series collection (C# property, line 173). ---
    ThresholdSeries& threshold_series() { return threshold_series_; }
    const ThresholdSeries& threshold_series() const { return threshold_series_; }
    void set_threshold_series(ThresholdSeries series) {
        threshold_series_ = std::move(series);
    }

    // The full time series in chronological order (C# property, line 203). Lazy: rebuilds
    // when the cached size no longer matches TotalRecordLength() -- the C# getter's own
    // rebuild condition, minus the locking (see the invalidation-strategy note). The
    // returned list must be treated as read-only. Const (M9): the C# getter is reachable
    // from read-only paths (the nonstationary likelihoods evaluate it on every call), so
    // the cache member is `mutable` and the getter/rebuild are logically const.
    const std::vector<std::unique_ptr<Data>>& full_time_series() const {
        int total = total_record_length();
        if (total == 0 || static_cast<int>(full_time_series_.size()) == total)
            return full_time_series_;
        create_full_time_series();
        return full_time_series_;
    }

    // Returns the number of low outliers in the exact data series (C# line 224).
    int number_of_low_outliers() const { return number_of_low_outliers_; }

    // --- The low outlier threshold value (C# property, line 236). ---
    double low_outlier_threshold() const { return low_outlier_threshold_; }
    void set_low_outlier_threshold(double threshold) { low_outlier_threshold_ = threshold; }

    // --- The plotting position parameter; default 0.0 = Weibull (C# property, line 256;
    // alternatives: 0.40 Cunnane, 0.44 Gringorten, 0.50 Hazen). The setter validates
    // eagerly (C# throws ArgumentOutOfRangeException for a non-finite value or one outside
    // [0, 1), BestFit v2.0.0) and, on a real change, recomputes plotting positions. The C#
    // recalculation now goes through RecalculatePlottingPositionsAfterEdit, which swallows
    // an InvalidOperationException raised by a transiently-invalid threshold state while an
    // interactive GUI edit is in progress -- a WPF-only concern with no C++ counterpart (see
    // the file header's SKIP note), so this setter calls calculate_plotting_positions()
    // directly and lets any throw propagate, matching every other explicit-trigger method
    // on this class. ---
    double plotting_parameter() const { return plotting_parameter_; }
    void set_plotting_parameter(double plotting_parameter) {
        if (!numerics::is_finite(plotting_parameter) || plotting_parameter < 0.0 ||
            plotting_parameter >= 1.0) {
            throw std::out_of_range(
                "The plotting parameter must be finite, greater than or equal to zero, "
                "and less than one.");
        }
        if (plotting_parameter_ != plotting_parameter) {
            plotting_parameter_ = plotting_parameter;
            calculate_plotting_positions();
        }
    }

    // The version stamp for data-frame plotting-position inputs (C# internal
    // PlottingPositionVersion, line 168; BestFit v2.0.0). Bumped once per
    // calculate_plotting_positions() call so consumers (Task 15's BivariateDistribution
    // pseudo-observation cache) can detect a stale cache without rescanning the series.
    // PORTED NON-GUI CORE ONLY: the C# XML-round-trip stamp handling (ToXElement /
    // FromXElement recomputation) and the wider event-driven bump on every series/item
    // mutation (Interlocked.Increment inside the ExactSeries/UncertainSeries/
    // IntervalSeries/ThresholdSeries property setters and their PropertyChanged handlers)
    // are WPF/XML-only concerns with no C++ counterpart -- this port's invalidation
    // strategy already requires an explicit calculate_plotting_positions() call after any
    // mutation (see the file header), so bumping there alone is sufficient and observably
    // equivalent for every caller that follows the documented invalidation contract.
    //
    // CORRECTION (Task 15 finding): the "WPF/XML-only" characterization above is NOT
    // fully accurate -- the C# `ExactDataChanged`/`UncertainDataChanged`/`IntervalDataChanged`/
    // `ThresholdDataChanged` handlers ALSO bump `_plottingPositionVersion` (via
    // `Interlocked.Increment`, no recalculation) whenever a bare `Data.PlottingPosition`
    // property write fires `PropertyChanged`, independent of any WPF binding -- so in the
    // real C#, directly mutating a single `ExactData.PlottingPosition` (bypassing
    // `CalculatePlottingPositions()` entirely) DOES invalidate a version-keyed cache such as
    // `BivariateDistribution`'s PseudoLikelihood validation cache
    // (bivariate_distribution.hpp). This port's version counter does NOT catch that specific
    // bare-mutation case (`Data::set_plotting_position()` has no back-pointer to bump its
    // owning DataFrame's counter), so a caller here MUST re-run
    // `calculate_plotting_positions()` explicitly for a version-keyed cache to observe the
    // change -- a real, documented behavior gap versus the C#, not exercised by any current
    // fixture (BivariateDistribution's only oracle-verified PseudoLikelihood scenario is the
    // first-validation auto-repair, which this counter handles correctly).
    std::int64_t plotting_position_version() const { return plotting_position_version_; }

    // The average number of events per index (C# line 273).
    double lambda() const { return lambda_; }

    // Sets the lambda value directly without calculation (C# line 1938).
    void set_lambda(double lambda) { lambda_ = lambda; }

    // Calculates the average number of events per index (C# line 1947): exact-series
    // count over its index span, 0 when either is empty. Explicit trigger -- see the
    // invalidation-strategy note.
    void calculate_lambda() {
        double events = static_cast<double>(exact_series_.count());
        double span = static_cast<double>(exact_series_.index_span());
        lambda_ = (events <= 0.0 || span <= 0.0) ? 0.0 : events / span;
    }

    // Hirsch-Stedinger censored plotting positions (C# CalculatePlottingPositions,
    // line 1140). Defined in data_frame_plotting.hpp (included below); calls
    // process_threshold_series() first, exactly like the C#.
    void calculate_plotting_positions();

    // Apply the Langbein conversion to the plotting positions (C#
    // ApplyLangbeinConversion, line 1458). Defined in data_frame_plotting.hpp.
    void apply_langbein_conversion(double lambda);

    // Validates the current state of the data frame and reports any issues found
    // (C# line 527): plotting-parameter range plus the four series validations, in the
    // C# order.
    ValidationResult validate() const {
        ValidationResult result;

        // Unreachable via set_plotting_parameter() (which now validates eagerly, BestFit
        // v2.0.0), but kept for structural parity with the C# -- there the raw
        // `_plottingParameter` field can still be populated out of range via the
        // (unported) XML deserialization constructor, bypassing the property setter.
        if (!numerics::is_finite(plotting_parameter_) || plotting_parameter_ < 0.0 ||
            plotting_parameter_ >= 1.0) {
            result.validation_messages.push_back(
                "Error: The plotting parameter must be finite, greater than or equal to 0, "
                "and less than 1.");
            result.is_valid = false;
        }

        append(result, exact_series_.validate());
        append(result, uncertain_series_.validate(this));
        append(result, interval_series_.validate(this));
        append(result, threshold_series_.validate());

        return result;
    }

    // Returns the total record length of the data frame (C# line 573): explicit data
    // points plus each threshold's NumberBelow + NumberAbove (call
    // process_threshold_series() first after mutations).
    int total_record_length() const {
        int n = static_cast<int>(exact_series_.count() + uncertain_series_.count() +
                                 interval_series_.count());
        for (std::size_t i = 0; i < threshold_series_.count(); i++) {
            n += threshold_series_[i].number_below() + threshold_series_[i].number_above();
        }
        return n;
    }

    // Computes the relative frequency of data points less than or equal to zero
    // (C# line 587).
    double zero_value_relative_frequency() const {
        double total_count = 0;
        double total_zero_count = 0;
        if (exact_series_.count() > 0) {
            total_count = static_cast<double>(exact_series_.count());
            for (std::size_t i = 0; i < exact_series_.count(); i++)
                if (exact_series_[i].value() <= 0.0) total_zero_count += 1;
        }
        if (uncertain_series_.count() > 0) {
            total_count += static_cast<double>(uncertain_series_.count());
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                if (uncertain_series_[i].value() <= 0.0) total_zero_count += 1;
        }
        if (interval_series_.count() > 0) {
            total_count += static_cast<double>(interval_series_.count());
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                if (interval_series_[i].value() <= 0.0) total_zero_count += 1;
        }
        if (total_count == 0) return 0.0;
        return total_zero_count / total_count;
    }

    // Process the threshold data to ensure exclusivity by adjusting counts for
    // overlapping data (C# line 618): NumberBelow = Duration - source NumberAbove -
    // (explicit interval/uncertain/exact points inside the window), clamped at 0; the
    // EFFECTIVE NumberAbove is zeroed when the explicit points account for every remaining
    // year (nBelow == 0). Idempotent (BestFit v2.0.0): each pass starts from
    // source_number_above() -- the stable user-supplied input -- rather than from
    // number_above() (a previous pass's possibly-already-reduced effective value), and
    // writes the result via set_processed_counts() rather than the public setter, so
    // repeated calls (or a mutation of the data series followed by a fresh call) never
    // compound the reduction across multiple passes.
    void process_threshold_series() {
        for (std::size_t i = 0; i < threshold_series_.count(); i++) {
            ThresholdData& threshold_data = threshold_series_[i];
            int n_above = threshold_data.source_number_above();
            int n_below = threshold_data.duration() - n_above;
            // Check interval data
            for (std::size_t j = 0; j < interval_series_.count(); j++) {
                if (interval_series_[j].index() >= threshold_data.start_index() &&
                    interval_series_[j].index() <= threshold_data.end_index()) {
                    n_below -= 1;
                }
            }
            // Check uncertain data
            for (std::size_t j = 0; j < uncertain_series_.count(); j++) {
                if (uncertain_series_[j].index() >= threshold_data.start_index() &&
                    uncertain_series_[j].index() <= threshold_data.end_index()) {
                    n_below -= 1;
                }
            }
            // Check exact data
            for (std::size_t j = 0; j < exact_series_.count(); j++) {
                if (exact_series_[j].index() >= threshold_data.start_index() &&
                    exact_series_[j].index() <= threshold_data.end_index()) {
                    n_below -= 1;
                }
            }
            // Zero out the effective NumberAbove when explicit data account for every
            // remaining year.
            threshold_data.set_processed_counts(n_below == 0 ? 0 : n_above,
                                                std::max(0, n_below));
        }
    }

    // Creates a full time series in chronological order by expanding threshold data into
    // per-index left/right-censored clones and combining all series (C# line 669; the
    // concurrency snapshot/retry machinery is not ported). Const (M9): only refreshes the
    // mutable full-time-series cache, so the lazy const getter above can call it.
    void create_full_time_series() const {
        std::vector<std::unique_ptr<Data>> new_list;

        // Occupied indexes (Exact, Interval, Uncertain), built once per call as upstream.
        std::unordered_set<int> occupied;
        for (std::size_t k = 0; k < exact_series_.count(); k++)
            occupied.insert(exact_series_[k].index());
        for (std::size_t k = 0; k < interval_series_.count(); k++)
            occupied.insert(interval_series_[k].index());
        for (std::size_t k = 0; k < uncertain_series_.count(); k++)
            occupied.insert(uncertain_series_[k].index());

        // Threshold data
        for (std::size_t i = 0; i < threshold_series_.count(); i++) {
            const ThresholdData& threshold = threshold_series_[i];
            // Left (below) thresholds
            for (int j = threshold.start_index();
                 j <= threshold.end_index() - threshold.number_above(); j++) {
                if (occupied.count(j) != 0) continue;
                auto t_data = std::make_unique<ThresholdData>(threshold.clone());
                t_data->set_start_index(j);
                t_data->set_end_index(j);
                t_data->set_number_above(0);
                t_data->set_number_below(1);
                new_list.push_back(std::move(t_data));
            }
            // Right (above) thresholds
            for (int j = threshold.end_index() - threshold.number_above() + 1;
                 j <= threshold.end_index(); j++) {
                if (occupied.count(j) != 0) continue;
                auto t_data = std::make_unique<ThresholdData>(threshold.clone());
                t_data->set_start_index(j);
                t_data->set_end_index(j);
                t_data->set_number_above(1);
                t_data->set_number_below(0);
                new_list.push_back(std::move(t_data));
            }
        }
        // Exact data
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            new_list.push_back(std::make_unique<ExactData>(exact_series_[i].clone()));
        // Uncertain data
        for (std::size_t i = 0; i < uncertain_series_.count(); i++)
            new_list.push_back(std::make_unique<UncertainData>(uncertain_series_[i].clone()));
        // Interval data
        for (std::size_t i = 0; i < interval_series_.count(); i++)
            new_list.push_back(std::make_unique<IntervalData>(interval_series_[i].clone()));

        // Sort data by index (stable, so equal indexes keep the threshold -> exact ->
        // uncertain -> interval insertion order deterministically; the C# unstable
        // List<T>.Sort leaves tie order unspecified).
        std::stable_sort(new_list.begin(), new_list.end(),
                         [](const std::unique_ptr<Data>& x, const std::unique_ptr<Data>& y) {
                             return x->index() < y->index();
                         });

        full_time_series_ = std::move(new_list);
    }

    // Clear the low outlier results (C# line 794).
    void clear_low_outliers() {
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            exact_series_[i].set_is_low_outlier(false);
        number_of_low_outliers_ = 0;
    }

    // Estimates and sets the low outliers using the Multiple Grubbs Beck Test (MGBT);
    // exact data only (C# line 805). Throws std::invalid_argument (C# ArgumentException)
    // when the exact series has errors or fewer than 10 items.
    void set_low_outliers_from_mgbt() {
        if (!exact_series_.validate().is_valid)
            throw std::invalid_argument("The exact data series has errors.");
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before evaluating low "
                "outliers.");

        clear_low_outliers();
        low_outlier_threshold_ = 0;

        std::vector<double> values = exact_series_.values_to_list();

        // Compute the number of low outliers using the Multiple Grubbs Beck Test.
        number_of_low_outliers_ = numerics::data::MultipleGrubbsBeckTest::function(values);

        // Set the threshold value as the first value larger than N.
        std::sort(values.begin(), values.end());
        low_outlier_threshold_ =
            number_of_low_outliers_ > 0
                ? values[static_cast<std::size_t>(number_of_low_outliers_)]
                : 0.0;

        // Flag every exact data point below the threshold.
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            exact_series_[i].set_is_low_outlier(exact_series_[i].value() <
                                                low_outlier_threshold_);
    }

    // Estimates and sets the low outliers using the low outlier threshold value; exact
    // data only (C# line 852). Throws std::invalid_argument (C# ArgumentException) when
    // the exact series has errors, has fewer than 10 items, or the threshold would censor
    // more than 50 percent of the values.
    void set_low_outliers_from_threshold() {
        if (!exact_series_.validate().is_valid)
            throw std::invalid_argument("The exact data series has errors.");
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before evaluating low "
                "outliers.");
        if (low_outlier_threshold_ > exact_series_.upper_middle_value())
            throw std::invalid_argument(
                "The low outlier threshold value cannot be set to a value that would "
                "censor more than 50 percent of the values.");

        number_of_low_outliers_ = 0;
        for (std::size_t i = 0; i < exact_series_.count(); i++) {
            bool is_low = exact_series_[i].value() < low_outlier_threshold_;
            exact_series_[i].set_is_low_outlier(is_low);
            if (is_low) number_of_low_outliers_ += 1;
        }
    }

    // --- Hypothesis-test facades (C# #region Hypothesis Testing, lines 927-1082; P4 Task
    // 5). Every facade below reads ExactSeries ONLY -- no censoring filter, no threshold
    // machinery -- and selects log10_value() vs value() on use_log10, exactly mirroring
    // the C# `useLog10 ? ... Log10Value ... : ... Value` ternaries. The two-sample splits
    // filter on Data::index() (matching C#'s `x.Index < index` / `x.Index >= index`), NOT
    // on array position, so a non-contiguous or out-of-order index still splits correctly.
    // unimodality_test and summary_hypothesis_test landed in P5 alongside the
    // GaussianMixtureModel port; the region is complete. ---

    // Returns the sample of the exact series' Value or Log10Value column, in series order
    // (the common `useLog10 ? Select(Log10Value) : Select(Value)` projection every
    // single-sample facade below performs).
    std::vector<double> exact_sample(bool use_log10) const {
        std::vector<double> sample;
        sample.reserve(exact_series_.count());
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            sample.push_back(use_log10 ? exact_series_[i].log10_value()
                                       : exact_series_[i].value());
        return sample;
    }

    // Splits the exact series' Value or Log10Value column into two samples at `index`,
    // filtering on Data::index() -- `< index` goes to the first sample, `>= index` to the
    // second (the common two-sample projection every two-sample facade below performs).
    std::pair<std::vector<double>, std::vector<double>> split_exact_sample(
        int index, bool use_log10) const {
        std::vector<double> sample1;
        std::vector<double> sample2;
        for (std::size_t i = 0; i < exact_series_.count(); i++) {
            double v = use_log10 ? exact_series_[i].log10_value() : exact_series_[i].value();
            if (exact_series_[i].index() < index)
                sample1.push_back(v);
            else
                sample2.push_back(v);
        }
        return {std::move(sample1), std::move(sample2)};
    }

    // The Jarque-Bera test for normality; exact data only (C# JarqueBeraTest, line 936).
    double jarque_bera_test(bool use_log10 = false) const {
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before performing "
                "hypothesis tests.");
        return numerics::data::hypothesis_tests::jarque_bera_test(exact_sample(use_log10));
    }

    // The Ljung-Box test for nonzero autocorrelation; exact data only (C# LjungBoxTest,
    // line 949).
    double ljung_box_test(int lag_max = -1, bool use_log10 = false) const {
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before performing "
                "hypothesis tests.");
        return numerics::data::hypothesis_tests::ljung_box_test(exact_sample(use_log10),
                                                                 lag_max);
    }

    // The equal-variance (Student's) t-test for a difference in two sample means; exact
    // data only (C# EqualVarianceTtest, line 962). Throws "Invalid index." when the split
    // at `index` leaves either side with fewer than 2 points.
    double equal_variance_t_test(int index, bool use_log10 = false) const {
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before performing "
                "hypothesis tests.");
        auto [sample1, sample2] = split_exact_sample(index, use_log10);
        if (sample1.size() < 2 || sample2.size() < 2)
            throw std::invalid_argument("Invalid index.");
        return numerics::data::hypothesis_tests::equal_variance_t_test(sample1, sample2);
    }

    // The unequal-variance (Welch's) t-test for a difference in two sample means; exact
    // data only (C# UnequalVarianceTtest, line 978). Throws "Invalid index." when the
    // split at `index` leaves either side with fewer than 2 points.
    double unequal_variance_t_test(int index, bool use_log10 = false) const {
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before performing "
                "hypothesis tests.");
        auto [sample1, sample2] = split_exact_sample(index, use_log10);
        if (sample1.size() < 2 || sample2.size() < 2)
            throw std::invalid_argument("Invalid index.");
        return numerics::data::hypothesis_tests::unequal_variance_t_test(sample1, sample2);
    }

    // The F-test for a difference in two sample variances; exact data only (C# Ftest,
    // line 994). Throws "Invalid index." when the split at `index` leaves either side
    // with fewer than 2 points.
    double f_test(int index, bool use_log10 = false) const {
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before performing "
                "hypothesis tests.");
        auto [sample1, sample2] = split_exact_sample(index, use_log10);
        if (sample1.size() < 2 || sample2.size() < 2)
            throw std::invalid_argument("Invalid index.");
        return numerics::data::hypothesis_tests::f_test(sample1, sample2);
    }

    // The linear trend test for stationarity (trend); exact data only (C#
    // LinearTrendTest, line 1010). NOTE (transcription note 1): the C# method does NOT
    // call the equivalent `HypothesisTests.LinearTrendTest` static (which exists and is
    // identical), instead inlining the LinearRegression + StudentT computation itself; it
    // also computes a local `d` that is assigned but never used and then recomputes the
    // identical expression inline in the `return`. Both oddities are mirrored here rather
    // than "cleaned up" to a call through numerics::data::hypothesis_tests::linear_trend_test.
    double linear_trend_test(bool use_log10 = false) const {
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before performing "
                "hypothesis tests.");
        std::vector<double> indexes;
        indexes.reserve(exact_series_.count());
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            indexes.push_back(static_cast<double>(exact_series_[i].index()));
        std::vector<double> values = exact_sample(use_log10);

        numerics::math::linalg::Matrix x_vals(static_cast<int>(indexes.size()), 1, indexes);
        numerics::math::linalg::Vector y_vals(values);
        numerics::data::regression::LinearRegression lm(x_vals, y_vals, true);
        numerics::distributions::StudentT tdist(static_cast<double>(lm.degrees_of_freedom()));
        double d = std::fabs(lm.parameters()[1] / lm.parameter_standard_errors()[1]);
        (void)d;  // upstream computes this and never uses it -- see the note above.
        return (1.0 - tdist.cdf(std::fabs(lm.parameters()[1] /
                                          lm.parameter_standard_errors()[1]))) *
               2.0;
    }

    // The Wald-Wolfowitz runs test for independence and stationarity; exact data only (C#
    // WaldWolfowitzTest, line 1050).
    double wald_wolfowitz_test(bool use_log10 = false) const {
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before performing "
                "hypothesis tests.");
        return numerics::data::hypothesis_tests::wald_wolfowitz_test(exact_sample(use_log10));
    }

    // The Mann-Whitney U test for homogeneity and stationarity (jump); exact data only
    // (C# MannWhitneyTest, line 1063). Requires at least 20 exact points (a distinct,
    // higher floor than every other facade's 10) and throws "Invalid index." when the
    // split at `index` leaves either side with fewer than 3 points. The smaller sample is
    // always passed first, matching C#'s `sample1.Count <= sample2.Count` dispatch.
    double mann_whitney_test(int index, bool use_log10 = false) const {
        if (exact_series_.count() < 20)
            throw std::invalid_argument(
                "The exact data series must have at least 20 items before performing this "
                "hypothesis test.");
        auto [sample1, sample2] = split_exact_sample(index, use_log10);
        if (sample1.size() < 3 || sample2.size() < 3)
            throw std::invalid_argument("Invalid index.");
        return sample1.size() <= sample2.size()
                   ? numerics::data::hypothesis_tests::mann_whitney_test(sample1, sample2)
                   : numerics::data::hypothesis_tests::mann_whitney_test(sample2, sample1);
    }

    // The Mann-Kendall test for homogeneity and stationarity (trend); exact data only (C#
    // MannKendallTest, line 1078).
    double mann_kendall_test(bool use_log10 = false) const {
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before performing "
                "hypothesis tests.");
        return numerics::data::hypothesis_tests::mann_kendall_test(exact_sample(use_log10));
    }

    // The Gaussian-mixture-model test for unimodality; exact data only (C# UnimodalityTest,
    // line 1021). P5: deferred at P4 because it needs the then-unported
    // Numerics.MachineLearning.GaussianMixtureModel. A SMALL p-value is evidence against
    // unimodality; a numerically failed mixture fit returns NaN rather than throwing (the
    // catch lives in the numerics static, matching C#).
    double unimodality_test(bool use_log10 = false) const {
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before performing "
                "hypothesis tests.");
        return numerics::data::hypothesis_tests::unimodality_test(exact_sample(use_log10));
    }

    // Returns all ten hypothesis-test p-values in one ordered key/value list (C#
    // SummaryHypothesisTest, line 1077). P5: deferred at P4 alongside unimodality_test,
    // which it calls.
    //
    // The result is an ORDERED list, not a map: C# builds a Dictionary<string, double> whose
    // INSERTION ORDER is what the GUI and the fixture (which selects by label) read.
    //
    // Five transcription notes, each an upstream oddity a "cleanup" would silently change:
    //
    //  1. All ten calls sit in ONE try/catch that CLEARS the dictionary and refills all ten
    //     keys with NaN if ANY single call throws. This is exactly why P4 could not ship the
    //     method with a throwing unimodality arm -- it would have silently NaN'd nine
    //     otherwise-working results.
    //  2. The split index is clamped with
    //     `if (index < 0 || index <= minIndex || index > maxIndex) index =
    //      ExactSeries[(int)((double)values.Length / 2)].Index;`
    //     -- the clamp reads `values.Length`, which under use_log10 counts only the POSITIVE
    //     values, but then indexes ExactSeries, which is NOT filtered. The mismatch is
    //     reproduced.
    //  3. Under use_log10 the `indexes`/`values` arrays and the v1/v2 splits filter on
    //     `Value > 0`; the real-space branch does not. The two branches are not symmetric.
    //  4. LjungBoxTest is called with the DEFAULT lagMax here, unlike the standalone
    //     ljung_box_test facade, which takes one.
    //  5. The Mann-Whitney call builds each argument with its own ternary
    //     (`v1.Count <= v2.Count ? v1 : v2` and `v1.Count > v2.Count ? v1 : v2`). That reads
    //     oddly next to the standalone facade's single ternary but is equivalent at every
    //     count relationship -- checked against the actual semantics and recorded in
    //     docs/upstream-csharp-issues.md rather than re-derived here.
    std::vector<std::pair<std::string, double>> summary_hypothesis_test(
        int index = -1, bool use_log10 = false) const {
        static const char* const kKeys[10] = {
            "Jarque-Bera test for normality",
            "Ljung-Box test for independence",
            "Wald-Wolfowitz test for independence and stationarity (trend)",
            "Mann-Whitney test for homogeneity and stationarity (jump)",
            "Mann-Kendall test for homogeneity and stationarity (trend)",
            "Linear trend test for stationarity (trend)",
            "Equal variance t-test for differences in the means of two samples",
            "Unequal variance t-test for differences in the means of two samples",
            "F-test for differences in the variances of two samples",
            "Mixture model test for unimodality"};
        // (The NaN is spelled inline rather than hoisted to a local constant: a function-local
        // `const`/`constexpr` used inside a capture-less lambda is what raises MSVC C3493, and
        // capturing it explicitly then trips clang's -Wunused-lambda-capture. `kKeys` has static
        // storage duration, so it needs no capture.)
        auto all_nan = []() {
            std::vector<std::pair<std::string, double>> r;
            r.reserve(10);
            for (int i = 0; i < 10; i++)
                r.emplace_back(kKeys[i], std::numeric_limits<double>::quiet_NaN());
            return r;
        };

        if (exact_series_.count() < 10) return all_nan();

        // Note 3: the log10 branch filters on Value > 0; the real-space branch does not.
        std::vector<double> indexes;
        std::vector<double> values;
        for (std::size_t i = 0; i < exact_series_.count(); i++) {
            if (use_log10 && !(exact_series_[i].value() > 0.0)) continue;
            indexes.push_back(static_cast<double>(exact_series_[i].index()));
            values.push_back(use_log10 ? exact_series_[i].log10_value()
                                       : exact_series_[i].value());
        }

        // Note 2: clamp the split index to the data range so that both samples have at least
        // one value. An out-of-range index (e.g. 0 from an uninitialized slider) would leave
        // one sample empty, causing the two-sample tests to throw. The fallback indexes
        // ExactSeries by `values.size() / 2`, which under use_log10 counts a DIFFERENT
        // collection -- mirrored as written.
        int min_index = exact_series_[0].index();
        int max_index = exact_series_[exact_series_.count() - 1].index();
        if (index < 0 || index <= min_index || index > max_index)
            index = exact_series_[static_cast<std::size_t>(
                                      static_cast<int>(static_cast<double>(values.size()) / 2.0))]
                        .index();

        std::vector<double> v1;
        std::vector<double> v2;
        for (std::size_t i = 0; i < exact_series_.count(); i++) {
            if (use_log10 && !(exact_series_[i].value() > 0.0)) continue;
            double v = use_log10 ? exact_series_[i].log10_value() : exact_series_[i].value();
            if (exact_series_[i].index() < index) {
                v1.push_back(v);
            } else {
                v2.push_back(v);
            }
        }

        namespace ht = numerics::data::hypothesis_tests;
        try {
            std::vector<std::pair<std::string, double>> result;
            result.reserve(10);
            result.emplace_back(kKeys[0], ht::jarque_bera_test(values));
            result.emplace_back(kKeys[1], ht::ljung_box_test(values));  // note 4: default lag
            result.emplace_back(kKeys[2], ht::wald_wolfowitz_test(values));
            result.emplace_back(kKeys[3],
                                ht::mann_whitney_test(v1.size() <= v2.size() ? v1 : v2,
                                                       v1.size() > v2.size() ? v1 : v2));
            result.emplace_back(kKeys[4], ht::mann_kendall_test(values));
            result.emplace_back(kKeys[5], ht::linear_trend_test(indexes, values));
            result.emplace_back(kKeys[6], ht::equal_variance_t_test(v1, v2));
            result.emplace_back(kKeys[7], ht::unequal_variance_t_test(v1, v2));
            result.emplace_back(kKeys[8], ht::f_test(v1, v2));
            result.emplace_back(kKeys[9], ht::unimodality_test(values));
            return result;
        } catch (...) {
            // Note 1: any single failure NaNs the whole ten-key result.
            return all_nan();
        }
    }

    // Computes nonparametric central moments [mean, stdDev, skewness, kurtosis] of the
    // data, optionally log10-transformed (C# GetNonparametricMoments, line 1659; ported
    // additively in B9). Combines exact, uncertain, and interval data with their
    // Hirsch-Stedinger plotting position complements into an EmpiricalDistribution, then
    // computes central moments via numerical integration with 1000 points. Returns an
    // empty optional (the C# null) when there are fewer than 4 data points. Reference:
    // Hirsch, R.M. and Stedinger, J.R. (1987). Plotting positions for historical floods
    // and their precision. Water Resources Research, 23(4), 715-727.
    std::optional<std::vector<double>> get_nonparametric_moments(
        bool use_log10_values = false) const {
        if (exact_series_.count() < 4) return std::nullopt;

        int total_count = static_cast<int>(exact_series_.count() + uncertain_series_.count() +
                                           interval_series_.count());
        if (total_count < 4) return std::nullopt;

        // Build sorted values
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(total_count));
        if (use_log10_values) {
            for (std::size_t i = 0; i < exact_series_.count(); i++)
                values.push_back(exact_series_[i].log10_value());
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                values.push_back(uncertain_series_[i].log10_value());
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                values.push_back(interval_series_[i].log10_value());
        } else {
            for (std::size_t i = 0; i < exact_series_.count(); i++)
                values.push_back(exact_series_[i].value());
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                values.push_back(uncertain_series_[i].value());
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                values.push_back(interval_series_[i].value());
        }
        std::sort(values.begin(), values.end());

        // Build sorted plotting position complements
        std::vector<double> probs = plotting_position_complements();

        // Repeated X-values collapse into a single right-continuous CDF step (BestFit
        // v2.0.0); a degenerate sample (fewer than two distinct values, e.g. all-identical)
        // reports unavailable rather than constructing a one-point EmpiricalDistribution.
        std::optional<numerics::distributions::EmpiricalDistribution> dist =
            create_empirical_distribution_with_unique_values(values, probs);
        if (!dist.has_value()) return std::nullopt;
        return dist->central_moments(1000);
    }

    // Computes nonparametric central moments using Regression on Order Statistics (ROS)
    // to impute values for low outliers below the censoring threshold (C#
    // GetNonparametricMomentsROS, line 1734; ported additively in B9). Algorithm: fit a
    // simple linear regression of value vs. standard normal quantile z = Phi^-1(pp)
    // through the uncensored exact points only, then replace each low outlier's value
    // with the regression prediction at its z before computing the empirical moments.
    // Falls back to get_nonparametric_moments() when there are no low outliers or fewer
    // than 2 uncensored exact points; returns an empty optional (the C# null) when there
    // are fewer than 4 data points. References: Helsel, D.R. and Cohn, T.A. (1988).
    // Estimation of descriptive statistics for multiply censored water quality data.
    // Water Resources Research, 24(12), 1997-2004; Helsel, D.R. (2005). Nondetects and
    // Data Analysis. Wiley, New York.
    std::optional<std::vector<double>> get_nonparametric_moments_ros(
        bool use_log10_values = false) const {
        // Fall back to standard method when there are no low outliers to impute
        if (number_of_low_outliers_ == 0) return get_nonparametric_moments(use_log10_values);

        if (exact_series_.count() < 4) return std::nullopt;

        int total_count = static_cast<int>(exact_series_.count() + uncertain_series_.count() +
                                           interval_series_.count());
        if (total_count < 4) return std::nullopt;

        // Separate exact data into uncensored and censored (low outlier) sets
        std::vector<double> uncensored_values;
        std::vector<double> uncensored_quantiles;
        numerics::distributions::Normal std_normal(0, 1);

        // Build paired (value, quantile) lists for exact series
        for (std::size_t i = 0; i < exact_series_.count(); i++) {
            double value = use_log10_values ? exact_series_[i].log10_value()
                                            : exact_series_[i].value();
            double z = std_normal.inverse_cdf(exact_series_[i].plotting_position_complement());

            if (!exact_series_[i].is_low_outlier()) {
                uncensored_values.push_back(value);
                uncensored_quantiles.push_back(z);
            }
        }

        // Need at least 2 uncensored points to fit a regression line
        if (uncensored_values.size() < 2) return get_nonparametric_moments(use_log10_values);

        // Fit linear regression: value = a + b * z using uncensored points only
        // (the C# single-column Matrix(double[]) ctor is not on the ported Matrix; the
        // same layout is built through the (rows, cols) ctor).
        numerics::math::linalg::Matrix x_matrix(
            static_cast<int>(uncensored_quantiles.size()), 1);
        for (std::size_t i = 0; i < uncensored_quantiles.size(); i++)
            x_matrix(static_cast<int>(i), 0) = uncensored_quantiles[i];
        numerics::math::linalg::Vector y_vector(uncensored_values);
        numerics::data::regression::LinearRegression regression(x_matrix, y_vector, true);
        double intercept = regression.parameters()[0];
        double slope = regression.parameters()[1];

        // Build combined values list with ROS-imputed values for low outliers
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(total_count));
        for (std::size_t i = 0; i < exact_series_.count(); i++) {
            if (exact_series_[i].is_low_outlier()) {
                // Impute from regression line at this point's normal quantile
                double z =
                    std_normal.inverse_cdf(exact_series_[i].plotting_position_complement());
                values.push_back(intercept + slope * z);
            } else {
                values.push_back(use_log10_values ? exact_series_[i].log10_value()
                                                  : exact_series_[i].value());
            }
        }

        // Add uncertain and interval series values (not subject to low-outlier imputation)
        if (use_log10_values) {
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                values.push_back(uncertain_series_[i].log10_value());
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                values.push_back(interval_series_[i].log10_value());
        } else {
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                values.push_back(uncertain_series_[i].value());
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                values.push_back(interval_series_[i].value());
        }
        std::sort(values.begin(), values.end());

        // Build sorted plotting position complements
        std::vector<double> probs = plotting_position_complements();

        // Repeated X-values collapse into a single right-continuous CDF step (BestFit
        // v2.0.0); a degenerate sample (fewer than two distinct values) reports
        // unavailable rather than constructing a one-point EmpiricalDistribution.
        std::optional<numerics::distributions::EmpiricalDistribution> dist =
            create_empirical_distribution_with_unique_values(values, probs);
        if (!dist.has_value()) return std::nullopt;
        return dist->central_moments(1000);
    }

    // The twenty summary-dictionary keys, in insertion order (C# `Dictionary<string,
    // double>` -- an ordered key/value list here since insertion order is oracle-visible
    // and the R/Python glue selects by label). Shared by both summary methods below.
    static const std::vector<std::string>& summary_statistics_keys() {
        static const std::vector<std::string> keys = {
            "Record Length",     "Events Per Index (λ)", "Low Outliers",
            "Minimum",           "Maximum",                   "Mean",
            "Std Dev",           "Skewness",                  "Kurtosis",
            "Mean (of log)",     "Std Dev (of log)",          "Skewness (of log)",
            "Kurtosis (of log)", "1%",                        "5%",
            "25%",               "50%",                       "75%",
            "95%",               "99%"};
        return keys;
    }

    // The all-NaN summary result both summary methods below return when there are fewer
    // than 10 exact points (C# lines 1789-1808 / 1849-1868).
    std::vector<std::pair<std::string, double>> unavailable_summary_statistics() const {
        std::vector<std::pair<std::string, double>> result;
        result.reserve(summary_statistics_keys().size());
        for (const std::string& key : summary_statistics_keys())
            result.emplace_back(key, std::numeric_limits<double>::quiet_NaN());
        return result;
    }

    // Returns summary statistics for exact data only (C# SummaryStatisticsExactDataOnly,
    // line 1786; P4 Task 5). Reports NaN for all twenty keys when there are fewer than 10
    // exact points. NOTE (transcription note 2): Kurtosis here is `moments[3] + 3` (raw
    // excess kurtosis shifted back to Pearson's kurtosis); summary_statistics_all_data()
    // below reports the SAME moments[3] slot with NO +3 -- the asymmetry is upstream's,
    // not a port bug. NOTE (transcription note 5): the C# `ExactSeries.Count <= 2 ? ... :
    // ...` moment/percentile guards below are dead code -- the outer `Count < 10` guard
    // above has already returned by the time they run -- but are ported anyway for
    // structural fidelity (a byte-for-byte mirror of the C# branch, not a "cleaned up"
    // simplification), matching this codebase's standing convention of preserving
    // upstream oddities rather than silently removing them.
    std::vector<std::pair<std::string, double>> summary_statistics_exact_data_only() const {
        if (exact_series_.count() < 10) return unavailable_summary_statistics();

        std::vector<double> values = exact_sample(/*use_log10=*/false);
        std::vector<double> log_values = exact_sample(/*use_log10=*/true);

        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> moments = exact_series_.count() <= 2
                                          ? std::vector<double>{kNaN, kNaN, kNaN, kNaN}
                                          : numerics::data::product_moments(values);
        std::vector<double> log_moments = exact_series_.count() <= 2
                                              ? std::vector<double>{kNaN, kNaN, kNaN, kNaN}
                                              : numerics::data::product_moments(log_values);
        std::vector<double> percentiles =
            exact_series_.count() <= 2
                ? std::vector<double>{kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN}
                : numerics::data::percentile(
                      values, std::vector<double>{0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99});

        std::vector<std::pair<std::string, double>> result;
        result.reserve(summary_statistics_keys().size());
        result.emplace_back("Record Length", static_cast<double>(exact_series_.count()));
        result.emplace_back("Events Per Index (λ)", lambda_);
        result.emplace_back("Low Outliers", static_cast<double>(number_of_low_outliers_));
        result.emplace_back("Minimum", numerics::data::minimum(values));
        result.emplace_back("Maximum", numerics::data::maximum(values));
        result.emplace_back("Mean", moments[0]);
        result.emplace_back("Std Dev", moments[1]);
        result.emplace_back("Skewness", moments[2]);
        result.emplace_back("Kurtosis", moments[3] + 3.0);
        result.emplace_back("Mean (of log)", log_moments[0]);
        result.emplace_back("Std Dev (of log)", log_moments[1]);
        result.emplace_back("Skewness (of log)", log_moments[2]);
        result.emplace_back("Kurtosis (of log)", log_moments[3] + 3.0);
        result.emplace_back("1%", percentiles[0]);
        result.emplace_back("5%", percentiles[1]);
        result.emplace_back("25%", percentiles[2]);
        result.emplace_back("50%", percentiles[3]);
        result.emplace_back("75%", percentiles[4]);
        result.emplace_back("95%", percentiles[5]);
        result.emplace_back("99%", percentiles[6]);
        return result;
    }

    // Returns summary statistics for all data, from a nonparametric distribution over the
    // combined exact/uncertain/interval series (C# SummaryStatisticsAllData, line 1845;
    // P4 Task 5). Reports NaN for all twenty keys when there are fewer than 10 exact
    // points. NOTE (transcription note 3): `values`, `log_values`, and `probs` below are
    // three INDEPENDENTLY sorted parallel arrays -- the same quirk
    // create_empirical_distribution_with_unique_values() already documents, carried
    // through unchanged from the two GetNonparametricMoments methods it shares this tail
    // with (`probs` is `plotting_position_complements()`, the same private helper).
    // Central moments use 1000 quadrature steps (transcription note 4: SetStandardizedValues
    // below uses 200 -- different, on purpose). Kurtosis here is the RAW moments[3] slot
    // with no +3 (see transcription note 2 on summary_statistics_exact_data_only() above).
    std::vector<std::pair<std::string, double>> summary_statistics_all_data() const {
        if (exact_series_.count() < 10) return unavailable_summary_statistics();

        std::vector<double> values;
        values.reserve(exact_series_.count() + uncertain_series_.count() +
                       interval_series_.count());
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            values.push_back(exact_series_[i].value());
        for (std::size_t i = 0; i < uncertain_series_.count(); i++)
            values.push_back(uncertain_series_[i].value());
        for (std::size_t i = 0; i < interval_series_.count(); i++)
            values.push_back(interval_series_[i].value());
        std::sort(values.begin(), values.end());

        std::vector<double> probs = plotting_position_complements();

        std::vector<double> log_values;
        log_values.reserve(exact_series_.count() + uncertain_series_.count() +
                           interval_series_.count());
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            log_values.push_back(exact_series_[i].log10_value());
        for (std::size_t i = 0; i < uncertain_series_.count(); i++)
            log_values.push_back(uncertain_series_[i].log10_value());
        for (std::size_t i = 0; i < interval_series_.count(); i++)
            log_values.push_back(interval_series_[i].log10_value());
        std::sort(log_values.begin(), log_values.end());

        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        std::optional<numerics::distributions::EmpiricalDistribution> dist =
            create_empirical_distribution_with_unique_values(values, probs);
        std::vector<double> moments =
            dist.has_value() ? dist->central_moments(1000)
                             : std::vector<double>{kNaN, kNaN, kNaN, kNaN};
        std::optional<numerics::distributions::EmpiricalDistribution> log_dist =
            create_empirical_distribution_with_unique_values(log_values, probs);
        std::vector<double> log_moments =
            log_dist.has_value() ? log_dist->central_moments(1000)
                                 : std::vector<double>{kNaN, kNaN, kNaN, kNaN};

        create_full_time_series();

        std::vector<std::pair<std::string, double>> result;
        result.reserve(summary_statistics_keys().size());
        result.emplace_back("Record Length",
                             static_cast<double>(full_time_series_.size()));
        result.emplace_back("Events Per Index (λ)", lambda_);
        result.emplace_back("Low Outliers", static_cast<double>(number_of_low_outliers_));
        result.emplace_back("Minimum", numerics::data::minimum(values));
        result.emplace_back("Maximum", numerics::data::maximum(values));
        result.emplace_back("Mean", moments[0]);
        result.emplace_back("Std Dev", moments[1]);
        result.emplace_back("Skewness", moments[2]);
        result.emplace_back("Kurtosis", moments[3]);
        result.emplace_back("Mean (of log)", log_moments[0]);
        result.emplace_back("Std Dev (of log)", log_moments[1]);
        result.emplace_back("Skewness (of log)", log_moments[2]);
        result.emplace_back("Kurtosis (of log)", log_moments[3]);
        result.emplace_back("1%", dist.has_value() ? dist->inverse_cdf(0.01) : kNaN);
        result.emplace_back("5%", dist.has_value() ? dist->inverse_cdf(0.05) : kNaN);
        result.emplace_back("25%", dist.has_value() ? dist->inverse_cdf(0.25) : kNaN);
        result.emplace_back("50%", dist.has_value() ? dist->inverse_cdf(0.5) : kNaN);
        result.emplace_back("75%", dist.has_value() ? dist->inverse_cdf(0.75) : kNaN);
        result.emplace_back("95%", dist.has_value() ? dist->inverse_cdf(0.95) : kNaN);
        result.emplace_back("99%", dist.has_value() ? dist->inverse_cdf(0.99) : kNaN);
        return result;
    }

    // Sets standardized values (for a Q-Q plot) on every exact/uncertain/interval item
    // (C# SetStandardizedValues, line 2165; P4 Task 5). A no-op guard at fewer than 4
    // exact points (a plain early return, NOT the 10-item / NaN-fill convention the two
    // summary methods above use -- upstream's own inconsistency, mirrored here). NOTE
    // (transcription note 3, continued): `values`, `log_values`, and `probs` are again
    // three independently sorted parallel arrays. NOTE (transcription note 4): central
    // moments here use 200 quadrature steps, not the 1000 the summary methods use --
    // different, on purpose. Control flow mirrors C# exactly: if EITHER distribution is
    // unavailable, BOTH standardized fields on every item are set to NaN and the method
    // returns; otherwise StandardizedValue is set from a Normal(moments) fit (or set to
    // NaN alone, if the real-space moments are non-finite) and StandardizedLog10Value is
    // set independently from a Normal(log_moments) fit (or set to NaN alone) -- so a
    // failure on one side does not touch the other side's already-written field.
    void set_standardized_values() {
        if (exact_series_.count() < 4) return;

        std::vector<double> values;
        values.reserve(exact_series_.count() + uncertain_series_.count() +
                       interval_series_.count());
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            values.push_back(exact_series_[i].value());
        for (std::size_t i = 0; i < uncertain_series_.count(); i++)
            values.push_back(uncertain_series_[i].value());
        for (std::size_t i = 0; i < interval_series_.count(); i++)
            values.push_back(interval_series_[i].value());
        std::sort(values.begin(), values.end());

        std::vector<double> probs = plotting_position_complements();

        std::vector<double> log_values;
        log_values.reserve(exact_series_.count() + uncertain_series_.count() +
                           interval_series_.count());
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            log_values.push_back(exact_series_[i].log10_value());
        for (std::size_t i = 0; i < uncertain_series_.count(); i++)
            log_values.push_back(uncertain_series_[i].log10_value());
        for (std::size_t i = 0; i < interval_series_.count(); i++)
            log_values.push_back(interval_series_[i].log10_value());
        std::sort(log_values.begin(), log_values.end());

        std::optional<numerics::distributions::EmpiricalDistribution> dist =
            create_empirical_distribution_with_unique_values(values, probs);
        std::optional<numerics::distributions::EmpiricalDistribution> log_dist =
            create_empirical_distribution_with_unique_values(log_values, probs);

        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        if (!dist.has_value() || !log_dist.has_value()) {
            for (std::size_t i = 0; i < exact_series_.count(); i++) {
                exact_series_[i].set_standardized_value(kNaN);
                exact_series_[i].set_standardized_log10_value(kNaN);
            }
            for (std::size_t i = 0; i < uncertain_series_.count(); i++) {
                uncertain_series_[i].set_standardized_value(kNaN);
                uncertain_series_[i].set_standardized_log10_value(kNaN);
            }
            for (std::size_t i = 0; i < interval_series_.count(); i++) {
                interval_series_[i].set_standardized_value(kNaN);
                interval_series_[i].set_standardized_log10_value(kNaN);
            }
            return;
        }

        std::vector<double> moments = dist->central_moments(200);
        std::vector<double> log_moments = log_dist->central_moments(200);

        if (std::isnan(moments[0]) || std::isnan(moments[1])) {
            for (std::size_t i = 0; i < exact_series_.count(); i++)
                exact_series_[i].set_standardized_value(kNaN);
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                uncertain_series_[i].set_standardized_value(kNaN);
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                interval_series_[i].set_standardized_value(kNaN);
            return;
        }

        numerics::distributions::Normal normal(moments[0], moments[1]);
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            exact_series_[i].set_standardized_value(
                normal.inverse_cdf(exact_series_[i].plotting_position_complement()));
        for (std::size_t i = 0; i < uncertain_series_.count(); i++)
            uncertain_series_[i].set_standardized_value(
                normal.inverse_cdf(uncertain_series_[i].plotting_position_complement()));
        for (std::size_t i = 0; i < interval_series_.count(); i++)
            interval_series_[i].set_standardized_value(
                normal.inverse_cdf(interval_series_[i].plotting_position_complement()));

        if (std::isnan(log_moments[0]) || std::isnan(log_moments[1])) {
            for (std::size_t i = 0; i < exact_series_.count(); i++)
                exact_series_[i].set_standardized_log10_value(kNaN);
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                uncertain_series_[i].set_standardized_log10_value(kNaN);
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                interval_series_[i].set_standardized_log10_value(kNaN);
            return;
        }

        numerics::distributions::Normal log_normal(log_moments[0], log_moments[1]);
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            exact_series_[i].set_standardized_log10_value(
                log_normal.inverse_cdf(exact_series_[i].plotting_position_complement()));
        for (std::size_t i = 0; i < uncertain_series_.count(); i++)
            uncertain_series_[i].set_standardized_log10_value(
                log_normal.inverse_cdf(uncertain_series_[i].plotting_position_complement()));
        for (std::size_t i = 0; i < interval_series_.count(); i++)
            interval_series_[i].set_standardized_log10_value(
                log_normal.inverse_cdf(interval_series_[i].plotting_position_complement()));
    }

    // Create a deep copy of the data frame (C# Clone, line 1907, which round-trips
    // through XElement; the direct deep clone here preserves the same state: the four
    // series, NumberOfLowOutliers, LowOutlierThreshold, PlottingParameter, and Lambda).
    // Like the C#, the clone starts with an empty full time series; the first
    // full_time_series() access triggers a lazy rebuild on the clone.
    DataFrame clone() const {
        DataFrame copy;
        copy.exact_series_ = exact_series_.clone();
        copy.uncertain_series_ = uncertain_series_.clone();
        copy.interval_series_ = interval_series_.clone();
        copy.threshold_series_ = threshold_series_.clone();
        copy.lambda_ = lambda_;
        copy.number_of_low_outliers_ = number_of_low_outliers_;
        copy.low_outlier_threshold_ = low_outlier_threshold_;
        copy.plotting_parameter_ = plotting_parameter_;
        return copy;
    }

    // ===================== Time-series import (P6) =====================
    // Ported from the C# `#region Create Series Methods` (DataFrame.cs 2339-2400), un-gated by
    // the P6 TimeSeries container. Both replace the exact series wholesale from a time series,
    // and both build their ExactData ordinates from a DATE -- so each ordinate's index is the
    // observation's YEAR and the date rides along for the seasonal PointProcess path.

    // Replaces the exact series with one value per time block (C# CreateBlockSeries, line 2339).
    // Sets lambda to 1 UNCONDITIONALLY: a block series has exactly one event per block by
    // construction, whatever the underlying record's rate was.
    void create_block_series(numerics::data::TimeSeries& time_series,
                             numerics::data::TimeBlockWindow time_block =
                                 numerics::data::TimeBlockWindow::WaterYear,
                             numerics::data::BlockFunctionType block_function =
                                 numerics::data::BlockFunctionType::Maximum,
                             numerics::data::SmoothingFunctionType smoothing_function =
                                 numerics::data::SmoothingFunctionType::None,
                             int start_month = 10, int end_month = 9, int period = 1) {
        numerics::data::TimeSeries blocked(numerics::data::TimeInterval::Irregular);
        switch (time_block) {
            case numerics::data::TimeBlockWindow::CalendarYear:
                blocked = time_series.calendar_year_series(block_function, smoothing_function,
                                                           period);
                break;
            case numerics::data::TimeBlockWindow::WaterYear:
                blocked = time_series.custom_year_series(start_month, block_function,
                                                          smoothing_function, period);
                break;
            case numerics::data::TimeBlockWindow::CustomYear:
                blocked = time_series.custom_year_series(start_month, end_month, block_function,
                                                          smoothing_function, period);
                break;
            case numerics::data::TimeBlockWindow::Quarter:
                blocked = time_series.quarterly_series(block_function, smoothing_function, period);
                break;
            case numerics::data::TimeBlockWindow::Month:
                blocked = time_series.monthly_series(block_function, smoothing_function, period);
                break;
        }

        exact_series_.clear();
        for (int i = 0; i < blocked.count(); ++i)
            exact_series_.add(ExactData(blocked[i].index(), blocked[i].value()));
        lambda_ = 1;
    }

    // Replaces the exact series with the peaks over a threshold (C#
    // CreatePeaksOverThresholdSeries, line 2383). Unlike the block series, lambda is the observed
    // RATE: the event count over the record's span in years (inclusive of both end years).
    void create_peaks_over_threshold_series(numerics::data::TimeSeries& time_series,
                                            double threshold, int min_steps_between_peaks = 1,
                                            numerics::data::SmoothingFunctionType
                                                smoothing_function =
                                                    numerics::data::SmoothingFunctionType::None,
                                            int period = 1) {
        numerics::data::TimeSeries pot = time_series.peaks_over_threshold_series(
            threshold, min_steps_between_peaks, smoothing_function, period);

        exact_series_.clear();
        for (int i = 0; i < pot.count(); ++i)
            exact_series_.add(ExactData(pot[i].index(), pot[i].value()));
        double events = static_cast<double>(exact_series_.count());
        double span = static_cast<double>(time_series.end_date().year() -
                                          time_series.start_date().year() + 1);
        lambda_ = events / span;
    }

    // ===================== Bootstrap Methods (A3) =====================
    // Ported from the C# `#region Bootstrap Methods` (DataFrame.cs 2059-2543). The C#
    // suppresses collection-changed events for speed inside these methods
    // (SuppressCollectionChanged = true / false); the C++ port has no events, so those
    // lines drop (the invalidation strategy is documented at the top of this file --
    // recompute explicitly via process_threshold_series() / lazily via full_time_series()).

    // Generate a jackknife data frame by leaving out one observation (C# JackKnife,
    // line 2066). Returns a new reduced frame: clone() the frame, then remove the single
    // observation at the global `index`. The C# searches the series in order Exact ->
    // Uncertain -> Interval to remove the ordinate; for a Threshold hit it decrements the
    // covering threshold's NumberAbove/NumberBelow by the sub-range (above region
    // [StartIndex, StartIndex + NumberAbove); below region [StartIndex + NumberAbove,
    // EndIndex]) rather than erasing a stored point. Consumed by A8 AccelerationConstants.
    DataFrame JackKnife(int index) {
        DataFrame dataframe = clone();

        // Exact data
        for (std::size_t i = 0; i < dataframe.exact_series_.count(); i++) {
            if (dataframe.exact_series_[i].index() == index) {
                dataframe.exact_series_.remove_at(i);
                return dataframe;
            }
        }
        // Uncertain data
        for (std::size_t i = 0; i < dataframe.uncertain_series_.count(); i++) {
            if (dataframe.uncertain_series_[i].index() == index) {
                dataframe.uncertain_series_.remove_at(i);
                return dataframe;
            }
        }
        // Interval data
        for (std::size_t i = 0; i < dataframe.interval_series_.count(); i++) {
            if (dataframe.interval_series_[i].index() == index) {
                dataframe.interval_series_.remove_at(i);
                return dataframe;
            }
        }
        // Threshold data
        for (std::size_t t = 0; t < dataframe.threshold_series_.count(); t++) {
            ThresholdData& data = dataframe.threshold_series_[t];
            // Number above: [StartIndex, StartIndex + NumberAbove)
            for (int i = data.start_index(); i < data.start_index() + data.number_above();
                 i++) {
                if (index == i) {
                    data.set_number_above(data.number_above() - 1);
                    return dataframe;
                }
            }
            // Number below: [StartIndex + NumberAbove, EndIndex]
            for (int i = data.start_index() + data.number_above(); i <= data.end_index();
                 i++) {
                if (index == i) {
                    data.set_number_below(data.number_below() - 1);
                    return dataframe;
                }
            }
        }

        return dataframe;
    }

    // Generate a resampled data frame using nonparametric bootstrap resampling with
    // replacement (C# Resample, line 2164). Draws indices via the ranged
    // next_integers(prng, startIndex, endIndex + 1, FullTimeSeries.Count, replace = true),
    // sorts them, then rebuilds the frame by index lookup across all four series (including
    // the threshold bins above/below). The above/below split uses the C# Resample
    // convention: above region [StartIndex, StartIndex + NumberAbove); below region
    // [StartIndex + NumberAbove, EndIndex].
    DataFrame Resample(numerics::sampling::MersenneTwister& prng,
                       bool create_full_time_series = false) {
        // FullTimeSeries getter rebuilds lazily when empty (C# CreateFullTimeSeries guard).
        const std::vector<std::unique_ptr<Data>>& full = full_time_series();
        DataFrame dataframe;
        if (full.empty()) return dataframe;

        int start_index = full.front()->index();
        int end_index = full.back()->index();
        std::vector<int> indexes = numerics::utilities::next_integers(
            prng, start_index, end_index + 1, static_cast<int>(full.size()), true);
        std::sort(indexes.begin(), indexes.end());

        for (std::size_t k = 0; k < indexes.size(); k++) {
            int idx = indexes[k];
            bool found = false;

            // Exact data
            for (std::size_t i = 0; i < exact_series_.count(); i++) {
                if (exact_series_[i].index() == idx) {
                    dataframe.exact_series_.add(exact_series_[i].clone());
                    found = true;
                    break;
                }
            }
            if (found) continue;

            // Uncertain data
            for (std::size_t i = 0; i < uncertain_series_.count(); i++) {
                if (uncertain_series_[i].index() == idx) {
                    dataframe.uncertain_series_.add(uncertain_series_[i].clone());
                    found = true;
                    break;
                }
            }
            if (found) continue;

            // Interval data
            for (std::size_t i = 0; i < interval_series_.count(); i++) {
                if (interval_series_[i].index() == idx) {
                    dataframe.interval_series_.add(interval_series_[i].clone());
                    found = true;
                    break;
                }
            }
            if (found) continue;

            // Threshold data - check each threshold period
            for (std::size_t i = 0; i < threshold_series_.count(); i++) {
                const ThresholdData& threshold = threshold_series_[i];
                // Above-threshold region
                if (idx >= threshold.start_index() &&
                    idx < threshold.start_index() + threshold.number_above()) {
                    ThresholdData* existing = find_threshold_bin(
                        dataframe.threshold_series_, threshold.start_index(),
                        threshold.end_index());
                    if (existing != nullptr) {
                        existing->set_number_above(existing->number_above() + 1);
                    } else {
                        ThresholdData new_threshold = threshold.clone();
                        new_threshold.set_number_above(1);
                        new_threshold.set_number_below(0);
                        dataframe.threshold_series_.add(std::move(new_threshold));
                    }
                    found = true;
                    break;
                }
                // Below-threshold region
                if (idx >= threshold.start_index() + threshold.number_above() &&
                    idx <= threshold.end_index()) {
                    ThresholdData* existing = find_threshold_bin(
                        dataframe.threshold_series_, threshold.start_index(),
                        threshold.end_index());
                    if (existing != nullptr) {
                        existing->set_number_below(existing->number_below() + 1);
                    } else {
                        ThresholdData new_threshold = threshold.clone();
                        new_threshold.set_number_above(0);
                        new_threshold.set_number_below(1);
                        dataframe.threshold_series_.add(std::move(new_threshold));
                    }
                    found = true;
                    break;
                }
            }
        }

        if (create_full_time_series) dataframe.create_full_time_series();

        return dataframe;
    }

    // Generates a parametric bootstrap data frame by simulating the physical observation
    // process (C# BootstrapDataFrame, line 2336). Per-series behaviour mirrors the C#:
    // exact -> unconditional inverse_cdf(U); uncertain -> shift the measurement-error
    // distribution to center on a simulated value; interval -> reclassify against the
    // original bounds; threshold -> clone systematic (NumberBelow == 0) bins, else resample
    // the exceedance count from Binomial(n, 1 - F(threshold)). Finally re-flags low
    // outliers and calls process_threshold_series() (the C# ordering). Consumed by A8's
    // parametric bootstrap. Reference: Davison, A.C. and Hinkley, D.V. (1997). Bootstrap
    // Methods and Their Application, Sections 3.5 and 7.3.
    DataFrame BootstrapDataFrame(
        const numerics::distributions::UnivariateDistributionBase& distribution,
        numerics::sampling::MersenneTwister& prng, bool create_full_time_series = false) {
        DataFrame dataframe;
        bool filter_low_outliers = number_of_low_outliers_ > 0;

        // Exact data: unconditional draw from the fitted distribution.
        for (std::size_t i = 0; i < exact_series_.count(); i++) {
            double simulated_value = distribution.inverse_cdf(prng.next_double());
            dataframe.exact_series_.add(ExactData(exact_series_[i].index(), simulated_value));
        }

        // Uncertain data: draw a "true" magnitude, then shift the error distribution.
        for (std::size_t i = 0; i < uncertain_series_.count(); i++) {
            double simulated_value = distribution.inverse_cdf(prng.next_double());
            auto shifted_dist =
                shift_distribution(uncertain_series_[i].distribution(), simulated_value);
            dataframe.uncertain_series_.add(
                UncertainData(uncertain_series_[i].index(), std::move(shifted_dist)));
        }

        // Interval data: unconditional draw, then re-classify against original bounds.
        for (std::size_t i = 0; i < interval_series_.count(); i++) {
            const IntervalData& data = interval_series_[i];
            double simulated_value = distribution.inverse_cdf(prng.next_double());
            if (simulated_value < data.lower_value()) {
                double upper = data.lower_value();
                double lower =
                    std::min(upper - 1E-8,
                             distribution.inverse_cdf(numerics::kDoubleMachineEpsilon));
                double mid = 0.5 * (lower + upper);
                dataframe.interval_series_.add(
                    IntervalData(data.index(), lower, mid, upper));
            } else if (simulated_value > data.upper_value()) {
                double lower = data.upper_value();
                double upper = std::max(
                    lower + 1E-8,
                    distribution.inverse_cdf(1.0 - numerics::kDoubleMachineEpsilon));
                double mid = 0.5 * (lower + upper);
                dataframe.interval_series_.add(
                    IntervalData(data.index(), lower, mid, upper));
            } else {
                dataframe.interval_series_.add(IntervalData(
                    data.index(), data.lower_value(),
                    0.5 * (data.lower_value() + data.upper_value()), data.upper_value()));
            }
        }

        // Threshold data: systematic clone vs. historical Binomial resample.
        for (std::size_t i = 0; i < threshold_series_.count(); i++) {
            const ThresholdData& data = threshold_series_[i];
            if (data.number_below() == 0) {
                // Systematic threshold: clone with original counts to avoid spurious
                // NumberAbove double-counting the exact observations.
                dataframe.threshold_series_.add(data.clone());
            } else {
                // Historical threshold: resample the exceedance count from Binomial(n, p),
                // p = P(X > threshold) under the fitted distribution.
                double p = 1.0 - distribution.cdf(data.value());
                int n = data.duration() - data.number_above();
                int n_above;
                if (n <= 0 || p <= 0.0) {
                    n_above = 0;
                } else if (p >= 1.0) {
                    n_above = n;
                } else {
                    numerics::distributions::Binomial binomial_dist(p, n);
                    n_above = std::max(
                        0, static_cast<int>(std::floor(
                               binomial_dist.inverse_cdf(prng.next_double()))));
                    n_above = std::min(n_above, n);
                }
                int n_below = data.duration() - n_above;
                ThresholdData new_threshold(data.start_index(), data.end_index(),
                                            data.value());
                new_threshold.set_number_above(n_above);
                new_threshold.set_number_below(n_below);
                dataframe.threshold_series_.add(std::move(new_threshold));
            }
        }

        // Low outliers: re-flag resampled exact values below the threshold. Inline marking
        // (not set_low_outliers_from_threshold) to avoid the >50%-censored validation guard.
        if (filter_low_outliers) {
            dataframe.low_outlier_threshold_ = low_outlier_threshold_;
            dataframe.number_of_low_outliers_ = 0;
            for (std::size_t i = 0; i < dataframe.exact_series_.count(); i++) {
                if (dataframe.exact_series_[i].value() < low_outlier_threshold_) {
                    dataframe.exact_series_[i].set_is_low_outlier(true);
                    dataframe.number_of_low_outliers_++;
                }
            }
        }

        // Post-processing (mirror the C# ordering of the final steps).
        dataframe.process_threshold_series();
        if (create_full_time_series) dataframe.create_full_time_series();

        return dataframe;
    }

    // Creates a new distribution of the same type, shifted so that its center is at
    // new_center while preserving the original measurement error spread (C# ShiftDistribution,
    // line 2485). For additive-error families the distribution is shifted by
    // `new_center - original.Mean`; for multiplicative-error families (LogNormal, Gamma) a
    // ratio-based shift preserves the CV. Unrecognized types fall back to a clone.
    //
    // The C# declares this `private static`; the C++ port exposes it `public static`
    // following the ThresholdData::set_number_below precedent (a C# `internal`/`private`
    // widened for DataFrame and for the C++-only test harness, not for end users) so the
    // arm-by-arm shift behaviour is directly testable. Access modifier only -- no numerical
    // deviation.
    static std::unique_ptr<numerics::distributions::UnivariateDistributionBase>
    shift_distribution(
        const numerics::distributions::UnivariateDistributionBase& original,
        double new_center) {
        namespace nd = numerics::distributions;
        double original_mean = original.mean();
        double shift = new_center - original_mean;

        // Guard against degenerate cases (C#: double.IsNaN || double.IsInfinity).
        if (std::isnan(shift) || std::isinf(shift)) return original.clone();

        switch (original.type()) {
            case nd::UnivariateDistributionType::Normal: {
                const auto& n = static_cast<const nd::Normal&>(original);
                return std::make_unique<nd::Normal>(n.mu() + shift, n.sigma());
            }
            case nd::UnivariateDistributionType::TruncatedNormal: {
                const auto& tn = static_cast<const nd::TruncatedNormal&>(original);
                return std::make_unique<nd::TruncatedNormal>(
                    tn.mu() + shift, tn.sigma(), tn.min_param() + shift,
                    tn.max_param() + shift);
            }
            case nd::UnivariateDistributionType::StudentT: {
                const auto& st = static_cast<const nd::StudentT&>(original);
                return std::make_unique<nd::StudentT>(st.mu() + shift, st.sigma(),
                                                      st.degrees_of_freedom());
            }
            case nd::UnivariateDistributionType::LogNormal: {
                const auto& ln = static_cast<const nd::LogNormal&>(original);
                // Multiplicative shift in log-space preserves the CV.
                double ratio = (original_mean > 0 && new_center > 0)
                                   ? new_center / original_mean
                                   : 1.0;
                double new_mu = ln.mu() + std::log(std::max(ratio, 1e-12));
                return std::make_unique<nd::LogNormal>(new_mu, ln.sigma());
            }
            case nd::UnivariateDistributionType::LnNormal: {
                const auto& lnn = static_cast<const nd::LnNormal&>(original);
                // C# uses the real-space (Mean, StandardDeviation) constructor.
                return std::make_unique<nd::LnNormal>(lnn.mean() + shift,
                                                      lnn.standard_deviation());
            }
            case nd::UnivariateDistributionType::GammaDistribution: {
                const auto& g = static_cast<const nd::GammaDistribution&>(original);
                // Scale shift preserves shape (and CV).
                double ratio = (original_mean > 0 && new_center > 0)
                                   ? new_center / original_mean
                                   : 1.0;
                double new_scale = g.theta() * ratio;
                return std::make_unique<nd::GammaDistribution>(std::max(new_scale, 1e-12),
                                                               g.kappa());
            }
            case nd::UnivariateDistributionType::Uniform: {
                const auto& u = static_cast<const nd::Uniform&>(original);
                return std::make_unique<nd::Uniform>(u.min() + shift, u.max() + shift);
            }
            case nd::UnivariateDistributionType::Triangular: {
                const auto& t = static_cast<const nd::Triangular&>(original);
                return std::make_unique<nd::Triangular>(
                    t.min_val() + shift, t.most_likely() + shift, t.max_val() + shift);
            }
            case nd::UnivariateDistributionType::Pert: {
                const auto& p = static_cast<const nd::Pert&>(original);
                return std::make_unique<nd::Pert>(p.min_val() + shift,
                                                  p.most_likely() + shift,
                                                  p.max_val() + shift);
            }
            case nd::UnivariateDistributionType::GeneralizedBeta: {
                const auto& gb = static_cast<const nd::GeneralizedBeta&>(original);
                return std::make_unique<nd::GeneralizedBeta>(
                    gb.alpha(), gb.beta(), gb.min_val() + shift, gb.max_val() + shift);
            }
            default:
                // Unrecognized distribution type -- clone as-is. The C# logs a
                // Debug.WriteLine here; ported as a silent no-throw path per the global
                // constraint.
                return original.clone();
        }
    }

   private:
    // Finds the threshold bin in `series` matching (start_index, end_index) -- the C#
    // FirstOrDefault(t => t.StartIndex == ... && t.EndIndex == ...) used by Resample.
    // Returns nullptr when no bin matches.
    static ThresholdData* find_threshold_bin(ThresholdSeries& series, int start_index,
                                             int end_index) {
        for (std::size_t i = 0; i < series.count(); i++) {
            if (series[i].start_index() == start_index &&
                series[i].end_index() == end_index) {
                return &series[i];
            }
        }
        return nullptr;
    }

    // The sorted plotting-position complements of the exact + uncertain + interval series
    // (the shared tail of both C# GetNonparametricMoments methods; B9).
    std::vector<double> plotting_position_complements() const {
        std::vector<double> probs;
        probs.reserve(exact_series_.count() + uncertain_series_.count() +
                      interval_series_.count());
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            probs.push_back(exact_series_[i].plotting_position_complement());
        for (std::size_t i = 0; i < uncertain_series_.count(); i++)
            probs.push_back(uncertain_series_[i].plotting_position_complement());
        for (std::size_t i = 0; i < interval_series_.count(); i++)
            probs.push_back(interval_series_[i].plotting_position_complement());
        std::sort(probs.begin(), probs.end());
        return probs;
    }

    // Creates an empirical distribution after collapsing repeated sorted values (C#
    // CreateEmpiricalDistributionWithUniqueValues, BestFit v2.0.0; the shared tail of both
    // GetNonparametricMoments methods). A CDF is right-continuous, so a repeated X-value
    // retains the LARGEST cumulative probability assigned to it (not the first or an
    // average) rather than producing a duplicate-X EmpiricalDistribution, which is either
    // invalid or degrades interpolation accuracy. Returns an empty optional (the C# null)
    // when fewer than two distinct values remain -- e.g. an all-identical sample -- since a
    // one-point empirical distribution has no defined shape/moments. `sorted_values` and
    // `sorted_probabilities` are independently-sorted parallel arrays (see the two call
    // sites: pre-existing, unrelated-to-this-fix quirk carried through unchanged).
    static std::optional<numerics::distributions::EmpiricalDistribution>
    create_empirical_distribution_with_unique_values(
        const std::vector<double>& sorted_values,
        const std::vector<double>& sorted_probabilities) {
        std::vector<double> unique_values;
        std::vector<double> unique_probabilities;
        unique_values.reserve(sorted_values.size());
        unique_probabilities.reserve(sorted_probabilities.size());
        for (std::size_t i = 0; i < sorted_values.size(); i++) {
            double value = sorted_values[i];
            double probability = sorted_probabilities[i];
            if (!unique_values.empty() && value == unique_values.back()) {
                if (probability > unique_probabilities.back())
                    unique_probabilities.back() = probability;
            } else {
                unique_values.push_back(value);
                unique_probabilities.push_back(probability);
            }
        }
        if (unique_values.size() < 2) return std::nullopt;
        return numerics::distributions::EmpiricalDistribution(std::move(unique_values),
                                                               std::move(unique_probabilities));
    }

    static void append(ValidationResult& result, const ValidationResult& partial) {
        if (partial.is_valid) return;
        result.validation_messages.insert(result.validation_messages.end(),
                                          partial.validation_messages.begin(),
                                          partial.validation_messages.end());
        result.is_valid = false;
    }

    ExactSeries exact_series_;
    UncertainSeries uncertain_series_;
    IntervalSeries interval_series_;
    ThresholdSeries threshold_series_;
    // Mutable: a logically-const cache, lazily rebuilt by the const full_time_series()
    // getter exactly as the C# property getter does (see the M9 note there).
    mutable std::vector<std::unique_ptr<Data>> full_time_series_;

    double lambda_ = 1.0;
    int number_of_low_outliers_ = 0;
    double low_outlier_threshold_ = 0;
    double plotting_parameter_ = 0.0;
    // C# `_plottingPositionVersion` (BestFit v2.0.0); starts fresh at 0 on a clone, exactly
    // like the C# (Clone() round-trips through the XElement constructor, which never
    // restores this field either).
    std::int64_t plotting_position_version_ = 0;
};

// ---------------------------------------------------------------------------
// Out-of-line series Validate definitions (declared in interval_series.hpp /
// uncertain_series.hpp; they cross-reference the DataFrame defined above). Both keep the
// C# quirk of running the duplicate-index and overlap checks INSIDE the per-item loop
// (so an empty series skips them and a multi-item series repeats them).
// ---------------------------------------------------------------------------

// C# IntervalSeries.Validate(DataFrame), line 153.
inline ValidationResult IntervalSeries::validate(const DataFrame* data_frame) const {
    ValidationResult result;

    for (std::size_t i = 0; i < count(); i++) {
        ValidationResult data_valid = (*this)[i].validate();
        if (!data_valid.is_valid) {
            result.validation_messages.insert(result.validation_messages.end(),
                                              data_valid.validation_messages.begin(),
                                              data_valid.validation_messages.end());
            result.is_valid = false;
        }

        // Check for duplicate indexes (in-loop, per the C#).
        std::vector<int> duplicates = duplicate_indices();
        if (!duplicates.empty()) {
            std::string joined;
            for (std::size_t k = 0; k < duplicates.size(); k++) {
                if (k > 0) joined += ", ";
                joined += std::to_string(duplicates[k]);
            }
            result.validation_messages.push_back(
                "Error: Duplicate indexes found in interval series: " + joined);
            result.is_valid = false;
        }

        // Check for overlaps with the exact and uncertain series.
        if (data_frame != nullptr) {
            std::unordered_set<int> exact_indexes;
            for (std::size_t k = 0; k < data_frame->exact_series().count(); k++)
                exact_indexes.insert(data_frame->exact_series()[k].index());
            std::unordered_set<int> uncertain_indexes;
            for (std::size_t k = 0; k < data_frame->uncertain_series().count(); k++)
                uncertain_indexes.insert(data_frame->uncertain_series()[k].index());

            for (std::size_t k = 0; k < count(); k++) {
                int idx = (*this)[k].index();
                if (exact_indexes.count(idx) != 0) {
                    result.validation_messages.push_back(
                        "Error: Interval data at index " + std::to_string(idx) +
                        " overlaps with exact data.");
                    result.is_valid = false;
                }
                if (uncertain_indexes.count(idx) != 0) {
                    result.validation_messages.push_back(
                        "Error: Interval data at index " + std::to_string(idx) +
                        " overlaps with uncertain data.");
                    result.is_valid = false;
                }
            }
        }
    }

    return result;
}

// C# UncertainSeries.Validate(DataFrame), line 153.
inline ValidationResult UncertainSeries::validate(const DataFrame* data_frame) const {
    ValidationResult result;

    for (std::size_t i = 0; i < count(); i++) {
        ValidationResult data_valid = (*this)[i].validate();
        if (!data_valid.is_valid) {
            result.validation_messages.insert(result.validation_messages.end(),
                                              data_valid.validation_messages.begin(),
                                              data_valid.validation_messages.end());
            result.is_valid = false;
        }

        // Check for duplicate indexes (in-loop, per the C#).
        std::vector<int> duplicates = duplicate_indices();
        if (!duplicates.empty()) {
            std::string joined;
            for (std::size_t k = 0; k < duplicates.size(); k++) {
                if (k > 0) joined += ", ";
                joined += std::to_string(duplicates[k]);
            }
            result.validation_messages.push_back(
                "Error: Duplicate indexes found in uncertain series: " + joined);
            result.is_valid = false;
        }

        // Check for overlaps with the exact series.
        if (data_frame != nullptr) {
            std::unordered_set<int> exact_indexes;
            for (std::size_t k = 0; k < data_frame->exact_series().count(); k++)
                exact_indexes.insert(data_frame->exact_series()[k].index());

            for (std::size_t k = 0; k < count(); k++) {
                int idx = (*this)[k].index();
                if (exact_indexes.count(idx) != 0) {
                    result.validation_messages.push_back(
                        "Error: Uncertain data at index " + std::to_string(idx) +
                        " overlaps with exact data.");
                    result.is_valid = false;
                }
            }
        }
    }

    return result;
}

}  // namespace corehydro::models

// Out-of-line definitions of DataFrame::calculate_plotting_positions() and
// DataFrame::apply_langbein_conversion() (split out purely for file size; the header
// must not be included directly).
#include "corehydro/models/data_frame/data_frame_plotting.hpp"  // NOLINT
