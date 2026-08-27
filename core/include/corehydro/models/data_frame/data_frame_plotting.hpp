// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataFrame.cs @ c2e6192
// (the "Plotting Positions" region)
//
// Out-of-line definitions of DataFrame::calculate_plotting_positions() and
// DataFrame::apply_langbein_conversion() -- the Hirsch-Stedinger censored plotting
// positions (Hirsch & Stedinger 1987; Bulletin 17C Appendix 5) and the Langbein
// partial-duration conversion. Split from data_frame.hpp purely for file size; this
// header is included by data_frame.hpp after the class definition and must not be
// included directly.
//
// BestFit v2.0.0 REWRITE (peakFQ-faithful ARRANGE2 / PPLOT2 / PLPOS): CalculatePlottingPositions
// was rewritten from the old sequential Kj/Kl threshold-band scheme to a documented port of
// peakFQ's ARRANGE2 (arrange explicit observations against their OWN covering perception
// threshold), PPLOT2 (compute per-band nonexceedance-interval boundaries from aggregate
// above/below counts), and PLPOS (rank-based plotting position within a band) sequence. The
// governing idea: each explicit observation is classified above/below its threshold by ITS OWN
// window (via FindThresholdForPlotting, a binary search over threshold windows sorted by
// StartIndex), not by a globally-sorted band index as before -- this is what "peakFQ-faithful"
// means and is the reason multi-threshold, tied-value frames can shift plotting-position oracles
// under this rewrite even though the formula shape is unchanged for a single-threshold or
// no-threshold frame (verified: the six named-formula ctests and both full Bulletin 17C worked
// examples below reproduce byte-identical values under the new algorithm).
//
// STRICT VALIDATION (new): CalculatePlottingPositions now THROWS (std::runtime_error, mirroring
// C# InvalidOperationException) on: a non-finite plotting parameter (also guarded eagerly by
// DataFrame::set_plotting_parameter()); a non-finite threshold value; a threshold window with
// StartIndex > EndIndex; processed threshold counts that are negative or exceed the window's
// Duration; overlapping threshold windows (after sorting by StartIndex/EndIndex); a processed
// NumberBelow/NumberAbove that cannot be placed within its own window's unoccupied indexes
// (infeasible against the explicit-data occupancy); a non-finite explicit observation value; or a
// computed plotting position that is non-finite or not strictly inside (0, 1) -- see
// detail::set_strict_plotting_position(). The C# skips this strictness during INTERACTIVE GUI
// edits via a private RecalculatePlottingPositionsAfterEdit wrapper that catches
// InvalidOperationException while `ThresholdSeries.Validate()` reports invalid (a transient state
// while a user/API request is still populating the series) and defers the recalculation with a
// Debug.WriteLine; that wrapper is WPF-editing-workflow-only and is NOT ported (this port's
// invalidation strategy already requires an EXPLICIT calculate_plotting_positions() call after
// every mutation -- see the file header in data_frame.hpp -- so there is no "transient partial
// edit" state to tolerate). DataFrame::set_plotting_parameter() therefore calls
// calculate_plotting_positions() directly here too and lets any throw propagate.
//
// EnsureDistinctPlottingPositions (new): peakFQ's PPLOT2 equations rank censored observations
// independently within each threshold band, so distinct bands can algebraically produce the SAME
// plotting position even though no upstream C# oracle in this corpus exercises duplicate
// exact-data plotting positions from a single band (Weibull-style formulas are already strictly
// monotonic within one band). A tied run (grouped by AlmostEquals, 1e-15 absolute) is reordered by
// decreasing observed value (with Index/Ordinal tiebreakers -- a genuine total order, so this
// re-sort does NOT need the introsort tie-permutation port below) and spread symmetrically within
// the local interval bounded by the midpoints to its neighboring untied positions, preserving the
// tie's H-S center and the global exceedance ordering. Throws if the run cannot be separated into
// finite, strictly-ordered, open-interval positions.
//
// TIE ORDERING (unchanged, still load-bearing): within a single threshold band, PLPOS ranks
// finite detected observations by decreasing value, split into the SAME legacy above/below lists
// the pre-rewrite algorithm used (partitioned at the GLOBAL MINIMUM FINITE THRESHOLD across all
// arranged windows, not per-band) -- "Preserve only the legacy tie permutation" in the C# source's
// own comment is the load-bearing instruction: detection status itself comes from the
// observation's OWN threshold (ARRANGE2), but when two observations tie exactly on value, WHICH
// one is treated as "detected first" still has to reproduce C#'s `List<T>.Sort` order, because
// List<T>.Sort is a deterministic-but-not-stable introspective sort (not the same as a stable
// sort, and not the same as any other correct sort of the same keys). std::sort/std::stable_sort
// therefore cannot reproduce the C# results for a tied legacy-order sort; detail::dotnet_list_sort
// below (a faithful port of .NET's ArraySortHelper<T>.IntrospectiveSort) REMAINS the sort for
// legacy_above_order/legacy_below_order, verified against dotnet 10 output for the tie patterns in
// the B17C examples. Every OTHER sort in the new algorithm (threshold ordering by
// StartIndex/EndIndex; the EnsureDistinctPlottingPositions comparators; censored-entry ordering by
// Index/Ordinal; placeholder-index ordering) carries a full tiebreak chain with no possible ties
// among genuinely distinct observations, so std::sort is safe there.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/utilities/dotnet_sort.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

// (Included by data_frame.hpp; all types it needs are already declared there.)

namespace corehydro::models {

namespace detail {

// The .NET introsort port and the `System.Double.CompareTo` ordering it runs on were MOVED
// VERBATIM to numerics/utilities/dotnet_sort.hpp in P5 Task 1, so the `numerics/` layer
// (KNearestNeighbors) can reach them without depending on `models/`. They are re-exported here
// under their original names, so every call site below is unchanged and the plotting-position
// fixtures are the proof the move altered nothing. See that header for why the tie permutation
// is load-bearing.
using corehydro::numerics::utilities::compare_double;
using corehydro::numerics::utilities::dotnet_list_sort;

// --- H-S rewrite helpers (BestFit v2.0.0) -----------------------------------------------

// Finds the perception threshold covering an observation index (C#
// FindThresholdForPlotting). `thresholds_by_index` is sorted ascending by (StartIndex,
// EndIndex) and nonoverlapping (both preconditions enforced by the caller before this is
// ever invoked). Binary search keeps per-observation threshold association O(log m).
// Returns nullptr when the observation is outside every threshold window (ARRANGE2 then
// uses a synthetic negative-infinity threshold).
inline const ThresholdData* find_threshold_for_plotting(
    const std::vector<const ThresholdData*>& thresholds_by_index, int index) {
    int low = 0;
    int high = static_cast<int>(thresholds_by_index.size()) - 1;
    int candidate = -1;
    while (low <= high) {
        int middle = low + ((high - low) / 2);
        if (thresholds_by_index[static_cast<std::size_t>(middle)]->start_index() <= index) {
            candidate = middle;
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }
    if (candidate >= 0 &&
        index <= thresholds_by_index[static_cast<std::size_t>(candidate)]->end_index()) {
        return thresholds_by_index[static_cast<std::size_t>(candidate)];
    }
    return nullptr;
}

// C# `Array.BinarySearch(levels, value)` exact-match lookup. `levels` is strictly
// ascending and duplicate-free (built from a std::set<double>/C# SortedSet<double>).
// Returns -1 when not found (both call sites below treat that as an arrangement error).
inline int find_level_exact(const std::vector<double>& levels, double value) {
    auto it = std::lower_bound(levels.begin(), levels.end(), value);
    if (it != levels.end() && *it == value) return static_cast<int>(it - levels.begin());
    return -1;
}

// Finds the highest arranged threshold level that does not exceed a detected value (C#
// FindPlottingLevel). Throws std::runtime_error (C# InvalidOperationException) when the
// value falls below every arranged level.
inline int find_plotting_level(const std::vector<double>& levels, double value) {
    auto it = std::lower_bound(levels.begin(), levels.end(), value);
    if (it != levels.end() && *it == value) return static_cast<int>(it - levels.begin());
    int level_index = static_cast<int>(it - levels.begin()) - 1;
    if (level_index < 0) {
        throw std::runtime_error("A detected observation falls below every arranged threshold.");
    }
    return level_index;
}

// Assigns a finite, open-interval exceedance plotting position to an explicit observation
// (C# SetStrictPlottingPosition). Boundary probabilities are rejected rather than clamped
// because either boundary makes downstream inverse-CDF calculations infinite and indicates
// invalid arrangement counts.
inline void set_strict_plotting_position(Data& data, double plotting_position) {
    if (!numerics::is_finite(plotting_position) || plotting_position <= 0.0 ||
        plotting_position >= 1.0) {
        throw std::runtime_error(
            "The Hirsch-Stedinger routine produced an invalid plotting position.");
    }
    data.set_plotting_position(plotting_position);
}

// Separates duplicate plotting positions produced by independent censored threshold bands
// (C# EnsureDistinctPlottingPositions). See the file header for the algorithm summary.
inline void ensure_distinct_plotting_positions(std::vector<Data*>& explicit_data) {
    if (explicit_data.size() < 2) return;

    std::vector<double> positions;
    positions.reserve(explicit_data.size());
    for (Data* d : explicit_data) positions.push_back(d->plotting_position());
    std::sort(positions.begin(), positions.end());

    bool has_duplicate = false;
    for (std::size_t i = 1; i < positions.size(); i++) {
        if (numerics::utilities::almost_equals(positions[i - 1], positions[i])) {
            has_duplicate = true;
            break;
        }
    }
    if (!has_duplicate) return;

    struct OrderedEntry {
        Data* source;
        double position;
        int ordinal;
    };
    std::vector<OrderedEntry> ordered;
    ordered.reserve(explicit_data.size());
    for (std::size_t i = 0; i < explicit_data.size(); i++) {
        ordered.push_back(
            {explicit_data[i], explicit_data[i]->plotting_position(), static_cast<int>(i)});
    }

    // A genuine total order (Index/Ordinal always resolve any remaining tie), so std::sort
    // is safe -- unlike the legacy above/below sort, this is NOT tie-permutation-sensitive.
    std::sort(ordered.begin(), ordered.end(), [](const OrderedEntry& left, const OrderedEntry& right) {
        if (left.position != right.position) return left.position < right.position;
        if (left.source->value() != right.source->value())
            return left.source->value() > right.source->value();  // descending value
        if (left.source->index() != right.source->index())
            return left.source->index() < right.source->index();
        return left.ordinal < right.ordinal;
    });

    std::size_t tie_start = 0;
    while (tie_start < ordered.size()) {
        std::size_t tie_end = tie_start;
        while (tie_end + 1 < ordered.size() &&
               numerics::utilities::almost_equals(ordered[tie_end].position,
                                                  ordered[tie_end + 1].position)) {
            tie_end++;
        }

        std::size_t tie_count = tie_end - tie_start + 1;
        if (tie_count > 1) {
            std::sort(ordered.begin() + static_cast<std::ptrdiff_t>(tie_start),
                     ordered.begin() + static_cast<std::ptrdiff_t>(tie_end) + 1,
                     [](const OrderedEntry& left, const OrderedEntry& right) {
                         if (left.source->value() != right.source->value())
                             return left.source->value() > right.source->value();
                         if (left.source->index() != right.source->index())
                             return left.source->index() < right.source->index();
                         return left.ordinal < right.ordinal;
                     });

            double tie_center = 0.0;
            for (std::size_t i = tie_start; i <= tie_end; i++) tie_center += ordered[i].position;
            tie_center /= static_cast<double>(tie_count);

            double lower_boundary =
                tie_start == 0 ? 0.0 : (ordered[tie_start - 1].position + tie_center) / 2.0;
            double upper_boundary = tie_end == ordered.size() - 1
                                        ? 1.0
                                        : (tie_center + ordered[tie_end + 1].position) / 2.0;
            double half_span = std::min(tie_center - lower_boundary, upper_boundary - tie_center);
            double increment = (2.0 * half_span) / (static_cast<double>(tie_count) + 1.0);
            double first_boundary = tie_center - half_span;

            if (!numerics::is_finite(increment) || increment <= 0.0) {
                throw std::runtime_error(
                    "Duplicate Hirsch-Stedinger plotting positions could not be separated.");
            }

            for (std::size_t i = 0; i < tie_count; i++) {
                double plotting_position = first_boundary + (static_cast<double>(i) + 1.0) * increment;
                set_strict_plotting_position(*ordered[tie_start + i].source, plotting_position);
            }
        }

        tie_start = tie_end + 1;
    }

    double previous = ordered[0].source->plotting_position();
    for (std::size_t i = 1; i < ordered.size(); i++) {
        double current = ordered[i].source->plotting_position();
        if (current <= previous || numerics::utilities::almost_equals(current, previous)) {
            throw std::runtime_error(
                "Duplicate Hirsch-Stedinger plotting positions could not be separated.");
        }
        previous = current;
    }
}

}  // namespace detail

// Provides plotting positions for censored data using the Hirsch-Stedinger plotting
// position formula (C# CalculatePlottingPositions). See the file header for the BestFit
// v2.0.0 ARRANGE2/PPLOT2/PLPOS rewrite summary and the validation/tie-ordering notes.
//
// References: Hirsch & Stedinger (1987) WRR; Cohn, Lane & Baier (1997) WRR; Cohn, Lane
// & Stedinger (2001) WRR; Bulletin 17C Appendix 5 (USGS 2018); the PeakfqSA FORTRAN
// source.
inline void DataFrame::calculate_plotting_positions() {
    double alpha = plotting_parameter_;
    if (!numerics::is_finite(alpha) || alpha < 0.0 || alpha >= 1.0) {
        throw std::runtime_error(
            "Plotting positions require a finite plotting parameter greater than or equal "
            "to zero and less than one.");
    }

    // NumberBelow and NumberAbove are inputs to the ARRANGE2 counts.
    process_threshold_series();
    ++plotting_position_version_;

    // Explicit observations, in the C# AddRange order (interval, then uncertain, then
    // exact) -- Ordinal (this vector's index) is a load-bearing tiebreaker downstream, so
    // this order must match exactly. Pointers into the owned series storage: the plotting-
    // position writes below must land on the real ordinates.
    std::vector<Data*> explicit_data;
    explicit_data.reserve(interval_series_.count() + uncertain_series_.count() +
                          exact_series_.count());
    for (std::size_t i = 0; i < interval_series_.count(); i++)
        explicit_data.push_back(&interval_series_[i]);
    for (std::size_t i = 0; i < uncertain_series_.count(); i++)
        explicit_data.push_back(&uncertain_series_[i]);
    for (std::size_t i = 0; i < exact_series_.count(); i++)
        explicit_data.push_back(&exact_series_[i]);

    // Validate every processed threshold, then sort by (StartIndex, EndIndex) and reject
    // overlapping windows.
    std::vector<const ThresholdData*> thresholds_by_index;
    thresholds_by_index.reserve(threshold_series_.count());
    for (std::size_t i = 0; i < threshold_series_.count(); i++) {
        const ThresholdData& threshold = threshold_series_[i];
        if (!numerics::is_finite(threshold.value()))
            throw std::runtime_error("Perception threshold values must be finite.");
        if (threshold.start_index() > threshold.end_index()) {
            throw std::runtime_error(
                "Perception threshold start indexes must not exceed their end indexes.");
        }
        long long total = static_cast<long long>(threshold.number_below()) +
                          static_cast<long long>(threshold.number_above());
        if (threshold.number_below() < 0 || threshold.number_above() < 0 ||
            total > threshold.duration()) {
            throw std::runtime_error(
                "Processed perception-threshold counts must be nonnegative and must not "
                "exceed the threshold duration.");
        }
        thresholds_by_index.push_back(&threshold);
    }

    std::sort(thresholds_by_index.begin(), thresholds_by_index.end(),
             [](const ThresholdData* left, const ThresholdData* right) {
                 if (left->start_index() != right->start_index())
                     return left->start_index() < right->start_index();
                 return left->end_index() < right->end_index();
             });

    for (std::size_t i = 1; i < thresholds_by_index.size(); i++) {
        if (thresholds_by_index[i]->start_index() <= thresholds_by_index[i - 1]->end_index())
            throw std::runtime_error("Perception threshold windows must not overlap.");
    }

    // Arrange every threshold level (the values of thresholds that still carry a processed
    // count) plus, for every explicit observation, the value of the threshold covering its
    // OWN index (or negative infinity when it is outside every window) -- the ARRANGE2
    // per-observation classification.
    std::unordered_set<int> occupied_indexes;
    std::set<double> threshold_level_set;

    struct PlottingObservation {
        Data* source;
        double threshold_value;
        int ordinal;
        bool is_detected;
    };
    std::vector<PlottingObservation> observations;
    observations.reserve(explicit_data.size());

    for (const ThresholdData* threshold : thresholds_by_index) {
        if (threshold->number_below() > 0 || threshold->number_above() > 0)
            threshold_level_set.insert(threshold->value());
    }

    for (std::size_t i = 0; i < explicit_data.size(); i++) {
        Data* source = explicit_data[i];
        if (!numerics::is_finite(source->value()))
            throw std::runtime_error("Explicit observation values must be finite.");

        occupied_indexes.insert(source->index());
        const ThresholdData* threshold =
            detail::find_threshold_for_plotting(thresholds_by_index, source->index());
        double threshold_value =
            threshold != nullptr ? threshold->value() : -std::numeric_limits<double>::infinity();
        threshold_level_set.insert(threshold_value);
        observations.push_back(
            {source, threshold_value, static_cast<int>(i), source->value() >= threshold_value});
    }

    std::vector<double> levels(threshold_level_set.begin(), threshold_level_set.end());

    if (!levels.empty()) {
        struct CensoredEntry {
            Data* source;
            int index;
            int ordinal;
        };
        std::vector<std::vector<Data*>> detected_by_level(levels.size());
        std::vector<std::vector<CensoredEntry>> censored_by_level(levels.size());
        std::vector<std::vector<int>> left_placeholder_indexes(levels.size());

        // Place NumberBelow/NumberAbove placeholders within each threshold's own window,
        // among its unoccupied indexes, validating feasibility. NumberAbove placeholders
        // need no per-index tracking (they are grouped generically at the top of the
        // highest band below), only a feasibility count.
        std::int64_t right_placeholder_count = 0;
        for (const ThresholdData* threshold : thresholds_by_index) {
            if (threshold->number_below() == 0 && threshold->number_above() == 0) continue;

            int level_index = detail::find_level_exact(levels, threshold->value());
            if (level_index < 0)
                throw std::runtime_error("A processed perception threshold could not be arranged.");

            bool track_left = threshold->number_above() > 0;
            std::unordered_set<int> selected_left_indexes;
            int selected_below = 0;
            for (long long candidate = threshold->start_index();
                 candidate <= threshold->end_index() && selected_below < threshold->number_below();
                 candidate++) {
                int index = static_cast<int>(candidate);
                if (occupied_indexes.count(index) != 0) continue;
                left_placeholder_indexes[static_cast<std::size_t>(level_index)].push_back(index);
                if (track_left) selected_left_indexes.insert(index);
                selected_below++;
            }
            if (selected_below != threshold->number_below()) {
                throw std::runtime_error(
                    "The processed number below a perception threshold exceeds its "
                    "available unoccupied indexes.");
            }

            int selected_above = 0;
            for (long long candidate = threshold->end_index();
                 candidate >= threshold->start_index() && selected_above < threshold->number_above();
                 candidate--) {
                int index = static_cast<int>(candidate);
                if (occupied_indexes.count(index) != 0 ||
                    (track_left && selected_left_indexes.count(index) != 0)) {
                    continue;
                }
                selected_above++;
            }
            if (selected_above != threshold->number_above()) {
                throw std::runtime_error(
                    "The processed number above a perception threshold exceeds its "
                    "available unoccupied indexes.");
            }

            right_placeholder_count += threshold->number_above();
        }

        // Split detected observations into the legacy above/below lists (see the file
        // header's TIE ORDERING note), partitioned at the global minimum finite threshold
        // -- NOT the observation's own threshold, which already decided IsDetected above.
        std::vector<Data*> legacy_above_order;
        std::vector<Data*> legacy_below_order;
        double minimum_finite_threshold = std::numeric_limits<double>::infinity();
        for (const ThresholdData* threshold : thresholds_by_index)
            minimum_finite_threshold = std::min(minimum_finite_threshold, threshold->value());

        for (const PlottingObservation& observation : observations) {
            if (observation.is_detected) {
                if (observation.source->value() >= minimum_finite_threshold)
                    legacy_above_order.push_back(observation.source);
                else
                    legacy_below_order.push_back(observation.source);
            } else {
                int level_index = detail::find_level_exact(levels, observation.threshold_value);
                if (level_index < 0) {
                    throw std::runtime_error(
                        "A censored observation threshold could not be arranged.");
                }
                censored_by_level[static_cast<std::size_t>(level_index)].push_back(
                    {observation.source, observation.source->index(), observation.ordinal});
            }
        }

        // Preserve only the legacy tie permutation (value-only, introsort-ported).
        auto descending_by_value = [](const Data* x, const Data* y) {
            return -detail::compare_double(x->value(), y->value());
        };
        detail::dotnet_list_sort(legacy_above_order, descending_by_value);
        detail::dotnet_list_sort(legacy_below_order, descending_by_value);

        for (int group = 0; group < 2; group++) {
            std::vector<Data*>& ordered = group == 0 ? legacy_above_order : legacy_below_order;
            for (Data* observation : ordered) {
                int level_index = detail::find_plotting_level(levels, observation->value());
                detected_by_level[static_cast<std::size_t>(level_index)].push_back(observation);
            }
        }

        std::vector<std::int64_t> detected_count(levels.size());
        std::vector<std::int64_t> censored_count(levels.size());
        for (std::size_t i = 0; i < levels.size(); i++) {
            detected_count[i] = static_cast<std::int64_t>(detected_by_level[i].size());
            censored_count[i] = static_cast<std::int64_t>(censored_by_level[i].size()) +
                                static_cast<std::int64_t>(left_placeholder_indexes[i].size());
        }
        // NumberAbove entries are right-censored values. PLPOS orders them above every
        // finite observation, so they occupy the end of the highest band.
        detected_count.back() += right_placeholder_count;

        // ARRANGE2 cumulative "not observable" recurrence.
        std::vector<std::int64_t> not_observable_at_level(levels.size());
        not_observable_at_level[0] = censored_count[0];
        for (std::size_t i = 1; i < levels.size(); i++) {
            not_observable_at_level[i] =
                not_observable_at_level[i - 1] + censored_count[i] + detected_count[i - 1];
        }

        // PPLOT2 nonexceedance interval boundaries, computed top-down from high to low
        // thresholds.
        std::vector<double> interval_boundary(levels.size() + 1, 0.0);
        for (std::size_t k = levels.size(); k-- > 0;) {
            double conditional_detection_probability =
                static_cast<double>(detected_count[k]) /
                (static_cast<double>(std::max<std::int64_t>(1, detected_count[k])) +
                 static_cast<double>(not_observable_at_level[k]));
            interval_boundary[k] = interval_boundary[k + 1] +
                                  (1.0 - interval_boundary[k + 1]) * conditional_detection_probability;
        }

        for (std::size_t i = 0; i < levels.size(); i++) {
            const std::vector<Data*>& detected = detected_by_level[i];
            double denominator = static_cast<double>(detected_count[i]) + 1.0 - 2.0 * alpha;
            for (std::size_t j = 0; j < detected.size(); j++) {
                double nonexceedance_probability =
                    (1.0 - interval_boundary[i]) +
                    (interval_boundary[i] - interval_boundary[i + 1]) *
                        ((static_cast<double>(detected.size()) - static_cast<double>(j) - alpha) /
                         denominator);
                detail::set_strict_plotting_position(*detected[j], 1.0 - nonexceedance_probability);
            }

            std::vector<CensoredEntry>& censored = censored_by_level[i];
            std::sort(censored.begin(), censored.end(),
                     [](const CensoredEntry& left, const CensoredEntry& right) {
                         if (left.index != right.index) return left.index < right.index;
                         return left.ordinal < right.ordinal;
                     });

            std::vector<int>& placeholder_indexes = left_placeholder_indexes[i];
            std::sort(placeholder_indexes.begin(), placeholder_indexes.end());

            std::int64_t rank = 0;
            std::size_t placeholder_cursor = 0;
            denominator = static_cast<double>(censored_count[i]) + 1.0 - 2.0 * alpha;
            for (const CensoredEntry& entry : censored) {
                while (placeholder_cursor < placeholder_indexes.size() &&
                       placeholder_indexes[placeholder_cursor] < entry.index) {
                    placeholder_cursor++;
                    rank++;
                }
                rank++;
                double nonexceedance_probability =
                    (1.0 - interval_boundary[i]) * ((static_cast<double>(rank) - alpha) / denominator);
                detail::set_strict_plotting_position(*entry.source, 1.0 - nonexceedance_probability);
            }
        }
    }

    detail::ensure_distinct_plotting_positions(explicit_data);
}

// Apply the Langbein conversion to the plotting positions (C# ApplyLangbeinConversion,
// line 1458). `lambda` is the number of events per block. The SuppressCollectionChanged
// bracketing and RaisePropertyChange are WPF event plumbing with no C++ counterpart.
inline void DataFrame::apply_langbein_conversion(double lambda) {
    for (std::size_t i = 0; i < exact_series_.count(); i++) {
        exact_series_[i].set_plotting_position(
            1 - std::exp(-lambda * exact_series_[i].plotting_position()));
    }
    for (std::size_t i = 0; i < uncertain_series_.count(); i++) {
        uncertain_series_[i].set_plotting_position(
            1 - std::exp(-lambda * uncertain_series_[i].plotting_position()));
    }
    for (std::size_t i = 0; i < interval_series_.count(); i++) {
        interval_series_[i].set_plotting_position(
            1 - std::exp(-lambda * interval_series_[i].plotting_position()));
    }
}

}  // namespace corehydro::models
