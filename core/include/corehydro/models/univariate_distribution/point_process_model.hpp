// ported from: src/RMC.BestFit/Models/UnivariateDistribution/PointProcessModel.cs @ c2e6192
//
// Point-process model for peaks-over-threshold (POT) data: an underlying Poisson process for
// exceedance times and GEV (or seasonal two-GEV) models for magnitudes, wrapped around the
// ported Numerics `CompetingRisks` with GEV-only components. Supports censored, interval,
// threshold, and uncertain data, parameter priors, and optional quantile priors. Derives
// from the M8 `UnivariateDistributionModelBase` and implements
// `ISimulatable<std::vector<double>>` (C# ISimulatable<double[]>) and `IUnivariateModel`.
// Sibling of the M9/M10/M11 models; mirrors the C# method-for-method.
//
// Nullability mapping:
//   - C# `CompetingRisks? Distribution` is genuinely nullable, so the wrapped distribution
//     is a `std::unique_ptr<CompetingRisks>`; `distribution()` returns the raw pointer (the
//     M10 convention; it doubles as the covariant IUnivariateModel::distribution()
//     override).
//   - C# `DataFrame = null` maps to the base's never-set optional (see the base header).
//
// THE SEASONAL DATA PATH landed in P6, un-gated by the ported Numerics `TimeSeries` container
// and the `DateTime` value type under it (and by `ExactData.DateTime`, deferred with them). Both
// members it covered are live:
//   - the irregular-TimeSeries branch of SetAMSData (C# lines 526-556: CreateBlockSeries,
//     ShiftDatesByMonth, water-year shifting, the POT day-of-year `_potDays` population).
//     Through P5 this branch was a silent no-op, so the AMS frame and pot_days() stayed empty
//     when IsSeasonal was true, and every downstream consequence (NaN seasonal defaults,
//     vanishing seasonal likelihood terms, a Validate() complaint) followed from that. They no
//     longer do. The C# swallow-all catch around the body is still mirrored, and is now
//     LOAD-BEARING: this method runs from the StartMonth setter, so an out-of-range month
//     reaches CustomYearSeries's guard and must be swallowed for Validate() to report it.
//   - `GeneratePOTTimeSeries(DateTime, double, int)` (C# line 2017), the synthetic
//     peaks-over-threshold record. Its private `SamplePoisson` helper was already ported (the
//     M14 seeded-stream work needed the complete RNG surface); P6 added the caller.
// Everything else seasonal -- the IsSeasonal/TimeBlock/StartMonth properties and setter
// cascades, the two-GEV SetDistribution branch, the change-point parameters, the seasonal
// branches of every likelihood method, SetParameterValues/GetDistribution's seasonal
// marginal-correction transform, and the seasonal Validate checks -- is pure computation
// with no TimeSeries/DateTime dependency and is PORTED in full.
//
// SetAMSData (non-seasonal) ordering note: the C# builds
// `ExactSeries.GroupBy(item => item.Index).ToDictionary(k, max)` and enumerates the
// Dictionary. LINQ GroupBy is documented to yield groups in first-appearance order of the
// key; ToDictionary inserts in that order, and a Dictionary<K,V> that has only been
// inserted into enumerates in insertion order. The observable output order is therefore
// FIRST-APPEARANCE ORDER of each index in the exact series, which this port reproduces
// explicitly.
//
// Clone(): C# aliases the DataFrame into `new PointProcessModel(DataFrame, Distribution!)`;
// the value-typed move-only C++ frame is DEEP-COPIED instead (M9/M10/M11 precedent), and a
// model with no frame or a null distribution cannot be cloned (std::invalid_argument -- the
// C# would NRE through the chained ctor). M9-lesson re-sync audit: NOT needed here -- the
// ctor clones the wrapped CompetingRisks with its full parameter state via the Numerics
// Clone(), and no later side effect resets it. Like the C#, the clone's Lambda is the
// ctor-derived value (the C# object initializer writes `_totalYears` without re-running
// CalculateLambda) and `_totalYearsExplicit` resets to false.
//
// SKIPPED (project-wide deferrals): the XElement ctor (C# line 71) and ToXElement (line
// 1736) -- XML serialization; INotifyPropertyChanged / RaisePropertyChange / event
// (un)subscription; the [Category]/[DisplayName]/[Description]/[Browsable] attributes.
// The C# DataFrame_PropertyChanged handler (line 390) ports as the explicit-invalidation
// data_frame_property_changed() override (the M4->M8 cadence; see the base header).
// ModelParameter.Name for GEV parameters: the C# reads `ParametersToString[j, 0]`;
// ParametersToString is not on the ported distribution base (Phase 4 decision,
// display-only), so GEV parameter names stay "" while the change-point names
// ("Change Point K<sub>") and OwnerName ("D1"/"D2" when seasonal) port fully. The
// pointwise Jeffreys labels port verbatim ("Jeffreys Scale: Scale" /
// "Jeffreys Scale: D<j>.Scale" -- unlike M11, the C# hardcodes "Scale" here). The
// unsupported-component Validate message carries the component index but not the enum name
// (no enum-name surface on the ported type; M10/M11 precedent).
//
// EXCEPTION-TYPE MAPPING for THIS file: C# ArgumentNullException/ArgumentException ->
// std::invalid_argument; ArgumentOutOfRangeException -> std::out_of_range;
// InvalidOperationException (and the C# `Distribution!` NullReferenceException path in
// SetParameterValues, plus the IStandardError InvalidCastException path in the
// multi-quantile prior branch) -> std::runtime_error.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
#include "corehydro/numerics/data/probability.hpp"
#include "corehydro/numerics/data/time_series/support/time_block_window.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "corehydro/numerics/utilities/dotnet_sort.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/competing_risks.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/generalized_extreme_value.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/math/integration/integration.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class PointProcessModel : public UnivariateDistributionModelBase,
                          public ISimulatable<std::vector<double>>,
                          public IUnivariateModel {
   public:
    using DistributionBase = numerics::distributions::UnivariateDistributionBase;
    using DistributionType = numerics::distributions::UnivariateDistributionType;
    using CompetingRisks = numerics::distributions::CompetingRisks;
    using GeneralizedExtremeValue = numerics::distributions::GeneralizedExtremeValue;
    using TimeBlockWindow = numerics::data::TimeBlockWindow;

    // --- Construction (C# region, lines 42-124) ---------------------------------------------

    // C# parameterless ctor (line 48): a default (non-seasonal) competing risks
    // distribution via SetDistribution() -- a FIELD assignment in C#, so no
    // Distribution-setter side effects run here.
    PointProcessModel() { set_distribution(); }

    // C# `PointProcessModel(DataFrame, CompetingRisks)` (line 59): the Distribution
    // PROPERTY setter first (clone + SetDefaultParameters/SetDefaultQuantilePriors, both
    // effectively empty with no frame yet), then the DataFrame setter cascade.
    PointProcessModel(DataFrame data_frame, const CompetingRisks& distribution) {
        set_distribution(clone_competing_risks(distribution));
        set_data_frame(std::move(data_frame));
    }

    // SKIPPED: the XElement ctor (C# line 71) -- XML serialization is a project-wide
    // deferral.

    // Move-only, like the base (deep copies go through clone()).
    PointProcessModel(PointProcessModel&&) = default;
    PointProcessModel& operator=(PointProcessModel&&) = default;

    // --- Members (C# region, lines 126-385) ------------------------------------------------

    // C# `IsSupportedDistributionType` (line 145): the point process likelihood requires
    // Generalized Extreme Value (GEV) distributions (the C# _supportedDistributionTypes
    // HashSet, line 132).
    static bool is_supported_distribution_type(DistributionType distribution_type) {
        return distribution_type == DistributionType::GeneralizedExtremeValue;
    }

    // C# `DataFrame` override (line 171): store, reset the explicit-TotalYears flag (the
    // new record invalidates a previously explicit exposure), reprocess thresholds at the
    // model boundary (the M4->M8 cadence), rebuild the AMS data, then the UseDefaults /
    // UseDefaultFlatPriors cascades in C# statement order.
    void set_data_frame(DataFrame data_frame) override {
        data_frame_ = std::move(data_frame);

        // The user is supplying new data; the previously-explicit TotalYears (if any) no
        // longer reflects this record.
        total_years_explicit_ = false;

        data_frame_->process_threshold_series();
        set_ams_data();

        if (use_defaults_) set_default_threshold_and_total_years(std::nullopt, true);

        if (use_default_flat_priors()) set_default_parameters();
    }

    // IUnivariateModel's data_frame() accessors (the base's non-virtual accessors carry the
    // same unguarded-deref contract; check has_data_frame() first where the frame may be
    // absent).
    DataFrame& data_frame() override { return *data_frame_; }
    const DataFrame& data_frame() const override { return *data_frame_; }

    // C# `Distribution` property (line 212; nullable -- nullptr == C# null). The const
    // accessor is the covariant IUnivariateModel::distribution() override (the C# explicit
    // `IUnivariateModel.Distribution => _distribution`, line 225).
    bool has_distribution() const { return distribution_ != nullptr; }
    CompetingRisks* distribution() { return distribution_.get(); }
    const CompetingRisks* distribution() const override { return distribution_.get(); }

    // C# setter (line 215): assign, then SetDefaultParameters + SetDefaultQuantilePriors
    // (unconditionally, even for null).
    void set_distribution(std::unique_ptr<CompetingRisks> distribution) {
        distribution_ = std::move(distribution);
        set_default_parameters();
        set_default_quantile_priors();
    }

    // C# `IsNonstationary => false` (line 233): POT models in RMC-BestFit are currently
    // stationary; exceedance intensity and mark distribution parameters are constant.
    bool is_nonstationary() const override { return false; }

    // C# `AMSDataFrame` (line 238): the pre-processed annual maximum (block) data frame
    // derived from the input POT data.
    const DataFrame& ams_data_frame() const { return ams_data_frame_; }

    // C# `POTDays` (line 243): the day-of-year for each POT event. Always empty in this
    // port (populated by the seasonal data path in set_ams_data()).
    const std::vector<int>& pot_days() const { return pot_days_; }

    // C# `Lambda` (line 248): the average number of POT events per year.
    double lambda() const { return lambda_; }

    // C# `Threshold` (line 257): the threshold above which events are modeled by the point
    // process. The setter is guarded by AlmostEquals (no side effects beyond the store).
    double threshold() const { return threshold_; }
    void set_threshold(double value) {
        if (!almost_equals(threshold_, value)) {
            threshold_ = value;
        }
    }

    // C# `TotalYears` (line 277): the total number of years over which POT events were
    // observed. An (almost-)changed value marks the exposure as explicitly set and
    // refreshes Lambda.
    double total_years() const { return total_years_; }
    void set_total_years(double value) {
        if (!almost_equals(total_years_, value)) {
            total_years_ = value;
            total_years_explicit_ = true;  // user (or caller) explicitly set the value
            calculate_lambda();
        }
    }

    // C# `UseDefaults` (line 300): whether the default threshold and total years should be
    // inferred from the data. Turning it on re-applies the defaults immediately.
    bool use_defaults() const { return use_defaults_; }
    void set_use_defaults(bool value) {
        if (use_defaults_ != value) {
            use_defaults_ = value;
            if (use_defaults_) set_default_threshold_and_total_years(std::nullopt, true);
        }
    }

    // C# `IsSeasonal` (line 323): seasonal (two GEVs) vs non-seasonal (single GEV). The
    // setter rebuilds the distribution, the AMS data, and the default parameters.
    bool is_seasonal() const { return is_seasonal_; }
    void set_is_seasonal(bool value) {
        if (is_seasonal_ != value) {
            is_seasonal_ = value;
            set_distribution();
            set_ams_data();
            set_default_parameters();
        }
    }

    // C# `TimeBlock` (line 346): the time block window used to build the block series
    // (default water year). Consumed by the seasonal data path; the property
    // and its setter cascade port in full.
    TimeBlockWindow time_block() const { return time_block_; }
    void set_time_block(TimeBlockWindow value) {
        if (time_block_ != value) {
            time_block_ = value;
            set_ams_data();
            if (use_default_flat_priors()) set_default_parameters();
        }
    }

    // C# `StartMonth` (line 369): the starting month for the water year (default October).
    int start_month() const { return start_month_; }
    void set_start_month(int value) {
        if (start_month_ != value) {
            start_month_ = value;
            set_ams_data();
            if (use_default_flat_priors()) set_default_parameters();
        }
    }

    // --- Methods (C# region, lines 387-2138) ------------------------------------------------

    // C# `SetDefaultThresholdAndTotalYears(double? peaksOverThreshold = null, bool
    // forceTotalYears = false)` (line 425). Threshold: the supplied POT threshold when it
    // is finite and below every exact event, otherwise BitDecrement(exact minimum); with no
    // exact data, BitDecrement of the robust minimum across the non-empty uncertain /
    // interval series. TotalYears: the exact-series index span (Poisson exposure for exact
    // POT events only -- threshold/uncertain/interval data contribute to the censored
    // magnitude likelihood but not the exposure duration), written directly to the field
    // (no explicit-flag side effect) and only when forced or not explicitly set. Lambda is
    // refreshed BEFORE the C# TotalYears change notification (the ordering rule the
    // upstream Test_..._TotalYearsEventSeesUpdatedLambda pins).
    void set_default_threshold_and_total_years(
        std::optional<double> peaks_over_threshold = std::nullopt,
        bool force_total_years = false) {
        if (!has_data_frame()) return;

        if (data_frame().exact_series().count() > 0) {
            double exact_minimum = data_frame().exact_series().minimum_value();
            if (peaks_over_threshold.has_value() &&
                numerics::is_finite(*peaks_over_threshold) &&
                *peaks_over_threshold < exact_minimum) {
                set_threshold(*peaks_over_threshold);
            } else {
                set_threshold(bit_decrement(exact_minimum));
            }
        } else {
            // Robust min across non-empty non-exact series; ignore series with no data.
            std::vector<double> minima;

            if (data_frame().uncertain_series().count() > 0)
                minima.push_back(data_frame().uncertain_series().minimum_value());
            if (data_frame().interval_series().count() > 0)
                minima.push_back(data_frame().interval_series().minimum_value());

            if (!minima.empty())
                set_threshold(bit_decrement(*std::min_element(minima.begin(), minima.end())));
        }

        // TotalYears is the Poisson exposure duration for exact POT events only. If exact
        // events miss the first/last observed years, this is only a heuristic; users can
        // turn off UseDefaults and enter the better exposure.
        if (force_total_years || !total_years_explicit_) {
            double inferred = data_frame().exact_series().count() > 0
                                  ? static_cast<double>(data_frame().exact_series().index_span())
                                  : 1.0;

            if (!almost_equals(total_years_, inferred)) {
                total_years_ = inferred;
                // (The C# raises the TotalYears change notification AFTER CalculateLambda
                // below; no events in this port.)
            }
            total_years_explicit_ = false;
        }

        calculate_lambda();
    }

    // C# `CalculateLambda` (line 484): the average number of POT events per year, NaN when
    // there is no frame or no positive exposure.
    void calculate_lambda() {
        if (!has_data_frame() || std::isnan(total_years_) || total_years_ <= 0) {
            lambda_ = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        double events = static_cast<double>(data_frame().exact_series().count() +
                                            data_frame().uncertain_series().count() +
                                            data_frame().interval_series().count());

        lambda_ = events / total_years_;
    }

    // C# `SetAMSData` (line 504): preprocesses the annual maximum (block) data. The
    // non-seasonal branch groups the exact series by index and keeps the per-index maximum,
    // in first-appearance order of the index (the GroupBy/ToDictionary reading in the file
    // header). The seasonal branch (P6) builds an irregular series from the ordinates' dates,
    // reduces it to block maxima, and records each event's day of year. The C# wraps the whole
    // body in a swallow-all catch (Debug.WriteLine only) so imperfect time-series data never
    // crashes the model, and that catch is LOAD-BEARING now that the seasonal branch is live:
    // this method runs from the StartMonth setter, so assigning an out-of-range month reaches
    // CustomYearSeries's own guard and the throw has to be swallowed for Validate() to be the
    // thing that reports it. Mirrored below.
    void set_ams_data() {
        ams_data_frame_ = DataFrame();
        pot_days_.clear();

        if (!has_data_frame() || data_frame().exact_series().count() == 0) {
            return;
        }

        try {
        if (!is_seasonal_) {
            // Get block-maximum values over each unique index (first-appearance order).
            std::vector<std::pair<int, double>> max_by_index;
            std::unordered_map<int, std::size_t> position;
            for (std::size_t i = 0; i < data_frame().exact_series().count(); ++i) {
                const ExactData& item = data_frame().exact_series()[i];
                auto found = position.find(item.index());
                if (found == position.end()) {
                    position.emplace(item.index(), max_by_index.size());
                    max_by_index.emplace_back(item.index(), item.value());
                } else if (item.value() > max_by_index[found->second].second) {
                    max_by_index[found->second].second = item.value();
                }
            }
            for (const auto& item : max_by_index) {
                ams_data_frame_.exact_series().add(ExactData(item.first, item.second));
            }
        } else {
            // The seasonal branch (C# lines 526-556), un-gated by the P6 TimeSeries container:
            // build an irregular series from the exact data's dates, reduce it to block maxima,
            // and record each event's day of the year.
            numerics::data::TimeSeries ts(numerics::data::TimeInterval::Irregular);
            for (std::size_t i = 0; i < data_frame().exact_series().count(); ++i) {
                const ExactData& exact = data_frame().exact_series()[i];
                numerics::data::DateTime dt = exact.date_time();
                // An ordinate built from a bare index carries no date (tick 0), so the C#
                // substitutes January 1 of its index year.
                if (dt == numerics::data::DateTime()) dt = numerics::data::DateTime(exact.index(), 1, 1);
                ts.add(numerics::data::TimeSeries::Ordinate(dt, exact.value()));
            }

            ams_data_frame_.create_block_series(ts, time_block_,
                                                numerics::data::BlockFunctionType::Maximum,
                                                numerics::data::SmoothingFunctionType::None,
                                                start_month_);

            // The day of year is read AFTER the water-year shift, so a POT day is expressed in
            // the same year convention the block maxima were taken in.
            if (time_block_ == numerics::data::TimeBlockWindow::WaterYear) {
                int shift = start_month_ != 1 ? 12 - start_month_ + 1 : 0;
                if (start_month_ != 1) ts = ts.shift_dates_by_month(shift);
            }

            // UPSTREAM DEFECT, mirrored (see docs/upstream-csharp-issues.md): on the water-year
            // path `ts` has just been through ShiftDatesByMonth, which SORTS BY TIME, so
            // `pot_days_` comes out in DATE order -- while the seasonal likelihood pairs it
            // POSITIONALLY with the exact series, which is in whatever order the caller supplied.
            // A record that is not already date-sorted therefore has each magnitude scored
            // against another event's day of year. Do not "fix" it here; the fix belongs upstream
            // and would move every seasonal oracle.
            for (int i = 0; i < ts.count(); ++i) pot_days_.push_back(ts[i].index().day_of_year());
        }
        } catch (const std::exception&) {
            // C# swallows every exception here (Debug.WriteLine only): the AMS frame and the POT
            // day list are simply left as they are, which is what Validate() then reports.
        }
    }

    // C# `SetDistribution` (line 572): (re)builds the underlying competing risks
    // distribution with one (non-seasonal) or two (seasonal) GEV components. A FIELD
    // assignment in C# -- the Distribution property-setter side effects do NOT run.
    void set_distribution() {
        std::vector<std::unique_ptr<DistributionBase>> components;
        components.push_back(std::make_unique<GeneralizedExtremeValue>());
        if (is_seasonal_) components.push_back(std::make_unique<GeneralizedExtremeValue>());

        auto dist = std::make_unique<CompetingRisks>(std::move(components));
        dist->minimum_of_random_variables = false;
        dist->set_dependency(numerics::data::probability::DependencyType::Independent);
        distribution_ = std::move(dist);
    }

    // C# `SetDefaultParameters` override (line 593): seasonal change-point parameters (in
    // day-of-year) when there are two components, then per component the constraint-driven
    // GEV ModelParameters (OwnerName "D<i>" when seasonal, Uniform(lower, upper) prior,
    // IsPositive when the lower bound is the machine epsilon), with the constraint sample
    // taken from the AMS (block maxima) frame.
    void set_default_parameters() override {
        // (Handler removal/re-add is INPC plumbing, skipped.)
        parameters_.clear();

        if (distribution_ == nullptr || !has_data_frame() ||
            !data_frame().validate().is_valid || distribution_->component_count() == 0) {
            return;
        }

        // Seasonal change-point parameters (in day-of-year).
        if (distribution_->component_count() > 1) {
            parameters_.emplace_back(
                "", "Change Point K" + to_subscript(1), 90.0, 10.0, 170.0,
                std::make_unique<numerics::distributions::Uniform>(10.0, 170.0));
            parameters_.emplace_back(
                "", "Change Point K" + to_subscript(2), 250.0, 171.0, 330.0,
                std::make_unique<numerics::distributions::Uniform>(171.0, 330.0));
        }

        // Priors for GEV distribution parameters.
        std::vector<double> sample;
        sample.reserve(ams_data_frame_.exact_series().count());
        for (std::size_t i = 0; i < ams_data_frame_.exact_series().count(); ++i)
            sample.push_back(ams_data_frame_.exact_series()[i].value());

        for (int i = 0; i < distribution_->component_count(); ++i) {
            const DistributionBase& gev = distribution_->component(i);
            const auto& mle_component =
                dynamic_cast<const numerics::distributions::IMaximumLikelihoodEstimation&>(
                    gev);

            std::vector<double> initials, lowers, uppers;
            mle_component.get_parameter_constraints(sample, initials, lowers, uppers);

            // Name stays "" -- ParametersToString is not on the ported distribution base
            // (see the file header).
            for (int j = 0; j < gev.number_of_parameters(); ++j) {
                std::size_t sj = static_cast<std::size_t>(j);
                parameters_.emplace_back(
                    is_seasonal_ ? "D" + std::to_string(i + 1) : "", "", initials[sj],
                    lowers[sj], uppers[sj],
                    std::make_unique<numerics::distributions::Uniform>(lowers[sj],
                                                                       uppers[sj]),
                    /*is_positive=*/lowers[sj] == numerics::kDoubleMachineEpsilon);
            }
        }
    }

    // C# `SetDefaultQuantilePriors` override (line 673): null distribution -> no-op;
    // disabled -> clear both lists; enabled -> one LnNormal prior per allowed quantile
    // (alpha = 0.1, 0.01, ... with mu = Round(InverseCDF(1 - alpha), 2) and sigma =
    // Round(0.15 * mu, 2)); a single GEV with UseSingleQuantile off allows up to three
    // (one per parameter). An existing list is trimmed from the back or extended by
    // refining the last alpha by a factor of 10. Ends with ProcessQuantilePriors().
    void set_default_quantile_priors() override {
        if (distribution_ == nullptr) return;

        // (Handler removal for the old priors is INPC plumbing, skipped.)

        // If quantile priors are not enabled, clear the list and return.
        if (!enable_quantile_priors()) {
            quantile_priors_.clear();
            quantile_priors_true_.clear();
            return;
        }

        std::vector<QuantilePrior> priors;
        std::size_t q_count = 1;

        // For single GEV, allow up to 3 quantile priors (one per parameter).
        if (!use_single_quantile() && distribution_->component_count() == 1) {
            q_count = 3;
        }

        // Respect existing priors when possible.
        if (quantile_priors_.size() >= 1) {
            priors = quantile_priors_;

            // Trim excess priors.
            while (priors.size() > q_count) priors.pop_back();

            // Add missing priors by refining the last alpha.
            while (priors.size() < q_count) {
                priors.emplace_back(priors.back().alpha() / 10.0,
                                    std::make_unique<numerics::distributions::LnNormal>());
                double mu = round_half_even_2(
                    distribution_->inverse_cdf(1.0 - priors.back().alpha()));
                double sigma = round_half_even_2(mu * 0.15);
                priors.back().distribution().set_parameters({mu, sigma});
            }
        } else {
            // Create default priors: alpha = 0.1, 0.01, 0.001, ... up to qCount.
            for (std::size_t i = 1; i <= q_count; ++i) {
                priors.emplace_back(std::pow(10.0, -static_cast<double>(i)),
                                    std::make_unique<numerics::distributions::LnNormal>());
                double mu = round_half_even_2(
                    distribution_->inverse_cdf(1.0 - priors[i - 1].alpha()));
                double sigma = round_half_even_2(mu * 0.15);
                priors[i - 1].distribution().set_parameters({mu, sigma});
            }
        }

        // Reset the quantile priors with the new list (handler re-adds skipped).
        quantile_priors_ = std::move(priors);

        process_quantile_priors();
    }

    // C# `ProcessQuantilePriors` override (line 744): a single prior is cloned through; the
    // single-GEV three-prior configuration keeps the first prior and converts the remaining
    // priors to Gamma distributions on the differences between consecutive random
    // quantiles.
    void process_quantile_priors() override {
        quantile_priors_true_.clear();

        if (quantile_priors_.size() == 1) {
            quantile_priors_true_.push_back(quantile_priors_[0].clone());
            return;
        }

        if (!use_single_quantile() && distribution_ != nullptr &&
            distribution_->component_count() == 1 && quantile_priors_.size() == 3) {
            // First quantile prior remains unchanged.
            quantile_priors_true_.push_back(quantile_priors_[0].clone());

            // Convert remaining priors to gamma distributions on the differences between
            // random quantiles.
            for (std::size_t i = 1; i < quantile_priors_.size(); ++i) {
                double mu_diff = quantile_priors_[i].distribution().mean() -
                                 quantile_priors_[i - 1].distribution().mean();
                double sigma_diff =
                    std::sqrt(quantile_priors_[i].distribution().variance() +
                              quantile_priors_[i - 1].distribution().variance());

                auto gamma_dist =
                    std::make_unique<numerics::distributions::GammaDistribution>();
                gamma_dist->set_parameters(
                    gamma_dist->parameters_from_moments({mu_diff, sigma_diff}));

                quantile_priors_true_.emplace_back(quantile_priors_[i].alpha(),
                                                   std::move(gamma_dist));
            }
        }
    }

    // C# `LogLikelihood` override (line 786): explicit Data + Prior with non-finite results
    // collapsed to -infinity at every step (the C# remark: the base implementation's
    // IsFinite check is bypassed by a NegativeInfinity sentinel, so this override forces
    // the canonical contract).
    double log_likelihood(std::vector<double>& parameters) const override {
        const double neg_inf = -std::numeric_limits<double>::infinity();

        double data_log_lh = data_log_likelihood(parameters);
        if (!numerics::is_finite(data_log_lh) || data_log_lh <= neg_inf) return neg_inf;

        double prior_log_lh = prior_log_likelihood(parameters);
        if (!numerics::is_finite(prior_log_lh) || prior_log_lh <= neg_inf) return neg_inf;

        double log_lh = data_log_lh + prior_log_lh;
        return numerics::is_finite(log_lh) ? log_lh : neg_inf;
    }

    // C# `DataLogLikelihood` override (line 801): the Poisson-GEV point-process likelihood
    // (Coles parameterization: shape = -kappa) over the exact data plus the rate term
    // -Ny*G(u), with the seasonal variant splitting the exposure by the change points; the
    // uncertain / interval / threshold data are evaluated against the composite
    // (competing-risks) likelihood -- they are block-indexed, not date-indexed, so
    // day-of-year season dispatch does not apply.
    double data_log_likelihood(std::vector<double>& parameters) const override {
        const double neg_inf = -std::numeric_limits<double>::infinity();

        if (distribution_ == nullptr || !has_data_frame()) return neg_inf;

        std::unique_ptr<DistributionBase> model_owner = distribution_->clone();
        CompetingRisks& model = static_cast<CompetingRisks&>(*model_owner);
        double log_lh = 0.0;

        double u = threshold_;
        double n_y = total_years_;

        if (std::isnan(u) || std::isnan(n_y) || n_y <= 0) return neg_inf;

        double k1 = 0.0;
        double k2 = 0.0;
        double ny1 = 0.0;
        double ny2 = 0.0;

        // Set model parameters (accounting for seasonal transformation).
        if (!is_seasonal_) {
            model.set_parameters(parameters);
        } else {
            if (parameters.size() < 8) return neg_inf;

            k1 = parameters[0];
            k2 = parameters[1];

            if (!(k1 >= 1 && k1 <= 366 && k2 >= 1 && k2 <= 366 && k1 < k2)) return neg_inf;

            ny1 = n_y * (k1 + (366.0 - k2)) / 366.0;
            ny2 = n_y * (k2 - k1) / 366.0;

            model.set_parameters(skip_first_two(parameters));
        }

        const DataFrame& df = data_frame();

        // Exact data contribution.
        if (!is_seasonal_) {
            const auto* gev =
                dynamic_cast<const GeneralizedExtremeValue*>(&model.component(0));
            if (gev == nullptr) return neg_inf;

            double loc = gev->xi();
            double scl = gev->alpha();
            double shp = -gev->kappa();  // Convert to Coles parameterization

            if (scl <= 0.0) return neg_inf;

            if (std::fabs(shp) < 1E-4) {
                // Gumbel (shape approximately zero).
                for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
                    double x = df.exact_series()[i].value();
                    log_lh += -std::log(scl) - (x - loc) / scl;
                }

                log_lh += -n_y * std::exp(-(u - loc) / scl);
            } else {
                // General case (shape != 0).
                for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
                    double x = df.exact_series()[i].value();
                    double z = 1.0 + shp * ((x - loc) / scl);
                    if (z <= 0.0) return neg_inf;

                    log_lh += -std::log(scl) - (1.0 + 1.0 / shp) * std::log(z);
                }

                double zu = 1.0 + shp * ((u - loc) / scl);
                if (zu <= 0.0) return neg_inf;

                log_lh += -n_y * std::pow(zu, -1.0 / shp);
            }
        } else {
            const auto* gev1 =
                dynamic_cast<const GeneralizedExtremeValue*>(&model.component(0));
            const auto* gev2 =
                dynamic_cast<const GeneralizedExtremeValue*>(&model.component(1));

            if (gev1 == nullptr || gev2 == nullptr) return neg_inf;

            double loc1 = gev1->xi();
            double scl1 = gev1->alpha();
            double shp1 = -gev1->kappa();

            double loc2 = gev2->xi();
            double scl2 = gev2->alpha();
            double shp2 = -gev2->kappa();

            if (scl1 <= 0.0 || scl2 <= 0.0) return neg_inf;

            // Loop over POT days and assign to season 1 or 2.
            for (std::size_t i = 0;
                 i < pot_days_.size() && i < df.exact_series().count(); ++i) {
                double x = df.exact_series()[i].value();
                int day = pot_days_[i];

                if (day < k1 || day >= k2) {
                    // Season 1
                    if (std::fabs(shp1) < 1E-4) {
                        log_lh += -std::log(scl1) - (x - loc1) / scl1;
                    } else {
                        double z = 1.0 + shp1 * ((x - loc1) / scl1);
                        if (z <= 0.0) return neg_inf;

                        log_lh += -std::log(scl1) - (1.0 + 1.0 / shp1) * std::log(z);
                    }
                } else {
                    // Season 2
                    if (std::fabs(shp2) < 1E-4) {
                        log_lh += -std::log(scl2) - (x - loc2) / scl2;
                    } else {
                        double z = 1.0 + shp2 * ((x - loc2) / scl2);
                        if (z <= 0.0) return neg_inf;

                        log_lh += -std::log(scl2) - (1.0 + 1.0 / shp2) * std::log(z);
                    }
                }
            }

            // Seasonal exceedance components.
            if (std::fabs(shp1) < 1E-4) {
                log_lh += -ny1 * std::exp(-(u - loc1) / scl1);
            } else {
                double zu1 = 1.0 + shp1 * ((u - loc1) / scl1);
                if (zu1 <= 0.0) return neg_inf;

                log_lh += -ny1 * std::pow(zu1, -1.0 / shp1);
            }

            if (std::fabs(shp2) < 1E-4) {
                log_lh += -ny2 * std::exp(-(u - loc2) / scl2);
            } else {
                double zu2 = 1.0 + shp2 * ((u - loc2) / scl2);
                if (zu2 <= 0.0) return neg_inf;

                log_lh += -ny2 * std::pow(zu2, -1.0 / shp2);
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
                // Normalize by retained ME mass from the 1E-8 probability window.
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

    // C# `PointwiseDataLogLikelihood` override (line 1042): per-observation contributions
    // for WAIC / LOO-CV. Exact data get the per-event GEV log-PDF plus a 1/N_exact share of
    // the global Poisson rate term; uncertain / interval / threshold data get the composite
    // likelihood only (block-indexed, no rate share), with the rate term attached to the
    // first non-exact entry when there is no exact data so the sum invariant
    // sum(pointwise) == DataLogLikelihood holds.
    std::vector<double> pointwise_data_log_likelihood(
        const std::vector<double>& parameters) const override {
        const double neg_inf = -std::numeric_limits<double>::infinity();

        if (distribution_ == nullptr || !has_data_frame()) return {};

        std::unique_ptr<DistributionBase> model_owner = distribution_->clone();
        CompetingRisks& model = static_cast<CompetingRisks&>(*model_owner);

        double u = threshold_;
        double n_y = total_years_;

        if (std::isnan(u) || std::isnan(n_y) || n_y <= 0) return {};

        std::vector<double> result;
        const DataFrame& df = data_frame();

        std::size_t n_exact = df.exact_series().count();
        std::size_t total_obs = n_exact + df.uncertain_series().count() +
                                df.interval_series().count() +
                                df.threshold_series().count();

        if (total_obs == 0) return {};

        double k1 = 0.0;
        double k2 = 0.0;
        double ny1 = 0.0;
        double ny2 = 0.0;

        // Set model parameters
        if (!is_seasonal_) {
            model.set_parameters(parameters);
        } else {
            if (parameters.size() < 8) return {};

            k1 = parameters[0];
            k2 = parameters[1];

            if (!(k1 >= 1 && k1 <= 366 && k2 >= 1 && k2 <= 366 && k1 < k2)) return {};

            ny1 = n_y * (k1 + (366.0 - k2)) / 366.0;
            ny2 = n_y * (k2 - k1) / 366.0;

            model.set_parameters(skip_first_two(parameters));
        }

        // Compute the global rate term to be distributed across observations
        double rate_term = 0.0;

        if (!is_seasonal_) {
            const auto* gev =
                dynamic_cast<const GeneralizedExtremeValue*>(&model.component(0));
            if (gev == nullptr) return {};

            double loc = gev->xi();
            double scl = gev->alpha();
            double shp = -gev->kappa();

            if (scl <= 0.0) return {};

            if (std::fabs(shp) < 1E-4) {
                rate_term = -n_y * std::exp(-(u - loc) / scl);
            } else {
                double zu = 1.0 + shp * ((u - loc) / scl);
                if (zu <= 0.0) return {};

                rate_term = -n_y * std::pow(zu, -1.0 / shp);
            }
        } else {
            const auto* gev1 =
                dynamic_cast<const GeneralizedExtremeValue*>(&model.component(0));
            const auto* gev2 =
                dynamic_cast<const GeneralizedExtremeValue*>(&model.component(1));

            if (gev1 == nullptr || gev2 == nullptr) return {};

            double loc1 = gev1->xi();
            double scl1 = gev1->alpha();
            double shp1 = -gev1->kappa();

            double loc2 = gev2->xi();
            double scl2 = gev2->alpha();
            double shp2 = -gev2->kappa();

            if (scl1 <= 0.0 || scl2 <= 0.0) return {};

            // Season 1 rate term
            if (std::fabs(shp1) < 1E-4) {
                rate_term += -ny1 * std::exp(-(u - loc1) / scl1);
            } else {
                double zu1 = 1.0 + shp1 * ((u - loc1) / scl1);
                if (zu1 <= 0.0) return {};

                rate_term += -ny1 * std::pow(zu1, -1.0 / shp1);
            }

            // Season 2 rate term
            if (std::fabs(shp2) < 1E-4) {
                rate_term += -ny2 * std::exp(-(u - loc2) / scl2);
            } else {
                double zu2 = 1.0 + shp2 * ((u - loc2) / scl2);
                if (zu2 <= 0.0) return {};

                rate_term += -ny2 * std::pow(zu2, -1.0 / shp2);
            }
        }

        // Distribute the global Poisson rate term across exact observations ONLY; if there
        // is no exact data, attach it to the first non-exact observation as a fallback so
        // the sum invariant is preserved (the C# comments, lines 1168-1177).
        double rate_per_exact = n_exact > 0 ? rate_term / static_cast<double>(n_exact) : 0.0;
        bool fallback_rate_applied = n_exact > 0;

        // Exact data contribution (with per-observation PDF + rate share)
        if (!is_seasonal_) {
            const auto& gev =
                static_cast<const GeneralizedExtremeValue&>(model.component(0));
            double loc = gev.xi();
            double scl = gev.alpha();
            double shp = -gev.kappa();

            for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
                double x = df.exact_series()[i].value();
                double ll;

                if (std::fabs(shp) < 1E-4) {
                    ll = -std::log(scl) - (x - loc) / scl;
                } else {
                    double z = 1.0 + shp * ((x - loc) / scl);
                    if (z <= 0.0) {
                        ll = neg_inf;
                    } else {
                        ll = -std::log(scl) - (1.0 + 1.0 / shp) * std::log(z);
                    }
                }
                result.push_back(ll + rate_per_exact);
            }
        } else {
            const auto& gev1 =
                static_cast<const GeneralizedExtremeValue&>(model.component(0));
            const auto& gev2 =
                static_cast<const GeneralizedExtremeValue&>(model.component(1));

            double loc1 = gev1.xi();
            double scl1 = gev1.alpha();
            double shp1 = -gev1.kappa();

            double loc2 = gev2.xi();
            double scl2 = gev2.alpha();
            double shp2 = -gev2.kappa();

            for (std::size_t i = 0;
                 i < pot_days_.size() && i < df.exact_series().count(); ++i) {
                double x = df.exact_series()[i].value();
                int day = pot_days_[i];
                double ll;

                if (day < k1 || day >= k2) {
                    // Season 1
                    if (std::fabs(shp1) < 1E-4) {
                        ll = -std::log(scl1) - (x - loc1) / scl1;
                    } else {
                        double z = 1.0 + shp1 * ((x - loc1) / scl1);
                        ll = z <= 0.0 ? neg_inf
                                      : -std::log(scl1) - (1.0 + 1.0 / shp1) * std::log(z);
                    }
                } else {
                    // Season 2
                    if (std::fabs(shp2) < 1E-4) {
                        ll = -std::log(scl2) - (x - loc2) / scl2;
                    } else {
                        double z = 1.0 + shp2 * ((x - loc2) / scl2);
                        ll = z <= 0.0 ? neg_inf
                                      : -std::log(scl2) - (1.0 + 1.0 / shp2) * std::log(z);
                    }
                }
                result.push_back(ll + rate_per_exact);
            }
        }

        // Uncertain Data -- composite GEV likelihood only (no rate-term share).
        for (std::size_t i = 0; i < df.uncertain_series().count(); ++i) {
            const DistributionBase& dist = df.uncertain_series()[i].distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            double ll;
            if (numerics::is_finite(a) && numerics::is_finite(b) &&
                numerics::is_finite(mass) && mass > 0.0 && a < b) {
                double ep = numerics::math::integration::Integration::gauss_legendre20(
                                [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                            mass;
                ll = ep > 0 ? std::log(ep) : neg_inf;
            } else {
                ll = neg_inf;
            }
            // Degenerate-case fallback: if there are no exact observations, attach the
            // global rate term to the first non-exact entry so the sum invariant is
            // preserved.
            if (!fallback_rate_applied) {
                ll += rate_term;
                fallback_rate_applied = true;
            }
            result.push_back(ll);
        }

        // Interval Data -- composite GEV likelihood only.
        for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
            const IntervalData& data = df.interval_series()[i];
            double ll = model.log_likelihood_intervals(data.lower_value(), data.upper_value());
            if (!fallback_rate_applied) {
                ll += rate_term;
                fallback_rate_applied = true;
            }
            result.push_back(ll);
        }

        // Threshold Data -- composite GEV likelihood only.
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            double ll = 0.0;
            if (data.number_below() > 0)
                ll += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                ll += model.log_likelihood_right_censored(data.value(), data.number_above());
            if (!fallback_rate_applied) {
                ll += rate_term;
                fallback_rate_applied = true;
            }
            result.push_back(ll);
        }

        return result;
    }

    // C# `PointwiseDataLogLikelihoodComponents` override (line 1326): wraps the raw
    // pointwise log-likelihoods in DataComponents (uncertain reports the ME mean, interval
    // the midpoint, thresholds type LeftCensored with the combined count and a
    // "Threshold N" label).
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& parameters) const override {
        if (distribution_ == nullptr || !has_data_frame()) return {};

        // Get the raw log-likelihoods
        std::vector<double> log_liks = pointwise_data_log_likelihood(parameters);
        std::vector<DataComponent> result;
        result.reserve(log_liks.size());

        std::size_t idx = 0;
        const DataFrame& df = data_frame();

        // Exact Data
        for (std::size_t i = 0; i < df.exact_series().count() && idx < log_liks.size();
             ++i) {
            const ExactData& data = df.exact_series()[i];
            result.emplace_back(static_cast<int>(idx), log_liks[idx], data.value(),
                                DataComponentType::Exact, 1, std::to_string(data.index()));
            ++idx;
        }

        // Uncertain Data
        for (std::size_t i = 0; i < df.uncertain_series().count() && idx < log_liks.size();
             ++i) {
            const UncertainData& data = df.uncertain_series()[i];
            result.emplace_back(static_cast<int>(idx), log_liks[idx],
                                data.distribution().mean(), DataComponentType::Uncertain, 1,
                                std::to_string(data.index()));
            ++idx;
        }

        // Interval Data
        for (std::size_t i = 0; i < df.interval_series().count() && idx < log_liks.size();
             ++i) {
            const IntervalData& data = df.interval_series()[i];
            double midpoint = (data.lower_value() + data.upper_value()) / 2.0;
            result.emplace_back(static_cast<int>(idx), log_liks[idx], midpoint,
                                DataComponentType::Interval, 1, std::to_string(data.index()));
            ++idx;
        }

        // Threshold Data
        for (std::size_t i = 0; i < df.threshold_series().count() && idx < log_liks.size();
             ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            int total_count = data.number_below() + data.number_above();
            result.emplace_back(static_cast<int>(idx), log_liks[idx], data.value(),
                                DataComponentType::LeftCensored, total_count,
                                "Threshold " + std::to_string(i + 1));
            ++idx;
        }

        return result;
    }

    // C# `PriorLogLikelihood` override (line 1375): parameter priors (including the
    // change-point parameters when seasonal), the Jeffreys 1/scale term per GEV component
    // (the GEV scale is parameter 1), and the quantile priors (single, or the single-GEV
    // three-prior difference form with its quantile Jacobian). Like the C#, the raw sum is
    // returned (no finite collapse; LogLikelihood collapses). The C# `Parameters is null`
    // guard has no C++ analogue (the vector always exists).
    double prior_log_likelihood(std::vector<double>& parameters) const override {
        const double neg_inf = -std::numeric_limits<double>::infinity();

        if (distribution_ == nullptr) return neg_inf;

        std::unique_ptr<DistributionBase> model_owner = distribution_->clone();
        CompetingRisks& model = static_cast<CompetingRisks&>(*model_owner);
        int k = model.component_count();
        double log_lh = 0.0;

        // Set model parameters
        if (!is_seasonal_) {
            model.set_parameters(parameters);
        } else {
            model.set_parameters(skip_first_two(parameters));
        }

        // Parameter priors (including change-point parameters if seasonal).
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            log_lh += parameters_[i].prior_distribution().log_pdf(parameters[i]);
        }

        // Jeffreys rule on scale for each GEV component.
        if (use_jeffreys_rule_for_scale()) {
            for (int j = 0; j < k; ++j) {
                double scale = model.component(j).get_parameters()[1];
                log_lh -= scale > 0 ? std::log(scale)
                                    : std::numeric_limits<double>::infinity();
            }
        }

        // Quantile Priors
        if (enable_quantile_priors() && use_single_quantile() &&
            quantile_priors_true_.size() == 1) {
            log_lh += quantile_priors_true_[0].distribution().log_pdf(
                model.inverse_cdf(1.0 - quantile_priors_true_[0].alpha()));
        } else if (enable_quantile_priors() && !use_single_quantile() &&
                   distribution_ != nullptr && distribution_->component_count() == 1 &&
                   quantile_priors_true_.size() == 3) {
            // First quantile prior
            log_lh += quantile_priors_true_[0].distribution().log_pdf(
                model.inverse_cdf(1.0 - quantile_priors_true_[0].alpha()));

            // Differences
            std::vector<double> p_vals(quantile_priors_true_.size());
            p_vals[0] = 1.0 - quantile_priors_true_[0].alpha();

            for (std::size_t i = 1; i < quantile_priors_true_.size(); ++i) {
                p_vals[i] = 1.0 - quantile_priors_true_[i].alpha();

                double q_curr = model.inverse_cdf(1.0 - quantile_priors_true_[i].alpha());
                double q_prev = model.inverse_cdf(1.0 - quantile_priors_true_[i - 1].alpha());
                double diff = q_curr - q_prev;

                log_lh += quantile_priors_true_[i].distribution().log_pdf(diff);
            }

            // Jacobian determinant for transformation from parameters to quantiles.
            double d = quantile_jacobian_determinant(model, p_vals);
            log_lh += d != 0 ? std::log(std::fabs(d)) : neg_inf;
        }

        return log_lh;
    }

    // C# `PointwisePriorLogLikelihood` override (line 1446): per-parameter prior
    // components, per-component Jeffreys components ("Jeffreys Scale: Scale", or
    // "Jeffreys Scale: D<j>.Scale" when seasonal -- the C# hardcodes "Scale" here), and the
    // quantile-prior components (with "(diff)" and "Quantile Jacobian" entries in the
    // three-prior form).
    std::vector<PriorComponent> pointwise_prior_log_likelihood(
        const std::vector<double>& parameters) const override {
        const double neg_inf = -std::numeric_limits<double>::infinity();
        std::vector<PriorComponent> result;

        if (distribution_ == nullptr) return result;

        std::unique_ptr<DistributionBase> model_owner = distribution_->clone();
        CompetingRisks& model = static_cast<CompetingRisks&>(*model_owner);
        int k = model.component_count();

        // Set model parameters
        if (!is_seasonal_) {
            model.set_parameters(parameters);
        } else {
            model.set_parameters(skip_first_two(parameters));
        }

        // Parameter priors (including change-point parameters if seasonal)
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            double ll = parameters_[i].prior_distribution().log_pdf(parameters[i]);
            const std::string& param_name = parameters_[i].owner_name().empty()
                                                ? parameters_[i].name()
                                                : parameters_[i].owner_name();
            result.emplace_back("Parameter Prior: " + param_name, ll,
                                PriorComponentType::ParameterPrior);
        }

        // Jeffreys rule on scale for each GEV component
        if (use_jeffreys_rule_for_scale()) {
            for (int j = 0; j < k; ++j) {
                double scale = model.component(j).get_parameters()[1];
                double ll = scale > 0 ? -std::log(scale) : neg_inf;
                std::string scale_name =
                    is_seasonal_ ? "D" + std::to_string(j + 1) + ".Scale" : "Scale";
                result.emplace_back("Jeffreys Scale: " + scale_name, ll,
                                    PriorComponentType::JeffreysScalePrior);
            }
        }

        // Quantile Priors (alpha formatted like the C# :G4, approximated with %.4g).
        if (enable_quantile_priors() && use_single_quantile() &&
            quantile_priors_true_.size() == 1) {
            double quantile = model.inverse_cdf(1.0 - quantile_priors_true_[0].alpha());
            double ll = quantile_priors_true_[0].distribution().log_pdf(quantile);
            result.emplace_back(
                "Quantile Prior: p=" + format_g4(quantile_priors_true_[0].alpha()), ll,
                PriorComponentType::QuantilePrior);
        } else if (enable_quantile_priors() && !use_single_quantile() &&
                   distribution_ != nullptr && distribution_->component_count() == 1 &&
                   quantile_priors_true_.size() == 3) {
            // First quantile prior
            double q0 = model.inverse_cdf(1.0 - quantile_priors_true_[0].alpha());
            double ll0 = quantile_priors_true_[0].distribution().log_pdf(q0);
            result.emplace_back(
                "Quantile Prior: p=" + format_g4(quantile_priors_true_[0].alpha()), ll0,
                PriorComponentType::QuantilePrior);

            // Differences
            std::vector<double> p_vals(quantile_priors_true_.size());
            p_vals[0] = 1.0 - quantile_priors_true_[0].alpha();

            for (std::size_t i = 1; i < quantile_priors_true_.size(); ++i) {
                p_vals[i] = 1.0 - quantile_priors_true_[i].alpha();

                double q_curr = model.inverse_cdf(1.0 - quantile_priors_true_[i].alpha());
                double q_prev = model.inverse_cdf(1.0 - quantile_priors_true_[i - 1].alpha());
                double diff = q_curr - q_prev;

                double ll = quantile_priors_true_[i].distribution().log_pdf(diff);
                result.emplace_back(
                    "Quantile Prior: p=" + format_g4(quantile_priors_true_[i].alpha()) +
                        " (diff)",
                    ll, PriorComponentType::QuantilePrior);
            }

            // Jacobian determinant for transformation from parameters to quantiles
            double d = quantile_jacobian_determinant(model, p_vals);
            double jacobian_ll = d != 0 ? std::log(std::fabs(d)) : neg_inf;
            result.emplace_back("Quantile Jacobian", jacobian_ll,
                                PriorComponentType::Jacobian);
        }

        return result;
    }

    // C# `SetParameterValues` override (line 1529): null/count guards with the C# exception
    // semantics (the null guard has no C++ analogue -- a vector cannot be null), write the
    // ModelParameter values, then push the (seasonally transformed, when applicable)
    // parameters into the wrapped distribution. The C# dereferences `Distribution!`
    // unguarded (NullReferenceException on a null distribution); the C++ throws
    // std::runtime_error instead of crashing.
    void set_parameter_values(const std::vector<double>& parameters) override {
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters()))
            throw std::invalid_argument("The length of the parameter list is incorrect.");

        for (std::size_t i = 0; i < parameters.size(); ++i)
            parameters_[i].set_value(parameters[i]);

        if (distribution_ == nullptr)
            throw std::runtime_error("Distribution is null.");  // C# NRE guard

        // Set parameters
        if (!is_seasonal_) {
            distribution_->set_parameters(parameters);
        } else {
            // Seasonal case: convert GEV parameters so that each season has the correct
            // marginal exceedance behavior.
            distribution_->set_parameters(seasonal_transformed_parameters(parameters));
        }
    }

    // C# `GetDistribution(IList<double>)` (line 1623): a clone of the competing risks
    // distribution with the specified parameter values applied (accounting for seasonal
    // transformations). The C# null-parameters guard has no C++ analogue.
    std::unique_ptr<CompetingRisks> get_distribution(
        const std::vector<double>& parameters) const {
        if (distribution_ == nullptr)
            throw std::runtime_error("Distribution is null.");  // C# `Distribution!` NRE

        std::unique_ptr<DistributionBase> base = distribution_->clone();
        auto model =
            std::unique_ptr<CompetingRisks>(static_cast<CompetingRisks*>(base.release()));

        // Set parameters
        if (!is_seasonal_) {
            model->set_parameters(parameters);
        } else {
            model->set_parameters(seasonal_transformed_parameters(parameters));
        }
        return model;
    }

    // C# `Clone()` override (line 1705): deep, independent copy through the (DataFrame,
    // CompetingRisks) ctor, then the C# object-initializer field writes (no setter side
    // effects -- Lambda keeps the ctor-derived value and _totalYearsExplicit stays false,
    // exactly like the C#), then SetAMSData + ProcessQuantilePriors on the result. See the
    // file header for the DataFrame deep-copy divergence.
    PointProcessModel clone() const {
        if (!has_data_frame())
            throw std::invalid_argument("dataFrame");  // C++ divergence: frame required
        if (distribution_ == nullptr)
            throw std::invalid_argument(
                "distribution");  // the C# ctor would NRE on a null distribution

        std::vector<ModelParameter> parms;
        parms.reserve(parameters_.size());
        for (std::size_t i = 0; i < parameters_.size(); ++i)
            parms.push_back(parameters_[i].clone());

        std::vector<QuantilePrior> quants;
        quants.reserve(quantile_priors_.size());
        for (std::size_t i = 0; i < quantile_priors_.size(); ++i)
            quants.push_back(quantile_priors_[i].clone());

        PointProcessModel result(data_frame().clone(), *distribution_);
        // C# object-initializer field writes (no setter side effects).
        result.threshold_ = threshold_;
        result.total_years_ = total_years_;
        result.use_defaults_ = use_defaults_;
        result.is_seasonal_ = is_seasonal_;
        result.time_block_ = time_block_;
        result.start_month_ = start_month_;
        result.use_default_flat_priors_ = use_default_flat_priors_;
        result.use_jeffreys_rule_for_scale_ = use_jeffreys_rule_for_scale_;
        result.enable_quantile_priors_ = enable_quantile_priors_;
        result.use_single_quantile_ = use_single_quantile_;
        result.parameters_ = std::move(parms);
        result.quantile_priors_ = std::move(quants);

        result.set_ams_data();
        result.process_quantile_priors();
        return result;
    }

    // SKIPPED: ToXElement (C# line 1736) -- XML serialization is a project-wide deferral.

    // C# `Validate` override (line 1773).
    ValidationResult validate() const override {
        ValidationResult result;

        // Data frame checks
        if (!has_data_frame()) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: Data frame is null.");
            return result;
        }

        // Validate data frame
        ValidationResult data_valid = data_frame().validate();
        if (!data_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              data_valid.validation_messages.begin(),
                                              data_valid.validation_messages.end());
        }

        // Distribution checks
        if (distribution_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Competing risks distribution is null.");
        } else if (distribution_->component_count() == 0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Competing risks distribution has no component distributions.");
        }

        // Validate uncertain-data ME bounds before likelihood evaluation. The point-process
        // magnitude likelihood normalizes uncertain integrals by retained 1E-8 probability
        // mass.
        for (std::size_t i = 0; i < data_frame().uncertain_series().count(); ++i) {
            const UncertainData& data = data_frame().uncertain_series()[i];
            const DistributionBase& dist = data.distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            // Validate the retained quantile window directly; endpoints at 0/1 are infinite
            // for common ME distributions such as Normal.
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

        // Component distribution type check (must all be GEV). (The C# message quotes the
        // enum name; the ported enum has no name surface, so the message carries the index
        // only -- M10/M11 precedent.)
        if (distribution_ != nullptr) {
            for (int i = 0; i < distribution_->component_count(); ++i) {
                if (!is_supported_distribution_type(distribution_->component(i).type())) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Component distribution " + std::to_string(i + 1) +
                        " has an unsupported type but the point process model requires "
                        "Generalized Extreme Value (GEV) distributions.");
                }
            }
        }

        // Threshold / years / lambda
        if (std::isnan(threshold_) || !numerics::is_finite(threshold_)) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Threshold is not set or is not finite.");
        }

        if (std::isnan(total_years_) || !numerics::is_finite(total_years_) ||
            total_years_ <= 0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: TotalYears must be positive and finite.");
        }

        if (std::isnan(lambda_) || !numerics::is_finite(lambda_) || lambda_ <= 0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Lambda (average events per year) must be positive and finite.");
        }

        // Seasonal-specific checks
        if (is_seasonal_) {
            if (start_month_ < 1 || start_month_ > 12) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: StartMonth must be between 1 and 12 for seasonal models.");
            }

            if (ams_data_frame_.exact_series().count() == 0) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: AMSDataFrame has no exact data for seasonal model.");
            }

            // Change points (if present)
            if (parameters_.size() >= 2 &&
                starts_with_ci(parameters_[0].name(), "Change Point") &&
                starts_with_ci(parameters_[1].name(), "Change Point")) {
                double k1 = parameters_[0].value();
                double k2 = parameters_[1].value();

                if (!(k1 >= 1 && k1 <= 366 && k2 >= 1 && k2 <= 366 && k1 < k2)) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Seasonal change points K1 and K2 must satisfy "
                        "1 <= K1 < K2 <= 366.");
                }
            }
        }

        // Parameter priors
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            ValidationResult valid = parameters_[i].validate();
            if (!valid.is_valid) {
                result.is_valid = false;
                result.validation_messages.insert(result.validation_messages.end(),
                                                  valid.validation_messages.begin(),
                                                  valid.validation_messages.end());
            }
        }

        // Quantile priors
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

    // C# `GenerateRandomValues(int sampleSize, int seed = -1)` (line 1950, ISimulatable):
    // guards, then per sample one inverse-CDF draw from EVERY component of the wrapped
    // distribution (each consuming one uniform from the shared Mersenne Twister stream),
    // combined by the distribution's min/max rule. The C# duplicates this loop rather than
    // delegating to CompetingRisks.GenerateRandomValues; the port mirrors it, including the
    // C# combining seeds (double.MaxValue / double.NegativeInfinity -- the Numerics
    // override seeds the max combine with double.MinValue instead).
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size <= 0) throw std::out_of_range("Sample size must be positive.");
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution cannot be null when generating random values.");

        numerics::sampling::MersenneTwister rng =
            seed > 0 ? numerics::sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                     : numerics::sampling::MersenneTwister();
        std::vector<double> result(static_cast<std::size_t>(sample_size));
        bool use_minimum = distribution_->minimum_of_random_variables;

        for (int i = 0; i < sample_size; ++i) {
            // Generate values from all components.
            double combined_value = use_minimum ? std::numeric_limits<double>::max()
                                                : -std::numeric_limits<double>::infinity();

            for (int j = 0; j < distribution_->component_count(); ++j) {
                double value = distribution_->component(j).inverse_cdf(rng.next_double());
                if (use_minimum)
                    combined_value = std::min(combined_value, value);
                else
                    combined_value = std::max(combined_value, value);
            }

            result[static_cast<std::size_t>(i)] = combined_value;
        }

        return result;
    }

    // C# `GeneratePOTTimeSeries(DateTime startDate, double durationYears, int seed)` (line 2017),
    // un-gated by the P6 TimeSeries container: a synthetic peaks-over-threshold record. The event
    // COUNT is Poisson with mean lambda * durationYears; each event's DATE is uniform over the
    // span (365.25 days per year, averaging over leap years) and its MAGNITUDE is drawn from the
    // marginal conditional on exceeding the threshold, by inverting
    // F_cond(y) = (F(y) - F(u)) / (1 - F(u)).
    //
    // The PRNG draw order is oracle-visible and is transcribed as written: the Poisson count
    // first, then per event one `next_double()` for the date offset and one for the magnitude.
    // For a SEASONAL model the first two parameters are the season-change days k1 and k2, and the
    // event's day of year selects which of the two GEV marginals it is drawn from -- the same
    // classification DataLogLikelihood applies.
    numerics::data::TimeSeries generate_pot_time_series(
        const numerics::data::DateTime& start_date, double duration_years, int seed = -1) const {
        if (duration_years <= 0)
            throw std::out_of_range("Duration must be positive.");
        if (distribution_ == nullptr)
            throw std::runtime_error("Distribution cannot be null when generating POT samples.");
        if (!has_data_frame())
            throw std::runtime_error("DataFrame cannot be null when generating POT samples.");

        double u = threshold_;
        double ny = total_years_;
        if (std::isnan(u) || std::isnan(ny) || ny <= 0)
            throw std::runtime_error("Threshold and TotalYears must be set on the model.");

        double lambda = static_cast<double>(data_frame().exact_series().count()) / ny;
        double expected_events = lambda * duration_years;

        numerics::sampling::MersenneTwister rng =
            seed > 0 ? numerics::sampling::MersenneTwister(seed)
                     : numerics::sampling::MersenneTwister();

        int total_events = sample_poisson(expected_events, rng);

        numerics::data::TimeSeries result(numerics::data::TimeInterval::OneDay);
        if (total_events <= 0) return result;

        std::vector<double> parameters;
        parameters.reserve(parameters_.size());
        for (const auto& p : parameters_) parameters.push_back(p.value());

        std::unique_ptr<CompetingRisks> local_model(
            static_cast<CompetingRisks*>(distribution_->clone().release()));
        double k1 = 0.0, k2 = 0.0;
        if (is_seasonal_ && parameters.size() >= 2) {
            k1 = parameters[0];
            k2 = parameters[1];
            local_model->set_parameters(
                std::vector<double>(parameters.begin() + 2, parameters.end()));
        } else {
            local_model->set_parameters(parameters);
        }

        // 365.25 days per year averages over leap years, as the C# comment says.
        double total_days = duration_years * 365.25;

        std::vector<std::pair<numerics::data::DateTime, double>> events;
        events.reserve(static_cast<std::size_t>(total_events));
        for (int i = 0; i < total_events; ++i) {
            double offset_days = rng.next_double() * total_days;
            numerics::data::DateTime event_date = start_date.add_days(offset_days);

            const DistributionBase* mark = nullptr;
            if (is_seasonal_ && local_model->component_count() >= 2) {
                int day = event_date.day_of_year();
                mark = (day < k1 || day >= k2) ? &local_model->component(0)
                                               : &local_model->component(1);
            } else {
                mark = &local_model->component(0);
            }

            double fu = mark->cdf(u);
            if (fu >= 1.0 - 1e-12) fu = 1.0 - 1e-12;
            double p_unif = fu + (1.0 - fu) * rng.next_double();
            if (p_unif >= 1.0) p_unif = 1.0 - 1e-12;
            events.emplace_back(event_date, mark->inverse_cdf(p_unif));
        }

        // C# `events.Sort((a, b) => a.date.CompareTo(b.date))` -- List<T>.Sort is the unstable
        // .NET introsort, so tied dates keep ITS permutation, not a stable one.
        numerics::utilities::dotnet_list_sort(
            events, [](const std::pair<numerics::data::DateTime, double>& a,
                       const std::pair<numerics::data::DateTime, double>& b) {
                return a.first.compare_to(b.first);
            });
        for (const auto& e : events)
            result.add(numerics::data::TimeSeries::Ordinate(e.first, e.second));

        return result;
    }

   protected:
    // C# `DataFrame_PropertyChanged` override (line 390): the explicit-invalidation
    // equivalent for in-place frame mutations (see the base header's cadence note; the
    // PlottingParameter/PlottingPosition filtering has no C++ trigger to filter). NEVER
    // called from the likelihood paths.
    void data_frame_property_changed() override {
        if (!has_data_frame()) return;

        data_frame().process_threshold_series();

        set_ams_data();

        if (use_defaults_) set_default_threshold_and_total_years(std::nullopt, true);

        if (use_default_flat_priors()) set_default_parameters();
    }

   private:
    // C# `SamplePoisson(double mean, MersenneTwister rng)` (line 2112): an integer event
    // count from Poisson(mean) -- normal approximation for mean > 30, Knuth's algorithm
    // otherwise. Called by generate_pot_time_series above (P6 un-gated it; the helper itself
    // was ported early so the model's RNG surface was complete for the M14 seeded-stream work).
    // C# (int)Math.Round(...) is banker's rounding -> std::nearbyint (round-half-even under
    // the default FE mode; the codebase's Math.Round precedent).
    static int sample_poisson(double mean, numerics::sampling::MersenneTwister& rng) {
        if (mean <= 0.0) return 0;

        if (mean > 30.0) {
            // Normal approximation: N(mean, mean) is accurate for mean > 30.
            double z = numerics::distributions::Normal::standard_z(rng.next_double());
            int n = static_cast<int>(std::nearbyint(mean + std::sqrt(mean) * z));
            return std::max(0, n);
        }

        // Knuth's small-mean algorithm.
        double l = std::exp(-mean);
        int k = 0;
        double p = 1.0;
        while (true) {
            ++k;
            p *= rng.next_double();
            if (p <= l) return k - 1;
            // Safety guard for pathological RNG sequences.
            if (k > 1000000) return k - 1;
        }
    }

    // C# `Distribution = (CompetingRisks)distribution.Clone()`.
    static std::unique_ptr<CompetingRisks> clone_competing_risks(
        const CompetingRisks& distribution) {
        std::unique_ptr<DistributionBase> base = distribution.clone();
        return std::unique_ptr<CompetingRisks>(static_cast<CompetingRisks*>(base.release()));
    }

    // C# `ExtensionMethods.AlmostEquals(this double a, double b, double epsilon = 1E-15)`:
    // |a - b| <= epsilon; false whenever either side is NaN (so a NaN-valued property IS
    // overwritten by the guarded setters, exactly like the C#).
    static bool almost_equals(double a, double b, double epsilon = 1E-15) {
        return std::fabs(a - b) <= epsilon;
    }

    // C# `Math.BitDecrement(x)`: the next representable double toward -infinity.
    static double bit_decrement(double value) {
        return std::nextafter(value, -std::numeric_limits<double>::infinity());
    }

    // C# `parameters.Skip(2).ToArray()` (never throws; an under-length input yields a
    // shorter -- possibly empty -- array).
    static std::vector<double> skip_first_two(const std::vector<double>& parameters) {
        if (parameters.size() <= 2) return {};
        return std::vector<double>(parameters.begin() + 2, parameters.end());
    }

    // The C# seasonal marginal-correction transform shared verbatim by SetParameterValues
    // (lines 1546-1612) and GetDistribution (lines 1636-1699): convert each season's GEV
    // parameters so it has the correct marginal exceedance behavior, using the Gumbel limit
    // when kappa is near zero and keeping the raw parameters when p^-kappa overflows.
    static std::vector<double> seasonal_transformed_parameters(
        const std::vector<double>& parameters) {
        double k1 = parameters[0];
        double k2 = parameters[1];

        double xi1 = parameters[2];
        double alpha1 = parameters[3];
        double kappa1 = parameters[4];

        double xi2 = parameters[5];
        double alpha2 = parameters[6];
        double kappa2 = parameters[7];

        // Correction factors
        double p1 = (k1 + (366.0 - k2)) / 366.0;
        double p2 = (k2 - k1) / 366.0;

        // Season 1 - use Gumbel limit when kappa is near zero
        double xi_hat1, alpha_hat1;
        if (std::fabs(kappa1) < 1e-8) {
            // Gumbel limit: xi_hat = xi - alpha * log(p), alpha_hat = alpha
            xi_hat1 = xi1 - alpha1 * std::log(p1);
            alpha_hat1 = alpha1;
        } else {
            double pow_term1 = std::pow(p1, -kappa1);
            // Guard against overflow
            if (std::isinf(pow_term1) || std::isnan(pow_term1)) {
                xi_hat1 = xi1;
                alpha_hat1 = alpha1;
            } else {
                xi_hat1 = xi1 + (alpha1 / kappa1) * (1.0 - pow_term1);
                alpha_hat1 = alpha1 * pow_term1;
            }
        }

        // Season 2 - use Gumbel limit when kappa is near zero
        double xi_hat2, alpha_hat2;
        if (std::fabs(kappa2) < 1e-8) {
            xi_hat2 = xi2 - alpha2 * std::log(p2);
            alpha_hat2 = alpha2;
        } else {
            double pow_term2 = std::pow(p2, -kappa2);
            if (std::isinf(pow_term2) || std::isnan(pow_term2)) {
                xi_hat2 = xi2;
                alpha_hat2 = alpha2;
            } else {
                xi_hat2 = xi2 + (alpha2 / kappa2) * (1.0 - pow_term2);
                alpha_hat2 = alpha2 * pow_term2;
            }
        }

        return {xi_hat1, alpha_hat1, kappa1, xi_hat2, alpha_hat2, kappa2};
    }

    // C# `((IStandardError)model.Distributions[0]).QuantileJacobian(pVals, out var D)`: the
    // ported IStandardError surface lives on the concrete GEV
    // (generalized_extreme_value.hpp, added in M12); a non-GEV component maps the C#
    // InvalidCastException to std::runtime_error.
    static double quantile_jacobian_determinant(const CompetingRisks& model,
                                                const std::vector<double>& p_vals) {
        const auto* gev =
            dynamic_cast<const GeneralizedExtremeValue*>(&model.component(0));
        if (gev == nullptr)
            throw std::runtime_error(
                "Component distribution does not support the quantile Jacobian.");
        double determinant = 0.0;
        gev->quantile_jacobian(p_vals, determinant);
        return determinant;
    }

    // C# `Math.Round(value, 2)` (MidpointRounding.ToEven); the
    // univariate_distribution_model.hpp precedent.
    static double round_half_even_2(double value) {
        return std::nearbyint(value * 100.0) / 100.0;
    }

    // Case-insensitive ASCII prefix test (C# String.StartsWith(...,
    // StringComparison.OrdinalIgnoreCase)).
    static bool starts_with_ci(const std::string& text, const std::string& prefix) {
        if (text.size() < prefix.size()) return false;
        for (std::size_t i = 0; i < prefix.size(); ++i) {
            char a = text[i], b = prefix[i];
            if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
            if (a != b) return false;
        }
        return true;
    }

    // C# alpha label format ":G4" (general, four significant digits), approximated with
    // printf %.4g (display-only; M11 precedent).
    static std::string format_g4(double value) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.4g", value);
        return std::string(buf);
    }

    std::unique_ptr<CompetingRisks> distribution_;  // C# _distribution (line 150)
    DataFrame ams_data_frame_;                      // C# _amsDataFrame (line 151)
    std::vector<int> pot_days_;                     // C# _potDays (line 152)
    bool is_seasonal_ = false;                      // C# _isSeasonal (line 153)
    TimeBlockWindow time_block_ = TimeBlockWindow::WaterYear;  // C# _timeBlock (line 154)
    int start_month_ = 10;                          // C# _startMonth (line 155)
    double threshold_ = std::numeric_limits<double>::quiet_NaN();    // C# _threshold
    double total_years_ = std::numeric_limits<double>::quiet_NaN();  // C# _totalYears
    bool use_defaults_ = true;                      // C# _useDefaults (line 158)
    double lambda_ = std::numeric_limits<double>::quiet_NaN();  // C# _lambda (line 159)
    // Tracks whether the user has explicitly set TotalYears; reset by the DataFrame setter
    // so the heuristic for new data is re-applied (C# _totalYearsExplicit, line 168).
    bool total_years_explicit_ = false;
};

}  // namespace corehydro::models
