// ported from: RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/UnivariateDistribution.cs
// @ c2e6192 -- stationary path (Phase 5, M8) + nonstationary trends, quantile priors,
// Clone, and seeded simulation (M9). The Phase 4 T6 slice was exact-data-only.
//
// The univariate distribution model: a parent distribution plus a DataFrame of exact /
// uncertain / interval / threshold observations, with the FULL censored likelihood --
// stationary (M8) and nonstationary via per-parameter trend models (M9). The C++ class
// keeps the Phase 4 name `UnivariateDistributionModel` (the C# class is
// `UnivariateDistribution`; the suffix avoids colliding with the Numerics distribution
// layer). It derives from `UnivariateDistributionModelBase` (base/...) which supplies the
// DataFrame, UseJeffreysRuleForScale, and IQuantilePriors state, and (M9) from
// `ISimulatable<std::vector<double>>` (C# ISimulatable<double[]>).
//
// Ported surface (stationary):
//   - Ctors: parameterless (C# line 32: LogPearsonTypeIII, no data), (DataFrame,
//     distribution) (line 46), (DataFrame, type) (line 63), plus two Phase 4 compatibility
//     shims taking a plain std::vector<double> of exact data (they build a DataFrame with an
//     ExactSeries internally; the fixture runner and the R/Python glue construct through
//     them and their semantics are unchanged).
//   - `IsSupportedDistributionType` / `CreateDistribution` statics (lines 263, 534): the
//     15-family whitelist. create_distribution delegates to the Numerics factory after the
//     whitelist check (same construction result as the C# if-chain of `new X()` defaults;
//     avoids 15 concrete includes here). Unsupported -> std::out_of_range (C#
//     ArgumentOutOfRangeException).
//   - `Distribution` property (line 328): set_distribution runs SetDefaultParameters +
//     SetDefaultQuantilePriors like the C# setter (its TrendModels reset is M9; the
//     _isDeserializing XML suppression is out of scope). `DistributionType` (line 368).
//   - `SetDefaultParameters` (line 571), stationary: when a valid DataFrame with exact data
//     is present, `GetParameterConstraints` supplies (initials, lowers, uppers) and one
//     ModelParameter per distribution parameter gets Value/bounds/IsPositive (lowers[i] ==
//     Tools.DoubleMachineEpsilon, C# line 628)/Uniform prior; otherwise
//     (the C# null/empty-frame skip path) one default ModelParameter per distribution
//     parameter (upstream these come from the default ConstantTrend per parameter; the
//     trend layer joins this class in M9 -- observably identical for the stationary case,
//     with Name cleared to "" as the C# does).
//     RETAINED PHASE 4 DEVIATION: after the constraint path the distribution itself is set
//     to `initials`; the C# leaves the distribution at its previous parameters until
//     SetParameterValues. Kept because the Phase 4 tests/fixtures pin it and no ported
//     compute path reads the stored parameters before setting them (the likelihoods all set
//     a working copy from the proposal first).
//   - `LogLikelihood` (1116), `PriorLogLikelihood` (1143), `StationaryData_LogLikelihood`
//     (1181), `DataLogLikelihood` (1361), `PointwiseDataLogLikelihood` (1373) ->
//     `StationaryPointwiseLogLikelihood` (1632), `PointwiseDataLogLikelihoodComponents`
//     (1385) -> `StationaryPointwiseLogLikelihoodComponents` (1402), `Prior_LogLikelihood`
//     (1822), `PointwisePriorLogLikelihood` (1882), `SetParameterValues` (1977),
//     `GetParameterValues` (2006), `SetDistributionParameterValues` (2020), `Validate`
//     (2127) -- each stationary branch term-for-term; see the per-method comments.
//   - The uncertain-data (measurement-error) branch integrates
//     `dist.PDF(q) * model.PDF(q)` over the 1e-8 probability window with
//     `Integration::gauss_legendre20` and normalizes by the retained mass, with the
//     finiteness/positivity guards exactly as the C# writes them.
//
// Validity-check mechanism: the C# `model.ValidateParameters(parameters, false)` returns
// null when valid / a message when invalid, and the model is only `SetParameters`ed when
// valid. The C++ distribution base has no validate-without-set; every concrete
// `set_parameters` sets the `parameters_valid_` flag instead. This port therefore probes on
// a THROWAWAY clone (set, then read `parameters_valid()`) and only sets the working copy
// when the probe is valid -- functionally identical, and it preserves the C# semantic that
// the working copy keeps its previous parameters on an invalid proposal (which matters for
// the Jeffreys term below, read off the working copy's parameters as the C# does).
//
// Divergence note (retained from Phase 4): `Normal::set_parameters` clamps a tiny
// non-negative scale up to 1E-16 rather than marking it invalid (mirroring the C#
// `Normal.SetParameters` clamp), so a proposed sigma in [0, 1E-16) rounds up identically on
// both sides of the port.
//
// M9 additions (each with per-method comments; the big trend-driven bodies are defined
// out-of-line in univariate_distribution_model_trends.hpp, included at the bottom of this
// file purely for file size -- the data_frame_plotting.hpp precedent):
//   - Nonstationary state: IsNonstationary (398; the set-false branch resets the trends to
//     one ConstantTrend per parameter and calls SetDefaultParameters; the set-true branch
//     builds the full time series, re-centers ParameterTimeIndex off the exact series, and
//     re-anchors every trend's StartIndex), ParameterTimeIndex (451; the setter re-applies
//     the current parameter values so the distribution re-syncs at the new index), Alpha
//     (473), TrendModels (386), SetTrendModel (794), and the nonstationary branches of the
//     DataFrame setter (275) / DataFrame_PropertyChanged (493) / SetDefaultParameters (571)
//     / SetParameterValues (1977) / Validate (2274) / PriorLogLikelihood (1143).
//   - Nonstationary likelihoods: NonstationaryData_LogLikelihood (1256),
//     NonstationaryPointwiseLogLikelihood (1716), NonstationaryPointwiseLogLikelihood-
//     Components (1509), dispatched from LogLikelihood / DataLogLikelihood / the pointwise
//     pair via the C# `IsNonstationary ?` ternaries. FullTimeSeries cadence: the C# getter
//     rebuilds lazily on every access; the C++ DataFrame getter (M4) has the same lazy
//     rebuild and is now const (M9), so the likelihood paths read it exactly as the C# does.
//     Model-boundary rebuild triggers (CreateFullTimeSeries) mirror the C#: the DataFrame
//     setter, the IsNonstationary setter, DataFrame_PropertyChanged, and the
//     SetDefaultParameters/SetTrendModel staleness guard (C# 609/868).
//   - Parameter plumbing: GetParameterValues(index) predicts through the trends (2006),
//     SetDistributionParameterValues (2020), the SetParameterValues ->
//     SetDistributionParameterValues(ParameterTimeIndex) call site (1998), and
//     GetNonstationaryReturnLevel (2035; the C# `null` returns map to an empty vector).
//   - Quantile priors: SetDefaultQuantilePriors (1024) fully ported (the M8 throwing stub
//     is gone, and with it the flag-stays-true-on-throw wart); the single-quantile terms of
//     Prior_LogLikelihood (1854-1857) / PointwisePriorLogLikelihood (1941-1946). DEFERRED:
//     the multi-quantile branch (1858-1875 / 1947-1971) -- its Jacobian term calls
//     IStandardError.QuantileJacobian, which is NOT on the ported distribution base
//     (Numerics' IStandardError surface is unported). Porting only the LogPDF terms would
//     silently drop the Jacobian and misfit, so the branch throws std::logic_error until
//     QuantileJacobian is ported. NOTE (C# fidelity): SetDefaultQuantilePriors does NOT
//     call ProcessQuantilePriors -- _quantilePriorsTrue stays empty (quantile terms inert)
//     until the QuantilePriors setter, Clone(), or an explicit ProcessQuantilePriors runs.
//   - ISimulatable / GenerateRandomValues (2303): guards, then delegates to
//     Distribution.GenerateRandomValues -- the Numerics inverse-CDF Mersenne Twister stream
//     (seed > 0 deterministic, else clock-seeded, exactly the C# semantics).
//   - Clone (2058): deep-copies Parameters/QuantilePriors/TrendModels + the nonstationary
//     flags and calls ProcessQuantilePriors() on the result. DIVERGENCE (the M8-flagged M9
//     decision): the C# clone ALIASES the DataFrame; the value-typed move-only C++ frame
//     cannot alias, so clone() DEEP-COPIES the frame (DataFrame::clone()). Mutating the
//     clone's frame therefore does NOT affect the original's, unlike C#. Like the C# (whose
//     (DataFrame, distribution) ctor throws ArgumentNullException), cloning a model with no
//     frame throws std::invalid_argument. Because the C++ ctor's retained Phase 4 deviation
//     resets the freshly constructed clone's distribution to constraint initials (the C#
//     SetDefaultParameters leaves Distribution's parameters intact), clone() re-syncs the
//     clone's distribution to the original's current parameters after the field writes,
//     restoring the C# end state (seeded simulation from the clone bit-matches the
//     original).
//
// IUnivariateModel (P1): this class now implements the marginal interface (the B1
// BivariateDistribution unblock). The collision the Phase 4 note flagged -- the interface's
// NULLABLE `const UnivariateDistributionBase* distribution() const` vs this class's Phase 4
// `const DistributionBase& distribution() const` reference accessor (two functions differing
// only in return type is ill-formed) -- is resolved by keeping the MUTABLE `distribution()`
// as the reference workhorse and making only the CONST accessor the covariant pointer
// override (they differ in const-qualification, a valid overload set). data_frame() is
// re-exposed as an override (the base's is non-virtual), and is_nonstationary()/validate()
// gained `override`. Zero behavior change to any existing method body.
//
// X1: ParameterNames IS now ported onto the distribution base
// (UnivariateDistributionBase::parameter_names) and wired through the trend/ModelParameter
// naming path. set_distribution / set_default_parameters / set_is_nonstationary /
// set_trend_model set each trend's owner_name from `distribution_->parameter_names()`
// (mirroring the C# TrendModels[i].OwnerName = parameterNames[i] sites), and the trend's
// set_owner_name propagates the name to its ModelParameters. So `owner_name()` is populated
// and PriorInfluenceDiagnostics keeps the distinctly-named prior components (no collapse).
//
// Still deliberately NOT ported:
//   - XML (XElement ctor, ToXElement), INotifyPropertyChanged, [attributes] (WPF display
//     concerns).
//   - C# `_parameters.AddRange(TrendModels[i].Parameters)` ALIASES the trend parameters
//     into the model's list; the C++ ModelParameter is a value type, so `parameters_`
//     holds COPIES and SetParameterValues writes both (which the C# does too -- its trend
//     update is redundant only because of the aliasing). Callers must mutate through
//     set_parameter_values, not by poking parameters() elements, to keep both in sync.
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

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/support/data_component.hpp"
#include "corehydro/models/support/i_univariate_model.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/models/trend_functions/support/i_trend_model.hpp"
#include "corehydro/models/trend_functions/support/trend_model_type.hpp"
#include "corehydro/models/trend_functions/trend_model_factory.hpp"
#include "corehydro/models/univariate_distribution/base/univariate_distribution_model_base.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/math/integration/integration.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class UnivariateDistributionModel : public UnivariateDistributionModelBase,
                                    public ISimulatable<std::vector<double>>,
                                    public IUnivariateModel {
   public:
    using DistributionBase = numerics::distributions::UnivariateDistributionBase;
    using DistributionType = numerics::distributions::UnivariateDistributionType;
    using ITrendModel = trend_functions::ITrendModel;
    using TrendModelType = trend_functions::TrendModelType;

    // C# parameterless ctor (line 32): Log-Pearson Type III and no data.
    UnivariateDistributionModel() {
        set_distribution(create_distribution(DistributionType::LogPearsonTypeIII));
    }

    // C# `UnivariateDistribution(DataFrame, UnivariateDistributionBase)` (line 46). The C#
    // clones the passed distribution; here the caller transfers sole ownership instead.
    UnivariateDistributionModel(DataFrame data_frame,
                                std::unique_ptr<DistributionBase> distribution) {
        if (distribution == nullptr)
            throw std::invalid_argument("distribution");  // C# ArgumentNullException
        set_distribution(std::move(distribution));
        set_data_frame(std::move(data_frame));
    }

    // C# `UnivariateDistribution(DataFrame, UnivariateDistributionType)` (line 63): builds
    // the distribution through the model's supported-type factory.
    UnivariateDistributionModel(DataFrame data_frame, DistributionType distribution_type) {
        set_distribution(create_distribution(distribution_type));
        set_data_frame(std::move(data_frame));
    }

    // Phase 4 compatibility shims (corehydro additions): exact observations as a plain vector.
    // A DataFrame with an ExactSeries (sequential 0-based indexes) is built internally, so
    // the fixture/binding construction paths keep their exact Phase 4 semantics.
    UnivariateDistributionModel(DistributionType distribution_type, std::vector<double> exact_data)
        : UnivariateDistributionModel(make_exact_data_frame(exact_data), distribution_type) {}

    UnivariateDistributionModel(std::unique_ptr<DistributionBase> distribution,
                                std::vector<double> exact_data)
        : UnivariateDistributionModel(make_exact_data_frame(exact_data), std::move(distribution)) {}

    // --- Statics -----------------------------------------------------------------------

    // C# `IsSupportedDistributionType` (line 263): the 15-family whitelist this model
    // supports (the C# _supportedDistributionTypes HashSet).
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

    // C# `CreateDistribution` (line 534): whitelist-checked construction. Delegates to the
    // Numerics factory (identical default-constructed results as the C# `new X()` chain).
    static std::unique_ptr<DistributionBase> create_distribution(
        DistributionType distribution_type) {
        if (!is_supported_distribution_type(distribution_type)) {
            // C# ArgumentOutOfRangeException("Unsupported distribution type.").
            throw std::out_of_range("Unsupported distribution type.");
        }
        return numerics::distributions::create_distribution(distribution_type);
    }

    // --- Distribution property (C# line 328) ---------------------------------------------
    // The mutable accessor is the type-specific workhorse (a reference, non-owning; the
    // internal compute paths and the ctest callers deref it directly). It always points at a
    // live distribution on every construction path, so a reference is faithful there. The
    // const accessor is the covariant IUnivariateModel override (P1): the C# property is
    // `UnivariateDistributionBase? Distribution`, so the interface member is a NULLABLE raw
    // non-owning pointer -- it returns distribution_.get() (nullptr when unset, never
    // deref'd). The two overloads differ in const-qualification (a valid overload set); a
    // const model therefore yields the pointer contract, a mutable one the reference.
    DistributionBase& distribution() { return *distribution_; }
    const DistributionBase* distribution() const override { return distribution_.get(); }

    // IUnivariateModel's DataFrame accessors (P1): the base's same-named accessors are
    // non-virtual and live in an unrelated base, so the interface's pure virtuals are
    // satisfied by re-exposing them here (the point_process_model precedent). They forward to
    // the base state -- the unguarded-deref contract is unchanged; check has_data_frame()
    // where the frame may be absent.
    DataFrame& data_frame() override { return *data_frame_; }
    const DataFrame& data_frame() const override { return *data_frame_; }

    // C# setter: null check, assign, reset the trend models to one ConstantTrend per
    // distribution parameter with OwnerName from the distribution's ParameterNames (C#
    // 338-353), then SetDefaultParameters() and SetDefaultQuantilePriors() (unconditionally --
    // not gated on UseDefaultFlatPriors; the XML `_isDeserializing` suppression is out of
    // scope).
    void set_distribution(std::unique_ptr<DistributionBase> distribution) {
        if (distribution == nullptr)
            throw std::invalid_argument("Distribution");  // C# ArgumentNullException
        distribution_ = std::move(distribution);

        // X1: reset the trends off the distribution's parameter names (C# 339-353). Each
        // ConstantTrend { OwnerName = parameterNames[i] } -- the trend's set_owner_name
        // propagates the name onto its ModelParameters.
        std::vector<std::string> parameter_names = distribution_->parameter_names();
        trend_models_.clear();
        for (int i = 0; i < distribution_->number_of_parameters(); ++i) {
            auto trend = std::make_unique<trend_functions::ConstantTrend>();
            if (static_cast<std::size_t>(i) < parameter_names.size())
                trend->set_owner_name(parameter_names[static_cast<std::size_t>(i)]);
            trend_models_.push_back(std::move(trend));
        }

        set_default_parameters();
        set_default_quantile_priors();
    }

    // C# `DistributionType` property (line 368).
    DistributionType distribution_type() const { return distribution_->type(); }
    void set_distribution_type(DistributionType distribution_type) {
        if (distribution_ != nullptr && distribution_->type() == distribution_type) return;
        set_distribution(create_distribution(distribution_type));
    }

    // --- TrendModels (C# line 386: public get, private set -- list contents mutable). ---
    std::vector<std::unique_ptr<ITrendModel>>& trend_models() { return trend_models_; }
    const std::vector<std::unique_ptr<ITrendModel>>& trend_models() const {
        return trend_models_;
    }

    // --- Nonstationarity (C# region, lines 388-486) ---------------------------------------

    // C# `IsNonstationary` (line 398). Set-false: reset the trends to one ConstantTrend per
    // parameter and SetDefaultParameters (unconditional). Set-true: build the full time
    // series, re-center ParameterTimeIndex off the exact series, re-anchor every trend's
    // StartIndex, then SetDefaultParameters when UseDefaultFlatPriors.
    bool is_nonstationary() const override { return is_nonstationary_; }
    void set_is_nonstationary(bool value) {
        if (is_nonstationary_ == value) return;
        is_nonstationary_ = value;

        if (!is_nonstationary_ && distribution_ != nullptr) {
            // X1: reset the trends off the distribution's parameter names (mirrors C# 339-353).
            std::vector<std::string> parameter_names = distribution_->parameter_names();
            trend_models_.clear();
            for (int i = 0; i < distribution_->number_of_parameters(); ++i) {
                auto trend = std::make_unique<trend_functions::ConstantTrend>();
                if (static_cast<std::size_t>(i) < parameter_names.size())
                    trend->set_owner_name(parameter_names[static_cast<std::size_t>(i)]);
                trend_models_.push_back(std::move(trend));
            }
            set_default_parameters();
        } else {
            if (has_data_frame()) {
                data_frame().create_full_time_series();
                if (data_frame().exact_series().count() > 0) {
                    set_parameter_time_index(static_cast<int>(
                        std::ceil((data_frame().exact_series().minimum_index() +
                                   data_frame().exact_series().maximum_index()) /
                                  2.0)));
                }
                const std::vector<std::unique_ptr<Data>>& fts = data_frame().full_time_series();
                if (!fts.empty()) {
                    int start_index = fts.front()->index();
                    for (std::size_t i = 0; i < trend_models_.size(); ++i)
                        trend_models_[i]->set_start_index(start_index);
                }
            }
            if (use_default_flat_priors()) set_default_parameters();
        }
    }

    // C# `ParameterTimeIndex` (line 451): the time step index at which the nonstationary
    // distribution is evaluated. The setter re-applies the current parameter values so the
    // distribution re-syncs at the new index (C# SetParameterValues call, line 461).
    int parameter_time_index() const { return parameter_time_index_; }
    void set_parameter_time_index(int value) {
        if (parameter_time_index_ == value) return;
        parameter_time_index_ = value;
        std::vector<double> values;
        values.reserve(parameters_.size());
        for (const ModelParameter& mp : parameters_) values.push_back(mp.value());
        set_parameter_values(values);
    }

    // C# `Alpha` (line 473): the exceedance probability for the nonstationary chronology;
    // default 0.5 (the 2-year return period).
    double alpha() const { return alpha_; }
    void set_alpha(double value) { alpha_ = value; }

    // C# concrete `DataFrame` setter override (line 275): store, reprocess thresholds at
    // the model boundary (the M4->M8 cadence), the nonstationary block (build the full time
    // series, re-center an out-of-range ParameterTimeIndex off the exact series, re-anchor
    // trend StartIndex), then SetDefaultParameters when UseDefaultFlatPriors.
    void set_data_frame(DataFrame data_frame) override {
        data_frame_ = std::move(data_frame);
        data_frame_->process_threshold_series();

        if (is_nonstationary_) {
            data_frame_->create_full_time_series();
            const std::vector<std::unique_ptr<Data>>& fts = data_frame_->full_time_series();
            if (!fts.empty()) {
                int first_index = fts.front()->index();
                int last_index = fts.back()->index();

                if (parameter_time_index_ < first_index ||
                    parameter_time_index_ > last_index + 100) {
                    if (data_frame_->exact_series().count() > 0) {
                        set_parameter_time_index(static_cast<int>(
                            std::ceil((data_frame_->exact_series().minimum_index() +
                                       data_frame_->exact_series().maximum_index()) /
                                      2.0)));
                    }
                }

                for (std::size_t i = 0; i < trend_models_.size(); ++i)
                    trend_models_[i]->set_start_index(first_index);
            }
        }

        if (use_default_flat_priors()) set_default_parameters();
    }

    // C# `SetTrendModel(int index, TrendModelType type)` (line 794): replaces the indexed
    // trend with a freshly constructed one, applies the constraint-driven defaults when a
    // valid frame with exact data exists, and rebuilds the Parameters list from all trends.
    // Defined out-of-line in univariate_distribution_model_trends.hpp.
    void set_trend_model(int index, TrendModelType type);

    // --- SetDefaultParameters (C# line 571, both paths) ------------------------------------
    // Constraint path: trend defaults + StartIndex anchoring + the per-trend-type bound
    // blocks (Linear/Quadratic/Cubic/Exponential/Logistic/Sinusoidal/StepFunction), with
    // IsPositive = (lowers[i] == Tools.DoubleMachineEpsilon) on the location parameter (C#
    // 628) AND on the StepFunction level parameter (C# 728 -- the M8 review ledger item).
    // Skip path (null/invalid/empty frame): the trends keep their own defaults. Either way
    // the Parameters list is rebuilt from all trend parameters, with names cleared for the
    // first NumberOfParameters entries when stationary (C# 756-760). Defined out-of-line in
    // univariate_distribution_model_trends.hpp.
    void set_default_parameters() override;

    // --- Quantile priors (see the file-header notes on the multi-quantile deferral) -------

    // C# `SetDefaultQuantilePriors` (line 1024). Disabled: clear both lists (C# 1037-1043).
    // Enabled: one LnNormal prior per quantile (1 with UseSingleQuantile, else one per
    // distribution parameter) at alpha = 10^-i, mu = Round(InverseCDF(1 - alpha), 2), sigma
    // = Round(0.15 * mu, 2); an existing list is shrunk from the back or extended with
    // alpha = last / 10 (C# 1045-1078). NOTE: like the C#, this does NOT reprocess --
    // _quantilePriorsTrue is untouched by the enabled branch.
    void set_default_quantile_priors() override {
        if (distribution_ == nullptr) return;
        // (Handler removal for the old priors is INPC plumbing, skipped.)
        if (!enable_quantile_priors()) {
            quantile_priors_.clear();
            quantile_priors_true_.clear();
            return;
        }

        std::vector<QuantilePrior> priors;
        std::size_t q_count = use_single_quantile()
                                  ? 1u
                                  : static_cast<std::size_t>(distribution_->number_of_parameters());

        // See if there are already quantile priors (the C# ToList copy aliases the prior
        // objects; the C++ copy is deep -- same observable end state since the old list is
        // replaced wholesale below).
        if (!quantile_priors_.empty()) {
            priors = quantile_priors_;
            if (priors.size() > q_count) {
                while (priors.size() != q_count) priors.pop_back();
            } else if (priors.size() < q_count) {
                while (priors.size() < q_count) {
                    priors.emplace_back(priors.back().alpha() / 10.0,
                                        std::make_unique<numerics::distributions::LnNormal>());
                    double mu = round_half_even_2(
                        distribution_->inverse_cdf(1.0 - priors.back().alpha()));
                    double sigma = round_half_even_2(mu * 0.15);
                    priors.back().distribution().set_parameters({mu, sigma});
                }
            }
        } else {
            for (std::size_t i = 1; i <= q_count; ++i) {
                priors.emplace_back(std::pow(10.0, -static_cast<double>(i)),
                                    std::make_unique<numerics::distributions::LnNormal>());
                double mu =
                    round_half_even_2(distribution_->inverse_cdf(1.0 - priors[i - 1].alpha()));
                double sigma = round_half_even_2(mu * 0.15);
                priors[i - 1].distribution().set_parameters({mu, sigma});
            }
        }

        // Reset the quantile priors with the new list (handler re-adds skipped).
        quantile_priors_ = std::move(priors);
    }

    // C# `ProcessQuantilePriors` (line 1089): the first prior stays; the remaining priors
    // convert to Gamma distributions of the difference in random quantiles.
    void process_quantile_priors() override {
        quantile_priors_true_.clear();
        if (quantile_priors_.size() == 1) {
            quantile_priors_true_.push_back(quantile_priors_[0].clone());
            return;
        }
        if (!use_single_quantile() &&
            static_cast<int>(quantile_priors_.size()) == distribution_->number_of_parameters()) {
            // The first quantile prior remains the same.
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

    // --- Likelihood surface ----------------------------------------------------------------

    // C# `LogLikelihood` override (line 1116): ONE working copy of the distribution is
    // threaded through the data likelihood and then the prior likelihood (so the Jeffreys
    // term sees the parameters the data step set), then the finite collapse.
    double log_likelihood(std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error("Distribution must be set before computing log likelihood.");

        std::unique_ptr<DistributionBase> model = distribution_->clone();
        double data_log_lh = is_nonstationary_ ? nonstationary_data_log_likelihood(*model, p)
                                               : stationary_data_log_likelihood(*model, p);
        double prior_log_lh = prior_log_likelihood(*model, p);

        double log_lh = data_log_lh + prior_log_lh;
        return numerics::is_finite(log_lh) ? log_lh : -std::numeric_limits<double>::infinity();
    }

    // C# `PriorLogLikelihood` override (line 1143): priors are evaluated against the
    // distribution at the LAST TIME STEP for nonstationary models (ParameterTimeIndex is
    // for prediction only, per the C# remark) or at the supplied parameters when
    // stationary. Invalid proposals leave the working copy at its previous parameters (see
    // the validity-check note in the file header).
    double prior_log_likelihood(std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution must be set before computing prior log likelihood.");
        if (p.size() != parameters_.size()) return -std::numeric_limits<double>::infinity();

        std::unique_ptr<DistributionBase> model = distribution_->clone();
        std::vector<double> dist_params;
        if (is_nonstationary_) {
            if (!has_data_frame() || data_frame().full_time_series().empty())
                return -std::numeric_limits<double>::infinity();
            dist_params = get_parameter_values(data_frame().full_time_series().back()->index());
        } else {
            dist_params.assign(
                p.begin(),
                p.begin() + static_cast<std::ptrdiff_t>(model->number_of_parameters()));
        }

        std::unique_ptr<DistributionBase> probe = distribution_->clone();
        probe->set_parameters(dist_params);
        if (probe->parameters_valid()) model->set_parameters(dist_params);

        return prior_log_likelihood(*model, p);
    }

    // C# `StationaryData_LogLikelihood` (line 1181): the full stationary censored
    // likelihood -- the type-switch over the four series.
    double stationary_data_log_likelihood(DistributionBase& model,
                                          const std::vector<double>& p) const {
        if (!has_data_frame())
            throw std::runtime_error("DataFrame must be set before computing log likelihood.");

        // Check if proposed parameters are valid (probe clone; see file header). The C#
        // returns -inf WITHOUT setting the working copy.
        std::unique_ptr<DistributionBase> probe = model.clone();
        probe->set_parameters(p);
        if (!probe->parameters_valid()) return -std::numeric_limits<double>::infinity();

        // Set model parameters.
        model.set_parameters(p);

        const DataFrame& df = data_frame();
        double log_lh = 0.0;

        // Exact Data (low outliers -> left-censored at the frame's LowOutlierThreshold).
        for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
            const ExactData& data = df.exact_series()[i];
            if (!data.is_low_outlier()) {
                log_lh += model.log_likelihood(data.value());
            } else {
                log_lh += model.log_likelihood_left_censored(df.low_outlier_threshold(), 1);
            }
        }

        // Uncertain Data (measurement error): integrate dist.PDF(q) * model.PDF(q) over the
        // 1e-8 probability window and normalize by the retained mass, with the C# guards.
        for (std::size_t i = 0; i < df.uncertain_series().count(); ++i) {
            const DistributionBase& dist = df.uncertain_series()[i].distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            if (!numerics::is_finite(a) || !numerics::is_finite(b) ||
                !numerics::is_finite(mass) || mass <= 0.0 || a >= b) {
                log_lh += -std::numeric_limits<double>::infinity();
                continue;
            }

            // Normalize by retained ME mass so the likelihood remains conditional on the
            // same 1E-8 probability window used across uncertain-data models.
            double ep = numerics::math::integration::Integration::gauss_legendre20(
                            [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                        mass;
            log_lh += ep > 0 ? std::log(ep) : -std::numeric_limits<double>::infinity();
        }

        // Interval Data.
        for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
            const IntervalData& data = df.interval_series()[i];
            log_lh += model.log_likelihood_intervals(data.lower_value(), data.upper_value());
        }

        // Threshold Data (a zero count on either censored side contributes exactly zero).
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            if (data.number_below() > 0)
                log_lh += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                log_lh += model.log_likelihood_right_censored(data.value(), data.number_above());
        }

        return log_lh;
    }

    // C# `NonstationaryData_LogLikelihood(model, parameters)` (line 1256; public upstream):
    // trend-model clones carry the proposed parameter slices, and every full-time-series
    // ordinate is evaluated at its own Predict(index) parameter set. Defined out-of-line in
    // univariate_distribution_model_trends.hpp.
    double nonstationary_data_log_likelihood(DistributionBase& model,
                                             const std::vector<double>& p) const;

    // C# `DataLogLikelihood` override (line 1361).
    double data_log_likelihood(std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution must be set before computing data log likelihood.");
        std::unique_ptr<DistributionBase> model = distribution_->clone();
        double log_lh = is_nonstationary_ ? nonstationary_data_log_likelihood(*model, p)
                                          : stationary_data_log_likelihood(*model, p);
        if (!numerics::is_finite(log_lh)) return -std::numeric_limits<double>::infinity();
        return log_lh;
    }

    // C# `PointwiseDataLogLikelihood` override (line 1373).
    std::vector<double> pointwise_data_log_likelihood(const std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution must be set before computing pointwise log likelihood.");
        std::unique_ptr<DistributionBase> model = distribution_->clone();
        return is_nonstationary_ ? nonstationary_pointwise_log_likelihood(*model, p)
                                 : stationary_pointwise_log_likelihood(*model, p);
    }

    // C# `PointwiseDataLogLikelihoodComponents` override (line 1385).
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution must be set before computing pointwise log likelihood components.");
        std::unique_ptr<DistributionBase> model = distribution_->clone();
        return is_nonstationary_ ? nonstationary_pointwise_log_likelihood_components(*model, p)
                                 : stationary_pointwise_log_likelihood_components(*model, p);
    }

    // C# `Prior_LogLikelihood(model, parameters)` (line 1822): parameter priors + the
    // Jeffreys 1/scale term read off the WORKING COPY's parameters (not the raw proposal --
    // this is why the callers set `model` first) + the quantile-prior terms (single-quantile
    // ported; the multi-quantile branch is DEFERRED, see the file header).
    double prior_log_likelihood(const DistributionBase& model,
                                const std::vector<double>& p) const {
        double log_lh = 0.0;

        // Parameter priors.
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            log_lh += parameters_[i].prior_distribution().log_pdf(p[i]);
        }

        // T14 (BestFit v2.0.0, commit 1abe795): TryGetJeffreysScaleParameter reports `false`
        // -- rather than indexing GetParameters out of range -- for a single-parameter family
        // with no applicable scale term, in which case the Jeffreys term is skipped entirely
        // (no contribution, positive or negative, to log_lh). This is now a straight port:
        // the base's guard matches C#'s own guard, not a documented divergence (see the base
        // header's T14 note).
        double scale_param;
        if (use_jeffreys_rule_for_scale() &&
            try_get_jeffreys_scale_parameter(model, scale_param)) {
            // Jeffreys prior requires a positive scale parameter: return -inf directly
            // rather than subtracting +inf (C# early-return, line 1849).
            if (scale_param <= 0) return -std::numeric_limits<double>::infinity();
            log_lh -= std::log(scale_param);
        }

        // Quantile priors (C# 1853-1875).
        if (enable_quantile_priors() && use_single_quantile() &&
            quantile_priors_true_.size() == 1) {
            log_lh += quantile_priors_true_[0].distribution().log_pdf(
                model.inverse_cdf(1.0 - quantile_priors_true_[0].alpha()));
        } else if (enable_quantile_priors() && !use_single_quantile() &&
                   static_cast<int>(quantile_priors_true_.size()) ==
                       distribution_->number_of_parameters()) {
            // DEFERRED (C# 1858-1875): the difference-prior terms are portable but the
            // branch's Jacobian term calls IStandardError.QuantileJacobian, which is not on
            // the ported distribution base. Omitting the Jacobian would silently misfit, so
            // the whole branch throws until QuantileJacobian is ported (file-header note).
            throw std::logic_error(
                "UnivariateDistributionModel: the multi-quantile prior branch requires "
                "IStandardError::QuantileJacobian, which is not ported yet; use a single "
                "quantile prior (use_single_quantile) or disable quantile priors.");
        }

        if (!numerics::is_finite(log_lh)) return -std::numeric_limits<double>::infinity();
        return log_lh;
    }

    // C# `PointwisePriorLogLikelihood` override (line 1882): the per-parameter prior
    // components plus one JeffreysScale component -- appended only when the proposed
    // distribution parameters are VALID (C# `valid is null`, line 1917), unlike the scalar
    // Prior_LogLikelihood which evaluates the term unconditionally off the working copy --
    // plus the quantile-prior components (single-quantile ported; multi-quantile DEFERRED,
    // see the file header). Nonstationary: the distribution parameters come from the last
    // time step (matches PriorLogLikelihood; ParameterTimeIndex is for prediction).
    // Component label: "Jeffreys Scale: {scaleName}", the ParameterNames entry at the scale
    // index (C# line 1912; X1 ported ParameterNames onto the distribution base, so the name
    // is available here too).
    std::vector<PriorComponent> pointwise_prior_log_likelihood(
        const std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution must be set before computing pointwise prior log likelihood.");

        std::vector<PriorComponent> result;
        // Length guard (deviation, documented: the C# indexes parameters[i] unguarded and
        // would throw on a mismatch; empty mirrors the ModelBase base behavior).
        if (p.size() != parameters_.size()) return result;

        std::unique_ptr<DistributionBase> model = distribution_->clone();
        std::vector<double> dist_params;
        if (is_nonstationary_) {
            // C# 1894-1899: no frame / empty full series -> empty component list.
            if (!has_data_frame() || data_frame().full_time_series().empty()) return result;
            dist_params = get_parameter_values(data_frame().full_time_series().back()->index());
        } else {
            dist_params.assign(
                p.begin(),
                p.begin() + static_cast<std::ptrdiff_t>(model->number_of_parameters()));
        }
        std::unique_ptr<DistributionBase> probe = distribution_->clone();
        probe->set_parameters(dist_params);
        bool valid = probe->parameters_valid();
        if (valid) model->set_parameters(dist_params);

        // Parameter priors.
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            double ll = parameters_[i].prior_distribution().log_pdf(p[i]);
            const std::string& param_name = parameters_[i].owner_name().empty()
                                                ? parameters_[i].name()
                                                : parameters_[i].owner_name();
            result.emplace_back("Parameter Prior: " + param_name, ll,
                                PriorComponentType::ParameterPrior);
        }

        // Jeffreys rule for scale parameter (only when the proposal is valid, per the C#).
        // TryGetJeffreysScaleParameter reports `false` for a single-parameter family with no
        // applicable scale term, in which case no component is appended (T14; see the base
        // header's note).
        if (use_jeffreys_rule_for_scale() && valid) {
            double scale;
            std::string scale_name;
            if (try_get_jeffreys_scale_parameter(*model, scale, scale_name)) {
                double ll =
                    scale > 0 ? -std::log(scale) : -std::numeric_limits<double>::infinity();
                result.emplace_back("Jeffreys Scale: " + scale_name, ll,
                                    PriorComponentType::JeffreysScalePrior);
            }
        }

        // Quantile priors (C# 1939-1972; components only when the proposal is valid).
        if (enable_quantile_priors() && valid) {
            if (use_single_quantile() && quantile_priors_true_.size() == 1) {
                double quantile = model->inverse_cdf(1.0 - quantile_priors_true_[0].alpha());
                double ll = quantile_priors_true_[0].distribution().log_pdf(quantile);
                // C# label $"Quantile Prior: p={Alpha:G4}"; :G4 approximated with %.4g
                // (display-only, the PriorComponent::to_string precedent).
                char alpha_buf[64];
                std::snprintf(alpha_buf, sizeof alpha_buf, "%.4g",
                              quantile_priors_true_[0].alpha());
                result.emplace_back(std::string("Quantile Prior: p=") + alpha_buf, ll,
                                    PriorComponentType::QuantilePrior);
            } else if (!use_single_quantile() &&
                       static_cast<int>(quantile_priors_true_.size()) ==
                           distribution_->number_of_parameters()) {
                // DEFERRED: same IStandardError::QuantileJacobian gap as the scalar
                // Prior_LogLikelihood branch (file-header note).
                throw std::logic_error(
                    "UnivariateDistributionModel: the multi-quantile prior branch requires "
                    "IStandardError::QuantileJacobian, which is not ported yet; use a single "
                    "quantile prior (use_single_quantile) or disable quantile priors.");
            }
        }

        return result;
    }

    // C# `SetParameterValues` override (line 1977): writes the model parameters, updates
    // the trend models with the per-trend parameter slices, then sets the distribution
    // parameters via SetDistributionParameterValues(ParameterTimeIndex) -- stationary, the
    // ConstantTrend chain collapses to the parameter values themselves.
    void set_parameter_values(const std::vector<double>& p) override {
        if (p.size() != static_cast<std::size_t>(number_of_parameters())) {
            throw std::invalid_argument("The length of the parameter list is incorrect.");
        }
        for (std::size_t i = 0; i < p.size(); ++i) parameters_[i].set_value(p[i]);

        std::size_t t = 0;
        for (std::size_t i = 0; i < trend_models_.size(); ++i) {
            std::size_t n = static_cast<std::size_t>(trend_models_[i]->number_of_parameters());
            std::vector<double> parms(p.begin() + static_cast<std::ptrdiff_t>(t),
                                      p.begin() + static_cast<std::ptrdiff_t>(t + n));
            trend_models_[i]->set_parameter_values(parms);
            t += n;
        }

        set_distribution_parameter_values(parameter_time_index_);
    }

    // C# `GetParameterValues(int index)` (line 2006): the distribution parameters at a time
    // step, predicted through the trend models (stationary ConstantTrends collapse to the
    // parameter values, index unused).
    std::vector<double> get_parameter_values(int index) const {
        std::vector<double> values(
            static_cast<std::size_t>(distribution_->number_of_parameters()));
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = trend_models_[i]->predict(index);
        return values;
    }

    // C# `SetDistributionParameterValues(int index)` (line 2020).
    void set_distribution_parameter_values(int index) {
        distribution_->set_parameters(get_parameter_values(index));
    }

    // C# `GetNonstationaryReturnLevel()` (line 2035): InverseCDF(1 - Alpha) at every index
    // from the first full-time-series index to max(ParameterTimeIndex, last index). The C#
    // returns null when the model is stationary or the series is unavailable; the C++
    // `null` maps to an empty vector.
    std::vector<double> get_nonstationary_return_level() const {
        if (!is_nonstationary_ || !has_data_frame()) return {};
        const std::vector<std::unique_ptr<Data>>& fts = data_frame().full_time_series();
        if (fts.empty()) return {};

        std::unique_ptr<DistributionBase> dist = distribution_->clone();
        // C# Tools.Sequence(first, Math.Max(ParameterTimeIndex, last)): inclusive step-1.
        int first_index = fts.front()->index();
        int last_index = std::max(parameter_time_index_, fts.back()->index());

        std::vector<double> result;
        result.reserve(static_cast<std::size_t>(last_index - first_index + 1));
        for (int index = first_index; index <= last_index; ++index) {
            dist->set_parameters(get_parameter_values(index));
            result.push_back(dist->inverse_cdf(1.0 - alpha_));
        }
        return result;
    }

    // C# `GenerateRandomValues(int sampleSize, int seed = -1)` (line 2303, the
    // ISimulatable<double[]> surface): guards, then delegates to
    // Distribution.GenerateRandomValues -- the Numerics inverse-CDF Mersenne Twister stream
    // at the distribution's CURRENT parameters (seed > 0 deterministic, else clock-seeded).
    // C# ArgumentOutOfRangeException -> std::out_of_range per the file-header mapping.
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size <= 0)
            throw std::out_of_range("Sample size must be positive.");
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution cannot be null when generating random values.");

        return distribution_->generate_random_values(sample_size, seed);
    }

    // C# `Clone()` (line 2058): deep-copies Parameters, QuantilePriors, TrendModels, and
    // the nonstationary flags, then calls ProcessQuantilePriors() on the result. C# returns
    // IModel; the C++ Clone returns the concrete model by value (ModelBase carries no
    // clone(), a Phase 4 decision). DIVERGENCE (see the file header): the C# ALIASES the
    // DataFrame into the clone; the value-typed C++ frame is DEEP-COPIED instead. Like the
    // C# ctor the clone routes through, a model with no frame cannot be cloned
    // (ArgumentNullException -> std::invalid_argument).
    UnivariateDistributionModel clone() const {
        if (!has_data_frame())
            throw std::invalid_argument("dataFrame");  // C# ctor ArgumentNullException

        std::vector<ModelParameter> parms;
        parms.reserve(parameters_.size());
        for (std::size_t i = 0; i < parameters_.size(); ++i)
            parms.push_back(parameters_[i].clone());

        std::vector<QuantilePrior> quants;
        quants.reserve(quantile_priors_.size());
        for (std::size_t i = 0; i < quantile_priors_.size(); ++i)
            quants.push_back(quantile_priors_[i].clone());

        std::vector<std::unique_ptr<ITrendModel>> trends;
        trends.reserve(trend_models_.size());
        for (std::size_t i = 0; i < trend_models_.size(); ++i)
            trends.push_back(trend_models_[i]->clone());

        // C# `new UnivariateDistribution(DataFrame, Distribution)` (the ctor clones the
        // distribution; the C++ ctor takes ownership of a clone) followed by the object
        // initializer -- direct field writes here mirror the C# field-initializer syntax
        // (no setter side effects).
        UnivariateDistributionModel result(data_frame().clone(), distribution_->clone());
        result.use_default_flat_priors_ = use_default_flat_priors_;
        result.use_jeffreys_rule_for_scale_ = use_jeffreys_rule_for_scale_;
        result.enable_quantile_priors_ = enable_quantile_priors_;
        result.use_single_quantile_ = use_single_quantile_;
        result.is_nonstationary_ = is_nonstationary_;
        result.parameter_time_index_ = parameter_time_index_;
        result.alpha_ = alpha_;
        result.parameters_ = std::move(parms);
        result.quantile_priors_ = std::move(quants);
        result.trend_models_ = std::move(trends);

        // Re-sync the cloned distribution to the original's current parameters. The C#
        // ctor's `Distribution = distribution.Clone()` keeps them (its SetDefaultParameters
        // never calls Distribution.SetParameters); the C++ ctor's retained Phase 4
        // deviation resets the clone's distribution to constraint initials, so without
        // this write the clone's ISimulatable stream silently diverges from the original's.
        result.distribution_->set_parameters(distribution_->get_parameters());

        result.process_quantile_priors();
        return result;
    }

    // C# `Validate` override (line 2127), including the nonstationary checks (2274-2293).
    ValidationResult validate() const override {
        ValidationResult result;

        // DataFrame checks (C# null -> error + early return; the C++ "null" is the
        // never-set optional -- see the base header's nullability note).
        if (!has_data_frame()) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The input DataFrame is null.");
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

        // Distribution checks (defensive; every C++ construction path sets it).
        if (distribution_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The parent distribution is null.");
            return result;
        }

        // Distribution type check.
        if (!is_supported_distribution_type(distribution_type())) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The distribution type is not supported.");
        }

        // Uncertain-data quadrature bounds at the 1E-8 probability window (C# 2165-2185).
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

        // Log distribution support checks (C# 2187-2224).
        if (distribution_type() == DistributionType::LnNormal ||
            distribution_type() == DistributionType::LogNormal ||
            distribution_type() == DistributionType::LogPearsonTypeIII) {
            bool non_positive_exact = false;
            for (std::size_t i = 0; i < data_frame().exact_series().count(); ++i) {
                const ExactData& data = data_frame().exact_series()[i];
                if (!data.is_low_outlier() && data.value() <= 0.0) {
                    non_positive_exact = true;
                    break;
                }
            }

            bool non_positive_uncertain = false;
            for (std::size_t i = 0; i < data_frame().uncertain_series().count(); ++i) {
                const DistributionBase& dist = data_frame().uncertain_series()[i].distribution();
                double lower = dist.inverse_cdf(1E-8);
                // A positive ME mean is not enough; the retained 1E-8 lower support must be
                // positive.
                if (!numerics::is_finite(lower) || lower <= 0.0) {
                    non_positive_uncertain = true;
                    break;
                }
            }

            if (non_positive_exact || non_positive_uncertain) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: Log based distributions cannot be used because some data values "
                    "are non positive.");
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

        // Quantile priors (C# 2241-2271): per-prior validation plus the strictly-decreasing
        // alpha / increasing mean / increasing 5-95 bound ordering rules.
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

        // Nonstationary checks (C# 2274-2293). The frame is non-null here (early return
        // above); the C# reads DataFrame.FullTimeSeries through its lazy getter.
        if (is_nonstationary_) {
            const std::vector<std::unique_ptr<Data>>& fts = data_frame().full_time_series();
            if (!fts.empty()) {
                int first_index = fts.front()->index();
                int last_index = fts.back()->index();

                if (parameter_time_index_ < first_index ||
                    parameter_time_index_ > last_index + 100) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: ParameterTimeIndex is outside the valid range for the full "
                        "time series.");
                }
            }

            if (alpha_ <= 0.0 || alpha_ >= 1.0) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: Alpha must be strictly between 0 and 1 for nonstationary "
                    "analysis.");
            }
        }

        return result;
    }

   protected:
    // C# `DataFrame_PropertyChanged` override (line 493): the explicit-invalidation
    // equivalent for in-place frame mutations (see the base header's cadence note) -- the
    // PlottingParameter/PlottingPosition filtering has no C++ trigger to filter. Reprocess
    // thresholds, SetDefaultParameters when flat priors, then the nonstationary side
    // effects (rebuild the full series, re-center ParameterTimeIndex, re-anchor trend
    // StartIndex). NEVER called from the likelihood paths.
    void data_frame_property_changed() override {
        if (!has_data_frame()) return;

        data_frame().process_threshold_series();

        if (use_default_flat_priors()) set_default_parameters();

        if (is_nonstationary_) {
            if (data_frame().exact_series().count() > 0) {
                data_frame().create_full_time_series();
                set_parameter_time_index(static_cast<int>(
                    std::ceil((data_frame().exact_series().minimum_index() +
                               data_frame().exact_series().maximum_index()) /
                              2.0)));
                const std::vector<std::unique_ptr<Data>>& fts = data_frame().full_time_series();
                if (!fts.empty()) {
                    int start_index = fts.front()->index();
                    for (std::size_t i = 0; i < trend_models_.size(); ++i)
                        trend_models_[i]->set_start_index(start_index);
                }
            }
        }
    }

   private:
    // Builds the exact-data frame the Phase 4 vector ctors wrap (sequential 0-based
    // indexes, exactly what ExactSeries(values) constructs).
    static DataFrame make_exact_data_frame(const std::vector<double>& values) {
        DataFrame df;
        df.set_exact_series(ExactSeries(values));
        return df;
    }

    // C# `Math.Round(value, 2)` (MidpointRounding.ToEven): std::nearbyint under the default
    // FE_TONEAREST mode rounds half-to-even. (.NET's decimal-digit rounding can differ in
    // rare representation edge cases; these feed default priors only, not oracle values.)
    static double round_half_even_2(double value) {
        return std::nearbyint(value * 100.0) / 100.0;
    }

    // The C# SetTrendModel construction chain (line 817): default ConstantTrend, then the
    // type if-chain. GeneralLinear has no branch upstream and falls through to
    // ConstantTrend, mirrored by the default case. The type -> concrete-class switch itself
    // is hoisted to trend_functions::make_trend_model (trend_model_factory.hpp), shared with
    // models::spec::build_spec_trend (model_spec.hpp), so there is one such switch rather than
    // two; this method is now a thin instance-method forwarder kept for its call sites below.
    static std::unique_ptr<ITrendModel> make_trend_model(TrendModelType type) {
        return trend_functions::make_trend_model(type);
    }

    // C# `NonstationaryPointwiseLogLikelihood` (line 1716) and
    // `NonstationaryPointwiseLogLikelihoodComponents` (line 1509). Defined out-of-line in
    // univariate_distribution_model_trends.hpp.
    std::vector<double> nonstationary_pointwise_log_likelihood(
        DistributionBase& model, const std::vector<double>& p) const;
    std::vector<DataComponent> nonstationary_pointwise_log_likelihood_components(
        DistributionBase& model, const std::vector<double>& p) const;

    // C# `StationaryPointwiseLogLikelihood` (line 1632).
    std::vector<double> stationary_pointwise_log_likelihood(DistributionBase& model,
                                                            const std::vector<double>& p) const {
        if (!has_data_frame())
            throw std::runtime_error(
                "DataFrame must be set before computing pointwise log likelihood.");

        const DataFrame& df = data_frame();

        // Invalid parameters -> one -inf per observation across the four series.
        std::unique_ptr<DistributionBase> probe = model.clone();
        probe->set_parameters(p);
        if (!probe->parameters_valid()) {
            std::size_t n = df.exact_series().count() + df.uncertain_series().count() +
                            df.interval_series().count() + df.threshold_series().count();
            return std::vector<double>(n, -std::numeric_limits<double>::infinity());
        }

        model.set_parameters(p);

        std::vector<double> result;

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
                result.push_back(ep > 0 ? std::log(ep)
                                        : -std::numeric_limits<double>::infinity());
            } else {
                result.push_back(-std::numeric_limits<double>::infinity());
            }
        }

        // Interval Data.
        for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
            const IntervalData& data = df.interval_series()[i];
            result.push_back(
                model.log_likelihood_intervals(data.lower_value(), data.upper_value()));
        }

        // Threshold Data -- each threshold contributes one observation.
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            double log_lh = 0.0;
            if (data.number_below() > 0)
                log_lh += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                log_lh += model.log_likelihood_right_censored(data.value(), data.number_above());
            result.push_back(log_lh);
        }

        return result;
    }

    // C# `StationaryPointwiseLogLikelihoodComponents` (line 1402).
    std::vector<DataComponent> stationary_pointwise_log_likelihood_components(
        DistributionBase& model, const std::vector<double>& p) const {
        if (!has_data_frame())
            throw std::runtime_error(
                "DataFrame must be set before computing pointwise log likelihood components.");

        const DataFrame& df = data_frame();
        std::vector<DataComponent> result;
        int idx = 0;
        const double neg_inf = -std::numeric_limits<double>::infinity();

        // Invalid parameters -> list of invalid components with per-type placeholder values
        // (C# 1410-1431).
        std::unique_ptr<DistributionBase> probe = model.clone();
        probe->set_parameters(p);
        if (!probe->parameters_valid()) {
            for (std::size_t i = 0; i < df.exact_series().count(); ++i)
                result.emplace_back(idx++, neg_inf, df.exact_series()[i].value(),
                                    DataComponentType::Exact);
            for (std::size_t i = 0; i < df.uncertain_series().count(); ++i)
                result.emplace_back(idx++, neg_inf,
                                    df.uncertain_series()[i].distribution().mean(),
                                    DataComponentType::Uncertain);
            for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
                const IntervalData& interval = df.interval_series()[i];
                result.emplace_back(idx++, neg_inf,
                                    (interval.lower_value() + interval.upper_value()) / 2.0,
                                    DataComponentType::Interval);
            }
            for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
                const ThresholdData& threshold = df.threshold_series()[i];
                result.emplace_back(idx++, neg_inf, threshold.value(),
                                    DataComponentType::LeftCensored,
                                    threshold.number_below() + threshold.number_above(),
                                    std::to_string(threshold.start_index()) + "-" +
                                        std::to_string(threshold.end_index()));
            }
            return result;
        }

        model.set_parameters(p);

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

        // Threshold Data (LeftCensored type with the combined count, per the C#).
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            double log_lh = 0.0;
            if (data.number_below() > 0)
                log_lh += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                log_lh += model.log_likelihood_right_censored(data.value(), data.number_above());

            int total_count = data.number_below() + data.number_above();
            result.emplace_back(idx++, log_lh, data.value(), DataComponentType::LeftCensored,
                                total_count,
                                std::to_string(data.start_index()) + "-" +
                                    std::to_string(data.end_index()));
        }

        return result;
    }

    std::unique_ptr<DistributionBase> distribution_;
    std::vector<std::unique_ptr<ITrendModel>> trend_models_;  // C# TrendModels (line 386)
    bool is_nonstationary_ = false;                           // C# _isNonstationary (line 270)
    int parameter_time_index_ = 0;                            // C# _parameterTimeIndex (line 271)
    double alpha_ = 0.5;                                      // C# _alpha (line 272)
};

}  // namespace corehydro::models

// Out-of-line definitions of the trend-driven surfaces (SetDefaultParameters,
// SetTrendModel, and the three nonstationary likelihood bodies), split out purely for file
// size like data_frame_plotting.hpp; the companion header must not be included directly.
#include "corehydro/models/univariate_distribution/univariate_distribution_model_trends.hpp"  // NOLINT
