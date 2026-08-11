// corehydro ADDITION -- no upstream C# counterpart.
//
// The shared `model_estimation` fixture model builder. A fixture case's `construct.model`
// object (see fixtures/README.md, "model_estimation") names one of the Phase 5 model types
// (univariate_distribution / mixture / competing_risks / point_process) or one of the Phase 7a
// families (time_series / spatial_gev / rating_curve / bivariate) and carries its data (a flat
// `dataset` reference or a full `data_frame` of exact / interval / threshold / uncertain
// arrays), optional nonstationary trend specs, and optional explicit parameter values. Like
// mcmc/model_registry.hpp, this header exists so the three
// fixture harnesses (C++ test runner, corehydror glue, corehydropy glue) build the EXACT same
// model from the same spec through ONE construction path: each harness re-serializes its
// native `construct.model` structure to a JSON string (nlohmann `dump()` in C++,
// `jsonlite::toJSON(digits = I(17))` in R, `json.dumps()` in Python -- all three round-trip
// doubles exactly) and calls `build_model_from_json`, which parses with models/json_lite.hpp
// and dispatches on the spec's `type` discriminator. There is no closed model-name registry:
// the "registry" is this one dispatch plus the same univariate distribution factory every
// other fixture kind uses.
//
// Spec shape (all types; `type` defaults to "univariate_distribution"):
//   { "type": "univariate_distribution", "family": "<factory name>",
//     "dataset": "<datasets key>" | "data_frame": { ... },
//     "trends": [ { "parameter": p, "type": "<TrendModelType name>",
//                   "start_index"?: i, "values"?: [ ... ] } ],
//     "parameter_values"?: [ ... ] }
// Every type additionally accepts the optional model-level keys `use_default_flat_priors` (bool)
// and `parameters` (a per-parameter override block), both applied by apply_parameter_values
// before the flat `parameter_values` vector:
//   "parameters": [ { "index": i, "value"?: x, "lower"?: lo, "upper"?: hi,
//                     "is_positive"?: b, "is_fixed"?: b, "name"?: "<label>",
//                     "prior"?: { "family": "<factory name>", "parameters": [ ... ] } } ]
//   { "type": "mixture", "families": [ ... ], "zero_inflated"?: b, <data>, ... }
//   { "type": "competing_risks", "families": [ ... ], <data>, ... }
//   { "type": "point_process", <data>, "use_defaults"?: b, "threshold"?: x,
//     "total_years"?: y, ... }
//   { "type": "time_series", "subtype": "ar"|"ma"|"arima"|"arimax", <data>,
//     "orders": { "p"?, "d"?, "q"?, "b"? }, "include_intercept"?: b, "transform"?: "<name>",
//     "trend"?, "include_seasonality"?, "covariates"?, "parameter_values"?: [ ... ] }
//   { "type": "spatial_gev", "coordinates": [[x,y],...], "at_site_data": [[...],...],
//     <use_*_errors / use_copula_dependence / use_log_link_* flags>, "parameter_values"?: [...] }
//   { "type": "rating_curve", "segments"?: 1..3, "stage": [ ... ], "discharge": [ ... ],
//     "parameter_values"?: [ ... ] }
//   { "type": "bivariate", "copula": "<CopulaType>", "estimation_method"?: "<name>",
//     "marginal_x": { "family", "data", "parameter_values" }, "marginal_y": { ... },
//     "parameter_values"?: [ ... ] }
// `dataset` is resolved by the CALLING harness (the file-level `datasets` map never reaches
// this builder); the resolved values arrive as the `dataset` argument and become an
// ExactSeries with sequential 0-based indexes -- exactly the Phase 4 vector-ctor path.
// `data_frame` arrays carry their values inline:
//   exact:     { "index": i, "value": x, "is_low_outlier"?: b }
//   interval:  { "index": i, "lower": lo, "value": x, "upper": hi }
//   threshold: { "start_index": i, "end_index": j, "value": x, "number_above": n }
//   uncertain: { "index": i, "distribution": { "family": "<name>", "parameters": [ ... ] } }
// plus an optional frame-level "low_outlier_threshold" and an optional frame-level
// "mgbt_low_outliers" flag (M14) that triggers the public set_low_outliers_from_mgbt() path
// after the series are assigned.
//
// `trends` attach via UnivariateDistributionModel::set_trend_model (which supplies the
// data-driven defaults exactly like the C# SetTrendModel); `start_index` then overrides the
// trend's anchor, and explicit trend `values` / the model-level `parameter_values` are
// applied through ONE final set_parameter_values call -- the sync-safe setter the model
// header mandates (poking parameters() elements directly would desync the trend copies).
// Attaching any trend first sets is_nonstationary(true).
//
// Every model derives from ModelBase (what the estimators accept) -- including
// CompetingRisksModel, which is NOT an IUnivariateModel (the C# omits it) -- so the builder
// returns std::unique_ptr<ModelBase>. Callers needing the seeded-simulation surface
// dynamic_cast the result to ISimulatable<std::vector<double>>* (implemented by every family
// EXCEPT the bivariate model, which is ISimulatable<Matrix2D>: its seeded draw is dispatched
// through a separate matrix arm in each harness -- flattened ROW-MAJOR -- see fixtures/README.md).
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/bivariate_distribution/bivariate_distribution.hpp"
#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/json_lite.hpp"
#include "corehydro/models/rating_curve/rating_curve.hpp"
#include "corehydro/models/spatial_extremes/spatial_gev.hpp"
#include "corehydro/models/support/i_univariate_model.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/time_series/arima.hpp"
#include "corehydro/models/time_series/arimax.hpp"
#include "corehydro/models/time_series/auto_regressive.hpp"
#include "corehydro/models/time_series/moving_average.hpp"
#include "corehydro/models/time_series/transform_type.hpp"
#include "corehydro/models/trend_functions/general_linear_function.hpp"
#include "corehydro/models/trend_functions/support/trend_model_type.hpp"
#include "corehydro/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "corehydro/models/univariate_distribution/competing_risks_model.hpp"
#include "corehydro/models/univariate_distribution/mixture_model.hpp"
#include "corehydro/models/univariate_distribution/point_process_model.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model_trends.hpp"
#include "corehydro/numerics/data/time_series/support/time_interval.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_estimation_method.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"

namespace corehydro::models::spec {

inline trend_functions::TrendModelType parse_trend_model_type(const std::string& name) {
    using TT = trend_functions::TrendModelType;
    if (name == "Constant") return TT::Constant;
    if (name == "Cubic") return TT::Cubic;
    if (name == "Exponential") return TT::Exponential;
    if (name == "Linear") return TT::Linear;
    if (name == "Logistic") return TT::Logistic;
    if (name == "Power") return TT::Power;
    if (name == "Quadratic") return TT::Quadratic;
    if (name == "Reciprocal") return TT::Reciprocal;
    if (name == "Sinusoidal") return TT::Sinusoidal;
    if (name == "StepFunction") return TT::StepFunction;
    if (name == "GeneralLinear") return TT::GeneralLinear;
    throw std::runtime_error("unknown trend model type: " + name);
}

// A `{ "family": ..., "parameters": [...] }` distribution spec -> a parameterized
// distribution through the same factory every other fixture kind uses.
inline std::unique_ptr<numerics::distributions::UnivariateDistributionBase>
build_spec_distribution(const JsonValue& spec) {
    auto dist = numerics::distributions::create_distribution(spec.at("family").as_string());
    if (spec.contains("parameters")) dist->set_parameters(spec.at("parameters").as_double_vector());
    return dist;
}

// A `data_frame` spec object -> a DataFrame (threshold processing happens later, at the
// model boundary -- every model's set_data_frame runs process_threshold_series itself).
inline DataFrame build_data_frame(const JsonValue& spec) {
    DataFrame df;

    if (spec.contains("exact")) {
        std::vector<ExactData> items;
        for (const JsonValue& e : spec.at("exact").items())
            items.emplace_back(e.at("index").as_int(), e.at("value").as_double(), 0.0,
                               e.value_or("is_low_outlier", false));
        df.set_exact_series(ExactSeries(items));
    }

    if (spec.contains("interval")) {
        std::vector<IntervalData> items;
        for (const JsonValue& e : spec.at("interval").items())
            items.emplace_back(e.at("index").as_int(), e.at("lower").as_double(),
                               e.at("value").as_double(), e.at("upper").as_double());
        df.set_interval_series(IntervalSeries(items));
    }

    if (spec.contains("threshold")) {
        std::vector<ThresholdData> items;
        for (const JsonValue& e : spec.at("threshold").items()) {
            ThresholdData td(e.at("start_index").as_int(), e.at("end_index").as_int(),
                             e.at("value").as_double());
            td.set_number_above(e.at("number_above").as_int());
            items.push_back(td);
        }
        df.set_threshold_series(ThresholdSeries(items));
    }

    if (spec.contains("uncertain")) {
        std::vector<UncertainData> items;
        for (const JsonValue& e : spec.at("uncertain").items())
            items.emplace_back(e.at("index").as_int(),
                               build_spec_distribution(e.at("distribution")));
        df.set_uncertain_series(UncertainSeries(items));
    }

    if (spec.contains("low_outlier_threshold"))
        df.set_low_outlier_threshold(spec.at("low_outlier_threshold").as_double());

    // Optional MGBT trigger (M14): the public set_low_outliers_from_mgbt() path, run at the
    // frame boundary (before the model ctor sees the frame) exactly like a C# caller would --
    // flags low outliers, sets low_outlier_threshold, and thereby left-censors the flagged
    // values in the model's likelihood. Mutually exclusive with explicit `is_low_outlier`
    // flags / `low_outlier_threshold` by fixture convention (MGBT clears both first).
    if (spec.value_or("mgbt_low_outliers", false)) df.set_low_outliers_from_mgbt();

    // Optional explicit-threshold trigger: the sibling public set_low_outliers_from_threshold()
    // path. Unlike a bare `low_outlier_threshold` (which only records the value, leaving the
    // per-ordinate flags to the caller -- the fixture convention above), this derives the flags
    // AND the low-outlier count from the threshold, which is what a user asking to censor below
    // a level means. Opt-in so the existing bare-threshold fixtures are untouched.
    if (spec.value_or("threshold_low_outliers", false)) df.set_low_outliers_from_threshold();

    // Honour the port's invalidation contract (see the strategy note in data_frame.hpp): the C#
    // recomputes plotting positions off the INotifyPropertyChanged cascade that fires when the
    // low-outlier flags change, and this port replaced that plumbing with an explicit call the
    // mutating caller owns. This IS that caller. It matters beyond plotting: the Bulletin 17C
    // ROS imputation (get_nonparametric_moments_ros) reads each ordinate's plotting position,
    // and without this a censored frame reaches it with every position still at its 0.0 default,
    // which throws out of the moment machinery.
    //
    // Guarded on the count rather than run unconditionally: only the two public setters above
    // maintain number_of_low_outliers_, so this fires exactly for frames they touched and leaves
    // every frame built from explicit per-ordinate flags exactly as it was.
    if (df.number_of_low_outliers() > 0) df.calculate_plotting_positions();

    return df;
}

// Resolves a model spec's data source: the inline `data_frame` object when present,
// otherwise an exact-only frame over the harness-resolved `dataset` values (sequential
// 0-based indexes -- the same frame the Phase 4 vector ctors build).
inline DataFrame build_model_data_frame(const JsonValue& model,
                                        const std::vector<double>& dataset) {
    if (model.contains("data_frame")) return build_data_frame(model.at("data_frame"));
    if (model.contains("dataset")) {
        DataFrame df;
        df.set_exact_series(ExactSeries(dataset));
        return df;
    }
    throw std::runtime_error("model spec requires either 'dataset' or 'data_frame'");
}

// Optional model-level `parameters` block: per-parameter bounds / fixed flag / prior
// distribution / starting value, written straight onto the ModelParameter elements the model
// built (models/support/model_parameter.hpp). Every entry names its target by 0-based `index`
// into the model's own parameter vector, so the block composes with `trends` (which decides the
// layout) and runs BEFORE the flat `parameter_values` vector.
//
// Operates on the parameter vector rather than the model so the one implementation serves both
// hierarchies: ModelBase-derived models AND Bulletin17CDistribution, which is an IGMMModel and
// shares only the `parameters()` / `set_parameter_values()` surface.
inline void apply_parameter_overrides(std::vector<ModelParameter>& parameters,
                                      const JsonValue& spec) {
    if (!spec.contains("parameters")) return;
    for (const JsonValue& entry : spec.at("parameters").items()) {
        int index = entry.at("index").as_int();
        if (index < 0 || static_cast<std::size_t>(index) >= parameters.size())
            throw std::runtime_error("parameter spec 'index' is out of range for this model");
        ModelParameter& p = parameters[static_cast<std::size_t>(index)];

        // Bounds first: the prior guard below reads them, and an estimator reads them off the
        // element regardless of whether a prior was supplied.
        if (entry.contains("lower")) p.set_lower_bound(entry.at("lower").as_double());
        if (entry.contains("upper")) p.set_upper_bound(entry.at("upper").as_double());
        if (entry.contains("is_positive")) p.set_is_positive(entry.at("is_positive").as_bool());
        if (entry.contains("is_fixed")) p.set_is_fixed(entry.at("is_fixed").as_bool());
        if (entry.contains("name")) p.set_name(entry.at("name").as_string());
        if (entry.contains("prior")) p.set_prior_distribution(build_spec_distribution(entry.at("prior")));
        if (entry.contains("value")) p.set_value(entry.at("value").as_double());
    }
}

// Optional model-level `use_default_flat_priors` + `parameters` + `parameter_values`, applied in
// that order. The flat-priors flag goes first because a model whose defaults cascade is still
// armed would otherwise overwrite a caller-supplied prior; `parameter_values` goes last because
// it is the sync-safe whole-vector setter (trend copies stay in step) and must win over a
// per-parameter `value`.
inline void apply_parameter_values(ModelBase& model, const JsonValue& spec) {
    if (spec.contains("use_default_flat_priors"))
        model.set_use_default_flat_priors(spec.at("use_default_flat_priors").as_bool());
    apply_parameter_overrides(model.parameters(), spec);
    if (!spec.contains("parameter_values")) return;
    model.set_parameter_values(spec.at("parameter_values").as_double_vector());
}

// `families` -> distribution types through the factory (the string->type mapping every
// other fixture kind already relies on).
inline std::vector<numerics::distributions::UnivariateDistributionType> parse_families(
    const JsonValue& model) {
    std::vector<numerics::distributions::UnivariateDistributionType> types;
    for (const JsonValue& f : model.at("families").items())
        types.push_back(numerics::distributions::create_distribution(f.as_string())->type());
    return types;
}

inline std::unique_ptr<ModelBase> build_univariate_distribution_model(
    const JsonValue& model, const std::vector<double>& dataset) {
    auto dist = numerics::distributions::create_distribution(model.at("family").as_string());

    std::unique_ptr<UnivariateDistributionModel> m;
    if (model.contains("data_frame")) {
        m = std::make_unique<UnivariateDistributionModel>(
            build_data_frame(model.at("data_frame")), std::move(dist));
    } else if (model.contains("dataset")) {
        // The Phase 4 vector-ctor shim, byte-for-byte (mle_normal_smoke.json et al.).
        m = std::make_unique<UnivariateDistributionModel>(std::move(dist), dataset);
    } else {
        throw std::runtime_error("model spec requires either 'dataset' or 'data_frame'");
    }

    if (model.contains("trends")) {
        const std::vector<JsonValue>& trends = model.at("trends").items();
        m->set_is_nonstationary(true);

        // Pass 1: attach every trend (set_trend_model supplies the C#-mirrored defaults),
        // then override the anchor where the spec asks.
        for (const JsonValue& t : trends) {
            int p = t.at("parameter").as_int();
            m->set_trend_model(p, parse_trend_model_type(t.at("type").as_string()));
            if (t.contains("start_index"))
                m->trend_models()[static_cast<std::size_t>(p)]->set_start_index(
                    t.at("start_index").as_int());
        }

        // Pass 2 (after the parameter layout is final): explicit per-trend values overwrite
        // their slice of the full parameter vector, applied through ONE
        // set_parameter_values call (the sync-safe setter -- see the file header).
        bool has_values = false;
        std::vector<double> full;
        full.reserve(m->parameters().size());
        for (const ModelParameter& mp : m->parameters()) full.push_back(mp.value());
        for (const JsonValue& t : trends) {
            if (!t.contains("values")) continue;
            has_values = true;
            int p = t.at("parameter").as_int();
            std::size_t offset = 0;
            for (int j = 0; j < p; ++j)
                offset += static_cast<std::size_t>(
                    m->trend_models()[static_cast<std::size_t>(j)]->number_of_parameters());
            std::vector<double> values = t.at("values").as_double_vector();
            std::size_t n = static_cast<std::size_t>(
                m->trend_models()[static_cast<std::size_t>(p)]->number_of_parameters());
            if (values.size() != n)
                throw std::runtime_error(
                    "trend spec 'values' length does not match the trend's parameter count");
            for (std::size_t k = 0; k < n; ++k) full[offset + k] = values[k];
        }
        if (has_values) m->set_parameter_values(full);
    }

    apply_parameter_values(*m, model);
    return m;
}

inline std::unique_ptr<ModelBase> build_mixture_model(const JsonValue& model,
                                                      const std::vector<double>& dataset) {
    auto m = std::make_unique<MixtureModel>(build_model_data_frame(model, dataset),
                                            parse_families(model),
                                            model.value_or("zero_inflated", false));
    apply_parameter_values(*m, model);
    return m;
}

inline std::unique_ptr<ModelBase> build_competing_risks_model(
    const JsonValue& model, const std::vector<double>& dataset) {
    auto m = std::make_unique<CompetingRisksModel>(build_model_data_frame(model, dataset),
                                                   parse_families(model));
    apply_parameter_values(*m, model);
    return m;
}

inline std::unique_ptr<ModelBase> build_point_process_model(
    const JsonValue& model, const std::vector<double>& dataset) {
    // Ctor surface: default-construct (non-seasonal GEV competing-risks distribution),
    // assign the frame (set_data_frame runs the AMS/defaults cascade), then the optional
    // knobs in C#-property order: UseDefaults before the explicit Threshold/TotalYears so
    // an explicit value is never clobbered by the defaults cascade. The deferred seasonal
    // path is deliberately NOT exposed.
    auto m = std::make_unique<PointProcessModel>();
    m->set_data_frame(build_model_data_frame(model, dataset));
    if (model.contains("use_defaults")) m->set_use_defaults(model.at("use_defaults").as_bool());
    if (model.contains("threshold")) m->set_threshold(model.at("threshold").as_double());
    if (model.contains("total_years")) m->set_total_years(model.at("total_years").as_double());
    apply_parameter_values(*m, model);
    return m;
}

// --- Phase 7a model families (P3) ----------------------------------------------------------
//
// The four remaining ModelBase families wired into the shared spec builder. Each returns a
// std::unique_ptr<ModelBase>, mirroring the Phase 4/5 helpers above. C# governs every ctor /
// setter cascade (verified against the model headers). The seeded-draw surface rides the
// existing ISimulatable arm for the five ISimulatable<std::vector<double>> families; the
// bivariate model is ISimulatable<Matrix2D>, handled by its own matrix arm in each harness.

// A `TransformType` name (transform_type.hpp) -> the Transform enum. Default None.
inline Transform parse_transform(const std::string& name) {
    if (name == "None") return Transform::None;
    if (name == "Logarithmic") return Transform::Logarithmic;
    if (name == "BoxCox") return Transform::BoxCox;
    if (name == "YeoJohnson") return Transform::YeoJohnson;
    throw std::runtime_error("unknown time_series transform: " + name);
}

// A `TimeInterval` name (time_interval.hpp) -> the enum. Default OneDay (the C# field default).
inline numerics::data::TimeInterval parse_time_interval(const std::string& name) {
    using TI = numerics::data::TimeInterval;
    if (name == "OneMinute") return TI::OneMinute;
    if (name == "FiveMinute") return TI::FiveMinute;
    if (name == "FifteenMinute") return TI::FifteenMinute;
    if (name == "ThirtyMinute") return TI::ThirtyMinute;
    if (name == "OneHour") return TI::OneHour;
    if (name == "SixHour") return TI::SixHour;
    if (name == "TwelveHour") return TI::TwelveHour;
    if (name == "OneDay") return TI::OneDay;
    if (name == "SevenDay") return TI::SevenDay;
    if (name == "OneMonth") return TI::OneMonth;
    if (name == "OneQuarter") return TI::OneQuarter;
    if (name == "OneYear") return TI::OneYear;
    if (name == "Irregular") return TI::Irregular;
    throw std::runtime_error("unknown time_series time_interval: " + name);
}

// Wraps a flat value vector into the P2 TimeSeries adapter (data ctor: interval + start index +
// values). `time_interval` defaults OneDay, `start_index` defaults 0 -- the index is only a
// join key (rating_curve) / a sequence position (the ARMA family), never calendar arithmetic.
inline numerics::data::TimeSeries build_time_series(const JsonValue& spec,
                                                    const std::vector<double>& values) {
    numerics::data::TimeInterval interval =
        spec.contains("time_interval") ? parse_time_interval(spec.at("time_interval").as_string())
                                       : numerics::data::TimeInterval::OneDay;
    long start = static_cast<long>(spec.value_or("start_index", 0));
    // Irregular is rejected by the (interval, start, values) ctor in BOTH C# and this port (that
    // ctor walks a REGULAR +1 step). Build it the way the C# regression tests do -- an empty
    // series on the interval, then one Add per value. Index spacing is inert for every model
    // here (the ARMA families index by POSITION), so a +1 walk keeps the C++/C# builders
    // value-identical. Task 21: needed by the TimeInterval.Irregular Validate guard fixtures.
    if (interval == numerics::data::TimeInterval::Irregular) {
        numerics::data::TimeSeries ts(interval);
        for (std::size_t i = 0; i < values.size(); ++i)
            ts.add(numerics::data::TimeSeries::Ordinate(start + static_cast<long>(i), values[i]));
        return ts;
    }
    return numerics::data::TimeSeries(interval, start, values);
}

// Resolves a time-series data source: an inline `data` array when present, otherwise the
// harness-resolved `dataset` vector (like every other kind).
inline std::vector<double> time_series_values(const JsonValue& spec,
                                              const std::vector<double>& dataset) {
    if (spec.contains("data")) return spec.at("data").as_double_vector();
    if (spec.contains("dataset")) return dataset;
    throw std::runtime_error("time_series model requires either 'dataset' or 'data'");
}

// `type: "time_series"` -- `subtype` selects AutoRegressive / MovingAverage / ARIMA / ARIMAX.
// `orders` carries p/d/q/b as the subtype needs; `transform` names a TransformType; `include_
// intercept` defaults true. ARIMAX adds `trend` (Trend enum), `include_seasonality`, and inline
// `covariates` (a list of value arrays). The series data comes from the file-level datasets map
// (or an inline `data` array), wrapped into the P2 TimeSeries adapter.
inline std::unique_ptr<ModelBase> build_time_series_model(const JsonValue& model,
                                                          const std::vector<double>& dataset) {
    std::string subtype = model.at("subtype").as_string();
    numerics::data::TimeSeries ts = build_time_series(model, time_series_values(model, dataset));

    bool include_intercept = model.value_or("include_intercept", true);
    const JsonValue* orders = model.contains("orders") ? &model.at("orders") : nullptr;
    auto order = [&](const char* key, int dflt) {
        return orders && orders->contains(key) ? orders->at(key).as_int() : dflt;
    };

    std::unique_ptr<ModelBase> result;
    if (subtype == "ar") {
        auto m = std::make_unique<AutoRegressive>(ts, order("p", 1), include_intercept);
        if (model.contains("transform")) m->set_transform_type(parse_transform(model.at("transform").as_string()));
        result = std::move(m);
    } else if (subtype == "ma") {
        auto m = std::make_unique<MovingAverage>(ts, order("q", 1), include_intercept);
        if (model.contains("transform")) m->set_transform_type(parse_transform(model.at("transform").as_string()));
        result = std::move(m);
    } else if (subtype == "arima") {
        auto m = std::make_unique<ARIMA>(ts, order("p", 1), order("d", 0), order("q", 0),
                                         include_intercept);
        if (model.contains("transform")) m->set_transform_type(parse_transform(model.at("transform").as_string()));
        result = std::move(m);
    } else if (subtype == "arimax") {
        auto m = std::make_unique<ARIMAX>(ts);
        if (model.contains("transform")) m->set_transform_type(parse_transform(model.at("transform").as_string()));
        m->set_include_intercept(include_intercept);
        if (model.contains("trend")) {
            const std::string& t = model.at("trend").as_string();
            using Trend = ARIMAX::Trend;
            Trend tr = t == "Linear" ? Trend::Linear
                       : t == "Quadratic" ? Trend::Quadratic
                       : t == "Cubic"     ? Trend::Cubic
                                          : Trend::None;
            m->set_trend_type(tr);
        }
        if (model.contains("include_seasonality")) m->set_include_seasonality(model.at("include_seasonality").as_bool());
        m->set_ar_order_p(order("p", 1));
        m->set_diff_order_d(order("d", 0));
        m->set_ma_order_q(order("q", 0));
        m->set_x_order_b(order("b", 0));
        if (model.contains("covariates")) {
            std::vector<numerics::data::TimeSeries> covariates;
            for (const JsonValue& c : model.at("covariates").items())
                covariates.push_back(build_time_series(model, c.as_double_vector()));
            m->set_covariates(covariates);
        }
        result = std::move(m);
    } else {
        throw std::runtime_error("unknown time_series subtype: " + subtype);
    }
    apply_parameter_values(*result, model);
    return result;
}

// `type: "spatial_gev"` -- SpatialGEV(at_site_data [obs][sites], coordinates [sites][2], and the
// three level-2 GeneralLinearFunction trends). The smoke path uses intercept-only trends (no
// covariate design); the optional gating flags (copula dependence / per-parameter errors /
// link toggles) are applied after construction, before parameter_values.
inline std::unique_ptr<ModelBase> build_spatial_gev_model(const JsonValue& model,
                                                          const std::vector<double>& /*dataset*/) {
    std::vector<std::vector<double>> at_site;
    for (const JsonValue& row : model.at("at_site_data").items())
        at_site.push_back(row.as_double_vector());
    std::vector<std::vector<double>> coords;
    for (const JsonValue& row : model.at("coordinates").items())
        coords.push_back(row.as_double_vector());

    // Intercept-only level-2 trends (std::nullopt covariate matrix). Covariate designs are a P4
    // concern -- the smoke path proves the base 3-parameter model builds + fits.
    trend_functions::GeneralLinearFunction location("Location");
    trend_functions::GeneralLinearFunction scale("Scale");
    trend_functions::GeneralLinearFunction shape("Shape");

    auto m = std::make_unique<spatial_extremes::SpatialGEV>(std::move(at_site), std::move(coords),
                                                            std::move(location), std::move(scale),
                                                            std::move(shape));
    if (model.contains("use_copula_dependence")) m->set_use_copula_dependence(model.at("use_copula_dependence").as_bool());
    if (model.contains("use_location_errors")) m->set_use_location_errors(model.at("use_location_errors").as_bool());
    if (model.contains("use_scale_errors")) m->set_use_scale_errors(model.at("use_scale_errors").as_bool());
    if (model.contains("use_shape_errors")) m->set_use_shape_errors(model.at("use_shape_errors").as_bool());
    if (model.contains("use_log_link_for_location")) m->set_use_log_link_for_location(model.at("use_log_link_for_location").as_bool());
    if (model.contains("use_log_link_for_scale")) m->set_use_log_link_for_scale(model.at("use_log_link_for_scale").as_bool());
    apply_parameter_values(*m, model);
    return m;
}

// `type: "rating_curve"` -- RatingCurve(stage TimeSeries, discharge TimeSeries, segments). Both
// series are wrapped from inline arrays sharing a start index so the date-inner-join aligns
// them 1:1 (the model enforces MinimumAlignedObservations = 10).
inline std::unique_ptr<ModelBase> build_rating_curve_model(const JsonValue& model,
                                                           const std::vector<double>& /*dataset*/) {
    numerics::data::TimeSeries stage = build_time_series(model, model.at("stage").as_double_vector());
    numerics::data::TimeSeries discharge =
        build_time_series(model, model.at("discharge").as_double_vector());
    auto m = std::make_unique<RatingCurve>(stage, discharge, model.value_or("segments", 1));
    apply_parameter_values(*m, model);
    return m;
}

// A `CopulaType` name (copula_type.hpp) -> the enum.
inline numerics::distributions::copulas::CopulaType parse_copula_type(const std::string& name) {
    using CT = numerics::distributions::copulas::CopulaType;
    if (name == "AliMikhailHaq") return CT::AliMikhailHaq;
    if (name == "Clayton") return CT::Clayton;
    if (name == "Frank") return CT::Frank;
    if (name == "Gumbel") return CT::Gumbel;
    if (name == "Joe") return CT::Joe;
    if (name == "Normal") return CT::Normal;
    if (name == "StudentT") return CT::StudentT;
    throw std::runtime_error("unknown copula type: " + name);
}

// A `CopulaEstimationMethod` name -> the enum. Default InferenceFromMargins (the B1 default).
inline numerics::distributions::copulas::CopulaEstimationMethod parse_copula_estimation_method(
    const std::string& name) {
    using CEM = numerics::distributions::copulas::CopulaEstimationMethod;
    if (name == "FullLikelihood") return CEM::FullLikelihood;
    if (name == "PseudoLikelihood") return CEM::PseudoLikelihood;
    if (name == "InferenceFromMargins") return CEM::InferenceFromMargins;
    throw std::runtime_error("unknown copula estimation method: " + name);
}

// A bivariate marginal spec -> a pre-fit UnivariateDistributionModel (an IUnivariateModel). The
// marginal carries its own inline `data` array (its exact series -- the copula joint sample) and
// pinned distribution `parameter_values` (per B1 the marginals stay FIXED during the copula fit).
inline std::unique_ptr<UnivariateDistributionModel> build_bivariate_marginal(
    const JsonValue& spec) {
    auto dist = numerics::distributions::create_distribution(spec.at("family").as_string());
    auto m = std::make_unique<UnivariateDistributionModel>(std::move(dist),
                                                           spec.at("data").as_double_vector());
    if (spec.contains("parameter_values"))
        m->set_parameter_values(spec.at("parameter_values").as_double_vector());
    return m;
}

// BivariateDistribution holds its marginals by NON-owning pointer (mirroring the C# reference-
// typed marginals); this thin wrapper keeps the two marginal models alive for the model's
// lifetime so the returned unique_ptr<ModelBase> is self-contained.
class OwningBivariateDistribution : public BivariateDistribution {
   public:
    OwningBivariateDistribution() = default;
    OwningBivariateDistribution(OwningBivariateDistribution&&) noexcept = default;
    std::unique_ptr<UnivariateDistributionModel> owned_marginal_x;
    std::unique_ptr<UnivariateDistributionModel> owned_marginal_y;
};

// `type: "bivariate"` -- a copula-coupled BivariateDistribution. `copula` names the CopulaType,
// `marginal_x`/`marginal_y` are pre-fit IUnivariateModel marginals (held FIXED), and
// `estimation_method` selects the CopulaEstimationMethod (default InferenceFromMargins).
inline std::unique_ptr<ModelBase> build_bivariate_model(const JsonValue& model,
                                                        const std::vector<double>& /*dataset*/) {
    auto m = std::make_unique<OwningBivariateDistribution>();
    m->owned_marginal_x = build_bivariate_marginal(model.at("marginal_x"));
    m->owned_marginal_y = build_bivariate_marginal(model.at("marginal_y"));
    m->set_marginal_x(m->owned_marginal_x.get());
    m->set_marginal_y(m->owned_marginal_y.get());
    m->set_copula_type(parse_copula_type(model.value_or("copula", "Normal")));
    m->set_copula_estimation_method(model.contains("estimation_method")
                                        ? parse_copula_estimation_method(
                                              model.at("estimation_method").as_string())
                                        : numerics::distributions::copulas::
                                              CopulaEstimationMethod::InferenceFromMargins);
    apply_parameter_values(*m, model);
    return m;
}

// The `construct.model` dispatch: `type` defaults to the Phase 4 behavior.
inline std::unique_ptr<ModelBase> build_model(const JsonValue& model,
                                              const std::vector<double>& dataset) {
    std::string type = model.value_or("type", "univariate_distribution");
    if (type == "univariate_distribution") return build_univariate_distribution_model(model, dataset);
    if (type == "mixture") return build_mixture_model(model, dataset);
    if (type == "competing_risks") return build_competing_risks_model(model, dataset);
    if (type == "point_process") return build_point_process_model(model, dataset);
    if (type == "time_series") return build_time_series_model(model, dataset);
    if (type == "spatial_gev") return build_spatial_gev_model(model, dataset);
    if (type == "rating_curve") return build_rating_curve_model(model, dataset);
    if (type == "bivariate") return build_bivariate_model(model, dataset);
    throw std::runtime_error("unknown model_estimation model type: " + type);
}

// The single shared entry point all three harnesses call (see the file header).
inline std::unique_ptr<ModelBase> build_model_from_json(const std::string& model_json,
                                                        const std::vector<double>& dataset) {
    return build_model(parse_json(model_json), dataset);
}

// A `type: "bulletin17c"` spec -> a concrete Bulletin17CDistribution (B11).
//
// WIRING DECISION (deviation from the brief -- documented): the brief assumed
// Bulletin17CDistribution "derives from ModelBase" and could ride the build_model_from_json ->
// ModelBase path. It does NOT: it derives from IGMMModel + ISimulatable + IUnivariateModel, and
// NONE of those derive from ModelBase (IUnivariateModel is a deliberately minimal capability
// mixin, i_univariate_model.hpp). So a Bulletin17CDistribution* is not convertible to a
// ModelBase*, and this needs its OWN construction entry point returning the CONCRETE type. That
// concrete pointer is exactly what both consumers want anyway: the GMM glue passes it straight
// as the IGMMModel& the estimator ctor takes (no cross-cast), and the seeded-draw glue calls
// its ISimulatable generate_random_values directly.
//
// The spec carries the distribution `family` (any of the six IsSupportedDistributionType
// families -- Exponential, Gamma, LogNormal, LogPearsonTypeIII (default), Normal,
// PearsonTypeIII; an unsupported type throws from the B17C ctor), the Phase 5 DataFrame block or
// a flat `dataset` (via build_model_data_frame, carrying the same low_outlier_threshold /
// mgbt_low_outliers keys), and optional explicit `parameter_values` (applied last, exactly like
// every other model type).
inline std::unique_ptr<Bulletin17CDistribution> build_bulletin17c_model(
    const JsonValue& model, const std::vector<double>& dataset) {
    DataFrame df = build_model_data_frame(model, dataset);
    numerics::distributions::UnivariateDistributionType type =
        model.contains("family")
            ? numerics::distributions::create_distribution(model.at("family").as_string())->type()
            : numerics::distributions::UnivariateDistributionType::LogPearsonTypeIII;
    auto m = std::make_unique<Bulletin17CDistribution>(std::move(df), type);
    // Bulletin17CDistribution is an IGMMModel, not a ModelBase (see the wiring note above), so
    // it gets the parameter-vector overload directly; it has no UseDefaultFlatPriors property in
    // C# either, so that spec key is silently inapplicable here.
    apply_parameter_overrides(m->parameters(), model);
    if (model.contains("parameter_values"))
        m->set_parameter_values(model.at("parameter_values").as_double_vector());
    return m;
}

inline std::unique_ptr<Bulletin17CDistribution> build_bulletin17c_from_json(
    const std::string& model_json, const std::vector<double>& dataset) {
    return build_bulletin17c_model(parse_json(model_json), dataset);
}

}  // namespace corehydro::models::spec
