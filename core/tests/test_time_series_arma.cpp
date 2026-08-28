// T1 support ctest (C++-only): the AutoRegressive (AR(p)) and MovingAverage (MA(q)) TimeSeries
// leaf models plus the shared Transform enum. AR/MA are ModelBase families; their full-fit
// likelihood / MLE / MAP / posterior oracles come from the P4 dotnet emitter, NOT T1, so this
// file transcribes only the STRUCTURAL / DETERMINISM / TRIVIALLY-ANALYTIC assertions (internal
// support test territory -- hardcoded oracles are correct here; public-API oracle values stay
// in fixtures/).
//
// Structural oracles transcribed (values unaltered where they exist) from
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/TimeSeriesModels/AutoRegressiveTests.cs @ fc28c0c
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/TimeSeriesModels/MovingAverageTests.cs   @ fc28c0c
//
// The C# fixtures draw their data from System.Random (a .NET LCG, not the ported Mersenne
// Twister), so the exact data VALUES are unreproducible in C++. This file regenerates
// structurally-equivalent data (same AR(1)/MA(1) recursion shape, ported bit-exact
// MersenneTwister, positive mean ~500 so the log/Box-Cox transforms are well defined), then
// applies the same STRUCTURAL assertions (parameter counts/names/bounds, finiteness, sentinels,
// stationarity/invertibility predicates, seeded-stream determinism, validation outcomes). No
// data-value oracle is claimed for the generated fixtures.
//
// Skipped C# test methods (see task-T1-report.md for the full list + reasons):
//   - XML tests (Test_Constructor_XElement_RestoresModel, Test_ToXElement_*,
//     Test_RoundTrip_PreservesAllProperties, *_XmlSerialization): XML is a project-wide non-port.
//   - Clone tests (Test_Clone_*): the C++ core has no virtual IModel::Clone (see model_base.hpp);
//     no fit path in T1 needs a clone, so clone() is omitted (S4 precedent).
//   - Test_SetParameterValues_NullParameters_ThrowsException: VACUOUS (const-ref vector; no null).
//   - Test_GenerateRandomSeries_* assert on the heavy TimeSeries container return; the REQUIRED
//     simulation entry point is generate_random_values (ISimulatable), tested here for length +
//     seeded determinism. GenerateRandomSeries itself is deferred (see header note in the models).
//   - Test_Predict_ConvergesToMean / Test_SetDefaultParameters_InterceptInitializedToMean assert
//     analytic (mean-recursion) values, kept here as determinism/closed-form checks -- NOT MLE
//     oracles. AreSame identity checks are adapted to structural (count/value) equality since the
//     ported TimeSeries is a value type.
//
// P4 landed (see test_ar_ma_p4_fixed_param_oracles below + fixtures/estimation/*):
//   - exact AR/MA DataLogLikelihood + Residuals at fixed parameters, dumped from the real
//     RMC.BestFit via the dotnet emitter and asserted here (route b, C++-only, 1e-9 abs);
//   - the exact MLE fit (parameters, log-likelihood, AIC) and the seeded cross-language
//     GenerateRandomValues draw are oracle-verified against the real C# in
//     time_series_ar_smoke.json / time_series_ma_smoke.json / time_series_ar_sim.json.
// Still deferred (severable follow-up, lower value): exact Residuals under the Log/BoxCox/
//   YeoJohnson transforms and exact multi-step Predict() forecast values -- both deterministic
//   transforms of the now-verified None-transform fit + closed-form model equations.
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/time_series/auto_regressive.hpp"
#include "corehydro/models/time_series/moving_average.hpp"
#include "corehydro/models/time_series/transform_type.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "check.hpp"

namespace {
// P6 index swap: the TimeSeries container indexes by DateTime rather than by an integer offset.
// Nothing in this suite reads the index for meaning (the ARMA families index by POSITION, the
// rating curve by date EQUALITY), so every series here is anchored at one arbitrary date; the
// mapping is monotone, so every oracle below is unchanged by the swap.
const corehydro::numerics::data::DateTime kT0(1900, 1, 1);
// The (interval, start, end) constructor takes an END DATE since the P6 index swap; these suites
// expressed the span as a step count, so this walks the interval that many times.
inline corehydro::numerics::data::DateTime t0_plus(int steps,
                                                   corehydro::numerics::data::TimeInterval ti) {
    corehydro::numerics::data::DateTime d = kT0;
    for (int i = 0; i < steps; ++i)
        d = corehydro::numerics::data::TimeSeries::add_time_interval(d, ti);
    return d;
}



using corehydro::models::AutoRegressive;
using corehydro::models::MovingAverage;
using corehydro::models::Transform;
using corehydro::numerics::data::TimeInterval;
using corehydro::numerics::data::TimeSeries;
using corehydro::numerics::sampling::MersenneTwister;

// Compile-time mirror of Model_InheritsFromModelBase / Model_ImplementsISimulatable.
static_assert(std::is_base_of<corehydro::models::ModelBase, AutoRegressive>::value,
              "AutoRegressive must derive from ModelBase");
static_assert(
    std::is_base_of<corehydro::models::ISimulatable<std::vector<double>>, AutoRegressive>::value,
    "AutoRegressive must implement ISimulatable<std::vector<double>>");
static_assert(std::is_base_of<corehydro::models::ModelBase, MovingAverage>::value,
              "MovingAverage must derive from ModelBase");
static_assert(
    std::is_base_of<corehydro::models::ISimulatable<std::vector<double>>, MovingAverage>::value,
    "MovingAverage must implement ISimulatable<std::vector<double>>");

// ---- Test-data helpers (mirror the C# private fixtures; data regenerated with the ported RNG) ----

// 50 annual observations, AR(1)-like, positive mean ~500 (mirrors CreateSampleTimeSeries shape).
TimeSeries make_sample_series() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(49, TimeInterval::OneYear));
    MersenneTwister rng(12345);
    const double mean = 500.0, phi = 0.7, sigma = 50.0;
    double prev = mean;
    for (int i = 0; i < ts.count(); ++i) {
        double innovation = rng.next_double() * 2.0 - 1.0;
        double value = mean + phi * (prev - mean) + sigma * innovation;
        ts[i].set_value(value);
        prev = value;
    }
    return ts;
}

// 15 annual observations, deterministic 100 + 5i (mirrors CreateShortTimeSeries).
TimeSeries make_short_series() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(14, TimeInterval::OneYear));
    for (int i = 0; i < ts.count(); ++i) ts[i].set_value(100.0 + i * 5.0);
    return ts;
}

// 50 annual observations, linear trend 100 + 2i (mirrors CreateTrendTimeSeries).
TimeSeries make_trend_series() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(49, TimeInterval::OneYear));
    for (int i = 0; i < ts.count(); ++i) ts[i].set_value(100.0 + i * 2.0);
    return ts;
}

// 50 annual observations, zero-mean-ish uniform in [-50, 50] (mirrors NoInterceptWithZeroMean).
TimeSeries make_zero_mean_series() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(49, TimeInterval::OneYear));
    MersenneTwister rng(12345);
    for (int i = 0; i < ts.count(); ++i) ts[i].set_value(rng.next_double() * 100.0 - 50.0);
    return ts;
}

// 5 annual observations -> too short for the 10-observation validation guard.
TimeSeries make_tiny_series() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(4, TimeInterval::OneYear));
    for (int i = 0; i < ts.count(); ++i) ts[i].set_value(i);
    return ts;
}

bool messages_contain(const corehydro::models::ValidationResult& r, const std::string& needle) {
    for (const auto& m : r.validation_messages)
        if (m.find(needle) != std::string::npos) return true;
    return false;
}

// 60 annual observations (1960-2019 by index) that make the Box-Cox lambda objective
// non-finite: a zero at index 0 fails can_fit_lambda's positivity requirement (mirrors C#'s
// CreateBoxCoxLambdaFailureTimeSeries).
TimeSeries make_box_cox_lambda_failure_series() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(59, TimeInterval::OneYear));
    for (int i = 0; i < ts.count(); ++i) ts[i].set_value(i == 0 ? 0.0 : 10.0);
    return ts;
}

// 60 annual observations, all -DBL_MAX: a degenerate constant sample fails can_fit_lambda's
// non-degeneracy requirement for Yeo-Johnson (mirrors C#'s
// CreateYeoJohnsonLambdaFailureTimeSeries).
TimeSeries make_yeo_johnson_lambda_failure_series() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(59, TimeInterval::OneYear));
    for (int i = 0; i < ts.count(); ++i) ts[i].set_value(-std::numeric_limits<double>::max());
    return ts;
}

// ============================ AutoRegressive ============================

void test_ar_constructors() {
    AutoRegressive empty;
    CHECK_EQ(empty.order(), 1);
    CHECK_TRUE(empty.include_intercept());

    TimeSeries ts = make_sample_series();
    AutoRegressive m1(ts);
    CHECK_EQ(m1.time_series().count(), ts.count());  // adapted AreSame -> structural equality
    CHECK_EQ(m1.order(), 1);

    AutoRegressive m3(ts, 3);
    CHECK_EQ(m3.order(), 3);

    AutoRegressive mni(ts, 1, false);
    CHECK_TRUE(!mni.include_intercept());
}

void test_ar_properties() {
    TimeSeries ts = make_sample_series();

    AutoRegressive m;
    m.set_time_series(ts);
    CHECK_EQ(m.time_series().count(), ts.count());

    AutoRegressive mo(ts);
    mo.set_order(4);
    CHECK_EQ(mo.order(), 4);

    AutoRegressive mc(ts, 1);
    int initial = mc.number_of_parameters();
    mc.set_order(3);
    CHECK_EQ(mc.number_of_parameters(), initial + 2);

    AutoRegressive mi(ts);
    mi.set_include_intercept(false);
    CHECK_TRUE(!mi.include_intercept());

    AutoRegressive mic(ts, 1, true);
    int initial2 = mic.number_of_parameters();
    mic.set_include_intercept(false);
    CHECK_EQ(mic.number_of_parameters(), initial2 - 1);
}

void test_ar_set_default_parameters() {
    TimeSeries ts = make_sample_series();

    AutoRegressive ar1_ni(ts, 1, false);
    CHECK_EQ(ar1_ni.number_of_parameters(), 2);  // phi + sigma

    AutoRegressive ar1(ts, 1, true);
    CHECK_EQ(ar1.number_of_parameters(), 3);  // mu + phi + sigma

    AutoRegressive ar3(ts, 3, true);
    CHECK_EQ(ar3.number_of_parameters(), 5);  // mu + phi1..3 + sigma

    AutoRegressive ar2(ts, 2, true);
    CHECK_TRUE(ar2.parameters()[0].name().find("Intercept") != std::string::npos);
    CHECK_TRUE(ar2.parameters()[1].name().find("AR") != std::string::npos);
    CHECK_TRUE(ar2.parameters()[2].name().find("AR") != std::string::npos);
    CHECK_TRUE(ar2.parameters()[3].name().find("Scale") != std::string::npos);

    // AR coefficient bounds [-2, 2]; the phi is at index 1 (after intercept).
    CHECK_NEAR(ar1.parameters()[1].lower_bound(), -2.0, 0.0);
    CHECK_NEAR(ar1.parameters()[1].upper_bound(), 2.0, 0.0);

    // Scale is positive with a strictly positive lower bound (last parameter).
    const auto& scale = ar1.parameters().back();
    CHECK_TRUE(scale.is_positive());
    CHECK_TRUE(scale.lower_bound() > 0.0);

    // Intercept initialized to the training mean (full-series training).
    AutoRegressive arm(ts);
    arm.set_use_default_training_steps(false);
    arm.set_training_time_steps(ts.count());
    CHECK_NEAR(arm.parameters()[0].value(), ts.mean_value(), 1e-6);
}

void test_ar_log_likelihood() {
    TimeSeries ts = make_sample_series();

    AutoRegressive ar1(ts, 1);
    auto p1 = ar1.parameter_values();
    double d1 = ar1.data_log_likelihood(p1);
    CHECK_TRUE(!std::isnan(d1));
    CHECK_TRUE(!std::isinf(d1) || d1 < 0.0);  // not +inf

    AutoRegressive ar2(ts, 2);
    auto p2 = ar2.parameter_values();
    CHECK_TRUE(!std::isnan(ar2.data_log_likelihood(p2)));

    AutoRegressive ar3(ts, 3);
    auto p3 = ar3.parameter_values();
    CHECK_TRUE(!std::isnan(ar3.data_log_likelihood(p3)));

    // Null time series -> -inf sentinel.
    AutoRegressive nullar;
    std::vector<double> pnull = {0, 0, 1};
    CHECK_TRUE(std::isinf(nullar.data_log_likelihood(pnull)) &&
               nullar.data_log_likelihood(pnull) < 0.0);

    // NaN parameter -> -inf sentinel.
    std::vector<double> pnan = {std::numeric_limits<double>::quiet_NaN(), 0, 1};
    double rnan = ar1.data_log_likelihood(pnan);
    CHECK_TRUE(std::isinf(rnan) && rnan < 0.0);

    // Prior finite.
    double pr = ar1.prior_log_likelihood(p1);
    CHECK_TRUE(!std::isnan(pr));
    CHECK_TRUE(!(std::isinf(pr) && pr > 0.0));

    // Pointwise count = TrainingTimeSteps - Order (full-series training).
    AutoRegressive arp(ts, 2);
    arp.set_use_default_training_steps(false);
    arp.set_training_time_steps(ts.count());
    auto pp = arp.parameter_values();
    auto pw = arp.pointwise_data_log_likelihood(pp);
    CHECK_EQ(static_cast<int>(pw.size()), ts.count() - 2);

    // Sum of pointwise == total data log-likelihood.
    AutoRegressive ars(ts, 1);
    ars.set_use_default_training_steps(false);
    ars.set_training_time_steps(ts.count());
    auto ps = ars.parameter_values();
    auto pws = ars.pointwise_data_log_likelihood(ps);
    double sum = 0.0;
    for (double v : pws) sum += v;
    CHECK_NEAR(sum, ars.data_log_likelihood(ps), 1e-6);

    // Components all Exact.
    auto comps = ar1.pointwise_data_log_likelihood_components(p1);
    for (const auto& c : comps)
        CHECK_TRUE(c.type() == corehydro::models::DataComponentType::Exact);
}

void test_ar_predict() {
    TimeSeries ts = make_sample_series();

    AutoRegressive m(ts);
    m.set_use_default_training_steps(false);
    m.set_training_time_steps(ts.count());

    auto p0 = m.predict(0);
    CHECK_EQ(static_cast<int>(p0.size()), ts.count());

    auto p10 = m.predict(10);
    CHECK_EQ(static_cast<int>(p10.size()), ts.count() + 10);

    // Deterministic (seed -1) reproducible.
    auto a = m.predict(5, -1);
    auto b = m.predict(5, -1);
    for (std::size_t i = 0; i < a.size(); ++i) CHECK_NEAR(a[i], b[i], 1e-10);

    // Seeded reproducible.
    auto s1 = m.predict(5, 12345);
    auto s2 = m.predict(5, 12345);
    for (std::size_t i = 0; i < s1.size(); ++i) CHECK_NEAR(s1[i], s2[i], 1e-10);

    // Different seeds differ somewhere in the forecast tail.
    auto d1 = m.predict(5, 12345);
    auto d2 = m.predict(5, 54321);
    bool any_diff = false;
    for (int i = m.training_time_steps(); i < static_cast<int>(d1.size()); ++i)
        if (std::fabs(d1[i] - d2[i]) > 1e-10) any_diff = true;
    CHECK_TRUE(any_diff);

    // Null time series -> throws.
    AutoRegressive nullm;
    CHECK_THROWS(nullm.predict());

    // Converges to mean for a stationary coefficient.
    AutoRegressive cm(ts, 1);
    cm.set_use_default_training_steps(false);
    cm.set_training_time_steps(ts.count());
    cm.parameters()[1].set_value(0.5);
    double mean = cm.parameters()[0].value();
    auto conv = cm.predict(100, -1);
    CHECK_NEAR(conv.back(), mean, std::fabs(mean) * 0.1);
}

void test_ar_generate_random_values() {
    TimeSeries ts = make_sample_series();
    AutoRegressive m(ts);

    auto g = m.generate_random_values(100, 12345);
    CHECK_EQ(static_cast<int>(g.size()), 100);

    auto g1 = m.generate_random_values(50, 12345);
    auto g2 = m.generate_random_values(50, 12345);
    for (std::size_t i = 0; i < g1.size(); ++i) CHECK_NEAR(g1[i], g2[i], 1e-10);

    // sampleSize <= 0 throws.
    CHECK_THROWS(m.generate_random_values(0, 12345));
}

void test_ar_stationarity() {
    TimeSeries ts = make_sample_series();

    AutoRegressive ar1(ts, 1);
    ar1.parameters()[1].set_value(0.5);
    CHECK_TRUE(ar1.is_stationary());
    ar1.parameters()[1].set_value(1.1);
    CHECK_TRUE(!ar1.is_stationary());
    ar1.parameters()[1].set_value(1.0);
    CHECK_TRUE(!ar1.is_stationary());  // unit root

    AutoRegressive ar2(ts, 2);
    ar2.parameters()[1].set_value(0.3);
    ar2.parameters()[2].set_value(0.2);
    CHECK_TRUE(ar2.is_stationary());  // phi1+phi2<1, phi2-phi1<1, |phi2|<1
    ar2.parameters()[1].set_value(0.8);
    ar2.parameters()[2].set_value(0.5);
    CHECK_TRUE(!ar2.is_stationary());  // phi1+phi2 = 1.3 > 1

    AutoRegressive ar3(ts, 3);
    ar3.parameters()[1].set_value(0.2);
    ar3.parameters()[2].set_value(0.2);
    ar3.parameters()[3].set_value(0.2);
    CHECK_TRUE(ar3.is_stationary());  // sum|phi| = 0.6 < 1
}

void test_ar_validate() {
    TimeSeries ts = make_sample_series();

    AutoRegressive valid(ts);
    CHECK_TRUE(valid.validate().is_valid);

    AutoRegressive nullm;
    auto rn = nullm.validate();
    CHECK_TRUE(!rn.is_valid);
    CHECK_TRUE(messages_contain(rn, "Time series"));

    TimeSeries tiny = make_tiny_series();
    AutoRegressive shortm(tiny);
    auto rs = shortm.validate();
    CHECK_TRUE(!rs.is_valid);
    CHECK_TRUE(messages_contain(rs, "10 observations"));

    AutoRegressive bo(ts);
    bo.set_order(0);
    CHECK_TRUE(!bo.validate().is_valid);

    AutoRegressive ho(ts);
    ho.set_order(15);
    CHECK_TRUE(!ho.validate().is_valid);

    TimeSeries shortts = make_short_series();
    AutoRegressive ol(shortts);
    ol.set_order(shortts.count() + 1);
    CHECK_TRUE(!ol.validate().is_valid);

    // Non-stationary parameters produce a warning mentioning "stationar".
    AutoRegressive ns(ts, 1);
    ns.parameters()[1].set_value(1.5);
    auto rns = ns.validate();
    CHECK_TRUE(messages_contain(rns, "stationar"));
}

// Transcribes AutoRegressiveTests.cs's Test_{BoxCoxTransform,YeoJohnsonTransform}Transform_
// LambdaFitFailure_ReturnsValidationError (BestFit v2.0.0, f140c4d): a degenerate series that
// makes FitLambda return a non-finite value must be captured as a validation error message
// (not a crash or a silently-NaN-propagating fit).
void test_ar_transform_lambda_failure() {
    AutoRegressive bc(make_box_cox_lambda_failure_series(), 1);
    bc.set_transform_type(Transform::BoxCox);
    auto rbc = bc.validate();
    CHECK_TRUE(!rbc.is_valid);
    CHECK_TRUE(messages_contain(rbc, "Box-Cox lambda estimation failed"));

    AutoRegressive yj(make_yeo_johnson_lambda_failure_series(), 1);
    yj.set_transform_type(Transform::YeoJohnson);
    auto ryj = yj.validate();
    CHECK_TRUE(!ryj.is_valid);
    CHECK_TRUE(messages_contain(ryj, "Yeo-Johnson lambda estimation failed"));
}

void test_ar_set_parameter_values() {
    TimeSeries ts = make_sample_series();
    AutoRegressive m(ts, 1);
    std::vector<double> nv = {500.0, 0.6, 40.0};
    m.set_parameter_values(nv);
    CHECK_NEAR(m.parameters()[0].value(), 500.0, 0.0);
    CHECK_NEAR(m.parameters()[1].value(), 0.6, 0.0);
    CHECK_NEAR(m.parameters()[2].value(), 40.0, 0.0);

    CHECK_THROWS(m.set_parameter_values(std::vector<double>{1.0}));
}

void test_ar_engineering_and_edges() {
    // AnnualStreamflow: valid + finite data log-likelihood.
    TimeSeries ts = make_sample_series();
    AutoRegressive as(ts, 1);
    CHECK_TRUE(as.validate().is_valid);
    auto pas = as.parameter_values();
    double das = as.data_log_likelihood(pas);
    CHECK_TRUE(!(std::isinf(das) && das < 0.0));

    AutoRegressive hp(ts, 2);
    CHECK_TRUE(hp.validate().is_valid);

    TimeSeries trend = make_trend_series();
    AutoRegressive td(trend, 1);
    CHECK_TRUE(td.validate().is_valid);

    // Minimum valid time series (15 obs, full training).
    TimeSeries shortts = make_short_series();
    AutoRegressive mv(shortts, 1);
    mv.set_use_default_training_steps(false);
    mv.set_training_time_steps(shortts.count());
    CHECK_TRUE(mv.validate().is_valid);

    AutoRegressive mo(ts, 10);
    CHECK_TRUE(mo.validate().is_valid);

    TimeSeries zm = make_zero_mean_series();
    AutoRegressive zi(zm, 1, false);
    CHECK_TRUE(zi.validate().is_valid);
}

void test_ar_transform_training_jeffreys() {
    TimeSeries ts = make_sample_series();

    AutoRegressive tt(ts);
    tt.set_transform_type(Transform::Logarithmic);
    CHECK_TRUE(tt.transform_type() == Transform::Logarithmic);

    AutoRegressive lt(ts, 1);
    lt.set_transform_type(Transform::Logarithmic);
    CHECK_TRUE(lt.validate().is_valid);

    // TrainingTimeSteps get/set + ForecastingTimeSteps derived.
    AutoRegressive tr(ts);
    tr.set_use_default_training_steps(false);
    tr.set_training_time_steps(35);
    CHECK_EQ(tr.training_time_steps(), 35);

    AutoRegressive fc(ts);
    fc.set_use_default_training_steps(false);
    fc.set_training_time_steps(40);
    CHECK_EQ(fc.forecasting_time_steps(), ts.count() - 40);

    AutoRegressive ud(ts);
    ud.set_use_default_training_steps(false);
    CHECK_TRUE(!ud.use_default_training_steps());

    // Jeffreys rule toggle + finite prior both ways.
    AutoRegressive jr(ts, 1);
    jr.set_use_jeffreys_rule_for_scale(false);
    CHECK_TRUE(!jr.use_jeffreys_rule_for_scale());

    AutoRegressive j1(ts, 1);
    j1.set_use_jeffreys_rule_for_scale(true);
    auto pj1 = j1.parameter_values();
    double prj1 = j1.prior_log_likelihood(pj1);
    CHECK_TRUE(!std::isnan(prj1) && !(std::isinf(prj1) && prj1 > 0.0));

    AutoRegressive j0(ts, 1);
    j0.set_use_jeffreys_rule_for_scale(false);
    auto pj0 = j0.parameter_values();
    double prj0 = j0.prior_log_likelihood(pj0);
    CHECK_TRUE(!std::isnan(prj0) && !(std::isinf(prj0) && prj0 > 0.0));
}

// ============================ MovingAverage ============================

void test_ma_constructors() {
    MovingAverage empty;
    CHECK_EQ(empty.order(), 1);
    CHECK_TRUE(empty.include_intercept());

    TimeSeries ts = make_sample_series();
    MovingAverage m1(ts);
    CHECK_EQ(m1.time_series().count(), ts.count());
    CHECK_EQ(m1.order(), 1);

    MovingAverage m3(ts, 3);
    CHECK_EQ(m3.order(), 3);

    MovingAverage mni(ts, 1, false);
    CHECK_TRUE(!mni.include_intercept());
}

void test_ma_properties() {
    TimeSeries ts = make_sample_series();

    MovingAverage mo(ts);
    mo.set_order(4);
    CHECK_EQ(mo.order(), 4);

    MovingAverage mc(ts, 1);
    int initial = mc.number_of_parameters();
    mc.set_order(3);
    CHECK_EQ(mc.number_of_parameters(), initial + 2);

    MovingAverage mic(ts, 1, true);
    int initial2 = mic.number_of_parameters();
    mic.set_include_intercept(false);
    CHECK_EQ(mic.number_of_parameters(), initial2 - 1);
}

void test_ma_set_default_parameters() {
    TimeSeries ts = make_sample_series();

    MovingAverage ma1_ni(ts, 1, false);
    CHECK_EQ(ma1_ni.number_of_parameters(), 2);

    MovingAverage ma1(ts, 1, true);
    CHECK_EQ(ma1.number_of_parameters(), 3);

    MovingAverage ma3(ts, 3, true);
    CHECK_EQ(ma3.number_of_parameters(), 5);

    MovingAverage ma2(ts, 2, true);
    CHECK_TRUE(ma2.parameters()[0].name().find("Intercept") != std::string::npos);
    CHECK_TRUE(ma2.parameters()[1].name().find("MA") != std::string::npos);
    CHECK_TRUE(ma2.parameters()[2].name().find("MA") != std::string::npos);
    CHECK_TRUE(ma2.parameters()[3].name().find("Scale") != std::string::npos);

    CHECK_NEAR(ma1.parameters()[1].lower_bound(), -2.0, 0.0);
    CHECK_NEAR(ma1.parameters()[1].upper_bound(), 2.0, 0.0);

    const auto& scale = ma1.parameters().back();
    CHECK_TRUE(scale.is_positive());
    CHECK_TRUE(scale.lower_bound() > 0.0);

    MovingAverage mam(ts);
    mam.set_use_default_training_steps(false);
    mam.set_training_time_steps(ts.count());
    CHECK_NEAR(mam.parameters()[0].value(), ts.mean_value(), 1e-6);
}

void test_ma_log_likelihood() {
    TimeSeries ts = make_sample_series();

    MovingAverage ma1(ts, 1);
    auto p1 = ma1.parameter_values();
    double d1 = ma1.data_log_likelihood(p1);
    CHECK_TRUE(!std::isnan(d1));
    CHECK_TRUE(!(std::isinf(d1) && d1 > 0.0));

    MovingAverage ma2(ts, 2);
    auto p2 = ma2.parameter_values();
    CHECK_TRUE(!std::isnan(ma2.data_log_likelihood(p2)));

    MovingAverage nullm;
    std::vector<double> pnull = {0, 0, 1};
    double rn = nullm.data_log_likelihood(pnull);
    CHECK_TRUE(std::isinf(rn) && rn < 0.0);

    std::vector<double> pnan = {std::numeric_limits<double>::quiet_NaN(), 0, 1};
    double rnan = ma1.data_log_likelihood(pnan);
    CHECK_TRUE(std::isinf(rnan) && rnan < 0.0);

    double pr = ma1.prior_log_likelihood(p1);
    CHECK_TRUE(!std::isnan(pr) && !(std::isinf(pr) && pr > 0.0));

    // MA pointwise count == TrainingTimeSteps (ALL observations; differs from AR).
    MovingAverage map(ts, 2);
    map.set_use_default_training_steps(false);
    map.set_training_time_steps(ts.count());
    auto pp = map.parameter_values();
    auto pw = map.pointwise_data_log_likelihood(pp);
    CHECK_EQ(static_cast<int>(pw.size()), ts.count());

    MovingAverage mas(ts, 1);
    mas.set_use_default_training_steps(false);
    mas.set_training_time_steps(ts.count());
    auto ps = mas.parameter_values();
    auto pws = mas.pointwise_data_log_likelihood(ps);
    double sum = 0.0;
    for (double v : pws) sum += v;
    CHECK_NEAR(sum, mas.data_log_likelihood(ps), 1e-6);

    auto comps = ma1.pointwise_data_log_likelihood_components(p1);
    for (const auto& c : comps)
        CHECK_TRUE(c.type() == corehydro::models::DataComponentType::Exact);
}

void test_ma_predict() {
    TimeSeries ts = make_sample_series();

    MovingAverage m(ts);
    m.set_use_default_training_steps(false);
    m.set_training_time_steps(ts.count());

    CHECK_EQ(static_cast<int>(m.predict(0).size()), ts.count());
    CHECK_EQ(static_cast<int>(m.predict(10).size()), ts.count() + 10);

    auto a = m.predict(5, -1);
    auto b = m.predict(5, -1);
    for (std::size_t i = 0; i < a.size(); ++i) CHECK_NEAR(a[i], b[i], 1e-10);

    auto s1 = m.predict(5, 12345);
    auto s2 = m.predict(5, 12345);
    for (std::size_t i = 0; i < s1.size(); ++i) CHECK_NEAR(s1[i], s2[i], 1e-10);

    auto d1 = m.predict(5, 12345);
    auto d2 = m.predict(5, 54321);
    bool any_diff = false;
    for (int i = m.training_time_steps(); i < static_cast<int>(d1.size()); ++i)
        if (std::fabs(d1[i] - d2[i]) > 1e-10) any_diff = true;
    CHECK_TRUE(any_diff);

    MovingAverage nullm;
    CHECK_THROWS(nullm.predict());

    // MA forecast decays to the intercept after q steps (memory exhausted).
    MovingAverage cm(ts, 2);
    cm.set_use_default_training_steps(false);
    cm.set_training_time_steps(ts.count());
    cm.parameters()[0].set_value(500.0);
    cm.parameters()[1].set_value(0.3);
    cm.parameters()[2].set_value(0.2);
    cm.parameters()[3].set_value(50.0);
    auto conv = cm.predict(10, -1);
    for (int i = cm.training_time_steps() + 2; i < static_cast<int>(conv.size()); ++i)
        CHECK_NEAR(conv[i], 500.0, 0.1);
}

void test_ma_generate_random_values() {
    TimeSeries ts = make_sample_series();
    MovingAverage m(ts);

    CHECK_EQ(static_cast<int>(m.generate_random_values(100, 12345).size()), 100);

    auto g1 = m.generate_random_values(50, 12345);
    auto g2 = m.generate_random_values(50, 12345);
    for (std::size_t i = 0; i < g1.size(); ++i) CHECK_NEAR(g1[i], g2[i], 1e-10);

    CHECK_THROWS(m.generate_random_values(0, 12345));
}

void test_ma_invertibility() {
    TimeSeries ts = make_sample_series();

    MovingAverage ma1(ts, 1);
    ma1.parameters()[1].set_value(0.5);
    CHECK_TRUE(ma1.is_invertible());
    ma1.parameters()[1].set_value(1.1);
    CHECK_TRUE(!ma1.is_invertible());
    ma1.parameters()[1].set_value(1.0);
    CHECK_TRUE(!ma1.is_invertible());

    MovingAverage ma2(ts, 2);
    ma2.parameters()[1].set_value(0.3);
    ma2.parameters()[2].set_value(0.2);
    CHECK_TRUE(ma2.is_invertible());
    ma2.parameters()[1].set_value(0.8);
    ma2.parameters()[2].set_value(0.5);
    CHECK_TRUE(!ma2.is_invertible());  // sum = 1.3 > 1
}

void test_ma_validate() {
    TimeSeries ts = make_sample_series();

    CHECK_TRUE(MovingAverage(ts).validate().is_valid);

    MovingAverage nullm;
    auto rn = nullm.validate();
    CHECK_TRUE(!rn.is_valid);
    CHECK_TRUE(messages_contain(rn, "Time series"));

    TimeSeries tiny = make_tiny_series();
    auto rs = MovingAverage(tiny).validate();
    CHECK_TRUE(!rs.is_valid);
    CHECK_TRUE(messages_contain(rs, "10 observations"));

    MovingAverage bo(ts);
    bo.set_order(0);
    CHECK_TRUE(!bo.validate().is_valid);

    MovingAverage ho(ts);
    ho.set_order(15);
    CHECK_TRUE(!ho.validate().is_valid);

    MovingAverage ns(ts, 1);
    ns.parameters()[1].set_value(1.5);
    CHECK_TRUE(messages_contain(ns.validate(), "invertibility"));
}

// Transcribes MovingAverageTests.cs's Test_{BoxCoxTransform,YeoJohnsonTransform}Transform_
// LambdaFitFailure_ReturnsValidationError (BestFit v2.0.0, f140c4d).
void test_ma_transform_lambda_failure() {
    MovingAverage bc(make_box_cox_lambda_failure_series(), 1);
    bc.set_transform_type(Transform::BoxCox);
    auto rbc = bc.validate();
    CHECK_TRUE(!rbc.is_valid);
    CHECK_TRUE(messages_contain(rbc, "Box-Cox lambda estimation failed"));

    MovingAverage yj(make_yeo_johnson_lambda_failure_series(), 1);
    yj.set_transform_type(Transform::YeoJohnson);
    auto ryj = yj.validate();
    CHECK_TRUE(!ryj.is_valid);
    CHECK_TRUE(messages_contain(ryj, "Yeo-Johnson lambda estimation failed"));
}

void test_ma_set_parameter_values() {
    TimeSeries ts = make_sample_series();
    MovingAverage m(ts, 1);
    std::vector<double> nv = {500.0, 0.4, 40.0};
    m.set_parameter_values(nv);
    CHECK_NEAR(m.parameters()[0].value(), 500.0, 0.0);
    CHECK_NEAR(m.parameters()[1].value(), 0.4, 0.0);
    CHECK_NEAR(m.parameters()[2].value(), 40.0, 0.0);

    CHECK_THROWS(m.set_parameter_values(std::vector<double>{1.0}));
}

void test_ma_engineering_and_edges() {
    TimeSeries ts = make_sample_series();

    MovingAverage me(ts, 1);
    CHECK_TRUE(me.validate().is_valid);
    auto pme = me.parameter_values();
    double dme = me.data_log_likelihood(pme);
    CHECK_TRUE(!(std::isinf(dme) && dme < 0.0));

    MovingAverage ho(ts, 5);
    CHECK_TRUE(ho.validate().is_valid);
    CHECK_EQ(ho.number_of_parameters(), 7);  // mu + theta1..5 + sigma

    TimeSeries shortts = make_short_series();
    MovingAverage mv(shortts, 1);
    mv.set_use_default_training_steps(false);
    mv.set_training_time_steps(shortts.count());
    CHECK_TRUE(mv.validate().is_valid);

    MovingAverage mo(ts, 10);
    CHECK_TRUE(mo.validate().is_valid);

    TimeSeries zm = make_zero_mean_series();
    MovingAverage zi(zm, 1, false);
    CHECK_TRUE(zi.validate().is_valid);
}

void test_ma_transform_training_jeffreys() {
    TimeSeries ts = make_sample_series();

    MovingAverage tt(ts);
    tt.set_transform_type(Transform::Logarithmic);
    CHECK_TRUE(tt.transform_type() == Transform::Logarithmic);

    MovingAverage lt(ts, 1);
    lt.set_transform_type(Transform::Logarithmic);
    CHECK_TRUE(lt.validate().is_valid);

    MovingAverage tr(ts);
    tr.set_use_default_training_steps(false);
    tr.set_training_time_steps(35);
    CHECK_EQ(tr.training_time_steps(), 35);

    MovingAverage fc(ts);
    fc.set_use_default_training_steps(false);
    fc.set_training_time_steps(40);
    CHECK_EQ(fc.forecasting_time_steps(), ts.count() - 40);

    MovingAverage jr(ts, 1);
    jr.set_use_jeffreys_rule_for_scale(false);
    CHECK_TRUE(!jr.use_jeffreys_rule_for_scale());

    MovingAverage j1(ts, 1);
    j1.set_use_jeffreys_rule_for_scale(true);
    auto pj1 = j1.parameter_values();
    double prj1 = j1.prior_log_likelihood(pj1);
    CHECK_TRUE(!std::isnan(prj1) && !(std::isinf(prj1) && prj1 > 0.0));

    MovingAverage j0(ts, 1);
    j0.set_use_jeffreys_rule_for_scale(false);
    auto pj0 = j0.parameter_values();
    double prj0 = j0.prior_log_likelihood(pj0);
    CHECK_TRUE(!std::isnan(prj0) && !(std::isinf(prj0) && prj0 > 0.0));
}

// v2.0.0 ResetDefaultTrainingStepsForNewTimeSeries (AutoRegressive.cs:347, MovingAverage.cs:347):
// attaching a DIFFERENT response series is a new calibration problem, so the setter discards the
// previous manual training-window edit and restores the default 80% split.
//
// PROVENANCE, stated exactly: the C# regression test
// TimeSeries_NewSeries_ResetsTrainingSplitToDefault exists ONLY in ARIMAXTests.cs (line 211);
// AutoRegressiveTests.cs, MovingAverageTests.cs, ARIMATests.cs and TimeSeriesModelTests.cs have
// no such test. The C# PRODUCTION method, by contrast, is present verbatim in all four model
// files, so this block is an EXTRAPOLATION of the ARIMAX test to the two sibling families whose
// production code is identical -- not a transcription of a test that exists for them. The
// expected values follow the same C# default-split formula the ARIMAX test asserts. The manual window (25)
// is deliberately NOT the default the replacement produces (floor(0.8*50) = 40), so the check
// fails on the pre-port code path (which kept both `use_default = false` and 25).
//
// The empty-series arm (C# zeroes TrainingTimeSteps when the new series is null/empty) is
// exercised too -- the C# `TimeSeries = null` assignment maps to attaching an EMPTY series here,
// since this port holds the series by value (std::optional), never by nullable reference.
void test_ar_ma_new_series_resets_training_split() {
    TimeSeries original = make_sample_series();    // 50 obs
    TimeSeries replacement = make_trend_series();  // 50 obs, different values
    TimeSeries empty(TimeInterval::OneYear);

    AutoRegressive ar(original, 1, true);
    ar.set_use_default_training_steps(false);
    ar.set_training_time_steps(25);
    CHECK_EQ(ar.training_time_steps(), 25);
    ar.set_time_series(replacement);
    CHECK_TRUE(ar.use_default_training_steps());
    CHECK_EQ(ar.training_time_steps(), 40);

    MovingAverage ma(original, 1, true);
    ma.set_use_default_training_steps(false);
    ma.set_training_time_steps(25);
    ma.set_time_series(replacement);
    CHECK_TRUE(ma.use_default_training_steps());
    CHECK_EQ(ma.training_time_steps(), 40);

    // Empty-series arm: attaching a series with no observations zeroes the window. Default
    // flat priors are turned OFF first only to keep the assertion on the reset itself -- the
    // trailing SetDefaultParameters call indexes the STALE differenced series at
    // TrainingTimeSteps-1 == -1, which is an IndexOutOfRangeException in C# and UB here (a
    // pre-existing shared hazard on a path the C# regression suite never drives).
    AutoRegressive are(original, 1, true);
    are.set_use_default_flat_priors(false);
    are.set_use_default_training_steps(false);
    are.set_training_time_steps(25);
    are.set_time_series(empty);
    CHECK_TRUE(are.use_default_training_steps());
    CHECK_EQ(are.training_time_steps(), 0);
}

// v2.0.0 TimeInterval.Irregular Validate guard (AutoRegressive.cs:995, MovingAverage.cs:985).
// Transcribed from the new Test_Validate_IrregularTimeSeries_ReturnsFalse regression tests. The
// cross-language oracle lives in fixtures/estimation/time_series_irregular_interval.json; this
// block is the C++ RED/GREEN discriminator for the two leaf families.
void test_ar_ma_irregular_interval_validate() {
    // C# `new TimeSeries(TimeInterval.Irregular)` + Add: the (interval, start, end) ctor
    // rejects Irregular in BOTH C# and this port, so build it ordinate-by-ordinate with the
    // C# test's own i*i+1 index spacing.
    TimeSeries irregular(TimeInterval::Irregular);
    for (int i = 0; i < 50; ++i)
        irregular.add(TimeSeries::Ordinate(kT0.add_days(static_cast<double>(i) * i + 1), i + 1.0));

    auto rar = AutoRegressive(irregular, 1, true).validate();
    CHECK_TRUE(!rar.is_valid);
    CHECK_TRUE(messages_contain(rar, "regular time interval"));

    auto rma = MovingAverage(irregular, 1, true).validate();
    CHECK_TRUE(!rma.is_valid);
    CHECK_TRUE(messages_contain(rma, "regular time interval"));

    // The same series on a REGULAR interval passes -- the guard is interval-specific.
    CHECK_TRUE(AutoRegressive(make_sample_series(), 1, true).validate().is_valid);
}

}  // namespace

// P4 oracles (route b): exact fixed-parameter DataLogLikelihood + Residuals dumped from the REAL
// RMC.BestFit AutoRegressive / MovingAverage via tools/oracle_emitter (Numerics @ a2c4dbf,
// RMC-BestFit @ fc28c0c) on the shared 20-point fixture series. Deterministic (no fit), so these
// are hardcoded C++-only oracles at 1e-9 abs (permitted for internal-support ctests). The exact
// MLE fit + seeded GenerateRandomValues draw are oracle-verified cross-language in
// fixtures/estimation/time_series_ar_smoke.json + time_series_ma_smoke.json + _ar_sim.json.
TimeSeries make_p4_fixture_series() {
    return TimeSeries(TimeInterval::OneDay, kT0,
                      std::vector<double>{10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0,
                                          13.1, 11.7, 12.5, 13.9, 15.2, 14.1, 12.9, 13.6, 15.0,
                                          16.2, 14.8});
}

void test_ar_ma_p4_fixed_param_oracles() {
    // AutoRegressive(1)+intercept, params [intercept, phi1, sigma] = [12.5, 0.4, 1.1].
    AutoRegressive ar(make_p4_fixture_series(), 1, true);
    ar.set_transform_type(Transform::None);
    std::vector<double> ar_p{12.5, 0.4, 1.1};
    CHECK_NEAR(ar.data_log_likelihood(ar_p), -33.621469348823844, 1e-9);
    std::vector<double> ar_res = ar.residuals(ar_p);
    CHECK_NEAR(ar_res[0], 0.0, 1e-9);
    CHECK_NEAR(ar_res[5], -0.9599999999999991, 1e-9);
    CHECK_NEAR(ar_res[18], 2.6999999999999993, 1e-9);

    // MovingAverage(1)+intercept, params [intercept, theta1, sigma] = [12.8, 0.3, 1.4].
    MovingAverage ma(make_p4_fixture_series(), 1, true);
    ma.set_transform_type(Transform::None);
    std::vector<double> ma_p{12.8, 0.3, 1.4};
    CHECK_NEAR(ma.data_log_likelihood(ma_p), -35.92910180608853, 1e-9);
    std::vector<double> ma_res = ma.residuals(ma_p);
    CHECK_NEAR(ma_res[0], -2.6000000000000014, 1e-9);
    CHECK_NEAR(ma_res[10], -1.0972291048400002, 1e-9);
}

int main() {
    test_ar_constructors();
    test_ar_properties();
    test_ar_set_default_parameters();
    test_ar_log_likelihood();
    test_ar_predict();
    test_ar_generate_random_values();
    test_ar_stationarity();
    test_ar_validate();
    test_ar_transform_lambda_failure();
    test_ar_set_parameter_values();
    test_ar_engineering_and_edges();
    test_ar_transform_training_jeffreys();

    test_ma_constructors();
    test_ma_properties();
    test_ma_set_default_parameters();
    test_ma_log_likelihood();
    test_ma_predict();
    test_ma_generate_random_values();
    test_ma_invertibility();
    test_ma_validate();
    test_ma_transform_lambda_failure();
    test_ma_set_parameter_values();
    test_ma_engineering_and_edges();
    test_ma_transform_training_jeffreys();

    test_ar_ma_new_series_resets_training_split();
    test_ar_ma_irregular_interval_validate();

    test_ar_ma_p4_fixed_param_oracles();

    return chtest::summary("time_series_arma");
}
