// T2 support ctest (C++-only): the ARIMA(p,d,q) TimeSeries model built on top of the T1 ARMA
// leaf models and the P2 TimeSeries adapter. ARIMA is a ModelBase family; its full-fit
// likelihood / MLE / MAP / posterior oracles come from the P4 dotnet emitter, NOT T2, so this
// file transcribes only the STRUCTURAL / DETERMINISM / TRIVIALLY-ANALYTIC assertions (internal
// support test territory -- hardcoded oracles are correct here; public-API oracle values stay
// in fixtures/).
//
// Structural oracles transcribed (values unaltered where they exist) from
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/TimeSeriesModels/ARIMATests.cs @ fc28c0c
//
// The C# fixtures draw their data from System.Random (a .NET LCG, not the ported Mersenne
// Twister), so the exact data VALUES are unreproducible in C++. This file regenerates
// structurally-equivalent data (same AR(1)-with-noise / monthly seasonal+trend shapes, ported
// bit-exact MersenneTwister, positive mean ~1000 so the log/Box-Cox transforms are well
// defined), then applies the same STRUCTURAL assertions (parameter counts/names/bounds,
// finiteness, sentinels, stationarity/invertibility predicates, differencing lengths,
// seeded-stream determinism, validation outcomes). No data-value oracle is claimed for the
// generated fixtures.
//
// Skipped C# test methods (see task-T2-report.md for the full list + reasons):
//   - XML tests (Test_Constructor_XElement_RestoresModel, Test_ToXElement_*,
//     Test_RoundTrip_PreservesAllProperties, *_XmlSerialization): XML is a project-wide non-port.
//   - Clone tests (Test_Clone_*): the C++ core has no virtual IModel::Clone (see model_base.hpp);
//     no fit path in T2 needs a clone, so clone() is omitted (T1 / S4 precedent).
//   - Test_SetParameterValues_NullParameters_ThrowsException: VACUOUS (const-ref vector; no null).
//   - Test_GenerateRandomSeries_* assert on the heavy TimeSeries container return; the REQUIRED
//     simulation entry point is generate_random_values (ISimulatable), tested here for length +
//     seeded determinism. GenerateRandomSeries itself is deferred (mirrors T1).
//
// P4 landed (see test_arima_p4_fixed_param_oracles below + fixtures/estimation/*):
//   - exact ARIMA DataLogLikelihood + Residuals at fixed parameters, dumped from the real
//     RMC.BestFit via the dotnet emitter and asserted here (route b, C++-only, 1e-9 abs);
//   - the exact MLE fit (parameters, log-likelihood, AIC) and the seeded cross-language
//     GenerateRandomValues draw are oracle-verified against the real C# in
//     time_series_arima_smoke.json.
// Still deferred (severable follow-up): exact Residuals under the Log/BoxCox/YeoJohnson
//   transforms and exact Predict() forecast values (the reverse-differencing integration).
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/time_series/arima.hpp"
#include "corehydro/models/time_series/transform_type.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"
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



using corehydro::models::ARIMA;
using corehydro::models::Transform;
using corehydro::numerics::data::TimeInterval;
using corehydro::numerics::data::TimeSeries;
using corehydro::numerics::sampling::MersenneTwister;

// Compile-time mirror of Model_InheritsFromModelBase / Model_ImplementsISimulatable.
static_assert(std::is_base_of<corehydro::models::ModelBase, ARIMA>::value,
              "ARIMA must derive from ModelBase");
static_assert(std::is_base_of<corehydro::models::ISimulatable<std::vector<double>>, ARIMA>::value,
              "ARIMA must implement ISimulatable<std::vector<double>>");

// ---- Test-data helpers (mirror the C# private fixtures; data regenerated with the ported RNG) ----

// 50 annual observations, AR(1)-with-noise, positive mean ~1000 (mirrors CreateSampleTimeSeries).
TimeSeries make_sample_series() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(49, TimeInterval::OneYear));
    MersenneTwister rng(12345);
    const double mean = 1000.0, phi = 0.6, sigma = 100.0;
    double prev = mean;
    for (int i = 0; i < ts.count(); ++i) {
        double innovation = rng.next_double() * 2.0 - 1.0;
        double value = mean + phi * (prev - mean) + sigma * innovation;
        ts[i].set_value(value);
        prev = value;
    }
    return ts;
}

// 15 annual observations, deterministic 100 + 10i (mirrors CreateShortTimeSeries).
TimeSeries make_short_series() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(14, TimeInterval::OneYear));
    for (int i = 0; i < ts.count(); ++i) ts[i].set_value(100.0 + i * 10.0);
    return ts;
}

// 240 monthly observations, seasonal + trend + noise (mirrors CreateMonthlyTimeSeries).
TimeSeries make_monthly_series() {
    TimeSeries ts(TimeInterval::OneMonth, kT0, t0_plus(239, TimeInterval::OneMonth));
    MersenneTwister rng(54321);
    for (int i = 0; i < ts.count(); ++i) {
        double seasonal = 50.0 * std::sin(2.0 * corehydro::numerics::kPi * i / 12.0);
        double trend = 0.5 * i;
        double noise = rng.next_double() * 20.0 - 10.0;
        ts[i].set_value(500.0 + seasonal + trend + noise);
    }
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

// 60 annual observations that make the Box-Cox lambda objective non-finite: a zero at index 0
// fails can_fit_lambda's positivity requirement (mirrors C#'s
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

// ============================ Constructors ============================

void test_arima_constructors() {
    ARIMA empty;
    CHECK_EQ(empty.p_order(), 1);
    CHECK_EQ(empty.q_order(), 0);
    CHECK_EQ(empty.d_order(), 0);
    CHECK_TRUE(empty.include_intercept());

    TimeSeries ts = make_sample_series();
    ARIMA m1(ts);
    CHECK_EQ(m1.time_series().count(), ts.count());  // adapted AreSame -> structural equality
    CHECK_EQ(m1.p_order(), 1);
    CHECK_EQ(m1.q_order(), 0);

    ARIMA mo(ts, 2, 0, 1);
    CHECK_EQ(mo.p_order(), 2);
    CHECK_EQ(mo.q_order(), 1);

    ARIMA mni(ts, 1, 0, 0, false);
    CHECK_TRUE(!mni.include_intercept());
}

// ============================ Properties ============================

void test_arima_properties() {
    TimeSeries ts = make_sample_series();

    ARIMA m;
    m.set_time_series(ts);
    CHECK_EQ(m.time_series().count(), ts.count());

    ARIMA mp(ts);
    mp.set_p_order(3);
    CHECK_EQ(mp.p_order(), 3);

    ARIMA mpc(ts, 1, 0, 0);
    int initial = mpc.number_of_parameters();
    mpc.set_p_order(3);
    CHECK_EQ(mpc.number_of_parameters(), initial + 2);

    ARIMA mq(ts);
    mq.set_q_order(2);
    CHECK_EQ(mq.q_order(), 2);

    ARIMA mqc(ts, 1, 0, 0);
    int initialq = mqc.number_of_parameters();
    mqc.set_q_order(2);
    CHECK_EQ(mqc.number_of_parameters(), initialq + 2);

    ARIMA mi(ts);
    mi.set_include_intercept(false);
    CHECK_TRUE(!mi.include_intercept());

    ARIMA mic(ts, 1, 0, 0, true);
    int initial2 = mic.number_of_parameters();
    mic.set_include_intercept(false);
    CHECK_EQ(mic.number_of_parameters(), initial2 - 1);
}

// ============================ SetDefaultParameters ============================

void test_arima_set_default_parameters() {
    TimeSeries ts = make_sample_series();

    ARIMA ar1_ni(ts, 1, 0, 0, false);
    CHECK_EQ(ar1_ni.number_of_parameters(), 2);  // phi + sigma

    ARIMA ar1(ts, 1, 0, 0, true);
    CHECK_EQ(ar1.number_of_parameters(), 3);  // mu + phi + sigma

    ARIMA a21(ts, 2, 0, 1, true);
    CHECK_EQ(a21.number_of_parameters(), 5);  // mu + phi1 + phi2 + theta1 + sigma

    CHECK_TRUE(a21.parameters()[0].name().find("Intercept") != std::string::npos);
    CHECK_TRUE(a21.parameters()[1].name().find("AR") != std::string::npos);
    CHECK_TRUE(a21.parameters()[2].name().find("AR") != std::string::npos);
    CHECK_TRUE(a21.parameters()[3].name().find("MA") != std::string::npos);
    CHECK_TRUE(a21.parameters()[4].name().find("Scale") != std::string::npos);

    // AR coefficient bounds [-2, 2]; phi at index 1 (after intercept).
    CHECK_NEAR(ar1.parameters()[1].lower_bound(), -2.0, 0.0);
    CHECK_NEAR(ar1.parameters()[1].upper_bound(), 2.0, 0.0);

    // MA coefficient bounds [-2, 2]; pure MA(1) -> theta at index 1.
    ARIMA ma1(ts, 0, 0, 1);
    CHECK_NEAR(ma1.parameters()[1].lower_bound(), -2.0, 0.0);
    CHECK_NEAR(ma1.parameters()[1].upper_bound(), 2.0, 0.0);

    // Scale is positive with a strictly positive lower bound (last parameter).
    const auto& scale = ar1.parameters().back();
    CHECK_TRUE(scale.is_positive());
    CHECK_TRUE(scale.lower_bound() > 0.0);
}

// ============================ LogLikelihood ============================

void test_arima_log_likelihood() {
    TimeSeries ts = make_sample_series();

    ARIMA a10(ts, 1, 0, 0);
    auto p1 = a10.parameter_values();
    double d1 = a10.data_log_likelihood(p1);
    CHECK_TRUE(!std::isnan(d1));
    CHECK_TRUE(!(std::isinf(d1) && d1 > 0.0));  // not +inf

    ARIMA a20(ts, 2, 0, 0);
    auto p2 = a20.parameter_values();
    CHECK_TRUE(!std::isnan(a20.data_log_likelihood(p2)));

    ARIMA a11(ts, 1, 0, 1);
    auto p11 = a11.parameter_values();
    CHECK_TRUE(!std::isnan(a11.data_log_likelihood(p11)));

    // Null time series -> -inf sentinel.
    ARIMA nullm;
    std::vector<double> pnull = {0, 0, 1};
    double rn = nullm.data_log_likelihood(pnull);
    CHECK_TRUE(std::isinf(rn) && rn < 0.0);

    // NaN parameter -> -inf sentinel.
    std::vector<double> pnan = {std::numeric_limits<double>::quiet_NaN(), 0, 1};
    double rnan = a10.data_log_likelihood(pnan);
    CHECK_TRUE(std::isinf(rnan) && rnan < 0.0);

    // Prior finite.
    double pr = a10.prior_log_likelihood(p1);
    CHECK_TRUE(!std::isnan(pr));
    CHECK_TRUE(!(std::isinf(pr) && pr > 0.0));

    // Pointwise count = TrainingTimeSteps - maxOrder (full-series training, p=2).
    ARIMA ap(ts, 2, 0, 0);
    ap.set_use_default_training_steps(false);
    ap.set_training_time_steps(ts.count());
    auto pp = ap.parameter_values();
    auto pw = ap.pointwise_data_log_likelihood(pp);
    CHECK_EQ(static_cast<int>(pw.size()), ts.count() - 2);

    // Sum of pointwise == total data log-likelihood.
    ARIMA as(ts, 1, 0, 0);
    as.set_use_default_training_steps(false);
    as.set_training_time_steps(ts.count());
    auto ps = as.parameter_values();
    auto pws = as.pointwise_data_log_likelihood(ps);
    double sum = 0.0;
    for (double v : pws) sum += v;
    CHECK_NEAR(sum, as.data_log_likelihood(ps), 1e-6);

    // Components all Exact.
    auto comps = a10.pointwise_data_log_likelihood_components(p1);
    for (const auto& c : comps)
        CHECK_TRUE(c.type() == corehydro::models::DataComponentType::Exact);
}

// ============================ Predict ============================

void test_arima_predict() {
    TimeSeries ts = make_sample_series();

    ARIMA m(ts);
    m.set_use_default_training_steps(false);
    m.set_training_time_steps(ts.count());

    CHECK_EQ(static_cast<int>(m.predict(0).size()), ts.count());
    CHECK_EQ(static_cast<int>(m.predict(10).size()), ts.count() + 10);

    // Deterministic (seed -1) reproducible.
    ARIMA d(ts, 1, 0, 0);
    d.set_use_default_training_steps(false);
    d.set_training_time_steps(ts.count());
    auto a = d.predict(5, -1);
    auto b = d.predict(5, -1);
    for (std::size_t i = 0; i < a.size(); ++i) CHECK_NEAR(a[i], b[i], 1e-10);

    // Seeded reproducible.
    auto s1 = d.predict(5, 12345);
    auto s2 = d.predict(5, 12345);
    for (std::size_t i = 0; i < s1.size(); ++i) CHECK_NEAR(s1[i], s2[i], 1e-10);

    // Different seeds differ somewhere in the forecast tail.
    auto d1 = d.predict(5, 12345);
    auto d2 = d.predict(5, 54321);
    bool any_diff = false;
    for (int i = d.training_time_steps(); i < static_cast<int>(d1.size()); ++i)
        if (std::fabs(d1[i] - d2[i]) > 1e-10) any_diff = true;
    CHECK_TRUE(any_diff);

    // Differencing round-trips through Predict (ARIMA(1,1,0) integration path).
    ARIMA di(ts, 1, 0, 0);
    di.set_use_default_training_steps(false);
    di.set_training_time_steps(ts.count());
    di.set_d_order(1);
    auto pi0 = di.predict(0);
    CHECK_EQ(static_cast<int>(pi0.size()), ts.count());
    for (double v : pi0) CHECK_TRUE(!std::isnan(v));

    // Null time series -> throws.
    ARIMA nullm;
    CHECK_THROWS(nullm.predict());
}

// ============================ GenerateRandomValues ============================

void test_arima_generate_random_values() {
    TimeSeries ts = make_sample_series();
    ARIMA m(ts);

    CHECK_EQ(static_cast<int>(m.generate_random_values(100, 12345).size()), 100);

    auto g1 = m.generate_random_values(50, 12345);
    auto g2 = m.generate_random_values(50, 12345);
    for (std::size_t i = 0; i < g1.size(); ++i) CHECK_NEAR(g1[i], g2[i], 1e-10);

    CHECK_THROWS(m.generate_random_values(0, 12345));
}

// ============================ Stationarity / Invertibility ============================

void test_arima_stationarity_invertibility() {
    TimeSeries ts = make_sample_series();

    // AR(1): phi at index (intercept?1:0).
    ARIMA a10(ts, 1, 0, 0);
    int idx = a10.include_intercept() ? 1 : 0;
    a10.parameters()[idx].set_value(0.5);
    CHECK_TRUE(a10.is_stationary());
    a10.parameters()[idx].set_value(1.1);
    CHECK_TRUE(!a10.is_stationary());

    // MA(1): theta at index (intercept?1:0).
    ARIMA m01(ts, 0, 0, 1);
    int idxm = m01.include_intercept() ? 1 : 0;
    m01.parameters()[idxm].set_value(0.5);
    CHECK_TRUE(m01.is_invertible());
    m01.parameters()[idxm].set_value(1.1);
    CHECK_TRUE(!m01.is_invertible());

    // No AR component -> IsStationary true.
    ARIMA noAr(ts, 0, 0, 1);
    CHECK_TRUE(noAr.is_stationary());

    // No MA component -> IsInvertible true.
    ARIMA noMa(ts, 1, 0, 0);
    CHECK_TRUE(noMa.is_invertible());
}

// ============================ Validation ============================

void test_arima_validate() {
    TimeSeries ts = make_sample_series();

    ARIMA valid(ts);
    CHECK_TRUE(valid.validate().is_valid);

    ARIMA nullm;
    auto rn = nullm.validate();
    CHECK_TRUE(!rn.is_valid);
    CHECK_TRUE(messages_contain(rn, "Time series"));

    TimeSeries tiny = make_tiny_series();
    auto rs = ARIMA(tiny).validate();
    CHECK_TRUE(!rs.is_valid);
    CHECK_TRUE(messages_contain(rs, "10 observations"));

    ARIMA bp(ts);
    bp.set_p_order(-1);
    CHECK_TRUE(!bp.validate().is_valid);

    ARIMA bq(ts);
    bq.set_q_order(-1);
    CHECK_TRUE(!bq.validate().is_valid);

    ARIMA both(ts, 0, 0, 0);
    auto rb = both.validate();
    CHECK_TRUE(!rb.is_valid);
    CHECK_TRUE(messages_contain(rb, "greater than 0"));

    // Non-stationary parameters produce a warning mentioning "stationar".
    ARIMA ns(ts, 1, 0, 0);
    int idx = ns.include_intercept() ? 1 : 0;
    ns.parameters()[idx].set_value(1.5);
    CHECK_TRUE(messages_contain(ns.validate(), "stationar"));
}

// Transcribes ARIMATests.cs's Test_{BoxCoxTransform,YeoJohnsonTransform}Transform_
// LambdaFitFailure_ReturnsValidationError (BestFit v2.0.0, f140c4d).
void test_arima_transform_lambda_failure() {
    ARIMA bc(make_box_cox_lambda_failure_series(), 1, 0, 0);
    bc.set_transform_type(Transform::BoxCox);
    auto rbc = bc.validate();
    CHECK_TRUE(!rbc.is_valid);
    CHECK_TRUE(messages_contain(rbc, "Box-Cox lambda estimation failed"));

    ARIMA yj(make_yeo_johnson_lambda_failure_series(), 1, 0, 0);
    yj.set_transform_type(Transform::YeoJohnson);
    auto ryj = yj.validate();
    CHECK_TRUE(!ryj.is_valid);
    CHECK_TRUE(messages_contain(ryj, "Yeo-Johnson lambda estimation failed"));
}

// ============================ SetParameterValues ============================

void test_arima_set_parameter_values() {
    TimeSeries ts = make_sample_series();
    ARIMA m(ts, 1, 0, 0);
    std::vector<double> nv = {1000.0, 0.5, 50.0};
    m.set_parameter_values(nv);
    CHECK_NEAR(m.parameters()[0].value(), 1000.0, 0.0);
    CHECK_NEAR(m.parameters()[1].value(), 0.5, 0.0);
    CHECK_NEAR(m.parameters()[2].value(), 50.0, 0.0);

    CHECK_THROWS(m.set_parameter_values(std::vector<double>{1.0}));
}

// ============================ Engineering / edge cases ============================

void test_arima_engineering_and_edges() {
    TimeSeries ts = make_sample_series();

    // Streamflow persistence: AR(1) valid + finite (not -inf) data log-likelihood.
    ARIMA sp(ts, 1, 0, 0);
    CHECK_TRUE(sp.validate().is_valid);
    auto psp = sp.parameter_values();
    double dsp = sp.data_log_likelihood(psp);
    CHECK_TRUE(!(std::isinf(dsp) && dsp < 0.0));

    // Monthly analysis: ARIMA(1,0,1) valid.
    TimeSeries monthly = make_monthly_series();
    ARIMA ma(monthly, 1, 0, 1);
    CHECK_TRUE(ma.validate().is_valid);

    // Higher-order ARIMA(3,0,2): 7 parameters, valid.
    ARIMA ho(ts, 3, 0, 2);
    CHECK_TRUE(ho.validate().is_valid);
    CHECK_EQ(ho.number_of_parameters(), 7);  // mu + phi1..3 + theta1..2 + sigma

    // Minimum valid time series (15 obs, full training).
    TimeSeries shortts = make_short_series();
    ARIMA mv(shortts, 1, 0, 0);
    mv.set_use_default_training_steps(false);
    mv.set_training_time_steps(shortts.count());
    CHECK_TRUE(mv.validate().is_valid);

    // Large orders.
    ARIMA lo(ts, 5, 0, 5);
    CHECK_EQ(lo.p_order(), 5);
    CHECK_EQ(lo.q_order(), 5);

    // Pure MA / pure AR both valid.
    ARIMA pma(ts, 0, 0, 2);
    CHECK_TRUE(pma.validate().is_valid);
    ARIMA par(ts, 2, 0, 0);
    CHECK_TRUE(par.validate().is_valid);
}

// ============================ DOrder (differencing) ============================

void test_arima_dorder() {
    TimeSeries ts = make_sample_series();

    ARIMA m(ts);
    m.set_d_order(1);
    CHECK_EQ(m.d_order(), 1);

    // ARIMA(1,1,0) validates.
    ARIMA a110(ts, 1, 0, 0);
    a110.set_d_order(1);
    CHECK_TRUE(a110.validate().is_valid);

    // ARIMA(1,2,1) validates.
    ARIMA a121(ts, 1, 0, 1);
    a121.set_d_order(2);
    CHECK_TRUE(a121.validate().is_valid);

    // DifferencedSeries length = TrainingTimeSteps - DOrder (full training, d=1).
    ARIMA d1(ts, 1, 0, 0);
    d1.set_use_default_training_steps(false);
    d1.set_training_time_steps(ts.count());
    d1.set_d_order(1);
    CHECK_EQ(d1.differenced_series().count(), ts.count() - 1);

    // DifferencedSeries with d=0 == training length.
    ARIMA d0(ts, 1, 0, 0);
    d0.set_use_default_training_steps(false);
    d0.set_training_time_steps(ts.count());
    d0.set_d_order(0);
    CHECK_EQ(d0.differenced_series().count(), ts.count());
}

// ============================ Transform / training / Jeffreys ============================

void test_arima_transform_training_jeffreys() {
    TimeSeries ts = make_sample_series();

    ARIMA tt(ts);
    tt.set_transform_type(Transform::Logarithmic);
    CHECK_TRUE(tt.transform_type() == Transform::Logarithmic);

    ARIMA lt(ts, 1, 0, 0);
    lt.set_transform_type(Transform::Logarithmic);
    CHECK_TRUE(lt.validate().is_valid);

    ARIMA bc(ts, 1, 0, 0);
    bc.set_transform_type(Transform::BoxCox);
    CHECK_TRUE(bc.validate().is_valid);

    // TrainingTimeSteps get/set + ForecastingTimeSteps derived.
    ARIMA tr(ts);
    tr.set_use_default_training_steps(false);
    tr.set_training_time_steps(35);
    CHECK_EQ(tr.training_time_steps(), 35);

    ARIMA fc(ts);
    fc.set_use_default_training_steps(false);
    fc.set_training_time_steps(40);
    CHECK_EQ(fc.forecasting_time_steps(), ts.count() - 40);

    ARIMA ud(ts);
    ud.set_use_default_training_steps(false);
    CHECK_TRUE(!ud.use_default_training_steps());

    // Jeffreys rule toggle + finite prior both ways.
    ARIMA jr(ts, 1, 0, 0);
    jr.set_use_jeffreys_rule_for_scale(false);
    CHECK_TRUE(!jr.use_jeffreys_rule_for_scale());

    ARIMA j1(ts, 1, 0, 0);
    j1.set_use_jeffreys_rule_for_scale(true);
    auto pj1 = j1.parameter_values();
    double prj1 = j1.prior_log_likelihood(pj1);
    CHECK_TRUE(!std::isnan(prj1) && !(std::isinf(prj1) && prj1 > 0.0));

    ARIMA j0(ts, 1, 0, 0);
    j0.set_use_jeffreys_rule_for_scale(false);
    auto pj0 = j0.parameter_values();
    double prj0 = j0.prior_log_likelihood(pj0);
    CHECK_TRUE(!std::isnan(prj0) && !(std::isinf(prj0) && prj0 > 0.0));
}

// v2.0.0 ResetDefaultTrainingStepsForNewTimeSeries (ARIMA.cs:411): attaching a DIFFERENT response
// series discards the previous manual training-window edit and restores the default 80% split.
//
// PROVENANCE, stated exactly: the C# regression test
// TimeSeries_NewSeries_ResetsTrainingSplitToDefault exists ONLY in ARIMAXTests.cs (line 211);
// ARIMATests.cs has no such test. The C# PRODUCTION method, by contrast, is present verbatim in
// all four model files, so this block is an EXTRAPOLATION of the ARIMAX test to ARIMA, whose
// production code is identical -- not a transcription of a test that exists for ARIMA. The
// expected values follow the same C# default-split formula the ARIMAX test asserts.
//
// The manual window (25) is deliberately NOT the default the 240-observation replacement produces
// (floor(0.8*240) = 192), so this fails on the pre-port code path.
void test_arima_new_series_resets_training_split() {
    TimeSeries original = make_sample_series();      // 50 annual obs
    TimeSeries replacement = make_monthly_series();  // 240 monthly obs
    TimeSeries empty(TimeInterval::OneYear);

    ARIMA m(original, 1, 0, 0);
    m.set_use_default_training_steps(false);
    m.set_training_time_steps(25);
    CHECK_EQ(m.training_time_steps(), 25);
    m.set_time_series(replacement);
    CHECK_TRUE(m.use_default_training_steps());
    CHECK_EQ(m.training_time_steps(), 192);

    // Empty-series arm: attaching a series with no observations zeroes the window. Default
    // flat priors are turned OFF first only to keep the assertion on the reset itself -- the
    // trailing SetDefaultParameters call indexes the STALE differenced series at
    // TrainingTimeSteps-1 == -1, which is an IndexOutOfRangeException in C# and UB here (a
    // pre-existing shared hazard on a path the C# regression suite never drives).
    ARIMA me(original, 1, 0, 0);
    me.set_use_default_flat_priors(false);
    me.set_use_default_training_steps(false);
    me.set_training_time_steps(25);
    me.set_time_series(empty);
    CHECK_TRUE(me.use_default_training_steps());
    CHECK_EQ(me.training_time_steps(), 0);
}

// v2.0.0 TimeInterval.Irregular Validate guard (ARIMA.cs:1269). Transcribed from the new
// Test_Validate_IrregularTimeSeries_ReturnsFalse regression test; the cross-language oracle lives
// in fixtures/estimation/time_series_irregular_interval.json.
void test_arima_irregular_interval_validate() {
    // C# `new TimeSeries(TimeInterval.Irregular)` + Add: the (interval, start, end) ctor
    // rejects Irregular in BOTH C# and this port, so build it ordinate-by-ordinate with the
    // C# test's own i*i+1 index spacing.
    TimeSeries irregular(TimeInterval::Irregular);
    for (int i = 0; i < 50; ++i)
        irregular.add(TimeSeries::Ordinate(kT0.add_days(static_cast<double>(i) * i + 1), i + 1.0));

    auto r = ARIMA(irregular, 1, 0, 0).validate();
    CHECK_TRUE(!r.is_valid);
    CHECK_TRUE(messages_contain(r, "regular time interval"));

    CHECK_TRUE(ARIMA(make_sample_series(), 1, 0, 0).validate().is_valid);
}

}  // namespace

// P4 oracles (route b): exact fixed-parameter DataLogLikelihood + Residuals dumped from the REAL
// RMC.BestFit ARIMA via tools/oracle_emitter (Numerics @ a2c4dbf, RMC-BestFit @ fc28c0c) on the
// shared 20-point fixture series. Deterministic (no fit) -> hardcoded C++-only oracles at 1e-9 abs.
// The exact MLE fit + seeded draw are oracle-verified cross-language in
// fixtures/estimation/time_series_arima_smoke.json.
void test_arima_p4_fixed_param_oracles() {
    // ARIMA(1,1,1)+intercept, params [diff-intercept, phi1, theta1, sigma] = [0.0, -0.5, 0.5, 1.3].
    TimeSeries series(TimeInterval::OneDay, kT0,
                      std::vector<double>{10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0,
                                          13.1, 11.7, 12.5, 13.9, 15.2, 14.1, 12.9, 13.6, 15.0,
                                          16.2, 14.8});
    ARIMA arima(series, 1, 1, 1, true);
    arima.set_transform_type(Transform::None);
    std::vector<double> p{0.0, -0.5, 0.5, 1.3};
    CHECK_NEAR(arima.data_log_likelihood(p), -31.02818097335498, 1e-9);
    std::vector<double> res = arima.residuals(p);
    CHECK_NEAR(res[0], 0.0, 1e-9);
    CHECK_NEAR(res[8], -0.9050781250000004, 1e-9);
}

int main() {
    test_arima_constructors();
    test_arima_properties();
    test_arima_set_default_parameters();
    test_arima_log_likelihood();
    test_arima_predict();
    test_arima_generate_random_values();
    test_arima_stationarity_invertibility();
    test_arima_validate();
    test_arima_transform_lambda_failure();
    test_arima_set_parameter_values();
    test_arima_engineering_and_edges();
    test_arima_dorder();
    test_arima_transform_training_jeffreys();

    test_arima_new_series_resets_training_split();
    test_arima_irregular_interval_validate();

    test_arima_p4_fixed_param_oracles();

    return chtest::summary("arima");
}
