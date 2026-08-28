// T3 support ctest (C++-only): the ARIMAX TimeSeries model -- the richest ModelBase family in
// the TimeSeries slice (ARIMA differenced+transformed mean PLUS an inline polynomial trend, an
// inline Fourier seasonality term, and exogenous covariate regression). Its full-fit likelihood
// / MLE / MAP / posterior oracles come from the P4 dotnet emitter, NOT T3, so this file
// transcribes only the STRUCTURAL / DETERMINISM / TRIVIALLY-ANALYTIC assertions (internal
// support test territory -- hardcoded oracles are correct here; public-API oracle values stay
// in fixtures/).
//
// Structural oracles transcribed (values unaltered where they exist) from
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/TimeSeriesModels/ARIMAXTests.cs @ fc28c0c
//
// The C# fixtures draw their data from System.Random (a .NET LCG, not the ported Mersenne
// Twister), so the exact data VALUES are unreproducible in C++. This file regenerates
// structurally-equivalent data (same AR(1)+trend / monthly seasonal+trend shapes, ported
// bit-exact MersenneTwister, positive mean ~1000 so the log/Box-Cox transforms are well
// defined), then applies the same STRUCTURAL assertions (parameter counts/names/bounds,
// finiteness, sentinels, differencing/seasonality/trend layout, seeded-stream determinism,
// validation outcomes). No data-value oracle is claimed for the generated fixtures.
//
// Skipped C# test methods (see task-T3-report.md for the full list + reasons):
//   - XML tests (Test_Constructor_XElement_RestoresModel, Test_ToXElement_*,
//     Test_RoundTrip_PreservesAllProperties, Test_CovariateExtension_XmlSerialization):
//     XML is a project-wide non-port.
//   - Clone tests (Test_Clone_*): the C++ core has no virtual IModel::Clone (see model_base.hpp);
//     no fit path in T3 needs a clone, so clone() is omitted (T1 / T2 / S4 precedent).
//   - Test_SetParameterValues_NullParameters_ThrowsException: VACUOUS (const-ref vector; no null).
//   - The CovariateExtension Predict/Generate tests (Test_Predict_CovariateExtension*,
//     Test_GenerateRandomValues_CovariateExtension*): the covariate forecast-tail extension
//     (BlockBootstrap/KNN) needs the DEFERRED heavy TimeSeries container (ResampleWithBlockBootstrap
//     / ResampleWithKNN); deferred with that container per PHASE7_PLAN.md.
//
// P4 landed (see test_arimax_p4_fixed_param_oracles below + fixtures/estimation/*):
//   - exact ARIMAX DataLogLikelihood + Residuals at fixed parameters (intercept + linear trend +
//     AR(1)), dumped from the real RMC.BestFit via the dotnet emitter and asserted here (route b,
//     C++-only, 1e-9 abs);
//   - the exact MLE fit (parameters, log-likelihood, AIC) and the seeded cross-language
//     GenerateRandomValues draw are oracle-verified against the real C# in
//     time_series_arimax_smoke.json.
// Still deferred (severable follow-up): exact Residuals under the seasonality / covariate /
//   Transform / differencing configs and exact Predict() forecast values.
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/time_series/arimax.hpp"
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



using corehydro::models::ARIMAX;
using corehydro::models::Transform;
using corehydro::numerics::data::TimeInterval;
using corehydro::numerics::data::TimeSeries;
using corehydro::numerics::sampling::MersenneTwister;

// Compile-time mirror of Model_InheritsFromModelBase / Model_ImplementsISimulatable.
static_assert(std::is_base_of<corehydro::models::ModelBase, ARIMAX>::value,
              "ARIMAX must derive from ModelBase");
static_assert(std::is_base_of<corehydro::models::ISimulatable<std::vector<double>>, ARIMAX>::value,
              "ARIMAX must implement ISimulatable<std::vector<double>>");

// ---- Test-data helpers (mirror the C# private fixtures; data regenerated with the ported RNG) ----

// 60 annual observations, AR(1)-with-trend, positive mean ~1000 (mirrors CreateSampleTimeSeries).
TimeSeries make_sample_series() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(59, TimeInterval::OneYear));
    MersenneTwister rng(12345);
    const double mean = 1000.0, phi = 0.6, sigma = 100.0, trend_slope = 5.0;
    double prev = mean;
    for (int i = 0; i < ts.count(); ++i) {
        double innovation = rng.next_double() * 2.0 - 1.0;
        double trend = trend_slope * i;
        double value = mean + trend + phi * (prev - mean - trend_slope * (i - 1)) + sigma * innovation;
        ts[i].set_value(value);
        prev = value;
    }
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

// 15 annual observations, deterministic 100 + 10i (mirrors CreateShortTimeSeries).
TimeSeries make_short_series() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(14, TimeInterval::OneYear));
    for (int i = 0; i < ts.count(); ++i) ts[i].set_value(100.0 + i * 10.0);
    return ts;
}

// Covariate aligned to the target series (mirrors CreateCovariateTimeSeries).
TimeSeries make_covariate_series(const TimeSeries& target) {
    TimeSeries ts(target.time_interval(), kT0,
                  t0_plus(target.count() - 1, target.time_interval()));
    MersenneTwister rng(67890);
    for (int i = 0; i < ts.count(); ++i)
        ts[i].set_value(0.5 * target[i].value() + rng.next_double() * 50.0);
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

void test_arimax_constructors() {
    ARIMAX empty;
    CHECK_EQ(empty.ar_order_p(), 1);
    CHECK_EQ(empty.diff_order_d(), 0);
    CHECK_EQ(empty.ma_order_q(), 0);
    CHECK_TRUE(empty.include_intercept());

    TimeSeries ts = make_sample_series();
    ARIMAX m(ts);
    CHECK_EQ(m.time_series().count(), ts.count());  // adapted AreSame -> structural equality
    CHECK_EQ(m.ar_order_p(), 1);
}

// ============================ Properties ============================

void test_arimax_properties() {
    TimeSeries ts = make_sample_series();

    ARIMAX m;
    m.set_time_series(ts);
    CHECK_EQ(m.time_series().count(), ts.count());

    ARIMAX mp(ts);
    mp.set_ar_order_p(3);
    CHECK_EQ(mp.ar_order_p(), 3);

    ARIMAX mq(ts);
    mq.set_ma_order_q(2);
    CHECK_EQ(mq.ma_order_q(), 2);

    ARIMAX md(ts);
    md.set_diff_order_d(1);
    CHECK_EQ(md.diff_order_d(), 1);

    ARIMAX mx(ts);
    mx.set_x_order_b(2);
    CHECK_EQ(mx.x_order_b(), 2);

    ARIMAX mi(ts);
    mi.set_include_intercept(false);
    CHECK_TRUE(!mi.include_intercept());

    ARIMAX mtr(ts);
    mtr.set_trend_type(ARIMAX::Trend::Quadratic);
    CHECK_TRUE(mtr.trend_type() == ARIMAX::Trend::Quadratic);

    TimeSeries monthly = make_monthly_series();
    ARIMAX ms(monthly);
    ms.set_include_seasonality(true);
    CHECK_TRUE(ms.include_seasonality());

    ARIMAX mt(ts);
    mt.set_transform_type(Transform::Logarithmic);
    CHECK_TRUE(mt.transform_type() == Transform::Logarithmic);
}

// ============================ Parameter layout ============================

void test_arimax_number_of_parameters() {
    TimeSeries ts = make_sample_series();

    // AR(1) with intercept: mu + phi + sigma = 3.
    ARIMAX ar1(ts);
    ar1.set_ar_order_p(1);
    ar1.set_ma_order_q(0);
    ar1.set_trend_type(ARIMAX::Trend::None);
    ar1.set_include_seasonality(false);
    CHECK_EQ(ar1.number_of_parameters(), 3);

    // ARMA(1,1) with intercept: mu + phi + theta + sigma = 4.
    ARIMAX arma11(ts);
    arma11.set_ar_order_p(1);
    arma11.set_ma_order_q(1);
    arma11.set_trend_type(ARIMAX::Trend::None);
    CHECK_EQ(arma11.number_of_parameters(), 4);

    // AR(1) + linear trend: mu + gamma + phi + sigma = 4.
    ARIMAX lin(ts);
    lin.set_ar_order_p(1);
    lin.set_ma_order_q(0);
    lin.set_trend_type(ARIMAX::Trend::Linear);
    CHECK_EQ(lin.number_of_parameters(), 4);

    // AR(1) + quadratic trend: mu + gamma1 + gamma2 + phi + sigma = 5.
    ARIMAX quad(ts);
    quad.set_ar_order_p(1);
    quad.set_ma_order_q(0);
    quad.set_trend_type(ARIMAX::Trend::Quadratic);
    CHECK_EQ(quad.number_of_parameters(), 5);

    // Trend increments: +1 linear, +2 quadratic, +3 cubic.
    ARIMAX tr(ts);
    tr.set_trend_type(ARIMAX::Trend::None);
    int base = tr.number_of_parameters();
    tr.set_trend_type(ARIMAX::Trend::Linear);
    CHECK_EQ(tr.number_of_parameters(), base + 1);
    tr.set_trend_type(ARIMAX::Trend::None);
    tr.set_trend_type(ARIMAX::Trend::Quadratic);
    CHECK_EQ(tr.number_of_parameters(), base + 2);
    tr.set_trend_type(ARIMAX::Trend::None);
    tr.set_trend_type(ARIMAX::Trend::Cubic);
    CHECK_EQ(tr.number_of_parameters(), base + 3);

    // Seasonality adds a sin/cos Fourier pair.
    TimeSeries monthly = make_monthly_series();
    ARIMAX seas(monthly);
    seas.set_include_seasonality(false);
    int base_s = seas.number_of_parameters();
    seas.set_include_seasonality(true);
    CHECK_TRUE(seas.number_of_parameters() > base_s);

    // Removing the intercept drops exactly one parameter.
    ARIMAX ni(ts);
    ni.set_include_intercept(false);
    int without = ni.number_of_parameters();
    ni.set_include_intercept(true);
    CHECK_EQ(ni.number_of_parameters(), without + 1);
}

void test_arimax_parameter_names_bounds() {
    TimeSeries ts = make_sample_series();

    ARIMAX m(ts);
    m.set_ar_order_p(2);
    m.set_ma_order_q(1);
    m.set_trend_type(ARIMAX::Trend::None);
    // Layout: Intercept, AR1, AR2, MA1, Scale.
    CHECK_TRUE(m.parameters()[0].name().find("Intercept") != std::string::npos);
    CHECK_TRUE(m.parameters()[1].name().find("AR") != std::string::npos);
    CHECK_TRUE(m.parameters()[2].name().find("AR") != std::string::npos);
    CHECK_TRUE(m.parameters()[3].name().find("MA") != std::string::npos);
    CHECK_TRUE(m.parameters().back().name().find("Scale") != std::string::npos);

    // AR coefficient bounds [-2, 2] (index 1, after intercept).
    CHECK_NEAR(m.parameters()[1].lower_bound(), -2.0, 0.0);
    CHECK_NEAR(m.parameters()[1].upper_bound(), 2.0, 0.0);

    // Scale is positive with a strictly positive lower bound (last parameter).
    const auto& scale = m.parameters().back();
    CHECK_TRUE(scale.is_positive());
    CHECK_TRUE(scale.lower_bound() > 0.0);

    // Trend / seasonality parameter names.
    ARIMAX tr(ts);
    tr.set_trend_type(ARIMAX::Trend::Linear);
    CHECK_TRUE(tr.parameters()[1].name().find("Trend") != std::string::npos);

    TimeSeries monthly = make_monthly_series();
    ARIMAX seas(monthly);
    seas.set_include_seasonality(true);
    bool has_seasonality = false;
    for (const auto& p : seas.parameters())
        if (p.name().find("Seasonality") != std::string::npos) has_seasonality = true;
    CHECK_TRUE(has_seasonality);
}

// ============================ SetParameterValues ============================

void test_arimax_set_parameter_values() {
    TimeSeries ts = make_sample_series();
    ARIMAX m(ts);
    m.set_ar_order_p(1);
    m.set_ma_order_q(0);
    m.set_trend_type(ARIMAX::Trend::None);

    std::vector<double> nv = {1000.0, 0.5, 50.0};
    m.set_parameter_values(nv);
    CHECK_NEAR(m.parameters()[0].value(), 1000.0, 0.0);
    CHECK_NEAR(m.parameters()[1].value(), 0.5, 0.0);
    CHECK_NEAR(m.parameters()[2].value(), 50.0, 0.0);

    CHECK_THROWS(m.set_parameter_values(std::vector<double>{1.0}));
}

// ============================ LogLikelihood ============================

void test_arimax_log_likelihood() {
    TimeSeries ts = make_sample_series();

    ARIMAX m(ts);
    auto p = m.parameter_values();
    double d = m.data_log_likelihood(p);
    CHECK_TRUE(!std::isnan(d));
    CHECK_TRUE(!(std::isinf(d) && d > 0.0));  // not +inf

    // With a linear trend, still finite (not NaN).
    ARIMAX tr(ts);
    tr.set_trend_type(ARIMAX::Trend::Linear);
    auto ptr = tr.parameter_values();
    CHECK_TRUE(!std::isnan(tr.data_log_likelihood(ptr)));

    // ARMA(2,1) finite.
    ARIMAX a21(ts);
    a21.set_ar_order_p(2);
    a21.set_ma_order_q(1);
    auto p21 = a21.parameter_values();
    CHECK_TRUE(!std::isnan(a21.data_log_likelihood(p21)));

    // Null time series -> DataLogLikelihood is 0.0 (ARIMAX sentinel; C# governs, ARIMAXTests.cs:459).
    ARIMAX nullm;
    auto pnull = nullm.parameter_values();
    double rn = nullm.data_log_likelihood(pnull);
    CHECK_NEAR(rn, 0.0, 0.0);

    // Prior finite.
    double pr = m.prior_log_likelihood(p);
    CHECK_TRUE(!std::isnan(pr));
    CHECK_TRUE(!(std::isinf(pr) && pr > 0.0));

    // Pointwise: positive count not exceeding the series length.
    ARIMAX ap(ts);
    ap.set_ar_order_p(2);
    auto pap = ap.parameter_values();
    auto pw = ap.pointwise_data_log_likelihood(pap);
    CHECK_TRUE(static_cast<int>(pw.size()) > 0);
    CHECK_TRUE(static_cast<int>(pw.size()) <= ts.count());

    // Sum of pointwise == total data log-likelihood.
    auto pws = m.pointwise_data_log_likelihood(p);
    double sum = 0.0;
    for (double v : pws) sum += v;
    CHECK_NEAR(sum, m.data_log_likelihood(p), 1e-6);

    // Components all Exact.
    auto comps = m.pointwise_data_log_likelihood_components(p);
    for (const auto& c : comps)
        CHECK_TRUE(c.type() == corehydro::models::DataComponentType::Exact);
}

// ============================ Predict ============================

void test_arimax_predict() {
    TimeSeries ts = make_sample_series();

    ARIMAX m(ts);
    auto p = m.parameter_values();

    auto r0 = m.predict_components(p);
    CHECK_TRUE(r0.y.size() > 0);
    CHECK_TRUE(static_cast<int>(r0.y.size()) >= m.training_time_steps());

    // Same-seed reproducible.
    auto s1 = m.predict_components(p, 0, 12345).y;
    auto s2 = m.predict_components(p, 0, 12345).y;
    for (std::size_t i = 0; i < s1.size(); ++i) CHECK_NEAR(s1[i], s2[i], 1e-10);

    // All tuple components present with equal length.
    ARIMAX m11(ts);
    m11.set_ar_order_p(1);
    m11.set_ma_order_q(1);
    auto p11 = m11.parameter_values();
    auto rc = m11.predict_components(p11);
    std::size_t n = rc.y.size();
    CHECK_EQ(rc.intercept_part.size(), n);
    CHECK_EQ(rc.trend_part.size(), n);
    CHECK_EQ(rc.seasonality_part.size(), n);
    CHECK_EQ(rc.covariate_part.size(), n);
    CHECK_EQ(rc.ar_part.size(), n);
    CHECK_EQ(rc.ma_part.size(), n);
}

// ============================ GenerateRandomValues ============================

void test_arimax_generate_random_values() {
    TimeSeries ts = make_sample_series();
    ARIMAX m(ts);

    CHECK_EQ(static_cast<int>(m.generate_random_values(100, 12345).size()), 100);

    auto g1 = m.generate_random_values(50, 12345);
    auto g2 = m.generate_random_values(50, 12345);
    for (std::size_t i = 0; i < g1.size(); ++i) CHECK_NEAR(g1[i], g2[i], 1e-10);

    // Different seeds diverge somewhere.
    auto d1 = m.generate_random_values(50, 12345);
    auto d2 = m.generate_random_values(50, 54321);
    bool any_diff = false;
    for (std::size_t i = 0; i < d1.size(); ++i)
        if (std::fabs(d1[i] - d2[i]) > 1e-10) any_diff = true;
    CHECK_TRUE(any_diff);

    CHECK_THROWS(m.generate_random_values(0, 12345));
}

// ============================ Validation ============================

void test_arimax_validate() {
    TimeSeries ts = make_sample_series();

    ARIMAX valid(ts);
    CHECK_TRUE(valid.validate().is_valid);

    ARIMAX nullm;
    auto rn = nullm.validate();
    CHECK_TRUE(!rn.is_valid);
    CHECK_TRUE(messages_contain(rn, "Time series"));

    TimeSeries tiny = make_tiny_series();
    auto rs = ARIMAX(tiny).validate();
    CHECK_TRUE(!rs.is_valid);
    CHECK_TRUE(messages_contain(rs, "10 observations"));

    // 15-obs short series: default training steps (30) exceed the series length -> invalid.
    TimeSeries shortts = make_short_series();
    auto rsh = ARIMAX(shortts).validate();
    CHECK_TRUE(!rsh.is_valid);

    // Bad AR order.
    ARIMAX bp(ts);
    bp.set_ar_order_p(-1);
    CHECK_TRUE(!bp.validate().is_valid);

    // Non-stationary AR parameters produce a warning mentioning "stationarity".
    ARIMAX ns(ts);
    ns.set_ar_order_p(1);
    int idx = ns.include_intercept() ? 1 : 0;
    ns.parameters()[idx].set_value(1.5);
    CHECK_TRUE(messages_contain(ns.validate(), "stationarity"));
}

// Transcribes ARIMAXTests.cs's Test_{BoxCoxTransform,YeoJohnsonTransform}Transform_
// LambdaFitFailure_ReturnsValidationError (BestFit v2.0.0, f140c4d).
void test_arimax_transform_lambda_failure() {
    ARIMAX bc(make_box_cox_lambda_failure_series());
    bc.set_transform_type(Transform::BoxCox);
    auto rbc = bc.validate();
    CHECK_TRUE(!rbc.is_valid);
    CHECK_TRUE(messages_contain(rbc, "Box-Cox lambda estimation failed"));

    ARIMAX yj(make_yeo_johnson_lambda_failure_series());
    yj.set_transform_type(Transform::YeoJohnson);
    auto ryj = yj.validate();
    CHECK_TRUE(!ryj.is_valid);
    CHECK_TRUE(messages_contain(ryj, "Yeo-Johnson lambda estimation failed"));
}

// ============================ Trend / seasonality / differencing / transform ============================

void test_arimax_trend_seasonality_diff_transform() {
    TimeSeries ts = make_sample_series();

    // Streamflow with AR(1) + linear trend: valid + DataLL not -inf.
    ARIMAX sp(ts);
    sp.set_ar_order_p(1);
    sp.set_trend_type(ARIMAX::Trend::Linear);
    CHECK_TRUE(sp.validate().is_valid);
    auto psp = sp.parameter_values();
    double dsp = sp.data_log_likelihood(psp);
    CHECK_TRUE(!(std::isinf(dsp) && dsp < 0.0));

    // Monthly with AR(1) + seasonality: valid + DataLL finite.
    TimeSeries monthly = make_monthly_series();
    ARIMAX ms(monthly);
    ms.set_ar_order_p(1);
    ms.set_include_seasonality(true);
    CHECK_TRUE(ms.validate().is_valid);
    auto pms = ms.parameter_values();
    CHECK_TRUE(!std::isnan(ms.data_log_likelihood(pms)));

    // ARIMA(1,1,1) valid.
    ARIMAX a111(ts);
    a111.set_ar_order_p(1);
    a111.set_diff_order_d(1);
    a111.set_ma_order_q(1);
    CHECK_TRUE(a111.validate().is_valid);

    // First / second differencing: DataLL not NaN.
    ARIMAX d1(ts);
    d1.set_diff_order_d(1);
    auto pd1 = d1.parameter_values();
    CHECK_TRUE(!std::isnan(d1.data_log_likelihood(pd1)));

    ARIMAX d2(ts);
    d2.set_diff_order_d(2);
    auto pd2 = d2.parameter_values();
    CHECK_TRUE(!std::isnan(d2.data_log_likelihood(pd2)));

    // Log transform: DataLL not NaN.
    ARIMAX lt(ts);
    lt.set_transform_type(Transform::Logarithmic);
    auto plt = lt.parameter_values();
    CHECK_TRUE(!std::isnan(lt.data_log_likelihood(plt)));

    // No transform: DataLL not NaN.
    ARIMAX nt(ts);
    nt.set_transform_type(Transform::None);
    auto pnt = nt.parameter_values();
    CHECK_TRUE(!std::isnan(nt.data_log_likelihood(pnt)));

    // High orders can be set.
    ARIMAX ho(ts);
    ho.set_ar_order_p(5);
    ho.set_ma_order_q(3);
    CHECK_EQ(ho.ar_order_p(), 5);
    CHECK_EQ(ho.ma_order_q(), 3);
}

// ============================ Covariates ============================

void test_arimax_covariates() {
    TimeSeries ts = make_sample_series();

    ARIMAX m(ts);
    m.set_ar_order_p(1);
    m.set_ma_order_q(0);
    m.set_x_order_b(0);
    m.set_use_default_training_steps(false);
    m.set_training_time_steps(ts.count());
    m.set_default_parameters();
    int base = m.number_of_parameters();

    TimeSeries cov = make_covariate_series(ts);
    m.set_covariates(std::vector<TimeSeries>{cov});
    CHECK_TRUE(m.number_of_parameters() > base);

    // DataLogLikelihood remains finite after attaching a covariate.
    auto p = m.parameter_values();
    double d = m.data_log_likelihood(p);
    CHECK_TRUE(!std::isnan(d));
    CHECK_TRUE(!(std::isinf(d) && d > 0.0));
}

// P4 oracles (route b): exact fixed-parameter DataLogLikelihood + Residuals dumped from the REAL
// RMC.BestFit ARIMAX via tools/oracle_emitter (Numerics @ a2c4dbf, RMC-BestFit @ fc28c0c) on the
// shared 20-point fixture series. Deterministic (no fit) -> hardcoded C++-only oracles at 1e-9 abs.
// The exact MLE fit + seeded draw are oracle-verified cross-language in
// fixtures/estimation/time_series_arimax_smoke.json.
void test_arimax_p4_fixed_param_oracles() {
    // ARIMAX(1,0,0)+intercept+linear trend, params [intercept, trend, phi1, sigma] =
    // [10.9, 0.05, 0.3, 1.0]; construction mirrors model_spec.hpp's build_time_series_model.
    TimeSeries series(TimeInterval::OneDay, kT0,
                      std::vector<double>{10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0,
                                          13.1, 11.7, 12.5, 13.9, 15.2, 14.1, 12.9, 13.6, 15.0,
                                          16.2, 14.8});
    ARIMAX arimax(series);
    arimax.set_transform_type(Transform::None);
    arimax.set_include_intercept(true);
    arimax.set_trend_type(ARIMAX::Trend::Linear);
    arimax.set_ar_order_p(1);
    arimax.set_diff_order_d(0);
    arimax.set_ma_order_q(0);
    arimax.set_x_order_b(0);
    std::vector<double> p{10.9, 0.05, 0.3, 1.0};
    CHECK_NEAR(arimax.data_log_likelihood(p), -46.43434463088877, 1e-9);
    std::vector<double> res = arimax.residuals(p);
    CHECK_NEAR(res[0], 0.0, 1e-9);
    CHECK_NEAR(res[7], 1.7300000000000004, 1e-9);
}

// v2.0.0 ResetDefaultTrainingStepsForNewTimeSeries (ARIMAX.cs:591). Transcribed from the new
// TimeSeries_NewSeries_ResetsTrainingSplitToDefault regression test (which uses exactly this
// original/replacement pair): attaching a DIFFERENT response series discards the manual
// training-window edit and restores floor(0.8 * count) = 192 for the 240-observation monthly
// replacement. The manual window (40) is the C# test's own value and is not the new default.
void test_arimax_new_series_resets_training_split() {
    TimeSeries original = make_sample_series();      // 60 annual obs
    TimeSeries replacement = make_monthly_series();  // 240 monthly obs
    TimeSeries empty(TimeInterval::OneYear);

    ARIMAX m(original);
    m.set_use_default_training_steps(false);
    m.set_training_time_steps(40);
    CHECK_EQ(m.training_time_steps(), 40);
    m.set_time_series(replacement);
    CHECK_TRUE(m.use_default_training_steps());
    CHECK_EQ(m.training_time_steps(), 192);

    // Empty-series arm: attaching a series with no observations zeroes the window. Default
    // flat priors are turned OFF first only to keep the assertion on the reset itself -- the
    // trailing SetDefaultParameters call indexes the STALE differenced series at
    // TrainingTimeSteps-1 == -1, which is an IndexOutOfRangeException in C# and UB here (a
    // pre-existing shared hazard on a path the C# regression suite never drives).
    ARIMAX me(original);
    me.set_use_default_flat_priors(false);
    me.set_use_default_training_steps(false);
    me.set_training_time_steps(40);
    me.set_time_series(empty);
    CHECK_TRUE(me.use_default_training_steps());
    CHECK_EQ(me.training_time_steps(), 0);
}

// Task 21 review item 1: the empty-series path with default flat priors LEFT ON -- the arm the
// other empty-series blocks in this file deliberately dodge.
// reset_default_training_steps_for_new_time_series() zeroes TrainingTimeSteps while the STALE
// differenced series survives, so the trailing set_default_parameters() reaches the trend-scale
// block with last_idx = min(TrainingTimeSteps - 1, count - 1) = min(-1, count - 1) = -1.
// TimeSeries::operator[] is unchecked, so before the clamp that read ordinates_[SIZE_MAX] --
// undefined behavior, and this test aborted (exit 139) rather than failing an assertion. C#
// raises IndexOutOfRangeException on the same expression (ARIMAX.cs:929). With the clamp the call
// simply completes with a zeroed training window; see the divergence note at the clamp.
void test_arimax_empty_series_with_default_priors_is_index_safe() {
    ARIMAX m(make_sample_series());
    CHECK_TRUE(m.use_default_flat_priors());  // deliberately NOT disabled -- that is the point
    m.set_time_series(TimeSeries(TimeInterval::OneYear));
    CHECK_TRUE(m.use_default_training_steps());
    CHECK_EQ(m.training_time_steps(), 0);

    // Trend ON drives the clamped index all the way through the delta1/delta2/delta3 trend-parameter
    // scales, the only consumer of the clamped read.
    ARIMAX t(make_sample_series());
    t.set_trend_type(ARIMAX::Trend::Linear);
    t.set_time_series(TimeSeries(TimeInterval::OneYear));
    CHECK_EQ(t.training_time_steps(), 0);

    // A manual window on top of the same path (use_default_training_steps false -> reset forces it
    // back on and zeroes the window), still with default flat priors on.
    ARIMAX u(make_sample_series());
    u.set_use_default_training_steps(false);
    u.set_training_time_steps(40);
    u.set_time_series(TimeSeries(TimeInterval::OneYear));
    CHECK_TRUE(u.use_default_training_steps());
    CHECK_EQ(u.training_time_steps(), 0);
}

// v2.0.0 InferSeasonalPeriod: the OneYear cycle length changed 1 -> 10 (ARIMAX.cs:880), so annual
// records now carry a DECADAL Fourier cycle instead of a degenerate one-step cycle. Transcribed
// from the new MonthlyTimeSeries_SeasonalPeriod_IsTwelve / AnnualTimeSeries_SeasonalPeriod_IsTen /
// AnnualSeasonality_FourierTerms_AreNonDegenerate regression tests. At the old S = 1 the Fourier
// pair collapses (sin(2*pi*t) == 0, cos(2*pi*t) == 1 at integer t), so the seasonality part is
// flat and the range check below fails -- that is the RED/GREEN discriminator.
void test_arimax_annual_seasonal_period() {
    CHECK_EQ(ARIMAX(make_monthly_series()).seasonal_period(), 12);
    CHECK_EQ(ARIMAX(make_sample_series()).seasonal_period(), 10);

    ARIMAX annual(make_sample_series());
    annual.set_include_seasonality(true);
    std::vector<double> p = annual.parameter_values();
    int sin_index = -1, cos_index = -1;
    for (std::size_t i = 0; i < annual.parameters().size(); ++i) {
        const std::string& name = annual.parameters()[i].name();
        if (name.rfind("Seasonality Sin", 0) == 0) sin_index = static_cast<int>(i);
        if (name.rfind("Seasonality Cos", 0) == 0) cos_index = static_cast<int>(i);
    }
    CHECK_TRUE(sin_index >= 0 && cos_index >= 0);
    p[static_cast<std::size_t>(sin_index)] = 1.0;
    p[static_cast<std::size_t>(cos_index)] = 0.0;

    ARIMAX::PredictResult result = annual.predict_components(p, 0, -1);
    double lo = result.seasonality_part[0], hi = result.seasonality_part[0];
    for (int t = 0; t < 10; ++t) {
        lo = std::min(lo, result.seasonality_part[static_cast<std::size_t>(t)]);
        hi = std::max(hi, result.seasonality_part[static_cast<std::size_t>(t)]);
    }
    CHECK_TRUE(hi - lo > 1.0);
}

// v2.0.0 TimeInterval.Irregular Validate guard (ARIMAX.cs:2016). Transcribed from the new
// Test_Validate_IrregularTimeSeries_ReturnsFalse regression test; the cross-language oracle lives
// in fixtures/estimation/time_series_irregular_interval.json.
void test_arimax_irregular_interval_validate() {
    // C# `new TimeSeries(TimeInterval.Irregular)` + Add: the (interval, start, end) ctor
    // rejects Irregular in BOTH C# and this port, so build it ordinate-by-ordinate with the
    // C# test's own i*i+1 index spacing.
    TimeSeries irregular(TimeInterval::Irregular);
    for (int i = 0; i < 50; ++i)
        irregular.add(TimeSeries::Ordinate(kT0.add_days(static_cast<double>(i) * i + 1), i + 1.0));

    auto r = ARIMAX(irregular).validate();
    CHECK_TRUE(!r.is_valid);
    CHECK_TRUE(messages_contain(r, "regular time interval"));

    CHECK_TRUE(ARIMAX(make_sample_series()).validate().is_valid);
}

}  // namespace

int main() {
    test_arimax_constructors();
    test_arimax_properties();
    test_arimax_number_of_parameters();
    test_arimax_parameter_names_bounds();
    test_arimax_set_parameter_values();
    test_arimax_log_likelihood();
    test_arimax_predict();
    test_arimax_generate_random_values();
    test_arimax_validate();
    test_arimax_transform_lambda_failure();
    test_arimax_trend_seasonality_diff_transform();
    test_arimax_covariates();

    test_arimax_new_series_resets_training_split();
    test_arimax_empty_series_with_default_priors_is_index_safe();
    test_arimax_annual_seasonal_period();
    test_arimax_irregular_interval_validate();

    test_arimax_p4_fixed_param_oracles();

    return chtest::summary("arimax");
}
