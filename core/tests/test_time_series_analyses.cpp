// Structural / behavioral tests for the TimeSeries analyses (D2):
//   corehydro::analyses::ARAnalysis, MAAnalysis, ARIMAAnalysis, ARIMAXAnalysis.
//
// These transcribe the STRUCTURAL C# tests from
//   RMC.BestFit.Tests/TimeSeriesAnalysis/ARAnalysisTests.cs
//   RMC.BestFit.Tests/TimeSeriesAnalysis/MAAnalysisTests.cs
//   RMC.BestFit.Tests/TimeSeriesAnalysis/ARIMAAnalysisTests.cs
//   RMC.BestFit.Tests/TimeSeriesAnalysis/ARIMAXAnalysisTests.cs
// There are NO numeric MCMC oracles for these analyses (the real parity lives in the absent
// RMC.BestFit.Verification project); every assertion here is a constructor-validity check, a
// same-instance identity, a config default, a model-order property, a ForecastingTimeSteps clamp,
// a Validate() outcome, or a return-null / is_estimated()==false-when-unestimated check. TimeSeries
// data fixtures are regenerated with the ported bit-exact Mersenne Twister (the C# fixtures draw
// from System.Random, so the exact data VALUES are unreproducible -- structural shape matches).
//
// SKIPPED / ADAPTED C# test methods (reasons in task-D2-report.md):
//   * XmlSerialization_RoundTrip_PreservesConfiguration, XmlSerialization_NullXElement_*,
//     ToXElement_CreatesValidStructure, XmlSerialization_PreservesBayesianSettings -- XML
//     (de)serialization surface, dropped project-wide. The round-trip config-preservation intent
//     is covered by the ForecastingTimeSteps set/get + clamp tests instead.
//   * PropertyChange_ForecastingTimeSteps_RaisesPropertyChanged,
//     PropertyChange_SameValue_DoesNotRaisePropertyChanged -- INotifyPropertyChanged; no
//     notification system in this port (the clamp/dedup logic they lean on IS ported and tested
//     via the set/get + clamp checks).
#include <algorithm>
#include <memory>
#include <vector>

#include "corehydro/analyses/time_series/ar_analysis.hpp"
#include "corehydro/analyses/time_series/arima_analysis.hpp"
#include "corehydro/analyses/time_series/arimax_analysis.hpp"
#include "corehydro/analyses/time_series/ma_analysis.hpp"
#include "corehydro/models/time_series/arima.hpp"
#include "corehydro/models/time_series/arimax.hpp"
#include "corehydro/models/time_series/auto_regressive.hpp"
#include "corehydro/models/time_series/moving_average.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "check.hpp"

using corehydro::analyses::ARAnalysis;
using corehydro::analyses::ARIMAAnalysis;
using corehydro::analyses::ARIMAXAnalysis;
using corehydro::analyses::MAAnalysis;
using corehydro::models::ARIMA;
using corehydro::models::ARIMAX;
using corehydro::models::AutoRegressive;
using corehydro::models::MovingAverage;
using corehydro::numerics::data::TimeInterval;
using corehydro::numerics::data::TimeSeries;
using corehydro::numerics::sampling::MersenneTwister;

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



// ---- Deterministic fixtures (ported Mersenne Twister is bit-exact) ----------------------

// 60-observation annual streamflow, AR(1)-ish (mean ~5000, phi ~0.6, sigma ~600). Mirrors the
// C# CreateAnnualStreamflowTimeSeries shape; data regenerated with the ported RNG.
TimeSeries annual_streamflow() {
    TimeSeries ts(TimeInterval::OneYear, kT0, t0_plus(59, TimeInterval::OneYear));
    MersenneTwister rng(12345);
    const double mean = 5000.0, phi = 0.6, sigma = 600.0;
    double prev = mean;
    for (int i = 0; i < ts.count(); ++i) {
        double innovation = (rng.next_double() * 2.0 - 1.0) * sigma;
        double value = std::max(100.0, mean + phi * (prev - mean) + innovation);
        ts[i].set_value(value);
        prev = value;
    }
    return ts;
}

std::unique_ptr<AutoRegressive> make_ar_model(int order = 1) {
    return std::make_unique<AutoRegressive>(annual_streamflow(), order);
}
std::unique_ptr<MovingAverage> make_ma_model(int order = 1) {
    return std::make_unique<MovingAverage>(annual_streamflow(), order);
}
std::unique_ptr<ARIMA> make_arima_model(int p = 1, int d = 0, int q = 1) {
    return std::make_unique<ARIMA>(annual_streamflow(), p, d, q);
}
std::unique_ptr<ARIMAX> make_arimax_model() {
    return std::make_unique<ARIMAX>(annual_streamflow());
}

// =========================================================================================
// ARAnalysis
// =========================================================================================

// C# Test_Constructor_ValidModel_InitializesCorrectly (+ ForecastingTimeSteps default,
// IsEstimated false, AnalysisResults null).
void test_ar_constructor_initializes() {
    auto model = make_ar_model(1);
    AutoRegressive* raw = model.get();
    ARAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.auto_regressive() == raw);
    CHECK_EQ(analysis.forecasting_time_steps(), 0);
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

// C# Test_Constructor_NullModel_ThrowsArgumentNullException.
void test_ar_null_model_throws() {
    CHECK_THROWS(ARAnalysis(std::unique_ptr<AutoRegressive>{}));
}

// C# Test_Constructor_BayesianAnalysis_HasCorrectModel.
void test_ar_bayesian_has_correct_model() {
    auto model = make_ar_model(1);
    AutoRegressive* raw = model.get();
    ARAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.bayesian_analysis().model() == raw);
}

// C# Test_Validate_ValidConfiguration_ReturnsTrue / Test_EdgeCase_HigherOrderAR / NoIntercept.
void test_ar_validate_valid() {
    ARAnalysis analysis(make_ar_model(1));
    CHECK_TRUE(analysis.validate().is_valid);
    ARAnalysis higher(make_ar_model(5));
    CHECK_TRUE(higher.validate().is_valid);
}

// C# Test_Validate_InvalidModel_PropagatesModelValidation (no time series).
void test_ar_validate_invalid_model() {
    ARAnalysis analysis(std::make_unique<AutoRegressive>());
    CHECK_TRUE(!analysis.validate().is_valid);
}

// C# Test_ForecastingTimeSteps_SetAndGet + the [0,100] setter clamp (Validate rejects <0 / >100).
void test_ar_forecasting_clamp() {
    ARAnalysis analysis(make_ar_model(1));
    analysis.set_forecasting_time_steps(20);
    CHECK_EQ(analysis.forecasting_time_steps(), 20);
    analysis.set_forecasting_time_steps(-5);  // clamps up to 0
    CHECK_EQ(analysis.forecasting_time_steps(), 0);
    analysis.set_forecasting_time_steps(150);  // clamps down to 100
    CHECK_EQ(analysis.forecasting_time_steps(), 100);
}

// C# Test_ClearResults_ResetsAllResults.
void test_ar_clear_results() {
    ARAnalysis analysis(make_ar_model(1));
    analysis.clear_results();
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

// C# Test_AutoRegressive_ReturnsCorrectModelReference (model-order property).
void test_ar_model_order() {
    ARAnalysis analysis(make_ar_model(2));
    CHECK_EQ(analysis.auto_regressive().order(), 2);
}

// C# Test_BayesianAnalysis_Iterations_CanBeConfigure + HasDefaultSettings.
void test_ar_bayesian_config() {
    ARAnalysis analysis(make_ar_model(1));
    CHECK_TRUE(analysis.bayesian_analysis().iterations() > 0);
    analysis.bayesian_analysis().set_iterations(10000);
    CHECK_EQ(analysis.bayesian_analysis().iterations(), 10000);
}

// =========================================================================================
// MAAnalysis
// =========================================================================================

void test_ma_constructor_initializes() {
    auto model = make_ma_model(1);
    MovingAverage* raw = model.get();
    MAAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.moving_average() == raw);
    CHECK_EQ(analysis.forecasting_time_steps(), 0);
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

void test_ma_null_model_throws() {
    CHECK_THROWS(MAAnalysis(std::unique_ptr<MovingAverage>{}));
}

void test_ma_bayesian_has_correct_model() {
    auto model = make_ma_model(1);
    MovingAverage* raw = model.get();
    MAAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.bayesian_analysis().model() == raw);
}

void test_ma_validate_valid() {
    MAAnalysis analysis(make_ma_model(1));
    CHECK_TRUE(analysis.validate().is_valid);
}

void test_ma_validate_invalid_model() {
    MAAnalysis analysis(std::make_unique<MovingAverage>());
    CHECK_TRUE(!analysis.validate().is_valid);
}

void test_ma_forecasting_clamp() {
    MAAnalysis analysis(make_ma_model(1));
    analysis.set_forecasting_time_steps(20);
    CHECK_EQ(analysis.forecasting_time_steps(), 20);
    analysis.set_forecasting_time_steps(-5);
    CHECK_EQ(analysis.forecasting_time_steps(), 0);
    analysis.set_forecasting_time_steps(150);
    CHECK_EQ(analysis.forecasting_time_steps(), 100);
}

void test_ma_clear_results() {
    MAAnalysis analysis(make_ma_model(1));
    analysis.clear_results();
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

// C# Test_MovingAverage_ReturnsCorrectModelReference: analysis.MovingAverage.Order == 2.
void test_ma_model_order() {
    MAAnalysis analysis(make_ma_model(2));
    CHECK_EQ(analysis.moving_average().order(), 2);
}

void test_ma_bayesian_config() {
    MAAnalysis analysis(make_ma_model(1));
    CHECK_TRUE(analysis.bayesian_analysis().iterations() > 0);
    analysis.bayesian_analysis().set_iterations(10000);
    CHECK_EQ(analysis.bayesian_analysis().iterations(), 10000);
}

// =========================================================================================
// ARIMAAnalysis
// =========================================================================================

void test_arima_constructor_initializes() {
    auto model = make_arima_model(1, 0, 1);
    ARIMA* raw = model.get();
    ARIMAAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.arima() == raw);
    CHECK_EQ(analysis.forecasting_time_steps(), 0);
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

void test_arima_null_model_throws() {
    CHECK_THROWS(ARIMAAnalysis(std::unique_ptr<ARIMA>{}));
}

void test_arima_bayesian_has_correct_model() {
    auto model = make_arima_model();
    ARIMA* raw = model.get();
    ARIMAAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.bayesian_analysis().model() == raw);
}

void test_arima_validate_valid() {
    ARIMAAnalysis analysis(make_arima_model(1, 0, 1));
    CHECK_TRUE(analysis.validate().is_valid);
}

void test_arima_validate_invalid_model() {
    ARIMAAnalysis analysis(std::make_unique<ARIMA>());
    CHECK_TRUE(!analysis.validate().is_valid);
}

void test_arima_forecasting_clamp() {
    ARIMAAnalysis analysis(make_arima_model());
    analysis.set_forecasting_time_steps(20);
    CHECK_EQ(analysis.forecasting_time_steps(), 20);
    analysis.set_forecasting_time_steps(-5);
    CHECK_EQ(analysis.forecasting_time_steps(), 0);
    analysis.set_forecasting_time_steps(150);
    CHECK_EQ(analysis.forecasting_time_steps(), 100);
}

void test_arima_clear_results() {
    ARIMAAnalysis analysis(make_arima_model());
    analysis.clear_results();
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

// C# Test_ARIMA_ReturnsCorrectModelReference: POrder == 2, QOrder == 1.
void test_arima_model_order() {
    ARIMAAnalysis analysis(make_arima_model(2, 1, 1));
    CHECK_EQ(analysis.arima().p_order(), 2);
    CHECK_EQ(analysis.arima().q_order(), 1);
    CHECK_EQ(analysis.arima().d_order(), 1);
}

void test_arima_bayesian_config() {
    ARIMAAnalysis analysis(make_arima_model());
    CHECK_TRUE(analysis.bayesian_analysis().iterations() > 0);
    analysis.bayesian_analysis().set_iterations(10000);
    CHECK_EQ(analysis.bayesian_analysis().iterations(), 10000);
}

// =========================================================================================
// ARIMAXAnalysis
// =========================================================================================

void test_arimax_constructor_initializes() {
    auto model = make_arimax_model();
    ARIMAX* raw = model.get();
    ARIMAXAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.arimax() == raw);
    CHECK_EQ(analysis.forecasting_time_steps(), 0);
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

void test_arimax_null_model_throws() {
    CHECK_THROWS(ARIMAXAnalysis(std::unique_ptr<ARIMAX>{}));
}

void test_arimax_bayesian_has_correct_model() {
    auto model = make_arimax_model();
    ARIMAX* raw = model.get();
    ARIMAXAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.bayesian_analysis().model() == raw);
}

void test_arimax_validate_valid() {
    ARIMAXAnalysis analysis(make_arimax_model());
    CHECK_TRUE(analysis.validate().is_valid);
}

void test_arimax_validate_invalid_model() {
    ARIMAXAnalysis analysis(std::make_unique<ARIMAX>());
    CHECK_TRUE(!analysis.validate().is_valid);
}

void test_arimax_forecasting_clamp() {
    ARIMAXAnalysis analysis(make_arimax_model());
    analysis.set_forecasting_time_steps(20);
    CHECK_EQ(analysis.forecasting_time_steps(), 20);
    analysis.set_forecasting_time_steps(-1);  // C# Validate_WithNegativeForecastingSteps clamps to 0
    CHECK_EQ(analysis.forecasting_time_steps(), 0);
    analysis.set_forecasting_time_steps(150);
    CHECK_EQ(analysis.forecasting_time_steps(), 100);
}

void test_arimax_clear_results() {
    ARIMAXAnalysis analysis(make_arimax_model());
    analysis.clear_results();
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

// C# WithARComponent / WithMAComponent / WithDifferencing: AROrderP / MAOrderQ / DiffOrderD.
void test_arimax_model_orders() {
    auto model = make_arimax_model();
    model->set_ar_order_p(2);
    model->set_ma_order_q(1);
    model->set_diff_order_d(1);
    ARIMAXAnalysis analysis(std::move(model));
    CHECK_EQ(analysis.arimax().ar_order_p(), 2);
    CHECK_EQ(analysis.arimax().ma_order_q(), 1);
    CHECK_EQ(analysis.arimax().diff_order_d(), 1);
}

void test_arimax_bayesian_config() {
    ARIMAXAnalysis analysis(make_arimax_model());
    CHECK_TRUE(analysis.bayesian_analysis().iterations() > 0);
    analysis.bayesian_analysis().set_iterations(10000);
    CHECK_EQ(analysis.bayesian_analysis().iterations(), 10000);
}

}  // namespace

int main() {
    // ARAnalysis
    test_ar_constructor_initializes();
    test_ar_null_model_throws();
    test_ar_bayesian_has_correct_model();
    test_ar_validate_valid();
    test_ar_validate_invalid_model();
    test_ar_forecasting_clamp();
    test_ar_clear_results();
    test_ar_model_order();
    test_ar_bayesian_config();

    // MAAnalysis
    test_ma_constructor_initializes();
    test_ma_null_model_throws();
    test_ma_bayesian_has_correct_model();
    test_ma_validate_valid();
    test_ma_validate_invalid_model();
    test_ma_forecasting_clamp();
    test_ma_clear_results();
    test_ma_model_order();
    test_ma_bayesian_config();

    // ARIMAAnalysis
    test_arima_constructor_initializes();
    test_arima_null_model_throws();
    test_arima_bayesian_has_correct_model();
    test_arima_validate_valid();
    test_arima_validate_invalid_model();
    test_arima_forecasting_clamp();
    test_arima_clear_results();
    test_arima_model_order();
    test_arima_bayesian_config();

    // ARIMAXAnalysis
    test_arimax_constructor_initializes();
    test_arimax_null_model_throws();
    test_arimax_bayesian_has_correct_model();
    test_arimax_validate_valid();
    test_arimax_validate_invalid_model();
    test_arimax_forecasting_clamp();
    test_arimax_clear_results();
    test_arimax_model_orders();
    test_arimax_bayesian_config();

    return chtest::summary("time_series_analyses");
}
