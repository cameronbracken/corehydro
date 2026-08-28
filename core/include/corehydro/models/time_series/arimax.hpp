// ported from: upstream/RMC-BestFit/src/RMC.BestFit/Models/TimeSeries/ARIMAX.cs @ c2e6192
//
// v2.0.0 (upstream-sync Task 16, f140c4d + 0d6821d): SetTrainingData's BoxCox/YeoJohnson
// branches (Step 1, transforming the whole raw series) now guard against a non-finite fitted
// lambda (Task 2's hardened fit_lambda already returns NaN instead of throwing for a
// degenerate/unsupported sample -- see box_cox.hpp / yeo_johnson.hpp). On failure:
// lambda_/log_jacobian_ reset to 0, transformed_time_series_/diff_series_/training_time_series_
// ALL replaced with an empty series, transform_fit_validation_message_ set to the C#-exact
// message text, and the branch returns early (Steps 2-4 -- differencing, the training-series
// slice, and the raw-value log-jacobian -- never run). Validate appends that message (and flips
// is_valid false) as the LAST check, mirroring the C# ordering. Deliberately NOT ported: the C#
// `catch (ArithmeticException)` branch around FitLambda (with its "Solver message: ..." suffix)
// has no C++ analog -- the ported fit_lambda never throws, it always returns a double (possibly
// NaN), so only the C# `!double.IsFinite(_lambda)` branch (the plain message, no solver-text
// suffix) is reachable here. Also SKIPPED (XML/GUI-only, no oracle-visible C++ surface, not in
// this task's scope): the XElement ctor's TrainingTimeSteps/UseDefaultTrainingSteps attribute
// restoration + explicit SetTrainingData() call. Task 21 CLOSED the three v2.0.0 deltas this note
// previously deferred -- ResetDefaultTrainingStepsForNewTimeSeries (the TimeSeries-setter
// training-window reset), the TimeInterval.Irregular Validate guard, and the InferSeasonalPeriod
// OneYear cycle-length change (1 -> 10, ARIMAX.cs:880) -- all three are model-layer library
// surface reachable from the public setters, not GUI/XML, and all three are ported below.
//
// AutoRegressive Integrated Moving Average with eXogenous variables (ARIMAX). The richest
// TimeSeries ModelBase family: it extends the ARIMA (T2) differenced + transformed mean with
//   - an INLINE polynomial trend        gamma1*t + gamma2*t^2 + gamma3*t^3   (Trend enum),
//   - an INLINE Fourier seasonality     psi1*sin(2*pi*t/S) + psi2*cos(2*pi*t/S),
//   - INLINE exogenous covariate regression  sum_i (beta_i0*X_i[t] + sum_j beta_ij*X_i[t-j]).
// The full mean function evaluated on the transformed + differenced scale is:
//   mu + trend(t) + seasonality(t) + sum_i beta_i.X_i(t) + AR(mean-centered) + MA(CSS),
//   eps(t) ~ N(0, sigma^2).
// Built on ModelBase (the shared likelihood surface the Estimation layer optimizes/samples) and
// ISimulatable<std::vector<double>> (C# ISimulatable<double[]>). Supports None/Logarithmic/
// Box-Cox/Yeo-Johnson data transforms, differencing of order d, a training/forecasting split,
// and the optional Jeffreys (1/sigma) scale prior.
//
// Structural mirroring: the class/member/method layout follows ARIMAX.cs. This file mirrors the
// T2 ARIMA header structure (transform + difference machinery, ARMA-on-differenced core, the
// anchor-chain integration in Predict) and adds the three inline mean-function extensions plus
// exogenous covariates. The never-mutate rule is RELAXED for these stateful model objects (per
// .claude/CLAUDE.md), matching the upstream mutable WPF-binding design. Mirrored copies of the
// shared plumbing are the sanctioned choice here (this project favors STRUCTURAL MIRRORING over
// premature helper extraction).
//
// KEY DIFFERENCES FROM ARIMA (C# governs, verified against ARIMAX.cs):
//   - maxOrder = max(AROrderP, MAOrderQ, XOrderB) in the likelihood path (ARIMAX.cs:1098/1261);
//     the AR mean-centering uses the FULL mean[t-p] (trend/seasonality/covariate included), not
//     just mu, and the AR block is gated on t >= AROrderP (not t >= maxOrder) (ARIMAX.cs:1362).
//   - SetTrainingData transforms the WHOLE raw series (_transformedTimeSeries), then differences
//     via the P2 TimeSeries.Difference(1, d) (NOT ARIMA's clone-based private method); the
//     training series is the first effectiveTrainingSteps of _diffSeries (ARIMAX.cs:588).
//   - DataLogLikelihood has NO "-inf when no series" guard: with no TimeSeries, TrainingTimeSteps
//     is 0 so Residuals is empty and DataLogLikelihood returns 0.0 (ARIMAX.cs:1082; verified by
//     ARIMAXTests.cs:453 Test_DataLogLikelihood_NullTimeSeries_ReturnsZero). This DIVERGES from
//     ARIMA's -inf sentinel -- ARIMAX's C# behavior governs.
//   - No public IsStationary / IsInvertible: stationarity/invertibility appear ONLY as Validate()
//     warnings when sum|.| >= 1 for the AR/MA blocks, indexing the flat parameter list through the
//     GetARParameterStartIndex()/GetMAParameterStartIndex() helpers (ARIMAX.cs:2073/2097).
//
// THE COVARIATE FORECAST-TAIL EXTENSION landed in P6, un-gated by the ported TimeSeries
// container. The nested CovariateExtensionMethod enum (None/BlockBootstrap/KNN, ARIMAX.cs:175)
// now drives real covariate resampling in both places C# does: Predict (C# 1558-1650) and the
// forecast branch of GenerateRandomValues (C# 2333-2378). Through P5 both sites threw a
// "deferred" error when the horizon ran past the observed covariates; they no longer do.
// Three details the two sites do NOT share, transcribed as written: Predict keeps the observed
// span EXACTLY as observed and extends only the tail (resampling the observed period would
// inflate the training-period credible interval, since beta * X[t] would move every
// realization); Predict has a DETERMINISTIC branch that appends each covariate's mean when no
// seed was given, and GenerateRandomValues has none; and their resample-seed offsets differ
// (+1000 vs +2000). CovariateExtensionMethod defaults to BlockBootstrap (C# field initializer,
// ARIMAX.cs:198).
//
// Deliberately NOT ported (documented; C# governs -- deviations noted in task-T3-report.md):
//   - XML (ToXElement / the XElement ctor, ARIMAX.cs:72-128,1852): project-wide non-port.
//   - INotifyPropertyChanged / RaisePropertyChange / Parameter_PropertyChanged /
//     TimeSeries_CollectionChanged: WPF data-binding plumbing, ported as silent no-ops
//     (parameters_ is a plain std::vector; no change notification is threaded through it).
//   - IModel Clone() (ARIMAX.cs:1819): the C++ core has no virtual IModel::Clone (see
//     model_base.hpp) and no T3 fit path needs a clone, so it is omitted (T1/T2/S4 precedent).
//   - GenerateRandomSeries (ARIMAX.cs:1792): returns a date-stamped TimeSeries rather than the
//     bare vector generate_random_values (the ISimulatable member) returns. Not ported: no caller
//     in scope needs the dates, and the container it would wrap them in is now ported, so this is
//     a one-method addition whenever one does.
//   - AppendTail / AppendConstantTail (ARIMAX.cs:799/819) were unported through P5 because nothing
//     called them; P6's forecast-tail extension does, so both are ported below.
//
// RNG deviation (documented): C# Predict() draws its stochastic-forecast noise from System.Random
// (a .NET LCG with no ported equivalent); this port substitutes the ported MersenneTwister.
// Same-seed reproducibility and different-seed divergence hold, but the exact seeded forecast
// VALUES are not C#-reproducible. The exact forecast oracles are a P4 concern.
// GenerateRandomValues uses the ported MersenneTwister exactly as the C# does (bit-exact).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/support/data_component.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/support/subscript_formatter.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/models/time_series/transform_type.hpp"
#include "corehydro/numerics/data/box_cox.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "corehydro/numerics/data/yeo_johnson.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class ARIMAX : public ModelBase, public ISimulatable<std::vector<double>> {
   public:
    using TimeSeries = numerics::data::TimeSeries;

    // Trend types for the deterministic polynomial trend (ARIMAX.cs:158).
    enum class Trend { None, Linear, Quadratic, Cubic };

    // Methods for extending covariate data beyond available observations (ARIMAX.cs:175).
    // DEFERRED with the heavy TimeSeries container -- see the file header. The enum is ported for
    // structural parity; BlockBootstrap/KNN resampling is not.
    enum class CovariateExtensionMethod { None, BlockBootstrap, KNN };

    // Prediction decomposition (C# tuple (Y, InterceptPart, TrendPart, SeasonalityPart,
    // CovariatePart, ARPart, MAPart)).
    struct PredictResult {
        std::vector<double> y;
        std::vector<double> intercept_part;
        std::vector<double> trend_part;
        std::vector<double> seasonality_part;
        std::vector<double> covariate_part;
        std::vector<double> ar_part;
        std::vector<double> ma_part;
    };

    // --- Construction (ARIMAX.cs:48-65). ---

    // Empty ARIMAX with default AR(1) order and intercept (C#:51).
    ARIMAX() { set_default_parameters(); }

    // ARIMAX over the given time series (C#:60). Setting the series triggers the same
    // InferSeasonalPeriod/SetDefaultTrainingSteps/SetTrainingData cascade as the C# setter, then
    // the ctor body re-runs SetTrainingData()/SetDefaultParameters().
    explicit ARIMAX(const TimeSeries& time_series) {
        set_time_series(time_series);
        set_training_data();
        set_default_parameters();
    }

    // --- Properties. ---

    bool has_time_series() const { return time_series_.has_value(); }
    const TimeSeries& time_series() const { return *time_series_; }
    void set_time_series(const TimeSeries& value) {
        // C# unsubscribes/subscribes CollectionChanged here -> no-op in this port.
        time_series_ = value;
        seasonal_period_ = infer_seasonal_period();  // C# UpdateSeasonalPeriod().
        reset_default_training_steps_for_new_time_series();
        set_training_data();
        // RaisePropertyChange -> no-op.
        if (use_default_flat_priors()) set_default_parameters();
    }

    Transform transform_type() const { return transform_type_; }
    void set_transform_type(Transform value) {
        if (transform_type_ != value) {
            transform_type_ = value;
            set_training_data();
            set_default_parameters();
        }
    }

    // Transformed + differenced training series used for calibration (C# TrainingTimeSeries).
    bool has_training_time_series() const { return training_time_series_.has_value(); }
    const TimeSeries& training_time_series() const { return *training_time_series_; }

    // Differenced (but training-window-trimmed only via _diffSeries) series (C# DifferencedSeries).
    bool has_differenced_series() const { return diff_series_.has_value(); }
    const TimeSeries& differenced_series() const { return *diff_series_; }

    // Exogenous regression covariates (C# Covariates).
    bool has_covariates() const { return covariates_.has_value(); }
    const std::vector<TimeSeries>& covariates() const { return *covariates_; }

    CovariateExtensionMethod covariate_extension() const { return covariate_extension_; }
    void set_covariate_extension(CovariateExtensionMethod value) {
        if (covariate_extension_ != value) covariate_extension_ = value;
    }

    // Seasonal period inferred from the time-series interval (C# SeasonalPeriod).
    int seasonal_period() const { return seasonal_period_; }

    bool include_intercept() const { return include_intercept_; }
    void set_include_intercept(bool value) {
        if (include_intercept_ != value) {
            include_intercept_ = value;
            set_default_parameters();
        }
    }

    bool include_seasonality() const { return include_seasonality_; }
    void set_include_seasonality(bool value) {
        if (include_seasonality_ != value) {
            include_seasonality_ = value;
            set_default_parameters();
        }
    }

    Trend trend_type() const { return trend_type_; }
    void set_trend_type(Trend value) {
        if (trend_type_ != value) {
            trend_type_ = value;
            set_default_parameters();
        }
    }

    int ar_order_p() const { return ar_order_p_; }
    void set_ar_order_p(int value) {
        if (ar_order_p_ != value) {
            ar_order_p_ = value;
            set_default_parameters();
        }
    }

    // DiffOrderD setter re-runs SetTrainingData() (re-differences _diffSeries) before
    // SetDefaultParameters (ARIMAX.cs:406-419).
    int diff_order_d() const { return diff_order_d_; }
    void set_diff_order_d(int value) {
        if (diff_order_d_ != value) {
            diff_order_d_ = value;
            set_training_data();
            set_default_parameters();
        }
    }

    int ma_order_q() const { return ma_order_q_; }
    void set_ma_order_q(int value) {
        if (ma_order_q_ != value) {
            ma_order_q_ = value;
            set_default_parameters();
        }
    }

    int x_order_b() const { return x_order_b_; }
    void set_x_order_b(int value) {
        if (x_order_b_ != value) {
            x_order_b_ = value;
            set_default_parameters();
        }
    }

    bool use_jeffreys_rule_for_scale() const { return use_jeffreys_rule_for_scale_; }
    void set_use_jeffreys_rule_for_scale(bool value) {
        if (use_jeffreys_rule_for_scale_ != value) use_jeffreys_rule_for_scale_ = value;
    }

    int training_time_steps() const { return training_time_steps_; }
    void set_training_time_steps(int value) {
        if (training_time_steps_ != value) {
            training_time_steps_ = value;
            set_training_data();
            if (use_default_flat_priors()) set_default_parameters();
        }
    }

    bool use_default_training_steps() const { return use_default_training_steps_; }
    void set_use_default_training_steps(bool value) {
        if (use_default_training_steps_ != value) {
            use_default_training_steps_ = value;
            if (use_default_training_steps_ && time_series_) set_default_training_steps();
        }
    }

    // Number of steps reserved for out-of-sample forecasting.
    int forecasting_time_steps() const {
        return time_series_ ? time_series_->count() - training_time_steps_ : 0;
    }

    // Sets the list of exogenous covariate time series (C#:559). Re-runs SetDefaultParameters.
    void set_covariates(const std::vector<TimeSeries>& covariates) {
        covariates_ = covariates;
        // RaisePropertyChange -> no-op.
        set_default_parameters();
    }

    // Sets the transform parameters manually (C#:676).
    void set_transform_parameters(double lambda1 = 0, double lambda2 = 0) {
        lambda_ = lambda1;
        lambda2_ = lambda2;
    }

    // Convenience: current parameter values (C# Parameters.Select(x => x.Value).ToArray()).
    std::vector<double> parameter_values() const {
        std::vector<double> v;
        v.reserve(parameters_.size());
        for (const auto& p : parameters_) v.push_back(p.value());
        return v;
    }

    // --- SetDefaultParameters (C#:773). Data-driven bounds from _diffSeries when non-empty. ---
    void set_default_parameters() override {
        int n_steps = 0;
        double mean = 0, min = -10, max = 10;
        double range = max - min;
        double delta1 = 0, delta2 = 0, delta3 = 0;
        double sigma = 1;
        double sigma_lb = numerics::kDoubleMachineEpsilon;
        double sigma_ub = 10;

        if (diff_series_ && diff_series_->count() > 0) {
            n_steps = training_time_steps_;
            mean = diff_series_->mean_value();
            sigma = diff_series_->standard_deviation();

            double sign = (mean > 0) - (mean < 0);
            double temp_min = sign * std::pow(10.0, std::floor(std::log10(std::fabs(mean)) - 1.0));
            double temp_max = sign * std::pow(10.0, std::ceil(std::log10(std::fabs(mean)) + 1.0));
            min = std::min(temp_min, temp_max);
            max = std::max(temp_min, temp_max);

            if (std::isnan(min) || std::isinf(min)) min = -1000;
            if (std::isnan(max) || std::isinf(max)) max = 1000;
            if (min >= max) {
                min = mean - 100;
                max = mean + 100;
            }

            range = max - min;

            // Trend parameter scales (ARIMAX.cs:929).
            //
            // The `std::max(0, ...)` clamp is a C++-only memory-safety guard with NO effect on
            // any path where the C# expression is itself well-defined. C# evaluates
            // `_diffSeries[Math.Min(TrainingTimeSteps - 1, _diffSeries.Count - 1)]` unguarded;
            // when TrainingTimeSteps is 0 the index is -1 and C# raises
            // IndexOutOfRangeException. `TimeSeries::operator[]` here is UNCHECKED (see
            // time_series.hpp), so the same -1 would index `ordinates_[SIZE_MAX]` -- undefined
            // behavior, not an exception. Documented-divergence precedent: the guarded
            // truncating casts at chi_squared.hpp / binomial.hpp / inverse_chi_squared.hpp.
            //
            // Reachability: TrainingTimeSteps is 0 while `diff_series_` is non-empty exactly when
            // an EMPTY series is attached over a model that already had one --
            // reset_default_training_steps_for_new_time_series() zeroes the window while
            // set_training_data() returns early and leaves the previous differenced series in
            // place. On that path the clamped read feeds `delta1 = (d[0] - d.first()) / 0`, i.e.
            // NaN deltas, which are consumed only by add_trend_param's bounds -- and the model is
            // already invalid (validate() rejects a null/short series), so no fit or likelihood
            // can consume them. The divergence is therefore "C# throws where this port continues
            // with a degenerate, unusable parameter set", never a differing NUMBER on any path a
            // fit can reach.
            int last_idx = std::max(0, std::min(training_time_steps_ - 1, diff_series_->count() - 1));
            delta1 = ((*diff_series_)[last_idx].value() - diff_series_->first().value()) / n_steps;
            delta2 = delta1 / n_steps;
            delta3 = delta2 / n_steps;
            delta1 = std::pow(10.0, std::floor(std::log10(std::fabs(delta1)) + 1.0));
            delta2 = std::pow(10.0, std::floor(std::log10(std::fabs(delta2)) + 1.0));
            delta3 = std::pow(10.0, std::floor(std::log10(std::fabs(delta3)) + 1.0));

            sigma_ub = std::pow(10.0, std::ceil(std::log10(sigma) + 1.0));
            if (std::isnan(sigma_ub) || std::isinf(sigma_ub)) sigma_ub = 100;
        }

        parameters_.clear();  // C# `_parameters = new List<>()`; PropertyChanged unsub -> no-op.

        // Initial estimates for the trend parameters via polynomial regression (ARIMAX.cs:831).
        double intercept_init = mean;
        double gamma1_init = 0, gamma2_init = 0, gamma3_init = 0;

        if (trend_type_ != Trend::None && diff_series_ && diff_series_->count() > 1) {
            int n = std::min(diff_series_->count(), training_time_steps_);
            std::vector<double> y(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) y[static_cast<std::size_t>(i)] = (*diff_series_)[i].value();

            if (trend_type_ == Trend::Linear) {
                double sum_t = 0, sum_t2 = 0, sum_y = 0, sum_ty = 0;
                for (int t = 0; t < n; ++t) {
                    sum_t += t;
                    sum_t2 += static_cast<double>(t) * t;
                    sum_y += y[static_cast<std::size_t>(t)];
                    sum_ty += t * y[static_cast<std::size_t>(t)];
                }
                double denom = n * sum_t2 - sum_t * sum_t;
                if (std::fabs(denom) > 1e-10) {
                    gamma1_init = (n * sum_ty - sum_t * sum_y) / denom;
                    intercept_init = (sum_y - gamma1_init * sum_t) / n;
                }
            } else if (trend_type_ == Trend::Quadratic) {
                double s0 = n, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
                double sy = 0, sty = 0, st2y = 0;
                for (int t = 0; t < n; ++t) {
                    double t2 = static_cast<double>(t) * t;
                    s1 += t;
                    s2 += t2;
                    s3 += t * t2;
                    s4 += t2 * t2;
                    sy += y[static_cast<std::size_t>(t)];
                    sty += t * y[static_cast<std::size_t>(t)];
                    st2y += t2 * y[static_cast<std::size_t>(t)];
                }
                double det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s2 * s3) +
                             s2 * (s1 * s3 - s2 * s2);
                if (std::fabs(det) > 1e-10) {
                    intercept_init = (sy * (s2 * s4 - s3 * s3) - s1 * (sty * s4 - st2y * s3) +
                                      s2 * (sty * s3 - st2y * s2)) /
                                     det;
                }
            } else if (trend_type_ == Trend::Cubic) {
                intercept_init = y[0];
            }
        }

        // Intercept.
        if (include_intercept_) {
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/"Intercept (\xCE\xBC)", /*value=*/intercept_init,
                /*lower_bound=*/min, /*upper_bound=*/max,
                std::make_unique<numerics::distributions::Uniform>(min, max));
        }

        // Trend parameters.
        if (trend_type_ == Trend::Linear) {
            add_trend_param("Trend (\xCE\xB3)", gamma1_init, delta1);
        } else if (trend_type_ == Trend::Quadratic) {
            add_trend_param(std::string("Trend (\xCE\xB3") + to_subscript(1) + ")", gamma1_init, delta1);
            add_trend_param(std::string("Trend (\xCE\xB3") + to_subscript(2) + ")", gamma2_init, delta2);
        } else if (trend_type_ == Trend::Cubic) {
            add_trend_param(std::string("Trend (\xCE\xB3") + to_subscript(1) + ")", gamma1_init, delta1);
            add_trend_param(std::string("Trend (\xCE\xB3") + to_subscript(2) + ")", gamma2_init, delta2);
            add_trend_param(std::string("Trend (\xCE\xB3") + to_subscript(3) + ")", gamma3_init, delta3);
        }

        // Seasonality parameters -- Fourier series (ARIMAX.cs:963).
        if (include_seasonality_) {
            double amplitude = range / 4;
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/std::string("Seasonality Sin (\xCF\x88") + to_subscript(1) + ")",
                /*value=*/0.0, /*lower_bound=*/-amplitude, /*upper_bound=*/amplitude,
                std::make_unique<numerics::distributions::Uniform>(-amplitude, amplitude));
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/std::string("Seasonality Cos (\xCF\x88") + to_subscript(2) + ")",
                /*value=*/0.0, /*lower_bound=*/-amplitude, /*upper_bound=*/amplitude,
                std::make_unique<numerics::distributions::Uniform>(-amplitude, amplitude));
        }

        // Covariate parameters (ARIMAX.cs:987).
        if (covariates_ && !covariates_->empty()) {
            for (int i = 1; i <= static_cast<int>(covariates_->size()); ++i) {
                if (x_order_b_ == 0) {
                    add_covariate_param(std::string("Covariate (\xCE\xB2") + to_subscript(i) + ")");
                } else {
                    for (int j = 0; j <= x_order_b_; ++j) {
                        std::string lag_label = j == 0 ? "" : (",-" + std::to_string(j));
                        add_covariate_param(std::string("Covariate (\xCE\xB2") + to_subscript(i) +
                                            lag_label + ")");
                    }
                }
            }
        }

        // AR parameters.
        for (int i = 1; i <= ar_order_p_; ++i) {
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/std::string("AR (\xCF\x86") + to_subscript(i) + ")",
                /*value=*/0.0, /*lower_bound=*/-2.0, /*upper_bound=*/2.0,
                std::make_unique<numerics::distributions::Uniform>(-2.0, 2.0));
        }

        // MA parameters.
        for (int i = 1; i <= ma_order_q_; ++i) {
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/std::string("MA (\xCE\xB8") + to_subscript(i) + ")",
                /*value=*/0.0, /*lower_bound=*/-2.0, /*upper_bound=*/2.0,
                std::make_unique<numerics::distributions::Uniform>(-2.0, 2.0));
        }

        // Scale (standard error) parameter.
        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/"Scale (\xCF\x83)", /*value=*/sigma,
            /*lower_bound=*/sigma_lb, /*upper_bound=*/sigma_ub,
            std::make_unique<numerics::distributions::Uniform>(sigma_lb, sigma_ub),
            /*is_positive=*/true);
    }

    // --- SetParameterValues (C#:1070): the base ModelBase body is byte-for-byte the C# body. ---

    // --- Residuals (C#:1252): the mean-function heart driving the likelihood. ---
    std::vector<double> residuals(const std::vector<double>& parameters) const {
        int effective = diff_series_
                            ? std::min(training_time_steps_, diff_series_->count())
                            : training_time_steps_;
        if (effective < 0) effective = 0;
        std::vector<double> y(static_cast<std::size_t>(effective), 0.0);
        std::vector<double> mean(static_cast<std::size_t>(effective), 0.0);
        std::vector<double> epsilon(static_cast<std::size_t>(effective), 0.0);
        std::vector<double> res(static_cast<std::size_t>(effective), 0.0);
        int max_order = std::max(ar_order_p_, std::max(ma_order_q_, x_order_b_));

        int k = 0;
        double mu = 0;
        std::vector<double> gamma;
        std::vector<double> psi;
        std::vector<std::vector<double>> beta;  // [covariate][lag]
        std::vector<double> phi(static_cast<std::size_t>(ar_order_p_));
        std::vector<double> theta(static_cast<std::size_t>(ma_order_q_));

        if (include_intercept_) mu = parameters[static_cast<std::size_t>(k++)];
        extract_trend(parameters, k, gamma);
        if (include_seasonality_)
            psi = {parameters[static_cast<std::size_t>(k++)], parameters[static_cast<std::size_t>(k++)]};
        extract_beta(parameters, k, beta);
        for (int i = 0; i < ar_order_p_; ++i) phi[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(k++)];
        for (int i = 0; i < ma_order_q_; ++i) theta[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(k++)];

        for (int t = 0; t < effective; ++t) {
            mean[static_cast<std::size_t>(t)] = mu;
            mean[static_cast<std::size_t>(t)] += trend_value(gamma, t);
            mean[static_cast<std::size_t>(t)] += seasonality_value(psi, t);

            // Covariate component.
            if (!beta.empty() && covariates_) {
                for (int i = 0; i < static_cast<int>(covariates_->size()); ++i) {
                    if (x_order_b_ == 0) {
                        mean[static_cast<std::size_t>(t)] +=
                            beta[static_cast<std::size_t>(i)][0] * (*covariates_)[static_cast<std::size_t>(i)][t].value();
                    } else {
                        if (t < (*covariates_)[static_cast<std::size_t>(i)].count())
                            mean[static_cast<std::size_t>(t)] +=
                                beta[static_cast<std::size_t>(i)][0] *
                                (*covariates_)[static_cast<std::size_t>(i)][t].value();
                        for (int j = 1; j <= x_order_b_ && t - j >= 0; ++j)
                            if (t - j < (*covariates_)[static_cast<std::size_t>(i)].count())
                                mean[static_cast<std::size_t>(t)] +=
                                    beta[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                                    (*covariates_)[static_cast<std::size_t>(i)][t - j].value();
                    }
                }
            }

            // AR component (mean-centered on the FULL mean function).
            double ar = 0;
            if (t >= ar_order_p_) {
                for (int p = 1; p <= ar_order_p_; ++p)
                    ar += phi[static_cast<std::size_t>(p - 1)] *
                          ((*diff_series_)[t - p].value() - mean[static_cast<std::size_t>(t - p)]);
            }

            // MA component.
            double ma = 0;
            for (int q = 1; q <= std::min(t, ma_order_q_); ++q)
                ma += theta[static_cast<std::size_t>(q - 1)] * epsilon[static_cast<std::size_t>(t - q)];

            if (t < max_order) {
                y[static_cast<std::size_t>(t)] = (*diff_series_)[t].value();
            } else {
                y[static_cast<std::size_t>(t)] = mean[static_cast<std::size_t>(t)] + ar + ma;
            }

            epsilon[static_cast<std::size_t>(t)] =
                (*diff_series_)[t].value() - y[static_cast<std::size_t>(t)];
            res[static_cast<std::size_t>(t)] = epsilon[static_cast<std::size_t>(t)];
        }
        return res;
    }

    // --- DataLogLikelihood (C#:1082): conditional Gaussian from t = maxOrder + logJacobian.
    //     NOTE: NO "-inf when no series" guard -- returns 0.0 for the no-series case (C# governs;
    //     ARIMAXTests.cs:453). ---
    double data_log_likelihood(std::vector<double>& parameters) const override {
        for (double v : parameters)
            if (std::isnan(v)) return -std::numeric_limits<double>::infinity();

        double sigma = parameters.back();
        if (sigma <= 0) return -std::numeric_limits<double>::infinity();
        numerics::distributions::Normal norm(0.0, sigma);
        std::vector<double> res = residuals(parameters);
        int max_order = std::max(ar_order_p_, std::max(ma_order_q_, x_order_b_));
        double log_lh = 0.0;
        for (int t = max_order; t < static_cast<int>(res.size()); ++t)
            log_lh += norm.log_pdf(res[static_cast<std::size_t>(t)]);
        return log_lh + log_jacobian_;
    }

    // --- PointwiseDataLogLikelihood (C#:1111). ---
    std::vector<double> pointwise_data_log_likelihood(
        const std::vector<double>& parameters) const override {
        int max_order = std::max(ar_order_p_, std::max(ma_order_q_, x_order_b_));
        int effective = diff_series_ ? std::min(training_time_steps_, diff_series_->count())
                                     : training_time_steps_;
        int n = effective - max_order;
        if (n <= 0) return {};

        for (double v : parameters)
            if (std::isnan(v))
                return std::vector<double>(static_cast<std::size_t>(n),
                                           -std::numeric_limits<double>::infinity());

        double sigma = parameters.back();
        numerics::distributions::Normal norm(0.0, sigma);
        std::vector<double> res = residuals(parameters);
        std::vector<double> result(static_cast<std::size_t>(n));

        double jacobian_per_obs = log_jacobian_ / n;
        int idx = 0;
        for (int t = max_order; t < static_cast<int>(res.size()); ++t)
            result[static_cast<std::size_t>(idx++)] =
                norm.log_pdf(res[static_cast<std::size_t>(t)]) + jacobian_per_obs;
        return result;
    }

    // --- PointwiseDataLogLikelihoodComponents (C#:1151). ---
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& parameters) const override {
        int max_order = std::max(ar_order_p_, std::max(ma_order_q_, x_order_b_));
        int effective = diff_series_ ? std::min(training_time_steps_, diff_series_->count())
                                     : training_time_steps_;
        int n = effective - max_order;
        if (n <= 0) return {};

        std::vector<DataComponent> result;
        result.reserve(static_cast<std::size_t>(n));
        std::vector<double> response =
            training_time_series_ ? training_time_series_->values_to_array() : std::vector<double>{};

        for (double v : parameters) {
            if (std::isnan(v)) {
                for (int j = 0; j < n; ++j) {
                    int t_idx = max_order + j;
                    double value = t_idx < static_cast<int>(response.size())
                                       ? response[static_cast<std::size_t>(t_idx)]
                                       : 0.0;
                    result.emplace_back(j, -std::numeric_limits<double>::infinity(), value,
                                        DataComponentType::Exact, 1,
                                        std::string("t=") + std::to_string(t_idx));
                }
                return result;
            }
        }

        double sigma = parameters.back();
        numerics::distributions::Normal norm(0.0, sigma);
        std::vector<double> res = residuals(parameters);
        double jacobian_per_obs = log_jacobian_ / n;

        int idx = 0;
        for (int t = max_order; t < static_cast<int>(res.size()); ++t) {
            double log_lh = norm.log_pdf(res[static_cast<std::size_t>(t)]) + jacobian_per_obs;
            double value = t < static_cast<int>(response.size())
                               ? response[static_cast<std::size_t>(t)]
                               : 0.0;
            result.emplace_back(idx++, log_lh, value, DataComponentType::Exact, 1,
                                std::string("t=") + std::to_string(t));
        }
        return result;
    }

    // --- PriorLogLikelihood (C#:1199): parameter priors + optional Jeffreys 1/sigma. ---
    double prior_log_likelihood(std::vector<double>& parameters) const override {
        if (parameters.size() < parameters_.size())
            return -std::numeric_limits<double>::infinity();

        double sigma = parameters.back();
        double log_lh = 0.0;
        for (std::size_t i = 0; i < parameters_.size(); ++i)
            log_lh += parameters_[i].prior_distribution().log_pdf(parameters[i]);

        if (use_jeffreys_rule_for_scale_)
            log_lh -= sigma > 0 ? std::log(sigma) : std::numeric_limits<double>::infinity();

        if (!numerics::is_finite(log_lh)) return -std::numeric_limits<double>::infinity();
        return log_lh;
    }

    // --- PointwisePriorLogLikelihood (C#:1224). ---
    std::vector<PriorComponent> pointwise_prior_log_likelihood(
        const std::vector<double>& parameters) const override {
        std::vector<PriorComponent> result;
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            double ll = parameters_[i].prior_distribution().log_pdf(parameters[i]);
            const std::string& param_name = parameters_[i].owner_name().empty()
                                                ? parameters_[i].name()
                                                : parameters_[i].owner_name();
            result.emplace_back("Parameter Prior: " + param_name, ll,
                                PriorComponentType::ParameterPrior);
        }
        if (use_jeffreys_rule_for_scale_) {
            double sigma = parameters.back();
            double ll = sigma > 0 ? -std::log(sigma) : -std::numeric_limits<double>::infinity();
            result.emplace_back("Jeffreys Scale: \xCF\x83", ll,
                                PriorComponentType::JeffreysScalePrior);
        }
        return result;
    }

    // C# `AppendTail` (line 799): clone the observed covariate and append the resampled tail,
    // walking the dates forward at the series' own interval.
    static TimeSeries append_tail(const TimeSeries& observed, const TimeSeries& tail) {
        TimeSeries extended = observed.clone();
        numerics::data::DateTime next =
            extended.count() > 0
                ? TimeSeries::add_time_interval(extended[extended.count() - 1].index(),
                                                extended.time_interval())
                : observed.start_date();
        for (int j = 0; j < tail.count(); ++j) {
            extended.add(TimeSeries::Ordinate(next, tail[j].value()));
            next = TimeSeries::add_time_interval(next, extended.time_interval());
        }
        return extended;
    }

    // C# `AppendConstantTail` (line 819): the deterministic counterpart, used when no seed was
    // given so a "deterministic forecast" does not depend on one arbitrary resampled path.
    static TimeSeries append_constant_tail(const TimeSeries& observed, double constant,
                                           int tail_length) {
        TimeSeries extended = observed.clone();
        numerics::data::DateTime next =
            extended.count() > 0
                ? TimeSeries::add_time_interval(extended[extended.count() - 1].index(),
                                                extended.time_interval())
                : observed.start_date();
        for (int j = 0; j < tail_length; ++j) {
            extended.add(TimeSeries::Ordinate(next, constant));
            next = TimeSeries::add_time_interval(next, extended.time_interval());
        }
        return extended;
    }

    // --- Predict (C#:1425): the ARMA recursion on the differenced scale with the full inline mean
    //     function, then reverse-differencing integration and inverse transform. Returns Y + parts.
    //     The covariate forecast tail is extended per CovariateExtension (P6). ---
    PredictResult predict_components(const std::vector<double>& parameters, int forecast_steps = 0,
                                     int seed = -1) const {
        int total_steps = training_time_steps_ + forecast_steps;
        if (total_steps < 0) total_steps = 0;

        std::vector<double> y(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> intercept_part(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> trend_part(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> seasonality_part(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> covariate_part(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> ar_part(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> ma_part(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> mean(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> epsilon(static_cast<std::size_t>(total_steps), 0.0);

        int max_order = std::max(diff_order_d_, std::max(ar_order_p_, std::max(ma_order_q_, x_order_b_)));

        // C# uses System.Random here (no ported equivalent); ported MersenneTwister substituted.
        std::optional<numerics::sampling::MersenneTwister> prng;
        std::optional<numerics::distributions::Normal> err_dist;
        if (seed >= 0) {
            prng.emplace(static_cast<std::uint32_t>(seed));
            err_dist.emplace(0.0, parameters.back());
        }

        // Covariate forecast-tail extension (C# 1558-1650), un-gated in P6 by the ported
        // TimeSeries container. The observed span is always kept EXACTLY as observed -- only the
        // tail of length forecast_steps is extended -- because resampling the observed period
        // would inflate the training-period credible interval, every realization moving
        // beta * X[t]. Without a seed the tail is each covariate's MEAN rather than one arbitrary
        // resampled path, so a deterministic forecast does not depend on the extension method.
        const std::vector<TimeSeries>* use_covariates = covariates_ ? &*covariates_ : nullptr;
        std::vector<TimeSeries> extended_covariates;
        if (forecast_steps > 0 && covariates_ && !covariates_->empty()) {
            bool deterministic = seed < 0;
            int resample_seed = seed >= 0 ? seed + 1000 : 12345;
            switch (covariate_extension_) {
                case CovariateExtensionMethod::None: {
                    int required_length = training_time_steps_ + forecast_steps;
                    for (std::size_t i = 0; i < covariates_->size(); ++i) {
                        if ((*covariates_)[i].count() < required_length)
                            throw std::runtime_error(
                                "Covariate " + std::to_string(i) + " has " +
                                std::to_string((*covariates_)[i].count()) + " observations but " +
                                std::to_string(required_length) + " are required for " +
                                std::to_string(forecast_steps) +
                                " forecast steps. Either provide forecast covariates or set "
                                "CovariateExtension to BlockBootstrap or KNN.");
                    }
                    break;
                }
                case CovariateExtensionMethod::BlockBootstrap: {
                    for (std::size_t i = 0; i < covariates_->size(); ++i) {
                        const TimeSeries& cov = (*covariates_)[i];
                        if (deterministic) {
                            extended_covariates.push_back(
                                append_constant_tail(cov, cov.mean_value(), forecast_steps));
                        } else {
                            int block_size = std::max(1, std::min(10, cov.count() / 4));
                            extended_covariates.push_back(append_tail(
                                cov, cov.resample_with_block_bootstrap(
                                         forecast_steps, block_size,
                                         resample_seed + static_cast<int>(i))));
                        }
                    }
                    use_covariates = &extended_covariates;
                    break;
                }
                case CovariateExtensionMethod::KNN: {
                    for (std::size_t i = 0; i < covariates_->size(); ++i) {
                        const TimeSeries& cov = (*covariates_)[i];
                        if (deterministic) {
                            extended_covariates.push_back(
                                append_constant_tail(cov, cov.mean_value(), forecast_steps));
                        } else {
                            int knn = std::max(3, cov.count() / 10);
                            extended_covariates.push_back(append_tail(
                                cov, cov.resample_with_knn(forecast_steps, knn,
                                                           resample_seed + static_cast<int>(i))));
                        }
                    }
                    use_covariates = &extended_covariates;
                    break;
                }
            }
        }

        int k = 0;
        double mu = 0;
        std::vector<double> gamma;
        std::vector<double> psi;
        std::vector<std::vector<double>> beta;
        std::vector<double> phi(static_cast<std::size_t>(ar_order_p_));
        std::vector<double> theta(static_cast<std::size_t>(ma_order_q_));

        if (include_intercept_) mu = parameters[static_cast<std::size_t>(k++)];
        extract_trend(parameters, k, gamma);
        if (include_seasonality_)
            psi = {parameters[static_cast<std::size_t>(k++)], parameters[static_cast<std::size_t>(k++)]};
        // Covariate beta layout follows useCovariates.Count == Covariates.Count in the fit window.
        if (use_covariates && !use_covariates->empty()) {
            beta.assign(use_covariates->size(),
                        std::vector<double>(static_cast<std::size_t>(x_order_b_ + 1), 0.0));
            for (std::size_t i = 0; i < use_covariates->size(); ++i)
                for (int j = 0; j <= x_order_b_; ++j)
                    beta[i][static_cast<std::size_t>(j)] = parameters[static_cast<std::size_t>(k++)];
        }
        for (int i = 0; i < ar_order_p_; ++i) phi[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(k++)];
        for (int i = 0; i < ma_order_q_; ++i) theta[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(k++)];

        for (int t = 0; t < total_steps; ++t) {
            mean[static_cast<std::size_t>(t)] = mu;
            intercept_part[static_cast<std::size_t>(t)] = mu;

            double trend = trend_value(gamma, t);
            mean[static_cast<std::size_t>(t)] += trend;
            trend_part[static_cast<std::size_t>(t)] = trend;

            double seasonality = seasonality_value(psi, t);
            mean[static_cast<std::size_t>(t)] += seasonality;
            seasonality_part[static_cast<std::size_t>(t)] = seasonality;

            double covariate = 0;
            if (!beta.empty() && use_covariates && !use_covariates->empty() &&
                t < (*use_covariates)[0].count()) {
                for (std::size_t i = 0; i < use_covariates->size(); ++i) {
                    if (x_order_b_ == 0) {
                        covariate += beta[i][0] * (*use_covariates)[i][t].value();
                    } else {
                        covariate += beta[i][0] * (*use_covariates)[i][t].value();
                        for (int j = 1; j <= x_order_b_ && t - j >= 0 &&
                                        t - j < (*use_covariates)[i].count();
                             ++j)
                            covariate += beta[i][static_cast<std::size_t>(j)] *
                                         (*use_covariates)[i][t - j].value();
                    }
                }
            }
            mean[static_cast<std::size_t>(t)] += covariate;
            covariate_part[static_cast<std::size_t>(t)] = covariate;

            double ar = 0;
            if (t >= ar_order_p_) {
                for (int p = 1; p <= ar_order_p_; ++p) {
                    if (diff_series_ && t - p < training_time_steps_ && t - p < diff_series_->count())
                        ar += phi[static_cast<std::size_t>(p - 1)] *
                              ((*diff_series_)[t - p].value() - mean[static_cast<std::size_t>(t - p)]);
                    else
                        ar += phi[static_cast<std::size_t>(p - 1)] *
                              (y[static_cast<std::size_t>(t - p)] - mean[static_cast<std::size_t>(t - p)]);
                }
            }
            ar_part[static_cast<std::size_t>(t)] = ar;

            double ma = 0;
            for (int q = 1; q <= std::min(t, ma_order_q_); ++q)
                ma += theta[static_cast<std::size_t>(q - 1)] * epsilon[static_cast<std::size_t>(t - q)];
            ma_part[static_cast<std::size_t>(t)] = ma;

            if (t < max_order && diff_series_ && t < diff_series_->count()) {
                y[static_cast<std::size_t>(t)] = (*diff_series_)[t].value();
            } else {
                y[static_cast<std::size_t>(t)] = mean[static_cast<std::size_t>(t)] + ar + ma;
            }

            if (diff_series_ && t < training_time_steps_ && t < diff_series_->count())
                epsilon[static_cast<std::size_t>(t)] =
                    (*diff_series_)[t].value() - y[static_cast<std::size_t>(t)];

            if (prng && t >= max_order) {
                double mt = y[static_cast<std::size_t>(t)];
                double error = err_dist->inverse_cdf(prng->next_double());
                y[static_cast<std::size_t>(t)] += error;
                if (t >= training_time_steps_)
                    epsilon[static_cast<std::size_t>(t)] = y[static_cast<std::size_t>(t)] - mt;
            }
        }

        // Step A -- integrate (reverse differencing) up through the differencing levels, anchoring
        // each level to the observed intermediate difference of _transformedTimeSeries (C#:1714).
        if (diff_order_d_ > 0) {
            std::vector<double> integrated = y;

            std::vector<TimeSeries> anchor_series;
            bool has_anchor = transformed_time_series_ && transformed_time_series_->count() > 0;
            if (has_anchor) {
                anchor_series.push_back(*transformed_time_series_);
                for (int level = 1; level < diff_order_d_; ++level)
                    anchor_series.push_back(anchor_series[static_cast<std::size_t>(level - 1)].difference(1, 1));
            }

            for (int d = 0; d < diff_order_d_; ++d) {
                int anchor_level = diff_order_d_ - 1 - d;
                const TimeSeries* anchor =
                    has_anchor && anchor_level < static_cast<int>(anchor_series.size())
                        ? &anchor_series[static_cast<std::size_t>(anchor_level)]
                        : nullptr;
                bool use_observed_anchor = anchor && anchor->count() > 0;

                if (use_observed_anchor) {
                    integrated[0] = (*anchor)[0].value();

                    int anchor_end = std::min(training_time_steps_, total_steps);
                    for (int i = 1; i < anchor_end; ++i) {
                        int obs_idx = i - 1;
                        if (obs_idx < anchor->count())
                            integrated[static_cast<std::size_t>(i)] =
                                (*anchor)[obs_idx].value() + integrated[static_cast<std::size_t>(i)];
                        else
                            integrated[static_cast<std::size_t>(i)] =
                                integrated[static_cast<std::size_t>(i - 1)] +
                                integrated[static_cast<std::size_t>(i)];
                    }
                    for (int i = std::max(1, anchor_end); i < total_steps; ++i)
                        integrated[static_cast<std::size_t>(i)] =
                            integrated[static_cast<std::size_t>(i - 1)] +
                            integrated[static_cast<std::size_t>(i)];
                } else {
                    for (int i = 1; i < total_steps; ++i)
                        integrated[static_cast<std::size_t>(i)] =
                            integrated[static_cast<std::size_t>(i - 1)] +
                            integrated[static_cast<std::size_t>(i)];
                }
            }

            y = std::move(integrated);
        }

        // Step B -- inverse transform back to the original scale.
        if (transform_type_ == Transform::Logarithmic || transform_type_ == Transform::BoxCox) {
            for (int t = 0; t < total_steps; ++t)
                y[static_cast<std::size_t>(t)] =
                    numerics::data::BoxCox::inverse_transform(y[static_cast<std::size_t>(t)], lambda_);
        } else if (transform_type_ == Transform::YeoJohnson) {
            for (int t = 0; t < total_steps; ++t)
                y[static_cast<std::size_t>(t)] = numerics::data::YeoJohnson::inverse_transform(
                    y[static_cast<std::size_t>(t)], lambda_);
        }

        return {std::move(y),
                std::move(intercept_part),
                std::move(trend_part),
                std::move(seasonality_part),
                std::move(covariate_part),
                std::move(ar_part),
                std::move(ma_part)};
    }

    // Predict with the current parameter values, returning Y only.
    std::vector<double> predict(int forecast_steps = 0, int seed = -1) const {
        return predict_components(parameter_values(), forecast_steps, seed).y;
    }

    // --- GenerateRandomValues (C#:2146): ISimulatable entry point. Bit-exact MersenneTwister.
    //     The covariate forecast-tail extension is DEFERRED (see file header). ---
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size <= 0) throw std::out_of_range("Sample size must be positive.");

        numerics::sampling::MersenneTwister rng =
            seed >= 0 ? numerics::sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                      : numerics::sampling::MersenneTwister();

        int k = 0;
        double mu = 0;
        std::vector<double> gamma;
        std::vector<double> psi;
        std::vector<std::vector<double>> beta;

        if (include_intercept_) mu = parameters_[static_cast<std::size_t>(k++)].value();
        if (trend_type_ == Trend::Linear) {
            gamma = {parameters_[static_cast<std::size_t>(k++)].value()};
        } else if (trend_type_ == Trend::Quadratic) {
            gamma = {parameters_[static_cast<std::size_t>(k++)].value(),
                     parameters_[static_cast<std::size_t>(k++)].value()};
        } else if (trend_type_ == Trend::Cubic) {
            gamma = {parameters_[static_cast<std::size_t>(k++)].value(),
                     parameters_[static_cast<std::size_t>(k++)].value(),
                     parameters_[static_cast<std::size_t>(k++)].value()};
        }
        if (include_seasonality_)
            psi = {parameters_[static_cast<std::size_t>(k++)].value(),
                   parameters_[static_cast<std::size_t>(k++)].value()};

        const std::vector<TimeSeries>* use_covariates = nullptr;
        std::vector<TimeSeries> extended_covariates;
        if (covariates_ && !covariates_->empty()) {
            beta.assign(covariates_->size(),
                        std::vector<double>(static_cast<std::size_t>(x_order_b_ + 1), 0.0));
            for (std::size_t i = 0; i < covariates_->size(); ++i)
                for (int j = 0; j <= x_order_b_; ++j)
                    beta[i][static_cast<std::size_t>(j)] = parameters_[static_cast<std::size_t>(k++)].value();

            // C# 2333-2378: extend the covariates when the requested sample runs past them.
            // Unlike Predict, this path has no deterministic branch -- a generated realization is
            // stochastic by definition -- and its resample seed offset is 2000, not 1000.
            if (sample_size > (*covariates_)[0].count()) {
                int resample_seed = seed > 0 ? seed + 2000 : 54321;
                switch (covariate_extension_) {
                    case CovariateExtensionMethod::None:
                        throw std::runtime_error(
                            "Covariates have " + std::to_string((*covariates_)[0].count()) +
                            " observations but " + std::to_string(sample_size) +
                            " are required. Either provide generate covariates or set "
                            "CovariateExtension to BlockBootstrap or KNN.");
                    case CovariateExtensionMethod::BlockBootstrap:
                        for (std::size_t i = 0; i < covariates_->size(); ++i) {
                            const TimeSeries& cov = (*covariates_)[i];
                            int block_size = std::max(1, std::min(10, cov.count() / 4));
                            int tail_size = sample_size - cov.count();
                            extended_covariates.push_back(append_tail(
                                cov, cov.resample_with_block_bootstrap(
                                         tail_size, block_size,
                                         resample_seed + static_cast<int>(i))));
                        }
                        break;
                    case CovariateExtensionMethod::KNN:
                        for (std::size_t i = 0; i < covariates_->size(); ++i) {
                            const TimeSeries& cov = (*covariates_)[i];
                            int knn = std::max(3, cov.count() / 10);
                            int tail_size = sample_size - cov.count();
                            extended_covariates.push_back(append_tail(
                                cov, cov.resample_with_knn(tail_size, knn,
                                                           resample_seed + static_cast<int>(i))));
                        }
                        break;
                }
                use_covariates = &extended_covariates;
            } else {
                use_covariates = &*covariates_;
            }
        }

        std::vector<double> phi(static_cast<std::size_t>(ar_order_p_));
        for (int i = 0; i < ar_order_p_; ++i)
            phi[static_cast<std::size_t>(i)] = parameters_[static_cast<std::size_t>(k++)].value();
        std::vector<double> theta(static_cast<std::size_t>(ma_order_q_));
        for (int i = 0; i < ma_order_q_; ++i)
            theta[static_cast<std::size_t>(i)] = parameters_[static_cast<std::size_t>(k++)].value();

        double sigma = parameters_[static_cast<std::size_t>(k)].value();
        numerics::distributions::Normal normal(0.0, sigma);

        std::vector<double> series(static_cast<std::size_t>(sample_size), 0.0);
        std::vector<double> mean(static_cast<std::size_t>(sample_size), 0.0);
        std::vector<double> epsilon(static_cast<std::size_t>(sample_size), 0.0);

        // Pre-generate the Gaussian noise (matches the C# draw order).
        std::vector<double> noise(static_cast<std::size_t>(sample_size));
        for (int t = 0; t < sample_size; ++t)
            noise[static_cast<std::size_t>(t)] = normal.inverse_cdf(rng.next_double());

        // Pre-compute the mean at each time step (matches Residuals).
        for (int t = 0; t < sample_size; ++t) {
            mean[static_cast<std::size_t>(t)] = mu;
            mean[static_cast<std::size_t>(t)] += trend_value(gamma, t);
            mean[static_cast<std::size_t>(t)] += seasonality_value(psi, t);
            if (!beta.empty() && use_covariates && t < (*use_covariates)[0].count()) {
                for (std::size_t i = 0; i < use_covariates->size(); ++i) {
                    if (x_order_b_ == 0) {
                        mean[static_cast<std::size_t>(t)] += beta[i][0] * (*use_covariates)[i][t].value();
                    } else {
                        mean[static_cast<std::size_t>(t)] += beta[i][0] * (*use_covariates)[i][t].value();
                        for (int j = 1; j <= x_order_b_ && t - j >= 0; ++j)
                            mean[static_cast<std::size_t>(t)] +=
                                beta[i][static_cast<std::size_t>(j)] * (*use_covariates)[i][t - j].value();
                    }
                }
            }
        }

        for (int t = 0; t < sample_size; ++t) {
            double ar = 0;
            if (t >= ar_order_p_) {
                for (int p = 1; p <= ar_order_p_; ++p)
                    ar += phi[static_cast<std::size_t>(p - 1)] *
                          (series[static_cast<std::size_t>(t - p)] - mean[static_cast<std::size_t>(t - p)]);
            }
            double ma = 0;
            for (int q = 1; q <= std::min(t, ma_order_q_); ++q)
                ma += theta[static_cast<std::size_t>(q - 1)] * epsilon[static_cast<std::size_t>(t - q)];

            double deterministic = mean[static_cast<std::size_t>(t)] + ar + ma;

            if (transform_type_ == Transform::None) {
                series[static_cast<std::size_t>(t)] = deterministic + noise[static_cast<std::size_t>(t)];
            } else if (transform_type_ == Transform::Logarithmic ||
                       transform_type_ == Transform::BoxCox) {
                series[static_cast<std::size_t>(t)] = numerics::data::BoxCox::inverse_transform(
                    numerics::data::BoxCox::transform(deterministic, lambda_) +
                        noise[static_cast<std::size_t>(t)],
                    lambda_);
            } else if (transform_type_ == Transform::YeoJohnson) {
                series[static_cast<std::size_t>(t)] = numerics::data::YeoJohnson::inverse_transform(
                    numerics::data::YeoJohnson::transform(deterministic, lambda_) +
                        noise[static_cast<std::size_t>(t)],
                    lambda_);
            }

            epsilon[static_cast<std::size_t>(t)] = series[static_cast<std::size_t>(t)] - deterministic;
        }

        // Integrate back through the differencing levels, seeding each level with mu (C#:2383).
        if (diff_order_d_ > 0) {
            std::vector<double> integrated = series;
            for (int d = 0; d < diff_order_d_; ++d) {
                integrated[0] = mu + integrated[0];
                for (int i = 1; i < sample_size; ++i)
                    integrated[static_cast<std::size_t>(i)] =
                        integrated[static_cast<std::size_t>(i - 1)] +
                        integrated[static_cast<std::size_t>(i)];
            }
            return integrated;
        }
        return series;
    }

    // --- Validate (C#:1879). Includes the AR/MA sum-abs warnings via the start-index helpers. ---
    ValidationResult validate() const override {
        ValidationResult result;  // is_valid = true

        if (!time_series_) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: Time series data is null.");
            return result;
        }

        if (time_series_->count() < 10) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Time series must have at least 10 observations.");
        }
        if (time_series_->time_interval() == numerics::data::TimeInterval::Irregular) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Time series analysis requires a regular time interval. Resample or "
                "convert the series to a regular interval before estimating.");
        }
        if (training_time_steps_ < number_of_parameters()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Training time steps (" + std::to_string(training_time_steps_) +
                ") must be at least equal to the number of parameters (" +
                std::to_string(number_of_parameters()) + ").");
        }
        if (training_time_steps_ > time_series_->count()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Training time steps cannot exceed time series length.");
        }
        if (diff_order_d_ > 0 && training_time_steps_ > time_series_->count() - diff_order_d_) {
            result.validation_messages.push_back(
                "Warning: TrainingTimeSteps (" + std::to_string(training_time_steps_) +
                ") exceeds the differenced series length (" +
                std::to_string(time_series_->count() - diff_order_d_) +
                "). Effective training will use " +
                std::to_string(time_series_->count() - diff_order_d_) + " time steps.");
        }
        if (ar_order_p_ < 0 || ar_order_p_ > 10) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: AR order (p) must be between 0 and 10.");
        }
        if (diff_order_d_ < 0 || diff_order_d_ > 2) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Differencing order (d) must be 0, 1, or 2.");
        }
        if (ma_order_q_ < 0 || ma_order_q_ > 10) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: MA order (q) must be between 0 and 10.");
        }
        if (x_order_b_ < 0 || x_order_b_ > 10) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Exogenous lag order (b) must be between 0 and 10.");
        }
        if (ar_order_p_ == 0 && ma_order_q_ == 0 && !include_intercept_ &&
            trend_type_ == Trend::None && !include_seasonality_ &&
            (!covariates_ || covariates_->empty())) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Model must have at least one component (AR, MA, intercept, trend, "
                "seasonality, or covariates).");
        }
        if (diff_order_d_ > 0 && trend_type_ != Trend::None) {
            result.validation_messages.push_back(
                "Warning: Using both differencing and trend parameters is typically redundant.");
        }
        if (diff_order_d_ > 0 && include_seasonality_) {
            result.validation_messages.push_back(
                "Warning: Using both differencing and Fourier seasonality can cause parameter "
                "identification issues.");
        }
        if (covariates_ && !covariates_->empty()) {
            for (int i = 0; i < static_cast<int>(covariates_->size()); ++i)
                if ((*covariates_)[static_cast<std::size_t>(i)].count() != time_series_->count()) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Covariate " + std::to_string(i + 1) + " length (" +
                        std::to_string((*covariates_)[static_cast<std::size_t>(i)].count()) +
                        ") does not match time series length (" +
                        std::to_string(time_series_->count()) + ").");
                }
        }
        for (const auto& p : parameters_) {
            ValidationResult v = p.validate();
            if (!v.is_valid) {
                result.is_valid = false;
                for (const auto& m : v.validation_messages) result.validation_messages.push_back(m);
            }
        }
        if ((transform_type_ == Transform::Logarithmic || transform_type_ == Transform::BoxCox) &&
            time_series_->count() > 0 && time_series_->min_value() <= 0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Log-based transformations require all time series values to be strictly "
                "positive. Use Yeo-Johnson for series with non-positive values.");
        }

        // AR stationarity warning (sum-abs; conservative). C# formats the sum to 3 decimals; the
        // ASCII-normalized message keeps the "stationarity" token the tests key on.
        if (ar_order_p_ > 0) {
            int ar_start = get_ar_parameter_start_index();
            if (ar_start >= 0 && ar_start + ar_order_p_ <= number_of_parameters()) {
                double sum_abs_ar = 0;
                for (int i = 0; i < ar_order_p_; ++i)
                    sum_abs_ar += std::fabs(parameters_[static_cast<std::size_t>(ar_start + i)].value());
                if (sum_abs_ar >= 1.0)
                    result.validation_messages.push_back(
                        "Warning: AR coefficients may violate stationarity (sum of absolute "
                        "values = " + format3(sum_abs_ar) + " >= 1). Consider checking "
                        "characteristic equation roots.");
            }
        }
        // MA invertibility warning (sum-abs; conservative).
        if (ma_order_q_ > 0) {
            int ma_start = get_ma_parameter_start_index();
            if (ma_start >= 0 && ma_start + ma_order_q_ <= number_of_parameters()) {
                double sum_abs_ma = 0;
                for (int i = 0; i < ma_order_q_; ++i)
                    sum_abs_ma += std::fabs(parameters_[static_cast<std::size_t>(ma_start + i)].value());
                if (sum_abs_ma >= 1.0)
                    result.validation_messages.push_back(
                        "Warning: MA coefficients may violate invertibility (sum of absolute "
                        "values = " + format3(sum_abs_ma) + " >= 1). Consider checking "
                        "characteristic equation roots.");
            }
        }
        if (transform_fit_validation_message_) {
            result.is_valid = false;
            result.validation_messages.push_back(*transform_fit_validation_message_);
        }
        return result;
    }

    // --- GetARParameterStartIndex (C#:2073): the flat index where AR parameters begin. ---
    int get_ar_parameter_start_index() const {
        int idx = 0;
        if (include_intercept_) idx++;
        if (trend_type_ == Trend::Linear) idx += 1;
        else if (trend_type_ == Trend::Quadratic) idx += 2;
        else if (trend_type_ == Trend::Cubic) idx += 3;
        if (include_seasonality_) idx += 2;
        if (covariates_ && !covariates_->empty())
            idx += static_cast<int>(covariates_->size()) * (x_order_b_ + 1);
        return idx;
    }

    // --- GetMAParameterStartIndex (C#:2097). ---
    int get_ma_parameter_start_index() const {
        return get_ar_parameter_start_index() + ar_order_p_;
    }

   private:
    // Helper: adds a trend ModelParameter with symmetric [-delta, delta] bounds.
    void add_trend_param(const std::string& name, double value, double delta) {
        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/name, /*value=*/value, /*lower_bound=*/-delta,
            /*upper_bound=*/delta,
            std::make_unique<numerics::distributions::Uniform>(-delta, delta));
    }

    // Helper: adds a covariate ModelParameter with [-10, 10] bounds (ARIMAX.cs:997).
    void add_covariate_param(const std::string& name) {
        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/name, /*value=*/0.0, /*lower_bound=*/-10.0,
            /*upper_bound=*/10.0, std::make_unique<numerics::distributions::Uniform>(-10.0, 10.0));
    }

    // Extracts the trend coefficient vector, advancing k (mirrors the C# per-Trend blocks).
    void extract_trend(const std::vector<double>& parameters, int& k, std::vector<double>& gamma) const {
        if (trend_type_ == Trend::Linear) {
            gamma = {parameters[static_cast<std::size_t>(k++)]};
        } else if (trend_type_ == Trend::Quadratic) {
            gamma = {parameters[static_cast<std::size_t>(k++)], parameters[static_cast<std::size_t>(k++)]};
        } else if (trend_type_ == Trend::Cubic) {
            gamma = {parameters[static_cast<std::size_t>(k++)], parameters[static_cast<std::size_t>(k++)],
                     parameters[static_cast<std::size_t>(k++)]};
        }
    }

    // Extracts the beta[covariate][lag] matrix, advancing k (mirrors the C# double loop).
    void extract_beta(const std::vector<double>& parameters, int& k,
                      std::vector<std::vector<double>>& beta) const {
        if (covariates_ && !covariates_->empty()) {
            beta.assign(covariates_->size(),
                        std::vector<double>(static_cast<std::size_t>(x_order_b_ + 1), 0.0));
            for (std::size_t i = 0; i < covariates_->size(); ++i)
                for (int j = 0; j <= x_order_b_; ++j)
                    beta[i][static_cast<std::size_t>(j)] = parameters[static_cast<std::size_t>(k++)];
        }
    }

    // Inline polynomial trend value at t (ARIMAX.cs:1321).
    double trend_value(const std::vector<double>& gamma, int t) const {
        if (gamma.empty()) return 0.0;
        double td = t;
        if (trend_type_ == Trend::Linear) return gamma[0] * td;
        if (trend_type_ == Trend::Quadratic) return gamma[0] * td + gamma[1] * td * td;
        if (trend_type_ == Trend::Cubic)
            return gamma[0] * td + gamma[1] * td * td + gamma[2] * td * td * td;
        return 0.0;
    }

    // Inline Fourier seasonality value at t (ARIMAX.cs:1332). Uses kPi -- NEVER M_PI.
    double seasonality_value(const std::vector<double>& psi, int t) const {
        if (psi.empty()) return 0.0;
        double angle = 2.0 * numerics::kPi * t / seasonal_period_;
        return psi[0] * std::sin(angle) + psi[1] * std::cos(angle);
    }

    // --- InferSeasonalPeriod (C#:725): seasonal cycle length from the time interval. ---
    int infer_seasonal_period() const {
        if (!time_series_) return 12;
        switch (time_series_->time_interval()) {
            case numerics::data::TimeInterval::OneMinute: return 1440;
            case numerics::data::TimeInterval::FiveMinute: return 288;
            case numerics::data::TimeInterval::FifteenMinute: return 96;
            case numerics::data::TimeInterval::ThirtyMinute: return 48;
            case numerics::data::TimeInterval::OneHour: return 24;
            case numerics::data::TimeInterval::SixHour: return 4;
            case numerics::data::TimeInterval::TwelveHour: return 2;
            case numerics::data::TimeInterval::OneDay: return 365;
            case numerics::data::TimeInterval::SevenDay: return 52;
            case numerics::data::TimeInterval::OneMonth: return 12;
            case numerics::data::TimeInterval::OneQuarter: return 4;
            // v2.0.0 (ARIMAX.cs:880): annual records infer a DECADAL cycle. Was 1, which
            // collapsed the Fourier pair (sin(2*pi*t) == 0, cos(2*pi*t) == 1 at integer t).
            case numerics::data::TimeInterval::OneYear: return 10;
            default: return 12;
        }
    }

    // --- SetDefaultTrainingSteps (C#:569): 80% of data, min 30 or parameter count. ---
    // --- ResetDefaultTrainingStepsForNewTimeSeries (ARIMAX.cs:591) -----------------------------------
    //
    // v2.0.0: attaching a DIFFERENT response series is a new calibration problem, so the setter
    // discards any manual training-window edit and restores the default split. An empty series
    // zeroes the window (the C# `TimeSeries = null` arm -- this port holds the series BY VALUE in
    // a std::optional, so "no usable series" is the empty series, never a null reference).
    // C# RaisePropertyChange calls -> no-ops here.
    //
    // Divergence, unavoidable and inert for this port: C# short-circuits the whole setter on
    // `ReferenceEquals(_timeSeries, value)`, so re-assigning the SAME object preserves a manual
    // window. Value semantics have no object identity to test, so re-assigning an equal series
    // here resets. The public surface never re-assigns the same series (the fixture/binding
    // builders assign once at construction), and C#'s companion CollectionChanged hook -- which
    // preserves the manual window when the ATTACHED series is edited in place -- has no analog
    // either, since this port hands out the series by const reference and cannot be edited in
    // place.
    void reset_default_training_steps_for_new_time_series() {
        use_default_training_steps_ = true;
        if (!time_series_ || time_series_->count() == 0) {
            training_time_steps_ = 0;
            return;
        }
        set_default_training_steps();
    }

    void set_default_training_steps() {
        if (!time_series_ || time_series_->count() == 0) return;
        int min_steps = std::max(30, static_cast<int>(parameters_.size()));
        set_training_time_steps(
            std::max(min_steps, static_cast<int>(std::floor(0.8 * time_series_->count()))));
    }

    // --- SetTrainingData (C#:588): transform the WHOLE raw series, then difference by DiffOrderD
    //     via the P2 TimeSeries.Difference(1, d); training series is the first
    //     effectiveTrainingSteps of _diffSeries. Jacobian window: max(0, d + maxOrder) .. TTS-1. ---
    void set_training_data() {
        transform_fit_validation_message_.reset();
        if (!time_series_ || training_time_steps_ == 0) return;

        int max_order = std::max(ar_order_p_, std::max(ma_order_q_, x_order_b_));

        // Step 1: transform the whole raw series.
        TimeSeries transformed(time_series_->time_interval());
        if (transform_type_ == Transform::None) {
            lambda_ = 0;
            for (int i = 0; i < time_series_->count(); ++i) transformed.add((*time_series_)[i].clone());
        } else if (transform_type_ == Transform::Logarithmic) {
            lambda_ = 0;
            for (int i = 0; i < time_series_->count(); ++i) {
                transformed.add((*time_series_)[i].clone());
                transformed[i].set_value(
                    numerics::data::BoxCox::transform((*time_series_)[i].value(), lambda_));
            }
        } else if (transform_type_ == Transform::BoxCox) {
            lambda_ = numerics::data::BoxCox::fit_lambda(time_series_->values_to_list());
            if (!numerics::is_finite(lambda_)) {
                lambda_ = 0;
                log_jacobian_ = 0;
                transformed_time_series_ = TimeSeries(time_series_->time_interval());
                diff_series_ = TimeSeries(time_series_->time_interval());
                training_time_series_ = TimeSeries(time_series_->time_interval());
                transform_fit_validation_message_ =
                    "Error: Box-Cox lambda estimation failed. Select a different transform or "
                    "revise the time-series data.";
                return;
            }
            for (int i = 0; i < time_series_->count(); ++i) {
                transformed.add((*time_series_)[i].clone());
                transformed[i].set_value(
                    numerics::data::BoxCox::transform((*time_series_)[i].value(), lambda_));
            }
        } else if (transform_type_ == Transform::YeoJohnson) {
            lambda_ = numerics::data::YeoJohnson::fit_lambda(time_series_->values_to_list());
            if (!numerics::is_finite(lambda_)) {
                lambda_ = 0;
                log_jacobian_ = 0;
                transformed_time_series_ = TimeSeries(time_series_->time_interval());
                diff_series_ = TimeSeries(time_series_->time_interval());
                training_time_series_ = TimeSeries(time_series_->time_interval());
                transform_fit_validation_message_ =
                    "Error: Yeo-Johnson lambda estimation failed. Select a different transform "
                    "or revise the time-series data.";
                return;
            }
            for (int i = 0; i < time_series_->count(); ++i) {
                transformed.add((*time_series_)[i].clone());
                transformed[i].set_value(
                    numerics::data::YeoJohnson::transform((*time_series_)[i].value(), lambda_));
            }
        }
        transformed_time_series_ = transformed;

        // Step 2: difference the transformed series (P2 TimeSeries.Difference(1, d)).
        diff_series_ =
            diff_order_d_ > 0 ? transformed.difference(1, diff_order_d_) : transformed.clone();

        int effective = std::min(training_time_steps_, diff_series_->count());

        // Step 3: training series is the first effective entries of _diffSeries.
        TimeSeries tts(diff_series_->time_interval());
        for (int i = 0; i < effective; ++i) tts.add((*diff_series_)[i].clone());
        training_time_series_ = tts;

        // Step 4: Jacobian on the RAW values whose densities the likelihood evaluates.
        if (transform_type_ == Transform::None) {
            log_jacobian_ = 0;
        } else {
            int start_raw_idx = diff_order_d_ + max_order;
            int end_raw_idx = training_time_steps_ - 1;
            if (end_raw_idx >= start_raw_idx && end_raw_idx < time_series_->count()) {
                std::vector<double> raw =
                    subset(time_series_->values_to_array(), start_raw_idx, end_raw_idx);
                log_jacobian_ = transform_type_ == Transform::YeoJohnson
                                    ? numerics::data::YeoJohnson::log_jacobian(raw, lambda_)
                                    : numerics::data::BoxCox::log_jacobian(raw, lambda_);
            } else {
                log_jacobian_ = 0;
            }
        }
    }

    // Numerics ExtensionMethods.Subset(start, end): inclusive slice, clamped to bounds.
    static std::vector<double> subset(const std::vector<double>& v, int start, int end) {
        std::vector<double> out;
        if (start < 0) start = 0;
        if (end >= static_cast<int>(v.size())) end = static_cast<int>(v.size()) - 1;
        for (int i = start; i <= end; ++i) out.push_back(v[static_cast<std::size_t>(i)]);
        return out;
    }

    // Formats a double to 3 decimals (mirrors the C# "{value:F3}" in the Validate warnings).
    static std::string format3(double v) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.3f", v);
        return buf;
    }

    std::optional<TimeSeries> time_series_;
    std::optional<TimeSeries> transformed_time_series_;
    std::optional<TimeSeries> diff_series_;
    std::optional<TimeSeries> training_time_series_;
    std::optional<std::vector<TimeSeries>> covariates_;

    Transform transform_type_ = Transform::None;
    double lambda_ = 0;
    double lambda2_ = 0;
    double log_jacobian_ = 0;
    // v2.0.0: set by set_training_data() when BoxCox/YeoJohnson FitLambda returns a non-finite
    // lambda; surfaced by validate() (C# _transformFitValidationMessage).
    std::optional<std::string> transform_fit_validation_message_;
    bool include_intercept_ = true;
    bool include_seasonality_ = false;
    int seasonal_period_ = 12;
    Trend trend_type_ = Trend::None;
    int ar_order_p_ = 1;
    int diff_order_d_ = 0;
    int ma_order_q_ = 0;
    int x_order_b_ = 0;
    bool use_jeffreys_rule_for_scale_ = true;
    int training_time_steps_ = 0;
    bool use_default_training_steps_ = true;
    CovariateExtensionMethod covariate_extension_ = CovariateExtensionMethod::BlockBootstrap;
};

}  // namespace corehydro::models
