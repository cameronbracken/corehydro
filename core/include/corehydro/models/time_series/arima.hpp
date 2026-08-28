// ported from: RMC.BestFit/Models/TimeSeries/ARIMA.cs @ c2e6192
//
// v2.0.0 (upstream-sync Task 16, f140c4d + 0d6821d): SetTrainingData's BoxCox/YeoJohnson
// branches now guard against a non-finite fitted lambda (Task 2's hardened fit_lambda already
// returns NaN instead of throwing for a degenerate/unsupported sample -- see box_cox.hpp /
// yeo_johnson.hpp). On failure: lambda_/log_jacobian_ reset to 0, training_time_series_ AND
// diff_series_ both replaced with an empty series, transform_fit_validation_message_ set to the
// C#-exact message text, and the branch returns early (the happy-path jacobian/transform loop
// and the trailing difference() call never run). Validate appends that message (and flips
// is_valid false) as the LAST check, mirroring the C# ordering. Deliberately NOT ported: the C#
// `catch (ArithmeticException)` branch around FitLambda (with its "Solver message: ..." suffix)
// has no C++ analog -- the ported fit_lambda never throws, it always returns a double (possibly
// NaN), so only the C# `!double.IsFinite(_lambda)` branch (the plain message, no solver-text
// suffix) is reachable here. Also SKIPPED (XML/GUI-only, no oracle-visible C++ surface, not in
// this task's scope): the XElement ctor's TrainingTimeSteps/UseDefaultTrainingSteps attribute
// restoration + explicit SetTrainingData() call. Task 21 CLOSED the two v2.0.0 deltas this note
// previously deferred: ResetDefaultTrainingStepsForNewTimeSeries (the TimeSeries-setter
// training-window reset) and the TimeInterval.Irregular Validate guard are both ported below --
// both are model-layer library surface reachable from the public setters, not GUI/XML.
//
// AutoRegressive Integrated Moving Average ARIMA(p,d,q) time-series model:
//   W(t) = (1-B)^d Y(t)  (d successive first-differences of the transformed series)
//   W(t) = mu + phi_1*(W(t-1) - mu) + ... + phi_p*(W(t-p) - mu)
//                + eps(t) + theta_1*eps(t-1) + ... + theta_q*eps(t-q),  eps(t) ~ N(0, sigma^2)
// on top of ModelBase (the shared likelihood surface the Estimation layer optimizes/samples) and
// ISimulatable<std::vector<double>> (C# ISimulatable<double[]>). Supports None/Logarithmic/
// Box-Cox/Yeo-Johnson data transforms, differencing of order d, a training/forecasting split, and
// the optional Jeffreys (1/sigma) scale prior.
//
// Structural mirroring: the class/member/method layout follows ARIMA.cs. The property setters that
// re-run SetTrainingData()/SetDefaultParameters() on change (POrder/DOrder/QOrder/IncludeIntercept/
// TransformType/TimeSeries/TrainingTimeSteps) preserve that effective behavior here; the
// never-mutate rule is RELAXED for these stateful model objects (per .claude/CLAUDE.md), matching
// the upstream mutable WPF-binding design. This file combines the T1 AutoRegressive mean-centered
// recursion and MovingAverage CSS error recursion into the ARMA-on-differenced core, adding the
// differencing (SetTrainingData -> _diffSeries) and the reverse-differencing integration in
// Predict; the mirrored copies of the shared plumbing are the sanctioned choice here (this project
// favors STRUCTURAL MIRRORING over premature helper extraction).
//
// KEY DIFFERENCES FROM AutoRegressive/MovingAverage (C# governs, verified against ARIMA.cs):
//   - Residuals/likelihood run the ARMA recursion on the DIFFERENCED series _diffSeries, with the
//     conditional Gaussian summed from t = maxOrder = max(POrder, QOrder) (ARIMA.cs:688).
//   - effectiveTrainingSteps = min(TrainingTimeSteps - DOrder, _diffSeries.Count) (ARIMA.cs:619).
//   - IsStationary uses the sum-abs test for POrder >= 2 (NOT AutoRegressive's exact order-2
//     triangle); IsStationary/IsInvertible short-circuit true when the respective order is 0.
//   - GenerateRandomValues seeds the MersenneTwister on seed >= 0 and draws eps ~ Normal(0, sigma)
//     directly (like AR), combining AR + MA terms (ARIMA.cs:1242).
//   - Validate allows POrder/QOrder == 0 (0..10), requires at least one > 0, bounds DOrder to
//     0..2, and guards TimeSeries.Count <= maxOrder + DOrder (ARIMA.cs:1155).
//
// Deliberately NOT ported (documented; C# governs -- deviations noted in task-T2-report.md):
//   - XML (ToXElement / the XElement ctor, ARIMA.cs:77-119,1080): project-wide non-port.
//   - INotifyPropertyChanged / RaisePropertyChange / Parameter_PropertyChanged /
//     TimeSeries_CollectionChanged: WPF data-binding plumbing, ported as silent no-ops
//     (parameters_ is a plain std::vector; no change notification is threaded through it).
//   - IModel Clone() (ARIMA.cs:1055): the C++ core has no virtual IModel::Clone (see
//     model_base.hpp) and no T2 fit path needs a clone, so it is omitted (T1 / S4 precedent).
//   - GenerateRandomSeries (ARIMA.cs:1030): returns a date-stamped TimeSeries rather than the
//     bare vector generate_random_values (the ISimulatable member) returns. Not ported: no caller
//     in scope needs the dates. Since P6 ported the container, this is a one-method addition
//     whenever one does.
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

class ARIMA : public ModelBase, public ISimulatable<std::vector<double>> {
   public:
    using TimeSeries = numerics::data::TimeSeries;

    // Prediction decomposition (C# tuple (double[] Y, InterceptPart, ARPart, MAPart)).
    struct PredictResult {
        std::vector<double> y;
        std::vector<double> intercept_part;
        std::vector<double> ar_part;
        std::vector<double> ma_part;
    };

    // --- Construction (ARIMA.cs:46-70). ---

    // Empty ARIMA(1,0,0) model with default orders and intercept (C#:46).
    ARIMA() {
        p_order_ = 1;
        d_order_ = 0;
        q_order_ = 0;
        include_intercept_ = true;
        set_default_parameters();
    }

    // ARIMA model over the given time series (C#:63). Setting the series triggers the same
    // SetDefaultTrainingSteps/SetTrainingData/SetDefaultParameters cascade as the C# setter.
    explicit ARIMA(const TimeSeries& time_series, int p_order = 1, int d_order = 0, int q_order = 0,
                   bool include_intercept = true) {
        p_order_ = p_order;
        d_order_ = d_order;
        q_order_ = q_order;
        include_intercept_ = include_intercept;
        set_time_series(time_series);
    }

    // --- Properties. ---

    bool has_time_series() const { return time_series_.has_value(); }
    const TimeSeries& time_series() const { return *time_series_; }
    void set_time_series(const TimeSeries& value) {
        // C# unsubscribes/subscribes CollectionChanged here -> no-op in this port.
        time_series_ = value;
        reset_default_training_steps_for_new_time_series();
        set_training_data();
        // RaisePropertyChange -> no-op.
        if (use_default_flat_priors()) set_default_parameters();
    }

    int p_order() const { return p_order_; }
    void set_p_order(int value) {
        if (p_order_ != value) {
            p_order_ = value;
            set_default_parameters();
        }
    }

    // DOrder setter re-runs SetTrainingData() (re-differences _diffSeries) before
    // SetDefaultParameters (ARIMA.cs:207-217); POrder/QOrder setters do not.
    int d_order() const { return d_order_; }
    void set_d_order(int value) {
        if (d_order_ != value) {
            d_order_ = value;
            set_training_data();
            set_default_parameters();
        }
    }

    int q_order() const { return q_order_; }
    void set_q_order(int value) {
        if (q_order_ != value) {
            q_order_ = value;
            set_default_parameters();
        }
    }

    bool include_intercept() const { return include_intercept_; }
    void set_include_intercept(bool value) {
        if (include_intercept_ != value) {
            include_intercept_ = value;
            set_default_parameters();
        }
    }

    Transform transform_type() const { return transform_type_; }
    void set_transform_type(Transform value) {
        if (transform_type_ != value) {
            transform_type_ = value;
            set_training_data();
            set_default_parameters();
        }
    }

    // Transformed training series used for calibration (C# TrainingTimeSeries).
    bool has_training_time_series() const { return training_time_series_.has_value(); }
    const TimeSeries& training_time_series() const { return *training_time_series_; }

    // Differenced series after transformation (C# DifferencedSeries, ARIMA.cs:291).
    bool has_differenced_series() const { return diff_series_.has_value(); }
    const TimeSeries& differenced_series() const { return *diff_series_; }

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

    // Number of steps reserved for out-of-sample forecasting (C#:345).
    int forecasting_time_steps() const {
        return time_series_ ? time_series_->count() - training_time_steps_ : 0;
    }

    bool use_jeffreys_rule_for_scale() const { return use_jeffreys_rule_for_scale_; }
    void set_use_jeffreys_rule_for_scale(bool value) {
        if (use_jeffreys_rule_for_scale_ != value) use_jeffreys_rule_for_scale_ = value;
    }

    // Sets the transform parameters manually (C#:510).
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

    // --- SetDefaultParameters (C#:517). Data-driven bounds from _diffSeries when non-empty. ---
    void set_default_parameters() override {
        parameters_.clear();  // C# `_parameters = new List<>()`; PropertyChanged unsub -> no-op.

        double mean = 0, sigma = 1, min = -10, max = 10, sigma_ub = 10;

        if (diff_series_ && diff_series_->count() > 0) {
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

            sigma_ub = std::pow(10.0, std::ceil(std::log10(sigma) + 1.0));
            if (std::isnan(sigma_ub) || std::isinf(sigma_ub)) sigma_ub = 100;
        }

        if (include_intercept_) {
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/"Intercept (\xCE\xBC)", /*value=*/mean,
                /*lower_bound=*/min, /*upper_bound=*/max,
                std::make_unique<numerics::distributions::Uniform>(min, max));
        }

        for (int i = 1; i <= p_order_; ++i) {
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/std::string("AR (\xCF\x86") + to_subscript(i) + ")",
                /*value=*/0.0, /*lower_bound=*/-2.0, /*upper_bound=*/2.0,
                std::make_unique<numerics::distributions::Uniform>(-2.0, 2.0));
        }

        for (int i = 1; i <= q_order_; ++i) {
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/std::string("MA (\xCE\xB8") + to_subscript(i) + ")",
                /*value=*/0.0, /*lower_bound=*/-2.0, /*upper_bound=*/2.0,
                std::make_unique<numerics::distributions::Uniform>(-2.0, 2.0));
        }

        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/"Scale (\xCF\x83)", /*value=*/sigma,
            /*lower_bound=*/numerics::kDoubleMachineEpsilon, /*upper_bound=*/sigma_ub,
            std::make_unique<numerics::distributions::Uniform>(numerics::kDoubleMachineEpsilon,
                                                               sigma_ub),
            /*is_positive=*/true);
    }

    // --- SetParameterValues (C#:600): count-check then write each value. ---
    // The base ModelBase::set_parameter_values body is byte-for-byte the C# body; not overridden.

    // --- Residuals (C#:616): ARMA recursion on _diffSeries, warm-up res/eps=0 for t<maxOrder. ---
    std::vector<double> residuals(const std::vector<double>& parameters) const {
        int effective = diff_series_ ? std::min(training_time_steps_ - d_order_, diff_series_->count())
                                     : training_time_steps_;
        if (effective < 0) effective = 0;
        std::vector<double> res(static_cast<std::size_t>(effective), 0.0);
        std::vector<double> epsilon(static_cast<std::size_t>(effective), 0.0);
        int max_order = std::max(p_order_, q_order_);

        int k = 0;
        double mu = include_intercept_ ? parameters[static_cast<std::size_t>(k++)] : 0.0;
        std::vector<double> phi(static_cast<std::size_t>(p_order_));
        for (int i = 0; i < p_order_; ++i) phi[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(k++)];
        std::vector<double> theta(static_cast<std::size_t>(q_order_));
        for (int i = 0; i < q_order_; ++i) theta[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(k++)];

        for (int t = 0; t < effective; ++t) {
            if (t < max_order) {
                res[static_cast<std::size_t>(t)] = 0.0;
                epsilon[static_cast<std::size_t>(t)] = 0.0;
            } else {
                double prediction = mu;
                for (int p = 1; p <= p_order_; ++p)
                    prediction += phi[static_cast<std::size_t>(p - 1)] *
                                  ((*diff_series_)[t - p].value() - mu);
                for (int q = 1; q <= std::min(t, q_order_); ++q)
                    prediction += theta[static_cast<std::size_t>(q - 1)] *
                                  epsilon[static_cast<std::size_t>(t - q)];
                epsilon[static_cast<std::size_t>(t)] = (*diff_series_)[t].value() - prediction;
                res[static_cast<std::size_t>(t)] = epsilon[static_cast<std::size_t>(t)];
            }
        }
        return res;
    }

    // --- DataLogLikelihood (C#:667): conditional Gaussian from t = maxOrder + logJacobian. ---
    double data_log_likelihood(std::vector<double>& parameters) const override {
        if (!diff_series_ || diff_series_->count() == 0)
            return -std::numeric_limits<double>::infinity();

        for (double v : parameters)
            if (std::isnan(v)) return -std::numeric_limits<double>::infinity();

        std::vector<double> res = residuals(parameters);
        double sigma = parameters.back();
        if (sigma <= 0) return -std::numeric_limits<double>::infinity();
        numerics::distributions::Normal norm(0.0, sigma);
        double log_lh = 0.0;
        int max_order = std::max(p_order_, q_order_);
        for (int t = max_order; t < static_cast<int>(res.size()); ++t)
            log_lh += norm.log_pdf(res[static_cast<std::size_t>(t)]);
        return log_lh + log_jacobian_;
    }

    // --- PointwiseDataLogLikelihood (C#:697). ---
    std::vector<double> pointwise_data_log_likelihood(
        const std::vector<double>& parameters) const override {
        if (!diff_series_ || diff_series_->count() == 0) return {};

        int effective = std::min(training_time_steps_ - d_order_, diff_series_->count());
        int max_order = std::max(p_order_, q_order_);
        int n = effective - max_order;
        if (n <= 0) return {};

        for (double v : parameters)
            if (std::isnan(v))
                return std::vector<double>(static_cast<std::size_t>(n),
                                           -std::numeric_limits<double>::infinity());

        std::vector<double> res = residuals(parameters);
        double sigma = parameters.back();
        numerics::distributions::Normal norm(0.0, sigma);
        std::vector<double> result(static_cast<std::size_t>(n));

        double jacobian_per_obs = log_jacobian_ / n;
        int idx = 0;
        for (int t = max_order; t < static_cast<int>(res.size()); ++t)
            result[static_cast<std::size_t>(idx++)] =
                norm.log_pdf(res[static_cast<std::size_t>(t)]) + jacobian_per_obs;
        return result;
    }

    // --- PointwiseDataLogLikelihoodComponents (C#:733). ---
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& parameters) const override {
        if (!diff_series_ || diff_series_->count() == 0) return {};

        int effective = std::min(training_time_steps_ - d_order_, diff_series_->count());
        int max_order = std::max(p_order_, q_order_);
        int n = effective - max_order;
        if (n <= 0) return {};

        std::vector<DataComponent> result;
        result.reserve(static_cast<std::size_t>(n));
        std::vector<double> response = diff_series_->values_to_array();

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

        std::vector<double> res = residuals(parameters);
        double sigma = parameters.back();
        numerics::distributions::Normal norm(0.0, sigma);
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

    // --- PriorLogLikelihood (C#:777): parameter priors + optional Jeffreys 1/sigma. ---
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

    // --- PointwisePriorLogLikelihood (C#:802). ---
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
            result.emplace_back("Jeffreys' rule for \xCF\x83", ll,
                                PriorComponentType::ParameterPrior);
        }
        return result;
    }

    // --- Predict (C#:830). Runs the ARMA recursion on the differenced scale, then integrates
    //     (reverse-differencing) back up through the differencing levels. Returns Y + parts. ---
    PredictResult predict_components(const std::vector<double>& parameters, int forecast_steps = 0,
                                     int seed = -1) const {
        if (!time_series_) throw std::runtime_error("TimeSeries must be set.");

        int total_steps = training_time_steps_ + forecast_steps;
        if (total_steps < 0) total_steps = 0;

        std::vector<double> y(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> intercept_part(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> ar_part(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> ma_part(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> epsilon(static_cast<std::size_t>(total_steps), 0.0);

        int k = 0;
        double mu = include_intercept_ ? parameters[static_cast<std::size_t>(k++)] : 0.0;
        std::vector<double> phi(static_cast<std::size_t>(p_order_));
        for (int i = 0; i < p_order_; ++i) phi[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(k++)];
        std::vector<double> theta(static_cast<std::size_t>(q_order_));
        for (int i = 0; i < q_order_; ++i) theta[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(k++)];
        double sigma = parameters[static_cast<std::size_t>(k)];

        // C# uses System.Random here (no ported equivalent); ported MersenneTwister substituted.
        std::optional<numerics::sampling::MersenneTwister> prng;
        std::optional<numerics::distributions::Normal> err_dist;
        if (seed >= 0) {
            prng.emplace(static_cast<std::uint32_t>(seed));
            err_dist.emplace(0.0, sigma);
        }

        int max_order = std::max(p_order_, q_order_);

        for (int t = 0; t < total_steps; ++t) {
            intercept_part[static_cast<std::size_t>(t)] = mu;

            double ar = 0.0;
            double ma = 0.0;
            if (t >= max_order) {
                for (int p = 1; p <= p_order_; ++p) {
                    if (diff_series_ && t - p < training_time_steps_ &&
                        t - p < diff_series_->count()) {
                        ar += phi[static_cast<std::size_t>(p - 1)] *
                              ((*diff_series_)[t - p].value() - mu);
                    } else {
                        ar += phi[static_cast<std::size_t>(p - 1)] *
                              (y[static_cast<std::size_t>(t - p)] - mu);
                    }
                }
                for (int q = 1; q <= std::min(t, q_order_); ++q)
                    ma += theta[static_cast<std::size_t>(q - 1)] *
                          epsilon[static_cast<std::size_t>(t - q)];
            }
            ar_part[static_cast<std::size_t>(t)] = ar;
            ma_part[static_cast<std::size_t>(t)] = ma;

            if (t < max_order && diff_series_ && t < diff_series_->count()) {
                y[static_cast<std::size_t>(t)] = (*diff_series_)[t].value();
                epsilon[static_cast<std::size_t>(t)] = 0.0;
            } else {
                y[static_cast<std::size_t>(t)] = mu + ar + ma;
            }

            if (diff_series_ && t >= max_order && t < training_time_steps_ &&
                t < diff_series_->count()) {
                epsilon[static_cast<std::size_t>(t)] =
                    (*diff_series_)[t].value() - y[static_cast<std::size_t>(t)];
            }

            if (prng && t >= max_order) {
                double mt = y[static_cast<std::size_t>(t)];
                double error = err_dist->inverse_cdf(prng->next_double());
                y[static_cast<std::size_t>(t)] += error;
                if (t >= training_time_steps_)
                    epsilon[static_cast<std::size_t>(t)] = y[static_cast<std::size_t>(t)] - mt;
            }
        }

        // Step A -- integrate (reverse differencing) up through the differencing levels. For
        // DOrder >= 2 we precompute the chain of intermediate differences and anchor each level
        // to the appropriate observed intermediate-difference inside the fit window (ARIMA.cs:941).
        if (d_order_ > 0) {
            std::vector<double> integrated = y;

            std::vector<TimeSeries> anchor_series;
            bool has_anchor = training_time_series_ && training_time_series_->count() > 0;
            if (has_anchor) {
                anchor_series.push_back(*training_time_series_);
                for (int level = 1; level < d_order_; ++level)
                    anchor_series.push_back(difference(anchor_series[static_cast<std::size_t>(level - 1)], 1));
            }

            for (int d = 0; d < d_order_; ++d) {
                int anchor_level = d_order_ - 1 - d;
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

        return {std::move(y), std::move(intercept_part), std::move(ar_part), std::move(ma_part)};
    }

    // Predict with the current parameter values (C#:1018), returning Y only.
    std::vector<double> predict(int forecast_steps = 0, int seed = -1) const {
        return predict_components(parameter_values(), forecast_steps, seed).y;
    }

    // --- IsStationary (C#:1105). POrder >= 2 uses the conservative sum-abs test. ---
    bool is_stationary() const {
        if (p_order_ == 0 || parameters_.empty()) return true;

        int k = include_intercept_ ? 1 : 0;
        if (static_cast<int>(parameters_.size()) <= k) return true;

        std::vector<double> phi(static_cast<std::size_t>(p_order_));
        for (int i = 0; i < p_order_; ++i)
            phi[static_cast<std::size_t>(i)] = parameters_[static_cast<std::size_t>(k + i)].value();

        if (p_order_ == 1) return std::fabs(phi[0]) < 1.0;

        double sum_abs_phi = 0.0;
        for (int i = 0; i < p_order_; ++i) sum_abs_phi += std::fabs(phi[static_cast<std::size_t>(i)]);
        return sum_abs_phi < 1.0;
    }

    // --- IsInvertible (C#:1131). QOrder >= 2 uses the conservative sum-abs test. ---
    bool is_invertible() const {
        if (q_order_ == 0 || parameters_.empty()) return true;

        int k = include_intercept_ ? 1 : 0;
        k += p_order_;
        if (static_cast<int>(parameters_.size()) <= k) return true;

        std::vector<double> theta(static_cast<std::size_t>(q_order_));
        for (int i = 0; i < q_order_; ++i)
            theta[static_cast<std::size_t>(i)] = parameters_[static_cast<std::size_t>(k + i)].value();

        if (q_order_ == 1) return std::fabs(theta[0]) < 1.0;

        double sum_abs_theta = 0.0;
        for (int i = 0; i < q_order_; ++i)
            sum_abs_theta += std::fabs(theta[static_cast<std::size_t>(i)]);
        return sum_abs_theta < 1.0;
    }

    // --- Validate (C#:1155). ---
    ValidationResult validate() const override {
        ValidationResult result;  // is_valid = true

        if (!time_series_) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: Time series is null.");
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
        if (p_order_ < 0 || p_order_ > 10) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: AR order (p) must be between 0 and 10.");
        }
        if (d_order_ < 0 || d_order_ > 2) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Differencing order (d) must be between 0 and 2.");
        }
        if (q_order_ < 0 || q_order_ > 10) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: MA order (q) must be between 0 and 10.");
        }
        if (p_order_ == 0 && q_order_ == 0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: At least one of AR order (p) or MA order (q) must be greater than 0.");
        }
        if (training_time_steps_ > time_series_->count()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Training time steps cannot exceed time series length.");
        }

        int max_order = std::max(p_order_, q_order_);
        if (time_series_->count() <= max_order + d_order_) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Time series length (" + std::to_string(time_series_->count()) +
                ") must exceed max order + differencing (" + std::to_string(max_order + d_order_) +
                ").");
        }

        if (d_order_ > 0 && training_time_steps_ > time_series_->count() - d_order_) {
            result.validation_messages.push_back(
                "Warning: TrainingTimeSteps (" + std::to_string(training_time_steps_) +
                ") exceeds the differenced series length (" +
                std::to_string(time_series_->count() - d_order_) +
                "). Effective training will use " +
                std::to_string(time_series_->count() - d_order_) + " time steps.");
        }

        for (const auto& p : parameters_) {
            ValidationResult v = p.validate();
            if (!v.is_valid) {
                result.is_valid = false;
                for (const auto& m : v.validation_messages) result.validation_messages.push_back(m);
            }
        }
        if (p_order_ > 0 && !is_stationary()) {
            // C# message uses Greek/em-dash glyphs; ASCII-normalized here (keeps the "Warning" /
            // "stationarity" tokens the tests key on). See task-T2-report.md.
            result.validation_messages.push_back(
                "Warning: AR parameters do not satisfy the sum-of-absolute-values stationarity "
                "sufficient condition (sum|phi_i| < 1). The model may still be stationary - this "
                "check is conservative for orders >= 3 - but forecasts may be unstable if it is "
                "not.");
        }
        if (q_order_ > 0 && !is_invertible()) {
            result.validation_messages.push_back(
                "Warning: MA parameters do not satisfy the sum-of-absolute-values invertibility "
                "sufficient condition (sum|theta_i| < 1). The model may still be invertible - this "
                "check is conservative for orders >= 2 - but uniqueness of the MA representation is "
                "not guaranteed if it is not.");
        }
        if (transform_fit_validation_message_) {
            result.is_valid = false;
            result.validation_messages.push_back(*transform_fit_validation_message_);
        }
        return result;
    }

    // --- GenerateRandomValues (C#:1242): ISimulatable entry point. Bit-exact MersenneTwister. ---
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size <= 0)
            throw std::out_of_range("Sample size must be positive.");

        numerics::sampling::MersenneTwister rng =
            seed >= 0 ? numerics::sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                      : numerics::sampling::MersenneTwister();

        int param_index = 0;
        double intercept = 0.0;
        if (include_intercept_) {
            intercept = parameters_[static_cast<std::size_t>(param_index)].value();
            param_index++;
        }
        std::vector<double> phi(static_cast<std::size_t>(p_order_));
        for (int i = 0; i < p_order_; ++i)
            phi[static_cast<std::size_t>(i)] =
                parameters_[static_cast<std::size_t>(param_index + i)].value();
        param_index += p_order_;

        std::vector<double> theta(static_cast<std::size_t>(q_order_));
        for (int i = 0; i < q_order_; ++i)
            theta[static_cast<std::size_t>(i)] =
                parameters_[static_cast<std::size_t>(param_index + i)].value();
        param_index += q_order_;

        double sigma = parameters_[static_cast<std::size_t>(param_index)].value();
        numerics::distributions::Normal normal(0.0, sigma);

        std::vector<double> series(static_cast<std::size_t>(sample_size));
        std::vector<double> epsilon(static_cast<std::size_t>(sample_size));

        for (int t = 0; t < sample_size; ++t)
            epsilon[static_cast<std::size_t>(t)] = normal.inverse_cdf(rng.next_double());

        int max_order = std::max(p_order_, q_order_);
        for (int t = 0; t < max_order; ++t)
            series[static_cast<std::size_t>(t)] = intercept + epsilon[static_cast<std::size_t>(t)];

        for (int t = max_order; t < sample_size; ++t) {
            double value = intercept;
            for (int j = 0; j < p_order_; ++j)
                value += phi[static_cast<std::size_t>(j)] *
                         (series[static_cast<std::size_t>(t - 1 - j)] - intercept);
            value += epsilon[static_cast<std::size_t>(t)];
            for (int j = 0; j < q_order_; ++j)
                value += theta[static_cast<std::size_t>(j)] *
                         epsilon[static_cast<std::size_t>(t - 1 - j)];
            series[static_cast<std::size_t>(t)] = value;
        }
        return series;
    }

   private:
    // --- SetDefaultTrainingSteps (C#:386): 80% of data, min 30 or parameter count. ---
    // --- ResetDefaultTrainingStepsForNewTimeSeries (ARIMA.cs:411) -----------------------------------
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

    // --- Difference (C#:397): d successive first-differences; clones ordinates (drops index 0
    //     each pass). Value/count match the P2 TimeSeries::difference; ported here directly to
    //     mirror the C# private method and its no-throw behavior on short series. ---
    static TimeSeries difference(const TimeSeries& series, int order) {
        if (order <= 0) return series;  // C# returns the same reference; we return a copy.
        TimeSeries result = series;
        for (int d = 0; d < order; ++d) {
            TimeSeries diffed(result.time_interval());
            for (int i = 1; i < result.count(); ++i) {
                TimeSeries::Ordinate ord = result[i];  // clone
                ord.set_value(result[i].value() - result[i - 1].value());
                diffed.add(ord);
            }
            result = std::move(diffed);
        }
        return result;
    }

    // --- SetTrainingData (C#:419): apply the active transform, then difference by DOrder. The
    //     transform jacobian window starts at max(0, DOrder + maxOrder) (ARIMA.cs:441/460/479). ---
    void set_training_data() {
        transform_fit_validation_message_.reset();
        if (!time_series_ || training_time_steps_ == 0) return;

        int effective = std::min(training_time_steps_, time_series_->count());
        TimeSeries tts(time_series_->time_interval());
        int max_order = std::max(p_order_, q_order_);
        int start_idx = std::max(0, d_order_ + max_order);
        int end_idx = effective - 1;

        if (transform_type_ == Transform::None) {
            lambda_ = 0;
            log_jacobian_ = 0;
            for (int i = 0; i < effective; ++i) tts.add((*time_series_)[i].clone());
        } else if (transform_type_ == Transform::Logarithmic) {
            lambda_ = 0;
            if (end_idx >= start_idx) {
                std::vector<double> data = subset(time_series_->values_to_array(), start_idx, end_idx);
                log_jacobian_ = numerics::data::BoxCox::log_jacobian(data, lambda_);
            }
            for (int i = 0; i < effective; ++i) {
                tts.add((*time_series_)[i].clone());
                tts[i].set_value(numerics::data::BoxCox::transform((*time_series_)[i].value(), lambda_));
            }
        } else if (transform_type_ == Transform::BoxCox) {
            lambda_ = numerics::data::BoxCox::fit_lambda(time_series_->values_to_list());
            if (!numerics::is_finite(lambda_)) {
                lambda_ = 0;
                log_jacobian_ = 0;
                training_time_series_ = TimeSeries(time_series_->time_interval());
                diff_series_ = TimeSeries(time_series_->time_interval());
                transform_fit_validation_message_ =
                    "Error: Box-Cox lambda estimation failed. Select a different transform or "
                    "revise the time-series data.";
                return;
            }
            if (end_idx >= start_idx) {
                std::vector<double> data = subset(time_series_->values_to_array(), start_idx, end_idx);
                log_jacobian_ = numerics::data::BoxCox::log_jacobian(data, lambda_);
            }
            for (int i = 0; i < effective; ++i) {
                tts.add((*time_series_)[i].clone());
                tts[i].set_value(numerics::data::BoxCox::transform((*time_series_)[i].value(), lambda_));
            }
        } else if (transform_type_ == Transform::YeoJohnson) {
            lambda_ = numerics::data::YeoJohnson::fit_lambda(time_series_->values_to_list());
            if (!numerics::is_finite(lambda_)) {
                lambda_ = 0;
                log_jacobian_ = 0;
                training_time_series_ = TimeSeries(time_series_->time_interval());
                diff_series_ = TimeSeries(time_series_->time_interval());
                transform_fit_validation_message_ =
                    "Error: Yeo-Johnson lambda estimation failed. Select a different transform "
                    "or revise the time-series data.";
                return;
            }
            if (end_idx >= start_idx) {
                std::vector<double> data = subset(time_series_->values_to_array(), start_idx, end_idx);
                log_jacobian_ = numerics::data::YeoJohnson::log_jacobian(data, lambda_);
            }
            for (int i = 0; i < effective; ++i) {
                tts.add((*time_series_)[i].clone());
                tts[i].set_value(
                    numerics::data::YeoJohnson::transform((*time_series_)[i].value(), lambda_));
            }
        }

        training_time_series_ = tts;
        diff_series_ = difference(tts, d_order_);
    }

    // Numerics ExtensionMethods.Subset(start, end): inclusive slice, clamped to bounds.
    static std::vector<double> subset(const std::vector<double>& v, int start, int end) {
        std::vector<double> out;
        if (start < 0) start = 0;
        if (end >= static_cast<int>(v.size())) end = static_cast<int>(v.size()) - 1;
        for (int i = start; i <= end; ++i) out.push_back(v[static_cast<std::size_t>(i)]);
        return out;
    }

    std::optional<TimeSeries> time_series_;
    std::optional<TimeSeries> training_time_series_;
    std::optional<TimeSeries> diff_series_;
    int p_order_ = 1;
    int d_order_ = 0;
    int q_order_ = 0;
    bool include_intercept_ = true;

    Transform transform_type_ = Transform::None;
    double lambda_ = 0;
    double lambda2_ = 0;
    double log_jacobian_ = 0;
    // v2.0.0: set by set_training_data() when BoxCox/YeoJohnson FitLambda returns a non-finite
    // lambda; surfaced by validate() (C# _transformFitValidationMessage).
    std::optional<std::string> transform_fit_validation_message_;

    int training_time_steps_ = 0;
    bool use_default_training_steps_ = true;
    bool use_jeffreys_rule_for_scale_ = true;
};

}  // namespace corehydro::models
