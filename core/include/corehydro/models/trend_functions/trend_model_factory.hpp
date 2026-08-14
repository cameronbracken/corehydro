// corehydro ADDITION -- no upstream C# counterpart.
//
// The type -> concrete-trend-class mapping, hoisted out of
// UnivariateDistributionModel::make_trend_model (a private static helper in
// univariate_distribution_model.hpp) so there is exactly one place that maps a
// TrendModelType to a freshly class-defaulted ITrendModel. Two callers use it:
// UnivariateDistributionModel::set_trend_model (univariate_distribution_model_trends.hpp),
// which takes the instance this returns and further data-drives its parameter values from the
// model's distribution and DataFrame before installing it; and
// models::spec::build_spec_trend (model_spec.hpp), which builds a standalone trend for direct
// evaluation (the toolbox `trend` group; numerics/support/toolbox/trend.hpp) and applies only
// what a spec gives explicitly (`start_index`/`values`) on top of the class defaults.
//
// Those two callers are NOT unified into one operation: set_trend_model's default-value
// computation depends on the owning model's fitted distribution and observed data (see its own
// header comment in univariate_distribution_model_trends.hpp), which a bare TrendModelType has
// no access to. This factory is the part of that path that IS state-free -- a pure type ->
// instance switch -- and is the one piece safe to share without inventing behavior neither
// caller had before.
//
// GeneralLinear falls through to ConstantTrend, mirroring the C# `SetTrendModel` if-chain
// (Models/UnivariateDistribution.cs line ~817), which has no GeneralLinear branch: that
// trend needs a covariate matrix and an owner name at construction
// (GeneralLinearFunction(ownerName, covariates)), neither of which a bare TrendModelType
// carries, so upstream itself never reaches it through this dispatch. The C++ mirrors that
// omission via the same `default:` fallthrough used here.
#pragma once
#include <memory>

#include "corehydro/models/trend_functions/constant_trend.hpp"
#include "corehydro/models/trend_functions/cubic_trend.hpp"
#include "corehydro/models/trend_functions/exponential_trend.hpp"
#include "corehydro/models/trend_functions/linear_trend.hpp"
#include "corehydro/models/trend_functions/logistic_trend.hpp"
#include "corehydro/models/trend_functions/power_trend.hpp"
#include "corehydro/models/trend_functions/quadratic_trend.hpp"
#include "corehydro/models/trend_functions/reciprocal_trend.hpp"
#include "corehydro/models/trend_functions/sinusoidal_trend.hpp"
#include "corehydro/models/trend_functions/step_function.hpp"
#include "corehydro/models/trend_functions/support/i_trend_model.hpp"
#include "corehydro/models/trend_functions/support/trend_model_type.hpp"

namespace corehydro::models::trend_functions {

inline std::unique_ptr<ITrendModel> make_trend_model(TrendModelType type) {
    switch (type) {
        case TrendModelType::Cubic:
            return std::make_unique<CubicTrend>();
        case TrendModelType::Exponential:
            return std::make_unique<ExponentialTrend>();
        case TrendModelType::Linear:
            return std::make_unique<LinearTrend>();
        case TrendModelType::Logistic:
            return std::make_unique<LogisticTrend>();
        case TrendModelType::Power:
            return std::make_unique<PowerTrend>();
        case TrendModelType::Quadratic:
            return std::make_unique<QuadraticTrend>();
        case TrendModelType::Reciprocal:
            return std::make_unique<ReciprocalTrend>();
        case TrendModelType::Sinusoidal:
            return std::make_unique<SinusoidalTrend>();
        case TrendModelType::StepFunction:
            return std::make_unique<StepFunction>();
        default:
            return std::make_unique<ConstantTrend>();
    }
}

}  // namespace corehydro::models::trend_functions
