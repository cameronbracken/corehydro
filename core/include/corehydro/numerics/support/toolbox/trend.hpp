// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `trend` group's `predict`/`parameters` dispatch arm over
// models::spec::build_spec_trend's `{"type": ..., "start_index"?: i, "values"?: [...]}` spec
// (model_spec.hpp): `type` is one of the eleven `TrendModelType` names, `start_index` and
// `values` are both optional and applied on top of the trend's own class defaults. Includes
// model_spec.hpp (the whole Models layer) rather than only trend_model_factory.hpp because
// build_spec_trend itself lives there, next to parse_trend_model_type -- see that file's header
// for why a standalone trend builder is a new, narrower operation than the model-attached
// `trends` array loop, not a second copy of it.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/model_spec.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/trend_functions/support/i_trend_model.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

inline ToolboxResult run_trend(const std::string& method,
                               const std::vector<std::vector<double>>& data,
                               const JsonValue& options) {
    if (!options.contains("trend"))
        throw std::runtime_error("toolbox group 'trend' needs a 'trend' spec in its options");
    std::unique_ptr<models::trend_functions::ITrendModel> t =
        models::spec::build_spec_trend(options.at("trend"));
    if (method == "predict") {
        ToolboxResult r;
        for (double i : data_at(data, 0, "trend", method))
            r.values.push_back(t->predict(static_cast<int>(i)));
        return r;
    }
    if (method == "parameters") {
        ToolboxResult r;
        for (const models::ModelParameter& p : t->parameters()) {
            r.values.push_back(p.value());
            r.names.push_back(p.name());
        }
        return r;
    }
    throw std::runtime_error("unknown trend method: " + method);
}

}  // namespace corehydro::numerics::support::detail
