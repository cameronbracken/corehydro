// corehydro ADDITION -- toolbox_runner.hpp group header, no upstream C# counterpart.
//
// Holds the `regression` group's dispatch arm (fit/covariance/residuals/predict/
// prediction_intervals) plus build_regression, the one helper it needs to assemble a
// LinearRegression from the flattened predictor matrix. Included by toolbox_runner.hpp, which
// defines the shared ToolboxResult/data_at/scalar helpers used here; not meant to be included
// directly.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/regression/linear_regression.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"

namespace corehydro::numerics::support::detail {

// Predictors cross as one flattened row-major vector plus `rows` and `columns`, because the
// binding layer has no matrix type in common between R (column-major) and Python (numpy). The
// wrappers transpose on the way in; this is the one place that layout is assumed.
inline numerics::data::regression::LinearRegression build_regression(
    const std::vector<std::vector<double>>& data, const JsonValue& options,
    const std::string& method) {
    namespace ml = math::linalg;
    const std::vector<double>& flat = data_at(data, 0, "regression", method);
    const std::vector<double>& yv = data_at(data, 1, "regression", method);
    int rows = options.value_or("rows", static_cast<int>(yv.size()));
    int cols = options.value_or("columns", 1);
    bool intercept = options.value_or("intercept", true);
    if (flat.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols))
        throw std::runtime_error("regression predictors hold " + std::to_string(flat.size()) +
                                 " values, expected rows * columns = " +
                                 std::to_string(rows * cols));
    return numerics::data::regression::LinearRegression(ml::Matrix(rows, cols, flat),
                                                         ml::Vector(yv), intercept);
}

inline ToolboxResult run_regression(const std::string& method,
                                    const std::vector<std::vector<double>>& data,
                                    const JsonValue& options) {
    namespace ml = math::linalg;
    numerics::data::regression::LinearRegression fit = build_regression(data, options, method);
    ToolboxResult r;

    if (method == "fit") {
        const std::vector<double>& beta = fit.parameters();
        const std::vector<double>& se = fit.parameter_standard_errors();
        for (std::size_t i = 0; i < beta.size(); ++i) {
            r.values.push_back(beta[i]);
            r.names.push_back("beta_" + std::to_string(i + 1));
        }
        for (std::size_t i = 0; i < se.size(); ++i) {
            r.values.push_back(se[i]);
            r.names.push_back("se_" + std::to_string(i + 1));
        }
        r.values.push_back(fit.r_squared());                                r.names.push_back("r_squared");
        r.values.push_back(fit.adj_r_squared());                            r.names.push_back("adj_r_squared");
        r.values.push_back(fit.standard_error());                           r.names.push_back("sigma");
        r.values.push_back(static_cast<double>(fit.degrees_of_freedom()));  r.names.push_back("df");
        r.values.push_back(static_cast<double>(fit.sample_size()));         r.names.push_back("n");
        return r;
    }
    if (method == "covariance") {
        const ml::Matrix& c = fit.covariance();
        for (int i = 0; i < c.number_of_rows(); ++i)
            for (int j = 0; j < c.number_of_columns(); ++j) r.values.push_back(c(i, j));
        r.dims = {c.number_of_rows(), c.number_of_columns()};
        return r;
    }
    if (method == "residuals") {
        r.values = fit.residuals();
        return r;
    }
    if (method == "predict" || method == "prediction_intervals") {
        const std::vector<double>& nd = data_at(data, 2, "regression", method);
        int nrows = options.at("predict_rows").as_int();
        int ncols = options.value_or("columns", 1);
        ml::Matrix xp(nrows, ncols, nd);
        if (method == "predict") {
            r.values = fit.predict(xp);
            return r;
        }
        double alpha = options.value_or("alpha", 0.1);
        ml::Matrix pi = fit.prediction_intervals(xp, alpha);
        for (int i = 0; i < pi.number_of_rows(); ++i)
            for (int j = 0; j < pi.number_of_columns(); ++j) r.values.push_back(pi(i, j));
        r.names = {"lower", "upper", "mean"};
        r.dims = {pi.number_of_rows(), pi.number_of_columns()};
        return r;
    }
    throw std::runtime_error("unknown regression method: " + method);
}

}  // namespace corehydro::numerics::support::detail
