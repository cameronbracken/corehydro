// corehydro ADDITION -- toolbox group header, no upstream C# counterpart.
//
// Holds the `ml` group's dispatch arms over the ported Machine Learning layer
// (numerics/machine_learning/, a port of the C# `Numerics.MachineLearning` namespace): the five
// supervised learners (DecisionTree, RandomForest, KNearestNeighbors, NaiveBayes,
// GeneralizedLinearModel) and the three unsupervised ones (KMeans, GaussianMixtureModel,
// JenksNaturalBreaks).
//
// DATA LAYOUT, identical to the `regression` group's and assumed in exactly one place (the
// `build_*` helpers below):
//   data[0]  the training predictor matrix, flattened ROW-MAJOR
//   data[1]  the response vector (EMPTY for the three unsupervised methods)
//   data[2]  the optional new-data matrix, flattened row-major
// with `rows`, `columns` and `predict_rows` in `options`. The binding layer transposes on the way
// in, because R is column-major and numpy is row-major and there is no matrix type in common.
// `jenks_*` is the one exception: it reads `data[0]` as a plain vector, since it classifies one
// dimension.
//
// EVERY METHOD RE-TRAINS. There is no fitted-object cache here, exactly as the `regression`
// group's `fit`/`covariance`/`residuals` each rebuild their `LinearRegression`. Seeded training
// is deterministic, so two calls agree; the cost is only visible for `random_forest_predict`,
// whose one call already returns everything a caller needs (all four interval columns).
//
// Methods, their options and their result shapes:
//
//   kmeans_means           dims {k, p}          options: k, seed, kmeans_plus_plus, max_iterations
//   kmeans_labels          n values (0-based)   same
//   kmeans_iterations      scalar               same
//   gmm_means              dims {k, p}          options: the kmeans set plus tolerance
//   gmm_weights            k values             same
//   gmm_labels             n values (0-based)   same
//   gmm_sigmas             dims {k * p, p}      same (component c occupies rows c*p .. c*p+p-1)
//   gmm_log_likelihood     scalar               same
//   gmm_iterations         scalar               same
//   jenks_breaks           k values             options: n_clusters, is_data_sorted
//   jenks_gvf              scalar               same
//   jenks_clusters         dims {k, 8}, named   same
//   decision_tree_predict  n_test values        options: seed, is_regression, features,
//                                                        minimum_split_size, max_depth
//   random_forest_predict  dims {n_test, 4}     the tree set plus number_of_trees, alpha
//                          names lower/median/upper/mean
//   knn_predict            n_test values        options: k, is_regression
//   knn_neighbors          dims {n_test, k}     options: k
//   knn_bootstrap_predict  n_test values        options: k, is_regression, seed
//   knn_prediction_intervals  dims {n_test, 4}  options: k, seed, realizations, alpha
//   naive_bayes_means      dims {c, p}          (no options)
//   naive_bayes_sds        dims {c, p}          (no options)
//   naive_bayes_priors     c values             (no options)
//   naive_bayes_classes    c values             (no options)
//   naive_bayes_predict    n_test values        (no options)
//   glm_fit                named flat result    options: intercept, link, local_method, robust_se
//   glm_covariance         dims {p, p}          same
//   glm_residuals          n values             same
//   glm_predict            n_test values        same
//   glm_predict_intervals  dims {n_test, 3}     same plus alpha; names lower/mean/upper
#pragma once
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/functions/link_function_type.hpp"
#include "corehydro/numerics/machine_learning/supervised/decision_tree.hpp"
#include "corehydro/numerics/machine_learning/supervised/generalized_linear_model.hpp"
#include "corehydro/numerics/machine_learning/supervised/k_nearest_neighbors.hpp"
#include "corehydro/numerics/machine_learning/supervised/naive_bayes.hpp"
#include "corehydro/numerics/machine_learning/supervised/random_forest.hpp"
#include "corehydro/numerics/machine_learning/unsupervised/gaussian_mixture_model.hpp"
#include "corehydro/numerics/machine_learning/unsupervised/jenks_natural_breaks.hpp"
#include "corehydro/numerics/machine_learning/unsupervised/k_means.hpp"
#include "corehydro/numerics/math/optimization/support/local_method.hpp"
#include "corehydro/numerics/support/toolbox/common.hpp"

namespace corehydro::numerics::support::detail {

namespace mlns = corehydro::numerics::machine_learning;

// Builds the training predictor matrix from data[0] plus the `rows`/`columns` options.
inline math::linalg::Matrix build_ml_x(const std::vector<std::vector<double>>& data,
                                       const JsonValue& options, const std::string& method) {
    const std::vector<double>& flat = data_at(data, 0, "ml", method);
    int rows = options.value_or("rows", 0);
    int cols = options.value_or("columns", 1);
    if (rows <= 0) rows = cols > 0 ? static_cast<int>(flat.size()) / cols : 0;
    if (flat.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols))
        throw std::runtime_error("ml predictors hold " + std::to_string(flat.size()) +
                                 " values, expected rows * columns = " +
                                 std::to_string(rows * cols));
    return math::linalg::Matrix(rows, cols, flat);
}

// Builds the new-data matrix from data[2] plus `predict_rows` and the same `columns`.
inline math::linalg::Matrix build_ml_newdata(const std::vector<std::vector<double>>& data,
                                             const JsonValue& options,
                                             const std::string& method) {
    const std::vector<double>& flat = data_at(data, 2, "ml", method);
    int cols = options.value_or("columns", 1);
    int rows = options.value_or("predict_rows", 0);
    if (rows <= 0) rows = cols > 0 ? static_cast<int>(flat.size()) / cols : 0;
    if (flat.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols))
        throw std::runtime_error("ml new data holds " + std::to_string(flat.size()) +
                                 " values, expected predict_rows * columns = " +
                                 std::to_string(rows * cols));
    return math::linalg::Matrix(rows, cols, flat);
}

// Flattens a k-by-p vector-of-rows into a ToolboxResult with dims.
inline ToolboxResult ml_matrix_result(const std::vector<std::vector<double>>& m) {
    ToolboxResult r;
    int rows = static_cast<int>(m.size());
    int cols = rows > 0 ? static_cast<int>(m[0].size()) : 0;
    for (const std::vector<double>& row : m)
        for (double v : row) r.values.push_back(v);
    r.dims = {rows, cols};
    return r;
}

// Flattens a Matrix into a ToolboxResult with dims.
inline ToolboxResult ml_matrix_result(const math::linalg::Matrix& m) {
    ToolboxResult r;
    for (int i = 0; i < m.number_of_rows(); ++i)
        for (int j = 0; j < m.number_of_columns(); ++j) r.values.push_back(m(i, j));
    r.dims = {m.number_of_rows(), m.number_of_columns()};
    return r;
}

inline ToolboxResult ml_vector_result(const std::vector<double>& v) {
    ToolboxResult r;
    r.values = v;
    return r;
}

inline ToolboxResult ml_vector_result(const std::vector<int>& v) {
    ToolboxResult r;
    r.values.reserve(v.size());
    for (int x : v) r.values.push_back(static_cast<double>(x));
    return r;
}

// Resolves the `local_method` option to the ported LocalMethod enum.
//
// DELIBERATELY SEPARATE from optimizer_runner.hpp's `parse_local_method`, and it accepts MORE:
// that one is shared by `mlsl` and `multi_start`, which construct only BFGS / NelderMead /
// Powell (ADAM and GradientDescent throw "Unsupported local method" inside both, so rejecting
// them by name there is clearer than surfacing the inner message).
// `GeneralizedLinearModel.SetOptimizer` constructs all FIVE. The asymmetry is upstream's, so the
// two option surfaces genuinely differ and one shared parser would have to be wrong for one of
// them.
inline math::optimization::LocalMethod parse_glm_local_method(const std::string& s) {
    namespace opt = math::optimization;
    if (s == "nelder_mead") return opt::LocalMethod::NelderMead;
    if (s == "bfgs") return opt::LocalMethod::BFGS;
    if (s == "powell") return opt::LocalMethod::Powell;
    if (s == "adam") return opt::LocalMethod::ADAM;
    if (s == "gradient_descent") return opt::LocalMethod::GradientDescent;
    throw std::runtime_error("unknown local method '" + s +
                             "'; expected nelder_mead, bfgs, powell, adam, or gradient_descent");
}

// Resolves the `link` option to the ported LinkFunctionType enum. Only the five families
// GeneralizedLinearModel implements a log-likelihood for are accepted; the other two
// LinkFunctionType members (yeo_johnson, fisher_z) have no GLM family and would throw from
// inside the objective on the first evaluation, so they are rejected here by name instead.
inline functions::LinkFunctionType parse_glm_link(const std::string& s) {
    using functions::LinkFunctionType;
    if (s == "identity") return LinkFunctionType::Identity;
    if (s == "log") return LinkFunctionType::Log;
    if (s == "logit") return LinkFunctionType::Logit;
    if (s == "probit") return LinkFunctionType::Probit;
    if (s == "complementary_log_log") return LinkFunctionType::ComplementaryLogLog;
    throw std::runtime_error(
        "unknown glm link '" + s +
        "'; expected identity, log, logit, probit, or complementary_log_log");
}

inline ToolboxResult run_ml(const std::string& method,
                            const std::vector<std::vector<double>>& data,
                            const JsonValue& options) {
    // --- Unsupervised: Jenks natural breaks (a one-dimensional classifier) ---------------
    if (method == "jenks_breaks" || method == "jenks_gvf" || method == "jenks_clusters") {
        const std::vector<double>& x = data_at(data, 0, "ml", method);
        int n_clusters = options.value_or("n_clusters", 2);
        bool is_sorted = options.value_or("is_data_sorted", false);
        mlns::JenksNaturalBreaks jenks(x, n_clusters, is_sorted);
        if (method == "jenks_breaks") return ml_vector_result(jenks.breaks());
        if (method == "jenks_gvf") return scalar(jenks.goodness_of_variance_fit());
        ToolboxResult r;
        for (const mlns::JenksCluster& c : jenks.clusters()) {
            r.values.push_back(static_cast<double>(c.start_index()));
            r.values.push_back(static_cast<double>(c.end_index()));
            r.values.push_back(static_cast<double>(c.count()));
            r.values.push_back(c.min_value());
            r.values.push_back(c.max_value());
            r.values.push_back(c.sum());
            r.values.push_back(c.average());
            r.values.push_back(c.variance());
        }
        r.names = {"start_index", "end_index", "count", "min",
                   "max",         "sum",       "average", "variance"};
        r.dims = {static_cast<int>(jenks.clusters().size()), 8};
        return r;
    }

    // --- Unsupervised: k-means ------------------------------------------------------------
    if (method == "kmeans_means" || method == "kmeans_labels" || method == "kmeans_iterations") {
        mlns::KMeans k_means(build_ml_x(data, options, method), options.value_or("k", 2));
        k_means.set_max_iterations(options.value_or("max_iterations", 1000));
        k_means.train(options.value_or("seed", -1), options.value_or("kmeans_plus_plus", true));
        if (method == "kmeans_means") return ml_matrix_result(k_means.means());
        if (method == "kmeans_labels") return ml_vector_result(k_means.labels());
        return scalar(static_cast<double>(k_means.iterations()));
    }

    // --- Unsupervised: Gaussian mixture model ---------------------------------------------
    if (method == "gmm_means" || method == "gmm_weights" || method == "gmm_labels" ||
        method == "gmm_sigmas" || method == "gmm_log_likelihood" || method == "gmm_iterations") {
        mlns::GaussianMixtureModel gmm(build_ml_x(data, options, method),
                                       options.value_or("k", 2));
        gmm.set_max_iterations(options.value_or("max_iterations", 1000));
        gmm.set_tolerance(options.value_or("tolerance", 1e-8));
        gmm.train(options.value_or("seed", -1), options.value_or("kmeans_plus_plus", true));
        if (method == "gmm_means") return ml_matrix_result(gmm.means());
        if (method == "gmm_weights") return ml_vector_result(gmm.weights());
        if (method == "gmm_labels") return ml_vector_result(gmm.labels());
        if (method == "gmm_log_likelihood") return scalar(gmm.log_likelihood());
        if (method == "gmm_iterations") return scalar(static_cast<double>(gmm.iterations()));
        // gmm_sigmas: the k covariance matrices stacked vertically.
        ToolboxResult r;
        int p = gmm.dimension();
        for (const math::linalg::Matrix& s : gmm.sigmas())
            for (int i = 0; i < p; ++i)
                for (int j = 0; j < p; ++j) r.values.push_back(s(i, j));
        r.dims = {gmm.k() * p, p};
        return r;
    }

    // --- Supervised: they all take a response in data[1] -----------------------------------
    //
    // The method name is validated BEFORE the training data is read, so an unknown method reports
    // itself rather than surfacing "ml method 'bogus' needs 2 data vector(s)" from `data_at` --
    // the same single-dispatch discipline models/data_frame_runner.hpp adopted after the P4
    // whole-branch review (finding C7).
    static const std::vector<std::string> kSupervised = {
        "decision_tree_predict",  "random_forest_predict", "knn_predict",
        "knn_neighbors",          "knn_bootstrap_predict", "knn_prediction_intervals",
        "naive_bayes_means",      "naive_bayes_sds",       "naive_bayes_priors",
        "naive_bayes_classes",    "naive_bayes_predict",   "glm_fit",
        "glm_covariance",         "glm_residuals",         "glm_predict",
        "glm_predict_intervals"};
    if (std::find(kSupervised.begin(), kSupervised.end(), method) == kSupervised.end())
        throw std::runtime_error("unknown ml method: " + method);

    math::linalg::Matrix x = build_ml_x(data, options, method);
    math::linalg::Vector y(data_at(data, 1, "ml", method));

    if (method == "decision_tree_predict") {
        mlns::DecisionTree tree(x, y, options.value_or("seed", -1));
        tree.set_is_regression(options.value_or("is_regression", true));
        if (options.contains("features")) tree.set_features(options.at("features").as_int());
        tree.set_minimum_split_size(options.value_or("minimum_split_size", 2));
        tree.set_max_depth(options.value_or("max_depth", 100));
        tree.train();
        math::linalg::Matrix nd = build_ml_newdata(data, options, method);
        std::optional<std::vector<double>> p = tree.predict(nd);
        if (!p.has_value())
            throw std::runtime_error("ml decision_tree_predict: the new data has " +
                                     std::to_string(nd.number_of_columns()) +
                                     " columns, expected " + std::to_string(tree.dimensions()));
        return ml_vector_result(*p);
    }

    if (method == "random_forest_predict") {
        mlns::RandomForest rf(x, y, options.value_or("seed", -1));
        rf.set_is_regression(options.value_or("is_regression", true));
        if (options.contains("features")) rf.set_features(options.at("features").as_int());
        rf.set_minimum_split_size(options.value_or("minimum_split_size", 2));
        rf.set_max_depth(options.value_or("max_depth", 100));
        rf.set_number_of_trees(options.value_or("number_of_trees", 1000));
        rf.train();
        std::optional<math::linalg::Matrix> p =
            rf.predict(build_ml_newdata(data, options, method), options.value_or("alpha", 0.1));
        if (!p.has_value())
            throw std::runtime_error(
                "ml random_forest_predict: the new data column count does not match the training "
                "matrix");
        ToolboxResult r = ml_matrix_result(*p);
        r.names = {"lower", "median", "upper", "mean"};
        return r;
    }

    if (method == "knn_predict" || method == "knn_neighbors" ||
        method == "knn_bootstrap_predict" || method == "knn_prediction_intervals") {
        mlns::KNearestNeighbors knn(x, y, options.value_or("k", 1));
        knn.set_is_regression(options.value_or("is_regression", true));
        math::linalg::Matrix nd = build_ml_newdata(data, options, method);
        if (method == "knn_neighbors") {
            std::optional<std::vector<int>> n = knn.get_neighbors(nd);
            if (!n.has_value()) throw std::runtime_error("ml knn_neighbors: invalid new data");
            ToolboxResult r = ml_vector_result(*n);
            r.dims = {nd.number_of_rows(), knn.k()};
            return r;
        }
        if (method == "knn_prediction_intervals") {
            math::linalg::Matrix pi =
                knn.prediction_intervals(nd, options.value_or("seed", -1),
                                          options.value_or("realizations", 1000),
                                          options.value_or("alpha", 0.1));
            ToolboxResult r = ml_matrix_result(pi);
            r.names = {"lower", "median", "upper", "mean"};
            return r;
        }
        std::optional<std::vector<double>> p =
            method == "knn_predict" ? knn.predict(nd)
                                     : knn.bootstrap_predict(nd, options.value_or("seed", -1));
        if (!p.has_value())
            throw std::runtime_error("ml " + method + ": the new data has " +
                                     std::to_string(nd.number_of_columns()) +
                                     " columns, expected " +
                                     std::to_string(knn.number_of_features()));
        return ml_vector_result(*p);
    }

    if (method == "naive_bayes_means" || method == "naive_bayes_sds" ||
        method == "naive_bayes_priors" || method == "naive_bayes_classes" ||
        method == "naive_bayes_predict") {
        mlns::NaiveBayes nb(x, y);
        if (method == "naive_bayes_classes") return ml_vector_result(nb.classes());
        nb.train();
        if (method == "naive_bayes_means") return ml_matrix_result(nb.means());
        if (method == "naive_bayes_sds") return ml_matrix_result(nb.standard_deviations());
        if (method == "naive_bayes_priors") return ml_vector_result(nb.priors());
        std::optional<std::vector<double>> p = nb.predict(build_ml_newdata(data, options, method));
        if (!p.has_value())
            throw std::runtime_error(
                "ml naive_bayes_predict: the new data column count does not match the training "
                "matrix");
        return ml_vector_result(*p);
    }

    if (method == "glm_fit" || method == "glm_covariance" || method == "glm_residuals" ||
        method == "glm_predict" || method == "glm_predict_intervals") {
        mlns::GeneralizedLinearModel model(
            x, y, options.value_or("intercept", true),
            parse_glm_link(options.value_or("link", "identity")));
        model.set_use_robust_se(options.value_or("robust_se", false));
        if (options.contains("local_method"))
            model.set_optimizer(parse_glm_local_method(options.at("local_method").as_string()));
        model.train();

        if (method == "glm_covariance") return ml_matrix_result(model.covariance());
        if (method == "glm_residuals") return ml_vector_result(model.residuals());
        if (method == "glm_predict")
            return ml_vector_result(model.predict(build_ml_newdata(data, options, method)));
        if (method == "glm_predict_intervals") {
            ToolboxResult r = ml_matrix_result(model.predict_intervals(
                build_ml_newdata(data, options, method), options.value_or("alpha", 0.1)));
            r.names = {"lower", "mean", "upper"};
            return r;
        }
        // glm_fit: the same flat named shape the `regression` group's `fit` returns.
        ToolboxResult r;
        const std::vector<double>& beta = model.parameters();
        for (std::size_t i = 0; i < beta.size(); ++i) {
            r.values.push_back(beta[i]);
            r.names.push_back("beta_" + std::to_string(i + 1));
        }
        for (std::size_t i = 0; i < beta.size(); ++i) {
            r.values.push_back(model.parameter_standard_errors()[i]);
            r.names.push_back("se_" + std::to_string(i + 1));
        }
        for (std::size_t i = 0; i < beta.size(); ++i) {
            r.values.push_back(model.parameter_z_scores()[i]);
            r.names.push_back("z_" + std::to_string(i + 1));
        }
        for (std::size_t i = 0; i < beta.size(); ++i) {
            r.values.push_back(model.parameter_p_values()[i]);
            r.names.push_back("p_" + std::to_string(i + 1));
        }
        r.values.push_back(model.standard_error());                          r.names.push_back("sigma");
        r.values.push_back(static_cast<double>(model.degrees_of_freedom())); r.names.push_back("df");
        r.values.push_back(static_cast<double>(model.sample_size()));        r.names.push_back("n");
        r.values.push_back(model.aic());                                     r.names.push_back("aic");
        r.values.push_back(model.aicc());                                    r.names.push_back("aicc");
        r.values.push_back(model.bic());                                     r.names.push_back("bic");
        return r;
    }

    throw std::runtime_error("unknown ml method: " + method);
}

}  // namespace corehydro::numerics::support::detail
