// corehydro ADDITION -- no upstream C# counterpart. Shared types for the callback groups,
// sibling of numerics/support/toolbox/common.hpp.
//
// Holds the CallbackSet (every callback signature a group may ask for), the flat CallbackResult
// every binding and every fixture assertion reads, and the option-reading helpers each group
// header uses. This header is self-contained and may be included before any group header.
#pragma once
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::support {

using corehydro::models::spec::JsonValue;

// What a GMM moment-condition callback returns: upstream's `(Vector G, Matrix S)` tuple
// (GeneralizedMethodOfMoments.cs's MomentConditionFunction delegate), carried across the binding
// boundary in the plainest shape each host language can build. `g` is the mean moment-condition
// vector, length q; `s` is their q x q covariance, FLATTENED ROW-MAJOR with its own shape in
// `s_rows`/`s_cols`.
//
// The shape travels with the value rather than being assumed from q for one reason: the wrong
// shape is the mistake this callback invites, and reporting "your s is 2 x 3" is a different
// quality of error from "your s has 6 entries". R hands back a COLUMN-major matrix and its glue
// transposes; Python's nested sequence is already row-major. Neither language's converter may
// guess -- see the note on each.
struct MomentConditionReturn {
    std::vector<double> g;
    std::vector<double> s;
    int s_rows = 0;
    int s_cols = 0;
};

// What a covariance-aware fit callback returns: upstream's `BootstrapFit` (the return type of
// `Bootstrap<TData>.FitWithCovarianceFunction`, the delegate the PIVOTAL bootstrap fits through),
// carried across the binding boundary in the plainest shape each host language can build.
// `parameters` is the fitted vector, length p; `covariance` is their p x p covariance, FLATTENED
// ROW-MAJOR with its own shape in `rows`/`cols`.
//
// The shape travels with the value for the reason MomentConditionReturn's does: the wrong shape is
// the mistake this callback invites, and "your covariance is 2 x 3" is a different quality of
// error from "your covariance has 6 entries". R hands back a COLUMN-major matrix and its glue
// transposes; Python's nested sequence is already row-major.
struct FitWithCovarianceReturn {
    std::vector<double> parameters;
    std::vector<double> covariance;
    int rows = 0;
    int cols = 0;
};

// Every callback a group may need. A caller fills only the members its group uses; the rest stay
// empty and each group validates the ones it requires. One struct rather than a variant keeps the
// four bindings' call sites uniform: build the set, name the group and method, call.
struct CallbackSet {
    // f(x) -> y. Root finding, quadrature, single-variable differentiation.
    std::function<double(double)> scalar;
    // f'(x) -> y. The analytic derivative Newton-Raphson root finding takes alongside `scalar`
    // (its own f). Named distinctly from `scalar` rather than reused positionally, because
    // root_find_newton needs both live at once -- see callback/math.hpp's root_find_newton arm,
    // which guards this and `scalar` with a SHARED abort state for the reason gmm.hpp's file
    // header gives for its own multi-callback groups.
    std::function<double(double)> scalar_deriv;
    // f(x, y) -> z. The 2D integrand math/quadrature_2d takes (AdaptiveSimpsonsRule2D's
    // function), P2 "math extras". Its own member rather than a reuse of `scalar`, because the
    // arity differs; see callback/math.hpp's quadrature_2d arm.
    std::function<double(double, double)> scalar_xy;
    // f(theta) -> y. Log-likelihood, gradient/hessian target, GMM penalty.
    std::function<double(const std::vector<double>&)> vector_scalar;
    // f(theta) -> vector. Gradient callback, GMM moment conditions, bootstrap statistic.
    std::function<std::vector<double>(const std::vector<double>&)> vector_vector;
    // f(theta, rng) -> vector. Gibbs proposal.
    std::function<std::vector<double>(const std::vector<double>&,
                                      corehydro::numerics::sampling::MersenneTwister&)>
        vector_rng;
    // f(data, theta, rng) -> data. Bootstrap resample.
    std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&,
                                      corehydro::numerics::sampling::MersenneTwister&)>
        data_rng;
    // f(data) -> theta. Bootstrap fit.
    std::function<std::vector<double>(const std::vector<double>&)> data_vector;
    // f(data) -> (theta, covariance). Bootstrap covariance-aware fit, upstream's
    // FitWithCovarianceFunction -- the delegate the pivotal run type fits through.
    std::function<FitWithCovarianceReturn(const std::vector<double>&)> data_covariance;
    // f(data, index) -> data. Bootstrap jackknife.
    std::function<std::vector<double>(const std::vector<double>&, int)> data_index;
    // f(theta) -> matrix, row-major with dims. GMM jacobian, pointwise moment conditions.
    std::function<std::pair<std::vector<double>, std::vector<int>>(const std::vector<double>&)>
        vector_matrix;
    // f(theta) -> (g, s). GMM moment conditions, upstream's MomentConditionFunction.
    std::function<MomentConditionReturn(const std::vector<double>&)> moment_conditions;
};

// Flat result surface every binding and every fixture assertion reads. `dims` takes one of three
// shapes: EMPTY when the group has no shape to report (a scalar result, or a plain vector whose
// length is `values.size()` -- math/root_find and math/derivative); {n} for a vector the group
// wants shaped explicitly (math/gradient); {rows, cols} for a matrix, with `values` flattened
// row-major (math/hessian). `names` labels `values` where a group has something to say (parameter
// names, statistic names); otherwise empty.
struct CallbackResult {
    std::vector<double> values;
    std::vector<std::string> names;
    std::vector<int> dims;
    std::string status;
};

namespace detail {

inline JsonValue parse_options(const std::string& options_json) {
    if (options_json.empty()) return JsonValue{};
    return corehydro::models::spec::parse_json(options_json);
}

inline double require_double(const JsonValue& o, const char* key, const char* group) {
    if (!o.contains(key))
        throw std::invalid_argument(std::string(group) + " requires the option '" + key + "'");
    return o.at(key).as_double();
}

inline std::vector<double> require_vector(const JsonValue& o, const char* key, const char* group) {
    if (!o.contains(key))
        throw std::invalid_argument(std::string(group) + " requires the option '" + key + "'");
    return o.at(key).as_double_vector();
}

}  // namespace detail
}  // namespace corehydro::numerics::support
