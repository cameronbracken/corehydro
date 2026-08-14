// corehydro ADDITION -- no upstream C# counterpart. The math group of callback_runner.hpp:
// the ported Numerics routines whose input is a user function rather than serializable data.
// Root finding is Brent (numerics/math/rootfinding/brent.hpp); differentiation is the ported
// NumericalDerivative (numerics/math/differentiation/numerical_derivative.hpp). Quadrature is
// added in Task 2, once AdaptiveGaussKronrod is a complete port.
//
// Options grammar (JSON object, see run_math below for which keys each method requires):
//
//   root_find:  {"lower": 0.0, "upper": 2.0, "tolerance": 1e-8, "max_iterations": 1000}
//   derivative: {"point": 2.0, "step_size": -1}
//   gradient:   {"point": [1.0, 1.0]}
//   hessian:    {"point": [1.0, 2.0]}
//
// `tolerance`/`max_iterations`/`step_size` are optional and default to the ported routine's own
// defaults (Brent::solve's 1e-8 / 1000; central_difference's adaptive step, selected by any
// step_size <= 0).
//
// EVERY drive site here wraps the ported call in try/catch and prefers the guard's stored host
// exception (see callback_guard.hpp's "USE IT IN PAIRS" note). This is not defensive
// boilerplate: the sentinel a guard returns after an abort is itself a value the ported routine
// may reject, and it does. A NaN from an aborted root_find drives Brent to its 1000-iteration
// "failed to find root" throw; a -inf from an aborted gradient/hessian trips
// NumericalDerivative's "f(theta) is not finite" domain_error on the very first evaluation. Both
// of those would otherwise reach the caller INSTEAD of the user's own R/Python error -- the exact
// bug the optimizer surface shipped with for "bfgs"/"mlsl" before it was closed.
#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"
#include "corehydro/numerics/support/callback/common.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"

namespace corehydro::numerics::support::detail {

namespace rootfinding = corehydro::numerics::math::rootfinding;
namespace differentiation = corehydro::numerics::math::differentiation;

inline CallbackResult run_math(const std::string& method, const JsonValue& o,
                               const CallbackSet& cbs) {
    CallbackResult r;
    r.status = "Success";

    if (method == "root_find") {
        if (!cbs.scalar) throw std::invalid_argument("math/root_find requires a scalar function");
        GuardedCall<double, double> g(cbs.scalar, std::numeric_limits<double>::quiet_NaN());
        double lower = require_double(o, "lower", "math/root_find");
        double upper = require_double(o, "upper", "math/root_find");
        double tolerance = o.value_or("tolerance", 1e-8);
        int max_iterations = o.value_or("max_iterations", 1000);
        double root = 0.0;
        try {
            root = rootfinding::solve([&g](double x) { return g(x); }, lower, upper, tolerance,
                                      max_iterations);
        } catch (...) {
            g.rethrow_if_aborted();
            throw;
        }
        g.rethrow_if_aborted();
        r.values = {root};
        r.names = {"root"};
        return r;
    }

    if (method == "derivative") {
        if (!cbs.scalar) throw std::invalid_argument("math/derivative requires a scalar function");
        GuardedCall<double, double> g(cbs.scalar, std::numeric_limits<double>::quiet_NaN());
        double point = require_double(o, "point", "math/derivative");
        double step_size = o.value_or("step_size", -1.0);
        double d = 0.0;
        try {
            d = differentiation::derivative([&g](double x) { return g(x); }, point, step_size);
        } catch (...) {
            g.rethrow_if_aborted();
            throw;
        }
        g.rethrow_if_aborted();
        r.values = {d};
        r.names = {"derivative"};
        return r;
    }

    if (method == "gradient" || method == "hessian") {
        if (!cbs.vector_scalar)
            throw std::invalid_argument("math/" + method + " requires a vector function");
        GuardedCall<double, const std::vector<double>&> g(
            cbs.vector_scalar, -std::numeric_limits<double>::infinity());
        std::string where = "math/" + method;
        std::vector<double> point = require_vector(o, "point", where.c_str());
        auto f = [&g](const std::vector<double>& p) { return g(p); };
        if (method == "gradient") {
            try {
                r.values = differentiation::gradient(f, point);
            } catch (...) {
                g.rethrow_if_aborted();
                throw;
            }
            g.rethrow_if_aborted();
            r.dims = {static_cast<int>(r.values.size())};
        } else {
            std::vector<std::vector<double>> h;
            try {
                h = differentiation::hessian(f, point);
            } catch (...) {
                g.rethrow_if_aborted();
                throw;
            }
            g.rethrow_if_aborted();
            int n = static_cast<int>(h.size());
            r.values.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
            for (const auto& row : h) r.values.insert(r.values.end(), row.begin(), row.end());
            r.dims = {n, n};
        }
        return r;
    }

    throw std::invalid_argument("unknown math method: " + method);
}

}  // namespace corehydro::numerics::support::detail
