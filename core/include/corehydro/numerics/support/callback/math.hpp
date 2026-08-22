// corehydro ADDITION -- no upstream C# counterpart. The math group of callback_runner.hpp:
// the ported Numerics routines whose input is a user function rather than serializable data.
// Root finding is Brent (numerics/math/rootfinding/brent.hpp) plus, since P2 "math extras",
// Bisection/Secant/NewtonRaphson (numerics/math/rootfinding/{bisection,secant,newton_raphson}.hpp);
// differentiation is the ported NumericalDerivative
// (numerics/math/differentiation/numerical_derivative.hpp); quadrature is the ported
// AdaptiveGaussKronrod (numerics/math/integration/adaptive_gauss_kronrod.hpp).
//
// Options grammar (JSON object, see run_math below for which keys each method requires):
//
//   root_find:  {"method": "brent", "lower": 0.0, "upper": 2.0, "tolerance": 1e-8,
//                "max_iterations": 1000}
//   root_find_newton:  {"first_guess": 1.0, "lower": 0.0, "upper": 2.0, "tolerance": 1e-8,
//                       "max_iterations": 1000}
//   root_find_system:  {"first_guess": [0.0, 0.0], "tolerance": 1e-8, "max_iterations": 1000}
//   derivative: {"point": 2.0, "step_size": -1}
//   gradient:   {"point": [1.0, 1.0]}
//   hessian:    {"point": [1.0, 2.0]}
//   quadrature: {"lower": 0.0, "upper": 3.0, "absolute_tolerance": 1e-8,
//                "relative_tolerance": 1e-8, "max_function_evaluations": 10000000}
//   quadrature_2d: {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0,
//                   "absolute_tolerance": 1e-8, "relative_tolerance": 1e-8,
//                   "min_depth": 0, "max_depth": 100}
//   quadrature_nd: {"method": "monte_carlo", "min": [0.0, 0.0], "max": [1.0, 1.0], "seed": 12345,
//                   "use_sobol": true, "sobol_path": "...", "max_function_evaluations": 100000,
//                   "max_iterations": 2000, "min_iterations": 100, "relative_tolerance": 1e-3,
//                   "fraction": 0.1, "min_subregion_points": 15, "min_bisections": 60,
//                   "dither": 0.0}
//   quadrature_vegas: {"min": [0.0, 0.0], "max": [1.0, 1.0], "seed": 12345, "use_sobol": true,
//                      "sobol_path": "...", "independent_evaluations": 5, "function_calls": 2000,
//                      "alpha": 1.5, "number_of_bins": 50, "tail_focus_parameter": 1.0,
//                      "initialize": 0, "check_convergence": true, "target_probability": 1e-5}
//
// `quadrature`'s `method` (P2 "math extras") is OPTIONAL and one of "gauss_kronrod" (the
// default, preserving every fixture that predates this key), "simpsons", "trapezoidal",
// "adaptive_simpsons", "gauss_lobatto" (the five ported Integrator subclasses -- SimpsonsRule,
// TrapezoidalRule, AdaptiveSimpsonsRule, AdaptiveGaussLobatto, AdaptiveGaussKronrod itself), or
// "gauss_legendre", "gauss_legendre20", "simpsons_fixed", "trapezoidal_fixed", "midpoint" (the
// five ported Integration statics). The Integrator-class arms keep the existing result triple
// {integral, function_evaluations, standard_error} + a status read off the driven object's own
// IntegrationStatus, standard_error 0.0 where the class has none (SimpsonsRule, TrapezoidalRule,
// AdaptiveGaussLobatto); the static arms return values={integral}, names={"integral"}, status
// "Success" unconditionally, since none of them has an Integrator to report one. `min_depth`/
// `max_depth` are optional and apply to "adaptive_simpsons" alone; `steps` (default 2, per the
// ported statics' own default) is optional and applies to "simpsons_fixed"/"trapezoidal_fixed"/
// "midpoint" alone. Every arm shares the absolute_tolerance/relative_tolerance plumbing the
// gauss_kronrod arm already has, where the driven class has the member to set (the statics take
// none).
//
// `quadrature_2d` (P2 "math extras") is a SEPARATE method rather than a `quadrature` arm, because
// it needs a second callback shape (`cbs.scalar_xy`, f(x, y) rather than f(x)) -- the 2D
// counterpart of why root_find_newton is its own method rather than a root_find option. It always
// drives AdaptiveSimpsonsRule2D and always returns the result triple + status, exactly as
// quadrature's Integrator-class arms do.
//
// `quadrature_nd` (P2 "math extras") drives the two ported stochastic multidimensional
// integrators, MonteCarloIntegration (the default `method`) and Miser, over `cbs.vector_scalar`
// (the same shape `gradient`/`hessian` already take) -- `min`/`max` give the per-dimension bounds
// AND, via their length, the dimensionality, so no separate `dimensions` key exists. `seed`
// PRESENT seeds the class's `random` member the same way the C# oracle emitter constructs `new
// MersenneTwister(seed)`; ABSENT leaves the default-constructed (clock-seeded) generator in
// force, matching the ported class's own field initializer. `sobol_path` is a corehydro addition
// (see the file's SOBOL PATH note below) -- required whenever `min`/`max` carry more than one
// dimension, because Miser's ctor unconditionally constructs a SobolSequence over it regardless
// of `use_sobol`; MonteCarloIntegration never reads it (`use_sobol_sequence` is a documented dead
// property on that class alone -- see monte_carlo_integration.hpp's file header). Two of
// MonteCarloIntegration's own Integrator-base fields are exposed here even though they carry no
// C# test-oracle-grammar counterpart in RMC's own delegate options, because MonteCarloIntegration
// checks its OWN early-exit only against `relative_tolerance` inside a loop bounded by
// `max_iterations` (`max_function_evaluations` is checked only AFTER the loop, to choose the
// reported status, not to end it early -- a faithfully-ported C# characteristic, not a bug here):
// without a way to shrink `max_iterations` or loosen `relative_tolerance`, a `method =
// "monte_carlo"` run with the ported defaults (1e7 iterations, 1e-8 relative tolerance) would
// call the host-language callback up to ten million times before returning, which is far too slow
// for an R or Python closure. `min_iterations`/`max_iterations`/`relative_tolerance` are therefore
// accepted for `method = "monte_carlo"` alongside `max_function_evaluations` (which the C# oracle
// emitter DOES honor by construction, setting the same Integrator field). Miser's own
// `max_function_evaluations` IS the real throttle for that method (it is `npts`, the sample-point
// budget its private `miser()` recursion is handed directly), plus its own four members
// (`fraction`, `min_subregion_points` -> `minimum_number_of_subregion_points`, `min_bisections` ->
// `minimum_number_of_bisections`, `dither`). Result is always the triple
// {integral, function_evaluations, standard_error} + the driven class's own IntegrationStatus,
// exactly as `quadrature`'s Integrator-class arms report.
//
// `quadrature_vegas` (P2 "math extras") drives the ported Vegas class over `cbs.vector_weight`
// (f(x, weight) -> y, upstream's own Vegas integrand shape) -- its OWN method rather than a
// `quadrature_nd` arm, because Vegas needs the weight argument no other integrator here does. The
// same `min`/`max`/`seed`/`use_sobol`/`sobol_path` rules as `quadrature_nd` apply, plus Vegas's
// own members: `independent_evaluations`, `function_calls` (the two knobs whose PRODUCT bounds
// total evaluations per run -- Vegas, unlike MonteCarloIntegration, checks `max_function_evaluations`
// as a real per-call stopping condition, so no extra throttle is needed here), `alpha`,
// `number_of_bins` (routed through `set_number_of_bins()`, which re-initializes the class's
// parameter arrays exactly as the C# property setter does), `tail_focus_parameter`, `initialize`,
// `check_convergence`. `target_probability`, PRESENT, calls `configure_for_rare_events()` --
// applied LAST, after every other option, so it can override `number_of_bins`/`alpha` the same way
// the ported method itself does when a caller supplies both. Result is
// {integral, function_evaluations, standard_error, chi_squared} + status -- the one arm on this
// surface reporting a chi-squared, since Vegas is the one integrator here that has one.
//
// SOBOL PATH, a corehydro addition with no C# counterpart (see miser.hpp's and vegas.hpp's own
// file headers): C#'s `SobolSequence` reads its direction numbers from a compiled resource: this
// port's instead takes a filesystem path, resolved by the WRAPPER exactly as
// numerics/support/toolbox/sampling.hpp's own `sobol` arm documents (R: `system.file()` against
// `inst/extdata`; Python: `importlib.resources` against corehydropy's own `data/` directory; the
// C++ fixture runner and the dotnet oracle emitter resolve their own copies the same way
// test_fixtures.cpp's `g_sobol_path` already does for the `sampling` toolbox group). This header
// only ever reads `sobol_path` as a plain string option and hands it straight to the Miser/Vegas
// constructor's own trailing `sobol_path` argument.
//
// `root_find`'s `method` is OPTIONAL and one of "brent" (the default, preserving every fixture
// that predates this key), "bisection", or "secant". `lower`/`upper` are required by all three;
// "bisection" additionally requires `first_guess` (the running root Bisection.Solve seeds itself
// with -- the bracket only seeds the initial step direction, exactly as bisection.hpp's own
// header note explains). "secant" does not take `first_guess`: Secant.Solve picks its own
// starting guess off the bracket, the bound with the smaller |f|.
//
// `root_find_newton` is a SEPARATE method, not a fourth `root_find` arm, because it is the one
// root finder needing a second callback (`cbs.scalar_deriv`, the analytic derivative) rather than
// a second option. `first_guess` is required; `lower` and `upper` are optional, and it is their
// PRESENCE -- both together -- that selects `newton_raphson_robust_solve` over the plain
// `newton_raphson_solve`, not a method sub-key, matching the ported class's own two static
// methods (`NewtonRaphson.Solve` / `NewtonRaphson.RobustSolve`) rather than inventing a knob C#
// does not have.
//
// `root_find_system` solves a system of equations: `cbs.vector_vector` (F, theta -> vector) and
// `cbs.vector_matrix` (J, theta -> matrix, ROW-major with its own shape, the same shape gmm's
// jacobian callback already carries) are reused rather than adding new CallbackSet members, since
// both shapes already exist for the mcmc/gmm groups. `first_guess` gives the system's dimension
// n; F must return a vector of length n and J an n x n matrix, both checked by name rather than
// left to a linear-algebra exception with no context. The result carries `dims = {n}` rather than
// leaving it empty, matching gradient's own vector-shaped result.
//
// `tolerance`/`max_iterations`/`step_size` are optional and default to the ported routine's own
// defaults (Brent::solve's 1e-8 / 1000, shared by every root finder here via
// rootfinding::kDefaultTolerance/kDefaultMaxIterations; central_difference's adaptive step,
// selected by any step_size <= 0). Quadrature's three optional keys likewise leave the ported
// Integrator base's own defaults untouched when absent, and an ABSENT key is not the same as a key
// carrying the default: the key is only written when present, so a later change to a C# default
// lands here.
//
// `quadrature` is the only method returning more than the answer: `values` is
// {integral, function_evaluations, standard_error} and `status` is the ported IntegrationStatus
// rather than the unconditional "Success" the other three report, because AdaptiveGaussKronrod is
// the only one of the four whose C# class has an Integrator base to report one. The standard error
// is the C# StandardError property (the square root of the accumulated squared Gauss-Kronrod
// differences); it is returned rather than dropped because it is the run's own error estimate and
// a caller integrating an unfamiliar function wants it beside the status.
//
// EVERY drive site here wraps the ported call in try/catch and prefers the guard's stored host
// exception (see callback_guard.hpp's "USE IT IN PAIRS" note). This is not defensive
// boilerplate: the sentinel a guard returns after an abort is itself a value the ported routine
// may reject, and it does. A NaN from an aborted root_find drives Brent to its 1000-iteration
// "failed to find root" throw; a -inf from an aborted gradient/hessian trips
// NumericalDerivative's "f(theta) is not finite" domain_error on the very first evaluation. Both
// of those would otherwise reach the caller INSTEAD of the user's own R/Python error -- the exact
// bug the optimizer surface shipped with for "bfgs"/"mlsl" before it was closed.
//
// Quadrature is the case where the TRAILING half carries the weight rather than the try/catch:
// AdaptiveGaussKronrod never rejects a NaN, so an aborted run neither throws nor converges. It
// subdivides until MaxFunctionEvaluations stops it, returns a NaN Result and a
// MaximumFunctionEvaluationsReached status, and the caller would read that as an answer if the
// trailing rethrow were missing. Both halves are kept on every arm precisely because which one
// fires is a property of the ported routine, not of this file.
#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/integration/adaptive_gauss_kronrod.hpp"
#include "corehydro/numerics/math/integration/adaptive_gauss_lobatto.hpp"
#include "corehydro/numerics/math/integration/adaptive_simpsons_rule.hpp"
#include "corehydro/numerics/math/integration/adaptive_simpsons_rule_2d.hpp"
#include "corehydro/numerics/math/integration/integration.hpp"
#include "corehydro/numerics/math/integration/miser.hpp"
#include "corehydro/numerics/math/integration/monte_carlo_integration.hpp"
#include "corehydro/numerics/math/integration/simpsons_rule.hpp"
#include "corehydro/numerics/math/integration/trapezoidal_rule.hpp"
#include "corehydro/numerics/math/integration/vegas.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/math/rootfinding/bisection.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"
#include "corehydro/numerics/math/rootfinding/newton_raphson.hpp"
#include "corehydro/numerics/math/rootfinding/secant.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/support/callback/common.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"

namespace corehydro::numerics::support::detail {

namespace rootfinding = corehydro::numerics::math::rootfinding;
namespace differentiation = corehydro::numerics::math::differentiation;
namespace integration = corehydro::numerics::math::integration;
namespace linalg = corehydro::numerics::math::linalg;
namespace bfsampling = corehydro::numerics::sampling;

inline CallbackResult run_math(const std::string& method, const JsonValue& o,
                               const CallbackSet& cbs) {
    CallbackResult r;
    r.status = "Success";

    if (method == "root_find") {
        if (!cbs.scalar) throw std::invalid_argument("math/root_find requires a scalar function");
        GuardedCall<double, double> g(cbs.scalar, std::numeric_limits<double>::quiet_NaN());
        auto fx = [&g](double x) { return g(x); };
        // Absent means "brent", preserving every fixture that predates this key. See the file
        // header on why the other two arms are options on THIS method rather than the separate
        // root_find_newton is.
        std::string root_method = o.value_or("method", "brent");
        double lower = require_double(o, "lower", "math/root_find");
        double upper = require_double(o, "upper", "math/root_find");
        // An ABSENT key leaves the PORTED default in force, and this file does not carry a copy
        // of that default -- the same rule the quadrature arm below follows with o.contains(),
        // and the one both packages' root_find() help pages state ("the value is not restated
        // here, so a change to it lands in one place"). The two constants live in brent.hpp,
        // beside the signature they default, because `max_iterations` cannot be passed
        // positionally without also naming `tolerance`.
        double tolerance = o.contains("tolerance") ? o.at("tolerance").as_double()
                                                   : rootfinding::kDefaultTolerance;
        int max_iterations = o.contains("max_iterations") ? o.at("max_iterations").as_int()
                                                          : rootfinding::kDefaultMaxIterations;
        double root = 0.0;
        try {
            if (root_method == "brent") {
                root = rootfinding::solve(fx, lower, upper, tolerance, max_iterations);
            } else if (root_method == "bisection") {
                double first_guess = require_double(o, "first_guess", "math/root_find (bisection)");
                root = rootfinding::bisection_solve(fx, first_guess, lower, upper, tolerance,
                                                    max_iterations);
            } else if (root_method == "secant") {
                root = rootfinding::secant_solve(fx, lower, upper, tolerance, max_iterations);
            } else {
                throw std::invalid_argument("math/root_find: unknown 'method' '" + root_method +
                                            "'; expected 'brent', 'bisection', or 'secant'");
            }
        } catch (...) {
            g.rethrow_if_aborted();
            throw;
        }
        g.rethrow_if_aborted();
        r.values = {root};
        r.names = {"root"};
        return r;
    }

    if (method == "root_find_newton") {
        if (!cbs.scalar)
            throw std::invalid_argument("math/root_find_newton requires a scalar function 'f'");
        if (!cbs.scalar_deriv)
            throw std::invalid_argument(
                "math/root_find_newton requires a scalar derivative function 'df'");
        // ONE abort state for both guards -- see gmm.hpp's file header on why a group with more
        // than one live callback must share rather than let each guard protect only itself: an
        // aborted f() must stop df() from re-entering the host too.
        CallbackAbortStatePtr abort = make_abort_state();
        GuardedCall<double, double> gf(cbs.scalar, std::numeric_limits<double>::quiet_NaN(), abort);
        GuardedCall<double, double> gdf(cbs.scalar_deriv, std::numeric_limits<double>::quiet_NaN(),
                                        abort);
        auto fx = [&gf](double x) { return gf(x); };
        auto dfx = [&gdf](double x) { return gdf(x); };
        double first_guess = require_double(o, "first_guess", "math/root_find_newton");
        double tolerance = o.contains("tolerance") ? o.at("tolerance").as_double()
                                                   : rootfinding::kDefaultTolerance;
        int max_iterations = o.contains("max_iterations") ? o.at("max_iterations").as_int()
                                                          : rootfinding::kDefaultMaxIterations;
        // Both present -- not a method sub-key -- selects the robust (bracketed) variant, matching
        // the ported class's own two static methods. See the file header.
        const bool has_bracket = o.contains("lower") && o.contains("upper");
        double root = 0.0;
        try {
            if (has_bracket) {
                double lower = o.at("lower").as_double();
                double upper = o.at("upper").as_double();
                root = rootfinding::newton_raphson_robust_solve(fx, dfx, first_guess, lower, upper,
                                                                 tolerance, max_iterations);
            } else {
                root = rootfinding::newton_raphson_solve(fx, dfx, first_guess, tolerance,
                                                          max_iterations);
            }
        } catch (...) {
            rethrow_if_aborted(abort);
            throw;
        }
        rethrow_if_aborted(abort);
        r.values = {root};
        r.names = {"root"};
        return r;
    }

    if (method == "root_find_system") {
        if (!cbs.vector_vector)
            throw std::invalid_argument(
                "math/root_find_system requires a system function 'F' (theta -> vector)");
        if (!cbs.vector_matrix)
            throw std::invalid_argument(
                "math/root_find_system requires a jacobian function 'J' (theta -> matrix)");
        std::vector<double> first_guess = require_vector(o, "first_guess", "math/root_find_system");
        if (first_guess.empty())
            throw std::invalid_argument(
                "math/root_find_system requires 'first_guess' to hold at least one value");
        const std::size_t n = first_guess.size();
        double tolerance = o.contains("tolerance") ? o.at("tolerance").as_double()
                                                   : rootfinding::kDefaultTolerance;
        int max_iterations = o.contains("max_iterations") ? o.at("max_iterations").as_int()
                                                          : rootfinding::kDefaultMaxIterations;

        // ONE abort state for both guards, for the reason root_find_newton's does.
        CallbackAbortStatePtr abort = make_abort_state();
        using MatrixReturn = std::pair<std::vector<double>, std::vector<int>>;

        // THE SHAPE IS CHECKED INSIDE THE GUARDED FUNCTION, exactly as gmm.hpp's jacobian guard
        // does -- so a wrong-shaped return is treated identically to a host-language exception: it
        // latches the shared abort state and short-circuits the other guard, rather than escaping
        // unguarded and racing the ported routine's own linear algebra to report first.
        GuardedCall<std::vector<double>, const std::vector<double>&> gF(
            [fn = cbs.vector_vector, n](const std::vector<double>& x) {
                std::vector<double> out = fn(x);
                if (out.size() != n)
                    throw std::invalid_argument(
                        "the system function 'F' must return a vector of length " +
                        std::to_string(n) + " (matching 'first_guess'); it returned " +
                        std::to_string(out.size()));
                return out;
            },
            std::vector<double>(n, 0.0), abort);
        GuardedCall<MatrixReturn, const std::vector<double>&> gJ(
            [fn = cbs.vector_matrix, n](const std::vector<double>& x) {
                MatrixReturn out = fn(x);
                const bool shaped = out.second.size() == 2 &&
                                    out.second[0] == static_cast<int>(n) &&
                                    out.second[1] == static_cast<int>(n);
                if (!shaped || out.first.size() != n * n)
                    throw std::invalid_argument(
                        "the jacobian function 'J' must return a " + std::to_string(n) + " x " +
                        std::to_string(n) + " matrix; it returned " +
                        (out.second.size() == 2
                             ? std::to_string(out.second[0]) + " x " + std::to_string(out.second[1])
                             : std::to_string(out.first.size()) + " values with no shape"));
                return out;
            },
            MatrixReturn{std::vector<double>(n * n, 0.0),
                        std::vector<int>{static_cast<int>(n), static_cast<int>(n)}},
            abort);

        auto Ffn = [&gF](const linalg::Vector& x) { return linalg::Vector(gF(x.to_array())); };
        auto Jfn = [&gJ, n](const linalg::Vector& x) {
            MatrixReturn out = gJ(x.to_array());
            return linalg::Matrix(static_cast<int>(n), static_cast<int>(n), out.first);
        };

        linalg::Vector root(static_cast<int>(n));
        try {
            root = rootfinding::newton_raphson_solve_system(Ffn, Jfn, linalg::Vector(first_guess),
                                                             tolerance, max_iterations);
        } catch (...) {
            rethrow_if_aborted(abort);
            throw;
        }
        rethrow_if_aborted(abort);
        r.values = root.to_array();
        r.dims = {static_cast<int>(n)};
        return r;
    }

    if (method == "derivative") {
        if (!cbs.scalar) throw std::invalid_argument("math/derivative requires a scalar function");
        GuardedCall<double, double> g(cbs.scalar, std::numeric_limits<double>::quiet_NaN());
        double point = require_double(o, "point", "math/derivative");
        double d = 0.0;
        // `step_size` is the last parameter, so an absent key simply drops it and the ported
        // adaptive default applies -- no copy of it here either. See the root_find arm above.
        auto fx = [&g](double x) { return g(x); };
        try {
            d = o.contains("step_size")
                    ? differentiation::derivative(fx, point, o.at("step_size").as_double())
                    : differentiation::derivative(fx, point);
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

    if (method == "quadrature") {
        if (!cbs.scalar) throw std::invalid_argument("math/quadrature requires a scalar function");
        GuardedCall<double, double> g(cbs.scalar, std::numeric_limits<double>::quiet_NaN());
        auto fx = [&g](double x) { return g(x); };
        // Absent means "gauss_kronrod", preserving every fixture that predates this key -- the
        // same rule root_find's own `method` option follows above. See the file header for the
        // full method list and the result-shape split between the Integrator-class arms and the
        // static arms.
        std::string qmethod = o.value_or("method", "gauss_kronrod");

        if (qmethod == "gauss_kronrod") {
            double lower = require_double(o, "lower", "math/quadrature");
            double upper = require_double(o, "upper", "math/quadrature");
            integration::AdaptiveGaussKronrod agk(fx, lower, upper);
            if (o.contains("absolute_tolerance"))
                agk.absolute_tolerance = o.at("absolute_tolerance").as_double();
            if (o.contains("relative_tolerance"))
                agk.relative_tolerance = o.at("relative_tolerance").as_double();
            if (o.contains("max_function_evaluations"))
                agk.max_function_evaluations = o.at("max_function_evaluations").as_int();
            try {
                agk.integrate();
            } catch (...) {
                g.rethrow_if_aborted();
                throw;
            }
            g.rethrow_if_aborted();
            r.values = {agk.result(), static_cast<double>(agk.function_evaluations()),
                        agk.standard_error()};
            r.names = {"integral", "function_evaluations", "standard_error"};
            r.status = integration::status_name(agk.status());
            return r;
        }

        if (qmethod == "simpsons" || qmethod == "trapezoidal" || qmethod == "adaptive_simpsons" ||
            qmethod == "gauss_lobatto") {
            double lower = require_double(o, "lower", "math/quadrature");
            double upper = require_double(o, "upper", "math/quadrature");
            // A small local closure over `integ` (rather than one shared polymorphic Integrator*)
            // because each class's extras differ: AdaptiveSimpsonsRule alone takes min_depth/
            // max_depth, and only it and AdaptiveGaussKronrod have a standard_error() to read --
            // SimpsonsRule, TrapezoidalRule, and AdaptiveGaussLobatto report 0.0, per the file
            // header's rule.
            double integral = 0.0, standard_error = 0.0;
            int function_evaluations = 0;
            integration::IntegrationStatus status = integration::IntegrationStatus::None;
            auto apply_tolerances = [&](integration::Integrator& integ) {
                if (o.contains("absolute_tolerance"))
                    integ.absolute_tolerance = o.at("absolute_tolerance").as_double();
                if (o.contains("relative_tolerance"))
                    integ.relative_tolerance = o.at("relative_tolerance").as_double();
                if (o.contains("max_function_evaluations"))
                    integ.max_function_evaluations = o.at("max_function_evaluations").as_int();
            };
            try {
                if (qmethod == "simpsons") {
                    integration::SimpsonsRule integ(fx, lower, upper);
                    apply_tolerances(integ);
                    integ.integrate();
                    integral = integ.result();
                    function_evaluations = integ.function_evaluations();
                    status = integ.status();
                } else if (qmethod == "trapezoidal") {
                    integration::TrapezoidalRule integ(fx, lower, upper);
                    apply_tolerances(integ);
                    integ.integrate();
                    integral = integ.result();
                    function_evaluations = integ.function_evaluations();
                    status = integ.status();
                } else if (qmethod == "adaptive_simpsons") {
                    integration::AdaptiveSimpsonsRule integ(fx, lower, upper);
                    apply_tolerances(integ);
                    if (o.contains("min_depth")) integ.min_depth = o.at("min_depth").as_int();
                    if (o.contains("max_depth")) integ.max_depth = o.at("max_depth").as_int();
                    integ.integrate();
                    integral = integ.result();
                    function_evaluations = integ.function_evaluations();
                    standard_error = integ.standard_error();
                    status = integ.status();
                } else {  // gauss_lobatto
                    integration::AdaptiveGaussLobatto integ(fx, lower, upper);
                    apply_tolerances(integ);
                    integ.integrate();
                    integral = integ.result();
                    function_evaluations = integ.function_evaluations();
                    status = integ.status();
                }
            } catch (...) {
                g.rethrow_if_aborted();
                throw;
            }
            g.rethrow_if_aborted();
            r.values = {integral, static_cast<double>(function_evaluations), standard_error};
            r.names = {"integral", "function_evaluations", "standard_error"};
            r.status = integration::status_name(status);
            return r;
        }

        if (qmethod == "gauss_legendre" || qmethod == "gauss_legendre20" ||
            qmethod == "simpsons_fixed" || qmethod == "trapezoidal_fixed" ||
            qmethod == "midpoint") {
            double lower = require_double(o, "lower", "math/quadrature");
            double upper = require_double(o, "upper", "math/quadrature");
            // `steps` applies to the three fixed-step statics alone; gauss_legendre/
            // gauss_legendre20 take no such option. Absent means the ported statics' own default
            // of 2, per the file header's rule that an absent key is not a copy of the default
            // made here.
            int steps = o.value_or("steps", 2);
            double integral = 0.0;
            try {
                if (qmethod == "gauss_legendre") {
                    integral = integration::Integration::gauss_legendre(fx, lower, upper);
                } else if (qmethod == "gauss_legendre20") {
                    integral = integration::Integration::gauss_legendre20(fx, lower, upper);
                } else if (qmethod == "simpsons_fixed") {
                    integral = integration::Integration::simpsons_rule(fx, lower, upper, steps);
                } else if (qmethod == "trapezoidal_fixed") {
                    integral = integration::Integration::trapezoidal_rule(fx, lower, upper, steps);
                } else {  // midpoint
                    integral = integration::Integration::midpoint(fx, lower, upper, steps);
                }
            } catch (...) {
                g.rethrow_if_aborted();
                throw;
            }
            g.rethrow_if_aborted();
            r.values = {integral};
            r.names = {"integral"};
            r.status = "Success";
            return r;
        }

        throw std::invalid_argument("math/quadrature: unknown 'method' '" + qmethod + "'");
    }

    if (method == "quadrature_2d") {
        if (!cbs.scalar_xy)
            throw std::invalid_argument("math/quadrature_2d requires a scalar function 'f(x, y)'");
        GuardedCall<double, double, double> g(cbs.scalar_xy,
                                              std::numeric_limits<double>::quiet_NaN());
        auto fxy = [&g](double x, double y) { return g(x, y); };
        double min_x = require_double(o, "min_x", "math/quadrature_2d");
        double max_x = require_double(o, "max_x", "math/quadrature_2d");
        double min_y = require_double(o, "min_y", "math/quadrature_2d");
        double max_y = require_double(o, "max_y", "math/quadrature_2d");
        integration::AdaptiveSimpsonsRule2D integ(fxy, min_x, max_x, min_y, max_y);
        if (o.contains("absolute_tolerance"))
            integ.absolute_tolerance = o.at("absolute_tolerance").as_double();
        if (o.contains("relative_tolerance"))
            integ.relative_tolerance = o.at("relative_tolerance").as_double();
        if (o.contains("min_depth")) integ.min_depth = o.at("min_depth").as_int();
        if (o.contains("max_depth")) integ.max_depth = o.at("max_depth").as_int();
        try {
            integ.integrate();
        } catch (...) {
            g.rethrow_if_aborted();
            throw;
        }
        g.rethrow_if_aborted();
        r.values = {integ.result(), static_cast<double>(integ.function_evaluations()),
                    integ.standard_error()};
        r.names = {"integral", "function_evaluations", "standard_error"};
        r.status = integration::status_name(integ.status());
        return r;
    }

    if (method == "quadrature_nd") {
        if (!cbs.vector_scalar)
            throw std::invalid_argument("math/quadrature_nd requires a vector function 'f(x)'");
        GuardedCall<double, const std::vector<double>&> g(
            cbs.vector_scalar, std::numeric_limits<double>::quiet_NaN());
        auto fx = [&g](const std::vector<double>& x) { return g(x); };
        std::vector<double> min_v = require_vector(o, "min", "math/quadrature_nd");
        std::vector<double> max_v = require_vector(o, "max", "math/quadrature_nd");
        if (min_v.size() != max_v.size())
            throw std::invalid_argument("math/quadrature_nd: 'min' and 'max' must be the same length");
        int dimensions = static_cast<int>(min_v.size());
        // Absent means "monte_carlo", preserving the same "first arm listed is the default" rule
        // `quadrature`'s own `method` option follows. See the file header for the full grammar.
        std::string qmethod = o.value_or("method", "monte_carlo");
        bool use_sobol = o.value_or("use_sobol", true);
        std::string sobol_path = o.value_or("sobol_path", "");

        double integral = 0.0, standard_error = 0.0;
        int function_evaluations = 0;
        integration::IntegrationStatus status = integration::IntegrationStatus::None;
        try {
            if (qmethod == "monte_carlo") {
                integration::MonteCarloIntegration mc(fx, dimensions, min_v, max_v);
                if (o.contains("seed"))
                    mc.random = bfsampling::MersenneTwister(
                        static_cast<std::uint32_t>(o.at("seed").as_int()));
                mc.use_sobol_sequence = use_sobol;
                // See the file header: these three are the real throttle on this class's own
                // loop, `max_function_evaluations` alone is not.
                if (o.contains("min_iterations")) mc.min_iterations = o.at("min_iterations").as_int();
                if (o.contains("max_iterations")) mc.max_iterations = o.at("max_iterations").as_int();
                if (o.contains("relative_tolerance"))
                    mc.relative_tolerance = o.at("relative_tolerance").as_double();
                if (o.contains("max_function_evaluations"))
                    mc.max_function_evaluations = o.at("max_function_evaluations").as_int();
                mc.integrate();
                integral = mc.result();
                standard_error = mc.standard_error();
                function_evaluations = mc.function_evaluations();
                status = mc.status();
            } else if (qmethod == "miser") {
                integration::Miser miser(fx, dimensions, min_v, max_v, sobol_path);
                if (o.contains("seed"))
                    miser.random = bfsampling::MersenneTwister(
                        static_cast<std::uint32_t>(o.at("seed").as_int()));
                miser.use_sobol_sequence = use_sobol;
                if (o.contains("max_function_evaluations"))
                    miser.max_function_evaluations = o.at("max_function_evaluations").as_int();
                if (o.contains("fraction")) miser.fraction = o.at("fraction").as_double();
                if (o.contains("min_subregion_points"))
                    miser.minimum_number_of_subregion_points = o.at("min_subregion_points").as_int();
                if (o.contains("min_bisections"))
                    miser.minimum_number_of_bisections = o.at("min_bisections").as_int();
                if (o.contains("dither")) miser.dither = o.at("dither").as_double();
                miser.integrate();
                integral = miser.result();
                standard_error = miser.standard_error();
                function_evaluations = miser.function_evaluations();
                status = miser.status();
            } else {
                throw std::invalid_argument("math/quadrature_nd: unknown 'method' '" + qmethod +
                                            "'; expected 'monte_carlo' or 'miser'");
            }
        } catch (...) {
            g.rethrow_if_aborted();
            throw;
        }
        g.rethrow_if_aborted();
        r.values = {integral, static_cast<double>(function_evaluations), standard_error};
        r.names = {"integral", "function_evaluations", "standard_error"};
        r.status = integration::status_name(status);
        return r;
    }

    if (method == "quadrature_vegas") {
        if (!cbs.vector_weight)
            throw std::invalid_argument(
                "math/quadrature_vegas requires a vector-weight function 'f(x, weight)'");
        GuardedCall<double, const std::vector<double>&, double> g(
            cbs.vector_weight, std::numeric_limits<double>::quiet_NaN());
        auto fxw = [&g](const std::vector<double>& x, double w) { return g(x, w); };
        std::vector<double> min_v = require_vector(o, "min", "math/quadrature_vegas");
        std::vector<double> max_v = require_vector(o, "max", "math/quadrature_vegas");
        if (min_v.size() != max_v.size())
            throw std::invalid_argument(
                "math/quadrature_vegas: 'min' and 'max' must be the same length");
        int dimensions = static_cast<int>(min_v.size());
        std::string sobol_path = o.value_or("sobol_path", "");

        integration::Vegas vegas(fxw, dimensions, min_v, max_v, sobol_path);
        if (o.contains("seed"))
            vegas.random =
                bfsampling::MersenneTwister(static_cast<std::uint32_t>(o.at("seed").as_int()));
        vegas.use_sobol_sequence = o.value_or("use_sobol", true);
        if (o.contains("independent_evaluations"))
            vegas.independent_evaluations = o.at("independent_evaluations").as_int();
        if (o.contains("function_calls")) vegas.function_calls = o.at("function_calls").as_int();
        if (o.contains("alpha")) vegas.alpha = o.at("alpha").as_double();
        if (o.contains("number_of_bins")) vegas.set_number_of_bins(o.at("number_of_bins").as_int());
        if (o.contains("tail_focus_parameter"))
            vegas.tail_focus_parameter = o.at("tail_focus_parameter").as_double();
        if (o.contains("initialize")) vegas.initialize = o.at("initialize").as_int();
        if (o.contains("check_convergence"))
            vegas.check_convergence = o.at("check_convergence").as_bool();
        // PRESENT -> configure_for_rare_events, applied LAST so it can override number_of_bins/
        // alpha the same way the ported method itself does. See the file header.
        if (o.contains("target_probability"))
            vegas.configure_for_rare_events(o.at("target_probability").as_double());

        try {
            vegas.integrate();
        } catch (...) {
            g.rethrow_if_aborted();
            throw;
        }
        g.rethrow_if_aborted();
        r.values = {vegas.result(), static_cast<double>(vegas.function_evaluations()),
                    vegas.standard_error(), vegas.chi_squared()};
        r.names = {"integral", "function_evaluations", "standard_error", "chi_squared"};
        r.status = integration::status_name(vegas.status());
        return r;
    }

    throw std::invalid_argument("unknown math method: " + method);
}

}  // namespace corehydro::numerics::support::detail
