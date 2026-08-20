// corehydro ADDITION -- no upstream C# counterpart. The gmm group of callback_runner.hpp:
// running GeneralizedMethodOfMoments against the user's own moment-condition delegates.
//
// This is upstream's OWN API, not an addition. GeneralizedMethodOfMoments.cs declares four
// file-scope delegates (C# 18-58) and has TWO constructors: one taking an `IGMMModel` (C# 106) and
// one taking the delegates directly (C# 143). The packages could reach only the first, and
// `IGMMModel` has exactly one implementation -- `Bulletin17CDistribution` -- so `fit_gmm()` fits an
// LP3 flood-frequency model and nothing else. This group reaches the second constructor, which is
// how a C# caller writes their own moment conditions.
//
// One method, `fit`. Options grammar (JSON object):
//
//   {"initial": [...], "lower": [...], "upper": [...], "sample_size": 100,
//    "number_of_moment_conditions": 2, "optimizer": "BFGS", "strategy": "Iterative",
//    "max_gmm_iterations": 20}
//
// `initial`, `lower`, `upper` and `sample_size` are REQUIRED. The three bound vectors must be the
// same length, which IS the parameter count p; there is no unbounded form, because every ported
// optimizer this can dispatch to takes a box and the numerical Jacobian's step selection is
// bounds-aware. `sample_size` is the n the sandwich covariance divides by: the number of
// observations behind the moment conditions, which only the caller knows -- the delegates hand
// over averages, not data.
//
// `optimizer` and `strategy` are parsed by the same two functions fit_gmm() uses
// (estimation/support/estimator_names.hpp), so the two verbs cannot drift apart on what they
// accept. An absent `max_gmm_iterations` leaves the estimator's own cap in force.
//
// `number_of_moment_conditions` is OPTIONAL and is a cross-check, not a declaration: q is measured
// by evaluating the moment-condition callback once at `initial` before anything ported is built
// (the up-front probe, below). Supplying a q that disagrees with what the callback returns is
// refused by name rather than silently believed.
//
// THE UP-FRONT PROBE. `moment_conditions(initial)` is evaluated once, directly and unguarded,
// before the estimator exists. It is unguarded on purpose: nothing ported is running yet, so an
// exception from it is already the user's own and travels straight out. It costs one extra
// evaluation of a pure function and it buys two things -- q, which the caller then does not have to
// state, and a sentinel of the right SHAPE for the guard (see below).
//
// THE RETURN SHAPE IS THE MISTAKE THIS SURFACE INVITES, because the callback returns two things at
// once: upstream's `(Vector G, Matrix S)` tuple, the mean moment-condition vector and its q x q
// covariance. Both are validated INSIDE the guarded function -- so a wrong shape aborts the fit
// exactly as a host-language error does -- and every message names `g` or `s` explicitly rather
// than reporting a length. A caller who returns the two the wrong way round gets told which one is
// wrong, not a covariance matrix quietly built from a moment vector.
//
// GUARD DISCIPLINE. Three callbacks can be live at once -- the moment conditions, the optional
// Jacobian and the optional penalty -- and all three guards share ONE abort state. That is
// load-bearing rather than tidy: with private states, an R error raised inside the moment
// conditions latches only that guard, and `Optimizer::minimize()` -- whose catch-all
// GeneralizedMethodOfMoments::minimize_with_fallback swallows without rethrowing -- goes on to the
// next iteration and calls the Jacobian, re-entering R with an unwind already pending. Sharing
// makes the first throw short-circuit every callback in the group (see callback_guard.hpp's "USE A
// SHARED STATE" note); core/tests/test_callback_runner.cpp asserts it per delegate, by making one
// throw and giving another a body that reports being entered afterwards.
//
// EACH GUARD'S SENTINEL IS A VALUE ITS PORTED CONSUMER CAN DO ARITHMETIC ON WITHOUT THROWING OR
// RUNNING AWAY, because an aborted fit still has to reach the trailing rethrow:
//
//   - moment conditions: g = 0, S = the q x q identity. A zero g makes Q = g'Wg = 0 everywhere, so
//     the optimizer stops almost at once instead of wandering; the identity is unconditionally
//     symmetric positive definite and invertible, so neither MatrixRegularization nor the two-step
//     W = S^-1 can throw an INTERNAL exception that would replace the user's own message. An
//     all-zeros S would be singular, which is exactly the trap the wrap below exists for and not a
//     trap worth walking into.
//   - jacobian: a q x p zero matrix. It is the right SHAPE, which is the load-bearing part -- the
//     bread D'WD is a shape-checked matrix product, and a mis-sized sentinel throws from inside the
//     linear algebra rather than reporting the user's error.
//   - penalty: 0, which is what "no penalty" means to Q().
//
// NON-FINITE VALUES follow the same rule the bootstrap group draws, applied to what each value
// MEANS. `Q()` ends with `is_finite(qv) ? qv : double.max` -- a non-finite quadratic form is the
// ported estimator's own vocabulary for "this trial point is unusable", which is how a bounded
// optimizer is supposed to learn that a corner of its box is infeasible. So a non-finite `g` (and a
// non-finite penalty) passes straight through. `s` and the Jacobian are the opposite case: nothing
// in the ported class tests either for finiteness, and a NaN there propagates silently into the
// weighting matrix or the covariance. Both bindings' converters therefore refuse NA/NaN in `s` and
// in the Jacobian and allow it in `g` and the penalty; see the note in each glue.
//
// The drive site is wrapped AND the trailing rethrow is kept, and it is worth recording which of
// the two the tests actually pin rather than implying both. Deleting the TRAILING rethrow fails
// seven checks in core/tests/test_callback_runner.cpp, all of them "did not throw": an aborted run
// walks to the end -- its zero-moment sentinel makes the moment residual covariance exactly
// singular, `post_process` throws, and that throw is deliberately swallowed just below -- so
// nothing else would ever report the user's exception, and the fit would come back looking
// finished. Deleting the WRAP fails nothing today, because the one path it protects is unreachable
// with this sentinel: `estimate()` returning false while an abort is latched would raise this
// file's own "estimate() failed with optimizer ..." in place of the user's message, and a zero
// moment vector always leaves the optimizer a usable point. It stays anyway -- it is the contract
// callback_guard.hpp states for every drive site, it costs one branch, and a future sentinel or
// option combination that makes estimate() fail is exactly what it is there for.
//
// THE POST-PROCESSING CALL IS A SECOND DRIVE SITE, and an easy one to miss: post_process()
// recomputes S, the Jacobian and g at the fitted parameters, so it re-enters the user's moment
// conditions AFTER estimate() has returned, and its own throw is swallowed just below. It carries
// the same abort check for the same reason. What the tests DO pin there is the observable promise
// -- a moment function that throws for the first time inside that re-entry surfaces the user's own
// message, not the ported "Singular matrix in LU decomposition" the sentinel provokes (see
// core/tests/test_callback_runner.cpp and both package suites, which throw on the LAST call a
// clean run makes and so land inside post_process by construction). What they do NOT pin is the
// abort check inside the catch itself: every exception the guards can latch is a std::exception, so
// the old narrower `catch (const std::exception&)` reached the trailing rethrow anyway and reported
// the same message. The check is the contract, not a fix for a reachable wrong answer, and saying
// so is better than implying a test discriminates it.
//
// Result layout, all of it named (`dims` is `{p, p}`, the covariance's shape):
//
//   parameter[j], standard_error[j]                 -- BestParameterSet.Values, GetStandardErrors()
//   covariance[i,j], correlation[i,j]               -- GetCovarianceMatrix(), GetCorrelationMatrix()
//   j_stat, j_stat_pval, degree_of_freedom          -- Jstat(), JstatPval(), max(0, q - p)
//   gmm_iterations, converged_within_tolerance, optimizer_fallback_count
//   sample_size, number_of_parameters, number_of_moment_conditions
//
// J-STATISTIC NOTE, so nobody reads the NaN as a bug: `JstatPval` is NaN whenever q == p. A
// just-identified fit has zero degrees of freedom and cannot test its own specification -- there is
// no over-identifying restriction left over to test -- and the ported post_process() writes NaN
// there deliberately. `j_stat` itself is then ~0 by construction (g(theta-hat) is driven to zero)
// and is pure floating-point cancellation noise, not reproducible across compilers -- and not always
// COMPUTABLE, since V is theoretically zero and inverting it either returns that noise or throws
// (see the swallow at the drive site, which reports NaN rather than failing the fit); see
// docs/upstream-csharp-issues.md and fixtures/estimation/gmm_bulletin17c_smoke.json, which pins the
// same NaN for the always-just-identified Bulletin 17C fit.
//
// NOT EXPOSED, deliberately: the fourth C# delegate, `PointwiseMomentConditionFunction` (C# 52-58).
// Its only consumer is the GMM Influence/Leverage Diagnostics region, which the packages reach
// through fit_diagnostics() on a model-based fit; accepting a delegate here that nothing on this
// surface would ever call would be a worse answer than leaving it out. Reaching the diagnostics
// from a delegate-built fit is a severable follow-up, not part of this group.
#pragma once
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/estimation/generalized_method_of_moments.hpp"
#include "corehydro/estimation/gmm_delegates.hpp"
#include "corehydro/estimation/support/estimator_names.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/math/optimization/support/optimization_status.hpp"
#include "corehydro/numerics/support/callback/common.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"

namespace corehydro::numerics::support::detail {

namespace chest = corehydro::estimation;
namespace chlin = corehydro::numerics::math::linalg;

// The shape contract of one moment-condition return, checked in exactly one place so the up-front
// probe and every guarded call afterwards refuse the same things with the same words. `q` is 0 on
// the probe (nothing to compare against yet) and the measured count on every call after it.
inline void check_moment_condition_return(const MomentConditionReturn& out, std::size_t q) {
    if (out.g.empty())
        throw std::invalid_argument(
            "the moment condition function must return at least one moment condition; its 'g' (the "
            "moment vector) was empty");
    if (q != 0 && out.g.size() != q)
        throw std::invalid_argument(
            "the moment condition function's 'g' (the moment vector) must hold one value per "
            "moment condition; it returned " +
            std::to_string(q) + " the first time and " + std::to_string(out.g.size()) +
            " this time");
    const std::size_t rows = out.g.size();
    if (out.s_rows != static_cast<int>(rows) || out.s_cols != static_cast<int>(rows))
        throw std::invalid_argument(
            "the moment condition function's 's' (the weighting matrix) must be a square matrix "
            "with one row and one column per moment condition; 'g' has " +
            std::to_string(rows) + " and 's' is " + std::to_string(out.s_rows) + " x " +
            std::to_string(out.s_cols));
    if (out.s.size() != rows * rows)
        throw std::invalid_argument(
            "the moment condition function's 's' (the weighting matrix) holds " +
            std::to_string(out.s.size()) + " values but says it is " + std::to_string(out.s_rows) +
            " x " + std::to_string(out.s_cols));
}

inline CallbackResult run_gmm(const std::string& method, const JsonValue& o,
                              const CallbackSet& cbs) {
    if (method != "fit") throw std::invalid_argument("unknown gmm method: " + method);
    if (!cbs.moment_conditions)
        throw std::invalid_argument(
            "gmm/fit requires a moment condition function, called with (parameters) and returning "
            "the moment vector 'g' and the weighting matrix 's'");

    std::vector<double> initial = require_vector(o, "initial", "gmm/fit");
    std::vector<double> lower = require_vector(o, "lower", "gmm/fit");
    std::vector<double> upper = require_vector(o, "upper", "gmm/fit");
    if (initial.empty())
        throw std::invalid_argument("gmm/fit requires 'initial' to hold at least one value");
    if (lower.size() != initial.size() || upper.size() != initial.size())
        throw std::invalid_argument(
            "gmm/fit requires 'initial', 'lower' and 'upper' to be the same length; they are " +
            std::to_string(initial.size()) + ", " + std::to_string(lower.size()) + " and " +
            std::to_string(upper.size()));
    const int p = static_cast<int>(initial.size());

    const int sample_size = static_cast<int>(require_double(o, "sample_size", "gmm/fit"));
    if (sample_size < 1)
        throw std::invalid_argument(
            "gmm/fit requires 'sample_size' to be at least 1: it is the number of observations "
            "behind the moment conditions, which the sandwich covariance divides by");

    // Parsed by fit_gmm()'s own two functions, so the two verbs accept exactly the same names.
    const chest::OptimizationMethod optimizer =
        chest::support::parse_optimizer(o.value_or("optimizer", "BFGS"));
    const auto strategy = chest::support::parse_gmm_strategy(o.value_or("strategy", "Iterative"));
    const int max_gmm_iterations = o.contains("max_gmm_iterations")
                                       ? o.at("max_gmm_iterations").as_int()
                                       : 0;

    // The up-front probe -- see the file header. Unguarded: nothing ported is running yet.
    MomentConditionReturn probe = cbs.moment_conditions(initial);
    check_moment_condition_return(probe, 0);
    const std::size_t q = probe.g.size();
    if (o.contains("number_of_moment_conditions")) {
        const int declared = o.at("number_of_moment_conditions").as_int();
        if (declared != static_cast<int>(q))
            throw std::invalid_argument(
                "'number_of_moment_conditions' says " + std::to_string(declared) +
                " but the moment condition function returned " + std::to_string(q) +
                " values in its 'g' (the moment vector)");
    }

    // ONE abort state for all three guards. See the file header.
    CallbackAbortStatePtr abort = make_abort_state();

    MomentConditionReturn moment_sentinel;
    moment_sentinel.g.assign(q, 0.0);
    moment_sentinel.s.assign(q * q, 0.0);
    for (std::size_t i = 0; i < q; ++i) moment_sentinel.s[i * q + i] = 1.0;  // the identity
    moment_sentinel.s_rows = static_cast<int>(q);
    moment_sentinel.s_cols = static_cast<int>(q);

    GuardedCall<MomentConditionReturn, const std::vector<double>&> moment(
        [fn = cbs.moment_conditions, q](const std::vector<double>& parameters) {
            MomentConditionReturn out = fn(parameters);
            check_moment_condition_return(out, q);
            return out;
        },
        moment_sentinel, abort);

    using MatrixReturn = std::pair<std::vector<double>, std::vector<int>>;
    GuardedCall<MatrixReturn, const std::vector<double>&> jacobian(
        cbs.vector_matrix
            ? std::function<MatrixReturn(const std::vector<double>&)>(
                  [fn = cbs.vector_matrix, q, p](const std::vector<double>& parameters) {
                      MatrixReturn out = fn(parameters);
                      const bool shaped = out.second.size() == 2 &&
                                          out.second[0] == static_cast<int>(q) &&
                                          out.second[1] == p;
                      if (!shaped || out.first.size() != q * static_cast<std::size_t>(p))
                          throw std::invalid_argument(
                              "the jacobian function must return a " + std::to_string(q) + " x " +
                              std::to_string(p) +
                              " matrix -- one ROW per moment condition, one COLUMN per parameter; "
                              "it returned " +
                              (out.second.size() == 2 ? std::to_string(out.second[0]) + " x " +
                                                            std::to_string(out.second[1])
                                                      : std::to_string(out.first.size()) +
                                                            " values with no shape"));
                      return out;
                  })
            : std::function<MatrixReturn(const std::vector<double>&)>(),
        MatrixReturn{std::vector<double>(q * static_cast<std::size_t>(p), 0.0),
                     std::vector<int>{static_cast<int>(q), p}},
        abort);

    GuardedCall<double, const std::vector<double>&> penalty(cbs.vector_scalar, 0.0, abort);

    chest::MomentConditionFunction moment_delegate =
        [&moment, q](const std::vector<double>& parameters) {
            MomentConditionReturn out = moment(parameters);
            return chest::MomentConditionResult{
                chlin::Vector(out.g), chlin::Matrix(static_cast<int>(q), static_cast<int>(q),
                                                    out.s)};
        };
    chest::JacobianFunction jacobian_delegate;
    if (cbs.vector_matrix)
        jacobian_delegate = [&jacobian, q, p](const std::vector<double>& parameters) {
            MatrixReturn out = jacobian(parameters);
            chlin::Matrix2D rows(q, std::vector<double>(static_cast<std::size_t>(p), 0.0));
            for (std::size_t i = 0; i < q; ++i)
                for (int j = 0; j < p; ++j)
                    rows[i][static_cast<std::size_t>(j)] =
                        out.first[i * static_cast<std::size_t>(p) + static_cast<std::size_t>(j)];
            return rows;
        };
    chest::PenaltyFunction penalty_delegate;
    if (cbs.vector_scalar)
        penalty_delegate = [&penalty](const std::vector<double>& parameters) {
            return penalty(parameters);
        };

    chest::GeneralizedMethodOfMoments gmm(moment_delegate, p, static_cast<int>(q), sample_size,
                                          initial, lower, upper, std::nullopt, jacobian_delegate,
                                          penalty_delegate, nullptr);
    gmm.set_optimizer_method(optimizer);
    gmm.set_estimation_strategy(strategy);
    if (max_gmm_iterations > 0) gmm.set_max_gmm_iterations(max_gmm_iterations);

    // See the file header on why BOTH halves are load-bearing here, and why the rethrow is taken
    // off the shared `abort` state rather than off whichever of the three guards sits closest.
    try {
        if (!gmm.estimate())
            throw std::runtime_error(
                "GeneralizedMethodOfMoments::estimate() failed with optimizer " +
                std::string(o.value_or("optimizer", "BFGS")));
    } catch (...) {
        rethrow_if_aborted(abort);
        throw;
    }

    // The same post-processing fit_gmm() runs -- the sandwich covariance and the J-statistic --
    // except that a J-statistic that cannot be computed is reported as NaN instead of failing an
    // otherwise perfect fit. See the file header's J-STATISTIC NOTE: for a just-identified fit
    // (q == p, which is the ordinary case here) the moment residual covariance V is THEORETICALLY
    // ZERO, so `V.inverse()` in post_process either returns cancellation noise or throws "Singular
    // matrix in LU decomposition", and which of the two happens turns on the last bits of a
    // structurally zero matrix -- measured: the same moment conditions under NelderMead throw from
    // R and Python and do not from the C++ fixture runner, whose compiled callback contracts
    // `a * a - theta` into an FMA and lands a few ulp away. Failing the fit on that is indefensible:
    // the parameters are exact, and the quantity that threw is one the estimator itself declares
    // meaningless by writing NaN into its p-value. `sigma_` is assigned BEFORE the J-statistic
    // inside post_process, so the covariance survives the catch.
    //
    // This is also what makes the TRAILING rethrow below load-bearing: an aborted run's sentinel
    // drives V singular, that throw is swallowed here, and nothing else would report the user's
    // exception.
    bool jstat_computed = true;
    try {
        gmm.post_process(/*use_sandwich=*/true, /*compute_jstat=*/true);
    } catch (...) {
        // `catch (...)` WITH the abort check, not `catch (const std::exception&)` without it. This
        // is the one drive site in the file that RE-ENTERS the user's moment conditions after
        // estimate() has returned -- post_process recomputes S, the Jacobian and g at the fitted
        // parameters -- so it is a place a host-language error can be raised for the first time,
        // and callback_guard.hpp's contract says a drive site that can latch must consult the latch
        // before it swallows anything. Without the check the swallow below is the last word on any
        // exception this frame does not recognize as a std::exception.
        rethrow_if_aborted(abort);
        // Otherwise deliberately silent, in the ported class's own idiom (its GetCovariance and
        // MinimizeWithFallback swallow the same way). The flag is load-bearing rather than
        // decorative: the ported members initialize to 0.0, not NaN (C# `JStat { get; } = 0`), so
        // WITHOUT it a J-statistic that could not be computed would be reported as a p-value of
        // exactly zero -- a decisive rejection of the specification, which is the worst possible
        // reading of "this could not be computed".
        jstat_computed = false;
    }
    rethrow_if_aborted(abort);

    CallbackResult r;
    r.status = corehydro::numerics::math::optimization::status_name(gmm.status());
    auto push = [&r](const std::string& name, double value) {
        r.names.push_back(name);
        r.values.push_back(value);
    };
    auto push_matrix = [&push, p](const char* label, const chlin::Matrix& m) {
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j)
                push(std::string(label) + "[" + std::to_string(i) + "," + std::to_string(j) + "]",
                     m(i, j));
    };

    const std::vector<double>& best = gmm.best_parameter_set().values;
    for (int j = 0; j < p; ++j) push("parameter[" + std::to_string(j) + "]", best[static_cast<std::size_t>(j)]);
    std::vector<double> se = gmm.get_standard_errors();
    for (int j = 0; j < p; ++j)
        push("standard_error[" + std::to_string(j) + "]", se[static_cast<std::size_t>(j)]);
    push_matrix("covariance", gmm.get_covariance_matrix());
    push_matrix("correlation", gmm.get_correlation_matrix());
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    push("j_stat", jstat_computed ? gmm.jstat() : kNaN);
    push("j_stat_pval", jstat_computed ? gmm.jstat_pval() : kNaN);
    push("degree_of_freedom", static_cast<double>(gmm.degree_of_freedom()));
    push("gmm_iterations", static_cast<double>(gmm.gmm_iterations()));
    push("converged_within_tolerance", gmm.converged_within_tolerance() ? 1.0 : 0.0);
    push("optimizer_fallback_count", static_cast<double>(gmm.optimizer_fallback_count()));
    push("sample_size", static_cast<double>(gmm.sample_size()));
    push("number_of_parameters", static_cast<double>(p));
    push("number_of_moment_conditions", static_cast<double>(q));

    r.dims = {p, p};
    return r;
}

}  // namespace corehydro::numerics::support::detail
