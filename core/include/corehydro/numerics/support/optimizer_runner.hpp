// corehydro ADDITION -- no upstream C# counterpart (sibling of toolbox_runner.hpp,
// distributions/support/dist_runner.hpp, and estimation/support/fit_runner.hpp).
//
// The single place a general-purpose optimizer runs against a live objective function in this
// repo. Unlike every other toolbox/fixture surface (bulk data, scalars, and enum names, all
// serializable), an optimizer's INPUT is a callable -- so this is a separate runner, not a group
// under toolbox_runner.hpp's run_toolbox dispatch, and it is the first place a host-language
// (R/Python) callback crosses into this shared core. Four callers drive it and none owns any
// optimization logic: the cpp11 glue (corehydror/src/toolbox.cpp), the pybind11 glue
// (corehydropy/src/bindings/toolbox.cpp), the C++ fixture runner (core/tests/test_fixtures.cpp),
// and the dotnet oracle emitter, which serializes the identical spec grammar and dispatches the
// real C# TestFunctions delegates.
//
// No objective registry lives here. `core/tests/optimization_test_functions.hpp` already carries
// every C# TestFunctions.cs objective for the ctest/fixture-runner path; the R and Python fixture
// runners write their own native closures reproducing the same formulas, so every fixture case
// exercises the real host callback path -- the thing this file exists to make safe. A user's own
// `optim_minimize()`/`optim_maximize()` call always supplies the objective directly, in R or
// Python, never by name.
//
// Two of the six ported optimizers -- NelderMead and BrentSearch -- deliberately do NOT derive
// from the Optimizer base (see optimizer.hpp's file header); this runner handles that difference
// explicitly rather than forcing a common base onto them. Their maximize()/minimize() have no
// OptimizationStatus, no function_evaluations()/iterations() accessor, and (NelderMead only) no
// fitness accessor at all -- see run_optimizer's "nelder_mead"/"brent" arms below for how each
// gap is closed without touching either class's ported algorithm.
#pragma once
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/json_lite.hpp"
#include "corehydro/numerics/math/optimization/bfgs.hpp"
#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "corehydro/numerics/math/optimization/differential_evolution.hpp"
#include "corehydro/numerics/math/optimization/mlsl.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/optimization/powell.hpp"
#include "corehydro/numerics/math/optimization/support/optimization_status.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"

namespace corehydro::numerics::support {

namespace opt = corehydro::numerics::math::optimization;
using corehydro::models::spec::JsonValue;

// The user-facing objective signature: takes the trial point, returns the objective value. A
// plain const-ref signature -- deliberately narrower than the ported Optimizer::Objective (a
// mutable reference, kept for C# array-reference-semantics fidelity; see optimizer.hpp's
// MUTABLE-POINT SEMANTICS note) -- because a host-language callback (an R closure, a Python
// function) has no business mutating the optimizer's own working vector, and every existing
// caller of this runner (fixtures, the R/Python glue) hands over a read-only function anyway.
using Objective = std::function<double(const std::vector<double>&)>;

// Flat result surface every binding and every fixture assertion reads. `hessian`/`hessian_dims`
// are empty unless the method both supports a Hessian (the four Optimizer-base methods only --
// NelderMead and BrentSearch never compute one, see the file header) and the spec's
// `control.compute_hessian` requested it.
struct OptimResult {
    std::vector<double> parameters;
    double value = std::numeric_limits<double>::quiet_NaN();
    int iterations = 0;
    int function_evaluations = 0;
    std::string status = "None";
    std::vector<double> hessian;
    std::vector<int> hessian_dims;
};

// Wraps the caller's objective so a host-language exception (an R error arriving as a
// cpp11::unwind_exception, a Python error as pybind11::error_already_set) cannot be swallowed by
// Optimizer::minimize()'s catch-all (see optimizer.hpp's file header: a genuine objective-side
// exception only survives that catch-all as a rethrow while `report_failure` is true, and even
// then only after passing back through the ArgumentException-kind filter -- a raw host exception
// is neither an ArgumentException nor safe to let a C++ stack-unwind carry across an R/Python
// frame it did not originate in). The first throw is stored, an abort flag latches, and every
// later evaluation short-circuits without re-entering the host, so the run cannot trigger a
// second host-language exception (or a second unwind) once the first has been captured.
// run_optimizer rethrows the stored exception once the optimizer returns.
//
// The guard itself now lives in callback_guard.hpp, shared with callback_runner.hpp. The only
// optimizer-specific part is the sentinel: the optimizer is always driven through ITS OWN
// minimize()/maximize() (never negate-and-always-minimize; see run_optimizer's header comment for
// why), so the sentinel must be the worst value in the RAW objective's own sign convention.
// Minimizing wants a small value, so a huge one is worst (+inf); maximizing wants a large one, so
// a hugely negative one is worst (-inf). Either way the optimizer treats the aborted point as
// unconditionally rejected. `call_count()` is the real (non-short-circuited) call count into the
// host objective -- the only function-evaluation count available for NelderMead/BrentSearch (see
// the file header); redundant with, but not necessarily identical to, the four Optimizer-base
// classes' own function_evaluations() (incremented only inside Optimizer::evaluate(), so -- like
// that counter -- it does NOT include the post-success Hessian-differentiation probes, which call
// the objective function directly and bypass evaluate() entirely).
using GuardedObjective = GuardedCall<double, const std::vector<double>&>;

inline GuardedObjective make_guarded_objective(const Objective& fn, bool maximize) {
    return GuardedObjective(fn, maximize ? -std::numeric_limits<double>::infinity()
                                         : std::numeric_limits<double>::infinity());
}

namespace detail {

// A thin forwarder to the one definition, which lives beside the enum
// (optimization/support/optimization_status.hpp). This file and fit_runner.hpp each carried an
// identical copy of the switch until callback/gmm.hpp would have made a third.
inline std::string optim_status_name(opt::OptimizationStatus s) { return opt::status_name(s); }

inline std::vector<double> spec_vector(const JsonValue& spec, const char* key) {
    if (!spec.contains(key)) return {};
    return spec.at(key).as_double_vector();
}

// Applies the three tolerance/iteration knobs every one of the six optimizer classes exposes
// (max_iterations, absolute_tolerance, relative_tolerance), only when the spec's control object
// carries the key -- an absent key leaves the ported class's own default untouched.
template <typename TOpt>
void apply_common_controls(TOpt& o, const JsonValue& control) {
    if (control.contains("max_iterations")) o.max_iterations = control.at("max_iterations").as_int();
    if (control.contains("absolute_tolerance"))
        o.absolute_tolerance = control.at("absolute_tolerance").as_double();
    if (control.contains("relative_tolerance"))
        o.relative_tolerance = control.at("relative_tolerance").as_double();
}

// Applies the extra knobs only the four real Optimizer subclasses (DE/BFGS/Powell/MLSL) expose:
// max_function_evaluations, report_failure, compute_hessian. NelderMead/BrentSearch have none of
// these (see the file header), so they never call this helper.
template <typename TOpt>
void apply_optimizer_controls(TOpt& o, const JsonValue& control) {
    if (control.contains("max_function_evaluations"))
        o.max_function_evaluations = control.at("max_function_evaluations").as_int();
    if (control.contains("report_failure")) o.report_failure = control.at("report_failure").as_bool();
    if (control.contains("compute_hessian")) o.compute_hessian = control.at("compute_hessian").as_bool();
}

// Fills the common block from any of the four real Optimizer subclasses (DE/BFGS/Powell/MLSL),
// which share the full public accessor surface (best_parameter_set/iterations/
// function_evaluations/status/hessian). `value` un-applies the internal function_scale sign
// convention (see optimizer.hpp: fitness = function_scale * raw objective, function_scale = -1
// under maximize()) so it always reports the same sign the user's own objective returns --
// mirroring, e.g., R's optimize(f, maximum = TRUE)$objective, which is not negated either.
template <typename TOpt>
void fill_optimizer_result(OptimResult& r, const TOpt& o, bool maximize) {
    const auto& best = o.best_parameter_set();
    r.parameters = best.values;
    r.value = maximize ? -best.fitness : best.fitness;
    r.iterations = o.iterations();
    r.function_evaluations = o.function_evaluations();
    r.status = optim_status_name(o.status());
    if (o.hessian().has_value()) {
        const auto& h = *o.hessian();
        int n = h.number_of_rows();
        int m = h.number_of_columns();
        r.hessian.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(m));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                r.hessian[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) +
                          static_cast<std::size_t>(j)] = h(i, j);
        r.hessian_dims = {n, m};
    }
}

}  // namespace detail

// Runs the optimizer named by `spec_json["method"]` (one of "de", "bfgs", "powell", "mlsl",
// "nelder_mead", "brent") against `objective`, and returns a flat OptimResult. Spec grammar:
//
//   {"method": "de|bfgs|powell|mlsl|nelder_mead|brent",
//    "lower": [...], "upper": [...], "initial": [...],
//    "maximize": false, "seed": 12345,
//    "control": {"max_iterations": 1000, "max_function_evaluations": 100000,
//                "absolute_tolerance": 1e-8, "relative_tolerance": 1e-8,
//                "report_failure": true, "compute_hessian": false,
//                "population_size": 30}}
//
// "de"/"brent" need "lower"/"upper" only; "bfgs"/"powell"/"mlsl"/"nelder_mead" additionally need
// "initial" (all three are then required to be the same length -- the underlying ctors validate
// this; see each optimizer's own header). "seed" only applies to the two stochastic methods
// ("de", "mlsl": both default their own prng_seed to 12345 when omitted, matching the ported
// class field default). Every "control" key is applied only when present; an absent key leaves
// the ported class's own default. Argument-shape validation beyond what the ported constructors
// already do (missing bounds/initial, mismatched lengths) is deliberately NOT duplicated here --
// see the file header on this being a thin dispatcher, and R/toolbox: optim_run()/
// corehydropy.optim for the complete, symmetric, user-facing validation.
//
// DECISION (documented per the task brief): a `"maximize": true` request calls the optimizer's
// OWN maximize() rather than negating the objective and calling minimize(). maximize() matches
// the real C# call site (Minimize()/Maximize() are separate methods there too), so a seeded run's
// PRNG draw sequence -- and therefore its result -- reproduces the C# run exactly: the ported
// classes already compute `fitness = function_scale * raw_objective(values)` internally
// (function_scale flips to -1 under maximize()), so calling maximize() and calling minimize() on
// a pre-negated objective would drive DE/MLSL's PRNG through the identical draw sequence either
// way -- but negating the objective ourselves would be reimplementing that sign flip a second
// time on top of the ported one, and every other optimizer's convergence test
// (Optimizer::check_convergence, NelderMead's own) compares raw fitness values, so getting the
// sign wrong anywhere in a hand-rolled negation would silently break convergence detection.
// Calling the class's own maximize() needs no such care. See make_guarded_objective() above for
// the one place this decision has an observable consequence: which infinity is "worst" for the
// aborted-objective path.
inline OptimResult run_optimizer(const std::string& spec_json, const Objective& objective) {
    JsonValue spec = corehydro::models::spec::parse_json(spec_json);
    std::string method = spec.at("method").as_string();
    bool maximize = spec.value_or("maximize", false);
    bool has_control = spec.contains("control");
    JsonValue control = has_control ? spec.at("control") : JsonValue{};

    std::vector<double> lower = detail::spec_vector(spec, "lower");
    std::vector<double> upper = detail::spec_vector(spec, "upper");
    std::vector<double> initial = detail::spec_vector(spec, "initial");

    GuardedObjective guarded = make_guarded_objective(objective, maximize);
    // Adapts GuardedObjective's const-ref call operator to the ported Optimizer-base and
    // NelderMead mutable-ref Objective signatures (see optimizer.hpp's MUTABLE-POINT SEMANTICS
    // note) -- an implicit const-ref-to-mutable-ref binding is not legal, so this lambda is the
    // adapter, not a special case.
    auto adapted = [&guarded](std::vector<double>& v) { return guarded(v); };

    OptimResult result;

    if (method == "de") {
        int D = static_cast<int>(lower.size());
        opt::DifferentialEvolution de(adapted, D, lower, upper);
        if (spec.contains("seed")) de.prng_seed = spec.at("seed").as_int();
        if (has_control) {
            detail::apply_common_controls(de, control);
            detail::apply_optimizer_controls(de, control);
            if (control.contains("population_size"))
                de.population_size = control.at("population_size").as_int();
        }
        try {
            if (maximize) de.maximize(); else de.minimize();
        } catch (...) {
            guarded.rethrow_if_aborted();
            throw;
        }
        guarded.rethrow_if_aborted();
        detail::fill_optimizer_result(result, de, maximize);
    } else if (method == "bfgs") {
        int D = static_cast<int>(initial.size());
        opt::BFGS bfgs(adapted, D, initial, lower, upper);
        if (has_control) {
            detail::apply_common_controls(bfgs, control);
            detail::apply_optimizer_controls(bfgs, control);
        }
        try {
            if (maximize) bfgs.maximize(); else bfgs.minimize();
        } catch (...) {
            guarded.rethrow_if_aborted();
            throw;
        }
        guarded.rethrow_if_aborted();
        detail::fill_optimizer_result(result, bfgs, maximize);
    } else if (method == "powell") {
        int D = static_cast<int>(initial.size());
        opt::Powell powell(adapted, D, initial, lower, upper);
        if (has_control) {
            detail::apply_common_controls(powell, control);
            detail::apply_optimizer_controls(powell, control);
        }
        try {
            if (maximize) powell.maximize(); else powell.minimize();
        } catch (...) {
            guarded.rethrow_if_aborted();
            throw;
        }
        guarded.rethrow_if_aborted();
        detail::fill_optimizer_result(result, powell, maximize);
    } else if (method == "mlsl") {
        int D = static_cast<int>(initial.size());
        opt::MLSL mlsl(adapted, D, initial, lower, upper);
        if (spec.contains("seed")) mlsl.prng_seed = spec.at("seed").as_int();
        if (has_control) {
            detail::apply_common_controls(mlsl, control);
            detail::apply_optimizer_controls(mlsl, control);
        }
        try {
            if (maximize) mlsl.maximize(); else mlsl.minimize();
        } catch (...) {
            guarded.rethrow_if_aborted();
            throw;
        }
        guarded.rethrow_if_aborted();
        detail::fill_optimizer_result(result, mlsl, maximize);
    } else if (method == "nelder_mead") {
        int D = static_cast<int>(initial.size());
        if (static_cast<int>(lower.size()) != D || static_cast<int>(upper.size()) != D)
            throw std::runtime_error(
                "optimizer 'nelder_mead' needs 'initial', 'lower' and 'upper' of the same length");
        opt::NelderMead nm(adapted, D, initial, lower, upper);
        if (has_control) detail::apply_common_controls(nm, control);
        try {
            if (maximize) nm.maximize(); else nm.minimize();
        } catch (...) {
            guarded.rethrow_if_aborted();
            throw;
        }
        guarded.rethrow_if_aborted();
        // NelderMead exposes no fitness/iterations/status accessor at all (see the file header) --
        // it is deliberately left as the standalone Phase 0 class, not extended with a
        // Hessian/status surface no current caller needs. `value` is recovered with one extra,
        // UNGUARDED call to the raw objective at the already-found optimum: this happens strictly
        // after the optimizer's own loop has returned (guarded.aborted() just tested false above),
        // so a host-language exception here propagates out of run_optimizer normally -- nothing
        // downstream still holds an Optimizer::minimize()-style catch-all that could swallow it,
        // so no guard is needed for it (going through `guarded` instead would risk this call's own
        // exception being caught and silently turned into a sentinel `value` rather than
        // propagating). `function_evaluations` is GuardedObjective's own counter (the only one
        // available for this class -- see its header comment), and therefore does not include this
        // one extra probe. `iterations` is unavailable and left at OptimResult's default (0);
        // `status` is always "Success", matching the class's own documented convention (a
        // max-iterations exit silently returns the best point found, exactly as MLSL's
        // NelderMeadLocalSolver -- see mlsl.hpp's note 5 -- already documents for the same
        // standalone class).
        result.parameters = nm.best_parameters();
        result.value = objective(result.parameters);
        result.function_evaluations = guarded.call_count();
        result.status = "Success";
    } else if (method == "brent") {
        if (lower.empty() || upper.empty())
            throw std::runtime_error("optimizer 'brent' needs 'lower' and 'upper' bounds");
        opt::BrentSearch brent(
            [&guarded](double x) { return guarded(std::vector<double>{x}); }, lower[0], upper[0]);
        if (has_control) detail::apply_common_controls(brent, control);
        try {
            if (maximize) brent.maximize(); else brent.minimize();
        } catch (...) {
            guarded.rethrow_if_aborted();
            throw;
        }
        guarded.rethrow_if_aborted();
        // BrentSearch exposes best_parameter() + best_fitness() (added in a prior task for
        // Powell's LineMinimization) but no iterations()/status (see the file header) -- iterations
        // left at OptimResult's default (0), status always "Success", matching NelderMead's
        // convention above. best_fitness() is already in the internal function_scale sign
        // convention (its own header documents this), so it un-applies exactly like
        // fill_optimizer_result's `value` above. function_evaluations is GuardedObjective's own
        // counter, every call here having gone through it (unlike NelderMead's arm, there is no
        // extra unguarded probe -- best_fitness() already carries the value).
        result.parameters = {brent.best_parameter()};
        result.value = maximize ? -brent.best_fitness() : brent.best_fitness();
        result.function_evaluations = guarded.call_count();
        result.status = "Success";
    } else {
        throw std::runtime_error("unknown optimizer method: " + method);
    }

    return result;
}

}  // namespace corehydro::numerics::support
