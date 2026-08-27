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
// Two of the fourteen ported optimizers -- NelderMead and BrentSearch -- deliberately do NOT derive
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
#include "corehydro/numerics/math/optimization/adam.hpp"
#include "corehydro/numerics/math/optimization/augmented_lagrange.hpp"
#include "corehydro/numerics/math/optimization/bfgs.hpp"
#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "corehydro/numerics/math/optimization/constraint/constraint.hpp"
#include "corehydro/numerics/math/optimization/constraint/constraint_type.hpp"
#include "corehydro/numerics/math/optimization/constraint/i_constraint.hpp"
#include "corehydro/numerics/math/optimization/differential_evolution.hpp"
#include "corehydro/numerics/math/optimization/golden_section.hpp"
#include "corehydro/numerics/math/optimization/gradient_descent.hpp"
#include "corehydro/numerics/math/optimization/mlsl.hpp"
#include "corehydro/numerics/math/optimization/multi_start.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/optimization/particle_swarm.hpp"
#include "corehydro/numerics/math/optimization/powell.hpp"
#include "corehydro/numerics/math/optimization/shuffled_complex_evolution.hpp"
#include "corehydro/numerics/math/optimization/simulated_annealing.hpp"
#include "corehydro/numerics/math/optimization/support/local_method.hpp"
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

// A gradient callback: the trial point in, one partial derivative per parameter out. Read only by
// the two gradient-taking methods ("adam", "gradient_descent"), which mirror the C# classes'
// optional `Gradient` field -- absent, both fall back to the ported NumericalDerivative.Gradient
// exactly as a null C# delegate does.
using Gradient = std::function<std::vector<double>(const std::vector<double>&)>;

// Everything a run may need from the host language. `objective` is always required; `gradient` is
// read only by the "adam"/"gradient_descent" methods, and `constraints` only by
// "augmented_lagrange", whose spec's `constraints[i]` object pairs POSITIONALLY with
// `constraints[i]` here (the serializable half of a constraint -- its type, value and tolerance --
// travels in the spec; its function half travels here, because a constraint is a callable). Every
// callback present is guarded, and all the guards share ONE abort state, so a throw in any of them
// short-circuits the rest instead of re-entering the host mid-unwind (see callback_guard.hpp's
// contract).
struct OptimCallbacks {
    Objective objective;
    Gradient gradient;
    std::vector<Objective> constraints;
};

// Flat result surface every binding and every fixture assertion reads. `hessian`/`hessian_dims`
// are empty unless the method both supports a Hessian (the nine Optimizer-base methods only --
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
    // The three Lagrange multiplier vectors, in AugmentedLagrange's own naming: `lambda` for the
    // equality constraints, `mu` for the "lesser than or equal to" ones, `nu` for the "greater
    // than or equal to" ones. Each is sized by COUNTING the constraints of that type (see
    // augmented_lagrange.hpp's transcription note 3), so a problem with no constraint of a type
    // gets an empty vector for it. All three are empty for every method except
    // "augmented_lagrange", which is the only one that has multipliers at all.
    std::vector<double> lambda;
    std::vector<double> mu;
    std::vector<double> nu;
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
// the file header); redundant with, but not necessarily identical to, the nine Optimizer-base
// classes' own function_evaluations() (incremented only inside Optimizer::evaluate(), so -- like
// that counter -- it does NOT include the post-success Hessian-differentiation probes, which call
// the objective function directly and bypass evaluate() entirely).
using GuardedObjective = GuardedCall<double, const std::vector<double>&>;

inline GuardedObjective make_guarded_objective(const Objective& fn, bool maximize) {
    return GuardedObjective(fn, maximize ? -std::numeric_limits<double>::infinity()
                                         : std::numeric_limits<double>::infinity());
}

// The gradient guard, for the two gradient-taking methods. Its sentinel is an EMPTY vector: unlike
// an objective value there is no "worst gradient" a ported consumer treats as rejected, and there
// is no length to fill at construction time anyway. Both ported classes index the returned vector
// over [0, D) with no length check, so the two arms below re-fill an ABORTED call's result with
// zeros -- a zero gradient takes no step -- rather than handing either class an empty vector; the
// run's result is discarded by rethrow_if_aborted() in any case. It shares the objective guard's
// abort state, so a throw in either callback stops both, and one rethrow_if_aborted() covers both.
using GuardedGradient = GuardedCall<std::vector<double>, const std::vector<double>&>;

namespace detail {

// A thin forwarder to the one definition, which lives beside the enum
// (optimization/support/optimization_status.hpp). This file and fit_runner.hpp each carried an
// identical copy of the switch until callback/gmm.hpp would have made a third.
inline std::string optim_status_name(opt::OptimizationStatus s) { return opt::status_name(s); }

inline std::vector<double> spec_vector(const JsonValue& spec, const char* key) {
    if (!spec.contains(key)) return {};
    return spec.at(key).as_double_vector();
}

// Applies the three tolerance/iteration knobs every one of the fourteen optimizer classes exposes
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

// Parses the LocalMethod control shared by "mlsl" and "multi_start". Only the three methods the
// two classes actually construct are accepted (ADAM/GradientDescent throw "Unsupported local
// method" inside both -- see local_method.hpp), so rejecting them here names the option rather
// than surfacing the inner class's message.
inline opt::LocalMethod parse_local_method(const std::string& s) {
    if (s == "bfgs") return opt::LocalMethod::BFGS;
    if (s == "nelder_mead") return opt::LocalMethod::NelderMead;
    if (s == "powell") return opt::LocalMethod::Powell;
    throw std::runtime_error("unknown local_method: " + s +
                             " (expected \"bfgs\", \"nelder_mead\" or \"powell\")");
}

// The short spec names for the three ConstraintType members, used by "augmented_lagrange"'s
// `constraints[i].type`. Short rather than the C# enum spelling because they are what the R and
// Python `optim_constraint()`/`Constraint` surfaces take, and those read as operators at the call
// site ("le" for <=), not as class names.
inline opt::ConstraintType parse_constraint_type(const std::string& s) {
    if (s == "eq") return opt::ConstraintType::EqualTo;
    if (s == "le") return opt::ConstraintType::LesserThanOrEqualTo;
    if (s == "ge") return opt::ConstraintType::GreaterThanOrEqualTo;
    throw std::runtime_error("unknown constraint type: " + s +
                             " (expected \"eq\", \"le\" or \"ge\")");
}

// Applies the extra knobs only the real Optimizer subclasses (DE/ParticleSwarm/SCE/
// SimulatedAnnealing/MultiStart/MLSL/BFGS/Powell/ADAM/GradientDescent/GoldenSection) expose:
// max_function_evaluations, report_failure, compute_hessian. NelderMead/BrentSearch have none of
// these (see the file header), so they never call this helper.
template <typename TOpt>
void apply_optimizer_controls(TOpt& o, const JsonValue& control) {
    if (control.contains("max_function_evaluations"))
        o.max_function_evaluations = control.at("max_function_evaluations").as_int();
    if (control.contains("report_failure")) o.report_failure = control.at("report_failure").as_bool();
    if (control.contains("compute_hessian")) o.compute_hessian = control.at("compute_hessian").as_bool();
}

// Fills the common block from any real Optimizer subclass, all of which share the full public
// accessor surface (best_parameter_set/iterations/
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

// One optimizer's construction inputs, read off a spec object. "augmented_lagrange"'s `inner`
// sub-spec has exactly the same shape as the top-level spec, so both are read by read_build() and
// built by make_optimizer() below -- the reason the eleven Optimizer-subclass arms live in one
// helper rather than inline in run_optimizer, where the constrained arm would have had to
// duplicate all eleven to build its inner optimizer.
struct OptimizerBuild {
    std::string method;
    std::vector<double> lower;
    std::vector<double> upper;
    std::vector<double> initial;
    bool has_seed = false;
    int seed = 0;
    bool has_control = false;
    JsonValue control;
};

inline OptimizerBuild read_build(const JsonValue& spec) {
    OptimizerBuild b;
    b.method = spec.at("method").as_string();
    b.lower = spec_vector(spec, "lower");
    b.upper = spec_vector(spec, "upper");
    b.initial = spec_vector(spec, "initial");
    b.has_seed = spec.contains("seed");
    if (b.has_seed) b.seed = spec.at("seed").as_int();
    b.has_control = spec.contains("control");
    if (b.has_control) b.control = spec.at("control");
    return b;
}

// Builds one fully configured optimizer from `b`, for the top-level dispatch AND for
// augmented_lagrange's `inner` sub-spec. Returns NULL for the two standalone classes
// ("nelder_mead"/"brent"), which do not derive from the Optimizer base at all (see the file
// header) and therefore can be neither returned through this pointer nor used as an
// AugmentedLagrange inner optimizer -- run_optimizer keeps its own arms for those two, and the
// "augmented_lagrange" arm rejects them by name before ever calling this.
//
// `grad_fn` is the already-guarded analytic gradient, empty unless the caller supplied one; only
// the "adam"/"gradient_descent" branches read it. The seed and every method-specific control key
// are applied here on the DERIVED type (the base has neither); the six base-wide control keys are
// applied at the bottom, after construction, because MultiStart's constructor sets max_iterations
// to 100 itself and a caller-supplied value has to win.
inline std::unique_ptr<opt::Optimizer> make_optimizer(const OptimizerBuild& b,
                                                      const opt::Optimizer::Objective& adapted,
                                                      const opt::ADAM::GradientFunction& grad_fn) {
    const std::string& method = b.method;
    const JsonValue& control = b.control;
    const bool has_control = b.has_control;
    const int Dl = static_cast<int>(b.lower.size());
    const int Di = static_cast<int>(b.initial.size());
    std::unique_ptr<opt::Optimizer> o;

    if (method == "de") {
        auto de = std::make_unique<opt::DifferentialEvolution>(adapted, Dl, b.lower, b.upper);
        if (b.has_seed) de->prng_seed = b.seed;
        if (has_control && control.contains("population_size"))
            de->population_size = control.at("population_size").as_int();
        o = std::move(de);
    } else if (method == "particle_swarm") {
        auto ps = std::make_unique<opt::ParticleSwarm>(adapted, Dl, b.lower, b.upper);
        if (b.has_seed) ps->prng_seed = b.seed;
        if (has_control && control.contains("population_size"))
            ps->population_size = control.at("population_size").as_int();
        o = std::move(ps);
    } else if (method == "sce") {
        auto sce = std::make_unique<opt::ShuffledComplexEvolution>(adapted, Dl, b.lower, b.upper);
        if (b.has_seed) sce->prng_seed = b.seed;
        if (has_control) {
            if (control.contains("complexes")) sce->complexes = control.at("complexes").as_int();
            // cce_iterations is a FIELD defaulting to 0 that the ctor sets to 2D + 1, so an absent
            // key leaves the C# default in place exactly as every other control key does.
            if (control.contains("cce_iterations"))
                sce->cce_iterations = control.at("cce_iterations").as_int();
            if (control.contains("tolerance_steps"))
                sce->tolerance_steps = control.at("tolerance_steps").as_int();
        }
        o = std::move(sce);
    } else if (method == "simulated_annealing") {
        auto sa = std::make_unique<opt::SimulatedAnnealing>(adapted, Dl, b.lower, b.upper);
        if (b.has_seed) sa->prng_seed = b.seed;
        if (has_control) {
            if (control.contains("initial_temperature"))
                sa->initial_temperature = control.at("initial_temperature").as_double();
            if (control.contains("min_temperature"))
                sa->min_temperature = control.at("min_temperature").as_double();
            if (control.contains("cooling_rate"))
                sa->cooling_rate = control.at("cooling_rate").as_double();
            if (control.contains("update_cycles"))
                sa->update_cycles = control.at("update_cycles").as_int();
            if (control.contains("temperature_cycles"))
                sa->temperature_cycles = control.at("temperature_cycles").as_int();
            // SimulatedAnnealing declares and validates tolerance_steps and then never reads it
            // (see simulated_annealing.hpp's hazard 1); applied anyway so the class's own
            // validation still sees what the caller asked for.
            if (control.contains("tolerance_steps"))
                sa->tolerance_steps = control.at("tolerance_steps").as_int();
        }
        o = std::move(sa);
    } else if (method == "multi_start") {
        auto ms = std::make_unique<opt::MultiStart>(adapted, Di, b.initial, b.lower, b.upper);
        if (b.has_seed) ms->prng_seed = b.seed;
        if (has_control) {
            if (control.contains("local_method"))
                ms->method = parse_local_method(control.at("local_method").as_string());
            if (control.contains("local_absolute_tolerance"))
                ms->local_absolute_tolerance = control.at("local_absolute_tolerance").as_double();
            if (control.contains("local_relative_tolerance"))
                ms->local_relative_tolerance = control.at("local_relative_tolerance").as_double();
            if (control.contains("polish")) ms->polish = control.at("polish").as_bool();
        }
        o = std::move(ms);
    } else if (method == "mlsl") {
        auto mlsl = std::make_unique<opt::MLSL>(adapted, Di, b.initial, b.lower, b.upper);
        if (b.has_seed) mlsl->prng_seed = b.seed;
        if (has_control && control.contains("local_method"))
            mlsl->method = parse_local_method(control.at("local_method").as_string());
        o = std::move(mlsl);
    } else if (method == "bfgs") {
        o = std::make_unique<opt::BFGS>(adapted, Di, b.initial, b.lower, b.upper);
    } else if (method == "powell") {
        o = std::make_unique<opt::Powell>(adapted, Di, b.initial, b.lower, b.upper);
    } else if (method == "adam" || method == "gradient_descent") {
        // The only two methods that read `grad_fn`. Both mirror the same C# shape (a settable
        // optional `Gradient` field; null means finite differences), so they share one branch --
        // the classes differ only in ADAM's two extra decay factors. `alpha` is a CONSTRUCTOR
        // argument in both, so it is read here rather than with the other control keys.
        double alpha = 0.001;  // both ctors' own default
        if (has_control && control.contains("alpha")) alpha = control.at("alpha").as_double();
        if (method == "adam") {
            auto adam =
                std::make_unique<opt::ADAM>(adapted, Di, b.initial, b.lower, b.upper, alpha, grad_fn);
            if (has_control) {
                if (control.contains("beta1")) adam->beta1 = control.at("beta1").as_double();
                if (control.contains("beta2")) adam->beta2 = control.at("beta2").as_double();
            }
            o = std::move(adam);
        } else {
            o = std::make_unique<opt::GradientDescent>(adapted, Di, b.initial, b.lower, b.upper,
                                                       alpha, grad_fn);
        }
    } else if (method == "golden_section") {
        if (b.lower.empty() || b.upper.empty())
            throw std::runtime_error("optimizer 'golden_section' needs 'lower' and 'upper' bounds");
        // The 1-D objective is built exactly as the "brent" arm's is, but GoldenSection IS an
        // Optimizer subclass (unlike BrentSearch -- see the file header), so it carries the full
        // best_parameter_set/iterations/function_evaluations/status/hessian surface and belongs
        // here with the rest. `adapted` is captured BY VALUE so the built optimizer does not
        // depend on this helper's argument outliving it.
        o = std::make_unique<opt::GoldenSection>(
            [adapted](double x) {
                std::vector<double> v{x};
                return adapted(v);
            },
            b.lower[0], b.upper[0]);
    } else if (method == "nelder_mead" || method == "brent") {
        return nullptr;
    } else {
        throw std::runtime_error("unknown optimizer method: " + method);
    }

    if (has_control) {
        apply_common_controls(*o, control);
        apply_optimizer_controls(*o, control);
    }
    return o;
}

}  // namespace detail

// Runs the optimizer named by `spec_json["method"]` (one of "de", "particle_swarm", "sce",
// "simulated_annealing", "multi_start", "mlsl", "bfgs", "powell", "adam", "gradient_descent",
// "nelder_mead", "brent", "golden_section", "augmented_lagrange") against `callbacks.objective`,
// and returns a flat OptimResult. Spec grammar:
//
//   {"method": "de|particle_swarm|sce|simulated_annealing|multi_start|mlsl|bfgs|powell|adam|
//               gradient_descent|nelder_mead|brent|golden_section|augmented_lagrange",
//    "lower": [...], "upper": [...], "initial": [...],
//    "maximize": false, "seed": 12345,
//    "constraints": [{"type": "eq|le|ge", "value": 22.0, "tolerance": 1e-8}],
//    "inner": {"method": "bfgs", "initial": [...], "lower": [...], "upper": [...],
//              "control": {...}},
//    "control": {"max_iterations": 1000, "max_function_evaluations": 100000,
//                "absolute_tolerance": 1e-8, "relative_tolerance": 1e-8,
//                "report_failure": true, "compute_hessian": false,
//                "population_size": 30, "complexes": 5, "cce_iterations": 5,
//                "tolerance_steps": 20, "initial_temperature": 10, "min_temperature": 0.1,
//                "cooling_rate": 0.95, "update_cycles": 4, "temperature_cycles": 10,
//                "local_method": "bfgs", "local_absolute_tolerance": 1e-8,
//                "local_relative_tolerance": 1e-8, "polish": true,
//                "alpha": 0.001, "beta1": 0.9, "beta2": 0.999}}
//
// "de"/"particle_swarm"/"sce"/"simulated_annealing"/"brent"/"golden_section" need "lower"/"upper"
// only; "bfgs"/"powell"/"mlsl"/"multi_start"/"adam"/"gradient_descent"/"nelder_mead" additionally
// need "initial" (all three are then required to be the same length -- the underlying ctors
// validate this; see each optimizer's own header). "brent" and "golden_section" are one-dimensional and read only the
// first bound of each. "seed" only applies to the six stochastic methods ("de", "particle_swarm",
// "sce", "simulated_annealing", "multi_start", "mlsl": all six default their own prng_seed to
// 12345 when omitted, matching the ported class field default). Every "control" key is applied
// only when present; an absent key leaves the ported class's own default -- which for
// "multi_start" includes max_iterations, set to 100 by ITS CONSTRUCTOR rather than by a field
// initializer, so the control application below must (and does) come after construction. Each
// method reads only the control keys of its own class; the R and Python surfaces reject the rest
// by name rather than letting them look like they did something (see R/optim.R's kOptimMethods).
// "alpha" is read by "adam" and "gradient_descent"; "beta1"/"beta2" by "adam" alone.
// `callbacks.gradient` is likewise read only by those two methods -- absent, both fall back to the
// ported NumericalDerivative.Gradient exactly as a null C# `Gradient` delegate does.
//
// "constraints" and "inner" belong to "augmented_lagrange" and to nothing else. Each
// `constraints[i]` object carries the SERIALIZABLE half of one constraint (its type, the value it
// is compared against, and the feasibility tolerance, default 1E-8); its FUNCTION half is
// `callbacks.constraints[i]`, paired positionally. "inner" names the borrowed inner optimizer,
// which may be any method except "augmented_lagrange" itself and the two standalone classes
// ("nelder_mead"/"brent", which do not derive from the Optimizer base); each vector it omits falls
// back to the top-level one, and an absent "inner" means BFGS over the top-level vectors, the
// shape every upstream C# test uses. NOTE that AugmentedLagrange::optimize() always drives the
// INNER optimizer through minimize(), whatever the outer request -- upstream behavior, mirrored,
// not corrected. It is also, MEASURED, a wrong answer reported as Success: the augmented
// Lagrangian is built from the RAW objective, so `"maximize": true` flips the outer bookkeeping
// and not the search direction, and the run returns the constrained MINIMUM. This runner keeps
// mirroring it (a fixture case must be able to pin upstream behavior), and the guard lives on the
// two PUBLIC verbs instead: R/optim.R's kOptimMinimizeOnlyMethods and optim.py's
// _MINIMIZE_ONLY_METHODS both reject `optim_maximize(method = "augmented_lagrange")` by name.
// Argument-shape validation beyond what the ported constructors
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
inline OptimResult run_optimizer(const std::string& spec_json, const OptimCallbacks& callbacks) {
    const Objective& objective = callbacks.objective;
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

    // The eleven Optimizer-subclass methods all take the same shape: build and configure the
    // class (detail::make_optimizer, shared with the "augmented_lagrange" arm's inner optimizer),
    // drive it through ITS OWN minimize()/maximize(), and read the base's accessor surface.
    // "nelder_mead" and "brent" are handled by their own arms below because neither derives from
    // that base; "augmented_lagrange" has its own arm because it takes a borrowed inner optimizer
    // and a set of constraint callbacks that no other method has.
    if (method == "adam" || method == "gradient_descent") {
        // The only two methods that take a SECOND host-language callback. Its guard shares the
        // objective guard's abort state so a throw in either callback stops both, and so the
        // single rethrow below covers both.
        int D = static_cast<int>(initial.size());
        GuardedGradient guarded_grad(callbacks.gradient, std::vector<double>{},
                                     guarded.abort_state());
        // The ported classes call the gradient unconditionally when the field is set, so the field
        // stays EMPTY unless the host supplied one -- an empty std::function is exactly what C#'s
        // null Gradient means, and both classes already branch on it.
        opt::ADAM::GradientFunction grad_fn = nullptr;
        if (callbacks.gradient) {
            grad_fn = [&guarded_grad, D](const std::vector<double>& p) {
                std::vector<double> g = guarded_grad(p);
                // See GuardedGradient's comment: an aborted call returns the empty sentinel, which
                // both classes would index out of range.
                if (guarded_grad.aborted()) g.assign(static_cast<std::size_t>(D), 0.0);
                if (static_cast<int>(g.size()) != D)
                    throw std::runtime_error(
                        "the gradient must return one value per parameter; got a value of length " +
                        std::to_string(static_cast<long long>(g.size())) + " for " +
                        std::to_string(static_cast<long long>(D)) + " parameters");
                return g;
            };
        }
        std::unique_ptr<opt::Optimizer> o =
            detail::make_optimizer(detail::read_build(spec), adapted, grad_fn);
        try {
            if (maximize) o->maximize(); else o->minimize();
        } catch (...) {
            guarded.rethrow_if_aborted();
            throw;
        }
        guarded.rethrow_if_aborted();
        detail::fill_optimizer_result(result, *o, maximize);
    } else if (method == "augmented_lagrange") {
        // The constrained arm. Three things distinguish it from every other method:
        //
        //   1. It BORROWS an inner optimizer, whose objective AugmentedLagrange's constructor
        //      replaces with the augmented Lagrangian (see augmented_lagrange.hpp's note 1). The
        //      inner optimizer therefore has to outlive the solver, which the local unique_ptr
        //      below gives it, and it is built by the same detail::make_optimizer the eleven
        //      unconstrained methods go through -- so an "inner" sub-spec may name any of them.
        //   2. It takes one host-language callback PER CONSTRAINT beside the objective. Each gets
        //      its own GuardedObjective (a Constraint's function has the same const-ref scalar
        //      shape as an objective) sharing the objective guard's abort state, so the first host
        //      exception raised in any of them short-circuits all the rest.
        //   3. Its result carries the three multiplier vectors, which nothing else has.
        //
        // The spec's `constraints[i]` object (type/value/tolerance) pairs POSITIONALLY with
        // `callbacks.constraints[i]`; the two halves are split by the caller (R/optim.R,
        // corehydropy/optim.py, the fixture runners) because one half is serializable and the
        // other is a live function. That pairing is the contract, so a length mismatch is an
        // error rather than a silent truncation.
        if (!spec.contains("constraints") || spec.at("constraints").items().empty())
            throw std::runtime_error(
                "optimizer 'augmented_lagrange' needs at least one entry in 'constraints'");
        const std::vector<JsonValue>& constraint_specs = spec.at("constraints").items();
        if (constraint_specs.size() != callbacks.constraints.size())
            throw std::runtime_error(
                "'constraints' carries " +
                std::to_string(static_cast<long long>(constraint_specs.size())) +
                " entries but " +
                std::to_string(static_cast<long long>(callbacks.constraints.size())) +
                " constraint function(s) were supplied; the two pair positionally");

        // The inner optimizer: the "inner" sub-spec when present, otherwise BFGS over the
        // top-level vectors (the shape every upstream C# test uses). An "inner" sub-spec that
        // omits a vector falls back to the top-level one, so naming only a method is enough.
        detail::OptimizerBuild inner_build;
        if (spec.contains("inner")) {
            inner_build = detail::read_build(spec.at("inner"));
            if (inner_build.lower.empty()) inner_build.lower = lower;
            if (inner_build.upper.empty()) inner_build.upper = upper;
            if (inner_build.initial.empty()) inner_build.initial = initial;
        } else {
            inner_build.method = "bfgs";
            inner_build.lower = lower;
            inner_build.upper = upper;
            inner_build.initial = initial;
        }
        // Rejected by name BEFORE construction: AugmentedLagrange's own constructor rejects a
        // nested one (its message names the argument, not the method), and the two standalone
        // classes cannot be an inner optimizer at all because they do not derive from the
        // Optimizer base -- make_optimizer returns null for them rather than throwing, so this is
        // the one place that difference can be explained to the caller.
        if (inner_build.method == "augmented_lagrange")
            throw std::runtime_error(
                "the inner optimizer cannot also be an augmented_lagrange optimizer");
        if (inner_build.method == "nelder_mead" || inner_build.method == "brent")
            throw std::runtime_error(
                "optimizer '" + inner_build.method +
                "' cannot be an augmented_lagrange inner optimizer: it does not derive from the "
                "ported Optimizer base");
        std::unique_ptr<opt::Optimizer> inner =
            detail::make_optimizer(inner_build, adapted, nullptr);

        // One guard per constraint, all sharing the objective guard's abort state. Held through
        // unique_ptr so the lambdas below can capture a pointer that stays valid however the
        // vector grows. The sentinel is NaN: unlike an objective there is no "worst" constraint
        // value a ported consumer treats as rejected, and NaN is the one double that cannot be
        // mistaken for a real constraint reading -- the run's result is discarded by the rethrow
        // below in any case.
        std::vector<std::unique_ptr<GuardedObjective>> constraint_guards;
        std::vector<std::shared_ptr<opt::IConstraint>> constraints;
        constraint_guards.reserve(constraint_specs.size());
        constraints.reserve(constraint_specs.size());
        for (std::size_t i = 0; i < constraint_specs.size(); ++i) {
            const JsonValue& cs = constraint_specs[i];
            opt::ConstraintType type = detail::parse_constraint_type(cs.at("type").as_string());
            double value = cs.at("value").as_double();
            double tolerance = cs.value_or("tolerance", 1E-8);
            constraint_guards.push_back(std::make_unique<GuardedObjective>(
                callbacks.constraints[i], std::numeric_limits<double>::quiet_NaN(),
                guarded.abort_state()));
            GuardedObjective* g = constraint_guards.back().get();
            constraints.push_back(std::make_shared<opt::Constraint>(
                [g](const std::vector<double>& x) { return (*g)(x); },
                inner->number_of_parameters(), value, type, tolerance));
        }

        opt::AugmentedLagrange solver(adapted, *inner, constraints);
        if (has_control) {
            detail::apply_common_controls(solver, control);
            detail::apply_optimizer_controls(solver, control);
        }
        try {
            if (maximize) solver.maximize(); else solver.minimize();
        } catch (...) {
            // More than one guard is live here, so the abort state is asked directly rather than
            // through whichever guard happens to be in scope -- see callback_guard.hpp's
            // rethrow_if_aborted(state) on why picking one guard to stand in for a group is a trap.
            support::rethrow_if_aborted(guarded.abort_state());
            throw;
        }
        support::rethrow_if_aborted(guarded.abort_state());
        detail::fill_optimizer_result(result, solver, maximize);
        result.lambda = solver.lambda();
        result.mu = solver.mu();
        result.nu = solver.nu();
    } else if (method != "nelder_mead" && method != "brent") {
        std::unique_ptr<opt::Optimizer> o =
            detail::make_optimizer(detail::read_build(spec), adapted, nullptr);
        try {
            if (maximize) o->maximize(); else o->minimize();
        } catch (...) {
            guarded.rethrow_if_aborted();
            throw;
        }
        guarded.rethrow_if_aborted();
        detail::fill_optimizer_result(result, *o, maximize);
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
    }

    return result;
}

// The objective-only form, kept so every caller that needs no second callback -- which is every
// method except "adam"/"gradient_descent" -- compiles and reads unchanged.
inline OptimResult run_optimizer(const std::string& spec_json, const Objective& objective) {
    return run_optimizer(spec_json, OptimCallbacks{objective, nullptr, {}});
}

}  // namespace corehydro::numerics::support
