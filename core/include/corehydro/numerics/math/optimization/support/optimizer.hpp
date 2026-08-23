// ported from: Numerics/Mathematics/Optimization/Support/Optimizer.cs @ 2a0357a
//
// Abstract base class for all optimization methods: inputs (tolerances, iteration/
// evaluation caps, ReportFailure/RecordTraces/ComputeHessian flags), outputs
// (Iterations, FunctionEvaluations, BestParameterSet, ParameterSetTrace, Status,
// Hessian), and the Evaluate/RepairParameter/UpdateStatus/CheckConvergence machinery
// every concrete optimizer (DifferentialEvolution here; NelderMead/BrentSearch/etc. in
// later phases) is built on.
//
// RELATIONSHIP TO nelder_mead.hpp / brent_search.hpp: Phase 0 shipped GEV's MLE fit
// against two standalone local optimizers that each fold a SUBSET of this base's
// best-tracking / bound-repair / convergence-check / minimize-maximize-sign-handling
// logic directly into their own file (see nelder_mead.hpp's header comment), because at
// the time the full Optimizer hierarchy didn't exist yet. That was a deliberate,
// documented shortcut, not an oversight -- this file is the FIRST port of the real base
// class, added now because DifferentialEvolution (and, via it, the MCMC MAP-init path)
// needs it. nelder_mead.hpp and brent_search.hpp are DELIBERATELY LEFT UNTOUCHED:
// refactoring them onto this base would touch code with oracle-locked Phase 0/1 fixtures
// (GEV MLE, and every distribution whose estimation path uses BrentSearch) for zero
// behavioral gain, and risks reintroducing the libm-ULP transcendental divergences noted
// in the Task 3 progress-ledger entry. A future cleanup task may fold them onto this base
// once there is a concrete reason to (e.g. a caller needing their Hessian/trace output);
// until then they intentionally stay as-is.
//
// Exception-handling fidelity: C# UpdateStatus() throws a plain `ArgumentException`
// (paramName = nameof(MaxIterations) or nameof(MaxFunctionEvaluations)) for the two
// "soft" statuses, or rethrows the ORIGINAL causing exception for OptimizationStatus.Failure
// -- and only `if (ReportFailure)` in either case. Minimize()/Maximize() wrap Optimize()
// in `catch (ArgumentException ex) { if (ex.ParamName != nameof(MaxIterations) &&
// ex.ParamName != nameof(MaxFunctionEvaluations)) UpdateStatus(Failure, ex); }` followed
// by a catch-all that always routes to `UpdateStatus(Failure, ex)`. Net effect (verified
// by tracing every path): hitting MaxIterations/MaxFunctionEvaluations NEVER causes
// Minimize()/Maximize() to throw, REGARDLESS of ReportFailure -- when ReportFailure is
// true the exception is thrown and then unconditionally swallowed by the ArgumentException
// filter above; when false, UpdateStatus never throws in the first place, so Optimize()
// just returns normally. Genuine failures (bad config, an objective-function exception, or
// any other unexpected exception) DO surface as a thrown exception from Minimize()/
// Maximize() when ReportFailure is true, and are swallowed (Status left at Failure, best-
// so-far BestParameterSet retained) when ReportFailure is false -- this is the "swallowed-
// failure path" the MCMC MAP-init call site (a later task) relies on: it sets
// `DE.ReportFailure = false` before `DE.Maximize()` so any optimizer hiccup falls back to
// whatever BestParameterSet was found so far instead of aborting the whole MCMC run.
//
// C++ has no .NET-style typed exception with a `ParamName`, so `ArgumentException` below
// carries an explicit `ArgumentErrorKind` tag standing in for that filter. Every throw
// site in this file and in differential_evolution.hpp that corresponds to a C#
// `ArgumentException`/`ArgumentOutOfRangeException`/`ArgumentNullException` uses this same
// type, so `minimize()`/`maximize()`'s `catch (const ArgumentException&)` clause catches
// them uniformly, exactly as C#'s `catch (ArgumentException ex)` catches all three
// subtypes.
//
// OBJECTIVE-FUNCTION SETTER: C# declares `ObjectiveFunction` as a public settable property
// (whose setter throws `ArgumentNullException` on null) rather than a constructor-only field,
// because AugmentedLagrange REPLACES the inner optimizer's objective with its own augmented
// Lagrangian after construction. `set_objective_function()` below is that setter; the
// constructor still assigns the backing member directly, exactly as the C# ctor assigns
// through the property before validating NumberOfParameters.
//
// minimize()/maximize(): kept as two separate methods with duplicated try/catch bodies
// (rather than factored into one shared private helper) to mirror the C# source
// structurally line-for-line, per this port's file/method-layout convention -- see
// CLAUDE.md's "Structural mirroring" note.
#pragma once
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/support/optimization_status.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"

namespace corehydro::numerics::math::optimization {

// Tag distinguishing which C# `ArgumentException`-family error a thrown
// `ArgumentException` represents, standing in for `Exception.ParamName` (see file header).
enum class ArgumentErrorKind { Other, MaxIterations, MaxFunctionEvaluations };

// Stand-in for C#'s `System.ArgumentException` (and the `ArgumentOutOfRangeException`/
// `ArgumentNullException` subtypes this module's validation throws) -- close enough for
// `Optimizer::minimize()`/`maximize()` to replicate the C# `catch (ArgumentException ex)`
// filter described in the file header.
class ArgumentException : public std::runtime_error {
   public:
    explicit ArgumentException(const std::string& message, ArgumentErrorKind kind = ArgumentErrorKind::Other)
        : std::runtime_error(message), kind_(kind) {}

    ArgumentErrorKind kind() const { return kind_; }

   private:
    ArgumentErrorKind kind_;
};

class Optimizer {
   public:
    // MUTABLE-POINT SEMANTICS (M14, C#-fidelity): C# objectives are `Func<double[], double>`
    // and .NET arrays are reference types, so an objective CAN write back into the point the
    // optimizer hands it -- and the real RMC.BestFit MixtureModel does exactly that
    // (`Mixture.SetParameters(ref parameters)` normalizes the weight entries in place during
    // every DataLogLikelihood/PriorLogLikelihood evaluation, re-projecting the optimizer's own
    // working vectors onto the normalized-weights manifold and thereby steering the search
    // path). The Objective signature is therefore a NON-CONST reference. Callables that take
    // `const std::vector<double>&` still convert to this std::function unchanged, so
    // non-mutating callers (every Phase 0-4 objective) are unaffected.
    using Objective = std::function<double(std::vector<double>&)>;

    // Construct a new optimization method. Validation order matches C# exactly: the ctor
    // assigns `ObjectiveFunction` (whose property setter null-checks) BEFORE validating
    // `NumberOfParameters`, so a caller passing both an empty objective and an invalid
    // parameter count observes the objective-function error, not the parameter-count one.
    Optimizer(Objective objective_function, int number_of_parameters)
        : objective_function_(std::move(objective_function)), number_of_parameters_(number_of_parameters) {
        if (!objective_function_) throw ArgumentException("The objective function cannot be null.");
        if (number_of_parameters < 1)
            throw ArgumentException("There must be at least 1 parameter to evaluate.");
    }

    virtual ~Optimizer() = default;

    // --- Inputs --------------------------------------------------------------------------

    // The maximum number of optimization iterations allowed. Default = 10,000.
    int max_iterations = 10000;

    // The maximum number of function evaluations allowed. Default = int max.
    int max_function_evaluations = std::numeric_limits<int>::max();

    // The desired absolute tolerance for the solution. Default = ~Sqrt(Machine Epsilon), or 1E-8.
    double absolute_tolerance = 1E-8;

    // The desired relative tolerance for the solution. Default = ~Sqrt(Machine Epsilon), or 1E-8.
    double relative_tolerance = 1E-8;

    // Determines if an exception will be thrown if the optimization solver fails to converge.
    bool report_failure = true;

    // Determines whether the parameter set traces are saved or not. Default = true.
    bool record_traces = true;

    // Determines whether to compute a numerically differentiated Hessian matrix when the
    // optimization was successful.
    bool compute_hessian = true;

    // The number of parameters to evaluate in the objective function.
    int number_of_parameters() const { return number_of_parameters_; }

    // The objective function to evaluate. C# exposes this as a public settable property whose
    // setter throws on null; AugmentedLagrange REPLACES the inner optimizer's objective with its
    // own augmented Lagrangian through it, which is why it is public rather than protected.
    const Objective& objective_function() const { return objective_function_; }
    void set_objective_function(Objective value) {
        if (!value) throw ArgumentException("The objective function cannot be null.");
        objective_function_ = std::move(value);
    }

    // --- Outputs -------------------------------------------------------------------------

    // Returns the number of iterations required to find the solution.
    int iterations() const { return iterations_; }

    // Returns the number of function evaluations required to find the solution.
    int function_evaluations() const { return function_evaluations_; }

    // The optimal point, or parameter set.
    const ParameterSet& best_parameter_set() const { return best_parameter_set_; }

    // A trace of the parameter set and fitness evaluated until convergence.
    const std::vector<ParameterSet>& parameter_set_trace() const { return parameter_set_trace_; }

    // Determines the optimization method status.
    OptimizationStatus status() const { return status_; }

    // The numerically differentiated Hessian matrix. Only populated when the optimization
    // was successful (mirrors C#'s nullable `Matrix? Hessian`).
    const std::optional<linalg::Matrix>& hessian() const { return hessian_; }

    // Clears the results.
    virtual void clear_results() {
        iterations_ = 0;
        function_evaluations_ = 0;
        best_parameter_set_ = ParameterSet();
        parameter_set_trace_.clear();
        status_ = OptimizationStatus::None;
        hessian_.reset();
    }

    // Finds the parameter set that minimizes the objective function.
    virtual void minimize() {
        validate();
        clear_results();
        function_scale_ = 1;
        try {
            optimize();
            if (status_ == OptimizationStatus::Success && compute_hessian) {
                // By-value copy: the differentiation helper's point is const, but Objective
                // takes a mutable reference (see the Objective note above).
                hessian_ = linalg::Matrix(differentiation::hessian(
                    [this](std::vector<double> x) { return objective_function_(x); },
                    best_parameter_set_.values));
            }
        } catch (const ArgumentException& ex) {
            if (ex.kind() != ArgumentErrorKind::MaxIterations && ex.kind() != ArgumentErrorKind::MaxFunctionEvaluations)
                update_status(OptimizationStatus::Failure, std::current_exception());
        } catch (...) {
            update_status(OptimizationStatus::Failure, std::current_exception());
        }
    }

    // Finds the parameter set that maximizes the objective function.
    virtual void maximize() {
        validate();
        clear_results();
        function_scale_ = -1;
        try {
            optimize();
            if (status_ == OptimizationStatus::Success && compute_hessian) {
                // By-value copy: the differentiation helper's point is const, but Objective
                // takes a mutable reference (see the Objective note above).
                hessian_ = linalg::Matrix(differentiation::hessian(
                    [this](std::vector<double> x) { return objective_function_(x); },
                    best_parameter_set_.values));
            }
        } catch (const ArgumentException& ex) {
            if (ex.kind() != ArgumentErrorKind::MaxIterations && ex.kind() != ArgumentErrorKind::MaxFunctionEvaluations)
                update_status(OptimizationStatus::Failure, std::current_exception());
        } catch (...) {
            update_status(OptimizationStatus::Failure, std::current_exception());
        }
    }

   protected:
    Objective objective_function_;
    int number_of_parameters_;

    // Objective function scaling factor (set to -1 for a maximization problem). By
    // default it is a minimization problem.
    int function_scale_ = 1;

    int iterations_ = 0;
    int function_evaluations_ = 0;
    ParameterSet best_parameter_set_;
    std::vector<ParameterSet> parameter_set_trace_;
    OptimizationStatus status_ = OptimizationStatus::None;
    std::optional<linalg::Matrix> hessian_;

    // Validate inputs.
    virtual void validate() {
        if (max_iterations < 10)
            throw ArgumentException("The maximum number of optimization iterations must be greater than 10.");
        if (max_function_evaluations < 10)
            throw ArgumentException("The maximum number of function evaluations must be greater than 10.");
        if (relative_tolerance <= 0 || relative_tolerance > 1)
            throw ArgumentException("The relative tolerance must be between 0 and 1.");
        if (absolute_tolerance <= 0 || absolute_tolerance > 1)
            throw ArgumentException("The absolute tolerance must be between 0 and 1.");
    }

    // Implements the actual optimization algorithm. This method should minimize the
    // objective function.
    virtual void optimize() = 0;

    // Evaluates the objective function and returns the fitness. `values` is a non-const
    // reference (see the Objective note above): a mutating objective's write-back lands in the
    // caller's own working vector, and the best-parameter-set copy below is taken AFTER the
    // call -- both exactly like the C# base (`BestParameterSet = new ParameterSet(
    // (double[])values.Clone(), fitness)` after `ObjectiveFunction(values)`).
    virtual double evaluate(std::vector<double>& values, bool& cancel) {
        double fitness = static_cast<double>(function_scale_) * objective_function_(values);

        // Keep track of the best fit parameter set. `values.empty()` stands in for C#'s
        // `BestParameterSet.Values == null` (see parameter_set.hpp's file header).
        if (best_parameter_set_.values.empty() || fitness <= best_parameter_set_.fitness) {
            best_parameter_set_ = ParameterSet(values, fitness);
        }

        // Update trace. This is tracked every evaluation.
        if (record_traces) parameter_set_trace_.push_back(best_parameter_set_.clone());

        // Update evaluation counter.
        function_evaluations_ += 1;
        if (function_evaluations_ >= max_function_evaluations) {
            cancel = true;
            update_status(OptimizationStatus::MaximumFunctionEvaluationsReached);
        }

        return fitness;
    }

    // Repair the trial parameter to keep it within bounds.
    virtual double repair_parameter(double value, double lower_bound, double upper_bound) const {
        if (value < lower_bound) value = lower_bound;
        if (value > upper_bound) value = upper_bound;
        return value;
    }

    // Update the optimization status. Exceptions will be thrown depending on the status
    // (see the file header for the full ReportFailure/swallowed-exception analysis).
    virtual void update_status(OptimizationStatus status, std::exception_ptr exception = nullptr) {
        status_ = status;
        if (status == OptimizationStatus::MaximumIterationsReached) {
            if (report_failure)
                throw ArgumentException("The maximum number of iterations has been reached.",
                                         ArgumentErrorKind::MaxIterations);
        } else if (status == OptimizationStatus::MaximumFunctionEvaluationsReached) {
            if (report_failure)
                throw ArgumentException("The maximum number of function evaluations has been reached.",
                                         ArgumentErrorKind::MaxFunctionEvaluations);
        } else if (status == OptimizationStatus::Failure) {
            if (report_failure && exception) std::rethrow_exception(exception);
        }
    }

    // Checks convergence.
    virtual bool check_convergence(double old_value, double new_value) const {
        if (std::isnan(old_value) || std::isnan(new_value) || std::isinf(old_value) || std::isinf(new_value))
            return false;
        return 2.0 * std::fabs(new_value - old_value) /
                   (std::fabs(new_value) + std::fabs(old_value) + absolute_tolerance) <
               relative_tolerance;
    }
};

}  // namespace corehydro::numerics::math::optimization
