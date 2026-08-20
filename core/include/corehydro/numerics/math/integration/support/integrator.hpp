// ported from: Numerics/Mathematics/Integration/Support/Integrator.cs @ 2a0357a
//
// Abstract base class for all integration methods: inputs (iteration and function-evaluation
// bounds, absolute/relative tolerance, the ReportFailure flag), outputs (Iterations,
// FunctionEvaluations, Result, Status), and the ClearResults / Validate / UpdateStatus /
// EvaluateConvergence machinery every concrete integrator is built on. Sibling of
// optimization/support/optimizer.hpp, whose C# counterpart it mirrors closely -- the two base
// classes were written to the same shape upstream, and this port keeps that symmetry.
//
// AdaptiveGaussKronrod is the only concrete integrator ported so far (the C# Integration folder
// also holds AdaptiveGaussLobatto, AdaptiveSimpsonsRule, AdaptiveSimpsonsRule2D, TrapezoidalRule,
// SimpsonsRule, MonteCarloIntegration, Miser and Vegas; each is a separate upstream file and none
// has a caller here yet). Add them beside adaptive_gauss_kronrod.hpp when a caller arrives.
//
// EXCEPTION MAPPING, repo-wide convention: C# ArgumentOutOfRangeException -> std::out_of_range.
// Validate() throws one per violated bound, in the C# order.
//
// UpdateStatus(): C# takes an optional inner Exception and rethrows it when the status is Failure
// and ReportFailure is set. C++ has no ambient exception object, so the parameter is an
// std::exception_ptr (the same stand-in optimizer.hpp uses), rethrown with std::rethrow_exception.
// AdaptiveGaussKronrod does not route through UpdateStatus -- it assigns Status directly and
// rethrows inside its own catch, exactly as the C# class does -- so the method exists here for
// fidelity and for the integrators still to be ported.
#pragma once
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

#include "corehydro/numerics/math/integration/support/integration_status.hpp"

namespace corehydro::numerics::math::integration {

class Integrator {
   public:
    virtual ~Integrator() = default;

    // --- Inputs ----------------------------------------------------------------------------

    // The minimum number of integration iterations allowed. Default = 1.
    int min_iterations = 1;

    // The maximum number of integration iterations allowed. Default = 1E7.
    int max_iterations = 10000000;

    // The minimum number of function evaluations allowed. Default = 1.
    int min_function_evaluations = 1;

    // The maximum number of function evaluations allowed. Default = 1E7.
    int max_function_evaluations = 10000000;

    // The desired absolute tolerance for the solution. Default = ~Sqrt(Machine Epsilon), or 1E-8.
    double absolute_tolerance = 1E-8;

    // The desired relative tolerance for the solution. Default = ~Sqrt(Machine Epsilon), or 1E-8.
    double relative_tolerance = 1E-8;

    // Determines if an exception will be thrown if the integration method fails to converge.
    bool report_failure = true;

    // --- Output ----------------------------------------------------------------------------

    // Returns the number of iterations required to find the solution.
    int iterations() const { return iterations_; }

    // Returns the number of function evaluations required to find the solution.
    int function_evaluations() const { return function_evaluations_; }

    // The numerically computed result of the definite integral. C# exposes a public setter over
    // the protected `_result` field, so this port keeps a public writer beside the reader.
    double result() const { return result_; }
    void set_result(double value) { result_ = value; }

    // Determines the integration method status.
    IntegrationStatus status() const { return status_; }

    // --- Methods ---------------------------------------------------------------------------

    // Evaluates the integral.
    virtual void integrate() = 0;

   protected:
    // Protected result property.
    double result_ = 0.0;

    int iterations_ = 0;
    int function_evaluations_ = 0;
    IntegrationStatus status_ = IntegrationStatus::None;

    // Clears the results.
    virtual void clear_results() {
        iterations_ = 0;
        function_evaluations_ = 0;
        result_ = std::numeric_limits<double>::quiet_NaN();
        status_ = IntegrationStatus::None;
    }

    // Validate inputs.
    virtual void validate() {
        if (min_iterations < 1)
            throw std::out_of_range(
                "The minimum number of iterations must be greater than or equal to 1.");
        if (min_function_evaluations < 1)
            throw std::out_of_range(
                "The minimum number of function evaluations must be greater than or equal to 1.");
        if (max_iterations < 1)
            throw std::out_of_range("The maximum number of iterations must be greater than 1.");
        if (max_function_evaluations < 1)
            throw std::out_of_range(
                "The maximum number of function evaluations must be greater than 1.");
        if (relative_tolerance < 1E-15 || relative_tolerance > 1)
            throw std::out_of_range("The relative tolerance must be between 1E-15 and 1.");
        if (absolute_tolerance < 1E-15 || absolute_tolerance > 1)
            throw std::out_of_range("The absolute tolerance must be between 1E-15 and 1.");
    }

    // Update the integration status. Exceptions will be thrown depending on the status.
    virtual void update_status(IntegrationStatus status, std::exception_ptr exception = nullptr) {
        status_ = status;
        if (status == IntegrationStatus::Failure) {
            if (report_failure && exception) std::rethrow_exception(exception);
        }
    }

    // Evaluate convergence. Returns true for convergence.
    virtual bool evaluate_convergence(double old_value, double new_value) const {
        if (std::isnan(old_value) || std::isnan(new_value) || std::isinf(old_value) ||
            std::isinf(new_value))
            return false;
        return std::fabs(new_value - old_value) < absolute_tolerance &&
               std::fabs(new_value - old_value) / std::fabs(new_value) < relative_tolerance;
    }
};

}  // namespace corehydro::numerics::math::integration
