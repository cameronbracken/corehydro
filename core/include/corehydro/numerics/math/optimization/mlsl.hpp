// ported from: Numerics/Mathematics/Optimization/Global/MLSL.cs @ 2a0357a
//
// The Multi-Level Single Linkage (MLSL) global optimization method (Kan, Boender &
// Timmer 1985). Each iteration draws N uniform sample points over the bounded search
// region (seeded MersenneTwister -> inverse-CDF of one Uniform per dimension, so the
// whole run is deterministic for a given PRNGSeed), sorts the accumulated sample by
// fitness, reduces it to the best ceil(Gamma*k*N) points, spawns a local optimizer from
// every reduced point that is not within the critical distance rk of a better sample
// point or known local minimum, and stops on a Bayesian estimate of the number of local
// minima combined with Min/MaxNoImprovement counters. A REAL `Optimizer` subclass
// deriving from the ported base exactly as bfgs.hpp / powell.hpp do.
//
// Transcription notes:
//
// 1. Reference semantics. C#'s `List<SamplePoint>` holds references: the reduced sample
//    Rk shares the SAME SamplePoint objects as SampledPoints (so `point.Minimized = true`
//    marks the master list too) and the `point == other` skip is reference equality.
//    Ported as vectors of `std::shared_ptr<SamplePoint>`; shared_ptr copies preserve
//    identity-sharing and pointer equality mirrors C# reference equality.
//
// 2. Sort. C# `List.Sort((x, y) => x.Fitness.CompareTo(y.Fitness))`. `double.CompareTo`
//    is a total order with NaN below every number and equal to itself; the comparator
//    below reproduces that (std::sort with a bare `<` would be UB if a NaN fitness ever
//    appeared). Both List.Sort and std::sort are unstable, but not the same algorithm,
//    so exactly-tied fitness values could order differently -- unobservable in practice
//    (fitness ties across continuous random draws) and untestable upstream for the same
//    reason.
//
// 3. In-place repair. C#'s GetLocalOptimizer takes `IList<double> initialValues` and
//    repairs it IN PLACE -- mutating the caller's array (a stored SamplePoint's values,
//    or BestParameterSet.Values in the Polish step). get_local_optimizer therefore takes
//    `std::vector<double>&` and repairs in place too, and its call sites pass the same
//    owned vectors the C# passes (the child ctor then copies, as C# `ToArray()` does).
//
// 4. localCancel closure. C# captures `localCancel` in the child-objective lambda and
//    computes `cancel = cancel || localCancel` BEFORE the child ever runs (so a child-
//    run cancellation never propagates into MLSL's own `cancel` -- a faithful quirk,
//    kept). The heap closure is mirrored with a shared_ptr<bool> so the flag outlives
//    get_local_optimizer's frame. Budget exhaustion still stops the whole run the same
//    way as C#: the parent's evaluate() throws through the child (swallowed by the
//    child's minimize() catch filter), and the next parent-side evaluation re-throws out
//    of optimize() into the parent's own minimize() filter.
//
// 5. NelderMead branch. The C# constructs the real NelderMead Optimizer subclass. In
//    this port NelderMead is the deliberately-standalone Phase 0 class (see
//    nelder_mead.hpp's header -- oracle-locked, not refactored onto the base), so the
//    branch wraps it in the private NelderMeadLocalSolver below: a real Optimizer
//    subclass whose optimize() drives the standalone simplex loop through the BASE's
//    evaluate(), so best-tracking, evaluation counting, the MaxFunctionEvaluations
//    budget, and the throw-through-cancel cascade all match the real C# subclass. Two
//    documented residual deviations: (a) it always reports Success after the simplex
//    loop returns (the standalone class does not expose converged-vs-max-iterations;
//    same standing limitation as estimation/support/optimizer_adapters.hpp, which this
//    class deliberately does NOT reuse -- that header lives in the higher `estimation`
//    layer, and numerics must not depend on it); and (b) with report_failure = false a
//    budget-exhausted child keeps iterating to its own max_iterations instead of
//    returning early (the standalone loop has no cancel flag to check). MLSL leaves
//    child report_failure at the default true, so (b) is unreachable from this file.
//
// 6. SamplePoint. C# declares only the parameterless ctor and sets ParameterSet /
//    Minimized via object initializers; the two-argument ctor below stands in for that
//    initializer syntax at the three construction sites.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/math/optimization/bfgs.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/optimization/support/nelder_mead_solver.hpp"
#include "corehydro/numerics/math/optimization/powell.hpp"
#include "corehydro/numerics/math/optimization/support/local_method.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::optimization {

class MLSL : public Optimizer {
   public:
    // Class for storing sampled points (C# nested `SamplePoint`; see note 6).
    class SamplePoint {
       public:
        // The sample point parameter set.
        ParameterSet parameter_set;

        // Determines if the sample point has already been minimized
        bool minimized = false;

        // Create new sample point
        SamplePoint() : parameter_set() {}

        // Object-initializer stand-in (see note 6).
        SamplePoint(ParameterSet parameter_set, bool minimized)
            : parameter_set(std::move(parameter_set)), minimized(minimized) {}
    };

    // Constructs a new multi-level single linkage (MLSL) optimization method.
    //   objective_function:   the objective function to evaluate.
    //   number_of_parameters: the number of parameters in the objective function.
    //   initial_values:       an array of initial values to evaluate.
    //   lower_bounds:         an array of lower bounds (inclusive) of the interval
    //                         containing the optimal point.
    //   upper_bounds:         an array of upper bounds (inclusive) of the interval
    //                         containing the optimal point.
    //   method:               the local search method to use. Default = BFGS.
    MLSL(Objective objective_function, int number_of_parameters,
         std::vector<double> initial_values, std::vector<double> lower_bounds,
         std::vector<double> upper_bounds, LocalMethod method = LocalMethod::BFGS)
        : Optimizer(std::move(objective_function), number_of_parameters) {
        // Check if the length of the initial, lower and upper bounds equal the number of
        // parameters. (C# ArgumentOutOfRangeException -> the ArgumentException stand-in
        // documented in optimizer.hpp's file header, as in bfgs.hpp / powell.hpp.)
        if (static_cast<int>(initial_values.size()) != number_of_parameters ||
            static_cast<int>(lower_bounds.size()) != number_of_parameters ||
            static_cast<int>(upper_bounds.size()) != number_of_parameters) {
            throw ArgumentException(
                "The initial values and lower and upper bounds must be the same length as the "
                "number of parameters.");
        }
        // Check if the initial values are between the lower and upper values.
        for (std::size_t j = 0; j < initial_values.size(); j++) {
            if (upper_bounds[j] < lower_bounds[j]) {
                throw ArgumentException("The upper bound cannot be less than the lower bound.");
            }
            if (initial_values[j] < lower_bounds[j] || initial_values[j] > upper_bounds[j]) {
                throw ArgumentException(
                    "The initial values must be between the upper and lower bounds.");
            }
        }
        initial_values_ = std::move(initial_values);
        lower_bounds_ = std::move(lower_bounds);
        upper_bounds_ = std::move(upper_bounds);
        this->method = method;  // the parameter shadows the member, as `Method = method` in C#
    }

    // An array of initial values to evaluate.
    const std::vector<double>& initial_values() const { return initial_values_; }

    // An array of lower bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }

    // An array of upper bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

    // The pseudo random number generator (PRNG) seed.
    int prng_seed = 12345;

    // The local search method to use. Default = BFGS.
    LocalMethod method = LocalMethod::BFGS;

    // The desired absolute tolerance for the local solution. Default = ~Sqrt(1E-16), or 1E-8.
    double local_absolute_tolerance = 1E-8;

    // The desired relative tolerance for the local solution. Default = ~Sqrt(1E-16), or 1E-8.
    double local_relative_tolerance = 1E-8;

    // If true (default), then a final local search is used to polish the best population
    // member at the end, which can improve the optimization slightly.
    bool polish = true;

    // The number of random samples to evaluate each iteration. Default = 30
    int sample_size = 30;

    // Determines the reduced sample size. Must be between 0 and 1. Default = 0.05.
    double gamma = 0.05;

    // The scale parameter for determining the critical distance. Default = 2.0.
    double sigma = 2.0;

    // The list of all sampled points.
    const std::vector<std::shared_ptr<SamplePoint>>& sampled_points() const {
        return sampled_points_;
    }

    // The list of all local optimums.
    const std::vector<std::shared_ptr<SamplePoint>>& local_minimums() const {
        return local_minimums_;
    }

    // The minimum number of iterations to carry out with no improvement. Default = 5.
    int min_no_improvement = 5;

    // The maximum number of iterations to carry out with no improvement. Default = 10.
    int max_no_improvement = 10;

   protected:
    void optimize() override {
        if (sample_size < 4)
            throw ArgumentException("The sample size must be greater than or equal to 4.");
        if (gamma <= 0 || gamma >= 1)
            throw ArgumentException("The reduction parameter must be between 0 and 1.");
        if (sigma <= 0) throw ArgumentException("The scale parameter must be positive.");
        if (min_no_improvement < 1)
            throw ArgumentException(
                "The minimum number of iterations to carry out with no improvement must be "
                "greater than 0.");
        if (max_no_improvement < 1)
            throw ArgumentException(
                "The maximum number of iterations to carry out with no improvement must be "
                "greater than 0.");
        if (max_no_improvement < min_no_improvement)
            throw ArgumentException(
                "The maximum number of iterations must be greater than or equal to the minimum "
                "number.");

        int D = number_of_parameters_;
        int N = sample_size;
        double old_fit = std::numeric_limits<double>::max();
        int no_improvement = 0;
        bool cancel = false;
        std::unique_ptr<Optimizer> solver;
        auto prng = sampling::MersenneTwister(static_cast<std::uint32_t>(prng_seed));

        // Set lower and upper bounds and
        // create uniform distributions for each parameter
        std::vector<distributions::Uniform> uniform_dists;
        for (int i = 0; i < D; i++)
            uniform_dists.emplace_back(lower_bounds_[i], upper_bounds_[i]);

        // Critical distance
        double rkfactor = std::sqrt(kPi) * std::pow(special::function(D) * sigma, 1.0 / D);
        for (int i = 0; i < D; i++)
            rkfactor *= std::pow(upper_bounds_[i] - lower_bounds_[i], 1.0 / D);

        // Create lists for sample points
        sampled_points_.clear();
        local_minimums_.clear();

        while (iterations_ < max_iterations) {
            // Step 1. Generate sample points and function values
            // Add N points, drawn from a uniform distribution over S, to the (initially
            // empty) set of sample points, and evaluate f(x) at each new sample point.

            if (iterations_ == 0) {
                // On the first iteration, add the user-defined initial starting points
                // This can often be very close to the true minimum
                double init_fitness = evaluate(initial_values_, cancel);
                sampled_points_.push_back(std::make_shared<SamplePoint>(
                    ParameterSet(initial_values_, init_fitness), true));

                // Perform local minimizations from initial values
                solver = get_local_optimizer(initial_values_, local_relative_tolerance,
                                             local_absolute_tolerance, cancel);
                solver->minimize();
                if (cancel) return;

                local_minimums_.push_back(std::make_shared<SamplePoint>(
                    solver->best_parameter_set().clone(), true));
            }

            // Sample a point from the uniform distribution over S.
            for (int i = 0; i < (iterations_ == 0 ? N - 1 : N); i++) {
                std::vector<double> values(static_cast<std::size_t>(D));
                for (int j = 0; j < D; j++)
                    values[j] = uniform_dists[j].inverse_cdf(prng.next_double());

                double fitness = evaluate(values, cancel);
                if (cancel) return;
                sampled_points_.push_back(
                    std::make_shared<SamplePoint>(ParameterSet(values, fitness), false));
            }

            // Step 2. Reduce the sample points
            // Sort the entire sample of kN points in order of increasing object function
            // values. Select the gamma*kN points with the lowest objective function
            // values. This resultant set, Rk, is called the reduced sample.

            std::sort(sampled_points_.begin(), sampled_points_.end(),
                      [](const std::shared_ptr<SamplePoint>& x,
                         const std::shared_ptr<SamplePoint>& y) {
                          return compare_to(x->parameter_set.fitness,
                                            y->parameter_set.fitness) < 0;
                      });
            int gkN = static_cast<int>(std::ceil(gamma * (iterations_ + 1) * N));
            std::vector<std::shared_ptr<SamplePoint>> Rk(
                sampled_points_.begin(),
                sampled_points_.begin() +
                    std::min<std::size_t>(static_cast<std::size_t>(gkN),
                                          sampled_points_.size()));

            // Step 3. Select start points for local searches
            // Determine a (possibly empty) subset of the sample points from which to
            // start local searches.
            //
            // At iteration k, each sample point x is selected as a start point for a
            // local minimization if it has:
            // -> not been used as a start point at a previous iteration
            // -> there is no sample point y within the critical distance rk of x with a
            //    lower function value: ||x - y|| <= rk and also f(x) < f(y)

            double rk = rkfactor *
                        std::pow(std::log(static_cast<double>((iterations_ + 1) * N)) /
                                     ((iterations_ + 1) * N),
                                 1.0 / D);

            for (auto& point : Rk) {
                // Check if local search has already been performed
                if (point->minimized) continue;

                bool can_minimize = true;

                // Check if there is any other point y in Rk such that ||xi - y|| <= rk
                // and also f(x) < f(y)
                for (auto& other : Rk) {
                    if (point == other) continue;
                    if (normalized_distance(point->parameter_set.values,
                                            other->parameter_set.values, lower_bounds_,
                                            upper_bounds_) <= rk &&
                        other->parameter_set.fitness < point->parameter_set.fitness) {
                        can_minimize = false;
                        break;
                    }
                }

                // Check if there is any other local minimum y such that ||xi - y|| <= rk
                // and also f(x) < f(y)
                if (can_minimize) {
                    for (auto& local_min : local_minimums_) {
                        if (normalized_distance(point->parameter_set.values,
                                                local_min->parameter_set.values, lower_bounds_,
                                                upper_bounds_) <= rk &&
                            local_min->parameter_set.fitness < point->parameter_set.fitness) {
                            can_minimize = false;
                            break;
                        }
                    }
                }

                // Check if point is a potential minimizer
                if (can_minimize) {
                    // Perform local minimizations from all start points
                    solver = get_local_optimizer(point->parameter_set.values,
                                                 local_relative_tolerance,
                                                 local_absolute_tolerance, cancel);
                    solver->minimize();
                    if (cancel) return;

                    point->minimized = true;
                    local_minimums_.push_back(std::make_shared<SamplePoint>(
                        solver->best_parameter_set().clone(), true));
                }
            }

            // Step 4. Decide whether to stop
            // If stopping rule is satisfied, regard the lowest local minimizer as the
            // global minimizer, otherwise go to step 1.
            double w = static_cast<double>(local_minimums_.size());
            double s = static_cast<double>(sampled_points_.size());
            double B1 = (w * (s - 1.0)) / (s - w - 2.0);
            double B2 = (s - w - 1.0) * (s + w) / (s * (s - 1.0));

            if (iterations_ >= 1 && w >= 1 && old_fit == best_parameter_set_.fitness) {
                no_improvement++;
            } else {
                no_improvement = 0;
                old_fit = best_parameter_set_.fitness;
            }

            // And the Bayesian stopping rule is satisfied
            if (no_improvement >= max_no_improvement ||
                (no_improvement >= min_no_improvement && w >= 1 && B1 - w < 0.5 &&
                 B2 >= 0.995)) {
                // Polish the final result
                if (polish) {
                    solver = get_local_optimizer(best_parameter_set_.values, relative_tolerance,
                                                 absolute_tolerance, cancel);
                    solver->minimize();
                    update_status(solver->status());
                    return;
                }

                // Stop and return
                update_status(OptimizationStatus::Success);
                return;
            }

            iterations_ += 1;
        }

        // If we made it to here, the maximum iterations were reached.
        update_status(OptimizationStatus::MaximumIterationsReached);
    }

   private:
    std::vector<double> initial_values_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    std::vector<std::shared_ptr<SamplePoint>> sampled_points_;
    std::vector<std::shared_ptr<SamplePoint>> local_minimums_;

    // C# double.CompareTo total order: NaN sorts below every number and equal to itself
    // (see note 2). Returns <0, 0, >0.
    static int compare_to(double x, double y) {
        if (x < y) return -1;
        if (x > y) return 1;
        if (x == y) return 0;
        bool xnan = std::isnan(x), ynan = std::isnan(y);
        if (xnan && ynan) return 0;
        return xnan ? -1 : 1;
    }

    // The NelderMead branch's Optimizer adapter (see note 5) lives in
    // support/nelder_mead_solver.hpp: it was written here first as a private nested class and
    // promoted to a shared header when GeneralizedLinearModel needed the same adapter for the
    // same reason. Behavior is unchanged -- it is the same code, still routing every simplex
    // evaluation through the base evaluate().
    using NelderMeadLocalSolver = NelderMeadSolver;

    // Returns an optimizer for the local search.
    //   initial_values:     an array of initial values to evaluate (repaired IN PLACE,
    //                       mutating the caller's vector exactly as the C# repairs the
    //                       caller's IList -- see note 3).
    //   relative_tolerance: the desired relative tolerance for the solution.
    //   absolute_tolerance: the desired absolute tolerance for the solution.
    //   cancel:             by ref. Determines if the solver should be canceled.
    std::unique_ptr<Optimizer> get_local_optimizer(std::vector<double>& initial_values,
                                                   double relative_tolerance,
                                                   double absolute_tolerance, bool& cancel) {
        // Heap closure standing in for the C# captured local `localCancel` (see note 4).
        auto local_cancel = std::make_shared<bool>(false);
        std::unique_ptr<Optimizer> solver;

        // Make sure the parameters are within the bounds.
        for (int i = 0; i < number_of_parameters_; i++)
            initial_values[i] =
                repair_parameter(initial_values[i], lower_bounds_[i], upper_bounds_[i]);

        if (method == LocalMethod::BFGS) {
            solver = std::make_unique<BFGS>(
                [this, local_cancel](std::vector<double>& x) { return evaluate(x, *local_cancel); },
                number_of_parameters_, initial_values, lower_bounds_, upper_bounds_);
        } else if (method == LocalMethod::NelderMead) {
            solver = std::make_unique<NelderMeadLocalSolver>(
                [this, local_cancel](std::vector<double>& x) { return evaluate(x, *local_cancel); },
                number_of_parameters_, initial_values, lower_bounds_, upper_bounds_);
        } else if (method == LocalMethod::Powell) {
            solver = std::make_unique<Powell>(
                [this, local_cancel](std::vector<double>& x) { return evaluate(x, *local_cancel); },
                number_of_parameters_, initial_values, lower_bounds_, upper_bounds_);
        } else {
            // C# NotSupportedException; a plain std::runtime_error routes through the
            // base minimize()/maximize() catch-all to Failure, exactly like the C#
            // catch-all handles NotSupportedException.
            throw std::runtime_error("Unsupported local method");
        }
        solver->relative_tolerance = relative_tolerance;
        solver->absolute_tolerance = absolute_tolerance;
        solver->max_function_evaluations = max_function_evaluations - function_evaluations_;
        cancel = cancel || *local_cancel;
        return solver;
    }
};

}  // namespace corehydro::numerics::math::optimization
