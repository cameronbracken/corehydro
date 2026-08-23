// ported from: Numerics/Mathematics/Optimization/Global/SimulatedAnnealing.cs @ 2a0357a
//
// The adaptive simulated annealing global optimizer (Corana et al., "Minimizing Multimodal
// Functions of Continuous Variables with 'Simulated Annealing' Algorithm", 1987). Each outer
// iteration cools the temperature once, then runs TemperatureCycles x UpdateCycles sweeps of
// single-coordinate random moves accepted by the Metropolis criterion, adapting the per-coordinate
// step size toward a 50% acceptance rate after every temperature cycle.
//
// Transcription hazards:
//
// 1. THIS ALGORITHM NEVER CONVERGES EARLY. There is no CheckConvergence call anywhere in
//    Optimize(); the only exit is the terminal UpdateStatus(MaximumIterationsReached), reached
//    after MaxIterations outer iterations. `ToleranceSteps` ("the number of successive temperature
//    reductions to test for termination") is declared, documented, and VALIDATED -- and then never
//    read by the algorithm. Both facts are upstream as written; neither is "fixed" here. A
//    consequence worth knowing: a run always reports MaximumIterationsReached, so it always burns
//    its full iteration budget, and with report_failure = true the terminal update_status() throws
//    the MaxIterations ArgumentException that minimize()/maximize() then swallow (see
//    optimizer.hpp's file header for that whole path).
//
// 2. The out-of-bounds `continue` skips the proposal WITHOUT evaluating it -- so a rejected-by-
//    bounds move costs no function evaluation and, critically, consumes no further PRNG draw for
//    that coordinate. Moving the bounds check after the evaluation, or repairing the parameter
//    instead of skipping it, would change both the evaluation count and the random stream.
//
// 3. The Metropolis test short-circuits: `df < 0 || prng.NextDouble() < Math.Exp(-df / T)`. An
//    improving move therefore does NOT consume a uniform draw. C++'s `||` short-circuits
//    identically, so the expression is transcribed verbatim rather than split into statements.
//
// 4. At the end of every outer iteration the current point is RESET to the best point found so
//    far (`x = new Vector(BestParameterSet.Values); fx = BestParameterSet.Fitness;`), which is what
//    makes this "adaptive" rather than a plain annealing walk. The reset happens after the
//    iteration counter increments, so it also applies on the final iteration. NO TEST GUARDS THIS
//    LINE -- deleting it leaves the whole test_global_optimizers suite green, because the reset
//    only changes anything once an uphill move has been accepted, and that requires the
//    probabilistic exp() branch, which no deterministic assertion can pin across platforms. See
//    the SA block in that file's SUPPLEMENT section for the measurement and the reasoning. Do not
//    "simplify" it away.
//
// 5. Corana step-size update. `rate = (double)acceptances[j] / UpdateCycles` deliberately divides
//    by UpdateCycles and not by UpdateCycles * D, so the "rate" can exceed 1; the acceptance
//    counters are then zeroed (`acceptances.Fill(0)`, a Numerics array extension -> std::fill)
//    after every temperature cycle, not after every update cycle. Both are preserved as written.
//
// 6. Vector clone semantics. C#'s `xp.Clone()` and `new Vector(BestParameterSet.Values)` both
//    produce copies; the ported linalg::Vector has value semantics, so plain assignment is the
//    exact equivalent and no clone() method is needed. `Evaluate(x.ToArray(), ...)` passes a COPY
//    in C# too, so a mutating objective's write-back is discarded there as here -- but the base's
//    evaluate() takes a non-const reference (see optimizer.hpp's Objective note), so the copy has
//    to be a named local rather than a temporary.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::math::optimization {

class SimulatedAnnealing : public Optimizer {
   public:
    // Construct a new simulated annealing optimization method.
    //   objective_function:   the objective function to evaluate.
    //   number_of_parameters: the number of parameters in the objective function.
    //   lower_bounds:         an array of lower bounds (inclusive) of the interval containing the
    //                         optimal point.
    //   upper_bounds:         an array of upper bounds (inclusive) of the interval containing the
    //                         optimal point.
    SimulatedAnnealing(Objective objective_function, int number_of_parameters,
                       std::vector<double> lower_bounds, std::vector<double> upper_bounds)
        : Optimizer(std::move(objective_function), number_of_parameters),
          lower_bounds_(std::move(lower_bounds)),
          upper_bounds_(std::move(upper_bounds)) {
        // Check if the length of the lower and upper bounds equal the number of parameters
        if (static_cast<int>(lower_bounds_.size()) != number_of_parameters_ ||
            static_cast<int>(upper_bounds_.size()) != number_of_parameters_) {
            throw ArgumentException(
                "The lower and upper bounds must be the same length as the number of parameters.");
        }
        // Check if the lower bounds are less than the upper bounds
        for (std::size_t i = 0; i < lower_bounds_.size(); ++i) {
            if (upper_bounds_[i] <= lower_bounds_[i])
                throw ArgumentException("The upper bound cannot be less than or equal to the lower bound.");
        }
    }

    // An array of lower bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }

    // An array of upper bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

    // The pseudo random number generator (PRNG) seed.
    int prng_seed = 12345;

    // The initial temperature at the start of the algorithm. Default = 10.
    double initial_temperature = 10;

    // The minimum temperature allowable. The temperature will be held constant when it reaches
    // this point.
    double min_temperature = 0.1;

    // The cooling rate for the annealing schedule. Default = 0.95.
    double cooling_rate = 0.95;

    // The number of cycles before updating the step size. Default = 10.
    // (C#'s doc comment says 10; the initializer says 4. Both are transcribed as-is.)
    int update_cycles = 4;

    // The number of cycles before reducing the temperature. 10.
    int temperature_cycles = 10;

    // The number of successive temperature reductions to test for termination. Default = 20.
    // Validated below and then never used by the algorithm -- see transcription hazard 1.
    int tolerance_steps = 20;

   protected:
    void optimize() override {
        int i, j, k, D = number_of_parameters_;

        // Validate. (The initial-temperature check compares against 1 while its message says
        // "greater than 0"; transcribed as written.)
        if (initial_temperature < 1)
            throw ArgumentException("The initial temperature must be greater than 0.");
        if (update_cycles < 4)
            throw ArgumentException(
                "The number of cycles before updating the step size must be at least 4.");
        if (temperature_cycles < 4)
            throw ArgumentException(
                "The number of cycles before reducing the temperature must be at least 4.");
        if (tolerance_steps < 4)
            throw ArgumentException("The number of tolerance steps must be greater than 3.");

        // Setup variables
        double T = initial_temperature;
        bool cancel = false;
        auto x = linalg::Vector(D);
        auto xp = linalg::Vector(D);
        // C# also declares `var fitness = new List<double>();` here and never reads or writes it;
        // a dead local with no behavior to mirror, so it is not transcribed.
        sampling::MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));

        // Initial evaluation
        for (j = 0; j < D; j++)
            x[j] = 0.5 * (lower_bounds_[static_cast<std::size_t>(j)] +
                          upper_bounds_[static_cast<std::size_t>(j)]);
        std::vector<double> xa = x.to_array();
        double fx = evaluate(xa, cancel);
        double fxp, df;

        // variables for corona update
        std::vector<int> acceptances(static_cast<std::size_t>(D), 0);
        std::vector<double> v(static_cast<std::size_t>(D));
        std::vector<double> c(static_cast<std::size_t>(D));
        for (j = 0; j < D; ++j) {
            v[static_cast<std::size_t>(j)] = 1.0 / initial_temperature;
            c[static_cast<std::size_t>(j)] = 2.0;
        }

        // Perform adaptive simulated annealing
        while (iterations_ < max_iterations) {
            // Reduce temperature
            T *= cooling_rate;
            T = std::max(min_temperature, T);

            for (i = 1; i <= temperature_cycles; i++) {
                for (j = 1; j <= update_cycles; j++) {
                    // Perform a cycle of random moves, each along one coordinate direction.
                    // Accept or reject each point according to the Metropolis criterion.
                    // Record the optimum point reached so far.
                    for (k = 0; k < D; k++) {
                        std::size_t kk = static_cast<std::size_t>(k);

                        // Create basis vector
                        auto basis = linalg::Vector(D, 0);
                        basis[k] = 1;

                        // Create proposal vector
                        xp = x + basis * v[kk] * (prng.next_double() * 2.0 - 1.0);

                        // If XP is out of bounds, continue (see transcription hazard 2)
                        if (xp[k] < lower_bounds_[kk] || xp[k] > upper_bounds_[kk]) continue;

                        // Evaluate proposal vector
                        std::vector<double> xpa = xp.to_array();
                        fxp = evaluate(xpa, cancel);
                        if (cancel) return;
                        df = fxp - fx;

                        // Metropolis rule (short-circuiting -- see transcription hazard 3)
                        if (df < 0 || prng.next_double() < std::exp(-df / T)) {
                            // Accept proposal
                            acceptances[kk] += 1;
                            x = xp;
                            fx = fxp;
                        }
                    }
                }

                // Do Corana et al. update
                // Goal is to keep the acceptance rate near 50%
                for (j = 0; j < D; j++) {
                    std::size_t jj = static_cast<std::size_t>(j);
                    // Get acceptance rate
                    double rate = static_cast<double>(acceptances[jj]) / update_cycles;
                    if (rate > 0.6) {
                        v[jj] *= 1.0 + c[jj] * (rate - 0.6) / 0.4;
                    } else if (rate < 0.4) {
                        v[jj] /= 1.0 + c[jj] * (0.4 - rate) / 0.4;
                    }
                }
                std::fill(acceptances.begin(), acceptances.end(), 0);
            }

            iterations_ += 1;

            // Set current point to the optimum.
            x = linalg::Vector(best_parameter_set_.values);
            fx = best_parameter_set_.fitness;
        }

        // If we made it to here, the maximum iterations were reached.
        update_status(OptimizationStatus::MaximumIterationsReached);
    }

   private:
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
};

}  // namespace corehydro::numerics::math::optimization
