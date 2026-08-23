// ported from: Numerics/Mathematics/Optimization/Global/ParticleSwarm.cs @ 2a0357a
//
// The Particle Swarm (PSO) global optimizer (Kochenderfer & Wheeler, "Algorithms for
// Optimization", 2019): a population of particles moves through the search space, each pulled
// toward its own personal best and the swarm's global best. No gradient needed. Structurally
// this is DifferentialEvolution's sibling -- same two-independently-seeded-stream population
// initialization, same RunningStatistics convergence test, same MaximumIterationsReached exit
// -- so differential_evolution.hpp is the style model here.
//
// Transcription hazards:
//
// 1. Two independent streams from ONE seed. C# constructs `new MersenneTwister(PRNGSeed)` for
//    the velocity draws AND passes the same `PRNGSeed` to `LatinHypercube.Random(PopulationSize,
//    D, PRNGSeed)` for the positions. Those are two separately-seeded MT streams, not one shared
//    stream (the same arrangement DifferentialEvolution uses; see its transcription hazard 3).
//    The initialization loop therefore consumes exactly D `prng.NextDouble()` draws per particle
//    -- one per velocity component -- with the positions coming entirely off the LHS matrix.
//
// 2. Inertia-weight arithmetic. C# writes `var w = wmax - (wmax - wmin) * Iterations /
//    MaxIterations;` with `Iterations` and `MaxIterations` both `int`. This is NOT integer
//    division: `(wmax - wmin)` is a `double`, so `* Iterations` promotes to `double` and the
//    subsequent `/ MaxIterations` is a double divide. The expression is transcribed with the
//    same operand order so the rounding matches term for term.
//
// 3. The weight is recomputed INSIDE the parameter (j) loop, once per dimension, even though it
//    depends only on the iteration counter. Redundant, but harmless and preserved as written --
//    hoisting it out of the loop would be a "cleanup" this port does not make.
//
// 4. The four Alam (2016) constants are C# method-body locals; they live at file scope here as
//    `constexpr` per this repo's MSVC C3493 rule (a `const` function-local used implicitly
//    inside a capture-less lambda is an error there, and file-scope constants sidestep the whole
//    class of problem). Values and use sites are unchanged.
#pragma once
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "corehydro/numerics/data/running_statistics.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/sampling/latin_hypercube.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::math::optimization {

// These parameters are recommended by Alam "Particle Swarm Optimization: Algorithm and its Codes
// in MATLAB" (2016). C# declares them as locals inside Optimize(); see transcription hazard 4.
inline constexpr double kParticleSwarmWMin = 0.4;
inline constexpr double kParticleSwarmWMax = 0.9;
inline constexpr double kParticleSwarmC1 = 2.05;
inline constexpr double kParticleSwarmC2 = 2.05;

class ParticleSwarm : public Optimizer {
   public:
    // Construct a new particle swarm optimization method.
    ParticleSwarm(Objective objective_function, int number_of_parameters,
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

    // The total population size. Default = 30.
    int population_size = 30;

    // The pseudo random number generator (PRNG) seed.
    int prng_seed = 12345;

   protected:
    void optimize() override {
        if (population_size < 1) throw ArgumentException("The population size must be greater than 0.");

        int D = number_of_parameters_;
        bool cancel = false;

        // Initialize the population of points. Two independently-seeded streams from the SAME
        // PRNGSeed -- see transcription hazard 1 above.
        sampling::MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        auto rnd = sampling::LatinHypercube::random(population_size, D, prng_seed);

        std::vector<Particle> Xp;
        Xp.reserve(static_cast<std::size_t>(population_size));
        for (int i = 0; i < population_size; ++i) {
            std::vector<double> values(static_cast<std::size_t>(D));
            std::vector<double> velocity(static_cast<std::size_t>(D));
            for (int j = 0; j < D; ++j) {
                std::size_t jj = static_cast<std::size_t>(j);
                values[jj] = lower_bounds_[jj] +
                             rnd[static_cast<std::size_t>(i)][jj] * (upper_bounds_[jj] - lower_bounds_[jj]);
                velocity[jj] = 0.1 * (upper_bounds_[jj] - lower_bounds_[jj]) * (prng.next_double() - 0.5);
            }

            double fitness = evaluate(values, cancel);
            if (cancel) return;

            Particle particle;
            particle.parameter_set = ParameterSet(values, fitness);
            particle.best_parameter_set = ParameterSet(values, fitness);
            particle.velocity = std::move(velocity);
            Xp.push_back(std::move(particle));
        }

        iterations_ += 1;

        // Perform Particle Swarm
        while (iterations_ < max_iterations) {
            // Keep track of population statistics to assess convergence
            data::RunningStatistics statistics;

            // Update momentum for each particle
            for (int i = 0; i < population_size; ++i) {
                std::size_t ii = static_cast<std::size_t>(i);
                // Generate trial vector
                std::vector<double> u(static_cast<std::size_t>(D));
                std::vector<double> v(static_cast<std::size_t>(D));
                for (int j = 0; j < D; ++j) {
                    std::size_t jj = static_cast<std::size_t>(j);
                    // New velocity (the weight is recomputed per dimension -- hazard 3; the
                    // arithmetic order is the C# one -- hazard 2).
                    double w = kParticleSwarmWMax -
                               (kParticleSwarmWMax - kParticleSwarmWMin) * iterations_ / max_iterations;
                    v[jj] = w * Xp[ii].velocity[jj] +
                            kParticleSwarmC1 * prng.next_double() *
                                (Xp[ii].best_parameter_set.values[jj] - Xp[ii].parameter_set.values[jj]) +
                            kParticleSwarmC2 * prng.next_double() *
                                (best_parameter_set_.values[jj] - Xp[ii].parameter_set.values[jj]);
                    // New position
                    u[jj] = Xp[ii].parameter_set.values[jj] + v[jj];
                    u[jj] = repair_parameter(u[jj], lower_bounds_[jj], upper_bounds_[jj]);
                }

                // Evaluate fitness
                double fitness = evaluate(u, cancel);
                if (cancel) return;

                Xp[ii].parameter_set = ParameterSet(u, fitness);
                Xp[ii].velocity = v;

                // Update population
                if (fitness <= Xp[ii].best_parameter_set.fitness) {
                    Xp[ii].best_parameter_set = ParameterSet(u, fitness);
                }

                // Keep running stats of population
                statistics.push(Xp[ii].best_parameter_set.fitness);
            }

            // Evaluate convergence
            if (iterations_ >= 10 &&
                statistics.standard_deviation() <
                    absolute_tolerance + relative_tolerance * std::fabs(statistics.mean())) {
                update_status(OptimizationStatus::Success);
                return;
            }

            iterations_ += 1;
        }

        // If we made it to here, the maximum iterations were reached.
        update_status(OptimizationStatus::MaximumIterationsReached);
    }

   private:
    // Class for storing particles.
    struct Particle {
        // The current parameter set (position) of the particle in the search space.
        ParameterSet parameter_set;

        // The best parameter set (position) this particle has found during the search -- the
        // particle's "personal best" or "pbest" in PSO terminology.
        ParameterSet best_parameter_set;

        // The velocity vector of the particle in parameter space, updated from the particle's
        // personal best and the global best.
        std::vector<double> velocity;
    };

    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
};

}  // namespace corehydro::numerics::math::optimization
