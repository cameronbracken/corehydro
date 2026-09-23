// ported from: Numerics/Mathematics/Optimization/Global/DifferentialEvolution.cs @ 7e8e8d1
//
// The Differential Evolution (DE) global optimizer (Storn & Price, 1997/1998): maintains a
// population of candidate solutions, mutates/recombines them each generation, and keeps
// whichever trial has the best fitness. No gradient needed. This is the optimizer the
// MCMC MAP-initialization path (a later task) uses to seed its proposal distribution, so
// its seeded search must reproduce the real C# call-by-call random-draw sequence exactly
// -- see the three transcription hazards called out below.
//
// Transcription hazards:
//
// 1. Random triple-index pick. C# `Enumerable.Range(0, Np).Where(x => x !=
//    i).OrderBy(_ => prng.NextDouble()).Take(3)` draws EXACTLY ONE `prng.NextDouble()` per
//    surviving candidate, in SOURCE ORDER (0, 1, ..., Np-1, skipping i), building a
//    (candidate, key) pair for each; LINQ's `OrderBy` is a STABLE sort, so ties (which
//    never actually occur for `double` keys in practice, but the algorithm's correctness
//    doesn't depend on that) preserve source order. The consumed PRNG stream therefore
//    does NOT depend on which candidates end up selected -- every candidate draws a key
//    regardless. This is ported as an explicit `(key, index)` array filled in source
//    order, followed by `std::stable_sort` on the key, taking the first three -- NOT a
//    partial/nth_element selection or a draw-as-you-compare algorithm, either of which
//    would consume the PRNG stream in a different order or break stability.
//
// 2. `jRand` type. C# declares `double jRand = prng.Next(0, D);` -- an `int` result
//    implicitly widened to `double` -- and later compares it against the loop index `j`
//    (an `int`) via `j == jRand` (widening `j` to `double` for the comparison). Ported
//    verbatim as a `double`, not narrowed to `int`: functionally the comparison always
//    behaves like an exact integer equality check here (`Next(0, D)` only ever returns
//    integer values exactly representable as `double` for any realistic `D`), but the
//    type is transcribed as-is per this port's fidelity convention rather than "simplified".
//
// 3. Dither/draw order. Per candidate, per generation, the draw order is: (a) 3
//    `prng.NextDouble()` calls consumed by the stable-sort key array above (one per
//    surviving candidate, so `PopulationSize - 1` draws, not 3 -- the "Take(3)" only
//    slices the RESULT, it does not limit how many keys get drawn); (b) 1
//    `prng.NextDouble()` for the dither check, plus a 2nd `prng.NextDouble()` if dithering
//    fires (`<= DitherRate`); (c) 1 `prng.Next(0, D)` for `jRand`; (d) `D` more
//    `prng.NextDouble()` calls, one per dimension, for the crossover decision. Population
//    initialization draws from `LatinHypercube.Random(PopulationSize, D, PRNGSeed)` (its
//    own already-ported, already-seeded MT stream -- see latin_hypercube.hpp), which is
//    entirely separate from the `prng` instance used during the generation loop; both are
//    seeded from the SAME `PRNGSeed` (C# constructs a fresh `MersenneTwister(PRNGSeed)`
//    for `prng` AND passes the same `PRNGSeed` to `LatinHypercube.Random` -- two
//    independently-seeded streams, not one shared stream), mirrored exactly here.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/data/running_statistics.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/sampling/latin_hypercube.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::math::optimization {

class DifferentialEvolution : public Optimizer {
   public:
    // Construct a new differential evolution optimization method.
    DifferentialEvolution(Objective objective_function, int number_of_parameters,
                           std::vector<double> lower_bounds, std::vector<double> upper_bounds)
        : Optimizer(std::move(objective_function), number_of_parameters),
          lower_bounds_(std::move(lower_bounds)),
          upper_bounds_(std::move(upper_bounds)) {
        if (static_cast<int>(lower_bounds_.size()) != number_of_parameters_ ||
            static_cast<int>(upper_bounds_.size()) != number_of_parameters_) {
            throw ArgumentException(
                "The lower and upper bounds must be the same length as the number of parameters.");
        }
        for (std::size_t i = 0; i < lower_bounds_.size(); ++i) {
            if (upper_bounds_[i] <= lower_bounds_[i])
                throw ArgumentException("The upper bound cannot be less than or equal to the lower bound.");
        }
        // C#'s field initializer (`= 30`) runs before this ctor body, then is
        // unconditionally overwritten here -- see population_size's declaration below.
        population_size = 10 * number_of_parameters_;
    }

    // An array of lower bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }

    // An array of upper bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

    // The total population size. Default = 10 * D (Storn & Price, 1997); the ctor always
    // overwrites this field initializer (see ctor body).
    int population_size = 30;

    // The pseudo random number generator (PRNG) seed.
    int prng_seed = 12345;

    // The mutation constant or differential weight, in [0, 2]. Default = 0.75.
    double mutation = 0.75;

    // Determines how often the mutation constant dithers between 0.5 and 1.0.
    double dither_rate = 0.9;

    // The crossover probability or recombination constant, in [0, 1]. Default = 0.9.
    double crossover_probability = 0.9;

   protected:
    const std::vector<double>& parameter_lower_bounds() const override { return lower_bounds_; }
    const std::vector<double>& parameter_upper_bounds() const override { return upper_bounds_; }

    void optimize() override {
        if (population_size < 1) throw ArgumentException("The population size must be greater than 0.");
        if (mutation < 0 || mutation > 2)
            throw ArgumentException("The mutation parameter must be between 0 and 2.");
        if (dither_rate < 0 || dither_rate > 1)
            throw ArgumentException("The dithering rate must be between 0 and 1.");
        if (crossover_probability < 0 || crossover_probability > 1)
            throw ArgumentException("The crossover probability must be between 0 and 1.");

        int D = number_of_parameters_;
        bool cancel = false;

        // Initialize the population of points. Two independently-seeded streams from the
        // SAME PRNGSeed -- see transcription hazard 3 above.
        sampling::MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        auto rnd = sampling::LatinHypercube::random(population_size, D, prng_seed);

        std::vector<ParameterSet> Xp;
        Xp.reserve(static_cast<std::size_t>(population_size));
        for (int i = 0; i < population_size; ++i) {
            std::vector<double> values(static_cast<std::size_t>(D));
            for (int j = 0; j < D; ++j) {
                values[static_cast<std::size_t>(j)] =
                    lower_bounds_[static_cast<std::size_t>(j)] +
                    rnd[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                        (upper_bounds_[static_cast<std::size_t>(j)] - lower_bounds_[static_cast<std::size_t>(j)]);
            }
            double fitness = evaluate(values, cancel);
            Xp.emplace_back(values, fitness);
            if (cancel) return;
        }

        iterations_ += 1;

        // Perform Differential Evolution.
        while (iterations_ < max_iterations) {
            // Keep track of population statistics to assess convergence.
            data::RunningStatistics statistics;

            // Mutate and recombine population.
            for (int i = 0; i < population_size; ++i) {
                // Randomly select three vector indexes -- see transcription hazard 1.
                std::vector<std::pair<double, int>> keyed;
                keyed.reserve(static_cast<std::size_t>(population_size > 0 ? population_size - 1 : 0));
                for (int x = 0; x < population_size; ++x) {
                    if (x == i) continue;
                    keyed.emplace_back(prng.next_double(), x);
                }
                // C# `indices[0], indices[1], indices[2]` throws IndexOutOfRangeException when
                // fewer than 3 OTHER population members exist (PopulationSize < 4, since index
                // `i` is excluded above); `std::vector::operator[]` would instead be UB. Guarded
                // to reproduce the observable C# throw (COVSRT precedent, multivariate_normal.hpp).
                // Unreachable via the public API today (the ctor always sets population_size =
                // 10 * D), but population_size is a public settable field, so this is reachable
                // if a caller shrinks it after construction.
                if (keyed.size() < 3)
                    throw std::out_of_range(
                        "The population size must be at least 4 (each candidate needs 3 other "
                        "distinct population members to mutate from).");
                std::stable_sort(keyed.begin(), keyed.end(),
                                  [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                                      return a.first < b.first;
                                  });
                int r0 = keyed[0].second, r1 = keyed[1].second, r2 = keyed[2].second;

                // Generate trial vector.
                double G = prng.next_double() <= dither_rate ? 0.5 + prng.next_double() * 0.5 : mutation;
                // jRand is a double -- see transcription hazard 2.
                double jRand = static_cast<double>(prng.next(0, D));
                std::vector<double> u(static_cast<std::size_t>(D));
                for (int j = 0; j < D; ++j) {
                    double rr = prng.next_double();
                    if (rr <= crossover_probability || static_cast<double>(j) == jRand) {
                        u[static_cast<std::size_t>(j)] =
                            Xp[static_cast<std::size_t>(r0)].values[static_cast<std::size_t>(j)] +
                            G * (Xp[static_cast<std::size_t>(r1)].values[static_cast<std::size_t>(j)] -
                                 Xp[static_cast<std::size_t>(r2)].values[static_cast<std::size_t>(j)]);
                        if (u[static_cast<std::size_t>(j)] < lower_bounds_[static_cast<std::size_t>(j)]) {
                            u[static_cast<std::size_t>(j)] =
                                0.5 * Xp[static_cast<std::size_t>(i)].values[static_cast<std::size_t>(j)] +
                                0.5 * lower_bounds_[static_cast<std::size_t>(j)];
                        } else if (u[static_cast<std::size_t>(j)] > upper_bounds_[static_cast<std::size_t>(j)]) {
                            u[static_cast<std::size_t>(j)] =
                                0.5 * Xp[static_cast<std::size_t>(i)].values[static_cast<std::size_t>(j)] +
                                0.5 * upper_bounds_[static_cast<std::size_t>(j)];
                        }
                    } else {
                        u[static_cast<std::size_t>(j)] = Xp[static_cast<std::size_t>(i)].values[static_cast<std::size_t>(j)];
                    }
                }

                // Evaluate fitness.
                double fitness = evaluate(u, cancel);
                if (cancel) return;

                // Update population. `<=` (not `<`): a tying trial always replaces the
                // incumbent, matching Optimizer::evaluate()'s own `<=` best-tracking rule.
                if (fitness <= Xp[static_cast<std::size_t>(i)].fitness) {
                    Xp[static_cast<std::size_t>(i)] = ParameterSet(u, fitness);
                }

                // Keep running stats of population.
                statistics.push(Xp[static_cast<std::size_t>(i)].fitness);
            }

            // Evaluate convergence.
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
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
};

}  // namespace corehydro::numerics::math::optimization
